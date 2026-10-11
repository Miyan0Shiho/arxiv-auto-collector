# EC-RAG: Event Chain Retrieval-Augmented Generation for Long Video Understanding

**Authors**: Yuhao Qin, Junbo Wang, Yuke Li, Yining Zhu

**Published**: 2026-10-06 16:57:46

**PDF URL**: [https://arxiv.org/pdf/2610.08674v1](https://arxiv.org/pdf/2610.08674v1)

## Abstract
Current large video-language models (LVLMs) still face challenges when dealing with long videos, mainly because frames are often processed independently, making it difficult to capture temporal dependencies across events. Although retrieval-augmented approaches have been introduced to provide additional context, most of them operate at the frame or snippet level, which limits their ability to model how events evolve over time and relate to each other. In this paper, we propose Event Chain Retrieval-Augmented Generation (EC-RAG), a training-free framework that organizes video content into an explicit event chain before question answering. Instead of retrieving isolated frames or text segments, EC-RAG first partitions the video into semantically coherent segments, represents each segment using multi-modal signals, and then links them into a structured chain that preserves temporal order and captures inter-event relationships. Given a query, the system identifies relevant events within this chain and gathers supporting evidence from the associated modalities. Our approach offers several practical advantages: (i) event-level abstraction that better reflects how video content is naturally structured, enabling more reliable localization compared to frame-level retrieval; (ii) structured multi-modal fusion that aggregates speech, text, and visual cues at the event level, allowing complementary information to be more effectively utilized during reasoning; and (iii) plug-and-play compatibility with existing LVLM backbones, requiring no additional training or reliance on proprietary models. Experiments on Video-MME, MLVU, and LongVideoBench show that this event-centric design consistently outperforms frame-level retrieval baselines, highlighting the importance of modeling temporal structure for long-video understanding.

## Full Text


<!-- PDF content starts -->

EC-RAG: Event Chain Retrieval-Augmented Generation for Long
Video Understanding
Yuhao Qin
qinyuhaoswu@163.com
Northwestern Polytechnical University
Xi’an, ChinaJunbo Wang
jbwang@nwpu.edu.cn
Northwestern Polytechnical University
Xi’an, China
Yuke Li
liyuke@nwpu.edu.cn
Northwestern Polytechnical University
Xi’an, ChinaYining Zhu
yiningzhu@nwpu.edu.cn
Northwestern Polytechnical University
Xi’an, China
Abstract
Current large video-language models (LVLMs) still face challenges
when dealing with long videos, mainly because frames are often pro-
cessed independently, making it difficult to capture temporal depen-
dencies across events. Although retrieval-augmented approaches
have been introduced to provide additional context, most of them
operate at the frame or snippet level, which limits their ability to
model how events evolve over time and relate to each other. In this
paper, we propose Event Chain Retrieval-Augmented Generation
(EC-RAG), a training-free framework that organizes video content
into an explicit event chain before question answering. Instead of
retrieving isolated frames or text segments, EC-RAG first partitions
the video into semantically coherent segments, represents each
segment using multi-modal signals, and then links them into a
structured chain that preserves temporal order and captures inter-
event relationships. Given a query, the system identifies relevant
events within this chain and gathers supporting evidence from
the associated modalities. Our approach offers several practical ad-
vantages: (i) event-level abstraction that better reflects how video
content is naturally structured, enabling more reliable localization
compared to frame-level retrieval; (ii) structured multi-modal fu-
sion that aggregates speech, text, and visual cues at the event level,
allowing complementary information to be more effectively utilized
during reasoning; and (iii) plug-and-play compatibility with exist-
ing LVLM backbones, requiring no additional training or reliance
on proprietary models. Experiments on Video-MME, MLVU, and
LongVideoBench show that this event-centric design consistently
outperforms frame-level retrieval baselines, highlighting the impor-
tance of modeling temporal structure for long-video understanding.
Keywords
Long Video Understanding, Retrieval-Augmented Generation, Event
Chain Reasoning
1 Introduction
With the rapid advancement of Large Video-Language Models
(LVLMs), significant progress has been made in video understand-
ing tasks [ 2,14,17,18,37,41]. However, when processing long
videos spanning tens of minutes to hours, current LVLMs face crit-
ical challenges: the limited context window restricts the number
of frames that can be processed, while simply increasing sampled
frames leads to information redundancy and reasoning degradation.
Figure 1: Advantages of our EC-RAG. EC-RAG provides an
event-centric, training-free pipeline with structured tempo-
ral understanding that is easily compatible with any LVLM.
Recent studies have sought to extend the context length of
LVLMs for long video understanding, such as LongLLaVA [ 29]
scales token capacity to ingest more frames. However, this brute-
force expansion proves brittle—Video-MME [ 6] benchmark shows
accuracy drops when additional frames are supplied beyond a
threshold. This outcome suggests that simply increasing the num-
ber of sampled frames not only leads to information redundancy
but also imposes additional challenges for the model to handle
complex reasoning. A parallel research direction employs Retrieval-
Augmented Generation (RAG) to supplement video queries with ex-
ternally retrieved information. Early efforts in this direction [ 16,39]
demonstrate that augmenting visual inputs with auxiliary textual
information can effectively enhance video understanding, particu-
larly for content that is difficult to capture through visual frames
alone. These methods typically extract three types of auxiliary texts
from videos: automatic speech recognition (ASR) for transcribing
spoken dialogue and narration, optical character recognition (OCR)
for capturing on-screen text such as titles and labels, and object
detection (DET) for identifying visual entities and their spatial
arrangements. Building upon this foundation, Video-RAG [ 20] pro-
poses visually-aligned auxiliary texts from OCR, ASR, and DET,
achieving notable improvements with minimal computational over-
head. TV-RAG [ 1] further introduces temporal-window retrieval
1
arXiv:2610.08674v1  [cs.CV]  6 Oct 2026

Qin et al.
with entropy-based keyframe selection to handle semantic drift in
long videos. More recently, AdaVideoRAG [ 35] employs adaptive
difficulty routing and graph databases for complex multi-hop rea-
soning. Despite these advances, existing video RAG methods share
a fundamental limitation: they retrieve at the frame level or text seg-
ment level, treating video as a collection of independent units rather
than a coherent narrative of interconnected events. Long videos
inherently contain sequences of causally connected events—a lec-
ture progresses through topics, a movie unfolds through scenes, a
cooking video follows recipe steps. Without modeling this event
structure, current methods produce fragmented retrieval results
that fail to support questions requiring temporal reasoning.
Based on this observation, we focus on the following question:
Can we design a video RAG framework that explicitly captures tem-
poral event dynamics and inter-event relationships for coherent long
video understanding?
To this end, we propose Event Chain Retrieval-Augmented Gen-
eration (EC-RAG), a training-free framework that organizes video
content into explicit event chains before question answering. In-
stead of retrieving isolated frames or text snippets, EC-RAG first
partitions the video into semantically coherent temporal segments,
aggregates multi-modal signals (ASR, OCR, DET) within each seg-
ment to form events, and links them into a structured chain that
preserves temporal order. For multi-modal extraction, we introduce
CLIP-Variance guided keyframe selection that combines query-
semantic relevance with visual informativeness, ensuring selected
frames are both relevant and information-rich. Given a user query,
EC-RAG performs event-level localization to identify the most rele-
vant events along with their temporal context, and gathers support-
ing evidence from associated modalities for answer generation.
We evaluate EC-RAG on Video-MME [ 6] , MLVU [ 42] , and
LongVideoBench [ 33] benchmarks. By applying EC-RAG to five
different open-source LVLMs, we achieve an average performance
improvement of 4% on Video-MME. Furthermore, compared with
existing frame-level retrieval methods such as Video-RAG and TV-
RAG, EC-RAG consistently demonstrates superior performance,
validating the effectiveness of our event-chain-based approach. Our
contributions are summarized as follows:
•We propose EC-RAG, a novel framework that explicitly
models temporal event chains for long video understand-
ing. Unlike existing methods operating at frame or snippet
level, EC-RAG captures inter-event relationships and en-
ables coherent reasoning over extended temporal spans.
•We introduce CLIP-Variance guided keyframe selection that
jointly considers semantic relevance and visual informa-
tiveness, improving the quality of multi-modal information
extraction for event construction.
•We design a query-guided event localization mechanism
that identifies relevant events within the temporal chain
and aggregates multi-modal evidence, providing structured
context for LVLM answer generation.2 Related Work
2.1 Video Understanding with Large Language
Models
The emergence of powerful LLMs has spurred efforts to develop
comprehensive video-language systems. Early works like Video-
ChatGPT [ 21] and VideoChat [ 12] adopt frame-wise feature extrac-
tion followed by temporal aggregation. Video-LLaVA [ 13] intro-
duces a unified projection to bridge visual and textual modalities,
while LLaVA-NeXT-Video [ 39] further refines this through video-
specific adaptation. These models perform well on short videos but
struggle with longer content due to frame sampling constraints.
To address this limitation, recent methods explore expanding the
context window for extended sequences. LongVA [ 38] and Long-
LLaVA [ 29] leverage pre-training on lengthy documents to develop
transferable sequence modeling capabilities. INTP [ 26] reorganizes
visual tokens to accommodate more information within fixed con-
text limits. However, densely sampling frames quickly becomes
computationally prohibitive, and the high redundancy in video
data often limits the benefits of processing more frames.
2.2 LLM-based Agents for Video Understanding
An alternative paradigm employs LLMs as controllers to orchestrate
specialized tools for video comprehension [ 7,22,30,32,36]. Repre-
sentative methods such as VideoAgent [ 4] and MM-VID [ 15] enable
dynamic interaction with video content through iterative retrieval
and reasoning. While flexible, these approaches introduce substan-
tial latency from multiple tool calls and often rely on proprietary
APIs, limiting broader adoption in open-source settings.
2.3 Retrieval-Augmented Generation for Video
RAG-based methods retrieve relevant information from pre-constructed
databases to augment video understanding. Video-RAG [ 20] ex-
tracts multimodal cues including speech transcripts and detected
objects to build searchable knowledge bases. TV-RAG [ 1] enhances
this with temporal-aware retrieval, while AdaVideoRAG [ 35] em-
ploys adaptive routing for queries of varying complexity. Current
approaches, however, retrieve content at the frame or segment
granularity without capturing the temporal progression of events,
limiting performance on questions requiring understanding of how
actions unfold across time.
3 Method
We proposeEC-RAG, a training-free retrieval-augmented frame-
work for large video-language models that organizes video content
as structured event chains. As illustrated in Figure 2, our pipeline
comprises four stages: (i)Query Decoupling: The user query is
decomposed into modality-specific retrieval requests for ASR, OCR,
and DET. (ii)CLIP-Variance Guided Multimodal Extraction: In-
formative keyframes are selected through joint CLIP similarity and
variance weighting, followed by parallel extraction of speech, text,
and object information. (iii)Temporal Event Chain Construc-
tion: The video is segmented into temporal units and organized as
a chain of semantically coherent events with explicit inter-event
relations. (iv)Evidence-augmented Multimodal Reasoning:
Query-guided localization identifies relevant events, and the LVLM
2

EC-RAG for Long Video Understanding
Figure 2: Overview of the EC-RAG framework. The pipeline comprises four stages: (i) Query Decoupling parses the user question
into modality-specific retrieval requests for ASR, OCR, and DET; (ii) CLIP-Variance Guided Multimodal Extraction selects
informative keyframes through weighted scoring of visual similarity and frame variance, then extracts speech transcripts,
on-screen text, and scene graphs via open-vocabulary object detection; (iii) Temporal Event Chain Construction segments
the video into 30-second units and builds a chain of semantically coherent events, where each event integrates its associated
multimodal evidence with explicit temporal relations; (iv) Evidence-augmented Multimodal Reasoning performs query-guided
event localization to retrieve relevant events, then feeds the event chain summary, located events, and raw evidence to the
LVLM for answer generation. By organizing video content as structured event sequences, EC-RAG captures temporal dynamics
and inter-event dependencies for complex video understanding.
generates answers grounded in the event chain context and multi-
modal evidence.
Problem Formulation.Given a video 𝑉and a user query 𝑄, a
frame sampler extracts 𝑁framesF ={𝐹 1,𝐹2,...,𝐹 𝑁}. These frames
are encoded by a visual encoder (e.g.,CLIP-L[ 24]) to obtain visual
featuresF 𝑣. The standard LVLM inference can be formulated as:
𝑂=LVLM(F 𝑣,𝑄)(1)
However, this direct approach struggles with long videos where
critical information may be sparse across extended durations.EC-
RAGaddresses this limitation by constructing anevent chain
(EC)Ethat captures the temporal structure of video content, and
performingquery-guided event localizationto retrieve relevant
context before final reasoning:
𝑂=LVLM(F 𝑣,𝑄,E,A)(2)whereEdenotes the temporal event chain andArepresents the
retrieved multimodal evidence. This formulation enables the model
to reason over both visual frames and structured textual context
derived from the video.
3.1 Query Decoupling
Before processing the video content, we first analyze the user query
to understand what types of information are needed to answer it.
Motivated by recent methods [ 13,20], this query decoupling step
separates the retrieval planning from the actual video processing,
allowing the system to focus computational resources on relevant
modalities. Upon receiving a user query,EC-RAGdecouples it into
structured retrieval requests without accessing video frames. The
LVLM analyzes the query to determine what information sources
are needed:
R=LVLM(𝑃 𝑑,𝑄)={𝑅 asr,𝑅ocr,𝑅det}(3)
3

Qin et al.
where𝑃𝑑is the decoupling prompt enhanced with few-shot exam-
ples that demonstrate how to extract retrieval targets from various
question types. Each request may be NULL if the corresponding
modality is unnecessary for the given query. Specifically, 𝑅asrspeci-
fies key phrases to retrieve from speech transcripts, which is partic-
ularly useful for questions about dialogue content or narration. 𝑅ocr
defines keywords to search in on-screen text, targeting questions
about titles, labels, or displayed information. 𝑅detindicates physical
objects to detect in video frames, essential for visual grounding
questions.
The decoupling prompt guides the LVLM to identify concrete re-
trieval targets through structured examples. For detection requests,
we enforce that 𝑅detcontains only physical entities rather than ab-
stract concepts, ensuring compatibility with vision-based detection
models. This constraint prevents the system from attempting to
detect non-visual concepts like emotions or intentions.
3.2 CLIP-Variance Guided Multimodal
Extraction
This stage selects informative keyframes and extracts multimodal
information from the video in parallel.
Keyframe Selection.A critical challenge in video understand-
ing is selecting which frames to process for detailed analysis. Uni-
form sampling may miss important moments, while processing
all frames is computationally prohibitive. Rather than uniformly
sampling frames or relying solely on semantic similarity, inspired
by query-guided frame selection in prior work [ 1,20], we propose
aCLIP-variance (CV) guided selectionmechanism that jointly
considers query relevance and visual informativeness. The intuition
is that informative frames should both relate to the query content
and contain rich visual details rather than static or redundant im-
agery. For each frame𝐹 𝑡, we compute a weighted score:
𝑠𝑡=sim CLIP(𝑅det,𝐹𝑡)·𝜎 𝑡 (4)
where sim CLIP(·)denotes the CLIP similarity between the detection
query and frame content. This term ensures that selected frames
are semantically relevant to the objects or entities mentioned in the
query. The second factor 𝜎𝑡is thenormalized visual varianceof
frame𝐹 𝑡:
𝜎𝑡=Var(𝐹 𝑡)
max 𝑗Var(𝐹 𝑗)(5)
The variance term captures the visual complexity of each frame,
computed from the pixel intensity distribution. Frames with higher
variance typically contain more visual structure and details, while
low-variance frames often correspond to static backgrounds, black
screens, or uniform scenes that provide limited information. By
multiplying these two factors, we favor frames that are both query-
relevant and visually informative. Keyframes are selected by thresh-
olding on the weighted scores:
Fkey={𝐹 𝑡|𝑠𝑡≥𝜏𝑠}(6)
where𝜏𝑠is a score threshold, with constraints on minimum and
maximum frame counts to ensure adequate coverage across the
video duration.
ASR Extraction.Audio information provides crucial context
that is often unavailable through visual analysis alone, including
dialogue, narration, and background audio cues. Following recentretrieval-augmented video systems [ 1,20], we extract the audio
track from the video and transcribe it using Whisper [ 25], a ro-
bust speech recognition model capable of handling diverse audio
conditions. The resulting speech transcripts 𝑇asrare segmented
by timestamps to preserve temporal alignment with video frames.
These segments are then encoded using Contriever [ 8], a dense re-
trieval encoder trained for semantic similarity matching, producing
embeddings that capture the semantic content of each transcript
segment. The embeddings are indexed in a FAISS [ 10] database for
efficient similarity search:
DBasrFAISS←−−−−E asr=Contriever(𝑇 asr)(7)
OCR Extraction.On-screen text frequently contains valuable
information such as titles, subtitles, labels, and other textual ele-
ments that complement visual and audio content. Current LVLMs
often struggle to accurately recognize text within video frames,
particularly for small or stylized fonts. To address this limitation,
we employ EasyOCR [ 9], a dedicated text recognition model, to
extract on-screen text from each sampled frame. Since OCR models
may produce noisy outputs, we apply a confidence threshold to
filter low-quality detections:
𝑇ocr={𝑡|𝑡=EasyOCR(𝐹),conf(𝑡)>𝜏 ocr}(8)
This filtering step removes unreliable text detections that could
introduce noise into the retrieval process. The filtered texts are
encoded and stored in a separate FAISS index following the same
procedure as ASR:
DBocrFAISS←−−−−E ocr=Contriever(𝑇 ocr)(9)
Object Detection with Scene Graph.While LVLMs demon-
strate strong capabilities in object recognition, they continue to
face challenges in precise object counting, localization, and under-
standing spatial relationships between objects. These limitations
often lead to hallucinations when answering questions that require
fine-grained visual understanding. To provide more accurate object-
level information, we employ APE [ 27], an open-vocabulary object
detection model that accepts textual prompts specifying which
objects to detect. For the selected keyframesF key, APE localizes
relevant objects based on the detection request:
𝐷𝑡=APE(𝐹 𝑡,𝑅det), 𝐹 𝑡∈Fkey (10)
The raw detection outputs consisting of object categories and
bounding box coordinates are not directly interpretable by language
models. To bridge this gap, inspired by scene graph representations
in visual reasoning [ 1,20], we transform the detection results into
semanticscene graphdescriptions expressed in natural language.
For each keyframe, we generate three complementary types of
information: (1)Object Location 𝐴locconverts bounding box co-
ordinates into natural language spatial descriptions (e.g., “left side
of frame”, “upper right corner”), making positional information ac-
cessible to the language model; (2)Object Count 𝐴cntaggregates
detection results to provide accurate counts for each object category,
addressing the counting limitations of LVLMs; (3)Object Relation
𝐴relcomputes and describes relative spatial relationships between
detected objects (e.g., “person A is to the left of person B”). The
complete scene graph representation combines these components:
4

EC-RAG for Long Video Understanding
𝑇det=SceneGraph(𝐴 loc,𝐴cnt,𝐴rel)(11)
3.3 Temporal Event Chain Construction
Unlike frame-level retrieval methods that treat video as a collection
of independent images, EC-RAG explicitly models the temporal
structure of video content by constructing an event chain. This
representation captures how events unfold over time and enables
reasoning about temporal relationships.
Video Segmentation.Long videos contain extended sequences
of content that cannot be processed as a single unit. We partition
the video into non-overlapping temporal segments of fixed duration
Δ:
S={𝑆 𝑖=[𝑡 𝑖,𝑡𝑖+Δ)|𝑖=0,1,...,⌊𝐿/Δ⌋}(12)
where𝐿is the total video duration. Each segment 𝑆𝑖serves as the
basic unit for event extraction and is associated with the multimodal
information (ASR, OCR, DET) extracted from frames within its
time range. This segmentation balances granularity and efficiency:
segments should be short enough to capture coherent events while
being long enough to contain meaningful content.
Event Extraction.Raw multimodal data from each segment
needs to be synthesized into a coherent semantic representation.
For each segment 𝑆𝑖, we aggregate the corresponding ASR, OCR,
and DET information and prompt the LVLM to extract a semantic
event description that summarizes what happens in that temporal
window:
𝑒𝑖=LVLM(𝑃 𝑒,𝑇(𝑖)
asr,𝑇(𝑖)
ocr,𝑇(𝑖)
det)(13)
where𝑃𝑒is the event extraction prompt that instructs the model
to integrate information across modalities. The prompt guides the
LVLM to focus on the main action or occurrence while considering
all available evidence. Each extracted event 𝑒𝑖comprises three com-
ponents: a description that concisely summarizes what happens in
this segment by integrating speech, text, and visual information; a
type that classifies the event as action, dialogue, scene transition,
or informational content; and a list of participants identifying key
entities (people, objects) involved in the event.
Event Chain Assembly.Individual events are connected to
form a coherent narrative structure. The extracted events are or-
ganized into atemporal event chain E={𝑒 0,𝑒1,...,𝑒 𝑛}with
explicit temporal ordering that reflects the video’s chronological
structure. We establish predecessor-successor relations between
adjacent events:
𝑒𝑖temporal−−−−−−→𝑒 𝑖+1,∀𝑖∈[0,𝑛−1](14)
The complete event chain provides a structured overview of
the video’s narrative progression, capturing how events unfold
and connect across time. This representation enables the model
to understand not just individual moments but also the broader
context and temporal dynamics of the video content.
3.4 Evidence-augmented Multimodal Reasoning
With the event chain constructed, EC-RAG performs query-guided
retrieval to identify relevant events and gather supporting evidence
for answer generation.
Query-guided Event Localization.Not all events in a video
are relevant to a given query. We identify the most relevant eventsthrough semantic similarity matching between the query and event
descriptions. Each event description is encoded usingContriever
and compared against the query embedding using cosine similarity:
sim𝑖=cos Contriever(𝑄),Contriever(𝑒 𝑖.desc)(15)
This text-to-text matching is more appropriate than image-text
matching (e.g., CLIP) since both the query and event descriptions
are textual. Events exceeding a relevance threshold 𝜏locare selected
as located events:
Eloc={𝑒 𝑖|sim 𝑖>𝜏loc}(16)
The located events represent the temporal regions most likely
to contain information needed to answer the query.
Evidence Retrieval.While event descriptions provide high-
level summaries, detailed evidence from the original multimodal
sources may be necessary for accurate reasoning. For each located
event𝑒𝑖∈E loc, we retrieve the associated multimodal evidence
from the constructed databases. Using the retrieval requestsRgener-
ated during query decoupling, we query the ASR and OCR databases
to obtain relevant text segments:
𝐴asr=Retrieve(DB asr,𝑅asr,𝜏), 𝐴 ocr=Retrieve(DB ocr,𝑅ocr,𝜏)
(17)
The DET evidence is collected from the scene graphs of keyframes
falling within the located events’ time ranges:
𝐴det={𝑇(𝑡)
det|𝐹𝑡∈Fkey, 𝑡∈𝑒 𝑖.span, 𝑒 𝑖∈E loc}(18)
where the retrieval function returns all database entries with similar-
ity scores exceeding threshold 𝜏. The DET evidence 𝐴detis directly
obtained from the scene graph descriptions of keyframes within
the located events’ time ranges, providing detailed object-level
information for visual grounding.
Context Construction and Answer Generation.The final
reasoning step integrates multiple levels of context to provide the
LVLM with comprehensive information. The context combines
three complementary components: (1)Event Chain Summary
provides a global overview of all events in the video, giving the
model awareness of the full narrative arc even for content not di-
rectly relevant to the query; (2)Located Eventscontain detailed
descriptions of query-relevant events along with their temporal
predecessors, enabling understanding of causal relationships and
temporal context; (3)Multimodal Evidenceincludes the raw ASR,
OCR, and DET information retrieved for the located events, provid-
ing specific details that support accurate reasoning. The complete
context is assembled as:
𝐶=Concat(Summary(E),Detail(E loc),𝐴 asr,𝐴ocr,𝐴det)(19)
Finally, the LVLM generates the answer by jointly reasoning
over the video frames, user query, and constructed context:
𝑂=LVLM(F 𝑣,𝑄,𝐶)(20)
4 Experiments
4.1 Datasets
We assess EC-RAG on three widely adopted long-video benchmarks
that span diverse durations, domains, and reasoning demands.
Video-MME[ 6] offers a systematic testbed for evaluating multi-
modal comprehension over real-world video content. Its 900 videos
5

Qin et al.
Table 1: Performance on the Video-MME [ 6] benchmark. #Text indicates the volume of retrieved textual context. EC-RAG is
integrated into four open-source LVLMs spanning 8–32 input frames.
Model #Text LLM Params Frames Short Medium Long Overall Gain
Proprietary LVLMs
GPT-4o [23] – – 384 80.0 70.3 65.3 71.9 –
Gemini-1.5-Pro [31] – – 0.5 fps 81.7 74.3 67.4 75.0 –
Open-Source LVLMs
Video-LLaVA [13] – 7B 8 44.6 38.3 35.8 39.6 –
Video-LLaVA + EC-RAG 2.0K 7B 8 50.1 44.6 43.4 46.0 +6.4
LLaVA-NeXT-Video [39] – 7B 16 49.4 43.0 36.7 43.0 –
LLaVA-NeXT-Video + EC-RAG 2.0K 7B 16 55.8 53.2 52.5 53.8 +10.8
Long-LLaVA [29] – 7B 32 60.3 51.4 44.1 52.0 –
Long-LLaVA + EC-RAG 1.9K 7B 32 67.6 60.3 60.0 62.6 +10.6
LLaVA-Video [40] – 7B 32 75.7 59.6 52.1 62.4 –
LLaVA-Video + Video-RAG [20] 2.0K 7B 32 70.9 63.2 59.0 64.3 +1.9
LLaVA-Video + TV-RAG [1] 2.0K 7B 32 72.4 64.1 59.8 65.4 +3.0
LLaVA-Video + EC-RAG 2.0K 7B 32 77.8 64.3 60.4 67.5 +5.1
LLaVA-Video [40] – 72B 32 78.0 63.7 59.6 67.1 –
LLaVA-Video + EC-RAG 2.1K 72B 32 81.9 72.4 72.7 75.7 +8.6
are grouped by length intoshort( <2 min),medium(4–15 min), and
long(30–60 min) subsets, yielding 2,700 multiple-choice questions
across 12 task types—ranging from object recognition and count-
ing to temporal reasoning and OCR—and 30 sub-categories drawn
from six broad domains including knowledge, sports, and film. This
breadth makes Video-MME particularly suitable for stress-testing
whether a retrieval-augmented pipeline can supply the right modal-
ity of evidence for each question type.
MLVU[ 42] assembles nine evaluation tasks from videos of 3 min
to 2 h (mean≈12 min), requiring models to sustain reasoning over
extended temporal ranges
LongVideoBench[ 33] focuses on multimodal retrieval and
compositional reasoning within lengthy footage. It provides 6,678
human-written multiple-choice questions spanning 17 thematic cat-
egories, testing a model’s ability to pinpoint and integrate evidence
scattered across long video timelines.
4.2 Implementation Details
All experiments were conducted on NVIDIA RTX 4090 and A800
80 GB GPUs. We adoptLLaVA-Video-7B[ 40] as the primary LVLM
backbone, with three additional 7B architectures and the 72B vari-
ant evaluated in Table 1 to verify cross-architecture and cross-scale
generalizability. Detection requests produced during query decou-
pling are post-filtered with spaCy to retain only concrete, CLIP-
responsive entities. Both the CLIP similarity threshold and the
FAISS [ 10] retrieval threshold are fixed at 𝜏=0.3, with IndexFlatIP
as the similarity backend. Videos are segmented into 30-second
windows for event chain construction. Speech is transcribed by
Whisper-Large [25], on-screen text is recognized by EasyOCR [9],
open-vocabulary detection is handled by APE [ 27], and dense text
retrieval is performed by Contriever [ 8]. Our ablation suite is built
around LLaVA-Video-7B at 32 frames, whose moderate context win-
dow makes it a practical testbed for probing the effect of individual
components and hyper-parameters.Video-MME.To verify that EC-RAG generalizes across archi-
tectures, model scales, and frame budgets, we integrate it into four
open-source LVLMs spanning 8–32 input frames, as reported in Ta-
ble 1. EC-RAG yields consistent improvements for every backbone,
with gains most pronounced onlongvideos where uniform sam-
pling alone cannot cover the query-relevant content. The benefit
is especially evident on frame-constrained backbones, confirming
that event-chain retrieval compensates effectively for sparse visual
input. The pipeline injects approximately 2 K tokens of supplemen-
tary textual context (the#Textcolumn), comparable to roughly 14
extra keyframes in payload, which complement the visual input
and help anchor temporal reasoning.
Scaling to the 72B LLaVA-Video, EC-RAG achieves 75.7% overall,
surpassing both GPT-4o [ 23] and Gemini-1.5-Pro [ 31], showing
that structured event-chain retrieval enables open-source models
to outperform proprietary systems without additional pretrain-
ing. EC-RAG also consistently outperforms Video-RAG [ 20] and
TV-RAG [ 1] under the same backbone setting, as event-level tem-
poral localization provides more precise context than frame-level
retrieval.
MLVU.Table 2 reports performance on the MLVU multiple-
choice task. EC-RAG lifts LLaVA-Video-7B from 70.8 to72.9, out-
performing both Video-RAG and TV-RAG on the same backbone.
Notably, this 7B model with only 64 frames also surpasses Oryx-
1.5 [19], a 32B model operating on 128 frames, highlighting the
efficiency of event-chain retrieval over brute-force scaling.
LongVideoBench.As shown in Table 3, EC-RAG also achieves
the best result on LongVideoBench, reaching59.5with LLaVA-
Video-7B at 64 frames. This benchmark requires retrieving and
composing evidence distributed across long timelines, a scenario
that directly benefits from EC-RAG’s temporal event chain design.
Compared with Video-RAG and TV-RAG under the same setting,
EC-RAG provides a more substantial boost over the unaugmented
6

EC-RAG for Long Video Understanding
Figure 3: Left: Grad-CAM heatmaps on query-relevant and query-irrelevant frames for the baseline and EC-RAG. Right: t-SNE
projection of query, vision, and text features, showing tighter cross-modal alignment when EC-RAG is applied.
Table 2: Overall accuracy on the multiple-choice split of the
MLVU [ 42] benchmark. EC-RAG achieves the best result
among all 7B-scale models.
Model #Params Frames Overall
Proprietary LVLMs
GPT-4o [23] – 0.5 fps 64.6
Open-Source LVLMs
Video-CCAM [5] 14B 96 63.1
Video-XL [28] 7B 256 64.9
Aria [11] 25.3B 256 70.6
LLaVA-Video [40] 7B 64 70.8
Oryx-1.5 [19] 32B 128 72.3
LLaVA-Video + Video-RAG [20] 7B 64 72.4
LLaVA-Video + TV-RAG [1] 7B 64 72.6
LLaVA-Video + EC-RAG 7B 64 72.9
Table 3: Performance on the LongVideoBench [ 33] validation
set.
Model #Params Frames Overall
VideoChat2-Mistral [12] 7B 8 39.3
ShareGPT4Video [3] 7B 8 39.7
LLaVA-Next-Mistral [39] 7B 8 49.1
PLLaVA [34] 34B 16 53.2
LLaVA-Video [40] 7B 64 56.6
LLaVA-Video + Video-RAG [20] 7B 64 58.7
LLaVA-Video + TV-RAG [1] 7B 64 58.8
LLaVA-Video + EC-RAG 7B 64 59.5
baseline, indicating that event-level localization is more effective
than flat retrieval when evidence is temporally scattered.
4.3 Ablation Studies
To quantify each component’s contribution, we ablate EC-RAG on
Video-MME with LLaVA-Video-7B at 32 frames (Table 4). Among allTable 4: Component ablation of EC-RAG on Video-MME with
LLaVA-Video-7B.
ASR OCR DET EC CV Short Medium Long Overall
×✓ ✓ ✓ ✓ 74.8 60.3 54.7 63.3
✓×✓ ✓ ✓ 76.5 62.8 59.9 66.4
✓ ✓×✓ ✓ 77.1 62.9 59.2 66.4
✓ ✓ ✓×✓ 77.0 61.6 58.4 65.7
✓ ✓ ✓ ✓× 76.4 63.5 58.9 66.3
✓ ✓ ✓✓✓ 77.8 64.3 60.4 67.5
modules, ASR has the greatest impact: removing it causes the most
significant performance degradation, particularly on medium and
long videos, confirming that speech transcription is the most infor-
mative auxiliary modality for video QA. The event chain module
ranks second in importance, as disabling temporal localization leads
to less targeted evidence retrieval. CLIP-variance guided keyframe
selection, OCR, and DET each provide meaningful but compara-
tively smaller gains, reflecting their complementary roles in select-
ing informative frames, capturing on-screen text, and grounding
physical entities respectively.
4.4 Qualitative Analysis
Figure 4 presents a representative case from Video-MME in which
the baseline LLaVA-Video fails but EC-RAG succeeds. The question
asks what happened in the car after the heroine returned from
shopping. With 32 uniformly sampled frames, the baseline captures
a frame showing the baby in a cheerful state and, lacking any tem-
poral context to disambiguate it from other segments, incorrectly
selects option C (“Baby happy”).
EC-RAG addresses this through its pipeline: the query decou-
pling stage extracts modality-specific retrieval cues (e.g., ASR key-
words “car, shopping, unhappy” and DET targets “baby, car seat”).
The event chain then localizes the relevant temporal segment,
whose ASR transcript explicitly states “He was just not happy in
the car.” Simultaneously, CLIP-variance guided keyframe selection
identifies a visually informative frame that is forwarded to the
object detector, revealing a baby being fed with a recently-cried
7

Qin et al.
Figure 4: Qualitative comparison between the baseline LLaVA-Video and EC-RAG on a Video-MME example. The baseline
confuses temporally distinct segments, whereas EC-RAG localizes the query-relevant event through its event chain and
aggregates ASR and DET evidence to arrive at the correct answer.
expression. Equipped with this temporally grounded multimodal
evidence, the model correctly selects option D (“Baby unhappy”).
Figure 3 further illustrates the effect of EC-RAG through Grad-
CAM heatmaps and t-SNE projections. The Grad-CAM visualiza-
tions compare the attention distribution of the baseline and EC-RAG
on both a query-relevant and a query-irrelevant frame. With the
supplementary textual evidence provided by EC-RAG, the model’s
attention concentrates more tightly on the semantically relevant
regions (e.g., the baby and car seat), whereas the baseline distributes
attention more diffusely. The t-SNE plot projects the query, vision,
and text features into a shared 2D space, showing that EC-RAG’s
auxiliary tokens pull the vision and query representations closer to-
gether, resulting in more aligned cross-modal features that facilitate
accurate reasoning.
5 Conclusion
We presented EC-RAG, a training-free retrieval-augmented genera-
tion framework that structures long-video understanding aroundtemporal event chains. By decomposing videos into semantically co-
herent events through the fusion of ASR, OCR, and open-vocabulary
detection, and localizing query-relevant segments before injecting
multimodal evidence into the LVLM prompt, EC-RAG consistently
improves diverse backbones across three benchmarks—Video-MME,
MLVU, and LongVideoBench—outperforming existing RAG-based
methods under every tested setting. Notably, with the 72B LLaVA-
Video backbone, EC-RAG surpasses both GPT-4o and Gemini-1.5-
Pro on Video-MME using a fully open-source pipeline, demonstrat-
ing that structured event-level retrieval enables open-source models
to rival and even surpass proprietary systems without additional
training.
Beyond the benchmarks evaluated in this work, we believe
the event-chain paradigm holds broader potential for other video-
centric tasks such as video summarization, temporal grounding,
and multi-turn video dialogue, where preserving temporal struc-
ture and aggregating cross-modal evidence are equally critical. The
training-free nature of EC-RAG also makes it readily applicable to
8

EC-RAG for Long Video Understanding
emerging LVLMs as they continue to evolve, offering a flexible and
scalable augmentation strategy.
References
[1] Zongsheng Cao, Yangfan He, Anran Liu, Jun Xie, Feng Chen, and Zhepeng Wang.
2025. Tv-rag: A temporal-aware and semantic entropy-weighted framework
for long video retrieval and understanding. InProceedings of the 33rd ACM
International Conference on Multimedia. 9071–9079.
[2] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng
Zhao, and Dahua Lin. 2024. Sharegpt4v: Improving large multi-modal models
with better captions. InEuropean Conference on Computer Vision. Springer, 370–
387.
[3] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui
Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al .2024. Sharegpt4video: Im-
proving video understanding and generation with better captions.Advances in
Neural Information Processing Systems37 (2024), 19472–19495.
[4] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. 2024.
Videoagent: A memory-augmented multimodal agent for video understanding.
InEuropean Conference on Computer Vision. Springer, 75–92.
[5]Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang.
2024. Video-ccam: Enhancing video-language understanding with causal cross-
attention masks for short and long videos.arXiv preprint arXiv:2408.14023(2024).
[6]Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang,
Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al .2025. Video-
mme: The first-ever comprehensive evaluation benchmark of multi-modal llms
in video analysis. InProceedings of the IEEE/CVF conference on computer vision
and pattern recognition. 24108–24118.
[7]Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Com-
positional visual reasoning without training. InProceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 14953–14962.
[8]Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv preprint arXiv:2112.09118
(2021).
[9]JaidedAI. 2023. EasyOCR. https://github.com/JaidedAI/EasyOCR. Accessed:
2026-03-30.
[10] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs.IEEE transactions on big data7, 3 (2019), 535–547.
[11] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao
Niu, Fan Zhou, Chengen Huang, Yanpeng Li, et al .2024. Aria: An open multi-
modal native mixture-of-experts model.arXiv preprint arXiv:2410.05993(2024).
[12] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang,
Limin Wang, and Yu Qiao. 2025. Videochat: Chat-centric video understanding.
Science China Information Sciences68, 10 (2025), 200102.
[13] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2024.
Video-llava: Learning united visual representation by alignment before pro-
jection. InProceedings of the 2024 conference on empirical methods in natural
language processing. 5971–5984.
[14] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song
Han. 2024. Vila: On pre-training for visual language models. InProceedings of the
IEEE/CVF conference on computer vision and pattern recognition. 26689–26699.
[15] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab,
Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al .
2023. MM-VID: Advancing Video Understanding with GPT-4V (ision). CoRR
abs/2310.19773 (2023).arXiv preprint arXiv:2310.19773(2023).
[16] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen,
and Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world
knowledge. https://llava-vl.github.io/blog/2024-01-30-llava-next/
[17] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual in-
struction tuning.Advances in neural information processing systems36 (2023),
34892–34916.
[18] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei,
Jianbin Jiao, Enhua Wu, and Jie Hu. 2026. Kangaroo: A Powerful Video-Language
Model Supporting Long-context Video Input: J. Liu et al.International Journal of
Computer Vision134, 3 (2026), 114.
[19] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao.
2024. Oryx mllm: On-demand spatial-temporal understanding at arbitrary reso-
lution.arXiv preprint arXiv:2409.12961(2024).
[20] Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin, Haojia Lin, Chaoyou Fu,
Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, et al .2024. Video-rag: Visually-aligned
retrieval-augmented long video comprehension.arXiv preprint arXiv:2411.13093
(2024).
[21] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. 2024.
Video-chatgpt: Towards detailed video understanding via large vision and lan-
guage models. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). 12585–12602.[22] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho, and Cordelia Schmid.
2024. Morevqa: Exploring modular reasoning models for video question answer-
ing. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition. 13235–13245.
[23] OpenAI. 2024. GPT-4o System Card. https://openai.com/index/gpt-4o-system-
card/. Accessed: 2026-03-30.
[24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sand-
hini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al .
2021. Learning transferable visual models from natural language supervision. In
International conference on machine learning. PmLR, 8748–8763.
[25] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and
Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision.
InInternational conference on machine learning. PMLR, 28492–28518.
[26] Yuzhang Shang, Bingxin Xu, Weitai Kang, Mu Cai, Yuheng Li, Zehao Wen, Zhen
Dong, Kurt Keutzer, Yong Jae Lee, and Yan Yan. 2024. Interpolating video-
llms: Toward longer-sequence lmms in a training-free manner.arXiv preprint
arXiv:2409.12963(2024).
[27] Yunhang Shen, Chaoyou Fu, Peixian Chen, Mengdan Zhang, Ke Li, Xing Sun,
Yunsheng Wu, Shaohui Lin, and Rongrong Ji. 2024. Aligning and prompting
everything all at once for universal visual perception. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13193–13203.
[28] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang,
Tiejun Huang, and Bo Zhao. 2025. Video-xl: Extra-long vision language model
for hour-scale video understanding. InProceedings of the Computer Vision and
Pattern Recognition Conference. 26160–26169.
[29] Yin Song, Chen Wu, and Eden Duthie. 2024. aws-prototyping/long-llava-qwen2-
7b. https://huggingface.co/aws-prototyping/long-llava-qwen2-7b. Accessed:
2026-03-30.
[30] Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference
via python execution for reasoning. InProceedings of the IEEE/CVF international
conference on computer vision. 11888–11898.
[31] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol
Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al .2024.
Gemini 1.5: Unlocking multimodal understanding across millions of tokens of
context.arXiv preprint arXiv:2403.05530(2024).
[32] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. 2024. Videoa-
gent: Long-form video understanding with large language model as agent. In
European Conference on Computer Vision. Springer, 58–76.
[33] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. Longvideobench: A
benchmark for long-context interleaved video-language understanding.Ad-
vances in Neural Information Processing Systems37 (2024), 28828–28857.
[34] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. 2024.
Pllava: Parameter-free llava extension from images to videos for video dense
captioning.arXiv preprint arXiv:2404.16994(2024).
[35] Zhucun Xue, Jiangning Zhang, Xurong Xie, Yuxuan Cai, Yong Liu, Xiangtai
Li, and Dacheng Tao. 2025. Adavideorag: Omni-contextual adaptive retrieval-
augmented efficient long video understanding.arXiv preprint arXiv:2506.13589
(2025).
[36] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit
Bansal, and Gedas Bertasius. 2024. A simple llm framework for long-range video
question-answering. InProceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing. 21715–21737.
[37] Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-llama: An instruction-tuned
audio-visual language model for video understanding. InProceedings of the
2023 conference on empirical methods in natural language processing: system
demonstrations. 543–553.
[38] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan
Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. 2024. Long
context transfer from language to vision.arXiv preprint arXiv:2406.16852(2024).
[39] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng,
Ziwei Liu, and Chunyuan Li. 2024. LLaVA-NeXT: A Strong Zero-shot Video
Understanding Model. https://llava-vl.github.io/blog/2024-04-30-llava-next-
video/
[40] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan
Li. 2024. Llava-video: Video instruction tuning with synthetic data.arXiv preprint
arXiv:2410.02713(2024).
[41] Yi-Fan Zhang, Qingsong Wen, Chaoyou Fu, Xue Wang, Zhang Zhang, Liang
Wang, and Rong Jin. 2024. Beyond LLaVA-HD: Diving into High-Resolution Large
Multimodal Models. arXiv:2406.08487 [cs.CV] https://arxiv.org/abs/2406.08487
[42] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong,
Bo Zhang, Tiejun Huang, and Zheng Liu. 2024. Mlvu: A comprehensive bench-
mark for multi-task long video understanding.arXiv preprint arXiv:2406.042642,
5 (2024), 6.
9

Qin et al.
Supplementary Material
A Additional Ablation Studies
Frame Sampling Budget.We investigate the sensitivity of EC-
RAG to the number of sampled frames by evaluating LLaVA-Video-
7B [40] under three frame budgets: 8, 16, and 32 on Video-MME [ 6].
As shown in Figure 5, EC-RAG consistently outperforms the base-
line at every frame budget across all duration categories. Notably,
the baseline shows diminishing returns as the frame count in-
creases—particularly on long videos, where doubling from 16 to
32 frames yields only marginal improvement, suggesting that sim-
ply adding more visual input cannot compensate for the lack of
temporal and textual context. In contrast, EC-RAG maintains sub-
stantial gains at all budgets, with the largest absolute improvement
observed at 16 frames, indicating that event-chain retrieval is es-
pecially effective when the visual input is moderately sparse and
the model most needs supplementary evidence to bridge temporal
gaps.
Retrieval Threshold.Table 5 investigates the effect of the re-
trieval similarity threshold 𝜏on both efficiency and accuracy. Lower
thresholds (𝜏=0.0–0.1) retain nearly all retrieved text, injecting up
to 4.4 K supplementary tokens and pushing inference time to over
46 s per question, yet only achieve 64.1–64.7% overall. As 𝜏in-
creases, the token budget and latency decrease steadily; at 𝜏=0.3,
EC-RAG processes 2.4 K tokens in 18 s while reaching the peak
overall accuracy of 65.1%, with the best short-video (73.4%) and
long-video (61.2%) scores. This non-monotonic pattern indicates
that unfiltered retrieval introduces noisy evidence that can mislead
the model. Beyond 𝜏=0.3, accuracy degrades as useful context is
progressively pruned: 𝜏=0.5drops to 60.8% and 𝜏=1.0—which ef-
fectively discards all retrieved text (0.0 K tokens)—falls to 58.9%.
We therefore adopt 𝜏=0.3as the default, which reduces latency by
roughly 65% compared to unfiltered retrieval while maximizing
accuracy.
Cross-Backbone RAG Comparison.Table 6 compares EC-
RAG with Video-RAG [ 20] and TV-RAG [ 1] under two additional
backbone settings. On Video-LLaVA-7B [ 13] with 8 frames, all three
RAG methods substantially improve over the baseline (39.6%), with
EC-RAG achieving the highest overall accuracy of 46.0%, outper-
forming TV-RAG by +0.5 and Video-RAG by +1.0. The advantage
is most pronounced on long videos (+0.8 over TV-RAG), where
event-chain retrieval provides more temporally precise context
than frame-level or segment-level approaches. On LongVA-7B [ 38]
with 32 frames, the same trend holds: EC-RAG reaches 62.8% over-
all, surpassing TV-RAG (61.7%) and Video-RAG (60.1%). EC-RAG
consistently leads across all duration categories under both back-
bones, confirming that structured event-level retrieval generalizes
across different LVLMs and frame budgets.
B Task-Type Breakdown
Table 7 presents a fine-grained breakdown of accuracy across all
12 task types on Video-MME under 8-frame and 16-frame set-
tings. With 8 frames, EC-RAG improves the average by +5.6 points
(54.6%→60.2%), with the largest gains on Information Synopsis
(+9.6), Spatial Reasoning (+9.0), and OCR (+7.9)—tasks that heavily
rely on textual or spatial cues where event-chain evidence provides
direct support. At 16 frames, the average gain further widens to +7.8Table 5: Performance with different retrieval thresholds on
Video-MME [6] with LLaVA-Video-7B [40].
𝜏#Token Time Short Medium Long Overall
0.0 4.4K 52s 72.4 61.6 60.0 64.7
0.1 4.0K 46s 71.8 60.5 59.9 64.1
0.2 3.5K 23s 70.3 60.4 57.8 62.8
0.3 2.4K 18s 73.4 60.7 61.2 65.1
0.4 1.6K 13s 70.1 60.6 57.5 62.7
0.5 1.2K 12s 68.9 58.2 55.3 60.8
1.0 0.0K 10s 66.5 56.4 53.7 58.9
Table 6: Comparison with other RAG methods across differ-
ent backbones on Video-MME [6].
Model Params Frames Short Medium Long Overall
X=Video-LLaVA [13] 7B 8 44.6 38.3 35.8 39.6
X+Video-RAG [20] 7B 8 49.5 43.0 42.5 45.0
X+TV-RAG [1] 7B 8 49.7 44.3 42.6 45.5
X+EC-RAG 7B 8 50.1 44.6 43.4 46.0
X=LongVA [38] 7B 32 60.9 49.3 44.0 51.4
X+Video-RAG [20] 7B 32 65.4 59.1 55.7 60.1
X+TV-RAG [1] 7B 32 66.2 62.1 58.7 61.7
X+EC-RAG 7B 32 67.1 62.5 58.9 62.8
Figure 5: Accuracy of LLaVA-Video-7B [ 40] with and without
EC-RAG under different frame budgets on Video-MME [ 6].
EC-RAG provides consistent gains at all budgets, with the
largest improvement at 16 frames.
points (58.0%→65.8%), with Action Reasoning seeing the most dra-
matic improvement (+13.7) followed by Attribute Perception (+10.4)
and Information Synopsis (+9.9). EC-RAG achieves the best score
on 10 out of 12 task types at 8 frames and 12 out of 12 at 16 frames,
demonstrating that the event-chain retrieval complements visual
features comprehensively across diverse question types rather than
benefiting only a narrow subset.
C More Qualitative Results
We present additional qualitative results of LLaVA-Video-7B [ 40]
with EC-RAG on two representative examples from Video-MME [ 6]
10

EC-RAG for Long Video Understanding
Table 7: Per-task-type accuracy (%) on Video-MME [ 6] with LLaVA-Video-7B [ 40] as the backbone. Gain denotes the absolute
improvement over the baseline. Best results per frame setting are in bold.
Method Frames TmpR ActR ActRec AttrP SpaP SpaR TmpP InfoS OCR ObjR ObjRec Count AVG Gain
LLaVA-Video [40] 8 41.8 47.4 55.6 69.4 59.3 71.4 56.4 70.0 48.9 51.1 58.5 37.7 54.6 –
LLaVA-Video + EC-RAG 8 46.9 49.8 55.9 73.9 59.6 80.4 60.0 79.6 56.8 60.6 63.6 40.3 60.2 +5.6
LLaVA-Video [40] 16 45.2 51.2 57.5 68.0 57.4 73.269.174.0 50.4 55.4 64.141.8 58.0 –
LLaVA-Video + EC-RAG 16 51.4 64.9 62.6 78.4 61.1 80.4 69.6 83.9 59.0 66.1 67.2 42.3 65.8 +7.8
Figure 6: Qualitative example 1: a history documentary where the question requires reasoning about the temporal order of
events after a key battle. The baseline confuses visually similar city depictions, while EC-RAG traces the event chain through
ASR narration to identify the correct subsequent event.
in Figures 6 and 7. Both cases involve temporal reasoning ques-
tions from long documentary videos, where the baseline model
fails due to the inherent ambiguity of sparsely sampled frames. As
illustrated, by constructing temporal event chains from multimodalevidence—particularly ASR transcripts that capture the chronologi-
cal progression of topics and events—EC-RAG enables the model to
accurately resolve temporal ordering and causal relationships that
are indistinguishable from visual information alone. The results
11

Qin et al.
Figure 7: Qualitative example 2: a science documentary where the question asks about the introduction order of four topics.
The baseline is misled by recurring visual elements across segments, while EC-RAG recovers the correct topic sequence from
the structured event chain.
demonstrate that event-level retrieval effectively bridges the gap between sparse visual sampling and the dense temporal reasoning
required for long video understanding.
12