# Empowering Agentic Video Analytics Systems with Video Language Models

**Authors**: Yuxuan Yan, Shiqi Jiang, Ting Cao, Yifan Yang, Qianqian Yang, Yuanchao Shu, Yuqing Yang, Lili Qiu

**Published**: 2025-05-01 02:40:23

**PDF URL**: [http://arxiv.org/pdf/2505.00254v1](http://arxiv.org/pdf/2505.00254v1)

## Abstract
AI-driven video analytics has become increasingly pivotal across diverse
domains. However, existing systems are often constrained to specific,
predefined tasks, limiting their adaptability in open-ended analytical
scenarios. The recent emergence of Video-Language Models (VLMs) as
transformative technologies offers significant potential for enabling
open-ended video understanding, reasoning, and analytics. Nevertheless, their
limited context windows present challenges when processing ultra-long video
content, which is prevalent in real-world applications. To address this, we
introduce AVA, a VLM-powered system designed for open-ended, advanced video
analytics. AVA incorporates two key innovations: (1) the near real-time
construction of Event Knowledge Graphs (EKGs) for efficient indexing of long or
continuous video streams, and (2) an agentic retrieval-generation mechanism
that leverages EKGs to handle complex and diverse queries. Comprehensive
evaluations on public benchmarks, LVBench and VideoMME-Long, demonstrate that
AVA achieves state-of-the-art performance, attaining 62.3% and 64.1% accuracy,
respectively, significantly surpassing existing VLM and video
Retrieval-Augmented Generation (RAG) systems. Furthermore, to evaluate video
analytics in ultra-long and open-world video scenarios, we introduce a new
benchmark, AVA-100. This benchmark comprises 8 videos, each exceeding 10 hours
in duration, along with 120 manually annotated, diverse, and complex
question-answer pairs. On AVA-100, AVA achieves top-tier performance with an
accuracy of 75.8%.

## Full Text


<!-- PDF content starts -->

arXiv:2505.00254v1  [cs.CV]  1 May 2025Empowering Agentic Video Analytics Systems with
Video Language Models
Yuxuan Yan2†, Shiqi Jiang1, Ting Cao1, Yifan Yang1, Qianqian Yang2,
Yuanchao Shu2, Yuqing Yang1, Lili Qiu1
1Microsoft Research2Zhejiang University
1{shijiang, ticao, yifanyang, yuqyang, liliqiu}@microsoft.com
2{yanyx44, qianqianyang20, ycshu}@zju.edu.cn
ABSTRACT
AI-driven video analytics has become increasingly impor-
tant across diverse domains. However, existing systems are
often constrained to specific, predefined tasks, limiting their
adaptability in open-ended analytical scenarios. The recent
emergence of Video-Language Models (VLMs) as transfor-
mative technologies offers significant potential for enabling
open-ended video understanding, reasoning, and analytics.
Nevertheless, their limited context windows present chal-
lenges when processing ultra-long video content, which is
prevalent in real-world applications. To address this, we
introduce Avas , a VLM-powered system designed for open-
ended, advanced video analytics. Avas incorporates two key
innovations: (1) the near real-time construction of Event
Knowledge Graphs (EKGs) for efficient indexing of long
or continuous video streams, and (2) an agentic retrieval-
generation mechanism that leverages EKGs to handle com-
plex and diverse queries. Comprehensive evaluations on pub-
lic benchmarks, LVBench and VideoMME-Long, demonstrate
thatAvas achieves state-of-the-art performance, attaining
62.3% and 64.1% accuracy, respectively, significantly surpass-
ing existing VLM and video Retrieval-Augmented Generation
(RAG) systems. Furthermore, to evaluate video analytics in
ultra-long and open-world video scenarios, we introduce
a new benchmark, Avas -100. This benchmark comprises 8
videos, each exceeding 10 hours in duration, along with 120
manually annotated, diverse, and complex question-answer
pairs. On Avas -100, Avas achieves top-tier performance with
an accuracy of 75.8%.
1 INTRODUCTION
Video analytics [ 6,17,19,26] has emerged as a transfor-
mative technology across a wide array of domains, such as
security surveillance, intelligent transportation systems, in-
dustrial automation, and retail monitoring. By harnessing
the power of deep learning models, video analytics systems
extract meaningful insights, identify patterns, and generate
†The Work is done during internship at Microsoft Research.actionable information from video data, enabling organiza-
tions to monitor, detect, and respond to events with greater
efficiency and precision. The desired features of video analyt-
ics systems necessitate a variety of capabilities. We classify
the intelligence levels of existing and future video analytics
systems into five levels, labeled as L1 through L5:
L1systems for specific classification, segmentation, and
detection using models e.g.,ResNet [ 15] and EfficientDet [ 34]
to extract spatial information from video data, including
object classes and bounding boxes, etc [6, 17, 19].
L2systems go beyond spatial information extraction by
enabling causal event detection and analytics, i.e.,identify-
ing short-term events. They use models like C3D [ 37] and
ActionFormer [ 45] to detect and localize events ( e.g.,actions,
activities, anomalies) through spatiotemporal modeling.
L3systems extend L2’s spatiotemporal detection by incor-
porating neural language processing (NLP). Using models
like CLIPBERT [ 21], they interpret and respond to natu-
ral language queries ( e.g.,"What animals appeared in the
videos?") instead of merely outputting fixed labels. Though
queries are still limited to specific domains, L3 systems greatly
improve user interaction for video analytics.
Despite significant advancements, current video analytics
systems [ 6,17,19,23,27,33,47,49] primarily focus on L1 to
L3 levels. These systems are designed for closed-end analytics,
relying on specialized models for specific tasks, which limits
their flexibility and adaptability. Consequently, we envision
L4 and L5 systems to enable open-end analytics.
L4systems enable open-ended video comprehension, rea-
soning, and analytics, marking a major advancement. They
support general-purpose analytics on video sources, han-
dling natural language queries and responses while enabling
complex, long-term spatiotemporal reasoning. For example,
they can answer questions like: "What abnormal events oc-
curred in the past ten hours?" ,"What caused the person to fall?" ,
or"How did the animals behave after appearing on camera?"
L5systems advance L4 by autonomously integrating exter-
nal public or domain-specific knowledge to uncover explicit

Yan and Jiang et al.
and implicit connections between videos and world knowl-
edge, fulfilling the ultimate goal of video analytics: deriving
meaningful insights and offering actionable automations.
In this paper, we delve into the development of L4 video
analytics systems, leveraging the transformative potential
of video-language models (VLMs). By combining vision and
language understanding, VLMs enable generalized visual
detection and advanced video comprehension, including
causal reasoning, key information retrieval, and human-
interpretable explanations. The integration of perception,
reasoning, and interaction makes VLMs highly adaptable,
positioning them as a key foundation for L4 systems.
However, integrating VLMs into video analytics poses
significant challenges, primarily due to the limited context
window of current VLMs compared to the extensive duration
of video sources in typical video analytics scenarios. While L1
to L3 systems handle spatial and short-term causal event de-
tection by processing frames independently or within small
sliding windows ( e.g.,a few seconds), L4 systems require
collective analysis of related frames for long-term causal de-
tection, summarization, and reasoning. Current VLMs, like
QwenVL [ 38], can processing up to 768 frames, covering
minutes or hours of video. However, video analytics often
involves much longer sources, spanning hundreds of hours
or continuous streams, far beyond the capabilities of existing
VLMs.
Recent studies [ 31,32,40] that attempt to extend the con-
text window of VLMs remain inadequate for processing video
sources spanning hundreds of hours. Retrieval-augmented
generation (RAG) frameworks [ 10,11,14,30] aim to address
similar limitations by first retrieving relevant frames from
massive contents and then generating final answers. How-
ever, these approaches still face significant challenges in han-
dling the video modality (as detailed in §7), leading to both
reduced analytics accuracy and substantial computational
overhead.
In this paper, we present Avas , a system that integrates
VLMs into video analytics to enable L4 capabilities. The core
innovation of Avas is its efficient indexing mechanism, de-
signed to handle extremely long video sources or unlimited
video streams, and by leveraging the index, Avas effectively
retrieves relevant information for a given query and gen-
erates accurate, robust responses. Specifically, Avas intro-
duces the following key features that distinguish it from
existing systems: 1) analyzing extremely long videos, span-
ning hundreds of hours or even unlimited video streams;
2) supporting for near-real-time ( e.g.,at more than 1 FPS)
index construction and analytics; and 3) handling diverse
and complex queries including temporal grounding and rea-
soning, summarization, event and entity understanding, and
key information retrieval.Particularly, Avas introduces two key components: near-
real-time index construction andagentic retrieval and genera-
tion. During the index construction phase (§4), we propose
event knowledge graphs (EKGs) as an indexing mechanism
for video analytics. Unlike traditional knowledge graphs
(KGs) used in text-based RAG systems [ 10,14,30], EKGs rep-
resent a flow of insightful events, effectively capturing video
dynamics and temporal consistency. Using a small VLM, such
as QwenVL-7B, Avas extracts information to construct the
EKG. To optimize this process, we introduce techniques that
enable near-real-time index construction, achieving more
than 5 FPS on typical edge servers equipped with 2 ×4090
GPUs.
In the retrieval and generation phase (§5), Avas employs
an agentic search mechanism instead of directly retrieving in-
formation from the constructed index. This approach allows
Avas to proactively retrieve more relevant information by
utilizing contextual hints captured within the EKG, enables
Avas to handle complex queries, including summarization
and multi-hop reasoning etc. Furthermore, we introduce
techniques to enhance Avas to robustly generate answers
based on the retrieved information.
We evaluate Avas on multiple public video understand-
ing benchmarks, including LVBench [ 39] and VideoMME-
Long [ 12]. These benchmarks collectively comprise approx-
imately 400 videos and 2,500 questions. We compare Avas
against a diverse range of baselines, including state-of-the-
art (SOTA) VLMs such as GPT-4o [ 4], Gemini [ 35], Phi-4-
Multimodal [ 1], Qwen2.5-VL-7B [ 38], InternVL2.5-8B [ 7],
and LLaVA-Video-7B [ 48], as well as typical video-RAG sys-
tems such as VideoTree [ 43], VideoAgent [ 42], DrVideo [ 25],
and VCA [ 44]. On both benchmarks, Avas establishes new
SOTA performance, achieving 62.3% on LVBench and 64.1%
on VideoMME-Long, respectively, significantly surpassing
baselines by up to 16.9% and 5.2%.
In addition to the public video understanding benchmarks,
we introduce a new benchmark, Avas -100, specifically de-
signed to evaluate L4 video analytics tasks. Avas -100 com-
prises 8 ultra-long videos, each exceeding 10 hours in dura-
tion, and includes a total of 120 manually annotated questions
and answers. The videos are carefully curated from typical
video analytics scenarios, and the questions cover multiple
key categories relevant to video analytics systems. Evalu-
ation results show that Avas achieves significantly better
performance on Avas -100 compared to various baselines,
with improvements of approximately 20.8%. In summary, we
make the following contributions in this paper:
•We propose Avas , the first L4 video analytics system
powered by VLMs, to the best of our knowledge.

Empowering Agentic Video Analytics Systems with Video Language Models
•Avas introduces near-real-time index construction
and agentic retrieval and generation, along with in-
novative techniques that enable key features for L4
video analytics, including open-ended analytics on
extremely long video sources in near-real-time.
•We evaluate Avas on two widely-used video under-
standing benchmarks, achieving SOTA performance
with 62.3% on LVBench and 64.1% on VideoMME-
Long.
•Furthermore, we present Avas -100, a benchmark specif-
ically designed for L4 video analytics systems, where
Avas demonstrates significant improvements, out-
performing baselines by approximately 20.8%.
2 RELATED WORK AND MOTIVATION
2.1 Video Analytics System and VLMs
The field of video analytics (VA) has seen significant ad-
vancements in recent years [ 6,19,23,27,33,47,49]. Lever-
aging emerging deep learning models, modern VA systems
can extract insightful information, such as object locations
or counts, from video streams processed on edge or cloud
servers.
Existing VA systems predominantly support closed-end an-
alytics (L1 to L3 systems as mentioed in §1), often relying on
shallow neural networks to extract predefined, task-specific,
and constrained information. For instance, Remix [ 17] lever-
ages fine-tuned EfficientDet [ 18] to generate bounding boxes
for pedestrians. Consequently, the flexibility and adaptabil-
ity of VA systems are fundamentally limited by the inherent
constraints of the specific models they employ.
Recently, video language models (VLMs), e.g.,GPT-4o [ 4],
Gemini [ 35,36], QwenVL [ 5,38] and Phi [ 1–3], have demon-
strated their transformative potential in video analytics tasks.
By leveraging the extensive world knowledge embedded in
large language models (LLMs), VLMs not only achieve gen-
eralized visual grounding but also exhibit advanced video
comprehension capabilities, such as zero-shot temporal and
spatial reasoning, contextual retrieval, and semantic under-
standing. More importantly, VLMs enable natural language
interaction, allowing users to dynamically query video con-
tent. This makes them particularly well-suited for addressing
diverse and unstructured open-ended analytics in real-world
scenarios, towards L4 VA systems.
However, adopting VLMs in VA systems is far from straight-
forward. Existing L1 to L3 VA systems, which rely on tra-
ditional DNNs, typically process each video frame indepen-
dently. In contrast, VLMs require related frames to be pro-
cessed collectively to infer causal relationships and temporal
dependencies across frames. This shift introduces significantShort (1.4 minutes) Medium (9.7 minutes) Long (39.7 mintues)
Total Needed Total Needed Total Needed
2144.8 12.1 (0.5%) 13924.1 68.1 (0.4%) 66847.1 82.3 (0.1%)
Table 1: A small portion of frames is required to answer
each specific question across the short, medium, and
long video subsets of the VideoMME [ 12] benchmark
using Qwen2-VL.
complexity, as existing VLMs are generally capable of han-
dling only minute-level or sub-hour-level videos due to the
limited context window inherent in language models.
In real-world video analytics scenarios, the scale of videos
to be analyzed is often vastly larger—spanning hundreds of
hours or more ( e.g.,monitoring wildlife behavior over an
entire month, as illustrated in Fig. 2). This creates a funda-
mental gap between the capabilities of VLMs and the de-
mands of VA systems, as the limited context window length
of VLMs directly restricts their ability to process videos of
such extensive durations effectively.
2.2 Long Video Understandings
Recent research also increasingly focuses on enabling long
video understandings [ 31,40]. Given the inherent limita-
tions of autoregressive language models, such as the con-
strained context window length, efforts have been directed
toward reducing the redundancy in video inputs to facilitate
the processing of extended video durations. For instance,
LongVU [ 31] and AdaRETAKE [ 40] introduce dynamic com-
pression mechanisms that prioritize video content based on
its relevance, selectively retaining frames or regions most per-
tinent to downstream language tasks. Similarly, NVILA [ 22]
addresses the efficiency-accuracy trade-off by optimizing
sampling strategies and resolution to fit within limited token
budgets.
While these approaches have succeeded in increasing the
number of frames that models can process and mitigating
the constraints of context windows to some extent, they fall
short of achieving a fundamental breakthrough. Most exist-
ing methods remain limited, with the supported video length
typically capped at approximately one hour [ 28], which is in-
sufficient to meet the demands of video analytics. Moreover,
as video length extends, the inference cost rises correspond-
ingly, further exacerbating the challenges of scaling these
systems.
2.3 Retrieval Augmented Generation
While the videos to be analyzed may span extensive dura-
tions, the frames necessary to respond to a specific query are
often limited. To validate this observation, we conducted an

Yan and Jiang et al.
...
 ... ...Semantic
Chunking
Event Desc.Temp.
RelationshipEntity
Extraction and
Linking
Entity Desc.
Entity-Event RelationshipEntity-Entity
Relationship
QueryThoughts
Consistency
Tri-View RetrievalLink to Events
Retrieved EventsEvents Frames
Agentic SearchingFR
BF
RB...
...
...
...RSA
SA
SA
SAFinal
Answer
Agentic Action SpaceVideo Streams Uniform Buffering Uniform Chunk Desc. Semantic Event Desc.
...
Event Knowledge Graph...
Near-Real-T ime Index ConstructionEKG with Entities
Entities
Agentic Retrieval and GenerationFRetrieve the next event on index
BRetrieve the previous event on index
RRe-query with new key words
Summary and answer SASmall VLM
Figure 1: The system overview of Avas .
experiment on VideoMME [ 25] using Qwen2-VL [ 38]. Specif-
ically, we first identified all questions for which Qwen2-VL
produced correct answers by uniformly sampling frames
from the videos at a rate of 1 FPS. For these questions, we
then determined the minimal set of frames required for the
VLM to generate the correct answer by iteratively reducing
the number of input frames using a binary search strategy1.
The results reveal that the frames necessary to answer a
given question constitute only a small fraction of the total
frames in the video, as shown in Table 1.
Based on this observation, an intuitive approach would
be to first retrieve the relevant frames corresponding to a
specific query and then generate the final answer based on
these frames, a method commonly referred to as Retrieval-
Augmented Generation (RAG). To retrieve potentially rele-
vant frames, a straightforward strategy involves vectorized
retrieval , where each frame of the video is embedded using a
vision-language model, such as CLIP [ 29]. At query time, the
embedding of the query is used to retrieve relevant frames
by comparing the similarity between the query embedding
and the vectorized frames.
However, the vectorized retrieval method’s limitations
stem from the detailed information contained in the query.
Notably, such an approach struggles to handle query-focused
1For example, we initially uniformly sample 100 frames from the video.
If the VLM can generate the correct answer based on these frames, we
then attempt to reduce the frame set to 50 frames. If 50 frames are still
sufficient to produce the correct answer, we further reduce the set to 25
frames. Conversely, if 50 frames are insufficient, we increase the set to 75
frames, iteratively refining the frame selection using binary search strategy.summaries [ 10] (e.g.,"What happened in the last few hours?")
or multi-hop queries [ 50] (e.g.,"What did the man do after
he opened the fridge?"), as the retrieved frames often fail to
capture key contexts that are not explicitly mentioned in the
query descriptions.
To enable effective retrieval, recent research has explored
two prominent approaches: video structuring and iterative
retrieval . For example, Video-RAG [ 24] structures videos by
utilizing various tools to extract information such as audio
transcripts (ASR), optical character recognition (OCR) results,
and object detection outputs. It then applies RAG techniques
to the structured information. However, this method is inher-
ently constrained by the tools employed for video structuring.
It is often impractical to predict in advance what types of
information need to be extracted and what corresponding
tools should be utilized, limiting its adaptability to diverse
and dynamic video analytics scenarios.
Alternatively, researchers have proposed obtaining rele-
vant frames through multiple iterative retrieval processes [ 25,
42–44]. For instance, VideoAgent [ 42] typically begins with
a coarse-grained sampling of video segments to establish
an initial high-level understanding. Based on this, the VLM
is prompted to decide which finer-grained segments to re-
trieve and analyze in subsequent iterations. However, these
approaches face significant challenges when applied to video
analytics scenarios involving extremely long videos. On one
hand, the initial coarse-grained sampling may become insuf-
ficient as video length increases, potentially missing critical
information. On the other hand, the iterative retrieval and

Empowering Agentic Video Analytics Systems with Video Language Models
analysis process becomes increasingly computationally ex-
pensive as video duration grows, making it impractical for
large-scale video analytics tasks.
Recent studies have advanced RAG techniques [ 10,11,14,
30] by incorporating knowledge graph construction to en-
hance the retrieval process. However, these works primarily
focus on text-only RAG problems, and adapting such ap-
proaches to video analytics remains a significant challenge
due to the complexity and multimodal nature of video data.
In this paper, we propose Avas , which, to the best of our
knowledge, is the first system to enable VLM-powered video
analytics by effectively addressing the aforementioned chal-
lenges.
3AVAS SYSTEM OVERVIEW
The key idea of Avas lies in leveraging a small VLM to ef-
ficiently structure video streams into discrete events , then
linking these events by extracting insights from each to con-
struct a comprehensive index. Given a specific query, Avas
leverages this index to proactively retrieve relevant infor-
mation from both the index and the associated raw frames.
Ultimately, the retrieved data are utilized by the VLM to
produce a coherent and contextually appropriate response.
To build Avas as the L4 video analytics system, we es-
tablished the following design principles: 1) The analytics
should be scalable to any volume of video data, i.e.,exceeding
hundreds of hours, while ensuring that the computational
overhead remains independent of the video length; 2) The
index construction must operate in near-real-time, allowing
the system to support timely event analytics; 3) The system
should accommodate not only fact-based retrieval queries
but also query-focused summarization and multi-hop queries,
supporting open-ended analytics.
To this end, as depicted in Fig. 1, Avas system is composed
of two primary components: near-real-time index construc-
tionandagentic retrieval and generation . Within each compo-
nent, we introduce a set of techniques designed to effectively
realize the established design principles.
In the index construction phase, our objective is to design
an effective index while ensuring construction efficiency. To
achieve this, we introduce the event knowledge graph (EKG)
to structure video streams (§4.1). An EKG is a specialized
form of a knowledge graph (KG) designed to represent and
organize at the granularity of events and their interconnec-
tions. Recognizing that events unfold across varying tem-
poral scales, we propose semantic chunking (§4.2) to extract
meaningful events from video streams. Specifically, video
streams are segmented into small, uniform chunks ( e.g.,3-
second intervals), and a small VLM, such as Qwen2.5-VL-7B,
is periodically employed to generate detailed content de-
scriptions for these chunks using carefully crafted prompts.Subsequently, neighboring chunks are merged into larger se-
mantic chunks by identifying semantically equivalent textual
descriptions with BertScore [ 9]. For each semantic chunk,
the small VLM extracts entities and their relationships. Iden-
tical entities across different events are linked to ensure
consistency and coherence. Ultimately, Avas facilitates the
continuous construction of an EKG for a given video stream,
regardless of its length, providing a comprehensive represen-
tation of semantic events, entities, and their interrelations
in near-real-time on typical edge servers.
In the retrieval and generation process, we aim to lever-
age the constructed index for efficiently retrieving essential
and minimal information, and to robustly generate the fi-
nal answer based on the retrieved data. To achieve this, we
first introduce the concept of tri-view retrieval . Specifically,
a given query undergoes simultaneous retrieval across three
dimensions: events, entities, and visual embeddings. This
approach ensures the acquisition of comprehensive and rele-
vant information pertaining to the query. To further support
complex queries in L4 VA systems e.g.,query-focused sum-
marization and multi-hop queries, we propose an agentic
searching mechanism. In particular, by utilizing the LLM as
an agent, Avas proactively explores to retrieve additional
information from events linked to those retrieved in earlier
steps. Avas explores multiple pathways to gather informa-
tion and formulates a response to the query based on the
collected data. Finally, we introduce the thoughts-consistency
strategy, which selects the most coherent and accurate final
answer from multiple generated candidates.
4 NEAR-REAL-TIME INDEX
CONSTRUCTION
4.1 Event Knowledge Graph
An Event Knowledge Graph (EKG) is a structured represen-
tation of events and their interconnections, linking entities,
timestamps, locations, and other contextual information to
offer a holistic understanding of events and their dependen-
cies. By employing an EKG, the content of a video can be
organized into a sequence of events, associating groups of
entities with specific events and capturing their intricate
relationships.
Although existing works, such as GraphRAG [ 10] and
LightRAG [ 14], utilize knowledge graphs (KGs) to construct
retrieval indices, we argue that EKGs are more suitable for
video data. The rationale lies in the fundamental difference
between the two: KGs focus on static entities ( e.g.,people,
locations, concepts) and their attribute-based relationships,
whereas EKGs prioritize modeling dynamic events and their
spatiotemporal evolution.
Fig. 2 illustrates an example video alongside its correspond-
ing KG and EKG. As depicted, the EKG effectively captures

Yan and Jiang et al.
Figure 2: An example of a constructed event knowledge
graph and a knowledge graph from wildlife surveil-
lance scenarios for video analytics.
key events and their transitions, while representing entities
with finer granularity within specific events. This enables
EKG-based retrieval to support more sophisticated queries,
such as event summaries, multi-hop temporal reasoning, and
other complex analyses. In contrast, KGs, which only encap-
sulate entities across the entire video, lack the capability to
fundamentally support such advanced queries.
Formally, we define our EKG Gas follows:
G=(E,U,R), (1)
whereE={𝑒𝑖}|E|
𝑖=1represents the temporally ordered set
of events,U={𝑢𝑗}|U|
𝑗=1denotes the entities extracted from
the video within each event, and R=R𝑒𝑒∪R 𝑢𝑢∪R 𝑢𝑒
encompasses three types of relationships: 1) temporal event-
event relationsR𝑒𝑒, such as before andafter , which encode
temporal logic constraints; 2) semantic entity-entity relations
R𝑢𝑢, akin to the relationships found in conventional KGs;
and 3) participation relations R𝑢𝑒, which associate entities
with their contextual roles within specific events.
4.2 Semantic Chunking
To construct the EKG as an index, it is essential to extract
events and their corresponding descriptions from videos.
Although current VLMs demonstrate remarkable capabilities
in event detection and transcription, their application in
video analytics scenarios remains challenging.
On one hand, large VLMs, such as Qwen2.5-VL-72B, can
achieve high accuracy in event detection and transcription,
but their substantial computational overhead makes it diffi-
cult to process video streams in near-real-time, particularly
on resource-constrained edge servers. On the other hand,
small VLMs, such as Qwen2.5-VL-7B, offer reduced latency
but suffer from performance degradation as the length of
Uniform Chunk IndexBERT Score
Manually labeled
chunks to be mergedFigure 3: Merging uniform chunks into semantic
chunks guided by the pairwise BERTScore distribu-
tion.
the video increases. Furthermore, both large and small VLMs
are limited by their constrained context windows. To handle
long video content, a common approach is to partition the
content uniformly, a process known as chunking. However,
events in videos naturally occur at varying and diverse tem-
poral scales. Inaccurate chunking can disrupt the coherence
of individual events, thereby increasing the difficulty for
VLMs to accurately detect and transcribe them.
To address this, we propose a semantic chunking approach.
The core idea involves processing a video stream in the fol-
lowing steps: First, we perform uniform buffering, e.g.,di-
viding the video into fixed-length chunks of 3 seconds each.
Next, a small VLM , e.g.,Qwen2.5-VL-7B, is employed to ex-
tract representative event descriptions from these chunks
with proper prompts. Based on the generated event descrip-
tions, we utilize a text embedding model, such as BERT [ 9],
to measure the similarity between neighboring events. Adja-
cent events with high similarity are then merged into a single
event. Ultimately, this process enables Avas to partition the
entire video into semantically meaningful chunks while si-
multaneously extracting their corresponding descriptions.
Particularly, an input video 𝑉is initially divided into uni-
form chunks 𝑐𝑖, and a small VLM is employed to generate
textual descriptions 𝑑𝑖for each chunk 𝑐𝑖. Subsequently, the
semantic similarity between any two uniform chunks is mea-
sured by computing the pairwise BertScore [ 46] for(𝑑𝑖,𝑑𝑗).
Higher similarity scores suggest that the same event may
occur across these chunks, making them candidates for se-
mantic merging. Specifically, we adopt two criteria to deter-
mine whether certain uniform chunks can be merged into
a single semantic chunk: 1) Within a semantic chunk, the
similarity between any two uniform chunks must exceed a

Empowering Agentic Video Analytics Systems with Video Language Models
predefined threshold ( e.g.,0.65 in our implementation); 2)
After merging, the similarity between the boundaries of ad-
jacent semantic chunks must fall below a sufficiently low
threshold. Figure 3 illustrates the semantic chunking process,
where a video initially divided into 18 uniform chunks is suc-
cessfully merged into 9 semantic chunks. Once merged, the
small VLM is further utilized to summarize each semantic
chunk.
It is important to highlight, although pairwise BertScore
computations are performed multiple times, Avas efficiently
schedules these computations in parallel, leveraging the hard-
ware parallelism (§6). Consequently, the semantic chunking
process does not become a bottleneck in the near-real-time
index construction phase, as detailed in §7.
4.3 Entity Extraction and Linking
In addition to extracting event information from videos, Avas
also identifies entities and their relationships, as illustrated
in Fig. 1. Similar to the approach in [ 10], we utilize a small
VLM to extract entities and their relationships from videos
using carefully designed prompts for each event.
The identified entities, however, tend to be highly redun-
dant across events within the EKG. Such redundancy not
only increases storage requirements but also hampers re-
trieval efficiency. Thus, it is necessary to de-duplicate and
link these entities. Existing works [ 10,11,14], which pri-
marily focus on text-only RAG problems, typically rely on
exact string matching strategies for entity de-duplication.
However, in the context of video analytics and EKG, entities
are independently extracted from each event by the VLM,
leading to potential inconsistencies in entity descriptions for
the same concept across different events, e.g.,, "raccoon" and
"procyon lotor".
To address this, Avas employs a text embedding model
e.g.,JinaCLIP [ 20], to encode all extracted entities into vector
representations. Using embedding similarities as a metric,
we apply a standard K-means clustering algorithm to group
entities. This approach ensures that semantically similar
entities are de-duplicated and linked by forming unified clus-
ters. Subsequently, the centroid of each cluster is used as the
representative for the linked entities.
Ultimately, the constructed EKG is stored in a database
comprising five tables: events, entities, event-to-event rela-
tionships, entity-to-entity relationships, and entity-to-event
relationships. Additionally, the raw video frames are vector-
ized using JinaCLIP [ 20] and linked to their corresponding
events, enabling comprehensive retrieval in the following
phase.
0.5 0.3 0.3 0.1
0.7 0.5 0.4 0.4
0.8 0.6 0.6 0.4Borda Counting
0.84
0.250.85
0.53
0.53
0.33 0.25 0.25 0.170.35 0.25 0.2 0.20.42 0.25 0.25 0.08 Similarities 
Event V iewNormalized Score
Entity V iew
Frame V iewSelected Top-4 EventsFigure 4: An illustration of tri-view retrieval and borda
counting on the retrieved events.
5 AGENTIC RETRIEVAL AND
GENERATION
In the agentic retrieval and generation stage, our primary ob-
jectives are to effectively retrieve relevant information using
the constructed EKG and to generate robust, contextually ac-
curate responses based on the retrieved data. To achieve this,
we introduce an agentic searching mechanism that explores
multiple retrieval pathways within the index to gather the
necessary information. Additionally, we propose a thought
consistency strategy, enabling Avas to generate and eval-
uate multiple responses from different retrieval pathways,
ultimately selecting the most appropriate one.
5.1 Tri-View Retrieval
To comprehensively retrieve relevant information from the
index for a given query, Avas employs a three-view retrieval
process: the first view targets events, enabling retrieval at
the event level to provide information for event summary-
related queries. Specifically, the query is encoded using the
text encoder JinaCLIP [ 20] and matched against the events
table in the constructed EKG. The second view focuses on
entities, offering insights into basic facts or item-specific
queries. For this, we leverage the entity centroids extracted
and aggregated as detailed in §4.3 to facilitate retrieval. The
third view utilizes vision embeddings of raw video frames as
complementary information. The retrieved entities and raw
frames are subsequently linked to their associated events
through the constructed EKG.
It is important to note that the retrieved events should be
ranked. Ranking is not only crucial for filtering noise from
the retrieved results but also essential for enabling agentic
searching, as detailed in §5.2. A straightforward ranking
method, such as similarity-based ranking, cannot be directly
applied to Avas due to the retrieved events originating from
three distinct views. To integrate these results, we propose
to use a weighted Borda counting approach.
Fig. 4 illustrates the process of using Borda counting to
integrate and rank retrieved events from the three views in
Avas . Specifically, we select the top 𝐾events from each view
and rank them based on their calculated similarities within

Yan and Jiang et al.
Root
SARQFB
SARQFB SA RQFB SA RQFB
SASASASASA SASASA SA
Figure 5: An example of agentic tree search with four
actions and a depth of three, yielding 13 distinct path-
ways for information gathering and response genera-
tion.
that view. Subsequently, the similarities of these 𝐾events
are normalized to compute their Borda scores:
𝑠𝑚(𝑒𝑗)=sim𝑚(𝑒𝑗)Í
𝑒𝑘∈E𝑚sim𝑚(𝑒𝑘), (2)
whereE𝑚represents the set of events retrieved from view
𝑚. The final Borda score for each event 𝑒𝑗is then obtained
by summing its scores across all views:
𝑠(𝑒𝑗)=∑︁
𝑚𝑠𝑚(𝑒𝑗), (3)
Finally, the aggregated Borda scores 𝑠(𝑒𝑗)are used to rank
all retrieved events.
5.2 Agentic Searching on Graph
The retrieved events mentioned above can be directly utilized
to generate the final answer. However, to support complex
queries, such as query-focused summaries and multi-hop
queries, Avas searches for additional relevant information
by leveraging the relationships between events and entities
within the constructed EKG. To enable efficient exploration,
we propose the agentic searching on graph approach.
This approach is inspired by human behavior when retriev-
ing and reasoning about information in videos. Specifically,
individuals typically begin by identifying key clips based
on retrieval keywords, then gather additional context by
exploring clips preceding or following the identified ones.
To achieve a more comprehensive understanding, they may
also refine their search by using alternative keywords, iterat-
ing through this process as needed. Similarly, in our agentic
searching process, we define the agentic action space as fol-
lows:
Forward (F) : this action extends the current retrieval by
including temporally subsequent events on the EKG for all
events in the event list. It reflects the natural tendency of
humans to seek forward narrative progression when trying
to understand what happens next or how a situation evolves
over time.Backward (B) : complementing the forward action, this ac-
tion retrieves temporally preceding events, enabling a back-
ward exploration of the narrative to uncover prior context
or causal factors.
Re-query (RQ) : this action generates a new query rep-
resented by a list of keywords via a LLM and retrieves com-
plementary events as outlined in §5.1. It reflects the human
tendency to gather information from multiple perspectives
to achieve a more comprehensive understanding.
Summary and answer (SA) : this action utilizes the de-
scriptions of the retrieved events from the EKG and generates
the response to the specific query by employing a LLM.
By leveraging these predefined agentic actions, we design
the agentic searching process as a tree search framework. The
search begins with an initial retrieval based on the original
query, producing a list of relevant events that serves as the
root node of the search tree. At each depth level, a one-step
rollout of all four predefined actions is performed for each
active node. If the SA action is reached, the corresponding
search path is terminated. The rollout process continues
until the maximum tree depth is reached. This tree search
process explores diverse pathways to gather information
from the EKG; As iilustrated in Fig. 5, a tree with a depth of 3
would result in 13 distinct information-gathering paths and
corresponding answers. All generated answers are evaluated,
and the optimal one is selected using the thought consistency
method, which is detailed in §5.3.
A practical issue in the tree search process is the exponen-
tial growth in the number of retrieved events as the tree depth
increases. This not only introduces computational overhead
but also results in the accumulation of noisy or irrelevant
information. To mitigate this, we use a length constraint on
the maintained event list during the search process, i.e.,16 in
our implementation. When the number of retrieved events
exceeds this limit, we employ a drop strategy to discard less
relevant events based on their rankings described in §5.1.
5.3 Consistency Enhanced Generation
During the agentic tree search, multiple candidate answers
are generated at SA nodes across different pathways. To
determine the final answer, it is necessary to either select
or synthesize from these candidates. A straightforward ap-
proach would be majority voting. However, due to the diver-
sity of retrieval paths, only a small subset of these nodes is
likely to access the essential information with minimal noise,
producing high-quality answers. To this end, we introduce
the thoughts-consistency mechanism to identify and select
the most reliable final answer.
At each SA node, instead of generating the answer a single
time, we repeatedly generate answers multiple times using
a Chain-of-Thought (CoT) prompting scheme. Following the

Empowering Agentic Video Analytics Systems with Video Language Models
principle of self-consistency [ 41], correct answers are more
likely to emerge consistently across multiple valid reason-
ing trajectories during repeated generations. Specifically, we
evaluate the consistency not only across the generated an-
swers but also within their associated CoT traces. Nodes
demonstrating strong internal coherence, where the reason-
ing process aligns logically with the conclusion, are assigned
higher scores.
To formalize this process, we propose a scoring framework
that integrates both answer agreement and thought consis-
tency . At each SA node, we perform 𝑛rounds of sampling
using a temperature setting between 0.5and0.7, resulting in
a set of𝑛candidate outputs denoted as {(𝑎𝑖,𝑟𝑖)}𝑛
𝑖=1, where
𝑎𝑖is the answer and 𝑟𝑖is the associated reasoning trace.
LetA={𝑎(1),𝑎(2),...,𝑎(𝑇)}be the set of unique answers
among the 𝑛samples, where 𝑇is the number of distinct
answers. The answer agreement score 𝑆(𝑡)
𝑎for a candidate
answer𝑎(𝑡)is defined as the proportion of times it appears
in the samples:
𝑆(𝑡)
𝑎=|{𝑖|𝑎𝑖=𝑎(𝑡)}|
𝑛(4)
The thought consistency score 𝑆(𝑡)
𝑟for𝑎(𝑡)is computed as
the average BERTScore between all pairs of reasoning traces
associated with 𝑎(𝑡):
𝑆(𝑡)
𝑟=2
𝑘(𝑘−1)∑︁
1≤𝑖<𝑗≤𝑘BERTScore(𝑟𝑖,𝑟𝑗), (5)
where𝑘is the number of times 𝑎(𝑡)appears in the 𝑛samples.
The final score for each candidate answer combines these
two components:
𝑆(𝑡)
final=𝜆𝑆(𝑡)
𝑎+(1−𝜆)𝑆(𝑡)
𝑟, (6)
where𝜆∈[0,1]is a weighting parameter controlling the
trade-off between answer agreement and thought consis-
tency. In our implementation, we set it to 0.3, the parameter
tuning would be discussed in §7.4.3.
For each SA node, the candidate answer with the highest
𝑆(𝑡)
finalis selected as its definitive response. To enhance the
reliability of this final answer, we propose an additional
agentic action, Check Frames and Answer (CA) . This
action retrieves the raw video frames associated with the
events from the EKG and utilizes the VLM to generate a
refined response to the specific query. By doing so, this action
effectively supplements any missing information relevant
to the query that may have been overlooked during the
construction phase.
Specifically, after ranking all candidate answers from the
SA nodes using the consistency-enhanced scoring mecha-
nism, the top-2 nodes with differing answers are selected.
The video frames corresponding to their retrieved events
are extracted, and the VLM is prompted to generate a newresponse by directly attending to the visual evidence. Fur-
thermore, the thought-consistency mechanism is applied to
the CA nodes to bolster the reliability of the final generated
answer.
6 IMPLEMENTATION
Avas uses Qwen2.5-VL-7B for constructing EKGs, Qwen2.5-
32B for SA, and Gemini-1.5-Pro for CA. Local VLMs and
LLMs are deployed using LMDeploy[ 8] to enable acceler-
ated inference. Additionally, Avas adopts batch inference for
several key stages—including description generation, descrip-
tion merging, entity extraction, and tree search—to improve
efficiency and maximize GPU utilization. For text and vi-
sion embedding, we utilize JinaCLIP[ 20]. And we employ
BERTScore with the deberta-xlarge-mnli[ 16] checkpoint.
The storage of EKG and vector representations is based on
the implementation of [ 14], upon which we make further
modifications to suit the specific requirements of Avas .
During the EKGs construction stage, we carefully design
prompts to guide the extraction of structured information
from video. For general-purpose video understanding, we
employ a unified prompt that avoids introducing bias or prior
assumptions:
"...Your task is to extract and provide a detailed description
of the video segment, focusing on all key visible details... " .
For scenario-specific videos, we design the prompts to
emphasize scenario-relevant information. For example, in the
case of wildlife surveillance scenario, key information may
includes the timestamps of the recording, animal activities
(e.g.,presence, species, number, specific behaviors, etc.), and
environmental changes.
7 EVALUATION
7.1 Evaluation Settings
7.1.1 Benchmark. Avas is evaluated on two widely used
public long-video benchmark and one ultra-long video bench-
marks proposed by us, covering a broad range of video sce-
narios and problem types.
LVBench [39] stands out among publicly available bench-
marks for its exceptionally long average video duration, ap-
proximately 4100 seconds per video. It comprises 103 videos
with a total of 1549 questions, covering six distinct video
domains and addressing six task types including temporal
grounding, summarization, and reasoning.
VideoMME-Long [12] is a subset of the VideoMME bench-
mark, focusing on videos exceeding 20 minutes in duration,
with an average length of 2400 seconds. Comprising a total of
300 videos and 900 questions, the benchmark covers a wide
range of video themes, spanning 6 primary visual domains
with 30 subfields to ensure broad scenario generalizability,
and includes 12 distinct task types.

Yan and Jiang et al.
05101520253035404550556065Accuracy (%)
VCA(GPT-4o)
VideoTree(GPT-4o)
VideoAgent(GPT-4o)
Qwen2.5-VL-7B V
LLaV A-Video-7B V
InternVL2.5-8B V
Phi4-Multimodal-5.8B V
Gemini-1.5-Pro V
GPT-4o V
Qwen2.5-VL-7B U
LLaV A-Video-7B U
InternVL2.5-8B U
Phi4-Multimodal-5.8B U
Gemini-1.5-Pro U
GPT-4o U
A V A Video-RAG Vectorized-Retrieval Uniform-Sampling
a LVBench
05101520253035404550556065Accuracy (%)
VCA(GPT-4o)
DrVideo(GPT-4)
VideoTree(GPT-4o)
VideoAgent(GPT-4o)
Qwen2.5-VL-7B V
LLaV A-Video-7B V
InternVL2.5-8B V
Phi4-Multimodal-5.8B V
Gemini-1.5-Pro V
GPT-4o V
Qwen2.5-VL-7B U
LLaV A-Video-7B U
InternVL2.5-8B U
Phi4-Multimodal-5.8B U
Gemini-1.5-Pro U
GPT-4o U
A V A Video-RAG Vectorized-Retrieval Uniform-Sampling b VideoMME-Long
05101520253035404550556065707580Accuracy (%)
Qwen2.5-VL-7B V
LLaV A-Video-7B V
InternVL2.5-8B V
Phi4-Multimodal-5.8B V
Gemini-1.5-Pro V
GPT-4o V
Qwen2.5-VL-7B U
LLaV A-Video-7B U
InternVL2.5-8B U
Phi4-Multimodal-5.8B U
Gemini-1.5-Pro U
GPT-4o U
A V A Vectorized-Retrieval Uniform-Sampling cAvas -100
Figure 6: The achieved accuracy of Avas and various baselines on the LVBench, VideoMME-Long, and Avas -100
benchmarks.
Avas -100 is proposed by us, which is an ultra-long video
benchmark specially designed to evaluate video analysis
capabilities Avas -100 consists of 8 videos, each exceeding
10 hours in length, and includes a total of 120 manually
annotated questions. The benchmark covers four typical
video analytics scenarios: human daily activities, city walk-
ing, wildlife surveillance, and traffic monitoring, each sce-
nario contains two videos. The human daily activity scenario
features egocentric videos selected and stitched from the
Ego4D[ 13]. City walking and wildlife surveillance videos
are curated from publicly available recordings on YouTube,
capturing urban exploration and animal monitoring respec-
tively. Traffic monitor videos are composed from clips in the
Bellevue Traffic Video Dataset [ 6]. All questions are carefully
designed by human annotators, who also provide reference
answers as the ground truth. In addition, GPT-4o is utilized
to generate plausible distractor options. The accuracy is eval-
uated by analyzing Avas ’s responses to multiple-choice ques-
tions included in the benchmarks.
7.2 Baselines
We conduct a comprehensive comparison between Avas
and a wide range of baseline models, encompassing both
mainstream VLMs and specialized Video-RAG methods. The
VLM baselines include GPT-4o [ 4], Gemini-1.5-Pro [ 35], Phi-
4-Multimodal [ 2], Qwen2.5-VL-7B [ 5], InternVL2.5-8B [ 7],
and LLaVA-Video-7B [ 48]. Each of these models is evaluated
with two typical strategies: uniform sampling and vectorized
retrieval, where a CLIP-based retriever selects the top-K
relevant frames based on the user query. In addition to VLMs,
we benchmark Avas against SOTA Video-RAG frameworks,
including VideoTree[ 43], VideoAgent[ 42], DrVideo[ 25], and
VCA[ 44]. Among these, VideoTree, VideoAgent, and VCA
are built upon GPT-4o, while DrVideo leverages GPT-4.7.3 Overall Evaluation
7.3.1 Overall Performance. Fig. 6 illustrates the overall
accuracy achieved by Avas compared to various baselines on
the LVBench, VideoMME-Long, and Avas -100 benchmarks.
Across all three benchmarks, Avas consistently outperforms
the baselines. Specifically, on LVBench, Avas delivers a re-
markable 16.9% improvement, while on VideoMME-Long, it
advances the SOTA by approximately 5.2%. On the Avas -100
benchmark, Avas achieves an accuracy of 75.8%, significantly
surpassing all competing methods.
In detail, compared to video-RAG methods, Avas achieves
improvements of 21% and 7.8% on LVBench and VideoMME-
Long, respectively. When compared to vectorized retrieval-
based methods, Avas demonstrates gains of 16.9% on LVBench
and 20.8% on Avas -100. Furthermore, against uniform sam-
pling baselines, Avas improves performance by approxi-
mately 19.6% and 26.9% on LVBench and Avas -100, respec-
tively.
Notably, on Avas -100, when evaluated with extremely
long videos, Avas maintains robust performance, whereas
the baselines degrade significantly. This highlights the effec-
tiveness of Avas in handling L4 video analytics tasks.
7.3.2 Performance on Different Query Categories. We
also evaluate the accuracy achieved by Avas across typi-
cal query categories on LVBench. As illustrated in Fig. 7,
our approach achieves improvements of 16%, 5.3%, 35.6%,
21.2%, 17.5%, and 18.9% across six key task types: Temporal
Grounding, Summarization, Reasoning, Entity Recognition,
Event Understanding, and Key Information Retrieval, respec-
tively, compared to the uniform sampling and vectorized re-
trieval baselines powered by Gemini-1.5-Pro. Notably, Avas
demonstrates particularly strong performance on reasoning
tasks, which require identifying causal relationships between
events and linking preceding and succeeding events within

Empowering Agentic Video Analytics Systems with Video Language Models
TG SU RE ER EU KIR
Task Type020406080Accuracy(%)Uniform
Vectorized Retrieval
A V A
Figure 7: The accuracy achieved by Avas and the base-
lines across typical query categories on LVBench: Tem-
poral Grounding (TG), Summarization (SU), Reasoning
(RE), Entity Recognition (ER), Event Understanding
(EU), and Key Information Retrieval (KIR).
05101520253035404550556065707580Accuracy (%)
LVBench VideoMME-Long A V A-100A V A(Qwen2.5-32B + Gemini-1.5-Pro)
A V A(Qwen2.5-14B + Gemini-1.5-Pro)
Gemini-1.5-Pro-Vectorized Retrieval
Gemini-1.5-Pro-Uniform SamplingA V A(Qwen2.5-32B + Qwen2.5-VL-7B)
A V A(Qwen2.5-14B + Qwen2.5-VL-7B)
Qwen2.5-VL-7B-Vectorized RetrievalQwen2.5-VL-7B-Uniform Sampling
A V A(Qwen2.5-32B)
A V A(Qwen2.5-14B)
Figure 8: The accuracy achieved by Avas and baselines
across three benchmarks when utilizing different LLMs
and VLMs.
the video. This highlights Avas ’s ability to effectively locate
and extract critical information from long videos, thereby
enabling advanced L4 video analytics systems.
7.3.3 Performance under Different Configurations.
Fig. 8 shows the performance of Avas using different mod-
els configurations for SA and CA. For SA, two models were
used: Qwen2.5 14B and 32B. For CA, two models were used:
Qwen2.5-VL-7B and Gemini-1.5-Pro. The results show that
across the three benchmarks, Avas using Gemini-1.5-Pro for
CA achieved improvements of 18.9%, 5.2%, and 20.8% respec-
tively compared to the best baseline result using the same
model, while using Qwen2.5-VL-7B yielded improvements
of 13%, 7.2%, and 15% respectively, fully demonstrating the
effectiveness of our method. Notably, even when only using
Qwen2.5-32B and Qwen2.5-7B based on the textual content
from EKG without accessing raw frames, Avas can surpass
the performance of Qwen2.5-VL-7B on the three benchmarks
and also outperform most models shown in Figs. 6a, 6b, and
6c.
1 5 10 15
#Concatenated Videos3035404550556065Accuracy (%)Qwen2.5-VL-7B Uniform
Qwen2.5-VL-7B Vectorized Retrieval
A V A(Qwen2.5 14B + Qwen2.5-VL-7B)
Gemini-1.5-Pro UniformGemini-1.5-Pro Vectorized Retrieval
A V A(Qwen2.5 14B + Gemini-1.5-Pro)
A V A(Qwen2.5 14B)
Average Duration
0246810
Average Duration (h)
Figure 9: The accuracy achieved by Avas and the base-
lines across varying video lengths via concatenating
videos from LVBench.
7.3.4 Performance on Different Video Lengths. To
evaluate the robustness of Avas with respect to video length,
we conducted experiments on videos of varying durations.
Specifically, sequences of 3.3, 6.6, and 10 hours were cre-
ated by concatenating videos from the VideoMME-Long
benchmark. Performance was measured using identical ques-
tions across these varying video lengths. As illustrated in
Fig. 9, both Qwen2.5-VL-7B and Gemini-1.5-Pro baselines
exhibit significant performance degradation as video length
increases. When extended to 10 hours, their performance
declines by 4.6% and 8.2%, respectively, under the uniform
sampling method, compared to the original VideoMME-Long
benchmark. For the vectorized retrieval setting, the perfor-
mance drops are 4.6% and 5.5%, respectively. These results
highlight the limitations of these methods in scaling effec-
tively with increasing video length. In contrast, Avas consis-
tently maintains stable performance across all video lengths,
underscoring its robustness and scalability in handling video
data of any duration.
7.3.5 System Overhead. Avas is designed to enable the
near real-time construction of EKGs. As shown in Fig. 10,
we measured the average processing speed (in FPS) of Avas
while constructing EKGs from LVBench videos across vari-
ous hardware platforms, with the input video stream fixed
at 2 FPS. On 2×A100 GPUs, Avas achieved an impressive
processing speed of 6.7 FPS, significantly exceeding the in-
put stream rate. On a single RTX 4090, a typical edge server
hardware, Avas maintained a processing speed of 4.4 FPS,
still well above the input frame rate. Even on a single RTX
3090, Avas performed effectively, achieving 2.5 FPS. This
performance demonstrates its capability to support efficient,
near real-time EKG construction for L4 video analytics. The
overhead during the retrieval and generation phases will be
discussed further in §7.4.

Yan and Jiang et al.
A100×2
A100×1
L40S×2
L40S×1
A6000×2
A6000×1
RT X4090×2
RT X4090×1
RT X3090×2
RT X3090×10.02.55.07.5Processing FPSProcessing FPS
Input FPS
Figure 10: Total index construction overhead evaluated
on various types of typical edge server hardware.
Method Acc. Overhead(h)
MiniRAG 28.1 3.49
LightRAG 30.6 3.52
Avas 39.7 0.31
Table 2: The achieved accuracy and construction over-
head evaluated when using EKG and KG as index in
Avas and baseline models on the subset of LVBench.
The total video duration is around 1.2 hours.
7.4 Ablation Evaluation
We randomly sampled 20 videos and 305 corresponding ques-
tions from LVBench for our ablation study. All ablation ex-
periments were conducted on 2 ×A100 GPUs.
7.4.1 Different Index Construction Methods. We com-
pare Avas ’s Event Knowledge Graph (EKG) construction
method with two representative knowledge graph-based
construction methods: LightRAG [ 14] and MiniRAG [ 11].
Since both of them only support text-only construction, we
use the full set of descriptions obtained through the seman-
tic chunking (§4.2) as their input textual corpus. We use
Qwen2.5 7B to construct EKG and KG for Avas and base-
lines, respectively. In the retrieval and generation phase, we
use the same LLM, Qwen2.5 14B and the same settings, e.g.,
maximum number of tokens of retrieved events or entities.
As shown in Table 2, Avas demonstrates a significant
performance advantage over the baselines. Specifically, it
achieves 11.6% higher accuracy than MiniRAG and 9.1%
higher accuracy than LightRAG. Crucially, this improved
performance comes with substantially less construction over-
head, requiring only 0.31 hours compared to 3.49 and 3.52
hours for the baselines. The rationale is that baselines con-
struct KG based on massive uniform chunks, while Avas
utilizes the semantic chunks. This substantial gap in both
effectiveness and efficiency highlights that Avas ’s EKG con-
struction method not only yields higher-quality knowledgeMethodTree Search Depth
1 2 3 4
Avas (Qwen2.5 14B) 34.1 36.1 40.9 39.5
Avas (Qwen2.5 14B + Qwen2.5VL 7B) 49.3 52.1 53.8 50.2
Avas (Qwen2.5 14B + Gemini-1.5-Pro) 54.2 58.4 61.5 52.7
Tree Search Overhead(s) 6.7 27.3 90.1 370.3
Table 3: The achieved accuracy and overhead when
applying different tree search depths in the agentic
search of Avas evalauted on the subset of LVBench.
representations but also drastically reduces the time needed
to build the graph.
7.4.2 Different Tree Search Depths. We also evaluate
the achieved accuracy and overhead applying different set-
tings in the retrieval phase, i.e.,tree search depth. The effect
of tree depth lies in a crucial trade-off: while shallower depths
may struggle to retrieve comprehensive information, increas-
ing the depth allows access to richer information from deeper
nodes. However, this comes with a significant increase in tree
search overhead, and the information from deeper levels can
introduce more noise, potentially negatively impacting the fi-
nal generation quality. Table 3 presents the results comparing
different tree search depths on performance and tree search
overhead. As shown, performance generally increases with
increasing tree depth up to a certain point. Specifically, for all
three Avas configurations evaluated, the highest accuracy is
achieved at a tree search depth of 3. Accuracy decreases when
the depth is further increased to 4, suggesting that excessive
depth leads to the retrieval of detrimental noise or irrelevant
information, outweighing the benefit of additional context.
Conversely, the tree search overhead increases sharply with
depth. Expanding the search from depth 1 (6.7s) to depth 2
(27.3s) incurs a moderate increase. Comparing the accuracy
improvements and the overhead increase, a tree search depth
of 3 offers the optimal balance.
7.4.3 Different Consistency Evaluation Settings. For
consistency-enhanced generation, Avas incorporates two
key parameters: 𝛼, which governs the balance between the
contributions of thought consistency and answer consis-
tency, and the number of generations for self-consistency
evaluation. Fig. 11a illustrates the impact of varying 𝛼values
on the accuracy achieved by Avas . Notably, the optimal per-
formance is observed when 𝛼is set to 0.3, highlighting the
importance of jointly considering both intermediate thought
consistency and final answer consistency to ensure robust
results. As depicted in Fig. 11b, the accuracy of Avas gradu-
ally improves as the number of self-consistency iterations
increases. However, this improvement comes at the expense
of significantly higher computational overhead. For example,
increasing the self-consistency iterations from 8 to 16 yields

Empowering Agentic Video Analytics Systems with Video Language Models
0.0 0.2 0.4 0.6 0.8 1.0
l36373839404142Accuracy (%)
a Balance between thoughts
and answer consistency
2 4 6 8 10 12 14 16
#Self-Consistency Times343638404244Accuracy (%)
Accuracy
Overhead
050100150200250
Overhead (s)
b Trade-offs using different
self-consistency times.
Figure 11: The performance of Avas under varying con-
sistency evaluation settings on the subset of LVBench.
only a 0.9% accuracy gain, while nearly doubling the com-
putational cost. This demonstrates a clear trade-off between
marginal accuracy improvements and resource efficiency.
Balancing this trade-off, we adopt 8 self-consistency itera-
tions in the implementation of Avas , ensuring a practical
balance between performance and computational overhead.
8 LIMITATIONS AND FUTURE WORK
There are also limitations in the current design of Avas ,
which we explore for future work. Specifically: 1) The ex-
isting agentic retrieval and generation mechanism relies on
a fixed tree-search strategy based on the Monte Carlo ap-
proach. While effective, this method is computationally ex-
pensive. The trajectories collected during the search process
could be leveraged as training data to develop a model ca-
pable of dynamically selecting optimal search actions and
depths based on the query and context. 2) Although the inte-
grated VLM demonstrates robust general video understand-
ing and reasoning capabilities, it may encounter challenges in
certain specialized visual tasks, such as precise object count-
ing. Incorporating lightweight, task-specific vision models
as tools within the system could improve accuracy for such
queries. Our future work will focus on enabling the VLM,
functioning as an autonomous agent, to intelligently invoke
these specialized tools, thereby addressing its limitations in
handling specific tasks.
9 CONCLUSION
This paper presents Avas , an advanced L4 video analytics
system powered by VLMs. Avas enables comprehensive un-
derstanding and open-ended query analysis of large-scale,
long-duration video data, overcoming the constraints of ex-
isting video analytics systems that are predominantly tai-
lored to specific, pre-defined tasks. The system introduces
novel designs, including near-real-time Event Knowledge
Graph (EKG) index construction and an agentic retrieval
and generation mechanism, facilitating efficient organization
and analysis of extended video content to address complex
queries. We demonstrate Avas ’s superior performance onpublic video understanding benchmarks, as well as on our
newly proposed benchmark, Avas -100, specifically designed
to evaluate video analytics tasks.
REFERENCES
[1]Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Am-
mar Ahmad Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari,
Jianmin Bao, Harkirat Behl, et al .2024. Phi-3 technical report: A
highly capable language model locally on your phone. arXiv preprint
arXiv:2404.14219 (2024).
[2]Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen
Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan
Javaheripi, Piero Kauffmann, et al .2024. Phi-4 technical report. arXiv
preprint arXiv:2412.08905 (2024).
[3]Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany
Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai,
Vishrav Chaudhary, Congcong Chen, et al .2025. Phi-4-mini techni-
cal report: Compact yet powerful multimodal language models via
mixture-of-loras. arXiv preprint arXiv:2503.01743 (2025).
[4]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge
Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt,
Sam Altman, Shyamal Anadkat, et al .2023. Gpt-4 technical report.
arXiv preprint arXiv:2303.08774 (2023).
[5]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo
Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al .2025. Qwen2.
5-vl technical report. arXiv preprint arXiv:2502.13923 (2025).
[6]Romil Bhardwaj, Zhengxu Xia, Ganesh Ananthanarayanan, Junchen
Jiang, Yuanchao Shu, Nikolaos Karianakis, Kevin Hsieh, Paramvir Bahl,
and Ion Stoica. 2022. Ekya: Continuous learning of video analytics
models on edge compute servers. In USENIX Symposium on Networked
Systems Design and Implementation (NSDI) .
[7]Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei
Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al .2024. Ex-
panding performance boundaries of open-source multimodal models
with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271
(2024).
[8]LMDeploy Contributors. 2023. LMDeploy: A Toolkit for Compressing,
Deploying, and Serving LLM. https://github.com/InternLM/lmdeploy.
[9]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.
2019. Bert: Pre-training of deep bidirectional transformers for language
understanding. In Proceedings of conference of the North American
chapter of the association for computational linguistics(NAACL) .
[10] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao,
Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Os-
azuwa Ness, and Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization. arXiv preprint
arXiv:2404.16130 (2024).
[11] Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang. 2025. Mini-
RAG: Towards Extremely Simple Retrieval-Augmented Generation.
arXiv preprint arXiv:2501.06713 (2025).
[12] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui
Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang,
et al.2024. Video-mme: The first-ever comprehensive evaluation
benchmark of multi-modal llms in video analysis. arXiv preprint
arXiv:2405.21075 (2024).
[13] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis,
Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao
Liu, Xingyu Liu, et al .2022. Ego4d: Around the world in 3,000 hours
of egocentric video. In Conference on Computer Vision and Pattern
Recognition (CVPR) .

Yan and Jiang et al.
[14] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024.
LightRAG: Simple and Fast Retrieval-Augmented Generation. arXiv
preprint arXiv:2410.05779 (2024).
[15] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep
residual learning for image recognition. In Conference on Computer
Vision and Pattern Recognition (CVPR) .
[16] Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2020.
Deberta: Decoding-enhanced bert with disentangled attention. arXiv
preprint arXiv:2006.03654 (2020).
[17] Shiqi Jiang, Zhiqi Lin, Yuanchun Li, Yuanchao Shu, and Yunxin Liu.
2021. Flexible High-Resolution Object Detection on Edge Devices with
Tunable Latency. In Proceedings of the 27th Annual International Con-
ference on Mobile Computing and Networking (New Orleans, Louisiana)
(MobiCom ’21) . Association for Computing Machinery, New York, NY,
USA, 559–572. https://doi.org/10.1145/3447993.3483274
[18] Shiqi Jiang, Zhiqi Lin, Yuanchun Li, Yuanchao Shu, and Yunxin Liu.
2021. Flexible high-resolution object detection on edge devices with
tunable latency. In ACM International Conference on Mobile Computing
and Networking (Mobicom) .
[19] Mehrdad Khani, Ganesh Ananthanarayanan, Kevin Hsieh, Junchen
Jiang, Ravi Netravali, Yuanchao Shu, Mohammad Alizadeh, and Victor
Bahl. 2023.{RECL}: Responsive{Resource-Efficient}continuous
learning for video analytics. In USENIX Symposium on Networked
Systems Design and Implementation (NSDI) .
[20] Andreas Koukounas, Georgios Mastrapas, Michael Günther, Bo Wang,
Scott Martens, Isabelle Mohr, Saba Sturua, Mohammad Kalim Akram,
Joan Fontanals Martínez, Saahil Ognawala, et al .2024. Jina clip: Your
clip model is also your text retriever. arXiv preprint arXiv:2405.20204
(2024).
[21] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit
Bansal, and Jingjing Liu. 2021. Less is More: ClipBERT for Video-and-
Language Learning via Sparse Sampling. arXiv:2102.06183 [cs.CV]
https://arxiv.org/abs/2102.06183
[22] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou,
Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al .
2024. NVILA: Efficient frontier visual language models. arXiv preprint
arXiv:2412.04468 (2024).
[23] Yan Lu, Shiqi Jiang, Ting Cao, and Yuanchao Shu. 2022. Turbo: Op-
portunistic enhancement for edge video analytics. In Proceedings of
ACM Conference on Embedded Networked Sensor Systems(Sensys) .
[24] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa
Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. 2024. Video-RAG:
Visually-aligned Retrieval-Augmented Long Video Comprehension.
arXiv preprint arXiv:2411.13093 (2024).
[25] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid
Rezatofighi, and Jianfei Cai. 2024. Drvideo: Document retrieval based
long video understanding. arXiv preprint arXiv:2406.12846 (2024).
[26] Ya Nan, Shiqi Jiang, and Mo Li. 2023. Large-scale Video Analytics
with Cloud–Edge Collaborative Continuous Learning. ACM Trans.
Sen. Netw. 20, 1, Article 14 (Oct. 2023), 23 pages. https://doi.org/10.
1145/3624478
[27] Arthi Padmanabhan, Neil Agarwal, Anand Iyer, Ganesh Anantha-
narayanan, Yuanchao Shu, Nikolaos Karianakis, Guoqing Harry Xu,
and Ravi Netravali. 2023. Gemel: Model merging for {Memory-
Efficient},{Real-Time}video analytics at the edge. In USENIX Sympo-
sium on Networked Systems Design and Implementation (NSDI) .
[28] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding,
Dahua Lin, and Jiaqi Wang. 2024. Streaming Long Video Under-
standing with Large Language Models. arXiv:2405.16009 [cs.CV]
https://arxiv.org/abs/2405.16009
[29] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel
Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, PamelaMishkin, Jack Clark, et al .2021. Learning transferable visual mod-
els from natural language supervision. In International Conference on
Machine Learning (ICML) .
[30] Microsoft Research. 2024. LazyGraphRAG: Setting a new standard for
quality and cost . https://www.microsoft.com/en-us/research/blog/
lazygraphrag-setting-a-new-standard-for-quality-and-cost/
[31] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun
Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadara-
jan, Florian Bordes, et al .2024. Longvu: Spatiotemporal adaptive
compression for long video-language understanding. arXiv preprint
arXiv:2410.17434 (2024).
[32] Yunhang Shen, Chaoyou Fu, Shaoqi Dong, Xiong Wang, Peixian
Chen, Mengdan Zhang, Haoyu Cao, Ke Li, Xiawu Zheng, Yan Zhang,
et al.2025. Long-VITA: Scaling Large Multi-modal Models to 1 Mil-
lion Tokens with Leading Short-Context Accuray. arXiv preprint
arXiv:2502.05177 (2025).
[33] Vibhaalakshmi Sivaraman, Pantea Karimi, Vedantha Venkatapathy,
Mehrdad Khani, Sadjad Fouladi, Mohammad Alizadeh, Frédo Durand,
and Vivienne Sze. 2024. Gemino: Practical and robust neural com-
pression for video conferencing. In USENIX Symposium on Networked
Systems Design and Implementation (NSDI) .
[34] Mingxing Tan, Ruoming Pang, and Quoc V Le. 2020. Efficientdet:
Scalable and efficient object detection. In Conference on Computer
Vision and Pattern Recognition (CVPR) .
[35] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac,
Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth,
Katie Millican, et al .2023. Gemini: a family of highly capable multi-
modal models. arXiv preprint arXiv:2312.11805 (2023).
[36] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai,
Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo
Wang, et al .2024. Gemini 1.5: Unlocking multimodal understanding
across millions of tokens of context. arXiv preprint arXiv:2403.05530
(2024).
[37] Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and
Manohar Paluri. 2015. Learning spatiotemporal features with 3d
convolutional networks. In International Conference on Computer Vi-
sion(ICCV) .
[38] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai,
Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al .2024. Qwen2-
vl: Enhancing vision-language model’s perception of the world at any
resolution. arXiv preprint arXiv:2409.12191 (2024).
[39] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang,
Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, et al .2024.
Lvbench: An extreme long video understanding benchmark. arXiv
preprint arXiv:2406.08035 (2024).
[40] Xiao Wang, Qingyi Si, Jianlong Wu, Shiyu Zhu, Li Cao, and Liqiang
Nie. 2025. AdaRETAKE: Adaptive Redundancy Reduction to Per-
ceive Longer for Video-language Understanding. arXiv preprint
arXiv:2503.12559 (2025).
[41] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sha-
ran Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-
Consistency Improves Chain of Thought Reasoning in Language Mod-
els. arXiv:2203.11171 [cs.CL] https://arxiv.org/abs/2203.11171
[42] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy.
2024. Videoagent: Long-form video understanding with large language
model as agent. In European Conference on Computer Vision(ECCV) .
[43] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng
Cheng, Gedas Bertasius, and Mohit Bansal. 2024. Videotree: Adaptive
tree-based video representation for llm reasoning on long videos. arXiv
preprint arXiv:2405.19209 (2024).
[44] Zeyuan Yang, Delin Chen, Xueyang Yu, Maohao Shen, and Chuang
Gan. 2024. VCA: Video Curious Agent for Long Video Understanding.

Empowering Agentic Video Analytics Systems with Video Language Models
arXiv preprint arXiv:2412.10471 (2024).
[45] Chen-Lin Zhang, Jianxin Wu, and Yin Li. 2022. Actionformer: Localiz-
ing moments of actions with transformers. In European Conference on
Computer Vision(ECCV) .
[46] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and
Yoav Artzi. 2020. BERTScore: Evaluating Text Generation with BERT.
InInternational Conference on Learning Representations (ICLR) .
[47] Xu Zhang, Yiyang Ou, Siddhartha Sen, and Junchen Jiang. 2021.
{SENSEI}: Aligning video streaming quality with dynamic user sensi-
tivity. In USENIX Symposium on Networked Systems Design and Imple-
mentation (NSDI) .
[48] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu,
and Chunyuan Li. 2024. Video instruction tuning with synthetic data.arXiv preprint arXiv:2410.02713 (2024).
[49] Yiwen Zhang, Xumiao Zhang, Ganesh Ananthanarayanan, Anand Iyer,
Yuanchao Shu, Victor Bahl, Z Morley Mao, and Mosharaf Chowdhury.
2024. Vulcan: Automatic Query Planning for Live {ML}Analytics. In
USENIX Symposium on Networked Systems Design and Implementation
(NSDI) .
[50] Siyun Zhao, Yuqing Yang, Zilong Wang, Zhiyuan He, Luna K. Qiu, and
Lili Qiu. 2024. Retrieval Augmented Generation (RAG) and Beyond:
A Comprehensive Survey on How to Make your LLMs use External
Data More Wisely. arXiv:2409.14924 [cs.CL] https://arxiv.org/abs/
2409.14924