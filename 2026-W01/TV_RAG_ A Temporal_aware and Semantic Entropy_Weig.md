# TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding

**Authors**: Zongsheng Cao, Yangfan He, Anran Liu, Feng Chen, Zepeng Wang, Jun Xie

**Published**: 2025-12-29 14:10:22

**PDF URL**: [https://arxiv.org/pdf/2512.23483v1](https://arxiv.org/pdf/2512.23483v1)

## Abstract
Large Video Language Models (LVLMs) have rapidly emerged as the focus of multimedia AI research. Nonetheless, when confronted with lengthy videos, these models struggle: their temporal windows are narrow, and they fail to notice fine-grained semantic shifts that unfold over extended durations. Moreover, mainstream text-based retrieval pipelines, which rely chiefly on surface-level lexical overlap, ignore the rich temporal interdependence among visual, audio, and subtitle channels. To mitigate these limitations, we propose TV-RAG, a training-free architecture that couples temporal alignment with entropy-guided semantics to improve long-video reasoning. The framework contributes two main mechanisms: \emph{(i)} a time-decay retrieval module that injects explicit temporal offsets into the similarity computation, thereby ranking text queries according to their true multimedia context; and \emph{(ii)} an entropy-weighted key-frame sampler that selects evenly spaced, information-dense frames, reducing redundancy while preserving representativeness. By weaving these temporal and semantic signals together, TV-RAG realises a dual-level reasoning routine that can be grafted onto any LVLM without re-training or fine-tuning. The resulting system offers a lightweight, budget-friendly upgrade path and consistently surpasses most leading baselines across established long-video benchmarks such as Video-MME, MLVU, and LongVideoBench, confirming the effectiveness of our model. The code can be found at https://github.com/AI-Researcher-Team/TV-RAG.

## Full Text


<!-- PDF content starts -->

TV-RAG: A Temporal-aware and Semantic Entropy-Weighted
Framework for Long Video Retrieval and Understanding
Zongsheng Cao∗
agiczsr@gmail.com
ResearcherYangfan He∗
he00577@umn.edu
UMNAnran Liu∗†
anniegogo1008@gmail.com
Researcher
Feng Chen
chenfeng@lenovo.com
PCIEZepeng Wang
wangzpb@lenovo.com
PCIEJun Xie†
xiejun@lenovo.com
PCIE
Abstract
Large Video Language Models (LVLMs) have rapidly emerged as
the focus of multimedia AI research. Nonetheless, when confronted
with lengthy videos, these models struggle: their temporal win-
dows are narrow, and they fail to notice fine-grained semantic
shifts that unfold over extended durations. Moreover, mainstream
text-based retrieval pipelines, which rely chiefly on surface-level
lexical overlap, ignore the rich temporal interdependence among
visual, audio, and subtitle channels. To mitigate these limitations,
we propose TV-RAG, a training-free architecture that couples tem-
poral alignment with entropy-guided semantics to improve long-
video reasoning. The framework contributes two main mechanisms:
(i)a time-decay retrieval module that injects explicit temporal off-
sets into the similarity computation, thereby ranking text queries
according to their true multimedia context; and (ii)an entropy-
weighted key-frame sampler that selects evenly spaced, information-
dense frames, reducing redundancy while preserving representa-
tiveness. By weaving these temporal and semantic signals together,
TV-RAG realises a dual-level reasoning routine that can be grafted
onto any LVLM without re-training or fine-tuning. The resulting
system offers a lightweight, budget-friendly upgrade path and con-
sistently surpasses most leading baselines across established long-
video benchmarks such as Video-MME, MLVU, and LongVideoBench,
confirming the effectiveness of our model. The code can be found
at https://github.com/AI-Researcher-Team/TV-RAG.
CCS Concepts
•Computing methodologies →Natural language processing .
Keywords
Video Understanding, Temporal-aware, Semantic Entropy-Weighted
∗Equal contribution
†Corresponding author
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full cita-
tion on the first page. Copyrights for components of this work owned by others than
the author(s) must be honored. Abstracting with credit is permitted. To copy other-
wise, or republish, to post on servers or to redistribute to lists, requires prior specific
permission and/or a fee. Request permissions from permissions@acm.org.
MM’25, October 27–31, 2025, Dublin, Ireland
© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-2035-2/2025/10
https://doi.org/10.1145/3746027.3755873ACM Reference Format:
Zongsheng Cao, Yangfan He, Anran Liu, Feng Chen, Zepeng Wang, and Jun
Xie. 2025. TV-RAG: A Temporal-aware and Semantic Entropy-Weighted
Framework for Long Video Retrieval and Understanding. In Proceedings of
the 33th ACM International Conference on Multimedia (MM ’25), October 27–
31,2025,Dublin,Ireland. ACM, New York, NY, USA, 9pages. https://doi.org/
10.1145/3746027.3755873
1 Introduction
Recent breakthroughs in large-scale language modelling have catal-
ysed rapid progress in multimodal research, ultimately leading to
a new class of Large Video–Language Models (LVLMs) [ 11,16,37].
Despite their impressive accuracy on short, clip-length inputs, cur-
rent LVLMs still face significant obstacles when tasked with ana-
lyzing and reasoning over very long videos.
Recent attempts to improve long-horizon reasoning in Large
Video Language Models, exemplified by LongVA [ 39] and LongLLAVA
[35], have largely focused on widening the context window. LongVA,
for instance, simply scales up the token capacity to ingest more
frames. This brute-force expansion, however, proves brittle when
faced with out-of-distribution footage: the Video-MME benchmark
shows its accuracy drops sharply as additional frames are supplied.
A parallel research thread employs Retrieval-Augmented Genera-
tion (RAG) to supplement video queries with externally retrieved
documents [ 15,18,40]. Yet, treating the video stream purely as
text ignores vital visual alignment cues, and the reliance on high-
capacity LVLM backbones translates into steep computational costs
and poor adaptability. Consequently, existing RAG pipelines strug-
gle to capture the intertwined temporal and semantic signals re-
quired for truly robust long-video understanding.
Based on the considerations above, we focus on the following
question: Can we develop a new model that can tackle the video
RAG task from both temporal and semantic perspectives in a unified
training-free framework?
Motivated by this limitation, several studies [ 15,18,40] advocate
replacing the bulky chain of visual tokens with proxy captions dis-
tilled directly from the video stream via off-the-shelf optical char-
acter recognitions (OCR), automatic speech recognition (ASR), and
object-detection models. These auxiliary texts are tightly coupled
to the visual scene and inject complementary cues unavailable in
raw pixels. For instance, Long-context LVLMs extend visual tokens
so the model sees every frame, yielding strong temporal awareness
but at the cost of heavy retraining and limited semantic depth. GPT-
based agent pipelines first convert videos into textual reports, thenarXiv:2512.23483v1  [cs.CV]  29 Dec 2025

MM’25, October 27–31, 2025, Dublin, Ireland Zongsheng Cao et al.
Figure 1: Advantages of our TV-RAG. TV-RAG provides a
temporal-aware, semantic-aware, and training-free pipeline
that is easily compatible with any LVLM.
query a proprietary LLM such as GPT-4o; while semantically pow-
erful, they lose fine-grained temporal cues and depend on closed-
source services. Yet, the strategy leaves two key issues unresolved:
C1Cross-modal dependencies among images, audio, and subti-
tles remain implicit, hampering performance on tasks that
demand joint reasoning over long temporal spans;
C2The fixed context window still cannot follow subtle seman-
tic drifts that emerge gradually throughout lengthy footage.
To this end, we introduce TV-RAG, a novel framework designed
to enhance the comprehension of long videos in LVLMs without
requiring additional training. Specifically, to address (C1), TV-RAG
employs a semantic entropy-based weighting strategy for key frame
selection to evenly distribute selected frames across time, reducing
redundancy, enhancing representativeness, and prioritizing the most
informative frames. To tackle (C2), it incorporates a temporal window-
based BM25 model that integrates time-aligned auxiliary data, in-
cluding OCR, ASR, and object detection (DET) outcomes. This ap-
proach enhances the relevance of text queries by aligning them
with the temporal context of multimedia content, ensuring that re-
trieved information is contextually appropriate.
By combining explicit temporal order with entropy-based se-
mantic salience, TV-RAG enables a two-stage reasoning process
that removes noise and filters out irrelevant content from long
videos. The module is lightweight and can be added to any exist-
ing LVLM without modifying its weights. This plug-and-play de-
sign allows TV-RAG to outperform popular models on benchmark
datasets such as Video-MME, MLVU, and LongVideoBench. We
further evaluate TV-RAG on six widely used open-source LVLM
backbones, confirming its effectiveness.
In summary, our contributions are three-fold:
•To mitigate frame redundancy and preserve cross-modal se-
mantics in long videos, we propose TV-RAG, including a
semantic-entropy key-frame selector that computes multi-
modal information gain and enforces uniform temporal cov-
erage, yielding compact yet representative frame sets for
downstream LVLM reasoning.•To handle semantic drift and maintain temporally coherent
retrieval, we introduce a temporal-window BM25 retriever
that ties lexical relevance to timestamp alignment across
auxiliary text streams, supplying the LVLM with context
that stays synchronized with the evolving video content, all
without modifying model weights.
•Extensive experiments on several well-established bench-
marks show that TV-RAG achieves state-of-the-art perfor-
mance and outperforms other models. It can also be used as
a plug-in for other models.
2 Related Work
2.1 Large Language Models for Video
The surge of large language models has sparked parallel efforts to
craft versatile video–language systems. Early work, Video-ChatGPT
[20], processes frames separately and later fuses their representa-
tions via spatial–temporal pooling. VideoChat [ 10] augments ap-
pearance embeddings with on-the-fly textual captions to form a
richer clip encoding. To minimise the gap between image and video
pathways, Video-LLaVA [ 12] introduces a shared projector that
aligns both outputs within a common language latent space. Its
successor, LLaVA-NeXT-Video [ 40], adapts the LLaVA-NeXT back-
bone [ 15] through targeted video fine-tuning. Despite these ad-
vances, retrieval-augmented schemes such as Video-RAG [ 18] still
fall short in tracking the subtle temporal dependencies and layered
semantics present in lengthy, information-dense footage.
2.2 Large Language Models for Long-context
Video
A prominent strand of recent work attempts to enlarge the effec-
tive context window so that models can reason over intricate video
narratives. LongVA [ 39] and Long-LLaVA [ 35] pursue this goal
by first pre-training language backbones on massive text corpora,
banking on the resulting long-sequence handling skills to trans-
fer to video tasks. In contrast, INTP [ 27] rearranges the incoming
video tokens without any additional training, stretches the win-
dow of an LVLM to ingest far more visual information. Yet, these
solutions confront a familiar trade-off: every extra batch of sam-
pled frames inflates computation while delivering only marginal
gains. Because videos contain significant redundancy, and model
capacity is finite. Under these circumstances, most models cannot
achieve satisfactory performance.
2.3 Video Understanding by GPT-based Agent
A complementary line of work investigates using an LLM as a con-
troller that calls external vision or audio modules to parse long-
form video data, especially for question–answering tasks [ 5,21,29,
32,36]. MM-VID [ 13] advances this idea by explicitly pairing each
frame with its textual caption, whereas VLog [ 14] first extracts
audio and appearance cues via multimodal pre-training and then
distills them into compact textual surrogates. Agent-style frame-
works such as VideoAgent [ 2], DrVideo [ 19], and OmAgent [ 38]
go a step further, issuing adaptive queries to retrieve relevant seg-
ments before reasoning over them. Despite their ingenuity, these
pipelines still suffer from two practical drawbacks: (i) iterative tool

TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding MM’25, October 27–31, 2025, Dublin, Ireland
Figure 2: The illustration of our framework TV-RAG. The pipeline begins with query decoupling, where the LVLM rewrites
the user question into an explicit evidence-retrieval request, cleanly separating search from reasoning. TV-RAG then executes
three tightly coupled steps: (i) a semantic-entropy selector distills high-information frames across visual, OCR, ASR and de-
tection streams; (ii) a temporal-decay BM25 retriever ranks auxiliary texts by both lexical relevance and timestamp proximity,
ensuring chronological coherence; and (iii) a bi-level reasoning routine drafts and self-verifies the answer using the retrieved
evidence. By closing the loop between asking, retrieving, and reasoning without altering LVLM weights, TV-RAG simultane-
ously sharpens retrieval precision and elevates answer faithfulness.
invocation inflates inference latency, and (ii) many rely on closed-
source models, limiting both efficiency and the ease with which the
community can replicate results using purely open-source stacks.
Moreover, there are also some advanced methods, such as [ 18];
however, they have flaws in capturing the joint associations of tem-
poral and semantic information. In this way, it is still an open issue
to be addressed.
3 Methodology
We introduce TV-RAG, a novel training-free process designed for
LVLMs that can be seamlessly integrated into any existing LVLM.
As shown in Fig. 2, the process consists of three main phases: (i) Se-
mantic entropy-based information extraction: After obtain-
ing the query, information is extracted based on semantic entropy
from different sources. (ii) Temporal decay-enhanced retrieval
model: In order to capture the important temporal information
in the video, the time window mechanism request is introduced
for obtaining the relevant information. (iii) Context-enhanced
reasoning-based response generation: In this final stage, the
auxiliary text retrieved based on the context-enhanced reasoning
mechanism is integrated with the user’s query and fed into the
LVLM to generate the final output.Problem Setup. LetVbe an input video. We then use a frame–
selection unit to extract 𝑁representative images FEach frame is
then mapped into a visual embedding via a frozen image encoder,
e.g., CLIP-L [ 23], yielding F𝑣from F. Finally, the visual tokens F𝑣
and a user query Qare supplied to a large video–language model
to generate the answer O:
O=LVLM (F𝑣,Q). (1)
In this way, we complete the RAG process for videos.
Two-stage Processes . In this paper, motivated by previous efforts
[12,18], upon receiving a user’s query regarding a video, the LVLM
follows a two-phase process. First, the LVLM decouples the query
to generate structured retrieval requests, which serve as auxiliary
inputs for later stages. In this phase, the LVLM focuses solely on
processing textual data without access to the video frames. These
requests are then parsed, and if any category does not require data,
it will be marked as NULL (indicating no need for that specific
request). This results in the structured retrieval requests, R={R𝑖},
which means the request categories, such as detection information.
Then, they are passed on to the next phase. The process can thus
be decoupled into two main stages, as formalized in the following
equations:
R=LVLM(P,Q),R={R𝑖}, (2)

MM’25, October 27–31, 2025, Dublin, Ireland Zongsheng Cao et al.
where Pis the prompt. In the second phase, these requests, R, along
with the video frames, F𝒗, are used to generate the final output:
O=LVLM(F𝒗,Q,R). (3)
In this phase, the LVLM processes the video frames, utilizes the
textual query, and incorporates the generated retrieval requests to
provide a more informed and contextually relevant answer.
3.1 Semantic Entropy-based Extraction
Auxiliary‐Text Extraction and Retrieval. Following the blue-
print laid out by recent retrieval-augmented systems [ 15,18,40],
we first transcribe the raw video into several complementary text
channels and then fetch the portions most useful for the down-
stream LVLM. Formally, we create a set R={ R𝑎𝑠𝑟,R𝑜𝑐𝑟,R𝑑𝑒𝑡},
where ( i)𝑟asRis automatic speech recognition to harvest dialogue
or soundtrack content; ( ii)𝑟detis salient entities present on screen;
and (iii)𝑟ocRis optical character recognition to capture any scene
text.
Longer clips naturally produce voluminous and often redundant
token streams, while the context budget of current open-source
LVLMs remains tight. To avoid overflow, we retrieve only those
snippets that align semantically with the user query. Before re-
trieval, three modality-specific repositories are built in parallel from
the video itself: an ASR base DasR, an OCR base DocR, and an
object-detection base Ddet. Subsequent look-ups are executed against
these lightweight databases, ensuring that irrelevant tokens do not
enter the LVLM’s limited context window.
Building Retrieval Repositories. Open source LVLMs tend to
misread scene text and spoken words, falling short of their propri-
etary counterparts. To curb such hallucinations and exploit frame
content more effectively, we offload text extraction to specialist
models. Concretely, EasyOCR [ 7] is run on each key frame to har-
vest on-screen captions, giving a pool of strings TocR; meanwhile,
the soundtrack is transcribed by Whisper [ 24], yielding an ASR
transcript TasRas advocated in prior work [ 15,18,40]. Both text
streams are embedded with the ContRieveR encoder [ 6] to ob-
tain dense vectors, which are written to two separate FAISS in-
dices [ 8]:DocRfor scene text and DasRfor speech. This design en-
ables low-latency, similarity-based retrieval of the most relevant
snippets during query time.
Key–Frame Selection. Although modern LVLMs excel at recog-
nising objects, they remain error-prone when asked to count in-
stances, pinpoint locations, or reason about complex interactions,
frequently hallucinating details when contextual cues are sparse.
To curb this issue, motivated by [ 12,18], we rank every sampled
frame F={𝐹𝑡}by the semantic affinity between the detector re-
quest𝑅𝑑𝑒𝑡and the frame content, modulated by a temporal impor-
tance weight:
𝐹𝑘𝑒𝑦=n
𝐹𝑡𝛼𝑡·CLIP 𝑅𝑑𝑒𝑡,𝐹𝑡≥𝜏o
,
where𝜏is a similarity threshold. The weight 𝛼𝑡captures how much
new information the segment contributes and is computed from
the normalised Shannon entropy of its visual features:
𝛼𝑡=𝐻(𝐹𝑡)Í
𝑗𝐻(𝐹𝑗).Object detection is then run only on this entropy-aware subset
FKey, ensuring that the LVLM processes the most informative and
contextually relevant frames.
The weight𝑤𝑡assigned to each frame is determined by the infor-
mation entropy 𝐻(𝐹𝑡), a measure of the uncertainty or variability
within the frame’s content. Then the entropy 𝐻(𝐹𝑡)is computed
as:
𝐻(𝐹𝑡)=−𝑝𝑡log𝑝𝑡, 𝑝𝑡=CLIP similarity (𝑅det,𝐹𝑡)Í
𝑗CLIP similarity (𝑅det,𝐹𝑗),(4)
where𝑝𝑡represents the normalized similarity between the object
retrieval query and each frame 𝐹𝑡, and𝐻(𝐹𝑡)captures how much
variance exists in the content of a given time segment. The se-
lection process begins by computing the CLIP similarity [ 23] be-
tween an object retrieval query R𝑑𝑒𝑡and the sampled video frames
F. High entropy values indicate significant changes in the visual
or auditory content, suggesting that this segment contains crucial,
non-redundant information.
RemaRK. In this way, it ensures that time segments with higher
variability in content are more likely to be selected, while segments
with low variability (i.e., those that are redundant) are deprioritized.
The goal is to maximize the representativeness of the selected frames
and minimize redundant or irrelevant content, which is particularly
important in the context of long-duration videos. By incorporating
this entropy-based weighting mechanism, we ensure that critical mo-
ments in the video, which may involve sudden shifts in object inter-
action or scene changes, are not overlooked.
3.2 Temporal Window-based Text Retrieval
Query–Aware Retrieval. Our pipeline introduces a dedicated re-
trieval routine that tightly couples the user prompt with the auxil-
iary captions mined from the video. Inspired by prior multimodal
RAG work [ 12,18], we begin by converting the query into a dense
representation e𝑟𝑒𝑞using the Contriever encoder.
To prevent the common temporal drift problem, where a lexi-
cally relevant caption is drawn from an incorrect moment in the
clip, we refine the top- 𝑘text hits with a temporal re-ranking stage.
Each candidate’s score is updated by a weighted sum of its lexical
similarity to e𝑟𝑒𝑞and its time-stamp distance from the query seg-
ment, ensuring the final selection is both semantically pertinent
and temporally on point.
Temporal Re-scoring of Retrieved Snippets. Assume that a
FAISS search over the OCR/ASR indices yields the relevant snip-
petsC. Let𝜏𝑞∈Rmark the moment in the video that anchors the
user’s query, and denote by 𝜏𝑖the original time-code of snippet 𝑐𝑖.
We refine each snippet’s lexical BM25 score by a time-aware decay:
˜𝑠𝑖=BM25 (ereq,𝑐𝑖)𝑒−Í2
𝑘=0𝜆𝑘|𝜏𝑘𝑞−𝜏𝑖|
3𝐾Õ
𝑗=1BM25 (ereq,𝑐𝑗)𝑒−Í2
𝑘=0𝜆𝑘|𝜏𝑘𝑞−𝜏𝑗|, (5)
where ereqis the Contriever embedding of the query and 𝜆>0
(set to 1in our study) controls the strength of the temporal penalty.
Here the three anchor points 𝜏𝑘
𝑞serve complementary roles: 𝜏0
𝑞is
the time-code of the lastvideo frame, 𝜏1
𝑞marks the firstframe, and
𝜏2
𝑞corresponds to the frame whose local context is semantically
most similar to the query. The rationale is that salient evidence can

TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding MM’25, October 27–31, 2025, Dublin, Ireland
appear near a video’s conclusion (e.g., a reveal or punch-line), at
its outset (introductory setup), or at an unpredictable location that
is best located via semantic affinity. By injecting all three anchors
into the exponential decay, the rescoring scheme remains agnostic
to where the key information actually lies, yet still rewards snip-
pets clustered around whichever temporal region proves relevant
for the user’s request.
The exponential term softly favours snippets that occur near 𝜏𝑘
𝑞
while still allowing distant, but lexically pertinent candidates to
survive with a lower confidence. Finally, we keep the top- 𝐾snip-
pets with the highest ˜𝑠𝑖:
C=TopK C,˜𝑠𝑖. (6)
Incorporating timestamp proximity into the similarity score reme-
dies two chronic weaknesses of standard retrieval pipelines. (i) It
sharply reduces the likelihood of selecting passages that look se-
mantically relevant yet come from the wrong moment in the footage,
a pitfall common to long videos with recurrent motifs. (ii) It main-
tains chronological consistency for the LVLM, so downstream rea-
soning and text generation remain anchored to the correct slice
of the narrative. Unlike hard sliding windows or ad-hoc alignment
rules, the proposed weighting scheme is continuous, differentiable,
and plug-compatible with existing multimodal retrieval frameworks.
Object Information Retrieval. After retrieving the relevant in-
formation A, the next step involves extracting object-related infor-
mation stored in the auxiliary database. Motivated by [ 15,18,40],
object detection models typically produce raw outputs that include
spatial attributes in the form of bounding boxes and other informa-
tion. These components collectively form a structured scene graph,
expressed as Ast, which can be regarded as a detection database and
enhances the ability of LVLMs to interpret temporal and spatial in-
formation. By incorporating this structured graph representation,
we enable LVLMs to reason over the spatial layout and temporal
dynamics of objects within the video.
3.3 Context-augmented Reasoning with Query
Reformulation
While retrieved auxiliary signals (e.g., OCR, ASR) can provide es-
sential external context, relying exclusively on these sources in-
troduces two key limitations: (i) they are often noisy, fragmented,
or partially misaligned with the user’s underlying intent; and (ii)
compressing them to fit the limited context window of LVLMs may
introduce semantic loss and degrade the coherence of downstream
reasoning.
To overcome these issues, we propose a context-augmented rea-
soning strategy that enhances retrieved evidence through a dual-
channel mechanism. This approach not only supplements the aux-
iliary inputs but also enhances the model’s understanding of the
user query via semantic rephrasing.
Level-1: Evidence-based Contextualization. We first aggregate
auxiliary textual signals from 𝐴𝑠𝑡. This aggregated context reflects
the explicit knowledge retrieved from the video content and acts
as grounded, observable evidence for reasoning.Level-2: Query-enhanced Latent Reasoning. Recognizing that
retrieved evidence may still omit latent or abstract concepts im-
plied by the user’s query, we introduce a generation-based aug-
mentation step. Instead of compressing auxiliary data, we instruct
the LVLM to perform two complementary operations: (1) generate
general background context relevant to the query to enrich model-
level knowledge grounding; (2) reformulate the user query into a
set of semantically similar paraphrases to enhance retrieval sensi-
tivity and decoding diversity.
Specifically, we define the prompt 𝐼𝑔𝑒𝑛to guide context genera-
tion as:
Given a question, first generate a helpful background
context. Then, provide 2-3 alternative phrasings
of the question with similar meaning.
This produces two outputs: a generated context Cgenand a set of
𝑘semantically equivalent queries {Q(𝑖)}𝑘
𝑖=1. These components en-
able the model to reason over both retrieved facts and inferred in-
tent.
Final Input Composition. The final structured input to the LVLM
integrates the sampled video frames, the retrieved auxiliary evi-
dence, the original query, and the generated latent context:
O=LVLM(𝐹𝑘𝑒𝑦,Concat (A𝑠𝑡,Q,{Q(𝑖)}𝑘
𝑖=1,Cgen)). (7)
RemaRK. By combining retrieval-grounded evidence with genera-
tive reasoning and query reformulation, this bi-level design strength-
ensbothexplicitandimplicitcontextunderstanding.Italsomitigates
alignment gaps between noisy auxiliary signals and the model’s pre-
trained knowledge space, resulting in more robust and semantically
coherent outputs.
4 Experiments
4.1 Datasets
We gauge the effectiveness of our framework on three publicly
accepted long-video testbeds. Video-MME [ 4] contains real-world
clips that run anywhere from 11 s to roughly one hour and probe
fine-grained understanding of everyday scenarios. MLVU [ 42] spans
nine evaluation tasks sourced from videos lasting 3 min to 2 h
(mean length about 12 min), challenging models to reason across
extended temporal ranges. Finally, LongVideoBench [ 33] targets
multimodal retrieval and reasoning in lengthy footage, offering
6,678 human-written multiple-choice questions distributed over 17
thematic categories.
4.2 Experimental Implementation
All experiments were executed on NVIDIA A100 GPUs. At the first
stage, we prune the LVLM’s detection prompts, keeping only con-
crete, CLIP-responsive objects and discarding abstract terms. For
following retrieval, we fix both the FAISS and CLIP acceptance
thresholds to 0.3; FAISS scores are computed with the IndexFlatIP
backend [ 8]. Our ablation suite is built around Long-LLaVA 7B [ 35],
whose extended context window and modest footprint make it
ideal for probing how RAG similarity cut-offs and frame-sampling
rates affect performance.

MM’25, October 27–31, 2025, Dublin, Ireland Zongsheng Cao et al.
Model #Text LLM Params Frames Short Medium Long Overall Gain
Proprietary LVLMs
GPT-4o [ 22] - - 384 80.0 70.3 65.3 71.9 -
Gemini-1.5-Pro [ 25] - - 0.5 fps 81.7 74.3 67.4 75.0 -
Open-Source LVLMs
Video-LLaVA [ 12] - 7B 8 44.6 38.3 35.8 39.6 -
Video-LLaVA + TV-RAG 2.0K 7B 8 49.7 44.3 42.6 44.1 +4.5
LLaVA-NeXT-Video [ 40] - 7B 16 49.4 43.0 36.7 43.0 -
LLaVA-NeXT-Video + TV-RAG 2.0K 7B 16 70.4 53.7 54.1 58.8 +15.8
LongVA [ 39] - 7B 32 60.9 49.3 44.0 51.4 -
LongVA + TV-RAG 1.8K 7B 32 66.2 62.1 58.7 61.7 +10.3
Long-LLaVA [ 35] - 7B 32 60.3 51.4 44.1 52.0 -
Long-LLaVA + TV-RAG 1.9K 7B 32 66.4 60.2 59.8 62.1 +10.1
Qwen2-VL [ 31] - 72B 32 75.0 63.3 56.3 64.9 -
Qwen2-VL + TV-RAG 2.1K 72B 32 76.3 70.5 72.1 74.9 +10.0
LLaVA-Video [ 41] - 72B 32 78.0 63.7 59.6 67.1 -
LLaVA-Video + TV-RAG 2.1K 72B 32 82.3 73.1 73.8 76.5 +9.4
Table 1: Performance evaluation on the Video-MME [ 4] benchmark.The #Text column logs the average volume of tokens that
TV–RAG appends for each sample. Most strikingly, the 72 B LLaVA-Video model [ 41] outfitted with our adapter edges past
the commercial Gemini-1.5-Pro [ 25]. The baseline results are from [ 18]. All open-source baselines and their TV–RAG variants
were re-benchmarked.
4.3 Main Results
Video-MME . To conduct a fair test of TV-RAG, we fixed every
candidate LVLM to the same 32-frame input budget, an especially
practical ceiling for the 72B-parameter models and a convenient
checkpoint for their 7B counterparts. Our benchmark therefore
covers six systems: four open 7B backbones (Video-LLaVA [ 12],
LLaVA-NeXT-Video [ 40], LongVA [ 39], Long-LLaVA [ 35]) and two
72B giants (Qwen2-VL [ 31], LLaVA-Video [ 41]). As reported in Ta-
ble1, attaching TV-RAG to the 72B pair lifts their scores beyond
the proprietary Gemini-1.5-Pro baseline [ 25]. Averaged over all six
backbones, the pipeline delivers a significant gain, with the most
dramatic jumps on extended footage. The boost stems from inject-
ing∼14 additional keyframes (about 2 000 tokens in total, given
the typical 144-token payload per frame), which compensates for
the weak visual grounding of LVLMs that were largely pre-trained
on text. These extra, text-rich frames act as semantic anchors and
enable sharper comprehension of complex video content.
MLVU . Table 2contrasts leading LVLMs on the benchmark’s multiple-
choice split. The vanilla 7B LLaVA-Video line scores 70.8, but once
TV-RAG is attached, the figure climbs to 72.6, an absolute gain of
+1.8that propels the small model ahead of every open-source rival
under 70B, including the 32B Oryx-1.5 (72.3). A similar but smaller
gain is observed for the 72B backbone, improving from 73.1 to 73.4
and further widening the gap with the proprietary GPT-4o base-
line (64.6). These improvements, achieved with the same 64-frame
budget, underline TV-RAG’s ability to supply complementary tex-
tual cues that sharpen video understanding, especially for lighter
models that still have headroom to benefit from richer context.
LongVideoBench . We next gauge the influence of TV–RAG on
both 7B and 72B LLaVA–Video backbones via the LongVideoBench
suite [ 33]. For clarity, we adopt the plain 64–frame feed and omitModel #Params Frames Overall
Proprietary LVLMs
GPT-4o [ 22] - 0.5 fps 64.6
Open-Source LVLMs
Video-CCAM [ 3] 14B 96 63.1
Video-XL [ 28] 7B 256 64.9
Aria [ 9] 25.3B 256 70.6
LLaVA-Video* [ 41] 7B 64 70.8
Oryx-1.5 [ 17] 32B 128 72.3
LLaVA-Video* [ 41] 72B 64 73.1
LLaVA-Video + TV-RAG 7B 64 72.6
LLaVA-Video + TV-RAG 72B 64 73.4
Table 2: The overall performance in the multiple-choice task
of the MLVU [ 42] benchmark. * donates the results of our
replication.
the benchmark’s optional interleaved prompt. Table 3reveals that
the 72B configuration, once augmented with our retrieval layer,
climbs to the runner-up position on the validation leaderboard:
its score edges past Gemini-1.5-Pro [ 25] by 1.6%and trails GPT-
4o [22] by merely 1.1%. The lighter 7B model also profits, posting a
+2.2%uplift under the same experimental regime. These results un-
derscore TV–RAG’s capacity to boost long-video reasoning across
model scales without altering input structure.
4.4 Ablation Studies
We probed the sensitivity of TV–RAG to the volume of visual evi-
dence by running the Long–LLaVA–7B backbone [ 35] under four

TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding MM’25, October 27–31, 2025, Dublin, Ireland
Model #Params Frames Overall
Proprietary LVLMs
Gemini-1.5-Pro [ 25] - 256 64.0
GPT-4o [ 22] - 256 66.7
Open-Source LVLMs
VideoChat2-Mistral [ 10] 7B 8 39.3
ShareGPT4Video [ 1] 7B 8 39.7
LLaVA-Next-Mistral [ 15] 7B 8 49.1
PLLaVA [ 34] 34B 16 53.2
LLaVA-Video [ 41] 7B 64 56.6
LLaVA-Video [ 41] 72B 64 61.9
LLaVA-Video + TV-RAG 7B 64 58.8
LLaVA-Video + TV-RAG 72B 64 65.6
Table 3: The overall performance on the validation set of
LongVideoBench [ 33].
Figure 3: Performance with different sampling frames rate
on Video-MME [ 4] when using Long-LLaVA-7B [ 35] as the
LVLM.
budgets: 8, 16, 32, and 64 sampled frames. Figure 3charts the ac-
curacies and reveals two clear trends. First, the retrieval layer de-
livers a measurable lift at every budget; second, the gap widens as
more frames are fed in, with the biggest jump on lengthy clips. In
the vanilla baseline (no retrieval), performance tops out at the 32-
frame mark, which demonstrates the effectiveness of our model.
We isolate each textual cue produced by the TV-RAG retriever,
including semantic-entropy weighting (SE), temporal-window gat-
ing (TW), OCR strings, ASR transcripts, and the context-enrichment
layer, and feed successive combinations to Long–LLaVA–7B [ 35].
Performance on Video-MME [ 4] (Table 4) climbs steadily with ev-
ery new source, with ASR and OCR delivering the significant jump
across short and long clips alike, confirming the unique value of
ASR and OCR transcripts.
SE (semantic-entropy) and TW (temporal-window) act as qual-
ity filters, discarding semantically weak or temporally mismatched
snippets so that only the most relevant text aligns with each frameSE TW OCR ASR Context Short Medium Long Overall
×✓ ✓ ✓ ✓ 65.3 56.5 49.1 57.0
✓ × ✓ ✓ ✓ 64.5 52.7 50.3 55.8
✓ ✓ × ✓ ✓ 68.9 57.8 55.0 60.6
✓ ✓ ✓ × ✓ 68.3 59.4 57.8 61.8
✓ ✓ ✓ ✓ × 65.4 55.6 56.4 59.1
✓✓ ✓ ✓ ✓ 72.4 64.1 59.3 65.3
Table 4: Results for combinations of different extracted texts
in Video-MME using LLaVA-Video-7B as the LVLM.
sequence. Taken together, these findings affirm the rationale be-
hind TV-RAG: multimodal retrieval, temporal alignment, and adap-
tive weighting converge to produce more faithful and grounded
video-language reasoning.
Figure 4: Qualitative result shown in Video-MME [ 4] bench-
mark when applying TV-RAG with LLaVA-Video [ 41].
Table 6reports how the similarity cutoff 𝜏shapes both efficiency
and accuracy when Long-LLaVA 7B answers the benchmark’s multiple-
choice questions. Lower bars ( 𝜏=0–0.1) flood the prompt with more
auxiliary tokens, pushing the inference time to 30-40 s but only
edge the overall score to 63-64. Raising the bar gradually prunes
redundant text and accelerates decoding; at 𝜏=0.3the model pro-
cesses 1.9 K tokens in 13 s yet still logs 61.6 overall, while post-
ing the single best medium-clip accuracy (61.0). Beyond this point,
the gains erode: 𝜏=0.5and𝜏=1.0shrink the token load to ≤0.3
K and time to ≤9s, but overall correctness drops sharply to 56.7
and 52.5, respectively. Hence 𝜏=0.3offers the most balanced trade-
off—halving latency relative to the lowest cutoffs, staying within
typical context limits ( ∼1.9 K extra tokens), and preserving strong
performance across clip lengths. For LVLMs with tighter windows,
a softer cutoff such as 𝜏=0.2may be preferred, still keeping the to-
ken budget under 3 K while avoiding the steep accuracy loss seen
at stricter thresholds.
To study the effectiveness of our model, we compare it with
Video-RAG [ 18]. As shown in Table 5, our method demonstrates a
clear performance boost over Video-RAG across all video lengths.
This improvement is observed in both the Long-Video and LLaVA-
Video models, with particularly noticeable gains in handling shorter
and longer videos. Overall, our model contributes to better model
performance by effectively addressing the challenges posed by dif-
ferent video durations. The observed gains can be attributed to our
framework’s ability to integrate temporal semantics more adap-
tively.

MM’25, October 27–31, 2025, Dublin, Ireland Zongsheng Cao et al.
Figure 5: Grad-CAM heatmaps of the final hidden state, alongside t-SNE projections of the user’s query and keyframe features,
are visualized for the example in Figure 4. The combined visualization makes clear that the supplementary texts retrieved by
TV-RAG tighten the vision–language link: they redirect the network’s attention toward the frames most pertinent to the
question, thereby boosting both answer precision and contextual fidelity.
𝜏#Token Time Short Medium Long Overall
0.0 3.6K 39s 66.3 58.4 57.1 63.4
0.1 3.4K 33s 68.7 57.3 59.1 62.9
0.2 2.7K 18s 66.4 60.2 58.2 60.5
0.3 1.9K 13s 67.4 61.0 58.8 61.6
0.4 0.8K 10s 64.6 58.8 58.5 60.3
0.5 0.3K 9s 63.4 54.5 50.1 56.7
1.0 0.0K 8s 60.4 51.3 44.7 52.5
Table 6: Performance with different thresholds of retrieval
on Video-MME [ 4] when using Long-LLaVA 7B [ 35] as the
LVLM. #Token and Time denote the total token number of
the extracted texts and the average inference time per ques-
tion, respectively.
Model Params Frames Short Medium Long Overall
X=LongVA 7B 32 60.9 49.3 44.0 51.4
X+Video-RAG 7B 32 65.4 59.1 55.7 60.1
X+ TV-RAG 7B 32 66.2 62.1 58.7 61.7
X=LLaVA-Video 72B 32 78.0 63.7 59.6 67.1
X+Video-RAG 72B 32 81.1 72.9 73.1 75.7
X+ TV-RAG 72B 32 82.3 73.1 73.8 76.5
Table 5: Performance with other model on Video-MME.
4.5 More Studies
We present more expriments for Video-MME [ 4] in Figure 4and
Figure 5to illustrate the effectiveness of our approach. As shown
in the figures, integrating external tools with LLaVA-Video to pro-
cess and retrieve extracted textual information from videos sig-
nificantly enhances the model’s ability to mitigate visual halluci-
nations. This integration leads to more accurate and contextually
grounded responses to user queries by reinforcing the alignment
between visual content and textual understanding.
Furthermore, the Grad-CAM [ 26] and t-SNE [ 30] visualization
results provide additional evidence supporting the impact of TV-
RAG in strengthening cross-modal alignment within the LVLM.Specifically, the Grad-CAM results highlight improved attention to
semantically relevant regions in the video frames, while the t-SNE
analysis reveals a more structured representation space, indicating
a refined relationship between visual and textual modalities. These
findings collectively underscore the effectiveness of our approach
in enhancing the robustness and interpretability of LVLM-based
video understanding.
In summary, the qualitative examples not only highlight the cor-
rectness of predictions but also provide insight into how the model
reasons through retrieved information.
5 Conclusion
We propose TV-RAG, a training-free adaptor that upgrades long-
video reasoning in large vision-language models. The scheme fuses
two key ideas: (i) a sliding temporal window that aligns retrieved
cues with the correct time span and (ii) an entropy-based score
that favours semantically rich segments, together compensating
for the narrow temporal receptive field of standard LVLMs and for
their difficulty in tracing subtle scene-level shifts. Contrary to re-
trievers that rely purely on text similarity or pipelines that demand
full model fine-tuning, TV-RAG clips onto any existing backbone
as a light, plug-and-play module. Extensive trials on Video-MME,
MLVU, and LongVideoBench confirm marked gains, allowing the
patched models to overtake contemporary leaders such as Gemini-
1.5-Pro and GPT-4o in some metrics. The future work targets finer
temporal gating and adaptive entropy scaling to further strengthen
cross-modal alignment and deep contextual reasoning. In future
work, we will extend TV-RAG to dynamically adjust its temporal
window size based on content uncertainty and explore curriculum-
style retrieval schedules to better adapt to ultra-long, open-world
video streams. We also plan to generalise the framework to audio-
visual co-reasoning and egocentric or 360° footage, broadening its
applicability across emerging multimodal benchmarks.
6 Acknowledgments
This work was supported by the Science and Technology Innova-
tion 2030-Key Project under Grant 2021ZD0201404.

TV-RAG: A Temporal-aware and Semantic Entropy-Weighted Framework for Long Video Retrieval and Understanding MM’25, October 27–31, 2025, Dublin, Ireland
References
[1]Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Ze-
hui Chen, Haodong Duan, Bin Lin, Zhenyu Tang, et al. 2024. ShareGPT4Video:
Improving Video Understanding and Generation with Better Captions. arXiv
preprint arXiv:2406.04325 (2024).
[2]Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. 2025.
Videoagent: A memory-augmented multimodal agent for video understanding.
InEuropean Conference on Computer Vision . Springer, 75–92.
[3]Jiajun Fei, Dian Li, Zhidong Deng, Zekun Wang, Gang Liu, and Hui Wang.
2024. Video-ccam: Enhancing video-language understanding with causal cross-
attention masks for short and long videos. arXiv preprint arXiv:2408.14023
(2024).
[4]Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan
Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. 2024. Video-MME:
The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in
Video Analysis. arXiv preprint arXiv:2405.21075 (2024).
[5]Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Composi-
tional visual reasoning without training. In Proceedings of the IEEE/CVF Confer-
ence on Computer Vision and Pattern Recognition . 14953–14962.
[6]Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning. arXiv preprint arXiv:2112.09118
(2021).
[7]JaidedAI. 2023. EasyOCR. https://github.com/JaidedAI/EasyOCR .
[8]Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs. IEEE Transactions on Big Data 7, 3 (2019), 535–547.
[9]Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao
Niu, Guoyin Wang, Bei Chen, and Junnan Li. 2024. Aria: An Open Multimodal
Native Mixture-of-Experts Model. arXiv preprint arXiv:2410.05993 (2024).
[10] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang,
Limin Wang, and Yu Qiao. 2023. Videochat: Chat-centric video understanding.
arXiv preprint arXiv:2305.06355 (2023).
[11] Yanwei Li, Chengyao Wang, and Jiaya Jia. 2025. Llama-vid: An image is worth
2 tokens in large language models. In European Conference on Computer Vision .
Springer, 323–340.
[12] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. 2023. Video-llava:
Learning united visual representation by alignment before projection. arXiv
preprint arXiv:2311.10122 (2023).
[13] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab,
Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. 2023.
Mm-vid: Advancing video understanding with gpt-4v (ision). arXiv preprint
arXiv:2310.19773 (2023).
[14] Qinghong Lin. 2023. VLog: Transform Video as a Document with ChatGPT,
CLIP, BLIP2, GRIT, Whisper, LangChain. https://github.com/showlab/VLog .
[15] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and
Yong Jae Lee. 2024. LLaVA-NeXT: Improved reasoning, OCR, and world knowl-
edge. https://llava-vl.github.io/blog/2024-01-30-llava-next/
[16] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao.
2024. Oryx MLLM: On-Demand Spatial-Temporal Understanding at Arbitrary
Resolution. arXiv preprint arXiv:2409.12961 (2024).
[17] Zuyan Liu, Yuhao Dong, Ziwei Liu, Winston Hu, Jiwen Lu, and Yongming Rao.
2024. Oryx MLLM: On-Demand Spatial-Temporal Understanding at Arbitrary
Resolution. arXiv preprint arXiv:2409.12961 (2024).
[18] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang,
Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. 2024. Video-RAG: Visually-
aligned Retrieval-Augmented Long Video Comprehension. arXiv preprint
arXiv:2411.13093 (2024).
[19] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi,
and Jianfei Cai. 2024. DrVideo: Document Retrieval Based Long Video Under-
standing. arXiv preprint arXiv:2406.12846 (2024).
[20] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan.
2023. Video-chatgpt: Towards detailed video understanding via large vision and
language models. arXiv preprint arXiv:2306.05424 (2023).
[21] Juhong Min, Shyamal Buch, Arsha Nagrani, Minsu Cho, and Cordelia Schmid.
2024. MoReVQA: Exploring Modular Reasoning Models for Video Question An-
swering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition . 13235–13245.[22] OpenAI. 2024. GPT-4o System Card. https://openai.com/index/gpt-4o-system-
card/ .
[23] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sand-
hini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.
2021. Learning transferable visual models from natural language supervision. In
International conference on machine learning . PMLR, 8748–8763.
[24] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey,
and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak super-
vision. In International conference on machine learning . PMLR, 28492–28518.
[25] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lil-
licrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Ju-
lian Schrittwieser, et al. 2024. Gemini 1.5: Unlocking multimodal understanding
across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).
[26] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedan-
tam, Devi Parikh, and Dhruv Batra. 2017. Grad-cam: Visual explanations from
deep networks via gradient-based localization. In Proceedings of the IEEE inter-
national conference on computer vision . 618–626.
[27] Yuzhang Shang, Bingxin Xu, Weitai Kang, Mu Cai, Yuheng Li, Zehao Wen, Zhen
Dong, Kurt Keutzer, Yong Jae Lee, and Yan Yan. 2024. Interpolating Video-LLMs:
Toward Longer-sequence LMMs in a Training-free Manner. arXiv preprint
arXiv:2409.12963 (2024).
[28] Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin, Junjie Zhou, Tiejun Huang,
and Bo Zhao. 2024. Video-XL: Extra-Long Vision Language Model for Hour-
Scale Video Understanding. arXiv preprint arXiv:2409.14485 (2024).
[29] Dídac Surís, Sachit Menon, and Carl Vondrick. 2023. Vipergpt: Visual inference
via python execution for reasoning. In Proceedings of the IEEE/CVF International
Conference on Computer Vision . 11888–11898.
[30] Laurens Van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-
SNE.Journal of machine learning research 9, 11 (2008).
[31] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin
Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-VL: Enhancing
Vision-Language Model’s Perception of the World at Any Resolution. arXiv
preprint arXiv:2409.12191 (2024).
[32] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. 2024. Videoa-
gent: Long-form video understanding with large language model as agent. arXiv
preprint arXiv:2403.10517 (2024).
[33] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. Longvideobench: A
benchmark for long-context interleaved video-language understanding. arXiv
preprint arXiv:2407.15754 (2024).
[34] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng.
2024. Pllava: Parameter-free llava extension from images to videos for video
dense captioning. arXiv preprint arXiv:2404.16994 (2024).
[35] Yin Song and Chen Wu and Eden Duthie. 2024. aws-prototyping/long-llava-
qwen2-7b. https://huggingface.co/aws-prototyping/long-llava-qwen2-7b
[36] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit
Bansal, and Gedas Bertasius. 2023. A simple llm framework for long-range video
question-answering. arXiv preprint arXiv:2312.17235 (2023).
[37] Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-llama: An instruction-
tuned audio-visual language model for video understanding. arXiv preprint
arXiv:2306.02858 (2023).
[38] Lu Zhang, Tiancheng Zhao, Heting Ying, Yibo Ma, and Kyusong Lee. 2024. Oma-
gent: A multi-modal agent framework for complex video understanding with
task divide-and-conquer. arXiv preprint arXiv:2406.16620 (2024).
[39] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuan-
han Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. 2024. Long
context transfer from language to vision. arXivpreprintarXiv:2406.16852 (2024).
[40] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi
Feng, Ziwei Liu, and Chunyuan Li. 2024. LLaVA-NeXT: A Strong Zero-shot
Video Understanding Model. https://llava-vl.github.io/blog/2024-04-30-llava-
next-video/
[41] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and
Chunyuan Li. 2024. Video Instruction Tuning With Synthetic Data.
arXiv: 2410.02713 [cs.CV] https://arxiv.org/abs/2410.02713
[42] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong,
Bo Zhang, Tiejun Huang, and Zheng Liu. 2024. MLVU: A Comprehensive Bench-
mark for Multi-Task Long Video Understanding. arXiv preprint arXiv:2406.04264
(2024).