# Everything Can Be Described in Words: A Simple Unified Multi-Modal Framework with Semantic and Temporal Alignment

**Authors**: Xiaowei Bi, Zheyuan Xu

**Published**: 2025-03-12 05:28:24

**PDF URL**: [http://arxiv.org/pdf/2503.09081v1](http://arxiv.org/pdf/2503.09081v1)

## Abstract
Long Video Question Answering (LVQA) is challenging due to the need for
temporal reasoning and large-scale multimodal data processing. Existing methods
struggle with retrieving cross-modal information from long videos, especially
when relevant details are sparsely distributed. We introduce UMaT (Unified
Multi-modal as Text), a retrieval-augmented generation (RAG) framework that
efficiently processes extremely long videos while maintaining cross-modal
coherence. UMaT converts visual and auditory data into a unified textual
representation, ensuring semantic and temporal alignment. Short video clips are
analyzed using a vision-language model, while automatic speech recognition
(ASR) transcribes dialogue. These text-based representations are structured
into temporally aligned segments, with adaptive filtering to remove redundancy
and retain salient details. The processed data is embedded into a vector
database, enabling precise retrieval of dispersed yet relevant content.
Experiments on a benchmark LVQA dataset show that UMaT outperforms existing
methods in multimodal integration, long-form video understanding, and sparse
information retrieval. Its scalability and interpretability allow it to process
videos over an hour long while maintaining semantic and temporal coherence.
These findings underscore the importance of structured retrieval and multimodal
synchronization for advancing LVQA and long-form AI systems.

## Full Text


<!-- PDF content starts -->

Everything Can Be Described in Words: A Simple Unified
Multi-Modal Framework with Semantic and Temporal
Alignment
Xiaowei Bi
Northwestern University
xiaoweibi2021@u.northwestern.eduZheyuan Xu
IEEE Member
cx1014@uw.edu
Abstract
Long Video Question Answering (LVQA)
is challenging due to the need for tempo-
ral reasoning and large-scale multimodal
data processing. Existing methods strug-
gle with retrieving cross-modal informa-
tion from long videos, especially when
relevant details are sparsely distributed.
We introduce UMaT (Unified Multi-modal
as Text), a retrieval-augmented genera-
tion (RAG) framework that efficiently pro-
cesses extremely long videos while main-
taining cross-modal coherence. UMaT
converts visual and auditory data into
a unified textual representation, ensur-
ing semantic and temporal alignment.
Short video clips are analyzed using a
vision-language model, while automatic
speech recognition (ASR) transcribes dia-
logue. These text-based representations
are structured into temporally aligned
segments, with adaptive filtering to re-
move redundancy and retain salient de-
tails. The processed data is embedded
into a vector database, enabling pre-
cise retrieval of dispersed yet relevant
content. Experiments on a benchmark
LVQA dataset show that UMaT outper-
forms existing methods in multimodal in-
tegration, long-form video understand-
ing, and sparse information retrieval. Its
scalability and interpretability allow it to
process videos over an hour long while
maintaining semantic and temporal co-
herence. These findings underscore the
importance of structured retrieval and
multimodal synchronization for advanc-
ing LVQA and long-form AI systems.
1 Introduction
The ability to analyze and interpret long-
form videos is essential for various appli-cations, including surveillance, medical di-
agnostics, educational indexing, and au-
tomated content summarization. While
short video comprehension has seen sig-
nificant advancements through multimodal
large language models (MLLMs) (Guo et al.,
2019)(Wang et al., 2022)(Ye et al., 2023)(Fu
et al., 2021)(Yang et al., 2022), long video
question answering (LVQA) remains a chal-
lenge due to the scale of multimodal data,
the difficulty of capturing long-range tempo-
ral dependencies, and the retrieval of key
details from sparsely distributed informa-
tion across extended video timelines. Tra-
ditional methods, such as long-range fea-
ture bank (Wu et al., 2019)(Cheng and Berta-
sius, 2022)(Zhang et al., 2022) or memory-
intensive (Wu et al., 2022) (Chen et al.,
2020)(Lee et al., 2021) architectures, often
lead to context loss, excessive redundancy,
or inefficient retrieval, making long-form
video understanding computationally expen-
sive and prone to missing critical informa-
tion.
To address these challenges, we intro-
duce UMaT (Unified Multi-modal as Text), a
Retrieval-Augmented Generation (Gao et al.,
2023) (RAG)-based framework designed to
process long-form videos while maintain-
ing semantic and temporal coherence effi-
ciently. Instead of attempting to process
entire videos at once, UMaT retrieves only
the most relevant multimodal content before
question answering, ensuring efficient rea-
soning and retrieval of sparsely occurring
yet contextually significant details. By sys-
tematically structuring, retrieving, and inte-
grating multimodal data, UMaT overcomes
fragmentation, redundancy, and inefficien-
1arXiv:2503.09081v1  [cs.CV]  12 Mar 2025

cies present in previous approaches.
A core aspect of UMaT is its fusion of multi-
modal data into structured textual represen-
tations, enabling seamless integration of vi-
sual and auditory content. Video frames are
converted into captions, while spoken dia-
logue is transcribed using automatic speech
recognition (ASR)(Radford et al., 2023). This
textual transformation ensures semantic
alignment between video and audio, enhanc-
ing retrieval precision and interpretability.
Structured temporal segmentation (Zhang
et al., 2024a) and adaptive deduplication fur-
ther optimize retrieved content by group-
ing short-term descriptions into coherent
segments while removing redundant infor-
mation, ensuring that the retained details
are concise yet comprehensive. This step is
crucial for retrieving contextually relevant
yet sparsely occurring details, prioritizing
salient moments embedded within long video
sequences.
To support question answering over ex-
tended durations, UMaT employs an efficient
retrieval strategy that dynamically selects
the most relevant video segments while man-
aging token constraints. Instead of process-
ing entire videos, multimodal content is em-
bedded into a structured retrieval system,
which prioritizes contextually significant seg-
ments to maintain semantic and temporal co-
herence. This retrieval-aware approach miti-
gates information overload, prevents loss of
critical but infrequent details, and ensures
that retrieved video-audio segments remain
aligned with the question context.
Unlike fixed-length summarization ap-
proaches that often truncate or dilute es-
sential details, UMaT adapts its retrieval
strategy based on video length and content
complexity, enabling precise content selec-
tion while avoiding hallucinations (Huang
et al., 2025)and irrelevant outputs. Its scal-
ability and adaptability make it a practical
solution for real-world LVQA applications,
allowing seamless integration with vision-
language models (Liu et al., 2023b)(Liu et al.,
2023a)(Liu et al., 2024), ASR techniques,
and retrieval architectures. Additionally, itsstructured retrieval mechanism enhances in-
terpretability, making long-form video analy-
sis more transparent and efficient.
By dynamically synchronizing multimodal
information and prioritizing key details from
sparsely distributed data, UMaT enables effi-
cient and scalable long-form video reasoning,
ensuring that even videos exceeding an hour
in length can be processed without unneces-
sary computational overhead.
1.1 Contributions
We summarize our key contributions as fol-
lows:
•UMaT: A Retrieval-Augmented LVQA
Framework that decomposes long-form
video understanding into structured re-
trieval and reasoning, enabling efficient
and scalable content selection.
•A unified textual representation strategy
that converts video and audio modalities
into structured text, ensuring semantic
and temporal alignment across multi-
modal data.
•A structured segmentation and dedupli-
cation mechanism that preserves contex-
tual integrity while filtering redundancy,
optimizing retrieval of sparse but criti-
cal details.
•An efficient retrieval-based LVQA
pipeline that dynamically prioritizes rel-
evant video-audio segments, enhancing
context-aware, token-efficient reason-
ing without unnecessary computational
overhead.
•A modular and interpretable framework,
adaptable to evolving vision-language
models and retrieval architectures, lay-
ing the foundation for future advance-
ments in long-form multimodal AI.
By addressing the challenges of long-form
video comprehension, UMaT establishes a
structured and scalable approach to mul-
timodal retrieval and reasoning, enabling
efficient processing of extended video se-
quences with sparsely distributed informa-
tion.
2

2 Related Work
2.1 Retrieval Augmented Generation
for LVQA Tasks
Utilizing the multi-modal generalization ca-
pability of LLMs in VQA tasks has been an
active research field. (Wang et al., 2023)
proposed a framework that enables LLMs
to proactively ask questions to gather more
information from images, thereby improving
the performance of open-domain knowledge-
based VQA tasks. (Lin and Byrne, 2022) pro-
posed a joint training scheme for VQA that
includes differentiable Dense Passage Re-
trieval (DPR) integrated with answer gener-
ation. The authors found that this approach
outperformed recent VQA systems that train
DPR separately from answer generation. Fol-
lowed by FVQA 2.0 (Lin et al., 2023), a new
dataset that contains adversarial samples to
address the limitations of the original FVQA
dataset. The authors showed that systems
trained with the original FVQA train sets can
be vulnerable to adversarial samples, and
demonstrated an augmentation scheme to
reduce this vulnerability without human an-
notations. PreFLMR (Lin et al., 2024) was
proposed, which is a pre-trained fine-grained
late-interaction multi-modal retriever, and
achieved state-of-the-art results in a range
of Knowledge-based Visual Question Answer-
ing (KB-VQA) tasks. (Luo et al., 2024) pro-
posed Video-RAG, which employs visually-
aligned auxiliary texts to improve the per-
formance of large visual-language models
(LVLMs), and leverages open-source exter-
nal tools to extract visually-aligned informa-
tion from pure video data alongside video
frames and queries, in a plug-and-play man-
ner.
2.2 Unified Representation of
Multimodal Data
One of the challenges for unlocking the mul-
timodal capability of LLMs is the lack of con-
sistency between different input modalities,
due to the differing dimensions and repre-
sentational power of each modality. (Xia
et al., 2024) introduced a new task calledCross-Modal Generalization (CMG) to learn
a unified discrete representation from paired
multimodal data. The authors proposed
Uni-Code, a new framework that facilitates
bidirectional supervision between modali-
ties and aligns semantically equivalent in-
formation in a shared discrete latent space.
(Huang et al., 2024) proposed two training-
free methods to enhance the performance of
the DCID model: Training-free Optimization
of Codebook (TOC) and Hierarchical Dual
Cross-modal Information Disentanglement
(H-DCID). TOC optimizes the codebook by
selecting important channels in the unified
space without retraining. H-DCID extends
information separation and alignment to two
levels, capturing more cross-modal details.
(Zhu and Li, 2023) proposed a symmetric
two-stream contrastive learning framework
for image-text retrieval, which consists of a
Bidirectional Encoder Representations from
Transformers (BERT) pre-trained text en-
coder and a Vision Transformer (ViT) pre-
trained image encoder. The authors utilized
a cross-modal contrastive loss and two sym-
metric uni-modal contrast losses to train the
model in an unsupervised manner. (Shu
et al., 2024) proposed Video-XL, a novel
approach that leverages MLLMs’ inherent
key-value (KV) sparsification capacity to con-
dense the visual input. Specifically, the au-
thors introduced a new special token, the
Visual Summarization Token (VST) for each
interval of the video, which summarizes the
visual information within the interval as its
associated KV .
2.3 Temporal Segmentation and
Content deduplication
(Tirumala et al., 2023) showed that careful
data selection via pre-trained model embed-
dings can speed up LLM pre-training and
improve downstream performance. The au-
thors found that repeating data intelligently
consistently outperforms baseline training
while repeating random data performs worse
than baseline training. (Liu et al., 2023c) in-
troduces a novel dynamic video token mask-
ing strategy along with masking video mod-
3

eling during training, which reduces the
length of sequences input to the LLM while
significantly improving the robustness of
videos of varying lengths during inference.
(Qian et al., 2024) proposed Momentor, a
video-LLM capable of accomplishing fine-
grained temporal understanding tasks. The
authors designed an automatic data gener-
ation engine to construct Moment-10M, a
large-scale video instruction dataset with
segment-level instruction data. They trained
Momentor on Moment-10M, enabling it to
perform segment-level reasoning and local-
ization. (Xu et al., 2024b) proposed SlowFast-
LLaVA, a training-free video LLM that uses a
two-stream SlowFast design to capture both
detailed spatial semantics and long-range
temporal context. The Slow pathway ex-
tracts features at a low frame rate while
keeping as much spatial detail as possible
(e.g., with 12 ×24 tokens), and the Fast path-
way operates on a high frame rate but uses
a larger spatial pooling stride (e.g., down-
sampling 6x) to focus on the motion cues.
(Park et al., 2024) presents LVNet, which in-
corporates a unique Hierarchical Keyframe
Selector that progressively reduces the num-
ber of keyframe candidates to reduce com-
putation costs. (Wang et al., 2024a) itera-
tively selects keyframes and gathers relevant
information to answer questions about the
video, which emphasizes reasoning and iter-
ative processing, achieving state-of-the-art
results on EgoSchema and NExT-QA bench-
marks while using fewer frames than previ-
ous methods.
2.4 Multimodal Content Integration
and Long-form Video
Understanding
(Weng et al., 2024) proposed LongVLM, a
Video-based Large Language Model (Vide-
oLLM) for long video understanding, which
decomposes long videos into multiple short-
term segments and encodes local features
for each segment via a hierarchical to-
ken merging module. These features are
concatenated in temporal order to main-
tain the storyline across sequential short-term segments. Additionally, LongVLM in-
tegrates global semantics into each local
feature to enhance context understanding.
(Zhang et al., 2024a) proposed LLoVi, a
language-based framework for long-range
video question-answering. LLoVi uses a
short-term visual captioner to generate tex-
tual descriptions of short video clips (0.5-8
seconds in length) densely sampled from a
long input video. Afterward, an LLM aggre-
gates the densely extracted short-term cap-
tions to answer a given question. (Li et al.,
2023b) focused on a novel task called Inten-
tQA, a specific type of VideoQA that centers
on intent reasoning. (Xu et al., 2024a) iden-
tifies challenges in adapting image-language
models to video tasks, including perfor-
mance vulnerability to prompt changes and
data scaling failures, and proposed a pooling
strategy to mitigate these issues.
3 Method
3.1 Short-term Video Clip Captioning
Given a long-form video input V, we seg-
ment it into Nvnon-overlapping short video
clips v={vm}N
m=1, where vm∈RTv×H×W×3
andTv,H,Wdenote the number of frames,
height and width of each short video clip,
respectively.
Each segment clip vmundergoes visual
captioning using a pre-trained short-term
vision-language model ϕ. This model gen-
erates textual descriptions cm=ϕ(vm),
where cm= (w1, ..., w Lm)andwirepresents
the i-th word in the caption of length Lm.
These short-term captions serve as the fun-
damental representation of the visual con-
tent within the video clips.
To ensure rich and structured visual de-
scriptions, we use a specialized prompt for
caption generation that focuses on describ-
ing scene composition, object attributes, ac-
tion, and spatial relationships while omitting
emotional interpretations, contrary to (Li
et al., 2023a), as we believe it improves the
factual accuracy and relevance of the cap-
tions, leading to better alignment between
video content and downstream question an-
swering. Example Caption Output:
4

1{
2 " caption ": "A red car moves
along a narrow road
surround by tall trees ."
3}
3.2 Automatic Speech Recognition
(ASR) Processing
Simultaneously, we apply an automatic
speech recognition (ASR) model such as
Whisper (Radford et al., 2023) to extract
spoken content from the corresponding au-
dio segments. Given an audio clip amcor-
responding to vm, the ASR model ψpro-
duces transcriptions tm=ψ(am), forming
the spoken-language representation of the
video segment.
The ASR output is structured as follows:
1{
2 " start ": 0,
3 " end ": 9,
4 " text ": "NT Live is all about
trying to emulate a live
performance ",
5},
6{
7 " start ": 9,
8 " end ": 17,
9 " text ": "It ’s a whole new
experience for those who
can ’t be",
10}
with start and ending timestamp specified,
as well as the transcribed text. The struc-
tured output enables precise temporal align-
ment between audio transcriptions and video
captions, ensuring multimodal consistency
across different modalities. By maintaining a
structured format, there ASR output allows
for efficient integration with downstream re-
trieval and reasoning tasks.
3.3 Long-range Video Representation
with Temporal Aggregation
To facilitate efficient long-form video un-
derstanding, we first chunk the short-termcaptions and ASR transcriptions into struc-
tured segments based on their timestamps.
Given Nvshort-term textural representa-
tions{(cm, tm)}Nv
m=1, we aggregate them into
fixed-length temporal segments Skof length
Ts:
Sk=(k+1)Ts−1[
m=kTs(cm, tm),∀k∈ {1, ...,Nv
Ts}
where each segment skmaintains sequential
consistency while reducing redundancy.
Once the textual representations are ag-
gregated into temporally aligned segments,
we concatenate the video captions and ASR
transcriptions within each segment to form
a unified multimodal representation. This
ensures alignment between visual and audi-
tory modalities while preserving contextual
coherence.
To further optimize text representation,
we employ deduplication to remove overlap-
ping or repeated content. We use Sequence-
Matcher API, which is part of Python difflib
module to detect the longest common sub-
strings between adjacent captions cmand
remove redundant text. Given two consec-
utive captions ciandci+1, we compute the
similarity score:
Sim(ci, ci+1) =2× |L|
|ci|+|ci+1|
where |L|is the length of the longest com-
mon substring. If Sim(ci, ci+1)exceeds a
predefined threshold, the redundant portion
ofci+1is removed.
3.4 Retrieval-Augmented Generation
for Context Selection
Given the structured video representations
{Sk}, directly feeding all segments into an
LLM would exceed its token limit. Instead,
we use retrieval-based selection to dynami-
cally fetch only the most relevant segments
for question answering.
3.4.1 Embedding and Indexing
Each aggregated segment Skis mapped into
a high-dimensional vector representation ek
5

using a sentence embedding model ϕe:
ek=ϕe(Sk)
All embeddings {ek}are stored in a vector
database, allowing for fast retrieval based
on semantic similarity. The study employs
FAISS (Douze et al., 2024)(Facebook AI Sim-
ilarity Search) with IndexFlatL2, a brute-
force nearest neighbor search that computes
L2 (Euclidean) distances between vectors,
which helps to ensure efficient similarity
matching for selecting relevant video seg-
ments.
3.4.2 Temporal Chunking and Text
Deduplication
We aggregate captions and transcripts into
longer time windows (30s or 60s) to reduce
retrieval complexity while preserving mean-
ingful context. This method prevents exces-
sive fragmentation of information and im-
proves the coherence of retrieved segments.
To prevent redundant inputs to the LLM,
a deduplication step removes repetitive text
while retaining semantically distinct details.
This ensures that each query receives a com-
pact yet comprehensive context for reason-
ing.
3.4.3 Query-Based Retrieval
During question answering, the given ques-
tion Q and answer choices A are encoded
using the same embedding model:
eQ=ϕe(Q, A)
we perform a similarity search between eQ
and{ek}using cosine similarity:
ˆS= arg max
kcos(eQ, ek)
where the top-K most similar segments ˆSare
selected to construct the final prompt for the
LLM.
4 Experiment
We evaluate UMaT on Video-MME (Fu et al.,
2024), a benchmark designed to assess long
video question answering (LVQA) across mul-
tiple reasoning tasks. Our evaluation com-
pares UMaT against state-of-the-art VLMmodels, measuring improvements in re-
trieval precision and long-range reasoning
accuracy.
4.1 Dataset: Video-MME
Video-MME is a comprehensive multimodal
evaluation benchmarking dataset designed
specifically for video-based large language
models (VLMs). Unlike previous video QA
datasets, which often focus on short videos
and limited scenarios, Video-MME provides:
•Diverse Video Types: The dataset spans
six primary visual domains—Knowledge,
Film & Television, Sports, Artistic Per-
formance, Life Record, and Multilin-
gual, further subdivided into 30 fine-
grained categories (e.g., astronomy, es-
ports, news reports, documentaries).
•Variable Video Lengths: Video durations
range from 11 seconds to 1 hour, ensur-
ing models are tested on both short-form
and long-form reasoning.
•Multimodal Integration: The dataset in-
cludes video frames, subtitles, and au-
dio, offering a holistic evaluation of mul-
timodal reasoning capabilities.
•High-Quality Annotations: The dataset
features 900 manually selected videos,
annotated with 2,700 expert-verified
question-answer pairs, ensuring accu-
racy and relevance.
4.2 Baseline Models
We compare UMaT against three state-of-the-
art VLM models, selected for their relevance
to retrieval-augmented long-video process-
ing:
•LLaVA-NeXT-Video (Liu et al., 2024) –
A vision-language model designed for
video reasoning, leveraging frame-level
alignment.
•LongVA (Zhang et al., 2024b) – An en-
hanced video QA model incorporating
temporal modeling for long-form videos.
6

Figure 1: Flowchart showing retrieval process, prompt construction and final LLM response genera-
tion. Raw videos are firstly fed into a pre-trained visual-language model (VLM) and an automatic
speech recognition (ASR) model, temporally segmented and captioned separately. The captions from
these two modalities are then fused in alignment and fed into an FAISS embedding engine, which
maintains a vector database under the hood. In Q&A phase, top K contents are retrieved from the
vector database for prompt assembly, which is then fed into a pre-trained large language model.
•Long-LLaVA (Wang et al., 2024b) – An
extended version of LLaVA, optimized
for long video comprehension.
UMaT is a model-agnostic framework, mean-
ing any VLM model can be applied within
its retrieval-augmented pipeline to improve
long-form video reasoning.
4.3 Quantitative Results
We report QA accuracy across short,
medium, and long videos, as shown in Ta-
ble 1
4.4 Qualitative Analysis
To better understand the effectiveness of our
approach, we conduct a qualitative analysis
by examining specific cases from the Video-
MME dataset. These cases highlight the ad-
vantages of our method and provide insights
into areas that need further improvement.
4.4.1 Effect of Video Captioning
Prompt Design
•In a video from the "Artistic Perfor-
mance" category, a ballet dancer is per-
forming a complex sequence involvingjumps and spins. Using a detailed de-
scriptive prompt, the generated caption
includes an exhaustive breakdown of
every motion: "The dancer executes
a pirouette, followed by a grand jeté,
arms extended gracefully." While rich
in details, this level of description clut-
ters the retrieval system with unneces-
sary information. On the other hand, a
concise summary prompt merely states:
"A dancer performs ballet on stage,"
omitting crucial interactions such as the
presence of props or stage lighting ef-
fects. Our balanced prompt generates:
"A ballet dancer performs a dynamic
routine, interacting with stage props
and dramatic lighting effects." This
maintains key details while ensuring ef-
ficient retrieval, leading to improved
question-answering accuracy. For in-
stance, when answering: "What prop
did the dancer interact with most in the
performance?" the balanced prompt al-
lows the model to correctly retrieve the
caption mentioning a silk ribbon, while
7

Table 1: Video-MME Benchmark Results
Model #Params Frames Short
(%)Medium
(%)Long
(%)Overall
LLaVA-NeXT-Video 7B 16 50.2 44.3 38.6 44.4
LLaVA-NeXT-Video
+ UMaT7B 5/s 60.2 51.2 49.2 53.5(+9.1)
LongVA 7B 32 59.2 50.3 45.0 51.5
LongVA + UMaT 7B 5/s 68.5 63.1 58.3 63.3(+11.8)
Long-LLaVA 7B 32 59.5 52.0 45.3 52.2
Long-LLaVA +
UMaT7B 5/s 70.3 65.1 62.2 65.9(+13.7)
the concise summary would lack this de-
tail, leading to an incorrect answer.
•In a video from the "Artistic Perfor-
mance" category, a dancer is perform-
ing a complex routine. When using a
detailed descriptive prompt, the cap-
tioning model generates an exhaustive
breakdown of each movement, resulting
in an overly verbose output. A concise
summary prompt, on the other hand,
fails to mention critical visual cues, such
as the interaction between the dancer
and stage props. Our balanced prompt
effectively highlights key actions while
removing redundant details, leading to
improved retrieval and reasoning by the
LLM.
•Using properly designed prompts allows
the model to retain essential informa-
tion while removing redundant details.
•This improves retrieval precision, lead-
ing to higher accuracy in reasoning-
based questions.
4.4.2 Effect of Video Captioning
Prompt Design
•In a "Documentary" video covering a sci-
entist’s life, the beginning of the video
details their early research on climate
change, while a later segment discusses
their contributions to renewable energy
policy. Without RAG, the LLM struggles
to correlate these two distant sections,
resulting in a fragmented or incorrectresponse. For example, when answer-
ing: "How did Dr. X’s early research
influence their later policies?" without
RAG, the model might focus only on the
latter part and provide an incomplete
answer. With RAG, the system retrieves
both sections, forming a coherent an-
swer: "Dr. X’s early studies on green-
house gases directly informed their later
push for renewable energy incentives,
shaping modern sustainability policies."
This demonstrates how RAG enhances
long-range reasoning by retrieving dis-
persed but interlinked segments of in-
formation.
•In a "Documentary" video, the sub-
ject discusses historical events spanning
decades. Without RAG, the LLM is over-
whelmed by the sheer amount of text,
resulting in responses that are either
too vague or missing crucial temporal
references. When RAG is employed,
only the most relevant excerpts from
the video captions and ASR transcripts
are retrieved, allowing the model to ac-
curately determine the correct timeline
of events. This significantly improves
the model’s performance, particularly
on long-duration videos.
•For short videos, the system can directly
aggregate all video captions and ASR
transcriptions for reasoning. However,
for long videos, the volume of textual in-
put exceeds LLM limits, requiring RAG
retrieval.
8

•RAG helps focus on the most relevant
information, improving long-video rea-
soning accuracy.
4.4.3 Effects of ASR on Model
Performance
•In a "Sports Competition" video of a bas-
ketball game, the play-by-play commen-
tary provides critical information about
the players and strategies. Without ASR,
the system relies only on visual captions,
producing descriptions like: "A player
makes a shot from a distance." This
lacks context about who made the shot
and what strategy was used. With ASR,
the model retrieves: "LeBron James ex-
ecutes a three-pointer from the corner
after a quick pass from Davis, securing
the lead with 10 seconds left." This ad-
ditional detail is essential for answering
questions like: "Who made the decisive
shot in the final moments?" or "What
strategy led to the final point?" ASR al-
lows the model to capture spoken in-
sights that are not visually apparent, sig-
nificantly improving answer accuracy.
•In a "Sports Competition" video, a com-
mentator describes a play-by-play break-
down of a soccer match. Without ASR,
the system relies solely on visual cap-
tions, failing to capture critical moments
like player names and strategy discus-
sions. With ASR integrated, the LLM
correctly identifies the key players in-
volved in the decisive goal and pro-
vides a more informed answer. This
demonstrates that ASR is essential for
question-answering tasks that depend
on spoken context rather than visual
cues.
•ASR enhances model accuracy, particu-
larly in speech-heavy videos.
•For short videos, the impact of ASR is
minor since visual captions often suffice.
•For long videos, ASR significantly boosts
comprehension, helping the model cor-
rectly answer speech-driven questions.5 Conclusion
We introduced UMaT (Unified Multi-modal
as Text), a retrieval-augmented framework
for long video question answering (LVQA)
that transforms multimodal content into a
unified textual representation. By integrat-
ing video captions and ASR transcriptions,
UMaT ensures semantic and temporal align-
ment, while structured segmentation and
adaptive deduplication optimize retrieval of
sparse but critical details. Its efficient re-
trieval strategy overcomes the fragmenta-
tion and token limitations of existing meth-
ods, enabling scalable and context-aware
reasoning over extended video sequences.
UMaT provides a modular and inter-
pretable foundation for multimodal AI, adapt-
able to evolving vision-language models and
retrieval architectures. Future work will en-
hance alignment strategies, sparsity-aware
retrieval, and dataset expansion, further ad-
vancing retrieval-augmented multimodal rea-
soning.
References
Yihong Chen, Yue Cao, Han Hu, and Liwei Wang.
2020. Memory enhanced global-local aggre-
gation for video object detection. In 2020
IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) , pages 10334–
10343.
Feng Cheng and Gedas Bertasius. 2022.
Tallformer: Temporal action localization
with&nbsp;a&nbsp;long-memory transformer.
InComputer Vision – ECCV 2022: 17th Eu-
ropean Conference, Tel Aviv, Israel, October
23–27, 2022, Proceedings, Part XXXIV , page
503–521, Berlin, Heidelberg. Springer-Verlag.
Matthijs Douze, Alexandr Guzhva, Chengqi
Deng, Jeff Johnson, Gergely Szilvasy, Pierre-
Emmanuel Mazaré, Maria Lomeli, Lucas Hos-
seini, and Hervé Jégou. 2024. The faiss library.
ArXiv .
Chaoyou Fu, Yuhan Dai, Yondong Luo, Lei Li,
Shuhuai Ren, Renrui Zhang, Zihan Wang,
Chenyu Zhou, Yunhang Shen, Mengdan Zhang,
et al. 2024. Video-mme: The first-ever com-
prehensive evaluation benchmark of multi-
modal llms in video analysis. arXiv preprint
arXiv:2405.21075 .
Tsu-Jui Fu, Linjie Li, Zhe Gan, Kevin Lin,
William Yang Wang, Lijuan Wang, and Zicheng
9

Liu. 2021. Violet : End-to-end video-language
transformers with masked visual-token model-
ing. ArXiv , abs/2111.12681.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang
Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
Qianyu Guo, Meng Wang, and Haofen Wang.
2023. Retrieval-augmented generation for
large language models: A survey. ArXiv ,
abs/2312.10997.
Daya Guo, Jiangshui Hong, Binli Luo, Qirui Yan,
and Zhangming Niu. 2019. Multi-modal repre-
sentation learning for short video understand-
ing and recommendation. 2019 IEEE Inter-
national Conference on Multimedia & Expo
Workshops (ICMEW) , pages 687–690.
Hai Huang, Yan Xia, Shengpeng Ji, Shulei Wang,
Hanting Wang, Jieming Zhu, Zhenhua Dong,
and Zhou Zhao. 2024. Unlocking the poten-
tial of multimodal unified discrete represen-
tation through training-free codebook opti-
mization and hierarchical alignment. ArXiv ,
abs/2403.05168.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong
Zhong, Zhangyin Feng, Haotian Wang, Qian-
glong Chen, Weihua Peng, Xiaocheng Feng,
Bing Qin, and Ting Liu. 2025. A survey on
hallucination in large language models: Prin-
ciples, taxonomy, challenges, and open ques-
tions. ACM Trans. Inf. Syst. , 43(2).
Sangmin Lee, Hak Gu Kim, Dae Hwi Choi, Hyung-
Il Kim, and Yong Man Ro. 2021. Video pre-
diction recalling long-term motion context
via memory alignment learning. In 2021
IEEE/CVF Conference on Computer Vision
and Pattern Recognition (CVPR) , pages 3053–
3062.
Cheng Li, Jindong Wang, Kaijie Zhu, Yixuan
Zhang, Wenxin Hou, Jianxun Lian, and Xingxu
Xie. 2023a. Large language models under-
stand and can be enhanced by emotional stim-
uli.Preprint , arXiv:2307.11760.
Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng
Fan. 2023b. Intentqa: Context-aware video
intent reasoning. In 2023 IEEE/CVF Interna-
tional Conference on Computer Vision (ICCV) ,
pages 11929–11940.
Weizhe Lin and Bill Byrne. 2022. Retrieval aug-
mented visual question answering with outside
knowledge. In Proceedings of the 2022 Con-
ference on Empirical Methods in Natural Lan-
guage Processing , pages 11238–11254, Abu
Dhabi, United Arab Emirates. Association for
Computational Linguistics.
Weizhe Lin, Jingbiao Mei, Jinghong Chen, and
Bill Byrne. 2024. PreFLMR: Scaling up fine-
grained late-interaction multi-modal retriev-
ers. In Proceedings of the 62nd Annual Meet-ing of the Association for Computational Lin-
guistics (Volume 1: Long Papers) , pages 5294–
5316, Bangkok, Thailand. Association for Com-
putational Linguistics.
Weizhe Lin, Zhilin Wang, and Bill Byrne. 2023.
FVQA 2.0: Introducing adversarial samples
into fact-based visual question answering.
InFindings of the Association for Computa-
tional Linguistics: EACL 2023 , pages 149–157,
Dubrovnik, Croatia. Association for Computa-
tional Linguistics.
Haotian Liu, Chunyuan Li, Yuheng Li, and
Yong Jae Lee. 2023a. Improved baselines with
visual instruction tuning.
Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuan-
han Zhang, Sheng Shen, and Yong Jae Lee.
2024. Llava-next: Improved reasoning, ocr,
and world knowledge.
Haotian Liu, Chunyuan Li, Qingyang Wu, and
Yong Jae Lee. 2023b. Visual instruction tuning.
Ruyang Liu, Chen Li, Haoran Tang, Yixiao Ge,
Ying Shan, and Ge Li. 2023c. St-llm: Large
language models are effective temporal learn-
ers.https://arxiv.org/abs/2404.00308 .
Yongdong Luo, Xiawu Zheng, Xiao Yang,
Guilin Li, Haojia Lin, Jinfa Huang, Ji-
ayi Ji, Fei Chao, Jiebo Luo, and Ron-
grong Ji. 2024. Video-rag: Visually-aligned
retrieval-augmented long video comprehen-
sion. Preprint , arXiv:2411.13093.
Jong Sung Park, Kanchana Ranasinghe, Kumara
Kahatapitiya, Wonjeong Ryoo, Donghyun Kim,
and Michael S. Ryoo. 2024. Too many frames,
not all useful: Efficient strategies for long-
form video qa. ArXiv , abs/2406.09396.
Long Qian, Juncheng Li, Yu Wu, Yaobo Ye, Hao
Fei, Tat-Seng Chua, Yueting Zhuang, and Sil-
iang Tang. 2024. Momentor: advancing video
large language model with fine-grained tem-
poral reasoning. In Proceedings of the 41st
International Conference on Machine Learn-
ing, ICML’24. JMLR.org.
Alec Radford, Jong Wook Kim, Tao Xu, Greg
Brockman, Christine McLeavey, and Ilya
Sutskever. 2023. Robust speech recognition
via large-scale weak supervision. In Proceed-
ings of the 40th International Conference on
Machine Learning , ICML’23. JMLR.org.
Yan Shu, Peitian Zhang, Zheng Liu, Minghao Qin,
Junjie Zhou, Tiejun Huang, and Bo Zhao. 2024.
Video-xl: Extra-long vision language model
for hour-scale video understanding. ArXiv ,
abs/2409.14485.
10

Kushal Tirumala, Daniel Simig, Armen Agha-
janyan, and Ari S. Morcos. 2023. D4: im-
proving llm pretraining via document de-
duplication and diversification. In Proceed-
ings of the 37th International Conference on
Neural Information Processing Systems , NIPS
’23, Red Hook, NY, USA. Curran Associates
Inc.
Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Ser-
ena Yeung-Levy. 2024a. Videoagent: Long-
form video understanding with large language
model as agent. In Computer Vision – ECCV
2024: 18th European Conference, Milan, Italy,
September 29–October 4, 2024, Proceedings,
Part LXXX , page 58–76, Berlin, Heidelberg.
Springer-Verlag.
Xidong Wang, Dingjie Song, Shunian Chen, Chen
Zhang, and Benyou Wang. 2024b. Longllava:
Scaling multi-modal llms to 1000 images effi-
ciently via a hybrid architecture. ArXiv .
Yi Wang, Kunchang Li, Yizhuo Li, Yinan He,
Bingkun Huang, Zhiyu Zhao, Hongjie Zhang,
Jilan Xu, Yi Liu, Zun Wang, Sen Xing, Guo
Chen, Junting Pan, Jiashuo Yu, Yali Wang,
Limin Wang, and Yu Qiao. 2022. Intern-
video: General video foundation models via
generative and discriminative learning. ArXiv ,
abs/2212.03191.
Ziyue Wang, Chi Chen, Peng Li, and Yang
Liu. 2023. Filling the image information
gap for vqa: Prompting large language mod-
els to proactively ask questions. Preprint ,
arXiv:2311.11598.
Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun
Chang, and Bohan Zhuang. 2024. Longvlm:
Efficient long video understanding via large
language models. In Computer Vision – ECCV
2024: 18th European Conference, Milan, Italy,
September 29–October 4, 2024, Proceedings,
Part XXXIII , page 453–470, Berlin, Heidelberg.
Springer-Verlag.
Chao-Yuan Wu, Christoph Feichtenhofer, Haoqi
Fan, Kaiming He, Philipp Krähenbühl, and
Ross Girshick. 2019. Long-term feature banks
for detailed video understanding. In 2019
IEEE/CVF Conference on Computer Vision and
Pattern Recognition (CVPR) , pages 284–293.
Chao-Yuan Wu, Yanghao Li, Karttikeya Man-
galam, Haoqi Fan, Bo Xiong, Jitendra Malik,
and Christoph Feichtenhofer. 2022. Memvit:
Memory-augmented multiscale vision trans-
former for efficient long-term video recogni-
tion. 2022 IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR) , pages
13577–13587.
Yan Xia, Hai Huang, Jieming Zhu, and Zhou Zhao.
2024. Achieving cross modal generalizationwith multimodal unified representation. Ad-
vances in Neural Information Processing Sys-
tems , 36.
Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin,
See Kiong Ng, and Jiashi Feng. 2024a. Pllava
: Parameter-free llava extension from images
to videos for video dense captioning. Preprint ,
arXiv:2404.16994.
Mingze Xu, Mingfei Gao, Zhe Gan, Hong-
You Chen, Zhengfeng Lai, Haiming Gang,
Kai Kang, and Afshin Dehghan. 2024b.
Slowfast-llava: A strong training-free base-
line for video large language models. ArXiv ,
abs/2407.15841.
Antoine Yang, Antoine Miech, Josef Sivic, Ivan
Laptev, and Cordelia Schmid. 2022. Zero-shot
video question answering via frozen bidirec-
tional language models. In Proceedings of the
36th International Conference on Neural In-
formation Processing Systems , NIPS ’22, Red
Hook, NY, USA. Curran Associates Inc.
Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye,
Ming Yan, Yi Zhou, Junyan Wang, Anwen
Hu, Pengcheng Shi, Yaya Shi, Chenliang Li,
Yuanhong Xu, Hehong Chen, Junfeng Tian,
Qiang Qi, Ji Zhang, and Feiyan Huang. 2023.
mplug-owl: Modularization empowers large
language models with multimodality. ArXiv ,
abs/2304.14178.
Ce Zhang, Taixi Lu, Md Mohaiminul Islam,
Ziyang Wang, Shoubin Yu, Mohit Bansal, and
Gedas Bertasius. 2024a. A simple LLM frame-
work for long-range video question-answering.
InProceedings of the 2024 Conference on Em-
pirical Methods in Natural Language Process-
ing, pages 21715–21737, Miami, Florida, USA.
Association for Computational Linguistics.
Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao
Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue
Wang, Haoran Tan, Chunyuan Li, and Ziwei
Liu. 2024b. Long context transfer from lan-
guage to vision. ArXiv , abs/2406.16852.
Zhuosheng Zhang, Aston Zhang, Mu Li, and
Alex Smola. 2022. Automatic chain of thought
prompting in large language models.
Yi Zhu and Xiu Li. 2023. Iterative uni-modal and
cross-modal clustered contrastive learning
for image-text retrieval. 2023 International
Conference on Pattern Recognition, Machine
Vision and Intelligent Algorithms (PRMVIA) ,
pages 15–23.
11