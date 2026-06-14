# Rethinking RAG in Long Videos: What to Retrieve and How to Use It?

**Authors**: Yuho Lee, Jisu Shin, Nicole Hee-Yeon Kim, Jihwan Bang, Juntae Lee, Kyuwoong Hwang, Fatih Porikli, Hwanjun Song

**Published**: 2026-06-11 10:05:49

**PDF URL**: [https://arxiv.org/pdf/2606.13141v1](https://arxiv.org/pdf/2606.13141v1)

## Abstract
Retrieval-augmented generation is moving beyond text into long, egocentric video, where systems must select query-relevant chunks across multiple modalities and temporal granularities. Yet progress in VideoRAG is limited by two gaps: existing benchmarks allow queries to be answered without the video, obscuring retrieval errors, and prior methods apply a single modality-granularity configuration per query, ignoring chunk-level variability. We address both by introducing V-RAGBench, a benchmark of $\langle$query, evidence chunk, answer$\rangle$ triplets that enables faithful, decoupled evaluation of retrieval and generation, and CARVE, a simple method that runs parallel retrievers across configurations and employs chunk-adaptive reranking to identify the winning configuration for each chunk. Each chunk then enters the generator under its winning configuration selected during retrieval, yielding an interleaved evidence form where the chunk-level decision propagates across both stages. CARVE outperforms eight recent VideoRAG baselines, with the chunks supplied to the generator interleaving multiple configurations rather than sharing a single one, a behavior unattainable by query-level methods.

## Full Text


<!-- PDF content starts -->

Rethinking RAG in Long Videos:
What to Retrieve and How to Use It?
Yuho Lee1, Jisu Shin1, Nicole Hee-Yeon Kim1, Jihwan Bang2, Juntae Lee2, Kyuwoong Hwang2,
Fatih Porikli2Hwanjun Song1
1KAIST,2Qualcomm AI Research†, Qualcomm Korea YH, Seoul, Republic of Korea
Retrieval-augmented generation is moving beyond text into long, egocentric video, where
systems must select query-relevant chunks across multiple modalities and temporal granularities.
Yet progress in VideoRAG is limited by two gaps: existing benchmarks allow queries to be
answered without the video, obscuring retrieval errors, and prior methods apply a single
modality-granularity configuration per query, ignoring chunk-level variability. We address
both by introducing V-RAGBench , a benchmark of ⟨query, evidence chunk, answer ⟩triplets
that enables faithful, decoupled evaluation of retrieval and generation, and CARVE , a simple
method that runs parallel retrievers across configurations and employs chunk-adaptive reranking
to identify the winning configuration for each chunk. Each chunk then enters the generator
under its winning configuration selected during retrieval, yielding an interleaved evidence
form where the chunk-level decision propagates across both stages. CARVE outperforms eight
recent VideoRAG baselines, with the chunks supplied to the generator interleaving multiple
configurations rather than sharing a single one, a behavior unattainable by query-level methods.
Date:June 11, 2026
Correspondence:Hwanjun Song (songhwanjun@kaist.ac.kr)
First Authors (equal contribution):
Yuho Lee (yuholee@kaist.ac.kr)
Jisu Shin (jisu3389@kaist.ac.kr)
Nicole Hee-Yeon Kim (nicole2kim@gmail.com)
†Qualcomm AI Research is an initiative of Qualcomm Technologies, Inc.
1 Introduction
Retrieval-augmented generation (RAG) grounds large language models (LLMs) in external knowl-
edge, improving factual reliability [ 13,22,56,86]. Its frontier is now moving beyond static text
corpora toward video-centric settings [ 28,77,84,71], where the primary source of knowledge is no
longer curated text butlong-form videoin theegocentricsetting, as wearable devices turn daily life
into long first-person videos [ 6,73] and agentic systems operate directly on accumulated personal
video data [ 8,51]. We refer to this setting, RAG over long videos, as VideoRAG , where the system
needs to retrieve query-relevant evidence from a long video. This shift does not merely extend
retrieval but fundamentally increases its retrieval and generation complexity. Given a target query, the
system must retrieve query-relevant video chunks by matching acrossmultimodal representations
[30,45,81], such as visual features and higher-level textual abstractions, as well as across multiple
temporal granularities[ 23,40], from frame-level details to clip-level segments. The problem thus
moves beyond what to retrieve to deciding "which modality and which granularity to retrieve from?"
and "how to exploit them jointly across retrieval and generation?"
Despite this growing interest [ 39,44,58], current benchmark datasets have not kept pace. While
hour-scale source videos are available through Ego4D [ 19] and EgoLife [ 73], the queries paired with
them are scarce and were not designed for VideoRAG . In particular, recent work [ 37] shows that over
half of widely used video QA samples can be answered without the video at all, relying on linguistic
priors, world knowledge, or static cues. Such queries are fatal for VideoRAG , since a system can reach
correct answers even when retrieval surfaces irrelevant chunks, sohigh generation accuracy does
not imply successful retrieval, and final accuracy fails as a proxy for the retrieval stage. Under such
evaluation, the right way to use modality and temporal granularity across retrieval and generation
cannot be identified, as each design choice is invisible behind the final accuracy.
The same gap appears on the methodological side. Recent work proposes techniques for VideoRAG ,
such as guided query refinement [ 66,71], modality fusion [ 42,74], and query decomposition [ 70],
while others delegate retrieval to iterative agentic loops [ 44,77] that act more as test-time scalingarXiv:2606.13141v1  [cs.AI]  11 Jun 2026

𝐶ℎ𝑢𝑛𝑘 5: TC
Text -Clip
(Clip -level description)
Visual -Frame
(Keyframe visual features)Text -Frame
(Keyframe descriptions)
Stage 1. Parallel Candidate Pooling
QueryModality -Interleaved 
Generation
Generator Model 
(MLLM)
Stage 2. Chunk -Adaptive Reranking
Multimodal 
Reranker
Parallel Retrievals
Visual -Clip
(Clip -level visual features)Top-kchunks 
Chunk 1…Chunk 2
Chunk k
Chunk 1
Chunk 2
Chunk k
Chunk 1
Chunk 2
Chunk k
Chunk 1
Chunk 2
Chunk k
Candidate Pool
(+ Configuration Tags)
… … …Multi -modal 
Cross -EncoderAssign 
Reranking Score
TC: Highest ScoresWinning 
Configuration
𝐶ℎ𝑢𝑛𝑘1: TF
𝐶ℎ𝑢𝑛𝑘 30: VF
Final top -k with 
mixed representations
I see various representations…
𝑐7
Representation TypeChunk Index𝑐7𝑐12𝑐35𝑐5
Compete based on 
Reranking Score𝑐1𝑐19𝑐7𝑐21
𝑐26𝑐18𝑐3𝑐10
𝑐2𝑐6𝑐30𝑐41TC TC TC TC
TF TF TF TF
VC VC VC VC
VF VF VF VF
Configuration Tag
Candidate PoolFigure 1: The overview of CARVE : Stage 1 builds a candidate pool via chunk-wise parallel retrieval,
while Stage 2 performs chunk-adaptive reranking. The final top- kevidence with their winning
configuration is passed to the generator in a modality-interleaved form.
than as principled retrieval. Although retrieval is the core component on which all these frameworks
ultimately rest, its validation is left at a shallow level, reduced to downstream QA accuracy on the
full pipeline. This is in sharp contrast to text-centric RAG, where retrieval is treated as a first-class
object of study and audited through stage-wise metrics on query-evidence pairs [13, 17, 34].
We address this gap by introducing V-RAGBench , a dedicated benchmark designed to make re-
trieval and generation in VideoRAG faithfully measurable both in isolation and in their interaction.
V-RAGBench consists of ⟨query, evidence chunk, answer ⟩triplets drawn from long egocentric videos
in Ego4D [ 19] and EgoLife [ 73], totaling 2,100 high-quality triplets across 216videos spanning 1–9
hours. Targeting the VideoRAG setting, V-RAGBench jointly enforces three properties that no existing
dataset satisfies together: each query is (i)tied to non-recurring evidence, so that the underlying
event does not repeat elsewhere in the video as a near-duplicate; (ii)visually-grounded, so that the
answer is not implied by the query itself or recoverable from parametric knowledge alone; and
(iii)evidence-localized, so that the answer cannot be reconstructed from generic visual cues that
incidentally recur across non-evidence chunks. Property (i) is enforced via temporal segmentation
and clustering, while (ii) and (iii) are enforced by dedicated post-hoc filters. Together, these properties
make generation causally dependent on retrieval, enabling a stage-wise view ofVideoRAG.
With V-RAGBench in hand, we shift the main question of VideoRAG from improving QA accuracy to
which representation of a video chunk best supports each stage of the pipeline, namely retrieval and
generation. Each chunk admits four candidate configurations1, formed by crossing two axes: modality
(visual embeddings vs. embeddings of textual summaries) and temporal granularity (frame-level vs.
clip-level). Existing retrieval strategies for VideoRAG [21,22,28,53] apply a single configuration, or
a fixed fusion of the four, uniformly to every candidate chunk given a query—aquery-leveldecision.
However, this is suboptimal because the best configuration is a property of the chunk’s content, not
of the query; a visually salient moment is most discriminable through visual embeddings, while a
semantically rich moment is better surfaced through its textual summary, and granularity follows
the same content dependence. We therefore propose CARVE (Chunk- Aware Reranking for Video
Evidence) in Figure 1, a novel framework that exploits multi-modal cross-encoder signals to assign
each candidate chunk its winning configuration based on its query relevance. The key idea is to
re-purpose reranking, which is a tool widely used in text-RAG [ 1,36,64] but largely unexplored in
VideoRAG, as a mechanism forchunk-levelconfiguration decision.
Concretely, CARVE operates in two stages. First, parallel retrieval runs four parallel retrievers, one
per configuration, each contributing its top- kchunks to a shared candidate pool. Second, a multi-
modal cross-encoder re-scores each candidate under its retrieving configuration (e.g., by query–text
relevance for textual retrievals and by query-visual relevance for visual retrievals). Sorting these
configuration-specific scores yields the final ranking, in which chunks from different configurations
are interleaved and each chunk carries its retrieving configuration onward. The final top- kchunks
then serve both stages of the VideoRAG pipeline. For retrieval, they are the output of CARVE , where
the chunk-level decision lies in which configuration scored each chunk highest. For generation, the
same chunks are augmented to the query, but each is supplied to the generator under its retrieving
configuration, so the representation fed to the generator varies per chunk. Importantly, our experiments
reveal that this chunk-level selection is not only beneficial at the retrieval stage, but becomes even
more impactful when carried into generation, leading to further gains in answer accuracy.
1Other retrieval signals, such as video knowledge graphs [ 16,26,55] or metadata (e.g., explicit temporal clue) [ 20,52], can
aid retrieval but are not consistently available. We focus on content-based representations, which are universally accessible.
2

Our experiments on V-RAGBench reveal four key findings: (1) CARVE significantly outperforms
eight baseline methods on both retrieval and generation, a comparison enabled by V-RAGBench ’s
stage-wise measurability; (2) a modality–granularity ablation reveals large performance gaps across
configurations at both stages, with no single configuration uniformly optimal, motivating chunk-level
rather than query-level decisions; (3) CARVE ’s chunk-level decisions distribute fairly evenly across
the four configurations, showing that per-chunk diversity is genuine rather than collapsing onto a
single dominant choice; and (4) at generation, CARVE even surpasses a trained query-level router that
learns to pick one configuration per query, without any additional training.
2 Related Work
Retrieval-Augmented Generation in Text.RAG improves generation by retrieving query-relevant
evidence from an external text corpus and conditioning the generator on it [ 2,13,29,34]. Progress has
centered on retrieval, since retrieval quality largely dictates final-answer quality. One axis advances
the retriever, from sparse and dense retrievers [ 17,31,72] to hybrid sparse–dense pipelines that
combine complementary signals [ 22,46,54]. A second reformulates the query through rewriting
[43,63,87] or expansion [ 4,13,83] to make retrieval intent explicit. A third addresses the limited
precision of first-stage retrievers through reranking, where a cross-encoder re-scores candidates
[1,36,64,78]. Together, these advances substantially improve retrieval quality, with gains that largely
translate into downstream generation accuracy. They are, however, developed for text and do not
readily generalize to retrieving evidence from video.
Retrieval-Augmented Generation in Long Video.Long-form video understanding has traditionally
relied on universal video representations such as sparse frame sampling [ 7,38,75], visual-token
compression [ 12,24,60,82], or full-video textualization [ 32,35,41,81]. Yet, these encode the entire
video into a query-independent view, diluting query-relevant evidence as length grows. This motivates
VideoRAG, which mirrors text RAG by retrieving query-relevant evidence before generation.
The main challenge in VideoRAG is that retrieval and generation become substantially more complex
than in text. A video chunk is continuous in time and inherently multimodal [ 30,45,81], expressible
through visual frames or derived textual abstractions, and at temporal granularities from individual
frames to segments [ 23,40]. Existing work explores these axes from complementary angles. On the
modality side, Jeong et al. [28] retrieves chunks using multimodal embeddings, while Luo et al. [42]
augments retrieval with visually-aligned auxiliary text such as OCR and ASR. On the granularity side,
hierarchical and tree-structured representations index the video at multiple temporal scales [ 53,65].
A separate line replaces one-shot retrieval with iterative, agentic probing, where an LLM repeatedly
fetches frames or segments to refine its evidence [8, 62, 71, 77].
However, these methods commit to a single modality–granularity choice (or a fixed fusion of all of
them) per query, and analysis is weighted toward downstream QA accuracy, leaving retrieval quality
itself less directly examined. As we discuss next, this stems from the benchmark side, where existing
datasets are not designed to evaluate retrieval separately from generation. Our work addresses both
gaps by treating modality and granularity as explicit design axes selected at the chunk-level, and by
introducing a benchmark that makes retrieval and generation independently measurable.
Video Benchmark Datasets for QA.Video QA benchmarks have progressed from short third-person
clips [ 27,33,69,79] toward longer videos and, increasingly, egocentric settings [ 15,19,47,48,73].
This shift mirrors the natural target of VideoRAG : personal video has emerged as a primary retrieval
domain, as wearable devices accumulate hour-scale first-person video and personal agents increasingly
operate over it [ 51,88]. However, existing benchmarks remain limited in this regime. For example,
EgoSchema (3-minute clips) [ 47] QaEgo4D (short Ego4D intervals) [ 15], Video-MME ( ≤60 min)
[18], and LongVideoBench (≤1 hour) [67].
While Ego4D [ 19] and EgoLife [ 73] provide hour-scale source video, their accompanying queries
fall short on two counts. First, public queries are scarce and serve different evaluation goals. Ego4D’s
queries are scarce and belong to the Episodic Memory benchmark, which evaluates retrieval alone,
while EgoLifeQA targets long-term memory aggregation across hours and days. Second, neither
targets VideoRAG evaluation, and thus lacks the three properties it requires: (i) non-recurring evidence,
(ii) visual grounding, and (iii) evidence localization. These ensure that retrieval and generation can
be assessed independently and jointly, with correct answers tied to the labeled evidence rather than
to spurious cues. CG-Bench [ 9] and MM-Lifelong [ 10] make partial progress by checking shortcut
3

answerability and clue grounding, addressing (ii) but leaving (i) and (iii) untouched. Our benchmark
closes this gap by jointly enforcing all three properties on hour-scale egocentric video.
3 Formulating VideoRAG along Two Design Axes
We formalize VideoRAG as a retrieve-then-generate process over long videos. While prior studies
[28,53,77] establish this pipeline, two design choices remain underspecified. First, the same chunk
admits multiple representations across modality (visual vs. textual) and granularity (frame vs. clip),
but these are typically combined at fixed scales rather than treated as decision variables. Second, the
relationship between the representation used for retrieval and the one used for generation has received
little attention, with the two typically tied to a single choice by default.
Following standard text RAG practice [ 2,56], aVideoRAG system partitions a long video Vinto non-
overlapping, fixed-interval temporal chunks that serve as retrieval units, yielding V={v 1, . . . , v |V|}.
We default to 2-minute chunks, matching the empirical 2–3 minute span of bounded semantic content
in long-form video [ 25]. Each such chunk, however, is fundamentally richer than a text passage: v
admits multiple representations ϕm,g(v)along two complementary axes: a modality m∈ {vis,text}
(visual features vs. textual abstractions) and a temporal granularity g∈ {frame,clip} (frame-
level details vs. clip-level summaries).2Given a query q, a retriever Rmatches qagainst chunk
representations to select kcandidate chunks Vq⊆V, and an MLLM-based generator Gproduces an
answer from those chunks under a chosen representation:
ba=G 
q,{ϕ mg,gg(v)}v∈Vq
whereV q=R(q,{ϕ mr,gr(v)}v∈V),(1)
(mr, gr)and(mg, gg)are the configurations chosen for retrieval and generation, respectively; and
ϕm,g(v)is the representation of chunk vunder modality mand granularity g. The goal is twofold:
Vqshould contain the chunk that supports the answer, and bashould match the reference answer a.
Notably, whereas a text chunk has a single canonical form, a video chunk forces VideoRAG to decide
which(m, g)to use for retrieval and which for generation, and these decisions need not coincide.
We now describe how the two design axes are instantiated in this work. Full implementation details
(e.g., decoding hyperparameters, temperature) are deferred to Appendix D, and the prompt templates
used are provided in Appendix G.
Chunk Modality.The modality axis m∈ {vis,text} exposes a chunk vthrough two complementary
channels. Thevisualmodality encodes vas raw visual features extracted by an MLLM-based
multimodal encoder (Qwen3-VL-Embedding-2B [ 5] by default), preserving fine-grained appearance
and motion cues that resist verbalization. In contrast, thetextualmodality describes vin natural
language by prompting an MLLM (Qwen3-VL-8B) to summarize the chunk v, and embeds the
resulting text with the same encoder, surfacing semantic structure (e.g., what is happening, where, and
with whom) at the cost of compressing pixel-level detail. Neither modality is universally preferable;
the more useful one depends on the query and chunk content, as textual abstractions surface semantics
invisible to raw pixels, while visual features capture appearance and motion.
Chunk Granularity.Even within a fixed modality, the temporal scope at which a chunk is read
can vary. The granularity axis g∈ {frame,clip} exposes a chunk vat two contrasting scales. The
frame-level retains fine-grained, per-frame information through a small set of keyframes. Specifically,
we cluster all frame embeddings in the chunk with k-means++ [ 3] (n= 5 by default) and take the
cluster centroids as keyframes, covering distinct visual states rather than uniform time samples. On
the other hand, theclip-level instead treatsvas a single unit.
They are then realized in a modality-specific way. Under the visual modality, the frame-level mean-
pools the keyframe embeddings while the clip-level encodes the entire chunk into one embedding.
Under the textual modality, the same MLLM produces a summary capped at 512tokens, applied
per keyframe (concatenated before embedding) for the frame-level and over the entire chunk for
the clip-level. This cap acts as an upper bound that accommodates detailed clip descriptions used in
prior multimodal summarization work [ 11,25] while preventing verbose outputs from dominating
retrieval and generation context. Neither granularity is universally preferable; frame-level forms
capture transient details but lose the surrounding flow, while clip-level forms retain temporal context
but blur transient details, so the more useful scale depends on the query and the chunk content.
2We instantiate MandGwith two values each, giving four (m, g) configurations for ease of exposition. Our formulation
readily accommodates additional modalities and granularities.
4

Scope and Objective.With (m, g) now made explicit, two questions follow:howit should be decided
andat what level. In other words, for each chunk v, which representation ϕm,g(v)should be passed
to the retriever and the generator? The simplest choice is to fix one (m, g) per query and apply it
uniformly across chunks [ 28,53]; agentic variants [ 77] issue multiple retrievals per query, but each
call still commits to a single (m, g) . This leaves room for finer-grained control, since the most useful
(m, g) may differ across chunks even within a single query. For instance, identifying a specific object
visible in only one keyframe is best handled by a frame-level visual embedding, while describing how
an activity unfolds over the chunk is better expressed by a clip-level textual summary. We therefore
explore deciding (m, g) per chunk, carrying the choice consistently into both retrieval and generation
while keeping the standard retrieve-then-generate pipeline intact.
4 Construction of V-RAGBench
Diagnosing how the two design axes interact at retrieval and generation requires a faithful benchmark
in which the two stages are measurable both in isolation and in their interaction, which in turn requires
every query to be tied to a uniquely sufficient evidence chunk. We therefore construct V-RAGBENCH,
a benchmark of ⟨query, evidence chunk, answer ⟩triplets built from hour-scale egocentric videos
through the four stages (see Figure 2)—namely source selection, event extraction and deduplication,
query generation, and filtering—described below. Our construction process is further verified by
human study (see Section C.5): on 577 sampled QA pairs, annotators show substantial inter-annotator
agreement (Cohen’sκ= 0.63), jointly flagging only 3.1% as unanswerable.
Source Video Selection.We focus on egocentric video as the source domain for VideoRAG . Unlike
third-person long videos (e.g., movies, lectures) organized around a single narrative arc, first-person
streams capture continuous, weakly structured daily activity in which the same actor revisits similar
scenes and routines, making retrieval over personal memory both more realistic and substantially
harder. Thus, we draw source videos from two complementary egocentric corpora, Ego4D [ 19] and
EgoLife [ 73], retaining only videoslonger than one hourand applying stratified sampling over
length bins so that the corpus spans a wide range of durations. This yields a search corpus of 216
personal videos in total, comprising 42EgoLife recordings (avg. 379.1 min., 177.8 –552.7 min.) and
174Ego4D videos (avg. 86.0 min., 60.0–423.6 min.); see Table 6. Reflecting the personal nature of
VideoRAG , retrieval is performed within each wearer’s own corpus, so every query is grounded in
their video history. Per-domain and per-length statistics are in Appendix C.1.
Event Extraction and Deduplication.This stage converts each long video into a small set of distinct
event chunks that form a non-redundant evidence pool for QA generation. For each video, we encode
every sampled frame with EV A02-E-14-plus [ 57] (a CLIP-based vision encoder) and apply kernel
temporal segmentation [ 49] on the resulting frame embeddings to cut the video into semantically
coherent, time-contiguous segments. The number of segments is capped at 150for the shorter Ego4D
videos and 900for the day-long EgoLife recordings, with each segment constrained to 5–300s. We
then re-embed each segment as a whole with Qwen3-VL-Embedding-8B [ 5] in video mode and
cluster the embeddings with k-Means++ ( k= 50 for Ego4D, k= 300 for EgoLife), retaining only
the segment closest to each centroid as its representative. This suppresses recurring routines (e.g.,
repeated meal preparation), leaving chunks that correspond todistinct events—a prerequisite for
QA generation, since queries grounded in recurring events would admit multiple sufficient evidence
chunks; while natural in the real world, such queries are excluded here for rigorous VideoRAG
evaluation. Details of deduplication and statistics of this stage are in appendix C.4.
Query Generation.For each event chunk, we prompt Gemini-3-flash-preview to generate up to
three candidate queries spanning three predefined categories (See Appendix C.3), with the model
itself deciding which are applicable rather than forcing all three. The prompt further enforces
contextual localization, requiring every query to embed anchor information from its source event
so that the query alone describes the situation enough to be matched against the right chunk; this
letsV-RAGBench dispense with direct temporal cues such as timestamps or when phrases, forcing
retrieval to rely purely on content (full prompt and examples in Appendix G). Each generation
produces a triple ⟨q,[t s, te], a⟩ of query, evidence time frames, andopen-endedform rather than
a multiple-choice option, ensuring that downstream evaluation reflects the generator’s true ability
rather than being inflated by random guessing over a fixed choice set. This yields 67,370 candidate
⟨query,evidence chunk,answer⟩triples for the filtering pipeline.
5

Post-hoc Filtering.While event deduplication enforces content uniqueness at the event level, the
generated queries are natural-language artifacts whose grounding and uniqueness must be verified at
the query level. Specifically, the two propertiesvisual grounding—the answer is not implied by the
query or recoverable from its priors—andevidence localization—the answer is not recoverable from
non-evidence chunks—are not yet guaranteed and must be enforced post-hoc.
We thus apply five filters targeting these properties along with query-level redundancy. (1)Semantic
Similarity: queries too close in Qwen3-VL-8B embedding space are removed, with per-video caps
of50(Ego4D) and 300(EgoLife) to prevent any single video from dominating; (2)Answerability:
GPT-5.2-chat is asked whether each query is answerable from its source clip, and unanswerable ones
are removed; (3)Shortcut Bias: queries GPT-5.2-chat answers correctly without visual context are
discarded; (4)Empirical Answerability: Gemini-2.5-flash must produce a correct answer from the
source clip, judged by an LLM-as-a-judge; and (5)Evidence Uniqueness: for each query, we retrieve
the top- 10chunks whileexcludingthe true evidence, and discard the query if Gemini-2.5-flash can
still answer correctly from this non-evidence set, indicating the answer is recoverable elsewhere.
The surviving queries are (1) lexically and semantically distinct, (2, 4) grounded in sufficient evidence,
(3) free of parametric and commonsense shortcuts, and (5) not recoverable from any other chunk of
the same video. Since the three query categories are unevenly populated after filtering, we randomly
subsample each category to match the smallest one ( n= 700 per category), yielding a balanced
dataset of2,100queries in total (1,800train,300test).3We defer per-stage details to Appendix C.5.
5 CARVE: Chunk-Aware Reranking for Video Evidence
We now introduce CARVE , a novel VideoRAG method that decides the modality–granularity configura-
tion(m, g) at thechunk-level rather than at the query-level, and propagates that decision consistently
from retrieval into generation. Concretely, CARVE operates in two simple stages, namely parallel
candidate pooling and chunk-adaptive reranking, as illustrated in Figure 1.
Stage 1: Parallel Candidate Pooling.Given a query q, this stage builds acandidate poolby perform-
ing four parallel retrievals, one per configuration (m, g) withm∈ {vis,text} andg∈ {frame,clip} .
The four embeddings ϕm,g(v)of a video chunk vdefined in Eq. (1), one per configuration, are
precomputed offline for every chunk in the corpus, so each retrieval at inference time only requires
encoding the query qinto an embedding and performing a fast nearest-neighbor lookup against the
precomputed chunk embeddings. That is, no additional chunk-encoding cost is incurred in this stage,
regardless of how many configurations are queried in parallel.
For each configuration (m, g) , we compute the dot-product similarity ⟨·,·⟩ between the query
embedding qand each chunk embedding ϕm,g(v), and keep the kchunks with the highest scores as
the top-kretrieval listC (m,g)(q)for that configuration:
C(m,g)(q) =top-kchunksvranked by⟨q, ϕ m,g(v)⟩.(2)
The four lists are then merged into a configuration-tagged candidate pool P(q) , where each retrieved
chunk is paired with the configuration(m, g)that retrieved the chunkv:
P(q) =[
(m,g)n 
v|{z}
chunk,(m, g)|{z}
tag
:v∈C (m,g)(q)o
.(3)
The same chunk vmay appear under multiple tags if it was retrieved by more than one configuration.
These tags are preserved into Stage 2 rather than collapsed away, since they are precisely the per-
configuration signal that lets the next stage decide a winning(m, g)for each chunk.
Stage 2: Chunk-Adaptive Reranking.Given the candidate pool P(q) from Stage 1, this stage selects
the final top- kchunks, while assigning each surviving chunk a singlewinningconfiguration (m∗, g∗).
Since the chunk embeddings in P(q) live in different modalities depending on the tag—textual when
m= text , visual when m= vis —we use a multi-modal cross-encoder CE(·,·)4that jointly attends
to a textual query and either textual or visual chunk modality. We use Qwen3-VL-Reranker-2B as the
default reranker, though performance remains consistent across rerankers, as discussed in Section F.5.
3The split train set is specifically used for the analysis in-depth study in Table 5, where we train two retrieval routers.
4Temporal granularity gdoes not change the modality of ϕm,g (v), since frame-level and clip-level forms under the same
modality share the same encoder branch and embedding space, differing only in temporal scope (Section 3).
6

Importantly, each chunk vinP(q) is re-scoredonlyunder the tag (m, g) that retrieved it, not under
all four configurations. The rerank score of a tagged candidate(v,(m, g))∈ P(q)is computed:
˜s 
q; (v,(m, g))
= CE 
q, ϕm,g(v)
.(4)
For each chunkv, we keep only its highest-scoring tag as its winning configuration:
(m∗
v, g∗
v) = arg max
(m,g) : (v,(m,g))∈P(q)˜s 
q; (v,(m, g))
.(5)
CEscores are comparable across modalities, so chunks can be ranked together by their winning
scores, yielding the final top-klist (V∗
q) with each chunk tagged by its winning configuration:
V∗
q=n 
v,(m∗
v, g∗
v)
:v∈top-kchunks ranked by˜s 
q; (v,(m∗
v, g∗
v))o
.(6)
This chunk-adaptive reranking improves retrieval by judging each chunk under its most discriminative
configuration and letting chunks from different configurations compete on a comparable scale, so the
top-knaturally interleaves the strongest evidence across modalities and granularities.
Modality-Interleaved Generation. V∗
qbenefits not only retrieval but also generation. Unlike prior
work that fixes a single (mg, gg)for the whole query [ 28,53] or fuses several representations uni-
formly, we expose each chunk to the generator Gonlythrough its winning representation ϕm∗v,g∗v(v):
ba=G 
q,
ϕm∗v, g∗v(v) : (v,(m∗
v, g∗
v))∈V∗
q	
.(7)
The generator context thus mixes representations chunk by chunk, with each chunk rendered under its
winning configuration. Empirically, this propagation from retrieval to generation—not the retrieval
gain alone—is whereCARVE’s strongest improvements appear.
6 Evaluation on V-RAGBench
Our evaluation of CARVE on the test set of V-RAGBench consists of three parts. First, we compare
CARVE against recent VideoRAG methods on both retrieval and generation to demonstrate its superi-
ority (Table 1). Second, we study how modality and temporal granularity should be selected under
our method, showing that CARVE ’s chunk-level choices outperform query-level alternatives (Table 2).
Third, we further show the efficiency and effectiveness of chunk-level decisions via latency, selection
distribution, and comparison against two trained query-level routers (Tables 3–5).
Compared Methods.We compare CARVE against eight RAG methods applicable to VideoRAG ,
which differ in modality, granularity, and fusion strategy. We group them into two categories. (i)
System-levelmethods apply a fixed policy regardless of the query, including VideoRAG-A [ 28], which
linearly interpolates text and visual modality scores; GQR [ 59], which refines query embeddings
at test time via complementary retriever signals; and the multimodal encoders, namely Freeret [ 89]
and GME [ 85], which retrieve in a single embedding space. (ii)Query-levelmethods adapt the
policy per query but apply the same decision to all chunks, including RRF [ 50], which expands a
query into variants and fuses retrieval ranks; DAT [ 22], which dynamically weights dense and sparse
retrievers via LLM-assessed reference scores; and VideoRAG-B [ 42] and UniversalRAG(-LoRA)
[76], which respectively decompose or route a query across auxiliary databases spanning different
modalities and granularities, where (-LoRA) denotes its LoRA-trained variant on the training set of
V-RAGBench . For fairness, all baselines share the same chunk embeddings as CARVE and differ only
in how representations are selected or combined. See Appendix D.2 for implementation details.
Evaluation Metrics.We evaluate retrieval and generation separately. For retrieval, we report two
metrics:nDCG@ k, which rewards methods that rank the annotated evidence higher; andRecall@ k,
which measures whether the annotated evidence appears among the top- kretrieved chunks. Since
retrieval relies on deterministic embedding similarity, these metrics carry no run-to-run variance. For
generation, we report the aggregatepass rate, defined as the fraction of queries whose generated
answer is judged correct against the reference by an LLM-as-a-judge (Qwen3.6-35B-A3B), which
is necessary since answers are in open-ended form rather than multiple choice (see the prompt
in Figure 8). To assess generalization across generators, we evaluate each retrieval method with
three backbones: {Qwen3-VL-8B, Qwen3-VL-32B, and Gemma-4-26B}. We fix k= 5 throughout
experiments, except in Appendix F.4 where we varykto show thatCARVEconsistently outperforms
baselines. Detailed definitions of all metrics are provided in Appendix D.3.
7

Table 1: Retrieval and generation performance on V-RAGBench across system-, query-, and chunk-
level methods for VideoRAG. Retrieval is measured by Recall@5 and nDCG@5, while generation is
reported as pass rate under three generator backbones. Best values are in bold.
Retrieval Performance Generation Performance (w. Three Backbones)
Category Method Recall@5 nDCG@5 Qwen3-VL-8B Qwen3-VL-32B Gemma-4-26B
System-Level VideoRAG-A [28] 0.510 0.332 0.250 0.317 0.307
GQR [59] 0.503 0.340 0.167 0.110 0.153
Freeret [89] 0.263 0.187 0.225 0.200 0.160
GME [85] 0.413 0.253 0.260 0.287 0.215
Query-Level RRF [50] 0.463 0.298 0.180 0.130 0.187
DAT [22] 0.460 0.312 0.223 0.147 0.200
VideoRAG-B [42] 0.487 0.325 0.315 0.317 0.240
UniversalRAG [76] 0.447 0.298 0.293 0.247 0.257
UniversalRAG-LoRA [76] 0.470 0.311 0.237 0.270 0.220
Chunk-Level CARVE (Ours) 0.603 0.433 0.357 0.367 0.320
6.1 Main Result: Performance on Retrieval and Generation
Table 1 reports retrieval and generation performance on V-RAGBench , comparing CARVE against
four system-level and five query-level baselines. On retrieval, CARVE achieves the best Recall@5
(0.603 ) and nDCG@5 ( 0.433 ), exceeding the strongest baseline (VideoRAG-A: 0.510 at Recall@5;
GQR: 0.340 at nDCG@5). Notably, this retrieval gain consistently translates into stronger generation
across all three generator backbones. With Qwen3-VL-8B, CARVE reaches a pass rate of 0.357 ,
outperforming the strongest baseline (VideoRAG-B: 0.315 ); with Qwen3-VL-32B, CARVE again
leads at 0.367 over VideoRAG-A and VideoRAG-B (both at 0.317 ); and on a different model family,
Gemma-4-26B, CARVE retains the top score at 0.320 , ahead of VideoRAG-A (0.307). These results
confirm CARVE ’s chunk-level decision strengthens both stages: at retrieval, chunk-adaptive reranking
surfaces more accurate evidence chunks via each chunk’s winning configuration; at generation, the
same winning configuration propagates as a chunk-level interleaved evidence form that is consistently
better consumed across generator scales and families. Further analyses are in the appendix: a per-
source breakdown over Ego4D and EgoLife showing the same trend (Appendix F.2), and a case study
on retrieval-success but generation-failure queries (Appendix F.3).
Table 2: Ablation over modality–granularity config-
urations in CARVE ; performance improves as more
configurations are considered.
Configuration R@5 nDCG@5 Pass Rate Latency
m={text}, g={frame} 0.430 0.301 0.207 2.5s
m={text}, g={clip} 0.433 0.293 0.157 1.5s
m={vis}, g={frame} 0.477 0.336 0.323 5.2s
m={vis}, g={clip} 0.507 0.369 0.293 7.3s
m={text, vis}, g={frame} 0.513 0.352 0.303 5.0s
m={text, vis}, g={clip} 0.543 0.398 0.300 5.5s
m={text}, g={frame, clip} 0.567 0.413 0.217 3.3s
m={vis}, g={frame, clip} 0.497 0.355 0.327 6.1s
m={text, vis}, g={frame, clip} 0.603 0.433 0.357 4.6sA natural question is whether multiple configura-
tions genuinely synergize in CARVE , or whether
a single dominant one already captures most
of the gain. To answer this, we ablate CARVE
by varying which modality–granularity config-
urations are made available across both paral-
lel retrieval and chunk-adaptive reranking. Ta-
ble 2 reports the result, where the top block re-
stricts CARVE to a single configuration, the mid-
dle block expands along one axis while fixing
the other, and the bottom row uses all four. In
detail, the best single configuration is visual clip-
level at 0.507 Recall@5, the best partial combi-
nation is textual frame-and-clip at 0.567 , and the
full setting reaches 0.603 Recall@5, 0.433 nDCG@5, and 0.357 pass rate. Yet, in some cases, com-
bining configurations is not always beneficial: the visual frame-and-clip combination drops to 0.497
Recall@5, even below visual clip-level alone ( 0.507 ). This suggests expanding across modalities is a
more dependable source of synergy than expanding granularity within a single modality.
Importantly, this synergy comes at a reasonable latency cost. The full setting runs at 4.6s per query,
which is modestly above text-only configurations ( 1.5–3.3s) but actually faster than visual-only
ones ( 5.2–7.3s). This is because each chunk is rendered through its winning configuration, so visual
evidence is passed only for chunks that benefit from it, not every retrieved chunk.
6.2 In-depth Analysis of Interleaved Chunk Representations
We further analyze how CARVE ’s interleaved chunk representations work in practice from three angles:
the distribution of winning configurations across ranks, the reranking strategy itself, and a comparison
against trained query-level routers at the generation stage.
8

Table 3: Distribution of winning config-
uration over chunk-aware reranking.
Configuration Rank 1 Rank 2 Rank 3 Rank 4 Rank 5
{text, frame} 0.37 0.27 0.25 0.27 0.25
{text, clip} 0.15 0.13 0.11 0.11 0.13
{vis, frame} 0.24 0.26 0.28 0.27 0.28
{vis, clip} 0.24 0.34 0.36 0.35 0.34Distribution of Winning Configurations.Table 3 shows
the distribution of winning configurations across the top-5
ranked chunks. The distribution does not collapse onto
a single dominant configuration. At every rank, all four
configurations are selected with non-trivial frequency, with
{vis, clip }and{text, frame }most often winning, while
{text, clip }and{vis, frame }also contribute consistently.
This confirms that CARVE ’s interleaving is genuine. Different chunks are best surfaced through
different modality–granularity configurations, supporting our design choice of treating both axes as
chunk-level decisions.
Table 4: Reranking configuration choice
under CARVE w. Qwen3-VL-2B-Rerank.
Method R@5 nDCG@5 Latency
m={text}, g={frame} 0.497 0.343 0.8s
m={text}, g={clip} 0.490 0.339 0.5s
m={vis}, g={frame} 0.547 0.391 4.1s
m={vis}, g={clip} 0.560 0.396 8.0s
Random 0.513 0.312 3.1s
Concatenation 0.513 0.339 8.2s
CARVE(Ours) 0.603 0.433 3.4sReranking Configuration Strategy.We next examine
whether CARVE ’s reranking rule of rescoring each candi-
date under its retrieving configuration in Eqs. (4)–(5)is the
right strategy. Table 4 compares against three alternatives:
(i) applying a single fixed configuration to all candidates
regardless of how they were retrieved, (ii) assigning a con-
figuration randomly to each candidate (Random), and (iii)
feeding all four configurations jointly into the reranker
without selection (Concatenation). CARVE substantially
outperforms all alternatives. This shows that scoring each
chunk under its retrieving configuration tag, rather than
under a fixed or heuristic alternative, is what drives the gain. In particular, Concatenation fails to
match CARVE despite using all four configurations jointly, indicating that the chunk-level alignment
between retrieval and reranking matters more than access to more representations.
CARVE also achieves a favorable accuracy–latency trade-off. Reranking visual features is costly due to
the larger token count ( 4.1s and 8.0s for visual-only configurations), but by passing only each chunk’s
winning configuration, CARVE runs at 3.4s while still achieving the best retrieval, substantially faster
than visual-only and Concatenation ( 8.2s). The same trend for retrieval performance and latency
holds under a different reranker backbone (LamRA-Rank-7B), as seen in Appendix F.5.
Table 5: Comparison of CARVE against
query-level routing in generation.
Method Pass Rate Latency
m={text}, g={frame} 0.207 3.0s
m={text}, g={clip} 0.158 1.7s
m={vis}, g={frame} 0.323 4.4s
m={vis}, g={clip} 0.293 5.9s
Non-LLM Router w. Training 0.329 5.7s
LLM Router w. Training 0.310 5.3s
CARVE(Ours)0.3574.6sComparison against Query-Level Routing.We finally
examine whether CARVE ’s chunk-level interleaving of rep-
resentations fed to the generator is more effective than
rendering all retrieved chunks under a single, query-level
representation. Table 5 compares CARVE with two base-
lines: (i) single-fixed routing, which always renders chunks
under the same configuration regardless of the query, and
(ii) trained query-level routers, which select one config-
uration per query and apply it uniformly to all chunks
for generation. The latter includes a non-LLM router (an
MLP head over a frozen BGE-large-en-v1.5 encoder) and
an LLM router (Qwen3-VL-2B-Instruct fine-tuned with LoRA), both trained on the V-RAGBench
training split (see training details in Appendix D.2). CARVE outperforms all baselines despite using no
training, while the routers are fine-tuned on the V-RAGBench training split. This shows that no single
configuration suits every chunk in a query, a limitation CARVE sidesteps by varying the representation
chunk by chunk through the same reranking step, at competitive generation latency.
7 Conclusion
In this paper, we address two emerging gaps in VideoRAG . On the benchmark side, we introduce
V-RAGBench , a collection of ⟨query, evidence chunk, answer ⟩triplets over hour-scale egocentric
videos that jointly enforce non-recurring evidence, visual grounding, and evidence localization
for decoupled evaluation of retrieval and generation. On the methodological side, CARVE reframes
modality and temporal granularity as a chunk-level decision via parallel candidate pooling and chunk-
adaptive reranking, propagating each chunk’s winning configuration into generation. On V-RAGBench ,
CARVE outperforms eight recent baselines for VideoRAG on both stages; and on generation side, it
even surpasses the trained query-level routers without additional training, with winning configurations
distributing evenly across all configurations rather than collapsing onto a dominant one.
9

References
[1]Abdelrahman Abdallah, Mahmoud Abdalla, Bhawna Piryani, Jamshid Mozafari, Mohammed
Ali, and Adam Jatowt. Rankarena: A unified platform for evaluating retrieval, reranking and
rag with human and llm feedback. InCIKM, 2025.
[2]Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. A survey on rag
with llms.Procedia Computer Science, 246:3781–3790, 2024.
[3]David Arthur, Sergei Vassilvitskii, et al. k-means++: The advantages of careful seeding. In
Soda, volume 7, pages 1027–1035, 2007.
[4]Hiteshwar Kumar Azad and Akshay Deepak. Query expansion techniques for information
retrieval: a survey.Information Processing & Management, 56(5):1698–1735, 2019.
[5]Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao
Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report.arXiv preprint
arXiv:2511.21631, 2025.
[6]Marius Bock, Hilde Kuehne, Kristof Van Laerhoven, and Michael Moeller. Wear: An outdoor
sports dataset for wearable and egocentric activity recognition.IMWUT, 2024.
[7]Mustafa Chasmai, Gauri Jagatap, Gouthaman KV , Grant Van Horn, Subhransu Maji, and
Andrea Fanelli. Moment sampling in video llms for long-form video qa.arXiv preprint
arXiv:2507.00033, 2025.
[8]Boyu Chen, Zhengrong Yue, Siran Chen, Zikang Wang, Yang Liu, Peng Li, and Yali Wang.
Lvagent: Long video understanding by multi-round dynamical collaboration of mllm agents. In
ICCV, 2025.
[9]Guo Chen, Yicheng Liu, Yifei Huang, Yuping He, Baoqi Pei, Jilan Xu, Yali Wang, Tong Lu,
and Limin Wang. Cg-bench: Clue-grounded question answering benchmark for long video
understanding.arXiv preprint arXiv:2412.12075, 2024.
[10] Guo Chen, Lidong Lu, Yicheng Liu, Liangrui Dong, Lidong Zou, Jixin Lv, Zhenquan Li, Xinyi
Mao, Baoqi Pei, Shihao Wang, et al. Towards multimodal lifelong understanding: A dataset and
agentic baseline.arXiv preprint arXiv:2603.05484, 2026.
[11] Lin Chen, Xilin Wei, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong
Duan, Bin Lin, Zhenyu Tang, et al. Sharegpt4video: Improving video understanding and
generation with better captions. InNeurIPS, 2024.
[12] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu,
Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-temporal modeling
and audio understanding in video-llms.arXiv preprint arXiv:2406.07476, 2024.
[13] Jeonghwan Choi, Minjeong Ban, Minseok Kim, and Hwanjun Song. Word2passage: Word-level
importance re-weighting for query expansion. InACL, 2025.
[14] Jacob Cohen. A coefficient of agreement for nominal scales.Educational and Psychological
Measurement, 20(1):37–46, 1960.
[15] Shangzhe Di and Weidi Xie. Grounded question-answering in long egocentric videos. InCVPR,
2024.
[16] Aradhya Dixit and Tianxi Liang. Semantic event graphs for long-form video question answering.
arXiv preprint arXiv:2601.06097, 2026.
[17] Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. Splade: Sparse lexical and
expansion model for first stage ranking. InSIGIR, 2021.
[18] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang,
Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive
evaluation benchmark of multi-modal llms in video analysis. InCVPR, 2025.
10

[19] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit
Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world
in 3,000 hours of egocentric video. InCVPR, 2022.
[20] Yongxin Guo, Jingyu Liu, Mingda Li, Dingxin Cheng, Xiaoying Tang, Dianbo Sui, Qingbin
Liu, Xi Chen, and Kevin Zhao. Vtg-llm: Integrating timestamp knowledge into video llms for
enhanced video temporal grounding. InAAAI, 2025.
[21] Gyuwon Han, Young Kyun Jang, and Chanho Eom. Cova: Text-guided composed video retrieval
for audio-visual content.arXiv preprint arXiv:2601.22508, 2026.
[22] Hsin-Ling Hsu and Jengnan Tzeng. Dat: Dynamic alpha tuning for hybrid retrieval in retrieval-
augmented generation.arXiv preprint arXiv:2503.23013, 2025.
[23] Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak
Shah, Raffay Hamid, Bing Yin, and Trishul Chilimbi. M-llm based video frame selection for
efficient video understanding. InCVPR, 2025.
[24] Xiaohu Huang, Hao Zhou, and Kai Han. Prunevid: Visual token pruning for efficient video
large language models. InACL, 2025.
[25] Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, and
Gedas Bertasius. Video recap: Recursive captioning of hour-long videos.arXiv preprint
arXiv:2402.13250, 2024.
[26] Md Mohaiminul Islam, Ngan Ho, Xitong Yang, Tushar Nagarajan, Lorenzo Torresani, and
Gedas Bertasius. Ravu: Retrieval augmented video understanding with compositional reasoning
over graph.arXiv preprint arXiv:2505.03173, 2025.
[27] Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward
spatio-temporal reasoning in visual question answering. InCVPR, 2017.
[28] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju Hwang. Videorag: Retrieval-
augmented generation over video corpus. InACL, 2025.
[29] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming
Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InEMNLP,
2023.
[30] Kumara Kahatapitiya, Kanchana Ranasinghe, Jongwoo Park, and Michael S Ryoo. Language
repository for long video understanding. InACL, 2025.
[31] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov,
Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering.
InEMNLP, 2020.
[32] Min Jung Lee, Dayoung Gong, and Minsu Cho. Video summarization with large language
models. InCVPR, 2025.
[33] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg. Tvqa: Localized, compositional video
question answering. InEMNLP, 2018.
[34] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. InNeurIPS, 2020.
[35] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin
Wang, and Yu Qiao. Videochat: Chat-centric video understanding.Science China Information
Sciences, 2025.
[36] Xiaonan Li, Changtai Zhu, Linyang Li, Zhangyue Yin, Tianxiang Sun, and Xipeng Qiu. Lla-
trieval: Llm-verified retrieval for verifiable generation. InNAACL, 2024.
11

[37] Geuntaek Lim, Minho Shim, Sungjune Park, Jaeyun Lee, Inwoong Lee, Taeoh Kim, Dongyoon
Wee, and Yukyung Choi. Video-oasis: Rethinking evaluation of video understanding.arXiv
preprint arXiv:2603.29616, 2026.
[38] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava:
Learning united visual representation by alignment before projection. InEMNLP, 2024.
[39] Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and
Wei Ping. Mm-embed: Universal multimodal retrieval with multimodal llms.arXiv preprint
arXiv:2411.02571, 2024.
[40] Yijie Lin, Jie Zhang, Zhenyu Huang, Jia Liu, Zujie Wen, and Xi Peng. Multi-granularity
correspondence learning from long-term noisy videos. InICLR, 2024.
[41] Hui Liu and Xiaojun Wan. Video paragraph captioning as a text summarization task. InACL,
2021.
[42] Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin, Haojia Lin, Chaoyou Fu, Jinfa Huang,
Jiayi Ji, Fei Chao, Jiebo Luo, et al. Video-rag: Visually-aligned retrieval-augmented long video
comprehension. InNeurIPS, 2025.
[43] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. Query rewriting in retrieval-
augmented large language models. InEMNLP, 2023.
[44] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, and Jianfei Cai.
Drvideo: Document retrieval based long video understanding. InCVPR, 2025.
[45] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards
detailed video understanding via large vision and language models. InACL, 2024.
[46] Priyanka Mandikal and Raymond Mooney. Sparse meets dense: A hybrid approach to enhance
scientific document retrieval.arXiv preprint arXiv:2401.04055, 2024.
[47] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic
benchmark for very long-form video language understanding. InNeurIPS, 2023.
[48] Alkesh Patel, Vibhav Chitalia, and Yinfei Yang. Advancing egocentric video question answering
with multimodal large language models.arXiv preprint arXiv:2504.04550, 2025.
[49] Danila Potapov, Matthijs Douze, Zaid Harchaoui, and Cordelia Schmid. Category-specific
video summarization. InECCV, 2014.
[50] Zackary Rackauckas. Rag-fusion: a new take on retrieval-augmented generation.arXiv preprint
arXiv:2402.03367, 2024.
[51] Aniket Rege, Arka Sadhu, Yuliang Li, Kejie Li, Ramya Korlakai Vinayak, Yuning Chai, Yong Jae
Lee, and Hyo Jin Kim. Agentic very long video understanding.arXiv preprint arXiv:2601.18157,
2026.
[52] Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive
multimodal large language model for long video understanding. InCVPR, 2024.
[53] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag:
Retrieval-augmented generation with extreme long-context videos. InKDD, 2026.
[54] Kunal Sawarkar, Abhilasha Mangal, and Shivam Raj Solanki. Blended rag: Improving rag
(retriever-augmented generation) accuracy with semantic search and hybrid query-based retriev-
ers. InMIPR, 2024.
[55] Xiaoqian Shen, Wenxuan Zhang, Jun Chen, and Mohamed Elhoseiny. Vgent: Graph-based
retrieval-reasoning-augmented generation for long video understanding. InNeurIPS, 2025.
[56] Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented
generation: A survey on agentic rag.arXiv preprint arXiv:2501.09136, 2025.
12

[57] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training
techniques for clip at scale.arXiv preprint arXiv:2303.15389, 2023.
[58] Shitong Sun, Ke Han, Yukai Huang, Weitong Cai, and Jifei Song. Egograph: Temporal knowl-
edge graph for egocentric video understanding.arXiv preprint arXiv:2602.23709, 2026.
[59] Omri Uzan, Asaf Yehudai, Eyal Shnarch, Ariel Gera, et al. Guided query refinement: Multimodal
hybrid retrieval with test-time optimization. 2026.
[60] Han Wang, Yuxiang Nie, Yongjie Ye, Yanjie Wang, Shuai Li, Haiyang Yu, Jinghui Lu, and Can
Huang. Dynamic-vlm: Simple dynamic visual token compression for videollm. InICCV, 2025.
[61] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing
Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception
of the world at any resolution.arXiv preprint arXiv:2409.12191, 2024.
[62] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form
video understanding with large language model as agent. InECCV, 2024.
[63] Yujing Wang, Hainan Zhang, Liang Pang, Binghui Guo, Hongwei Zheng, and Zhiming Zheng.
Maferw: Query rewriting with multi-aspect feedbacks for retrieval-augmented large language
models. InAAAI, 2025.
[64] Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi
Lei, Yuqing Ding, and Han Li. Infogain-rag: Boosting retrieval-augmented generation through
document information gain-based reranking and filtering. InEMNLP, 2025.
[65] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius,
and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on
long videos. InCVPR, 2025.
[66] Ziyang Wang, Honglu Zhou, Shijie Wang, Junnan Li, Caiming Xiong, Silvio Savarese, Mohit
Bansal, Michael S Ryoo, and Juan Carlos Niebles. Active video perception: Iterative evidence
seeking for agentic long video understanding.arXiv preprint arXiv:2512.05774, 2025.
[67] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for
long-context interleaved video-language understanding. InNeurIPS, 2024.
[68] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. C-pack: Packaged resources
to advance general chinese embedding, 2023.
[69] Dejing Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang.
Video question answering via gradually refined attention over appearance and motion. InMM,
2017.
[70] Zeyu Xu, Junkang Zhang, Qiang Wang, and Yi Liu. E-vrag: Enhancing long video understanding
with resource-efficient retrieval augmented generation.arXiv preprint arXiv:2508.01546, 2025.
[71] Zhucun Xue, Jiangning Zhang, Xurong Xie, Yuxuan Cai, Yong Liu, Xiangtai Li, and Dacheng
Tao. Adavideorag: Omni-contextual adaptive retrieval-augmented efficient long video under-
standing.arXiv preprint arXiv:2506.13589, 2025.
[72] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented
generation.arXiv preprint arXiv:2401.15884, 2024.
[73] Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang,
Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, et al. Egolife: Towards egocentric life
assistant. InCVPR, 2025.
[74] Songyuan Yang, Weijiang Yu, Ziyu Liu, Guijian Tang, Wenjing Yang, Huibin Tan, and Nong
Xiao. Graph-to-frame rag: Visual-space knowledge fusion for training-free and auditable video
reasoning.arXiv preprint arXiv:2604.04372, 2026.
[75] Linli Yao, Haoning Wu, Kun Ouyang, Yuanxing Zhang, Caiming Xiong, Bei Chen, Xu Sun,
and Junnan Li. Generative frame sampler for long video understanding. InACL, 2025.
13

[76] Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jinheon Baek, and Sung Ju Hwang. Univer-
salrag: Retrieval-augmented generation over corpora of diverse modalities and granularities.
2026.
[77] Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, and Sung Ju Hwang. Worldmm: Dynamic
multimodal memory agent for long video reasoning. InCVPR, 2026.
[78] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi,
and Bryan Catanzaro. Rankrag: Unifying context ranking with retrieval-augmented generation
in llms. InNeurIPS, 2024.
[79] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao.
Activitynet-qa: A dataset for understanding complex web videos via question answering. In
AAAI, 2019.
[80] Biao Zhang, Paul Suganthan, Gaël Liu, Ilya Philippov, Sahil Dua, Ben Hora, Kat Black, Gus
Martins, Omar Sanseviero, Shreya Pathak, et al. T5gemma 2: Seeing, reading, and understanding
longer.arXiv preprint arXiv:2512.14856, 2025.
[81] Ce Zhang, Taixi Lu, Md Mohaiminul Islam, Ziyang Wang, Shoubin Yu, Mohit Bansal, and
Gedas Bertasius. A simple llm framework for long-range video question-answering. InEMNLP,
2024.
[82] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language
model for video understanding. InEMNLP, 2023.
[83] Le Zhang, Yihong Wu, Qian Yang, and Jian-Yun Nie. Exploring the best practices of query
expansion with large language models. InEMNLP, 2024.
[84] Xiaoyi Zhang, Zhaoyang Jia, Zongyu Guo, Jiahao Li, Bin Li, Houqiang Li, and Yan Lu. Deep
video discovery: Agentic search with tool use for long-form video understanding.arXiv preprint
arXiv:2505.18079, 2025.
[85] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie,
Meishan Zhang, Wenjie Li, and Min Zhang. Gme: improving universal multimodal retrieval by
multimodal llms. 2025.
[86] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. Retrieval-augmented generation for ai-generated
content: A survey.Data Science and Engineering, pages 1–29, 2026.
[87] Yixuan Zhao, Zihao Fan, Yingying Cao, Zhengjia Lyu, and Jingyuan Li. Rewritegen: Au-
tonomous query optimization for retrieval-augmented large language models via reinforcement
learning.Electronics, 15(5):1058, 2026.
[88] Wenqi Zhou, Kai Cao, Hao Zheng, Yunze Liu, Xinyi Zheng, Miao Liu, Per Ola Kristensson,
Walterio Mayol-Cuevas, Fan Zhang, Weizhe Lin, et al. X-lebench: A benchmark for extremely
long egocentric video understanding.arXiv preprint arXiv:2501.06835, 2025.
[89] Yuhan Zhu, Xiangyu Zeng, Chenting Wang, Xinhao Li, Yicheng Xu, Ziang Yan, Yi Wang, and
Limin Wang. Freeret: Mllms as training-free retrievers. 2026.
14

A Limitations
We outline three limitations of this work, which also point to natural directions for future research.
Scope of modality and granularity axes.We instantiate the chunk-level configuration space along
two axes—modality (visual vs. textual) and granularity (frame vs. clip)—yielding four configura-
tions. This covers the most universally available content-based representations and is sufficient to
demonstrate that chunk-level decisions outperform query-level ones. By design, CARVE is agnostic to
the size of the configuration space: parallel candidate pooling and chunk-adaptive reranking extend
naturally to additional modalities and finer granularities without architectural changes, and we leave
a systematic study of this scaling behavior to future work.
One-shot retrieve-then-generate pipeline.We deliberately keep the standard one-shot retrieve-
then-generate structure intact to isolate the contribution of chunk-level configuration decisions and
enable clean comparison against system- and query-level baselines. This means we do not explore
iterative or agentic retrieval loops, where evidence could be progressively refined across multiple
turns. CARVE ’s chunk-level winning configurations are a natural per-step signal for such loops, and
integrating chunk-adaptive reranking into agenticVideoRAGframeworks is a promising avenue.
Egocentric video as the primary evaluation domain. V-RAGBench is deliberately built on ego-
centric video from Ego4D [ 19] and EgoLife [ 73], since first-person streams are the most realistic
and challenging target for VideoRAG . This focused scope is a strength for studying the core re-
trieval problem, but it does not directly evaluate domains such as instructional videos, films, or
third-person broadcasts, where narrative structure or production cues may shift the balance among
modality–granularity configurations. Since the three benchmark properties and the CARVE pipeline
are domain-agnostic, extending both to other domains is a promising direction.
B Societal Impact
We do not foresee specific societal risks arising directly from the proposed method or benchmark
beyond those generally associated with multimodal retrieval and video understanding research. Never-
theless, we acknowledge that egocentric video inherently captures personal, daily-life activities, and
any downstream deployment of VideoRAG systems should consider privacy, consent, and responsible
use of personal video data.
For the human evaluation components described in Appendix C.5 and Appendix D.4, we recruited
graduate-student annotators to verify the reliability of our filtering pipeline and the LLM-as-a-judge
evaluation. All annotators were compensated above the U.S. federal minimum wage, were informed
of the task scope and purpose in advance, and participated voluntarily. No personally identifiable
information about the annotators is disclosed in the dataset, paper, or supplementary materials, and
their contributions are used solely for the validation analyses reported in this work.
C Dataset
The overall pipeline ofV-RAGBenchconstruction is described in Figure 2.
C.1 Dataset statistics
V-RAGBench , is a benchmark for egocentric video understanding constructed from two comple-
mentary sources: long-form daily-life recordings from EgoLife and diverse activity-centric videos
from Ego4D. The dataset comprises 2,100 queries grounded in 216 uncut source videos. Detailed
dataset statistics, including the distribution of queries across source datasets, categories, domains,
and evidence chunk durations, are summarized in Table 6.
C.2 Definitions of the source video domains
Ego4D videos cover heterogeneous activities, while EgoLife videos are uniformly daily-life. We
group Ego4D videos into five high-level domains —Domestic,Social,Stationary,Manual, and
Outdoor— by collapsing the Ego4D metadata categories. For each in-distribution domain we sample
36training and 6test videos based on the within-domain length distribution; theOutdoordomain is
15

Stage 1: Source Video Selection
EgoLife
 Ego4D
>1hr
Stratified Sampling 
by Video Length
216 Long Egocentric Videos
Stage 2: Event Extraction & Deduplication
KMeans  ++
ClusteringSemantic 
DeduplicationSemantic Deduplication
Distinct Event ChunksTemporal Segmentation
(CLIP -Based Vision Encoder)
Kernel Temporal SegmentationStage 3: Candidate QA Generation
AI generating 
questionsQuery Category 1
Query Category 2
Query Category 3⋯⋯[𝑡𝑠,𝑡𝑒]
⋯⋯
⋯⋯
Contextual 
LocalizationQuery Category 1
Query Category 2
Query Category 3⋯⋯
⋯⋯
⋯⋯
Query Category 1
Query Category 2
Query Category 3⋯⋯
⋯⋯
⋯⋯
Candidate Queries
< Query, Evidence Timeframe, Open Answer >Stage 4: Query Filtering
F1) Query Semantic 
Similarity
F2) Answerability
F3) Shortcut Bias
F4) Empirical 
Answerability
F5) Evidence 
Uniqueness
2,100 High -Quality,
Uniquely Grounded Queries𝑞1 𝑞2≈
Is there answer?
Still 
answerable?
⋯ ⋯
Exclusive Evidence?Figure 2: Overview of the V-RAGBench Construction Pipeline. We construct V-RAGBench through
four stages: source video selection, event extraction and deduplication, candidate QA generation,
and query filtering. Hour-scale egocentric videos are segmented into distinct event chunks, used
to generate open-ended QA candidates, and then rigorously filtered for redundancy, answerability,
shortcut bias, empirical answerability, and evidence uniqueness, yielding a balanced set of uniquely
grounded queries.
Table 6: Dataset statistics detailing the distribution of 2,100 queries across categories, domains, and
chunk durations, alongside 216 uncut source videos.
Scope Attribute Sub-category Count (% / Details)
Source VideosTotal 216
SourceEgoLife 42 (Mean: 379.1 min,
Range: 177.8–552.7 min)
Ego4D 174 (Mean: 86.0 min,
Range: 60.0–423.6 min)
QueriesTotal 2,100
SourceEgoLife 1,320 (62.86%)
Ego4D 780 (37.14%)
CategoryAction Flow & Movement 700 (600 Train / 100 Test)
Interaction 700 (600 Train / 100 Test)
Object-Centric
Visual Understanding700 (600 Train / 100 Test)
DomainDaily 1,320 (62.86%)
Domestic 203 (9.67%)
Social 195 (9.29%)
Stationary 180 (8.57%)
Manual 169 (8.05%)
Outdoor 33 (1.57%)
Evidence
Chunk Duration<5s 330 (15.71%)
5–30s 1,350 (64.29%)
30–60s 168 (8.00%)
1–5 min 252 (12.00%)
held out entirely for test, providing an out-of-domain split. For EgoLife, one of the seven days per
participant is sampled as test, with non-overlapping day assignments across participants to prevent
leakage. The resulting test set spans both familiar and unseen domains, supporting both standard and
out-of-distribution evaluation. Formal definitions of these domains are provided in Table 7.
C.3 Definitions of the query categories
The 2,100 queries are organized into three reasoning categories that target distinct capabilities of
egocentric video understanding:Action Flow & Movement, which probes temporal and sequential
reasoning;Object-Centric Visual Understanding, which focuses on fine-grained recognition of
object attributes and states; andInteraction, which examines both physical manipulation and social
engagement. The precise definitions of each category are given in Table 8.
16

Table 7: Egocentric video domains covered inV-RAGBench
Source Dataset Domain Definition
EgoLife Daily Long-form egocentric daily-life videos from EgoLife, covering extended personal routines across
time. Unlike Domestic, which is a specific Ego4D activity domain, Daily refers to broader continuous
life-logging episodes that may include household routines, desk work, leisure, transitions, and other
ordinary activities across a day.
Ego4DDomestic Videos of household or home-centered everyday activities, such as cooking, cleaning, organizing,
eating, or moving around indoor living spaces. The key characteristic is routine domestic activity
within a home-like environment.
Manual Videos centered on hands-on physical manipulation, craftwork, repair, assembly, tool use, or labor-
intensive activities. The key characteristic is that the camera wearer is actively manipulating objects
or materials.
Social Videos involving interpersonal interaction, such as conversation, collaboration, instruction, assistance,
or shared activity with other people. The key characteristic is that understanding the scene depends on
social context, other people’s actions, or communicative behavior.
Stationary Low-motion egocentric videos where the camera wearer remains largely fixed in place and attends
to a stable target over time, such as a computer screen, television, desk, document, or workspace.
The key characteristic is not the presence of a screen itself, but the static first-person viewpoint and
sustained attention to the same visual area.
Outdoor Videos captured in outdoor or large-scale physical environments, often involving walking, navigation,
travel, sports, outdoor work, or other movement-heavy activities. The key characteristic is that the
camera wearer interacts with a changing external environment rather than a fixed indoor workspace or
household setting.
Table 8: Definitions of the query categories used in the dataset.
Category Description
Action Flow
& MovementQuestions requiring understanding of event sequences, movement, and scene transitions. These questions
emphasizes temporal flow before, during, and after key actions. The expected answer must describe a
temporal or sequential fact.
Object-Centric
Visual UnderstandingQuestions that require recognizing specific details about objects: identity, visual attributes (color, shape,
size), current state, or count. The expected answer must identify a specific object detail.
Interaction Questions about the camera wearer’s interaction with their environment, encompassing both the physical
manipulation of objects (how things are held, used, or transferred) and social engagement (roles,
relationships, and interpersonal behaviors). The expected answer must describe a physical interaction
with an object or a social fact involving other people.
C.4 Event selection and de-duplication details
Event Chunk Construction Details.Section 4 describes the overall event extraction and dedupli-
cation procedure. Here we clarify two implementation consequences of this design. First, temporal
segmentation and semantic deduplication play separate roles: segmentation determines valid time-
contiguous event candidates, while deduplication only selects among those candidates and does not
modify their temporal boundaries. Thus, each retained chunk remains an original contiguous video
interval.
Second, all event-level statistics are computed after deduplication. Since the retained chunks are
selected as representatives of semantically similar segment groups, the final evidence pool is inten-
tionally smaller and less redundant than the raw segmentation output. In particular, repeated daily
routines are counted only through their selected representative chunks, which reduces the chance that
a generated query can be answered from multiple near-duplicate events in the same video.
Because the representatives are not forced into a fixed temporal length, the resulting chunks naturally
span a range of durations. We report the duration distribution of these deduplicated event chunks in
Table 9 and Table 10.
Table 9: Descriptive statistics of event chunk durations (in seconds) by dataset.
Split Count Mean (s) Median (s) Min (s) Max (s)
All (Overall) 19,751 28.03 8.60 4.70 300.00
EgoLife 11,109 31.65 7.60 4.70 300.00
Ego4D 8,642 23.39 11.00 5.00 300.00
17

Table 10: Distribution of event chunk durations scaled to the final subset of 2,100 queries.
Duration Bin (s) Count Percentage
0–5 331 15.78%
5–10 810 38.59%
10–30 594 28.27%
30–60 167 7.97%
60–120 77 3.66%
120–300 116 5.51%
300–500 5 0.22%
Total 2,100 100.00%
C.5 QA Filter Details
Table 11 shows that the filtering pipeline is highly selective: only 5,907 of the initial 67,370 can-
didates remain after all five stages, corresponding to 8.77% of the original candidate pool. This
low retention rate reflects the strict grounding requirements of V-RAGBench : retained queries must
be semantically non-redundant, answerable from the annotated evidence, robust to shortcut bias,
empirically answerable by a strong VLLM, and not recoverable from non-evidence chunks in the
same video. The 5,907 retained queries form the post-filter candidate pool before category balancing.
Since the three query categories are unevenly populated after filtering, we balance the benchmark by
downsampling each category to the size of the smallest category, which isInteraction(See Table 8).
We therefore randomly sample 700queries per category, yielding the final balanced set of 2,100
queries.
Table 11: Filtering statistics for the combined dataset. The table reports the number of queries
remaining and the number filtered at each major post-hoc filtering stage. Percentages are relative to
the initial candidate count.
Filtering Stage Remaining Filtered (%)
Initial Candidates 67,370 –
Semantic Similarity 52,783 14,587 (21.65%)
Shortcut Bias 28,331 24,452 (36.30%)
Answerability 19,751 8,580 (12.74%)
Empirical Answerability 11,041 8,710 (12.93%)
Evidence Uniqueness 5,907 5,134 (7.62%)
Final 5,907 (8.77% kept)
Semantic Similarity Filter.Event-level visual deduplication does not fully eliminate redundancy
at the query level, since different retained events can still induce near-identical natural-language
questions. We therefore apply this filter over query embeddings rather than event embeddings, remov-
ing candidates that are semantically too close to already retained queries. This reduces benchmark
inflation from repeated phrasings of similar actions.
Answerability Filter.For each candidate, we evaluate the paired (q,event clip) with GPT-5.2-
chat and ask whether the answer can be determined from the visual evidence. Candidates judged
unanswerable from their source clip are discarded.
Shortcut Bias Filter.Some generated queries can be answered from language priors or commonsense
knowledge without inspecting the video. To identify such cases, we provide GPT-5.2-chat with the
query alone, without any visual context, and compare its answer against the reference answer. If
the blind answer matches the reference, the query is discarded as shortcut-biased. Table 12 shows
examples of queries filtered as they exhibit shortcut bias.
Empirical Answerability Filter.After the previous filters, we further check whether the query is
practically answerable by a strong VLLM using only the source evidence chunk. We run Gemini 2.5-
flash on the paired clip and evaluate the generated answer with the binary LLM-as-a-judge described
in Appendix D.4. Queries that the model fails to answer correctly are removed, since they are unlikely
to be reliably answerable in downstream evaluation.
Evidence Uniqueness Filter.To test whether the annotated evidence is uniquely sufficient, we
simulate retrieval over non-evidence clips from the same source video. Each video is split into
18

2-minute chunks, which are embedded with Qwen3-VL-8B. For every query, we retrieve the top- 10
most similar chunks while explicitly excluding the chunk containing the true evidence. Gemini 2.5-
flash is then asked to answer using only this non-evidence retrieval set, and the output is judged with
the same binary LLM-as-a-judge as in the empirical answerability filter. If the answer is still judged
correct, the query is removed because its answer is recoverable from elsewhere in the video rather
than uniquely grounded in the annotated evidence. Figure 3 presents the case study of queries whose
evidence is not uniquely grounded in a single evidence chunk.
Query 1:When everyone is gathered in the living room and talking in front of the whiteboard, what is the man in the pink shirt doing?Answer: The man in the pink shirt is watching the others as they discuss something.
Ground Truth ChunkSpuriously grounded chunk #1Spuriously grounded chunk#2SpuriousEvidenceSpuriousEvidenceTrue Evidence
Query 2:What color is the watch worn by the man in the orange shirt?Answer: The watch worn by the man in the orange shirt is black.
Ground Truth ChunkSpuriously grounded chunk #1Spuriously grounded chunk#2
SpuriousEvidenceSpuriousEvidenceTrue Evidence
Query 3:When the person is sitting outside at the wooden table, what object are they holding?Answer: The person is holding a transparent sheet with pink flower stickers on it.
Ground Truth ChunkSpuriously grounded chunk #1Spuriously grounded chunk#2
SpuriousEvidenceSpuriousEvidenceTrue Evidence
Figure 3: Examples of the queries not grounded with a unique evidence chunk.
Human Verification.To further validate the effectiveness of our answerability filters, we conduct
human evaluation on the retained queries. Specifically, we recruit two graduate student annotators
19

Table 12: Examples of benchmark video queries with shortcut bias, which are answerablewithout
access to the ground-truth source video. GPT-5.2-Chat resolved each query correctly from the question
text alone, via common sense, contextual cues, or parametric knowledge.
Q1: While I am standing at the wooden island with the carton of milk and metal mixing bowls, what is the color of the liquid inside
the clear plastic measuring cup?
Blind prediction:White.GT:White.
Why answerable:The question explicitly mentions a carton of milk on the same surface as the measuring cup. Common sense strongly
suggests the liquid being measured is the milk, making “white” inferable without watching the video.
Q2: What visible item indicates the woman’s professional role while she sits and talks to me?
Blind prediction:A visible name badge or ID badge.GT:A badge.
Why answerable:The phrase “visible item indicates professional role” is a near-defining description of a name badge or ID badge. Parametric
knowledge of workplace conventions makes this the default answer regardless of video content.
Q3: Upon approaching the man sitting at the white table, how does the person acknowledge my presence?
Blind prediction:He looks up and greets you (e.g., says hello or nods).GT:By greeting.
Why answerable:Standard social conventions dictate that acknowledging an approaching person involves looking up and greeting them.
The query frames the action as “acknowledge,” which already implies a greeting-style response.
and sample 577QA pairs at random (corresponding to a 95% confidence interval with a margin of
±4over the full set of 2,100 queries). To ensure annotation quality, we only recruit annotators who
have at least one first-author publication at a top-tier AI conference and possess English proficiency
equivalent to CEFR C2. Prior to annotation, each annotator received a 30-minute instructional briefing
on the task. Each annotator is then provided with the source video clip, the query, and the ground-truth
answer, and is asked to judge whether the answer can be reasonably inferred from the video.
We find that 96.9% of the sampled queries are judged as answerable by both annotators. The inter-
annotator agreement (IAA) is 0.63 (Cohen’s Kappa [ 14]), indicating substantial agreement. These
results further support that the retained queries are reliably answerable from their associated evidence.
D Implementation details
Model Specification.Table 13 lists all open-source models used in our experiments, while Table 14
summarizes the proprietary models. The total API cost incurred in our experiments is approximately
$1,200, including both QA generation and QA filtering.
Resources.All experiments run on8×NVIDIA H200 (141GB) GPUs with CUDA13.0.
Table 13: Pretrained models used in our pipeline.
Model Checkpoint Purpose
Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-8B-Instruct Video QA, VLM for construct Text Clip and Text
Frames in Section 3
Qwen3-VL-Embedding-32B Qwen/Qwen3-VL-32B-Instruct Video QA
Gemma-4-26B-A4B google/gemma-4-26B-A4B Video QA
Clip-vit-large-patch14 openai/clip-vit-large-patch14 Frame embedding extraction in Section 4
Qwen3-6-35B-A3B Qwen/Qwen3-6-35B-A3B LLM as Judge in Section 6
Qwen3-VL-Embedding-2B Qwen/Qwen3-VL-Embedding-2B Embedding model in Section 5
Qwen3-VL-Reranker-2B Qwen/Qwen3-VL-Reranker-2B Reranking in Section 5
LamRA-Rank-7B code-kunkun/LamRA-Rank Reranking in Section 5
Qwen3-VL-Embedding-8B Qwen/Qwen3-VL-Embedding-8B Semantic similarity check in Section 4
Table 14: Proprietary API used.
Model Purpose
gemini-3-flash-preview QA generation in Section 4
gemini-2.5-flash Uniqueness, empirical answerability filtering in Section 4
gpt-5.2-chat-latest Answerability, Shortcut bias filter in Section 4
20

D.1CARVEImplementations.
Representation Construction.We split each video into non-overlapping 120-second segments and
build four representations per segment, indexed identically across representations. The VF encodes 5
keyframes per segment; the VC encodes up to 64frames per segment (maximum 1,843,200 pixels
per frame, sampled at 1fps). The TF view captions the 5keyframes with Qwen3-VL-8B-Instruct [ 5]
up to512tokens; the TC view captions per-second frames at1fps.
Retrieval and Generation.At inference, we retrieve the top- K=5 chunks per query, with reranking
using the default Qwen3-VL-Reranker-2B [ 5]. The retrieved chunks are passed to Qwen3-VL-8B-
Instruct with a maximum of 512new tokens under its default decoding setup. Every generated answer
is judged against the reference by a binary judge (Qwen3.6-35B-A3B), using identical judge weights
and prompt across our method and all baselines.
D.2 Baseline Implementations
System-level Baselines.VideoRAG-A[ 28] is a retrieval method that encodes each video segment by
linearly interpolating its textual and visual representations. We set αas 0.5 in this set up.GQR[ 59]
refines the retriever’s query embedding by gradient descent against a mixture of the two retriever’s
score distributions, using KL minimization. We set two retrievers as text and video retrievers with
learning rate 5e−4 .FreeRet[ 89] repurposes an MLLM (Qwen2-VL-2B-Instruct [ 61] in our setup)
as a training-free two-stage retriever, extracting segment embeddings from the last self-attention
sublayer and reranking the candidates via MCQ-based scoring.GME[ 85] is a unified multimodal
embedding model based on Qwen2-VL-2B-Instruct [ 61]. We apply both methods directly to the raw
video segments and query.
Query-level Baselines.VideoRAG-B[ 42] prompts Qwen3-VL-8B-Instruct [ 5] to decouple each
query into three memory-specific sub-queries, retrieves from each memory, and merges the results
before passing them to the generator. Following the original paper, we use one visual retriever and two
text retrievers.RRF[ 50] is the RAG-Fusion variant of reciprocal rank fusion: the MLLM (Qwen3-VL-
8B-Instruct [ 5]) rewrites each query into two modality-targeted paraphrases — one for text-clip, one
for text-frames — which drive two retrievers operating on the text-clip and text-frame embeddings,
respectively. The resulting ranked lists are aggregated by the standard RRF formula with k=60 .
DAT[ 22] adaptively balances two retrievers per query using LLM. We set the paper’s dense and
sparse setting with text clip and frames, αthat interpolates the min-max-normalized similarities over
the candidate union.UniversalRAG[ 76] routes each query to a single optimal (m, g) configuration
using a trained classifier and applies the selected configuration uniformly to all retrieved chunks. We
implement two variants-an LLM-based router (Qwen3-VL-2B-Instruct fine-tuned with LoRA r=32 ,
BCE multi-hot supervision) and a lightweight encoder-based router (T5Gemma 2 270M [ 80] with a
linear classification head) — and replace the paper’s sigmoid-thresholded decoding with argmax at
inference for consistency with the top-Kretrieval setting used by all other baselines.
Router Implementations.To evaluate the benefits of chunk-level decisions, we compare CARVE
against two types of query-level routers that select a single configuration (m, g) per query. To train
LLM-based Router, following the training recipe of UniversalRAG [ 76], we fine-tune Qwen3-VL-2B-
Instruct [ 5] as a multi-label classifier over the four configurations. We apply LoRA ( r= 32 ) and train
for 5 epochs with a 2e−5 learning rate. Labels are derived from our train set, where a configuration
is labeled positive if it produces a correct answer. At inference, we select the configuration with the
highest predicted score (argmax) and apply it uniformly to all chunks for both retrieval and generation.
To train encoder based router, we train an MLP-based classifier on top of a frozen BGE-large-en-
v1.5 [ 68] encoder. We use the frozen [CLS] representation as input to a classification head consisting
of three hidden blocks with a size of 512. The model is optimized using binary cross-entropy with
a1e−3 learning rate. Similar to the LLM router, we useargmaxat inference to determine a single,
query-uniform configuration.
D.3 Evaluation Metrics
We evaluate retrieval and generation separately. Let Qdenote the set of evaluation queries. For
each query q∈ Q , letGqbe the set of ground-truth evidence chunks annotated for q, and let
21

RK
q= [r 1, r2, . . . , r K]be the ranked list of the top- Kretrieved chunks. In our main experiments, we
setK= 5.
Recall@K.Recall@K measures whether the ground-truth evidence appears in the top- Kretrieved
chunks. For a queryq, we define
Recall@K(q) =|RK
q∩Gq|
|Gq|.(8)
The dataset-level score is the average over all queries:
Recall@K =1
|Q|X
q∈QRecall@K(q).(9)
Since each query in V-RAGBench is designed to have a uniquely sufficient evidence chunk, this
reduces to checking whether that evidence chunk is retrieved within the top-K.
nDCG@K.nDCG@K evaluates not only whether the evidence is retrieved, but also how highly it is
ranked. We use binary relevance:
reli=1[r i∈Gq],(10)
wherer iis the chunk ranked at positioni. The discounted cumulative gain is
DCG@K(q) =KX
i=1reli
log2(i+ 1).(11)
The ideal discounted cumulative gain is
IDCG@K(q) =min(K,|G q|)X
i=11
log2(i+ 1).(12)
Then,
nDCG@K(q) =DCG@K(q)
IDCG@K(q).(13)
The final nDCG@K is averaged over all queries:
nDCG@K =1
|Q|X
q∈QnDCG@K(q).(14)
Pass Rate.For generation, we report aggregate pass rate. Let ˆaqbe the model-generated answer for
query q, and let aqbe the ground-truth answer. We use an LLM-as-a-judge J(·) to determine whether
ˆaqis correct:
J(q,ˆa q, aq)∈ {0,1},(15)
where 1indicates that the generated answer matches the key factual content of the reference answer.
The pass rate is
PassRate =1
|Q|X
q∈QJ(q,ˆa q, aq).(16)
Latency.We additionally report latency for experiments where methods differ in online inference
cost. For each query q, letTqdenote the wall-clock time required by the evaluated online pipeline.
Depending on the experiment, this includes query encoding, retrieval, reranking, evidence rendering,
and generation, but excludes offline preprocessing such as video chunking, memory construction and
index construction. The average latency is
Latency =1
|Q|X
q∈QTq.(17)
When reporting retrieval-only latency, Tqincludes only the online retrieval and reranking steps; when
reporting end-to-end latency, it additionally includes generation.
22

Table 15: Baselines with reranking
Method Recall@5 nDCG@5
VideoRAG-A 0.510 0.343
VideoRAG-B 0.487 0.330
GQR 0.503 0.341
RRF 0.463 0.324
UniversalRAG 0.447 0.324
UniversalRAG-LORA 0.470 0.344
DAT 0.460 0.315
Freeret 0.263 0.212
GME 0.450 0.305
CARVE0.603 0.433
D.4 LLM-as-a-Judge Details
For evaluating answer correctness on open-ended questions, we use Qwen3.6-35B-A3B as the
backbone model for the LLM-as-a-judge. The prompt used for judgment is shown in Figure 8.
Human Verification.To assess the reliability of the LLM-as-a-judge, we conduct human validation of
its outputs. We randomly sample 600instances from the full set of 2,100 predictions (corresponding
to a95% confidence interval with a margin of error of ±4). We apply the same annotator qualification
criteria as in Section C.5, namely (i) graduate students in the field of AI, (ii) at least one first-author
publication at a top-tier AI conference, (iii) English proficiency equivalent to CEFR C2, and (iv)
completion of a 30-minute instructional briefing prior to annotation. Each annotator is then provided
with the model-generated answer, the ground-truth answer, and the binary decision from the LLM-as-
a-judge indicating whether the prediction is correct, and is asked to independently assess whether
the binary judge’s evaluation label should be considered correct. We find substantial agreement
between human annotators and the LLM-as-a-judge, with a Cohen’s Kappa score of 0.61. Further
inspection shows that in 2.2% of the instances, both annotators agree that the LLM-as-a-judge made
an incorrect decision. These results suggest that while assessing open-ended questions with a binary
LLM-as-a-judge may introduce minor noise, it can be considered generally reliable for large-scale
evaluation.
ECARVEAlgorithm
Algorithm 1 shows the formalization of the algorithm ofCARVE.
F Additional Analysis
We conduct the additional analysis as follows: (i) a comparison against reranked baselines to decouple
the impact of chunk-level configuration selection, (ii) a performance breakdown per source video to
evaluate robustness across different video scales and (iii) an error analysis on the generation.
F.1 Baseline with Reranking
To verify that CARVE ’s gains are not solely attributable to the reranking stage, we apply all baselines
with the same reranker used in our method. As illustrated in Table 15, while the addition of the reranker
generally yields performance improvements, these gains remain marginal and, CARVE significantly
outperforms even the reranked baseline. This result demonstrates that the advantage of CARVE does
not merely stem from the reranking mechanism as a standalone tool. Instead, it originates from
the system’s unique ability to select and interleave the most discriminative modality-granularity
configurations at the chunk level.
F.2 Performance Breakdown by Source Dataset
To analyze the effectiveness of CARVE across different video scales,we specifically analyze its
performance depending on the source dataset. Ego4D already presents a challenging retrieval task with
videos exceeding one hour, while EgoLife extends this challenge further with day-long recordings.
23

Algorithm 1CARVE: Chunk-Aware Reranking for Video Evidence
Require:
Per-wearer video corpusV(2-min non-overlapping chunks)
Queryq
Multimodal encoderf(Qwen3-VL-Embedding-2B by default)
Multimodal cross-encoderCE(·,·)(Qwen3-VL-Reranker-2B by default)
Captioner MLLM (Qwen3-VL-8B) for textual summaries (cap512tokens)
Keyframe countn(default5)
Per-configuration retrieval capk 0, final list sizek
GeneratorG
Ensure:
Multi-representation indexI
Tagged final top-klistV⋆
q={(v,(m⋆
v, g⋆
v))}
Predicted answerba
OFFLINE: CHUNK-WISEMULTI-REPRESENTATIONINDEXING
1:foreach chunkv∈ Vdo
2:K(v)←nkeyframes fromk-means++centroidsover per-frame embeddings ofv
▷centroids cover distinct visual states, not uniform time samples
3:ϕ vis,frame (v)←MeanPool 
{f(κ) :κ∈ K(v)}
4:ϕ vis,clip (v)←f(v)▷single embedding over the entire chunk
5:ϕ text,frame (v)←f 
∥κ∈K(v) Summarize(κ)
▷per-keyframe summaries concatenated before encoding
6:ϕ text,clip (v)←f(Summarize(v))
7:I ← I ∪
(m, g, v, ϕ m,g(v)) : (m, g)∈ {vis,text} × {frame,clip}	
8:end for
ONLINE: RETRIEVAL, RERANKING,ANDGENERATION
// Stage 1: Parallel candidate pooling — one query embedding, four index shards
9:Computef(q)once
10:P(q)← ∅▷configuration-tagged candidate pool
11:foreach(m, g)∈ {vis,text} × {frame,clip}do
12:C (m,g)(q)←Top-k 0v∈V
f(q), ϕ m,g(v)
▷cf. Eq. (2)
13:P(q)← P(q)∪
(v,(m, g)) :v∈C (m,g)(q)	
▷cf. Eq. (3)
14:end for
// Stage 2: Chunk-adaptive reranking — score each candidate under the tag that surfaced it
15:foreach(v,(m, g))∈ P(q)do
16:˜s 
q; (v,(m, g))
←CE(q, ϕ m,g(v))▷cf. Eq. (4)
17:end for
// Per-chunk winning configuration: keep each chunk’s highest-scoring tag
18:foreach chunkvappearing inP(q)do
19:(m⋆
v, g⋆
v)←arg max
(m,g) : (v,(m,g))∈P(q)˜s 
q; (v,(m, g))
▷cf. Eq. (5)
20:end for
// Final cross-modal top-k: rank chunks by their winning scores on a comparable scale
21:V⋆
q←n
(v,(m⋆
v, g⋆
v)) :v∈top-kchunks by˜s 
q; (v,(m⋆
v, g⋆
v))o
▷cf. Eq. (6)
// Modality-interleaved generation: each chunk speaks through its winning representation
22:ba← G
q,
ϕm⋆v, g⋆v(v)	
(v,(m⋆v,g⋆v))∈V⋆q
▷cf. Eq. (7)
23:returnI, V⋆
q,ba
As shown in Table 16, CARVE outperforms all baselines in both videos, but the performance gap is
even more pronounced in the super-long recordings of EgoLife. Specifically, while the strongest
24

baseline achieves an nDCG@5 of 0.302 on EgoLife, CARVE reaches 0.394, significantly widening the
margin compared to its performance on the hour-scale Ego4D clips.
Table 16: Performance across EgoLife and Ego4D datasets. Generation model is Qwen3-VL-8B. Best
values in bold.
EgoLife Ego4D
Method R@5 nDCG@5 Pass R@5 nDCG@5 Pass
VideoRAG-A 0.459 0.295 0.253 0.523 0.365 0.246
VideoRAG-B 0.459 0.302 0.318 0.523 0.368 0.315
GQR 0.459 0.289 0.135 0.562 0.408 0.208
RRF 0.447 0.277 0.177 0.485 0.324 0.185
UniversalRAG 0.394 0.275 0.308 0.515 0.389 0.282
UniversalRAG-LORA 0.429 0.300 0.235 0.523 0.403 0.239
DAT 0.465 0.302 0.212 0.454 0.325 0.239
Freeret 0.194 0.145 0.197 0.354 0.242 0.239
GME 0.359 0.217 0.241 0.485 0.300 0.285
CARVE0.571 0.394 0.359 0.646 0.484 0.354
F.3 Error Analysis on Retrieval and Generation
To analyze the distribution of failures, we categorized the results into two primary types across Qwen3-
VL-8B and 32B scales.Case 1, represents both retrieval and generation fail, andCase 2, represents
retrieving success but the generator fails to produce the correct answer. As shown in Table 17, for
the 8B model, Case 1 and Case 2 occurred in n= 92 andn= 108 instances, respectively. As the
generator scale increased to 32B, these frequencies decreased ton= 86andn= 104.
Table 17: Retrieval cases where generation fails, across two generator scales.
Case Retrieval Qwen3-VL-8B Qwen3-VL-32B
Case 1 Fail 92 86
Case 2 Hit 108 104
F.4 Effect of Top-K
We also vary the final retrieval depth kto test whether CARVE is sensitive to the number of chunks
passed to the generator. This setting is important because increasing kcan improve recall by adding
more candidate evidence, but it can also introduce distractor chunks that make generation harder.
A robust VideoRAG method should therefore maintain strong performance across different retrieval
depths, rather than relying on a single tuned value ofk.
Across the tested values of k,CARVE remains stable and consistently competitive, showing that its
gains do not depend on a particular retrieval budget. This indicates that chunk-wise parallel reranking
improves the ordering of evidence itself, rather than merely benefiting from retrieving more chunks.
In other words, CARVE is effective because it selects better-ranked evidence, not because it relies on a
larger context window. This experiment shows thatchunk-adaptive reranking produces a robust
evidence ranking across retrieval depths, makingCARVEless sensitive to the final top-kchoice.
F.5 Reranker Backbone Robustness
Table 18 shows the retrieval performance per each representation configuration under different
reranker backbone model (LamRA-Rank-7B.).
G Prompts
G.1 QA Generation Prompt
Figure 5 presents the prompt used to generate visually grounded QA pairs from egocentric video
segments, with each question anchored to a unique observable event in the frames.
25

k=1 k=2 k=4 k=8 k=10TC
TF
VC
VF
CARVE0.22 0.29 0.40 0.53 0.58
0.16 0.28 0.39 0.53 0.60
0.22 0.30 0.46 0.61 0.64
0.21 0.30 0.42 0.55 0.62
0.27 0.39 0.55 0.71 0.75Recall@5
k=1 k=2 k=4 k=8 k=10TC
TF
VC
VF
CARVE0.22 0.23 0.28 0.33 0.35
0.16 0.23 0.28 0.35 0.38
0.22 0.25 0.34 0.42 0.43
0.21 0.24 0.30 0.37 0.40
0.27 0.32 0.41 0.48 0.49nDCG@5
0.200.300.400.500.600.70
0.200.250.300.350.400.45Figure 4: Heatmaps of Recall@5 and nDCG@5 under different final retrieval depths.
G.2 QA Filtering Prompt
Table 18: Reranking configuration choice
underCARVEw. LamRA-Rank-7B.
Method R@5 nDCG@5 Latency
m={text}, g={frame} 0.520 0.339 1.0s
m={text}, g={clip} 0.503 0.318 0.5s
m={vis}, g={frame} 0.537 0.357 4.5s
m={vis}, g={clip} 0.563 0.380 8.7s
Random 0.537 0.347 4.4s
Concatenation 0.543 0.367 11.3s
CARVE(Ours) 0.577 0.401 4.0sFigure 6 shows the prompt used to verify whether each
generated QA pair is answerable from the provided
frames and whether the answer is correct. In addition,
Figure 7 presents the prompt used to check whether a
question can be answered without visual input, using
only common sense and the question itself.
G.3 LLM Judge Prompt
Figure 8 presents the prompt used by the LLM judge
to evaluate whether a candidate answer matches the key
factual content of the reference answer.
G.4 Memory Generation Prompt
Figure 9 presents the prompt used to generate text-clip
memories that summarize what the camera wearer sees,
does, and experiences over an egocentric video clip. In
addition, Figure 10 presents the prompt used to generate text-keyframe memories that describe the
camera wearer’s visual experience at a specific moment in detail.
G.5 QA Answering Prompt
Figure 11 presents the prompt used to answer questions from retrieved textual descriptions, including
either keyframe or clip memory contexts. In addition, Figure 12 presents the prompt used to answer
questions from combined textual memories that include both clip-level descriptions and keyframe-
level descriptions. Furthermore, Figure 13 shows the prompt used to answer questions from retrieved
video segments, while Figure 14 shows the prompt used to answer questions from retrieved keyframe
images. Finally, Figure 15 presents the prompt used to answer questions from combined visual
evidence, including both video segments and keyframe images.
26

You will receive a sequence of frames from a segment of an egocentric video.# ObjectiveCreate up to 4QA pairs, grounded strictly in visibleevidence from the frames.-Generate fewer than 4 if the segment does not naturally support all categories.-Each question must localize a single, unique moment in the segment.-Do not use prior knowledge or common sense.# Category Rule-Attempt to create one QA pair per category from the list below.-However, ONLY generate a QA pair for a category if there is clear, strong visual evidence in the frames.-If a category is not clearly supported by the visible content, DO NOT force a question for that category.{category_block}# Step-by-step Procedure (DO NOT reveal these steps in the output)For each of the categories, do:Step 1) Select ONE visually unique event boundary (anchor).-The anchor must include:(1) a concrete object-Include color, material, location, or distinctive visual detail.-Avoid generic words -Avoid camera motion, viewpoint shift, reframing, gaze shift, zoom, panning, tilting, or scene transition as the anchor event.-Include specific identifying details such as color, material, relative location, shape, size, or any other distinctive visual attribute.(2) a clearly observable physical action(3) a resulting visible outcome or state changesStep 2) Draft a question using the anchor:-e.g., "After/When <ANCHOR EVENT>, <QUESTION>?" or any natural phrasing that clearly references the anchor event.-The anchor provides temporal grounding.-The question and answer must belong to the category.Step 3) Ensure the question asks about a DIFFERENT visual detail than the anchor.-The anchor must NOT contain the answer.-If the answer could be inferred from the anchor alone, rewrite.Step 4) Write the reasoning (justification).-Describe the specific visual evidence observed in the frames that is relevant to the question.-State what objects, actions, or state changes are visible and where they appear.Step 5) Based on the reasoning above, write the answer as exactly ONE concrete visual fact.-The answer must be derived from the evidence described in the reasoning.# Constraints-Each question anchors to a unique event.-Avoid anchors based on generic or common actions (e.g., holding or looking at a phone).-Avoid mechanical or formulaic phrasing.-Each answer uses only observable visual evidence.-Each answer states only a single fact; not multiple actions.-No timestamps or frame numbers.-Maintain an egocentric viewpoint; use "I" only for the wearer's visible actions.-Do NOT base questions on fine-grained phone screen details.-Do NOT ask about frequency, repetition, or habitual behavior.After producing the QA pairs, add a 1-2 line validation confirming all constraints are met. Revise output if needed.# OutputProduce 1 to 4 QA pairs, all grounded exclusively in the provided frames. Figure 5: Prompt for QA Generation.
27

Evaluate each QA pair below.For each, determine if the questioncan be answered from these frames and if the answer is correct.{qa_block}Figure 6: Prompt for GPT Answerability Filtering.
You have NO frames or visual input.Try to answer using ONLY common sense and the question.Question:{question}
Figure 7: Prompt for GPT Blind Check Filtering.
You are a strict evaluator judging whether a candidate answer to a question about video content is correct, using a reference answer as the ground truth.Question:{question}Reference Answer:{reference}Candidate Answer:{candidate}Evaluation criteria:-YES: The candidate answer captures the key factual content of the reference answer. It does not need to match word-for-word, but must convey the core information. Some additional details or differences in phrasing are acceptable.-NO: The candidate answer is factually wrong, contradicts the reference, misses the central point, is vague or generic where the reference is specific, or admits it cannot answer the question when the reference provides a clear answer.Important:Focus on whether the essential facts match, not surface-level wording. A shorter answer that gets the key facts right is YES. A longer answer that misses the main point is NO.Answer with ONLY "YES" or "NO" followed by a one-sentence reason.Format: [YES/NO]: [REASON]
Figure 8: Prompt for LLM Judge.
You are analyzing an egocentric (first-person) video clip from someone's day.Describe what the person is seeing, doing, and experiencing during this clip.Use first-person perspective ("I am... " , "I see... ")and be specific about:-Activities and actions being performed-Objects being handled or interacted with-Environment and location details-Any notable events or transitions-People present and interactions with them-Time of day indicators if visibleProvide a natural, coherent description that captures the essence of what happened during this time period.
Figure 9: Prompt for Text Clip Memory Generation.
28

You are analyzing an egocentric (first-person) image from someone's day.Describe in detail what I am seeing and doing at this exact moment.Use "I" perspective and be extremely specific and detailed. Focus on:-What I am currently looking at (center of my visual field)-What my hands are doing (if visible)-Objects I am interacting with or holding-The environment and location details-Any text, labels, or screens visible-Other people present and their actionsProvide a detailed, vivid description of this exact moment.Figure 10: Prompt for Text Key Frame Memory Generation.
You are a helpful assistant answering questions about video content. The following segments are language descriptions of a first-person view video, arranged in chronological order with timestamps.Retrieved context:{context}Question: {question}Based on the descriptions above, provide a clear and concise answer. If there is not enough information to answer, say so.
Figure 11: Prompt for Text Key Frame and Text Clip Answering.
You are a helpful assistant answering questions about video content. The following segments are language descriptions of a first-person view video, arranged in chronological order with timestamps.Each segment includes two levels of description:-[Coarse Description]: A high-level textual summary of the entire segment.-[Fine Descriptions]: Detailed textual descriptions of individual keyframe moments within the segment, capturing specific visual details.Retrieved context:{context}Question:{question}Provide a clear, concise answer based on the retrieved context above. Use both coarse and fine descriptions to form a comprehensive answer. If the context doesn't contain enough information to answer the question, say so.
Figure 12: Prompt for Text Combined Answering.
You are a helpful assistant answering questions about video content.  The video segments above are from a first-person view video, arranged in chronological order.Question:{question}Based on the video content above, provide a clear and concise answer. If there is not enough information to answer, say so.
Figure 13: Prompt for Visual Clip Answering.
29

You are a helpful assistant answering questions about video content. The keyframe images above are from a first-person view video, arranged in chronological order.Question:{question}Based on the images above, provide a clear and concise answer.If there is not enough information to answer, say so.Figure 14: Prompt for Visual Key Frame Answering.
You are a helpful assistant answering questions about video content. The video segments and keyframe images above are from a first-person view video, arranged in chronological order.Question:{question}Based on all the visual content above, provide a clear and concise answer. If there is not enough information to answer, say so.
Figure 15: Prompt for Visual Combined Answering.
30