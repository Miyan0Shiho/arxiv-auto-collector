# M$^3$KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation

**Authors**: Hyeongcheol Park, Jiyoung Seo, Jaewon Mun, Hogun Park, Wonmin Byeon, Sung June Kim, Hyeonsoo Im, JeungSub Lee, Sangpil Kim

**Published**: 2025-12-23 07:54:03

**PDF URL**: [https://arxiv.org/pdf/2512.20136v2](https://arxiv.org/pdf/2512.20136v2)

## Abstract
Retrieval-Augmented Generation (RAG) has recently been extended to multimodal settings, connecting multimodal large language models (MLLMs) with vast corpora of external knowledge such as multimodal knowledge graphs (MMKGs). Despite their recent success, multimodal RAG in the audio-visual domain remains challenging due to 1) limited modality coverage and multi-hop connectivity of existing MMKGs, and 2) retrieval based solely on similarity in a shared multimodal embedding space, which fails to filter out off-topic or redundant knowledge. To address these limitations, we propose M$^3$KG-RAG, a Multi-hop Multimodal Knowledge Graph-enhanced RAG that retrieves query-aligned audio-visual knowledge from MMKGs, improving reasoning depth and answer faithfulness in MLLMs. Specifically, we devise a lightweight multi-agent pipeline to construct multi-hop MMKG (M$^3$KG), which contains context-enriched triplets of multimodal entities, enabling modality-wise retrieval based on input queries. Furthermore, we introduce GRASP (Grounded Retrieval And Selective Pruning), which ensures precise entity grounding to the query, evaluates answer-supporting relevance, and prunes redundant context to retain only knowledge essential for response generation. Extensive experiments across diverse multimodal benchmarks demonstrate that M$^3$KG-RAG significantly enhances MLLMs' multimodal reasoning and grounding over existing approaches.

## Full Text


<!-- PDF content starts -->

M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced
Retrieval-Augmented Generation
Hyeongcheol Park1Jiyoung Seo1Jaewon Mun1Hogun Park2
Wonmin Byeon3Sung June Kim1Hyeonsoo Im4JeungSub Lee4Sangpil Kim1*
1Korea University2Sungkyunkwan University3NVIDIA Research4Hanhwa Systems
Abstract
Retrieval-Augmented Generation (RAG) has recently
been extended to multimodal settings, connecting multimodal
large language models (MLLMs) with vast corpora of ex-
ternal knowledge such as multimodal knowledge graphs
(MMKGs). Despite their recent success, multimodal RAG
in the audio-visual domain remains challenging due to 1)
limited modality coverage and multi-hop connectivity of
existing MMKGs, and 2) retrieval based solely on similar-
ity in a shared multimodal embedding space, which fails
to filter out off-topic or redundant knowledge. To address
these limitations, we propose M3KG-RAG, a Multi-hop Mul-
timodal Knowledge Graph-enhanced RAG that retrieves
query-aligned audio-visual knowledge from MMKGs, im-
proving reasoning depth and answer faithfulness in MLLMs.
Specifically, we devise a lightweight multi-agent pipeline
to construct multi-hop MMKG (M3KG), which contains
context-enriched triplets of multimodal entities, enabling
modality-wise retrieval based on input queries. Further-
more, we introduce GRASP (Grounded Retrieval And Se-
lective Pruning), which ensures precise entity grounding
to the query, evaluates answer-supporting relevance, and
prunes redundant context to retain only knowledge essential
for response generation. Extensive experiments across di-
verse multimodal benchmarks demonstrate that M3KG-RAG
significantly enhances MLLMs’ multimodal reasoning and
grounding over existing approaches.
1. Introduction
The advancements in Retrieval-Augmented Generation
(RAG) have substantially improved the factual accuracy and
faithfulness of large language models (LLMs) by connect-
ing them to vast external knowledge corpora [ 28,41]. Re-
cently, graph-based RAG methods [ 11,18,57] have further
pushed the progress by supporting structured reasoning and
precise, query-relevant retrieval. However, extending these
schemes to multimodal settings—jointly handling audio, vi-
sual, and textual signals—is non-trivial as heterogeneous
*Corresponding author.
(a) Lack of Multimodal Coverage
(b) Insufficient Knowledge
(c) 𝐌𝟑KG-RAGceramicsvasemade ofhas partstill life
Shared-embedding space Search
Modality-wise Search
Single-hop Retrieval
Modality-wiseSearch
Multi-hop Retrieval
manInaccurate responseIncorrect retrieval
Concept-level graphsUnimproved response
Multi-hop graphsFaithful response
[Reference]: In the room, two men played the instrument to the sound of the instrument.[Reference]: On the ground lay a raid siren, which was spinning at high speed, making a sharp sound.
[Reference]:Four men sat with a group of women, using trombones, trumpets, French horns and                           percussion instruments to play a joyous music.bodyholdingflowers
used for
congapercussionplayssitsroomtuneproduce sound[Prediction]:In the video, two men are playing percussion instruments in a studio... There are various  instruments, including conga drums, bongos, and…
[Prediction]:The video features a group of five musicians performing on a stage. Each holding a brass instrument …[Prediction]: The video shows a large red and silver speakeron the ground. The sound of the speaker is loud and …peoplebrass quintetfive menplaystadiummusiciansare ininstrumentplay[Question]: Describe in detail what is visually and audibly happening in the video, including actions, objects, people, sounds, and environment.
playsmakesFigure 1.Illustration of multimodal RAG scenarios.Incorrect
answers are shown in red, correct answers in blue. (a) Shared
embedding search misaligns with the audio-visual query. (b) Noisy,
single-hop facts provide little answer support. (c) M3KG-RAG uses
modality-wise multi-hop retrieval for answer-supporting context.
inputs raise complexity, motivating designs that explicitly ac-
count for multimodal structure. Recent work [ 27,40,55] ad-
dresses these challenges with multimodal knowledge graphs
(MMKGs) that organize cross-modal knowledge as entities
and relations, thereby delivering query-relevant evidence to
multimodal large language models (MLLMs) [1, 53].
However, existing MMKG-enhanced RAG methods ex-
hibit key limitations.First, existing MMKGs [ 26,34,55]
largely emphasize image-text and provide limited audio-
visual coverage, which hampers temporal and causal reason-
ing across modalities. Moreover, the modality gap [ 39] in
unified multimodal embedding spaces makes cross-modal re-
trieval inaccurate [ 54]. As illustrated in Fig. 1-(a), matching
an audio-visual query directly against a text-only knowl-
edge base often fails to retrieve truly related evidence, which
motivates modality-wise retrieval. While recent work [ 37]
builds an audio-visual MMKG for precise retrieval, it in-
1arXiv:2512.20136v2  [cs.CL]  24 Dec 2025

duces concept-level, single-hop graphs that rarely capture
temporal or causal dependencies. Therefore, constructing a
multi-hop, modality-aware knowledge source across audio
and visual streams is essential for query-relevant retrieval
and reliable spatio-temporal reasoning.
Second, most multimodal retrieval strategies [ 15,23,37,
54] rely on similarity search in shared embedding spaces,
which captures the query’s broad semantics but misses fine-
grained cues. They often fail to select fine-grained, query-
relevant knowledge, retrieve off-topic content, and add re-
dundant context. Even when retrieved knowledge aligns
with the query, facts that do not contribute to the answer
introduce noise in the MLLM context. As shown in Fig. 1-
(b), such simple RAG frameworks, even when they retrieve
context-matched knowledge for audio-visual queries, inject
noisy evidence that fails to improve response.
To address these limitations, we propose M3KG-RAG,
an end-to-end, graph-enhanced RAG framework that con-
structs a multi-hop MMKG for modality-wise retrieval and
supplies only query-aligned, answer-supportive knowledge.
Specifically, we transform raw multimodal corpora into a
multi-hop MMKG (M3KG) with a lightweight, collabora-
tive multi-agent pipeline in three steps. First, we perform
(i) Context-Enriched Triplet Extraction, which captures
knowledge-intensive entities and relations containing tempo-
ral and cross-modal cues. As triplets alone lack enough con-
text for reliable reasoning [ 37,55], we perform(ii) Knowl-
edge Groundingto obtain canonical entity identifiers and
descriptions using external resources and tools. Finally,(iii)
Context-Aware Description Refinementaligns entity de-
scriptions with the surrounding multimodal context to ensure
consistency and specificity. In addition, we incorporate a
Self-Reflection Loopto prevent possible hallucinated or
misaligned descriptions during construction.
Additionally, we introduceGrounded Retrieval And Se-
lective Pruning (GRASP)to keep only query-relevant and
answer-useful subgraphs. GRASP first leverages off-the-
shelf multimodal grounding models [ 35,51] to drop triplets
not appearing in the query, and then applies a light LLM [ 43]
to prune triplets that do not contribute to answering the
question. As shown in Fig. 1-(c), our framework retrieves
knowledge tightly linked to the query from the constructed
multi-hop MMKG and passes only answer-relevant evidence
to the MLLMs. Extensive experiments across diverse audio,
video, and audio-visual QA demonstrate that M3KG-RAG
achieves substantial performance gains over existing meth-
ods. Our contributions are summarized as follows:
•We present M3KG-RAG, an end-to-end framework that
integrates a multi-hop MMKG with RAG to enhance audio-
visual reasoning in MLLMs.
•We propose a three-step, multi-agent pipeline that builds a
multi-hop MMKG from raw multimodal corpora, enabling
scalable, modality-wise retrieval.•We introduce Grounded Retrieval And Selective Pruning
(GRASP), which discards graph elements absent from the
query or unhelpful for answering and retains only query-
relevant, answer-useful subgraphs for the MLLMs.
•Through extensive evaluations across diverse multimodal
benchmarks, we demonstrate that M3KG-RAG consis-
tently outperforms strong RAG baselines.
2. Related Work
2.1. Multimodal Large Language Model
Recent advances in large language models (LLMs) [ 1,3,
10,17,43,52] have showcased strong reasoning and genera-
tion capabilities within the language domain. This progress
has extended to multimodal settings (e.g., vision and au-
dio), leading to the emergence of multimodal large language
models (MLLMs). Early MLLMs, such as Flamingo [ 2]
and BLIP-2 [ 31], focused on vision–language understand-
ing through lightweight cross-modal interfaces built on top
of frozen LLM backbones. Subsequent works broadened
both the scale and reasoning capabilities of MLLMs. For
instance, LLaV A [ 32] enables general-purpose image–text
understanding through visual instruction tuning using a pre-
trained vision encoder and LLM. Later variants [ 29,30]
further extend the framework to incorporate video inputs.
Parallel efforts in the audio domain have produced MLLMs
capable of reasoning over auditory inputs. SALMONN [ 42]
integrates speech and general sound understanding to LLMs
through specialized audio encoders. More recently, Kimi-
Audio [ 9] achieves strong results across audio understanding,
generation, and conversational tasks as an audio foundation
model. Motivated by the success of both modalities, recent
MLLMs focus on enhancing joint audio-visual understand-
ing. Video-LLaMA2 [ 8] realizes this with a dual branch for
spatial–temporal video and audio cues, improving event and
scene comprehension under synchronized fusion. Qwen2.5-
Omni [ 49] further targets real-time interaction, supporting
perception and generation across multiple modalities in a
streaming manner. Complementing open-source models,
commercial models such as GPT-4o [ 21] extend multimodal
I/O to low-latency audio–visual dialogue, reflecting a shift
toward tightly integrated perception and reasoning.
2.2. Multimodal RAG
Retrieval-Augmented Generation (RAG) conditions LLM
generation on retrieved knowledge, improving grounding
and factuality [ 5,12,13,20,22,28]. To support multi-hop
compositional reasoning across entities and relations, graph-
based RAG represents knowledge as entity–relation graphs.
Along this line, GraphRAG [ 11] improves coherence and
coverage via graph-aware indexing and community sum-
maries, while LightRAG [ 18] uses dual-level retrieval for ef-
ficient, interpretable selection. HippoRAG2 [ 19] strengthens
2

Graph
Extractor
Semantic-level CaptionExternalKnowledge<Title>: Inside a Pilatus PC-12 at Lebanon Airport!<Description>: WATCH IN HD!!!!One of Goodspeed's Pilatus PC-12…
Multimodal Corpus
Step 1 : Context-Enriched Triplet Extraction
Context-Enriched Caption
“A plane slowly pulled out of the open runway, the propeller turning at high…
SearcherNormalizer
Step 2 : Knowledge Grounding- A Pilatus PC-12  plane- the propeller⋯
LLM CallbackSearchableUnsearchable
DescriptionCandidates
Selector
Step 3 : Context-Aware Description Refinement 
Original ConceptsHead : A Pilatus PC-12 is … ,Relation: has ,Tail : A mechanical device ...Context-Enriched Caption
External Knowledge
The Pilatus PC-12 is a pressurized, single-engined, aircraft ...You are an encyclopedic writer Goal: Write a neutral, generic, sentence description ….Inspector
Evaluate
Pass or Re-Run
Score 0-10 how well the description’s encyclopedic sense matches the intended sense of …Inspector
Evaluate
Pass or Re-Run
Score 0-10 how well the description’s encyclopedic sense matches the intended sense of …
"A Pilatus PC-12 plane slowly pulled out of the open runway, the propeller …""A Pilatus PC-12 plane slowly pulled out of the open runway, the propeller …"- Pilatus PC-12- PropellerSearchableConceptsOriginal ConceptsSelected description
Refiner
Rewriter
Refined descriptions
Figure 2.An overview of the M3KG construction pipeline.The pipeline consists of three steps: (i)Context-Enriched Triplet Extraction,
which rewrites multimodal captions into knowledge-intensive text and extracts entity–relation triplets; (ii)Knowledge Grounding, linking
normalized entities to open knowledge bases to obtain candidate descriptions; (iii)Context-Aware Description Refinement, selecting and
rewriting the most context-relevant descriptions for each entity; andSelf-Reflection Loop, where an inspector agent validates or re-runs
uncertain outputs to ensure graph quality.
multi-hop retrieval with PPR-style walks and on-the-fly pas-
sage integration under tighter online LLM. With the advent
of MLLMs, RAG extends beyond text to multimodal ground-
ing, yet text-only graph RAG methods do not account for
intrinsic multimodal semantics, making direct transfer dif-
ficult. In response, multimodal knowledge graph (MMKG)
based approaches have emerged, encoding entities and rela-
tions across modalities. MR-MKG [ 27] leverages MMKG
with graph encoding and cross-modal alignment to improve
multimodal reasoning. M2ConceptBase [ 55] organizes im-
age–text corpora into a concept-centric MMKG and couples
it with graph-aware retrieval, improving grounding and an-
swer accuracy for multimodal QA and description tasks.
Pushing beyond image–text, V AT-KG [ 37] integrates visual,
audio, and text into an MMKG and proposes a RAG proto-
col tailored to audio–visual MLLMs, improving faithfulness
under audio–visual queries. However, its graph construction
is largely single-hop, and its RAG framework mainly relies
on similarity-based search, which can admit off-topic neigh-
bors and redundant context. In contrast, we build a multi-
hop MMKG and execute fine-grained, query-conditioned
retrieval, supplying answer-useful context.
3. Method
In this section, we detail the M3KG-RAG paradigm, em-
phasizing its core architecture and contributions. Sec. 3.1
presents our multi-hop multimodal knowledge graph
(M3KG) construction pipeline. Sec. 3.2 introduces our multi-
modal RAG framework equipped with the proposed GRASP.
3.1. M3KG Construction with Multi Agents
Our M3KG-RAG improves the retrieval scheme by con-
structing a multi-hop MMKG that enables scalable, in-depthreasoning over multiple multimodal triplets. However, since
constructing a multi-hop MMKG requires multiple stages
beyond simple graph connectivity, we design a lightweight,
collaborative multi-agent pipeline with our own specialized
LLM agents—rewriter,extractor,normalizer,searcher,se-
lector,refiner, andinspector—to balance automation and
quality control, in line with recent advances in multi-agent
LLM systems and self-reflection [ 4,44]. The overall con-
struction process is illustrated in Fig. 2, and the detailed role
and method for each step are as follows.
Step 1: Context-Enriched Triplet ExtractionOur multi-
hop MMKG construction starts from a raw multimodal cor-
pusC={(xtext
n, xaudio
n, xvisual
n )}N
n=1 ofNaligned sam-
ples, where xtext
n, xaudio
n, xvisual
n denote the text, audio, and
visual data for sample n. Much of the text in Cis seman-
tically generic [ 7,25], which limits its utility as external
knowledge for MLLMs. Following prior MMKG construc-
tion studies [ 37], we develop a newrewriterthat converts
semantic-level caption xtext
ninto a context-enriched cap-
tion˜xtext
nby incorporating external knowledge—titles and
descriptions collected via a crawler—to supply knowledge-
intensive context. Motivated by recent advances in open
information extraction leveraging LLMs [ 56,58], we further
introduce anextractorthat parses ˜xtext
nand returns triplets
Tn={(h i, ri, ti)}Kn
i=1, where hi,ri, and tidenote the head
entity, relation, and tail entity and Kn=|T n|varies by
input. As the rewritten ˜xtext
nis knowledge-intensive and
summarizes the overall multimodal context, the extracted
triplets often capture relations among long-tail or uncommon
entities—cases that MLLMs commonly miss or misidentify.
Step 2: Knowledge GroundingWhile transforming the
corpus into a graph structure enables efficient entity-level
access, connections alone offer limited guidance to MLLMs.
3

To enrich the MMKG beyond connectivity, we ground ency-
clopedic descriptions to entities in this step. Head and tail
entities in Toften include modifiers or variant surface forms
that hinder look-up (e.g., “small brown dog” vs. “dog”).
Thus, thenormalizerfirst maps each entity mention to a
canonical, searchable concept by removing non-essential
modifiers, preserving the source word order when appropri-
ate, and standardizing to a singular noun phrase. Given the
normalized concepts, thesearcherqueries open knowledge
bases (e.g., Wikipedia, Wiktionary) and uses a crawler to re-
trieve a compact set of candidate descriptions for each entity.
Since open knowledge bases cannot cover every textual con-
cept, we include a lightweight LLM callback to fill missing
descriptions. Subsequently, we have a candidate description
setDfor every normalized concept.
Step 3: Context-Aware Description RefinementA single
term can carry multiple meanings, (e.g., “bank”: financial in-
stitution vs. a river bank). To make entities accurately infor-
mative, theselectorchooses the most context-appropriate de-
scription from the candidate set, using the context-enriched
caption fromStep 1as guidance for each normalized con-
cept. This keeps descriptions aligned with the context and
filters out off-topic ones. Since the selected descriptions
are written for a normalized concept rather than the original
heads and tails in T, therefineradapts the chosen descrip-
tion to the original concept’s phrasing to inject the original
semantics while preserving the selected content. After this
step, we obtain a refined description set ˆDfor all entities.
Self-Reflection LoopTo ensure knowledge graph quality,
we introduceInspectorthat implements a self-reflection loop
within our construction pipeline. When the task extends
beyond simple information extractions and instead relies on
the language model’s implicit knowledge (e.g., Step 2 LLM
Callback or Step 3 Rewriter), errors may occur. Accordingly,
the inspector reviews these outputs and either passes them
or returns a re-run signal to the producing agent.
Resulting M3KGStarting from the text data xtext
n in
multimodal corpus C, our multi-agent pipeline with a self-
reflection loop constructs a multi-hop knowledge graph
and links its triplets to the corresponding audio-visual data
(xaudio
n, xvisual
n ), yielding the following multi-hop multi-
modal knowledge graph:
G={E,R,T, ˆD,A,V,L},(1)
whereEis the set of entities; Rthe set of relations; T ⊆ E ×
R×E the set of triplets; ˆD={d e}e∈Ethe per-entity refined
descriptions; A={xaudio
n}N
n=1andV={xvisual
n}N
n=1the
audio/visual items; L ⊆ T ×(A ∪ V) the links from triplets
to associated audio/visual data. The resulting multi-hop
MMKG satisfies the following coverage property:
∀t∈ T,∃x∈(A ∪ V)s.t.(t, x)∈ L.(2)Consequently, since every triplet links to at least one
multimodal item, all facts are eligible for retrieval under
multimodal queries, providing full graph coverage.
3.2. Multimodal RAG Framework
To deliver query-relevant and answer-useful context only,
we design a multimodal RAG framework composed of (i)
Modality-Wise Retrieval over the MMKG to gather candi-
dates aligned with the input modalities, and (ii) GRASP to
keep only knowledge that is relevant to the multimodal query
and useful for answering the question. An overview of the
multimodal RAG framework is shown in Fig. 3.
Modality-Wise RetrievalShared embedding spaces from
multimodal encoders often exhibit a modality gap where
cross-modal distances are not comparably calibrated [ 54].
Consequently, querying a knowledge base indexed in a dif-
ferent modality—for instance, a video query against text
embeddings—often yields off-topic neighbors. To bridge the
modality gap, we find the items of the same modality as the
query in Gand lift them to triplets. This procedure is enabled
by Eq. 2, which guarantees that each triplet in Gis linked
to at least one audio or visual item. Concretely, we obtain
query embeddings with multimodal foundation models (e.g.,
InternVL2 [ 47] for video and CLAP [ 48] for audio) and
search a FAISS [24] index built overG’s audio/visual items
using L2 distance in the embedding space. We first retrieve
the top- knearest items and then keep only candidates within
a distance threshold τof the query to avoid off-topic neigh-
bors. When both audio and video are provided, we form a
simple vector concatenation and apply the same search. Let
S ⊆(A ∪ V) denote the set of audio/visual items selected
by the above retrieval, we then lift items to triplets to obtain
the query-relevant initial graph:
Ginit={t∈ T | ∃x∈ S,(t, x)∈ L}.(3)
GRASP (Grounded Retrieval And Selective Pruning)
After mitigating the modality gap, we obtain a query-aligned
initial graph Ginit. Yet, its similarity-only retrieval can lack
fine-grained alignment or include knowledge that may not be
useful to answer the question. For instance, if the question
asks“What instrument is being played?”, only knowledge
about the instruments present is useful, and the remaining
triplets mostly become noise for the MLLMs. To address
these limitations, we design a Grounding Retrieval And Se-
lective Pruning (GRASP) to align the graph finely with the
query and retain only answer-useful knowledge. Specifically,
we first use off-the-shelf multimodal grounding models to
verify whether entities or triplets in Ginitappear in the
query’s audio and/or visual streams. For visual grounding,
we use GroundingDINO [ 35] on four uniformly sampled
frames Ffrom the query video qv, obtaining per-frame de-
tection confidences Φv(e;f) for each entity e∈ G initon
4

Multimodal QuerySimilarity SearchTop-𝑘 retrieval(a) Modality-Wise Retrieval
MultimodalLarge Language Model
Question(b) GRASP—Grounded Retrieval And Selective Pruning
Multimodal Grounded RetrievalKeep only helpful knowledgefor question
Multi-hop Retrieved Knowledge Graphwoodenflooris locatedyellow dogmakes sounddogchewherselfmeatis_ongray Jack Russell mixflap
bonechews×𝐹[Visual-Grounding][Audio-Grounding]
“opossum”“dog”“meat”0.912.271.67“dog”“meat”2.27“wooden floor”1.67
⋮[dog]: 0.81[meat]: 0.75[bone]: 0.23[opossum]: 0.03
Final score0.81+0.75+0.71=2.270.75+0.03+0.13=0.910.81+0.23+0.63= 1.670.71 0.130.63 
foodeatslicksis
“wooden floor”eatsopossumlicks
rests on“dog eats meat”“opossum eats meat” “dog chews bone” 
MultimodalEncoder..
Figure 3.Overview of the Multimodal RAG framework.The framework consists of two components:(a) Modality-Wise Retrieval,
which retrieves multi-hop triplets aligned with the query from the M3KG; and(b) GRASP (Grounded Retrieval And Selective Pruning),
which uses visual and/or audio grounding models to check entity presence and prunes triplets that are off-topic or non-informative. The
resulting subgraph is then provided to an MLLM for query-relevant, evidence-grounded audio-visual reasoning.
each frame f. We then take the maximum across the sampled
frames as the visual presence score of each entity:
sv(e|q v) = max
f∈FΦv(e;f).(4)
We derive a triplet-level visual presence score by sum-
ming the presence scores of the head and tail entities and
prune triplets whose score falls belowη v.
For audio grounding, we use a Text-to-Audio Grounding
model (TAG) [ 51]. Unlike visual signals, audio is not as
easily factorized into independent snapshots. Accordingly,
we convert each triplet tinto a natural sentence σ(t) (e.g., "h
r t") and use the TAG scoring function Φato measure how
strongly σ(t) is grounded in the query audio qa, resulting in
the audio presence score:
sa(t|q a) = Φ a(σ(t);q a).(5)
Similar to the visual case, we drop triplets whose audio
presence score falls below ηa. When both audio and visual
streams are available, we simply sum their presence scores
and remove triplets whose fused score is below ηav. This
grounding step yields the grounded subgraphG grd.
After obtaining the grounded subgraph Ggrd, we remove
knowledge that is not helpful for answering the question. A
lightweight LLM [ 50] produces a binary mask over triplets
under a conservative keep-or-drop policy, yielding GGRASP .
This procedure filters unhelpful and off-topic triplets while
keeping answer-supportive knowledge.
Graph-Augmented GenerationOur M3KG-RAG targets
MLLMs that jointly reason over video, audio, and text.
Given the retrieved subgraph GGRASP , we condition the
MLLM by concatenating the multimodal query qwith the
graph context. For each triplet (h, r, t)∈ G GRASP , we in-
clude the relation rtogether with the entity–description pairs⟨h, d h⟩and⟨t, dt⟩, where dhanddtare the refined descrip-
tions of handt, respectively. Formally, our graph-enhanced
generation is defined as follows:
paug=q∥
[
(h,r,t)∈G GRASP⟨h, d h⟩r−→ ⟨t, d t⟩
.(6)
By providing paugto the MLLMs, we inject query-
relevant, answer-useful knowledge and supply the inter-
entity relations together with detailed entity attributions,
which improves the MLLM’s reasoning capabilities.
4. Experiments
4.1. Experimental Setup
DatasetsTo highlight the diverse modality coverage of
M3KG-RAG, we evaluate on three multimodal tasks: Audio
QA, Video QA, and Audio-Visual QA. For Audio-QA, we
use AudioCaps-QA [ 45], which provides human-annotated
QA pairs built on AudioCaps corpus [ 25]. For Video-QA, we
adopt the VideoChatGPT (VCGPT) benchmark [ 36], which
is built on videos curated from ActivityNet [ 6]. For Audio-
Visual QA, we evaluate on the V ALOR [ 33] benchmark,
which explicitly requires joint reasoning over synchronized
audio and visual streams for response.
MLLMsWe apply our M3KG-RAG to three MLLMs ca-
pable of joint audio-visual understanding to assess its effec-
tiveness. Specifically, we employ VideoLLaMA2 [ 8] and
Qwen2.5-Omni [ 49], both advanced open-source models that
jointly process audio and visual streams and serve as strong
baselines for audio–visual reasoning. We further adopt GPT-
4o [21], a strong commercial model with substantial implicit
5

MLLM MethodAudio QA Video QA Audio-Visual QA
AudioCaps-QA VCGPT V ALOR
VideoLLaMA2None 43.13 39.09 25.66
Wikidata 43.58 38.58 26.43
VTKG 43.02 38.88 25.92
M2ConceptBase 42.19 39.31 25.93
V AT-KG 44.60 39.42 28.30
M3KG-RAG 53.23 39.92 29.25
Qwen2.5-OmniNone 49.00 42.21 32.42
Wikidata 49.78 40.82 30.28
VTKG 48.95 42.96 32.70
M2ConceptBase 49.78 42.78 32.31
V AT-KG 51.30 43.50 35.44
M3KG-RAG 60.77 44.35 44.67
Table 1.Overall performance.We report Model-as-Judge (M.J.) scores (higher is better). Across all benchmarks and for both MLLMs,
M3KG-RAG provides the largest and most consistent gains over the no-retrieval and MMKG-based baselines. The best results arebolded.
knowledge and capacity, to examine whether our method
remains effective even for such high-capacity models.
Baseline MethodsFollowing prior multimodal RAG
work [ 37], we compare M3KG-RAG against five baselines:
(i) None—the MLLM answers without external knowledge;
(ii) Wikidata [ 46] + naïve RAG with retrieval in a shared
embedding space [ 47,48] between the text KG and the mul-
timodal query; (iii) VTKG [ 26] + naïve RAG (image–text
MMKG), matching visual queries to images in MMKG via a
vision–language space (e.g., CLIP [ 38]) and audio queries in
the CLAP [ 48] audio–text space; (iv) M2ConceptBase [ 55],
following VTKG’s protocol; and (v) V AT-KG [ 37], eval-
uated under its released RAG protocol that accounts for
audio–visual streams.
Implementation DetailsWe implement the multi-hop
MMKG construction pipeline with a lightweight multi-agent
stack built on a single backbone LLM (Qwen3-8B) [ 50],
using only the training splits of our evaluation benchmarks.
In modality-wise retrieval, we set k= 5 and select the top-
kbest-matching items per query, then expand each to its
connected multi-hop subgraph. Since our benchmarks span
different audio–visual distributions, we set the modality-
wise distance threshold τand GRASP presence threshold η
separately per benchmark as follows: for AudioCaps-QA,
τ= 0.3 andηa= 0.5 ; for VCGPT, τ= 0.15 andηv= 1.5 ;
and for V ALOR, τ= 4.5 andηav= 1.2 . These values are
held constant within each benchmark across all experiments.
All experiments use a single NVIDIA H100 GPU. Note that
further details are provided in the supplementary material.
Evaluation MetricsWe implement two evaluation
schemes. First, since our benchmarks consist of open-
ended QA with free-form responses, we adopt an off-the-
shelf Model-as-Judge (M.J.) protocol [ 45], where an LLMjudge [ 16] scores each response. Second, in line with es-
tablished RAG evaluation [ 11,18,40], we report a win-rate
preference protocol where the LLM judge compares the two
responses (ours vs. a baseline) and selects the preferred
response based on multiple criteria. We make it reference-
aware by providing the judge with the reference answer
during comparison, which reduces verbosity bias and yields
a more reliable, multi-dimensional assessment.
4.2. Quantitative Results
We compare M3KG-RAG against text-KG and multimodal-
KG baselines on Audio-QA, Video-QA, and Audio-Visual
QA. The overall results are summarized in Table 1. Across
all benchmarks, M3KG-RAG yields significant gains over
the base MLLMs, indicating that modality-wise retrieval
and GRASP deliver knowledge that is both tightly aligned
with the query and directly useful for answering. In con-
trast, other baselines tend not to consistently improve the
MLLMs. Specifically, text KG with naïve RAG (Wikidata)
yields weak or even negative deltas, as retrieval ignores
the temporal nature of audio–visual queries, often retriev-
ing off-context neighbors and injecting noisy facts that do
not support the answer. Image-text KGs with naïve RAG
(VTKG, M2ConceptBase) partially account for visual cues
via images in MMKGs but still miss query dynamics, leading
to limited impact on response quality and occasional degra-
dation. V AT-KG, which considers audio–visual streams,
improves all baselines uniformly. However, its largely single-
hop MMKG captures only local, concept-level facts. The
MLLM therefore receives only shallow, fragmentary context,
so the knowledge implicitly encoded in the underlying multi-
modal data is only partially exploited, and performance gains
remain mostly marginal. In contrast, M3KG-RAG builds
multi-hop neighborhoods that aggregate temporally and se-
6

AudioCaps-QA VCGPT V ALOR
BaselineOursBaselineOursBaselineOurs
Baseline: None
Comprehensiveness 15.9%84.1%47.6%52.4%39.8%60.2%
Diversity 20.3%79.7%37.8%62.2%45.5%54.5%
Empowerment 14.0%86.0%42.1%57.9%40.1%59.9%
Overall 15.2%84.8%47.0%53.0%39.8%60.2%
Baseline: Wikidata
Comprehensiveness 14.9%85.1%48.3%51.7%40.3%59.7%
Diversity 22.4%77.6%47.4%52.6% 55.5%44.5%
Empowerment 12.0%88.0%39.6%60.4%40.8%59.2%
Overall 13.7%86.3%44.5%55.5%40.8%59.2%
Baseline: VTKG
Comprehensiveness 20.8%79.2%49.1%50.9%39.1%60.9%
Diversity 33.8%66.2%45.9%54.1%45.2%54.8%
Empowerment 21.2%78.8%46.6%53.4%39.2%60.8%
Overall 21.2%78.8%49.1%50.9%39.4%60.6%
Baseline: M2ConceptBase
Comprehensiveness 21.2%78.8%41.8%58.2%38.2%61.8%
Diversity 28.3%71.7%43.9%56.1%45.1%54.9%
Empowerment 19.7%80.3%44.6%55.4%38.6%61.4%
Overall 21.0%79.0%44.3%55.7%38.3%61.7%
Baseline: VAT-KG
Comprehensiveness 26.1%73.9%48.4%51.6%41.4%58.6%
Diversity 34.8%65.2%46.6%53.4%48.3%51.7%
Empowerment 24.3%75.7%43.5%56.5%42.1%57.9%
Overall 25.6%74.4%47.6%52.4%41.8%58.2%
Table 2.Win-rate comparison.Pairwise win rates (%) of each
baseline versus M3KG-RAG across three benchmarks and four
criteria. Columns show the preference rate of theBaselineand
Ours, with the higher win rate in each pair highlighted inbold.
mantically related evidence across modalities and, together
with modality-wise retrieval and GRASP, delivers query-
focused, answer-supporting knowledge that more faithfully
reflects the multimodal query. Consequently, M3KG-RAG
achieves notable gains over V AT-KG on every benchmark.
These findings become more pronounced with a stronger
commercial MLLM. As shown in Table 3, even with substan-
tial built-in knowledge, GPT-4o paired with M3KG-RAG
improves across all benchmarks and exhibits larger gains
than with V AT-KG, reinforcing that multi-hop evidence to-
gether with GRASP provides a diverse, answer-supporting
context that the model can exploit more effectively.
Win-rate preference results in Table 2 corroborate the M.J.
scores, showing consistent preference for M3KG-RAG over
baselines across benchmarks and criteria. We observe higher
Comprehensivenessbecause richer multi-hop evidence ag-
gregates the key entities and relations needed to answer the
query end-to-end, aided by refined entity descriptions for
clarity.Diversityimproves as the multi-hop MMKG offers
several distinct evidence chains, while pruning removes off-
topic or duplicate content.Empowermentbenefits from
a strictly query-relevant context that reduces hallucination
and steers the model toward concrete, answer-supporting
details rather than generic filler. Together, these effects yield
strongerOverallpreferences in pairwise comparisons.MLLM Method Audiocaps-QA VCGPT V ALOR
GPT-4o None 56.74 49.68 46.02
GPT-4o V AT-KG 57.70 51.49 55.86
GPT-4o M3KG-RAG59.17 53.05 56.53
Table 3.Performance on Commercial MLLM (GPT-4o).We
report M.J. scores (higher is better). The best result isbolded.
MLLMMethodM.J↑
Modality-Wise
RetrievalGRASP
Qwen2.5-Omni✗ ✗ 36.62
✓ ✗ 40.91
✗ ✓ 36.96
✓ ✓ 44.67
Table 4.Ablation on V ALOR.Checkmarks denote enabled com-
ponents. We report the M.J. score; combining Modality-Wise Re-
trieval and GRASP gives the best score. The best result isbolded.
4.3. Ablation Study
To explore the effectiveness of our design, we conduct abla-
tions of modality-wise retrieval and GRASP on the V ALOR,
which requires joint audio–visual reasoning, using Qwen2.5-
Omni [ 49] as the base MLLM. For modality-wise retrieval,
we remove the cross-modal links Lthat connect multimodal
items ( A,V) to the triplet set Tin the M3KG, yielding a
text-only KG. We then convert each triplet tinto a natural
sentence σ(t) and index them using the text encoder of the
multimodal embedding model [ 47,48], enabling retrieval
for audio-visual queries in a shared embedding space.
As shown in Table 4, using modality-wise retrieval alone
keeps retrieval within the query’s modality and reduces mis-
matched evidence. However, relying solely on similarity
search cannot verify entity-level relevance to the query or en-
sure that retrieved evidence supports the answer, leading to
limited performance gains. Using GRASP improves faithful-
ness via fine-grained pruning, yet the initial graph retrieved
from a text-only KG in a shared space is weakly aligned with
the audio–visual cues, yielding only modest improvements.
Combining both offers the largest gain: modality-wise
retrieval supplies candidates aligned with the query’s audio
and visual streams, and GRASP retains only triplets that
directly support the question and removes redundancy. Taken
together, modality-wise retrieval and grounded pruning are
complementary and jointly necessary for multimodal RAG.
4.4. Qualitative Results
Fig. 4 presents qualitative results across Audio, Video, and
Audio–Visual QA. With M3KG-RAG, the MLLM produces
more specific, context-faithful answers by grounding genera-
tion in multi-hop evidence and concise entity descriptions
from our modality-wise retrieval and GRASP.
In the case of Audio-QA, the context supplied by M3KG-
RAG directly supports answering. For example, for the
7

QuestionDescribe in detail what is visually and audibly happening in the video, including actions, objects, people, sounds, and environment. Be specific and concise.[Audio-Visual Question Answering][Video Question Answering]
Qwen2.5-Omni + M𝟑KG-RAGQwen2.5-OmniQuestion
What type of social setting is depicted in the audio recording?“restaurant.”“The social setting depicted in the audio recording is a casual and friendly environment, likely a gathering or a party at someone's house.”
Qwen2.5-OmniQwen2.5-Omni + M𝟑KG-RAGQuestionWhat happened when the game started?“The game started with a man in a blue shirt serving the ball.”“When the game started, a group of people were playing dodgeball on a court. The players were wearing casual clothes and were actively ...
Qwen2.5-OmniQwen2.5-Omni + M𝟑KG-RAGQwen2.5-OmniQwen2.5-Omni + M𝟑KG-RAG"The video features a close-up shot of a cylindrical object, which appears to be a speaker, lying on its side on a bright yellow, shaggy carpet. The speaker is metallic and has a sleek, modern design with a white cable extending from it. …"In the video, there's a playing silver hairdryer lying on a yellow carpet. The hairdryer is emitting a relaxing dryer sound. There areno people or other objects visible in the video. The environment is just the carpet and the hairdryer …
( A sound of casual social gathering, likely a dinner party or a social gathering with friends )( A sound of duck in wetland or a park with a body of water )
"In the video, there is a group of men on a stage. They are wearing red checkered skirts and playing instruments. Two of them are playing Scottish bagpipes, and two are playing drums. The stage is lit up with spotlights, creating a bright and  … "The video features a group of four men dressed in traditional Scottish attire performing on stage. They are wearing kilts, sporran pouches, and sporran belts. Each man is playing a different instrument: two are playing bagpipes, one is playing a drum, and the fourth is playing a keyboard … 
Qwen2.5-Omni + M𝟑KG-RAGQwen2.5-OmniQuestion
What type of environment is likely to be recorded in the audio file?“park.”"The environment likely to be recorded in the audio file is a natural habitat where ducks are present, such as a pond, park, or wetland area.”
[Audio Question Answering]Figure 4.Qualitative results on various Question Answering tasks.Incorrect and insufficient model responses are highlighted in red,
while correct and sufficient responses are highlighted in blue.
question “What type of social setting is depicted in the audio
recording?” the base model responds “restaurant,” which
is loosely related yet misaligned with the asked social set-
ting and lacks sufficient detail. With M3KG-RAG, query-
conditioned retrieval selects the correct context, and GRASP
passes only answer-useful cues, producing “a gathering or
a party at someone’s house.” Likewise, for the environment
clip with water, the base MLLM overlooks water-related
acoustic attributes, whereas our retrieval highlights duck
calls and water ambience, producing “a natural habitat with
ducks, such as a pond, park, or wetland.”
In the case of Video-QA, the graph-enhanced context
enables precise answering. While the base model misses the
action occurring in the video, our RAG conditions retrieval
on the visual stream, identifies the scene as dodgeball, and
supplies action-aware context that enables a precise answer.
Retrieval conditioned on both audio and visual sharpens
predictions in Audio-Visual QA. For the stage-performance
clip, the model hallucinates a keyboard player. With M3KG-
RAG, based on the retrieved query-related context, the model
produces correct stage context and instrument roles (two
bagpipes, two drums). For the hair-dryer clip, the modeltags a "cylinder object" and misclassifies it as a speaker.
On the other hand, M3KG-RAG conditions retrieval on
the audio-visual query and passes context that links the
buzzing audio to a running hair dryer, enabling a precise re-
sponse. These results demonstrate that modality-conditioned
retrieval with GRASP supplies on-topic, fine-grained context
that improves the specificity and faithfulness of answers.
5. Conclusion
We introduce M3KG-RAG, a novel graph-augmented mul-
timodal RAG framework for enhancing audio–visual rea-
soning in MLLMs. Our lightweight multi-agent pipeline
constructs a multi-hop multimodal knowledge graph that sup-
ports precise modality-wise retrieval and robust knowledge
grounding. Furthermore, the proposed GRASP (Grounded
Retrieval And Selective Pruning) scores triplets with visual
and audio foundation models and retains only query-relevant,
answer-supporting evidence. Extensive evaluation across di-
verse multimodal benchmarks highlights the effectiveness of
M3KG-RAG, with consistent performance gains over strong
baselines. We believe M3KG-RAG will serve as a practical
foundation for future research in multimodal RAG.
8

References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad,
Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko
Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4
technical report.arXiv preprint arXiv:2303.08774, 2023. 1, 2
[2]Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine
Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch,
Katherine Millican, Malcolm Reynolds, et al. Flamingo: a
visual language model for few-shot learning. InAdvances in
Neural Information Processing Systems, 2022. 2
[3]Rohan Anil, Andrew M Dai, Orhan Firat, Melvin John-
son, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri,
Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2
technical report.arXiv preprint arXiv:2305.10403, 2023. 2
[4]Xiaohe Bo, Zeyu Zhang, Quanyu Dai, Xueyang Feng, Lei
Wang, Rui Li, Xu Chen, and Ji-Rong Wen. Reflective multi-
agent collaboration based on large language models.Ad-
vances in Neural Information Processing Systems, 37:138595–
138631, 2024. 3
[5]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George Bm
Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc,
Aidan Clark, et al. Improving language models by retriev-
ing from trillions of tokens. InInternational Conference on
Machine Learning, pages 2206–2240. PMLR, 2022. 2
[6]Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and
Juan Carlos Niebles. Activitynet: A large-scale video bench-
mark for human activity understanding. InProceedings of the
ieee conference on computer vision and pattern recognition,
pages 961–970, 2015. 5, 2
[7]Honglie Chen, Weidi Xie, Andrea Vedaldi, and Andrew Zis-
serman. Vggsound: A large-scale audio-visual dataset. In
ICASSP 2020-2020 IEEE International Conference on Acous-
tics, Speech and Signal Processing (ICASSP), pages 721–725.
IEEE, 2020. 3
[8]Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin
Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang
Luo, Deli Zhao, et al. Videollama 2: Advancing spatial-
temporal modeling and audio understanding in video-llms.
arXiv preprint arXiv:2406.07476, 2024. 2, 5
[9]Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong
Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi
Tang, et al. Kimi-audio technical report.arXiv preprint
arXiv:2504.18425, 2025. 2
[10] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-
hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The
llama 3 herd of models.arXiv e-prints, pages arXiv–2407,
2024. 2
[11] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley,
Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropoli-
tansky, Robert Osazuwa Ness, and Jonathan Larson. From
local to global: A graph rag approach to query-focused sum-
marization.arXiv preprint arXiv:2404.16130, 2024. 1, 2, 6,
4
[12] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang,
Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A sur-vey on rag meeting llms: Towards retrieval-augmented large
language models. InProceedings of the 30th ACM SIGKDD
conference on knowledge discovery and data mining, pages
6491–6501, 2024. 2
[13] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
Precise zero-shot dense retrieval without relevance labels. In
Proceedings of the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages
1762–1777, 2023. 2
[14] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren
Jansen, Wade Lawrence, R. Channing Moore, Manoj Plakal,
and Marvin Ritter. Audio set: An ontology and human-labeled
dataset for audio events. In2017 IEEE International Confer-
ence on Acoustics, Speech and Signal Processing (ICASSP),
pages 776–780, 2017. 2
[15] Sreyan Ghosh, Sonal Kumar, Chandra Kiran Reddy Evuru,
Ramani Duraiswami, and Dinesh Manocha. Recap: Retrieval-
augmented audio captioning. InICASSP 2024-2024 IEEE
International Conference on Acoustics, Speech and Signal
Processing (ICASSP), pages 1161–1165. IEEE, 2024. 2
[16] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Ab-
hinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al.
The llama 3 herd of models.arXiv preprint arXiv:2407.21783,
2024. 6, 4
[17] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi
Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning
capability in llms via reinforcement learning.arXiv preprint
arXiv:2501.12948, 2025. 2
[18] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. LightRAG: Simple and fast retrieval-augmented gen-
eration. InFindings of the Association for Computational Lin-
guistics: EMNLP 2025, pages 10746–10761, Suzhou, China,
2025. Association for Computational Linguistics. 1, 2, 6, 4
[19] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe
Zhou, and Yu Su. From RAG to memory: Non-parametric
continual learning for large language models. InForty-second
International Conference on Machine Learning, 2025. 2
[20] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and
Ming-Wei Chang. Realm: retrieval-augmented language
model pre-training. InProceedings of the 37th International
Conference on Machine Learning, pages 3929–3938, 2020. 2
[21] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman,
Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda,
Alan Hayes, Alec Radford, et al. Gpt-4o system card.arXiv
preprint arXiv:2410.21276, 2024. 2, 5
[22] Gautier Izacard and Edouard Grave. Leveraging passage
retrieval with generative models for open domain question
answering.arXiv preprint arXiv:2007.01282, 2020. 2
[23] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju
Hwang. VideoRAG: Retrieval-augmented generation over
video corpus. InFindings of the Association for Computa-
tional Linguistics: ACL 2025, pages 21278–21298, Vienna,
Austria, 2025. Association for Computational Linguistics. 2
[24] Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale
similarity search with gpus.IEEE Transactions on Big Data,
7(3):535–547, 2019. 4
9

[25] Chris Dongjoo Kim, Byeongchang Kim, Hyunmin Lee, and
Gunhee Kim. Audiocaps: Generating captions for audios in
the wild. InProceedings of the 2019 Conference of the North
American Chapter of the Association for Computational Lin-
guistics: Human Language Technologies, Volume 1 (Long
and Short Papers), pages 119–132, 2019. 3, 5, 2
[26] Jaejun Lee, Chanyoung Chung, Hochang Lee, Sungho Jo,
and Joyce Whang. Vista: Visual-textual knowledge graph
representation learning. InFindings of the association for
computational linguistics: EMNLP 2023, pages 7314–7328,
2023. 1, 6, 4
[27] Junlin Lee, Yequan Wang, Jing Li, and Min Zhang. Multi-
modal reasoning with multimodal knowledge graph.arXiv
preprint arXiv:2406.02030, 2024. 1, 3
[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike
Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-
augmented generation for knowledge-intensive nlp tasks.Ad-
vances in Neural Information Processing Systems, 33:9459–
9474, 2020. 1, 2
[29] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li,
Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Zi-
wei Liu, et al. Llava-onevision: Easy visual task transfer.
arXiv preprint arXiv:2408.03326, 2024. 2
[30] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo
Li, Wei Li, Zejun MA, and Chunyuan Li. LLaV A-neXT-
interleave: Tackling multi-image, video, and 3d in large mul-
timodal models. InThe Thirteenth International Conference
on Learning Representations, 2025. 2
[31] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-
2: bootstrapping language-image pre-training with frozen
image encoders and large language models. InProceedings
of the 40th International Conference on Machine Learning,
pages 19730–19742, 2023. 2
[32] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. Visual instruction tuning. InProceedings of the 37th
International Conference on Neural Information Processing
Systems, pages 34892–34916, 2023. 2
[33] Jing Liu, Sihan Chen, Xingjian He, Longteng Guo, Xinxin
Zhu, Weining Wang, and Jinhui Tang. Valor: Vision-audio-
language omni-perception pretraining model and dataset.
IEEE Transactions on Pattern Analysis and Machine Intelli-
gence, 2024. 5, 2, 6
[34] Junming Liu, Siyuan Meng, Yanting Gao, Song Mao, Pinlong
Cai, Guohang Yan, Yirong Chen, Zilin Bian, Ding Wang, and
Botian Shi. Aligning vision to language: Annotation-free
multimodal knowledge graph construction for enhanced llms
reasoning. InProceedings of the IEEE/CVF International
Conference on Computer Vision, pages 981–992, 2025. 1
[35] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao
Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang,
Hang Su, et al. Grounding dino: Marrying dino with grounded
pre-training for open-set object detection. InEuropean con-
ference on computer vision, pages 38–55. Springer, 2024. 2,
4, 3, 5
[36] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and
Fahad Khan. Video-chatgpt: Towards detailed video un-derstanding via large vision and language models. InPro-
ceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages
12585–12602, 2024. 5, 6
[37] Hyeongcheol Park, Jiyoung Seo, MinHyuk Jang, Hogun Park,
Ha Dam Baek, Gyusam Chang, Hyeonsoo Im, and Sang-
pil Kim. Vat-kg: Knowledge-intensive multimodal knowl-
edge graph dataset for retrieval-augmented generation.arXiv
preprint arXiv:2506.21556, 2025. 1, 2, 3, 6, 4
[38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. Learning transferable visual
models from natural language supervision. InProceedings
of the 38th International Conference on Machine Learning,
pages 8748–8763. PMLR, 2021. 6
[39] Sameera Ramasinghe, Violetta Shevchenko, Gil Avraham,
and Ajanthan Thalaiyasingam. Accept the modality gap:
An exploration in the hyperbolic space. InProceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 27263–27272, 2024. 1
[40] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei
Yin, and Chao Huang. Videorag: Retrieval-augmented gen-
eration with extreme long-context videos.arXiv preprint
arXiv:2502.01549, 2025. 1, 6, 4
[41] Chaitanya Sharma. Retrieval-augmented generation: A com-
prehensive survey of architectures, enhancements, and ro-
bustness frontiers.arXiv preprint arXiv:2506.00054, 2025.
1
[42] Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian
Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. Salmonn:
Towards generic hearing abilities for large language models.
arXiv preprint arXiv:2310.13289, 2023. 2
[43] Qwen Team et al. Qwen2 technical report.arXiv preprint
arXiv:2407.10671, 2(3), 2024. 2
[44] Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-
Viet Pham, Barry O’Sullivan, and Hoang D Nguyen. Multi-
agent collaboration mechanisms: A survey of llms.arXiv
preprint arXiv:2501.06322, 2025. 3
[45] Bin Wang, Xunlong Zou, Geyu Lin, Shuo Sun, Zhuohan Liu,
Wenyu Zhang, Zhengyuan Liu, AiTi Aw, and Nancy Chen.
Audiobench: A universal benchmark for audio large language
models. InProceedings of the 2025 Conference of the Nations
of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long
Papers), pages 4297–4316, 2025. 5, 6, 4
[46] Xiaozhi Wang, Tianyu Gao, Zhaocheng Zhu, Zhengyan
Zhang, Zhiyuan Liu, Juanzi Li, and Jian Tang. Kepler: A
unified model for knowledge embedding and pre-trained lan-
guage representation.Transactions of the Association for
Computational Linguistics, 9:176–194, 2021. 6, 3
[47] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He,
Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong
Shi, et al. Internvideo2: Scaling foundation models for mul-
timodal video understanding. InEuropean Conference on
Computer Vision, pages 396–416. Springer, 2024. 4, 6, 7
[48] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor
Berg-Kirkpatrick, and Shlomo Dubnov. Large-scale con-
10

trastive language-audio pretraining with feature fusion and
keyword-to-caption augmentation. InICASSP 2023-2023
IEEE International Conference on Acoustics, Speech and
Signal Processing (ICASSP), pages 1–5. IEEE, 2023. 4, 6, 7
[49] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He,
Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang,
et al. Qwen2. 5-omni technical report.arXiv preprint
arXiv:2503.20215, 2025. 2, 5, 7, 6
[50] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang,
Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa
Zhu, et al. Qwen3-omni technical report.arXiv preprint
arXiv:2509.17765, 2025. 5, 6
[51] Xuenan Xu, Ziyang Ma, Mengyue Wu, and Kai Yu. Towards
weakly supervised text-to-audio grounding.IEEE Transac-
tions on Multimedia, 2024. 2, 5, 3
[52] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan
Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang,
Chenxu Lv, et al. Qwen3 technical report.arXiv preprint
arXiv:2505.09388, 2025. 2, 1, 3
[53] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui,
Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He,
et al. Minicpm-v: A gpt-4v level mllm on your phone.arXiv
preprint arXiv:2408.01800, 2024. 1
[54] Woongyeong Yeo, Kangsan Kim, Soyeong Jeong, Jin-
heon Baek, and Sung Ju Hwang. Universalrag: Retrieval-
augmented generation over corpora of diverse modalities and
granularities.arXiv preprint arXiv:2504.20734, 2025. 1, 2, 4
[55] Zhiwei Zha, Jiaan Wang, Zhixu Li, Xiangru Zhu, Wei Song,
and Yanghua Xiao. M2conceptbase: A fine-grained aligned
concept-centric multimodal knowledge base. InProceedings
of the 33rd ACM International Conference on Information
and Knowledge Management, pages 3113–3123, 2024. 1, 2,
3, 6, 4
[56] Bowen Zhang and Harold Soh. Extract, define, canonicalize:
An llm-based framework for knowledge graph construction.
InProceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing, pages 9820–9836, 2024. 3
[57] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng
Yuan, Huachi Zhou, Zijin Hong, Hao Chen, Yilin Xiao,
Chuang Zhou, Junnan Dong, et al. A survey of graph retrieval-
augmented generation for customized large language models.
arXiv preprint arXiv:2501.13958, 2025. 1
[58] Zikang Zhang, Wangjie You, Tianci Wu, Xinrui Wang, Juntao
Li, and Min Zhang. A survey of generative information ex-
traction. InProceedings of the 31st International Conference
on Computational Linguistics, pages 4840–4870, 2025. 3
11

M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced
Retrieval-Augmented Generation
Supplementary Material
Overview
This supplementary material provides additional implemen-
tation details, experimental results, and qualitative analyses
for our proposed framework, M3KG-RAG.
•In Sec. A, we describe the implementation of M3KG-RAG
in detail, including the construction of the M3KG and the
multimodal RAG framework.
•In Sec. B, we present additional analyses of our multi-
modal RAG framework, including hyperparameter sensi-
tivity and ablations over key components with their com-
putational cost.
•In Sec. C, we provide additional qualitative results of
M3KG-RAG on multimodal benchmarks, including win-
rate evaluations.
•In Sec. D, we discuss the limitations of M3KG-RAG and
potential directions for improving robustness.
A. Extended Implementation Details
A.1. M3KG Construction
In this section, we describe how M3KG is built using a
lightweight multi-agent pipeline, outlining the roles and
designs of each agent. We also summarize the multimodal
corpora used for construction.
A.1.1. Multi-Agents
We provide detailed descriptions and prompt designs for the
multi-agent system used in M3KG construction, including
rewriter,extractor,normalizer,searcher,selector,refiner,
andinspector.
RewriterTherewritertransforms generic textual descrip-
tions from the raw multimodal corpus into knowledge-
intensive captions that are more informative for MLLMs.
For each data point, it leverages the crawled YouTube title
and description to inject unfamiliar concepts and background
knowledge that are not explicitly captured in the original text
caption. The detailed prompt design for therewriteragent is
provided in Table 5.
ExtractorTheextractortakes the rewritten, knowledge-
intensive captions produced by therewriterand extracts
structured knowledge in the form of triplets. Building on
the LLM-based open information extraction prompt used in
V AT-KG [ 37], we design the prompt for theextractoragent,
as shown in Table 6.
NormalizerThenormalizeroperates on the head and tail
entities in the triplets extracted by theextractorand standard-Rewriter agent
<system prompt>
You refine video captions using the video’s Title and Descrip-
tion.
ORIGINAL CAPTION always has priority. If Title/Description
are not clearly referring to the SAME scene/object/action, output
the ORIGINAL CAPTION exactly.
Allowed edits ONLY when clearly aligned: replace generic
nouns with specific terms (breed/species/instrument/model/
place/role), or add 1 short factual attribute. Keep the meaning
and keep the length roughly similar (±20%).
Disallowed: inventing new events, numbers, counts, or specula-
tive facts; adding ads/URLs/hashtags.
Keep the original style. Make the merge natural (not a concat).
Output: ONLY the final caption in English (no labels or expla-
nations).
<user prompt>
Title: {TITLE}
Description: {DESCRIPTION}
ORIGINAL CAPTION: {ORIGINAL_CAPTION}
Output:
Table 5. Prompt template for therewriteragent.
izes them into canonical, searchable concepts. The prompt
design for thenormalizeragent is provided in Table 7.
SearcherThesearchertakes the normalized entities pro-
duced by thenormalizerand queries external knowledge re-
sources (e.g., Wikipedia, Wiktionary) to obtain encyclopedic
descriptions for each concept. If it cannot find a descrip-
tion from these sources, it invokes an LLM callback [ 52] to
generate a brief description for the entity, leveraging context-
enriched caption. The prompt designs for the LLM callback
of thesearcheragent are provided in Table 8.
SelectorTheselectortakes multiple candidate descriptions
for each entity and uses the context-enriched caption pro-
duced by therewriteras context to select the most appropri-
ate description. The prompt design for theselectoragent is
provided in Table 9.
RefinerTherefinertakes the descriptions selected for each
entity’s canonical form and refines them to better match the
semantics and surface form of the original entity mention,
while preserving the underlying factual content. The prompt
design for therefineragent is provided in Table 10.
InspectorTheinspectorserves as a quality-control agent
that implements the self-reflection loop in our construction
pipeline. For each entity description produced either by the
1

Extractor agent
<system prompt>
You are an expert in extracting structured knowledge from text.
Given a video caption, extract all subject-relationship-object
triples in the form (h, r, t).
Extract multiple (h, r, t) triples if applicable.
Each triple must be meaningful and correctly represent relation-
ships in the text.
Output format: ONE triple per line as (h, r, t). No extra text or
explanation.
Use concise surface forms that appear (or are directly implied)
in the caption.
Do not invent entities or facts not supported by the caption.
Language: English only.
<user prompt>
Caption: {CAPTION}
Output:
Table 6. Prompt template for theextractoragent.
Normalizer agent
<system prompt>
You output exactly ONE KB-searchable concept noun phrase
(Wikipedia-title-like). Plain text only, no quotes or extra words.
Output MUST be in English. If CONCEPT is not in English,
translate the noun phrase into an English Wikipedia-style title;
transliterate proper names if needed (do not add extra words).
Otherwise, use ONLY words from CONCEPT ; keep order; you
may DROP words (no inventions/translation).
If the output noun phrase is in plural form, convert it to its
singular form (e.g., “dogs”→“dog”, “empires”→“empire”).
Must be a NOUN PHRASE; remove wrappers like “how to”,
“what is”, guides/tips, articles, years.
Prefer inner object NP (e.g., “history of jazz music” →“jazz
music”).
Prefer canonical/proper names; preserve original casing.
<user prompt>
CONCEPT: {CONCEPT}
Output:
Table 7. Prompt template for thenormalizeragent.
refineror by the LLM callback of thesearcher, theinspector
assigns a plausibility score on a 0–10scale, conditioned
on the context-enriched caption from Step 1. Descriptions
scoring below 7are sent back to the corresponding agent
for regeneration and re-scoring; we allow at most three such
iterations, after which persistently low-scoring descriptions
are discarded to avoid injecting low-quality facts into the
graph, while higher-scoring ones are accepted. The prompt
design for theinspectoragent is provided in Table 11.
A.1.2. Corpora for M3KG Construction
We construct M3KG from the training splits of three mul-
timodal corpora: AudioCaps [ 25], ActivityNet [ 6], andSearcher agent (LLM callback)
<system prompt>
You are an encyclopedic writer.
Write a neutral, generic, 1–2 sentence encyclopedic description
of a concept.
Use the caption ONLY to disambiguate the intended sense (do
not describe the scene).
Return ONLY the final description sentences in plain text—no
labels, no lists, no quotes, no extra commentary.
<user prompt>
Concept: {CONCEPT}
Caption (sense disambiguation only): {CAPTION}
Output:
Table 8. Prompt template for the LLM callback of thesearcher
agent.
Selector agent
<system prompt>
You are a selector for concept descriptions.
The CAPTION is from the same video, and the CONCEPT
refers to the concept that appears or is mentioned in this video.
Choose ONE candidate whose encyclopedic sense best matches
that concept in this video’s caption.
Use ONLY the CAPTION to resolve meaning (sense); do not
import outside facts.
Return EXACTLY one candidate’s text verbatim; do not edit,
merge, summarize, quote, or label it.
If multiple candidates are similarly valid, prefer the most spe-
cific non-speculative candidate.
If none clearly fits, choose the safest generic candidate (least
speculative).
Return ONLY the chosen candidate text (no extra text).
<user prompt>
CONCEPT: {CONCEPT}
CAPTION: {CAPTION}
CANDIDATES: {ENUMERATED_CANDIDATES}
Output:
Table 9. Prompt template for theselectoragent.
V ALOR [ 33]. For graph construction, we use only the raw
audio-visual content and their associated captions, without
accessing any QA annotations.
AudioCapsAudioCaps is an audio captioning corpus built
on 10-second clips from AudioSet [ 14] YouTube videos,
where each clip is paired with human-written natural lan-
guage descriptions of the acoustic scene and salient sound
events.
ActivityNetActivityNet is a large-scale video benchmark
of untrimmed videos covering diverse human activities, an-
notated with temporal activity boundaries and class labels.
In our construction pipeline, we segment each untrimmed
video into temporally localized clips using the provided an-
2

Refiner agent
<system prompt>
You are a refiner for concept descriptions.
Adapt the selected description so it fits the original con-
cept phrasing, preserving the original meaning and keeping
the content of the selected description as intact as possible
(minimal wording changes only—e.g., adjust possessives like
“my/our/their”, determiners, and surface phrasing to align with
the original concept).
Do NOT add, remove, or invent facts beyond what is in the
selected description.
Keep the meaning unchanged; only adapt phrasing to match the
original concept.
Concise: 1–2 sentences, plain text
(no lists/quotes/markdown/meta).
Do NOT output any reasoning.
Return ONLY the rewritten description sentences.
<user prompt>
Concept (original phrasing): {ORIGINAL_CONCEPT}
Searchable concept (KB term): {SEARCHABLE_CONCEPT}
Selected description (about the searchable concept):
{SELECTED_DESCRIPTION}
Output:
Table 10. Prompt template for therefineragent.
Inspector agent
<system prompt>
You are a judge that scores how well an encyclopedic DESCRIP-
TION matches the intended sense of a CONCEPT.
Sense scoring only.
Score 0–10 how well the DESCRIPTION’s encyclopedic sense
matches the intended sense of the CONCEPT (0 = differ-
ent/irrelevant sense, 10 = perfect sense match).
Output a single integer 0–10 with no extra text.
<user prompt>
CONCEPT: {CONCEPT}
DESCRIPTION: {DESCRIPTION}
OUTPUT:
Table 11. Prompt template for theinspectoragent.
notations, and use the corresponding activity class label for
each clip as a text signal when building M3KG.
V ALORV ALOR is a multimodal dataset of short video clips
with synchronized audio and human-authored audio-visual
captions, providing closely aligned triplets of vision, audio,
and text. We use the V ALOR-32K variant for constructing
M3KG.
A.2. Multimodal RAG Framework
In this section, we provide additional details of the multi-
modal RAG framework used in our experiments. Given a
multimodal query, we first perform modality-wise retrievalLLM-based GRASP filter
You are a selector that removes only unnecessary triples for
answering the query.
Keep triples that could be helpful to answer the query.
Remove triples that are clearly irrelevant, contradictory to the
query, or redundant duplicates.
When uncertain, prefer KEEPING the triple.
Preserve the ORIGINAL ORDER of kept indices (do NOT
rerank).
Query: {QUERY}
Triplets: {TRIPLETS}
Table 12. Instruction used for the LLM-based filter in GRASP over
retrieved triplets.
Multimodal RAG Prompt
You are a multimodal QA assistant. Prioritize PRIMARY evi-
dence from the input modalities you perceive.
Use the retrieved triples BELOW only as optional hints when
they are CLEARLY observed or corroborated in the input.
Procedure:
1) Detect whether any triple’s context appears in the input (enti-
ties, attributes, actions, time/place cues).
2) If matched, integrate the FULL triple (head, relation, tail)
into the answer, and enrich with head_desc/tail_desc.
– Do NOT contradict the primary evidence; if conflict exists,
ignore the triple.
3) If no triple is confidently matched, answer from the primary
evidence only.
Query : {QUERY}
Retrieved Triples : {TRIPLES_BLOCK}
Triple Format : [i] head={h} | relation={r} | tail={t} ||
head_description={hd} | tail_description={td}
Answer :
Table 13. Graph-augmented generation template used to instantiate
Eq. (6) in our multimodal RAG framework.
over M3KG. We then apply GRASP, which leverages mul-
timodal grounding models [ 35,51] and an LLM-based fil-
ter implemented with Qwen3-8B [ 52] using the instruc-
tion in Table 12 to obtain a compact set of query-relevant
and answer-supportive triplets, and finally inject this evi-
dence into the MLLM using the graph-augmented generation
scheme in Eq. (6) of the main paper. In practice, Eq. (6) is
realized by using the template summarized in Table 13.
A.2.1. Baselines for Multimodal RAG
Following prior multimodal RAG work [ 37], we compare
M3KG-RAG against four knowledge-graph baselines cou-
pled with RAG, in addition to theNonesetting where the
MLLMs answer without external knowledge.
Wikidata5MWikidata5M [ 46] is a million-scale text-only
knowledge graph constructed from Wikidata entities and rela-
tions aligned with their textual descriptions from Wikipedia.
3

Reference-Aware Win-rate Prompt
You will evaluate two answers to the same question using a Reference Answer. Base every judgment solely on alignment to the
Reference and the Question; do not reward verbosity or speculative content.
Question: {QUESTION}
Reference Answer (trusted ground truth): {REFERENCE}
Answer 1: {ANSWER_1}
Answer 2: {ANSWER_2}
Evaluate on the following criteria and return JSON in the exact schema below.
- Comprehensiveness: Which answer correctly covers more of the Reference’s essential points (paraphrase allowed) with fewer
mistakes or omissions? Do not reward length; penalize unsupported/contradictory claims.
- Diversity: Which answer offers greater variety in organizing the Reference’s facts (e.g., visual vs. audio facets) while avoiding new
attribute categories not stated or trivially entailed?
- Empowerment: Which answer better enables understanding or action through clear, concise, and reference-aligned guidance (no
filler, no meandering)?
- Overall Winner: Choose the answer that is most faithful to the Reference, with stronger correct coverage and clearer, more concise
presentation. Break ties by (1) correctness/faithfulness, (2) coverage, (3) concision/clarity.
Table 14. Reference-aware win-rate comparison template used for LLM-judged preferences.
VTKGVTKG [ 26] is an image-text multimodal knowledge
graph that augments a textual KG with visual evidence by
attaching images to entities and relational triples, together
with short textual descriptions. This design provides en-
tity–relation graphs grounded in visual examples, enabling
concept nodes to be linked not only by symbolic relations
but also by associated images.
M2ConceptBaseM2ConceptBase [ 55] is a concept-centric
multimodal knowledge base that represents each concept as
a node with multiple aligned visual examples and a detailed
textual description. It is explicitly designed to provide fine-
grained, cross-modal concept knowledge that can be passed
to MLLMs as grounded external evidence, helping mitigate
hallucinated or semantically inconsistent predictions.
V AT-KGV AT-KG [ 37] is a knowledge-intensive multimodal
knowledge graph that jointly integrates visual, audio, and tex-
tual signals into a unified concept-centric graph. Each triplet
is linked to multimodal evidence and enriched with concept
descriptions, providing an audio-visual KG backbone tai-
lored for retrieval-augmented generation under multimodal
queries.
A.2.2. Evaluation Protocol
As described in the main paper, our benchmarks consist
of open-ended QA with free-form responses, so we adopt
an off-the-shelf Model-as-Judge (M.J.) metric [ 45], where
an LLM judge [ 16] scores each generated answer on a 0-5
scale given the query and reference answer and reports the
resulting score on a 0-100 scale.
In addition, we report a pairwise win-rate between M3KG-
RAG and each baseline, following RAG evaluation pro-
tocols based on LLM preferences [ 11,18,40]. Unlike
prior work that compares two LLM-generated answers,
we make the win-rate judge reference-aware by provid-
ing the reference answer alongside the two candidates.This helps reduce verbosity bias (overly favoring longer re-
sponses) and discourages rewarding merely plausible but
unsupported generations, leading to a more faithful and
stable preference signal. The exact evaluation instruction
for the reference-aware win-rate judge is provided in Ta-
ble 14. The judge compares the two answers according to
three criteria—Comprehensiveness,Diversity, andEmpow-
erment—and selects a preferred answer for each criterion.
Based on these per-criterion preferences, it then decides
which answer is preferred overall for each query.
B. Additional Analysis
B.1. Hyperparameter Sensitivity Analysis
Our multimodal RAG framework has two scalar hyperpa-
rameters: (i) the modality-wise retrieval distance threshold τ
and (ii) the GRASP presence score threshold η. Both control
how much knowledge is injected into the MLLMs and there-
fore may affect downstream QA performance. To assess the
robustness of our framework to these choices, we conduct
a sensitivity study on the V ALOR benchmark by varying τ
andη avand measuring the resulting M.J. scores.
Modality-wise distance threshold τFor each query, we
embed it into the modality-specific representation spaces
of M3KG and compute distances to candidate items. Con-
cretely, an audio-only query is compared against audio items
in M3KG within the audio embedding space, and a visual-
only query is compared against visual items within the visual
embedding space. For an audio-visual query, we concatenate
its audio and visual embeddings and match it to audio-visual
items in the joint concatenated space. In each active modal-
itym, we compute an L2 distance d(qm, xm)between the
query representation qmand an item xmfrom M3KG, and
use it to perform top- kretrieval. We then apply the distance
threshold τto these kcandidates and keep only items with
4

40.7440.8744.8741.0840.0935.4432.2430.0034.0038.0042.0046.00
1.53.04.56.07.5M³KG-RAGVAT-KGNone
Distance Threshold 𝝉𝜼𝒂𝒗=𝟏.𝟐
44.6542.9944.8737.2534.4535.4432.2430.0034.0038.0042.0046.00
0.70.91.21.51.8Presence Score Threshold 𝜼𝒂𝒗𝝉=𝟒.𝟓Model-As-JudgeScoreModel-As-JudgeScoreFigure 5.Sensitivity analysis.M.J. score on V ALOR versus
modality-wise distance threshold τ(top) and GRASP presence
thresholdη av(bottom).
d(qm, xm)≤τ . The remaining retrieved items are lifted
into the graph via Eq. (3) in the main paper.
To understand how the choice of τaffects QA perfor-
mance, we conduct a sensitivity study on the V ALOR bench-
mark by varying τ∈ {1.5,3.0,4.5,6.0,7.5} while fixing
ηav= 1.2 , and visualize the resulting M.J. scores in the
top plot of Fig. 5. As shown in the figure, the M.J. score is
maximized at τ= 4.5 , while both smaller and larger thresh-
olds yield lower scores. Nonetheless, across all tested values
ofτ, M3KG-RAG consistently outperforms the baselines.
When τis set too small, only very close items are retained,
so the retrieved subgraphs become overly sparse and may
not provide sufficient evidence for the MLLM. Conversely, a
largeτallows many more distant items to pass the filter and
expands the subgraph, but also introduces multi-hop nodes
that are only weakly related to the query, increasing the risk
of noisy or distracting knowledge. These observations are
consistent with the intended role of τin balancing coverage
and noise in modality-wise retrieval.
GRASP presence score threshold ηGRASP assigns a
query-conditioned presence score s(t|q) to each triplet tGRASP Component GPU VRAM (GB) Avg time / query (s) M.J.
None 23.0 4.30 40.91
GDino (η v=0.8) 23.7 5.75 41.35
TAG (η a=0.4) 23.6 4.48 41.70
GDino + TAG (η av=1.2) 24.2 6.02 42.96
GDino + TAG + LLM Filter 39.8 7.0244.87
Table 15.Ablation on GRASP Components.We report GPU
VRAM usage, average inference time per query, and M.J. score.
in the retrieved subgraph using an off-the-shelf multimodal
grounding model [ 35,51]. We then apply the presence score
threshold ηand prune triplets whose scores fall below it,
keeping only those withs(t|q)≥η.
To examine how ηaffects QA performance, we fix
τ= 4.5 and vary ηav∈ {0.7,0.9,1.2,1.5,1.8} on the
V ALOR benchmark, visualizing the resulting M.J. scores
in the bottom plot of Fig. 5. The performance is highest at
ηav= 1.2 , with only minor differences between ηav= 0.7 ,
0.9, and 1.2, but it drops noticeably once ηincreases to
1.5and1.8. This pattern suggests that moderate grounding-
based pruning is beneficial, whereas overly aggressive thresh-
olds remove many triplets that still carry useful evidence,
leaving the MLLM with an under-informative subgraph.
B.2. Additional Ablation Studies
In Sec. 4.3 of the main paper, we ablate modality-wise
retrieval and GRASP as a whole. We further decompose
GRASP into its three submodules in the audio-visual setting:
visual grounding with GroundingDINO [ 35] (GDino), audio
grounding with TAG [ 51], and the final LLM-based filtering
stage. Tab. 15 reports GPU VRAM, average inference time
per query, and M.J. scores on the V ALOR benchmark as we
progressively enable these components, where GPU VRAM
is measured after loading each additional module.
Starting from the configuration without GRASP, adding
either GDino or TAG alone yields small but consistent gains
over the base M3KG-RAG model (from 40.91 to 41.35 and
41.70 M.J., respectively). Using both grounding modules
together further improves performance to 42.96 M.J., indicat-
ing that audio and visual grounding provide complementary
benefits. Importantly, the GPU memory footprint remains
almost unchanged when moving from a single grounding
module to both (about 23.6–24.2 GB), and the average la-
tency stays within 4.3–6.0 seconds per query.
Finally, enabling the LLM-based filtering stage on top
of GDino and TAG achieves the best performance of 44.87
M.J., a gain of nearly 4 points over the configuration without
GRASP and about 2 points over using only the grounding
modules. This improvement comes with a moderate increase
in resource usage (VRAM from 24.2 GB to 39.8 GB and
average time from 6.02 s to 7.02 s per query), while the LLM-
based filter helps focus the retrieved subgraph on answer-
supporting knowledge. Overall, these results show that each
GRASP submodule contributes positively to performance,
and that the full GRASP pipeline offers the best accuracy
5

with a relatively modest overhead compared to its benefits.
C. Additional Qualitative Results
In this section, we present additional qualitative com-
parisons of M3KG-RAG against V AT-KG [ 37] on multi-
modal QA benchmarks, including Audio QA (AudioCaps-
QA [ 45]), Video QA (VCGPT [ 36]), and Audio-Visual
QA (V ALOR [ 33]), using Qwen2.5-Omni [ 49] as the base
MLLM. For each benchmark, we show the knowledge
retrieved with V AT-KG and with M3KG-RAG, the corre-
sponding answers generated from these contexts, and the
win-rate judge’s preference and rationale. Both methods
construct MMKGs from raw multimodal corpora and sup-
port modality-wise retrieval. However, V AT-KG represents
each multimedia item with a single-hop graph and relies on
shallow similarity search, often yielding sparse or weakly
aligned evidence. In contrast, M3KG-RAG exploits multi-
hop knowledge and GRASP-based pruning to serve richer,
query-relevant context to the MLLM, leading to more faith-
ful and informative responses.
For the Audio QA case in Figure 6, the query asks what
animals can be heard in a clip where birds chirp in a forest
environment with background insect sounds. V AT-KG pri-
marily retrieves a single fact about a flock of birds in the
forest, leading the model to produce an answer that mentions
only birds. In contrast, M3KG-RAG retrieves multi-hop
knowledge that links both birds and crickets chirping in
a forest setting, providing richer cues about co-occurring
animal sounds. Conditioned on this context, the model iden-
tifies both birds and insects as audible in the scene, which
the win-rate judge prefers for covering all relevant animal
sources and better matching the reference audio.
For the Video QA case in Figure 7, the query video shows
a woman playing racquetball on an indoor court, and the
model is asked to describe in detail what happens in the
scene. V AT-KG performs coarse similarity-based retrieval
that includes knowledge about both squash and racquetball,
two related but distinct sports, which leads the model to
produce a hedged response that refers to a game similar to
squash or racquetball without clearly committing to the ac-
tual activity or capturing fine-grained details. In contrast,
M3KG-RAG, together with GRASP, prunes off-topic neigh-
bors based on fine-grained query relevance and supplies
racquetball-focused multi-hop evidence that matches the
video. Guided by this evidence, the model correctly identi-
fies the sport as racquetball and gives a more precise descrip-
tion of the player’s attire, court setting, and actions, which
the win-rate judge prefers for its specificity and semantic
alignment with the video.
For the Audio-Visual QA case in Figure 8, the multi-
modal query shows a man playing an electric guitar, and the
model is asked to describe the scene. V AT-KG, due to its
single-hop MMKG structure, mainly connects the man toa generic guitar and to a musician–acoustic-guitar relation,
providing only fragmentary, coarse knowledge. Without fine-
grained audio–visual relevance checking, it treats acoustic
and electric guitars as semantically interchangeable, which
leads the model to describe the scene as an acoustic guitar
performance and to miss surrounding contextual details. In
contrast, M3KG-RAG retrieves a multi-hop neighborhood
around the guitar that includes electric-guitar–specific con-
text (such as playing with an effects setup) together with
local scene cues (e.g., the man sitting on a chair in an indoor
room). With this richer, better-aligned evidence, the model
correctly identifies the instrument as an electric guitar and
produces a more detailed description of the player’s appear-
ance and environment, which the win-rate judge prefers for
both factual correctness and contextual richness.
D. Limitations
Despite the advancements presented in this work, several
limitations remain. First, the knowledge coverage of M3KG
is bounded by the raw multimodal corpora used for construc-
tion, which may underrepresent long-tail entities, rare rela-
tions, or domains not well captured in the source datasets.
Second, M3KG–RAG relies on the multimodal encoders
and grounding models used in the retrieval phase; when
their training domains do not cover the query distribution,
cross-modal miscalibration can surface off-topic neighbors
and degrade evidence quality. Extending M3KG-RAG with
better-calibrated encoders, broader and continually updated
corpora, and tighter grounding mechanisms is an important
direction for future work.
6

[Query]What animals can be heard in the audio?[Retrieved Subgraph]VAT-KG𝐌𝟑KG-RAG
[Answer 1] (V AT-KG)“The audio contains the sound of a bird chirping.”[Answer 2] (𝐌𝟑KG-RAG)“The animals that can be heard in the audio are birds and insects.”[Judge Response]"Comprehensiveness": "Winner": "Answer 2 (𝐌𝟑KG-RAG)""Explanation": "Answer 2 is more comprehensive because it provides more information about the animals that can be heard in the audio. It not only mentions birds, but also insects, giving a more detailed and thorough response to the question. Answer 1 only mentions a single bird chirping, which is a limited and incomplete description of the audio content.""Diversity": "Winner": "Answer 2""Explanation": "Answer 2 shows more diversity in its response by mentioning two different types of animals: birds and insects. This provides a richer and more varied answer, offering the reader a broader understanding of the audio content. Answer 1, on the other hand, only focuses on a single aspect, which limits its diversity and range.""Empowerment": "Winner": "Answer 2""Explanation": "Answer 2 empowers the reader more because it provides a more detailed and nuanced understanding of the audio content. By mentioning both birds and insects, it gives the reader a clearer picture of what they might expect to hear, allowing them to make more informed judgments about the audio. Answer 1, with its limited description, does not provide the same level of empowerment.""Overall Winner": "Winner": "Answer 2""Explanation": "Answer 2 is the overall winner because it excels in all three criteria. It provides a more comprehensive and detailed description of the audio content, offers a more diverse range of information, and empowers the reader to better understand and engage with the topic. Overall, Answer 2 is a more informative, thorough, and engaging response to the question."
flock of birdsforest(bird chirps and flies in the audio.)
chirpforestinsectchirpchirpsafari settingfliesbirdscricketsFigure 6.Qualitative Comparison on Audio QA.Comparing V AT-KG and M3KG-RAG with Qwen2.5-Omni, including retrieved
knowledge and win-rate judge preferences.
7

[Query]Can you describe the details of what happens in the video?[Retrieved Subgraph]VAT-KG𝐌𝟑KG-RAG
[Answer 1] (V AT-KG)"The video appears to show a person playing a sport in a walled court. The person is holding a racquet and seems to be engaged in a game similar to squash or racquetball. The setting and equipment suggest that the activity involves hitting a ball against the walls of the court. The person is actively moving around the court, indicating they are participating in a fast-paced game."[Answer 2] (𝐌𝟑KG-RAG)"The video shows a person playing racquetball in an indoor court. The individual is wearing athletic attire, including a white shirt and black shorts, and is using a racquet to hit the ball against the walls. The court's walls, floor, and ceiling serve as playing surfaces, which is typical for racquetball. The person appears to be actively engaged in the game, moving around to strike and return the ball as it rebounds."[Judge Response]"Comprehensiveness": "Winner": "Answer 2 (𝐌𝟑KG-RAG)""Explanation": "Answer 2 correctly covers more essential points from the Reference, including the indoor court, athletic attire, and active engagement in the game. Answer 1 omits these details and only provides a general description of the activity.""Diversity": "Winner": "Answer 2""Explanation": "Answer 2 provides a more varied perspective by incorporating details about the court's walls, floor, and ceiling as playing surfaces, which is typical for racquetball. Answer 1 lacks this specific insight.""Empowerment": "Winner": "Answer 2""Explanation": "Answer 2 offers clearer and more actionable guidance by providing specific details about the game, making it easier to understand the activity. Answer 1 is more general and lacks clear direction.""Overall Winner": "Winner": "Answer 2""Explanation": "Answer 2 wins due to its stronger correct coverage of the Reference, clearer and more concise presentation, and faithfulness to the original description. It is more comprehensive, diverse, and empowering than Answer 1."
mansquashpersonracquetballperformspersondynamic swingracquetballstudentfocused expressionhasplaysplaysFigure 7.Qualitative Comparison on Video QA.Comparing V AT-KG and M3KG-RAG with Qwen2.5-Omni, including retrieved
knowledge and win-rate judge preferences.
8

[Query]Describe in detail what is visually and audibly happening in the video, including actions, objects, people, sounds, and environment. Be specific and concise.
[Retrieved Subgraph]VAT-KG𝐌𝟑KG-RAG
[Answer 1] (V AT-KG)"The video features a man playing an acoustic guitar. He is wearing a black hat and a black shirt. The man is seated and playing the guitar with his right hand strumming the strings and his left hand pressing the strings on the fretboard. The video is split into two parts, with the top part showing a close-up of the man's hands on the guitar and the bottom part showing a wider view of the man playing the guitar. The video also includes text on the screen that reads Natural harmonics."[Answer 2] (𝐌𝟑KG-RAG)"In the video, there is a man playing an electric guitar. He is sitting on a chair and wearing a black shirt and a black cap. The guitar is a light brown color with a dark brown pickguard. The man is using his right hand to strum the strings and his left hand to press down on the frets. There is a close-up shot of his hands on the guitar. The sound of the guitar playing is beautiful music. The room where the man is playing reverberates the music."[Judge Response]"Comprehensiveness": "Winner": "Answer 2 (𝐌𝟑KG-RAG)""Explanation": "Answer 2 covers more essential points from the Reference, including the man playing guitar, the instrument's color, and the beautiful music. Answer 1 incorrectly states the guitar is acoustic and adds unnecessary information about the video being split into two parts and the text 'Natural harmonics’.""Diversity": "Winner": "Answer 2""Explanation": "Answer 2 provides a more varied perspective, describing both visual (man, guitar, chair) and audio (beautiful music) aspects of the Reference, while staying faithful to the original context. Answer 1 focuses mainly on the visual aspects.""Empowerment": "Winner": "Answer 2""Explanation": "Answer 2 is clearer and more concise, providing a direct and actionable description of the video content. It enables understanding of the scene without unnecessary details or speculative claims.""Overall Winner": "Winner": “Answer 2”"Explanation": "Answer 2 wins due to its stronger, more faithful coverage of the Reference's essential points, its varied yet concise perspective, and its clearer, more actionable presentation, making it the most faithful and empowering answer."
manguitarmusicianacoustic guitarusesguitarelectric guitar pedalmanbeautiful musicchairsitsplaysplaysroomreverberateFigure 8.Qualitative Comparison on Audio-Visual QA.Comparing V AT-KG and M3KG-RAG with Qwen2.5-Omni, including retrieved
knowledge and win-rate judge preferences.
9