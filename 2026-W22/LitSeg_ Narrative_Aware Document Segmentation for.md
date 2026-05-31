# LitSeg: Narrative-Aware Document Segmentation for Literary RAG

**Authors**: Ruikang Zhang, Zhanni Chen, Yiqiao Cai, Qi Su

**Published**: 2026-05-26 15:17:03

**PDF URL**: [https://arxiv.org/pdf/2605.27156v1](https://arxiv.org/pdf/2605.27156v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by incorporating external knowledge, particularly for long-tail domains such as literary works. However, the critical step of document segmentation in RAG remains largely underexplored. Existing strategies are typically semantically blind and overlook the complicated narrative structures of literary works, often resulting in fragmented plots and unclear references that severely hinder retrieval and generation performance. To address this, we propose LitSeg, a novel narrative-theory-guided segmentation framework. By employing multi-stage prompting, LitSeg explicitly extracts valid events, untangles narrative threads, clarifies narrative structures, and locates turning points to inform segmentation. To alleviate the computational overhead of multi-stage inference with large-scale models, we further introduce LitSeg-Lite, a lightweight single-pass chunker fine-tuned on LitSeg-generated data via a two-stage training strategy, distilling the complex process into a single inference pass. Extensive experiments demonstrate that with structurally independent text chunks, our methods significantly improve retrieval accuracy and context relevance over baselines, ultimately enhancing downstream QA performance, while ablation studies validate the efficacy of narratological guidance and data distillation.

## Full Text


<!-- PDF content starts -->

LitSeg: Narrative-Aware Document Segmentation for Literary RAG
Ruikang Zhang1Zhanni Chen1*Yiqiao Cai1*Qi Su1†
1Peking University, Beijing, China
{2300018416, 2300018109, 2300018316}@stu.pku.edu.cn,sukia@pku.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) en-
hances Large Language Models (LLMs) by in-
corporating external knowledge, particularly
for long-tail domains such as literary works.
However, the critical step of document segmen-
tation in RAG remains largely underexplored.
Existing strategies are typically semantically
blind and overlook the complicated narrative
structures of literary works, often resulting in
fragmented plots and unclear references that
severely hinder retrieval and generation perfor-
mance. To address this, we proposeLitSeg,
a novel narrative-theory-guided segmentation
framework. By employing multi-stage prompt-
ing, LitSeg explicitly extracts valid events, un-
tangles narrative threads, clarifies narrative
structures, and locates turning points to inform
segmentation. To alleviate the computational
overhead of multi-stage inference with large-
scale models, we further introduceLitSeg-Lite,
a lightweight single-pass chunker fine-tuned on
LitSeg-generated data via a two-stage training
strategy, distilling the complex process into a
single inference pass. Extensive experiments
demonstrate that with structurally independent
text chunks, our methods significantly improve
retrieval accuracy and context relevance over
baselines, ultimately enhancing downstream
QA performance, while ablation studies vali-
date the efficacy of narratological guidance and
data distillation.
1 Introduction
Retrieval-Augmented Generation (RAG) enhances
the generation process of Large Language Models
(LLMs) by retrieving relevant information from ex-
ternal corpora, thereby improving their access to
long-tail knowledge such as literary works. (Zhao
et al., 2026) A RAG system typically consists of
three parts: 1) indexing, in which raw documents
*Equal contribution.
†Corresponding author.are segmented into chunks, and then represented
and stored in vector databases; 2) retrieval, in
which the system retrieves relevant chunks given
a user query; 3) generation, in which an LLM for-
mulates a response given the query and retrieved
chunks (Ma et al., 2025). In such systems, doc-
ument segmentation affects the vector representa-
tion and text quality of chunks, thereby influencing
retrieval accuracy, the information accessible to
the generator, and ultimately the performance of
RAG systems on downstream tasks such as ques-
tion answering (QA) (Rajanidi et al., 2026) and
agentic tasks (Singh et al., 2025). However, exist-
ing work has primarily focused on index structures,
document retrieval, and response generation (e.g.,
Huang et al., 2025; Gao et al., 2023; Tu et al.,
2025), leaving document segmentation underex-
plored (Wang et al., 2025). Given the compli-
cated narrative structure of literary works (Shen
and Wang, 2010), inappropriate text segmentation
can distort the semantics of individual segments
and lead to fragmented plots and unclear references,
thereby degrading downstream retrieval and gen-
eration performance. Nevertheless, existing seg-
mentation strategies are either semantically blind
or overlook such high-order structures (see Sec-
tion 2.1), leaving this critical issue unresolved.
In light of this, we presentLitSeg(see Sec-
tion 3.1), a narrative-theory-guided segmentation
framework. Drawing on established theories, we
leverage explicit, multi-stage prompting to task an
LLM with extracting valid events, untangling narra-
tive threads, clarifying narrative structures, locating
key turning points and finally executing segmenta-
tion. We also provide the model with long context
and design an input-output format that numbers the
input text and represents each chunk as a collection
of sentence IDs, thereby supporting flexible recom-
bination of non-adjacent sentences while preserv-
ing broader narrative context. To further reduce
the overhead of multi-stage inference and large-arXiv:2605.27156v1  [cs.CL]  26 May 2026

LitSegLitSeg-Lite
EvaluationNarratological Theories1. Event-based Plot Model2. Simultaneity in Narrative      ……3. Aristotelian Plot Structure TheorySTEP 1: Extract Valid Events and Filter NoiseSTEP 2: Untangle Narrative Threads and Clarify StructuresSTEP 3: Locate Key Turning Points and Execute Segmentation
GRPOTheory-Guided Reward Model
LiteraryQA
promptTeacher ModelStudent ModelSFT
annotate
LiteraryQA Chunking Dataset Question:What kind of dog is Buck?Standard Answer: ['A St. Bernard-Scotch Shepherd.', 'ST BERNARD/SCOTCH SHEPHERD MIX’]Our Answer: Buck is a St. Bernard and Scotch shepherd dog mix.Chapter I. Into the Primitivestep_1_events: “…”step_2_threads: “…”step_3_turning_points: […]segments: [{SUBTITLE, FROM_IDX, TO_IDX, CONTEXT_IDX},…]
GutenQA
Format_Reward
{event_validity: [4.0, 3.0, 4.5],unity_of_action: [3.0, 4.0, 5.0],cut_point_logic: [5.0, 5.0, 5.0]}
Figure 1: Overview of our proposed framework. LitSeg leverages narratological theories to guide a high-capacity
teacher model through a multi-stage segmentation pipeline, producing narrative-aware chunking annotations. LitSeg-
Lite then distills this knowledge into a lightweight student model via SFT and GRPO with theory-guided rewards,
enabling efficient single-pass segmentation for RAG-based QA on literary benchmarks.
scale model usage, we also presentLitSeg-Lite
(see Section 3.2), a lightweight chunker that per-
forms segmentation in a single inference pass. It is
fine-tuned on LitSeg-generated data via a two-stage
training strategy, achieving performance compara-
ble to LitSeg. Experiments show that by producing
narratologically self-contained text chunks, LitSeg
and LitSeg-Lite substantially improve retrieval ac-
curacy, context relevance, and downstream answer
accuracy over baselines. Ablation study further
shows that both narrative theories and distillation
of LitSeg data contribute to the strong performance
of LitSeg-Lite (see Section 4). Figure 1 provides
an overview of our proposed framework.
Our main contributions are threefold:(1)We pro-
poseLitSeg, a novel narrative-theory-guided seg-
mentation framework for RAG systems targeting
literary works. By employing multi-stage prompt-
ing, LitSeg explicitly extracts narrative structure to
aid segmentation, effectively addressing the seman-
tic blindness of existing segmentation methods and
preserving high-order narrative structures.(2)We
introduceLitSeg-Lite, a lightweight, single-pass
chunker fine-tuned on LitSeg-generated data via
a two-stage training strategy. It distills the multi-
step inference process into a single inference pass,
achieving performance comparable to LitSeg withsubstantially lower inference overhead.(3)Com-
prehensive experiments demonstrate that our meth-
ods substantially improve retrieval accuracy and
downstream QA performance by leveraging seman-
tically independent text chunks. Ablation studies
validate the efficacy of narratological guidance and
data distillation.
2 Related Works
2.1 Document Segmentation in RAG
In RAG systems, the most prevalent segmenta-
tion methods typically split documents by fixed
token (Ma et al., 2025) or character counts. Re-
cursive character splitters (Chase, 2022) refine this
strategy by recursively applying common delim-
iters (e.g., newlines) until size constraints are met.
However, these heuristic approaches are semanti-
cally agnostic, often disrupting structural integrity
and semantic coherence. To address this, semantics-
aware methods have emerged, leveraging embed-
ding models or LLMs to capture textual seman-
tics. Embedding-based approaches detect seman-
tic shifts; for instance, SemanticChunker (Chase,
2022) inserts boundaries where the cosine distance
between adjacent sentence-group embeddings ex-
ceeds a percentile threshold. LLM-based methods

further exploit model internals or reasoning capabil-
ities: Perplexity Chunker (Zhao et al., 2024) iden-
tifies boundaries at local minima of the sentence-
level loss curve; Margin-Sampling Chunker (Zhao
et al., 2024) frames boundary detection as a binary
decision based on the LLM’s probability margin for
keeping adjacent text; and LumberChunker (Duarte
et al., 2024) uses a sliding window to prompt an
LLM to directly pinpoint content shifts. However,
these methods still overlook the high-order nar-
rative structures of literary works, where coherent
chunks often depend on long-range causal relations,
parallel narrative threads, and shifts in narrative
level or perspective. This limitation restricts their
effectiveness in literary RAG systems.
2.2 Distillation for Efficient Task-Specific
Models
Supervised Fine-Tuning (SFT) (Zhang et al., 2026)
remains the foundational step to adapt Large Lan-
guage Models (LLMs) to expert demonstrations.
While SFT effectively instills specific task knowl-
edge and stylistic consistency, it is inherently con-
strained by the quality and diversity of the offline
dataset (Chu et al., 2025). To address these limita-
tions, Group Relative Policy Optimization (GRPO)
(Shao et al., 2024) has emerged as an efficient Re-
inforcement Learning (RL) paradigm. GRPO gen-
erates a group of Grollouts {y1, y2, . . . , y G}for
each prompt xand computes the relative advantage
of each output by normalizing rewards within the
group. This mechanism bypasses the optimization
instabilities often associated with value function
estimation and significantly reduces memory over-
head of traditional Proximal Policy Optimization
(PPO) methods (Schulman et al., 2017) that rely
on a computationally expensive critic model to es-
timate value baselines. Building upon the group-
based efficiency of GRPO, Decoupled Clip and
Dynamic Sampling Policy Optimization (DAPO)
(Yu et al., 2025) further enhances training stabil-
ity and output diversity, particularly in scenarios
requiring long-sequence structural consistency.
2.3 Narrative Theory and NLP
Narrative theory is the foundational framework for
understanding how stories are constructed, struc-
tured, and communicated (Liveley, 2019), provid-
ing crucial theoretical underpinnings for various
Natural Language Processing (NLP) tasks.
In the realm of narrativity analysis, narrative the-
ory provides essential criteria for quantifying thedegree of narrativity and distinguishing narrative
discourse from non-narrative noise. Piper et al.
(2021) synthesizes classical and postclassical nar-
rative theories, guiding computational narrative re-
search by defining the core elements of narrativity.
To operationalize event narrativity, Gius and Vauth
(2022) proposes an event-based plot model classi-
fying textual components into distinct event types;
Piper and Bagga (2024) further applies LLMs to
annotate narrative discourse within the framework
of Genette (1980)’s narrative triangle concerning
story, discourse, and narrating.
Beyond identifying narration, narrative theory
also offers vital structural paradigms for organiz-
ing scattered events into coherent narrative threads,
thereby facilitating storyline extraction. V ossen
et al. (2021) introduces a narratology-based frame-
work for news that maps the theoretical notions
offabula, plot, and plot structure to explicit data
structures: timelines, causelines, and storylines. To
address the higher levels of non-linearity in fic-
tional works compared to news, Visser Solissa et al.
(2025) constructs a theoretical model for event an-
notation grounded in the theory ofsyuzhet(the
concrete presentation order of events) to achieve
its computational detection.
On the other hand, narrative theory empowers
text segmentation by guiding the identification of
logical breakpoints across different levels. At a
macro-structural level, Papalampidi et al. (2019)
models movie plots by identifying theory-grounded
narrative turning points. At a micro-linguistic level,
text boundaries are effectively pinpointed using
mutual information theory: Wagner et al. (2024)
detects zones of low Point-wise Mutual Informa-
tion (PMI) between adjacent sentences to segment
spoken testimonies, while Wang et al. (2023) intro-
duces M3Seg, a novel Maximum-Minimum Mutual
information paradigm, to partition ASR transcripts.
Despite significant strides in quantifying narra-
tive structures, these works are hindered by two
primary limitations: theoretically, they often fo-
cus on single-faceted features, lacking a refined
and synthesized comprehensive theoretical frame-
work; practically, they predominantly target iso-
lated computational tasks, thereby failing to induce
and execute an end-to-end theoretical pipeline for
the entire text segmentation process.

3 Methodology
3.1 LitSeg: Narrative Theory Guided
Segmentation Framework
3.1.1 Theoretical Framework
Our framework integrates classical and post-
classical narratology with computational narratol-
ogy, translating abstract literary concepts into op-
erational metrics to address the limitations of ex-
isting semantic segmenters. Specifically, LitSeg
explicitly decomposes the complex cognitive task
of document chunking into a three-step narratolog-
ical pipeline: Event Extraction, Thread Untangling,
and Turning Points Pinpointing. See Appendix D.1
for full prompts.
Extract Valid Events and Filter NoiseDistin-
guishing valid narrative progression from back-
ground information is the foundational step of seg-
mentation. Plot is defined by its dynamic, sequen-
tial nature; narrative elements only constitute a
plot when they drive this dynamism (Scholes et al.,
2006). Thus, the event-based plot model (Gius
and Vauth, 2022) is adopted to explicitly task the
LLM to filter text by retaining only changes of state
and process events, while discarding static events
and non-events (e.g., generic statements or coun-
terfactual passages). For ambiguous paragraphs,
the text’s degree of narrativity is quantified by ac-
tively evaluating specific feature densities, such as
the presence of an agent, sequential actions, spa-
tial/temporal specifications, and rationale (Piper
et al., 2021), to guarantee that only independent
and valid events are extracted for downstream pro-
cessing.
Untangle Narrative Threads and Clarify Struc-
turesLiterary texts frequently feature inter-
twined storylines. To systematically disentangle
these narratives, we adopt Aristotle’s principle of
unity of action (Butcher et al., 1902; Shen and
Wang, 2010), enabling the model to precisely clus-
ter events based on complete, core actions. Since
a well-arranged plot often builds the structural in-
tegrity of the whole story through the gradual un-
folding of secrets (Forster, 1927), tracking this rev-
elation process guides the model to connect scat-
tered events into cohesive narrative chains. Further-
more, to handle complex discourse, the theory of
simultaneity in narrative (Margolin, 2014) is incor-
porated to instruct the model to explicitly segment
concurrent and parallel event chains, thereby pre-
venting the erroneous merging of minor threadsinto major ones. We also map the nested narra-
tive hierarchy (e.g., extradiegetic, intradiegetic, or
metadiegetic levels) based on the theory of narra-
tive levels (Genette, 1980) to systematically pre-
vent context collisions caused by embedded narra-
tives. Finally, to capture the multidimensionality
of narrative discourse, we build upon the three core
narrative dimensions: time, setting, and point-of-
view, which Piper and Bagga (2024) operational-
ized for LLMs under Genette (1980) classical nar-
ratological framework. By combining these dimen-
sions with V ossen et al. (2021)’s narratology-based
framework for storyline extraction, we are able to
comprehensively extract timelines, causelines, and
critical shift points in time, space, and perspective
from this structured thread data.
Locate Key Turning Points and Execute Seg-
mentationUltimately, the framework utilizes
the established thread data to determine logical
breakpoints. Turning points are treated as crucial
functional shifts rather than mere plot occurrences
(Thompson, 1999), ensuring that chunk boundaries
are dictated by structural narrative transitions rather
than arbitrary length constraints or linear episodic
prolongations. Building upon this rationale, our
framework segments narratives through a four-
tiered location strategy. At the macro-structural
level, we integrate Aristotelian plot structure the-
ory, specifically the narrative shifts of reversal and
recognition (Butcher et al., 1902; Shen and Wang,
2010), with turning-point identification (Papalam-
pidi et al., 2019; Hauge, 2017) to capture major nar-
rative pivots: opportunity, change of plans, point
of no return, major setback and reversal, climax
and recognition, and aftermath and buffer. At the
contextual level, we leverage the established mul-
tidimensional thread data to pinpoint breakpoints
at transitions in time, space, or perspective, as well
as shifts between parallel or nested narrative levels.
At the micro-dynamic level, we align segmentation
with state transitions (from equilibrium to disequi-
librium to a new equilibrium) (Todorov and Wein-
stein, 1969) and the logic of narrative possibilities
(Bremond and Cancalon, 1980; Shen and Wang,
2010), capturing the precise moments where the
fulfillment or failure of an action is resolved. Fi-
nally, to pinpoint exact sentence-level boundaries
within these general turning points, we draw inspi-
ration from mutual information (MI) theory (Wag-
ner et al., 2024; Wang et al., 2023) to formulate an
LLM-assessable metric for semantic cohesion. The

model is prompted to actively evaluate the semantic
dependency between adjacent sentences, avoiding
segmenting within High Semantic Cohesion Zones
(e.g., spans with dense MI markers, such as con-
tinuous pronoun chains or adjacency pairs), and
instead targeting Low Semantic Cohesion Zones,
where semantic dependency is minimal. Collec-
tively, this four-tiered strategy ensures a rigorous
alignment between segmentation and the underly-
ing narrative architecture, thereby ensuring that the
resulting segments are narratively self-contained
and semantically coherent.
3.1.2 Implementation of LitSeg
The LitSeg pipeline comprises three sequential
phases: preprocessing, multi-round generation, and
validation with fallback mechanisms. More imple-
mentation details are in Appendix D.1 and H.2.
PreprocessingTo preserve authorial intent and
semantic integrity, we first perform structure-driven
splitting by isolating top-level narrative units (e.g.,
chapters, acts, scenes, sections, cantos, or titled
episodes). This ensures that the model receives
self-contained structural units while naturally ac-
commodating the context window limits for most
texts. When top-level unit metadata is unavailable,
or when a narrative unit exceeds a length threshold
n, we apply recursive word-level splitting, hier-
archically segmenting the text by paragraphs and
then sentences while greedily satisfying the n-word
constraint. We set nto the largest value allowed by
the model’s context budget and without substantial
long-context degradation. Subsequently, we assign
a unique sequential index to each sentence in the
candidate chunk.
Multi-stage GenerationWe implement a three-
stage pipeline corresponding to the narratological
workflow above. All stages share a unified sys-
tem prompt containing the full narratological work-
flow, providing the model with global task aware-
ness. The user prompt for each stage contains
stage-specific formatting instructions alongside the
model’s outputs from preceding stages. This de-
sign implements an explicit multi-stage prompt-
ing strategy, which avoids the token overhead of
unconstrained Chain-of-Thought (CoT) rationales
while ensuring high-quality segmentation. To fur-
ther optimize inference efficiency, we constrain the
model to output sentence indices rather than ver-
batim text, significantly reducing the overhead of
autoregressive decoding. Specifically, for each seg-mented plot block, the model outputs a main range,
represented as a closed interval between its start
and end sentence indices, and an optional context
list, represented as a sequence of discrete sentence
indices. Sentences shared by overlapping main
ranges are retained in all corresponding blocks, al-
lowing boundary sentences to serve as shared narra-
tive context. Unlike traditional chunking methods
that rely on fixed, adjacent overlaps, this index-
based representation allows the model to flexibly
incorporate non-adjacent sentences from anywhere
within the indexed candidate chunk that are crucial
for understanding the main block. Additionally, we
task the model with generating a concise subtitle
for each block, leveraging its generative capabili-
ties to summarize the core event without the burden
of extensive text generation.
Validation, Retry, and FallbackWe apply dual-
level validation to guarantee reliability. Syntax val-
idation ensures that the output JSON conforms to
the predefined schema. Semantic validation verifies
that all main ranges and context indices are valid
and that the main ranges, with overlaps permit-
ted, collectively and exhaustively cover all source
sentences, ensuring zero information loss. Invalid
outputs are retried up to Ntimes. If coverage gaps
persist after retries, a fallback mechanism greedily
appends the missing sentences to their left-adjacent
segment in source order (or the right one if unavail-
able). If the output remains unparsable after all
retries, the system falls back to yielding the whole
preprocessed chunk.
ReconstructionDuring text reconstruction, sen-
tences specified by the context list, except those
already included in the corresponding main
range, are ordered by their original positions and
prepended or appended to the main text block ac-
cordingly. The generated subtitle is then prepended
to form the final context-enriched chunk.
3.2 LitSeg-Lite: Distillation of Lightweight
Chunker
With the objective of improving inference effi-
ciency and facilitating deployment, we develop
LitSeg-Lite, a compact student model distilled
from LitSeg. Given the same sentence-indexed
candidate chunk as LitSeg, LitSeg-Lite predicts a
JSON output containing the internally summarized
narratological workflow and the final index-based
segmentation within a single inference pass. We
train LitSeg-Lite with LoRA (Hu et al., 2021) on

the LitSeg-annotated LiteraryQA dataset through
a two-stage optimization pipeline: initial super-
vised fine-tuning (SFT) followed by Reinforcement
Learning (RL) refinement. The SFT targets are the
validated LitSeg annotations, reformatted into the
single-stage output schema used by LitSeg-Lite.
To ensure that LitSeg-Lite acquires narratology-
aware segmentation logic and robust instruction-
following capabilities, we first perform SFT. While
SFT facilitates rapid task adaptation, it often results
in limited out-of-distribution generalization (Chu
et al., 2025). To overcome these limitations, we
integrate RL to further enhance the model’s behav-
ioral robustness and generalization. Specifically,
we employ GRPO with a DAPO-style objective,
guided by a specialized reward function grounded
in narratological theory. This function combines a
rule-based format validator with a model-based re-
ward evaluator; the latter conducts granular assess-
ments across three narrative-centric dimensions:
Event Validity, Unity of Action, and Cut Point
Logic, thereby targeting common weaknesses of
the student model (see Appendix D.3 for the com-
prehensive scoring rubrics). The model-based
scores are first averaged equally across the three di-
mensions and then over predicted segments to form
a narratological reward, which is combined with
the rule-based format reward through a weighted
sum as the final optimization objective. This re-
ward design explicitly aligns the student model
with narratological criteria for structural analysis.
See Appendix H.3.2 for training configurations and
reward details.
While LitSeg achieves high-fidelity results
through a sophisticated three-stage prompting
pipeline, such a multi-turn process introduces sig-
nificant computational overhead. For practical de-
ployment, LitSeg-Lite performs segmentation with
a single model call, eliminating LitSeg’s multi-
stage prompting overhead while maintaining com-
petitive performance. This offers a practical bal-
ance between narrative depth and operational ef-
ficiency. For prompts for LitSeg-Lite, see Ap-
pendix D.2. For inference configurations, see Ap-
pendix H.3.1.
4 Experiments
In this section, we comprehensively evaluate Lit-
Seg and LitSeg-Lite on two narrative QA bench-
marks, investigating their retrieval and generation
performance (Subsection 4.1), assessing the intrin-sic quality and structural independence of their
text segments (Subsection 4.2), and validating the
contribution of core design choices through abla-
tion studies (Subsection 4.3). Details of our RAG
pipeline are provided in Appendix H.1.
4.1 Retrieval and Generation Performance
Datasets and BaselinesWe employ two bench-
marks for evaluation. LiteraryQA (Bonomo et al.,
2025), a long-document narrative QA dataset fo-
cusing on diverse literary works (e.g., novels, nar-
rative poetry, and drama), with all reported results
evaluated on its official test split. However, as Lit-
eraryQA lacks gold evidence-span annotations for
precise retrieval evaluation, and to assess the gen-
eralization performance of LitSeg-Lite, which is
trained on LitSeg annotations of the LiteraryQA
training set in our experiments, we further adopt
GutenQA (Duarte et al., 2024), a retrieval-focused
benchmark featuring “needle-in-a-haystack” fac-
tual QA. Since LitSeg primarily serves as a high-
capacity teacher for generating segmentation anno-
tations rather than as a deployable chunker, we
use LiteraryQA to report both teacher and stu-
dent performance, while using GutenQA mainly
to evaluate the cross-dataset generalization of the
deployable LitSeg-Lite model. We compare our ap-
proaches against heuristic, embedding-based, and
LLM-based segmentation baselines. Detailed base-
line configurations are provided in Appendices G
and H.5.
Evaluation Metrics and Human JudgmentWe
comprehensively assess retrieval and generation us-
ing lexical and LLM-based automatic metrics (See
Appendices E and H.4 for metric definitions and im-
plementation details). While reliable for structured,
information-centric domains with well-defined evi-
dence boundaries, these metrics systematically pe-
nalize answers containing enriched, contextually
appropriate information not included in ground
truths, as they only measure surface-level literal
or factual matching against ground truths (Gerrits
et al., 2026), hindering their validity on nuanced
literary QA tasks. For instance, on GutenQA, our
richer narrative context prompts the generator to
produce highly accurate and informative, even in-
depth answers. This creates a systematic discrep-
ancy with the concise, factoid-style reference la-
bels, which are referenced by automated metrics
in isolation from the broader context, thereby un-
derestimating our method’s performance. Similar

phenomena have been observed in literary transla-
tion (Zhang et al., 2025) and other expert knowl-
edge tasks (Szymanski et al., 2025). To obtain a
reliable assessment, we therefore conduct a human
pair-wise evaluation on GutenQA. Annotators com-
pare LitSeg’s outputs against reference answers,
with both judged against the original text within
a±500-character context window. The primary
criterion is factual accuracy, followed by informa-
tiveness. See Appendix A for the full annotation
guidelines and inter-annotator agreement. Further-
more, our manual audit uncovers five categories of
inherent flaws in the existing benchmarks that fur-
ther undermine automated evaluation: 1) ambigu-
ous or erroneous questions; 2) incomplete ground
truth; 3) literal extractions misinterpreting narrative
subtext; 4) narrow reference answers, and 5) entity
and granularity mismatches. Detailed case studies
are provided in Appendix C.
Table 1: Generation results on LiteraryQA and
GutenQA. EM = Exact Match; R-L = ROUGE-L; Mtr
= METEOR; AA = Answer Accuracy; WR = Pairwise
Win Rate. Best results inbold.
Method GutenQA LiteraryQA
WR EM F1 R-L Mtr AA
RAG Baselines
Token 0.583 0.066 0.240 0.248 0.296 0.418
Recursive Char 0.552 0.078 0.253 0.259 0.298 0.415
Perplexity 0.603 0.072 0.235 0.245 0.280 0.373
+ Merge 0.618 0.079 0.253 0.259 0.302 0.423
Margin-Sampling 0.652 0.065 0.205 0.216 0.247 0.308
+ Merge 0.594 0.080 0.255 0.262 0.299 0.418
LumberChunker 0.584 0.076 0.250 0.255 0.299 0.435
+ Merge 0.588 0.077 0.252 0.257 0.303 0.438
Ours
LitSeg-Lite * 0.0790.258 0.263 0.3110.448
LitSeg —0.0810.256 0.261 0.3090.450
Table 2: Retrieval results on GutenQA and LiteraryQA.
CR = Context Relevance; MRR = Mean Reciprocal
Rank; H@k= Hit@k. Best results inbold.
Method GutenQA LitQA
CR MRR H@1 H@2 H@3 H@5 H@20 CR
RAG Baselines
Token 0.885 0.572 0.478 0.589 0.654 0.728 0.728 0.654
Recursive Char 0.871 0.614 0.531 0.631 0.689 0.748 0.748 0.664
Perplexity 0.861 0.555 0.463 0.573 0.640 0.702 0.702 0.665
+ Merge 0.889 0.599 0.514 0.619 0.679 0.734 0.734 0.724
Margin-Sampling 0.781 0.383 0.309 0.392 0.446 0.510 0.510 0.564
+ Merge 0.877 0.593 0.509 0.605 0.668 0.732 0.732 0.714
LumberChunker 0.885 0.633 0.552 0.652 0.708 0.761 0.761 0.739
+ Merge 0.893 0.656 0.574 0.674 0.736 0.785 0.785 0.752
Ours
LitSeg-Lite0.938 0.683 0.600 0.701 0.759 0.813 0.813 0.805
LitSeg — — — — — — — 0.799Results and AnalysisTables 1 and 2 report the
main results on both benchmarks. Our proposed
methods consistently achieve strong performance
across generation and retrieval metrics, validating
the effectiveness of our narrative-theory-guided
segmentation framework. Notably, significant
gains are observed in Context Relevance, where
LitSeg-Lite outperforms the best baselines by sub-
stantial margins (+0.045 on GutenQA; +0.053
on LiteraryQA), demonstrating that narratologi-
cal boundaries effectively preserve plot-complete
passages. This segmentation advantage is further
reflected in retrieval performance, evidenced by
the highest MRR (0.683) and substantial improve-
ments in Hit@ kmetrics (e.g., 0.813 on H@5 for
GutenQA), indicating that our method not only
retrieves the correct chunks but also ranks them
significantly higher. Consequently, this precise re-
trieval and prioritization of relevant context greatly
empowers the generator to utilize necessary infor-
mation, confirmed by LitSeg achieving the highest
Answer Accuracy on LiteraryQA (0.450), surpass-
ing the best baseline (0.438). On GutenQA, Lit-
seg consistently outperforms all alternative metrics
in pair-wise evaluations (e.g., securing a 65.2%
win rate against Margin-Sampling). Furthermore,
the distillation process proves highly effective, as
LitSeg-Lite not only matches but slightly exceeds
LitSeg’s Context Relevance (0.805 vs. 0.799) and
achieves comparable generation scores (e.g., An-
swer Accuracy of 0.448 vs. 0.450). This con-
firms that LitSeg-Lite successfully internalizes the
narrative-aware segmentation logic of LitSeg in a
single, efficient pass using a compact model.
To qualitatively illustrate how our framework
preserves narrative integrity and enhances down-
stream QA performance, we provide case studies
and detailed textual analyses in Appendix B. See
Appendix F for more metrics on GutenQA.
4.2 Segment Quality
To establish that our narratological-theory-guided
framework yields text chunks with superior struc-
tural and thematic independence, we conduct an
intrinsic evaluation of the segmentation quality.
Specifically, we measure the boundary semantic
transition by calculating the cosine similarity be-
tween the sentences immediately preceding and
succeeding each predicted boundary.
As illustrated in Table 3, LitSeg-Lite achieves
the lowest average boundary similarity across both
benchmarks (0.4025 on GutenQA and 0.3699

Table 3: Average semantic similarity of adjacent sen-
tences across predicted segment boundaries. Lower
values indicate better segmentation independence; best
results are inbold.
Method GutenQA LiteraryQA
RAG Baselines
Token 0.4536 0.4047
Recursive Char 0.4382 0.4219
Perplexity 0.4257 0.4064
+ Dynamic Merge 0.4378 0.4179
Margin-Sampling 0.4365 0.4148
+ Dynamic Merge 0.4468 0.4276
LumberChunker 0.4135 0.3805
+ Dynamic Merge 0.4096 0.3761
Ours
LitSeg — 0.3768
LitSeg-Lite0.4025 0.3699
on LiteraryQA). This empirical drop in cross-
boundary similarity indicates a sharper semantic
transition, demonstrating that LitSeg successfully
identifies profound plot transitions and narrative
boundaries, thereby rendering the chunks highly
self-contained and informative for retrieval.
4.3 Ablation Studies
Table 4: Ablation study on LiteraryQA and GutenQA
generation. Best results inbold.
Method GutenQA LiteraryQA
WR EM F1 R-L Mtr AA
LitSeg-Lite *0.079 0.258 0.263 0.311 0.448
w/o FT 0.593 0.077 0.253 0.256 0.306 0.446
+w/o Theory 0.584 0.073 0.250 0.255 0.301 0.438
+w/o Theo.S.1 0.553 0.074 0.248 0.252 0.298 0.441
+w/o Theo.S.2 0.564 0.076 0.250 0.254 0.300 0.441
+w/o Theo.S.3 0.548 0.078 0.255 0.259 0.308 0.445
Table 5: Ablation study on GutenQA and LiteraryQA
retrieval. Best results inbold.
Method GutenQA LitQA
CR MRR H@1 H@2 H@3 H@5 H@20 CR
LitSeg-Lite0.938 0.683 0.600 0.701 0.759 0.813 0.813 0.805
w/o FT 0.927 0.646 0.550 0.670 0.737 0.797 0.796 0.787
+w/o Theory 0.936 0.632 0.536 0.657 0.720 0.783 0.783 0.764
+w/o Theo.S.1 0.927 0.634 0.532 0.656 0.728 0.794 0.794 0.791
+w/o Theo.S.2 0.922 0.628 0.527 0.647 0.719 0.792 0.792 0.781
+w/o Theo.S.3 0.925 0.625 0.531 0.648 0.709 0.774 0.774 0.771
To validate the contributions of distilled knowl-
edge and narratological theory, we ablate LitSeg-
Lite into five variants:w/o FT(retains the
theory prompt but removes fine-tuning),+w/o
Theo.S.1/2/3(each ablates one specific theory-
driven stage beyondw/o FT), and+w/o Theory
(replaces the entire narratological pipeline with a
generic prompt beyondw/o FT). See Appendix D.4
for detailed prompt templates.Table 6: Ablation study on segmentation quality (aver-
age boundary semantic similarity). Best results are in
bold.
Method GutenQA LiteraryQA
LitSeg-Lite0.4025 0.3699
w/o FT 0.4350 0.3989
+w/o Theo.S.1 0.4349 0.3941
+w/o Theo.S.2 0.4385 0.4012
+w/o Theo.S.3 0.4403 0.4031
+w/o Theory 0.4324 0.3991
As shown in Tables 4 and 5, both distillation
and the multi-step narratological pipeline are in-
dispensable. Removing fine-tuning or skipping
any individual theory-driven stage consistently de-
grades performance, with retrieval metrics suffer-
ing the most pronounced declines. On GutenQA,
MRR drops from 0.683 to 0.646 (w/o FT), 0.634
(+w/o Theo.S.1), 0.628 (+w/o Theo.S.2), and 0.625
(+w/o Theo.S.3), with the fully generic prompt
(+w/o Theory) yielding a comparable 0.632. A
similar pattern emerges on LiteraryQA, where Con-
text Relevance degrades from 0.805 to 0.787 (w/o
FT) and further to 0.764 (+w/o Theory), indicating
that narratological knowledge is the primary driver
of retrieval quality. These retrieval deficits cascade
to downstream generation: Answer Accuracy on
LiteraryQA decreases from 0.448 to 0.438 (+w/o
Theory), while pair-wise evaluation on GutenQA
is consistently defeated by the full model, securing
a win rate of 58.4%(+w/o Theory). Ablating one
component even incurs degraded performance com-
pared to+w/o Theory, confirming that the holistic
integration of both distillation and theory-guided
pipelines are indispensable for robust document
retrieval and high-fidelity generation.
Beyond downstream performance, we conduct
an intrinsic evaluation of segmentation boundary
quality using average boundary semantic similar-
ity (Table 6), where lower values indicate more
semantically distinct cuts. Both fine-tuning and
narratological prompting reduce this similarity
across benchmarks (e.g., from 0.4324 to 0.4025
on GutenQA and from 0.3991 to 0.3699 on Lit-
eraryQA), confirming that our framework is es-
sential for regularizing the model to produce self-
contained, semantically coherent text segments.
5 Conclusion
In this work, we address the critical challenge of
document segmentation for literary RAG systems,
where existing methods overlook higher-order se-

mantics and severely disrupt narrative coherence,
resulting in suboptimal performance in QA sys-
tems. We introduce LitSeg, a narrative-theory-
guided framework employing multi-stage prompt-
ing to explicitly extract narrative structures, com-
plemented by a novel index-based output schema
that reduces autoregressive overhead and enables
flexible text reconstruction. To mitigate the com-
putational cost of this multi-stage pipeline, we fur-
ther present LitSeg-Lite, a lightweight single-pass
chunker distilled via a two-stage SFT and RL strat-
egy. Extensive experiments on GutenQA and Lit-
eraryQA demonstrate that our proposed methods
substantially outperform baselines, yielding signifi-
cant gains in context relevance, retrieval accuracy,
and downstream answer accuracy. Ablation stud-
ies confirm the substantial contributions of both
narratological guidance and knowledge distillation;
notably, distillation not only transfers the teacher’s
segmentation logic but also ensures near-perfect
output stability. Together, LitSeg and LitSeg-Lite
bridge the gap between narrative theory and practi-
cal RAG deployment, providing a robust and effi-
cient foundation for understanding literary texts.
Limitations
While our framework demonstrates strong perfor-
mance on English literary texts, its current evalua-
tion is confined to English. Generalizing LitSeg to
multilingual contexts remains an avenue for future
research.
Furthermore, dedicated benchmarks for the lit-
erary domain remain relatively scarce. Despite
achieving significant empirical gains across eval-
uated metrics, our evaluation is, as discussed pre-
viously, partially constrained by the inherent qual-
ity flaws prevalent in existing datasets. These
benchmark-level limitations prevent the quanti-
tative improvements from fully reflecting our
method’s true capabilities. A detailed qualitative
analysis of these flawed scenarios is presented in
Appendix C. Developing more robust benchmarks
for literary texts thus remains another vital direc-
tion for future research.
References
Satanjeev Banerjee and Alon Lavie. 2005. METEOR:
An automatic metric for MT evaluation with im-
proved correlation with human judgments. InPro-
ceedings of the ACL Workshop on Intrinsic and Ex-
trinsic Evaluation Measures for Machine Transla-tion and/or Summarization, pages 65–72, Ann Arbor,
Michigan. Association for Computational Linguis-
tics.
Steven Bird and Edward Loper. 2004. NLTK: The natu-
ral language toolkit. InProceedings of the ACL In-
teractive Poster and Demonstration Sessions, pages
214–217, Barcelona, Spain. Association for Compu-
tational Linguistics.
Tommaso Bonomo, Luca Gioffré, and Roberto Navigli.
2025. LiteraryQA: Towards effective evaluation of
long-document narrative QA. InProceedings of the
2025 Conference on Empirical Methods in Natural
Language Processing, pages 34086–34107, Suzhou,
China. Association for Computational Linguistics.
Claude Bremond and Elaine D Cancalon. 1980. The
logic of narrative possibilities.New Literary History,
11(3):387–411.
Samuel Henry Butcher and 1 others. 1902.The poetics
of Aristotle. Macmillan.
Harrison Chase. 2022. LangChain.
Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang
Tong, Saining Xie, Dale Schuurmans, Quoc V . Le,
Sergey Levine, and Yi Ma. 2025. Sft memorizes,
rl generalizes: A comparative study of foundation
model post-training.Preprint, arXiv:2501.17161.
André V . Duarte, João DS Marques, Miguel Graça,
Miguel Freire, Lei Li, and Arlindo L. Oliveira. 2024.
LumberChunker: Long-form narrative document seg-
mentation. InFindings of the Association for Com-
putational Linguistics: EMNLP 2024, pages 6473–
6486, Miami, Florida, USA. Association for Compu-
tational Linguistics.
Edward Morgan Forster. 1927.Aspects of the Novel.
Harcourt, Brace.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777,
Toronto, Canada. Association for Computational Lin-
guistics.
Gérard Genette. 1980.Narrative discourse: An essay
in method, volume 3. Cornell University Press.
Kyo Gerrits, Rik van Noord, and Ana Guerberof Arenas.
2026. Creativity bias: How machine evaluation strug-
gles with creativity in literary translations.Preprint,
arXiv:2605.13596.
Evelyn Gius and Michael Vauth. 2022. Towards an
event based plot model. a computational narratology
approach.Journal of Computational Literary Studies,
1(1).
Michael Hauge. 2017.Storytelling Made Easy: Per-
suade and Transform Your Audiences, Buyers, And
Clients-Simply, Quickly, and Profitably. BookBaby.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu
Chen. 2021. Lora: Low-rank adaptation of large
language models.CoRR, abs/2106.09685.
Haoyu Huang, Yongfeng Huang, Yang Junjie, Zhenyu
Pan, Yongqiang Chen, Kaili Ma, Hongzhi Chen, and
James Cheng. 2025. Retrieval-augmented generation
with hierarchical knowledge. InFindings of the Asso-
ciation for Computational Linguistics: EMNLP 2025,
pages 6044–6060, Suzhou, China. Association for
Computational Linguistics.
Chin-Yew Lin. 2004. ROUGE: A package for auto-
matic evaluation of summaries. InText Summariza-
tion Branches Out, pages 74–81, Barcelona, Spain.
Association for Computational Linguistics.
Genevieve Liveley. 2019.Narratology. Oxford Univer-
sity Press.
Yusong Ma, Hongxuan Nie, Chao Chen, Jiujie Zhang,
Jiali Jiang, Bisheng Wang, and Yuqin Xia. 2025. A
survey of retrieval-augmented generation (rag) for
large language models. In2025 International Con-
ference on Trustworthy Big Data and Artificial Intel-
ligence (ICTBAI), pages 7–13.
Uri Margolin. 2014.Simultaneity in Narrative, pages
777–786. De Gruyter, Berlin, München, Boston.
Pinelopi Papalampidi, Frank Keller, and Mirella Lapata.
2019. Movie plot analysis via turning point identi-
fication. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processing
and the 9th International Joint Conference on Natu-
ral Language Processing (EMNLP-IJCNLP), pages
1707–1717.
Andrew Piper and Sunyam Bagga. 2024. Using large
language models for understanding narrative dis-
course. InProceedings of the 6th Workshop on Nar-
rative Understanding, pages 37–46.
Andrew Piper, Richard Jean So, and David Bamman.
2021. Narrative theory for computational narrative
understanding. InProceedings of the 2021 confer-
ence on empirical methods in natural language pro-
cessing, pages 298–311.
Saikrishna Rajanidi, M. Anbazhagan, and G. R. Ramya.
2026. Rag in specialized domains: A survey of qa
chatbots. InData Science and Applications, pages
352–369, Cham. Springer Nature Switzerland.
Robert Scholes, James Phelan, and Robert Leland Kel-
logg. 2006.The nature of narrative: Revised and
expanded. OUP USA.
John Schulman, Filip Wolski, Prafulla Dhariwal,
Alec Radford, and Oleg Klimov. 2017. Prox-
imal policy optimization algorithms.Preprint,
arXiv:1707.06347.Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,
Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan
Zhang, Y . K. Li, Y . Wu, and Daya Guo. 2024.
Deepseekmath: Pushing the limits of mathemati-
cal reasoning in open language models.Preprint,
arXiv:2402.03300.
Dan Shen and Yali Wang. 2010.Western Narratol-
ogy: Classical and Post-Classical. Peking University
Press. In Chinese.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented
generation: A survey on agentic rag.Preprint,
arXiv:2501.09136.
Annalisa Szymanski, Noah Ziems, Heather A. Eicher-
Miller, Toby Jia-Jun Li, Meng Jiang, and Ronald A.
Metoyer. 2025. Limitations of the llm-as-a-judge
approach for evaluating llm outputs in expert knowl-
edge tasks. InProceedings of the 30th International
Conference on Intelligent User Interfaces, IUI ’25,
page 952–966, New York, NY , USA. Association for
Computing Machinery.
Kristin Thompson. 1999.Storytelling in the new Holly-
wood: Understanding classical narrative technique.
Harvard University Press.
Tzvetan Todorov and Arnold Weinstein. 1969. Struc-
tural analysis of narrative. InNOVEL: A forum on
fiction, volume 3, pages 70–76. JSTOR.
Yiteng Tu, Weihang Su, Yujia Zhou, Yiqun Liu, and
Qingyao Ai. 2025. Robust fine-tuning for retrieval
augmented generation against retrieval defects. In
Proceedings of the 48th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, SIGIR ’25, page 1272–1282, New
York, NY , USA. Association for Computing Machin-
ery.
Noa Visser Solissa, Andreas van Cranenburgh, and Fed-
erico Pianzola. 2025. Event detection between liter-
ary studies and nlp: A survey, a narratological reflec-
tion, and a case study.
Piek V ossen, Tommaso Caselli, and Roxane Segers.
2021. A narratology-based framework for storyline
extraction.Computational Analysis of Storylines:
Making Sense of Events, 125:125–140.
Eitan Wagner, Renana Keydar, Amit Pinchevski, and
Omri Abend. 2024. Automatic topic-guided segmen-
tation of holocaust survivor testimonies.Journal of
Computational Literary Studies, 2(1).
Ke Wang, Xiutian Zhao, Yanghui Li, and Wei Peng.
2023. M3seg: A maximum-minimum mutual infor-
mation paradigm for unsupervised topic segmenta-
tion in asr transcripts. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7928–7934.

Zhitong Wang, Cheng Gao, Chaojun Xiao, Yufei Huang,
Shuzheng Si, Kangyang Luo, Yuzhuo Bai, Wen-
hao Li, Tangjian Duan, Chuancheng Lv, Guoshan
Lu, Gang Chen, Fanchao Qi, and Maosong Sun.
2025. Document segmentation matters for retrieval-
augmented generation. InFindings of the Associa-
tion for Computational Linguistics: ACL 2025, pages
8063–8075, Vienna, Austria. Association for Compu-
tational Linguistics.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan,
Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin,
Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan
Tong, Chi Zhang, Mofan Zhang, Wang Zhang, and
16 others. 2025. Dapo: An open-source llm re-
inforcement learning system at scale.Preprint,
arXiv:2503.14476.
Ran Zhang, Wei Zhao, and Steffen Eger. 2025. How
good are LLMs for literary translation, really? liter-
ary translation evaluation with humans and LLMs.
InProceedings of the 2025 Conference of the Na-
tions of the Americas Chapter of the Association
for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers), pages 10961–
10988, Albuquerque, New Mexico. Association for
Computational Linguistics.
Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang,
Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tian-
wei Zhang, Guoyin Wang, and Fei Wu. 2026. In-
struction tuning for large language models: A survey.
ACM Comput. Surv., 58(7).
Jihao Zhao, Zhiyuan Ji, Yuchen Feng, Pengnian Qi,
Simin Niu, Bo Tang, Feiyu Xiong, and Zhiyu Li.
2024. Meta-chunking: Learning text segmentation
and semantic completion via logical perception.
Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhen-
gren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2026.
Retrieval-augmented generation for ai-generated con-
tent: A survey.Data Science and Engineering,
11(1):1–29.
A Details of Pairwise Evaluation
A.1 Expert Information and Annotation
Procedure
We recruited five volunteers to participate in the
evaluation. All volunteers are graduate students ma-
joring in English Literature, ensuring they possess
the necessary domain expertise, linguistic sensi-
tivity, and critical capacity to judge the nuances
of the text. The evaluation was conducted under
strict quality control to ensure objectivity and mini-
mize bias. We made sure all annotators completed
the tasks independently; no communication, dis-
cussion, or collaboration was permitted among the
volunteers during the entire annotation process. Athorough guideline was provided to maintain con-
sistency across all evaluations. For each sample,
the tri-binary preference (Win/Loss/Tie) of each
of the five annotators was recorded. We then com-
puted the average win and loss frequencies across
all five annotators for each instance as the overall
win rate of our strategy.
A.2 Guideline for Pairwise Evaluation
Thank you for participating in this evaluation. As a
literary professional, your scholarly expertise is
invaluable in assessing the accuracy, depth, and
nuance of these generated responses. Please use the
following guidelines to inform your judgment.
I. Evaluation Principle
Core Principle: "Factual Accuracy as the Baseline,
Informativeness as the Deciding Factor." Please rely
entirely on the provided Original Context as the
ultimate textual evidence. The Reference Answer is
provided merely as an auxiliary guide and may contain
interpretive or factual flaws; please apply your own
scholarly judgment and do not blindly trust it.
II. Evaluation Dimensions and Criteria
1. Dimension 1: Factual Accuracy and Textual Fidelity
This dimension serves as the absolute baseline. A
response that meets any "Fail" condition is deemed
factually invalid and fails the correctness check.
1.1 Textual Fidelity vs. Unwarranted Invention:
a) Pass: The response is strictly anchored in the
provided text and its claims are fully traceable to
the source.
b) Fail: The response invents unmentioned plot points,
fabricates character actions, or engages in baseless
subjective extrapolation beyond the textual evidence.
1.2 Decoding Narrative Subtext and True Intention:
a) Pass: The response successfully penetrates
characters'surface-level discourse (e.g., deception,
pretexts, politeness) to accurately reveal the
underlying narrative motives and dramatic irony.
b) Fail: The response relies superficially on literal
readings, mistakenly accepting character deceit or
posturing as truth, thereby distorting the narrative
logic.
1.3 Critical Engagement with Flawed Questions:
a) Pass: When confronted with questions containing
factual distortions or ambiguous referents, the
response actively corrects the flawed premise using
textual evidence.
b) Fail: The response succumbs to the query's flawed
assumptions, resulting in an invalid or logically
contradictory reading of the text.
2. Dimension 2: Informativeness and Interpretive Depth
Assuming both candidate responses meet the baseline of
factual accuracy, this dimension serves to
distinguish the superior reading.
2.1 Horizontal Breadth: Comprehensive Coverage of Parallel
Facts:
Superior: For complex events driven by concurrent
motivations or parallel actions, the response
synthesizes a comprehensive mosaic of textual clues.
This outperforms reductive answers that extract only
a single, isolated fact.
2.2 Longitudinal Breadth: Closure of Sequential Narrative
Arcs:
Superior: The response demonstrates an ability to
synthesize information across the text, capturing the
complete narrative arc from its initial setup to its
definitive resolution. This outperforms fragmented
answers that prematurely truncate an ongoing action.
2.3 Precision: Specificity of Textual Detail:

Superior: Supplying a more precise, fine-grained proper
noun or specific descriptive detail than the other
response demonstrates a higher-resolution close
reading and warrants a tie-breaking win.
III. Scoring Rules
Each pairwise comparison is evaluated as a zero-sum game (
awarding 1 point total per pair):
1. Win (Winner: 1.0, Loser: 0.0):
a) Superiority in Accuracy: One candidate response
passes the baseline check, while the other exhibits
one or more "Fail" conditions.
b) Superiority in Informativeness: Both candidate
responses pass the baseline accuracy check, but the
winning response meets a "Superior" condition by
demonstrating greater narrative breadth or more
precise textual detail.
2. Tie (0.5 points each): Both candidate responses meet a "
Fail" condition, or both are highly homogeneous in
their accuracy and informativeness.
A.3 Inter-Annotator Agreement
See Fig 2.
A1 A2 A3 A4 A5
AnnotatorA1
A2
A3
A4
A5AnnotatorA1=0.6109
Ag=74.85%=0.5581
Ag=71.31%=0.5512
Ag=70.85%=0.5281
Ag=69.38%
=0.6109
Ag=74.85%A2=0.5228
Ag=69.00%=0.5504
Ag=70.77%=0.5001
Ag=67.54%
=0.5581
Ag=71.31%=0.5228
Ag=69.00%A3=0.4938
Ag=67.00%=0.4873
Ag=66.62%
=0.5512
Ag=70.85%=0.5504
Ag=70.77%=0.4938
Ag=67.00%A4=0.4902
Ag=66.77%
=0.5281
Ag=69.38%=0.5001
Ag=67.54%=0.4873
Ag=66.62%=0.4902
Ag=66.77%A5
0.00.20.40.60.81.0
Cohen's kappa
Figure 2: Inter-annotator agreement heatmap for the
human pair-wise evaluation on GutenQA ( N= 1300 ).
This figure reports the pairwise Cohen’s kappa ( κ) and
raw agreement percentages (Ag) among five indepen-
dent annotators (A1, A2, A3, A4, and A5). The scores
demonstrate a reliable consensus, validating the robust-
ness of the human judgment used to mitigate the inher-
ent limitations of automated metrics in complex literary
QA tasks.
B Case Study on Baseline Chunking
Failure Modes
To provide a deeper qualitative understanding of
how our framework preserves narrative integrity
and enhances downstream question-answering
(QA) performance, we present three representative
case studies from our benchmarks. These exam-
ples contrast baseline segments against our distilled
model, LitSeg-Lite, illustrating three critical failure
modes of baseline chunking methods: prematureevent termination, mid-action truncation, and am-
biguous pronoun reference.
B.1 Overcoming Premature Event
Termination and Lack of Macro-Event
Perspective
The case study onBarchester Towersby Anthony
Trollope (illustrated in Fig 3a) exemplifies the fail-
ure mode of premature event termination. When
answering a question regarding Mrs. Bold’s re-
action to Mr. Slope’s proposal, traditional seg-
mentation methods impose a rigid boundary that
ends prematurely right after Eleanor expresses her
verbal disgust (“. . . unable to bear much more of
it.”). By terminating the segment early, the baseline
fails to capture the macro-perspective of the event—
specifically, its physical resolution. Deprived of the
full event horizon, the baseline QA model can only
infer an answer based solely on her verbal refusal,
delivering a fundamentally incomplete and incor-
rect response. Conversely, despite being a highly
compact model, LitSeg-Lite successfully internal-
izes macro-event boundaries by encapsulating the
entire proposal arc within a single chunk. By pre-
serving the definitive physical resolution where
she slaps him (“. . . dealt him a box on the ear. . . ”),
LitSeg-Lite equips the downstream generator with
the necessary context to formulate a precise and
complete answer.
B.2 Eliminating Mid-Action Truncation and
Omission of Crucial Information
As demonstrated in the analysis ofThe Vampyre
by John Polidori in Fig 3b, non-narratological ap-
proaches frequently suffer from mid-action trun-
cation, which breaks the textual sequence in the
middle of a continuous plotline. When querying
how Aubrey attempts to warn his sister, the base-
line chunker cuts the text abruptly at “but in vain.”,
entirely omitting the crucial information that imme-
diately follows: the acquisition of “pen and paper”
and the core action of “writing a letter”. This se-
vere information bottleneck leaves the generator in
an information vacuum, forcing the downstream
LLM to hallucinate a highly vivid but entirely fab-
ricated sequence regarding the trampling of a por-
trait. LitSeg-Lite effectively mitigates this issue
by respecting the sequential integrity of continu-
ous actions. LitSeg-Lite keeps the setup, the intent,
and the final action structurally intact within the
same context window, thereby ensuring clean and
complete retrieval of the golden evidence.

Q: What does Mrs. Bold do when Mr. Slope proposes?
à WRONG ANSWERàCORRECT ANSWER'Ah! Eleanor, will it not be sweet with the Lord's assistance, to travel hand in hand through this mortal valley which his mercies will make pleasant to us, till hereafter we shall dwell together at the foot of his throne?‘ ...'Ah! Eleanor—’ 'My name, MrSlope, is MrsBold,' said Eleanor, who, though determined to hear out the tale of his love, was too much disgusted by his blasphemy to be able to bear much more of it.Other Method Retrieves:... ‘Ah! Eleanor, will it not be sweet with the Lord’s assistance, to travel hand in hand through this mortal valley which his mercies will make pleasant to us, till hereafter we shall dwell together at the foot of his throne?‘ ...’Ah! Eleanor—’ ‘My name, MrSlope, is MrsBold,’ said Eleanor, who, though determined to hear out the tale of his love, was too much disgusted by his blasphemy to be able to bear much more of it. ‘Sweetest angel, be not so cold,’ said he, and as he said it the champagne broke forth, and he contrived to pass his arm around her waist. …She sprang from him as she would have jumped from an adder, …and then, quick as thought, she raised her little hand and dealt him a box on the ear with such right good will, that it sounded among the trees like a miniature thunder-clap. …Our Method Retrieves:
Other Method Answers:She responds by correcting him, stating 'My name, MrSlope, is MrsBold,' which implicitly rejects his proposal.Our Method Answers:She rejects him, corrects his address to "Mr. Slope," and eventually slaps him on the ear as he attempts to embrace her.【Std Ans: She slaps him.】
from Barchester Towers by Anthony Trollope(a) Case study onBarchester Towers. Baseline method fails to answer the question due to premature event termination.
Q: How does Aubrey attempt to tell his sister about Ruthven?
à WRONG ANSWERàCORRECT ANSWER…Lord Ruthven had called the morning after the drawing-room... Aubrey, when he was left by the physician and his guardians, attempted to bribe the servants, but in vain. Other Method Retrieves:…Lord Ruthven had called the morning after the drawing-room... Aubrey, when he was left by the physician and his guardians, attempted to bribe the servants, but in vain. He asked for pen and paper; it was given him; he wrote a letter to his sister, conjuring her, as she valued her own happiness, her own honour, and the honour of those now in the grave, who once held her in their arms as their hope and the hope of their house, to delay but for a few hours that marriage, on which he denounced the most heavycurses. …Our Method Retrieves:
Other Method Answers:He angrily trampled the portrait of Ruthven underfoot while congratulating her on their engagement.Our Method Answers:Aubrey attempts to write a secret letter to his sister asking her to delay the marriage and deliver it, but the physicians intercept the letter.【Std Ans: Through a letter.】
from The Vampyre by John PolidoriSo?
(b) Case study onThe Vampyre. Baseline method segments a coherent action into half, thereby failing to retrieve critical
information.
Figure 3: Qualitative comparison of text retrieval and QA results between baseline methods and our model
(Continued on the next page).

Q: What will prove Clifford's innocence?
à WRONG ANSWERàCORRECT ANSWERHis own death, so like that former one, yet attended by none of those suspicious circumstances, seems the stroke of God upon him, at once a punishment for his wickedness, and making plain the innocence of Clifford. But this flight,—it distorts everything! He may be in concealment, near at hand.Other Method Retrieves:…Now, there is a minute and almost exact similarity in the appearances connected with the death that occurred yesterday and those recorded of the death of Clifford’s uncle thirty years ago. It is true, there was a certain arrangement of circumstances... which made it possible... that old Jaffrey Pyncheon came to a violent death, and by Clifford's hands.” “They were arranged... by the man who sits in yonder parlor. …His own death, so like that former one, yet attended by none of those suspicious circumstances…making plain the innocence of Clifford. But this flight,—it distorts everything! He may be in concealment, near at hand. Could we but bring him back before the discovery of the Judge's death, the evil might be rectified.” “We must not hide this thing a moment longer!” said Phœbe. “…Clifford is innocent…” ...Our Method Retrieves:
Other Method Answers:Clifford's own death will prove his innocence.Our Method Answers:The Judge's death, which occurred in a manner similar to his ancestor's and without the suspicious circumstances that seemed to have orchestrated Clifford's uncle's death, will prove Clifford's innocence.【Standard Ans: New evidence in the crime.】?
from The House of the Seven Gables by Nathaniel Hawthorne(c) Case study onThe House of the Seven Gables. Our method correctly links narrative evidence while baseline method
misunderstands the text without sufficient context.
Figure 3: Qualitative comparison of text retrieval and QA results (Continued). Figures (a), (b), and (c) demonstrate
how LitSeg preserves narrative integrity and surpasses baseline methods in downstream performance.
B.3 Resolving Ambiguous Pronoun Reference
and Context Deprivation
Finally, the text fromThe House of the Seven
Gablesby Nathaniel Hawthorne, displayed in Fig
3c, highlights a pervasive linguistic vulnerability
in baseline text chunking methods: ambiguous pro-
noun reference resulting from context deprivation.
As marked by the dashed circle in the figure, the
baseline segment begins abruptly with an orphaned
possessive pronoun (“His own death. . . ”). Because
the upstream context containing the pronoun’s ac-
tual antecedent (the Judge) is completely cut off,
the baseline QA model suffers from reference mis-
alignment. Lacking the necessary contextual track-
ing to resolve the anaphora, the model misattributes
the pronoun toClifford, culminating in a logically
inverted and erroneous deduction. LitSeg-Lite ele-
gantly circumvents this limitation by dynamically
anchoring its boundaries around major narrative
shifts. By preserving the preceding context that in-
troduces the correct referent, LitSeg-Lite maintains
proper referential chains, allowing the reader model
to accurately synthesize the multi-layered plotline
and correctly establish Clifford’s innocence.C Case Study on Benchmark Quality
Issues
C.1 Ambiguous Phrasing or Factual Errors in
Questions
As highlighted in Fig 4, Some questions in the
dataset are ambiguous, flawed or incorrect, making
them unanswerable. On one hand, the benchmark
introduces severe linguistic ambiguity by employ-
ing unresolvable referents and incomplete semantic
predicates. Queries containing indeterminate pro-
nouns (e.g., "he") or isolated definite descriptions
(e.g., "the dog") fail to provide the localized con-
text necessary for proper coreference resolution,
leaving any evaluation of a model’s reading com-
prehension capacity completely invalid.
On the other hand, the dataset incorporates overt
factual errors and misleading premises that directly
invert or distort the logical relationships explicitly
stated in the source text. For example, by trun-
cating the phrasal verb "break out" to "break," the
question severely distorts the character’s physical
action. This is further compounded by blatant nar-
rative contradictions, such as asking how certain
characters convinced a counterparty to pay them
in gold, when the transaction is exactly opposite.
Forcing language models to navigate structurally

Flaw 1. Ambiguous Phrasing or Factual Errors in QuestionsQuestion: What type of dog is he trained to be in Canada? Who is “he”?Question: Who stole the dog?       Whichdog?
Question: How did Red Pierre and his companion convince McGuire to give themclothes and pay them in gold?     Factual error!!! It should be Red Pierre and his companion paying McGuire gold![Original Text] Jacqueline took one of them and Pierre the other under his left arm... “…here's what the clothes are worth to us.” And into the quaking hands of McGuirehe poured a chinking stream of gold pieces.Question: What did Buck have to break to win the wager?   “Break”? Or “break out”? [Original Text] “Buck can start a thousand pounds.” “And break it out? and walk off with it for a hundred yards?” demanded Matthewson... "And break it out, and walk off with it for a hundred yards”…Figure 4: An illustration of dataset flaws resulting from
ambiguous phrasing and factual errors in questions.
broken and factually misleading queries severely
compromises the integrity of the benchmark, pe-
nalizing high-performing models that detect these
contradictions, while artificially rewarding models
that succumb to hallucination or accept the query’s
flawed premises, thereby undermining the validity
of the evaluation metric.
C.2 Incomplete Ground-Truth Coverage
Question: What is the name of the dog that had to be shot? Answer: DaveFact: (1) Dave (2) DubQuestion: What does Reinhard do to impress Elisabeth?Answer: Writes fairy tales.       Fact: (1) Writes fairy tales. (2) Teaches her botany.(3)……Flaw 2. Incomplete Ground-Truth Coverage
Figure 5: An illustration of the incomplete ground-truth
coverage flaw within the dataset
As illustrated in Fig 5, the dataset also suffers
from incomplete ground-truth coverage, where theprovided answers fail to capture the full scope of
valid responses. This flaw manifests as a severe
under-reporting of facts, where questions with mul-
tiple, equally correct answers are falsely restricted
to a single target answer in the dataset. By omit-
ting alternative valid ground truths, this deficiency
creates a misleading evaluation bottleneck, unfairly
punishing models that successfully extract com-
plete and accurate information from the source text,
hindering the system’s ability to learn comprehen-
sive multi-perspective reasoning.
C.3 Literal Extractions Misinterpreting
Narrative Subtext
Question: Why does Elizabeth's father express concern about her acceptance of Mr. Darcy despite his wealth?Answer: Elizabeth's father is worried because he believes Mr. Darcy is a proud and unpleasant man.Flaw 3. Literal Extractions Misinterpreting Narrative Subtext
[Original Text] Have you any other objection,” said Elizabeth, “than your belief of my indifference?” “None at all. We all know him to be a proud, unpleasant sort of man; but this would be nothing if you really liked him…. But let me advise you to think better of it. I know your disposition, Lizzy. I know that you could be neither happy nor respectable, unless you truly esteemed your husband, unless you looked up to him as a superior. Your lively talents would place you in the greatest danger in an unequal marriage. You could scarcely escape discredit and misery. Better Answer: Because he believes she dislikes Darcy, warning that her lively disposition requires a husband she truly esteems to avoid the misery and discredit of an unequal marriage.
Figure 6: An illustration of dataset flaws stemming from
literal extractions misinterpreting narrative subtext.
As demonstrated in Fig 6, the dataset fails to
understand the connotations behind the text, but
relies on superficial keyword matching to construct
ground-truth answers, resulting in a severe reduc-
tion of literary meaning. In the first case, the
dataset’s literal answer mistakes a father’s voiced
observation for his actual motive, failing to decode
that his true anxiety stems from his daughter’s emo-
tional indifference and the possibility of an unequal
marriage. In the second case, the dataset naively
treats a character’s deceptive verbal pretense of
"delivering a message" as a literal fact, completely
blind to the narrative subtext and subsequent dra-
matic irony where the "message" is a fatal ambush

Question: Why did Pierre ask for McGurk at Gaffney's place?Answer: Because he had a messagespecifically meant for McGurk, which he did not want to pass through someone else.                             Flaw 3. Literal Extractions Misinterpreting Narrative Subtext (Continued)
[Original Text] 'Where's McGurk?' ...'What do you want with him?' 'Got a message for him.' 'Tell it to me, and I'll pass it along.' Pierre met the eye of the other and smiled faintly. 'Not this message.' 'Oh,' said the other, and then shouted: 'McGurk!' …The door opened and framed McGurk. He did not start, seeing Pierre. He said: "None of the rest of them had the guts even to bring me the message, eh?"..."That's not the message," answered a voice which Pierre did not recognize as his own. "Out with it, then." "It's in the leather on my hip." And he went for his gun...he scooped up the gun with his left and twisted. That movement made the third shot of McGurk fly wide and Pierre fired from the floor and saw a spasm of pain contract the face of the outlaw.Better Answer: Pierre asked for McGurk under the pretense of delivering a message, but his true intention was to confront and shoot him with the gun hidden on his hip.Figure 6: An illustration of dataset flaws stemming
from literal extractions misinterpreting narrative subtext
(Continued).
with a hidden firearm. By evaluating models based
on these literal readings, the dataset heavily penal-
izes advanced language systems capable of deep
narrative synthesis, ironic interpretation, and intent
modeling, thereby limiting its utility for complex
literary reading comprehension tasks.
C.4 Incomplete Reference Answers with
Insufficient Narrative Coverage
As demonstrated in Fig 7, the text constructs Ry-
der’s motivations across two distinct phases: first,
his letter explicitly states a dual purpose (revealing
Pierre’s illegitimate birth and seeing him before
dying); second, upon meeting, he makes his ul-
timate concrete request (burial in a specific plot).
Despite the significant narrative span between these
events, a valid reading comprehension evaluation
requires cross-contextual synthesis. By fragment-
ing this continuous arc, the incomplete ground truth
unfairly penalizes models capable of multi-hop ev-
idence integration.
C.5 Entity Name Variations and Granularity
Mismatches
As illustrated in Fig 8, the dataset enforces a single,
rigid ground-truth string, failing to account for the
Question: What is the main reason Martin Ryder is reaching out to Pierre? Answer: Because he fears dying alone and unattended, with his sons waiting to take his savings and leave him to rot, hoping that Pierre might come before he passes away.            Flaw 4. Reference Answers Lacking Narrative Breadth 
[Original Text](1) 'Morgantown, R.F.D. No. 4. SON PIERRE:...So I'm writing to you, Pierre, part to tell you what you ought to know; part because I got a sort of crazy idea that maybe you could get down here to me before I go out.(2)Son, you didn‘t come none too soon. ...Here’s all I ask: ...And down in the middle of Morgantown is the buryin‘-ground. I’ve ridden past it a thousand times an‘ watched a corner plot, where the grass grows quicker than it does anywhereselse in the cemetery. Pierre, I’d die plumb easy if I knew I was goin‘ to sleep the rest of time in that place." Our Answer: The main reason Martin Ryder is reaching out to Pierre is to reveal that Pierre is his illegitimate sonand to ask Pierre to help him arrange his burialin a specific corner plot at the Morgantown graveyard in his dying moments.Figure 7: An illustration of dataset flaws due to in-
complete reference answers with insufficient narrative
coverage.
Question: Who steals Buck from Judge Miller's home? Answer: The gardener's assistant. Flaw 5. Entity Name Variations and Granularity MismatchesOurAnswer: Manuel, a gardener's helper who owed money to a stranger, steals Buck from Judge Miller's home.
Question: What defeats Zog? Answer: The forces of good. OurAnswer: King Ankodefeats Zog by catching him in his coils and crushing his magic power with fear, causing him to turn into a mass of jelly-like pulp and die.Question: Who was Eva?Answer: Daughter of a Jewish financier that Tancred met. OurAnswer: Eva was the daughter of Besso, ….
Figure 8: An illustration of dataset flaws rooted in entity
name variations and granularity mismatches.

multi-dimensional ways a correct answer can be ex-
pressed, and consequently penalizing semantically
valid responses.
This rigidity becomes particularly problematic
when dealing with entity name variations, where
a single narrative character is naturally referenced
across a text via diverse linguistic expressions, such
as proper names, titles, or descriptive phrases. For
instance, a model extracting the precise proper
noun "Manuel" or "Besso" is falsely penalized sim-
ply because the ground truth records the functional
equivalents "the gardener’s assistant" or "a Jewish
financier."
This problem is further exacerbated when these
variations shift across levels of narrative scale, in-
troducing profound granularity mismatches where
valid responses operate at differing tiers of abstrac-
tion. Depending on the depth of text synthesis, an
event may be legitimately framed macroscopically
through broad thematic forces and collective fac-
tions, or microscopically through individual agents
and localized actions. As demonstrated in the text,
attributing an antagonist’s defeat to "the forces of
good" versus the physical intervention of "King
Anko" represents a divergence in granularity rather
than accuracy, yet both assertions remain factually
grounded. By failing to bridge these parallel axes
of linguistic variation and narrative scale, the eval-
uation metric introduces substantial noise, system-
atically penalizing advanced models that capture
high-precision, fine-grained entities.
D Full Prompts
D.1 Prompt for LitSeg
System Prompt Shared Across Stages 1–3
# Role: Narratology Segmentation Expert
## Profile:
- Language: English
- Description: You are a professional literary editor and
Natural Language Processing expert deeply versed in
structuralism and narratology. Your core task is to
perform fine-grained plot segmentation on literary
texts.
## Core Competencies
1. Fine-grained plot segmentation preserving original
literary aesthetics.
2. Extracting valid dynamic events and effectively
filtering out static or non-narrative noise.
3. Untangling complex multi-line, parallel, and nested
narrative structures based on "Unity of Action" and
internal story logic.
4. Pinpointing logical breakpoints by synthesizing macro-
structural shifts (spatial, temporal, tension-based,
etc.) with micro-level evaluations of semantic
dependency.
## Primary Objectives
1. Perform logical, narratology-driven plot segmentation on

literary texts.
2. Enable readers to independently read and understand each
segmented snippet out of its full context.
3. Output the exact analytical and segmentation results in
strict JSON format.
## Constraints
1. Role Consistency: Don't break character or bypass the
narratological framework under any circumstance.
2. Narrative-Driven Granularity: The recommended maximum
length for any single segment is 100 sentences. This
is a guidance threshold rather than an absolute hard
cap. When narrative coherence clearly requires it, a
segment may exceed this limit. NEVER unnaturally
merge distinct narrative blocks just to hit a
sentence count. Abandon any mathematical division
mindset. Breakpoints must be dictated organically by
narratological theories.
3. Conditional Contextual Overlap: A small overlap of key
sentences between adjacent plot blocks is allowed
ONLY to ensure narrative continuity. This is NOT
mandatory. If a natural narrative breakpoint is clear
and coherent, make a clean, hard cut. Do not blindly
copy previous sentences just for the sake of
overlapping.
4. Descriptive Subtitle: Extract a highly concise subtitle
that summarizes the core action or reversal event for
each plot block.
5. Full Coverage and Order: You must segment the entire
chapter in its original order. Do not skip any
sentences or paragraphs. Every sentence must belong
to at least one segment, i.e., covered in the index
range between "from_idx" and "to_idx" of at least one
segment.
6. Indexing Rule: The "from_idx" of the first segment MUST
be 1. The "to_idx" of the last segment MUST be the
last index of the chapter. Indices are 1-based and
inclusive like [1] in the original text.
7. Context Indices: For sentences crucial for understanding
a segment but outside its main range, include their
indices in the "context_idx" field. Sentences cannot
ONLY exist in "context_idx"; they must be part of
some segment's main range.
## Narratological Workflow
When determining logical breakpoints for plot segmentation,
you must execute the following three-step
narratological framework sequentially:
### Step 1: Extract Valid Events and Filter Noise
1. Focus on Dynamic Changes: Treat the plot as a sequence
of dynamic events. Ignore static character portraits
or pure scenery descriptions. Extract an event ONLY
when characters or objects undergo dynamic changes.
2. Filter by Event Type: Retain "Changes of state" (
physical or mental state changes) and "Process events
" (actions or happenings without state change, such
as talking, thinking, and feeling) as core events.
Discard "Stative events" (static physical or mental
states) and "Non-events" (generic statements,
counterfactuals, questions) to aggressively reduce
noise.
3. Handling Ambiguous Paragraphs: For borderline paragraphs
, gauge narrativity by checking for: a situation, an
agent, one or more sequential actions, a potential
object, a spatial location, a temporal specification,
and a rationale. A higher density of these features
indicates a valid, independent event.
### Step 2: Untangle Narrative Threads and Clarify
Structures
Group the events extracted in Step 1 into clear narrative
threads using the following criteria:
1. Apply "Unity of Action": Group events by a complete core
action, abandoning the "same character" stereotype.
Separate unrelated actions performed by the same
character; merge joint actions advanced by different
characters.
2. Track the "Revelation of Secrets": Use the gradual
unfolding of author-planted secrets as a tight
internal logic to connect scattered events into a
cohesive narrative chain.
3. Clarify Parallel & Minor Threads: Explicitly identify
and segment concurrent event chains (e.g., A1 -> B1
-> A2 -> B2). Never merge minor threads into major
ones; preserve their structural independence.4. Identify Nested Hierarchy: Classify the thread's
narrative level: [Extradiegetic] (story introduction)
, [Intradiegetic] (core main plot), or [Metadiegetic]
(a story within a story).
5. Establish Basic Data: For each thread, map the timeline,
causeline, and critical shift points in time, space,
and perspective.
### Step 3: Locate Key Turning Points and Execute
Segmentation
Use the data established in Step 2 to locate logical
breakpoints. Apply these markers organically,
segmenting the text only when shifts clearly and
naturally occur:
1. Macro-Structural & Plot Tension: Track tension through
the overarching arc (Exposition -> Predicament ->
Extrication). Consider segmenting the text at the
boundaries or transitions between these major driving
events:
- Opportunity: The introductory event that triggers the
story after the background is set.
- Change of Plans: The event where the main goal is
defined and action intensifies.
- Point of No Return: The event pushing characters to
fully commit to their goal.
- Major Setback & Reversal: The event where the
situation completely falls apart, marking a formal
reversal of fortune (peripeteia). This generally
serves as a strong signal for segmentation.
- Climax & Recognition: The final resolution and peak
tension, often accompanied by a cognitive shift from
ignorance to truth (anagnorisis).
- Aftermath & Buffer: Post-climax events that resolve
the predicament and lower tension, smoothly
transitioning into the final Extrication phase.
2. Contextual Shifts: Segment at definitive leaps in time,
space, or perspective. Cut when the narrative
switches between parallel threads or crosses nested
narrative levels.
3. Micro-Dynamic Shifts (State & Action): Segment when a
situation's fundamental state transitions (
equilibrium -> disequilibrium -> new equilibrium) or
at the precise moment an action's outcome (success/
failure) is revealed.
4. Sentence-Level Boundary Pinpointing: After narrowing
down a general turning point, actively evaluate the
semantic dependency between adjacent sentences to
pinpoint the exact breakpoint:
- High Semantic Cohesion Zone (Avoid Segmenting): Within
the scope of the potential turning point, identify
sentences with strong semantic dependency or
interlocking logic. Examples include adjacency pairs
(e.g., question/answer, attack/defense), continuous
pronoun chains, or an immediate physical/emotional
reaction to a specific action. Avoid executing a cut
within these sequences.
- Low Semantic Cohesion Zone (Execute the Cut): Within
the scope of the potential turning point, locate the
exact sentence where the semantic dependency drops to
its lowest level. Make the cut right before the
sentence that can semantically stand alone--one that
initiates a new topic, phase, or action, and requires
minimal context from the immediately preceding
sentence to be fully understood.
## Output Specifications
1. JSON Format: You must output STRICT JSON format. Escape
all internal double quotes within JSON string fields
using a backslash.
2. Content Requirement: For each segmented plot block,
output a concise subtitle summarizing the core action
, and the index ranges based on the constraints. The
exact JSON structure will be defined in the user
prompt based on the specific Step being executed.
## Initialization:
As a Narratology Segmentation Expert, you must strictly
follow the Constraints and the Narratological
Workflow. To initiate the analysis, await the
provision of the target Step and the Chapter Text
Content. Upon receipt, output the segmentation
results exclusively in the requested STRICT JSON
format.

Stage-1 User Prompt
You are executing ONLY Step 1 of the Workflow:
Extract Valid Events and Filter Noise.
[Chapter Text Content]
{chapter_content}
Return STRICT JSON only (no markdown):
{
"step_1_events": "A string briefly listing the core
dynamic events extracted (filtering out noise)."
}
Stage-2 User Prompt
You are executing ONLY Step 2: Untangle Narrative Threads
and Clarify Structures.
Use both the chapter text and Step 1 result as input
context.
[Chapter Text Content]
{chapter_content}
[Step 1 Result JSON]
{step_1_result}
Return STRICT JSON only (no markdown):
{
"step_2_threads": "A string briefly describing the
narrative threads and their nested hierarchy, based
on unity of action, secrets, and key shifts in time,
space, or perspective."
}
Stage-3 User Prompt
You are executing ONLY Step 3: Locate Key Turning Points
and Execute Segmentation.
Use the chapter text, Step 1 result, and Step 2 result as
input.
[Chapter Text Content]
{chapter_content}
[Step 1 Result JSON]
{step_1_result}
[Step 2 Result JSON]
{step_2_result}
Return STRICT JSON only (no markdown):
{
"chapter_title": "The chapter title or identifier",
"step_3_turning_points": [
{
"type": "The type of turning point, do not force all
predefined types.",
"description": "A string briefly describing the
specific event or shift that constitutes this turning
point, explicitly justified by a macro/micro
narrative transition or a drop in semantic cohesion."
}
],
"segments": [
{
"subtitle": "The subtitle of the plot block",
"from_idx": "Integer start index, with optional
overlap with previous segment if necessary. First
segment must start with index 1.",
"to_idx": "Integer end index, with optional overlap
with next segment if necessary. Last segment must end
with index {last_index}.",
"context_idx": "List of indices of sentences that
provide necessary context, if necessary."
}
]
}D.2 Prompt for LitSeg-Lite
System Prompt Single-Stage
# Role: Narratology Segmentation Expert
## Profile:
- Language: English
- Description: You are a professional literary editor and
Natural Language Processing expert deeply versed in
structuralism and narratology. Your core task is to
perform fine-grained plot segmentation on literary
texts.
## Core Competencies
1. Fine-grained plot segmentation preserving original
literary aesthetics.
2. Extracting valid dynamic events and effectively
filtering out static or non-narrative noise.
3. Untangling complex multi-line, parallel, and nested
narrative structures based on "Unity of Action" and
internal story logic.
4. Pinpointing logical breakpoints by synthesizing macro-
structural shifts (spatial, temporal, tension-based,
etc.) with micro-level evaluations of semantic
dependency
## Primary Objectives
1. Perform logical, narratology-driven plot segmentation on
literary texts.
2. Enable readers to independently read and understand each
segmented snippet out of its full context.
3. Output the exact analytical and segmentation results in
strict JSON format.
## Constraints
1. Role Consistency: Don't break character or bypass the
narratological framework under any circumstance.
2. Narrative-Driven Granularity: The recommended maximum
length for any single segment is 100 sentences. This
is a guidance threshold rather than an absolute hard
cap. When narrative coherence clearly requires it, a
segment may exceed this limit. NEVER unnaturally
merge distinct narrative blocks just to hit a
sentence count. Abandon any "mathematical division"
mindset. Breakpoints must be dictated organically by
narratological theories.
3. Conditional Contextual Overlap: A small overlap of key
sentences between adjacent plot blocks is allowed
ONLY to ensure narrative continuity. This is NOT
mandatory. If a natural narrative breakpoint is clear
and coherent, make a clean, hard cut. Do not blindly
copy previous sentences just for the sake of
overlapping.
4. Descriptive Subtitle: Extract a highly concise subtitle
that summarizes the core action or reversal event for
each plot block.
5. Full Coverage and Order: You must segment the entire
chapter in its original order. Do not skip any
sentences or paragraphs. Every sentence must belong
to at least one segment, i.e., covered in the index
range between "from_idx" and "to_idx" of at least one
segment.
6. Indexing Rule: The "from_idx" of the first segment MUST
be 1. The "to_idx" of the last segment MUST be the
last index of the chapter. Indices are 1-based and
inclusive like [1] in the original text.
7. Context Indices: For sentences crucial for understanding
a segment but outside its main range, include their
indices in the "context_idx" field. Sentences cannot
ONLY exist in "context_idx"; they must be part of
some segment's main range.
## Narratological Workflow
When determining logical breakpoints for plot segmentation,
you must execute the following three-step
narratological framework sequentially:
### Step 1: Extract Valid Events and Filter Noise
1. Focus on "Dynamic" Changes: Treat the plot as a sequence
of dynamic events. Ignore static character portraits
or pure scenery descriptions. Extract an event ONLY
when characters or objects undergo dynamic changes.
2. Filter by Event Type: Retain "Changes of state" (
physical or mental state changes) and "Process events
" (actions or happenings without state change, such

as talking, thinking, and feeling) as core events.
Discard "Stative events" (static physical or mental
states) and "Non-events" (generic statements,
counterfactuals, questions) to aggressively reduce
noise.
3. Handling Ambiguous Paragraphs: For borderline paragraphs
, gauge narrativity by checking for: a situation, an
agent, one or more sequential actions, a potential
object, a spatial location, a temporal specification,
and a rationale. A higher density of these features
indicates a valid, independent event.
### Step 2: Untangle Narrative Threads and Clarify
Structures
Group the events extracted in Step 1 into clear narrative
threads using the following criteria:
1. Apply "Unity of Action": Group events by a complete core
action, abandoning the "same character" stereotype.
Separate unrelated actions performed by the same
character; merge joint actions advanced by different
characters.
2. Track the "Revelation of Secrets": Use the gradual
unfolding of author-planted secrets as a tight
internal logic to connect scattered events into a
cohesive narrative chain.
3. Clarify Parallel & Minor Threads: Explicitly identify
and segment concurrent event chains (e.g., A1 -> B1
-> A2 -> B2). Never merge minor threads into major
ones; preserve their structural independence.
4. Identify Nested Hierarchy: Classify the thread's
narrative level: [Extradiegetic] (story introduction)
, [Intradiegetic] (core main plot), or [Metadiegetic]
(a story within a story).
5. Establish Basic Data: For each thread, map the timeline,
causeline, and critical shift points in time, space,
and perspective.
### Step 3: Locate Key Turning Points and Execute
Segmentation
Use the data established in Step 2 to locate logical
breakpoints. Apply these markers organically,
segmenting the text only when shifts clearly and
naturally occur:
1. Macro-Structural & Plot Tension: Track tension through
the overarching arc (Exposition -> Predicament ->
Extrication). Consider segmenting the text at the
boundaries or transitions between these major driving
events:
- Opportunity: The introductory event that triggers the
story after the background is set.
- Change of Plans: The event where the main goal is
defined and action intensifies.
- Point of No Return: The event pushing characters to
fully commit to their goal.
- Major Setback & Reversal: The event where the
situation completely falls apart, marking a formal
reversal of fortune (peripeteia). This generally
serves as a strong signal for segmentation.
- Climax & Recognition: The final resolution and peak
tension, often accompanied by a cognitive shift from
ignorance to truth (anagnorisis).
- Aftermath & Buffer: Post-climax events that resolve
the predicament and lower tension, smoothly
transitioning into the final Extrication phase.
2. Contextual Shifts: Segment at definitive leaps in time,
space, or perspective. Cut when the narrative
switches between parallel threads or crosses nested
narrative levels.
3. Micro-Dynamic Shifts (State & Action): Segment when a
situation's fundamental state transitions (
equilibrium -> disequilibrium -> new equilibrium) or
at the precise moment an action's outcome (success/
failure) is revealed.
4. Sentence-Level Boundary Pinpointing: After narrowing
down a general turning point, actively evaluate the
semantic dependency between adjacent sentences to
pinpoint the exact breakpoint:
- High Semantic Cohesion Zone (Avoid Segmenting): Within
the scope of the potential turning point, identify
sentences with strong semantic dependency or
interlocking logic. Examples include adjacency pairs
(e.g., question/answer, attack/defense), continuous
pronoun chains, or an immediate physical/emotional
reaction to a specific action. Avoid executing a cut
within these sequences.
- Low Semantic Cohesion Zone (Execute the Cut): Withinthe scope of the potential turning point, locate the
exact sentence where the semantic dependency drops to
its lowest level. Make the cut right before the
sentence that can semantically stand alone--one that
initiates a new topic, phase, or action, and requires
minimal context from the immediately preceding
sentence to be fully understood.
## Output Specifications
1. JSON Format: You must output STRICT JSON format. Escape
all internal double quotes within JSON string fields
using a backslash.
2. Content Requirement: For each segmented plot block,
output a concise subtitle summarizing the core action
, and the index ranges based on the constraints.
## Initialization:
As a Narratology Segmentation Expert, you must strictly
follow the Constraints and the Narratological
Workflow. To initiate the analysis, await the
provision of the Chapter Text Content. Upon receipt,
execute all three workflow steps internally and
output the segmentation results exclusively in the
requested STRICT JSON format.
Return STRICT JSON only (no markdown):
{
"step1": {
"step_1_events": "A string briefly listing the core
dynamic events extracted (filtering out noise)."
},
"step2": {
"step_2_threads": "A string briefly describing the
narrative threads and their nested hierarchy, based
on unity of action, secrets, and key shifts in time,
space, or perspective."
},
"step3": {
"chapter_title": "The chapter title or identifier",
"step_3_turning_points": [
{
"type": "The type of turning point, do not force
all predefined types.",
"description": "A string briefly describing the
specific event or shift that constitutes this turning
point, explicitly justified by a macro/micro
narrative transition or a drop in semantic cohesion."
}
],
"segments": [
{
"subtitle": "The subtitle of the plot block",
"from_idx": "Integer of the index where the segment
starts, with optional overlap with previous segment
if necessary. The first segment must start with index
1.",
"to_idx": "Integer of the index where the segment
ends, with optional overlap with next segment if
necessary. The last segment must end with index {
last_index}.",
"context_idx": "List of indices of sentences that
provide necessary context for understanding this
segment, if necessary."
}
]
}
}
D.3 Prompt for Reward Model
Reward Model System Prompt
# Role and Task
You are a Senior Narratologist and NLP Evaluation Expert.
Your task is to evaluate the provided segment against
the Original Passage based on three primary
dimensions: Event Validity, Event Phase & Unity of
Action, and Cut Point Logic. Evaluate each chunk.
# Evaluation Dimensions and Criteria

## Dimension 1: Event Validity
- Evaluation Scope: Focuses on two core areas: dynamic
event extraction and fidelity to the original text.
Dynamic events include "Changes of state" (physical
or mental state changes) and "Process events" (
actions or happenings without state change, such as
talking, thinking, and feeling). Assess whether the
segment accurately captures an event. Summaries must
be precise, avoiding subjective hallucinations or
distortion of facts.
- Core Pain Points: Deviating from the dynamic narrative
logic of the source; composed entirely of static
description, with no narrative events; subtitles that
are too vague (e.g., "The Island"); subtitles that
focus on a minor detail while ignoring the main plot
movement; misalignment between the title and the
actual text.
Scoring Criteria:
- 5 (Excellent): The entire segment revolves around a clear
, dynamic core event that is self-contained,
featuring a complete narrative function within the
chunk. Static descriptions (setting, character
psychology, background, etc.) are naturally preserved
as narrative support and are seamlessly integrated
to serve the progression of the plot. The subtitle
perfectly captures the core of the event.
- 4 (Good): The segment contains a clear dynamic event that
is largely self-contained. Although the proportion
of static description or exposition within the block
is relatively high, the primary narrative axis
remains easily identifiable, and the dynamic movement
is not obscured. The subtitle fidelity is acceptable.
- 3 (Passing): The segment contains at least one minor
valid event (e.g., a few lines of dialogue or a micro
-action) but captures only a fragment or a partial
slice of a larger event. Lengthy static descriptions
or background information heavily dilute the sense of
dynamic momentum in both length and perception. The
subtitle fidelity is acceptable.
- 2 (Poor): The vast majority of the segment consists of
static settings or chatter unrelated to the current
plot progression. The narrative is completely
fragmented, severely lacking dynamic change and
leaving the reader with a sense of narrative
stagnation. The subtitle may contain noticeable
factual distortions. There is a significant deviation
from narrative logic.
- 1 (Failure): 100%
- Special Notes: Assign a score of 5.0 to any chunk that
contains the entirety of the author's prologue/
monologue, background introduction, or copyright
information, regardless of the above scoring criteria.
## Dimension 2: Event Phase and Unity of Action
- Evaluation Scope: Focuses exclusively on the scope/focus
of events within the segment. Assess whether the
segment strictly adheres to the "Unity of Action" (a
single phase or a single thread) and whether it
merges multiple actions that should have remained
independent.
- Core Pain Points: "Stitching" together unrelated parallel
threads, actions of irrelevant characters, or multi-
level/nested events into a single segment that could
have been split into independent phases.
Scoring Criteria:
- 5 Points (Excellent): Perfect focus on a single phase.
Strictly locked into a pure action phase where the
cause and effect of the event are highly cohesive.
- 4 Points (Good): Clear primary action. Focuses on a
single event segment, occasionally including a very
minor prelude (one or two sentences) to the next
action phase without feeling bloated.
- 3 Points (Passing): Cluttered multi-phase actions. A
clear main thread exists, but the block contains two
or more consecutive events that could have been
clearly divided into independent segments, causing
the focus to lose its singularity.
- 2 Points (Poor): Forced stitching of unrelated threads.Unrelated parallel threads or different narrative
levels (e.g., reality mixed with long flashbacks) are
forced into the same block.
- 1 Point (Failure): Complete fragmentation of events. A
total "mishmash" where small fragments within the
block have no temporal or logical connection in terms
of action.
## Dimension 3: Cut Point Logic and Overlapping
Connectivity
- Evaluation Scope: Focuses on the "placement of the cut"
between each chunk and the neighboring one(s): , and
the "contextual bridging mechanism." Assess whether
the segmentation breaks a tight causal chain and
whether "Overlap" is reasonably utilized to ensure
the segment is independently readable.
- Critical Deficiencies:
- Front-end Fragmentation: Setting a chunk boundary
directly within a high semantic cohesion region
without "Overlap" buffer effectively "beheads" the
context, severely compromising semantic integrity and
rendering the segment unintelligible in isolation.
- Back-end Truncation: Setting a chunk boundary that
prematurely severs the causal link prevents the
eventual outcome or resolution of the event from
being included, resulting in an incomplete narrative
arc.
Scoring Criteria:
- 5 Points (Excellent): The cut points falls precisely at a
natural transition zone of low semantic cohesion. If
cutting within a tight action sequence, a highly
logical Overlap mechanism is used to keep the causal
chain intact, making the segment fully readable and
self-contained on its own.
- 4 Points (Good): Natural cut and basic independence. The
cut position is logical. Minor contextual dependency
exists and the Overlap mechanism isn't fully utilized
, or the Overlap is slightly redundant but doesn't
hinder independent reading.
- 3 Points (Passing): Lacks buffering and depends on
context. A "hard cut" was made between sentences with
some causal connection (without reasonable Overlap
buffering). When read out of context, the reader must
exert effort to "fill in the blanks" to follow the
flow.
- 2 Points (Poor): Illogical cut point and difficult to
read independently. Directly cuts through a zone of
extremely high semantic cohesion (e.g., continuous
causal chains or compact actions) without any
bridging buffer, leaving the segment nonsensical
without the preceding text, or suspended without the
following text.
- 1 Point (Failure): Purely mechanical/blind cutting. An
absurd physical break that completely destroys basic
semantic meaning.
- Special Notes: If the overlap buffer contains sentences
entirely from within the current chunk, it
constitutes an improper implementation of the
mechanism. Consequently, a **0.5-point penalty** will
be deducted from the final score. If the overlap
buffer is utilized effectively and provides crucial
supplementary context that allows the chunk to be
fully understood in isolation, an **additional 0.5
points** will be awarded.
# Output Format
Output must be in valid JSON. For each dimension, return a
list containing the score for each individual chunk.
Example (suppose there are 4 chunks in the segment):
{
"dimension1": [1.5, 1.0, 1.0, 1.0],
"dimension2": [1.0, 1.5, 1.0, 1.0],
"dimension3": [1.0, 1.0, 1.5, 1.0]
}
Reward Model User Prompt Template
Segment Pipeline Prompt:
{segment_pipeline_prompt}

Original Passage:
{original_passage}
Segment Pipeline:
{segment_pipeline}
D.4 Prompt for Ablation Studies
System Prompt for Ablation w/o Theory-guided Seg-
mentation
# Role: Semantic Segmentation Expert
## Profile:
- Language: English
- Description: You are an expert in breaking down texts
into segments. Your sole task is to split the text
based on semantic content.
## Core Competencies
1. Detecting semantic transitions and boundaries.
2. Producing segments that can be read and understood
clearly.
3. Ensuring every sentence is assigned to a segment, with
no omissions and no reordering.
## Primary Objectives
1. Perform a semantic segmentation of the provided text.
2. For each segment, output:
- A concise subtitle that captures its main idea or
subject.
- The inclusive sentence index range it covers.
3. Output the segmentation in strict JSON format.
## Constraints
1. **Full Coverage and Original Order**: Segment the entire
text in its original sentence order. Do not skip any
sentence. Every sentence must belong to at least one
segment's main index range.
2. **Indexing Rule**: Indices are 1-based and inclusive.
The "from_idx" of the first segment must be 1. The "
to_idx" of the last segment must equal the last index
of sentences in the text. Indices are like [1] in
the original text.
3. **Context Indices**: If a segment requires a sentence
from outside its main range to be fully
comprehensible, you may list that sentence's index in
"context_idx". However, a sentence cannot exist *
only* in "context_idx"; every sentence must appear as
a main member of exactly one segment (i.e., within
its "from_idx"-"to_idx" range).
4. **No External Knowledge**: Segment based only on the
semantic content of the text itself. Do not use any
external knowledge or assumptions about the text.
5. **Semantic Clarity**: Each segment should represent a
clear and coherent meaning.
6. **Segment Length**: The recommended maximum length for
any single segment is 100 sentences. This is a
guidance threshold rather than an absolute hard cap.
When necessary, a segment may exceed this limit. Do
not artificially merge or split to reach a target
length.
7. **Subtitle**: Provide a very short, descriptive subtitle
for each segment.
Return STRICT JSON only (no markdown):
{
"chapter_title": "The chapter title or identifier",
"segments": [
{
"subtitle": "The subtitle of the plot block",
"from_idx": "Integer of the index where the
segment starts, with optional overlap with previous
segment if necessary. The first segment must start
with index 1.",
"to_idx": "Integer of the index where the
segment ends, with optional overlap with next segment
if necessary. The last segment must end with index {
last_index}.",
"context_idx": "List of indices of sentences
that provide necessary context for understanding this
segment, if necessary."}
]
}
System Prompt for Ablation w/o Event Extraction
# Role: Narratology Segmentation Expert
## Profile:
- Language: English
- Description: You are a professional literary editor and
Natural Language Processing expert deeply versed in
structuralism and narratology. Your core task is to
perform fine-grained plot segmentation on literary
texts.
## Core Competencies
1. Fine-grained plot segmentation preserving original
literary aesthetics.
2. Untangling complex multi-line, parallel, and nested
narrative structures based on "Unity of Action" and
internal story logic.
3. Pinpointing logical breakpoints by synthesizing macro-
structural shifts (spatial, temporal, tension-based,
etc.) with micro-level evaluations of semantic
dependency.
## Primary Objectives
1. Perform logical, narratology-driven plot segmentation on
literary texts.
2. Enable readers to independently read and understand each
segmented snippet out of its full context.
3. Output the exact analytical and segmentation results in
strict JSON format.
## Constraints
1. Role Consistency: Don't break character or bypass the
narratological framework under any circumstance.
2. Narrative-Driven Granularity: The recommended maximum
length for any single segment is 100 sentences. This
is a guidance threshold rather than an absolute hard
cap. When narrative coherence clearly requires it, a
segment may exceed this limit. NEVER unnaturally
merge distinct narrative blocks just to hit a
sentence count. Abandon any "mathematical division"
mindset. Breakpoints must be dictated organically by
narratological theories.
3. Conditional Contextual Overlap: A small overlap of key
sentences between adjacent plot blocks is allowed
ONLY to ensure narrative continuity. This is NOT
mandatory. If a natural narrative breakpoint is clear
and coherent, make a clean, hard cut. Do not blindly
copy previous sentences just for the sake of
overlapping.
4. Descriptive Subtitle: Extract a highly concise subtitle
that summarizes the core action or reversal event for
each plot block.
5. Full Coverage and Order: You must segment the entire
chapter in its original order. Do not skip any
sentences or paragraphs. Every sentence must belong
to at least one segment, i.e., covered in the index
range between "from_idx" and "to_idx" of at least one
segment.
6. Indexing Rule: The "from_idx" of the first segment MUST
be 1. The "to_idx" of the last segment MUST be the
last index of the chapter. Indices are 1-based and
inclusive like [1] in the original text.
7. Context Indices: For sentences crucial for understanding
a segment but outside its main range, include their
indices in the "context_idx" field. Sentences cannot
ONLY exist in "context_idx"; they must be part of
some segment's main range.
## Narratological Workflow
When determining logical breakpoints for plot segmentation,
you must execute the following two-step
narratological framework sequentially:
### Step 1: Untangle Narrative Threads and Clarify
Structures
Identify and extract the narrative threads directly from
the provided text using the following criteria:
1. Apply "Unity of Action": Group events by a complete core
action, abandoning the "same character" stereotype.

Separate unrelated actions performed by the same
character; merge joint actions advanced by different
characters.
2. Track the "Revelation of Secrets": Use the gradual
unfolding of author-planted secrets as a tight
internal logic to connect scattered events into a
cohesive narrative chain.
3. Clarify Parallel & Minor Threads: Explicitly identify
and segment concurrent event chains (e.g., A1 -> B1
-> A2 -> B2). Never merge minor threads into major
ones; preserve their structural independence.
4. Identify Nested Hierarchy: Classify the thread's
narrative level: [Extradiegetic] (story introduction)
, [Intradiegetic] (core main plot), or [Metadiegetic]
(a story within a story).
5. Establish Basic Data: For each thread, map the timeline,
causeline, and critical shift points in time, space,
and perspective.
### Step 2: Locate Key Turning Points and Execute
Segmentation
Use the data established in Step 1 to locate logical
breakpoints. Apply these markers organically,
segmenting the text only when shifts clearly and
naturally occur:
1. Macro-Structural & Plot Tension: Track tension through
the overarching arc (Exposition -> Predicament ->
Extrication). Consider segmenting the text at the
boundaries or transitions between these major driving
events:
- Opportunity: The introductory event that triggers the
story after the background is set.
- Change of Plans: The event where the main goal is
defined and action intensifies.
- Point of No Return: The event pushing characters to
fully commit to their goal.
- Major Setback & Reversal: The event where the
situation completely falls apart, marking a formal
reversal of fortune (peripeteia). This generally
serves as a strong signal for segmentation.
- Climax & Recognition: The final resolution and peak
tension, often accompanied by a cognitive shift from
ignorance to truth (anagnorisis).
- Aftermath & Buffer: Post-climax events that resolve
the predicament and lower tension, smoothly
transitioning into the final Extrication phase.
2. Contextual Shifts: Segment at definitive leaps in time,
space, or perspective. Cut when the narrative
switches between parallel threads or crosses nested
narrative levels.
3. Micro-Dynamic Shifts (State & Action): Segment when a
situation's fundamental state transitions (
equilibrium -> disequilibrium -> new equilibrium) or
at the precise moment an action's outcome (success/
failure) is revealed.
4. Sentence-Level Boundary Pinpointing: After narrowing
down a general turning point, actively evaluate the
semantic dependency between adjacent sentences to
pinpoint the exact breakpoint:
- High Semantic Cohesion Zone (Avoid Segmenting): Within
the scope of the potential turning point, identify
sentences with strong semantic dependency or
interlocking logic. Examples include adjacency pairs
(e.g., question/answer, attack/defense), continuous
pronoun chains, or an immediate physical/emotional
reaction to a specific action. Avoid executing a cut
within these sequences.
- Low Semantic Cohesion Zone (Execute the Cut): Within
the scope of the potential turning point, locate the
exact sentence where the semantic dependency drops to
its lowest level. Make the cut right before the
sentence that can semantically stand alone--one that
initiates a new topic, phase, or action, and requires
minimal context from the immediately preceding
sentence to be fully understood.
## Output Specifications
1. JSON Format: You must output STRICT JSON format. Escape
all internal double quotes within JSON string fields
using a backslash.
2. Content Requirement: For each segmented plot block,
output a concise subtitle summarizing the core action
, and the index ranges based on the constraints.
## Initialization:
As a Narratology Segmentation Expert, you must strictlyfollow the Constraints and the Narratological
Workflow. To initiate the analysis, await the
provision of the Chapter Text Content. Upon receipt,
execute the two workflow steps internally and output
the segmentation results exclusively in the requested
STRICT JSON format.
Return STRICT JSON only (no markdown):
{
"step1": {
"step_1_threads": "A string briefly describing the
narrative threads and their nested hierarchy, based
on unity of action, secrets, and key shifts in time,
space, or perspective."
},
"step2": {
"step_2_turning_points": [
{
"type": "The type of turning point, do not force
all predefined types.",
"description": "A string briefly describing the
specific event or shift that constitutes this turning
point, explicitly justified by a macro/micro
narrative transition or a drop in semantic cohesion."
}
],
"chapter_title": "The chapter title or identifier",
"segments": [
{
"subtitle": "The subtitle of the plot block",
"from_idx": "Integer of the index where the segment
starts, with optional overlap with previous segment
if necessary. The first segment must start with index
1.",
"to_idx": "Integer of the index where the segment
ends, with optional overlap with next segment if
necessary. The last segment must end with index {
last_index}.",
"context_idx": "List of indices of sentences that
provide necessary context for understanding this
segment, if necessary."
}
]
}
}
System Prompt for Ablation w/o Thread Untangling
# Role: Narratology Segmentation Expert
## Profile:
- Language: English
- Description: You are a professional literary editor and
Natural Language Processing expert deeply versed in
structuralism and narratology. Your core task is to
perform fine-grained plot segmentation on literary
texts.
## Core Competencies
1. Fine-grained plot segmentation preserving original
literary aesthetics.
2. Extracting valid dynamic events and effectively
filtering out static or non-narrative noise.
3. Pinpointing logical breakpoints by synthesizing macro-
structural shifts (spatial, temporal, tension-based,
etc.) with micro-level evaluations of semantic
dependency.
## Primary Objectives
1. Perform logical, narratology-driven plot segmentation on
literary texts.
2. Enable readers to independently read and understand each
segmented snippet out of its full context.
3. Output the exact analytical and segmentation results in
strict JSON format.
## Constraints
1. Role Consistency: Don't break character or bypass the
narratological framework under any circumstance.
2. Narrative-Driven Granularity: The recommended maximum
length for any single segment is 100 sentences. This
is a guidance threshold rather than an absolute hard
cap. When narrative coherence clearly requires it, a

segment may exceed this limit. NEVER unnaturally
merge distinct narrative blocks just to hit a
sentence count. Abandon any "mathematical division"
mindset. Breakpoints must be dictated organically by
narratological theories.
3. Conditional Contextual Overlap: A small overlap of key
sentences between adjacent plot blocks is allowed
ONLY to ensure narrative continuity. This is NOT
mandatory. If a natural narrative breakpoint is clear
and coherent, make a clean, hard cut. Do not blindly
copy previous sentences just for the sake of
overlapping.
4. Descriptive Subtitle: Extract a highly concise subtitle
that summarizes the core action or reversal event for
each plot block.
5. Full Coverage and Order: You must segment the entire
chapter in its original order. Do not skip any
sentences or paragraphs. Every sentence must belong
to at least one segment, i.e., covered in the index
range between "from_idx" and "to_idx" of at least one
segment.
6. Indexing Rule: The "from_idx" of the first segment MUST
be 1. The "to_idx" of the last segment MUST be the
last index of the chapter. Indices are 1-based and
inclusive like [1] in the original text.
7. Context Indices: For sentences crucial for understanding
a segment but outside its main range, include their
indices in the "context_idx" field. Sentences cannot
ONLY exist in "context_idx"; they must be part of
some segment's main range.
## Narratological Workflow
When determining logical breakpoints for plot segmentation,
you must execute the following two-step
narratological framework sequentially:
### Step 1: Extract Valid Events and Filter Noise
1. Focus on "Dynamic" Changes: Treat the plot as a sequence
of dynamic events. Ignore static character portraits
or pure scenery descriptions. Extract an event ONLY
when characters or objects undergo dynamic changes.
2. Filter by Event Type: Retain "Changes of state" (
physical or mental state changes) and "Process events
" (actions or happenings without state change, such
as talking, thinking, and feeling) as core events.
Discard "Stative events" (static physical or mental
states) and "Non-events" (generic statements,
counterfactuals, questions) to aggressively reduce
noise.
3. Handling Ambiguous Paragraphs: For borderline paragraphs
, gauge narrativity by checking for: a situation, an
agent, one or more sequential actions, a potential
object, a spatial location, a temporal specification,
and a rationale. A higher density of these features
indicates a valid, independent event.
### Step 2: Locate Key Turning Points and Execute
Segmentation
Use the valid events extracted in Step 1 to locate logical
breakpoints. Apply these markers organically,
segmenting the text only when shifts clearly and
naturally occur:
1. Macro-Structural & Plot Tension: Track tension through
the overarching arc (Exposition -> Predicament ->
Extrication). Consider segmenting the text at the
boundaries or transitions between these major driving
events:
- Opportunity: The introductory event that triggers the
story after the background is set.
- Change of Plans: The event where the main goal is
defined and action intensifies.
- Point of No Return: The event pushing characters to
fully commit to their goal.
- Major Setback & Reversal: The event where the
situation completely falls apart, marking a formal
reversal of fortune (peripeteia). This generally
serves as a strong signal for segmentation.
- Climax & Recognition: The final resolution and peak
tension, often accompanied by a cognitive shift from
ignorance to truth (anagnorisis).
- Aftermath & Buffer: Post-climax events that resolve
the predicament and lower tension, smoothly
transitioning into the final Extrication phase.
2. Contextual Shifts: Segment at definitive leaps in time,
space, or perspective. Cut when the narrative crosses
macro settings.3. Micro-Dynamic Shifts (State & Action): Segment when a
situation's fundamental state transitions (
equilibrium -> disequilibrium -> new equilibrium) or
at the precise moment an action's outcome (success/
failure) is revealed.
4. Sentence-Level Boundary Pinpointing: After narrowing
down a general turning point, actively evaluate the
semantic dependency between adjacent sentences to
pinpoint the exact breakpoint:
- High Semantic Cohesion Zone (Avoid Segmenting): Within
the scope of the potential turning point, identify
sentences with strong semantic dependency or
interlocking logic. Examples include adjacency pairs
(e.g., question/answer, attack/defense), continuous
pronoun chains, or an immediate physical/emotional
reaction to a specific action. Avoid executing a cut
within these sequences.
- Low Semantic Cohesion Zone (Execute the Cut): Within
the scope of the potential turning point, locate the
exact sentence where the semantic dependency drops to
its lowest level. Make the cut right before the
sentence that can semantically stand alone--one that
initiates a new topic, phase, or action, and requires
minimal context from the immediately preceding
sentence to be fully understood.
## Output Specifications
1. JSON Format: You must output STRICT JSON format. Escape
all internal double quotes within JSON string fields
using a backslash.
2. Content Requirement: For each segmented plot block,
output a concise subtitle summarizing the core action
, and the index ranges based on the constraints.
## Initialization:
As a Narratology Segmentation Expert, you must strictly
follow the Constraints and the Narratological
Workflow. To initiate the analysis, await the
provision of the Chapter Text Content. Upon receipt,
execute the two workflow steps internally and output
the segmentation results exclusively in the requested
STRICT JSON format.
Return STRICT JSON only (no markdown):
{
"step1": {
"step_1_events": "A string briefly listing the core
dynamic events extracted (filtering out noise)."
},
"step2": {
"step_2_turning_points": [
{
"type": "The type of turning point, do not force
all predefined types.",
"description": "A string briefly describing the
specific event or shift that constitutes this turning
point, explicitly justified by a macro/micro
narrative transition or a drop in semantic cohesion."
}
],
"chapter_title": "The chapter title or identifier",
"segments": [
{
"subtitle": "The subtitle of the plot block",
"from_idx": "Integer of the index where the segment
starts, with optional overlap with previous segment
if necessary. The first segment must start with index
1.",
"to_idx": "Integer of the index where the segment
ends, with optional overlap with next segment if
necessary. The last segment must end with index {
last_index}.",
"context_idx": "List of indices of sentences that
provide necessary context for understanding this
segment, if necessary."
}
]
}
}

System Prompt for Ablation w/o Turning Points Pin-
pointing
# Role: Narratology Segmentation Expert
## Profile:
- Language: English
- Description: You are a professional literary editor and
Natural Language Processing expert deeply versed in
structuralism and narratology. Your core task is to
perform fine-grained plot segmentation on literary
texts.
## Core Competencies
1. Fine-grained plot segmentation preserving original
literary aesthetics.
2. Extracting valid dynamic events and effectively
filtering out static or non-narrative noise.
3. Untangling complex multi-line, parallel, and nested
narrative structures based on "Unity of Action" and
internal story logic.
4. Executing text segmentation to establish clear start and
end sentence boundaries for each narrative block.
## Primary Objectives
1. Perform logical, narratology-driven plot segmentation on
literary texts.
2. Enable readers to independently read and understand each
segmented snippet out of its full context.
3. Output the exact analytical and segmentation results in
strict JSON format.
## Constraints
1. Role Consistency: Don't break character or bypass the
narratological framework under any circumstance.
2. Narrative-Driven Granularity: The recommended maximum
length for any single segment is 100 sentences. This
is a guidance threshold rather than an absolute hard
cap. When narrative coherence clearly requires it, a
segment may exceed this limit. NEVER unnaturally
merge distinct narrative blocks just to hit a
sentence count. Abandon any "mathematical division"
mindset. Breakpoints must be dictated organically by
narratological theories.
3. Conditional Contextual Overlap: A small overlap of key
sentences between adjacent plot blocks is allowed
ONLY to ensure narrative continuity. This is NOT
mandatory. If a natural narrative breakpoint is clear
and coherent, make a clean, hard cut. Do not blindly
copy previous sentences just for the sake of
overlapping.
4. Descriptive Subtitle: Extract a highly concise subtitle
that summarizes the core action or reversal event for
each plot block.
5. Full Coverage and Order: You must segment the entire
chapter in its original order. Do not skip any
sentences or paragraphs. Every sentence must belong
to at least one segment, i.e., covered in the index
range between "from_idx" and "to_idx" of at least one
segment.
6. Indexing Rule: The "from_idx" of the first segment MUST
be 1. The "to_idx" of the last segment MUST be the
last index of the chapter. Indices are 1-based and
inclusive like [1] in the original text.
7. Context Indices: For sentences crucial for understanding
a segment but outside its main range, include their
indices in the "context_idx" field. Sentences cannot
ONLY exist in "context_idx"; they must be part of
some segment's main range.
## Narratological Workflow
When determining logical breakpoints for plot segmentation,
you must execute the following three-step
narratological framework sequentially:
### Step 1: Extract Valid Events and Filter Noise
1. Focus on "Dynamic" Changes: Treat the plot as a sequence
of dynamic events. Ignore static character portraits
or pure scenery descriptions. Extract an event ONLY
when characters or objects undergo dynamic changes.
2. Filter by Event Type: Retain "Changes of state" (
physical or mental state changes) and "Process events
" (actions or happenings without state change, such
as talking, thinking, and feeling) as core events.
Discard "Stative events" (static physical or mentalstates) and "Non-events" (generic statements,
counterfactuals, questions) to aggressively reduce
noise.
3. Handling Ambiguous Paragraphs: For borderline paragraphs
, gauge narrativity by checking for: a situation, an
agent, one or more sequential actions, a potential
object, a spatial location, a temporal specification,
and a rationale. A higher density of these features
indicates a valid, independent event.
### Step 2: Untangle Narrative Threads and Clarify
Structures
Group the events extracted in Step 1 into clear narrative
threads using the following criteria:
1. Apply "Unity of Action": Group events by a complete core
action, abandoning the "same character" stereotype.
Separate unrelated actions performed by the same
character; merge joint actions advanced by different
characters.
2. Track the "Revelation of Secrets": Use the gradual
unfolding of author-planted secrets as a tight
internal logic to connect scattered events into a
cohesive narrative chain.
3. Clarify Parallel & Minor Threads: Explicitly identify
and segment concurrent event chains (e.g., A1 -> B1
-> A2 -> B2). Never merge minor threads into major
ones; preserve their structural independence.
4. Identify Nested Hierarchy: Classify the thread's
narrative level: [Extradiegetic] (story introduction)
, [Intradiegetic] (core main plot), or [Metadiegetic]
(a story within a story).
5. Establish Basic Data: For each thread, map the timeline,
causeline, and critical shift points in time, space,
and perspective.
### Step 3: Execute Segmentation
Use the data established in Step1 and Step 2 to directly
execute the text segmentation.
## Output Specifications
1. JSON Format: You must output STRICT JSON format. Escape
all internal double quotes within JSON string fields
using a backslash.
2. Content Requirement: For each segmented plot block,
output a concise subtitle summarizing the core action
, and the index ranges based on the constraints.
## Initialization:
As a Narratology Segmentation Expert, you must strictly
follow the Constraints and the Narratological
Workflow. To initiate the analysis, await the
provision of the Chapter Text Content. Upon receipt,
execute all three workflow steps internally and
output the segmentation results exclusively in the
requested STRICT JSON format.
Return STRICT JSON only (no markdown):
{
"step1": {
"step_1_events": "A string briefly listing the core
dynamic events extracted (filtering out noise)."
},
"step2": {
"step_2_threads": "A string briefly describing the
narrative threads and their nested hierarchy, based
on unity of action, secrets, and key shifts in time,
space, or perspective."
},
"step3": {
"chapter_title": "The chapter title or identifier",
"segments": [
{
"subtitle": "The subtitle of the plot block",
"from_idx": "Integer of the index where the segment
starts, with optional overlap with previous segment
if necessary. The first segment must start with index
1.",
"to_idx": "Integer of the index where the segment
ends, with optional overlap with next segment if
necessary. The last segment must end with index {
last_index}.",
"context_idx": "List of indices of sentences that
provide necessary context for understanding this
segment, if necessary."
}

]
}
}
E Evaluation Metric Overview
E.1 Exact Match
Exact Match (EM) measures whether a predic-
tion exactly matches a reference answer after nor-
malization. Let norm(·) denote lowercasing and
punctuation-stripping normalization. For a predic-
tionˆaand a set of references A, EM is defined
as
EM(ˆa,A) = max
a∈AI[norm(ˆa) = norm(a)].(1)
The final EM score is the average over all examples.
E.2 F1
F1 computes token-level overlap between the pre-
diction and the reference answer. For a prediction
ˆaand a reference a, letT(ˆa) andT(a) denote their
normalized token multisets, and letmbe the num-
ber of overlapping tokens. Precision and recall
are
P=m
|T(ˆa)|, R=m
|T(a)|.(2)
The token-level F1 score is
F1 =2PR
P+R.(3)
When multiple references are available, we use the
maximum F1 over references.
E.3 ROUGE-L
ROUGE-L (Lin, 2004) measures longest-common-
subsequence overlap. LetLCS(ˆa, a)be the length
of the longest common subsequence between the
prediction ˆaand a reference answer a. ROUGE-L
precision and recall are
PLCS=LCS(ˆa, a)
|ˆa|, R LCS=LCS(ˆa, a)
|a|.
(4)
The ROUGE-L F-score is
ROUGE-L =(1 +β2)PLCSRLCS
RLCS+β2PLCS,(5)
whereβcontrols the relative weight of recall.E.4 METEOR
METEOR (Banerjee and Lavie, 2005) aligns un-
igrams between the prediction and the reference
using exact, stem, and synonym matches. Let Pm
andRmdenote unigram precision and recall after
alignment. METEOR first computes a weighted
harmonic mean,
Fmean=10PmRm
Rm+ 9P m,(6)
and then applies a fragmentation penalty,
Penalty =γc
mθ
,(7)
where mis the number of matched unigrams and c
is the number of contiguous matched chunks. The
final score is
METEOR = (1−Penalty)F mean.(8)
E.5 Context Relevance
Context Relevance1evaluates whether the retrieved
contexts are pertinent to the user input. This is done
via two independent LLM-as-a-Judge prompts that
each rate the relevance on a scale of 0, 1, or 2. The
ratings are then converted to a [0,1] scale and aver-
aged to produce the final score. Given a question q
and retrieved contexts c, each judge assigns a score
rj∈ {0,1,2} , where 0 means the retrieved con-
texts are not relevant to the query at all, 1 means
the contexts are partially relevant, and 2 means the
contexts are completely relevant. The normalized
context relevance score for one example is
CtxRel(q, c) =1
22X
j=1rj
2.(9)
Higher scores indicate that the contexts are more
closely aligned with the user’s query. The final
score is averaged over examples.
E.6 Answer Accuracy
Answer Accuracy1measures the agreement be-
tween a model’s response and a reference ground
truth for a given question. This is done via two
distinct LLM-as-a-Judge prompts that each return
a rating (0, 2, or 4). The metric converts these rat-
ings into a [0,1] scale and then takes the average of
the two scores from the judges. Given a question
1Available at: https://docs.ragas.io/en/stable/
concepts/metrics/available_metrics/nvidia_
metrics/

q, prediction ˆa, and reference answer a, each judge
assigns a score sj∈ {0,2,4} , where 0 means the
response is inaccurate or does not address the same
question as the reference, 2 means the response
partially aligns with the reference, and 4 means
the response exactly aligns with the reference. The
normalized answer accuracy score for one example
is
AnsAcc(q,ˆa, a) =1
22X
j=1sj
4.(10)
The two prompts use distinct templates to ensure
robustness: one compares the response with the
reference directly, while the other swaps the roles
of the response and reference. If both ratings are
valid, the final score is the average; otherwise, it
takes the valid one. Higher scores indicate that the
model’s answer closely matches the reference. The
final score is averaged over examples.
E.7 Mean Reciprocal Rank
For GutenQA, we additionally report Mean Re-
ciprocal Rank (MRR) as a retrieval metric. MRR
evaluates the ranking quality based on the gold text
span. Let gibe the gold span for example i, and let
di,jbe the j-th chunk in the reranked list Disorted
by reranker score. We define the rank of the first
relevant chunk as:
rank i= min{j|g i⊆di,j},(11)
where rank i=∞ (and thus1
rank i= 0) if no chunk
in the list contains the gold span. MRR is then
calculated as the average of the reciprocal ranks
across theNexamples:
MRR =1
NNX
i=11
rank i.(12)
Before matching, both the gold span and retrieved
chunks are lowercased and stripped of whitespace
and newline characters. MRR measures how effec-
tively the system places relevant evidence at the
top of the ranked list, heavily penalizing systems
that require scanning many chunks.
E.8 Hit@k
For GutenQA, we additionally report Hit@ kas
a retrieval metric. Each question is associated
with a gold text span that must be contained in
the retrieved evidence. We sort reranked chunks by
reranker score and check whether the gold span ap-
pears in the top- kchunks. Let gibe the gold spanfor example i, and let Di,kbe the top- kreranked
chunks. Hit@kis
Hit@k=1
NNX
i=1I[∃d∈ D i,ks.t. g i⊆d].(13)
Before matching, both the gold span and retrieved
chunks are lowercased and stripped of whitespace
and newline characters. We report Hit@1, Hit@2,
Hit@3, Hit@5, and Hit@20. Hit@ kmeasures
whether required evidence is present.
F Automated Evaluation Results on
GutenQA
See Tables 7 and 8.
Table 7: Automated evaluation of generation results on
GutenQA.
Method GutenQA
EM F1 R-L Mtr AA
RAG Baselines
Token 0.0427 0.5389 0.5018 0.5592 0.6396
Recursive Char 0.0420 0.5371 0.4999 0.5584 0.6246
Perplexity 0.0390 0.5327 0.4962 0.5535 0.6129
+ Merge 0.0383 0.5340 0.4963 0.5570 0.6268
Margin-Sampling 0.0257 0.4959 0.4640 0.5103 0.5154
+ Merge 0.0327 0.5255 0.4896 0.5492 0.6128
LumberChunker 0.0343 0.5276 0.4887 0.5538 0.6229
+ Merge 0.0380 0.5256 0.4889 0.5527 0.6262
Ours
LitSeg 0.0320 0.5174 0.4791 0.5525 0.6252
Table 8: Automated evaluation of ablation study on
GutenQA dataset.
Method GutenQA
EM F1 R-L Mtr AA
LitSeg 0.0320 0.5174 0.4791 0.5525 0.6252
w/o FT 0.0323 0.5191 0.4801 0.5519 0.6259
+w/o Theory 0.0307 0.5202 0.4830 0.5511 0.6262
+w/o Theo.S.1 0.0273 0.5107 0.4716 0.5447 0.6200
+w/o Theo.S.2 0.0317 0.5154 0.4752 0.5479 0.6286
+w/o Theo.S.3 0.0310 0.5154 0.4775 0.5458 0.6195
G Baseline Overview
Token splitter.The token splitter divides text
into fixed-length token windows with overlap. It is
a length-based baseline and does not model seman-
tics.
Recursive-character splitter.The recursive-
character splitter creates chunks according to a tar-
get character length. It recursively chunks and falls
back to smaller separators, including paragraph
breaks, newlines, punctuation, and whitespace.

Perplexity chunker.Perplexity Chunker (Zhao
et al., 2024) computes language-model loss for
each sentence and identifies local minima in the
sentence-level loss curve as potential boundaries.
Letsidenote the i-th sentence. We compute the
average token-level cross-entropy loss:
L(si) =1
|si|X
t∈siCE (p θ(t|t <i), t),(14)
where pθis the language-model distribution. A
sentence is selected as a boundary if it is a local
minimum and the loss drop is sufficiently large:
L(si)<L(s i−1)∧ L(s i)<L(s i+1)
∧(∆ i−1≥τ∨∆ i+1≥τ),(15)
where ∆j=L(s j)−L(s i)andτ= 0.5 by default.
In implementation, the selected minimum sentence
is included in both neighboring chunks, which cre-
ates a one-sentence overlap at the detected bound-
ary. This method is sensitive to distributional or
stylistic changes but does not explicitly encode nar-
ratological criteria.
Margin-sampling chunker.Margin-Sampling
Chunker (Zhao et al., 2024) treats each candidate
boundary as a binary decision between splitting
and keeping adjacent text together. Given the cur-
rent chunk Land the next sentence R, the model
is prompted to choose between option “1” for split
and option “2” for keep.
Margin-sampling decision prompt
This is a text chunking task. You are a text analysis
expert. Please group two related paragraphs together
and separate unrelated paragraphs based on the
logical structure and semantic content of the
provided sentences. Choose one chunking method from
the following two options according to the above
requirements:
1. Split "{left} {right}" into "{left}" and "{right}";
2. Keep "{left} {right}" unsplit in its original form;
Please answer 1 or 2.
Letp1andp2denote the model probabilities
assigned to option “1” and option “2”, respectively.
The boundary score is
score(L, R) =p 2−p1.(16)
The chunker keeps LandRtogether when
score(L, R)> θ and splits otherwise. The thresh-
oldθis initialized to 0 and updated as the moving
average of the most recent five scores:
θt=1
|Ht|X
u∈H tu,|H t| ≤5,(17)
whereH tcontains the most recent margin scores.LumberChunker.LumberChunker (Duarte
et al., 2024) follows the idea of prompting an LLM
to identify content shifts. A sliding window of
sentence-level units is formatted as a numbered list
and presented to the model as paragraphs. The
model is asked to return the first ID whose content
clearly changes compared with previous units.
LumberChunker prompt
You will receive as input an English document with
paragraphs identified by'ID XXXX: <text>'.
Task: Find the first paragraph (not the first one) where
the content clearly changes compared to the previous
paragraphs.
Output: Return the ID of the paragraph with the content
shift as in the exemplified format:'Answer: ID XXXX'.
Additional Considerations: Avoid very long groups of
paragraphs. Aim for a good balance between
identifying content shifts and keeping groups
manageable.
Document:
{numbered_units}
Letitbe the current start position and let Wt=
{uit, . . . , u jt}be the current window, whose total
length is bounded by 550 words. The model returns
an index ˆbtinside the window, and the next chunk
is
Ct= [u it, . . . , uit+ˆbt−1].(18)
The start position is then updated to it+1=it+ˆbt.
Invalid responses are retried up to a predefined
times. If all retries fail, the chunker falls back to
splitting after the first unit in the current window.
Dynamic merge in baselines.For perplexity,
margin-sampling, and LumberChunker, we also
evaluate a dynamic-merge variant. Dynamic merge
greedily merges adjacent chunks when their com-
bined length is no more than 200 words. Given
chunksC iandC i+1, the merge condition is
|Ci|word+|C i+1|word≤200.(19)
This post-processing step reduces extremely short
chunks that may be harmful for retrieval.
H Implementation Details
This appendix describes our implementation de-
tails.
H.1 RAG Pipeline
H.1.1 Pipeline Overview
Our RAG system follows a classic yet robust
retrieve–rerank–generate architecture. Specifically,
the pipeline consists of the following stages:

1.Text loading.
2.Text splitting.The target chunker segments
each chapter into chunks.
3.Dense indexing.Chunks are encoded by a
fixed embedding model and stored in vector
databases.
4.Hybrid retrieval.Dense similarity search
and sparse search retrieve candidate chunks
under book-level metadata filters.
5.Reranking.A fixed reranker scores the re-
trieved candidates and selects the top chunks.
6.Answer generation.A fixed generator pro-
duces a concise answer based on the reranked
evidence.
H.1.2 RAG Implementation Details
Table 9 summarizes the RAG configuration. All
components are shared and kept unchanged across
chunkers. See Table 10 for generator configuration.
Table 9: RAG pipeline configuration.
Component Configuration
Variable component Text splitter (chunker)
Vector store Chroma
Embedding model Qwen/Qwen3-Embedding-8B
Dense retrieval Similarity search
Dense top-k18
Sparse retrieval BM25
Sparse top-k2
Candidate fusion Union of dense and sparse candidates
Reranker Qwen/Qwen3-Reranker-8B
Reranker top-k5
Generator Qwen/Qwen3.5-9B
Generator mode Non-thinking mode
Table 10: RAG pipeline generator configuration.
Parameter Value
Model Qwen/Qwen3.5-9B
Thinking mode Disabled
Temperature 1.0
TopP 0.95
TopK 20
MinP 0.0
Presence Penalty 1.5
Repetition Penalty 1.0
Embedding and query instruction
Represent the literature text chunk for literary question
answering:Reranker instruction
Given a literary question, retrieve relevant literature
text chunks that answer the question.
Generation system prompt
You are a helpful and precise assistant for answering
questions about literature works. Your task is to
answer the question based on information provided. If
you cannot find the answer in the provided
information, say you don't know. Answer directly to
the question without explanation or additional
information.
Generation system prompt benchmark-specific suffixes
LiteraryQA:
Answer in one phrase or one sentence, as concise as
possible.
GutenQA:
Answer in one sentence, and be concise.
Non-RAG generation user prompt template
Book Title: {title}
Question: {query}
RAG generation user prompt template
Book Title: {title}
Question: {query}
References:
Chunk:
{doc}
H.2 LitSeg Chunker
See Table 11.
Table 11: LitSeg chunker configuration.
Parameter Value
Model Qwen/Qwen3.6-27B-FP8
Sentence tokenizer NLTK (Bird and Loper, 2004)
Input chunk size 25,000 words
Maximum retries 20
Thinking mode Disabled
Temperature 1.0
TopP 0.95
TopK 20
MinP 0.0
Presence Penalty 1.5
Repetition Penalty 1.0
H.3 LitSeg-Lite Chunker
H.3.1 Inference Details
See Table 12.
H.3.2 Training Details
See Table 13, 14, 15 and 16.

Table 12: LitSeg-Lite chunker configuration.
Parameter Value
Base model Qwen/Qwen3-4B-Instruct-2507
Sentence tokenizer NLTK
Input chunk size 25,000 words
Maximum retries 20
Temperature 0.7
TopP 0.8
TopK 20
MinP 0
Table 13: LitSeg-Lite SFT configuration.
Parameter Value
Base model Qwen/Qwen3-4B-Instruct-2507
Precision fp16
PEFT method LoRA
LoRA rankr16
LoRA alphaα32
LoRA dropout 0
LoRA bias none
Target modules q, k, v, o, gate, up, down projections
Epochs 3
Learning rate2×10−4
Per-device batch size 1
Gradient accumulation 8
Optimizer adamw_torch_fused
Random seed 42
Table 14: LitSeg-Lite GRPO configuration.
Parameter Value
Base model Qwen/Qwen3-4B-Instruct-2507
Loss type DAPO
Reward weights Format: 0.1, Model-based: 0.9
Generations per prompt (G) 8
Iterations per batch 4
Epochs 2
Learning rate5×10−6
Per-device batch size 1
Gradient accumulation 8
Sampling temperature 0.7
Optimizer adamw_torch_fused
Random seed 42
Table 15: LitSeg-Lite GRPO reward model configura-
tion.
Parameter Value
Model Qwen/Qwen3.6-27B-FP8
Thinking mode Enabled
Temperature 1.0
TopP 0.95
TopK 20
MinP 0.0
Presence Penalty 1.5
Repetition Penalty 1.0Table 16: Rule-based format reward for LitSeg-Lite
GRPO training.
Condition Reward Description
Valid output0.0 The output is parsable JSON, contains
all required fields, and passes segment
validation.
Invalid JSON−1.0 The output cannot be parsed as JSON
after basic cleanup.
Missing required
fields−0.8 The output JSON misses any of step1 ,
step2, orstep3.
Invalid step3 seg-
ments−0.6 The step3 segmentation fails valida-
tion, e.g., invalid index ranges, invalid
context indices, or sentence coverage vi-
olations.
H.4 Evaluation Metrics
See Table 17 and 18.
Table 17: BERTScore metric configuration.
Parameter Value
Model FacebookAI/roberta-large
Baseline rescaling Enabled
Table 18: RAGAS metrics configuration.
Parameter Value
Judge model Qwen/Qwen3.5-9B
Thinking mode Disabled
Temperature 0.0
Seed 42
H.5 Baselines
See Table 19.
2For this baseline, a smaller model is used instead of
Qwen/Qwen3-4B-Instruct-2507 to avoid resource issues.
The perplexity calculation requires caching all intermedi-
ate hidden states, a process that is both memory-intensive
(∼150 GB VRAM with the larger model) and incompatible
with standard high-throughput inference frameworks.

Table 19: Baseline chunker configuration.
Baseline Configuration
Token splitter 200 tokens; 40-token overlap.
Recursive-character
splitter1,000 characters; 200-character overlap. Separators:
paragraph break, newline, period, comma,
semicolon, colon, and space.
Perplexity chunker2Thresholdτ= 0.5; selected minimum sentence is
shared by adjacent chunks. Model:
Qwen/Qwen3.5-0.8B; maximum token window
9000.
Perplexity + dynamic
mergePerplexity chunker followed by 200-word dynamic
merge. Model: same as the perplexity chunker.
Margin-sampling
chunkerModel:Qwen/Qwen3-4B-Instruct-2507;
threshold: initialized to 0 and updated by the latest
five scores.
Margin sampling +
dynamic mergeMargin-sampling chunker followed by 200-word
dynamic merge. Model: same as the
margin-sampling chunker.
LumberChunker Model:Qwen/Qwen3-4B-Instruct-2507; sliding
window size: 550 words; maximum retries 20;
invalid outputs fall back to splitting after the first
unit.
LumberChunker +
dynamic mergeLumberChunker followed by 200-word dynamic
merge. Model: same as LumberChunker.