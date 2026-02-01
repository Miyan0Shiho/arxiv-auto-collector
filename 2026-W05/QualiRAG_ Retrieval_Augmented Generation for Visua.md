# QualiRAG: Retrieval-Augmented Generation for Visual Quality Understanding

**Authors**: Linhan Cao, Wei Sun, Weixia Zhang, Xiangyang Zhu, Kaiwei Zhang, Jun Jia, Dandan Zhu, Guangtao Zhai, Xiongkuo Min

**Published**: 2026-01-26 06:27:03

**PDF URL**: [https://arxiv.org/pdf/2601.18195v1](https://arxiv.org/pdf/2601.18195v1)

## Abstract
Visual quality assessment (VQA) is increasingly shifting from scalar score prediction toward interpretable quality understanding -- a paradigm that demands \textit{fine-grained spatiotemporal perception} and \textit{auxiliary contextual information}. Current approaches rely on supervised fine-tuning or reinforcement learning on curated instruction datasets, which involve labor-intensive annotation and are prone to dataset-specific biases. To address these challenges, we propose \textbf{QualiRAG}, a \textit{training-free} \textbf{R}etrieval-\textbf{A}ugmented \textbf{G}eneration \textbf{(RAG)} framework that systematically leverages the latent perceptual knowledge of large multimodal models (LMMs) for visual quality perception. Unlike conventional RAG that retrieves from static corpora, QualiRAG dynamically generates auxiliary knowledge by decomposing questions into structured requests and constructing four complementary knowledge sources: \textit{visual metadata}, \textit{subject localization}, \textit{global quality summaries}, and \textit{local quality descriptions}, followed by relevance-aware retrieval for evidence-grounded reasoning. Extensive experiments show that QualiRAG achieves substantial improvements over open-source general-purpose LMMs and VQA-finetuned LMMs on visual quality understanding tasks, and delivers competitive performance on visual quality comparison tasks, demonstrating robust quality assessment capabilities without any task-specific training. The code will be publicly available at https://github.com/clh124/QualiRAG.

## Full Text


<!-- PDF content starts -->

QualiRAG: Retrieval-Augmented Generation for Visual Quality Understanding
Linhan Cao1∗, Wei Sun2∗♡, Weixia Zhang1, Xiangyang Zhu3, Kaiwei Zhang1, Jun Jia1,
Dandan Zhu2, Guangtao Zhai1, Xiongkuo Min1†
1Shanghai Jiao Tong University
2East China Normal University
3Shanghai Artificial Intelligence Laboratory
Abstract
Visual quality assessment (VQA) is increasingly
shifting from scalar score prediction toward inter-
pretable quality understanding—a paradigm that
demandsfine-grained spatiotemporal perception
andauxiliary contextual information. Current ap-
proaches rely on supervised fine-tuning or rein-
forcement learning on curated instruction datasets,
which involve labor-intensive annotation and are
prone to dataset-specific biases. To address these
challenges, we proposeQualiRAG, atraining-free
Retrieval-AugmentedGeneration(RAG)frame-
work that systematically leverages the latent per-
ceptual knowledge of large multimodal models
(LMMs) for visual quality perception. Unlike con-
ventional RAG that retrieves from static corpora,
QualiRAG dynamically generates auxiliary knowl-
edge by decomposing questions into structured
requests and constructing four complementary
knowledge sources:visual metadata,subject local-
ization,global quality summaries, andlocal qual-
ity descriptions, followed by relevance-aware re-
trieval for evidence-grounded reasoning. Extensive
experiments show that QualiRAG achieves sub-
stantial improvements over open-source general-
purpose LMMs and VQA-finetuned LMMs on vi-
sual quality understanding tasks, and delivers com-
petitive performance on visual quality comparison
tasks, demonstrating robust quality assessment ca-
pabilities without any task-specific training. The
code will be publicly available at https://github.
com/clh124/QualiRAG.
1 Introduction
Visual quality assessment (VQA) [Minet al., 2024 ]is a fun-
damental component of modern multimedia systems, serv-
ing as the primary perceptual criterion for evaluating visual
content across the processing pipeline and guiding down-
stream decisions. Traditionally, VQA has focused onpredict-
ing scalar quality scores [Sunet al., 2022; Wuet al., 2023a;
∗Equal contribution.♡Project lead.†Corresponding authors.
Human AnnotatedThinking Trace
Any LMMSFT based LMMsTraining
Infer
TestedVideo&Question
RL based LMMsHigh annotation cost High risk of overfitting
Human AnnotatedFinalAnswerAny LMMTraining
Infer
Reward on Answer OnlyNo Constraint on Think
OurRAG based LMMs
Training free:Flexible:   QueryTools
Retrieval DatabaseAny LMM
Infer
Training free:Flexible:   Training free:Flexible:   
Figure 1: Comparison of training-based and training-free paradigms
for visual quality understanding. Unlike SFT- and RL-based LMMs
that require task-specific training with human annotations or reward
signals, our RAG-based LMM performs inference-time retrieval and
reasoning without any additional training, providing improved flex-
ibility and scalability.
Wuet al., 2023c; Sunet al., 2023; Sunet al., 2025 ]. While
such paradigms provide a compact summary of perceptual
quality, they are inherently limited in interpretability: they
neither explainwhyan input is judged to be of high or
low quality, nor identify the underlying distortion attributes.
As multimedia applications increasingly demand trustworthy
and explainable quality evaluation, VQA is undergoing a shift
from scalar score prediction toward interpretable quality un-
derstanding.
Recently, large multimodal models (LMMs) have shown
remarkable advances in visual perception and multimodal
reasoning [Liet al., 2024a; Baiet al., 2025; Zhuet al., 2025b;
Wanget al., 2025b; Yanget al., 2025 ], enabling them to
capture and articulate complex perceptual attributes that sup-
port visual quality understanding. To elicit such capabili-
ties, previous studies have explored supervised fine-tuning
(SFT) and reinforcement learning (RL) strategies to pro-
duce explainable quality assessment, as illustrated in Fig-
ure 1. Specifically, SFT-based approaches [Wuet al., 2024b;arXiv:2601.18195v1  [cs.CV]  26 Jan 2026

Jiaet al., 2025 ]depend heavily on the curation of large-scale
quality instruction datasets, such as Q-Pathway [Wuet al.,
2024b ]and OmniVQAChat-400K [Jiaet al., 2025 ]. These
datasets typically require a labor-intensive pipeline involving
proprietary model generation and human refinement, making
them expensive to scale and prone to annotation and content
biases. Alternatively, RL-based approaches [Liet al., 2025;
Caoet al., 2025 ]attempt to bypass explicit instruction data
by using quality scores as implicit supervision. However,
this often leads to misalignment between the generated ratio-
nale and the final prediction, resulting in inconsistent expla-
nations. Crucially, such training-based paradigms risk com-
promising general-purpose reasoning and often struggle to
generalize across unseen distortions and domains.
In contrast to training-based paradigms, visual qual-
ity understanding can be achieved by carefully organizing
inference-time visual observations and auxiliary contextual
cues. Existing LMMs [Wanget al., 2025b; Yanget al.,
2025 ]already encode rich knowledge related to visual dis-
tortions, temporal consistency, subject-dependent quality at-
tributes, etc. The key challenge thus lies in how to effectively
elicit and ground this knowledge with respect to the visual
content under assessment. This insight naturally leads to a
retrieval-augmented [Lewiset al., 2020 ]formulation that co-
ordinates visual inputs with auxiliary contextual retrieval, en-
abling fine-grained and interpretable quality reasoning with-
out any task-specific training.
Therefore, we proposeQualiRAG, atraining-free
agentic frameworkthat leveragesRetrieval-Augmented
Generation(RAG)for visual quality understanding. Given a
visual input and a quality-related question, QualiRAG adopts
a retrieval-augmented generation pipeline composed of four
collaborative modules: (1)Query Organizerthat reformu-
lates the input question into structured, retrieval-oriented re-
quests specifying subject, quality dimension, scope, and vi-
sual focus; (2)Source Augmenterthat constructs and popu-
lates multiple auxiliary quality knowledge databases, includ-
ingvisual metadata,subject localization,global quality sum-
maries, andlocal quality descriptions, to produce candidate
evidence; (3)Source Selectorthat retrieves knowledge se-
mantically aligned with the original question via relevance-
aware retrieval; and (4)Answer Generatorthat integrates the
visual input with the retrieved auxiliary knowledge to gen-
erate grounded reasoning and answers. These modules de-
compose visual quality understanding into sequential stages
of visual perception, knowledge augmentation, and evidence-
grounded reasoning, enabling interpretable and fine-grained
quality analysis. Extensive experiments show that QualiRAG
achieves substantial improvements over open-source general-
purpose LMMs and VQA-finetuned LMMs on quality under-
standing tasks, and delivers competitive performance on qual-
ity comparison tasks, demonstrating its effectiveness.
Our main contributions are summarized as follows:
• We presentQualiRAG, a training-free retrieval-
augmented generation framework for visual quality un-
derstanding that systematically activates the latent per-
ceptual knowledge of LMMs, without requiring any
task-specific fine-tuning.• We propose agranularity-aware quality reasoning
paradigmthat organizes visual evidence at both global
and local levels, enabling interpretable inference over
holistic quality perception and fine-grained, subject-
dependent distortions during inference.
• We instantiate this reasoning paradigm throughfour
complementary auxiliary quality knowledge sources:
visual metadata, subject localization, global quality
summaries, and local quality descriptions, and inte-
grate them through relevance-aware retrieval to support
evidence-grounded reasoning.
2 Related Work
LMMs for Visual Quality Assessment.
Recent advances in LMMs have sparked growing inter-
est in their application to visual quality assessment. Exist-
ingtraining-basedmethods can be broadly categorized into
three paradigms. The first paradigm [Wanget al., 2025a;
Liuet al., 2025 ]leverages LMMs as powerful feature ex-
tractors, where hidden states from LMM backbones are fed
into lightweight prediction heads to regress a scalar quality
score. Although benefiting from the powerful feature repre-
sentations of LMMs, these methods still adhere to a conven-
tional score-regression paradigm and fail to fully exploit the
generative and reasoning capabilities of LMMs. The second
paradigm [Jiaet al., 2024; Jiaet al., 2025 ]formulates VQA
as a vision-language instruction task. By constructing vi-
sual instruction–answer pairs, these methods fine-tune LMMs
via supervised learning to generate quality-related descrip-
tions. While this paradigm enhances interpretability and bet-
ter aligns model outputs with human perceptual judgments,
the collection of large-scale, high-quality instruction datasets
is expensive and difficult to scale. The third paradigm [Liet
al., 2025; Caoet al., 2025 ]introduces reinforcement learn-
ing into VQA, typically optimized via Group Relative Prefer-
ence Optimization (GRPO) [Shaoet al., 2024 ]. In this setting,
LMMs are trained to generate step-by-step reasoning before
predicting a quality score. The reward signal is computed
based on the final predicted score, while the intermediate rea-
soning process remains unsupervised. This may lead to rea-
soning traces that are inconsistent with or even contradictory
to the final quality prediction.
Beyond these training-based paradigms, several studies
have exploredtraining-freeapproaches for VQA. In-context
learning [Chenet al., 2023; Wuet al., 2024d ]exploits care-
fully designed prompts or a small number of exemplars to
elicit the quality perception capabilities of LMMs. Other
works [Zhuet al., 2025a; Xinget al., 2025 ]adopt agent-based
frameworks for visual quality perception, where fine-tuned
expert models or proprietary LMMs are employed as special-
ized perception modules to enhance quality assessment abil-
ity. Our QualiRAG follows the agent-based paradigm but dif-
fers fundamentally in that it leverages RAG to systematically
exploit the capabilities of LMMs, without relying on addi-
tional quality-labeled data for training or task-specific VQA
models as expert modules.
Retrieval-Augmented Generation for LMMs.RAG has
emerged as an effective paradigm for enhancing LMMs by

Question:How is the clarity of the first child that appears in the video?A…B...C…D…
…
Query OrganizerQuestion DecoupleStructured Request{"subject": subject referenced in the question,  ”dimension":relevant quality dimension,”scope“: spatial or temporal,”focus“: key spatial/temporalregion."}MetadataDatabase
{Resolution, Frame Rate,Duration, Bitrate,Bitrate_trend}Locationof[“subject”]High Level Feature
GlobalQuality Summary Database
LocalQuality Description DatabaseOverallperceptual quality description.Describe/Compare the[“dimension”]of the [“subject”].
Knowledge Retrieval
Question
…CalculateSimilarityQuestionMetadataVisual focusGlobal QualityLocal QualitySub Location
Sampled Visual Input
Final Answer
UserInput
Source AugmenterSubject Localization Database
Source SelectorDatabasesTextEmbeddingQuestion-Relevant Knowledge
AnswerGenerator
Low Level Feature
Visual Input:
Figure 2: The overall workflow of our QualiRAG. Given a visual input and a quality-related question, QualiRAG first decomposes the query
into structured requests via aQuery Organizer, then constructs complementary auxiliary knowledge sources through aSource Augmenter.
ASource Selectorretrieves question-relevant evidence from multiple knowledge databases, which is finally integrated with sampled visual
inputs by theAnswer Generatorto produce evidence-grounded and interpretable quality reasoning.
grounding generation on externally retrieved knowledge. By
incorporating relevant information from curated databases,
RAG enables LMMs to produce responses that are more fac-
tual, interpretable, and domain-aware, while mitigating hallu-
cination [Gaoet al., 2023b ]. A typical RAG pipeline consists
of three stages: (1) preprocessing and organizing raw data
into a structured knowledge base; (2) retrieving task-relevant
information conditioned on the input query; (3) condition-
ing the generation process on the retrieved content. RAG
has been successfully applied to a variety of multimodal
tasks, including visual question answering [Linet al., 2023;
Luoet al., 2024 ], visual captioning [Wuet al., 2024a;
Liet al., 2024b ], and multimodal content generation [Zhang
et al., 2024; Lyuet al., 2025 ]. In this work, we extend RAG to
visual quality understanding, demonstrating its effectiveness
in grounding quality-aware multimodal reasoning.
3 Method
The framework of QualiRAG is illustrated in Figure 2. Given
a visual inputVand a quality-related questionQ, QualiRAG
adopts a training-free, retrieval-augmented reasoning pipeline
composed of four collaborative modules: (1)Query Orga-
nizerthat reformulates the input question into structured,
retrieval-oriented requests; (2)Source Augmenterthat con-
structs and populates multiple auxiliary quality knowledge
databases to produce candidate evidence; (3)Source Selec-
torthat retrieves knowledge that is semantically aligned with
the original question; and (4)Answer Generatorthat inte-
grates visual inputs with the retrieved auxiliary knowledge to
generate grounded reasoning and the final answer. In prac-
tice, the framework is instantiated using two LMMs, denoted
asLMM mainandLMM aux. The main LMM,LMM main, per-
forms query interpretation and final reasoning, while the aux-
iliary LMM,LMM aux, is responsible for auxiliary knowledge
construction.
3.1 Query Organizer: Granularity-Aware Query
Formulation
The Query Organizer reformulates quality-related questions
into structured representations that shape the granularity of
subsequent perception and retrieval. Given a quality-relatedquestionQ, we perform structured question decoupling to
extract its key semantic components in a text-only manner,
where a LMM is prompted to analyze the question without
accessing any visual inputs and generate a structured request
that specifies what auxiliary information should be queried
for quality reasoning.
Concretely, we prompt the LMM with a decoupling in-
structionP dec(see Appendix B.1) to transform the original
questionQinto a structured requestRas follows:
R= LMM main(Pdec, Q),(1)
whereRis represented as a JSON object with four fields:
R={subject,dimension,scope,focus}.(2)
Here,subjectidentifies the primary visual entity refer-
enced in the question,dimensionspecifies the relevant
quality dimension (e.g., blur, noise, artifact, or temporal sta-
bility),scopeindicates whether the queried quality concerns
spatial or temporal aspects, andfocusdenotes the spatial
region or temporal segment emphasized by the question. If a
component is not explicitly mentioned or cannot be reliably
inferred from the question, the corresponding field is set to
NULL.
3.2 Source Augmenter: Multi-Source Knowledge
Generation
The Source Augmenter construct multiple auxiliary knowl-
edge databases, denoted asK, which provide query-specific,
multi-granularity contextual information beyond raw visual
input. Specifically, we build four complementary knowledge
databases: (1) a metadata databaseK meta, (2) a subject local-
ization databaseK loc, (3) a global quality summary database
KglobalQ , and (4) a local quality description databaseK localQ .
Metadata Database.Existing LMMs infer visual quality pri-
marily from perceptual cues in the visual content, without
explicitly observing low-level technical properties, such as
codec- and bitrate-level signals that characterize the under-
lying acquisition and compression process. Such technical
factors provide important context for interpreting compres-
sion artifacts and temporal variations in visual quality. To
make these signals available during reasoning, we extract a

set of technical metadata usingffprobe [FFmpeg Develop-
ers, 2025 ], including spatial resolutionr, frame ratef, video
durationd, average bitrateb avg, and temporal bitrate variation
trendsb trend. For image inputs, only the spatial resolution is
available and thus included as metadata. Formally, the meta-
data database is defined as
Kmeta(V) ={r, f, d, b avg, btrend}.(3)
Subject Localization Database.Although modern LMMs
demonstrate strong object recognition capabilities, their spa-
tial grounding is often coarse, which limits their effectiveness
for queries involvinglocal quality distortionsthat require ac-
curate region-level assessment. To provide explicit spatial
grounding, we construct a subject localization database that
identifies the spatial extent of the queried subject across sam-
pled frames. When a subject is specified inR, we localize it
in each frame ofFsusing APE [Shenet al., 2024 ], a prompt-
driven object detection model that takes natural language de-
scriptions as input and returns subject-specific bounding re-
gions. The subject localization database is defined as
Kloc(V, s) ={(t,B t)|t∈ F s},(4)
wheres=R[subject], andB tdenotes the predicted
bounding region of the queried subject in sampled framet.
Global Quality Summary Database.The global quality
summary database captures the overall perceptual character-
istics of a visual input at a coarse level. For each image or
video, we prompt the queried LMM with a unified global
quality queryQ global (see Appendix B.2) to generate a holis-
tic description that reflects coarse-grained quality attributes,
such as overall clarity, stability, color fidelity, and compres-
sion artifacts at the image or video level. Formally, the global
quality summary database is defined as
KglobalQ (V) = LMM aux(V, Q global).(5)
Local Quality Description Database.To complement
global perceptual summaries, the local quality descrip-
tion database captures on fine-grained, question-driven, and
subject-aware quality attributes. The local quality query
Qlocal (see Appendix B.2) is instantiated based on the
subjectanddimensionfields inR. WhenRcorre-
sponds to a single visual input,Q localprompts the LMM to
describe the specified quality dimension of the queried sub-
ject; for multiple visual inputs, it is adapted to elicit compar-
ative descriptions across visual inputs.
The specific visual input provided to the LMM is deter-
mined by thescopeandfocusfields inR: for spatial-
scope queries, subject-centric regions corresponding to the
queriedsubjectare extracted using the subject local-
ization databaseK loc, optionally constrained by the spec-
ifiedfocus, to support region-level quality analysis; for
temporal-scope queries, full video frames are retained to pre-
serve the temporal segment indicated byfocusand capture
quality variations over time.
To consolidate local quality information, the LMM is
queriedn ltimes, and consistent content across the generateddescriptions is aggregated to form the local quality descrip-
tion databaseK localQ :
KlocalQ(V,R) = Aggregate 
{LMM aux(V, Q local(R))}nl
i=1
,
(6)
where detailed implementation of the aggregation strategy is
provided in Appendix B.2.
3.3 Source Selector: Relevance-Aware Knowledge
Retrieval
The Source Selector aims to retrieve evidence that is seman-
tically aligned with the original quality-related question from
auxiliary knowledge databasesK, while suppressing redun-
dant or irrelevant information. This module serves two pur-
poses: (1) filtering noisy or weakly related knowledge candi-
dates, and (2) reducing the overall context length to improve
inference efficiency and reasoning focus.
Following Video-RAG [Luoet al., 2024 ], we encode the
input questionQand each candidate knowledge entryd∈ K
into a shared semantic embedding space using theContriever
framework [Izacardet al., 2021 ]. Semantic relevance be-
tween the question and each knowledge unit is computed via
dense inner-product similarity. To enable scalable and effi-
cient retrieval, we index all knowledge embeddings using the
FAISSlibrary [Johnsonet al., 2019 ]and perform approxi-
mate nearest-neighbor search. Only knowledge entries whose
similarity scores exceed a predefined thresholdτare retained
for subsequent reasoning. Formally, the filtered knowledge
set is defined as:
K∗(Q) ={d∈ K |sim(d, Q)≥τ},(7)
wheresim(·,·)denotes the dense inner-product similarity
computed via inner-product search.
3.4 Answer Generator: Knowledge-Grounded
Quality Reasoning
The Answer Generator constitutes the final stage of Quali-
RAG, where the visual inputV, the quality-related ques-
tionQ, and the retrieved auxiliary knowledgeK∗are jointly
integrated to produce grounded quality reasoning and an-
swers. By conditioning the LMM on these complementary
inputs, the model aligns perceptual observations with exter-
nal quality-related evidence, thereby mitigating hallucination
and improving reasoning fidelity. The final answerAis gen-
erated as
A= LMM main(V,K∗, Q).(8)
4 Experiments
4.1 Experimental Setups
We evaluate QualiRAG from two complementary perspec-
tives:visual quality understandingandvisual quality com-
parison. The former constitutes the primary task of our
model, while the latter serves as a dedicated evaluation pro-
tocol for visual quality rating, another important visual qual-
ity perception task, enabling direct comparison with models
specifically designed for quality rating.
Validation Benchmarks.Forvisual quality under-
standing, we evaluate the models on two representative

Table 1: Results on thedevsubset of Q-bench for the image quality understanding ability of LMMs. TheLabelscolumn indicates whether
a model requires VQA-lablled data for training. Models shown in gray text correspond to closed-sourced proprietary LMMs.Red,blue, and
boldmark the top three results after excluding proprietary models, consistent with the following tables.
Sub-categories Question Types Quality ConcernsOverall↑LMM LabelsYes-or-No↑What↑How↑Distortion↑Other↑I-C Distortion↑I-C Other↑
Open-sourced General-purpose LMMs∼8B
LLaV A-OneVision-7B✗81.27% 80.31% 68.56% 73.74%77.78%74.67% 84.08% 76.79%
mPLUG-Owl3-7B✗63.48% 77.85% 50.87% 57.80% 66.89% 67.33% 64.00% 63.40%
InternVL3-9B✗80.18% 78.76% 68.76% 72.76% 75.69% 73.68%86.12%75.99%
InternVL3.5-8B✗80.36% 79.87% 69.78% 75.10% 74.54% 76.64% 84.08% 76.72%
Qwen2.5-VL-7B✗81.45%83.19%68.97% 75.10% 76.85% 77.96%85.31%77.86%
Qwen3-VL-8B✗80.91%83.85% 70.99% 77.43% 77.08% 78.62%83.27%78.53%
GLM-4.1V-9B✗81.45% 80.53% 69.98% 76.07% 75.23% 76.64% 84.90% 77.39%
VQA-finetuned LMMs
Q-Instruct✔76.91% 65.04% 55.78% 64.01% 67.13% 64.80% 71.84% 66.35%
Co-Instruct✔82.00%76.77% 66.73% 75.29% 71.53% 74.67% 83.27% 75.38%
Q-Insight✔82.00%82.30% 68.56% 76.07% 76.39% 77.30% 83.67% 77.66%
VisualQuality-R1✔82.91%82.30%71.60% 77.63% 77.78% 79.61%83.27%79.00%
Closed-sourced Proprietary LMMs
GPT-5✗82.18% 84.51% 74.24% 77.63% 79.86% 80.92% 85.71% 80.27%
Gemini-3-Pro✗82.91% 88.48% 71.05% 77.33% 78.83% 83.80% 86.67% 80.63%
Our RAG-based LMM
QualiRAG✗85.82% 84.60% 74.59% 77.01% 81.71% 85.20% 87.35% 81.74%
Table 2: Results on thedevsubset of Q-Bench-Video for the video quality understanding ability of LMMs. Q-Router (GPT-4o) indicates that
GPT-4o is used as the reasoning backbone. Note that Q-Router integrates multiple label-trained no-reference VQA models and is therefore
categorized as a training-based method.
Sub-categories Question Types Quality ConcernsOverall↑LMM LabelsYes-or-No↑What-How↑Open-ended↑Tech.↑Aes.↑Temp.↑AIGC↑
Open-sourced General-purpose LMMs∼8B
LLaV A-OneVision-7B✗62.13% 52.23% 38.56% 48.74% 61.53% 48.81%44.57%52.12%
mPLUG-Owl3-7B✗60.82% 56.52% 35.84% 51.34% 60.46% 54.26% 37.30% 52.44%
InternVL3-9B✗56.57% 51.58% 30.40% 46.35% 59.80% 50.83% 31.51% 47.61%
InternVL3.5-8B✗61.97% 51.13% 35.31% 49.28%65.17%54.33% 29.14% 50.70%
Qwen2.5-VL-7B✗68.54%57.01%35.56% 54.94% 63.12% 57.27% 35.70% 55.30%
Qwen3-VL-8B✗62.44% 51.81% 38.92% 51.68% 62.85% 54.90% 31.94% 52.11%
GLM-4.1V-9B✗63.85% 47.96% 31.45% 49.28% 60.77% 50.59% 24.25% 49.15%
VQA-finetuned LMMs
VQA2✔73.81%56.40% 38.33%60.70%56.65% 61.09% 38.11% 56.67%
VQAThinker✔70.42% 56.56% 35.15% 55.95% 64.56%61.69%30.91% 55.70%
OmniVQA-Chatter✔75.51% 59.76% 40.37% 62.05%61.58%63.45%42.38%59.08%
Q-Router (GPT-4o)✔76.00% 57.01% 43.33%59.31%65.31% 61.46% 50.22% 60.07%
Closed-sourced Proprietary LMMs
GPT-5✗68.72% 56.49% 43.21% 56.58% 68.99% 59.91% 37.42% 57.22%
Gemini-3-Pro✗76.78% 61.00% 47.87% 61.00% 69.56% 59.49% 60.00% 63.02%
Our RAG-based LMM
QualiRAG✗73.18%62.95% 44.95% 61.34% 70.74% 61.69% 48.17% 61.66%
benchmarks,Q-Bench [Wuet al., 2023b ]andQ-Bench-
Video [Zhanget al., 2025 ], which assess LMMs’ ability to
perceive and reason about visual quality attributes through
question answering. Q-Bench targets image-level quality un-
derstanding with tasks probing both global and localized per-
ceptual factors, while Q-Bench-Video extends the evaluation
to videos, covering technical, aesthetic, temporal, and AIGC-
related distortions under both single-video and pairwise com-
parison settings. We report results on the publicly available
devsubsets of both benchmarks.
Forvisual quality comparison, we evaluate on three im-
age benchmarks: LIVE-C [Ghadiyaram and Bovik, 2015 ],
AGIQA [Liet al., 2023 ], and PIPAL [Jinjinet al., 2020 ],
and three video benchmarks: KoNViD-1k [Hosuet al.,
2017 ], VDPVE [Gaoet al., 2023a ], and LIVE-HFR [Mad-
husudanaet al., 2021 ]. Among these benchmarks, LIVE-
C and KoNViD-1k focus on user-generated content cap-
tured in unconstrained real-world scenarios with authentic
distortions. AGIQA evaluates image quality understanding
for AI-generated content, while PIPAL and VDPVE target
quality variations introduced by image and video processingpipelines, such as compression and enhancement. LIVE-HFR
specifically benchmarks temporal perceptual distortions aris-
ing from different frame rates. For each benchmark, we itera-
tively select one image or video and randomly sample another
instance to form a comparison pair. Once constructed, the
same set of pairs is used for all evaluated methods to ensure
fair comparison.
Competing Methods.Forvisual quality understanding,
we compare QualiRAG against three categories of LMMs:
•Open-source general-purpose LMMs, including
LLaV A-OneVision-7B [Liet al., 2024a ], mPLUG-
Owl3-7B [Yeet al., 2024 ], InternVL3-9B [Zhuet
al., 2025b ], InternVL3.5-8B [Wanget al., 2025b ],
Qwen2.5-VL-7B [Baiet al., 2025 ], Qwen3-VL-
8B[Yanget al., 2025 ], and GLM-4.1V-9B [Honget al.,
2025 ].
•VQA-finetuned LMMs, which are specifically adapted
for quality-related question answering. This group in-
cludes Q-Instruct [Wuet al., 2024b ], Co-Instruct [Wu
et al., 2024c ], Q-Insight [Liet al., 2025 ], and

Table 3: Performance comparison on the image quality compari-
son task.♢denotes methods trained specifically for quality scoring,
while♠indicates question-answering–based methods. These con-
ventions are consistent with those in Table 4.
BenchmarksLIVE-C AGIQA PIPALOverall↑Models Labels
Open-sourced General-purpose LMMs∼8B
♠LLaV A-OneVision-7B✗76.54%78.95%67.20% 76.12%
♠InternVL3-9B✗66.10% 64.05% 57.00% 63.15%
♠InternVL3.5-8B✗69.35% 64.86% 58.80% 64.70%
♠Qwen2.5-VL-7B✗78.42% 78.40%70.40%76.85%
♠Qwen3-VL-8B✗81.67% 77.19% 68.40% 76.50%
DNN-based VQA Methods
♢MUSIQ✔82.36% 70.82% 66.80% 72.66%
♢TRES✔81.85% 72.03% 68.00% 73.47%
♢CLIP-IQA+✔84.18% 74.74% 65.40% 75.07%
♢LIQE✔85.96%74.60% 68.40% 75.97%
♢TOPIQ✔82.36% 73.96% 66.40% 74.40%
LMM-based VQA Methods
♢Q-Align✔87.33% 79.07%67.00%78.60%
♠Q-Instruct✔56.68% 62.91% 60.60% 61.05%
♠Co-Instruct✔83.56% 77.20% 66.80% 76.62%
♢Q-Insight✔84.59% 76.94% 68.00% 76.94%
♢VisualQuality-R1✔86.30%78.29%68.80% 78.26%
Ours RAG-based Method
♠QualiRAG✗85.62%79.28% 73.40% 79.58%
Table 4: Performance comparison on the video quality comparison
task.
BenchmarksKoNViD-1K VDPVE LIVE-HFROverall↑Models Labels
Open-sourced General-purpose LMMs∼8B
♠LLaV A-OneVision-7B✗75.33% 67.30%62.92%70.29%
♠InternVL3-9B✗71.00% 61.34% 57.92% 65.29%
♠InternVL3.5-8B✗73.33% 68.74% 58.33% 68.94%
♠Qwen2.5-VL-7B✗73.50% 68.50% 60.42% 69.34%
♠Qwen3-VL-8B✗75.33% 69.69% 59.58% 70.45%
DNN-based VQA Methods
♢FAST-VQA✔84.17% 72.79% 60.42% 75.86%
♢DOVER✔85.50% 73.51% 62.50% 77.13%
♢COVER✔84.17% 73.03%63.33%76.49%
♢MinimalisticVQA✔83.33% 72.79% 56.67% 74.74%
LMM-based VQA Methods
♢Q-Align✔86.17%74.22% 62.08%77.60%
♠Q-Instruct✔56.50% 54.18% 49.17% 54.33%
♠Co-Instruct✔76.50% 65.87% 51.25% 68.15%
♢VQA2✔86.50% 76.13%60.00%78.00%
♢VQAThinker✔87.17% 77.33%62.08%79.11%
Ours RAG-based Method
♠QualiRAG✗80.33%74.46% 63.75%75.22%
VisualQuality-R1 [Wuet al., 2025 ]for image-based
quality understanding, as well as VQA2[Jiaet al.,
2024 ], VQAThinker [Caoet al., 2025 ], OmniVQA-
Chatter [Jiaet al., 2025 ], and Q-Router [Xinget al.,
2025 ]for video-based settings.
•Closed-sourced proprietary LMMs, including GPT-
5[OpenAI, 2025 ]and Gemini-3-Pro [DeepMind, 2025 ].
Forvisual quality comparison, we further compare Quali-
RAG with representative VQA methods specifically de-
signed for quality scoring. For images, these include
MUSIQ [Keet al., 2021 ], TRES [Golestanehet al., 2022 ],
CLIP-IQA+ [Wanget al., 2023 ], LIQE [Zhanget al.,
2023 ], TOPIQ [Chenet al., 2024 ]and Q-Align [Wuet
al., 2023c ]; for videos, we consider FAST-VQA [Wuet
al., 2022 ], DOVER [Wuet al., 2023a ], COVER [Heet
al., 2024 ]and MinimalisticVQA [Sunet al., 2024 ]. For
VQA methods trained to output scalar quality scores, we
compute score differences between paired images or videosand evaluate pairwise accuracy. For question-answering-
based methods, we report their classification accuracy on
the binary-choice questions:‘‘Which video/image
has better visual quality?’’with candidate an-
swers:(A) The first video/imageand(B) The
second video/image.
Implementation Details.The LMM mainand LMM auxare
instantiated by InternVL3-9B-Instruct and Qwen3-VL-8B-
Instruct respectively. As analyzed in Section 4.2, InternVL3-
9B-Instruct exhibits relatively limited performance on vi-
sual quality understanding tasks, thereby providing a suit-
able testbed to verify that the observed performance gains
are attributable to QualiRAG rather than the intrinsic capa-
bility of the backbone. In contrast, Qwen3-VL-8B-Instruct
demonstrates strong capability in perceptual quality descrip-
tion, making it well suited for constructing auxiliary quality
knowledge. For video inputs, we sample1fps as the visual
input. Visual inputs processed by Qwen3-VL-8B-Instruct are
kept at their original resolution to support fine-grained aux-
iliary knowledge construction, whereas inputs to InternVL3-
9B-Instruct are resized to448×448following its default in-
put configuration. The hyperparametersτin Eq. (7) andn lin
Eq. (6) are set to0.25and4, respectively.
4.2 Performance Analysis
Visual Quality Understanding.The results on thedevsub-
sets of Q-Bench and Q-Bench-Video are reported in Table 1
and Table 2, respectively. On Q-Bench, QualiRAG achieves
an overall accuracy of81.74%,outperforming all open-
source general LMMs, VQA-finetuned LMMs, and even
surpassing strong proprietary modelsGPT-5 and Gemini-
3-Pro. Similarly, on Q-Bench-Video, QualiRAG attains an
overall accuracy of61.66%,outperforming all open-source
general LMMs and VQA-finetuned LMMs, including Q-
route, an agentic framework that leverages a diverse set of
VQA expert models with GPT-4o [Achiamet al., 2023 ]as its
backbone. Moreover, QualiRAG surpasses GPT-5 and trails
Gemini-3-Pro by only1.36%. These results demonstrate that
retrieval-augmented generation with structured query decom-
position and relevance-aware retrieval over complementary
global and local quality evidence effectively improves vi-
sual quality understanding, without relying on task-specific
instruction tuning or VQA expert models.
Visual Quality Comparison.The performance on image
and video quality comparison tasks is reported in Table 3
and Table 4, respectively. For image quality comparison,
QualiRAG achieves the best overall performance among
all evaluated methods. Most competing approaches perform
well on LIVE-C, which serves as an in-domain benchmark
for methods trained on the KonIQ dataset [Hosuet al., 2020 ],
but exhibit noticeable performance degradation on AGIQA,
containing AI-generated images, and on PIPAL, focusing on
processing-induced distortions. In contrast, QualiRAG main-
tains consistently strong performance across all image bench-
marks. This indicates that the proposed training-free RAG
framework provides more robust image quality assessment
under out-of-domain settings. For video quality compar-
ison, QualiRAG also demonstrates strong competitiveness.

Table 5: Ablation study of the contributions of four complementary
knowledge databases to overall performance on thedevsubset of
Q-Bench-Video.∗indicates that LMM auxis replaced with the VQA-
finetuned model VQA2.
Kmeta Kloc KglobalQ KlocalQ Overall↑
47.61%
✔49.22%
✔ ✔51.19%
✔ ✔ ✔55.84%
✔ ✔ ✔ ✔61.66%
✔ ✔ ✔∗✔∗64.12%
It substantially outperforms open-source general-purpose
LMMs and achieves performance comparable to DNN-
based VQA modelstrained on the LSVQ dataset [Yinget
al., 2021 ]. While LMM-based methods fine-tuned on LSVQ
(e.g., VQAThinker) attain higher overall accuracy, Quali-
RAG delivers more favorable performance on out-of-domain
benchmarks such as VDPVE and LIVE-HFR. This observa-
tion further highlights the robustness of QualiRAG for visual
quality comparison, a task that directly reflects visual quality
rating capability.
4.3 Ablation Study
We perform ablation studies on the four complementary
quality-related knowledge databases and on different back-
bone LMMs used in the Answer Generator. All ablation stud-
ies are performed on Q-Bench-Video, as it poses a more chal-
lenging setting for visual quality understanding.
Knowledge Databases.The knowledge databasesKconsti-
tute a core component of QualiRAG for enhancing the visual
quality understanding capability. To quantify the contribu-
tion of each auxiliary database, we progressively enable the
four knowledge sources and additionally replace the auxiliary
modelLMM auxfrom Qwen3-VL-8B-Instruct with the VQA2
expert to derive more VQA-specific knowledge. This proce-
dure yields a clear monotonic improvement in performance.
Specifically, introducing the metadata databaseK meta
yields a1.61%gain, indicating that low-level technical meta-
data provides useful contextual cues for quality reasoning.
Adding the subject localization databaseK locfurther im-
proves performance by1.97%, demonstrating that explicit
spatial grounding helps the model focus on question-relevant
regions. Incorporating the global quality summary database
KglobalQ results in a more substantial improvement of4.65%,
highlighting the importance of holistic perceptual quality de-
scriptions. The local quality description databaseK localQ de-
livers the largest gain of5.82%, underscoring the critical
role of fine-grained, question-aligned local quality evidence
in visual quality understanding. WhenLMM auxis further
replaced with the VQA2expert model, the overall accuracy
reaches64.12%, representing the best performance among all
variants and even surpassing Gemini-3-Pro.
Backbone-Agnostic Performance.We investigate the im-
pact of the backbone LMM mainon QualiRAG by replacing the
default backbone with a diverse set of open-source general-
purpose LMMs. As shown in Figure 3, without QualiRAG,
the evaluated backbones exhibit notable performance differ-
ences, and even the strongest model, Qwen2.5-VL, achieves
only moderate accuracy. After incorporating QualiRAG, all
0.40.450.50.550.652.1%47.6%50.7%55.3%52.1%61.3%61.7%60.6%60.5%61.1%
LLaVA-OVInternVL3InternVL3.5Qwen2.5Qwen3AverageAfterBefore9%
14.1%
9.8%
5.3%
9.1%
9.5%
51.6%61.0%Figure 3: Performance of LLaV A-OneVision-7B, InternVL3-9B-
Instruct, InternVL3.5-8B-Instruct, Qwen2.5-VL-7B-Instruct, and
Qwen3-VL-8B-Instruct on thedevsubset of Q-Bench-Video, be-
fore and after applying QualiRAG.
backbones achieve consistent performance improvements and
converge to a similar level of approximately61%, with vari-
ations within1%. This convergence indicates that the effec-
tiveness of QualiRAG is largely backbone-agnostic, suggest-
ing that the observed improvements primarily stem from the
quality-related knowledge provided by the framework rather
than the intrinsic capabilities of individual backbones. These
results demonstrate that QUALIRAG is robust to backbone
choices and can be seamlessly applied to a wide range of
LMM architectures.
5 Discussion
Visual quality perception is a fundamental capability of the
human visual system, yet it has received limited attention
in the development of large multimodal models. In this
work, the proposed QualiRAG framework demonstrates that
grounding inference on quality-related evidence can substan-
tially enhance the visual quality perception ability of LMMs.
Compared to training-based paradigms, QualiRAG offers sig-
nificant advantages inscalability.First, QualiRAG is entirely
training-free, enabling seamless integration with updated or
more powerful vision-language backbones.Second, the aux-
iliary knowledge databases can be dynamically extended ac-
cording to specific evaluation requirements; for instance,
when assessing AI-generated content, domain-specific arti-
fact detectors [Huanget al., 2024 ]can be incorporated to
explicitly capture AIGC-related distortions.Third, Quali-
RAG is complementary to existing VQA models; for exam-
ple, replacing the quality description generators with VQA2
improves performance on Q-Bench-Video beyond Gemini-3-
Pro, highlighting their synergistic potential. Overall, Quali-
RAG provides a flexible and extensible approach to visual
quality assessment that can adapt to a broad range of real-
world scenarios.
6 Conclusion
We propose QualiRAG, a training-free retrieval-augmented
generation framework for visual quality understanding. In-
stead of relying on task-specific fine-tuning based on large-
scale quality instruction data, QualiRAG improves visual
quality perception ability by grounding inference on retrieved
quality-related evidence. By decomposing quality-related
questions into retrieval-oriented queries and integrating com-
plementary knowledge sources at different granularity levels,

QualiRAG enables effective and interpretable visual quality
analysis. Experimental results validate the effectiveness of
QualiRAG for both visual quality understanding and quality
comparison.
References
[Achiamet al., 2023 ]Josh Achiam, Steven Adler, Sandhini
Agarwal, Lama Ahmad, Ilge Akkaya, et al. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774, 2023.
[Baiet al., 2025 ]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, et al. Qwen2. 5-vl technical report.
arXiv preprint arXiv:2502.13923, 2025.
[Caoet al., 2025 ]Linhan Cao, Wei Sun, Weixia Zhang, Xi-
angyang Zhu, Jun Jia, et al. Vqathinker: Exploring gener-
alizable and explainable video quality assessment via re-
inforcement learning. InAAAI, 2025.
[Chenet al., 2023 ]Yixiong Chen, Li Liu, and Chris Ding.
X-iqe: explainable image quality evaluation for text-to-
image generation with visual large language models.arXiv
preprint arXiv:2305.10843, 2023.
[Chenet al., 2024 ]Chaofeng Chen, Jiadi Mo, Jingwen Hou,
Haoning Wu, Liang Liao, et al. Topiq: A top-down ap-
proach from semantics to distortions for image quality as-
sessment.TIP, 2024.
[DeepMind, 2025 ]DeepMind. Gemini 3 pro. https://
deepmind.google/models/gemini/pro/, 2025.
[Elo and Sloan, 1978 ]Arpad E Elo and Sam Sloan. The rat-
ing of chessplayers: Past and present.(No Title), 1978.
[FFmpeg Developers, 2025 ]FFmpeg Developers. ffprobe
documentation. https://ffmpeg.org/ffprobe.html, 2025.
[Gaoet al., 2023a ]Yixuan Gao, Yuqin Cao, Tengchuan Kou,
Wei Sun, Yunlong Dong, et al. Vdpve: Vqa dataset for
perceptual video enhancement. InCVPR, 2023.
[Gaoet al., 2023b ]Yunfan Gao, Yun Xiong, Xinyu Gao,
Kangxiang Jia, Jinliu Pan, et al. Retrieval-augmented
generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2023.
[Ghadiyaram and Bovik, 2015 ]Deepti Ghadiyaram and
Alan C Bovik. Massive online crowdsourced study of
subjective and objective picture quality.TIP, 2015.
[Golestanehet al., 2022 ]S Alireza Golestaneh, Saba Dad-
setan, and Kris M Kitani. No-reference image quality
assessment via transformers, relative ranking, and self-
consistency. InWACV, 2022.
[Heet al., 2024 ]Chenlong He, Qi Zheng, Ruoxi Zhu, Xi-
aoyang Zeng, Yibo Fan, et al. Cover: A comprehensive
video quality evaluator. InCVPR, 2024.
[Honget al., 2025 ]Wenyi Hong, Wenmeng Yu, Xiaotao Gu,
Guo Wang, et al. Glm-4.1 v-thinking: Towards versatile
multimodal reasoning with scalable reinforcement learn-
ing.arXiv preprint arXiv:2507.01006, 2025.
[Hosuet al., 2017 ]Vlad Hosu, Franz Hahn, Mohsen Je-
nadeleh, Hanhe Lin, Hui Men, et al. The konstanz natural
video database (konvid-1k). InQoMEX, 2017.[Hosuet al., 2020 ]Vlad Hosu, Hanhe Lin, Tamas Sziranyi,
and Dietmar Saupe. Koniq-10k: An ecologically valid
database for deep learning of blind image quality assess-
ment.TIP, 2020.
[Huanget al., 2024 ]Ziqi Huang, Yinan He, Jiashuo Yu, Fan
Zhang, et al. Vbench: Comprehensive benchmark suite for
video generative models. InCVPR, 2024.
[Izacardet al., 2021 ]Gautier Izacard, Mathilde Caron, Lu-
cas Hosseini, Sebastian Riedel, Piotr Bojanowski, et al.
Unsupervised dense information retrieval with contrastive
learning.arXiv preprint arXiv:2112.09118, 2021.
[Jiaet al., 2024 ]Ziheng Jia, Zicheng Zhang, Jiaying Qian,
Haoning Wu, Wei Sun, et al. Vqa2: Visual question an-
swering for video quality assessment.ACM MM, 2024.
[Jiaet al., 2025 ]Ziheng Jia, Zicheng Zhang, Zeyu Zhang,
Yingji Liang, Xiaorong Zhu, et al. Scaling-up perceptual
video quality assessment. InAAAI, 2025.
[Jinjinet al., 2020 ]Gu Jinjin, Cai Haoming, Chen Haoyu,
Ye Xiaoxing, Jimmy S Ren, et al. Pipal: a large-scale
image quality assessment dataset for perceptual image
restoration. InECCV, 2020.
[Johnsonet al., 2019 ]Jeff Johnson, Matthijs Douze, and
Herv ´e J´egou. Billion-scale similarity search with gpus.
IEEE TBD, 2019.
[Keet al., 2021 ]Junjie Ke, Qifei Wang, Yilin Wang, Pey-
man Milanfar, and Feng Yang. Musiq: Multi-scale image
quality transformer. InICCV, 2021.
[Lewiset al., 2020 ]Patrick Lewis, Ethan Perez, Aleksan-
dra Piktus, Fabio Petroni, Vladimir Karpukhin, et al.
Retrieval-augmented generation for knowledge-intensive
nlp tasks.NeurIPS, 2020.
[Liet al., 2023 ]Chunyi Li, Zicheng Zhang, Haoning Wu,
Wei Sun, et al. Agiqa-3k: An open database for ai-
generated image quality assessment.TCSVT, 2023.
[Liet al., 2024a ]Bo Li, Yuanhan Zhang, Dong Guo, Renrui
Zhang, Feng Li, et al. Llava-onevision: Easy visual task
transfer.arXiv preprint arXiv:2408.03326, 2024.
[Liet al., 2024b ]Wenyan Li, Jiaang Li, Rita Ramos,
Raphael Tang, and Desmond Elliott. Understanding re-
trieval robustness for retrieval-augmented image caption-
ing.arXiv preprint arXiv:2406.02265, 2024.
[Liet al., 2025 ]Weiqi Li, Xuanyu Zhang, Shijie Zhao,
Yabin Zhang, et al. Q-insight: Understanding image qual-
ity via visual reinforcement learning.NeurIPS, 2025.
[Linet al., 2023 ]Weizhe Lin, Jinghong Chen, Jingbiao Mei,
Alexandru Coca, and Bill Byrne. Fine-grained late-
interaction multi-modal retrieval for retrieval augmented
visual question answering.NeurIPS, 2023.
[Liuet al., 2025 ]Jie Liu, Gongye Liu, Jiajun Liang, Ziyang
Yuan, Xiaokun Liu, et al. Improving video generation with
human feedback.arXiv preprint arXiv:2501.13918, 2025.
[Luoet al., 2024 ]Yongdong Luo, Xiawu Zheng, Guilin Li,
Shukang Yin, Haojia Lin, et al. Video-rag: Visually-
aligned retrieval-augmented long video comprehension.
arXiv preprint arXiv:2411.13093, 2024.

[Lyuet al., 2025 ]Yuanhuiyi Lyu, Xu Zheng, Lutao Jiang,
Yibo Yan, Xin Zou, et al. Realrag: Retrieval-augmented
realistic image generation via self-reflective contrastive
learning.arXiv preprint arXiv:2502.00848, 2025.
[Mackinet al., 2018 ]Alex Mackin, Fan Zhang, and David R
Bull. A study of high frame rate video formats.TMM,
2018.
[Madhusudanaet al., 2021 ]Pavan C Madhusudana, Xi-
angxu Yu, Neil Birkbeck, Yilin Wang, Balu Adsumilli,
et al. Subjective and objective quality assessment of high
frame rate videos.IEEE Access, 2021.
[Minet al., 2024 ]Xiongkuo Min, Huiyu Duan, Wei Sun,
Yucheng Zhu, and Guangtao Zhai. Perceptual video qual-
ity assessment: A survey.SCIS, 2024.
[Mittalet al., 2012 ]Anish Mittal, Rajiv Soundararajan, and
Alan C Bovik. Making a “completely blind” image quality
analyzer.SPL, 2012.
[OpenAI, 2025 ]OpenAI. Gpt-5. https://openai.com/gpt-5/,
2025.
[Shaoet al., 2024 ]Zhihong Shao, Peiyi Wang, Qihao Zhu,
Runxin Xu, Junxiao Song, et al. Deepseekmath: Push-
ing the limits of mathematical reasoning in open language
models.arXiv preprint arXiv:2402.03300, 2024.
[Shenet al., 2024 ]Yunhang Shen, Chaoyou Fu, Peixian
Chen, Mengdan Zhang, Ke Li, et al. Aligning and prompt-
ing everything all at once for universal visual perception.
InCVPR, 2024.
[Sunet al., 2022 ]Wei Sun, Xiongkuo Min, Wei Lu, and
Guangtao Zhai. A deep learning based no-reference qual-
ity assessment model for ugc videos. InACM MM, 2022.
[Sunet al., 2023 ]Wei Sun, Xiongkuo Min, Danyang Tu, Si-
wei Ma, and Guangtao Zhai. Blind quality assessment for
in-the-wild images via hierarchical feature fusion and iter-
ative mixed database training.JSTSP, 2023.
[Sunet al., 2024 ]Wei Sun, Wen Wen, Xiongkuo Min, Long
Lan, Guangtao Zhai, et al. Analysis of video quality
datasets via design of minimalistic video quality models.
TPAMI, 2024.
[Sunet al., 2025 ]Wei Sun, Linhan Cao, Jun Jia, Zhichao
Zhang, et al. Enhancing blind video quality assessment
with rich quality-aware features.ESWA, 2025.
[Thomeeet al., 2016 ]Bart Thomee, David A Shamma,
Gerald Friedland, Benjamin Elizalde, Karl Ni, et al.
Yfcc100m: The new data in multimedia research.CACM,
2016.
[Wanget al., 2023 ]Jianyi Wang, Kelvin CK Chan, and
Chen Change Loy. Exploring clip for assessing the look
and feel of images. InAAAI, 2023.
[Wanget al., 2025a ]Jiarui Wang, Huiyu Duan, Guangtao
Zhai, Juntong Wang, and Xiongkuo Min. Aigv-assessor:
benchmarking and evaluating the perceptual quality of
text-to-video generation with lmm. InCVPR, 2025.
[Wanget al., 2025b ]Weiyun Wang, Zhangwei Gao, Lixin
Gu, Hengjun Pu, Long Cui, et al. Internvl3. 5: Advancingopen-source multimodal models in versatility, reasoning,
and efficiency.arXiv preprint arXiv:2508.18265, 2025.
[Wuet al., 2022 ]Haoning Wu, Chaofeng Chen, Jingwen
Hou, Liang Liao, Annan Wang, et al. Fast-vqa: Efficient
end-to-end video quality assessment with fragment sam-
pling. InECCV, 2022.
[Wuet al., 2023a ]Haoning Wu, Erli Zhang, Liang Liao,
Chaofeng Chen, Jingwen Hou, et al. Exploring video qual-
ity assessment on user generated contents from aesthetic
and technical perspectives. InICCV, 2023.
[Wuet al., 2023b ]Haoning Wu, Zicheng Zhang, Erli Zhang,
Chaofeng Chen, Liang Liao, et al. Q-bench: A bench-
mark for general-purpose foundation models on low-level
vision. InICLR, 2023.
[Wuet al., 2023c ]Haoning Wu, Zicheng Zhang, Weixia
Zhang, Chaofeng Chen, et al. Q-align: Teaching lmms for
visual scoring via discrete text-defined levels. InICML,
2023.
[Wuet al., 2024a ]Hao Wu, Zhihang Zhong, and Xiao Sun.
Dir: Retrieval-augmented image captioning with compre-
hensive understanding.arXiv preprint arXiv:2412.01115,
2024.
[Wuet al., 2024b ]Haoning Wu, Zicheng Zhang, Erli Zhang,
Chaofeng Chen, Liang Liao, et al. Q-instruct: Improv-
ing low-level visual abilities for multi-modality foundation
models. InCVPR, 2024.
[Wuet al., 2024c ]Haoning Wu, Hanwei Zhu, Zicheng
Zhang, Erli Zhang, Chaofeng Chen, et al. Towards open-
ended visual quality comparison. InECCV, 2024.
[Wuet al., 2024d ]Tianhe Wu, Kede Ma, Jie Liang, Yujiu
Yang, and Lei Zhang. A comprehensive study of mul-
timodal large language models for image quality assess-
ment. InECCV, 2024.
[Wuet al., 2025 ]Tianhe Wu, Jian Zou, Jie Liang, Lei Zhang,
and Kede Ma. Visualquality-r1: Reasoning-induced im-
age quality sssessment via reinforcement learning to rank.
arXiv preprint arXiv:2505.14460, 2025.
[Xinget al., 2025 ]Shuo Xing, Soumik Dey, Mingyang Wu,
Ashirbad Mishra, et al. Q-router: Agentic video quality
assessment with expert model routing and artifact local-
ization.arXiv preprint arXiv:2510.08789, 2025.
[Yanget al., 2025 ]An Yang, Anfeng Li, Baosong Yang, Be-
ichen Zhang, Binyuan Hui, et al. Qwen3 technical report.
arXiv preprint arXiv:2505.09388, 2025.
[Yeet al., 2024 ]Jiabo Ye, Haiyang Xu, Haowei Liu, An-
wen Hu, Ming Yan, et al. mplug-owl3: Towards long
image-sequence understanding in multi-modal large lan-
guage models. InICLR, 2024.
[Yinget al., 2021 ]Zhenqiang Ying, Maniratnam Mandal,
Deepti Ghadiyaram, and Alan Bovik. Patch-vq:’patching
up’the video quality problem. InCVPR, 2021.
[Zhanget al., 2023 ]Weixia Zhang, Guangtao Zhai, Ying
Wei, Xiaokang Yang, and Kede Ma. Blind image quality
assessment via vision-language correspondence: A multi-
task learning perspective. InCVPR, 2023.

[Zhanget al., 2024 ]Shiyue Zhang, Zheng Chong, Xujie
Zhang, Hanhui Li, Yuhao Cheng, et al. Garmentaligner:
Text-to-garment generation via retrieval-augmented multi-
level corrections. InECCV, 2024.
[Zhanget al., 2025 ]Zicheng Zhang, Ziheng Jia, Haoning
Wu, Chunyi Li, et al. Q-bench-video: Benchmark the
video quality understanding of lmms. InCVPR, 2025.
[Zhuet al., 2025a ]Hanwei Zhu, Yu Tian, Keyan Ding, Bao-
liang Chen, Bolin Chen, et al. Agenticiqa: An agentic
framework for adaptive and interpretable image quality as-
sessment.arXiv preprint arXiv:2509.26006, 2025.
[Zhuet al., 2025b ]Jinguo Zhu, Weiyun Wang, Zhe Chen,
Zhaoyang Liu, Shenglong Ye, et al. Internvl3: Exploring
advanced training and test-time recipes for open-source
multimodal models.arXiv preprint arXiv:2504.10479,
2025.

QualiRAG: Retrieval-Augmented Generation for Visual Quality Understanding
Appendix
A More Details of Our Testing Benchmarks
Table 6 provides an overview of our testing benchmarks,
which encompass diverse content types, resolutions, dura-
tions, frame rates, and distortion types. In the following, we
provide a detailed description of each benchmark.
•Q-Bench [Wuet al., 2023b ]: Q-Bench is a bench-
mark designed to systematically evaluate the low-level
visual perception capabilities of large multimodal mod-
els (LMMs) across diverse attributes and conditions.
It introduces the LLVisionQA dataset, which contains
2,990 images paired with human-authored low-level vi-
sual questions, covering a wide range of image sources,
including in-the-wild photography, AI-generated con-
tent (AIGC), and artificially distorted images. LLVi-
sionQA comprises three types of questions—Yes-or-No,
What, and How—and organizes low-level visual con-
cerns into four quadrants along two axes: distortion ver-
sus other low-level attributes, and global perception ver-
sus local content-related perception. Together, these de-
sign choices provide a holistic, diverse, and balanced
benchmark for assessing LMMs’ low-level visual per-
ception abilities.
•Q-Bench-Video [Zhanget al., 2025 ]:Q-Bench-Video
is a benchmark designed to systematically evaluate the
video quality understanding capabilities of large mul-
timodal models (LMMs). It covers a wide spectrum
of video content, including natural scenes, AIGC, and
computer graphics. To ensure a balanced distribution
of perceptual quality, videos are sampled from multi-
ple sources with available subjective annotations. The
benchmark includes both multiple-choice and open-
ended questions, enabling the evaluation of model per-
formance under diverse quality assessment settings.
Overall, Q-Bench-Video consists of 1,800 videos and
2,378 annotated question–answer pairs for validation,
providing a comprehensive testbed for assessing video
quality understanding in LMMs.
•LIVE-C [Ghadiyaram and Bovik, 2015 ]:LIVE-C
(LIVE In the Wild Image Quality Challenge Database)
is a large-scale visual quality assessment (VQA) bench-
mark designed to address the limitations of traditional
datasets built under controlled laboratory conditions.
Unlike synthetic-distortion datasets, LIVE-C contains
authentically distorted images captured using a wide
range of real-world mobile devices, reflecting com-
plex mixtures of distortions commonly encountered in
practice. The dataset is constructed via a large-scale
crowdsourcing study, comprising over 350,000 subjec-
tive opinion scores collected from more than 8,100 hu-
man observers on 1,169 images.
•PIPAL [Jinjinet al., 2020 ]:PIPAL (Perceptual Image
Processing Algorithms) is a large-scale VQA dataset de-
signed to evaluate both VQA methods and modern im-
age restoration algorithms, particularly those based onGenerative Adversarial Networks (GANs). The dataset
contains 29,000 distorted images generated from 40 dis-
tortion types and 116 distortion levels. PIPAL is an-
notated with over 1.13 million human judgments col-
lected using an Elo-based [Elo and Sloan, 1978 ]sub-
jective evaluation system, providing reliable perceptual
quality scores. In this work, we evaluate our model on
the validation subset containing 1,000 images.
•AGIQA [Liet al., 2023 ]:AGIQA is a large-scale sub-
jective image quality assessment dataset specifically de-
signed for AI-generated images (AGIs). It contains
2,982 images generated by six representative text-to-
image models, covering diverse generation paradigms
including GAN-, autoregressive-, and diffusion-based
models. The dataset is constructed by carefully varying
prompts and internal generation parameters to capture a
wide range of perceptual quality variations.
•KoNViD-1k [Hosuet al., 2017 ]:KoNViD-1k is an
authentic video quality assessment dataset comprising
1,200 unique test videos that exhibit a wide range of
real-world distortions. All videos are sampled from
the YFCC100M dataset [Thomeeet al., 2016 ]using a
feature-based selection strategy that accounts for blur,
colorfulness, contrast, spatial and temporal informa-
tion, as well as the no-reference image quality metric
NIQE [Mittalet al., 2012 ]. Each video is clipped from
the original source content and resized to 540p with a
landscape aspect ratio. The videos have frame rates of
24, 25, or 30 fps, and each clip has a fixed duration of 8
seconds.
•LIVE-YT-HFR [Madhusudanaet al., 2021 ]:LIVE-
YT-HFR is a video quality assessment dataset designed
to investigate the perceptual effects of frame rate vari-
ation and compression on video quality. The dataset
comprises 16 source sequences and 480 distorted videos,
generated by applying six different frame rates in con-
junction with five levels of VP9 compression, includ-
ing one lossless setting and four CRF-based compres-
sion levels. Among the source sequences, 11 are drawn
from the BVI-HFR dataset [Mackinet al., 2018 ], featur-
ing a resolution of 1920×1080 and a fixed duration of 10
seconds. The remaining five sequences consist of high-
motion sports content captured by Fox Media Group,
provided at a resolution of 3840×2160, with durations
ranging from 6 to 8 seconds.
•VDPVE [Gaoet al., 2023a ]:VDPVE is a video quality
assessment VQA dataset developed for perceptual video
enhancement. It comprises a total of 1,211 enhanced
videos, organized into three subsets: (1) 600 videos
with color, brightness, and contrast enhancements; (2)
310 videos with deblurring enhancements; and (3) 301
videos with deshaking enhancements. In this work, we
evaluate our model on an open-sourced subset of 839
videos from the training split.

Table 6: An overview of our testing Benchmarks.
Benchmark Year # of Data # of Scenes Resolution Duration Frame Rate Distortion Type
Benchmarks for Visual Quality Understanding
Q-Bench 2023 2,990 2,990 Diverse - - Diverse
Q-Bench-Video 2025 1,800 1,800 Diverse Diverse Diverse Diverse
Benchmarks for In-domain Visual Quality Rating
LIVE-C 2016 1,169 1,169 500x500 - - In-the-wild
KoNViD-1k 2017 1,200 1,200 540p 8 24, 25, 30 In-the-wild
Benchmarks for Out-of-domain Visual Quality Rating
PIPAL 2020 29,000 250 288x288 - - synthetic
AGIQA 2023 2,982 2,982 512x512 - - AIGC
LIVE-YT-HFR 2021 480 16 1080p 6-10 24, 30, 60, 82, 98, 120 Frame rate, VP9 compression
VDPVE 2023 1,211 79 Diverse 8-15 24, 25, 30 Enhancement
B More Details of Our QualiRAG Model
B.1 Query Organizer
The decoupling instructionP decused in QualiRAG for trans-
forming raw questions into structured retrieval requests is de-
fined as follows:
Decoupling InstructionP dec
You will analyze the following visual-question
and decompose it into a structured quality-analysis
schema.
Your job is NOT to answer the question.
Instead, extract:
1. The subject referenced in the question. (Identify
the concrete, visually identifiable entity mentioned
in the question (e.g., child, person, road, building,
mountain, object). Abstract terms such as video,
image quality, or distortion should not be considered
valid subjects. If no valid subject exists, set it to
”none”.)
2. The relevant quality dimension(s) such as clarity,
sharpness, blur, noise, blockiness, temporal, etc.
3. The main scope category: spatial or temporal.
4. Determine the most relevant visual area or time
segment the LMM should attend to when assessing
the specified subject and quality dimension.
Return your output in JSON format:
{
“subject”: “...”,
“dimension”: “...”,
“scope”: “...”,
“focus”: “...”
}
Here is the question:[question]
B.2 Source Augmenter
Metadata Database.For video inputs, the spatial resolution
r, frame ratef, video durationd, and average bitrateb avgare directly extracted usingffprobe [FFmpeg Developers,
2025 ]from the video stream metadata. These attributes de-
scribe the basic acquisition and encoding configuration of the
video and are obtained without any additional processing.
In addition to these static properties, we introduce a tempo-
ral bitrate variation indicatorb trendto capture coarse-grained
bitrate dynamics over time. To compute this indicator, we es-
timate the average video bitrate over two temporal segments
of the video: the head segment and the tail segment. Specif-
ically, we sample the first 10% and the last 10% of the video
duration, respectively, and compute the mean bitrate over all
frames within each segment using frame-level bitrate statis-
tics provided byffprobe. Let ¯bheadand¯btaildenote the es-
timated average bitrates of the head and tail segments. The
relative bitrate change ratio is then defined as
∆ =¯btail−¯bhead
max( ¯bhead,1).
Based on this ratio and the overall average bitrateb avg, we
categorize the temporal bitrate trendb trendinto four discrete
types:
•Increasing, if∆>0.1;
•Decreasing, if∆<−0.1;
•Constant high, if|∆| ≤0.1andb avgexceeds a prede-
fined thresholdτ b= 3Mbps;
•Constant low, if|∆| ≤0.1andb avg≤τ b.
Global Quality Summary Database.The detailed global
quality queryQ global is given as follows:
Global Quality QueryQ global
For the given image or video, please provide an over-
all perceptual quality description in terms of structure
and texture preservation, color and luminance repro-
duction, noise, contrast, sharpness, motion, artifacts,
stutter, and other low-level distortions, and explain
the reasons for your assessment.
Local Quality Description Database.Given a structured
requestR, we construct the local quality queryQ local
according to the number of visual inputs. For a single
visual input, the query takes the form:‘‘Describe
theR[dimension] of theR[subject]

in the image/video.’’For multiple visual
inputs, the query is adapted to:‘‘Compare the
R[dimension] of theR[subject] across
all images/videos.’’
To improve robustness and mitigate stochastic variability in
local quality description generation, we query the LMM mul-
tiple times to obtainn lindependent local quality descriptions
for each question. During generation, we adopt temperature-
based sampling withtemperature= 1.0and nucleus
sampling (top p= 0.95) to balance descriptive diversity
and generation stability. Hard top-ktruncation is disabled by
settingtop k= 0, thereby avoiding excessive pruning of
informative quality-related tokens. The prompt used to ag-
gregate consistent information across these descriptions is as
follows:
Aggregation Information Prompt
You are givenn ldifferent descriptions of the visual
quality of the SAME visual input.
Your task is to produce ONE final visual quality de-
scription that reflects the shared, consistent judgment
across these descriptions.
Important rules:
- Do NOT mention that multiple answers were given.
- Do NOT use phrases like “most responses agree”,
“overall”, or “in summary”.
- Write as if you directly observed the image/video
yourself.
Here are the descriptions:
{numbered descriptions}
Write ONE coherent visual quality description.
B.3 Source Selector
We adoptContrieverframework [Izacardet al., 2021 ]as the
text encoder to map both the quality-related questionQand
each candidate knowledge sentencedinto a shared embed-
ding space. Given an input textx, its vector representation is
obtained by mean pooling over the last-layer hidden states of
the encoder:
e(x) = MeanPool 
Contriever(x)
∈Rm.(9)
Both the question embedding and the candidate knowledge
sentence embeddings areℓ 2-normalized to compute inner-
product similarity:
ˆe(x) =e(x)
∥e(x)∥ 2,sim(Q, d) = ˆe(Q)⊤ˆe(d).(10)
All knowledge sentence embeddings are indexed using
FAISSlibrary [Johnsonet al., 2019 ]to enable efficient ap-
proximate nearest-neighbor search. Rather than selecting a
fixed top-k, we perform range search with a predefined simi-
larity thresholdτ, retaining only knowledge sentences whose
relevance scores exceedτ:
K∗(Q) ={d∈K|sim(Q, d)≥τ}.(11)The retrieved knowledge sentences are then concatenated
into a paragraph, which is subsequently provided to the An-
swer Generator for knowledge-grounded quality reasoning.
B.4 Answer Generator
The prompt used in the final Answer Generator is as follows.
For a single visual input:
Answer Generator Prompt
Now you will receive an image/video:
[visual token].
Here is the metadata for the image/video:[K∗
meta].
Here is the localization description of objects in the
image/video for reference:[K loc]orNULL.
Here is the global quality summary of the im-
age/video for reference:[K∗
globalQ ].
Here is the local quality description of the im-
age/video for reference:[K∗
localQ].
You are performing a visual quality understanding
task. Here is the question:
[question]
Please combine what you observe in the image/video
with the reference descriptions to make your judg-
ment. In your response, act as if your conclusion
is drawn solely from the visual content itself, rather
than from the reference descriptions.
Pay special attention toR[focus], as well as both
global and local distortions in the image/video,
including blur, noise, compression artifacts, color in-
consistencies, and motion instability. Your evaluation
should reflect how these distortions affect the overall
perceptual quality.
First, output your reasoning process enclosed in
<think>and</think>tags. Then, output
the selected option letter (if applicable) directly in
<answer>and</answer>tags.
For a pair of visual inputs:
Answer Generator Prompt
Now you will receive the first image/video:
[visual token] 1, the second image/video:
[visual token] 2.
Here is the metadata about the first image/video:
[K∗
meta]1.
Here is the metadata about the second image/video:
[K∗
meta]2.
This is the localization description of objects in the
the first image/video for reference:[K∗
loc]1orNULL.
This is the localization description of objects in the

second image/video for reference:[K∗
loc]2orNULL.
This is the global quality summary of the first im-
age/video for reference:[K∗
globalQ ]1.
This is the global quality summary of the second im-
age/video for reference:[K∗
globalQ ]2.
This is the local quality description of the two
images/videos for reference:[K∗
localQ].
You are performing a visual quality understanding
task. Here is the question:
[question]
Please combine what you observe in the image/video
with the reference descriptions to make your judg-
ment. In your response, act as if your conclusion
is drawn solely from the visual content itself, rather
than from the reference descriptions.
Pay special attention toR[focus], as well as both
global and local distortions in the image/video,
including blur, noise, compression artifacts, color in-
consistencies, and motion instability. Your evaluation
should reflect how these distortions affect the overall
perceptual quality.
First, output your reasoning process enclosed in
<think>and</think>tags. Then, output
the selected option letter (if applicable) directly in
<answer>and</answer>tags.
C More Visualizations of Performance
Figures 4 and 5 show additional examples of the visual qual-
ity understanding ability of our QualiRAG.

How severe is compression artifacts on the cat?A. NoneB. StrongC. Weak
Query Organizer
Source AugmenterQuery 1: MetadataQuery 2: LocalizationQuery 3: GlobalQualityQuery 4: localQuality
-  Resolution: 500x500- Frame Rate: -- Duration: -- Bitrate: -- Bitrate_trend: -Object 0 is a cat located at coordinates [154, 174] with dimensions 85x195.The image shows good overall quality with clear structure and texture, especially in the cat’s fur and the tree bark. Colors appear natural subject from the background...The compression artifacts on the cat are moderate. There is noticeable blurring and softening of fine details, particularly in the fur texture and the edges...Describe the severity of compression artifacts on the cat.Response1: ...Response2: ...Response3: ...Response4: ...
Source SelectorQuestion
<think> Upon examining the image, the cat's fur does appear slightly blurred, and the edges of the cat's body are not as sharply defined as they should be. There are subtle signs of blockiness or color bleeding, particularly around the hind legs and tail. These observations suggest that the compression artifacts are present but not extremely severe. Considering the options provided: Weak - This would imply minor compression artifacts, which is consistent with the observations. Therefore, the severity of compression artifacts on the cat is weak. </think><answer>C</answer>
Base Model(w/oRAG)Spatial subjectMultiple Description Aggregation
<think>In this image, the cat appears clear and detailed without any noticeable distortion or blurring. Therefore, the compression artifacts, if any, are not severe.</think><answer>A</answer>UserInput
AnswerGenerator
Is the coin in the image totally clear, partly clear, or totally blurred?A. Partly clearB. Totally blurredC. Totally clear
StructuredRequest:  {"subject": coin,  ”dimension":clarity,”scope": spatial, "focus": the area of the image containing the coin.}
Query Organizer
Source AugmenterQuery 1: MetadataQuery 2: LocalizationQuery 3: GlobalQualityQuery 4: localQuality
- Resolution: 1024x768- Frame Rate: -- Duration: -- Bitrate: -- Bitrate_trend: -Object 0 is a coin located at coordinates [469, 248] with dimensions 305x299.The clarity of the coin is high, with clear details of the texture and embossed patterns…The coin in the image has moderate clarity. The central design, featuring two leaping antelopes, is visible but shows some blurring and loss of fine detail…Response1: ...Response2: ...Response3: ...Response4: ...
Source SelectorQuestion
AnswerGenerator<think> The coin's design, featuring two leaping antelopes, is visible but shows some blurring and loss of fine detail. The text around the edge, "SHILINGI MIA MOJA" and "100," is legible but not perfectly sharp, with some letters appearing slightly softened or indistinct. The overall surface texture and metallic sheen are discernible, but the image lacks high-definition crispness. This indicates that the coin is not totally clear, nor is it totally blurred; it falls between the two, making it partly clear. </think><answer>A</answer>
Base Model(w/oRAG)Spatial subjectMultiple Description Aggregation
<think>The coin in the image is clearly visible with all its details, including the text and the image of the animals, being distinct and sharp. There is no blurriness or obstruction that would make it difficult to see the coin's features. Therefore, the coin is totally clear.</think><answer>C</answer>UserInput
Describe the clarity of the coin.
Is there any noise problem in the image?A. NoB. Yes
Query Organizer
Source AugmenterQuery 1: MetadataQuery 3: GlobalQualityQuery 4: localQuality
-  Resolution: 1024x768- Frame Rate: -- Duration: -- Bitrate: -- Bitrate_trend: -The image has a high level of noise, with noticeable graininess.The image exhibits noticeable digital noise, particularly in the darker areas and along high-contrast edges. This manifests as a grainy texture and a loss of fine detail, which is most apparent in the black wetsuit and the surrounding blue water. The noise appears as a subtle, speckled pattern…Describe the noise in the image.Response1: ...Response2: ...Response3: ...Response4: ...
Source SelectorQuestion
<think>The image shows a person in a scuba diving suit underwater. The overall quality appears to be low, with blurred and softened details. Upon closer inspection, there is a noticeable grainy texture, especially in the darker areas of the image, such as the wetsuit and the background. This graininess is consistent with digital noise. The reference descriptions mention the presence of visible grain or digital noise, especially in darker areas, which aligns with my observation. The noise reduces the clarity and detail of the image, making it less sharp and more difficult to discern finer details. </think><answer>B</answer>
Base Model(w/oRAG)Multiple Description Aggregation
<think>In this image, the details of the person's face and the surrounding environment appear clear and free of such distortions. The colors and textures are smooth and consistent, with no visible signs of noise or artifacts. Therefore, based on the analysis, there is no noise problem in the image. </think><answer>A</answer>UserInput
AnswerGenerator
StructuredRequest:  {"subject": none,  ”dimension": noise,”scope": spatial, "focus": entire image.}StructuredRequest:  {"subject": cat,  ”dimension":compression artifacts,”scope": spatial, "focus": the entire cat.}𝑸𝐥𝐨𝐜𝐚𝐥:
𝑸𝐥𝐨𝐜𝐚𝐥:
𝑸𝐥𝐨𝐜𝐚𝐥:Figure 4: Examples of image quality understanding on Q-Bench.

Question: How is the stuttering of the second video compared to the first one?A. More stutteringB. About the sameC. Much smootherD. Smoother…
StructuredRequest:  {"subject": none,  ”dimension":temporal, ”scope": temporal, "focus": entire video sequence in both videos.}
Query Organizer
Source AugmenterQuery 1: MetadataQuery 3: GlobalQualityQuery 4: localQuality
- Resolution: 1280x720- Frame Rate: 29.97 fps- Duration: 9.01 sec- Bitrate: 2284.1 kbps- Bitrate_trend: IncreasingThe video has moderate perceptual quality due to motion blur, uneven lighting, and low sharpness. While the colors and structure are somewhat preserved...The first video has smoother temporal flow, with consistent camera movement and a continuous. The second video appears more choppy, with abrupt cuts...Compare the temporal smoothness between the two videos.Response1: ...Response2: ...Response3: ...Response4: ...
Source SelectorQuestion
AnswerGenerator<think>The first video is described as having a smoother temporal flow with consistent camera movement and a continuous sequence of actions. The second video, however, is noted to have abrupt cuts and a less fluid progression between scenes. This suggests that the second video has more stuttering or choppiness compared to the first one. Additionally, The first video has moderate perceptual quality with several noticeable artifacts and distortions, while the second video also has moderate quality but with more pronounced issues such as motion blur and soft focus. These factors contribute to the perception of stuttering in the second video. Considering these observations, the second video exhibits more stuttering compared to the first one.</think><answer>A</answer>
Base Model(w/oRAG)Multiple Description Aggregation
<think>The first video shows a person with purple hair, and the second video shows a wedding ceremony. The first video has a noticeable stuttering effect, while the second video appears to be smoother and more continuous. The stuttering in the first video is more pronounced, indicating a lower video quality compared to the second video.</think> <answer>D</answer>UserInputQuestion: How is the clarity of the person kicking the ball in this video?A. Average B. Below average C. Very poor D. Good…StructuredRequest:   {"subject": person kicking the ball,  ”dimension":clarity,”scope": spatial, "focus": the person in motion while kicking the ball.}
Query Organizer
Source AugmenterQuery 1: MetadataQuery 2: LocalizationQuery 3: GlobalQualityQuery 4: localQuality
- Resolution: 320x240- Frame Rate: 29.97 fps- Duration: 10.01 sec- Bitrate: 244.3 kbps- Bitrate_trend: Increasing.… Frame 2: Object 0 is a person kicking the ball located at coordinates [265, 32] with dimensions 54x139.…The video likely originates from a low-resolution camera with poor lighting conditions, combined with high compression....The clarity of the person kicking the ball is low. The image is blurry and pixelated, making it difficult to discern fine details ...…Describe the clarity of the person kicking the ball.Response1: ...Response2: ...Response3: ...Response4: ...Source SelectorQuestion
<think>The video exhibits several indicators of low perceptual quality. The person kicking the ball is blurry and pixelated, making it difficult to discern fine details of their facial features, clothing, or the exact mechanics of the kick. The motion blur suggests the person was moving quickly when the photo was taken. Additionally, the overall image is grainy and lacks crispness, with a general lack of detail in both static and moving elements. The "Expert" and "Village" text overlays are also blurry and poorly rendered, further indicating low-quality encoding or resolution. These factors collectively result in a visually degraded and low-fidelity video.</think><answer>C</answer>
Base Model(w/oRAG)Spatial subjectMultiple Description Aggregation
<think>The video shows a person kicking a ball on a grassy field. The person is clearly visible, and the details of their actions, such as the motion of kicking the ball, are discernible. The image quality is sufficient to observe the actions and movements without any significant blur or distortion.</think><answer>D</answer>UserInput
Question: How realistic does the mountain goat appear in this video?A. Very poorB. GoodC. AverageD. Poor…
StructuredRequest:  {"subject": mountain goat,  ”dimension":realism,”scope": spatial, "focus": The mountain goat within the video.}
Query Organizer
Source AugmenterQuery 1: MetadataQuery 2: LocalizationQuery 3: GlobalQualityQuery 4: localQuality
- Resolution: 768x432- Frame Rate: 8.0 fps- Duration: 3.0 sec- Bitrate: 1137.6 kbps- Bitrate_trend: unknown… Frame 2: Object 0 is a mountain goat located at coordinates [246, 53] with dimensions 371x264…The video frame exhibits exceptional perceptual quality, likely due to advanced rendering techniques. The sheep’s wool texture is ....The images appear to be digitally generated or heavily edited, and the animal does not look like a real, living creature....…Response1: ...Response2: ...Response3: ...Response4: ...Source SelectorQuestion
AnswerGenerator<think>The video frames provided show a digitally generated or heavily edited image of a sheep, not a mountain goat. The sheep's wool appears unnaturally thick and sculpted, lacking the natural, fluffy texture of real sheep wool. The water stream coming from the sheep's mouth is also unrealistic, appearing as a digital effect rather than a natural stream. The overall texture and lighting have a synthetic look, further indicating that the image is not a photograph of a real animal. Therefore, the perceptual quality of the animal in the video is low, and it does not appear realistic.</think><answer>A</answer>
Base Model(w/oRAG)Spatial subjectMultiple Description Aggregation
<think>The video shows a mountain goat with a realistic appearance, standing in a natural setting with a flowing stream and green hills in the background. The goat's fur, horns, and overall posture are depicted with attention to detail, making it look quite realistic.</think><answer>B</answer>UserInput
Describe the realism of the mountain goat.
- Resolution: 640x360- Frame Rate: 24.0 fps- Duration: 9.0 sec- Bitrate: 441.7 kbps- Bitrate_trend: Constant lowThe video has moderate quality with acceptable color reproduction and minimal noise, but it suffers from low sharpness, motion blur, and soft focus...…
FirstVideo
SecondVideo
FirstVideoSecondVideoSecondVideoFirstVideoAnswerGenerator
𝑸𝐥𝐨𝐜𝐚𝐥:
𝑸𝐥𝐨𝐜𝐚𝐥:
𝑸𝐥𝐨𝐜𝐚𝐥:Figure 5: Examples of video quality understanding on Q-Bench-Video.