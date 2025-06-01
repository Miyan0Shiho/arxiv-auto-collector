# Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models

**Authors**: Yifan Jia, Kailin Jiang, Yuyang Liang, Qihan Ren, Yi Xin, Rui Yang, Fenze Feng, Mingcai Chen, Hengyang Lu, Haozhe Wang, Xiaoye Qu, Dongrui Liu, Lizhen Cui, Yuntao Du

**Published**: 2025-05-26 04:39:30

**PDF URL**: [http://arxiv.org/pdf/2505.19509v1](http://arxiv.org/pdf/2505.19509v1)

## Abstract
Large Multimodal Models(LMMs) face notable challenges when encountering
multimodal knowledge conflicts, particularly under retrieval-augmented
generation(RAG) frameworks where the contextual information from external
sources may contradict the model's internal parametric knowledge, leading to
unreliable outputs. However, existing benchmarks fail to reflect such realistic
conflict scenarios. Most focus solely on intra-memory conflicts, while
context-memory and inter-context conflicts remain largely investigated.
Furthermore, commonly used factual knowledge-based evaluations are often
overlooked, and existing datasets lack a thorough investigation into conflict
detection capabilities. To bridge this gap, we propose MMKC-Bench, a benchmark
designed to evaluate factual knowledge conflicts in both context-memory and
inter-context scenarios. MMKC-Bench encompasses three types of multimodal
knowledge conflicts and includes 1,573 knowledge instances and 3,381 images
across 23 broad types, collected through automated pipelines with human
verification. We evaluate three representative series of LMMs on both model
behavior analysis and conflict detection tasks. Our findings show that while
current LMMs are capable of recognizing knowledge conflicts, they tend to favor
internal parametric knowledge over external evidence. We hope MMKC-Bench will
foster further research in multimodal knowledge conflict and enhance the
development of multimodal RAG systems. The source code is available at
https://github.com/MLLMKCBENCH/MLLMKC.

## Full Text


<!-- PDF content starts -->

arXiv:2505.19509v1  [cs.LG]  26 May 2025Benchmarking Multimodal Knowledge Conflict for
Large Multimodal Models
Yifan Jia1∗, Kailin Jiang2*, Yuyang Liang1, Qihan Ren3, Yi Xin4,
Rui Yang1, Fenze Feng1, Mingcai Chen5, Hengyang Lu6, Haozhe Wang7,
Xiaoye Qu8, Dongrui Liu8, Lizhen Cui1, Yuntao Du1†
1Joint SDU-NTU Centre for Artificial Intelligence Research&School of Software , Shandong University
2University of Science and Technology of China3Shanghai Jiaotong University4Nanjing University,
5Nanjing University of Posts and Telecommunications6Jiangnan University
7The Hong Kong University of Science and Technology8Shanghai AI Laboratory
Abstract
Large Multimodal Models (LMMs) face notable challenges when encountering
multimodal knowledge conflicts, particularly under retrieval-augmented genera-
tion (RAG) frameworks, where the contextual information from external sources
may contradict the model’s internal parametric knowledge, leading to unreliable
outputs. However, existing benchmarks fail to reflect such realistic conflict sce-
narios. Most focus solely on intra-memory conflicts, while context-memory and
inter-context conflicts remain largely investigated. Furthermore, commonly used
factual knowledge-based evaluations are often overlooked, and existing datasets
lack a thorough investigation into conflict detection capabilities. To bridge this
gap, we propose MMKC-Bench , a benchmark designed to evaluate factual knowl-
edge conflicts in both context-memory and inter-context scenarios. MMKC-Bench
encompasses three types of multimodal knowledge conflicts and includes 1,573
knowledge instances and 3,381 images across 23 broad types, collected through
automated pipelines with human verification. We evaluate three representative
series of LMMs on both model behavior analysis and conflict detection tasks. Our
findings show that while current LMMs are capable of recognizing knowledge
conflicts, they tend to favor internal parametric knowledge over external evidence.
We hope MMKC-Bench will foster further research in multimodal knowledge
conflict and enhance the development of multimodal RAG systems. The source
code is available at https://github.com/MLLMKCBENCH/MLLMKC .
1 Introduction
The rapid advancement of large multimodal models (LMMs) and large language models (LLMs) has
led to remarkable performance across a wide range of multimodal understanding, generation, and
reasoning tasks [ 1,2,3,4,5]. Despite their impressive capabilities, static LMMs and LLMs often
suffer from limitations such as outdated or incorrect knowledge and hallucinations. To address these
issues, retrieval-augmented generation (RAG) techniques have been introduced [ 6,7], which enhance
model outputs by incorporating up-to-date information from external sources. However, this paradigm
introduces the challenge of knowledge conflict, where the retrieved contextual knowledge may
contradict the model’s internal (parametric) knowledge [ 8]. Recent studies have demonstrated that
such conflicts can undermine the trustworthiness and reliability of model predictions [ 9], highlighting
the need for a deeper understanding of model behavior under conflicting knowledge scenarios. To
∗Equal contribution
†Corresponding author
Preprint.

Original 
Knowledge
This image  is the Empire  
State  Building.
Empire  State  Building is 
an Australian  landmark 
building
This image  is the 
White  house . 
White  house  is an 
Australian  landmark  
building
Conflict 
Knowledge 1
Evaluation 
QuestionThis image  is the The 
Eiffel  Tower . 
The Eiffel  Tower  is an 
Australian  landmark 
building
Question :
What  is the building  
in the image?
Conflict 
Knowledge 2Entity Recognition Conflict Entity Knowledge Conflict Visual Semantic Conflict
The person  in the 
image  is Musk .  Musk  
was born in 1971.  
Musk  is an American . 
and Musk  is a 
businessman .
The person in the image  
is Musk .  Musk  was 
born in 1974. Musk  is 
an American . and Musk  
is a businessman .
The person in the image  
is Musk . Musk  was 
born in 1979. Musk  is 
an American,  and Musk  
is a businessman .
Question :
In Which  year was the 
person in the image  
born?
This is a gesture  
of rejection .
Question :
What  is the gesture  
in the image?This is a gesture  
of sarcasm
This is a gesture  
of OK.
Figure 1: Three types of multimodal knowledge conflict in MMKC-Bench. It is noted that the original
knowledge is shown to help understand what the conflict is, and is not contained in the dataset.
facilitate this, several benchmark datasets have been developed to study knowledge conflict both in
textual contexts [10, 11, 9] and in multimodal domains [12, 13, 14].
As outlined in the survey [ 8], knowledge conflicts in LLMs typically fall into three categories:
intra-memory conflicts, context-memory conflicts, and inter-context conflicts. Intra-memory conflict
arises from inconsistencies within the model’s own parametric knowledge, which are often introduced
during pretraining due to contradictory or noisy pretraining data. In contrast, context-memory and
inter-context conflicts typically emerge during inference. Context-memory conflict occurs when
external contextual information, such as user prompts or retrieved documents, contradicts the model’s
internal (parametric) knowledge. Inter-context conflict refers to inconsistencies among the external
contextual sources themselves. These latter forms of conflict are particularly relevant in RAG settings
and present unique challenges for ensuring the reliability and consistency of model outputs.
Several multimodal knowledge conflict datasets have been proposed to investigate a variety of per-
spectives, such as commonsense-based knowledge conflicts under the context-memory scenario [ 12],
cognition and perception conflicts under the intra-memory scenario by leveraging different model
capability to answer the same question (e.g., OCR vs. VQA)[ 13], and cross-modality conflicts in
intra-memory settings by framing the same question in both textual and multimodal formats[ 14].
While these efforts provide valuable insights, they fall short of capturing realistic conflict scenarios
within the RAG framework. First, existing datasets mainly focus on intra-memory conflicts, with
limited attention paid to context-memory and inter-context conflicts. Second, in practical scenarios,
external factual knowledge, such as identifying objects in an image or determining a person’s age, is
often necessary but remains underrepresented in current datasets. Lastly, current multimodal conflict
benchmarks are primarily designed to observe model behavior, rather than evaluating whether models
can effectively detect and acknowledge the presence of knowledge conflict.
To address these gaps, we propose MMKC-Bench, a multimodal knowledge conflict benchmark
aimed at evaluating factual knowledge conflicts under both context-memory and inter-context
scenarios . As illustrated in Fig. 1, MMKC-Bench focuses on three representative types of multimodal
conflicts: entity recognition conflict, entity knowledge conflict, and visual semantic conflict. The
entity recognition and visual semantic conflicts target inconsistencies in entity identification and
complex visual understanding, while the entity knowledge conflicts emphasize factual inconsistencies
related to specific attributes or quantitative data. In addition to analyzing model behavior in the
presence of conflict, MMKC-Bench also investigates whether models can detect and perceive conflicts
at both coarse-grained (given whole evidence) and fine-grained (given a subset of the evidence) levels.
To construct the benchmark, we first curate original multimodal knowledge from a variety of sources,
including Wikipedia, Google Images, and existing datasets. We then employ large language models
(LLMs) to generate conflicting knowledge through counterfactual editing, which involves modifying
the entity name, semantic content, or entity-related factual information. Based on the constructed
knowledge pairs, we use LLMs to generate both multiple-choice and open-ended evaluation questions,
2

along with candidate answers. All generated questions, answers, and the preceding data construction
steps undergo rigorous human verification, with samples being filtered or revised as needed to ensure
quality and accuracy. The final MMKC-Bench dataset comprises 1,573 knowledge instances and
3,381 images, spanning 23 broad knowledge categories.
We conduct experiments on nine representative LMMs from three prominent model families known
for their strong performance in multi-image reasoning: Qwen2.5-VL [ 1], InternVL3 [ 15], and GPT-4o
mini [ 16]. Our evaluation covers both model behavior analysis and conflict detection tasks. The
experimental findings reveal several key insights: (1) LMMs tend to rely more heavily on internal
(parametric) knowledge than on external evidence, a behavior that contrasts with previously observed
trends in LLMs [ 11,10]; (2) LMMs are more sensitive to knowledge-level conflicts (e.g., entity
knowledge) than to recognition-level conflicts (e.g., entity identification); (3) Larger models show a
stronger promoting effect across all conflict types; (4) LMMs are capable of accurately identifying
the presence of conflict in both coarse-grained and fine-grained scenarios.
To sum up, the contribution of this work is summarized as follows,
•We propose MMKC-Bench, a multimodal knowledge conflict benchmark focusing on factual
knowledge conflict under both context-memory and inter-context scenarios.
•We propose a novel pipeline to construct the benchmark that collects original knowledge,
generates conflict knowledge and produce evaluation with two question formats.
•Extensive experiments with various models under both context-memory and inter-context for
behavior understanding and conflict detection are conducted, revealing several characteristics
of existing LMMs.
2 Related work
2.1 Large Multimodal Model
The development of LMMs has significantly advanced the integration of visual and textual infor-
mation, enabling more sophisticated cross-modal understanding and reasoning. Modern LMMs
are typically composed of three core components: a language encoder, a vision encoder, and cross-
modality alignment modules [ 17]. The language encoder is usually based on large language models
such as LLaMA [ 18,19] and Qwen [ 20,20], while the vision encoder often adopts architectures like
ViT [ 21]. Cross-modality alignment modules play a crucial role in integrating visual features into
textual representations, allowing the language encoder to effectively interpret visual signals. Based
on this architecture, many state-of-the-art LMMs have been developed, including Qwen2.5-VL [ 1],
InternVL2.5 [ 22], LLaV A [ 3], and LLaV A-OneVision [ 23]. In parallel, a range of training strategies
have been introduced to strengthen cross-modal alignment. For instance, Qwen-VL integrates a visual
receptor and employs a three-stage training pipeline to boost performance. InternVL adopts a native
multimodal pre-training paradigm, jointly learning visual and linguistic knowledge from diverse
sources. Meanwhile, LLaV A and LLaV A-Next enhance vision-language integration with improved
visual grounding and reasoning capabilities. These advancements have led to impressive performance
across a wide spectrum of multimodal tasks [ 24,25], highlighting the rapid progress and evolving
design of LMM architectures and training methodologies. Besides, some LLM Agents [ 26,27,28]
utilizes visual tools to solve multimodal tasks, which is beyond our scope.
2.2 Knowledge Conflict in LLMs
Knowledge conflict refers to discrepancies between contextual inputs and a model’s internal paramet-
ric knowledge [ 29,30]. Such conflicts often arise from temporal misalignment or misinformation in
training corpora. In the context of LLMs, knowledge conflicts are typically categorized into three
types: context-memory conflicts, inter-context conflicts, and intra-memory conflicts. With the rise of
retrieval-augmented generation (RAG), research has largely focused on context-memory conflicts,
while other conflicts have received comparatively less attention. Most existing datasets are syntheti-
cally constructed, either via entity substitution [ 31,9] or LLM-generated contradictions [ 32]. Among
them, ConflictBank [ 11] is currently the largest benchmark, though its design simplifies conflicts
into binary factual mismatches, limiting its reflection of real-world complexity. More recent datasets,
such as WikiConflict [ 10], attempt to address this limitation by extracting natural contradictions
from Wikipedia articles. Beyond dataset construction, several studies have also explored conflict
detection [9, 33] and conflict mitigation [34, 35, 36], improving the reliability of LLM.
3

2.3 Knowledge Conflict in LMMs
Several studies have also explored knowledge conflicts in LMMs from different perspectives. For
example, existing benchmarks have examined commonsense-based knowledge conflicts in context-
memory scenarios [ 12], cognition-perception conflicts in intra-memory settings by leveraging differ-
ent model capabilities to answer the same question (e.g., OCR vs. VQA) [ 13], and cross-modality
conflicts within intra-memory scenarios by comparing model responses to the same question framed
in both textual and multimodal formats [ 14]. While these efforts offer valuable insights, they fall
short in capturing realistic conflict scenarios within the RAG framework. Specifically, current bench-
marks predominantly focus on intra-memory conflicts, with limited attention to context-memory
and inter-context conflicts. Moreover, in practical applications, resolving external factual knowledge
conflicts, such as identifying objects in images or estimating a person’s age, is often crucial, yet
remains underrepresented in existing datasets. Finally, most current multimodal conflict benchmarks
are primarily designed to observe model behavior, lacking an explicit focus on evaluating whether
models can detect and acknowledge the presence of knowledge conflicts.
3 Problem Definition
3.1 Original and Conflict Knowledge Representation
MMKC-Bench focuses on factual knowledge based knowledge conflict and encompasses three types:
entity recognition conflict, entity knowledge conflict, and visual semantic conflict. For these types,
each original piece of original knowledge is represented in a unified format k= (i, d), where idenotes
an image of the entity or semantic action, and dis the corresponding textual description. To construct
conflicting instances, the original knowledge is modified to form kc. It takes the form kc= (ic, dc),
where icis another image of the same entity or action, and dcis a conflicting description. The
examples of each type are shown in Fig 1.
3.2 Multimodal Knowledge Conflict Types
To comprehensively reflect real-world scenarios, MMKC-Bench includes three types of multimodal
knowledge conflicts: entity recognition conflict, entity knowledge conflict, and visual semantic
conflict.
Entity Recognition Conflict simulates cognitive inconsistencies where different sources identify the
same entity differently. This is achieved by keeping the entity image unchanged while replacing the
entity name in the description with that of another entity of the same type. For example, as shown in
Fig 1, describing the Empire State Building image using “Eiffel Tower” or “White House” as the
entity name.
Visual Semantic Conflict addresses inconsistencies in interpreting complex visual semantics, such
as gestures, body actions, or symbolic cues. Here, the semantic meaning associated with an action is
replaced with that of another action of the same type. For instance, as shown in Fig 1, changing the
meaning of an “OK” gesture to “rejection” or “sarcasm.”
Entity Knowledge Conflict centers on factual discrepancies surrounding entity attributes like
nationality, occupation, or birth year. This is simulated by substituting the tail entity in a factual triple
with another entity of the same type. For example, as shown in Fig 1, altering Elon Musk’s birth year
from the correct value to 1974 or 1979.
4 MMKC-Bench
In this section, we present the pipeline for constructing MMKC-Bench, a QA-based benchmark
comprising 1,573 instances that encompass three types of multimodal knowledge conflicts, as
illustrated in Fig 2.
4.1 Original Knowledge Collection
We begin with original knowledge collection by first listing candidate entity types or visual semantics.
Then, we collect the corresponding images and descriptions.
4

Figure 2: The construction pipeline of MMKC-Bench.
Figure 3: The data types of MMKC-Bench.For entity recognition and entity knowledge conflicts,
we manually define multiple candidate visual entity
types (e.g., person, building). For each type, we use a
large language model (LLM) to generate a list of the
most prominent entities (e.g., Messi under the “per-
son” category). Once the entity list is obtained, we
crawl their images from Google and retrieve entity de-
scriptions from Wikipedia summary dumps3, which
are then summarized by an LLM to retain essential
information. For visual semantic conflicts, we follow
prior work on knowledge editing with visual semantic
modifications [ 37], focusing on four categories of vi-
sual semantic knowledge: everyday gestures, human
body actions, human emotions, and symbol identifica-
tion. Since these types are already included in MMKC-
Bench, we directly obtain the original visual seman-
tic knowledge, consisting of paired images and their
meanings, from the MMKC-Bench dataset, rather than
collecting them manually.
4.2 Conflict Knowledge Generation
Considering the multimodal nature of LMMs, we generate conflict knowledge by deliberately
introducing misalignments across modalities, with the assistance of large language models (LLMs).
For entity recognition, visual semantic, and entity knowledge conflicts, we retain the original image
while modifying the textual component. Specifically, we replace the entity name, the meaning of
the action, or the tail entity in a factual triple with another instance of the same type. For example,
as illustrated in Fig 1, the name “Empire State Building” is replaced with “Eiffel Tower” or “White
House”, and Elon Musk’s birth year is altered to 1974 or 1979.
To simulate both context-memory conflicts and inter-context conflicts, we generate two conflicting
versions for each piece of original knowledge following the above procedures. In the context-memory
setting, one conflicting version is randomly selected as internal evidence. In the inter-context setting,
both conflicting versions are provided as internal evidence.
4.3 Evaluation Question Generation
We adopt a visual question answering (VQA) format to construct evaluation questions and answers,
leveraging LLMs for automatic generation. The specific prompts used are provided in the appendix.
We consider two types of questions: multiple-choice questions (MCQs) and open-ended questions
(OQAs). For entity recognition and semantic recognition conflicts, the questions focus on identifying
the entity name or interpreting the semantic meaning depicted in the image. For entity knowledge
conflicts, the questions target fine-grained factual knowledge, such as querying a person’s occupation
or age. In MCQs, each question includes four answer options: one answer within the models’ internal
knowledge, two answers from the conflicting knowledge variants, and one unrelated distractor option.
3[https://dumps.wikimedia.org/
5

4.4 Human Verification and Benchmark Statistics
During the construction of the benchmark, we conducted multiple rounds of manual collection,
review, and filtering to ensure data quality. In the original knowledge collection stage, all images
associated with each entity, semantic action, and piece of knowledge were manually reviewed to
ensure their accuracy and relevance. Furthermore, following counterfactual editing and question
generation, we performed additional manual verification, filtering out inappropriate samples, revising
ambiguous or ill-formed questions, and correcting incorrect answers. We believe that this extensive
human effort has been instrumental in ensuring the reliability and quality of the benchmark.
Table 1: The statistics of MMKC-Bench.
#Types #Instances #Images
Visual Entity Conflict 13 757 2,271
Entity Knowledge Conflict 6 669 669
Visual Semantic Conflict 4 147 441The statistical information of MMKC-Bench is
shown in Tab 1. As we can see, MMKC-Bench
encompasses three types of conflict knowledge,
containing 1,573 pieces of knowledge and 3,381
images. These knowledge spans 23 fine-grained
types, highlighting the diversity of MMKC-Bench.
5 Experiment
5.1 Setup
Models In multimodal knowledge conflict scenarios, the model input consists of multiple interleaved
images and texts. Therefore, we select LMMs that perform well in multi-image understanding.
Specifically, we conduct a comprehensive evaluation on 9 LMMs across 3 model series, with sizes
ranging from 3B to 72B. The selected models include: Qwen2.5-VL (3B, 7B, 32B, 72B) [ 1],
InternVL3 (8B, 14B, 38B, 78B) [15], and GPT-4o mini [16].
Settings We consider two conflict-related tasks: conflict behavior analysis and conflict detection.
The former investigates how models behave under conflicting scenarios, while the latter evaluates
whether models can correctly detect the presence of conflict.
For conflict behavior analysis, we consider two types of conflict scenarios: context-memory conflict
and inter-context conflict. In context-memory conflict, one piece of conflicting external evidence,
composed of an image and associated text, is provided as an in-context example. In inter-context
conflict, two conflicting pieces of evidence about the same knowledge are provided as in-context
examples. The model is then required to answer an evaluation question based on this context.
For conflict detection, we explore both coarse-grained and fine-grained conflict detection. In the
coarse setting, a full piece of evidence (either conflicting or non-conflicting) is provided in context,
and the model must determine whether a conflict exists by answering “yes” or “no”. Following
previous work [ 9], the fine-grained setting involves providing only one single sentence, which is the
subset of full evidence, and the model must again judge whether a conflict is present.
Evaluation Metrics For conflict behavior analysis, we assess how conflicting contexts influence the
model’s answers to QA pairs. Each model prediction under a conflict scenario is categorized into
one of three types: (1) consistent with the model’s answer in the non-conflict setting, (2) consistent
with the external conflicting evidence, and (3) inconsistent with both, referred to as an irrelevant
answer. To enable this, we first perform QA under a non-conflict setting to establish the model’s
internal knowledge. We then compute three ratios: Original Answer Ratio (OAR), Counter Answer
Ratio (CAR), and Irrelevant Answer Ratio (IAO), with OAR + CAR + IAO = 1 across the dataset.
For conflict detection, we treat this as a binary classification task. If a knowledge conflict exists, the
model should output “yes”; otherwise, it should output “no”, in both coarse and fine-grained settings.
Accordingly, we report the detection accuracy as the evaluation metric.
5.2 Model Behavior Analysis
The results under both context-memory and inter-context conflict scenarios, using multiple-choice
and open-ended question formats, are presented in Table 2, Table 3, Fig. 4, and Fig. 5. Based on these
results, we draw the following observations:
1) LMMs are more receptive to internal knowledge than to external evidence. As shown in
Table 2 and Table 3, under context-memory conflicts, the average OAR exceeds CAR in all cases
6

Table 2: Results of both context-memory and inter-context conflicts on MMKC-Bench using the
multiple-choice question format.
Qwen2.5-VL-7B InternVL3-8B GPT-4o mini
ER EK VS Avg. ER EK VS Avg. ER EK VS Avg.
Context-Memory Conflict
OAR 0.81 0.50 0.71 0.67 0.48 0.46 0.70 0.49 0.78 0.54 0.61 0.66
CAR 0.15 0.48 0.21 0.30 0.41 0.45 0.26 0.41 0.11 0.43 0.30 0.27
IAR 0.04 0.02 0.08 0.03 0.11 0.09 0.04 0.09 0.11 0.03 0.09 0.07
Inter-Context Conflict
OAR 0.87 0.51 0.72 0.70 0.41 0.47 0.66 0.46 0.76 0.47 0.57 0.62
CAR 0.10 0.48 0.25 0.28 0.53 0.51 0.30 0.50 0.19 0.50 0.40 0.34
IAR 0.03 0.01 0.03 0.02 0.07 0.02 0.05 0.04 0.05 0.03 0.03 0.04
Table 3: Results of both context-memory and inter-context conflicts on MMKC-Bench using the
open-ended question answering format.
Qwen2.5-VL-7B InternVL3-8B GPT-4o mini
ER EK VS Avg. ER EK VS Avg. ER EK VS Avg.
Context-Memory Conflict
OAR 0.66 0.26 0.13 0.44 0.60 0.27 0.27 0.43 0.76 0.36 0.45 0.56
CAR 0.20 0.62 0.48 0.40 0.19 0.53 0.15 0.33 0.02 0.47 0.05 0.21
IAR 0.14 0.10 0.39 0.15 0.22 0.20 0.58 0.24 0.22 0.17 0.50 0.22
Inter-Context Conflict
OAR 0.65 0.14 0.05 0.38 0.46 0.15 0.18 0.30 0.82 0.37 0.37 0.58
CAR 0.21 0.76 0.77 0.50 0.40 0.72 0.52 0.54 0.06 0.47 0.07 0.24
IAR 0.14 0.10 0.18 0.12 0.14 0.14 0.30 0.15 0.12 0.16 0.56 0.18
(6 out of 6), indicating that LMMs tend to favor internal knowledge. Closed-source GPT-4o mini
shows consistent results with open-source models, suggesting that even advanced closed models are
insensitive to external evidence. This differs from LLMs, which have shown high receptiveness to
external knowledge [ 11,10]. One reason for this contrast is the difference in training data formats:
LLMs are typically trained on long text contexts involving multiple information sources, while LMMs
are mostly trained on isolated image-text pairs. This limits their exposure to multi-source contexts
and reduces their ability to integrate external information during inference.
This finding is important for designing multimodal RAG systems, as it reveals that LMMs may not
naturally leverage retrieved evidence and instead rely on parametric knowledge. Thus, improving
LMMs’ ability to incorporate external information is important, which may require innovations in
training paradigms and model architecture.
2) LMMs are more sensitive to knowledge-related conflicts and less sensitive to recognition-
based conflicts. We group the three conflict types into recognition-based (entity recognition, visual se-
mantics) and knowledge-related (entity knowledge). LMMs show lower OARs on knowledge-related
conflicts than of recognition-based conflicts, indicating greater sensitivity to factual inconsistencies.
For example, entity recognition conflicts yield an OAR as low as 0.26 on Qwen2.5-VL-7B. While
entity recognition conflicts often show the highest OARs, suggesting LMMs more easily rely on
internal knowledge for perception tasks.
As shown in prior work [ 13,38], perception and cognition are core abilities of LMMs. Recognition-
based tasks rely on visual-text alignment (perception), whereas knowledge-related tasks involve
cognitive reasoning over facts. LMMs are mainly trained on perception tasks like VQA, grounding,
and captioning, with less exposure to cognitively demanding data. This imbalance results in stronger
perception than reasoning abilities. Therefore, LMMs lean on internal memory for recognition, but
may turn to external sources for knowledge-intensive tasks. This highlights the need to enrich training
data with cognitively challenging examples to strengthen LMM reasoning capabilities.
3) When provided with more external evidence, LMMs exhibit greater alignment with external
information, though the improvement remains limited. Compared to context-memory conflict
scenarios, models generally achieve higher CARs under inter-context conflicts, suggesting a slight
7

Figure 4: The results of Qwen2.5-VL with different model sizes under context-memory conflict with
multi-choice question format.
Figure 5: The results of Qwen2.5-VL with different model sizes under inter-context conflict with
multi-choice question format.
increase in reliance on external evidence. This is because, given more internal information, the model
output would be affected more. However, the overall improvement is limited: the largest increase in
CAR is 21% on average, while the smallest average improvement is only about -2%. These results
reaffirm that LMMs predominantly rely on their internal parametric knowledge, even when presented
with multiple external sources.
4) Larger models exhibit a stronger promoting effect across all conflict types. As illustrated in
Fig.4 and Fig.5, the Overall Agreement Rate (OAR) generally increases with model size within the
Qwen2.5-VL series. Specifically, the OAR improves progressively as the model scales from 3B to
7B, 13B, and 70B, reflecting gains across entity recognition conflict, entity knowledge conflict, and
visual semantic conflict. This trend suggests that larger models are more strongly influenced by their
internal knowledge. This enhanced capability may stem from exposure to more extensive training
data, enabling larger models to develop stronger mechanisms for resolving conflicts.
5) While performance differs between the two question formats, the overall trends remain
consistent. Under the two question formats, the models exhibit different performance levels. For
instance, in the open-ended question format, models tend to achieve higher IAR, suggesting that
the open-ended nature of the task introduces greater variability in the model outputs. Despite
these differences in absolute performance, the overall trend across both formats remains consistent,
demonstrating the robustness of the proposed benchmark across varying evaluation settings.
5.3 Conflict Detection Analysis
The results of both coarse-grained and fine-grained conflict detection are shown in Table 4. Based on
the results, we have the following findings:
1) LMMs can effectively identify the presence of knowledge conflicts and generally perform
better in recognizing conflicts under conflict scenarios than non-conflict scenarios. As shown in
Table 4, the average detection accuracy reaches 79%, 75%, and 76% for Qwen2.5-VL-7B, InternVL-
8B, and GPT-4o mini, respectively, indicating that LMMs are capable of reliably detecting the
existence of knowledge conflicts. Moreover, in most cases (5 out of 9), the detection accuracy
8

Table 4: Results of coarse-grained and fine-grained conflict detection on MMKC-Bench.
Coarse-Grained Detection Fine-Grained Detection
ER EK VS Avg. EK
Non-Conflict Conflict Avg. Non-Conflict Conflict Avg. Non-Conflict Conflict Avg. Non-conflict Conflict Avg.
Qwen2.5-VL-7B 0.92 0.87 0.89 0.89 0.51 0.70 0.67 0.89 0.78 0.79 0.76 0.65 0.71
InternVL3-8B 0.95 0.44 0.69 0.98 0.67 0.82 0.87 0.72 0.79 0.75 0.92 0.35 0.64
GPT-4o mini 0.73 0.88 0.80 0.66 0.76 0.71 0.63 0.82 0.73 0.76 0.82 0.61 0.72
under non-conflict scenarios is higher than that under conflict scenarios, suggesting that models
tend to detect more accurately when no knowledge conflict is present. Besides, it is also found that
open-source models have similar or even better performance than closed-source models, showing a
smaller gap between these models.
2) LMMs can effectively identify knowledge conflicts in both coarse-grained and fine-grained
scenarios. The detection accuracy under fine-grained scenarios is comparable to or slightly lower
than that under coarse-grained scenarios. These results indicate that LMMs are capable of recognizing
knowledge conflicts across both levels of granularity. However, the slightly lower performance in
fine-grained settings is consistent with observations in previous work [9].
5.4 Case Study
Two examples each of context-memory conflict and inter-context conflict involving entity recognition
and visual semantic conflicts are presented in Fig. 6. The answers without context reflect the models’
parametric (internal) knowledge, while the answers with context show the models’ behavior when
exposed to conflicting external evidence. As observed, across all models, the original answers
align with the ground truth. However, under context-memory conflict scenarios, most models tend
to rely on their internal knowledge, often ignoring the provided external evidence. When more
external evidence is introduced in the inter-context setting, models are more likely to refer to external
knowledge sources. Moreover, we also observe that under conflict scenarios, models may sometimes
produce answers that are inconsistent with both internal knowledge and external evidence, such as
“curvewaringsign” in the second example.
Figure 6: Case study of context-memory conflict and inter-context conflict involving entity recognition
and visual semantic conflicts.
6 Conclusion
In this paper, we propose MMKC-Bench, a multimodal knowledge conflict benchmark aimed at
analyzing factual conflicts in both context-memory and inter-context scenarios. Our benchmark
covers three types of multimodal knowledge conflicts and incorporates two distinct conflict settings.
Through extensive experiments on representative LMMs, we observe that most models are more
receptive to internal (parametric) knowledge and exhibit limited sensitivity to external conflicting
information. We hope our work will inspire further research in knowledge conflict resolution
and the development of multimodal retrieval-augmented generation (RAG) frameworks. As for
limitations, collecting real-world multimodal knowledge conflicts remains challenging. To address
this, we employ counterfactual editing to synthesize conflict instances, which inevitably introduces
a distribution gap between our benchmark and naturally occurring data. In the future, real-world
multimodal knowledge conflict benchmarks, such as [ 10], are needed to more accurately reflect
real-world scenarios and enhance the robustness of model evaluations.
9

References
[1]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang,
Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 ,
2025.
[2]Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong
Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning
for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition , pages 24185–24198, 2024.
[3]Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances
in neural information processing systems , 36:34892–34916, 2023.
[4]Can Cui, Yunsheng Ma, Xu Cao, Wenqian Ye, Yang Zhou, Kaizhao Liang, Jintai Chen, Juanwu
Lu, Zichong Yang, Kuei-Da Liao, et al. A survey on multimodal large language models for
autonomous driving. In Proceedings of the IEEE/CVF Winter Conference on Applications of
Computer Vision , pages 958–979, 2024.
[5]Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie
Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via
visual tool reinforcement learning. arXiv preprint arXiv:2505.08617 , 2025.
[6]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language
models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining , pages 6491–6501, 2024.
[7]Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. A survey of multimodal retrieval-
augmented generation. arXiv preprint arXiv:2504.08748 , 2025.
[8]Rongwu Xu, Zehan Qi, Zhijiang Guo, Cunxiang Wang, Hongru Wang, Yue Zhang, and Wei Xu.
Knowledge conflicts for llms: A survey. EMNLP , 2024.
[9]Yike Wang, Shangbin Feng, Heng Wang, Weijia Shi, Vidhisha Balachandran, Tianxing He, and
Yulia Tsvetkov. Resolving knowledge conflicts in large language models. COLM , 2024.
[10] Yufang Hou, Alessandra Pascale, Javier Carnerero-Cano, Tigran Tchrakian, Radu Marinescu,
Elizabeth Daly, Inkit Padhi, and Prasanna Sattigeri. Wikicontradict: A benchmark for evaluating
llms on real-world knowledge conflicts from wikipedia. Advances in Neural Information
Processing Systems , 37:109701–109747, 2024.
[11] Zhaochen Su, Jun Zhang, Xiaoye Qu, Tong Zhu, Yanshu Li, Jiashuo Sun, Juntao Li, Min Zhang,
and Yu Cheng. Conflictbank: A benchmark for evaluating the influence of knowledge conflicts
in llms. Advances in Neural Information Processing Systems , 37:103242–103268, 2024.
[12] Xiaoyuan Liu, Wenxuan Wang, Youliang Yuan, Jen-tse Huang, Qiuzhi Liu, Pinjia He, and
Zhaopeng Tu. Insight over sight? exploring the vision-knowledge conflicts in multimodal llms.
arXiv preprint arXiv:2410.08145 , 2024.
[13] Zirui Shao, Chuwei Luo, Zhaoqing Zhu, Hangdi Xing, Zhi Yu, Qi Zheng, and Jiajun Bu. Is
cognition consistent with perception? assessing and mitigating multimodal knowledge conflicts
in document understanding. arXiv preprint arXiv:2411.07722 , 2024.
[14] Tinghui Zhu, Qin Liu, Fei Wang, Zhengzhong Tu, and Muhao Chen. Unraveling cross-modality
knowledge conflicts in large vision-language models. arXiv preprint arXiv:2410.03659 , 2024.
[15] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan,
Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time
recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 , 2025.
[16] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark,
AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv
preprint arXiv:2410.21276 , 2024.
10

[17] Davide Caffagni, Federico Cocchi, Luca Barsellotti, Nicholas Moratelli, Sara Sarto, Lorenzo
Baraldi, Marcella Cornia, and Rita Cucchiara. The revolution of multimodal large language
models: a survey. ACL, 2024.
[18] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama
3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
[19] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei,
Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open
foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 , 2023.
[20] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint
arXiv:2412.15115 , 2024.
[21] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai,
Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al.
An image is worth 16x16 words: Transformers for image recognition at scale. ICLR , 2020.
[22] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shen-
glong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source
multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 ,
2024.
[23] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan
Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. Transactions on
Machine Learning Research , 2025.
[24] Jihyung Kil, Zheda Mai, Justin Lee, Arpita Chowdhury, Zihe Wang, Kerrie Cheng, Lemeng
Wang, Ye Liu, and Wei-Lun Harry Chao. Mllm-compbench: A comparative reasoning bench-
mark for multimodal llms. Advances in Neural Information Processing Systems , 37:28798–
28827, 2024.
[25] Jiaxing Huang and Jingyi Zhang. A survey on evaluation of multimodal large language models.
arXiv preprint arXiv:2408.15769 , 2024.
[26] Zhi Gao, Yuntao Du, Xintong Zhang, Xiaojian Ma, Wenjuan Han, Song-Chun Zhu, and Qing
Li. Clova: A closed-loop visual assistant with tool usage and update. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition , pages 13258–13268, 2024.
[27] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A
memory-augmented multimodal agent for video understanding. In European Conference on
Computer Vision , pages 75–92. Springer, 2024.
[28] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning
without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 14953–14962, 2023.
[29] Hung-Ting Chen, Michael JQ Zhang, and Eunsol Choi. Rich knowledge sources bring complex
knowledge conflicts: Recalibrating models to reflect conflicting evidence. EMNLP , 2022.
[30] Jian Xie, Kai Zhang, Jiangjie Chen, Renze Lou, and Yu Su. Adaptive chameleon or stubborn
sloth: Revealing the behavior of large language models in knowledge conflicts. In The Twelfth
International Conference on Learning Representations , 2023.
[31] Shayne Longpre, Kartik Perisetla, Anthony Chen, Nikhil Ramesh, Chris DuBois, and Sameer
Singh. Entity-based knowledge conflicts in question answering. EMNLP , 2021.
[32] Jiahao Ying, Yixin Cao, Kai Xiong, Yidong He, Long Cui, and Yongbin Liu. Intuitive or
dependent? investigating llms’ behavior style to conflicting prompts. ACL, 2024.
[33] Jierui Li, Vipul Raheja, and Dhruv Kumar. Contradoc: understanding self-contradictions in
documents with large language models. NAACL , 2024.
11

[34] Zorik Gekhman, Jonathan Herzig, Roee Aharoni, Chen Elkind, and Idan Szpektor. Trueteacher:
Learning factual consistency evaluation with large language models. EMNLP , 2023.
[35] Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, and Pengcheng He.
Dola: Decoding by contrasting layers improves factuality in large language models. ICLR ,
2024.
[36] Tsun-Hin Cheung and Kin-Man Lam. Factllama: Optimizing instruction-following language
models with external knowledge for automated fact-checking. In 2023 Asia Pacific Signal
and Information Processing Association Annual Summit and Conference (APSIPA ASC) , pages
846–853. IEEE, 2023.
[37] Yuntao Du, Kailin Jiang, Zhi Gao, Chenrui Shi, Zilong Zheng, Siyuan Qi, and Qing Li. Mmke-
bench: A multimodal editing benchmark for diverse visual knowledge. ICLR , 2025.
[38] Liang Chen, Yichi Zhang, Shuhuai Ren, Haozhe Zhao, Zefan Cai, Yuchi Wang, Peiyi Wang,
Xiangdi Meng, Tianyu Liu, and Baobao Chang. Pca-bench: Evaluating multimodal large
language models in perception-cognition-action chain. Findings of ACL , 2024.
12

G DBENCHMARK CONSTRUCTION
G.1 ORIGINAL KNOWLEDGE COLLECTION
In the process of collecting raw knowledge, we first select the most popular entities, then gather
the corresponding raw knowledge of these entities using Wikipedia. Then, we collect the images of
the entities to construct multimodal data, followed by building conflicting knowledge, and finally
generating corresponding evaluation questions.
For entity recognition tasks, we first identify visually grounded entity categories (e.g., building,
people) and then collect the most popular entities for each category using existing datasets or
generated by LLMs. Next, we retrieve the raw description for each entity from Wikipedia. To
ensure data quality, we apply an automated preliminary filtering process: excluding entities with
raw knowledge shorter than 30 words and leveraging Wikipedia’s API to prioritize the most popular
entities per category. Since raw knowledge may contain noise, logical inconsistencies, or fragmented
text, we employ LLMs to summarize it while preserving the original information. Subsequently, we
crawl images for each entity, manually inspect and remove low-quality visuals, and finally ensure a
minimum of three images per entity.
For entity knowledge, we extended the entity recognition data and constructed standardized knowledge
across two themes (individuals and brand logos) with three dimensions each. Specifically, for
individuals, we compiled knowledge on birth dates, nationalities, and occupations; for brand logos,
we organized data on establishment dates, founders, and primary products.
For visual semantic, we collected semantic knowledge across four categories: symbolic images,
common gestures, body movements, and facial expressions. We sourced candidate instances from the
MMKE dataset and constructed raw knowledge for each instance, primarily consisting of descriptive
narratives about the image semantics. Additionally, we ensured that every instance was supported by
at least three corresponding images retrieved from the MMKE dataset.
In summary, this benchmark encompasses a total of 1,573 pieces of knowledge and 3,381 images.
G.2 CONFLICT KNOWLEDGE GENERATION
For conflict knowledge construction in entity recognition, we primarily leverage large language
models to generate counterfactual conflict knowledge based on the original entity names while
keeping all other content unchanged.
For the construction of conflicting knowledge in entity knowledge categories, we further developed
multi-layered knowledge conflicts. Specifically, we employed large language models to generate coun-
terfactual conflicting knowledge for each individual dimension while preserving all other dimensions
unchanged.
For the construction of conflicting knowledge in image semantics, our focus lies on semantic
vocabulary substitution. Specifically, we replace an entity or action with another of the same
category—for example, substituting "happy" with "sad," or an "OK hand gesture" with a "phone hand
gesture."
H LLM Prompts for Different Steps
In this section, we provide a detailed list of all prompts for different steps, offering a clear reference
for understanding our experimental approach:
• The prompt for summary and organization of original knowledge is shown in Figure7.
•The prompt for generating conflicting Knowledge for Entity Recognition is shown in
Figure8.
• The prompt for generating conflicting Knowledge for character time is shown in Figure9.
•The prompt for generating conflicting Knowledge for character country of citizenship is
shown in Figure10.
•The prompt for generating conflicting Knowledge for character occupation is shown in
Figure11.
13

Figure 7: Prompt for summary of original knowledge
I EXPERIMENTS
We experimented with the VLMEvalKit library, which uses PyTorch and integrates several large
multimodal models. Experiments were conducted on NVIDIA L20 4BGB/A100 80GB GPUs.
MLLMs. To evaluate our benchmark, we conduct experiments on three representative MLLMs.
•Qwen2.5-VL: Qwen2.5 VL is a multimodal large model launched by Alibaba. It achieves
image-text joint understanding and generation by efficiently bridging the visual and language
modalities. Its design continues the advantages of the Qwen series in the field of language
models (LM), while combining visual encoders to form an end-to-end unified architecture.
•InternVL3: InternVL3 is the third-generation multimodal basic model launched by Shang-
hai AI Lab, focusing on the unification of general vision-language understanding and
cross-modal generation capabilities. Its core goal is to achieve deep integration of images,
videos, and texts through large-scale training and architecture innovation, and is suitable for
open world scenarios.
•GPT4o mini: GPT4o mini is a lightweight language model designed by OpenAI for edge
computing and low-cost deployment needs. While maintaining the core capabilities of
GPT-4o, it achieves a balance between performance and efficiency through architecture com-
pression and training optimization. It is suitable for scenarios such as real-time interaction
and mobile integration.
14

Figure 8: Prompt for Generate Entity Recognition conflicting knowledge
Figure 9: Prompt for Generate character time knowledge conflicting knowledge
15

Figure 10: Prompt for Generate character country of citizenship knowledge conflicting knowledge
Figure 11: Prompt for Generate character occupation knowledge conflicting knowledge
16

Figure 12: The results of Qwen2.5-VL with different model sizes under context-memory conflict
with open-ended question answering format
Figure 13: The results of Qwen2.5-VL with different model sizes under inter-context conflict with
open-ended question answering format
Figure 14: The results of InternVL3 with different model sizes under context-memory conflict with
multiple-choice question format
17

Figure 15: The results of InternVL3 with different model sizes under inter-context conflict with
open-ended question answering format
Figure 16: The results of InternVL3 with different model sizes under context-memory conflict with
open-ended question answering format
Figure 17: The results of InternVL3 with different model sizes under inter-context conflict with
open-ended question answering format
18

Table 5: Results of both context-memory and inter-context fine-grained Entity Knowledge conflicts
on MMKC-Bench using open-ended question answering format.
Qwen2.5-VL-7B InternVL3-8B GPT-4o mini
PT PL PC LT LC LO PT PL PC LT LC LO PT PL PC LT LC LO
Context-Memory Conflict
OAR 0.11 0.39 0.48 0.05 0.16 0.36 0.05 0.32 0.52 0.11 0.08 0.54 0.02 0.23 0.75 0.04 0.51 0.61
CAR 0.88 0.58 0.27 0.89 0.80 0.30 0.78 0.57 0.13 0.81 0.85 0.03 0.96 0.62 0.11 0.96 0.20 0.07
IAR 0.01 0.03 0.25 0.05 0.04 0.34 0.17 0.11 0.34 0.08 0.07 0.43 0.02 0.14 0.13 0.00 0.28 0.32
Inter-Context Conflict
OAR 0.08 0.20 0.35 0.03 0.05 0.14 0.03 0.19 0.19 0.04 0.03 0.39 0.03 0.30 0.68 0.03 0.51 0.66
CAR 0.89 0.78 0.38 0.97 0.92 0.64 0.71 0.75 0.62 0.89 0.96 0.36 0.96 0.60 0.01 0.97 0.23 0.05
IAR 0.03 0.02 0.27 0.00 0.03 0.23 0.26 0.06 0.19 0.07 0.01 0.23 0.01 0.10 0.30 0.00 0.26 0.28
Table 6: Results of both context-memory and inter-context fine-grained Entity Knowledge conflicts
on MMKC-Bench using the multiple-choice question format.
Qwen2.5-VL-7B InternVL3-8B GPT-4o mini
PT PL PC LT LC LO PT PL PC LT LC LO PT PL PC LT LC LO
Context-Memory Conflict
OAR 0.04 0.26 0.93 0.03 0.77 0.96 0.04 0.26 0.85 0.03 0.72 0.86 0.27 0.28 0.96 0.23 0.67 0.81
CAR 0.96 0.71 0.05 0.91 0.20 0.04 0.66 0.64 0.06 0.97 0.28 0.10 0.73 0.70 0.02 0.77 0.23 0.15
IAR 0.00 0.03 0.01 0.06 0.03 0.00 0.30 0.10 0.08 0.00 0.00 0.03 0.00 0.02 0.02 0.00 0.10 0.04
Inter-Context Conflict
OAR 0.03 0.35 0.97 0.02 0.74 0.93 0.01 0.22 0.87 0.04 0.80 0.86 0.24 0.13 0.83 0.22 0.66 0.74
CAR 0.94 0.62 0.02 0.97 0.24 0.07 0.94 0.77 0.11 0.94 0.20 0.12 0.76 0.85 0.10 0.78 0.29 0.24
IAR 0.00 0.02 0.01 0.00 0.02 0.00 0.05 0.02 0.02 0.01 0.00 0.01 0.00 0.02 0.08 0.00 0.05 0.01
J MORE RESULTS
In this section, we present more experimental results from the main paper to provide more guidance.
J.1 The results with different model sizes
We present the performance of the Qwen2.5-VL model series on open-ended question answering
under varying model sizes in the settings of context-memory conflict (Figure 12) and inter-context
conflict (Figure 13). For InternVL3, we show results across different model sizes using the multiple-
choice question format under context-memory conflict and inter-context conflict in Figure 14 and
Figure 15, respectively. In addition, the performance of InternVL3 with the open-ended question
format under both conflict settings is illustrated in Figure 16 (context-memory conflict) and Figure 17
(inter-context conflict). As we can see, larger models exhibit a stronger promoting effect across all
conflict types. The Overall Agreement Rate (OAR) generally increases with model size within both
the Qwen2.5-VL and InternVL3 series. This trend suggests that larger models are more strongly
influenced by their internal knowledge.
J.2 Model performance results for different types of knowledge
In this section, we show the performance of entity knowledge segmentation. We constructed two
entities and three dimensions of knowledge. As shown in Table 5, we show the performance of
Qwen2.5-VL, InternVL3, and GPT4o-mini models in the face of segmentation knowledge conflicts
under open questions. As shown in Table 6, we show the performance of Qwen2.5-VL, InternVL3,
and GPT4o mini models in the face of segmentation knowledge conflicts under selective questions.
It is seen that different types of data show different characteristics. For example, all models are
very sensitive to time-related knowledge conflicts. When time conflicts occur, the models tend to
be consistent with external documents. While for content-related knowledge conflicts, the models
tend to rely on their own memory.Note that we use abbreviations here. PT stands for character time
knowledge, PL stands for character nationality knowledge, and PC stands for character occupation
knowledge. LT stands for brand time knowledge, LC stands for brand creator knowledge, and LO
stands for brand product knowledge.
19

K Detailed examples of three types of multimodal knowledge conflicts in
MLLMKC-Bench
In this section, we present more examples of dataset in detail. For entity recognition, as shown in
Figure 18, we show eight types of data instances: plants, daily necessities, cartoon characters, brand
logos, characters, team logos, musical instruments, and transportation. For character knowledge,
as shown in Figure 20, we show data instances in three dimensions: character birth time, character
nationality, and character occupation. For brand knowledge, as shown in Figure 21, we show data
instances in three dimensions: brand creation time, brand creator, and brand main products. For visual
semantic, as shown in Figure19,we show three types of data instances: daily actions, expressions,
and traffic signs.
L Presentation of case studies of different conflict types.
In this section, we present the case studies of different conflict types. As shown in Figure22, we
present the case studies of different data instances when there is Context-Memory Conflict. As
shown in Figure23, we present the case studies of different data instances when there is Inter-Context
Conflict.
20

Figure 18: Data instance display diagram of eight entity recognition types.
Figure 19: Data instance display diagram of three visual semantic types.
21

Figure 20: A data example showing three dimensions of character knowledge.
Figure 21: Data example display of three dimensions of brand knowledge.
22

Figure 22: A diagram showing cases of different data instances under Context-Memory Conflict.
23

Figure 23: Case diagram showing different data instances under Inter-Context Conflict.
24