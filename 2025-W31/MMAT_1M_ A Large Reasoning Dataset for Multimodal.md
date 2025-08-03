# MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning

**Authors**: Tianhong Gao, Yannian Fu, Weiqun Wu, Haixiao Yue, Shanshan Liu, Gang Zhang

**Published**: 2025-07-29 15:39:14

**PDF URL**: [http://arxiv.org/pdf/2507.21924v1](http://arxiv.org/pdf/2507.21924v1)

## Abstract
Large Language Models (LLMs), enhanced through agent tuning, have
demonstrated remarkable capabilities in Chain-of-Thought (CoT) and tool
utilization, significantly surpassing the performance of standalone models.
However, the multimodal domain still lacks a large-scale, high-quality agent
tuning dataset to unlock the full potential of multimodal large language
models. To bridge this gap, we introduce MMAT-1M, the first million-scale
multimodal agent tuning dataset designed to support CoT, reflection, and
dynamic tool usage. Our dataset is constructed through a novel four-stage data
engine: 1) We first curate publicly available multimodal datasets containing
question-answer pairs; 2) Then, leveraging GPT-4o, we generate rationales for
the original question-answer pairs and dynamically integrate API calls and
Retrieval Augmented Generation (RAG) information through a multi-turn paradigm;
3) Furthermore, we refine the rationales through reflection to ensure logical
consistency and accuracy, creating a multi-turn dialogue dataset with both
Rationale and Reflection (RR); 4) Finally, to enhance efficiency, we optionally
compress multi-turn dialogues into a One-turn Rationale and Reflection (ORR)
format. By fine-tuning open-source multimodal models on the MMAT-1M, we observe
significant performance gains. For instance, the InternVL2.5-8B-RR model
achieves an average improvement of 2.7% across eight public benchmarks and 8.8%
on the RAG benchmark Dyn-VQA, demonstrating the dataset's effectiveness in
enhancing multimodal reasoning and tool-based capabilities. The dataset is
publicly available at https://github.com/VIS-MPU-Agent/MMAT-1M.

## Full Text


<!-- PDF content starts -->

MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning
Tianhong Gao*Yannian Fu*†Weiqun Wu Haixiao Yue Shanshan Liu Gang Zhang
Baidu Inc.
Abstract
Large Language Models (LLMs), enhanced through agent
tuning, have demonstrated remarkable capabilities in
Chain-of-Thought (CoT) and tool utilization, significantly
surpassing the performance of standalone models. How-
ever, the multimodal domain still lacks a large-scale, high-
quality agent tuning dataset to unlock the full potential
of multimodal large language models. To bridge this
gap, we introduce MMAT-1M, the first million-scale mul-
timodal agent tuning dataset designed to support CoT, re-
flection, and dynamic tool usage. Our dataset is con-
structed through a novel four-stage data engine: 1) We
first curate publicly available multimodal datasets contain-
ing question-answer pairs; 2) Then, leveraging GPT-4o,
we generate rationales for the original question-answer
pairs and dynamically integrate API calls and Retrieval
Augmented Generation (RAG) information through a multi-
turn paradigm; 3) Furthermore, we refine the rationales
through reflection to ensure logical consistency and accu-
racy, creating a multi-turn dialogue dataset with both Ra-
tionale and Reflection (RR); 4) Finally, to enhance effi-
ciency, we optionally compress multi-turn dialogues into a
One-turn Rationale and Reflection (ORR) format. By fine-
tuning open-source multimodal models on the MMAT-1M,
we observe significant performance gains. For instance, the
InternVL2.5-8B-RR model achieves an average improve-
ment of 2.7% across eight public benchmarks and 8.8% on
the RAG benchmark Dyn-VQA, demonstrating the dataset’s
effectiveness in enhancing multimodal reasoning and tool-
based capabilities. The dataset is publicly available at
https://github.com/VIS-MPU-Agent/MMAT-1M.
1. Introduction
In recent years, Multimodal Large Language Models
(MLLMs) exemplified by GPT-4o [51], Gemini [58], the
QwenVL series [1, 2, 65], the InternVL series [11–13], and
the LLaV A series [33, 34] have made remarkable strides. To
further enhance the reasoning and problem-solving capabil-
*Equal contribution.
†Corresponding author.
Figure 1. Performance comparison of multimodal large language
models fine-tuned on MMAT-1M dataset using One-turn Ratio-
nale and Reflection (ORR) and Rationale and Reflection (RR)
across eight benchmarks. Both strategies significantly boost per-
formance, demonstrating the effectiveness of structured reasoning
and MMAT-1M.
ities of these models, integrating Chain-of-Thought (CoT)
reasoning and external tools has proven to be an effective
approach, commonly referred to as “Agents”. Agents op-
erate through two primary methods: instruction-driven [17,
53, 66, 70, 71] and tuning-driven [5, 10, 62, 73, 75]. The
former involves designing prompts to enable LLMs to plan,
reason, and utilize tools, which demands strong prompt
comprehension. The latter employs specialized datasets
to fine-tune models, empowering even smaller models to
achieve agent capabilities comparable to proprietary large
models. Consequently, agent tuning has emerged as a
prominent and promising research direction.
In terms of existing research, several representative
works have emerged in the field of multimodal agent tun-
1arXiv:2507.21924v1  [cs.CV]  29 Jul 2025

ing. For instance, LLaV A-Plus [35] converts LLaV A-158K
dataset into a tool-use instruction format with 117K samples
through both user-oriented and skill-oriented dialogues, and
T3-Agent [18] constructs the MM-Traj dataset that contains
20K multimodal tasks with tool-usage trajectories. How-
ever, existing datasets commonly suffer from three critical
shortcomings: (1) They exhibit a relatively homogeneous
distribution, limiting improvements to diverse benchmarks;
(2) They lack mechanisms for reflecting on errors induced
by visual tools, resulting in weak model robustness against
interference; (3) They are deficient in flexible reasoning
and tool-usage mechanisms, reducing their feasibility for
real-world applications. Consequently, building a large-
scale tuning dataset that effectively addresses these chal-
lenges—diversity, robustness, and flexibility—has emerged
as a critical breakthrough for advancing the field.
To overcome these bottlenecks, we propose Multi-Modal
Agent Tuning—One Million (MMAT-1M), which, to the
best of our knowledge, is the first million-scale multi-
modal agent tuning dataset including diverse fundamental
visual tasks. Building on publicly available multimodal
datasets, we design a four-stage data synthesis framework.
First, we compile publicly accessible multimodal datasets
that encompass question-answer pairs. To ensure consis-
tency in input and output formats across diverse multimodal
datasets, we adapt the prompts for both inputs and outputs.
Then we generate iterative trajectories using CoT reasoning
and dynamic API calls, incorporating functionalities such
as Image Caption, Optical Character Recognition (OCR),
Open-V ocabulary Object Detection (OVD), Face Detection,
and RAG. Next, we evaluate these trajectories for logi-
cal inconsistencies and refine those requiring modification
through a reflection process. To enhance practical flexibil-
ity, we optionally consolidate iterative trajectories into a
one-turn format and prepend tool-usage results to the in-
put. Experimental results demonstrate that models fine-
tuned with the MMAT-1M dataset exhibit significant perfor-
mance advantages. As Figure 1 shows, after training on our
two formats of datasets, all three mainstream open-source
models achieve better performance compared to the base-
line. Taking the InternVL2.5-8B-RR model as an example,
it achieves an average improvement of 2.7% across eight
publicly available multimodal benchmarks compared to the
baseline model. Furthermore, on the Dyn-VQA benchmark,
which requires multi-hop reasoning and web search capa-
bilities, it demonstrates an improvement of 8.8%.
The main contributions of this study can be summarized
as follows: (1) We propose the first million-scale multi-
modal agent tuning dataset, MMAT-1M, addressing a criti-
cal gap in the domain of multimodal agent tuning. (2) We
establish a reflection mechanism that effectively mitigates
logical errors in the reasoning process, significantly enhanc-
ing the model’s robustness. (3) We offer datasets in bothone-turn and iterative formats, providing flexibility to bal-
ance precision and efficiency in practical applications.
2. Related Work
LLM-based Agents LLM-based agents are primarily large
models that harness the instruction-following capabilities
of LLMs to develop advanced reasoning and tool-usage
functionalities. Notable frameworks in this domain include
HuggingGPT [53], GPT4Tools [70], VisualChatGPT [66],
among others. ReAct [71], for instance, introduces a gen-
eral paradigm that integrates CoT reasoning with action
execution to address a broad spectrum of reasoning and
decision-making challenges. Similarly, AssistGPT [17],
proposes a “Learner” module that analyzes the prediction
process and facilitates reflection, aligning with ReAct’s
methodology. However, these approaches heavily rely on
the instruction comprehension capabilities of LLMs, which
restricts their effectiveness in handling longer or more com-
plex reasoning tasks. Additionally, the high computational
costs associated with invoking large models further raise the
barrier to practical application.
Multimodal Agent Tuning. Agent tuning is a specialized
subfield of language model fine-tuning, focused on enhanc-
ing the capabilities of LLMs in areas such as planning, rea-
soning, and tool usage. Among the earliest works in this
domain are AgentTuning [75] and Fireact [5], which laid
the foundation for subsequent advancements in agent tun-
ing. Subsequently, many efforts are dedicated to advanc-
ing agent tuning [10, 56, 62, 73]. However, these methods
primarily concentrate on optimizing LLMs, which, when
applied in the multimodal domain, can only access infor-
mation through multimodal tools. To address this limita-
tion, several studies have explored multimodal agent tuning
to improve reasoning and tool usage for multimodal chal-
lenges. For instance, LLaV A-Plus [35] represents the first
attempt to train a multimodal assistant through visual in-
struction tuning, enabling it to learn tool usage effectively.
Similarly, MLLM-Tool [63] is an agent system that inte-
grates multimodal encoders with open-source LLMs to per-
ceive and process instructions based on visual or audio in-
puts. Additionally, T3-Agent [18] generates a diverse range
of multimodal tasks with detailed trajectories and leverages
this data to fine-tune Vision-Language Models (VLMs) for
enhanced tool utilization.
Multimodal Agent and CoT Dataset. To achieve strong
performance in multimodal agent tuning, several datasets
have been developed to optimize agents using diverse
approaches. For instance, LLaV A-Plus transforms the
LLaV A-158K dataset into a tool-use instruction format.
Similarly, MLLM-Tool curates instruction-answer pairs en-
compassing 29 tasks sourced from HuggingFace. Mean-
while, T3-Agent introduces MM-Traj, a dataset comprising
20K trajectories, generated through a novel data collection
2

Statistics Component Number
Dataset CompositionVisual CoT [52] 434265
LLaV A-CoT [69] 98561
The Cauldron [29] 215680
TabMWP [41] 23059
Infoseek [9] 131400
Dialogue Turns1 turn 846389
2 turns 28646
3+ turns 27930
Rationale Steps2 turn 7909
3 turns 763212
4 turns 221440
5+ turns 97702
Operator CallsImage Caption 620644
OVD 156237
OCR 471866
Face Detection 20077
RAG 205682
Reflection CallsGeneral 46508
Math 11139
Table 1. Key statistics of the MMAT-1M dataset.
pipeline. Moreover, some agents, such as OmniSearch [31],
have designed the Dyn-VQA benchmark to evaluate capa-
bilities in RAG and multi-hop reasoning tasks. In addi-
tion to these multimodal agent datasets, several multimodal
CoT datasets share similar construction methodologies but
lack explicit information on tool usage, such as LLaV A-
CoT [68], Visual-CoT [52], and M3CoT [8].
3. MMAT-1M Dataset
In this section, we provide a comprehensive introduction to
MMAT-1M, detailing its key components and methodolo-
gies. The discussion is structured into three parts: (1) an
overview of the dataset, which outlines its scope, compo-
sition, and significance (Section 3.1); (2) the data engine,
which describes the iterative framework for generating and
refining high-quality trajectories (Section 3.2); and (3) the
multimodal agent tuning method, which explains the ap-
proaches for enhancing reasoning and tool-usage capabil-
ities (Section 3.3).
3.1. Overview of MMAT-1M
To build a diverse and comprehensive MMAT-1M dataset,
we consolidate data from five distinct sources. These
sources encompass a wide range of critical domains in
multimodal tasks, including visual understanding, logical
reasoning, mathematical computation, and knowledge re-
trieval. This integration ensures both the diversity and com-
pleteness of the dataset. The details of each dataset are as
follows:
Visual CoT [52] encompasses a variety of tasks, such
as document parsing, fine-grained understanding, generalDataset Size APIs Online Search CoT Reflection Turns
LLaV A-Plus-v1 [35] 117K ✓ ✓ multiple
Visual CoT [52] 438K ✓ one
LLaV A-CoT [69] 100K ✓ one
MM-Traj [18] 20K ✓ ✓ ✓ one
MMAT-1M 1M ✓ ✓ ✓ ✓ one&multiple
Table 2. Comparison of MMAT-1M with other training datasets.
visual question answering (VQA), chart analysis, and re-
lational reasoning. Its primary objective is to strengthen
models’ capabilities in focusing on localized visual regions
and executing step-by-step reasoning processes. LLaV A-
CoT [69] places a strong emphasis on complex reason-
ing and systematic thinking. It tackles a range of tasks,
including general VQA, scientific reasoning, mathemati-
cal reasoning, and document understanding, aiming to en-
hance models’ hierarchical reasoning capabilities and im-
prove their interpretability. The Cauldron [29] incorpo-
rates a wide array of multimodal data types, including in-
terleaved text-image documents, text-image pairs, OCR-
processed documents, and tables or charts. The diversity
of its data sources and task designs plays a pivotal role
in advancing models’ generalization capabilities, particu-
larly in the integration of visual and linguistic information.
TabMWP [41] focuses on mathematical reasoning tasks
that integrate both textual and tabular data, seeking to im-
prove models’ table parsing, numerical computation, and
complex reasoning capabilities. Infoseek [9] is centered on
visual information-seeking question answering, designed to
assess and enhance the performance of multimodal models
in knowledge-intensive visual question-answer tasks. These
tasks demand fine-grained reasoning that extends beyond
common sense and often relies on external knowledge bases
for accurate responses.
The statistical information of the MMAT-1M dataset
is shown in Table 1. The dataset comprises a total of
1,090,263 question-answer pairs and 902,965 dialogues,
distributed across distinct subsets to ensure diversity in data
sources. The second row of the table shows the number
of dialogue turns in the original data, which shows that the
one-turn dialogues represent the majority of samples, while
the multi-turn dialogues are comparatively less frequent. In
terms of reasoning complexity, the majority of data sam-
ples involve two-step and three-step reasoning processes,
which serve as the foundational level of reasoning. In con-
trast, tasks requiring more intricate, multi-step reasoning
constitute a smaller proportion, highlighting the dataset’s
inclusion of both basic and advanced cognitive challenges.
Meanwhile, among a wide range of operator calls, the invo-
cation of Image Caption and OCR is relatively high, indi-
cating the demand for basic information of images and text
in the reasoning process. RAG and OVD also account for
a notable proportion of operator invocations. Furthermore,
3

GPT-4o
UserHere are ocr results from the previous step:Miles driven. Day Number of miles Sunday 30 Monday 41 Tuesday 35 Wednesday 35 Thursday 35 Friday 34 Saturday 35Please continue your reasoning.Based on the above information, please summarize and give the answer to the initial question.<THOUGHT>I have extracted the numbers: 30, 41, 35, 35, 35, 34, 35.I will calculate the mean of these numbers.</THOUGHT><SIGNAL>end</SIGNAL>35
CheckMultimodalDatasetsQ:Gordon went on a road trip and tracked his driving each day. What is the mean of the numbers?GiveashortanswerA:35SampleAPI&RAG
<THOUGHT>To find the mean, I need to extract the numbers from the table and calculate their average. I have extracted the numbers: 30, 41, 35, 35, 35, 34, 35. I will calculate the mean of these numbers. Wait up, I think my reasoning up there could use a bit more math. Let me work on it. To find the mean of the numbers, first sum them up: 30 + 41 + 35 + 35 + 35 + 34 + 35 = 245. Then divide the total by the number of days (7): 245 / 7 = 35.</THOUGHT><ANSWER>The answer is: 35</ANSWER>Foundation
Gordon went on a road trip and tracked his driving each day. What is the mean of the numbers?Image
OptionallyTabmwp
LLaVACoT
TheCauldron
VisualCoT
Infoseekiterative
GPT-4o
Rationale
...<THOUGHT>Wait up, I think my reasoning up there could use a bit more math. Let me work on it.</THOUGHT><SIGNAL>reflection</SIGNAL><THOUGHT>To find the mean of the numbers, first sum them up: 30 + 41 + 35 + 35 + 35 + 34 + 35 = 245. Then divide the total by the number of days (7): 245/7 = 35.</THOUGHT><SIGNAL>end</SIGNAL>...ReflectionIntegrationOCRFaceDetectionImage DetectionGoogleSearchOVD
<THOUGHT>To find the mean, I need to extract the numbers from the table and calculate theiraverage.</THOUGHT><OPERATOR>ocr</OPERATOR><SIGNAL>continue</SIGNAL>Figure 2. The data engine pipeline follows four stages: foundation, rationale generation, reflection, and trajectory integration. It generates
datasets in two formats (RR and ORR) achieving a balance between precision and efficiency.
the reflection section encompasses both general reflection
and mathematical reasoning reflection, comprising a total
of approximately 57k data points. In summary, MMAT-1M
is distinguished by its large-scale data volume, diverse task
coverage, and hierarchical reasoning depth, collectively es-
tablishing a robust and flexible data foundation for advanc-
ing research in multimodal agent tuning.
We compare MMAT-1M with several similar agent tun-
ing and CoT datasets, including LLaV A-Plus-v1 [35], Vi-
sual CoT [52], LLaV A-CoT [69] and MM-Traj [18], as
shown in Table 2. It is evident that the scale of our dataset
substantially exceeds that of comparable datasets. Further-
more, our dataset is equipped with API and RAG tool invo-
cation capabilities, supports CoT reasoning and reflection,
and encompasses both one-turn and multi-turn reasoning
paradigms.
3.2. Data Engine
As shown in Figure 2, the data construction process is struc-
tured into four key stages: foundation, rationale generation,
reflection, and integration of trajectories.
Foundation. As an illustrative example, we randomly se-
lect an image and its corresponding question-answer pairfrom the original dataset. To ensure consistency in response
styles across different datasets, we optimize the phrasing
of the questions. For samples with shorter answers, we
append a response style constraint at the end of the ques-
tion, while keeping the original answer unchanged. Addi-
tionally, we prepare external tools for invocation, includ-
ing Image Caption, OVD, OCR, Face Detection, and RAG.
The Image Caption operator generates textual descriptions
of images, extracting key visual information and expressing
their semantics. Based on the CCoT [47], we use GPT-4o
to construct a scene graph and generate image descriptions
accordingly, enhancing semantic understanding and com-
positional reasoning capabilities. OVD leverages object in-
formation from the scene graph to identify and detect tar-
gets within an open vocabulary range, enabling the recog-
nition of novel categories that extend beyond a predefined
label set. This functionality is implemented using Ground-
ing DINO [36]. OCR utilizes PaddleOCR [15] to recognize
textual content within images. Face Detection, powered by
deepface [50], accurately locates facial regions in images.
Finally, for questions that require online search capabilities,
we leverage GPT-4o to generate search queries, which are
then used to invoke the Google API to retrieve the top-k
4

most relevant information.
Rationale. We employ an iterative diagram to generate ra-
tionales, where the annotation process is powered by GPT-
4o, ensuring the stability and efficiency of reasoning. Dur-
ing inference, the model adaptively invokes multimodal op-
erators RAG to maintain the completeness and interpretabil-
ity of the reasoning chain. The reasoning process initiates
with problem analysis, where the model selects appropriate
operators based on task requirements. If holistic seman-
tic understanding is required, the Image Caption operator
is invoked to extract a scene graph and generate an image
description. For tasks demanding object-level information,
the OVD operator is utilized to identify objects within an
open-vocabulary range. Similarly, the OCR operator and
Face Detection operator are employed for text recognition
and facial analysis, respectively. When operator outputs
are insufficient to support inference, the model formulates
RAG queries to retrieve and integrate external knowledge.
Each reasoning step is meticulously recorded in a structured
STRING format, capturing inference thoughts, operator in-
vocations, retrieval requests, and subsequent actions. This
adaptive multi-turn reasoning mechanism ensures the logi-
cal coherence of the reasoning chain, ultimately producing
accurate, interpretable, and well-documented rationales.
Reflection. In our observations, the rationales generated
through the process mentioned above exhibit two notable
issues. The first is incompleteness in the reasoning pro-
cess, particularly evident in the derivation of mathematical
problems. This occurs when certain steps are omitted, mak-
ing it challenging to arrive at the final answer. The sec-
ond issue is reasoning cheating behavior, where the ratio-
nale’s thought process does not logically lead to the final
answer, but GPT-4o forcibly aligns the reasoning with the
answer during label generation, creating an illusion of cor-
rectness. To address these issues, we introduce reflective
steps aimed at enhancing the model’s error-correction capa-
bilities during training and ensuring the reasoning process
remains logically sound. Specifically, for the first issue,
GPT-4o is tasked with identifying whether “step skipping”
behavior exists in the reasoning process. If such behavior
is detected, missing steps are supplemented to complete the
derivation. For the second issue, we employ GPT-4o to re-
evaluate whether the rationale’s thought process aligns with
the final answer. If a mismatch is identified, a reflective
process is implemented to make the rationale aware of the
cheating behavior and correct it accordingly.
Integration. The dataset generated through the approach
above adopts a multi-turn Rationale and Reflection (RR)
format, which may be impractical for real-world applica-
tions requiring time-sensitive responses. Inspired by the
LUMOS [73] model, we aim to create a dataset where the
model can deliberate and produce the final answer in one
turn. However, due to the constraints of the one-turn for-mat, we cannot dynamically incorporate the results of ex-
ternal operators during the output phase. To address this,
we integrate the results of all operators (excluding RAG)
into the input stage, clearly demarcated by brackets. At the
output stage, we consolidate multiple trajectories from the
multi-turn dialogue into a One-turn Rationale and Reflec-
tion (ORR) format. Our findings indicate that ORR not
only retains the ability to perform reasoning and integrate
external tool results but also significantly improves infer-
ence speed, making it more suitable for time-critical appli-
cations.
To assess potential GPT-4o hallucinations, we evaluated
all MMAT-1M samples on coherence, relevance, accuracy,
completeness, and image-text alignment, with over 89%
demonstrating high-quality reasoning. Evaluation criteria
are detailed in the supplementary material.
3.3. Multimodal Agent Tuning
Given a training sample: {{q1, r1}, ...{qi, ri}, ...{qn, A}},
where qiis i-th question, riindicates the rationale, and
A signifies the final answer. We select several open-
source multimodal models and employ supervised fine-
tuning (SFT) training schemes on these models.
SFT. We opt for low-rank adaptation (LoRA) [20], com-
pared to full parameters fine-tuning, which not only retains
the majority of the baseline model’s knowledge but also
save memory and computational space efficiently. The loss
function of it is designed as follows:
L=Loriginal +λX
i∥∆θi∥2
F, (1)
where Loriginal is the original loss function, ∆θiindicates
the update of the i-th weight matrix, λis the regularization
parameter, and ∥ · ∥Fdenotes the Frobenius norm.
4. Experiments
We conduct extensive experiments across multiple bench-
marks to evaluate the effectiveness of our approach. Sec-
tion 4.1 details the implementation settings. In Section 4.2,
we compare our method, which fine-tunes MLLMs with
One-turn Rationale and Reflection (ORR) and Rationale
and Reflection (RR) strategies on the MMAT-1M dataset,
against baselines. The evaluation spans eight benchmarks,
covering general and reasoning tasks, along with one bench-
mark for external knowledge retrieval. Section 4.3 presents
ablation studies and analyzes inference efficiency. Finally,
Section 4.4 provides qualitative results for further insights
into our method.
4.1. Implementation Details
In this section, we integrate MMAT-1M with various
MLLMs to showcase the broad applicability of our ap-
proach. We investigate two reasoning strategies, ORR and
5

Model Method Average MMStar MMMU MathVista MathVision AI2D OCRBench RealWorldQA HallusionBench
GPT-4o [23] / 65.6 65.1 70.7 60.0 30.4 84.9 806 76.5 56.2
Llama-3.2-11B-Vision-Instruct [46]Baseline 52.2 47.7 50.3 48.0 16.4 77.1 756 63.4 39.4
ORR 54.6 50.7 47.8 50.1 17.7 78.9 806 66.7 44.4
RR 55.3 51.4 51.0 49.1 16.8 77.9 784 69.3 48.3
MiniCPM-V-2.6 [72]Baseline 58.0 56.5 47.1 60.3 22.4 81.5 843 65.0 47.1
ORR 58.8 56.9 47.9 60.6 23.4 81.7 848 66.6 48.8
RR 59.9 58.5 49.2 61.9 25.3 82.0 840 68.0 50.0
InternVL2.5-2B [11]Baseline 52.7 53.6 43.2 50.1 16.1 75.1 804 60.5 42.6
ORR 54.4 55.4 44.7 50.1 14.1 77.5 819 69.5 42.4
RR 54.7 54.9 44.4 52.6 16.5 77.2 799 68.0 43.8
InternVL2.5-4B [11]Baseline 58.4 58.6 51.8 60.8 21.7 81.2 823 64.6 46.5
ORR 59.5 59.2 50.7 61.4 19.7 81.4 824 69.2 51.9
RR 60.6 60.9 53.1 62.0 22.4 82.7 805 72.2 50.7
InternVL2.5-8B [11]Baseline 60.7 62.4 53.1 64.5 20.1 84.1 819 69.4 49.8
ORR 62.4 64.8 55.4 63.8 20.8 83.5 849 73.0 53.3
RR 63.4 65.3 57.3 64.8 21.7 84.2 839 74.4 55.8
Table 3. Performance comparison of MLLMs with Baseline, ORR (One-turn Rationale and Reflection), and RR (Rationale and Reflection)
across eight benchmarks. Models trained on MMAT-1M with ORR and RR achieve overall gains, enhancing multimodal capabilities.
Model Query Golden Query
GPT-4o [23] 52.0 61.5
OmniSearch (GPT-4V) [31] 50.0 /
Llama-3.2-11B-Vision-Instruct [46] 29.4 34.6
Llama-3.2-11B-Vision-Instruct-RR 38.0 45.1
MiniCPM-V-2.6 [72] 32.7 39.2
MiniCPM-V-2.6-RR 35.9 44.4
InternVL2.5-2B [11] 19.3 26.0
InternVL2.5-2B-RR 30.9 38.8
InternVL2.5-4B [11] 23.3 31.1
InternVL2.5-4B-RR 35.4 42.1
InternVL2.5-8B [11] 27.0 35.2
InternVL2.5-8B-RR 36.8 44.0
Table 4. Results on the RAG Benchmark Dyn-VQA. RR strategy
significantly boosts performance across model scales, enhancing
multi-hop reasoning and retrieval.
RR, which guide multimodal models toward structured and
interpretable reasoning. ORR consolidates all reasoning
steps into a single query, enabling efficient inference while
maintaining strong accuracy. In contrast, RR follows a
multi-step reasoning process, dynamically selecting oper-
ators and retrieving external knowledge when needed. For
reasoning scenarios that require external knowledge injec-
tion, we employ Google Search to retrieve relevant infor-
mation. Each query returns up to three results (top-k=3),
providing the model with necessary contextual knowledge
while maintaining efficiency.
We apply these strategies to open-source multimodal
models, including Llama-3.2-11B-Vision-Instruct [46],
MiniCPM-V-2.6 [72], and the InternVL2.5 series [11],
which includes InternVL2.5-2B, InternVL2.5-4B, and
InternVL2.5-8B. Each model is separately fine-tuned with
ORR and RR on the MMAT-1M dataset, which consists
of 1,090,263 question-answer pairs, for one epoch with alearning rate of 4e-5. Detailed training parameters are pro-
vided in the supplementary material.
4.2. Main Results on Benchmark
Setup. We conduct a comprehensive evaluation of our
method using eight widely adopted and challenging bench-
marks: MMStar [7], MMMU [74], MathVista [42],
MathVision [64], AI2D [25], OCRBench [37], Real-
WorldQA [67], and HallusionBench [19]. Specifically,
MMStar and MMMU primarily assess multimodal reason-
ing and question-answering capabilities, while MathVista
and MathVision focus on mathematical and visual reason-
ing skills. AI2D examines the comprehension of scien-
tific diagrams, and OCRBench evaluates textual informa-
tion extraction from documents. RealWorldQA targets spa-
tial reasoning in real-world scenarios, whereas Hallusion-
Bench gauges susceptibility to language hallucinations and
visual illusions. For MathVista and MathVision, we adopt
the testmini set. To ensure fairness and reproducibility,
all evaluations are conducted using VLMEvalKit [16], an
open-source toolkit specifically designed for large vision-
language models. Beyond these benchmarks, we further
evaluate the RAG capabilities of the models with the Dyn-
VQA dataset proposed in OmniSearch [31]. Dyn-VQA
encompasses dynamic, multimodal, multi-hop reasoning
tasks, offering a comprehensive assessment of how effec-
tively models plan retrieval strategies and integrate relevant
information.
Main Results. Table 3 presents experimental results on
multiple benchmarks that evaluate the performance of var-
ious multimodal large models trained on MMAT-1M using
ORR and RR. The findings demonstrate that both methods
effectively enhance model performance across different pa-
rameter scales.
Training with our ORR on MMAT-1M improves the av-
6

Model API RAG Average MMStar MMMU MathVista MathVision AI2D OCRBench RealWorldQA HallusionBench Dyn-VQA
Baseline 57.9 62.4 53.1 65.1 20.1 84.1 819 69.4 49.8 35.2
Baseline-RR ✓ 59.8 65.0 56.2 64.2 20.4 84.1 839 74.3 55.0 35.4
Baseline-RR ✓ 57.3 60.1 52.6 61.1 21.0 81.8 797 67.8 48.0 43.4
Baseline-RR (w/o SFT) ✓ ✓ 55.0 60.6 49.8 60.9 15.1 82.8 825 68.9 43.2 31.5
Baseline-R ✓ ✓ 60.2 65.0 54.5 63.9 20.5 84.6 826 72.7 54.8 42.9
Baseline-ORR ✓ 59.6 64.8 55.4 63.8 20.8 83.5 849 73.0 53.3 36.6
Baseline-RR ✓ ✓ 61.3 65.3 57.3 64.8 21.7 84.2 839 74.4 55.8 44.0
Table 5. Ablation study evaluating the impact of SFT, API integration, structured reflection, and RAG-based retrieval on multimodal
reasoning performance. Results highlight the complementary benefits of fine-tuning, explicit rationale generation, and external knowledge
integration in enhancing multimodal reasoning performance.
erage score of InternVL2.5-8B from 60.7 to 62.4 compared
to the baseline, while our RR strategy further boosts it to
63.4. Notably, RR consistently outperforms the baseline
and achieves competitive results against GPT-4o. Specifi-
cally, InternVL2.5-8B with RR surpasses GPT-4o on MM-
Star (65.3 vs. 65.1) and MathVista (64.8 vs. 60.0), demon-
strating superior multimodal reasoning and mathematical-
visual understanding. It also outperforms GPT-4o on OCR-
Bench (839 vs. 806), reflecting stronger textual information
extraction. Additionally, it performs on par with GPT-4o on
AI2D (84.2 vs. 84.9) and HallusionBench (55.8 vs. 56.2),
indicating robust comprehension of scientific diagrams and
resilience to multimodal hallucinations.
Compared with baseline models such as InternVL2.5-
8B, MiniCPM-V-2.6, and Llama-3.2-11B-Vision-Instruct,
our ORR and RR particularly RR, have demonstrated gener-
ally similar optimization effects across various test sets. Our
RR on MiniCPM-V2.6 achieves a gain in average from 58.0
to 59.9, a 3.3% relative increase, while on Llama-3.2-11B-
Vision-Instruct achieves a gain from 52.2 to 55.3, a relative
improvement of 5.9%. This indicates that our methods have
broad applicability across different model series. Similarly,
our ORR and RR consistently deliver strong performance
across the InternVL2.5 series, including the 2B, 4B, and
8B parameter variants, demonstrating robust scalability and
wide-ranging applicability of our methodology.
In OCRBench, InternVL2.5-2B’s ORR strategy outper-
forms the baseline (804 to 819), while RR drops to 799,
a trend also seen in InternVL2.5-4B and 8B. The reason
for this phenomenon is that, although RR exhibits specific
error-correction capabilities, the OCR misrecognition neg-
atively impacts the final results. In contrast, ORR utilizes
image captioning to mitigate OCR errors, demonstrating su-
perior performance in OCRBench.
The comprehensive results confirm that training on
MMAT-1M with our ORR and RR leads to significant
improvements, particularly with RR, in tasks requiring
comprehensive reasoning, mathematical computation, and
cross-modal information fusion. This establishes MMAT-
1M as a valuable benchmark for advancing the reasoning
capabilities of vision-language models.Results on RAG Benchmark. The evaluation results
of Dyn-VQA [31] are shown in Table 4, based on the lat-
est version. Query refers to the input content used by the
model for information retrieval, while Golden Query de-
notes an optimized prompt focused on the final retrieval step
to maximize answer accuracy. To align with Dyn-VQA, we
adopt the same evaluation metric, F1-Recall, which mea-
sures the overlap between the model-generated response
and the ground truth. Results demonstrate that our ORR
and RR consistently enhance multi-hop reasoning and re-
trieval performance. Specifically, the RR improves Llama-
3.2-11B-Vision-Instruct by 29.3% relative to its original
performance (from 29.4 to 38.0) in Query and by 30.3%
relative to its original performance (from 34.6 to 45.1) in
Golden Query, while MiniCPM-V-2.6 shows improvements
of 9.8% and 13.3%, respectively. The InternVL2.5 se-
ries models similarly benefit, with relative gains ranging
from 31.9% to 60.1%, underscoring the effectiveness of our
methods across complex, knowledge-intensive tasks.
2B4B8B
2B4B8B
2B4B8B
Figure 3. Comparison of inference efficiency and performance
gains of ORR and RR across different InternVL2.5 model scales.
4.3. Further Analysis
Ablation Study. Table 5 presents an ablation study on the
effects of SFT, API integration, structured reflection, and
RAG on multimodal reasoning performance. The base-
line model, without external resources, achieves an aver-
7

Which university did this person receive her honorary Doctor of Fine Arts degree from?New York University<THOUGHT>The image shows a person…</THOUGHT><OPERATOR>face detection</OPERATOR>...Here are face detection results from the previous step:There is one facewithacoordinate: [454.0, 301.0, 883.0, 838.0]…<THOUGHT>…doesn‘t identify…useRAG…</THOUGHT><RAG>Which university did the person…</RAG>…Here…She was awarded an honorary Doctor of Fine Arts degree from Princeton Universityin 2002…<THOUGHT>…PrincetonUniversity…</THOUGHT><SIGNAL>end</SIGNAL>Based on the above information…Princeton UniversityWhich…Note,you can additionally use<OPERATOR>face recognition</OPERATOR> to recognize faces.<THOUGHT>The image shows a person…</THOUGHT><OPERATOR>face recognition</OPERATOR>…Here are face recognition results from the previous step:Taylor Swift …<THOUGHT>Now, I need to find out…</THOUGHT><RAG>Taylor Swift honorary Doctor of Fine Arts…</RAG>…Here…I rolled my eyes at Taylor Swift‘s honorary degree from NYUWhen I first saw the news about…New York University…<THOUGHT>The RAG results confirm...</THOUGHT><SIGNAL>end</SIGNAL>Based on the above information…New York UniversityWhich university did this person…? Please start yourreasoning.QAWhich university did this person receive her honorary Doctor of Fine Arts degree from?The person in the image received her honorary Doctor of Fine Arts degree from Middle Tennessee State University.Q&ARRBaselineRR+RecFigure 4. The zero-shot capability of invoking a celebrity recognition operator of InternVL2.5-8B-RR.
age score of 57.9. In the RR setting, Baseline-RR achieves
the highest score of 61.3 with both API and RAG. Remov-
ing API reduces performance to 57.3, while removing RAG
lowers it to 59.8. Without SFT, performance declines fur-
ther to 55.0. Additionally, Baseline-R, which retains ratio-
nale but omits reflection, scores 60.2, suggesting that re-
flection enhances reasoning ability. In the ORR setting,
performance declines to 59.6, primarily because the ORR
format does not incorporate RAG information, resulting in
a performance drop on the Dyn-VQA benchmark. On other
benchmarks, however, its performance remains comparable
to that of the RR format. These results confirm that SFT is
crucial for instruction adherence, while structured reflection
and external knowledge integration further improve multi-
modal reasoning.
Performance Efficiency Tradeoff between ORR and RR.
Figure 3 compares the inference efficiency and performance
gains of the ORR and RR methods across different In-
ternVL2.5 model scales. Although both ORR and RR con-
sistently enhance multimodal reasoning performance, their
inference times notably increase relative to the baseline.
ORR introduces a moderate inference overhead due to its
one-turn structured reasoning approach, while RR, involv-
ing multi-turn adaptive reasoning steps, incurs a slightly
higher computational cost. However, RR achieves greater
performance improvements compared to ORR, demonstrat-
ing a beneficial tradeoff between computational efficiency
and reasoning accuracy.
4.4. Qualitative Results
While the experiments, as mentioned above, have demon-
strated the benefits of invoking external tools for the model,
the capabilities of a fixed set of tools are inherently limited.For instance, MMAT-1M’s lack of a celebrity recognition
operator hinders the fine-tuned model from achieving cor-
rect results in cases requiring celebrity identification. To
address this, we conduct an experiment to verify whether
the fine-tuned model can invoke operators it has not been
explicitly trained on. As shown in Figure 4 , we test a
visual question with the InternVL2.5-8B model. Initially,
the baseline model provides an incorrect answer. As antici-
pated, the model fine-tuned on MMAT-1M, failing to recog-
nize the person, also returns a wrong answer due to unsuc-
cessful web search results. To address this limitation, we in-
struct the fine-tuned model to invoke a celebrity recognition
operator, which successfully identifies the correct answer.
This experiment demonstrates that the model fine-tuned on
our dataset exhibits a certain level of zero-shot capability
for invoking unseen tools. However, its performance re-
mains inferior to that achieved through explicit fine-tuning.
5. Conclusion
The introduction of MMAT-1M represents a significant ad-
vancement in multimodal agent tuning, offering a diverse
and flexible dataset for enhancing CoT reasoning and tool
usage in MLLMs. By addressing key limitations of exist-
ing multimodal agent tuning datasets, such as homogene-
ity, lack of reflection, and inflexible tool usage, it provides
a comprehensive solution that aligns with the demands of
real-world applications. While the dataset demonstrates ro-
bust performance on current multimodal benchmarks, fur-
ther research is essential to evaluate its adaptability to a
broader array of MLLMs and more intricate real-world sce-
narios.
8

References
[1] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan
Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren
Zhou. Qwen-vl: A versatile vision-language model for un-
derstanding, localization, text reading, and beyond. arxiv
2023. arXiv preprint arXiv:2308.12966 , 1(8), 2023. 1
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun
Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhao-
hai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren
Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen
Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Jun-
yang Lin. Qwen2.5-vl technical report. arXiv preprint
arXiv:2502.13923 , 2025. 1
[3] Ali Furkan Biten, Rub `en Tito, Andr ´es Mafla, Lluis Gomez,
Marc ¸al Rusi ˜nol, C.V . Jawahar, Ernest Valveny, and Dimos-
thenis Karatzas. Scene text visual question answering. In
2019 IEEE/CVF International Conference on Computer Vi-
sion (ICCV) , pages 4290–4300, 2019. 13
[4] Jie Cao and Jing Xiao. An augmented benchmark dataset for
geometric question answering through dual parallel text en-
coding. In Proceedings of the 29th International Conference
on Computational Linguistics , pages 1511–1520, Gyeongju,
Republic of Korea, 2022. International Committee on Com-
putational Linguistics. 13
[5] Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier,
Karthik Narasimhan, and Shunyu Yao. Fireact: Toward lan-
guage agent fine-tuning. arXiv preprint arXiv:2310.05915 ,
2023. 1, 2
[6] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui
He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v:
Improving large multi-modal models with better captions. In
European Conference on Computer Vision (ECCV) , 2024. 13
[7] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang
Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao,
Dahua Lin, et al. Are we on the right way for evaluating large
vision-language models? In The Thirty-eighth Annual Con-
ference on Neural Information Processing Systems , 2024. 6
[8] Qiguang Chen, Libo Qin, Jin Zhang, Zhi Chen, Xiao Xu,
and Wanxiang Che. M3cot: A novel benchmark for multi-
domain multi-step multi-modal chain-of-thought. arXiv
preprint arXiv:2405.16473 , 2024. 3
[9] Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, So-
ravit Changpinyo, Alan Ritter, and Ming-Wei Chang.
Can pre-trained vision and language models answer vi-
sual information-seeking questions? arXiv preprint
arXiv:2302.11713 , 2023. 3, 13
[10] Zehui Chen, Kuikun Liu, Qiuchen Wang, Wenwei Zhang,
Jiangning Liu, Dahua Lin, Kai Chen, and Feng Zhao. Agent-
flan: Designing data and methods of effective agent tuning
for large language models. arXiv preprint arXiv:2403.12881 ,
2024. 1, 2
[11] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhang-
wei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian,
Zhaoyang Liu, et al. Expanding performance boundaries of
open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 , 2024. 1, 6,
15
[12] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhang-
wei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng
Luo, Zheng Ma, et al. How far are we to gpt-4v? closing
the gap to commercial multimodal models with open-source
suites. Science China Information Sciences , 67(12):220101,
2024.
[13] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen,
Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu,
Lewei Lu, et al. Internvl: Scaling up vision foundation mod-
els and aligning for generic visual-linguistic tasks. In Pro-
ceedings of the IEEE/CVF conference on computer vision
and pattern recognition , pages 24185–24198, 2024. 1
[14] Adam Dahlgren Lindstr ¨om and Savitha Sam Abraham.
Clevr-math: A dataset for compositional language, visual
and mathematical reasoning. In International Joint Confer-
ence on Learning and Reasoning, 16th International Work-
shop on Neural-Symbolic Learning and Reasoning (NeSy
2022), Windsor, UK, September 28-30, 2022 , pages 155–
170. Technical University of Aachen, 2022. 13
[15] Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Weiwei
Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua Yang, Qingqing
Dang, et al. Pp-ocr: A practical ultra lightweight ocr system.
arXiv preprint arXiv:2009.09941 , 2020. 4
[16] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang,
Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang,
Jiaqi Wang, et al. Vlmevalkit: An open-source toolkit for
evaluating large multi-modality models. In Proceedings
of the 32nd ACM International Conference on Multimedia ,
pages 11198–11201, 2024. 6
[17] Difei Gao, Lei Ji, Luowei Zhou, Kevin Qinghong Lin, Joya
Chen, Zihan Fan, and Mike Zheng Shou. Assistgpt: A gen-
eral multi-modal assistant that can plan, execute, inspect, and
learn. arXiv preprint arXiv:2306.08640 , 2023. 1, 2
[18] Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaojian Ma, Tao
Yuan, Yue Fan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, and
Qing Li. Multi-modal agent tuning: Building a vlm-driven
agent for efficient tool usage. In The Thirteenth International
Conference on Learning Representations , 2025. 2, 3, 4
[19] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia
Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang,
Yaser Yacoob, et al. Hallusionbench: an advanced diagnos-
tic suite for entangled language hallucination and visual il-
lusion in large vision-language models. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition , pages 14375–14385, 2024. 6
[20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-
Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al.
Lora: Low-rank adaptation of large language models. ICLR ,
1(2):3, 2022. 5
[21] Zheng Huang, Kai Chen, Jianhua He, Xiang Bai, Dimosthe-
nis Karatzas, Shijian Lu, and CV Jawahar. Icdar2019 compe-
tition on scanned receipt ocr and information extraction. In
2019 International Conference on Document Analysis and
Recognition (ICDAR) , pages 1516–1520. IEEE, 2019. 13
[22] Drew A Hudson and Christopher D Manning. Gqa: A new
dataset for real-world visual reasoning and compositional
9

question answering. In Proceedings of the IEEE/CVF con-
ference on computer vision and pattern recognition , pages
6700–6709, 2019. 13
[23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perel-
man, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Weli-
hinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card.
arXiv preprint arXiv:2410.21276 , 2024. 6
[24] Justin Johnson, Bharath Hariharan, Laurens van der Maaten,
Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr:
A diagnostic dataset for compositional language and elemen-
tary visual reasoning. In Proceedings of the IEEE Confer-
ence on Computer Vision and Pattern Recognition (CVPR) ,
2017. 13
[25] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon
Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is
worth a dozen images. In Computer Vision–ECCV 2016:
14th European Conference, Amsterdam, The Netherlands,
October 11–14, 2016, Proceedings, Part IV 14 , pages 235–
251. Springer, 2016. 6, 13
[26] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk,
Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are
you smarter than a sixth grader? textbook question answer-
ing for multimodal machine comprehension. In 2017 IEEE
Conference on Computer Vision and Pattern Recognition
(CVPR) , pages 5376–5384, 2017. 13
[27] Douwe Kiela, Hamed Firooz, Aravind Mohan, Vedanuj
Goswami, Amanpreet Singh, Pratik Ringshia, and Davide
Testuggine. The hateful memes challenge: Detecting hate
speech in multimodal memes. In Advances in Neural Infor-
mation Processing Systems , pages 2611–2624. Curran Asso-
ciates, Inc., 2020. 13
[28] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Ui-
jlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan
Popov, Matteo Malloci, Alexander Kolesnikov, et al. The
open images dataset v4: Unified image classification, object
detection, and visual relationship detection at scale. Interna-
tional Journal of Computer Vision , 128(7):1956–1981, 2020.
13
[29] Hugo Laurenc ¸on, L ´eo Tronchon, Matthieu Cord, and Victor
Sanh. What matters when building vision-language models?
arXiv preprint arXiv:2405.02246 , 2024. 3, 13
[30] Junnan Li, Yongkang Wong, Qi Zhao, and Mohan S.
Kankanhalli. People in social context (pisc) dataset, 2017.
Data set. 13
[31] Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen
Zhang, Xinran Zheng, Hui Wang, Hai-Tao Zheng, Pengjun
Xie, Philip S Yu, et al. Benchmarking multimodal retrieval
augmented generation with dynamic vqa dataset and self-
adaptive planning agent. arXiv preprint arXiv:2411.02937 ,
2024. 3, 6, 7
[32] Fangyu Liu, Guy Edward Toh Emerson, and Nigel Collier.
Visual spatial reasoning. Transactions of the Association for
Computational Linguistics , 2023. 13
[33] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.
Visual instruction tuning. Advances in neural information
processing systems , 36:34892–34916, 2023. 1
[34] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee.
Improved baselines with visual instruction tuning. In Pro-ceedings of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition , pages 26296–26306, 2024. 1
[35] Shilong Liu, Hao Cheng, Haotian Liu, Hao Zhang, Feng Li,
Tianhe Ren, Xueyan Zou, Jianwei Yang, Hang Su, Jun Zhu,
et al. Llava-plus: Learning to use tools for creating multi-
modal agents. In European Conference on Computer Vision ,
pages 126–142. Springer, 2024. 2, 3, 4
[36] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao
Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang,
Hang Su, et al. Grounding dino: Marrying dino with
grounded pre-training for open-set object detection. In
European Conference on Computer Vision , pages 38–55.
Springer, 2024. 4
[37] Yuliang Liu, Zhang Li, Biao Yang, Chunyuan Li, Xucheng
Yin, Cheng-lin Liu, Lianwen Jin, and Xiang Bai. On the
hidden mystery of ocr in large multimodal models. arXiv
e-prints , pages arXiv–2305, 2023. 6
[38] Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang,
Xiaodan Liang, and Song-Chun Zhu. Inter-gps: Interpretable
geometry problem solving with formal language and sym-
bolic reasoning. In The Joint Conference of the 59th An-
nual Meeting of the Association for Computational Linguis-
tics and the 11th International Joint Conference on Natural
Language Processing (ACL-IJCNLP 2021) , 2021. 13
[39] Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao,
Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu.
Iconqa: A new benchmark for abstract diagram understand-
ing and visual language reasoning. In The 35th Conference
on Neural Information Processing Systems (NeurIPS) Track
on Datasets and Benchmarks , 2021. 13
[40] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei
Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and
Ashwin Kalyan. Learn to explain: Multimodal reasoning
via thought chains for science question answering. In The
36th Conference on Neural Information Processing Systems
(NeurIPS) , 2022. 13
[41] Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-
Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin
Kalyan. Dynamic prompt learning via policy gradient for
semi-structured mathematical reasoning. arXiv preprint
arXiv:2209.14610 , 2022. 3, 13
[42] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li,
Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel
Galley, and Jianfeng Gao. Mathvista: Evaluating mathe-
matical reasoning of foundation models in visual contexts.
InThe Twelfth International Conference on Learning Repre-
sentations , 2024. 6
[43] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Ena-
mul Hoque. ChartQA: A benchmark for question answer-
ing about charts with visual and logical reasoning. In Find-
ings of the Association for Computational Linguistics: ACL
2022 , pages 2263–2279, Dublin, Ireland, 2022. Association
for Computational Linguistics. 13
[44] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar.
Docvqa: A dataset for vqa on document images. In Proceed-
ings of the IEEE/CVF winter conference on applications of
computer vision , pages 2200–2209, 2021. 13
10

[45] Minesh Mathew, Viraj Bagal, Rub `en Tito, Dimosthenis
Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa.
InProceedings of the IEEE/CVF Winter Conference on Ap-
plications of Computer Vision , pages 1697–1706, 2022. 13
[46] AI Meta. Llama 3.2: Revolutionizing edge ai and vision
with open, customizable models. Meta AI Blog. Retrieved
December , 20:2024, 2024. 6, 15
[47] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei
Herzig. Compositional chain-of-thought prompting for large
multimodal models. In Proceedings of the IEEE/CVF Con-
ference on Computer Vision and Pattern Recognition , pages
14420–14431, 2024. 4, 13
[48] Panupong Pasupat and Percy Liang. Compositional semantic
parsing on semi-structured tables. In Proceedings of the 53rd
Annual Meeting of the Association for Computational Lin-
guistics and the 7th International Joint Conference on Nat-
ural Language Processing (Volume 1: Long Papers) , pages
1470–1480, Beijing, China, 2015. Association for Computa-
tional Linguistics. 13
[49] Bryan A Plummer, Liwei Wang, Chris M Cervantes,
Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazeb-
nik. Flickr30k entities: Collecting region-to-phrase corre-
spondences for richer image-to-sentence models. In Pro-
ceedings of the IEEE international conference on computer
vision , pages 2641–2649, 2015. 13
[50] Sefik Serengil and Alper ¨Ozpınar. A benchmark of facial
recognition pipelines and co-usability performances of mod-
ules. Bilis ¸im Teknolojileri Dergisi , 17(2):95–107, 2024. 4
[51] Sakib Shahriar, Brady D Lund, Nishith Reddy Man-
nuru, Muhammad Arbab Arshad, Kadhim Hayawi, Ravi
Varma Kumar Bevara, Aashrith Mannuru, and Laiba Batool.
Putting gpt-4o to the sword: A comprehensive evaluation of
language, vision, speech, and multimodal proficiency. Ap-
plied Sciences , 14(17):7782, 2024. 1
[52] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan
Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot:
Advancing multi-modal language models with a comprehen-
sive dataset and benchmark for chain-of-thought reasoning.
InThe Thirty-eight Conference on Neural Information Pro-
cessing Systems Datasets and Benchmarks Track , 2024. 3, 4,
13
[53] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li,
Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai
tasks with chatgpt and its friends in hugging face. Advances
in Neural Information Processing Systems , 36:38154–38180,
2023. 1, 2
[54] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and
Amanpreet Singh. Textcaps: a dataset for image caption-
ing with reading comprehension. In Computer Vision–ECCV
2020: 16th European Conference, Glasgow, UK, August 23–
28, 2020, Proceedings, Part II 16 , pages 742–758. Springer,
2020. 13
[55] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang,
Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus
Rohrbach. Towards vqa models that can read. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition , pages 8317–8326, 2019. 13[56] Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li,
and Bill Yuchen Lin. Trial and error: Exploration-based
trajectory optimization for llm agents. arXiv preprint
arXiv:2403.02502 , 2024. 2
[57] Benny J. Tang, Angie Boggust, and Arvind Satyanarayan.
VisText: A Benchmark for Semantically Rich Chart Cap-
tioning. In The Annual Meeting of the Association for Com-
putational Linguistics (ACL) , 2023. 13
[58] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-
Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a
family of highly capable multimodal models. arXiv preprint
arXiv:2312.11805 , 2023. 1
[59] Jordy Van Landeghem, Rub `en Tito, Łukasz Borchmann,
Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Ju-
rkiewicz, Micka ¨el Coustaty, Bertrand Anckaert, Ernest Val-
veny, et al. Document understanding dataset and evaluation
(dude). In Proceedings of the IEEE/CVF International Con-
ference on Computer Vision , pages 19528–19540, 2023. 13
[60] Catherine Wah, Steve Branson, Peter Welinder, Pietro Per-
ona, and Serge Belongie. The caltech-ucsd birds-200-2011
dataset. 2011. 13
[61] Bryan Wang, Gang Li, Xin Zhou, Zhourong Chen, Tovi
Grossman, and Yang Li. Screen2words: Automatic mobile
ui summarization with multimodal learning. In The 34th An-
nual ACM Symposium on User Interface Software and Tech-
nology , page 498–510, New York, NY , USA, 2021. Associa-
tion for Computing Machinery. 13
[62] Boshi Wang, Hao Fang, Jason Eisner, Benjamin Van Durme,
and Yu Su. Llms in the imaginarium: tool learning through
simulated trial and error. arXiv preprint arXiv:2403.04746 ,
2024. 1, 2
[63] Chenyu Wang, Weixin Luo, Qianyu Chen, Haonan Mai,
Jindi Guo, Sixun Dong, XM Xuan, Zhengxin Li, Lin
Ma, and Shenghua Gao. Mllm-tool: A multimodal large
language model for tool agent learning. arXiv preprint
arXiv:2401.10727 , 4, 2024. 2
[64] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing
Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Mea-
suring multimodal mathematical reasoning with math-vision
dataset. Advances in Neural Information Processing Sys-
tems, 37:95095–95169, 2025. 6
[65] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan,
Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin
Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui
Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Jun-
yang Lin. Qwen2-vl: Enhancing vision-language model’s
perception of the world at any resolution. arXiv preprint
arXiv:2409.12191 , 2024. 1
[66] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang,
Zecheng Tang, and Nan Duan. Visual chatgpt: Talking,
drawing and editing with visual foundation models. arXiv
preprint arXiv:2303.04671 , 2023. 1, 2
[67] X.AI. Grok-1.5 vision preview. https://x.ai/blog/
grok-1.5v , 2024. 6
[68] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and
Li Yuan. Llava-o1: Let vision language models reason step-
by-step. arXiv preprint arXiv:2411.10440 , 2024. 3
11

[69] Guowei Xu, Peng Jin, Hao Li, Yibing Song, Lichao Sun, and
Li Yuan. Llava-cot: Let vision language models reason step-
by-step, 2024. 3, 4, 13
[70] Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu
Li, and Ying Shan. Gpt4tools: Teaching large language
model to use tools via self-instruction. Advances in Neural
Information Processing Systems , 36:71995–72007, 2023. 1,
2
[71] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran,
Karthik Narasimhan, and Yuan Cao. React: Synergizing rea-
soning and acting in language models. In International Con-
ference on Learning Representations (ICLR) , 2023. 1, 2
[72] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui,
Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He,
et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv
preprint arXiv:2408.01800 , 2024. 6, 15
[73] Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi
Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin.
Agent lumos: Unified and modular training for open-source
language agents. In Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1:
Long Papers) , pages 12380–12403, 2024. 1, 2, 5
[74] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi
Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming
Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline
multimodal understanding and reasoning benchmark for ex-
pert agi. In Proceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition , pages 9556–
9567, 2024. 6
[75] Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao
Liu, Yuxiao Dong, and Jie Tang. Agenttuning: En-
abling generalized agent abilities for llms. arXiv preprint
arXiv:2310.12823 , 2023. 1, 2
[76] Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-
Chun Zhu. Raven: A dataset for relational and analogical
visual reasoning. In Proceedings of the IEEE Conference on
Computer Vision and Pattern Recognition (CVPR) , 2019. 13
[77] Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yun-
lin Mao, Daoze Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai,
Ang Wang, Wenmeng Zhou, and Yingda Chen. Swift:a scal-
able lightweight infrastructure for fine-tuning, 2024. 16
[78] Victor Zhong, Caiming Xiong, and Richard Socher. Seq2sql:
Generating structured queries from natural language using
reinforcement learning, 2017. 13
[79] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei.
Visual7w: Grounded question answering in images. In Pro-
ceedings of the IEEE conference on computer vision and pat-
tern recognition , pages 4995–5004, 2016. 13
12

MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning
Supplementary Material
A. Dataset Statistics
Table 6 presents detailed statistics of the datasets utilized to
construct MMAT-1M. These datasets are curated from vari-
ous prominent sources, including Visual CoT [52], LLaV A-
CoT [69], The Cauldron [29], TabMWP [41], and InfoS-
eek [9], collectively contributing to an extensive multimodal
reasoning dataset. Specifically, the table enumerates the
composition, the number of data entries, and the corre-
sponding QA pairs of each sub-dataset. In total, the MMAT-
1M dataset comprises 1,090,263 QA pairs, indicating sub-
stantial coverage and diversity in multimodal tasks.
Category Number of Data Entries Number of QA Pairs
Visual CoT [52]
Birds-200-2011 [60] 10.1k 10.1k
DocVQA [44] 33.5k 33.5k
DUDE [59] 11.7k 11.7k
Flickr30K [49] 135.7k 135.7k
GQA [22] 98.1k 98.1k
InfographicsVQA [45] 15.1k 15.1k
Open images [28] 43.1k 43.1k
SROIE [21] 2.5k 2.5k
TextCap [54] 32.2k 32.2k
TextVQA [55] 18.5k 18.5k
Visual7W [79] 30.5k 30.5k
VSR [32] 3.4k 3.4k
LLaV A-CoT [69]
ShareGPT4V [6] 31.3K 67.9k
ChartQA [43] 17.0k 25.6k
A-OKVQA [43] 16.1K 99.9K
AI2D [25] 11.4k 11.4k
GeoQA+ [4] 11.4k 11.4k
ScienceQA [40] 5.6k 5.6k
DocVQA [44] 4.0k 31.3k
PISC [30] 0.9k 0.9k
CLEVR [24] 0.5k 0.5k
CLEVR-Math [14] 0.5k 0.5k
The Cauldron [29]
HatefulMemes [27] 8.5k 8.5k
Screen2Words [61] 15.7k 15.7k
ST-VQA [3] 17.2k 23.1k
VisText [57] 10.0k 10.0k
WikiSQL [78] 75.0k 86.2k
WTQ [48] 38.2k 44.1k
IconQA [39] 27.3k 29.8k
RA VEN [76] 20.9k 20.9k
Inter-GPS [38] 1.3k 1.8k
TQA [26] 1.5k 6.5k
TabMWP [41]
TabMWP [41] 23.1k 23.1k
InfoSeek [9]
InfoSeek [9] 131.4k 131.4k
Table 6. Detailed statistics of datasets included in MMAT-1M.B. Data Engine
This section introduces the API operators, including scene
graph-based image caption generation. It then details
the prompts and their designs underlying the data engine,
clearly distinguishing between rationale generation and re-
flection components.
Inspired by the CCoT [47], GPT-4o is leveraged to con-
struct a scene graph and derive an image caption, enhancing
semantic understanding and compositional reasoning. The
scene graph prompt and image caption prompt are depicted
in 5.
{QUESTION}For the provided image and its associated question, generate only a scene graph in JSON format that includes the following:1. Objects that are relevant to answering the question2. Object attributes that are relevant to answering the question3. Object relationships that are relevant to answering the questionExample of a valid JSON response:{"objects": ["object_name1", "object_name2", ...],"attributes": [{"name": "object_name", "value": "attribute_value"},...],"relationships": [{"object_name1": {"relation_type":["object_name2"]}},...]}STRICTLY FOLLOW THE JSON RESPONSE FORMAT.Scene Graph Prompt 
Scene Graph: {SCENE_GRAPH}Use the scene graph as context to write a concise description for this image.Image Caption Prompt 
Figure 5. Scene graph and caption generation prompts.
For rationale generation, we guide GPT-4o through a
structured and adaptive multi-stage reasoning process. Dur-
ing inference, the model dynamically invokes multimodal
operators, including Image Caption, Open-V ocabulary Ob-
13

jection Detection (OVD), Optical Character Recognition
(OCR), and Face Detection, while also leveraging retrieval-
augmented generation when needed. Each reasoning step is
explicitly documented in structured JSON format to main-
tain transparency and logical coherence. Figure 6 illus-
trates the comprehensive design of the rationale generation
prompt.
You are an advanced multimodal reasoner. Your task is to reason step by step to answer the user's question using the available operators and rags, finally get the standard answer logically.In each step, decide which operators to call for the next step. You may choose none, one, or multiple operators. The operator results will be provided as input in the following step.If all information proviededby the operators are useless, then use RAG (Retrieval Augmented Generation) to solve the problem. It should be noted that rag only supports text-to-text search, not text-to-image or image-to-image search.Respond in JSON format with the following fields:1. thought: A concise reasoning step with reflection on previous results.2. operators: An array of operators to use next (e.g., "image_caption", "label_bbox", "ocr", "face_detection"). Leave it empty if no operators are needed.3. rag: An dictof RAG queries that can help resolve the issue. (e.g. {"needed": true, "query": "A query can help reasoning and solve the promblem."})4. next_action: Either "continue" if further steps are needed, or "final_answer" if you have the complete solution.Example of a valid JSON response:{"thought": "Identifying key information and reflecting on prior results","operators": ["image_caption"],"rag": {"needed": false, "query": null},"next_action": "continue"}Notice: you are merely using reasoning to approach the standard answer, but you should not use the standard answer directly or use the words like standard answer. The thought should be concise and logical.Data Generation Prompt
Figure 6. Rationale generation prompt.
For reflection, we designed two targeted prompts to en-
hance reasoning robustness. The general reflection prompt
is intended to detect and correct reasoning cheating behav-
iors. Specifically, it prompts GPT-4o to critically exam-ine cases where its reasoning process artificially aligns with
given answers rather than deriving them through genuine in-
ference. The prompt explicitly requests that the model iden-
tify and articulate any logical inconsistencies in its reason-
ing. The detailed structure of the general reflection prompt
is provided in Figure 7.
You are an image analysis and reasoning system. Based on the input image and question, you first determine the approach to solve the problem, then invoke external tools to assist in providing an answer, and ultimately combine the analysis process to give your response. The entire reasoning process takes the form of a multi-round dialogue, with the results as follows: {CONVERSATIONS_WO_ANSWER}As a user, I have noticed that some of your reasoning processes may involve cheating. A typical manifestation is: you think and invoke external tools, but due to the inadequacy of the external tools or the limitations of your own model's capabilities, you actually cannot obtain the correct answer. However, in order to appear as if you can derive the correct answer, you pretend in your final reasoning conclusion that you have derived the correct answer. This constitutes cheating.I hope you can review the above answers to determine if such cheating behavior exists. If it does, please provide the reason.Output format:{"reason": "The reason for the existence of cheating behavior"}The content of "reason" is is why you think the above reasoning process involves logical cheating.If there is no cheating behavior, the output will be empty: {}.General Reflection Prompt
Figure 7. General reflection prompt.
The math reflection prompt specifically targets com-
pleteness issues in mathematical reasoning. It instructs
GPT-4o to carefully inspect its mathematical derivations,
identifying instances where crucial calculation steps might
be omitted. By prompting the model to supplement missing
derivations explicitly, this ensures the integrity and clarity
of mathematical reasoning. Figure 8 depicts the math re-
flection prompt’s structure in detail.
Our MMAT-1M dataset supports both one-turn and
multi-turn reasoning frameworks, each with a dedicated
system prompt.
14

You are an image analysis and mathematical reasoning system that provides answers based on input images and questions in the form of multiple rounds of dialogue:{CONVERSATIONS}. However, as a user, I'm not good at mathematical derivations. If you omit necessary steps in your reasoning process, I may make mistakes in my reasoning and fail to obtain the final answer. Especially, I've noticed that in your answers, from the second-to-last round of dialogue: {CONVERSATION_Q}, to the last round: {CONVERSATION_V}, sometimes the specific mathematical calculation process is omitted. If this is the case, could you please provide additional reasoning to help me understand? If you believe the above answer is already very detailed and complete, then there is no need to add anything.Now, please respond in the following format:{"reason": "Specific reason","thought": "Supplementary reasoning process/null"}Where "reason" is the reason why additional reasoning is needed or not needed, and "thought" is the supplementary reasoning process. If no additional reasoning is needed, it must be "null".Note: 1.the output format must be a json.2.You need to carefully examine whether the logic from the second-to-last step to the final step is sufficient, there is no need to supplement it with additional reasoning for each question.Math Reflection PromptFigure 8. Math reflection prompt.
The one-turn rationale and reflection (ORR) prompt en-
ables the model to complete the entire reasoning process
within a single inference step. This design ensures effi-
ciency while maintaining strong reasoning capabilities. The
structured output format ensures clarity and consistency.
The specific prompt format for ORR is shown in Figure 9.
The rationale and reflection (RR) prompt guides the
model through an iterative reasoning process, dynami-
cally selecting multimodal operators and retrieving external
knowledge when necessary. This approach enhances inter-
pretability and reasoning depth. Figure 10 outlines the spe-
cific format designed for RR.
To assess potential GPT-4o hallucinations introduced
You are an advanced multimodal reasoner. Your task is to reason step by step to answer the user's question using the available operators, finally get the answer logically.Respond in the format like this:<THOUGHT>your reasoning</THOUGHT><ANSWER>The answer is: your answer</ANSWER>ORR System Prompt Figure 9. One-turn rationale and reflection (ORR) prompt.
You are an advanced multimodal reasoner. Your task is to reason step by step to answer the user's question using the available operators and rags, finally get the answer logically.In each step, decide which operators to call for the next step. You may choose none, one, or multiple operators. The operator results will be provided as input in the following step.You can try to use RAG (Retrieval Augmented Generation) to search more useful information.During the reasoning steps, your output format should be as follows:<THOUGHT>reasoning</THOUGHT><OPERATOR>names of the operators</OPERATOR><RAG>query for rag</RAG><SIGNAL>signal of the next action</SIGNAL>Note, if there is no need to call operators or RAG, <OPERATOR> or <RAG> can be omitted from the output.In the final step, your response must be the ultimate answer to the initial question.RR System Prompt 
Figure 10. Rationale and reflection (RR) prompt.
during the rationale and reflection generation stages, we
performed a large-scale quality evaluation of the entire
MMAT-1M dataset. Specifically, we employed the Doubao-
1.5-Vision-Pro-32K model to evaluate all samples against
five criteria: coherence, relevance, accuracy, completeness,
and image-text integration. Results indicate that over 89%
of the samples exhibit high-quality reasoning. Figure 11
presents the prompt used in this evaluation.
C. Training Hyperparameters
In this section, we present the main training parameters
for multiple models. For all models, including Llama-3.2-
11B-Vision-Instruct [46], MiniCPM-V-2.6 [72], and the In-
ternVL2.5 series [11], we adopt the same training configu-
15

You are an experienced evaluator of multimodal reasoning processes. Your task is to assess the quality of a structured multimodal Chain-of-Thought (CoT) reasoning process and assign an overall quality score.Evaluation Criteria:Coherence: Logical and fluent step-by-step reasoning.Relevance: All steps directly contribute to answering the query.Accuracy: Reasoning leads to the correct final answer.Completeness: Proper use of multimodal operators (e.g., Image Caption, OVD, OCR, Face Detection) and Retrieval-Augmented Generation (RAG).Visual Integration: Effective fusion of image and text-based reasoning.Assign a score from 0.0 (poor) to 1.0 (excellent), accurate to one decimal place.Output Format:Provide your evaluation in JSON format:{"quality_score": "A numeric score between 0.0 and 1.0, rounded to one decimal place.","scoring_reason": "A brief explanation for the score."}Reasoning evaluation promptFigure 11. Reasoning evaluation prompt.
ration and use the open-source framework ms-swift [77] for
training. The specific parameters are shown in Table 7.
Parameter Value
train type LoRA
num train epochs 1
train batch size 1
gradient accumulation steps 1
learning rate 4×10−5
weight decay 0.1
max length 16384
torch dtype BF16
seed 42
deepspeed ZeRO-2
Table 7. Configuration of hyperparameters used in training.
D. More Qualitative Examples
In this section, we present additional qualitative examples
highlighting the practical benefits of iterative reasoning,
self-reflection, and one-turn reasoning.Figure 12 illustrates how the model refines its reasoning
to correct an OCR error. The initial OCR result misiden-
tifies the text as “ADEDNI”, but by leveraging contex-
tual understanding and common knowledge, the model cor-
rectly recognizes it as “CALIFORNIA”. This highlights the
model’s ability to detect and correct errors through iterative
reasoning.
Figure 13 showcases how reflection improves mathemat-
ical reasoning. Initially, the model applies a direct calcula-
tion to determine the area of a parallelogram, but realizes its
approach lacks mathematical rigor. Through self-reflection,
it revises its reasoning and correctly applies the sine func-
tion, leading to an accurate computation of the area.
Figure 14 illustrates how the model integrates external
signals such as image caption and object detection to im-
prove decision-making. In a traffic scene, the baseline
model incorrectly determines the direction of the closest car.
However, by analyzing additional image information, the
model correctly identifies that the car is approaching from
the opposite lane, leading to the correct conclusion.
These examples underscore the model’s strengths in er-
ror correction, reasoning refinement, and effective use of
external knowledge for improved decision-making.
E. Limitations and Future Work
Despite its strengths, MMAT-1M has certain limitations.
The reliance on high-quality rationale training data and a
fixed set of tool usage may restrict its generalization ability.
Additionally, while the dataset demonstrates strong perfor-
mance on current benchmarks, further testing is needed to
evaluate its adaptability to more MLLMs and more complex
scenarios. Nevertheless, MMAT-1M sets a new pipeline for
multimodal agent tuning, and its performance on challeng-
ing tasks underscores its potential as a foundational visual
training dataset for future research.
16

Question:what is written in the image?.InternVL-2.5-8B-RR:Q:what is written in the image?Answer the question using a single word or phrase. Please start your reasoning.A:<THOUGHT>To determine the text in the image, using OCR will be the most effective approach.</THOUGHT><OPERATOR>ocr</OPERATOR><SIGNAL>continue</SIGNAL>Q:Here are ocr results from the previous step:-ocr information: ADEDNI Please continue your reasoning.A:<THOUGHT>The OCR result 'ADEDNI' seems to be a misinterpretation.The text might be 'CALIFORNIA' based on the context and common knowledge.</THOUGHT><SIGNAL>end</SIGNAL>Q:Based on the above information, please summarize and give the answer to the initial question: what is written in the image?Answer the question using a single word or phrase.A:california✓✓xFigure 12. Example of iterative rationale result. The OCR recognition is erroneous, but the rationale process corrects the mistake.
17

Question:Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: As shown in the figure, in the parallelogram ABCD, it is known that AB = 6.0, BC = 9.0, ∠B = 30.0, then the area of ​​the parallelogram ABCD is ()\nA.12\nB.18\nC. 27\nD. 54\nAnswerwith the option's letter from the given choices directly.InternVL-2.5-8B-RR:Q:Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: As shown in the figure, in the parallelogram ABCD, it is known that AB = 6.0, BC = 9.0, ∠B = 30.0, then the area of ​​the parallelogram ABCD is ()\nA.12\nB.18\nC. 27\nD. 54\nAnswerwith the option's letter from the given choices directly. Please start your reasoning.A:<THOUGHT>The image shows a parallelogram ABCD with given side lengths and angle. To find the area, we can use the formula: Area = base * height. The base is AB, and the height can be found using the sine of angle B.</THOUGHT>\n<SIGNAL>continue</SIGNAL>Q:No additional operator or rag results were selected in the previous step. Please continue your reasoning.A:<THOUGHT>Wait a bit, the way I was thinking about it might not have enough math backing it up. I'll improve it.</THOUGHT>\n<SIGNAL>reflection</SIGNAL>Q:No additional operator or rag results were selected in the previous step. Please make a reflection.A:<THOUGHT>To calculate the area of the parallelogram ABCD, we use the formula: Area = base ×height. Here, the base is AB = 6.0. The height can be found using the sine of angle B, which is 30°. The height is BC ×sin(30°) = 9.0 ×0.5 = 4.5. Therefore, the area of the parallelogram is 6.0 ×4.5 = 27. This confirms the answer is C.</THOUGHT>\n<SIGNAL>end</SIGNAL>Q:Based on the above information, please summarize and give the answer to the initial question: Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion: As shown in the figure, in the parallelogram ABCD, it is known that AB = 6.0, BC = 9.0, ∠B = 30.0, then the area of ​​the parallelogram ABCD is ()\nA.12\nB.18\nC. 27\nD. 54\nAnswerwith the option's letter from the given choices directly.A:C✓✓✓Figure 13. Example of reflection result. The inclusion of the reflection makes the mathematical derivation more complete and accurate.
18

Question:Is the car closest to us driving in the same direction as us or in the opposition direction from us.A.Same direction.B. Opposite direction.Answer with the option's letter from the given choices directly.InternVL-2.5-8B-Baseline:Q:Is the car closest to us driving in the same direction as us or in the opposition direction from us.A.Same direction.B. Opposite direction.Answer with the option's letter from the given choices directly.A:Ax
✓InternVL-2.5-8B-ORRQ:Is the car closest to us driving in the same direction as us or in the opposition direction from us.A.Same direction.B. Opposite direction.Answer with the option‘s letter from the given choices directly.Use the following API outputs to help answer thequestion:<|begin_api_invoke|>image_caption<|end_api_invoke|><|begin_result_of_api|>A divided road features vehicles traveling in opposite directions, with the closest car approaching from the opposite lane.<|end_result_of_api|><|begin_api_invoke|>label_bbox<|end_api_invoke|><|begin_result_of_api|>{'closest _ car': [[0.0, 0.46, 0.2, 0.72]], 'road _ lane': [[0.17, 0.47, 1.0, 0.99]]}<|end_result_of_api|>A:<THOUGHT>To determine the direction of the car closest to us, I need to analyze the image for the orientation of the vehicles. The image caption indicates that the closest car is approaching from the opposite lane, confirming it is driving in the opposite direction.</THOUGHT><ANSWER>The answer is: B</ANSWER>Figure 14. Example of one-turn rationale result. It correctly answers the question with the aid of image caption information.
19