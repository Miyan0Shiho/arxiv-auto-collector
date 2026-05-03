# ChipLingo: A Systematic Training Framework for Large Language Models in EDA

**Authors**: Lei Li, Xingwen Yu, Jianguo Ni, Junxuan Zhu, Jieqiong Zhang, Jian Zhao, Zhi Liu

**Published**: 2026-04-30 04:35:43

**PDF URL**: [https://arxiv.org/pdf/2604.27415v1](https://arxiv.org/pdf/2604.27415v1)

## Abstract
With the rapid advancement of semiconductor technology, Electronic Design Automation (EDA) has become an increasingly knowledge-intensive and document-driven engineering domain. Although large language models (LLMs) have shown strong general capabilities, applying them directly to EDA remains challenging due to limited domain expertise, cross-tool knowledge confusion, and degraded retrieval-augmented generation (RAG) performance after domain training. To address these issues, this paper presents ChipLingo, a systematic training pipeline for domain-adapted LLMs tailored to EDA scenarios.
  ChipLingo consists of three stages: domain corpus construction with multi-source data curation and QA augmentation, domain-adaptive pretraining with comparisons of different parameter training strategies, and instruction alignment with RAG scenario training under diverse retrieval conditions. We also curate an internal benchmark, EDA-Bench, covering representative EDA tool scenarios, with plans for public release.
  Experiments show that ChipLingo-8B achieves 59.7% accuracy on EDA-Bench, outperforming the same-scale base model and some larger general-purpose models. ChipLingo-32B reaches 70.02%, approaching leading closed-source commercial models. Further analysis shows that QA augmentation improves domain performance, Partial FT offers a better balance between adaptation and general capability retention than LoRA, and explicit RAG scenario training mitigates the decline in retrieval utilization after domain training. These results demonstrate the practical value of systematic domain training for knowledge-intensive EDA tasks and provide a foundation for future EDA agents and external-knowledge-driven systems.

## Full Text


<!-- PDF content starts -->

ChipLingo: A Systematic Training Framework for
Large Language Models in EDA
Lei Li Xingwen Yu Jianguo Ni Junxuan Zhu
Jieqiong Zhang Jian Zhao Zhi Liu
Ickylin AI Team
info@ickylin.com
https://www.ickylin.com
Abstract
With the rapid advancement of semiconductor technology, the complexity of Electronic De-
sign Automation (EDA) tools has increased substantially, making EDA a highly knowledge-
intensive and document-driven engineering domain. Although large language models (LLMs)
have demonstrated strong capabilities in general tasks, their direct application to the EDA
domain still faces significant challenges, including insufficient domain expertise, cross-tool
knowledge confusion, and degradation of retrieval-augmented generation (RAG) capabilities
after domain training. To address these challenges, this paper presents ChipLingo, a system-
atic training pipeline for domain-adapted large language models tailored to EDA scenarios.
The proposed pipeline consists of three stages. First, we construct domain-specific cor-
pora through multi-source data curation and question-answering (QA) augmentation. Sec-
ond, during the domain-adaptive pretraining phase, we compare different parameter training
strategies and examine their effects on domain adaptation performance and selected general
capability benchmarks. Third, through instruction alignment and RAG scenario training
targeting diverse retrieval conditions, we enhance the model’s ability to leverage external
knowledge. To evaluate model performance on EDA question-answering tasks, we curate an
internal benchmark called EDA-Bench, covering multiple representative EDA tool scenarios,
which is planned for public release.
Experimental results demonstrate that ChipLingo-8B achieves 59.7% accuracy on EDA-
Bench, outperforming the base model of the same scale and surpassing some larger general-
purpose models. ChipLingo-32B reaches 70.02% accuracy, approaching the performance of
leading closed-source commercial models on this benchmark. Furthermore, we observe that
under the current experimental settings, QA augmentation contributes to improved domain
performance, Partial FT achieves a better trade-off between domain adaptation and general
capability retention compared to LoRA, and explicit RAG scenario training can mitigate the
decline in retrieval information utilization that occurs after domain training. These results
indicate that a systematic domain training pipeline provides practical value for enhancing
LLM performance on knowledge-intensive EDA tasks, while also laying the foundation for
building EDA agents or harness capabilities that rely on external knowledge.
1arXiv:2604.27415v1  [cs.LG]  30 Apr 2026

Keywords:Large Language Models; Electronic Design Automation (EDA); Continued Pre-
training; Retrieval-Augmented Generation (RAG); Knowledge-Intensive Tasks
1 Introduction
1.1 Background
With the rapid advancement of semiconductor technology, the scale and complexity of modern
chip systems continue to grow. State-of-the-art chips now contain billions or even tens of billions
of transistors, with design workflows spanning multiple stages including logic synthesis, physi-
cal design, timing analysis, verification, and power optimization. To address these challenges,
Electronic Design Automation (EDA) tools have become indispensable infrastructure [1].
However, the EDA ecosystem exhibits substantial complexity. Different design stages typi-
cally rely on distinct specialized tools, each providing extensive command interfaces, parameter
configurations, and design constraint mechanisms. In practice, engineers must frequently consult
large volumes of technical documentation, user manuals, and community forums to understand
tool behavior, debug design issues, and optimize design flows. Moreover, significant differences
exist among EDA tools in terms of command syntax, design workflows, and configuration ap-
proaches, further raising the learning curve and barriers to entry [2].
Knowledge-Intensive Characteristics.In engineering practice, substantial knowledge
exists in the form of documentation, accumulated experience, and question-answering exchanges,
distributed across diverse sources. This knowledge-intensive characteristic makes EDA a typi-
cal “document-driven” engineering domain. For more sophisticated EDA intelligent assistants,
agents, or harness systems, models must not only answer questions but also maintain stable
knowledge alignment across external documents, tool feedback, and task workflows. Therefore,
reliable retrieval utilization capability can be viewed as a foundational ability for deploying
higher-level EDA intelligent systems.
1.2 Problem Definition and Challenges
1.2.1 Limitations of General LLMs in the EDA Domain
Large Language Models (LLMs) have achieved significant progress in natural language under-
standing, code generation, and knowledge-based question answering. General-purpose LLMs
have demonstrated strong knowledge representation and task generalization capabilities across
multiple domains [4]. Nevertheless, directly applying general LLMs to answer EDA domain
questions still faces notable limitations:
•Insufficient Domain Expertise: General models typically lack adequate EDA-specific
knowledge, leading to inaccurate or overly generic responses to domain-specific questions.
2

•Cross-Tool Knowledge Confusion: Due to significant differences in command inter-
faces and design constraints across EDA tools, models may confuse terminology and usage
patterns between different tools.
•Lack of Real-Time Knowledge: EDA domain questions often require reference to specific
documentation or tool version information, necessitating the ability to retrieve up-to-date
knowledge.
1.2.2 RAG Capability Degradation After Domain Training
Given that EDA domain knowledge updates frequently and relies heavily on documentation,
Retrieval-Augmented Generation (RAG) is considered an important technical approach for
addressing domain knowledge challenges [4, 5, 6]. However, after domain-specific continued
training, model RAG capabilities may exhibit notable degradation. After acquiring domain
knowledge, models tend to rely more on parametric knowledge for answering questions rather
than utilizing retrieved external knowledge. This phenomenon is particularly pronounced in
knowledge-intensive domains and poses new challenges for RAG system stability. Furthermore,
this issue affects not only single-turn question-answering scenarios but also directly impacts EDA
agent or harness systems that depend on external documents, tool feedback, and contextual state
for decision-making.
1.3 Main Contributions
This paper presentsChipLingo, developed around knowledge-intensive EDA question-answering
scenarios. The main contributions include:
1. We construct a systematic LLM training pipeline for the EDA domain, covering data prepa-
ration, domain-adaptive pretraining, instruction alignment, and RAG scenario training.
2. Under the current experimental settings, we systematically compare the effects of QA aug-
mentation, LoRA, Full FT, and Partial FT training strategies on EDA domain adaptation
and selected general capability benchmarks.
3. We observe and quantify changes in model retrieval information utilization after domain
training, demonstrating that RAG scenario training targeting diverse retrieval conditions
can mitigate this issue on the current benchmark.
4. We curate EDA-Bench, a benchmark covering multiple representative EDA tool scenar-
ios for evaluating model performance on EDA engineering question-answering tasks. This
benchmark is currently undergoing further curation and preparation for release.
3

2 Related Work
2.1 Domain-Specific Large Language Models
With the successful application of LLMs in natural language processing tasks, an increasing num-
ber of studies have explored how to build domain-specific large language models [7, 12]. These
works demonstrate that continued pretraining or task adaptation on domain-specific corpora
can significantly improve model performance in specialized scenarios.
In the EDA and chip design domain, prior works have explored domain-specific models for
chip design scenarios. ChipNeMo [3] systematically investigated domain-adaptive tokenizers,
continued pretraining, instruction alignment, and retrieval augmentation for chip design. Addi-
tionally, ChipExpert [2] and customized RAG/benchmark work for EDA documentation QA [8]
further demonstrate that general LLMs require domain adaptation to adequately handle EDA
tasks. Meanwhile, VerilogEval [9] and similar works have constructed more focused evaluation
frameworks from the perspective of hardware code generation, indicating that benchmarks and
application scenarios for chip design tasks are becoming increasingly refined.
Compared to these works, the EDA domain exhibits more complex knowledge structures.
EDA knowledge exists not only in technical documentation but is also distributed across engi-
neering experience, tool commands, and design workflows in various knowledge forms.
2.2 Domain Adaptation and Continued Pretraining
Domain Adaptation is an important approach for building domain-specific models [10, 11, 7]. A
common method involves performing continued pretraining on domain corpora based on general
foundation models to enhance model understanding of domain knowledge [12, 7, 10].
Inspired by research such as RAFT [10], this paper explores a QA-augmented domain contin-
ued pretraining strategy. Unlike traditional approaches that use only documents for pretraining,
we introduce question-answering format data during the pretraining phase, enabling the model
to learn the relationship between knowledge and tasks earlier in training.
2.3 Parameter-Efficient Fine-Tuning Methods
As large language model sizes continue to grow, the computational cost of full-parameter fine-
tuning has become a significant limiting factor for model deployment [16, 17]. Researchers have
proposed a series of parameter-efficient fine-tuning methods, including Adapter [13], Prefix-
Tuning [14], and LoRA [15].
However, recent research indicates that parameter-efficient fine-tuning methods may have
certain limitations in knowledge-intensive tasks [18]. Due to constraints on the expressiveness
of low-rank updates, models may struggle to fully absorb domain knowledge when tasks require
learning large amounts of fine-grained knowledge.
4

2.4 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has emerged as an important method for enhancing
LLM knowledge utilization capabilities in recent years [4, 19, 20, 21]. Researchers have pro-
posed various approaches to improve RAG, including Self-RAG [19], GraphRAG [20], and Hy-
bridRAG [21].
In this study, we further observe an important phenomenon: after domain training, model
RAG capabilities may exhibit notable degradation. This phenomenon is particularly pronounced
in knowledge-intensive domains and poses new challenges for RAG system design.
2.5 Distinctions from Existing Work
Compared to existing work, the main distinctions of this paper include:
•We propose a systematic LLM training framework for the EDA domain;
•We explore the role of QA-augmented pretraining in domain model training;
•We systematically analyze the limitations of parameter-efficient fine-tuning in knowledge-
intensive domains;
•We investigate the phenomenon of RAG capability degradation during domain training and
propose corresponding training strategies.
Furthermore, this paper views retrieval information utilization capability as a critical foun-
dational ability for higher-level EDA intelligent systems. In other words, the RAG grounding
problem addressed in this paper relates not only to single-turn question-answering quality but
also directly determines whether subsequent agent or harness systems can stably leverage exter-
nal knowledge and tool feedback in real engineering workflows.
3 Method
3.1 Problem Formalization
Before introducing the ChipLingo training framework, we first provide formal definitions for
EDA domain question-answering tasks and the model training process.
Definition 3.1(EDA Question-Answering Task).Given an EDA domain questionqand an
optional set of retrieved documentsC={c 1, c2, . . . , c k}, a large language modelM θis required
to generate an answer sequencea=M θ(q, C) such thatais semantically equivalent to the
ground-truth answera∗. WhenC=∅, the model relies solely on parametric knowledge for
answering; whenC̸=∅, the model must incorporate retrieved context to generate the answer.
Definition 3.2(RAG Gain Rate).LetC+
idenote the retrieved context containing the correct
answer. The RAG gain rate is defined as the improvement in model accuracy when provided
5

with correct retrieved context:
∆rag=1
NNX
i=1I 
Mθ(qi, C+
i) =a∗
i
−1
NNX
i=1I 
Mθ(qi,∅) =a∗
i
(1)
wherea∗
iis the ground-truth answer andI(·) is the indicator function. A higher ∆ ragindicates
that the model can more effectively utilize correct retrieved information.
Definition 3.3(Noise Impact Degree).LetC−
idenote retrieved context irrelevant to question
qi. The noise impact degree is defined as the change in model accuracy when irrelevant retrieved
context is introduced, relative to the no-retrieval condition:
∆noise=1
NNX
i=1I 
Mθ(qi, C−
i) =a∗
i
−1
NNX
i=1I 
Mθ(qi,∅) =a∗
i
(2)
This metric measures the degree of interference caused by irrelevant retrieved context on model
performance. A ∆ noisecloser to 0 indicates stronger model resistance to external noise.
Based on the above definitions, the three-stage training objectives of ChipLingo can be
formalized as:
θ1= arg min
θLPT 
Mθ, Dpt
(3)
θ2= arg min
θLSFT 
Mθ1, Dsft
(4)
θ∗= arg min
θLRAG 
Mθ2, Drag
(5)
Specifically, the loss functions for each stage are defined as follows:
•Domain-Adaptive Pretraining Stage: The standard causal language modeling loss is
employed, maximizing the log-likelihood over the pretraining corpusD pt:
LPT=−E (x)∼D pt
|x|X
t=1logP Mθ 
xt|x<t
 (6)
•Instruction Alignment Stage: Supervised fine-tuning loss is applied, optimizing the gen-
eration probability of final answers based on question-answer pairs (q, a)∈D sft. Auxiliary
reasoning annotations, when available in the raw instruction data, are not used as explicit
generation targets:
LSFT=−E (q,a)∼D sft
|a|X
t=1logP Mθ 
at|q, a <t
 (7)
•RAG Scenario Fine-Tuning Stage: Under conditions that include retrieved contextC,
6

the answer generation probability is optimized using training data (q, C, a)∈D rag:
LRAG=−E (q,C,a)∼D rag
|a|X
t=1logP Mθ 
at|q, C, a <t
 (8)
For thepartial parameter training strategy, let the complete model parameters beθ.
We freeze the parameters of several bottom layers, denoted asθ freeze⊂θ, and update only a
selected subset of the remaining parametersθ train⊂θ\θ freeze. During training, onlyθ trainis
updated:
θ(t+1)
train=θ(t)
train−η∇ θtrainL(9)
The overall architecture of the ChipLingo framework is illustrated in Figure 1.
Figure 1Overview of the ChipLingo three-stage training framework. Stage 1 (Domain-Adaptive Pre-
training) constructs domain pretraining corpora through multi-source data fusion and augmentation
strategies, employing partial parameter training to obtain a model with EDA domain expertise. Stage
2 (Instruction Alignment Training) leverages high-quality QA data distilled and rewritten from strong
models, using supervised fine-tuning to enable instruction comprehension and task execution capabilities.
Stage 3 (RAG Scenario Fine-Tuning) introduces training data covering various RAG scenarios including
correct retrieval, irrelevant retrieval, and incomplete retrieval, enabling the model to achieve a balance
between utilizing external knowledge and filtering noise.
3.2 Domain Data Preparation
EDA domain data is characterized by dispersed sources, complex structures, and diverse formats.
To construct high-quality training data, we systematically curated EDA documentation and
engineering QA data, and expanded the training corpus through various data augmentation
7

methods. The overall data composition and corpus organization are illustrated in Figure 2.
(a)Source distribution of pretraining raw data
(b)Organization from raw data to training corpora
Figure 2ChipLingo data composition and training corpus organization
3.2.1 Data Sources and Cleaning
Training data primarily comes from the following categories:
•EDA tool technical documentation
•Engineer question-answering records
•Technical papers and educational materials
•Tool usage examples and script documentation
The overall training corpus scale exceeds200,000 pages of technical documentation,
coveringmore than 50 EDA tools, spanning multiple design stages including logic synthesis,
physical implementation, simulation verification, and design-for-test analysis.
8

3.2.2 Data Augmentation Strategies
Relying solely on raw documentation data is often insufficient to form high-quality training
corpora. Therefore, we explored multiple data augmentation methods:
1.QA Generation: Automatically generating question-answer pairs from documentation.
2.Document Rewriting: Semantically-preserving rewriting of document content to increase
data diversity.
3.Cloze Generation: Generating fill-in-the-blank tasks from technical descriptions.
4.Multiple Choice Generation: Constructing training data in multiple-choice or single-
choice question formats.
Through comparative experiments on different augmentation strategies, we found thatQA-
format data exhibits high efficiency for domain knowledge learning, and thus consti-
tutes a significant proportion of the training data.
3.3 Domain-Adaptive Pretraining
After completing data preparation, we perform domain-adaptive pretraining on the base model
to enable it to acquire EDA-related knowledge.
3.3.1 QA-Augmented Pretraining
Traditional domain-adaptive pretraining typically uses only document text for training, with
question-answering capabilities learned during subsequent supervised fine-tuning stages. How-
ever, in knowledge-intensive domains, this training approach may struggle to establish connec-
tions between knowledge and tasks.
Therefore, we introduceQA-format dataduring the pretraining phase, enabling the model
to learn both domain knowledge and its application simultaneously. Experimental results in-
dicate that, under the current experimental settings, incorporating QA data leads to improved
performance on EDA tasks.
3.3.2 Partial Parameter Training Strategy
Domain-adaptive training often leads to degradation of model general capabilities. To achieve a
balance between domain knowledge acquisition and general capability preservation, we explored
apartial parameter training strategy. In this strategy, we freeze the parameters of several
bottom layers and update only a selected subset of the remaining parameters, thereby reducing
the impact of domain training on general capabilities.
9

3.3.3 Limitations of Parameter-Efficient Fine-Tuning
During experiments, we also compared the performance of full-parameter training and parameter-
efficient fine-tuning methods on EDA tasks. Experimental results indicate that in knowledge-
intensive tasks, parameter-efficient fine-tuning methods may underperform compared to full-
parameter training.
A possible explanation is that EDA knowledge possesses highly complex structures and high
information density, while low-rank updates may struggle to express such complex knowledge
representations. Therefore, full-parameter training still maintains certain advantages in specific
domain tasks.
3.4 Instruction Alignment and RAG Training
After completing domain-adaptive pretraining, we train the model’s instruction comprehension
capabilities through supervised fine-tuning and introduce retrieval scenario data to enhance the
model’s RAG capabilities.
3.4.1 Instruction Data Construction
The construction of instruction training data is a critical component in the ChipLingo framework
that bridges domain knowledge and task execution capabilities. This process consists of three
steps: raw QA collection, strong model distillation and rewriting, and quality filtering with
formatting.
Raw QA Collection.We collected a large volume of engineer QA records from real engi-
neering environments, covering multiple EDA tool scenarios including logic synthesis, physical
implementation, simulation verification, and design-for-test. These raw records possess high do-
main value but often suffer from colloquial expressions, missing context, and incomplete answers,
making them unsuitable for direct use in supervised fine-tuning.
Strong Model Distillation and Rewriting.To improve data quality, we utilized high-
performance language models to distill and rewrite the raw QA data. Specifically, we input raw
questions into strong models and requested structured reference answers that explicitly include
reasoning processes. Additionally, we completed ambiguous question formulations and expanded
overly brief answers, ensuring each sample contains clear question descriptions, complete rea-
soning chains, and accurate final answers. Through this step, the professional value of raw QA
data is preserved while data format and expression quality are improved.
Quality Filtering and Formatting.We conducted multiple rounds of filtering on the
distilled data, removing samples that were too short, unrelated to the EDA domain, or had low
answer credibility. The final instruction dataset containsapproximately 40,000 high-quality
samples, with each sample uniformly formatted as a triple (q i, ri, ai), whereq iis the question,
riis the reasoning process, anda iis the final answer. During instruction alignment training,
10

we use only the question-answer pair (q i, ai) as the supervised target, whiler iis retained as an
auxiliary annotation for quality control and analysis. This dataset is used not only for instruction
alignment training but also provides foundational corpora for subsequent RAG scenario training.
3.4.2 RAG Capability Degradation Phenomenon
During experiments, we observed that after domain training, the model’s RAG capabilities may
exhibit notable degradation. Specifically, after acquiring domain knowledge, the model tends to
rely more on parametric knowledge for answering while ignoring retrieved document information.
This phenomenon can be understood as the model’sparametric bias. As domain training
progresses, the model gradually acquires substantial domain knowledge, thus becoming more
inclined to rely on internal knowledge when answering questions rather than utilizing external
retrieval results. This leads to situations where model accuracy under correct retrieval context
may actually be lower than under no-retrieval conditions.
3.4.3 RAG Scenario Training Method
To mitigate RAG capability degradation, we designed a set ofretrieval scenario training
data. This data simulates various RAG application scenarios:
•Correct Knowledge Retrieved: The model learns to reference context for answer gen-
eration.
•Irrelevant Knowledge Retrieved: The model learns to ignore noise and rely on para-
metric knowledge.
•Incomplete Retrieval Results: The model learns to perform joint reasoning by combin-
ing context and parametric knowledge.
Algorithm 1 describes the construction process for RAG scenario training data.
Algorithm 1RAG Scenario Training Data Construction Algorithm
Input:Domain QA datasetD qa={(q i, ai)}N
i=1, retrieval corpusR
Output:RAG scenario training datasetD rag
1:D rag← ∅
2:foreach (q, a)∈D qado
3:C rel←Retrieve(R, q, k= 3)▷Retrieve relevant documents
4:D rag←D rag∪ {(q, C rel, a)}▷Scenario 1: Correct knowledge
5:C irr←SampleIrrelevant(R, q)▷Sample irrelevant documents
6:D rag←D rag∪ {(q, C irr, a)}▷Scenario 2: Irrelevant knowledge
7:C partial←Subsample(C rel,ratio = 0.5)▷Subsample partial documents
8:D rag←D rag∪ {(q, C partial , a)}▷Scenario 3: Incomplete knowledge
9:end for
10:returnD rag
11

3.5 Retrieval System Design
During inference, ChipLingo integrates a retrieval system to obtain EDA-related documents.
The retrieval system employs a hybrid retrieval strategy, combining semantic vector retrieval
and keyword-based retrieval methods to improve recall. Additionally, the system selects the
most relevant document segments through a document reranking mechanism and provides them
as context input to the model for generating final answers.
Through this approach, the model can leverage both parametric knowledge and access ex-
ternal document information when needed.
4 Experiments
4.1 Training Data and Implementation Details
ChipLingo’s training data primarily comes from EDA technical documentation, engineer QA
records, and related technical materials. For data augmentation, we explored multiple ap-
proaches including document rewriting, QA generation, cloze generation, and multiple-choice
generation. Experiments demonstrate thatcombining multiple augmentation methods
yields the best results, with QA generation being particularly effective for improving domain
question-answering capabilities.
In our experiments, we also observed that as training data scale increased from approximately
260k chunks to approximately 400k chunks, model performance on EDA tasks continued to
improve, though with diminishing gains, indicating marginal returns on data scale for model
performance.
4.2 Evaluation Benchmark: EDA-Bench
To evaluate model capabilities in EDA engineering scenarios, we constructed theEDA-Bench
evaluation benchmark. EDA-Bench contains thousands of questions from real engineering sce-
narios, uniformly organized as short-answer questions. These questions primarily involve tool
command usage, design flow comprehension, and common troubleshooting tasks.
The evaluation questions coverfour typical EDA tool categoriesin chip design workflows:
•Logic Synthesis Tools: Used for converting RTL descriptions to gate-level netlists;
•Physical Implementation Tools: Used for placement, routing, and timing optimization;
•Simulation Verification Tools: Used for functional verification and design behavior
analysis;
•Design-for-Test (DFT) Tools: Used for testability design and test pattern generation.
12

4.3 Evaluation Methodology
EDA-Bench test questions are all short-answer format, making traditional string matching meth-
ods inadequate for accurate answer quality assessment. Therefore, we adopt theLLM-as-a-
Judgeautomatic evaluation method, a paradigm that has been systematically studied and
validated in works such as MT-Bench, Chatbot Arena, and G-Eval [22, 23]. Specifically, we
use a high-performance language model as the evaluator to judge model-generated answers.
Evaluator inputs include: the question, ground-truth answer, and model-predicted answer. The
evaluation model’s task is to determine whether the predicted answer is consistent with the
ground-truth answer, outputting abinary classification result of correct or incorrect.
To verify the reliability of the automatic evaluation method, we conducted manual review
on a subset of samples, and results indicate high consistency between automatic evaluation and
human judgment.
The related evaluation framework and dataset are currently undergoing curation
and review, with plans for public release as an independent research contribution
to promote the development of large language model research in the EDA domain.
4.4 Baseline Models
In experiments, we compare ChipLingo with various general-purpose and industry-specific mod-
els. ChipLingo models are trained based on theQwen3 series models[27], with two model
scales: ChipLingo-8B and ChipLingo-32B. Baseline models include mainstream open-source
large language models, strong open-source models, and industry-leading commercial models.
All models are evaluated on the same EDA-Bench benchmark.
4.5 Main Experimental Results
4.5.1 Overall Performance Comparison
Table 1 presents the overall performance of different models on EDA-Bench.
Table 1Overall performance of different models on EDA-Bench
Model Parameters EDA-Bench Accuracy
Qwen3-8B 8B 26.85%
Qwen3-32B 32B 36.30%
DeepSeek-v3.2 671B (37B active) 56.28%
ChipLingo-8B 8B 59.70%
ChipLingo-32B 32B 70.02%
GPT-5.4 – 72.35%
Claude-Sonnet-4.5 – 71.11%
The proposed ChipLingo models demonstrate strong domain adaptability in semiconduc-
tor professional task evaluation. Specifically, ChipLingo-8B with only 8B parameters achieves
13

59.7% accuracy, substantially outperforming the same-scale base model Qwen3-8B (26.85%) and
surpassing the larger general-purpose model DeepSeek-v3.2 (56.28%). Meanwhile, ChipLingo-
32B reaches 70.02%, significantly improving over the same-scale general base model Qwen3-32B
(36.30%), with results approaching Claude-Sonnet-4.5 (71.11%).
4.5.2 Per-Tool Performance Analysis
Figure 3Performance comparison across different EDA tool categories
Figure 3 presents representative evaluation set results extracted by tool category, illustrating
relative performance trends across different tool scenarios. Results indicate that ChipLingo
demonstrates overall superior performance across multiple tool task types, with the most notable
improvements in DFT (Design-for-Test) related tasks. It should be noted that gains across
different tool categories are not entirely uniform; in certain tool subsets, model improvements are
relatively limited, suggesting that these scenarios still require more targeted corpus construction
and training strategies.
4.6 Ablation Studies
To understand the effects of different training strategies, we conducted multiple ablation exper-
iments. All ablation experiments below are based on Qwen3-8B.
14

4.6.1 Effectiveness of QA-Augmented Pretraining
Figure 4Impact of different augmentation strategies on domain performance
Experimental results (Figure 4) indicate that introducing multi-format constructed data
during the pretraining phase helps improve the model’s domain knowledge comprehension ca-
pabilities.
Figure 5Effect of multi-augmentation strategy pretraining on EDA domain learning
Comparing standard document pretraining and multi-augmentation strategy pretraining (All
Aug) training dynamics over 55,000 training steps, the curves show that multi-augmentation
strategies achieve higher final accuracy under current settings, while also exhibiting faster con-
vergence and lower training loss (0.24 vs 0.51).
15

Table 2Comparison of different parameter training strategies on EDA domain performance and general
capability retention
Method on Qwen3-8B EDA-Bench IFEval SimpleQA HumanEval General Avg.
Base 26.85 87.6 36.6 87.8 70.7
LoRA 46.8 85.2 32.4 83.6 67.1
Full FT 61.082.1 28.7 79.2 63.3
Partial FT 59.7 85.8 33.8 84.5 68.0
4.6.2 Parameter Training Strategy Comparison
Table 2 presents comparative results of different parameter training strategies on EDA-Bench
and three general capability benchmarks. IFEval [24] evaluates instruction-following capability,
SimpleQA [25] evaluates short-form factual question-answering capability, and HumanEval [26]
evaluates code generation capability. LoRA, while providing some domain performance improve-
ment over the base model, still significantly underperforms full-parameter and partial-parameter
training on EDA tasks. Full-parameter training achieves the highest accuracy on EDA-Bench,
indicating stronger expressiveness for domain knowledge absorption; however, it simultaneously
exhibits more significant performance degradation on IFEval, SimpleQA, and HumanEval. In
comparison, partial-parameter training achieves results close to full-parameter training on EDA-
Bench while maintaining better capability retention across all three general benchmarks. These
results indicate that under current experimental settings, partial-parameter training demon-
strates a more favorable empirical trade-off between domain adaptation effectiveness and general
capability preservation.
4.6.3 RAG Training Effect Analysis
Table 3 presents quantified results of RAG capability degradation and recovery. For the base
model Qwen3-8B, the relative improvement under the +correct retrieval condition is +7.3. After
domain-adaptive pretraining (DAP), this value drops to−5.5, indicating that providing correct
retrieval context actually degrades model performance. After subsequent supervised fine-tuning
(+DAP+SFT), the relative improvement under the +correct retrieval condition remains negative
at−3.8. After RAG scenario training, the relative improvement under the +correct retrieval
condition recovers to +5.1, while performance loss under the +irrelevant retrieval condition
improves from−4.6 to−2.3, indicating that model utilization of correct retrieval information is
restored and demonstrates certain noise robustness under irrelevant retrieval conditions.
16

Table 3Quantified comparison of RAG capability degradation and recovery
Model No Retrieval +Correct Retrieval +Irrelevant Retrieval
Qwen3-8B 24.5% 31.8% (+7.3) 23.1% (−1.4)
+DAP only 48.2% 42.7% (−5.5) 43.0% (−5.2)
+DAP+SFT 52.1% 48.3% (−3.8) 47.5% (−4.6)
+DAP+SFT+RAG 59.7% 64.8% (+5.1) 57.4% (−2.3)
Figure 6RAG comparison during fine-tuning process
As shown in Figure 6, without RAG training (+DAP+SFT), the model still exhibits negative
gain under the +correct retrieval condition. After introducing RAG training with multiple
scenario templates (+DAP+SFT+RAG), the model recovers positive gain under the +correct
retrieval condition and shows better robustness under the +irrelevant retrieval condition. In
addition, RAG training leads to a more stable convergence process.
4.7 Experimental Summary
Based on comprehensive experimental results, we draw the following important conclusions:
1. Under current experimental settings, domain-adaptive pretraining can improve model per-
formance on EDA tasks.
2. Experimental results indicate that introducing QA data during the pretraining phase helps
enhance domain knowledge capabilities.
3. Under current knowledge-intensive task settings, full-parameter training or partial-parameter
training outperforms parameter-efficient fine-tuning methods such as LoRA.
4. Multi-tool joint training can yield certain synergistic effects, but may also cause tool knowl-
edge confusion.
17

5. Experiments demonstrate that domain training may lead to RAG capability degradation,
while explicit RAG scenario training helps restore this capability.
These results support the effectiveness of the ChipLingo framework for EDA domain tasks.
5 Analysis and Discussion
5.1 Parameter Training Strategies and Knowledge Expressiveness
During domain-adaptive pretraining, we systematically examined the impact of different param-
eter training strategies on model performance. Experimental results indicate that as the model
absorbs EDA domain knowledge, its general capabilities often exhibit certain degrees of degra-
dation, and parameter-efficient fine-tuning methods significantly underperform full-parameter
training in such knowledge-intensive tasks. These observations reveal an inherentcapability
competitionwithin the model’s parameter space when simultaneously encoding general and
domain-specific knowledge.
From a learning mechanism perspective, large language models must represent massive
amounts of knowledge within a finite parameter space. When domain data occupies a large
proportion, the model gradually allocates portions of its parameter resources to encode domain
knowledge, thereby compressing the representation space for original general knowledge. Fur-
thermore, EDA knowledge is characterized by abundant technical details, complex command
structures, and fine-grained knowledge granularity, requiring the model to possess sufficient
knowledge expressiveness. Methods such as LoRA rely on low-rank matrix updates to model
weights, and their expressiveness may be insufficient to adequately capture such complex domain
knowledge representations.
To mitigate these issues, we adopt apartial parameter training strategythat freezes
the parameters of several bottom layers and updates only a selected subset of the remaining
parameters, enabling the model to learn domain knowledge while preserving original language
capabilities. Experimental results indicate that this strategy achieves performance close to full-
parameter training. Based on comprehensive considerations of training efficiency, resource costs,
and deployment feasibility, we ultimately adopt this approach for model training.
5.2 Knowledge Transfer and Confusion in Multi-Tool Training
A distinctive characteristic of the EDA domain is its complex tool ecosystem. Different tools
typically correspond to different design stages while sharing certain foundational concepts, such
as design constraints, timing analysis, and layout structures.
During experiments, we observed thatmulti-tool training exhibits certain knowledge
transfer effects. When the model simultaneously learns multiple EDA tools of similar types,
overall performance tends to improve. However, multi-tool training also introduces new chal-
18

lenges. In certain cases, the model uses commands from one tool to answer questions about
another tool, manifesting as cleartool knowledge confusion.
5.3 Causes and Remediation of RAG Degradation After Domain Training
During experiments, we observed that after domain-adaptive training, the model’sRAG ca-
pability may exhibit degradation. Specifically, even when the retrieval system returns
documents containing correct answers, the model may still ignore this information and directly
generate answers based on its parametric knowledge.
This phenomenon can be understood as the model’sparametric bias. As domain training
progresses, the model gradually acquires substantial domain knowledge, thus becoming more
inclined to rely on internal knowledge when answering questions rather than utilizing external
retrieval results. This manifests as negative relative improvement under +correct retrieval con-
ditions, meaning that accuracy when provided with correct retrieval context is actually lower
than accuracy under no-retrieval conditions.
To mitigate this issue, we further introduced diverseRAG scenario training dataduring
the supervised fine-tuning stage, exposing the model to various input scenarios including correct
retrieval, irrelevant retrieval, and incomplete retrieval during training. Experimental results
indicate that this strategy can restore the model’s ability to utilize external retrieval informa-
tion to a certain extent: after RAG scenario training, the model’s relative improvement under
+correct retrieval conditions recovers from negative to positive values, while performance loss
under +irrelevant retrieval conditions is reduced. This demonstrates that in knowledge-intensive
domains, RAG capability does not naturally strengthen with domain knowledge training but
requires targeted training mechanisms for maintenance. More importantly, such grounding capa-
bility is not only a requirement for traditional RAG question-answering modules but also serves
as the foundation for subsequent EDA agent or harness systems to stably leverage documents,
retrieval results, and tool feedback.
6 Conclusion
This paper presentsChipLingo, a large language model training pipeline combining domain-
adaptive pretraining, instruction alignment training, and RAG scenario training, developed
around knowledge-intensive Electronic Design Automation (EDA) question-answering scenarios.
Experimental results demonstrate that this pipeline can improve model comprehension and
question-answering performance for EDA-related knowledge.
To evaluate model performance on EDA tasks, we curatedEDA-Bench, an evaluation
benchmark. Experimental results indicate that under current evaluation settings, models with
domain training show improved performance on EDA-Bench compared to general-purpose mod-
els. ChipLingo-8B outperforms some larger general-purpose models on this benchmark, while
19

ChipLingo-32B achieves results approaching strong closed-source commercial models.
During training, we also observed several phenomena worthy of further investigation: under
current experimental settings, QA augmentation contributes to improved domain performance,
Partial FT demonstrates favorable empirical balance between domain adaptation effectiveness
and selected general benchmark performance; furthermore, domain training may alter how mod-
els utilize external retrieval information, while explicit RAG scenario training helps mitigate this
issue. These observations provide empirical references for understanding model adaptation in
knowledge-intensive vertical domains, and also indicate that stable model capability for utiliz-
ing external knowledge is an important prerequisite for further constructing EDA agents and
harness systems.
Building upon this work, we are continuing to expand ChipLingo’s research and applica-
tions, gradually extending from single-turn question-answering and retrieval utilization to more
complete harness capabilities, including more stable multi-step coordination combining external
documents, tool feedback, and task workflows. Meanwhile, we will further expand EDA task
coverage, continue scaling domain training data, and refine theEDA-Benchevaluation bench-
mark.EDA-Bench is currently undergoing review and optimization, with plans for
public release as an independent research contribution.
References
[1] He Z, Wu H, Zhang X, et al. ChatEDA: A Large Language Model Powered Autonomous
Agent for EDA. arXiv:2308.10204, 2023.
[2] Xu N, Zhang Z, Qi L, et al. ChipExpert: The Open-Source Integrated-Circuit-Design-
Specific Large Language Model. arXiv:2408.00804, 2024.
[3] Liu M, Ene T D, Kirby R, et al. ChipNeMo: Domain-Adapted LLMs for Chip Design.
arXiv:2311.00176, 2023.
[4] Lewis P, Perez E, Piktus A, et al. Retrieval-Augmented Generation for Knowledge-Intensive
NLP Tasks. arXiv:2005.11401, 2020.
[5] Gao Y, Xiong Y, Gao X, et al. Retrieval-Augmented Generation for Large Language Models:
A Survey. arXiv:2312.10997, 2023.
[6] Karakurt E, Akbulut A. Retrieval-Augmented Generation (RAG) and Large Language
Models (LLMs) for Enterprise Knowledge Management and Document Automation: A Sys-
tematic Literature Review.Applied Sciences, 16(1):368, 2026. doi:10.3390/app16010368.
[7] Colombo P, Pessoa Pires T, Boudiaf M, et al. SaulLM-7B: A pioneering Large Language
Model for Law. arXiv:2403.03883, 2024.
20

[8] Pu Y, He Z, Qiu T, et al. Customized Retrieval Augmented Generation and Benchmarking
for EDA Tool Documentation QA. arXiv:2407.15353, 2024.
[9] Thakur A, Chaturvedi S, Kothari P, et al. VerilogEval: Evaluating Large Language Models
for Verilog Code Generation. arXiv:2309.07544, 2023.
[10] Zhang T, Patil S G, Jain N, et al. RAFT: Adapting Language Model to Domain Specific
RAG. arXiv:2403.10131, 2024.
[11] Soudani H, Kanoulas E, Hasibi F. Fine Tuning vs. Retrieval Augmented Generation for
Less Popular Knowledge. arXiv:2403.01432, 2024.
[12] Xu N, Zhang Z, Shu S, et al. iScript: A Domain-Adapted Large Language Model and
Benchmark for Physical Design Tcl Script Generation. arXiv:2603.04476, 2026.
[13] Houlsby N, Giurgiu A, Jastrzebski S, et al. Parameter-Efficient Transfer Learning for NLP.
InProceedings of the 36th International Conference on Machine Learning (ICML),Pro-
ceedings of Machine Learning Research97:2790–2799, 2019.
[14] Li X L, Liang P. Prefix-Tuning: Optimizing Continuous Prompts for Generation.
arXiv:2101.00190, 2021.
[15] Hu E J, Shen Y, Wallis P, et al. LoRA: Low-Rank Adaptation of Large Language Models.
arXiv:2106.09685, 2021.
[16] Zhang Q, Chen M, Bukharin A, et al. AdaLoRA: Adaptive Budget Allocation for
Parameter-Efficient Fine-Tuning. arXiv:2303.10512, 2023.
[17] Dettmers T, Pagnoni A, Holtzman A, et al. QLoRA: Efficient Finetuning of Quantized
LLMs. arXiv:2305.14314, 2023.
[18] Pletenev S, Marina M, Moskovskiy D, et al. How Much Knowledge Can You Pack into
a LoRA Adapter without Harming LLM? InFindings of the Association for Computa-
tional Linguistics: NAACL 2025, pages 4309–4322, 2025. doi:10.18653/v1/2025.findings-
naacl.243.
[19] Asai A, Wu Z, Wang Y, et al. Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection. arXiv:2310.11511, 2023.
[20] Edge D, Trinh H, Cheng N, et al. From Local to Global: A Graph RAG Approach to
Query-Focused Summarization. arXiv:2404.16130, 2024.
[21] Sarmah B, Hall B, Rao R, et al. HybridRAG: Integrating Knowledge Graphs and Vector
Retrieval Augmented Generation for Efficient Information Extraction. arXiv:2408.04948,
2024.
21

[22] Zheng L, Chiang W L, Sheng Y, et al. Judging LLM-as-a-Judge with MT-Bench and Chat-
bot Arena. InAdvances in Neural Information Processing Systems36 (Datasets and Bench-
marks Track), 2023. arXiv:2306.05685.
[23] Liu Y, Iter D, Xu Y, Wang S, Xu R, Zhu C. G-Eval: NLG Evaluation using GPT-4 with Bet-
ter Human Alignment. InProceedings of the 2023 Conference on Empirical Methods in Nat-
ural Language Processing (EMNLP), pages 2511–2522, 2023. doi:10.18653/v1/2023.emnlp-
main.153.
[24] Zhou J, Lu T, Mishra S, et al. Instruction-Following Evaluation for Large Language Models.
arXiv:2311.07911, 2023.
[25] OpenAI. Measuring Short-Form Factuality in Large Language Models. 2024. Available at:
https://cdn.openai.com/papers/simpleqa.pdf.
[26] Chen M, Tworek J, Jun H, et al. Evaluating Large Language Models Trained on Code.
arXiv:2107.03374, 2021.
[27] Yang A, Li A, Yang B, et al. Qwen3 Technical Report. arXiv:2505.09388, 2025.
22