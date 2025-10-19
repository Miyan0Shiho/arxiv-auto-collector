# RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems

**Authors**: Jingru Lin, Chen Zhang, Stephen Y. Liu, Haizhou Li

**Published**: 2025-10-15 04:13:00

**PDF URL**: [http://arxiv.org/pdf/2510.13910v1](http://arxiv.org/pdf/2510.13910v1)

## Abstract
Retrieval-Augmented Generation (RAG) mitigates key limitations of Large
Language Models (LLMs)-such as factual errors, outdated knowledge, and
hallucinations-by dynamically retrieving external information. Recent work
extends this paradigm through agentic RAG systems, where LLMs act as agents to
iteratively plan, retrieve, and reason over complex queries. However, these
systems still struggle with challenging multi-hop questions, and their
intermediate reasoning capabilities remain underexplored. To address this, we
propose RAGCap-Bench, a capability-oriented benchmark for fine-grained
evaluation of intermediate tasks in agentic RAG workflows. We analyze outputs
from state-of-the-art systems to identify common tasks and the core
capabilities required for their execution, then construct a taxonomy of typical
LLM errors to design targeted evaluation questions. Experiments show that
"slow-thinking" models with stronger RAGCap performance achieve better
end-to-end results, underscoring the benchmark's validity and the importance of
enhancing these intermediate capabilities.

## Full Text


<!-- PDF content starts -->

RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval
Augmented Generation Systems
Jingru Lin1†Chen Zhang†Stephen Y. Liu Haizhou Li1,2
1National University of Singapore, Singapore
2The Chinese University of Hong Kong, Shenzhen, China
{jingrulin}@u.nus.edu
Abstract
Retrieval-Augmented Generation (RAG) miti-
gates key limitations of Large Language Mod-
els (LLMs)—such as factual errors, outdated
knowledge, and hallucinations—by dynami-
cally retrieving external information. Recent
work extends this paradigm through agentic
RAG systems, where LLMs act as agents to
iteratively plan, retrieve, and reason over com-
plex queries. However, these systems still strug-
gle with challenging multi-hop questions, and
their intermediate reasoning capabilities remain
underexplored. To address this, we propose
RAGCap-Bench, a capability-oriented bench-
mark for fine-grained evaluation of interme-
diate tasks in agentic RAG workflows. We
analyze outputs from state-of-the-art systems
to identify common tasks and the core capa-
bilities required for their execution, then con-
struct a taxonomy of typical LLM errors to
design targeted evaluation questions. Experi-
ments show that “slow-thinking” models with
stronger RAGCap performance achieve bet-
ter end-to-end results, underscoring the bench-
mark’s validity and the importance of enhanc-
ing these intermediate capabilities.
1 Introduction
Although large language models (LLMs) demon-
strate impressive performance across diverse do-
mains (Grattafiori et al., 2024; Hadi et al., 2023;
Xi et al., 2024), their reliance on internal knowl-
edge often leads to factual errors, outdated knowl-
edge, and hallucinations, reducing their reliability
on complex and dynamic queries (McKenna et al.,
2023; Ji et al., 2023; Huang et al., 2025). These
problems are alleviated with Retrieval Augmented
Generation (RAG) which equips the LLMs with an
external knowledge base (Lewis et al., 2020; Izac-
ard et al., 2023; Gao et al., 2023; Jiang et al., 2023).
However, such traditional RAG system still suffers
†Equal Contributionfrom problems such as limited access to real-time
information, shallow reasoning and context inte-
gration etc. (Singh et al., 2025). Recent progress
in LLM-based agentic RAG systems presents a
more promising approach to mitigate these issues
and improve task handling capabilities (Li et al.,
2025c; Wu et al., 2025; Jin et al., 2025b; Zheng
et al., 2025). An agentic RAG system enhances the
RAG pipeline by granting the LLM agency, where
LLMs interact with open web environments to dy-
namically retrieve and filter information, reason
logically with contexts, and plan adaptively and
systematically to answer a complex query.
Given the recent rapid advancement of agentic
RAG systems and their growing complexity (Asai
et al., 2024; Yao et al., 2023; Li et al., 2025b;
Feng et al., 2025; Zheng et al., 2025; Pan et al.,
2025; Rawat et al., 2025; Jin et al., 2025a; Song
et al., 2025; Li et al., 2025a), it is becoming cru-
cial to move beyond end-to-end QA-based eval-
uations (Krishna et al., 2025; Rein et al., 2023;
Zhou et al., 2025; Xi et al., 2025) and assess these
systems at a finer-grained level. Such granular eval-
uation requires a deeper understanding of the inter-
mediate processes that drive overall performance.
An agentic RAG system typically engages in
multiple rounds of planning, retrieval, and reason-
ing. Like solving a complex math problem, errors
in intermediate steps can propagate and degrade
the final outcome. To spotlight these critical in-
termediate tasks and the core model capabilities
they require, we introduce RAGCap-Bench, a fine-
grained benchmark for component-level evaluation
of common tasks within agentic RAG systems. We
identify the common tasks by analyzing several
recent open-source agentic RAG systems (Li et al.,
2025c; Jin et al., 2025b; Li et al., 2025a) and de-
sign four task types, includingPlanning,Evidence
Extraction,Grounded ReasoningandNoise Ro-
bustness. The evaluation questions for each task
type are derived from a taxonomy of typical er-
1arXiv:2510.13910v1  [cs.CL]  15 Oct 2025

Benchmark Language Open Web. Process
2WikiMultiHopQA (2020) En✗ ✗
MuSiQue (Trivedi et al., 2022) En✗ ✗
BrowseComp(Zhou et al., 2025) En✓✗
BrowseComp-ZH(Zhou et al., 2025) Zh✓✗
InfoDeepSeek(Xi et al., 2025) En&Zh✓✗
RAGCap-Bench (ours) En&Zh✓ ✓
Table 1: Comparisons between different benchmarks.
rors made by LLMs during intermediate task ex-
ecution and are presented in a Multiple-Choice
Question (MCQ) format. We evaluate the per-
formance of both fast-thinking and slow-thinking
LLMs on these MCQs and demonstrate that perfor-
mance on RAGCap-Bench reliably correlates with
their end-to-end performance in complex agentic
RAG workflows. Additionally, we show that LLM
performance on RAGCap-Bench is also indicative
of their ability to evaluate intermediate task out-
puts accurately. Compared to other benchmarks,
as shown in Table 1, RAGCap-Bench offers fine-
grained process-level evaluation, with realistic and
diverse information from open web environments.
Our key contributions are summarised as fol-
lows:
•RAGCap-Bench is the first comprehensive
evaluation benchmark focused on core inter-
mediate tasks common to most agentic RAG
workflows, moving beyond end-to-end assess-
ments that provide guidance on which specific
model capabilities need improvement.
•Our experiments show strong correlations be-
tween the performance of LLMs on RAGCap-
Bench and the performance on downstream
tasks.
•We also demonstrate that LLMs performing
well on RAGCap-Bench exhibit strong capa-
bilities in evaluating intermediate task outputs,
and the intermediate evaluation is indicative
of final end-to-end performance.
2 Problem Formulation
Figure 2 shows the general pipeline of the agen-
tic RAG system. Given a user query q∈ Q , the
LLM first designs its question-answering plan π0
= Plan (q)about how to tackle the question. In
the planning task, LLMs are required to break
down the question into sub-queries, extract the
question’s intents, or brainstorm ideas etc., de-
pending on the nature of the query. With the ini-
tial plan, the LLM will start to solve the problem,
29.8%
27.1%20.8%22.4%/glyph1197umber of Samples by Question Type
Planning
Evidence Extraction Grounded ReasoningNoise Robustness012345678Average Number of Options4.67.2
6.8
3.6Average /glyph1197umber of Options by Question Type
Planning
Evidence Extraction
Grounded Reasoning
Noise RobustnessFigure 1: Left: Proportion of samples in each MCQ
type. Right: Average number of options in each MCQ
type.
which requires iterative live searching, reasoning
and dynamic planning up to Tsteps. At each
stept, the agent retrieves a set of top-K evidence
Et={e 1, e2, ..., e k}from the open web using a
search engine. It then analyses the evidence set and
identifies useful evidences E′
tfrom the set. Based
on the evidences, the LLM starts to reason and
produces a reasoning chain Rt={r 1, r2, ..., r m}
where mis the number of reasoning steps. With
current evidence and reasoning, the LLM will then
refine its plan πt+1= Plan (q|E′
<t, R<t), in which it
determines whether current information is enough
to answer the question and makes a strategic plan
for the next step. If the agent thinks the current evi-
dence and reasoning are enough, it will stop search-
ing and proceed to generate the answer ˆy. Depend-
ing on the specific system design, some systems
incrementally gather relevant evidence through iter-
ative search and analysis, and then perform a final
reasoning step over the aggregated evidence set (as
indicated by the blue dotted box in Figure 2) before
generating the final answerˆy.
The complexity of the agentic RAG system
makes it hard to locate the actual problem. For
example, extracting a wrong evidence could lead
to incorrect reasoning, which propagates to later
processes. In RAGCap-Bench, we aim to propose
capability-oriented evaluations systematically as-
sess on the agentic systems at the process-level.
Formally, our goal is to evaluate how well the foun-
dation LLMs strategizes its question-answering
planπt, extract useful relevant evidence E′
tfrom
Etand reason Rtfaithfully with E′
t. Due to the
dynamic nature of the open web environments, the
evaluation of the noise robustness of LLMs toward
potentially low-quality, misleading and less credi-
ble sources of information is also important. In the
next section, we will introduce the construction of
this benchmark in detail.
2

3 RAGCap-Bench
Our benchmark comprises 255 curated questions.
To construct the dataset, we draw queries from mul-
tiple open-source deep research benchmarks (Zhou
et al., 2025; Xi et al., 2025; Krishna et al., 2025;
Chen et al., 2025) spanning diverse domains, in-
cluding entertainment, sports, arts, technology, and
medicine. Since the open-source datasets consist
solely of end-to-end query-answer pairs, we utilize
various agentic RAG systems to process the queries
and collect execution logs. From these logs, we ex-
tract the relevant intermediate information, which
is then used to generate task-specific MCQs. Fig-
ure 1 shows the number of samples and the average
number of options in RAGCap-Bench by task type.
More details about the open-source datasets and
agentic systems can be found in Appendix A.
3.1 MCQ Format
Motivated by prior works (Ross et al., 2025;
Hendrycks et al., 2020; Zellers et al., 2019), we
frame RAGCap-Bench questions as MCQs. Each
MCQ consists of a query, some intermediate out-
puts from the agentic RAG systems, and an instruc-
tion to ask the LLMs to choose the correct/incorrect
options from all options given. The general formats
of the MCQs are shown in the bottom of Figure 2.
Examples of MCQs are shown in Appendix E.
3.2 Dataset Construction
There are four MCQ task types:Planning,Evi-
dence Extraction,Grounded ReasoningandNoise
Robustness. The two main MCQs generation strate-
gies include 1) Vanilla Generation; 2) Error-Guided
Generation. The former involves directly extract-
ing and reformatting relevant information from ex-
ecution logs, framing it as MCQs to evaluate the
corresponding model capabilities. The latter in-
volves dedicated error-guided prompts to instruct
the LLMs to generate high-quality and challenging
MCQs. We mainly use GPT-4o (Hurst et al., 2024),
Qwen-Plus (Yang et al., 2025) and DeepSeeek-
V3 (Liu et al., 2024) for our dataset construction.
Below shows how the MCQs of each type are con-
structed:
Planning: As shown in Figure 2, planning capa-
bility is essential at two key stages within agentic
RAG systems. The first occurs when the LLM
receives the initial query and begins interpreting
it. Based on its understanding of the user’s in-
tent, the model formulates a strategic plan to guidethe question-answering process. The second stage
arises after the LLM has retrieved search results
and conducted some reasoning. At this point, the
model must assess what aspects of the query have
already been addressed, identify remaining gaps,
and dynamically adjust its plan based on the newly
acquired information. Depending on the nature of
the question and the current information the LLM
has gathered, the LLMs might need convergent
and divergent planning abilities at any stage. The
converging planning ability refers to the ability to
narrow down the search space, advancing the rea-
soning process efficiently. In contrast, divergent
planning entails broadening the search space to
enhance the diversity and coverage of retrieved
information, ensuring that important aspects are
not overlooked. In Appendix B, we show the typi-
cal use cases of convergent and divergent planning
abilities for different types of queries.
To curate MCQs for assessing both convergent
and divergent planning abilities, we gather relevant
thinking trajectories (CoT from the slow-thinking
agent) for the two stages of question-answering
planning, as mentioned above, from various agen-
tic RAG systems. To ensure the data quality, we
only keep the trajectories that lead to the cor-
rect final answer. Among all the systems ana-
lyzed, WebThinker (Li et al., 2025c) and HiRA (Jin
et al., 2025b) are particularly specialized in high-
level question-answering planning. Leveraging
this, we prompt LLMs to generate MCQs based on
their planning traces. Additionally, since humans
excel at both convergent and divergent thinking
when solving complex problems, we also collect
samples from open-source datasets where human-
annotated stepwise problem-solving strategies are
available (Chen et al., 2025; Du et al., 2025), and in-
struct the LLMs to generate MCQs based on these
strategies.
Appendix D provides the detailed prompts used
by LLMs to generate MCQs designed to assess the
convergent and divergent planning abilities. The
above-mentioned high-quality thinking trajectories
from both agentic RAG systems and humans make
up the correct options in the MCQs. For wrong
options, we instruct the LLMs to generate them.
Specifically, we employ Error-Guided Generation,
which includes the identified common mistakes
made by agentic RAG systems during the planning
process. The common mistakes made are listed in
Table 2.
Evidence Extraction: Vanilla Generation is
3

Deep Resear ch Systems
e.g. HiRA, WebThinker ,
WebDancer etc.
Search Tools
e.g. Baidu, Google, Jina etc.
⼀款 2D⽣存冒险游
戏、带有恐怖 ...Can a user of FMS software legally make a backup...?
Document 0: Title: Software License Agreement Overview;
Document1: Title: Ownership and Copyright of FMS
Software;...; Document N: ...
Reference Answer: A user of FMS software can legally
make a backup copy of the Software for ......
Multiple-Choice Questions
Initial Plan Search Reason Refine Plan
 Evidence Set Reason
 Extract Evidence
Vanilla Generation Error -Guided Generation
Answer
Grounded
ReasoningNoise
RobustnessPlanningEvidence
Extraction
Filtering
Step 1: Error Analysis &
Summarisation ；
Step 2: Error -Guided LLM
Generation
Annotate
[User Query]
[Available Information]
Which of the following
actions would be most
appropriate?
A. [Action 1]
B. [Action 2]
C. [Action 3]
...[User Query]
[Search Query]
Which of the following
webpages are not useful
for answering the query?
A. [W eb page 1]
B. [W eb page 2]
C. [W eb page 3]
...[User Query]
[Evidence Set]
Which of the following
statements are not supported
by the documents above?
A. [Statement 1]
B. [Statement 2]
C. [Statement 3]
...[User Query]
[Available Information]
Which of the following
plans would be most 
appropriate?
A. [Plan 1]
B. [Plan 2]
C. [Plan 3]
...[User Query]
[Evidence Set]
Based on the evidence
above, is the question
answerable? 
A. yes
B. no[Topic]
Please select low-quality ,
potentially misleading or
less credible web pages.
A. [W eb page 1]
B. [W eb page 2]
C. [W eb page 3]
D. [W eb page 4]
...Figure 2: General agentic RAG pipeline and data construction processes for RAGCap-Bench. We take queries
from open-source question-answering (QA) datasets. We then run different deep research systems and collect
the intermediate outputs. Two main strategies, Vanilla Generation and Error-Guided Generation are deployed to
generate the MCQs based on queries and the intermediate outputs. The generated MCQs are filtered to ensure the
quality and experts are recruited to provide the answers.
adopted for this task. Specifically, we collect the
search queries and retrieved web pages in the execu-
tion trajectories of different agentic RAG systems
and each web page is formatted into an MCQ op-
tion, as shown in Figure 2. The MCQs typically in-
clude more than four options, and the model being
evaluated is tasked with selecting the least relevant
evidence to answer the user query. Appendix D
presents the prompts for the MCQs generation.
Grounded Reasoning: We employ both Error-
Guided Generation and Vanilla Generation strate-
gies here. For Error-Guided Generation, we use in-
termediate outputs that lead to correct final answer.
The query, evidence set, the associated reasoning
steps from the intermediate outputs and the com-
mon mistakes for grounded reasoning as listed in
Table 2 are all included in the generation prompts.
The LLMs are then instructed to generate MCQs
to introduce errors into some statements in the rea-
soning steps. To facilitate a more realistic critique
of the actual agentic RAG systems, we also make
use of reasoning steps that lead to wrong answers.
For these reasoning steps, they already contain er-
rors. Therefore, we only employ Vanilla Genera-
tion, which prompts the LLMs to extract statements
from the reasoning steps and maximise the diver-
sity of the extracted statements. The prompts for
generation using both strategies are shown in Ap-
pendix D. Finally, an instruction is added to MCQsasking for the selection of the incorrect statements,
enabling the evaluation of the model’s ability to
identify erroneous reasoning.
Noise Robustness: Real-time search results may
include noise, such as low-quality, less credible, or
entirely irrelevant information. During inspection
of the execution logs of different agentic RAG sys-
tems, we find that they are prone to forcing their
reasoning based on the retrieved evidence even
though they may contain lots of noise. In this task,
we mainly assess whether the models are robust
against noisy search results. In the MCQs, we
provide the model to evaluate search results that
contain no or very limited useful information, and
prompt the model with two options: answer the
question or admit that the question cannot be an-
swered with the given search results. In this type
of question, we deem an LLM that admits the ques-
tions are not answerable as a noise-robust one. We
term the capability as noise-abstain.
We also assess whether the LLMs can identify
less credible sources. While it is still possible to
extract information from less credible sources, it
is hard to guarantee the quality of the informa-
tion. This capability is especially important when
answering queries from high-stakes areas such as
medicine, laws, academic and policies etc. Fur-
thermore, this capability prioritises more credi-
ble sources of information, which might be useful
4

Question Type Targeted Capability Common Mistakes
PlanningThe capability to interpret and break down the
problem, implement an optimal/strategic
problem-answering plan, and dynamically refine
its plan based on current information.Poor question interpretation
Omission of necessary constraints
Poor planning logic
Poor dynamic planning
Limited scope and depth
Evidence
ExtractionThe capability to identify useful evidence from a
large amount of retrieved documents.Shallow keyword matching
Fail to recognise implicit connections
Grounded
ReasoningThe capability to reason with grounding and
generate a well-supported statement.Hallucinated support
Contradiction with retrieved information
Failed to identify implicit reasoning gap
Noise
RobustnessThe capability to detect low-quality, less-reliable
information and the capability to abstain.Forced/Incorrect reasoning based on noisy informa-
tion
Unable to detect low-quality, misleading, and less
credible sources
Table 2: Question types, the targeted capabilities, and common mistakes made by LLMs.
when multiple sources present conflicting informa-
tion. Vanilla MCQs Generation is employed to
format the MCQs, which ask the LLMs to identify
less credible sources from all the MCQ options.
We term this capability as noise-reliability.
3.3 Dataset Post Processing
Despite employing Error-Guided Generation,
LLMs may still generate trivial cases and suffer
from issues such as hallucination and poor instruc-
tion following. To ensure the quality of the bench-
mark questions, we apply both difficulty filtering
and format filtering. Difficulty filtering removes
trivial cases by prompting multiple models to an-
swer the MCQs, and discarding those where all
LLMs provide the same answers. Format filtering
removes MCQs with poor formatting or ambiguity
using a combination of rule-based heuristics and
manual verification.
For more precise evaluations, all MCQs are an-
notated by human experts equipped with advanced
deep research tools. The ground-truth answers
to the MCQs is determined by the majority vote
among the annotators.
3.4 Evaluation Metrics
We adopt both Exact Match (EM) and instance-
wise macro F1 score as the evaluation metrics.
Planningquestions contain two sub-categories:
converging and diverging.Noise Robustnessalso
contains two sub-categories: abstain and reliability.
The EM and F1 scores are first averaged within
the group. The overall score is the average of four
scores from each question type.4 Evaluations
4.1 Setups
We evaluate a range of fast-thinking LLMs, includ-
ing Qwen2.5-72B-Instruct (Team, 2024), Qwen-
Plus w/o think (Yang et al., 2025), DeepSeek-
V3 (Liu et al., 2024), GPT-4o (Hurst et al., 2024)
and Gemini-1.5-Pro (Team et al., 2024), and slow-
thinking LLMs, including Qwen3-8B (Yang et al.,
2025), Qwen3-32B w/ think (Yang et al., 2025),
QwQ-32B (Yang et al., 2025), Qwen3-235B-A22B
w/ think (Yang et al., 2025), DeepSeek-R1 (Guo
et al., 2025), O1-mini (Jaech et al., 2024) and
Gemini-2.5-Flash (Comanici et al., 2025).
4.2 Benchmark Results
We evaluate the performance of fast-thinking and
slow-thinking models across the different task
types using both bare and informative prompts. All
prompts are shown in Appendix H. Bare prompts
refer to those with no error examples included. The
LLMs are given the query, MCQs, and simple in-
structions about the output format. Informative
prompts additionally include the few-shot exam-
ples about the common mistakes listed in Table 2.
The latter shows improved and more robust perfor-
mance. Table 3 shows the results for using informa-
tive prompts. The comparison with bare prompts
is shown in Section 4.3.
Planning: We present the scores for convergent
and divergent planning ability separately in Table 3.
The EM cand F1 cdenote the scores for conver-
gent planning ability. As for divergent planning
ability, we only compute EM d. For both planning
5

ModelPlanning Evidence Extraction Grounded Reasoning Noise Robustness Overall
EMcF1cEMd EM F1 EM F1 EM aEMrF1r EM F1
fast-thinking models
Qwen2.5-72B-Instruct 27.45 54.88 76.00 31.88 79.47 35.85 77.98 64.86 10.00 77.18 39.22 72.37
Qwen-Plus w/o think 49.02 74.10 84.00 27.54 74.44 35.85 80.82 64.86 25.00 74.45 43.71 75.95
Deepseek-v3 49.02 74.97 80.00 15.94 75.07 41.51 81.19 78.38 25.00 81.01 43.41 78.06
GPT-4o 35.29 59.48 72.00 34.78 78.27 50.94 81.89 97.30 10.00 65.96 48.26 71.40
Gemini-1.5-Pro 49.02 75.82 76.00 18.84 75.37 37.74 80.49 54.05 10.00 73.36 37.78 76.26
slow-thinking models
Qwen3-8B 45.10 67.52 72.00 18.84 74.10 49.06 83.54 56.76 50.00 83.25 44.96 76.26
Qwen3-32B w/ think 43.14 67.45 68.00 30.43 62.97 49.06 83.95 59.46 40.00 82.15 46.20 74.13
QwQ-32B 33.33 65.56 80.00 34.78 79.05 54.72 85.19 62.16 35.00 84.32 48.68 78.53
Qwen3-235B-A22B w/ think 47.06 70.92 80.00 39.13 73.39 56.60 88.70 54.05 40.00 82.68 51.57 79.94
DeepSeek-R1 52.94 74.38 84.00 36.23 81.34 52.83 85.89 70.27 35.00 80.92 52.54 80.63
O1-mini 33.33 70.10 64.00 30.40 77.05 43.40 78.87 62.16 20.00 76.14 40.89 75.52
Gemini-2.5-Flash 52.94 72.75 80.00 31.88 78.33 47.17 83.48 75.68 25.00 75.40 48.97 77.48
Table 3: Performance of different fast- and slow-thinking LLMs using informative prompts. EM cand F1 cdenote
the scores for converging ability. EM ddenotes the EM score for diverging ability. EM adenotes the EM score for
noise-abstain. EM rand F1 rdenote the scores for noise-reliability. The scores are presented as percentages for
clarity.
abilities, a high EM score demonstrates that the
LLM is capable of identifying the optimal solution
path to the final answer among all the non-optimal
paths. In the agentic RAG systems, implement-
ing a good plan not only leads to a final answer
but also means higher efficiency. For example,
with concise problem-solving steps, the systems
can avoid redundant search that leads to more use-
less evidence or repetitively checking on known
information. On the other hand, a high F1 score
indicates that both optimal and non-optimal paths
might be chosen. In agentic RAG systems, where
iterative reasoning is performed, it is still possible
for the system to reach the final solution even if
the system takes redundant steps. From Table 3,
we find that Gemini and DeepSeek series generally
exhibit consistent ability to identify optimal paths
among all fast-thinking and slow-thinking models.
DeepSeek-V3 and Gemini-1.5-Pro both obtain the
highest EM cscore of 49.02% among fast-thinking
models. DeepSeek-R1 and Gemini-2.5-Flash ob-
tain the highest EM cscore of 52.94% among slow-
thinking models. In terms of F1 c, Gemini-1.5-Pro
leads with a score of 75.82%, closely followed
by DeepSeek-v3 with a score of 74.97% among
fast-thinking models. DeepSeek-R1 obtains an F1 c
score of 74.38%, followed by Gemini-2.5-Flash,
with an F1 cscore of 72.75% among slow-thinking
models. Among the Qwen series, the planning
ability seems inconsistent. While Qwen-Plus w/o
think achieves the highest EM cand EM d, Qwen3-
235B-A22B w/ think, despite the large size, doesnot perform as good.
Evidence Extraction: A high EM score in evi-
dence extraction shows the LLMs’ ability to iden-
tify the relevant evidence and filter useless evidence
precisely. In the agentic RAG system, a clean set of
evidence provides a strong foundation for faithful
and accurate reasoning. From Table 3, we see that
all fast-thinking and slow-thinking models show
low EM scores of less than 40% on this question
type, which shows that the LLMs struggle to pro-
cess information from dynamic open-web environ-
ments. The best EM scores achieved are 34.78%
by GPT-4o among the fast-thinking models, and
39.13% by Qwen3-235B-A22B w/ think among
the slow-thinking models. This weakness can be
attributed to the mismatch between pretraining dis-
tributions and noisy, unstructured web data, and the
limitations in aggregating evidence across sources.
However, in the context of agentic RAG systems,
it is common for multiple sources to provide over-
lapping information. In such cases, the omission of
some repetitive evidence has a trivial impact on the
reasoning. Through analysis on the performance of
LLMs on this question type, we summarise three
common failure cases: a) omission of repetitive evi-
dence; b) omission of key information; c) inclusion
of irrelevant information. While a) is a trivial mis-
take, the effects of b) and c) are dependent on the
system’s planning and noise robustness capabilities.
We will elaborate on these interdependencies later
in this section. In this sense, F1, which provides
partial correctness, captures whether the systems
6

are able to extract at least some relevant informa-
tion to carry on with their reasoning process. From
Table 3, we can see the LLMs are generally able to
achieve F1 scores of more than 70%. Among all
models, DeepSeek-R1 achieves the best F1 score
of 81.34% among all models.
Grounded Reasoning: While an F1 score re-
flects partial correctness in reasoning statements,
it does not reflect deeper reasoning failures. A
mistake made by LLMs in any intermediate reason-
ing stage can be propagated through subsequent
stages of the agentic RAG systems. Conversely,
even perfect extraction can be undermined if the
model’s reasoning process is flawed. Therefore,
the EM score serves as a stronger indicator for
flawless reasoning grounded in given evidence. De-
spite high F1 scores (generally more than 80%)
achieved by many LLMs, the EM scores are much
lower. Among fast-thinking models, the best EM
score of 50.94% is obtained by GPT-4o. Among
slow-thinking models, the best EM score is 56.60%,
achieved by Qwen3-235B-A22B.
Noise Robustness: For noise-abstain, only EM
score, denoted as EM ais computed. Our results
show that most models are noise robust. When
provided with irrelevant information, most LLMs
admit that the question is not answerable. On the
other hand, most LLMs do not recognise the credi-
bility and reliability of the sources. Most models in
Table 3 have an EM rof less than 50% despite high
F1rscores. Models such as Qwen2.5-72B-Instruct,
GPT-4o and Gemini-1.5-Pro obtain an EM ras low
as 10.00%. This suggests a concerning trend: many
models exhibit a tendency to trust retrieved infor-
mation indiscriminately, regardless of the sources’
credibility. A closer examination of the models’
reasoning outputs reveals that many models priori-
tise the content of retrieved web pages over their
reliability. For example, the LLMs tend to accept
the sources as long as they appear informative, even
if they think the sources might come from a poten-
tially less credible authority. While this tendency
may be relatively safe when information across
sources is consistent, it becomes problematic in the
presence of conflicting evidence. In such cases, a
typical approach that humans would involve check-
ing the origin of the sources before drawing the
conclusions. Among all models, Qwen3-8B per-
forms surprisingly well in detecting the less cred-
ible sources. However, one should note that the
Qwen3-8B is also a weak model in evidence extrac-
tion, which partially explains its good performance
Qwen-2.5-72B-Instruct
Qwen-Plus w/o think
Deepseek-V3 GPT-4oGemini-1.5-Pro
2530354045505560(a) /glyph1197on-reasoning Models
bare
informativeQwen3-8B
Qwen3-32B
QwQ-32B
Qwen3-235B DeepSeek-R1O1-miniGemini-2.5-Flash
2530354045505560(b) Reasoning Models
bare
informativeFigure 3:RAGCap-Bench overall EM scores for different
fast-thinking (left) and slow-thinking models (right), with
informative (orange) and bare (blue) prompts.
on EM rdue to its disengagement with the content.
Although we define four capabilities of LLMs in
an agentic RAG system, it is important to recognize
that these capabilities do not operate in isolation.
Instead, they interact synergistically, influencing
one another and working together as an integrated
process to answer the query effectively. For exam-
ple, for evidence extraction and noise robustness,
the negative impacts of missing out key information
and selecting irrelevant information are possibly
alleviated if the LLMs are noise robust enough.
When conflicts occur between different sources of
information, a noise-robust LLM that recognises
more reliable sources of information can guide the
selection of evidence. For planning and noise ro-
bustness, a concise strategy can lead to fewer irrel-
evant search results, which relieves the burden of
handling the noisy search results.
4.3 With vs. Without Error-Guided Prompts
The detailed results for bare prompts (without error-
guided exemplars) are presented in Appendix F.
Here, we only show a comparison between the
overall EM scores for the two types of prompts.
From Figure 3, it is clear that the use of informa-
tive prompts, guided by our error identification in
Table 2, leads to improved performance in both
fast- and slow-thinking LLMs. This observation
underscores a central challenge in building agentic
RAG systems: even slow-thinking LLMs struggle
to interact with dynamic and noisy web informa-
tion without structured guidance. This explains
why building robust agentic RAG systems usually
requires intensive prompt engineering or other post-
training techniques (Asai et al., 2024; Zhang et al.,
2025).
4.4 Correlation with Downstream
A good benchmark for evaluating the intermedi-
ate tasks should faithfully reflect their effective-
7

Qwen3-8B Qwen3-32B Qwen3-235B1020304050Accuracy/EM
WebThinker
Qwen3-8B Qwen3-32B Qwen3-235B
HiRA
RAGCap-Bench InfoDeepSeek BrowseComp-ZhFigure 4:Correlation of performance on RAGCap-Bench
with performance on InfoDeepSeek and BrowseComp-Zh, for
Qwen3-8B, Qwen3-32B and Qwen3-235B.
ness with the end-to-end performance. There-
fore, we compare the results of different models
on RAGCap-Bench to the downstream question-
answering (QA) tasks. We use three LLMs of vary-
ing sizes, including Qwen3-8B, Qwen3-32B, and
Qwen3-235B-A22B, and two representative agen-
tic RAG pipelines, including WebThinker (ReACT-
based) and HiRA (Multi-Agent-based), yielding
a total of six configurations. For downstream
datasets, we use the QA pairs from BrowseComp-
Zh and InfoDeepSeek. We limit each inference to
10 Google Search API calls. Figure 4 shows the cor-
relation between the performance of these LLMs
on RAGCap-Bench and the downstream perfor-
mance when the LLMs act as the agents in respec-
tive pipelines. A positive correlation is observed
from the figure, showcasing that RAGCap-Bench
can reliably serve as an efficient alternative to the
more costly and time-consuming end-to-end QA
evaluation.
Furthermore, the performance of Qwen3-8B and
Qwen3-32B, while the former is much smaller in
size, are close in RAGCap-Bench, with the over-
all EM scores of 44.96% and 46.20% respectively.
This is also reflected in their downstream perfor-
mance. From Figure 4, we can see that the perfor-
mance of Qwen3-8B and Qwen3-32B is close on
both datasets for the HiRA workflow, as well as on
InfoDeepSeek for the WebThinker workflow.
4.5 LLMs as Evaluator
In this section, we examine the relationship be-
tween RAGCAP-Bench and direct evaluation of in-
termediate outputs within the reasoning trajectory.
We randomly sample 500 inference trajectories
from WebThinker experiments described in Sec-
tion 4.4 and apply the LLM-as-a-judge approach,
prompting different LLMs to score grounded rea-
soning (thought-action) and evidence extraction
(observation-extraction) on a 1–10 scale for each
thought-action-observation-extraction step. TheEvidence Extraction Direct Scoring RAGCAP-Bench
Correlation EM F1
Qwen3-8B 0.210 18.84 74.10
Qwen3-32B 0.113 30.43 62.97
Qwen3-235B 0.528 39.13 73.39
Grounded Reasoning Direct Scoring RAGCAP-Bench
Correlation EM F1
Qwen3-8B 0.291 49.06 83.54
Qwen3-32B 0.316 49.06 83.95
Qwen3-235B 0.338 56.60 88.70
Table 4: The second column reports the point-biserial
correlation coefficient between the evaluator scores and
the downstream performance. All correlation scores are
statistically significant (<0.05). The third and fourth
columns show corresponding EM and F1 scores on
RAGCap-Bench.
overall scores are averaged across all the steps in
the reasoning trajectory. The evaluation prompts
are provided in Appendix G. Due to a lack of
ground-truth human scores, we approximate the di-
rect evaluation performance using the point-biserial
correlation coefficient (Tate, 1954) between the
LLM overall scores and the downstream outcomes
of the 500 samples, measured as binary output (i.e.,
whether the final answer is correct or incorrect). We
choose three LLMs of varying sizes: Qwen3-8B,
Qwen3-32B, and Qwen3-235B. Table 4 presents
the comparisons. Notably, the correlation values of
the LLMs show a consistent trend with their respec-
tive performance for the evidence extraction and
grounded-reasoning categories on RAGCap-Bench.
Qwen3-235B’s evaluator scores have the highest
correlation with the results, and it also obtains the
best EM score in RAGCap-Bench. Qwen3-8B
and Qwen3-32B both have lower EM scores on
evidence extraction and grounded reasoning, and
hence, they also demonstrate weaker correlation
with the downstream task results. Qwen3-8B has
a slightly stronger correlation than Qwen3-32B on
Evidence Extraction, which might be explained by
the higher F1 scores in evidence extraction. As
discussed in Section 4.2, both EM and F1 scores
are significant indicators of the evidence extraction
capability.
5 Related Work
Agentic RAG Systems: While RAG equips the
LLMs with access to external knowledge bases,
mitigating issues such as factual errors and hal-
8

lucinations, it still struggles to solve complex
tasks (Singh et al., 2025). Agentic RAG has en-
hanced the capabilities of the traditional RAG sys-
tems. It no longer treats the LLM merely as a
passive text generator, but as an active agent ca-
pable of adaptive planning, dynamic information
seeking, and iterative reasoning (Asai et al., 2024;
Yao et al., 2023; Li et al., 2025b; Feng et al.,
2025; Zheng et al., 2025; Pan et al., 2025). No-
tably, this emerging paradigm has seen increas-
ing adoption in practical applications, including
OpenAI Deep Research (OpenAI, 2025), Gemini
Deep Research (Gemini, 2025), Perplexity Deep
Research (Perplexity Team, 2025) etc., all of which
leverage LLMs as autonomous agents.
RAG Benchmark: With the rise of agentic RAG
systems, comprehensive and systematic bench-
marking becomes increasingly important for un-
covering potential weaknesses and steering the
development of more effective and capable sys-
tems. Some benchmarks are proposed for this
purpose (Xi et al., 2025; Zhou et al., 2025; Wei
et al., 2025). However, most of these efforts focus
on question-answering (QA) tasks that evaluate a
system’s ability to answer challenging, multi-hop
questions. Although they are useful indicators of
end-task performance, they offer limited insight
into the intermediate planning, retrieving and rea-
soning tasks executed by the systems.
6 Conclusion
This work introduces RAGCap-Bench, a capability-
oriented benchmark designed for fine-grained,
component-wise evaluation for the agentic RAG
systems. RAGCap-Bench addresses the critical
gap in existing benchmarks, which lack evaluation
on the intermediate processes of the systems. Ex-
perimental results demonstrate that the RAGCap-
Bench scores are correlated with the downstream
task performance, highlighting its practical rele-
vance. In addition, we conduct exploratory exper-
iments to show the potential of using LLMs to
assess intermediate outputs of the agentic RAG sys-
tems. This paves the way for future research into
the integration of LLMs as a means of improving
the agentic RAG systems.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-RAG: Learning to
retrieve, generate, and critique through self-reflection.InThe Twelfth International Conference on Learning
Representations.
Kaiyuan Chen, Yixin Ren, Yang Liu, Xiaobo Hu, Hao-
tong Tian, Tianbao Xie, Fangfu Liu, Haoye Zhang,
Hongzhang Liu, Yuan Gong, Chen Sun, Han Hou,
Hui Yang, James Pan, Jianan Lou, Jiayi Mao, Jizheng
Liu, Jinpeng Li, Kangyi Liu, and 14 others. 2025.
xbench: Tracking agents productivity scaling with
profession-aligned real-world evaluations.Preprint,
arXiv:2506.13651.
Gheorghe Comanici, Eric Bieber, Mike Schaekermann,
Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Mar-
cel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and
1 others. 2025. Gemini 2.5: Pushing the frontier with
advanced reasoning, multimodality, long context, and
next generation agentic capabilities.arXiv preprint
arXiv:2507.06261.
Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang,
and Zhendong Mao. 2025. Deepresearch bench: A
comprehensive benchmark for deep research agents.
arXiv preprint arXiv:2506.11763.
Wenfeng Feng, Chuzhan Hao, Yuewei Zhang, Jingyi
Song, and Hao Wang. 2025. Airrag: Activat-
ing intrinsic reasoning for retrieval augmented gen-
eration using tree-based search.arXiv preprint
arXiv:2501.10053.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2(1).
Gemini. 2025. Gemini deep research.
https://gemini.google/overview/
deep-research/. Accessed: 2025-08-02.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Muhammad Usman Hadi, Rizwan Qureshi, Abbas Shah,
Muhammad Irfan, Anas Zafar, Muhammad Bilal
Shaikh, Naveed Akhtar, Jia Wu, Seyedali Mirjalili,
and 1 others. 2023. A survey on large language mod-
els: Applications, challenges, limitations, and practi-
cal usage.Authorea Preprints.
Dan Hendrycks, Collin Burns, Steven Basart, Andy
Zou, Mantas Mazeika, Dawn Xiaodong Song, and
Jacob Steinhardt. 2020. Measuring massive multitask
language understanding.ArXiv, abs/2009.03300.
9

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong,
Zhangyin Feng, Haotian Wang, Qianglong Chen,
Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 oth-
ers. 2025. A survey on hallucination in large lan-
guage models: Principles, taxonomy, challenges, and
open questions.ACM Transactions on Information
Systems, 43(2):1–55.
Aaron Hurst, Adam Lerer, Adam P Goucher, Adam
Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, and 1
others. 2024. Gpt-4o system card.arXiv preprint
arXiv:2410.21276.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richard-
son, Ahmed El-Kishky, Aiden Low, Alec Helyar,
Aleksander Madry, Alex Beutel, Alex Carney, and 1
others. 2024. Openai o1 system card.arXiv preprint
arXiv:2412.16720.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea
Madotto, and Pascale Fung. 2023. Survey of hal-
lucination in natural language generation.ACM com-
puting surveys, 55(12):1–38.
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun,
Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. 2023. Active retrieval
augmented generation. InProceedings of the 2023
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 7969–7992, Singapore. As-
sociation for Computational Linguistics.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025a. Search-R1: Training llms to reason and
leverage search engines with reinforcement learning.
Preprint, arXiv:2503.09516.
Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang,
Yutao Zhu, Yang Zhao, Hongjin Qian, and Zhicheng
Dou. 2025b. Decoupled planning and execution: A
hierarchical reasoning framework for deep search.
arXiv preprint arXiv:2507.02652.
Satyapriya Krishna, Kalpesh Krishna, Anhad Mo-
hananey, Steven Schwarcz, Adam Stambler, Shyam
Upadhyay, and Manaal Faruqui. 2025. Fact, fetch,
and reason: A unified evaluation of retrieval-
augmented generation. InProceedings of the 2025Conference of the Nations of the Americas Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Pa-
pers), pages 4745–4759, Albuquerque, New Mexico.
Association for Computational Linguistics.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen
Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan
Li, Zhengwei Tao, Xinyu Wang, and 1 others. 2025a.
Websailor: Navigating super-human reasoning for
web agent.arXiv preprint arXiv:2507.02592.
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang,
Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng
Dou. 2025b. Search-o1: Agentic search-enhanced
large reasoning models.Preprint, arXiv:2501.05366.
Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yu-
tao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng
Dou. 2025c. Webthinker: Empowering large rea-
soning models with deep research capability.arXiv
preprint arXiv:2504.21776.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang,
Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi
Deng, Chenyu Zhang, Chong Ruan, and 1 others.
2024. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437.
Nick McKenna, Tianyi Li, Liang Cheng, Mohammad
Hosseini, Mark Johnson, and Mark Steedman. 2023.
Sources of hallucination by large language models
on inference tasks. InFindings of the Association
for Computational Linguistics: EMNLP 2023, pages
2758–2774, Singapore. Association for Computa-
tional Linguistics.
OpenAI. 2025. Introducing deep re-
search. https://openai.com/index/
introducing-deep-research/ . Accessed:
2025-08-02.
Melissa Z Pan, Mert Cemri, Lakshya A Agrawal, Shuyi
Yang, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer,
Aditya Parameswaran, Kannan Ramchandran, Dan
Klein, and 1 others. 2025. Why do multiagent sys-
tems fail? InICLR 2025 Workshop on Building Trust
in Language Models and Applications.
Perplexity Team. 2025. Introducing per-
plexity deep research. https://
www.perplexity.ai/hub/blog/
introducing-perplexity-deep-research .
Accessed: 2025-08-02.
Mrinal Rawat, Ambuje Gupta, Rushil Goomer, Alessan-
dro Di Bari, Neha Gupta, and Roberto Pieraccini.
10

2025. Pre-act: Multi-step planning and reason-
ing improves acting in llm agents.arXiv preprint
arXiv:2505.09970.
David Rein, Betty Li Hou, Asa Cooper Stickland,
Jackson Petty, Richard Yuanzhe Pang, Julien Di-
rani, Julian Michael, and Samuel R. Bowman. 2023.
Gpqa: A graduate-level google-proof q&a bench-
mark.Preprint, arXiv:2311.12022.
Hayley Ross, Ameya Sunil Mahabaleshwarkar, and
Yoshi Suhara. 2025. When2call: When (not) to call
tools.arXiv preprint arXiv:2504.18851.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented gen-
eration: A survey on agentic rag.arXiv preprint
arXiv:2501.09136.
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen,
Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-
Rong Wen. 2025. R1-searcher: Incentivizing the
search capability in llms via reinforcement learning.
Preprint, arXiv:2503.05592.
Robert F Tate. 1954. Correlation between a discrete and
a continuous variable. point-biserial correlation.The
Annals of mathematical statistics, 25(3):603–607.
Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan
Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer,
Damien Vincent, Zhufeng Pan, Shibo Wang, and 1
others. 2024. Gemini 1.5: Unlocking multimodal
understanding across millions of tokens of context.
arXiv preprint arXiv:2403.05530.
Qwen Team. 2024. Qwen2.5: A party of foundation
models.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics.
Jason Wei, Zhiqing Sun, Spencer Papay, Scott McK-
inney, Jeffrey Han, Isa Fulford, Hyung Won Chung,
Alex Tachard Passos, William Fedus, and Amelia
Glaese. 2025. Browsecomp: A simple yet challeng-
ing benchmark for browsing agents.arXiv preprint
arXiv:2504.12516.
Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin,
Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun
Xi, Gang Fu, Yong Jiang, and 1 others. 2025. Web-
dancer: Towards autonomous information seeking
agency.arXiv preprint arXiv:2505.22648.
Yunjia Xi, Jianghao Lin, Menghui Zhu, Yongzhao
Xiao, Zhuoying Ou, Jiaqi Liu, Tong Wan, Bo Chen,
Weiwen Liu, Yasheng Wang, and 1 others. 2025.
Infodeepseek: Benchmarking agentic information
seeking for retrieval-augmented generation.arXiv
preprint arXiv:2505.15872.Yunjia Xi, Weiwen Liu, Jianghao Lin, Xiaoling Cai,
Hong Zhu, Jieming Zhu, Bo Chen, Ruiming Tang,
Weinan Zhang, and Yong Yu. 2024. Towards open-
world recommendation with knowledge augmenta-
tion from large language models. InProceedings of
the 18th ACM Conference on Recommender Systems,
pages 12–22.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.
2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak
Shafran, Karthik Narasimhan, and Yuan Cao. 2023.
React: Synergizing reasoning and acting in language
models. InInternational Conference on Learning
Representations (ICLR).
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali
Farhadi, and Yejin Choi. 2019. Hellaswag: Can a
machine really finish your sentence? InProceedings
of the 57th Annual Meeting of the Association for
Computational Linguistics, pages 4791–4800.
Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao
Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang,
Derong Xu, Zhaocheng Du, Huifeng Guo, and 1 oth-
ers. 2025. Process vs. outcome reward: Which is
better for agentic rag reinforcement learning.arXiv
preprint arXiv:2505.14069.
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai,
Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025.
Deepresearcher: Scaling deep research via reinforce-
ment learning in real-world environments.arXiv
preprint arXiv:2504.03160.
Peilin Zhou, Bruce Leon, Xiang Ying, Can Zhang,
Yifan Shao, Qichen Ye, Dading Chong, Zhiling
Jin, Chenxuan Xie, Meng Cao, and 1 others. 2025.
Browsecomp-zh: Benchmarking web browsing abil-
ity of large language models in chinese.arXiv
preprint arXiv:2504.19314.
11

Dataset Language Number of Samples
glaveai/RAG-v1 En 100
InfoDeepSeek (Xi et al., 2025) Zh 245
BrowseComp-Zh (Zhou et al., 2025) Zh 289
Frames (Krishna et al., 2025) En 230
XBench (Chen et al., 2025) Zh 100
Deep Research Bench (Du et al., 2025) En&Zh 100
Table 5: Datasets used for the construction of RAGCap-
Bench. Note that the original Frames contain 824 sam-
ples, we subsample 230 to run the agentic RAG systems
on these 230 samples only. In addition, the original
Glaveai/RAG-v1 contains 51.4k rows, we select the first
100 rows with longest list of retrieved documents.
A Datasets
All open-source datasets used for constructing
RAGCap-Bench are listed in Table 5. The agen-
tic RAG systems used include WebThinker (Li
et al., 2025c), WebSailor (Li et al., 2025a), Web-
Dancer (Wu et al., 2025) and HiRA (Jin et al.,
2025b). We run these systems on InfoDeepSeek,
BrowseComp-Zh and Frames to obtain the interme-
diate outputs. Glaveai/RAG-v11dataset includes
retrieved documents as part of its content. The
remaining datasets provide human-annotated step-
wise problem-solving strategies, which are used to
construct planning questions.
BConvergent and Divergent Capabilities
Figure 5 illustrates the roles of convergent and di-
vergent planning capabilities for different types of
queries. Convergent planning capability is required
to gradually narrow down the search space in order
to reach a final deterministic answer. Divergent
planning capability is required to expand a user
query to explore multiple perspectives, interpreta-
tions and possibilities, so that the final answer is
well-rounded and considers diverse viewpoints.
C Examples of Errors
Figure 6 shows examples of the actual errors col-
lected from agentic RAG pipelines. The errors are
highlighted in red.
D Generation of MCQs
Since different datasets and pipelines generate in-
termediate outputs with varying structures and for-
mats, the prompts must be customised accordingly.
Here, we provide some examples of our generation
prompts to illustrate the core idea. These prompts
1https://huggingface.co/datasets/glaiveai/RAG-v1
Query
Sub-query1 Sub-query2Candidates:
S1, S2, S3
Sub-
query_S1=> Convergent :
Sear ch S1 & S2
Reject S3
Sub-
query_S2Sub-
query_T1...
Query
Sub-query 1 Sub-query N Sub-query 2 ......
Reject
Answer
Sub-query
1_1Sub-query
1_2Sub-query
2_4Sub-query
N_1Sub-query
N_k... ...Divergent :
Explore from different perspectives
Divergent :
Explore from
different perspectives
Answer...Please identify the name of this artist based on the following clues: He studied at the Central
Academy of Fine Arts in China and the Düsseldorf Art Academy in Germany , and pursued further
studies in Germany . During his time in Germany , he studied under three well-known artists, one
of whom set the record in 2012 for the highest auction price ever achieved by a living artist.Example 1 (T ranslated fr om Chinese):"Artist, China,
Central Academy of Fine Arts and 
the Düsseldorf Art Academy""Living artists, highest
auction price record
in 2012"Candidates:
 T1, T2
=> Convergent :
Sear ch T1
Reject T2
"T1's student, 
Düsseldorf Art Academy " "S1's teacher "
...
Which Obsidian plugins can ef fectively replicate Notion's multi-view database functionality (including Table,
Kanban, Calendar , and List views)? Please provide a detailed comparison of the strengths and weaknesses
of these pluginsExample 2:"MPC" "A2A" "Problem Space Addressed by A2A"
"Definition""Architecture""Design
Principles""Heterogeneous
Agent Ecosystems""Communication
and Data Silos""S2's teacher "Figure 5: Convergent (top) and divergent (bottom) plan-
ning capabilities required for different user queries.
can be adapted as needed for each specific pipeline
and dataset.
Figure 7 and Figure 8 show the prompts for gen-
erating MCQs using Error-Guided Generation. For
most questions using Vanilla Generation, except
for grounded reasoning, LLM-based generation is
not needed. Figure 9 show the prompt for gener-
ating grounded reasoning questions using Vanilla
Generation with LLM. Other questions generated
with Vanilla Generation follow the formats in Fig-
ure 2. Here we only show an example of evidence
extraction in Figure 10.
E Examples of MCQs
Figure 11 and Figure 12 are MCQ exam-
ples from RAGCap-Becnh. The full dataset
can be assessed at https://github.com/
jingru-lin/RAGCap-Bench.
F Results with Bare Prompt
Table 6 shows the evaluation results on RAGCap-
Bench using bare prompts.
G G. Evaluation Prompts
Figure 13 provides the prompts used for evaluating
the intermediate outputs from WebThinker.
12

H Bare and Informative Prompts
Figure 14 and Figure 16 are bare and infor-
mative prompts for grounded reasoning. Fig-
ure 15 and Figure 17 are bare and informative
prompts for noise-reliability. All prompts for
evaluation are available in the prompts folder
athttps://anonymous.4open.science/
r/RAGCap-Bench-5C02/README.md.
13

某 90 后中国男歌⼿， 2014 年在某档节⽬中翻唱了⼀⾸发表于 1979 年的歌曲，并于2025 年跨年晚会上再次与原唱同台演唱这⼀⾸歌。这⾸歌的名字是什么？   …  - 周深在2014年《中国好声⾳》翻唱《乡恋》的知名度与年龄匹配； - 《乡恋》原唱李⾕⼀的年代与歌曲历史地位吻合； - 新⽣代与原唱的合作模式在跨年晚会中常⻅（如2021年案例） 。  好的，现在根据搜索结果，周深在2014年的《中国好声⾳第三季》中翻唱了李⾕⼀1979年的经典歌曲《乡恋》 。⽽李⾕⼀作为原唱，两⼈曾在2021年的央视跨年晚会合作过。虽然问题提到的是2025年的跨年晚会，但考虑到这类合作通常是为了纪念经典，且周深与李⾕⼀的合作模式符合题⽬的描述，可以推断这⾸歌就是《乡恋》 。因此，最终答案应为《乡恋》 。     某地点名称出⾃发⽣于某周朝诸侯国⼀次⾮战争⾏为，主⻆为⼀位职权较⾼者，⾯对潜在羞辱选择退⾄⼀条窄道中，⽇后双⽅和解传为佳话。此地重建碑⽂由⼀位地⽅书法家撰写，现已成为⽂化象征。请问该地点是哪⾥？ …  好的，现在根据第三次搜索的结果，找到了⼀个可能的答案：“虞芮之讼”的典故。以下是分析： 1. **事件背景**：虞国和芮国是周朝时期的诸侯国，他们因为争执前往周国寻求解决，结果被周⼈的谦让⻛⽓感动，⾃⾏和解，将争议地让出作为公共⽤地，称为“闲⽥”。这符合“⾮战争⾏为”和“和解传为佳话”的条件。 2. **地名关联**：争议⼟地现在位于⼭⻄省平陆县洪池乡，被称为“闲⽥春⾊”，这可能就是题⽬中提到的地点名称。 3. **时间匹配**：事件发⽣在周⽂王时期，属于周朝，符合题⽬中的周朝诸侯国要求。 4. **碑⽂问题**：虽然搜索结果中没有直接提到碑⽂由地⽅书法家撰写，但考虑到该地作为⽂化象征，可能在后续重建或纪念时有相关碑⽂。例如，可能在明清时期有地⽅官员或书法家为纪念此事⽽题写碑⽂，从⽽符合题⽬中的描述。 5. **窄道的隐喻**：虽然事件本⾝没有提到“窄道”，但“耕者让畔、路⼈让道”可能隐含在狭窄的⼟地或道路上的谦让⾏为，可能被后世引申为“窄道”的象征意义。 综合以上分析，最可能的答案是“闲⽥春⾊”对应的地点，即⼭⻄省平陆县的“闲⽥”所在地，也就是“虞芮之讼”发⽣地。因此，答案应为该地的现代名称或相关地名。 根据搜索结果，该地现称为“闲⽥春⾊”，属于平陆县洪池乡。因此，最终答案应该是这个地点名称。 Figure 6: Error examples from agentic RAG pipelines.
14

You are an advanced agentic AI. You are given: a. a question b. planning traces (multiple steps, each step has a Plan and Found) Your task is to create a multiple-choice question (MCQ) to assess how good a student is at problem-solving. Please follow the following instruction for generating the MCQs: 1. Randomly select a step and use the steps up to this selected step as past thinking traces. 2. Create an MCQ to ask for appropriate next step. 3. Each MCQ has **at least four** options, each option is an advice for the next search step. 4. **At least one** options should contain a good next step. You can use the suggested next step or come up with a better one. 5. **At least two** options should contain a bad next step. Some examples of bad next step include but not limited to: a. non-converging logic: advise a next search query that does not narrow down the search scope b. misunderstands the question: advise a next search query that shows misunderstanding of the question c. redundant search: advise an unnecessary next search step that appear to be relevant to the question 6. One option should suggest that current information is enough and no further search is necessary. 7. Your MCQ should begin with "为了回答问题，你进⾏了以下步骤：[Previous Steps]。\n\n下⼀步最合适的做法有哪些？".  Yo u  * * m u s t * *  r e p l a c e  [ P r e v i o u s  S t e p s ]  w i t h  a  c o n c i s e  C h i n e s e  s u m m a r y  o f  p r e v i o u s  s t e p s ,  **no** advice or next-step reasoning should be included. **Example Question:** [example]  **Question Content:** Query: [query] Planning Traces: [planning_traces] Output:  Ensure that your output strictly follows the output format above. Do not add any extra explanations or comments—simply generate the multiple-choice question.        Figure 7: Example prompt for generating planning questions using Error-Guided Generation.
15

  You are an expert in retrieving information and question answering. Your task now is to design a multiple-choice question (MCQ) to evaluate the student's ability to write answer statements that are well-supported by the information retrieved. You are provided with: a. a query b. webpages  c. reasoning  **Requirements: ** 1. Extract **at least six** statements from the reasoning. Try to maximise the diversity of the statements. 2. Each statement should attempt to answer the question. 3. Format each statement as an MCQ option, label them as A, B, C, D, E (or more if needed). 4. Modiﬁed **at least two** statements so that each of them demonstrates an error. Some of the errors include but not limited to: - hallucinated support from the webpages - irrelevant to the question - contradict with retrieved information - contains an incorrect number or statistic - lacks necessary context or conditions from the original question - misunderstands the question 5. Begin your MCQ with "根据搜索到的⽹⻚，以下哪项陈述是错误的？"  **Question Content:** [query] [webpages]  Reasoning: [reasoning]  Output:  You do not add any extra explanations or comments - simply generate the MCQ.      Figure 8: Example prompt for generating grounded reasoning questions using Error-Guided Generation.
16

 You are an expert in extracting statements. Your task now is to extract statements from the given reasoning. a. a query b. reasoning  **Requirements: ** 1. Extract **at least six** statements from the reasoning. Try to maximise the diversity of the statements. 2. Each statement should attempt to answer the question. 3. Format each statements as an MCQ option, label them as A, B, C, D, E (or more if needed). 4. Begin your MCQ with "根据搜索到的⽹⻚，以下哪项陈述是错误的？"  **Question Content:** [query]  Answer Report: [reasoning]  You do not add any extra explanations or comments - simply generate the MCQ.                    Figure 9: Example prompt for generating grounded reasoning questions using Vanilla Generation.
 [query] 为了回答以上问题，你对以下关键词进⾏了搜索： [search_queries]  以下每个选项是⼀个搜索到的⽹⻚，哪些⽹⻚对回答问题没有帮助？ [webpages_as_mcq_options]                              
Figure 10: Vanilla Generation for evidence extraction.
17

ModelPlanning Evidence Extraction Grounded Reasoning Noise Robustness Overall
EMc F1c EMd EM F1 EM F1 EM aEMr F1r EM F1
fast-thinking models
Qwen2.5-72B-Instruct 31.37 64.31 72.00 28.99 79.56 39.62 79.98 62.16 10.00 68.05 39.09 72.98
Qwen-Plus w/o think 27.45 67.26 84.00 24.64 77.40 30.19 76.55 54.05 15.00 65.67 36.27 71.72
Deepseek-v3 43.14 69.93 80.00 20.29 71.93 24.53 69.56 34.14 15.00 61.93 32.86 68.34
GPT-4o 35.29 56.99 64.00 30.43 79.65 50.94 81.29 54.05 10.00 54.69 40.76 68.16
Gemini-1.5-Pro 52.94 74.58 80.00 14.49 75.51 41.51 80.63 37.84 5.00 73.19 35.97 75.98
slow-thinking models
Qwen3-8B 37.25 63.20 72.00 23.19 74.85 39.62 83.01 37.84 15.00 75.51 35.96 74.14
Qwen3-32B w/ think 39.22 67.84 84.00 28.99 57.29 37.74 84.06 35.14 10.00 74.92 37.72 71.02
QwQ-32B 19.61 58.21 80.00 28.99 76.70 49.06 83.24 51.35 20.00 72.43 40.88 72.64
Qwen3-235B-A22B w/ think 49.02 72.29 80.00 42.03 75.15 52.83 86.27 56.76 5.00 55.37 46.56 72.27
DeepSeek-R1 45.10 70.33 84.00 31.88 79.11 56.60 88.73 56.76 35.00 80.03 49.73 79.55
O1-mini 21.57 60.20 64.00 23.19 72.27 41.51 76.72 54.05 15.00 64.59 35.50 68.45
Gemini-2.5-Flash 50.98 74.79 80.00 31.88 77.68 39.62 79.83 78.38 15.00 72.20 45.02 76.17
Table 6: Performance of different fast- and slow-thinking LLMs using bare prompts. EM cand F1 cdenote the scores
for converging ability. EM ddenotes the EM score for diverging ability. EM adenotes the EM score for noise-abstain.
EMrand F1 rdenote the scores for noise-reliability. The scores are presented as percentages for clarity.
政治家a。曾中状元，他55岁上下任⼀地知州，受当地百姓⾼度赞扬。a曾⼒荐某⽼将，该⽼将与某唐朝名相有⾎脉联系，并在5岁不到失去⽗亲，40岁被贬，年近50⼜被贬。政治家a在今天有⼀同籍贯⼥星，于2017年进⼊演艺圈。这位⼥星是谁？  为了回答问题，你进⾏了以下步骤： Step 0: 搜索历史上著名的状元及其官职和年龄情况。发现王⼗朋是南宋绍兴⼆⼗七年（1157年）的状元，中状元时45岁，之后历任饶州、夔州、湖州、泉州等地知州，55岁左右确实在任知州。 Step 1: 确认王⼗朋的政治贡献，包括推荐将领的记录。发现王⼗朋确实推荐了张浚作为北伐主帅，张浚符合与唐朝名相张九龄有⾎脉联系，5岁丧⽗，以及40岁和50岁两次被贬的条件。  下⼀步最合适的做法有哪些？ A. 搜索其他与张浚相关的历史⼈物，看看是否符合其他相关条件。 B. 确认王⼗朋是否有推荐过其他将领，并核对这些将领的背景信息。 C. 如果已知王⼗朋的籍贯为温州乐清，则检查是否有其他温州乐清籍的⼥明星在2017年进⼊演艺圈，以确认答案。 D. 停⽌搜索，已掌握⾜够信息回答问题。 E. 搜索王⼗朋的籍贯。 F.  搜索张浚的籍贯。                    
Figure 11: An example from RAGCap-Bench.
18

 Question: 巴勃罗·毕加索去世的那⼀年，平克·弗洛伊德发⾏了哪张专辑？,"为了回答以上问题，你对以下关键词进⾏了搜索： 1. 平克·弗洛伊德 1973 专辑; 2. 巴勃罗·毕加索 去世年份; 3. 平克·弗洛伊德 专辑列表; 4. 毕加索 平克·弗洛伊德 1973;   以下每个选项是⼀个搜索到的⽹⻚，哪些⽹⻚对回答问题没有帮助： A. ⽹⻚标题：绘画⼤师毕加索是什么时间死的? - 搜狗问问|||⽹⻚时间：2010年01⽉27⽇|||⽹⻚内容：姓名:巴勃罗·毕加索(Pablo Picasso) ⽣卒:1881年10⽉25⽇—1973年4⽉8⽇ 描述:画家 籍贯:⻄班⽛ ⻄班⽛画家、雕塑家。法国共产党党员。是现代艺术(⽴体派)的创始⼈,⻄⽅现代派绘画的主要代表。他是⻄班⽛⼈,⾃幼有⾮凡的艺术才能,他的⽗亲是个美术教师,⼜曾在美术学院接受过⽐较严格的绘画训练,具有坚实的造型能⼒。 1973年4⽉8⽇ B. ⽹⻚标题：画家和模特-毕加索 1970年7⽉5⽇|||⽹⻚时间：2022年01⽉28⽇|||⽹⻚内容：毕加索名画(784): 784、画家和模特 毕加索 ⻄班⽛ 1970年 粉彩画 80x100cm 美国圣地亚哥艺术博物馆 这幅《画家和模特》(Woman with bird),创作于1970年7⽉5⽇,是毕加索⽥园时期(1946-1973)的作品。这⼀时期毕加索的艺术特点表现为,仍然以⽴体主义、现实主义和超现实主义⼿法相结合的抽象画为主,⼿法运⽤更加灵活娴熟,画⾯粗狂⼲劲,和谐统⼀。造型虽有夸张,但并不算是最激烈的变形,笔触运⽤也相对精致周到。 C. ⽹⻚标题：⼥⼈的侧影-毕加索 1963|||⽹⻚时间：2022年01⽉28⽇|||⽹⻚内容：⼥⼈的侧影 毕加索 ⻄班⽛ 1963年 布⾯油画 私⼈收藏 这幅《 ⼥⼈的侧影》(Grand proﬁl),创作于1963年,是毕加索⽥园时期(1946-1973)的作品。这⼀时期毕加索的艺术特点表现为,仍然以⽴体主义、现实主义和超现实主义⼿法相结合的抽象画为主,⼿法运⽤更加灵活娴熟,画⾯粗狂⼲劲,和谐统⼀。造型虽有夸张,但并不算是最激烈的变形,笔触运⽤也相对精致周到。 D. ⽹⻚标题：平克 * 弗洛伊德专辑⾳乐 CD  eBay|||⽹⻚时间：2022年06⽉06⽇|||⽹⻚内容：已应⽤ 2 个过滤条件 艺⼈: 平克·弗洛伊德 已应⽤过滤条件 类型: 专辑 已应⽤过滤条件 全部清除 艺⼈ 平克·弗洛伊德 已应⽤过滤条件 查看全部 类型 专辑 已应⽤过滤条件 查看全部 类别 流⾏乐 古典 爵⼠ 查看全部 版本 精选辑 查看全部 ⻛格 物品状况 价格 购买形式 全部物品 已应⽤过滤条件 全部过滤条件 迷你 LP Sleeve 仅适⽤于 (2001) PINK FLOYD Dark Side of the Moon tocp - 65740 ⽇本 (3) 3 个商品评分 - Mini-LP Sleeve ONLY for (2001) Pink Floyd Dark Side Of The Moon TOCP-65740 Japan 83.38元 107.14元 运费 333.70元 115.75元 运费 PINK FLOYD THE LATER YEARS 1987-2019 (CD DIGI PAK) NEW SEALED (S) (12) 12 个商品评分 - PINK FLOYD THE LATER YEARS 1987-2019 (CD DIGI PAK) NEW SEALED (S) 58.09元 313.96元 运费 仅剩 1 件! (18) 18 个商品评分 - Pink Floyd - A Momentary Lapse of Reason CD 1987 CK 40599 DADC EARLY PRESS 73.70元 原价: 86.71元 53.40元 运费 仅剩 1 件! Pink Floyd: The Dark Side Of The Moon (SACD) (Analogue Productions) (1) 1 个商品评分 - Pink Floyd: The Dark Side Of The Moon (SACD) (Anal E. ⽹⻚标题：平克·弗洛伊德录⾳室专辑(1970-2015) - 歌单 - ⽹易云⾳乐|||⽹⻚时间：2016年01⽉29⽇|||⽹⻚内容：介绍: 英国摇滚乐队平克弗洛伊德1970年⾄2015年发售的所有录⾳室专辑。 《Atom Heart Mother》1970.10.2 《Meddle》1971.10.30 《Obscured by Clouds》1972.6.2 《The Dark Side of the Moon》1973.3.1 《Wish Y ... F.  ⽹⻚标题： 【摇滚乐】平克·弗洛伊德《⽉之暗⾯》 1994 [FLAC+CUE/整轨]|||⽹⻚时间：2023年07⽉03⽇|||⽹⻚内容：专辑名称:Dark Side of the Moon   Figure 12: An example from RAGCap-Bench.
19

You are an excellent evaluator, good at assessing deep search systems. You will be given a **User Query**, a **Search Query** , some **Retrieved Evidence** and **Evidence Analysis** . Your task is to evaluate the quality of the **Evidence Analysis** .  ### Quality Evaluation Criteria Scores are assigned using a 1–10 Likert scale, where higher scores indicate the systems have selected all useful evidence to answer the user query, and lower scores indicate that the evidence selection is incomplete, oJ-topic, inaccurate, or redundant. 1. **Relevance** – Does the retrieved evidence address the user query? Is it clearly related to the main topic or query? 2. **Completeness** – Does the system misses key evidence pieces that would help to answer the query? 3. **Speciﬁcity** – Is the evidence precise and focused, rather than vague or overly broad?  ### Key Failure Patterns to Watch (a) **Shallow Keyword Matching** – The evidence appears related only because it shares similar keywords with the search query or user query, but it fails to address the actual intent. Example: [example 1]  (b) **Fail to Recognise Implicit Connections** – The system missed the deeper semantic or logical relationship needed to connect the user query to relevant evidence. Example: [example 2]  (c) **Missing Potential Evidence** - The evidence might have relevant information, but it would require browsing the page to obtain the information Example: [example 3]  ### Evidence to evaluate **User Query** [user_query]  **Search Query** [search_query]  ### Output Format [output_format] Figure 13: Prompt for evaluation on the intermediate outputs from WebThinker.
20

You are an advanced agentic AI. You are good at analysing the documents retrieved and write well-supported statements to answer the question. Please read the following webpages and answer the question.  [query]  [mcq_question]  Ensure that you output option alphabets (A,B,C,D etc.) only without any explanation. You should select all possible answers. However, if you think none of the answer should be selected, output 'none' .                             Figure 14: Bare prompt for grounded reasoning.
  You are an advanced agentic AI. You are good at detecting low-quality, potentially misleading or less credible sources of information.  Let's say you are writing a report on [topic].  Please select low-quality, potentially misleading or less credible web page from the options. [mcq_question]  Ensure that you output option alphabets (A,B,C,D etc.) only without any explanation. You should select all possible answers. However, if you think none of the answer should be selected, output 'none' . 
Figure 15: Bare prompt for noise robustness (reliability).
21

"""You are an advanced agentic AI. You are good at analysing the documents retrieved and ensuring your reasoning and written statements are grounded in evidence.  The following are some cases of the common mistakes when reasoning based on retrieved evidence:   Evidence Set:[webpage 1 begin]浅草寺・雷⻔。 1865年12⽉12⽇发⽣的⽥原町⼤⽕曾⼀度将浅草寺的⼤⻔烧毁,现在的⼤⻔是于1960年,即距⽕灾发⽣95年后,由松下电器创始⼈松下幸之助捐赠修复的。现在,雷⻔已不单是浅草寺的庄严的正⻔,更作为浅草的象征⽽闻名全国。 [webpage 1 end] [webpage 2 begin]东京浅草寺,据传说起源于1000多 年前,⼀对以捕⻥为⽣的兄弟在隅⽥川中发现了观⾳像,就建了这座寺院进⾏供奉,历史真是很悠久。现在的浅草观⾳寺建筑群为⼆战后重建, 其位置在东京与银座、新宿、池袋、涩⾕同为繁华街区,浅草寺⻔前的仲⻅世⼤道,布满各种⽇本传统⻛情旅游品销售店,是熙熙攘攘海外参拜客光顾之地。[webpage 2 end] [webpage 3 begin]浅草寺是东京历史悠久、⼈⽓旺的寺院,也是浅草地区的中⼼,供奉的本尊是圣观⾳。它始建于7世纪,因屡遭⽕灾,后被重建。相传因三位渔⺠出海捕⻥时捞起了⼀座约5.5厘⽶⾼的⾦观⾳⼩雕像,才修建了这座庙宇。 东京浅草寺图⽚⼤全[webpage 3 end]  Statement 1: "浅草寺的⼤⻔于20世纪初曾经由当时的东京市政当局组织修缮，为⽇后松下幸之助的捐赠打下基础。" (wrong) Explanation: it shows hallucinated support. There is no mention of any early 20th-century repair with government involvement.  Statement 2: "浅草观⾳寺是东京少数完整保留战前原貌的历史建筑之⼀" (wrong) Explanation: it shows contradiction with retrieved information. Webpage 2 mentions "现在的浅草观⾳寺建筑群为⼆战后重建"   Statement 3: "浅草寺每年接收⽆数海内外游客" (correct) Explanation: it is not explicitly stated but logically inferable from webpage 2 ("是熙熙攘攘海外参拜客光顾之地") and 3 ("浅草寺是东京历史悠久、⼈⽓旺的寺院").  You should be careful with the mistakes above as well as other possible mistakes. Please analyse the following webpages and answer the question.  [query]  [mcq_question]  Ensure that you output options alphabets (A,B,C,D etc.) only without any explanation. You should select all possible answers. However, if you think none of the answer should be selected, output 'none' .   Figure 16: Informative prompt for grounded reasoning.
22

 You are an advanced agentic AI. You are good at detecting low-quality, potentially misleading or less credible sources of information.  Examples of such webpages include but not limited to: a. company's websites that primarily promotional or focused on revenue generation b. websites where any users can upload content c. unoBicial or unveriﬁed websites  Now, let's say you are writing a report on "[topic]".    Please select low-quality, potentially misleading or less credible web page from the options.  [mcq_question]  Answer: \n  Ensure that you output option alphabets (A,B,C,D etc.) only without any explanation. You should select all possible answers. However, if you think none of the answer should be selected, output 'none' .  Figure 17: Informative prompt for noise robustness (reliability).
23