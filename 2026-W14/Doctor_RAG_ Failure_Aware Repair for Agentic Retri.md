# Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation

**Authors**: Shuguang Jiao, Chengkai Huang, Shuhan Qi, Xuan Wang, Yifan Li, Lina Yao

**Published**: 2026-04-01 13:13:27

**PDF URL**: [https://arxiv.org/pdf/2604.00865v1](https://arxiv.org/pdf/2604.00865v1)

## Abstract
Agentic Retrieval-Augmented Generation (Agentic RAG) has become a widely adopted paradigm for multi-hop question answering and complex knowledge reasoning, where retrieval and reasoning are interleaved at inference time. As reasoning trajectories grow longer, failures become increasingly common. Existing approaches typically address such failures by either stopping at diagnostic analysis or rerunning the entire retrieval-reasoning pipeline, which leads to substantial computational overhead and redundant reasoning. In this paper, we propose Doctor-RAG (DR-RAG), a unified diagnose-and-repair framework that corrects failures in Agentic RAG through explicit error localization and prefix reuse, enabling minimal-cost intervention. DR-RAG decomposes failure handling into two consecutive stages: (i) trajectory-level failure diagnosis and localization, which attributes errors to a coverage-gated taxonomy and identifies the earliest failure point in the reasoning trajectory; and (ii) tool-conditioned local repair, which intervenes only at the diagnosed failure point while maximally reusing validated reasoning prefixes and retrieved evidence. By explicitly separating error attribution from correction, DR-RAG enables precise error localization, thereby avoiding expensive full-pipeline reruns and enabling targeted, efficient repair. We evaluate DR-RAG across three multi-hop question answering benchmarks, multiple agentic RAG baselines, and different backbone models. Experimental results demonstrate that DR-RAG substantially improves answer accuracy while significantly reducing reasoning token consumption compared to rerun-based repair strategies.

## Full Text


<!-- PDF content starts -->

Doctor-RAG: Failure-Aware Repair for Agentic
Retrieval-Augmented Generation
Shuguang Jiao
Harbin Institute of Technology
Shenzhen, China
24S151028@stu.hit.edu.cnChengkai Huang∗
Macquarie University and UNSW
Sydney, Australia
chengkai.huang1@unsw.edu.auShuhan Qi∗
Harbin Institute of Technology
Shenzhen, China
shuhanqi@cs.hitsz.edu.cn
Xuan Wang
Harbin Institute of Technology
Shenzhen, China
wangxuan@cs.hitsz.edu.cnYifan Li
Harbin Institute of Technology
Shenzhen, China
liyifan@cs.hitsz.edu.cnLina Yao
UNSW and CSIRO’s Data61
Sydney, Australia
lina.yao@unsw.edu.au
Abstract
Agentic Retrieval-Augmented Generation (Agentic RAG) has be-
come a widely adopted paradigm for multi-hop question answering
and complex knowledge reasoning, where retrieval and reasoning
are interleaved at inference time. As reasoning trajectories grow
longer, failures become increasingly common. Existing approaches
typically address such failures by either stopping at diagnostic anal-
ysis or rerunning the entire retrieval–reasoning pipeline, which
leads to substantial computational overhead and redundant reason-
ing. In this paper, we proposeDoctor-RAG (DR-RAG), a unified
diagnose-and-repair framework that corrects failures in Agentic
RAG through explicit error localization and prefix reuse, enabling
minimal-cost intervention. DR-RAG decomposes failure handling
into two consecutive stages: (i) Trajectory-level failure diagnosis
and localization, which attributes errors to a coverage-gated tax-
onomy and identifies the earliest failure point in the reasoning
trajectory; and (ii) Tool-conditioned local repair, which intervenes
only at the diagnosed failure point while maximally reusing vali-
dated reasoning prefixes and retrieved evidence. By explicitly sepa-
rating error attribution from correction, DR-RAG enables precise
error localization, thereby avoiding expensive full-pipeline reruns
and enabling targeted, efficient repair. We evaluate DR-RAG across
three multi-hop question answering benchmarks, multiple agen-
tic RAG baselines, and different backbone models. Experimental
results demonstrate that DR-RAG substantially improves answer
accuracy while significantly reducing reasoning token consumption
compared to rerun-based repair strategies.
CCS Concepts
•Information systems→Information retrieval.
Keywords
Retrieval-Augmented Generation, Large Language Model, Agentic
RAG
1 Introduction
In recent years, Agentic Retrieval-Augmented Generation (Agentic
RAG) has emerged as a dominant paradigm for multi-hop question
answering and complex knowledge reasoning [ 16,21,29,36,45].
By dynamically generating search queries, invoking retrievers, and
∗Corresponding authors.
87
27553
28
6
15888209
Total
362Format Error
Reasoning 
Error
Dataset
Noise
Search ErrorRetriever 
ErrorFormat ErrorDataset Noise
Cov<1
Cov=1Distribution of Error TypesFigure 1: The statistics of different error types in Agentic
RAG under the ReAct baseline on HotpotQA (500 samples).
performing multi-step reasoning during inference, Agentic RAG
enables models to progressively plan and solve complex problems,
substantially improving coverage and generalization over static
RAG systems [15, 32, 34, 35, 46, 48].
However, as reasoning chains become longer and interaction
rounds increase, failures in Agentic RAG become more frequent
and difficult to fully avoid [ 1,25,33]. Since errors in multi-step
reasoning may arise at arbitrary stages and accumulate over time,
simply improving single-step reasoning or retrieval performance is
often insufficient to ensure stable system behavior [ 2,8,41]. As a
result, the challenges faced by Agentic RAG extend beyond whether
a correct answer can be generated, and increasingly concern:
How can an Agentic RAG system localize the
cause of a failure and repair it with minimal
recomputation once the failure occurs?
Existing studies have analyzed and modeled failures in RAG
systems from different perspectives [ 1,3,11,25,33]. One line of
work focuses on systematically categorizing and diagnosing errors
in RAG outputs [ 33,47], but does not further address automatic
repair or performance improvement after failures occur. Another
line of work incorporates error analysis into agentic workflows [ 8],
triggering subsequent repair based on diagnostic feedback after
a complete reasoning execution; however, such repair typically
relies on a replanning–rerun strategy, discarding the original trajec-
tory and re-executing the retrieval–reasoning pipeline. AlthougharXiv:2604.00865v1  [cs.IR]  1 Apr 2026

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Shuguang Jiao et al.
these two approaches differ in how error analysis is conducted
and utilized, they either stop at diagnostic evaluation or depend on
expensive full-pipeline re-execution, and do not reuse validated pre-
fixes and previously retrieved evidence, and thus recompute most
steps during repair. Consequently, a fundamental question remains
unanswered:once a failure has occurred, how can an Agentic
RAG system repair it at minimal cost?Figure 1 illustrates the
distribution of failure types observed in Agentic RAG under the
ReAct baseline on HotpotQA. The results show that failures arise
from diverse sources, rather than a single dominant cause. This
diversity suggests that failures are systematic and multi-faceted,
making unconditional reruns or monolithic repair strategies inher-
ently inefficient.
To address this problem, we proposeDoctor-RAG (DR-RAG), a
unified diagnose-and-repair framework for Agentic RAG, designed
to enable efficient failure correction through explicit error attri-
bution and minimal-cost intervention. Unlike existing approaches
that directly retry or replan the entire retrieval–reasoning pipeline
upon failure, DR-RAG explicitly models failure correction as two
consecutive stages: error diagnosis and conditional repair.
Specifically, DR-RAG consists of two tightly coupled core com-
ponents.(1) Taxonomy-Constrained Error Diagnosis and Lo-
calization.When a reasoning failure occurs, DR-RAG treats the
complete agentic RAG reasoning trajectory as the object of diag-
nosis. Leveraging evidence coverage signals, the diagnosis mod-
ule performs coarse-to-fine failure attribution, mapping continu-
ous reasoning failures into a set of executable error types. Unlike
approaches that assign error labels only at the output level, DR-
RAG furtherlocalizes the earliest failure pointat the trajectory
level, identifying the first reasoning or retrieval step that deviates
from a valid solution path, thereby providing structured signals
required for minimal intervention in subsequent repair.(2) Tool-
Conditioned Local Repair.Given the diagnosed error type and
failure location, DR-RAG selects a corresponding tool-conditioned
repair operator rather than regenerating a complete solution plan.
Each repair operator intervenes only at the identified failure point
while maximizing reuse of previously validated reasoning prefixes
and retrieved evidence. Concretely, repair operations include light-
weight answer rewriting for format errors, evidence-preserving
re-reasoning under full coverage, query rewriting and retrieval
enhancement for retrieval failures, and planning-based repair for
reasoning-induced retrieval errors. By constraining repair to the
necessary components, DR-RAG significantly reduces the cost of
retrieval and reasoning repair while preserving coherence across
repaired trajectories.
Our main contributions are summarized as follows:
•We are the first to formally formulate failure handling in Agentic
RAG at the system level, modeling correction as diagnosis-guided
minimal-cost intervention rather than full-pipeline retry.
•We propose a coverage-gated error taxonomy and a trajectory-
based diagnosis method that explicitly distinguishes retrieval
and reasoning failures while localizing the earliest failure point.
•We design a set of tool-conditioned repair operators that enable
local repair with prefix reuse, reducing correction cost.•Experiments across multiple datasets, baselines, and model scales
show that DR-RAG substantially improves accuracy while greatly
reducing retrieval calls and reasoning token usage.
2 Related Work
2.1 Retrieval Augmented Generation
Large Language Models (LLMs) are prone to hallucinations and lack
access to up-to-date knowledge. Retrieval-Augmented Generation
(RAG) addresses these limitations by integrating external knowl-
edge sources at inference time, improving factual accuracy without
requiring model retraining [ 12,23]. Early work, exemplified by the
original RAG framework [26], established the retriever–generator
architecture that underpins retrieval-enhanced generation. Build-
ing on this paradigm, subsequent studies progressively extend RAG
with more advanced reasoning and retrieval mechanisms, including
Chain-of-Thought [ 45], Tree-of-Thought reasoning [ 20,44], and
adaptive retrieval strategies [ 18]. These advances enable effective
multi-step reasoning in RAG systems.
2.2 Agentic RAG
In recent years, Agentic RAG (e.g., ReAct [ 45], Search-o1 [ 27],
Search-R1 [ 21]) has formulated reasoning and tool invocation (re-
trieval [ 7], memory [ 28], and summarization [ 5]) as a unified se-
quential decision-making process. Through policy learning [ 6,9],
these approaches enable models to autonomously select search and
information processing actions during reasoning. Existing work
[4,10,39] typically optimizes reasoning and retrieval policies using
outcome-level rewards or preference signals, thereby improving
performance on multi-step search and complex reasoning tasks.
More recently, several studies have investigated robustness and
failure diagnosis in RAG and agentic RAG systems. RAGChecker
[33] is a claim-level entailment-based diagnostic framework that
evaluates RAG system performance but does not address how to
repair or improve the system after errors are identified. RAG-Critic
[8] responds to detected error types by replanning a new reason-
ing and retrieval trajectory, without localizing the earliest error
or reusing validated prefix trajectories. As a result, failed trajec-
tories are still treated as indivisible units, leading to redundant
computation and limited error correction efficiency.
In contrast, how torepaira failed agentic RAG trajectory in
a fine-grained and cost-efficient manner remains underexplored
[14,17,31]. Prior work largely lacks explicit error localization
within trajectories and does not consider reusing validated rea-
soning prefixes or retrieved evidence. This gap motivates our work,
which treats agentic RAG failures as structured objects that can be
diagnosed, localized, and selectively repaired rather than blindly
rerun.
3 Agentic RAG Task Formulation
We consider the agentic retrieval-augmented generation (RAG)
setting for multi-step question answering, where a language model
solves a query through an explicit sequence of interleavedreasoning,
search, andinformationactions.
Given a question 𝑥∈ X , the system executes a sequence of
actions, each corresponding to one of the following modules:

Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
•Reason(𝑟): generate an intermediate reasoning step based on
the current context and previously retrieved information;
•Search(𝑞): issue a search query derived from the current rea-
soning state;
•Information( 𝑑): retrieve documents in response to the query
and incorporate them into the context for subsequent reasoning.
The execution produces an action-level trajectory
𝑦={𝜌(1),𝜌(2),...,𝜌(𝐾),𝑎pred},(1)
where𝑘indexes actions in the trajectory and action 𝜌(𝑘)∈𝑟,𝑞,𝑑
corresponds to a reasoning, search, or information acquisition ac-
tion, respectively, and 𝑎preddenotes the final predicted answer pro-
duced by the model. We treat the final predicted answer 𝑎predas a
terminal action in the trajectory.
In typical executions, these actions follow an interleaved pat-
tern, such as reasoning actions followed by search and information
acquisition actions. However, the trajectory length and structure
may vary across executions, depending on the intermediate reason-
ing outcomes, especially after localized repair. This formulation is
consistent with the execution flow illustrated in Figure 2 (a).
4 Methodology
4.1 DR-RAG Overview
Figure 2 presents the overall workflow of DR-RAG, a diagnose-
and-repair framework for agentic RAG. Throughout this section,
trajectories are indexed at the action level, with 𝑘denoting the
position of an atomic reasoning or retrieval action. Unlike prior
approaches that rerun the entire reasoning–retrieval pipeline, DR-
RAG treats a failed trajectory as a reusable object and corrects it
through targeted and localized intervention.
We focus on repairable agentic RAG failures; Dataset Noise (e.g.,
ambiguous questions) are considered non-actionable and excluded
from the diagnosis and repair pipeline. The framework then pro-
ceeds in two sequential stages: (1) taxonomy-constrained error
diagnosis and localization, and (2) tool-conditioned local repair.
In the diagnosis stage, DR-RAG first assesses the sufficiency of
the retrieved documents and then performs coverage-constrained
error classification and localization based on observable trajectory
signals, which serves as a gating signal that constrains the space
of plausible failure causes. In the repair stage, DR-RAG selects
a specialized repair operator based on the diagnosis result and
intervenes only at the identified failure point, while efficiently
reusing all previously validated prefix steps.
4.2 Diagnosis Module: Taxonomy-Constrained
Error Diagnosis and Localization
This section presents a unified diagnosis module that jointly defines
(i) a taxonomy of agentic RAG failure modes and (ii) a coverage-
guided procedure for error classification and localization. The mod-
ule analyzes a failed trajectory without modifying it and outputs a
structured diagnosis that directly conditions downstream repair.
Given an input question 𝑥, the agentic RAG system generates
an action-level trajectory 𝑦={𝜌(1),...,𝜌(𝐾),𝑎pred}, where𝑎predis
treated as the terminal action. We focus on failure cases where the
predicted answer is incorrect, i.e.,EM(𝑎 pred,𝑎★)=0.Table 1: Coverage-gated error taxonomy for agentic RAG
trajectories with localized failure attribution.
Coverage Error Type Localized Component
Cov(𝑦)=1Format Error answer
Reasoning Error reasoning
Cov(𝑦)<1Format Error answer
Retriever Error information
Search Error search
The goal of the diagnosis module is to determine (i)whythe
trajectory fails, in terms of the error category; (ii)wherethe failure
first occurs along the trajectory at the action level. In contrast to
RAG-Critic, our diagnosis process does not use the ground-truth
answer as auxiliary supervision and relies solely on the generated
trajectory. Formally, we define a diagnosis operator:
D𝜙:(𝑥,𝑦)→(𝑐,𝑘†),(2)
where𝑐∈C denotes an error category from a predefined taxonomy,
and𝑘†is an optional localization index indicating the earliest action
in the trajectory that is identified as erroneous.
4.2.1 Coverage-Guided Diagnosis.As illustrated in Figure 2(a), di-
agnosis is guided by an assessment of evidence coverage as an initial
gating signal. Given the retrieved documents in trajectory 𝑦, the
module first determines whether the currently retrieved documents
are sufficient to answer the input question.
We define the coverage indicator Cov(𝑦)∈{ 0,1}as a sufficiency
signal indicating whether the retrieved documents in 𝑦provide
adequate evidence to answer the input question:
•Cov(𝑦)= 1if the retrieved documents in 𝑦are assessed to be
sufficient to answer𝑥;
•Cov(𝑦)=0otherwise.
In practice, Cov(𝑦) can be instantiated either using oracle evi-
dence annotations when available, or via an LLM-based sufficiency
judge when golden evidence is not provided by the dataset. This
coverage assessment determines whether additional retrieval is
required and constrains the set of admissible error categories con-
sidered in subsequent diagnosis. Notably, coverage is evaluated
solely based on the retrieved documents and the input question,
without access to any golden or oracle evidence.
4.2.2 Error Taxonomy under Coverage Constraints.Based on the
coverage assessment, the diagnosis module restricts the candidate
error space in a deterministic manner, eliminating implausible error
types. Table 1 summarizes the resulting coverage-gated error tax-
onomy and the corresponding localized components. The coverage
signal indicates whether the retrieved documents in the trajectory
are sufficient to support answering the input question, which in
turn constrains the set of plausible failure causes.
Full Evidence Coverage ( Cov(𝑦)= 1).When the retrieved doc-
uments are assessed to be sufficient, failures cannot be attributed
to missing or insufficient retrieval. In this case, the failure is local-
ized to the reasoning action that produces an invalid or incomplete
search query, rather than the retrieval module itself.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Shuguang Jiao et al.
Question
Predicted Answer
Reasoning ProcessFailed RAG Trajectory
 Cov(Trajectory)=1 ?    Coverage-based Split Error Candidates
Format Error
Reasoning Error
Retriever Error
Search Error
Format Error
LLM-based Diagnosis�†: Error Location�: Error Type
Error Taxonomy Constraints
Cov=1
Cov<1
<reason> <search> <info> <answer> <reason> <search> <info> ......
Reasoning Error Retriever Error Search Error Format Error
Question Trajectory
Format Repair Reasoning Repair Retriever Repair Search Repair
Cov<1 Cov=1 Cov=1/Cov<1Reuse prefixRepair 
Router Trajectory truncation at �†
[0,...,�†]Retrieve? No Retrieve? No Retrieve? Yes Retrieve? Yes
[�†,...,�]Input:     
Trajectory [0,...,�†)
Wrong AnswerInput:     
Trajectory [0,...,�†)
Docs in [�†,...,�] Input:     
Trajectory [0,...,�†)
Queries in [�†,...,�] Input:    
Trajectory [0,...,�†)
Answer RewritingRepair:
Reasoning
All Retrieved Docs
Query Rewriting
Search (Top-10)
SearchReasoningPlan Generation
Cache 
Documents/QueriesRepair: Repair: Repair:
×N
Reasoning
Reasoning
(a) Taxonomy-Constrained Error Diagnosis and Localization
(b) Tool-Conditioned Local Repair
Reasoning with LLM
Rewriting with LLM
Searching with Retriever
Planning a Solution PathChose operator from
{Format, Reasoning, Retriever, Search}Are all golden 
evidence retrieved?
Figure 2: Overview of DR-RAG. Given a failed agentic RAG trajectory, the framework performs taxonomy-constrained diagnosis
and localization, followed by tool-conditioned local repair that intervenes at the failure point while reusing validated prefixes.
The admissible error categories are therefore limited to:
•Format-Invalid Answer (Format Error), where the predicted
answer is semantically correct but violates strict output or EM
constraints, and is localized to the answer prediction step;
•Reasoning Logic Error (Reasoning Error), where the model
performs incorrect reasoning over sufficient evidence, leading to
an incorrect answer. This error type admits a well-defined failure
point and requires trajectory localization.
Partial Evidence Coverage ( Cov(𝑦)< 1).When the retrieved
documents are insufficient, failures may arise either from retrieval
itself or from earlier reasoning steps that misdirect retrieval. In this
case, the admissible error categories include:
•Format-Invalid Answer (Format Error), where the predicted
answer is semantically correct but violates strict output or EM
format constraints, and is localized to the answer prediction step,
regardless of evidence coverage;
•Retriever Failure (Retriever Error), where valid and well-
formed queries fail to retrieve relevant information, indicating a
limitation of the retrieval component; and•Reasoning-Induced Missing Retrieval (Search Error), where
incorrect reasoning leads to invalid or incomplete queries, caus-
ing relevant evidence to be missed.
This coverage-gated taxonomy ensures mutually exclusive and
trajectory-consistent error classification, and directly determines
when and where local repair is applied.
4.2.3 Taxonomy-Constrained Error Localization.Within the coverage-
constrained candidate set, the diagnosis module performs fine-
grained error classification based on observable trajectory signals,
including reasoning steps, issued queries, retrieved documents,
and the final predicted answer. No additional retrieval is triggered,
and no ground-truth answer or oracle information is used during
diagnosis. For error types that admit partial trajectory reuse, the
module further performs error localization at the action level. Given
the action-level trajectory 𝑦={𝜌(1),...,𝜌(𝐾)}defined above, the
diagnosis module performs error localization at the action level.
The error localization index is defined as:
𝑘†=min{𝑘|𝜌(𝑘)is classified as erroneous},(3)
where the classification is performed under the coverage-constrained
error taxonomy. Intuitively, 𝑘†identifies the earliest action that is

Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
inconsistent with the admissible behavior of its error category, such
as faulty reasoning, ineffective retrieval, or misdirected search.
All actions preceding 𝑘†are treated as conditionally valid and
form a reusable prefix of the trajectory. In contrast, the suffix start-
ing from𝑘†is regarded as unreliable and discarded during repair.
This localization enables targeted intervention at the earliest failure
point while preserving validated reasoning and retrieval steps.
4.2.4 Diagnosis Output.The diagnosis module outputs a struc-
tured tuple(𝑐,𝑘†), where𝑘†denotes the earliest localized failure
point along the trajectory. For format errors, the failure is local-
ized to the terminal answer prediction step. This diagnosis output
serves as the control signal for selecting and parameterizing tool-
conditioned local repair operators in the subsequent repair stage.
4.3 Repair Module: Tool-Conditioned Local
Repair
Given the diagnosis output (𝑐,𝑘†)from the diagnosis module, the re-
pair stage performs targeted intervention on the failed agentic RAG
trajectory. Rather than restarting the entire reasoning–retrieval
process, DR-RAG applies atool-conditioned repair operatorselected
according to the diagnosed error type. This design enables partial
reuse of conditionally valid trajectory prefixes and avoids redun-
dant computation.
Formally, we define a set of repair operators: F={𝑓 1,...,𝑓𝐿},
where each operator 𝑓ℓis specialized for repairing a specific error
category. Operator selection is diagnosis-driven: 𝑓=S(𝑐), where
S:C→Fmaps an error type𝑐to its repair operator.
4.3.1 Repair under Full Evidence Coverage ( Cov(𝑦)= 1).When the
diagnosis module determines that the retrieved documents in the
trajectory are sufficient, repair focuses on correcting reasoning or
answer generation, without issuing additional retrieval actions.
Format-Invalid Answer Repair.For format-invalid answer errors,
the trajectory reasoning and retrieved documents are treated as
conditionally valid. Repair is therefore restricted to rewriting the
final answer to satisfy task-specific output constraints. The lan-
guage model is prompted with the original question, the retrieved
documents, and formatting instructions, and is asked to re-generate
the final answer. No reasoning or retrieval actions are modified.
Reasoning Logic Error Repair.For reasoning logic errors under
full coverage, the failure originates from an incorrect reasoning
action over sufficient evidence. We truncate the trajectory at the
localized error index𝑘†and retain the prefix:
𝑦prefix={𝜌(1),...,𝜌(𝑘†−1)},(4)
where𝜌(𝑘)denotes the𝑘-th reasoning or retrieval action.
LetD(𝑦) denote the set of document observation actions in
trajectory𝑦. All retrieved documents are aggregated as:
𝐷(𝑦)=Ø
𝜌∈D(𝑦)𝜌.(5)
The model is then prompted to re-perform reasoning from the
retained prefix, grounded on the aggregated document set 𝐷(𝑦) ,
while discarding previously generated incorrect reasoning steps.
This operator corrects reasoning errors without additional retrieval.4.3.2 Repair under Partial Evidence Coverage ( Cov(𝑦)< 1).When
retrieved evidence is insufficient, repair must address retrieval fail-
ures as well as potential answer-generation issues.
Format-Invalid Answer Repair.As in the full-coverage case, format-
invalid answer errors under partial coverage are handled by rewrit-
ing the final answer. No reasoning or retrieval actions are altered.
Retriever Failure Repair.Retriever failure errors arise when well-
formed queries fail to retrieve sufficient information. We truncate
the trajectory at 𝑘†and retain the prefix 𝑦prefix. All retrieval queries
issued prior to the failure are collected as: Qprev={𝑞(1),...,𝑞(𝑘†−1)}.
These queries are rewritten to improve recall while preserving their
original intent. In addition, the retriever is configured with an in-
creased top- 𝑘to return a larger candidate document set. The model
then re-generates the final answer based on the retained prefix and
the augmented retrieval results.
Reasoning-Induced Missing Retrieval Repair.Since the failure orig-
inates from an incorrect reasoning action that misguides retrieval,
both reasoning and retrieval steps beyond 𝑘†are re-generated. After
truncating the trajectory at 𝑘†, the model is prompted to gener-
ate a high-level solution plan conditioned on the original question
and the retained prefix. This plan decomposes the task into a se-
quence of reasoning and retrieval steps, which are then executed
to produce a new trajectory suffix. Unlike retriever failure repair,
both reasoning and retrieval paths are allowed to diverge from the
original trajectory beyond the truncation point.
4.3.3 Summary of Repair Operators.Across all error categories,
repair operators share a common structure: they preserve condi-
tionally valid trajectory prefixes, intervene only at the diagnosed
failure point 𝑘†, and restrict modifications to the minimal com-
ponents required by the error type. This tool-conditioned design
forms a coherent interface with the diagnosis module and enables
efficient, localized repair for agentic RAG trajectories.
5 Experiments
5.1 Research Questions
In this section, we address the following Research Questions (RQs):
•RQ1:Does DR-RAG improve overall RAG performance while
reducing token consumption compared to full rerun baselines?
•RQ2:Do error taxonomy and error localization each play an
essential role in DR-RAG?
•RQ3:What mechanisms enable DR-RAG to achieve lower token
usage than retry-based repair methods?
•RQ4:How accurate are error diagnosis and localization across
different error types?
•RQ5:How does repair success vary across different error types?
•RQ6:What is the maximum achievable repair success rate under
oracle diagnosis and localization?
5.2 Experiment Settings
5.2.1 Datasets and Metrics.We conduct experiments on three
multi-hop question answering benchmarks: HotpotQA, 2Wiki, and
MuSiQue. HotpotQA [ 43] is a large-scale benchmark for multi-hop
reasoning over Wikipedia, 2WikiMultihopQA (2Wiki) [ 13] focuses

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Shuguang Jiao et al.
Table 2: Main results on three multi-hop QA benchmarks. We report EM, F1, ROUGE-L (R-L), Token Cost, and Repair Rate for
different agentic RAG baselines with DR-RAG, Rerun, and Step-wise retry. DR-RAG consistently improves answer quality
while reducing token consumption across datasets and backbone models. Best results are in bold.
Baseline MethodTokens
Cost(↓)Repair
Rate(↑)HotpotQA 2Wiki MuSiQue
ΔEMΔF1ΔR-LΔEMΔF1ΔR-LΔEMΔF1ΔR-L
Qwen3-8B
ReAct [45]Rerun 8,113 15.7 17.4 19.0 18.6 11.6 15.0 14.9 6.0 7.4 7.3
Step-wise 9,927 11.5 12.4 14.4 13.8 9.4 14.0 13.9 3.6 3.8 3.7
RAG-Critic [8] 15,610 17.9 18.0 18.6 18.4 13.4 14.5 14.5 8.6 8.1 8.5
DR-RAG 5,261 26.1 25.8 26.0 25.0 19.6 20.8 20.9 10.0 9.1 9.1
Search-o1 [27]Rerun 6,809 7.3 6.6 6.9 7.1 3.4 5.9 6.0 4.6 3.6 3.6
Step-wise 7,320 7.7 6.6 6.8 5.8 5.4 7.6 7.6 3.0 3.4 3.0
RAG-Critic [8] 19,763 9.7 6.2 6.6 7.0 6.2 6.0 6.3 7.6 8.5 8.8
DR-RAG 4,987 20.8 13.8 12.5 12.2 14.6 13.6 13.7 10.0 10.0 10.0
Search-R1 [21]Rerun 6,632 16.7 12.2 13.0 13.0 15.6 15.9 16.2 6.6 8.2 8.1
Step-wise 7,057 7.3 4.8 5.7 5.2 8.6 10.3 10.9 1.4 2.0 2.2
RAG-Critic [8] 16,099 11.9 7.4 8.0 8.0 10.2 9.3 10.1 7.8 8.7 9.5
DR-RAG 5,395 20.6 12.4 14.2 13.9 18.8 18.0 18.5 10.0 10.5 10.2
LLaMA-3.1-8B-Instruct
ReAct [45]Rerun 14,787 14.0 17.4 18.1 17.5 9.4 12.6 12.5 4.2 4.5 4.3
Step-wise 21,239 13.0 16.2 17.5 16.7 8.0 12.7 12.6 4.6 4.7 4.8
RAG-Critic [8] 8,128 10.7 12.4 12.2 11.7 7.4 8.3 8.1 4.0 2.2 2.3
DR-RAG 7,196 20.1 21.4 20.7 20.2 12.4 12.8 12.7 7.0 5.4 5.2
Search-o1 [27]Rerun 12,065 5.6 6.0 5.0 4.7 2.0 3.6 3.5 2.8 3.6 3.4
Step-wise 16,477 6.3 5.4 4.5 3.7 3.2 4.8 4.6 4.0 4.4 4.0
RAG-Critic [8] 9,486 4.6 4.2 3.5 3.6 2.8 2.9 2.9 1.8 1.6 1.7
DR-RAG 6,609 13.9 12.8 10.8 10.6 6.6 4.9 4.7 5.6 4.8 4.6
Search-R1 [21]Rerun 9,502 6.7 5.6 7.3 6.9 5.0 6.3 6.4 3.4 4.6 4.9
Step-wise 13,162 5.3 5.2 6.0 5.6 3.4 5.8 6.0 2.4 2.8 3.3
RAG-Critic [8] 7,519 2.6 2.0 0.8 0.7 2.4 1.5 1.7 1.0 0.5 0.6
DR-RAG 7,268 12.8 9.0 9.4 9.0 9.2 8.3 8.5 5.2 5.9 5.9
Qwen3-4B
ReAct [45]Rerun 8,773 15.0 17.0 18.4 17.9 11.2 15.1 14.9 5.0 5.8 5.6
Step-wise 9,871 9.8 11.4 12.0 11.7 6.2 9.9 9.9 4.2 3.3 3.3
RAG-Critic [8] 14,186 12.1 12.6 13.0 12.6 9.6 10.7 10.7 4.8 4.3 4.4
DR-RAG 5,432 23.1 22.2 21.1 20.4 16.0 17.3 17.1 8.4 8.5 9.1
Search-o1 [27]Rerun 7,881 7.7 7.2 7.4 7.0 3.8 6.4 6.5 4.0 4.4 4.2
Step-wise 7,664 5.3 4.0 3.9 3.4 4.0 6.6 6.7 2.6 2.3 1.9
RAG-Critic [8] 19,770 5.8 6.0 5.4 5.9 2.2 2.9 3.1 3.0 3.7 3.9
DR-RAG 5,732 17.1 10.2 9.3 8.6 11.2 10.7 10.6 9.2 10.5 10.4
Search-R1 [21]Rerun 7,874 13.5 8.4 10.3 10.4 15.0 14.8 14.8 4.2 5.9 6.3
Step-wise 6,162 5.7 3.8 4.5 4.4 5.4 9.0 9.5 2.6 3.7 3.9
RAG-Critic [8] 17,685 6.1 4.2 4.3 4.4 7.0 6.2 6.5 1.2 1.1 1.7
DR-RAG 5,728 17.9 10.4 10.9 10.9 15.4 14.8 15.3 5.6 6.7 6.7
on explicit multi-hop reasoning paths, and MuSiQue [ 38] contains
2–4 hop questions constructed from multiple single-hop datasets.
All three datasets provide gold supporting evidence, enabling fine-
grained analysis of retrieval coverage and error diagnosis.
For evaluation, we report standard question answering metrics,
including Exact Match (EM), F1, and ROUGE-L [ 30]. EM measures
whether the predicted answer exactly matches the ground truth. F1
scores evaluate the token-level overlap, while ROUGE-L measures
lexical similarity based on the longest common subsequence.
To evaluate error correction effectiveness, we report theRepair
Rate, defined as the proportion of instances that are initially in-
correct (i.e., EM= 0) but are successfully corrected to an exact
match after repair (i.e., repaired EM= 1). For overall performance,we report the absolute improvement, denoted as Δ, relative to the
unrepaired baseline. Specifically, ΔEM measures the net increase
in the aggregate EM score attributable to the repair process.
5.2.2 Baselines.We evaluate DR-RAG on three representative
agentic RAG baselines:ReAct[ 45],Search-o1[ 27], andSearch-
R1[21], all of which generate multi-step trajectories interleaving
reasoning and retrieval. We compare DR-RAG with the following
repair strategies applied to the same failed trajectories:
(1) Rerundiscards the entire failed trajectory and regenerates a
new one from scratch, treating all previous steps as unreliable and
incurring high computational cost. All tokens generated during the
rerun process are included in the reported token cost.

Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
(2) Step-wise Retrysequentially verifies reasoning steps along
the trajectory and reuses the validated prefix to regenerate sub-
sequent steps. While enabling partial trajectory reuse, it does not
classify error types or localize failures. All tokens consumed during
step verification and regeneration are counted in the token cost.
(3) RAG-Critic[ 8] follows a diagnose–replan–rerun paradigm
by identifying coarse-grained error categories (e.g., hallucination
or retrieval failure) and replanning a sequence of recovery actions.
However, diagnosis operates at the trajectory level without pre-
cise localization, often re-executing previously correct segments,
thereby incurring redundant and unnecessary token consumption.
All tokens consumed during diagnosis, replanning, and execution
of recovery actions are included in the reported token cost.
5.2.3 Implementation Details.Following prior work on agentic
RAG systems [ 19], we conduct experiments using three backbone
language models: Qwen3-8B [ 42], LLaMA-3.1-8B-Instruct [ 37], and
Qwen3-4B [ 42]. All baselines and diagnosis-repair strategies are
evaluated under identical backbone models to ensure fair compar-
ison. Following Karpukhin et al .[23], we use the full Wikipedia
2018 dump as the retrieval corpus. Our main experiments follow
the retrieval setup of Jiang et al .[19], employing FAISS [ 22] for
indexing,BGE-large-en-v1.5[ 40] as retriever, and retrieving top-
5 documents. All methods share the same retrieval settings. All
methods are evaluated in a post-hoc manner on failed trajectories
produced by the original agentic RAG baselines, without modifying
their underlying retrieval or reasoning components. For coverage
assessment of DR-RAG, we instantiate Cov(𝑦) using an LLM-based
sufficiency judge with the same backbone model as the evaluated
methods, without access to golden evidence. Notably, the diagnosis
module of DR-RAG does not use ground-truth answers and oper-
ates solely on the generated trajectories and retrieved documents.
All reported token costs for DR-RAG include tokens consumed
during both diagnosis and repair. Experiments are conducted on
four NVIDIA RTX Pro 6000 GPUs, using vLLM for inference [24].
5.3 Main Results (RQ1)
Table 2 reports the main experimental results across three multi-hop
question answering benchmarks. Across all baselines and datasets,
DR-RAGconsistently achieves substantial performance improve-
ments while maintaining lower token consumption. Compared with
full rerun and step-wise retry baselines, DR-RAG yields significant
gains in EM, F1, and ROUGE-L. For instance, under the Qwen3-
8B backbone with the ReAct baseline, DR-RAG improves EM by
+25.8%on HotpotQA,+19.6%on 2Wiki, and+10.0%on MuSiQue,
while reducing token usage by26.8%relative to the rerun strat-
egy. Across all settings, DR-RAG consistently incurs the lowest
token cost, validating that localized repair is more efficient than
full trajectory re-execution.
DR-RAG achieves simultaneous gains in effectiveness and effi-
ciency by adopting a failure-attribution-driven local repair para-
digm. Conventional rerun or step-wise methods implicitly treat the
entire trajectory as unreliable after a failure, leading to redundant
regeneration of reasoning and retrieval steps. In contrast, DR-RAG
first diagnoses whether the failure originates from formatting, rea-
soning, or retrieval, and then localizes the earliest point of failure,
thereby avoiding recomputation of previously verified reasoningTable 3: Ablation study of DR-RAG. We evaluate error taxon-
omy and error localization across three benchmarks, show-
ing gains in accuracy and efficiency.
BaselineHotpotQA 2Wiki MuSiQue
TokensΔEM TokensΔEM TokensΔEM
DR-RAG 4,483 17.3 5,697 17.7 5,464 10.0
w/o Taxonomy4,840 13.9 5,894 13.7 5,507 6.9
w/o Localization6,012 14.3 7,193 14.9 6,585 8.6
prefixes. This design allows repair to focus on the decisions that
actually cause the error, reducing unnecessary generation and the
likelihood of introducing new errors, and ultimately resulting in
higher overall repair success rates.
To evaluate generalization, we test DR-RAG with different back-
bone models and model scales, including Qwen3-4B, Qwen3-8B,
and LLaMA-3.1-8B-Instruct. As shown in Table 2, DR-RAG con-
sistently improves performance across all model families. Notably,
the gains are often larger for smaller models. Under the Qwen3-4B
setting, DR-RAG achieves up to a+22.2%EM improvement on Hot-
potQA while still using the lowest number of tokens. These results
show that DR-RAG does not rely on a specific backbone or model
scale and can effectively improve both strong and weaker models.
5.4 Ablation Study (RQ2)
We conduct an ablation study to examine the roles of error tax-
onomy and error localization in DR-RAG. As shown in Table 4,
removing either component consistently degrades performance
across all datasets. Thew/o Taxonomyvariant achieves smaller
ΔEM gains than DR-RAG, indicating that error classification is
important for selecting appropriate repair operators. Without tax-
onomy, the repair module cannot distinguish different failure types
and may apply mismatched repairs, which reduces repair effective-
ness. In contrast,w/o Localizationincurs around 1000 more tokens
on average, showing that without precise failure localization, the
model regenerates larger parts of the reasoning trajectory and intro-
duces redundant reasoning. These results confirm that taxonomy
and localization are complementary: taxonomy enables effective
repair, while localization ensures minimal intervention.
These results confirm that taxonomy and localization are com-
plementary: taxonomy enables effective repair, while localization
ensures minimal intervention.
5.5 Token and Time Efficiency (RQ3)
We evaluate the computational efficiency of DR-RAG in terms of
both token consumption and end-to-end inference time. As shown
in Table 4, DR-RAG consistently achieves the lowest cost across
all datasets and agent frameworks under both metrics. Full rerun
strategies regenerate entire reasoning trajectories after a failure,
resulting in substantial overhead in token usage and runtime. Step-
wise retry reduces some redundancy but still repeats downstream
reasoning and retrieval beyond the true failure point, as it does
not distinguish different error types and regenerates all subsequent
steps once a failure is detected. In contrast, DR-RAG performs

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Shuguang Jiao et al.
Table 4: Efficiency comparison of different repair strategies.
Tok.denotes the average token consumption, andTimerep-
resents the total time cost 𝑠(in seconds). The best results
(minimum values) are highlighted in bold.
Baseline MethodHotpotQA 2Wiki MuSiQue
Tok. Time Tok. Time Tok. Time
ReActRerun 8,714 713 10,934 842 9,820 927
Step-wise 11,524 1,031 9,822 915 11,315 990
RAG-Critic 14,196 5,036 15,608 5,003 17,026 7,111
DR-RAG 4,514 279 5,583 369 5,687 199
Search-o1Rerun 7,844 633 8,389 716 8,537 817
Step-wise 6,861 611 8,290 762 8,262 1,034
RAG-Critic 17,910 4,735 20,670 5,595 20,711 8,790
DR-RAG 4,340 274 5,659 384 4,963 220
Search-R1Rerun 6,457 652 7,877 474 7,930 773
Step-wise 5,848 530 7,696 702 6,259 734
RAG-Critic 15,203 5,511 15,997 5,193 17,097 8,852
DR-RAG 4,594 270 5,850 254 5,743 366
HotpotQA 2Wiki Musique020406080Accuracy (%)6671
656264
52 5160 60Cov=1
HotpotQA 2Wiki Musique6063
51 5166
56576972Cov<1React Search-o1 Search-R1 Avg.
Figure 3: Accuracy of the automated diagnosis module eval-
uated against human annotations across different datasets
and baselines.
diagnosis-driven localized repair, reusing verified prefix trajectories
and intervening only at the earliest failure step. As a result, DR-
RAG avoids unnecessary regeneration beyond the true failure cause,
leading to consistent reductions in both token cost and inference
time across all settings.
5.6 Error Diagnosis Reliability (RQ4)
We evaluate the reliability of the automated diagnosis module in DR-
RAG by comparing its predictions with human annotations across
datasets and agent frameworks. Figure 3 reports diagnosis accuracy
stratified by evidence coverage. Across all baselines and datasets,
diagnosis accuracy consistently falls within the 50–70% range. On
average, DR-RAG achieves an accuracy of61.6%under full coverage
(Cov=1) and60.3%under partial coverage (Cov<1), indicating stable
diagnostic performance regardless of evidence completeness. This
consistency suggests that the diagnosis module generalizes well
across agent architectures and datasets and provides sufficiently
reliable signals to guide localized repair.
Reasoning Error
Format Error (cov=1) Format Error (cov<1)Retriever ErrorSearch Error
Predicted LabelReasoning Error
Format Error (cov=1)
Format Error (cov<1)
Retriever Error
Search ErrorTrue Label0.73
0.20
0.00
0.00
0.000.27
0.80
0.00
0.00
0.000.00
0.00
0.39
0.07
0.060.00
0.00
0.47
0.75
0.280.00
0.00
0.14
0.17
0.66
0.00.10.20.30.40.50.60.70.8
Error Diagnosis Confusion Matrix(a) Confusion matrix for auto-
mated error diagnosis.
Reasoning 
Error
Search ErrorRetriever ErrorFormat Error 
(cov<1)10%20%30%40%
Rerun Step DiagnoseRepair Rates across Different Error Types
· Format Error (cov=1)(b) Average repair rates across er-
ror types on HotpotQA.
Figure 4: Automated error diagnosis and repair analysis. (a)
Confusion matrix aggregated across multiple datasets (2Wiki,
HotpotQA, and MuSiQue) and baselines (ReAct, Search-o1,
and Search-R1), showing high consistency between the di-
agnostic model and human annotations. (b) Comparison of
average repair rates across error types on HotpotQA, aggre-
gating results from three baselines under repair strategies.
Table 5: Repair success rates stratified by error type and re-
trieval coverage. Results are broken down by failure category
(format, reasoning, search, and retriever errors) and reported
separately for full coverage (Cov=1) and partial coverage
(Cov<1) across datasets and baselines.
Cov=1 Cov<1
Baseline DatasetFmt.
ErrReas.
ErrSear.
ErrRetr.
ErrFmt.
Err
ReActHotpotQA 43.40 60.71 27.22 39.77 15.00
2Wiki 30.23 52.63 27.27 25.61 25.00
MuSiQue 61.54 100.00 8.96 17.27 13.04
Search-o1HotpotQA 40.00 34.78 19.15 28.75 33.33
2Wiki 21.28 21.05 16.48 21.13 10.00
MuSiQue 43.75 75.00 9.72 14.56 12.00
Search-R1HotpotQA 36.36 21.88 22.67 22.78 14.29
2Wiki 27.27 17.24 29.00 25.33 45.45
MuSiQue 5.00 50.00 8.64 6.10 3.57
We further analyze diagnosis behavior by error type using the
confusion matrix in Figure 4(a). Overall, DR-RAG maintains high
accuracy across major failure categories, including reasoning, for-
mat, and retriever errors, with most confusions occurring between
closely related error types. Notably, under partial coverage (Cov<1),
formatting errors are more likely to be misclassified as retriever
errors. This reflects the ambiguity introduced by incomplete evi-
dence: when supporting documents are missing, the model may
attribute weakly supported answers to retrieval failure, even if the
generated reasoning is semantically plausible.
5.7 Repair Performance by Error Type (RQ5)
We first analyze repair performance by error type using Figure 4(b),
which comparesrerun,step-level repair, andDR-RAGon HotpotQA.

Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Question: Which of these is further south in China, the Pulandian District or Kaiyuan, Liaoning?
Rerun-based Repair Doctor-RAG Repair 
[1] <reason> I need to compare the geographic locations of Pulandian District 
and Kaiyuan, Liaoning to determine which one is further south. I should first 
retrieve information about Pulandian District. </reason>
[2] <search> Pulandian District location in China </search>
[3] <info> Pulandian District is one of the districts of Dalian, located in the south 
of Liaoning province. </info>
[4] <reason> Now I need to retrieve information about Kaiyuan, Liaoning to 
compare its location with Pulandian District. </reason>
[5] <search> Kaiyuan Liaoning location in China  </search>
[6] <info> Kaiyuan is a county-level city in the northeast of Liaoning province. 
</info>
[7] <reason> Pulandian District is in the south of Liaoning, and Kaiyuan is in the 
northeast of Liaoning. Northeastern areas are generally further south than 
southeastern areas. Therefore, Kaiyuan should be further south. </reason>
[8] <answer> Kaiyuan, Liaoning </answer>Rerun Entire Pipeline
I need to compare ...  
<search> 
south of Liaoning ...
I need to ...
Kaiyuan ... 
northeast of Liaoning ... 
Pulandian is more southTrajectory
Pulandian ... <reason> 
<info>
<reason> 
<search> 
<info>
<reason> 
<answer> Pulandian
Reasoning ErrorTrajectory
Locate in [7]Diagnose
Prefix Reuse
<reason> 
<answer> Pulandian is 
more south than 
Kaiyuan
Pulandian
Truncating trajectory
Figure 5: Case study of a reasoning logic error under full evidence coverage. DR-RAG localizes the earliest faulty reasoning
step and repairs the error by reusing retrieved evidence, avoiding unnecessary retrieval and full trajectory regeneration.
(a)DR-RAG achieves the highest repair rates across all five error
types, with the largest gains on easoning errors and format errors
under full coverage (Cov=1). These errors are local to generation or
reasoning and do not require retrieval changes, enabling targeted
correction after failure identification. In contrast, rerun and step-
level repair do not model error causes, limiting their improvements.
(b)For retriever errors and search errors, DR-RAG still outperforms
the baselines, but the overall repair rates are lower. These two error
types exhibit different limiting factors. For retriever errors, repair is
constrained by missing or incomplete evidence, while search errors
typically arise in more complex multi-hop questions that require
harder reasoning and planning beyond local correction.
Table 5 reports repair success rates under different error types
and retrieval coverage conditions. Overall, repair under partial
coverage (Cov<1) is consistently more difficult than under full cov-
erage (Cov=1). When retrieval coverage is incomplete, many errors
arise from complex reasoning and planning failures, which reflect
the upper bound of the model’s ability to analyze and solve hard
tasks. In such cases, even with a correct diagnosis, localized repair
becomes less effective. We also observe clear dataset-level differ-
ences. Across baselines and error types, repair rates on MuSiQue
are generally lower than those on HotpotQA and 2Wiki. This is
expected, as MuSiQue contains more complex multi-hop questions
and longer reasoning chains, which place higher demands on the
model’s reasoning capacity and limit the effectiveness of repair.
5.8 Oracle-based Upper Bound Analysis (RQ6)
Finally, we estimate the upper bound of DR-RAG by assuming
oracle access to error types and failure locations. As shown in
Table 6, oracle-guided DR-RAG achieves higher repair rates and
downstream QA performance across all datasets and baselines. TheTable 6: Oracle-guided upper bound results across datasets.
Token cost, Repair Rate, and Improvement are reported.
Baseline Dataset Tokens. Repair.ΔEMΔF1ΔR-L
ReActHotpotQA 4,520 39.4 27.8 26.9 25.9
2Wiki 5,504 29.3 20.2 21.0 20.6
MuSiQue 5,681 13.1 11.2 11.2 11.3
Search-o1HotpotQA 4,459 29.0 15.4 14.3 13.6
2Wiki 5,888 25.4 16.2 16.6 16.4
MuSiQue 5,422 14.6 12.2 12.2 11.7
Search-R1HotpotQA 4,571 21.8 13.6 14.4 14.3
2Wiki 6,231 29.0 19.4 18.3 18.9
MuSiQue 5,965 11.3 9.8 9.9 9.7
gap between oracle and automated diagnosis quantifies the remain-
ing headroom of the framework. Notably, once the correct failure
point is identified, the proposed repair operators yield strong im-
provements. This suggests that future gains in DR-RAG are more
likely to come from improving diagnosis and localization accuracy
than from introducing more complex repair mechanisms.
5.9 Case Study
We present a representative example from HotpotQA to illustrate
how DR-RAG repairs reasoning errors under full evidence coverage.
The question asks:“Which of these is further south in China, the
Pulandian District or Kaiyuan, Liaoning?”The agent retrieves the
correct documents for both locations, indicating full evidence cov-
erage. However, it incorrectly answers“Kaiyuan, Liaoning”, despite
the retrieved evidence stating that Pulandian District is located in
the south of Liaoning province, while Kaiyuan is in the northeast.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Shuguang Jiao et al.
According to our taxonomy, this failure is classified as a rea-
soning logic error under full coverage, where the error originates
from incorrect reasoning rather than missing evidence. Restart-
based baselines respond to this failure by regenerating the entire re-
trieval–reasoning trajectory, even though the retrieved documents
contain sufficient information to answer the question.
In contrast, DR-RAG diagnoses the earliest reasoning failure
and performs localized repair by reusing the retrieved evidence
and recomputing only the affected reasoning steps. This targeted
intervention corrects the error without additional retrieval and
avoids unnecessary recomputation, demonstrating the effectiveness
of diagnosis-guided localized repair.
6 Conclusion
In this work, we presented DR-RAG, a diagnose-and-repair frame-
work for Agentic Retrieval-Augmented Generation that enables
efficient failure correction without full-pipeline reruns. DR-RAG
models failure handling as trajectory-level diagnosis and localized
repair by identifying error types and earliest failure points within
agentic reasoning trajectories. Experiments show that DR-RAG
consistently improves accuracy and efficiency, supported by reli-
able automated diagnosis and stable repair behavior. Overall, this
work reframes failure handling in Agentic RAG as a structured
repair problem rather than a costly restart, providing a practical
foundation for more robust and efficient agentic reasoning systems
in complex multi-step reasoning scenarios.
References
[1]Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, and Huan Liu. 2024.
Mindful-RAG: A Study of Points of Failure in Retrieval Augmented Generation.
InFLLM. IEEE, 607–611.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InICLR. OpenReview.net.
[3]Tianyu Cao, Neel Bhandari, Akhila Yerukola, Akari Asai, and Maarten Sap. 2025.
Out of Style: RAG’s Fragility to Linguistic Variation.CoRRabs/2504.08231 (2025).
[4]Guoxin Chen, Zile Qiao, Xuanzhong Chen, Donglei Yu, Haotian Xu, Wayne Xin
Zhao, Ruihua Song, Wenbiao Yin, Huifeng Yin, Liwen Zhang, Kuan Li, Minpeng
Liao, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025. IterResearch:
Rethinking Long-Horizon Agents via Markovian State Reconstruction.CoRR
abs/2511.07327 (2025).
[5]Yuqin Dai, Guoqing Wang, Yuan Wang, Kairan Dou, Kaichen Zhou, Zhanwei
Zhang, Shuo Yang, Fei Tang, Jun Yin, Pengyu Zeng, Zhenzhe Ying, Can Yi,
Changhua Meng, Yuchen Zhou, Yongliang Shen, and Shuai Lu. 2025. EviNote-
RAG: Enhancing RAG Models via Answer-Supportive Evidence Notes.CoRR
abs/2509.00877 (2025).
[6]Guanting Dong, Licheng Bao, Zhongyuan Wang, Kangzhi Zhao, Xiaoxi Li, Jiajie
Jin, Jinghan Yang, Hangyu Mao, Fuzheng Zhang, Kun Gai, Guorui Zhou, Yutao
Zhu, Ji-Rong Wen, and Zhicheng Dou. 2025. Agentic Entropy-Balanced Policy
Optimization.CoRRabs/2510.14545 (2025).
[7]Guanting Dong, Yifei Chen, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Yutao Zhu, Hangyu
Mao, Guorui Zhou, Zhicheng Dou, and Ji-Rong Wen. 2025. Tool-Star: Empow-
ering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning.CoRR
abs/2505.16410 (2025).
[8]Guanting Dong, Jiajie Jin, Xiaoxi Li, Yutao Zhu, Zhicheng Dou, and Ji-Rong Wen.
2025. RAG-Critic: Leveraging Automated Critic-Guided Agentic Workflow for
Retrieval Augmented Generation. InACL (1). Association for Computational
Linguistics, 3551–3578.
[9]Guanting Dong, Hangyu Mao, Kai Ma, Licheng Bao, Yifei Chen, Zhongyuan
Wang, Zhongxia Chen, Jiazhen Du, Huiyang Wang, Fuzheng Zhang, Guorui
Zhou, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. 2025. Agentic Reinforced
Policy Optimization.CoRRabs/2507.19849 (2025).
[10] Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua
Zhu, and Yi Wu. 2025. Beyond Ten Turns: Unlocking Long-Horizon Agentic
Search with Large-Scale Asynchronous RL.CoRRabs/2508.07976 (2025).
[11] Gauthier Guinet, Behrooz Omidvar-Tehrani, Anoop Deoras, and Laurent Callot.
2024. Automated Evaluation of Retrieval-Augmented Language Models withTask-Specific Exam Generation. InICML. OpenReview.net.
[12] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. Retrieval Augmented Language Model Pre-Training. InICML (Proceedings
of Machine Learning Research, Vol. 119). PMLR, 3929–3938.
[13] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics, COLING 2020, Barcelona, Spain (Online), December 8-13, 2020. Interna-
tional Committee on Computational Linguistics, 6609–6625.
[14] Chengkai Huang, Xiaodi Chen, Hongtao Huang, Quan Z Sheng, and Lina Yao.
2026. Generative Chain of Behavior for User Trajectory Prediction.arXiv preprint
arXiv:2601.18213(2026).
[15] Chengkai Huang, Junda Wu, Yu Xia, Zixu Yu, Ruhan Wang, Tong Yu, Ruiyi Zhang,
Ryan A Rossi, Branislav Kveton, Dongruo Zhou, et al .2025. Towards agentic
recommender systems in the era of multimodal large language models.arXiv
preprint arXiv:2503.16734(2025).
[16] Chengkai Huang, Yu Xia, Rui Wang, Kaige Xie, Tong Yu, Julian McAuley, and
Lina Yao. 2025. Embedding-informed adaptive retrieval-augmented generation
of large language models. InProceedings of the 31st International Conference on
Computational Linguistics. 1403–1412.
[17] Hongtao Huang, Chengkai Huang, Junda Wu, Tong Yu, Julian McAuley, and
Lina Yao. 2025. Listwise Preference Diffusion Optimization for User Behavior
Trajectories Prediction.arXiv preprint arXiv:2511.00530(2025).
[18] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. 2024.
Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models
through Question Complexity. InNAACL-HLT. Association for Computational
Linguistics, 7036–7050.
[19] Jinhao Jiang, Jiayi Chen, Junyi Li, Ruiyang Ren, Shijie Wang, Xin Zhao, Yang Song,
and Tao Zhang. 2025. RAG-Star: Enhancing Deliberative Reasoning with Retrieval
Augmented Verification and Refinement. InNAACL (Long Papers). Association
for Computational Linguistics, 7064–7074.
[20] Shuguang Jiao, Xinyu Xiao, Yunfan Wei, Shuhan Qi, Chengkai Huang, Quan
Z. Michael Sheng, and Lina Yao. 2026. PruneRAG: Confidence-Guided
Query Decomposition Trees for Efficient Retrieval-Augmented Generation.
arXiv:2601.11024 [cs.IR]
[21] Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-R1: Training LLMs to Reason and Leverage Search Engines
with Reinforcement Learning.CoRRabs/2503.09516 (2025).
[22] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-Scale Similarity
Search with GPUs.IEEE Trans. Big Data7, 3 (2021), 535–547.
[23] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InEMNLP (1). Association for Computational
Linguistics, 6769–6781.
[24] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAttention.
InSOSP. ACM, 611–626.
[25] Kin Kwan Leung, Mouloud Belbahri, Yi Sui, Alex Labach, Xueying Zhang,
Stephen Anthony Rose, and Jesse C. Cresswell. 2026. Classifying and Ad-
dressing the Diversity of Errors in Retrieval-Augmented Generation Systems.
arXiv:2510.13975 [cs.CL]
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InNeurIPS.
[27] Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian
Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic Search-Enhanced Large
Reasoning Models.CoRRabs/2501.05366 (2025).
[28] Xiaoxi Li, Wenxiang Jiao, Jiarui Jin, Guanting Dong, Jiajie Jin, Yinuo Wang, Hao
Wang, Yutao Zhu, Ji-Rong Wen, Yuan Lu, and Zhicheng Dou. 2025. DeepAgent:
A General Reasoning Agent with Scalable Toolsets.CoRRabs/2510.21618 (2025).
[29] Yifan Li, Xuan Wang, Shuhan Qi, Chengkai Huang, Zoe L Jiang, Qing Liao, Jian
Guan, and Jiajia Zhang. 2021. Self-supervised learning-based weight adaptive
hashing for fast cross-modal retrieval.Signal, Image and Video Processing15, 4
(2021), 673–680.
[30] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries.
InText Summarization Branches Out. Association for Computational Linguistics,
Barcelona, Spain, 74–81.
[31] Haowei Lou, Chengkai Huang, Hye-young Paik, Yongquan Hu, Aaron Quigley,
Wen Hu, and Lina Yao. 2025. SpeechAgent: An End-to-End Mobile Infrastructure
for Speech Impairment Assistance.arXiv preprint arXiv:2510.20113(2025).
[32] Qi Luo, Xiaonan Li, Yuxin Wang, Tingshuo Fan, Yuan Li, Xinchi Chen, and Xipeng
Qiu. 2025. MARAG-R1: Beyond Single Retriever via Reinforcement-Learned
Multi-Tool Agentic Retrieval.CoRRabs/2510.27569 (2025).
[33] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang,
Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie
Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang, and Zheng

Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Zhang. 2024. RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-
Augmented Generation. InNeurIPS.
[34] Zhengliang Shi, Yiqun Chen, Haitao Li, Weiwei Sun, Shiyu Ni, Yougang Lyu,
Run-Ze Fan, Bowen Jin, Yixuan Weng, Minjun Zhu, Qiujie Xie, Xinyu Guo, Qu
Yang, Jiayi Wu, Jujia Zhao, Xiaqiang Tang, Xinbei Ma, Cunxiang Wang, Jiaxin
Mao, Qingyao Ai, Jen-Tse Huang, Wenxuan Wang, Yue Zhang, Yiming Yang,
Zhaopeng Tu, and Zhaochun Ren. 2025. Deep Research: A Systematic Survey.
arXiv:2512.02038 [cs.CL]
[35] Joykirat Singh, Raghav Magazine, Yash Pandya, and Akshay Nambi. 2025. Agentic
Reasoning and Tool Integration for LLMs via Reinforcement Learning.CoRR
abs/2505.01441 (2025).
[36] Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin
Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-Searcher: Incentivizing the Search
Capability in LLMs via Reinforcement Learning.CoRRabs/2503.05592 (2025).
[37] Llama Team. 2024. The Llama 3 Herd of Models.CoRRabs/2407.21783 (2024).
[38] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.Trans.
Assoc. Comput. Linguistics10 (2022), 539–554.
[39] Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Kaiyu He, Xinya Du, and
Zhiyu Chen. 2025. HiPRAG: Hierarchical Process Rewards for Efficient Agentic
Retrieval Augmented Generation.CoRRabs/2510.07794 (2025).
[40] Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-
Pack: Packaged Resources To Advance General Chinese Embedding.CoRR
abs/2309.07597 (2023).
[41] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective Retrieval
Augmented Generation.CoRRabs/2401.15884 (2024).
[42] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng
Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong
Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou,Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li,
Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao,
Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang
Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang
Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru
Zhang, Zhipeng Zhou, and Zihan Qiu. 2025. Qwen3 Technical Report.CoRR
abs/2505.09388 (2025).
[43] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium,
October 31 - November 4, 2018. Association for Computational Linguistics, 2369–
2380.
[44] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and
Karthik Narasimhan. 2023. Tree of Thoughts: Deliberate Problem Solving with
Large Language Models. InNeurIPS.
[45] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. InICLR. OpenReview.net.
[46] Juexiang Ye, Xue Li, Xinyu Yang, Chengkai Huang, Lanshun Nie, Lina Yao, and
Dechen Zhan. 2026. MemWeaver: Weaving Hybrid Memories for Traceable
Long-Horizon Agentic Reasoning.arXiv preprint arXiv:2601.18204(2026).
[47] Asaf Yehudai, Lilach Eden, Yotam Perlitz, Roy Bar-Haim, and Michal Shmueli-
Scheuer. 2025. CLEAR: Error Analysis via LLM-as-a-Judge Made Easy.CoRR
abs/2507.18392 (2025).
[48] Chao Zhang, Yuhao Wang, Derong Xu, Haoxin Zhang, Yuanjie Lyu, Yuhao Chen,
Shuochen Liu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, and Enhong Chen. 2025.
TeaRAG: A Token-Efficient Agentic Retrieval-Augmented Generation Framework.
CoRRabs/2511.05385 (2025).