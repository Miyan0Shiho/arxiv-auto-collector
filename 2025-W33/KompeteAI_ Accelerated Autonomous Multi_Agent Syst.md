# KompeteAI: Accelerated Autonomous Multi-Agent System for End-to-End Pipeline Generation for Machine Learning Problems

**Authors**: Stepan Kulibaba, Artem Dzhalilov, Roman Pakhomov, Oleg Svidchenko, Alexander Gasnikov, Aleksei Shpilman

**Published**: 2025-08-13 20:29:56

**PDF URL**: [http://arxiv.org/pdf/2508.10177v1](http://arxiv.org/pdf/2508.10177v1)

## Abstract
Recent Large Language Model (LLM)-based AutoML systems demonstrate impressive
capabilities but face significant limitations such as constrained exploration
strategies and a severe execution bottleneck. Exploration is hindered by
one-shot methods lacking diversity and Monte Carlo Tree Search (MCTS)
approaches that fail to recombine strong partial solutions. The execution
bottleneck arises from lengthy code validation cycles that stifle iterative
refinement. To overcome these challenges, we introduce KompeteAI, a novel
AutoML framework with dynamic solution space exploration. Unlike previous MCTS
methods that treat ideas in isolation, KompeteAI introduces a merging stage
that composes top candidates. We further expand the hypothesis space by
integrating Retrieval-Augmented Generation (RAG), sourcing ideas from Kaggle
notebooks and arXiv papers to incorporate real-world strategies. KompeteAI also
addresses the execution bottleneck via a predictive scoring model and an
accelerated debugging method, assessing solution potential using early stage
metrics to avoid costly full-code execution. This approach accelerates pipeline
evaluation 6.9 times. KompeteAI outperforms leading methods (e.g., RD-agent,
AIDE, and Ml-Master) by an average of 3\% on the primary AutoML benchmark,
MLE-Bench. Additionally, we propose Kompete-bench to address limitations in
MLE-Bench, where KompeteAI also achieves state-of-the-art results

## Full Text


<!-- PDF content starts -->

KompeteAI: Accelerated Autonomous Multi-Agent System for End-to-End
Pipeline Generation for Machine Learning Problems
Stepan Kulibaba*1, Artem Dzhalilov*1, Roman Pakhomov1, Oleg Svidchenko2, Alexander
Gasnikov3, Aleksei Shpilman2
1Research Center of the Artificial Intelligence Institute, Innopolis University, Innopolis, Russia
2Sberbank of Russia, AI4S Center, Russia
3Innopolis University, MIPT & Steklov Institute, Russia
kulibabast@gmail.com, artem.dzhalilov@gmail.com, r2087007@gmail.com, oasvidchenko@sbebrank.ru,
gasnikov@yandex.ru, eanamuravleva@sbebrank.ru
Abstract
Recent Large Language Model (LLM)-based AutoML sys-
tems demonstrate impressive capabilities but face significant
limitations such as constrained exploration strategies and a
severe execution bottleneck. Exploration is hindered by one-
shot methods lacking diversity and Monte Carlo Tree Search
(MCTS) approaches that fail to recombine strong partial so-
lutions. The execution bottleneck arises from lengthy code
validation cycles that stifle iterative refinement. To overcome
these challenges, we introduce KompeteAI, a novel AutoML
framework with dynamic solution space exploration. Un-
like previous MCTS methods that treat ideas in isolation,
KompeteAI introduces a merging stage that composes top
candidates. We further expand the hypothesis space by in-
tegrating Retrieval-Augmented Generation (RAG), sourcing
ideas from Kaggle notebooks and arXiv papers to incorpo-
rate real-world strategies. KompeteAI also addresses the ex-
ecution bottleneck via a predictive scoring model and an
accelerated debugging method, assessing solution potential
using early stage metrics to avoid costly full-code execu-
tion. This approach accelerates pipeline evaluation 6.9 times.
KompeteAI outperforms leading methods (e.g., RD-agent,
AIDE, and Ml-Master) by an average of 3% on the primary
AutoML benchmark, MLE-Bench. Additionally, we propose
Kompete-bench to address limitations in MLE-Bench, where
KompeteAI also achieves state-of-the-art results.
1 Introduction
Recent research has shifted towards the use of Large Lan-
guage Models (LLM) as the core reasoning engine for Au-
toML frameworks, enabling the autonomous generation and
testing of end-to-end pipelines that adapt to specific tasks (Li
et al. 2024; Jiang et al. 2025; Chi et al. 2024; Liu et al. 2025;
Trirat, Jeong, and Hwang 2024; Grosnit et al. 2024; Yang
et al. 2025). However, these approaches have critical limi-
tations. Initial ”one-shot” generation can yield diverse ideas
but lacks iterative refinement, so a single flawed component
- like suboptimal feature engineering — can undermine the
entire pipeline without a way to correct it.
More adaptive frameworks based on Monte Carlo Tree
Search (MCTS) address this by exploring a tree of potential
solutions, but often use constrained exploration and struggle
*These authors contributed equally.to recombine promising ideas from different branches. Even
recent frameworks that combine exploratory search with
LLM-based reasoning (Liu et al. 2025), despite achieving
state-of-the-art results on MLE-Bench (Chan et al. 2024),
are limited by architectural constraints. They may not sys-
tematically preserve or merge valuable insights from dis-
tant high-performing branches. As a result, valuable solution
components can be prematurely discarded.
Even an ideal internal recombination can only reshuf-
fles known ideas. Retrieval-Augmented Generation (RAG)
breaks this limit by injecting external, domain-specific
knowledge — enabling the agent to explore solutions be-
yond its pretrained hypothesis space. This potential remains
largely untapped in current implementations. Most systems
apply RAG only during early stages; as the problem shifts
— from statistical feature engineering to physics-based sim-
ulation or chemistry-driven descriptors — the agent cannot
retrieve fresh, relevant knowledge, leading to knowledge de-
cay as the pipeline evolves.
Beyond search and knowledge limitations, all frameworks
suffer from a severe execution bottleneck. Validating a sin-
gle solution requires full code execution, which can take
hours for complex models. This issue becomes even more
problematic during debugging, as errors late in the pipeline
force a complete retraining for every fix. As a result, the
slow feedback loop discourages major changes and presents
a serious scalability challenge.
In this work, we introduce KompeteAI, an autonomous
multi-agent framework for structured, multistage pipeline
generation. The key innovations compared to prior ap-
proaches is presented in Table 1. To efficiently explore and
exploit the search space, we employ two core operators.
Adding , which dynamically generates novel stage-specific
ideas by querying external knowledge sources via an adap-
tive RAG module, and merging , which intelligently com-
bines the most successful solutions. We address execution
bottlenecks through a predictive scoring model that prunes
weak solutions early, alongside an accelerated debugging
paradigm using simplified code and smaller data samples,
dramatically shortening the feedback loop.
Our experiments show that KompeteAI sets a new state-
of-the-art on MLE-Bench, outperforming prior methods by
an average of 3%, while accelerating the average test-timearXiv:2508.10177v1  [cs.AI]  13 Aug 2025

performance by a factor of 6.9.
Finally, the only publicly available competitive bench-
mark MLE-Bench suffers from two key limitations. It is
excessively large, constructs its test sets by partitioning the
original training data, and then compares the resulting scores
to the private leaderboard positions that cause evaluation
bias. To address these issues, we introduce a new benchmark
— Kompete-bench.
Our primary contributions:
•Stage-Decomposed Multi-Agent Architecture: A
state-of-the-art framework that partitions the ML work-
flow into discrete stages, enabling agents to specialize in
focused tasks, dynamically integrate external knowledge
sources to enhance exploration diversity, and systemati-
cally recombine optimal partial solutions through novel
addition and merging operators.
•An Accelerated Evaluation and Debugging
Paradigm: A two-part solution to the execution
bottleneck, combining a predictive scoring model and
a rapid debugging framework to drastically reduce
validation time.
•A New Benchmark: Kompete-bench — A curated
benchmark of recent real-world problems designed to
more rigorously evaluate a model’s genuine problem-
solving ability, minimizing the influence of memoriza-
tion or prior exposure.
2 Related Work
Classic AutoML Classic AutoML frameworks - TPOT,
AutoGluon, AutoKeras, LightAutoML, and others (Olson
and Moore 2016; Erickson et al. 2020; Jin, Song, and Hu
2019; Vakhrushev et al. 2021; Feurer et al. 2022; LeDell
and Poirier 2020; Thornton et al. 2013) — automate data
preprocessing, model selection, and hyperparameter tuning
via heuristic search, ensembling, and Bayesian optimization.
Despite their effectiveness, these systems operate within
static search spaces, require manual data preparation and
adaptation for each new task, and lack the dynamic co-
ordination and continual learning capabilities inherent to
multi-agent AutoML architectures.
AutoML based LLM LLM-based AutoML systems have
progressed rapidly, offering increasingly autonomous capa-
bilities through dynamic coordination and iterative planning
(Li et al. 2024; Chi et al. 2024; Jiang et al. 2025; Trirat,
Jeong, and Hwang 2024; Yang et al. 2025; Liu et al. 2025).
To analyze and compare these systems meaningfully, we fo-
cus on four key aspects that consistently shape their per-
formance and flexibility: exploration strategy, RAG, merge
methods, and debugging techniques. These components ad-
dress critical challenges related to how systems explore the
design space, incorporate external knowledge, manage idea
diversity, and implement solutions both efficiently and reli-
ably. Notably, RD-Agent’s merge approach amounts to sim-
ple, uncontrolled recombination driven by LLM rather than
a structured merge algorithm, which results in incoherent or
suboptimal pipeline integrations. A summary of how exist-ing systems approach these dimensions is provided in Ta-
ble 1.
Scoring Model Our approach to model scoring is in-
spired by similar performance prediction methods developed
in Neural Architecture Search (Elsken, Metzen, and Hut-
ter 2019). In NAS, a common approach is to use weight-
sharing supernets, where multiple architectures are jointly
trained by sharing parameters with a large model, and per-
formance is estimated by evaluating sampled architectures
on a validation set (Jawahar et al. 2023a). This method, how-
ever, faces challenges such as weight co-adaptation (Bender
et al. 2018), capacity bottlenecks (Jawahar et al. 2023b), and
gradient conflicts (Gong and Wang 2022). More recently
(Jawahar et al. 2023a), demonstrated that LLMs can serve
as effective performance predictors in NAS tasks, provid-
ing a promising alternative to traditional methods. In con-
trast to NAS, the AutoML setting typically involves a much
broader and less constrained search space, which motivates
the exploration of LLM-based performance prediction be-
yond neural architectures.
Benchmarks Using Kaggle competitions to evaluate au-
tonomous ML systems has become a modern approach in a
number of recent benchmarks due to clear metrics, variety
of tasks and the ability to compare with human solutions.
One of the first benchmarks to systematically evaluate au-
tonomous ML agents on Kaggle competitions was MLA-
gentBench (Huang et al. 2023), which focused on a small
set of tasks with simple baselines, measuring agents’ ability
to improve upon them. Later, DSBench (Jing et al. 2024),
expanded the scope but often relied on automated filtering,
which excluded many complex or non-standard competi-
tions. The most recent effort, MLE-Bench (Chan et al. 2024)
stands out for its scale and diversity, present a more chal-
lenging and realistic testbed for multi-agent AutoML sys-
tems. However, MLE-Bench also faces notable limitations,
including its large size (3.3 TB) and the fact that it constructs
its test sets by partitioning the original training data and then
compares the resulting scores to the private leaderboard po-
sitions that cause evaluation bias.
3 KompeteAI
The pipeline of KompeteAI is demonstrated in Figure 1. It is
designed to ensure robust, leak-free data handling, efficient
exploration of modeling ideas, and rapid iteration, all while
maintaining high code quality and logical consistency. This
section outlines the key stages and mechanisms that under-
pin the operation of our system. It consists of four main com-
ponents: Pipeline Setup, The Ideation Process, Tree-Guided
Exploration, which includes operations such as node addi-
tion and merging, and the Scoring Model, designed to accel-
erate overall pipeline evaluation.
In our framework, each node represents a segment of
the final code, and each tree level corresponds to a distinct
pipeline component: Exploratory Data Analysis (EDA),
Feature Engineering (FE), or Model Training (MT). Con-
nections between nodes indicate that they form parts of
the same pipeline. The main stage unfolds in three phases:

Table 1: Comparison of LLM-based AutoML systems. Phase-wise chain : sequential, stage-by-stage pipeline construction;
without context : searches that ignore information from parallel or prior branches; Multi-stage expansion : KompeteAI’s staged
node growth at every pipeline level; R&D-phase : retrieval only during initial planning, versus Dynamic RAG : on-the-fly lit-
erature/web queries; Incremental debugging : iterative draft–debug–refine cycles, versus Multistage debugging : KompeteAI’s
multi-agent, stage-based error checking; Simple recombination : implicit LLM-driven fusion of ideas, versus Controlled merger :
KompeteAI’s explicit algorithmic branch combination.
Method Exploration Strategy RAG system Debugging method Merge Method
AutoKaggle Phase-wise chain ✗ Three-stage phase & Unit tests ✗
SELA MCTS (UCT) without context ✗ ✗ ✗
AIDE MCTS (UCT) without context ✗ Incremental ✗
AutoML-Agent Retrieval-augmented planning R&D-phase Incremental ✗
R&D-Agent Multi-trace exploration R&D-phase Incremental Simple recombination
ML-Master Controlled MCTS with memory ✗ Incremental ✗
KompeteAI Multi-stage expansion Dynamic Multistage Controlled merger
tree initialization, adding, and merging. Tree initialization
constructs the primary tree structure. Subsequently, within
the allotted time budget, the system alternates between the
adding phase — where new ideas are injected into the most
promising branches — and the merging phase, which com-
bines two promising ideas into a single, more powerful
solution. The EDA phase is represented by a single root
node, which aggregates all insights derived from data ex-
ploration, such as visualizations, distribution analyses, and
summary statistics. This EDA node is not static: as the
search progresses, it can be dynamically enriched with ad-
ditional analyses prompted by downstream discoveries. In
contrast, nodes at the FE and MT levels represent concrete
instantiations of their respective phases, with edges between
nodes indicating their inclusion in the same candidate solu-
tion. While a node may have multiple children, it is restricted
to a single parent.
3.1 Pipeline Setup
This phase sets up the core components required for the
next stages. The dataset is ingested by the Reader Agent,
which analyzes its structure, produces a detailed task spec-
ification, and initializes the data for the RAG based on this
description. The Metric Agent constructs unit tests to sup-
port submission validation and defines the evaluation metric
function. The Validator Agent partitions the data according
to the task specification and applies appropriate preprocess-
ing methods to ensure a valid and reliable evaluation proto-
col. The Baseliner Agent generates an initial solution, estab-
lishing a lower-bound reference for expected performance.
Based on the baseline score, it also assesses the quality of the
data split and, if necessary, can trigger the Validator Agent
to re-partition the data using an alternative strategy.
3.2 The Ideation Process
This section focuses on the generation process for a single
node within the pipeline, emphasizing localized decision-
making rather than full-path construction. The procedure is
structured as a modular interaction among four specialized
agents. The Insighter Agent initiates the process by propos-
ing candidate ideas. These are then evaluated by the CheckerAgent for consistency and compatibility with the solution
context. Once validated, the Coder Agent translates the idea
into executable code. The output is again verified by the
Checker Agent, this time to ensure implementation correct-
ness. Finally, the Debugger Agent addresses potential run-
time errors and integration issues, completing the node’s ini-
tialization cycle.
Insighter The Insighter Agent generates ideas for down-
stream components, guided by factors such as EDA results
and the current solution stage. To overcome LLM limitations
in diversity and quality, it uses two key mechanisms.
• Tree Memory: The agent employs a memory module
over the entire ideation tree, analyzing previously gen-
erated nodes using diversity-driven strategies based on
embedding similarity — such as selecting nodes closest
or farthest from the parent, or sampling randomly. A top-
nsubset is selected for inclusion in the context, and the
memory is continuously updated as new nodes are added,
enabling dynamic promotion of diversity.
• Retrieval-Augmented Generation: The RAG mechanism
retrieves strong ideas from Kaggle (top- nsimilar tasks
and winning solutions) and arXiv (top- mpapers via task-
specific queries), forming a candidate pool. It then nor-
malizes texts, extracts key ideas, and selects kfor con-
text. The retrieval approach is adaptive, based on function
calling that is triggered only when external knowledge is
expected to contribute meaningfully, thereby minimizing
computational overhead.
Checker The Checker Agent is designed to validate the
logical consistency of outputs produced by other agents. It
uses unit tests, including schema validation and script exe-
cution, to assess correctness.
Coder The Coder Agent implements the input idea based
on several parameters, including a description of the input
data, available computational resources, and the idea itself.
It selects optimal hyperparameters and designs the architec-
ture to ensure the code executes within the given time con-
straints while remaining as efficient as possible.

DataRetrievalRetrievalRetrievalPipeline
TreeMainBack-propagationBack-propagationsetup
initializationloop
ExplorationknowledgeConclusionsBrief 
summaryoverview.txttrain.csvtest.csv...Metricagent
Reader
ValidatorAddingMergingBaseliner
traint est
EDA
BackpropagationGenerating-ideasNPassagenumber :N+1
Curren t Sta t e A na ly sisM emor y1 + 2→73 + 2→84 + 5→9XGBPCA
MTMTMTFEFE
MTFEMTMTFEFE
Scorefail
Adding-ideasMerge-ideasFE0.69MT0.69MT0.86New0.58FE0.720.58+0.862=Figure 1: The KompeteAI AutoML pipeline consists of three main stages: Pipeline Setup, Tree Initialization, and the Main
Loop. Pipeline Setup involves data ingestion, initial analysis, metric evaluation, data validation, and establishing performance
benchmarks. Tree Initialization uses EDA to generate initial insights, forming a tree structure where Feature Engineering nodes
are parents to Model Training nodes. The Main Loop iteratively refines the solution tree through adding new nodes, with their
performance subsequently evaluated via code acceleration, debugging, and return speed to get a score; backpropagation for
scoring FE ideas based on downstream model feedback; and merging strong partial solutions to recombine effective strategies
Debugger The Debugger agent efficiently resolves issues
with dependency installation, code generation, and submis-
sion formatting using a nested loop that iteratively debugs
code within a set limit. To reduce runtime overhead common
in existing systems, it accelerates debugging by minimizing
time-sensitive parameters like training iterations, enabling
fast error detection without full execution. Once debugging
succeeds, the original configuration is restored. The agent
also logs detailed metrics — such as error types, retries, and
outcomes — and skips debugging steps for recurring errors,
opting instead for direct code regeneration.
3.3 Tree-Guided Exploration
This section provides an explanation of the core principles
underlying operations on the ideation tree.
Tree Initialization The primary objective of the tree ini-
tialization stage is to generate an initial set of candidate
pipelines, providing a diverse foundation from which fur-
ther exploration can proceed. This stage seeds the search
space with promising ideas and establishes the initial struc-
ture for subsequent optimization. After the initial tree is con-
structed, we perform a backpropagation step to propagate
performance signals from the model training nodes (where
direct evaluation scores are available) up through the tree,
updating each node’s average score based on new informa-
tion to improve their representativeness within the search
tree.
Adding The adding stage governs the structured expan-
sion of the ideation tree Tt= (Vt, Et)by proposing new
nodes at the Feature Engineering and Model Training levels.
Before expansion, the agent examines the current tree stateand may selectively trigger additional exploratory analysis
via function calling, choosing specific data inspections that
are most informative for subsequent node generation.
The entire process is conditioned on a global context rep-
resentation ct, which captures the current pipeline state and
external knowledge in a structured and denoised form. We
define the context vector as:
ct=ϕEDA(Tt)+ϕreader (Tt)+ϕext(QueryExternal ())
where each mapping ϕ:X → C transforms raw, unstruc-
tured, or noisy inputs into structured semantic representa-
tions in a common context space C. Specifically:
•ϕEDA encodes statistical and structural information ex-
tracted from exploratory data analysis;
•ϕreader derives contextual insights from competition
metadata and prior solutions;
•ϕextincorporates knowledge retrieved from external
sources (e.g., Kaggle, arXiv).
Based on the aggregated context ct, the agent performs a
structured expansion of the ideation tree in three successive
steps. First, a set of candidate FE nodes is sampled from the
conditional distribution
qFE:C → P (VFE), q FE(· |ct)
where VFEdenotes the space of available FE transforma-
tions. Then, for each sampled FE node vFE∈ V FE, the
agent generates a corresponding set of MT nodes by sam-
pling from
qMT:VFE→ P(VMT)
which specifies a distribution over the MT configuration
spaceVMT. Finally, a subset of FE nodes from the current

tree (including both existing and newly added ones) is se-
lected based on their scores, reflecting their estimated util-
ity. For each selected FE node, additional MT nodes are ap-
pended as children.
The full procedure is detailed in Appendix A1.
Merging The merging stage enables the agent to consoli-
date multiple promising solutions at the Feature Engineering
and Model Training levels, yielding stronger and more gen-
eralizable configurations.
The merging process unfolds as follows.
1. A set of FE node pairs (vFE
i, vFE
j)is sampled, exclud-
ing those present in the long-term memory buffer Mlong.
Each valid pair is merged into a new node
vFE
ij=MergeFE (vFE
i, vFE
j)
which recombines structural and statistical traits of its
parents.
2. For each merged node vFE
ij, a set of child MT nodes is
generated from a conditional distribution. Then, for each
parent FE node vFE
iandvFE
j, the agent selects additional
MT nodes from their respective subtrees. These are sam-
pled stochastically with probabilities proportional to their
scores:
u(i)
k∼SampleTop (vFE
i)
The final child set of vFE
ijcombines the freshly gener-
ated MT nodes and the resampled top performers from
its parents.
3. Additionally, a subset Vmerge
FE ⊆VFE
tis selected, and
within each selected FE node, MT child pairs are merged
using
uMT
ij=MergeMT (ui, uj)
to form stronger model configurations.
To avoid redundant or destructive merges, the agent
employs dual memory buffers. The short-term memory
Mshort temporarily stores failed merge attempts, while
the long-term memory Mlongpermanently excludes re-
peatedly failing node pairs. Specifically, a pair that fails
to produce a beneficial merge θfailtimes is promoted from
Mshort toMlong, ensuring efficient resource allocation
and adaptive learning.
Merging procedure detailed in Appendix A2.
3.4 Scoring Model
The scoring model is a crucial component applied at the
model training nodes of our pipeline generation tree. Its pri-
mary purpose is to accelerate the overall pipeline evaluation
process, thereby enabling the exploration of a greater num-
ber and diversity of ideas within the solution space. By pro-
viding rapid performance estimates for candidate models,
the scoring model allows us to prioritize the most promising
approaches without the need for full-scale training, which
can be prohibitively time-consuming for complex models.
The core principle behind the scoring model is to pre-
dict the final performance of a candidate model based on
how similar models have performed on the same dataset. To
achieve this, we generate detailed descriptions for severalanchor ideas as well as for the candidate idea to be scored.
This step is essential, as even minor code differences —
such as the number of training epochs — can lead to signifi-
cant variations in model performance. These descriptions are
then inserted into a prompt to LLM with dataset description
and current model training specification.
To construct effective anchor ideas for the scoring model,
we first train several model architectures on a single feature
engineering idea to observe how different architectures be-
have on the dataset. Subsequently, we select one model ar-
chitecture and train it across all feature engineering nodes,
allowing us to assess the impact of each feature engineering
strategy on the final model score. This systematic approach
to few-shot example selection ensures that the scoring model
receives diverse and informative context, further enhancing
its predictive accuracy.
4 Kompete-bench
4.1 Motivation
Despite growing interest in multi-agent AutoML, empirical
evaluation remains limited by the lack of accessible and reli-
able benchmarks. Currently, the only public option is MLE-
Bench — a collection of 75 Kaggle competitions that offer a
rich mix of real-world ML challenges, clear evaluation met-
rics, and a competitive setup that closely mirrors practical
deployments.
While MLE-Bench is a significant step forward, it has
two key limitations that hinder its practical utility and re-
producibility. First, it is prohibitively large: even the ”Lite”
version with just 22 competitions takes up 158GB and re-
quires substantial compute to process. Second, MLE-Bench
constructs test sets by partitioning the original training data
and compares results to private leaderboard positions. This
introduces significant evaluation bias, as it does not reflect
actual test set performance. We show that scores on MLE-
Bench’s constructed test sets can diverge significantly from
real leaderboard rankings, leading to misleading conclusions
and undermining its reliability as a proxy for real-world out-
comes.
4.2 Methodology
We propose a concise two-part benchmark for evaluating
multi-agent AutoML systems, balancing historical, contem-
porary, and future-oriented challenges under standardized
computational constraints.
•Selection of Established Competitions. We curate 15
competitions from the ’lite’ MLEBench collection that
still accept late submissions on Kaggle and whose indi-
vidual dataset sizes do not exceed 1 GB. The aggregate
volume of this subset is 5.3 GB, providing a stable foun-
dation for baseline comparisons.
•Incorporation of New Competitions. To better reflect
the evolving landscape of AutoML tasks, we include 11
additional competitions from 2024 and 2025 years. These
datasets, totaling 4.9 GB, were selected primarily to en-
sure fair comparison with human performance. This is
because, given the recency of these competitions, both

human participants and current models have access to al-
most the same tools and libraries, minimizing discrepan-
cies due to technological advancements.
5 Experiment
5.1 Experiment Setup
Baselines To provide a comprehensive evaluation on
MLE-Bench, we compared our system against the top-
ranked methods on the leaderboard: AIDE, RD-agent, and
Ml-Master. Due to the high cost of running all baselines,
we report their leaderboard metrics as provided directly by
MLE-Bench. For evaluation on Kompete-bench we re-ran
only AIDE and RD-agent, as both are open-source, while
the implementation of Ml-Master was not publicly avail-
able at the time of our study and was therefore excluded.
All methods were tested using the same LLM backend —
gemini-2.5-flash (Comanici et al. 2025), which also pow-
ers our system. Additionally, to fairly assess the alignment
between medal distributions on MLE-Bench and real Kag-
gle leaderboards, we also evaluated all methods using o1-
preview (Jaech et al. 2024), the model on which most MLE-
Bench submissions achieved their highest scores.
Environment To evaluate our system on MLE-Bench, we
set a 6-hour runtime limit using 12 vCPUs, 64 GB of RAM,
and a single NVIDIA A100 GPU (40 GB). The same hard-
ware setup was used for all systems on Kompete-bench,
where we applied a stricter 6-hour time limit. Each config-
uration was executed three times, and results were averaged
to reduce variance. To ensure fairness and prevent systems
from exploiting test-time information, we additionally RAG
components from accessing any information about the com-
petition date on which the agent was evaluated.
Benchmarks We evaluate KompeteAI on the Lite sub-
set of MLE-Bench, a computationally feasible proxy for
the full benchmark. Although not identical, MLE-Bench re-
ports high alignment in system rankings between Lite and
full versions, making Lite a practical basis for compari-
son. For medal-based evaluation against real Kaggle leader-
boards, we include only Lite competitions that still accept
submissions. Additionally, we assess performance on our
custom benchmark — Kompete-bench, structured following
the same principles described in our methodology section,
by separating the data into two categories: MLE-subset Lite
and recent and a collection of newly curated competitions.
Evaluation Metrics On MLE-Bench, we adopt the offi-
cial leaderboard metric — the percentage of submissions
that receive a medal — which aligns with standard Kaggle
competition criteria. For Kompete-bench, we additionally
report the percent humans beaten metric, which measures
the percentage of human participants outperformed by the
agent on the corresponding Kaggle leaderboard. This metric
offers a more fine-grained and interpretable evaluation sig-
nal compared to medal thresholds, allowing us to distinguish
between systems that may not earn medals but differ mean-
ingfully in their relative competitiveness against human par-
ticipants. To assess pipeline efficiency, we report the number
of complete pipeline passes, where a single pass correspondsto one adding and one merging iteration performed by the
AutoML system.
5.2 Results
Evaluation on MLE-Bench We begin by evaluating our
system on the widely adopted MLE-Bench benchmark (Lite
subset). As presented in Table 2, KompeteAI attains the
highest overall performance, outperforming existing state-
of-the-art systems by an average margin of 3 percentage
points.
Name MLE-Bench medals %
AIDE (o1-preview) 34.3 ± 2.4
RD-agent (o1-preview) 48.18 ± 2.49
ML-Master (deepseek-r1) 48.5 ± 1.5
KompeteAI (gemini-2.5-flash) 51.5 ± 1.5
Table 2: Comparison of agents on MLE-Bench ’Lite’ subset.
For AIDE and RD-agent, we use results reported by MLE-
Bench. KompeteAI was run 3 times with different seeds; re-
sults are reported as mean ± SEM. Each run was given 6
hours.
Kaggle-Based Validation of Robustness To validate ro-
bustness beyond proxy benchmarks, we tested all agents on
the Lite subset using official Kaggle submissions (Table 3).
The analysis reveals a substantial gap between MLE-Bench
evaluations and actual leaderboard outcomes. In particular,
both AIDE and RD-agent exhibit inflated medal rates on
MLE-Bench compared to their real Kaggle performance.
This discrepancy highlights the limitations of relying solely
on MLE-Bench for comparison, especially in competitive
human-level settings.
Name MLE-Bench LB Real LB Contemporary part
AIDE (o1-preview) 20 7 0
RD-agent (o1-preview) 27 13 0
AIDE (gemini-2.5-flash) 13 7 0
RD-agent (gemini-2.5-flash) 13 7 0
Table 3: Medal rate gap between MLE-Bench leaderboard
and leaderboard of real Kaggle competition. For this experi-
ment we ran each system once and used the rules from Kag-
gle to determine the medal.
Evaluation on Kompete-Bench To address these limi-
tations, Kompete-Bench evaluates performance using real
Kaggle leaderboards, covering recent competitions with
high prize pools and strong engagement. On this Contem-
porary subset, traditional medal-based evaluation fails, as
all agents achieve zero medal rates. We therefore propose
a finer-grained metric: percent humans beaten . As shown
in Figure 2, KompeteAI demonstrates state-of-the-art re-
sults, surpassing human performance in 11.2% of cases —
significantly ahead of both RD-agent and AIDE. Neverthe-
less, the agent remains far behind top leaderboard teams,

P er c ent humans beat enMLE-bench (Lit e) par t010203040
Kompet eAIw\o RA Gw\o Mer gingw\o Sc oring m.RD-a gentC ont empor ary par t
AIDE46.440.639.636.032.532.511.28.17.97.47.63.838.5Figure 2: Comparison of our pipeline with AIDE and RD-
agent on Contemporary and MLE-Bench parts of Kompete-
bench. All systems use gemini-2.5-flash as the underlying
LLM. Each was run 3 times with different seeds, results are
averaged, and each run was limited to 6 hours.
reflecting how contemporary competitions demand capabil-
ities beyond local modeling — including large-scale fea-
ture engineering, synthetic data generation, and use of exter-
nal datasets. Human participants leverage broader tools and
resources, underscoring the competitiveness of these chal-
lenges and the remaining headroom for automated agents.
Impact of the Acceleration Methods Finally, we assess
the contribution of our acceleration framework. Table 4 re-
ports the number of iterations within a fixed budget. By in-
tegrating predictive scoring and an accelerated debugging
loop, our pipeline executes over 6.9× more iterations than
a baseline without accelerations, enabling stronger submis-
sions under identical constraints. Moreover, as shown in Fig-
ure 2, KompeteAI without accelerations exhibits a marked
performance drop, underscoring its ability to refine solu-
tions during test-time optimization. While we restrict accel-
eration measurements to KompeteAI due to tight pipeline
integration, the proposed paradigms are general and may
be adapted to other architectures with minimal conceptual
changes.
System Configuration Number of Iterations
w\o all accelerations 1.8 ± 0.3
w\o scoring model 4.1 ± 0.4
with all accelerations 12.5 ± 2.1
Table 4: Impact of acceleration techniques on the number
of completed iterations within a fixed time budget. Mean ±
95% CI estimated via Student’s t-distribution.
5.3 Ablation Study
As shown in Figure 2, all major components of KompeteAI
are crucial, with their impact particularly pronounced on the
Contemporary subset.
•W\o RAG The removal of RAG causes a sharper rela-
tivedecline on Contemporary tasks: performance dropsfrom 11.2% to 8.1%, compared to a more moderate re-
duction from 46.4% to 40.6% on MLE-Bench Lite. This
indicates that in recent competitions, producing strong
solutions in isolation is far harder, and the ability to incor-
porate external ideas and tools becomes disproportion-
ately important.
•W\o Merging The merging mechanism yields one of
the largest absolute improvements. Without it, perfor-
mance falls to 7.9% on Contemporary and 39.6% on
MLE-Bench Lite. While the agent can still generate di-
verse ideas, strong final submissions typically emerge
only when partial yet promising solutions are consoli-
dated.
•W\o Scoring model Removing the scoring model low-
ers results to 7.4% on Contemporary and 36.0% on MLE-
Bench Lite. By prioritizing high-potential candidates, the
scoring model enables broader exploration, allowing the
agent to test significantly more hypotheses under limited
compute.
6 Conclusions
We introduced KompeteAI, a new multi-agent AutoML
framework, and demonstrated its strong performance across
diverse and challenging tasks. Our work opens several
promising directions for future research.
First, while our acceleration paradigm significantly re-
duces computation time, it relies on a scoring model whose
accuracy may degrade over longer runs, potentially lead-
ing to cumulative errors. Improving this component — e.g.,
through adaptive retraining or uncertainty-aware corrections
— could enhance long-term robustness.
Second, we see strong potential in deepening the inte-
gration of LLM-based reasoning with algorithmic search.
By embedding structured computation within a language-
guided planning process, agents can explore solution spaces
more effectively. Extending this to coordinated multi-agent
fine-tuning — with shared representations and task-aware
interaction — could further strengthen system-level coher-
ence and adaptability.
Finally, systems like KompeteAI may prove valuable be-
yond competitive AutoML settings. While platforms like
Kaggle offer practical, open-ended challenges, they only
partially reflect the demands of real-world scientific discov-
ery. As autonomous agents begin contributing to hypothesis
generation, experimental design, and even paper drafting, re-
fining these tools for real research workflows becomes an
exciting and urgent frontier.
Acknowledgements
The study research has been supported by the Min-
istry of Economic Development of the Russian Federa-
tion (agreement No. 139-10-2025-034 dd. 19.06.2025, IGK
000000C313925P4D0002).
References
Bender, G.; Kindermans, P.-J.; Zoph, B.; Vasudevan, V .; and
Le, Q. 2018. Understanding and simplifying one-shot ar-

chitecture search. In International conference on machine
learning , 550–559. PMLR.
Chan, J. S.; Chowdhury, N.; Jaffe, O.; Aung, J.; Sherburn,
D.; Mays, E.; Starace, G.; Liu, K.; Maksin, L.; Patward-
han, T.; et al. 2024. Mle-bench: Evaluating machine learn-
ing agents on machine learning engineering. arXiv preprint
arXiv:2410.07095 .
Chi, Y .; Lin, Y .; Hong, S.; Pan, D.; Fei, Y .; Mei, G.; Liu, B.;
Pang, T.; Kwok, J.; Zhang, C.; et al. 2024. Sela: Tree-search
enhanced llm agents for automated machine learning. arXiv
preprint arXiv:2410.17238 .
Comanici, G.; Bieber, E.; Schaekermann, M.; Pasupat, I.;
Sachdeva, N.; Dhillon, I.; Blistein, M.; Ram, O.; Zhang,
D.; Rosen, E.; et al. 2025. Gemini 2.5: Pushing the fron-
tier with advanced reasoning, multimodality, long context,
and next generation agentic capabilities. arXiv preprint
arXiv:2507.06261 .
Elsken, T.; Metzen, J. H.; and Hutter, F. 2019. Neural ar-
chitecture search: A survey. Journal of Machine Learning
Research , 20(55): 1–21.
Erickson, N.; Mueller, J.; Shirkov, A.; Zhang, H.; Larroy,
P.; Li, M.; and Smola, A. 2020. Autogluon-tabular: Ro-
bust and accurate automl for structured data. arXiv preprint
arXiv:2003.06505 .
Feurer, M.; Eggensperger, K.; Falkner, S.; Lindauer, M.;
and Hutter, F. 2022. Auto-sklearn 2.0: Hands-free automl
via meta-learning. Journal of Machine Learning Research ,
23(261): 1–61.
Gong, C.; and Wang, D. 2022. Nasvit: Neural architec-
ture search for efficient vision transformers with gradient
conflict-aware supernet training. ICLR Proceedings 2022 .
Grosnit, A.; Maraval, A.; Doran, J.; Paolo, G.; Thomas, A.;
Beevi, R. S. H. N.; Gonzalez, J.; Khandelwal, K.; Iacobacci,
I.; Benechehab, A.; et al. 2024. Large language models or-
chestrating structured reasoning achieve kaggle grandmaster
level. arXiv preprint arXiv:2411.03562 .
Huang, Q.; V ora, J.; Liang, P.; and Leskovec, J. 2023. Mla-
gentbench: Evaluating language agents on machine learning
experimentation. arXiv preprint arXiv:2310.03302 .
Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky,
A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney,
A.; et al. 2024. Openai o1 system card. arXiv preprint
arXiv:2412.16720 .
Jawahar, G.; Abdul-Mageed, M.; Lakshmanan, L. V .;
and Ding, D. 2023a. LLM Performance Predictors are
good initializers for Architecture Search. arXiv preprint
arXiv:2310.16712 .
Jawahar, G.; Yang, H.; Xiong, Y .; Liu, Z.; Wang, D.; Sun,
F.; Li, M.; Pappu, A.; Oguz, B.; Abdul-Mageed, M.; et al.
2023b. Mixture-of-supernets: Improving weight-sharing su-
pernet training with architecture-routed mixture-of-experts.
arXiv preprint arXiv:2306.04845 .
Jiang, Z.; Schmidt, D.; Srikanth, D.; Xu, D.; Kaplan, I.; Ja-
cenko, D.; and Wu, Y . 2025. Aide: Ai-driven exploration in
the space of code. arXiv preprint arXiv:2502.13138 .Jin, H.; Song, Q.; and Hu, X. 2019. Auto-keras: An effi-
cient neural architecture search system. In Proceedings of
the 25th ACM SIGKDD international conference on knowl-
edge discovery & data mining , 1946–1956.
Jing, L.; Huang, Z.; Wang, X.; Yao, W.; Yu, W.; Ma, K.;
Zhang, H.; Du, X.; and Yu, D. 2024. DSBench: How Far Are
Data Science Agents from Becoming Data Science Experts?
arXiv preprint arXiv:2409.07703 .
LeDell, E.; and Poirier, S. 2020. H2O AutoML: Scalable
Automatic Machine Learning. In Proceedings of the 7th
ICML Workshop on Automated Machine Learning .
Li, Z.; Zang, Q.; Ma, D.; Guo, J.; Zheng, T.; Liu, M.; Niu,
X.; Wang, Y .; Yang, J.; Liu, J.; et al. 2024. Autokaggle: A
multi-agent framework for autonomous data science compe-
titions. arXiv preprint arXiv:2410.20424 .
Liu, Z.; Cai, Y .; Zhu, X.; Zheng, Y .; Chen, R.; Wen, Y .;
Wang, Y .; Chen, S.; et al. 2025. ML-Master: Towards AI-
for-AI via Integration of Exploration and Reasoning. arXiv
preprint arXiv:2506.16499 .
Olson, R. S.; and Moore, J. H. 2016. TPOT: A tree-based
pipeline optimization tool for automating machine learning.
InWorkshop on automatic machine learning , 66–74. PMLR.
Thornton, C.; Hutter, F.; Hoos, H. H.; and Leyton-Brown, K.
2013. Auto-WEKA: Combined selection and hyperparam-
eter optimization of classification algorithms. In Proceed-
ings of the 19th ACM SIGKDD international conference on
Knowledge discovery and data mining , 847–855.
Trirat, P.; Jeong, W.; and Hwang, S. J. 2024. Automl-agent:
A multi-agent llm framework for full-pipeline automl. arXiv
preprint arXiv:2410.02958 .
Vakhrushev, A.; Ryzhkov, A.; Savchenko, M.; Simakov, D.;
Damdinov, R.; and Tuzhilin, A. 2021. Lightautoml: Au-
toml solution for a large financial services ecosystem. arXiv
preprint arXiv:2109.01528 .
Yang, X.; Yang, X.; Fang, S.; Xian, B.; Li, Y .; Wang, J.; Xu,
M.; Pan, H.; Hong, X.; Liu, W.; et al. 2025. R&d-agent:
Automating data-driven ai solution building through llm-
powered automated research, development, and evolution.
arXiv preprint arXiv:2505.14738 .

Algorithm 1: Adding Stage at Iteration t
Input : Ideation tree Tt= (Vt, Et); context vector ct
Parameters :N(number of FE nodes), M(number of MT
nodes per parent)
Output : Updated tree Tt+1
1:ct←ct+ϕEDA(QueryEDA (Tt))
2:ct←ct+ϕext(QueryExternal ())
3:{vFE
j}N
j=1∼qFE(· |ct)
4:foreachvFE
jdo
5:{uMT
j,i}M
i=1∼qMT(· |vFE
j)
6: Assign scores {aj,i}M
i=1
7:end for
8:Tt←Backpropagation (Tt)
9:Transform scores:
sj,i=aj,i, if higher is better
−aj,i,otherwise
10:Compute probabilities:
pj,i=Softmax (sj,i)
11:Sample subset: S ∼Sample ({uMT
j,i},{pj,i})
12:foreachu∈ S do
13:{wMT
k}M
k=1∼qMT(· |u)
14:end for
15:Tt+1←Backpropagation (Tt)
16:return Tt+1
A Algorithms
A.1 Adding
This subsection formalizes the adding stage, as defined in
Algorithm 1. We define the ideation tree at iteration tas
Tt= (Vt, Et), and the global context vector as ct∈ C,
where Cdenotes the space of structured semantic represen-
tations.
The agent samples new Feature Engineering (FE) and
Model Training (MT) nodes from conditional distributions:
qFE:C → P (VFE), q MT:VFE→ P(VMT)
where VFEandVMTare the respective candidate spaces for
FE and MT nodes, and P(·)denotes the space of probability
distributions over a discrete set.
We also define a scoring function a:VMT→R, and
a softmax-based selection mechanism used to stochastically
expand the MT subtrees. The resulting tree Tt+1is obtained
by applying the additions and structural updates defined by
the algorithm.
A.2 Merging
This subsection defines the merging stage and its associated
memory mechanism. Let
Mshort,Mlong⊆ P(VFE
t)denote short- and long-term memory buffers tracking merge
success.
Algorithm 2: Merging Stage at Iteration t
Input : Ideation tree Tt= ( Vt, Et); memories
Mshort,Mlong
Parameters :N(number of FE merge candidates), M(MT
children per merged FE), θfail(failure threshold)
Output : Updated tree Tt+1, updated memo-
ries
1:{(vFE
i, vFE
j)}N
i=1∼qpair(· |Tt,Mlong)
2:foreach(vFE
i, vFE
j)do
3:vFE
ij←MergeFE (vFE
i, vFE
j)
4:{uMT
ij,k}M
k=1∼qMT(· |vFE
ij)
5:Si,Sj∼SampleTop (vFE
i, vFE
j)
6: Attach {uMT
ij,k} ∪ S i∪ SjtovFE
ij
7: ∆ij←Evaluate (vFE
ij)
8: ifPR
r=1I
IsFailure (∆(r)
ij)
⩾θfailthen
9: (vFE
i, vFE
j)∈ M long
10: else
11: ifIsFailure (∆ij)then
12: (vFE
i, vFE
j)∈ M short
13: else
14: (vFE
i, vFE
j)∈ M long
15: end if
16: end if
17:end for
18:Vmerge
FE∼qFE(· |Tt)
19:foreachvFE∈ Vmerge
FE do
20: (uMT
p, uMT
q)∼qpair(· |Children (vFE))
21: uMT
pq←MergeMT (uMT
p, uMT
q)
22: Attach uMT
pqtovFE
23:end for
24:Tt+1←Backpropagation (Tt)
25:return Tt+1,Mshort,Mlong
B Kompete-bench
Full descriptions of the competitions in Kompete-bench,
including the name, number of participants, metrics, and
medal thresholds, are listed in Table 5. The benchmark com-
prises 26 Kaggle competitions, totaling 10.2 GB in size, and
is divided into two distinct parts. The first part includes com-
petitions from MLE-Bench that remain open for submis-
sions on Kaggle and are each under 1 GB. These span from
2014 to 2017 and primarily feature straightforward tasks,
where strong leaderboard positions can be achieved without
complex modeling or novel ideas — sometimes even lever-

Figure A3: End dates for competitions presented in
Kompete-Bench
aging tools and libraries unavailable at the time of the origi-
nal competition.
The second part consists of more recent competitions
from 2024–2025, some of which are still ongoing. Achiev-
ing a medal in these requires creative approaches or training
a large number of models, reflecting the increased difficulty
and evolving standards of modern machine learning chal-
lenges.
The distribution of competition end dates is illustrated in
Figure A3, clearly showing the temporal separation between
the two parts of the benchmark.
For all competitions, we report the percentage of par-
ticipants outperformed (“percent users beaten”) as the pri-
mary metric. This choice is motivated by the observation
that current AutoML systems are unable to achieve medal
positions in the latest competitions, yet tracking progress on
these challenging tasks remains crucial. The ”percent users
beaten” metric is computed using the actual private Kaggle
leaderboard, averaged over three independent runs. If a sys-
tem fails to generate a valid submission, a score of 0% is
assigned for that run. For ongoing competitions, evaluation
is performed simultaneously on the public leaderboard.
C Setup details
We used the hyperparameter settings summarized in Table
8. For AIDE and the RD-agent, we retained their default
configurations as specified in the original implementations
except for the time limit. For KompeteAI, we empirically
selected a set of hyperparameters that strike a balance be-
tween the quality of component exploration and the compu-
tational time allocated to each. This tuning was guided by
the need to ensure efficient coverage of critical components
while maintaining tractable execution time. Below are the
hyperparameters and their descriptions for each system.
KompeteAI The list of hyperparameters used when run-
ning KompeteAI on benchmarks is given in the Table
6.The time runminutes parameter sets the maximum run-
time for the entire multi-agent system in minutes, after
which the process will terminate. The runtime error time
defines the time limit (in minutes) after which a gener-
ated code is will be stopped. The subset size inpercentspecifies the percentage of the dataset to be used for
quick validation. The validator size threshold sets a thresh-
old for the dataset size; if the data exceeds this value,
a subset is used for training and validation. The num-
berofideas eda determines how many exploratory data
analysis (EDA) ideas generates per iteration. Similarly,
number ofideas data and number ofideas modelling con-
trol the number of ideas related to feature engineering and
model training. The max addidea parameter limits how
many new ideas can be added to the idea pool in a sin-
gle adding iteration. The number ofselected node speci-
fies how many nodes are selected for adding expansion
at each step. The number ofiterations parents sets how
many iterations parent nodes participate in generating new
ideas, while number ofselected node merging determines
how many nodes are chosen for merging at each iteration.
The number ofideas min and number ofideas max define
the minimum and maximum number of ideas, which are
used as achor ideas for scoring model. The retrieve npapers
and retrieve ncompetitions parameters control how many
papers retrieves from arXiv papers and Kaggle solutions.
The number ragideas sets how many ideas are generated
using RAG. The memory size parameter determines how
many recent ideas, solutions, or states each agent remembers
for learning and decision-making. Alternatively, if mem-
orysize is set to nearest nodes, the agent’s memory consists
of the most similar nodes or ideas, rather than a fixed num-
ber, allowing for more contextually relevant recall.
Hyperparam name Value
time runminutes 360
runtime error time 30
subset size inpercent 10
validator size threshold 104
number ofideas eda 5
number ofideas data 2
number ofideas modelling 2
max addidea 2
number ofselected node 2
number ofiterations parents 2
number ofselected node merging 2
number ofiterations children 3
number ofideas min 2
number ofideas max 5
retrieve npapers 3
retrieve ncompetitions 3
number ragideas 5
Table 6: Hyperparameters used for running KompeteAI
RD-Agent The list of hyperparameters used when run-
ning RD-agent on benchmarks is given in the Table 7. The
debug timeout parameter sets the maximum time, in sec-
onds, that is allowed to debug one generated code. The
fulltimeout parameter defines the overall time limit for sys-
tem. The if action choosing based onUCB flag determines
whether the agents select their actions using the Upper Con-
fidence Bound (UCB) strategy. The enable knowledge base
flag indicates whether a shared knowledge base is enabled

Name Number of participants Metric Bronze Silver Gold Part
aerial-cactus-identification 1221 ROC-AUC ↑ 1 1 1 MLE-bench (Lite)
denoising-dirty-documents 162 RMSE ↓ 0.04517 0.02609 0.01794 MLE-bench (Lite)
dog-breed-identification 1281 log loss ↓ 0.04598 0.00539 0.0005 MLE-bench (Lite)
dogs-vs-cats-redux-kernels-edition 1315 log loss ↓ 0.06127 0.05038 0.03882 MLE-bench (Lite)
jigsaw-toxic-comment-classification-challenge 4539 mean col-wise ROC AUC ↑ 0.98639 0.98668 0.98740 MLE-bench (Lite)
leaf-classification 1596 log loss ↓ 0.01526 0.00791 0.00000 MLE-bench (Lite)
mlsp-2013-birds 81 ROC-AUC ↑ 0.87372 0.90038 0.93527 MLE-bench (Lite)
nomad2018-predict-transparent-conductors 879 RMSLE ↓ 0.06582 0.06229 0.05589 MLE-bench (Lite)
plant-pathology-2020-fgvc7 1318 ROC-AUC ↑ 0.97361 0.97465 0.97836 MLE-bench (Lite)
random-acts-of-pizza 462 ROC-AUC ↑ 0.6921 0.76482 0.97908 MLE-bench (Lite)
spooky-author-identification 1242 log loss ↓ 0.29381 0.26996 0.16506 MLE-bench (Lite)
tabular-playground-series-dec-2021 1189 ROC-AUC ↑ 0.95658 0.95658 0.9566 MLE-bench (Lite)
tabular-playground-series-may-2022 1152 ROC-AUC ↑ 0.99818 0.99822 0.99823 MLE-bench (Lite)
text-normalization-challenge-english-language 261 accuracy ↑ 0.99038 0.99135 0.99724 MLE-bench (Lite)
text-normalization-challenge-russian-language 163 accuracy ↑ 0.97592 0.98232 0.99012 MLE-bench (Lite)
eedi-mining-misconceptions-in-mathematics 1449 MAP@25 ↑ 0.46090 0.49136 0.56429 Contemporary
learning-agency-lab-automated-essay-scoring-2 2708 quadratic weighted kappa ↑ 0.83471 0.83518 0.83583 Contemporary
lmsys-chatbot-arena 1688 log loss ↓ 1.00472 0.99410 0.98392 Contemporary
pii-detection-removal-from-educational-data 2049 efficiency score ↑ 0.95714 0.95883 0.96615 Contemporary
um-game-playing-strength-of-mcts-variants 1610 RMSE ↓ 0.43050 0.42973 0.42591 Contemporary
llm-prompt-recovery 2176 Sharpened Cosine Similarity ↑0.6375 0.6513 0.6848 Contemporary
equity-post-HCT-survival-predictions 3327 C-index ↑ 0.69288 0.69320 0.69500 Contemporary
cmi-detect-behavior-with-sensor-data 2156 F1↑ 0.84 0.84 0.86 Contemporary
make-data-count-finding-data-references 833 F1↑ 0.548 0.564 0.620 Contemporary
neurips-open-polymer-prediction-2025 1539 wMAE ↓ 0.057 0.041 0.032 Contemporary
wsdm-cup-multilingual-chatbot-arena 890 categorization accuracy ↑ 0.696381 0.702772 0.711412 Contemporary
Table 5: Summary of tasks and metrics included in Kompete-bench: the table lists competition names, number of participants,
evaluation metrics, threshold values for medals and the part of the benchmark to which this competition belongs
for the agents. The loop n parameter specifies the num-
ber of main iterations the system will execute, set to 2000
to avoid early stopping not by the time limit. Finally, the
max trace num parameter limits the number of traces that
can be created during system execution.
Hyperparam name Value
debug timeout 3600
fulltimeout 21600
ifaction choosing based onUCB False
enable knowledge base False
loop n 2000
max trace num 3
Table 7: Hyperparameters used for running RD-agent
AIDE The list of hyperparameters used when running RD-
agent on benchmarks is given in the Table 8. The steps pa-
rameter defines the maximum number of steps the entire sys-
tem can perform during run. The max debug depth, speci-
fies the maximum depth for recursive code debugging. The
debug prob parameter suggests that debugging is enabled
for every generated code, meaning that all relevant infor-
mation will be recorded without any sampling. Finally, the
time limit parameter represents the maximum allowed wall-
clock time (in seconds) for experiment.Hyperparam name Value
steps 2000
max debug depth 20
debug prob 1
time limit 21600
Table 8: Hyperparameters used for running AIDE
D Metrics
Competition RD-agent AIDE KompeteAI
aerial-cactus-identification 58 5 79
denoising-dirty-documents 3 None 12
dog-breed-identification 41 62 58
dogs-vs-cats-redux-kernels-edition 86 6 91
jigsaw-toxic-comment-classification-challenge 42 26 88
leaf-classification 38 24 46
mlsp-2013-birds None 0 0
nomad2018-predict-transparent-conductors 78 33 22
plant-pathology-2020-fgvc7 12 62 68
random-acts-of-pizza 58 47 67
spooky-author-identification 71 47 59
tabular-playground-series-dec-2021 17 22 25
tabular-playground-series-may-2022 16 32 36
text-normalization-challenge-english-language 3 9 27
text-normalization-challenge-russian-language 17 14 18
Table 9: Full table by ”percent humans beaten” for AutoML
systems given in the article at each competition in MLE-
Bench part of Kompete-bench

Competition RD-agent AIDE KompeteAI
cmi-detect-behavior-with-sensor-data None None None
eedi-mining-misconceptions-in-mathematics 22 None 30
equity-post-HCT-survival-predictions None None None
learning-agency-lab-automated-essay-scoring-2 12 19 13
llm-prompt-recovery 12 9 None
lmsys-chatbot-arena None None None
make-data-count-finding-data-references 11 None 18
neurips-open-polymer-prediction-2025 13 None 15
pii-detection-removal-from-educational-data None None 29
um-game-playing-strength-of-mcts-variants None None None
wsdm-cup-multilingual-chatbot-arena 14 None 19
Table 10: Full table by ”percent humans beaten” for Au-
toML systems given in the article at each competition in
Contemporary part of Kompete-bench.
Results of AutoML system on Kompete-bench are pre-
sented in the Table 9 and Table 10. We report results using
the ”percent humans beaten” metric, which reflects the per-
centage of Kaggle leaderboard participants outperformed by
each system. For every competition, each system was evalu-
ated over three independent runs; the final score is the arith-
metic mean of these runs, rounded to the nearest integer. If a
system failed to generate any valid solution across all three
attempts, its score is reported as None (equivalent to 0 when
averaging across competitions). For individual runs where
no valid submission was produced, a score of 0 was as-
signed. Notably, the primary factor influencing overall per-
formance was the proportion of valid submissions: systems
that consistently generated correct code achieved substan-
tially higher scores.