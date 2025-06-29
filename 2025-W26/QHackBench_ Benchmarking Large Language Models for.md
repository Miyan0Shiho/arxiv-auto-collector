# QHackBench: Benchmarking Large Language Models for Quantum Code Generation Using PennyLane Hackathon Challenges

**Authors**: Abdul Basit, Minghao Shao, Haider Asif, Nouhaila Innan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique

**Published**: 2025-06-24 20:54:56

**PDF URL**: [http://arxiv.org/pdf/2506.20008v1](http://arxiv.org/pdf/2506.20008v1)

## Abstract
Recent advances in Large Language Models (LLMs) have demonstrated strong
potential in code generation, yet their effectiveness in quantum computing
remains underexplored. This paper benchmarks LLMs for PennyLane-based quantum
code generation using real-world challenges from the Quantum Hackathon (QHack).
We introduce QHackBench, a novel benchmark dataset derived from QHack
competitions, and evaluate model performance under vanilla prompting and
Retrieval-Augmented Generation (RAG). Our structured evaluation framework
assesses functional correctness, syntactic validity, and execution success
across varying challenge difficulties. Results indicate that RAG-enhanced
models, supplemented with an augmented PennyLane dataset, approximately
generate similar results as the standard prompting, particularly in complex
quantum algorithms. Additionally, we introduce a multi-agent evaluation
pipeline that iteratively refines incorrect solutions, further enhancing
execution success rates. To foster further research, we commit to publicly
releasing QHackBench, along with our evaluation framework and experimental
results, enabling continued advancements in AI-assisted quantum programming.

## Full Text


<!-- PDF content starts -->

arXiv:2506.20008v1  [cs.AI]  24 Jun 2025QHackBench: Benchmarking Large Language
Models for Quantum Code Generation Using
PennyLane Hackathon Challenges
Abdul Basit1, Minghao Shao1, Haider Asif1,2, Nouhaila Innan1,2, Muhammad Kashif1,2, Alberto Marchisio1,2, Muhammad Shafique1,2
1eBRAIN Lab, Division of Engineering New York University (NYU) Abu Dhabi , Abu Dhabi, UAE
2Center for Quantum and Topological Systems (CQTS), NYUAD Research Institute, New York University Abu Dhabi , UAE
{abdul.basit, shao.minghao, ma8183, nouhaila.innan, muhammadkashif, alberto.marchisio, muhammad.shafique }@nyu.edu
Abstract —Recent advances in Large Language Models (LLMs)
have demonstrated strong potential in code generation, yet
their effectiveness in quantum computing remains underexplored.
This paper benchmarks LLMs for PennyLane-based quantum
code generation using real-world challenges from the Quantum
Hackathon (QHack). We introduce QHackBench , a novel
benchmark dataset derived from QHack competitions, and
evaluate model performance under vanilla prompting and
Retrieval-Augmented Generation (RAG). Our structured evalua-
tion framework assesses functional correctness, syntactic validity,
and execution success across varying challenge difficulties. Results
indicate that RAG-enhanced models, supplemented with an
augmented PennyLane dataset, approximately generate similar
results as the standard prompting, particularly in complex
quantum algorithms. Additionally, we introduce a multi-agent
evaluation pipeline that iteratively refines incorrect solutions,
further enhancing execution success rates. To foster further
research, we commit to publicly releasing QHackBench , along
with our evaluation framework and experimental results,
enabling continued advancements in AI-assisted quantum
programming.
Index Terms —Quantum Computing, PennyLane, Large
Language Models, Code Generation, Benchmarking, Retrieval-
Augmented Generation.
I. Introduction and Motivation
Quantum computing has emerged as a transformative
technology, enabling solutions to classically intractable
problems [1]. With the increasing adoption of hybrid
quantum-classical computing, tools such as PennyLane have
gained prominence, facilitating Quantum Machine Learning
(QML) and variational circuit optimization [2]. However,
programming in quantum frameworks remains challenging due
to the complexity of quantum circuits and domain-specific
syntax.
Large Language Models (LLMs) such as GPT-4 have
demonstrated remarkable performance in classical code
generation and completion [3]. However, their capabilities
in quantum programming, particularly within the PennyLane
framework, remain largely unexamined. Existing benchmarks,
such as Qiskit HumanEval, primarily evaluate LLM perfor-
mance for IBM’s Qiskit framework [4], leaving a significant
gap in assessing LLM effectiveness for PennyLane-based
quantum programming. This gap is critical, as PennyLane
provides unique features such as automatic differentiation for
QML and seamless hardware integration [2], necessitating a
dedicated benchmark.
To address this issue, we introduce a novel benchmarking
approach leveraging Quantum Hackathon Challenges (QHack
Well–known QC FrameworksQiskitPennylaneQiskit Code AssistantLack of pennylane specific datasetInaccurate code generationAbsence of Standard Benchmark
Benchmark Dataset-QHackBenchCompiled Pennylane Qhack challenges from 2023-24 RAG-based Evaluation FrameworkSystematic Comparison of vanilla vs. RAG enhanced promptsOur ContributionsMulti-Agent Evaluation Pipelinemulti-agent system to iteratively refine LLM-generated codeQHackchallenges categorized in various difficulty levels/topics Self reasoning through runtime logs and debugging feedbackEnhanced prompts resulting in better/correct code generationFig. 1: An overview of our contributions.
2023-2024) as an evaluation dataset along with our baseline
evaluation framework. Our framework employs both vanilla
prompting and Retrieval-Augmented Generation (RAG) and
a more complicated agent setup in consideration of different
usage scenarios in cooperate with the modern LLM agentic
systems. leveraging a diverse set of model families for
enhanced flexibility and performance. Based on this, we
systematically analyze LLM-generated quantum code. We
compare a wide range of configurations within our evaluation
framework to establish a comprehensive baseline on our
proposed benchmark. Our study reveals that although
RAG improves accuracy in specific scenarios, such as
reducing hallucinated API calls and refining circuit structure,
existing foundation models still exhibit notable limitations in
generating reliable quantum code.
A. Motivational Case Study
In our evaluation using the o3-mini model, the generated
code was assessed on a challenge-by-challenge basis as either
correct or incorrect. In this study, we report only binary
outcomes—each challenge is either solved correctly or not.
Our experiments show that when the model’s queries were
supplemented with relevant PennyLane documentation via
RAG, the number of challenges solved were approximately
the same as the baseline approach. Furthermore, integrating a
multi-agent refinement pipeline further improved the execution
success rate significantly. These findings underscore the
importance of iterative debugging and the inclusion of
domain-specific documentation to enhance the reliability of
LLM-assisted quantum programming.
B. Research Challenges
This work addresses several key challenges in LLM-based
quantum code generation:
•Lack of PennyLane-Specific Training Data: Current LLMs
are trained on limited PennyLane-related code, often leading
to incorrect function calls.
•Inaccurate Code Generation: Without additional context,
LLMs struggle with function signatures, quantum execution

flow, and circuit correctness.
•Need for a Standardized Benchmark: Unlike Qiskit,
PennyLane lacks a dedicated benchmark for evaluating
LLM-assisted code generation.
C. Novel Contributions
To overcome the above challenges, we make the following
key contributions, which are also summarized in Fig. 1.
•Benchmark Dataset: We introduce QHackBench , the first
curated dataset of PennyLane coding challenges from
QHack 2023-2024, specifically designed to evaluate LLM
performance in quantum programming.
•Baseline Evaluation Agent Fraemwork: We propose a
structured evaluation framework, which includesiteratively
refines LLM-generated quantum code using automated
execution, self-reasoning, and debugging feedback with an
extra feature support of modern agent technologies such as
RAG and multi-agent support.
•Comprehensive baseline Evaluation: We systematically
compare LLMs with a variety of configurations, assessing
the impact of different features on quantum code correctness
and execution success.
Paper Organization: Section II discusses the basic concepts
to understand the rest of the paper and presents the related
work. Section III introduces our proposed QHackBench
methodology. Section IV discusses the experiment settings used
in the evaluation. Section V reports the experimental results.
Section VIII concludes the paper.
II. Background and Related Work
A. Large Language Models (LLMs) for Code Generation
LLMs have demonstrated remarkable capabilities in code
generation, as evidenced by models such as Codex [5],
StarCoder [6], and IBM’s Granite models [3]. These models are
trained on extensive programming datasets and benchmarked
using standardized datasets like HumanEval [5] and MBPP [7].
However, their effectiveness in niche domains such as quantum
computing is limited due to the scarcity of domain-specific
training data [8].
Prior research has explored the application of LLMs to
quantum programming, primarily focusing on IBM’s Qiskit
framework. The Qiskit Code Assistant was fine-tuned to
improve accuracy in quantum code generation tasks [9].
Additionally, Vishwakarma et al. introduced Qiskit HumanEval,
a benchmark designed to assess LLM-generated Qiskit code [4].
However, despite PennyLane’s increasing adoption in QML,
there has been limited research on evaluating LLMs for
PennyLane-based quantum programming [2]. Our work fills
this gap by systematically benchmarking LLM performance on
PennyLane tasks using real-world coding challenges.
B. Quantum Programming and PennyLane
Quantum computing frameworks facilitate various ap-
proaches to quantum programming. IBM’s Qiskit focuses
on hardware-oriented circuit execution, while Google’s Cirq
optimizes for near-term quantum devices [10]. PennyLane,
developed by Xanadu, is distinguished by its emphasis on
quantum differentiability, making it particularly well-suited
for hybrid quantum-classical computations [2]. By enablingautomatic differentiation of quantum circuits, PennyLane
supports the optimization of variational quantum algorithms
and machine learning models.
Despite PennyLane’s advantages, there is currently no
dedicated benchmark for evaluating LLM-generated code within
this framework. The lack of large-scale PennyLane datasets
presents a unique challenge, necessitating alternative evaluation
methods such as RAG [11]. Our work introduces QHackBench ,
the first structured benchmark dataset for PennyLane-based
LLM evaluation, leveraging real-world challenges from QHack
2023-2024.
C. Retrieval-Augmented Generation (RAG) for Quantum Code
RAG has emerged as an effective technique for enhancing
LLMs by incorporating relevant external knowledge sources
dynamically [12]. In classical programming, RAG has been
employed to improve code completion by retrieving real-time
documentation references [11]. In the quantum computing
domain, AWS Braket has integrated RAG-based solutions to
improve LLM-generated quantum code by providing access to
up-to-date quantum-specific knowledge.
Similarly, PennyLang , a curated dataset of PennyLane-
specific code examples and quantum computing resources,
serves as a structured retrieval corpus that enhances
LLM prompts with relevant contextual information. By
leveraging PennyLang, LLMs can access previously solved
quantum problems, reducing API hallucinations and improving
functional correctness in PennyLane programming [13]. The
dataset consists of four primary sources: unofficial GitHub
repositories, official PennyLane documentation, open-source
contributions from PennyLaneAI, and quantum computing
books. The composition of the dataset is shown in Fig. 2.
Fig. 2: Composition analysis of the PennyLang dataset (3,347 samples).
D. Multi-Agent Systems for Quantum Code Generation
Multi-agent systems have been widely explored in AI
research to enhance problem-solving by enabling specialized
agents to collaborate on complex tasks [14]. In programming,
multi-agent frameworks have been employed for code synthesis
and debugging, where different agents handle distinct roles

QHack Challenges
Query
PennyLang
DatasetIndexing Multi-Agent System
RetrievalChallenge Description
Retrieval Code Samples
Augment Samples in Query
Evaluation on Test BenchEmbeddingsCode Samples
FunctionalityChunks Vectors
YesNo
Challenge Template
Claude 3.7
DeepSeek V3
GPT 4.1
LLaMa 4
Qwen 3GPT o3-Mini
DeepSeek-V3
...Base Query
Augmented
Query
CorrectIncorrect12
Automated Execution
Evaluation on Test BenchValidator
Agent
(Same LLM)
FunctionalityEnhanced
Augmented
QueryCorrect IncorrectNo Solution
Found
3
Incorrect Results
Test Bench ErrorsRuntime LogsSelf-Reasoning Iteration < 5Reformulate
Query
Fig. 3: Overview of the QHackBench .1⃝Base query. The QHack challenge description and template are provided to LLMs (GPT-4.1, Claude 3.7, ...) for
initial code generation. 2⃝Retrieval-augmented query If enabled, relevant PennyLang code snippets are retrieved, embedded, and incorporated into the query
to enhance accuracy. 2⃝Multi-agent pipeline: a validator agent, implemented with the same LLM, inspects run-time logs and hidden-test feedback, performs
self-reasoning, reformulates the query, and asks the builder to regenerate code. This ensures correctness evaluation, error diagnosis, and improved execution.
such as code generation, execution validation, and iterative
refinement [15].
For quantum programming, a multi-agent approach is
particularly valuable due to the inherent complexity of
debugging quantum circuits, where execution errors often
stem from subtle query misinterpretations or incorrect circuit
logic [16].
Our experimental results demonstrate that integrating a
multi-agent framework significantly improves execution success
rates compared to standalone prompting or RAG-based
approaches. By systematically evaluating LLM performance in
a multi-agent setting, we provide the first in-depth analysis of
iterative refinement in quantum code generation.
III. QHackBench Methodology
QHackBench systematically evaluates LLM performance in
quantum code generation through an automated benchmarking
framework for QHack challenges. Our methodology integrates
RAG with the PennyLang dataset to enhance model
responses, benchmarking them against non-RAG baselines. The
evaluation assesses correctness, execution success, and iterative
improvements within a structured pipeline.
A. Automated Evaluation Pipeline
To ensure consistency and reproducibility in benchmarking
LLMs for quantum code generation, we design a structured
evaluation pipeline that systematically assesses model
performance across different configurations. The overall
evaluation pipeline is illustrated in Algorithm 1. The process
begins with dataset preparation, where QHack challenge
descriptions, template notebooks, and corresponding test scripts
from the years 2023 and 2024 are gathered to establish a
standardized benchmark. Next, we integrate RAG by retrieving
relevant code snippets from the PennyLang dataset [13],
providing models with enhanced contextual knowledge for more
accurate code completion. To analyze the effectiveness of RAG,
we evaluate multiple LLMs, including GPT-4o mini (RAG and
non-RAG) and GPT-o3 Mini (RAG and non-RAG), allowing
a direct comparison between retrieval-enhanced responses and
baseline prompting.
Once the solutions are generated, we employ an automatedexecution system that runs the code in an isolated environment,
verifying functional validity and correctness against predefined
test cases. This is followed by the performance evaluation, where
we measure key metrics such as Pass@ 𝑘, execution success
rate, and error classification to quantify model effectiveness.
Additionally, we incorporate an iterative refinement process
using a multi-agent system, wherein an automated validation
mechanism analyzes execution errors and refines incorrect
solutions iteratively. By following this structured sequence,
spanning code generation, retrieval augmentation, correctness
verification, iterative improvement, and automated execution,
our pipeline provides a robust and scalable framework for
benchmarking LLMs in quantum programming, enabling
precise evaluation of their capabilities in solving QHack
challenges.
Algorithm 1 Automated Benchmarking Pipeline for QHack
Challenges
Require: Quantum challenge dataset D, LLM modelsM, PennyLang retrieval corpus
R
1:foreach challenge(𝐶𝑖,𝑇𝑖)∈D do
2: Retrieve relevant context S𝑖fromRusing embedding similarity.
3: foreach model𝑀𝑗∈M do
4: Generate quantum code Q𝑗
𝑖using RAG-assisted and non-RAG prompting.
5: ExecuteQ𝑗
𝑖in a controlled environment.
6: Collect execution logs L𝑗
𝑖and correctness status.
7: end for
8: Perform error analysis and refine Q𝑗
𝑖iteratively (if multi-agent mode is enabled).
9:end for
10: Compute benchmark metrics: Pass@ 𝑘, execution success rate, error classification.
B. Retrieval-Augmented Generation with PennyLang Dataset
To enhance LLM-based quantum code generation, we
integrate retrieval from the PennyLang dataset, a curated
collection of resources relevant to PennyLane-based quantum
programming. For each challenge, a similarity-based retriever
selects the most relevant context from Rusing a vector database.
This retrieved knowledge is appended to the model prompt,
improving accuracy and reducing hallucinations. The RAG
process is outlined in Algorithm 2.
C. Multi-Agent System for Iterative Code Refinement
To improve the reliability of LLM-generated quantum code,
we employ a multi-agent system that iteratively refines incorrect
solutions. The system consists of two main agents:
•Code Generation Agent (GPT-o3-Mini, DeepSeek-V3):

Algorithm 2 Retrieval-Augmented Generation (RAG) for
Quantum Code
Require: Query input𝑄, retrieval corpusR, embedding model E
1:Compute query embedding 𝑒𝑄=E(𝑄)
2:Retrieve top-𝑘documentsS=arg max 𝑆𝑖∈Rsimilarity(𝑒𝑄,E(𝑆𝑖))
3:Construct prompt:P=system prompt+S+𝑄
4:Generate response: Q=LLM(P)
5:return Generated codeQ
Fig. 4: Difficulty distribution among all the challenges.
Generates initial quantum code solutions using either RAG
or non-RAG methods.
•Validation & Correction Agent (GPT-o3-Mini): Executes
the generated code, analyzes errors from execution logs, and
iteratively refines the solution through self-reasoning and
debugging.
The multi-agent system follows a structured pipeline designed
to iteratively improve LLM-generated quantum code by
leveraging both RAG and automated debugging. The process
begins with the Code Generation Agent, which produces an
initial quantum circuit implementation based on the QHack
challenge description. This generated solution is then executed
within a controlled Jupyter environment, where its correctness
is evaluated against predefined test cases. If execution fails,
the Validation & Correction Agent analyzes the runtime logs,
extracting error messages, incorrect results, and test bench
failures. Based on this analysis, the agent formulates a structured
debugging report, identifying potential issues and suggesting
modifications. The revised query, incorporating these debugging
insights, is then sent back to the Code Generation Agent for
solution refinement. If RAG is enabled, the retrieved context
from the PennyLang dataset is also included in the updated
query to enhance code corrections. This process is repeated
iteratively, with a maximum of five refinement attempts,
ensuring that the solution is progressively improved until it
successfully passes all test cases or reaches the retry limit.
IV. Experimental Setup
A. Benchmark Dataset
We curated a dataset comprising 49 unique QHack challenges
from 2023-2024 [17], [18]. These challenges span multiple QC
domains, including quantum algorithms, QML, and quantum
chemistry. The dataset is structured based on the yearly challenge
formats and categorized by difficulty levels and thematic
divisions.
B. Structure of QHack Challenges
QHack challenges adhere to a structured format, with
problems assigned to predefined categories and difficulty levels.The dataset is categorized as follows:
•2023: Challenges were structured within five narrative-based
thematic sets: A Tale of Timbits, Bending Bennett’s Laws,
Fall of Sqynet, and Office Hijinks. Each narrative contained
five challenges, with a standardized point distribution ranging
from 100 to 500 points. A total of 28 challenges were included,
with 8 tutorial challenges assigned 0 points. The introduction
of a narrative-driven format added an engaging element to
problem-solving.
•2024: The challenge structure evolved further with a
thematic-based organization, introducing five core themes:
Boson Beach, Dipole Desert, FemtoForest, Tensor Tundra,
and Qutropolis. This iteration featured 21 challenges with
difficulty levels spanning 100 to 600 points. Higher-difficulty
problems were more prevalent in later themes, reflecting an
increased focus on complex quantum problem-solving.
TABLE I: Full challenge list of QHackBench.
Year Theme Challenge Points Difficulty2023 (28 Challenges)TutorialsTutorial 1 (C1-E-23) 0 Easy
Tutorial 2 (C2-E-23) 0 Easy
Tutorial 3 (C3-E-23) 0 Easy
Tutorial 4 (C4-E-23) 0 Easy
Tutorial 5 (C5-E-23) 0 Easy
Tutorial 6 (C6-E-23) 0 Easy
Tutorial 7 (C7-E-23) 0 Easy
Tutorial 8 (C8-E-23) 0 Easy
A Tale of
TimbitsThe Magic 5 Ball (C9-E-23) 100 Easy
Cascadar (C10-I-23) 200 Intermediate
A Pauli Worded Problem (C11-I-23) 300 Intermediate
Entangled Robot Swarms (C12-A-23) 400 Advanced
One Bit Wonder (C13-A-23) 500 Advanced
Bending
Bennett’s
LawsCtrl-Z (C14-E-23) 100 Easy
Sub Superdense Coding (C15-I-23) 200 Intermediate
Secrets in Spacetime (C16-I-23) 300 Intermediate
A Halfway Decent Photocopier (C17-A-23) 400 Advanced
The Itch to Switch (C18-A-23) 500 Advanced
Fall of
SqynetSqy Trotter (C19-E-23) 100 Easy
Unitary Operators (C20-I-23) 200 Intermediate
Don’t Hit the Ground (C21-I-23) 300 Intermediate
Desperate Measures (C22-A-23) 400 Advanced
Ising Uprising (C23-A-23) 500 Advanced
Office
HijinksTick Tock Bloch (C24-E-23) 100 Easy
The Super Parameter (C25-I-23) 200 Intermediate
The Change of Qubit (C26-I-23) 300 Intermediate
The Lazy Colleague (C27-A-23) 400 Advanced
The False Proof (C28-A-23) 500 Advanced2024 (21 Challenges)Boson
BeachHacking for the UpGrayde (C1-E-24) 100 Easy
Save QHack Beach (C2-I-24) 200 Intermediate
Mathematicians at the Resort (C3-I-24) 300 Intermediate
Mojito HHLime Twist (C4-A-24) 400 Advanced
The Three Shipping Companies (C5-A-24) 500 Advanced
Dipole
DesertDunes out of Context (C6-E-24) 100 Easy
The Oasis Ruins (C7-I-24) 200 Intermediate
The Grand Hideaway Zone Inn (C8-I-24) 300 Intermediate
A Market of Quantum Trinkets (C9-A-24) 400 Advanced
The Wormhole Airdrome (C10-A-24) 500 Advanced
Femto
ForestTo View or Not to View (C11-E-24) 100 Easy
Rainy Days at the Retreat (C12-I-24) 200 Intermediate
The Mach-Zehnder Cabin (C13-I-24) 300 Intermediate
Stay Out of This Swamp! (C14-A-24) 400 Advanced
Lazy Workers at the Terminal (C15-A-24) 500 Advanced
Tensor
TundraThe Chalet of the Random Gate (C16-E-24) 100 Easy
Hockey Night in the Cave (C17-I-24) 200 Intermediate
Critical Coffee Conundrum (C18-I-24) 300 Intermediate
The Triple H Hotel (C19-A-24) 400 Advanced
Travelling on the Eigentracksly (C20-A-24) 500 Advanced
Qutropolis Fireworks in Qutropolis (C21-A-24) 600 Advanced
Each challenge consists of a problem statement, a Jupyter
Notebook or Python template for completing the code, and a
structured testing methodology. The dataset includes problem
descriptions, solution templates, and test functions.
To assess trends in quantum computing problem-solving,
we conduct a statistical analysis of the challenge dataset (see

C1-E-23C2-E-23C3-E-23C4-E-23C5-E-23C6-E-23C7-E-23C8-E-23C9-E-23C10-I-23C11-I-23C12-A-23C13-A-23C14-E-23C15-I-23C16-I-23C17-A-23C18-A-23C19-E-23C20-I-23C21-I-23C22-A-23C23-A-23C24-E-23C25-I-23C26-I-23C27-A-23C28-A-23C1-E-24C2-I-24C3-I-24C4-A-24C5-A-24C6-E-24C7-I-24C8-I-24C9-A-24C10-A-24C11-E-24C12-I-24C13-I-24C14-A-24C15-A-24C16-E-24C17-I-24C18-I-24C19-A-24C20-A-24C21-A-24CLAUDE 3.7  RAG
CLAUDE 3.7  Non-RAG
DeepSeek V3  RAG
DeepSeek V3  Non-RAG
GPT 4.1  RAG
GPT 4.1  Non-RAG
LLama-4  RAG
LLama-4  Non-RAG
Qwen-3  RAG
Qwen-3  Non-RAG
×××××××××××××××××××××××××××××××××××××××××××××××××
1××3×1213××4×5×4××2××××1××××1×××××××××12×5×2×××××
11×××1×××××××1××1×××1×11××××1×××××××××1××××××××××
1×11×11111×××1×21×1×4×11×1××1××××1×1××11×1×3×2×1×
13×3×13××××2×1×4××××1××1××××1×××××××××2××××4×××3×
1××××13×4××××12×××2×2××1××××1××××2××××1521×2×4×××
1×22×1×132×××2×11×3×3×1152××1×××××××××2234×3××2××
1434×121323××1×11×34××11××××1××××3××××32×3××××2××
×××××××××××××1×××××××××1××××4×××××××××1××××××××××
22×××××××2××××××××××××11××××2××××××××××××××××××××Challenge Iterations  RAG vs Non-RAG (2023 + 2024)
12345Number of IterationsFig. 5: Per–challenge performance heat-map for all models on the complete QHack 2023 & 2024 challenge suites. Each row corresponds to a model–prompting
method pair (RAG or Non-RAG); each column is a single challenge, ordered first by 2023 codes ( C1{E{23 . . .C28{A{23 ) followed by 2024 codes ( C1{E{24 . . .
C21{A{24 ). The integer written inside every solved cell is the number of iterations required before the model produced an accepted solution (lower is better). Color
shading encodes the same metric monotonically from one to five iterations, while light-grey cells marked with “ ×” indicate that the challenge remained unsolved
within the five-attempt budget. This compact view highlights (i) model-specific strengths, (ii) the impact of retrieval-augmented generation (compare each model’s
RAG vs. Non-RAG row), and (iii) year-to-year differences in challenge difficulty across the full benchmark.
Table I). Each challenge was assigned a difficulty score based
on complexity: Easy (0–100 points), intermediate (200–300
points), and advanced (400–600 points). Key observations
include:
•Challenge Distribution: Each year maintains a balanced
distribution of challenges, with a gradual shift in complexity.
In 2023, 28 challenges are introduced, spanning easy,
intermediate, and advanced levels, while in 2024, 21
challenges are included, reflecting an increasing focus on more
complex problem-solving.
•Difficulty Levels: Over time, the proportion of advanced-level
challenges increased slightly, with 8 Advanced challenges
in 2023 rising to 9 in 2024. Additionally, no tutorial-level
challenges were included initially, but a 600-point
problem was introduced in 2024, marking a shift toward
higher-difficulty problem-solving.
Our evaluation considers functional correctness, i.e., the
code execution must match expected outputs. Figure 4 shows
a clear alignment between point values and difficulty levels:
easy challenges cluster at 0–100 points, intermediate ones peak
at 200–300, and advanced challenges dominate the 400–600
range. The sharp drop in count at 600 points highlights its rarity
and difficulty. This reflects a deliberate and consistent mapping
from score to challenge complexity.
C. Evaluation Framework
To benchmark the effectiveness of the multi-agent refinement
process, we evaluate the system on QHack coding challenges
from 2023 and 2024. Two evaluation settings are considered:
•Single-agent generation. A “builder” LLM receives the
challenge and must solve it in at most five attempts, with
or without RAG .
•Multi-agent refinement. A validator LLM inspects failing
runs and reformulates the query; the builder then regenerates
code. Only this pipeline uses the lightweight validator models
listed below.
The following models are considered for Multi-agent
evaluation but the pipeline is extendable.
•GPT-o3-Mini (RAG and non-RAG) : Lighter-weight model
used for both code generation and debugging.
•DeepSeek-V3 : Specialized model for advanced code
completion.1) Execution Pipeline
Each challenge is processed through the following pipeline:
1)Initial Code Generation: The base query (problem
description and template) is passed to the Code Generation
Agent.
2)Solution Execution: The generated solution is injected into
a Jupyter notebook and executed using Papermill .
3)Validation and Debugging: The Validator Agent extracts
runtime logs, test errors, and execution failures.
4)Solution Refinement: Based on debugging feedback, the
next iteration generates a revised solution.
5)Iteration Limit: The process repeats until a correct solution
is found or a maximum of five attempts is reached.
2) Retrieval-Augmented Debugging
Using RAG, the validator performs a ChromaDB vector
search ( MMR , top- 𝑘=3) over the PennyLang corpus and injects
the most relevant code chunks into the follow-up query.
3) Evaluation Metrics
To compare the efficiency of RAG-based and non-RAG
debugging, we measure:
•Pass@ 𝑘:Probability of obtaining a correct solution within
the first 𝑘attempts.
•Execution Success Rate: Percentage of solutions that execute
without runtime errors.
•Correction Efficiency: Average number of iterations needed
for successful code refinement.
4) Computational Resources
All experiments were conducted in a high-performance
computing environment with:
•Software: Python 3.10, LangChain, OpenAI / Anthropic /
DeepSeek APIs, JupyterLab, and Papermill for execution
monitoring.
•Retrieval Setup: ChromaDB for vector-based retrieval,
optimized for PennyLang dataset queries.
By testing multiple LLMs across different retrieval
and iteration strategies, we assess how retrieval-augmented
debugging enhances quantum code generation accuracy.
V. Results
Benchmarking Results : Out evaluation reveals substantial
performance heterogeneity, as detailed in Table II. LLaMa 4

emerges as the most robust model overall, achieving a high
average accuracy of 47.0% with RAG and 44.0% in its standard
configuration, with corresponding scores of 2950 and 2600,
respectively. While LLaMa 4 shows strong all-around capability,
DeepSeek V3 establishes itself as the definitive leader in non-
RAG performance, attaining the highest average accuracy of
49.4% and the top score of 3000. In contrast, GPT-4.1 maintains
relatively stable, albeit more moderate, performance across both
modes.
A significant finding is the pronounced year-over-year
performance degradation observed in some models, suggesting
the 2024 challenge set introduced greater difficulty. For instance,
LLaMa 4 ’s RAG accuracy dropped from 60.7% in 2023 to
33.3% in 2024. Even more dramatically, DeepSeek V3 ’s RAG
performance fell from 32.1% to a near-failure 4.8%. This trend
indicates that model robustness over evolving task complexity
remains a critical area for improvement across more diverse
challenge sets. One possible explanation for the 2024 drop is
that some models may have inadvertently memorized parts
of the 2023 challenge set during pretraining, leading to data
contamination in the earlier year.
TABLE II: Model Accuracy Performance
Model RAG 2023 2024 Avg.
Acc. (%) Score Acc. (%) Score Acc. Score
Claude 3.7 ✓ 0.0 0 0.0 0 0.0 0
✗ 42.9 2400 19.0 1500 31.0 1950
DeepSeek V3 ✓ 32.1 1900 4.8 200 18.5 1050
✗ 60.7 3600 38.1 2400 49.4 3000
GPT 4.1 ✓ 39.3 1900 14.3 700 26.8 1300
✗ 35.7 2300 33.3 2300 34.5 2300
LLaMa 4 ✓ 60.7 3600 33.3 2300 47.0 2950
✗ 64.3 3500 23.8 1700 44.0 2600
Qwen 3 ✓ 0.0 0 0.0 0 0.0 0
✗ 21.4 1100 0.0 0 10.7 550
Legend: Complete Failure (0%) — Best — Good (≥35%)
Attempt Analysis : Table II shows important insights into
model problem-solving dynamics and computational resource
allocation. Successful models typically converge within 1–3
iterations on elementary tasks, with iteration counts increasing
systematically for more complex problems. DeepSeek V3
displays remarkable consistency across modes, often solving
early-stage challenges in a single iteration and maintaining
stable performance in medium-difficulty tasks. LLaMa 4 ,
while efficient in RAG mode during the 2023 set, exhibits
greater iteration variability in 2024, suggesting difficulty in
adapting to new problem formulations. The universal failure
ofClaude 3.7 under RAG conditions, juxtaposed with its
sporadic non-RAG successes, reinforces the earlier hypothesis
of architectural misalignment with RAG paradigms.
For advanced-level challenges across both years, most models
either exhaust the 5-iteration limit or fail to produce valid
solutions, marking clear performance boundaries. This behavior
affirms the discriminative strength of the QHackBench and
highlights current limitations in scaling AI systems toward
expert-level reasoning and problem-solving.
RAG vs non-RAG : The comparative analysis between
RAG and non-RAG uncovers a crucial binary compatibility
C8-I-24C14-A-24C16-E-24C18-I-24C6-E-24C21-A-24C1-E-24C17-I-24C15-A-24C13-I-24C9-A-24C3-I-24C4-A-24C11-E-24C2-I-24C20-A-24C19-A-24C10-A-24Non RAG
RAG
Multi Agent
×
 ×
 ×
 ×
 ×
 ×
×
 ×
 ×
 ×
 ×
 ×
 ×
×
 ×
 ×
 ×Challenge Success Comparison on o3 mini ({Method: Non RAG, RAG, Multi Agent})
Solved ( )
Not Solved Solved (×)
C8-I-24C14-A-24C16-E-24C18-I-24C6-E-24C21-A-24C1-E-24C17-I-24C15-A-24C13-I-24C9-A-24C3-I-24C4-A-24C11-E-24C2-I-24C20-A-24C19-A-24C10-A-24o3 mini
DeepSeek V3112225153232415325
515525155555222225Challenge Iterations with Multi-Agent Approach
(o3 mini vs. DeepSeek V3 Multi agent)
BestWorst
IterationsFig. 6: Top Panel: Matrix depicting the correctness of results given by each of the
Non-RAG, RAG, and Multi-Agent system methods for each problem. All three
methods use the o3-mini model as the model performing the code completion.
The Multi-Agent system further uses the o3-mini model to debug prior incorrect
responses. Bottom Panel: Heatmap comparing iteration counts (fewer is better)
for each challenge under the o3-mini and DeepSeek V3 multi-agent methods.
Green cells indicate lower iteration counts (better performance), while red cells
represent higher iteration counts or unsolved tasks (worse performance).
split among models. A particularly striking observation is
the complete failure of certain architectures when integrated
with RAG. Both Claude 3.7 and Qwen 3 achieve zero
accuracy in all RAG test conditions. However, they retain
partial functionality in non-RAG mode, with Claude 3.7
achieving 31.0% accuracy (1950 score) and Qwen 3 reaching
10.7% (550 score). This contrast underscores that for RAG
to be effective, a model must possess baseline conceptual
understanding compatible with the retrieval mechanism, which
is further discussed in Section VI-B.
Among the RAG-enhanced models, performance tendencies
diverge significantly. DeepSeek V3 exhibits a strong preference
for non-RAG operation, with a 31-percentage-point accuracy
advantage (49.4% vs. 18.5%), suggesting difficulties in utilizing
retrieved context. In contrast, LLaMa 4 demonstrates relatively
balanced performance, with a slight preference for RAG (47.0%
vs. 44.0%), while GPT-4.1 also favors non-RAG mode (34.5%
vs. 26.8%). These results indicate that beyond requiring a
baseline understanding of task-relevant concepts and knowledge,
models differ markedly in their ability to incorporate retrieved
information, especially in tasks where training data is limited.
Fig. 5 provides a granular view of this behavior,
revealing differences in performance consistency. DeepSeek
V3(non-RAG) and LLaMa 4 (in both modes) demonstrate broad
success across a wide array of individual challenges. In contrast,
the successes of GPT 4.1 andQwen 3 (non-RAG) are more
sporadic, indicating a narrower range of solvable problems. This
suggests that top-performing models are distinguished not just
by their average accuracy, but by their ability to generalize across
a diverse set of tasks.
Evaluation on multi-agent Figure 6 compares three
prompting methods, vanilla, RAG, and multi-agent, on QHack
2024 challenges using the o3-mini model. The multi-agent
approach consistently outperforms both baselines, solving more
tasks, including several that RAG or vanilla failed to address.
When paired with DeepSeek V3, the multi-agent strategy
further demonstrates lower or comparable iteration counts
across many challenges, indicating faster convergence. These
improvements highlight the effectiveness of an internal feedback

TABLE III: Comparative Analysis of Error Rates on QHackBench by year and model ( %)
Error Types: Syn = Syntax Errors, NoOut = No Output, Inc = Incomplete Output, Run = Runtime Errors
Model 2023 2024 Average
Syn NoOut Inc Run Syn NoOut Inc Run Syn NoOut Inc Run
LLaMA 4 5.2 57.8 8.4 28.6 9.2 52.9 13.7 24.2 7.2 55.4 11.1 26.4
Qwen3 1.8 96.0 1.1 1.1 2.0 88.4 1.0 8.6 1.9 92.8 1.1 4.2
Deepseek-V3 0.0 96.3 0.0 3.7 1.3 91.6 0.0 7.1 0.6 94.0 0.0 5.3
GPT-4.1 5.3 74.9 9.6 9.6 11.2 66.5 6.8 15.5 8.0 71.0 8.3 12.4
Claude 3.7 0.4 96.3 0.4 2.9 1.0 94.2 0.0 4.9 0.7 95.3 0.2 3.8
Color Scale: ■0-25% ■25-50% ■50-75% ■75-100%
loop, where agents iteratively refine and debug their outputs.
Overall, the results affirm that multi-agent prompting enhances
model capability in quantum programming tasks by leveraging
structured self-correction.
VI. Discussions
A. Failure Analysis
Table III presents an analysis of the common failure modes
observed during our baseline evaluation across the five tested
models. The most prevalent failure type is the absence of
generated output, with suppression rates ranging from 55.4%
forLLaMA 4 to 95.3% for Claude 3.7 . This widespread
suppression behavior is primarily attributable to the models’
insufficient understanding of quantum programming tasks,
which prevents them from generating valid solution code. These
results highlight a critical limitation in current LLMs , as quantum
programming appears to exceed their capability thresholds,
leading many models to suppress output entirely rather than
risk producing incorrect responses. The disparity in suppression
rates is particularly notable: while Claude 3.7 ,DeepSeek-V3 ,
andQwen3 all exhibit suppression rates exceeding 90%, LLaMA
4demonstrates considerably better responsiveness at 55.4%.
This variation underscores substantial differences among model
families, likely driven by factors such as model architecture,
training data composition, and learning strategies, which in turn
result in differing levels of adaptability to novel and conceptually
demanding tasks.
For code that is successfully generated, runtime errors
emerge as the most common form of active failure. This is
particularly pronounced in LLaMA 4 , which shows a 26.4%
runtime error rate, significantly higher than 3.8%–12.4%
range observed in other models. This pattern suggests that even
if LLMs generally grasp the syntax of quantum programming on
some challenges, they struggle with semantic correctness. The
generated code is often syntactically valid but fails at execution
due to logical flaws, such as improper gate sequencing or invalid
qubit operations. In contrast, syntax errors , including, but
not limited to, malformed function signatures or attempts to
access non-existent properties or functions, remain consistently
low (0.6%–8.0%). This indicates that structural language
generation is largely a solved problem, specifically, Python on
this task, with most failures stemming from deeper conceptual
misunderstandings rather than surface-level formatting.
Insight : These failure patterns highlight fundamental
limitations in current LLMs’ understanding of quantum
computing. First, the predominance of no output failure suggestsmodels recognize quantum programming as distinct from
classical programming but lack the training data or architectural
adaptations to engage confidently with such tasks. Second,
the high rate of runtime errors among models that do
generate output indicates syntactic fluency alone is insufficient,
as quantum programming demands deeper understanding of
quantum principles, circuit constraints, and hardware-specific
behaviors. Third, these challenges raise concerns about whether
general-purpose LLMs can perform effectively in specialized
scientific domains without architectural innovation or require
purpose-built models trained on domain-specific data and
evaluation protocols.
B. Case Studies
To better understand RAG’s inconsistent performance in
quantum code generation on QHackBench, we present two case
studies on Tutorial 1 from QHack 2023 highlighting successful
and failed use cases. We analyze instances where RAG improves
generation quality (LLaMA-4) and introduces critical errors
(Qwen-3), conducting detailed error analysis to trace how
retrieval noise causes failure. These cases illustrate how RAG’s
effectiveness depends on both context quality and the model’s
ability to integrate relevant information into executable code.
Case 1: LLaMA-4 - RAG vs Non-RAG (Tutorial #1)
Challenge summary. Tutorial #1 asks competitors to
implement add numbers so that the function passes
a hidden test-suite checking 𝑥+𝑦within 10−4relative
error. The task is simple and most of the models solve it
in first attempt.
1# RAG solution
2def add_numbers(numbers):
3 """
4 Return x + y , where numbers == [x, y].
5 Matches QHack it_adds_up checker tolerance
(1e-4).
6 """
7 x, y = numbers
8 return float(x) + float(y)
9
10# non-RAG solution
11def add_numbers(numbers):
12 x, y = map(float , numbers)
13 return x + y
Listing 1: LLaMA-4 Implementations (RAG equals Non-RAG)
In Case 1, both RAG and non-RAG approaches yield correct
solutions with LLaMA-4 , with RAG offering model more slights

on quantum code.
Case 2: Qwen-3 - RAG vs Non-RAG (Tutorial #1)
Key observation. Contrary to the LLaMA-4 case, Qwen-3’s RAG-augmented
variant exhausted the five-attempt budget without passing the checker.
Inspection of the retrieved context revealed irrelevant mathematical references
that distracted the model from the concise task specification.
Limitations Observed (RAG run):
•Retrieved passages contained floating-point precision caveats that led the
model to over-engineer a tolerance loop and fail hidden tests.
•The function returned Decimal objects instead of float , breaking the
autograder’s type check.
•No explicit fallback when the retrieval step returned empty context.
1def add_numbers(numbers):
2 x, y = map(float , numbers)
3 return x + y
Listing 2: Qwen-3 Implementation (Non-RAG: Correct ✓)
RAG trace (excerpt showing two rejected attempts):
1# RAG attempt #1 (rejected)
2<think> </think> ‘‘‘python #These tokens are causing
syntax error.
3def add_numbers(numbers):
4 result = x + y
5 return str(result)
6
7# RAG attempt #5 (rejected)
8### Problematic Code Example
9
10Here’s an example of what might be causing the error:
11‘‘‘python
12def add_numbers(x, y):
13 return x + y
14‘‘‘
15Even though the logic of ’return x + y’ is correct, the
character makes Python throw a ’SyntaxError’.
16---
17### Corrected Code
18To fix it, simply remove the invalid characters and ensure
your function looks like this:
19‘‘‘python
20def add_numbers(x, y):
21 return x + y
22‘‘‘
Listing 3: Qwen-3 Implementation (RAG: Incorrect ✗)
The model repeatedly introduces either incorrect tokens or other text which
causes a syntax error in the running of the generated code, The extra tokens
are caused by LLM hallucination.
This case highlights how RAG can degrade performance
when irrelevant context misleads the model, demonstrating
the importance of retrieval quality and fallback strategies.
The retrieved mathematical references caused Qwen-3 to
over-engineer the solution with unnecessary precision handling
and type conversions, ultimately leading to systematic failures
across all five attempts.
VII. Limitations and Future Work
While QHackBench offers a structured benchmark for
evaluating LLMs in quantum computing, our results highlight
key limitations. (1) Even with RAG, models frequently produce
hallucinated APIs and incorrect quantum circuits, revealing
a lack of domain-specific understanding despite access to
documentation. (2) Binary pass@k fails to capture semantic
errors; many outputs are syntactically correct but fail at runtime
due to flawed quantum logic. Future work should focus on
quantum-specific training frameworks, richer datasets, and
integrating quantum principles into training objectives. Agentic
models tailored to quantum tasks are a promising direction, withQHackBench serving as both benchmark and testbed.
VIII. Conclusion
We introduced QHackBench , the first benchmark for
evaluating LLMs on PennyLane-based quantum code generation
using QHack 2023–2024 challenges, alongside a baseline
framework for low-data settings. Comparing vanilla prompting,
RAG, and a simple multi-agent refinement method, we
find that RAG boosts correctness, while the multi-agent
approach improves execution success via iterative reasoning.
Our results show RAG’s effectiveness depends on the
model’s quantum understanding, and agentic methods offer
further gains. Evaluations on GPT-4o, GPT-o3-Mini, and
DeepSeek-V3 highlight both capabilities and limitations
in quantum programming, demonstrating the promise of
agent-based techniques in this domain.
References
[1] F. Arute et al. , “Quantum supremacy using a programmable
superconducting processor,” Nature , 2019.
[2] V. Bergholm et al. , “PennyLane: Automatic differentiation of hybrid
quantum-classical computations,” arXiv:1811.04968 , 2018.
[3] N. Dupuis et al. , “Qiskit Code Assistant: Training LLMs for generating
Quantum Computing Code,” arXiv preprint arXiv:2405.19495 , 2024.
[4] S. Vishwakarma et al. , “Qiskit HumanEval: An Evaluation Benchmark
for Quantum Code Generative Models,” arXiv:2406.14712 , 2024.
[5] M. Chen, J. Tworek, H. Jun et al. , “Evaluating large language models
trained on code,” arXiv preprint arXiv:2107.03374 , 2021.
[6] R. Li, A. Q. Jiang, C. Qian et al. , “Starcoder: May the source be with
you!” arXiv preprint arXiv:2305.06161 , 2023.
[7] J. Austin et al. , “Program synthesis with large language models,” arXiv
preprint arXiv:2108.07732 , 2021.
[8] L. Watson, “QHack 2022 - the one-of-a-kind celebration of quantum
computing,” PennyLane Blog, uRL: https://pennylane.ai/blog/2022/03/
qhack-2022-the-one-of-a-kind-celebration-of-quantum-computing.
[9] D. G. Almeida, J. Cruz-Benito, and R. Davis, “Introducing qiskit code
assistant,” IBM Quantum Computing Blog (Oct 9, 2024), uRL: https:
//www.ibm.com/quantum/blog/qiskit-code-assistant.
[10] C. Gidney and M. Newman, “Cirq: A python framework for creating,
editing, and invoking noisy intermediate scale quantum (nisq) circuits,”
Quantum Science and Technology , vol. 6, no. 1, p. 014002, 2021.
[11] Y. Kharkov, Z. Mohammad, M. Beach, and E. Kessler, “Accelerate
quantum software development on amazon braket with claude-3,” AWS
Quantum Technologies Blog (18 Sep 2024), uRL: https://aws.amazon.c
om/blogs/quantum-computing/accelerate-quantum-software-developme
nt-on-amazon-braket-with-claude-3/.
[12] P. Lewis et al. , “Retrieval-augmented generation for knowledge-intensive
nlp tasks,” arXiv preprint arXiv:2005.11401 , 2020.
[13] A. Basit et al. , “Pennylang: Pioneering llm-based quantum code generation
with a novel pennylane-centric dataset,” arXiv preprint arXiv:2503.02497 ,
2025.
[14] M. Wooldridge, An Introduction to MultiAgent Systems . John Wiley &
Sons, 2009.
[15] H. Jin, Z. Sun, and H. Chen, “Rgd: Multi-llm based agent debugger via
refinement and generation guidance,” 2024.
[16] W. Yang et al. , “Coast: Enhancing the code debugging ability of llms
through communicative agent based data synthesis,” in Findings of the
Association for Computational Linguistics: NAACL 2025 , 2025.
[17] Xanadu AI, “Qhack 2023 coding challenges.” [Online]. Available:
https://github.com/XanaduAI/qhack 2023 coding challenges
[18] Xanadu AI, “Qhack 2024 coding challenges.” [Online]. Available:
https://github.com/XanaduAI/QHack2024-coding-challenges