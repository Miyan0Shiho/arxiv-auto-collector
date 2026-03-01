# Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs

**Authors**: Guangsheng Ou, Qiming Zhang, Sirong Chen, Anji Li, Dong Xu, Tiancheng Luo, Dekun Dai, Cuiyun Gao, Long Wang, Jun Zhou, Mingwei Liu, Zibin Zheng

**Published**: 2026-02-24 11:36:34

**PDF URL**: [https://arxiv.org/pdf/2602.20799v1](https://arxiv.org/pdf/2602.20799v1)

## Abstract
In the context of newly release software frameworks, large language models (LLMs) often exhibit poor performance and a high rate of hallucination, as they are not exposed to such environments during training. Although inference-time augmentation techniques such as retrieval-augmented generation (RAG) can partially mitigate hallucinations, knowledge injection through prompting alone is insufficient to enable models to fully understand the intrinsic relationships among different components of a codebase, or to reason about the correct compositions and apply. Although explicit knowledge injection can be achieved through post-training, compared with public code domains, unseen codebases typically provide only source code and lack large volumes of high-quality, usage-oriented code that can be directly leveraged as training data. Consequently, existing data synthesis approaches are insufficient to adequately capture unseen codebases usage scenarios when restricted to source code alone. To address these challenges, we propose UCD-Training, a two-stage training framework for reasoning-aware data synthesis grounded in a code graph constructed from unseen codebases. UCD-Training first parses the source code to build a code graph, then conducts dependency-preserving continued pretraining (CPT) using file-level dependency data, followed by graph-grounded supervised fine-tuning (SFT) on three types of synthesized data augmented with explicit reasoning traces: (1) single-hop relation reasoning data, (2) compositional API reasoning data, and (3) codebase utilization data. We further introduce a new benchmark, UnseenCodeBench, for code generation on unseen codebases and conduct comprehensive experiments across multiple codebases.

## Full Text


<!-- PDF content starts -->

Unseen-Codebases-Domain Data Synthesis and Training
Based on Code Graphs
GUANGSHENG OU†#,Sun Yat-sen University, China
QIMING ZHANG†‡,WeChat Pay, Tencent, China
SIRONG CHEN#,WeChat Pay, Tencent, China
ANJI LI,Sun Yat-sen University, China
DONG XU,Sun Yat-sen University, China
TIANCHENG LUO,The Chinese University of Hong Kong, Shenzhen, China
DEKUN DAI#,Sun Yat-sen University, China
CUIYUN GAO,The Chinese University of Hong Kong, China
LONG WANG*,WeChat Pay, Tencent, China
JUN ZHOU,WeChat Pay, Tencent, China
MINGWEI LIU*,Sun Yat-sen University, China
ZIBIN ZHENG,Sun Yat-sen University, China
In the context of newly release software frameworks, large language models (LLMs) often exhibit poor
performance and a high rate of hallucination, as they are not exposed to such environments during training.
Although inference-time augmentation techniques such as retrieval-augmented generation (RAG) can partially
mitigate hallucinations, knowledge injection through prompting alone is insufficient to enable models to
fully understand the intrinsic relationships among different components of a codebase, or to reason about the
correct compositions and apply. Although explicit knowledge injection can be achieved through post-training,
compared with public code domains, unseen codebases typically provide only source code and lack large
volumes of high-quality, usage-oriented code that can be directly leveraged as training data. Consequently,
existing data synthesis approaches are insufficient to adequately capture unseen codebases usage scenarios
when restricted to source code alone.
To address these challenges, we propose UCD-Training, a two-stage training framework for reasoning-aware
data synthesis grounded in a code graph constructed from unseen codebases. UCD-Training first parses the
source code to build a code graph, then conducts dependency-preserving continued pretraining (CPT) using file-
level dependency data, followed by graph-grounded supervised fine-tuning (SFT) on three types of synthesized
data augmented with explicit reasoning traces: (1) single-hop relation reasoning data, (2) compositional API
reasoning data, and (3) codebase utilization data. We further introduce a new benchmark, UnseenCodeBench,
for code generation on unseen codebases and conduct comprehensive experiments across multiple codebases.
Results show that UCD-Training consistently outperforms existing baselines, demonstrating strong generality
across programming languages, model scales, and architectures—for example, achieving a 25.5% absolute
improvement over RAG with GPT5.1. Compared with existing data synthesis methods, UCD-Training yields
#Work done during internship at Tencent.
†Equal contribution.
‡Project leader.
*Corresponding author.
Authors’ Contact Information: Guangsheng Ou†#, Sun Yat-sen University, China, ougsh3@mail2.sysu.edu.cn; Qim-
ing Zhang†‡, WeChat Pay, Tencent, China, qmzhangzz@hotmail.com; Sirong Chen#, WeChat Pay, Tencent, China,
sirongchen49@outlook.com; Anji Li, Sun Yat-sen University, China, lianj8@mail2.sysu.edu.cn; Dong Xu, Sun Yat-sen
University, China, xudong7@mail2.sysu.edu.cn; Tiancheng Luo, The Chinese University of Hong Kong, Shenzhen,
China, 123090401@link.cuhk.edu.cn; Dekun Dai#, Sun Yat-sen University, China, daidk@mail2.sysu.edu.cn; Cuiyun Gao,
The Chinese University of Hong Kong, China, cuiyungao@outlook.com; Long Wang*, WeChat Pay, Tencent, China,
oliverlwang@tencent.com; Jun Zhou, WeChat Pay, Tencent, China, anderszhou@tencent.com; Mingwei Liu*, Sun Yat-sen
University, China, liumw26@mail.sysu.edu.cn; Zibin Zheng, Sun Yat-sen University, China, zhzibin@mail.sysu.edu.cn.
2026.
, Vol. 1, No. 1, Article . Publication date: February 2026.arXiv:2602.20799v1  [cs.SE]  24 Feb 2026

2 Guangsheng Ou, Qiming Zhang et al.
performance gains ranging from 7.2% to 26.1% of Avg. pass@1 on UnseenCodeBench. Moreover, we simulate
realistic enterprise scenarios involving multiple languages and codebases, where a single model needs to
adapt to multiple unseen codebases. In these real-world settings, UCD-Training remains effective, achieving a
pass@1 of 36.0% on UnseenCodeBench.
1 Introduction
Large language models (LLMs) have demonstrated strong capabilities across a wide range of software
engineering tasks, including code generation, completion, translation, and refactoring [ 10,17,22–
24,32]. These successes, however, largely rely on the availability of abundant training data drawn
from mature public code ecosystems (e.g. GitHub) [4, 19].
In practice,developers frequently work with codebases that are absent from LLM pretrain-
ing corpora, such as newly released libraries, domain-specific frameworks, customized
enterprise systems, or proprietary third-party components. We refer to such codebases as
unseen codebases. In this setting, LLMs lack the domain-specific knowledge required to correctly
understand and utilize the codebase [ 54], and consequently exhibit severely degraded performance
and high hallucination rates [ 12]. For instance, prior studies report a performance drop of37–57%
on code generation tasks when models are applied to unseen or private repositories [ 51]. LLMs in
these settings frequently misuse components, generate invalid compositions, or rely on non-existent
functionalities.
A natural first attempt to mitigate this problem is to rely on inference-time augmentation
techniques, such as prompt engineering or retrieval-augmented generation (RAG) [ 26,35,49,55,58],
which provide access to relevant source code during generation. While such approaches can reduce
obvious hallucinations, they do not fundamentally address the underlying issue. Correct code
generation in unseen codebases often requires knowledge of how different components are intended
to be used together—knowledge that reflects latent usage patterns and conventions learned during
training, rather than the source code that can be reliably retrieved alone. As a result, inference-time
techniques are usually insufficient to enable robust generalization to unseen codebases.
These observations suggest thateffectively adapting LLMs to unseen codebases requires
explicit knowledge injection through training such as continue pre-training (CPT) [ 57]
and supervised fine-tuning (SFT) [ 13], rather than relying solely on inference-time aug-
mentation. However, constructing suitable training data for this purpose is inherently challenging.
Unlike public code domains, unseen codebases, typically those newly-released codebases, provide
only source code implementation, while high-quality, usage-oriented examples that demonstrate
correct component composition are scarce or entirely absent. This data scarcity severely limits the
applicability of existing post-training approaches.
Although several data synthesis methods have been proposed to automatically generate training
data for LLMs in code generation tasks [ 28,30,45], they remain insufficient for handling unseen
codebases. For example, approaches such as OSS-Instruct [ 45] typically rely on observable usage
patterns from public repositories, implicitly assuming that there are rich real-world examples
adequately demonstrating how components should be composed. This assumption fails for newly
released or private codebases, where code implementation is often the only available signal.
To address these challenges,we propose UCD-Training(Unseen-Codebase-DomainTraining),
a two-stage training framework, consisting of dependency-preserving CPT data and graph-
grounded SFT data, designed to inject codebase-specific knowledge into LLMs using only
source code implementation from unseen codebases. Our approach first constructs a code
graph within the codebases based on code dependency analysis, and constructs dependency-
preserving CPT data to capture structural and semantic relationships. We perform CPT to help the
model memorize how codebases are implemented. We then synthesize usage-oriented SFT training
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 3
data according to three complementary data types: (1) single-hop relation data, (2) compositional
API data, and (3) codebase utilization data. These data are all augmented with natural reasoning
traces to improve LLMs’ both interpretability and performance.
To systematically evaluate this problem,we also introduce a new benchmark Unseen-
CodeBench, targeting unseen codebases across two widely used programming languages,
C++ and Python. Extensive experiments demonstrate that UCD-Training consistently outperforms
retrieval-based and training-based baselines by 7.2% ot 26.1% on pass@1, generalizes across model
architectures and scales from 7B to 32B parameters, and remains effective in realistic scenarios
where one single model needs to adapt to multiple unseen codebases, achieving a pass@1 of 36.0%
on UnseenCodeBench and still surparssing all baselines by 4.7% to 23.6%. These results indicate
that reasoning-aware data synthesis from code implementation alone can provide a practical and
effective solution for adapting LLMs to unseen codebases.
In summary, this paper makes the following contributions:
•We proposeUCD-Training, a two-stage training framework that injects domain-specific
knowledge into LLMs for unseen codebases by performing reasoning-aware data synthesis
based on a code graph constructed from source code.
•We introduce a benchmark,UnseenCodeBench, for code generation on unseen codebases
across widely used programming languages, providing a standardized evaluation testbed for
future research.
•We conductextensive experimentsto evaluate the effectiveness and generality of UCD-
Training, demonstrating that it consistently improves LLMs’ understanding and application
of unseen codebases across different languages, model scales, and architectures.
2 Approach
To address the challenge that existing LLMs cannot effectively utilize unseen codebases to solve
the requirements, we propose UCD-Training, a two-stage CPT and SFT training framework.
UCD-Training first parses the source code of unseen codebases to build a code graph, and then
constructs code dependency-preserving CPT data and graph-grounded SFT data. SFT data are
synthesized from three views: single-hop relation data, compositional API data, and codebase
utilization data, and augmented with natural reasoning traces. These data types enable the model
to perform repository-grounded reasoning and generation over the unseen codebase. Figure 4
illustrates the overall workflow. The framework operates in the following stages:
(1)Code Graph Construction. We perform program analysis over the source code of the
codebase to extract code entities, including files, classes, functions, methods, and global
variables as nodes, along with their implementations as their properties. We then parse their
dependency, call, include, and other relations as edges, thereby constructing the code graph
of the codebase.
(2)Dependency-Preserving CPT. The CPT stage begins with data construction. We extract
file nodes and their dependency edges from the code graph to obtain a directed acyclic graph
that captures file-level dependencies. Starting from nodes with in-degree zero, we perform
a depth-first traversal over this graph and concatenate the file content according to the
resulting file path order. We then apply a sliding-window truncation strategy to the data
to construct training samples that satisfy training context length constraints. This process
ensures that code files with direct dependency relationships appear within the same training
instance, thereby forming a dependency-preserving source code training corpus. We then
mix this corpus with general-domain data and perform CPT.
, Vol. 1, No. 1, Article . Publication date: February 2026.

4 Guangsheng Ou, Qiming Zhang et al.
(3)Graph-Grounded SFT. The SFT stage also begins with data construction. The synthesized
SFT data consists of three parts: single-hop relation reasoning data, compositional API
reasoning data, and codebase utilization data. The single-hop relation reasoning data are
constructed from each edge in the code graph and its two endpoint nodes. The compositional
API reasoning data are synthesized as multiple types of code examination questions based
on API combinations extracted from the codebase’s internal test cases that cover different
functional scenarios of the project. The codebase utilization data is obtained by transforming
the codebase’s internal test cases into function-generation tasks. We then synthesize the
reasoning traces for these three types of data by providing ground-truth-level context, thereby
completing the data construction for the SFT stage. Finally, the synthetic data are mixed with
general-purpose data to perform SFT training.
These three stages form the core of the UCD-Training methodology. Detailed implementation,
training dataset construction procedures, and descriptive statistics are provided in the following
sections.
2.1 Code Graph Construction
In this stage, we parse the source code of unseen codebases to construct a structured code graph.
By transforming raw source code into a graph-based representation, we effectively capture the
intrinsic relationships among different components of the target codebases, as well as the potential
combinations of APIs in application scenarios. This code graph serves as the foundation for training
data generation in both the dependency-preserving CPT stage and the graph-grounded SFT stage.
Specifically, we perform program analysis to extract code entities, including files, classes, func-
tions, methods, and global variables as nodes, along with their names, types, implementations, and
file locations as properties. We then parse their dependency, call, include, and locate relations as
edges, thereby constructing the code graph of the codebase as shown in Figure 4.
Single Hop 
Relationship
Compilation  
and Repairing
Compositional
APIs
Internal 
Test Cases
File Class Function Global Variable
depends on
contains
contains
containscontains
containsdepends on
contains
invokescontains
invokes invokes invokescontains
Rule -based  
ConstructionLLM -driven  
Augmentation
Test Case 
Transforming
Data Synthesis
Examination Qu -
estions  Generation 
Seed for SFT
Complete Code 
Context ExtractionReasoning Traces Generation
Complete  Content  
of Files
Complete Code 
Context Extraction
Complete 
Dependencies
Ground -Truth -Level  Context
TestingChat Model
Compilation
Post Filtering
Sampling  ResultSingle  Hop
Ground  Truth
Rule -based  
Filtering
Sampling  
ResultMulti -API Refe -
rence  Answer
depends on
depends on
depends on
Source File Dependency GraphDFS Traversal  with
Context Window
Source  Code  
Data  Construction
(1) Code Graph Construction
Source  code of  
Unseen  Codebases
Code Graph
Seed for CPT(2) Dependency -Preserving CPT
CPT Training 
Corpus
General  Data
Base  ModelCPT 
Trained Model UCD -Traing ModelSFT
name
file path
implementation  
codeMeta Data
General  Data
Code  Analysis
SFT Training Corpus
Internal Test Case For 
Codebase Utilization DataCompositional APIs
(3) Graph -Grounded  SFT
Chat Model
Consistency EvaluationConsistency Evaluation
Reasoning 
Traces 
Sampling
Fig. 1. Overview of UCD-Training
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 5
Algorithm 1Training Sample Generation via DFS Traversal
Require:Directed acyclic subgraph𝐺, context window limit𝐿
Ensure:Training samplesS
1:Identify all nodes in𝐺with zero in-degree as start nodes
2:foreach start node𝑣do
3:Perform depth-first traversal to obtain pathsP
4:foreach path𝑝∈Pdo
5:𝑙←1,𝑟←1
6:while𝑟≤|𝑝|do
7:iflength(𝑝[𝑙:𝑟])≤𝐿then
8:𝑟←𝑟+1
9:else
10:Emit𝑝[𝑙:𝑟−1]as a training sample
11:𝑙←𝑟−1,𝑟←𝑙
12:end if
13:end while
14:end for
15:end for
2.2 Dependency-Preserving CPT
2.2.1 Construction of File-level Dependency Data from Code Graph .We extract only file nodes and
inter-file dependency relations from the code graph, resulting in a directed acyclic subgraph, which
will be used to construct the pretraining code corpus. Unlike previous works [ 14], which process
project source code by taking a topological ordering for each connected component in the subgraph
and then splitting it according to the context window limit to obtain the final file-level dependency
data, our approach addresses a key limitation of this strategy. Files that are actually dependent
on each other are not necessarily adjacent in topological order. Therefore, when a topological
sequence is split due to the context window length limit, some files with dependency relations may
not appear in the same training sample, leading to an insufficient description of how these files
collaborate within the same codebase.
To address this issue, we perform a depth-first traversal over the directed acyclic subgraph as
illustrated in Algorithm 1. We start from nodes with zero in-degree, i.e., files that are not depended
on by any other files, and find all depth-first paths in the subgraph. This strategy ensures that any
pair of files with a true dependency relationship appears as adjacent nodes in at least one traversal
path. For each path, we generate training samples using a sliding-window strategy. Specifically, we
fix the left pointer at the beginning of the traversal path and move the right pointer incrementally,
until the total length of all file content within the current sequence exceeds the model’s context
limit. The concatenated content of the files in the current sequence constructs a CPT training
sample. Then, we update the left point location by one step and find the proper right pointer in the
same way. We repeat such a process for all traversal paths and construct the dependency-preserving
CPT data.
This method effectively guarantees that files with actual dependency relations will appear
at least once adjacently in a single training sequence. As a result, the model can more clearly
capture dependency relations among different code elements in the unseen codebase, thereby
improving its understanding of the codebase and its ability to reason about cross-file, repository-
level dependencies.
, Vol. 1, No. 1, Article . Publication date: February 2026.

6 Guangsheng Ou, Qiming Zhang et al.
2.2.2 CPT Data Corpus Composition and Training.We mix the constructed file-level dependency
source code data with general code-relevant data to form the final training corpus for the CPT
stage. This is because continual pre-training that uses only domain-specific data can undermine
models’ original general capability [ 46]. In such a case, even though the model’s understanding of
unseen code codebases improves after CPT, its overall coding ability may actually degrade.
2.3 Graph-Grounded SFT
2.3.1 Construction of Infrastructure and Applications Data with Reasoning Traces from Code Graph.
Each data type undergoes three phases: data synthesis, ground-truth–level contextual reasoning
traces generation, and post-filtering to eliminate low-quality samples, resulting in high-quality
training samples augmented with explicit reasoning traces for understanding and applying unseen
codebases.
(1) Single-hop relation reasoning dataare derived from the code graph to capture fine-grained
structural relations among code entities in an unseen codebase, which are essential for understand-
ing its implementation patterns and architectural organization.
Data Synthesis.Specifically, we synthesize single-hop relation data through a two-stage process
consisting of rule-based construction followed by LLM-driven augmentation. For each single-hop
relation𝐴→𝐵 , we adopt a rule-based approach to map it into a textual representation of the
following form:
[type of𝐴][name of𝐴][relation between A and B][type of𝐵][name of𝐵]
To improve robustness and prevent overfitting to homogeneous positive samples, we further employ
an LLM-based approach to diversify each data instance and generate negative samples. Specifically,
each original instance is expanded into 𝑁1diversified paraphrased samples and 𝑁2negative samples,
where𝑁 1=5and𝑁 2=1.
Reasoning Traces Generation.For positive samples in single-hop relation data, ground-
truth–level context refers to the complete contents of the files in which nodes 𝐴and𝐵are located.
This context explicitly contains the surrounding information of both nodes as well as the associa-
tions between them. For negative samples, ground-truth–level context consists of a list of all code
entity names within the unseen codebases that share the same categories as nodes 𝐴and𝐵, clearly
demonstrating that the fabricated code entity referenced in the query does not exist. We provide
the original instruction together with the constructed contextual information to a powerful LLM
for rejection sampling, then record its outputs and reasoning traces.
Post Filtering.Directly assessing the correctness of a model’s reasoning trace is challenging, as
the underlying reasoning is often lengthy, information-dense, and difficult to evaluate in isolation.
Therefore, we simplify the evaluation by judging the correctness of the model’s output and using it
as a proxy for the correctness of the associated reasoning trace. Specifically, we employ a modern
LLM to determine whether the output produced by the reference model is semantically consistent
with the constructed ground-truth response and filter out the samples judged as inconsistent.
The validated reasoning trace is then embedded into the original data instances, completing the
construction of the single-hop relation reasoning data training set.
(2) Compositional API Reasoning Dataare synthesized as multiple types of code examination
questions based on coherent API combinations extracted from the codebase’s internal test cases,
covering diverse functional scenarios of the project. Rather than randomly composing APIs, which
often results in semantically inconsistent or impractical usage patterns, we leverage internal test
cases as high-quality priors. Since such test cases are written by maintainers to validate concrete
functionalities, the extracted API combinations naturally reflect realistic usage scenarios and
engineering constraints.
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 7
Data Synthesis.To generate high-quality compositional reasoning data, we simulate the process
of human-designed examination tasks and formulate a set of explicit task-design principles to guide
an LLM in task generation. These principles ensure that the synthesized tasks are structurally
well-formed, semantically coherent, and tightly aligned with the target codebase. Specifically, we
constrain the task scope to applications, design principles, and code analysis directly related to
the codebase, preventing overly generic or irrelevant tasks. For each task, the model is required to
output not only the problem statement, but also a reference answer and explicit grading criteria,
forming a complete task–solution–evaluation loop suitable for supervised training. To further
increase information density, we additionally require each task to explicitly list its scoring points
and align them with corresponding knowledge units, encouraging the model to focus on the
codebase’s core concepts and functionalities.
We synthesize three task formats: (1) question–answer tasks targeting conceptual understanding
of the codebase design, (2) fill-in-the-blank tasks for context-aware code completion, and (3)
programming tasks for end-to-end code generation grounded in the codebase. Task difficulty is
explicitly controlled by specifying the number of knowledge points required, providing a concrete
and interpretable difficulty signal beyond subjective labels. For contextual information, we provide
not only the implementation code of the selected API combinations, but also precise location
information and the complete implementations of all required dependencies. This dependency-
closure context enables the model to reason over API interactions holistically, rather than relying
solely on abstract signatures.
Reasoning Traces Generation.Ground-truth–level context for compositional API reasoning
data is defined as a subset of the contextual code information used during data synthesis. Since it is
difficult to determine which specific pieces of context the model relies on, we perform rejection
sampling by providing the problem statement together with the original contextual information
and recording the generated responses and reasoning traces. Samples are retained only when the
responses are judged to be valid under the provided context.
Post Filtering.Compositional API reasoning data employ a two-stage filtering strategy. The first
stage is rule-based and applied to the synthesized task descriptions and reference answers. We
extract code entity names from task descriptions and parse the reference code using tree-sitter to
identify invoked functions and invocation patterns. We then validate (1) whether each function
exists in the provided contextual code or is a language built-in, and (2) whether its invocation
prefix and argument count match the corresponding function signature. This stage filters out
unseen-codebases-domain syntactic errors introduced during task synthesis.
The second stage is applied after reasoning trace generation and combines LLM-based and
rule-based validation. A chat model checks semantic consistency between the reference answer
and the generated response, while rule-based checks again verify unseen-codebases-domain syn-
tactic correctness. This stage further removes samples with functional or reasoning-level errors,
preventing hallucinated behaviors from being propagated to the final training data.
(3) Codebase utilization dataare obtained by transforming the internal test cases of the unseen
codebases. Although such test cases are typically designed to evaluate atomic functionalities,
the API invocation patterns they contain and the implicit functional scenarios they encode can
substantially enhance a model’s understanding of how project APIs are used and the fundamental
application scenarios the project supports.
Data Synthesis.Specifically, as shown in Figure 2, we decompose each internal test case into two
components: (1) the functional implementation that involves codebase’s API invocations and (2) the
assertion statements that verify whether the API behavior matches the expected outcomes. Using
an LLM-based few-shot approach, we perform a test-case decomposition and generate function
comments that describe the intended functionality of each implementation. We then apply an
, Vol. 1, No. 1, Article . Publication date: February 2026.

8 Guangsheng Ou, Qiming Zhang et al.
iterative compile-and-repair procedure to ensure that both the extracted functional code and the
new test cases are syntactically and semantically correct.
Reasoning Traces Generation.Ground-truth-level context of the codebase utilization data is
constructed by collecting all API dependencies involved in the functional implementation and
corresponding information. Specifically, we leverage the code graph to extract functions, data
types, and global variables that have explicit call relations with the original test case. For each
dependency, we retrieve its concrete implementation, file location, and enclosing namespace. This
dependency-complete context enables the model to reason over which APIs should be invoked,
how they interact, and which components are irrelevant to the target functionality. We provide the
instruction together with this ground-truth dependency information to a modern LLM, record the
generated outputs and reasoning traces, and apply rejection sampling
Post Filtering.Similarly, we use the correctness of the generated code as a proxy for the validity
of the associated reasoning trace. Each sampled instance is compiled and executed against the test
suite, and only those that successfully pass both compilation and testing are retained. The validated
reasoning traces are then embedded into the corresponding instances, yielding the final codebase
utilization dataset.
2.3.2 SFT Data Corpus Composition and Training.We mix the constructed data with general code
relevant data to form the final training corpus for the SFT stage to maintain its general capability
as CPT stage.
3 Benchmark
Unseen code domains are pervasive in industrial settings. Most enterprises either adapt existing
open-source frameworks to meet internal business requirements or develop proprietary third-party
libraries to improve development efficiency and security. Although a variety of benchmarks have
been introduced to evaluate LLMs across different software engineering dimensions, the majority
of these efforts focus primarily on public code scenarios. While prior research has established
benchmarks for private codebases [ 43,51], it typically concentrates on a single codebase in a
single programming language, failing to capture realistic enterprise environments that often
involve multiple codebases across multiple programming languages simultaneously. Therefore,we
construct an unseen codebases evaluation benchmark, UnseenCodeBench, to assess the
effectiveness of UCD-Training.
Construction.We choose C++ and Python as our target languages because of their wide adoption
and popularity [ 40]. In particular, C++ is widely used for building system-level infrastructure
[36], while Python is extensively used for developing application software [ 34]. We construct the
evaluation benchmark with LLM assistance followed by manual double-checking. Inspired by
previous work [ 51,59] that evaluates a model’s understanding of a specific codebase by testing
its ability to use the codebase’s APIs, we first analyze each target codebase to extract all publicly
TEST(mysql, test_aggregations_with_nullable) {
  ……
  const auto get_children =
      select_from <Person>(
          avg("age"_c).as<"avg_age">(), count().as<"num_children">(),
          max("age"_c).as<"max_age">(), min("age"_c).as<"min_age">(),
          sum("age"_c).as<"sum_age">(),
          count_distinct("last_name"_c).as<"num_last_names">()) |
      where("age"_c < 18 or "age"_c.is_null()) | to<Children>;
……
  EXPECT_EQ(children.num_children, 3);
  EXPECT_EQ(children.num_last_names, 1);
  …...
}functional implementation that
involves codebase's API invocations
assertion statementsOriginal Internal Test Case
Codebase Utilization Data
New Test Cases
Function 
Implementation
File
Fig. 2. Example for Generating Codebase Utilization Data
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 9
Table 1. Details of Selected Projects for Benchmark Construction.PL: Programming Language, FLD: File-level
Dependencies, D.: Dependencies
PL Project Name Created Time Stars #LOC Number of Files Avg. FLD of File Avg. D. of Function Number of Test Cases
C++sqlgen[11] 2025.3 131 14,164 213 3.5 5.9 60
reaction[27] 2025.3 618 5,720 26 2.7 4.0 64
Hexi[8] 2025.3 279 6,108 29 3.1 2.0 64
python LEANN[50] 2025.6 8,038 7,491 14 1.6 5.1 64
available functions, global variables, and custom data types (e.g., classes). We then manually select
combinations of these elements that can jointly implement certain functionalities provided by the
codebase, and design corresponding target task scenarios. These are fed into an LLM to generate
an evaluation instance consisting of a task description, a reference implementation, and test cases.
Subsequently, we manually check and refine the generated data along several dimensions: the
rationality of the task setup, the correctness of the functional description, the executability of the
reference implementation and its corresponding test cases, and the consistency between the code
and the tests. After this first pass, another author repeats the same quality-checking process to
confirm the quality of the evaluation data by double-checking. The entire construction process
is carried out by five engineers, each with 3 to 8 years of programming experience, and in total
consumes 100 person-hours.
Quality Control Mechanism.To ensure that the selected target codebases have not appeared
in the training corpora of the evaluated models, thus achieve the effect of truly unseen codebases
evaluation, we adopt a two-stage verification process. In the first stage, we ensure that the creation
time of each selected project is later than the knowledge cutoff of the models. Based on the
knowledge cutoff dates of all the models we evaluate, we take their union and obtain a time threshold
𝑇, which we currently set to 2025-03-01 . By querying GitHub with created:>=2025-03-01 , we
crawl all projects that satisfy this constraint, and then apply an initial filtering mechanism with
stars:>100 andsize:>1000 to eliminate projects that are too small in scale, have low recognition,
or lack sufficient activity as previous works[ 15,33], thereby ensuring the quality of the selected
projects. Finally, we manually review and select the final target codebases, covering a diverse range
of domains, including database systems, binary data stream processing, programming frameworks,
and high-performance retrieval engines. Detailed information of selected codebases is shown in
Table 1. In the second stage, we evaluate the performance of the base models on our constructed
evaluation set. After constructing the evaluation benchmark, we measure the performance of the
base models without any external knowledge augmentation. This allows us to assess how well
existing models master unseen-codebases-domain knowledge before any unseen-codebases-domain
post-training. As shown in Table 3, all selected SOTA models with only base model exhibit a dramatic
performance drop on our constructed evaluation set compared to existing public benchmarks, which
further indicates that the chosen codebases did not appear in the training corpora of these models.
This demonstrates that our constructed evaluation benchmark can effectively measure models’ real
performance in unseen-codebases-domain scenarios.
4 Experiment Setup
We evaluate the effectiveness, usefulness, and generalization of UCD-Training by answering the
following seven research questions (RQs):
•RQ1 (Performance of UCD-Training): How does UCD-Training perform compared to other
methods?
•RQ2 (Ablation Study): How does each component of UCD-Training contribute to the final
performance (including different categories of training data and filtering mechanisms)?
, Vol. 1, No. 1, Article . Publication date: February 2026.

10 Guangsheng Ou, Qiming Zhang et al.
•RQ3 (Generality of UCD-Training): How generalizable is UCD-Training across different
programming languages, model sizes, and model families?
•RQ4 (Application in Real-World Enterprise Scenarios): How does UCD-Training perform in
simulated real-world enterprise scenarios, i.e., can a single model simultaneously handle multiple
unseen codebases across multiple languages?
•RQ5 (Configuration): How do different configurations of hyperparameters in UCD-Training
affect the results?
4.1 Baseline
We evaluate the performance of UCD-Training with two types of methods: (1) Explicit knowledge
injection through training: OSS-Instruct [ 45] and Cotton [ 48], which fine-tune the model with
synthetic data to enhance its coding capabilities; (2) Inference-time augmentation: RAG [ 55], which
uses retrieval to mitigate the model’s hallucination phenomenon.
OSS-Instruct[ 45] is a proven and influential program synthesis approach, distinguished by its
effectiveness in leveraging synthetic data to fine-tune small-parameter models. It serves as a mature,
robust baseline in code generation and is widely adopted in the field of code data synthesis. We
compare this method with our proposed data synthesis approach to demonstrate the effectiveness
of our data synthesis pipeline.
COTTON[ 48] is a Chain-of-Thought (CoT) based program synthesis approach specifically designed
for lightweight language models. It automatically generates high-quality CoTs without relying on
manually crafted rationales to fine-tune models so as to improve the LLMs’ reasoning capability,
thereby enhancing their code generation ability. We compare the CoT synthesis method in COTTON
with our proposed reasoning-process synthesis method to validate the effectiveness of our approach
for generating reasoning traces.
RAG[ 55] retrieves code snippets from the project that are similar to the current task and incor-
porates them into the input prompt to help LLMs better understand the requirements and gain
awareness of specific factual knowledge and project contexts.
4.2 Studied LLMs
As shown in Table 2, For the RAG-based baselines, we select several state-of-the-art models, covering
both general-purpose LLMs and code-oriented LLMs, as well as open-source and closed-source
models. We also ensure that the knowledge cutoff of each chosen model is earlier than the creation
time of the projects used to build our evaluation benchmark. Although DeepSeek-V3.1-Terminus
and Qwen3-Coder-480B do not publicly disclose their exact knowledge cutoff dates, based on LLMs’
training timelines, the association between release time and knowledge cutoff of other models, and
their base-model performance on our benchmark, we reasonably infer that the selected projects
are not included in the training corpora of these two models either.
For UCD-Training and the training-based baselines, following previous work [ 29,37], we adopt
four models from the Qwen family, which are widely used in post-training scenarios: Qwen3-8B-
Base, Qwen3-14B-Base, Qwen2.5-Coder-14B-Base, and Qwen2.5-Coder-32B-Base. This selection
covers multiple parameter scales and different model types, enabling a comprehensive evaluation
of our method under diverse model capacities and architectures.
4.3 Metrics
Following the previous methods [ 4,41], we employ the execution-based metricsCompilation@k
andPass@kto assess the quality of generated code through the execution status of the code.
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 11
Table 2. Studied LLMs
Model Type Model Name Open-source Reasoning Time Knowledge Cutoff Size
Models for RAG
General LLMClaude-Sonnet-4.5 [1] ✗✓2025.9 2025.1 -
GPT-5.1 [31] ✗✓2025.11 2024.9 -
Gemini-3.0-Pro [6] ✗✓2025.11 2025.1 -
DeepSeek-V3.1-Terminus [7] ✓ ✓2025.9 - 671B
Code LLM Qwen3-Coder-480B [38] ✓✗2025.7 - 480B
Models for Training
General LLMQwen3-8B-Base [38] ✓✗2025.4 - 8B
Qwen3-14B-Base [38] ✓✗2025.4 - 14B
Code LLMQwen2.5-Coder-14B-Base [18] ✓✗2024.11 - 14B
Qwen2.5-Coder-32B-Base [18] ✓✗2024.11 - 32B
•Compilation@k: Compilation@k is a crucial evaluation metric that measures the proportion
of translated code snippets that compiles successfully without errors within 𝑘rounds, directly
reflecting the syntactic and structural correctness of the translated code. 𝐶(𝑖,𝑘)= 1if the𝑖𝑡ℎ
code sample compiles successfully within𝑘attempts; otherwise𝐶(𝑖,𝑘)=0.
•Pass@k: This widely-used metric [ 4] calculates the percentage of tasks correctly solved based
on𝑘generated code samples per task. A task is considered solved if at least one of the generated
code samples passes all the corresponding test cases.
4.4 Implementation Details
Code Parsing.During the code graph construction stage, C++ codebases are parsed and compiled
using Clang [ 25], while Python codebases are statically analyzed using Jedi [ 5] and the built-in
AST module. For rule-based post-filtering of compositional API usage reasoning data, we adopt
Tree-sitter [39] to perform fine-grained syntactic analysis.
Data Construction.For the dependency-preserving CPT stage, we incorporate general-domain
data from Mixture-of-Thought [ 9], retaining only categories relevant to code-related capabilities,
including math,code, and scientist reasoning. For the graph-grounded SFT stage, we use AM-
distilled-data[ 56] as general-domain supervision, which includes not only code-related data but
also instruction-following data. The modern model used during graph-grounded SFT is DeepSeek-
v3.1-Terminus[ 7] in reasoning mode, while the model used for consistency evaluation is DeepSeek-
v3.1-Terminus in chat mode. Due to variations in programming languages and repository scales,
the sizes of the synthesized datasets differ across codebases: 35k samples for sqlgen, 20k samples
for Reaction and Hexi, and 6k samples for Leann. For a fair comparison, training-based baselines
are trained using the same data scales.
Training Setup.Due to variations in programming languages and codebases scales, the number of
epochs used in the dependency-preserving CPT stage differs across codebases. For C++ codebases,
the CPT stage is conducted for 1 epoch on sqlgen and for 2 epochs on reaction and hexi. For
the Python codebase leann, CPT is performed for 1 epoch. In the graph-grounded SFT stage, all
codebases are trained for 3 epochs. For other hyperparameters, both CPT and SFT stages employ
full-parameter fine-tuning with a learning rate of5 .0𝑒−5, a warmup ratio of 0.1, a context window
length of 32,768 tokens, and a gradient accumulation step of 1. All experiments are conducted on
32 NVIDIA H20 GPUs. The complete configuration and training details are available on GitHub.
5 Evaluation
5.1 RQ1: Performance of UCD-Training
Design.To systematically evaluate the effectiveness of UCD-Training in unseen codebase scenarios,
we compare it against two representative categories of methods: inference-time augmentation
, Vol. 1, No. 1, Article . Publication date: February 2026.

12 Guangsheng Ou, Qiming Zhang et al.
Table 3. Comparison of Ours and Baselines
Tech. Base Modelsqlgen reaction Hexi LEANN UnseenCodeBench
compilation@1 pass@1 compilation@1 pass@1 compilation@1 pass@1 pass@1 Avg. pass@1
Base ModelClaude-Sonnet-4.5 1.8% 1.0% 3.8% 0.5% 16.1% 14.1% 20.8% 9.1%
GPT-5.1 0.0% 0.0% 2.8% 1.9% 11.9% 11.1% 10.6% 5.9%
Gemini-3.0-Pro 3.3% 1.7% 3.0% 0% 20.2% 17.0% 4.7% 5.9%
DeepSeek-V3.1-Terminus 0.0% 0.0% 3.3% 2.0% 16.3% 14.5% 10.3% 6.7%
Qwen3-Coder-480B 1.3% 0.7% 3.1% 1.6% 20.5% 16.7% 12.7% 7.9%
RAG [55]Claude-Sonnet-4.5 11.2% 9.5% 11.6% 2.2% 44.2% 39.4% 34.5% 21.4%
GPT-5.1 4.2% 4.2% 14.7% 5.5% 31.6% 28.8% 13.4% 13.0%
Gemini-3.0-Pro 8.3% 6.5% 11.3% 4.5% 43.6% 38.9% 9.1% 14.8%
DeepSeek-V3.1-Terminus 4.5% 4.2% 13.4% 4.2% 29.8% 25.8% 18.8% 13.3%
Qwen3-Coder-480B 7.8% 7.7% 5.9% 1.7% 23.6% 20.0% 14.4% 11.0%
OSSInstruct [45]Qwen3-8B-Base 1.2% 0.0% 5.8% 2.5% 21.3% 18.3% 17.0% 9.5%
Qwen3-14B-Base 0.7% 0.2% 2.2% 1.7% 24.5% 20.3% 27.5% 12.4%
COTTON [48]Qwen3-8B-Base 18.0% 11.2% 30.2% 17.0% 43.9% 35.0% 42.7% 26.5%
Qwen3-14B-Base 20.5% 16.2% 33.3% 18.8% 44.8% 39.1% 50.9% 31.3%
UCD-TrainingQwen3-8B-Base 22.8% 19.1% 34.5% 20.0% 45.8% 38.8% 47.8% 31.5%
Qwen3-14B-Base 29.3% 22.2% 42.0% 29.1% 54.2% 47.2% 55.2% 38.5%
Qwen2.5-Coder-14B-Base 23.3% 18.2% 48.8% 27.3% 52.5% 45.3% 48.1% 34.7%
UCD-Training
(combined)Qwen3-14B-Base 23.3% 19.8% 44.4% 27.2% 54.7% 46.1% 50.9% 36.0%
and training-time augmentation. For inference-time augmentation, we select Retrieval-Augmented
Generation (RAG) as a representative baseline and five state-of-the-art models as shown in Table 2
as base model. For training-time augmentation, we consider two widely adopted data synthesis
and fine-tuning approaches: OSS-Instruct and COTTON and select the same base model as UCD-
Training: Qwen3-8B-Base and Qwen3-14B-Base We adopt compilation@1 and pass@1 for C++
codebases and pass@1 for Python codebase LEANN, to assess code generation performance from
both syntactic correctness and semantic correctness perspectives in unseen codebases domain.
Result.Table 3 presents the overall performance comparison between UCD-Training and baseline
methods across different unseen codebases. The results reveal several notable findings.
First, compared with inference-time augmentation methods,models trained with UCD-
Training significantly outperform RAG-based state-of-the-art baselines on UnseenCodeBench, even
at the 8B parameter scale. Specifically, our 8B model achieves a pass@1 score of 31.5%, exceeding
the best-performing RAG baseline (Claude-Sonnet-4.5) by 10.1%. While RAG partially alleviates the
difficulty faced by LLMs without domain-specific adaptation in understanding and using unseen
codebases, the resulting improvements remain limited. In particular, as codebases grow in scale
and structural complexity, the effectiveness of RAG diminishes substantially. On the relatively
simple Hexi codebase, the performance of an 8B model trained with UCD-Training is comparable to
that of the best RAG-based model. However, on the most complex and largest codebase, sqlgen, in
UnseenCodeBench, the gap becomes pronounced. The best RAG-based model is Claude-Sonnet-4.5,
achieving only 17.2% compilation@1 and 9.5% pass@1, while under UCD-Training, even an 8B model
reaches 22.8% compilation@1 and 19.1% pass@1. These results indicate that merely retrieving and
injecting local code snippets is insufficient for enabling models to capture implicit inter-component
sqlgen::read<std::vector<Event>> |       
sqlgen::order_by("created_at", sqlgen:: desc)|
sqlgen::limit(n)).value();
Reasoning Traces:
…
For example, order_by ("created _at", 
sqlgen ::desc ) and then limit(n).
…
ERROR: ‘desc’ is not a member of ‘ sqlgen ’Baseline1: OSS -Instruct
// without using namespace sqlgen::literals;
auto query = sqlgen::read<std::vector<Event>>| 
sqlgen::order_by("created_at"_c.desc())  | sqlgen::limit(n);
Reasoning Traces:
…
Step 6. **Use column literals** by employing the `_c` suffix on 
column names (e.g., `" created_at"_c `) to reference database 
columns in a type -safe manner.
…
ERROR: unable to find string literal operator ‘ operator””_c ’Baseline2: COTTON
using namespace sqlgen::literals;
auto query = sqlgen::read<std::vector<Event>> | 
sqlgen::order_by("created_at"_c. desc() ) | 
sqlgen::limit(n);
Reasoning Traces:
…
We need to use the `sqlgen::literals` to get the 
column literal for `created_at` . We can use 
`"created_at"_c` to get the column.
…Ours: UCD -Training
Fig. 3. A Comparative Example Illustrating the Outputs on the Same Problem Using Baselines and UCD-
Training. The example is the task with ID limit_recent_events from the sqlgen in UnseenCodeBench. The
complete model outputs can be found on GitHub.
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 13
dependencies and usage conventions, highlighting the necessity of post-training–based knowledge
injection for unseen codebases.
Second, compared with other explicit knowledge injection through training methods,UCD-
Training consistently yields superior performance under the same model scale across all codebases.
It is worth noting that on larger and more complex codebases, the performance gains brought by
UCD-Training become even more pronounced. In such settings, models trained with UCD-Training
at 8B parameters even outperform 14B-parameter models trained using baseline methods. This
advantage stems from two key factors: (1) UCD-Training synthesizes richer architectural and usage-
oriented data grounded in unseen codebases, substantially enhancing the model’s understanding
and application of unseen codebases; and (2) the reasoning traces generated by our method are
more realistic and natural than generic Chain-of-Thought data, providing models with stronger
reasoning capabilities and enabling them to solve problems effectively in more complex scenarios.
We further compare UCD-Training and existing training-based data synthesis methods through
a concrete case study. As shown in Figure 3, UCD-Training achieves a pass@1 score of 90% on
this task, while all other baselines fail completely with 0%. OSS-Instruct struggles in this setting
because it relies on pre-existing usage-oriented code and cannot adequately cover realistic usage
scenarios when only source code is available, leading to hallucinated or invalid API invocations.
Although COTTON correctly identifies all required dependencies, the synthesized CoT traces often
diverge from the model’s intrinsic reasoning patterns, resulting that models trained on such traces
suffer from degraded reasoning robustness and exhibit poor performance in complex scenarios. In
contrast, UCD-Training not only accurately identifies all necessary dependencies but also correctly
infers and applies the appropriate usage patterns for each dependency, enabling reliable end-to-end
code generation.
Finding 1:UCD-Training effectively injects repository-level structural knowledge and reason-
ing processes during training, achieving a peak performance of 38.5% on UnseenCodeBench.
This result surpasses the best-performing RAG baseline by 10.1% and the strongest training-
based baseline by 7.2%.
5.2 RQ2: Ablation Study
Design.To gain a deeper understanding of how each key design component of UCD-Training
contributes to the overall performance, we conduct a systematic ablation study along two dimen-
sions: training data sources, and data quality control mechanisms. All ablation experiments are
performed using the Qwen3-14B-base model and evaluated across all unseen codebases and metrics
with comprehensive experiments, ensuring fair and comparable results.
Specifically, we design our ablations from the following perspectives:(1) Ablation on SFT
Data Types.To verify the complementarity of the three types of graph-grounded SFT data, we
selectively remove each data type:w/o codebase utilization data, removing codebase utilization data
to assess the importance of realistic usage patterns and end-to-end generation supervision;w/o
single-hop realtion reasoning data, removing single-hop relation reasoning data to evaluate the role
of fine-grained structural modeling and local dependency learning;w/o compositional API reasoning
data, removing compositional API reasoning data to measure the contribution of synthesized
unseen-codebase API usage data in the SFT stage.(2) Ablation on General-Domain Data in SFT.
We further introduce awith only general datasetting, where all unseen-codebase–specific data are
removed. This setup aims to demonstrate that general-domain data mainly serves to preserve the
model’s generic code capabilities, while providing little benefit to domain-specific understanding
and usage in unseen codebases.(3) Ablation on Data Filtering Mechanisms.To analyze the
impact of our two-stage filtering strategy on synthesized data quality and downstream performance,
, Vol. 1, No. 1, Article . Publication date: February 2026.

14 Guangsheng Ou, Qiming Zhang et al.
Table 4. Ablation Study on base model Qwen3-14B-Base. CUD: codebase utilization data, SHRRD: single-hop
relation reasoning data, CARD: compositional API reasoning data
Tech.sqlgen reaction Hexi LEANN
compilation@1 pass@1 compilation@1 pass@1 compilation@1 pass@1 pass@1
UCD-Training 29.3% 22.2% 42.0% 29.1% 54.2% 47.2% 55.2%
w/o CUD 23.8% (↓5.5) 21.7% (↓0.5) 33.1% (↓8.9) 21.6% (↓7.5) 53.4% (↓0.8) 46.1% (↓1.1) 49.8% (↓5.4)
w/o SHRRD 27.5% (↓1.8) 21.3% (↓0.9) 33.0% (↓9.0) 19.8% (↓9.3) 50.6% (↓3.6) 44.2% (↓3.0) 54.6% (↓0.6)
w/o CARD 12.7% (↓16.6) 9.8% (↓12.4) 37.8% (↓4.2) 24.2% (↓4.9) 34.4% (↓19.8) 30.8% (↓16.4) 25.5% (↓29.7)
with only general data 0.5% (↓28.8) 0.5% (↓21.7) 0.5% (↓41.5) 0.3% (↓28.8) 28.8% (↓25.4) 24.4% (↓22.8) 10.5% (↓44.7)
w/o filter of problem and reference
answers of CARD18.2% (↓11.1) 15.2% (↓7.0) 40.0% (↓2.0) 27.2% (↓1.9) 52.5% (↓1.7) 44.7% (↓2.5) 39.1% (↓16.1)
w/o filter of reasoning
content of CARD25.3% (↓4.0) 19.0% (↓3.2) 42.0% (+0.0) 28.0% (↓1.1) 55.9% (↑1.7) 46.7% (↓0.5) 50.2% (↓5.0)
we independently remove:w/o filter of problem and reference answers, which eliminates filtering at
the question–answer pair level andw/o filter of reasoning content, which removes filtering applied
to reasoning traces and final responses.
Result.Table 4 reports the performance of the full UCD-Training and its ablated variants across
different tasks. The results clearly demonstrate the critical contributions of each component.
First, for ablation on SFT data types, the result shows that all three types of SFT data are
indispensable for improving performance on unseen codebases. Removing any single data type
consistently leads to noticeable degradation, indicating that they provide complementary benefits
for understanding and utilizing unseen codebases. Among them, compositional API reasoning data
is the most critical: its removal causes a dramatic performance drop—for example, on LEANN,
pass@1 decreases from 55.2% to 25.5%, a decline of 29.7%. This result underscores a key strength
of our approach: by automatically synthesizing multi-API usage tasks, we effectively mitigate the
scarcity of high-quality, usage-oriented code beyond implementation-level source code, substantially
enhancing the model’s ability to solve tasks on the target codebase.
Second, for ablation on general-domain data in SFT, general-domain data mainly serves to
preserve the model’s generic capabilities and contributes little to domain-specific generalization
on its own. Under the with only general data setting, the model achieves near-zero pass@1 on
sqlgen and reaction with 0.5% and 0.3% respectively, and only 10.5% on LEANN. These results
closely mirror the behavior of base models in RQ1, further confirming that without explicit private-
domain knowledge injection, LLMs struggle to generalize to unseen codebases. Notably, the model
performs better on Hexi than all base models reported in RQ1, even under this setting. This
observation indicates that dependency-preserving CPT alone already injects meaningful private-
domain knowledge, enabling the model to acquire a non-trivial understanding of the unseen
codebase structure, despite the absence of graph-grounded SFT data.
Finally, for ablation on data filtering mechanisms, the strict data filtering mechanism plays
a crucial role in ensuring stable performance improvements. Removing any stage of the filtering
pipeline leads to consistent performance degradation, indicating that each filtering stage contributes
non-trivially to data quality. Moreover, compared with removing the filtering of reasoning content,
removing the filtering of problem statements and reference answers results in a substantially
larger performance drop. It is also worth noting that removing problem–answer filtering leads
to substantial performance degradation, with pass@1 drops of 7.0% and 16.1% on sqlgen and
LEANN, respectively—significantly larger than those observed on reaction (1.9%) and Hexi (2.5%).
This discrepancy stems from the larger scale and higher structural complexity of these unseen
codebases. Without post-training on the target codebase, the modern model is more likely to
generate problems and reference answers that misuse or misinterpret repository-specific APIs
and design assumptions. When filtering is removed, such hallucinations are propagated into the
synthesized data, ultimately impairing the model’s understanding and utilization of the target
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 15
codebase. Consequently, post-filtering becomes increasingly critical as codebase complexity grows.
Finding 2:Each component of UCD-Training contributes positively to the performance of
model. Among them, the compositional API reasoning data in the SFT stage has the largest
impact, with a performance drop of up to 29.7% in w/o CARD setting.
5.3 RQ3: Generality of UCD-Training
Design.To demonstrate the robustness and generality of UCD-Training across diverse settings, we
evaluate its performance under variations along three key dimensions: programming languages,
base model types, and model scales. First, to assess language generalization, we conduct experiments
on unseen codebases written in two widely used programming languages: Python and C++. Second,
to examine model-type robustness, we evaluate both general LLMs and code LLMs as base models.
Third, to analyze scalability with respect to model size, we consider models with 8B, 14B, and 32B
parameters. For the 32B setting, we adopt Qwen2.5-Coder-32B-Base as the base model, since Qwen3
does not currently provide a released base model at this scale. Due to computational resource
constraints, experiments with 32B-scale models are performed on one representative codebase per
programming language.
Result.The experimental results demonstrate that UCD-Training can be applied effectively in
diverse settings. As shown in Table 3, UCD-Training achieves consistent and significant improve-
ments on both C++ and Python codebases, indicating its robustness across programming languages.
Moreover, Table 4 shows that, in the Python setting, each key design component of UCD-Training
continues to contribute positively to enhancing the model’s understanding and usage of unseen
codebases, further validating the necessity of our overall design. In addition, UCD-Training is
effective for both general LLMs and code LLMs. As illustrated in Table 3, Qwen2.5-Coder-14B-Base
consistently outperforms all baseline methods across all evaluated codebases, demonstrating that
our method is model-agnostic and does not rely on a specific class of base models. Importantly,
the effectiveness of UCD-Training scales favorably with model size. As shown in Figure 4, when
increasing the model size from 8B to 14B parameters, pass@1 improves from 20.0% to 29.1% on one
codebase and from 47.8% to 55.2% on another. When further scaling from 14B to 32B parameters,
performance continues to increase, from 27.3% to 33.8% and from 48.1% to 58.8%, respectively. These
results indicate the practical potential of UCD-Training in real-world scenarios with larger base
models when given sufficient computational resources.
Finding 3:UCD-Training consistently enhances models’ understanding and application of
target codebases across different programming languages, model types, and model sizes. More-
over, its strong scaling behavior highlights its practical potential for deployment in real-world,
industrial-scale software development scenarios.
5.4 RQ4: Application in Real-World Enterprise Scenarios
Design.Section 5.3 demonstrates the potential of UCD-Training for deployment in realistic settings.
To further evaluate its effectiveness in real-world enterprise scenarios, where models are typically
required to handle multiple codebases across multiple programming languages simultaneously,
we conduct an additional experiment in which knowledge from all codebases included in Unseen-
CodeBench is injected into a single model. We then evaluate this unified model on each individual
codebase to assess its ability to retain and apply knowledge across heterogeneous codebases. For
consistency with previous experiments, we adopt Qwen3-14B-Base as the base model in this setting.
, Vol. 1, No. 1, Article . Publication date: February 2026.

16 Guangsheng Ou, Qiming Zhang et al.
Result.As shown in Table 3, although the performance of UCD-Training under combined training
exhibits a slight degradation compared to separate training, the performance drop remains modest,
ranging from only 1.1% to 4.3% across different codebases. This behavior is expected and can be
attributed to the well-known trade-off and interference effects that commonly arise when training
on mixed-domain data.
Despite this minor degradation, models trained with UCD-Training under the combined setting
consistently outperform all baseline methods across every target codebase, achieving an average
improvement of 4.7% to 23.6% in pass@1. This demonstrates that UCD-Training is robust to cross-
codebases and cross-language knowledge integration, and does not rely on isolated, single-codebase
adaptation.
Finding 4:UCD-Training remains effective in real-world industrial scenarios involving multi-
ple programming languages and multiple codebases, achieving a pass@1 of 36.0% on Unseen-
CodeBench and surparssing all baselines by 4.7% to 23.6%
5.5 RQ5: Configuration
Design.We conduct comprehensive experiments to investigate how different configurations under
UCD-Training affect model performance. Specifically, we explore: (1) the impact of varying the
number of CPT epochs under a fixed SFT configuration; (2) the impact of varying the number of
SFT epochs under a fixed CPT configuration; and (3) the effect of different scales of compositional
API reasoning data, which contributes most significantly to the model’s performance on unseen
codebases.
Due to computational constraints, we select one representative C++ codebase and one Python
codebase for these experiments. Since the overall scale of compositional API reasoning data varies
across codebases, the staged data sizes used in the experiments are not strictly identical. We adopt
Qwen3-14B-Base as the base model for all settings.
Result.As shown in Figures 5a and 5b, the performance of model exhibits a consistent upward trend
as the volume of the compositional API reasoning data increases on both codebases, demonstrating
the scalability of our proposed data synthesis method. It’s important to note that even with the
smallest amount of injected knowledge, performance on the Hexi and LEANN codebases improves
substantially, from 30.8% to 41.6% and from 25.5% to 40.6%, respectively. This highlights the high
knowledge density of the compositional API reasoning data.
Figures 6a and 6b further show that the optimal number of epochs for the CPT stage is 1.
Increasing the number of CPT epochs beyond this point leads to gradual overfitting. A similar trend
is observed in the SFT stage as shown in Figures 7a and 7b, although the optimal epoch setting
differs slightly between CPT and SFT.
Qwen3-8B-Base Qwen3-14B-Base
Qwen2.5-Coder-14B-Base Qwen2.5-Coder-32B-Base01020304050Accuracy (%)34.542.048.350.6
20.029.127.333.8compile@1
pass@1
a reaction
Qwen3-8B-Base Qwen3-14B-Base
Qwen2.5-Coder-14B-Base Qwen2.5-Coder-32B-Base0102030405060Accuracy (%)47.855.2
48.158.8
pass@1 b leann
Fig. 4. Generality of Different Models. 4a shows performance on reaction and 4b on leann.
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 17
The discrepancy between the optimal epoch numbers for CPT and SFT can be attributed to the
nature of the training data. While CPT focuses primarily on learning structural and dependency-
level information from source code, SFT involves application-oriented data, which requires more
training iterations for the model to internalize complex usage patterns and implicit functional
relationships.
Finding 5:Our data synthesis approach demonstrates strong scalability, with model perfor-
mance improving consistently as the data scale increases. Moreover, compared to source-code-
only data, application-level data requires more training epochs to effectively capture its richer
relational structure and more complex usage semantics.
6 Threats To Validity
Threats in Language Coverage.Our evaluation focuses on only two programming languages,
C++ and Python, and does not include other languages. This limited language coverage may pose a
threat to the generalizability of our findings. To further examine the applicability of UCD-Training,
we plan to extend our experiments to a broader range of programming languages and development
environments in future work, thereby more thoroughly validating the robustness and generalization
capability of our method.
Threats in Training Configuration.Due to computational resource constraints, we were unable
to exhaustively explore all possible training configurations during the training process. As a result,
the configurations adopted in our experiments may not represent the optimal settings. In future
work, we plan to investigate optimal training configurations for unseen codebase domains through
both theoretical analysis and more extensive empirical exploration.
0 5,000 10,000 15,000 20,000
Scales of Synthetic Data303540455055Accuracy (%)
34.447.748.852.854.2
30.841.642.745.547.2compile@1
pass@1
a Scale-Hexi
0 2,000 4,000 6,000
Scales of Synthetic Data25303540455055Accuracy (%)
25.540.648.155.2pass@1
b Scale-leann
Fig. 5. Model performance un-
der different synthetic data scales.
Base model is Qwen3-14B-Base. 5a
is model’s performance under dif-
ferent scales of synthetic data on
reaction, 5b is model’s performance
under different scales of synthetic
data on leann.
1 2 3
Epochs of CPT182022242628Accuracy (%)
29.3
24.0
23.0
22.2
21.3
18.3compile@1
pass@1a CPT-sqlgen
1 2 3
Epochs of CPT4648505254Accuracy (%)
55.2
46.447.6pass@1
b CPT-leann
Fig. 6. Model performance under
different CPT epochs. Base model
is Qwen3-14B-Base. 6a is Different
Model’s Performance on sqlgen, 6b
is Different Model’s Performance
on leann.
1 2 3 4
Epochs of SFT141618202224262830Accuracy (%)
 18.527.029.3
25.7
13.721.322.2
20.7compile@1
pass@1a SFT-sqlgen
1 2 3 4
Epochs of SFT4849505152535455Accuracy (%)
47.751.955.255.0 pass@1
b SFT-leann
Fig. 7. Model performance under
different SFT epochs. Base model
is Qwen3-14B-Base. 7a is Different
model’s Performance on sqlgen, 7b
is Different model’s Performance
on leann
, Vol. 1, No. 1, Article . Publication date: February 2026.

18 Guangsheng Ou, Qiming Zhang et al.
7 Related Work
7.1 Data Synthesis for Code LLMs
Data synthesis techniques aim to automatically construct high-quality instruction–response pairs.
Early methods relied on template-based prompting to generate synthetic training data [ 53], among
which Self-Instruct [ 42] is a seminal approach. It iteratively expands an instruction set by boot-
strapping from a small number of human-written seeds using an LLM’s own generation capability.
This paradigm was later extended to the code domain by Code Alpaca [ 3], which used GPT-3.5 to
produce 20K instruction–code pairs. Building on this idea, Evol-Instruct [ 28] proposed instruction
evolution to increase complexity and diversity via in-depth and in-breadth transformations. More
recently, OSS-Instruct [ 45] leveraged randomly sampled open-source code snippets as inspiration,
prompting LLMs to synthesize diverse programming tasks.
However, most existing data synthesis techniques rely on pre-existing usage-oriented code from
the target codebase and are ineffective when only source code is available. To address this limitation,
UCD-Training starts directly from source code, performs systematic code analysis to construct a
code graph, and then synthesizes the training data base on the code graph.
7.2 Unseen Domain Code Generation
Different from generic code generation, unseen-domain code generation requires LLMs to under-
stand domain-specific logic, private APIs, and internal frameworks. Most existing approaches rely
on Retrieval-Augmented Generation (RAG), which augments general-purpose code LLMs with
externally retrieved knowledge at inference time without updating model parameters [44]. A key
challenge lies in effectively leveraging private-library APIs. Prior work [ 12,47,51,52] commonly
adopts multi-stage pipelines that retrieve relevant APIs and incorporate them as contextual in-
puts for code generation. In practice, domain-specific code generation has been explored across
industries. MedCoder [ 2] applies LLM-based extraction and retrieval for automated ICD coding;
Koziolek et al. [ 20] generate IEC 61131-3 ST control code via RAG; AnalogCoder [ 21] focuses on
analog circuit design through prompt engineering, while VerilogCoder [ 16] integrates graph-based
planning and AST–waveform tracking for Verilog generation and verification.
However, most existing methods that rely on retrieval-augmented generation are insufficient for
enabling models to fully understand the intrinsic relationships among different components of a
codebase. Our work synthesizes high-quality architectural and application-level data grounded in
the target codebase and leverages post-training to effectively enhance models’ understanding and
practical usage of the target codebase.
8 Conclusion
In this work, we propose UCD-Training, a two-stage training framework that performs reasoning-
aware data synthesis based on a code relation graph constructed from unseen codebases. UCD-
Training first parses the source code of unseen codebase to build the code graph, then performs
continued pretraining (CPT) using file-level dependency data, and finally performs supervised
fine-tuning (SFT) with three types of synthesized data augmented with explicit reasoning traces.
We also conduct a new code generation benchmark, UnseenCodeBench, targeting unseen code-
bases and perform comprehensive experiments to demonstrate that UCD-Training consistently
outperforms existing baselines across multiple codebases and two programming languages. Further-
more, our method exhibits strong generality: across codebases in different programming languages,
models of varying parameter scales, and diverse model architectures, UCD-Training effectively im-
proves models’ understanding and application capabilities in unseen codebases domains compared
to existing data synthesis baselines, achieving performance gains ranging from 7.2% to 26.1% on
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 19
UnseenCodeBench. We also simulate realistic enterprise scenarios involving multiple languages
and multiple target codebase, and show that our approach remains effective in real-world software
development settings, achieving a pass@1 of 36.0% on UnseenCodeBench and still surparssing all
baselines by 4.7% to 23.6%
9 Data availability
To facilitate the replication study, we have released our data and benchmark at: https://github.com/
ooggss/Unseen-Codebases-Domain-Data-Synthesis-and-Training-Based-on-Code-Graphs.
References
[1] Anthropic. 2025. Claude Sonnet 4.5. https://www.anthropic.com/news/claude-sonnet-4-5
[2]Krishanu Das Baksi, Elijah Soba, John J Higgins, Ravi Saini, Jaden Wood, Jane Cook, Jack I Scott, Nirmala Pudota, Tim
Weninger, Edward Bowen, and Sanmitra Bhattacharya. [n. d.]. MedCodER: A Generative AI Assistant for Medical
Coding. InProceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 3: Industry Track), 2025. 449–459.
[3]Sahil Chaudhary. 2023. Code Alpaca: An Instruction-following LLaMA model for code generation. https://github.com/
sahil280114/codealpaca.
[4]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards,
Yuri Burda, Nicholas Joseph, Greg Brockman, et al .2021. Evaluating large language models trained on code.arXiv
preprint arXiv:2107.03374(2021).
[5] davidhalter. 2026. https://github.com/davidhalter/jedi
[6] Deepmind. 2025. Gemini-3.0-Pro. https://deepmind.google/models/gemini/pro/
[7] DeepSeek-AI. 2025. DeepSeek-V3.1-Teminus. https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus
[8] EmberEmu. 2025. Hexi. https://github.com/EmberEmu/Hexi
[9] Hugging Face. 2025. Open R1: A fully open reproduction of DeepSeek-R1. https://github.com/huggingface/open-r1
[10] Angela Fan, Beliz Gokkaya, Mark Harman, Mitya Lyubarskiy, Shubho Sengupta, Shin Yoo, and Jie M. Zhang. 2023.
Large Language Models for Software Engineering: Survey and Open Problems. InIEEE/ACM International Conference
on Software Engineering: Future of Software Engineering, ICSE-FoSE 2023, Melbourne, Australia, May 14-20, 2023. IEEE,
31–53.
[11] getml. 2025. sqlgen. https://github.com/getml/sqlgen
[12] Xiaodong Gu, Meng Chen, Yalan Lin, Yuhan Hu, Hongyu Zhang, Chengcheng Wan, Zhao Wei, Yong Xu, and Juhong
Wang. 2025. On the Effectiveness of Large Language Models in Domain-Specific Code Generation.ACM Transactions
on Software Engineering and Methodology34, 3 (2025), 78:1–78:22.
[13] Beliz Gunel, Jingfei Du, Alexis Conneau, and Ves Stoyanov. 2020. Supervised contrastive learning for pre-trained
language model fine-tuning.arXiv preprint arXiv:2011.01403(2020).
[14] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y Wu, YK Li, et al .
2024. DeepSeek-Coder: When the Large Language Model Meets Programming–The Rise of Code Intelligence.arXiv
preprint arXiv:2401.14196(2024).
[15] Junxiao Han, Shuiguang Deng, Xin Xia, Dongjing Wang, and Jianwei Yin. 2019. Characterization and prediction of
popular projects on github. In2019 IEEE 43rd annual computer software and applications conference (COMPSAC), Vol. 1.
IEEE, 21–26.
[16] Chia-Tung Ho, Haoxing Ren, and Brucek Khailany. [n. d.]. VerilogCoder: Autonomous Verilog Coding Agents with
Graph-based Planning and Abstract Syntax Tree (AST)-based Waveform Tracing Tool. InProceedings of the AAAI
Conference on Artificial Intelligence, 2025, Vol. 39. 300–307.
[17] Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, and Haoyu
Wang. 2024. Large language models for software engineering: A systematic literature review.ACM Transactions on
Software Engineering and Methodology33, 8 (2024), 1–79.
[18] Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai
Dang, et al. 2024. Qwen2. 5-Coder Technical Report.arXiv preprint arXiv:2409.12186(2024).
[19] Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. 2024. A survey on large language models for code
generation.ACM Transactions on Software Engineering and Methodology(2024).
[20] Heiko Koziolek, Sten Grüner, Rhaban Hark, Virendra Ashiwal, Sofia Linsbauer, and Nafise Eskandani. [n. d.]. LLM-based
and Retrieval-augmented Control Code Generation. InProceedings of the 1st International Workshop on Large Language
Models for Code, 2024. 22–29.
, Vol. 1, No. 1, Article . Publication date: February 2026.

20 Guangsheng Ou, Qiming Zhang et al.
[21] Yao Lai, Sungyoung Lee, Guojin Chen, Souradip Poddar, Mengkang Hu, David Z Pan, and Ping Luo. [n. d.]. Analogcoder:
Analog Circuit Design via Training-free Code Generation. InProceedings of the AAAI Conference on Artificial Intelligence,
2025, Vol. 39. 379–387.
[22] Anji Li, Mingwei Liu, Zhenxi Chen, Zheng Pei, Zike Li, Dekun Dai, Yanlin Wang, and Zibin Zheng. 2025. KTester: Lever-
aging Domain and Testing Knowledge for More Effective LLM-based Test Generation.arXiv preprint arXiv:2511.14224
(2025).
[23] Caihua Li, Lianghong Guo, Yanlin Wang, Wei Tao, Zhenyu Shan, Mingwei Liu, Jiachi Chen, Runze Liu, Haoyu Song,
Duyu Tang, et al .2026. Advances, Frontiers, and Future of Issue Resolution in Software Engineering: A Comprehensive
Survey.Authorea Preprints(2026).
[24] Mingwei Liu, Zheng Pei, Yanlin Wang, Zihao Wang, Zikang Li, Enci Lin, Xin Peng, and Zibin Zheng. 2025. Framework-
Aware Code Generation with API Knowledge Graph-Constructed Data: A Study on HarmonyOS.arXiv preprint
arXiv:2512.00380(2025).
[25] LLVM. 2026. https://clang.llvm.org/
[26] Shuai Lu, Nan Duan, Hojae Han, Daya Guo, Seung-won Hwang, and Alexey Svyatkovskiy. 2022. ReACC: A Retrieval-
Augmented Code Completion Framework. InProceedings of the 60th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), ACL 2022. Association for Computational Linguistics, 6227–6240.
[27] lumia431. 2025. reaction. https://github.com/lumia431/reaction
[28] Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin,
and Daxin Jiang. 2023. WizardCoder: Empowering Code Large Language Models with Evol-Instruct.arXiv preprint
arXiv:2306.08568(2023).
[29] Yichuan Ma, Yunfan Shao, Peiji Li, Demin Song, Qipeng Guo, Linyang Li, Xipeng Qiu, and Kai Chen. 2025. UnitCoder:
Scalable Code Synthesis from Pre-training Corpora. InProceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing. 5623–5641.
[30] Mihai Nadă s,, Laura Dio s,an, and Andreea Tomescu. 2025. Synthetic data generation using large language models:
Advances in text and code.IEEE Access(2025).
[31] OpenAI. 2025. GPT-5.1. https://openai.com/index/gpt-5-1/
[32] Guangsheng Ou, Mingwei Liu, Yuxuan Chen, Xueying Du, Shengbo Wang, Zekai Zhang, Xin Peng, and Zibin Zheng.
2025. Enhancing llm-based code translation in repository context via triple knowledge-augmented.arXiv preprint
arXiv:2503.18305(2025).
[33] Guangsheng Ou, Mingwei Liu, Yuxuan Chen, Xin Peng, and Zibin Zheng. 2025. Repository-level code translation
benchmark targeting rust. In2025 38th IEEE/ACM International Conference on Automated Software Engineering (ASE).
[34] AS Saabith, MMM Fareez, and T Vinothraj. 2019. Python current trend applications-an overview.International Journal
of Advance Engineering and Research Development6, 10 (2019).
[35] Disha Shrivastava, Hugo Larochelle, and Daniel Tarlow. 2023. Repository-level prompt generation for large language
models of code. InInternational Conference on Machine Learning. PMLR, 31693–31715.
[36] Bjarne Stroustrup. 2013.The C++ programming language. Pearson Education.
[37] Chaofan Tao, Jierun Chen, Yuxin Jiang, Kaiqi Kou, Shaowei Wang, Ruoyu Wang, Xiaohui Li, Sidi Yang, Yiming Du,
Jianbo Dai, et al .2026. SWE-Lego: Pushing the Limits of Supervised Fine-tuning for Software Issue Resolving.arXiv
preprint arXiv:2601.01426(2026).
[38] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https://arxiv.org/abs/2505.09388
[39] tree sitter. 2023. https://github.com/tree-sitter/tree-sitter
[40] Darko Ðurđev. 2024. Popularity of programming languages.AIDASCO Reviews2, 2 (2024), 24–29.
[41] Xin Wang, Yasheng Wang, Yao Wan, Fei Mi, Yitong Li, Pingyi Zhou, Jin Liu, Hao Wu, Xin Jiang, and Qun Liu. 2022.
Compilable neural code generation with compiler feedback.arXiv preprint arXiv:2203.05132(2022).
[42] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi.
2023. Self-instruct: Aligning language models with self-generated instructions. InProceedings of the 61st annual meeting
of the association for computational linguistics (volume 1: long papers). 13484–13508.
[43] Yunkun Wang, Yue Zhang, Zhen Qin, Chen Zhi, Binhua Li, Fei Huang, Yongbin Li, and Shuiguang Deng. 2025. Explo-
raCoder: Advancing code generation for multiple unseen APIs via planning and chained exploration. InProceedings of
the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 18124–18145.
[44] Zora Zhiruo Wang, Akari Asai, Frank F Xu, Yiqing Xie, Graham Neubig, Daniel Fried, et al .2025. Coderag-bench:
Can retrieval augment code generation?. InFindings of the Association for Computational Linguistics: NAACL 2025.
3199–3214.
[45] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. 2023. Magicoder: Empowering code generation
with oss-instruct.arXiv preprint arXiv:2312.02120(2023).
[46] Tongtong Wu, Trang Vu, Linhao Luo, and Gholamreza Haffari. 2025. Continual Learning of Large Language Models.
InProceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Tutorial Abstracts. 16–17.
, Vol. 1, No. 1, Article . Publication date: February 2026.

Unseen-Codebases-Domain Data Synthesis and Training Based on Code Graphs 21
[47] Conghui Yang, Lei Yu, Huafeng Su, and Xiang Zhou. [n. d.]. APICoder: A Multi-Role Large Language Model Framework
for API Service Call Code Generation. In2025 IEEE International Conference on Web Services, 2025. 849–851.
[48] Guang Yang, Yu Zhou, Xiang Chen, Xiangyu Zhang, Terry Yue Zhuo, and Taolue Chen. 2024. Chain-of-thought in
neural code generation: From and for lightweight language models.IEEE Transactions on Software Engineering(2024).
[49] Zezhou Yang, Sirong Chen, Cuiyun Gao, Zhenhao Li, Xing Hu, Kui Liu, and Xin Xia. 2025. An empirical study of
retrieval-augmented code generation: Challenges and opportunities.ACM Transactions on Software Engineering and
Methodology(2025).
[50] yichuan w. 2025. LEANN. https://github.com/yichuan-w/LEANN
[51] Daoguang Zan, Bei Chen, Zeqi Lin, Bei Guan, Wang Yongji, and Jian-Guang Lou. 2022. When language model meets
private library. InFindings of the Association for Computational Linguistics: EMNLP 2022. 277–288.
[52] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen.
2023. RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation. InProceedings of
the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023. Association for Computational
Linguistics, 2471–2484.
[53] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang,
Guoyin Wang, et al .2026. Instruction tuning for large language models: A survey.Comput. Surveys58, 7 (2026), 1–36.
[54] Yuanliang Zhang, Yifan Xie, Shanshan Li, Ke Liu, Chong Wang, Zhouyang Jia, Xiangbing Huang, Jie Song, Chaopeng
Luo, Zhizheng Zheng, et al .2025. Unseen Horizons: Unveiling the Real Capability of LLM Code Generation Beyond
the Familiar. In 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE).IEEE Computer Society
(2025), 619–619.
[55] Ziyao Zhang, Chong Wang, Yanlin Wang, Ensheng Shi, Yuchi Ma, Wanjun Zhong, Jiachi Chen, Mingzhi Mao, and Zibin
Zheng. 2025. Llm hallucinations in practical code generation: Phenomena, mechanism, and mitigation.Proceedings of
the ACM on Software Engineering2, ISSTA (2025), 481–503.
[56] Han Zhao, Haotian Wang, Yiping Peng, Sitong Zhao, Xiaoyu Tian, Shuaiting Chen, Yunjie Ji, and Xiangang Li. 2025. 1.4
Million Open-Source Distilled Reasoning Dataset to Empower Large Language Model Training. arXiv:2503.19633 [cs.CL]
https://arxiv.org/abs/2503.19633
[57] Da-Wei Zhou, Hai-Long Sun, Jingyi Ning, Han-Jia Ye, and De chuan Zhan. 2024. Continual Learning with Pre-Trained
Models: A Survey. InInternational Joint Conference on Artificial Intelligence. https://api.semanticscholar.org/CorpusID:
267312447
[58] Shuyan Zhou, Uri Alon, Frank F Xu, Zhengbao Jiang, and Graham Neubig. 2022. Docprompting: Generating code by
retrieving the docs. InThe Eleventh International Conference on Learning Representations.
[59] Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf,
Haolan Zhan, Junda He, Indraneil Paul, and et al. 2025. BigCodeBench: Benchmarking Code Generation with Diverse
Function Calls and Complex Instructions. InThe Thirteenth International Conference on Learning Representations, ICLR
2025. OpenReview.net.
, Vol. 1, No. 1, Article . Publication date: February 2026.