# MRG-Bench: Evaluating and Exploring the Requirements of Context for Repository-Level Code Generation

**Authors**: Haiyang Li

**Published**: 2025-08-05 01:53:45

**PDF URL**: [http://arxiv.org/pdf/2508.02998v1](http://arxiv.org/pdf/2508.02998v1)

## Abstract
Large Language Models (LLMs) have demonstrated impressive capabilities in
code generation. However, current evaluation datasets suffer from issues such
as the lack of runnable test cases, deviation from the distribution of
real-world code, and the ability to evaluate only the Python language. These
limitations undermine the credibility of the evaluation results.
  To address these limitations, we introduce \textbf{MRG-Bench} (Multi-language
Repository-level Code Generation Benchmark), a novel dataset that provides a
more accurate evaluation of LLMs in practical repository-level code generation
tasks. MRG-Bench has three main features: (1) practical data sourced from
real-world code repositories that align to the practical distribution, (2)
multiple programming languages support, including Python, Java, and Go, and (3)
project-level runnable test cases to assess the quality of the generated code.
  Based on MRG-Bench, we conducted extensive experiments including large
language models, long-context models, and RAG-related methods. These evaluation
results demonstrate that \textbf{current repository-level code generation
techniques suffer from significant performance deficiencies}. To further
investigate why models fail, we designed novel experiments to annotate the
underlying causes of generation errors. The results explicitly show that the
majority of methods suffer from "\textbf{difficulty in understanding user
requirements}," failing to comprehend their assigned tasks accurately.
Moreover, the impact of different repository-level contexts on this issue
exhibits significant disparities across different programming languages,
suggesting that, in practice, specialized contextual information needs to be
designed for different languages.

## Full Text


<!-- PDF content starts -->

MRG-Bench: Evaluating and Exploring the Requirements of
Context for Repository-Level Code Generation
Haiyang Li
Peking University
Beijing, ChinaQing Gao
Peking University
Beijing, ChinaShikun Zhang
Peking University
Beijing, China
Abstract
Large Language Models (LLMs) have demonstrated impressive ca-
pabilities in code generation. However, current evaluation datasets
suffer from issues such as the lack of runnable test cases, deviation
from the distribution of real-world code, and the ability to eval-
uate only the Python language. These limitations undermine the
credibility of the evaluation results.
To address these limitations, we introduce MRG-Bench (Multi-
language Repository-level Code Generation Benchmark), a novel
dataset that provides a more accurate evaluation of LLMs in practi-
cal repository-level code generation tasks. MRG-Bench has three
main features: (1) practical data sourced from real-world code repos-
itories that align to the practical distribution, (2) multiple program-
ming languages support, including Python, Java, and Go, and (3)
project-level runnable test cases to assess the quality of the gener-
ated code.
Based on MRG-Bench, we conducted extensive experiments in-
cluding large language models, long-context models, and RAG-
related methods. These evaluation results demonstrate that current
repository-level code generation techniques suffer from sig-
nificant performance deficiencies . To further investigate why
models fail, we designed novel experiments to annotate the under-
lying causes of generation errors. The results explicitly show that
the majority of methods suffer from " difficulty in understanding
user requirements ," failing to comprehend their assigned tasks
accurately. Moreover, the impact of different repository-level con-
texts on this issue exhibits significant disparities across different
programming languages, suggesting that, in practice, specialized
contextual information needs to be designed for different languages.
CCS Concepts
•Software and its engineering →Software creation and man-
agement .
Keywords
Large Language Models, Benchmark of LLM, Repository Level Code
Generation,
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym, Venue
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXACM Reference Format:
Haiyang Li, Qing Gao, and Shikun Zhang. 2018. MRG-Bench: Evaluating
and Exploring the Requirements of Context for Repository-Level Code
Generation. In Proceedings of Conference Name (Conference acronym). ACM,
New York, NY, USA, 12 pages. https://doi.org/XXXXXXX.XXXXXXX
60
 40
 20
 0 20 40 6040
20
02040A: Semantic Distribution of Real, LLM Generated and MRG Comments
Sampled Real Comments
MRG Comments
LLM Generated Comments
60
 40
 20
 0 20 40 6075
50
25
0255075B: Distribution of Python, Java and Go Function Comment Embeddings
Python Doc Embeddings
Java Doc Embeddings
Go Doc Embeddings
(a) The semantic distributions of function comments across three
different programming languages exhibit significant differences,
highlighting the importance of evaluating each language indepen-
dently.
60
 40
 20
 0 20 40 6040
20
02040A: Semantic Distribution of Real, LLM Generated and MRG Comments
Sampled Real Comments
MRG Comments
LLM Generated Comments
60
 40
 20
 0 20 40 6075
50
25
0255075B: Distribution of Python, Java and Go Function Comment Embeddings
Python Doc Embeddings
Java Doc Embeddings
Go Doc Embeddings
(b) The semantic distributions of function comments of real-world
data, LLM-generated data, and MRG-bench.
1 Introduction
The remarkable performance of large language models (LLMs) in
code generation tasks has garnered significant attention in recent
years, with related research steadily emerging [ 10,11]. The intro-
duction of commercial products like GitHub Copilot [ 9] further
highlights the practical potential of LLMs. These models not only
assist developers in generating code but also enhance efficiency
in various aspects of the development process. However, current
evaluation methods [ 7,20,26,28] for LLMs in code generation
predominantly focus on standalone code fragments, which limits
their applicability in real-world development scenarios. In prac-
tice, development is typically conducted within independent code
repositories, involving complex code dependencies, modular archi-
tectures, and multiple programming languages. Thus, evaluating
the code generation capabilities of LLMs within the context of an
actual code repository has become a key focus in ongoing research.arXiv:2508.02998v1  [cs.SE]  5 Aug 2025

Conference acronym, Date, Venue Haiyang Li, Qing Gao, and Shikun Zhang
Table 1: Comparison of different datasets
dataset multi-language support repository information project runnable environment practical data source
NumpyEval [27] ✗ ✓ ✗ ✗
ClassEval [7] ✗ ✗ ✗ ✗
CoderEval [26] ✓ ✗ ✗ ✓
EvolCodeBench [17] ✗ ✓ ✓ ✗
AgentBench [30] ✗ ✓ ✓ ✗
SWEBench [13] ✗ ✓ ✓ ✓
RepoCoder [29] ✗ ✓ ✗ ✓
MRG-Bench ✓ ✓ ✓ ✓
Several efforts [ 17,18,29] have proposed repository-level code
generation datasets, but they exhibit the following limitations: (1)
Limited programming languages: These datasets predominantly
focus on specific programming languages, such as Python. How-
ever, as shown in Figure 1a, the distributions of semantic embed-
dings from three different programming languages exhibit distinct
differences. In fact, different programming languages often excel
in specific domains of code. Therefore, a multi-language evalu-
ation dataset can better assess the performance of LLMs (Large
Language Models) in real-world programming scenarios. (2) Incon-
sistent with the practical situation: Some work [ 17] attempts to use
large language models to generate code summaries, treating these
summaries as natural language descriptions of the code. As shown
in Figure 1b, there is a significant semantic discrepancy between
the requirement descriptions generated by LLMs and the actual
code requirements. (3) Lack of a runnable environment: Although
some approaches generate code snippets that appear highly similar
to the target code, these snippets may fail to compile or pass tests
when executed in a real-world environment. (4) Limited Evaluation
Scope: Previous studies have predominantly attempted to extract
fixed contexts from repositories and evaluate large language models
accordingly. However, in practice, different retrieval methods are
often employed to recall relevant code snippets from repositories,
or more powerful reasoning models are utilized for generation.
The absence of evaluation for these methods in prior work has
impacted the accurate assessment of current technologies. These
three limitations undermine the reliability of previous evaluations
of repository-level code generation tasks.
Addressing the problems above, we first propose a new evalu-
ation dataset, MRG-Bench(Multi-language repository level code
Generation Benchmark), designed to more accurately reflect the
real-world performance of LLMs in repository-level code genera-
tion. Subsequently, we conducted extensive evaluation experiments
based on this dataset, encompassing multiple language models,
long-context models, reasoning models, as well as common RAG
methods. Furthermore, we attempted to design a novel data annota-
tion dimension to investigate why current methods perform poorly
on MRG-Bench.
For the dataset, MRG-Bench has three key features: (1) Practical
data source: All data in the dataset is collected from real-world
code repositories, ensuring that the code snippets reflect the ac-
tual distribution of code in real development scenarios. (2) Multi-
language coverage: In addition to Python (the most common lan-
guage), MRG-Bench includes other programming languages, suchas Java and Go, enabling a more comprehensive evaluation of LLMs’
code generation capabilities in different languages. (3) Runnable
test cases: Each project in the dataset includes executable test
cases, ensuring that the code generated by the model can be run
in real-world environments, thereby enhancing the credibility of
the evaluation results. As shown in Table 1, current datasets ex-
hibit significant gaps across critical features, whereas MRG-Bench
demonstrates more comprehensive assessment capabilities. It is
noteworthy that although we include SWE-Bench [ 13] in the table,
its target scenario differs fundamentally from MRG-Bench; SWE-
Bench is designed for bug fixing according to real issues rather than
code generation scenarios.
For the evaluation, we conduct an extensive evaluation of state-
of-the-art models, including large language models, reasoning mod-
els, long-context models, and several RAG-based methods. Our
experiments highlight the following key findings: (1) Poor perfor-
mance of LLMs on MRG-Bench : Current large language models
perform inadequately on the MRG-Bench dataset. Even the best
language model, Claude-3.5-Sonnet, achieves an average Pass@1
score of only 32.5%, indicating significant room for improvement.
(2)Language bias in large language models : LLMs exhibit a no-
ticeable bias towards specific programming languages. All models
perform better in Python when relevant contextual information is
lacking. (3) Ineffectiveness of RAG-related methods : Retrieval-
Augmented Generation (RAG) methods perform poorly, even worse
than simply providing the file information containing the function.
Based on the findings before, we further investigated the under-
lying causes of failures in code generation across different contexts.
Specifically, we decomposed the model generation process into two
perspectives: (1) understanding user requirements (knowing What
to do) and (2) implementing user requirements (knowing How
to do it). We employed model-based annotation for failed cases.
The results reveal several interesting findings: (1) The majority
of failure cases occur in "What to do", where models fail
to comprehend what users intend to generate. (2) Different
repository contexts exhibit pronounced disparities in their
effects on performance. For instance, intra-file content signifi-
cantly aids model comprehension of user requirements in Java and
Go, while showing negligible impact in Python. These two experi-
mental findings indicate that current models are most struggling at
requirement understanding. Moreover, methods for enhancing this
stage vary across different programming languages. Consequently,

MRG-Bench: Evaluating and Exploring the Requirements of Context for Repository-Level Code Generation Conference acronym, Date, Venue
One Samle of MRG-Bench
// DispatchingStrategyRoundRobin distributes messages in a
rotating sequential manner .
// If the channel capacity is exceeded, the next channel will be
selected and so on.
func DispatchingStrategyRoundRobin[T  any](msg T, index
uint64, channels []<-chan T) int 
func DispatchingStrategyRoundRobin[T  any](msg T, index
uint64, channels []<-chan T) int {
for {
i := int(index % uint64(len(channels)))
if channelIsNotFull(channels[i]) {
return i
}
index++
time.Sleep(10 * time.Microsecond) // prevent CPU
from burning 🔥
}
}TestDispatchingStrategyRoundRobin
... ...
channel.go::channelIsNotFull
... ...
├── math_example_test.go
├── math_test.go
├── parallel
│   ├ ── slice.go
│   └── slice_test.go
├── retry .go
├── retry_example_test.go
... ...（1）
（2）
（3）（4）
（5）
（6）
（1）  Function Annotations （2）  Function Signature （3）  Reference Function Body 
（4）  Test Cases （5）  Called Functions （6）  Repository Info
Figure 2: All Information in One Sample of MRG-Bench
in real-world engineering applications, we may need to dynami-
cally adjust supplementary contextual content based on the specific
programming language to achieve superior performance.
In conclusion, our work makes the following contributions:
1.Practical dataset with multiple languages: We published a
multi-language dataset, MRG-Bench, with a real-world data distri-
bution and runnable test cases, offering a more realistic evaluation
benchmark for future studies. Besides, we released the function call
graph analysis framework which enables researchers to quickly
generate private evaluation datasets for their own research.
2.Comprehensive evaluation and analysis: We performed
an extensive evaluation of the performance of popular large lan-
guage models on MRG-Bench, along with an analysis of the basic
RAG methods. Our results demonstrate that current models need
improvement when applied to real-world tasks.
3.Novel Analytical Methodology : We designed a novel an-
alytical approach to investigate the underlying causes of failures
and to examine the influence of varying contextual information
on the results. Our experimental findings demonstrate that the pri-
mary cause of failures lies in the inability to accurately comprehend
user inputs, thereby indicating that enhancements targeting this
particular perspective can rapidly improve the final performance.
We hope this work provides valuable insights into the application
of large language models in real-world coding scenarios and inspires
future research.
2 MRG-Bench Description
MRG-Bench includes three programming languages and contains a
total of 383 samples. Each sample is a function-level code generation
task which includes the following information, as shown in Figure 2:
(1)Function annotations: Natural language comment used for
generating the function.
(2)Function signature: function name, parameters, and return
type (if any).
(3) Referenced Function Body: code snippet from repository.
(4)Test Cases: unit test or integration test cases for this sample.Table 2: Language Statistics
Language Samples Avg Query Tokens Avg Target Tokens
Python 124 80.90 283.00
Java 96 70 152.83
Go 163 38.36 153.23
(5)Called Private Functions: A list of other functions in the
repository that the function calls.
(6) Repository Information: The original repository.
3 MRG-Bench Construction
Our dataset construction adheres to three primary principles:
Practical: The dataset is derived from real-world open-source
projects, avoiding artificially constructed data. The function sig-
natures, natural language descriptions of code functions, and test
cases are all sourced from actual projects. Besides, the distribution
of queries should closely approximate the distribution of queries in
real-world code repositories.
Multi-Language: The dataset should include different program-
ming languages to facilitate comprehensive evaluations.
Executability: The ultimate evaluation criterion for the dataset
is that the code generated by models should work in real-world
scenarios, passing both unit and integration tests in the project.
The construction of our dataset is divided into following key
steps.
3.1 Programming language selection
We select three languages in this version of MRG-Bench: Python,
Go, and Java, which have robust package management approaches
and unit test frameworks. The primary reason for excluding C and
C++ was the difficulty of managing conflicting dependencies in con-
structing runnable environments in Linux Docker containers (plan-
ning for next version, see Section 6). These projects often depend
on specific Linux distributions and various dynamic libraries, for
which there is no efficient way to manage environments. Although
JavaScript has a robust package management system that allows for
quick configuration of its runtime environment, we found that the
code comment ratio and unit test coverage were low. Additionally,
most test suites were integration tests, making it challenging to
match each test case to its corresponding function through auto-
mated analysis. Therefore, we finally selected Python, Java, and Go
as the supported languages of our datasets.
3.2 Selection of code repositories
For the selected programming languages, we utilized the GitHub
API to retrieve repositories. To ensure that the large language
model (LLM) has limited prior knowledge of the repositories, we
focused on repositories created after January 2023, following pre-
vious work [ 29]. We ranked these repositories by the number of
stars and manually reviewed repositories over 50 stars, following
previous work [ 17]. The repositories were filtered based on the
following criteria:
(1)The repository should include a complete test suite, contain-
ing at least 10 unit test functions.

Conference acronym, Date, Venue Haiyang Li, Qing Gao, and Shikun Zhang
(2) The repository must be able to run on the Linux platform.
(3)The repository should compile and execute successfully, pass-
ing its most test cases.
Based on these criteria, we finally selected 152 repositories from
a total of 1,000 repositories across 3 languages.
3.3 Function dependency construction
After obtaining a project that can be compiled and tested locally, we
extracted the functions required for our dataset. These functions
were selected based on the following criteria:
(1)They have dependencies on other functions within the repos-
itory.
(2) They include developer-written function comments.
(3) They have corresponding test cases within the project.
To extract functions that meet these requirements, we devel-
oped project-level function call graph analyzers for each language.
Specifically, we first utilized Tree-sitter to parse the source code
into function definitions and class method definitions. For each
function, we analyzed the files (import info and package info) and
symbols referenced (variable defined info) within the function, and
matched function calls to project-defined functions. This process al-
lowed us to construct a function call graph for all custom functions
in the project.
Next, we identified test functions by checking whether the func-
tion name, class name, file name, or path contained the keyword
"test." Each test function was then linked to the function(s) it called,
marking it as a test case for those functions. Finally, we saved all
non-test functions that contain documentation and corresponding
test cases as candidate data. In this phase, we further filtered out
repositories that did not yield valid data (i.e., those that did not pro-
duce functions containing both test cases and function annotations),
ultimately leaving 23 projects, 580 samples remaining.
3.4 Manual inspection and Test Coverage
Filtering
In the final step of dataset construction, we reviewed the obtained
dataset to ensure that each function includes meaningful annota-
tions. We identified and filtered out annotations that do not provide
meaningful descriptions of the function’s behavior, such as generic
comments like "return String" or "implement interface ClientInter-
face." Additionally, we manually executed the unit tests of these
projects and utilized corresponding tools (i.e., pytest-cov ,Jacoco ,
andgo test -cover ) to generate function-level test coverage in-
formation. We then removed all functions that did not achieve 100%
line coverage. Ultimately, we obtained a total of 383 samples from
22 projects.
3.5 Evaluation Method and Metrics
We have configured the runtime environment and test command
for each project in a docker container. When executing the tests, we
copy the function to be tested to the target position of the file in the
project, and then run the corresponding test case for that sample.
Specifically, for Python projects, we use pytest to execute the test
cases. For Java projects, we utilize Maven to manage dependenciesand run unit tests using the mvn test command. For Go projects,
we use go test to execute the specified test cases.
All projects are configured within an executable Linux container,
where the necessary environments have already been set up. Given
the complexity and time required to build this container from
scratch, we do not recommend that researchers attempt to do so.
Instead, you can pull our pre-configured Docker image from Docker
Hub to quickly verify your method locally.
We use the Pass@k metric as the performance evaluation met-
ric for MRG-Bench. Pass@n represents the proportion of samples
which has at least one generation result that passes all test cases.
4 Experiment Setting
4.1 Model Selection
We selected six large language models of varying sizes for experi-
ments. These models have demonstrated strong coding capabilities
and achieved competitive scores on the HumanEval dataset [ 2].
Open-source models include DeepSeek-Coder-33B [ 11], CodeLLaMA-
13B [ 10], LLaMA-3.1-8B-Instruct1, and StarChat2-15B [ 19], while
the closed-source models are Claude3.5-Sonnet2and GPT-4o3.
For reasoning models, we select o3-mini4, Deepseek-R1 [ 3], and
Qwen2.5-QwQ-32B [25].
4.2 Experimental Setup
For the open-source models, we deployed them locally using vLLM
[15]. For the closed-source models, we accessed GPT-4o via the
Azure API5and Claude3.5-Sonnet through VertexAI’s API6. We
set the temperature to 0.6 to balance the stability and creativity
of the model’s generated outputs, while the remaining parameters
were configured using either the API defaults or the default settings
provided by vLLM. Due to the request frequency limitations of
closed-source models, we could not include them in all experiments.
For the subsequent RAG-based experiments, we utilized DeepSeek-
Coder-33B, as it performed best in the previous evaluation in open-
source LLMs. The RAG experiments were conducted using the
LangChain7framework. All the code for our experiments is publicly
available on GitHub8.
4.3 Research Questions
In this paper, we conduct experiments to answer the following
research questions:
RQ1: Why do we need MRG-Bench? Corresponding to the
motivations for constructing MRG-Bench outlined in Section 1,
we designed experiments to demonstrate the advantages of MRG-
Bench over existing datasets from two perspectives.
RQ1.1 Is MRG-Bench more representative compared to ex-
isting repository-level code generation datasets? We analyzed
the semantic distribution of samples in MRG-Bench to show that
our dataset aligns more closely with real-world code distributions
1https://github.com/meta-llama/llama-models
2https://claude.ai/
3https://chatgpt.com/
4https://chatgpt.com/
5https://azure.microsoft.com/
6https://console.cloud.google.com/vertex-ai/
7https://www.langchain.com/
8https://github.com/MRG-Bench/MRG-Bench

MRG-Bench: Evaluating and Exploring the Requirements of Context for Repository-Level Code Generation Conference acronym, Date, Venue
-60 -40 -20 0 20 40 60-40-2002040Distribution of Python function docstring across different datasets
EvolCodeBench
random_samples
MRG-bench
real_samples
(a) Distribution of samples for different Python
datasets
Python Java GO0.50.60.70.80.91.01.11.2Reconstruction Error of Different Datasets with real samples
EvolCodeBench
CoderEval
MRG-Bench
random_samples
(b) Reconstruction Error between different datasets
and real data on 3 languages
compared to existing datasets. Consequently, results obtained on
this dataset better reflect the true performance of models.
RQ1.2 Is Multi-language evaluation necessary for current
LLMs? We conducted experiments on mainstream large language
models to empirically demonstrate the necessity of a multi-language
dataset for evaluating current models, thereby highlighting the
advantage of MRG-Bench’s multi-language support.
RQ2: How do current LLMs perform on MRG-Bench with
different contexts?
Lacking information about the function to be generated will de-
crease the performance of the models. In this question, we explore
the impact of different contextual information on the model. In this
question, we primarily designed two types of experimental settings:
static context and dynamic context. The static context encompasses
2 configurations: in-file context and available utility function infor-
mation. In the dynamic context experiments, we evaluated multiple
RAG-related methods, including basic RAG approaches based on
BM25 retrieval and embedding retrieval methods, as well as Re-
poCoder, an algorithm specifically designed for repository-level
code generation.
RQ3: Why do current methods perform poorly on MRG-
Bench?
In this question, we attempt to annotate the failure cases and
analyze why the models fail on these samples, as well as examine
the impact of different contexts on the model’s failure cases.5 Result and analysis
5.1 RQ1: Why do we need MRG-Bench?
5.1.1 RQ1.1: Is MRG-Bench more representative compared to ex-
isting repository-level code generation datasets? One of the most
critical design principles of MRG-Bench is Practicality , as we
aim for MRG-Bench to align closely with the actual distribution
of open-source code. While some existing works have attempted
to construct repository-level code generation datasets, only two
datasets, EvoCodeBench [ 17] and AgentBench [ 14], provide com-
plete repository information with executable unit tests. However,
as the source code for AgentBench is unavailable, we compare the
data from EvolBench with MRG-Bench in this section.
Specifically, we selected 500 high-quality, popular open-source
projects (excluding the 22 projects in MRG-Bench) in Python, Java,
and Go to create a representative dataset. From these projects, we
randomly sampled 10,000 functions, extracting their docstrings and
function bodies to represent "real-world projects." To effectively
analyze the semantic distribution, we employed NV-Embedding-
2 [16], one of the state-of-the-art embedding models, to embed each
docstring into a 4,096-dimensional semantic vector.
For comparison, we first established a random control group,
referred to as the random-sampled dataset , for each language by
randomly extracting a dataset of equivalent size from open-source
repositories. Additionally, we selected queries from existing repository-
level code generation datasets for comparison. For Python, we
chose EvolCodeBench [17], and for Java, we selected data from the
CoderEval-Java [26] subset. Although CoderEval was not specifi-
cally designed as a repository-level code generation dataset, its data
is indeed sourced from real open-source repositories. Unfortunately,
there is currently no available dataset for the Go language, so we
could only compare it with the randomly sampled dataset. We mea-
sure the gap between each dataset and the real query distribution
(10,000 samples) using the Reconstruction Error of the sampled
data in reconstructing the real distribution, Reconstruction Error
is defined as follows:
Reconstruction of 𝐵using𝐴: For each𝑏∈𝐵, find its nearest
neighbor𝑎∈𝐴then compute the reconstruction error, smaller is
better :
Reconstruction Error =1
|𝐵|∑︁
𝑏∈𝐵min
𝑎∈𝐴∥𝑎−𝑏∥2
|𝐴|and|𝐵|represent the sizes (number of elements) of sets 𝐴
and𝐵, respectively.∥·∥ 2denotes the Euclidean norm. A smaller
reconstruction error indicates that the sampled data is closer to the
real data distribution.
Figure 3b shows the reconstruction errors between different
datasets to real-world data. We can find that MRG-Bench is signifi-
cantly closer to the real-world distribution compared to EvoCodeBench
and CoderEval-Java. The reconstruction error values of MRG-Bench
are very close to those of random sampling. These results demon-
strate that MRG-Bench represents the real-world code distribution
more accurately. Consequently, the results obtained on this dataset
better reflect the real performance of models.
To more clearly illustrate the distribution of the MRG-Bench,
we used t-SNE to reduce the dimensionality of the data and visual-
ized the data distribution for Python as an example, as shown in

Conference acronym, Date, Venue Haiyang Li, Qing Gao, and Shikun Zhang
Figure 3a. The gray data points in the figure represent real data,
and for clarity of the plot, we further sampled 2,000 data points
from the 10,000 total samples for visualization. We can find that the
queries from EvolCodeBench (in blue) deviate significantly from
the semantic distribution of real queries, with many semantics left
uncovered. In contrast, MRG-Bench aligns more closely with the
real distribution and the results of random sampling.
Takeaway-1 : MRG-Bench demonstrates a significantly lower re-
construction error compared to other datasets, indicating its superior
ability to represent real-world code semantics. Visualizations using
t-SNE further confirm that MRG-Bench closely mirrors the semantic
distribution of actual code, making it a more reliable benchmark for
evaluating model performance in practical scenarios.
5.1.2 RQ1.2 Why do we need a multi-language evaluation dataset?
To demonstrate the multi-language advantages of MRG-Bench, we
conducted experiments on the performance of LLMs across differ-
ent programming languages. Specifically, we provided the models
with the annotations and signatures of functions and asked them
to complete the specified function. To verify the baseline capabil-
ity differences of the models across languages, we did not provide
any additional contextual information beyond the function details.
The experimental results are shown in Table 3. According to the
results, it is evident that every model performs significantly better
on Python compared to Java and Go. Moreover, Go, being a rel-
atively niche language, exhibits the poorest performance among
the three languages. For example, LLaMA-3.1-8B-Instruct achieves
7.3% Pass@1 on Python but 1.2% Pass@1 on GO. This indicates
that existing models exhibit significant bias toward different pro-
gramming languages. Even Java, a relatively mainstream language,
shows notably weaker performance compared to Python. These
findings cannot be revealed by current evaluation datasets, which
are predominantly focused on Python, which presents a challenge
to previous evaluation efforts, as most earlier work focused on
single-language evaluations, often concentrated on Python. This
imbalance may lead to an overestimation of large language mod-
els’ performance in other programming languages. The finding
highlights the value of MRG-Bench’s multi-language support—not
only for evaluating performance on specific languages, but also for
uncovering significant performance disparities of models across
different programming languages.
Takeaway-2 : MRG-Bench’s multi-language evaluation reveals
significant performance disparities in LLMs across programming lan-
guages, illustrating how current benchmarks (often Python-centric)
fail to capture these gaps.
5.2 RQ2: How do current LLMs perform on
MRG-Bench with different contexts?
5.2.1 RQ2.1: How do different models perform when provided with
fixed context? In this experiment, we employ two types of fixed
contextual information: (1) Infile context: The source code contain-
ing the target function to be generated. (2) Callee Context: The
callee function information of the target function, efficiently ex-
tracted through our function call analysis framework. These two
information respectively represent the context surrounding the tar-
get function and available custom function definitions that may beinvoked. In addition to contextual settings, we also assessed the per-
formance of state-of-the-art models on MRG-Bench to understand
their capabilities in repository-level code generation.
We first investigated how much improvement can be achieved
by providing the file content in which the function is located, as
this information is the easiest to obtain in practice. Specifically, we
prepended the file content to the beginning of the function genera-
tion prompt and asked the model to generate the function based
on the given annotations and function signature. Since different
models use different tokenizers, we used the tokenizer from GPT-
3.5-Turbo to truncate the file content to a maximum of 6000 tokens,
ensuring that no model exceeded its maximum token length. Due
to the shorter context window of StarChat2-15B, this model was
excluded from this experiment. The results are shown in the Table
4.
Based on the results, we find that providing contextual infor-
mation about the function significantly improves the Pass@1 rate.
For example, the Pass@1 of CodeLLaMA-13B shows noticeable
improvement from an average of 7.4% to 13.3%. After incorporating
relevant context, DeepSeek-Coder-33B demonstrates a substan-
tial advantage, outperforming CodeLLaMA-13B and LLaMA-3.1-8B.
Moreover, DeepSeek-Coder-33B exhibits strong competitiveness
with closed-source models; for instance, its Pass@3 score is compa-
rable to the pass@1 performance of GPT-4o and Claude3.5-Sonnet.
Additionally, we find that while the performance gap between
languages was substantial in RQ1.2, the differences between models
across different languages narrowed significantly after providing
rich context. This implies that evaluation outcomes may vary sig-
nificantly across programming languages under different contex-
tual conditions, necessitating an expansion of current assessment
methodologies.
Takeaway-3 : Providing in-file contextual information signifi-
cantly improves the performance of large language models (LLMs).
Additionally, the performance gap between languages narrowed when
context was included. This implies that evaluation outcomes may
vary significantly across programming languages under different con-
textual conditions, necessitating an expansion of current assessment
methodologies.
A major challenge in repository-based code generation is that
the model lacks knowledge of the available functions within the
repository. Thanks to our analysis framework, we are able to extract
the repository functions utilized by the target function during the
generation process and provide them to the model as context. In
this section, we conducted experiments on the previously identi-
fied optimal open-source model, DeepSeek-Coder-33B [ 11]. All the
callable functions are concatenated into the prompt and provided
as input to the model, which is tasked with generating the specified
function based on the given annotations and function signature.
The experimental results are shown in Table 5.
Based on the data in the table, we can observe that while pro-
viding available functions brings some improvement to the perfor-
mance (DeepSeek-Coder-33B achieves a Pass@1 score of 19.62% on
Python), these improvements are far smaller than those brought
by providing file content. This indicates that when generating the
code we need, the model is not simply encountering difficulties
inhow to accomplish the expected functionality (since we have
already provided all custom functions). This motivates us to further

MRG-Bench: Evaluating and Exploring the Requirements of Context for Repository-Level Code Generation Conference acronym, Date, Venue
Table 3: Pass@1 and Pass@3 for different models on MRG-Bench.
Model Python Java Go Average
pass@1 pass@3 pass@1 pass@3 pass@1 pass@3 pass@1 pass@3
CodeLLaMA-13B 9.8% 14.6% 7.8% 10.4% 4.5% 6.1% 7.4% 10.4%
DeepSeek-Coder-33B 7.9% 10.6% 5.9% 8.3% 5.3% 7.4% 6.4% 8.8%
StarChat2-15B 7.9% 10.6% 5.9% 9.4% 5.3% 7.4% 6.4% 9.1%
LLaMA-3.1-8B-Instruct 7.3% 8.9% 5.6% 8.9% 1.2% 1.2% 4.7% 6.3%
GPT-4o 8.7% 10.6% 5.6% 8.9% 4.9% 4.9% 6.4% 8.1%
Claude3.5-Sonnet 10.6% 15.9% 7.8% 10.4% 7.3% 11.6% 8.6% 12.6%
Table 4: Pass@1 and Pass@3 Providing In-file Context to Models.
Model Python Java Go Average
Pass@1 Pass@3 Pass@1 Pass@3 Pass@1 Pass@3 Pass@1 Pass@3
CodeLLaMA-13B 13.6% 24.4% 12.3% 21.5% 13.9% 23.9% 13.3% 23.3%
DeepSeek-Coder-33B 25.5% 34.1% 21.5% 26.1% 19.6% 29.4% 22.2% 29.9%
LLaMA-3.1-8B-Instruct 18.1% 22.8% 16.9% 20.0% 13.5% 20.8% 16.2% 21.2%
GPT-4o 33.3% 38.2% 24.6% 29.2% 28.8% 33.7% 28.9% 33.7%
Claude3.5-Sonnet 33.3% 41.5% 29.2% 40.0% 35.0% 47.2% 32.5% 42.9%
Table 5: Performance Comparison with Different Contexts
Different Context Python Java Go Average
Pass@1 Pass@3 Pass@1 Pass@3 Pass@1 Pass@3 Pass@1 Pass@3
callee-funcbody 19.60% 22.80% 11.30% 15.10% 7.50% 10.60% 12.8% 16.2%
callee-signature 13.30% 15.80% 7.60% 10.40% 5.10% 8.30% 8.7% 11.5%
in-file 25.50% 34.10% 21.50% 26.10% 19.60% 29.40% 22.2% 29.9%
Table 6: Pass@1 Performance Comparison of Different Meth-
ods and Models
Method Model Python Java Go
In-file Context Claude3.5-Sonnet 33.3% 29.2% 35.0%
Long ContextClaude3.5-Sonnet 31.7% 33.9% 33.7%
DeepSeek-V2.5 25.2% 32.3% 28.8%
Reasoning ModelDeepseek-R1 34.15 % 33.33% 39.26 %
O3-mini 34.15 %38.54 % 32.52%
QWQ-32B 13.01% 15.62% 16.56%
investigate the model’s erroneous results to explore what repository
context the model truly needs during generation.
Takeaway-4: Providing available functions yields less improve-
ment than providing in-file context, suggesting that the model’s diffi-
culties lie not simply in how to accomplish the expected functional-
ity, but rather in other underlying issues.
In order to further explore the optimal performance of the current
models, we conducted experiments on longer context window and
stronger reasoning models. For long context models, we tested the
optimal open-source model, DeepSeek-V3, and the optimal closed-
source model, Claude3.5-Sonnet. DeepSeek’s official API supports
a 128K token context window. Therefore, we concatenated all code
from the folder containing the target function into a single context,
which was then input into the model for generation. To ensure con-
sistency between the two models, we standardized the context sizeto 100K tokens, truncating any data exceeding this length. The re-
sults for both models are presented in Table 6. For reasoning models,
we select O3-mini, Deepseek-R1 [ 3] and Qwen2.5-QWQ-32B [ 25].
We use the same in-file context in RQ2.1 for the experiments of
reasoning models. The results are shown in Table 6.
As shown in the data, while the performance of long-context
models surpasses that of In-file-context method, the improvement
is not substantial. For instance, Claude3.5-Sonnet get worse per-
formance on Python and Go with longer context. This suggests
that, although longer contexts can provide more useful informa-
tion, they may also introduce additional noise. A more promising
approach may be to selectively provide different contexts tailored
to the requirements of each programming language.
Furthermore, our experiments reveals that reasoning-enhanced
models, when combined with in-file contextual information, demon-
strate superior performance. Specifically, DeepSeek-R1 and O3-mini
achieve state-of-the-art results on MRG-Bench. However, even the
most advanced large language models (LLMs) currently available
exhibit a Pass@1 accuracy below 40%, underscoring the significant
challenge MRG-Bench poses to current LLMs.
Notably, QWQ-32B underperforms compared to DeepSeek-Coder-
33B, despite sharing the same parameter scale. We hypothesize
that this performance gap may stem from QWQ-32B’s reasoning-
oriented training, which could compromise its foundational code
generation capabilities.

Conference acronym, Date, Venue Haiyang Li, Qing Gao, and Shikun Zhang
Takeaway-5: While long-text models perform better than the
In-file-context method on MRG-Bench, the improvements are not
substantial. The reasoning models achieved the highest performance,
yet none surpassed 40% in Pass@1 accuracy, demonstrating that MRG-
Bench remains a highly challenging benchmark for state-of-the-art
models.
5.2.2 RQ2.2: How do retrieval-augmented generation (RAG) ap-
proaches perform on MRG-Bench when supplying dynamically re-
trieved context? Repository-level code generation can be catego-
rized as a Retrieval-Augmented Generation (RAG) task, wherein
the model retrieves relevant information to support the generation
process. To evaluate the performance of various methods on MRG-
Bench, we implemented several basic RAG techniques. Specifically,
we designed four sets of experiments: (1) the RAG method using
BM25 retriever, (2) the RAG method using embedding-based re-
triever, (3) a mixed RAG method combining both above retrieval
techniques, and (4) Repocoder, a code generation method designed
specifically for repository-level tasks. For all methods, we employed
DeepSeek-Coder-33B as the generation model, while the BGE-M3
model, recognized for its strong performance, was used for em-
beddings the code snippet. We adopted the code document block
segmentation strategy provided by LangChain, segmenting the
codebase into blocks within 500 tokens with an overlap of 50 to-
kens. During retrieval, we selected the top 5 most relevant code
blocks as the context for generation. For Repocoder, we used the
hyperparameters specified in the paper. The experimental results
are presented in the Table 7.
The results reveal a substantial performance disparity between
RAG-based models across different programming languages. Among
the three languages evaluated, Python achieved the highest per-
formance, while Go performed the worst. For instance, in the
mix-RAG method, the Pass@1 score of Python was double that
of Go. Moreover, when using BM25 as a retriever, pass@1 scores for
all languages were lower compared to using embedding-based re-
trieval. After excluding the BM25 retriever, the performance of the
embedding-only RAG method surpassed that of the mixed retrieval
approach, indicating that BM25 can be replaced by embedding-
based methods without sacrificing performance.
Repocoder, designed for repository-level code generation, did
not outperform the basic RAG method for Python but demonstrated
superior performance in Java and Go, significantly surpassing the
RAG baseline. Notably, Repocoder showed consistent performance
across all three languages, suggesting it does not exhibit language
bias. We attribute this balanced performance to Repocoder’s use
of pre-generated code for retrieve, ensuring that both the query
and retrieved information reside in the same semantic space, thus
improving retrieval accuracy. In contrast, other methods suffer
from inadequate retrieve information, leading to similar issues
as observed in RQ1 and RQ2, with Python outperforming other
languages.
Takeaway-6: Although RAG-related methods achieve certain per-
formance improvements, their effectiveness is significantly inferior to
that of providing in-file context alone. This further suggests that the
primary challenges faced by current methods and the key contextual
You are a professional code analysis expert. I am conducting a repository-
level code generation task experiment. Your task is to help me annotate the
code snippets generated by the model based on the ground truth answers.
We categorize the information required for the model to generate the
necessary code into two types:
1. What : This type of information guides the model on what
functionality the code snippet should implement. The absence of this
information will cause the model-generated results to be
inconsistent with the actual code in terms of code functionality .
2. How : This type of information guides the model on how to generate
the code snippet. The absence of this information will cause the
model-generated results to be inconsistent with the actual code in
terms of code implementation.
Your task is to help me annotate which of the above two types of
information is missing from the model-generated incorrect code snippets,
based on the provided standard answer code segments.
You need to first analyze the two code segments, then provide the final
judgment result in <label></label> and explain the reason.Figure 4: Prompt for label fail cases to How and What.
no-context in-file rag callee
Different Context0.20.30.40.50.60.70.8Rate
Comparison of What vs How rate across Different Languages and Methods
py - what
py - how
go - what
go - how
java - what
java - how
Figure 5: Annotation result of fail reason.
information require further exploration. Notably, RepoCoder demon-
strates exceptional capability in eliminating language bias, which
warrants deeper investigation.
5.3 RQ3:Why do current methods fails on
MRG-Bench?
From the previous two RQs, we observe several intriguing phe-
nomena: on MRG-Bench, RAG algorithms show no significant ad-
vantages, and providing available functions for code generation
offers minimal benefit to models. Instead, simply providing the file
containing the target code proves more effective. To explain these
phenomena, we attempt to annotate the causes of model failures,
offering novel insights into the contextual requirements of models.
To our knowledge, this is the first research effort to explore the
relationship between model failure causes and context types.
Inspired by Takeaway-4 (RQ2.1), we found that providing cus-
tom functions needed to "implement target functionality" yields
limited model improvement, suggesting that generation difficulties
may lie in other perspectives. Following intuitive problem-solving

MRG-Bench: Evaluating and Exploring the Requirements of Context for Repository-Level Code Generation Conference acronym, Date, Venue
Table 7: Pass@1 and Pass@3 of Different RAG Methods Using DeepSeek-Coder-33B on MRG-Bench
RAG Method Python Java Go Average
Pass@1 Pass@3 Pass@1 Pass@3 Pass@1 Pass@3 Pass@1 Pass@3
mix-rag 15.80% 18.40% 11.30% 15.10% 9.80% 14.20% 12.3% 15.9%
bm25-rag 13.30% 17.10% 9.40% 14.20% 7.50% 10.60% 10.1% 14.0%
bge3-rag 17.10% 19.00% 11.30% 16.00% 10.20% 15.00% 12.9% 16.7%
repocoder 15.80% 18.40% 20.80% 21.70% 17.70% 20.10% 18.1% 20.1%
logic, we categorize the information required for models to gen-
erate specified code into two types: (1) What to do - information
that guides models to understand the functional logic required in
the target function, potentially including function definitions, com-
ments, classes, sibling functions, or functions with similar names.
We call this " What information ". (2) How to do - information
that guides models on implementation approaches, such as appro-
priate frameworks or callable custom functions. we call this " How
information ". To understand which type of missing information
causes model failures, we annotate failure cases from the currently
best-performing model, Claude-3.5-Sonnet.
We employ a carefully designed prompt (shown in Figure 4)
and utilize five top-performing models (GPT-4o, Claude-3.5-Sonnet,
Gemini-2.5-Pro, DeepSeek-V3, and Qwen2.5-72B-Instruct) for an-
notation and voting to produce final results, requiring models to
identify failure causes in unsuccessful cases. To maintain annota-
tion stability, all generation processes use temperature=0. To further
ensure annotation reliability, we only retain samples with voting
ratios of 5:0 and 4:1, discarding highly controversial 3:2 cases. The
final retention rate is 86.3%, with most cases receiving consistent
annotations across models.
Our annotation results are shown in Figure 5:
The results reveal commonalities across different languages. First,
over 68% of failures stem from missing "What information" - models
cannot understand what code functionality the user requirements
correspond to within the current repository context. Providing
different contextual information from repositories can effectively
reduce this proportion, helping models understand the working
environment and make more precise judgments. However, these
improvements are quite limited (infile, RAG), suggesting we should
further enhance retrieval techniques for this type of information,
such as repository README documents, feature descriptions, and
specific application scenarios.
Takeaway-7 : Missing "What information" is the primary cause
of current model failures; models cannot accurately understand what
functionality requirement descriptions correspond to within the spe-
cific repository context.
Benefiting from MRG-Bench’s multilingual feature, we find that
different programming languages exhibit varying contextual pref-
erences. Python is particularly notable (recall that Python was also
special in evaluation of RQ2). Python demonstrates a distinctly
high "What Information" tendency. Across different contexts (ex-
cept No Context), Python’s lowest "What" proportion equals the
maximum values of the other two languages. This indicates that in
Python, once models struggle to understand specified requirements,
repository context rarely provides useful information. Combined
with conclusions from RQ1.2, this yields an interesting inference:Python language itself exhibits weaker semantic associations be-
tween functions compared to other languages. Therefore, even
without context, models can achieve good performance (Takeaway-
2, RQ1.2). If models fail, providing context rarely resolves the issue.
Only providing truly needed sub-functionalities (callable functions)
for target functions proves effective, but such context is extremely
difficult to obtain in real applications.
When attempting to improve model performance in practical sce-
narios, for Go and Java, we can follow in-file and RAG approaches
to further mine "What Information" while eliminating ineffective
information, expecting better results. However, for Python, these
approaches are ineffective. We need to start from RepoCoder’s ap-
proach, attempting to use fine-grained information from target
functions to reversely summarize "What information," presenting
new challenges for further improving practical application effec-
tiveness.
Takeaway-8 : Python exhibits markedly different contextual pref-
erences compared to the other two languages. Following RepoCoder’s
pre-generation approach to further mine "What Information" rep-
resents a potentially effective direction. Different languages in real
application scenarios may require different context extraction strate-
gies.
6 Threats to Validity
Our work faces the following threats:
Low Coverage of Languages and Projects. Although we ini-
tially screened 7 languages and 1,400 projects, only 3 languages and
22 projects were retained in the final dataset. This may lead to a vari-
ance in our evaluation results. However, our experiments (RQ1.1)
demonstrate that MRG-Bench, despite encompassing fewer projects,
exhibits a closer alignment with real-world data distributions com-
pared to alternative datasets. In practice, contemporary high-quality
open-source projects typically comprise complex, multi-module
systems containing code segments with diverse functionalities. Our
streamlined selection strategy—which neither imposes restrictions
on function dependencies nor modifies the original project structure
during test environment construction—enables maximal preserva-
tion of source code distributions. In contrast, other datasets incor-
porating more projects often introduce distributional biases during
their construction processes due to sophisticated sampling criteria
and artificial dependency pruning. Even though, we plan to expand
MRG-Bench’s project coverage in future iterations to cover more
functionalities, and expand language of C/C++ and JavaScript by
building isolate docker environment for each project.
Lack of agent based method. We did not include agent-related
methods in the paper because these often require extensive prompt
configuration and involve network search functions, which may

Conference acronym, Date, Venue Haiyang Li, Qing Gao, and Shikun Zhang
cause data leakage and affect the fairness of the evaluation. Besides,
current mainstream open-source agent methods are primarily de-
signed for bug-fixing scenarios in SWE-Bench, making them diffi-
cult to adapt to the generation scenarios in MRG-Bench. We look
forward to the emergence of agent methods specifically designed
for general code generation, and will update the leaderboard with
relevant approaches when they become available.
Table 8: Data Leakage Detection Results
Dataset Model Data Leakage Ratio(CDD)
HumanEval GPT-4o 41.47%
MRG-BenchGPT-4o 7.2%
Claude3.5-Sonnet 7.8%
DeepSeek-Coder-33B 6.0%
CodeLLaMA-13B 4.9%
LLaMA-3.1-8B-Instruct 7.7%
Data Leakage Problem. Data leakage of dataset is one of the
most critical issues in evaluating large language models. We address
this problem from two aspects. First, we select newer repositories
that have less relevant information available online, are less imitated
and cited by other repositories, and with which the models are less
familiar. Second, we use data leakage detection approach CDD [ 6]
to check the dataset leakage ratio (Using default parameters in their
paper). CDD can detect whether LLMs have been trained on specific
benchmarks by compare the generated result in different sampling
tempreture. The detection results are shown in Table 8. Compared
to HumanEval [ 2], the leakage ratio on MRG-Bench is low. Given
that the generated outputs share identical function signatures, doc-
umentation, and partial reuse of public APIs/code elements, it is
inherently challenging to reduce this metric to zero. Based on the
information above and the suboptimal performance of current mod-
els on MRG-Bench, we can conclude that the benchmark exhibits
low susceptibility to data leakage.
7 Related Work
7.1 Large Language Models for Code Generation
The advent of pre-training technology has significantly propelled
the field of code generation, leading to remarkable advancements
in both academia and industry [ 8,21,24]. This surge has resulted
in the emergence of various large language models (LLMs) that
have demonstrated excellent performance in code generation tasks.
Notable examples include Codex [ 23], ChatGPT [ 22], CodeLLaMA
[10], DeepSeek Coder [11], and StarCoder2 [19].
7.2 Code Generation Benchmarks
Early benchmarks for code generation primarily evaluated LLMs on
generating relatively simple Python functions. HumanEval [ 2] and
MBPP [ 1] consist of manually designed programming questions
that provide function signatures, comments, and unit tests. LLMs
are tasked with crafting function bodies based on these inputs, and
their outputs are evaluated by the success in passing the provided
unit tests. However, these benchmarks focus on self-contained
functions with only built-in language dependencies, which do not
fully represent real-world development scenarios [26].To address these limitations, more complex benchmarks have
been developed. APPS [ 12] evaluates code generation on more
challenging competition-style problems. CoderEval [ 26] introduce
non-standalone programs derived from real GitHub projects, align-
ing better with actual development settings that rely on multiple
public libraries and project files.
Multi-task benchmarks like CodeXGLUE [ 20] and XCodeEval
[14] incorporate a broad range of programming questions and tasks,
establishing a comprehensive framework for evaluating LLMs. How-
ever, CodeXGLUE relies on textual similarity metrics such as BLEU
and CodeBLEU [ 5], which may not fully capture the functional
correctness of code. CoderUJB [ 28] fills a critical gap by providing
a benchmark that includes multiple programming tasks that are
executable and match real-world development scenarios.
Recently, benchmarks for repository-level tasks have been pro-
posed. CrossCodeEval [ 4], RepoBench [ 18], and RepoEval [ 29] are
code completion benchmarks that aim to evaluate LLMs on code
completion tasks within repositories. However, they lack necessary
runnable environment, limiting their applicability for code genera-
tion evaluation. SWE-bench [ 13] focuses on repairing repository
issues by revising existing programs rather than generating new
code.
However, existing datasets present several issues, including in-
consistencies with pratical code information, a lack of runnable
test cases, and limited language coverage. To address these short-
comings, we propose MRG-Bench . Our dataset encompasses three
programming languages and exclusively includes data from high-
quality open-source projects. This ensures that MRG-Bench pro-
vides an effective evaluation framework for subsequent repository-
oriented code generation tasks.
8 Conclusion and Future Work
In this paper, we present a code generation evaluation dataset based
on an analysis of high-quality open-source code repositories. Our
dataset exhibits three key characteristics: Practicality: The dataset
is derived from real-world open-source projects, avoiding the use
of artificially constructed data. Multilingualism: The dataset in-
cludes a diverse range of programming languages to enable com-
prehensive evaluations across different languages. Executability:
The primary evaluation criterion for the dataset is that the code
generated by models should be functional in real-world scenarios.
In addition, we conducted a comprehensive evaluation of current
large language models and Retrieval-Augmented Generation (RAG)
methods using MRG-Bench. Our experimental results indicate that
the accuracy of existing large language models in solving real-world
code generation tasks remains low. While providing context can
improve performance, the highest Pass@1 score does not exceed
40%. This highlights the need for further attention to evaluating
large language models in real application scenarios.
Finally, we designed a novel annotation methodology to inves-
tigate the underlying causes of current method failures. Experi-
mental results demonstrate that the primary challenge for current
approaches stems from " understanding user input, i.e., deter-
mining what to do. " The repository context that enhances this ca-
pability exhibits distinct preferences across different programming

MRG-Bench: Evaluating and Exploring the Requirements of Context for Repository-Level Code Generation Conference acronym, Date, Venue
languages, suggesting the need for more diverse context extrac-
tion techniques. Notably, Python demonstrates markedly different
characteristics from other languages in both performance and con-
textual preferences. Therefore, we encourage future research to
expand the scope of programming languages to further explore the
characteristics of both models and languages.
All our data, code, experiment results(generated code and test
log) are public available on Github9
References
[1]Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk
Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le,
et al .2021. Program synthesis with large language models. arXiv preprint
arXiv:2108.07732 (2021).
[2]Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. 2023. Teaching
large language models to self-debug. arXiv preprint arXiv:2304.05128 (2023).
[3]DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning. arXiv:2501.12948 [cs.CL] https://arxiv.org/abs/2501.
12948
[4]Yangruibo Ding, Zijian Wang, Wasi Ahmad, Hantian Ding, Ming Tan, Nihal Jain,
Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth,
et al.2024. Crosscodeeval: A diverse and multilingual benchmark for cross-file
code completion. Advances in Neural Information Processing Systems 36 (2024).
[5]Yihong Dong, Jiazheng Ding, Xue Jiang, Zhuo Li, Ge Li, and Zhi Jin.
2023. CodeScore: Evaluating Code Generation by Learning Code Execution.
arXiv:2301.09043 (Jan. 2023). doi:10.48550/arXiv.2301.09043 arXiv:2301.09043
[cs].
[6]Yihong Dong, Xue Jiang, Huanyu Liu, Zhi Jin, Bin Gu, Mengfei Yang, and Ge
Li. 2024. Generalization or Memorization: Data Contamination and Trustwor-
thy Evaluation for Large Language Models. In Findings of the Association for
Computational Linguistics: ACL 2024 , Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
12039–12050. doi:10.18653/v1/2024.findings-acl.716
[7]Xueying Du, Mingwei Liu, Kaixin Wang, Hanlin Wang, Junwei Liu, Yixuan Chen,
Jiayi Feng, Chaofeng Sha, Xin Peng, and Yiling Lou. 2023. Classeval: A manually-
crafted benchmark for evaluating llms on class-level code generation. arXiv
preprint arXiv:2308.01861 (2023).
[8]Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi,
Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. 2023. InCoder:
A Generative Model for Code Infilling and Synthesis. arXiv:2204.05999 (April
2023). doi:10.48550/arXiv.2204.05999 arXiv:2204.05999 [cs].
[9] GitHub. 2024. Copilot . https://github.com/features/copilot
[10] Wenhan Xiong Grattafiori, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo
Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Syn-
naeve. 2023. Code Llama: Open Foundation Models for Code. arXiv preprint
arXiv:2308.12950 (2023).
[11] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang,
Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al .2024. DeepSeek-Coder: When the
Large Language Model Meets Programming–The Rise of Code Intelligence. arXiv
preprint arXiv:2401.14196 (2024).
[12] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora,
Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al .2021. Mea-
suring coding challenge competence with apps. arXiv preprint arXiv:2105.09938
(2021).
[13] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir
Press, and Karthik Narasimhan. 2023. Swe-bench: Can language models resolve
real-world github issues? arXiv preprint arXiv:2310.06770 (2023).
[14] Mohammad Abdullah Matin Khan, M Saiful Bari, Xuan Long Do, Weishi Wang,
Md Rizwan Parvez, and Shafiq Joty. 2023. xcodeeval: A large scale multilin-
gual multitask benchmark for code understanding, generation, translation and
retrieval. arXiv preprint arXiv:2303.03004 (2023).
[15] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Mem-
ory Management for Large Language Model Serving with PagedAttention. In
Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles .
[16] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi,
Bryan Catanzaro, and Wei Ping. 2024. NV-Embed: Improved Techniques for
Training LLMs as Generalist Embedding Models. arXiv preprint arXiv:2405.17428
(2024).
[17] Jia Li, Ge Li, Xuanming Zhang, Yihong Dong, and Zhi Jin. 2024. EvoCodeBench:
An Evolving Code Generation Benchmark Aligned with Real-World Code
9https://github.com/MRG-Bench/MRG-BenchRepositories. arXiv:2404.00599 (March 2024). http://arxiv.org/abs/2404.00599
arXiv:2404.00599 [cs].
[18] Tianyang Liu, Canwen Xu, and Julian McAuley. 2023. Repobench: Benchmarking
repository-level code auto-completion systems. arXiv preprint arXiv:2306.03091
(2023).
[19] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-
Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei,
et al.2024. Starcoder 2 and the stack v2: The next generation. arXiv preprint
arXiv:2402.19173 (2024).
[20] Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambro-
sio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al .2021.
Codexglue: A machine learning benchmark dataset for code understanding and
generation. arXiv preprint arXiv:2102.04664 (2021).
[21] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou,
Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language
model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474
(2022).
[22] OpenAI. 2024. ChatGPT . https://chatgpt.com/
[23] Luca Pasquini, Stefano Cristiani, Ramón García López, Martin Haehnelt, Michel
Mayor, Jochen Liske, Antonio Manescau, Gerardo Avila, Hans Dekker, Olaf Iwert,
et al.2010. Codex. In Ground-based and Airborne Instrumentation for Astronomy
III, Vol. 7735. SPIE, 957–968.
[24] Sijie Shen, Xiang Zhu, Yihong Dong, Qizhi Guo, Yankun Zhen, and Ge Li. 2022.
Incorporating domain knowledge through task augmentation for front-end
javascript code generation. In Proceedings of the 30th ACM Joint European Software
Engineering Conference and Symposium on the Foundations of Software Engineering .
1533–1543.
[25] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang
Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu,
Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 Technical Report. arXiv
preprint arXiv:2412.15115 (2024).
[26] Hao Yu, Bo Shen, Dezhi Ran, Jiaxin Zhang, Qi Zhang, Yuchi Ma, Guangtai Liang,
Ying Li, Qianxiang Wang, and Tao Xie. 2024. Codereval: A benchmark of prag-
matic code generation with generative pre-trained models. In Proceedings of the
46th IEEE/ACM International Conference on Software Engineering . 1–12.
[27] Daoguang Zan, Bei Chen, Dejian Yang, Zeqi Lin, Minsu Kim, Bei Guan, Yongji
Wang, Weizhu Chen, and Jian-Guang Lou. 2022. CERT: Continual Pre-training on
Sketches for Library-oriented Code Generation. In Proceedings of the Thirty-First
International Joint Conference on Artificial Intelligence, IJCAI-22 , Lud De Raedt
(Ed.). International Joint Conferences on Artificial Intelligence Organization,
2369–2375. doi:10.24963/ijcai.2022/329
[28] Zhengran Zeng, Yidong Wang, Rui Xie, Wei Ye, and Shikun Zhang. 2024.
CoderUJB: An Executable and Unified Java Benchmark for Practical Programming
Scenarios. arXiv preprint arXiv:2403.19287 (2024).
[29] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao,
Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-level code com-
pletion through iterative retrieval and generation. arXiv preprint arXiv:2303.12570
(2023).
[30] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. 2024. CodeAgent: Enhancing
Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level
Coding Challenges. arXiv:2401.07339 (Jan. 2024). http://arxiv.org/abs/2401.07339
arXiv:2401.07339 [cs].
References
[1]Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk
Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le,
et al .2021. Program synthesis with large language models. arXiv preprint
arXiv:2108.07732 (2021).
[2]Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. 2023. Teaching
large language models to self-debug. arXiv preprint arXiv:2304.05128 (2023).
[3]DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning. arXiv:2501.12948 [cs.CL] https://arxiv.org/abs/2501.
12948
[4]Yangruibo Ding, Zijian Wang, Wasi Ahmad, Hantian Ding, Ming Tan, Nihal Jain,
Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth,
et al.2024. Crosscodeeval: A diverse and multilingual benchmark for cross-file
code completion. Advances in Neural Information Processing Systems 36 (2024).
[5]Yihong Dong, Jiazheng Ding, Xue Jiang, Zhuo Li, Ge Li, and Zhi Jin.
2023. CodeScore: Evaluating Code Generation by Learning Code Execution.
arXiv:2301.09043 (Jan. 2023). doi:10.48550/arXiv.2301.09043 arXiv:2301.09043
[cs].

Conference acronym, Date, Venue Haiyang Li, Qing Gao, and Shikun Zhang
[6]Yihong Dong, Xue Jiang, Huanyu Liu, Zhi Jin, Bin Gu, Mengfei Yang, and Ge
Li. 2024. Generalization or Memorization: Data Contamination and Trustwor-
thy Evaluation for Large Language Models. In Findings of the Association for
Computational Linguistics: ACL 2024 , Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
12039–12050. doi:10.18653/v1/2024.findings-acl.716
[7]Xueying Du, Mingwei Liu, Kaixin Wang, Hanlin Wang, Junwei Liu, Yixuan Chen,
Jiayi Feng, Chaofeng Sha, Xin Peng, and Yiling Lou. 2023. Classeval: A manually-
crafted benchmark for evaluating llms on class-level code generation. arXiv
preprint arXiv:2308.01861 (2023).
[8]Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi,
Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. 2023. InCoder:
A Generative Model for Code Infilling and Synthesis. arXiv:2204.05999 (April
2023). doi:10.48550/arXiv.2204.05999 arXiv:2204.05999 [cs].
[9] GitHub. 2024. Copilot . https://github.com/features/copilot
[10] Wenhan Xiong Grattafiori, Alexandre Défossez, Jade Copet, Faisal Azhar, Hugo
Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Syn-
naeve. 2023. Code Llama: Open Foundation Models for Code. arXiv preprint
arXiv:2308.12950 (2023).
[11] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang,
Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al .2024. DeepSeek-Coder: When the
Large Language Model Meets Programming–The Rise of Code Intelligence. arXiv
preprint arXiv:2401.14196 (2024).
[12] Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora,
Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al .2021. Mea-
suring coding challenge competence with apps. arXiv preprint arXiv:2105.09938
(2021).
[13] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir
Press, and Karthik Narasimhan. 2023. Swe-bench: Can language models resolve
real-world github issues? arXiv preprint arXiv:2310.06770 (2023).
[14] Mohammad Abdullah Matin Khan, M Saiful Bari, Xuan Long Do, Weishi Wang,
Md Rizwan Parvez, and Shafiq Joty. 2023. xcodeeval: A large scale multilin-
gual multitask benchmark for code understanding, generation, translation and
retrieval. arXiv preprint arXiv:2303.03004 (2023).
[15] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Mem-
ory Management for Large Language Model Serving with PagedAttention. In
Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles .
[16] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi,
Bryan Catanzaro, and Wei Ping. 2024. NV-Embed: Improved Techniques for
Training LLMs as Generalist Embedding Models. arXiv preprint arXiv:2405.17428
(2024).
[17] Jia Li, Ge Li, Xuanming Zhang, Yihong Dong, and Zhi Jin. 2024. EvoCodeBench:
An Evolving Code Generation Benchmark Aligned with Real-World Code
Repositories. arXiv:2404.00599 (March 2024). http://arxiv.org/abs/2404.00599
arXiv:2404.00599 [cs].
[18] Tianyang Liu, Canwen Xu, and Julian McAuley. 2023. Repobench: Benchmarking
repository-level code auto-completion systems. arXiv preprint arXiv:2306.03091
(2023).
[19] Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-
Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei,
et al.2024. Starcoder 2 and the stack v2: The next generation. arXiv preprint
arXiv:2402.19173 (2024).
[20] Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambro-
sio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al .2021.
Codexglue: A machine learning benchmark dataset for code understanding and
generation. arXiv preprint arXiv:2102.04664 (2021).
[21] Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou,
Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language
model for code with multi-turn program synthesis. arXiv preprint arXiv:2203.13474
(2022).
[22] OpenAI. 2024. ChatGPT . https://chatgpt.com/
[23] Luca Pasquini, Stefano Cristiani, Ramón García López, Martin Haehnelt, Michel
Mayor, Jochen Liske, Antonio Manescau, Gerardo Avila, Hans Dekker, Olaf Iwert,
et al.2010. Codex. In Ground-based and Airborne Instrumentation for Astronomy
III, Vol. 7735. SPIE, 957–968.
[24] Sijie Shen, Xiang Zhu, Yihong Dong, Qizhi Guo, Yankun Zhen, and Ge Li. 2022.
Incorporating domain knowledge through task augmentation for front-end
javascript code generation. In Proceedings of the 30th ACM Joint European Software
Engineering Conference and Symposium on the Foundations of Software Engineering .
1533–1543.
[25] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang,
Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang
Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue,
Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang
Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu,
Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 Technical Report. arXivpreprint arXiv:2412.15115 (2024).
[26] Hao Yu, Bo Shen, Dezhi Ran, Jiaxin Zhang, Qi Zhang, Yuchi Ma, Guangtai Liang,
Ying Li, Qianxiang Wang, and Tao Xie. 2024. Codereval: A benchmark of prag-
matic code generation with generative pre-trained models. In Proceedings of the
46th IEEE/ACM International Conference on Software Engineering . 1–12.
[27] Daoguang Zan, Bei Chen, Dejian Yang, Zeqi Lin, Minsu Kim, Bei Guan, Yongji
Wang, Weizhu Chen, and Jian-Guang Lou. 2022. CERT: Continual Pre-training on
Sketches for Library-oriented Code Generation. In Proceedings of the Thirty-First
International Joint Conference on Artificial Intelligence, IJCAI-22 , Lud De Raedt
(Ed.). International Joint Conferences on Artificial Intelligence Organization,
2369–2375. doi:10.24963/ijcai.2022/329
[28] Zhengran Zeng, Yidong Wang, Rui Xie, Wei Ye, and Shikun Zhang. 2024.
CoderUJB: An Executable and Unified Java Benchmark for Practical Programming
Scenarios. arXiv preprint arXiv:2403.19287 (2024).
[29] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao,
Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-level code com-
pletion through iterative retrieval and generation. arXiv preprint arXiv:2303.12570
(2023).
[30] Kechi Zhang, Jia Li, Ge Li, Xianjie Shi, and Zhi Jin. 2024. CodeAgent: Enhancing
Code Generation with Tool-Integrated Agent Systems for Real-World Repo-level
Coding Challenges. arXiv:2401.07339 (Jan. 2024). http://arxiv.org/abs/2401.07339
arXiv:2401.07339 [cs].
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009