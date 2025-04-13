# A System for Comprehensive Assessment of RAG Frameworks

**Authors**: Mattia Rengo, Senad Beadini, Domenico Alfano, Roberto Abbruzzese

**Published**: 2025-04-10 14:41:34

**PDF URL**: [http://arxiv.org/pdf/2504.07803v1](http://arxiv.org/pdf/2504.07803v1)

## Abstract
Retrieval Augmented Generation (RAG) has emerged as a standard paradigm for
enhancing the factual accuracy and contextual relevance of Large Language
Models (LLMs) by integrating retrieval mechanisms. However, existing evaluation
frameworks fail to provide a holistic black-box approach to assessing RAG
systems, especially in real-world deployment scenarios. To address this gap, we
introduce SCARF (System for Comprehensive Assessment of RAG Frameworks), a
modular and flexible evaluation framework designed to benchmark deployed RAG
applications systematically. SCARF provides an end-to-end, black-box evaluation
methodology, enabling a limited-effort comparison across diverse RAG
frameworks. Our framework supports multiple deployment configurations and
facilitates automated testing across vector databases and LLM serving
strategies, producing a detailed performance report. Moreover, SCARF integrates
practical considerations such as response coherence, providing a scalable and
adaptable solution for researchers and industry professionals evaluating RAG
applications. Using the REST APIs interface, we demonstrate how SCARF can be
applied to real-world scenarios, showcasing its flexibility in assessing
different RAG frameworks and configurations. SCARF is available at GitHub
repository.

## Full Text


<!-- PDF content starts -->

arXiv:2504.07803v1  [cs.CL]  10 Apr 2025A System for Comprehensive Assessment of RAG
Frameworks
Mattia Rengo , Senad Beadini , Domenico Alfano , Roberto Abbruzzese
R&D Department, Eustema S.p.A.
Napoli, Italy
{m.rengo, s.beadini, d.alfano, r.abbruzzese }@eustema.it
Code : https://github.com/Eustema-S-p-A/SCARF
Abstract —Retrieval Augmented Generation (RAG) has
emerged as a standard paradigm for enhancing the factual accu-
racy and contextual relevance of Large Language Models (LLMs)
by integrating retrieval mechanisms. However, existing evaluation
frameworks fail to provide a holistic black-box approach to
assessing RAG systems, especially in real-world deployment
scenarios. To address this gap, we introduce SCARF (System for
Comprehensive Assessment of RAG Frameworks), a modular and
flexible evaluation framework designed to benchmark deployed
RAG applications systematically. SCARF provides an end-to-
end, black-box evaluation methodology, enabling a limited-effort
comparison across diverse RAG frameworks. Our framework
supports multiple deployment configurations and facilitates au-
tomated testing across vector databases and LLM serving strate-
gies, producing a detailed performance report. Moreover, SCARF
integrates practical considerations such as response coherence,
providing a scalable and adaptable solution for researchers
and industry professionals evaluating RAG applications. Using
the REST APIs interface, we demonstrate how SCARF can
be applied to real-world scenarios, showcasing its flexibility in
assessing different RAG frameworks and configurations. SCARF
is available at GitHub repository.
Index Terms —Retrieval-Augmented Generation, RAG Evalu-
ation, LLM
I. I NTRODUCTION
Retrieval-Augmented Generation (RAG) represents a re-
markable advancement in Natural Language Processing
(NLP), significantly enhancing the performance of generative
Language Models (LMs). By combining the capabilities of
LMs with external knowledge bases, RAG allows responses
that are not only more accurate but also contextually relevant.
Merging information retrieval with language generation, RAG
systems address a major limitation of standalone LMs: their
tendency to generate responses that, while coherent, may
lack factual accuracy or grounding. Since traditional Large
Language Models (LLMs) rely solely on pre-trained data, they
may generate factually incorrect information and unreliable
outputs. RAG systems address this limitation by retrieving
relevant information in real-time, making them well-suited for
tasks that require up-to-date, accurate, and context-aware re-
sponses, such as answering questions, generating content, and
supporting various real-world applications. RAG approaches[1] have evolved rapidly, resulting in a wide variety of system
variants [2]–[5] and benchmarking frameworks [6]. Many
existing evaluation methods [7], [8] focus on assessing specific
components, such as the relevance of retrieved documents or
the quality of generated responses. However, these methods
often lack a comprehensive perspective, failing to provide
a holistic, end-to-end evaluation that considers not only the
interplay between the retrieval and generation components
but also the flexibility to experiment with and optimize the
underlying technical tools used in these processes. Another
critical yet often overlooked aspect is the variability introduced
by different deployment frameworks used for LLMs. Tools
such as vLLM [9], OpenLLM [10], and Ollama [11] im-
plement diverse optimization strategies, including techniques
like quantization, batching, and caching. These strategies can
have a significant impact on key performance metrics, such as
latency, system efficiency, and overall responsiveness. These
variations are particularly important in real-world scenarios,
where deployment choices can directly affect the user experi-
ence and system scalability. Despite their importance, existing
evaluation methods rarely integrate the ability to manage and
measure the influence of these deployment frameworks sys-
tematically. To address these limitations, we propose SCARF,
a low-level evaluation framework designed to assess RAG
systems comprehensively while maintaining extreme modu-
larity. This framework, in addition to offering a wide range
of capabilities, also enables an easy replacement of individual
system components, such as testing various vector databases
for the retrieval phase or employing different LLM deployment
frameworks in the generation phase. By providing this flexibil-
ity, our framework supports targeted experiments to understand
the impact of specific system changes, including deployment
strategies and their optimizations. Another strength of our
approach is its ability to generate a detailed final report that
provides test results of the RAG pipeline. This report allows
for precise measurement of system performance in specific
scenarios, such as addressing particular types of questions or
solving specialized tasks. Moreover, our evaluation framework
extends beyond traditional metrics by leveraging state-of-the-
art LLM-based evaluation methods [12], including RAGAS
[13]. To summarize, our framework offers the following key

advantages:
1)Easy to integrate and use : Researchers and practition-
ers can set up and evaluate RAG systems with minimal
configuration.
2)Highly modular and customization : Our framework
supports modular component replacement, enabling sys-
tematic experimentation.
3)Comprehensive and insightful : By producing a final
report, the framework enables a deep understanding
of system performance. The report provides valuable
insights that can guide to hyper-params tuning and
improvements for specific use cases or scenarios.
II. B ACKGROUND AND RELATED WORK
The evaluation of LLMs and RAG systems necessitates
specialized frameworks capable of assessing both retrieval
performance—such as relevance and recall—and generation
quality, including coherence and factual accuracy. In this
section, we critically examine representative solutions, empha-
sizing their approaches to data handling, metric evaluation, and
deployment scenarios.
Evaluating RAG systems is a complex task due to their
dual nature of retrieving relevant information and generating
coherent responses [6]. Traditional evaluation methods for
language models, such as perplexity or human annotations, are
often inadequate for capturing the nuanced interplay between
these two components. For example, specialized metrics are
needed to assess how effectively the retrieved context supports
the generation process and ensures the accuracy and relevance
of the output. Recent research has increasingly leveraged
LLM-as-a-judge [12] techniques to develop advanced evalua-
tion metrics for retrieval-augmented generation pipelines. [18]
proposed GPT-Score that leverages the generative capabilities
of pre-trained language models—such as GPT-3 [19] or GPT-
4 [20]—to assess the quality of a generated text. RAGAS [13]
further refines these approaches by defining three specialized
metrics specifically designed for RAG applications: faithful-
ness, answer relevance, and context relevance.
Faithfulness [13] measures the consistency of the generated
response with the retrieved context. It determines whether
the claims made in the response are substantiated by the
retrieved passages. A response that aligns closely with the
context receives a high faithfulness score, while deviations
or hallucinations result in lower scores. Within RAGAS,
faithfulness is assessed using a structured scoring system,
allowing evaluators—both human and automated—to rate the
degree of alignment between the generated content and the
supporting context. This metric is critical to ensuring that RAG
systems produce factual and trustworthy outputs.
Answer relevancy [13] evaluates how well the generated
response addresses the user’s query. This metric is essential for
ensuring that the response is not only accurate but also directly
relevant to the user’s needs. In the RAGAS framework, answer
relevance is assessed by analyzing the relationship between the
query’s intent and the content of the response. High scores in
answer relevance reflect responses that provide meaningful andprecise information, making this metric central to determining
the usefulness of RAG outputs.
Context relevancy [13] focuses on the quality and speci-
ficity of the information retrieved by the system. It examines
whether the retrieved passages are relevant to the input query
and sufficiently focused to support accurate and coherent
response generation. High context relevance scores indicate
that the retrieved information is appropriate and effectively
contributes to the final response. This metric is pivotal for
evaluating the performance of the retrieval component within
RAG systems, as it directly influences the overall quality of
the output.
Other common metrics in NLP, such as ROUGE [21] and
BLEU [22], can be applied to evaluate certain properties of
RAG-generated outputs. These metrics specifically measure
the overlap between generated outputs and reference answers,
making them suitable for assessing lexical similarity. However,
their reliance on overlap-based evaluation limits their ability
to capture other aspects of RAG performance, which are better
addressed by the specialized metrics provided by RAGAS.
A. Key Feature of RAG Evaluators
In this section, we systematically review and analyze the
essential characteristics that any framework designed to eval-
uate framework RAG systems should possess. Our objective
is to dissect the key features of these tools and elucidate how
they overcome the challenges of assessing RAG’s retrieval and
generation components. The main features are presented in
Table I. Below, we provide a brief explanation of each column:
•Retrieval Metrics: indicates whether the framework sup-
ports the evaluation of the retrieval phase by measuring
metrics such as recall, precision, or relevance of the
retrieved documents. Tools that can directly score this
stage of an RAG pipeline or generate retrieval-centric
metrics are noted here.
•Generation Metrics: reflects the framework’s capability
to assess the generated text’s quality. This includes stan-
dard metrics (e.g. BLEU [22], ROUGE [21]) as well as
specialized LLM-based metrics.
•Synthetic Data Gen: denotes whether the tool is
equipped to automatically generate new test data or
augment existing datasets—typically by creating new
question-answer pairs from a given knowledge base. This
feature helps extend evaluations to new domains without
requiring extensive manual annotation.
•Multi-RAG Testing: specifies if the tool can simulta-
neously compare or evaluate multiple RAG solutions.
This multi-system testing allows for side-by-side bench-
marking under consistent conditions, thereby facilitating
comprehensive performance comparisons.
•External RAG Support: determines whether the frame-
work is capable of interfacing with fully deployed, third-
party RAG endpoints. We call this also “black-box”
testing capability. This feature is essential when direct
access to internal components is not feasible.

TABLE I: Comparison of Frameworks. This comparison highlights the strengths and limitations of each framework, providing
insight into their capabilities and identifying areas where improvements or extensions may be needed. The table presents
an overview of the different frameworks and their support for key features. A ✗symbol indicates that the feature is not
implemented in the respective framework. A ✓signifies full support, meaning the feature is fully integrated and functional
without limitations. The ⊖symbol represents partial support, meaning the feature is present but lacks completeness.
FrameworkRetrieval
MetricsGeneration
MetricsSynthetic
Data GenMulti-RAG
TestingExternal
RAG
SupportConfig &
Auto
TestingAPI
Integration
Langchain Bench [8] ⊖ ✓ ✗ ✓ ✗ ✗ ✗
RAG Evaluator [14] ⊖ ✓ ✗ ✗ ✗ ✗ ✗
Giskard (RAGET) [15] ⊖ ✓ ✓ ✗ ✗ ✗ ✗
Rageval [7] ✓ ✓ ✗ ✗ ✗ ✗ ✗
Promptfoo [16] ✗ ✓ ✗ ✗ ✗ ✗ ✗
ARES [17] ✓ ✓ ✓ ⊖ ✗ ✓ ✗
SCARF (ours) ✓ ✓ ✗ ✓ ✓ ✓ ✓
•Config & Auto Testing: indicates if the framework
supports a configuration-based or script-based approach
that enables automatic execution of tests—including tasks
such as data uploads and query submission—without
requiring extensive manual intervention.
•API Integration: this column highlights whether the
framework is designed to integrate via standard APIs,
thus enhancing interoperability and efficiency in real-
world deployments. In industrial settings, systems are fre-
quently exposed through standardized APIs (e.g., REST),
thus this feature may be essential for the usability of a
framework.
B. Related Frameworks
In this section, we examine several promising tools for RAG
evaluation, assessing each one based on the key features de-
scribed in the preceding discussion. By systematically analyz-
ing how each tool addresses these criteria, we highlight their
respective strengths, limitations, and potential gaps—thereby
providing a clear perspective on the current state of RAG
evaluation practices.
RAG evaluator [14] is a Python library that supports a
broad set of text-generation metrics like BLEU [22], ROUGE
[21], Bert-Score [23], METEOR [24] and can detect certain
bias or hate-speech aspects. This makes it straightforward to
evaluate generation quality on a precollected dataset. However,
it does not inherently integrate a procedure for evaluating
external RAG systems; users typically provide the generated
text and references offline rather than hooking into a live RAG
framework deployment.
Giskard [15] offers a comprehensive testing and scanning
approach for AI systems, including LLMs and RAG. Its
RAG Evaluation Toolkit (RAGET) can auto-generate synthetic
question-and-answer pairs from a knowledge base, perform
correctness checks, and detect harmful or biased content. How-
ever, it primarily assumes local or in-house RAG integration,
focusing on diagnosing and improving the pipeline rather than
systematically comparing different third-party RAG frame-works. Rageval [7] is a fine-grained evaluator that splits RAG
into multiple subtasks (query rewriting, document ranking,
evidence verification, etc.), each with dedicated metrics. It
can distinguish between “answer correctness” (comparing to
ground truth) and “groundedness” (checking alignment with
retrieved passages). Like RAG Evaluator, it is primarily used
offline and may require additional scripts to interact with a de-
ployed RAG endpoint. Promptfoo [16] is a developer-focused
CLI that emphasizes rapid iteration over prompts, comparing
outputs from different LLMs or prompt variants. It includes
a “red teaming” mode to detect potential vulnerabilities (e.g.
prompt injections), making it valuable for improving prompt
design. Promptfoo, however, does not offer built-in retrieval
metrics and is not designed for direct integration with large-
scale, black-box RAG frameworks. Langchain Benchmark
[8] is an open-source tool designed to evaluate various tasks
related to LLMs. Organized around end-to-end use cases, it
heavily utilizes LangSmith for its benchmarking processes.
Nevertheless, it primarily focuses on local development en-
vironments and does not facilitate RAG external black-box
testing. Consequently, users must set up and manage evalu-
ations within their own infrastructure, and the benchmarking
process is not fully automated.
Table I provides a comprehensive comparison of the existing
solutions across the key characteristics discussed in the previ-
ous section. While each tool has distinct strengths—focusing
on specific aspects of the RAG pipeline—there exists a sig-
nificant gap in the landscape for a more unified, low-level
“black-box” evaluator. Such an evaluator should facilitate:
•Seamless interaction with fully deployed, third-party
RAG services via APIs, encompassing essential capabil-
ities such as file uploads, query handling, and inference
evaluation.
•Effortless flexibility in replacing essential compo-
nents—including vector databases and LLM serving en-
gines (e.g. Ollama, vLLM)—or even entire RAG frame-
works, all with minimal configuration overhead and no
disruption to the overall evaluation process.

•The ability to provide systematic, granular metrics that
thoroughly assess the entire RAG pipeline, encompassing
critical aspects such as also response latency and resource
consumption.
III. D ESCRIPTION OF THE FRAMEWORK
In this section, we discuss SCARF, our evaluation frame-
work designed to address existing limitations in RAG eval-
uators. At its core, SCARF offers a modular Python-based
suite that treats RAG platforms as interchangeable ”plugins”,
allowing comprehensive end-to-end evaluations in realistic
deployment scenarios. We begin by describing the architecture
of SCARF, highlighting its main components and how they
interoperate to support flexible testing. Next, we provide a
step-by-step guide on using SCARF, illustrating how it can
be leveraged into existing pipelines to evaluate RAG systems.
This design enables researchers and practitioners to fill the
current gaps in the evaluation of RAG by facilitating more
scalable and adaptable testing on diverse platforms and con-
figurations.
A. Framework Architecture
SCARF is designed following a plug-and-play principle,
enabling systematic evaluation of different deployed platforms
without modifying their core implementation . Figure 1 clearly
illustrates the architecture, detailing the distinct functional
blocks organized within the project’s repository. The reposi-
tory is structured into four primary sections. The SCARF Core
includes core testing scripts and configuration files that specify
necessary test datasets and queries required for conducting
evaluations. In addition, SCARF Modules and APIs are pro-
vided, containing dedicated API adapter modules specifically
tailored for integration with various RAG platforms. Another
part of the repository consists of Docker Compose files and
configuration resources necessary for deploying supported
RAG frameworks and for integrating both local and remote
LLM engines or vector databases. Although these resources
are provided, SCARF does not automatically manage the
deployment of these services. Lastly, a dedicated Configuration
Settings area provides comprehensive settings and configu-
rations, enabling users to easily manage different evaluation
scenarios and customize the evaluation processes according to
their specific testing requirements.
Thus SCARF, highlighting its ability to:
•Interact with multiple frameworks as black boxes:
users can test RAG frameworks (e.g. AnythingLLM [25]
, CheshireCat [26]) without having to replicate or fully
understand their internal workings , simply by wrapping
them with a module that exposes consistent methods for
uploading data and querying if such adapter is not already
provided by SCARF.
•Leverage different vector databases: SCARF supports
quick reconfiguration of the underlying vector database
when the remote RAG permits such changes. Thanks to
thevectorDB-local-providers , users can seam-
lessly switch to any local vector database (e.g. Qdrant[27], Milvus [28]) with minimal updates to a Docker
Compose file. This ensures minimal overhead when
adapting to different storage solutions and maintains
SCARF’s goal of providing a flexible and extensible
evaluation environment.
•Use local or remote LLM providers: within
llm-local-providers , users have access to all nec-
essary components for executing local LLM inference
using engines such as Ollama, vLLM, or alternative im-
plementations with Docker Compose. Additionally, they
can configure a remote API (e.g., OpenAI, Anthropic,
OpenRouter) to interface with their preferred models.
•Test and compare frameworks: a single Python entry
point can spin up multiple tests, collecting results and
saving them in standard .csv or.json formats. This
architecture enables SCARF to efficiently assess and
compare different RAG frameworks, assisting users in
identifying the most suitable solution for their specific
use case.
•Ability to delve into metrics at different levels:
SCARF outputs per-question results, including text re-
sponses, expected answers, and metadata. An optional
EvaluatorGPT module can measure correctness or
consistency (using LLM-as-a-judge approach [12]). Users
may also integrate other specialized evaluators due to the
modular nature of SCARF.
SCARF CoreRAG CoreEvaluatorLocal / Remote LLM
Inference
Retrieval
Vector DBSCARF 
Modules / APIsSCARF 
SCARF settingsConfiguration,
{Q, Exp A}
EmbedderRAG Framework
Local /
Remote LLM
Fig. 1: High-level SCARF architecture showing modular in-
tegration points for RAG frameworks, vector databases, and
LLM engines.
B. Scenarios
SCARF supports various testing scenarios to accommodate
different user goals:
1)Evaluating a Single RAG Framework :Users may want
to validate that a particular RAG framework correctly retrieves
and generates answers from a given dataset. They may also
experiment with different parameters (e.g. model temperature,
retrieval thresholds, embedder, vectordb, LLM provider) to
optimize performance for their own datasets. In this scenario,
the system:
•connect to the framework’s endpoint.

•uploads documents to the RAG knowledge base through
the framework API.
•executes a set of queries (generic or file-specific) and
saves responses.
•optionally runs an evaluator module (like
evaluator_gpt.py ) to assess correctness, relevancy,
or other NLP metrics.
2)Comparing Multiple RAG Frameworks on the Same
Dataset :SCARF also supports comparisons among multi-
ple frameworks (any RAG framework for which an adapter
module is available or has been written by a user). SCARF
runs identical test queries against each deployed framework in
sequence, then it combines the results into a single .csv or
.json . This simplifies questions like:
•Cross-framework analysis: e.g. “Which RAG frame-
work, with these specific settings, is the most accurate
in domain X?’
•Performance benchmarks: “Which approach yields the
fastest response with the same hardware or number of
documents?”
IV. F RAMEWORK IN ACTION
This section provides a detailed look at how SCARF carries
out its end-to-end evaluation processes in real-world scenarios.
It highlights key points that facilitate the testing of multiple
RAG platforms.
A. SCARF workflow
In Fig. 2, we show the workflow of our framework.
The main entry point for SCARF evaluations is the script
test_rag_frameworks.py which orchestrates all the
procedures. Below, we summarize the high-level stages of
SCARF’s operational flow:
1) The system reads the input configuration to determine
which files should be uploaded, identifies the specific
queries that need to be executed, and selects the appro-
priate files from the knowledge base to ingest into the
RAG platform.
2) Checking command-line flags (e.g. --apikey ) to de-
termine which framework(s) to target and any necessary
credentials.
3) Dynamically loading a corresponding API adapter (e.g.
CheshireCatAPI ,AnythingLLMAPI ).
4) Submitting queries to each RAG framework sequentially
and collecting responses in memory via Modules/API.
5) Optionally calling EvaluatorGPT to produce auto-
mated scores or annotations for each answer. Finally,
SCARF saves both raw responses and computed evalu-
ation metrics in standardized formats (e.g., csv).
6) Output Export: .csv ,.json . This step consolidates all
data, ensures compatibility with common data process-
ing tools (e.g., Pandas, Excel), and provides a complete
snapshot of the experimental run.
Algorithm 1 presents a more detailed and comprehensive
description of these steps in the form of pseudocode, providing
further clarity on the procedural aspects involved.B. Writing an adapter Module for a Custom RAG Framework
SCARF is designed to be highly extensible , recognizing that
practitioners may need to evaluate emerging or proprietary
RAG solutions. Developers can integrate any system by creat-
ing a SCARF-compliant adapter module and placing it in the
modules/ directory. Each adapter must implement two key
methods:
•upload_document(file_path: str) ->
Dict[str, Any] : Responsible for adding a local
file to the framework’s knowledge base. In some RAG
systems, this may involve splitting the file into smaller
chunks before embedding and indexing. SCARF captures
any returned metadata (e.g., document IDs, potential
error messages).
•send_message(message: str) ->
Dict[str, Any] : Submits a text-based query to
the RAG endpoint and captures both the raw text
response and any auxiliary diagnostic data (e.g., top-
ranked document identifiers, partial token sequences).
After placing this new module (e.g. mynewrag_api.py )
in the modules/ folder, you can specify it in
test_rag_frameworks.py or via command-line
flags. After integrating your custom module, you can interact
with your deployed RAG platform and begin running tests,
queries, and performance evaluations. Moreover, once you
have refined your queries and expected responses, you have
the chance to modify default metrics.
SCARFRAG FrameworkSCARF Flow
Knowledge
BaseInput
EvaluatorCore
Intermediate
Report
RAG FrameworkModules
/ APIs
Final
ReportOutputConfiguration
{Q, Exp A}
Fig. 2: Flow showing how SCARF interact with data and RAG
frameworks to produce the output.

Algorithm 1 SCARF Workflow
1:Input: Configuration file ( config.json ), command-
line arguments, and optional API key.
2:Output: Evaluation results saved as CSV and JSON files.
3:Begin:
4:Load configuration from config.json .
5:for all selected RAG frameworks do
6: Dynamically load the corresponding API adapter.
7: for all warmup queries (not associated with a docu-
ment) do
8: response ←send_message(warmup_query)
9: Append the warmup response and metadata to the
results list.
10: end for
11: for all document in the dataset do
12: upload_document(document_path)
13: for all queries associated with the current document
do
14: response ←send_message(query)
15: Save the response
16: end for
17: end for
18:end for
19:ifevaluation mode is enabled then
20: CallEvaluatorGPT for each response.
21: Merge evaluation scores with the raw responses.
22: Save the aggregated results to test_results.csv
ortest_results.json .
23:end if
24:End.
V. C ONCLUSION AND LIMITATIONS
In this technical report, we presented SCARF, a highly
flexible and modular evaluation framework for RAG systems.
Unlike many existing tools, which often focus on single
components or assume local integration, SCARF operates at
a “black-box” level. It can connect to any already-deployed
RAG solution through a minimal adapter module, making it
easy for researchers and practitioners to assess multiple RAG
frameworks side by side on the same dataset, enabling direct
comparisons of many performance indicators.
Our approach complements the capabilities of existing RAG
evaluation frameworks, which may focus more narrowly on
metrics for retrieval or generation. SCARF allows users to
benchmark a range of real-world scenarios—from single-
framework tuning to large-scale, multi-framework using a
single, consistent interface. This modularity is particularly
valuable in environments where organizations need to validate
not only the quality of the model but also determine which
RAG framework is the most suitable for their specific use
case and data.
Despite its flexibility, SCARF has some limitations that
could be addressed in future iterations. Currently, SCARFrequires users to manually provide queries for evaluation,
which can be time-consuming and may introduce biases in the
assessment process. A promising direction for improvement is
the integration of synthetic query generation, allowing SCARF
to create diverse test cases and reducing human intervention.
Additionally, while SCARF supports a range of metrics,
future versions could incorporate additional evaluation criteria,
such as system response time, latency, stability under load, and
scalability, to provide a more holistic performance assessment
of different RAG frameworks.
Another area for improvement is the visualization and user
experience. Currently, SCARF primarily focuses on providing
numerical results and logs. The development of a graphical
user interface (GUI) and a dedicated control panel would
greatly enhance the user experience, making it easier to
navigate through results, compare frameworks visually, and
explore detailed insights across multiple experiments.
As the RAG landscape continues to evolve, we believe
SCARF’s ability to integrate seamlessly with emerging tools
will remain a key advantage. By addressing these limitations
and expanding its capabilities, SCARF can become an even
more comprehensive and user-friendly framework for evaluat-
ing RAG systems.
REFERENCES
[1] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,” arXiv preprint arXiv:2312.10997 , 2023.
[2] A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi, “Self-rag: Learning
to retrieve, generate, and critique through self-reflection,” arXiv preprint
arXiv:2310.11511 , 2023.
[3] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “Interleav-
ing retrieval with chain-of-thought reasoning for knowledge-intensive
multi-step questions,” arXiv preprint arXiv:2212.10509 , 2022.
[4] S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Milli-
can, G. B. Van Den Driessche, J.-B. Lespiau, B. Damoc, A. Clark et al. ,
“Improving language models by retrieving from trillions of tokens,” in
International conference on machine learning . PMLR, 2022, pp. 2206–
2240.
[5] J. Deng, L. Pang, H. Shen, and X. Cheng, “Regavae: A retrieval-
augmented gaussian mixture variational auto-encoder for language mod-
eling,” arXiv preprint arXiv:2310.10567 , 2023.
[6] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu, “Evaluation of
retrieval-augmented generation: A survey,” in CCF Conference on Big
Data . Springer, 2024, pp. 102–120.
[7] K. Zhu, Y . Luo, D. Xu, R. Wang, S. Yu, S. Wang, Y . Yan, Z. Liu,
X. Han, Z. Liu et al. , “Rageval: Scenario specific rag evaluation dataset
generation framework,” arXiv preprint arXiv:2408.01262 , 2024.
[8] langchain ai, “langchain-benchmarks,” https://langchain-ai.github.io/
langchain-benchmarks/notebooks/retrieval/langchain docs qa.html,
2025, accessed: 2025-02-09.
[9] W. Kwon, Z. Li, S. Zhuang, Y . Sheng, L. Zheng, C. H. Yu, J. Gonzalez,
H. Zhang, and I. Stoica, “Efficient memory management for large
language model serving with pagedattention,” in Proceedings of the 29th
Symposium on Operating Systems Principles , 2023, pp. 611–626.
[10] bentoml, “Openllm,” https://github.com/bentoml/OpenLLM, 2025, ac-
cessed: 2025-02-15.
[11] ollama, “ollama,” https://github.com/ollama/ollama, 2025, accessed:
2025-02-15.
[12] L. Zheng, W.-L. Chiang, Y . Sheng, S. Zhuang, Z. Wu, Y . Zhuang, Z. Lin,
Z. Li, D. Li, E. Xing et al. , “Judging llm-as-a-judge with mt-bench
and chatbot arena,” Advances in Neural Information Processing Systems ,
vol. 36, pp. 46 595–46 623, 2023.
[13] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “Ragas: Au-
tomated evaluation of retrieval augmented generation,” arXiv preprint
arXiv:2309.15217 , 2023.

[14] A. Anytime, “rag-evaluator,” https://github.com/AIAnytime/
rag-evaluator, 2025, accessed: 2025-02-09.
[15] Giskard-AI, “giskard,” https://github.com/Giskard-AI/giskard, 2025, ac-
cessed: 2025-02-09.
[16] promptfoo, “promptfoo,” https://github.com/promptfoo/promptfoo,
2025, accessed: 2025-02-09.
[17] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “Ares: An
automated evaluation framework for retrieval-augmented generation
systems,” arXiv preprint arXiv:2311.09476 , 2023.
[18] J. Fu, S.-K. Ng, Z. Jiang, and P. Liu, “Gptscore: Evaluate as you desire,”
arXiv preprint arXiv:2302.04166 , 2023.
[19] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al. , “Language mod-
els are few-shot learners,” Advances in neural information processing
systems , vol. 33, pp. 1877–1901, 2020.
[20] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman,
D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat et al. , “Gpt-4
technical report,” arXiv preprint arXiv:2303.08774 , 2023.
[21] C.-Y . Lin, “Rouge: A package for automatic evaluation of summaries,”
inText summarization branches out , 2004, pp. 74–81.
[22] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “Bleu: a method for
automatic evaluation of machine translation,” in Proceedings of the 40th
annual meeting of the Association for Computational Linguistics , 2002,
pp. 311–318.
[23] T. Zhang, V . Kishore, F. Wu, K. Q. Weinberger, and Y . Artzi, “Bertscore:
Evaluating text generation with bert,” arXiv preprint arXiv:1904.09675 ,
2019.
[24] S. Banerjee and A. Lavie, “Meteor: An automatic metric for mt evalua-
tion with improved correlation with human judgments,” in Proceedings
of the acl workshop on intrinsic and extrinsic evaluation measures for
machine translation and/or summarization , 2005, pp. 65–72.
[25] Mintplex-Labs, “anything-llm,” https://github.com/Mintplex-Labs/
anything-llm, 2025, accessed: 2025-02-15.
[26] cheshire-cat ai, “core,” https://github.com/cheshire-cat-ai/core, 2025,
accessed: 2025-02-15.
[27] qdrant, “qdrant,” https://github.com/qdrant/qdrant, 2025, accessed: 2025-
02-15.
[28] milvus, “milvus,” https://github.com/milvus-io/milvus, 2025, accessed:
2025-02-15.