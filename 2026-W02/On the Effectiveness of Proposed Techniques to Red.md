# On the Effectiveness of Proposed Techniques to Reduce Energy Consumption in RAG Systems: A Controlled Experiment

**Authors**: Zhinuan, Guo, Chushu Gao, Justus Bogner

**Published**: 2026-01-05 19:50:49

**PDF URL**: [https://arxiv.org/pdf/2601.02522v1](https://arxiv.org/pdf/2601.02522v1)

## Abstract
The rising energy demands of machine learning (ML), e.g., implemented in popular variants like retrieval-augmented generation (RAG) systems, have raised significant concerns about their environmental sustainability. While previous research has proposed green tactics for ML-enabled systems, their empirical evaluation within RAG systems remains largely unexplored. This study presents a controlled experiment investigating five practical techniques aimed at reducing energy consumption in RAG systems. Using a production-like RAG system developed at our collaboration partner, the Software Improvement Group, we evaluated the impact of these techniques on energy consumption, latency, and accuracy.
  Through a total of 9 configurations spanning over 200 hours of trials using the CRAG dataset, we reveal that techniques such as increasing similarity retrieval thresholds, reducing embedding sizes, applying vector indexing, and using a BM25S reranker can significantly reduce energy usage, up to 60% in some cases. However, several techniques also led to unacceptable accuracy decreases, e.g., by up to 30% for the indexing strategies. Notably, finding an optimal retrieval threshold and reducing embedding size substantially reduced energy consumption and latency with no loss in accuracy, making these two techniques truly energy-efficient. We present the first comprehensive, empirical study on energy-efficient design techniques for RAG systems, providing guidance for developers and researchers aiming to build sustainable RAG applications.

## Full Text


<!-- PDF content starts -->

On the Effectiveness of Proposed Techniques to Reduce Energy
Consumption in RAG Systems: A Controlled Experiment
Zhinuan (Otto) Guo
Vrije Universiteit Amsterdam
Amsterdam, The Netherlands
guozhinuan2@gmail.comChushu Gao
Software Improvement Group
Amsterdam, The Netherlands
chushu.gao@softwareimprovementgroup.comJustus Bogner
Vrije Universiteit Amsterdam
Amsterdam, The Netherlands
j.bogner@vu.nl
Abstract
The rising energy demands of machine learning (ML), e.g., imple-
mented in popular variants like retrieval-augmented generation
(RAG) systems, have raised significant concerns about their environ-
mental sustainability. While previous research has proposed green
tactics for ML-enabled systems, their empirical evaluation within
RAG systems remains largely unexplored. This study presents a con-
trolled experiment investigating five practical techniques aimed at
reducing energy consumption in RAG systems. Using a production-
like RAG system developed at our collaboration partner, the Soft-
ware Improvement Group, we evaluated the impact of these tech-
niques on energy consumption, latency, and accuracy.
Through a total of 9 configurations spanning over 200 hours of
trials using the CRAG dataset, we reveal that techniques such as
increasing similarity retrieval thresholds, reducing embedding sizes,
applying vector indexing, and using a BM25S reranker can signif-
icantly reduce energy usage, up to 60% in some cases. However,
several techniques also led to unacceptable accuracy decreases, e.g.,
by up to 30% for the indexing strategies. Notably, finding an opti-
mal retrieval threshold and reducing embedding size substantially
reduced energy consumption and latency with no loss in accuracy,
making these two techniques truly energy-efficient. We present
the first comprehensive, empirical study on energy-efficient design
techniques for RAG systems, providing guidance for developers
and researchers aiming to build sustainable RAG applications.
CCS Concepts
•Software and its engineering →Designing software;Soft-
ware performance;•Computing methodologies →Machine
learning.
Keywords
green ML engineering, retrieval-augmented generation, energy
consumption, latency, accuracy, controlled experiment
ACM Reference Format:
Zhinuan (Otto) Guo, Chushu Gao, and Justus Bogner. 2026. On the Ef-
fectiveness of Proposed Techniques to Reduce Energy Consumption in
RAG Systems: A Controlled Experiment. In2026 IEEE/ACM 48th Inter-
national Conference on Software Engineering (ICSE-SEIS ’26), April 12–18,
2026, Rio de Janeiro, Brazil.ACM, New York, NY, USA, 12 pages. https:
//doi.org/10.1145/3786581.3786932
This work is licensed under a Creative Commons Attribution 4.0 International License.
ICSE-SEIS ’26, Rio de Janeiro, Brazil
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2424-4/2026/04
https://doi.org/10.1145/3786581.37869321 Introduction
Large language models (LLMs) as the most recent wave of machine
learning (ML) have shown impressive performance in natural lan-
guage processing tasks and beyond [ 5]. However, they also face
several challenges, including hallucination [ 20,46], outdated knowl-
edge, and opaque or untraceable reasoning processes [ 9]. Retrieval-
augmented generation (RAG) has emerged as a popular technique
to address shortcomings by integrating knowledge from external
databases into the generation process [20].
As ML-enabled systems are increasingly adopted, more attention
is also spent on their carbon footprint [ 3,28]. Due to factors such
as increased technology adoption, cryptocurrency trends, and the
rising demand for artificial intelligence, global data center electricity
consumption is projected to range between 620 and 1,050 TWh by
2026 [ 18]. The growing popularity of RAG systems is adding to this
consumption, as they are increasingly applied in diverse domains
such as code generation [ 33,51], question answering [ 16,17], and
AI for science [ 43]. Their energy consumption and carbon footprint
should therefore be a subject of concern.
While research on the environmental impact of ML-enabled
systems has received more interest in recent years [ 10,41], the
quality attribute (QA) of environmental sustainability has received
little direct attention in RAG systems. Existing research predomi-
nantly emphasizes other QAs, such as accuracy and latency, while
largely neglecting energy efficiency. For instance, Wang et al . [42]
investigated various RAG configurations but focused primarily on
performance metrics like accuracy and latency. Similarly, Järvenpää
et al. [19] have proposed 30 green architectural tactics for generic
ML-enabled systems, but the practical applicability of these tactics
within the context of RAG systems remains largely unexplored.
This underscores a clear research gap: studying how to reduce en-
ergy consumption as a core quality concern in RAG systems and
systematically exploring trade-offs with other system attributes.
While some techniques have been proposed to improve the resource
utilization or latency of RAG systems, no systematic study has ex-
amined the energy efficiency impact of RAG techniques to provide
guidance to practitioners about their usage.
To close this gap, the primary objective of this research is to ana-
lyze the energy consumption of various techniques in RAG systems
while also examining potential trade-offs with response latency
and the accuracy of generated answers. We conducted a controlled
experiment [ 44] involving five techniques, some tested under dif-
ferent parameter settings, resulting in a total of nine configurations
that we compared to a baseline industrial RAG system. To the best
of our knowledge, this is the first study to conduct an in-depth
investigation into different energy techniques for RAG systems.
Our findings provide valuable insights for future research on the
1arXiv:2601.02522v1  [cs.SE]  5 Jan 2026

ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil Guo et al.
environmental sustainability of such systems and offer guidance on
navigating trade-offs between energy usage, latency, and answer
accuracy.
2 Related Work
In this section, we discuss related work in the area of ML energy
efficiency and empirical studies about RAG systems.
Järvenpää et al . [19] presented a synthesis of green architectural
tactics for ML-enabled systems. They compiled a catalog of 30
tactics, derived from an extensive review of Green AI literature
and validated through an expert focus group. These tactics span six
categories, namely data-centric strategies, algorithm design, model
optimization, model training, deployment, and management, aimed
at reducing energy consumption and enhancing computational
efficiency throughout the ML system lifecycle. While their work
provides valuable, actionable guidance for building sustainable ML-
enabled systems, the proposed tactics are not specifically tailored to
RAG systems, and evidence for their effectiveness in this context is
missing. In this study, we focus specifically on production-level RAG
systems and empirically evaluate the impact of selected techniques
on energy efficiency.
Using experiments to demonstrate that current LLM agents and
RAG systems consume substantial amounts of energy, Wu et al .
[45] introduced the “Sustainable AI Trilemma”, highlighting the
tensions between AI capability, negative environmental impact, and
digital inequality. They reported that common optimization steps,
e.g., query optimization or compression, drastically increase energy
use with diminishing returns and that LLM-dependent methods
consume orders of magnitude more energy than non-LLM alter-
natives. While their paper crucially highlights these issues and
inefficiencies within RAG systems, it stops at diagnosis and does
not propose or study specific green techniques as a solution. In
contrast, our research focuses explicitly on evaluating proposed
technique variations for designing green RAG systems.
Current techniques targeting RAG systems primarily focus on
improving other key performance metrics, such as response time
or reducing computational resource usage. In several cases, this
can also lead to reductions in energy consumption. For example,
Arefeen et al . [1] examined the effect of threshold k selection in
chunk filtering within the iRAG framework, demonstrating that
a balanced selection of k enhances query efficiency and reduces
computational waste. While increasing k results in longer query
processing and greater computational overhead due to processing
more candidate chunks, effective chunk filtering mitigates this by
minimizing unnecessary computation while preserving high recall.
However, there remains a lack of broad empirical evidence on the
effectiveness and trade-offs of various energy-saving techniques
specifically designed for RAG systems, especially for realistic sys-
tems from an industry context.
Similarly, Şakar and Emekci [52] conducted a comprehensive
evaluation of various RAG systems, identifying optimal config-
urations that balance critical performance metrics like response
accuracy, token efficiency, runtime, and hardware utilization across
diverse domains. Their findings indicate that Reciprocal RAG, which
generates and ranks multiple query variations to resolve ambiguity,
achieved the highest similarity score but at the cost of significantlyhigher token usage and longer run times. In contrast, the Stuff
method, which simply stuffs all retrieved documents into a sin-
gle prompt, was the fastest and most token-efficient approach but
sacrificed response accuracy by not addressing query ambiguity.
However, their study does not offer any insights on the energy
consumption of the methods, omitting a crucial dimension for a
comprehensive system evaluation. Our research puts energy con-
sumption at the center and also studies potential trade-offs with
latency and accuracy.
In a study from the medical domain, Kartiyanta et al . [23] sys-
tematically evaluated the end-to-end performance of RAG sys-
tems on the RAGEval DragonBall dataset, employing retrieval
metrics (recall, precision, MAP) and generation metrics (RAGAs,
BERTScore) to compare cost-effective sparse and dense retrieval
methods against commercial embeddings. Their key finding was
that the sparse method BM25 outperformed all dense and com-
mercial embedding models in the retrieval stage and achieved the
highest scores in most generation metrics. While their research pro-
vides valuable insight into the effectiveness of open sparse retrieval
methods like BM25, it is limited to a single technique and medical
domain benchmark dataset. In contrast, our research extends this
investigation to a real-world production application by evaluating
a RAG system adopted by our industry partner. This allows us to
explore the performance and challenges of these retrieval methods
under practical operational conditions and constraints.
Overall, there is a lack of empirical evidence about techniques
that directly optimize the energy consumption in RAG systems.
Despite the increasing popularity of RAG as a framework within
ML-enabled applications, its energy consumption characteristics
and techniques to improve them remain underexplored. Most ex-
isting studies on energy efficiency focus on general ML-enabled
systems rather than RAG. While these more general techniques can
still be of situational value for designing greener RAG architectures,
e.g., by selecting energy-efficient algorithms [ 22], more specific
green RAG techniques and evidence about their effectiveness and
trade-offs are needed. Our study aims to start closing this gap.
3 Study Design
To ensure both academic rigor and industrial relevance, we formed
an academia-industry collaboration between our university and
the Software Improvement Group (SIG), a software consultancy
firm specialized in software quality and digital sustainability with
roughly 160 employees.1We started by choosing and analyzing one
of their suitable RAG systems. After that, we together reviewed and
discussed the scientific literature on RAG systems to identify and
select suitable techniques for their system. Finally, we conducted an
industry-informed controlled experiment with selected techniques.
The primary objective of our research was to evaluate the effec-
tiveness of current techniques in reducing energy consumption in
RAG systems and to study potential trade-offs with relevant QAs.
Towards this end, our study addresses the following two research
questions:
RQ1 How effective are proposed techniques for reducing the en-
ergy consumption of RAG systems?
1https://www.softwareimprovementgroup.com
2

On the Effectiveness of Proposed Techniques to Reduce Energy Consumption in RAG Systems ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil
RQ2 How does the application of these techniques impact accu-
racy and latency?
To maintain a manageable scope for the trade-off analysis (RQ2),
we focused on two QAs that are essential for the effectiveness and
acceptable user experience of industrial RAG systems for question
answering: the accuracy and latency of responses. Therefore, under-
standing the trade-off between energy consumption and these two
QAs is crucial. Unnecessary retrieval may increase latency and com-
putational costs, while insufficient retrieval may lead to incomplete
or incorrect answers [ 40]. Thus, investigating the trade-offs that
techniques cause among these quality concerns holds significant
value for practitioners and researchers alike.
3.1 Experiment Objects
Research on techniques for improving the environmental sustain-
ability of RAG systems is unfortunately still scarce, and selecting
promising candidates as experimental objects based on academic
literature is nontrivial. Therefore, we also included techniques that
have not been specifically conceptualized in the context of energy
efficiency but seemed promising in terms of reducing energy con-
sumption. To guide our selection of techniques, we applied the
following inclusion criteria:
•Potentially energy-saving:The adopted techniques must
either have been directly recommended to lower energy con-
sumption or seem reasonable to assume energy-related bene-
fits due to, e.g., reduced resource consumption.
•Not tied to proprietary solutions:If a technique is closely
related to proprietary solutions from companies like OpenAI
or Bedrock, it is difficult to generalize the experiment results,
and it may not be locally implementable.
•Suitability for the current environment:Techniques must
be compatible with the existing system architecture and exper-
iment environment. For instance, our GPU provides approxi-
mately 24 GB of VRAM, which limits how many models we
can run simultaneously. Techniques requiring several LLMs,
such as LLM-based context compression, are therefore infea-
sible.
•Ease of implementation:The technique should be imple-
mentable in the experimental system with reasonable effort.
Using these criteria, we finally selected five techniques, which
are shown below.
T1 – Increase similarity threshold of pgvector queries:The
threshold in pgvector2refers to the similarity score used to filter re-
sults when querying vector embeddings in PostgreSQL. This score
determines which documents are returned based on their similarity
to the user query. A higher threshold returns fewer, more relevant
documents, thereby reducing the amount of context retrieved in
a RAG system. This reduction may lead to reduced energy con-
sumption, as less data needs to be processed. Bulgakov [4]showed
that filtering semantically incoherent documents via thresholding
significantly improves retrieval quality while minimizing memory
and compute overhead. The original baseline threshold in the RAG
system under study was set to 0.58. To better understand the im-
pact of this value, we evaluated 100 test queries and found that
2https://github.com/pgvector/pgvectorthe mean similarity score was 0.78. This provided a statistically
grounded reference point for typical retrieval behavior and helped
avoid arbitrary threshold selection. Since real production data is
typically significantly larger and more diverse than the test sample,
using the mean similarity score as a reference is more reasonable
for generalization, as it captures the central tendency of similarity
scores over a broader range of queries. Based on this, we chose ad-
ditional thresholds at regular intervals around these two values to
examine how varying similarity cutoffs affect energy consumption
and the other QAs (0.58 as the baseline, 0.68, 0.78, and 0.88).
T2 - Introduce lightweight reranking algorithm:Rerank-
ing typically serves as an enhancement technique to improve the
accuracy of RAG systems [ 38]. Since GPU inference typically con-
sumes more energy than CPU operations, reducing the load on the
GPU can help to lower overall energy usage. In our experiment, we
applied a lightweight reranking method using BM25S [ 29] to filter
and prioritize candidate documents before they are processed by
the frozen LLM. This approach reduces the number of documents
passed to the LLM, potentially decreasing energy consumption
while maintaining performance.
T3 - Reduce embedding sizes of the embedding model:Re-
ducing the size of word embeddings can improve their efficient
use in memory-constrained devices, benefiting real-world applica-
tions [ 37]. In our experiment, we replaced the default embedding
model in the baseline system (e5-large-v23, 1024 dimensions) with
smaller variants from the same provider: e5-base-v24(768 dimen-
sions) and e5-small-v25(384 dimensions).
T4 - Apply an efficient vector search indexing strategy:
The indexing strategy of the vector search in RAG systems impacts
how quickly the search is performed but also which and how many
documents are selected. Previous studies indicate that indexing via
Hierarchical Navigable Small World (HNSW) [ 30] or Inverted File
with Flat Quantization (IVFFlat) [ 32] can enhance efficiency, partic-
ularly with binary embeddings [ 14]. HNSW indexing constructs a
graph-based structure that facilitates efficient and robust approx-
imate nearest neighbor search by traversing hierarchical layers
of nodes. IVFFlat indexing divides the data into clusters and em-
ploys inverted lists to rapidly narrow down the search space. Since
both indexing methods are well-suited for fast approximate nearest
neighbor searches in high-dimensional spaces and also supported
by pgvector, we included them both in our experiment.
T5 - Cache intermediate retrieval states via knowledge
trees:If several queries show decent similarity with each other,
caching the intermediate knowledge in memory for reuse can en-
hance system efficiency by reducing redundant computation. To
allow this, Jin et al . [21] have proposed the RAGCache approach
that uses special structures calledknowledge trees. This allows new
queries to skip recomputation for shared prefixes and has been
shown to improve overall latency and throughput in RAG sys-
tems [ 6,26,27,48]. We adopt a configuration in vLLM for this
called enable_prefix_caching [24], which enables the key-value
(KV) cache of existing queries.
3https://huggingface.co/intfloat/e5-large-v2
4https://huggingface.co/intfloat/e5-base-v2
5https://huggingface.co/intfloat/e5-small-v2
3

ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil Guo et al.
Table 1: ChatRAG Baseline System Characteristics
Component Technology Explanation
Backend Spring Original ChatRAG code
Embedding
ModelE5 larger (V2) em-
bedding model with
1024 dimension6Open-source, supports different
embedding sizes
Frozen LLM Llama 3.1 8B In-
struct7Popular open-source LLM, sup-
ports quantization
Vector Data-
basePostgreSQL 178Stable open-source database;
same database as ChatRAG
Reranking
Model– Discarded: not usable with exper-
iment hardware
3.2 Experiment Materials
To ground the experiment design in real-world application require-
ments, we selected ChatRAG as our experiment system, a RAG
system developed by SIG. ChatRAG is a query-based chatbot that
enhances its responses with insights from retrieved internal doc-
uments by feeding them directly into the initial stage of the gen-
erator [ 49]. The specific architecture of ChatRAG is illustrated in
Fig. 1, while the detailed configuration of the baseline system and
its components is provided in Table 1.
To collect energy consumption data, we used Kepler9, which
attributes power usage to containers and Kubernetes Pods. Kepler
gathers real-time power consumption metrics from node compo-
nents using Intel’s Running Average Power Limit (RAPL) for CPU
and DRAM power, and the NVIDIA Management Library (NVML)
6https://huggingface.co/intfloat/e5-large-v2
7https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
8https://www.postgresql.org/docs/current/release-17.html
9https://sustainable-computing.io/for GPU power. The collected and estimated container-level data
is then stored using Prometheus. To host the model in a Kuber-
netes environment with GPU resources, we adopted vLLM [ 25].
vLLM is a high-throughput and memory-efficient inference and
serving engine for LLMs, supporting scalable deployment on Ku-
bernetes. With the assistance of the integration tool Helm10, we
could efficiently deploy our models on Kubernetes.
To send and evaluate API calls to ChatRAG, we required an eval-
uation tool capable of measuring system performance. We adopted
LLMPerf11, which supports customization for issuing calls and col-
lecting results related to latency and accuracy. In our experiment,
we extended this tool, which was originally based on the Ray dis-
tributed framework for emerging AI applications [ 31], to support
request handling, energy data collection, and accuracy tracking.
This extended version of the tool is publicly available on GitHub.12
In real-world applications, direct energy measurements are of-
ten not feasible, especially when systems are hosted on virtualized
public cloud infrastructure or if LLMs are accessed through enter-
prise APIs. To enable accurate energy monitoring in a controlled
environment, we therefore deployed the ChatRAG system variants
on a server cluster in the Experiment Lab of our university, a fa-
cility specifically designed for research on energy efficiency. The
machine used for the experiments was equipped with an NVIDIA
GeForce RTX 4090 GPU (24 GB VRAM), an Intel Core i9-14900KF
CPU (24 cores), and 128 GB of RAM. The overall experiment setup is
illustrated in Fig. 2. The frozen LLM was hosted on the GPU, while
other components ran on CPUs. A local machine with LLMPerf
sent API calls to the backend pod and collected energy data from
the Kepler pod.
10https://helm.sh
11https://github.com/ray-project/llmperf.git
12https://github.com/KafkaOtto/rageval
Figure 1: ChatRAG Software Architecture and Workflow
4

On the Effectiveness of Proposed Techniques to Reduce Energy Consumption in RAG Systems ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil
Figure 2: Experiment Environment and Workflow
3.3 Experiment Variables
The primaryindependent variablein our experiment is the sys-
tem variations created by applying each selected technique. Some
techniques also included multiple configurations. Specifically, T1
had three configurations with the similarity thresholds 0.68, 0.78,
and 0.88 (baseline: 0.58). T3 involved two configurations based on
embedding sizes: 768 and 384 (baseline: 1024). T4 included two
configurations based on indexing strategies in pgvector: HNSW
and IVFFlat (baseline: no indexing strategies). T2 and T5 each con-
sisted of a single configuration. In total, that led to 9 technique
configurations plus the baseline, i.e., 10 treatments in total. The
dependent variablesfor RQ1 were energy consumption (J), while
they were accuracy (%) and latency (s) for RQ2.
3.4 Dataset
Several criteria needed to be considered for selecting a benchmark
dataset. Firstly, the dataset had to be manageable, as very large-
scale datasets were not suitable for our cluster. For example, the
KILT benchmark [ 35], designed for intensive language tasks, con-
tains raw knowledge sources of approximately 35 GB, which would
not be suitable for our experiment. Secondly, the dataset had to
be rich and cover a diverse set of topics, as dataset diversity can
make the results more convincing. Based on these criteria, we se-
lected the CRAG dataset [ 47], which contains a manageable total of
2,706 questions. It also includes a diverse range of question types,
such as comparison and multi-hop questions, which are reasonably
close to real-world challenges commonly encountered in practical
applications. CRAG highlights the inherent trade-offs between ac-
curacy and latency across various state-of-the-art RAG systems,
offering useful context for evaluating ChatRAG’s performance. The
CRAG benchmark employs a structured scoring method to assess
the quality of responses generated by RAG systems, categorizing
each answer as perfect, missing, or incorrect. Following this ap-
proach, we implemented a similar labeling process. In our eval-
uation, we relied on an LLM-as-Judge approach [ 11] to validate
the correctness of the final answers, excluding responses such as
“I don’t know”. LLM-as-Judge approaches provide efficient ways
to evaluate query responses that go beyond simple requests likemultiple-choice questions and have proven remarkably effective
for many benchmarks, matching even crowdsourced human evalu-
ations [ 50]. We employed DeepSeek-V2 [ 8] as our LLM judge due
to its low cost and competent question-answering capabilities. The
details of this evaluation and the used prompts are available in our
replication package13.
3.5 Experiment Execution
Before executing the experiment, we first had to prepare the CRAG
documents for pgvector. For the chunking strategy, we used the
TokenTextSplitter of Spring AI.14Due to the maximum token
limit of 512 in the downstream embedding model, we set the chunk
size window to 320, the same value as used in ChatRAG. This
resulted in a total of 501,916 split chunks. Among these, 10 chunks
were rejected by the embedding model because their token count
exceeded the 512-token limit. We chose to disregard these errors,
as the overall number of chunks remains manageable.
For a test run of 1,335 queries, the total elapsed time was approx-
imately 2 hours and 26 minutes. We observed that it took around 5
minutes for the CPU to reach a stable usage level. Since fluctuating
CPU utilization could act as a confounder for measuring energy
consumption, we therefore introduced a warm-up period to ensure
that each trial was conducted under stable conditions. The dataset
was divided into two parts: a warm-up dataset consisting of 100
random queries and an experiment dataset containing the remain-
ing queries. The warm-up dataset was executed before each trial to
stabilize the environment. Additionally, we introduced a cool-down
period of 5 minutes after each trial to allow system resources to
reset. The used dataset is available on Zenodo.15
Furthermore, each experiment configuration, i.e., 9 technique
variations + 1 baseline, was tested over a period of 10 trials to
ensure the reliability of measurements and to limit the influence
of potential random confounders. A second reason for doing this
was that we used the default configuration of Llama 3.1 8B Instruct,
which employs a non-deterministic decoding strategy with a non-
zero temperature and sampling enabled. As a result, the model
may produce slightly different outputs across runs for the same
input. While this increased variability was beneficial for external
validity, the 10 trials per treatment were required to average out
potential suboptimal generations. We therefore report the average
accuracy of all treatment runs. In total, we ran 100 experiment trials.
Given that a single experiment trial took approximately 2 hours
and 26 minutes, the total time required to complete the experiment
was around 240 hours, i.e., slightly over 10 days. This substantial
runtime also explains why we could only consider a single dataset.
3.6 Data Analysis
After we collected all the data from the experiments, we assessed
the normality of each data subset using the Shapiro-Wilk test [ 13].
Since all datasets in our experiment passed the normality test, we
used the t-test [ 39] to evaluate the significance levels. Based on
our hypotheses, we used a one-tailed test for energy consumption
13https://anonymous.4open.science/r/green-rag-techniques-experiment/running/
README.md
14https://docs.spring.io/spring-ai/docs/current/api/org/springframework/ai/
transformer/splitter/TokenTextSplitter.html
15https://doi.org/10.5281/zenodo.16569517
5

ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil Guo et al.
and a two-tailed test for performance. Since multiple hypotheses
were tested simultaneously, we had to guard against the multiple
comparisons problem, where the probability of incorrectly rejecting
at least one true null hypothesis (Type I error) increases with the
number of tests performed [ 2]. To mitigate this, we applied the
Holm-Bonferroni correction [ 15], which adjusts the p-values to
control the family-wise error rate. After correction, an adjusted
p-value < 0.05 indicates a statistically significant difference, while
a value > 0.05 suggests no significant difference. For statistically
significant differences, we calculated Cohen’s d [ 7] to estimate the
effect size. According to Cohen’s guidelines, values between 0.0
and 0.2 indicate a negligible effect, 0.2 to 0.5 a small effect, 0.5 to
0.8 a moderate effect, and 0.8 and above a large effect. Additionally,
we calculated mean values for each configuration with a significant
effect and then computed the percentage change relative to the
baseline (control).
4 Results
In this section, we present our quantitative experiment results
according to the research questions. While we provide some inter-
pretation for unexpected results, more explanations can be found
in the discussion section.
4.1 Reducing Energy Consumption (RQ1)
Based on the t-test results in Fig. 3a, we found that 7 of the 9
technique configurations had a statistically significant impact on
the system’s energy consumption. The exceptions were T1 with
a similarity threshold of 0.68, for which the small reduction was
not significant, and T5 (enabling caching prefixes), which even
led to slightly increased energy consumption. For the significant
techniques, all effects were very large (smallest Cohen’s 𝑑of 1.48),
with some even reaching a Cohen’s𝑑of 4.0 and above.
For the different thresholds of T1, we found that the difference
between 0.68 and the 0.58 baseline was minor. However, as the
threshold increased to 0.78 and 0.88, energy consumption decreased
significantly, with the 0.88 threshold showing the greatest reduction.
This outcome aligns with our expectations, as the average similarity
score for the test data was approximately 0.78. Therefore, thresholds
of 0.58 and 0.68 retrieved documents with fairly close similarity
scores, whereas thresholds of 0.78 and above filtered out more
documents, reducing the computational load. As shown in Fig. 3b,
using a threshold of 0.78 resulted in a 20.7% reduction in energy
consumption, while increasing the threshold to 0.88 achieved an
impressive 51.0% reduction.
For the lightweight reranker T2, we observed a 32.6% reduction in
energy consumption. Interestingly, as shown in Fig. 4, both applying
T1 with a threshold of 0.88 and T2 led to the same GPU energy
usage percentage, with 28.3% of the total energy, the lowest of all
experiment configurations. While the numerical similarity may be
coincidental, the results suggest that both a high retrieval threshold
and an effective reranking method can help filter irrelevant inputs
early on. This reduces the workload of the LLM, which is typically
the most energy-intensive component of the system.
Regarding T3 (reducing the embedding size), both dimensions of
768 and 384 resulted in notable decreases in energy consumption
1
012345Cohen's d Effect Size
T10.68 0.78 0.88
T2bm25s
T3768 384
T4ivfflat hnsw
T5prefixd=0.14p=0.6651d=1.48p=0.0017d=3.96p<1e-4
d=1.97p=4e-4
d=1.58p=0.0015d=3.67p<1e-4d=4.98p<1e-4
d=4.22p<1e-4
d=-0.59p=0.9529Positive effect
Negative effect
Not significant(a) T-Test Results for Energy Consumption
60
50
40
30
20
10
0Percentage Changes(-%)
T10.78 0.88
T2bm25s
T3768 384
T4ivfflat hnsw-20.68-51.02
-32.64
-20.00-38.12-59.76 -58.63Positive effect
Negative effect
(b) Relative Energy Difference to Baseline for Significant Techniques
Figure 3: Energy Consumption Experiment Results
01234Energy (mJ)×103
T1base 0.58 0.68 0.78 0.88
T2bm25s
T3768 384
T4ivfflat hnsw
T5prefix
combo384 & 0.7837.5%62.5%
38.0%62.0%
40.1%59.9%
28.3%71.7%
28.3%71.7%
42.2%57.8%
47.0%53.0%
59.1%40.9%
58.9%41.1%
36.7%63.3%
51.5%48.5%Other Energy GPU Energy
Figure 4: GPU Contribution to Total Energy Consumption.
(20.0% and 38.1%). Since the embedding size defines the dimen-
sionality of vector representations, smaller dimensions generated
smaller vectors, which required less memory and computational
resources and was therefore beneficial for energy use in practice.
Among all evaluated techniques, T4 yielded the most substantial
energy reductions, with both indexing methods achieving approx-
imately a 60% decrease. Indexing likely enabled faster and more
efficient similarity searches, which not only lowered query latency
and CPU usage for the database itself but also reduced overall sys-
tem resource contention. Additionally, the indexing strategies also
reduced the number of fetched documents that were passed on to
the LLM, which likely contributed to achieving the largest energy
reductions among all evaluated techniques.
6

On the Effectiveness of Proposed Techniques to Reduce Energy Consumption in RAG Systems ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil
For T5, we enabled prefix caching on the frozen LLM during
the generation stage. However, no significant differences in energy
consumption or performance were observed, and energy use even
slightly increased compared to the baseline. As shown in the Llama
pod logs (Listing 1), cache utilization remained very low, with GPU
hit rates of only∼1.1% and no hits on the CPU.
INFO 06 -28 02:52:53 metrics .py :471] Prefix
cache hit rate : GPU : 1.09% , CPU : 0.00%
INFO 06 -28 02:52:58 metrics .py :455] Avg
prompt throughput : 0.0 tokens /s, Avg
generation throughput : 54.8 tokens /s,
Running : 1 reqs , Swapped : 0 reqs ,
Pending : 0 reqs , GPU KV cache usage :
23.7% , CPU KV cache usage : 0.0%.
Listing 1: Llama Pod Logs for T5
This limited effectiveness is likely due to the short prompt lengths
and minimal overlap between retrieved documents across queries.
To quantify query similarity, we applied TF-IDF vectorization [ 36]
combined with cosine similarity. The average similarity score was
0.0294 (standard deviation 0.0437), indicating that most queries dif-
fer substantially. As a result, the prefix cache was rarely reused, lead-
ing to negligible improvements. In contrast, systems with higher
query similarity might benefit more from prefix caching.
4.2 Trade-off With Latency (RQ2a)
In this section, we examine the impact of the evaluated techniques
on system latency and explore the relationship to the energy con-
sumption results. Fig. 5a shows the t-test results for latency. We
observe the same trend as for energy consumption: all seven tech-
niques that significantly reduced energy consumption also signif-
icantly reduced latency, with the exceptions again being T1 with
the 0.68 threshold and T5. However, the strengths of the reductions
differed notably between the two QAs, highlighting that energy
consumption and latency are not in a perfect linear relationship for
complex distributed RAG systems.
For T1, as the threshold increased to 0.78 and 0.88, latency im-
proved by 24.8% and 42.0%, respectively. The 0.68 threshold was
again too similar to the baseline to lead to significant changes.
Regarding T2 (the BM25S reranker), the improvement in latency
was less pronounced compared to the improvement in energy con-
sumption (20.3% vs. 32.6%). This suggests that the additional re-
sponse time introduced by the reranker component offset some of
the overall latency gains, but its filtering capability still led to more
substantial downstream energy savings.
For T3, embedding size reductions to 768 and 384 improved
latency by 28.6% and 50.7% respectively. Notably, going from the
1024 dimensions of the baseline to 384 led to a greater reduction in
latency than increasing the T1 threshold from 0.58 to 0.88, despite
the 0.88 threshold T1 variant yielding more substantial energy
reduction. This again highlights that using latency reductions as a
proxy for energy reductions is not always reliable.
Moreover, the T4 indexing strategies again yielded the most
substantial improvement in latency among all techniques. Both
IVFFlat and HNSW resulted in similar latency reductions of about
76% compared to the baseline.
1
0123456Cohen's d Effect Size
T10.68 0.78 0.88
T2bm25s
T3768 384
T4ivfflat hnsw
T5prefixd=0.17p=0.6095d=1.42p=0.0061d=2.72p<1e-4
d=1.00p=0.0341d=1.85p=0.0012d=4.09p<1e-4d=5.79p<1e-4
d=5.24p<1e-4
d=-0.65p=0.1381Positive effect
Negative effect
Not significant(a) T-Test Results for Latency
80
60
40
20
0Percentage Changes(-%)
T10.78 0.88
T2bm25s
T3768 384
T4ivfflat hnsw-24.76-41.95
-20.34-28.63-50.65-76.96 -76.26Positive effect
Negative effect
(b) Relative Latency Difference to Baseline for Significant Techniques
Figure 5: Latency Experiment Results
Lastly, T5 was similarly ineffective for latency reductions as it
was for energy consumption. The reason was likely the same: used
queries and documents were not similar enough to make good use
of the caching functionality.
4.3 Trade-off With Accuracy (RQ2b)
In this section, we first examine the baseline’s ability to answer
questions accurately and then evaluate the impact of the proposed
techniques on system accuracy. Among all 2,606 questions from
our benchmark dataset, the baseline system produced, on average,
correct answers for 619 questions, resulting in an accuracy of 23.75%.
This is in line with the published CRAG benchmark results [ 47],
which mention an accuracy of 23.7% for Llama 3.1 8B Instruct. Our
baseline system achieved almost identical accuracy, which validates
the realism and effectiveness of our base setup, but also of our LLM-
as-Judge approach.
We show the t-test results for accuracy in Fig. 6a and the respec-
tive percentage change of significant techniques in Fig. 6b. For T1,
increasing the retrieval threshold to 0.68 and 0.78 did not signifi-
cantly harm accuracy, with the latter even significantly improving
it by 1.7%. However, the 0.88 threshold had a substantial negative
impact, with accuracy dropping by a massive 71.3%. When the
threshold is low, the system retains more data points, which helps
maintain accuracy but limits improvement on energy consumption
and latency. As the threshold increases, fewer data points are used,
improving efficiency at the cost of potentially discarding important
information, which explains the sharp drop in accuracy at 0.88.
This highlights a promising direction for future research: efficiently
7

ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil Guo et al.
020406080Cohen's d Effect Size
T10.68 0.78 0.88
T2bm25s
T3768 384
T4ivfflat hnsw
T5prefixd=0.25p=0.4533
d=-1.39p=0.0071d=76.68p<1e-4
d=11.59p<1e-4
d=-0.40p=0.4768d=-1.44p=0.0068d=9.62p<1e-4
d=2.71p<1e-4
d=0.48p=0.4900Positive effect
Negative effect
Not significant
(a) T-Test Results for Accuracy
80
60
40
20
0Percentage Changes(-%)
T10.78 0.88
T2bm25s
T3384
T4ivfflat hnsw1.71-71.29
-11.02
1.73-32.02
-21.99Positive effect
Negative effect
(b) Relative Accuracy Difference to Baseline for Significant Tech-
niques
Figure 6: Accuracy Experiment Results
identifying the “golden threshold” that optimally balances energy
consumption and latency without compromising accuracy for a
given RAG system. Since T1 with the 0.78 threshold also contributed
to improvements in energy consumption and latency, it was one of
the few techniques with advantageous outcomes for all three QAs.
Regarding T2 with the BM25S reranker, its observed improve-
ments in both latency and energy consumption unfortunately came
at the cost of accuracy, with a decrease of approximately 11%. This
suggests that the reranker may filter out some critical documents
retrieved by pgvector that are essential for generating accurate
responses. As a result, the final set of documents passed to the
language model is less informative, leading to a reduction in overall
accuracy. This trade-off might be acceptable for some practical RAG
use cases, while it may disqualify the technique for many others.
For T3, reducing the embedding size to 768 did not significantly
affect accuracy. Additionally, when reducing the embedding size
further to 384, there was even a slight but significant increase in
accuracy (1.7%). As discussed earlier, both T3 variants also signifi-
cantly improved energy consumption and latency, making T3 the
second technique with no drawbacks for our three QAs. This sug-
gests that decreasing the embedding size did not compromise the
model’s ability to retrieve relevant documents from pgvector. One
possible explanation is that the queries in our dataset are relatively
short, resulting in the system consistently retrieving the full top- 𝑘
documents (150). Consequently, the retrieved document set and
its size remained stable across embedding sizes, which might have
been different for longer queries.While both T4 indexing strategies had substantial benefits for en-
ergy consumption and latency, they unfortunately also significantly
reduced accuracy. Both IVFFlat and HNSW had a very similar effect
on the first two QAs, but their negative impact on accuracy notably
differed. The use of HNSW resulted in an accuracy decrease of 22.0%,
whereas IVFFlat led to an even larger drop of 32%. This suggests
that, under default parameter settings, HNSW may be the better
choice when balancing energy consumption and accuracy, but that
both are still unlikely to be usable in practice: their accuracy drops
will simply be unacceptable for most RAG use cases.
Lastly, the prefix caching of T5 did not significantly impact ac-
curacy, similar to the other two QAs, making it the least impactful
technique in our experiment.
010203040506070Percentage Change (%)
T10.78
T3384
combo384 & 0.7820.738.146.7
24.850.762.3
1.7 1.70.0Energy reduction
Latency reduction
Accuracy changes
Figure 7: Relative Differences for T1-0.78 and T3-384 Combi-
nation Compared to Baseline
4.4 Combining the Two Best Techniques
Two technique configurations (T1 with the 0.78 threshold and T3
with 384 dimensions) emerged as the most beneficial candidates,
improving energy consumption, latency, and even slightly accuracy.
To understand whether their benefits compound when applied to-
gether, we ran an additional experiment with a system variation
that implemented both of them simultaneously. As shown in Fig. 7,
the combination of the two led to even greater improvements in en-
ergy consumption and latency. Specifically, T1-0.78 alone achieved a
20.7% reduction in energy consumption, while T3-384 alone yielded
a 38.1% reduction. Their combination, however, reduced energy
consumption by 46.7%. The results for latency were similarly bene-
ficial. Although each individual configuration results in a modest
accuracy improvement of 1.7%, their combination did not produce
a statistically significant change in accuracy anymore. According
to Gu et al . [12] , as the dimensionality of embeddings increases, the
distribution of pairwise cosine similarities tends to converge toward
a stable distribution with finite variance. In higher-dimensional
spaces, cosine similarity scores cluster more tightly around a cen-
tral value, reducing the variance between similarity scores across
different pairs. In practice, high-dimensional embeddings tend to
make random vectors more orthogonal, concentrating pairwise co-
sine similarities near zero with low variance. This shifts the natural
baseline similarity upward, e.g., from a slightly negative value in
lower dimensions to near zero, which likely requires a threshold
recalibration. Nonetheless, since the combination also did not harm
8

On the Effectiveness of Proposed Techniques to Reduce Energy Consumption in RAG Systems ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil
Table 2: Combined Experiment Results for All Dependent Variables
Technique Config MetricΔ(%) p-value Cohen’s d
T10.68Energy Consumption – 0.66219 –
Latency – 0.64539 –
Accuracy – 0.42672 –
0.78Energy Consumption -20.68 0.00107 1.657
Latency -24.76 0.00207 1.664
Accuracy 1.71 0.00062 -2.026
0.88Energy Consumption -51.02 <0.0001 4.052
Latency -41.95 <0.0001 2.919
Accuracy -71.29 <0.0001 79.965
T2 BM25SEnergy Consumption -32.64 <0.0001 2.425
Latency -20.34 0.00831 1.289
Accuracy -11.02 <0.0001 10.235
T3768Energy Consumption -20.00 0.00173 1.48
Latency -28.63 0.00204 1.722
Accuracy – 0.75527 –
384Energy Consumption -38.12 <0.0001 2.964
Latency -50.65 <0.0001 3.567
Accuracy 1.73 0.01490 -1.227
T4IVFFlatEnergy Consumption -59.76 <0.0001 4.806
Latency -76.96 <0.0001 5.680
Accuracy -32.02 <0.0001 10.342
HNSWEnergy Consumption -58.63 <0.0001 6.061
Latency -76.26 <0.0001 6.594
Accuracy -21.99 <0.0001 2.677
T5 Prefix cachingEnergy Consumption – 0.95776 –
Latency – 0.09637 –
Accuracy – 0.26845 –
accuracy, applying both techniques at once remains a powerful
option to improve energy efficiency.
5 Discussion
In this section, we summarize the key findings of our study, provide
additional explanations, and discuss their implications. Table 2
provides an aggregated summary of the impact of each technique on
the studied dependent variables, which serves as a quick reference
for identifying the most effective techniques and understanding
their associated trade-offs.
As shown in our results,applying indexing strategies (T4)
led to the strongest energy decreases, with both IVFFlat and
HNSW indexing achieving reductions of about 59%. One reason
for that is that the energy consumption of the database dropped
substantially due to the indexing (see Fig. 8). In all other configura-
tions, the database accounts for more than 4% of total energy usage,
whereas this share is noticeably reduced when indexing is applied.
Additionally, system-level processes like background services and
OS tasks also consumed less energy. This indicates that indexing en-
abled faster and more efficient similarity searches, lowering query
latency and CPU usage for the database itself while also reducing
overall system resource contention. But despite their energy ben-
efits, indexing comes with a trade-off: since they also reduce the
number of retrieved documents, both IVFFlat and HNSW decrease
accuracy by more than 20%, which is likely unacceptable in practice.However, it is important to note that we did not explore different
parameter configurations for IVFFlat. For example, increasing the
probes parameter is known to improve recall, potentially mitigat-
ing some accuracy loss. Future research could further investigate
the trade-offs of different IVFFlat settings to better optimize energy
consumption without sacrificing accuracy.
Another major result was thatsimilarity threshold increases
(T1) and embedding size reductions (T3) were the most bene-
ficial techniques. Unlike other techniques, these two significantly
reduced energy consumption without sacrificing latency or accu-
racy. In fact, each of them led to a modest accuracy improvement
of 1.7%. This suggests that RAG systems should carefully deter-
mine an optimal threshold based on dataset sampling and select an
appropriate embedding model for balanced system performance.
Interestingly, the combination of T1 and T3 returned the accuracy
to the baseline. This finding reveals thatthe optimal similarity
threshold in RAG systems is not fixed but can shift depend-
ing on embedding configurations.Therefore, when replacing
a high-dimensional embedding model with a lower-dimensional
one to reduce energy and latency, it is crucial to recalculate the
similarity threshold to preserve potential accuracy gains.
Moreover, our experiment showed thatcontrolling the over-
all content size is critical for improving energy efficiency
in enterprise RAG systems.As shown in Fig. 4, both threshold
adjustment and indexing strategies significantly reduced GPU en-
ergy consumption. Since the LLM itself remains fixed on the GPU,
9

ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil Guo et al.
01234Energy (Megajoule)
T1 T2 T3 T4 T5base 0.58 0.68 0.78 0.88 bm25s 768 384 ivfflat hnsw prefix37.4 %4.2 %46.6 %10.4 %
38.1 %4.5 %44.7 %11.2 %
40.9 %4.3 %43.4 %10.1 %
28.4 %5.3 %53.2 %11.3 %
27.1 %5.2 %54.8 %10.8 %
43.3 %4.1 %41.4 %9.9 %
48.4 %4.0 %37.2 %9.3 %
62.2 %28.3 %7.8 %
62.1 %28.5 %7.8 %
36.0 %4.1 %47.8 %10.6 %Other
SystemReranking
DatabaseGeneration
EmbeddingBackend
Figure 8: Energy Consumption Breakdown by Component
these results demonstrate that such techniques can effectively reg-
ulate the amount of data flowing into the generation model, a stage
that accounts for a substantial portion of energy consumption (see
Fig. 8). This highlights the importance of our study, especially in
real-world scenarios where frozen LLMs are commonly used.
Finally, our results highlight that reducing the energy consump-
tion in RAG systems is not difficult, as 7 of 9 treatments achieved
this. However, only two techniques could do so without unaccept-
able trade-offs. This underscores that the real challenge in Green AI
isachieving energy efficiency, i.e., reducing energy consumption
without harming other important QAs, especially accuracy. Future
research will have to focus on such techniques, as practitioners will
not adopt techniques with substantial drawbacks.
6 Threats to Validity
We assess potential threats to the validity of our experiment fol-
lowing the framework of Wohlin et al. [44].
Internal Validity.Chunking is critical for RAG systems, as
larger chunks can improve recall but may reduce precision [ 34].
We used the TokenTextSplitter of Spring AI with a chunk size of
320 tokens to balance model compatibility and manageable chunk
counts. Despite this, 10 out of 501,916 chunks failed, which could
marginally affect accuracy if they contained relevant information.
Experiments were conducted on a university lab machine with
controlled access via a shared Google calendar and a monitoring
script that automatically logged out unauthorized users. Addition-
ally, we stopped unnecessary background processes before the ex-
periment. Nevertheless, some residual background processes cannot
be fully excluded. Environmental factors, such as room temperature,
may also affect components sensitive to thermal throttling, such as
GPUs. To reduce the impact of potential short-term fluctuations,
we ran each treatment 10 times with sufficient duration to allow
stabilization. We also executed a warm-up dataset of 100 queries be-
fore each trial to further stabilize the system. We therefore believe
our energy and latency measurements to be fairly reliable.
LLM-as-Judge approaches enable efficient accuracy evaluations
of complex NLP tasks, but they are also subject to potential LLM
hallucinations. While our DeepSeek-V2 judge reported a similar
accuracy for the baseline system as previous benchmarks, it is pos-
sible that some answers were evaluated wrongly. Overall, we still
believe that the accuracy changes between treatments are reliable,
even though the absolute numbers could be slightly different.External Validity.Our setup mirrored a production RAG sys-
tem at SIG, and our close collaboration ensured realistic design
choices. However, alternative implementations exist, such as run-
ning all components in a single process, as in the CRAG benchmark.
Experiments were conducted in a controlled, on-premise environ-
ment with fixed hardware (24 GB VRAM GPUs), which may limit
generalization to cloud or distributed deployments, where features
like microservice autoscaling can improve latency. While our re-
sults should be transferable to similar chatbot RAG systems, we
have to be careful with broader generalization.
Conclusion Validity.To control Type I errors from multiple
comparisons, we applied the Holm-Bonferroni correction. Each
treatment was also executed 10 times, yielding stable, normally dis-
tributed measurements. While more trials could improve stability,
the current setup was sufficient to support valid conclusions.
7 Conclusion
In this paper, we provided a thorough analysis of proposed tech-
niques that could reduce the energy consumption of RAG systems
while also studying potential trade-offs with latency and accuracy.
We identified valuable findings for researchers and practitioners,
such as the importance of selecting an optimal threshold in the
retrieval stage. An appropriately chosen similarity threshold (T1)
can significantly reduce energy consumption and latency without
compromising accuracy. However, setting the threshold too high
may severely degrade answer quality. Similarly, using smaller em-
bedding models (T3) is beneficial when handling short queries and
a large number of retrieved documents, offering energy and latency
benefits with no loss in accuracy and, in some cases, even a slight
increase. We also showed that applying indexing strategies like
IVFFlat and HNSW in pgvector (T4) or integrating a lightweight
reranker (T2) can significantly reduce energy consumption and la-
tency. Nonetheless, these enhancements come at the cost of reduced
accuracy, highlighting a crucial trade-off that future research must
continue to explore. Our results can support RAG practitioners in
substantially optimizing energy consumption, which has important
societal implications.
Since the energy efficiency of RAG systems is a relatively new
research area, future work should build upon our findings to explore
this important topic further. For instance, examining combinations
of existing techniques may reveal new patterns of interaction. This
could lead to the discovery of more effective configurations that
optimize energy usage without sacrificing system performance
or accuracy. To support such studies, we make all experimental
artifacts and implementation details publicly available.16
Acknowledgments
This experiment was conducted with the support of the Green Lab
at VU Amsterdam and additional colleagues at SIG. We gratefully
acknowledge everyone’s assistance with this study.
References
[1]Md Adnan Arefeen, Biplob Debnath, Md Yusuf Sarwar Uddin, and Srimat Chakrad-
har. 2024. iRAG: Advancing RAG for Videos with an Incremental Approach. In
Proceedings of the 33rd ACM International Conference on Information and Knowl-
edge Management. 4341–4348.
16https://doi.org/10.5281/zenodo.17990845
10

On the Effectiveness of Proposed Techniques to Reduce Energy Consumption in RAG Systems ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil
[2]Mitchell J. Barnett, Shadi. Doroudgar, Vista. Khosraviani, and Eric J. Ip. 2022.
Multiple comparisons: To compare or not to compare, that is the question.Re-
search in Social and Administrative Pharmacy18, 2 (2022), 2331–2334. doi:10.
1016/j.sapharm.2021.07.006
[3]Noman Bashir, Priya Donti, James Cuff, Sydney Sroka, Marija Ilic, Vivienne Sze,
Christina Delimitrou, and Elsa Olivetti. 2024. The Climate and Sustainability
Implications of Generative AI.An MIT Exploration of Generative AI(mar 27 2024).
https://mit-genai.pubpub.org/pub/8ulgrckc.
[4]Vitaly Bulgakov. 2024. Optimization of Retrieval-Augmented Generation Context
with Outlier Detection. arXiv:2407.01403 [cs.IR] https://arxiv.org/abs/2407.01403
[5]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao
Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang,
Philip S. Yu, Qiang Yang, and Xing Xie. 2024. A Survey on Evaluation of Large
Language Models.ACM Transactions on Intelligent Systems and Technology15, 3
(June 2024), 1–45. doi:10.1145/3641289
[6]Yihua Cheng, Kuntai Du, Jiayi Yao, and Junchen Jiang. 2024. Do Large Language
Models Need a Content Delivery Network?arXiv preprint arXiv:2409.13761
(2024).
[7]Jacob Cohen. 2013.Statistical power analysis for the behavioral sciences. routledge.
[8]DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang
Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli
Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting
Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui
Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong
Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai
Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia,
Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua
Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng
Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe
Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang
Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou,
Shuiping Yu, Shunfeng Zhou, Size Zheng, T. Wang, Tian Pei, Tian Yuan, Tianyu
Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun
Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong
Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie,
Xiaowen Sun, Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi
Zhou, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu,
Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui
Wang, Yi Zheng, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Ying Tang,
Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo,
Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun Zha, Yunxian Ma, Yuting Yan,
Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhen Huang,
Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen, Zhipeng
Xu, Zhongyu Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei
Xie. 2024. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts
Language Model. arXiv:2405.04434 [cs.CL] https://arxiv.org/abs/2405.04434
[9]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[10] Bahar Gezici and Ayça Kolukısa Tarhan. 2022. Systematic literature review on
software quality for AI-based software.Empirical Software Engineering27, 3
(2022), 66. doi:10.1007/s10664-021-10105-2
[11] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu,
Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang,
Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. 2025. A Survey on LLM-as-
a-Judge. arXiv:2411.15594 [cs.CL] https://arxiv.org/abs/2411.15594
[12] Weiwei Gu, Aditya Tandon, Yong-Yeol Ahn, and Filippo Radicchi. 2021. Principled
approach to the selection of the embedding dimension of networks.Nature
Communications12, 1 (2021), 3772.
[13] Zofia Hanusz, Joanna Tarasinska, and Wojciech Zielinski. 2016. Shapiro–Wilk
Test with Known Mean.REVSTAT-Statistical Journal14, 1 (Feb. 2016), 89–100.
doi:10.57805/revstat.v14i1.180
[14] Elias Herranen. 2024. Sustainable and Efficient Vector Search Solutions: A Com-
parative Analysis of Quantization Techniques on Multilingual Text Embeddings.
(2024).
[15] Sture Holm. 1979. A simple sequentially rejective multiple test procedure.Scan-
dinavian journal of statistics(1979), 65–70.
[16] Xixin Hu, Xuan Wu, Yiheng Shu, and Yuzhong Qu. 2022. Logical Form Generation
via Multi-task Learning for Complex Question Answering over Knowledge Bases.
InProceedings of the 29th International Conference on Computational Linguistics,
Nicoletta Calzolari, Chu-Ren Huang, Hansaem Kim, James Pustejovsky, Leo Wan-
ner, Key-Sun Choi, Pum-Mo Ryu, Hsin-Hsi Chen, Lucia Donatelli, Heng Ji, Sadao
Kurohashi, Patrizia Paggio, Nianwen Xue, Seokhwan Kim, Younggyun Hahm,
Zhong He, Tony Kyungil Lee, Enrico Santus, Francis Bond, and Seung-Hoon
Na (Eds.). International Committee on Computational Linguistics, Gyeongju,
Republic of Korea, 1687–1696. https://aclanthology.org/2022.coling-1.145/[17] Xin Huang, Jung-Jae Kim, and Bowei Zou. 2021. Unseen Entity Handling in
Complex Question Answering over Knowledge Base via Language Generation.
InFindings of the Association for Computational Linguistics: EMNLP 2021, Marie-
Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.).
Association for Computational Linguistics, Punta Cana, Dominican Republic,
547–557. doi:10.18653/v1/2021.findings-emnlp.50
[18] International Energy Agency. 2024. Electricity 2024. https://www.iea.org/reports/
electricity-2024. Licence: CC BY 4.0.
[19] Heli Järvenpää, Patricia Lago, Justus Bogner, Grace Lewis, Henry Muccini,
and Ipek Ozkaya. 2024. A Synthesis of Green Architectural Tactics for ML-
Enabled Systems. InProceedings of the 46th International Conference on Soft-
ware Engineering: Software Engineering in Society(Lisbon, Portugal)(ICSE-
SEIS’24). Association for Computing Machinery, New York, NY, USA, 130–141.
doi:10.1145/3639475.3640111
[20] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination
in Natural Language Generation.Comput. Surveys55, 12 (March 2023), 1–38.
doi:10.1145/3571730
[21] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Xin Liu, Xuanzhe Liu, and Xin
Jin. 2024. RAGCache: Efficient Knowledge Caching for Retrieval-Augmented
Generation. arXiv:2404.12457 [cs.DC] https://arxiv.org/abs/2404.12457
[22] Lynn H. Kaack, Priya L. Donti, Emma Strubell, George Kamiya, Felix Creutzig,
and David Rolnick. 2022. Aligning artificial intelligence with climate change
mitigation.Nature Climate Change12, 6 (2022), 518–527. doi:10.1038/s41558-
022-01377-7
[23] Maggiore Alica Kartiyanta, Eugenia Ancilla, and Kenny Jingga. 2025. Performance
Evaluation for Cost-Effective Retrieval Process for Multi-Document Retrieval-
Augmented Generation on a Domain-Specific Dataset. In2025 IEEE International
Conference on Industry 4.0, Artificial Intelligence, and Communications Technology
(IAICT). 719–725. doi:10.1109/IAICT65714.2025.11101522
[24] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Mem-
ory Management for Large Language Model Serving with PagedAttention. In
Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.
[25] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAttention.
arXiv:2309.06180 [cs.LG] https://arxiv.org/abs/2309.06180
[26] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng
Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, et al .2024.
Cachegen: Kv cache compression and streaming for fast large language model
serving. InProceedings of the ACM SIGCOMM 2024 Conference. 38–56.
[27] Songshuo Lu, Hua Wang, Yutian Rong, Zhi Chen, and Yaohua Tang. 2024.
TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed
KV Caches for Chunked Text. arXiv:2410.07590 [cs.CV] https://arxiv.org/abs/
2410.07590
[28] Sasha Luccioni, Yacine Jernite, and Emma Strubell. 2024. Power Hungry Pro-
cessing: Watts Driving the Cost of AI Deployment?. InThe 2024 ACM Con-
ference on Fairness, Accountability, and Transparency (FAccT ’24). ACM, 85–99.
doi:10.1145/3630106.3658542
[29] Xing Han Lù. 2024. BM25S: Orders of magnitude faster lexical search via eager
sparse scoring. arXiv:2407.03618 [cs.IR] https://arxiv.org/abs/2407.03618
[30] Yu A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.
IEEE Transactions on Pattern Analysis and Machine Intelligence42, 4 (2020), 824–
836. doi:10.1109/TPAMI.2018.2889473
[31] Philipp Moritz, Robert Nishihara, Stephanie Wang, Alexey Tumanov, Richard
Liaw, Eric Liang, Melih Elibol, Zongheng Yang, William Paul, Michael I. Jordan,
and Ion Stoica. 2018. Ray: A Distributed Framework for Emerging AI Applications.
arXiv:1712.05889 [cs.DC] https://arxiv.org/abs/1712.05889
[32] Marius Muja and David G. Lowe. 2014. Scalable Nearest Neighbor Algorithms
for High Dimensional Data.IEEE Transactions on Pattern Analysis and Machine
Intelligence36, 11 (2014), 2227–2240. doi:10.1109/TPAMI.2014.2321376
[33] Md Rizwan Parvez, Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and
Kai-Wei Chang. 2021. Retrieval Augmented Code Generation and Summarization.
arXiv:2108.11601 [cs.SE] https://arxiv.org/abs/2108.11601
[34] Robin D. Pesl, Jerin G. Mathew, Massimo Mecella, and Marco Aiello. 2025.
Retrieval-Augmented Generation for Service Discovery: Chunking Strategies
and Benchmarking. arXiv:2505.19310 [cs.SE] https://arxiv.org/abs/2505.19310
[35] Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani,
Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Mail-
lard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021. KILT: a
Benchmark for Knowledge Intensive Language Tasks. arXiv:2009.02252 [cs.CL]
https://arxiv.org/abs/2009.02252
[36] Anand Rajaraman and Jeffrey David Ullman. 2011.Data Mining. Cambridge
University Press, 1–17.
11

ICSE-SEIS ’26, April 12–18, 2026, Rio de Janeiro, Brazil Guo et al.
[37] Vikas Raunak, Vivek Gupta, and Florian Metze. 2019. Effective Dimensionality
Reduction for Word Embeddings. InProceedings of the 4th Workshop on Represen-
tation Learning for NLP (RepL4NLP-2019), Isabelle Augenstein, Spandana Gella,
Sebastian Ruder, Katharina Kann, Burcu Can, Johannes Welbl, Alexis Conneau,
Xiang Ren, and Marek Rei (Eds.). Association for Computational Linguistics,
Florence, Italy, 235–243. doi:10.18653/v1/W19-4328
[38] Sahel Sharifymoghaddam, Ronak Pradeep, Andre Slavescu, Ryan Nguyen, An-
drew Xu, Zijian Chen, Yilin Zhang, Yidi Chen, Jasper Xian, and Jimmy Lin. 2025.
RankLLM: A Python Package for Reranking with LLMs. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information
Retrieval(Padua, Italy)(SIGIR ’25). Association for Computing Machinery, New
York, NY, USA, 3681–3690. doi:10.1145/3726302.3730331
[39] Student. 1908. The probable error of a mean.Biometrika(1908), 1–25.
[40] Jinyan Su, Jennifer Healey, Preslav Nakov, and Claire Cardie. 2025. Fast or Better?
Balancing Accuracy and Cost in Retrieval-Augmented Generation with Flexible
User Control. arXiv:2502.12145 [cs.IR] https://arxiv.org/abs/2502.12145
[41] Roberto Verdecchia, June Sallou, and Luis Cruz. 2023. A Systematic Review of
Green AI.WIREs Data Mining and Knowledge Discovery13, 4 (July 2023), e1507.
doi:10.1002/widm.1507
[42] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu,
Tianyuan Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng Yin, Changze
Lv, Xiaoqing Zheng, and Xuanjing Huang. 2024. Searching for Best Practices in
Retrieval-Augmented Generation. arXiv:2407.01219 [cs.CL] https://arxiv.org/
abs/2407.01219
[43] Zichao Wang, Weili Nie, Zhuoran Qiao, Chaowei Xiao, Richard Baraniuk, and
Anima Anandkumar. 2023. Retrieval-based Controllable Molecule Generation.
arXiv:2208.11126 [q-bio.QM] https://arxiv.org/abs/2208.11126
[44] C. Wohlin, P. Runeson, M. Höst, M.C. Ohlsson, B. Regnell, and A. Wesslén. 2012.
Experimentation in Software Engineering - An Introduction. Kluwer Academic
Publishers.
[45] Hui Wu, Xiaoyang Wang, and Zhong Fan. 2025. Addressing the sustainable
AI trilemma: a case study on LLM agents and RAG. arXiv:2501.08262 [cs.CY]
https://arxiv.org/abs/2501.08262
[46] Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. 2025. Hallucination is Inevitable:
An Innate Limitation of Large Language Models. arXiv:2401.11817 [cs.CL] https://arxiv.org/abs/2401.11817
[47] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal
Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong,
Brian Moran, Jiaqi Wang, Yifan Ethan Xu, An Yan, Chenyu Yang, Eting Yuan,
Hanwen Zha, Nan Tang, Lei Chen, Nicolas Scheffer, Yue Liu, Nirav Shah, Rakesh
Wanga, Anuj Kumar, Wen-tau Yih, and Xin Luna Dong. 2024. CRAG - Compre-
hensive RAG Benchmark. InAdvances in Neural Information Processing Systems,
A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang
(Eds.), Vol. 37. Curran Associates, Inc., 10470–10490. https://proceedings.
neurips.cc/paper_files/paper/2024/file/1435d2d0fca85a84d83ddcb754f58c29-
Paper-Datasets_and_Benchmarks_Track.pdf
[48] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
Kuntai Du, Shan Lu, and Junchen Jiang. 2024. CacheBlend: Fast Large Language
Model Serving with Cached Knowledge Fusion.arXiv preprint arXiv:2405.16444
(2024).
[49] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng,
Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2024.
Retrieval-Augmented Generation for AI-Generated Content: A Survey.
arXiv:2402.19473 [cs.CV] https://arxiv.org/abs/2402.19473
[50] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhang-
hao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing,
Hao Zhang, Joseph E Gonzalez, and Ion Stoica. 2023. Judging LLM-
as-a-Judge with MT-Bench and Chatbot Arena. InAdvances in Neu-
ral Information Processing Systems, A. Oh, T. Naumann, A. Globerson,
K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36. Curran Associates,
Inc., 46595–46623. https://proceedings.neurips.cc/paper_files/paper/2023/file/
91f18a1287b398d378ef22505bf41832-Paper-Datasets_and_Benchmarks.pdf
[51] Shuyan Zhou, Uri Alon, Frank F Xu, Zhiruo Wang, Zhengbao Jiang, and Graham
Neubig. 2022. Docprompting: Generating code by retrieving the docs.arXiv
preprint arXiv: 2207.05987(2022).
[52] Tolga Şakar and Hakan Emekci. 2025. Maximizing RAG efficiency: A comparative
analysis of RAG methods.Natural Language Processing31, 1 (2025), 1–25. doi:10.
1017/nlp.2024.53
12