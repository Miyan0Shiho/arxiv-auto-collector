# Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks

**Authors**: Qiang Ke, Yanjie Zhao, Hongjin Leng, Shengming Zhao, Haoyu Wang

**Published**: 2026-05-14 07:47:44

**PDF URL**: [https://arxiv.org/pdf/2605.14503v1](https://arxiv.org/pdf/2605.14503v1)

## Abstract
While Retrieval-Augmented Generation (RAG) is increasingly adopted to ground Large Language Models (LLMs) in software artifacts, the optimal configuration of its components remains an open question for software engineering (SE) tasks. The lack of systematic guidance forces practitioners into costly, ad-hoc experimentation. This paper presents a comprehensive, component-wise empirical study that dissects the RAG pipeline, evaluating over 21 distinct models and methods. Our study systematically isolates and evaluates 4 query processing techniques, 7 retrieval models spanning sparse, dense, and hybrid paradigms, 4 context refinement methods, and 6 distinct generators. We test these components on a suite of 3 core SE tasks: code generation, summarization, and repair. Our empirical findings reveal a crucial insight: the retriever-side components, particularly the choice of the retrieval algorithm, often exert a more significant influence on final system performance than the selection of the generator model. Strikingly, the classic lexical retriever BM25 demonstrates exceptionally robust performance across diverse tasks. Our analysis provides a practical, data-driven roadmap for researchers and practitioners, offering clear guidance on prioritizing optimization efforts when constructing effective RAG systems for software engineering contexts.

## Full Text


<!-- PDF content starts -->

Not All RAGs Are Created Equal: A Component-Wise
Empirical Study for Software Engineering Tasks
QIANG KE,Huazhong University of Science and Technology, China
YANJIE ZHAO∗,Huazhong University of Science and Technology, China
HONGJIN LENG,Xiamen University Malaysia, Malaysia
SHENGMING ZHAO,Fudan University, China
HAOYU WANG,Huazhong University of Science and Technology, China
While Retrieval-Augmented Generation (RAG) is increasingly adopted to ground Large Language Models
(LLMs) in software artifacts, the optimal configuration of its components remains an open question for
software engineering (SE) tasks. The lack of systematic guidance forces practitioners into costly, ad-hoc
experimentation. This paper presents a comprehensive, component-wise empirical study that dissects the RAG
pipeline, evaluating over21distinct models and methods. Our study systematically isolates and evaluates
4query processing techniques,7retrieval models spanning sparse, dense, and hybrid paradigms,4context
refinement methods, and6distinct generators. We test these components on a suite of3core SE tasks: code
generation, summarization, and repair. Our empirical findings reveal a crucial insight: the retriever-side
components, particularly the choice of the retrieval algorithm, often exert a more significant influence on
final system performance than the selection of the generator model. Strikingly, the classic lexical retriever
BM25demonstrates exceptionally robust performance across diverse tasks. Our analysis provides a practical,
data-driven roadmap for researchers and practitioners, offering clear guidance on prioritizing optimization
efforts when constructing effective RAG systems for software engineering contexts.
CCS Concepts:•Software and its engineering;
Additional Key Words and Phrases: retrieval-augmented methods, software engineering, empirical study
ACM Reference Format:
Qiang Ke, Yanjie Zhao, Hongjin Leng, Shengming Zhao, and Haoyu Wang. 2026. Not All RAGs Are Created
Equal: A Component-Wise Empirical Study for Software Engineering Tasks.Proc. ACM Softw. Eng.3, FSE,
Article FSE183 (July 2026), 23 pages. https://doi.org/10.1145/3808190
1 Introduction
Retrieval-Augmented Generation (RAG) has become a widely adopted paradigm for enhancing
Large Language Models (LLMs) in software engineering applications [ 14,26]. By incorporating
external knowledge bases such as project-specific documentation and code repositories, RAG
∗Corresponding author (Yanjie_Zhao@hust.edu.cn).
Authors’ Contact Information: Qiang Ke, Hubei Key Laboratory of Distributed System Security, Hubei Engineering Research
Center on Big Data Security, School of Cyber Science and Engineering, Huazhong University of Science and Technology,
Wuhan, China, qiangke@hust.edu.cn; Yanjie Zhao, Hubei Key Laboratory of Distributed System Security, Hubei Engineering
Research Center on Big Data Security, School of Cyber Science and Engineering, Huazhong University of Science and
Technology, Wuhan, China, Yanjie_Zhao@hust.edu.cn; Hongjin Leng, Xiamen University Malaysia, Sepang, Malaysia,
CYS2409015@xmu.edu.my; Shengming Zhao, Fudan University, Shanghai, China, smzhao25@m.fudan.edu.cn; Haoyu Wang,
Hubei Key Laboratory of Distributed System Security, Hubei Engineering Research Center on Big Data Security, School of
Cyber Science and Engineering, Huazhong University of Science and Technology, Wuhan, China, haoyuwang@hust.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and
the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses,
contact the owner/author(s).
©2026 Copyright held by the owner/author(s).
ACM 2994-970X/2026/7-ARTFSE183
https://doi.org/10.1145/3808190
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.arXiv:2605.14503v1  [cs.SE]  14 May 2026

FSE183:2 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
reduces factual inaccuracies and improves the contextual relevance of generated outputs for code-
related tasks [34, 38].
However, constructing optimal RAG pipelines for software engineering tasks lacks system-
atic guidance. Software engineering presents unique challenges for RAG systems, including the
structured nature of code [ 2], diverse query types ranging from natural language questions to
code snippets [ 39,43], and the need to retrieve from heterogeneous knowledge sources such as
documentation, API references, and code repositories [ 54]. A RAG system consists of multiple
interconnected components, each presenting numerous design options [ 14]. Practitioners must
make critical decisions regarding query processing, retrieval algorithms, context refinement, and
generator model selection. The absence of systematic, empirical evidence to inform these choices
specifically for software engineering contexts results in a costly, iterative experimentation process.
The limited body of prior empirical work on RAG for code, while providing valuable initial
insights, is constrained by a narrow scope. Existing studies have focused almost exclusively on
the single task of code generation, investigating specific aspects of the retrieval stage, such as the
effectiveness of different information sources like APIs [ 17] or the quality and fusion of retrieved
code snippets [ 52]. Consequently, their findings, while important, are confined to a single component
and a single task.
To address this limitation, this paper presents a systematic, component-wise empirical study
of RAG pipelines for software engineering tasks. Our investigation methodically examines the
pipeline to isolate and evaluate the impact of core architectural decisions. We aim to establish a
data-driven understanding of how each component contributes to overall system performance. To
guide our study, we formulate the following four research questions:
RQ1: How does the query processing stage impact performance?We investigate whether
transforming the user’s initial query, for instance, by simplifying or elaborating it, can enhance the
retrieval of relevant information and improve the quality of the final output.
RQ2: Which retrieval strategy is most effective for code tasks?This question examines the
core of the RAG pipeline by comparing different retrieval paradigms: sparse (e.g., keyword-based),
dense (e.g., semantic-based), and hybrid approaches to determine which is best suited for the unique
characteristics of code and natural language queries in software engineering.
RQ3: What is the utility of the context refinement stage?We evaluate post-retrieval
techniques such as re-ranking and compression. This inquiry seeks to understand whether these
optional steps effectively increase the signal-to-noise ratio for the generator or if they risk discarding
critical information.
RQ4: What is the relative importance of the generator versus the retriever?The final
question assesses the interplay between the retrieval and generation stages. We aim to determine
whether final performance is more sensitive to the quality of the retrieved context or the intrinsic
capabilities of the generative LLMs.
Through answering these questions, this work makes the following principal contributions:
•We design and implement amodular testbedfor RAG systems in software engineering
(hereafter referred to as Code RAG) that decouples the pipeline’s core stages, enabling flexible
and reproducible component-wise experimentation.
•We conduct alarge-scale empirical study, evaluating over20distinct models across3
diverse code tasks on4datasets to build a foundational knowledge base for practitioners.
•We develop and open-source aprototype adaptive RAG frameworkthat leverages our
empirical findings to automatically recommend a near-optimal pipeline configuration based
on task-specific features.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:3
2 Background and Motivations
The adoption of RAG has been pivotal in advancing the capabilities of LLMs for SE. By grounding
generation in targeted, real-time context, RAG promises to make LLMs more reliable development
partners. However, realizing this potential is hindered by several fundamental challenges that
motivate our work.
Challenge 1: Performance Instability and Context-Dependency.A primary challenge is
RAG’s performance instability across different SE tasks. Our preliminary experiments, corroborated
by findings in the literature [ 14], reveal that no single RAG configuration is universally optimal.
The ideal choice of components is highly dependent on the task context. For instance, a pipeline
tuned for code generation (text-to-code) may fail on code repair (code-to-code), meaning success on
one benchmark rarely guarantees generalizability to diverse development scenarios.
Challenge 2: The Expanding and Evolving RAG Design Space.The RAG design space
is exponentially growing. The classic pipeline forces choices among sparse, dense, or hybrid
retrievers [ 9,25], alongside post-retrieval refinements like re-ranking or compression [ 23,41].
Furthermore, while advanced, multi-step agentic paradigms are emerging [ 3,11], they fundamen-
tally build upon these classic components. Their complexity makes a systematic, component-wise
evaluation intractable without first establishing a foundational understanding of the core building
blocks. Therefore, this study focuses on providing this essential empirical foundation for the classic
RAG pipeline, which remains the most widely adopted and fundamental architecture in practice.
The vast and poorly understood set of trade-offs within even this classic design makes it nearly
impossible to select an optimal configuration without systematic guidance.
Challenge 3: Lack of Systematic Empirical Guidance.The majority of existing research
introduces a novel component or a new architectural pattern and demonstrates its effectiveness on
specific benchmarks. While such studies are valuable, they contribute to a fragmented understanding
of the RAG ecosystem. Even comprehensive surveys that categorize RAG techniques [ 14] often
do not provide comparative empirical data on how different componentcombinationsperform
across avarietyof SE task characteristics. Consequently, practitioners are forced to rely on costly
trial-and-error. The field urgently needs an empirical roadmap mapping task features to optimal
configurations.
Our Approach.To address these challenges, we undertake asystematic, component-wise
empirical studymapping the RAG design space for SE. By rigorously evaluating and comparing a
wide array of components in a controlled testbed, we aim to transform the “alchemy” of current
RAG development into a more principled, scientific practice. Our goal is to provide the empirical
foundations necessary for building more effective and reliable RAG systems in SE.
3 Approach
This section details the methodology for our empirical study of RAG on code tasks. We first establish
the foundations of our study by compiling a suite of code tasks and constructing a large-scale
retrieval corpus. Next, we introduce our modular testbed, a plug-and-play architecture designed
for rigorous, component-wise experimentation. Finally, guided by our experimental insights, we
propose a prototype of an LLM-driven adaptive framework that recommends an optimal pipeline
configuration tailored to specific task features. The overall architecture is illustrated in Figure 1.
3.1 Foundations of the Empirical Study
Our empirical study is built upon two foundations: a curated set of code tasks (processed in § 4.1.1)
and a large-scale retrieval corpus (constructed in § 4.1.2). These were designed to systematically
evaluate RAG components across diverse development scenarios.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:4 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
Filtering
Corpus DatasetSourcing
Query Processing Stage
——   RQ1  ——
Query Rewriting
•Keyword Extraction
•Query Expansion
Retrieval Stage
——   RQ2  ——
Sparse Retrieval
Dense Retrieval
•Text-based
•Code -based 
Hybrid RetrievalContext Refinement Stage
——   RQ3  ——
Reranking
Compression
Generation Stage
——   RQ4  ——
LLM Generator
Code Summary Code Repair Code GenerationCode Tasks
Query Transformation
LLMlingua2
RecompEncoder -only
Decoder -onlyDeepSeek -V3
Llama 3.3
Phi-4
Qwen2.5
•Instruct
•Coder
Empirical Rules
Dataset profile
 Code tasks
Code generation: converting queries into more detailed queries
Code summary: using hybrid retrieval with text dense retrieval
Specific rules Final answers
RAGNormalization
Language & Quality
Relevance & Tokens
Simplification
Standardization
Required
Optional
Retrieval Paradigms
Query Augmentation
•HyDE
Fig. 1. The overall architecture of our RAG framework.
3.1.1 Code Tasks.Our evaluation suite comprises three fundamental tasks.Code Generationis
atext-to-codetask that synthesizes code from natural language, testing the retrieval of relevant
algorithms and APIs.Code Summarization, acode-to-texttask, generates a concise description for
a code snippet to assess semantic comprehension. Finally,Code Repairis a complexcode-to-code
task where the system must understand buggy code and retrieve correct patterns to guide fixes.
Crucially, rather than being isolated applications, these tasks were strategically selected as
representative proxies for the three foundational input-output modalities in SE [ 7,15,29]. Conse-
quently, empirical insights derived from this triad inherently generalize to a broader spectrum of
SE scenarios, such as automated test generation (text-to-code), documentation writing (code-to-text),
and code refactoring (code-to-code).
3.1.2 Retrieval Corpus.A comprehensive, up-to-date corpus suitable for RAG across our diverse
evaluation tasks is not readily available. We therefore constructed a new, large-scale knowledge
source by integrating and processing data from five authoritative sources: Stack Overflow, GitHub,
LeetCode, Python API docstrings, and Hugging Face. To ensure the integrity of our evaluation and
prevent data leakage, we implemented a strict decontamination. We removed any document from
our constructed corpus if it contained code that was an exact match to any code solution present in
the test splits of our evaluation datasets. This process yielded a realistic and decontaminated corpus
with a wide spectrum of coding styles and complexities, essential for a robust retrieval evaluation.
3.2 The Modular Testbed for Code RAG
At the heart of our empirical study is a modular testbed for Code RAG, designed withmodularity
andplug-and-playcapability as its core principles [ 47]. This allows us to isolate, evaluate, and
combine various RAG components seamlessly. The testbed decouples the RAG workflow into
distinct, interchangeable stages, a fundamental architecture for conducting controlled experiments
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:5
and attributing performance changes directly to the component being evaluated. The key stages
are detailed below.
3.2.1 Query Processing Stage.This optional stage investigates whether refining raw queries en-
hances retrieval and downstream performance. Our testbed implements two primary refinement
paradigms. To ensure robust findings unbiased by specific model characteristics, we employ the
open-weightDeepSeek-V3[ 10] and proprietaryGPT-4o[ 33] as core engines. Their powerful rea-
soning capabilities ensure high-fidelity query transformations. The evaluation is detailed in RQ1
(§ 4.3).
Query Transformation.This approach rewrites the query to improve its semantic clarity and
keyword relevance, inspired by recent work [30]. We explore two distinct strategies:
•Simplified Query:This method condenses the problem description into a concise format,
retaining only core algorithmic keywords and essential constraints to reduce noise.
•Elaborated Query:In contrast, this method expands the query into a more structured
format via chain-of-thought prompting [ 49]. The model is instructed to include detailed I/O
specifications, potential edge cases, and a pseudo-code sketch of a solution. This implicitly
incorporates the core principle ofHyDE[ 13] by proactively generating ahypothetical answer
within the query itself.
Query Augmentation.In contrast to rewriting, this method enriches the query by appending
a known, correct solution. While the originalHyDEframework [ 13] generates a hypothetical
document, our implementation adapts this for an experimental setting by using the ground-truth
answer. This serves as an oracle-based approach to create a semantically ideal query embedding,
providing a practical upper bound on performance achievable through query enrichment.
3.2.2 Retrieval Stage.The Retrieval Stage is a mandatory step that recalls an initial set of candidate
documents from the code corpus. Our framework systematically evaluates three distinct retrieval
paradigms. For each paradigm, we select a suite of representative and commonly used models,
which are detailed below and summarized in Table 1. Their performance is analyzed in RQ2 (§ 4.4).
Sparse Retrieval.This paradigm relies on classic, non-deep-learning models that excel at exact
keyword matching based on term frequency (TF) and inverse document frequency (IDF) [ 40]. In
our study,BM25[ 36] serves as a crucial lexical baseline, representing the robust performance of
term-based search.
Dense Retrieval.This paradigm utilizes deep learning models for semantic search, encoding
queries and documents into high-dimensional vectors (embeddings) within a shared space [ 25]. We
use the same model for both queries and documents to ensure a consistent mapping. Embeddings
are indexed using the Faiss library [ 24] with theIndexFlatL2algorithm to ensure exact similarity
search. We evaluate two categories of embedders:
•General-purpose Embedders:Pre-trained on diverse text corpora [ 46], these models possess
a broad understanding of natural language semantics.
•Code-specialized Embedders:Fine-tuned on large-scale source code [ 12], these models
are trained to understand the nuances of programming languages for better retrieval of
functionally relevant code.
Hybrid Retrieval.This approach combines the outputs of sparse and dense retrieval to syn-
thesize their respective strengths. We use the Reciprocal Rank Fusion (RRF) algorithm [ 9], which
computes a new score for each document𝑑as follows:
Score𝑅𝑅𝐹(𝑑)=∑︁
𝑟∈𝑅1
𝑘+rank𝑟(𝑑)(1)
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:6 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
Table 1. Retrieval models evaluated in the study.
Retrieval Paradigm Model Type Model Name Parameter
Sparse RetrievalN/A BM25 N/A
Dense RetrievalGeneral-purposeintfloat/multilingual-e5-small (E5) [45] 118M
ibm-granite/granite-embedding-278m-multilingual (Granite) [31] 278M
Alibaba-NLP/gte-multilingual-base (GTE) [27] 305M
BAAI/bge-m3 (BGE) [6] 1.3B
Code-specializedjinaai/jina-embeddings-v2-base-code (Jina) [18] 161M
Salesforce/SFR-Embedding-Code-400M_R (SFR) [28] 434M
where 𝑅is the set of ranked lists, rank𝑟(𝑑)is the rank of document 𝑑in list 𝑟, and 𝑘is a
smoothing constant. In our experiments, we set 𝑘=60, following the established academic and
industry standard [9], as it is mathematically proven to optimally mitigate the influence of lower-
ranked outlier documents without requiring further parameter tuning. We fuse the ranked list from
BM25with that of the top-performing dense retriever from our analysis.
3.2.3 Context Refinement Stage.This optional stage refines retrieved context to improve the signal-
to-noise ratio for the LLM. We systematically evaluate two influential post-retrieval techniques:
Re-ranking and Compression, with their effectiveness investigated in RQ3 (§ 4.5).
Re-ranking.This process employs a more powerful but computationally intensive model to
re-evaluate the top- 𝑘(where 𝑘=20, a standard threshold balancing computational overhead and
recall ceiling) candidates from the initial retrieval, aiming for a more precise relevance ordering [ 32].
We contrast two dominant architectural paradigms:
•Encoder-only:Conventional cross-encoders concatenate the query and a document, allowing
a bidirectional encoder like XLM-RoBERTa to perform deep, token-level interaction [ 8]. We
use the state-of-the-artBAAI/bge-reranker-v2-m3[6].
•Decoder-only:Termed generative re-rankers, these LLMs treat re-ranking as a conditional
scoring task [ 48]. We evaluateQwen/Qwen3-Reranker-0.6B[ 53], a specialized model prompted
to act as a relevance assessor.
Compression.This technique distills retrieved documents into a more concise format to mitigate
issues of limited context windows and noise [ 21]. We evaluate two representative query-aware
methods:
•LLMLingua-2:This technique uses a smaller language model for fine-grained compres-
sion [ 23]. It preserves key information by estimating the perplexity of document tokens
conditioned on the query, selectively removing those that are least essential.
•Zero-shot Recomp Adaptation:We adapt the Recomp framework, a two-stage (extractive-
abstractive) compression method [ 50]. Our zero-shot version (Figure 2) avoids the need
for task-specific fine-tuning. The workflow is sequential: first,Extractive Compression
isolates code blocks and segments text, which is then scored for relevance by a pre-trained
cross-encoder. Second, inAbstractive Compression, the top- 𝑁chunks are synthesized into
a summary byDeepSeek-V3. As shown in Table 2, this provides a clear mapping between the
number of intermediate chunks ( 𝑁) and their total token count, enabling a direct comparison
against the strict token budgets of methods likeLLMLingua-2.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:7
Corpus 
Dataset
Extractive Compression Abstractive CompressionSummaryQueryTop-N 
Chunks
Synthesis 
PromptText  Chunks
[> 15 tokens]
Code Chunks
[isolated]
Cross -Encoder
Top-N [C₁:S₁, C₂:S₂,...]
Fig. 2. The Extractive-Abstractive Pipeline of the Zero-shot Recomp Adaptation.
Table 2. Average token count of the top-𝑁chunks after the extractive compression step.
Number of Chunks (𝑁)10 20 40 80
Avg. Context Tokens1533.47 3578.81 7818.43 17105.42
3.2.4 Generation Stage.In the final stage, a generator synthesizes the query and the context to
produce the output. Our evaluation covers a diverse set of highly capable LLMs to represent a
spectrum of architectures and sizes. Their impact is evaluated in RQ4 (§ 4.6).
•Frontier Generalist Models:To establish the performance ceiling driven by massive rea-
soning capacity, we evaluateGPT-4o[33] andDeepSeek-V3 (671B)[10].
•Efficient Generalist Models:To assess the capabilities of widely adopted open-weight
models across a diverse range of scales and architectures, we evaluateMeta Llama 3.3 (70B)
[16],Qwen2.5-32B-Instruct(Qwen2.5) [51], andPhi-4 (14B)[1].
•Code-Specialized Models:To explicitly measure the effect of domain-specific training on
vast corpora of source code, we evaluateQwen2.5-Coder-32B-Instruct(Qwen2.5-C) [51].
3.3 LLM-driven Adaptive Configuration
To translate our empirical findings into a practical solution, we developed a dynamic, LLM-driven
adaptive RAG framework. Its core objective is to move beyond static, one-size-fits-all pipelines by
autonomously recommending an optimal component configuration for any given SE task. It is built
upon a knowledge base of rules and a two-stage reasoning process.
3.3.1 A Knowledge Base of Empirical Rules.The framework’s foundation is a knowledge base of
high-level rules derived from our empirical study. These rules connect observable task features to
the performance of specific RAG components across the pipeline’s main stages:
•Query Processing Rules:Guide the choice of query transformation or augmentation based
on the query’s initial clarity and detail.
•Retrieval Rules:Determine the optimal retrieval paradigm by considering the task’s I/O
modality (e.g.,text-to-code,code-to-text, orcode-to-code).
•Context Refinement Rules:Govern the activation of optional stages likeRe-rankingand
Compressionbased on factors like query ambiguity and context window.
•Generation Rules:Provide guidance on selecting a generator by weighing the trade-offs
between generalist and code-specialized models based on task complexity.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:8 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
3.3.2 Two-Stage Adaptive Process.To apply this knowledge base automatically, we developed
a prototype adaptive framework. As visualized in Figure 3, this framework employs a dynamic,
two-stage inference process powered by a large language model (DeepSeek-V3).
Stage 1: Automated Task Profiling.The framework parses the natural language description
of an unseen SE task to generate a structuredTask Profile. It evaluates the input across critical
dimensions (query modality, clarity, complexity, and key information type) to accurately capture
the task’s semantic and structural demands.
Stage 2: Knowledge-Driven Configuration.The extracted profile is dynamically mapped against
our codifiedKnowledge Base of Empirical Rules. Governed by the decision trees in Figure 3, the
framework bypasses static defaults to autonomously recommend the optimal pipeline.
By grounding its decisions entirely in our empirical evidence, the framework ensures strict
theoretical alignment. The external validation of this framework’s generalization capabilities is
presented in § 5.2.
Debug a Python data aggregation module where incorrect 
conditional logic causes duplicated records in the final output.
Input & Task Profiling
Natural Language Task Description (Input)
Stage 1: Reasoning Analysis
task_profile :
 task_type : Code Repair
 query_modality : Text-and-Code -to-Code
 task_complexity : Medium
 key_information_type : Semantic Concepts
 primary_success_metric : Functional CorrectnessYAML -formatted Task Profile  (Output)Rule -Based Strategy
System Prompt Summary
Role:  Task Profiling Analyst
Instruction:  Analyze the input and produce a structured 
7-dimension Task Profile in YAML format .
1.0      Query:
Rule 1.1 High Clarity -> Baseline, 
Rule 1.2: Low Clarity -> Query Elaboration
2.0      Retrieval:
Rule 2.2 Precise Keywords -> BM25,
Rule 2.3: Semantic Concepts -> Dense Retrieval
3.0      Refinement:
Rule 3.1: Code Tasks -> Avoid Re -ranking
4.0      Generation:
Rule 4.1: Algorithmic Logic -> Logic -tuned Model
Rule 4.3 k=3 optimal
Codified Knowledge Base
Core Empirical Rules Summary
Role:  Pipeline Strategist
Instruction:  Select optimal components per 
stage based on Task Profile and empirical rules; 
omit detrimental stages.Stage 2: Configuration Strategist System Prompt Instructions
Optimized RAG Pipeline
Reasoning:
Applies Rules to Profile
YAML -formatted Final Pipeline Configuration
(Derived from Profile + Rules)
Query Processing :
 Strategy : "Expansion"
 Rule: "Medium Clarity → Elaboration“
Retrieval :
 Type : "Dense Retrieval (e.g., Contriever )"
 Rule: "Semantic Concepts → Dense“
Context Refinement :
 Strategy : None
 Rule: "Avoid Re -ranking for code generation"
Generation :
 Model : "DeepSeek -V3 (Instruction -tuned)"
 Rule: "Medium/High Complexity → Logic -tuned"
 Context Size (k) : 3
 Note : "Performance peaks at k > 3"
Fig. 3. The Decision Logic of the LLM-Driven Adaptive RAG Framework.
4 Evaluation
4.1 Datasets and Corpora
All experiments focus on the Python language to ensure a controlled environment, leveraging its
widespread adoption and the abundance of high-quality public data.
4.1.1 Task Datasets.We selected three datasets for our evaluation suite, each tailored to a core
code-related task and preprocessed to create focused, balanced evaluation sets.
•Code Generation:We employ the APPS benchmark [ 20], constructing a balanced 300-
problem subset by randomly sampling 100 entries from each of its three difficulty tiers.
•Code Summarization:Using the Python subset of CodeXGLUE [ 29], we explicitly select
the 100 longest snippets (Avg: ∼1,510 tokens vs. Global: ∼206) to mitigate ceiling effects and
stress-test RAG capabilities. To isolate syntactic reliance, we created an obfuscated version
(CodeXGLUE-OBF): using Python’s AST, we popped docstrings and deterministically mapped
four user identifier types (functions, classes, arguments, variables) to sequential placeholders
(e.g., var_1 ), while strictly preserving built-ins, imported modules, attributes, and magic
methods.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:9
•Code Repair:We utilize DebugBench [ 42], curating a 300-sample test set by selecting 25
samples from each of the 12 difficulty-category combinations (coveringsyntax,reference,
logic, andmultiplebug types).
4.1.2 Retrieval Corpora.To construct a comprehensive knowledge base for our RAG system, we
aggregated and processed data from five diverse sources, summarized in Table 3.
Table 3. Document Counts of the Retrieval Corpora
Corpus Stack Overflow Python API LeetCode CodeSearchNet Code-Contests Total
Documents164,085 38,352 3,174 13,590 13,432232,633
The foundation of the corpus combines community-driven knowledge with canonical documen-
tation via strict quantitative filtering. ForStack Overflow, we extracted Python-related posts by
enforcing quality thresholds (Question Score ≥10, Answer Score≥5), retaining only the top-5
answers truncated to 512 tokens. This is complemented byPython APIentries, programmati-
cally parsed from 33 widely adopted industry libraries (e.g., numpy ,pandas ) to serve as functional
references. We curatedLeetCodeby pairing algorithmic problems with 1-2 high-quality refer-
ence solutions, alongside deduplicated function-docstring pairs fromCodeSearchNet. Finally, we
processedCode-Contestsby filtering strictly for Python solutions and flattening the structure,
retaining a maximum of 10 unique problem-solution entries per problem.
As performance generally peaks at 𝑘= 3(§ 5.3.1), we standardize this retrieval depth across
RQ1-RQ3 to strictly isolate the impact of individual RAG components. Additionally, we validate
the necessity of our heterogeneous corpus via source ablation (§ 5.3.2).
4.2 Experimental Setup
4.2.1 Evaluation Metrics.We employ metrics tailored to each task’s unique objectives, citing their
foundational sources to ensure methodological rigor.
ForCode GenerationandCode Repair, we evaluate functional correctness and structural
quality through the following metrics:
•Weighted Pass@1 (W-Pass@1)is based on thePass@k[ 7] which estimates the probability
that at least one of𝑘generated samples is correct.
Pass@k=E Problems"
1− 𝑛−𝑐
𝑘
 𝑛
𝑘#
(2)
where 𝑛is the samples per problem and 𝑐is the number of correct ones. For our experiments
(𝑘= 1), we calculate a weighted average to account for problem difficulty. Informed by
prior work on difficulty-based sample weighting [ 55], we assign a higher weight to more
challenging problems to better reflect their significance. The final metric is:
W-Pass@1=∑︁
𝑝∈𝑃𝑤𝑝·Pass@1𝑝where𝑤 𝑝=𝑑𝑝Í
𝑝′∈𝑃𝑑𝑝′(3)
•CodeBLEUserves as a structure-aware metric forCode Repair. It complements binary
correctness by comparing a solution’s syntactic (AST) and data-flow similarity to the refer-
ence [35]. The metric is a weighted combination of these components:
CodeBLEU=𝛼·BLEU weighted+𝛽·BLEU ngram+𝛾·Match AST+𝛿·Match DF (4)
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:10 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
ForCode Summarization, we evaluate the semantic alignment with the ground truth.
•Semantic Similarity (Sim Emb)computes the cosine similarity between the vector em-
beddings of the candidate and reference summaries. We generate embeddings using the
intfloat/e5-mistral-7b-instructmodel [45].
Sim Emb(𝑐, 𝑟)=v𝑐·v𝑟
∥v𝑐∥∥v𝑟∥(5)
wherev𝑐andv𝑟are the corresponding vector embeddings.
4.2.2 Experimental Environment.The experiments were conducted on a server equipped with
an NVIDIA A100 GPU with 80GB of memory. The server operates under a Linux environment,
specifically, Linux version 5.15.0-97-generic, compiled using GCC (Ubuntu 11.4.0-1ubuntu1 22.04)
version 11.4.0 and GNU ld (GNU Binutils for Ubuntu) version 2.38.
The various models in our testbed were deployed as follows:DeepSeek-V3andQwen2.5were
accessed via the SiliconFlow API, while other generators, such asLlama-3.3-70BandPhi-4-14B, were
run locally using the Ollama framework. All models for the intermediate RAG stages—including
dense retrievers, rerankers (e.g., BAAI/bge-reranker-v2-m3), and compressors—were hosted and
executed within our local environment.
4.3 RQ1: Impact of theQuery Processing Stage
We investigate the impact of query refinement by evaluating the four strategies detailed in § 3.2.1:
Baseline(the original query),Simplified,Elaborated, and the oracle-augmentedHyDE. The
primary analysis, summarized in Table 4, focuses on performance when retrieving the top-3
documents (𝑘=3).
Table 4. Performance comparison of Query Processing strategies at 𝑘= 3. Parentheses denote relative
percentage changes vs.Baseline. Best results inbold. (CXG: CodeXGLUE, DB: DebugBench, O: Obfuscated).
Task Dataset (Metric) BaselineSimplified ElaboratedHyDE
DeepSeek-V3 GPT-4o DeepSeek-V3 GPT-4o
Code Gen. APPS (W-Pass@1)38.3627.91 (-27.2%) 25.81 (-32.7%) 33.72 (-12.1%) 22.04 (-42.5%) 31.45 (-18.0%)
Code Sum.CXG (Sim Emb) 87.75 87.78 (+0.03%) 87.50 (-0.28%)87.84 (+0.10%)87.61 (-0.16%) 87.71 (-0.05%)
CXG-O (Sim Emb) 74.50 80.21 (+7.66%) 79.83 (+7.15%)80.38 (+7.89%)80.11 (+7.53%) 75.05 (+0.74%)
Code RepairDB (W-Pass@1) 76.24 69.44 (-8.92%) 75.45 (-1.04%) 73.08 (-4.14%)77.35 (+1.46%)75.88 (-0.47%)
DB (CodeBLEU) 46.3256.65 (+22.3%)51.93 (+12.1%) 47.74 (+3.07%) 49.87 (+7.66%) 47.24 (+1.99%)
4.3.1 Cross-Task Performance Analysis.The effectiveness of query processing is highly task-
dependent, a phenomenon observed consistently across both transformation models.
For the APPS code generation task, which characterizeswell-defined tasksfeaturing explicit
problem statements, strict I/O examples, and clear constraints, the unalteredBaselinequery is
superior (W-Pass@138.36). All LLM-based transformations severely degrade performance, with
SimplifiedandElaboratedqueries causing drops up to 32.7% and 42.5% across both models. This
suggests for tasks with explicit requirements and robust examples, automated transformations may
strip away critical lexical details essential for exact-match retrieval.
Conversely, query transformation is highly beneficial forambiguous or noisy inputs, such as
obfuscated identifiers or informal, incomplete user queries. On the CodeXGLUE-OBF dataset, the
Elaboratedstrategy improvesSim Embby 7.89% withDeepSeek-V3and 7.53% withGPT-4o. Without
explicit semantic identifiers, elaboration forces the LLM to hypothesize the underlying structural
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:11
intent, generating useful contextual keywords for retrieval. However, on the standard CodeXGLUE
dataset, the same method offers a negligible 0.1% gain at most. This indicates that while query
transformation is directionally helpful for noisy code, its impact is far more constrained than in
natural language contexts [30].
DebugBench results highlight a critical trade-off. TheSimplifiedquery dramatically improves
theCodeBLEUscore (up to +22.3%), indicating better structural similarity to the reference. However,
this gain sacrifices functional correctness (W-Pass@1declines of 8.92% and 1.04%), suggesting that
simplification may prioritize superficial structure over function.
4.3.2 Findings of RQ1.Based on this analysis, we derive the following findings:
•Finding 1:The original, unaltered query is a powerful baseline, often outperforming auto-
mated transformations on tasks with clear, unambiguous inputs.
•Finding 2:Query elaboration is more effective for ambiguous or noisy inputs, where it can
provide crucial context and significantly improve retrieval relevance.
•Finding 3:Query simplification can trade functional correctness for structural similarity.
•Finding 4:All LLM-based query transformations risk the unintentional alteration of a query’s
core intent or the omission of critical details.
Answer to RQ1: The impact of the Query Processing stage is highly context-dependent. For well-
defined tasks, the original query is most robust. For ambiguous tasks, query elaboration can provide
significant benefits, but automated transformations should be applied cautiously due to the risk of
semantic drift, regardless of the LLM used.
4.4 RQ2: Comparative Analysis of theRetrieval Stage
In this section, we evaluate the core of the RAG pipeline by comparing the three retrieval paradigms
detailed in § 3.2.2:Sparse Retrieval,Dense Retrieval, andHybrid Retrieval. We benchmark
their performance against a non-RAGZero-Shotbaseline and anOraclesetup, representing the
theoretical upper bound. The detailed results are presented in Table 5.
4.4.1 Cross-Task Performance Analysis.The results unequivocally show that all retrieval paradigms
substantially outperform theZero-Shotbaseline, confirming the fundamental value of RAG for code
tasks. However, the optimal strategy is highly task-dependent.
The most striking result is the exceptional performance ofBM25, a classicSparse Retrieval
method. Unlike general-domain NLP tasks where dense retrievers typically dominate [ 14,25], SE
tasks often impose hard lexical constraints. On the APPS task,BM25’sW-Pass@1of 38.00 (+150.3%)
highlights that precise keyword matching of algorithms is crucial; dense models may suffer from
semantic drift, retrieving conceptually similar but syntactically incompatible code. This exact-
match advantage persists on code-based queries:BM25achieves the highestSim Embon standard
CodeXGLUE (87.90) and noisy CodeXGLUE-OBF (75.61), suggesting it robustly captures struc-
tural control-flow keywords even when semantic identifiers vanish. Finally, it leads in functional
correctness on DebugBench with aW-Pass@1of 76.53, likely by matching exact error traces.
Dense Retrieval, while powerful, exhibits more nuanced performance. Our findings reveal two
key insights. First, model scale is not a decisive factor; the smaller, 305M-parameterGTEmodel
consistently outperforms the largest 1.3B-parameterBGEmodel, particularly on APPS (34.45 vs.
25.82W-Pass@1). Second, code-specialization does not guarantee an advantage, as the code-trained
JinaandSFRmodels failed to surpass the best general-purpose models likeGTE.
Hybrid Retrievalproves to be a reliable but conservative strategy, delivering consistent middle-
of-the-pack results but failing to unlock any significant synergistic gains. Finally, the DebugBench
results highlight a critical trade-off: RAG methods consistently improve functional correctness at
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:12 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
Table 5. Comparative performance of Retrieval strategies at 𝑘=3. Parentheses denote relative percentage
changes compared to theZero-Shotbaseline. Best results for each metric are highlighted inbold.
Code Generation Code Summarization Code Repair
Retrieval ParadigmAPPS CodeXGLUE CodeXGLUE-OBF DebugBench
(W-Pass@1) (Sim Emb) (Sim Emb) (W-Pass@1) (CodeBLEU)
Baselines
Zero-Shot 15.18 87.38 75.24 64.03 66.26
Oracle 55.99 (+268.8%) 88.28 (+1.0%) 86.47 (+15.0%) 70.01 (+9.3%) 75.28 (+13.6%)
Sparse Retrieval
BM2538.00 (+150.3%) 87.90 (+0.6%) 75.61 (+0.5%) 76.53 (+19.5%)46.39 (-30.0%)
Dense Retrieval (General-purpose)
E5 (118M) 33.72 (+122.1%) 87.86 (+0.5%) 74.82 (-0.6%) 73.70 (+15.1%) 46.38 (-30.0%)
Granite (278M) 35.63 (+134.7%) 87.73 (+0.4%) 75.35 (+0.1%) 74.44 (+16.3%) 46.91 (-29.2%)
GTE (305M) 34.45 (+126.9%)87.90 (+0.6%)75.05 (-0.3%) 76.37 (+19.3%) 44.87 (-32.3%)
BGE (1.3B) 25.82 (+70.1%) 87.61 (+0.3%) 74.67 (-0.8%) 68.13 (+6.4%)63.57 (-4.1%)
Dense Retrieval (Code-specialized)
Jina (161M) 27.09 (+78.4%) 87.67 (+0.3%) 74.93 (-0.4%) 68.49 (+7.0%) 59.73 (-9.8%)
SFR (434M) 34.27 (+125.7%) 87.79 (+0.5%) 74.56 (-0.9%) 72.84 (+13.8%) 47.60 (-28.2%)
Hybrid Retrieval
Hybrid 33.54 (+120.9%) 87.68 (+0.3%) 75.22 (-0.0%) 75.04 (+17.2%) 47.37 (-28.5%)
the cost of structural similarity. For instance, whileBM25boostsW-Pass@1by +19.5%, it degrades
theCodeBLEUscore by 30.0%, suggesting that retrieving externally correct but different code
patterns steers the generator away from minimal, structurally-similar edits.
4.4.2 Findings of RQ2.
•Finding 5:RAG is fundamentally beneficial for code tasks, as all evaluated retrieval paradigms
significantly outperform the non-retrievalZero-Shotbaseline.
•Finding 6:The lexical-basedBM25is an exceptionally powerful and versatile retriever,
excelling across text-heavy, clean code, and noisy code queries via precise term matching.
•Finding 7:Model scale is not a decisive factor in dense retrieval performance. We found that
smaller, efficient models likeGTEconsistently surpassed the largest model,BGE.
•Finding 8:Domain specialization on code does not guarantee superior performance, as the
best general-purpose retrievers were unbeaten even on code-centric tasks.
•Finding 9:Hybrid Retrievalis a safe but conservative strategy that, in our experiments, did
not yield performance gains beyond the single best retriever for a given task.
Answer to RQ2: No single retrieval strategy is universally superior, as the optimal choice depends
on the task’s query modality. However, our findings strongly suggest that a dual-retriever approach,
leveraging the exceptional lexical precision of BM25 alongside the semantic understanding of a
robust general-purpose dense retriever like GTE, provides the most consistent and high-performing
foundation for a wide array of code-related tasks.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:13
4.5 RQ3: Contribution of theContext Refinement Stage
In this section, we evaluate the two primary techniques of the Context Refinement stage as defined
in § 3.2.3:Re-rankingandCompression. Our objective is to determine if these post-retrieval
steps improve the signal-to-noise ratio and enhance final task performance.
4.5.1 Re-ranking Performance Analysis.We analyzed the impact of re-ranking by applying two
state-of-the-art models to the outputs of three representative retrievers: the top sparse model
(BM25), an efficient dense model (GTE), and a weaker dense model (BGE). As shown in Table 6, our
findings indicate that re-ranking is an unstable and often detrimental step for code-related tasks.
Table 6. Performance of Reranker-Retriever combinations at 𝑘=3. Parentheses denote relative percentage
changes against the corresponding un-reranked baselines (Table 5). Best results are highlighted inbold.
Code Generation Code Summarization Code Repair
Reranker Base RetrieverAPPS CodeXGLUE CodeXGLUE-OBF DebugBench DebugBench
(W-Pass@1) (Sim Emb) (Sim Emb) (W-Pass@1) (CodeBLEU)
BEG-RerankerBM25 33.36 (-12.21%) 87.87 (-0.03%) 74.72 (-1.18%)76.76 (+0.30%)46.74 (+0.75%)
GTE 32.72 (-5.02%) 87.44 (-0.51%) 75.00 (-0.07%) 75.55 (-1.07%) 47.73 (+6.37%)
BGE 23.82 (-7.75%) 87.36 (-0.29%) 74.73 (+0.08%) 67.48 (-0.95%) 62.92 (-1.02%)
Qwen-RerankerBM25 34.72 (-8.63%)87.90 (0.00%)75.59 (-0.03%) 72.32 (-5.50%) 47.79 (+3.02%)
GTE 34.09 (-1.04%) 87.68 (-0.24%) 75.04 (-0.01%) 73.27 (-4.06%) 48.23 (+7.49%)
BGE 23.36 (-9.53%) 87.52 (-0.10%) 74.67 (0.00%) 68.74 (+0.90%) 62.91 (-1.04%)
For Code Generation, re-ranking consistently harms functional correctness.W-Pass@1scores
drop substantially in all configurations; the best retriever,BM25, declines by up to 12.21%. Notably,
re-ranking also fails to salvage weaker retrievals fromBGE, further degrading its score. For Code
Summarization, the impact is negligible, failing to justify the added computational cost.
DebugBench results are particularly unreliable. WhileBGE-Rerankeroffers a marginal +0.30%
W-Pass@1gain forBM25, this lone outlier contrasts with a 5.50% correctness drop when paired with
Qwen-Reranker. Although some configurations slightly improve structural similarity (CodeBLEU),
these minor gains cannot offset the significant, unpredictable degradation in functional correctness.
4.5.2 Compression Performance Analysis.We evaluate context compression by comparing two
distinct methodologies: the token-level pruning ofLLMLingua-2and variants of ourZero-Shot
Recomp Adaptationwhich use different cross-encoders (GTE,BGE, and the code-specializedSFR).
As illustrated in Figure 4, we benchmarked these against aZero-Shotfloor and an uncompressed
Optimal Retrieverceiling.
For APPS code generation, all compressors improve performance as the token budget increases.
The code-specializedSFRadaptation proves most effective, peaking around an 8000-token budget.
Conversely, compression is largely ineffective for Code Summarization. On standard CodeXGLUE,
all methods trail the uncompressed retriever; even on the obfuscated version, gains remain minor,
suggesting the task is too nuance-sensitive for current techniques.
DebugBench reveals a complex trade-off. For functional correctness (W-Pass@1), theSFRand
GTEadaptations show a remarkable upward trend, surpassing both theZero-Shotbaseline and the
uncompressedOptimal Retriever, suggesting compression effectively filters noise. However, the
CodeBLEUchart tells the opposite story: all curves trend downwards as token budgets increase.
This confirms a critical trade-off: compression achieves functional correctness by progressively
discarding original stylistic and structural context.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:14 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
10152025303540
2k 4k 8k 16kW-Pass@1(a) Code Gen. --APPS
86.88787.287.487.687.8
2k 4k 8k 16kSim_Emb(b-1) Code Sum. --CodeXGLUE
55657585
2k 4k 8k 16kW-Pass@1(c-1) Code Repair --DebugBench
0.350.450.550.650.75
2k 4K 8K 16KCodeBLUE(c-2) Code Repair --DebugBench
73.574.575.576.5
2k 4k 8k 16kSim_Emb(b-2) Code Sum. --CodeXGLUE -OBF
llmlingua2
bge
gte
sfr
Zero-Shot
Fig. 4. Performance of context compression methods across five evaluation scenarios. The charts illustrate
how performance (y-axis) changes with an increasing token budget (x-axis) for each compressor. The dashed
line represents theZero-Shotbaseline.
4.5.3 Findings of RQ3.Our analysis of the context refinement stage leads to the following findings:
•Finding 10:Re-ranking is an unreliable component that fails to justify its computational
overhead. It often degrades the performance of strong, keyword-based retrievals by disrupting
an already optimal ordering.
•Finding 11:The corrective potential of re-ranking is minimal. It fails to consistently or
substantially improve results from weaker initial retrievals, making it an ineffective tool for
salvaging performance.
•Finding 12:The utility of context compression is critically dependent on the token budget
and the choice of compressor. Code-specialized models with a sufficient budget are more
robust, while aggressive compression is consistently harmful.
Answer to RQ3: The Context Refinement stage is not a guaranteed improvement and should be
applied with caution. Our findings show thatRe-rankingis generally an inadvisable step due to
its high cost and unreliable, often negative, impact.Compressionreserved for models with strict
context length limitations. If used, a high token budget ( ≥8000) and a compatible compressor are
critical to mitigate performance loss. Ultimately, employing a generator with a native long-context
window is a superior strategy to relying on this stage.
4.6 RQ4: Influence of theGenerator Stage
In this section, we evaluate the impact of the generator (§ 3.2.4), analyzing how the capabilities of
different LLMs establish the performance ceiling of the RAG system, as illustrated in Figure 5.
4.6.1 Cross-Task Performance Analysis.Universally, all evaluated models benefit from retrieval,
with RAG-based scores at 𝑘=1,3,5consistently outperforming their respectiveZero-Shotbaselines.
Oraclecontext performance further confirms the profound impact of context quality on generation.
On the APPS code generation task,DeepSeek-V3is the clear frontrunner, alongside highly
competitive performance fromGPT-4o. Both frontier models demonstrate superior problem-solving
capabilities via distinctly higher performance curves (Figure 5(a)). For most models, performance
peaks at 𝑘= 3before plateauing or declining, suggesting that a small number of high-quality
documents is optimal while excess context introduces noise.
On the standard CodeXGLUE dataset, top models includingGPT-4o,Qwen2.5-C, andDeepSeek-V3
are tightly clustered near the ceiling. However, the challenging obfuscated dataset reveals clearer
model strengths. Here, the raw reasoning capacity ofDeepSeek-V3and the domain specialization
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:15
102030405060
K=1 K=3 K=5 Zero-Shot OracleW-Pass@1(a) Code Gen. --APPS Task
7075808590
K=1 K=3 K=5 Zero-Shot OracleSim_Emb(b-1) Code Sum. --CodeXGLUE
60708090
K=1 K=3 K=5 Zero-Shot OracleSim_Emnb(b-2) Code Sum. --CodeXGLUE -OBF
Phi4
Qwen2.5
Qwen2.5-C
Llama3.3
DeepSeek-V3
GPT-4o30507090
K=1 K=3 K=5 Zero-Shot OracleW-Pass@1(c-1) Code Repair --DebugBench
30507090
K=1 K=3 K=5 Zero-Shot OracleCodeBLEU(c-2) Code Repair --DebugBench
Fig. 5. Performance comparison of different generators across all tasks. RAG performance is shown for
𝑘=1,3,5retrieved documents, withZero-ShotandOracleresults as baselines.
ofQwen2.5-Cbecome apparent. This validates the distinct benefits of massive model scale and
code-specific training on structurally ambiguous inputs.
The DebugBench repair task highlights clear task-specific advantages. WhileDeepSeek-V3domi-
nated generation,Qwen2.5andGPT-4oprove highly effective for code repair, achieving the highest
functional correctness (W-Pass@1). Crucially,CodeBLEUresults (Figure 5(c-2)) confirm a recurring
finding: for most generators, RAG improves functional correctness at the cost of structural similarity
when compared to theZero-Shotbaseline.
4.6.2 Findings of RQ4.Our analysis of the generator stage leads to the following findings:
•Finding 13:A generator’s effectiveness depends on both its intrinsic reasoning capability
(standalone performance) and its ability to synthesize retrieved context.
•Finding 14:Model strengths are highly task-dependent. Frontier models likeDeepSeek-V3
andGPT-4oexcel at complex generative reasoning, while specialized models likeQwen2.5-C
are highly effective for structural code comprehension.
•Finding 15:More retrieved context is not always better. For complex tasks like code gen-
eration, performance often peaks with a small number of documents ( 𝑘=3), as additional
context can introduce noise.
•Finding 16:The trade-off between functional correctness and structural similarity in code
repair is a consistent phenomenon observed across most generators when using RAG.
Answer to RQ4: The generator is a crucial component whose capabilities establish the performance
ceiling for a RAG system. The optimal choice is task-dependent, requiring a balance between the
model’s intrinsic ability and its capacity to leverage context. Effective retrieval augmentation is
therefore not just about selecting a powerful generator, but also about providing it with the optimal
amount of high-quality information for the specific task at hand.
5 Discussion
5.1 Summary of Key Findings and Implications
Our empirical study on RAG for SE reveals several critical insights. Our findings point not to a
simple hierarchy of components, but to a complex interplay between task profiles and RAG pipeline
design. We synthesize our key findings and their implications below:
•The Primacy of the Retriever:Our results show that the initial retrieval stage is the most
critical determinant of pipeline performance. Strikingly, the classic lexical retriever,BM25,
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:16 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
proved exceptionally effective, frequently outperforming larger, dense models across diverse
query modalities (§ 4.4). Its success in matching specific algorithm names and error messages
underscores the enduring importance of lexical precision in code tasks (Finding 6). For
practitioners, this implies that prioritizing a robust and suitable retriever is often a more
impactful investment than focusing solely on the generator’s scale.
•The Situational Value of Advanced Components:Our findings caution against building
unnecessarily complex pipelines, as the value of advanced components is highly situational
and their indiscriminate use is often detrimental. While query elaboration benefited ambigu-
ous inputs (Finding 2), most transformations introduced semantic drift on well-defined tasks
(Finding 4). Similarly, re-ranking failed to correct weak retrievals and actively harmed the
ordering of strong ones (Finding 10). Practitioners should therefore view these components
not as default enhancements, but as specialized tools for targeted problems.
•No One-Size-Fits-All:A central conclusion is that no single RAG configuration is universally
optimal. The ideal component choice is coupled with the task’s nature, as the best retrieval
strategy (§ 4.4), query method (§ 4.3), and generator (§ 4.6) all varied significantly. For
instance,DeepSeek-V3excelled at code generation whileQwen2.5-Cis more effective for code
comprehension (Finding 14). This highlights a critical implication for researchers and tool
builders: the future of Code RAG lies in moving beyond static pipelines towards adaptive
frameworks that dynamically configure the optimal component chain.
5.2 Validation of the Adaptive Framework
As introduced in our approach (§ 3.3), we developed and open-sourced an adaptive RAG framework.
To validate the framework’s generalization beyond initial datasets, we conducted simulations
using problem descriptions from three external SE studies:Move Method Refactoring[ 5],Test
Generation[37], andLibrary Code Generation[54].
For each case, the framework autonomously profiled the task and recommended architectural
choices perfectly aligning with the original authors’ manual best practices (Table 7). Crucially, its
internal reasoning dynamically synthesizes our empirical findings across the entire RAG pipeline:
•Bridging Semantic Gaps:ForMove Method Refactoring[ 5] andLibrary Code Generation[ 54],
the framework identified semantic concepts as the key information type, overriding the default
lexical preference (Finding 6) to utilizeDense Retrieval. It further tailored the pipeline by
employing query elaboration (Finding 2) and a code-specialized generator (Finding 14) to
extract intent for low-clarity refactoring, while retaining the baseline query (Finding 4) to
preserve informal user intent in library generation.
•Preserving Structural Precision:Conversely, forTest Generation[ 37], the framework
detected a high-clarity reliance on explicit structural patterns. It retained the baseline query
to prevent semantic drift (Findings 1 & 4) and recommendedBM25for exact keyword
matching (Finding 6). By omitting re-ranking to protect optimal lexical ordering (Finding
10), the framework’s decisions mirrored the original authors’ observation that exact matching
avoids introducing irrelevant context noise.
This confirms our framework possesses robust generalization capabilities, synthesizing empirical
findings to autonomously architect theoretically sound pipelines for novel SE tasks.
5.3 Retrieval Dynamics: Depth and Source Distribution
5.3.1 The Impact of Retrieval Depth ( 𝑘).Our analysis of the number of retrieved documents ( 𝑘)
reveals a trade-off between signal and noise. As illustrated for the APPS task in Figure 6, the
relationship between 𝑘and performance is not linear. For most components, performance improves
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:17
Table 7. External validation of the adaptive framework across diverse SE tasks, detailing the generated profiles,
recommended pipelines, and their rationales mapped to our empirical findings (F denotes Finding).
Dimension Move Method Refactoring[ 5]Test Generation[37]Library Code Gen.[54]
1. Generated Task Profile
Modality[Code]-to-[Code] [Text+Code]-to-[Code] [Text]-to-[Code]
ClarityLow High Medium
ComplexityHigh Medium High
Info TypeSemantic Concepts Structural Patterns Semantic Concepts
MetricStructural Similarity Functional Correctness Functional Correctness
2. Recommended Pipeline
Query Proc.Elaborated Baseline Baseline
RetrievalDense BM25 Dense
RefinementNone None None
GeneratorCode-Specialized Frontier Generalist Frontier Generalist
3. Rationale & Empirical Alignment
Query Logic F2:Elaborates ambiguous code
smells to extract structural in-
tent.F1, F4:Retains baseline query
for high-clarity tasks to prevent
semantic drift.F4:Avoids rewriting to preserve
informal user intent and prevent
information loss.
Retrieve Logic F6:Overrides default lexical
search; uses dense embeddings
for conceptual bridging.F6:Relies on exact keyword
matching for precise method sig-
natures.F6:Employs dense retrieval to
resolve severe vocabulary mis-
matches.
Other Logic F10, F14:Omits re-ranking; se-
lects specialized generator for
deep code comprehension.F10, F15:Preserves lexical or-
dering (no re-ranking) and lim-
its context size to avoid noise.F10, F14:Omits refinement; al-
locates frontier model for com-
plex generative reasoning.
significantly from 𝑘=1to𝑘=3, highlighting the need for a sufficiently rich context. However,
performance typically plateaus or declines when moving to 𝑘= 5, as additional documents are
more likely to introduce distracting noise. This is especially pronounced for the top-performing
retriever,BM25.
252729313335373941
k=1 k=3 k=5W-Pass@1(a) Query Processing Stage
Simplified Elaborated HyDE Baseline2025303540
k=1 k=3 k=5W-Pass@1(b) Retrieval Stage
BM25 E5 Granite GTE
BGE Jina SFR Hybrid2025303540
K=1 K=3 K=5W-Pass@1(c) Context Refinement Stage (Reranking)
BEG-Reranker (BM25) BEG-Reranker (GTE)
BEG-Reranker (BGE) Qwen-Reranker (BM25)
Qwen-Reranker (GTE) Qwen-Reranker (BGE)
Fig. 6. Impact of the number of retrieved documents ( 𝑘) onW-Pass@1for different RAG stages on the APPS.
The optimal 𝑘is not universal. For instance, the condensedSimplifiedquery uniquely benefits up
to𝑘=5to gather sufficient information, whileRe-rankingrenders𝑘’s impact unpredictable. This
reinforces a key design principle for Code RAG:retrieval depth ( 𝑘) must be carefully optimized,
not blindly maximized, to balance essential signal against overwhelming noise.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:18 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
5.3.2 Retrieval Source Ablation.To validate our heterogeneous corpus design, we analyze the
retrieval source distribution at 𝑘= 3. Table 8 demonstrates that no single source universally
satisfies the diverse queries of code tasks.
Table 8. Retrieval source distribution at 𝑘=3. Percentages indicate the proportion of documents retrieved
from each corpus source across the evaluated tasks.
Task Dataset Python API Stack Overflow CodeSearchNet Code-Contests LeetCode
Code Gen. APPS 1.41% 2.81% 0.44%83.83%11.50%
Code Sum.CodeXGLUE 31.85% 24.00%39.00%4.77% 0.38%
CodeXGLUE-OBF 14.47% 29.15%43.41%10.10% 2.88%
Code Repair DebugBench 1.34% 2.69% 1.31% 14.03%80.63%
Task-specific dominance is quantitatively evident. Code generation retrieves 83.83% of its context
from Code-Contests, suggesting models may heavily rely on competitive programming boilerplate
to synthesize complex solutions. Conversely, code repair draws 80.63% from LeetCode, indicating
that logic debugging inherently demands precise algorithmic ground truth. Meanwhile, code
summarization exhibits a balanced triangulation: CodeSearchNet provides 39.00% of structural
context, while the Python API and Stack Overflow contribute 31.85% and 24.00% respectively to
supply functional definitions.
Code obfuscation induces a critical retrieval shift. When explicit identifiers vanish, Python API
retrievals drop substantially from 31.85% to 14.47%, forcing the system to compensate by extracting
syntactic patterns from CodeSearchNet and community discussions from Stack Overflow, which
correspondingly increase. These dynamic adaptations confirm that a heterogeneous corpus can
effectively handle varying query clarities, proving its construction is a functional necessity rather
than an arbitrary design choice.
5.4 Robustness of Findings: A Cross-Temporal Validation
To determine whether pre-training memorization on legacy benchmarks invalidates our empirical
findings, we conducted a cross-temporal trend analysis. We compared the legacy APPS dataset
against a zero-contamination validation set from LiveCodeBench (LCB) [ 22], comprising 100
problems published post-January 2025(with a balanced 1:1:1 difficulty ratio) to strictly bypass the
knowledge cutoff of our evaluated models.
Crucially, while absolute generation scores naturally degrade on the uncontaminated LCB dataset,
the relative ranking of RAG configurations remains remarkably invariant (Figure 7).BM25strictly
dominates both dense and hybrid retrievers across both temporal splits (Findings 6 & 9), proving
that the necessity for precise lexical matching in code generation is an intrinsic task property,
not a byproduct of retrieving memorized snippets. Heavy query transformations (e.g.,Elaborated,
HyDE) consistently underperformBaselineandSimplifiedinputs (Findings 1 & 4), confirming
that semantic drift in over-engineered queries penalizes novel problem-solving just as it does on
legacy data. Context refinement introduces universal regressions; models likeBGE-Rerankerstrictly
degrade baseline retrieval accuracy (Finding 10), demonstrating their persistent misalignment
with underlying algorithmic logic regardless of data recency. Finally, context compression strategies
display a converging upward trajectory with expanding token budgets (Finding 12), corroborating
their stable utility as noise filters.
Ultimately, this temporal validation yields a definitive conclusion:data leakage on legacy
benchmarks inflates absolute generation scores but does not confound the underlying RAG
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:19
dynamics. The comparative advantages of specific pipeline components are intrinsically tied to
the algorithmic nature of the tasks, verifying that our empirical insights generalize robustly to
contemporary, unseen SE challenges.
1020304050
Baseline Simpified Elaborated HyDE(a) Query Proccessing Stage
LCB (ds) LCB (gpt)
APPS (ds) APPS (gpt)020406080
bm25 gte e5 rag_hybrid(b) Retrieval Stage 
LCB APPS10203040
Baseline BEG-Reranker Qwen-Reranker(c) Context Re -ranking
LCB (bm25) LCB (gte)
APPS (bm25) APPS (gte)1520253035
2000 4000 8000 16000(d) Context Compression
LCB (llmlingua2) LCB (recomp-bge) LCB (zero-shot)
APPS (llmlingua2) APPS (recomp-bge) APPS (zero-shot)
Fig. 7. Cross-temporal validation of core RAG findings. The relative performance trends across (a) Query
Processing, (b) Retrieval, (c) Context Re-ranking, and (d) Context Compression remain highly consistent.
5.5 Cost and Feasibility Analysis
To assess real-world feasibility, we profile the computational and token overheads across RAG
components (Table 9) using our standardized infrastructure (§ 4.2.2).
Computational Latency.Resource demands vary drastically.BM25is computationally negligi-
ble (<10ms), whereas deep-learning bottlenecks are severe.Re-ranking(BGE-Reranker) consumes
815ms alongside massive VRAM ( ∼12.8GB), orders of magnitude slower than retrieval. This quan-
titatively reinforcesFinding 10, confirming that indiscriminate re-ranking renders pipelines
engineered-infeasible for latency-sensitive SE workflows.
Token Economics.Cost is driven by prompt inflation.Elaboratedqueries incur a heavy 987-
token overhead, roughly equivalent to retrieving 2-3 extra documents, explicitly justifying our
restriction to low-clarity tasks (Finding 2). Furthermore, tripling the input volume with 𝑘=3(e.g.,
1,760→5,781 in CodeXGLUE) leaves output length stable. Thus, Code RAG costs scale predictably
with context ingestion, avoiding erratic generation bloat.
Table 9. Cost and feasibility profiling.Left:Computational overhead per query (Ret.: Retrival, BGE-R: BGE-
Reranker).Right:Token consumption dynamics (Simp.: Simplified, Elab.: Elaborated, CXG: CodeXGLUE, DB:
DebugBench, “-”: non-applicability).
Panel A: Computational Overhead
Stage Method Device (V/RAM) Latency
Sparse Ret.BM25CPU (∼1.5G)<10ms
Dense Ret.E5 (118M)GPU (∼3.5G) 205ms
Re-rankingBGE-RGPU (∼12.8G) 815msPanel B: Average Token Consumption
Component Setting Input Tokens Output Tokens
Query ProcessionSimp./Elab.+85 / +987 -
Generator (𝑘=0→3) APPS 740→2,926 266→217
Generator (𝑘=0→3) CXG 1,760→5,781 111→109
Generator (𝑘=0→3) DB 845→2,518 234→187
5.6 Threats to Validity
External Validity:Our findings are based on the Python language and three specific tasks, which
may limit generalizability. We partially mitigate the risk of dataset-specific conclusions by evaluating
diverse query transformations in § 4.3. These transformations simulate the diverse input patterns
one might encounter across different datasets for the same task. Furthermore, the rapidly evolving
LLM landscape means our model selection is a snapshot in time; future models may exhibit different
sensitivities to the RAG components we studied.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:20 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
Construct Validity:Our metrics, while standard, have limitations.W-Pass@1evaluates only
functional correctness, ignoring readability and efficiency. Additionally, lowerCodeBLEUscores
in code repair don’t necessarily indicate RAG failure; rather, RAG often prioritizes functionally
correct external solutions over structurally minimal edits.
Internal Validity:We mitigated errors by leveraging well-established libraries and pre-trained
models. The primary source of potential internal risk lies in the custom code written to connect
these disparate components into a cohesive experimental pipeline. While this “glue code” was
carefully tested, the possibility of subtle, undetected bugs cannot be entirely eliminated.
6 Related Work
Retrieval-Augmented Generation for SE.The application of RAG to SE is a burgeoning field,
with research demonstrating its value across various tasks. Early works established that retrieving
relevant code snippets significantly improves code completion and summarization [ 34,38]. Subse-
quent research has explored more sophisticated retrieval sources beyond simple code snippets. For
instance, DocPrompting showcased the effectiveness of retrieving from API documentation to han-
dle libraries unseen during training [ 54], while others have proposed using structured knowledge,
such as representing a codebase as a knowledge graph, to enhance retrieval for repository-level
tasks [ 4]. More recent work has even begun to explore RAG for specialized domains like test
generation [ 37]. As categorized in recent surveys [ 14], these studies typically propose and validate a
single, novel RAG architecture or component. While pioneering new methods, they do not provide
a comparative analysis of the interchangeable components (e.g., different retrievers, re-rankers)
that constitute the pipeline. Our work differs fundamentally by providing the first systematic,
component-wise empirical study of the RAG design space for code.
Empirical Studies and Benchmarking of LLMs for Code.Our research also builds upon the
strong tradition of empirical evaluation in SE and AI. The advancement of LLMs for code has been
driven by standardized benchmarks that enable rigorous comparison, such as APPS for code gener-
ation [ 19], CodeXGLUE for a variety of code intelligence tasks [ 29], and HumanEval for evaluating
foundational model capabilities [ 7]. Within this context, several studies have conducted ablation
or comparative analyses on specific aspects of the code generation process, such as the impact
of different prompting techniques [ 49] or the effect of model fine-tuning versus prompting [ 44].
However, these empirical studies have largely focused on the LLM itself, treating it as a monolithic
component. To the best of our knowledge, no prior work has applied this rigorous, empirical
methodology to systematically dissect and evaluate the multi-stage RAG pipeline specifically for
code-related tasks. Our study fills this critical gap by providing a foundational empirical roadmap
that connects task features to optimal RAG pipeline configurations.
7 Conclusion
Our large-scale, component-wise study of RAG for software engineering leads to a clear directive:
practitioners shouldprioritize the retrieverand embracearchitectural parsimony. We found
that pipeline performance is dominated by the retrieval stage, where classic lexical models likeBM25
often outperform larger dense counterparts. Advanced components like query transformers and
re-rankers, in contrast, provide only situational benefits and can even be detrimental. Furthermore,
our results underscore the highly task-dependent nature of the entire pipeline—from retriever to
generator—challenging the efficacy of static, one-size-fits-all systems. This work therefore strongly
advocates for a paradigm shift towards theadaptive, task-aware frameworkswe prototype.
Ultimately, our study provides a foundational empirical roadmap for engineering RAG systems for
code in a more principled, data-driven manner.
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:21
Data Availability
The artifact of this paper, containing all experimental data, source code, and documentation neces-
sary to reproduce our component-wise empirical study of RAG systems for software engineering
tasks, can be accessed via https://github.com/security-pride/RAG-Empirical-SE.
References
[1]Marah Abdin, Jyoti Aneja, Hany Awadalla, et al .2024. Phi-3 Technical Report: A Highly Capable Language Model
Locally on Your Phone. arXiv:2404.14219 [cs.CL]
[2]Uri Alon, Meital Zilberstein, Omer Levy, and Eran Yahav. 2019. code2vec: Learning Distributed Representations of
Code.Proc. ACM Program. Lang.3, POPL (2019), 40:1–40:29. doi:10.1145/3290353
[3]Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. 2024. A Survey on RAG with LLMs.
Procedia Computer Science246 (2024), 3781–3790. doi:10.1016/j.procs.2024.09.178
[4]Mihir Athale and Vishal Vaddina. 2025. Knowledge Graph Based Repository-Level Code Generation. In2025 IEEE/ACM
International Workshop on Large Language Models for Code (LLM4Code). IEEE, Piscataway, NJ, USA, 169–176. doi:10.
1109/llm4code66737.2025.00026
[5]Abhiram Bellur, Fraol Batole, Mohammed Raihan Ullah, Malinda Dilhara, Yaroslav Zharov, Timofey Bryksin, Kai
Ishikawa, Haifeng Chen, Masaharu Morimoto, Takeo Hosomi, Tien N. Nguyen, Hridesh Rajan, Nikolaos Tsantalis, and
Danny Dig. 2025. Together We are Better: LLM, IDE and Semantic Embedding to Assist Move Method Refactoring. In
Proceedings of the 41st IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE, Piscataway,
NJ, USA, 1–13. doi:10.1109/ICSME64153.2025.00046
[6]Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. BGE M3-Embedding: Multi-Lingual,
Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]
[7]Mark Chen, Jerry Tworek, Heewoo Jun, et al .2021. Evaluating Large Language Models Trained on Code.
arXiv:2107.03374 [cs.LG]
[8]Alexis Conneau, Kartikay Khandelwal, Naman Goyal, et al .2020. Unsupervised Cross-lingual Representation Learning
at Scale. arXiv:1911.02116 [cs.CL]
[9]Gordon V. Cormack, Charles L. A. Clarke, and Stefan Bottcher. 2009. Reciprocal rank fusion outperforms condorcet
and individual rank learning methods. InProceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval, James Allan, Javed A. Aslam, Mark Sanderson, ChengXiang Zhai, and Justin
Zobel (Eds.). ACM, New York, NY, USA, 758–759. doi:10.1145/1571941.1572114
[10] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, et al. 2025. DeepSeek-V3 Technical Report. arXiv:2412.19437 [cs.CL]
[11] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A
Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models. InProceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining. ACM, Barcelona Spain, 6491–6501. doi:10.1145/3637528.
3671470
[12] Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, et al .2020. Codebert: A pre-trained model for programming
and natural languages. InFindings of the Association for Computational Linguistics: EMNLP 2020. Association for
Computational Linguistics, Stroudsburg, PA, USA, 1536–1547.
[13] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot Dense Retrieval without Relevance
Labels. InProceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki (Eds.).
Association for Computational Linguistics, Stroudsburg, PA, USA, 1762–1777. doi:10.18653/V1/2023.ACL-LONG.99
[14] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen
Wang. 2024. Retrieval-Augmented Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
[15] Luca Gazzola, Daniela Micucci, and Leonardo Mariani. 2019. Automatic Software Repair: A Survey.IEEE Transactions
on Software Engineering45, 1 (2019), 34–67. doi:10.1109/TSE.2017.2755013
[16] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al .2024. The Llama 3 Herd of Models. arXiv:2407.21783 [cs.AI]
[17] Wenchao Gu, Juntao Chen, Yanlin Wang, Tianyue Jiang, Xingzhe Li, Mingwei Liu, Xilin Liu, Yuchi Ma, and Zibin
Zheng. 2025. What to Retrieve for Effective Retrieval-Augmented Code Generation? An Empirical Study and Beyond.
arXiv:2503.20589 [cs.SE]
[18] Michael Günther, Jackmin Ong, Isabelle Mohr, Alaeddine Abdessalem, Tanguy Abel, Mohammad Kalim Akram,
Susana Guzman, Georgios Mastrapas, Saba Sturua, Bo Wang, Maximilian Werk, Nan Wang, and Han Xiao. 2024. Jina
Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents. arXiv:2310.19923 [cs.CL]
[19] Dan Hendrycks, Steven Basart, Saurav Kadavath, et al .2021. Measuring Coding Challenge Competence With APPS.
arXiv:2105.09938 [cs.SE]
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

FSE183:22 Q. Ke, Y. Zhao, H. Leng, S. Zhao, and H. Wang
[20] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt.
2021. Measuring Mathematical Problem Solving With the MATH Dataset. arXiv:2103.03874 [cs.LG]
[21] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with Generative Models for Open Domain
Question Answering. arXiv:2007.01282 [cs.CL]
[22] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik
Sen, and Ion Stoica. 2025. LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for
Code. InThe Thirteenth International Conference on Learning Representations, ICLR 2025. OpenReview.net, Amherst,
MA, USA, 1–15.
[23] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. LongLLM-
Lingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression. InProceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024,
Bangkok, Thailand August 11-16, 2024. Association for Computational Linguistics, Stroudsburg, PA, USA, 1658–1677.
doi:10.18653/v1/2024.acl-long.91
[24] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-Scale Similarity Search with GPUs.IEEE Transactions on
Big Data7, 3 (2021), 535–547. doi:10.1109/TBDATA.2019.2921572
[25] Vladimir Karpukhin, Barlas Oguz, Sewon Min, et al .2020. Dense Passage Retrieval for Open-Domain Question
Answering. arXiv:2004.04906 [cs.CL]
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler,
Mike Lewis, Wen tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. arXiv:2005.11401 [cs.CL]
[27] Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards General Text
Embeddings with Multi-stage Contrastive Learning. arXiv:2308.03281 [cs.CL]
[28] Ye Liu, Rui Meng, Shafiq Jot, Silvio Savarese, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. 2025. CodeXEmbed: A
Generalist Embedding Model Family for Multiligual and Multi-task Code Retrieval. arXiv:2411.12644 [cs.CL]
[29] Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, et al .2021. CodeXGLUE: A Machine Learning Benchmark Dataset for
Code Understanding and Generation. arXiv:2102.04664 [cs.SE]
[30] Xinbei Ma, Yeyun Gong, Pengcheng He, Hai Zhao, and Nan Duan. 2023. Query Rewriting for Retrieval-Augmented
Large Language Models. arXiv:2305.14283 [cs.CL]
[31] Mayank Mishra, Matt Stallone, Gaoyuan Zhang, et al .2024. Granite Code Models: A Family of Open Foundation
Models for Code Intelligence. arXiv:2405.04324 [cs.AI]
[32] Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage Re-ranking with BERT. arXiv:1901.04085 [cs.IR]
[33] OpenAI. 2024. GPT-4o System Card. arXiv:2410.21276 [cs.CL]
[34] Md Rizwan Parvez, Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021. Retrieval
Augmented Code Generation and Summarization. arXiv:2108.11601 [cs.SE]
[35] Shuo Ren, Daya Guo, Shuai Lu, et al .2020. CodeBLEU: a Method for Automatic Evaluation of Code Synthesis.
arXiv:2009.10297 [cs.SE]
[36] S. E. Robertson and S. Walker. 1994. Some simple effective approximations to the 2-Poisson model for probabilistic
weighted retrieval. InSIGIR ’94: Proceedings of the 17th annual international ACM SIGIR conference on Research and
development in information retrieval. Springer-Verlag, London, UK, 232–241.
[37] Jiho Shin, Reem Aleithan, Hadi Hemmati, and Song Wang. 2024. Retrieval-Augmented Test Generation: How Far Are
We? arXiv:2409.12682 [cs.SE]
[38] Lu Shuai, Duan Nan, Han Hojae, Guo Daya, Hwang Seung-won, and Svyatkovskiy Alexey. 2022. ReACC: A Retrieval-
Augmented Code Completion Framework. InProceedings of the 60th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Dublin, Ireland, 6227–6240. doi:10.
18653/v1/2022.acl-long.431
[39] Jonathan Sillito, Frank Maurer, Seyed Mehdi Nasehi, and Chris Burns. 2012. What makes a good code example? A
study of programming Q and A in StackOverflow. InProceedings of the 2012 IEEE International Conference on Software
Maintenance (ICSM). IEEE, Piscataway, NJ, USA, 25–34. doi:10.1109/ICSM.2012.6405249
[40] Karen Spärck Jones. 1972. A statistical interpretation of term specificity and its application in retrieval.Journal of
Documentation28, 1 (1972), 11–21.
[41] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, et al .2024. Is ChatGPT Good at Search?
Investigating Large Language Models as Re-Ranking Agents. InProceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, Houda Bouamor, Juan Pino, and Kalika Bali
(Eds.). Association for Computational Linguistics, Stroudsburg, PA, USA, 14918–14937. doi:10.18653/V1/2023.EMNLP-
MAIN.923
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.

Not All RAGs Are Created Equal: A Component-Wise Empirical Study for Software Engineering Tasks FSE183:23
[42] Runchu Tian, Yining Ye, Yujia Qin, Xin Cong, Yankai Lin, Yinxu Pan, Yesai Wu, Haotian Hui, Weichuan Liu,
Zhiyuan Liu, and Maosong Sun. 2024. DebugBench: Evaluating Debugging Capability of Large Language Mod-
els. arXiv:2401.04621 [cs.SE]
[43] Christoph Treude, Ohad Barzilay, and Margaret-Anne Storey. 2011. How do programmers ask and answer questions
on the web? (NIER track). InProceedings of the 33rd International Conference on Software Engineering. ACM, New York,
NY, USA, 804–807. doi:10.1145/1985793.1985907
[44] Chaozheng Wang, Yuanhang Yang, Cuiyun Gao, Yun Peng, Hongyu Zhang, and Michael R. Lyu. 2022. No more
fine-tuning? an experimental evaluation of prompt tuning in code intelligence. InProceedings of the 30th ACM Joint
European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE 22).
ACM, New York, NY, USA, 382–394. doi:10.1145/3540250.3549113
[45] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, et al .2024. Text Embeddings by Weakly-Supervised Contrastive
Pre-training. arXiv:2212.03533 [cs.CL]
[46] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Improving Text
Embeddings with Large Language Models. arXiv:2401.00368 [cs.CL]
[47] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang,
Shizheng Li, Qi Qian, Ruicheng Yin, Changze Lv, Xiaoqing Zheng, and Xuanjing Huang. 2024. Searching for Best
Practices in Retrieval-Augmented Generation. arXiv:2407.01219 [cs.CL]
[48] Yuan Wang, Xuyang Wu, Hsin-Tai Wu, Zhiqiang Tao, and Yi Fang. 2024. Do Large Language Models Rank Fairly? An
Empirical Study on the Fairness of LLMs as Rankers. arXiv:2404.03192 [cs.IR]
[49] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. 2022.
Chain-of-thought prompting elicits reasoning in large language models. InAdvances in Neural Information Processing
Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates, Inc.,
Red Hook, NY, USA, 24824–24837.
[50] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. RECOMP: Improving Retrieval-Augmented LMs with Compression
and Selective Augmentation. arXiv:2310.04408 [cs.CL]
[51] An Yang, Baosong Yang, Beichen Zhang, et al. 2025. Qwen2.5 Technical Report. arXiv:2412.15115 [cs.CL]
[52] Zezhou Yang, Sirong Chen, Cuiyun Gao, Zhenhao Li, Xing Hu, Kui Liu, and Xin Xia. 2025. An Empirical Study of
Retrieval-Augmented Code Generation: Challenges and Opportunities.ACM Trans. Softw. Eng. Methodol.34, 7 (2025),
188:1–188:28. doi:10.1145/3717061
[53] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng
Liu, Junyang Lin, Fei Huang, and Jingren Zhou. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking
Through Foundation Models. arXiv:2506.05176 [cs.CL]
[54] Shuyan Zhou, Uri Alon, Frank F. Xu, Zhiruo Wang, Zhengbao Jiang, and Graham Neubig. 2023. DocPrompting:
Generating Code by Retrieving the Docs. arXiv:2207.05987 [cs.SE]
[55] Xiaoling Zhou, Ou Wu, Weiyao Zhu, and Ziyang Liang. 2022. Understanding Difficulty-Based Sample Weighting
with a Universal Difficulty Measure. InMachine Learning and Knowledge Discovery in Databases: European Conference,
ECML PKDD 2022, Grenoble, France, September 19–23, 2022, Proceedings, Part III. Springer-Verlag, Cham, Switzerland,
68–84. doi:10.1007/978-3-031-26409-2_5
Received 2026-02-25; accepted 2026-03-24
Proc. ACM Softw. Eng., Vol. 3, No. FSE, Article FSE183. Publication date: July 2026.