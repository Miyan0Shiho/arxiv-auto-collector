# Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation

**Authors**: Zoe Fingleton, Nazanin Siavash, Armin Moin

**Published**: 2026-04-16 17:41:49

**PDF URL**: [https://arxiv.org/pdf/2604.15270v1](https://arxiv.org/pdf/2604.15270v1)

## Abstract
In this paper, we focus on automating two of the widely used Verification and Validation (V&V) activities in the Software Development Lifecycle (SDLC): Software testing and software inspection (also known as review). Concerning the former, we concentrate on automated test case generation using Large Language Models (LLMs). For the latter, we enable inspection of the source code by LLMs. To address the known LLM hallucination problem, in which LLMs confidently produce incorrect outputs, we implement a Retrieval Augmented Generation (RAG) pipeline to integrate supplementary knowledge sources and provide additional context to the LLM. Our experimental results indicate that incorporating external context via the RAG pipeline has a generally positive impact on both test case generation and code inspection. This novel approach reduces the total project cost by saving human testers'/inspectors' time. It also improves the effectiveness and efficiency of these V&V activities, as evidenced by our experimental study.

## Full Text


<!-- PDF content starts -->

Enhancing Large Language Models with Retrieval
Augmented Generation for Software Testing and Inspection
Automation
ZOE FINGLETON,Department of Computer Science
University of Colorado Colorado Springs (UCCS), USA and
Smith College, Northampton, MA, USA
NAZANIN SIAVASH,Department of Computer Science
University of Colorado Colorado Springs (UCCS), USA
ARMIN MOIN,Department of Computer Science
University of Colorado Colorado Springs (UCCS), USA
Abstract:In this paper, we focus on automating two of the widely used Verification and Validation (V&V)
activities in the Software Development Lifecycle (SDLC): Software testing and software inspection (also
known as review). Concerning the former, we concentrate on automated test case generation using Large
Language Models (LLMs). For the latter, we enable inspection of the source code by LLMs. To address the
known LLMhallucinationproblem, in which LLMs confidently produce incorrect outputs, we implement a
Retrieval Augmented Generation (RAG) pipeline to integrate supplementary knowledge sources and provide
additional context to the LLM. Our experimental results indicate that incorporating external context via
the RAG pipeline has a generally positive impact on both test case generation and code inspection. This
novel approach reduces the total project cost by saving human testers’/inspectors’ time. It also improves the
effectiveness and efficiency of these V&V activities, as evidenced by our experimental study.
CCS Concepts:•Software and its engineering →Software verification and validation;•Computing
methodologies→Artificial intelligence.
Additional Key Words and Phrases: large language models, software testing, code inspection, code review,
ai4se, verification and validation
ACM Reference Format:
Zoe Fingleton, Nazanin Siavash, and Armin Moin. 2026. Enhancing Large Language Models with Retrieval
Augmented Generation for Software Testing and Inspection Automation. 1, 1 (April 2026), 18 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
The Software Development Life-Cycle (SDLC), also known as the software process, can look different
across various companies and teams. However, it often comprises four fundamental stages: (i)
Software Requirements Engineering (RE), (ii) Software development (which includes both software
Authors’ Contact Information: Zoe Fingleton, zfingleton@smith.edu, Department of Computer Science
University of Colorado Colorado Springs (UCCS), USA and
Smith College, Northampton, MA, USA; Nazanin Siavash, nsiavash@uccs.edu, Department of Computer Science
University of Colorado Colorado Springs (UCCS), USA; Armin Moin, amoin@uccs.edu, Department of Computer Science
University of Colorado Colorado Springs (UCCS), USA.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the
full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM XXXX-XXXX/2026/4-ART
https://doi.org/XXXXXXX.XXXXXXX
, Vol. 1, No. 1, Article . Publication date: April 2026.arXiv:2604.15270v1  [cs.SE]  16 Apr 2026

2 Fingleton et al.
design and implementation/programming), (iii) Software Verification and Validation (V&V), and
(iv) Software deployment, evolution, and maintenance. This work is concerned with V&V, which is
a crucial stage throughout the software lifecycle. Software testing, software inspection (review),
and formal verification are popular examples of V&V activities. Formal verification is not always
possible. Moreover, it is very costly. Thus, it is typically pursued for safety-critical or mission-critical
systems. However, it is the only V&V activity that can prove the absence of errors to the extent
that it is verified with mathematical proofs.
On the other extreme of the spectrum, software inspection (also known as review) resides. This
is the most cost-effective V&V activity. On average, it can find around 60% of software defects
and inefficiencies in the design and implementation, which is impressive [ 1]. However, manual
inspection is tedious and time-consuming. Large Language Models (LLMs) have ‘observed’ many
examples of software source code during their training. Therefore, they are capable of taking over
the code inspection task for widely used programming languages, such as Python.
Another well-established V&V activity is Testing. In terms of the cost, testing lies somewhere
between the two extremes of formal verification and inspection. Similar to inspection, and unlike
formal verification, testing can only show the presence of errors, not their absence. However,
software testing is essential to ensure fidelity and adherence to the functional and non-functional
requirements of software systems as specified in the RE stage of SDLC. Functional requirements
are related to features and functionalities of the software system and the services that it should
provide. However, non-functional requirements (also known as quality attributes) are concerned
with the constraints on the software and its operation, as well as horizontal qualities of the system,
such as security, privacy-preserving, reliability, adaptability, acceptability, maintainability, and
sustainability.
In this work, we are interested in automated code inspection and test case generation using
Artificial Intelligence (AI). Specifically, we are interested in the subdiscipline of Machine Learning,
and in particular, state-of-the-art pre-trained Machine Learning models, called Large Language
Models (LLMs). LLMs are fundamentally transforming the landscape of software engineering [ 2].
Over the past few years, researchers and practitioners have increasingly leveraged LLMs to support
or automate a broad spectrum of software engineering activities, including automated test case
generation, input data synthesis, test oracle construction, bug detection, debugging, and program
repair. Among these, software testing has emerged as one of the most active and impactful areas
of application. Given the ever-growing complexity of modern systems and the scale of testing
required, organizations face significant pressure to automate testing workflows. In this context,
the generative and ‘reasoning’ abilities of LLMs offer a powerful solution, enabling more efficient
handling of labor-intensive tasks.
Despite these advances, ensuring consistency and reliability in large-scale testing projects
remains a considerable challenge. Software systems continuously evolve, and maintaining alignment
between source code, documentation, and test artifacts is both time-consuming and error-prone
[3–5]. Discrepancies between code and descriptive artifacts, such as comments or test specifications,
not only reduce comprehensibility but may also introduce confusion, propagate errors, and degrade
developer productivity. To mitigate such risks, the research community has proposed a variety of
automated techniques over the past decade [ 5]. However, many of these methods struggle with the
inherently informal and inconsistent nature of human-written documentation, which often includes
domain-specific jargon, acronyms, inconsistent grammar, incomplete sentences, and typographical
errors.
The emergence of advanced LLM architectures, exemplified by the Generative Pre-trained
Transformer (GPT) series and similar models, has introduced new opportunities to overcome these
limitations. Unlike traditional approaches that rely on rigid pattern matching or shallow Natural
, Vol. 1, No. 1, Article . Publication date: April 2026.

Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation 3
Language Processing (NLP) techniques, LLMs can analyze software repositories in a context-aware
manner, capturing semantic nuances and subtle dependencies across code and documentation [ 5,6].
This capability enables more robust detection of inconsistencies and provides richer support for
developers in maintaining coherent software artifacts throughout the lifecycle.
Rather than relying solely on manual, rule-based generation, or search-based generation strate-
gies for test cases, which often require significant manual configuration, LLMs can dynamically
synthesize test cases by reasoning about the program intent, requirements, and contextual cues
extracted from code and natural language documentation. This shift represents a move toward
intelligent, knowledge-driven automation, where models are not only capable of generating test
cases but also refining them iteratively to improve coverage, correctness, and maintainability. By
integrating LLMs into the testing pipeline, software engineering is moving closer to a future in
which testing is not just automated but also adaptive, context-sensitive, and capable of evolving
alongside the software itself.
To make automated code analysis and test case generation more context-aware while also
improving the reliability of LLMs and reducing hallucinations, we employ a Retrieval-Augmented
Generation (RAG) approach. The retrieval stage is designed to interpret the input prompt and
gather supporting information from external knowledge bases [ 7,8]. Such techniques have already
proven effective in domains including content generation, question answering systems, chatbots,
and conversational agents.
In NLP tasks, RAG provides additional guidance by integrating external evidence into the
generation process, effectively serving as contextual reinforcement for both the input and the
produced output [ 9]. Typically, the system first deploys a retriever to locate and extract the most
relevant documents from the underlying database. These retrieved items are then fused with the
user’s original query, creating an enriched context that steers the model toward more accurate and
coherent responses [10].
To quantify the hallucination problem in LLM-based code analysis, we estimate hallucination
metrics through overall accuracy. In other words, the proportion of correct outputs produced by
the model serves as an indicator of how frequently the model avoids fabricating or generating
misleading information. For this purpose, we experiment with multiple LLMs, each of which can be
applied in different stages of the V&V activities. Employing different LLMs allows us to capture the
variation in their capabilities and ensures that our evaluation is not biased toward the strengths or
weaknesses of a single model.
To rigorously validate and evaluate our proposed approach for automated code inspection, we
compare the performance of the models against a human baseline. Specifically, we consider an
average detection rate of human code inspectors, which has been reported to be approximately
60%. By using this reference point, we are able to assess not only whether the LLMs can automate
parts of the inspection process, but also whether they can reach or surpass the effectiveness of
human experts in detecting errors and defects.
In addition, to validate and evaluate the proposed framework for automated test case generation,
we adopt a reference benchmark dataset introduced by Wang et al. [ 11]. The TestEval dataset has
been specifically designed to measure the quality of automatically generated test cases across a
wide range of tasks, making it a reliable choice for this study. The dataset enables us to assess the
coverage of the generated test cases in a controlled and consistent setting. Moreover, by using
this dataset, we can ensure that the evaluation is both reproducible and comparable with other
state-of-the-art techniques. The evaluations allow us to demonstrate not only the feasibility but
also the relative advantages and limitations of applying LLMs in automated code inspection and
test case generation.
, Vol. 1, No. 1, Article . Publication date: April 2026.

4 Fingleton et al.
The contribution of this paper is twofold: (i) We propose a novel approach to automated code
inspection based on LLMs and RAGs. (ii) We propose a novel approach to automated test case
generation using LLMs and RAGs. Therefore, we answer the following Research Questions (RQs)
in our experimental study: RQ1. Can LLMs w/o RAG pipelines perform code inspection effectively
and efficiently? RQ2. Can LLMs w/o RAG pipelines perform test case generation effectively and
efficiently?
The rest of this paper is structured as follows: Section 2 provides some background information on
LLMs and RAG. Afterward, Section 3 reviews the literature of automated V&V. Further, we propose
our novel approach in Section 4. The experimental study in Section 5 validates and evaluates the
proposed approach. Potential threats to validity are discussed in Section 6. Finally, we conclude
and suggest future work in Section 7.
2 Background
2.1 Large Language Models (LLMs)
LLMs are a significant advancement in AI, trained on vast corpora containing billions of words.
These models are upscaled iterations of language models (LMs), which learn intricate patterns
of natural language from a dataset, enabling them to predict and generate contextually relevant
information [ 12]. The driving force behind many LLMs is the self-attention mechanism found in the
Transformer architecture [ 13], which enables understanding of context. Through a process known as
pre-training, LLMs learn some general knowledge from a dataset. Often, this is accomplished using
the aforementioned self-attention. Furthermore, beyond this ability to learn a general language
understanding from initial training, LLMs can befine-tunedfor a specific task. Fine-tuning redirects
the model to focus on a specific field for enhanced domain-specific performance [ 14]. Fine-tuning
gained prominence with BERT’s introduction in 2019 [ 15,16], which integrated pre-training and
fine-tuning approaches. Since then, these techniques have been widely adopted in industry [ 17]
and scaled [18].
2.2 LLM Hallucination
As the performance of LLMs in NLP tasks has rapidly progressed, the generation of non-factual
responses has become a pressing concern. Known ashallucination, LLMs have a tendency to produce
content that is either unverifiable or contradictory to their knowledge base [ 19]. This often occurs
when LLMs lack sufficient information to answer a user’s query.
We can generally group hallucinations into two categories: factuality hallucinations and faithful-
ness hallucinations. Factuality hallucination occurs when an LLM generates a result that is factually
incorrect within the broader scheme of known knowledge. In contrast, faithfulness hallucination
directly contradicts an LLM’s learned knowledge or user instructions [19].
2.3 Retrieval Augmented Generation (RAG)
Introducing a context-specific knowledge source to the LLM’s accessible data is one approach to
mitigating the hallucination issue. The presence of this supplementary information reduces the
likelihood that the LLM will lack context [ 20]. To access knowledge base data, documents in the
RAG knowledge base are first encoded into embeddings. The general methodology for performing
these embeddings involves using a sentence transformer model or an encoder-based architecture
to find sentence embeddings through contrastive learning [19].
With these documents in vector storage, the pipeline can perform a similarity search for a
specific query and return relevant context. Using this methodology, an LLM can receive additional
context-specific knowledge pertinent to a specific prompt.
, Vol. 1, No. 1, Article . Publication date: April 2026.

Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation 5
Existing results, particularly in the domain of NLP tasks, show that RAG-enhanced models
can outperform the state-of-the-art in several language generation tasks [ 21,22]. Subsequently,
RAG pipelines have been integrated into domain-specific contexts, such as code generation and
summarization [ 23], mathematics [ 24], and medicine [ 25], with similarly successful results [ 26]. In
this work, we propose deploying RAG in the V&V activities, specifically for test case generation
and source code review.
3 Related Work
3.1 Code Inspection
Code inspection is a crucial activity within the V&V process; however, traditional approaches
relying on manual review, where one or more developers examine their peers’ code [ 27,28], are
both time-consuming and labor-intensive. This limitation has motivated research into automation
techniques, particularly with the rise of LLMs. In this area, a recent systematic survey of code review
tasks analyzed 691 publications and identified 24 key studies published between 2015 and 2024,
providing a structured overview of models, metrics, baselines, and results in automated code review
[29]. Recent advances in state-of-the-art LLMs demonstrate strong performance in automated
code inspection tasks [ 30,31], especially when models are fine-tuned for domain-specific software
engineering applications [ 32]. Additionally, Cihan et al. [ 33] highlighted the gap in longitudinal
studies by examining the impact of automated code review tools in software development processes.
Their research extends the scope by investigating an LLM-based automated code review tool
deployed in real-world industry settings, with particular attention to its influence on development
artifacts and developer perceptions. Another line of research has explored the role of LLMs in
processing code review feedback. Studies show that models can effectively interpret inspection
comments and generate appropriate revisions [ 34]. In parallel, automated comment generation
has gained traction, where Machine Learning and NLP methods are leveraged to analyze code
and produce defect-focused annotations or fix suggestions [ 35,36]. These approaches improve
review efficiency, but remain limited in their ability to express uncertainty or identify gaps in
understanding [ 37]. In another study, Sun et al. [ 38] provided the first systematic investigation of
the adoption and usage of AI-based code review actions on GitHub, categorizing available tools by
their review granularity (pull request, file, or hunk level) as well as by their behavior, such as the
specific triggers that initiate automated actions. Building on this, an LLM-assisted framework has
been introduced for evaluating the actionability of code review comments and verifying whether
they have been addressed, achieving accuracy levels comparable to human annotations. A further
challenge arises when dealing with obfuscated or intentionally complex code, where LLMs struggle
to provide meaningful analyses and often exhibit degraded performance [ 39]. These limitations
underscore the importance of continued research into integrating LLM-based inspection tools
within practical software engineering workflows, balancing their demonstrated strengths with
recognition of their current shortcomings.
3.2 Test Case Generation
Beyond applications for code inspection, LLMs show potential for adaptability in other areas of
software development. One such area is computational test case generation, a well-established
field [ 40] that has recently begun to incorporate LLMs to facilitate complete automation. Recent
research into automated unit test generation and evaluation suggests that LLMs have surpassed the
current state-of-the-art in some forms of test case generation, such as JavaScript test generation
programs [ 41]. Furthermore, LLMs demonstrate potential for automating test case generation from
bug reports [ 42]. For instance, Chen et al. [ 43] introduced ChatUniTest, a unit test generation
, Vol. 1, No. 1, Article . Publication date: April 2026.

6 Fingleton et al.
framework powered by LLMs. The framework leveraged novel mechanisms such as adaptive
focal context and a generation–validation–repair pipeline, culminating in the development of the
ChatUniTest Core. This system was designed to support both researchers and tool builders by
improving the reliability and adaptability of automated unit test generation. Complementing this
direction, Li et al. [ 44] proposed TestChain, a multi-agent framework that decouples the generation
of test inputs from test outputs. TestChain adopts a ReAct-style conversational chain, enabling LLMs
to interact directly with a Python interpreter to produce more accurate and executable outputs.
Experimental results demonstrated its effectiveness, with TestChain significantly outperforming
baseline approaches. Notably, when using GPT-4 as the backbone model, TestChain achieved a
13.84% improvement in test case accuracy over the baseline on the LeetCode-hard dataset.
Nevertheless, LLM performance remains inconsistent across datasets and code contexts, with a
substantial influence from the LLMs’ training sets [ 45]. In some cases, LLMs have a similar perfor-
mance to current state-of-the-art benchmarks, while in others, they are significantly outperformed.
As one solution, empirical research into the performance of specific models, namely the ChatGPT
suite, has shown potential for improvement upon out-of-the-box model generation ability through
the implementation of iterative test refinement [46].
4 Proposed Approach
Fig. 1. Overview of the proposed RAG-based framework for automated code inspection and test case genera-
tion. The pipeline begins with dataset ingestion, where (a) theBug In the Code Stackdataset from [ 47] is used
for automated code inspection, and (b) the benchmark dataset from Wang et al. [ 11] is used for automated
test case generation. External knowledge (e.g., MBPP dataset [ 48,49]) is embedded using MiniLM-L6-V2 via
the Sentence Transformer library. In the retrieval stage, query vectors are matched with stored embeddings
using cosine similarity to identify the top-K relevant documents. These retrieved contexts are then passed to
the LLM for code analysis and test case generation, followed by evaluation to measure effectiveness.
While LLMs show promise across software development, their use in V&V remains limited
due to persistent challenges with modelhallucinations, where outputs diverge from the input,
context, or factual knowledge[ 50]. Such errors threaten the reliability of tasks like code inspection
and automated test generation, undermining trust in LLM-driven workflows. To address this, we
propose integrating an RAG pipeline that supplies domain-specific external knowledge during
generation with the goal of reducing hallucination risks. In Figure 1, we outline the proposed
approach.
, Vol. 1, No. 1, Article . Publication date: April 2026.

Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation 7
4.1 Automated Code Inspection
For automated code inspection, we begin by initializing our model to work with a user-specified
LLM setup. Several arguments are allowed for this configuration, including the model, temperature,
max number of tokens, and whether to enable a RAG knowledge base.
Following the processing of user arguments, we instantiate a pre-trained sentence transformer
embedding model that later enables us to encode query data into embeddings. This marks the
beginning of the Pipeline Setup component of the model pipeline. In our base configuration, we
use all-MiniLM-L6-v2 [ 51] as our embedding model. Once our embedding model is configured, we
implement in-memory vector storage. This will store our embeddings for easy access. With the
framework for our model now established, we can load data and our knowledge documents.
In parallel, we prepare our knowledge base corpus by loading JSON/JSONL data from a GitHub
repository. As a foundational knowledge base, this pipeline uses the Mostly Basic Python Program-
ming (MBPP) dataset developed by Austin et al. [ 48,49]. This dataset contains 1000 crowd-sourced
Python code snippets, useful as examples of well-written and error-freegolden standards.
Our model parses this data line by line for a task ID, content, and metadata; each task and
accompanying data are saved as a separate ‘document.’
Following data loading and model activation, we move to the second step: knowledge base
configuration (i.e., Retrieval). Here, if the RAG flag in the command-line arguments is enabled, we
prepare our RAG pipeline. Using the embedding model and loaded documents from our pipeline
setup, we generate embeddings for these documents and load the complete embedded corpus into
the vector storage.
Following the RAG setup, we enter the main processing loop. The most focal component of this
step is the query object. To create an instance of the query class, our model iterates through each
data point in the loaded dataset and prepares the data as a query object. To do this, we process
the data and save class variables for each relevant piece of metadata. Next, we run a function that
randomly selects one of the eight code snippets. This function returns the ground truth of the
snippet’s validity status (bug or no bug) and the selected data point.
From here, the selected data point is encoded into an embedding, which is input into the vector
store for a similarity search. To find relevant documents, we utilize the Python library NumPy to
compute cosine similarity, which involves calculating the dot product of the query and document
vectors and normalizing by their magnitudes, and then returning the topkresults.
Using these sourced documents and the previously formatted query, the model generates a
prompt with the selected output and enriches it with knowledge base context. From here, the
completed prompt (see Figure 2) is fed into an LLM, which then returns a code analysis of the
provided snippet.
Once code analysis is produced for each data point, the data are saved and funneled into the
evaluation process. With this data, we evaluate our model’s code analysis results by contrasting
the generated bug type with the labeled bug type collected during the selection process. If the LLM
correctly identified the root of the bug (if present), its response is labeled as a ‘match. ’ Alternatively,
if the LLM failed to identify a bug within the code snippet or misclassified the bug type, its response
is labeled as a ‘mismatch. ’ We run this evaluation through a new LLM conversation, with OpenAI’s
gpt-3.5 Turbo as the default model.
To assess the effectiveness of our model’s automated code inspection, we first compare the error
detection rate of the RAG-enhanced and standard LLMs with the average performance benchmark
for human code inspectors, which is previously established at 60% in this paper’s introduction [ 1].
Then, we compare performance with and without the presence of a RAG knowledge base.
, Vol. 1, No. 1, Article . Publication date: April 2026.

8 Fingleton et al.
Fig. 2. Prompt for code inspection
4.2 Automated Test Case Generation
For the pipeline and knowledge base setup of the automated test case generation, we prepare our
model using the same methodology as automated code inspection, creating and configuring our
embedding model and vector storage.
In our main processing loop, we use the same general framework as automated code inspection,
with some systemic differences. To prepare our queries after processing each data point as a query
class object, we save only the program code, function name, and function description. From here,
we embed the query and use the same similarity search functionality previously described.
Fig. 3. Prompt for test generation
Following embedding, the pipeline provides the specified LLM with the program under test, the
function name and description, and the knowledge base context (see Figure 3). The completed
prompt is fed into an LLM, which then returnsnunique test cases.
To ensure that each test case is independent of previous generations, we first run the query and
document context through a single test generation function. Following the completion of this cycle,
the result is fed into a multi-round generation function that takes all previous generations as input.
, Vol. 1, No. 1, Article . Publication date: April 2026.

Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation 9
Once raw test cases are produced for each data point, the data are saved and funneled into
evaluation. We pre-process our data by formatting the results into a structured form. With this
data, we evaluate our model’s test case generation results in terms of overall, line, path, and branch
coverage. To achieve this, we utilize TestEval’s source code, which cross-references generated test
cases with a complementary metadata category, ‘branches,’ that details all the task’s branches.
Coverage is determined by the proportion of lines/branches covered by at least one test case using
pytest-cov.
Table 1. Results of automated test generation on the TestEval dataset (210 tasks) [52]
Model RAGCompilation success Test case coverage Line𝑐𝑜𝑣@𝑘Branch𝑐𝑜𝑣@𝑘
syntax execution line branch𝑘=1𝑘=2𝑘=5𝑘=1𝑘=2𝑘=5
GPT-3.5 Turbo X 99.95 94.61 93.26 89.96 85.66 87.47 89.76 78.05 80.80 84.43
GPT-3.5 Turbo✓100 98.76 96.64 93.20 88.35 90.03 92.63 80.35 83.02 87.10
GPT-4o X99.9098.80 98.27 96.84 89.00 91.13 93.66 81.56 84.90 89.03
GPT-4o✓99.8099.33 98.51 96.85 89.39 91.45 94.02 81.89 85.12 89.19
Table 2. Efficiency of automated test generation on the TestEval dataset (210 tasks) [52]
Model RAG Runtime
GPT-3.5 Turbo X 1:32:30
GPT-3.5 Turbo✓1:28:59
GPT-4o X1:55:26
GPT-4o✓2:09:43
5 Experimental Results
5.1 Experimental Dataset
For our automated code inspection experimental data, we use the Bug In the Code Stack dataset
provided by Lee et al [ 47]. This dataset comprises 4,010 programming tasks. For each task in this
dataset, there are eight code snippets: a bug-free code sample and sevenbuggyversions (e.g., the
correct code missing a colon or misusing a keyword as a variable name). The dataset also contains
metadata specifying the line at which buggy code occurs.
For our automated test case generation dataset, we locally use the benchmark dataset provided
by Wang et al [ 11]. This repository’s data comprises coding assessments from LeetCode, an online
programming platform, and contains code for computing line and branch coverage.
As the central knowledge base, both pipelines will be using the Mostly Basic Python Programming
(MBPP) dataset seen in the code inspection model architecture [ 48,49]. This dataset is relevant
to automated inspection, as it contains 1000 examples of well-written code, and is relevant to
automated test case generation, as it contains corresponding sets of Python problems, solutions,
and test cases.
5.2 Experimental Setup
In this experimental study, we evaluate three commercial LLMs from the OpenAI GPT line (GPT-3.5
Turbo, GPT-4o, and GPT-4.1) using the Chat Completions API. All experiments were conducted
, Vol. 1, No. 1, Article . Publication date: April 2026.

10 Fingleton et al.
between May and Aug 2025 to ensure consistency in model availability. To minimize stochastic
variation, we fixed the temperature at 0, yielding deterministic outputs, and constrained responses
to a maximum of 256 tokens, which provides concise yet sufficiently informative outputs.
Prompts were designed in a zero-shot format and kept identical across models to ensure fair
comparison. Each prompt was executed programmatically through a Python-based evaluation
framework to guarantee reproducibility and consistent logging.
5.3 Experimental Analysis
5.3.1 Automated Test Case Generation.For automated test case generation, we test the coverage of
LLM-provided test cases. To evaluate automated test case generation ability, we directly implement
a RAG pipeline alongside TestEval’s source code. Using command line arguments as a RAG ‘toggle’,
we can compare performance on the same data with and without a supplementary knowledge base.
With this toggle determining whether a RAG database will be provided, we prompt our model to
generateNunique test cases. In this experimental configuration,N= 20.
Following the generation of the provided quantity of test cases, we evaluate the model’s perfor-
mance by computing compilation success, in terms of syntax and execution, test case coverage, in
terms of line and branch coverage, and a proprietary coverage metric developed by Wang et al. in
TestEval,cov@k.
All evaluation methodology for test case generation remains unchanged from TestEval’s source
code, as of July 2025. To compute compilation success, the program attempts to execute without
syntax errors. For coverage metrics, pytest-cov, a plugin that produces coverage reports, is used to
determine the proportion of branches covered by at least one test case. Finally,cov@krandomly
sampleskgenerated test cases and finds accuracy. Our findings for test case coverage are outlined
in Table 1.
Regarding these results, several observable trends are apparent. Firstly, under the current exper-
imental configuration, RAG-enhanced LLMs consistently outperform standard models in terms
of execution success. Furthermore, RAG-enhanced LLMs have a greater rate of coverage across
all areas of measured line and branch coverage, includingcov@k. Compilation success in terms
of syntax appears to be relatively uninfluenced by the presence of a RAG pipeline. These results
strongly indicate that the addition of a RAG pipeline has an overall positive influence on automated
test case generation, with no substantial drawbacks visible at this time. According to Table 2 and
also the visualization in Figure 4 , for automated test generation on the TestEval dataset, GPT-3.5
Turbo shows a clear runtime benefit from RAG, completing tasks about 3.5 minutes faster (1:32:30 →
1:28:59), suggesting retrieval helped streamline generation. Conversely, GPT-4o’s runtime increases
by over 14 minutes with RAG (1:55:26 →2:09:43), indicating that retrieval overhead outweighed
any efficiency gains. Overall, GPT-3.5 Turbo with RAG is optimal for speed in this setting, while
GPT-4o performs best without RAG.
5.3.2 Automated Code Inspection.For automated code inspection, we test LLM accuracy in correctly
identifying bugs in code snippets. Using the same configurable RAG pipeline outlined in automated
test case generation, we prompt an LLM to locate and classify code defects in a provided snippet of
code. For our evaluation, we compare the LLM-assigned bug label with the ground truth sourced
from the Bug In the Code Stack dataset. This comparison is performed using OpenAI’s GPT-3.5
Turbo [ 53]. Our findings are outlined in Tables 4 and 3, and Figure 8, 9, 10, and 11. Also, Figure 5
illustrates the results of code analysis performed by GPT-3.5-Turbo with external context enabled
through the RAG pipeline. The model identifies multiple bug types, including missing colons in
function definitions, missing commas in list syntax, use of reserved keywords as identifiers, and
mismatched brackets. In some cases, the model correctly reports that the code is free of errors.
, Vol. 1, No. 1, Article . Publication date: April 2026.

Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation 11
Fig. 4. Efficiency of Automated Test Generation
Fig. 5. Code Analysis Predictions using GPT-3.5-turbo, where (a) theBug In the Code Stackdataset from [ 47]
is used as an input dataset
When comparing the performance of standard models to those enhanced with RAG, several
notable patterns emerge. First, in Table 4, we observe that all models show improved accuracy in
bug detection when they have access to a RAG database.
In Figures 8, 9, 10 , and 11, we can evaluate model performance in classifying specific bug types.
RAG-enhanced LLMs consistently perform better in identifying ‘missing comma’, ‘mismatched
quotation’, ‘mismatched bracket’, and ‘keyword as identifier’ type bugs. The ‘bug-free code’,
‘missing colon’, ‘missing parenthesis’, and ‘missing quotation’ bug types are more variable, although
RAG-enhanced LLMs have a better overall performance.
Table 3 shows overall trends in which bug types are more challenging for the LLMs to classify.
There are fewer consistent patterns present in these data, although we do see a uniformly stronger
performance by RAG-enhanced models in identifying ‘mismatched quotation’, ‘mismatched bracket’,
and ‘keyword as identifier’ types of bugs. Consistent with our findings at this point, the ‘bug-free
code’, ‘missing colon’, and ‘missing parenthesis’ bug-types are less evenly distributed, as well
as the ‘missing comma’ label. However, again, we see that RAG-enhanced LLMs have a better
overall performance. Our results show that incorporating a RAG pipeline consistently enhances the
accuracy of automated code inspection. Furthermore, RAG-enhanced LLMs frequently outperform
standard LLMs in code inspection across bug types.
Also, the results in Table 5 and also the visualization in Figure 6 show that GPT-3.5 Turbo without
RAG achieves the fastest runtime (1:05:28), while adding RAG increased its runtime by about 15
minutes due to retrieval overhead. In contrast, GPT-4o benefits the most from RAG, reducing
runtime by roughly 8 minutes (1:27:35 →1:19:26), suggesting that retrieval helped streamline
, Vol. 1, No. 1, Article . Publication date: April 2026.

12 Fingleton et al.
its code inspection process. GPT-4.1 sees only a marginal improvement of about 1 minute with
RAG, indicating limited runtime sensitivity to retrieval. Overall, GPT-3.5 Turbo without RAG is
optimal for speed, GPT-4o with RAG offers the best runtime improvement, and GPT-4.1’s runtime
performance is largely unaffected by retrieval.
Fig. 6. Efficiency of Automated Code Inspection
The evaluation results of the code analysis using GPT-3.5-Turbo and GPT-4.1 are presented in
Figure 7.
Fig. 7. Evaluation Results of Code Analysis/Inspection Predictions
Note that the results in Figures 5 and 7 come from a separate run compared to the results in the
tables. The difference in outcomes is due to the inherent variability of the LLM, which can occur
even with the temperature set to 0.
Table 3. Mismatch type rate (proportional to overall mismatches)
Model RAG Bug-free code Missing colon Missing parenthesis Missing quotation Missing comma Mismatched quotation Mismatched bracket Keyword as identifier
GPT-3.5 Turbo X2.55%27.03% 13.74%5.78%7.13% 5.41% 6.53% 31.83%
GPT-3.5 Turbo✓47.35%8.47% 8.2%10.32%6.88% 4.5% 0.53% 13.76%
GPT-4 X24.46%28.3%2.62% 5.9% 3.75%5.06% 4.22% 25.68%
GPT-4✓50.0%8.96%8.46% 7.46% 6.97%3.48% 1.0% 13.68%
GPT-4o X 48.55%4.34%7.07%6.755.47% 4.18% 2.09% 21.54%
GPT-4o✓48.52%10.59%6.4%12.07%4.43% 3.94% 0.74% 13.3%
, Vol. 1, No. 1, Article . Publication date: April 2026.

Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation 13
Fig. 8. GPT-3.5 Turbo: Mismatch Type Count Comparison (RAG vs Non-RAG)
Bug-free code Missing colon
Missing parenthesisMissing quotationMissing comma
Mismatched quotationMismatched bracket Keyword as identifier0200400CountWithout RAG
With RAG
Fig. 9. GPT-4: Mismatch Type Count Comparison (RAG vs Non-RAG)
Bug-free code Missing colon
Missing parenthesisMissing quotationMissing comma
Mismatched quotationMismatched bracket Keyword as identifier0200400CountWithout RAG
With RAG
Fig. 10. GPT-4o: Mismatch Type Count Comparison (RAG vs Non-RAG)
Bug-free code Missing colon
Missing parenthesisMissing quotationMissing comma
Mismatched quotationMismatched bracket Keyword as identifier0200400CountWithout RAG
With RAG
Fig. 11. Mismatch Type Count Comparison (GPT-3.5, GPT-4, GPT-4o) between RAG and Non-RAG
, Vol. 1, No. 1, Article . Publication date: April 2026.

14 Fingleton et al.
Table 4. Automated code inspection results on the Bug In the Code Stack dataset (4,010 tasks) [54]
Model RAG Matches Mismatches Accuracy
GPT-3.5 Turbo X 2678 1332 66.78
GPT-3.5 Turbo✓3632 378 90.57
GPT-4 X 2943 1067 73.39
GPT-4✓3608 402 89.98
GPT-4o X 3388 622 84.49
GPT-4o✓3604 406 89.88
Table 5. Efficiency of automated code inspection on the Bug In the Code Stack dataset (4,010 tasks) [54]
Model RAG Runtime
GPT-3.5 Turbo X1:05:28
GPT-3.5 Turbo✓1:20:59
GPT-4o X 1:27:35
GPT-4o✓1:19:26
GPT-4.1 X 1:25:18
GPT-4.1✓1:24:20
6 Threats to Validity
While our proposed approach aims to mitigate hallucinations in LLM-based V&V, several validity
threats remain. In the following, we discuss threats to Internal, External, and Construct Validity.
6.1 Internal Validity
Our evaluation depends on specific datasets (e.g., MBPP) and selected LLM configurations. Biases
or limitations in these datasets (such as incomplete coverage of real-world code defects or limited
test case diversity) may have influenced the observed effectiveness of the RAG-enhanced pipeline.
Moreover, hyperparameter settings (e.g., temperature, context length, retrieval top-k) could affect
model behavior in ways not fully controlled in our experiments. As part of our mitigation strategy,
in future work, we will expand to additional datasets, incorporate diverse benchmark suites, and
perform hyperparameter tuning. We also plan to adopt adaptive retrieval strategies that dynamically
adjust retrieval parameters to reduce configuration-induced variance.
6.2 External Validity
The generalizability of our findings is constrained by the domains and datasets used. While MBPP
and TestEval are widely adopted in prior studies, they may not fully capture the complexity of
large-scale industrial software systems. As such, our results may not directly transfer to highly
domain-specific or safety-critical environments (e.g., healthcare or finance), where codebases, defect
types, and testing requirements differ significantly. Future studies will evaluate the pipeline on
large, real-world, domain-specific repositories and integrate additional external knowledge sources
as context.
6.3 Construct Validity
The metrics used to measure effectiveness, such as error detection rate for inspection and coverage
metrics (line, branch, path) for test generation, provide useful proxies for reliability but do not
, Vol. 1, No. 1, Article . Publication date: April 2026.

Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation 15
fully capture the broader notion of software quality. For instance, high test coverage does not
guarantee defect detection, and identifying a bug type may not equate to producing actionable fixes.
Additionally, hallucination reduction was assessed indirectly through improvements in factual
correctness and alignment with ground truth, which may not capture all forms of unfaithfulness.
To address these threats, we plan to incorporate semantic-oriented metrics such as CodeBLEU,
extend the evaluation to include fix viability and actionable repair quality, and adopt finer-grained
hallucination assessments.
6.4 Conclusion Validity
Our comparative analysis between RAG-enhanced and baseline LLMs is based on limited experi-
mental runs and a single human performance benchmark (60% accuracy). Statistical fluctuations or
variations in LLM outputs may affect reproducibility. Furthermore, conclusions regarding efficiency
and effectiveness gains must be interpreted cautiously, as retrieval introduces computational over-
head that may scale differently in larger deployments. To mitigate these threats, future work will
include repeated trials, cross-run variance analysis, and statistical significance testing to ensure
robust conclusions. We will also profile retrieval overhead more extensively and examine scalability
under larger workloads and alternative deployment configurations.
7 Conclusion and Future Work
In this paper, we have proposed a novel approach to automated software test case generation
and automated code inspection using LLMs. Our experimental results indicate that LLMs hold
significant potential for applications in software testing and inspection. Regarding our initial
research questions, ‘Can LLMs w/o RAG perform code inspection effectively and efficiently?’ and
‘Can LLMs w/o RAG perform test case generation effectively and efficiently?’, we have validated
these questions. In particular, we have found that the addition of a RAG pipeline generally increases
the effectiveness of these models. Notably, RAG-enhanced models consistently outperform their
standard counterparts in terms of test case generation coverage and their overall accuracy in
identifying and classifying bugs during code inspection.
Future studies will investigate adaptive retrieval strategies that trigger retrieval only when they
are likely to be beneficial, thereby reducing unnecessary overhead. Scaling the experiments to
a broader range of models, especially more recent ones such as GPT-5, which was not available
at the time of our evaluation, would help confirm the extent to which the observed runtime and
performance patterns generalize better. Detailed profiling of the retrieval pipeline may also reveal
optimization opportunities, while exploring hybrid frameworks that combine selective retrieval
with in-context learning could offer a better balance between speed and accuracy. In addition, future
work will evaluate alternative external knowledge sources to understand how different context
types influence code generation quality and efficiency. Complementary metrics, such as CodeBLEU,
will be used to assess semantic correctness, providing a more holistic evaluation. Finally, we plan
to incorporate statistical significance testing to determine whether the observed differences are
robust and not due to random variation.
Data Availability
The source code and data are accessible at [ 55]. This repository contains code to run and evaluate
automated test case generation and automated code review using OpenAI, Ollama, and Hugging
Face models. Additionally, this work uses data from Google’s Mostly Basic Python Programming
dataset [ 49], the Bug In the Code Stack dataset by Lee et al. [ 47], and the TestEval dataset from
Wang et al. [ 11]. The relevant experimental data used in this study are accessible in the source code
, Vol. 1, No. 1, Article . Publication date: April 2026.

16 Fingleton et al.
repository. To reproduce the experimental setup, users will need to download TestEval’s code [ 52]
from the source. Details are specified in the README.md for this paper’s repository.
Acknowledgment
This material is based upon work supported by the U.S. National Science Foundation (NSF) under
Grant No. 2349452. Any opinions, findings, conclusions, or recommendations expressed in this
material are those of the authors and do not necessarily reflect the views of the NSF. In preparing
this work, we used generative AI models and tools, including the OpenAI GPT models, to assist
with generating and revising content, including code and text. Finally, the authors would like to
thank Himon Thakur for discussions and comments.
References
[1]I. Sommerville,Software engineering. Always learning, Boston Columbus Indianapolis New York San Francisco Hoboken
Amsterdam Cape Town Dubai London: Pearson, tenth edition ed., 2016.
[2]S. Wang, Y. Yu, R. Feldt, and D. Parthasarathy, “Automating a complete software test process using llms: An automotive
case study,”arXiv preprint arXiv:2502.04008, 2025.
[3]F. Rabbi and M. S. Siddik, “Detecting code comment inconsistency using siamese recurrent network,” inProceedings of
the 28th international conference on program comprehension, pp. 371–375, 2020.
[4]Y. Zhou, R. Gu, T. Chen, Z. Huang, S. Panichella, and H. Gall, “Analyzing apis documentation and code to detect
directive defects,” in2017 IEEE/ACM 39th International Conference on Software Engineering (ICSE), pp. 27–37, IEEE, 2017.
[5]Y. Zhang, “Detecting code comment inconsistencies using llm and program analysis,” inCompanion Proceedings of the
32nd ACM International Conference on the Foundations of Software Engineering, pp. 683–685, 2024.
[6]Z. Zhang, X. Liu, Y. Lin, X. Gao, H. Sun, and Y. Yuan, “Reference-based retrieval-augmented unit test generation,”ACM
Transactions on Software Engineering and Methodology, 2025.
[7]Z. Deng, Z. Lai, Y. Ding, H. Kong, and X. Wu, “Deep scaling factor quantization network for large-scale image retrieval, ”
inProceedings of the 2024 international conference on multimedia retrieval, pp. 851–859, 2024.
[8]W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li, “A survey on rag meeting llms: Towards
retrieval-augmented large language models, ” inProceedings of the 30th ACM SIGKDD conference on knowledge discovery
and data mining, pp. 6491–6501, 2024.
[9]U. Khandelwal, O. Levy, D. Jurafsky, L. Zettlemoyer, and M. Lewis, “Generalization through memorization: Nearest
neighbor language models,”arXiv preprint arXiv:1911.00172, 2019.
[10] G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open domain question answering,”
arXiv preprint arXiv:2007.01282, 2020.
[11] W. Wang, C. Yang, Z. Wang, Y. Huang, Z. Chu, D. Song, L. Zhang, A. R. Chen, and L. Ma, “TESTEVAL: Benchmarking
Large Language Models for Test Case Generation,” Feb. 2025. arXiv:2406.04531 [cs].
[12] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, Y. Du, C. Yang, Y. Chen,
Z. Chen, J. Jiang, R. Ren, Y. Li, X. Tang, Z. Liu, P. Liu, J.-Y. Nie, and J.-R. Wen, “A Survey of Large Language Models,”
2023. Version Number: 16.
[13] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is all
you need,”Advances in neural information processing systems, vol. 30, 2017.
[14] H. Naveed, A. U. Khan, S. Qiu, M. Saqib, S. Anwar, M. Usman, N. Akhtar, N. Barnes, and A. Mian, “A Comprehensive
Overview of Large Language Models,”ACM Transactions on Intelligent Systems and Technology, June 2025. Publisher:
Association for Computing Machinery (ACM).
[15] K. W. Church, Z. Chen, and Y. Ma, “Emerging trends: A gentle introduction to fine-tuning,”Natural Language Engineering,
vol. 27, pp. 763–778, Nov. 2021.
[16] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language
understanding,” inProceedings of the 2019 conference of the North American chapter of the association for computational
linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.
[17] V. B. Parthasarathy, A. Zafar, A. Khan, and A. Shahid, “The Ultimate Guide to Fine-Tuning LLMs from Basics to
Breakthroughs: An Exhaustive Review of Technologies, Research, Best Practices, Applied Research Challenges and
Opportunities,” Oct. 2024. arXiv:2408.13296 [cs].
[18] B. Zhang, Z. Liu, C. Cherry, and O. Firat, “When Scaling Meets LLM Finetuning: The Effect of Data, Model and
Finetuning Method,” Feb. 2024. arXiv:2402.17193 [cs].
[19] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, and T. Liu, “A Survey on
Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions,”ACM Transactions
, Vol. 1, No. 1, Article . Publication date: April 2026.

Enhancing Large Language Models with Retrieval Augmented Generation for Software Testing and Inspection Automation 17
on Information Systems, vol. 43, pp. 1–55, Mar. 2025.
[20] P. Béchard and O. M. Ayala, “Reducing hallucination in structured outputs via Retrieval-Augmented Generation,” in
Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 6: Industry Track), pp. 228–238, 2024. arXiv:2404.08189 [cs].
[21] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel,
and D. Kiela, “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,” Apr. 2021. arXiv:2005.11401 [cs].
[22] Y. Lyu, Z. Li, S. Niu, F. Xiong, B. Tang, W. Wang, H. Wu, H. Liu, T. Xu, and E. Chen, “CRUD-RAG: A Comprehensive
Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models,”ACM Transactions on Information
Systems, vol. 43, pp. 1–32, Mar. 2025. Publisher: Association for Computing Machinery (ACM).
[23] M. R. Parvez, W. U. Ahmad, S. Chakraborty, B. Ray, and K.-W. Chang, “Retrieval Augmented Code Generation and
Summarization,” Sept. 2021. arXiv:2108.11601 [cs].
[24] Z. Levonian, C. Li, W. Zhu, A. Gade, O. Henkel, M.-E. Postle, and W. Xing, “Retrieval-augmented Generation to Improve
Math Question-Answering: Trade-offs Between Groundedness and Human Preference,” Nov. 2023. arXiv:2310.03184
[cs].
[25] X. Yang, M. Ye, Q. You, and F. Ma, “Writing by Memorizing: Hierarchical Retrieval-based Medical Report Generation,”
May 2021. arXiv:2106.06471 [cs].
[26] P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y. Geng, F. Fu, L. Yang, W. Zhang, J. Jiang, and B. Cui, “Retrieval-Augmented
Generation for AI-Generated Content: A Survey,” June 2024. arXiv:2402.19473 [cs].
[27] G. Bavota and B. Russo, “Four eyes are better than two: On the impact of code reviews on software quality,” in2015
IEEE International Conference on Software Maintenance and Evolution (ICSME), pp. 81–90, IEEE, 2015.
[28] M. E. Fagan, “Design and code inspections to reduce errors in program development,”IBM Systems Journal, vol. 38,
no. 2.3, pp. 258–287, 1999.
[29] R. Heumüller and F. Ortmeier, “Previously on... automating code review,”arXiv preprint arXiv:2508.18003, 2025.
[30] Q. Guo, J. Cao, X. Xie, S. Liu, X. Li, B. Chen, and X. Peng, “Exploring the Potential of ChatGPT in Automated Code
Refinement: An Empirical Study, ” inProceedings of the IEEE/ACM 46th International Conference on Software Engineering,
(Lisbon Portugal), pp. 1–13, ACM, Feb. 2024.
[31] Y. Hao, W. Chen, Z. Zhou, and W. Cui, “E&V: Prompting Large Language Models to Perform Static Analysis by
Pseudo-code Execution and Verification,” Dec. 2023. arXiv:2312.08477 [cs].
[32] C. Pornprasit and C. Tantithamthavorn, “Fine-tuning and prompt engineering for large language models-based code
review automation,”Information and Software Technology, vol. 175, p. 107523, Nov. 2024.
[33] U. Cihan, V. Haratian, A. Icoz, M. K. Gul, O. Devran, E. F. Bayendur, B. M. Ucar, and E. Tuzun, “Automated code review
in practice,” in2025 IEEE/ACM 47th International Conference on Software Engineering: Software Engineering in Practice
(ICSE-SEIP), pp. 425–436, 2025.
[34] H. Y. Lin, C. Liu, H. Gao, P. Thongtanunam, and C. Treude, “CodeReviewQA: The Code Review Comprehension
Assessment for Large Language Models,” Mar. 2025. arXiv:2503.16167.
[35] O. Ben Sghaier and H. Sahraoui, “Improving the learning of code review successive tasks with cross-task knowledge
distillation,”Proceedings of the ACM on Software Engineering, vol. 1, no. FSE, pp. 1086–1106, 2024.
[36] Z. Li, S. Lu, D. Guo, N. Duan, S. Jannu, G. Jenks, D. Majumder, J. Green, A. Svyatkovskiy, S. Fu,et al., “Automating
code review activities by large-scale pre-training,” inProceedings of the 30th ACM Joint European Software Engineering
Conference and Symposium on the Foundations of Software Engineering, pp. 1035–1047, 2022.
[37] Y. Zhao, Z. Luo, Y. Tian, H. Lin, W. Yan, A. Li, and J. Ma, “CodeJudge-Eval: Can Large Language Models be Good Judges
in Code Understanding?,” Sept. 2024. arXiv:2408.10718 [cs].
[38] K. Sun, H. Kuang, S. Baltes, X. Zhou, H. Zhang, X. Ma, G. Rong, D. Shao, and C. Treude, “Does ai code review lead to
code changes? a case study of github actions,”arXiv preprint arXiv:2508.18771, 2025.
[39] C. Fang, N. Miao, S. Srivastav, J. Liu, R. Zhang, R. Fang, R. Tsang, N. Nazari, H. Wang, and H. Homayoun, “Large
language models for code analysis: Do {LLMs} really do their job?,” in33rd USENIX Security Symposium (USENIX
Security 24), pp. 829–846, 2024.
[40] W. Miller and D. Spooner, “Automatic Generation of Floating-Point Test Data,”IEEE Transactions on Software Engineering,
vol. SE-2, pp. 223–226, Sept. 1976.
[41] M. Schäfer, S. Nadi, A. Eghbali, and F. Tip, “An Empirical Evaluation of Using Large Language Models for Automated
Unit Test Generation,”IEEE Transactions on Software Engineering, vol. 50, pp. 85–105, Jan. 2024.
[42] S. Kang, J. Yoon, and S. Yoo, “Large Language Models are Few-shot Testers: Exploring LLM-based General Bug
Reproduction,” in2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pp. 2312–2323, May
2023. ISSN: 1558-1225.
[43] Y. Chen, Z. Hu, C. Zhi, J. Han, S. Deng, and J. Yin, “Chatunitest: A framework for llm-based test generation,” in
Companion Proceedings of the 32nd ACM International Conference on the Foundations of Software Engineering, pp. 572–
576, 2024.
, Vol. 1, No. 1, Article . Publication date: April 2026.

18 Fingleton et al.
[44] K. Li and Y. Yuan, “Large language models as test case generators: Performance evaluation and enhancement,”arXiv
preprint arXiv:2404.13340, 2024.
[45] M. L. Siddiq, J. C. Da Silva Santos, R. H. Tanvir, N. Ulfat, F. Al Rifat, and V. Carvalho Lopes, “Using Large Language
Models to Generate JUnit Tests: An Empirical Study,” inProceedings of the 28th International Conference on Evaluation
and Assessment in Software Engineering, (Salerno Italy), pp. 313–322, ACM, June 2024.
[46] Z. Yuan, Y. Lou, M. Liu, S. Ding, K. Wang, Y. Chen, and X. Peng, “No More Manual Tests? Evaluating and Improving
ChatGPT for Unit Test Generation,” May 2024. arXiv:2305.04207 [cs].
[47] H. Lee, S. Sharma, and B. Hu, “Bug In the Code Stack: Can LLMs Find Bugs in Large Python Code Stacks,” June 2024.
arXiv:2406.15325 [cs].
[48] J. Austin, A. Odena, M. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang, C. Cai, M. Terry, Q. Le, and C. Sutton,
“Program Synthesis with Large Language Models,” Aug. 2021. arXiv:2108.07732 [cs].
[49] Google Research, “Mbpp: Mostly basic python problems (google research github repository). ” https://github.com/google-
research/google-research/tree/master/mbpp, 2025. Accessed: 2025-08-13.
[50] Z. Zhang, C. Wang, Y. Wang, E. Shi, Y. Ma, W. Zhong, J. Chen, M. Mao, and Z. Zheng, “Llm hallucinations in practical
code generation: Phenomena, mechanism, and mitigation,”Proceedings of the ACM on Software Engineering, vol. 2,
no. ISSTA, pp. 481–503, 2025.
[51] Hugging Face, “sentence-transformers/all-minilm-l6-v2.” https://huggingface.co/sentence-transformers/all-MiniLM-
L6-v2, jan 2024. Accessed: 2025-07-21.
[52] LLM4SoftwareTesting, “LLM4SoftwareTesting: Testeval,” Jan. 2025.
[53] OpenAI, “GPT-3.5 Turbo,” 2024.
[54] HammingHQ, “HammingHQ: Bug in the code stack,” June 2024.
[55] QAS Lab, “QAS Lab: Zoe fingleton, month = jul, howpublished = https://github.com/qas-lab/reu-zoe-fingleton, note =
Accessed: 2025-07-31.”
, Vol. 1, No. 1, Article . Publication date: April 2026.