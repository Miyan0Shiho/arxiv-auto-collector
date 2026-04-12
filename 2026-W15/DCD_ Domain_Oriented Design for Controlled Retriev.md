# DCD: Domain-Oriented Design for Controlled Retrieval-Augmented Generation

**Authors**: Valeriy Kovalskiy, Nikita Belov, Nikita Miteyko, Igor Reshetnikov, Max Maximov

**Published**: 2026-04-08 20:47:51

**PDF URL**: [https://arxiv.org/pdf/2604.07590v1](https://arxiv.org/pdf/2604.07590v1)

## Abstract
Retrieval-Augmented Generation (RAG) is widely used to ground large language models in external knowledge sources. However, when applied to heterogeneous corpora and multi-step queries, Naive RAG pipelines often degrade in quality due to flat knowledge representations and the absence of explicit workflows. In this work, we introduce DCD (Domain-Collection-Document), a domain-oriented design to structure knowledge and control query processing in RAG systems without modifying the underlying language model. The proposed approach relies on a hierarchical decomposition of the information space and multi-stage routing based on structured model outputs, enabling progressive restriction of both retrieval and generation scopes. The architecture is complemented by smart chunking, hybrid retrieval, and integrated validation and generation guardrail mechanisms. We describe the DCD architecture and workflow and discuss evaluation results on synthetic evaluation dataset, highlighting their impact on robustness, factual accuracy, and answer relevance in applied RAG scenarios.

## Full Text


<!-- PDF content starts -->

DCD: Domain-Oriented Design for Controlled
Retrieval-Augmented Generation
Valeriy Kovalskiy, Nikita Belov, Nikita Miteyko, Igor Reshetnikov, Max Maximov
red_mad_robot
Abstract
Retrieval-Augmented Generation (RAG) is widely used to ground large language models in external knowledge
sources. However, when applied to heterogeneous corpora and multi-step queries, Naive RAG pipelines often
degrade in quality due to flat knowledge representations and the absence of explicit workflows. In this work, we
introduce DCD (Domain–Collection–Document), a domain-oriented design to structure knowledge and control
query processing in RAG systems without modifying the underlying language model. The proposed approach
relies on a hierarchical decomposition of the information space and multi-stage routing based on structured model
outputs, enabling progressive restriction of both retrieval and generation scopes. The architecture is complemented
by smart chunking, hybrid retrieval, and integrated validation and generation guardrail mechanisms. We describe
the DCD architecture and workflow and discuss evaluation results on synthetic evaluation dataset, highlighting
their impact on robustness, factual accuracy, and answer relevance in applied RAG scenarios.
1. Introduction
RAG has gained widespread adoption as a practical approach for integrating language models with external knowledge
sources [Lewis et al., 2020]. Even basic Naive RAG implementations can effectively address a broad range of applied tasks,
from customer support query handling to enterprise document analysis [Izacard et al., 2021].
However, as data volumes grow and user queries become more complex, the limitations of standard RAG pipelines become
increasingly apparent. In particular, the linear architecture of Naive RAG exhibits degraded performance on multi-step
queries that require sequential interpretation, retrieval across multiple knowledge slices, and result aggregation [Wei et al.,
2023]. Another critical issue is the sensitivity of RAG systems to document segmentation parameters. These factors limit
the applicability of Naive RAG in scenarios demanding high accuracy and predictable behavior.
In this work, we consider an architectural approach to organizing knowledge and workflows around a language model,
aimed at improving the accuracy and safety of RAG systems without modifying the models themselves. We propose DCD
(Domain–Collection–Document), a domain-oriented design that introduces an explicit knowledge hierarchy and a controlled
query-processing workflow. The proposed approach demonstrates substantial improvements in answer quality for multi-step
queries while ensuring reproducibility and controllability in applied settings.
2. Preliminaries
2.1. Retrieval-Augmented Generation
We focus on improving the accuracy and robustness of RAG systems in scenarios involving multi-step queries and
heterogeneous knowledge corpora. Specifically, we consider architectural approaches that enable:
• restricting the retrieval space to relevant subsets of knowledge,
• explicit control over query processing workflows,
• reproducibility and quality assurance of generated outputs.
1arXiv:2604.07590v1  [cs.IR]  8 Apr 2026

Throughout this work, Naive RAG refers to a linear pipeline without explicit knowledge structuring, query decomposition,
or multi-stage generation control. We deliberately focus not on modifying the language model itself, but on organizing
knowledge and workflows around it.
In many practical RAG implementations, retrieval is performed once at the beginning of the pipeline and remains fixed
throughout the generation process. This design assumes that a single retrieval step can provide all information required to
answer the query. However, when the knowledge corpus contains documents from multiple semantic domains, similarity-
based retrieval may return fragments that are locally relevant but belong to different knowledge contexts. As a result,
the retrieved context may contain partially compatible evidence, forcing the language model to reconcile inconsistent
information during generation [Li et al., 2025].
We conceptualize a RAG system as a composition of the following stages:
• user query interpretation,
• selection of a relevant knowledge subset,
• retrieval and aggregation of contextual information,
• answer generation by a language model,
• validation and quality control of the output [Guu et al., 2020].
In standard implementations, these stages are often implicit or collapsed into a single step. In contrast, our approach treats
each stage as an explicit workflow component, enabling fine-grained control and additional validation.
2.2. Chunking
The quality of a RAG system is highly dependent on document segmentation strategies. Common chunking approaches
include paragraph-based segmentation, fixed-length chunks, and sliding-window techniques [Gao et al., 2021]. In our
implementation, we apply a sliding-window segmentation strategy with overlap ranging from 5–10% of the chunk size. A
5% overlap is typically sufficient to preserve contextual continuity between adjacent chunks. For documents containing
semantically similar or repetitive sections, the overlap is increased to 10% to better capture contextual dependencies.
2.3. Guardrails
Classical RAG implementations typically rely on built-in safety mechanisms of language models and lack explicit architec-
tural procedures for validating generated outputs [Ji et al., 2023]. In particular, they provide no systematic means to verify
alignment between generated answers and retrieved context or to refuse generation outside the system’s functional scope
[Lin et al., 2022]. Our system incorporates a guardrails mechanism combined with a hallucination prevention module. This
mechanism consists of two interrelated components. The first component analyzes user queries for prohibited topics using a
composite dictionary that includes both a base set of stop-words and custom elements defined by the system owner. The
second component analyzes generated outputs, ensuring the absence of prohibited content and detecting hallucinations. This
is achieved by comparing the original query, the retrieved context, and the generated response, enabling early detection of
deviations during generation.
3. Method
3.1. Key Assumption
The central methodological assumption of this work is that answer quality significantly improves when retrieval and
generation are constrained to semantically homogeneous knowledge regions — a subset of the corpus whose documents
share a common topical scope, terminology, and expected user intent, while remaining clearly distinguishable from
documents belonging to other regions [Yao et al., 2023]. Within such a region, documents address closely related subject
areas and therefore compete primarily with semantically relevant sources during retrieval.
This constraint allows the retrieval process to operate within a bounded semantic space, reducing interference from unrelated
documents and improving the alignment between retrieved evidence and the informational structure of the query. Rather
2

Figure 1.The difference between the RAG and DCD approaches
than operating over a monolithic document store, we assume a prior decomposition of the corpus into independent subspaces,
within which documents compete only with semantically similar sources [Khot et al., 2023].
This issue is closely related to the organization of the knowledge space itself. When retrieval operates over a heterogeneous
corpus without explicit semantic boundaries, documents from different knowledge regions compete within the same search
space. In such settings, embedding similarity alone may not reflect the structural dependencies between pieces of information
required to answer the query. As a result, retrieved evidence may be individually relevant but collectively insufficient for
coherent reasoning [Li et al., 2025].
Evidence retrieval studies for long documents demonstrate that coarse-to-fine search strategies can improve the identification
of relevant evidence compared to flat fragment-level retrieval. In such approaches, higher-level structural segments are
first identified and only then used to guide fine-grained passage selection. By narrowing the search space to semantically
coherent regions of a document, retrieval systems preserve global contextual relationships and improve the reliability of
downstream question answering [Nair et al., 2023].
The proposed DCD architecture addresses this limitation through explicit decomposition of the knowledge space prior to
retrieval. By constraining retrieval to semantically homogeneous regions, the system reduces cross-domain interference and
improves alignment between retrieved context and the reasoning structure of the query.
3.2. Experimental Validation of the Assumption
To validate this assumption, we conducted experiments on a synthetic dataset constructed to approximate real-world RAG
scenarios. The corpora consist of heterogeneous documents with varying structures and semantic granularity — conditions
under which pipelines most frequently degrade due to flat knowledge organization and unstable retrieval.
All datasets were anonymized for confidentiality: client identifiers and sensitive metadata were removed. The paper
reports procedures and observed effects without disclosing document contents. In addition to quality metrics, we recorded
operational characteristics of the pipeline, including total evaluation time, mean end-to-end response time, and time to first
token, allowing analysis of the trade-off between accuracy and latency under increased workflow complexity.
3

4. DCD: Domain–Collection–Document
The DCD (Domain–Collection–Document) Design is an approach to organizing knowledge in RAG systems through explicit
hierarchical segmentation of the information space. Its primary goal is to minimize overlap between knowledge areas and
prevent irrelevant context from being passed to the language model [Makin, 2024].
DCD structures data across three levels:
• Domain — a high-level subject area,
• Collection — a thematically homogeneous subset within a domain,
• Document — an atomic knowledge unit with metadata.
4.1. Hierarchical Levels
A Domain defines top-level search boundaries and is constructed to minimize semantic overlap with other domains. A
Collection represents a narrower thematic slice within a domain and further restricts the retrieval space. Examples include
legal documents, reference materials, or user FAQs within a single domain. A Document is the basic unit used in the RAG
process. Documents are enriched with metadata and may be further segmented into chunks for context retrieval. Information
retrieval follows a top-down principle: from domain to collection to document.
Navigation through the hierarchy is performed by the DCD Router, which sequentially selects the most relevant domain,
collection, and document for a given query. At each level, routing is formulated as a selection task over a limited candidate
set [Shazeer et al., 2017]. Router decisions are produced using structured outputs from the language model, ensuring
transparency, reproducibility, and intermediate result caching. To improve robustness, fallback elements (main domain, main
collection, main document) are defined at each level.
4.2. DCD Router
The functional core of domain segmentation is the DCD Router, a specialized module responsible for identifying the target
knowledge segment for retrieval. Like most system components, it relies on structured language model outputs (JSON),
enabling high-precision processing and transparent decision-making [OpenAI, 2024].
Initial configuration involves defining domain and collection structures and assigning knowledge assets accordingly.
Documents retain their original titles, while domains and collections receive manually defined identifiers. Upon receiving
a user query, the system submits it — along with the registry of available domains — to the language model to identify
the most relevant domain. This procedure is repeated hierarchically for collections and documents. If the language model
produces incorrect outputs, fallback elements at each level ensure system robustness and guarantee a response even after
multiple failed routing attempts.
Higher hierarchy levels are described with richer metadata to reduce routing errors, since mistakes at upper levels may
propagate downstream and restrict the search space to an incorrect semantic region of the knowledge base. Providing clearer
semantic descriptions for domains and collections improves the reliability of the initial routing decisions and helps maintain
stable navigation across the hierarchy.
In the present study, routing accuracy at individual hierarchy levels was not evaluated as a separate classification metric.
Instead, the routing mechanism is considered as part of the overall retrieval pipeline, and its effectiveness is reflected in
downstream system behaviour, including answer relevance, factual accuracy, and retrieval coverage.
4.3. Integration with the RAG Pipeline
After a document is selected, the system retrieves relevant fragments using semantic and full-text search, followed by
deduplication and reranking. The resulting context is passed to the language model for answer generation.
By constraining the retrieval space upfront, DCD reduces the likelihood of irrelevant context inclusion and improves
generation robustness on multi-step queries. While Naive RAG employs a linear processing pipeline — chunking, embedding,
retrieval, and generation — DCD preserves a similar user interface but introduces a fundamentally different internal
organization based on multi-level processing.
4

Figure 2.The difference between the RAG and DCD approaches
5

4.4. Smart Chunking
Smart chunking is an integral component of the DCD design. Although DCD can operate with classical chunking methods,
the selected strategy provided the most stable behaviour within the proposed architecture, as it better preserves the structural
properties of the underlying knowledge base.
In this approach, documents are segmented using a sliding window mechanism with partial overlap between neighbouring
fragments. This segmentation strategy preserves local semantic continuity and reduces the likelihood that relevant contextual
information will be lost at chunk boundaries. As a result, logically connected statements that span multiple sentences or
paragraphs remain accessible during retrieval, which is particularly important for queries requiring compositional reasoning.
Figure 3.Smart chunking pipeline
Each fragment is additionally enriched with chunk-level metadata derived from the document structure. These metadata
describe the fragment’s position within the document hierarchy and maintain references to the corresponding document-level
entity. Such structural annotations allow the retrieval stage to operate not only on semantic similarity but also on contextual
relationships between fragments originating from the same document or collection.
The combination of overlapping segmentation and structured metadata improves the consistency of fragment retrieval and
reduces the probability of assembling answers from semantically unrelated contexts. This design enables the retrieval
pipeline to better align selected evidence with the conceptual structure of the query and contributes to more stable answer
generation in retrieval-augmented workflows, as reflected in the experimental evaluation presented later in the paper.
4.5. Fast Guardrails
In addition to the main retrieval and generation pipeline, the system introduces a fast guardrails mechanism for early response
validation. The system includes a guardrails mechanism combined with hallucination prevention and a fast pre-check
function that evaluates responses before full generation completes [Manakul et al., 2023].
The guardrails workflow is as follows:
• the user query undergoes stop-word screening,
• compliant queries are passed to the DCD RAG pipeline,
• during response streaming, the first 150 tokens are analyzed,
• based on query, context, and partial output, the system decides whether to display or block the response.
This mechanism filters unsupported queries and validates generated content in parallel with generation, avoiding additional
latency.
6

4.6. System Completeness Summary
The DCD architecture demonstrates significant advantages in adaptability when processing heterogeneous data and struc-
turally complex queries. Its flexibility yields substantial accuracy improvements in multi-level knowledge scenarios while
integrating critical validation and control stages absent in Naive RAG systems. While initial configuration requires additional
setup effort, these costs are offset by superior performance and rapid amortization through improved operational efficiency.
5. Metrics
To comprehensively evaluate the proposed DCD approach, we employ a metric suite extending beyond standard evaluations.
The assessment relies on structured generation quality evaluation using an LLM as an assessor [Liu et al., 2023].
5.1. Strict Binary Answer Relevance & Completeness
SBARCis a strict binary metric assessing whether an answer is both relevant and complete with respect to a question.
S(Q, A) =D(Q, A)∧P(Q, A)∧Sp(Q, A)∧ ¬V(A)(1)
Q, A— the question and the answer, respectively,
S(Q, A)— the binary output verdict,
D(Q, A)— the direct-answer criterion,
P(Q, A)— the completeness criterion,
Sp(Q, A)— the specificity criterion,
V(A)— the vagueness criterion,
SB ARC=PN
i=1S(Q, A i)
N(2)
N— the total number of pairs (question, answer) in the evaluation dataset.
5.2. Strict Binary Context Recall
SBCRevaluates whether all relevant contextual information is utilized in the answer.
U(C, A) =1 
Fused≡F relevant
(3)
C, A— the context and the answer, respectively,
U(C, A)— the binary output verdict,
Frelevant — the set of all relevant facts present in the context,
Fused— the set of facts from the context that are used in the answer,
SB CR=NX
i=1U(C i, Ai)
N(4)
N— the total number of pairs (question, answer) in the evaluation dataset.
7

5.3. Strict Binary Factual Accuracy
SBFAevaluates factual correctness and hallucination absence.
F(C, A) =1(S(A, C)∧ ¬C(A, C)∧ ¬H(A, C))(5)
C, A— the context and the answer, respectively,
F(C, A)— the binary output verdict,
Supported(C, A)— all statements in the answer are explicitly present in or directly entailed by the context,
¬Contradicts(A, C)— the answer does not contradict the information in the context,
¬Hallucinates(A, C)— the answer does not contain information that is absent from the context,
SB FA=PN
i=1F(C i, Ai)
N(6)
N— the total number of pairs (question, answer) in the evaluation dataset.
5.4. Retrieval Coverage Score (RCS)
RCS measures how completely the retrieved context reproduces the original (reference) context and therefore characterizes
the quality of the retriever.
LetCorigbe the original context and Cretrthe context retrieved by the retriever. The evaluation function S(C orig, Cretr)is
defined as follows:
S(C orig, Cretr) =

0,ifCov(C orig, Cretr) =none
1,ifCov(C orig, Cretr) =partial
2,ifCov(C orig, Cretr) =complete(7)
Here, Coverage(S(C orig, Cretr))is either an expert assessment or the result of an LLM-based evaluator that compares the
two texts.
S= 0— the retrieved contextC retrdoes not contain information from the original contextC orig,
S= 1— the retrieved contextC retrcontains part of the information fromC orig, but not all of it,
S= 2— all information from the original contextC retris present in the retrieved contextC orig.
The finalRCSvalue for a dataset is computed as the arithmetic mean:
RCS=PN
i=1S(Ci
orig, Ci
retr)
N(8)
WhereNis the total number of queries in the evaluation dataset.
6. Experiment
The goal of the experiment was to evaluate the effectiveness of the DCD approach compared to a baseline Naive RAG
pipeline. The research process consisted of five sequential stages:
1. Generation of a text dataset
2. Generation of evaluation data (question–answer–context),
3. Construction of a vector database,
8

4. Inference with DCD and Naive RAG pipelines,
5. Metric computation.
At the first stage, a synthetic text dataset was generated. Using the language model gpt-oss-120b and a set of predefined
templates, we synthesized texts describing different domains. Ten residential complexes were used as domains. Within
each domain, several document collections were created corresponding to different sections, such as infrastructure, security,
and complex description. Dataset variability was achieved via template-based generation of semantically aligned prompt
instances, where numerical parameters and variables were systematically varied while maintaining the same underlying
reasoning pattern. This approach allowed us to generate a document corpus with a controlled semantic structure.
At the second stage, question–answer pairs were generated for the resulting document corpus. Using the language model
gpt-oss-120b, relevant questions, answers, and the supporting context were synthesized for each document. The resulting
question–answer–context dataset served as the reference set for evaluating the retrieval and generation pipelines.
The third stage involved constructing a vector database. ChromaDB was used as the storage solution. Documents were split
using a window-based splitter with a fixed window size of 300 tokens and an overlap of 20%. Chunk embeddings were
generated using the bge-m3 embedding model.
At the fourth stage, both approaches were evaluated. For each query from the evaluation dataset, the system retrieved
relevant context and generated an answer using an LLM. Query embeddings were computed using bge-m3, while answer
generation was performed by qwen2.5-7b-instruct. This procedure was executed separately for the DCD pipeline and the
Naive RAG pipeline.
Finally, we computed the metrics described in Section 5.
Table 1.Experimental results
Metrics DCD Naive
Strict Binary Answer Relevance 0.86 0.87
Strict Binary Context Recall 0.95 0.59
Strict Binary Factual Accuracy 0.89 0.40
Retrieval Coverage Score 1.76 1.28
Figure 4.Model performance
In terms of answer generation quality, both approaches demonstrated comparable results. At the same time, the evaluation of
the retriever shows a clear advantage for DCD: the proposed approach provides more accurate retrieval of relevant context.
On a dataset with a template-based document structure, DCD significantly outperforms Naive RAG in retrieving relevant
information.
7. Limitations
7.1. Configuration Complexity
The proposed DCD approach introduces additional configuration complexity as the size and heterogeneity of the knowledge
base increase. The difficulty of maintaining a correct domain segmentation grows proportionally with the number of
semantically disconnected knowledge areas [Yao et al., 2023], a trade-off commonly observed in systems relying on explicit
reasoning structure and controlled search spaces.
9

7.2. Fully Unstructured Data
DCD assumes the presence of at least minimal semantic organization within the knowledge corpus. When applied to fully
unstructured data, the effectiveness of structured retrieval and routing mechanisms degrades, often converging toward the
behavior of advanced naive RAG pipelines [Gao et al., 2021].
7.3. Computational Overhead
It is important to consider the specifics of the generated dataset: all documents have a template-based structure in which
texts within each domain are almost identical and differ only in specific values such as names and numerical characteristics,
while preserving the same overall syntactic and semantic structure. [Shazeer et al., 2017].
For this type of data, characterized by a high degree of contextual overlap, DCD predictably demonstrates higher retrieval
quality compared to the naive RAG approach.
7.4. Evaluation Methodology
Answer quality in this study was assessed using an LLM-as-a-Judge setup with the gpt-oss-120b model. The judge evaluated
generated answers according to the evaluation metrics defined in Section 5, receiving as input the user query, the retrieved
context, and the generated answer. The evaluation prompts instructed the model to assess responses according to the metric
definitions and to produce a structured justification before assigning the final decision. The temperature for these evaluations
was set to 0 to ensure deterministic scoring.
Retrieval quality was evaluated separately using the Retrieval Coverage Score. In this setting, the judge compared the
retrieved context with the reference context and assigned a score on a three-point scale from 0 to 2, reflecting whether
the retrieved information was absent, partially sufficient, or fully matched the reference context. For this evaluation, the
temperature was set to 0.1 to allow limited variability in explanatory reasoning.
8. Conclusion
We introduced DCD, a domain-oriented RAG design based on explicit knowledge hierarchies and controlled multi-stage
workflows. Experiments on production data demonstrate consistent quality improvements for heterogeneous corpora and
multi-step queries at a predictable computational cost. Future work includes replacing general-purpose LLMs in routing
tasks with specialized lightweight models and integrating backward navigation mechanisms with contextual relevance
validation.
9. Resources
Dataset: Huggin Face
Code repository: GitHub
Both resources are maintained by the AI R&D team at red_mad_robot.
10. References
Lewis, P., Perez, E., Piktus, A., et al.Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in
Neural Information Processing Systems (NeurIPS), 2020.
Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M.-W.REALM: Retrieval-Augmented Language Model Pre-Training.
International Conference on Machine Learning (ICML), 2020.
Izacard, G., Grave, E. LeveragingPassage Retrieval with Generative Models for Open Domain Question Answering.
European Chapter of the ACL (EACL), 2021.
Khot, T. Khot, T., Sabharwal, A., Clark, P.Decomposed Prompting: A Modular Approach for Solving Complex Tasks.
International Conference on Learning Representations (ICLR), 2023.
Wei, J., Wang, X., Schuurmans, D., et al.Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
10

NeurIPS, 2022.
Yao, S., Zhao, J., Yu, D., et al.Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv preprint
arXiv:2305.10601, 2023.
Shazeer, N., Mirhoseini, A., Maziarz, K., et al.Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-
Experts Layer. ICLR, 2017.
OpenAIStructured Outputs in Language Models. Technical report / documentation, 2024.
Gao, L., Ma, X., Lin, J., Callan, J.Precise Zero-Shot Dense Retrieval without Relevance Labels. ACL, 2021.
Zheng, L., Chiang, W.-L., Sheng, Y ., et al.Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS, 2023.
Ji, Z., Lee, N., Frieske, R., et al. Survey of Hallucination in Natural Language Generation. ACM Computing Surveys, 2023.
Li, Y ., Weizhi Zhang, Yuyao Yang, et al. Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning
Systems in LLMs. arXiv preprint, 2025.
Liu, Y . Iter, D., et al. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. arXiv preprint arXiv:2303.16634,
2023.
Liu, Y . Lin, S., Hilton, J., Evans, O.TruthfulQA: Measuring How Models Mimic Human Falsehoods. ACL, 2022.
Manakul, P., Liusie, A., Gales, M. Manakul, P.. EMNLP, 2023.
Makin, A.Ontology-Driven Knowledge Management Systems Enhanced by Large Language Models. ResearchGate preprint,
2024.
Nair, I. Somasundaram, S., Saxena, A., Goswami, K. Drilling Down into the Discourse Structure with LLMs for Long
Document Question Answering. arXiv preprint, 2023.
11