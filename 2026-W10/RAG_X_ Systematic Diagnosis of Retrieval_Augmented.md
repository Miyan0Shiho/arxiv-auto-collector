# RAG-X: Systematic Diagnosis of Retrieval-Augmented Generation for Medical Question Answering

**Authors**: Aswini Sivakumar, Vijayan Sugumaran, Yao Qiang

**Published**: 2026-03-03 21:55:42

**PDF URL**: [https://arxiv.org/pdf/2603.03541v1](https://arxiv.org/pdf/2603.03541v1)

## Abstract
Automated question-answering (QA) systems increasingly rely on retrieval-augmented generation (RAG) to ground large language models (LLMs) in authoritative medical knowledge, ensuring clinical accuracy and patient safety in Artificial Intelligence (AI) applications for healthcare. Despite progress in RAG evaluation, current benchmarks focus only on simple multiple-choice QA tasks and employ metrics that poorly capture the semantic precision required for complex QA tasks. These approaches fail to diagnose whether an error stems from faulty retrieval or flawed generation, limiting developers from performing targeted improvement. To address this gap, we propose RAG-X, a diagnostic framework that evaluates the retriever and generator independently across a triad of QA tasks: information extraction, short-answer generation, and multiple-choice question (MCQ) answering. RAG-X introduces Context Utilization Efficiency (CUE) metrics to disaggregate system success into interpretable quadrants, isolating verified grounding from deceptive accuracy. Our experiments reveal an ``Accuracy Fallacy", where a 14\% gap separates perceived system success from evidence-based grounding. By surfacing hidden failure modes, RAG-X offers the diagnostic transparency needed for safe and verifiable clinical RAG systems.

## Full Text


<!-- PDF content starts -->

RAG-X: Systematic Diagnosis of
Retrieval-Augmented Generation for Medical
Question Answering
Aswini Sivakumar, Vijayan Sugumaran, Yao Qiang
Oakland University, Rochester, MI, USA
Email:{aswinisivakumar, sugumara, qiang}@oakland.edu
Abstract—Automated question-answering (QA) systems in-
creasingly rely on retrieval-augmented generation (RAG) to
ground large language models (LLMs) in authoritative medical
knowledge, ensuring clinical accuracy and patient safety in
Artificial Intelligence (AI) applications for healthcare. Despite
progress in RAG evaluation, current benchmarks focus only
on simple multiple-choice QA tasks and employ metrics that
poorly capture the semantic precision required for complex QA
tasks. These approaches fail to diagnose whether an error stems
from faulty retrieval or flawed generation, limiting developers
from performing targeted improvement. To address this gap,
we propose RAG-X, a diagnostic framework that evaluates
the retriever and generator independently across a triad of
QA tasks: information extraction, short-answer generation, and
multiple-choice question (MCQ) answering. RAG-X introduces
Context Utilization Efficiency (CUE) metrics to disaggregate
system success into interpretable quadrants, isolating verified
grounding from deceptive accuracy. Our experiments reveal an
“Accuracy Fallacy”, where a 14% gap separates perceived system
success from evidence-based grounding. By surfacing hidden
failure modes, RAG-X offers the diagnostic transparency needed
for safe and verifiable clinical RAG systems.
Index Terms—Large Language Models, Retrieval Augmented
Generation, Natural Language Processing, Evaluation Frame-
works, Healthcare AI.
I. INTRODUCTION
The reasoning capabilities of Large Language Models (LLMs)
have spurred their rapid integration into healthcare applica-
tions, such as clinical decision support [1], automated medical
note generation [2], and medical education [3]. However, the
safe deployment of standalone LLMs is constrained by well-
known risks such as hallucinations [4], outdated knowledge
[5], and a lack of source verifiability [6].
Recently, Retrieval Augmented Generation (RAG) has
emerged as the standard architecture to mitigate these issues
by grounding LLMs’ responses in authoritative, up-to-date
knowledge [7]. The RAG system effectively separates the
task into two complementary components: a retriever, which
searches external knowledge sources to gather relevant context,
and a generator (the LLM), which leverages this context to
produce an informed, factually accurate response. However,
unlocking its full potential requires rigorous evaluation and
systematic diagnosis of each component, ensuring that the
system operates reliably and safely in critical medical domains.Although the evaluation of RAG systems is steadily ad-
vancing, existing frameworks suffer from a “diagnostic” gap.
Most prominent approaches rely on aggregate metrics such as
accuracy or F1, which provide a high-level snapshot of system
performance but fail to reveal the underlying causes of failure.
As a result, developers cannot determine whether a wrong
answer stems from a retriever that returned irrelevant context
or a generator that misinterpreted relevant information. This
limitation exposes a critical research gap in the medical do-
main:the absence of evaluation frameworks that rigorously
assess RAG systems on diverse, clinically realistic tasks
while providing actionable, component-level diagnostics.
To bridge this gap, we present RAG-X, a diagnostic frame-
work that disaggregates system success into interpretable
failure modes. Unlike conventional evaluations, RAG-X pro-
vides a multi-dimensional analysis of the retrieval-generation
interface, as shown in Figure 1. For retrieval, RAG-X moves
beyond standard ranking metrics, such as Mean Average
Precision, to reveal structural inefficiencies. For instance, in
our evaluation of a best-performing RAG pipeline, RAG-X
identified that over one-fifth (22%) of the retrieved evidence
was redundant, effectively wasting the model’s limited context
window despite having an adequate recall rate.
The core innovation of RAG-X lies in itsContext Uti-
lization Efficiency (CUE)metrics, which synchronize re-
triever success with generator behavior to expose theAccuracy
Fallacy, a scenario where the system appears to be highly
accurate but is actually ungrounded. CUE isolates critical
attribution errors in which a model’s high adherence score
masks a lack of true source grounding. In our experiments,
RAG-X uncovered a 14% gap between overall accuracy and
actual evidence-based grounding, showing more than one-third
(33.9%) of the responses were ungrounded “lucky guesses”.
These insights are essential for determining whether a system
is genuinely following clinical evidence or merely appearing
correct without verifiable support.
We state our primary contributions as follows:
•RAG-X: A Unified Diagnostic Framework for RAG
Systems.We introduce a multi-dimensional evalua-
tion framework that replaces aggregate metrics with
component-level diagnostics tailored for the high-
precision requirements of medical QA.arXiv:2603.03541v1  [cs.CL]  3 Mar 2026

Fig. 1: Overview of RAG System and RAG-X Diagnostics. This figure illustrates the workflow of an RAG system, where
external knowledge is indexed into a vector database. A user query interacts with the retriever to fetch relevant information,
which is then passed to a generator to produce a response. The RAG-X framework adds diagnostic modules for both retrieval
(e.g., ranking, semantic relevance, fine-grained evaluation) and generation (e.g., surface-level similarity, structured output,
semantic similarity, and LLM-based judgment) to provide detailed performance analysis.
•Context Utilization Efficiency (CUE).Our framework
systematically categorizes RAG outputs into diagnos-
tic quadrants, such as Effective Use and Hallucina-
tion, isolating grounded successes from deceptive “lucky
guesses”.
•Attribution Error Identification.We provide a system-
atic method to identify theAdherence Paradox, demon-
strating how RAG-X exposes cases where high adherence
scores mask a fundamental lack of source grounding.
•Comprehensive Empirical Study.We evaluate RAG-
X across three clinical datasets, validating its ability to
identify actionable bottlenecks that standard Accuracy
and F1-scores obscure.
II. RELATEDWORK
A. Application of RAG in Medical Domain
Recently, meta-analyses [8] indicate that integrating retrieval
mechanisms yields a 1.35 odds ratio over baseline LLM
performance and can deliver accuracy improvements exceed-
ing 18% using advanced prompting alone. These gains have
motivated the adoption of RAG systems in diverse health-
care applications, such as clinical decision support [1; 9],
automated medical note generation [2], medical QA [10],
and medical education [3]. However, medical applications
pose domain-specific challenges that differ significantly from
those in general LLM applications. Medical information is
inherently heterogeneous, encompassing unstructured clini-
cal narratives [11], semi-structured EHRs [12], and highly
structured guideline repositories [13]. Clinical queries extend
beyond basic fact retrieval and require ”multi-hop/multi-step”
reasoning [14]. This involves synthesizing information from
multiple documents to reach a conclusion. Moreover, factualaccuracy and source verifiability are non-negotiable due to the
life-critical nature of clinical decisions [15]. Errors arising
from incomplete retrieval or hallucination can result in se-
vere patient harm [16], underscoring the need for specialized
evaluation methods that assess robustness under these domain-
specific constraints [6].
B. Evaluation of RAG in Medical Domain
The need for robust evaluation has spurred the development of
several RAG benchmarks. In the general domain, RAGAS [17]
has established a foundational reference-free approach to score
answer faithfulness, context relevance, and retrieval precision
using carefully designed heuristic prompts. However, these
heuristic scoring prompts often fail to adapt when applied
to new domains or corpora. To mitigate this, ARES [18]
introduced a pipeline for fine-tuning lightweight LLM judges
on synthetic data and applying Prediction-Powered Inference
(PPI) to derive statistically reliable confidence intervals for its
evaluation metrics. While this framework evaluates the RAG
components separately, each cross-domain shift requires in-
domain passages and few-shot query examples to reconfigure
ARES’s judges. This limits its direct applicability to special-
ized domains or information-extraction tasks, underscoring the
need for domain-specific evaluation frameworks.
In the medical domain, frameworks such as MIRAGE
[10] benchmark RAG systems against clinical QA datasets,
focusing on the impact of different retrieval modules on target
accuracy. MedRGB [19] extended this to evaluate informa-
tion sufficiency, multi-document integration, and robustness
to factual errors. Although these frameworks mark significant
progress, they still face two major limitations. First, most ex-
isting benchmarks, such as MIRAGE and MedRGB, primarily

Capability RAGAS MIRAGE MedRGB RAG-X
Primary Focus General Medical Medical Medical
Metric Engine Heuristic
PromptsAccuracy Prompt
AnalysisMulti-level
Domain Alignment None None None Logic-
aware
Diagnostic Decoupling Limited Minimal Limited Full
Task Coverage Broad MCQ MCQ Broad
TABLE I: Comparison of capabilities across RAG evaluation
frameworks. RAG-X uniquely incorporates independent com-
ponent analysis and medical logic normalization, capabilities
absent from existing benchmarks.
focus on multiple-choice or factoid-style QA. These rely on
simple accuracy-based metrics and fail to capture the seman-
tic precision and reasoning adequacy required for complex
medical tasks such as guideline recommendation or evidence
synthesis. Second, while general evaluation frameworks, such
as ARES and RAGAS, provide component-level scores, they
lack diagnostic granularity to identify why a retrieval failed.
To address these evaluation issues, we propose RAG-X,
the first diagnostic framework tailored to medical RAG sys-
tems. Unlike prior benchmarks, RAG-X decouples retrieval
and generation performance, enabling fine-grained error attri-
bution. Furthermore, it introduces multi-dimensional metrics
that evaluate coverage, redundancy, faithfulness, and reason-
ing adequacy, offering actionable insights beyond a single
score. Table I compares RAG-X with existing frameworks,
highlighting its unique ability to provide component-level
diagnostics essential for safe and effective deployment of RAG
in healthcare settings.
III. METHOD
A. RAG Pipeline and Medical Normalization
The RAG pipeline evaluated in this study follows the standard
three-stage architecture (indexing, retrieval, and generation)
augmented with a domain-specific normalization layer.
– Indexingtransforms the structured text chunks from raw
documents into high-dimensional embeddings using a pre-
trained embedding model. The embeddings are indexed using
the Hierarchical Navigable Small World Indexing algorithm
and stored in a vector database. This indexing method enables
data to be stored in a multi-layer structure, making approxi-
mate nearest neighbor searches fast and efficient.
– Retrievalsearches for the most relevant segments from the
vector database based on their proximity to the query vector.
We employ a hybrid search strategy that integrates traditional
lexical matching with semantic similarity search using the
reciprocal rank fusion (RRF) algorithm. The contribution of
the sparse and dense retrieval is controlled by a weighting
parameterα. This approach is particularly important in the
medical domain, where effective retrieval must capture both
rigid medical terminology match and the nuanced, conceptual
nature of the clinical queries [20].
– Generationintegrates the user query with the retrieved
evidence to produce a response that is both factually grounded
and contextually aligned. This is achieved by constraining themodel’s reasoning to the provided evidence rather than relying
solely on its internal knowledge.
– Medical Normalization Pre-processing:We applied a
specialized normalization process to the retrieved contexts,
generated responses, and ground truths to ensure robust evalua-
tion. This includes medical abbreviations to full terms mapping
(e.g., “AAA” to “abdominal aortic aneurysm”), standardizing
age-related thresholds, and unifying gender-specific phrasing.
B. RAG-X Approach
Traditional RAG evaluations often fall short in three key
areas: (1) distinguishing component-level accountability (re-
triever versus generator errors), (2) incorporating semantic and
human-aligned measures of relevance and factuality, and (3)
providing transparency into retrieval failure modes such as
coverage gaps and redundancy. RAG-X addresses these limi-
tations through a comprehensive diagnostic framework that in-
tegrates quantitative information retrieval metrics, embedding-
based semantic similarity, and LLM-as-a-judge evaluations.
Specifically, RAG-X organizes metrics into three primary cat-
egories: retrieval diagnostics, generation quality, and context
utilization efficiency to support fine-grained, component-level
interpretability and systematic failure-mode analysis across the
entire RAG pipeline.
– Retrieval Diagnostic Metrics:Retrieval quality fundamen-
tally constrains the overall performance of any RAG system,
as the generator cannot produce accurate or comprehensive
answers without access to relevant evidence. Consequently,
robust evaluation of the retriever is essential for diagnosing
pipeline bottlenecks and guiding system improvements. To
achieve this, we categorize our retrieval diagnostic metrics into
three complementary groups:
(1)Ranking-based Metrics, such as Recall@k, Mean Average
Precision (MAP), Mean Reciprocal Rank (MRR), and Nor-
malized Discounted Cumulative Gain (nDCG) to evaluate the
ranking quality of the retrieved contexts and measure how
effectively the retriever surfaces relevant evidence.
(2)LLM-based Context Relevancycomplements traditional
ranking metrics by leveraging an LLM-as-a-judge to assign a
context relevancy score (0.0 - 1.0) to each retrieved segment,
assessing how well that context can potentially answer the
question [21].
(3)Fine-grained Retrieval Diagnosticsoffer deeper insights
into specific retrieval failure modes. RAG-X introduces a set
of diagnostic metrics specifically designed to characterize the
underlying retrieval dynamics and evidence distribution. Let
Qbe the set of all queries in the evaluation set. For each
queryq∈Q, letC q={c 1, c2, . . . , c k}be the ranked set of
kretrieved contexts andGT qbe the ground truth answer. We
define a cascading binary relevance functionR(c i, GT q)as:
R(c i, GT q) =(
1,ifc icontains correct answer forq,
0,otherwise.(1)
To compute this relevance signal robustly, we implement a
multi-level context-checking approach that checks for exact
substring matches, followed by token-level overlap (≥0.80),

and finally sentence-level semantic similarity using embed-
dings (≥0.75). Using this function, we derive the following
diagnostic metrics:
•Context-kHit Rate measures the percentage of queries
where the ground truth appears in thek-th retrieved doc-
ument, enabling analysis of retrieval performance across
different ranks.
•No-Hit Rate captures the percentage of queries for which
none of the retrieved contexts contain the ground truth,
providing a direct measure of retrieval coverage failures.
A high rate signals a critical weakness in the retriever’s
ability to locate relevant information within the corpus.
•Exclusive Hit Rate reports the percentage of queries
where the ground truth appears in only a single retrieved
context, indicating whether the retriever is providing
unique sources of evidence or redundant variations of the
same information.
– Generation Quality Metricsultimately determines the
usability and trustworthiness of a RAG system. Even when
relevant evidence is retrieved, the generator must accurately
synthesize this information to produce responses that are
both correct and contextually grounded. Poor generation can
undermine the benefits of high-quality retrieval, making it
essential to evaluate this component thoroughly. To capture the
multifaceted nature of generation quality, RAG-X organizes its
generation metrics into three complementary dimensions:
(1)Surface-Level Similaritymeasures lexical overlap to assess
how closely the generated answer matches the reference text at
the word and phrase level. Common metrics in this category
include Exact Match, Fuzzy Match, ROUGE-L, and Token-
level F1 (see Appendix Table??). While useful for evaluating
completeness, these metrics cannot capture paraphrasing or
deeper semantic equivalence.
(2)Semantic Similaritydetects whether the generated re-
sponses preserve the intended meaning of the reference text
beyond superficial word overlap. To achieve this, we com-
pute cosine similarity between sentence embeddings of the
normalized answers, providing a semantic-level measure of
alignment.
(3)Structured Outputevaluates the correctness of answers that
require enumerating multiple elements, such as lists of risk
factors or symptoms. Traditional token-level metrics often fall
short in these scenarios, as they do not account for the structure
and order of itemized responses. To address this, we introduce
a List-Component F1-score, which splits the generated and
reference responses into individual items and computes an
F1-score based on their overlap, ensuring fair and accurate
assessment of list-based outputs.
(4)LLM-Judgmentleverages an LLM to evaluate two crit-
ical dimensions of the generated answer: answer relevancy,
whether the response directly and completely addresses the
user’s query, and context adherence, whether the response
is factually grounded in the retrieved evidence [22]. This
approach offers a human-aligned perspective on accuracy and
factual consistency, serving as an essential complement to
traditional numerical metrics.– Context Utilization Efficiencyframework synthesizes re-
triever and generator performance to assess how effectively
the system utilizes retrieved evidence. By cross-referencing
retrieval success determined through the relevance function
with the generator’s context adherence score (threshold≥0.7),
we categorize each question into four diagnostic quadrants: (1)
Effective use occurs when the retriever successfully found the
answer, and the generator correctly utilized it. (2) Information
blindness describes cases where the retriever found the answer,
but the generator failed to utilize the provided context. (3)
Hallucination (Lucky Guess) refers to situations where the
retriever failed to find the answer, but the generator produced
a correct response by claiming adherence to irrelevant context
or relying on internal weights. (4) Correct rejection captures
instances where the retriever failed to find the answer, and the
generator correctly exhibited low adherence to the irrelevant
context.
IV. EXPERIMENTSETTINGS
We demonstrate the diagnostic capabilities of RAG-X by: (1)
evaluating various RAG configurations across diverse medical
QA tasks; (2) comparing against non-RAG baselines; (3)
performing ablations to validate pipeline configurations
Datasets and Knowledge Bases:We evaluate RAG-X on
three medical QA benchmarks, each representing a distinct
modality and paired with a domain-specific retrieval cor-
pus.PubMedQAprovides 1,000 labeled multiple-choice ques-
tions (yes/no/maybe) grounded in biomedical abstracts [23].
GuidelineQAincludes 59 expert-curated questions targeting
information extraction from a corpus of 11 comprehensive
United States Preventive Services Task Force (USPSTF) clini-
cal guideline documents[11]. This dataset emphasizes high-
precision retrieval and extraction.MedQuAD-GHRcontains
500 QA pairs drawn from MedQuAD [24] evaluated against
1,084 genetic condition files scraped from MedlinePlus Ge-
netics Home Reference (GHR).
Baselines:To quantify the effectiveness of the retrieval step,
we compare RAG against two non-retrieval baselines: (1)
Direct Zero-shot(prompts question only), testing the model’s
parametric knowledge; (2)Long-Context Zero-shot(prompts
question plus the entire document), evaluating the model’s
ability to perform in-context learning.
RAG Pipeline Configuration:
– Indexing: Documents were segmented into chunks of 1,024
tokens with an overlap of 100 tokens and stored in a Weaviate
[25] vector database collection. We evaluated three embed-
ding models: Qwen3-Embedding-8B [26], MedCPT-Article-
Encoder [27], and NV-EmbedQA-E5-v5 [28].
– Retrieval: We employed a hybrid search strategy that inte-
grated BM25 for lexical matching with vector-based similarity
search using RRF. For all experiments, we retrieved the top
k= 3context chunks and varied the weighting parameterα.
– Generation: We utilized three instruction-tuned models for
response generation: Llama-3.1-8B-Instruct [29], gemma-2-
9b-it [30], and Qwen2.5-7B-Instruct [31]. For RAG-X metrics,

TABLE II: Performance comparison of backbone LLMs across
different settings on the GuidelineQA dataset. We report
Token-level F1-Scoreto measure lexical precision and recall,
Semantic Similarityto capture correctness in meaning, and
Accuracyfor the fraction of correct answers. An answer is
considered accurate if it matches the ground truth using at
least one of the following criteria: (1) Exact Match, (2) Fuzzy
Match, (3) List-Component F1 score≥0.7, or (4) Semantic
Similarity score≥0.7
LLM Setting F1-Score Semantic Similarity Accuracy
Direct Zero-Shot 0.21 0.52 0.35
Llama-3.1-8B-Instruct Long-Context0.340.63 0.54
RAG 0.320.69 0.71
Direct Zero-Shot 0.16 0.50 0.33
gemma-2-9b-it Long-Context 0.51 0.70 0.61
RAG0.56 0.75 0.69
Direct Zero-Shot 0.31 0.60 0.50
Qwen2.5-7B-Instruct Long-Context 0.27 0.52 0.47
RAG0.44 0.68 0.54
all-MiniLM-L6-v2 [32] sentence-transformers model was em-
ployed to encode semantic similarity, and DeepSeek-V3.1 [33]
served as the LLM-judge for all qualitative evaluations.
V. RESULTS ANDDISCUSSIONS
A. Comparison of Backbone LLMs
We first evaluated three backbone LLMs, i.e., Llama-3.1-
8B-Instruct, gemma-2-9b-it, and Qwen2.5-7B-Instruct, across
three settings: Direct Zero-Shot, Long-Context, and RAG. The
results, summarized in Table II, demonstrate the definitive
superiority of the RAG architecture. In nearly every instance,
the RAG configuration outperforms the non-RAG baselines
across the primary evaluation metrics.
These results highlight that accuracy alone is insufficient for
RAG evaluation, particularly in complex information extrac-
tion tasks. On the GuidelineQA dataset, Llama-3.1-8B-Instruct
achieves its highest accuracy (0.71) in the RAG setting, but
its highest lexical precision (0.34) actually occurs in the
Long-Context setting. This mismatch shows that standard F1-
scores overly penalize minor phrasing differences, while our
comprehensive list of metrics provides a more nuanced view
of model behavior. In particular, our fuzzy match and List-
Component F1 score capture all the essential clinical elements
that a token-level F1 score may miss. By breaking down these
metrics, RAG-X measures the information extraction ability
of the model that the aggregate accuracy would otherwise
overlook.
Finally, we observe a clear performance pattern across
the generators. In the RAG setting, gemma-2-9b-it achieves
the best generation quality with an F1-score of 0.56 and a
semantic similarity of 0.75, reflecting its strong ability to
synthesize information when guided by targeted context. In
contrast, Qwen2.5-7B-Instruct struggles in the Long-Context
setting, where its F1-score (0.27) and semantic similarity
(0.52) dropped below its Direct Zero-shot performance. This
“lost-in-the-middle” behavior underscores the role of a re-
triever as it transforms the challenging information extractionfrom long clinical guidelines into a manageable grounded-
generation task.
B. Comparison of Retrieval Models
To isolate the impact of the retriever, we evaluated the
three retrieval models across three datasets, as summarized in
Table III. While Qwen3-Embedding-8B achieves the highest
MRR and Recall, especially on GuidelineQA and MedQuAD-
GHR, RAG-X’s decoupled metrics expose important qualita-
tive differences in information density and retrieval risk.
A key signal comes from theExclusive Hit Rate
(EHR@1). On GuidelineQA, Qwen3 reaches its highest recall
atα= 1, yet EHR@1 drops to 0.1, showing high information
redundancy where the answer appears across multiple chunks.
In contrast, MedQuAD-GHR shows higher EHR@1 values (up
to 0.30), indicating a “single-source-of-truth” pattern where
the generator’s success is entirely reliant on its ability to attend
to the first retrieved evidence.
RAG-X also explains the “all-or-nothing” behavior in
domain-specific retrievers. On PubMedQA, MedCPT achieves
the best Context Relevancy and EHR@1 despite lower recall.
This confirms that, when successful, it retrieves uniquely infor-
mative evidence. Finally, the drop in Recall and EHR@1 for
NV-EmbedQA-E5-v5 when moving from hybrid to pure dense
search on GuidelineQA shows the importance of retaining
lexical matching for clinically rigid terminology.
C. Case Study: Diagnosing RAG Pipeline with RAG-X
In Table IV, we demonstrate RAG-X’s value using our
best-performing setup (Llama-3.1-8B-Instruct + Qwen3-
Embedding-8B atα= 1), which achieved the top accuracy
(0.71) and MAP (0.44) on GuidelineQA. Although these
metrics indicate strong performance, RAG-X’s fine-grained
diagnostics show that this success is partly misleading.
The Accuracy Fallacy (CUE Analysis):RAG-X shows that
much of the system’s apparent success is not truly grounded.
Although 49.2% of responses reflectEffective Use, a sub-
stantial 33.9% areLucky Guess, where the retriever misses
the answer but the model still generates a confident, correct-
looking response. Without RAG-X, this behavior would be
hidden under the overall Accuracy score.
Quantifying Retrieval Waste:Although standard Recall
(57.6%) suggests adequate coverage, RAG-X uncovers a major
inefficiency, with 22.0% Pairwise redundancy between the
top contexts and only 6.8% Exclusive Hit Rate at Rank 2.
Practically, the retriever is returning overlapping evidence
rather than complementary context, wasting retrieval capacity.
Diagnostic of Attribution Error:RAG-X reveals a 14%
gap between Accuracy (71%) and overall Context Hit Rate
(57.6%), showing that a few correct answers come from the
model’s parametric knowledge rather than retrieved evidence.
The high Context Adherence Score (0.84) then incorrectly
attributes these answers to the context, demonstrating that ad-
herence alone can give a false sense of grounding highlighting
the critical role of CUE metric in making these errors visible.

TABLE III: Comprehensive retrieval performance across all three datasets and retriever configurations. Each block corresponds
to a retriever model, with rows indicating the hybrid search weight (α). We report four key retrieval metrics: MRR (Mean
Reciprocal Rank), Recall@3, Ctx Rel. (LLM-judged Context Relevancy score), and EHR@1 (Exclusive hit rate for Context1).
RetrieverαPubMedQA GuidelineQA MedQuad-GHR
MRR Recall Ctx Rel. EHR@1 MRR Recall Ctx Rel. EHR@1 MRR Recall Ctx Rel. EHR@1
BM25 0 0.31 0.34 0.44 0.015 0.32 0.41 0.39 0.14 0.65 0.77 0.30 0.19
Qwen3-Embedding-8B0.5 0.31 0.34 0.51 0.02 0.43 0.54 0.49 0.14 0.72 0.81 0.31 0.27
1.0 0.31 0.34 0.52 0.021 0.45 0.58 0.52 0.10 0.70 0.79 0.30 0.30
NV-EmbedQA-E5-v50.5 0.31 0.34 0.49 0.016 0.39 0.54 0.42 0.14 0.66 0.79 0.31 0.20
1.0 0.31 0.34 0.49 0.018 0.28 0.41 0.36 0.03 0.62 0.74 0.30 0.22
MedCPT-Query-Encoder0.5 0.24 0.31 0.55 0.034 0.27 0.34 0.35 0.08 0.40 0.54 0.21 0.10
1.0 0.23 0.30 0.50 0.033 0.26 0.31 0.32 0.10 0.30 0.39 0.18 0.11
TABLE IV: Fine-grained diagnostics from our RAG-X framework with suggested actions for the best-performing Llama-3.1-
8B-Instruct + Qwen3-Embedding-8B atα= 1pipeline (MAP = 0.44) on GuidelineQA dataset.
Diagnostic Metric Value Interpretation Actionable Insight
Effective Use (CUE) 49.2% The Only truly grounded success System is reliable for nearly half of the queries.
Lucky Guess (CUE) 33.9% High risk; Generator produces answers without evidence
support.Improve retriever coverage.
Information Blindness (CUE) 8.5% Retriever was successful, but the generator missed the an-
swerRefine generation prompt for extraction.
Pairwise Redundancy (C1 and C2) 22.0% Significant overlap between top ranks. Implement Maximum Marginal Relevance (MMR) for di-
versity.
Exclusive Hit Rate (Context-2) 6.8% Rank 2 rarely provides unique information. Implement diverse re-ranking.
Context Adherence (LLM Judgment) 0.84 Generator is highly faithful to context. No Action
VI. CONCLUSION
The safe integration of AI in high-stakes medical domains de-
pend on RAG systems that are not only capable but verifiably
grounded. This work introduced RAG-X, a diagnostic frame-
work designed to move beyond aggregate accuracy by offering
a component-level X-ray of the RAG pipeline. By breaking
down the system behavior through the CUE framework, we
show that high accuracy can be misleading, often masking
grounding failures and deceptive lucky guesses that standard
benchmarks overlook. As RAG systems become increasingly
integrated into clinical workflows, the diagnostic transparency
offered by RAG-X becomes critical for ensuring safety and
trustworthiness. This work provides the targeted validation
needed to transform LLMs from general-purpose assistants
to reliable, evidence-based clinical tools, thereby setting a
stronger standard for responsible medical AI systems.
REFERENCES
[1] F. Gaber, M. Shaik, F. Allega, A. J. Bilecz, F. Busch,
K. Goonet al., “Evaluating large language model work-
flows in clinical decision support for triage and referral
and diagnosis,”npj Digital Medicine, vol. 8, p. 263, 2025.
[2] H. Wang, C. Gao, B. Liu, Q. Xu, G. Hussein, M. E.
Labbanet al., “Towards adapting open-source large lan-
guage models for expert-level clinical note generation,”
inFindings of the Association for Computational Lin-
guistics: ACL 2025, 2025, pp. 12 084–12 117.
[3] A. Abd-alrazaq, R. AlSaad, D. Alhuwail, A. Ahmed,
P. M. Healy, S. Latifiet al., “Large language models in
medical education: Opportunities, challenges, and futuredirections,”JMIR Medical Education, vol. 9, p. e48291,
2023.
[4] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y . Xuet al.,
“Survey of hallucination in natural language generation,”
ACM Comput. Surv., vol. 55, no. 12, 2023.
[5] J. Clusmann, F. R. Kolbinger, H. S. Muti, Z. I. Carrero,
J.-N. Eckardt, N. G. Lalehet al., “The future landscape
of large language models in medicine,”Communications
medicine, vol. 3, no. 1, p. 141, 2023.
[6] K. Wuet al., “How well do LLMs cite relevant medical
references? An evaluation framework and analyses,”
2024, arXiv:2402.02008 [cs.CL]. [Online]. Available:
https://arxiv.org/abs/2402.02008
[7] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyalet al., “Retrieval-augmented generation for
knowledge-intensive NLP tasks,” inProc. NeurIPS, 2020.
[8] S. Liu, A. B. McCoy, and A. Wright, “Improv-
ing large language model applications in biomedicine
with retrieval-augmented generation: a systematic re-
view, meta-analysis, and clinical development guide-
lines,”Journal of the American Medical Informatics
Association, vol. 32, no. 4, pp. 605–615, 2025.
[9] Y . Lu, X. Zhao, and J. Wang, “ClinicalRAG: enhancing
clinical decision support through heterogeneous knowl-
edge retrieval,” inProc. KnowLLM 2024, 2024, pp. 64–
68.
[10] G. Xiong, Q. Jin, Z. Lu, and A. Zhang, “Benchmarking
retrieval-augmented generation for medicine,” inFind-
ings of the Association for Computational Linguistics:
ACL 2024, 2024, pp. 6233–6251.
[11] X. Luo, F. M. Tahabi, T. Marc, L. A. Haunert, and

S. Storey, “Zero-shot learning to extract assessment cri-
teria and medical services from the preventive healthcare
guidelines using large language models,”Journal of
the American Medical Informatics Association, vol. 31,
no. 8, pp. 1743–1753, 2024.
[12] L. Li, J. Zhou, Z. Gao, W. Hua, L. Fan, H. Yuet al.,
“A scoping review of using large language models
(LLMs) to investigate electronic health records (EHRs),”
2024, arXiv:2405.03066 [cs.CL]. [Online]. Available:
https://arxiv.org/abs/2405.03066
[13] A. G. Levra, M. Gatti, R. Mene, D. Shiffer,
G. Costantino, M. Solbiatiet al., “A large language
model-based clinical decision support system for syncope
recognition in the emergency department: A framework
for clinical workflow integration,”European journal of
internal medicine, vol. 131, pp. 113–120, 2025.
[14] S. Krishna, K. Krishna, A. Mohananey, S. Schwarcz,
A. Stambler, S. Upadhyay, and M. Faruqui, “Fact, fetch,
and reason: A unified evaluation of retrieval-augmented
generation,” inProc. NAACL, 2025, pp. 4745–4759.
[15] O. K. Gargari and G. Habibi, “Enhancing medical AI
with retrieval-augmented generation: A mini narrative
review,”Digital Health, vol. 11, p. 20552076251337177,
2025.
[16] Y . Kimet al., “Medical hallucinations in
foundation models and their impact on healthcare,”
2025, arXiv:2503.05777 [cs.CL]. [Online]. Available:
https://arxiv.org/abs/2503.05777
[17] S. Es, J. James, L. Espinosa Anke, and S. Schockaert,
“RAGAs: Automated evaluation of retrieval augmented
generation,” inProc. EACL, 2024, pp. 150–158.
[18] J. Saad-Falcon, O. Khattab, C. Potts, and M. Za-
haria, “ARES: An Automated Evaluation Framework
for Retrieval-Augmented Generation Systems,” inProc.
NAACL, 2024.
[19] N. T. Ngo, C. V . Nguyen, F. Dernoncourt, and
T. H. Nguyen, “Comprehensive and practical evaluation
of retrieval-augmented generation systems for medical
question answering,” 2024, arXiv:2411.09213 [cs.CL].
[Online]. Available: https://arxiv.org/abs/2411.09213
[20] Y .-F. Luo, W. Sun, and A. Rumshisky, “A hybrid normal-
ization method for medical concepts in clinical narrative
using semantic matching,”AMIA Summits on Transla-
tional Science Proceedings, p. 732, 2019.
[21] L. Zheng, W.-L. Chiang, Y . Sheng, S. Zhuang, Z. Wu,
Y . Zhuanget al., “Judging llm-as-a-judge with mt-bench
and chatbot arena,”Advances in neural information pro-
cessing systems, vol. 36, pp. 46 595–46 623, 2023.
[22] D. Cohen, L. Burg, and G. Barkan, “RAGXplain: From
explainable evaluation to actionable guidance of rag
pipelines,” 2025, arXiv:2505.13538 [cs.IR]. [Online].
Available: https://arxiv.org/abs/2505.13538
[23] Q. Jin, B. Dhingra, Z. Liu, W. W. Cohen, and X. Lu,
“PubMedQA: A dataset for biomedical research question
answering,” inProc. EMNLP-IJCNLP, 2019, pp. 2567–
2577.[24] A. Ben Abacha and D. Demner-Fushman, “A question-
entailment approach to question answering,”BMC Bioin-
formatics, vol. 20, no. 1, p. 511, 2019.
[25] Weaviate, “Weaviate: An open-source vector search en-
gine,” https://weaviate.io, 2024, accessed: 2026-01-30.
[26] Y . Zhanget al., “Qwen3 embedding: Advancing text
embedding and reranking through foundation models,”
2025, arXiv:2506.05176 [cs.CL]. [Online]. Available:
https://arxiv.org/abs/2506.05176
[27] Q. Jinet al., “MedCPT: Contrastive pre-trained trans-
formers with large-scale pubmed search logs for zero-
shot biomedical information retrieval,”Bioinformatics,
vol. 39, no. 11, p. btad651, 2023.
[28] NVIDIA, “Nvidia nim: nv-embedqa-e5-v5,”
https://docs.api.nvidia.com/nim/reference/nvidia-nv-
embedqa-e5-v5, 2024, accessed: 2026-01-31.
[29] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle,
A. Letmanet al., “The Llama 3 herd of models,”
2024, arXiv:2407.21783 [cs.AI]. [Online]. Available:
https://arxiv.org/abs/2407.21783
[30] Google Research, “Gemma 2 9b-it model card,” https:
//huggingface.co/google/gemma-2-9b-it, 2024, [Online;
accessed Jan. 30, 2026].
[31] A. Yanget al., “Qwen2 technical report,” 2024,
arXiv:2407.10671 [cs.CL]. [Online]. Available: https:
//arxiv.org/abs/2407.10671
[32] S. Transformers, “Pretrained models — sentence
transformers documentation,” https://www.sbert.net/docs/
sentence transformer/pretrained models.html, 2024, ac-
cessed: 2026-01-31.
[33] DeepSeek-AI, “Deepseek-V3 technical report,” 2024,
arXiv:2412.19437 [cs.CL]. [Online]. Available: https:
//arxiv.org/abs/2412.19437