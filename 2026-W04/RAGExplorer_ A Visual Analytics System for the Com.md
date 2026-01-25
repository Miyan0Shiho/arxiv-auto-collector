# RAGExplorer: A Visual Analytics System for the Comparative Diagnosis of RAG Systems

**Authors**: Haoyu Tian, Yingchaojie Feng, Zhen Wen, Haoxuan Li, Minfeng Zhu, Wei Chen

**Published**: 2026-01-19 12:09:56

**PDF URL**: [https://arxiv.org/pdf/2601.12991v1](https://arxiv.org/pdf/2601.12991v1)

## Abstract
The advent of Retrieval-Augmented Generation (RAG) has significantly enhanced the ability of Large Language Models (LLMs) to produce factually accurate and up-to-date responses. However, the performance of a RAG system is not determined by a single component but emerges from a complex interplay of modular choices, such as embedding models and retrieval algorithms. This creates a vast and often opaque configuration space, making it challenging for developers to understand performance trade-offs and identify optimal designs. To address this challenge, we present RAGExplorer, a visual analytics system for the systematic comparison and diagnosis of RAG configurations. RAGExplorer guides users through a seamless macro-to-micro analytical workflow. Initially, it empowers developers to survey the performance landscape across numerous configurations, allowing for a high-level understanding of which design choices are most effective. For a deeper analysis, the system enables users to drill down into individual failure cases, investigate how differences in retrieved information contribute to errors, and interactively test hypotheses by manipulating the provided context to observe the resulting impact on the generated answer. We demonstrate the effectiveness of RAGExplorer through detailed case studies and user studies, validating its ability to empower developers in navigating the complex RAG design space. Our code and user guide are publicly available at https://github.com/Thymezzz/RAGExplorer.

## Full Text


<!-- PDF content starts -->

©2026 IEEE. This is the author’s version of the article that has been published in IEEE Transactions on Visualization and
Computer Graphics. The final version of this record is available at: xx.xxxx/TVCG.201x.xxxxxxx/
RAGExplorer: A Visual Analytics System for the Comparative
Diagnosis of RAG Systems
Haoyu Tian∗, Yingchaojie Feng∗†, Zhen Wen, Haoxuan Li, Minfeng Zhu†, and Wei Chen†
A
a
cdebBC
DQwen3-Embedding-8B
Qwen3-Reranker-8B
2000 tokensCorrect → Not in Context
Check the 
details of 
Test 1Check the 
details of 
ChunkSwitch to 
test modeSelect/remove a 
chunk in test 
modeQwen3-Embedding-0.6B
Qwen3-Reranker-8B
2000 tokens
Fig. 1: An overview of our visual analytic system. Our system integrates four coordinated views: Component Configuration View (A)
allows users to define an experimental space by selecting options for each RAG component. Performance Overview View (B) uses a
matrix and coordinated bar charts to display the ranked performance of all configurations, allowing users to identify top performers
and assess the average impact of each component choice. The Failure Attribution View (C) visualizes the flow of questions between
different failure attribution when comparing two RAG configurations, allowing users to see exactly how a design change impacts RAG’s
failure points. The Instance Diagnosis View (D) uses a three-panel layout, allowing users to select questions from a list (c), use a
central dual-track view to visually compare contexts and interactively test hypotheses (d), and inspect their full text in a details panel (e).
Abstract— The advent of Retrieval-Augmented Generation (RAG) has significantly enhanced the ability of Large Language Models
(LLMs) to produce factually accurate and up-to-date responses. However, the performance of a RAG system is not determined by a
single component but emerges from a complex interplay of modular choices, such as embedding models and retrieval algorithms. This
creates a vast and often opaque configuration space, making it challenging for developers to understand performance trade-offs and
identify optimal designs. To address this challenge, we present RAGExplorer, a visual analytics system for the systematic comparison
and diagnosis of RAG configurations. RAGExplorer guides users through a seamless macro-to-micro analytical workflow. Initially, it
empowers developers to survey the performance landscape across numerous configurations, allowing for a high-level understanding of
which design choices are most effective. For a deeper analysis, the system enables users to drill down into individual failure cases,
investigate how differences in retrieved information contribute to errors, and interactively test hypotheses by manipulating the provided
context to observe the resulting impact on the generated answer. We demonstrate the effectiveness of RAGExplorer through detailed
case studies and user studies, validating its ability to empower developers in navigating the complex RAG design space. Our code and
user guide are publicly available athttps://github.com/Thymezzz/RAGExplorer.
Index Terms—RAG, Visual Analytics, Interactive Visualization
1 INTRODUCTION
• Haoyu Tian, Zhen Wen, Haoxuan Li, and Wei Chen are with the State Key
Lab of CAD&CG, Zhejiang University. E-mail: {thyme | wenzhen |
lihaoxuan | chenvis}@zju.edu.cn.
• Yingchaojie Feng is with School of Computing, National University of
Singapore. E-mail: feng.y@nus.edu.sg.
•Minfeng Zhu is with Zhejiang University. E-mail: minfeng_zhu@zju.edu.cn.
• * Haoyu Tian and Yingchaojie Feng contributed equally to this work.The rapid development of Large Language Models (LLMs) has sig-
nificantly advanced the capabilities of artificial intelligence. How-
ever, their reliance on static training data limits their effectiveness in
knowledge-intensive tasks that require timely information, often re-
sulting in factually inaccurate or outdated responses, a phenomenon
•†Minfeng Zhu, Yingchaojie Feng and Wei Chen are the corresponding
authors.
1arXiv:2601.12991v1  [cs.HC]  19 Jan 2026

commonly known as “hallucination.” The Retrieval-Augmented Gener-
ation (RAG) paradigm [35] was developed to address this limitation.
RAG systems dynamically retrieve relevant documents from external
knowledge sources to augment the LLM’s context, thereby improving
the generated content’s accuracy, relevance, and reliability.
Despite its promise, the performance of a RAG system is not deter-
mined by a single component but emerges from a complex interplay
of modular choices, such as document chunking strategies, embedding
models, and reranking algorithms. This creates a vast and often opaque
configuration space, where the interdependencies between components
are difficult to decipher. Consequently, model developers face a critical
challenge:How can they systematically navigate this vast configura-
tion space to understand the performance trade-offs of different design
choices and effectively diagnose the root causes of failures?
Existing approaches offer only partial solutions. While formal evalu-
ation methodologies [47] and automated reporting tools [13] provide
valuable metrics and summaries, their static nature hinders intuitive,
user-centered exploration of performance data. More importantly, re-
cent visual analytics tools like RAGTrace [12], RAGViz [60], and
RAGGY [32], though valuable for debugging, are fundamentally de-
signed to diagnose a single RAG pipeline. They focus on the internal
workings of one system, rather than enabling the crucial comparative
analysis needed to understand performance trade-offs across different
configurations. This leaves a significant gap in supporting the holistic
optimization of RAG systems.
To bridge this gap, we introduce RAGExplorer, a visual analytics
system designed to facilitate the in-depth comparison and diagnosis of
RAG configurations, guiding model developers toward more effective
designs. The analytical workflow begins in the Component Configu-
ration View (Fig. 1A), where users define the set of configurations for
evaluation. The Performance Overview View (Fig. 1B) then presents
the results in a matrix layout, providing a broad overview that allows
users to survey all configurations and identify key configurations that
can lead to significant performance change. From there, users can
select two configurations for a side-by-side comparison in the Failure
Attribution View (Fig. 1C). This view uses a Sankey diagram to visu-
ally articulate how failure patterns shift between them, enabling users
to quantify the impact of the design changes and form concrete hy-
potheses about the root causes of performance differences. To validate
these hypotheses, users can drill down into the Instance Diagnosis View
(Fig. 1D) for in-depth investigation. It features a novel Dual-Track
Context Comparison to visually pinpoint differences in retrieved docu-
ments and rankings, alongside an interactive sandbox. In this sandbox,
users can directly manipulate the context—for example, by removing a
suspected distractor chunk—and immediately observe the impact on
the generated answer. This allows them to confirm the root cause of a
specific failure.
The main contributions of this paper are:
1.A novel comparative diagnosis workflow that shifts the focus of
RAG analysis from single-pipeline debugging to the systematic
evaluation of multiple configurations.
2.RAGExplorer, an integrated visual analytics system featuring
novel designs for visualizing failure pattern shifts and interactively
verifying the root causes of failures.
3.Two case studies and one expert interview that demonstrate that
RAGExplorer effectively helps model developers understand com-
plex performance trade-offs and make informed design decisions.
2 RELATED WORK
The related work involves the optimization and evaluation of RAG
systems and the visualization for understanding NLP models.
2.1 Optimization and Evaluation of RAG Systems
Optimizing and evaluating RAG is a key research area focused
on navigating its vast configuration space and diagnostic complex-
ity [20,22,26,64,71]. To this end, researchers have proposed numerous
optimization techniques. One line of work introduces architectural
innovations, ranging from flexible modular frameworks [23] to agenticframeworks [44, 48] that incorporate mechanisms for self-reflection [5]
and correction [65]. Others leverage structured knowledge via knowl-
edge graphs [17, 37, 41] or design minimalist frameworks for smaller
models [19]. Another line of work focuses on component-level op-
timizations. These enhance retrieval with techniques like metadata
filtering [43] and chunk filtering [49], or refine generation through
methods such as data synthesis for model finetuning [11, 72] and
credibility-aware attention [15, 40]. Furthermore, interactive tools like
FlashRAG [29] and RAGGY [32] accelerate the development cycle.
However, these works focus on how to build a better RAG, not on why
one configuration outperforms another.
Answering this “why” requires effective evaluation and failure anal-
ysis. Theoretical work has identified seven common failure points [7]
and highlighted LLM limitations, such as performance degradation
with more documents [34] and the “Lost in the Middle” effect [38].
Building on this, various evaluation methods have emerged. Some
offer reference-free automated evaluation [18], while others provide
fine-grained diagnostics by checking claim-level entailment [45] or
assessing document contributions [1, 46, 60]. Large-scale benchmarks
like RAGBench [21] provide comprehensive standards, supplemented
by specialized tests for challenges like unanswerability [42] and unified
comparisons for long context [24] and multi-hop queries [54, 68].
To complement these evaluation approaches, several methodologies
and tools have emerged to support RAG system analysis. Simon et
al. [47] propose a systematic comparative methodology for RAG evalu-
ation, though their tabular presentation format limits intuitive insight
extraction. Meanwhile, tools like RAGXplain [13] focus on automated
suggestions rather than enabling developer-led cross-configuration com-
parison. Building on this foundation, RAGExplorer integrates the sys-
tematic comparative methodology [47] with multi-granularity failure
attribution analysis, incorporating established failure points [7] and
comparisons against documents selected by user to enable hypothesis-
driven debugging.
2.2 Visualization for Understanding NLP Models
Visual analytics is an essential tool for understanding the “black
box” nature of Natural Language Processing (NLP) models and sys-
tems [10, 27, 39, 62, 67, 69, 73]. These visual analytics approaches
can be categorized into two paradigms: single-instance analysis and
comparative analysis.
Single-instance analysis focuses on understanding the behavior of
an individual model or system workflow. Strobelt et al. visualized the
computational mechanisms of traditional NLP models [50,51], followed
by other researchers using various types of visualizations to interpret
attention or neuron distributions of modern NLP models [16, 28, 30, 57,
63, 70]. Other single-instance approaches allow users to interactively
explore model behavior by performance evaluation [2] or prompt-based
methods [14, 53, 61] for LLMs. This analytical paradigm also extends
to explaining RAG systems. These works visually interpret specific
components within a single RAG workflow, including the knowledge
corpus [33], context utilization [60], the end-to-end pipeline [12, 59],
and output factuality [66].
Comparative analysis evaluates systems by highlighting the dif-
ferences between them. Early applications of this approach include
comparing classifiers by exploring performance on data subsets [25] or
using discriminator models [58]. In NLP, comparative techniques have
been used to visualize structural differences in embedding spaces [8]
and to compare textual outputs from various LLMs given the same
prompt [4,52]. More recently, LLM Comparator [31] was developed to
support interactive, side-by-side evaluation of LLM outputs. Similar
comparative techniques are also applied to study the behavioral impacts
of model compression on LLMs [9]. Departing from prior work, RAG-
Explorer introduces the first visual analytics system for the comparative
evaluation of multiple RAG configurations. Our approach adapts the
side-by-side comparison methodology to the full system level, revealing
the performance impact of different component choices.
2

©2026 IEEE. This is the author’s version of the article that has been published in IEEE Transactions on Visualization and
Computer Graphics. The final version of this record is available at: xx.xxxx/TVCG.201x.xxxxxxx/
3 DESIGNPROCESS
In this section, we first illustrate the background on RAG, then distill
key challenges and design requirements from a formative study with
domain experts, followed by a detailed data description.
3.1 Background on RAG
RAG is a paradigm that enhances LLMs by dynamically grounding
them in external knowledge sources. This approach is designed to
mitigate common LLM failure modes such as factual inaccuracies
(“hallucinations”) and reliance on outdated internal knowledge. The
performance of a RAG system is highly dependent on the configuration
of its modular pipeline, which consists of the following core stages:
Indexing.A knowledge corpus is processed and structured for efficient
search. This involves segmenting documents into text chunks, a process
governed by parameters like Chunk Size andChunk Overlap . These
settings control the information’s granularity and influence the trade-off
between context completeness and retrieval precision.
Encoding.Text chunks are transformed into high-dimensional numer-
ical vectors using an embedding model. These embeddings capture
the semantic meaning of the text, enabling effective similarity compar-
isons between a query and the document chunks. Models for this task
include open-source options like bge-m3 and API-based services like
text-embedding-3-small.
Retrieval.Given a user query, the system encodes it and performs a
vector similarity search to find the most relevant chunks. The num-
ber of initial chunks retrieved is set by the Top K parameter. Op-
tionally, a reranker can be applied to this initial set. Rerankers (e.g.,
BGE-reranker-v2-m3 ) use a cross-encoder architecture to more pre-
cisely re-evaluate the relevance of each chunk to the query, yielding a
better-ordered context.
Generation.The final retrieved context and the original query are
formatted into a prompt and passed to a generator LLM. The LLM’s
task is to synthesize this information into a coherent, contextually-
grounded response. This role can be filled by a range of models, such
asGemini 2.5 Flash,GPT-4o-mini, orDeepSeek V3.
3.2 Formative Study
To better support analyzing and diagnosing RAG processes, we conduct
a formative study with four domain experts. Two of them (E1-2) are
model practitioners from a local technology company who focus on
improving the robustness and reliability of LLM. The other two experts
(E3-4) are academic researchers who have studied RAG technology for
several years and have published research papers at relevant academic
conferences. During our regular meetings, we discuss the traditional
pipeline of RAG evaluation and diagnosis, analyze the pain points
within the process, and distill the design requirements of our system to
address these problems.
Through these collaborative sessions, we synthesized the experts’
feedback on the traditional RAG evaluation workflow. We observed
that despite their different backgrounds and task scenarios, the experts
consistently faced a common set of challenges when relying on standard
tools like code notebooks and spreadsheets. We summarized three
primary challenges as follows.
C1Lack of a Holistic Performance Overview.The development of
RAG systems involves tuning numerous parameters, creating a
vast and complex configuration space. Typically, users evaluate
different parameter combinations and present the performance
metrics in spreadsheets. However, this format may hinder users
from grasping the global performance landscape and discovering
key patterns, leading to the risk of overlooking optimal configura-
tions or failing to understand key parameter interactions.
C2Difficulty in Attributing Performance Differences.While ag-
gregate metrics like accuracy are useful for overall evaluation,
they fail to explain why one configuration outperforms another.
A higher score can mask the introduction of new, critical failures,
obscuring the inherent trade-offs of a design choice. Lackingsupport for systematic diagnosis, users must resort to laborious
manual review to investigate the root causes, which severely hin-
ders efficient optimization.
C3Inefficient Instance-Level Debugging.Pinpointing the root
cause of individual failures often requires a deep dive into the
retrieved context to investigate issues. In current practice, users
typically need to manually inspect raw text within a plain text
environment and repeatedly modify the original text to test the
hypothesis. This fractured loop between analysis and experi-
mentation can be time-consuming and may impose a significant
cognitive load, potentially disrupting the analytical flow.
3.3 Design Requirements
Based on the challenges identified in our formative study, we derive
and iteratively refine a set of design requirements that guide the de-
velopment of our visual analysis system. These requirements aim to
address the limitations of traditional workflows by enabling a more
systematic and efficient analytical process.
R1Provide a Holistic Performance Overview.The system should
help developers grasp the global performance landscape across
all configurations (C1), enabling users to quickly identify high-
performing or anomalous configurations and discover high-level
performance patterns. Moreover, it should support the assessment
of a single design choice (e.g., a specific model or parameter).
This helps users understand the contribution of individual compo-
nents to the system’s overall performance.
R2Facilitate a Diagnostic Comparison of Configurations.Aggre-
gate performance metrics often obscure the nuanced trade-offs
inherent in design choices. To help developers understand why
one configuration outperforms another (C2), the system must sup-
port in-depth diagnostic comparisons. This allows developers to
precisely attribute performance gains or losses to specific failure
modes, revealing the inherent trade-offs of a design decision. For
instance, it can show how a reduction in one type of failure cases
may come at the expense of introducing another. By doing so,
the system enables a balanced assessment of each configuration’s
relative strengths and weaknesses.
R3Streamline Instance-Level Debugging.To mitigate the fractured
process of diagnosing individual failures (C3), the system should
provide an integrated sandbox for interactive debugging. It should
allow users to directly and intuitively perturb the context of failure
cases to conduct rapid what-if analysis. For instance, users can
interactively choose different chunks as context to identify the
noise information in different chunks that lead to failure. This
enables users to efficiently verify the failure causes.
3.4 Data Description
Our evaluation is conducted on the MultiHop-RAG dataset [54],
which comprises 2,556 questions derived from an English news cor-
pus. A key characteristic of this dataset is its focus on multi-hop
questions. Each question necessitates retrieving and reasoning over
evidence scattered across multiple documents to formulate a correct
answer, which is a challenging task for general RAG pipelines. The
dataset’s inherent complexity, including both multi-hop reasoning and
instances with insufficient evidence, provides a testbed for diagnosing
the nuanced failure modes of RAG systems. The dataset also includes
ground-truth evidence annotations for the exact evidence sentences
required to answer each question. Inspired by LLM self-explanation
techniques, to enable granular, automated analysis of the generation
process, we mandate a structured output format. The LLM is prompted
to return its response as a JSON object containing two mandatory keys:
supporting_sentences , an array of the exact evidence sentences
used for reasoning, and final_answer , a concise answer constrained
to a maximum of three words.
3

Configur ation InputIndexingChunk Size = 500Chunk Size = 1000EncodingBge-m3Qwen3-Embedding-8Btext-embedding-3-small...Retrie vingTop K ChunksRerank or NotRerank RangeGener ating
...AP erformance Ov erviewAccuracyChunk
Size50010002000K135
B
Quantitativ e P erformance AnalysisQueryChunksEvidenceRetrieval Details
RecallMAPMRRResponse
Evaluate ModelGround TruthAccuracyEF ailur e AttributionConfig1CorrectMissing ContentNot in ContextNot ExtractedConfig2CorrectMissing ContentNot in ContextNot ExtractedC
Hier ar chical F ailur e Attribution
ResponseRule-basedRetrieval DetailsRetrieval FPs
LL M-basedQuery Ground TruthGeneration FPsF
Instance D iagnosisConfig10 .7
0 .6 00 .7
0 .6 0Config2
O riginalO riginalTest 1C orr ectIncorr ectE videnceD
Compar ativ e Causal V erificationQuestionContextDifferen t
Context
Respons e
Model
Response
Ne w
ResponseGFig. 2: RAGExplorer helps users optimize RAG pipelines through four main stages. After a user (A) defines a configuration space, the system
generates (B) a performance overview based on (E) retrieval metrics ( Recall ,MRR,MAP) orAccuracy . Based on this overview, users can (C) select
two configurations to compare, for which (F) an automated algorithm provides a hierarchical attribution of their failure points. This allows users to (D)
analyze the context similarity distribution. They can also (G) modify the context to regenerate answers and verify the impact of a specific instance.
4 FRAMEWORK
The core of our framework is a three-stage analytical methodology de-
signed to move from high-level metrics to specific root causes (Fig. 2).
The process begins with (B) Quantitative Performance Benchmarking
that computes a suite of metrics for system evaluation. To diagnose
the underlying reasons for performance gaps, we employ (C) a Hier-
archical Failure Attribution algorithm that diagnoses and categorizes
the root causes of failures. Finally, the framework facilitates (D) Inter-
active Causal Verification, where users can test hypotheses by directly
intervening in the reasoning process.
4.1 Quantitative Performance Analysis
To quantitatively assess RAG performance, our methodology(Fig. 2E)
evaluates the distinct contributions of the retrieval and generation stages.
We adopt a set of standard metrics from information retrieval, inspired
by recent surveys on RAG evaluation [71].
For the retrieval stage, we measure the efficacy of evidence retrieval
using three established metrics:
•Recall@kquantifies the proportion of relevant documents cap-
tured within the top-k results, indicating if the necessary evi-
dence was available to the generator. It is defined as Recall@k=
|Relevant∩Retrieved k|
|Total Relevant|.
•Mean Reciprocal Rank (MRR)assesses the system’s ability
to rank thefirstrelevant document as high as possible. This
is critical for efficiency and user trust, calculated as MRR=
1
|Q|∑|Q|
i=11
rank i, where rank iis the rank of the first relevant docu-
ment for thei-th query.
•Mean Average Precision (MAP)provides a holistic measure
of ranking quality. Unlike MRR,MAP considers the rank ofall
relevant documents, rewarding systems that place more correct
documents at the top of the list. It averages the precision at each
relevant document’s position across all queries.
For the end-to-end generation stage, we measure the quality of the
final output using the following metric:
•Factual Correctness: This metric quantifies the factual accu-
racy of the generated final_answer . We adopt the LLM-as-a-
Judge paradigm, a methodology validated by automated evalu-
ation frameworks like RAGAS [18]. An adjudicator model is
employed to assess whether the generated answer is semanticallyequivalent to the ground truth. The final score is the proportion
of answers evaluated as correct.
4.2 Hierarchical Failure Attribution
To systematically diagnose RAG failures, this work introduces an auto-
mated, hierarchical failure attribution algorithm (Fig. 3).
Our method is inspired by the catalogue of failure points with Barnett
et al. [7]. It begins by identifying correct answers. It labels them
correct and stops. For all incorrect answers, the algorithm proceeds
through a sequence of checks to assign a single, primary failure point
(FP).
The first check identifies Missing Content (FP1) , which as-
sesses the system’s handling of unanswerable queries. For questions
where the ground truth is “insufficient information,” the RAG system is
prompted to produce the same output. If the system generates any other
answer, this mismatch is categorized as FP1. The next check is for
Wrong Format (FP5) . It evaluates the RAG’s instruction-following
capability. Our parser can often handle malformed JSON and still
extract afinal_answer. But the system logs this case asFP5.
If the initial checks pass, the analysis proceeds to the retrieval stage.
The algorithm examines the evidence within the “rerank range.” A fail-
ure is categorized as Missed the Top Ranked Documents (FP2)
if the proportion of ground-truth evidence sentences in this range falls
below a 70% threshold. While a 100% threshold is theoretically ideal,
we adopted 70% based on empirical observations that RAG systems
can often output correct answers with partial evidence. This may be
due to the synthetic nature of the dataset or the existence of valid rea-
soning paths not captured in the ground truth. The system might find
sufficient evidence (>=70%) in the rerank range. But the final top-k
context passed to the LLM may be insufficient. We apply the same
70% threshold. This failure is classified asNot in Context (FP3).
Failures not caught in the preceding stages are attributed to the gen-
eration module. Among these, Not Extracted (FP4) is identified
under a strict condition: the final context provided to the LLM con-
tained 100% of the required ground-truth evidence, yet the answer was
still incorrect. This indicates the generator failed to either identify the
evidence or correctly reason with the complete information provided.
For more nuanced generation failures, we employ an LLM-based
adjudicator. The adjudicator classifies these cases into Incorrect
Specificity (FP6) (e.g., answering “France” when the question
asks for “Paris”) or Incomplete (FP7) (missing parts of the an-
swer). To perform this classification, the LLM judge is provided
4

©2026 IEEE. This is the author’s version of the article that has been published in IEEE Transactions on Visualization and
Computer Graphics. The final version of this record is available at: xx.xxxx/TVCG.201x.xxxxxxx/
CorrectMissing 
ContentWrong 
FormatMissed Top 
RankedNot in 
ContextNot 
ExtractedIncompleteIncorrect 
SpecificityUnknownResponseGenerationRetrievalQueryDatasetJSONRetrieved
Details
Evalution
LLM-basedreturn FP1return Correctreturn FP5return FP2return FP3return FP4
return FP7return Unknownreturn FP6Ground Truth
No EvidenceCheck formatRerank rangeTop KTop K
Definition
Definition
Definition
Fig. 3: A hierarchical algorithm for diagnosing RAG failure points. It
applies a prioritized cascade of checks to assign a single, primary failure
point to each incorrect answer.
with a prompt containing the failure definitions, the original query ,
theground_truth answer, the model’s final_answer , and its
raw_JSON_response . An Unknown category is included for cases
the adjudicator finds ambiguous or difficult to classify.
4.3 Comparative Causal Verification
To systematically investigate how the composition of a retrieved context
influences the final generated answer, we employ a method of compara-
tive causal verification (Fig. 2G). This process begins by establishing a
content-based correspondence between the chunk sets of two configu-
rations, CAandCB, which is essential when their underlying chunking
strategies or embedding models may differ. A pairwise similarity score
is computed for every chunk ci∈C Aandcj∈C Busing the Jaccard
index to quantify their textual overlap:
J(ci,cj) =|Wi∩W j|
|Wi∪W j|,(1)
where WiandWjrepresent the sets of unique words in chunks ciand
cj. The resulting similarity matrix reveals the informational overlap
and divergence between the two contexts. Based on the matrix, we
retain only chunk pairs whose similarity exceeds a user-adjustable
threshold(default 0.3), balancing recall of nuanced overlaps against
the precision of stronger matches. Having these connections, we can
conduct a causal contrastive analysis along the paths of context preser-
vation, substitution, or omission, tracing which contextual changes
truly drive the differences in generated answers.
Building on this comparative analysis, causal hypotheses are tested
via context perturbation. Let fgenbe the generation function and Corig
be an original context that yields an answer Ans orig=f gen(Corig). A
new, curated context, Cpert, is then constructed by modifying the com-
position of Corig. Such modifications can include removing chunks
hypothesized to be irrelevant distractors or substituting chunks withhigh-similarity counterparts from the alternate configuration. The verifi-
cation is completed by generating a new answer, Ans pert=fgen(Cpert),
and observing the change relative to Ans orig. For instance, if an in-
correct answer becomes correct after a specific chunk is removed, it
provides evidence for a causal link between that chunk and the initial
failure. This method enables an empirical validation of hypotheses
regarding phenomena like the model’s susceptibility to distraction or
the “Lost-in-the-Middle” [38] effect.
5 SYSTEMDESIGN
RAGExplorer is a visual analysis system designed to facilitate the
systematic comparison and diagnosis of RAG systems. We introduce
its workflow, followed by detailed descriptions of four visualization
views. Figure 1 shows an overview of RAGExplorer’s interface.
5.1 Workflow
RAGExplorer follows a multi-stage workflow, moving from a broad
overview of system performance to a detailed investigation of individ-
ual instances. The workflow begins from the Component Configuration
View (Fig. 1A), where the user defines a configuration space for analy-
sis. The user specifies candidate models for modular components of
RAG systems, including response models (generator), embedding
models , and rerank models (reranker), as well as key parameters,
such aschunk sizeandoverlap.
Upon evaluating the specified configurations, the user proceeds to
the Performance Overview View (Fig. 1B) to survey the performance
landscape. The matrix-based layout enables a systematic overview
across all configurations (R1), facilitating the identification of high-
performing designs, unexpected outliers, and overarching patterns. To
diagnose the discrepancy between distinct configurations, the user se-
lects the two configurations of interest for a side-by-side comparison in
the Failure Attribution View (Fig. 1C). This view employs a Sankey dia-
gram to visually articulate the precise shifts in instance-level outcomes.
By tracing the flows between states, the user can quantify the impact of
a design change and form concrete hypotheses about the root causes
of performance differences (R2). For example, they might attribute
the performance drop to a significant flow of instances from Correct
toFP4: Not Extracted failures, suggesting that while the correct
information was retrieved, it was ultimately ignored by the generator.
To validate the hypothesis, the user needs to select the corresponding
flow in the Sankey diagram, which filters the Instance Diagnosis View
(Fig. 1D) to the relevant set of questions. This final view enables
an interactive causal verification and debugging loop at the instance
level (R3). Suspecting that an irrelevant “distractor” chunk retrieved
alongside the correct evidence is causing the failure, the user can
temporarily remove it from the context and regenerate the answer. An
immediate change in the output from incorrect to correct provides direct,
empirical evidence confirming the chunk’s causal role in the failure,
thus completing the analytical process from high-level observation to
validated, instance-level insight.
5.2 Component Configuration View
The Component Configuration View (Fig. 1A) serves as the entry point
of analysis, designed to define a manageable and targeted subset of this
space for systematic exploration. The interface consists of a series of
selection panels, one for each modular component of the RAG pipeline
(e.g., embedding models ) and its key parameters (e.g., chunk size ).
Within each panel, the user can select one or more options. The system
then computes the Cartesian product of these selections to generate the
full set of configurations to be evaluated. Once the user finalizes their
selection and initiates the evaluation, the system executes the pipelines
for all questions in the dataset [54] in parallel to expedite the process.
5.3 Performance Overview View
To facilitate a holistic performance assessment (R1), the Performance
Overview View (Fig. 1B) is designed to support two key analytical
tasks: identifying the highest-performing overall configurations and
discerning the general effectiveness of individual components. We
employ a coordinated matrix and multi-chart layout to support analysis.
5

Inspired by UpSet [36], our design visualizes the intersections among
multiple configuration sets, enabling users to intuitively observe how
different parameter combinations influence overall performance.
The central element of this view is a configuration matrix, where
each column represents a unique RAG pipeline and each row corre-
sponds to a specific component choice. A glyph at each intersection
visually links a configuration to its constituent parts. This matrix is
coordinated with two summary bar chart visualizations:
•Global performancevisualization is positioned above the ma-
trix. Each bar corresponds to a configuration column, and its
length encodes a selected performance metric, including Accuracy,
Recall, MRR, and MAP . This provides a ranked overview for
quickly identifying the best- and worst-performing pipelines.
•Component-wise summary bar chartvisualization is positioned
to the right of the matrix. Each bar represents a single component
choice (e.g., the bge-m3 embedding model ), and its length en-
codes the average performance of all configurations that include that
component. This facilitates an assessment of a component’s general
effectiveness, independent of any single pipeline.
The user can change the primary metric, which re-sorts the configu-
rations in the global chart and the matrix. The principal action is the
selection of two configuration columns, which initiates a comparative
analysis by populating the Failure Attribution View (Fig. 1C).
5.4 Failure Attribution View
To facilitate the systematic diagnosis of performance shifts between
two configurations (R2), the Failure Attribution View (Fig. 1C) moves
beyond aggregate metrics to visualize how instance-level outcomes
change. We employ a Sankey diagram for this purpose, as its flow
metaphor is well-suited to representing the transition of questions be-
tween discrete states. The diagram’s two columns represent the selected
configurations, while the nodes correspond to outcome states: Correct
or a specific failure category from our hierarchical failure attribution
algorithm (Sec 4.2), such as FP2: Missed Top Ranked . The width
of the flows connecting nodes between the two columns is proportional
to the number of questions that transitioned from the source state to
the target state. This encoding provides an immediate visual summary
of the impact of a design change, such as a large flow from FP2in the
baseline toCorrectin the variant, indicating improved retrieval.
To enable a seamless drill-down from pattern to detail, the view
supports two interactions: (1) hovering over a flow or node reveals
a tooltip with the exact instance count; (2) clicking on a flow filters
the Instance Diagnosis View (Fig. 1D) to show only the questions that
constitute that specific transition, preserving the analytical context.
5.5 Instance Diagnosis View
To support efficient instance-level debugging and hypothesis testing
(R3), the Instance Diagnosis View (Fig. 1D) integrates three coordi-
nated views for detailed analysis of individual questions.
Question List.This view (Fig. 1c) is populated based on the user
selection in the Failure Attribution View, which filters questions from
the dataset. This list serves as a navigator, where each entry includes
a circular glyph that visually encodes the proportion of ground-truth
evidence retrieved, offering a pre-attentive cue to retrieval quality.
Dual-Track Context Comparison.This view (Fig. 1d) enables a direct
side-by-side inspection of the retrieved contexts from two configura-
tions. It presents two parallel vertical tracks, one for each configuration.
Within each track, retrieved text chunks are represented as horizontal
bars, with their vertical position determined by their relevance score
(top for highly relevant). Connecting lines between the tracks highlight
textually similar chunks, making critical differences—such as presence,
absence, or changes in rank—immediately apparent. This visualization
supports a human-in-the-loop process of hypothesis testing, such as
hypothesizing a chunk acting as a distractor, by manually adding or
removing the chunk in context (selecting or deselecting the chunk bar
in the tracks) and regenerating the answer to observe its causal impact.
Text Details Panel.This view (Fig. 1e) on the right supports this inspec-
tion by providing the full text of selected item. To facilitate evidence
ID.9      Questio n Which company has spent billions to maintain its default search engine 
status on various platforms and is also accused of harming news 
publishers’ revenue through its business practices?R etrie v ed Ch unksChunk 336Similarity: 0.580In our last roundup, we learned how Google spent $26.3 billion in 2021 
making itself the default search engine across platforms and how Google 
tried to have Chrome preinstalled on iPhones. Over the past couple of 
weeks, more of the inner workings of Google has come to light, including 
some of the search engine’s most lucrative search queries, what the 
revenue-share agreements between Google and Android OEMs look like 
and why Expedia has a bone to pick with Google.Chunk 7949Similarity: 0.576Google paid $26 billion in 2021 to be everyone’s default search 
engine When Google’s search head Prabhakar Raghavan testified in court 
on October 28, he revealed that the tech giant had paid $26.3 billion in 
2021 to multiple browsers, phones and platforms, from companies 
including Apple, Samsung and Mozilla, The Verge reports.Fig. 4: This figure illustrates question ID.9 with retrieved chunk 336 and
chunk 7949. Irrelevant text is grayed out, supporting_sentences are
underlined in blue, and evidence is highlighted in orange. These stylings
may be combined where the text passages overlap.
tracing, the panel uses targeted highlighting: ground-truth evidence
found within a chunk is underlined in orange, while sentences cited in
the model’s response are underlined in blue (Fig. 4).
6 CASESTUDY
Our case studies were conducted by two experts (E1, E2) from the
formative study (Sec. 3.2). Our first case study demonstrates how an
expert (E1) uses our system to resolve a contradiction between industry
consensus and recent research. Our second case study demonstrates
how an expert (E2) uses our system to diagnose a counter-intuitive
component failure, revealing how a “stronger-is-better” hypothesis can
be a performance trap.
6.1 Resolving the Overlap Contradiction
E1 was motivated by a conflict: “industry consensus” often suggests
a small overlap (e.g., 10–20% of chunk size ) [56] to avoid se-
mantically splitting text or too much noise. However, some recent
research [3] suggests an overlap =0 strategy may be superior. To iso-
late the parameter’s effect, he fixed other variables ( Top K ,embedding
model ) and disabled the reranker. He then tested overlap =0 (manu-
ally added in View A) against consensus values (100 and 200) across
differentchunk size(500 and 2000).
E1’s initial results in the Performance Overview (View B) appeared
puzzling (Fig. 5A). They failed to resolve the conflict: the average
accuracy scores for all overlap configurations (0, 100, and 200) were
nearly identical. The expert decided to check other metrics (Fig. 5B).
He examined retrieval-specific statistics, including Recall ,MAP, and
MRR, and found these “simple statistics” also showed almost no differ-
ence between the configurations. At this point, a user relying only on
statistical tables would be completely misled. They would wrongly
conclude that overlap is an unimportant parameter and that both the
consensus and the new research were incorrect, at least for this dataset.
The expert then used the Failure Attribution View (View C) to find
the truth hidden beneath the averages. He compared two of the top-
ranked configurations (Fig. 5D): overlap =0,chunk size =500 and
overlap =200, chunk size =500. The view immediately revealed a
non-obvious finding that all statistical metrics had been completely
masked. Although their final scores were nearly identical, their fail-
ure patterns were significantly different. The overlap =0 configura-
tion had noticeably fewer FP2 (missed top-ranked documents) .
6

©2026 IEEE. This is the author’s version of the article that has been published in IEEE Transactions on Visualization and
Computer Graphics. The final version of this record is available at: xx.xxxx/TVCG.201x.xxxxxxx/
A P erformance Ov erview of Accur acy
Accuracy
Chunk
Size500
2000
Overlap0
100
20046% 46% 45% 45% 45% 44%
1 2
 345%
45%
45%
45%
45%
B P erformance Ov erview of MRR
MRR
Chunk
Size500
2000
Overlap0
100
2000.54 0.54 0.54 0.46 0.46 0.46
3 1 20.46
0.54
0.50
0.50
0.50
C F ailur e Attribution of 2&3
2
Correct
FP 2
FP3
FP 43
Correct
FP 2
FP3
FP 4
D F ailur e Attribution of 1&2
1
Correct
FP 2
FP 3
FP 42
Correct
FP 2
FP3
FP 4
E Instance Diagnosis Comparison of 1&2
1
0.67
0.570.67
0.552
Original
Test 1OriginalCorrect
Incorrect
EvidenceQ.34810-20% overlap
0 overlap
Conclusion
Overlap is 
Irrelevant
The end
Conclusion
0 overlap has 
masked benefit
We need a 
reranker!
Fig. 5: This case study illustrates how an expert resolves a conflict about overlap using our system. (A–B) In the Performance Overview of
Accuracy andMRR, alloverlap settings (0, 100, 200) appear identical, suggesting that overlap is irrelevant. (C–D) Guided by the Failure Attribution
View, the expert finds that overlap =100 and 200 behave similarly, but overlap =0 shows a distinct failure pattern—fewer FP2 (missed top-ranked
documents) yet more FP3 (not in context) . (E) The Instance Diagnosis Comparison of case Q.348 reveals why: overlap =0 retrieves more
evidence into the rerank range, though not ranked in top-k. This uncovers a benefit and suggests the next step: adding a reranker.
At the same time, it had noticeably more FP3 (Not in context) .
To confirm this was a unique property of overlap =0, he also com-
pared overlap=100 ,chunk size =500, and overlap =200, chunk
size =500, and found they were nearly identical in the Failure Attribu-
tion View (Fig. 5C).
To confirm why this failure mode had shifted, the expert drilled
down into the Instance Diagnosis View (View D). He selected the main
source of the difference: the flow of questions that were FP2in the
overlap =200 config but became FP3in the overlap =0 config. The
Instance Diagnosis View (Fig. 5E) provided a visual proof. For these
questions, the overlap =0 pipeline often successfully retrieved more
groundtruth evidence into the rerank range (which led to fewer FP2).
However, he also observed that this new evidence was often ranked low,
failing to make the final top-k (thus causing moreFP3).
This complete diagnostic process provided the expert with a clear
path for an informed improvement. He now understood the contra-
diction: the overlap=0 pipeline had superior recall (fewer FP2) but its
gains were being hidden by a new bottleneck at the next sequential step
of RAG (more FP3). His next concrete step was to create a new config-
uration ( overlap =0 combined with a newly added reranker) to exploit
this proven recall benefit and fix the identified ranking bottleneck.
6.2 Diagnosing Counter-Intuitive Component Failure
An expert (E2) begins by testing a “stronger-is-better” hypothesis.
In the Component Configuration View, he selects a range of param-
eters he wants to compare, including Qwen3-Embedding-0.6B/8B ,
Qwen3-Reranker-0.6B/8B , and chunk size set to 2000. The sys-
tem then runs all possible combinations. He expects Config A ( emb-8B
+reranker-8B +chunk size=2000 ) to perform best. The Perfor-
mance Overview (Fig. 6) reveals a counter-intuitive finding: Config
B (emb-0.6B +reranker-0.6B +chunk size=2000 ) achieves 59%
accuracy, while Config A is 4% worse, scoring only 55%.
Why “stronger” components lead to a decrease? The expert uses
the Failure Attribution View to compare Config A (55%) with Con-
fig B (59%). The view (Fig. 1C) immediately reveals that Config A
(emb-8B +reranker-8B ) has a massive new bottleneck, with signif-
icantly more FP3 (Not in context) andFP4 (Not extracted) .
Many of these new failures were questions that Config B had answeredcorrectly. The Instance Diagnosis View confirms why: the emb-8B
model’s “over-retrieval” of similar but distracted chunks is confusing
thereranker-8B . For example, in Q.442 (Fig. 1D), the Dual-Track
Context Comparison shows that while Config B correctly ranked the
evidence in the top-k, Config A puts them at the bottom instead.
This diagnosis leads to the final step of the expert’s exploration.
The initial “stronger-is-better” hypothesis was proven wrong. If chunk
size=2000 with the most “powerful” embedding model brings too
much noise and causes more failure points, how about a smaller chunk
size with a less powerful embedding model? He thought it could be a
promising improvement.
P erformance Ov erview
59% 59% 55% 54%
59%
55%
56%
57%
57%
1 2Accuracy
EmbeddingQwen3-Emb-0.6B
Qwen3-Emb-8B
RerankerQwen3-Reranker-0.6B
Qwen3-Reranker-8B
ChunkSize 2000
Fig. 6: The Performance Overview results for the expert’s initial
“stronger-is-better” hypothesis. The view shows the “Strongest” configu-
ration (Config A: emb-8B +reranker-8B +chunk size=2000 ) achieves
55% accuracy, while the “Budget” configuration (Config B: emb-0.6B +
reranker-0.6B+chunk size=2000) achieves 59% accuracy.
To test this, he adds the emb-4B model and chunk-500 to his Com-
ponent Configuration View, broadening his search to include mid-range
and less-noisy options. After running the new combinations, he returns
to the Performance Overview View (Fig. 1B). He immediately discov-
ers that his new idea worked. The top-performing configuration is now
7

Config C ( emb-4B +reranker-0.6B +chunk-500 ), which achieves
60% accuracy.
This new configuration represents a +5% measurable improvement
over the failed “strongest” configuration (Config A, 55%). The expert
used a multi-step diagnostic loop: he (1) diagnosed the emb-8B model’s
“Paradox of Riches” failure pattern, (2) formed a new “less-noisy” idea,
and (3) confirmed it by finding a final configuration that is measurably
better, cheaper, and more precise.
7 EXPERTEVALUATION
We conducted a qualitative study with domain experts to evaluate
our system. Specifically, we aimed to assess the effectiveness of its
individual views for analysis and the usability of the overall workflow.
7.1 Participants
We invited four domain experts (E3-E6) for this evaluation. E3 and E4
were the two industry practitioners from our formative study (Sec. 3.2).
The other two experts, E5 and E6, were new participants from industry.
They specialize in AI agent development and are experienced with
RAG technologies.
7.2 Procedure and Tasks
The evaluation with each expert was structured into three phases. The
session began with a 15-minute introduction, during which we pre-
sented the system’s background and core motivation. We also provided
a live demonstration of each view. After this, participants entered a 30-
minute task-oriented exploration phase. We guided them with the two
case studies presented in Sec. 6. While following the main path, partic-
ipants were encouraged to explore freely and vocalize their reasoning
and any discoveries with a think-aloud protocol. The entire interactive
session was audio- and screen-recorded for subsequent analysis. The
session concluded with a 20-minute questionnaire and a semi-structured
interview. Participants rated the system on a five-point Likert scale,
assessing the effectiveness of the views, the system’s usability, and
the coherence of the overall workflow. The final interview allowed
us to gather more in-depth qualitative feedback and elaborate on their
questionnaire responses.
7.3 Results Analysis
This section analyzes the expert feedback and questionnaire data
(Fig. 7). We evaluate the effectiveness of each system view and discuss
the overall usability and workflow.
7.3.1 Effectiveness of System Views
In this section, we detail the expert feedback on the three core views of
our system. The evaluation reveals that while each view was consid-
ered valuable for its specific analytical purpose, experts also identified
distinct usability challenges and offered targeted suggestions for im-
provement related to information density, terminological clarity, and
workflow continuity.
ThePerformance Overview Viewserves as a valuable entry point
for analysis but imposes a visual burden (Q1, µ=4.0/5). While E3
described it as a “perfect start” for finding patterns and generating hy-
potheses, E4 felt “lost” among the numerous configuration points. E5
mentioned “it’s hard to quickly locate and compare the configurations
I want to control with so many points”. Similarly, E6 noted “finding
specific configurations was difficult.” E5 suggested adding direct filter-
ing controls like a drop-down box, while E6 proposed “enhancing the
visual encoding to make performance differences more apparent”.
TheFailure Attribution Viewwas highly rated for its ability to identify
differences between configurations (Q2, µ=4.75/5). E3 highlighted
its ability to help him “lock down” key performance differences and
praised the fluid “brushing and linking” interaction. E5 noted that
“the flows within the Sankey diagram allowed me to very quickly see
changes”. But E4 was “confused” by the specialized definitions of the
Failure Points. He explained that a RAG expert is not necessarily an
expert in failure attribution and may struggle to understand by its name
alone. E3 suggested an insight module to generate textual summaries,and E6 built on this by proposing a “one-click analysis” feature to
automatically highlight the most significant changes.
TheInstance Diagnosis Viewwas also confirmed for its utility in in-
depth case validation (Q3, µ=4.75/5). E5 praised the dual-axis design,
stating it answered a key diagnostic question “at a glance”: “whether
a document was retrieved but failed due to low ranking”. Conversely,
E3 found the experience “abrupt” when a specific case contradicted
a hypothesis proposed previously. This situation forced him to check
numerous cases manually, to determine “if a contradictory case was
an outlier or a general pattern”. Experts suggested an information
hierarchy with summaries (E3) and collapsible text regions (E4). Addi-
tionally, E3 recommended using an LLM to explain “how this evidence
contributes to the conclusion” rather than simply highlighting text. E4
also proposed a feature to track a document chunk’s rank change.
7.3.2 Overall Workflow and System Usability
Beyond the individual views, we assessed the user experience, focusing
on the system’s overall workflow and usability. Experts universally
praised the system’s learnability and expressed a strong willingness to
reuse it. However, this positive reception was balanced by a nuanced
discussion of cognitive load and usability, which received more varied
ratings and highlighted a tension between the system’s analytical power
and the mental effort it requires.
Design Clarity and Cognitive Load.Experts found the system easy to
understand (Q4, µ=4.75/5). E3 described “the layout as clear but noted
a risk of information overload in the Text Details Panel.” Cognitive
load received a more varied rating (Q5, µ=4.25/5). E4 explained that
the system requires him to “actively think and find the reason” for
observed phenomena. E5 felt the pressure came “specifically from the
information density of the Performance Overview View”. In contrast,
E6 finds the information level appropriate and not overwhelming.
Learnability and Usability.The system’s learnability received a
perfect score from all experts (Q6, µ=5.0/5). E6 commented that he
could “tell what it does at a glance”. Usability, in contrast, was rated
more variably (Q7, µ=4.5/5). E5 noted it was very effective for an
expert user wanting to “verify a conclusion I had already predicted.”
And E6 called the interaction “simple and direct”. However, E3 noted
that “the pairwise-comparison design required repeated analyses” and
“imposed a heavy reading burden”. E4 indicated insufficient system
guidance in certain views.
Overall Value and Willingness to Reuse.All experts are willing to
reuse (Q8, µ=5.0/5). E3 called it a “huge improvement” over traditional
analysis, noting it can accelerate the process and provide “convincing
visual evidence” for team collaboration. E4 considered it a valuable
“guided tuning tool” for scenario-specific optimization, since no “silver
bullet” for RAG exists. E5 described it as a practical “debug tool for
RAG workflow”. E6 added that “the system’s comparative nature is
ideal for facilitating analysis akin to ablation studies.” Finally, E3
suggested that “the system’s analytical paradigm has good extensibility
for more complex scenarios, such as GraphRAG”.
7.4 Findings
From the expert evaluation and case studies, we derive four key find-
ings. These insights operate on two distinct levels: the first two reveal
fundamental principles about the RAG optimization process, discov-
ered through our system. The latter two are synthesized from the
experts’ feedback on the analytical workflow, identifying a valuable
new paradigm for AI diagnostics and pointing toward a future of LLM-
augmented collaborative analysis.
Aggregate Statistics Mask Hidden Performance GainsA key find-
ing from our case study is that aggregate performance metrics (like
Accuracy ,MAP, orMRR) are insufficient and can be highly misleading
for RAG optimization. In our case, standard “simple statistics” high-
lighted an apparent lack of difference between several configurations,
showing nearly identical scores that suggested a key parameter was
irrelevant. The Failure Attribution View, however, provided the true
causal explanation. It revealed a hidden failure mode transfer: one
8

©2026 IEEE. This is the author’s version of the article that has been published in IEEE Transactions on Visualization and
Computer Graphics. The final version of this record is available at: xx.xxxx/TVCG.201x.xxxxxxx/
Q5.
Q6.
Q7.No.
Q1.
Q2.
Q3.
Q4.
Q8.Question Avg.
 Scor e Distribution
Strongly Disagr ee Disagr ee Neutral Agree Strongly Agree4.00
4.75
4.75
4.75
4.25
5.00
4.50
5.00The Performance Overview View helps me 
compare configurations and find entry points.
The Failure Attribution View helps me attribute 
differences to specific failure types.
The Instance Diagnosis View supports the 
in-depth diagnosis of individual cases.
The system is easy to learn.
The system is easy to use.
I will use this system again.The design is easy to understand.
The design is not causing cognitive overload.
2 1 1
3 1
2 2
1
 1 2
3 1
4
4
3 1
Fig. 7: The results of the expert interview questionnaire regarding the
effectiveness and usability of the RAGExplorer and workflow.
configuration was proven to be objectively superior in recall (fewer
FP2errors), but this gain was being offset by a new bottleneck in the
reranking stage (more FP3errors). This reveals that aggregate metrics
can mask objective performance gains in early pipeline stages. These
gains are often simply canceled out by a new, downstream bottleneck,
a performance pattern our system is designed to reveal.
Component Synergy Outweighs Individual Strength.Our anal-
ysis also revealed that simply combining the “strongest” compo-
nents does not guarantee the best performance. RAGExplorer’s
Performance Overview first highlighted this counter-intuitive find-
ing, where a “strongest” configuration ( Qwen3-Embedding-8B
+Qwen3-Reranker-8B +chunk size=2000 ) was measurably
outperformed by a “Budget” one ( Qwen3-Embedding-0.6B +
Qwen3-Reranker-0.6B +chunk size=2000 ). This finding offers
a significant practical insight: RAG optimization is not about maximiz-
ing individual component strength, but about finding optimal synergy.
A “weaker” component that provides a “cleaner” signal can be a more
resource-effective and higher-performing strategy than simply deploy-
ing the largest models.
The Value of a Comparative and Hierarchical Workflow.Our study
reveals the effectiveness of an analytical workflow that integrates pair-
wise comparison with hierarchical analysis for diagnosing complex
RAG systems. E6 aptly characterized this workflow’s value as en-
abling “ablation studies”. This approach allows developers to trace
the systemic impact of a single component change, from a high-level
performance shift down to its root causes within the pipeline. This inte-
grated paradigm moves beyond the limitations of analyzing metrics in
isolation, providing a causal, step-by-step narrative of how a configura-
tion choice influences specific failure modes. The finding confirms that
for multi-component AI systems, such a workflow is critical for moving
from simple performance comparison to deep, actionable diagnostics.
The Need for LLM-Augmented Collaborative Analysis.A consis-
tent theme from expert feedback was the demand for the system to
evolve from a passive visualization tool into a proactive analytical part-
ner. This signals an emerging need for a new paradigm of human-AI
collaborative analysis where the exploratory burden is shared. Experts
proposed features for automated discovery, such as an “insight module”
(E3) and a “one-click analysis” (E6), to automatically surface the most
significant performance changes. More profoundly, feedback called for
deeper explanatory augmentation. A key suggestion was to leverage an
LLM to explain “how this evidence contributes to the conclusion” (E3),
helping users interpret complex or contradictory cases. These sugges-
tions collectively point toward a future where interactive visualization
is augmented by LLM-driven reasoning, allowing human experts to
concentrate on strategic decision-making and validation.8 DISCUSSION
The analysis of RAG systems is complicated by the modularity of their
architecture, where end-to-end performance is an emergent property of
interacting components. This work introduces RAGExplorer, a visual
analytics system designed to address this complexity. We contribute
a methodology for comparative diagnosis, which shifts the analytical
objective from single-configuration debugging to a systematic analysis
of performance differences and failure mode transitions between mul-
tiple configurations. This approach provides a structured paradigm to
explore the design space and understand its inherent trade-offs.
Generalizability.In this work, we have demonstrated how our system
effectively works on a fixed set of RAG configurations. To support
generalization with more user-defined configurations, the RAGExplorer
is developed as a plug-in system where users can register custom com-
ponents to extend additional configurations. On this basis, our visual
analytics paradigm can be further generalized to other modular RAG
pipelines. Its comparative visualization is effective for diagnosing
system-level behaviors in multi-stage pipelines, e.g., agentic workflows.
By focusing on visualizing the differences between configurations,
such as failure transitions or content disparities, our approach allows
analysts to debug emergent issues that are invisible to component-level
inspection. Benefiting from our visual and interactive interface, this
comparative analysis can be further strengthened by enabling active
intervention for users to directly test causal hypotheses.
Applicability.Our framework relies on the ground-truth data to calcu-
late the evaluation metrics and failure points. To improve applicability
in production where the ground truth is missing, RAGExplorer can be
adapted to operate using LLM-as-a-Judge evaluators for metrics like
context relevance and faithfulness. This preserves the core capability
of understanding the impact of configuration changes for comparative
diagnosis by visualizing shifts between these automatically assessed
metrics, thus relaxing the dependency on annotated ground-truth data.
Scalability.The current system is designed for the focused, in-depth
comparison of a small number of configurations, which is typical for
root-cause analysis. While the Performance Overview View supports
the initial exploration of dozens of configurations, the approach does
not scale directly to thousands of experiments common in large-scale
hyperparameter searches. Addressing this would require future work
on automated methods to supplement the visual exploration, such as
algorithms to detect salient patterns across the configuration space or to
recommend the most diagnostically insightful comparisons to the user.
Further future work could incorporate sampling [6, 55] or hierarchical
search of the parameter space to reduce computational overhead.
Study Limitations.The evaluation was scoped to question-answering
tasks with annotated ground-truth data. While we have discussed
paths to generalize, the system’s effectiveness on open-ended tasks
without such annotations remains a key area for future work. Besides,
our user study was conducted exclusively with domain experts. The
learnability and utility of RAGExplorer for novice users, who may lack
deep domain knowledge, is an important area for future investigation.
9 CONCLUSION
In this paper, we introduced RAGExplorer, a visual analytics system
designed to untangle the complexity of configuring RAG systems. We
presented a novel analytical workflow that shifts the focus from de-
bugging single pipelines to the systematic, comparative diagnosis of
multiple configurations. This approach empowers developers to move
seamlessly from a high-level performance overview to an in-depth,
interactive investigation of instance-level failures. Our case studies
and user evaluation demonstrated that this comparative paradigm effec-
tively uncovers critical, non-obvious insights into RAG system behavior.
These results underscore the necessity of a holistic, system-level ap-
proach to optimization. By providing a structured framework and an
interactive interface for comparative analysis, RAGExplorer not only
offers a practical tool for AI developers but also establishes a visual
analytics paradigm poised to bring clarity to the emergent behaviors of
a wide range of complex, modular AI systems.
9

ACKNOWLEDGMENTS
We thank anonymous reviewers for their insightful reviews. This work
was supported by the National Key R&D Program of China (Grant
Nos. 2024YFB4505500 and 2024YFB4505503), the National Natural
Science Foundation of China (Grant Nos. 62132017, 62421003, and
62302435), and the Zhejiang Provincial Natural Science Foundation of
China (Grant No. LD24F020011).
REFERENCES
[1]A. Alinejad, K. Kumar, and A. Vahdat. Evaluating the retrieval component
in LLM-based question answering systems.CoRR, abs/2406.06458, 2024.
doi: 10.48550/arXiv.2406.06458 2
[2]S. Amershi, M. Chickering, S. M. Drucker, B. Lee, P. Simard, and J. Suh.
ModelTracker: Redesigning performance analysis tools for machine learn-
ing. InProc. CHI, pp. 337–346, Apr. 2015. doi: 10.1145/2702123.
2702509 2
[3]M. Amiri and T. Bocklitz. Chunk twice, embed once: A systematic study
of segmentation and representation trade-offs in chemistry-aware retrieval-
augmented generation.CoRR, abs/2506.17277, 2025. doi: 10.48550/arXiv
.2506.17277 6
[4]I. Arawjo, C. Swoopes, P. Vaithilingam, M. Wattenberg, and E. L. Glass-
man. ChainForge: A visual toolkit for prompt engineering and LLM
hypothesis testing. InProc. CHI, pp. 304:1–304:18, May 2024. doi: 10.
1145/3613904.3642016 2
[5]A. Asai, Z. Wu, Y . Wang, A. Sil, and H. Hajishirzi. Self-RAG: Learning
to retrieve, generate, and critique through self-reflection. InProc. ICLR,
May 2024. 2
[6]M. Barker, A. Bell, E. Thomas, J. Carr, T. Andrews, and U. Bhatt. Faster,
cheaper, better: Multi-objective hyperparameter optimization for LLM
and RAG systems.CoRR, abs/2502.18635, 2025. doi: 10.48550/arXiv.
2502.18635 9
[7]S. Barnett, S. Kurniawan, S. Thudumu, Z. Brannelly, and M. Abdelrazek.
Seven failure points when engineering a retrieval augmented generation
system. InProc. CAIN, pp. 194–199, June 2024. doi: 10.1145/3644815.
3644945 2, 4
[8]A. Boggust, B. Carter, and A. Satyanarayan. Embedding comparator:
Visualizing differences in global structure and local neighborhoods via
small multiples. InProc. IUI, pp. 746–766, Mar. 2022. doi: 10.1145/
3490099.3511122 2
[9]A. Boggust, V . Sivaraman, Y . Assogba, D. Ren, D. Moritz, and F. Hohman.
Compress and compare: Interactively evaluating efficiency and behavior
across ML model compression experiments.IEEE Trans. Vis. Comput.
Graph., 31(1):809–819, Jan. 2025. doi: 10.1109/TVCG.2024.3456371 2
[10] R. Brath, D. A. Keim, J. Knittel, S. Pan, P. Sommerauer, and H. Strobelt.
The role of interactive visualization in explaining (large) NLP models:
From data to inference.CoRR, abs/2301.04528, 2023. doi: 10.48550/arXiv
.2301.04528 2
[11] M. Chen, X. Chen, and W.-t. Yih. Few-shot data synthesis for open domain
multi-hop question answering. InProc. EACL, pp. 190–208, Mar. 2024.
doi: 10.18653/v1/2024.eacl-long.12 2
[12] S. Cheng, J. Li, H. Wang, and Y . Ma. RAGTrace: Understanding and
refining retrieval-generation dynamics in retrieval-augmented generation.
InProc. UIST, pp. 176:1–176:20, Sept. 2025. doi: 10.1145/3746059.
3747741 2
[13] D. Cohen, L. Burg, and G. Barkan. RAGXplain: From explainable evalu-
ation to actionable guidance of RAG pipelines.CoRR, abs/2505.13538,
2025. doi: 10.48550/arXiv.2505.13538 2
[14] A. Coscia and A. Endert. KnowledgeVIS: Interpreting language models
by comparing fill-in-the-blank prompts.IEEE Trans. Vis. Comput. Graph.,
30(9):6520–6532, Sept. 2024. doi: 10.1109/TVCG.2023.3346713 2
[15] B. Deng, W. Wang, F. Zhu, Q. Wang, and F. Feng. CrAM: Credibility-
aware attention modification in LLMs for combating misinformation in
RAG. InProc. AAAI, pp. 23760–23768, Apr. 2025. doi: 10.1609/aaai.
v39i22.34547 2
[16] J. F. DeRose, J. Wang, and M. Berger. Attention flows: Analyzing and
comparing attention mechanisms in language models.IEEE Trans. Vis.
Comput. Graph., 27(2):1160–1170, Feb. 2021. doi: 10.1109/TVCG.2020.
3028976 2
[17] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Tru-
itt, D. Metropolitansky, R. O. Ness, and J. Larson. From local to
global: A graph RAG approach to query-focused summarization.CoRR,
abs/2404.16130, 2024. doi: 10.48550/arXiv.2404.16130 2[18] S. Es, J. James, L. Espinosa Anke, and S. Schockaert. RAGAs: Automated
evaluation of retrieval augmented generation. InProc. EACL, pp. 150–158,
Mar. 2024. doi: 10.18653/v1/2024.eacl-demo.16 2, 4
[19] T. Fan, J. Wang, X. Ren, and C. Huang. MiniRAG: Towards extremely
simple retrieval-augmented generation.CoRR, abs/2501.06713, 2025. doi:
10.48550/arXiv.2501.06713 2
[20] W. Fan, Y . Ding, L. Ning, S. Wang, H. Li, D. Yin, T.-S. Chua, and Q. Li.
A survey on RAG meeting LLMs: Towards retrieval-augmented large
language models. InProc. KDD, pp. 6491–6501, Aug. 2024. doi: 10.
1145/3637528.3671470 2
[21] R. Friel, M. Belyi, and A. Sanyal. RAGBench: Explainable benchmark
for retrieval-augmented generation systems.CoRR, abs/2407.11005, 2024.
doi: 10.48550/arXiv.2407.11005 2
[22] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, M. Wang,
and H. Wang. Retrieval-Augmented generation for large language models:
A survey.CoRR, abs/2312.10997, 2024. doi: 10.48550/arXiv.2312.10997
2
[23] Y . Gao, Y . Xiong, M. Wang, and H. Wang. Modular RAG: Transform-
ing RAG systems into LEGO-like reconfigurable frameworks.CoRR,
abs/2407.21059, 2024. doi: 10.48550/arXiv.2407.21059 2
[24] Y . Gao, Y . Xiong, W. Wu, B. Li, Y . Zhong, and H. Wang. U-NIAH:
Unified RAG and LLM evaluation for long context needle-in-a-haystack.
ACM Trans. Inf. Syst., Dec. 2025. doi: 10.1145/3786609 2
[25] M. Gleicher, A. Barve, X. Yu, and F. Heimerl. Boxer: Interactive compari-
son of classifier results.Computer Graphics Forum, 39(3):181–193, July
2020. doi: 10.1111/cgf.13972 2
[26] S. Gupta, R. Ranjan, and S. N. Singh. A comprehensive survey of retrieval-
augmented generation (RAG): Evolution, current landscape and future
directions.CoRR, abs/2410.12837, 2024. doi: 10.48550/arXiv.2410.
12837 2
[27] F. Hohman, M. Kahng, R. Pienta, and D. H. Chau. Visual analytics in
deep learning: An interrogative survey for the next frontiers.IEEE Trans.
Vis. Comput. Graph., 25(8):2674–2693, Aug. 2019. doi: 10.1109/TVCG.
2018.2843369 2
[28] B. Hoover, H. Strobelt, and S. Gehrmann. exBERT: A visual analysis
tool to explore learned representations in Transformer models. InPro-
ceedings of the 58th Annual Meeting of the Association for Computational
Linguistics: System Demonstrations, pp. 187–196, July 2020. doi: 10.
18653/v1/2020.acl-demos.22 2
[29] J. Jin, Y . Zhu, Z. Dou, G. Dong, X. Yang, C. Zhang, T. Zhao, Z. Yang, and
J.-R. Wen. FlashRAG: a modular toolkit for efficient retrieval-augmented
generation research. InProc. WWW, pp. 737–740, May 2025. doi: 10.
1145/3701716.3715313 2
[30] M. Kahng, P. Y . Andrews, A. Kalro, and D. H. Chau. ActiVis: Visual
exploration of industry-scale deep neural network models.IEEE Trans.
Vis. Comput. Graph., 24(1):88–97, Jan. 2018. doi: 10.1109/TVCG.2017.
2744718 2
[31] M. Kahng, I. Tenney, M. Pushkarna, M. X. Liu, J. Wexler, E. Reif,
K. Kallarackal, M. Chang, M. Terry, and L. Dixon. LLM comparator:
Interactive analysis of side-by-side evaluation of large language models.
IEEE Trans. Vis. Comput. Graph., 31(1):503–513, Jan. 2025. doi: 10.
1109/TVCG.2024.3456354 2
[32] Q. R. Lauro, S. Shankar, S. Zeighami, and A. Parameswaran. RAG
without the lag: Interactive debugging for retrieval-augmented generation
pipelines.CoRR, abs/2504.13587, 2025. doi: 10.48550/arXiv.2504.13587
2
[33] S. Y .-T. Lee and K.-L. Ma. HINTs: Sensemaking on large collections of
documents with hypergraph visualization and INTelligent agents.IEEE
Trans. Vis. Comput. Graph., 31(9):5532–5546, Sept. 2025. doi: 10.1109/
TVCG.2024.3459961 2
[34] S. Levy, N. Mazor, L. Shalmon, M. Hassid, and G. Stanovsky. More
documents, same length: Isolating the challenge of multiple documents
in RAG. InFindings of the Association for Computational Linguistics:
EMNLP, pp. 19539–19547, Nov. 2025. doi: 10.18653/v1/2025.findings
-emnlp.1064 2
[35] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küt-
tler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-
augmented generation for knowledge-intensive NLP tasks. InProc.
NeurIPS, pp. 9459–9474, Dec. 2020. 2
[36] A. Lex, N. Gehlenborg, H. Strobelt, R. Vuillemot, and H. Pfister. UpSet:
Visualization of intersecting sets.IEEE Trans. Vis. Comput. Graph.,
20(12):1983–1992, Dec. 2014. doi: 10.1109/TVCG.2014.2346248 6
[37] F. Li, P. Fang, Z. Shi, A. Khan, F. Wang, W. Wang, Zhangxin-hw,
10

©2026 IEEE. This is the author’s version of the article that has been published in IEEE Transactions on Visualization and
Computer Graphics. The final version of this record is available at: xx.xxxx/TVCG.201x.xxxxxxx/
and C. Yongjian. CoT-RAG: Integrating chain of thought and retrieval-
augmented generation to enhance reasoning in large language models. In
Findings of the Association for Computational Linguistics: EMNLP, pp.
3119–3171, Nov. 2025. doi: 10.18653/v1/2025.findings-emnlp.168 2
[38] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and
P. Liang. Lost in the middle: How language models use long contexts.
TACL, 12:157–173, Feb. 2024. doi: 10.1162/tacl_a_00638 2, 5
[39] S. Liu, X. Wang, M. Liu, and J. Zhu. Towards better analysis of ma-
chine learning models: A visual analytics perspective.Visual Informatics,
1(1):48–56, Mar. 2017. doi: 10.1016/j.visinf.2017.01.006 2
[40] R. Pan, B. Cao, H. Lin, X. Han, J. Zheng, S. Wang, X. Cai, and L. Sun.
Not all contexts are equal: Teaching LLMs credibility-aware generation.
InProc. EMNLP, pp. 19844–19863, Nov. 2024. doi: 10.18653/v1/2024.
emnlp-main.1109 2
[41] B. Peng, Y . Zhu, Y . Liu, X. Bo, H. Shi, C. Hong, Y . Zhang, and S. Tang.
Graph retrieval-augmented generation: A survey.ACM Trans. Inf. Syst.,
44(2), article no. 35, Dec. 2025. doi: 10.1145/3777378 2
[42] X. Peng, P. K. Choubey, C. Xiong, and C.-S. Wu. Unanswerability evalu-
ation for retrieval augmented generation. InProc. ACL, pp. 8452–8472,
July 2025. doi: 10.18653/v1/2025.acl-long.415 2
[43] M. Poliakov and N. Shvai. Multi-Meta-RAG: Improving RAG for multi-
hop queries using database filtering with LLM-extracted metadata. In
Proc. ICTERI, pp. 334–342. Springer Nature Switzerland, Feb. 2025. doi:
10.1007/978-3-031-81372-6_25 2
[44] C. Ravuru, S. S. Srinivas, and V . Runkana. Agentic retrieval-augmented
generation for time series analysis.CoRR, abs/2408.14484, 2024. doi: 10.
48550/arXiv.2408.14484 2
[45] D. Ru, L. Qiu, X. Hu, T. Zhang, P. Shi, S. Chang, J. Cheng, C. Wang,
S. Sun, H. Li, Z. Zhang, B. Wang, J. Jiang, T. He, Z. Wang, P. Liu,
Y . Zhang, and Z. Zhang. RAGCHECKER: A fine-grained framework for
diagnosing retrieval-augmented generation. InProc. NeurIPS, article no.
692, Dec. 2024. 2
[46] A. Salemi and H. Zamani. Evaluating retrieval quality in retrieval-
augmented generation. InProc. SIGIR, pp. 2395–2400, July 2024. doi: 10
.1145/3626772.3657957 2
[47] S. Simon, A. Mailach, J. Dorn, and N. Siegmund. A methodology for
evaluating RAG systems: A case study on configuration dependency
validation.CoRR, abs/2410.08801, 2024. doi: 10.48550/arXiv.2410.
08801 2
[48] A. Singh, A. Ehtesham, S. Kumar, and T. T. Khoei. Agentic retrieval-
augmented generation: A survey on agentic RAG.CoRR, abs/2501.09136,
2025. doi: 10.48550/arXiv.2501.09136 2
[49] I. S. Singh, R. Aggarwal, I. Allahverdiyev, M. Taha, A. Akalin, K. Zhu,
and S. O’Brien. ChunkRAG: Novel LLM-chunk filtering method for RAG
systems. InProc. NAACL Student Research Workshop, 2025. doi: 10.
48550/arXiv.2410.19572 2
[50] H. Strobelt, S. Gehrmann, M. Behrisch, A. Perer, H. Pfister, and A. M.
Rush. Seq2seq-Vis: A visual debugging tool for sequence-to-sequence
models.IEEE Trans. Vis. Comput. Graph., 25(1):353–363, Jan. 2019. doi:
10.1109/TVCG.2018.2865044 2
[51] H. Strobelt, S. Gehrmann, H. Pfister, and A. M. Rush. LSTMVis: A tool
for visual analysis of hidden state dynamics in recurrent neural networks.
IEEE Trans. Vis. Comput. Graph., 24(1):667–676, Jan. 2018. doi: 10.
1109/TVCG.2017.2744158 2
[52] H. Strobelt, B. Hoover, A. Satyanaryan, and S. Gehrmann. LMdiff: A
visual diff tool to compare language models. InProceedings of the 2021
Conference on Empirical Methods in Natural Language Processing: Sys-
tem Demonstrations, pp. 96–105, Nov. 2021. doi: 10.18653/v1/2021.
emnlp-demo.12 2
[53] H. Strobelt, A. Webson, V . Sanh, B. Hoover, J. Beyer, H. Pfister, and
A. M. Rush. Interactive and visual prompt engineering for ad-hoc task
adaptation with large language models.IEEE Trans. Vis. Comput. Graph.,
29(1):1146–1156, Jan. 2023. doi: 10.1109/TVCG.2022.3209479 2
[54] Y . Tang and Y . Yang. MultiHop-RAG: benchmarking retrieval-augmented
generation for multi-hop queries.CoRR, abs/2401.15391, 2024. doi: 10.
48550/arXiv.2401.15391 2, 3, 5
[55] C. Thornton, F. Hutter, H. H. Hoos, and K. Leyton-Brown. Auto-WEKA:
Combined selection and hyperparameter optimization of classification
algorithms. InProc. KDD, pp. 847–855, Aug. 2013. doi: 10.1145/2487575
.2487629 9
[56] B. Tuychiev. Best chunking strategies for RAG in 2025. https://www.
firecrawl.dev/blog/best-chunking-strategies-rag-2025 ,
Oct. 2025. Accessed: 2025-11-10. 6[57] J. Vig. A multiscale visualization of attention in the Transformer model.
InProceedings of the 57th Annual Meeting of the Association for Compu-
tational Linguistics: System Demonstrations, pp. 37–42, July 2019. doi:
10.18653/v1/P19-3007 2
[58] J. Wang, L. Wang, Y . Zheng, C.-C. M. Yeh, S. Jain, and W. Zhang.
Learning-from-disagreement: A model comparison and visual analyt-
ics framework.IEEE Trans. Vis. Comput. Graph., 29(9):3809–3825, Sept.
2023. doi: 10.1109/TVCG.2022.3172107 2
[59] K. Wang, B. Pan, Y . Feng, Y . Wu, J. Chen, M. Zhu, and W. Chen.
XGraphRAG: Interactive visual analysis for graph-based retrieval-
augmented generation. InProc. PacificVis, pp. 1–11, Apr. 2025. doi:
10.1109/PacificVis64226.2025.00005 2
[60] T. Wang, J. He, and C. Xiong. RAGViz: Diagnose and visualize retrieval-
augmented generation. InProc. EMNLP, pp. 320–327, Nov. 2024. doi: 10
.18653/v1/2024.emnlp-demo.33 2
[61] X. Wang, R. Huang, Z. Jin, T. Fang, and H. Qu. CommonsenseVIS:
Visualizing and understanding commonsense reasoning capabilities of
natural language models.IEEE Trans. Vis. Comput. Graph., 30(1):273–
283, Jan. 2024. doi: 10.1109/TVCG.2023.3327153 2
[62] X. Wang, Z. Wu, W. Huang, Y . Wei, Z. Huang, M. Xu, and W. Chen.
VIS+AI: Integrating visualization with artificial intelligence for efficient
data analysis.Frontiers of Computer Science, 17(6):176709, June 2023.
doi: 10.1007/s11704-023-2691-y 2
[63] Z. J. Wang, R. Turko, and D. H. Chau. Dodrio: Exploring Transformer
models with interactive visualization. InProceedings of the 59th Annual
Meeting of the Association for Computational Linguistics and the 11th
International Joint Conference on Natural Language Processing: System
Demonstrations, pp. 132–141, Aug. 2021. doi: 10.18653/v1/2021.acl
-demo.16 2
[64] S. Wei, Y . Tong, Z. Zhou, Y . Xu, J. Gao, T. Wei, T. He, and W. Lv.
Federated reasoning LLMs: A survey.Frontiers of Computer Science,
19(12):1912613, June 2025. doi: 10.1007/s11704-025-50480-3 2
[65] S.-Q. Yan, J.-C. Gu, Y . Zhu, and Z.-H. Ling. Corrective retrieval aug-
mented generation.CoRR, abs/2401.15884, 2024. doi: 10.48550/arXiv.
2401.15884 2
[66] Y . Yan, Y . Hou, Y . Xiao, R. Zhang, and Q. Wang. KNowNet: Guided
health information seeking from LLMs via knowledge graph integration.
IEEE Trans. Vis. Comput. Graph., 31(1):547–557, Jan. 2025. doi: 10.
1109/TVCG.2024.3456364 2
[67] W. Yang, M. Liu, Z. Wang, and S. Liu. Foundation models meet visu-
alizations: Challenges and opportunities.Computational Visual Media,
10(3):399–424, June 2024. doi: 10.1007/s41095-023-0393-x 2
[68] Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. Cohen, R. Salakhutdinov, and
C. D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop
question answering. InProc. EMNLP, pp. 2369–2380, Oct. 2018. doi: 10.
18653/v1/D18-1259 2
[69] Y . Ye, J. Hao, Y . Hou, Z. Wang, S. Xiao, Y . Luo, and W. Zeng. Gener-
ative AI for visualization: State of the art and future directions.Visual
Informatics, 8(2):43–66, June 2024. doi: 10.1016/j.visinf.2024.04.003 2
[70] C. Yeh, Y . Chen, A. Wu, C. Chen, F. Viegas, and M. Wattenberg. Atten-
tionViz: A global view of Transformer attention.IEEE Trans. Vis. Comput.
Graph., 30(1):262–272, Jan. 2024. doi: 10.1109/TVCG.2023.3327163 2
[71] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu. Evaluation of
retrieval-augmented generation: A survey. InBig Data, pp. 102–120, Jan.
2025. doi: 10.1007/978-981-96-1024-2_8 2, 4
[72] W. Yu, H. Zhang, X. Pan, P. Cao, K. Ma, J. Li, H. Wang, and D. Yu. Chain-
of-Note: Enhancing robustness in Retrieval-Augmented Language Models.
InProc. EMNLP, pp. 14672–14685, Nov. 2024. doi: 10.18653/v1/2024.
emnlp-main.813 2
[73] Z. Zhou, M. Zhu, and W. Chen. A human-centric perspective on inter-
pretability in large language models.Visual Informatics, 9(1):A1–A3, Mar.
2025. doi: 10.1016/j.visinf.2025.03.001 2
11