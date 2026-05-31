# From Norms to Indicators (N2I-RAG): An Agentic Retrieval-Augmented Generation Framework for Legal Indicator Computation

**Authors**: Youssef Al Mouatamid, Marie Bonnin, Jihad Zahir

**Published**: 2026-05-26 12:21:48

**PDF URL**: [https://arxiv.org/pdf/2605.26926v1](https://arxiv.org/pdf/2605.26926v1)

## Abstract
Computing legal indicators from normative texts is a key task in legal monitoring and policy evaluation, but presents significant challenges due to the complexity, scale, and interpretive nature of legal language, as well as the variability in available document quality. Existing natural language processing techniques and generative models can assist in legal analysis, but often suffer from high risk of hallucinations and lack the interpretability and evidence grounding required for reliable indicator computation. This paper presents N2I-RAG (From Norms to Indicators), an agentic retrieval-augmented generation framework designed to automate the computation of legal indicators in a transparent and traceable way. We integrate adaptive retrieval, llm-based agents, and validation mechanisms in a modular pipeline, where each component performs a defined role in filtering, retrieving, and assessing evidence, and in producing binary legal outcomes linked to identifiable legal provisions. The framework emphasizes traceability by requiring explicit explanations of intermediate decisions and final indicator assignments. We evaluate N2I-RAG using an in-house constructed French marine environmental law corpus that includes both scanned and digital sources. Comparative experiments with multiple language model families demonstrate that the proposed approach consistently outperforms baseline systems, and generalizes well when tested on 2 different bans. The results indicate that agentic retrieval-augmented generation can bridge open-text legal language and standardized indicator computation, offering a foundation for transparent and scalable legal observatories.

## Full Text


<!-- PDF content starts -->

From Norms to Indicators (N2I-RAG): An Agentic
Retrieval-Augmented Generation Framework for
Legal Indicator Computation
Youssef Al Mouatamid1,2*, Marie Bonnin3and Jihad Zahir1,4
1LISI Laboratory, Cadi Ayyad University, Marrakesh, 40000, Morocco.
2LEMAR, Univ Brest, Plouzane, F-29280, France.
3IRD, Univ Brest, CNRS, Ifremer, LEMAR, Plouzane, F-29280, France.
4UMMISCO, IRD France Nord, Bondy, F-93143, France.
*Corresponding author(s). E-mail(s):
youssef.almouatamid@univ-brest.fr;
Contributing authors: marie.bonnin@ird.fr; j.zahir@uca.ac.ma;
Abstract
Computing legal indicators from normative texts is a key task in legal monitoring
and policy evaluation, but presents significant challenges due to the complex-
ity, scale, and interpretive nature of legal language, as well as the variability in
available document quality. Existing natural language processing techniques and
generative models can assist in legal analysis, but often suffer from high risk
of hallucinations and lack the interpretability and evidence grounding required
for reliable indicator computation. This paper presents N2I-RAG (From Norms
to Indicators), an agentic retrieval-augmented generation framework designed
to automate the computation of legal indicators in a transparent and traceable
way. We integrate adaptive retrieval, llm-based agents, and validation mecha-
nisms in a modular pipeline, where each component performs a defined role
in filtering, retrieving, and assessing evidence, and in producing binary legal
outcomes linked to identifiable legal provisions. The framework emphasizes trace-
ability by requiring explicit explanations of intermediate decisions and final
indicator assignments. We evaluate N2I-RAG using an in-house constructed
French marine environmental law corpus that includes both scanned and digital
sources. Comparative experiments with multiple language model families demon-
strate that the proposed approach consistently outperforms baseline systems,
and generalizes well when tested on 2 different bans. The results indicate that
agentic retrieval-augmented generation can bridge open-text legal language and
1arXiv:2605.26926v1  [cs.AI]  26 May 2026

standardized indicator computation, offering a foundation for transparent and
scalable legal observatories.
Keywords:Generative AI, Retrieval-Augmented Generation (RAG), Legal indicator
computation, Multi-agent systems, Explainable legal AI
1 Introduction
Legal indicators are structured measures that quantify legal implementation, compli-
ance, or performance, serving as analytical tools for monitoring governance and policy
outcomes.
In fields such as marine environmental law, legal indicators provide key evidence
for assessing whether regulatory frameworks effectively address ecological challenges,
from pollution to biodiversity loss. They translate legal obligations, often expressed in
complex and qualitative terms, into standardized and comparable metrics usable by
policymakers and researchers.
Several initiatives have demonstrated the value of legal indicators for environmental
governance. For example, Michel Prieur et al. [1] have developed an environmental
law application indicator based on variables extracted from expert surveys. However,
the construction of these indicators remains labor-intensive and depends on expert
interpretation and extensive document analysis. As legal systems expand in scope and
complexity, these practices face growing limitations in terms of scalability, consistency,
and transparency. This creates a need for reliable and reproducible methods to com-
pute legal indicators at scale.
The design of relevant legal indicators requires not only a clear conceptual frame-
work but also a reproducible methodology and outcomes. Odeline Billant’s thesis [2]
offers a comprehensive framework for legal indicator construction, emphasizing the
alignment between legal norms, empirical observations, and operational metrics. This
framework highlights the importance of traceability, as each indicator must be sup-
ported by explicit textual evidence. However, implementing this framework in practice
is challenging owing to the large -scale, heterogeneous, and linguistically complex legal
texts. Manually identifying relevant norms and rules is both time-consuming and vul-
nerable to subjective interpretation. Automating legal indicator construction through
natural language processing techniques and generative AI is therefore a key element
in enabling consistent, large-scale, and comprehensible monitoring of complex legal
systems.
Natural Language Processing (NLP) has made progress toward automating aspects
of legal analysis. Early work focused on tasks such as named entity recognition, doc-
ument classification, and information retrieval have enabled partial automation to a
certain degree of legal analysis workflows [3–10]. While these methods support specific
subtasks, they typically rely on limited annotated datasets and domain-specific rules,
which limits their generalizability when faced with new topics or languages. Moreover,
their systems often run as “black boxes”, offering little or no insight into the reasoning
leading to how their outputs are produced. This lack of interpretability is particularly
2

problematic for legal indicator computation, where transparency and justification are
essential. In practice, models trained on court decisions or general legislation often
underperform when applied to environmental law. Traditional NLP therefore remain
insufficient for the level of consistency, interpretability and precision required in this
context.
Large language models (LLMs) have taken natural language processing (NLP) to
another level by enabling general purpose language understanding and generation,
raising expectations regarding their potential contribution to legal analysis.
Their ability to perform zero-shot and few-shot reasoning makes them particularly
appealing for legal text processing, where annotated data is scarce [11]. Nevertheless,
LLMs remain limited by their frozen training knowledge, their poor generalization
to domain-specific queries, and, above all, their tendency to generate hallucinations,
the phenomenon whereby the model outputs factually incorrect answers [12, 13].
LLMs may confidently produce factually incorrect statements, lack up-to-date legal
knowledge, and/or are insensitive to jurisdictional nuances. Recent work [14, 15] have
demonstrated that, while LLMs can be valuable tools for information extraction, their
outputs often lack factual grounding. These studies show that even though LLMs have
remarkable linguistic capabilities, their ungrounded nature limits their reliability for
computing legal indicators, where every conclusion must be traceable and justified by
verifiable evidence.
Retrieval-Augmented Generation (RAG) addresses some of these issues by grounding
model outputs in external documents [16]. Instead of encoding facts directly in model
parameters, RAG systems retrieve relevant text snippets to support generation. How-
ever, standard RAG pipelines follow a fixed sequence of retrieval and generation steps
[17]. This rigidity limits their ability to dynamically adapt queries, refine retrieval
strategies, or evaluate the reliability of the intermediate results. Consequently, despite
their potential, conventional RAG systems still lack the flexibility and adaptivity
required to operate effectively in the legal field. Such approaches offer limited control
over why and how legal texts are selected and evaluated which constrains their suit-
ability for legal indicator construction.
Recent advances in agent-based systems extend RAG by introducing multiple reason-
ing steps, loops and workflows into these frameworks. These enhancements enable the
agent to review and refine its own outputs. In medicine, agentic RAG has improved
factual reliability and interpretability [18]. However, in the legal domain, and partic-
ularly for legal indicator computation, these methods remains underexplored despite
the need for explicit evidence tracking. Extending agentic RAG to legal contexts
represents a critical step toward building AI systems that can support trustworthy,
interpretable, and adaptive legal analysis.
To address these limitations, this paper focuses on the automated construction of legal
indicators from legal norms. We introduce N2I-RAG (From Norms to Indicators), an
agentic retrieval-augmented framework designed to compute legal indicators from col-
lections of digitized legal documents. It combines multi-agent coordination, semantic
embeddings, and self-adaptive retrieval mechanisms to generate contextualized and
traceable results. N2I-RAG formulates indicator computation as a structured infor-
mation extraction problem, in which relevant texts are identified, then semantically
3

aligned with Indicator queries, to be finally validated based on contextual relevance.
These informations are then synthesized in a structured and traceable form, allowing
experts to inspect how each indicator was derived. This approach aims to make legal
indicator computation both automatable and interpretable, bridging qualitative legal
analysis and quantitative monitoring.
The architecture is designed to be explainable and its outputs are interpretable in
an operational sense. Each agent in the system produces, alongside its output, an
explicit explanation that describes the reasoning process by which the result was
obtained, including the sources used and the criteria applied. In this work, explain-
ability denotes the availability of explanations at each stage of the pipeline, enabling
inspection of intermediate decisions, while interpretability denotes the alignment of
final outputs with the indicator definition and their direct linkage to identifiable legal
articles. Requiring agents to produce such explanations constrains generation and sup-
ports consistent, evidence-based outputs.
The main contributions of this paper are:
1.Agentic RAG architecture:A multi-agent pipeline that performs adaptive
retrievals, contextual evaluations, query disambiguation, and evidence-based gen-
eration.
2.Evaluation and case study:A comparative study and analysis of N2I-RAG’s
performance against baseline RAG systems on marine environmental law.
3.Dataset construction:A French marine environmental law corpus consisting of
10,596 legal articles drawn from five collections, covering legislation and regulations
from both scanned and digital sources, structured and indexed to support retrieval-
based analysis and indicator construction.
Together, these contributions show how N2I-RAG can support the consistent and
transparent computation of legal indicators across complex regulatory domains.
2 Methodology
The N2I-RAG framework converts natural language legal queries into traceable binary
decisions that populate an assessment grid representing legal indicators. The system
connects document analysis to indicator calculation through a structured, multi-agent
pipeline. Each step, from text extraction to decision-making, ensures that intermediate
reasoning is visible and verifiable. As illustrated in Figure 1, the process begins with
OCR (Optical Character Recognition) processing of the document. The cleaned text
is semantically indexed to enable accurate retrieval. An agentic reasoning loop then
handles retrieval, grading, generating, and validating before returning a binary result.
These outputs fill the evaluation grid, where each entry corresponds to a distinct legal
indicator. This design ensures that each computational step remains interpretable,
traceable, and meets the legal reliability standards required for policy and compliance
analysis.
4

Fig. 1Overview of the N2I-RAG Framework
2.1 Visual-Language Processing
Precise text extraction is vital for downstream legal analysis, especially when source
documents include scanned and degraded documents. Many legislative documents,
notably in developing countries, are only available as low-quality scans with stamps,
signatures, or inconsistent layouts. Conventional OCR often fails on these materials,
leading to missing or corrupted text. To address this, N2I-RAG uses a hybrid OCR
strategy built on a Visual–Language Model (VLM) that integrates a vision encoder
with a language decoder. This allows the system to capture both textual and visual
elements (layout, formatting, and embedded elements) in order to recover full legal
texts. The model corrects noisy segments, ignores irrelevant visual features, and gen-
erates missing text. Each document is then segmented into legal articles and enhanced
with a set of standardized metadata such as jurisdiction, publication date, revision
date, institutional source, and text type (law, regulation, directive,etc.). These articles
are encoded using the BGE-M3 model [11] , a multi-lingual, multi-function, multi-
granularity embedding model and indexed in ChromaDB for semantic search. This
approach guarantees that even imperfect documents provide usable data, producing a
comprehensive, coherent, and searchable database suitable for indicator generation.
5

2.2 Multi-Agent Framework
The multi-agent pipeline structures N2I-RAG’s reasoning process into verifiable stages.
Distributing tasks across specialized agents improves transparency and reduces cumu-
lative errors. Each agent plays a specialized and complementary role, while recording
its reasoning, so that errors and biases can be identified and addressed. This mod-
ular structure prevents information loss and reduces the risk of hallucination. The
pipeline includes eight different specialized agents. Table 1 provides an overview of
these agents, their roles, and their interactions. Collectively, these agents turn complex
natural language reasoning into a transparent workflow featuring factual consistency
and interpretability controls. In the following subsections, we detail how each agent
works individually, highlighting their specific contribution to controlling hallucinations
and improving response quality.
Table 1Description of agents used in the architecture
Agent Role Input Output
Metadata Retriever Identify and parse metadata
associated with a queryUser query Metadata
JSON
Context Retriever Extract relevant articles Query Potential
articles
Context Grader Evaluate the semantic relevance
and specificity of the retrieved
articlesRetrieved
articlesScore +
Explanation
Generator Agent Generate responses based on the
selected contextFiltered and
graded contextNatural
language
answer
Groundedness
GraderEvaluates whether each claim is
directly supported by cited contextGenerated
answer +
ContextScore +
Explanation
Answer Relevance
GraderCheck the consistency between the
generated response and the initial
requestGenerated
answer +
queryScore +
Explanation
Query
DisambiguatorReformulate ambiguous or general
queriesUser query Reformulated
query
Binary QA Agent Assign a binary output (0/1) for
the final evaluation gridGenerated
answerBinary label +
Explanation
2.2.1 Metadata Retriever
Metadata retrieval is an essential step in enriching the query and improving the accu-
racy of the results. Metadata such as date, jurisdiction, or thematic provide contextual
dimensions that complement purely textual information and enable more targeted
searches. In our framework, the module automatically identifies the metadata associ-
ated with a query and converts it to JSON, facilitating its use by subsequent agent.
6

This standardized structure ensures consistency and interoperability within the sys-
tem. Thus, the integration of metadata enhances the accuracy and contextualization
of the retrieval process, paving the way for more appropriate responses.
2.2.2 Context Retriever
Cosine similarity is a particularly suitable approach for retrieving relevant documents
in Natural Language Processing applications. When associated with ChromaDB and
its HNSW (Hierarchical Navigable Small World) index, it offers a combination of high
semantic accuracy and optimized performance at scale, which significantly improves
the efficiency of the search system.
This measure evaluates the directional alignment of vectors in semantic space, regard-
less of their magnitude. This property neutralizes the text length dependency, which
is essential when processing articles of mixed sizes and structures, a problem often
encountered in information retrieval systems.
By using this metric with an HNSW index in ChromaDB, we benefit from approximate
nearest neighbor (ANN) search, which makes the retrieval process fast and efficient,
even for large data collections.
In our system, each article in our database is preprocessed and transformed into a vec-
tor representation using the BGE-M3 embedding model. These vectors are indexed in
ChromaDB, which uses the HNSW index to perform cosine similarity searches. When
a query is submitted, it is also converted into a vector and compared to document
vectors using cosine similarity. The relevance score is calculated as shown by equation
1:
d= 1.0−P(A i×B i)P(A2
i)×P(B2
i)(1)
Therefore, by using this equation 1 as a distance function in ChromaDB’s HNSW
index, we optimize both semantic accuracy in article retrieval and the scalability and
efficiency of the process, ensuring relevant and contextually appropriate item selection.
2.2.3 Context Grader
Context grading is crucial for sorting out the truly useful content from the superficially
similar ones. Indeed, not all retrieved articles are equally relevant, and the linguistic
complexity of legal texts can introduce noise, affecting the quality of the generated
answers.
This agent grades each article based on two criteria: semantic relevance and speci-
ficity. Each grade is supported by a detailed explanation, ensuring transparency and
interpretability.
By selecting and filtering the most informative contexts and justifying these decisions,
this module guarantees the reliability and explainability of the data transmitted to
the following steps, ensuring that their reasoning is based exclusively on reliable and
contextually relevant information.
7

2.2.4 Generator Agent
The generator agent is the core of the process, as it produces the final response based
on the selected information. Controlled and contextualized generation is essential to
avoid hallucinations and ensure that responses accurately reflect the content of the
selected documents. This module relies exclusively on filtered and evaluated texts.
The generation process follows a strict set of rules to maintain consistency and fac-
tual accuracy while producing a coherent and understandable output. As a result,
this module is able to provide accurate and reliable answers that are grounded in a
validated context.
2.2.5 Groundedness Grader
We introduce a groundedness agent designed to check the factual reliability of
responses and limiting hallucinations. A generative model can produce convincing
statements without any source reference. However, without explicit grounding, the
reliability of these assumptions is questionable, jeopardizing the credibility of the sys-
tem. This module verifies that each statement in the final answer is explicitly linked to
a text passage from the retrieved documents. Any statements without a source refer-
ence are flagged as unreliable. This approach allows a direct link to be traced between
the source data and the generated content, enhancing the transparency of the sys-
tem. Therefore, the grounding assessment acts as a safeguard against hallucinations,
ensuring that each response element is based on documented evidence.
2.2.6 Answer Relevance Grader
Assessing the relevance of the response ensures that the output generated by the
system remains consistent with the user’s original intention. A response could appear
factually correct and yet be irrelevant to the original query. Although this is not
strictly a factual hallucination, it significantly reduces the usability of the system. This
module systematically evaluates whether the initial query and the generated response
are coherent. Hence, this module helps to ensure that the generated answers are not
only correct, but also relevant and useful to the initial query.
2.2.7 Query Disambiguator
Query disambiguation and rewriting improves system performance by optimizing
query formulation[19]. Users may formulate queries that are ambiguous, incomplete,
or too general. If not reformulated, retrieval is likely to be inefficient or produce too
much noise. This module automatically reformulates queries while preserving their
initial intent while making them more specific and operational for retrieval. It thus
reduces noise and improves the relevance of candidate articles. Query disambiguation
is therefore an effective means of increasing the accuracy and coverage of the search
system.
8

2.2.8 Binary QA Agent
The binary response module plays a strategic role in our framework, as it translates
the responses generated into decisions suitable for integration into a 0/1 evaluation
grid. In a setting where AI results are feeding a legal evaluation system, clarity and
standardization of decisions are essential. Generative systems often produce nuanced
or ambiguous responses that are difficult to encode in a binary format. However, to
integrate them into a normative grid and ensure comparability between instances, we
need to transform these nuances into a discrete signal (1 if the response is entirely
affirmative, 0 otherwise). This step requires a level of decision-making rigor that goes
beyond text generation.
The agent relies on two sets of instructions to guide its decision-making:
1. Total affirmation vs. total negation: The response is evaluated as either entirely
affirmative (1) or otherwise (0).
2. Full coverage of the question: the agent assigns a score of 1 only if the answer covers
all parts of the question, without omission or ambiguity. Any partial, contradictory,
or incomplete answer automatically receives a score of 0.
These guidelines are implemented via specific prompts, and the agent returns its
verdict in JSON format containing both the binary score and a textual explanation.
The score is fed into the evaluation grid, while the explanation provides essential trace-
ability for expert review and decision-making justification.
Consequently, the binary response module constitutes a bridge joining the flexibility
of natural language and the normative strictness of the legal evaluation grid. By bind-
ing the system’s outputs to a standardized binary format, it ensures the reliability,
transparency, and applicability of its results as AI contributes directly to the legal
evaluation.
The N2I-RAG framework combines multi-modal document processing and llm-
based agents to translate unstructured legal texts into meaningful binary indicators.
In each phase, from document processing to decision-making, N2I-RAG features a
transparent and traceable workflow aligned with legal reliability standards.The inte-
gration of semantic retrieval, multi-agent and ground-based validation controls makes
N2I-RAG stand out from other conventional solutions that prioritize performance over
interpretability. This methodology paves the way for automated, evidence-based legal
indicator computation, thus filling the existing gap between formal legal texts and
quantifiable/measurable compliance outcomes.
3 Experimental Setup and Case Study: Marine
Environmental Law
3.1 Case Study Focus
We evaluated N2I-RAG on a case study in marine environmental law, to assess its abil-
ity to automatically calculate reliable legal indicators from heterogeneous documents.
This domain is particularly well suited for evaluation due to the growing demand from
international organizations for standardized legal indicators to monitor environmental
9

protection. Marine environmental law is constantly expanding in both scope and com-
plexity as governments and international institutions tackle urgent ecological crises,
including climate change, biodiversity loss, and pollution. The process of evaluating
these legal measures remains time-consuming and requires in-depth expertise, making
it a appropriate testing ground for norm-to-indicator computation. The study focuses
on two prohibitions with measurable legal impact:
•The ban on plastic bags.
•The ban on bottom trawling.
Each ban is assessed through an evaluation grid for legal indicators covering 11 stan-
dardized questions, designed jointly by legal, ecological, and economic experts, detailed
in the Table 2. These questions measure the applicability, characteristics, scope, and
operationalization of the prohibition. This setup therefore provides a clear and repro-
ducible setting for testing N2I-RAG’s performance in a real-world application, to
translate legal norms into structured, indicator-level outputs.
Table 2: The legal indicator is an agregation of 11 metrics. Blue and green metrics quantify the
reach of the ban (scope extent and exception scarcity, respectively), while red and yellow
represent operationalization (penalties and control mechanisms).
Evaluation metrics Output
1. The ban is specified by a legal article
0 or 12. The ban is nationwide in scope
3. The ban is permanent in nature
4. Details of banned activities is documented
5. There are no exceptions to the rule
6. Exemptions are restricted to a few specific cases
7. A monetary fine is imposed
8. A jail sentence is outlined
9. A designated authority is tasked with enforcement
10. A control process with a defined duration is established
11. A location-specific control procedure is detailed
10

3.2 Dataset Construction
A consistent and high-quality corpus is essential for testing the reliability of retrieval-
augmented legal analysis. Legal documents are typically available as scanned images,
with mixed quality and layout. Accurate computation of legal indicators requires
both clean textual content as well as meaningful metadata. Our scope covers African
countries where French is an administrative language. Documents were acquired from
FAOLEX1, a database maintained by the Food and Agriculture Organization of
the United Nations. It contains a vast collection of legislative, regulatory, and policy
texts covering areas such as agriculture, natural resources, and the environment on a
global scale. This source was chosen for its comprehensive coverage. This study focuses
on African countries where French is used as an administrative language. We pro-
cessed the corpus through the N2I-RAG preprocessing pipeline described in Section
2.2. All documents were processed and cleaned using the hybrid OCR system and
the Visual–Language Model (VLM), which corrected any noise present in the text.
They were then segmented into articles, enhanced with the standardized metadata,
and embedded using BGE-M3 representations. Vectors were stored in ChromaDB for
semantic retrieval. Through this process, we have built a comprehensive database
consisting of five distinct collections, totaling 10,596 articles, structured and indexed
for research, analysis, and automated legal reasoning tasks, as shown in the Table
3. We intentionally introduced three additional prohibitions to obtain a total of five
collections in addition to the initial two, in order to challenge the retrieval process.
Therefore, our vector database does not only contain articles related to our case study.
Table 3Number of Articles in the Corpus by Country and Ban
CountryBan
Coastal
ConstructionWhaling Bottom
TrawlingHydrocarbon
dischargePlastic
Bags
Algeria 43 107 67 1,052 -
Benin 109 223 92 762 -
Cameroon - 105 105 93 24
Congo - 155 16 246 11
Ivory Coast 66 121 362 136 14
Djibouti - 12 - - -
France - 13 - - -
Gabon 163 18 63 163 25
Guinea 1,129 469 33 2,538 -
Morocco 56 5 24 59 15
Mauritania 701 101 67 - 8
Niger - - - - 15
DRC - 91 - - 17
Senegal 363 138 - - 42
Togo - 28 - - 18
Tunisia 100 60 - 147 6
Total2,730 1,646 829 5,196 195
1https://www.fao.org/faolex
11

This provides legally structured and semantically searchable references, offering a
steady basis for indicator computation.
3.3 Models and Implementation
All experiments were executed locally to ensure reproducibility and control over
resources. The implementation relies on LangChain and LangGraph for agent orches-
tration, and on Ollama for language model execution. All agents are implemented as
specialized LLMs, with the exception of the retrieval module, which uses ChromaDB
with bge-m3 vector representations and a search limited to the 10 closest results
(top results=10).
We tested three state-of-the-art open-source large language models from different fam-
ilies: Llama3.2:3B, Qwen3:8b, and Mistral-Nemo:12b, in order to demonstrate the
robustness of the proposed architecture.
Two generating settings were adopted:
•For deterministic tasks (evaluation, scoring, binary decision), a temperature of 0
was used.
•For tasks requiring reformulation or creativity (rewriting queries, alternative
generation), a temperature of 0.9 was used.
3.4 Evaluation Metrics
Performance was assessed according to standard binary classification metrics, but
interpreted from a legal perspective:
•Accuracy:Overall rate of correct answers (whether or not a law is correctly
identified).
•Sensitivity (Recall):The model’s ability to correctly identify situations where a
law exists. A low recall would mean that effective laws are being ignored, which
undermines the reliability of the legal diagnosis.
•Specificity:The system’s ability to correctly identify cases where no law exists.
This limits false positives, where a non-existent law is identified, which could be
misleading
•Precision:Rate of responses correctly asserting the existence of a law among all
those indicating that a law exists.
•F1-Score:Balance between precision and recall, reflecting the robustness of the
system in the face of heterogeneous legal contexts.
•Balanced Accuracy:Average between sensitivity and specificity, helpful in cases
of unbalanced data distribution (existing vs. non-existing law)
•False Positive Rate (FPR):Rate of falsely identifying a rule as existing.
•False Negative Rate (FNR):Rate of falsely identifying an existing rule as not
existing.
This interpretation allows technical metrics to be grounded in a legal and operational
interpretation, which is essential for regulatory assessment.
12

3.5 Experimental Protocol
The experiments were designed to evaluate both the global performance of N2I-RAG
and the contribution of its key elements. By comparing full and reduced configurations,
we can detect the impact of the different pipeline modules, especially those that reduce
hallucinations.
Three configurations were tested:
1.Full Pipeline:all agents are activated.
2.Retrieval-only Baseline:Only vector search via ChromaDB is used, without
LLM agents.
3.Pipeline without hallucination control agents:Keep the main agents, except
those dedicated to check relevance, groundedness, and coherence.
Each configuration was run across the same indicators and both case study topics,
ensuring a consistent comparison.
This ablation approach allows quantitative comparison and qualitative inspection,
highlighting how agentic control improves factual grounding and stability, supporting
a clear assessment of N2I-RAG’s overall reliability in legal contexts.
4 Results and Discussion
4.1 Global Performance of N2I-RAG
The experimental results (Figure 2) show how the integration of the N2I-RAG archi-
tecture significantly improves the quality of results across all LLM variants, with
Mistral-Nemo:12B standing out in particular from the Qwen3:8B and Llama3.2:3B
models. The multi-agent pipeline reduces hallucinations, reinforces factual grounding,
and ensures better consistency between retrieved legal texts and generated responses.
As shown in Figure 2, Mistral Nemo with N2I-RAG achieves a Balanced Accuracy
Fig. 2Comprehensive and comparative metrics heatmap
13

of 0.933 and an F1-Score of 0.943, compared to only 0.768 without the architecture.
Llama and Qwen also improve with N2I-RAG (0.77 and 0.784 in F1-Score, respec-
tively), but remain behind. Specificity particularly illustrates this contribution: Mistral
Nemo goes from 0.612 to 0.919.
These results confirm that the performance gains of N2I-RAG do not stem from the
individual capacity of a model, but rather from its structured orchestration. The
framework remains effective across architectures, proving that the proposed approach
is LLM-agnostic but that its potential is maximized with more complex models, where
combining representation ability and multi-agent control produces optimal output.
4.2 Comparative Analysis Across Models
N2I-RAG maintains stable performance across distinct LLM families, confirming its
portability and robustness. Because the proposed framework separates reasoning con-
trol from model output generation, it can adapt to models of different sizes and
linguistic capacities. The architecture does not rely on memorized facts, but on active
retrieval and validation loops. While all models benefit, Mistral-Nemo consistently
delivers the best results, suggesting that models with larger contextual windows are
better at leveraging multi-agent interactions. Qwen and Llama still outperform their
baselines, confirming that N2I-RAG’s gains are not model-dependent but rather result
from structured orchestration. This cross-model consistency supports N2I-RAG’s gen-
eralizability. It can be integrated into various LLM setups, enabling scalable and
comprehensible automation for legal observatories.
4.3 Ablation Study
To assess the contribution of the core component, we conducted an ablation study
comparing the full pipeline, a variant without hallucination control, and a simple base-
line. This analysis clarifies how each design component supports factual consistency
and balanced performance
Fig. 3Ablation test results for Mistral-Nemo by configuration
14

Table 4Ablation Tests (Mistral-Nemo)
Ban Configuration Accuracy Precision Recall Specificity F1-
ScoreBalanced
Acc.
Plastic
bagFull 0.975 1.000 0.962 1.000 0.980 0.981
Without-Hall 0.843 0.862 0.848 0.836 0.855 0.842
Baseline 0.777 0.760 0.864 0.673 0.809 0.768
Bottom
trawlingFull 0.872 0.821 0.914 0.837 0.865 0.876
Without-Hall 0.782 0.765 0.743 0.814 0.753 0.778
Baseline 0.564 0.512 0.600 0.535 0.553 0.567
Figure 3 and Table 4 present the comparative results across three configurations
obtained with the Mistral-Nemo model, selected for its superior performance. We note
that the balanced accuracy for the plastic bag ban increased from 0.768 (Baseline)
to 0.842 (Without-Hall) and then to 0.981 (Full). Similarly, for bottom trawling ban,
performance improved from 0.567 (Baseline) to 0.876 (Full). Specificity is the most
affected metric: with N2I-RAG, Nemo achieves 1.000 (plastic) and 0.837 (trawling),
compared to only 0.836 and 0.535 without hallucination control.
These findings demonstrate that performance does not depend exclusively on the
quality of LLM representations, but also on the design of the evaluation and con-
trol pipeline. The proposed multi-agent approach provides a key regulatory layer,
particularly in sensitive contexts where legal false positives must be minimized.
4.4 Legal Error Analysis
The legal-oriented metrics enhance the relevance of the assessment by identifying two
types of errors with different regulatory implications.
For the ban on plastic bags, Mistral-Nemo with N2I-RAG achieves Recall = 0.962 and
Specificity = 1.000, illustrating a remarkable balance between recognition and relia-
bility. Meanwhile, variants without anti-hallucination agents see their specificity drop
to 0.535 in the case of trawling, which is a worrying false positive rate for legal appli-
cations.
The proposed architecture therefore provides a legal robustness guarantee by reducing
critical biases related to false positives and false negatives. This outcome is partic-
ularly decisive when it comes to analyzing legal prohibitions in environmental law,
where an erroneous claim can have serious regulatory and political implications.
Technical performance alone does not capture the practical reliability of legal AI sys-
tems. N2I-RAG was therefore analyzed from a legal-error perspective, focusing on the
types and implications of false predictions. Despite its robustness, N2I-RAG’s remain-
ing errors reveal important insights about the nature of legal data and the boundaries
of automated interpretation. For instance, in assessing the ban on plastic bags, the
errors observed are not solely due to technical limitations of language models, but also
reflect specific challenges related to the interpretation of the standard corpus used.
Two main categories of legal errors have been identified:
15

1.Data gaps:In some cases (e.g., Ivory Coast), the sanctions associated with the
ban are not mentioned in the articles accessible via the database, but appear in a
separate decree which is not listed in Faolex. The lack of this secondary document
led the system to incorrectly conclude that there were no sanctions.
From a technical standpoint, this is not an error but a positive outcome, as the
system did not produce incorrect information. However, from a legal perspective,
this highlights the challenges of evaluating a law based on partial sources. It also
shows the need to consider supplementary texts (like decrees and implementing
orders), while emphasizing their limited online accessibility.
2.Transitional misinterpretation:The model interpreted the phase-out period
for plastic bags in the Democratic Republic of Congo as a temporal constraint on
the ban. However, from a legal perspective, this measure is transitional and aims
to regulate the progressive implementation of the ban.
In legal terms, false positives mean over-claiming a regulation that does not exist,
while false negatives omit and ignore an actual ban. Both have judicial consequences:
the former can mislead compliance assessments, while the latter can hide regulatory
progress.
These outcomes confirm that automating legal indicator computation requires both
reliable data sources and structured interpretability mechanisms. N2I-RAG effectively
reproduces expert reasoning, but depends on the comprehensiveness of accessible legal
texts. Its results thus provide a transparent, evidence-based foundation for legal obser-
vatories aiming to systematically monitor norms, extending the conceptual foundations
proposed by Billant (2023)[2] to an AI-supported operational workflow.
5 Conclusion
This paper introduced N2I-RAG, an agentic retrieval-augmented generation frame-
work designed to compute legal indicators in a reliable, interpretable, and automatable
way. It shows that structured multi-agent reasoning can bridge the gap between open-
textured legal language and standardized assessment grids used in policy and judicial
evaluation.
N2I-RAG integrates search, generation, and validation within a modular pipeline.
Each agent performs a specific function, whether it be filtering, retrieving, grading,
or producing binary legal results. This structure allows the system to analyze com-
plex legal documents while maintaining full traceability of its decisions and reasoning
process. Unlike conventional RAG systems that rely on a single generation step, N2I-
RAG applies multi-level control to align the model’s linguistic output with the logic
of legal indicators.
Experimental results demonstrate that the complete N2I-RAG architecture substan-
tially outperforms its simplified variants, especially in sensitivity and specificity, two
metrics essential for legal interpretation. Mistral-Nemo 12B achieved the highest per-
formance, while results from Qwen and LLaMA confirm that the framework remains
model-agnostic. Additionally, error analysis reveals that the challenges encountered
arise primarily from the subtleties of the law (e.g., transitional periods, gaps in doc-
umentation, or implicit exceptions) rather than technical shortcomings , thus paving
16

the way for further interdisciplinary work.
Beyond its technical performance, N2I-RAG contributes to the broader goal of
automating legal observatories. It provides a foundation for continuous, transparent,
and scalable monitoring of environmental law. This would offer ongoing legal moni-
toring, enhance visibility, and provide a practical decision-making tool for researchers,
legal professionals, and even the general public. For future work, we intend to extend
this framework to multilingual and multi-jurisdictional contexts, integrating richer
databases and real-time data sources to support more adaptive and inclusive legal
indicator computation.
Acknowledgements.This work has been supported by AIME and ISblue project,
Interdisciplinary graduate school for the blue planet (ANR-17-EURE-0015) and co-
funded by a grant from the French government under the program”Investissements
d’Avenir”embedded in France 2030.
References
[1] Prieur, M., Bastin, C., Mekouar, A.: Measuring the Effectivity of Environmen-
tal Law. Peter Lang Verlag, Bruxelles, Belgique (2021). https://doi.org/10.3726/
b18559 .https://www.peterlang.com/document/1114411
[2] Billant, O.: Lib´ erer les oc´ eans du plastique par le droit : mirage ou horizon ? Une
exp´ erimentation en droit num´ erique compar´ e sur les rives de l’oc´ ean Atlantique.
Theses, Universit´ e de Bretagne occidentale - Brest (November 2022). https://
theses.hal.science/tel-04052691
[3] Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., Androutsopou-
los, I.: LEGAL-BERT: The muppets straight out of law school. In: Cohn,
T., He, Y., Liu, Y. (eds.) Findings of the Association for Computational
Linguistics: EMNLP 2020, pp. 2898–2904. Association for Computational Lin-
guistics, Online (2020). https://doi.org/10.18653/v1/2020.findings-emnlp.261 .
https://aclanthology.org/2020.findings-emnlp.261/
[4] Oliveira, V., Nogueira, G., Faleiros, T., Marcacini, R.: Combining prompt-based
language models and weak supervision for labeling named entity recognition on
legal documents. Artificial Intelligence and Law33(2), 361–381 (2025) https:
//doi.org/10.1007/s10506-023-09388-1
[5] Wang, J., Wang, K., Weng, Y., Li, X.: Adversarial training flat-lattice transformer
for named entity recognition of chinese legal texts. Artificial Intelligence and Law
(2025) https://doi.org/10.1007/s10506-025-09476-4
[6] Costa, Y.D.R., Oliveira, H., Nogueira, V., Massa, L., Yang, X., Barbosa, A.,
Oliveira, K., Vieira, T.: Automating petition classification in brazil’s legal system:
a two-step deep learning approach. Artificial Intelligence and Law33(1), 227–251
(2025) https://doi.org/10.1007/s10506-023-09385-4
17

[7] Al Mouatamid, Y., Zahir, J., Bonnin, M., Mousannif, H.: Assessing ocean’s
legal protection using ai: A new dataset and a bert-based classifier. In: Legal
Knowledge and Information Systems. Frontiers in Artificial Intelligence and
Applications, pp. 263–268. IOS Press, The Netherlands (2023). https://doi.org/
10.3233/FAIA230972
[8] Araujo, P.H., Campos, T.E., Oliveira, R.R.R., Stauffer, M., Couto, S., Bermejo,
P.: Lener-br: A dataset for named entity recognition in brazilian legal text. In:
Villavicencio, A., Moreira, V., Abad, A., Caseli, H., Gamallo, P., Ramisch, C.,
Gon¸ calo Oliveira, H., Paetzold, G.H. (eds.) Computational Processing of the
Portuguese Language, pp. 313–323. Springer, Cham (2018)
[9] Chen, Y., Sun, Y., Yang, Z., Lin, H.: Joint entity and relation extraction for
legal documents with legal feature enhancement. In: Scott, D., Bel, N., Zong,
C. (eds.) Proceedings of the 28th International Conference on Computational
Linguistics, pp. 1561–1571. International Committee on Computational Lin-
guistics, Barcelona, Spain (Online) (2020). https://doi.org/10.18653/v1/2020.
coling-main.137 .https://aclanthology.org/2020.coling-main.137/
[10] Pais, V., Mitrofan, M., Gasan, C.L., Coneschi, V., Ianov, A.: Named entity
recognition in the Romanian legal domain. In: Aletras, N., Androutsopoulos, I.,
Barrett, L., Goanta, C., Preotiuc-Pietro, D. (eds.) Proceedings of the Natural
Legal Language Processing Workshop 2021, pp. 9–18. Association for Computa-
tional Linguistics, Punta Cana, Dominican Republic (2021). https://doi.org/10.
18653/v1/2021.nllp-1.2 .https://aclanthology.org/2021.nllp-1.2/
[11] Breton, J., Billami, M.M., Chevalier, M., Nguyen, H.T., Satoh, K., Trojahn,
C., Zin, M.M.: Leveraging llms for legal terms extraction with limited anno-
tated data. Artificial Intelligence and Law (2025) https://doi.org/10.1007/
s10506-025-09448-8
[12] Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto,
A., Fung, P.: Survey of hallucination in natural language generation. ACM
Comput. Surv.55(12) (2023) https://doi.org/10.1145/3571730
[13] Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng,
W., Feng, X., Qin, B., Liu, T.: A survey on hallucination in large language mod-
els: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems43(2), 1–55 (2025) https://doi.org/10.1145/3703155
[14] Faria, J., Xie, H., Steffek, F.: Information extraction from employment tribunal
judgments using a large language model. Artificial Intelligence and Law (2025)
https://doi.org/10.1007/s10506-025-09443-z
[15] Mouatamid, Y.A., Bonnin, M., Zahir, J.: Elevating legal understanding: A dedi-
cated instruction dataset and an optimized llm for legal information extraction to
18

support the juridical interpretation of marine environmental law. In: 2025 Inter-
national Conference on Intelligent Systems: Theories and Applications (SITA),
pp. 1–6 (2025). https://doi.org/10.1109/SITA67914.2025.11273384
[16] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., K¨ uttler, H.,
Lewis, M., Yih, W.-t., Rockt¨ aschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive nlp tasks. In: Proceedings of the 34th Interna-
tional Conference on Neural Information Processing Systems. NIPS ’20. Curran
Associates Inc., Red Hook, NY, USA (2020)
[17] Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang,
M., Wang, H.: Retrieval-Augmented Generation for Large Language Models: A
Survey (2024). https://arxiv.org/abs/2312.10997
[18] Zhao, X., Liu, S., Yang, S.-Y., Miao, C.: MedRAG: Enhancing Retrieval-
augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare
Copilot (2025). https://arxiv.org/abs/2502.04413
[19] Min, S., Michael, J., Hajishirzi, H., Zettlemoyer, L.: AmbigQA: Answering
ambiguous open-domain questions. In: Webber, B., Cohn, T., He, Y., Liu, Y.
(eds.) Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pp. 5783–5797. Association for Computational
Linguistics, Online (2020). https://doi.org/10.18653/v1/2020.emnlp-main.466 .
https://aclanthology.org/2020.emnlp-main.466/
19