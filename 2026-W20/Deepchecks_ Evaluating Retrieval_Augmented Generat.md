# Deepchecks: Evaluating Retrieval-Augmented Generation (RAG)

**Authors**: Assaf Gerner, Netta Madvil, Nadav Barak, Alex Zaikman, Jonatan Liberman, Liron Hamra, Rotem Brazilay, Shay Tsadok, Yaron Friedman, Neal Harow, Noam Bresler, Shir Chorev, Philip Tannor, Lior Rokach

**Published**: 2026-05-14 07:27:50

**PDF URL**: [https://arxiv.org/pdf/2605.14488v1](https://arxiv.org/pdf/2605.14488v1)

## Abstract
Large Language Models (LLMs) augmented with Retrieval-Augmented Generation (RAG) techniques are revolutionizing applications across multiple domains, such as healthcare, finance, and customer service. Despite their potential, evaluating RAG systems remains a complex challenge due to the stochastic nature of generated outputs and the intricate interplay between retrieval and generation components. This paper introduces Deepchecks, a comprehensive framework tailored for evaluating RAG applications. Deepchecks' evaluation framework addresses RAG applications evaluation through a multi-faceted approach, root cause analysis and production monitoring. By ensuring alignment with application-specific requirements, Deepchecks framework provides a robust foundation for assessing reliability, relevance, and user satisfaction in RAG systems.

## Full Text


<!-- PDF content starts -->

Deepchecks: Evaluating Retrieval-Augmented
Generation (RAG)
Assaf Gerner1, Netta Madvil1, Nadav Barak1, Alex Zaikman1, Jonatan
Liberman1, Liron Hamra1, Rotem Brazilay1, Shay Tsadok1, Yaron Friedman1,
Neal Harow1, Noam Bresler1, Shir Chorev1, Philip Tannor1, and Lior Rokach2
1Deepchecks, Ramat Gan, Israel
2Ben-Gurion University, Beer Sheva, Israelliorrk@bgu.ac.il
Abstract.Large Language Models (LLMs) augmented with Retrieval-
Augmented Generation (RAG) techniques are revolutionizing applica-
tions across multiple domains, such as healthcare, finance, and customer
service. Despite their potential, evaluating RAG systems remains a com-
plex challenge due to the stochastic nature of generated outputs and
the intricate interplay between retrieval and generation components.
This paper introduces Deepchecks, a comprehensive framework tailored
for evaluating RAG applications. Deepchecks’ evaluation framework ad-
dresses RAG applications evaluation through a multi-faceted approach,
root cause analysis and production monitoring. By ensuring alignment
with application-specific requirements, Deepchecks framework provides a
robustfoundationforassessingreliability,relevance,andusersatisfaction
in RAG systems.
1 Introduction
Retrieval-Augmented Generation (RAG) is a framework that enhances the ca-
pabilities of Large Language Models (LLMs) by integrating external knowledge
sources into the generation process. By combining the text generation prowess
of LLMs with information retrieval functions, RAG enables the production of
precise and contextually relevant information. This approach addresses several
critical limitations of traditional LLMs, such as their propensity to generate
generic responses or false claims, often referred to as hallucinations.
The RAG process typically involves the following steps:
– External Data Curation:External data from various sources, such as
APIs,databases,ordocumentrepositories,isconvertedintonumericalrepre-
sentationsusingembeddinglanguagemodelsandstoredinavectordatabase.
– Information Retrieval:Upon receiving a user query, the RAG system
retrieves relevant information from the external knowledge base.
– Augmented Generation:The retrieved information is combined with the
user query and fed into the LLM, which uses this augmented context along
with its training data to generate a response.arXiv:2605.14488v1  [cs.AI]  14 May 2026

Fig. 1.: The core principles of a RAG system: (1) Offline – store external data, serving
as a source of truth, in a vector database. (2) Online – augment the user query by
retrieving and incorporating relevant documents from the database.
Despite the promise of RAG systems, evaluating their performance remains
a challenge. RAG systems comprise multiple interconnected components, each
requiring unique evaluation methodologies. In turn, this complicates end-to-end
assessment; i.e., providing a comprehensive unified scoring for RAG pipelines.
Addressing these challenges demands a multifaceted and reference-free ap-
proach that integrates automated metrics capable of capturing the unique char-
acteristics of RAG systems. The evaluation of RAG applications has become a
critical area of research, focusing on systematic methodologies to assess model
capabilitiesacrossdimensionssuchasalignment,robustness,anddomain-specific
performance. Ensuring alignment with user needs, domain-specific requirements,
and performance goals is essential for advancing this technology.
This paper explores the challenges associated with evaluating RAG appli-
cations and introduces the Deepchecks framework as a comprehensive solution,
assisting evaluation, iterating system improvements, and production monitoring.
2

By addressing key limitations in existing evaluation methodologies, Deepchecks
offers a robust and modular approach to assess RAG pipelines effectively, ensur-
ing their reliability and practical applicability.
2 Related Work
Evaluating Retrieval-Augmented Generation (RAG) systems has become a crit-
ical focus in modern NLP research, spurring the development of several frame-
worksdesignedtoproviderobust,scalable,andtask-agnosticevaluationmethod-
ologies. This section reviews prior work, organizing the discussion around evalu-
ation metrics and methods, domain-specific and general frameworks, and emerg-
ing techniques.
2.1 Retrieval Evaluation
2.1.1 Classic Methods
Retrieval evaluation using classic methods relies on well-established metrics
and benchmarks to assess both retrieval quality and generation fidelity. For in-
stance, retrieval metrics such as NDCG@k and MRR@k evaluate the ranking
effectiveness of retrieved results. NDCG@k evaluates the ranking quality of re-
trieved results, rewarding highly relevant items appearing at the top. It is useful
when multiple levels of relevance matter and the overall ranking order is impor-
tant. MRR@k focuses on the position of the first relevant result, making it ideal
for tasks where retrieving the correct answer quickly is crucial. While NDCG@k
is preferred for assessing all retrieved results, MRR@k is best suited for scenarios
where finding the first relevant result is the priority.
Effectiveevaluationalsorequiresstandardizationintheformofrobustbench-
marks. For instance, the BEIR and MS-Marco benchmarks facilitate compre-
hensive assessment, standardization, and real-world applicability of retrieval-
augmentedgenerationsystems.BEIR(BenchmarkingIR)isawidelyusedbench-
mark for evaluating information retrieval tasks, BEIR provides a standardized
frameworkwithdiversedatasets,supportingthoroughtestingacrossmultipledo-
mains. MS-Marco focuses on question-answering and ranking tasks, MS-Marco
features large-scale real-world queries, enabling the evaluation of retrieval per-
formance in practical settings. By leveraging these benchmarks, researchers can
enhance the effectiveness and reliability of RAG systems.
2.1.2 LLM-based evaluation
LLM-based evaluation, particularly the LLM-as-a-Judge approach, offers a
powerful and flexible method for assessing retrieval due to its ability to capture
nuanced aspects of textual data. The evaluator LLM is provided with specific
criteria and prompted to assess various aspects of the retrieved content.
3

Several automated metrics powered by LLMs have emerged for specific re-
trieval evaluation dimensions.Context relevanceassessment measures the pro-
portion of retrieved documents actually needed to answer the query, calculated
throughLLM-basedsentence-levelnecessityanalysis[12].Multimodal recalleval-
uation, as implemented in [20], uses vision-language models to assess product
image-text alignment in e-commerce retrieval scenarios. For multilingual sys-
tems, [20] developed a framework where LLMs generate localized annotation
guidelines before performing culture-aware relevance judgments.
Using LLMs as judges for evaluation comes with several limitations. Their
assessments can be subjective, leading to variability and potential biases in
judgment. Additionally, the computational cost of running multiple LLMs for
evaluation is high, making large-scale assessments resource-intensive. Ensuring
consistency across different prompts and models is another challenge, as varia-
tions in responses can lead to inconsistent evaluations, reducing the reliability
of the assessment process. Studies show that without proper calibration, LLM
judges can exhibit position bias favoring earlier documents in retrieval lists [7].
Hybrid approaches combining LLM scoring with traditional metrics like BM25
termoverlaphelpmaintaingroundinginlexicalrelevance[9].Recentbenchmarks
indicate that GPT-4-based evaluation achieves 0.81 Spearman correlation with
human judgments when using chain-of-thought prompting for retrieval assess-
ment [5].
2.2 Generation Evaluation
2.2.1 Reference-based Evaluation
For assessing generated responses, BLEU, ROUGE, and METEOR are com-
monly used to compare generated text with reference outputs, capturing lexical
overlap and coherence. However, these classic methods have limitations when
applied to RAG systems, as they often fail to account for contextual alignment,
factuality, and nuanced semantic correctness, necessitating the development of
more specialized evaluation frameworks.
BERTScore is a neural-based metric that evaluates the quality of generated
text by comparing it to reference outputs using contextual embeddings rather
than exact word overlap. Unlike traditional metrics like BLEU and ROUGE,
which rely on n-gram matching, BERTScore leverages pre-trained transformer
models (such as BERT) to compute token-level embeddings and measure their
similarity via cosine similarity. This approach allows BERTScore to capture se-
mantic meaning, making it particularly useful for evaluating RAG-generated
responses where lexical variations do not necessarily imply a difference in intent.
By going beyond surface-level string matching, BERTScore provides a nuanced
assessment of how well a generated response aligns with its reference in terms
of meaning and relevance.
Utilizing LLM-as-a-judge methods leverages models’ robust semantic capa-
bilities to compare generated responses against reference texts through dynamic
4

reasoning processes [15], usually surpassing performance of classic methods and
BERTScore (which tends to utilize small language models - SLMs).
Modern implementations employ three principal LLM-judge architectures for
reference-based assessment:
– Cross-attention Scoring: The judge LLM analyzes alignment between
reference and generated text through attention mechanism analysis, partic-
ularly effective for technical documentation verification [1].
– Chain-of-Thought Comparison: The model generates step-by-step rea-
soning comparing semantic equivalence between reference and response [2].
– Multi-Aspect Rubric Evaluation: LLM judges score outputs against ref-
erence texts using predefined dimensional rubrics (e.g., factual consistency,
stylistic match) [6].
Forinstance,theAmazonBedrockRAGEvaluationServiceexemplifiesproduction-
grade implementation, combining reference-based LLM judging with automatic
rubric generation [1]. Their three-phase process:
1.Reference Analysis: LLM extracts key claims and semantic structures
from reference text
2.Cross-Examination:Generatedresponseischeckedagainstreference-derived
constraints
3.Divergence Scoring: Model quantifies deviations using context-aware sim-
ilarity measures
This approach achieved 92% precision in detecting factual inconsistencies
in legal document analysis benchmarks, compared to 78% for BERTScore-based
evaluations [1]. The system’s explainability features provide natural language ra-
tionales for each discrepancy, addressing a critical limitation of black-box neural
metrics.
2.2.2 Reference-free Evaluation
While reference-based metrics, relying on human-annotated ground truth
answers, remain valuable, they are often expensive to obtain and may not fully
capturethenuancesofRAGperformance,particularlyinopen-domainscenarios.
This has led to a surge of interest inreference-freeevaluation metrics. These
metrics offer the advantage of assessing RAG outputs without requiring pre-
defined gold standard answers.
Reference-free metrics can consider different combinations of input and out-
put elements, such as assessing the relevance of the generated answer to the
input query, evaluating whether the generated answer is grounded in the re-
trieved context, or measuring the overall quality and coherence of the generated
text itself. In essence, reference-free metrics provide a practical and scalable way
to evaluate RAG systems, offering valuable insights into different aspects of their
performance.
5

Roughly speaking, there are three main types of reference-free metrics for
evaluating the generation phase of a RAG system:
1. Answer Relevance and Quality:RAG evaluation assesses answer rel-
evance and quality without requiring a ground truth. Metrics include sentence
embedding similarity (e.g., Sentence-BERT [13]) and LLM-based scoring for rel-
evance, helpfulness and completeness.
2.ContextGroundednessandFaithfulness:Answersshouldbegrounded
in retrieved context. Metrics evaluate factuality by checking if the answer is en-
tailed by the context (e.g., using NLI models [4, 18]) and measure how much the
answer relies on the retrieved content.
3. Human Alignment and Safety:Evaluating alignment ensures the sys-
tem follows instructions and user intent, using compliance scoring and user feed-
back simulation (e.g., Pistis-RAG [3]). Safety measures include bias detection,
sentiment analysis, and content moderation to prevent harmful outputs.
2.3 Holistic and Multi-Dimensional Evaluation Benchmarks
Evaluating RAG systems requires holistic frameworks that integrate retrieval
and generation assessment while ensuring scalability and task generalizability.
Recent advancements have introduced unified, automated, and modular ap-
proaches to streamline evaluation across diverse applications.
RAGChecker [14] and CoFE-RAG [10] provide end-to-end evaluation by
simultaneously analyzing retrieval effectiveness and generation fidelity. These
frameworks offer diagnostic tools for transparency, error localization, and inter-
pretability, enabling systematic debugging and optimization.
RAGAS [8] minimizes reliance on ground-truth annotations by leveraging
unsupervised, language model-based metrics. This approach facilitates scalable
evaluation of retrieval relevance, faithfulness, and context alignment, making it
adaptable to low-resource and dynamic domains.
By integrating multi-dimensional assessment methodologies, these frame-
works establish a foundation for benchmarking RAG systems effectively, pro-
moting reliability, reproducibility, and generalizability in diverse real-world ap-
plications.
2.4 Domain Specific Evaluation Benchmarks
WhilegenericbenchmarksofferinitialassessmentsofRAGmodels,theyoftenfail
to capture real-world performance in specialized fields. Since RAG effectiveness
depends on domain-specific information quality, developing targeted evaluation
benchmarks is crucial for assessing models within particular subject areas, data
types, and tasks.
For industry applications, domain-specific benchmarks complement, rather
thanreplace,evaluationsusingdatasetsreflectiveofreal-worlddeployment.Pub-
lic benchmarks rarely match the data a system encounters in production, making
customized evaluation datasets essential.
Several research efforts have addressed this need:
6

– Code Generation and Analysis:Domain-specific evaluation is vital for
RAG models handling code tasks like completion, search, bug fixing, and
synthesis. CodeRAG-Bench [17] emphasizes evaluating models on code exe-
cution, syntax, and functional correctness, aspects often missing in generic
text benchmarks.
– Medical Applications:RAG in healthcare requires high accuracy and
reliability. Medical RAG benchmarks [19] assess clinical accuracy, factual
correctness, and safety using medical knowledge bases, guidelines, and real-
world patient data.
– Legal Domain:Legal RAG tasks, such as case law analysis and legal doc-
ument retrieval, require benchmarks that assess legal reasoning, precedent
identification, and accurate statutory interpretation [11].
– Financial Domain:RAG models used for financial analysis and market
predictions must be evaluated on financial datasets, considering accuracy,
risk assessment, and regulatory adherence [16].
Domain-specific benchmarks provide realistic performance assessments, enable
targeted optimizations, and clarify RAG strengths and weaknesses across fields.
Futureworkshouldfocusonstandardizingbenchmarkcreation,developingdomain-
specific metrics, and fostering cross-domain collaboration to advance RAG eval-
uation.
3 Deepchecks’ RAG Evaluation Framework
Deepchecks is a modular end-to-end evaluation system for LLM-based appli-
cations, designed to deal with various use-cases, including RAG. Deepchecks’
framework incorporates core principles from leading RAG evaluation method-
ologies and provides an intuitive user-interface for evaluation and production
monitoring.
Deepchecks evaluation framework decomposes complex tasks into smaller,
manageable sub-tasks, referred to as "properties". Each property is evaluated
independently, enabling error tracing and iterative improvements in pipeline
performance. In addition to property-level assessments, Deepchecks employs a
learned aggregation mechanism to derive an end-to-end evaluation metric. This
feature allows for a holistic assessment of the pipeline’s performance while main-
taining transparency in individual component evaluations.
The above framework supplies a solid foundation for evaluating the perfor-
mance of a RAG pipeline. Deepchecks complements it with tools for root cause
analysis (RCA), comparing performance of different system versions, and moni-
toring it once it’s deployed in a production environment.
3.1 Properties for RAG Evaluation
In the context of RAG evaluation, Deepchecks decomposes the task into three
main properties, accompanied by several safety properties:
7

– Retrieval Relevance:Measures that the retrieved information is relevant
for addressing the input question.
– Grounded in Context:Measures that the generated answer is factually
consistent with the source of truth, synonymous with faithfulness and ad-
herence to the source of truth.
– Completeness:measures that the answer comprehensively addresses all
parts of the question.
– Safety Properties:Include metrics for evaluating safety concerns such as
toxicity, inclusion of personally identifiable information (PII), jailbreak at-
tempts, and other ethical considerations. As in any LLM-based application,
safetypropertiesareessentialmetricsforRAGthatensuretheLLMoperates
within ethical, legal, safe, and unbiased boundaries.
Theseprinciplesformthefoundationofseveralpopularframeworks([8],[14]),
and Deepchecks combines them into an integrated system while excelling in ac-
curately and efficiently measuring these properties. For each property, it utilizes
state-of-the-art techniques alongside proprietary methods designed to perform
exceptionally well in real-world scenarios. An example of these methods will be
presented in section 4.
In cases where a reference answer that can serve as ground truth is available,
Deepchecks provides theExpected Output Similarityproperty. This metric
evaluates whether the generated output captures the main arguments present
in the reference output, functioning as either a sufficient evaluation metric or a
complementary measure to reference-free properties.
3.2 End-to-end RAG Evaluation
In addition to property evaluation, Deepchecks emphasizes capturing human in-
tuitionbydeterminingwhetherinteractionsarepositiveornegative,ensuringthe
evaluation framework aligns with real-world expectations and user requirements.
Deepchecks’ evaluation is the only framework to date to generate a holistic met-
ric for RAG pipelines.
This approach enables Deepchecks to deliver a nuanced and accurate assess-
ment of RAG pipelines, ensuring alignment with user expectations and practical
use cases. A thorough analysis of Deepchecks’ holistic evaluation will be pre-
sented in section 5.
3.3 Tools for Research and Production Life-Cycle
Deepchecks’ RAG evaluation capabilities extend beyond calculating metrics and
generating end-to-end scores. By offering advanced tools for RCA, version com-
parison, and production monitoring, Deepchecks provides actionable insights
that drive informed decision-making and system improvements. This section ex-
plores how these tools complement the metrics calculated by Deepchecks.
8

3.3.1 Root Cause Analysis (RCA)Deepchecks’ RCA tools allow practi-
tioners to identify the underlying factors contributing to performance degrada-
tions or anomalies in RAG systems. Using the detailed breakdown of properties,
users can pinpoint specific components within the pipeline that require improve-
ment.
For example, RCA can help determine whether retrieval failures or inad-
equate grounding in the retrieved context are responsible for suboptimal re-
sponses. This targeted approach accelerates debugging and enables focused iter-
ation on the weakest components of the system.
Fig. 2.: Deepchecks’ RCA tools, such as annotation breakdown (top-left), insights
based on properties’ scores (top-right) and ungrounded content highlighting (bottom)
assist in pinpointing specific components within the pipeline that require improvement.
3.3.2 Version ComparisonThe version comparison feature is a critical tool
for assessing the impact of changes made to a RAG system. By juxtaposing
the metrics of different system versions side-by-side, users can evaluate whether
modifications have led to improvements in key properties.
Deepchecks allows for both per-metric comparisons (e.g., the effect of a new
retrieval model on context relevance) and per-interaction analyses (e.g., compar-
ing outputs for specific queries). This structured comparison helps users under-
stand trade-offs and ensures that system upgrades align with desired objectives.
3.3.3 Production MonitoringOnce deployed, a RAG system must main-
tain its performance under varying real-world conditions. Deepchecks’ produc-
tion monitoring tools track key reference-free metrics over time, enabling users
to detect and respond to performance drifts or anomalous behavior.
9

Fig. 3.: Deepchecks’ production monitoring tracks key metrics over time, indicating
performance degradation due to data distribution shifts and assists in prompting fo-
cused system upgrades.
For instance, an increase of answer avoidance score over time likely indi-
cates data distribution shifts. By continuously evaluating system metrics, orga-
nizations can ensure their RAG systems remain reliable and effective, even as
external conditions evolve.
4 Grounded in Context
As mentioned in the introduction, one of the most pressing challenges in gen-
erative AI is addressing hallucinations—cases where a language model produces
inaccurate or unsupported claims. Researchers are actively exploring a variety
of approaches for hallucination detection, aiming to enhance the reliability of
LLM-generated content. Table 1 compares the factual grounding faithfulness of
Deepchecks, RAGAS, and Langsmith across several datasets.
Fig. 4.: Deepchecks’ Grounded in Context method. Calculating an entailment score
for each factual statement based on the chunks of the documents most relevant for it,
and then aggregating the scores to a unified Grounded in Context score.
10

RAGAS is an open-source evaluation framework for RAG applications, pro-
viding metrics to assess faithfulness, relevance, and retrieval quality. It integrates
with popular LLMs to diagnose and improve RAG system performance. Lang-
SmithisaLangChaintoolfordebugging,monitoring,andevaluatingLLMappli-
cations, offering insights into prompt execution, latency, and model behavior. It
enables developers to trace and optimize their pipelines with structured logging
and performance analytics.
Both RAGAS and LangSmith rely solely on external LLMs to evaluate vari-
ous metrics, using their own predefined prompts for the process.
Deepchecks’ property for evaluating the faithfulness of a generated output
to a source of truth is calledGrounded in Context, and it is comprised of a
multi-step pipeline (see Figure 4):
1. Extracting factual statements from the LLM’s output using a proprietary
small language model (SLM) for information density classification.
2. Retrieving relevant passages from the context for each statement.
3. Classifying whether a statement is entailed by the retrieved passages using
a proprietary SLM trained on production-like data.
4. Aggregating the entailment scores using a dedicated averaging method.
4.1 Benchmarking Grounded in Context
To benchmark this method, we compared its performance against other estab-
lished approaches. The datasets used in this comparison included context as the
source of truth, model outputs, and annotations indicating whether the outputs
were factually grounded. The classification setup required each method to score
the factual grounding of the outputs, with results evaluated using the ROC-AUC
metric.
Dataset Deepchecks
Grounded
in ContextRAGAS
Faithfulness
(GPT-4o)Langsmith
Answer
Faithfulness
(GPT-4o)
TRUE 0.86 0.65 0.63
SQuAD 0.92 0.83 0.65
PubmedQA 0.84 0.80 0.78
Table 1.Performance of Deepchecks, RAGAS, and Langsmith faithfulness measure-
ment methods in evaluating factual grounding across several datasets.
The following datasets were used in the evaluation:
– TRUE Benchmark:A comprehensive collection of 11 manually annotated
datasets from various domains. We sampled 100 entries from each dataset.
11

– Corrupted Open Datasets:GPT-4 was used to generate two types of
negatives—hallucinationsandcontradictions—basedontheSQuADdataset.
Each dataset contained 62 positive samples, 31 with hallucinations, and 31
with contradictions.
– PubmedQA: a biomedical question answering (QA) dataset collected from
PubMed abstracts. We sampled 1000 entries.
5 RAG End-to-End Evaluation
Deepchecks’ end-to-end (E2E) evaluation integrates the scores from the core
properties—retrievalrelevance,groundedincontext,completeness,andsafety—through
a learned aggregation mechanism. This mechanism classifies each triplet (ques-
tion, retrieved-context, and answer) into one of three categories:
– Negative: A single failure in any of the core properties automatically clas-
sifies the interaction as negative.
– Positive: To qualify as positive, the answer must meet a high threshold
across all core properties, ensuring robustness and reliability.
– Unknown: Interactions that don’t fail any metrics but also fall short of the
stringent criteria for positive classification are labeled as unknown.
While Deepchecks provides pre-configured settings for comprehensive RAG eval-
uation, the platform allows for customization of threshold values across the core
metrics—groundedness, relevance, completeness, and safety—as well as incorpo-
ration of additional metrics to align with specific evaluation requirements.
5.1 Benchmarking End-to-End Evaluation
To assess the quality of Deepchecks’ automatic evaluation pipeline, we collected
various annotated Retrieval-Augmented Generation (RAG) datasets and fed the
pipeline with triplets of question, context, and answer. A positive triplet is de-
fined as one where the answer is both correct and faithful (i.e., grounded in the
provided context). Deepchecks’ evaluation of the answers was then compared
with human-provided annotations.
We compared our results with two alternative LLM evaluation frameworks:
Langsmith and RAGAS. Neither framework offers an evaluation pipeline that
provides a general annotation (positive/negative). Thus, we aggregated their
supplied metrics in a way that yields the best possible score for each. To en-
sure a fair comparison, we used only the property-based evaluation component
of Deepchecks’ pipeline, as competitors’ systems do not learn from client data.
This approach facilitates a balanced assessment across all systems while under-
representing the full capabilities of Deepchecks’ complete pipeline.
Each dataset was balanced between positive and negative examples, and per-
formance was evaluated using the accuracy metric (number of correct predictions
/ total number of samples). Accuracy is most suitable for balanced binary clas-
sification tasks where both labels are of interest.
12

5.1.1 Client Datasets
–Real-world RAG use cases from two of our clients, featured in this demo
with their consent.
–Provided with client annotations based on specific needs.
–Most representative of production use cases.
–Most challenging to analyze correctly.
–Each dataset contains 100 samples with balanced binary labels (annotated
by the clients).
–Derived from two use cases: Q&A on technical support data and job candi-
dates’ work histories.
5.1.2 Public Datasets
–WidelyknownandreliablepublicRAGuse-casedatasets:SQUAD,RAG-dataset-12000,
andHAGRID.
–Originally contained only positive samples; negatives were generated using
several corruption methods:
•Contradiction:Partially modifying the answer to create internal in-
consistencies.
•Hallucination:Altering facts in the answer to produce outputs that
are not fully grounded in the given context.
•Irrelevant (but faithful):Replacing the answer and context with sim-
ilar but unrelated information (most similar negatives in the dataset).
–Each dataset contains 100 samples with balanced binary labels (original
samples annotated as positive; corrupted samples annotated as negative).
5.1.3 ResultsTheresultsoftheevaluationaresummarizedinTable2.Deepchecks
demonstrates superior or competitive performance compared to Langsmith and
RAGAS across all datasets, particularly excelling in client datasets representa-
tive of real-world production use cases.
Dataset Deepchecks RAGAS (GPT-4o) Langsmith (GPT-4o)
RAG-dataset-120000.960.650.96
SQUAD0.940.78 0.93
HAGRID0.940.59 0.84
Client 1 (Tech Support)0.700.55 0.58
Client 2 (Job Profiles)0.810.46 0.54
Table 2.Performance comparison of Deepchecks, RAGAS, and Langsmith across pub-
lic and client datasets, measured by accuracy.
13

6 Conclusions
In this study, we presented a comprehensive evaluation of Deepchecks, a mod-
ular and end-to-end system for assessing the quality of Retrieval-Augmented
Generation (RAG) pipelines. By focusing on core properties such as retrieval
relevance, contextual grounding, completeness, and safety, Deepchecks provides
arobustframeworkfordiagnosingandimprovingtheperformanceofRAG-based
systems.
Through detailed benchmarking against alternative solutions like RAGAS
andLangsmith,wedemonstratedthatDeepchecksoutperformsormatchesexist-
ing methods across both public and client datasets. Specifically, Deepchecks ex-
cels in scenarios involving real-world, production-level complexity, as evidenced
by its higher accuracy on client-provided data. This highlights the framework’s
ability to meet practical demands and align with user expectations in diverse
deployment contexts.
Our results further underscore the importance of property-based evaluations
in mitigating common challenges associated with LLMs, such as hallucinations,
contradictions, and irrelevant outputs. By incorporating a learned aggregation
mechanismforend-to-endassessments,Deepchecksisuniquelypositionedtopro-
vide holistic evaluations that capture both granular and global quality measures.
In conclusion, Deepchecks framework represents a significant step forward in
the evaluation of LLM-driven RAG systems, offering a reliable, accurate, and
user-aligned tool for advancing the state of the art in generative AI applications.
14

Bibliography
[1] Amazon Web Services: New RAG evaluation and llm-as-a-judge ca-
pabilities in Amazon Bedrock.AWS Blog(2025), retrieved from
https://aws.amazon.com/blogs/aws/new-rag-evaluation-and-llm-as-a-
judge-capabilities-in-amazon-bedrock/
[2] Arize AI: Llms as judges: A comprehensive survey on LLM-based evaluation
methods.Arize AI Blog(2025), retrieved from https://arize.com/blog/llm-
as-judge-survey-paper/
[3] Bai, Y., Miao, Y., Chen, L., Wang, D., Li, D., Ren, Y., Xie, H., Yang, C.,
Cai, X.: Pistis-rag: Enhancing retrieval-augmented generation with human
feedback. arXiv preprint arXiv:2407.00072 (2024)
[4] Bowman, S.R., Angeli, G., Potts, C., Manning, C.D.: A large annotated cor-
pus for learning natural language inference. Proceedings of the 2015 Con-
ference on Empirical Methods in Natural Language Processing pp. 632–642
(2015), https://aclanthology.org/D15-1075
[5] Confident AI: Leveraging llm-as-a-judge for automated evaluation. Con-
fident AI Blog (2024), https://www.confident-ai.com/blog/why-llm-as-a-
judge-is-the-best-llm-evaluation-method
[6] Confident AI: Leveraging llm-as-a-judge for automated evaluation.
Confident AI Blog(2025), retrieved from https://www.confident-
ai.com/blog/why-llm-as-a-judge-is-the-best-llm-evaluation-method
[7] Engineering,V.:Improvingretrievalwithllm-as-a-judge.Vespa Blog(2024),
https://blog.vespa.ai/improving-retrieval-with-llm-as-a-judge/
[8] Es, S., James, J., Espinosa-Anke, L., Schockaert, S.: RAGAS: Au-
tomated evaluation of retrieval augmented generation. arXiv preprint
arXiv:2309.15217 (2023)
[9] Lin, J., et al.: Generative information retrieval evaluation. In: Proceedings
of the 47th International ACM SIGIR Conference. ACM (2024)
[10] Liu, J., Ding, R., Zhang, L., Xie, P., Huang, F.: Cofe-rag: A comprehensive
full-chain evaluation framework for retrieval-augmented generation with en-
hanced data diversity. arXiv preprint arXiv:2410.12248 (2024)
[11] Pipitone, N., Alami, G.H.: Legalbench-rag: A benchmark for retrieval-
augmented generation in the legal domain. arXiv preprint arXiv:2408.10343
(2024)
[12] Promptfoo Team: Evaluating rag pipelines. Promptfoo Documentation
(2024), https://www.promptfoo.dev/docs/guides/evaluate-rag/
[13] Reimers, N., Gurevych, I.: Sentence-bert: Sentence embeddings using
siamese bert-networks. Proceedings of the 2019 Conference on Empir-
ical Methods in Natural Language Processing pp. 3982–3992 (2019),
https://aclanthology.org/D19-1410
[14] Ru,D.,Qiu,L.,Hu,X.,Zhang,T.,Shi,P.,Chang,S.,Jiayang,C.,Wang,C.,
Sun, S., Li, H., et al.: Ragchecker: A fine-grained framework for diagnosing
retrieval-augmented generation. arXiv preprint arXiv:2408.08067 (2024)

[15] Snowflake Engineering: Benchmarking llm-as-a-judge for the
RAG triad metrics.Snowflake Blog(2025), retrieved from
https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-
as-a-judge-RAG-triad-metrics/
[16] Wang, S., Tan, J., Dou, Z., Wen, J.R.: Omnieval: An omnidirectional and
automatic rag evaluation benchmark in financial domain. arXiv preprint
arXiv:2412.13018 (2024)
[17] Wang, Z.Z., Asai, A., Yu, X.V., Xu, F.F., Xie, Y., Neubig, G., Fried,
D.: Coderag-bench: Can retrieval augment code generation? (2024),
https://arxiv.org/abs/2406.14497
[18] Williams,A.,Nangia,N.,Bowman,S.R.:Abroadcoveragechallengedataset
for sentence understanding through inference. Proceedings of the 2018 Con-
ference on Empirical Methods in Natural Language Processing pp. 1112–
1126 (2018), https://aclanthology.org/D18-1145
[19] Xiong, G., Jin, Q., Lu, Z., Zhang, A.: Benchmarking retrieval-augmented
generation for medicine (2024), https://arxiv.org/abs/2402.13178
[20] Zaheer, M., et al.: Retrieve, annotate, evaluate, repeat: Leveraging multi-
modal llms for large-scale product retrieval evaluation. Tech. rep., arXiv
preprint (2024)
16