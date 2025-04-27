# Retrieval Augmented Generation Evaluation in the Era of Large Language Models: A Comprehensive Survey

**Authors**: Aoran Gan, Hao Yu, Kai Zhang, Qi Liu, Wenyu Yan, Zhenya Huang, Shiwei Tong, Guoping Hu

**Published**: 2025-04-21 06:39:47

**PDF URL**: [http://arxiv.org/pdf/2504.14891v1](http://arxiv.org/pdf/2504.14891v1)

## Abstract
Recent advancements in Retrieval-Augmented Generation (RAG) have
revolutionized natural language processing by integrating Large Language Models
(LLMs) with external information retrieval, enabling accurate, up-to-date, and
verifiable text generation across diverse applications. However, evaluating RAG
systems presents unique challenges due to their hybrid architecture that
combines retrieval and generation components, as well as their dependence on
dynamic knowledge sources in the LLM era. In response, this paper provides a
comprehensive survey of RAG evaluation methods and frameworks, systematically
reviewing traditional and emerging evaluation approaches, for system
performance, factual accuracy, safety, and computational efficiency in the LLM
era. We also compile and categorize the RAG-specific datasets and evaluation
frameworks, conducting a meta-analysis of evaluation practices in high-impact
RAG research. To the best of our knowledge, this work represents the most
comprehensive survey for RAG evaluation, bridging traditional and LLM-driven
methods, and serves as a critical resource for advancing RAG development.

## Full Text


<!-- PDF content starts -->

Front. Comput. Sci., 2025, 0(0): 1–18
https: //doi.org /10.1007 /sxxxxx-yyy-zzzz-1
REVIEW ARTICLE
Retrieval Augmented Generation Evaluation in the Era of
Large Language Models: A Comprehensive Survey
Aoran GAN1, Hao YU2, Kai ZHANG1, Qi LIU(B)1, Wenyu YAN1, Zhenya HUANG1,
Shiwei TONG3, Enhong CHEN1, Guoping HU1,4
1 State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China, Hefei, China
2 McGill University, Montreal, Canada
3 Tencent Company, Shenzhen, China
4 Artificial Intelligence Research Institute, iFLYTEK Co., Ltd, Hefei, China
©Higher Education Press 2025
Abstract Recent advancements in Retrieval-Augmented Gen-
eration (RAG) have revolutionized natural language process-
ing by integrating Large Language Models (LLMs) with ex-
ternal information retrieval, enabling accurate, up-to-date, and
verifiable text generation across diverse applications. How-
ever, evaluating RAG systems presents unique challenges due
to their hybrid architecture that combines retrieval and gen-
eration components, as well as their dependence on dynamic
knowledge sources in the LLM era. In response, this paper
provides a comprehensive survey of RAG evaluation meth-
ods and frameworks , systematically reviewing traditional and
emerging evaluation approaches, for system performance, fac-
tual accuracy, safety, and computational e fficiency in the LLM
era. We also compile and categorize the RAG-specific datasets
and evaluation frameworks, conducting a meta-analysis of
evaluation practices in high-impact RAG research. To the
best of our knowledge, this work represents the most com-
prehensive survey for RAG evaluation, bridging traditional
and LLM-driven methods, and serves as a critical resource
for advancing RAG development.
Keywords Retrieval Augmented Generation, System Eval-
uation, Large Language Model
1 Introduction
Retrieval Augmented Generation (RAG) has emerged as a
powerful methodology that enhances natural language gener-
ation by incorporating information from external knowledge.
Received month dd, yyyy; accepted month dd, yyyy
E-mail: qiliuql@ustc.edu.cnThis approach significantly improves Large Language Mod-
els through non-parametric learning, multi-source knowledge
integration, and specialized domain adaptation [1, 2]. By
connecting LLMs with external databases, RAG produces re-
sponses that are both contextually appropriate and grounded
in authoritative, up-to-date information, marking a substan-
tial advancement in developing more sophisticated natural
language processing (NLP) systems [3, 4].
As a sophisticated and expansive system that encompasses
numerous elements from both the LLM and retrieval domains,
RAG can be approximately segmented into two principal sec-
tions from a macroscopic viewpoint: retrieval and genera-
tion. The retrieval section typically entails diverse operations
including preprocessing, dense or sparse retrieval, rerank-
ing and pruning, etc [5, 6]. The generation section com-
prises components such as retrieval planning, the integration
of multi-source knowledge, and logical reasoning [7, 8]. Ad-
ditionally, RAG systems incorporate interconnected upstream
and downstream elements such as document chunking, em-
bedding generation, and mechanisms for ensuring security
and credibility [9]. The overall performance of RAG systems
depends not only on each individual component but also on
their interactions and integrated functionality.
When faced with such complex systems, a fundamental
and practical question arises regarding the evaluation frame-
work for assessing the e fficacy of architectural methodologies
governing both the holistic system and its constituent compo-
nents. This challenge proves particularly pronounced in RAG
systems, where three factors - the expansive scope of im-
plementation domains, the heterogeneity of internal compo-
nents, and the dynamic progression of current developments
- collectively render the establishment of a unified system-arXiv:2504.14891v1  [cs.CL]  21 Apr 2025

2 Front. Comput. Sci., 2025, 0(0): 1–18
atic evaluation paradigm an ongoing research frontier. In re-
sponse to this, we conducted this survey on RAG Evaluation
to gather methods for multi-scale assessment of RAG in re-
cent years. The comprehensiveness of this survey is demon-
strated in four aspects: 1) Systematic completeness, encom-
passing both the evaluation of RAG’s internal components
and the system as a whole; 2) Methodological variety, in-
cluding both traditional statistically-based evaluation metrics
and the innovative methods characteristic of the LLM era;
3) Source diversity, incorporating both structured evaluation
frameworks, as well as cutting-edge methods scattered across
various papers; and 4) Practicality, both in terms of metrics’
definition to be evaluated and their subsequent application.
Through this multi-dimensional approach, we aim to provide
researchers and practitioners with a comprehensive toolkit for
evaluating and improving RAG systems.
The remainder of this paper is organized as follows: Sec-
tion 2 o ffers a concise review of the existing LLM-based RAG
system to provide the reader with relevant background knowl-
edge. Our comprehensive evaluation is divided into two dis-
tinct sections: Internal Evaluation (Section 3) and External
Evaluation (Section 4). Internal Evaluation assesses com-
ponent level performance and methodology-specific metrics
within basic RAG systems, focusing on technical advance-
ment. External evaluation examines system-wide factors like
safety and e fficiency, emphasizing practical viability. We
pay particular attention to the emerging trend of LLM-based
evaluation methods, which represent a novel assessment ap-
proach unique to the current era. Section 5 presents exist-
ing RAG evaluation frameworks, datasets, and methods, pro-
viding a practical resource for researchers. Furthermore, we
compiled a comprehensive collection of high-level RAG stud-
ies spanning multiple dimensions in recent years, and con-
ducted a preliminary analysis and discussion from the per-
spective of evaluation (Section 6).
2 Background
2.1 Large Language Model (LLM)
Large Language Models, with billions of parameters, are a
class of generative neural language models trained on exten-
sive natural language data [10, 11]. Due to the wide cover-
age of the training corpus, LLMs are considered to implicitly
integrate world knowledge [12]. LLMs are capable of ad-
hering to human instructions or requests though instruction
tuning, thus being able to e ffectively understand and gener-
ate human-like text [13]. Its generalization open up a wide
range of applications, such as NLP, signal processing, and
recommender systems [14, 15]. However, LLM’s capabil-
ity remains circumscribed by their training data. It is some-
times predisposed to generating factually inconsistent out-
puts (hallucinations), particularly when processing novel in-
formation beyond training data [16]. Despite the adaptability
of LLMs to diverse downstream tasks through post-trainingor fine-tuning on specific datasets, these methods encounter
challenges related to arithmetic, timeliness, flexibility, or us-
ability (on close models). Optimization techniques during
the LLM inference phase have thus garnered significant at-
tention. One of the representative techniques is Prompt En-
gineering, in which artificially constructed task descriptions
and commands are used to enhance LLMs’ understanding
of task objectives. In-context learning is designed to enable
LLMs to analyze patterns and generalize from task samples,
offering substantial advantages in few-shot scenarios [17,18].
Unlike these approaches, RAG aims to address the issue of
knowledge limitations inherent in LLM by incorporating ex-
ternal knowledge. Both LLM and RAG possess complemen-
tary strengths: RAG can e ffectively leverage the superior rea-
soning capabilities of LLMs, combined with the broad knowl-
edge scope of external data, to explore the potential applica-
tions of LLMs more extensively [19]. On the other hand,
LLMs can serve as crucial components in RAG, functioning
as the decision maker, reasoner, generator, or even evaluating
certain aspects of RAG [20, 21].
2.2 Retrieval Augmented Generation (RAG)
RAG is a technical framework that enhances NLP systems by
integrating external knowledge retrieval, whose core innova-
tion enables extra non-parametric optimization of parameter-
fixed neural language models after training, e ffectively ex-
panding their operational domains while maintaining archi-
tectural stability [22]. Prior to the widespread adoption of
LLM, scholarly investigations had already established meth-
ods for enhancing NLP tasks through external knowledge in-
fusion [23]. Initial researches on RAG adhered to an ele-
mentary indexing and reading paradigm [24, 25]. Later for-
mulations delineated two core components: (1) the retriever,
which identifies, indexes, filters, and structures relevant knowl-
edge fragments from external data sources; (2) the generator,
which synthesizes the curated segments through analysis and
logical reasoning to produce outputs [9]. Figure 1 shows the
workflow of an RAG system with recommendations of com-
ponents implementation using LLMs at present. We provide
a concise description of each module’s process below.
The retrieval component of RAG systems is inspired by
the retrieval technologies in multiple domains, such as infor-
mation retrieval [26], open-domain question answering [27],
and recommender systems [28, 29]. Before the retrieval, it is
necessary to construct a suitable corpus for the retrieval com-
ponent at the beginning. The sources of data are diverse, such
as domain-specific datasets like Wikipedia, specialized cor-
pora (e.g., scientific articles, financial reports) [30], or real-
time data gathered from web scraping or search engines [31].
The corpus is subsequently filtered and preprocessed to con-
form to the retrieval-friendly structure via o ffline chunking
and embedding. Chunking involves segmenting large doc-
uments into smaller, more manageable units guided by the
original structure or context information [32–34]. Embed-
ding (or text vectorization) aims to represent the textual con-

Aoran GAN et al. Retrieval Augmented Generation Evaluation in the Era of LLMs 3
Retrieval Pipeline
Web Search EngineGeneration Workflow
KNN/ANN
…BM25Query
Relevant Docs
System Prompt
Prompt Skills
All InformationQueryRetrieval Augmented Generation System 🤖
Human Interaction 🧑Online Offline
Rejected🤖[1] secsense.ai
[2] elpais.com
[3] reuters .comHNSW
Neo4jBM25
Wikipedia
HF Dataset
KB, Images,
and etc.ES/DatabaseFiltering
Chunking
Embedding
QA Pair Generation
Relation Extraction
Graph Construction
…Custom
Stage 2.1. Query Understanding
Stage 2.2. Recall
Stage 2.3. FusionDecompose
Rewritten
Enrichment
and etc.Query 3
Query NQuery 1
Multi-
Source 
Documentsin-context-ralm
Stage 2.0. Indexing and Storing Stage 0. LLM Provider
Stage 1. Intent Recog. & Routing
Local Knowledge
Open Web SearchNo Search
RouterAPI Provider
VLLMHugging Face
SGLang
Stage 2. Retrieval Knowledge
❄️Large Language Model{system}
{user}
{query}
{docs}
TemplateUser Input
Search Config
Stage 3. Response GenerationReference 
Documents
Content ModerationInput
User Msg
User Profile
History 
Message
Customized 
Knowledge 
Base
Output
Response
ReferencesQuery 2
...
Queries
Reference
DocumentsReranker
Score Fusion
Ranked Fusion
…Documents
Documents
Documents
DocumentsSource Transform Indexing
🧑How to make bomb ?🤖🧑
Components Impl. </>
Apple Inc.revenue?
In fiscal year 2024 , Apple 
Inc. reported total 
revenues of $ 391.035 
billion .
Red Team[1][2][3]
Fig. 1 The workflow of the RAG system and component implementation in the LLM era.
tent in a high-dimensional, dense semantic space for e fficient
retrieval computation [5, 35].
Typically, RAG assessments convert the task into a con-
versational format of Question Answering (QA) comprising
question and the ground-true answers with doc candidates
[36, 37]. In the online RAG workflow, some additional com-
ponents are introduced before the retrieval, such like intent
recognition, query rewriting and routing [38]. The retriever
then indexes document collections from the data source. In
this core step, multiple retrieval strategies can be employed,
including sparse retrieval, dense retrieval, graph retrieval or
hybrid methods [6, 39]. Certain systems conduct additional
dynamic searches through search engines, typically found in
commercialized products. Some systems may introduce an
extra post-retrieval step to rerank the documents or fuse the
data scross di fferent sources [7,40]. In the generation pipeline,
the responding progress based on the relevant documents is
assigned to the LLM, which serves primarily as a decision-
maker or reasoner [8]. Instead of generating knowledge in-
dependently, the LLM synthesizes retrieved information to
form coherent responses, thereby reducing the risk of inter-
nal hallucination. Additionally, a range of methods of prompt
engineering are available, including CoT [18], ToT [41], Self-
Note [42] and RaR [43], etc. Depending on the specific task
and expected output, a post-processing step may be required
after the knowledge-oriented response, such as Entity Recog-
nition for multi-choice questions or classification task, and
the translation component for multilingual task. Moreover,
the utility of the model’s application is a point of concern,
particularly regarding safety and e fficiency [44].2.3 Related Surveys
Li et al. [23] summerrized and formalized the key definitions
of RAG while providing a synthesis of early-stage method-
ologies and practical applications. Expanding the scope be-
yond NLP, Zhao et al. [45] traced the developmental trajec-
tory of multimodal RAG across the broader AIGC landscape.
The emergence of LLM has since triggered an accelerated
development of RAG methods, with numerous survey papers
emerging to document this growing research domain [1,9,19,
20, 46]. Current researches mainly focus on collecting meth-
ods or applications, but lack substantive discussion about sys-
tematic evaluation mechanisms. While Yu et al. [21] pro-
vided an initial review outlining conceptual approaches for
RAG evaluation, their analysis was predominantly confined
to mainstream the frameworks, o ffering limited insights into
emerging assessment methods applicable to diverse contexts.
Building upon previois foundational work, this comprehen-
sive survey extends beyond these limitations, o ffering deeper
insights into emerging evaluation methods.
This study extends the research [21] by incorporating a
broader array of RAG evaluation methods within a systems
theory context. We di fferentiate between internal and exter-
nal evaluations: the former examines the RAG component
assessments and their interactive processes within the system
architecture, while the latter focuses on holistic system eval-
uation and environmental considerations, where environment
specifically denotes the external tasks or particular evaluation
contexts. We extend our horizons beyond collecting concep-
tual definitions of evaluation methods to exploring and ana-

4 Front. Comput. Sci., 2025, 0(0): 1–18
lyzing their practical application in the actual RAG studies.
Simultaneously, we focuses on RAG evaluation in LLM con-
texts, prioritizing unstructured text retrieval as the prevail-
ing paradigm. Domain-specific variants of RAG evaluation
(e.g., knowledge graph, multimodal retrieval) are excluded
due to fundamental architectural gaps. Unless otherwise in-
dicated, all the ‘RAG’ hereafter pertain to the narrow opera-
tional training-free framework employing unstructured docu-
ments as external knowledge resources.
3 Internal Evaluation
In this section, we summarize and organize the evaluations of
the internal components with their interactions within a RAG
system from prior studies. We deconstruct the evaluation of a
whole RAG system, focusing on internal component interac-
tions. A range of evaluation approaches are then introduced,
from traditional to new ones. The elements mentioned and
the implication of internal evaluation point to a framework
forevaluating the strengths of the RAG system’s core func-
tionality , that is, generating accurate and credible output.
3.1 Evaluation Target
The diverse components of the RAG system can be boiled
down to solving two core problems: the retrieval of the ground
truth, and the generation of the response that closely aligns
with the gold answer. They correspond to the respective eval-
uation objectives of the retrieval and generation modules.
Figure 2 summarizes the evaluation targets of the retrieval
and generation component. The retrieval component includes
two main stages, recall and ranking. The outputs, relevant
documents, for both are similar to evaluate. Then we can con-
struct several pairwise relationships for the retrieval compo-
nent by defining the target as follows:
Relevance (Relevant Documents ↔Query ) evaluates how
well the retrieved documents match the information needed
expressed in the query. It measures the precision and speci-
ficity of the retrieval process.
Comprehensiveness (Relevant Documents ↔Relevant Doc-
uments ) evaluates the diversity and coverage of the retrieved
documents. This metric assesses how well the system cap-
tures a wide range of relevant information, ensuring that the
retrieved documents provide a comprehensive view of the
topic according to the query.Correctness (Relevant Documents ↔Documents Candi-
dates ) assesses how accurate the retrieved documents are in
comparison to a set of candidate documents. It is a measure of
the system’s ability to identify and score relevant documents
higher than less relevant or irrelevant ones.
The similar pairwise relations and targets for the genera-
tion component are outlined below.
Relevance (Response↔Query ) measures how well the
generated response aligns with the intent and content of the
initial query. It ensures that the response is related to the
query topic and meets the query’s specific requirements.
Faithfulness (Response↔Relevant Documents ) evalu-
ates how the generated response accurately reflects the infor-
mation contained in the relevant documents and measures the
consistency between the generated and source documents.
Correctness (Response↔Sample Response ) Similar to
the accuracy in the retrieval component, this measures the ac-
curacy of the generated response against a sample response,
which serves as a ground truth. It checks if the response is
correct in terms of factual information and appropriate in the
context of the query.
3.2 Conventional Evaluation Methods
RAG is a cross-disciplinary system founded on traditional re-
search fields including information retrieval (IR) and natu-
ral language generation (NLG). Adhering to the conventional
methods of them, numerous traditional metrics are employed
to evaluate the retrieval and generation of RAG as follows.
3.2.1 IR-related Metrics
The IR-related metrics refer to the indicators associated with
conventional retrieval systems. These metrics are categorized
into two groups based on their correlation to ranking:
•Non-rank-based Metrics
The non-rank-based metrics typically evaluate binary outcomes,
that is, whether an item is relevant or not, without taking into
account the item’s position in a ranked list.
Accuracy /Hit@K is the proportion of true results (both
true positives and true negatives) among the cases examined.
Accuracy =T P+T N
TotalNumber
where T Pis the number of true positives, T Nis the number
of true negatives in the response.
Fig. 2 The evaluation target of the Retrieval and Generation component in RAG.

Aoran GAN et al. Retrieval Augmented Generation Evaluation in the Era of LLMs 5
Recall@K is the portion of relevant instances that have
been retrieved over the total amount of relevant cases, con-
sidering only the top kresults.
Recall =|RD∩Top kd|
|RD|
where RDis the relevant documents, and Top kdis the top-k
retrieved documents.
Precision@K is the fraction of relevant instances among
the retrieved instances, considering only the top kresults.
Precision =T P
T P+FP
where T Prepresents true positives and FPrepresents false
positives, respectively.
F1 Score measures the balance between precision and re-
call, defined as the Harmonic Mean of the two.
F1=2×Precision×Recall
Precison +Recall
•Rank-Based Metrics
The rank-based metrics focuse on the sequential presentation
of relevant items, assigning greater significance to the posi-
tioning of these items within the ranking list.
MRR (Mean Reciprocal Rank) is the average of the recip-
rocal ranks of the first correct answer for a set of queries.
MRR =1
|Q||Q|X
i=11
rank i
where|Q|is the number of queries and rank iis the rank posi-
tion of the first relevant document for the i-th query.
NDCG (Normalized Discounted Cumulative Gain) accounts
for the position of the relevant documents by penalizing rele-
vant documents that appear lower in the search results [47].
NDCG @k=DCG @k
IDCG @k
where DCG @kis the Discounted Cumulative Gain at rank k
andIDCG @kis the Ideal Discounted Cumulative Gain at
rank k, which represents the maximum possible DCG @k.
DCG @kis defined as:
DCG @k=kX
i=12reli−1
log2(i+1)
with relibeing the graded relevance of the result at position i.
MAP (Mean Average Precision) is the mean of the average
precision scores for each query.
MAP =1
|Q||Q|X
q=1Pn
k=1(P(k)×rel(k))
|relevant documents q|
where P(k) is the precision at cuto ffkin the list, rel(k) is an
indicator function equaling 1 if the item at rank kis a relevant
document in the nretrieved documents, 0 otherwise.3.2.2 NLG-related Metrics
The NLG-related metrics focus on the content of the text out-
put, dedicated to the evaluation on the char or semantic level.
EM (Exact Match) is a simple, stringent and widely-used
evaluation metric that assesses the accuracy of model-generated
answers compared to the ground truth. It scores as 1 if a gen-
erated answer precisely aligns with the standard otherwise 0.
Typically, the responses need standardization and preprocess-
ing (e.g., conversion to lowercase, removal of punctuation,
elimination of articles, and standardization of number for-
mats) before comparison. A general approach involves com-
bining EM and Precision /Recall /F1 or edit distance [48,49].
ROUGE (Recall-Oriented Understudy for Gisting Evalu-
ation) [50] is a set of metrics designed to evaluate the quality
of summaries by comparing them to human-generated refer-
ence summaries. ROUGE can be indicative of the content
overlap between the generated text and the reference text.
The variants of ROUGEs measure the overlap of n-grams
(ROUGE-N, ROUGGE-W), word subsequences (ROUGE-L,
ROUGGE-S), and word pairs between the system-generated
summary and the reference summaries.
BLEU (Bilingual Evaluation Understudy) [51] is a metric
for evaluating the quality of machine-translated text against
one or more reference translations. BLEU calculates the pre-
cision of n-grams in the generated text compared to the ref-
erence text and then applies a brevity penalty to discourage
overly short translations. Beyond machine translation evalua-
tion, BLEU can also be used for supervised comparison eval-
uation for general natural language generation. BLEU has
limitations, such as not accounting for the fluency or gram-
maticality of the generated text.
METEOR [52] is a metric designed to assess the quality
of machine translation or text generation. It enhances BLEU
by incorporating mechanisms like synonymization, stemming
matching, and word order penalties, demonstrating a stronger
correlation with results obtained from manual evaluations.
METEOR is defined as:
METEOR =(1−p)(α2+1)Precision×Recall
Recall +αPrecision,
whereαis the balanced factor, and pis the penalization factor
for word order.
BertScore [53] leverages the contextual embedding from
pre-trained transformers like BERT to evaluate the semantic
similarity between generated text and reference text. BertScore
computes token-level similarity using contextual embedding
and produces precision, recall, and F1 scores. Unlike n-gram-
based metrics, BertScore captures the meaning of words in
context, making it more robust to paraphrasing and more sen-
sitive to semantic equivalence. It has multiple variants, in-
cluding backbone advanced pre-trained models (e.g. BERT,
RoBERTa and BART) and supervised evaluation based on ex-
ternal classifier design.
Textual Similarity measures the semantic variety in re-
trieved documents. It can be calculated using metrics like

6 Front. Comput. Sci., 2025, 0(0): 1–18
Intra-Document Similarity orInter-Document Similarity , which
assess the similarity between documents within a set.
Similarity =1
|D|2|D|X
i=1|D|X
j=1sim(di,dj)
where Dis the set of retrieved documents, dianddjare em-
beddings of individual documents, and sim(di,dj) is a simi-
larity measure (e.g.,the most commonly used cosine similar-
ity) between the two documents.
Coverage measures the proportion of relevant documents
retrieved from the total number of relevant documents avail-
able in the dataset. It quantifies how comprehensively the
system captures all pertinent information across the corpus,
across topics, categories, or entities defined by humans or in
the knowledge base.
Coverage =|RD∩Retrieved|
|RD|
where RDis the set of relevant documents and the notation
Retrieved is the set of retrieved documents. The coverage
can also be calculated at the group level, where the relevant
documents are grouped into di fferent categories or topics.
Coverage =|Relevant Groups∩Retrieved Groups|
|Relevant Groups|
Perplexity (PPL) gauges a language model’s predictive
prowess, illustrating its level of uncertainty concerning test
data. Essentially, it is an exponential variation of cross-entropy,
quantifying the model’s fit to the probability distribution of
the text. It is defined base on the generative LM output as
Perplexity =exp−1
NNX
i=1logp(wi|w1,w2,..., wi−1).
It’s important to note that the IR-related and NLG-related
methods are not directly equivalent to retrieval and generation
assessment methods. In RAG systems, retrieval and genera-
tion operations typically alternate. For instance, the query un-
derstanding and document fusion component are considered
as pre- and post-retrieval operations in the retriever, respec-
tively, yet the evaluation is sometimes based on the NLG-like
methods. SCARF [54] used BLEU /ROUGE to evaluate the
query relevance of the retriever. Blagojevic et al. [40] uti-
lized cosine similarity to assess the retrieval diversity. Addi-
tionally, the metrics can be adapted into various designs with
new label based on the specific subject of study, such as Ed-
itDist [55], Fresheval [56], etc.
3.2.3 Upstream Evaluation
Given the rapid advancement of RAG systems, it is crucial
to emphasize the significance of o ffline preprocessing of the
corpus. We supplement the evaluation method of preprocess-
ing modules, including chunking and embedding.The evaluation of chunking methods can be conducted at
two levels. First, chunk-specific evaluation focuses on in-
trinsic metrics such as Accuracy , measured by Full Keyword
Coverage—the percentage of required keywords present in at
least one retrieved chunk—and the Tokens To Answer met-
ric, which tracks the index of the first fully comprehensive
chunk and cumulative token count needed for full context
coverage [57]. Second, extrinsic evaluation analyzes how dif-
ferent chunking approaches influence retrieval performance
on downstream tasks. For example, [34] and [58] evaluate
chunking methods by comparing retrieval recall, precision,
and response quality using metrics like ROUGE, BLEU, and
F1 scores against ground truth evidence paragraphs, while
considering computational overhead. Other works extend this
evaluation using domain-specific datasets, such as financial
reports [57], to observe how structure-based and semantic
chunking improves retrieval accuracy while reducing latency
and token usage during inference.
Before retrieval, the embedding model determines the ac-
tual performance of retrieving relevant documents. Compre-
hensive benchmarks like Massive Text Embedding Bench-
mark (MTEB) [59] and Massive Multicultural Text Embed-
ding Benchmark (MMTEB) [60] have become standard for
the evaluation of embedding models. MTEB introduced the
first large-scale benchmark covering 8 embedding tasks across
58 datasets and 112 languages, establishing that no single
embedding method excels across all tasks. MMTEB sig-
nificantly expanded this work through a community-driven
effort, encompassing over 500 evaluation tasks across 250 +
languages and introducing novel challenges like instruction
following, long-document retrieval, and code retrieval.
Although the models of chunking and embedding have
broad applications, they primarily serve as an upstream com-
ponent of the retriever in RAG. The primary benefit to the en-
tire system, involving chunking and embedding, is reflected
in the enhancement of the retriever’s evaluation metrics.
3.3 Evaluation Methods via LLMs
The advancement of LLM has catalyzed refined investiga-
tions into RAG system architectures. Contemporary studies
increasingly employ LLM-driven assessment metrics, which
establish quantifiable benchmarks for iterative improvements
across di fferent RAG modules. They can be broadly catego-
rized into the output and representation based methods.
3.3.1 LLM Output based Methods
The LLM-output based evaluation methods perform content
identification or statistical analysis of the text-format output
of the RAG components assumed by the LLM. These meth-
ods feature a concise and easily understandable process with-
out restrictions regarding whether the LLM is open or closed.
The most straightforward approach is to instruct the LLM
to explicitly evaluate or score the textual output of the compo-
nent by prompt engineering. Methods like RAGAS [61] and

Aoran GAN et al. Retrieval Augmented Generation Evaluation in the Era of LLMs 7
Databricks Eval [62] prompt GPT-based judges with explicit
instructions, such as “Check if the response is supported by
the retrieved context. ” or“Assess completeness with respect
to the user query. ” Zhang et al. [63] utilized GPT-4 with a
few-shot prompt design to determine whether the generated
answer matches the gold ones comprehensively. Finsås et
al. [64] implemented a multi-agent LLM framework to eval-
uate the retrieval performance and reported a higher relevance
with the human preference than the traditional methods. Patil
et al. [65] proposed an Abstract Syntax Tree (AST) based
method to measure the hallucination in RAG, which indicates
the accuracy of calling external APIs in the RAG system.
These methods typically benefit from CoT reasoning.
In addition, numerous researchers have proposed novel
definitions of statistical metrics derived from the LLM out-
put, facilitating a multi-perspective approach to evaluating
the RAG components.
Dai et al. [66] proposed a new metric Semantic Perplexity
(SePer) to capture the LLM’s internal belief about the cor-
rectness of the generated answer. Given the query qand the
reference answers a∗,SePer is defined as the output sequence
likelihood with clustered entity target as:
S ePer M(q,a∗)=PM(a∗|q)≈X
Ci∈Ck(Ci,a∗)pM(Ci|q),
where Mis the specific LLM. Cis the cluster set that the an-
other clustering model groupes the responses into. pM(Ci|
q) means the probability that a response generated by Mis
mapped to the cluster Ci.k(Ci,a∗) is a simple kernal fuc-
tion to measure the distance between the meaning of semantic
cluster Cianda∗by utilizing char-level matching or simply
asking the LLM to get a True /False response.
Qi et al. [67] introduced the key point extraction to the
RAG evaluation and designed KPR metric to evaluate the ex-
tent to which LLMs incorporate key points extracted from the
retrieved documents into their generated responses:
KPR (·)=1
|Q|X
q∈QP
x∈xqI(x,M(q∥dq))
|xq|,
where Qis the global query set, and I(x,M(q∥dq)) is a fuc-
tion to judge whether a single LLM output sequence M(q∥dq)
based on the query qand the recalled documents dqentails the
predefined key points xq.
To evaluate the inconsistency of the di fferent retrievers in
RAG, Li et al. [68] proposed a pair of naive metrics called
Mean Relative Win /Lose Ratio (MRWR /MRLR). Given M
different retrieversR={r1,r2,...,rM}and the dataset with N
query & answer pairs, the correctness of model response for
each sample <qn,an>is first cauculated, denoted by Im(n)=
1 if the retriever rmanswers correctly on sample snotherwise
0. Then the Relative Win Ratio (RWR) of retriever riover
another retriever rjis defined as:
RWR (i,j)=PN
n=1Ii(n)∗(1−Ij(n))
PN
n=11−Ij(n),which represents the proportion of questions answered incor-
rectly by retriever rjthat were correctly answered by retriever
ri. MRWR and MRLR are calculated by respectively averag-
ing RWR across rows and columns among the retrievers:
MRWR( i)=1
M−1X
j,iRWR( i,j),
MRLR( i)=1
M−1X
j,iRWR( j,i).
Especially, MRLR( i)=0 implies that retriever riconsistently
outperforms all of the other ones.
Min et al. [69] proposed FactScore to messure whether the
generated content matches the given knowledge source by
breaking the generations into atomic facts. Chiang et al. [70]
further consideder the synonym expression and proposed the
advanced D-FAatScore .FactScore is a simple statistical de-
termination whether the factual content ain the generated text
ymatches the external knowledge base C:
FS(y)=1
|Ay|X
a∈A yI[ais supported byC].
D-FActScore links synonymous entities into the same cluster
Ayiand consider a cluster-level evaluation:
DFS( y)=1
|Ay|X
Ayi∈A yX
a∈A yiI[ais supported by C∗
i].
To evaluate the risk in the generator’s response, Chen et
al. [71] introduced the divided cases of the generated an-
swer, answerable(A) and unanswerble(U), along with the dif-
ferent prediction process in the RAG system, keep(K) and
discard(D). Four risk-aware evaluation metrics from various
aspects are defined as:
1)Risk that measures the proprotion of risky casess among
the kept samples:
Risk=|UK|
|AK|+|UK|
2)Care f ulness indicates the percentage of incorrect and dis-
carded samples that are equivalent to recall for the unanswer-
able samples:
Care f ulness =|UD|
|UK|+|UD|
3)Alignment refers to the proportion of samples in which the
system’s judgment align with the assigned labels:
Alignment =|AK|+|UD|
|AK|+|AD|+|UK|+|UD|
4)Coverage quantifies the proportion of samples retained:
Coverage =|AK|+|UK|
|AK|+|AD|+|UK|+|UD|

8 Front. Comput. Sci., 2025, 0(0): 1–18
3.3.2 LLM Representation based Methods
The representation-based methods, conversely, captures valu-
able metrics by modeling vector representation in the inter-
mediate or final layers of the LLM. These methods can mit-
igate overreliance on surface lexical patterns, but they may
lose interpretability since the final numeric similarity does
not necessarily clarify which factual detail is correct or not.
Certain methods are inspired by the conventional metrics,
demonstrated as expansions of existing metrics on the LLM.
For instance, GPTScore [72] is a GPT based LLM-scoring
method inspired by BertScore, which has been widely used
as a convincing metric. ARES [73] combined a classifier with
LLM embeddings to check whether a generative answer is se-
mantically aligned with ground-truth evidence. RAGAS [61]
uses a cosine similarity approach on LLM-generated embed-
dings to gauge answer relevance.
Moreover, numerous researchers have developed novel rep-
resentation based metrics, which serve not only to evaluate
the components but also to guide the further enhancement.
Zhao et al. [74] introduced a novel metric, Thrust , which
assesses the LLM’s knowledgeability by leveraging the repre-
sentation distribution of the instances produced by the LLM.
A hypothesis was proposed that if an LLM has acquired ad-
equate knowledge pertaining to a task, it should e ffectively
cluster samples related to that task through its hidden states.
TheThrust metric was defined as:
sthrust(q)=1
N·KNX
l=1KX
k=1|Ckl|
∥dkl(q)∥2·dkl(q)
∥dkl(q)∥,
where Nis the number of classes for the specific task, Kis
the number of clusters per class, |Ckl|denotes the cardinality
of the set. dkl(q) is a vector pointing from the representacion
of the query to the centroid.
Zhu et al. [75] introduced the information bottleneck the-
ory into retrieval component to messure the relevance of the
recalled document and candidate document. Moreover, a new
information bottleneck-based loss function was derived and
used to train a better noise filter for the retriever. Given the
sample{q,x,y}from the dataset and the noise filter p( ˜x|x,q)
(need tuning), the information bottleneck in the RAG task is
derived and formulated as:
IB( ˜x)=PLLM(x|[q,˜x,y])−αPLLM(y|[q,˜x]),
where [·] means the concatenation operation. PLLM means
the final output probability of the LLM.
Li et al. [76] proposed a new metric GECE based on ME-
TEOR for assessing the extent of the long-tailness of the gen-
erated text in RAG:
GECE =|METEOR( pred,re f)−1
nPn
i=1PLLM(ti)|
α·[E(▽ins)·▽ins],
whereαis the average word frequency, ▽insandE(▽ins) are
the gradient w.r.t. the current instance and the mean gradientof the total dataset, separately. A long-tail instance usually
has a smaller αand▽ins, obtaining a larger GECE , which
implies larger degree of long-tailness.
To assess the extent to which external knowledge is uti-
lized in the RAG response, Sun et al. [77] proposed External
Context ScoreE, which is defined on the response level as:
El,h
r=1
|r|X
t∈rEl,h
t=1
|r|X
t∈re·xL
t
∥e∥∥xL
t∥,
where|r|means the length of the response r,xL
tis the t-th
token’s vector logit of the last layer L.eis a pooled vector
of the most relevant vectors of xL
taccording to the attention
weights in the middle layer:
e=1
|Il,h
t|X
j∈Il,h
txL
j,
whereIl,h
tmeans the attended times where the token has
larger than top-k% attention scores with xL
tin the l-th layer.
Noted that some of these LLM based evaluation metrics
represent research specializations. While they may not be di-
rectly targeted towards an actual RAG system, their presen-
tation is an integral part of advancing researches in the field
of RAG, indicating significant contributions as well.
4 External Evaluation
We have dissected the components of RAG and provided a
comprehensive account of its internal evaluation. This sec-
tion shifts our focus to the external utility that RAG, as a com-
plete system , encounters. We summarize the external utility
in two areas: safety and e fficiency, the evaluation of whom
are introduced below.
4.1 Safety Evaluation
Safety pertains to the RAG system’s capacity to ensure the
generation of stable and harmless content within a dynamic,
even noisy or hazardous environment. As RAG systems con-
tinue widespread deployment, safety concerns have intensi-
fied beyond those of standalone LLMs. The incorporation of
external knowledge sources introduces unique vulnerabilities
requiring specialized evaluation frameworks [20].
Robustness evaluations focus on system behavior when
processing misleading information in retrieval results. The
RECALL benchmark [78] tests discrimination between reli-
able and counterfactual knowledge using BLEU, ROUGE-L,
and specialized metrics like Misleading Rate. Wu et al. [79]
quantify susceptibility to semantically related but irrelevant
information using Misrepresentation Ratio and Uncertainty
Ratio. SafeRAG [80] categorizes challenges like ”inter-context
conflict” with specific evaluation metrics, while C-RAG [81]
provides theoretical guarantees on generation risks using con-
formal risk analysis and ROUGE-L. Cheng et al. [82] intro-
duce two metrics to evaluate the RAG system: 1) Resilience

Aoran GAN et al. Retrieval Augmented Generation Evaluation in the Era of LLMs 9
Rate, aiming to emphases the system’s stability and robust-
ness, quantifies the percentage of instances where the sys-
tem’s responses remain accurate, both prior to and follow-
ing retrieval augmentation. 2) Boost Rate quantifies the pro-
portion of instances initially answered erroneously that were
subsequently corrected upon the introduction of a retrieved
document, evaluating the e ffectiveness of RAG.
Factuality focuses on generating accurate information and
avoiding plausible but incorrect statements (hallucinations),
especially with noisy or conflicting retrieval results [78, 83,
84]. Key metrics include Factual Accuracy , using standard
QA metrics (EM, F1, accuracy, etc.) when the context might
be misleading [78]; the Hallucination Rate , the frequency of
generated information not supported by or contradicting re-
trieved documents, often measured via LLM-as-judge [85] or
human evaluation; Citation Accuracy , assessing correct attri-
bution to sources using Citation Precision andCitation Re-
call[20, 85]; and Faithfulness Metrics , evaluating how accu-
rately the output reflects retrieved information [83].
Adversarial attacks target specific components within the
RAG pipeline. Knowledge database poisoning (PoisonedRAG
[86]) targets the retrieval corpus by injecting malicious texts
that trigger predetermined outputs when retrieved. This at-
tack vector is evaluated using Attack Success Rate (ASR)
and retrieval-focused Precision /Recall /F1 metrics. Retrieval
hijacking (HijackRAG [87]) exploits ranking algorithms to
prioritize malicious content during retrieval, with evaluation
focusing on attack transferability across models. Phantom at-
tacks [88] use trigger-activated documents evaluated through
Retrieval Failure Rate (Ret-FR), while jamming attacks [89]
insert ‘blocker’ documents that force response refusal, as-
sessed through oracle-based metrics.
Privacy assess information exposure risks from retrieval
databases or user queries [90]. Evaluation often involves sim-
ulated attacks [91,92]. Key metrics about privacy include the
Extraction Success Rate , the frequency or success rate of at-
tacks extracting specific private information (e.g., names, PII)
from the knowledge base, often measured by the count of
successfully extracted items [90]; the PII Leakage Rate , the
amount or percentage of Personally Identifiable Information
inadvertently revealed in generated outputs, typically found
via pattern matching or inspection [93]; and the Membership
Inference Attack Success , which measures an attacker’s abil-
ity to determine if a specific data record was in the RAG sys-
tem’s knowledge base.
Fairness examines if the RAG system exhibits or ampli-
fies biases from retrieved documents or training, leading to
inequitable outputs [94]. Bias Metrics are used to analyze
the outputs for disparities, which are quantitative measures
of performance disparities (e.g., error rates, sentiment scores)
across demographic groups [94]. Stereotype Detection mea-
sures the frequency or severity of harmful stereotypes in gen-
erated text, assessed via lists or human evaluation. Coun-
terfactual Fairness checks if outputs change inappropriately
when sensitive attributes in queries or context are altered.Transparency /Accountability assesses the understand-
ability and traceability of the RAG system’s reasoning pro-
cess, enabling verification of sources and justification [95,
96]. Metrics are often qualitative or user-focused, such as
Explanation Quality , based on human ratings of the clarity,
completeness, and usefulness of explanations or provenance
information [96]; Traceability , the ease of linking the final
output back to specific source documents or passages; and
Citation Accuracy (precision /recall) [20].
Comprehensive safety benchmarks standardize evaluation
across multiple dimensions. SafeRAG [80] classifies attack
tasks into four categories with tailored datasets. VERA frame-
work [97] uses bootstrap sampling for confidence bounds on
safety metrics, while DeepTeam’s red teaming approach [93]
identifies vulnerabilities through systematic testing. In addi-
tion, current research indicates defense mechanisms remain
insufficient against sophisticated attacks [86–88]. Evalua-
tions reveal significant vulnerabilities in current RAG sys-
tems [87, 88], underscoring the need for robust benchmarks
and metrics addressing the unique safety challenges arising
from the retrieval-generation interplay. Further e fforts are re-
quired to evaluate the safety of RAG.
4.2 E fficiency Evaluation
Efficiency is another crucial aspect of RAG’s utility, directly
linked to the real-world significance of a system’s popularity,
cost, and e ffectiveness.
Latency evaluation typically focuses on two critical met-
rics. Time to first token (TTFT) [98] measures the time taken
by the system to produce its initial output token after receiv-
ing a query, which is particularly crucial for user experience
as it directly impacts perceived responsiveness. This met-
ric is especially important in interactive applications where
immediate feedback maintains user engagement. Addition-
ally, complete response time (total latency) measures the du-
ration from query submission to the generation of the entire
response. This encompasses retrieval time, processing time,
and generation time for all tokens. Hofstatte et al. [99] pro-
posed Single Query Latency that refers to the complete end-
to-end time taken to process a single query, including both
complete retrieval and generation phases.
Resources and Money Cost evaluation of RAG systems is
another critical component for assessing the e fficiency. Cost
evaluation methodologies typically focus on quantifying both
direct expenditures and e fficiency metrics that impact overall
system economics. The total cost of RAG systems can be
categorized into several key components [126]:
•Infrastructure Costs : Computing local resources for
embedding generation, vector database maintenance,
and LLM inference for open models.
•Token-based Expenses : API charges for external LLM
services based on input and output token usage.
•Storage Costs : Vector database hosting and mainte-
nance expenses that scale with corpus size.

10 Front. Comput. Sci., 2025, 0(0): 1–18
Table 1 Overview of RAG benchmarks and their evaluation datasets. Source Domain indicates the data origin (e.g., real-time news, specialized corpora),
and Special Points highlight unique or novel features (like domain-specific tasks, dynamic changes, or false-premise data).
Benchmark Time Dataset Name(s) Source Domain Special Points
RAGAS [61] 2023.09 WikiEval Post-2022 Wikipedia Manually labeled for faithfulness
FreshLLMs [56] 2023.11 FRESHQA Real-time news /web queries Dynamic QA with false-premise detection
RECALL [78] 2023.11 EventKG, UJ Multilingual KGs, sci. terms Edited /counterfactual context tests
ARES [73] 2023.11NQ [100], HotpotQA [101], FEVER [102],
WoW [103], MultiRC [104], ReCoRD [105]KILT and SuperGLUE corpora Re-uses classic QA sets, multi-domain
RGB [85] 2023.12 Custom corpus Latest news articles Emphasizes info integration, noise rejections
MultiHop-RAG [7] 2024.01 Generated corpus Daily news segments via mediastack Multi-hop cross-document queries
CRUD-RAG [106] 2024.02 Generated corpus, UHGEval Chinese news, domain texts Create /Read/Update /Delete tasks
MedRAG [107] 2024.02 MIRAGE Medical QA corpora Healthcare domain knowledge
FeB4RAG [108] 2024.02 FeB4RAG, BEIR [109] Federated search tasks Multi-domain, multi-engine retrieval
RAGBench [110] 2024.06PubMedQA, CovidQA, HotpotQA,
MS Marco, CUAD, DelucionQA,
EManual, TechQA, FinQA, TAT-QAMulti-domain corpora Faithfulness with TRACe (Util, Rel, Adh, Compl)
ReEval [111] 2024.05 NQ (MRQA) +RealTimeQA Wikipedia, real-time QAAdversarial test cases
for hallucination detection
DomainRAG [112] 2024.06 Generated admission QA College docs with yearly updates Single- /multi-doc, single- /multi-turn QA
Telecom RAG Eval. [113] 2024.07 TeleQuAD 3GPP-based domain docs Triple-labeled QA from SMEs (telecom context)
LegalBench-RAG [114] 2024.08PrivacyQA, CUAD, MAUD,
ContractNLIExpert-annotated legal corpora Emphasizes strict retrieval of legal text
RAGEval [115] 2024.08 DragonBall Finance, law, medical docs Schema-based generation, scenario-specific
CoURAGE [116] 2024.09 RealTimeQA [117], NQ [100] Online QA +KILT tasks Hallucination resilience, dynamic updates
RAG Unfairness [118] 2024.09 TREC22 FairRank, BBQ Wikipedia-based track +socioecon. QA Fairness metrics, group disparity
CoFE-RAG [119] 2024.10 CoFE data PDF, DOC, multi-lingual docs Fine-grained chunking, multi-keyword approach
OCR Hinders RAG [55] 2024.12 1,261 PDFs +8,561 images OCR text from scanned docs Evaluates noise from OCR errors
OmniEval [120] 2024.12 Finance domain set Financial docs, numeric tasks Emphasizes numeric correctness /factual QA
CRAG [121] 2024.12 KG+web corpus Knowledge graphs +web pages Multi-entity queries, curated dynamic facts
RAG Playground [122] 2024.12 319 QA pairs Curated multi-domain tasks Prompt engineering /user flows
MTRAG [123] 2025.01 CLAPNQ, FiQA, Govt, Cloud Wikipedia, finance, gov, tech docs Multi-turn, bridging queries
CDQA [124] 2025.01 Chinese Dynamic QA Recent Chinese news queries Time-varying evolving answers
U-NIAH [125] 2025.03 Starlight Academy Synthetic “needle-in-haystack” data Evaluates extremely long contexts
SCARF [54] 2025.04 (User-provided) Generic multi-domainModular or black-box approach
integrates wide metrics (LLM judge)
•Operational Overhead : Human supervision, system main-
tenance, and regular updates to knowledge bases.
•Development Costs : Initial implementation, integra-
tion, and customization expenses.
For more details in the token-based expenses, LLM providers
such as OpenAI and Google o ffer token usage metrics that
track input and output token consumption during evaluation
processes. This approach calculates costs by multiplying to-
ken counts by their respective pricing rates [127]. Researchers
have developed metrics to evaluate the economic e fficiency of
RAG implementations:
•Cost-E ffectiveness Ratio : Measures performance im-
provement per unit of cost, allowing for standardized
comparison between di fferent RAG configurations [127].
•Retrieval Precision ROI : Quantifies the economic re-
turn of improving retrieval precision by measuring the
reduction in irrelevant context processing costs [127].
This metric demonstrated that optimizing retrieval can
improve cost e fficiency by up to around 50% through
reducing token consumption during LLM inference.
•User-Controllable Cost-Accuracy Tradeo ffs: Su et al.
[128] propose evaluation methods using an interpretable
control parameter ( α) that allows systematic assessment
of the relationship between retrieval costs and accu-
racy. This approach enables evaluating RAG systemsacross a spectrum of cost constraints rather than at fixed
operating points.
•Comparative Cost Analysis : Methodologies for eval-
uating relative cost e fficiency between di fferent RAG
implementations for specific use cases, considering both
direct costs and long-term economic sustainability [129].
5 Resources
The evaluation methodologies previously examined are com-
prehensive, though not necessarily abundant. This section
systematically compiles, categorizes, and presents the imple-
mented RAG evaluation frameworks, benchmarks, analytical
tools, and datasets that have emerged in the large language
model era. To our knowledge, this compilation constitutes the
most exhaustive collection of RAG evaluation frameworks
currently documented in the literature.
Datasets . We compiled the benchmarks along with the as-
sociated datasets in recent years. Early works focus on static
general-purpose QA datasets (e.g., NQ [100], HotpotQA [101]),
providing well-established baselines but lack recency or do-
main specificity. Recent benchmarks counter these limita-
tions by 1) sourcing live news or rapidly updated online doc-
uments (e.g., RGB [85], MultiHop-RAG [7]) to test time-
sensitive capabilities; 2) curating domain-specific corpora in

Aoran GAN et al. Retrieval Augmented Generation Evaluation in the Era of LLMs 11
Table 2 RAG evaluation frameworks, highlighting principal evaluation targets and methods. Retrieval focuses mainly on Relevance (R), Correctness
(C) or Comprehensiveness, whereas generation (right) focuses on Faithfulness (F), Correctness (C), or Relevance (R). External evaluation targets ( safety ,
efficiency ) or other statements appear in italics.
Type Framework Time Raw Targets Retrieval Metrics Generation Metrics
Research FiD-Light [99] 2023.07 Latency – –
Research Diversity Reranker [40] 2023.08 Diversity Cosine Distances –
Benchmark RAGAS [61] 2023.09 Context R, Answer R, F LLM as JudgeLLM CosSim,
LLM as Judge
Tool TruEra RAG Triad [130] 2023.10 Context R, Answer R, Groundedness LLM as Judge LLM as Judge
Tool LangChain Bench. [131] 2023.11 C, F, ExecutionTime ,EmbCosDist Exact-match LLM as Judge
Benchmark FreshLLMs [56] 2023.11 Response C, Fast-changing ,False premise (retrieval logs)STRICT /RELAXED,
FRESHEVAL (LLM-based)
Tool RECALL [78] 2023.11 Response Quality, Robustness – BLEU, ROUGE-L
Benchmark ARES [73] 2023.11 Context R, Answer F, Answer R LLM +ClassifierLLM +Classifier,
LLM +Classifier
Benchmark RGB [85] 2023.12Info Integration, NoiseRobust ,
NegRejection ,Counterfact– Accuracy
Tool Databricks Eval [62] 2023.12 C,Readability , Comprehensiveness – LLM as Judge
Benchmark MultiHop-RAG [7] 2024.01 Retrieval C, Response C MAP, MRR, Hit@K LLM as Judge
Benchmark CRUD-RAG [106] 2024.02 Create, Read, Update, Delete –ROUGE, BLEU,
RAGQuestEval
Benchmark MedRAG [107] 2024.02 Accuracy (medical) – Exact-match, Acc.
Benchmark FeB4RAG [108] 2024.02 Consistency, C, Clarity ,Coverage –Human Eval,
Human Eval
Benchmark Arabic RAG Eval. [132] 2024.05 Doc R, Answer R nDCG, MRR, mAP Possibly CosSim to QA
Benchmark RAGBench [110] 2024.06Context R, Answer R, Explainability,
TRACe =Util, Rel, Adh, Compl.LLM-based Eval LLM-based Eval, TRACe Metrics
Benchmark ReEval [111] 2024.05Hallucination
Adversarial Attack–F1, EM, Entailment
LLM or Human Eval
Benchmark DomainRAG [112] 2024.06 C, F, NoiseRobust ,StructOutput –F1, EM,
ROUGE-L, LLM
Benchmark CoURAGE [116] 2024.06 Hallucination –F1, EM, LLM as Judge,
Human Eval
Tool Telecom RAG Eval. [113] 2024.07Context R, Faithfulness,
CorrectnessLLM-based Metrics RAGAS-based, LLM Eval
Benchmark LegalBench-RAG [114] 2024.08 Doc-level Precision, Citation Rel. Precision, Recall –
Benchmark RAGEval [115] 2024.08Completeness, Hallucination,
IrrelevanceLLM-based Scoring LLM-based, Human Alignment
Benchmark RAG Unfairness [118] 2024.09 Fairness , C, C MRR@K EM, ROUGE
Benchmark CoFE-RAG [119] 2024.10Fine-grained Retrieval, Resp Quality,
DiversityRecall, Correctness,
Multi-keywordBLEU, ROUGE-L, LLM as Judge
Benchmark Toward Instr.-Following [133] 2024.10 Instr. Relevance, Constraint –LLM as Judge,
Atomic Pass Rate
Benchmark OmniEval [120] 2024.12 Factual Acc., Domain Tasks Rule+LLM Manual or LLM FT
Benchmark CRAG [121] 2024.12Accuracy, Dynamism ,Complex Facts ,
R, CWeighted scoring Accuracy, Truthfulness measure
Benchmark OCR Hinders RAG [55] 2024.12Accuracy, OCR Noise ,
Semantic vs. Format NoiseEditDist ,LCS F1-score
Benchmark RAG Playground [122] 2024.12Retrieval Strategy,
Prompt Eng.Comparison-based LLM-based Eval
Benchmark MTRAG [123] 2025.01 Multi-turn Quality, Conv. C Recall, nDCG LLM as Judge
Benchmark CDQA [124] 2025.01 Accuracy – F1
Benchmark U-NIAH [125] 2025.03 Needle Detect, LongContext , No Halluc. Recall LLM Judge, Heatmap
Tool eRAG [134] 2024.04 Doc-level Rel. , Downstream Quality Doc-level LLM Kendall’sτ
Tool SCARF [54] 2025.04Context R, Answer R,
FaithfulnessLLM-based or
BLEU /ROUGERAGAS-like Relevance,
LLM-based
(Black-box Integration)
law, healthcare, or finance (e.g., MedRAG [107], OmniEval
[120], LegalBench-RAG [114]); or 3) generating synthetic
data or specialized QA pairs, possibly with false-premise or
counterfactual elements (e.g., FreshLLMs [56], RAGEval [115])
to assess robustness and misinformation handling. We fur-
ther provide a concise description of the original domains and
characteristics according to the original resource, as shown
in Table 1. Noted that only the datasets containing retrieved
ground truth documents are included, indicating a concern for
more in-depth system component evaluation.
Frameworks with Evaluation Methods . We compiled
and summarized the evaluation methods devised by exist-ing frameworks, as illustrated in Table 2. These e fforts span
from initial, point-level researches [40, 99] to later, multi-
component evaluation tools and benchmarks [73, 131], en-
compassing a remarkably comprehensive collection of assess-
ment frameworks. The evaluation methods employed are var-
ied, encompassing both traditional [78, 132] and LLM-based
metrics [106, 110]. Additionally, there are frameworks that
facilitate safety-focused evaluations [85, 116], or are tailored
to specific downstream domains like document [55,125], tele-
com [113], medicine [107], etc. Referencing the component
evaluation objectives outlined in section 3.1, we categorize
and highlight the evaluation elements and specific metrics.

12 Front. Comput. Sci., 2025, 0(0): 1–18
Retrieval Generation Safety Efficiency020406080100Percentage (%)89.3% 90.9%
9.1%47.3%
Fig. 3 Statistics on the distribution of RAG studies across four key areas:
retrieval, generation, safety, and e fficiency. A paper may utilize evaluation
methods in more than one areas.
Fig. 4 Frequency statistics wordcloud of evaluation metrics in RAG stud-
ies. The LLM-based methods are categorized based on the targets and pre-
sented with the su ffix ‘-LLM’. F-score refers to the expanded F1-score.
6 Discussion
6.1 Statistics and Analysis of RAG Evaluation
The proliferation of LLM has contributed to a significant di-
versification of RAG evaluation methods. Current researches,
while demonstrating comprehensive coverage of RAG eval-
uation dimensions, often subjectively assert their respective
utility statements. To assess the popularity of these evaluation
methods, we conducted a statistical analysis of the available
methods from a survey perspective. This can also be viewed
as a research-oriented simple meta-evaluation. We crawled
the collection of the papers since 2022 autumn with keywords
about RAG in the accepted papers of the high-level confer-
ences about NLP & AI, and extracted the component as well
as the evalauation metrics the papers focus and utilize. We
finally amassed a total of 582 PDF manuscripts. All the in-
cluded papers have undergone rigorous peer review, demon-
strating scholarly merit with complete experimental method-
ologies and logically structured evaluation procedures.
Research Focus . Figure 3 illustrates the statistical distri-
bution of evaluation methods used across the four di fferent
segments in RAG studies (Retrieval /Generation /Safety /
Efficiency). The data suggests a prevailing focus on internal
research and evaluation of RAG systems, as indicated by the
extensive coverage of the retrieval and generation processes.
In contrast, external evaluations, particularly those related to
safety, have garnered less attention.
Metric Preference . Word frequency counts were con-
ducted for the assessment metrics mentioned in the papers,
with the wordcloud displayed in Figure 4. Whenever a met-
ric is formally introduced in the body of a paper or reported in
the table of experimental results, its word frequency count is
set+1. We manually merged and mapped synonymous met-
2022 H2 2023 H1 2023 H2 2024 H1 2024 H2 2025 H1
Time010203040Count of LLM-as-Judge Papers
1 111
743
12Fig. 5 The number of papers explicitly mentioning LLM-based evaluation
on RAG. The 2025 H1 collection is up to March 31st.
rics in the same session and excluded the words with global
occurrences lower than twice. It is observed that traditional
metrics predominantly dominate the evaluation usage, while
LLM-based methods have not yet gained widespread accep-
tance among researchers. This phenomenon is attributed to
the simplicity and reliability of the conventional metrics. Con-
versely, the LLM-based methods often require more e ffort
and involve multiple settings that are di fficult to keep the
same across di fferent researches, such as the LLM version
and prompt design.
Trend of LLM Usage . Despite the potential issues with
LLM-based methods, there is an observable trend of increas-
ing application, as shown in Figure 5. 2024 H2 and 2025 H1
have the top two highest numbers. LLM judges are ultimately
capable of handling more complex designs, drawing closer to
real-world applications. LLM itself, additionally, has contin-
ued to evolve in recent years, with the performance progres-
sively improving, and the supported functions expanding.
6.2 Challenges and Future Directions
This section addresses several challenges inherent in contem-
porary RAG evaluation.
Limitations of LLM-based Methods . The current evalu-
ation design does not su fficiently address the timeliness and
the black-box nature inherent in the LLM. The method of
employing LLMs for assessments, particularly through di-
rect prompts, raises latent risk about stability and security.
Future research should focus on enhancing the robustness of
the evaluation process itself and minimizing the likelihood of
LLM errors in the RAG system.
Cost of Evaluation . The cost associated with the RAG
system has garnered attention. Nevertheless, a thorough eval-
uation remains expensive due to the vast scale of the tools
and datasets involved. Determining an e fficient method for
system evaluation, or striking a balance between cost and ef-
fectiveness, is one of the directions for future research.
Advanced Evaluation Methods . As LLMs continue to
evolve, the components of RAG systems are becoming more
diverse. Currently, many of these components are evaluated
using end-to-end RAG ontology metrics, with a lack of com-
prehensive functional decomposition evaluation or theoreti-
cal analysis. Concurrently, there remains untapped potential
in the functionalities of LLMs themselves. For instance, the

Aoran GAN et al. Retrieval Augmented Generation Evaluation in the Era of LLMs 13
evaluation about deep thinking models (e.g. openai-o1 [135])
along with the thinking process of LLMs in conjunction with
RAG’s retrieval and generation process, is still inadequate.
These in-depth evaluation strategies require further research
and development in the future.
Comprehensiveness of the Evaluation Framework . De-
spite the abundant evaluation frameworks at present, individ-
ual ones are somewhat limited in their metrics and methods of
evaluation. Moreover, most contemporary frameworks con-
centrate on widely used languages such as English and Chi-
nese. There is an urgent need for frameworks that are not
only methodologically but also linguistically diverse.
7 Conclusion
In this paper, we have presented the first comprehensive sur-
vey of RAG evaluation methodologies in the LLM era. Our
systematic analysis reveals several important insights for re-
searchers and practitioners working with these increasingly
prevalent systems. For the evaluation of internal RAG per-
formance, we dissect the internal components of RAG sys-
tems, define the assessment objectives, and gather a range of
methods and metrics from traditional to innovative. More-
over, we investigate the external evaluation related to system
integrity such as safety and e fficiency, which are underex-
plored in RAG research according to our statistical analy-
sis. Additionally, we compile and categorize the current eval-
uation datasets and frameworks to elucidate the unique at-
tributes and assessment focuses of the resources. Last but not
least, we analyze the implementation of existing evaluation
methods and synthesize the challenges and future directions
of RAG evaluation in the LLM era.
Acknowledgements
Competing interests The authors declare that they have no competing
interests or financial conflicts to disclose.
References
1. Fan W, Ding Y , Ning L, Wang S, Li H, Yin D, Chua T S, Li Q. A sur-
vey on rag meeting llms: Towards retrieval-augmented large language
models. In: Proceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining. 2024, 6491–6501
2. Guti ´errez B J, Shu Y , Gu Y , Yasunaga M, Su Y . Hipporag: Neuro-
biologically inspired long-term memory for large language models.
arXiv preprint arXiv:2405.14831, 2024
3. Zhang Y , Khalifa M, Logeswaran L, Lee M, Lee H, Wang L. Merg-
ing Generated and Retrieved Knowledge for Open-Domain QA. In:
Bouamor H, Pino J, Bali K, eds, Proceedings of the 2023 Conference
on Empirical Methods in Natural Language Processing. December
2023, 4710–47284. Yao J Y , Ning K P, Liu Z H, Ning M N, Yuan L. Llm lies: Hallu-
cinations are not bugs, but features as adversarial examples. arXiv
preprint arXiv:2310.01469, 2023
5. Wang L, Yang N, Huang X, Jiao B, Yang L, Jiang D, Majumder
R, Wei F. Text embeddings by weakly-supervised contrastive pre-
training. arXiv preprint arXiv:2212.03533, 2022
6. Robertson S, Zaragoza H, others . The probabilistic relevance frame-
work: Bm25 and beyond. Foundations and Trends ®in Information
Retrieval, 2009, 3(4): 333–389
7. Tang Y , Yang Y . Multihop-rag: Benchmarking retrieval-augmented
generation for multi-hop queries. arXiv preprint arXiv:2401.15391,
2024
8. Sun J, Xu C, Tang L, Wang S, Lin C, Gong Y , Shum H Y , Guo J.
Think-on-graph: Deep and responsible reasoning of large language
model with knowledge graph. CoRR, 2023
9. Gao Y , Xiong Y , Gao X, Jia K, Pan J, Bi Y , Dai Y , Sun J, Wang H,
Wang H. Retrieval-augmented generation for large language models:
A survey. arXiv preprint arXiv:2312.10997, 2023, 2
10. Brown T, Mann B, Ryder N, Subbiah M, Kaplan J D, Dhariwal P,
Neelakantan A, Shyam P, Sastry G, Askell A, others . Language mod-
els are few-shot learners. Advances in neural information processing
systems, 2020, 33: 1877–1901
11. Zhao W X, Zhou K, Li J, Tang T, Wang X, Hou Y , Min Y , Zhang B,
Zhang J, Dong Z, others . A survey of large language models. arXiv
preprint arXiv:2303.18223, 2023
12. Yildirim I, Paul L. From task structures to world models: what do
llms know? Trends in Cognitive Sciences, 2024
13. Zhang S, Dong L, Li X, Zhang S, Sun X, Wang S, Li J, Hu R, Zhang
T, Wu F, others . Instruction tuning for large language models: A
survey. arXiv preprint arXiv:2308.10792, 2023
14. Verma P, Pilanci M. Towards signal processing in large language
models. arXiv preprint arXiv:2406.10254, 2024
15. Lyu H, Jiang S, Zeng H, Xia Y , Wang Q, Zhang S, Chen R, Leung C,
Tang J, Luo J. Llm-rec: Personalized recommendation via prompting
large language models. In: Findings of the Association for Computa-
tional Linguistics: NAACL 2024. 2024, 583–612
16. Zhang B, Liu Z, Cherry C, Firat O. When scaling meets llm fine-
tuning: The e ffect of data, model and finetuning method. In: ICLR.
2024
17. Reynolds L, McDonell K. Prompt programming for large language
models: Beyond the few-shot paradigm. In: Extended abstracts of the
2021 CHI conference on human factors in computing systems. 2021,
1–7
18. Wei J, Wang X, Schuurmans D, Bosma M, Xia F, Chi E, Le Q V , Zhou
D, others . Chain-of-thought prompting elicits reasoning in large lan-
guage models. Advances in neural information processing systems,
2022, 35: 24824–24837
19. Huang Y , Huang J. A survey on retrieval-augmented text generation
for large language models. arXiv preprint arXiv:2404.10981, 2024
20. Zhou Y , Liu Y , Li X, Jin J, Qian H, Liu Z, Li C, Dou Z, Ho T Y ,
Yu P S. Trustworthiness in retrieval-augmented generation systems:
A survey. arXiv preprint arXiv:2409.10102, 2024

14 Front. Comput. Sci., 2025, 0(0): 1–18
21. Yu H, Gan A, Zhang K, Tong S, Liu Q, Liu Z. Evaluation of retrieval-
augmented generation: A survey. In: CCF Conference on Big Data.
2024, 102–120
22. Lewis P, Perez E, Piktus A, Petroni F, Karpukhin V , Goyal N, K ¨uttler
H, Lewis M, Yih W t, Rockt ¨aschel T, others . Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in neural
information processing systems, 2020, 33: 9459–9474
23. Li H, Su Y , Cai D, Wang Y , Liu L. A survey on retrieval-augmented
text generation. arXiv preprint arXiv:2202.01110, 2022
24. Dinan E, Roller S, Shuster K, Fan A, Auli M, Weston J. Wiz-
ard of wikipedia: Knowledge-powered conversational agents. arXiv
preprint arXiv:1811.01241, 2018
25. Qin L, Galley M, Brockett C, Liu X, Gao X, Dolan W B, Choi Y , Gao
J. Conversing by reading: Contentful neural conversation with on-
demand machine reading. In: Proceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics. 2019, 5427–
5436
26. Kobayashi M, Takeda K. Information retrieval on the web. ACM
computing surveys (CSUR), 2000, 32(2): 144–173
27. Lee H, Yang S, Oh H, Seo M. Generative multi-hop retrieval. In:
Proceedings of the 2022 Conference on Empirical Methods in Natural
Language Processing. 2022, 1417–1436
28. Zhang S, Yao L, Sun A, Tay Y . Deep learning based recommender
system: A survey and new perspectives. ACM Computing Surveys,
2019, 52(1): 1–38
29. Wang W, Lin X, Feng F, He X, Chua T S. Generative recom-
mendation: Towards next-generation recommender paradigm. arXiv
preprint arXiv:2304.03516, 2023
30. Karpukhin V , Oguz B, Min S, Lewis P, Wu L, Edunov S, Chen D, Yih
W t. Dense passage retrieval for open-domain question answering.
In: Webber B, Cohn T, He Y , Liu Y , eds, Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing
(EMNLP). November 2020, 6769–6781
31. Google . Programmable Search Engine |Google for Developers, 2024
32. Yepes A J, You Y , Milczek J, Laverde S, Li R. Financial report
chunking for e ffective retrieval augmented generation. arXiv preprint
arXiv:2402.05131, 2024
33. Fan W, Ding Y , Ning L, Wang S, Li H, Yin D, Chua T S, Li Q. A sur-
vey on rag meeting llms: Towards retrieval-augmented large language
models. In: Proceedings of the 30th ACM SIGKDD Conference on
Knowledge Discovery and Data Mining. 2024, 6491–6501
34. Singh I S, Aggarwal R, Allahverdiyev I, Taha M, Akalin A, Zhu K,
O’Brien S. Chunkrag: Novel llm-chunk filtering method for rag sys-
tems. arXiv preprint arXiv:2410.19572, 2024
35. Multi-Granularity M L M F. M3-embedding: Multi-linguality,
multi-functionality, multi-granularity text embeddings through self-
knowledge distillation. 2024
36. Mao Y , He P, Liu X, Shen Y , Gao J, Han J, Chen W. Generation-
augmented retrieval for open-domain question answering. In: Zong
C, Xia F, Li W, Navigli R, eds, Proceedings of the 59th Annual Meet-
ing of the Association for Computational Linguistics and the 11th
International Joint Conference on Natural Language Processing (V ol-ume 1: Long Papers). August 2021, 4089–4100
37. Mekala D, Vu T, Schick T, Shang J. Leveraging QA datasets to
improve generative data augmentation. In: Goldberg Y , Kozareva
Z, Zhang Y , eds, Proceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing. December 2022, 9737–
9750
38. Asai A, Wu Z, Wang Y , Sil A, Hajishirzi H. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection. In: The Twelfth
International Conference on Learning Representations. 2023
39. Douze M, Guzhva A, Deng C, Johnson J, Szilvasy G, Mazar ´e P E,
Lomeli M, Hosseini L, J ´egou H. The faiss library. CoRR, 2024
40. Blagojevic V . Enhancing RAG Pipelines in Haystack: Introducing
DiversityRanker and LostInTheMiddleRanker, August 2023
41. Besta M, Blach N, Kubicek A, Gerstenberger R, Podstawski M, Gi-
aninazzi L, Gajda J, Lehmann T, Niewiadomski H, Nyczyk P, others
. Graph of thoughts: Solving elaborate problems with large language
models. In: Proceedings of the AAAI Conference on Artificial Intel-
ligence. 2024, 17682–17690
42. Lanchantin J, Toshniwal S, Weston J, Sukhbaatar S, others . Learning
to reason and memorize with self-notes. Advances in Neural Infor-
mation Processing Systems, 2023, 36: 11891–11911
43. Deng Y , Zhang W, Chen Z, Gu Q. Rephrase and respond: Let large
language models ask better questions for themselves. CoRR, 2023
44. Wang C, Liu X, Yue Y , Tang X, Zhang T, Jiayang C, Yao Y , Gao
W, Hu X, Qi Z, others . Survey on factuality in large language
models: Knowledge, retrieval and domain-specificity. arXiv preprint
arXiv:2310.07521, 2023
45. Zhao P, Zhang H, Yu Q, Wang Z, Geng Y , Fu F, Yang L, Zhang W,
Cui B. Retrieval-augmented generation for ai-generated content: A
survey. CoRR, 2024
46. Cheng M, Luo Y , Ouyang J, Liu Q, Liu H, Li L, Yu S, Zhang B, Cao
J, Ma J, others . A survey on knowledge-oriented retrieval-augmented
generation. arXiv preprint arXiv:2503.10677, 2025
47. J ¨arvelin K, Kek ¨al¨ainen J. Cumulated gain-based evaluation of ir tech-
niques. ACM Transactions on Information Systems (TOIS), 2002,
20(4): 422–446
48. Sanko ffD, Kruskal J B. Time warps, string edits, and macro-
molecules: the theory and practice of sequence comparison. Reading:
Addison-Wesley Publication, 1983
49. Yujian L, Bo L. A normalized levenshtein distance metric. IEEE
transactions on pattern analysis and machine intelligence, 2007,
29(6): 1091–1095
50. Lin C Y . ROUGE: A package for automatic evaluation of summaries.
In: Text Summarization Branches Out. July 2004, 74–81
51. Papineni K, Roukos S, Ward T, Zhu W J. Bleu: a method for auto-
matic evaluation of machine translation. In: Isabelle P, Charniak E,
Lin D, eds, Proceedings of the 40th Annual Meeting of the Associa-
tion for Computational Linguistics. July 2002, 311–318
52. Banerjee S, Lavie A. Meteor: An automatic metric for mt evaluation
with improved correlation with human judgments. In: Proceedings
of the acl workshop on intrinsic and extrinsic evaluation measures for
machine translation and /or summarization. 2005, 65–72

Aoran GAN et al. Retrieval Augmented Generation Evaluation in the Era of LLMs 15
53. Zhang T, Kishore V , Wu F, Weinberger K Q, Artzi Y . BERTScore:
Evaluating Text Generation with BERT. In: 8th International Con-
ference on Learning Representations, ICLR 2020, Addis Ababa,
Ethiopia, April 26-30, 2020. 2020
54. Rengo M, Beadini S, Alfano D, Abbruzzese R. A system for
comprehensive assessment of rag frameworks. arXiv preprint
arXiv:2504.07803, 2025
55. Zhang J, Zhang Q, Wang B, Ouyang L, Wen Z, Li Y , Chow K H, He C,
Zhang W. Ocr hinders rag: Evaluating the cascading impact of ocr
on retrieval-augmented generation. arXiv preprint arXiv:2412.02592,
2024
56. Vu T, Iyyer M, Wang X, Constant N, Wei J, Wei J, Tar C, Sung Y H,
Zhou D, Le Q, Luong T. FreshLLMs: Refreshing large language
models with search engine augmentation. In: Ku L W, Martins A,
Srikumar V , eds, Findings of the Association for Computational Lin-
guistics: ACL 2024. August 2024, 13697–13720
57. Sælemyr J, Femdal H T. Chunk smarter, retrieve better: Enhancing
llms in finance: An empirical comparison of chunking techniques in
retrieval augmented generation for financial reports. Master’s thesis,
NORWEGIAN SCHOOL OF ECONOMICS, 2024
58. Finardi P, Avila L, Castaldoni R, Gengo P, Larcher C, Piau M, Costa
P, Carid ´a V . The chronicles of rag: The retriever, the chunk and the
generator. arXiv preprint arXiv:2401.07883, 2024
59. Muennigho ffN, Tazi N, Magne L, Reimers N. Mteb: Massive text
embedding benchmark. arXiv preprint arXiv:2210.07316, 2022
60. Enevoldsen K, Chung I, Kerboua I, Kardos M, Mathur A, Stap D,
Gala J, Siblini W, Krzemi ´nski D, Winata G I, others . Mmteb:
Massive multilingual text embedding benchmark. arXiv preprint
arXiv:2502.13595, 2025
61. Es S, James J, Anke L E, Schockaert S. Ragas: Automated evalua-
tion of retrieval augmented generation. In: Proceedings of the 18th
Conference of the European Chapter of the Association for Compu-
tational Linguistics: System Demonstrations. 2024, 150–158
62. Leng Q, Uhlenhuth K, Polyzotis A. Best practices for llm eval-
uation of rag applications (2023). URL https: //www. databricks.
com/blog/LLM-auto-eval-best-practices-RAG
63. Zhang H, Semnani S, Ghassemi F, Xu J, Liu S, Lam M. Spaghetti:
Open-domain question answering from heterogeneous data sources
with retrieval and semantic parsing. In: Findings of the Association
for Computational Linguistics ACL 2024. 2024, 1663–1678
64. Finsås M, Maksim J. Optimizing rag systems for technical support
with llm-based relevance feedback and multi-agent patterns. Master’s
thesis, NTNU, 2024
65. Patil S G, Zhang T, Wang X, Gonzalez J E. Gorilla: Large language
model connected with massive apis. Advances in Neural Information
Processing Systems, 2024, 37: 126544–126565
66. Dai L, Xu Y , Ye J, Liu H, Xiong H. Seper: Measure retrieval util-
ity through the lens of semantic perplexity reduction. arXiv preprint
arXiv:2503.01478, 2025
67. Qi Z, Xu R, Guo Z, Wang C, Zhang H, Xu W. Long2rag: Evalu-
ating long-context & long-form retrieval-augmented generation with
key point recall. In: Findings of the Association for ComputationalLinguistics: EMNLP 2024. 2024, 4852–4872
68. Li M, Li X, Chen Y , Xuan W, Zhang W. Unraveling and mitigating
retriever inconsistencies in retrieval-augmented large language mod-
els. In: Findings of the Association for Computational Linguistics
ACL 2024. 2024, 4833–4850
69. Min S, Krishna K, Lyu X, Lewis M, Yih W t, Koh P, Iyyer M, Zettle-
moyer L, Hajishirzi H. Factscore: Fine-grained atomic evaluation
of factual precision in long form text generation. In: Proceedings
of the 2023 Conference on Empirical Methods in Natural Language
Processing. 2023, 12076–12100
70. Song Y , Kim Y , Iyyer M. Veriscore: Evaluating the factuality
of verifiable claims in long-form text generation. arXiv preprint
arXiv:2406.19276, 2024
71. Chen L, Zhang R, Guo J, Fan Y , Cheng X. Controlling risk of
retrieval-augmented generation: A counterfactual prompting frame-
work. In: Findings of the Association for Computational Linguistics:
EMNLP 2024. 2024, 2380–2393
72. Fu J, Ng S K, Jiang Z, Liu P. Gptscore: Evaluate as you desire. In:
Proceedings of the 2024 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies (V olume 1: Long Papers). 2024, 6556–6576
73. Saad-Falcon J, Khattab O, Potts C, Zaharia M. Ares: An automated
evaluation framework for retrieval-augmented generation systems. In:
Proceedings of the 2024 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies (V olume 1: Long Papers). 2024, 338–354
74. Zhao X, Zhang H, Pan X, Yao W, Yu D, Chen J. Thrust: Adaptively
propels large language models with external knowledge. Advances in
Neural Information Processing Systems, 2023, 36: 69930–69948
75. Zhu K, Feng X, Du X, Gu Y , Yu W, Wang H, Chen Q, Chu Z,
Chen J, Qin B. An information bottleneck perspective for e ffec-
tive noise filtering on retrieval-augmented generation. arXiv preprint
arXiv:2406.01549, 2024
76. Li D, Yan J, Zhang T, Wang C, He X, Huang L, Xue H, Huang J. On
the role of long-tail knowledge in retrieval augmented large language
models. arXiv preprint arXiv:2406.16367, 2024
77. Sun Z, Zang X, Zheng K, Song Y , Xu J, Zhang X, Yu W, Li H. Re-
deep: Detecting hallucination in retrieval-augmented generation via
mechanistic interpretability. arXiv preprint arXiv:2410.11414, 2024
78. Liu Y , Huang L, Li S, Chen S, Zhou H, Meng F, Zhou J, Sun X. Re-
call: A benchmark for llms robustness against external counterfactual
knowledge. arXiv preprint arXiv:2311.08147, 2023
79. Wu S, Xie J, Chen J, Zhu T, Zhang K, Xiao Y . How easily do ir-
relevant inputs skew the responses of large language models? arXiv
preprint arXiv:2404.03302, 2024
80. Liang X, Niu S, Li Z, Zhang S, Wang H, Xiong F, Fan J Z, Tang
B, Song S, Wang M, others . Saferag: Benchmarking security
in retrieval-augmented generation of large language model. arXiv
preprint arXiv:2501.18636, 2025
81. Kang M, G ¨urel N M, Yu N, Song D, Li B. C-rag: Certified genera-
tion risks for retrieval-augmented language models. In: International
Conference on Machine Learning. 2024, 22963–23000

16 Front. Comput. Sci., 2025, 0(0): 1–18
82. Cheng X, Wang X, Zhang X, Ge T, Chen S Q, Wei F, Zhang H, Zhao
D. xrag: Extreme context compression for retrieval-augmented gen-
eration with one token. arXiv preprint arXiv:2405.13792, 2024
83. Asai A, Wu Z, Wang Y , Sil A, Hajishirzi H. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection. In: The Twelfth
International Conference on Learning Representations. 2023
84. Trivedi H, Balasubramanian N, Khot T, Sabharwal A. Interleav-
ing retrieval with chain-of-thought reasoning for knowledge-intensive
multi-step questions. In: ACL (1). 2023, 10014–10037
85. Chen J, Lin H, Han X, Sun L. Benchmarking large language mod-
els in retrieval-augmented generation. In: Proceedings of the AAAI
Conference on Artificial Intelligence. 2024, 17754–17762
86. Zou W, Geng R, Wang B, Jia J. Poisonedrag: Knowledge corruption
attacks to retrieval-augmented generation of large language models.
arXiv preprint arXiv:2402.07867, 2024
87. Zhang Y , Li Q, Du T, Zhang X, Zhao X, Feng Z, Yin J. Hijackrag:
Hijacking attacks against retrieval-augmented large language models.
arXiv preprint arXiv:2410.22832, 2024
88. Chaudhari H, Severi G, Abascal J, Jagielski M, Choquette-Choo C A,
Nasr M, Nita-Rotaru C, Oprea A. Phantom: General trigger at-
tacks on retrieval augmented language generation. arXiv preprint
arXiv:2405.20485, 2024
89. Shafran A, Schuster R, Shmatikov V . Machine against the rag: Jam-
ming retrieval-augmented generation with blocker documents. arXiv
preprint arXiv:2406.05870, 2024
90. Zeng S, Zhang J, He P, Liu Y , Xing Y , Xu H, Ren J, Chang Y , Wang
S, Yin D, others . The good and the bad: Exploring privacy issues in
retrieval-augmented generation (rag). In: Findings of the Association
for Computational Linguistics ACL 2024. 2024, 4505–4524
91. Cheng P, Ding Y , Ju T, Wu Z, Du W, Yi P, Zhang Z, Liu G. Trojan-
rag: Retrieval-augmented generation can be backdoor driver in large
language models. arXiv preprint arXiv:2405.13401, 2024
92. Chaudhari H, Severi G, Abascal J, Jagielski M, Choquette-Choo C A,
Nasr M, Nita-Rotaru C, Oprea A. Phantom: General trigger at-
tacks on retrieval augmented language generation. arXiv preprint
arXiv:2405.20485, 2024
93. Perez E, Huang S, Song F, Cai T, Ring R, Aslanides J, Glaese A,
McAleese N, Irving G. Red teaming language models with language
models, 2022
94. Shrestha R, Zou Y , Chen Q, Li Z, Xie Y , Deng S. Fairrag: Fair
human generation via fair retrieval augmentation. CoRR, 2024,
abs/2403.19964
95. Zhou Y , Liu Z, Jin J, Nie J Y , Dou Z. Metacognitive retrieval-
augmented large language models. In: WWW. 2024, 1453–1463
96. Sudhi V , Bhat S R, Rudat M, Teucher R. Rag-ex: A generic frame-
work for explaining retrieval augmented generation. In: SIGIR. 2024,
2776–2780
97. Ding T, Banerjee A, Mombaerts L, Li Y , Borogovac T, Weinstein
J P D l C. Vera: Validation and evaluation of retrieval-augmented
systems. arXiv preprint arXiv:2409.03759, 2024
98. Anthropic . Reducing latency, January 2025
99. Hofst ¨atter S, Chen J, Raman K, Zamani H. FiD-Light: E fficient andEffective Retrieval-Augmented Text Generation. In: Proceedings of
the 46th International ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, SIGIR ’23. July 2023, 1437–
1447
100. Kwiatkowski T, Palomaki J, Redfield O, Collins M, Parikh A, Alberti
C, Epstein D, Polosukhin I, Devlin J, Lee K, Toutanova K, Jones L,
Kelcey M, Chang M W, Dai A M, Uszkoreit J, Le Q, Petrov S. Natu-
ral questions: A benchmark for question answering research. Trans-
actions of the Association for Computational Linguistics, 2019, 7:
453–466
101. Yang Z, Qi P, Zhang S, Bengio Y , Cohen W W, Salakhutdinov R,
Manning C D. HotpotQA: A dataset for diverse, explainable multi-
hop question answering. In: Conference on Empirical Methods in
Natural Language Processing (EMNLP). 2018
102. Thorne J, Vlachos A, Christodoulopoulos C, Mittal A. FEVER: a
large-scale dataset for fact extraction and VERification. In: NAACL-
HLT. 2018
103. Dinan E, Roller S, Shuster K, Fan A, Auli M, Weston J. Wizard
of Wikipedia: Knowledge-powered conversational agents. In: Pro-
ceedings of the International Conference on Learning Representations
(ICLR). 2019
104. DeYoung J, Jain S, Rajani N F, Lehman E, Xiong C, Socher R, Wal-
lace B C. Eraser: A benchmark to evaluate rationalized nlp models.
In: Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics. 2020, 4443–4458
105. Zhang S, Liu X, Liu J, Gao J, Duh K, Van Durme B. Record: Bridging
the gap between human and machine commonsense reading compre-
hension. arXiv preprint arXiv:1810.12885, 2018
106. Lyu Y , Li Z, Niu S, Xiong F, Tang B, Wang W, Wu H, Liu H, Xu T,
Chen E. Crud-rag: A comprehensive chinese benchmark for retrieval-
augmented generation of large language models. ACM Trans. Inf.
Syst., 2025, 43(2)
107. Xiong G, Jin Q, Lu Z, Zhang A. Benchmarking retrieval-augmented
generation for medicine. In: Findings of the Association for Compu-
tational Linguistics ACL 2024. 2024, 6233–6251
108. Wang S, Khramtsova E, Zhuang S, Zuccon G. Feb4rag: Evaluat-
ing federated search in the context of retrieval augmented generation.
In: Proceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 2024, 763–773
109. Kamalloo E, Thakur N, Lassance C, Ma X, Yang J H, Lin J. Re-
sources for brewing beir: reproducible reference models and an o ffi-
cial leaderboard, 2023
110. Friel R, Belyi M, Sanyal A. Ragbench: Explainable bench-
mark for retrieval-augmented generation systems. arXiv preprint
arXiv:2407.11005, 2024
111. Yu X, Cheng H, Liu X, Roth D, Gao J. ReEval: Automatic halluci-
nation evaluation for retrieval-augmented large language models via
transferable adversarial attacks. In: Duh K, Gomez H, Bethard S, eds,
Findings of the Association for Computational Linguistics: NAACL
2024. June 2024, 1333–1351
112. Wang S, Liu J, Song S, Cheng J, Fu Y , Guo P, Fang K, Zhu Y , Dou
Z. Domainrag: A chinese benchmark for evaluating domain-specific

Aoran GAN et al. Retrieval Augmented Generation Evaluation in the Era of LLMs 17
retrieval-augmented generation. CoRR, 2024
113. Roychowdhury S, Soman S, Ranjani H, Gunda N, Chhabra V , BALA
S K. Evaluation of rag metrics for question answering in the telecom
domain. ICML 2024 Workshop on Foundation Models in the Wild,
2024
114. Pipitone N, Alami G H. Legalbench-rag: A benchmark for
retrieval-augmented generation in the legal domain. arXiv preprint
arXiv:2408.10343, 2024
115. Zhu K, Luo Y , Xu D, Wang R, Yu S, Wang S, Yan Y , Liu Z, Han
X, Liu Z, others . Rageval: Scenario specific rag evaluation dataset
generation framework. CoRR, 2024
116. Galla D, Hoda S, Zhang M, Quan W, Yang T D, V oyles J. Courage: A
framework to evaluate rag systems. In: Rapp A, Di Caro L, Meziane
F, Sugumaran V , eds, Natural Language Processing and Information
Systems. 2024, 392–407
117. Kasai J, Sakaguchi K, Le Bras R, Asai A, Yu X, Radev D, Smith
N A, Choi Y , Inui K, others . Realtime qa: What’s the answer right
now? Advances in neural information processing systems, 2023, 36:
49025–49043
118. Wu X, Li S, Wu H T, Tao Z, Fang Y . Does RAG introduce unfairness
in LLMs? evaluating fairness in retrieval-augmented generation sys-
tems. In: Rambow O, Wanner L, Apidianaki M, Al-Khalifa H, Euge-
nio B D, Schockaert S, eds, Proceedings of the 31st International Con-
ference on Computational Linguistics. January 2025, 10021–10036
119. Liu J, Ding R, Zhang L, Xie P, Huang F. Cofe-rag: A comprehensive
full-chain evaluation framework for retrieval-augmented generation
with enhanced data diversity. arXiv preprint arXiv:2410.12248, 2024
120. Wang S, Tan J, Dou Z, Wen J R. Omnieval: An omnidirectional
and automatic rag evaluation benchmark in financial domain. arXiv
preprint arXiv:2412.13018, 2024
121. Yang X, Sun K, Xin H, Sun Y , Bhalla N, Chen X, Choudhary S, Gui
R D, Jiang Z W, Jiang Z, Kong L, Moran B, Wang J, Xu Y E, Yan
A, Yang C, Yuan E, Zha H, Tang N, Chen L, Sche ffer N, Liu Y , Shah
N, Wanga R, Kumar A, Yih W t, Dong X L. Crag - comprehensive
rag benchmark. In: Globerson A, Mackey L, Belgrave D, Fan A,
Paquet U, Tomczak J, Zhang C, eds, Advances in Neural Information
Processing Systems. 2024, 10470–10490
122. Papadimitriou I, Gialampoukidis I, Vrochidis S, others . Rag
playground: A framework for systematic evaluation of retrieval
strategies and prompt engineering in rag systems. arXiv preprint
arXiv:2412.12322, 2024
123. Katsis Y , Rosenthal S, Fadnis K, Gunasekara C, Lee Y S, Popa L,
Shah V , Zhu H, Contractor D, Danilevsky M. Mtrag: A multi-turn
conversational benchmark for evaluating retrieval-augmented gener-
ation systems. arXiv preprint arXiv:2501.03468, 2025
124. Xu Z, Li Y , Ding R, Wang X, Chen B, Jiang Y , Zheng H, Lu W, Xie
P, Huang F. Let llms take on the latest challenges! a chinese dynamic
question answering benchmark. In: Proceedings of the 31st Interna-
tional Conference on Computational Linguistics. 2025, 10435–10448
125. Gao Y , Xiong Y , Wu W, Huang Z, Li B, Wang H. U-niah: Unified
rag and llm evaluation for long context needle-in-a-haystack. arXiv
preprint arXiv:2503.00353, 2025126. Selvaraj T. Calculate the total cost of a retrieval augmented generation
(rag) solution, February 2024
127. Zhang J, Li G, Su J. Sage: A framework of precise retrieval for rag.
arXiv preprint arXiv:2503.01713, 2025
128. Su J, Healey J, Nakov P, Cardie C. Fast or better? balancing accuracy
and cost in retrieval-augmented generation with flexible user control.
CoRR, 2025
129. S ¸akar T, Emekci H. Maximizing rag e fficiency: A comparative analy-
sis of rag methods. Natural Language Processing, 2025, 31(1): 1–25
130. Datta A, Fredrikson M, Leino K, Lu K, Sen S, Shih R, Wang Z. Ex-
ploring conceptual soundness with trulens. In: NeurIPS 2021 Com-
petitions and Demonstrations Track. 2022, 302–307
131. LangChain . Evaluating rag architectures on benchmark tasks,
November 2023
132. Mahboub A, Za’ter M E, Al-Rfooh B, Estaitia Y , Jaljuli A, Hak-
ouz A. Evaluation of semantic search and its role in retrieved-
augmented-generation (rag) for arabic language. arXiv preprint
arXiv:2403.18350, 2024
133. Dong G, Song X, Zhu Y , Qiao R, Dou Z, Wen J R. Toward general
instruction-following alignment for retrieval-augmented generation.
arXiv preprint arXiv:2410.09584, 2024
134. Salemi A, Zamani H. Evaluating retrieval quality in retrieval-
augmented generation. In: Proceedings of the 47th International
ACM SIGIR Conference on Research and Development in Informa-
tion Retrieval, SIGIR ’24. 2024, 2395–2400
135. Jaech A, Kalai A, Lerer A, Richardson A, El-Kishky A, Low A, Hel-
yar A, Madry A, Beutel A, Carney A, others . Openai o1 system card.
arXiv preprint arXiv:2412.16720, 2024
Aoran Gan is working toward the PhD
degree in the School of Artificial In-
telligence and Data Science, University
of Science and Technology of China.
His research interests include text min-
ing, knowledge graph and large language
models.
Hao Yu is pursuing a MS degree at
McGill University and is a ffiliated with
Quebec Artificial Intelligence Institute.
His research focuses on multilingual and
low-resource NLP, as well as RAG sys-
tems for misinformation detection.

18 Front. Comput. Sci., 2025, 0(0): 1–18
Kai Zhang is an Associate Researcher at
the University of Science and Technology
of China. His general area of research is
natural language processing and knowl-
edge discovery. He is a member of ACM,
SIGIR, AAAI, and CCF.
Qi Liu is a professor in the School of Ar-
tificial Intelligence and Data Science at
USTC. His area of research is data mining
and knowledge discovery. He has pub-
lished prolifically in refereed journals and
conferences. He is an Associate Editor of
IEEE TBD and Neurocomputing.
Wenyu Yan is currently pursuing MS de-
gree in University of Science and Tech-
nology of China. His research interests
focus on conversational search, retrieval-
augmented generation, etc.
Zhenya Huang is currently an Associate
Professor with USTC. His main research
interests include data mining, knowledge
reasoning, natural language processing,
and intelligent education. He has pub-
lished more than 50 papers in refereed
journals and conference proceedings.
Shiwei Tong is a senior data scientist at
Tencent Company. His research focuses
on Game Data Mining and Game Appli-
cations driven by Large Language Mod-
els.
Enhong Chen is a professor in the School
of Computer Science and Technology at
USTC. His general area of research in-
cludes data mining and machine learn-
ing, social network analysis, and recom-
mender systems. He was on program
committees of numerous conferences in-
cluding SIGKDD, ICDM, and SDM.
Guoping Hu is senior vice president of
iFLYTEK, director of the National Key
Laboratory of Cognitive Intelligence. He
has been honored with the First Prize of
State Science and Technology Advance-
ment Award and garnered over 300 autho-
rized patents.