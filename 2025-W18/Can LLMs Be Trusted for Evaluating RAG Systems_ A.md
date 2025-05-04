# Can LLMs Be Trusted for Evaluating RAG Systems? A Survey of Methods and Datasets

**Authors**: Lorenz Brehme, Thomas Ströhle, Ruth Breu

**Published**: 2025-04-28 08:22:19

**PDF URL**: [http://arxiv.org/pdf/2504.20119v2](http://arxiv.org/pdf/2504.20119v2)

## Abstract
Retrieval-Augmented Generation (RAG) has advanced significantly in recent
years. The complexity of RAG systems, which involve multiple components-such as
indexing, retrieval, and generation-along with numerous other parameters, poses
substantial challenges for systematic evaluation and quality enhancement.
Previous research highlights that evaluating RAG systems is essential for
documenting advancements, comparing configurations, and identifying effective
approaches for domain-specific applications. This study systematically reviews
63 academic articles to provide a comprehensive overview of state-of-the-art
RAG evaluation methodologies, focusing on four key areas: datasets, retrievers,
indexing and databases, and the generator component. We observe the feasibility
of an automated evaluation approach for each component of a RAG system,
leveraging an LLM capable of both generating evaluation datasets and conducting
evaluations. In addition, we found that further practical research is essential
to provide companies with clear guidance on the do's and don'ts of implementing
and evaluating RAG systems. By synthesizing evaluation approaches for key RAG
components and emphasizing the creation and adaptation of domain-specific
datasets for benchmarking, we contribute to the advancement of systematic
evaluation methods and the improvement of evaluation rigor for RAG systems.
Furthermore, by examining the interplay between automated approaches leveraging
LLMs and human judgment, we contribute to the ongoing discourse on balancing
automation and human input, clarifying their respective contributions,
limitations, and challenges in achieving robust and reliable evaluations.

## Full Text


<!-- PDF content starts -->

arXiv:2504.20119v2  [cs.IR]  1 May 2025Can LLMs Be Trusted for Evaluating RAG
Systems? A Survey of Methods and Datasets
1stLorenz Brehme
Department of Computer Science
University of Innsbruck
Innsbruck, Austria
https://orcid.org/0009-0009-4711-25642ndThomas Str¨ ohle
Department of Computer Science
University of Innsbruck
Innsbruck, Austria
https://orcid.org/0000-0002-1954-64123rdRuth Breu
Department of Computer Science
University of Innsbruck
Innsbruck, Austria
https://orcid.org/0000-0001-7093-4341
Abstract —Retrieval-Augmented Generation (RAG) has ad-
vanced signiﬁcantly in recent years. The complexity of RAG
systems, which involve multiple components—such as indexi ng,
retrieval, and generation—along with numerous other param e-
ters, poses substantial challenges for systematic evaluat ion and
quality enhancement. Previous research highlights that ev aluating
RAG systems is essential for documenting advancements, com -
paring conﬁgurations, and identifying effective approach es for
domain-speciﬁc applications. This study systematically r eviews
63 academic articles to provide a comprehensive overview of
state-of-the-art RAG evaluation methodologies, focusing on four
key areas: datasets, retrievers, indexing and databases, a nd
the generator component. We observe the feasibility of an
automated evaluation approach for each component of a RAG
system, leveraging an LLM capable of both generating evalua tion
datasets and conducting evaluations. In addition, we found that
further practical research is essential to provide compani es with
clear guidance on the do’s and don’ts of implementing and
evaluating RAG systems. By synthesizing evaluation approa ches
for key RAG components and emphasizing the creation and
adaptation of domain-speciﬁc datasets for benchmarking, w e
contribute to the advancement of systematic evaluation met hods
and the improvement of evaluation rigor for RAG systems.
Furthermore, by examining the interplay between automated ap-
proaches leveraging LLMs and human judgment, we contribute
to the ongoing discourse on balancing automation and human
input, clarifying their respective contributions, limita tions, and
challenges in achieving robust and reliable evaluations.
I. I NTRODUCTION
In recent years, Large Language Models (LLMs) have made
signiﬁcant progress in research and have grown increasingl y
popular [1]. However, LLMs face several challenges, includ -
ing issues with hallucinations caused by insufﬁcient conte xt
[2], as well as limitations in their learned content, which
prevent them from addressing questions requiring speciﬁc
or proprietary information [1]. To address these issues, [3 ]
introduced Retrieval-Augmented Generation (RAG), which
extends LLMs by integrating external knowledge sources for
knowledge-intensive natural language processing (NLP) ta sks.
By incorporating domain-speciﬁc information, RAG systems
enable tailored responses for specialized topics, improvi ng
accuracy, relevance, and contextual understanding. Since 2022,
numerous RAG systems have demonstrated the effectiveness
of this approach in overcoming some limitations of LLMs [1].These systems are applied across a wide range of NLP tasks
and outperform in domain-speciﬁc scenarios by leveraging
specialized knowledge to enhance performance and relevanc e
[1]. RAG systems operate through three interconnected com-
ponents: (1) indexing, which structures and organizes ex-
ternal knowledge bases; (2) retrieval, which identiﬁes and
extracts relevant documents from these sources; (3) and gen -
eration, which combines retrieved information with the inp ut
to produce a coherent, contextually relevant response usin g
an LLM and prompt engineering techniques. These systems
offer numerous customization options, including ﬁne-tuni ng
retrieval mechanisms, optimizing prompting techniques, r e-
ﬁning generation models, and customizing knowledge base
design [4], [5]. This ﬂexibility raises the question of what
settings should be used to conﬁgure RAG systems, and how
these systems can be compared and evaluated to determine
the most effective system for speciﬁc domains. This particu lar
task of identifying optimal parameters in RAG systems is
commonly referred to in the literature as RAG evaluation
[6], [7]. During the RAG evaluation, each component of the
RAG system is systematically evaluated within a predeﬁned
workﬂow to ensure that the system meets the overall quality
standards [6]. The process begins with an input query from a
prepared QA evaluation dataset, which is used to compute
metrics such as retrieval accuracy and response relevance,
providing statistical insight into the performance of the R AG
components and thus a comprehensive understanding of their
effectiveness [6]. RAG evaluation presents several challe nges:
A major hurdle is the deﬁnition of robust methods for assess-
ing the quality of the system’s responses; frameworks such
as RAGAS [6] and ARES [7] provide comprehensive metrics
for evaluating the generator’s responses in terms of releva nce,
accuracy, and overall performance. Retriever evaluation f o-
cuses on evaluating the relevance of retrieved documents an d
determining whether the selected chunks effectively contr ibute
to answering the query [8], while indexing evaluation prima rily
emphasizes performance metrics such as indexing and retrie val
speed [5], [9]. Equally critical is the selection of an appro priate
database that both contains domain-speciﬁc information an d is
complex enough to effectively test the system’s ability to m eet
quality benchmarks [10], [11]. Furthermore, creating QA ev al-
uation datasets requires signiﬁcant human effort and domai n This paper has been accepted for presentation at the SDS25. © 2025 IEEE.

expertise, so they are often augmented with an artiﬁcial dat aset
generated by an LLM using pieces of given domain knowledge
to streamline the process [6], [11], [12]. Previous literat ure
reviews have either focused exclusively on evaluation of th e
retriever and generator components, overlooking indexing or
broader aspects of dataset generation and enhancement [13] , or
focused primarily on evaluation metrics and existing datas ets,
similarly omitting evaluation of indexing [14]. However, t here
is a growing body of research emphasizing the need for datase t
generation and enhancement [10], the importance of includi ng
indexing strategies and their impact on system performance
[9], and the importance of automation through LLMs [6] in
RAG evaluation. The aim of this paper is therefore to conduct
a systematic literature review (SLR) on the evaluation of RA G
systems, synthesizing existing research to identify best p ractice
and provide guidance on effective evaluation methodologie s.
We focus on examining datasets and the processes used to
produce and improve them, with the aim of understanding
their impact on the reliability of evaluations. Our analysi s
examines the evaluation of indexing, retrieval and generat ion
components, highlighting the metrics and methodologies us ed.
We also assess the interplay between automation, particula rly
through LLM, and human judgement, clarifying their roles
and limitations in the evaluation process. By distinguishi ng
between RAG tasks and their evaluation techniques, we offer
insights into the unique challenges associated with each ta sk
type. Finally, we offer a coherent overview that integrates these
ﬁndings and advances the understanding of effective RAG
system evaluations.
In Section II, we outline our approach to selecting relevant
literature and conducting the SLR. The subsequent sections
examine the evaluation of each RAG component. We begin
with the dataset evaluation in Section III, followed by an
assessment of indexing and database performance in Section
IV. Next, we evaluate the retriever and its methodologies in
Section V. Finally, in Section VI, we analyze the generator
tasks within the system. We conclude with a summary of the
current state of RAG evaluation.
II. R EASEARCH METHOD
For our analysis, we conducted an SLR following the
guidelines of [15]. The process began with identifying rele vant
studies using advanced search tools in online databases. We
selected keywords and formulated a search query based on
two key frameworks: RAGAS, an early automated evaluation
approach for RAGs [6], and [10], which benchmarks RAG sys-
tems for multi-hop questions. Boolean operators were appli ed,
and the query was iteratively reﬁned to improve relevance.
The ﬁnal search query was: ”RAG” OR ”Retrieval-Augmented
Generation” OR ”Retrieval Augmented Generation” OR ”re-
triev* augment* generation”; AND Evaluation OR ”Quality
Assessment” OR ”Benchmark” OR ”Performance Evalua-
tion” OR ”evaluat*” . Searches were conducted on November
11, 2024, with papers restricted to those published from 202 1
onward. The search covered ACM, IEEE, arXiv, Elsevier,
Google Scholar, and Web of Science—databases commonlyused in computer science research [15]. This initial search
yielded 71 papers: 50 from arXiv, four from ACM, one from
Elsevier, nine from Google Scholar, four from IEEE, two from
Springer, and one from Web of Science. We reviewed abstracts
and excluded papers that did not primarily focus on evaluati ng
or benchmarking RAG systems, narrowing the selection to 48
papers. A forward and backward search using Google Scholar
added 21 more papers, of which six did not meet our criteria.
Ultimately, our SLR included 63 relevant papers. Previous
literature reviews analyzed only 12 papers, with [13] focus ed
on retriever and generator evaluation but omitted indexing
and dataset enhancement, while [14] provided an overview of
evaluation datasets and methods, primarily discussing met rics
without addressing indexing evaluation. To structure our ﬁ nd-
ings, we categorized papers based on their focus: evaluatio n
of retrievers, generators, or embeddings. Additionally, w e
grouped them by dataset usage, distinguishing between gen-
erated datasets, enhanced datasets, and pre-existing data sets.
III. E VALUATION DATASETS
In this survey, we found 87 existing question and answer
(QA) datasets that were used to benchmark RAG systems.
These datasets differ in the types of questions, the do-
main—such as legal [16], medicine [17], or energy [18]—the
length of the questions, and the type of context they pro-
vide, each introducing unique challenges and requirements
for RAG system evaluation. The datasets contain question-
answer-context tuples, where the question serves as input
to the RAG system, and the context and answer act as
evaluation references. For example, for What is the capital
of Switzerland? , the context might be Bern is the capital of
Switzerland , and the answer Bern . Questions are categorized to
enable structured evaluation of the RAG system’s capabilit ies.
We identiﬁed ﬁve distinct categories of questions as being
used to evaluate RAG systems in this study: One prevalent typ e
includes questions with (1) short answers , which typically
consist of a single word, a number, or a Boolean variable. Thi s
category also encompasses single-choice and multiple-cho ice
questions, often requiring straightforward responses [19 ]. In
contrast, (2) long answer questions demand more detailed
responses, providing logical and well-reasoned explanati ons
[20]. Building on the complexity of long-answer questions,
(3) multi-hop questions extend the challenge by requiring
the integration of several pieces of context to form a cohere nt
answer. These questions test a RAG system’s capacity to
utilize multiple contexts and engage in logical reasoning
across several steps [10]. Some evaluations include (4) error-
type questions designed to expose system weaknesses by
presenting incorrect or illogical contexts, requiring the RAG
system to identify inconsistencies and respond appropriat ely
[21], [7], or trigger speciﬁc error cases within the system
[22], [23]. Datasets vary in structure, including question s,
answers, context, and sometimes misleading information. (5)
Misleading context datasets test a system’s ability to detect
and handle false information. In general, we observed that
datasets for evaluating RAG systems require more complex

questions and answers. To achieve this, multi-hop question s
and long answers were utilized [10].
There are existing datasets that are publicly available for QA
tasks that are in different domains. The most commonly used
datasets are HotPotQA, NaturalQuestions, and MSMarco, wit h
HotPotQA being particularly prominent. It includes questi ons
with long and short answers, multi-hop reasoning, and fake
context, making it a highly complex dataset. These datasets are
created based on publicly available knowledge, not proprie tary
knowledge, which presents a challenge because the RAG
system then becomes redundant since the LLM has already
been trained on this knowledge [24]. To avoid this issue,
existing databases were modiﬁed, or a complete new dataset
which could include the proprietary knowledge was created
[24]. These new datasets are particularly important for RAG
systems, which depend on domain-speciﬁc knowledge and are
customized to meet the needs of specialized areas. As a resul t,
evaluating them with public QA datasets is usually impracti cal.
In such cases, custom datasets must be created manually [25] ,
[26].
a) Creation by Humans: In [25], [27], [26], human
annotators develop datasets, with [25], [27] focusing on qu es-
tions crafted by humans in a speciﬁc ﬁeld and subsequently
evaluated by domain experts. One approach begins with
16 Yes/No questions on a domain-speciﬁc topic, iteratively
reﬁning concepts and developing complex, context-enriche d
questions [25]. When a QA dataset lacks complex answers, a
human annotator formulates detailed responses, which are t hen
reviewed by experts for a more comprehensive evaluation [26 ].
b) Enhancing existing datasets: Three methods were
identiﬁed to enhance existing datasets and develop new ones .
For instance, when datasets contain only questions and cont ext
without answers, an LLM can generate the answers based
on the provided context [22]. The second method generates
answers to reﬂect speciﬁc error types to test evaluators and
assess their robustness and reliability [22], [23], [21]. A n-
other method involves rewriting questions to enhance their
complexity and suitability for RAG evaluation [28], [29].
However, a risk with existing datasets is that LLMs may recal l
answers without using the given context, which is mitigated
by evolving and regenerating questions for novelty [30].
c) Generation of Datasets using an LLM: Dataset cre-
ation is time-consuming; to address this, evaluation frame -
works have been developed that generate datasets using LLMs .
A given context was used to generate a speciﬁc question
related to that context [6], [10]. Additionally, the LLM gen -
erated an answer corresponding to the question and context,
completing the tuple. The resulting questions were then ﬁlt ered
to reﬁne the dataset and improve its quality [6], [10]. The
generation process uses several prompting strategies, suc h as
chain-of-thought prompting, along with speciﬁc metrics to
derive the best questions for RAG evaluation [6], [31], [32] .
It has been demonstrated that LLMs can answer questions
using prior knowledge, even in the absence of context [24].
To prevent this, post-training articles were used for quest ion
generation, except in domain-speciﬁc areas where LLMs lackprior exposure.
To enhance evaluation datasets, [10] proposed generating
multi-hop questions requiring reasoning across multiple c on-
texts. For instance, [12] used one to ﬁve randomly sam-
pled chunks from a domain-speciﬁc topic to create multi-
hop queries. The approach was enhanced by using an LLM
to add keywords to each chunk and QA pair to address
missing critical information [33], [31], [30]. In addition , one
method addresses the generation of domain-speciﬁc questio ns
by evaluating different aspects of domain-speciﬁc tasks: T hese
datasets include a multi-hop dataset, a dataset focused on t able
information, and a dataset designed to test the faithfulnes s
of RAGs, where LLMs generate questions from documents
requiring speciﬁc knowledge [11], [18]. Another method is
to generate a noisy dataset[11], [18], [7], [34], [35], [36] ,
where irrelevant, misleading, or conﬂicting context is add ed to
the Question-Answer dataset. This approach aims to evaluat e
the ability of evaluators to detect hallucinations, assess the
faithfulness of the RAG, and test the performance of the
retriever.
Once the questions and answers have been created, they
need to be ﬁltered and selected for inclusion in the ﬁnal data set
to ensure quality and accuracy by an LLM or manually by
humans. In [18], [31], [32], [33], domain experts evaluate
the generated questions for relevance and clarity. In contr ast,
an LLM is used to evaluate datasets by assessing query-
context alignment, answerability, query type relevance, o r
independence, assigning a score for each criterion [10], [6 ].
IV. I NDEXING AND DATABASE EVALUATION
One crucial component of a RAG system is the database
used by the retriever. Key factors such as indexing, em-
beddings, and chunk size signiﬁcantly impact the system’s
performance. Two studies speciﬁcally focused on evaluatin g
these components [9], [5]. In [5] the authors evaluated data base
performance using four key metrics: upload time, indexing
time, retrieval speed, and throughput, providing quantita -
tive performance measurements. Additionally, [37] evalua tes
different embeddings, chunking strategies and databases b y
analyzing their impact on overall performance. In [9] they
focus on evaluating embedding models, particularly in the
context of RAG. The authors investigate the similarity of
embedding models by comparing their representations and th e
similarity of retrieved results for speciﬁc queries. To mea sure
the similarity of embeddings, [9] employs Centered Kernel
Alignment (CKA). For evaluating the similarity of retrieve d
contexts, the Jaccard similarity coefﬁcient was utilized a nd
the RankSimilarity score to account the ranking of retrieve d
text chunks were introduced. In total, the indexing compone nt
is primarily evaluated on performance metrics like indexin g
and retrieval speed, while other factors are assessed as par t of
overall system performance [9], [5].
V. E VALUATION OF RETRIEVER
A crucial aspect of evaluating a RAG system is assessing
the performance of its retrieval component. In our survey,

we identiﬁed 24 different papers that speciﬁcally address t he
evaluation (see Table I). In two studies, researchers adjus t the
retriever parameters and evaluate the generator’s perform ance
to analyze how changes in the retriever affect the overall
system [37], [5], while the remaining ones focus on evaluati on
methods tailored to assess the retriever itself. The primar y goal
of retriever evaluation is to determine whether the retriev ed
chunks are relevant to the given query. We identiﬁed seven di s-
tinct methods for determining document relevance, followe d
by the application of metrics as listed in table I to calculat e a
score reﬂecting the quality of the retriever. Among these, M ean
Reciprocal Rank (MRR) and Discounted Cumulative Gain
(NDCG) are the only metrics that account for the order of the
retrieved documents. Another approach evaluates addition al
the fairness of the retriever by examining whether the retri eved
documents fairly represent protected groups [38].
TABLE I
LIST OF METRICS FOCUSING ON RETRIEVER EVALUATION
Context relevance Metric Ref.
Labeled NDCG [39]
Accuracy, MRR [38]
Recall, MRR, Mean Average
Precision (MAP), NDCG[40]
Recall [12]
Recall, MRR, NDCG [41]
MRR, MAP, Hit Rate [10], [29]
Precision, F1-Score, Recall [25]
Only Context Relevance [20]
Keyword Labeling Recall, Accuracy [31]
Recall, Effective Information
Rate[33]
Human judges Accuracy [42]
LCS Only Context Relevance [43]
LLM Judge MRR [44]
Context Utilization [45]
Precision, Recall [46]
Only Context Relevance [7], [28]
LLM judge (Indirect) Precision, Recall, MRR,
MAP, NDCG, Hit Rate[8]
Only Context Relevance [47], [6], [18]
No Context Relevance Retrieval Time [5]
For evaluation, the context relevance metric is used to
calculate most of the other metrics. This metric determines
whether a context is relevant or not. In the following, the
different methods to assess the relevance of a context are
presented.
The simplest way to determine if a document is relevant to a
question is by using a labelled dataset . In such datasets, each
question is paired with a set of relevant contexts. A documen t
is considered relevant if it matches the contexts labelled a s
relevant in the dataset [40], [12]. However, these datasets often
contain only a predeﬁned set of documents, leading to errors
if truly relevant documents are excluded and wrongly labele d
as irrelevant [31], [33]. To address this issue, some approa ches
label each context with a keyword , and each question is also
associated with these keywords (see Section III-0c). If the
retrieved context contains the keyword associated with the
question, the context is deemed relevant [31], [33].
In cases where no labelled dataset is available, [42] relies on
human judges to determine whether the context is relevant.Additionally, in some instances, an extra human preference
validation set is created to compare against the newly de-
veloped approach for evaluation purposes [7], [28]. In [43] ,
traditional retrieval evaluation strategies are used. The
ground truth evidence is available, and the Longest Common
Subsequence (LCS) is calculated to measure the quality of
the retriever. One method of measuring context relevance is
to employ an LLM as judge to calculate a score indicating
whether the retrieved chunk is relevant. There are differen t
methods to use the LLM to measure this. One method involves
using the LLM directly as a binary classiﬁer to assess whether
a document is relevant to the question, with only slight
variations in the prompts used for different evaluations.[ 7].
The eRAG framework [8] evaluates chunk relevance using an
LLM indirectly by generating a question from each chunk
and assessing the correctness of the LLM’s answer, labeling
the context as relevant if accurate. Another approach evalu ates
relevance by comparing answers generated from retrieved
documents to those derived from golden documents, which are
essential for solving the question [47]. In RAGAS, relevanc e
is measured by the proportion of sentences used from the
provided context [6].
VI. E VALUATION OF GENERATOR
This section outlines the evaluation of the generator based
on a review of 56 relevant papers. These studies were grouped
into four categories: short answer evaluation , focusing on
methods for assessing multiple-choice or categorization t asks;
classical and embedding-based approaches , representing
traditional evaluation techniques predating large langua ge
models; LLMs as evaluators , where large language models
perform evaluations; and human evaluation , involving assess-
ments conducted by human judges.
A. Short Answers Evaluation
TABLE II
LIST OF METRICS FOR SHORT ANSWER EVALUATION
Metrics Papers
F1-Micro Score [48]
Precision [49], [50]
Recall [49], [50], [48], [51]
Error Rejection/Detection/
Correction Rate[35]
F1-Score [49], [51], [52], [43]
Accuracy [53], [54], [17], [35], [38], [19], [52], [10]
Error Rate [48]
One approach to evaluating the RAG system is to simply
determine whether the answer is correct. This can be au-
tomated when the answers are very short. The correctness
of an answer is measured using exact match metrics. This
approach applies to tasks such as short answers [35], multip le-
choice [53] and binary [49] or multi-class categorization [ 51].
Evaluation datasets include the correct answers or categor ies,
enabling automated detection of whether the RAG’s response
is correct. Based on this, the relevant metrics are calculat ed, as
summarized in Table II. This involves noisy datasets to calc u-
late error metrics to test the model’s robustness when handl ing

noisy or erroneous input data [35]. This evaluation approac h is
commonly used in simpler RAGs, where answers can be easily
assessed as correct or incorrect. However, for more complex
RAGs, where responses are not as straightforward, addition al
methods are required for comprehensive evaluation.
B. Human Evaluators
TABLE III
LIST OF METRICS FOR HUMAN EVALUATORS
Method Metrics Papers
Direct Correctness [55], [34], [56], [42]
Precision, Recall, Accuracy [16], [57]
Quality [55], [42]
Readability, Usefulness [42]
Confusing Questions Detection [23]
Score [54]
Comprehensive Coverage, Consistency, Correct-
ness, Clarity[29]
In eleven approaches, human evaluators were employed to
manually assess the responses of RAG systems. We identiﬁed
two distinct methods for assessing the quality of responses :
The ﬁrst method involves direct evaluation, where human eva l-
uators assess the quality of responses based on various metr ics,
as listed in Table III, along with the corresponding papers.
For example, one study tested the RAG system’s ability to
handle confusing questions by deliberately introducing er rors
in the context. Human evaluators then determined whether th e
RAG correctly identiﬁed or addressed these errors [23]. The
second method is the comprehensive evaluation, introduced in
Feb4Rag [29]. In this approach, two answers are presented
side by side, and human evaluators judge which answer is
superior based on predeﬁned metrics. Human evaluation was
especially applied in domain-speciﬁc contexts. For exampl e,
in the Legal-Bench RAG, domain experts assessed the cor-
rectness of responses to legal questions. These evaluation s
were then used to calculate metrics [16], [57]. Additionall y,
eight approaches utilized human judgment as a benchmark to
compare the performance of RAG systems against methods
where an LLM acted as a judge [58], [59], [26], [42], [20],
[44] or against classical evaluation methods [52], [42], [5 5].
C. Classical and Embedding-Based Approaches
TABLE IV
LIST OF METRICS FOR EMBEDDING BASED APPROACHES
Method Metrics Papers
Embedding SAS [5]
SBERT [52], [42]
BERTScore [42], [4]
N-Gram BLEU [31], [40], [12], [42], [44]
ROGUE-n [38], [5], [42], [50]
ROGUE-L [31], [40], [12], [11], [60],
[5], [42], [50], [44]
Unigram Precision/Recall [61]
Model Unieval [12]
Token Similarity [27], [61]
To automatically evaluate the performance of the generator
in a RAG system, the generated response is compared to
the ground truth answer. We identiﬁed four distinct methods
for automatically evaluating the generator’s performance . Onewidely used approach is embedding -based evaluation, which
assesses the semantic similarity between the generated ans wer
and the reference answer. Within this category, we identiﬁe d
three primary techniques. The ﬁrst is the Semantic Answer
Similarity (SAS) score, which employs a trained cross-enco der
architecture to evaluate the semantic alignment between ge ner-
ated and reference answers [5]. The second technique involv es
Sentence-BERT (SBERT), a specialized adaptation of the
BERT model designed for comparing sentences and evaluating
their semantic similarity [62], [63]. SBERT is particularl y
effective for tasks requiring textual alignment, making it
suitable for comparing RAG-generated answers with referen ce
answers [52], [42]. Lastly, BERTScore, another embedding-
based model built upon the BERT architecture, compares
token-level embeddings of sentences to measure similarity
[64], [42], [4]. In addition, n-gram -based metrics are com-
monly used. These metrics calculate scores automatically b y
analyzing sequences of words (n-grams) of a speciﬁed length .
Table IV lists the speciﬁc metrics used in this category. The
third method involves model -based evaluation, such as the
UniEval score. UniEval is designed to assess natural langua ge
generation by providing a comprehensive evaluation score
based on coherence, consistency, ﬂuency, and relevance. In
this approach, a model is trained using the ground truth as a
reference for evaluation [65], [12]. Finally, the fourth me thod
istoken-based evaluation, which measures the quality of a
generated response by calculating the ratio of overlapping
tokens between the generated answer and the ground truth,
using word segmentation tools [27]. Alternatively, this ca n be
achieved using cosine similarity, which measures the cosin e
of the angle between two vectors [61]. In four cases, these
metrics are extended by using a large language model (LLM)
as a judge [31], [60], [42], [50] or a human evaluator for
validation [52], [42].
D. LLM as a Judge
LLMs can serve as judges to evaluate the performance of
RAG systems, and we identiﬁed 41 papers that employed
LLMs for this purpose. Table V lists all metrics and the
corresponding papers used for evaluating the generator wit h
an LLM. We observed a growing trend in using LLMs for
automating evaluation. Initial studies have shown a positi ve
correlation between human evaluation and LLM-based eval-
uation [58], [59], highlighting the potential of LLMs for
assessing RAG systems. Based on our analysis, we identiﬁed
ﬁve distinct methods for calculating these metrics.
The ﬁrst method integrates the exact match (EM) metric,
referred to as LLM + EM . This approach leverages an
LLM when the exact match fails, such as in cases where
the generated answer is too long. In such instances, the
LLM determines whether the answer is correct [37], [54].
Additionally, the LLM can provide detailed explanations ab out
the errors, offering a more nuanced understanding of the RAG
system’s capabilities. For example, it can highlight the RA G’s
ability to reﬂect information absent in the document or to
identify factual inaccuracies [35].

TABLE V
LIST OF METRICS FOR LLM AS A JUDGE
Method Metrics Papers
EM+LLM Correctness [37], [54]
Error Explanation [35]
Direct Faithfulness/Factual
Consistency[6], [7], [31], [11], [66], [67],
[46], [20], [68], [36]
Truthfulness/Correctness [50], [31], [54], [28], [44], [66],
[32], [69], [26], [24], [43], [42]
Hallucination [37], [54], [33], [20]
Relevance [6], [7], [31], [44], [67], [33],
[69], [24], [42], [46]
Redundancy [11]
Noise Sensitivity [6], [20], [36]
Completeness [44], [33], [69], [26]
Precision [50]
Helpfulness [66], [26]
Missing [37], [54]
Deﬁciency [11]
Coherence [68]
Score [37], [67], [54], [58], [59]
Indirect Precision, recall [4]
Relevance [6]
KPR [30]
Fact/Logic Consistency [22]
OPI [70]
Comparative Kendall /acute.ts1s tau [7]
RAGElo [71], [44], [60]
Own LLM Lynx [21]
The second approach involves direct measurement, where
an LLM is prompted to calculate a speciﬁc metric directly.
This method takes the generated answer, the question, the
retrieved context, and the ground truth answer as input to
compute the metric [7]. Here metrics like Faithfulness or
Truthfulness were described in a speciﬁc prompt and the
LLM outputs a score for these. Faithfulness describes for
example the factual consistent by the retrieved context [6] . The
direct measurement has been the most widely used method for
evaluating the generator component. In contrast, the indirect
measurement method focuses on a more complex evaluation
process. Unlike direct measurement, this approach does not
rely on the direct use of the context, question, and ground
truth in the prompt. Instead, it preprocesses the generated
answer in a speciﬁc manner before calculating the metric. Fo r
instance, RAGQuestEval evaluates the ground truth referen ce
by generating questions derived from it and then determinin g
whether the generated answers can correctly address these
questions [4]. Other methods in this category include count ing
key points in the generated answer [30] or generating questi ons
based on the answer itself [6].
Forcomparative evaluation, metrics such as Kendall’s tau
are commonly employed to compare different RAG conﬁg-
urations [7]. A more comprehensive framework, RAGElo,
was also introduced, combining individual judgments with a
comparative scoring mechanism [71], [44], [60]. This frame -
work allows an LLM to evaluate two answers to the same
question along with their retrieved contexts, determining which
answer is better. Based on these comparisons, an Elo score
is calculated, serving as a comparative metric for evaluati ng
the performance of RAG systems [44], [60]. Finally, the ﬁfth
method leverages a specialized own LLM for evaluation. For
example, Lynx, an open-source LLM trained speciﬁcally todetect hallucinations, has been used as a standalone evalua tor
to assess the accuracy and reliability of RAG systems [21].
These diverse methods illustrate the versatility of LLMs as
evaluators.
VII. C ONCLUSION
A key challenge in the evaluation workﬂow lies in selecting
an appropriate dataset. This process must address several
issues. First, the dataset must be designed to prevent the
LLM from leveraging its base knowledge to answer questions
independently [24]. The questions should also be neither to o
simple nor too general; instead, they should be challenging
and aligned with the RAG system’s intended use case. Ad-
ditionally, the dataset should contain labels for the conte xt
relevance if the retriever evaluation requires these. Anot her
challenge is that every part of the evalution can be automate d
with an LLM. Here the challange lies in ensuring conﬁdence
in the automation process, especially in tasks involving LL Ms.
Unresolved questions persist, such as whether the quality o f
evaluation is compromised when an LLM generates questions,
answers them, and ultimately evaluates its own output. This
raises concerns about potential biases when an LLM evaluate s
itself, and whether the evaluation remains reliable withou t
human involvement. Determining where human expertise is
still necessary in this workﬂow remains a critical area for
further exploration. The rapid development of LLMs present s
another signiﬁcant challenge, as advancements in models
could render previous evaluation results invalid. A new mod el
might produce entirely different outcomes, raising the iss ue of
how to adapt prior results and establish a standard evaluati on
framework that remains consistent and independent of speci ﬁc
LLM versions. Additionally, the lack of a standardized prom pt
for LLM-based evaluations complicates the ability to make
results comparable across different studies.
In light of these evolving challenges, this study identiﬁes
and synthesizes best practices across key components of
RAG system evaluation. The indexing component is evaluated
primarily based on its performance metrics and as part of the
overall system performance [9], [5]. The retriever’s quali ty is
assessed by the relevance of retrieved documents, often usi ng
labeled datasets [8], [39]. In the absence of such datasets i n
RAG systems, domain experts manually evaluated relevance
[42], or LLMs were employed for efﬁcient, automated assess-
ment [8]. The generator component was evaluated through fou r
methods: exact match detection for short answers [48], trad i-
tional metrics like BLEU or ROUGE for complex questions
[42], human validation by experts for accuracy and relevanc e
[55], and LLM-based evaluation using metrics like faithful ness
[6]. The trend and best practice point toward using an LLM
to automate this evaluation task. This study also highlight ed
the crucial role of datasets in evaluating RAG systems. Thre e
types of datasets were identiﬁed: existing datasets, enhan ced
datasets, and newly created datasets [16], [22], [6]. The cr e-
ation of completely new datasets was particularly signiﬁca nt
for domain-speciﬁc RAG systems where no publicly available
datasets existed [11]. In general, the study demonstrated t hat

almost every aspect of the evaluation process could poten-
tially be automated using LLMs. Datasets can be generated
automatically, while both answers and retrieved chunks can
be evaluated by LLMs. However, human expertise remains
essential for speciﬁc tasks, particularly in domain-speci ﬁc
RAG systems where nuanced judgment is required.
Only six studies compared LLM judges with human judges,
ﬁnding a positive correlation between their evaluations [5 8],
[59], [26], [42], [20], [44]. While promising, these ﬁnding s
underscore the ongoing need for human oversight. Future
research should further investigate the reliability of LLM -
based evaluation, as current evidence suggests that LLMs ca n
be trusted to some extent, but their validity remains to be
thoroughly established.
In conclusion, while signiﬁcant progress has been made
in RAG systems and their evaluation, the practical dos and
don’ts remain largely unexplored, especially in real-worl d
application domains. As businesses continue to invest in
RAG implementations, the need for practical frameworks and
actionable recommendations has become increasingly urgen t.
A notable gap in current evaluation approaches is the lack of
consideration for RAG-speciﬁc requirements, such as ensur ing
systems stay updated with new knowledge or how retrievers
effectively incorporate and maintain access to the most cur rent
information. Further research should address these gaps by
offering clear guidance and developing practical tools to
enhance the effectiveness and adaptability of RAG systems
in dynamic, real-world environments.
REFERENCES
[1] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun , M. Wang,
and H. Wang, “Retrieval-augmented generation for large lan guage
models: A survey.” [Online]. Available: http://arxiv.org /abs/2312.10997
[2] Y . Zhang, Y . Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zh ao,
Y . Zhang, Y . Chen, L. Wang, A. Luu, W. Bi, F. Shi, and S. Shi, “Si ren’s
Song in the AI Ocean: A Survey on Hallucination in Large Langu age
Models,” ArXiv , Sep. 2023.
[3] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N . Goyal,
H. K¨ uttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel, S. Riedel , and
D. Kiela, “Retrieval-Augmented Generation for Knowledge- Intensive
NLP Tasks,” Apr. 2021, arXiv:2005.11401. [Online]. Availa ble:
http://arxiv.org/abs/2005.11401
[4] Y . Lyu, Z. Li, S. Niu, F. Xiong, B. Tang, W. Wang, H. Wu,
H. Liu, T. Xu, and E. Chen, “CRUD-RAG: A comprehensive chines e
benchmark for retrieval-augmented generation of large lan guage
models.” [Online]. Available: http://arxiv.org/abs/240 1.17043
[5] S. Kukreja, T. Kumar, V . Bharate, A. Purohit, A. Dasgupta , and
D. Guha, “Performance evaluation of vector embeddings with retrieval-
augmented generation,” in 2024 9th International Conference on
Computer and Communication Systems (ICCCS) , 2024, pp. 333–340.
[Online]. Available: https://ieeexplore.ieee.org/docu ment/10603291
[6] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert, “RA GAS:
Automated evaluation of retrieval augmented generation.” [Online].
Available: http://arxiv.org/abs/2309.15217
[7] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia, “AR ES: An
automated evaluation framework for retrieval-augmented g eneration
systems.” [Online]. Available: http://arxiv.org/abs/23 11.09476
[8] A. Salemi and H. Zamani, “Evaluating retrieval qual-
ity in retrieval-augmented generation.” [Online]. Availa ble:
http://arxiv.org/abs/2404.13781
[9] L. Caspari, K. G. Dastidar, S. Zerhoudi, J. Mitrovic, and
M. Granitzer, “Beyond benchmarks: Evaluating embedding mo del
similarity for retrieval augmented generation systems.” [ Online].
Available: http://arxiv.org/abs/2407.08275[10] Y . Tang and Y . Yang, “MultiHop-RAG: Benchmarking retri eval-
augmented generation for multi-hop queries.” [Online]. Av ailable:
http://arxiv.org/abs/2401.15391
[11] S. Wang, J. Liu, S. Song, J. Cheng, Y . Fu, P. Guo, K. Fang, Y . Zhu,
and Z. Dou, “DomainRAG: A chinese benchmark for evaluating
domain-speciﬁc retrieval-augmented generation.” [Onlin e]. Available:
http://arxiv.org/abs/2406.05654
[12] Y . Pu, Z. He, T. Qiu, H. Wu, and B. Yu, “Customized retriev al
augmented generation and benchmarking for EDA tool documen tation
QA.” [Online]. Available: http://arxiv.org/abs/2407.15 353
[13] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu, “Evalu ation of
retrieval-augmented generation: A survey,” 2024. [Online ]. Available:
http://arxiv.org/abs/2405.07437
[14] S. Knollmeyer, O. Caymazer, L. Koval, M. Akmal, S. Asif, S. Mathias,
and D. Großmann, “Benchmarking of retrieval augmented gene ration:
A comprehensive systematic literature review on evaluatio n dimensions,
evaluation metrics and datasets,” in Proceedings of the 16th International
Joint Conference on Knowledge Discovery, Knowledge Engine ering and
Knowledge Management - KMIS , INSTICC. SciTePress, 2024, pp. 137–
148.
[15] B. Kitchenham and S. Charters, “Guidelines for perform ing systematic
literature reviews in software engineering,” vol. 2, 2007.
[16] N. Pipitone and G. H. Alami, “LegalBench-RAG: A benchma rk
for retrieval-augmented generation in the legal domain.” [ Online].
Available: http://arxiv.org/abs/2408.10343
[17] G. Xiong, Q. Jin, Z. Lu, and A. Zhang, “Benchmarking
retrieval-augmented generation for medicine.” [Online]. Available:
http://arxiv.org/abs/2402.13178
[18] R. Meyur, H. Phan, S. Wagle, J. Strube, M. Halappanavar,
S. Horawalavithana, A. Acharya, and S. Munikoti, “WeQA: A
benchmark for retrieval augmented generation in wind energ y domain.”
[Online]. Available: http://arxiv.org/abs/2408.11800
[19] D. Oberst. How to evaluate LLMs for RAG? [Online]. Avail able:
https://medium.com/@darrenoberst/how-accurate-is-ra g-8f0706281fd9
[20] D. Ru, L. Qiu, X. Hu, T. Zhang, P. Shi, S. Chang, C. Jiayang , C. Wang,
S. Sun, H. Li, Z. Zhang, B. Wang, J. Jiang, T. He, Z. Wang, P. Liu ,
Y . Zhang, and Z. Zhang, “RAGChecker: A ﬁne-grained framewor k
for diagnosing retrieval-augmented generation.” [Online ]. Available:
http://arxiv.org/abs/2408.08067
[21] S. S. Ravi, B. Mielczarek, A. Kannappan, D. Kiela, and R. Qian,
“Lynx: An open source hallucination evaluation model.” [On line].
Available: http://arxiv.org/abs/2407.08488
[22] Y . Xu, T. Cai, J. Jiang, and X. Song, “Face4rag: Factual c onsistency
evaluation for retrieval augmented generation in chinese. ” [Online].
Available: http://arxiv.org/abs/2407.01080
[23] Z. Peng, J. Nian, A. Evﬁmievski, and Y . Fang, “RAG-Confu sionQA:
A benchmark for evaluating LLMs on confusing questions.” [O nline].
Available: http://arxiv.org/abs/2410.14567
[24] T. Kenneweg, P. Kenneweg, and B. Hammer, “Retrieval aug mented
generation systems: Automatic dataset creation, evaluati on and boolean
agent setup.” [Online]. Available: http://arxiv.org/abs /2403.00820
[25] T. Schimanski, J. Ni, R. Spacey, N. Ranger, and M. Leippo ld, “ClimRe-
trieve: A benchmarking dataset for information retrieval f rom corporate
climate disclosures.” [Online]. Available: http://arxiv .org/abs/2406.09818
[26] R. Han, Y . Zhang, P. Qi, Y . Xu, J. Wang, L. Liu, W. Y . Wang, B . Min,
and V . Castelli, “RAG-QA arena: Evaluating domain robustne ss for
long-form retrieval augmented question answering.” [Onli ne]. Available:
http://arxiv.org/abs/2407.13998
[27] Y . Li, Y . Li, X. Wang, Y . Jiang, Z. Zhang, X. Zheng, H. Wang ,
H.-T. Zheng, P. S. Yu, F. Huang, and J. Zhou, “Benchmarking
multimodal retrieval augmented generation with dynamic VQ A
dataset and self-adaptive planning agent.” [Online]. Avai lable:
http://arxiv.org/abs/2411.02937
[28] K. Xie, P. Laban, P. K. Choubey, C. Xiong, and C.-S.
Wu, “Do RAG systems cover what matters? evaluating and
optimizing responses with sub-question coverage.” [Onlin e]. Available:
http://arxiv.org/abs/2410.15531
[29] S. Wang, E. Khramtsova, S. Zhuang, and G. Zuccon, “FeB4r ag:
Evaluating federated search in the context of retrieval aug mented
generation.” [Online]. Available: http://arxiv.org/abs /2402.11891
[30] Z. Qi, R. Xu, Z. Guo, C. Wang, H. Zhang, and W. Xu,
“Long$ˆ2$RAG: Evaluating long-context & long-form retrie val-
augmented generation with key point recall.” [Online]. Ava ilable:
http://arxiv.org/abs/2410.23000

[31] J. Liu, R. Ding, L. Zhang, P. Xie, and F. Huang, “CoFE-RAG :
A comprehensive full-chain evaluation framework for retri eval-
augmented generation with enhanced data diversity.” [Onli ne]. Available:
http://arxiv.org/abs/2410.12248
[32] S. Krishna, K. Krishna, A. Mohananey, S. Schwarcz, A. St ambler,
S. Upadhyay, and M. Faruqui, “Fact, fetch, and reason: A uniﬁ ed
evaluation of retrieval-augmented generation.” [Online] . Available:
http://arxiv.org/abs/2409.12941
[33] K. Zhu, Y . Luo, D. Xu, R. Wang, S. Yu, S. Wang, Y . Yan,
Z. Liu, X. Han, Z. Liu, and M. Sun, “RAGEval: Scenario speciﬁc
RAG evaluation dataset generation framework.” [Online]. A vailable:
http://arxiv.org/abs/2408.01262
[34] T. Sun, A. Somalwar, and H. Chan, “Multimodal re-
trieval augmented generation evaluation benchmark,” in 2024
IEEE 99th Vehicular Technology Conference (VTC2024-Sprin g),
2024, pp. 1–5, ISSN: 2577-2465. [Online]. Available:
https://ieeexplore.ieee.org/document/10683437/?arnu mber=10683437
[35] J. Chen, H. Lin, X. Han, and L. Sun, “Benchmarking large l anguage
models in retrieval-augmented generation.” [Online]. Ava ilable:
http://arxiv.org/abs/2309.01431
[36] Y . Ming, S. Purushwalkam, S. Pandit, Z. Ke, X.-P. Nguyen , C. Xiong,
and S. Joty, “FaithEval: Can your language model stay faithf ul to
context, even if ”the moon is made of marshmallows”.” [Onlin e].
Available: http://arxiv.org/abs/2410.03727
[37] S. Yu, M. Cheng, J. Yang, and J. Ouyang, “A knowledge-cen tric
benchmarking framework and empirical study for retrieval- augmented
generation.” [Online]. Available: http://arxiv.org/abs /2409.13694
[38] X. Wu, S. Li, H.-T. Wu, Z. Tao, and Y . Fang, “Does RAG intro duce
unfairness in LLMs? evaluating fairness in retrieval-augm ented
generation systems.” [Online]. Available: http://arxiv. org/abs/2409.19804
[39] G. d. S. P. Moreira, R. Ak, B. Schifferer, M. Xu, R. Osmuls ki,
and E. Oldridge, “Enhancing q&a text retrieval with ranking models:
Benchmarking, ﬁne-tuning and deploying rerankers for RAG. ” [Online].
Available: http://arxiv.org/abs/2409.07691
[40] Y . Cheng, K. Mao, Z. Zhao, G. Dong, H. Qian, Y . Wu,
T. Sakai, J.-R. Wen, and Z. Dou, “CORAL: Benchmarking multi- turn
conversational retrieval-augmentation generation.” [On line]. Available:
http://arxiv.org/abs/2410.23090
[41] H. Lin, S. Zhan, J. Su, H. Zheng, and H. Wang, “IRSC: A zero -
shot evaluation benchmark for information retrieval throu gh semantic
comprehension in retrieval-augmented generation scenari os.” [Online].
Available: http://arxiv.org/abs/2409.15763
[42] A. Afzal, A. Kowsik, R. Fani, and F. Matthes, “Towards op timizing and
evaluating a retrieval augmented QA chatbot using LLMs with human
in the loop.” [Online]. Available: http://arxiv.org/abs/ 2407.05925
[43] Y . Hui, Y . Lu, and H. Zhang, “UDA: A benchmark suite for re trieval
augmented generation in real-world document analysis.” [O nline].
Available: http://arxiv.org/abs/2406.15187
[44] Z. Rackauckas, A. Cˆ amara, and J. Zavrel, “Evaluating R AG-fusion
with RAGElo: an automated elo-based framework.” [Online]. Available:
http://arxiv.org/abs/2406.14783
[45] R. Friel, M. Belyi, and A. Sanyal, “RAGBench: Explainab le benchmark
for retrieval-augmented generation systems.” [Online]. A vailable:
http://arxiv.org/abs/2407.11005
[46] T. Ding, A. Banerjee, L. Mombaerts, Y . Li, T. Borogovac,
and J. P. D. l. C. Weinstein, “VERA: Validation and
evaluation of retrieval-augmented systems.” [Online]. Av ailable:
http://arxiv.org/abs/2409.03759
[47] A. Alinejad, K. Kumar, and A. Vahdat, “Evaluating the re trieval
component in LLM-based question answering systems.” [Onli ne].
Available: http://arxiv.org/abs/2406.06458
[48] V . Katranidis and G. Barany, “FaaF: Facts as a function f or the evaluation
of generated text.” [Online]. Available: http://arxiv.or g/abs/2403.03888
[49] S. Simon, A. Mailach, J. Dorn, and N. Siegmund, “A method ology for
evaluating RAG systems: A case study on conﬁguration depend ency
validation.” [Online]. Available: http://arxiv.org/abs /2410.08801
[50] D. Rau, H. D´ ejean, N. Chirkova, T. Formal, S. Wang,
V . Nikoulina, and S. Clinchant, “BERGEN: A benchmarking
library for retrieval-augmented generation.” [Online]. A vailable:
http://arxiv.org/abs/2407.01102
[51] S. Khaled, E. H. Mohamed, and W. Medhat, “Evaluating lar ge
language models for arabic sentiment analysis: A comparati ve
study using retrieval-augmented generation,” Procedia ComputerScience , vol. 244, pp. 363–370, 2024. [Online]. Available:
https://www.sciencedirect.com/science/article/pii/S 1877050924030114
[52] X. Yu, H. Cheng, X. Liu, D. Roth, and J. Gao, “ReEval:
Automatic hallucination evaluation for retrieval-augmen ted large
language models via transferable adversarial attacks,” in Findings
of the Association for Computational Linguistics: NAACL 20 24,
K. Duh, H. Gomez, and S. Bethard, Eds. Association for
Computational Linguistics, 2024, pp. 1333–1351. [Online] . Available:
https://aclanthology.org/2024.ﬁndings-naacl.85
[53] G. Guinet, B. Omidvar-Tehrani, A. Deoras, and L. Callot , “Automated
evaluation of retrieval-augmented language models with ta sk-speciﬁc
exam generation.” [Online]. Available: http://arxiv.org /abs/2405.13622
[54] X. Yang, K. Sun, H. Xin, Y . Sun, N. Bhalla, X. Chen, S. Chou dhary,
R. D. Gui, Z. W. Jiang, Z. Jiang, L. Kong, B. Moran, J. Wang,
Y . E. Xu, A. Yan, C. Yang, E. Yuan, H. Zha, N. Tang, L. Chen,
N. Scheffer, Y . Liu, N. Shah, R. Wanga, A. Kumar, W.-t. Yih, an d
X. L. Dong, “CRAG – comprehensive RAG benchmark.” [Online].
Available: http://arxiv.org/abs/2406.04744
[55] C. Lang, R. Schneider, and N. D. T. Tu, “Automatic questi on answering
for the linguistic domain – an evaluation of LLM knowledge ba se
extension with RAG,” in Natural Language Processing and Information
Systems , A. Rapp, L. Di Caro, F. Meziane, and V . Sugumaran, Eds.
Springer Nature Switzerland, 2024, pp. 161–171.
[56] S. Alghisi, M. Rizzoli, G. Roccabruna, S. M. Mousavi, an d G. Riccardi,
“Should we ﬁne-tune or RAG? evaluating different technique s to adapt
LLMs for dialogue.” [Online]. Available: http://arxiv.or g/abs/2406.06399
[57] A. Onan and E. D. Dursun, “Benchmarking retrieval augme nted genera-
tion in quantitative ﬁnance,” in Intelligent and Fuzzy Systems , C. Kahra-
man, S. Cevik Onar, S. Cebi, B. Oztaysi, A. C. Tolga, and I. Uca l Sari,
Eds. Springer Nature Switzerland, 2024, pp. 64–74.
[58] Q. Leng. Best practices for LLM evalua-
tion of RAG applications. [Online]. Available:
https://www.databricks.com/blog/LLM-auto-eval-best- practices-RAG
[59] Y . Wang, A. G. Hernandez, R. Kyslyi, and N. Kersting, “Ev aluating
quality of answers for retrieval-augmented generation: A s trong LLM
is all you need.” [Online]. Available: http://arxiv.org/a bs/2406.18064
[60] N. Thakur, S. Kazi, G. Luo, J. Lin, and A. Ahmad, “MIRAGE-
bench: Automatic multilingual benchmark arena for retriev al-augmented
generation systems.” [Online]. Available: http://arxiv. org/abs/2410.13716
[61] G. B and A. Purwar, “Evaluating the efﬁcacy of open-sour ce LLMs in
enterprise-speciﬁc RAG systems: A comparative study of per formance
and scalability.” [Online]. Available: http://arxiv.org /abs/2406.11424
[62] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training
of deep bidirectional transformers for language understan ding,” version:
2. [Online]. Available: http://arxiv.org/abs/1810.0480 5
[63] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence
embeddings using siamese BERT-networks.” [Online]. Avail able:
http://arxiv.org/abs/1908.10084
[64] T. Zhang, V . Kishore, F. Wu, K. Q. Weinberger, and Y . Artz i,
“BERTScore: Evaluating text generation with BERT.” [Onlin e].
Available: http://arxiv.org/abs/1904.09675
[65] M. Zhong, Y . Liu, D. Yin, Y . Mao, Y . Jiao, P. Liu, C. Zhu, H. Ji,
and J. Han, “Towards a uniﬁed multi-dimensional evaluator f or text
generation.” [Online]. Available: http://arxiv.org/abs /2210.07197
[66] Evaluation concepts | LangSmith. [Online]. Available:
https://docs.smith.langchain.com/evaluation/concept s
[67] D. Huang and Z. Wang, “Evaluation of orca 2 against other LLMs
for retrieval augmented generation,” in Trends and Applications in
Knowledge Discovery and Data Mining , Z. Wang and C. W. Tan, Eds.
Springer Nature, 2024, pp. 3–19.
[68] T.-L. Kuo, F.-T. Liao, M.-W. Hsieh, F.-C. Chang, P.-C. H su,
and D.-S. Shiu, “RAD-bench: Evaluating large language mode ls
capabilities in retrieval augmented dialogues.” [Online] . Available:
http://arxiv.org/abs/2409.12558
[69] S. Sivasothy, S. Barnett, S. Kurniawan, Z. Rasool, and R . Vasa,
“RAGProbe: An automated approach for evaluating RAG applic ations.”
[Online]. Available: http://arxiv.org/abs/2409.19019
[70] J. Hu, Y . Zhou, and J. Wang, “Intrinsic evaluation of
RAG systems for deep-logic questions.” [Online]. Availabl e:
http://arxiv.org/abs/2410.02932
[71] Zeta Alpha Vector, “Ragelo,” 2023. [Online]. Availabl e:
https://github.com/zetaalphavector/RAGElo