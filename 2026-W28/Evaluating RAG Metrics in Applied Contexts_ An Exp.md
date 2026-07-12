# Evaluating RAG Metrics in Applied Contexts: An Experiment, Its Findings and Its Limitations

**Authors**: Quentin Brabant

**Published**: 2026-07-08 11:44:28

**PDF URL**: [https://arxiv.org/pdf/2607.07302v1](https://arxiv.org/pdf/2607.07302v1)

## Abstract
This paper reports an empirical study evaluating the relevance of several RAG metrics. The experiment is based on a question-answering dataset created by human annotators from business data. The generated responses and retrieved spans of a RAG system are scored using evaluation metrics from four libraries (Ragas, DeepEval, RAGChecker, Opik). These metrics are compared to scores given by two evaluators, as well as to standard metrics such as recall. An analysis of correlations is conducted. Finally, we highlight certain limitations of our methodology, compare it to those used in the literature, and suggest some avenues for future research. This paper is an English translation of a paper originally published in the French-speaking workshop EvalLLM (Brabant, 2026).

## Full Text


<!-- PDF content starts -->

Evaluating RAG Metrics in Applied Contexts: An
Experiment, Its Findings and Its Limitations
Quentin Brabant
Orange Research, Lannion, France
quentin.brabant@orange.com
Abstract
This paper reports an empirical study evaluating the relevance of several
RAG metrics. The experiment is based on a question-answering dataset cre-
ated by human annotators from business data. The generated responses and
retrieved spans of a RAG system are scored using evaluation metrics from four
libraries (Ragas, DeepEval, RAGChecker, Opik). These metrics are compared
to scores given by two evaluators, as well as to standard metrics such as re-
call. An analysis of correlations is conducted. Finally, we highlight certain
limitations of our methodology, compare it to those used in the literature, and
suggest some avenues for future research. This paper is an English translation
of a paper originally published in the French-speaking workshop EvalLLM 2026
(Brabant, 2026).
1 Introduction
Evaluating and comparing RAG (Retrieval Augmented Generation) systems remains
a challenging task today: even when a sufficiently large set of test questions with
reference answers is available, automatically evaluating a system’s responses against
these references is far from trivial. A popular approach is to use so-called LLM-
as-a-judge metrics to perform this evaluation. Although these metrics generally
seem more relevant than classical metrics such as BLEU, it is difficult to know
in advance what the relevance of a specific metric will be on the dataset under
consideration, especially since evaluation criteria can vary (relevance, factuality,
completeness of responses, etc.). It is therefore useful, when evaluating a RAG
system under development, to conduct an evaluation of available metrics in order
to verify that they provide acceptable approximations of the criterion considered,
and that they will thus enable reliable comparison of different iterations of the
RAG system being developed. Generally, metrics are evaluated by measuring their
correlation with scores given by humans.
This article reports an experiment of this type. This experiment is based on a
question-answering dataset created by annotators from business data. The responses
1
arXiv:2607.07302v1  [cs.CL]  8 Jul 2026

produced and the documents retrieved by a RAG system are scored using RAG met-
rics from four libraries: Ragas1(Es et al., 2024), DeepEval2, RAGChecker3(Ru et al.,
2024), and Opik4. These metrics are compared to reference evaluations: human
evaluations for assessing generated responses, and recall for evaluating retrieval.
Note that our objective is not to compare the different metrics evaluated, as
the study results are strongly dependent on the choices made during its design.
The reported experiments rather aim to apply a methodology in order to test its
advantages and limitations. Unfortunately, the question-answering dataset used
cannot be made public; however, we share the raw scores and the code used for the
statistical analyses5.
The article is organized as follows. Section 2 describes the application context
and the data used, including manual annotation and evaluation processes. Section 3
describes the methodology applied to compare reference evaluations with evaluations
produced by metrics from the tested libraries. Section 4 reports and analyzes the
results of this experiment. Certain limitations of our methodology are highlighted
in Section 5. In Section 6, our methodology is compared to those employed in the
literature. We finally propose some research perspectives aimed at facilitating the
application of reliable methodologies for evaluating RAG metrics, in Section 7.
2 Context and Data
Our company is developing a RAG solution designed to answer questions in French
about a business domain. The developed system processes each given question via
two key modules: first, aretriever, whose role is to retrieve relevant spans within
a business document database (also in French), which combines a dense approach
with BM25; then agenerator, which injects the user’s question and the top 5 spans
from the retriever into a prompt for GPT 3.5, so that it generates the answer to the
user’s question. The retriever is a hybrid system combining a dense approach with
BM25.
In order to evaluate the performance of this system, a dataset was created from
the corpus of business documents used by the RAG system. This dataset is a set of
question-answer pairs accompanied by reference spans.
2.1 The Question-Answering Dataset
This dataset consists of 96 questions. Each question is associated with a reference
answer, as well as one or more spans from business documents; these spans contain
the information necessary to answer the question with which they are associated.
1https://docs.ragas.io/
2https://docs.confident-ai.com/
3https://github.com/amazon-science/RAGChecker
4https://www.comet.com/site/products/opik/
5https://github.com/Orange-OpenSource/evalllm2026-metric-correlation-analysis
2

The question-answering dataset is created based on the existing business docu-
ment database on the telecommunications domain, containing information on, for
instance, different offers and customer relations. It contains 479 documents, with
length ranging from 11 to 14,025 words (1,112 on average). The questions and
answers are written by annotators who are company employees, as follows:
1. Documents are divided into “pages” of maximum 1,000 words, relying on the
structure of titles and sections to preserve content coherence. This size limit
is chosen arbitrarily, with the aim of limiting the amount of information to
process for each annotation.
2. The obtained pages are distributed in random order to annotators.
3. Each annotator annotates one by one the pages assigned to them. The anno-
tation of a page proceeds as follows.
(a) The annotator writes a question related to the page content, as well as
a correct answer to this question (if the answer can be given from the
information contained in the page).
(b) They select one or more spans in the page that allow answering the
question.
These steps can be repeated to produce up to 5 questions per page.
Annotators were asked to diversify, as much as possible, the type of questions pro-
duced, and to indicate the type of each question within a dropdown list. The possible
question types and their numbers are summarized in Table 1. Additionally, 15 ques-
tions are not associated with any answer, because the necessary information is not
available in the business documents. In this case, the system is expected to produce
an answer such as ”I don’t know”.
The number of spans per question varies from 0 (for questions without answers)
to 4, with an average of 1.3. We observe that the size of spans associated with
questions varies greatly, ranging from 1 word (the annotator having selected a single
keyword corresponding to the answer) to 202 words. The implications in terms of
evaluation metric choice are discussed in Section 2.3.
2.2 RAG System Evaluation Procedure
The question-answering dataset is intended to enable evaluation of the RAG system:
first, questions are given as input to the system, and the system outputs (generated
responses and retrieved spans) are collected; then the system outputs are evaluated
using different metrics.
Comparing generated responses with reference answers produces an overall score,
while comparing retrieved spans with reference spans produces a retriever perfor-
mance score, which can be useful for estimating what proportion of system failures
3

Question Type Number Example
boolean 19 Does a customer of X who cancels due to
the November 2022 price increase have to
pay fees?
who/what/where/when 22 Who should I contact if I have an issue with
solution X?
how 17 How can I consult the documents and op-
erating procedures for offer X?
why 14 Why was the X provided by Y chosen?
conditional 17 What happens if a customer wishes to op-
pose a judicial transfer?
how many 17 How many X are included in Y?
Table 1: Question types present in the question-answering dataset. Some questions
belong to multiple types.
is due to the retriever and which is due to the generator, and thus targeting im-
provement efforts appropriately.
2.3 Available Metrics
This subsection briefly describes the metrics used to evaluate the RAG system.
We follow the terminology of (Ru et al., 2024), where three types of metrics are
distinguished:
•retrievalmetrics, which evaluate the relevance of retrieved spans relative to a
given question;
•overallmetrics, which evaluate the response generated by the system for the
question;
•generationmetrics, which evaluate the generator’s behavior relative to the
content of spans from the retrieval.
Retrieval Metrics.In the context of information retrieval, several metrics are
traditionally used to evaluate retrieval quality: recall, precision, nDCG, MAP, etc.
In this study, we choose to report recall scores, which will be used as reference
scores to evaluate retrieval metrics from the tested libraries.Recallis defined as
the proportion of relevant elements actually present in the topkspans returned by
the retriever. A common approach is to calculate recall at the document level, i.e.,
by considering that a reference span is present in the retriever results if at least
4

one span from the same document is present. However, this condition does not
imply that the reference span is actually contained in a returned span: the reference
span can be disjoint from the retrieved span or partially overlapping. This problem
arises particularly here, since our references are made of short spans compared to
the documents from which they originate. We therefore calculate recall at the word
level, i.e.: the percentage of words from reference spans present in the top k retrieved
spans. Note that words are identified by their positions in the text, not by their
value: returned spans completely disjoint from reference spans will therefore give a
score of 0, even if they contain identical words. The choice of recall over other metrics
(such as precision) is justified by the fact that (1) its scores are easy to interpret, (2)
since reference spans tend to be relatively small, we seek to know if they are included
in the fixed-size spans returned by the retriever, which is a reasonable expectation
and corresponds to the definition of recall; conversely, precision at the word level
would give necessarily low scores. Finally, the choice of this metric is supported by
the fact that it correlates much better with evaluators’ average scores (r= 0.35)
than other cited metrics, and notably than document-level recall (r= 0.05).
The retrieval metrics proposed by the tested libraries differ from traditional met-
rics on two main respects: first, some of them do not require reference spans, so they
can be applied even when these are not available; second, they often use language
models, which allows them to calculate the final score based on semantically impor-
tant elements of the considered spans. Like recall, these metrics evaluate retrieval
quality from the topkretrieved spans. We choose to fixk= 5 for all metrics, so
that the spans on which retrieval is evaluated correspond to those actually inserted
into the generator’s prompt.
Overall Metrics.Overall metrics evaluate the quality of the generated response
according to specific criteria. Many overall metrics, corresponding to different cri-
teria, are proposed by the tested libraries. Although these criteria are diverse, it
can be noted that most are concerned with the factuality of the generated response,
generally decomposed into two aspects:precision(“is all the information given in
the response correct?”), andrecall(“is all the expected information given in the
response?”). In addition to factuality,relevanceis sometimes considered: “is all the
information given in the response related to the question?” In addition to metrics
evaluating factuality and relevance, we integrate Opik’smoderationandusefulness
metrics: the former checks for the absence of harmful or inappropriate content, while
the latter combines various criteria to obtain a general score.
Generation Metrics.Generation metrics evaluate the generator’s behavior rela-
tive to the content of spans from the retrieval. The primary purpose of these metrics
is to study certain generator behaviors and not its performance in the strict sense;
however, one may want to use them to approximate overall criteria. For example,
DeepEval’sfaithfulnessmetric, which aims to measure the response’s faithfulness to
5

retrieved spans, can be used to approximate the precision criterion. We therefore
integrate into our study some generation metrics, which will be evaluated as overall
metrics (these are the four metrics namedhallucinationorfaithfulness, see Figure
1).
3 Metric Evaluation: Methodology
We seek to evaluate to what extent the metrics mentioned in the previous section
provide a good approximation of a given evaluation criterion. We choose to define
an overall criterion simultaneously measuring aspects of factuality and relevance.
This criterion is evaluated on a scale of 1 to 5 and defined by the rubric in Table 2.
Score Description
5 Factual and relevant response with the right amount of detail.
4 Factual response, but lacking useful information or containing too much
unimportant information.
3 Partially correct response, with small errors or approximations OR “I
don’t know” when the reference answer contains the requested informa-
tion.
2 Factually incorrect response.
1 Off-topic response.
Table 2: Rubric used during human evaluation of RAG system responses.
This rubric is then applied by two evaluators (company employees with expertise
in natural language processing and generative models, one of whom participated in
the annotation phase during dataset creation), to score each RAG system output
on the 96 instances of the question-answering dataset. We calculate the Pearson
correlation6between the two series of scores obtained, to verify the robustness of
the rubric and estimate the maximum performance that can be expected from an
automatic metric. The correlation obtained is 0.85. To simplify subsequent analyses,
we base ourselves on the average scores obtained per response. The scores thus
obtained are calledreference scores.
We then proceed to the correlation analysis intended to evaluate the metrics.
The following section reports and analyzes:
•correlations between overall metrics and reference scores;
6We calculate Pearson correlation rather than Cohen’s or Fleiss’s kappa, as these are poorly
suited when annotations are ordinal in nature, as is the case here.
6

•correlations of retrieval metrics with recall: although recall is itself an imper-
fect metric, it is expected that a stronger correlation of a retrieval metric with
recall indicates better reliability;
•correlations between retrieval metrics and reference scores: indeed, since better
retrieval correlates with better responses, retrieval metrics should also corre-
late with scores from our rubric.
All reported correlations correspond to Pearson’s coefficient (performing analyses
with Spearman correlation gives results similar to those reported).
4 Results
-1.00 -0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00DeepEval Ragas OpikRAG
Checkerbleu
rougeLsum
meteor
bertscore-f1
hallucination
answer relevancy
faithfulness
faithfulness
factual correctness:f1
factual correctness:recall
factual correctness:accuracy
hallucination
moderation
answer relevance
usefulness
context precision
context recall
f1
recall
precision
Figure 1: Pearson correlation of overall metrics with average human rates.
Figure 1 summarizes the correlations obtained for overall metrics. We first note
that the width of confidence intervals is substantial, due to the modest size of our
sample (96). However, it is possible to see certain trends emerge.
7

First, we note that metrics sometimes considered obsolete such as METEOR
correlate surprisingly well with reference scores. Next, we observe that generation
metrics used as overall metrics correlate weakly with reference scores, which is ex-
pected, since they do not have access to the reference answer. Similarly, Opik’s
moderationmetric does not correlate with reference scores. Conversely, we observe
a very strong correlation for RAGChecker metrics, particularly recall.
-1.00 -0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00DeepEval RagasRAG
Checkercontextual precision∗
contextual recall∗
contextual relevancy∗
LLM context precision w/o ref∗
LLM context precision w/ ref∗
non-LLM context precision w/ ref
LLM context recall∗
non-LLM context recall
context entity recall∗
claim recall
context precision
Figure 2: Pearson correlation of overall metrics with recall. Metrics that do not rely
of reference spans are marked with an asterisk.
Figure 2 summarizes the correlations of retrieval metrics with recall. We observe
larger overlaps of confidence intervals. However, it is interesting to note that a
metric such as DeepEval’scontextual precisionobtains a correlation above 0.5 with
recall, without using reference spans. We observe that conversely, Ragas’s non-LLM
metrics obtain very weak correlation. This is probably explained by the fact that
it compares retrieved spans to reference spans via Levenshtein distance, which does
not give a relevant value when compared spans are of very different sizes, as is the
case here.
If we observe Figure 3, we note a surprising phenomenon: several metrics show
very strong correlation with reference scores. This correlation is sometimes stronger
than the correlation with recall. RAGChecker’sclaim recallcorrelation appears
close to 0.7. It does not seem plausible that such a correlation is due solely to this
metric’s ability to measure the relevance of retrieved documents. We comment on
this phenomenon in more detail in the following section.
8

-1.00 -0.75 -0.50 -0.25 0.00 0.25 0.50 0.75 1.00DeepEval RagasRAG
Checkercontextual precision∗
contextual recall∗
contextual relevancy∗
LLM context precision w/o ref∗
LLM context precision w/ ref∗
non-LLM context precision w/ ref
LLM context recall∗
non-LLM context recall
context entity recall∗
claim recall
context precision
Figure 3: Pearson correlations of retrieval metrics with average human rates. Metrics
that do not rely on reference spans are marked with an asterisk.
5 Limitations
The main limitation of our study is related to the interpretation of correlations.
It is difficult to know a priori what sophisticated metrics using LLMs rely on to
produce their scores. Consequently, observing that a given metric correlates well
with human judgment is not sufficient to affirm that it measures what we want
it to measure, given the configuration of our experiment. This difficulty can be
illustrated by imagining a metric measuring question difficulty, without knowledge
of generated responses. Such a metric would give higher scores to easy questions,
and these indeed tend to obtain better scores. It would therefore have a positive
correlation with reference scores, while its relevance for evaluating and comparing
RAG systems would be null (it would give exactly the same scores to all systems). A
less extreme illustration of this phenomenon is probably RAGChecker’sclaim recall
metric, whose correlation with evaluators’ scores is very high. It seems likely that
this metric does not measure only retrieval quality; a plausible interpretation is that
it partially captures other characteristics such as, for example, the ease with which
response information can be extracted from retrieved spans, or question difficulty.
This limitation is partly due to the configuration of our experiment: since it involves
only one RAG system, it is impossible to observe metric behavior as a function of
system outputs independently of the input question.
Note that, despite this weakness, our methodology produces certain exploitable
results by eliminating certain candidate metrics: indeed, we can affirm that metrics
with poor correlation with the evaluation criterion considered do not measure this
criterion.
9

6 Related Work
Metric evaluations through correlation studies with human judgment have been re-
ported in several recent publications. Although most of them consider the evaluation
of NLG (Natural Language Generation) tasks other than RAG, the evaluation of
these tasks involves issues common to those of evaluating RAG system responses.
This section offers a (non-exhaustive) overview of these publications, grouped ac-
cording to their evaluation methodologies.
First, some of these publications rely on a methodology similar to ours (corre-
lation analysis on a single system). We can cite, for example: (Liu et al., 2024)
which evaluates a system dedicated to evaluating various aspects of various natural
language generation tasks, or (Yeginbergen et al., 2025) which observes the corre-
lation of several LLM-as-a-judge systems with human evaluations of automatically
generated counter-arguments. These studies suffer from the same methodological
limitation as ours.
Other studies evaluate metrics by measuring their adequacy with a preference
order expressed on pairs of outputs corresponding to the same input. More precisely:
each task input is associated with two alternative outputs, for which a preference
order is available; all pairs for which the metric score conforms to the preference
order are considered successes, and the success ratio forms the metric’s performance
score. This approach helps guard against confounding factors related to question
characteristics that limit the interpretations of our results. Among studies using
this approach, we can cite: (Ke et al., 2024; Zhu et al., 2025; Lambert et al., 2025).
Some studies combine this approach with correlation analysis on a single system,
for example: (Xu et al., 2023; Kim et al., 2024; Xiong et al., 2025).
Finally, some publications report correlation analyses involving multiple systems.
The statistical treatments performed can then vary, since the scores generated by a
metric, like reference scores, are then arranged into a matrix with one row per system
and one column per evaluated output. There are indeed several ways to calculate
a correlation between two matrices: (Gao et al., 2025) studies four of them, two of
which seem relevant to us in the context of evaluating RAG metrics. The first is to
calculate the correlation of each system’s average scores noted by the metric with
each system’s average reference score. The second is to calculate, for each input,
the correlation of metric scores with reference scores across different systems, then
to average the correlations thus obtained. These two approaches are also studied
by (Deutsch et al., 2021). Both studies empirically conclude that the second ap-
proach has greater power to discriminate among tested metrics. Moreover, it has
been demonstrated that this approach considerably reduces the importance of con-
founding factors in correlations measured between machine translation evaluation
metrics and human judgment (Perrella et al., 2024). A similar approach, applied in
(Dinh et al., 2024), consists of calculating correlation on all scores, normalized by
inputs.
10

7 Research Directions
The empirical results of our study are in line with some of the results mentioned in
the previous section, and confirm the usefulness of integrating several (at least two)
RAG systems into correlation studies aimed at evaluating RAG metrics. It should
be noted that the study results will then be partially dependent on the chosen RAG
systems: it is therefore appropriate to choose a set of systems representative of all
systems to which we wish to apply the metrics. Moreover, integrating too many
systems risks making the annotation task costly.
These difficulties open interesting research perspectives. For example, how to
estimate the probable contribution of an additional model or question to a correla-
tion study? Is it possible to develop procedures to choose and adapt the number of
systems and questions to integrate into the empirical study, as human evaluations
are collected? Answers to such questions could enable professionals implementing
RAG systems to validate their evaluation metrics more reliably while minimizing
the costs associated with this validation.
References
Quentin Brabant. ´Evaluation de m´ etriques de rag dans un contexte applicatif : une
exp´ erience, ses conclusions et ses limites. InEvalLLM Workshop, 2026.
Daniel Deutsch, Rotem Dror, and Dan Roth. A Statistical Analysis of Summariza-
tion Evaluation Metrics Using Resampling Methods.Transactions of the Associa-
tion for Computational Linguistics, 9:1132–1146, October 2021. ISSN 2307-387X.
doi: 10.1162/tacl a00417. URLhttps://doi.org/10.1162/tacl_a_00417.
Tu Anh Dinh, Carlos Mullov, Leonard B¨ armann, Zhaolin Li, Danni Liu, Simon
Reiß, Jueun Lee, Nathan Lerzer, Jianfeng Gao, Fabian Peller-Konrad, Tobias
R¨ oddiger, Alexander Waibel, Tamim Asfour, Michael Beigl, Rainer Stiefelhagen,
Carsten Dachsbacher, Klemens B¨ ohm, and Jan Niehues. SciEx: Benchmarking
Large Language Models on Scientific Exams with Human Expert Grading and
Automatic Grading. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen,
editors,Proceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, pages 11592–11610, Miami, Florida, USA, November 2024.
Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.
647. URLhttps://aclanthology.org/2024.emnlp-main.647/.
Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. RAGAs:
Automated Evaluation of Retrieval Augmented Generation. In Nikolaos Ale-
tras and Orphee De Clercq, editors,Proceedings of the 18th Conference of
the European Chapter of the Association for Computational Linguistics: Sys-
tem Demonstrations, pages 150–158, St. Julians, Malta, March 2024. Associa-
11

tion for Computational Linguistics. doi: 10.18653/v1/2024.eacl-demo.16. URL
https://aclanthology.org/2024.eacl-demo.16/.
Mingqi Gao, Xinyu Hu, Li Lin, and Xiaojun Wan. Analyzing and Evaluating
Correlation Measures in NLG Meta-Evaluation. In Luis Chiruzzo, Alan Rit-
ter, and Lu Wang, editors,Proceedings of the 2025 Conference of the Nations
of the Americas Chapter of the Association for Computational Linguistics: Hu-
man Language Technologies (Volume 1: Long Papers), pages 2199–2222, Al-
buquerque, New Mexico, April 2025. Association for Computational Linguis-
tics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.111. URL
https://aclanthology.org/2025.naacl-long.111/.
Pei Ke, Bosi Wen, Andrew Feng, Xiao Liu, Xuanyu Lei, Jiale Cheng, Shengyuan
Wang, Aohan Zeng, Yuxiao Dong, Hongning Wang, Jie Tang, and Minlie Huang.
CritiqueLLM: Towards an Informative Critique Generation Model for Evalua-
tion of Large Language Model Generation. In Lun-Wei Ku, Andre Martins, and
Vivek Srikumar, editors,Proceedings of the 62nd Annual Meeting of the Asso-
ciation for Computational Linguistics (Volume 1: Long Papers), pages 13034–
13054, Bangkok, Thailand, August 2024. Association for Computational Linguis-
tics. doi: 10.18653/v1/2024.acl-long.704. URLhttps://aclanthology.org/
2024.acl-long.704/.
Seungone Kim, Juyoung Suk, Shayne Longpre, Bill Yuchen Lin, Jamin Shin,
Sean Welleck, Graham Neubig, Moontae Lee, Kyungjae Lee, and Minjoon Seo.
Prometheus 2: An Open Source Language Model Specialized in Evaluating Other
Language Models. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, ed-
itors,Proceedings of the 2024 Conference on Empirical Methods in Natural Lan-
guage Processing, pages 4334–4353, Miami, Florida, USA, November 2024. As-
sociation for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.248.
URLhttps://aclanthology.org/2024.emnlp-main.248/.
Nathan Lambert, Valentina Pyatkin, Jacob Morrison, LJ Miranda, Bill Yuchen Lin,
Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, Noah A.
Smith, and Hannaneh Hajishirzi. RewardBench: Evaluating Reward Models for
Language Modeling. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors,Find-
ings of the Association for Computational Linguistics: NAACL 2025, pages 1755–
1797, Albuquerque, New Mexico, April 2025. Association for Computational Lin-
guistics. ISBN 979-8-89176-195-7. doi: 10.18653/v1/2025.findings-naacl.96. URL
https://aclanthology.org/2025.findings-naacl.96/.
Minqian Liu, Ying Shen, Zhiyang Xu, Yixin Cao, Eunah Cho, Vaibhav Kumar,
Reza Ghanadan, and Lifu Huang. X-Eval: Generalizable Multi-aspect Text
Evaluation via Augmented Instruction Tuning with Auxiliary Evaluation As-
pects. In Kevin Duh, Helena Gomez, and Steven Bethard, editors,Proceed-
12

ings of the 2024 Conference of the North American Chapter of the Associa-
tion for Computational Linguistics: Human Language Technologies (Volume 1:
Long Papers), pages 8560–8579, Mexico City, Mexico, June 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.473. URL
https://aclanthology.org/2024.naacl-long.473/.
Stefano Perrella, Lorenzo Proietti, Alessandro Scir` e, Edoardo Barba, and Roberto
Navigli. Guardians of the Machine Translation Meta-Evaluation: Sentinel Metrics
Fall In! In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors,Proceedings
of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 16216–16244, Bangkok, Thailand, August 2024.
Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.856.
URLhttps://aclanthology.org/2024.acl-long.856/.
Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang,
Cheng Jiayang, Cunxiang Wang, Shichao Sun, Huanyu Li, Zizhao Zhang, Bin-
jie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang,
and Zheng Zhang. RAGChecker: A Fine-grained Framework for Diagnosing
Retrieval-Augmented Generation. 2024. doi: 10.48550/arXiv.2408.08067. URL
http://arxiv.org/abs/2408.08067.
Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu,
Heng Huang, and Chunyuan Li. LLLaVA-Critic: Learning to Evaluate Multi-
modal Models.2025 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 13618–13628, June 2025. doi: 10.1109/CVPR52734.
2025.01271. URLhttps://ieeexplore.ieee.org/document/11093772/. Con-
ference Name: 2025 IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) ISBN: 9798331543648 Place: Nashville, TN, USA.
Wenda Xu, Danqing Wang, Liangming Pan, Zhenqiao Song, Markus Freitag,
William Wang, and Lei Li. INSTRUCTSCORE: Towards Explainable Text Gen-
eration Evaluation with Automatic Feedback. In Houda Bouamor, Juan Pino, and
Kalika Bali, editors,Proceedings of the 2023 Conference on Empirical Methods in
Natural Language Processing, pages 5967–5994, Singapore, December 2023. As-
sociation for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.365.
URLhttps://aclanthology.org/2023.emnlp-main.365/.
Anar Yeginbergen, Maite Oronoz, and Rodrigo Agerri. Dynamic Knowledge Inte-
gration for Evidence-Driven Counter-Argument Generation with Large Language
Models. 2025. doi: 10.48550/ARXIV.2503.05328. URLhttps://arxiv.org/
abs/2503.05328. Version Number: 2.
Lianghui Zhu, Xinggang Wang, and Xinlong Wang. JudgeLM: Fine-tuned Large
Language Models are Scalable Judges, March 2025. URLhttp://arxiv.org/
abs/2310.17631. arXiv:2310.17631 [cs].
13