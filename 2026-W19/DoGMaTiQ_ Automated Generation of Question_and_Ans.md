# DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation

**Authors**: Bryan Li, William Walden, Yu Hou, Gabrielle Kaili-May Liu, Dawn Lawrie, Jame Mayfield, Eugene Yang, Chris Callison-Burch, Laura Dietz

**Published**: 2026-05-06 03:34:46

**PDF URL**: [https://arxiv.org/pdf/2605.04458v1](https://arxiv.org/pdf/2605.04458v1)

## Abstract
Evaluation of long-form, citation-backed reports has lately received significant attention due to the wide-scale adoption of retrieval-augmented generation (RAG) systems. Core to many evaluation frameworks is the use of atomic facts, or nuggets, to assess a report's coverage of query-relevant information attested in the underlying collection. While nuggets have traditionally been represented as short statements, recent work has used question-answer (QA) representations, enabling fine-grained evaluations that decouple the information need (i.e. the question) from the potentially diverse content that satisfies it (i.e. its answers).
  A persistent challenge for nugget-based evaluation is the need to manually curate sets of nuggets for each topic in a test collection -- a laborious process that scales poorly to novel information needs. This challenge is acute in cross-lingual settings, where information is found in multilingual source documents. Accordingly, we introduce DoGMaTiQ, a pipeline for generating high-quality QA-based nugget sets in three stages: (1) document-grounded nugget generation, (2) paraphrase clustering, and (3) nugget subselection based on principled quality criteria. We integrate DoGMaTiQ nuggets with AutoArgue -- a recent nugget-based evaluation framework -- to enable fully automatic evaluation of generated reports. We conduct extensive experiments on two cross-lingual TREC shared tasks, NeuCLIR and RAGTIME, showing strong rank correlations with both human-in-the-loop and fully manual judgments. Finally, detailed analysis of our pipeline reveals that a strong LLM nugget generator is key, and that the system rankings induced by DoGMaTiQ are robust to outlier systems. We facilitate future research in report evaluation by publicly releasing our code and artifacts at https://github.com/manestay/dogmatiq.

## Full Text


<!-- PDF content starts -->

DoGMaTiQ: Automated Generation of Question-and-Answer
Nuggets for Report Evaluation
Bryan Li∗
University of Pennsylvania
Philadelphia, PA, USAWilliam Walden†
Johns Hopkins University
Baltimore, MD, USAYu Hou
University of Maryland
College Park, MD, USA
Gabrielle Kaili-May Liu
Yale University
New Haven, CT, USADawn Lawrie
Johns Hopkins University
Baltimore, MD, USAJames Mayfield
Johns Hopkins University
Baltimore, MD, USA
Eugene Yang
Johns Hopkins University
Baltimore, MD, USAChris Callison-Burch
University of Pennsylvania
Philadelphia, PA, USALaura Dietz
University of New Hampshire
Durham, NH, USA
Abstract
Evaluation of long-form, citation-backed reports has lately received
significant attention due to the wide-scale adoption of retrieval-
augmented generation (RAG) systems. Core to many evaluation
frameworks is the use of atomic facts, ornuggets, to assess a report’s
coverage of query-relevant information attested in the underlying
collection. While nuggets have traditionally been represented as
short statements, recent work has used question-answer (QA) rep-
resentations, enabling fine-grained evaluations that decouple the
information need (i.e. the question) from the potentially diverse
content that satisfies it (i.e. its answers).
A persistent challenge for nugget-based evaluation is the need to
manually curatesetsof nuggets for each topic in a test collection—a
laborious process that scales poorly to novel information needs.
This challenge is acute in cross-lingual settings, where information
is found in multilingual source documents. Accordingly, we intro-
duceDoGMaTiQ, a pipeline for generating high-quality QA-based
nugget sets in three stages: (1) document-groundednugget genera-
tion, (2)paraphrase clustering, and (3)nugget subselectionbased on
principled quality criteria. We integrateDoGMaTiQnuggets with
Auto-ARGUE—a recent nugget-based evaluation framework—to
enable fully automatic evaluation of generated reports. We conduct
extensive experiments on two cross-lingual TREC shared tasks,
NeuCLIR and RAGTIME, showing strong rank correlations with
both human-in-the-loop and fully manual judgments. Finally, de-
tailed analysis of our pipeline reveals that a strong LLM nugget
generator is key, and that the system rankings induced byDoGMa-
TiQare robust to outlier systems. We facilitate future research in
report evaluation by publicly releasing our code and artifacts.1
CCS Concepts
•Information systems →Multilingual and cross-lingual re-
trieval;Information retrieval;Evaluation of retrieval results.
Keywords
Report evaluation, information nuggets, question generation
∗Currently at Google Inc. Email: bryanli.ca@gmail.com
†Corresponding author. Email: wwalden1@jh.edu
1https://github.com/manestay/dogmatiq1 Introduction
The recent surge of interest in retrieval-augmented generation
(RAG) has created a parallel need for effective evaluation of RAG-
based tasks. Of these, generation of long-form, citation-backed re-
sponses to complex queries—report generation—is among the most
challenging and has featured in numerous TREC shared tasks, in-
cluding RAG [ 25], NeuCLIR [ 16], RAGTIME [ 17], and BioGen [ 13].
Extending a line of research dating to the TREC 2003 Question
Answering track [ 32,33], recent automatic evaluation methods
assess responses for how effectively they recover certain key pieces
of information, dubbednuggets, from the underlying collection
[15,23,26]. The majority of these methods, including AutoNugge-
tizer [ 26] and GINGER [ 15], represent nuggets as short factual
statements. A recall-based metric is then computed assessing the
proportion of the target topic’s nugget set attested by the report.
However, statement-based nuggets struggle in linguistically di-
verse environments and scale poorly to complex information needs.
To address this, we introduceDoGMaTiQ,2a nugget generation
pipeline that produces nuggets in the form ofquestion-answer (QA)
pairs. In our cross-lingual setting, this representation is essential:
user information needs are expressed in English while supporting
evidence may come from non-English documents. We argue QA
pairs offer three critical advantages over statement-based represen-
tations. First, QA pairs decouple an information need (the question)
from the content that satisfies it (the potentially diverse and multi-
lingual answers). Second, they prevent the arbitrary overweighting
of duplicate statements by consolidating redundant information
into a single nugget with multiple grounded answers. Third, QA en-
ables fine-grained assessments of nugget support via the choice of
aggregation over answers—e.g. requiringallanswers to be provided
versus requiring only a single answer.
DoGMaTiQadapts a generate-cluster-filter paradigm for the
structural complexities of nugget QA representation and report
evaluation. Given a set of retrieved passages for a user query,
DoGMaTiQaddresses QA curation in three stages: (1) document-
groundednugget generation, (2) QA-specificparaphrase clustering
(which handles the unique semantic overlap of questions versus
2Document- Grounded, Merged andTop-Cr iteria Question-based Nuggets.Dogmatic
means adherence to a set of principles, which accurately characterizes our approach.arXiv:2605.04458v1  [cs.CL]  6 May 2026

Li et al.
Generate QA nuggets(Where is Machu Picchu?, Peru)(Which continent is MP on?, South America)(When was MP built?, 1500s)(What is a mystery of Machu Picchu?, its exact purpose)Cluster nugget Qs(In which year was MP created?,  1440)(Which emperor built MP?, Pachacuti)(Who led the MP expedition?, Rualdo Menegat)
1. (What is a mystery of Machu Picchu?, [its exact purpose])2. (Where is Macchu Picchu?, [Peru])(When was MP built? [1500s, 1440])…K. (Which emperor built MP?, [Pachacuti])(Who led the MP expedition?, [Rualdo Menegat])(Which continent is MP on?, [South America])…summarize docs nativelygen QA nuggets in Englishembed Qs, get cos simverify close QsSelect top nugget Qs
cluster links
most commonSVM on metrics
hybrid+
Pick canonical Q; Clean As; Assign OR/AND
fasruszho
postprocessing
Figure 1: Illustration of theDoGMaTiQpipeline, showing the three main stages of (1) generating QA nuggets, (2) clustering
nugget questions, and (3) selecting the top nuggets for inclusion in a topic’s final nugget bank. Each stage consists of substeps,
where icons designate an LLM, programmatic method, or ML model. QA postprocessing occurs between stages (2) and (3).
answers), and (3) a finalnugget subselectionstep driven by a learned
model trained on principled linguistic and utility criteria. Critically,
nuggetanswersremain grounded in their original source documents,
while nuggetquestionsanchor related answers across documents.
Whereas prior approaches treat automatically generated nuggets
as intermediary artifacts for automatic report generation [ 6,15],
our experiments treat them instead as primary units of study for
automatic report evaluation.
We demonstrate the effectiveness ofDoGMaTiQthrough exper-
iments on the report generation shared tasks from the NeuCLIR
track at TREC 2024 and the RAGTIME track at TREC 2025. First,
we study the viability of usingDoGMaTiQ-generated nugget sets
in lieu of human-written sets, evaluating system-level rank corre-
lations based on nugget recall scores when using each set. Here,
we find strong correlations withDoGMaTiQ, outperforming corre-
lations when using nuggets generated from baseline systems and
from GINGER. Second, drawing on the recent Auto-ARGUE [ 34]
implementation of the ARGUE framework from Mayfield et al . [23] ,
we compare fully automatic nugget recall scoring usingDoGMa-
TiQ-generated nuggets with LLM-based nugget alignment to the
fully manual scores (including manual nugget alignment) provided
by track organizers—again obtaining favorable results. Finally, fur-
ther analyses validates the design decisions we make for different
stages of theDoGMaTiQpipeline. In sum, our contributions are:
•We introduceDoGMaTiQ, a pipeline for automatic gener-
ation of high-quality, document-grounded nuggets which
enable fully automatic QA nugget-based report evaluation.
•We show that system rankings based onDoGMaTiQ-generated
nuggets correlate highly with those based on human-writtennuggets, and modestly with rankings from fully manual
judgments.
•We confirm the quality ofDoGMaTiQ-generated nuggets
through several analyses, showing thatDoGMaTiQis ro-
bust to the influence of low-performing systems, and that
the nugget recall scores it induces correlate more highly
with scores from organizer-provided nuggets than with
scores based on synthetic nuggets from other systems.
2 Background
Nuggetsare atomic units of information and have been widely
used to evaluate long-form text [ 33]. Nugget-based evaluations can
broadly be split into two phases: (1)nugget creation(for a target
topic) and (2)nugget-to-text alignment, used to determine which
nuggets are attested in the target text. Here, we focus on the first
phase, developing an automated pipeline for nugget generation. For
the second phase, we use existing systems and judgments. Below,
we review the two most common nugget representations as well as
the literature on each phase.
Nugget Representations.Nuggets have largely been represented
in one of two forms. Historically, short statements orclaimsthat
describe key facts have been the most widely used form within
IR [26]. More recently, however, some works have focused instead
on representing nuggets as QA pairs [ 23], which have notably also
been used for summarization evaluation [ 4,8]. The QA form has a
key advantage over the claim form in that it decouples the infor-
mation need (represented by the question) from the content that
satisfies it (the answers), whereas a claim both implies an informa-
tion need and asserts the response. This decoupling thus enables
fine-grained assessments of information needs with multiple valid

DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation
responses, and also enables evaluation for multilingual retrieval,
where questions and answers may be in different languages.
2.1 Nugget Creation
Manual nuggets.Nuggets have traditionally been manually cu-
rated, and this has remained the dominant practice for benchmark-
ing in recent years. For example, organizers of the recent TREC 2024
NeuCLIR and TREC 2025 RAGTIME report generation tasks used
manually curated nuggets for official evaluation [ 16,17]. However,
this curation process requires substantial human effort, as assessors
must become familiar with new topics, read through documents on
those topics, and distill the most important information into clear
statements or QA pairs. For this reason, manual annotation does
not easily scale to new topics—a significant limitation.
Automatic nuggets.On the other end of the spectrum are fully-
automatic nugget generation approaches such as the Rubric Auto-
grader Workbench [ 5,10]. This system generates nuggets solely
from the query and is only sometimes competitive with manually
created questions [ 9]. To remedy this, recent approaches ground
nuggets directly in retrieved documents. The GINGER system ex-
tracts candidate statement nuggets from relevant documents, clus-
ters them, and uses an LLM to rerank the clusters for final selec-
tion [ 15]. Similarly, the Crucible system generates QA nuggets,
clusters them, and selects a top-ranked set [6].
At a high level,DoGMaTiQshares a generate-cluster-filter par-
adigm with GINGER and Crucible, but our framework makes three
concrete contributions that distinguish it from prior work. First,
compared to GINGER, each stage ofDoGMaTiQis explicitly tai-
lored to cross-lingual evaluation: we use QA nuggets, cluster on
questions to aggregate multilingual answers, and rank candidates
with explicit SVM weighting over textual criteria rather than opaque
LLM reranking. Second, both GINGER and Crucible are designed
forreport generation, whereasDoGMaTiQis designed as an in-
dependent framework forreport evaluation. Using the same LLM
paradigms for both generation and evaluation can introduce circu-
larity biases, which we evaluate in §6.2. Finally, through extensive
experiments on cross-lingual shared tasks, we identify the key com-
ponents ofDoGMaTiQ’s success: a strong LLM nugget generator
and an evaluation mechanism robust to outlier systems.
2.2 Nuggets for Evaluation
Given a nugget set for a target topic, the next phase of nugget-based
evaluation is to judge whether a report attests each nugget in the
set. The most intuitive approach is through nugget recall: for each
nugget, determine whether it is attested by the report. This is highly
compatible with the nugget statement format. Aside from reports,
nuggets have been used for summarization evaluation [ 19,24] and
ranked retrieval evaluation [27].
ARGUE.Mayfield et al . [23] propose ARGUE, a report evalua-
tion framework that evaluates reports along several dimensions:
completenessin their coverage of the report request,accuracywith
their presentation of facts, andverifiabilityof facts. A series of eight
binary judgments are made, which are used to compute two pri-
mary evaluation metrics: nugget recall and sentence precision. Thelatter metric assesses whether RAG systems are able to properly
cite documents for each factual statement in their reports.
Automating Report Evaluation.Substantial improvements in LLM
capabilities have led to their widespread use in evaluative annota-
tion tasks, such as pairwise preference labeling [ 2,22]—an evalua-
tion paradigm often calledLLM-as-a-judge. AutoNuggetizer, one
such LLM-as-a-judge framework, was introduced as part of the
TREC RAG 2024 shared task for automatic judgments of reports
by nugget recall [ 26]. Manual judgments were also collected, and
the organizers found a 𝜏= 0.87rank correlation with the auto-
mated approach [ 31]. As an alternative to binary nugget recall, the
Rubric Autograder Workbench approach asks an LLM to rate how
effectively a report addresses a specific nugget on a scale of 0–5 [ 5].
Auto-ARGUE is a robust, LLM-based implementation of the
ARGUE evaluation framework [ 34]. The authors show that this sys-
tem’s automated judgments demonstrate good system-level rank
correlations with human judgments on the NeuCLIR shared task.
However, because these results relied on the existence of manually-
created nugget sets, they can only be considered a partially auto-
mated (human-in-the-loop) report evaluation approach. In this work,
we address this remaining manual bottleneck withDoGMaTiQ.
2.3 Risks of LLM-Based Evaluation
LLM-as-a-judge evaluation frameworks have achieved strong cor-
relations with human judgments, as discussed above. However,
researchers have cautioned that such evaluations are prone to cer-
tain vulnerabilities that can lead to inflated results. This is especially
the case when correlations are demonstrated on only a limited set of
domains and in tightly controlled settings. Our paper on automated
evaluation is thus aware of, and assesses, the following risks:
Circular judgments.It is increasingly common for the same LLM
that generated a given text to also be used to score that text. This
can lead tocircular judgments[ 3,7], which may inflate metric scores.
Critically, this circularity may not be deliberate, and may occur
due toincidentalexposures to the same artifact, whether via the
same underlying LLM or a similar prompt. This paper contains one
study on circularity (§6.2), where we find that reports generated by
Crucible, a QA-nugget aware system [ 6], achieve outlier perfor-
mance when using the same LLM generator (Llama). Fortunately,
DoGMaTiQavoids circularity by using a different LLM (Claude).
LLM ranking correlations can be inflated by low-ranked systems.
Clarke and Dietz [3]note that it is most critical for evaluation
systems to identify meaningful differences betweentop-ranked
systems, rather than across the whole distribution. When there are
many underperforming systems that can easily be distinguished,
the resulting rank correlations are inflated. Clarke and Dietz [3]
demonstrate this by starting with a 𝜏=0.89result between manual
and automatic assessment on 75 systems, from Upadhyay et al . [31] .
They progressively drop lower-ranked systems and find a decrease
to𝜏=0.51for the top 20. We perform a similar experiment in §6.1,
and find thatDoGMaTiQis robust to such issues, with subset rank
correlations remaining nearly equivalent to those for the full set.

Li et al.
3 TheDoGMaTiQPipeline
DoGMaTiQconsists of three main stages, each with its own sub-
steps. The key idea is to decompose the broader task of creating an
ideal nugget bank for some topic into subtasks that can be addressed
with existing NLP methods. The goal is to generate nuggets that
are both highly relevant (by grounding nugget answers in relevant
documents) and comprehensive (by producing nugget questions
that cover diverse facets of the user’s query). Figure 1 visualizes the
full pipeline. For each topic, we first generate QA nuggets for each
document in some input set. Second, we identify and merge para-
phrases of nugget questions. Finally, we rank the merged nuggets
and select the most salient ones for inclusion in the final nugget
bank. Note that, as our pipeline focuses on report evaluation, we do
not consider retrieval of input documents as part ofDoGMaTiQ.
As such, we use existing IR systems to retrieve this set.
3.1 Stage 1: Document-Grounded QA Nugget
Generation
Generation of QA pairs from single documents is a well-explored
NLP task [ 12,14,20,29]. In our case, we wish to generate English
QA pairs from documents that may or may not be in English. While
cross-linguality was achieved in earlier work through separate
English generation and machine translation systems [ 18], a single
contemporary LLM suffices to handle multiple source and target
languages.3Here, we use a summarize-then-generate approach.
Summarization.Directly generating QA nuggets from a docu-
ment often over-indexes on document-specific details, rather than
the broader information need. This was first observed by Liu et al .
[21], who used LLMs to generate QA pairs over multilingual doc-
uments. We thus follow their advice to first prompt an LLM to
summarize each document, retaining only the most salient informa-
tion. Summaries are generated in the document’s original language
to minimize loss of key information due to translation errors.
Generating QA Pairs.Given a summary and a user query, we
prompt an LLM to generate 1–6 QA pairs, each covering a single
fact that is verifiable from the document. Regardless of the summary
language, we elicit the QA pairs in English. Our prompt has clear
output specifications and includes several carefully curated few-
shot exemplars. Answers are generated before their questions so as
to minimize hallucinations. Thus, by construction, each nugget is
grounded in a specific document, and this grounding information
is propagated through the rest of the pipeline. We apply the sum-
marization and generation steps to each relevant document for a
topic, obtaining a large set of candidate nuggets.
3.2 Stage 2A: Identifying Question Paraphrases
through Clustering
Since QA pairs are generated independently for each document, the
resulting candidate nuggets may have significant semantic overlap.
For example, consider the following two QA pairs:
(What is the Statue of Liberty’s height?, 305 feet)
(How tall is the Statue of Liberty?, 93 meters)
3Provided the languages are relatively high-resource, which they are in this work.Both the questions and the answers are synonymous (modulo the
answers’ unit conversion). Using the QA format enables us to merge
these two nuggets into one, preserving just one of the question
texts but retaining both answers (either of which is permissible),
along with their document grounding information. In contrast, with
the statement-based nuggets used in GINGER and AutoNuggetizer,
these would be unnaturally treated as distinct statements.
Paraphrase Identification.We treat question deduplication as a
paraphrase identification task. Although prompting LLMs to de-
termine whether two sentences are paraphrases of each other is
quite effective [ 37], this is expensive for large 𝑁given the quadratic
number of comparisons.4Thus, we first identify candidate pairs
with a small, fine-tuned embedding model (see §4), and label pairs
for which the embeddings’ cosine similarity exceeds 0.9 as para-
phrases.5We then elicit more reliable paraphrase judgments from
an LLM on this smaller candidate set. Our prompt for this task has
clear instructions along with several difficult few-shot exemplars.
The resulting set of verified paraphrase pairs is treated as edges in
a graph, and we find larger paraphrase clusters through single-link
clustering (i.e. connected components).
3.3 Stage 2B: Nugget Refinement
Following clustering, we apply several further refinement steps
to ensure that nuggets have high factual precision and satisfy the
structural requirements of the ARGUE framework.
Canonical Question Selection.From each cluster of paraphrased
questions, we select a single canonical question that best represents
the latent information need. We zero-shot prompt an LLM to do
this, providing the original query as additional context.
Answer Set Validation.Despite our quality assurance measures
in stages (1) and (2A), generated nugget answers may still be unin-
formative or inaccurate. We address these issues with two layers of
validation on answer sets. First, we use a regular expression to filter
out uninformative responses (e.g. ‘none,’ ‘null,’ ‘no answer,’ ‘un-
known’). Second, we verify the factual consistency of answer sets
by prompting an LLM to remove any answers it deems implausible
or contradictory given the others in the set.6As this involves some
subjective judgment, we again ensure that our prompt includes
detailed instructions and few-shot exemplars.
Finally, we cull any nugget questions with no answers remaining
at the end of this process. Note that our validation can onlyremove
answers, thus improving theprecisionof the answer sets.Recall
is much less of a concern due to the large number of candidate
nuggets that still remain after this step.
Answer Aggregator Selection.ARGUE supports specifying a logi-
cal aggregation over nugget answers to enable accurate determina-
tion of when a nugget question is correctly addressed. In this final
refinement step, we prompt an LLM to analyze the relationship
among answers to each nugget question and assign that nugget an
4With only 1000 sentences, this is already 1000
2= 499500 comparisons.
5This threshold was determined via manual experimentation on a small set of queries.
6We note that for some scenarios, preserving contradictory responses may be desirable.

DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation
aggregator: ORis assigned whenany(one) answer suffices, while
ANDis assigned whenallmust be provided.7
3.4 Stage 3: Final Nugget Selection
The previous stages yield a large, comprehensive set of candidate
nuggets. However, reports are almost always limited in content
relative to the amount of relevant material in the collection they
draw on. Thus, nugget-based evaluations must identify a privileged
set of the most critical nuggets thatanyhigh-quality report on
the target topic must cover; this is the goal of stage (3). Given a
ranking of nuggets, we select the top 20 for each topic, observing
that human-authored nugget banks for topics in NeuCLIR and
RAGTIME generally feature 10–30 nuggets. We consider several
methods forimportance-based ranking of nugget questions:
Method A: MostCommon.This method adopts the rationale of the
pyramid method [ 24]: the number of distinct documents in which a
fact is mentioned can serve as a signal for the fact’s importance. We
thus rank nuggets by their occurrence frequency, based primarily on
the number of questions in the paraphrase cluster for each nugget,
breaking ties based on the total number of grounding documents
across all answers.
Method B: Quality Criteria (DoGMaTiQ).While document fre-
quency is a strong signal of importance, it risks overlooking nuggets
that are attested in only a handful of documents but that are nonethe-
less essential to addressing the request. This method takes an alter-
native approach, representing each nugget as a vector of 19 explicit
quality criteria (detailed in Table 1). These criteria, adapted from
prior works [ 28,36], fall into two categories:linguistic quality(e.g.
fluency and reading level) andinformation utility(e.g. personaliza-
tion and criticality). We author LLM prompts to obtain most of these
labels. We then train a Support Vector Machine (SVM) to act as a
weighted average to effectively distinguish between high-quality,
critical nuggets and lower-quality candidates.
Method C:Sample.This method randomly selects 20 nuggets
from the fullDoGMaTiQ-generated candidate set, effectively ablat-
ing the subselection process.
4 Experimental Setup
Datasets.Our primary study on the efficacy ofDoGMaTiQfo-
cuses on the TREC 2025 RAGTIME collection [ 17],8which contains
news documents in English, Arabic, Chinese, and Russian. We con-
sider the 16 test topics from the report generation task, and the 59
valid system submissions. We compare system-level correlations be-
tween rankings based onAuto-ARGUE nugget recall scores using
(1) ourDoGMaTiQ-generated nugget sets and (2) human-authored
nugget sets, with the latter drawn from the official evaluation.
Additionally, we study rank correlations between manual nugget
recall judgments and automated ones. As manual judgments are not
available for RAGTIME at the time of writing,9we instead consider
7In principle, combinations of ANDs and ORscould be used for a single nugget. We do
not pursue such an extension in this work.
8https://trec-ragtime.github.io/index.html
9We have obtained these from the organizers after the completion of this draft. We
include the fully automatic vs. fully manual report evaluation for RAGTIME in Appen-
dix A.1. These results reinforce our overall findings in §5.1 and §5.2.Table 1: Quality criteria used for assessing nuggets. All crite-
ria are assessed via LLM prompting except for 1 and 2.
# Criterion ScaleBasic1 Reading Level 4.0–13.0
2 Complexity 1.0–6.0
3 Vitality BinaryPersonalization4 Goal Match 0.0–1.0
5 Background Match 0.0–1.0
6 Role Match 0.0–1.0
7 Communication Match 0.0–1.0
8 Scope Match 0.0–1.0
9 Overall 0.0–1.0General10 Fluency 1.0–5.0
11 Clarity 1.0–5.0
12 Ambiguity 1.0–5.0
13 Relevance 1.0–5.0
14 Incompleteness 1.0–5.0
15 Assumptiveness 1.0–5.0
16 Multifaceted 1.0–5.0
17 Knowledge Intensiveness 1.0–5.0
18 Subjectiveness 1.0–5.0
19 Reasoning Intensiveness 1.0–5.0
assessor judgments on the 21 topics and 51 systems from the cross-
lingual report generation pilot task from the TREC 2024 NeuCLIR
track [ 16]. NeuCLIR was the predecessor to RAGTIME and also has
documents in the news domain, but in Chinese, Russian, and Farsi.
Aside from rank correlations, §6 has experiments that plot the
actual scores output by different automatic nugget systems as (𝑥,𝑦)
points vs. manual nuggets, as well as a qualitative study onDoG-
MaTiQnuggets for a single NeuCLIR topic.
Metrics.Following ARGUE [ 23], we evaluate systems vianugget
recall, defined as the proportion of nuggets for a given topic that are
correctly addressed in a system report.10Scores are macro-averaged
across topics to produce an overall evaluation score for each system.
To assess the quality of different nugget sets, we examine their effect
on nugget recall using an automatic nugget scanning system. In
our experiments, we useAuto-ARGUE, although other tools such
as the RUBRIC Autograder Workbench [ 5] or AutoNuggetizer [ 26]
could be used instead.
Our objective is to identify nugget sets that yield a leaderboard
that is maximally similar to the official leaderboard. We quantify this
similarity using Spearman’s 𝜌as well as weighted and unweighted
Kendall’s 𝜏, determining which nugget set best matches the official
ranking. As a measure of statistical significance, we follow [ 34] in
reporting Wilcoxon paired accuracy (WPA), the probability that
two Wilcoxon tests agree on the relative ranking of any given pair
of runs, computed over all pairs.
Baseline: GINGER.We consider another nugget generation ap-
proach, GINGER [ 15] as a baseline, and apply it to RAGTIME topics
and documents to obtain alternative nugget sets for each topic.
We useAuto-ARGUE to score reports with GINGER-generated
10ARGUE also evaluates citation support via asentence precisionmetric, though we
ignore this given our focus on nuggets.

Li et al.
nuggets,11and consider nugget recall measure. We rank systems by
this measure to obtain the leaderboard under the GINGER method.
DoGMaTiQPipeline Configuration.Below we provide implemen-
tation details for each stage of our pipeline.
Stage 0 (Retrieval) We use PLAID-X [ 35] as our cross-lingual
IR system to retrieve input documents.
Stage 1 We use Claude 3.5 Sonnet [ 1] to summarize the docu-
ments and generate candidate nuggets.
Stage 2 For paraphrase identification (stage (2A)), we fine-
tune a pretrained bge-large-en-v1.5 checkpoint on the Quora
duplicate questions dataset.12For all refinement steps (stage
(2B)), we use Llama 3.3 70B Instruct [11].
Stage 3 We train an SVM classifier to select the final nuggets
based on the 19 quality criteria discussed above. Positive
training examples were drawn from human-written nuggets
and negative examples were then mined from nuggets gen-
erated byDoGMaTiQthat were judgednotto be para-
phrases of the positive examples.
Compared Systems.We empirically compare:
Manual Official manual evaluation scores for each system.
This is our gold standard for our leaderboard correlation
experiment. These scores were obtained by manually de-
signing nuggets, then manually aligning them to responses.
DoGMaTiQ Our full approach with quality criteria (method
B), as described in §3.4. We use these nuggets with the
automated evaluation systemAuto-ARGUE.
Common A variant of our approach where we only use most
common nuggets (method A).
CommonLSame as above but using Llama for generation.
Sample An ablation of our approach, where we choose a
subset of 20 randomly sampled nuggets (method C).
SampleLSame as above but using Llama for generation.
Manual Nuggets Using official manually created nuggets
withAuto-ARGUE.
GINGER Nuggets Using nuggets created by the GINGER
system [15], withAuto-ARGUE.
5 Main Results
Our main experiments show the efficacy ofDoGMaTiQnuggets
for automated report evaluation, comparing run scores from our au-
tomated nuggets to those obtained using manually written nuggets.
For manually written nuggets, we consider manual assessments of
nugget recall on NeuCLIR and automatic assessments withAuto-
ARGUE on RAGTIME. For automated nuggets, we compare the
three nugget selection methods from stage (3) ofDoGMaTiQ.
5.1 Report Evaluation: Manual Nuggets vs.
DoGMaTiQNuggets
Table 2 shows results on RAGTIME, and we summarize our main
findings below. We first compare systems when using Claude 3.5
Sonnet for candidate nugget generation. Depending on the metric,
11For compatibility with GINGER, in this baseline experiment we modifiedAuto-
ARGUE’s prompts to take nugget statements rather than QA nuggets.
12https://huggingface.co/datasets/sentence-transformers/quora-duplicatesTable 2: Main results on RAGTIME report generation runs.
Each row shows rank correlation metrics based on nugget
recall for different nugget systems, all automated. The refer-
ence leaderboard underlying these correlations uses manual
nuggets scored withAuto-ARGUE.
Nugget System 𝜌 𝜏Wtd.𝜏WPAClaudeGINGER Nuggets 0.793 0.640 0.594 0.819
Sample 0.737 0.591 0.639 0.793
Common 0.694 0.541 0.476 0.772
DoGMaTiQ 0.899 0.734 0.747 0.866LlamaSampleL 0.506 0.366 0.175 0.681
CommonL 0.442 0.296 0.101 0.648
DoGMaTiQL 0.465 0.319 0.152 0.656
Table 3: Main results on NeuCLIR report generation runs.
Each row shows rank correlation metrics based on nugget
recall for different nugget systems (manual or automated).
The reference leaderboard underlying the correlations uses
manual nuggets scored by human assessors.
Nugget system 𝜌 𝜏Wtd.𝜏WPA
Sample 0.720 0.552 0.571 0.775
DoGMaTiQ 0.768 0.586 0.614 0.792
Manual Nuggets 0.804 0.630 0.640 0.815
DoGMaTiQhas either very high ( 𝜌= 0.899) or moderate corre-
lation ( 𝜏= 0.734) with official leaderboards based on manually
designed nuggets that are manually matched.
Sample( 𝜌=0.737) andCommon( 𝜌=0.694) substantially under-
perform, even though they draw from the same candidate nuggets as
DoGMaTiQ. GINGER is a distant second best ( 𝜌=0.793, 𝜏= 0.640),
as its nugget filtering stage is a simpler reranking. This demon-
strates the effectiveness ofDoGMaTiQ’s utilization of 19 qual-
ity criteria with a learned SVM weighting. Results with Wilcoxon
paired accuracy (WPA) exhibit the same trends as the other metrics.
We found that using nuggets generated by Llama is detrimental
to rank correlations, with weighted 𝜏dropping as low as 0.101.
Manual inspection confirmed these nuggets were well-formed and
factually grounded, but tended to be less personalized than Claude
nuggets. Circular judgments may be the culprit, as 10 systems
adopted a nugget-aware approach that used Llama (§6.2). In con-
trast, the mainDoGMaTiQpipeline used Claude and avoided this.
5.2 Report Evaluation: Fully Automatic vs. Fully
Manual Scoring
We next assess the degree to which fully automated evaluation
usingDoGMaTiQfor nuggets andAuto-ARGUE for judgments,
can simulate fully manual evaluation with human-written nuggets
and assessors on NeuCLIR. Results are shown in Table 3. Recall that
Manual Nuggets is a human-in-the-loop approach that also uses
Auto-ARGUE, thus serving as an upper bound for correlations.13
13The “Manual Nuggets” setting represents a reproduction of an experiment from [ 34].
While they report 𝜏=0.73, we find 𝜏=0.63. We are currently discussing with those

DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation
Table 4: Results on a subset of RAGTIME systems marked
as “highest priority” for each team (12 total). Each system is
evaluated withAuto-ARGUE. Each cell also shows the dif-
ference 𝛿between the metric value on these 12 runs and the
same metric value on all 59 runs. The first column indicates
whether nuggets were generated with Llama or Claude.
Nuggets 𝜌 𝜏Wtd.𝜏WPAClaudeGINGER Nuggets 0.657
(𝛿-0.136)0.545
(𝛿-0.095)0.528
(𝛿-0.066)0.773
(𝛿-0.046)
Sample 0.796
(𝛿
+0.059)0.677
(𝛿
+0.086)0.713
(𝛿
+0.074)0.833
(𝛿
+0.040)
DoGMaTiQ 0.883
(𝛿-0.016)0.748
(𝛿
+0.014)0.687
(𝛿-0.060)0.864
(𝛿-0.002)LlamaSampleL 0.608
(𝛿
+0.102)0.515
(𝛿
+0.149)0.455
(𝛿
+0.280)0.758
(𝛿
+0.077)
DoGMaTiQL 0.580
(𝛿
+0.115)0.485
(𝛿
+0.166)0.373
(𝛿
+0.221)0.742
(𝛿
+0.086)
We find that, depending on the metric, bothSampleandDoGMaTiQ
achieve 92% to 97% of the maximum possible (Manual) correlation,
showing that fully automatic evaluation withDoGMaTiQnuggets
is viable while alleviating assessor burden.
Since the initial submission of this paper, the RAGTIME manual
judgments have been released. We include these results in Appen-
dix A.1, which largely concur with our findings here for NeuCLIR.
6 Analysis
6.1 Robustness to Underperforming Systems
In this experiment, we address the risk of rank correlations being
inflated by many low-performing systems ([ 3]; see §2.3). We simu-
late a plausible scenario in which one wants to useDoGMaTiQto
rank a subset of the strongest systems for a shared task (here: RAG-
TIME), but without prior knowledge of the outcome of a manual
assessment. As we cannot know which systems will be strongest,
we use participants’ own priority labels for runs as a proxy, taking
the highest priority run for each team. This yields a 12-run subset
on which we perform rank correlation analysis.
Table 4 presents the results. We find that the metrics forDoGMa-
TiQare stable across the board, with much smaller relative changes
compared to all other systems ( 𝜏=+ 0.01or 2% relative to 𝜏com-
puted over all runs). In contrast, GINGER suffers a 𝜏=− 0.1/14.8%
relative drop.
Interestingly, all other variants of our approach see relatively
large improvements in correlation over the subset compared to the
full set of runs. This could be due to teams submitting multiple very
similar systems, and to the weaker variants ofDoGMaTiQthus
struggling to discriminate between them. That the full version of
authors what could account for this discrepancy. The results in this section do not
depend on their reported results, however, and thus remain valid.
Figure 2: Scatterplot comparing the macro-average nugget
recall scores under manual nuggets vs. automated nugget sets
for each leaderboard system. This is for all runs submitted
to the RAGTIME report generation task. Points above the
diagonal represent systems whose manual score is higher
than underDoGMaTiQ, while below-diagonal points denote
systems where the manual score is lower.
DoGMaTiQdoes not suffer from this issue illustrates its robustness
to variation in the set of systems under evaluation.
6.2 Raw Nugget Recall Scores
Inspection of raw nugget recall scores under different evaluation
methods reveals additional insights. This experiment follows the
RAGTIME setting withAuto-ARGUE judgments from §5.1. Figure 2
shows a scatterplot where the 𝑥-coordinate of each point gives the
nugget recall score using automatically generated nuggets from a
given system and where the 𝑦-coordinate gives the nugget recall
score using manually created nuggets.
We observe that points forDoGMaTiQlie closest to the diagonal,
showing a reasonably narrow spread and no outliers. Points for
GINGER show a wider spread and consistently higher scores than
manual nugget evaluations. In contrast,Sampleconsistently yields
scores lower than those from manually generated nuggets, with a
narrow spread and a few outliers. Finally,DoGMaTiQLshows the
lowest scores and the widest spread.
Figure 2 thus illustrates thatDoGMaTiQproduces nugget sets
whose scores most closely match those found under evaluation
with human-written nuggets. GINGER instead tends to produce
more easily answerable nuggets, whileSampleandDoGMaTiQL
tend to produce more difficult ones.
Study on Circularity.Figure 2 also offers insights into circular
judgments (§2.3). Ten systems usedCrucible, a nugget-aware re-
port generation system [ 6]. Considering the yellow points forDoG-
MaTiQL, we observe a cluster of relative outliers towards the right

Li et al.
Figure 3: Scatterplot comparing nugget recall scores using
manual nuggets vs.DoGMaTiQnuggets, scored withAuto-
ARGUE. Red dots are systems’ macro-average scores over
topics, while pink dots plot systems’ per-topic scores.
of others, which strongly contribute to the decreased rank corre-
lations. Further evidence comes fromDoGMaTiQLscores being
much higher on the top-system subset than on all systems (§6.1).
In contrast,DoGMaTiQavoids this circularity by using a differ-
ent LLM (Claude). The observed performance drop forDoGMaTiQL
is therefore primarily a symptom of evaluation circularity inherent
to the RAGTIME track composition, rather than a structural flaw
in our pipeline. These results thus emphasize the importance of
decoupling the evaluation model from generation models [3, 7].
6.3 System-Level vs. Topic-Level Scores
So far, we have considered only system-level scores, which repre-
sent macro-averages across topics. We now turn to per-topic scores:
Figure 3 presents a scatterplot analogous to the one in Figure 2, but
where points represent nugget recall scores for individual runs on
individual topics usingDoGMaTiQnuggets.14We corroborate find-
ings from [ 30] that despite strong system-level rank correlations,
per-topic points exhibit high variance. Such variance is unsurpris-
ing: Creating nugget sets is an inherently subjective process, and
different annotators—whether human or machine—will differ in
which pieces of information they deem most central to a given
topic [ 16]. Running a system over multiple topics can be viewed
analogously to drawing samples from a larger population of valid
nuggets. Upadhyay et al . [30] estimate that having scores on 10 top-
ics suffices to achieve good rank correlations, which is consistent
with our results usingDoGMaTiQon these 16 RAGTIME topics.
6.4 Rank Correlations Between Nugget Sets
Next, we consider how the nugget sets generated by different auto-
matic systems compare to one another. Figure 4 shows system-level
14Appendix Figure 6 shows these scatterplots forDoGMaTiQ,Sample,DoGMaTiQL,
and GINGER nuggets.
Figure 4: Heatmaps showing rank correlation with official
fully manual leaderboards ( 𝜏and 𝜌) when scoring RAG-
TIME topics with different nugget sets. We compare three
automated systems with official leaderboards, yielding nine
unique comparisons. The bottom row numbers are also pro-
vided in Table 2. This shows thatDoGMaTiQobtains the best
correlation with manual leaderboards.
rank correlations between nugget recall scores on RAGTIME runs
under nugget sets generated by different methods. First, we note
that scores usingDoGMaTiQnuggets correlate better with those
based on manual nuggets ( 𝜌= 0.899, 𝜏= 0.734) than with those
based on any automatic system—once again testifying toDoGMa-
TiQ’s effectiveness. Second, we find thatDoGMaTiQand GINGER
rankings correlate more strongly with each other than either does
withSample—emphasizing that these systems’ nugget selection
step is important to their success. Third, we observe that correla-
tions among all four automated systems largely fall in the moderate
range. This could be due in part to limitations of the LLM judge from
Auto-ARGUE used here to assess nugget recall, and it is possible
we would observe higher correlations under human assessments of
nugget correctness using the same nugget sets.
6.5 Qualitative Analysis
Lastly, we present qualitative analysis comparing the set of 10
nuggets created by human assessors against the set generated by
DoGMaTiQfor topic 361 (on the “Murchison meteorite, its dis-
covery and composition”) from NeuCLIR. We use our paraphrase
detection model (§4) and a stable matching algorithm to assign each
manually written nugget to its most similar nugget from the set
generated byDoGMaTiQ.
Figure 5 presents the resulting alignment. The matched pairs in
(1) and (2) are effectively synonymous; the questions differ slightly
in their level of detail, but they seek the same information and fea-
ture the same answers (not pictured). Although the remaining eight
pairs in the alignment are not synonymous, they nonetheless cover
many of the same topics (e.g. the composition of the meteorite,
the age of the materials it contains, and scientific analysis con-
ducted on it). Furthermore, the level of specificity and the style are
not noticeably different between the two sets. Briefly highlighting
several pairs, we note that for pair (5), the GOLD nugget is under-
specified and requires additional context, while the GEN nugget
is specific and standalone. The questions in pair (6) are entirely

DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation
Clearly matched: 2
1) GEN: What are the ancient particles in the Murchison meteorite called?
GOLD: The presolar grains in the Murchison meteorite were mostly composed of what? SIM: 0.732
2) GEN: Where did the Murchison meteorite fall in 1969?
GOLD: Where did the Murchison meteorite make landfall? SIM: 0.745
Unclearly matched: 8
3) GEN: How many pre-solar grains were analyzed by the research team?
GOLD: How does the age of the grains compare to that of our Solar System? SIM: 0.643
4) GEN: How many ancient dust particles were extracted from the Murchison meteorite?
GOLD: How heavy was the Murchison meteorite? SIM: 0.634
5) GEN: How old is 60% of the stardust in the Murchison meteorite?
GOLD: How old were these grains? SIM: 0.521
6) GEN: Which university's researchers discovered asymmetric molecules in meteorites?
GOLD: What advances could the discovery of water molecules in the Murchison meteorite lead to?
SIM: 0.693
7) GEN: What ancient materials were found in the Murchison meteorite?
GOLD: What did the scientists use pieces of the meteorite for? SIM: 0.608
8) GEN: What method was used to discover sugars in the Murchison meteorite?
GOLD: What does the discovery of sugars in the meteorite indicate about the origin of life on Earth?
SIM: 0.874
9) GEN: What sugars were found in the Murchison meteorite?
GOLD: What if any organic compounds were found in the Murchison meteorite? SIM: 0.691
10) GEN: How long before the solar system was the oldest material in the Murchison meteorite formed?
GOLD: What other materials do we have that are older than the Solar System? SIM: 0.605
Figure 5: Comparison of nuggets fromDoGMaTiQ(GEN) and human-written (GOLD) for NeuCLIR topic 361. SIM is
the pairwise cosine similarity from a paraphrase detection model. Each GOLD nugget is assigned to its closest GEN
nugget through stable matching. “Clearly” and “Unclearly” matched pairs are judged manually by the authors.
distinct, though the GOLD nugget is more narrowly tailored to the
topic. Finally, for pair (9), although the questions are similar and
the answers are the same, the GEN nugget is more specific in the
types of compounds it asks about (sugars vs. organic compounds).
7 Conclusion
We have introducedDoGMaTiQ, a pipeline for generating high-
quality, document-grounded QA nuggets. Prior work in automated
report evaluation has focused on simple, statement-based nuggets.
In contrast,DoGMaTiQproduces QA nuggets, which allow for a
much richer representation of information needs versus the infor-
mation that satisfies them, and are compatible with the ARGUE
framework for report evaluation. We showed that usingDoGMa-
TiQnugget sets for automated report evaluation achieves strong
correlation scores with human-written nugget sets.
We further investigated the design decisions for each stage of
theDoGMaTiQpipeline, finding that success relies heavily on com-
bining a strong LLM for nugget generation, and weighted quality
criteria for nugget selection. Each stage of our pipeline mitigates er-
rors and refines precision from earlier stages. Our further analyses
show thatDoGMaTiQsystem rankings are robust to underper-
forming systems, avoid the evaluation circularity pitfalls of prior
paradigms, and yield macro-level scores that correspond much
better to those from human-written nuggets.We emphasize thatDoGMaTiQand related methods are meant to
complement, rather than replace, expert human effort. Automated
methods enable fast, scalable evaluations for arbitrary generated
reports, helping researchers understand the topics and user profiles
where current systems underperform. As generative AI continues to
reshape how information is created and consumed, it has never been
easier to use these systems for generative purposes, making the
challenge of evaluating quality ever more urgent. WithDoGMaTiQ,
we take a step toward closing that gap: we provide automated tools
to assess the degree to which different information systems can
satisfy information needs.
A Additional Figures and Diagrams
A.1 Fully Automatic vs. Fully Manual Scoring
for RAGTIME’25
Table 5 shows the rank correlation metrics with the manual judg-
ments and manual nuggets from the official RAGTIME assessments.
Recall that the first three rows are fully automated evaluations,
while Manual Nuggets is human-in-the-loop, since it still uses Auto-
ARGUE.15We see thatDoGMaTiQis the best performing system
again, with only 13% lower 𝜏than manual, handily beatingSample
and GINGER.
15Our Manual Nugget 𝜏= 0.578on nugget recall closely aligns to the 𝜏= 0.549
reported in Table 6 of the of RAGTIME’25 paper [17].

Li et al.
Table 5: Main results on RAGTIME report generation runs.
Each row shows rank correlation metrics based on nugget
recall for different nugget systems, manual or automated.
The reference leaderboard underlying the correlations are
manual nuggets scored by humans (the official assessors).
Nugget system 𝜌 𝜏Wtd.𝜏WPA
DoGMaTiQ 0.664 0.505 0.383 0.750
Sample 0.507 0.372 0.291 0.686
Manual Nuggets 0.748 0.578 0.589 0.786
A.2 System-level vs. Topic-level Scatterplots
Appendix Figure 6 shows the system-level vs. topic-level scatter-
plots forDoGMaTiQ,Sample,DoGMaTiQL, and GINGER nuggets.
Acknowledgments
This research project was pursued as a part of the SCALE 2025
program at the HLTCOE at Johns Hopkins University. We thank
the fellow SCALE participants for their discussions and feedback
throughout the course of this project, including Hannah Recknor,
Rebecca Kotula, Jia-Huei Ju, and Dayeon Ki.

DoGMaTiQ: Automated Generation of Question-and-Answer Nuggets for Report Evaluation
Figure 6: Scatterplots comparing nugget recall scores using manual nuggets vs. different automated nuggets, scored with
Auto-ARGUE. Darker dots are systems’ macro-average scores over topics, while lighter dots plot systems’ per-topic scores.
References
[1]Anthropic. 2024.Introducing Claude 3.5 Sonnet. https://www.anthropic.com/
news/claude-3-5-sonnet Accessed: 2026-01-20.
[2]Negar Arabzadeh and Charles L. A. Clarke. 2025. Benchmarking LLM-Based
Relevance Judgment Methods. InProceedings of the 48th International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR ’25).
ACM. doi:10.1145/3726302.3744382
[3]Charles LA Clarke and Laura Dietz. 2024. LLM-based relevance assessment
still can’t replace human relevance assessment.arXiv preprint arXiv:2412.17156
(2024).
[4]Daniel Deutsch, Tania Bedrax-Weiss, and Dan Roth. 2021. Towards question-
answering as an automatic metric for evaluating the content quality of a summary.
Transactions of the Association for Computational Linguistics9 (2021), 774–789.
[5]Laura Dietz. 2024. A workbench for autograding retrieve/generate systems.
InProceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval. 1963–1972.
[6]Laura Dietz, Bryan Li, Gabrielle Liu, Jia-Huei Ju, Eugene Yang, Dawn Lawrie,
William Walden, and James Mayfield. 2026. Incorporating Q&A Nuggets into
Retrieval-Augmented Generation. InProceedings of the 48th European Conference
on Information Retrieval (ECIR 2026).[7] Laura Dietz, Bryan Li, Eugene Yang, Dawn Lawrie, William Walden, and James
Mayfield. 2026. Insider Knowledge: How Much Can RAG Systems Gain from
Evaluation Secrets?. InProceedings of the 48th European Conference on Information
Retrieval (ECIR 2026).
[8] Matan Eyal, Tal Baumel, and Michael Elhadad. 2019. Question answering as an
automatic evaluation metric for news article summarization. In2019 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies, NAACL HLT 2019. Association for Computational
Linguistics (ACL), 3938–3948.
[9] Naghmeh Farzi and Laura Dietz. 2024. Exam++: Llm-based answerability met-
rics for ir evaluation. InProceedings of LLM4Eval: The First Workshop on Large
Language Models for Evaluation in Information Retrieval.
[10] Naghmeh Farzi and Laura Dietz. 2024. Pencils down! automatic rubric-based
evaluation of retrieve/generate systems. InProceedings of the 2024 ACM SIGIR
International Conference on Theory of Information Retrieval. 175–184.
[11] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Ab-
hishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schel-
ten, Alex Vaughan, et al .2024. The llama 3 herd of models.arXiv preprint
arXiv:2407.21783(2024).
[12] Shash Guo, Lizi Liao, Cuiping Li, and Tat-Seng Chua. 2024. A survey on neural
question generation: methods, applications, and prospects. InProceedings of the

Li et al.
Thirty-Third International Joint Conference on Artificial Intelligence(Jeju, Korea)
(IJCAI ’24). Article 889, 10 pages. doi:10.24963/ijcai.2024/889
[13] Deepak Gupta, Dina Demner-Fushman, William Hersh, Steven Bedrick, and Kirk
Roberts. 2024. Overview of TREC 2024 biomedical generative retrieval (BioGen)
track.arXiv preprint arXiv:2411.18069(2024).
[14] Kalpesh Krishna and Mohit Iyyer. 2019. Generating Question-Answer Hi-
erarchies. InProceedings of the 57th Annual Meeting of the Association for
Computational Linguistics, Anna Korhonen, David Traum, and Lluís Màrquez
(Eds.). Association for Computational Linguistics, Florence, Italy, 2321–2334.
doi:10.18653/v1/P19-1224
[15] Weronika Łajewska and Krisztian Balog. 2025. Ginger: Grounded information
nugget-based generation of responses. InProceedings of the 48th International
ACM SIGIR Conference on Research and Development in Information Retrieval.
2723–2727.
[16] Dawn Lawrie, Sean MacAvaney, James Mayfield, Paul McNamee, Douglas W
Oard, Luca Soldaini, and Eugene Yang. 2025. Overview of the trec 2024 neuclir
track.arXiv preprint arXiv:2509.14355(2025).
[17] Dawn Lawrie, Sean MacAvaney, James Mayfield, Luca Soldaini, Eugene Yang,
and Andrew Yates. 2025. Overview of the trec 2025 RAGTIME track.The
Thirty-Fourth Text REtrieval Conference Proceedings (TREC2025)(2025).
[18] Bryan Li and Chris Callison-Burch. 2023. PAXQA: Generating Cross-lingual
Question Answering Examples at Training Scale. InFindings of the Association
for Computational Linguistics: EMNLP 2023. 439–454.
[19] Jimmy Lin and Dina Demner-Fushman. 2006. Will pyramids built of nuggets
topple over?. InProceedings of the Human Language Technology Conference of the
NAACL, Main Conference. 383–390.
[20] Zefeng Lin, Weidong Chen, Yan Song, and Yongdong Zhang. 2024. Prompting
Few-shot Multi-hop Question Generation via Comprehending Type-aware Se-
mantics. InFindings of the Association for Computational Linguistics: NAACL
2024, Kevin Duh, Helena Gomez, and Steven Bethard (Eds.). Association for
Computational Linguistics, Mexico City, Mexico, 3730–3740. doi:10.18653/v1/
2024.findings-naacl.236
[21] Wei Liu, Sony Trenous, Leonardo F. R. Ribeiro, Bill Byrne, and Felix Hieber. 2025.
XRAG: Cross-lingual Retrieval-Augmented Generation. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2025. Association for Computational
Linguistics, Suzhou, China, 15669–15690. https://aclanthology.org/2025.findings-
emnlp.849/
[22] Sean MacAvaney and Luca Soldaini. 2023. One-shot labeling for automatic rele-
vance estimation. InProceedings of the 46th International ACM SIGIR Conference
on Research and Development in Information Retrieval. 2230–2235.
[23] James Mayfield, Eugene Yang, Dawn Lawrie, Sean MacAvaney, Paul McNamee,
Douglas W. Oard, Luca Soldaini, Ian Soboroff, Orion Weller, Efsun Kayi, Kate
Sanders, Marc Mason, and Noah Hibbler. 2024. On the Evaluation of Machine-
Generated Reports. InProceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval(Washington DC, USA)(SI-
GIR ’24). Association for Computing Machinery, New York, NY, USA, 1904–1915.
doi:10.1145/3626772.3657846
[24] Ani Nenkova, Rebecca Passonneau, and Kathleen McKeown. 2007. The pyramid
method: Incorporating human content selection variation in summarization
evaluation.ACM Transactions on Speech and Language Processing (TSLP)4, 2
(2007), 4–es.
[25] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick
Craswell, and Jimmy Lin. 2024. Initial nugget evaluation results for the trec 2024
rag track with the autonuggetizer framework.arXiv preprint arXiv:2411.09607
(2024).
[26] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick
Craswell, Ian Soboroff, Hoa Trang Dang, and Jimmy Lin. 2025. The great nugget
recall: Automating fact extraction and rag evaluation with large language models.
InProceedings of the 48th International ACM SIGIR Conference on Research and
Development in Information Retrieval. 180–190.
[27] Shahzad K Rajput, V Pavlu, and Javed A Aslam. 2011. A nugget-based evaluation
paradigm to address the scalability and reusability issues of information retrieval
test collections. InProceedings of the 34th international ACM SIGIR conference on
Research and development in Information Retrieval. ACM, 1003–1012.
[28] Corby Rosset, Ho-Lam Chung, Guanghui Qin, Ethan C Chau, Zhuo Feng, Ahmed
Awadallah, Jennifer Neville, and Nikhil Rao. 2024. Researchy questions: A dataset
of multi-perspective, decompositional questions for llm web agents.arXiv
preprint arXiv:2402.17896(2024).
[29] Yuichi Sasazawa, Sho Takase, and Naoaki Okazaki. 2019. Neural Question
Generation using Interrogative Phrases. InProceedings of the 12th International
Conference on Natural Language Generation, Kees van Deemter, Chenghua Lin,
and Hiroya Takamura (Eds.). Association for Computational Linguistics, Tokyo,
Japan, 106–111. doi:10.18653/v1/W19-8613
[30] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Daniel Campos, Nick
Craswell, Ian Soboroff, Hoa Trang Dang, and Jimmy Lin. 2024. A large-scale
study of relevance assessments with large language models: An initial look.arXiv
preprint arXiv:2411.08275(2024).[31] Shivani Upadhyay, Ronak Pradeep, Nandan Thakur, Nick Craswell, and Jimmy
Lin. 2024. UMBRELA: UMbrela is the (Open-Source Reproduction of the) Bing
RELevance Assessor.arXiv preprint arXiv:2406.06519(2024).
[32] Ellen M. Voorhees. 2003. Evaluating Answers to Definition Questions. InCom-
panion Volume of the Proceedings of HLT-NAACL 2003 - Short Papers. 109–111.
https://aclanthology.org/N03-2037/
[33] Ellen M Voorhees and Hoa Trang Dang. 2003. Overview of the TREC 2003
question answering track.. InTrec, Vol. 2003. 54–68.
[34] William Walden, Marc Mason, Orion Weller, Laura Dietz, John Conroy, Neil
Molino, Hannah Recknor, Bryan Li, Gabrielle Kaili-May Liu, Yu Hou, et al .
2025. Auto-argue: Llm-based report generation evaluation.arXiv preprint
arXiv:2509.26184(2025).
[35] Eugene Yang, Dawn Lawrie, James Mayfield, Douglas W Oard, and Scott Miller.
2024. Translate-distill: learning cross-language dense retrieval by translation and
distillation. InEuropean Conference on Information Retrieval. Springer, 50–65.
[36] Zhuohao Yu, Jiali Zeng, Weizheng Gu, Yidong Wang, Jindong Wang, Fan-
dong Meng, Jie Zhou, Yue Zhang, Shikun Zhang, and Wei Ye. 2025. Rewar-
dAnything: Generalizable Principle-Following Reward Models.arXiv preprint
arXiv:2506.03637(2025).
[37] Chao Zhou, Cheng Qiu, Lizhen Liang, and Daniel E Acuna. 2025. Paraphrase
identification with deep learning: A review of datasets and methods.IEEE Access
(2025).