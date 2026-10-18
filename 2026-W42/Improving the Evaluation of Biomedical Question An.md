# Improving the Evaluation of Biomedical Question Answering with Few Human Preference Signals

**Authors**: Shakila Mahjabin Tonni, Xiang Dai, Sarvnaz Karimi, Xiaomei Li, Samuel C. Andrew, Stephen Wan

**Published**: 2026-10-12 00:17:35

**PDF URL**: [https://arxiv.org/pdf/2610.16008v1](https://arxiv.org/pdf/2610.16008v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems are increasingly used for biomedical Question Answering (QA). However, evaluating their effectiveness remains challenging due to the lack of reliable reference resources and the frequent misalignment between automatic evaluation metrics and human judgments. One practical question is how evaluation results should be used to guide development decisions, especially when they correlate weakly with human judgments and may even lead to different conclusions. We explore how to use a small amount of human preference signals to calibrate direct assessment evaluation metrics, and show that calibrated metrics consistently outperform baselines that either treat all metrics equally or rely on a single best metric across multiple datasets.

## Full Text


<!-- PDF content starts -->

Improving the Evaluation of Biomedical Question Answering with Few
Human Preference Signals
Shakila Mahjabin Tonni1∗Xiang Dai2Sarvnaz Karimi3
Xiaomei Li4Samuel C. Andrew2Stephen Wan2
1Suncorp Bank, Australia2CSIRO, Australia
3Monash University, Australia4Commonwealth Bank, Australia
∗shakila.tonni@suncorp.com.au
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems are increasingly used for biomedical Ques-
tion Answering (QA). However, evaluating
their effectiveness remains challenging due to
the lack of reliable reference resources and the
frequent misalignment between automatic eval-
uation metrics and human judgments. One prac-
tical question is how evaluation results should
be used to guide development decisions, es-
pecially when they correlate weakly with hu-
man judgments and may even lead to different
conclusions. We explore how to use a small
amount of human preference signals to cali-
brate direct assessment evaluation metrics, and
show that calibrated metrics consistently out-
perform baselines that either treat all metrics
equally or rely on a single best metric across
multiple datasets.
1 Introduction
Large Language Models (LLMs) (Luo et al., 2022;
Bolton et al., 2024; Labrak et al., 2024; Kafkas
et al., 2025) have shown promise in answering
biomedical questions. Interest in Question Answer-
ing (QA) applications using Retrieval-Augmented
Generation (RAG) techniques has also been on the
rise (Lyu et al., 2024; Jin et al., 2024; Minbyul
et al., 2024; Wang et al., 2024, for example). As a
result, a recent direction has been to examine the
use of RAG systems for discovery-oriented scien-
tific questions that require long-form responses. A
key problem, however, is the evaluation of their
effectiveness, which remains underexplored.
Consider the question“What proportion of Tran-
scriptomic studies on plants found ICE1 Transcrip-
tion factors to be associated with cold acclima-
tion?”as an example. Acquiring ground-truth an-
swers for comparison is often infeasible due to
the high cost of knowledge acquisition. In the
absence of ground truth references, metric-based
approaches to evaluation are a logical option. How-
ever, a naive use of metrics is not straightforward.When applying existing reference-free direct as-
sessment evaluation metrics to evaluate domain-
specific QA, e.g., measuring response faithfulness
or entailment with relevant resources (Schimanski
et al., 2024), previous studies have reported incon-
sistent findings regarding the effectiveness of indi-
vidual metrics (Roychowdhury et al., 2024; Oliva
et al., 2025). Furthermore, we highlight that met-
rics often lead to inconclusive assessments (§5.1)
when applied to evaluate individual components of
a RAG-based QA system. This raises the question
of how to interpret such evaluation results to guide
development decisions (e.g., should different met-
rics be weighted equally and combined, or should
a single best metric be identified?).
Beyond problems with the application of metrics,
Bavaresco et al. (2025) showed that domain experts
apply stricter domain-specific criteria than auto-
mated metrics when evaluating LLM responses.
Additionally, direct assessment itself is challeng-
ing for experts compared to pairwise comparison or
reference-based assessment (Novikova et al., 2018).
To this end, we work with biologists to collect
human preferences for long-form answers from a
RAG-based QA system and investigate how to cali-
brate the weights of direct assessment metrics from
collected preferences. We find that a small amount
of human preference signals is highly effective for
calibrating weights, making the calibrated metrics
substantially better than baselines with uniform or
random weighting. In contrast, calibration based
on LLM preferences, used as a low-cost proxy for
human preferences, did not significantly improve
alignment with human judgments.
Our contributions are: (1) a preference-guided
framework for calibrating weights of direct assess-
ment evaluation metrics using correlation analy-
sis and logistic regression; (2) two human prefer-
ence datasets FROSTTOLGENEand OZWHEAT-
GENE, for RAG-based QA in plant and agricul-
tural genomics, experiments on which (and an ex-
arXiv:2610.16008v1  [cs.CL]  12 Oct 2026

Figure 1: An overview of our RAG-based QA system.
isting SCIARENAbenchmark) show that calibrated
metrics align better with human judgments than
uniform-weight or single-metric baselines; and (3)
an exploration of LLM-based pairwise preference
collection, as a cheap proxy, using generic and
domain-specific rubric-based judges.
2 Problem Statement
We consider a 3-stage RAG-based system: (1)
query formulation from user question; (2) retrieval
and corresponding creation of content surrogates
(i.e., summarizing retrieved documents) for inclu-
sion in the prompt for answer generation; and (3)
LLM answer generation. In an applied setting, de-
cisions are needed about which variation is most
effective for the proposed use case. Such deci-
sions relate to the choice of LLM for each compo-
nent or to algorithmic variations in any of the three
above-mentioned steps.
Building on the RAGAS evaluation framework
proposed by Es et al. (2024), we consider the sce-
nario where ground -truth answers are unavailable,
yet a practical evaluation approach is still needed
to inform design decisions for the RAG -based QA
system. Here, we first collect a pool of answers
for a given question by running different variations
of a QA system. Then, using existing automatic
evaluation metrics, we rank the candidate answers
and select a few candidates for human judgment.
Finally, the collected human judgments and feed-
back are used to build better evaluation metrics and
to inform design decisions for the QA system.
3 RAG-based QA System
To enable our study, we first built an in -house RAG
QA system, GENERAG, since the existing biomed-
ical QA systems are either not publicly available
or focused solely on generating short answers. Fig-
ure 1 shows the pipeline of our system and design
options for each stage.Query Formulation.Our focus user group, con-
sisting of plant genomics researchers, typically re-
lies on NCBI’s Entrez system1to search databases
(e.g., PubMed) for relevant scientific information.
To search via the Entrez API, we first convert a user
question to search terms. To do this, we prompt
LLMs with few -shot examples to: (1) generate a
single query term for a given question (this setting
is denoted asSQ); and, (2) create sub-questions
from the given question and then generate a query
term for each sub-question (denoted asMQ).
Context Retrieval.We retrieve relevant abstracts
from PubMed focusing only on the open-access
subset. A brief description of the API is provided
in the App. §A. In theMQsetting, where multi-
ple query terms are generated, two search variants
are implemented in GENERAG: (1) concatenat-
ing all query terms into a single query string us-
ing the ORoperator (MQ +Cat ); and (2) retrieving
abstracts separately for each query term and con-
catenating them (MQ +Stack ). After retrieving the
abstracts, we follow PaperQA2 (Skarlinski et al.,
2024) pipeline, usingGPT4o-minito chunk, filter,
re-rank, and summarize them. The final context
is a set of summarized relevant excerpts from the
abstracts.
Answer Generation.Given the user’s question
and the context from the previous step, the system
prompts an LLM to generate the final answer.
Models Selection.In theSQsetting, we use four
LLMs:GPT3.5-turbo,GPT4o-mini,GEMMA2-9B,
andMistral-Nemo-12B. In theMQsetting, we use
GPT4o-miniandMistral-Nemo-12B. For answer
generation, we useGPT4o-miniandQWEN-3-8B.
We evaluated several domain-specific (BioMistral-
7B, MedLLAMA-7B, PMC-LLAMA-7B and
BioGPT-Large) and generic LLMs (QWEN2.5-
32B and Claude-3-Sonnet) on a question subset,
selecting models via a quality control inspection.
4 Datasets
In collaboration with two plant genomics scien-
tists, we compiled two new datasets. The scientists
selected candidate wheat genes identified through
their own genomic annotation research and posed
questions regarding the association of these genes
with specific traits. The resulting questions are
divided into two sets (examples in App. Table 11):
1https://www.ncbi.nlm.nih.gov/search/

•FROSTTOLGENE, a set of 42 questions about
the relevance of genes to frost tolerance
and cold acclimation, using genes identified
through a literature search by the domain ex-
perts.
•OZWHEATGENE, a set of 23 questions from
genome-wide association analysis that iden-
tified genes linked to crop yield and compo-
nent traits for crop yield for the genes from
Ozwheatdiversity panel (Hyles et al., 2024).
These questions are related to multiple genes
and physical traits of wheat.
4.1 Collecting Human Preferences
We collected expert preferences over pairs of an-
swers from the FROSTTOLGENEand OZWHEAT-
GENEdatasets through an internally developed sur-
vey application (details in App. §H). For each ques-
tion, we generated a pool of 12 answers using dif-
ferent variants (i.e., different combinations of RAG
system parameters) using our QA system.
In cases where the answer space is combinatorial,
it is common to sample and choose candidate LLM
answers for human annotations, given the complex-
ity and time requirement. For example, Feng et al.
(2025) selected the top 10 candidates via maximum-
discrepancy competition and WildBench (Lin et al.,
2025) discarded highly similar chats from evalua-
tion. Furthermore, Pan et al. (2026) showed that in
the Direct Preference Optimization (DPO) context,
the human preference data should cover both good
and bad responses. Considering the substantial hu-
man annotation effort required to judge pairwise
answers with large retrieved contexts, we also sam-
pled candidate answers from the answer pool (66
unique combinations of answer pairs per question).
Given our focus on calibrating metric weights for
only a specific set of metrics, we used the aggre-
gated evaluation scores as the basis for selecting
three candidate answers per question, correspond-
ing to the highest-, moderate-, and lowest-quality
answers.
We adopted a tournament-style sampling strat-
egy, where the winner from the first random pair
moves forward and is paired with the remaining
third answer, as we found when comparing annota-
tions on ‘all’ pairs and fewer pairs in tournament-
style setup (paired randomly), they both achieve
close inter-annotator agreements Cohen’s κof 0.68
and 0.64, respectively. Critically, the annotators
were not aware of the quality ranks of the answers.During the annotation process, the annotators could
only see the generated contexts and the answers,
and did not have access to the formulated query
or the retrieved set of abstracts. In addition to se-
lecting their preferred answer, the annotators were
instructed to provide a short explanation of the
problems they identified in the unpreferred answer
(referred to later as thehuman reasonings).
Of the 65 questions across both datasets, two
questions (one from each dataset) were reserved
for annotator training and excluded from analy-
sis. For the remaining 63 questions, we collected
annotations on answer pairs, resulting in 166 pair-
wise comparisons. Three domain expert annota-
tors annotated ten questions, and the remaining 53
were single-annotated. For the multi-annotations,
we further considered the mode over the multi-
annotations and concatenated the reasons. The final
data set totalled 130 unique annotations.
4.2 Preferences by LLM-as-a-Judge
To compensate for the high cost of human expert
annotations, which invariably limits the number
of collected human annotations, we developed a
rubric-based LLM-as-a-judge baseline (Hashemi
et al., 2024; Lin et al., 2025; Zhao et al., 2025) and
constructed a synthetic dataset containing LLM
preferences over all pairwise answers. Following
Lin et al. (2025), we first created a set of generic
rubric questions and a set of domain-specific rubric
questions in consultation with domain experts. We
then prompted aGPT-4o-minimodel to give pref-
erences over pairwise answers using the rubric as
a guideline. We first collected LLM preferences
over the human-annotated subset under four set-
tings: LLMDIRECT—direct LLM preference with-
out rubrics; LLMGENERIC—rubrics containing
only generic questions; LLMDOMAIN—rubrics
containing only domain-specific questions; and
LLMALL—rubrics containing both generic and
domain-specific questions. The prompts are in-
cluded in App. §E. After generating the prefer-
ences, we measured human–LLM agreement using
Cohen’s κand correlation using Pearson’s r. As
shown in Table 2, the LLMDOMAINachieved the
highest agreement and correlation with human pref-
erences. Therefore, we used the LLMDOMAIN
prompts to collect 4,290 LLM preferences over all
answer pairs for each question.

Quality Metric Evaluation
Dimension Dimension
Semantic Similarity ROUGE-1 (Lin, 2004),Qr2Q,
BERTScore (Zhang et al., 2020),C2Ab,A2C
CosSim. (Reimers and Gurevych, 2019)
Relevance Context Relevance (Es et al., 2024)C2Q
(LLM-based) Answer Relevance (Es et al., 2024)A2Q
Factual consistency FactCC (Kryscinski et al., 2020)C2Ab,A2C
AlignScore (Zha et al., 2023)Qr2Q,C2Q,A2Q,
C2Ab,A2C
Faithfulness HHEM-2.1-open (Es et al., 2024)C2Ab,A2C
Table 1: Evaluation metrics selected for different evalua-
tion dimensions. Qr2Q =query-to-question, C2Q =context-
to-question, A2Q =answer-to-question, C2Ab =context-to-
abstract andA2C=Answer-to-Context.
Prompt Strategy Cohen’sκPearson’sr
LLMDIRECT0.46 0.409
LLMDOMAIN0.55 0.530
LLMGENERIC0.50 0.486
LLMALL0.53 0.515
Table 2: Agreement between human vs. LLM preferences in
four LLM-as-a-Judge settings (p-val<.001).
4.3 Inter-annotator Agreements
We computed agreement on the common-answer
pairs shared across human annotators. The overall
agreement level is moderate with κ=0.68. The
agreements between the annotator pairs are 0.53,
0.66 and 0.72, indicating that the annotators have
higher variability in individual alignment with the
remaining annotators. This also highlights the chal-
lenge of conducting human evaluation in highly
specialized domains, since domain experts usu-
ally have slightly different background knowledge,
which can strongly influence their judgments.
5 Automatic Evaluation vs. Preferences
Drawing from existing evaluation frameworks in
the literature, we first identified a set of quality
dimensions and corresponding metrics for the eval-
uation of RAG-based systems. Beyond assessing
only the quality of final answers, following Es et al.
(2024), we used these metrics to assess individual
components of the GENERAG pipeline.
5.1 Metrics and Dimensions
We applied multiple metrics across five evaluation
dimensions. Specifically, we assessed the quality
of the formulated query, the retrieved context, and
the final answers to the user questions, which we
refer to as in the Qr2Q ,A2Q andC2Q evalua-
tion, respectively. Additionally, in A2C evalua-
tion, the generated answers were checked againstSetting Query AnsQr2Q C2Q C2Ab A2Q A2C
LLM LLM Avg. Avg. Avg. Avg. Avg.
SQGPT3.5 GPT4 0.606 0.154 0.465 0.793 0.408
GPT3.5 QWEN 0.8130.497
GPT4 GPT4 0.596 0.175 0.486 0.789 0.414
GPT4 QWEN0.8230.484
GEMMA GPT4 0.596 0.183 0.471 0.774 0.422
GEMMA QWEN 0.804 0.478
Mistral GPT4 0.567 0.220 0.481 0.792 0.391
Mistral QWEN 0.811 0.473
MQ +Cat GPT4 GPT40.6070.2640.5210.788 0.392
GPT4 QWEN 0.779 0.399
Mistral GPT4 0.564 0.268 0.517 0.807 0.421
Mistral QWEN 0.785 0.433
MQ +Stack GPT4 GPT4 0.5790.3130.511 0.804 0.399
GPT4 QWEN 0.748 0.423
Mistral GPT4 0.578 0.312 0.511 0.812 0.399
Mistral QWEN 0.743 0.422
Table 3: Metric scores for FROSTTOLGENEquery, context
and answer evaluation in three settings:SQ,MQ +Cat and
MQ+Stack (Bestbolded, and second-best underlined ).
the context, and in C2Ab evaluation, the gener-
ated contexts by PaperQA2 were evaluated against
the retrieved abstracts. We selected the semantic
similarity, relevance, factual consistency and faith-
fulness quality dimensions and adopted suitable
metrics for each dimension. The metric list is pre-
sented in Table 1. Altogether, we considered 18
evaluation metrics and applied min-max scaling to
bring the scores within a range between 0 and 1.
(Details in App. §C.)
From the metric scores, we found that none of
the variantsaloneis consistently performing well
across different evaluation dimensions, and it is not
possible to conclusively decide which LLMs and
design options to use in the various stages of the
RAG system based on them. For example, sum-
marized results on FROSTTOLGENEin Table 3
shows, the best context was obtained by GPT4
in theMQ +Stack setting for C2Q , but for C2Ab ,
MQ +Cat setting obtained the highest results. More
results and discussion are in the App. §D. This
raised our next question: if these metrics cannot
reach consensus, how can we use them to guide
design decisions for the QA system?
5.2 Metric vs. Preferences
We started by analysing the alignment of the hu-
man preferences and aggregated metric scores on
FROSTTOLGENEand OZWHEATGENEdatasets.
We converted the scores and the human into binary
preference labels. Formally, for each question q
and answer pair (a, b) , a label of 0 is assigned if
annotators prefer a, and 1 if they prefer b. Metric-

based preferences are derived analogously:
Pref(a, b|S) =(
0,ifS(a)≥ S(b),
1,otherwise.(1)
where uniform weight S(ai) =1
nPn
j=1Mj(ai)
calculated over Mj∈ M metrics. Using these la-
bels, we computed the rate of cases when human
annotators selected metric-based preferred answers.
All metrics were considered equally, equivalent to
anuniform-weighting. The resulting agreement
percentages and Pearson’s correlation coefficients
r, along with the statistical significance reported
in App. Table 17. Overall, experts agreed with the
metric-based preferences in 61.45% of the cases
(Kappa κ= 0.30). We also measured a moderate
but significant Pearson correlation coefficient, r
= 0.300 (p < 0.01) between human-metric prefer-
ences.
Further, between human and individual metrics-
based preferences, we found that the agreement
rates vary substantially across metrics. We
found A2C -AlignScorehas the highest human-
metric preference agreement of 65.75%. Addition-
ally,A2Q -AlignScoreachieved 61.64% agreement,
C2Ab -FactCC61.45% and A2C -Faithfulness
achieved 61.64% human-metric preference agree-
ments.
In light of the findings, we infer that not all au-
tomatic metrics correlate uniformly with human
preferences, and it is not possible to have a conclu-
sive decision on the metrics to choose. Thus, in the
next section, we explore ways to incorporate hu-
man preferences into metric reliability calibration.
6 Preference-Guided Weight Calibration
Before applying or aggregating evaluation metrics,
we want to determine which metrics align better
with human preferences. The human–metric pref-
erence agreement rates computed using uniformly
weighted metrics in the previous section allow us
to identify which metrics contribute more strongly
to alignment with human judgments. These agree-
ments can be considered as a proxy for human
preference alignments.
Based on this, we propose a preference-guided
approach for calibrating metric reliability weights
using human preferences. Formally, let the agree-
ment rate between human and metric preferences
for metric Mjbe denoted as the calibration weight
wj. Using these weights, the adjusted aggregated
score for an answer aiis computed as Sadj(ai) =1
nPn
j=1wj×M j(ai). The corresponding pref-
erence labels, Pref(a, b| S adj), are then ob-
tained using Eq. 1. We consider two approaches
to calibrate the metric weights—directly using
human–metric preference agreement as weights
(DMW), and learning the weights using a logistic
regression model (LMW).
Direct Metric Weighting (DMW)We consid-
ered human-metric preference agreements directly
as the metric weights. By dividing the dataset into
a calibration and validation set and computing hu-
man–metric preference agreement scores for each
metric on the calibration set, we obtained the cal-
ibrated weights for the metrics. We then applied
these weights to the validation set for aggregat-
ing the metric scores and obtaining the weighted-
metric preferences, Metdir+hum . Finally, we mea-
sured the human- Metdir+hum agreement rate and
compared it with the agreement rates between hu-
man vs. uniform- and random-weighted metric
preferences, Metunif and Metrand, respectively.
Similarly, we calibrated weights using LLM prefer-
ences and obtained metric preferences Metdir+llm
over the human validation sets and measured the
agreement rates against human preferences.
Learned Metric Weighting (LMW)To learn
the metric weights from the human preferences, we
built separate logistic regression models for each
metric. For each metric, we prepared the dataset
with pairwise high and low metric scores (implying
metric-based preference) for each answer pair, and
human preference [0,1] as the label. The model’s
prediction task was essentially a binary classifica-
tion task to predict human preference for a pair of
high-low metric scores. We then considered the
training accuracy of the model as the metric weight.
The learned metric weights were then applied to
the human validation subsets to obtain Metlog+hum
and validated against the human preferences.
6.1 Experimental Setup
We randomly split the human-annotated dataset
into a 60%-40% calibration–validation split, yield-
ing 78 calibration and 52 validation instances per
set, repeating this process 100 times to report the
mean and standard deviation of the agreement
rates of the human-metric preferences. For ex-
periments involving LLM-generated preferences,
we randomly sampled a larger calibration set by
taking 80% of the annotations to derive the cal-
ibrated weights for k= 100 iterations in both

No Calibration
Uniform weighting 61.44±5.95
Random weighting 55.42±7.21
W/ Calibration
DMW LMW
Human pref. 64.30±4.34 70.00±4.05
LLM pref. 53.17±2.45 62.95±4.10
Table 4: Agreement rate (%) and std. dev. (CI=95%) between
human and metric-based preferences, computed via Direct
Metric Weighting (DMW) and Learned Metric Weighting
(LMW), and compared against random and uniform weights
on the validation set.
DMW and LMW settings. For DMW, we av-
eraged the 100 calibration weights to generate a
single set of weights. Validation was done on the
100 human-annotated validation folds described
above. We also considered two baselines: uniform
weighting and random weighting, where all met-
rics are assigned equal weights or random weights,
respectively.
6.2 Results and Discussion
Calibration with Human Preferences.Human
preference-based calibration leads to a significant
improvement in alignment between human and
metric preferences on the validation set, with
an increase in the average agreement rate from
61.44% with uniform metric-weights to 64.30%
with DMW and 70.00% with LMW methods (Ta-
ble 4). This result is also significantly higher than
therandom-weightsbaseline with an agreement
level of 55.42%.
We observed a moderate Pearson’s correlation
coefficient r= 0.315 forMetdir+hum -human and
r= 0.408 forMetlog+hum -human preference la-
bels, compared to a lower correlation of r= 0.281
forMetunif-human labels across all kvalidation
samples (for all, p<0.0001). Furthermore, we found
that the improvement in the human-metric agree-
ment rates is also statistically significant. With
a one-tailed (Group A > Group B) Wilcoxon’s
signed-rank test (Riffenburgh, 2006), we measure
the agreement rates using DMW vs. uniform-
weighting over ksamples and observed a statis-
tically significant improvement z= 2.865 (p<0.01,
significant at α= 0.005 level) and which is for
LMW, z= 0.178 (p<0.01, significant at α= 0.05
level), implying a positive significance with a
tighter bound of standard deviations.
Comparison of Best Metrics and Subsets.To
decide on which evaluation dimensions were mostEval. Dimension Uniform DMW LMW
All 61.44 64.30 70.00
A2C-AlignScore (best)66.03
Qr2Q50.9252.88 52.17
C2Q52.8452.85 51.88
C2Ab45.07 48.54 47.64
C2Q+C2Ab48.57 49.57 48.66
A2Q48.19 50.80 46.58
A2C61.15 62.03 58.85
A2Q+A2C58.54 60.35 61.55
Table 5: Human-metric preference agreements(%) for calibra-
tion using subsets of metrics compared to ‘All’ metrics. The
top 4 best-performing subsets are bolded for each setting.
effective in weight calibration, we use Qr2Q ,
C2Ab ,C2Q +C2Ab ,A2Q +A2C subsets sepa-
rately and the best metric, A2C -AlignScore (based
on App. Table 17) to calibrate weights. The re-
sults are reported in Table 5. Under the DMW
setting, the single best metric achieved 64.19%,
which was close to the 64.30% obtained when us-
ing all metrics, implying that under direct metric
calibration, considering one best metric is suffi-
cient. In contrast, under the LMW setting, neither
the best-performing metric nor any individual sub-
set of metrics achieved a score close to the 70%
obtained using metrics from all dimensions. Fur-
thermore, the Qr2Q ,A2Q , and A2C metrics ap-
pear to contribute the most toward learning the
metric weights.
Calibration with LLM-preferences.We found
that, in the absence of human-annotated data, LLM-
generated preferences can also be used to calibrate
metric weights. As shown in Table 4, the DMW
setting achieves only 53.17% agreement with hu-
man judgments, which is lower than both uniform
and random-weighted baselines. In contrast, the
LMW setting improves this agreement to 62.95%.
Although calibration using LLM preferences does
not reach the 70% agreement achieved with human-
preference-based calibration, it provides a better
alternative to uniform metric aggregation when
human annotations are unavailable. Across all k
samples, we also observe a Pearson correlation of
r= 0.293 (p < 0.001) between human- Metdir+llm
labels, and r= 0.396 (p < 0.001) between hu-
man and Metlog+llm labels, further indicating that
LLM-preference in LMW is relatively closer to
human-preference-based calibration.
Evaluation on the SCIARENABenchmark.To
test the generalizability of the calibration tech-
niques on an out-of-domain dataset, we conducted

Unif. Weighting DMW LMW
Combined 53.47 59.8260.51
Biomed. 52.1460.1759.82
Medicine 53.9158.7758.75
Neuroscience 54.00 61.0962.18
Pharmacy 52.42 60.0560.57
Public Health 51.77 59.0061.23
Table 6: SCIARENAhuman-metric preference agreements(%).
experiments on SCIARENA(Zhao et al., 2025),
which is a benchmarking dataset with annotations
collected from the researcher community over pair-
wise LLM-generated responses on scientific liter-
ature tasks. We considered‘Biomedical Engineer-
ing’,‘Medicine’,‘Neuroscience’,‘Pharmacy’and
‘Public Health’subsets, as the domains were close
to our focus. Details of the data size of each sub-
set are reported in the App. §G, and an example
QA pair is in Table 25. We applied both DMW
and LMW techniques on each subset separately
and combined. Results are in Table 6. We observe
that in all cases human-metric agreement rates im-
proved from the uniform weighting, and the best
agreement rate of 60.51%, on the combined set,
was achieved in the LMW setting. Note, in the
SciArena dataset, only answers and context (cita-
tions) are available, and we used only 9 metrics
covering C2Q ,A2Q andA2C . As shown in §6.2,
the LMW setting performed better with a larger
set of metrics, particularly with Qr2Q ,A2Q , and
A2C , which might be a possible reason for the
improvement in agreement from DMW to LMW.
Quality Dimensions from Expert Reasons.To
understand which quality aspects the experts pri-
oritize, we prompted aGPT4o-minimodel with
a one-shot example to group the human rationals
and assign interpretable labels (referred to ascrite-
ria). In case multiple reasons were present in one
rationale, we split and consider them separately.
Using another prompt by adding these dimensions
in-prompt, we asked the LLM to assign a label
to each of the human reasons, which were further
corrected through manual checking. The prompt
and resulting criteria are in App. §F. We found that
the main factors influencing human judgments are
answer and context relevance, lack of detail in the
answer and incorrect or irrelevant context (App. Ta-
ble 22). This validates our contextual appraisal of
quality based on a set of component-wise metrics
for RAG systems.
We also observed 35 cases where annotatorsnoted that the answers were very similar, resem-
bling a ‘tie’ between two answers and that both
answers are good/bad (Chiang et al., 2024). Two
such examples are in App. Table 23.
In addition, human experts also considered an-
swer clarity, fluency, organization, and structure.
This finding resonates with prior work that incor-
porates stylistic qualities (D’Souza et al., 2025)
and answer usefulness (Malaviya et al., 2024) in
LLM-based evaluation.
We further analyzed cases where metric-based
preferences differ from human judgments (App. Ta-
ble 21), highlighting the metrics’ inability to cover
certain evaluation criteria. For example, when only
C2Ab metrics are used to derive metric preferences,
they disagree with 30 human preferences in cases
where humans based their judgments on ‘Relevance
of Answer’. The lowest amount of mismatching
was obtained when all metrics were considered to
generate preferences.
Finally, we observed that the qualitative aspects
of the answers can be a matter of subjective judg-
ments, varying across expertise levels. App. Ta-
ble 24 gives an example of such an annotation,
where both annotators considered different ratio-
nales for the same preference. Therefore, effec-
tively selecting an evaluation strategy for RAG
systems requires incorporating metrics from all
components of the RAG pipeline and human-in-
the-loop evaluation strategies.
Practicality and Overall Recommendation. In
our work, we aim to improve the alignment be-
tween automatic evaluation metrics and human
judgments. In other words, we aim to identify
a better and more cost-effective proxy for human
judgments to facilitate downstream RAG system
development. Our main goal is to show that, even
with only a small number of human annotations,
they can be used to effectively calibrate the met-
ric weights. The calibrated metric reaches a per-
formance level close to human–human agreement,
suggesting that this is a practical upper bound given
the variability in human judgments. We also ob-
serve that, although, LMW is better-performing in
all cases, DMW provides only modest improve-
ment and, as we reported in the Table 5, per-
forms worse than the single best metric. This sug-
gests that metric weight calibration is not always
needed. Therefore, we recommend comparing the
calibrated metric with the single best metric and
using whichever achieves higher agreement with

human preferences.
7 Related Work
We first explored the limitations of different QA
datasets and language models in the biomedical
domain. Then, we discussed challenges with the
existing automated metrics for RAG-based QA sys-
tems and how human preferences are integrated to
improve LLM-based evaluations.
Biomedical QA Datasets.Several general and
specialized datasets support biomedical QA.
MedQA (Jin et al., 2021) and MedRedQA (Nguyen
et al., 2023) targeted general medical knowledge,
whereas GeneTuring (Shang et al., 2025) intro-
duced gene-centric questions, PubMedQA (Jin
et al., 2019) and BIOASQ-QA (Krithara et al.,
2023) focused on literature-grounded questions.
More recently, ExpertQA (Malaviya et al., 2024) in-
troduced expert-curated questions across multiple
domains. However, these datasets are not suitable
for evaluating long-form, open-ended responses.
Our study introduced qualitative aspects– such as
relevance and alignment with the question and sup-
porting context–through calibrated metrics by hu-
man preferences.
LLMs for Answering Domain-specific Ques-
tions.LLMs have been pretrained or fine-tuned
on biomedical corpora to support domain-specific
QA (Bolton et al., 2024; Luo et al., 2022; Wu et al.,
2024; Labrak et al., 2024; Yasunaga et al., 2022).
Some models focused on gene-related tasks, such
as Geneverse (Liu et al., 2024) and GP-GPT (Lyu
et al., 2024). In parallel, RAG has been explored
for biomedical QA using literature retrieval (Skar-
linski et al., 2024; Patel et al., 2026), retrieval-
based gap-analysis and knowledge assertion (Ning
et al., 2025) or combining databases with external
tools (Wang et al., 2024). However, such systems
rely on restricted resources, limiting public avail-
ability and reproducibility.
Evaluation of RAG QA systems.Evaluating
RAG-based QA models remain challenging due
to the lack of reliable reference answers. RAGAS
(Es et al., 2024) proposed a reference-free evalua-
tion framework judging faithfulness, relevance, and
groundedness. Adlakha et al. (2024) evaluated cor-
rectness and faithfulness in instruction-following
QA and Zha et al. (2023) evaluated factual consis-
tency via AlignScore. Benchmarks such as LitQA2
(Skarlinski et al., 2024) and LabBench (Laurentet al., 2024) assessed QA accuracy and biologi-
cal reasoning skills, while ExpertQA (Malaviya
et al., 2024) evaluated qualitative dimensions in-
cluding usefulness and cite-worthiness. However,
guidance on metric selection for domain-specific
QA remains limited.
Human-preferences in LLM Evaluators.
LLMs were used as automated evaluators to judge
AI-generated responses (Li et al., 2025; Hijazi
et al., 2024; D’Souza et al., 2025). However,
there are significant gaps between coverage of
human-centred criteria by LLMs (Miller and Tang,
2025) or inconsistent correlations with human
judgment (Oliva et al., 2025). Consequently,
several approaches now integrate human feedback
into LLM-evaluators for automated evaluation of
reasoning chains (Jacovi et al., 2024) and to align
automated evaluation with human preferences
(Yue et al., 2025; Polo et al., 2025; Ye et al., 2025).
8 Conclusions
Evaluating the effectiveness of RAG-based QA sys-
tems for generating long-form answers to biomedi-
cal questions remains a challenging task, regardless
of the availability of several automatic evaluation
metrics. Our analysis revealed substantial variabil-
ity across metrics when evaluating different com-
ponents of a RAG system. Such variability makes
it difficult to interpret results, select appropriate
metrics, or identify a single reliable metric for eval-
uation. Furthermore, we found that no individual
metric consistently aligns with expert judgments.
To address this issue, we considered the hu-
man–metric preference agreement rates as indi-
cators of each metric’s contribution toward hu-
man alignment and proposed calibrating metric
weights before applying or aggregating them. Our
results showed that, without calibration, relying
on the single best-performing metric can achieve
performance comparable to direct metric weight-
ing (DMW). However, the strongest alignment
with human preferences was achieved when met-
rics across all evaluation dimensions—query, con-
text, and answer—were considered and the weights
were learned using the LMW approach.
We further conducted similar weight calibra-
tion using LLM preferences collected through a
rubric-based LLM-as-a-judge baseline. The re-
sults showed that only the learning-based LMW
approach improved human alignment of the metrics
over uniform weighting. However, the performance

remained well below the human-preference-based
calibration.
Additionally, to conduct the weight calibrations,
we create two expert-curated plant genomics QA
datasets that address the scarcity of non-human ge-
netics QA benchmarks, and extended the analysis
to the SCIARENAbenchmark dataset.
Future work includes extending this analysis
to other domains or task types to test the gener-
ality of preference-guided metric calibration and
adapting other learning techniques for improved
metric weight calibration. In addition, integrating
preference-guided metrics into training objectives
for RAG-based QA models—through reinforce-
ment learning or self-refinement—could further
improve alignment with expert human judgments.
9 Limitations
As a proof of concept, we propose using a small
amount of human preference signals to calibrate
existing direct assessment metrics. We select repre-
sentative metrics that aim to cover different quality
dimensions. We do not claim that our choice of
metrics is exhaustive. Moreover, given the com-
plexity of discovery-oriented scientific questions
and long-form answers, these expert annotations
are inherently subjective and challenging to scale
consistently, which constitutes a key limitation.
10 Ethics Statement
This work received ethics approval through
CSIRO’s ethics approval process for the collec-
tion of human annotations. Three domain scien-
tists served as annotators, two of whom are co-
authors of this paper. The scientists were recruited
via email invitations from within CSIRO and were
involved in genomic research. Participants were
provided with a document describing the project,
including the annotation task, data collection and
anonymisation procedures, and ethics approval de-
tails. Participation was voluntary, and participants
were asked to provide informed consent via email
before taking part in the study.
References
Vaibhav Adlakha, Parishad BehnamGhader, Xing Han
Lu, Nicholas Meade, and Siva Reddy. 2024. Eval-
uating correctness and faithfulness of instruction-
following models for question answering.Trans-
actions of the Association for Computational Linguis-
tics, 12:681–699.Anna Bavaresco, Raffaella Bernardi, Leonardo Berto-
lazzi, Desmond Elliott, Raquel Fernández, Albert
Gatt, Esam Ghaleb, Mario Giulianelli, Michael
Hanna, Alexander Koller, Andre Martins, Philipp
Mondorf, Vera Neplenbroek, Sandro Pezzelle, Bar-
bara Plank, David Schlangen, Alessandro Suglia,
Aditya K Surikuchi, Ece Takmaz, and Alberto
Testoni. 2025. LLMs instead of human judges? a
large scale empirical study across 20 NLP evaluation
tasks. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics, pages
238–255, Vienna, Austria. Association for Computa-
tional Linguistics.
Elliot Bolton, Abhinav Venigalla, Michihiro Yasunaga,
David Hall, Betty Xiong, Tony Lee, Roxana
Daneshjou, Jonathan Frankle, Percy Liang, Michael
Carbin, et al. 2024. "biomedlm: A 2.7b parameter
language model trained on biomedical text".arXiv
preprint arXiv:2403.18421.
Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anasta-
sios Nikolas Angelopoulos, Tianle Li, Dacheng Li,
Banghua Zhu, Hao Zhang, Michael Jordan, Joseph E.
Gonzalez, and Ion Stoica. 2024. Chatbot arena: An
open platform for evaluating LLMs by human pref-
erence. InProceedings of the 41st International
Conference on Machine Learning, volume 235 of
Proceedings of Machine Learning Research, pages
8359–8388. PMLR.
Jennifer D’Souza, Hamed Babaei Giglou, and Quentin
Münch. 2025. YESciEval: Robust LLM-as-a-judge
for scientific question answering. InProceedings of
the 63rd Annual Meeting of the Association for Com-
putational Linguistics, pages 13749–13783, Vienna,
Austria. Association for Computational Linguistics.
Shahul Es, Jithin James, Luis Espinosa Anke, and
Steven Schockaert. 2024. RAGAs: Automated evalu-
ation of retrieval augmented generation. InProceed-
ings of the 18th Conference of the European Chap-
ter of the Association for Computational Linguistics:
System Demonstrations, pages 150–158.
Kehua Feng, Keyan Ding, Tan Hongzhi, Kede Ma, Zhi-
hua Wang, Shuangquan Guo, Cheng Yuzhou, Ge Sun,
Guozhou Zheng, Qiang Zhang, and Huajun Chen.
2025. Sample-efficient human evaluation of large
language models via maximum discrepancy competi-
tion. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics, pages
10913–10947, Vienna, Austria. Association for Com-
putational Linguistics.
Helia Hashemi, Jason Eisner, Corby Rosset, Benjamin
Van Durme, and Chris Kedzie. 2024. LLM-rubric: A
multidimensional, calibrated approach to automated
evaluation of natural language texts. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics, pages 13806–13834,
Bangkok, Thailand. Association for Computational
Linguistics.

Hashem Hijazi, Diego Molla, Vincent Nguyen, and
Sarvnaz Karimi. 2024. Using large language models
to evaluate biomedical query-focused summarisation.
InProceedings of the 23rd Workshop on Biomedical
Natural Language Processing, pages 236–242.
Jessica E. Hyles, Howard A. Eagles, Kerrie Ramm,
Bjorg Sherman, Andrew Gock, Sandra Stops, Tanya
Phongkham, Emmett Leyne, Tina Rathjen, Radoslaw
Suchecki, Lauren Stevens, Louise Ord, Nick S.
Fradgley, Meredith D. McNeil, Annelie Marquardt,
Samuel C. Andrew, Kerrie Forrest, Russell F. East-
wood, Adam Norman, Annette Tredrea, Richard
Trethowan, Ben Trevaskis, and Shannon K. Dillon.
2024. Ozwheat: a genome-to-phenome platform to
resolve complex traits for wheat pre-breeding and
research.bioRxiv.
Alon Jacovi, Yonatan Bitton, Bernd Bohnet, Jonathan
Herzig, Or Honovich, Michael Tseng, Michael
Collins, Roee Aharoni, and Mor Geva. 2024. A
chain-of-thought is as strong as its weakest link: A
benchmark for verifiers of reasoning chains. InPro-
ceedings of the 62nd Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 4615–
4634, Bangkok, Thailand. Association for Computa-
tional Linguistics.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng,
Hanyi Fang, and Peter Szolovits. 2021. What disease
does this patient have? a large-scale open domain
question answering dataset from medical exams.Ap-
plied Sciences, 11(14).
Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Co-
hen, and Xinghua Lu. 2019. PubMedQA: A dataset
for biomedical research question answering. InPro-
ceedings of the 2019 Conference on Empirical Meth-
ods in Natural Language Processing and the 9th In-
ternational Joint Conference on Natural Language
Processing, EMNLP-IJCNLP, pages 2567–2577.
Qiao Jin, Yifan Yang, Qingyu Chen, and Zhiyong Lu.
2024. Genegpt: augmenting large language models
with domain tools for improved access to biomedical
information.Bioinformatics, 40(2).
¸ Senay Kafkas, Marwa Abdelhakim, Azza Altha-
gafi, Sumyyah Toonsi, Malak Alghamdi, Paul N.
Schofield, and Robert Hoehndorf. 2025. The applica-
tion of large language models to the phenotype-based
prioritization of causative genes in rare disease pa-
tients.Scientific Reports, 15(1).
Anastasia Krithara, Anastasios Nentidis, Konstantinos
Bougiatiotis, and Georgios Paliouras. 2023. BioASQ-
QA: A manually curated corpus for Biomedical Ques-
tion Answering.Scientific Data, 10(1).
Wojciech Kryscinski, Bryan McCann, Caiming Xiong,
and Richard Socher. 2020. Evaluating the factual
consistency of abstractive text summarization. In
Proceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing, EMNLP,
pages 9332–9346, Online. Association for Computa-
tional Linguistics.Yanis Labrak, Adrien Bazoge, Emmanuel Morin, Pierre-
Antoine Gourraud, Mickael Rouvier, and Richard
Dufour. 2024. BioMistral: A collection of open-
source pretrained large language models for medical
domains. InFindings of the Association for Compu-
tational Linguistics, pages 5848–5864.
Jon M Laurent, Joseph D Janizek, Michael Ruzo,
Michaela M Hinks, Michael J Hammerling, Sid-
dharth Narayanan, Manvitha Ponnapati, Andrew D
White, and Samuel G Rodriques. 2024. Lab-bench:
Measuring capabilities of language models for biol-
ogy research.arXiv preprint arXiv:2407.10362.
Qintong Li, Leyang Cui, Lingpeng Kong, and Wei Bi.
2025. Exploring the reliability of large language
models as customized evaluators for diverse NLP
tasks. InProceedings of the 31st International Con-
ference on Computational Linguistics, pages 10325–
10344.
Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Abhi-
lasha Ravichander, Valentina Pyatkin, Nouha Dziri,
Ronan Le Bras, and Yejin Choi. 2025. Wildbench:
Benchmarking llms with challenging tasks from
real users in the wild. InInternational Conference
on Learning Representations, volume 2025, pages
47852–47870.
Chin-Yew Lin. 2004. ROUGE: A package for automatic
evaluation of summaries. InText Summarization
Branches Out, pages 74–81.
Tianyu Liu, Yijia Xiao, Xiao Luo, Hua Xu, Wenjin
Zheng, and Hongyu Zhao. 2024. Geneverse: A col-
lection of open-source multimodal large language
models for genomic and proteomic research. InFind-
ings of the Association for Computational Linguistics,
EMNLP, pages 4819–4836.
Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng
Zhang, Hoifung Poon, and Tie-Yan Liu. 2022.
Biogpt: generative pre-trained transformer for
biomedical text generation and mining.Briefings
Bioinform., 23(6).
Yanjun Lyu, Zihao Wu, Lu Zhang, Jing Zhang, Yiwei
Li, Wei Ruan, Zhengliang Liu, Xiaowei Yu, Chao
Cao, Tong Chen, et al. 2024. Gp-gpt: Large language
model for gene-phenotype mapping.arXiv preprint
arXiv:2409.09825.
Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth
Sieber, Mark Yatskar, and Dan Roth. 2024. Ex-
pertQA: Expert-curated questions and attributed an-
swers. InProceedings of the 2024 Conference of
the North American Chapter of the Association for
Computational Linguistics: Human Language Tech-
nologies, pages 3025–3045.
Justin K Miller and Wenjia Tang. 2025. Evaluating
llm metrics through real-world capabilities.arXiv
preprint arXiv:2505.08253.

Jeong Minbyul, Sohn Jiwoong, Sung Mujeen, and Kang
Jaewoo. 2024. Improving medical reasoning through
retrieval and self-reflection with retrieval-augmented
large language models.Bioinformatics, 40.
Vincent Nguyen, Sarvnaz Karimi, Maciej Rybinski, and
Zhenchang Xing. 2023. MedRedQA for medical con-
sumer question answering: Dataset, tasks, and neural
baselines. InProceedings of the 13th International
Joint Conference on Natural Language Processing
and the 3rd Conference of the Asia-Pacific Chapter of
the Association for Computational Linguistics, pages
629–648.
Yingpeng Ning, Yuanyuan Sun, Ling Luo, Yanhua
Wang, Yuchen Pan, and Hongfei Lin. 2025. Medtrust-
rag: Evidence verification and trust alignment for
biomedical question answering. In2025 IEEE
International Conference on Bioinformatics and
Biomedicine (BIBM), pages 1791–1794.
Jekaterina Novikova, Ond ˇrej Dušek, and Verena Rieser.
2018. RankME: Reliable human ratings for natural
language generation. InProceedings of the 2018
Conference of the North American Chapter of the
Association for Computational Linguistics: Human
Language Technologies, pages 72–78, New Orleans,
Louisiana. Association for Computational Linguis-
tics.
Maria Paz Oliva, Adriana D. Correia, Ivan Vankov, and
Viktor Botev. 2025. The illusion of a perfect metric:
Why evaluating AI’S words is harder than it looks.
InProceedings of the 15th International Conference
on Recent Advances in Natural Language Processing
- Natural Language Processing in the Generative AI
Era, pages 832–842, Varna, Bulgaria. INCOMA Ltd.,
Shoumen, Bulgaria.
James M. Ostell. 2012. Entrez: The ncbi search and
discovery engine. InData Integration in the Life
Sciences, pages 1–4.
Yu Pan, Zhongze Cai, Huaiyang Zhong, Guanting Chen,
and Chonghuan Wang. 2026. What matters in data
for dpo?Advances in Neural Information Processing
Systems, 38.
Vatsal Pravinbhai Patel, Elena Jolkver, and Anne
Schwerk. 2026. MedDiscover: A domain-
specific retrieval-augmented generation framework
for evidence-grounded knowledge extraction in
metabolomics.Comput. Struct. Biotechnol. J., 35(1).
Felipe Maia Polo, Xinhe Wang, Mikhail Yurochkin,
Gongjun Xu, Moulinath Banerjee, and Yuekai Sun.
2025. Bridging human and LLM judgments: Under-
standing and narrowing the gap. InThe Thirty-Ninth
Annual Conference on Neural Information Process-
ing Systems.
Nils Reimers and Iryna Gurevych. 2019. Sentence-
BERT: Sentence embeddings using Siamese BERT-
networks. InProceedings of the 2019 Conference on
Empirical Methods in Natural Language Processingand the 9th International Joint Conference on Nat-
ural Language Processing, EMNLP-IJCNLP, pages
3982–3992, Hong Kong, China.
Robert H. Riffenburgh. 2006. Chapter 6 - statistical
testing, risks, and odds in medical decisions. In
Statistics in Medicine, second edition, pages 93–114.
Academic Press, Burlington.
Sujoy Roychowdhury, Sumit Soman, HG Ranjani,
Neeraj Gunda, Vansh Chhabra, and Sai Krishna Bala.
2024. Evaluation of rag metrics for question answer-
ing in the telecom domain. InICML Workshop on
Foundation Models in the Wild.
Tobias Schimanski, Jingwei Ni, Mathias Kraus, Elliott
Ash, and Markus Leippold. 2024. Towards faith-
ful and robust LLM specialists for evidence-based
question-answering. InProceedings of the 62nd An-
nual Meeting of the Association for Computational
Linguistics, pages 1913–1931.
Xinyi Shang, Xu Liao, Zhicheng Ji, and Wenpin Hou.
2025. Benchmarking large language models for ge-
nomic knowledge with geneturing.Briefings in Bioin-
formatics, 26(5).
Michael D Skarlinski, Sam Cox, Jon M Laurent,
James D Braza, Michaela Hinks, Michael J Hammer-
ling, Manvitha Ponnapati, Samuel G Rodriques, and
Andrew D White. 2024. Language agents achieve
superhuman synthesis of scientific knowledge.arXiv
preprint arXiv:2409.13740.
Tatiana Tatusova. 2010. Genomic databases and re-
sources at the national center for biotechnology in-
formation. InData Mining Techniques for the Life
Sciences, pages 17–44.
Chengrui Wang, Qingqing Long, Meng Xiao, Xunxin
Cai, Chengjun Wu, Zhen Meng, Xuezhi Wang, and
Yuanchun Zhou. 2024. Biorag: A rag-llm framework
for biological question reasoning.arXiv preprint
arXiv:2408.01107.
Chaoyi Wu, Weixiong Lin, Xiaoman Zhang, Ya Zhang,
Weidi Xie, and Yanfeng Wang. 2024. Pmc-llama:
toward building open-source language models for
medicine.Journal of the American Medical Infor-
matics Association, 31(9).
Michihiro Yasunaga, Jure Leskovec, and Percy Liang.
2022. LinkBERT: Pretraining language models with
document links. InProceedings of the 60th Annual
Meeting of the Association for Computational Lin-
guistics, pages 8003–8016.
Ziyi Ye, Xiangsheng Li, Qiuchi Li, Qingyao Ai, Yujia
Zhou, Wei Shen, Dong Yan, and Yiqun Liu. 2025.
Learning llm-as-a-judge for preference alignment. In
The Thirteenth International Conference on Learning
Representations, ICLR.
Tan Yue, Rui Mao, Xuzhao Shi, Shuo Zhan, Zuhao
Yang, and Dongyan Zhao. 2025. QAEval: Mixture of
evaluators for question-answering task evaluation. In

Proceedings of the 63rd Annual Meeting of the Asso-
ciation for Computational Linguistics, pages 14717–
14730.
Yuheng Zha, Yichi Yang, Ruichen Li, and Zhiting Hu.
2023. AlignScore: Evaluating factual consistency
with a unified alignment function. InProceedings
of the 61st Annual Meeting of the Association for
Computational Linguistics, pages 11328–11348.
Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q.
Weinberger, and Yoav Artzi. 2020. Bertscore: Evalu-
ating text generation with BERT. In8th International
Conference on Learning Representations, ICLR.
Yilun Zhao, Kaiyan Zhang, Tiansheng Hu, Sihong Wu,
Ronan Le Bras, Taira Anderson, Jonathan Bragg,
Joseph Chee Chang, Jesse Dodge, Matt Latzke, Yixin
Liu, Charles McGrady, Xiangru Tang, Zihang Wang,
Chen Zhao, Hannaneh Hajishirzi, Doug Downey,
and Arman Cohan. 2025. Sciarena: An open evalu-
ation platform for foundation models in scientific
literature tasks. InProceedings of the NeurIPS
2025 Datasets & Benchmarks Track. Available at
arXiv:2507.01001.

Appendix
A Entrez E-utils Framework
The National Center for Biotechnology Informa-
tion (NCBI) is a primary repository for genomic
data, integrating diverse datasets for genomes,
genes, proteins, and more, with analytical and re-
trieval tools through its website (Ostell, 2012). The
Entrez Programming Utilities (E-utilities) (Ostell,
2012), a text-based search system, allows easy nav-
igation across biological databases. Entrez offers
a stable interface for querying and retrieving data
from 38 NCBI databases, including gene records
and biomedical literature (Tatusova, 2010). We
make two API calls on the PMC open access sub-
set of the articles:E-Search, to find the PubMed
IDs (PMIDs) of the most relevant articles for a
given query, andE-Fetch, to extract the abstracts
associated with those PMIDs.
API/Method Parameters
NCBI API DB: PubMed, Sort: relevance
(Entrez) Min-year: 1990, Max-year: 2024
Max-articles:30, Filter: PMC open access
PaperQA2 Top evidences: 5, Evidence summary length: 50
Max answer source: 3, Answer length: 100
LLM: GPT4o-2024-08-06
Table 7: Parameter settings for NCBI API and PaperQA2
Prompt: Query Reformulation—SQsetting
Question: {question}
From the above question, your task is to create a search
phrase to find relevant articles. The search phrase should
contain only one or a few words. Do not include any
word that is not present in the question. Do not generate
numbers and punctuation. Convert any initials to their
full form.
Below are some examples: {samples}
However, below query term(s) didn’t return any article:
{query-error}
Your response should only contain the search phrase.
Search phrase:
Table 8: Few-shot LLM prompt to formulate single query
terms (SQ) to search articles from the NCBI API.
B GENERAG Settings
GENERAG retrieves the top 30 articles based on
theirrelevanceto the query terms and provides
them to PaperQA2 (Skarlinski et al., 2024), which
uses a ranking technique for creating relevant ab-
stract summaries (to use where LLM input context
windows are limited). ForMQ +Cat , we generatedPrompt: Query Reformulation—MQsetting
Your task is to create multiple search phrases to find
relevant articles that may assist in answering a question.
1. If the question is simple and asking about one single
topic, keep the question as it is.
2. If the question is asking a relation between multiple
topics, create sub-questions that will be able to help get
the final answer.
3. Create the most important search phrases for the
questions to be able to search relevant articles.
The search phrase should contain only one or a few
words. Do not include any search term that is not present
in the question. Do not generate numbers and punctua-
tions, unless present in the question. Try to convert any
initials to their full form.
Below are some samples:
Question: {sample-q}
Reasoning: {sample-sub-questions}
Search phrases: [{sample-query-terms}]
. . .
However, the query term(s) didn’t return any article:
{query-error}
Now create n search phrases following the above method
for the following question. Your response should con-
tain the sub-question, if created, and the search phrases.
Question:{question}
Your response:
Table 9: Few-shot LLM prompt to formulate multiple query
terms (MQ) to search articles from the NCBI API.
Prompt: Answer Generation
Question: {question}
For the above question, we retrieve the following con-
text:
Context: {context}
Try to answer the question with the help of the given
context in a maximum {length} words with citation in
the format (citation key - pmid).
If the context provides insufficient information, try to
reply with as much helpful information as you can give.
Your response should only contain the answer to the
question.
Answer:
Table 10: LLM prompt for generating answers for the given
question based on the retrieved context.
3 search terms and concatenated them into a single
string for article search. In the case ofMQ +Stack ,
we formulated 10 search terms per question and
found 10 abstracts for each term. Then we stacked
them in a single list of abstracts to create the con-
text using PaperQA2. In the context produced
by PaperQA2, we have top-3 evidence summaries
inSQandMQ +Cat settings, and top-5 evidence
summaries inMQ +Stack setting. We are using

Question Answer (GPT4)
FROSTTOLGENE—What proportion
of Transcriptomic studies on plants
found ICE1 Transcription factors to be
associated with cold acclimation?
[Contains both gene name and trait]The provided context indicates that ICE1 transcription factors are associated with cold
acclimation in multiple plant studies. Specifically, Wu (2021) discusses the regulation of
cold acclimation pathways by ICE1 and ICE2 in plants (Wu 2021 - 34639192). Addition-
ally, Xu (2015) identifies ICE gene homologs in grapevines that play a crucial role in cold
acclimation, highlighting the involvement of VaICE1 as a transcription factor responding
to cold stress (Xu 2015 - 25019620). However, the exact proportion of studies reporting
this association is not specified in the context. Thus, while ICE1 transcription factors are
recognized for their role in cold acclimation, quantifying the proportion of studies is not
possible based on the given information.
OZWHEATGENE—Candidate gene
TraesCS4D03G0067100 on genome
GCF _018294505.1 has the annotation
of being a GAI-like protein 1 gene.
Are there any known associations or
links between this family of genes and
the trait height in wheat research?
Please also describe the evidence for
the association.[Contains both gene
name and connection to pre-existing
trait]Yes, there are known associations between the GAI-like protein family and height traits
in wheat. The Reduced height-1 (Rht-B1 and Rht-D1) loci, which are orthologues of the
Arabidopsis GAI gene, encode proteins that function as nuclear transcription factors and
are implicated in the regulation of plant height by modulating the response to gibberellin,
a growth hormone (Peng 1999 - 10421366). The introduction of mutant dwarfing alleles at
these loci has led to the development of shorter wheat varieties, significantly contributing
to the green revolution’s impact on wheat yields (Peng 1999 - 10421366). Additionally,
studies of mutant GAI alleles in other crops, such as rice, demonstrate that they also
result in reduced gibberellin responses and dwarfism, suggesting that manipulating GAI-
like proteins could be a strategy to enhance yield across various species (Peng 1999 -
10421366).
Table 11: Example gene-trait association questions from the FROSTTOLGENEand OZWHEATGENEdatasets. Answers are
generated by GPT4.
PaperQA2 specifically to find answer-related pas-
sages for the answer generation stage. PaperQA2
breaks the retrieved abstracts into smaller chunks
and ranks them by their perplexity with respect to
the question. Then, the top 5 abstract chunks are
selected to generate evidence summaries. Finally,
the summaries are further filtered out by re-ranking
them by an LLM-prompt. The top 3 of the evidence
summaries by this re-ranking were then returned as
the final context to be provided to the answer gener-
ation stage. The parameter settings for both Entrez
system and PaperQA2 are outlined in Table 7.
We also provide the used LLM prompts for query
reformulation inSQandMQsettings in Table 8
& 9 and answer generation in Table 10. We pro-
vide some sample questions and answers for the
explored datasets in Table 11.
Query & Context (Prompt Setting–Query LLM)
Eval. Dim. FROSTTOLGENEOZWHEATGENE
Qr2QMQ +Stack -GPT4SQ-GEMMA
C2QMQ +Stack -MistralMQ +Stack -GPT4
C2AbSQ-GEMMAMQ +Cat -Mistral
OverallSQ-GPT3.5 SQ-GPT3.5
Answer ((Prompt Setting–Query LLM–Answer LLM)
A2QSQ-GPT3.5-GPT4MQ +Cat -Mistral-QWEN
A2CMQ +Cat -Mistral-GPT4MQ +Cat -GPT4-GPT4
OverlallMQ +Cat -Mistral-GPT4 MQCAT-GPT4-GPT4
Table 12: Best-performing LLMs for all datasets, based on
metric scores computed for multiple evaluation dimensions.C Metric Details
The evaluation metrics used in this work are as
follows:
Semantic similaritymeasures how closely a text
aligns with a reference text in terms of meaning.
We chose ROUGE-1 (Lin, 2004) (F1-score), cosine
similarity based on SentenceTransformer (Reimers
and Gurevych, 2019) embeddings (CosSim), and
BERTScore (Zhang et al., 2020), and applied them
toQr2Q,C2AbandA2Cevaluation.
Relevanceis an LLM-based metric adopted from
the RAGAS framework (Es et al., 2024) and mea-
suresContext RelevanceandAnswer Relevanceto
the questions for C2Q andA2Q evaluations.An-
swer Relevanceis measured by prompting an LLM
to create several possible questions that could have
led to the given answer. Finally, averagring the co-
sine similarities between these generated questions
and the original question to get the score.Context
Relevanceis measured by asking an LLM to ex-
tract sentences from context that are essential for
answering the question and computing the ratio of
relevant sentences to total context sentences.
Factual consistencyevaluates whether the
claims in the generated answer can be supported
by the source material. We use two metrics: (1)
FactCC (Kryscinski et al., 2020), which checks
the factual correctness; and (2) AlignScore (Zha
et al., 2023) that measures NLI-based textual entail-
ments between two sets of texts. Here, we applied

Prompt Query Query to Ques. (Qr2Q) Context to Ques. (C2Q) Context to Abstracts (C2Ab)
Setting LLM ROUGE-1 BERTSc. CosSim. AlignSc. Query Avg. AlignSc. Relev. Avg. ROUGE-1 BERTSc. AlignSc. FactCC Faith. Avg. Context Avg. Overall Avg.
SQGPT3.5 0.072 0.877 0.817 0.659 0.606 0.202 0.106 0.1540.0700.772 0.461 0.879 0.142 0.465 0.376 0.460
GPT4 0.062 0.877 0.782 0.663 0.596 0.227 0.123 0.175 0.063 0.788 0.487 0.880 0.210 0.486 0.397 0.469
GEMMA 0.068 0.869 0.824 0.623 0.596 0.217 0.149 0.183 0.065 0.781 0.460 0.883 0.165 0.471 0.389 0.464
Mistral0.0890.870 0.758 0.549 0.567 0.229 0.210 0.220 0.047 0.806 0.470 0.839 0.241 0.481 0.406 0.464
MQ +Cat GPT4 0.0290.884 0.843 0.670 0.6070.228 0.300 0.264 0.043 0.826 0.4550.935 0.347 0.521 0.448 0.505
Mistral 0.041 0.870 0.781 0.564 0.564 0.2060.329 0.268 0.0480.8480.460 0.906 0.324 0.517 0.446 0.489
MQ +Stack GPT4 0.047 0.855 0.777 0.636 0.5790.4310.195 0.3130.068 0.8110.5410.930 0.204 0.511 0.454 0.499
Mistral 0.046 0.855 0.777 0.635 0.578 0.430 0.194 0.312 0.067 0.811 0.540 0.931 0.207 0.511 0.454 0.500
Table 13: Metric scores for FROSTTOLGENEquery and context evaluation. Queries are formulated by LLMs in three settings,
and contexts are produced by PaperQA2. Best scores arebolded, and second-best scores are underlined .
Prompt Query Answer Ans. to Ques. (A2Q) Ans. to Context (A2C)
Setting LLM LLM AlignSc. Relev. Avg. AlignSc. ROUGE-1 BERTSc. FactCC Faith. Avg. Avg.
SQGPT3.5 GPT4 0.710 0.875 0.793 0.676 0.013 0.818 0.345 0.187 0.408 0.518
GPT4 GPT4 0.719 0.859 0.789 0.695 0.018 0.822 0.254 0.283 0.414 0.521
GEMMA GPT4 0.715 0.832 0.774 0.691 0.016 0.821 0.332 0.249 0.422 0.522
Mistral GPT4 0.743 0.840 0.792 0.690 0.022 0.831 0.208 0.206 0.391 0.506
GPT3.5 QWEN 0.758 0.867 0.8130.7540.012 0.8050.7360.179 0.497 0.587
GPT4 QWEN 0.7530.893 0.8230.730 0.017 0.813 0.684 0.175 0.484 0.581
GEMMA QWEN 0.760 0.847 0.804 0.710 0.013 0.814 0.629 0.226 0.478 0.571
Mistral QWEN 0.768 0.854 0.811 0.696 0.029 0.824 0.572 0.244 0.473 0.570
MQ +Cat GPT4 GPT4 0.777 0.798 0.788 0.718 0.0420.8620.091 0.249 0.392 0.505
Mistral GPT4 0.756 0.858 0.807 0.6880.0430.858 0.265 0.252 0.421 0.531
GPT4 QWEN 0.716 0.841 0.779 0.659 0.042 0.854 0.060 0.379 0.399 0.507
Mistral QWEN 0.717 0.852 0.785 0.653 0.042 0.851 0.311 0.307 0.433 0.533
MQ +Stack GPT4 GPT40.7920.816 0.804 0.746 0.038 0.848 0.088 0.273 0.399 0.514
Mistral GPT40.7920.831 0.812 0.746 0.038 0.848 0.088 0.273 0.399 0.517
GPT4 QWEN 0.675 0.821 0.748 0.719 0.038 0.842 0.1100.403 0.423 0.515
Mistral QWEN 0.675 0.810 0.743 0.719 0.038 0.842 0.108 0.402 0.422 0.513
Table 14: FROSTTOLGENEmetric scores evaluating the answer quality. Answers are generated by GPT4 and QWEN.
the pretrained BERT-based FactCC model2and
RoBERTa-basedAlignScore-basemodel3. We ap-
plied AlignScore in testing components against the
user questions ( Qr2Q ,C2Q andA2Q ). For C2Ab
andA2C evaluation, we applied both FactCC and
AlignScore.
Faithfulnessrefers to the extent to which the
LLM’s output remains grounded in the evidence.
For the C2Ab andA2C evaluations, the informa-
tion in the context and the answer should be sup-
ported by the biomedical evidence given in the
reference abstracts and contexts, respectively, with-
out adding irrelevant or hallucinated claims. We
used the pretrained hallucination detection model
HHEM-2.1-Open model from Vectara4.
D Automatic Metric Results
We found that there is no single language model
that performs well across all stages of the RAG
pipeline. As shown in the results in Table 13 on
the FROSTTOLGENEdataset, the highest average
metric scores were achieved by different models
2https://huggingface.co/manueldeprada/FactCC
3https://huggingface.co/yzha/AlignScore
4https://shorturl.at/naxUxfor different evaluation dimensions. For example,
for the query-to-question ( Qr2Q ) quality, GPT4
generated queries inMQ +Cat had the best scores.
However, GPT4 inMQ +Stack setting generated the
best contexts aligned to the question (context-to-
question, C2Q ) and GPT4 inMQ +Cat generated
the best contexts with respect to the retrieved ab-
stracts (context-to-abstract,C2Ab).
For the answer quality (Table 14), none of the
contexts that were generated by the better quality
LLMs were able to produce better answers aligned
to the question (answer-to-question, A2C ) or con-
text (answer-to-context, A2C ). Looking at the sum-
mary of the best performing LMs in different set-
tings from Table 12, we observe a similar trend
across all datasets. These results essentially in-
dicate that rather than depending on one LLM at
each stage of the pipeline, it is better to select a
combination of LLMs through multiple evaluation
dimensions for optimal outcome.
Furthermore, we present the detailed individual
metrics scores and averaged scores on each dimen-
sion in Table 13-14 on FROSTTOLGENEand Ta-
ble 15-16 on OZWHEATGENEdatasets. Looking
at these detailed scores we observe that, because
different evaluation metrics emphasise different

Prompt Query Query to Ques. (Qr2Q) Context to Ques. (C2Q) Context to Abstracts (C2Ab)
Setting LLM ROUGE-1 BERTSc. CosSim. AlignSc. Query Avg. AlignSc. Relev. Avg. ROUGE-1 BERTSc. AlignSc. FactCC Faith. Avg. Context Avg. Overall Avg.
SQGPT3.50.0490.853 0.761 0.544 0.552 0.273 0.028 0.151 0.075 0.753 0.578 0.989 0.054 0.650 0.393 0.523
GPT4 0.046 0.839 0.696 0.483 0.516 0.280 0.085 0.183 0.032 0.7600.5840.912 0.083 0.634 0.391 0.509
GEMMA 0.046 0.839 0.696 0.483 0.516 0.280 0.112 0.196 0.032 0.7600.584 0.9950.083 0.651 0.407 0.519
Mistral 0.046 0.851 0.703 0.543 0.536 0.289 0.070 0.1800.0930.765 0.572 0.898 0.096 0.625 0.398 0.511
MQ +Cat GPT4 0.036 0.8580.8060.605 0.576 0.424 0.372 0.398 0.045 0.828 0.494 0.3040.317 0.398 0.398 0.463
Mistral 0.027 0.852 0.732 0.575 0.547 0.4140.402 0.408 0.0840.8300.514 0.419 0.226 0.415 0.413 0.461
MQ +Stack GPT4 0.0400.8610.8010.657 0.5900.488 0.270 0.379 0.043 0.813 0.493 0.205 0.204 0.352 0.359 0.443
Mistral 0.036 0.840 0.716 0.647 0.5600.5600.296 0.4280.042 0.810 0.528 0.254 0.163 0.359 0.379 0.445
Table 15: Metric scores for OZWHEATGENEquery and context evaluation. Queries are formulated by multiple LLM settings
and contexts are produced by PaperQA (GPT4).
Prompt Query Answer Ans. to Ques. (A2Q) Ans. to Context (A2C)
Setting LLM LLM AlignSc. Relev. Avg. AlignSc. ROUGE-1 BERTSc. FactCC Faith. Avg. Avg.
SQGPT3.5 GPT4 0.678 0.835 0.7570.655 0.006 0.796 0.651 0.080 0.438 0.529
GPT4 GPT4 0.688 0.798 0.743 0.671 0.006 0.797 0.633 0.091 0.440 0.526
GEMMA GPT4 0.688 0.801 0.745 0.671 0.006 0.797 0.633 0.091 0.440 0.527
Mistral GPT4 0.666 0.783 0.725 0.667 0.009 0.797 0.416 0.035 0.385 0.482
GPT3.5 QWEN 0.687 0.816 0.752 0.593 0.004 0.788 0.592 0.089 0.413 0.510
GPT4 QWEN 0.654 0.818 0.736 0.601 0.005 0.7920.6560.072 0.425 0.514
GEMMA QWEN 0.654 0.820 0.737 0.601 0.005 0.792 0.655 0.072 0.425 0.514
Mistral QWEN 0.659 0.813 0.736 0.576 0.008 0.792 0.562 0.052 0.398 0.495
MQ +Cat GPT4 GPT4 0.620 0.788 0.704 0.765 0.0430.8740.458 0.396 0.507 0.563
Mistral GPT4 0.619 0.796 0.708 0.753 0.040 0.870 0.509 0.396 0.514 0.569
GPT4 QWEN 0.649 0.724 0.687 0.6510.0500.841 0.294 0.464 0.460 0.525
Mistral QWEN 0.592 0.722 0.657 0.680 0.036 0.849 0.512 0.483 0.512 0.553
MQ +Stack GPT4 GPT4 0.687 0.811 0.7490.7870.039 0.868 0.2310.609 0.507 0.576
Mistral GPT4 0.615 0.789 0.702 0.759 0.040 0.863 0.372 0.480 0.503 0.560
GPT4 QWEN 0.621 0.734 0.678 0.686 0.038 0.845 0.107 0.489 0.433 0.503
Mistral QWEN 0.617 0.811 0.714 0.683 0.042 0.845 0.114 0.483 0.433 0.514
Table 16: OZWHEATGENEmetric scores evaluating the answer quality. Answers are generated by GPT4 and QWEN.
quality aspects of the evaluated texts, no single
metric provides a complete picture. For example,
from the FactCC and Faithfulness metric scores for
FROSTTOLGENEand OZWHEATGENEdatasets
in the Table 14 and 16, we observe that the an-
swers generated by using the contexts in the single
query setting are factually more accurate, but less
faithful than the others. A combination of met-
rics is therefore more effective, with each metric
contributing differently to the assessment of LLM
response quality.
Through an analysis of the Pearson correla-
tion coefficient racross different metric combi-
nations, we observe varying levels of correlation
between evaluation metrics, as illustrated in the
Figure 2. Across all datasets, metrics assessing
answer–context alignment (the A2C dimension)
exhibit relatively higher correlation with metrics
measuring context–question alignment (the C2Q
dimension). In contrast, query–question evaluation
metrics ( Qr2Q ) show consistently lower correla-
tion with metrics from other evaluation dimensions.
Beyond these trends, no consistent correlation pat-
terns are observed across the remaining metric com-
binations. This analysis indicates that there is no
one-size-fits-all metric configuration for evaluat-ing RAG systems, and it is crucial to incorporate
human guidance when selecting and combining
evaluation metrics.
In Table 17, we report the agreement rates be-
tween uniform-weighted metric-based preferences
and human preferences (human annotated set) and
LLM preferences (synthetic annotations).
E LLM-as-a-Judge
Following the work of Lin et al. (2025), we create
in-prompt rubrics containing questions to check
the quality of the answers as a guideline for the
LLM to generate preferences over pairwise an-
swers. The used rubric questions are listed in Ta-
ble 18. The generic rubric is created by taking
motivation from WildBench (Lin et al., 2025) and
LLM-rubric (Hashemi et al., 2024) and connecting
with the evaluation dimension explored in our work.
To create the domain-specific rubric, we consulted
the domain experts.
F Quality Dimensions vs. human
reasonings
We promptGPT4o-miniin a one-shot prompt set-
ting to group similar human reasonings and label
the quality aspects considered in the reasons. Ta-

(a)
(b)
Figure 2: Pearson’s correlation coefficients ( r) be-
tween the evaluation metrics for a) FROSTTOLGENEb)
OZWHEATGENEdatasets.
ble 19 shows the used prompt for this purpose.
Some example annotations are illustrated in Ta-
bles 23&24. The resultant criteria and the human
reasoning under each criterion are presented in Ta-
ble 22. Based on the criteria labels, the frequency
of each criterion in the case of mismatched human-
metric criteria for ‘all’ and a subset of metrics is in
Table 21.
G SciArena Dataset
The SCIARENA(Zhao et al., 2025) is an open,
crowdsourced benchmarking platform designed to
evaluate foundation models on scientific literature-
grounded tasks. The dataset5contains 13,204 hu-
man preferences over pairwise LLM-generated an-
swers on 25 subjects. From this dataset, we in-
5https://huggingface.co/datasets/yale-nlp/
SciArenaMetric Hum-Metric Pearson’s LLM-Metric Pearson’s
Agree. (%)rAgree. (%)r
All 61.450.300 (p=0.002)47.1630.267 (p=0.005)
Qr2Q
AlignScore 59.04 0.164 (p=0.035) 48.27 -0.044 (p=0.005)
ROUGE 52.41 -0.010 (p=0.902) 46.99 -0.069 (p<0.001)
BERTScore 50.60 -0.003 (p=0.974) 46.77 -0.079 (p<0.001)
CosSim. 51.20 0.002 (p=0.976) 47.85 -0.050 (p=0.001)
C2Q
AlignScore 56.63 0.094 (p=0.228) 64.370.262 (p<0.001)
Relevance 51.81 -0.008 (p=0.919) 55.750.082 (p<0.001)
C2Ab
AlignScore 53.01 0.032 (p=0.679) 49.23 -0.023 (p=0.138)
ROUGE 49.40 -0.046 (p=0.557) 38.82 -0.226 (p<0.001)
BERTScore 50.60 -0.021 (p=0.790) 54.72 0.055 (p = 0.003)
FactCC 61.450.298 (p=0.010)42.77 -0.138 (p<0.001)
Faithfulness 50.60 -0.018 (p=0.822) 57.18 0.106 (p<0.001)
A2Q
AlignScore 61.640.301 (p=0.050)56.42 0.124 (p<0.001)
Relevance 39.73 -0.230 (p=0.051) 43.02 -0.127 (p<0.001)
A2C
AlignScore 65.750.313 (p=0.007)68.000.370 (p<0.001)
ROUGE 52.05 0.245 (p=0.704) 58.85 0.143 (p<0.001)
BERTScore 57.53 0.253 (p=0.197) 68.200.354 (p<0.001)
FactCC 57.53 0.245 (p=0.219) 40.83 -0.148 (p<0.001)
Faithfulness 61.640.303 (p=0.039)61.820.208 (p<0.001)
Table 17: Rates of agreements between uniform-weighted
metric-based preferences against human preferences (human
annotated set) and LLM preferences (synthetic annotations).
Bold indicates statistical significance.
Rubric: Generic
- Does the response answer the user’s question?
- Is the answer easily understandable and complete with non-
repetitive information?
- Are the references useful to answer the question?
- Does the answer cite the references correctly for each claim?
- Can the claims in the answers be accurately supported by the
given references?
Rubric: Domain-specific
- Does the answer include a short description of the function of
the candidate gene?
- Does the answer state if it can or cannot find evidence linking
the candidate gene to the target trait?
- If evidence has been identified, are the type or types of evidence
described?
- Does the answer reference the source of all evidence linking the
candidate gene to the target trait?
Table 18: Used generic and domain-specific rubrics to guide
the LLM in judging the pairwise answers
vestigate metric weight calibration on the subjects
closer to our focus domain. The list of subjects and
their data size is reported in Table 20 and an exam-
ple question and answer pair is given in Table 25.
Note, we excluded the ‘tie’ or ‘bad’ annotations
from the investigation.
H Human Annotation App
The annotation application was hosted on an in-
ternal server. We provided secure logins to the
annotators to access the survey application. We
presented both answer pairs on a single screen for

Prompt: Grouping Human Ratoinals
human reasonings: {reasons}
Given the above list of reasons for not preferring some
LLM-generated answers, try to group them into reason-
ing categories.
Give a short category name for each category.
Return the output in a JSON firmat, where "criteria"
is the "<category label>" and "reasons" is the list of
reasons fall into the category.
For example, "criteria": "answer relevance", "reasons":
[’The answer is not as relevant as it also does not provide
specific information regarding the proportion of studies
linking Ethylene-responsive transcription factors with
cold acclimation.’, ’The answer does not provide any rel-
evant supporting evidence that links ethylene-responsive
transcription factors to cold acclimation, making it less
relevant.’]
Your response:
Table 19: One-shot LLM prompt to group the human reason-
ings for not preferring an answer and create quality dimension.
Subject Datasize
Combined 2251
Biomed. 662
Medicine 801
Neuroscience 206
Pharmacy 334
Public Health 248
Table 20: Dataset size of the different subjects in SCIARENA
Total
Similarity to the Selected Answer
Relevance of Answer
Lack of Details
Lack of Supporting Evidence
Incorrect or irrelevant context
Organization and Structure
Answer Clarity and Fluency
Inaccurate Claims
All 54 16 15 9 4 4 3 2 2
Qr2Q 70 15 28 11 7 4 1 - 1
C2Q 62 18 22 10 6 3 2 - 1
C2Ab 75 17 30 12 7 6 1 1 1
C2Q+C2Ab 70 18 27 11 6 6 - 1 1
A2Q 72 21 26 8 6 5 3 1 2
A2C 56 19 15 8 5 4 2 2 1
A2Q+A2C 57 19 14 9 4 5 3 2 1
Table 21: Criteria from human reasons grouped by LLM for
mismatched human-metric preference cases.
each question. One answer in the second pair isloaded based on the choice of the first answer. The
reasons for not preferring an answer are collected
as free texts. Also, we employed a separate text box
for the annotators to provide stylistic reasons. For
ease of reading, we highlighted the same PubMed
IDs (PMIDs) with one color.

Criteria Frequency Example Reasons
Relevance of Answer 54 ‘the answer is not as relevant’, ‘answer is not relevant to the question’,‘the
answer is neither sufficiently relevant nor complete.’, ‘not as relevant to
the candidate gene’, ‘the answer is not as relevant and does not provide
inline citations’, ‘the answer is very similar to the selected answer but
less relevant’, ‘not as relevant does not get to answer’, ‘not as relevant, no
description of gene function’, ‘the answer is neither sufficiently relevant
nor complete.’
Similarity to the Selected Answer 35 ‘the answer is very similar to the selected answer but not as concise.’, ‘the
answer has less information about the other reference.’, ‘similar, irrelevant
context’, ‘both bad, no description of gene function’, ‘the answer is almost
similar to the selected answer’
Lack of details 16 ‘the answer is not as detailed. The paragraph structure is not helpful for
directly answer the question.’, ‘the answer has less information’, ‘the
answer is not as detailed.’, ‘the answer has less information.’, ‘details
provided could be misleading to a novice.’, ‘answer is not as detailed’,
‘less detail’, ‘the answer is neither sufficiently relevant nor com plete.’,
‘the answer is not complete’
Incorrect or Irrelevant Context 11 ‘incorrect context’, ‘irrelevant context, no description of gene function’,
‘context has low relevance’
Lack of Supporting Evidence 11 ‘no supporting evidence’, ‘the answer does not provide inline citations.’,
‘the answer is not as relevant and does not provide inline citations’, ‘did
not define function of gene’, ‘irrelevant; did not define function of gene’,
‘not as relevant, no description of gene function’, ‘part of the answer is
not as relevant and there is no citations’, ‘the answer is not as relevant, no
supporting evidence’
Organization and Structure 7 ‘the paragraph structure is not helpful for directly answer the question.’,
‘the paragraph structure and organization are not as useful’, ‘the answer
is not as relevant paragraph structure and organization are not helpful
(not using topic sentence structure)’, ‘incomplete, intro too long’, ‘The
phrasing in the answer is not as fluent.’
Answer Clarity and Fluency 4 ‘The phrasing in the answer is not as fluent.’, ‘Answer did not properly
address the two parts of the question.’, ‘Answer did not fully address the
question.’, ‘The answer is very similar to the selected answer but has less
information.The phrasing in the answer is not as fluent.’
Inaccurate Claims 2 ‘Answer has incorrect claims / evidence’, ‘not as accurate’
Total reasons 140
Table 22: LLM generated quality dimensions from the human reasons.

Example #1
Question:What proportion of Transcriptomic studies on plants found Gpcr-type g protein 2 gene
to be associated with cold acclimation?
Human reason:The answer has less information
Human preference:Answer 2Metric preference:Answer 1
Answer 1:"The provided context does not mention any specific transcriptomic studies on plants that
associate the GPCR-type G protein 2 gene with cold acclimation. The cited studies focus on viroid
effects on gene expression (Joubert 2022 - 35887025), stress responses via RNA-seq and single-cell
RNA-seq (Rurek 2024 - 39524930), and single-cell transcriptomics for plant development (Yu
2023 - 36540021). None of these contexts address GPCR-type G protein 2 gene functionality or its
relationship with cold acclimation. Therefore, the proportion cannot be determined.",
Anwer 2:"The provided context does not specify the proportion of transcriptomic studies on plants
that found the G-protein coupled receptor (GPCR) type G protein 2 gene to be associated with
cold acclimation. It emphasizes the general mechanisms of cold acclimation, such as physiological
adjustments and changes in gene expression involving C-repeat binding factors (CBFs) (Barrero-Gil
2019 - 30288701), as well as the role of various proteins in regulating this process (Barrero-Gil
2013 - 23498862). However, specific information regarding the GPCR-type G protein 2 gene’s
association with cold acclimation is not included in the provided studies. Thus, I cannot provide a
definitive answer regarding the proportion."
Example #2
Question:Candidate gene TraesCS6B03G1296700 on genome GCF_018294505.1 has the annota-
tion of being a SKIP interacting protein 16 gene. Are there any known associations or links between
this family of genes and the trait height in wheat research? Please also describe the evidence for the
association.
Human preference:Answer 1Metric preference:answer 2
Answer 1:"The context does not provide direct evidence linking the SKIP interacting protein 16
gene (TraesCS6B03G1296700) to wheat height. While SKIP is described as a conserved splicing
factor and transcriptional regulator in plants (Li 2022 - 35640153), its role in wheat or specific
associations with plant height are not mentioned. The TB1 gene in wheat is noted to influence plant
height through allelic variation (Dixon 2021 - 32449515), but this is unrelated to SKIP. Additionally,
SKIP’s role in inhibiting sphingosine kinase (Ghazaly 2020 - 32161116)."
Answer 2:"There is limited direct evidence linking SKIP interacting protein 16 genes, including
TraesCS6B03G1296700, to height traits in wheat. However, studies on plant height as an important
agronomic trait highlight the role of various genes in influencing this characteristic. The Reduced
height-B1b (Rht-B1b) and Rht-D1b alleles have been specifically associated with semi-dwarf
plant architecture, enhancing lodging resistance (Song 2023 - 37100915). Furthermore, QTL
analyses have been effective in identifying and estimating the effects of genes related to plant height
(Paccapelo 2022 - 35597886; Zhou 2023 - 37349717). While the specific role of SKIP interacting
protein 16 in height traits remains to be elucidated, its interaction in gene networks may warrant
further investigation in the context of height-related QTLs."
Table 23: Example of human annotation and metric-based preference mismatch when two answers are very similar.

Question:
Candidate gene TraesCS6B03G0835400 on genome GCF_018294505.1 has the annotation
of being a SBP (S-ribonuclease-binding protein) family protein gene. Are there any known
associations or links between this family of genes and the trait spike length in wheat research?
Please also describe the evidence for the association.
Answer 1:
The SBP (S-ribonuclease-binding protein) family gene TraesCS6B03G0835400 is linked to spike
length in wheat. Research indicates that SBP genes, such as BdSBP1 and BdSBP3, play roles
in spike development, with their expression patterns during spikelet initiation and responses to
environmental factors like vernalization and temperature (Tripathi 2020 - 32929136). A T-DNA
mutant of Bdsbp9 showed reduced spike length, highlighting the functional importance of SBP
genes in spike development (Tripathi 2020 - 32929136).
Answer 2:
There is currently no direct evidence linking the SBP (S-ribonuclease-binding protein) family genes,
such as TraesCS6B03G0835400, to spike length traits in wheat research based on the provided
context. The context discusses the association of specific haplotypes in different genes with spike
length traits (Zhang 2023 - 37310852) but does not mention SBP family genes or their role in
determining spike length. Additionally, while SPL genes are mentioned in relation to plant growth
traits (Zhao 2022 - 36011366), they do not directly address the SBP family or spike length in wheat.
Therefore, further research would be necessary to explore any potential associations between SBP
genes and spike length in wheat.
Annot. 1 Annot. 2
Preference: #1 Preference: #1
Rationale: Rationale:
Incorrect claim or evidence The answer is not as relevant
Table 24: Different reasonings for the same answer pair and the same preferences by two annotators.

Subject:Biomed. Engi.
Question:What recent progress characterises novel monolithic stationary phases for liquid chromatogra-
phy?
Llama-4-Scout:The development of novel monolithic stationary phases has significantly advanced liquid
chromatography, offering improved separation efficiency, selectivity, and versatility. Recent progress in
this area includes the introduction of nanomaterial-based stationary phases, such as nanoporous silica,
zirconia, and graphene, which provide ultra-high surface areas and nanoscale confinement effects (Ratnam
et al. 2023b).
Monolithic columns, made from organic polymers, inorganic materials, or hybrid structures, have been
engineered to offer tailored characteristics for specific analyses (Nechv ˘00e1talov ˘00e1 et al. 2022a). These
columns have demonstrated superior permeability, efficiency, and ease of preparation, making them
attractive alternatives to traditional particle-packed columns (Gregu ˘0161 et al. 2023a). The incorporation
of metal ions, such as Co(II), into molecularly imprinted polymers has enhanced selectivity for separating
enantiomeric compounds (Amalia et al. 2022b). Additionally, zwitterionic sulfobetaine-based monoliths
have been developed for hydrophilic interaction chromatography, exhibiting good selectivity and efficiency
(Wang et al. 2022a).
The use of bio-inspired materials, like lipase-immobilized monoliths, has also shown potential for
chiral separations (Fouad et al. 2023a). Furthermore, the integration of monolithic columns with mass
spectrometry has enabled sensitive and high-throughput analysis of complex samples (Shamsi et al. 2022a).
Overall, these advancements in monolithic stationary phases are transforming liquid chromatography,
enabling faster, more sensitive, and high-resolution separations for a wide range of applications (Ratnam
et al. 2023b).
Deepseek-R1:Recent advancements in monolithic stationary phases for liquid chromatography (LC)
emphasize enhanced selectivity, fabrication techniques, and application-specific designs. Gradient mono-
lithic phases, achieved through controlled chemical modifications like the controlled rate infusion method
or photo-grafting, enable gradual changes in surface functionality along the column, improving resolution
for complex biological samples (Alla et al. 2022a). Metal-mediated molecularly imprinted polymers
(MMIPs) have emerged for chiral separations, with cobalt(II)-based monolithic columns demonstrating
high enantioselectivity for racemic citronellal, leveraging metal coordination to enhance template recogni-
tion (Amalia et al. 2022b).
Hybrid organic-inorganic monoliths, such as polymer-bound silica phases, combine high permeability with
exceptional efficiency ( 288,000 plates/m for peptides), addressing challenges in biomolecule separation
(Ali et al. 2024b). Innovations in miniaturization include microbore columns (1 mm i.d.) that reduce
solvent consumption while maintaining performance, ideal for environmentally friendly LC systems
(Amalia et al. 2022a). Zwitterionic sulfobetaine-based monoliths, fabricated via rapid photoinitiated
copolymerization, offer superior hydrophilic interaction chromatography (HILIC) capabilities, benefiting
polar analyte separations (Wang et al. 2022a).
Composite monoliths integrating nanomaterials (e.g., carbon nanotubes) or metal-organic frameworks
(MOFs) enhance surface area and selectivity, particularly for proteomic and pharmaceutical analyses
(Nechv ˘00e1talov ˘00e1 et al. 2022a). Additionally, advancements in nano-LC utilize silica-based mono-
lithic columns with sub-1 ˘00b5m pores for sensitive food analysis, achieving low back-pressure and
high mass transfer kinetics (Aydo ˘011fan et al. 2023a). These developments highlight trends toward
multifunctional, application-tailored monoliths, improved fabrication reproducibility, and integration with
miniaturized systems, expanding their utility in biomedicine, environmental monitoring, and industrial
chromatography (Gregu ˘0161 et al. 2023a, Ratnam et al. 2023c).
Human preference: 1
Table 25: Example questions and answers from SCIARENAdataset for Biomedical Engineering domains.