# META-RAG: Meta-Analysis-Inspired Evidence-Re-Ranking Method for Retrieval-Augmented Generation in Evidence-Based Medicine

**Authors**: Mengzhou Sun, Sendong Zhao, Jianyu Chen, Haochun Wang, Bin Qin

**Published**: 2025-10-28 02:18:09

**PDF URL**: [http://arxiv.org/pdf/2510.24003v1](http://arxiv.org/pdf/2510.24003v1)

## Abstract
Evidence-based medicine (EBM) holds a crucial role in clinical application.
Given suitable medical articles, doctors effectively reduce the incidence of
misdiagnoses. Researchers find it efficient to use large language models (LLMs)
techniques like RAG for EBM tasks. However, the EBM maintains stringent
requirements for evidence, and RAG applications in EBM struggle to efficiently
distinguish high-quality evidence. Therefore, inspired by the meta-analysis
used in EBM, we provide a new method to re-rank and filter the medical
evidence. This method presents multiple principles to filter the best evidence
for LLMs to diagnose. We employ a combination of several EBM methods to emulate
the meta-analysis, which includes reliability analysis, heterogeneity analysis,
and extrapolation analysis. These processes allow the users to retrieve the
best medical evidence for the LLMs. Ultimately, we evaluate these high-quality
articles and show an accuracy improvement of up to 11.4% in our experiments and
results. Our method successfully enables RAG to extract higher-quality and more
reliable evidence from the PubMed dataset. This work can reduce the infusion of
incorrect knowledge into responses and help users receive more effective
replies.

## Full Text


<!-- PDF content starts -->

META-RAG: Meta-Analysis-Inspired Evidence-Re-Ranking Method for
Retrieval-Augmented Generation in Evidence-Based Medicine
Mengzhou Sun1, Sendong Zhao1, Jianyu Chen1, Haochun Wang1, Bin Qin1
1Faculty of Computing, Harbin Institute of Technology
mzsun@ir.hit.edu.cn, sdzhao@ir.hit.edu.cn
Abstract
Evidence-based medicine (EBM) holds a crucial role in clin-
ical application. Given suitable medical articles, doctors ef-
fectively reduce the incidence of misdiagnoses. Researchers
find it efficient to use large language models (LLMs) tech-
niques like RAG for EBM tasks. However, the EBM main-
tains stringent requirements for evidence, and RAG applica-
tions in EBM struggle to efficiently distinguish high-quality
evidence. Therefore, inspired by the meta-analysis used in
EBM, we provide a new method to re-rank and filter the med-
ical evidence. This method presents multiple principles to fil-
ter the best evidence for LLMs to diagnose. We employ a
combination of several EBM methods to emulate the meta-
analysis, which includes reliability analysis, heterogeneity
analysis, and extrapolation analysis. These processes allow
the users to retrieve the best medical evidence for the LLMs.
Ultimately, we evaluate these high-quality articles and show
an accuracy improvement of up to 11.4% in our experiments
and results. Our method successfully enables RAG to extract
higher-quality and more reliable evidence from the PubMed
dataset. This work can reduce the infusion of incorrect knowl-
edge into responses and help users receive more effective
replies.
Introduction
Currently, Evidence-Based Medicine (EBM) is gradually
being embraced by doctors as an essential discipline in
the medical field (Subbiah 2023). Using EBM can signif-
icantly reduce the risk of misdiagnosis by referring to the
retrieved medical articles. As the volume of medical evi-
dence grows, doctors start to rely on artificial intelligence
(AI) technology to assist in the practice of EBM (Djulbe-
govic and Guyatt 2017). The key requirement from AI is to
leverage all available resources, extracting and synthesizing
all relevant evidence to arrive at a comprehensive conclu-
sion (Clusmann et al. 2023). However, due to the limitation
of memory capacity, small-scale models often struggle to
deal with a large amount of evidence (Friedman, Rindflesch,
and Corn 2013; Nadkarni, Ohno-Machado, and Chapman
2011). Recently, Large Language Models (LLMs) have been
presented, which are equipped with a long input restric-
tion and exceptional comprehension ability. There have been
breakthroughs in using LLMs to assist EBM.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Query  how many 
months will it take for 
me to be discharged 
from the hospital?
?
30 daysRelative evidence ：
?
10 days
10 days
??
30 days?
30 days
Random selected
Answer： 10 daysThinking ：
More agree 10 days 
Query  how many 
months will it take for 
me to be discharged 
from the hospital?
30 daysRe-ranked evidence ：
10 days
10 days
30 daysScore 6.2
30 days
High -quality
Answer：30 daysThinking ：
30 days more 
reliable Score 5.1Meta -RAG:Base RAG:
Score 4.4Score 3.1
Score 2.9
Figure 1: When traditional RAG processes a query, it
probably retrieves a large volume of unhelpful and non-
professional evidence. This evidence may include condi-
tional results and outdated conclusions. This will mislead
the generator to mistakes.
With the iterative advancements in LLM technol-
ogy, innovative methods like Retrieval-augmented Genera-
tion(RAG) and knowledge fine-tuning have emerged (Alam,
Giglou, and Malik 2023). They can minimize the knowl-
edge errors made by LLMs (Zhang et al. 2023; Huang et al.
2023). The core process of RAG, which involves retriev-
ing evidence and generating diagnoses, closely aligns with
the fundamental principles of EBM. As a result, RAG has
the most potential to enhance the efficiency of EBM. How-
ever, RAG faces several limitations when applied to clinical
medicine. EBM requires a highly rigorous process for select-
ing and filtering the retrieved evidence (Sackett, Richardson,
and Rosenberg 2008). Traditional RAG fails to adequately
address this process because of the complexity of medicalarXiv:2510.24003v1  [cs.CL]  28 Oct 2025

articles. This oversight often leads to the retrieval of con-
flicting and redundant evidence. For instance, as illustrated
in Figure 1, the vanilla RAG probably retrieves a large vol-
ume of unhelpful and non-reliable evidence. This evidence
may include conditional results and outdated conclusions.
Consequently, RAG selects this evidence to mislead the re-
sponse, which will significantly restrict the accuracy.
To address the above issues, we develop META-RAG for
evidence re-ranking and filtering in RAG for EBM. By ac-
quiring more reliable and valid evidence, this method en-
ables RAG to retrieve evidence that is both more trustwor-
thy and consistent, thereby reducing erroneous judgments.
We emulate the principles of meta-analysis, which focuses
on three key aspects: (1) reliability, (2) heterogeneity, and
(3) extrapolation (Lipsey 2001; Egger, Smith, and Phillips
1997; Hansen, Steinmetz, and Block 2022). META-RAG fil-
ters out inconsistent evidence and presents reliable and rig-
orous evidence to the response model. As shown in Figure 2,
first, we gather the related medical articles and assign a base
score to each article based on its publication type. Then,
we assess the information of evidence to judge the reliabil-
ity score accordingly and filter the heterogeneous articles.
We evaluate the extrapolation by considering the limitations
of the experimental results for the users. Finally, the reli-
able and high-quality articles are passed to the generator.
We present the experiments and results to prove our method
effectively resolves the issues of low-quality and conflicting
evidence. This method can significantly improve the accu-
racy of the RAG process in providing correct responses.
Our contributions can be summarized in three aspects:
• Inspired by meta-analysis, we re-rank the evidence by
adopting the evaluation dimensions from meta-analysis,
assessing the evidence based on its grade, methodologi-
cal reliability, and extrapolation.
• We utilize LLM agents to analyze the extrapolative and
reliable potential of the evidence, reducing subjectivity
in the evidence selection process.
• We conduct an evaluation method for the quality of evi-
dence. By scoring the contribution of the articles to each
option, we can observe the improvement of evidence.
Related Works
EBM and Meta-Analysis
Healthcare professionals have recognized and accepted
EBM as an important discipline in recent years. EBM aims
to make the best clinical decisions by integrating the best
research evidence, clinical expertise, and patient prefer-
ences (Subbiah 2023; McMurray and Packer 2021). Many
practitioners of EBM attempt to use advanced AI technolo-
gies to aid in the search process. However, doctors cannot
trust AI models because of hallucinations. They would like
to choose the time-consuming and subjective manual ap-
proach unless the LLMs (Li et al. 2024).
To eliminate biases arising from subjective choices, re-
searchers propose the method known as meta-analysis.
Meta-analysis is a quantitative research technique designedto systematically integrate the results of multiple inde-
pendent studies to provide more rigorous conclusions. It
is widely used in fields like medicine, social science,
and education, especially in studies derived from experi-
ments (Borenstein et al. 2021). In meta-analysis, researchers
aggregate data from multiple independent studies and con-
duct uniform statistical analyses to determine overall effect
sizes or other relevant statistical metrics (Hansen, Stein-
metz, and Block 2022). However, each meta-analysis re-
quires manually compiling more relevant literature, which
is highly complex. Therefore, we hope to utilize the core
comparative elements of meta-analysis and employ LLMs
to assist users in evaluating evidence.
RAG for EBM
LLMs have recently made significant progress in natural
language processing. High-performance models like GPT-
4 (Achiam et al. 2023) have achieved substantial break-
throughs in fields such as medicine, military, and law.
Google MED-PALM (Singhal et al. 2023) suggests that
LLMs can be applied in many tasks within clinical. With
RAG method, the LLMs can deal with these complex tasks
with few hallucinations (Lewis et al. 2020). The princi-
ple of EBM, which relies on extensive medical evidence
for decision-making, aligns well with this approach. RAG
generative method is particularly well-suited for EBM and
serves as an effective tool for assisting doctors in resolving
clinical issues.
However, medicine constantly evolves at a rapid pace,
leading to inconsistencies in viewpoints among publications
like the articles in PubMed (White 2020). RAG may retrieve
outdated, incorrect, and restricted theories. They may have
once been accepted but no longer right because of the pro-
posal of a new theory. This phenomenon will result in some
conclusions being inapplicable to the actual situation.
Evidence Re-Ranking
Currently, there are three main methods for optimizing the
evidence retrieved during the RAG process: scoring based
on rules, trained models, and LLMs that have re-ranking
capabilities (Gao et al. 2023b). Researchers tend to em-
ploy existing rules for the task, relying on predefined met-
rics such as diversity, relevance, and Mean Reciprocal Rank
(MRR) (Gao et al. 2023b). By calculating specific values
for these articles, those with higher values are prioritized in
the ranking. Model-based approaches used traditional Trans-
former models like SpanBERT (Joshi et al. 2020).
The third method utilizes some specialized re-ranking
models like Cohere re-rank or be-ranked-large and general-
purpose LLMs like GPT (Gao et al. 2023a). Filtering evi-
dence also effectively optimizes evidence quality. There is
another Filter-re-ranker paradigm combining the strengths
of different models (Ma et al. 2023). The smaller model acts
as a filter while the LLM serves as a re-ranking agent. An-
other simple and effective method involves LLMs evaluating
the retrieved content before generating the final answer, al-
lowing the LLM to self-assess and filter out documents with
poor relevance.

61
2
3Reliability analysis
Heterogeneity analysis Extrapolation analysisEvidence retrieval
Generator 
Result:
Answer: A
Explaination:……  Prompt:
You are a board-certified 
physician…….
Input:Prompt +  
question + evidence
Input 
Corpus EvidencePublicat
ion type：
Date：
……
Reliability 
score：6.3
Background
Background: A 46-year-old 
woman comes to the 
physician because of ……
Question: Which of the 
following helps explain the 
condition of this patient?
A.    B.     C.     D.     E.   
Population 
Intervention 
OutcomesSp
Si
SoS4Figure 2: The pipeline of META-RAG includes (1) reliability analysis, (2) heterogeneity analysis, and (3) extrapolation analysis.
Our method incorporates these three stages to re-rank and filter evidence, providing as high-quality evidence as possible to (4)
generator LLM.
Method
Task Definition
To align with the principles of EBM, we aim not only to
deliver convincing answers but also to present high-quality
evidence. We define medical queriesQfrom users as sys-
tem inputs and then respondAand retrieved evidenceE
as the output. As shown in Figure 2, our main pipeline fo-
cuses on the re-ranking and filtering steps of the evidence in
RAG. At the end of the re-ranking and filtering section, we
pass high-quality articles with their orders to the generator.
In this task, we evaluate the evidence across three distinct
dimensions: reliability analysis, heterogeneity analysis, and
extrapolation analysis. These analyses enable us to assess
the reliability of the evidence, exclude untrustworthy find-
ings, and determine whether the results can be applied to the
patient. After re-ranking evidence, the most effective pieces
of evidence and their order are passed to the response model
to generate recommendations for the queries.
Evidence Retrieval
In the first step, we conduct evidence retrieval based on
query similarity with the datasets. However, there are too
many article types in PubMed (White 2020). A substantial
proportion of the articles lack an abstract. To address these
problems, we employ a hybrid retrieval approach. We si-
multaneously search the article titles, abstracts, and MeSH
(Medical Subject Headings) keys in the articles. By calculat-
ing and aggregating the similarity scores across these three
different tags and then ranking them, we ultimately select the
evidenceEwith the highest scores as potential evidence.
Reliability Analysis
After obtaining highly relevant evidence, we first grade the
articles by their fundamental information. As shown in Fig-ure 3, we mainly score the evidenceEwith the rules of the
publication type, publication date, and LLM judgments. Ini-
tially, we access the publication type from the information
and evaluate the evidence quality level. We assign scores
ranging from 1 to 7 based on the medical principles (Polit
and Beck 2004). Recognizing that the publication date of an
article can significantly influence its conclusions, we then
sort the articles by their publication dates. We award an ex-
tra point to the most recently published articles on their base
score. And as the article becomes less recent, the score we
reward gradually decreases in tiers. This process results in
our base score derived from rule-based filtering.
We also employ an LLM for a more fine-grained reliabil-
ity analysis. Meta-analyses typically analyze the randomiza-
tion of literature, data integrity, presence of bias, and choices
regarding blinding. These principles can reflect the validity
of the experimental conclusions in the article. We implement
this method, evaluating the evidence by three questions as
detailed in Figure 3. The detailed architecture of prompts
and questions is provided in the appendix. Ultimately, this
process provides the reliability scorer iwithE i. A largerr i
signify a more rigorous methodology.
Heterogeneity Analysis
After we score each piece of evidence on reliability, we
then apply heterogeneity analysis. This analysis can remove
studies with low quality and high heterogeneity. This step
guarantees that only valid evidence is fed to the generative
model. We perform heterogeneity detection for each article-
claim pair. In this analysis, we apply the definition of het-
erogeneity in the DerSimonian-Laird method (DerSimonian
and Laird 2015) to filter the evidence. Based on the charac-
teristics of this dataset, we approximate part of the model
parameters and define the measurement metric to represent
the stance of each article.First, we create claims by combin-

2025.07  
2023.02Date Score  
5
4Date score (Data) Publication score(Pub)
Experiment score (Exp)
You are a professional medical expert. Please analyze the 
following content and answer the following questions: 
1) Does the article's conclusions have ethnic or biological 
limitations? 
2) Are the experimental results specifically targeted toward a 
particular regional population?
3) Are the conditions for these limitations clearly stated?Meta -
analysis
LetterPublication Score  
7
3
Query 
Retrieved evidence ：
……  
Reliability score(R)= αData + βPub + γExp
R=3.3No，No，Yes
R=2.7
……  
R=5.6Reliability analysis Figure 3: The pipeline of the reliability analysis. We synthe-
size the information and the judgments of LLM to show the
reliability of each evidence.
ing the query with each option. Each option defines a sep-
arate claim. We ask LLMs to determine the stance of each
piece of evidence on each claim. We define the label of each
evidence asy i, and mark these pairs as support, oppose, or
irrelevant.
yi=

1,ifilabeled “Support”,
0,ifilabeled “Oppose”,
NaN,ifilabeled “Irrelevant”.(1)
Then we need to compute the heterogeneity of the evidence
set associated with each query. We compute the random-
effects varianceτ2
DL, the pooled effectθ RE, and the study
weightsw reat this step. We definekas the total number of
studies retrieved for a single query andv ias the variance es-
timate of theithstudy. Most original studies do not report
standard errors, so we setv iasσ. Definew ias the weight of
theithobservation in the fixed-effect model. Then compute
the fixed-effect combined estimate ˆθFE. Formally,
ˆθFE=Pk
i=1wiyiPk
i=1wiwherew i=1
vi(2)
Afterwards, we calculate the heterogeneity statisticQ.
This variable represents the total standard deviation of theentire set of articles. It serves as a preliminary indicator of
the consistency of stances within the article cluster.
Q=kX
i=1wi 
yi−ˆθFE2(3)
Then calculate the heterogeneityτ2
DLunder the DerSimo-
nian–Laird random-effects model. These metrics measure
the fixed-effect estimate dispersion and the true effect vari-
ability across studies. Formally,
τ2
DL= max(
Q−(k−1)
Pk
i=1wi−Pk
i=1w2
iPk
i=1wi,0)
(4)
We useτ2
DLandv ito derive the random-effects weightW i
and the overall estimate ˆθRE. We also compute each study’s
outlier measureQ iand the leave-one-out heterogeneity re-
duction ratio∆ i. Formally,
ˆθRE=Pk
i=1WiyiPk
i=1WiwhereW i=1
vi+τ2
DL(5)
Qi= 
yi−ˆθRE2
vi(6)
Finally, we defineSas the set of allkstudies andS(−i)
as the set obtained by removing studyifromS. As for the
formula 4, we compute theτ2 (−i)
DL for theS(−i)and cal-
culate the decrease caused by this evidence. We define an
acceptable maximum heterogeneity contributionMand a
minimum reliability scoreR c. Based on the final outcomes,
we determine whether each article should be excluded. For-
mally,
∆i=τ2
DL−τ2 (−i)
DL
τ2
DL(7)
Algorithm 1 summarizes this process.
Extrapolation Analysis
To prevent large gaps between the background of the user
and the experimental conditions in the evidence, we adjust
the extrapolation score of each evidence. This process is im-
plemented based on the similarity between them. We di-
vide the process into three clear steps. First, we split the
query into the user background and the clinical question.
Then, we use LLM with a carefully designed prompt to
compare the background information from the query and
the evidence across the population, intervention, and out-
comes(PIO) (Methley et al. 2014). Through this process,
each piece of evidence is assigned a fine-grained score along
each of these dimensions, and the detailed architecture of
this process is provided in the appendix. Finally, we com-
pute an overall extrapolation score for each evidence rela-
tive to the user’s background. We calculate the final ranking
scoreSby both the extrapolation score and the reliability
score.
Algorithm 2 summarizes this process.

Algorithm 1: Heterogeneity Analysis
Input:Queryq, Evidence(E, R) =
{(E1, r1), . . .(E k, rk)}, HyperparameterM, R c
Output:filtered evidenceE f={E 1, . . . E m}
1:vi←σ ▷Initialize
2:c1, c2, . . .← C(q)▷Combine claims
3:E f← {}
4:forc i∈ C(q)do
5:y← G(P 0(c, E i))▷Generate evidence labels
6:τ2
DL←maxD(q, y, v i, k)▷Calculate DL variance
7:fore∈ {E i|i= 0. . . k}do
8:ComputeQ i=M(y i, vi)
9:Compute∆ iby Eq. (7)
10:if∆ i< M∧e /∈E fthen
11:AddetoE f
12:end if
13:if∆ i≥M∧e /∈E f∧ri> Rcthen
14:AddetoE f
15:end if
16:end for
17:end for
18:returnE f ▷Return the filtered evidence
Algorithm 2: Extrapolation Analysis
Input:Queryq, EvidenceE f, Rf =
{(E1, r1), . . .(E f, rf)}, Hyperparameterα, β, γ
Output:Scored evidence(E f, S) =
{(E1, S1), . . .(E m, Sm)}
1:S← {}▷Initialize
2:Back, Que← C(q)▷split the background
3:fore∈ {E j|j= 0. . . m}do
4:T p← G(P 0(Back, E j))▷Generate Population
score
5:T i← G(P 1(Back, E j))▷Generate Intervention
score
6:T o← G(P 2(Back, E j))▷Generate outcome score
7:T j←αT p+βT i+γT o▷Calculate Extrapolation
score
8:S j←r2
jTj ▷Calculate total ranking score
9:end for
10:return(E f, S)▷Return the filtered evidence
Experiments and Results
Experiments Setup
DatasetsIn our experiment, we first select various medi-
cal Q&A datasets and literature databases for our query re-
source. To ensure our experimental results are clear and fair,
we select a five-option multiple-choice Q&A dataset as the
task format. During data selection, we guarantee that each
question includes sufficient patient information to support
extrapolation analysis. We ultimately focus our method on
the MedQA (Jin et al. 2020) and MMLU (He, Fu, and Tu
2019), which contain more comprehensive and professional
user queries. The MedQA dataset typically comprises real
cases that carry patient information, allowing us to perform
c"Systematic Review"
"Meta -Analysis"Level 7
Level 6
Level 5
Level 4
Level 3
Level 2
Level 1"Randomized Controlled Trial",
"Clinical Trial, Phase I", 
"Controlled Clinical Trial", 
"Multicenter Study"
"Observational Study"
"Comparative Study"
"Cohort Study"
"Validation Study"
"Evaluation Study"
"Case Reports", 
"Clinical Trial", 
"Comment", 
"Letter"
"Journal Article", 
"Review", 
"Editorial", 
"Research Support, U.S. Gov't, P.H.S.",
"Research Support, U.S. Gov't, Non -P.H.S.",
"Research Support, Non -U.S. Gov't", 
"Research Support, N.I.H., Extramural"
Others
Weakest evidenceStrongest evidenceFigure 4: We divide the evidence type into 7 levels. In reli-
ability analysis, we categorize evidence from different pub-
lication types and LLM judgments. The higher level of evi-
dence means a better publication type score.
extrapolative analysis more accurately.
Evidence corpusAlso, we take the PubMed dataset as the
literature database, which is widely used in medical meta-
analysis work. This dataset provides a thorough organization
of information from the literature. As shown in Figure 5,
the PubMed data set provides all the information we need
for better retrieval and evaluation. In the step of reliability
analysis, we also divide these articles into different levels
by the rules shown in Figure 4. The top articles have the
strongest evidence grade due to their publication type. Be-
cause the classification of PubMed articles is more detailed
compared to traditional medical evidence grades, the order
in our ranking is based on a fusion of multiple medical field
evidence grading systems. Our ranking primarily follows the
(Polit and Beck 2004) method, categorizing these articles
into seven levels.
Main Result
We select three different baselines to show the performance
of the META. The experiments are calculated based on 5000
queries extracted from the MedQA datasets and 300 queries
extracted from MMLU and MedQA datasets. Due to limited
computational resources, all the evidence used in the experi-
ments is extracted from over four million medical articles in
PubMed. For each query, 15 articles are initially retrieved as
a baseline, and then different methods are used for filtering
and re-ranking.
w/o Evi. To test the base performance of each model,
we give no evidence to the LLM as w/o Evi. This base-
line can test whether the LLM has studied this query. Ac-
tually, some LLMs like GPT-4o-mini reach an accuracy of

methodMedQA MMLU
D1 D2 D3 D4 Best D1 D2 D3 D4 Best
Llama-3.0-8BMeta 44.038.0 40.7 39.344.0 42.742.0 42.0 39.342.7
w/o Evi - - - - 38.7 - - - - 36.2
Ran-Evi 25.3 29.3 31.9 32.6 32.6 25.3 29.3 31.9 32.6 32.6
Self-Evi 38.0 30.0 33.3 28.3 38.0 36.0 40.0 42.7 37.2 42.7
Qwen2.5-7BMeta 51.552.048.5 42.552.0 49.3 46.050.746.750.7
w/o Evi - - - - 49.6 - - - - 49.3
Ran-Evi 44.5 43.5 42.5 43.5 44.5 43.3 43.7 48.4 44.3 48.4
Self-Evi 42.5 39.5 43.5 41.5 41.5 48.0 48.7 48.0 48.4 48.7
Mistral-7BMeta 47.545.0 46.5 46.547.5 45.0 47.348.047.748.0
w/o Evi - - - - 43.5 - - - - 44.0
Ran-Evi 42.0 42.5 40.5 45.5 45.5 43.3 45.0 46.8 45.4 46.8
Self-Evi 42.5 39.5 43.5 41.5 43.5 43.3 44.7 45.3 46.7 46.7
Gemma-1.1-7BMeta 41.0 41.543.040.043.0 36.0 34.7 35.340.0 40.0
w/o Evi - - - - 40.5 - - - - 34.7
Ran-Evi 34.0 31.5 30.0 31.0 34.0 35.3 36.6 35.5 37.1 37.1
Self-Evi 31.0 29.5 30.0 31.0 31.0 34.7 29.3 33.3 34.7 34.7
Table 1: Accuracy (%) of Meta-RAG and other baselines in 300 queries of MedQA and MMLU datasets. w/o Evi: unrelated
evidence provided; Ran-Evi: evidence randomly selected by correlation; Self-Evi: evidence selected by the generator LLM. All
the other LLMs below use the re-rank method like Self-Evi. The numbers D1, D2, D3, and D4 under the dataset name represent
the number of evidence articles provided to the model during generation.
Figure 5: The specific information structures retrieved from
the PubMed dataset. By analyzing this detailed information
of articles, we can comprehensively assess whether the liter-
ature is sufficiently authoritative.
over 90%. This performance may have no increase by the
META method.
Ran-Evi. We provide evidence extracted based on the
LLM as Ran-Evi. We use a random function to shuffle the
extracted evidence, serving as the most straightforward con-
trol group for our method.
Self-Evi. We provide a random order of evidence as the
Self-Evi. This baseline is designed to demonstrate that our
method offers a significant improvement over traditional
LLM-based re-ranking approaches. We utilized the inherent
capability of each large model to rank the relevance of docu-
ments in the evidence pool. The top-ranked documents from
this sorted list were then selected as evidence and providedmethodMedQA
D1 D2 D3 D4 Best
0.5BMeta 25.0024.70 23.92 23.8825.00
w/o - - - - 24.64
Ran- 23.80 23.50 23.62 23.80 23.80
Self- 23.50 23.78 24.00 23.34 24.00
1.5BMeta 28.42 30.26 30.5630.82 30.82
w/o - - - -35.08
Ran- 26.52 28.84 28.12 27.40 28.84
Self- 25.98 28.48 27.94 27.08 28.48
7BMeta 50.7651.0450.74 50.5651.04
w/o - - - - 50.58
Ran- 47.04 47.08 46.66 47.18 47.18
Self- 49.70 49.12 49.52 49.18 49.70
14BMeta 59.20 60.00 60.7260.90 60.90
w/o - - - - 58.36
Ran- 56.58 56.76 56.32 56.94 56.94
Self- 55.48 55.90 55.88 55.66 55.90
32BMeta 63.28 64.08 64.0064.32 64.32
w/o - - - - 62.06
Ran- 59.14 60.24 60.30 60.10 60.30
Self- 59.46 60.24 60.10 60.18 60.24
Table 2: Accuracy (%) of the responses of different sizes
of Qwen-2.5 LLMs. All the responses are based on 5000
queries of the MedQA datasets.
to the generation model.
Experimental results are presented in Table 1 and Table 2.
Table 1 shows the performance of different types of LLMs,
while Table 2 shows the performance of different sizes. We

can get the following analysis:
1) Our method consistently improves upon the baselines
of almost all LLMs. Across LLMs of different sizes and
types, our Meta-RAG achieves substantial improvements
over traditional evidence-ranking methods.
2) In the response by Qwen-2.5, we observe a clear ac-
curacy gain when the model answers directly. In the1.5B
model, w/o Evi reaches an accuracy of 35.08%. Both Ran-
Evi and Self-Evi, which include additional evidence, per-
form worse than w/o Evi. One possible explanation for this
phenomenon is that the Qwen model has memorized this
particular question, so the extra input tokens harm the per-
formance. For other differently sized models, however, our
high-quality evidence still yields improved accuracy.
3) When given different amounts of evidence, our method
delivers steadily increasing performance on stronger mod-
els. Small models suffer significant drops when exposed to
too many input tokens. As a result, Table 1 shows large over-
all performance fluctuations. However, for Qwen-14B and
Qwen-32B, adding more evidence consistently improves ac-
curacy, indicating that our evidence has low heterogeneity.
Ablation Study
To validate the efficiency of each step, we set multiple ab-
lation experiments. For the reliability ablation experiments,
we ablate the reliability analysis module of the model. We
set all the reliability scores sent to the heterogeneity analy-
sis same. When calculating the highest-scoring evidence for
heterogeneity analysis, we randomly select the first piece of
evidence from the list. After conducting this ablation, we ob-
serve that both the quality of the evidence and the accuracy
of the responses decrease, as shown in Table 3.
methodMedQA
1 2 3 4 Best
Meta 44.038.0 40.7 39.344.0
w/o R 36.7 40.0 34.0 37.3 40.0
w/o H 34.0 38.7 37.3 35.3 38.7
w/o E 34.0 33.3 34.7 34.7 34.7
Table 3: Ablation of each part of the Metaw/o Rmeans no
reliability checking and all the reliability scores set to 1.w/o
Hmeans no heterogeneity analysis.w/o Emeans no extrap-
olation analysis
Subsequently, we analyze the impact of heterogeneity on
the model. As shown in Table 3, we remove the heterogene-
ity judgment process and directly re-rank the extracted ev-
idence based on reliability and extrapolation. We observe
a noticeable decline in the evidence contribution score and
accuracy. This phenomenon suggests that some highly reli-
able but paradoxical evidence scores remain in the evidence
and mislead the judgment of the generation model. The gen-
eration model then produces responses closer to the incor-
rect options. Since our calculation of evidence grades in-
volves selecting both the evidence grade and the similarity
of the evidence to each option for assessment, this aspect
Figure 6: The similarity of each method between the pro-
vided evidence and the ground-truth answer. We use this
metric to evaluate whether Meta-RAG can better guide the
model to the correct answer.
inevitably becomes affected when heterogeneity scoring is
removed.
Is the Evidence Better?
As shown in Figure 6, we employ another evaluation met-
ric to evaluate the evidence quality. We assess the similarity
of our input evidence and the claim with the gold option to
show the contribution to a correct answer. The higher simi-
larity means better evidence is provided. We observe that the
average quality of the evidence is effectively enhanced af-
ter Meta-RAG. Additionally, we can also analyze that as the
model size grows, the Self-Evi group becomes more sen-
sitive to good evidence. In addition, the self-ranking abil-
ity varies between LLMs. Our experiments show that some
models can select higher-quality evidence. Yet they do not
achieve higher accuracy when using self-selected evidence.
We believe that this happens because, although the chosen
evidence is more similar to the correct answer, it is also con-
troversial. That controversy raises the similarity to incorrect
answers. As a result, the generative model becomes con-
fused. Therefore, most baselines cannot surpass Meta-RAG.
Conclusion
EBM currently needs robust automated tools to assist in
medical tasks. However, existing RAG for EBM cannot
ensure the evidence meets the stringent requirements of
medicine. Therefore, inspired by the principles of meta-
analysis, we propose a META-RAG filtering and re-ranking
method to ensure the evidence is effective and reliable. We
conduct practical experiments on our method and verify its
improvements in accuracy and evidence quality. We hope
this work will assist researchers in the medical field, promot-
ing safer and more effective deployment of LLMs in medical
applications.

References
Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.;
Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.;
Anadkat, S.; et al. 2023. Gpt-4 technical report.arXiv
preprint arXiv:2303.08774.
Alam, F.; Giglou, H. B.; and Malik, K. M. 2023. Auto-
mated clinical knowledge graph generation framework for
evidence based medicine.Expert Systems with Applications,
233: 120964.
Borenstein, M.; Hedges, L. V .; Higgins, J. P.; and Rothstein,
H. R. 2021.Introduction to meta-analysis. John Wiley &
Sons.
Clusmann, J.; Kolbinger, F. R.; Muti, H. S.; Carrero, Z. I.;
Eckardt, J.-N.; Laleh, N. G.; L ¨offler, C. M. L.; Schwarzkopf,
S.-C.; Unger, M.; Veldhuizen, G. P.; et al. 2023. The future
landscape of large language models in medicine.Communi-
cations Medicine, 3(1): 141.
DerSimonian, R.; and Laird, N. 2015. Meta-analysis in clin-
ical trials revisited.Contemporary clinical trials, 45: 139–
145.
Djulbegovic, B.; and Guyatt, G. H. 2017. Progress in
evidence-based medicine: a quarter century on.The lancet,
390(10092): 415–423.
Egger, M.; Smith, G. D.; and Phillips, A. N. 1997. Meta-
analysis: principles and procedures.Bmj, 315(7121): 1533–
1537.
Friedman, C.; Rindflesch, T. C.; and Corn, M. 2013. Natural
language processing: state of the art and prospects for sig-
nificant progress, a workshop sponsored by the National Li-
brary of Medicine.Journal of biomedical informatics, 46(5):
765–773.
Gao, Y .; Sheng, T.; Xiang, Y .; Xiong, Y .; Wang, H.; and
Zhang, J. 2023a. Chat-rec: Towards interactive and explain-
able llms-augmented recommender system.arXiv preprint
arXiv:2303.14524.
Gao, Y .; Xiong, Y .; Gao, X.; Jia, K.; Pan, J.; Bi, Y .; Dai, Y .;
Sun, J.; and Wang, H. 2023b. Retrieval-augmented gener-
ation for large language models: A survey.arXiv preprint
arXiv:2312.10997.
Hansen, C.; Steinmetz, H.; and Block, J. 2022. How to con-
duct a meta-analysis in eight steps: a practical guide.
He, J.; Fu, M.; and Tu, M. 2019. Applying deep matching
networks to Chinese medical question answering: a study
and a dataset.BMC medical informatics and decision mak-
ing, 19: 91–100.
Huang, L.; Yu, W.; Ma, W.; Zhong, W.; Feng, Z.; Wang, H.;
Chen, Q.; Peng, W.; Feng, X.; Qin, B.; et al. 2023. A sur-
vey on hallucination in large language models: Principles,
taxonomy, challenges, and open questions.arXiv preprint
arXiv:2311.05232.
Jin, D.; Pan, E.; Oufattole, N.; Weng, W.-H.; Fang, H.; and
Szolovits, P. 2020. What Disease does this Patient Have?
A Large-scale Open Domain Question Answering Dataset
from Medical Exams.arXiv preprint arXiv:2009.13081.Joshi, M.; Chen, D.; Liu, Y .; Weld, D. S.; Zettlemoyer, L.;
and Levy, O. 2020. Spanbert: Improving pre-training by rep-
resenting and predicting spans.Transactions of the associa-
tion for computational linguistics, 8: 64–77.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; et al. 2020. Retrieval-augmented generation for
knowledge-intensive nlp tasks.Advances in Neural Infor-
mation Processing Systems, 33: 9459–9474.
Li, J.; Deng, Y .; Sun, Q.; Zhu, J.; Tian, Y .; Li, J.; and Zhu,
T. 2024. Benchmarking large language models in evidence-
based medicine.IEEE Journal of Biomedical and Health
Informatics.
Lipsey, M. W. 2001. Practical meta-analysis.Thousand
Oaks.
Ma, Y .; Cao, Y .; Hong, Y .; and Sun, A. 2023. Large lan-
guage model is not a good few-shot information extrac-
tor, but a good reranker for hard samples!arXiv preprint
arXiv:2303.08559.
McMurray, J. J.; and Packer, M. 2021. How should we se-
quence the treatments for heart failure and a reduced ejection
fraction? A redefinition of evidence-based medicine.Circu-
lation, 143(9): 875–877.
Methley, A. M.; Campbell, S.; Chew-Graham, C.; McNally,
R.; and Cheraghi-Sohi, S. 2014. PICO, PICOS and SPI-
DER: a comparison study of specificity and sensitivity in
three search tools for qualitative systematic reviews.BMC
health services research, 14(1): 1–10.
Nadkarni, P. M.; Ohno-Machado, L.; and Chapman, W. W.
2011. Natural language processing: an introduction.Jour-
nal of the American Medical Informatics Association, 18(5):
544–551.
Polit, D. F.; and Beck, C. T. 2004.Nursing research: Prin-
ciples and methods. Lippincott Williams & Wilkins.
Sackett, D.; Richardson, W.; and Rosenberg, W. 2008. What
is evidence-based medicine (EBM).Patient care model, 36:
26–33.
Singhal, K.; Azizi, S.; Tu, T.; Mahdavi, S. S.; Wei, J.; Chung,
H. W.; Scales, N.; Tanwani, A.; Cole-Lewis, H.; Pfohl, S.;
et al. 2023. Large language models encode clinical knowl-
edge.Nature, 620(7972): 172–180.
Subbiah, V . 2023. The next generation of evidence-based
medicine.Nature medicine, 29(1): 49–58.
White, J. 2020. PubMed 2.0.Medical reference services
quarterly, 39(4): 382–387.
Zhang, Y .; Li, Y .; Cui, L.; Cai, D.; Liu, L.; Fu, T.; Huang,
X.; Zhao, E.; Zhang, Y .; Chen, Y .; et al. 2023. Siren’s song
in the AI ocean: a survey on hallucination in large language
models.arXiv preprint arXiv:2309.01219.