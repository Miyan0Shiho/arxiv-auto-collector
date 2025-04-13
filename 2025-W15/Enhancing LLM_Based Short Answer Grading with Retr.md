# Enhancing LLM-Based Short Answer Grading with Retrieval-Augmented Generation

**Authors**: Yucheng Chu, Peng He, Hang Li, Haoyu Han, Kaiqi Yang, Yu Xue, Tingting Li, Joseph Krajcik, Jiliang Tang

**Published**: 2025-04-07 17:17:41

**PDF URL**: [http://arxiv.org/pdf/2504.05276v1](http://arxiv.org/pdf/2504.05276v1)

## Abstract
Short answer assessment is a vital component of science education, allowing
evaluation of students' complex three-dimensional understanding. Large language
models (LLMs) that possess human-like ability in linguistic tasks are
increasingly popular in assisting human graders to reduce their workload.
However, LLMs' limitations in domain knowledge restrict their understanding in
task-specific requirements and hinder their ability to achieve satisfactory
performance. Retrieval-augmented generation (RAG) emerges as a promising
solution by enabling LLMs to access relevant domain-specific knowledge during
assessment. In this work, we propose an adaptive RAG framework for automated
grading that dynamically retrieves and incorporates domain-specific knowledge
based on the question and student answer context. Our approach combines
semantic search and curated educational sources to retrieve valuable reference
materials. Experimental results in a science education dataset demonstrate that
our system achieves an improvement in grading accuracy compared to baseline LLM
approaches. The findings suggest that RAG-enhanced grading systems can serve as
reliable support with efficient performance gains.

## Full Text


<!-- PDF content starts -->

Enhancing LLM-Based Short Answer Grading with
Retrieval-Augmented Generation
Yucheng Chu
Michigan State University
chuyuch2@msu.eduPeng He
Washington State University
peng.he@wsu.eduHang Li
Michigan State University
lihang4@msu.edu
Haoyu Han
Michigan State University
hanhaoy1@msu.eduKaiqi Y ang
Michigan State University
kqyang@msu.eduYu Xue
Washington State University
yu.xue@wsu.edu
Tingting Li
Washington State University
tingting.li1@wsu.eduJoseph Krajcik
Michigan State University
krajcik@msu.eduJiliang Tang
Michigan State University
tangjili@msu.edu
ABSTRACT
Short answer assessment is a vital component of science
education, allowing evaluation of students’ complex three-
dimensional understanding. Large language models (LLMs)
that possess human-like ability in linguistic tasks are increas-
ingly popular in assisting human graders to reduce their
workload. However, LLMs’ limitations in domain knowl-
edge restrict their understanding in task-specific require-
ments and hinder their ability to achieve satisfactory per-
formance. Retrieval-augmented generation (RAG) emerges
as a promising solution by enabling LLMs to access rele-
vant domain-specific knowledge during assessment. In this
work, we propose an adaptive RAG framework for auto-
mated grading that dynamically retrieves and incorporates
domain-specific knowledge based on the question and stu-
dent answer context. Our approach combines semantic search
and curated educational sources to retrieve valuable refer-
ence materials. Experimental results in a science education
dataset demonstrate that our system achieves an improve-
ment in grading accuracy compared to baseline LLM ap-
proaches. The findings suggest that RAG-enhanced grading
systems can serve as reliable support with efficient perfor-
mance gains.
Keywords
Automated Short Answer Grading, LLM, RAG, Learning
Assessments, Constructed Responses
1. INTRODUCTION
Assessment and analysis of student understanding in science
education extend far beyond simple grading. With the emer-
gence of new frameworks for K-12 science education, there is
an increasing emphasis on analyzing students’ multidimen-sional comprehension of scientific understanding. The Na-
tional Research Council’s Framework for K-12 Science Edu-
cation Council et al. [2012] established three critical dimen-
sions of science learning: disciplinary core ideas (DCIs), sci-
ence and engineering practices (SEPs), and crosscutting con-
cepts (CCCs). To demonstrate their achievement, students
should apply their 3D understanding to interpret compelling
phenomena or solve real-world problems. This framework
has fundamentally transformed how we assess student learn-
ing, shifting away from traditional multiple-choice questions
toward short-answer assessments that better capture stu-
dents’ authentic understanding He et al. [2023b]. Short-
answer questions have become particularly valuable in this
context, as they require students to demonstrate their three-
dimensional knowledge by explaining real-world phenomena
and solving complex problems He et al. [2023a]. These re-
sponses provide rich insights into students’ conceptual un-
derstanding, revealing both their mastery and misunder-
standing, which is essential for teachers to adapt their in-
structional strategies and support students’ self-regulated
learning He et al. [2024].
However, the analysis of these short answers presents sig-
nificant challenges as it is time-intensive, requires deep ex-
pertise across all three dimensions, and makes it difficult
for teachers to provide timely, actionable feedback to in-
dividual students. Fortunately, artificial intelligence (AI),
particularly automatic short answer grading (ASAG) sys-
tems, has emerged as a promising solution to these chal-
lenges. Traditional ASAG approaches using machine learn-
ing (ML) techniques have shown success in providing con-
sistent and objective scoring but are limited by their re-
liance on large training datasets Burrows et al. [2015], Sul-
tan et al. [2016]. The recent advent of large language models
(LLMs) has opened new possibilities for training ASAG sys-
tems on smaller datasets, due to its possession of broad gen-
eral knowledge from pre-training. However, current LLM-
based approaches face two critical limitations in science ed-
ucation assessment. First, they exhibit deceptive perfor-
mance in technical domains, often generating plausible-sounding
but scientifically false evaluations when in-depth domain
knowledge is required. Their output can be unreliable when
evaluating answers that contain ambiguous terminology orarXiv:2504.05276v1  [cs.CL]  7 Apr 2025

complex scientific concepts. Second, they lack sufficient
task-specific knowledge about the grading criteria Lewis et al.
[2020]. They often fail to understand the nuanced require-
ments of specific educational frameworks such as the three-
dimensional learning approach, which can lead to misalign-
ment between automated scores and expert evaluations.
To address these limitations, we introduce GradeRAG, a
novel framework that enhances LLM-based grading through
retrieval-augmented generation (RAG) Zeng et al. [2024].
Our approach implements a specialized RAG pipeline that
provides the LLM with access to curated domain-specific
knowledge bases, enabling a more accurate assessment of
scientific concepts across the three dimensions (DCIs, SEPs,
and CCCs). This knowledge-focused approach is partic-
ularly crucial in science education, where effective assess-
ment requires domain-specific expertise of specialized con-
cepts and their interconnections He et al. [2023a], Beatty
et al. [2014]. As part of our knowledge retrieval strategy,
we incorporate expert-annotated scoring rationales as a spe-
cialized form of knowledge source. These examples con-
tain detailed scoring rationales that explicitly identify crit-
ical scientific terminology and reasoning patterns in stu-
dent responses. By treating these examples as retrievable
knowledge, LLMs are guided to emulate expert analysis pro-
cesses, further improving scoring performance. We evaluate
GradeRAG on a dataset of student short answers on science
assessments. Results demonstrate improvements in both
grading accuracy and consistency compared to the baseline
approaches. Our results suggest that integrating special-
ized knowledge retrieval systems can bridge the gap between
automated efficiency and expert-level assessment in science
education, leading to more reliable evaluation of complex
scientific understanding.
2. RELATED WORK
Automatic short answer grading. Automatic short an-
swer grading (ASAG) has evolved significantly over the past
decades. Early approaches mainly focus on text matching
and statistics-based methods Mohler et al. [2011]. With the
advance in ML, later systems employ feature engineering
techniques with supervised training methods Leacock and
Chodorow [2003]. These traditional approaches typically
require large training datasets and often struggle with cap-
turing the semantic differences in student answers. Recent
works utilizing neural networks demonstrate improvement in
understanding semantic relationships in short answers Has-
san et al. [2018]. Large language models (LLMs) further en-
able enhanced capabilities in ASAG through zero-shot and
few-shot Lee et al. [2024] techniques. However, as noted
by Li et al. [2023], LLMs still face challenges in special-
ized domains like science education, where domain-specific
knowledge is crucial for accurate assessment.
Retrieval-augmented generation. Retrieval-augmented gen-
eration (RAG) has emerged as an effective approach to en-
hance LLMs’ reasoning ability by providing access to ex-
ternal knowledge sources. RAG has demonstrated effec-
tiveness in knowledge-intensive tasks requiring factual ac-
curacy Lewis et al. [2020] and domain expertise Guu et al.
[2020]. In the educational context, the application of RAGremains relatively unexplored, with most works focusing on
content generation Miladi et al. [2024] rather than assess-
ment. Recent studies have begun investigating RAG for
automated assessment. For example, Harshavardhan and
Singh [2024] proposes a rubric-centric approach for auto-
mated test correction using RAG, while Sundar et al. [2024]
evaluated RAG frameworks for grading open-ended written
responses. However, the application of RAG, specifically for
short answer grading in science education, where domain-
specific knowledge across multiple dimensions (DCIs, SEPs,
CCCs) is crucial, represents a novel contribution to our
work.
3. METHOD
3.1 Problem Statement
The task of automatic short answer grading (ASAG) in-
volves assigning appropriate scores to open-ended student
responses. Given a student answer x, an ASAG system maps
it to one of Cpredefined score levels {s1, ..., s C}. Formally,
an ASAG system Fgenerates score predictions ˆ yi=F(xi)
for each response xiacross each dimension. This work ex-
tends traditional ASAG through retrieval-augmented gen-
eration. We leverage two types of external knowledge re-
sources to enhance grading performance. First, we utilize
domain-specific knowledge from an external database KD,
retrieving relevant background information Ithat contains
educational materials and scoring guidelines. Second, we in-
corporate a specialized knowledge collection KEofNexpert-
annotated graded examples E={(xi, ri, yi)|i= 1, ..., N},
where each entry consists of a student response xi, a scoring
rationale ri, and the corresponding ground truth score yi.
By combining these knowledge sources, our enhanced ASAG
system F(xi, I,E) creates a more comprehensive context for
LLM-as-a-Grader’s accurate assessment.
3.2 RAG-Based Grading System
In this section, we present GradeRAG, a framework that
combines in-context learning with RAG for automatic short
answer grading in science education.
3.2.1 Knowledge Base Construction
As shown in Figure 1, our system constructs a triple-index
knowledge base optimized for scientific assessment. KD1
andKD2consists of project-related information and grad-
ing guidelines, while KE(which is optinal) contains expert-
annotated scoring examples. The first index KD1contains
general assessment documents such as the Framework of K-
12 Science Education Council et al. [2012] and the NGSS
standard document States [2013]. These materials are sys-
tematically chunked into coherent segments of approximately
512 tokens, which balances content completeness with the
retrieval granularity. This chunking size preserves sufficient
context while enabling precise retrieval of relevant concepts,
as smaller chunks would fragment scientific explanations while
larger chunks can introduce excessive irrelevant information.
Each chunk is tagged with metadata including the document
source and dimension type to facilitate efficient filtering dur-
ing retrieval.
The second index KD2consists of task-specific materials,
such as the three-dimensional learning progression materi-
als He et al. [2023a], which includes unpacking DCIs, SEPs,

Figure 1: An illustration of the proposed GradeRAG frame-
work.
and CCCs materials and the holistic levels in the three as-
sessment tasks. For these more specialized resources, we em-
ploy a finer-grained, manual chunking strategy. Specifically,
we divide the content into single sentences or meaningful seg-
ments containing one or two bullet points. This approach
is adopted as task-specific materials contain dense, concen-
trated information where even small segments can provide
critical assessment criteria. With this fine-grained chunk-
ing strategy, our system can retrieve precisely the relevant
assessment standards without including extraneous informa-
tion. Each chunk is tagged with metadata indicating the di-
mension type, level number, and chunk identifier to enable
targeted filtering during retrieval.
The third index KEcomprises expert-annotated examples,
each encoded with metadata such as score assignments and
dimension-specific annotations. These examples are main-
tained as complete units rather than being chunked, which
preserves the complete reasoning logic of expert scoring.
This design choice ensures that our framework can access
the full scoring demonstration with coherent guidance for
emulating human-like reasoning processes, thus improving
the alignment between model scores and human evaluation.
3.2.2 Dual Retrieval Strategy
For each student response x, GradeRAG employs a dual
retrieval mechanism to gather complementary information.
Knowledge Retrieval. The system first identifies relevant
dimension-specific content using a task-aware retrieval strat-
egy. Each task is associated with specific dimension levels
(e.g., Task 1 corresponds to DCI Level 1 and 2, while Task
2 encompasses Level 4, 5 and 6). This retrieval mechanism
operates in two steps. First, it retrieves dimension-specific
materials using a semantic similarity search over the indexed
content that corresponds to different levels. The top- kper-
tinent information is retrieved. Second, to optimize rele-vance to the student answer while accounting for the con-
text window constraint, we implement a reranking mech-
anism that combines three weighted components: seman-
tic similarity (40% weight), text matching based on word
overlap (30% weight), and domain-specific concept coverage
(30% weight). We deliberately combine these three scores
to address the different aspects of information relevance in
science education assessment. Semantic similarity captures
contextual relevance and identifies materials that share sim-
ilar meanings. Text matching based on normalized word
overlap between the student response and knowledge chunks
can identify the lexical connections. This ensures that spe-
cific terminology in student answers is represented in the
retrieved content. Lastly, the concept matching score eval-
uates the coverage of key scientific concepts necessary for
assessment, such as properties, substances, chemical reac-
tions, and identifications in the knowledge chunk. The final
top-kpassages are selected based on this combined score.
By weighting and combining these three signals, our system
achieves more comprehensive retrieval.
Example Retrieval. In parallel with knowledge retrieval,
our system retrieves similar expert-annotated examples us-
ing semantic similarity matching. These examples include
expert rationales that explain the scoring criteria and high-
light the critical keywords in student responses that signal
understanding.
Figure 2 shows a concrete example of the expert-annotated
grading sample. We incorporate demonstrations (human-
graded examples) into the grading prompt using a scale-
based approach. The number of demonstrations scales with
the complexity of the scoring scheme. For binary classifica-
tion tasks with two score classes, we utilize six demonstra-
tions, while for ternary classification tasks with three score
classes nine demonstrations are employed. This scaling en-
sures sufficient examples across all possible score categories
while maintaining prompt efficiency.
3.2.3 Grading
After retrieving relevant knowledge and examples, GradeRAG
performs the grading process. For each student response,
GradeRAG assembles a unified grading context by com-
bining the original expert-designed task guidelines, the re-
trieved top- kknowledge chunks, and similar expert-annotated
examples with their rationales. This comprehensive con-
tent combination augments the expert-designed base grad-
ing guideline, providing additional context and criteria for
more accurate assessment. The combined prompt (shown
in Figure 3) is passed to the LLM-powered Grader agent
for score generation. Formally, the Grader generates ˆ y1:T=
F(x, K, E ) where ˆ y1:T−1represents the reasoning rationale
containing T−1 tokens that explain the grading decision,
and ˆyTis the final token that represents the numerical score.
The input consists of the student answer x, retrieved knowl-
edge chunks K, and expert examples E. The scoring process
is performed separately for each dimension to ensure a fo-
cused evaluation of the different aspects of scientific under-
standing. The Grader’s output includes both a numerical
score ˆ yTindicating the student’s achievement level and a
detailed explanation ˆ y1:T−1that references the specific evi-
dence from the student response and connects it to the re-

Examplar of Expert-Annotated Sample with Scoring
Rationale (SEP)
Student’s Short Answer: When coconut oil is
mixed with lye, a chemical reaction occurs because
soap and glycerol are new substances. From the
table, I found that the odor, density, solubility in
water, and melting point are different from each
other. They are properties that can be used to
identify substances and whether a chemical reaction
occurs.
Score: SEP-1
Scoring Rationale: “When coconut oil is mixed with
lye, a chemical reaction occurs because soap and
glycerol are new substances.” – this part meets the
partial SEP criteria that mentioned a descriptive
explanation, including a claim of a chemical reac-
tion occurs and observed evidence of the data before
and after the process. However, the response did
not use the evidence to connect to the phenomenon
– new substance produced and a chemical reaction
occurred.
Figure 2: An example of expert-elaborated rationales for a
short-answer sample on the SEP dimension.
trieved knowledge. This approach simulates the expert an-
notation process to enhance grading alignment.
Prompt for Grader
Task: <Task Description >
Question: Write a scientific explanation about
whether a chemical reaction occurs in the described
scenario. Make sure your explanation includes a
claim, evidence, and reasoning.
Step 1 : Review and apply the following detailed
learning goals and scoring criteria for grading stu-
dent answer:
•Criteria: <Initial Expert Criteria >
•Knowledge Materials: <Retrieved Knowledge >
Step 2 : Examine following example graded answers.
Analyze how each one is assessed as explained in
scoring rationales: <Retrieved Examples >
Step 3 : Now assess the fulfillment of student answer
based on the description of task, question, criteria,
and graded examples. Give a score of 0, 1, or 2 with
your reasoning: <Student Answer >
Figure 3: An example prompt for grader
4. EXPERIMENT
This section presents our experimental evaluation for our
GradeRAG system. Specifically, we intend to investigatewhether incorporating external knowledge through RAG im-
proves automated scoring performance compared to the base-
line approaches. We further investigate whether GradeRAG
is compatible with in-context learning by conducting an ab-
lation study.
4.1 Dataset
Our evaluation uses a testing dataset Dtestcomprising stu-
dent responses collected from two middle schools in the Mid-
western United States. The dataset includes student re-
sponses for each of the three NGSS-aligned science tasks,
completed in standard classroom settings. The detailed re-
sponse count for Dtestis reported in Table 1. To estab-
lish ground truth scores, two content experts independently
graded all responses using standardized rubrics and demon-
stration examples. Any scoring discrepancies were resolved
through expert discussion to ensure high-quality gold stan-
dard annotations. Table 1 shows the details of level coverage
for each question task.
Table 1: Detailed statistics of different questions in dataset
Dtest. The number of samples in each label category is shown
asCi.
Question Assessment LevelResponse Count
Total C1 / C2 / C3
Q1DCI (0, 1, 2) 28/14/5
SEP (0, 1, 2) 47 44/3/0
CCC (0, 1, 2) 44/3/0
Q2DCI (0, 4, 5, 6) 13/15/1/2
SEP (0, 1, 2) 31 12/3/16
CCC (0, 1, 2) 10/18/3
Q3DCI (0, 6, 7) 40/3/3
SEP (0, 1, 2, 3) 46 12/26/5/3
CCC (0, 1, 2, 3) 14/1/28/3
4.2 Experimental Setting
We implement our grading system using gpt-4o-mini-2024-
07-18 as the core Grader agent. To ensure reproducibility
and consistent evaluation, we set the temperature parame-
ter to 0, eliminating stochastic variations in the model’s out-
put. The embedding model used for indexing documents is
text-embedding-ada-002. Since the original student answers
were collected in image format, we first process them using
gpt-4-vision-preview for transcription. These transcribed re-
sponses are then passed to the Retriever and Grader agent
for separately assessing each task dimension (e.g., DCI, SEP,
CCC). We run experiments for testing our system under two
experimental conditions: NonRAG and GradeRAG. Under
naive prompt, the system processes student responses us-
ing only the basic grading rubric without additional context
or examples from the RAG component. Under GradeRAG
with zero-shot prompting, the system incorporates the re-
trieved materials as described in Section 3.2, but without
expert-annotated examples.
We evaluate performance using standard metrics, includ-
ing accuracy, weighted F1-score, and Cohen’s kappa. To
be specific, Accuracy ( Acc) measures the proportion of cor-
rect predictions. Weighted F1-score (F1 weighted ) accounts

for class imbalance by computing the F1-score for each class
and taking their weighted average. Cohen’s kappa ( κ) mea-
sures the inter-rater reliability by accounting for the possi-
bility of agreement occurring by chance. The formulas for
these metrics are as follows:
Acc=1
NNX
i=1I(yi=ˆyi), κ=po−pe
1−pe,
F1weighted =X
k= 1Knk·2·Pk·Rk
Pk+Rk
where pois the observed agreement between the model and
expert scores, peis the expected agreement by chance, nk
is the proportion of samples in class k,PkandRkare the
precision and recall for class k.
Table 2: Performance comparison between NonRAG and
GradeRAG on Dtestwith zero-shot prompting (shot num-
berC=0, retrieval size k=4).
QuestionNonRAG GradeRAG
DCI SEP CCC DCI SEP CCC
Accuracy (Acc)
Q1 0.217 0.717 0.739 0.348 0.804 0.957
Q2 0.355 0.387 0.581 0.484 0.419 0.645
Q3 0.717 0.478 0.565 0.783 0.522 0.674
Weighted F1 Score (F1)
Q1 0.170 0.799 0.816 0.257 0.858 0.935
Q2 0.399 0.463 0.544 0.527 0.499 0.595
Q3 0.752 0.443 0.543 0.792 0.481 0.656
Cohen’s Kappa (κ)
Q1 -0.001 0.188 0.074 0.100 0.119 0.000
Q2 0.129 0.232 0.188 0.245 0.260 0.288
Q3 0.124 0.083 0.123 0.196 0.174 0.331
4.3 Main Results
Table 2 presents the comparative performance of our pro-
posed GradeRAG framework against the non-retrieval base-
line (NonRAG) across three questions ( Q1,Q2, and Q3) and
three assessment dimensions (DCI, SEP, and CCC). The re-
sults demonstrate general performance improvements with
GradeRAG across metrics, questions, and dimensions. For
all three metrics (accuracy, weighted F1 score, and Cohen’s
Kappa), GradeRAG outperforms the non-retrieval baseline
in most cases. The most substantial gains are observed in
Q1for the DCI dimension (13.1% accuracy increase) and
CCC dimension (21.8% accuracy increase), with Q3’s CCC
dimension showing the largest Kappa improvement (20.8%
increase). While the Kappa values remain relatively low
overall (ranging from 0.000 to 0.331), GradeRAG generally
achieves better alignment with expert grading patterns com-
pared to NonRAG.
Analyzing performance across questions reveals patterns re-
lated to question complexity. Q1, which has a relatively
simple assessment structure with three levels (0, 1, 2) acrossTable 3: Performance comparison between NonRAG and
GradeRAG under in-context learning (ICL) on Dtest.
QuestionNonRAG-ICL GradeRAG-ICL
DCI SEP CCC DCI SEP CCC
shot number C= 3, retrieval size k= 1
Accuracy (Acc)
Q1 0.609 0.543 0.826 0.609 0.565 0.913
Q2 0.742 0.226 0.387 0.742 0.323 0.548
Q3 0.674 0.500 0.304 0.696 0.587 0.348
Weighted F1 Score (F1)
Q1 0.591 0.664 0.875 0.599 0.682 0.927
Q2 0.717 0.229 0.315 0.742 0.340 0.537
Q3 0.726 0.473 0.300 0.741 0.555 0.327
Cohen’s Kappa (κ)
Q1 0.230 0.087 0.281 0.303 0.094 0.292
Q2 0.536 0.055 0.394 0.554 0.187 0.259
Q3 0.170 0.108 0.142 0.189 0.231 0.142
shot number C= 6, retrieval size k= 2
Accuracy (Acc)
Q1 0.609 0.696 0.891 0.630 0.717 0.913
Q2 0.677 0.290 0.419 0.774 0.355 0.419
Q3 0.761 0.413 0.391 0.761 0.413 0.348
Weighted F1 Score (F1)
Q1 0.584 0.786 0.913 0.609 0.798 0.927
Q2 0.665 0.320 0.333 0.758 0.395 0.333
Q3 0.785 0.400 0.317 0.784 0.392 0.272
Cohen’s Kappa (κ)
Q1 0.210 0.053 0.238 0.276 0.178 0.292
Q2 0.434 0.139 0.120 0.600 0.213 0.120
Q3 0.259 -0.011 0.129 0.259 -0.006 0.068

all dimensions and highly imbalanced class distributions,
demonstrates the largest average performance gains (14.5%
in accuracy). This suggests that for questions with clear-
cut scoring criteria and imbalanced response distributions,
retrieved knowledge can provide particularly effective guid-
ance. In contrast, Q2andQ3involve more complex assess-
ment criteria with up to four score levels and more balanced
class distributions, resulting in more moderate performance
improvements. In general, this variety in complexity levels
across questions suggests how retrieval effectiveness may de-
pend on both the nature of the question and the distribution
of student responses across performance levels.
Across dimensions, we observe that CCC assessments gen-
erally benefit most from knowledge augmentation, with an
average improvement of 13.0% in accuracy. This aligns with
expectations as crosscutting concepts often require broader
contextual understanding that can be enhanced through re-
trieved knowledge. The DCI dimension shows the second-
largest improvements, while SEP shows more modest gains,
suggesting that scientific practices may be more challenging
to evaluate through retrieved content alone.
4.4 Ablation Study
In this section, we conduct two ablation experiments. First,
to investigate whether the proposed GradeRAG is compat-
ible with in-context learning (ICL), we conducted ablation
experiments comparing NonRAG-ICL and GradeRAG-ICL
across different shot settings. Tables 3 presents our findings
with 3-shot and 6-shot settings.
Effect of Retrieved Knowledge Across Shot Settings.
Comparing across shot settings reveals several patterns. In
the zero-shot scenario, adding retrieved knowledge consis-
tently improves performance across nearly all dimensions
and questions. For 3-shot learning, the addition of knowl-
edge retrieval also provides benefits, with GradeRAG-ICL
outperforming NonRAG-ICL in 7 out of 9 dimension-question
combinations. For DCI in both Q1andQ2, retrieval pro-
vides no additional benefit, suggesting the examples may
already contain sufficient domain knowledge. With 6-shot
learning, GradeRAG-ICL outperforms NonRAG-ICL in only
5 out of 9 cases. For Q3, knowledge retrieval provides no ac-
curacy improvement in any dimension, even with a minor
drop in CCC. This suggests that with sufficient examples,
additional knowledge becomes redundant for this particular
question.
Effect of Increasing Example Count. For NonRAG, in-
creasing examples from 0 to 3 shots substantially improves
performance in most cases, particularly for DCI dimensions.
However, further increasing from 3 to 6 shots yields mis-
cellaneous results, as we observe some dimensions improve
while others deteriorate. For GradeRAG, the impact of in-
creasing examples follows a similar pattern. The transitions
from zero-shot to 3-shot generally improve performance, yet
the magnitude of improvement is typically smaller than for
NonRAG. This suggests that retrieval partially compensates
for the lack of examples. The transitions from 3 to 6 shots
within GradeRAG show varying effects. Positive gains ap-
pear in all questions in DCI and 2 out of 3 in SEP, yet
(a) Retrieval Size Ablation For Q1
(b) Retrieval Size Ablation For Q2
(c) Retrieval Size Ablation For Q3
Figure 4: The impact of the retrieval size ( k).

performance decreases for 1 out of 3 in both SEP and CCC.
These two observations suggest the following findings re-
garding the compatibility of GradeRAG with in-context learn-
ing. First, adding more examples might lead to potential in-
formation excessiveness, causing diminishing utility or even
performance decreases. Second, the optimal balance be-
tween retrieval and examples depends heavily on the specific
dimension and question complexity. Third, simply adding
more examples or more retrieved content does not guarantee
improved performance. Different dimensions require differ-
ent balances between the retrieval size and example number
to achieve optimal performance.
Effect of Knowledge Retrieval Size. We further conducted
a systematic ablation study on the impact of retrieval size
k(number of retrieved knowledge chunks) across different
in-context learning settings. Figures 4 represent the accu-
racy results for all questions across all dimensions as we
vary the retrieval size kfrom 0 to the maximum available
chunks. For Q1, increasing kin zero-shot settings gener-
ally improves zero-shot performance across all dimensions,
with CCC showing the most drastic gains. For Q2, perfor-
mance typically peaks at moderate kvalues ( k=2 or 3) be-
fore beginning to decline. Q3shows more results, with DCI
benefiting from increasing kup to 3 in zero-shot settings,
while SEP achieves the best performance at k=1 under 3-
shot settings and CCC shows optimal results with moderate
kvalues in zero-shot settings. Across all three questions, we
observe that the optimal retrieval size kdecreases as more
examples are provided. In zero-shot scenario, k=4 generally
leads to the best performance; while k=1 typically performs
the best in 3-shot scenarios and k=2 the best in 6-shot sce-
narios. This suggests that complementary information can
be provided by retrieved knowledge or in-context examples.
5. CONCLUSION
This study investigates into the effectiveness of GradeRAG,
a novel framework that integrates retrieval-augmented gen-
eration into automatic short answer grading for science ed-
ucation. Our experiments across three science assessment
tasks demonstrate that incorporating domain-specific knowl-
edge through RAG significantly improves scoring accuracy,
with consistent performance gains across multiple dimen-
sions of scientific understanding. Our experimental results
show several advantages of GradeRAG. First, it effectively
enhances the alignment between expert graders and auto-
mated grading systems by providing access to relevant do-
main knowledge. Second, our dual-index strategy and the
comprehensive retrieval mechanism ensure that the most
pertinent information is retrieved for student responses. Lastly,
GradeRAG demonstrates compatibility with in-context learn-
ing, exhibiting the complementary relationship between the
retrieved knowledge and expert examples, which further en-
hances grading performance.
References
Alexandra S Beatty, Judith A Koenig, Mark R Wilson, and
James W Pellegrino. Developing assessments for the next
generation science standards. 2014.
Steven Burrows, Iryna Gurevych, and Benno Stein. Theeras and trends of automatic short answer grading.
International journal ofartificial intelligence ineducation,
2015.
National Research Council et al. A framework for k-12 sci-
ence education: Practices, crosscutting concepts, and core
ideas. National Academy ofSciences, 2012.
Kelvin Guu et al. Retrieval augmented language model pre-
training. In ICML, pages 3929–3938. PMLR, 2020.
G Harshavardhan and Kulvinder Singh. A rubric-centric
approach for automated test correction utilizing rag and
fine tuning. In ICTACS, 2024.
Sarah Hassan, Aly A. Fahmy, and Mohammad El-Ramly.
Automatic short answer scoring based on paragraph em-
beddings. IJACSA, 2018.
Peng He, Xiaoming Zhai, Namsoo Shin, and Joseph Krajcik.
Applying rasch measurement to assess knowledge-in-use
in science education. In Advances inApplications ofRasch
Measurement inScience Education. Springer, 2023a.
Peng He, Namsoo Shin, Xiaoming Zhai, and Joseph Krajcik.
A design framework for integrating artificial intelligence
to support teachers’ timely use of knowledge-in-use assess-
ments. Uses ofArtificial Intelligence inSTEM Education,
2024.
Peng He et al. Predicting student science achievement us-
ing post-unit assessment performances in a coherent high
school chemistry project-based learning system. Journal
ofResearch inScience Teaching, 2023b.
Claudia Leacock and Martin Chodorow. C-rater: Auto-
mated scoring of short-answer questions. Computers and
theHumanities, 2003.
Gyeong-Geon Lee et al. Applying large language models and
chain-of-thought for automatic scoring. arXiv preprint
arXiv:2312.03748, 2024.
Patrick Lewis et al. Retrieval-augmented generation for
knowledge-intensive nlp tasks. NeurIPS, 2020.
Qingyao Li et al. Adapting large language models for ed-
ucation: Foundational capabilities, potentials, and chal-
lenges. arXiv preprint arXiv:2401.08664, 2023.
Fatma Miladi, Val´ ery Psych´ e, and Daniel Lemire. Com-
parative performance of gpt-4, rag-augmented gpt-4, and
students in moocs. In International Conference on
Breaking Barriers with Generative Intelligence, pages 81–
92. Springer, 2024.
Michael Mohler, Razvan Bunescu, and Rada Mihalcea.
Learning to grade short answer questions using semantic
similarity measures and dependency graph alignments. In
ACL 2011, 2011.
NGSS Lead States. Next generation science standards: For
states, bystates. National Academies Press, 2013.
Md Arafat Sultan, Cristobal Salazar, and Tamara Sumner.
Fast and easy short answer grading with high accuracy.
InNAACL, 2016.

Koushik Sundar, Eashaan Manohar, K Vijay, and Sajay
Prakash. Revolutionizing assessment: Ai-powered eval-
uation with rag and llm technologies. In ICSSAS, pages
43–48. IEEE, 2024.
Shenglai Zeng et al. The good and the bad: Exploring pri-
vacy issues in retrieval-augmented generation (rag). arXiv
preprint arXiv:2402.16893, 2024.