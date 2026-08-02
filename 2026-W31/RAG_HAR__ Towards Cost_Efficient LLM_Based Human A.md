# RAG-HAR+: Towards Cost-Efficient LLM-Based Human Activity Recognition for Edge Deployment

**Authors**: Hansi Karunarathna, Nirhoshan Sivaroopan, Chamara Madarasingha, Anura Jayasumana, Kanchana Thilakarathna

**Published**: 2026-07-29 08:57:03

**PDF URL**: [https://arxiv.org/pdf/2607.26631v1](https://arxiv.org/pdf/2607.26631v1)

## Abstract
Human Activity Recognition (HAR) from wearable sensors supports applications in healthcare, rehabilitation, fitness tracking, and smart environments. Yet, existing deep learning approaches require dataset-specific training, large labeled corpora, and repeated adaptation to new sensor settings or activity taxonomies. Retrieval-Augmented Generation for Human Activity Recognition (RAG-HAR) addresses this by framing HAR as a training-free, retrieval-augmented task, in which statistical descriptions of sensor windows are used to retrieve similar labeled examples that guide LLM-based classification. We introduce RAG-HAR+, a retrieval-first and cost-optimized extension that strengthens retrieval while reducing dependence on LLM-based inference. RAG-HAR+ uses an offline Retrieval Designer Agent to design dataset-specific feature groups from a diverse pool of motion descriptors, enabling sensor windows to be compared using features better aligned with dataset-specific activity patterns. During inference, RAG-HAR+ uses majority voting over retrieved neighbors for samples with strong retrieval evidence and defers only uncertain cases to an LLM-based Ambiguity Resolver Agent. Across six HAR benchmarks, RAG-HAR+ maintains competitive or improved performance while reducing LLM usage, token consumption, and inference time. We further extend the RAG-HAR mobile prototype to demonstrate the practical feasibility of retrieval-first, LLM-assisted HAR in mobile sensing scenarios.

## Full Text


<!-- PDF content starts -->

1
RAG-HAR+: Towards Cost-Efficient LLM-Based
Human Activity Recognition for Edge Deployment
Hansi Karunarathna, Nirhoshan Sivaroopan, Chamara Madarasingha, Anura Jayasumana, and Kanchana
Thilakarathna
Abstract—Human Activity Recognition (HAR) from wearable
sensors supports applications in healthcare, rehabilitation, fitness
tracking, and smart environments. Yet, existing deep learn-
ing approaches require dataset-specific training, large labeled
corpora, and repeated adaptation to new sensor settings or
activity taxonomies. Retrieval-Augmented Generation for Human
Activity Recognition (RAG-HAR) addresses this by framing
HAR as a training-free, retrieval-augmented task, in which
statistical descriptions of sensor windows are used to retrieve
similar labeled examples that guide LLM-based classification.
We introduce RAG-HAR+, a retrieval-first and cost-optimized
extension that strengthens retrieval while reducing dependence
on LLM-based inference. RAG-HAR+ uses an offline Retrieval
Designer Agent to design dataset-specific feature groups from a
diverse pool of motion descriptors, enabling sensor windows to
be compared using features better aligned with dataset-specific
activity patterns. During inference, RAG-HAR+ uses majority
voting over retrieved neighbors for samples with strong retrieval
evidence and defers only uncertain cases to an LLM-based
Ambiguity Resolver Agent. Across six HAR benchmarks, RAG-
HAR+ maintains competitive or improved performance while
reducing LLM usage, token consumption, and inference time. We
further extend the RAG-HAR mobile prototype to demonstrate
the practical feasibility of retrieval-first, LLM-assisted HAR in
mobile sensing scenarios.
Index Terms—Human Activity Recognition, Retrieval-
Augmented Generation, Large Language Models, Cost
Optimization, Wearable Sensors
I. INTRODUCTION
Human Activity Recognition (HAR) from wearable and
mobile sensor data underpins continuous monitoring, anomaly
detection, and personalized intervention across healthcare,
rehabilitation, fitness tracking, and industrial workflow moni-
toring [1]–[3]. Because these applications run continuously on
resource-constrained wearables, smartphones, and edge gate-
ways, the practical value of a HAR system depends not only on
recognition accuracy but also on per-inference computational
cost, latency, and connectivity requirements.
Deep learning (DL) has become the dominant paradigm
for HAR, with convolutional, recurrent, and attention-based
models achieving strong benchmark performance [4]–[6].
However, DL-based HAR depends on dataset-specific training
pipelines and hyperparameter tuning, degrades under domain
shift across subjects, placements, and devices, and requires
large labeled corpora that are costly to collect and annotate.
Hansi Karunarathna and Nirhoshan Sivaroopan contributed equally to this
work.
Hansi Karunarathna, Nirhoshan Sivaroopan and Kanchana Thilakarathna
are with the University of Sydney, Australia.
Chamara Madarasingha is with Curtin University, Australia.
Anura Jayasumana is with Colorado State University, USA.As a result, HAR still lacks a solution that is simultaneously
training-free, generalizable, and scalable. Large Language
Models (LLMs) offer a different path for learning. Pretrained
on broad corpora, they bring semantic knowledge of human
activities, strong pattern matching that may support cross-
domain generalization, and reasoning over context-dependent
tasks. However, raw sensor streams cannot be supplied to an
LLM directly because they are continuous, multi-channel time
series, whereas LLMs operate over tokenized text and long
windows of time series quickly exhaust the context budget.
LLMs also lack grounding in activity-specific knowledge
(sensor placement, activity definitions, and examples of easily
confused motions) that is needed for accurate prediction.
Retrieval-Augmented Generation (RAG) addresses both issues
by grounding the model in concrete labeled evidence.
Our prior work, RAG-HAR [7] applies this idea to HAR:
it describes each sensor window with multi-segment-view
statistics, indexes these descriptions in a vector database,
retrieves semantically similar labeled examples, and uses the
retrieved context to guide LLM-based classification, recogniz-
ing activities without training or fine-tuning a dataset-specific
model. This retrieval-augmented formulation is attractive for
training-free, scalable HAR, but its current realization of time-
series analysis carries two limitations that become acute under
continuous edge deployment. First, the retrieval representation
is fixed: predefined temporal views are always described with
the same basic statistics (mean, max, min, std, p25, p75,
median) assuming one segment-level representation suffices
across all activities and datasets. In practice, different motions
call for different retrieval cues (i.e,. locomotion is better
characterized by periodicity and frequency content, postures
by signal magnitude and orientation, and short industrial
gestures by intensity, shape, or local motion change) so a
fixed temporal-view, basic-statistics descriptor may miss the
most discriminative evidence. Second, the LLM is invoked for
every test sample. In a typical deployment, sensing runs on a
wearable or phone while the LLM is hosted and reached over
the network, so each prediction incurs a network round-trip
and a latency that grows with prompt length, and the retrieval-
augmented prompt is already large, carrying the query de-
scription together with multiple retrieved exemplars and their
summaries. The result is recurring token cost, variable latency,
and a hard dependence on connectivity, paid even for windows
whose retrieved neighbors already agree on a confident answer.
For inference that repeats indefinitely over streaming data, this
per-sample reliance on an online LLM, rather than accuracy,
becomes the dominant barrier to deployment.
This paper presents RAG-HAR+, a retrieval-centric and
arXiv:2607.26631v1  [cs.LG]  29 Jul 2026

2
cost-optimized framework for efficient edge HAR. Its central
idea is to changewhenandwherethe LLM is used. Rather
than acting as an online classifier invoked for every sample,
the LLM is moved upstream in the pipeline into a one-time
offline stage that designs a stronger retrieval representation
and is retained online only as the Ambiguity Resolver Agent
for the small fraction of ambiguous windows. Offline, given
dataset context, candidate feature descriptions, and retrieval
feedback, the offline Retrieval Designer Agent selects compact
feature groups from a broad pool of statistical, temporal, spec-
tral, and signal-shape descriptors; these groups are indexed
as separate vector fields for multi-vector search, letting the
retrieval representation adapt to the motion characteristics of
each dataset. The benefit of the offline retrieval-design stage
is reflected during online inference. By producing retrieved
contexts that are more informative and better aligned with the
query activity, the system can classify most windows directly
through majority voting over the retrieved neighbors. The
Ambiguity Resolver Agent is therefore invoked only when the
retrieved evidence is ambiguous. Online computational cost
therefore scales with prediction difficulty rather than with the
number of test samples. The offline design cost is incurred
once per dataset and amortized across all later inference,
preserving the training-free nature of the framework while
substantially lowering deployment cost and latency.
The main contributions of this paper are as follows:
•We reframe LLM usage in retrieval-augmented HAR as a
two-stage process: a one-time offline retrieval-refinement
stage, where the LLM designs dataset-specific feature
groups, and an online Ambiguity Resolver Agent, where
the LLM is invoked only when retrieved neighbors do
not provide a clear majority-vote prediction. Thus, online
inference cost scales with prediction ambiguity rather
than the number of test samples.
•We propose an iterative, LLM-guided Retrieval Designer
Agent that uses dataset context and retrieval feedback
to construct compact multi-vector retrieval groups from
statistical, temporal, spectral, and signal-shape features,
replacing fixed segment-level statistical groups.
•We implement a smartphone-based prototype that embeds
RAG-HAR+ as an activity-recognition widget in a mobile
fitness application, demonstrating on-device latency and
the practical deployability of retrieval-first, LLM-assisted
HAR in a mobile edge setting.
•We provide an extended evaluation across six HAR
benchmarks, with ablations isolating the different com-
ponent of Retrieval Designer Agent and Ambuiguity
Resolver Agent. RAG-HAR+ improves performance on
four datasets, remains competitive on the other two, and
substantially reduces online LLM usage by 89.3–99.9%,
quantifying the accuracy–cost–latency trade-off.
II. RELATEDWORK
A. Large Language Models for HAR
LLMs have recently been explored for time-series analysis
and sensor-based reasoning [8]–[10], making them attractive
for HAR, where low-level sensor patterns must be mappedto high-level activities. Recent methods connect sensor data
and LLMs through different forms of adaptation: Sensor-
LLM uses channel-specific tokens and task-aware tuning [11],
LLM4HAR combines sensor adaptation and efficiency mod-
ules [12], and PH-LLM fine-tunes an LLM on expert case
studies and wearable data for personalized coaching [13].
Other works pair learned sensor encoders with LLM reasoning,
such as SimCLR-based representations for comparing test
and reference samples [14], and extend LLMs to related
sensing tasks including user recognition and activity-event
description [15], [16]. While these studies show the promise
of LLMs for sensor reasoning, they require fine-tuning, task-
specific adaptation, or learned encoders, reducing the training-
free advantage and limiting adaptability to new sensor con-
figurations or activity taxonomies. Direct prompting methods
such as HARGPT avoid training but degrade on fine-grained
activities and higher-dimensional sensor inputs [17]. These
limitations motivate grounding LLM-based HAR in relevant
labeled examples from the target sensing context.
B. Retrieval-Augmented HAR
RAG grounds LLM reasoning in external evidence by
retrieving relevant examples or documents and providing them
as context for prediction. In HAR, this allows a query sensor
window to be compared with similar labeled activity windows
from the target dataset, giving the LLM concrete examples
of the motion patterns it must distinguish. Our prior work,
RAG-HAR [7], introduced a training-free retrieval-augmented
framework that computes lightweight statistical descriptors
from sensor windows, retrieves similar labeled samples from
a vector database, and uses the retrieved context for LLM-
based activity classification. However, RAG-HAR relies on
fixed segment-level statistical views for retrieval and invokes
the LLM for every test sample. RAG-HAR+ extends this line
of work by improving retrieval and reducing online LLM
usage. Instead of fixed temporal segment views and the same
basic statistics across datasets, it uses LLM-guided feature-
group selection to build adaptive retrieval representations from
a broader feature pool. It then uses retrieval confidence to
classify clear samples directly through neighbor voting and
routes only ambiguous samples to the Ambiguity Resolver
Agent. Thus, the LLM shifts from being a mandatory per-
sample classifier to serving as an offline Retrieval Designer
Agent and an online fallback for ambiguous cases.
III. METHODOLOGY
A. RAG-HAR+ Overview
RAG-HAR+ extends RAG-HAR by introducing an adap-
tive retrieval representation and a cost-optimized inference
strategy. The proposed framework is training-free. In this
work,training-freemeans that RAG-HAR+ does not train or
fine-tune a dataset-specific deep learning model for activity
classification. The framework still uses labeled training sam-
ples to construct the retrieval index and a validation split to
select feature groups and retrieval weights, but these steps
configure the retrieval pipeline rather than train a dataset-
specific classifier or fine-tune the LLM.

3
LLMGi1Gi2Gi3Build Indexing DBEvaluate on Validation Setround_idx(i) < N roundsSelect the Best Round YesNoG*1G*2G*3Channel m (Cm)Full (Si)…
NormalizationWindow SegmentationData Pre-processing1HAR data
 
 
Select top-q samplesIndexing phase3
 Repeat for all channels up to Cm  G*2 G*1 G*3f1f2f3….fkf1f2f3….fmf1f2f3….fnNumerical vectors{ vi,1 , vi,2 , vi,3 }Indexing data
 w1 w2 w3Feature Group Computation2
𝜏Vector Databasevtri,1vtri,3vtri,2
ANNANNANNvtex,1vtex,2vtex,3
 
 
  Dp,1 Dp,3 Dp,2Testdata
Re-rank Test Phase4
Normalization
Multi Vector Search4.1Top-q Neighbor Retrieval4.2Majority Voting4.3Top-q SamplesS1(label A)S2(label A)S3(label B)…..Sq(label C)Count votes for each activity among top-q samples
Clear majority among Top-q samples ?ActivityYes4.4Create prompt(with top q samples)ActivityLLM 
 Ambiguity Resolver Agent4.5NoInitial PromptImprovement Prompt
Retrieval Designer Agent: LLM is used in offline feature group designingAAmbiguity Resolver Agent: LLM is used online as a fallback classifier for ambiguous samplesB(a)(b)Construct Improvement Prompt
Fig. 1. Overview of RAG-HAR+. (a) Offline Retrieval Designer Agent for selecting dataset-specific feature groups. (b) Feature indexing and inference
pipeline, where training samples are indexed for retrieval and test samples are classified using majority voting or the Ambiguity Resolver Agent.
RAG-HAR+ represents HAR windows using lightweight
numerical feature vectors for retrieval and converts selected
windows into structured feature descriptions only when LLM
fallback is required for classification. Unlike RAG-HAR,
which uses the same fixed set of statistical descriptors over
predefined temporal segments and invokes the LLM for every
test sample, RAG-HAR+ uses the LLM in two distinct roles.
First, theRetrieval Designer Agentuses the LLM offline to
select dataset-specific feature groups for retrieval. Second, the
Ambiguity Resolver Agentuses the LLM online to resolve only
the ambiguous test samples.
Fig. 1 summarizes the overall RAG-HAR+ workflow. As
shown in Fig. 1(a), the LLM is first used offline by the
Retrieval Designer Agent to select dataset-specific feature
groups. Starting from an initial prompt, the agent proposes
three candidate feature groups, which are used to build an
indexing database and are evaluated on a validation set. The
validation feedback is then used to construct an improvement
prompt for the next round. After multiple rounds, the best-
performing feature groups, denoted asG∗
1,G∗
2, andG∗
3, are
selected as the final feature groups.
Fig. 1(b) shows how the selected feature groups are used
during indexing and inference. The pipeline begins with data
preprocessing 1, where raw HAR sensor streams are nor-
malized and segmented into fixed-size windows. The selected
feature groups from Fig. 1(a) are then computed for each
window across all sensor channels 2. The resulting feature-
group vectors are normalized and stored in the vector database
together with their activity labels 3. During the test phase 4,
the same preprocessing and feature-group computation steps
are applied to the query window. RAG-HAR+ then performs
multi-vector search, re-ranks the retrieved samples, selects
the Top-qneighbors, and applies majority voting over theirlabels, corresponding to steps 4.1–4.3. If a single activity
label obtains the highest vote count, the activity is predicted
directly from retrieval 4.4. If two or more labels share the
highest vote count, the sample is treated as ambiguous and
routed to the Ambiguity Resolver Agent 4.5. This selective
fallback mechanism reduces unnecessary online LLM calls
while retaining LLM reasoning for difficult cases.
B. Offline Feature-Group Design
1)Candidate Feature Pool Generation:RAG-HAR+ de-
fines a candidate feature pool as the search space for con-
structing feature groups. The pool contains feature descriptors
spanning statistical, temporal, spectral, motion-intensity, and
signal-shape categories [18]. Statistical features capture distri-
butional properties of each channel, temporal features describe
local signal changes and movement dynamics, spectral features
capture frequency-domain behavior, and shape-based features
describe waveform morphology. The pool is provided to the
Retrieval Designer Agent as a flat list of feature names and
short descriptions, from which the agent selects the feature
groups.
2)Retrieval Designer Agent:RAG-HAR+ uses the LLM
offline as a Retrieval Designer Agent to construct compact,
dataset-specific feature groups. This is a one-time process
performed on each dataset before test-time inference. Given
the candidate feature pool, the agent proposes three feature
groups at iterationi, denoted asG(i)={G(i)
1, G(i)
2, G(i)
3}.
Each group contains a subset of features from the pool and
forms one vector field in the multi-vector retrieval index. The
choice of three groups is motivated by the ablation results in
Section IV-E1b, where three groups achieve the best retrieval
performance.

4
The selected groups are not restricted to a single feature
category. Instead, each group may combine statistical, tem-
poral, spectral, motion-intensity, and signal-shape descriptors
to provide complementary representations of the same sensor
window. This allows retrieval to compare activity windows
from multiple perspectives rather than relying on a single fixed
descriptor set. Each selected feature is assigned to only one
group to reduce redundancy across feature groups and keep
the retrieval representation compact.
As shown in Fig. 1(a), the feature-group design process
is performed iteratively using an initial prompt followed by
improvement prompts. The structure of these prompts is illus-
trated in Fig. 2. The initial prompt in Fig. 2(a) provides the Re-
trieval Designer Agent with the task overview, dataset context,
system constraints, candidate feature pool, and expected output
format. The dataset context includes information such as the
dataset name, sensor setup, sensor locations, activity classes,
and known confusing activity pairs. The system constraints
specify that the agent should assign features to three compact
groups, avoid assigning the same feature to multiple groups,
and reduce redundant or subject-dependent features.
At iterationi, the proposed configurationG(i)is used to
build a temporary multi-vector retrieval index on the training
data and is evaluated on a validation split. The validation stage
uses two metrics: retrieval F1 score and RAG hit rate. The
retrieval F1 score measures the classification performance ob-
tained by applying majority voting over the retrieved neighbors
and comparing the predicted activity label with the ground-
truth label. RAG hit rate measures whether the correct activity
label appears among the retrieved candidates. Thus, retrieval
F1 score evaluates whether retrieval alone is sufficient for
direct classification, while RAG hit rate evaluates whether the
retrieved neighbors contain at least one correct-label example
that can support the Ambiguity Resolver Agent when fallback
is needed.
After each round, an improvement prompt is constructed
using the validation feedback, as illustrated in Fig. 2(b). This
prompt includes the best-performing configuration so far, over-
all retrieval F1 score, RAG hit rate, weak activity classes, per-
class retrieval behavior, recent feature-group selection history,
and stagnation warnings when the selections do not improve
across rounds. It also instructs the agent to focus on poorly
performing classes, choose features that better distinguish
confusing activities, avoid redundant descriptors, and explore
alternative grouping strategies when needed. This allows the
LLM to adjust the feature groups based on observed retrieval
errors, for example by adding features that better distinguish
confusing activities or removing redundant features that do
not improve retrieval. After multiple refinement rounds, the
configuration with the strongest validation Retrieval F1 score
is selected as the final retrieval representation, denoted as
G∗={G∗
1, G∗
2, G∗
3}.
Table I summarizes the final feature groups selected by the
Retrieval Designer Agent for all six datasets. The table shows
that the agent does not simply reuse a fixed descriptor set
across datasets. Instead, it consistently preserves some gener-
ally useful descriptors, such as standard deviation, interquartile
range, and autocorrelation, while adapting the remaining de-scriptors and their group assignments to the sensing setup and
activity classes of each dataset. This supports the intended role
of the agent: to design compact, non-overlapping, and dataset-
specific feature groups that provide complementary retrieval
representations.
C. Indexing and Inference Pipeline
1)Data Preprocessing:Given raw HAR sensor streams,
RAG-HAR+ first applies dataset-specific preprocessing to
generate fixed-size windows. Each dataset contains multiple
sensor channels, such as accelerometer or gyroscope axes,
collected from one or more body locations or devices. Let
C={C 1, C2, . . . , C m}denote the set of sensor channels. As
shown in Fig. 1(b), Step 1, data preprocessing consists of two
main steps.i)Normalization:All raw sensor channels are
standardized using Z-score normalization. For each channel,
the mean and standard deviation are estimated exclusively
from the training split and applied unchanged to the test
splits.ii)Sliding window-based segmentation:Following the
protocols used in prior HAR studies [19]–[22], the normal-
ized time-series data are partitioned into fixed-size windows
using a sliding window with a predefined window size and
step size. We denote the resulting labeled window set as
D={(S i, yi)}N
i=1, whereS idenotes thei-th sensor window
andy idenotes its activity label. This process captures short-
term temporal context and produces fixed-size units for feature
computation in Step 2of Fig. 1(b).
2)Feature-Group Computation and Vector Database In-
dexing:After the final feature-group configuration is selected,
RAG-HAR+ computes feature-group vectors for the labeled
training windows and stores them in the vector database. Let
the labeled indexing set be denoted asDtr={(S i, yi)}Ntr
i=1,
whereS iis thei-th training window,y iis its activity label,
andiindexes the training windows. Let the final selected
feature-group configuration beG∗={G∗
1, G∗
2, G∗
3}, where
k∈ {1,2,3}indexes the selected feature groups.
For each training windowS i, RAG-HAR+ computes one
feature vector from each selected groupG∗
k. The features in
G∗
kare computed across all sensor channels. IfG∗
kcontainsd k
features and the dataset hasmsensor channels, the resulting
vector for groupkhasd k×mdimensions. Each feature-group
vector is then normalized using the per-dimension mean and
standard deviation computed from the indexing split.
Thus, each indexed windowS iis represented by three
normalized feature-group vectors:
Vtr
i=
vtr
i,1,vtr
i,2,vtr
i,3	
,(1)
wherevtr
i,kdenotes the normalized vector generated from
windowS iusing feature groupG∗
k.
RAG-HAR+ then builds a multi-vector index using these
feature-group vectors. Here, indexing refers to storing each
labeled training window as a retrievable reference item in
the vector database. Unlike a single-vector index, RAG-HAR+
stores each window as one database record with three separate
vector fields, one for each selected feature group:
ri= 
vtr
i,1,vtr
i,2,vtr
i,3, yi
,(2)

5
Initial PromptImprovement promptTASK OVERVIEWYou are selecting features for a Human Activity Recognition (HAR) retrieval system. Assign each feature to exactly one of group_1, group_2, and group_3. Each group must contain 5–12 features and should mix statistical, temporal, and spectral features. Do not group features by category. Avoid redundant features within the same group. Prefer subject-invariant features.DATASET CONTEXT[dataset name, sensor setup, activity classes, known confusion pairs]SYSTEM ARCHITECTURESelect features and assign them to exactly 3 groups. Each group becomes a separate vector in a multi-vector retrieval system. Features are computed per axis for [N] sensor channels, so each selected feature adds [N] dimensions. Retrieval returns top-[q] labeled examples, and prediction uses label voting over the retrieved neighbors.AVAILABLE FEATURES[ﬂat pool: each feature name with a one-line description]OUTPUT FORMAT[Output Format]TASK OVERVIEWYou are optimizing feature selection and grouping for a HAR system. Select 5–12 features per group. Each feature must appear in exactly one group. Each group should mix statistical, temporal, and spectral features rather than grouping by category. Avoid redundant features within the same group.DATASET CONTEXT[dataset name, sensor setup, activity classes, known confusion pairs]SYSTEM ARCHITECTURESelect features and assign them to exactly 3 groups. Each group becomes a separate vector in a multi-vector retrieval system. Features are computed per axis for [N] sensor channels, so each selected feature adds [N] dimensions. Retrieval returns top-[q] labeled examples, and prediction uses label voting over the retrieved neighbors.ROUND FEEDBACK CONTEXT- Best round so far: Round [idx], Retrieval F1 score [..], RAG hit rate [..], Features: [best feature_selection]- Worst-class targets: 5 worst classes [ranked list]- Best-round per-class performance: retrieval F1 score and RAG hit rate [breakdown]- Recent round history: retrieval F1 score, RAG hit rate, and features and feature selections- Stagnation warning: [if any group stayed unchanged for 3 rounds]AVAILABLE FEATURES[ﬂat pool: each feature name with a one-line description]OPTIMIZATION STRATEGYFocus on the worst classes and choose features that better distinguish them. Prefer richer selections closer to 30 features when useful. If recent rounds did not improve, try a meaningfully different grouping strategy, including untested features.OUTPUT FORMAT[Output Format](a)(b)
Fig. 2. Prompt structure used by the Retrieval Designer Agent. (a) Initial prompt used to generate the first feature-group configuration. (b) Improvement
prompt used in subsequent rounds to refine feature groups based on retrieval feedback.
wherer iis the indexed database record for windowS i.
Therefore, each activity window has a single label but can
be retrieved through three complementary feature groups.
During inference, the same feature groups and normalization
statistics are applied to the query window to generate three
query vectors. Each query vector is searched against the
corresponding vector field in the database, and the retrieved
results are combined during weighted multi-vector retrieval.
3)Weighted Multi-Vector Search:For an unseen test win-
dowS x, RAG-HAR+ applies the same preprocessing, feature-
group computation, and normalization steps used during in-
dexing. This produces three query vectors:
Vx={vte
x,1,vte
x,2,vte
x,3}.(3)
Approximate nearest-neighbor search is performed sepa-
rately for each feature-group vector. For each selected feature
groupG∗
k, the corresponding query vectorvte
x,kis used to
retrieve the Top-pcandidate windows from the indexed vectors
of the same feature group. We denote the candidate set
retrieved from feature groupkasD(p)
x,k. The candidates from
the three feature groups are then merged and re-ranked to
select the final Top-qneighbors. Lets i,kdenote the similarity
between the query windowS xand an indexed windowS i
under feature groupk.
ρi=3X
k=1w∗
ksi,k,(4)
wherew∗
kis the optimized weight assigned to feature group
k,w∗
k≥0, andP3
k=1w∗
k= 1.
The weights{w∗
1, w∗
2, w∗
3}are selected through a validation-
set grid search over candidate weight combinations as de-
scribed in Section IV-E1d. The validation set is constructed
only from the training split, and the weight configuration that
gives the best validation retrieval objective is fixed for final
test-set inference. The merged candidates are re-ranked using
ρi, and the Top-qsamples are selected as the final retrieved
neighbor set:
D(q)
x={(S j, yj, ρj)}q
j=1.(5)4)Retrieval-Based Classification:After Top-qretrieval,
RAG-HAR+ decides whether the retrieved evidence alone is
sufficient to classify the query or whether the query should
be deferred to the Ambiguity Resolver Agent. This decision
is based on the agreement among the labels of the retrieved
neighbors. Note that inD(q)
xthe scoreρ jis used only to select
and order the Top-qneighbors during retrieval; in the voting
stage, it plays no further role, and each retained neighbor
contributes a single, equally weighted vote. For each activity
classc∈ Y, the vote count over the retrieved neighbor set
is computed using an indicator function. Here,I(y j=c)is
defined as:
I(yj=c) =(
1,ify j=c,
0,otherwise.(6)
The vote count for classcis then:
V(c) =X
(Sj,yj,ρj)∈D(q)
xI(yj=c).(7)
The retrieval-based prediction is:
ˆyret= arg max
c∈YV(c).(8)
RAG-HAR+ then applies a deterministic routing rule. If one
activity label obtains a unique highest vote count, the retrieved
neighbors are in clear agreement and the query is classified
directly from retrieval,ˆy= ˆy ret. If two or more labels share
the highest vote count, majority voting is undecided, and the
query is routed to the Ambiguity Resolver Agent.
We deliberately adopt parameter-free majority voting as the
routing rule. Because the database contains many close sam-
ples, we empirically observed that the similarity scores among
the top retrieved neighbors often do not have a large margin.
Majority voting avoids additional hyperparameters, requires no
calibration data, and keeps the routing rule deterministic and
training-free.

6
TABLE I
SELECTED FEATURE GROUPS FOR EACH DATASET BY THERETRIEVALDESIGNERAGENT. ORANGE FEATURES ARE COMMON TO ALL SIX DATASETS BUT
APPEAR IN DIFFERENT FEATURE GROUPS ACROSS DATASETS. PURPLE ITALICIZED FEATURES ARE NOT SHARED BY ALL SIX DATASETS.
DatasetG∗
1 G∗
2 G∗
3
USC-HAD mid-frequency bandpower,high-frequency band-
power, dominant frequency, dominant-frequency ra-
tio,autocorrelation at lag 10,peak count,peak-
frequency ratio,skewness,positive ratio,mean sec-
ond difference,spectral centroidsample entropy,count above mean,standard de-
viation, mean absolute change,very-low-frequency
bandpower,low-frequency bandpower,low–high
bandpower ratio,spectral flatness,mean-crossing
rate,spectral entropyRMS,total power,crest factor,impulse factor,wave-
form length, turning-points ratio,autocorrelation at
lag 1,autocorrelation at lag 5,autocorrelation at
lag 2,interquartile range
Skodastandard deviation,signal energy,ratio beyond one
standard deviation,count above mean,interquartile
range, mean absolute change,crest factor,autocor-
relation at lag 1,zero-crossing rateskewness,kurtosis, sample entropy,Hjorth complex-
ity,waveform length, turning-points ratio,mean-
crossing rate,coefficient of variation,spectral en-
tropyautocorrelation at lag 2, dominant frequency,
dominant-frequency ratio,low-frequency
bandpower, mid-frequency bandpower,spectral
centroid,spectral spread,linear trend slope,start–
end difference
GOTOVstandard deviation,interquartile range,coefficient of
variation,kurtosis,ratio beyond one standard devi-
ation, sample entropy,spectral flatness,spectral en-
tropy,very-low-frequency bandpower,spectral slopeskewness,mean change,positive ratio,linear trend
slope,start–end difference,autocorrelation at lag
1, mean absolute change,mean second difference,
turning-points ratio,waveform lengthdominant frequency, dominant-frequency ratio,
peak-frequency ratio,autocorrelation at lag 5,
autocorrelation at lag 10,zero-crossing rate,
mean-crossing rate,low-frequency bandpower, mid-
frequency bandpower,high-frequency bandpower
MHEALTHstandard deviation,RMS, dominant frequency,
dominant-frequency ratio,peak-frequency ratio,
mid-frequency bandpower,high-frequency band-
power,low-frequency bandpower,low–high band-
power ratio,total power,autocorrelation at lag 10,
peak countratio beyond one standard deviation, sample en-
tropy,spectral flatness,mean second difference,
count above mean,positive ratio,median,interquar-
tile mean,spectral centroid,spectral spread,mean-
crossing rate,very-low-frequency bandpowermean absolute change,Hjorth mobility,skewness,
kurtosis,linear trend slope,start–end difference,im-
pulse factor,clearance factor, turning-points ratio,
waveform length,crest factor
HHAR dominant frequency, dominant-frequency ratio,
peak-frequency ratio,autocorrelation at lag 1,
zero-crossing rate, mid-frequency bandpower,
low–high bandpower ratiolinear trend slope,start–end difference,first-half–
second-half difference,mean second difference,
mean change,crest factor,impulse factor,skewnessratio beyond one standard deviation, sample entropy,
count above mean,spectral flatness, mean abso-
lute change, turning-points ratio,interquartile mean,
trimmed mean (10%)
PAMAP2signal energy, mean absolute change,autocorre-
lation at lag 1, mid-frequency bandpower,high-
frequency bandpower,low-frequency bandpower,
low–high bandpower ratio, dominant frequency,
dominant-frequency ratio,mean-crossing rateratio beyond one standard deviation, sample entropy,
spectral flatness,mean second difference,spectral
entropy,zero-crossing rate,standard deviation,in-
terquartile range,crest factor,impulse factorlinear trend slope,start–end difference,first-half–
second-half difference,autocorrelation at lag 10,
peak count,peak-frequency ratio,waveform length,
turning-points ratio,second dominant frequency,
spectral centroid
5)Ambiguity Resolver Agent:For queries routed to the
Ambiguity Resolver Agent, RAG-HAR+ constructs a struc-
tured prompt using the query window, the retrieved examples,
their activity labels, and the dataset-specific activity defini-
tions. Although adaptive feature groups are used for retrieval,
these high-dimensional vectors are not directly exposed to the
LLM. This is because LLMs reason more effectively over
structured textual descriptions with semantically meaningful
fields than over dense numerical vectors optimized for simi-
larity search.
Instead, we use the segment-wise statistical prompt format
from RAG-HAR, where each window is described using basic
descriptors computed over the full window and the Start,
Mid, and End sub-segments. This introduces only a small
additional computational cost because these descriptors are
lightweight and are computed only for the query and the
Top-qretrieved examples when fallback is triggered. The
separation allows RAG-HAR+ to use adaptive feature groups
for retrieving stronger labeled neighbors, while presenting the
Ambiguity Resolver Agent with a compact and temporally
ordered representation suitable for comparison and reasoning.
The LLM then selects the most likely activity label from the
valid label set,ˆy= ˆy LLM. This selective fallback mechanism
preserves LLM reasoning for ambiguous cases while avoiding
unnecessary LLM calls when the retrieved neighbors already
provide sufficient evidence.
IV. EVALUATION
We evaluate RAG-HAR+ on six public HAR bench-
mark datasets used in the original RAG-HAR study: USC-HAD [23], PAMAP2 [24], MHEALTH [25], GOTOV [26],
HHAR [27], and Skoda [3]. These datasets cover diverse sens-
ing conditions, including single-IMU setups, multi-location
body-worn sensors, smartwatch-based sensing, and high-
dimensional industrial sensor arrays. They also include a wide
range of activity categories, from locomotion and postural
activities to daily living activities and assembly-line gestures.
We follow the same preprocessing, windowing, indexing, and
testing protocols used in RAG-HAR [7] to ensure a fair
comparison with the original method.1
To analyze retrieval quality, we report the retrieval F1 score
and RAG hit rate. Retrieval F1 score (RF1) measures the
classification performance obtained by applying majority vot-
ing over the retrieved neighbors and comparing the predicted
activity label with the ground-truth label. RAG hit rate (RHR)
measures whether the ground-truth label appears among the
Top-qretrieved neighbors.
A. Benchmarking RAG-HAR+
Table II compares the RAG-HAR+ pipeline with the RAG-
HAR and the strongest prior HAR baselines reported for each
dataset. We mention only the best non-LLM and LLM base-
lines in the table. Additional baselines for RAG-HAR bench-
marking are reported in [7]. Overall, RAG-HAR+ achieves
competitive performance across all six datasets while reducing
the reliance on LLM-based inference. RAG-HAR+ achieves
1Complete details of the dataset, protocols, LLM model, database, features
and hyper-parameters can be found in the Supplementary material.

7
TABLE II
END-TO-END BENCHMARK COMPARISON WITH THESOTA
NON-LLMANDLLM HARBASELINES.
Dataset Method Acc. F1
USC-HADTriplet LSTM (HTL-SB) [19] – 62.80
Transformer-like architecture [19] – 55.00
DeepConvLSTM [19] – 46.00
Sensor-LLM [11] 62.60 61.20
RAG-HAR 57.20 58.63
RAG-HAR+60.62 60.12
PAMAP2Triplet LSTM (OTL) [19] – 90.40
b-LSTM-S [19] – 86.80
ADFE [20] 85.69 77.84
HARGPT [17] 32.11 31.57
LLM as Virtual Annotators [28] 56.70 53.80
RAG-HAR91.60 91.12
RAG-HAR+ 90.35 90.60
MHEALTHADFE [20] 96.72 96.47
METIER [20] 94.42 94.09
MC-CNN [20] 89.69 87.42
RAG-HAR 96.91 96.74
RAG-HAR+98.35 98.19
GOTOVALAE-TAE-CutMix+ [21] – 79.40
Attend and Discriminate [21] - 76.20
Att. Model [21] - 70.70
RAG-HAR81.55 79.92
RAG-HAR+ 76.51 75.97
HHAREnhanced CPC [22] – 59.25
CPC [22] - 59.17
SimCLR [22] - 52.84
RAG-HAR 58.61 59.86
RAG-HAR+62.70 61.02
SkodaALAE-TAE-CutMix+ [21] – 94.80
Attend and Discriminate [21] - 92.80
b-LSTM-S [21] - 92.10
RAG-HAR 96.04 95.21
RAG-HAR+97.73 97.74
0K 25K 50K 75K 100K 125K 150K 175K 200K
T okens / Sample (K)USC-HADPAMAP2MHEALTHGOTOVHHARSkoda
14.9K67.7K53.1K27.2K15.7K132.9K
1.6K1.0K0.1K1.1K0.8K2.1K(a)
RAG-HAR
RAG-HAR+
0 5 10 15 20 25 30
Latency / Sample (s)USC-HADPAMAP2MHEALTHGOTOVHHARSkoda
18.86s13.83s18.25s18.84s8.45s18.33s
2.04s1.05s0.41s1.15s1.47s0.83s(b)
RAG-HAR
RAG-HAR+Fig. 3. (a) Average online LLM token usage. (b) Average inference latency
RAG-HARRAG-HAR+RAG-HARRAG-HAR+RAG-HARRAG-HAR+RAG-HARRAG-HAR+RAG-HARRAG-HAR+RAG-HARRAG-HAR+020406080100Score (%)
USC-HAD PAMAP2 MHEALTH GOTOV Skoda HHAR+14.0+19.0+15.2
+3.8+14.5
+4.2RHR (Upper bound for final classification score)
RAG-HAR: LLM classification (Final F1)
RAG-HAR: RF1 (Retrieved-context quality)RAG-HAR+: RF1 (Retrieved-neighbor majority voting)
RAG-HAR+: Ambiguity Resolver Agent contribution
Fig. 4. Comparison of RAG-HAR and RAG-HAR+. For RAG-HAR, the blue
bars indicate retrieval F1, while the red bars show the final F1 after LLM
classification. For RAG-HAR+, the blue bars show the F1 achieved through
retrieved-neighbor majority voting, and the stacked red segments indicate the
additional gain contributed by the Ambiguity Resolver Agent.
higher performance than RAG-HAR on four datasets: USC-
HAD, MHEALTH, HHAR, and Skoda, demonstrating that
the adaptive feature-group representation improves retrieval
quality and contributes to stronger classification outcomes.
Compared with prior deep learning baselines, the results
further show that RAG-HAR+ remains highly competitive
while preserving the training-free retrieval-augmented design
of RAG-HAR.
Although the performance gains over RAG-HAR are modest
on some datasets, they are noteworthy because RAG-HAR+
achieves these results with substantially lower online LLM in-
ference. As shown in Fig. 3, RAG-HAR+ substantially reduces
per-sample token usage and inference latency compared with
RAG-HAR. Unlike RAG-HAR, which invokes the LLM for
every sample, RAG-HAR+ first relies on retrieval-based clas-
sification and uses the LLM only for ambiguous cases. RAG-
HAR+ therefore provides a more efficient operating point by
maintaining competitive recognition performance while reduc-
ing token usage, API cost, and inference latency. This makes
the proposed pipeline more suitable for resource-constrained
mobile environments where continuous sensor streams can
otherwise lead to repeated and costly LLM invocations.B. LLM Cost Analysis
This subsection analyzes the LLM usage and cost of RAG-
HAR+. We report offline and online costs separately because
they occur at different stages of the pipeline. The offline
cost corresponds to the Retrieval Designer Agent, which is
executed once for each dataset configuration. After this step,
the selected feature groups and weights are fixed and reused
during inference. Therefore, for the offline stage, we report
the total token usage per dataset in Table III. Since absolute
dollar estimates may vary with changes in API pricing, we use
token counts and relative reductions as more stable measures
of cost efficiency.
For online inference, we compare RAG-HAR+ with RAG-
HAR in terms of token usage. In RAG-HAR, every test sample
is sent to the LLM after retrieval, so the online cost grows
directly with the number of test samples. In contrast, RAG-
HAR+ invokes the Ambiguity Resolver Agent only when
majority voting over the retrieved neighbors is undecided. To
make the comparison clear, we report both the total token
usage over the full test set and the average token usage
per test sample. Total token usage reflects the actual cost
of evaluating each dataset, while per-sample token usage

8
normalizes for differences in test-set size. These results are
summarized in Table IV and Fig. 3(a). The per-sample prompt
size generally increases with the number of sensor channels,
since more channel-wise descriptors must be included in the
prompt. For example, the Skoda dataset has 60 channels,
resulting in approximately 132.9K tokens (per sample), which
fits comfortably within the 400k-token context window of
gpt-5-mini[29].
The offline feature-group design stage uses 132,176 tokens
across all six datasets, corresponding to an estimated total
cost of $0.075. Since this cost is incurred only once per
dataset configuration, it is small compared with the recurring
online inference cost. For online inference, RAG-HAR uses
220.8M tokens across the six test sets. RAG-HAR+ reduces
this to 5M tokens by routing only ambiguous samples to
the Ambiguity Resolver Agent. This corresponds to a 97.7%
reduction in online LLM token usage. The reduction varies
across datasets because the fallback rate depends on label
agreement among the retrieved neighbors. When the selected
feature groups produce consistent retrieved neighbors, most
samples are classified directly through retrieval-based voting.
MHEALTH represents the strongest case, where only one
sample is routed to the Ambiguity Resolver Agent, resulting in
only one online LLM classification cost. Even in USC-HAD,
which shows the smallest reduction, RAG-HAR+ still reduces
online cost by 89.3%. These results show that RAG-HAR+
substantially reduces recurring LLM usage while preserving
the retrieval-augmented classification framework.
Fig. 4 further separates the contribution of retrieval-based
classification from that of the Ambiguity Resolver Agent.
Across all datasets, most of the final classification F1 is
achieved directly through retrieval-based voting. Compared
with RAG-HAR, RAG-HAR+ improves the retrieval-only F1
and narrows the gap between retrieval-based prediction and
final LLM-assisted F1, indicating that the adaptive feature-
group retriever provides stronger evidence before fallback.
The Ambiguity Resolver Agent then adds 0.1–3.3 percentage
points, with the largest gains on USC-HAD and HHAR. These
results show that RAG-HAR+ resolves most predictions with-
out invoking the Ambiguity Resolver Agent, while retaining
it as a useful fallback for ambiguous samples. Thus, selective
fallback substantially reduces online LLM usage while pre-
serving the accuracy gains provided by LLM-based reasoning.
RAG-HAR+ does not eliminate LLM usage entirely; instead,
it changes where and how the LLM is used. A small offline
TABLE III
OFFLINELLMTOKEN USAGE BYRETRIEVALDESIGNERAGENT IN
RAG-HAR+. THIS ONE-TIME COST IS INCURRED ONCE PER DATASET
CONFIGURATION.
Dataset Input Output Total
USC-HAD 17,929 3,996 21,925
PAMAP2 18,011 3,536 21,547
MHEALTH 18,985 4,044 23,029
GOTOV 18,546 5,095 23,641
HHAR 16,712 3,427 20,139
Skoda 18,084 3,811 21,895
Total 108,267 23,909 132,176LLM cost is invested in constructing a stronger retrieval repre-
sentation, while online LLM calls are reserved for cases where
retrieval evidence is insufficient. This makes the inference
cost scale with ambiguity rather than with the total number
of test samples. The reported costs account only for LLM
API usage. They do not include non-LLM components such
as preprocessing, feature extraction, vector indexing, nearest-
neighbor retrieval, storage, or system overhead.
C. Inference latency
Fig. 3(b) compares the per-sample latency of RAG-HAR
and RAG-HAR+. RAG-HAR+ reduces latency on every
dataset, with speed-ups ranging from5.7×on HHAR to44.5×
on MHEALTH. This reduction comes from the different infer-
ence strategy used by RAG-HAR+. In the RAG-HAR pipeline,
every test window is routed to the LLM after retrieval, which
introduces a network round-trip and LLM generation delay for
each sample. In contrast, RAG-HAR+ first attempts retrieval-
based classification and routes only ambiguous samples to the
Ambiguity Resolver Agent. As a result, most samples avoid
the online LLM stage, and the per-sample latency moves closer
to the cost of feature computation and retrieval alone.
The speed-up varies across datasets because the cost of
LLM-based inference and retrieval-based inference differs
with dataset characteristics. MHEALTH obtains the largest
reduction, from 18.25 s to 0.41 s per sample (44.5×), because
only one test sample is routed to the Ambiguity Resolver
Agent. Skoda achieves the second-largest speed-up of22.1×,
while HHAR shows the smallest, but still substantial, speed-
up of5.7×. Nevertheless, RAG-HAR+ still reduces latency
substantially on this dataset. The remaining latency of RAG-
HAR+ mainly reflects the retrieval pipeline, including feature
extraction over the query window, feature normalization, and
approximate nearest-neighbor search over the indexed training
corpus. This cost depends jointly on the number of channels,
window length, selected feature groups, and index size, rather
than on channel count alone. For example, USC-HAD shows
the highest RAG-HAR+ latency, while Skoda remains near
the lower end despite having the largest number of channels.
Across all datasets, the cost-optimized pipeline keeps latency
between 0.41s and 2.04s per sample, which is compatible with
continuous, near-real-time activity recognition on mobile and
edge devices.
D. Retriever Impact: RAG-HAR+ vs. RAG-HAR
This subsection analyzes how the redesigned RAG-HAR+
retriever improves retrieval quality over the original RAG-
HAR retriever. RAG-HAR represents each window using fixed
temporal segment views with the same basic statistical descrip-
tors, whereas RAG-HAR+ uses the Retrieval Designer Agent
to select three dataset-specific feature groups from statistical,
temporal, spectral, and signal-shape descriptors. These groups
form complementary vector fields in the multi-vector index,
allowing retrieval to capture activity-relevant cues that fixed
temporal summaries may miss.

9
TABLE IV
ONLINELLMTOKEN USAGE COMPARISON BETWEENRAG-HARANDRAG-HAR+FOR EACH DATASET. IN. =INPUT TOKENS, OUT. =OUTPUT TOKENS,
TOTAL=TOTAL TOKENS,ANDLLM CALLS=NUMBER OF INFERENCE-TIMELLMINVOCATIONS. REDUCTION IS COMPUTED FROM TOTAL TOKENS.
Dataset TestRAG-HAR RAG-HAR+ Token Usage
Reduction (%)In. Out. Total LLM Calls In. Out. Total LLM Calls
USC-HAD 492 6,734,004 617,066 7,351,070 492 715,790 68,667 784,457 53 89.3
PAMAP2 342 22,820,018 348,395 23,168,414 342 333,414 4,642 338,056 5 98.6
MHEALTH 666 34,587,378 781,484 35,368,862 666 52,250 597 52,847 1 99.9
GOTOV 1324 34,883,029 1,101,832 35,984,861 1324 1,400,776 89,228 1,490,004 53 95.9
HHAR 858 12,861,077 637,237 13,498,314 858 641,636 73,329 714,965 65 94.7
Skoda 793 104,775,918 639,317 105,415,235 793 1,639,389 16,422 1,655,811 14 98.4
Total 4475 216,661,424 4,125,331 220,786,756 4475 4,783,255 252,885 5,036,140 191 97.7
(a) RAG-HAR (b) RAG-HAR+ 
Fig. 5. t-SNE analysis of RAG-HAR vs. RAG-HAR+ retriever performance
on MHEALTH.
Fig. 4 compares the RAG-HAR retriever with the final
optimized RAG-HAR+ retriever produced by the Retrieval De-
signer Agent. For RAG-HAR, RF1 is reported only as a diag-
nostic retrieval-quality measure computed by applying voting
over its retrieved neighbors; the original RAG-HAR pipeline
still invokes the LLM for final classification of every sample.
In contrast, RAG-HAR+ RF1 reflects its actual retrieval-based
voting path before fallback. Across datasets, RAG-HAR+
improves RF1 and maintains high RHR, indicating that the
correct activity label is usually present among the retrieved
neighbors. The reduced gap between RF1 and final LLM-
assisted F1 further shows that the adaptive retriever provides
stronger evidence before fallback, allowing most samples to
be classified without invoking the Ambiguity Resolver Agent.
Fig. 5 provides a qualitative view of this effect using
MHEALTH retrieval embeddings. The RAG-HAR retriever
produces a more entangled embedding space, where several
activity classes overlap because fixed segment-level statistics
capture only coarse window-level patterns. In contrast, the
RAG-HAR+ retriever forms more coherent and better sepa-
rated clusters, showing that the selected feature groups capture
activity-specific retrieval cues more effectively. This observa-
tion is consistent with Fig. 4: on MHEALTH, RAG-HAR+
improves retrieval F1 from82.9%to98.1%and increases
RHR from95.7%to99.2%. A more separable retrieval space
increases the likelihood that the Top-qneighbors agree on the
correct label, enabling direct classification through voting.
The t-SNE visualization also explains why RAG-HAR+
reduces online LLM usage. When clusters are compact andwell separated, the deterministic routing rule can identify con-
fident retrieval evidence and avoid the latency and cost of an
LLM call. However, Fig. 5(b) also shows that some dynamic
activities with similar repetitive motion patterns remain close
in the embedding space. These boundary cases justify retaining
the Ambiguity Resolver Agent. Thus, RAG-HAR+ does not
remove the LLM entirely; it uses retrieval for clear cases and
reserves the LLM for genuinely ambiguous ones.
E. Ablation Studies
This subsection evaluates the principal design choices of
RAG-HAR+. We first evaluate the contribution of the adaptive
feature-group design. We then evaluate the representation used
to construct the retrieval index and analyze the combined effec-
tiveness of the resulting retriever. Finally, we study the online
inference stage by varying the number of retrieved neighbors
and the textual representation used by the Ambiguity Resolver
Agent.
1) Offline Retrieval Designer Agent:
a)Feature-Group Selection:We evaluate whether the
Retrieval Designer Agent selects genuinely useful retrieval
features, or whether the gains come only from using a richer
candidate pool than the basic statistical representation. To iso-
late feature selection from grouping effects, all configurations
in this experiment are concatenated into a single numerical
vector and evaluated under the same train/validation/test splits,
normalization, similarity function, and retrieval depth. On
USC-HAD, the Retrieval Designer Agent selects 31 features
in total; therefore, the budget-matched baselines also select
31 features from the same 67-feature candidate pool. Fig. 6(a)
reports the results. Three observations follow. First, expanding
the feature pool improves retrieval quality: using all 67 candi-
date features increases RF1 from 33.7% to 52.4%, showing
that richer motion descriptors provide more discriminative
retrieval cues than basic statistics alone. Second, feature selec-
tion further improves performance beyond pool expansion. The
agent-selected 31 features achieve 56.3% RF1, outperforming
the full 67-feature pool while retaining less than half of the
candidate features. This suggests that the agent removes noisy
or redundant descriptors rather than simply benefiting from
a larger representation. Third, the Retrieval Designer Agent
outperforms both classical supervised filter-based selection
and trained model-based selection: minimum Redundancy
Maximum Relevance (mRMR) achieves 44.9% RF1, while
Random-Forest-based feature selection achieves 47.6%. Thus,

10
RF1 RHR020406080100Score (%)
33.7
60.652.4
87.644.9
77.247.6
83.756.3
91.5(a)
Basic statistics
Retrieval Designer AgentFeature pool
Random ForestmRMR
RF1 RHR020406080100Score (%)
57.9
91.559.1
92.060.3
92.656.8
90.6(b)
1 group 2 groups 3 groups 4 groups
USC-HAD
RF1USC-HAD
RHRMHEALTH
RF1MHEALTH
RHR020406080100Score (%)
32.6
87.2
72.8
95.760.3
92.6
98.5
100.0(c)
T ext embedding Numerical vector
Fig. 6. Ablation of the offline retrieval design. (a) Held-out USC-HAD test performance of the basic-statistics, all-candidate, mRMR-selected, and Retrieval
Designer Agent-selected feature sets. (b) USC-HAD validation performance for different feature groups. (c) Validation performance of vectorizer index
representations on USC-HAD and MHEALTH.
the LLM-guided retrieval design provides a stronger training-
free alternative to conventional feature selection, improving
retrieval quality without training a dataset-specific classifier.
b)Number of Feature Groups:We analyze the effect
of the number of feature groups used for retrieval in RAG-
HAR+. Since different activity patterns may be better captured
by different feature groups, using multiple feature groups can
improve the diversity and relevance of retrieved examples.
However, increasing the number of feature groups may also
introduce redundant or less informative retrieval evidence.
Therefore, we conduct an ablation study on the USC-HAD
dataset by varying the number of feature groups from 1 to 4
while keeping all other components unchanged.
Fig. 6(b) reports the validation retrieval F1 score and RAG
hit rate for each configuration. The validation retrieval F1
score increases fromK=1toK=3, reaching its highest value
atK=3, and then drops atK=4. This suggests that three
feature groups provide sufficient complementary information
without introducing excessive redundancy or fragmenting the
per-group feature pool. Based on this result, we useK=3for
all subsequent experiments.
c)Feature-Group Assignment:The agent decides not
onlywhichfeatures to select but alsohowto partition them
into the three vector fields. To isolate the contribution of this
assignment, we fix the exact 31 features selected for USC-
HAD and vary only their organization, evaluating all condi-
tions with equal group weights so that weight optimization
cannot mask grouping effects. The agent’s grouping achieves
a test retrieval F1 of 56.1%, compared with 54.6±1.6% for
20 random partitions into the same group sizes and 55.3%
for a single concatenated vector. We conclude that the benefit
of the Retrieval Designer Agent is attributable to the specific
group assignment as well as feature selection.
d)Feature-Group Weighting:We next evaluate the ef-
fect of feature-group weighting in Table V. The number of fea-
ture groups is fixed at three, while their relative contributions
to the retrieval score are varied. Because the selected feature
groups capture different characteristics of human motion, they
may not contribute equally to retrieving relevant samples. We
therefore select dataset-specific weights rather than assigning
equal importance to all three groups. The weights are selected
through a grid search performed on a validation set as part of
the Retrieval Designer Agent. The validation set is constructed
exclusively from the training split, ensuring that the test split
remains unseen during weight selection. For the multi-subjectTABLE V
VALIDATION PERFORMANCE FOR DIFFERENT FEATURE-GROUP WEIGHT
COMBINATIONS ON THEUSC-HADVALIDATION SET.
w1 w2 w3 Val RF1 Val RHR
0.2 0.6 0.2 60.33 92.60
0.4 0.4 0.2 60.12 92.60
0.6 0.2 0.2 58.83 89.81
0.2 0.4 0.4 58.13 90.84
0.2 0.2 0.6 57.34 89.40
0.4 0.2 0.4 56.04 88.17
datasets MHEALTH, HHAR, PAMAP2, GOTOV and USC-
HAD, two complete subjects are held out from the training
split for validation. For Skoda, we use a random 20% window-
level validation split with a fixed seed.
For each dataset, the weights(w 1, w2, w3)are constrained
to sum to one, and each weight is varied from 0 to 1 in
increments of 0.2. The configuration that achieves the highest
validation retrieval F1-score is selected and fixed for the final
test-set evaluation. This procedure is performed independently
for each dataset.
2) Offline Index Construction:
a)Index Representation:We analyze whether retrieved
samples should be represented as text embeddings or numer-
ical vectors. Fig. 6(c) evaluates the effect of vector repre-
sentation. To isolate the effect of the retrieval representation,
both settings use the same LLM-selected feature groups. We
then compare two ways of storing and retrieving the resulting
feature values: representing them as textual descriptions and
embedding them using a text embedding model versus directly
storing the normalized feature values as numerical vectors.
This isolates whether numerical vector retrieval is more suit-
able than text-embedding retrieval for sensor-feature matching.
The results show that the retriever benefits more from numer-
ical vectors than from language-style text embeddings.
3) Ambiguity Resolver Agent:
a)Number of Retrieved Neighbors:We analyze the ef-
fect of the number of retrieved samples,q, on the performance
and efficiency of RAG-HAR+. Since retrieved samples are
used both for retrieval-based classification and as contextual
examples in the Ambiguity Resolver Agent prompt, increasing
qcan provide richer evidence but also increases prompt length
and may introduce less relevant neighbors. As shown in Fig. 7,
q= 10achieves the best final classification F1-score of
63.1%, improving overq= 5while maintaining similar
retrieval quality. Larger values ofqdo not provide further

11
020040060080010001200
T otal token usage (K)
689.1K 773.1K 733.2K 1105.0K
5 10 15 20
Retrieved samples (q)5859606162636465F1 score (%)
61.0
60.3
60.0 59.961.563.1
62.062.5Retrieval F1
Classification F1
T otal token usage
Fig. 7. Effect of retrieval depthqon USC-HAD validation performance and
token usage.
gains. Althoughq= 15slightly reduces total token usage
compared withq= 10(733.2K vs. 773.1K tokens), this saving
is relatively small and comes with a drop in final F1-score
from 63.1% to 62.0%. Atq= 20, the average tokens per LLM
call increase further, resulting in the highest total token usage
among the evaluated settings. Overall,q= 10provides the
best accuracy-efficiency trade-off, achieving the highest final
classification F1-score while keeping total token usage within
a reasonable range. Therefore, we use 10 as the retrieval depth
in RAG-HAR+.
b)Ambiguity Resolver Agent Prompt Representation:
In the Ambiguity Resolver Agent, the LLM receives the test
sample to be classified together with the retrieved samples
from the vector database. Since raw signals are not directly
suitable for LLM input, both the test sample and the retrieved
samples are converted into textual feature descriptions before
being included in the prompt. This ablation studies how
different prompt-level feature representations affect the classi-
fier’s performance and token usage. We compare three feature
representation strategies. First, in theselected feature-group
representation, the feature groups selected by the Retrieval
Designer Agent are computed over the full sensor window and
represented as feature descriptions in the formfeature =
value. Second, in thesegment-wise selected feature-group
representation, the same selected feature groups are com-
puted separately over four temporal segments: start, middle,
end, and full window. These segment-level values are then
arranged as textual feature descriptions. Third, in thesegment-
wise basic statistical representation, basic statistical features
are computed over the start, middle, end, and full-window
segments, following the representation offeature = value. As
shown in Fig. 8, the Retrieval Designer Agent-selected feature-
group representation achieves 62.4% accuracy and 61.6% F1-
score, with an average token usage of 21.7K tokens per LLM
call. Extending the selected feature groups across temporal
segments slightly improves performance to 62.6% accuracy
and 61.8% F1-score, but substantially increases the average
token usage to 81.2K tokens per LLM call. In contrast, the
segment-wise basic statistical representation achieves the best
classification performance, with 63.6% accuracy and 63.1%
F1-score, while requiring the lowest token usage of 14.8K to-
kens per LLM call. This result supports separating the retrieval
Classification
AccuracyClassification
F1Avg T okens /
LLM call020406080100Classification Score (%)
62.4
62.6
63.6
61.6
61.8
63.1Retrieval Designer Agent-selected feature groups (full window)
Retrieval Designer Agent-selected feature groups (4 temporal segments)
Basic statistical features (4 temporal segments)
0K20K40K60K80K100K120K140K
Avg T okens / LLM call21.7K
81.2K
14.8KFig. 8. USC-HAD validation-set ablation of the textual representation
supplied to the Ambiguity Resolver Agent.
representation from the LLM prompt representation: adaptive
feature groups improve neighbor retrieval, while compact
segment-wise statistical summaries provide a more effective
and token-efficient textual representation for the Ambiguity
Resolver Agent.
F . Discussion
The effectiveness of RAG-HAR+ is primarily governed by
retrieval quality. RAG-HAR+ deliberately uses a simple deter-
ministic majority-voting rule for routing. This choice keeps the
inference path parameter-free, reproducible, and training-free,
which is important for deployment across new datasets without
requiring calibration data. However, the same design also
opens a useful direction for future work. Rather than routing
only when the top retrieved labels tie, future systems could
use a more adaptive uncertainty-estimation strategy based
on the full retrieval distribution. For example, the entropy
of the neighbour-label distribution, the vote margin between
the top two classes, the agreement across feature groups,
or the similarity-score concentration could be combined to
decide when LLM fallback is worthwhile. Such an entropy-
based LLM fallback strategy could better balance accuracy
and cost by invoking the LLM for genuinely uncertain cases
while avoiding unnecessary calls when retrieval evidence is
already strong. As future work, we will investigate the privacy
implications of sharing sensor data with cloud-hosted LLMs;
in this version, we scope the study to computational cost and
latency reduction.
V. PILOTMOBILEPROTOTYPEIMPLEMENTATION
To examine the practical deployability of RAG-HAR+,
we implement a pilot mobile prototype [30] as an activity-
recognition widget within a Flutter-based fitness-tracking ap-
plication [31]. The prototype runs on a Samsung Galaxy S23
Ultra smartphone equipped with a Snapdragon 8 Gen 2 Mobile
Platform for Galaxy, 12 GB RAM, and built-in accelerom-
eter and gyroscope sensors. The prototype separates offline
preparation from online mobile inference. The laptop-hosted
backend runs the RAG-HAR+ Retrieval Designer Agent once
on mobile-collected labeled data to design feature groups and
build the multi-vector Milvus index [32]. At deployment, the
smartphone runs the RAG-HAR+ inference pipeline in real

12
(a) (b) (c) (d) (e) 
Fig. 9. RAG-HAR+ widget embedded within a mobile fitness-tracking application: (a) activity prediction, (b) activity insights and history, (c) labeled motion-
data collection, (d) real-time motion capture, and (e) application settings and device/data-store management.
0 20 40 60 80 100
F1-score (%)WalkingRunningSittingLyingJumpingStandingActivity
89.285.096.097.286.596.4
Fig. 10. Per-activity F1-scores of the mobile-integrated RAG-HAR+ prototype
on smartphone-collected activity data.
time, invoking the remotely hosted Ambiguity Resolver Agent
only for rare ambiguous cases.
The application also supports labeled data collection and
incremental retrieval-index expansion. As shown in Fig. 9,
users can record labeled motion samples through the mobile
interface, after which the corresponding feature vectors and
labels can be added to the retrieval index without retraining
a classifier. The Retrieval Designer Agent only needs to be
rerun when the sensing configuration, activity set, or candi-
date feature pool changes substantially. Thus, the prototype
demonstrates how RAG-HAR+ can support mobile activity
recognition while avoiding continuous transmission of raw
sensor streams and reducing unnecessary online LLM calls.
We evaluate the prototype on smartphone-collected activity
data to assess runtime behavior under real deployment rather
than benchmark conditions. RAG-HAR+ slightly improves
macro-F1 compared with RAG-HAR (91.7%vs.91.0%) while
reducing average per-sample latency from5400to360ms,
corresponding to a roughly15×speed-up. This latency re-
duction is larger than in the benchmark experiments because
the prototype captures real network round-trips to the hosted
LLM: RAG-HAR incurs this cost for every prediction, whereas
RAG-HAR+ resolves most windows locally through neighborvoting and routes only ambiguous cases to the Ambiguity
Resolver Agent. Fig. 10 shows stronger recognition for static
and postural activities and lower performance for dynamic
locomotion activities. Overall, the pilot prototype maintains
benchmark-level accuracy while substantially reducing per-
sample latency, supporting the feasibility of retrieval-first,
LLM-assisted HAR in a real mobile setting.
VI. CONCLUSION
RAG-HAR introduced a training-free HAR framework that
combines retrieval-augmented generation with LLM reason-
ing, enabling activity recognition through similarity search
over sensor-window descriptors and contextual reasoning over
retrieved examples, without dataset-specific training or fine-
tuning. This direction is timely beyond HAR as well, since
recent surveys in adjacent domains show rapidly growing
research activity around LLM-based methods [33]. Compared
with prior supervised HAR pipelines, RAG-HAR achieved
state-of-the-art performance across six benchmarks and further
demonstrated open-set reasoning by inferring unseen activities
using the LLM’s prior knowledge. RAG-HAR+ extends this
paradigm toward practical edge deployment by strengthening
retrieval and changing the role of the LLM from a per-sample
online classifier to an offline Retrieval Designer Agent and an
online Ambiguity Resolver Agent used only for difficult cases.
Across six benchmarks, RAG-HAR+ preserves competitive
or improved recognition performance while reducing online
LLM usage by89.3–99.9%and lowering per-sample latency
by up to an order of magnitude, with a smartphone prototype
confirming the feasibility of retrieval-first, LLM-assisted HAR
in mobile settings. Overall, RAG-HAR and RAG-HAR+ show
that retrieval, LLM reasoning, and cost-aware routing can
provide a scalable, training-free alternative to conventional
HAR pipelines, where inference cost scales with ambiguity
rather than data volume.

13
REFERENCES
[1] A. Bulling, U. Blanke, and B. Schiele, “A tutorial on human activity
recognition using body-worn inertial sensors,”ACM Computing Surveys,
vol. 46, no. 3, pp. 1–33, 2014.
[2] O. D. Lara and M. A. Labrador, “A survey on human activity recognition
using wearable sensors,”IEEE Communications Surveys & Tutorials,
vol. 15, no. 3, pp. 1192–1209, 2013.
[3] P. Zappi, C. Lombriser, T. Stiefmeier, E. Farella, D. Roggen, L. Benini,
and G. Tr ¨oster, “Activity recognition from on-body sensors: Accuracy-
power trade-off by dynamic sensor selection,” inProceedings of the
European Conference on Wireless Sensor Networks. Springer, 2008,
pp. 17–33.
[4] J. Yang, M. N. Nguyen, P. P. San, X. Li, and S. Krishnaswamy,
“Deep convolutional neural networks on multichannel time series for
human activity recognition,” inProceedings of the International Joint
Conference on Artificial Intelligence, 2015, pp. 3995–4001.
[5] F. J. Ord ´o˜nez and D. Roggen, “Deep convolutional and lstm recurrent
neural networks for multimodal wearable activity recognition,”Sensors,
vol. 16, no. 1, p. 115, 2016.
[6] L. Chen, Y . Zhang, and L. Peng, “Metier: A deep multi-task
learning-based activity and user recognition model using wearable
sensors,”Proceedings of the ACM on Interactive, Mobile, Wearable and
Ubiquitous Technologies, vol. 4, no. 1, Mar. 2020. [Online]. Available:
https://doi.org/10.1145/3381012
[7] N. Sivaroopan, H. Karunarathna, C. Madarasingha, A. Jayasumana,
and K. Thilakarathna, “Rag-har: Retrieval augmented generation-based
human activity recognition,” in2026 IEEE International Conference on
Pervasive Computing and Communications (PerCom). IEEE, 2026, pp.
1–12.
[8] N. Gruver, M. Finzi, S. Qiu, and A. G. Wilson, “Large language models
are zero-shot time series forecasters,”Advances in Neural Information
Processing Systems, vol. 36, 2023.
[9] A. Garza, C. Challu, and M. Mergenthaler-Canseco, “Timegpt-
1,”arXiv preprint arXiv:2310.03589, 2023. [Online]. Available:
https://arxiv.org/abs/2310.03589
[10] M. Jin, S. Wang, L. Ma, Z. Chu, J. Y . Zhang, X. Shi, P.-Y . Chen,
Y . Liang, Y .-F. Li, S. Pan, and Q. Wen, “Time-llm: Time series
forecasting by reprogramming large language models,” inInternational
Conference on Learning Representations, 2024.
[11] Z. Li, S. Deldari, L. Chen, H. Xue, and F. D. Salim, “Sensorllm:
Aligning large language models with motion sensors for human activity
recognition,” inProceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, 2025, pp. 354–379.
[12] Z. Hong, Y . Song, Z. Li, A. Yu, S. Zhong, Y . Ding, T. He, and D. Zhang,
“Llm4har: Generalizable on-device human activity recognition with
pretrained llms,” inProc of the 31st ACM SIGKDD conf on Knowledge
Discovery and Data Mining V . 2, 2025.
[13] J. Khasentino, A. Belyaeva, X. Liuet al., “A personal health
large language model for sleep and fitness coaching,”Nature
Medicine, vol. 31, pp. 3394–3403, 2025. [Online]. Available:
https://www.nature.com/articles/s41591-025-03888-0
[14] B. Khaertdinov, E. Ghaleb, and S. Asteriadis, “Contrastive self-
supervised learning for sensor-based human activity recognition,” in
2021 IEEE International Joint Conference on Biometrics (IJCB), 2021,
pp. 1–8.
[15] X. Chen, J. Cumin, F. Ramparany, and D. Vaufreydaz, “Towards llm-
powered ambient sensor based multi-person human activity recognition,”
in2024 IEEE 30th International Conference on Parallel and Distributed
Systems (ICPADS), 2024, pp. 609–616.
[16] Q. Xia, T. Maekawa, and T. Hara, “Unsupervised human activity
recognition through two-stage prompting with chatgpt,”arXiv preprint
arXiv:2306.02140, 2023. [Online]. Available: https://arxiv.org/abs/2306.
02140
[17] S. Ji, X. Zheng, and C. Wu, “Hargpt: Are llms zero-shot human activity
recognizers?” in2024 IEEE International Workshop on Foundation
Models for Cyber-Physical Systems & Internet of Things (FMSys), 2024,
pp. 38–43.
[18] M. Barandas, D. Folgado, L. Fernandes, S. Santos, M. Abreu, P. Bota,
H. Liu, T. Schultz, and H. Gamboa, “Tsfel: Time series feature
extraction library,”SoftwareX, vol. 11, p. 100456, Jan. 2020. [Online].
Available: http://dx.doi.org/10.1016/j.softx.2020.100456
[19] B. Khaertdinov, E. Ghaleb, and S. Asteriadis, “Deep triplet networks
with attention for sensor-based human activity recognition,” in2021
IEEE International Conference on Pervasive Computing and Communi-
cations (PerCom), 2021, pp. 1–10.[20] S. Suh, V . F. Rey, and P. Lukowicz, “Adversarial deep feature extraction
network for user independent human activity recognition,” in2022 IEEE
Percom, 2022.
[21] N. Ahmad and H.-f. Leung, “Alae-tae-cutmix+: Beyond the state-of-the-
art for human activity recognition using wearable sensors,” in2023 IEEE
International Conference on Pervasive Computing and Communications
(PerCom), 2023, pp. 222–231.
[22] H. Haresamudram, I. Essa, and T. Pl ¨otz, “Investigating enhancements
to contrastive predictive coding for human activity recognition,” in
2023 IEEE International Conference on Pervasive Computing and
Communications (PerCom), 2023, pp. 232–241.
[23] M. Zhang and A. A. Sawchuk, “Usc-had: A daily activity dataset for
ubiquitous activity recognition using wearable sensors,” inProceedings
of the ACM Conference on Ubiquitous Computing, 2012, pp. 1036–1043.
[24] A. Reiss and D. Stricker, “Introducing a new benchmarked dataset for
activity monitoring,” inProceedings of the International Symposium on
Wearable Computers. IEEE, 2012, pp. 108–109.
[25] O. Banos, R. Garcia, J. A. Holgado-Terriza, M. Damas, H. Pomares,
I. Rojas, A. Saez, and C. Villalonga, “mhealthdroid: A novel framework
for agile development of mobile health applications,” inProceedings
of the International Work-Conference on Ambient Assisted Living.
Springer, 2014, pp. 91–98.
[26] S. Paraschiakos, R. Cachucho, M. Moed, D. van Heemst, S. Mooijaart,
E. P. Slagboom, A. Knobbe, and M. Beekman, “Activity recognition
using wearable sensors for tracking the elderly,”User Modeling and
User-Adapted Interaction, 2020.
[27] A. Stisen, H. Blunck, S. Bhattacharya, T. S. Prentow, M. B. Kjærgaard,
A. K. Dey, T. Sonne, and M. M. Jensen, “Smart devices are different:
Assessing and mitigating mobile sensing heterogeneities for activity
recognition,” inProceedings of the ACM Conference on Embedded
Networked Sensor Systems, 2015, pp. 127–140.
[28] A. Hota, S. Chatterjee, and S. Chakraborty, “Evaluating large language
models as virtual annotators for time-series physical sensing data,”ACM
Transactions on Intelligent Systems and Technology, 2025.
[29] OpenAI, “GPT-5 mini Model,” https://developers.openai.com/api/docs/
models/gpt-5-mini, 2026, accessed: 2026-07-02.
[30] H. Karunarathna, N. Sivaroopan, C. Madarasingha, A. Jayasumana, and
K. Thilakarathna, “Large language models meet wearable sensing: A
demo of training-free har with rag-har,” in2026 IEEE International
Conference on Pervasive Computing and Communications Workshops
and other Affiliated Events (PerCom Workshops), 2026, pp. 470–472.
[31] Google, “Flutter: Build apps for any screen,” https://flutter.dev.
[32] Milvus, “Milvus vector database,” https://milvus.io//.
[33] N. Sivaroopan, K. Silva, C. Madarasingha, T. Dahanayaka, G. Jourjon,
A. Jayasumana, and K. Thilakarathna, “A comprehensive survey on
synthetic network traffic generation,”IEEE Communications Surveys &
Tutorials, vol. 28, pp. 5949–5983, 2026.