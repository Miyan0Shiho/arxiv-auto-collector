# MetaPlate: Counterfactual-Guided RAG-LLM Tool for Personalized Food Recommendation and Hyperglycemia Prevention

**Authors**: Asiful Arefeen, Carol Johnston, Hassan Ghasemzadeh

**Published**: 2026-06-08 19:52:08

**PDF URL**: [https://arxiv.org/pdf/2606.10120v2](https://arxiv.org/pdf/2606.10120v2)

## Abstract
Postprandial hyperglycemia is a key risk factor for metabolic disorders; however, existing dietary guidance is often static, impractical, and insufficiently personalized, providing recommendations that are difficult to follow or not impactful. While recent advances leverage continuous glucose monitoring (CGM) and machine learning to predict glycemic responses, these approaches are largely predictive and lack actionable guidance. Moreover, recommendation systems are often misaligned with user goals and require extensive input. We present MetaPlate, a counterfactual explanation (CF) guided, context-aware decision-support framework that generates personalized meal recommendations to mitigate postprandial glucose excursions in healthy adults. MetaPlate integrates multimodal data, including CGM readings, wearable-derived physiological signals, and user-provided meal inputs from $25$ individuals to model pre-meal context. A machine learning model predicts glucose response, while a CF optimization module adjusts meal composition modifying macronutrient amounts to maintain glucose levels within a target range ($\leq 140$ mg/dL). An LLM-based retrieval-augmented generation (RAG) layer enhances interpretability by producing human-readable recommendations using constrained search of the USDA food database. We evaluate MetaPlate via a structured expert-in-the-loop assessment with registered dietitians (RDs), comparing performance before and after prompt refinement. Results show improvements in meal realism, portion suitability, and recommendation likelihood, with expert feedback indicating a shift from clinically implausible outputs to actionable, contextually appropriate recommendations. Our findings emphasize the importance of domain knowledge and structured constraints in LLM-driven systems and highlight the potential of MetaPlate as a real-time personalized dietary decision-support tool.

## Full Text


<!-- PDF content starts -->

IEEE JOURNAL OF BIOMEDICAL AND HEAL TH INFOR-
MA TICS 1
MetaPlate: Counterfactual-Guided RAG-LLM
Personalized F ood Recommendation for
Hyperglycemia Prevention
Asiful Arefeen1,2*, Carol Johnston1, Hassan Ghasemzadeh1
Abstract— Postprandial hyperglycemia is a key risk factor
for metabolic disorders; however, existing dietary guidance is
often static, impractical, and insuﬀiciently personalized, fre-
quently providing low-level recommendations that are diﬀicult
to follow or high-level recommendations that are not impactful.
While recent advances leverage continuous glucose monitoring
(CGM) and machine learning to predict glycemic responses,
these approaches are predominantly predictive and lack action-
able dietary guidance. Moreover, existing recommendation sys-
tems are often insuﬀiciently aligned with user goals and may re-
quire extensive user input. T o address these gaps in research, we
present MetaPlate, a counterfactual explanation (CF)-guided,
context-aware decision-support framework that generates per-
sonalized meal recommendations to mitigate postprandial glu-
cose excursions. MetaPlate integrates multimodal data, in-
cluding CGM readings, wearable-derived physiological signals,
and user-provided meal inputs to model pre-meal context. A
machine learning model predicts postprandial glucose response
for a given meal and context (RMSE of 16.46 mg/dL), while a
CF optimization module adjusts meal composition—modifying
macro-nutrient amounts—to maintain glucose levels within a
target range (i.e., ≤140 mg/dL). An LLM-based retrieval-
augmented generation (RAG) layer further enhances inter-
pretability and understanding by producing human-readable
meal recommendations and explanations utilizing constrained
search from the USDA food database. W e first evaluate Meta-
Plate using data collected in an observational study involving
25 healthy adults. MetaPlate achieves higher validity ( 0.660
vs.0.540 for other techniques) while requiring smaller meal
modifications (normalized L1 distance: 0.230 ). W e then assess
MetaPlate through a structured, expert-in-the-loop assessment
involving registered dietitians (RDs), comparing system per-
formance before and after prompt refinement. Results demon-
strate substantial improvements in portion suitability (from
4.88 to7.67), alignment with dietary guidelines (from 5.38
to7.97), and recommendation likelihood (from 3.45 to7.10)
on a 10-point Likert scale, with expert feedback indicating
a shift from clinically implausible outputs toward actionable
and contextually appropriate recommendations. Our findings
emphasizes the importance of adding domain knowledge and
structured constraints in LLM-driven systems and underscore
the potential of MetaPlate as a real-time, personalized dietary
decision-support tool for hyperglycemia prevention.
Index T erms— Counterfactual explanations, Diabetes, Ex-
plainable AI, Glucose management, Large language model,
1College of Health Solutions, Arizona State University, Phoenix,
AZ 85004, USA
2School of Computing and Augmented Intelligence, Arizona State
University, Tempe, AZ 85281, USA
*Corresponding author: aarefeen@asu.edu
**Code available at: github.com/Arefeen06088/MetaPlateMeal recommendations, W earable sensors
I. INTRODUCTION
POSTPRANDIAL hyperglycemia is defined as blood
glucose concentrations exceeding 140 mg/dL in
healthy individuals [1]. Persistent hyperglycemia con-
tributes to the onset of diabetes mellitus [2] and is
associated with elevated cardiovascular risk [3]. There-
fore, maintaining in-range postprandial glucose levels is
important not only for individuals with diabetes but
also for otherwise healthy adults. Notably , recent studies
using continuous glucose monitoring (CGM) have demon-
strated substantial inter-individual variability in glycemic
responses to identical meals, exposing the limitations of
generalized, one-size-fits-all dietary guidelines [4]. These
risk factors, together with the challenges associated with
glucose variability , emphasize the need for personalized,
context-aware dietary strategies to mitigate glycemic ex-
cursions in healthy adults prior to the onset of overt
metabolic disease.
Recent work has increasingly leveraged CGM and
wearable or physiological sensing to model and forecast
glycemic response. Seminal work by Zeevi et al. demon-
strated that glycemic responses to real-life meals can be
predicted using machine learning models that combine
meal characteristics with individual-level biological and
behavioral features [5]. Subsequent studies have extended
this direction by integrating recent CGM traces, activity
signals, and multimodal wearable data to predict post-
prandial glucose response or hyperglycemia risk [6]–[12].
However, these systems largely stop at forecasting: they
estimate what is likely to happen after a meal, but do not
directly address the more actionable question of what a
user should eat instead under a given pre-meal context.
In parallel, food and nutrition recommendation sys-
tems have advanced rapidly from preference-based recom-
menders [13] toward goal-driven [14], health-aware and
RAG-enabled LLM frameworks [15] for meal planning
tools. Recent approaches like NutriGen [16] leverages
prompt engineering to generate personalized meal plans
aligned with user preferences and constraints, while Chat-
Diet [17] combines LLMs with population and person-level
models to recommend meals that minimize glycemic ex-
cursions while improving explainability and interactivity .arXiv:2606.10120v2  [cs.IR]  10 Jun 2026

2 IEEE JOURNAL OF BIOMEDICAL AND HEAL TH
INFORMA TICS
MOPI-HFRS [18] optimizes user preferences, personalized
healthiness, and nutritional diversity to generate meal
recommendations, followed by LLM-aided interpretation.
Broader work in food recommendation has also empha-
sized the importance of personalization, contextual aware-
ness, and health objectives. Despite these advances, most
existing food recommendation systems remain centered
on dietary preferences, calorie targets, or general wellness
constraints rather than on immediate physiological state
and explicit glycemic goals. They also often depend on
extensive user-specified inputs and therefore less suitable
for just-in-time decision support at mealtime.
CF s offer a promising paradigm for bridging the gap
between prediction and intervention. Rather than merely
explaining why a harmful outcome is predicted, CF meth-
ods identify minimal, feasible changes to inputs that would
alter the outcome. In the dietary setting, this translates to
the question: how should the meal be changed so that the
predicted postprandial glucose response remains within
a healthy range? This framing is particularly well-suited
for behavior change, because it translates model output
into concrete, goal-directed suggestions. Prior work has
highlighted the potential of CF not only as interpretability
tools but also as a basis for actionable health interventions
[19].
Despite this promise, current CF-based methods for
glycemic management still exhibit important limita-
tions. Existing approaches like ExAct [20], GlyMan, and
GlyT win [21] primarily generate interventions at the level
of behavioral variables or macronutrient adjustments, like
changing carbohydrate intake [20], [21], insulin dosage
[21], [22], or other numerical treatment parameters [20].
While useful, these low-level recommendations do not
fully realize the practical potential of explainable AI in
everyday dietary decision-making [23], as users ultimately
make choices in terms of foods and portion sizes rather
than abstract nutrient quantities. As a result, the final
step from CF reasoning to human-understandable meal
guidance often remains underdeveloped.
T o address these gaps, we present MetaPlate, a CF-
guided, context-aware decision-support framework for per-
sonalized meal recommendation in healthy adults. Meta-
Plate integrates multimodal pre-meal information from
CGM, wearable-derived physiological signals, and user
meal input to model an individual’s current context. A
machine learning model first predicts the postprandial
glucose response of a planned meal; a CF optimization
module then adjusts meal macronutrient composition to
satisfy a glycemic target; finally , an LLM-based retrieval-
augmented layer grounds the recommendation in the
USDA food database and converts these optimized nu-
tritional targets into food-level, human-readable meal
suggestions. In addition, we integrate an expert-in-the-
loop refinement process to improve the clinical plausi-
bility and usability of generated meals, ensuring that
recommendations are not only glycemically effective but
also realistic and actionable in practice. Thus, Meta-
Plate moves beyond passive forecasting and low-level CFoutputs to deliver interpretable, user-centered, and goal-
aligned dietary guidance.
W e make the following contributions:
•W e introduce a context-aware multimodal framework
that combines CGM, wearable sensing, and user meal
input to support pre-meal glycemic decision-making
in healthy adults.
•W e formulate meal recommendation as a CF inter-
vention problem and program an optimization system
to generate meal modifications targeted to maintain
postprandial glucose at or below a desired threshold.
•W e bridge the gap between numeric crude level CF
outputs and practical dietary guidance through an
LLM-based RAG layer that maps optimized macronu-
trient targets to food-level recommendations using the
USDA database.
•W e introduce an expert-in-the-loop evaluation and
refinement pipeline realizing that domain knowledge
significantly improves meal realism, portion suitabil-
ity , and clinical usability of LLM-generated recom-
mendations.
•W e evaluate the practical validity of MetaPlate using
a new dataset collected in free-living condition and
structured expert assessment, focusing on clinical
appropriateness, nutritional soundness, and practical
utility after iterative refinement.
II. P ROBLEM STATEMENT
LetD={(xi,mi, gi)}n
i=1 denote a dataset of neating
events collected from individuals, where each instance con-
sists of a pre-meal context xi∈Rd, a meal representation
mi∈Rk, and the corresponding observed postprandial
glucose outcome gi∈R (e.g., peak glucose within a
specified post-meal window).
The pre-meal context vector xi= [x1
i, x2
i, . . . , xd
i]in-
cludes multimodal physiological and behavioral features
derived from CGM and wearable sensors. These features
may include recent glucose trends, activity levels, heart
rate, and other contextual signals. The meal vector mi=
[m1
i, m2
i, . . . , mk
i]encodes the nutritional composition of
the meal, such as macronutrient quantities (e.g., carbohy-
drates, protein, and fat).
W e assume a predictive function
f:Rd×Rk→R, (1)
where Rd×Rkdenotes the joint input space of context
and meal features, which are concatenated in practice.
The function fmaps a pre-meal context xand a meal m
to a predicted postprandial glucose response, i.e.,
f(x,m) = ˆg (2)
Given a target glycemic threshold τ(e.g., τ= 140 mg/dL),
the desired outcome is:
f(x,m)≤τ. (3)
However, for a planned meal m0, it is often observed
that:
f(x,m0)> τ, (4)

AREFEEN et al.: MET APLA TE: COUNTERF ACTUAL-GUIDED PERSONALIZED FOOD RECOMMENDA TIONS 3
indicating a potential hyperglycemic response.
The objective is therefore to identify a modified meal
m∗that minimizes deviation from the original meal while
satisfying the glycemic constraint:
m∗= arg min
mD(m,m0) (5)
subject to:
f(x,m)≤τ,m∈ M , (6)
where D(·)is a distance function measuring the magnitude
of modification (e.g., changes in macronutrient composi-
tion), and M denotes the space of feasible meals under
nutritional, dietary , and practical constraints.
Additionally , we distinguish between actionable and
non-actionable components within the input space. While
the pre-meal context x contains largely non-actionable
physiological signals, the meal vector m represents the
primary actionable component. Therefore, the interven-
tion is constrained to modifying m while keeping xfixed.
In practice, the optimized meal m∗must be realizable
as a combination of discrete food items and portion sizes.
Thus, beyond satisfying the optimization constraints, the
modified meal m∗should be: (i) nutritionally valid, (ii)
practically realizable as a combination of real food items
and portion sizes, and (iii) interpretable to the end user.
In summary , the problem can be formulated as a
constrained counterfactual optimization task: given a pre-
meal context xand an initial meal m0, can we generate
a minimally modified, feasible meal m∗that achieves the
desired glycemic outcome while remaining actionable and
interpretable?
III. P ROPOSED METHOD
A. Postprandial Glucose F orecasting Model
Given a pre-meal context x∈Rdand a meal represen-
tation m∈Rk, the goal of the forecasting model is to
predict the corresponding postprandial glucose response.
W e learn a function
fθ:Rd×Rk→R, (7)
parameterized by θ, that estimates the peak postprandial
glucose level:
ˆg=fθ(x,m). (8)
In practice, the inputs xandm are concatenated into
a joint feature representation and passed through a model
that captures temporal and contextual dependencies. The
model is trained on the dataset D={(xi,mi, gi)}n
i=1 by
minimizing a regression loss:
Lpred=1
nnX
i=1(fθ(xi,mi)−gi)2. (9)
This predictive model serves as the foundation for
downstream counterfactual optimization.B. Counterfactual Optimization for Meal Adjustment
Given a pre-meal context xand an initial planned meal
m0, our objective is to generate a modified meal m∗such
that the predicted glucose response satisfies the target
constraint fθ(x,m∗)≤τ, while minimizing deviation from
the original meal and ensuring it is practically realizable.
This is formulated as a CF optimization problem, where
the goal is to identify the minimal intervention on the
actionable input m such that the predicted postprandial
glucose response remains within a clinically acceptable
range.
Inspired by prior work on counterfactual explanations
and intervention modeling, we require the generated meal
m∗to satisfy the following properties:
•Interventional: The modified meal must achieve the
desired glycemic outcome, i.e., f(x,m∗)≤τ.
•Minimal: The modification from the original meal m0
should be as small as possible.
•Actionable: Only meal-related variables are modified,
while the pre-meal context xremains fixed.
•Plausible: The modified meal must satisfy nutritional
and practical constraints, ensuring feasibility in real-
world settings.
T o implement the CF optimization framework, one
straightforward option can be using an existing CF
framework like DiCE [24], NICE [25] or CFNOW [26].
However, to exert more control over the optimization (e.g.
direction and limit of changes) and to generate realistic
looking CF s (e.g. not suggesting to increase carbohydrate
intake to reduce post-meal BGL), we choose to design
our own optimization method. W e formulate this as a
constrained optimization problem using a differentiable
objective function:
L(m) = [max (0 , f(x,m)−τ)]2
| {z }
target loss+λX
j∈Amj−m0,j
sj2
| {z }
distance regularizer,
(10)
where the first term enforces the glycemic target by
penalizing predictions that exceed the threshold τ, and
the second term is a distance regularizer that encourages
minimal deviation from the original meal m0. Here, A
denotes the set of actionable nutritional components (e.g.,
carbohydrates, protein, and fat), sjis a scaling factor
for normalization across different nutrient ranges, and
λ controls the trade-off between achieving the glycemic
target and preserving similarity to the original meal.
The optimization is subject to domain-specific con-
straints that ensure nutritional validity and practical
feasibility:
mC≤m0,C,
mP≤Pmax,
mF≤Fmax,
m∈ M ,(11)
where mC,mP, and mFdenote carbohydrate, protein, and
fat components of the meal, respectively . The constraint

4 IEEE JOURNAL OF BIOMEDICAL AND HEAL TH
INFORMA TICS
mC≤m0,C reflects the dominant role of carbohydrate
intake in postprandial glucose excursions, while Pmax
and Fmax impose upper bounds to maintain nutritional
balance. The feasible set M further ensures that the
optimized meal adheres to dietary guidelines and practical
consumption constraints.
The optimized solution is obtained as:
m∗= arg min
mL(m). (12)
This formulation differs from prior counterfactual ap-
proaches that operate on high-dimensional behavioral or
clinical feature spaces and often produce low-level recom-
mendations (e.g., adjusting macronutrient quantities or
treatment parameters). In contrast, MetaPlate constrains
the intervention space to meal composition and explicitly
bridges the gap between numerical optimization and
actionable dietary guidance. The optimized macronutrient
targets obtained from this step are subsequently translated
into food-level recommendations through an LLM-based
retrieval and reasoning module.
C. LLM-Based Retrieval-Augmented F ood
Recommendation
While the counterfactual optimization produces an
adjusted macronutrient vector m∗, these outputs are not
directly interpretable by end users. T o bridge this gap, we
introduce a RAG layer that maps optimized nutritional
targets to concrete food-level recommendations.
W e construct a structured food database F derived from
the USDA F oodData Central repository , where each food
item is associated with its nutritional profile. Given the
optimized macronutrient target m∗, a retrieval module
identifies candidate food combinations whose aggregate
nutritional composition approximates m∗.
F ormally , the retrieval step identifies a set of food items
{f1, f2, . . . , f k} ⊂ F such that:
X
jϕ(fj)≈m∗, (13)
where ϕ(fj)denotes the nutrient vector of food item fj.
An LLM is then used to generate human-readable meal
recommendations conditioned on the retrieved candidates
and user context. The LLM refines the output by adding
dietary coherence, portion sizes, and natural language
explanations, producing a final recommendation that is
interpretable, actionable, and aligned with the optimized
nutritional targets.
This RAG-based layer enables MetaPlate to translate
low-level counterfactual outputs into practical dietary
guidance suitable for real-world use.
D. Prompt Optimization
A baseline prompt was initially designed to generate
meal recommendations that satisfy target macronutrient
constraints using foods from the USDA F oodData Central
database. The outputs produced by this prompt werereviewed by a cohort of domain experts with backgrounds
in nutrition and dietetics.
Based on this evaluation, the prompt was iteratively
refined to address identified limitations, including issues
related to clinical plausibility and meal composition.
Specifically , additional constraints were added to enforce
balanced meal structure (e.g., inclusion of protein, carbo-
hydrate, and non-starchy produce components), improve
portion realism, and discourage implausible food combi-
nations.
The finalized prompt, reflecting these refinements, is
used in all subsequent experiments.
E. Pipeline
Figure 1presents an overview of the MetaPlate pipeline
for personalized meal recommendation under glycemic
constraints. The framework consists of five stages: data
collection, data processing, model development, CF opti-
mization, and LLM-based RAG recommendation.
In the data collection stage, multimodal data is obtained
from healthy adults in free-living conditions using CGM,
wearable sensors, and smartphone-based meal logs. These
data include physiological signals (e.g., glucose, activity ,
heart rate) and meal information. The data processing
stage performs cleaning, synchronization, missing value
handling, and feature engineering to construct pre-meal
context and meal representations.
In the model development stage, a predictive function is
learned to estimate 2-hour postprandial glucose peak given
the pre-meal context and meal. This model serves as the
basis for subsequent optimization and is not restricted to
a specific architecture.
The CF optimization stage modifies the planned meal
to obtain a minimally changed alternative that satisfies a
target glycemic constraint. The optimization is performed
over macronutrient components under feasibility and nu-
tritional constraints.
Finally , the LLM-based RAG module maps the op-
timized macronutrient targets to food-level recommen-
dations by retrieving candidate items from the USDA
database and generating interpretable meal suggestions
with portioning and explanations.
IV. D ATA
A. Data Collection
Data collection was conducted between May 2025 and
April 2026 involving 13 healthy individuals with no self-
reported history of diabetes (IRB #15102). All partici-
pants were familiar with meal logging. F ollowing informed
consent, participants were equipped with a Dexcom G6
Pro CGM sensor worn on the upper arm and an Embrace
Plus wristband worn on the dominant wrist. The wrist-
band was recharged every other night. Participants used
the MyFitnessPal Calorie T racker application on their
smartphones to log dietary intake.
F or each participant, approximately 10 days of recording
was collected under free-living conditions. The dataset

AREFEEN et al.: MET APLA TE: COUNTERF ACTUAL-GUIDED PERSONALIZED FOOD RECOMMENDA TIONS 5
Free-living condition
Counterfactual optimization
RAG-LLM framework
Cleaning & Synchronization•Handle missing data•Align timestamps across sources•Resample to common intervalFeature Engineering•Time window aggregation (last 2h)•Activity, glucose, meal features•Setting X (features), y (target)•Train (75%) and test (25%) split
Model Training & Validation•Algorithm selection (XGBoost, NN)•Train on test set•Tune hyperparameters•Evaluation: RMSE, MAE, MAPE, R2
1. Data Collection
2. Model Development3. Meal Macronutrient Adjustment4. Human Translatable Information Retrieval 5. Healthy Meal Suggestion
Subject to:
Meal data
Glucose data
Mobility data
Activity data
Fig. 1.MetaPlate framework consists of multiple phases: (1) data acquisition from healthy adults in free-living condition using CGM sensor,
wristband and smartphone application, (2) data processing, feature engineering, model training and validation for forecasting post-meal
blood glucose peak, (3) CF optimization for meal macro-nutrient adjustment to achieve in-range post-meal glucose level, (4) LLM-RAG
module to retrieve human understandable healthy meals from USDA database satisfying constraints generated through CF optimization.
includes continuous glucose measurements from the CGM,
physiological signals (pulse rate, skin temperature, and
electrodermal activity), and activity-related measures
(step count, metabolic equivalents (MET s), and activity
counts) from the wristband, as well as meal timing and
macro- and micronutrient information from food logs. In
total, the dataset comprises approximately 3120 hours of
glucose data and 636 logged meals.
Data from all 13 participants (age: 26.46±6.81 years; 10
female; BMI: 26.23±5.65) is used for subsequent analysis
and model development. T able Isummarizes participant
demographics.
B. Data Processing
F or robust modeling of postprandial glycemic response
under free-living conditions, a multi-stage preprocessing
pipeline is developed to synchronize, clean and transformTABLE I
DEMOGRAPHIC INFORMATION OF THE DATASET COLLECTED FROM
HEALTHY ADULTS .
nAge
(mean ±SD)Gender
(F/M)Height
(mean ±SD)
in.Weight
(mean ±SD)
lbs.BMI
(mean ±SD)
1326.46 ±6.8110/3 65.19 ±3.59158.69 ±34.1826.23 ±5.65
multimodal data sources: glucose, physiological, activity
data and nutrition logs.
1) Data Synchronization and Alignment :Data streams
from CGM, wristband, and nutrition logs are first tempo-
rally aligned. Since Embrace Plus records data in UTC,
all timestamps are converted to a common local timezone
to ensure consistency with CGM and dietary logs. The
CGM-derived signals are then upsampled to a uniform

6 IEEE JOURNAL OF BIOMEDICAL AND HEAL TH
INFORMA TICS
TABLE II
EXAMPLES OF PROCESSED SAMPLES FROM THE DATASET . PREBGSLOPE , PREBGANDPOSTBGREFER TO PRE-MEAL BLOOD GLUCOSE
SLOPE , PRE-MEAL BLOOD GLUCOSE AND 2HPOST-MEAL MAX BLOOD GLUCOSE ,RESPECTIVELY . THE MODIFIABLE FEATURES ARE MARKED WITH
TEAL COLOR BOXES .
Age BMIStep count Activity count EDA Skin temp. Pulse ratePre BG slope Pre BG Carb Prot F at Post BG
mean std mean std mean std mean std mean std
40 28.6 17.6 21.1 71.1 42.2 3.8 4.6 31.9 0.7 87.7 9.7 -0.23 93 52 7 4 106
25 23.3 7.1 15.3 42.5 34.7 2.4 0.2 32.3 0.2 76.6 9.5 0.03 88 74 11.1 7.5 169
one-minute resolution to match the temporal granularity
of wristband data. Missing values arising from sensor
dropouts are handled using forward filling for short gaps
(≤30 minutes) and exclusion for longer discontinuities
(>30 minutes).
2) Meal Event Identification and Aggregation :Meal events
are extracted from nutrition logs. Due to the tendency
of users to log multiple food items within a short time
span, temporally adjacent meal entries occurring within
a 30-minute interval are grouped into a single meal
cluster. Each cluster is represented by its latest timestamp,
corresponding to the final logged food item. The total
macronutrient composition of the meal is computed as the
sum of carbohydrates, protein and fat across all entries
within the cluster.
3) F eature Extraction :F or each event at time tm, a
feature vector x is constructed using data from a two-
hour pre-meal window [tm−2h, tm). The extracted features
include statistical summaries (mean and standard devia-
tion) of wristband-derived signals, including step counts,
activity counts, MET s, EDA, skin temperature, and pulse
rate. Subjects’ age and BMI are also concatenated to x
as stationary features.
Additionally , glycemic context features are computed
from CGM data. Specifically , the pre-meal glucose level is
defined as the last observed glucose value prior to tm, and
the short-term glucose trend is quantified as the slope
of glucose values over the preceding 30-minute interval,
estimated via linear regression. Both features capture the
baseline glycemic state and recent dynamics prior to food
intake.
4) T arget V ariable Construction :The forecast target y
was defined as the maximum glucose value observed within
a two-hour postprandial window (tm, tm+ 2 h]. This def-
inition aligns with clinical understanding of postprandial
glucose excursions and captures peak glycemic response
following a meal.
5) T rain/test split :T o prevent subject-level data leakage
and ensure better generalization, the dataset is partitioned
at subject level. Specifically , a 10/3 subject-wise split is
followed, with ten participants being randomly selected
for model training and the remaining three participants
are set aside for evaluation and meal plan generation.
The data processing pipeline leaves us with 498 factual
samples. T rain/test split leaves 376 for training and 122
for test. T wo samples are shown in T able II as examples.
Of the 17 features, we consider Carb, Protein, and F at as
modifiable for meal recommendation.6) Supplementing the training data :Given the limited
size of the train dataset ( 376 samples), the training
data is supplemented with an additional dataset from
the MealMeter [27] project (IRB #15102) comprising 12
subjects ( n= 168 samples). In this cohort, participants
were monitored in a controlled lab setting while consuming
standardized meals and wearing a Dexcom G6 Pro and an
Empatica E4 wrist-worn device.
The integration of these datasets introduces a modality
mismatch due to differences in sensing platforms. The
original dataset, collected using the Empatica Embrace
Plus, provides device-derived features like step counts,
MET s, and activity counts by default. In contrast, the
Empatica E4 provides raw high-frequency tri-axial ac-
celerometry without such pre-computed features. Since
several activity-related features are not directly available
in the supplementary dataset, they are reconstructed from
accelerometry using established methods. Physical activ-
ity intensity is estimated using Euclidean Norm Minus
One (ENMO), computed from tri-axial acceleration and
expressed in milli-gravitational units (mg), which has been
widely adopted as a proxy for movement intensity in wrist-
worn accelerometry [28], [29]. Step counts are estimated
using a peak-detection approach applied to band-pass
filtered acceleration (0.5–3 Hz), with additional cadence-
based constraints to retain physiologically plausible walk-
ing periods. This approach is consistent with open-source
wrist accelerometry frameworks like GGIR and related
implementations [30].
In contrast, MET cannot be reliably estimated from
wrist accelerometry alone without population-specific cal-
ibration and additional physiological signals. Prior work
has demonstrated substantial variability in energy expen-
diture estimation from wrist-worn devices across activity
types and individuals [31], [32]. Therefore, MET is not
reconstructed and is instead omitted entirely from training
data to avoid introducing biased or unreliable estimates.
With instances from MealMeter, the training data has
544 samples.
V. E XPERIMENTAL SETUP
A. Model
The forecasting task is formulated as a supervised
regression problem. T o evaluate the effectiveness of the
proposed feature representation, we considered a set of
eight regression models spanning linear, ensemble, and
gradient-boosting paradigms: Ridge regression, Elastic

AREFEEN et al.: MET APLA TE: COUNTERF ACTUAL-GUIDED PERSONALIZED FOOD RECOMMENDA TIONS 7
Net, Random F orest, Extra T rees, Gradient Boosting,
Histogram-based Gradient Boosting, XGBoost, and Light-
GBM.
F or robust and reproducible model selection, hyperpa-
rameters for all models were optimized using randomized
search over predefined distributions. Specifically , 3-fold
cross-validation strategy is followed with shuffled splits
to evaluate candidate configurations, using root mean
squared error (RMSE) as the primary selection criterion.
After hyperparameter tuning, each model is retrained
using the optimal configuration on the full training dataset
and evaluated on the held-out test set.
B. Counterfactual Optimizer
Building on the aforementioned CF formulation, CF
optimization is implemented using differential evolution,
a population-based global optimizer. CF s are generated
independently for all eight predictive models by optimizing
the modifiable macronutrient variables, while keeping the
remaining pre-meal context fixed. T o improve robustness,
instances whose original predicted glucose response was
already below 140 mg/dL are skipped and not used
for CF generation. The search space is constrained to
physiologically plausible values using ϵ= 10−6as the
lower bound for all macronutrients. Differential evolution
is run with a population size of 15 and a maximum of 80
iterations.
C. LLMs
Some of the latest LLMs have been used to map
the CF generated macronutrient constraints into human
understandable food items-
•gemma-4-26b-a4b-it or Gemma 4 26B (April, 2026)
•GPT-5.4 (March, 2026)
•GPT-5.4-mini (March, 2026)
•llama-3.3-70b-versatile or Llama 3.3 70B (December,
2024)
•moonshotai-kimi-k2-instruct-0905 or Kimi K2 In-
struct (September, 2025)
D. V alidation Metrics
Different set of validation metrics have been used for
evaluating different components of the MetaPlate pipeline.
1) Regression model :The forecasting model is evaluated
using a set of standard regression performance metrics:
RMSE, MAE, median absolute error (MedAE), coeﬀicient
of determination ( R2), explained variance (Exp. V ar.),
Pearson correlation coeﬀicient ( r), MAPE, and sMAPE.
2) Counterfactuals :CF s are validated using the metrics
below-
V alidity assesses whether the generated CF s achieve the
desired glycemic outcome under a regression setting. A CF
is considered valid if the predicted postprandial glucose
level falls below a predefined evaluation threshold τeval=
140 mg/dL.validity =1
|X|X
(x,m0)∈X/x31 
fθ(x,m∗)≤τeval
(14)
where, X is the set of factual instances
Distance measures how close a counterfactual meal m∗
is to the original meal m0, computed over the actionable
feature set A. Lower distance indicates smaller and more
realistic modifications.
W e compute both L1andL2distances:
L1(m0,m∗) =X
j∈A|m∗
j−m0,j| (15)
L2(m0,m∗) =sX
j∈A(m∗
j−m0,j)2 (16)
T o ensure comparability across methods, both distances
are max-normalized across all valid counterfactuals:
˜Lk=Lk
max( Lk), k∈ {1,2} (17)
Sparsity measures the number of actionable features
that change between the original meal m0and the counter-
factual meal m∗. Lower sparsity improves interpretability
by minimizing the number of modifications required.
T o avoid counting negligible variations, a tolerance of 1
gram is applied. A feature is considered changed only if:
|m∗
j−m0,j|>1, j∈ A (18)
The sparsity is computed as:
sparsity =1
|CF valid|X
m∗∈CF validX
j∈A/x31 
|m∗
j−m0,j|>1
(19)
3) LLM mappings :The LLM-based meal mapping mod-
ule is evaluated along three dimensions-
Constraint Satisfaction (RMSE) measures how closely
the LLM generated meal matches the target macronutrient
profile using root mean squared error (RMSE) for each
macronutrient:
RMSE j=vuut1
NNX
i=1 
mLLM
ij−m∗
ij2(20)
where j∈ {C, P, F }denotes carbohydrates, protein, and
fat, respectively , m∗
ij is the target amount for nutrient j
in the i-th meal, and mLLM
ij is the corresponding amount
generated by the LLM. Lower RMSE indicates better
adherence to the target macronutrient constraints.
Glycemic Consistency assesses whether the generated
meals preserve the intended glycemic outcome by re-
evaluating them using the forecasting model. Specifically ,
we compute the proportion of generated meals that satisfy
the target glucose constraint:

8 IEEE JOURNAL OF BIOMEDICAL AND HEAL TH
INFORMA TICS
Consistency =1
NNX
i=1/x31 
fθ(xi,mLLM
i )≤τ
(21)
where N denotes the total number of generated meals,
/x31(·)is the indicator function, fθ(xi,mLLM
i )is the pre-
dicted postprandial glucose level for the i-th generated
meal, and τis the target glucose threshold.
F ood-Item Diversity quantifies diversity among the food
items produced by an LLM. Each generated food item
is embedded using a pretrained text embedding model
and compute the average pairwise distance between all
item embeddings. Let eidenote the embedding of food
item i, and let N be the total number of generated items.
Diversity is defined as:
Diversity =2
N(N−1)N−1X
i=1NX
j=i+1d(ei,ej) (22)
where d(·,·) is a Euclidean distance function in the
embedding space. Higher values indicate greater diversity
among the generated food items. F ood-item embeddings
are generated using a pretrained Sentence-BER T model
(all-MiniLM-L6-v2) [33].
4) Expert-Based V alidation of the Interventions :W e evalu-
ate the clinical relevance and practical applicability of the
generated meal interventions through expert assessment.
Case-level Evaluation: Experts were provided with the
subject context, predicted postprandial glucose response,
and the corresponding MetaPlate-generated meal recom-
mendation for each case. Each intervention was evaluated
along four dimensions:
•Glycemic appropriateness (ability to maintain post-
prandial glucose below target)
•Portion suitability
•Nutritional alignment with dietary guidelines
•Recommendation likelihood
Each dimension was rated on a 10-point Likert scale.
Performance is summarized by averaging ratings across
all cases and experts for each evaluation dimension.
Specifically , for a given criterion k, the average score is
computed as:
¯sk=1
NMNX
i=1MX
j=1sijk (23)
where N denotes the total number of evaluated cases,
M denotes the number of experts, and sijk represents the
rating assigned by the j-th expert to the k-th criterion for
thei-th case.
System-level Evaluation: In addition to case-level as-
sessment, MetaPlate is also evaluated with established
human-centered evaluation frameworks for explainable AI
[34]. Accordingly , experts evaluated the MetaPlate system
across five aspects:
•Ease of use
•Consistency with clinical knowledge
•T rustworthiness of recommendations•Practical usability
•Interpretability of meal composition
VI. R ESULTS
A. F orecasting Performance
Post-meal glucose forecasting performance across mod-
els is summarized in T able III . The predictive accuracies
are comparable. Therefore, the models are further assessed
based on the quality of CF s generated during the CF
optimization process using the models. As reported in
T able IV , LightGBM achieves the lowest RMSE toward
the target value of 140, the smallest l2distance, and
the second-lowest l1distance among all models. Hence,
LightGBM is used in all subsequent experiments.
B. Counterfactual Performance
T able V presents a comparative evaluation of the
proposed MetaPlate CF optimization framework against
two widely used baseline methods, namely W achter et al.
and DiCE, across four key metrics: validity , normalized
L1 and L2 distance and sparsity .
MetaPlate achieves the highest validity 0.660 , substan-
tially outperforming both W achter 0.540 and DiCE 0.500 .
This indicates that a larger proportion of generated CF
meals successfully satisfy the target glycemic constraint
(≤140 mg/dL). The improvement in validity can be
attributed to the explicit addition of domain-specific
constraints (e.g., limiting carbohydrate increase) and the
use of a target-aware loss function that directly penalizes
deviations from the glycemic threshold.
In terms of proximity to the original meal, MetaPlate
also yields lower normalized L1(0.230) andL2(0.258) dis-
tances compared to W achter (0.326 and 0.350) and DiCE
(0.419 and 0.465) . This means that MetaPlate produces
more minimal, realistic modifications and preserves the
structure of the original meal while achieving the desired
outcome. In contrast, DiCE exhibits the largest distances,
which indicates substantial deviations from the original
meal composition and may reduce practical feasibility
despite its ability to generate diverse CF s.
DiCE achieves the lowest sparsity (1.800) i.e., fewer
features are modified on average. However, this comes at
the cost of reduced validity and significantly larger per-
turbations in magnitude. MetaPlate exhibits a balanced
trade-off between the number of macro-nutrient changes
and their magnitude by maintaining a competitive sparsity
level (2.030) , slightly higher than DiCE but lower than
W achter (2.222) . Importantly , the slightly higher sparsity
in MetaPlate reflects deliberate adjustments across mul-
tiple macronutrients to ensure both physiological plausi-
bility and adherence to glycemic constraints.
Our analysis shows that MetaPlate achieves a favorable
balance between effectiveness and practicality , generating
counterfactual meal recommendations that are more likely
to meet glycemic targets while requiring smaller and more
logical modifications compared to existing CF methods.

AREFEEN et al.: MET APLA TE: COUNTERF ACTUAL-GUIDED PERSONALIZED FOOD RECOMMENDA TIONS 9
TABLE III
REGRESSION PERFORMANCE ON THE TEST SET . LOWER IS BETTER FOR RMSE, MAE, M EDAE, MAPE, AND S MAPE; HIGHER IS BETTER FOR
R2, EXPLAINED VARIANCE ,ANDPEARSON r.
ModelRMSE
↓MAE ↓MedAE
↓R2↑Exp. V ar.
↑Pearson r
↑MAPE (%)
↓sMAPE (%)
↓
ElasticNet 16.458 12.586 10.003 0.472 0.473 0.689 8.832 8.843
ExtraT rees 16.573 12.552 10.234 0.464 0.465 0.691 8.951 8.848
XGBoost 16.602 12.588 10.587 0.463 0.465 0.685 8.813 8.821
Ridge 16.975 12.867 10.550 0.438 0.447 0.685 8.897 9.019
GradientBoosting 17.532 13.406 11.515 0.401 0.401 0.651 9.526 9.448
LightGBM 17.559 13.651 11.501 0.399 0.402 0.643 9.630 9.663
RandomF orest 17.721 13.673 11.698 0.388 0.388 0.631 9.781 9.631
HistGradientBoosting 17.896 13.571 11.151 0.375 0.379 0.615 9.545 9.525
TABLE IV
MODEL COMPARISON USING NORMALIZED l1,l2DISTANCE AND RMSE
TO TARGET 140. L OWER SCORES ARE BETTER
Model RMSE to 140 ↓ l1Dist. ↓ l2Dist. ↓
ElasticNet 14.487 0.331 0.355
ExtraTrees 11.537 0.225 0.266
GradientBoosting 7.816 0.183 0.228
HistGradientBoosting 11.414 0.140 0.187
LightGBM 7.030 0.142 0.176
RandomForest 9.807 0.166 0.210
Ridge 7.067 0.352 0.401
XGBoost 8.970 0.183 0.221
C. LLM Performance
The LLMs are assessed on the basis of macronutrient
constraint satisfaction, glycemic consistency , and food-
item diversity . The results are listed in T able VI-B .
The radar plot in Figure 2 provides a normalized
visualization of the same results, where the RMSE-based
metrics are inverted such that higher values indicate better
adherence to the target nutritional profile.
Among the evaluated LLMs, Gemma 4 26B achieves
the best overall nutritional fidelity , with the low-
est carbohydrate, protein, and fat RMSE values
(0.187,0.147,and 0.211 , respectively ) and the highest
glycemic consistency (0.700) . This indicates that the
LLM most accurately maps the optimized macronutrient
targets to feasible meal suggestions while preserving the
intended glycemic effect. The corresponding radar profile
in Figure 2is the most balanced and expansive across
the accuracy , consistency and diversity axes, reflecting its
superior alignment with the CF targets.
In contrast, GPT-5.4, GPT-5.4-mini, and Llama 3.3
70B exhibit substantially larger nutrient RMSE values,
indicating weaker adherence to the prescribed macronu-
TABLE V
COMPARISON OF CFMETHODS USING VALIDITY ,NORMALIZED
DISTANCE ,AND SPARSITY
Method Validity ↑ L1Dist. ↓ L2Dist. ↓Sparsity ↓
MetaPlate 0.660 0.230 0.258 2.030
Wachter 0.540 0.326 0.350 2.222
DiCE 0.500 0.419 0.465 1.800trient constraints. Although Llama 3.3 70B attains a
moderate glycemic consistency of 0.440 , its nutrient-level
errors remain comparatively high, suggesting that the
generated meals are less tightly aligned with the optimized
nutritional profile. The radar plot reflects this behavior
through a narrower polygon on the nutrient accuracy
dimensions.
Kimi K2 Instruct shows an intermediate profile. While
its carbohydrate, protein, and fat RMSE values are larger
than those of Gemma 4 26B, it attains the highest food-
item diversity (1.105) and the largest number of unique
items used for diversity analysis (121) , indicating broader
lexical and compositional variability in the generated rec-
ommendations. However, this diversity does not translate
into the best macronutrient matching, which suggests a
trade-off between recommendation variety and constraint
fidelity . This pattern is also evident in Figure 2, where the
diversity axis is relatively strong despite weaker nutrient-
accuracy axes.
Based on our analysis, Gemma 4 26B appears to be
the most suitable LLM for the MetaPlate retrieval-and-
generation layer as it provides the most reliable balance
between nutritional accuracy and glycemic consistency .
At the same time, the diversity observed in Kimi K2
Instruct suggests that models with broader generative
variability may be useful when recommendation richness
is prioritized, albeit at the expense of tighter adherence
to the target macronutrient constraints.
D. Quality of the Interventions
W e evaluate the quality of MetaPlate-generated meal
recommendations through a two-stage expert assessment
involving registered dietitians. In the first round, four
experts evaluated the meals generated using an initial
prompt formulation. Based on their feedback, the prompt
was redesigned by the team and the new meals are re-
evaluated in a second round with six experts. All eval-
uators are registered dietitians with substantial clinical
experience.
Figure 3 presents a comparison of expert ratings
before and after prompt refinement across both case-
level and system-level evaluation dimensions. At the case
level, consistent improvements are observed across all

10 IEEE JOURNAL OF BIOMEDICAL AND HEAL TH
INFORMA TICS
TABLE VI
LLM P ERFORMANCE COMPARISON ACROSS NUTRITIONAL ACCURACY , GLYCEMIC CONSISTENCY ,ANDDIVERSITY
LLM nCarb RMSE ↓Prot RMSE ↓Fat RMSE ↓Gly. Const. ↑Diversity ↑# Items ↑Unique items ↑
Gemma 4 26B 50 0.187 0.147 0.211 0.700 1.123 199 80
GPT-5.4 50 4.215 2.753 1.729 0.320 1.040 192 37
GPT-5.4-mini 50 6.334 2.941 3.189 0.320 0.962 185 27
Llama 3.3 70B 50 11.168 3.216 5.012 0.440 0.952 171 26
Kimi K2 Instruct 50 8.596 8.580 6.652 0.520 1.105 195 121
carb accuracy
protein accuracy
fat accuracy glycemic consistencyfood diversitygemma-4-26b-a4b-it
gpt-5.4
gpt-5.4-mini
llama-3.3-70b-versatile
moonshotai-kimi-k2-instruct-0905
Fig. 2. LLM comparison across RMSE, glycemic consistency,
and diversity. For visualization purposes, RMSE-based metrics are
normalized and inverted to accuracy such that higher values indicate
better performance. Specifically, lower RMSE corresponds to higher
normalized accuracy in the radar plot.
0 2 4 6 8 10
ScoreMeal compositionUsabilityTrustworthinessConsistency with
clinical knowledgeEase of useRecommendation
likelihoodAlignment with
dietary guidelinesPortion suitabilityGlycemic
appropriateness
3.25
6.72±2.77
±2.753.00
6.43±2.92
±2.763.75
7.00±2.17
±2.383.00
7.00±2.92
±1.833.50
7.71±3.20
±2.143.45
7.10±2.94
±2.365.38
7.97±2.50
±1.624.88
7.67±2.59
±2.067.15
8.04±1.76
±1.79Before
After
Fig. 3. Comparison of expert evaluation scores before and after
prompt refinement across case-level (red) and system-level (blue)
dimensions.Ratingsarereportedona10-pointLikertscalewitherror
bars indicating standard deviation across experts and cases. Sub-
stantial improvements are observed across all dimensions following
prompt redesign, particularly in portion suitability, recommendation
likelihood, ease of use, and trustworthiness. Impact of expert-in-the-
loop refinement are also noticeable on the realism, usability, and
clinical alignment of MetaPlate-generated meal recommendations.
four criteria. Glycemic appropriateness improves modestly
(7.15 to8.04), indicating that even the initial formulation
was reasonably aligned with glycemic goals. However, sub-
stantially larger gains are observed in portion suitability
(4.88 to7.67) , nutritional alignment (5.38 to7.97) , and
recommendation likelihood (3.45 to7.10) . The trends sug-gest that the primary limitations of the initial system were
not in glycemic correctness, but in producing realistic,
nutritionally coherent, and practically acceptable meals.
System-level evaluation further reinforces this trend.
Metrics like ease of use (3.50 to7.71) , consistency
with clinical knowledge (3.00 to7.00) , trustworthiness
(3.75 to7.00) , usability (3.00 to6.43) , and meal com-
position quality (3.25 to7.72) all show substantial im-
provement after refinement. Notably , the largest gains are
observed in ease of use and perceived clinical validity . This
means that the resulting meals from the redesigned system
is significantly more aligned with expert expectations for
real-life applicability .
The improvements can be attributed to changes made
during prompt redesign. The initial prompt emphasized
strict macronutrient matching, which often resulted in
unrealistic or fragmented meal compositions, including
snack-like outputs, excessive reliance on filler ingredients
(e.g., nuts, almonds), and implausible portion sizes. In
contrast, the redesigned prompt explicitly enforces meal
structure (protein + carbohydrate + produce), realistic
portioning, and dietary coherence, while slightly relax-
ing strict macro constraints when necessary to preserve
plausibility . Additionally , it prioritizes clinical realism and
balanced meal composition over exact numerical matching,
and introduces explicit checks for meal validity .
The results show that expert-in-the-loop prompt refine-
ment significantly enhances the quality , interpretability ,
and practical usability of generated interventions. This is
consistent with the fact that co-creation and integration
of domain knowledge and human feedback is necessary
when deploying LLM-based systems for clinically relevant
decision support.
In addition to quantitative improvements, qualitative
analysis of expert feedback reveals a clear shift in the
nature and severity of system limitations between the
two evaluation rounds. In the initial evaluation, feedback
was dominated by fundamental structural issues, including
non-meal-like outputs (e.g., “this is a snack, not a meal”),
unrealistic food combinations (e.g., “meat and almonds
only”), and extreme macronutrient modifications that
were deemed clinically impractical (e.g., excessive car-
bohydrate restriction). Experts frequently questioned the
plausibility , palatability , and clinical applicability of the
recommendations and indicated that the system outputs
were often not usable in real-world settings.
F ollowing prompt refinement, these concerns were sub-
stantially reduced. F eedback from the second round shifted

AREFEEN et al.: MET APLA TE: COUNTERF ACTUAL-GUIDED PERSONALIZED FOOD RECOMMENDA TIONS 11
toward fine-grained adjustments, like modest portion tun-
ing, minor macronutrient balancing, and flavor enhance-
ments. Experts more frequently described the recommen-
dations as “reasonable”, “balanced”, and “clinically ap-
propriate”, with suggestions focusing on improving rather
than rejecting the outputs. This transition from rejection-
level feedback to refinement-level feedback indicates that
the redesigned system successfully addresses core struc-
tural and plausibility issues, resulting in recommendations
that are both interpretable and actionable in practice.
These qualitative observations align with the quantita-
tive improvements reported in Figure 3, where the largest
gains are observed in portion suitability , nutritional
alignment, and recommendation likelihood — dimensions
directly related to meal realism and practical usability .
W e categorize the primary failure modes of the initial
system into three groups: (1) structural issues, includ-
ing generation of snack-like or incomplete meals; (2)
quantitative inconsistencies, like unrealistic macronutrient
distributions and portion sizes; and (3) practical usability
concerns, including lack of dietary coherence, variety ,
and palatability . These failure modes are substantially
mitigated after prompt refinement.
Notably , the absence of repeated concerns such as
over-reliance on almonds and lack of meal structure in
the second round suggests that including explicit meal
composition constraints and diversity controls effectively
mitigates common failure modes of LLM-based food gen-
eration.
VII. D ISCUSSION
Motivation for LLM-RAG-Based F ood Mapping: Map-
ping user-provided meal descriptions to concrete food
items in a structured database (e.g., USDA F oodData
Central) is inherently challenging due to the noisy , het-
erogeneous, and context-dependent nature of food text.
Prior work has shown that naive or straightforward
lexical matching approaches struggle to accurately identify
specific food items from such databases, as superficial word
overlap can lead to incorrect matches and misclassification
in the presence of brand names, preparation details, and
ambiguous terminology [35]. In the context of MetaPlate,
a purely lexical or rule-based retrieval approach would
be insuﬀicient as it cannot reason over combinations of
foods, provide healthy food suggestions, enforce dietary
coherence, or ensure that aggregated nutrient profiles align
with the target constraints. T o overcome this challenges,
MetaPlate uses a LLM-based RAG module that enables
semantic matching and compositional reasoning over can-
didate foods.
Implications for clinical usability and adoption: Expert
feedback also highlights an important consideration for
clinical adoption: systems that recommend entirely new
meals may face resistance if they do not align with users’
existing dietary habits. Instead, approaches that preserve
familiar food choices while suggesting realistic modifica-
tions are more consistent with human adaptability based
on dietary counseling practices.Design considerations for LLM-based dietary systems:
Insights from experts illustrates a broader limitation of
LLM-based generation systems in healthcare applications,
where unconstrained optimization may produce techni-
cally valid but clinically implausible outputs. Integrat-
ing domain-specific constraints, structured reasoning, and
expert feedback is therefore prerequisite to ensure both
accuracy and usability before deployment.
VIII. L IMITATIONS AND FUTURE PLAN
Limited dataset size: The current study is based on
a relatively small cohort collected under an ongoing data
acquisition effort, which may limit the generalizability and
robustness of the learned models. Although we partially
mitigate this constraint through data augmentation using
an external dataset, the overall sample size remains
modest compared to large-scale studies. As a result, the
predictive and counterfactual components may not fully
capture the variability in glycemic responses across diverse
populations and contexts. In future work, we plan to
expand the cohort by recruiting additional participants
and collecting longer-duration recordings under free-living
conditions. This will unlock the development of more ro-
bust models, including deep learning architectures capable
of capturing complex temporal and multimodal dependen-
cies, and support more comprehensive and statistically
rigorous evaluations of the proposed framework.
Evaluation in clinical trials: While MetaPlate demon-
strates promising performance in offline evaluation and
expert-based assessments, it has not yet been validated
in an active intervention setting. As such, its practical
effectiveness, user adherence, and feasibility under real-life
constraints, like dietary preferences, accessibility of food
items, and behavioral factors—remain to be established.
The current evaluation does not capture longitudinal
outcomes or the impact of repeated system use on glycemic
control. T o address this limitation, future work will involve
deploying MetaPlate as a smartphone-based application
and conducting a prospective clinical trial. Specifically ,
we plan to collect an initial baseline period of four weeks
without intervention, followed by a four-week interven-
tion phase during which participants will actively use
MetaPlate for meal decision support. Assessment will
include real-world usability , adherence, and effectiveness
in maintaining in-range postprandial glucose levels.
LLM-generated meal inconsistencies: Despite prompt
refinement and constraint-based generation, the LLM
occasionally produces meal suggestions that deviate from
intended guidelines, like including restricted items (e.g.,
nuts or almonds) or generating snack-like combinations in-
stead of structured meals. These inconsistencies highlight
limitations in controllability and faithfulness of current
LLM outputs, particularly when translating numerical
macronutrient targets into discrete food items. Although
rare, discrepancies may arise between the generated meal
composition and the corresponding nutritional values due
to hallucination or imperfect grounding in the underlying
food database. T o address this limitation, future work

12 IEEE JOURNAL OF BIOMEDICAL AND HEAL TH
INFORMA TICS
will incorporate a post-generation verification and filtering
module that cross-checks generated meals against the
USDA F oodData Central database. This step will ensure
that selected food items and portion sizes accurately
reflect the target macronutrient composition and adhere
to predefined dietary constraints, thereby improving reli-
ability and reducing hallucination in the recommendation
pipeline.
F ood availability and practicality constraints: The cur-
rent formulation of MetaPlate assumes access to the
recommended food items, which may not always hold in
real scenarios due to availability , cost, or user preferences.
As a result, some generated meal suggestions may be
impractical or diﬀicult to adopt at the time of decision-
making. F urthermore, the system currently focuses on
generating alternative meals rather than adapting to what
the user already has available. T o address this limitation,
future work will introduce mechanisms for availability-
aware recommendation, where users can specify accessible
food items or constraints. Additionally , the system can be
extended to support modification-based recommendations
by adjusting components of the user’s intended meal (e.g.,
reducing portion sizes or substituting specific ingredients).
F or instance, if a user plans to consume a meal such as
chicken with rice, MetaPlate could suggest reducing the
rice portion or rebalancing macronutrients to achieve the
desired glycemic outcome. Enhancements like these would
improve the practicality and applicability of the system.
Device-specific generalizability: MetaPlate is currently
developed using data from specific devices (Dexcom G6
Pro and Empatica Embrace Plus), which may limit its
generalizability across different sensing platforms. W e plan
to extend the framework to widely available consumer
devices like Dexcom G7, Abbott F reeStyle Libre 3, Apple
W atch and Samsung Galaxy W atch, which offer compa-
rable sensing modalities. This transition is expected to be
feasible given the overlap in sensor capabilities and would
improve scalability and real-life adoption.
IX. C ONCLUSION
This paper presents MetaPlate, a counterfactual-guided,
context-aware decision-support framework for personal-
ized meal recommendation to mitigate postprandial hy-
perglycemia in healthy adults. MetaPlate extends be-
yond prediction to provide actionable dietary interven-
tions. The framework formulates meal recommendation
as a constrained counterfactual optimization problem and
translates optimized macronutrient targets into human-
understandable meal suggestions through an LLM-based
RAG layer grounded in the USDA F oodData Central
database. Experimental results demonstrate that Meta-
Plate generates valid and minimally modified counter-
factual meals that satisfy glycemic constraints, while
expert evaluation by registered dietitians confirms the
clinical appropriateness, nutritional validity , and practical
usability of the recommendations. F urthermore, expert-in-
the-loop prompt refinement significantly enhances the re-
alism, coherence, and trustworthiness of generated meals,underscoring the importance of incorporating domain
expertise in LLM-driven health systems. MetaPlate holds
potential as a scalable and interpretable tool for real-time,
personalized dietary guidance and early prevention of
glycemic dysregulation, with future work directed toward
clinical validation and deployment in real-world settings.
ACKNOWLEDGMENT
This work was supported in part by the National
Institute of Diabetes and Digestive and Kidney Dis-
eases of the National Institutes of Health under grant
T32DK137525 and the National Science F oundation under
grant IIS-2402650. Any opinions, findings, conclusions, or
recommendations expressed in this material are those of
the authors and do not necessarily reflect the views of the
funding organization.
[1]P. R. E. Jarvis, J. L. Cardin, P. M. Nisevich-Bede, and
J. P. McCarter, “Continuous glucose monitoring in a healthy
population: understanding the post-prandial glycemic response
in individuals without diabetes mellitus.” Metabolism: clinical
and experimental, p. 155640, 2023.
[2]B. Giri, S. Dey, T. Das, M. Sarkar, J. Banerjee, and S. K. Dash,
“Chronic hyperglycemia mediated physiological alteration and
metabolic distortion leads to organ dysfunction, infection, can-
cer progression and other pathophysiological consequences: An
update on glucose toxicity.” Biomedicine & pharmacotherapy =
Biomedecine & pharmacotherapie, vol. 107, pp. 306–328, 2018.
[3]E. B. Levitan, Y. Song, E. S. Ford, and S. Liu, “Is nondiabetic
hyperglycemia a risk factor for cardiovascular disease? a meta-
analysis of prospective studies.” Archives of internal medicine,
vol. 164 19, pp. 2147–55, 2004.
[4]N. E. Phillips, T.-H. Collet, and F. Naef, “Uncovering person-
alized glucose responses and circadian rhythms from multiple
wearable biosensors with bayesian dynamical modeling,” Cell
Reports Methods, vol. 3, 2023.
[5]D. A. Zeevi, T. Korem, N. Zmora, D. Israeli, D. Roth-
schild, A. Weinberger, O. Ben-Yacov, D. Lador, T. Avnit-Sagi,
M. Lotan-Pompan, J. Suez, J. A. Mahdi, E. Matot, G. Malka,
N. Kosower, M. Rein, G. Zilberman-Schapira, L. Dohnalová,
M. Pevsner-Fischer, R. Bikovsky, Z. Halpern, E. Elinav, and
E. Segal, “Personalized nutrition by prediction of glycemic
responses.” Cell, vol. 163 5, pp. 1079–1094, 2015.
[6]W. P. van Doorn, Y. D. Foreman, N. C. Schaper, H. H.
Savelberg, A. Koster, C. J. H. van der Kallen, A. Wesselius,
M. T. Schram, R. M. A. Henry, P. C. Dagnelie, B. E. de Galan,
O. Bekers, C. D. A. Stehouwer, S. J. R. Meex, and M. C.
Brouwers, “Machine learning-based glucose prediction with use
of continuous glucose and physical activity monitoring data:
The maastricht study,” PLoS ONE, vol. 16, 2021.
[7]E. Farahmand, R. R. Azghan, N. T. Chatrudi, E. Kim,
G. K. Gudur, E. Thomaz, G. Pedrielli, P. K. Turaga, and
H. Ghasemzadeh, “Attengluco: Multimodal transformer-based
bloodglucoseforecastingonai-readidataset,”202547thAnnual
International Conference of the IEEE Engineering in Medicine
and Biology Society (EMBC), pp. 1–7, 2025.
[8]A. Machiraju, E. Farahmand, S. B. Soumma, A. Arefeen,
C.Johnston,andH.Ghasemzadeh,“Time-awarecross-attention
for multi-modal sensor-based blood glucose forecasting,” 2025
IEEE 21st International Conference on Body Sensor Networks
(BSN), pp. 1–4, 2025.
[9]T. Zhu, L. Kuang, C. Piao, J. Zeng, K. Li, and P. Georgiou,
“Population-specific glucose prediction in diabetes care with
transformer-based deep learning on the edge,” IEEE Transac-
tions on Biomedical Circuits and Systems, vol. 18, pp. 236–246,
2024.
[10]P.Shroff,A.Arefeen,andH.Ghasemzadeh,“Glucoseassist:Per-
sonalized blood glucose level predictions and early dysglycemia
detection,” 2023 IEEE 19th International Conference on Body
Sensor Networks (BSN), pp. 1–4, 2023.

AREFEEN et al.: MET APLA TE: COUNTERF ACTUAL-GUIDED PERSONALIZED FOOD RECOMMENDA TIONS 13
[11]S. Khamesian, A. Arefeen, M. A. Grando, B. Thompson, and
H. Ghasemzadeh, “Glycemic-aware and architecture-agnostic
training framework for blood glucose forecasting in type 1
diabetes,” 2025.
[12]S. B. Soumma and H. Ghasemzadeh, “Glyrag: Context-aware
retrieval-augmented framework for blood glucose forecasting,”
ArXiv, vol. abs/2601.05353, 2026.
[13]K.Kalpakoglou,L.Calderón-Pérez,N.Boqué,M.Guldas,Çağla
Erdoğan Demir, L. P. Gymnopoulos, and K. Dimitropoulos,
“An ai-based nutrition recommendation system: technical vali-
dation with insights from mediterranean cuisine,” Frontiers in
Nutrition, vol. 12, 2025.
[14]A. Arefeen, N. Jaribi, B. J. Mortazavi, and H. Ghasemzadeh,
“Computationalframeworkforsequentialdietrecommendation:
Integrating linear optimization and clinical domain knowledge,”
2022 IEEE/ACM Conference on Connected Health: Applica-
tions, Systems and Engineering Technologies (CHASE), pp. 91–
98, 2022.
[15]A. K. Gavai and J. van Hillegersberg, “Ai-driven personalized
nutrition: Rag-based digital health solution for obesity and type
2 diabetes,” PLOS Digital Health, vol. 4, 2025.
[16]S. Khamesian, A. Arefeen, S. M. Carpenter, and
H. Ghasemzadeh, “Nutrigen: Personalized meal plan generator
leveraging large language models to enhance dietary and
nutritional adherence,” 2025 47th Annual International
Conference of the IEEE Engineering in Medicine and Biology
Society (EMBC), pp. 1–7, 2025.
[17]Z. Yang, E. Khatibi, N. Nagesh, M. Abbasian, I. Azimi, R. Jain,
and A. M. Rahmani, “Chatdiet: Empowering personalized
nutrition-oriented food recommender chatbots through an llm-
augmented framework,” Smart Health, vol. 32, p. 100465, 2024.
[18]Z. Zhang, Z. Wang, T. Ma, V. S. Taneja, S. Nelson, N. H. L.
Le, K. Murugesan, M. Ju, N. V. Chawla, C. Zhang, and
Y. Ye, “Mopi-hfrs: A multi-objective personalized health-aware
food recommendation system with llm-enhanced interpreta-
tion,” Proceedings of the 31st ACM SIGKDD Conference on
Knowledge Discovery and Data Mining V.1, 2024.
[19]R. Crupi, A. Castelnovo, D. Regoli, and B. S. M. Gonzalez,
“Counterfactual explanations as interventions in latent space,”
Data Mining and Knowledge Discovery, vol. 38, pp. 2733 – 2769,
2021.
[20]A. Arefeen and H. Ghasemzadeh, “Designing user-centric be-
havioral interventions to prevent dysglycemia with novel coun-
terfactual explanations,” ArXiv, vol. abs/2310.01684, 2023.
[21]A. Arefeen, S. Khamesian, M. A. Grando, B. Thompson, and
H. Ghasemzadeh, “Glytwin: Digital twin for glucose control in
type 1 diabetes through optimal behavioral modifications using
patient-centric counterfactuals,” ArXiv, vol. abs/2504.09846,
2025.
[22]——, “Glyman: Glycemic management using patient-centric
counterfactuals,” 2024 IEEE EMBS International Conference
on Biomedical and Health Informatics (BHI), pp. 1–5, 2024.
[23]P. Barbiero, G. Ciravegna, F. Giannini, P. Li’o, M. Gori,
and S. Melacci, “Entropy-based logic explanations of neural
networks,” ArXiv, vol. abs/2106.06804, 2021.
[24]R. K. Mothilal, A. Sharma, and C. Tan, “Explaining ma-
chine learning classifiers through diverse counterfactual expla-
nations,” in Proceedings of the 2020 Conference on Fairness,
Accountability, and Transparency, 2020, pp. 607–617.
[25]D. Brughmans and D. Martens, “Nice: an algorithm for nearest
instance counterfactual explanations,” Data Mining and Knowl-
edge Discovery, pp. 1–39, 2021.
[26]R. M. B. de Oliveira, K. Sörensen, and D. Martens, “A
model-agnostic and data-independent tabu search algorithm to
generate counterfactuals for tabular, image, and text data,”
European Journal of Operational Research, 2023.
[27]A. Arefeen, S. N. Fessler, S. M. Mostafavi, C. Johnston,
and H. Ghasemzadeh, “Mealmeter: Using multimodal sensing
and machine learning for automatically estimating nutrition
intake,”202547thAnnualInternationalConferenceoftheIEEE
Engineering in Medicine and Biology Society (EMBC), pp. 1–6,
2025.
[28]V. T. van Hees, L. Gorzelniak, E. C. D. León, M. Eder, M. R.
Pias, S. Taherian, U. Ekelund, F. Renström, P. W. Franks,
A. Horsch, and S. Brage, “Separating movement and gravity
components in an acceleration signal and implications for theassessmentofhumandailyphysicalactivity,”PLoSONE,vol.8,
2013.
[29]M. Hildebrand, V. T. van Hees, B. H. Hansen, and U. Ekelund,
“Age group comparability of raw accelerometer output from
wrist- and hip-worn monitors.” Medicine and science in sports
and exercise, vol. 46 9, pp. 1816–24, 2014.
[30]V.T.vanHees,F.Renström,A.Wright,A.Gradmark,M.Catt,
K. Y. Chen, M. Löf, L. J. C. Bluck, J. Pomeroy, N. J. Wareham,
U. Ekelund, S. Brage, and P. W. Franks, “Estimation of daily
energy expenditure in pregnant and non-pregnant women using
a wrist-worn tri-axial accelerometer,” PLoS ONE, vol. 6, 2011.
[31]S. J. Strath, L. A. Kaminsky, B. E. Ainsworth, U. Ekelund,
P. S. Freedson, R. A. Gary, C. R. Richardson, D. T. Smith, and
A. M. Swartz, “Guide to the assessment of physical activity:
Clinical and research applications: a scientific statement from
the american heart association.” Circulation, vol. 128 20, pp.
2259–79, 2013.
[32]J. H. Migueles, C. Cadenas-Sánchez, U. Ekelund, C. D. Nys-
tröm, J. Mora-Gonzalez, M. Löf, I. Labayen, J. R. Ruiz, and
F. B. Ortega, “Accelerometer data collection and processing
criteria to assess physical activity and other outcomes: A sys-
tematic review and practical considerations,” Sports Medicine,
vol. 47, pp. 1821–1845, 2017.
[33]W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and
M. Zhou, “Minilm: Deep self-attention distillation for task-
agnostic compression of pre-trained transformers,” ArXiv, vol.
abs/2002.10957, 2020.
[34]A. Holzinger, A. M. Carrington, and H. Müller, “Measuring
the quality of explanations: The system causability scale (scs),”
Kunstliche Intelligenz, vol. 34, pp. 193 – 198, 2019.
[35]A. A. Metwally, A. K. Leong, A. Desai, A. Nagarjuna, D. Perel-
man, and M. P. Snyder, “Learning personal food preferences via
food logs embedding,” 2021 IEEE International Conference on
Bioinformatics and Biomedicine (BIBM), pp. 2281–2286, 2021.

IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS 1
Supplementary Materials for MetaPlate
I. M ODEL FINE-TUNING PROCEDURE
Hyperparameter optimization was conducted using
randomized search. For linear and tree-based models
(Ridge, Elastic Net, Random Forest, Extra Trees, Gradi-
ent Boosting, HistGradientBoosting, and XGBoost), we
used RandomizedSearchCV with 3-fold cross-validation
(KFold, n= 3, shufﬂe=True, random state=42), opti-
mizing for negative RMSE. Search budgets were set to
30 iterations for linear/tree models and 15 for boosting
models.
Search spaces were deﬁned using bounded uniform
and log-uniform distributions. Linear models were tuned
over regularization strength ( α), solver, tolerance, and
Elastic Net mixing ratio. Tree-based models were opti-
mized over the number of estimators, depth, minimum
samples per split/leaf, feature subsampling, and bootstrap
options. Gradient boosting models additionally included
learning rate and subsampling. XGBoost tuning included
learning rate, depth, subsampling, column sampling, and
regularization terms ( α,λ,γ), with monotonic con-
straints applied to clinically relevant features.
For LightGBM, a hold-out strategy was used for
efﬁciency: an 80/20 trainvalidation split with early stop-
ping (30 rounds). Hyperparameters were sampled us-
ingParameterSampler , and the best conﬁguration was
selected based on validation RMSE. The ﬁnal model
was retrained on the full training set using the optimal
number of boosting iterations. Monotonic constraints
were enforced for key physiological features (e.g., car-
bohydrate intake and pre-meal glucose).
II. S URVEY DETAILS
The survey evaluates the clinical usefulness, practi-
cality, and nutritional appropriateness of MetaPlate , a
decision-support system for designing meals that main-
tain postprandial blood glucose below 140 mg/dL.
For each case, the system:
•Reviews pre-meal context (e.g., blood glucose, ac-
tivity level)
•Predicts postprandial glucose response using a ma-
chine learning model
•Suggests an alternative meal to improve glycemic
outcomes
Experts evaluated 10 independent cases generated by
different LLMs and provided ratings.A. Participant Background
1) Professional Role :
•Registered Dietitian / Nutritionist
•Endocrinologist / Certiﬁed Diabetes Educator
•Physician
•Nurse / Nurse Practitioner
•Other
2) Y ears of Experience :
•0–2 years
•3–5 years
•6–10 years
•11–15 years
•>15 years
3) Comfort with Meal Design :Rated on a 10-point
Likert scale (1 = Not at all, 10 = Very comfortable).
B. Evaluation Criteria
For each case, participants rated the following (1–10
Likert scale):
1)Glycemic appropriateness (maintaining glucose
<140 mg/dL)
2)Portion size appropriateness
3)Alignment with dietary guidelines
4)Likelihood of recommendation
Participants could also provide optional free-text com-
ments.
C. Case Descriptions
1) Case 1 :Subject: 23 y, Female, BMI 32
Pre-meal: 51 g carb, 27.5 g protein, 21.7 g fat at 113.7
mg/dL
Predicted peak glucose: 147 mg/dL
MetaPlate Recommendation:
Roasted chicken breast (113 g), Brown rice (148 g),
Boiled broccoli (91 g), Olive oil (17 g)
Nutritional Summary: 43 g carbs, 32.5 g protein,
19.6 g fat, 475 kcal
2) Case 2 :Subject: 23 y, Female, BMI 32
Pre-meal: 29 g carb, 47 g protein, 8.3 g fat at 110 mg/dL
Predicted peak: 150 mg/dL
Recommendation: Chicken breast (140 g), white rice
(150 g), asparagus (100 g)
Nutritional Summary: ∼30 g carbs, ∼40 g protein,
∼6.3 g fat, ∼491 kcal

2 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS
TABLE I : Hyperparameter search spaces for forecasting models
Model Hyperparameters Search Space
Ridge , solver, tolerance, intercept ∼logU(10−3;103); solver ∈ {auto, svd, cholesky, lsqr, sag };
tolerance ∼logU(10−5;10−2); intercept ∈ {True, False }
Elastic Net ,l1ratio, selection, tolerance ∼logU(10−4;101);l1ratio∼U(0;1);
selection ∈ {cyclic, random }; tolerance ∼logU(10−5;10−2)
Random Forest estimators, depth, split, leaf, features, bootstrap estimators ∈[200 ;800]; depth ∈ {None, 3–20 };
split∈[2;15]; leaf∈[1;8]; features ∈ {sqrt, log2, None, 0.3–0.7 }
Extra Trees estimators, depth, split, leaf, features, bootstrap estimators ∈[300 ;1000] ; depth ∈ {None, 3–25 };
split∈[2;15]; leaf∈[1;8]; features ∈ {sqrt, log2, None, 0.3–0.7 }
Gradient Boosting estimators, learning rate, depth, subsample estimators ∈[100 ;600]; learning rate ∼logU(10−3;10−1);
depth∈[2;5]; subsample ∼U(0:6;1:0)
HistGradientBoosting iterations, learning rate, depth, regularization iterations ∈[100 ;600]; learning rate ∼logU(10−3;10−1);
depth∈ {None, 2–12 };L2∼logU(10−8;1)
XGBoost estimators, learning rate, depth, subsample, reg-
ularizationestimators ∈[100 ;700]; learning rate ∼logU(10−3;10−1);
depth∈[3;8]; subsample ∼U(0:7;1:0);
; ;  ∼logU(10−8;1)
LightGBM estimators, learning rate, leaves, depth, subsam-
ple, regularizationestimators ∈[300 ;1500] ; learning rate ∼logU(0:01;0:08);
leaves ∈[16;96]; depth ∈[3;10];
subsample, colsample ∼U(0:7;1:0);
; ∼logU(10−3;3); min split gain ∼logU(10−4;0:3)
3) Case 3 :Subject: 23 y, Female, BMI 32
Pre-meal: 112 g carb, 33 g protein, 42 g fat at 111
mg/dL
Predicted peak: 158 mg/dL
Recommendation: Chicken breast (155 g), sweet
potato (200 g), broccoli, olive oil
Nutritional Summary: ∼45 g carbs, ∼54 g protein,
∼38 g fat, ∼600 kcal
4) Case 4 :Subject: 28 y, Female, BMI 26.4
Pre-meal: 35 g carb, 13 g protein, 10 g fat at 118 mg/dL
Predicted peak: 160 mg/dL
Recommendation: Shrimp (155 g), asparagus (100
g), butter and olive oil
Nutritional Summary: ∼17.1 g carbs, ∼40.5 g pro-
tein,∼26 g fat, ∼465 kcal
5) Case 5 :Subject: 26 y, Female, BMI 22.2
Pre-meal: 83 g carb, 25 g protein, 24 g fat at 121 mg/dL
Predicted peak: 151 mg/dL
Recommendation: Salmon (90 g), sweet potato (155
g), berry sauce, walnuts
Nutritional Summary: ∼45 g carbs, ∼25 g protein,
∼25 g fat, ∼505 kcal
6) Case 6 :Subject: 26 y, Female, BMI 22.2
Pre-meal: 32 g carb, 7 g protein, 8 g fat at 137 mg/dL
Predicted peak: 153 mg/dL
Recommendation: Tuna (90 g), whole wheat crack-
ers, avocado
Nutritional Summary: ∼15.5 g carbs, ∼25 g protein,
∼9 g fat, ∼243 kcal7) Case 7 :Subject: 26 y, Female, BMI 22.2
Pre-meal: 20 g carb, 10 g protein, 6 g fat at 110 mg/dL
Predicted peak: 150 mg/dL
Recommendation: Greek yogurt, blueberries, al-
monds
Nutritional Summary: ∼17 g carbs, ∼11 g protein,
∼9.2 g fat, ∼195 kcal
8) Case 8 :Subject: 28 y, Female, BMI 26.4
Pre-meal: 101 g carb, 29 g protein, 25 g fat at 106
mg/dL
Predicted peak: 161 mg/dL
Recommendation: Salmon, quinoa, broccoli, olive oil
Nutritional Summary: ∼39 g carbs, ∼34 g protein,
∼26.4 g fat, ∼511 kcal
9) Case 9 :Subject: 28 y, Female, BMI 26.4
Pre-meal: 48 g carb, 37 g protein, 17 g fat at 105 mg/dL
Predicted peak: 148 mg/dL
Recommendation: Ground turkey, brown rice, green
beans, olive oil
Nutritional Summary: ∼40 g carbs, ∼39 g protein,
∼21 g fat, ∼505 kcal
10) Case 10 :Subject: 23 y, Female, BMI 32
Pre-meal: 45.5 g carb, 15.5 g protein, 15.5 g fat at 125
mg/dL
Predicted peak: 165 mg/dL
Recommendation: Greek yogurt, walnuts, honey,
egg, blueberries
Nutritional Summary: ∼22 g carbs, ∼20 g protein,
∼26 g fat, ∼402 kcal

AREFEEN et al. : GLYTWIN: ENHANCING DIGITAL TWIN FOR GLUCOSE CONTROL IN TYPE 1 DIABETES USING PATIENT-CENTRIC COUNTERFACTUAL
TREATMENTS 3
D. System-Level Evaluation
Participants rated the following statements (1 = Not
at all, 10 = Very much):
•MetaPlate is easy to use
•Recommendations are clinically consistent
•Recommendations are trustworthy
•Willingness to use in practice
•Recommendations are sufﬁciently explainable
Notes*
•All cases were evaluated independently
•No correct or incorrect answers were speciﬁed
•The cases in second round are different from those
in the ﬁrst round
III. S UMMARY OF EXPERTS ’ COMMENTS
Below, we present the GPT 5.5 summarized inputs
from the experts on the ten cases and on the MetaPlate
technology-
Expert feedback across all cases indicated that Meta-
Plate recommendations were generally aligned with clin-
ical goals for glycemic control and nutritional balance,
but several areas for improvement were identiﬁed. A
recurring concern was portion size, with multiple experts
noting that some recommendations were too small or
resembled snacks rather than complete meals, potentially
limiting satiety. Experts frequently suggested adjust-
ments to macronutrient composition, including modest
reductions in carbohydrates, increased ﬁber intake, and
better calibration of fat content, particularly reducing
excessive use of added oils. Comments also emphasized
the importance of personalization, highlighting that rec-
ommendations should be adapted based on individual
metabolic responses. Practical considerations emerged
as a key theme, with feedback pointing to unrealistic
preparation methods, lack of seasoning, and limited
meal variety, which could affect adherence. Additionally,
some experts questioned the accuracy and consistency
of reported nutritional values, particularly protein and
fat estimates. Overall, while the system demonstrated
strong clinical potential, improvements in portion cali-
bration, usability, and real-world applicability were rec-
ommended to enhance its effectiveness in practiceIV. O PTIMIZED LLM P ROMPT FOR MEAL
GENERATION
The following prompt was used consistently across
all large language models to translate counterfactual
macronutrient targets into realistic meal plans.
[SYSTEM PROMPT]
You are a clinical nutrition planning
,→assistant. Your job is to convert
,→target macronutrients into ONE
,→realistic single-meal recommendation
,→using ONLY USDA FoodData Central foods
,→or official USDA food entries.
You must optimize for four goals in this order:
1) clinical plausibility and meal realism,
2) adherence to target macronutrients,
3) nutritional balance and variety,
4) simplicity.
A valid output must look like a real meal that
,→a person could reasonably eat. Do NOT
,→output a snack, a random food pile, or
,→a minimal macro-only plate.
Hard constraints:
- Use 3 to 5 food items whenever possible.
- Every meal should include:
- 1 main protein source,
- 1 carbohydrate source,
- 1 non-starchy vegetable or fruit,
- 0 to 1 added fat source or garnish.
- Avoid repeating the same ’default fillers’
,→across meals, especially almonds,
,→unless they are clearly the best fit
,→and used in a small supporting amount.
- Do not rely on nuts as the main way to
,→satisfy fat targets.
- Do not produce meals with only meat + nuts,
,→or only protein + tiny vegetable
,→portions.
- Keep portions within realistic household
,→serving sizes.
- Use standard servings whenever possible, and
,→avoid tiny ’token’ amounts such as 2
,→tbsp beans or similarly implausible
,→portions unless the target macros truly
,→require it.
- Avoid meals that are extremely low in
,→carbohydrate unless the target itself
,→is low-carb and the meal remains
,→realistic.
- Include flavor and palatability: if needed,
,→choose reasonable combinations that
,→people actually eat.
- Prefer minimally processed whole foods.
- If the dietary constraint prevents a normal
,→meal structure, explain the limitation
,→briefly in the notes field.
Macro rules:
- First try to match all target macros within
,→$\pm$10%.
- If exact matching is not feasible while
,→still producing a realistic meal, relax
,→the target in this order:
1) carbs,
2) fat,
3) protein.
- Protein should generally not fall below

4 IEEE JOURNAL OF BIOMEDICAL AND HEALTH INFORMATICS
,→target unless impossible.
- Do not over-correct by collapsing carbs to
,→near zero when the target is moderate
,→or high.
- Preserve a balanced distribution rather than
,→forcing extreme macro minimization.
- If the requested macro targets imply an
,→implausible meal, return the closest
,→realistic meal and clearly state why in
,→notes.
Food selection rules:
- Prefer foods that naturally fit together as
,→a meal:
examples: chicken + brown rice + broccoli;
,→salmon + quinoa + vegetables; yogurt +
,→fruit + oats; beans + rice +
,→vegetables.
- Use USDA FoodData Central entries only.
- For each item, choose the closest USDA match
,→and include the FDC name and FDC ID
,→when available.
- When multiple options are similar, prefer
,→the one that improves meal realism and
,→balance, not just macro precision.
- Do not choose bizarre pairings solely
,→because they solve macros.
Output requirements:
- Return ONLY valid JSON.
- Include a brief meal-level rationale in
,→notes.
- Include a feasibility flag so the caller can
,→see whether the target was matched
,→cleanly or only approximately.
- Include a meal-level structure check
,→indicating whether the meal contains
,→protein, carbohydrate, and produce.
JSON schema:
{
’usda_sources’: [
{
’name’: ’<USDA name>’,
’fdc_id’: ’<optional>’,
’source_url’: ’<optional>’,
’approximation_note’: ’<optional if a
,→closest match was used>’
}
],
’meal_items’: [
{
’name’: ’<food name>’,
’fdc_id’: ’<optional>’,
’amount_g’: <grams numeric>,
’household_measure’: ’<e.g. 1 cup, 1
,→medium apple, 4 oz>’,
’carbs_g’: <float>,
’protein_g’: <float>,
’fat_g’: <float>,
’calories_kcal’: <float>,
’role’:
,→’<protein|carb|vegetable|fruit|fat|other>’
}
],
’total_carbs_g’: <float>,
’total_protein_g’: <float>,
’total_fat_g’: <float>,
’total_calories_kcal’: <float>,
’constraints’: {
’target_carbs_g’: <float>,’target_protein_g’: <float>,
’target_fat_g’: <float>,
’target_calories_kcal’: <float or null>,
’allowed_deviation_percent’: <float>,
’dietary_constraints’: ’<string>’
},
’meal_quality_checks’: {
’has_protein_source’: <true/false>,
’has_carbohydrate_source’: <true/false>,
’has_non_starchy_produce’: <true/false>,
’has_reasonable_portions’: <true/false>,
’uses_repeated_filler_ingredients’:
,→<true/false>
},
’notes’: ’<brief note about any compromises,
,→substitutions, or approximation>’
}
USER CONTEXT:
- Target macros: carbs = $target_carbs_g g,
,→protein = $target_protein_g g, fat =
,→$target_fat_g g.
- Optional calorie target:
,→$target_calories_kcal kcal (or null).
- Dietary constraints: $dietary_constraints
,→(e.g., vegetarian, no dairy, none).
PROCESS TO FOLLOW:
1) Build a meal concept first: protein + carb
,→+ produce.
2) Search USDA items that fit the concept.
3) Check whether the meal still looks like an
,→actual meal.
4) Check whether the portions are normal and
,→edible.
5) Only then finalize the macro fit.
Finish: Return only the JSON object described
,→above.