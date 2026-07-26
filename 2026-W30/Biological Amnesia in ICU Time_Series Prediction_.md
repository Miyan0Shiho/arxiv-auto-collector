# Biological Amnesia in ICU Time-Series Prediction: A Drift-Adaptive Two-Stream Architecture with Temporal Retrieval

**Authors**: Fatema Ferdous Tamanna, K. M. Merajul Arefin, Md. Abdul Masud

**Published**: 2026-07-21 12:07:36

**PDF URL**: [https://arxiv.org/pdf/2607.19020v1](https://arxiv.org/pdf/2607.19020v1)

## Abstract
Background: Clinical decision support systems degrade silently as treatment protocols evolve, yet standard adaptation methods treat models as monolithic blocks, unable to distinguish stable patient physiology from shifting institutional practice. Methods: We propose an adaptive clinical intelligence architecture for ICU intervention prediction that structurally decouples physiological from treatment representations, confining parameter updates to the treatment stream upon a dual distributional and accuracy trigger. Automated audit logs record which treatment features drove each adaptation event and how their importance shifted. At inference, an attribution-driven Temporal RAG module grounds each prediction in patient-specific, era-matched PubMed evidence anchored to the patient's dominant physiological features. Experiments used 84,792 MIMIC-IV stays (2008-2022) under strict chronological split. Results: Drift localised entirely to the treatment stream, validating the structural prior. Selective adaptation improved vasopressor and septic shock discrimination and calibration over the static source model. A fully retrained baseline yielded marginally higher aggregate discrimination but missed 26 septic shock cases the framework correctly identified, with none in the reverse direction; retrieval consistency with the pre-adaptation source model was preserved by the framework but degraded substantially in the retrained baseline. Conclusions: Structurally constraining adaptation to drifting components while preserving stable physiological representations enables clinical AI to evolve with practice without distorting learned patient biology. This architecture offers a template for governable, interpretable deployment of adaptive models in high-stakes clinical environments.

## Full Text


<!-- PDF content starts -->

Biological Amnesia in ICU Time-Series Prediction: A Drift-Adaptive
Two-Stream Architecture with Temporal Retrieval
Fatema Ferdous Tamanna *1, K. M. Merajul Arefin2, and Md. Abdul Masud1
1Dept. of Computer Science and Information Technology, Patuakhali Science and Technology University,
Bangladesh
2Dept. of Computer Science and Engineering, University of Dhaka, Bangladesh
Abstract
Background: Clinical decision support systems degrade
silently as treatment protocols evolve, yet standard adapta-
tion methods treat models as monolithic blocks, unable to
distinguish stable patient physiology from shifting institu-
tional practice.Methods: We propose an adaptive clinical
intelligence architecture for ICU intervention prediction that
structurally decouples physiological from treatment represen-
tations, confining parameter updates to the treatment stream
upon a dual distributional and accuracy trigger. Automated
audit logs record which treatment features drove each adapta-
tion event and how their importance shifted. At inference, an
attribution-driven Temporal RAG module grounds each pre-
diction in patient-specific, era-matched PubMed evidence an-
chored to the patient’s dominant physiological features. Ex-
periments used 84,792 MIMIC-IV stays (2008–2022) under
strict chronological split.Results: Drift localised entirely to
the treatment stream, validating the structural prior. Selec-
tive adaptation improved vasopressor and septic shock dis-
crimination and calibration over the static source model. A
fully retrained baseline yielded marginally higher aggregate
discrimination but missed 26 septic shock cases the frame-
work correctly identified, with none in the reverse direction;
retrieval consistency with the pre-adaptation source model
was preserved by the framework but degraded substantially in
the retrained baseline.Conclusions: Structurally constrain-
ing adaptation to drifting components while preserving sta-
ble physiological representations enables clinical AI to evolve
with practice without distorting learned patient biology. This
architecture offers a template for governable, interpretable
deployment of adaptive models in high-stakes clinical envi-
ronments.
Keywords:Clinical concept drift, continual learning, clin-
ical decision support systems, explainable AI, retrieval-
augmented generation
*Corresponding author:tamanna16@cse.pstu.ac.bd1 Introduction
Clinical decision support systems (CDSS) deployed in in-
tensive care units face a fundamental challenge: the clinical
world does not hold still, but deployed models do. Treatment
protocols evolve in response to growing evidence and institu-
tional guidelines. The Surviving Sepsis Campaign regularly
updates its vasopressor and fluid recommendations [1, 2], se-
dation practice shifted after adoption of the ABCDEF bun-
dle [3], and the COVID-19 pandemic compressed years of
protocol evolution into months [1, 4]. Such drift can silently
erode model calibration without triggering an obvious failure
signal.
This mismatch between training and deployment distribu-
tions is concept drift [5, 6]. Its clinical impact is well doc-
umented: performance drops in ICU models are driven by
shifts in coding and treatments rather than patient biology [7],
and single retrospective splits structurally underestimate real-
world degradation [8–10]. Current solutions fall short. Full
retraining risksbiological amnesia—a domain-specific mani-
festation of catastrophic forgetting [11] in which stable phys-
iological representations are unintentionally distorted as the
model over-indexes on shifting protocols—and creates gov-
ernance problems when adaptations are unmonitored [12].
Transfer learning [13, 14] and online learning [15] treat the
model as a monolithic block, ignoring a fundamental ICU
asymmetry: patient physiology is governed by stable human
biology, while treatment patterns reflect mutable institutional
practice. Furthermore, deployed CDSS often suffer from ex-
planatory staleness, citing outdated guidelines to justify cur-
rent predictions, and standard prediction tasks frequently suf-
fer from label leakage that obscures true clinical utility.
We address this with a drift-adaptive continual learning
framework whose contributions are:
1.Ongoing-need label formulation: replaces initiation-
only treatment labels with a formulation asking whether
treatment is needed during the prediction horizon, elim-
inating anti-correlation leakage and clinical relevance
loss in standard MIMIC-IV labelling.
2.Two-stream architecture with composite drift moni-
1
arXiv:2607.19020v1  [cs.LG]  21 Jul 2026

toring and selective adaptation: structurally decouples
physiological dynamics (LSTM) from treatment con-
text (MLP), using a composite detector combining PSI,
Kolmogorov–Smirnov, and AUROC signals to trigger
adaptation confined exclusively to the treatment stream,
leaving physiological representations bitwise identical
to the source model.
3.∆-Attribution metric and governance audit logs: a
model-class-agnostic measure of feature-importance re-
calibration that identifies which treatment features drove
each adaptation event and operationalises biological am-
nesia for cross-architecture comparison.
4.Attribution-driven Temporal RAG: derives patient-
specific PubMed queries from per-instance Integrated
Gradients attributions and conditions retrieval on the de-
tected drift era, structurally guaranteeing evidence con-
sistency via the frozen physiology stream.
We validate all four contributions on 84,792 MIMIC-IV
stays [18] under a strict chronological split (2008–2022), in-
cluding formal ablation studies and a clinician-rated retrieval
scaffold. To our knowledge, no prior study combines these
elements in a single clinically governable pipeline.
2 Methods
2.1 Dataset, Cohort, and Label Formulation
All experiments used MIMIC-IV (v3.1) [18], a freely ac-
cessible critical care database comprising de-identified elec-
tronic health records from Beth Israel Deaconess Medical
Center. We extracted adult ICU stays≥14 hours, excluding
stays with zero heart-rate variance (recording artefact), yield-
ing 84,792 stays. A strict chronological split was applied:
source training (2008–2013;n train=35,765,n val=8,784)
and a chronological test stream (2014–2022;n=40,243).
This design directly mirrors the conditions under which a de-
ployed CDSS would encounter distributional shift, and differs
from the single retrospective splits common in prior MIMIC-
based work [7, 10]. During post-drift adaptation a subject-
level split (30/10/60%) was used. Cohort characteristics are
summarised in Table 1.
Standard MIMIC-IV treatment-prediction labels define
positive only when treatmentinitiatesin the horizon. This
creates two simultaneous pathologies: (i)anti-correlation
leakage—patients already on vasopressors get a positive
feature but a negative label; (ii)clinical relevance loss—
continuation and weaning decisions are discarded. We re-
solve both with theongoing-needformulation: given obser-
vationX∈R6×86spanningh 0–h5and a two-hour gap[h 6,h7]
providing sufficient lead time for clinical action [19],
yt=1[∃h∈[h 8,h14]: treatment t(h) =1](1)
generating three binary labels per stay:y vaso,y intub,
yshock. Septic shock is operationalised via all three concur-
rent Sepsis-3 criteria [20] programmatically identified fromTable 1: Baseline Patient Characteristics Across Temporal
Cohorts
CharacteristicTrain/Val
(2008–13)Pre-Drift
(2014–19)Post-Drift
(2020–22)
Total Stays (N) 44,549 30,641 9,602
Age, mean±SD 65.4±16.7 64.1±16.9 63.8±16.5
Male, N (%) 24,291 (54.5) 17,210 (56.2) 5,591 (58.2)
White, N (%) 30,909 (69.4) 19,768 (64.5) 5,522 (57.5)
Black, N (%) 6,207 (13.9) 2,598 (8.5) 662 (6.9)
LOS days, mean±SD 2.8±4.7 3.2±5.3 4.0±7.0
In-hosp. mortality (%) 9.3 10.0 11.7
Vasopressor, N (%) 7,614 (17.1) 6,067 (19.8) 921 (9.6)
Intubation, N (%) 12,098 (27.2) 8,750 (28.6) 2,787 (29.0)
Septic Shock, N (%) 830 (1.9) 707 (2.3) 106 (1.1)
MIMIC-IV tables, strictly requiring refractory hypotension:
lactate>2mmol/L, active vasopressor, and invasive MAP
<65mmHg—the most severe subset of Sepsis-3, stricter than
the full definition (which includes patients achieving MAP
≥65mmHg with vasopressor support).
A programmatic audit confirmed absolute temporal
isolation across partitions: the intersection of unique
subject ids across every partition combination is exactly
zero. Post-drift adaptation used a subject-level split of the
9,602 post-drift stays (30% train / 10% validation / 60% eval-
uation), yielding 2,885 adaptation training stays, 968 valida-
tion stays, and 5,749 held-out evaluation stays (5,749 rather
than 5,761 owing to subject-level deduplication removing 12
stays across partition boundaries), plus a 500-stay pre-drift
replay buffer.
2.2 Feature Engineering
Features are split into two disjoint sets mirroring the archi-
tectural decomposition. Thephysiological matrixX∈R6×86
contains hourly vital signs (HR, BP, SpO 2, RR, tempera-
ture), laboratory values (lactate, creatinine, bilirubin, white
cell count, platelet count, BUN, glucose, electrolytes, and ar-
terial blood gas parameters), binary observation masks, and
derived rolling statistics (baseline-normalised deltas, ratios,
and short-horizon mean/SD over 3–6 hour windows). Miss-
ing values are handled by within-patient forward-filling fol-
lowed by median imputation computed exclusively from the
training cohort [18]. Thetreatment vectorZ∈R12captures
static treatment context (crystalloid volume, antibiotic tim-
ing, steroid orders, sex, age, medication count) with all di-
rect label proxies excluded to prevent anti-correlation leak-
age. All continuous treatment features are z-score normalised
using training-set statistics. Timing features occurring after
the observation window are clipped to a value beyond the ob-
servation window length to prevent temporal leakage.
2.3 Two-Stream Neural Architecture
The model has three components. Thephysiology stream
processesXthrough a two-layer LSTM [21] (hidden dim
64, LayerNorm [22]), producingh phys∈R64—frozen after
2

source training in Runs A and B. Layer normalisation is
preferred over batch normalisation given the variable-length
and irregular-sampling properties of clinical time series. The
treatment streamprocesseszthrough a two-layer MLP:
ttreat=LayerNorm(ReLU(W 2D(ReLU(W 1z+b 1))+b 2))
(2)
withW 1∈R64×12,W2∈R32×64, dropoutp=0.3. Thefusion
headconcatenates both representations and projects to three
outputs:
fjoint= [h phys;ttreat]∈R96(3)
through two dense layers (dims 64, 32; ReLU; dropout
0.3/0.225) followed by a linear projection:
ˆy=W 5ReLU(W 4D2(ReLU(W 3D1(fjoint)+b 3))+b 4)+b 5
(4)
whereD 1andD 2denote the dropout operators for the first
and second fusion layers. No activation is used on the final
layer, as BCEWithLogitsLoss applies sigmoid internally dur-
ing training; at inference, probabilities are obtained viaσ(ˆy).
Training minimises a focal binary cross-entropy loss [23]
with per-target positive-class reweighting and label smooth-
ing. ForNpatients andT=3 targets:
L=1
NT∑
i,t(1−p∗
i,t)γ[−α t˜yi,tlogp i,t−(1−˜y i,t)log(1−p i,t)]
(5)
wherep∗
i,tis the predicted probability of the true class (focal
modulating factor),γ=1.5 continuously during both source
training and adaptation, ˜y i,t=y i,t(1−ε) +ε/2 is the label-
smoothed target (ε=0.02 source for all phases), andα tis
the target-specific positive-class weight clipped at 20.0.
Four neural configurations span the design space:Run
A(static source, all frozen);Run B(selective adaptation—
proposed: physiology frozen, treatment MLP and fusion head
updated);Run C(full adaptation, all layers updated);Run D
(single-stream monolithic LSTM baseline: all features con-
catenated at every timestep; during adaptation only the fi-
nal output head is updated, mirroring last-layer fine-tuning
without structural decomposition). XGBoost [24] trained on
flattened physiological statistics concatenated with treatment
features (282-dim) serves as the tabular baseline, in both
static-source and fully-retrained configurations.
2.4 Dual-Signal Drift Detection and Selective
Adaptation
Drift is monitored via a dual OR-gate trigger over ten treat-
ment features (excluding age and medication count as sta-
ble demographic and administrative covariates). For continu-
ous features, the Population Stability Index (PSI) is computed
over 10 quantile bins with Laplace smoothing:
PSI=10
∑
i=1(Pi−Q i)ln(P i/Qi)(6)For binary features, a dedicated binary PSI is computed over
outcome categories{0,1}:
PSI binary=∑
k∈{0,1}
ˆpref
k−ˆpcur
k
lnˆpref
k
ˆpcur
k
(7)
with proportions clipped to[10−6,1−10−6]for numerical
stability. A binary feature is flagged by absolute rate shift
>0.05 or relative shift>20%. For each feature, three pair-
wise comparisons are computed: training versus pre-drift
(sanity check), training versus post-drift (primary drift sig-
nal), and pre-drift versus post-drift (temporal shift within
the test stream). Statistical significance is assessed via two-
sample Kolmogorov–Smirnov test for continuous features
and chi-squared test for binary features (both atα=0.01);
Cohen’sdeffect size is reported for continuous features.
Each feature is assigned a categorical drift severity label:
stable(PSI<0.10),minor(0.10≤PSI<0.20),moderate
(0.20≤PSI<0.35), orsevere(PSI≥0.35). The feature-drift
leg combines continuous-feature PSI and binary rate-shifts
into a composite distributional score, while factoring in KS-
test significance to satisfy a minimum drifted-feature count;
the accuracy leg monitors AUROC degradation relative to the
source-era baseline; either leg crossing its respective thresh-
olds activates the OR-gate. Drift is triggered if: (1) composite
distributional score, computed as PSI cont+0.5∆bin, exceeds
twice the training-era baseline (floor 0.20); or (2) AUROC
drops>0.02 on any target.
Upon trigger, adaptation enforces:
∇θphysL=0,θ(t+1)
m←θ(t)
m−η∇ θmL,m∈{treat,fusion}
(8)
using a 500-stay pre-drift replay buffer to maintain backward
compatibility [11]. Adaptation is run over 5 seeds; median-
validation-AUROC model is selected. Automated audit logs
record per-feature attribution shifts (via permutation-based
feature ablation over a 512-sample adaptation subset) at each
adaptation event [25].
2.5∆-Attribution and Biological Amnesia De-
tection
To quantify internal recalibration, we define the∆-Attribution
metric. Global feature importance is:
Φj(f,X) =E x∼X[|φj(f,x)|](9)
whereφ jis the local attribution score (SHAP for tree ensem-
bles; Integrated Gradients for neural models). Population-
level∆Φ jis computed via SHAP over the full post-drift
held-out set for XGBoost only; audit logs use permutation-
based ablation over the 512-sample adaptation subset. The
adaptation-induced shift is:
∆Φ j=Φ j(fadapt,Xpost)−Φ j(fsrc,Xpost)(10)
Evaluating both model states on identical post-drift dataX post
isolates weight recalibration from distributional change. All
∆Φ jestimates are overn=5,749 post-drift stays withB=
1,000 bootstrap resamples.
3

2.6 Attribution-Driven Temporal Retrieval
While standard Retrieval-Augmented Generation (RAG) [17]
systems synthesize natural language responses, our module
adapts this paradigm for predictive retrieval. It avoids static
corpora by coupling retrieval to per-instance model attribu-
tions and conditions the corpus window on the detected drift
era, producing a closed loop between adapted model be-
haviour and adapted evidence retrieval.
At inference, Integrated Gradients [16] are computed over
inputs using a zero baseline and 20 steps:
φj= (x j−x′
j)Z1
0∂F(x′+α(x−x′))
∂xjdα(11)
Two sub-queries are built independently: aphysiology sub-
queryfrom the top-5|φ j|features ofX(mask columns ex-
cluded) and atreatment sub-queryfrom the top-4 features of
Z. Because the physiology LSTM weights are bitwise iden-
tical between Run A and Run B by construction, physiology
sub-queries remainempirically stable—linking the architec-
tural freeze to retrieval consistency. Queries are prefixed with
label anchors (e.g., “septic shock ICU”), encoded via Med-
CPT [26]—whose asymmetric design (separate encoders for
short keyword queries and long article passages) suits the
structural mismatch between attribution-derived queries and
PubMed abstracts—and matched against an era-conditioned
PubMed corpus (871 ICU abstracts retrieved via NCBI E-
utilities; 522 retained after restriction to 2009–2019, cover-
ing the 10-year window ending the year before detected drift
onset). Retrieved sets are merged by PMID, re-ranked by
similarity, and the top-klist is returned with the prediction.
Retrieval stability is measured as Jaccard overlap of retrieved
PMID sets against the source model, assessed on 300 post-
drift patients (100 per label). Retrieval quality is evaluated
via canonical MeSH descriptors (MeSH P@5, nDCG@5) and
a 45-document clinician-rated scaffold (rated by the corre-
sponding author, blinded to model identity; single-rater as-
sessment is a limitation).
2.7 Evaluation Protocol
Primary metrics are AUROC and area under the precision-
recall curve (AUPRC) per target. Uncertainty is quantified
via bootstrap CIs (B=1,000): BCa for paired AUROC∆,
percentile for AUPRC∆[27].Standard 95% bootstrap confi-
dence intervals (α=0.05) are reported for the primary AU-
ROC family; AUPRC, Brier scores, and RAG metrics are ex-
ploratory. The source model is selected as the median of five-
seed ensemble (seeds 42, 123, 7, 2024, 99) on validation AU-
ROC. The two-stream model was trained for up to 50 epochs
with early stopping (patience=8) using the Adam optimiser
(learning rate 1×10−3, weight decay 1×10−5). Adaptation
uses Adam (η=3×10−4, up to 40 epochs, patience 8).
An XGBoost classifier [24] was trained on a flattened fea-
ture representation comprising last-value, mean, standard de-
viation, minimum, and maximum of each physiological fea-
ture across the observation window, concatenated with the 12
Figure 1: Distribution shifts in key treatment features: (A)
total crystalloid volume and (B) insulin infusion rate. Physi-
ological features (max PSI=0.100) show no comparable shift.
treatment features; mask columns are excluded, leaving ap-
proximately 54 non-mask sequential features, yielding 5×
54+12=282 dimensions. Two configurations are evaluated:
XGBoost-source (no adaptation) and XGBoost-adapted (re-
trained on the same post-drift partition used for Run B), serv-
ing as our proxy for the industry-standard sliding-window
periodic retraining strategy. Post-hoc SHAP (TreeExplainer)
analysis [28] was performed on both XGBoost configurations
to examine feature attribution stability under drift. No post-
hoc probability calibration (Platt scaling or isotonic regres-
sion) was applied to any model; all reported probabilities are
native model outputs.
3 Results
3.1 Drift Detection and Localisation
The dual-gated detector maintained stability through 2014–
2019 despite incremental protocol shifts (Sepsis-3 adoption
2016, SSC guideline updates 2016–2018): neither gate was
crossed, confirming stable model performance throughout
the pre-drift period. The 2020–2022 cohort triggered a se-
vere alert via the distributional leg (composite distributional
score 0.2324 vs. training baseline 0.0095); the accuracy leg
did not independently fire, though vasopressor’s AUROC
drop (0.0197) approached the 0.02 threshold. Five treatment
features exceeded drift thresholds:total crystalloid ml
(PSI=0.76; mean volume fell from 791 to 255 mL), insulin
infusion prevalence (−8%), blood product use (−8%), PRBC
volume, and antibiotic timing—all consistent with docu-
mented COVID-era conservative resuscitation [1].
PSI analysis across all 54 physiological features confirmed
distributional stability (mean PSI=0.016, max PSI=0.100;
53/54 in the stable range (PSI<0.10)). This empirically val-
idates the architectural prior: drift was localised predomi-
nantly to the treatment domain, with one physiological fea-
ture at the minor-drift boundary (PSI = 0.10, 53/54 features
stable) (Fig. 1).
3.2 Pre-Drift Sanity Check and Baseline
Equivalence
On the pre-drift cohort (2014–2019), Runs A, B, and C
are mathematically identical by construction (∆AUROC=
0.0000 across all labels), since adaptation is not yet triggered.
4

Run D uses independently trained source weights to establish
its distinct pre-drift baseline. XGBoost-Adapted is identical
to XGBoost-Source, as retraining has not occurred. The sta-
ble performance across all architectures prior to adaptation
rules out initialisation bias: any post-drift divergence is at-
tributable solely to the differing adaptation strategies and ar-
chitectural priors rather than baseline discrepancies.
3.3 Post-Drift Discrimination and Calibration
Post-drift evaluation set:n=5,749 stays (12 removed by
subject-level deduplication from the nominal 60% of 9,602).
Of the 106 septic shock cases in the full post-drift cohort
(n=9,602; prevalence 1.1%), 76 fell within this held-out
evaluation partition (prevalence 1.3%), with the remainder al-
located to adaptation training and validation.
Results are in Tables 2 and 4 and Fig. 2. Run B (selec-
tive adaptation) achieved mAUROC=0.9316, outperforming
frozen Run A (0.8965) with the largest gain for vasopres-
sor (∆= +0.0713, BCa 95% CI[+0.0622,+0.0822],p<
0.001) and a significant gain for septic shock (∆= +0.0303,
BCa 95% CI[+0.0113,+0.0497],p<0.001). Run B sur-
passed both unconstrained full adaptation (Run C, 0.9249)
and single-stream last-layer fine-tuning (Run D, 0.9010), con-
firming that structural decomposition—not merely selective
freezing—is the operative mechanism. XGBoost-adapted
achieved marginally higher mean AUROC (0.9382), but this
aggregate advantage has hidden bedside costs detailed in Sec-
tion 3.4. Per-label bootstrap 95% confidence intervals for all
four neural configurations are reported in Table 3. Mean AU-
ROC is a necessary but not sufficient criterion for model se-
lection in high-stakes clinical settings [32].
AUPRC results are decisive for the rare septic shock con-
dition (1.3% prevalence): Run B improved from 0.3100
(Run A) to 0.4131 (+0.1031, 95% CI[+0.0484,+0.1553]).
Bootstrap 95% CIs on all paired AUPRC gains confirm sta-
tistical reliability: vasopressor[+0.1721,+0.2301], intuba-
tion[+0.0007,+0.0159], septic shock[+0.0484,+0.1553].
The vasopressor gain is substantial and precisely estimated;
the septic shock CI reflects the small positive class (n=76);
the modest intubation gain is consistent with the absence
of ventilation-adjacent treatment features. By comparison,
Run D deteriorated septic shock AUPRC to 0.2149, fur-
ther demonstrating that entangled adaptation damages rare-
class performance. XGBoost-adapted’s septic shock AUPRC
degradedfrom 0.3313 to 0.2600 after retraining—evidence
of probability mass compression toward the majority class,
indicating monolithic retraining actively traded precision-
recall performance on the rarest, most dangerous class in
exchange for aggregate discrimination. Run B restored
calibration across all targets (vasopressor Brier 0.1861→
0.1241; intubation 0.1223→0.1016; septic shock 0.0613→
0.0184). XGBoost-adapted’s nominal Brier score (septic
shock 0.0116) reflects probability compression rather than
genuine calibration—a model assigning near-zero probabili-
ties to all patients minimises squared error under severe class
imbalance without detecting any positives at a meaningful
Figure 2: Post-drift AUROC trajectories across three labels
(vasopressor, intubation, septic shock). Runs A/B/C are iden-
tical on pre-drift data; adaptation diverges only post-2020.
Run B (selective) outperforms both frozen Run A and single-
stream Run D.
Table 2: Post-Drift AUROC and Mean AUROC (n=5,749).
Model Vaso Intub Shock Mean
Monolithic Baselines
XGBoost (Source) 0.91480.94750.9367 0.9330
XGBoost (Adapted)0.93300.9413 0.94040.9382
Neural Baselines
Run A: Static Source 0.8233 0.9367 0.9296 0.8965
Run C: Full Adapt 0.8779 0.9435 0.9533 0.9249
Run D: Single-Stream 0.8269 0.9414 0.9347 0.9010
Proposed Framework
Run B: Selective Adapt0.8947 0.94030.95990.9316
Table 3: Post-Drift AUROC with 95% Bootstrap Confidence
Intervals (percentile method,n=5,749). XGBoost point es-
timates are in Table 2; bootstrap CIs were not computed for
it.
Label Run A Run B Run C Run D
Vasopressor 0.823 [0.805, 0.841] 0.895 [0.879, 0.910] 0.878 [0.862, 0.893] 0.827 [0.810, 0.843]
Intubation 0.937 [0.930, 0.943] 0.940 [0.934, 0.947] 0.943 [0.937, 0.950] 0.941 [0.935, 0.948]
Septic Shock 0.930 [0.904, 0.953] 0.960 [0.934, 0.978] 0.953 [0.925, 0.975] 0.935 [0.907, 0.957]
Table 4: Post-Drift AUPRC and Septic Shock Brier (n=
5,749).
Model Vaso Intub Shock Brier (↓)
Monolithic Baselines
XGBoost (Source) 0.64870.87570.3313 0.0123
XGBoost (Adapted)0.66120.8586 0.26000.0116
Neural Baselines
Run A: Static Source 0.4187 0.8475 0.3100 0.0613
Run C: Full Adapt 0.5714 0.8632 0.3775 0.0167
Run D: Single-Stream 0.3544 0.8598 0.2149 0.0324
Proposed Framework
Run B: Selective Adapt0.6182 0.85580.41310.0184
threshold, confirmed threshold-free by the AUPRC collapse
from 0.3313 to 0.2600.
3.4 Clinical Safety and Bedside Disagreement
The aggregate AUROC advantage of XGBoost-adapted is ac-
companied by a severe unidirectional failure pattern. Defin-
5

Table 5: Bedside Disagreement (catches/misses vs.
XGBoost-Adapted,n=5,749). Catch:p≥0.50; Miss:
p<0.10.
Model Vaso (n=
554)Intub (n=
1657)Shock (n=
76)
Run B: Selective Adapt19/ 04/ 026/ 0
Run C: Full Adapt 17 / 0 7 / 0 20 / 0
Run D: Single-Stream 14 / 0 6 / 0 25 / 0
Table 6: XGBoost-Adapted vs. Run B atτ=0.5 (TP/FP/FN
and Sensitivity/Precision).
Target Model TP FP FN Recall Prec.
Septic Shock
(n=76)XGBoost (adapted) 5 9 71 6.6% 35.7%
Run B (proposed) 5312423 69.7%29.9%
Vasopressor
(n=554)XGBoost (adapted) 355 239 199 64.1% 59.8%
Run B (proposed) 441836113 79.6%34.5%
Intubation
(n=1,657)XGBoost (adapted) 1428 368 229 86.2% 79.5%
Run B (proposed) 1464553193 88.4%72.6%
ing a “Catch” asp≥0.50 and a “Miss” asp<0.10 on
a ground-truth positive, Table 5 shows that Run B caught
26 true-positive septic shock cases that XGBoost-adapted
critically missed, withzerocases in the reverse direction.
This 26/0 asymmetry persists across thresholds (31/0 at catch
≥0.40; 32/0 at catch≥0.30; 19/0 at miss<0.05) and is
confirmed threshold-free by the AUPRC collapse. Run C
(20/0) and Run D (25/0) share this directional advantage, con-
firming monolithic retraining suppresses septic shock sensi-
tivity across all comparators; Table 6 reports the full confu-
sion matrix atτ=0.5 for both models on all three targets.
Notably, XGBoost’s own septic-shock performance degrades
on both axes under adaptation—not merely a precision-
recall tradeoff—dropping from 34.2% recall / 40.0% preci-
sion (source) to 6.6% recall / 35.7% precision (adapted), in-
dicating that retraining does not sharpen the model but rather
suppresses it into near-silence on the rarest, most lethal target.
Relative to XGBoost-adapted, Run B raises septic-shock re-
call from 6.6% to 69.7% (5 vs. 53 of 76 true cases caught) at
a cost of 115 additional false positives out of 5,673 total neg-
atives (an absolute false-alarm-rate increase of 2.0 percentage
points), or approximately 2.4 extra alarms per additional true
case caught. This tradeoff is markedly less favourable for va-
sopressor (6.9 extra alarms per additional catch) and intuba-
tion (5.1), indicating that recall-prioritisation is best justified
specifically for the rarest and most lethal target rather than as
a blanket property of the architecture.
This failure is illustrated by specific patient trajectories.
For vasopressor prediction (Patient A, stay id 35773744),
Run B predicted 90.4% vs. XGBoost’s 15.5%. For septic
shock (Patient B, stay id 31656477), Run B assigned 85.6%
probability; XGBoost assigned 4.0%—a critically low prob-
ability for a confirmed lethal condition. These cases confirm
that monolithic retraining suppresses true-positive alerts de-
spite acceptable aggregate metrics.
Figure 3: Biological amnesia: illustrative single-patient
SHAP comparison pre/post-adaptation. XGBoost-adapted
shows∆=1.454 SHAP units ontotal crystalloid ml
with widespread physiological recalibration; the Two-Stream
framework shows zero physiological shift.
3.5 Biological Amnesia via∆-Attribution
The Two-Stream architecture provides a mathematically ex-
act guarantee: LSTM physiology parameters are bitwise
identical between Run A and Run B (physio mean rel∆=
0.0000). Any attribution shift in Run B originates exclusively
from the updated fusion head re-weighting a structurally pre-
served biological state—a property unreplicable by post-hoc
methods on monolithic models [29].
The following∆Φanalysis is self-contained within XG-
Boost (SHAP values); attribution magnitudes are not com-
pared across architectures as SHAP and Integrated Gradients
operate on incomparable scales. In XGBoost-adapted,∆Φ
analysis reveals treatment shifts that are clinically coherent
(mean|SHAP|fortotal crystalloid ml: 1.057→2.185,
∆Φ= +1.128 for vasopressor), accompanied by simulta-
neous recalibration ofstablephysiological features (PSI<
0.10):max lactatelost∆Φ=−0.438[−0.442,−0.434]∗;
stdphvenousgained+0.847[+0.835,+0.859]∗—a near-
inversion of the canonical haemodynamic severity signal
(∗95% CI excludes zero). This is the biological amnesia in-
troduced above: the measurable, population-level overwriting
of stable physiological representations in a monolithic model
(Fig. 3). Population-level∆-Attribution analysis identified
statistically significant importance shifts across 93.6% of fea-
tures (264 of 282) for vasopressor (95.7% for septic shock,
92.9% for intubation); for vasopressor, gains outnumbered
losses among physiological features (160 vs. 93), while for
septic shock the pattern reversed, with losses dominating (226
vs. 32), confirming widespread physiological recalibration in
both directions across labels.
3.6 Attribution-Driven Temporal Retrieval
The frozen physiology LSTM structurally anchors evidence
retrieval. Table 7 reports Jaccard overlap of retrieved PMID
6

Table 7: Evidence Retrieval Stability (n=300 post-drift pa-
tients). Jaccard overlap of retrieved PubMed documents vs.
source model; higher is better.
Metric Run B XGBoost
(Adp.)
Physiology Jaccard (↑)0.5730.330
Treatment Jaccard (↑)0.5400.448
Merged Jaccard (↑)0.4380.323
Rank corr., physio Spearman (↑)0.4650.131
Rank corr., treatment Spearman (↑)0.5920.439
Token Jaccard, physio (↑)0.7290.341
Token Jaccard, treatment (↑)0.7170.602
sets against the source model across 300 post-drift patients
(100 per label). Run B maintains physiology-stream Jac-
card 0.573 vs. 0.330 for XGBoost-adapted (∆= +0.243);
the treatment stream adapts as expected (0.540 vs. 0.448).
Per-label physiology Jaccard: vasopressor 0.485/0.260, in-
tubation 0.619/0.293, septic shock 0.614/0.438. Query-level
token Jaccard confirms the structural cause: Run B physi-
ology queries share 72.9% of tokens with the source model
vs. 34.1% for XGBoost—the direct consequence of identical
LSTM weights between Run A and Run B.
The mechanistic link between architectural freeze and re-
trieval stability is confirmed via Spearman correlation be-
tween per-patient∆ physio and Jaccard divergence: for Run B,
r=−0.197 (p=0.0006, computed onn=241 cases);
for XGBoost-adapted,r=−0.169 (p=0.0033, computed
on onlyn=137 cases, as the remaining 163 cases were
completely excluded due to zero document overlap with
the source queries). The negative sign for Run B reflects
that its residual attribution delta originates from the fusion
head—not physiological features—so retrieval remains an-
chored to source documents regardless of fusion-head re-
calibration magnitude. Automatic retrieval quality (Ta-
ble 8) shows equivalent MeSH P@5 across all configurations
(0.635/0.635/0.632), confirming selective adaptation does not
degrade topical relevance; per-label, vasopressor achieves the
highest precision (P@5 0.710–0.760) and intubation the low-
est (P@5 0.208–0.252), reflecting the broader MeSH vocabu-
lary of mechanical ventilation literature. Clinician-rated P@5
favours Run B (0.800 vs. XGBoost 0.467)—a gap substan-
tially wider than the automatic MeSH P@5 gap (0.635 vs.
0.632)—while XGBoost’s higher automatic nDCG@5 (0.913
vs. 0.838, Table 8) reflects superior ranking of the relevant
documents it retrieves, rather than broader topical coverage.
Worked examples.The clinical value is best understood
at the individual patient level. We present three representa-
tive cases drawn from the RAG evaluation cohort– including
a mechanistic deep-dive into the vasopressor disagreement
case introduced in Section 3.4—to illustrate the clinical value
at the individual patient level. These cases capture both res-
cue scenarios—where the baseline model critically misses the
diagnosis—and spurious reasoning, where the baseline model
predicts correctly but for physiologically incoherent reasons.
Patient A, vasopressor (stay 35773744).True label=Table 8: Automatic Retrieval Quality. MeSH P@5 and
nDCG@5 over 300 post-drift cases using canonical MeSH
descriptors as oracle.
Label / Overall Source Run B XGBoost
(Adp.)
P@5 Overall0.635 0.6350.632
P@5 Vasopressor 0.7500.7600.710
P@5 Intubation0.2520.248 0.208
P@5 Septic Shock 0.902 0.8960.978
nDCG@5 Overall 0.860 0.8380.913
1: Run B assigned probability 0.904; XGBoost-adapted
assigned 0.155—a confident false negative. Integrated
Gradients identifiedlactate,lactate baseline, and
diastolic blood pressureas the dominant physiology-
stream contributors. The physiology sub-query retrieved
haemodynamic management abstracts with identical PMID
sets for source and Run B (Jaccard=1.000); the XGBoost
query—shifted toward fluid and medication volume terms by
retraining drift—retrieved a substantially different and less
clinically relevant document set.
Patient B, intubation (stay 31841598).Both Run B (0.926)
and XGBoost (0.960) correctly predicted mechanical ventila-
tion. Integrated Gradients identifiedgcs verbal,gcs eye,
andrespiratory rateas the dominant physiology con-
tributors. Source and Run B physiology queries retrieved
identical documents (Jaccard=1.000); crucially, XGBoost’s
internal query—incorporatingWBC,age, and antibiotic tim-
ing from the flattened feature representation—retrieved a sub-
stantially different set skewed toward antibiotic management
literature. This illustrates spurious reasoning: a correct pre-
diction grounded in the wrong physiological evidence, expos-
ing attribution redistribution despite an accurate outcome.
Patient C, septic shock (stay 30855786).Run B assigned
probability 0.805; XGBoost-adapted assigned 0.039—
missing a highly lethal condition. Integrated Gradients iden-
tifieddiastolic blood pressure,lactate baseline,
andurine outputas the dominant physiology contribu-
tors. The Run B physiology sub-query retrieved the same
documents as the source model (Jaccard=1.000), includ-
ing abstracts on cardiovascular determinants of sepsis re-
suscitation; the XGBoost query—shifted toward respira-
tory and metabolic terms by attribution redistribution under
retraining—retrieved a divergent set. The frozen physiology
stream thus provides two complementary benefits: preserving
both the predictive signal and the evidence grounding for the
physiological features defining each patient’s clinical presen-
tation.
3.7 Sensitivity Analysis and Ablation Studies
To validate robustness, we re-ran Run B selective adaptation
varying the post-drift training split ratio (20/30/40%, fixed
500-stay replay buffer) and the replay buffer size (0/250/500
stays, fixed 30% split). The 30% ratio yielded the strongest
7

performance among the three tested values. Across re-
play buffer sizes, post-drift discrimination was essentially
unchanged (mean AUROC within 0.001–0.002 of one an-
other), confirming that the frozen physiology stream—not
the replay buffer—is the primary anti-forgetting mechanism.
The 500-stay buffer achieved strongest pre-drift retention
among the three tested, serving a complementary backward-
compatibility role.
4 Discussion
Results suggest that CDSS temporal degradation is driven
predominantly by treatment-protocol drift rather than physi-
ological change—a distinction current monolithic adaptation
methods cannot enforce structurally. The two-stream archi-
tecture exploits this asymmetry by design: freezing the phys-
iology LSTM is not a heuristic choice but an empirically justi-
fied structural prior (treatment max PSI 0.76 vs. physiological
max PSI 0.10). The result is a mathematically exact guarantee
that no post-hoc explanation method can provide: physiolog-
ical representations are bitwise identical between source and
adapted model, so any attribution shift originates solely from
the updated fusion head.
The 26/0 unidirectional septic shock disagreement is the
most clinically striking result. XGBoost-adapted achieved
marginally higher mean AUROC, but this aggregate advan-
tage actively suppressed true-positive alerts on the rarest
and most lethal condition—trading precision-recall perfor-
mance for aggregate rank-ordering. The∆-Attribution analy-
sis explains why: monolithic retraining simultaneously recal-
ibrated up to 95.7% of features (label-dependent), including
stable physiological features whose PSI confirmed no distri-
butional change, producing shifts that are architecturally im-
possible to distinguish from legitimate co-adaptation. This
is biological amnesia as a measurable, population-level phe-
nomenon. Prior work (Nestor et al. [7], Futoma et al. [10])
characterised the problem; this paper provides both an archi-
tectural remedy and a formal metric to detect and audit it.
The attribution-driven Temporal RAG closes a comple-
mentary gap: explanatory staleness. Unlike Almanac [30]
and similar static-corpus systems, our module couples re-
trieval to per-instance attributions and to the publication era
of the source training window, ensuring a 2022 prediction is
grounded in contemporaneous evidence (2009–2019) rather
than potentially divergent post-pandemic guidelines.
Clinical implications.Drift detection confirmed that
treatment-side shift can be clinically substantial even when
patient physiology remains stable—the five drifted features
(conservative crystalloid resuscitation PSI=0.76, reduced in-
sulin infusion−8%, reduced blood product use−8%, PRBC
volume, accelerated antibiotic timing) are all consistent with
documented COVID-era protocol changes [1]. This supports
recent calls for ongoing post-deployment performance mon-
itoring rather than one-time validation [12, 31]. Calibration
improvements are equally significant: selective adaptation
recalibrates probability estimates, not merely ranks (vaso-pressor Brier 0.1861→0.1241; intubation 0.1223→0.1016;
septic shock 0.0613→0.0184). Since clinical interven-
tion thresholds are probability-based, CDSS deployed against
miscalibrated outputs can produce systematically biased rec-
ommendations even at high AUROC [32]. The attribution
audit logs address this governance requirement directly: by
documenting per-feature importance shifts at each adaptation
event, the framework makes model updates interpretable and
contestable by clinicians—a property monolithic retraining
cannot provide by design.
Relation to prior work.Prior work on temporal gener-
alisation in clinical ML has predominantly characterised the
problem rather than offered architectural remedies. Nestor
et al. [7] demonstrated feature-importance instability across
MIMIC-III eras but did not propose adaptation; Futoma et al.
[10] argued that single-split evaluation systematically over-
states clinical utility. Compared to continual-learning regu-
larisation (EWC [11]), which imposes a soft hyperparameter-
dependent penalty, our structural freeze is unconditional:
a physiologically stable feature with small source-domain
weights is not protected by EWC but is fully protected by
our architecture. Domain-adversarial approaches [33] require
environmental labels during source training and do not ex-
plicitly isolate treatment shifts; head-to-head comparison is
a planned extension. On the explanation side, recent clinical
RAG systems such as Almanac [30] rely on static corpora;
our framework couples retrieval to per-instance attributions
and to publication-era filtering, producing patient-specific
and time-appropriate evidence in a single closed loop. The∆-
Attribution metric extends concept-drift detection via model
explanation [25] to a formally defined, population-level mea-
sure of physiological recalibration under adaptation—a dis-
tinction Dem ˇsar and Bosni ´c did not address. Together, these
elements constitute a pipeline in which architectural design,
adaptation governance, and evidence retrieval are structurally
coupled rather than independently applied—a combination
not present in prior clinical ML literature to our knowledge.
Limitations.Evaluation is limited to a single institution; ex-
ternal validation on eICU-CRD is the natural next step. The
treatment vector lacks non-leaking ventilation-adjacent fea-
tures, limiting intubation-specific adaptation benefit; identi-
fying such proxies is a natural extension. Septic shock re-
sults (76 positives) warrant replication on larger post-2020
cohorts. The RAG module currently surfaces abstract-level
evidence; production use would benefit from full-text re-
trieval with LLM summarisation. The RAG evaluation re-
lies on a single-rater clinician scaffold; multi-rater validation
would strengthen the reliability of the P@5 and nDCG@5 es-
timates. Ethnicity indicators in the treatment vector carry eq-
uity risks requiring fairness evaluation. Head-to-head bench-
marking against EWC and DANN is left to future work. All
analysis is retrospective; prospective impact measurement is
needed [8].
8

5 Conclusion
We have presented a governable clinical intelligence architec-
ture that moves the CDSS paradigm from static, monolithic
models toward dynamic, attribution-aware systems. By struc-
turally decoupling stable human physiology from evolving in-
stitutional treatment protocols, our framework overcomes the
biological amnesia inherent in standard monolithic retrain-
ing. We demonstrated that clinical AI can evolve safely—not
as an opaque update, but as a transparent, governed process
grounded in causal audit logs and per-instance evidence re-
trieval. Our results validate this approach on the MIMIC-IV
cohort, where selective adaptation achieved superior bedside
safety—most notably catching 26 critical septic shock cases
missed by standard retraining—while maintaining retrieval
consistency as the model evolved. Ultimately, this framework
enables CDSS to keep pace with clinical practice while pre-
serving the integrity of fundamental patient biology.
Ethics and Data
MIMIC-IV (v3.1) is de-identified and publicly avail-
able via PhysioNet (https://physionet.org/content/
mimiciv/3.1/); IRB approval was not required. Code and
the curated PubMed corpus will be released at the corre-
sponding author’s GitHub upon acceptance. The authors de-
clare no competing interests and received no specific funding.
Acknowledgment
During the preparation of this work, the authors used Claude
(Anthropic), Grok (xAI), and Gemini (Google) for language
editing, L ATEX formatting, and structural organisation of the
manuscript text. No AI system was used to generate, analyse,
or interpret the research data or results. The authors reviewed
and edited all AI-assisted content and take full responsibility
for the accuracy and integrity of the published work.
References
[1] L. Evans et al., “Surviving sepsis campaign: in-
ternational guidelines for management of sep-
sis and septic shock 2021,”Crit. Care Med.,
vol. 49, no. 11, pp. e1063–e1143, 2021, doi:
10.1097/CCM.0000000000005337.
[2] A. Rhodes et al., “Surviving sepsis campaign: interna-
tional guidelines for management of sepsis and septic
shock: 2016,”Crit. Care Med., vol. 45, no. 3, pp. 486–
552, 2017, doi: 10.1097/CCM.0000000000002255.
[3] J. W. Devlin et al., “Clinical practice guidelines for pain,
agitation/sedation, delirium, immobility, and sleep in
ICU patients,”Crit. Care Med., vol. 46, no. 9, pp. e825–
e873, 2018, doi: 10.1097/CCM.0000000000003299.[4] G. Grasselli et al., “Baseline characteristics and out-
comes of 1591 patients infected with SARS-CoV-
2 admitted to ICUs of the Lombardy region, Italy,”
JAMA, vol. 323, no. 16, pp. 1574–1581, 2020, doi:
10.1001/jama.2020.5394.
[5] J. Gama et al., “A survey on concept drift adaptation,”
ACM Comput. Surveys, vol. 46, no. 4, Art. no. 44, 2014,
doi: 10.1145/2523813.
[6] J. Lu et al., “Learning under concept drift:
a review,”IEEE Trans. Knowl. Data Eng.,
vol. 31, no. 12, pp. 2346–2363, 2019, doi:
10.1109/TKDE.2018.2876857.
[7] B. Nestor et al., “Feature robustness in non-stationary
health records: caveats to deployable model perfor-
mance in common clinical machine learning tasks,” in
Proc. MLHC, vol. 106, 2019, pp. 381–405.
[8] C. J. Kelly et al., “Key challenges for delivering clinical
impact with artificial intelligence,”BMC Med., vol. 17,
no. 1, Art. no. 195, 2019, doi: 10.1186/s12916-019-
1382-x.
[9] A. Wong et al., “External validation of a widely
implemented proprietary sepsis prediction model in
hospitalized patients,”JAMA Intern. Med., vol. 181,
no. 8, pp. 1065–1070, 2021, doi: 10.1001/jamaintern-
med.2021.2626.
[10] J. Futoma et al., “The myth of generalisability in clinical
research and machine learning in health care,”Lancet
Digit. Health, vol. 2, no. 9, pp. e489–e492, 2020, doi:
10.1016/S2589-7500(20)30186-2.
[11] J. Kirkpatrick et al., “Overcoming catastrophic for-
getting in neural networks,”PNAS, vol. 114, no. 13,
pp. 3521–3526, 2017, doi: 10.1073/pnas.1611835114.
[12] J. Feng et al., “Clinical artificial intelligence quality im-
provement: towards continual monitoring and updat-
ing of AI algorithms in healthcare,”npj Digit. Med.,
vol. 5, no. 1, Art. no. 66, 2022, doi: 10.1038/s41746-
022-00611-y.
[13] M. B. McDermott et al., “Reproducibility in machine
learning for health research: still a ways to go,”Sci.
Transl. Med., vol. 13, no. 586, Art. no. eabb1655, 2021,
doi: 10.1126/scitranslmed.abb1655.
[14] H. Zhang et al., “Shifting machine learning for health-
care from development to deployment and from models
to data,”npj Digit. Med., vol. 5, no. 1, Art. no. 40, 2022,
doi: 10.1038/s41746-022-00698-3.
[15] V . L ¨osing, B. Hammer, and H. Wersing, “Incremental
on-line learning: a review and comparison of state of the
art algorithms,”Neurocomputing, vol. 275, pp. 1261–
1274, 2018, doi: 10.1016/j.neucom.2017.06.084.
9

[16] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic at-
tribution for deep networks,” inICML, 2017, pp. 3319–
3328.
[17] P. Lewis et al., “Retrieval-augmented generation for
knowledge-intensive NLP tasks,” inNeurIPS, vol. 33,
2020, pp. 9459–9474.
[18] A. E. Johnson et al., “MIMIC-IV , a freely accessible
electronic health record dataset,”Sci. Data, vol. 10,
no. 1, Art. no. 1, 2023, doi: 10.1038/s41597-022-
01899-x.
[19] M. Moor et al., “Early prediction of sepsis in the
ICU without target leakage: a new benchmark,”
Front. Med., vol. 8, Art. no. 607952, 2021, doi:
10.3389/fmed.2021.607952.
[20] M. Singer et al., “The third international consen-
sus definitions for sepsis and septic shock (Sepsis-
3),”JAMA, vol. 315, no. 8, pp. 801–810, 2016, doi:
10.1001/jama.2016.0287.
[21] S. Hochreiter and J. Schmidhuber, “Long short-term
memory,”Neural Comput., vol. 9, no. 8, pp. 1735–1780,
1997, doi: 10.1162/neco.1997.9.8.1735.
[22] J. L. Ba, J. R. Kiros, and G. E. Hinton, “Layer normal-
ization,” arXiv:1607.06450 [stat.ML], 2016.
[23] T.-Y . Lin et al., “Focal loss for dense object
detection,” inICCV, 2017, pp. 2980–2988, doi:
10.1109/ICCV .2017.324.
[24] T. Chen and C. Guestrin, “XGBoost: a scalable tree
boosting system,” inKDD, 2016, pp. 785–794, doi:
10.1145/2939672.2939785.
[25] J. Dem ˇsar and Z. Bosni ´c, “Detecting concept drift
in data streams using model explanation,”Expert
Syst. Appl., vol. 92, pp. 546–559, 2018, doi:
10.1016/j.eswa.2017.10.003.
[26] Q. Jin et al., “MedCPT: contrastive pre-trained trans-
formers with large-scale PubMed search logs for zero-
shot biomedical information retrieval,”Bioinformat-
ics, vol. 39, no. 11, Art. no. btad651, 2023, doi:
10.1093/bioinformatics/btad651.
[27] B. Efron, “Better bootstrap confidence intervals,”J.
Amer. Stat. Assoc., vol. 82, no. 397, pp. 171–185, 1987,
doi: 10.1080/01621459.1987.10478410.
[28] S. M. Lundberg and S.-I. Lee, “A unified approach to
interpreting model predictions,” inNeurIPS, vol. 30,
2017, pp. 4765–4774.
[29] C. Rudin, “Stop explaining black box machine learning
models for high stakes decisions and use interpretable
models instead,”Nature Mach. Intell., vol. 1, no. 5,
pp. 206–215, 2019, doi: 10.1038/s42256-019-0048-x.[30] C. Zakka et al., “Almanac—retrieval-augmented lan-
guage models for clinical medicine,”NEJM AI,
vol. 1, no. 2, Art. no. AIoa2300068, 2024, doi:
10.1056/AIoa2300068.
[31] S. G. Finlayson et al., “The clinician and dataset shift in
artificial intelligence,”N. Engl. J. Med., vol. 385, no. 3,
pp. 283–286, 2021, doi: 10.1056/NEJMc2104626.
[32] B. Van Calster et al., “Calibration: the Achilles heel of
predictive analytics,”BMC Med., vol. 17, no. 1, Art.
no. 230, 2019, doi: 10.1186/s12916-019-1466-7.
[33] Y . Ganin et al., “Domain-adversarial training of neu-
ral networks,”J. Mach. Learn. Res., vol. 17, no. 59,
pp. 2096–2130, 2016.
10