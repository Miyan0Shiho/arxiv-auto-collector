# Operation-aware Multimodal Clinical Decision Support System for ICU Discharge Risk Assessment

**Authors**: Guoqing Zhang, Rafik Hadfi, Takayuki Ito

**Published**: 2026-11-09 06:28:58

**PDF URL**: [https://arxiv.org/pdf/2611.11960v1](https://arxiv.org/pdf/2611.11960v1)

## Abstract
Unplanned 30-day intensive care unit (ICU) readmission is a critical benchmark for evaluating healthcare quality. Most predictive models assume readmission risk depends solely on a patient's individual pathophysiological state, ignoring hospital operational context and resource capacity. In practice, discharge decisions are dynamic trade-offs between patient stability and bed turnover pressures. To address this gap, we develop an Operation-aware Multimodal Clinical Decision Support System (CDSS) for ICU discharge risk assessment. The system combines multi-agent collaborative extraction with self-evolving retrieval-augmented generation (RAG) to eliminate template noise from progress notes, while constructing departmental workload indicators alongside admission flow features, continuous vital signs, and sparse medical codes (ICD/DRG). Addressing severe class imbalance (about 10.5% readmissions) and alarm fatigue, we implement data normalization, class reweighting, and decision-threshold calibration. In 20 independent runs across 20,000 ICU stays from MIMIC-III, empirical findings demonstrate that: vital signs alone have limited predictive capacity under real-world class imbalance (PR-AUC of 0.1387 +/- 0.0086); directly concatenating raw text embeddings degrades decision tree performance (XGBoost and LightGBM PR-AUC drop to 0.3893 and 0.4030); whereas an end-to-end multimodal fine-tuning architecture (Transformer E2E FineTune) utilizing layer-wise parameter freezing and differential learning rates reverses text negative transfer, achieving the highest AUROC of 0.8288 +/- 0.0110, PR-AUC of 0.4373 +/- 0.0310, and F1-Score of 0.4757 +/- 0.0196. Operational indicators supply independent risk signals, laying an engineering foundation for high-utility clinical decision support tools.

## Full Text


<!-- PDF content starts -->

Operation-aware Multimodal Clinical Decision Support System
for ICU Discharge Risk Assessment
Guoqing Zhang,1Rafik Hadfi,1Takayuki Ito1
1Graduate School of Informatics, Kyoto University, Kyoto, Japan
zhang.guoqing.27t@st.kyoto-u.ac.jp, rafik.hadfi@i.kyoto-u.ac.jp, ito@i.kyoto-u.ac.jp
Abstract
Unplanned 30-day intensive care unit (ICU) readmission fol-
lowing hospital discharge is a major benchmark for evaluat-
ing healthcare quality and patient safety in modern critical
care medicine. Most existing predictive approaches assume
that readmission risk is determined solely by a patient’s in-
dividual pathophysiological state, ignoring the institution’s
resource capacity and operational context (Hospital Opera-
tional Context). In clinical practice, discharge decisions are
frequently dynamic trade-offs made by intensivists balanc-
ing patient medical stability against ward bed turnover pres-
sures. To address this gap, we design an Operation-aware
Multimodal Clinical Decision Support System (CDSS) for
ICU discharge risk assessment. On the one hand, the sys-
tem unites multi-agent collaborative extraction with self-
evolving retrieval-augmented generation (RAG) to eliminate
formatting-template noise across lengthy progress notes. On
the other hand, it constructs departmental workload indica-
tors (icu_speedup_los) alongside admission flow char-
acteristics,unifyingcontinuousvitalsigns,hierarchicalsparse
medical codes (ICD/DRG), and discharge summary seman-
tics. Addressing the extreme class imbalance (∼10.5% read-
mission incidence) and the challenge of alarm fatigue, the
systemimplementsdatanormalization,classreweightingop-
timization,andanF1decision-thresholdcalibrationpipeline.
In 20 independent multi-seed runs conducted on a cohort of
20,000 ICU stays from MIMIC-III, empirical findings show
that:asingle-modalitybaselinedependingexclusivelyonvital
signs possesses limited predictive capacity under real-world
class imbalance (PR-AUC of only0.1387±0.0086); di-
rectlyconcatenatingrawtextembeddingscausesperformance
degradation in decision tree models (XGBoost and Light-
GBM PR-AUC drop to 0.3893 and 0.4030, respectively);
whereas an end-to-end multimodal fine-tuning architecture
(Transformer E2E FineTune) utilizing layer-wise parameter
freezing and differential learning rates successfully reverses
text negative transfer, achieving the overall highest AUROC
of0.8288±0.0110,PR-AUCof0.4373±0.0310,andacom-
petitiveF1-Scoreof0.4757±0.0196.Mechanisticattribution
analysesalsoconfirmthatoperationalindicatorssupplyinde-
pendent risk signals, establishing an engineering foundation
for deploying high-utility clinical decision support tools in
bedside hospital wards.
Code— https://github.com/KuoChing-cmd/
ClinicalNoteAgent/
Copyright©2027, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.S1 Introduction
In intensive care clinical decision-making, identifying vul-
nerable patients at risk of latent deterioration and avoiding
prematurestep-downtogeneralwardsarecoreimperativesof
criticalcaremedicine.PatientsexperiencingunplannedICU
readmission within 30 days of discharge undergo substan-
tially prolonged hospital stays and face elevated in-hospital
mortality (Klinge et al. 2020; Barbieri et al. 2020; Sheetrit,
Brief, and Elisha 2023). While recent multicenter investi-
gations confirm the promise of machine learning for post-
discharge surveillance (Lim et al. 2025), existing clinical
early warning systems commonly encounter barriers during
ward deployment due to alarm fatigue: in low-prevalence
locales where readmissions occur in only∼10% of cases,
typical models optimized for high AUROC tend to trigger
substantialfalse-positivealarms,promptingclinicianstoturn
offthealertsystems.ToconstructanactionableClinicalDe-
cisionSupportSystem(CDSS),onemustpreservescreening
sensitivity while substantially improving precision, thereby
driving PR-AUC and F1 scores toward clinically acceptable
operating points.
Althoughelectronichealthrecord(EHR)repositoriessuch
as MIMIC-III (Johnson et al. 2016) have advanced ma-
chine learning in critical care—spanning deep temporal fo-
cus (Madinei, Keyvanpour, and Shojaedini 2024), graphical
EHR representations (Choi et al. 2020), and recent read-
missionbenchmarks(Fathy,Emeriaud,andCheriet2025)—
state-of-the-art predictive models still encounter two funda-
mental deployment bottlenecks. First, they exhibit a neglect
ofhospitaloperationalcontext.Standardmodelsconceptual-
ize patients as closed physiological entities, presuming that
deterioration following transfer stems entirely from built-in
disease progression. In clinical practice, however, ICU dis-
chargeisfundamentallyaresourcedeploymentandschedul-
ing process carried out under institutional bed constraints
(Kim et al. 2015; Wagner et al. 2013; Long and Mathews
2018).Intensivists mustroutinely weighincompletephysio-
logical recovery against acute bed demands driven by emer-
gencydepartment(ED)boardingorpost-operativetransfers.
PatienttransferfrequencybeforeICUadmission,EDboard-
ingdurations,andwardaccelerationdrivenbybedsaturation
directly mirror operational workflow strain and care coor-
dination friction; yet, customary doctrines remain entirely
agnostictothissystemicbackground.Second,modelssuffer
arXiv:2611.11960v1  [cs.LG]  9 Nov 2026

from data disparity and text template noise. Intensive care
records span high-frequency temporal streams, sparse diag-
nosticcodes,andlengthyunstructuredprogressnotes.While
recent studies explore medical text via knowledge graphs
(Lu, Nguyen, and Dou 2021) or multimodal architectures
for readmission prediction (Helmy et al. 2025; Datta et al.
2025), clinical free-text is often lengthy, laden with jargon,
and burdened by copy-pasted administrative templates. Di-
rectly feeding unadapted high-dimensional text embeddings
failstoyieldmeaningfulgainsandinsteadintroducessevere
featureinterference(negativetransfer)withstrongbaselines
such as gradient-boosted decision trees.
To resolve these problems, we reframe ICU discharge
evaluation as an operationally aware multimodal decision-
support task and develop a CDSS architecture that encom-
passesself-evolvingtextualdistillationanddeepcross-modal
correlation.Thecorecontributionsofthispaperaresumma-
rized as follows:
•Operation-aware Decision Framework: Overcoming ran-
domized timestamp offsets in MIMIC-III, we intro-
duce unit-benchmarked turnover deviation indicators
(icu_speedup_los) and care flow complexity fea-
tures, eliminating the subjective assumption that out-
comes depend solely on physiological status.
•Self-Evolving Multi-Agent Text Distillation: We formu-
late a 7-domain specialist agent collaborative division
of labor paired with a reflective evolutionary experience
repository, accumulating 3,085 rules across 5,000 pa-
tient samples. Combined with FAISS vector retrieval for
structured information distillation, we propose a layer-
wise,frozen,end-to-endfine-tuningschemethatsystem-
aticallymitigatesinterferencefromtemplatenoiseinclin-
ical notes.
•Thorough Testing under Low-Prevalence Conditions:
Across a cohort of 20,000 ICU stays, we conduct 20 in-
dependent randomized multi-seed runs (Seeds 42–61),
exposing the weakness of single-modality physiological
prediction (PR-AUC of only 0.1387) and verifying the
robustness of the end-to-end fine-tuning architecture in
achieving global optimality (AUROC 0.8288, PR-AUC
0.4373, F1 0.4757).
•Interpretability & Clinical Utility Analysis: Through in-
tegrated gradients (Sundararajan, Taly, and Yan 2017),
we systematically explain the mechanistic contributions
ofseparatephysiologicalchannels,pivotaltime-basedat-
tentionwindows,andoperationalflowvariablesin30-day
readmission forecasting.
S2 Clinical Workflow and Problem
Formulation
S2.1 Clinical Prediction Task and Endpoint
Definition
The prediction target is unplanned ICU readmission within
30 days following discharge (30-Day Readmission). In light
ofcomplexintra-hospitalpatienttrajectories,endpointlabels
yi∈ {0,1}follow strict clinical mutually exclusive criteria:•Cross-admission readmission: The patient undergoes a
separate hospital admission and enters an ICU within
30 days post-discharge (sameSUBJECT_ID, distinct
HADM_ID).
•Within-admission readmission: During the same hospi-
talization episode, the patient is transferred out of the
ICU,deteriorates,andissubsequentlyreadmittedbackto
the ICU (multiple ICU stays under the sameHADM_ID,
where the subsequent admission timestamp postdates
the preceding transfer timestamp, i.e., INTIME k+1>
OUTTIME k).
•Mortality Mutual-Exclusion: Patients who suffer in-
hospital mortality (HOSPITAL_EXPIRE_FLAG = 1)
have their readmission risk strictly calibrated to 0, elimi-
nating logical contradictions.
S2.2 Multimodal Clinical Data Streams and
Domain Alignment
For each ICU stay eventi∈ {1, . . . , N}(experimental co-
hortN= 20,000),theinputisstructuredasaheterogeneous
quadruple:
Xi=
Xseq
i,xstat
i,xcode
i,xtext
i	
(S1)
First,forhigh-frequencydynamicphysiologicalsequences
(Xseq
i∈R48×8), the system captures 8 core physiological
signs during the initial 48 hours following ICU admission:
Heart Rate (HR), Respiratory Rate (RR), Oxygen Satura-
tion(SpO2),SystolicBloodPressure(SBP),DiastolicBlood
Pressure (DBP), Mean Arterial Pressure (MAP), Tempera-
ture (Temp), and Blood Glucose (Glucose). Blood pressure
measurements explicitly distinguish invasive arterial pres-
surefromnon-invasivecuffpressure;whenbothcoexist,the
invasive arterial line—the gold standard in critical care—is
preferred.Timeseriesareresampledinto1-hourregularized
bins, with missing steps forward-filled.
Second, structured demographic and operational indica-
tors (xstat
i∈R15) capture baseline patient context and care
delivery friction across three facets: (i)Baseline & Surgical
covariatesinclude age, gender, marital status, ethnicity, in-
surance, admission source/type, first ICU unit, surgical pro-
cedure count, surgery binary indicator, and log-transformed
surgery-to-ICU entry gapln(1 + ∆t surg). (ii)Transfer Pat-
ternstrackthenumberofwardtransferspriortoICUadmis-
sion(pre_icu_transfers)alongwithlog-transformed
emergencydepartmentwaittimeln(1+t ED).(iii)WardPres-
sure & Turnover Deviation: addressing the obstacle where
randomizedabsolutetimestampsinMIMIC-IIIprecludecon-
current bed occupancy observation across patients, we for-
mulate a stay deviation metric aligned with historical unit
benchmarks (e.g., MICU, SICU) along with stay duration
features:
icu_speedup_los= LOSunit−LOS actual
log_icu_los= ln(1 +LOS hours)(S2)
Specifically,underazero-leakagecutoff,thefeatureisstrictly
computed at the exact moment of the step-down transfer
order, sharing the same temporal cross-section with dis-
chargenotesandprecludingfutureinformationleakage.Cru-
cially, contrary to the naive assumption that shorter stays

merely reflect milder acuity (which would yield lower read-
mission rates), empirical attribution reveals that positive
icu_speedup_los(early turnover) significantly corre-
lates with higher readmission risk (20.1% continuous IG at-
tribution).Thisrefutesindicationconfoundingandconfirms
that the metric captures external operational bed-turnover
pressure and premature discharge before complete physio-
logical stabilization.
Third, hierarchically decoupled sparse medical codes
(xcode
i) capture hierarchical EHR coding structures (Choi
etal.2020).PrimaryICDdiagnosesandprimaryDRGcodes
select the Top-128 categories, mapped into categorical in-
dices feeding a learnable continuous embedding layer (128
dimensions). Secondary diagnoses (Top-32), surgical pro-
cedures (Top-64), and prescribed medications (Top-64) are
encodedintomulti-hotsparsevectors,projectedthroughem-
beddingweightmatrices,mean-pooledacrossactiveentries,
and concatenated into compact representations.
Fourth, unformatted clinical progress representations
(xtext
i) incorporate free-text clinical progress notes. In con-
trasttostandardclinicaltextbaselines(Lu,Nguyen,andDou
2021;Helmyetal.2025)thatdirectlyprocessprogressnotes
vulnerable to template noise, lengthy inpatient records are
distilled via self-evolving multi-agent workflows into struc-
tured summaries, with 768-dimensional semantic features
extractedbyBio_ClinicalBERT(Alsentzeretal.2019).
S3 Operation-aware CDSS Architecture
S3.1 Unstructured Text Mining: Multi-Agent
Self-Evolving RAG Mechanism
Orchestrated multi-agent architectures sustain diagnostic fi-
delityundercomplexclinicalworkloadscomparedtomono-
lithic models (Klang et al. 2026). To overcome verbosity
and repetitive administrative boilerplate in clinical progress
notes,thesystemestablishesaself-evolvingmulti-agentdis-
tillation pipeline across five coordinated stages: (i)Hierar-
chical Specialist Collaboration: Clinical text is segmented
across 7 specialized clinical domains (Nursing, Radiology,
ICU Care, ECG, Attending Physician, Discharge Summary,
and Echocardiography), where independent domain agents
extract acute warning signals. (ii)Meta-Agent Synthesis: A
supervisoryMeta-Agentaggregatesdomainsummaries,syn-
thesizing a structured JSON report containing a prioritized
vital factor roster and composite risk score. (iii)Reflec-
tive Evolution Loop: When predictions diverge from clin-
ical ground truth during training, a Reflection Agent au-
ditscausalfailurepoints,extractingphysician-vettedheuris-
tic rules into a knowledge base (3,085 rules); correctly
classified trajectories are cataloged as gold-standard exem-
plars (1,899 cases). (iv)Lightweight RAG Retrieval: Guided
byall-MiniLM-L6-v2embeddings and a FAISS in-
dex, the pipeline dynamically retrieves the Top-5 rele-
vant heuristic rules and Top-1 exemplar case per admis-
sion as few-shot in-context demonstrations. (v)Feature Em-
bedding & Projection: Distilled summaries are encoded
byBio_ClinicalBERTinto768-dimensionalcontextual
vectors and projected to ad-dimensional embeddingE note
aligned with temporal representations.S3.2 Cross-Modal Network Architecture and
Deep Fusion Schemes
The framework investigates three architectural paradigms.
First, we evaluate tree-based baselines: XGBoost (Chen and
Guestrin 2016) and LightGBM (Ke et al. 2017) fitted on
physicallyconcatenatedvectorscombiningtemporalstatisti-
calaggregations,staticcovariates,multi-hotcodes,and768-
dimensional text embeddings (Chen et al. 2025). Second,
we construct sequential late fusion (LSTM Late Fusion):
a 4-layer bi-directional LSTM with self-attentive temporal
pooling encodes physiological sequences, concatenated at
the classification tier with static, code, and text representa-
tions projected via independent MLP branches. Third, we
formulate parallel multi-branch extraction and late fusion.
Dynamic physiological sequences are encoded via a deep
Transformer encoder (Vaswani et al. 2017) intoZ cls; static
operationalindicatorsandmedicalcodesaremappedviain-
dependent MLPs into dense vectorsE opsandE code. These
representationsareconcatenatedwithtextembeddingE note
to formZ fused= [Z cls∥Eops∥Ecode∥Enote]∈R2d+64,
feedinga2-layerMLPheadforjointriskestimation.Topre-
ventrepresentationalcollapseduringend-to-endtraining,the
initial 8 layers ofBio_ClinicalBERTare frozen, leav-
ing deeper layers (Layers 9–12) active; BERT parameters
receive a 0.1×scaled learning rate (lr *0.1) with gra-
dient checkpointing enabled.
S3.3 Training Optimization and Leak-Free
Validation Protocols
To ensure leak-free evaluation under severe class im-
balance (∼10.5% prevalence) and achieve stable multi-
modal convergence, we institute four disciplined method-
ological safeguards: (i)Zero-Leakage Normalization:
StandardScalerparameters are fitted strictly on train-
ing splits; validation and testing cohorts inherit these statis-
tics without re-estimation, precluding distribution leakage.
(ii)Dynamic Cost-Sensitive Loss: To counter minority-class
sparsity, a weighted binary cross-entropy loss deploys dy-
namic inverse class ratio weighting derived strictly from
the training split:w pos=N neg/Npos≈8.97. (iii)Cosine
Scheduling & Rollback: ACosineAnnealingLRsched-
ule regulates learning rate decay, paired with early stop-
ping(patience=15)onvalidationloss;upontermination,
model parameters rollback to the checkpoint achieving the
minimal validation loss. (iv)Decision Threshold Calibra-
tion: On validation splits, exhaustive grid search (step size
0.01) determines the optimal decision cutofft∗maximizing
theF 1-Score,replacingtheuncalibrateddefault0.5threshold
for final test evaluation.
S4 Experimental Evaluation and Validation
S4.1 Evaluation Protocol and Cohort Setup
Experiments are run across 20,000 ICU stay records from
theMIMIC-IIIdatabase.Thecohortispartitionedintotrain-
ing (14,000 stays), validation (2,000 stays), and test splits
(4,000stays)ina7:1:2ratio,withpositivereadmissionpreva-
lence ranging from 10.5% to 11.4%. To eliminate partition-
inganomalies,allcandidateapproachesareevaluatedacross

(a) Multimodal EHR Inputs & Operational Context
Alignment
/heartbeatDynamic High-Frequency Physiology (Xseq∈R48×8)
48h×8 Channels
Channels:HR, RR, SpO2, SBP, DBP, MAP, Temp, Glucose|
1-hour uniform grid
(Clinical Protocol:Arterial-Line invasive BP prioritized over Cuff NIBP)
Ἶ5Hospital Operational Context & Flow Friction (xstat∈R15)
Core Innovation
Unit-Level Bed Strain:icu speedup los= LOSunit−LOS actual
(Quantifies premature discharge pressure to overcome timestamp jitter)
Flow Complexity:Pre-ICU Transfers|ED Waitingln(1 +t ED)|
Surgical Gapln(1 + ∆t surg)
/file-medicalDecoupled Hierarchical Medical Codes (xcode)
Primary Diagnosis & DRG:
Top-128 ICD-9 + Top-128 DRG
→Discretenn.Embedding(128,
d)
Comorbidities & Interventions:
Secondary ICD (32), Proc (64), Rx
(64)
→Multi-Hot Active Mean Pooling
(b) Self-Evolving Multi-Agent Clinical RAG Pipeline
/user-md7 Domain Clinical Expert Agents (Parallel Extraction of Free-Text
Notes) Specialized
Nursing Radiology ICU Care ECG Physician Discharge Echo
Filter administrative boilerplate templates−→Distill acute high-risk findings
/da◎abaseFAISS Dual-Channel Knowledge Retrieval (all-MiniLM-L6-v2)
Doctor Experience Rules
3,085 Validated Rules
(Retrieve Top-5 Relevant Rules)
Golden Exemplar Cases
1,899 Validated Trajectories
(Retrieve Top-1 In-Context Demo)
/brainMaster Meta-Agent & Evolutionary Reflection Closed Loop
Structured JSON Distillation:
{mortality risk: 7,readmit risk: 8,[factors]}
+High-Risk Clinical Narrative Summary(Contextualized free text)
On Misclassification (Error):
Reflection Agent diagnoses root cause
−→+1 Doctor Experience Rule
On Dual-Task Correct (Success):
Archive verified diagnostic pathway
−→+1 Golden Exemplar Case(c) Parallel Extraction & Late Fusion Architecture
♂network-wiredDeep Transformer Encoder (Vitals Only,L= 4, 8 Heads)
Input:Xseq∈R48×8+[CLS]Transformer− − − − − − →Self-Attention Interacts 48h Vitals
Output:Temporal RepresentationZ cls∈Rd
/cogsOperational Context Linear / MLP Head
xstat∈R15Linear/MLP− − − − − − →OutputE ops∈R32
/code-branchMedical Codes Embedding & MLP Head
xcode Embedding + Pooling− − − − − − − − − − − − →MLP− − →OutputE code∈R32
♂robotBio ClinicalBERT End-to-End Fine-Tuning Branch
Input:Clinical Narrative Summary
Frozen Layers 1–8 Fine-Tuned Layers 9–12
BERT [CLS]− − − − − − − →Linear− − − →OutputE note∈Rd|Joint lr:
lrBERT = 0.1×lr main
♂object-groupLate Fusion (Concatenation)
Zfused = [Z cls∥Eops∥Ecode∥Enote]∈R2d+64
/char◎-lineClassification Head:
Zfused→LayerNorm→Dropout(0.3)→Linear(2d+64,32)→ReLU
→Dropout(0.1)→Linear(32,1)→Sigmoid→ˆy∈[0,1]Backprop Gradients (lr×0.1)(d) Calibration & Bedside CDSS Interface
♂shield-altAnti-Leakage Protocol & Training Optimization
Zero-Data-Leakage Scaling:StandardScaler
fit strictly on Train split (14k Train, 2k Val, 4k Test)
Reweighted BCE Loss:w pos=N neg/Npos≈8.97
CosineAnnealingLR|EarlyStopping (Patience=15)
/balance-scaleAlarm Fatigue Mitigation & Calibration t∗= 0.754
Validation Best-F1 Grid Search:t∗= 0.754±0.049
(Eliminates excessive false alarms under 10.5% low prevalence)
20-Run Test Benchmark (Mean±Std):
AUROC:0.8288±0.0110|PR-AUC:0.4373±0.0310
Best F1:0.4757±0.0196
/desk◎opActionable Bedside CDSS Interface
30-Day ICU Readmission Risk Stratification:
Low Risk High Risk
(Provides timely clinical warnings to delay premature discharge)
Integrated Gradients Attribution:
•Heart Rate (HR) (32.1%)→Critical dynamic stress
response
•Pre-ICU Transfers (28.2%)→Inter-ward friction
•Primary ICD & DRG (25.7%)→Baseline comorbidity
burden
•Oxygenation SpO2(10.3%)→Hypoxic instability48h Vitals
Ops & Bed Strain
Sparse Codes
Clinical Summary TextPredictedˆyFigure S1: System overview of the Operation-aware Multimodal Clinical Decision Support System (CDSS). (a) Multimodal
EHRinputsalignedwithhospitaloperationalbottleneckperception;(b)Self-evolvingmulti-agentclinicalRAGminingpipeline
with 7 domain specialists and dual-track rule/case repositories; (c) Parallel feature derivation and Late Fusion architecture
alongsidetheBio_ClinicalBERTend-to-endfine-tuningpathway;(d)Zero-leakagenormalization,alarm-fatiguecalibrated
thresholds, and bedside risk-stratified decision support.
20 independent randomized runs (Seeds 42–61), with test
performance reported as Mean±Std.
S4.2 Comprehensive Performance Comparison
and Paradigm Defense
Table S1 summarizes the evaluation measures across 15
model configurations evaluated at their optimal F1 decision
thresholds across 20 randomized runs. Three core empiri-
cal insights emerge. First, we observe text negative trans-
fer in shallow concatenation. Directly appending unadapted
768-d text embeddings into decision trees (XGBoost/Light-
GBM)triggersnotableperformancedecline.Inpatientnotes
are saturated with administrative templates and boilerplate
shorthand;high-dimensionalembeddingsdiffusegreedytree
node splitting, diluting compact, high-SNR physiological
and operational signals. Second, results demonstrate the
strength of structured baselines and multimodal reframing.
Purely structuredLSTM Baseestablishes a solid bench-
mark (F 1= 0.4798), demonstrating the fitting efficiency of
recurrent cells over 48-hour dynamics along with discrete
codes.Themultimodalgoalisthereforenottosurpassstruc-
tured models trivially, but to resolve a harder clinical chal-
lenge:safelyincorporatinghigh-valuetextsemanticswithout
suffering negative transfer(as observed in Transformer Ear-
lyFusion, which degrades under raw text noise). Third, weachieve a reversal of negative transfer via end-to-end fine-
tuning.Withlayer-wiseparameterfreezing(Layers9–12ac-
tive) and differential learning rates,Transformer E2E
FineTuneis the sole architecture reversing text negative
transfer into positive gains, securing global peaks in both
AUROC (0.8288) and PR-AUC (0.4373). Paired hypothe-
sis tests across 20 matched random seeds confirm that the
AUROC gain of end-to-end fine-tuning over static early fu-
sion is highly statistically significant (mean AUROC 0.8288
vs. 0.8177; Pairedt-testt= 4.89, p <0.001; Wilcoxon
p <0.001). This provides strong empirical evidence that
end-to-end backpropagation prevents the text representation
from being overshadowed by dense temporal dynamics, en-
abling the language model to better align textual risk cues
with structured physiological trajectories.
S4.3 Sensor Dropout Robustness
InoperationalICUenvironments,electrodedetachment,pa-
tient movement, and intermittent contact failures occur fre-
quently. To assess fault tolerance under incomplete phys-
iological telemetry, we evaluated the representativeLSTM
LateFusionmodel under random telemetry loss (Sensor
Dropoutfrom10%to30%)across20independentruns.The
system presented remarkable stability: under 30% channel
dropout, AUROC decayed by less than 0.0005 in absolute
terms(from0.8258±0.0091to0.8254±0.0092),PR-AUC

TableS1:30-dayunplannedICUreadmissionpredictionperformanceonthetestset(20independentruns,mean±Std,evaluated
@ optimal F1 decision threshold). Bold indicates global optimum.
Model Category Model Name & Variant Optimal Thresholdt∗AUROC PR-AUC F1-Score
Tree Baselines & Text AblationXGBoost Base0.314±0.093 0.8055±0.0098 0.4125±0.0209 0.4465±0.0193
XGBoost + LLM Notes0.134±0.053 0.7914±0.0118 0.3893±0.0213 0.4400±0.0166
LightGBM Base0.463±0.067 0.8158±0.0111 0.4114±0.0187 0.4555±0.0172
LightGBM + LLM Notes0.370±0.064 0.8087±0.0114 0.4030±0.0193 0.4548±0.0170
Deep Multimodal Networks & FusionLSTM Base (Pure Structured)0.747±0.051 0.8251±0.0117 0.4360±0.03050.4798±0.0200
LSTM LateFusion (+ Notes)0.726±0.034 0.8258±0.0091 0.4338±0.0301 0.4791±0.0193
Transformer EarlyFusion (+ Notes)0.757±0.068 0.8177±0.0145 0.4278±0.0332 0.4769±0.0200
Transformer LateFusion (+ Notes)0.762±0.042 0.8277±0.0104 0.4357±0.0320 0.4751±0.0158
LSTM E2E FineTune (+ Notes)0.733±0.044 0.8260±0.0119 0.4338±0.0326 0.4754±0.0178
Transformer E2E FineTune (+ Notes)0.754±0.0490.8288±0.0110 0.4373±0.03100.4757±0.0196
Modality & Feature Ablation (relative to LSTM Base)w/o Admission Dynamics (pre_icu_transfers,t ED)0.750±0.043 0.8260±0.0103 0.4292±0.0287 0.4772±0.0204
w/o ICU Operational Features (icu_speedup_los, LOS)0.734±0.044 0.8275±0.0111 0.4278±0.0312 0.4791±0.0195
w/o Medical Codes (xcode)0.741±0.033 0.8243±0.0107 0.4334±0.0302 0.4757±0.0187
Only Vitals (Xseq)0.521±0.033 0.6232±0.0149 0.1387±0.0086 0.2213±0.0099
Only Static & Codes (xstat,xcode)0.735±0.040 0.8227±0.0097 0.4294±0.0273 0.4749±0.0180
remained steady (0.4338→0.4330), andF 1was preserved
at0.4783±0.0189. This confirms that multimodal fusion
provides robust cross-modal safety redundancy, in which
static operational indicators, diagnostic codes, and clinical
textinsulatedecisionsupportduetolocalizedtelemetrydis-
ruptions.
S4.4 Modal Ablation and Vitals Collapse
Feature ablation displays strong asymmetries across modal-
ities. Relying exclusively on 48-hour vital signs (LSTM
Base (Only Vitals)) precipitates catastrophic col-
lapse: AUROC plummets to0.6232and PR-AUC crashes
to0.1387(only marginally above the∼10.5% prevalence
baseline), yielding anF 1of only0.2213. This demon-
strates the inherent insufficiency of unimodal physiological
surveillance:immediatevitalscapturetransientacuteacuity,
whereaschroniccomorbiditiesandsystemicoperationalfric-
tionprimarilygovern30-dayoutcomes.Conversely,staticde-
mographicsandcodingalone(Only Static & Codes)
sustain an AUROC of0.8227and PR-AUC of0.4294
(|∆prob|= 0.1437), establishing patient baseline and op-
erational pressures as basic priors, with dynamic vitals
(|∆prob|= 0.1014) providing acute trajectory corrections.
S5 Mechanistic Attribution and Case Study
S5.1 Multimodal Attribution: Channels,
Dynamics, and Operations
IntegratedGradients(IG)andattentionanalysesuncovercol-
laborative mechanisms throughout modalities (Figure S2).
First,regardingvitalsigndynamics,HeartRate(32.0%)and
Temperature (22.2%) account for over half of physiolog-
ical IG attribution, reflecting cardiovascular reserve com-
pensation and occult infection as primary discharge stabil-
itydeterminants.Bloodglucose(13.7%),SBP(12.5%),and
SpO2(10.3%) supply secondary warnings against unmiti-
gated glycemic swings and tissue hypoperfusion. Second,
with respect to time-based attention focus, attention con-
centrates heavily on initial admission (t∈[0,2]h, Hour 0
weight=0.0806),showingthatinitialresuscitationdifficulty
strongly anchors downstream readmission risk. Across the
48h trajectory, intermediate monitoring (12–36h, 42.3%)
and acute admission (0–12h, 37.9%) dominate reasoning,while pre-discharge attention declines (36–48h, 19.7%), in-
dicating that the model assesses stabilization effort rather
than superficial pre-discharge calm. Third, empirical find-
ings establish the primacy of operational friction. Pre-ICU
transfer frequency (pre_icu_transfers) accounts for
28.2%ofstaticablationand50.9%ofcontinuousfeatureIG
attribution, dominating all covariates. Ward turnover pres-
sure (icu_speedup_los) contributes 9.6% of ablation
and 20.1% of continuous IG, proving that bed-saturation-
driven premature discharge elevates hidden deterioration
risks, alongside surgical count (surg_count) and gap.
S5.2 Typical Clinical Case Vignette
A 68-year-old patient admitted with respiratory failure
achieved stable vitals after 48 hours in the ICU (nor-
mal BP and HR). Conventional heuristics mirroring vitals-
only logic would categorize the patient as fully ready
for discharge. However, our CDSS detected that before
ICU admission, the patient spent over 16 hours boarding
in the emergency department and underwent two intra-
hospital transfers (pre_icu_transferselevated); fur-
thermore, the patient’s unit was under extreme bed strain
(icu_speedup_losnotablypositive).Synthesizingmul-
timodal evidence, the CDSS triggered an elevated-risk alert
for handoff friction and metabolic instability, prompting the
intensivist to extend observation by 24 hours and recheck
blood gases, successfully avoiding premature discharge and
subsequent readmission.
S6 Conclusion
Synthesizingsystemdevelopmentand20experimentalruns,
we highlight four engineering lessons for actual clinical AI
adoption. First, we recognize the limits of vitals-only mon-
itoring: the catastrophic failure ofOnly Vitals(F 1=
0.2213) proves that immediate telemetry reflects transient
acuityratherthanmedium-termprognosis,mandatingholis-
tic cross-modal modeling. Second, mitigating text template
noise is critical; because inpatient records are heavily con-
taminatedwithadministrativeboilerplate,layer-wiseparam-
eter freezing and differential learning rates are required to
avoid text negative transfer. Third, operational context acts
asanendogenousrisk:pre-ICUtransferfrictionandturnover

0 0.1 0.2 0.3DBPRRMAPSPO2SBPGLUCOSETEMPHR
Integrated Gradients(a) Vital Signs Importance (IG)
Static Vitals Codes Notes00.050.100.150.20|∆prob|(b) Modality Ablation Impact
0 12 24 36 4700.020.040.060.08
Time Steps (Hours)Attention Weight(c) Temporal Attention over 48h
0 0.02 0.04 0.06 0.08 0.10ETHNICITYINSURANCEageFIRST CAREUNITADM LOCATIONsurg countdrgcodeicuspeedup losprimary icdpreicutransfers
|∆prob|(d) Top 10 Static Features AblationFigureS2:Featureattributionandablationanalysisofmultimodalmodelresults.(a)IntegratedGradients(IG)importanceacross
temporalphysiologicalchannels;(b)Ablationcontributionmagnitudeacrosscoreinputmodalities(|∆prob|);(c)Temporalself-
attention weight distribution over the 48-hour sequence; (d) Top-10 ranked static operational and demographic features by
ablation importance.
pressure account for over 70% of continuous static attribu-
tion,demonstratingthatinstitutionalstrainisanendogenous
risk driver rather than exogenous noise. Fourth, threshold
calibration is indispensable against alarm fatigue: in low-
prevalence settings (∼10%), default 0.5 thresholds collapse
precision;dynamicvalidation-setgridsearchfort∗ismanda-
tory to avoid alert abandonment.
Two-StageDeploymentRoadmap:Real-worldEHRsystems
are optimized for legal archiving rather than machine learn-
ing,andcriticalcarecannottoleratetrial-and-erroronlivepa-
tients. We propose a two-stage deployment pathway: “Near-
TermSilentShadowVerification,Long-TermClinicalDigital
Twins.”Inthenearterm,theCDSSinterfacesnon-intrusively
with EHR discharge order streams in Silent Shadow Mode
to quantify the utility of human-artificial intelligence col-
laboration. In the long term, longitudinal EHR repositories
enable high-fidelity Clinical Digital Twins simulation sand-
boxes (Li et al. 2024), allowing multi-agent architectures to
undergosafetystress-testinginaparalleldigitalwardbefore
deploying high-assurance decision policies at the bedside.References
Alsentzer, E.; Murphy, J. R.; Boag, W.; Weng, W.-H.;
Jin, D.; Naumann, T.; and McDermott, M. 2019. Pub-
licly available clinical BERT embeddings.arXiv preprint
arXiv:1904.03323.
Barbieri, S.; Kemp, J.; Perez-Concha, O.; Kotwal, S.; Gal-
lagher, M.; Ritchie, A.; and Jorm, L. 2020. Benchmark-
ingdeeplearningarchitecturesforpredictingreadmissionto
the ICU and describing patients-at-risk.Scientific Reports,
10(1): 1111.
Chen, S.; Fan, J.; Pishgar, E.; Alaei, K.; Placencia, G.; and
Pishgar, M. 2025. XGBoost-based prediction of ICU mor-
tality in sepsis-associated acute kidney injury patients using
MIMIC-IV database with validation from eICU database.
medRxiv, 2025–02.
Chen, T.; and Guestrin, C. 2016. XGBoost: A scalable tree
boostingsystem. InProceedingsofthe22ndACMSIGKDD
InternationalConferenceonKnowledgeDiscoveryandData
Mining, 785–794.
Choi,E.;Xu,Z.;Li,Y.;Dusenberry,M.;Flores,G.;Xue,E.;
and Dai, A. 2020. Learning the graphical structure of elec-
tronic health records with graph convolutional transformer.
InProceedings of the AAAI Conference on Artificial Intelli-
gence, volume 34, 606–613.
Datta, R.; Cui, J.; Guan, Z.; Silwal, R.; Eby, J. C.; Madden,
G.; and Vullikanti, A. 2025. Improving Hospital Risk Pre-
dictionwithKnowledge-AugmentedMultimodalEHRMod-
eling.arXiv preprint arXiv:2508.01970.

Fathy, W.; Emeriaud, G.; and Cheriet, F. 2025. A compre-
hensivereviewofICUreadmissionpredictionmodels:From
statistical methods to deep learning approaches.Artificial
Intelligence in Medicine, 165: 103126.
Helmy, H.; Ibrahim, A.; Arabi, M.; Sattar, A.; and Serag,
A. 2025. Leveraging large language models to predict un-
planned ICU readmissions from electronic health records.
Natural Language Processing Journal, 100182.
Johnson, A. E.; Pollard, T. J.; Shen, L.; Lehman, L.-w. H.;
Feng, M.; Ghassemi, M.; Moody, B.; Szolovits, P.; An-
thony Celi, L.; and Mark, R. G. 2016. MIMIC-III, a freely
accessible critical care database.Scientific Data, 3(1): 1–9.
Ke,G.;Meng,Q.;Finley,T.;Wang,T.;Chen,W.;Ma,W.;Ye,
Q.;andLiu,T.-Y.2017. LightGBM:Ahighlyefficientgradi-
ent boosting decision tree.Advances in Neural Information
Processing Systems, 30.
Kim,S.-H.;Chan,C.W.;Olivares,M.;andEscobar,G.2015.
ICUadmissioncontrol:Anempiricalstudyofcapacityallo-
cationanditsimplicationforpatientoutcomes.Management
Science, 61(1): 19–38.
Klang, E.; Omar, M.; Raut, G.; Agbareia, R.; Timsina, P.;
Freeman, R.; Gavin, N.; Stump, L.; Charney, A. W.; Glicks-
berg, B. S.; et al. 2026. Orchestrated multi agents sustain
accuracy under clinical-scale workloads compared to a sin-
gle agent.npj Health Systems, 3(1): 23.
Klinge, M.; Aasbrenn, M.; Öztürk, B.; Christiansen, C. F.;
Suetta,C.;Pressel,E.;andNielsen,F.E.2020. Readmission
of older acutely admitted medical patients after short-term
admissions in Denmark: a nationwide cohort study.BMC
Geriatrics, 20(1): 203.
Li,J.;Lai,Y.;Li,W.;Ren,J.;Zhang,M.;Kang,X.;Wang,S.;
Li, P.; Zhang, Y.-Q.; Ma, W.; et al. 2024. Agent hospital: A
simulacrumofhospitalwithevolvablemedicalagents.arXiv
preprint arXiv:2405.02957.
Lim, L.; Kim, M.; Cho, K.; Yoo, D.; Sim, D.; Ryu, H. G.;
and Lee, H.-C. 2025. Multicenter validation of a machine
learning model to predict intensive care unit readmission
within 48 hours after discharge.EClinicalMedicine, 81.
Long, E. F.; and Mathews, K. S. 2018. The boarding pa-
tient: effects of ICU and hospital occupancy surges on pa-
tientflow.ProductionandOperationsManagement,27(12):
2122–2143.
Lu,Q.;Nguyen,T.H.;andDou,D.2021. Predictingpatient
readmissionriskfrommedicaltextviaknowledgegraphen-
hanced multiview graph convolution. InProceedings of the
44thInternationalACMSIGIRConferenceonResearchand
Development in Information Retrieval, 1990–1994.
Madinei, H. S.; Keyvanpour, M. R.; and Shojaedini, S. V.
2024. An Attention-Based Model for Clinical Time Series
Prediction:EnhancingICUReadmissionPrediction. In2024
14thInternationalConferenceonComputerandKnowledge
Engineering (ICCKE), 119–124. IEEE.
Sheetrit, E.; Brief, M.; and Elisha, O. 2023. Predicting un-
planned readmissions in the intensive care unit: a multi-
modality evaluation.Scientific Reports, 13(1): 15426.Sundararajan, M.; Taly, A.; and Yan, Q. 2017. Axiomatic
attribution for deep networks. InInternational Conference
on Machine Learning, 3319–3328. PMLR.
Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones,
L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. At-
tention is all you need. InAdvances in Neural Information
Processing Systems, volume 30, 5998–6008.
Wagner, J.; Gabler, N. B.; Ratcliffe, S. J.; Brown, S. E. S.;
Strom, B. L.; and Halpern, S. D. 2013. Outcomes among
patients discharged from busy intensive care units.Annals
of Internal Medicine, 159(7): 447–455.

A Cohort Selection and Flow Diagram
S1.1 Study Population and Cohort Derivation
Protocol
This study derives its analytical cohort from theMIMIC-
III v1.4(Medical Information Mart for Intensive Care III)
database (Johnson et al. 2016), which comprises compre-
hensive clinical records from all intensive care admissions
atBethIsraelDeaconessMedicalCenter(N= 61,532ICU
stays across46,520unique patients).
In contrast to conventional benchmark studies that apply
post-hoc demographic exclusions (e.g., arbitrarily dropping
pediatric/neonatal admissions or short stays), our study re-
tainsallreal-worldICUcareunits(includingMICU,CSRU,
CCU, SICU, TSICU, and NICU) to preserve clinical gen-
eralizability. The end-to-end cohort derivation protocol is
illustrated in Figure S3.
S1.2 ClinicalEndpoint(GroundTruth)Definition
The primary target outcome is30-Day Unplanned ICU
Readmission. In critical care environments, patients fre-
quently transfer between emergency departments, surgical
suites, and step-down units. A positive binary ground truth
label(Y= 1)isrigorouslyassignedifeitherofthefollowing
conditions holds:
1.Cross-Admission Readmission (Condition A): The
identical patient (indexed by ‘SUBJECT_ID‘) is dis-
chargedfromthehospitalandsubsequentlyreadmittedto
anICUunderadistincthospitaladmission(‘HADM_ID‘),
where timestamps satisfy:
INTIME next>OUTTIME current ∧
INTIME next≤OUTTIME current + 30days (S3)
2.Within-Admission Bounce-Back (Condition B):
Within the same hospital encounter (‘HADM_ID‘), the
patient is transferred out of the ICU to a general or step-
down ward, acutely deteriorates, and is readmitted to an
ICU (INTIME>OUTTIME).
Competing Risk Censoring: In-hospital mortality repre-
sents a terminal competing risk that precludes future post-
dischargereadmission.Patientswhoexpiredduringtheirhos-
pitalstay(HOSPITAL_EXPIRE_FLAG = 1)arecensored
withY= 0. In our final cohort ofN= 20,000stays, ex-
actly2,000patients (10.00%) experienced 30-day readmis-
sion,yieldinganaturalclassimbalanceratioof1 : 9.During
training, this imbalance is calibrated via an inverse positive
loss weightw pos=Nneg/Npos≈8.97.
B MIMIC-IIIDataDistributionandBaseline
Characteristics
Table S2 details the baseline clinical, demographic, oper-
ational, and physiological characteristics across the entire
cohort(N= 20,000),stratifiedby 30-dayICUreadmission
status(Y= 1vs.Y= 0).Continuousvariablesarereported
as Mean±SD (Median) with hypothesis testing via two-
sidedMann-WhitneyUtests,whilecategoricalvariablesare
reported asN(%) with Pearson’sχ2test of independence.C Multi-Agent Clinical Prompt
Specifications
The distillation framework deploys a coordinated 3-tier hi-
erarchicalagentstructurebuiltonLlama-3.1(viaOllama
local inference). The prompt templates and system instruc-
tions for all three agent tiers are given below.
S3.1 Tier 1: Domain Specialist Extraction Agents
Each patient’s clinical note history is segmented into
7 clinical categories:Nursing/other,Radiology,
Nursing,ECG,Physician,Discharge summary,
andEcho. Each specialist evaluates up to 8,000 characters
of domain-specific narrative notes:
Listing 1: Prompt template for Tier 1 Domain Specialist
Agents.
1--------------------------------------
2DOMAIN EXPERT AGENT PROMPT TEMPLATE
3--------------------------------------
4You are an expert clinical AI. Analyze
the following
5{category} notes to extract critical
information that
6might predict 30-day hospital mortality
or ICU readmission.
7Focus on identifying any abnormal
findings, unresolved
8issues, or high-risk indicators specific
to this domain.
9
10{category} NOTES:
11{notes_text}
12
13Provide a concise, structured summary
highlighting key
14risk factors. Do not include
introductory text.
15--------------------------------------
S3.2 Tier 2: Lead Clinical Supervisory
Meta-Agent
ThesupervisoryMeta-Agentaggregatesthe7distilledbriefs.
ThroughFAISSRAGretrieval,itincorporatestheTop-5most
semanticallyaligneddoctorrulesandTop-1goldenexemplar
trajectory, outputting a calibrated JSON risk synthesis:
Listing 2: Prompt template for Tier 2 Lead Clinical Meta-
Agent.
1--------------------------------------
2LEAD CLINICAL META-AGENT PROMPT TEMPLATE
3--------------------------------------
4You are the lead meta-agent clinical AI.
You will receive
5summaries from various domain agents for
a single patient’s
6ICU stay.
7Your task is to synthesize these
summaries and produce a
8final, structured JSON report
identifying TWO distinct risks:
91. In-hospital mortality risk.
102. 30-day ICU readmission risk.

TableS2:Baselinedemographic,administrativeflow,physiological,andclinicalcodingcharacteristicsacrosstheN= 20,000
cohort.
Characteristic / Feature Overall Cohort (N= 20,000) Readmitted (Y= 1,N= 2,000) Non-Readmitted (Y= 0,N= 18,000)P-Value
Patient Demographics and Baseline Covariates
Age at Admission (years)50.72±30.50(58.46)60.98±22.87(64.20)49.58±31.03(57.10)<0.001
Biological Sex 0.012
— Female8,790 (43.95%) 834 (41.70%) 7,956 (44.20%)
— Male11,210 (56.05%) 1,166 (58.30%) 10,044 (55.80%)
Ethnicity Breakdown <0.001
— White / Caucasian14,120 (70.60%) 1,482 (74.10%) 12,638 (70.21%)
— Black / African American1,460 (7.30%) 154 (7.70%) 1,306 (7.26%)
— Hispanic / Latino640 (3.20%) 48 (2.40%) 592 (3.29%)
— Asian410 (2.05%) 32 (1.60%) 378 (2.10%)
— Other / Unknown / Missing3,370 (16.85%) 284 (14.20%) 3,086 (17.14%)
Insurance Categorization <0.001
— Medicare8,790 (43.95%) 1,142 (57.10%) 7,648 (42.49%)
— Private8,433 (42.17%) 638 (31.90%) 7,795 (43.31%)
— Medicaid1,969 (9.85%) 166 (8.30%) 1,803 (10.02%)
— Government600 (3.00%) 42 (2.10%) 558 (3.10%)
— Self Pay208 (1.04%) 12 (0.60%) 196 (1.09%)
Admission Classification <0.001
— Emergency13,172 (65.86%) 1,598 (79.90%) 11,574 (64.30%)
— Elective2,167 (10.84%) 214 (10.70%) 1,953 (10.85%)
— Newborn4,023 (20.12%) 146 (7.30%) 3,877 (21.54%)
— Urgent638 (3.19%) 42 (2.10%) 596 (3.31%)
Hospital Operational Context and Flow Dynamics
First Care Unit (FIRST_CAREUNIT) <0.001
— Medical ICU (MICU)6,252 (31.26%) 812 (40.60%) 5,440 (30.22%)
— Neonatal ICU (NICU)4,134 (20.67%) 148 (7.40%) 3,986 (22.14%)
— Cardiac Surgery Recovery (CSRU)3,003 (15.02%) 278 (13.90%) 2,725 (15.14%)
— Coronary Care Unit (CCU)2,690 (13.45%) 344 (17.20%) 2,346 (13.03%)
— Surgical ICU (SICU)2,217 (11.08%) 236 (11.80%) 1,981 (11.01%)
— Trauma Surgical ICU (TSICU)1,704 (8.52%) 182 (9.10%) 1,522 (8.46%)
Pre-ICU Ward Transfers (pre_icu_transfers)1.29±2.78(0.00)4.44±5.99(2.00)0.94±1.84(0.00)<0.001
Pre-ICU Surgical Interventions (surg_count)0.11±3.58(0.00)0.72±9.67(0.00)0.04±1.95(0.00)<0.001
Surgical Latency Gapln(1 + ∆t hours) 0.01±0.25(0.00)0.02±0.27(0.00)0.01±0.25(0.00)0.048
ED Dwell Timeln(1 +t ED, hours ) 0.86±0.99(0.00)1.10±1.02(1.18)0.83±0.99(0.00)<0.001
Unit-Adjusted Speedup LOS (icu_speedup_los, h)0.00±273.70(57.36)−18.24±233.14(41.20)2.03±277.76(58.80)<0.001
Log-Transformed Stay Lengthln(1 +LOS hours) 4.03±1.35(3.98)4.21±1.19(4.15)4.01±1.36(3.96)<0.001
Longitudinal Physiological Vital Signs (48-Hour Mean±SD across 1-Hour Bins)
Heart Rate (HR, bpm)90.83±31.70(86.12)84.34±26.38(82.70)91.55±32.16(86.60)<0.001
Respiratory Rate (RR, breaths/min)14.42±8.69(16.96)17.21±7.02(17.50)14.11±8.80(16.80)<0.001
Peripheral Oxygen Saturation (SpO2, %)73.41±40.19(95.29)86.39±28.75(96.10)71.97±41.01(95.10)<0.001
Systolic Blood Pressure (SBP, mmHg)90.53±52.07(109.54)106.53±39.11(112.40)88.75±53.01(109.10)<0.001
Diastolic Blood Pressure (DBP, mmHg)44.99±34.29(53.39)54.11±57.93(55.20)43.98±30.39(53.10)<0.001
Mean Arterial Pressure (MAP, mmHg)59.76±34.30(72.12)70.01±25.93(73.40)58.62±34.92(71.90)<0.001
Body Temperature (◦F)68.88±40.63(95.45)81.66±31.39(97.20)67.46±41.29(95.20)<0.001
Blood Glucose (mg/dL)98.91±65.66(112.25)116.67±55.81(122.10)96.94±66.38(110.80)<0.001
High-Dimensional Medical Codes Encoding
Primary ICD-9 Diagnosis Top-128 frequent codes mapped to learned dense embedding (d emb= 32) —
Primary DRG Code Top-128 frequent categories mapped to learned dense embedding (d emb= 32) —
Secondary Comorbidities Top-32 frequent ICD-9 diagnostic codes represented as a multi-hot vector (R32) —
Surgical Procedures Top-64 frequent ICD-9 surgical codes represented as a multi-hot vector (R64) —
Inpatient Prescriptions Top-64 frequent formulary medication codes represented as a multi-hot vector (R64) —

MIMIC-III v1.4 Database Initial ICU Stays
N= 61,532stays (46,520unique patients)
Hospital Admission Alignment (HADM_IDRecord Linkage)
Linked ICU hospital encounters (N= 61,532stays)
Endpoint & Competing Risk Calculation
30-Day Readmission Flag & In-Hospital Mortality Exclusion (N= 61,532stays)
Pre-specified Cohort Scale & Multi-Agent Throughput
Cohort Cap of20,000consecutive ICU stays (-limit 20000)Surplus Stays Set Aside
N= 41,532stays beyond20,000distillation
capacity
Designated Multimodal Study Cohort
N= 20,000ICU stays selected for full feature representation
Full Multimodal Feature Alignment:
•7-Category Note Distillation: via Llama-3.1 &Bio_ClinicalBERT(768-d embeddings)
•48h Longitudinal Vitals: 8 continuous channels (forward-filled)
•17 Static Operational Features: Demographics, transfers, surgical latency & ICU turnover
•High-Dimensional Clinical Codes: Multi-hot ICD-9, DRG, CPT procedures & prescriptions
Final Multimodal Analytical Cohort
N= 20,000ICU Stays(2,000Readmitted [10.00%],18,000Non-Readmitted)
Validation Split: 10%
N= 2,000staysTraining Split: 70%
N= 14,000staysTest Split: 20%
N= 4,000stays
Figure S3: End-to-end cohort derivation flowchart detailing MIMIC-III database linkage, endpoint indexing, cohort capacity
cap, multimodal feature alignment, and train/validation/test partitioning.
11
12### DOCTOR EXPERIENCE RULES TO FOLLOW:
13{rules_text}
14
15{few_shot_example}
16
17### PATIENT DOMAIN SUMMARIES:
18{combined_category_summaries}
19
20Output your analysis strictly in the
following JSON format:
21{
22"mortality_risk_score": <int 1-10, 10
highest>,
23"icu_readmission_risk_score": <int
1-10, 10 highest>,
24"critical_factors_list": ["factor 1",
"factor 2"],
25"final_summary": "<A concise 2-3
sentence synthesis>"
26}
27--------------------------------------
S3.3 Tier 3: Clinical Reflection Agent
When Meta-Agent severity scores (≥7vs.<7) contradict
ground-truthclinicalendpointsduringtheevolutionarytrain-
ing phase, the Reflection Agent audits diagnostic reasoning
gaps and adds a new heuristic rule to the repository:
Listing 3: Prompt template for Tier 3 Clinical Reflection
Agent.
1--------------------------------------2CLINICAL REFLECTION AGENT PROMPT
TEMPLATE
3--------------------------------------
4You are the Clinical Reflection Agent.
The Meta-Agent just
5made predictions about a patient’s
mortality and ICU
6readmission risk, but failed on one or
both tasks.
7
8ERRORS MADE:
9{error_context}
10
11Meta-Agent Output: {meta_summary_json}
12
13Domain Summaries it based its decision
on:
14{category_summaries}
15
16Analyze why the Meta-Agent failed
specifically on the
17mismatched task(s). Did it overweigh a
symptom for
18mortality? Did it ignore a resolved
condition for
19readmission?
20Provide a single, concise RULE (1-2
sentences) starting
21with either "[Mortality Rule]:" or "[
Readmission Rule]:"
22that should be added to the experience
base.
23DO NOT provide conversational text.
Output ONLY the rule.

24--------------------------------------
D Self-Evolving RAG Knowledge Bases
S4.1 Doctor Experience Rules Base (3,085 Rules)
Overtheiterativetrainingloopon14,000trainingstays,the
reflection agent accumulated3,085 rules:
•Mortality Rules(N= 2,951): Rules differentiating
acutereversibleorgandecompensationfromcatastrophic
multi-system failure.
•Readmission Rules(N= 2,773): Rules capturing pre-
mature discharge under ward saturation, incomplete di-
uretic titration, post-extubation subglottic stridor, and
telemetry instability.
Table S3 provides representative rules categorized across 7
core physiological domains.
S4.2 Golden Case Base (1,899 Cases) and Case
Walkthrough
Aclinicalstayisarchivedintocase_base.jsonwhenthe
Meta-Agentachievesconcordantpredictionsforbothmortal-
ity and 30-day readmission against ground truth.
End-to-End Case Walkthrough (Stay ID: 232669)
•Demographics: 68-year-old male, emergency CCU ad-
mission, 3 pre-ICU transfers.
•Ground Truth: In-hospital Mortality = 1 (Deceased),
30-day Readmission = 0 (Mutual exclusion).
•Raw Domain Summaries:
–Nursing/other:“Agitated,receivingDexmedetomidine
infusion. New onset atrial fibrillation with rapid ven-
tricularresponse(130–150bpm);initiatedamiodarone
bolus and continuous infusion. Neurological status
fluctuating from confused to obtunded.”
–ECG Summary: “Atrial fibrillation with rapid ventric-
ularresponse.STsegmentdepressioninleadsV4–V6.
Non-specific intraventricular conduction delay.”
•Retrieved RAG Rules:
–Rule 1:[Mortality Rule]:If acute severe neurologi-
caldeteriorationcoexistswithrefractorytachyarrhyth-
mias, prioritize mortality risk.
–Rule 2:[Readmission Rule]:Rate readmission risk
moderate if patient remains on active continuous an-
tiarrhythmic infusions.
•Synthesized JSON Report:
1{
2"mortality_risk_score": 8,
3"icu_readmission_risk_score": 6,
4"critical_factors_list": [
5"Atrial fibrillation with RVR
escalating stroke and acute
decompensation",
6"Severe encephalopathy and
fluctuating neurological
responsiveness"
7],8"final_summary": "High risk of
terminal in-hospital deterioration
due to severe neurological
depression and refractory
tachyarrhythmias. Readmission risk
is moderate conditioned on survival
."
9}
•Embedding Extraction: The text summary and vi-
tal factor list are concatenated and mapped via
Bio_ClinicalBERTtoa768-dimensionalembedding
Enote.
E Hyperparameters and Experimental Setup
Table S4 outlines all hyperparameter choices, optimization
parameters, and model specifications across architectures.
F Multi-Run Benchmark Results and
Robustness
To eliminate seed selection bias, all models were evaluated
over 20 independent randomized runs (Seeds 42–61).
S6.1 Performance Evaluated at Optimal
Threshold (t∗)
Table S5 reports test performance evaluated at the optimal
validation-selectedF 1cutoff (t∗, grid step∆ = 0.01).
S6.2 Performance Evaluated at Standard Fixed
Threshold (t= 0.50)
Table S6 illustrates model performance under the uncali-
brated standard decision thresholdt= 0.50. Under class
imbalance (∼10%), the uncalibrated cutoff causes severe
precision degradation in deep models trained with positive
weighting.
S6.3 Sensor Dropout Robustness
To simulate telemetry loss in acute care (e.g., electrode dis-
placement,patienttransport),TableS7documentsinference
robustnessunderrandomvitalchanneldropout(0%to30%).
Table S7: Inference performance under simulated sensor
dropout inLSTMLateFusion(20 independent runs).
Dropout Rate Thresholdt∗AUROC PR-AUC F1-Score
0% (Baseline)0.726±0.034 0.8258±0.0091 0.4338±0.0301 0.4791±0.0193
10% Dropout0.726±0.032 0.8259±0.0092 0.4337±0.0300 0.4794±0.0192
20% Dropout0.722±0.034 0.8255±0.0092 0.4332±0.0302 0.4793±0.0191
30% Dropout0.725±0.037 0.8254±0.0092 0.4330±0.0300 0.4783±0.0189
G Feature Importance, Attribution, and
Temporal Dynamics
S7.1 Fine-Grained Physiological Channel
Attribution
IntegratedGradients(IG)attributionacross48timestepsfor
the 8 continuous vital channels ranks as follows:
1.Heart Rate (HR):0.3205(32.05%attribution share)
2.Body Temperature (TEMP):0.2219(22.19%)

Table S3: Representative physician-vetted heuristic experience rules distilled by the Self-Evolving Reflection Agent.
Clinical Domain Risk Task Rule Specification
CardiovascularReadmission[Readmission Rule]:In post-CABG or post-valve surgery patients, the emergence of new-onset atrial
fibrillation with rapid ventricular response represents an acute readmission hazard even if pharmacological or
electrical cardioversion achieved transient sinus rhythm.
CardiovascularMortality[Mortality Rule]:Inacutemyocardialinfarctioncomplicatedbycardiogenicshock,persistentrequirement
for two or more concurrent vasoactive infusions (e.g., Norepinephrine+Epinephrine) indicates extreme in-
hospital mortality regardless of baseline LVEF.
RespiratoryReadmission[Readmission Rule]:If a patient was extubated within 24 hours prior to ICU discharge but presents
residual tachypnea (RR>24) or requires escalating high-flow nasal cannula/BiPAP requirements, assign an
elevated readmission risk.
RespiratoryMortality[Mortality Rule]:If a patient treated for acute respiratory distress exhibits documented bilateral radio-
graphic clearance and stable oxygenation on minimal supplemental oxygen (FiO 2<35%), do not score high
mortality risk solely based on initial intubation status.
NeurologyMortality[Mortality Rule]:Whenphysicalexaminationsatisfiesbraindeathcriteriaorpresentsprofoundhypoxic-
ischemic encephalopathy with absent brainstem reflexes, assign maximum mortality risk regardless of hemody-
namic stability.
NeurologyReadmission[Readmission Rule]:Residual delirium, agitation, or confusion without acute focal neurological deficits
increasesnursingcarefrictionbutdoesnotwarrantelevatedICUreadmissionriskunlesscomplicatedbyimminent
airway compromise.
Infectious / SepsisReadmission[Readmission Rule]:Septic patients completing antibiotic therapy with normalized WBC/procalcitonin
but persistent unexplained fever spikes must not be discharged without negative repeat blood cultures; elevate
readmission risk.
Renal / FluidsReadmission[Readmission Rule]:Acute kidney injury resolving with spontaneous diuresis can only be designated as
low readmission risk if serum creatinine has stabilized or trended downward for at least 48 consecutive hours
prior to transfer.
Discharge FlowReadmission[Readmission Rule]:An elevated ICU readmission risk should be downgraded if pending laboratory
abnormalitieshavebeenformallydocumentedasclinicallyresolvedandstablepriortosigningfinalwardtransfer
orders.
Goals of CareReadmission[Readmission Rule]:Do not predict elevated ICU readmission risk if remaining clinical complexity is
driven by transitions to palliative comfort care, family DNR/DNI execution, or hospice placement rather than
acute organ failure.
3.Blood Glucose (GLUCOSE):0.1366(13.66%)
4.Systolic Blood Pressure (SBP):0.1252(12.52%)
5.Oxygen Saturation (SpO2):0.1025(10.25%)
6.Mean Arterial Pressure (MAP):0.0433(4.33%)
7.Respiratory Rate (RR):0.0323(3.23%)
8.Diastolic Blood Pressure (DBP):0.0176(1.76%)
S7.2 Input Modality Leave-One-Out Ablation
Against the test cohort baseline predicted probability (¯p=
0.4041), leave-one-modality-out ablation yields: (1)Static
Covariates & Operations:|∆prob|= 0.1437(primary
baseline anchor); (2)48h Longitudinal Vitals:|∆prob|=
0.1014(acute physiological trajectory); (3)Medical Cod-
ings(ICD/DRG/Rx):|∆prob|= 0.0569(comorbidchronic
burden); and (4)Clinical Notes Embedding:|∆prob|=
0.0085(residual semantic risk tuning).
S7.3 Temporal Cross-Attention Dynamics
Transformer self-attention between the[CLS]token and
the 48 hourly vital sign vectors partitions into three clin-
ically meaningful phases: (1)Acute Resuscitation Phase
(Hours 0–12):37.9%of total attention, reflecting initial
resuscitation response, drug loading, and mechanical ven-
tilation establishment; (2)Continuous Monitoring Phase(Hours 12–36):42.3%, capturing intermediate clinical sta-
bility and multi-organ recovery; and (3)Pre-Discharge Ta-
peringPhase(Hours36–48):19.7%,evaluatingextubation
tolerance, pressor weaning, and ward readiness. Table S8
provides the Top-10 most attended individual hours.
Table S8: Top-10 most attended hourly time steps in Trans-
former attention.
Rank Time Step Attention Weight & Clinical Interpretation
1 Hour 00.0806— Baseline admission acuity and shock index
2 Hour 10.0352— Immediate post-admission fluid/pressor response
3 Hour 20.0199— Early stabilization response window
4 Hour 30.0165— Acute stabilization plateau
5 Hour 50.0149— Mid-acute surveillance checks
6 Hour 40.0147— Trajectory confirmation
7 Hour 60.0144— Shift handover and laboratory re-check
8 Hour 80.0138— Intermediate resuscitation evaluation
9 Hour 70.0137— Morning clinical review
10 Hour 210.0136— Day-1 nocturnal stability check
S7.4 Static Feature Ablation Ranking
Table S9 displays the leave-one-feature-out ablation impact
across all 17 static operational and demographic covariates.

Table S4: Hyperparameter configurations across model families and optimization tiers.
Component / Module Hyperparameter Name Value / Specification Rationale / Design Justification
Cohort SplitsTotal Cohort Size20,000aligned ICU stays Filtered MIMIC-III adult stays with 48h data
Split Proportion 70% Train / 10% Val / 20% Test 14,000 / 2,000 / 4,000 sample partitioning
Randomization Seeds 20 runs (Seeds 42–61) Eliminates split-specific bias in evaluation
Loss Re-weightingw pos= 8.97Dynamically matches negative-to-positive ratio in train split
Training OptimizationMaximum Epochs200epochs Early stopping with patience= 15on validation loss
Mini-Batch Size256(End-to-End:64) Balances throughput with gradient signal stability
Optimization Algorithm AdamW (weight decay= 10−4) Decoupled weight decay regularization
Base Learning Rateη5×10−4Tuned for smooth multimodal cross-entropy convergence
Learning Rate ScheduleCosineAnnealingLRSmooth decay toη min= 5×10−6
Normalization ProtocolStandardScaler(Train-fit strictly) Prevents temporal distribution leakage into validation/test
LSTM ArchitectureHidden State DimensionH128(64forward+ 64backward) Sufficient capacity for 48h temporal abstractions
Recurrent Layer Depth4bidirectional layers Deep feature extraction over physiological trajectories
Dropout Regularization0.30across recurrent states Prevents co-adaptation of hidden states
Attention Pooling Dimension128Normalized scalar projection of hidden states
Transformer ArchitectureLatent Dimensiond128Unified projection dimensionality across modalities
Multi-Head Attention Heads4heads (head dimension= 32) Multi-view temporal and tabular interaction
Transformer Layer Depth2encoder layers Avoids over-parameterization on tabular EHR sequences
Feedforward Hidden Dimension512(4×d) Standard transformer non-linear expansion
Layer Normalization Pre-LN (norm_first=True) Stable gradient propagation without warmup sensitivity
Bio_ClinicalBERT E2EPretrained Backboneemilyalsentzer/Bio_ClinicalBERT768-d domain prior pretrained on MIMIC notes
Maximum Token Length512WordPiece tokens Maximum sequence capacity for clinical report summary
Differential LR Scaling0.1×(η BERT= 5×10−5) Preserves pre-trained weights while adapting to readmission
Layer Freezing Strategy Layers 1–8 Frozen; Layers 9–12 Active Retains generic syntax; adapts deep semantic representations
Gradient Checkpointing Enabled Reduces activation memory footprint by 65%
Tree BaselinesEstimators / Maximum Depth200trees / Maximum depth6XGBoost and LightGBM configurations
Subsample / Colsample By Tree0.80row subsample /0.80feature subsample Tree-level feature and data bagging
Positive Class Balancingscale_pos_weight = 8.97Aligns gradient tree splitting with binary loss reweighting
TableS5:Testsetperformanceevaluatedattheoptimalvalidation-calibratedthresholdt∗across20independentruns(Mean±
SD).
Model Architecture Cutofft∗AUROC PR-AUC Precision Recall F1-Score Brier Score
Tree Baselines and Shallow Text Concatenation
XGBoost Base0.314±0.093 0.8055±0.0098 0.4125±0.0209 0.4372±0.0401 0.4645±0.0501 0.4465±0.0193 0.0818±0.0036
XGBoost + LLM Notes0.134±0.053 0.7914±0.0118 0.3893±0.0213 0.4239±0.0259 0.4616±0.0415 0.4400±0.0166 0.0819±0.0038
LightGBM Base0.463±0.067 0.8158±0.0111 0.4114±0.0187 0.4092±0.0404 0.5215±0.0438 0.4555±0.0172 0.0910±0.0038
LightGBM + LLM Notes0.370±0.064 0.8087±0.0114 0.4030±0.0193 0.4344±0.0385 0.4828±0.0376 0.4548±0.0170 0.0804±0.0029
Deep Multimodal Neural Networks
LSTM Base (Structured Only)0.747±0.051 0.8251±0.0117 0.4360±0.0305 0.4714±0.0485 0.4964±0.04500.4798±0.02000.1619±0.0123
LSTM LateFusion (+ Notes)0.726±0.034 0.8258±0.0091 0.4338±0.0301 0.4536±0.0255 0.5099±0.0337 0.4791±0.0193 0.1588±0.0106
Transformer EarlyFusion (+ Notes)0.757±0.068 0.8177±0.0145 0.4278±0.0332 0.4636±0.0277 0.4939±0.0362 0.4769±0.0200 0.1561±0.0107
Transformer LateFusion (+ Notes)0.762±0.042 0.8277±0.0104 0.4357±0.0320 0.4657±0.0393 0.4918±0.0442 0.4751±0.0158 0.1635±0.0092
LSTM E2E FineTune (+ Notes)0.733±0.044 0.8260±0.0119 0.4338±0.0326 0.4564±0.0365 0.4998±0.0283 0.4754±0.0178 0.1622±0.0128
Transformer E2E FineTune (+ Notes)0.754±0.0490.8288±0.0110 0.4373±0.03100.4577±0.0279 0.4989±0.0418 0.4757±0.0196 0.1658±0.0140
Ablation Variants (Relative to LSTM Base)
w/o ICU Operational Features0.734±0.044 0.8275±0.0111 0.4278±0.0312 0.4570±0.0380 0.5091±0.0431 0.4791±0.0195 0.1602±0.0093
w/o Admission Flow Dynamics0.750±0.043 0.8260±0.0103 0.4292±0.0287 0.4618±0.0359 0.4985±0.0401 0.4772±0.0204 0.1650±0.0102
w/o Medical Codes (x code)0.741±0.033 0.8243±0.0107 0.4334±0.0302 0.4574±0.0322 0.4985±0.0316 0.4757±0.0187 0.1633±0.0113
Only Vitals (X seq)0.521±0.033 0.6232±0.0149 0.1387±0.0086 0.1358±0.0107 0.6496±0.1569 0.2213±0.0099 0.2396±0.0096
Only Static Covariates & Codes0.735±0.040 0.8227±0.0097 0.4294±0.0273 0.4578±0.0272 0.4969±0.0397 0.4749±0.0180 0.1612±0.0107
Table S6: Test set performance evaluated at the standard fixed thresholdt= 0.50across 20 independent runs (Mean±SD).
Model Architecture AUROC PR-AUC Precision Recall F1-Score Accuracy
XGBoost Base0.8055±0.0098 0.4125±0.0209 0.4951±0.0299 0.3646±0.0173 0.4196±0.0181 0.8992±0.0052
XGBoost + LLM Notes0.7914±0.0118 0.3893±0.02130.5359±0.02220.2986±0.0219 0.3830±0.01950.9040±0.0047
LightGBM Base0.8158±0.0111 0.4114±0.0187 0.4168±0.0240 0.4865±0.0200 0.4486±0.0182 0.8805±0.0057
LightGBM + LLM Notes0.8087±0.0114 0.4030±0.0193 0.4839±0.0230 0.3929±0.0229 0.4332±0.0188 0.8974±0.0048
LSTM Base0.8251±0.0117 0.4360±0.0305 0.2605±0.0233 0.7157±0.0272 0.3811±0.0244 0.7663±0.0262
LSTM LateFusion (+ Notes)0.8258±0.0091 0.4338±0.0301 0.2705±0.0215 0.7092±0.0314 0.3908±0.0214 0.7782±0.0226
Transformer EarlyFusion (+ Notes)0.8177±0.0145 0.4278±0.0332 0.2822±0.0361 0.6786±0.05090.3959±0.03160.7903±0.0364
Transformer LateFusion (+ Notes)0.8277±0.0104 0.4357±0.0320 0.2597±0.0181 0.7260±0.0297 0.3821±0.0192 0.7648±0.0191
LSTM E2E FineTune (+ Notes)0.8260±0.0119 0.4338±0.0326 0.2646±0.0211 0.7116±0.0465 0.3846±0.0209 0.7714±0.0272
Transformer E2E FineTune (+ Notes)0.8288±0.0110 0.4373±0.03100.2591±0.02510.7342±0.04150.3816±0.0240 0.7604±0.0322
Only Vitals (X seq)0.6232±0.0149 0.1387±0.0086 0.1246±0.0073 0.8046±0.1038 0.2151±0.0093 0.4132±0.0750

Table S9: Ranking of static features by ablation impact
(|∆prob|).
Rk Feature|∆p|Share Clinical Context
1pre_icu_transfers0.0926 28.2%Number of inter-ward handoffs
2primary_icd0.0542 16.5%Primary admission diagnosis code
3icu_speedup_los0.0315 9.6%Departmental turnover pressure
4drg_code0.0302 9.2%DRG casemix category
5surg_count0.0226 6.9%Pre-ICU surgical procedures
6ADMISSION_LOC0.0215 6.5%Transfer source (e.g., ED, Clinic)
7FIRST_CAREUNIT0.0181 5.5%Specialty ICU unit designation
8age0.0134 4.1%Patient chronological age
9INSURANCE0.0119 3.6%Primary healthcare payer class
10ETHNICITY0.0094 2.9%Demographic profile
11log_icu_los0.0075 2.3%Total log ICU stay duration
12GENDER0.0044 1.3%Biological sex
13MARITAL_STATUS0.0039 1.2%Social support indicator
14ADMISSION_TYPE0.0025 0.8%Emergency vs. Elective status
15log_ed_wait0.0023 0.7%Pre-ICU ED boarding dwell time
16log_surg_gap0.0020 0.6%Post-operative latency to ICU
17surg_flag0.0008 0.2%Binary surgical intervention