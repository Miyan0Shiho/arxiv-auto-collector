# Knowledge-Graph-Guided Retrieval-Augmented LLMs for Explainable Root Cause Analysis in Automotive HiL Validation

**Authors**: Hamza Ouarrad, Mohammad Abboush, Andreas Rausch

**Published**: 2026-08-11 10:19:33

**PDF URL**: [https://arxiv.org/pdf/2608.11277v1](https://arxiv.org/pdf/2608.11277v1)

## Abstract
Hardware-in-the-Loop validation of automotive software systems generates large multivariate time-series recordings whose manual analysis is time-consuming and often limited to anomaly detection and fault classification rather than root-cause analysis. Although deep learning methods have shown strong performance in fault detection and classification, they usually require task-specific training or retraining when new fault locations, systems, or operating conditions are introduced. They also tend to treat localization as a classification task, without explicitly representing the spatial and functional relationships between fault locations, sensors, and downstream subsystem effects. This limits their generalizability and their usefulness for engineering root cause analysis and diagnosis. This paper proposes a knowledge-graph-guided retrieval-augmented large language model framework for RCA (root cause analysis) and fault localization in automotive HiL data. The method converts raw time-series recordings into compact diagnostic evidence, enriches this evidence with sensor-to-location and propagation knowledge, and retrieves similar historical cases to support the final reasoning step. The LLM is then used as a decision and explanation layer rather than as a direct time-series classifier, producing a ranked fault-location prediction together with an interpretable RCA explanation. The framework is evaluated on two automotive HiL case studies: an ASM gasoline engine and an electric vehicle system. The best-performing model achieves Top-1 accuracies of 90\% and 94\%, respectively, while recording-level aggregation reaches perfect file-level fault localization in the evaluated subset. These results demonstrate the potential of KG-guided RAG-LLM reasoning for explainable and generalizable HiL RCA.

## Full Text


<!-- PDF content starts -->

Knowledge-Graph-Guided Retrieval-Augmented
LLMs for Explainable Root Cause Analysis in
Automotive HiL Validation
1stHamza Ouarrad*
Technische Universit ¨at Clausthal
Clausthal-Zellerfeld, Germany
hamza.ouarrad@tu-clausthal.de
*Corresponding author2ndMohammad Abboush
Technische Universit ¨at Clausthal
Clausthal-Zellerfeld, Germany
mohammad.abboush@tu-clausthal.de3rdAndreas Rausch
Technische Universit ¨at Clausthal
Clausthal-Zellerfeld, Germany
andreas.rausch@tu-clausthal.de
Abstract—Hardware-in-the-Loop validation of automotive soft-
ware systems generates large multivariate time-series recordings
whose manual analysis is time-consuming and often limited
to anomaly detection and fault classification rather than root-
cause analysis. Although deep learning methods have shown
strong performance in fault detection and classification, they
usually require task-specific training or retraining when new fault
locations, systems, or operating conditions are introduced. They
also tend to treat localization as a classification task, without
explicitly representing the spatial and functional relationships
between fault locations, sensors, and downstream subsystem
effects. This limits their generalizability and their usefulness
for engineering root cause analysis and diagnosis. This paper
proposes a knowledge-graph-guided retrieval-augmented large
language model framework for RCA (root cause analysis) and
fault localization in automotive HiL data. The method converts
raw time-series recordings into compact diagnostic evidence,
enriches this evidence with sensor-to-location and propagation
knowledge, and retrieves similar historical cases to support the
final reasoning step. The LLM is then used as a decision and
explanation layer rather than as a direct time-series classifier,
producing a ranked fault-location prediction together with an
interpretable RCA explanation. The framework is evaluated on
two automotive HiL case studies: an ASM gasoline engine and
an electric vehicle system. The best-performing model achieves
Top-1 accuracies of 90% and 94%, respectively, while recording-
level aggregation reaches perfect file-level fault localization in the
evaluated subset. These results demonstrate the potential of KG-
guided RAG-LLM reasoning for explainable and generalizable
HiL RCA.
Index Terms—Hardware-in-the-Loop, fault localization, root-
cause analysis, large language models, retrieval-augmented gen-
eration, knowledge graph, automotive software validation, time-
series diagnosis
I. INTRODUCTION
In the automotive domain, the development of safety-critical
software systems is governed by the ISO 26262 functional
safety standard [1], which prescribes a rigorous verification
and validation process across the V-model life cycle, collec-
tively referred to as X-in-the-Loop (XiL) testing [2]. In the
concluding validation phase preceding series production, real
test drives on public roads, complemented by hardware-in-the-
loop (HIL) simulations integrating actual electronic controlunits (ECUs) and physical vehicle components, are conducted
to validate the implemented system against its functional safety
requirements under representative operating conditions [3],
[4]. A direct consequence of modern HIL validation campaigns
is the generation of large-volume test recordings that manifest
as multivariate time series data with nonlinear dynamics and
multi-mode behaviour. Industry-certified tools predominantly
rely on rule-based analysis and predefined thresholds, which
are effective for flagging anomalies but provide little support
for identifying the actual location and root cause of a fault. In
practice, an anomaly observed in a single signal may originate
from sensors, communication channels, software components,
controllers, or actuators [5], and the corresponding fault effects
often propagate through interconnected components before be-
coming visible in the recordings [6]. Consequently, validation
engineers must manually inspect large numbers of signals
and combine numerical observations with prior experience,
historical fault cases, component knowledge, sensor dependen-
cies, and system architecture information. There is therefore
a pressing need for intelligent diagnostic frameworks that
integrate data-driven evidence with engineering knowledge to
support explainable and efficient root cause analysis in HIL
validation.
Within the literature, signal-based [7], knowledge-based
[8], and model-based [9] approaches have been proposed
for the analysis of test recordings, yet each suffers from
well-known limitations related to noise sensitivity, manual
engineering effort, and dependence on reference model fidelity.
More recently, machine learning (ML) and deep learning (DL)
methods have demonstrated remarkable performance in fault
detection and diagnosis (FDD) by automatically learning dis-
criminative representations from data [10], [11]. Nevertheless,
irrespective of the underlying paradigm, existing approaches
generally answer only whether and what type of anomaly has
occurred, and do not address the more critical question faced
by validation engineers: where the fault is located and why a
given location constitutes the most probable root cause [6].
Recent advances in Large Language Models (LLMs) have
opened new opportunities for knowledge-intensive reason-
arXiv:2608.11277v1  [cs.CR]  11 Aug 2026

ing in engineering diagnosis [12], [13], while Retrieval-
Augmented Generation (RAG) techniques [14] have been suc-
cessfully employed for industrial troubleshooting and knowl-
edge retrieval [15], [16]. However, most existing LLM-based
diagnostic approaches treat the language model as an end-
to-end classifier and do not incorporate structured diagnostic
evidence or fault localization knowledge into the reasoning
process [17]. Furthermore, the literature lacks knowledge-
guided retrieval-based frameworks tailored to fault localization
in automotive HIL environments: existing methods rarely
integrate numerical time-series evidence, historical fault cases,
sensor-to-component mappings, and propagation relationships
within a unified reasoning framework [18], and most retrieval-
based LLM systems rely on semantic embeddings rather
than engineered diagnostic evidence derived from multivariate
time-series behaviour [14]. In addition, LLM-based diagnos-
tic systems remain susceptible to hallucinations and model-
specific biases, which limits their direct deployment in safety-
critical contexts without additional grounding mechanisms
[18], [19]. Consequently, a significant research gap exists at the
intersection of automotive HIL validation, retrieval-augmented
reasoning, case-based diagnosis, structured knowledge integra-
tion, and LLM-assisted fault localization.
To address this gap, the present article proposes a novel
knowledge-guided retrieval-augmented LLM framework for
root cause analysis and fault localization in the real-time HIL
validation of automotive software systems. The framework
transforms raw multivariate HIL recordings into structured
diagnostic evidence, including abnormality scores, signal re-
action times, deviation directions, and inter-signal correla-
tion changes. These engineered features drive a lightweight
case-based retrieval mechanism, in contrast to conventional
embedding-based retrieval [14], [17], and are combined with
structured domain knowledge describing sensor roles, compo-
nent relationships, and fault propagation paths [15]. Unlike
conventional LLM-based classification approaches, the LLM
acts as an evidence-fusion, reasoning, and reranking layer
rather than as a direct classifier: it receives the structured
evidence, retrieved cases, domain knowledge, and candidate
fault locations, and produces a ranked list of the most probable
root causes accompanied by interpretable engineering explana-
tions, supporting both single and combined fault hypotheses.
To the best of the authors’ knowledge, this is the first study
that unifies automotive HIL recordings, structured diagnostic
evidence, case-based retrieval, knowledge-guided diagnosis,
root cause analysis, and Retrieval-Augmented Large Language
Models within a single framework. The principal contributions
of this work are summarised as follows:
•A knowledge-guided retrieval-augmented LLM frame-
work is proposed for fault localization and root cause
analysis in automotive HIL validation.
•A diagnostic evidence extraction module converts raw
HIL recordings into interpretable features, including ab-
normality scores, reaction times, deviation directions, and
correlation changes.•A lightweight case-based retrieval approach combined
with domain knowledge captures sensor-location relation-
ships and fault propagation patterns.
•An LLM-based reasoning layer integrates evidence, re-
trieved cases, and engineering knowledge to generate
ranked fault-location hypotheses with root-cause expla-
nations.
The remainder of this article is organised as follows. Section 2
reviews the related literature. Section 3 presents the proposed
approach. Section 4 describes the implementation structure and
case study. Section 5 reports and discusses the experimental
results. Section 6 concludes the article and outlines directions
for future research.
II. RELATED WORK
A. Deep Learning-Based Fault Detection and Diagnosis in
Automotive Systems
Data-driven FDD in the automotive domain has progressed
from shallow classifiers anchored in hand-engineered features
towards deep architectures that learn discriminative tempo-
ral representations directly from raw multivariate signals.
Convolutional–dense pipelines have been applied to sensor-
fault detection, isolation and prognostic health-index fore-
casting in autonomous-driving stacks, with Safavi et al. [20]
reporting 99.84% detection accuracy on the Audi A2D2
dataset but restricting evaluation to four canonical fault classes
injected via a static normal-distribution model. Recurrent
architectures have dominated diagnostics in electrified pow-
ertrains, exemplified by the LSTM model of Kaplan et al.
[21], which reduced MAPE from 10.13% to 2.06% over a
shallow ANN baseline, although fault injection was performed
at the Simulink level rather than under real-time constraints.
Ensemble and hybrid formulations have broadened coverage
and robustness: a two-class/one-class ensemble attaining an
F2-score of 77% on OBD-II road-trial recordings under known
and unseen faults [22]; a multi-label LSTM–Random Forest
ensemble augmented by a GRU-based denoising autoencoder
reaching 99.43% accuracy and 91.2% F1 on concurrent sensor
faults under HIL-generated noise [23]; and a Random Forest
classifier coupled with sliding-mode fault-tolerant control on a
TruckMaker-based HIL platform for heavy-duty brakes [24];
complementary reviews further note that many algorithms
still simplify actuator dynamics and neglect parameter drift,
limiting transferability to production conditions [25]. Collec-
tively, these contributions advance detection and classification
accuracy but answer only whether and what type of anomaly
has occurred, providing no explicit support for identifying
where a fault is located or why a particular location is the most
probable root cause — a gap that has motivated two parallel
responses in the literature: knowledge-augmented language
models and dedicated root-cause-diagnosis methods.
B. Knowledge Graph Enhanced Large Language Models for
Fault Diagnosis
The integration of Knowledge Graphs (KGs) with Large
Language Models (LLMs) has emerged as a leading paradigm

for grounding LLM reasoning in structured engineering knowl-
edge and mitigating hallucinations in industrial fault diagnosis.
A first line treats the KG as an external memory queried by an
LLM agent: To-FD-EKG [26] couples a table-sequence triplet
extractor with an LLM that interacts with a virtual digital-twin
environment through perception–action loops, reaching 82.3%
average cosine similarity on wind-turbine maintenance logs,
whilst LC-CoT [27] extends the paradigm to compound faults
by combining causal hypergraphs with reinforcement-learning
fine-tuning. A second line embeds graph structure directly into
the LLM through prefix-tuning and subgraph generation, illus-
trated by the joint KG–LLM aviation-assembly system of [28],
which reaches 98.5% accuracy on industrial localisation cases,
and by joint-extraction frameworks raising entity-relation F1
from 0.907 to 0.968 [29].
A third and increasingly dominant line employs Retrieval-
Augmented Generation over a domain KG. CNC-focused sys-
tems consolidate diagnostic cases, work orders and equipment
data into KGs queried by fine-tuned LLMs, outperforming
experienced engineers on benchmark tasks [30]; wind-turbine
pipelines combine DeepSeek-R1 with dual-channel graph and
vector retrieval and Graph Attention Networks for multi-
hop reasoning [31]; LLM-FDQA reports 96.1% Hit@1 entity
linking and 94.04% end-to-end diagnostic accuracy on real-
world CNC data through explicit evidence-chain reasoning
[32]; Graph-RAG for train-bogie diagnosis achieves 92.67%
attribution consistency through hybrid sparse–dense–graph re-
trieval [33]; and dynamic multimodal frameworks fuse vi-
bration spectrograms with text KGs through DeepSeek-V3,
sustaining 85.3% accuracy at−5dB SNR [34]. Causal exten-
sions for automotive chip production further couple knowledge
graphs with text-grounded causal event graphs to support long-
chain reasoning [35]. Despite this momentum, three limitations
recur. Validation is overwhelmingly concentrated in process
industries—wind turbines, CNC machinery, aviation and rail—
with virtually no exposure to automotive HIL recordings.
Retrieval remains predominantly embedding-based and op-
erates on unstructured maintenance logs, leaving the use of
engineered diagnostic features derived from multivariate time-
series behaviour largely unexplored. And the LLM is most
often deployed as an end-to-end classifier or generator rather
than as an evidence-fusion and reranking layer grounded in
sensor-to-component relationships and fault-propagation paths.
C. Root Cause Diagnosis of Industrial Faults
Root cause diagnosis (RCD) of industrial faults has been
pursued along three complementary directions. The first relies
on causal discovery over multivariate process recordings:
CASCN [36] combines LASSO-based fault isolation with a
channel-attentive sparse causal TCN, achieving on the Ten-
nessee Eastman process (TEP) a 23.10% improvement in true-
positive rate and a 60.82% reduction in false-positive rate over
mainstream baselines; REASON [37] integrates hierarchical
graph neural networks for topological causal discovery with
Extreme Value Theory for individual-entity scoring; MPRA
combines Lasso–Granger single-level causality with attention-based high-level similarity for grouped RCD [38]; and ex-
tended convergent cross mapping with redundancy pruning
[39], reinforcement-learning structure learning [40], and im-
proved symbolic transfer entropy [41] extend causal discovery
to aero-engine, chemical and electromechanical processes.
A second direction couples anomaly detection with post-
hoc attribution to recover causes from effects. Deep Root
Cause Analysis [42] pairs hierarchical prediction networks
with saliency maps and explicitly demonstrates on TEP that
high-deviation variables often correspond to symptoms rather
than direct causes; LSTM-A V AGMM with Kernel SHAP
supports anomaly explanation and incremental adaptation in
wind-turbine SCADA streams [43]; V AE–RCAP–SHAP sep-
arates internal-sensor anomaly manifestation from external
operational drivers on container-ship engines [44]; physics-
guided graphs fuse explicit knowledge rules with multivariate
GNNs for backtracking root causes across wind farms, coal
pulverisers and sugar factories [45]; and a data-space attention-
CNN with LSTM–OLS captures causal propagation paths in
float-glass and TEP [46]. A third, knowledge-centric direction
integrates KGs and case-based reasoning with anomaly de-
tection, exemplified by an informed self-supervised one-class
learner coupled with FMEA-grounded SPARQL, symbolic-
driven neural reasoning and case-based retrieval for smart-
factory predictive maintenance [47]. Within the automotive
domain, evidence is limited: a causal Seq2Seq with attention
[48] addresses fault identification and basic causal relations on
UCI/Ford GoBike/Nissan datasets, but does not localise faults
to physical components under HIL constraints.
Across these contributions, three challenges remain un-
resolved. Causal-discovery methods deliver structural trans-
parency but depend on offline training over substantial fault
corpora and offer limited grounding in engineering knowl-
edge such as sensor-to-component mappings and propagation
paths. Attribution-based methods supply local explanations
yet frequently confuse symptoms with causes and rarely
support multi-fault hypotheses. Knowledge-graph and case-
based approaches incorporate expert structure but have not
been integrated with retrieval-augmented LLM reasoning over
engineered time-series evidence, and automotive HIL valida-
tion under safety-critical concurrent-fault conditions remains
absent altogether. These observations motivate the present
investigation, which unifies structured diagnostic evidence
extracted from HIL recordings, lightweight case-based re-
trieval over engineered features, structured sensor–component–
propagation knowledge, and an LLM reasoning-and-reranking
layer into a single framework for fault localisation and root
cause analysis in automotive HIL validation.
III. METHODOLOGY
The proposed method is designed for root-cause diagnosis in
multivariate temporal HiL time-series data. Given a recording
X∈RT×d, whereTdenotes the number of time steps andd
the number of sensor signals, the objective is to identify the
most likely fault location from a predefined set of candidates.
Fig. 1 summarizes the complete KG-guided RAG pipeline. The

method first converts raw recordings into compact diagnostic
evidence, then combines knowledge-graph-based candidate
reasoning with retrieval of similar historical cases, and finally
uses an LLM to produce a ranked fault-location prediction
with an explanatory root-cause diagnosis.
A. Preprocessing and Healthy-Reference Evidence Extraction
The raw HiL recordings are first converted into fault-
relevant diagnostic evidence. Each experimental run is loaded
as a multivariate CSV time series and sorted according to the
time column. Non-signal fields, such as the time stamp and
optional status labels, are excluded from the diagnostic feature
set. Since the measurements are generated under closed-
loop driving conditions, absolute signal values alone are not
sufficient for fault localization. The same signal may vary
naturally because of the driving profile, controller response, or
vehicle dynamics. Therefore, each faulty recording is evaluated
relative to a corresponding healthy reference run.
The analysis focuses on the fault-active region of the
experiment. A pre-fault interval is used as baseline behavior,
while the stable fault interval is used to describe the abnormal
response. In this work, the baseline window is defined as 2–
11 s and the stable fault interval as 15–45 s. For window-wise
diagnosis, the stable fault interval is further segmented into
fixed-length temporal windows. This preserves the temporal
structure of the fault response and allows the method to capture
whether a signal reacts early, remains persistently abnormal,
or appears mainly as a later propagation effect.
For each signalx j, the faulty recording is compared with
the time-aligned healthy reference over the same interval. Let
xj(t)denote the query signal andxh
j(t)the corresponding
healthy reference signal. The raw mean deviation for a window
Wis computed as
∆µj(W) =1
|W|X
t∈Wxj(t)−1
|W|X
t∈Wxh
j(t)(1)
This quantity captures the direction and magnitude of the
deviation in the original signal units. To make signals with
different scales comparable, the deviation is also normalized
using a robust reference standard deviation. A small standard-
deviation floor is used to avoid unstable z-scores for nearly
constant HiL signals:
zj(W) =|∆µ j(W)|
max(σh
j(W), ϵ)(2)
In addition to the mean-shift score, the maximum deviation
from the healthy reference is computed in normalized form.
The first abnormal reaction time is also estimated by detecting
the first time point at which the reference deviation exceeds
a threshold. This timing feature is important for root-cause
reasoning: a direct fault-location signal that reacts early is usu-
ally more diagnostic than a downstream signal that becomes
abnormal only after propagation. The extracted evidence for
each window therefore contains raw mean difference, mean-
shift z-score, maximum-deviation z-score, first abnormal reac-
tion time, and an overall abnormality score. The abnormalityscore combines persistent deviation and peak deviation, while
clipping extreme values to prevent a single transient spike
from dominating the ranking. The resulting evidence table is
sorted by abnormality score to obtain the most relevant direct
and propagated signal deviations. This compact representation
is then passed to the knowledge-graph and retrieval stages
instead of sending the full raw time series to the LLM.
B. KG-Guided Candidate Ranking and RAG
After the diagnostic evidence has been extracted, the next
step is to translate abnormal signal behavior into plausible
fault-location candidates. This is done using a knowledge
graph that links each candidate fault location to its physically
related signals. The graph contains three types of information:
signal-to-fault mappings, direct and propagation relationships,
and diagnostic rules. Direct mappings define which signal rep-
resents a possible injected fault location, such as accelerator-
pedal position for an accelerator-pedal fault or battery voltage
for a high-voltage battery fault. Propagation relations describe
how a fault at one location may affect downstream signals
through the vehicle dynamics, powertrain, or electrical sub-
system.
For each candidate fault locationc, a KG-guided evidence
score is computed from primary, direct, and propagation signal
evidence:
Sevid(c) =w pSp(c) +w dSd(c) +w gSg(c)(3)
whereS p(c),S d(c), andS g(c)denote the primary-signal,
direct-signal, and propagation-signal evidence, respectively.
The weights give the strongest importance to primary and
direct KG evidence, while propagation signals provide sup-
porting evidence only. This design reflects the diagnostic
assumption that downstream effects can confirm a propagation
path, but they should not dominate the root-cause decision
when the direct fault-location signal provides clear evidence.
Retrieval in inference time is performed by comparing the
query evidence vector with reference evidence vectors from
the historical case library. The similarity between a query
vectorv qand a reference vectorv ris computed using cosine
similarity:
sim(v q,vr) =v⊤
qvr
∥vq∥2∥vr∥2(4)
wherev qis the query evidence vector andv ris a reference
vector from the historical case library.
To avoid retrieval leakage, a leave-one-recording-out strat-
egy is applied. When a query recording is evaluated, all
windows from the same recording are removed from the
retrieval library and from the few-shot example pool. The
retrieved cases therefore come only from other recordings,
although they may contain the same fault location under a
different gain or severity. This setting is important because
the method should generalize to unseen recordings rather than
retrieve near-duplicate windows from the same experiment.

real-time
FaultInjection1. HiL  FI Data
ASM Gasoline Engine
L1
L2
L3
..
EV System
L1
L2
L3
..
2. Pr eprocessing &
Windowing (LORO)
Time alignment with
healthy reference
Signal cleaning &
numeric signal selection
Window-wise
segmentation
5 s windows
...
Leave-One-Recording-
Out (LORO) Split3. Healthy-Refer ence
Evidence Extraction
Faulty W indow Healthy windowDiagnostic
evidence
-
Per-signal diagnostic features
Raw mean difference
Mean-shift z-score
Max-deviation z-score
First abnormal time
Abnormality score
Top abnormal signals
(ranked by score)4. Knowledge-Graph Evidence
Signal to fault
mapping
Direct vs
Propagation
relations
Evidence
scoring per
fault locationCandidate
ranking
5. Retrieval Branch
Query vector
z-diagnostic
vector
Case library for
historical fault
windows
Top-k similar
cases via cosine
similarity
6. KG-Guided Few-Shot Pr ompt Construction
Query Evidence
(top abnormal
signals & featuresKG Facts 
(signal-fault relations
& subsystems)
Retrieved Cases
(similar historical
cases)
Predicted T op-1 & T op-
3 fault locations
Explanation & RCA
reasoning
8. Pr ediction of the fault location with
explanation
Unseen
Data Preprocessing
& WindowingKG-Guided Few-
Shot Prompt
 Gemma-3 27B7. LLM Root-Cause
Diagnosis
Few-Shot
Examples.
Example N
(JSON evidence
+ correct class)Output Schema
(strict JSON)
Model Deployment
Test
Enginee r Evidence
extraction+ + + +Fig. 1. Overview of the proposed KG-guided RAG RCA workflow for HiL time-series data.
The final candidate ranking combines KG-guided evidence
with retrieval support. For each candidatec, the score is
computed as
S(c) =αS evid(c) + (1−α)S ret(c)(5)
whereS ret(c)is obtained from the retrieved cases andα
controls the relative influence of direct diagnostic evidence and
retrieval. In the ASM case study, retrieval is used as a support-
ing signal beside strong KG-based evidence, while in the EV
case study its influence is kept smaller because the available
EV reference library is more limited. The ranked candidates
are then passed to the prompt-construction stage together with
the extracted evidence, KG facts, retrieved reference cases,
and the structural relations between fault locations, directly
affected sensors, and downstream subsystems. These spatial
and functional relationships are important because faults in
HiL systems do not only appear as isolated signal deviations;
they can propagate through physically connected components,
such as from driver inputs to engine behavior or from battery-
side deviations to DC-link and propulsion responses. In thisway, the retrieval component does not simply retrieve similar
examples. It augments the LLM input with structured historical
evidence that is constrained by the knowledge graph, the sub-
system relationships, and the candidate-ranking mechanism.
C. Few-Shot Prompt Construction, LLM Reasoning, and Pre-
diction Aggregation
At inference time, each query recording or query win-
dow is processed through the same evidence-extraction and
candidate-ranking pipeline as the historical cases. The query
evidence vector is compared with the historical case library,
and the top-(k) most similar cases are retrieved. To avoid
label leakage, the query filename and hidden ground-truth
label are excluded from the prompt, since file names contain
fault-location information. The LLM therefore receives only
observable diagnostic evidence, KG facts, retrieved reference
cases, few-shot examples, and the ranked candidate locations.
The prompt is constructed as a compact diagnostic report
rather than as raw time-series input. It includes the query
identifier, analyzed time interval, strongest direct-location

signal deviations, relevant propagation evidence, KG/domain
facts, retrieved similar cases, few-shot decision examples,
and the calibrated candidate ranking. The few-shot examples
are selected from historical reference windows and exclude
windows from the same recording as the query. Their purpose
is to demonstrate the expected reasoning pattern: primary and
direct KG evidence should guide the root-cause decision, while
propagation effects and retrieved cases provide supporting
context. The model is instructed to return valid JSON only.
The output contains the predicted Top-1 fault location, the
Top-3 ranked candidate list, a confidence level, candidate-level
reasoning, the most likely propagation path, uncertainty notes,
and a short RCA explanation. This structure makes the output
automatically parsable while still useful for test engineers, who
receive not only a predicted label but also an explanation of
the supporting evidence and possible downstream effects.
After inference, the response is parsed and validated by
checking the JSON structure, required fields, and predicted
labels against the predefined fault-location set. For window-
wise experiments, the LLM produces one prediction per
evaluated fault window. These predictions are aggregated into
a recording-level diagnosis using rank-based voting over the
Top-3 lists, where higher-ranked candidates receive stronger
support. The final file-level prediction is the candidate with
the highest aggregated support, which reduces the influence
of locally ambiguous windows and provides a more stable
diagnosis for the complete HiL recording.
IV. CASESTUDY ANDDATADESCRIPTION
A. ASM Gasoline Engine
The first case study is based on the ASM gasoline engine
system from dSPACE, integrated into a closed-loop auto-
motive Hardware-in-the-Loop (HiL) environment. The setup
combines a detailed gasoline-engine plant model with ve-
hicle dynamics, driver/environment models, and a controller
layer, enabling fault-injection experiments under realistic real-
time operating conditions [49]. The plant is executed on the
dSPACE SCALEXIO real-time platform, while the control
logic is deployed through a dSPACE MicroAutoBox II, which
emulates the ECU behavior during the experiments.
The gasoline-engine model is considered as part of a
coupled vehicle system rather than as an isolated engine
component. It includes several interacting subsystems,
such as the air path, fuel system, piston engine, exhaust
path, and cooling system. This coupling is important
for the proposed root-cause localization task because a
fault injected at one location can propagate to physically
related signals in other subsystems. In this study, five
fault locations are considered for the ASM gasoline-engine
case: accelerator pedal, brake pedal, steering wheel, engine
speed, and throttle position. These locations are represented
by the signalsPos AccPedal[%],Pos BrakePedal[%],
Angle SteeringWheel[deg],n Engine[rpm], and
Pos Throttle[%], respectively.B. Electric Vehicle System
The second case study uses an electric vehicle system inte-
grated into the same HiL-based validation concept. This case
study complements the gasoline-engine setup by considering
a different propulsion architecture, where the relevant fault
effects are related to electrical power flow, battery behavior,
electric-machine dynamics, and vehicle-level responses. The
plant and environment models are executed on the dSPACE
SCALEXIO real-time platform, while the controller logic
is executed through dSPACE MicroAutoBox II in order to
emulate ECU behavior in the closed-loop experiment. The EV
model includes the main subsystems required for closed-loop
electric powertrain validation, including the high-voltage bat-
tery, DC-link, rear electric machine, steering system, vehicle
dynamics, and driver/environment model. These subsystems
are physically and functionally coupled, so a fault injected
in one component may also affect downstream electrical or
vehicle-level signals. In this study, three EV fault locations are
considered: the high-voltage battery, the rear electric machine
speed, and the steering system. These locations are represented
by the signalsV Bat HV[V],omega EM Rear[rad/s], and
TrqSteeringWheel[Nm], respectively.
C. Dataset Description
The dataset used in this study consists of multivariate time-
series recordings generated from automated fault-injection
experiments in the ASM gasoline-engine and electric-vehicle
HiL setups. Each experimental run is stored as a separate
CSV file. The ASM gasoline-engine files contain 15 columns,
including the time column and sensor measurements from
the driver, vehicle, and engine/powertrain subsystems. The
electric-vehicle files contain 17 columns, also including the
time column, and cover sensor measurements from the battery,
DC-link, propulsion, steering, and vehicle-level subsystems.
Each recording contains approximately 7,700 samples with a
fixed sampling interval of 0.01 s.
The fault type considered in the evaluated dataset is a
gain fault. For the ASM gasoline-engine case study, two
faulty recordings with different gain values are available for
each of the five fault locations, resulting in ten ASM fault-
injection runs. For the electric-vehicle case study, one faulty
recording is available for each of the three fault locations,
resulting in three EV fault-injection runs. In total, the evaluated
fault-injection subset therefore contains 13 CSV recordings.
The recordings include healthy and faulty operating phases,
allowing the injected behavior to be analyzed relative to
a time-aligned healthy reference trajectory. The experiments
capture the temporal system response before, during, and after
fault activation. In the considered recordings, the fault is active
approximately between 12 s and 47 s. This structure makes
the data suitable for evaluating both window-level fault local-
ization and recording-level aggregation. The ASM gasoline-
engine case study considers five fault locations: accelerator
pedal, brake pedal, steering wheel, engine speed, and throttle
position. The electric-vehicle case study considers three fault
locations: high-voltage battery, rear electric machine speed,

and steering system. The recorded signals therefore provide
system-level observations that can be used to evaluate fault
localization under different propulsion architectures and sub-
system interactions.
V. RESULTS ANDDISCUSSION
A. Fault-Localization Performance and Recording-Level Ag-
gregation
TABLE I
PERFORMANCE COMPARISON OFLLMS FORKG-GUIDED FAULT
LOCALIZATION ON THEASMGASOLINE ENGINE ANDEVCASE STUDIES.
Scenario Model Top-1 Acc. Top-3 Acc. MRR F1 Precision Recall MCC
ASM Gasoline Engine Llama-3.3 70B FP8 0.800 0.800 0.800 0.800 0.850 0.800 0.750
Mistral-Small 3.2 24B 0.850 0.950 0.883 0.839 0.870 0.850 0.820
DeepSeek-R1-Qwen 32B 0.850 0.950 0.900 0.844 0.880 0.850 0.820
Qwen3 32B 0.900 1.000 0.933 0.899 0.910 0.900 0.878
Gemma-3 27B 0.900 1.000 0.933 0.899 0.910 0.900 0.878
EV Llama-3.3 70B FP8 0.889 1.000 0.926 0.886 0.917 0.889 0.849
Mistral-Small 3.2 24B 0.833 1.000 0.888 0.823 0.887 0.833 0.783
DeepSeek-R1-Qwen 32B 0.833 1.000 0.888 0.823 0.887 0.833 0.783
Qwen3 32B 0.944 1.000 0.963 0.944 0.952 0.944 0.921
Gemma-3 27B 0.944 1.000 0.963 0.944 0.952 0.944 0.921
MRR: mean reciprocal rank; MCC: Matthews correlation coefficient.
Table I reports the fault-localization performance of the
evaluated LLMs on the ASM gasoline engine and EV case
studies. The results show that the proposed KG-guided RAG
pipeline provides reliable fault-location ranking across both
systems, despite the different signal structures and fault mech-
anisms. In the ASM gasoline engine case, Gemma-3 27B and
Qwen3 32B achieve the strongest performance, with a Top-1
accuracy of 0.900, a Top-3 accuracy of 1.000, and an MRR
of 0.933. This means that the correct injected fault location
is always contained in the three highest-ranked candidates
and is predicted as the first candidate in most windows.
In the EV case study, the same two models again obtain
the best results, reaching a Top-1 accuracy of 0.944. These
results indicate that the proposed prompting strategy does not
depend on a single vehicle architecture, but can transfer to
a second HiL system when the corresponding KG mapping
and diagnostic signal summaries are available. The high Top-
3 accuracy is particularly relevant for root-cause analysis
because engineers often need a ranked list of plausible fault
locations rather than a single label. The results indicate that
the evidence-oriented prompt helps preserve the correct fault
location among the leading candidates, even when the Top-
1 prediction is affected by propagation effects or ambiguity
between physically coupled components.
Differences between models are still visible. Llama-3.3
70B FP8 achieves lower accuracy on the ASM gasoline
engine case, with a Top-1 accuracy of 0.800 and a Top-3
accuracy of 0.800. This indicates that, although the model is
computationally efficient, it is less reliable in preserving the
correct candidate under the KG-guided prompt. Mistral-Small
3.2 24B and DeepSeek-R1-Qwen 32B show intermediate
performance, especially on the ASM case, where both reacha Top-1 accuracy of 0.850 and a Top-3 accuracy of 0.950. In
contrast, Gemma-3 27B and Qwen3 32B consistently achieve
the strongest ranking behavior across both case studies. Since
Gemma-3 27B reaches the same accuracy level as Qwen3 32B
while requiring lower average inference time, it is treated as
the most favorable model for the proposed pipeline in the
subsequent analysis.
TABLE II
LEAVE-ONE-RECORDING-OUT WINDOW-LEVEL AND FILE-LEVEL FAULT
LOCALIZATION RESULTS. FILE-LEVEL PREDICTIONS ARE OBTAINED BY
AGGREGATING WINDOW-LEVEL PREDICTIONS OVER EACH RECORDING.
Case Study Model Win. Files Win. T1 Win. T3 Win. MRR File T1 File MRR
ASM Gasoline Engine Gemma-3 27B 60 10 0.833 0.917 0.8691.000 1.000
EV Gemma-3 27B 18 3 0.944 1.000 0.9631.000 1.000
Win.: windows; T1: Top-1 accuracy; T3: Top-3 accuracy; MRR: mean reciprocal rank.
Table II further evaluates the effect of aggregating window-
level predictions at the recording level. The evaluation follows
a leave-one-recording-out strategy: when a query recording is
evaluated, all windows from that same recording are excluded
from the retrieval library and from the prompt examples. This
prevents the model from retrieving or seeing windows from
the test recording itself and therefore provides a more realistic
estimate of generalization to unseen fault recordings. This
step is important because individual windows do not always
contain the complete fault signature. Some windows capture
the early onset of the fault, while others contain stronger
propagated effects after the fault has influenced downstream
signals. As a result, a single window can be locally ambiguous.
For example, a fault injected into an actuator or engine-related
signal may later affect vehicle velocity, throttle behavior,
pressure signals, or other coupled measurements. The same
effect appears in the EV case, where battery-related faults can
propagate to electric-machine or DC-link signals. Therefore,
evaluating each window independently is useful for temporal
localization, but it can underestimate the reliability of the final
diagnosis at the recording level.
The aggregation results show that this ambiguity is reduced
when predictions are combined over all windows of the same
recording. In the ASM gasoline engine case, the window-level
Top-1 accuracy is 0.833, while the file-level Top-1 accuracy
reaches 1.000. Similarly, in the EV case, the window-level
Top-1 accuracy is 0.944, and the file-level Top-1 accuracy
also reaches 1.000. This shows that the final decision becomes
more stable when multiple fault-active windows contribute
to the prediction. The result is relevant for HiL-based fault
localization because a recorded experiment is usually analyzed
as a complete test run rather than as an isolated window.
Consequently, the recording-level aggregation step aligns the
evaluation more closely with the intended diagnostic use case:
identifying the most likely injected fault location for a full HiL
experiment.

Fig. 2. Retrieval baseline comparison on the ASM gasoline engine case study.
The structured z-only diagnostic representation achieves the strongest Top-3
retrieval accuracy and is selected for the final KG-guided RAG pipeline.
TABLE III
REPRESENTATIVE EXPLANATION EXAMPLES FROMGEMMA-3 27BFOR
CORRECT PREDICTIONS.
Case Study Prediction / Fault Model explanation
ASM Gasoline En-
ginen_Engine/n_Engine
Engine speed
High confidenceGemma-3 27B identifiesn_Engineas the root-cause
location. The explanation refers to direct KG evidence
fromn_Engine[rpm], which has the strongest diagnos-
tic score. The calibrated ranking placesn_Engineahead
of Throttle and AccPedal, supporting the diagnosis.
EVBAT_HV/BAT_HV
High-voltage battery
High confidenceGemma-3 27B identifiesBAT_HVas the root-cause lo-
cation. The explanation refers to direct battery-related
evidence fromV_Bat_HV[V],V_Bat_HV_Plant[V],
SOC_Bat_HV[%], andSOC_HV[%]. These signals
are more diagnostic than propagated effects from
Omega_EM_Rearor Steering.
B. Retrieval Representation and KG-Guided Evidence
The retrieval component was evaluated independently to
identify the most suitable representation for retrieving diag-
nostically similar HiL fault windows. As shown in Fig. 2,
the structured z-only diagnostic representation achieves the
strongest performance on the ASM gasoline engine case
study, with 0.50 Top-1 accuracy and 1.00 Top-3 accuracy.
This representation uses healthy-reference mean-shift and
maximum-deviation z-scores, which makes deviations compa-
rable across signals with different physical units and dynamic
ranges. In contrast, raw numerical vectors, text-based repre-
sentations, sentence embeddings, and MiniRocket embeddings
show weaker retrieval performance. The final KG-guided RAG
pipeline therefore, uses the structured z-only representation,
since it preserves physically meaningful deviation patterns
while remaining compact enough for prompt construction.
The explanation examples in Table III show that the LLM
decisions are consistent with the KG-guided evidence. In the
ASM case, Gemma-3 27B correctly predictsn_Engineand
refers to the direct signaln_Engine[rpm], which also has
the strongest deviation score in Table IV. In the EV case, the
model correctly identifiesBAT_HVand supports the decision
using battery-related signals such asV_Bat_HV[V]and
V_Bat_HV_Plant[V]. These examples illustrate the benefit
of combining numerical evidence with KG relations: the
model can distinguish likely source signals from downstream
propagation effects, instead of treating all abnormal signals asTABLE IV
TOP DIAGNOSTIC SIGNAL DEVIATIONS FOR THE SELECTED CORRECT
GEMMA PREDICTIONS. SCORES ARE COMPUTED FROM DEVIATIONS
AGAINST THE TIME-ALIGNED HEALTHY REFERENCE TRAJECTORY.
Case Study Window ID Signal ScoreRaw Mean
Diff.Mean
Shift zMax
Dev. zFirst
Abn. Time
ASM Gasoline EngineASM009_w001 n_Engine[rpm]20.000 6971.069 21.782 24.838 20.00s
ASM Gasoline EngineASM009_w001 p_Out_EGR[Pa]4.547 -84559.420 4.298 4.921 20.23s
ASM Gasoline EngineASM009_w001 p_Rail[bar]1.952 -0.057 0.008 4.868 22.44s
ASM Gasoline EngineASM009_w001 Pos_Throttle[%]1.826 -56.950 1.402 2.463 None
ASM Gasoline EngineASM009_w001 Pos_BrakePedal[%]1.684 -1.226 0.368 3.656 22.15s
EVEV001_w000 T_Bat_HV[degC]15.870 3.237 15.794 15.985 15.00s
EVEV001_w000 V_Bat_HV_Plant[V]5.204 -23.049 2.350 9.486 15.00s
EVEV001_w000 V_Bat_HV[V]5.204 -23.049 2.350 9.486 15.00s
EVEV001_w000 V_DCLink[V]5.202 -23.063 2.350 9.481 15.00s
EVEV001_w000 I_Bat_HV[A]3.651 -50.653 0.684 8.101 15.00s
Abn.: abnormal; Dev.: deviation.
equivalent.
C. Computational Complexity and Accuracy–Latency Trade-
off
Fig. 3. Accuracy–complexity trade-off of the evaluated LLMs for KG-guided
fault localization. Gemma-3 27B achieves the same highest mean Top-1
accuracy as Qwen3 32B while requiring lower average inference time, making
it the preferred accuracy–latency trade-off.
The practical usefulness of the proposed KG-guided RAG
pipeline depends on both localization performance and com-
putational cost. The LLM experiments were executed using a
vLLM serving setup on an NVIDIA RTX PRO 6000 Blackwell
GPU with 97,887 MiB of available GPU memory, CUDA 13.2,
and driver version 595.71.05. Table V summarizes the resource
measurements of the evaluated models. Inference time and
token counts are averaged over the evaluation prompts, while
peak memory denotes the total GPU memory reserved during
inference on the evaluation hardware.
The results show a clear accuracy-latency trade-off. Llama-
3.3 70B FP8 is the fastest model, requiring 2.856 s per prompt
and reaching the highest throughput of 0.351 prompts/s.
However, its localization performance is lower than the best-
performing models, especially on the ASM gasoline engine
case. In contrast, Qwen3 32B and Gemma-3 27B achieve
the strongest predictive performance across both case studies.
Both models obtain the best overall Top-1 accuracy, Top-3
accuracy, MRR, F1-score, and MCC values. Their runtime
differs, however: Qwen3 32B requires 6.614 s per prompt,
whereas Gemma-3 27B requires 4.773 s per prompt. Thus,

TABLE V
RESOURCE AND COMPLEXITY COMPARISON OF THE EVALUATEDLLMS
USING THEKG-GUIDED FEW-SHOT FAULT-LOCALIZATION PROMPT.
VALUES ARE AVERAGED OVER THE EVALUATION PROMPTS.
ModelInf.
time [s]Thr.Prompt
tokensOutput
tokensLoad
time [s]Mem.
load [GB]Peak
mem. [GB]
Gemma 3 27B 4.773 0.210 1502.644 101.056 206.055 86.013 86.013
Qwen3 32B 6.614 0.152 1502.644 110.087 174.051 86.241 86.243
DeepSeek R1 Distill Qwen 32B 5.879 0.171 1502.644 92.515 82.025 86.271 86.271
Mistral Small 3.2 24B 4.940 0.203 1502.644 112.817 108.034 86.353 86.353
Llama 3.3 70B FP82.856 0.3511502.64468.03984.023 86.333 86.333
Thr.: throughput. Prompt-token counts are computed with each model’s native tokenizer.
Peak memory denotes the total GPU memory reserved during inference, not the minimal
model footprint.
Gemma reaches the same highest localization performance
while reducing the average inference time by approximately
28%. Fig. 3 visualizes this trade-off between mean Top-
1 accuracy and inference time. Gemma-3 27B lies on the
Pareto frontier because no other evaluated model is both
more accurate and faster. Although Llama-3.3 70B FP8 is
computationally efficient, its lower accuracy makes it less
suitable as the final diagnostic model. Qwen3 32B provides the
same accuracy as Gemma, but at a higher inference cost. Since
model loading is a one-time cost during a serving session and
GPU memory usage is similar across models at approximately
86 GB, the average inference time per prompt is the most
relevant runtime criterion for the offline HiL analysis work-
flow. Gemma-3 27B provides the most favorable accuracy-
latency trade-off and is selected as the preferred model for the
proposed KG-guided RAG-based fault-localization pipeline.
VI. CONCLUSION
This paper presented a KG-guided retrieval-augmented
LLM framework for root cause analysis and fault localization
in automotive HiL validation data. The proposed method trans-
forms raw multivariate recordings into healthy-reference diag-
nostic evidence, enriches this evidence with sensor-to-location
and propagation knowledge, retrieves similar historical cases,
and uses an LLM as a reasoning layer to generate ranked
fault-location predictions with explanatory RCA output.
The framework was evaluated on two HiL case studies:
an ASM gasoline engine and an electric vehicle system. The
results show that structured diagnostic evidence, KG-guided
candidate ranking, and retrieval-augmented prompting can
support reliable and explainable fault localization across dif-
ferent vehicle architectures. In particular, the strongest models
achieved high Top-1 localization accuracy, while recording-
level aggregation improved the stability of the final diagnosis
by reducing the influence of locally ambiguous windows.
The current study has some limitations. The evaluation
dataset is still limited in size and mainly focuses on gain
faults. In addition, the present experiments address single-fault
localization, while concurrent faults may create overlapping
direct and propagated effects that are more difficult to sep-
arate. The quality of the RCA explanation also depends onthe completeness of the knowledge graph and the available
historical case library.
Finally, future work will extend the framework to concur-
rent and interacting fault scenarios, larger HiL datasets, and
additional fault types. Another direction is to integrate richer
spatial or causal representations, such as dynamic knowledge
graphs or graph neural networks, to model fault propagation
more explicitly.
REFERENCES
[1] International Organization for Standardization, “Iso 26262-10:2018-
road vehicles—functional safety—part 10: Guideline on iso 26262,”
https://www.iso.org/standard/68392.html, 2018, accessed on: Feb. 14,
2026, [Online].
[2] M. Abboush, D. Bamal, C. Knieke, and A. Rausch, “Intelligent fault
detection and classification based on hybrid deep learning methods
for hardware-in-the-loop test of automotive software systems,”Sensors,
vol. 22, no. 11, p. 4066, 2022.
[3] M. Abboush, C. Knieke, and A. Rausch, “A virtual testing framework for
real-time validation of automotive software systems based on hardware
in the loop and fault injection,”Sensors, vol. 24, no. 12, p. 3733, 2024.
[4] M. Abboush, E. Ghannoum, and A. Rausch, “An explainable hybrid
deep learning-enabled intelligent fault detection and diagnosis approach
for automotive software systems validation,”Knowledge-Based Systems,
p. 114922, 2025.
[5] S. Salih and R. Olawoyin, “Fault injection in model-based system failure
analysis of highly automated vehicles,”IEEE Open Journal of Intelligent
Transportation Systems, vol. 2, pp. 417–428, 2021.
[6] Z. Wen, J. Crossman, J. Cardillo, and Y . L. Murphey, “Case-base rea-
soning in vehicle fault diagnostics,” inProceedings of the International
Joint Conference on Neural Networks, 2003., vol. 4. IEEE, 2003, pp.
2679–2684.
[7] E. Olsson, P. Funk, and N. Xiong, “Fault diagnosis in industry using
sensor readings and case-based reasoning,”Journal of Intelligent &
Fuzzy Systems, vol. 15, no. 1, pp. 41–46, 2004.
[8] V . Venkatasubramanian, R. Rengaswamy, and S. N. Kavuri, “A review
of process fault detection and diagnosis: Part ii: Qualitative models and
search strategies,”Computers & chemical engineering, vol. 27, no. 3,
pp. 313–326, 2003.
[9] R. Isermann, “Model-based fault-detection and diagnosis–status and
applications,”Annual Reviews in control, vol. 29, no. 1, pp. 71–85,
2005.
[10] Y . Liu, Y . Zhou, Y . Liu, Z. Xu, and Y . He, “Intelligent fault diagnosis
for cnc through the integration of large language models and domain
knowledge graphs,”Engineering, 2025.
[11] L. Lin, S. Zhang, S. Fu, and Y . Liu, “Fd-llm: Large language model
for fault diagnosis of complex equipment,”Advanced Engineering
Informatics, vol. 65, p. 103208, 2025.
[12] H. Zhang, W. Wang, L. Zhang, S. Shao, Q. Wang, J. Li, and J. Hu,
“Large language models for prognostic analysis in mechanical fault
diagnosis,”PLoS One, vol. 20, no. 11, p. e0337203, 2025.
[13] Z. Qi, J. Ren, W. Gan, and P. S. Yu, “Large language models for
fault diagnosis,” in2025 IEEE International Conference on Big Data
(BigData). IEEE, 2025, pp. 6982–6991.
[14] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.
[15] Y . Ma, S. Zheng, Z. Yang, H. Pan, and J. Hong, “A knowledge-graph
enhanced large language model-based fault diagnostic reasoning and
maintenance decision support pipeline towards industry 5.0,”Interna-
tional Journal of Production Research, pp. 1–22, 2025.
[16] H. Zhang, Y . Zhao, B. Sun, Y . Wu, Z. Fu, and X. Xiao, “Large language
model based intelligent fault information retrieval system for new energy
vehicles,”Applied Sciences, vol. 15, no. 7, p. 4034, 2025.
[17] S. Lukens, M. Bishof, N. Siddiqui, and D. West, “An evaluation
framework for fault diagnosis using technical manuals in retrieval-
augmented large language models,” inAnnual Conference of the PHM
Society, vol. 17, no. 1, 2025.

[18] J. Xu, Z. Chen, Z. Jiang, D. Zhao, and L. Li, “Iterative search-based rag:
Using iterative retrieval for efficient large language model enhancement
in industrial fault diagnosis,” in2025 IEEE 8th International Conference
on Industrial Cyber-Physical Systems (ICPS). IEEE, 2025, pp. 1–6.
[19] W. Lin and K. Miao, “An automotive fault diagnosis framework based
on knowledge graphs and large language models,”Electronics, vol. 14,
no. 21, p. 4180, 2025.
[20] S. Safavi, M. A. Safavi, H. Hamid, and S. Fallah, “Multi-sensor fault
detection, identification, isolation and health forecasting for autonomous
vehicles,”Sensors, vol. 21, no. 7, p. 2547, 2021.
[21] H. Kaplan, K. Tehrani, and M. Jamshidi, “A fault diagnosis design based
on deep learning approach for electric vehicle applications,”Energies,
vol. 14, no. 20, p. 6599, 2021.
[22] A. Theissler, “Detecting known and unknown faults in automotive
systems using ensemble-based anomaly detection,”Knowledge-Based
Systems, vol. 123, pp. 163–173, 2017.
[23] M. Abboush, C. Knieke, and A. Rausch, “Intelligent identification of
simultaneous faults of automotive software systems under noisy and
imbalanced data using ensemble lstm and random forest,”IEEE Access,
vol. 11, pp. 140 022–140 040, 2023.
[24] R. Raveendran, K. Devika, and S. C. Subramanian, “Brake fault iden-
tification and fault-tolerant directional stability control of heavy road
vehicles,”IEEE Access, vol. 8, pp. 169 229–169 246, 2020.
[25] W. Pietrowski, M. Puskarczyk, and J. Szymenderski, “Fault detection
methods for electric power steering system using hardware in the loop
simulation,”Energies, vol. 17, no. 14, p. 3486, 2024.
[26] C. Men, Y . Han, P. Wang, J. Tao, and C.-G. Huang, “The interpretable
reasoning and intelligent decision-making based on event knowledge
graph with llms in fault diagnosis scenarios,”IEEE Transactions on
Instrumentation and Measurement, 2025.
[27] F. Li, X. Li, and J. Bao, “Long causal chain-of-thought for compound
fault diagnosis: enhancing hypergraph-based causal-llm intervention
reasoning,”Journal of Manufacturing Systems, vol. 86, pp. 218–246,
2026.
[28] L. Peifeng, L. Qian, X. Zhao, and B. Tao, “Joint knowledge graph and
large language model for fault diagnosis and its application in aviation
assembly,”IEEE Transactions on Industrial Informatics, vol. 20, no. 6,
pp. 8160–8169, 2024.
[29] L. Li, A. Haruna, W. Ying, K. Noman, and Y . Li, “Knowledge graph-
driven fault diagnosis for aviation equipment: Integrating improved joint
extraction with large language model,”Journal of Industrial Information
Integration, p. 101039, 2025.
[30] Q. Nie, J. Geng, D. Tang, and C. Liu, “Industrial knowledge-enhanced
fault diagnosis method: Integrating llm and knowledge graph for fault
reasoning and maintenance recommendation in cnc machine tools,”
Computers & Industrial Engineering, p. 111879, 2026.
[31] F. Wang, X. Zhang, R. Tang, and J. Jiang, “Wt-fault-llm: Knowledge
graph enhanced large model for wind-turbine faults,”Procedia Computer
Science, vol. 266, pp. 385–390, 2025.
[32] Y . Gao, Y . Qin, T. Wang, Y . Cui, and P. Wang, “Knowledge graph
enhanced large language model framework for causal chain reasoning in
industrial fault diagnosis,”Expert Systems with Applications, p. 132027,
2026.
[33] R. Miao, T. Wu, and Z. Zhang, “Graph rag-based fault diagnosis for train
bogies using knowledge graphs and large language model,”Knowledge-
Based Systems, p. 114855, 2025.
[34] J. Zhuang, J. Yang, W. Li, J. Chen, Y . Zheng, and Z. Chen, “Large
model for fault diagnosis of industrial equipment based on a knowledge
graph construction,”Applied Soft Computing, p. 113936, 2025.
[35] S. Liu, H. Xie, Y . Lei, J. Zhao, and X. Song, “Llm-augmented causal-
knowledge heterogeneous graph framework for interpretable reasoning
and collaborative knowledge fusion in automotive chip production,”
Expert Systems with Applications, p. 131343, 2026.
[36] S. Zheng and D. Pi, “Root cause diagnosis for time series data faults
based on fault isolation and channel-attentive sparse causal networks,”
IEEE Sensors Journal, 2025.
[37] D. Wang, Z. Chen, J. Ni, L. Tong, Z. Wang, Y . Fu, and H. Chen, “Inter-
dependent causal networks for root cause localization,” inProceedings
of the 29th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining, 2023, pp. 5051–5060.
[38] L. Zhao, P. Song, and C. Zhao, “Causal similarity learning with multi-
level predictive relation aggregation for grouped root cause diagnosis
of industrial faults,”Control Engineering Practice, vol. 154, p. 106140,
2025.[39] S. Ji, J. Wang, Z. Wang, and F. Zhang, “An explainable causal graph-
based method for diagnosis and root cause analysis in process systems,”
Process Safety and Environmental Protection, p. 107953, 2025.
[40] Z. Pan, Y . Hu, J.-H. Choi, D. Zhang, Z. Zhou, S. Zhang, and Y . Zhang,
“Temporal causal discovery-enhanced hierarchical monitoring for root
cause diagnosis of aero-engine faults,”Knowledge-Based Systems, p.
115921, 2026.
[41] R. Wang, X. Gao, J. Gao, Z. Gao, and J. Kang, “An information transfer
based novel framework for fault root cause tracing of complex elec-
tromechanical systems in the processing industry,”Mechanical Systems
and Signal Processing, vol. 101, pp. 121–139, 2018.
[42] H. Huang, T. Shah, J. Karigiannis, and S. Evans, “Deep root cause
analysis: unveiling anomalies and enhancing fault detection in industrial
time series,” in2024 international joint conference on neural networks
(ijcnn). IEEE, 2024, pp. 1–8.
[43] C. Zhang, D. Hu, and T. Yang, “Research of artificial intelligence
operations for wind turbines considering anomaly detection, root cause
analysis, and incremental training,”Reliability Engineering & System
Safety, vol. 241, p. 109634, 2024.
[44] M. Park, H. Kim, S. Lee, D. Kim, and J. Lee, “Explainable ai-based root-
cause diagnosis of maritime engine anomalies with external factors,”
Computers & Industrial Engineering, p. 112028, 2026.
[45] C. Feng, C. Liu, and D. Jiang, “Root cause localization for wind
turbines using physics guided multivariate graphical modeling and fault
propagation analysis,”Knowledge-Based Systems, vol. 295, p. 111838,
2024.
[46] L. Qiao, X. Li, X. Wang, and K. Peng, “Root cause diagnosis and fault
propagation path identification for complex industrial processes based
on data space,”Measurement, vol. 226, p. 114219, 2024.
[47] P. Klein, L. Malburg, and R. Bergmann, “Combining informed data-
driven anomaly detection with knowledge graphs for root cause anal-
ysis in predictive maintenance,”Engineering Applications of Artificial
Intelligence, vol. 145, p. 110152, 2025.
[48] W. Huang and J. Ma, “Analysis of vehicle fault diagnosis model based
on causal sequence-to-sequence in embedded systems,” 2025.
[49] M. Abboush, D. Bamal, C. Knieke, and A. Rausch, “Hardware-in-the-
loop-based real-time fault injection framework for dynamic behavior
analysis of automotive software systems,”Sensors, vol. 22, no. 4, p.
1360, 2022.