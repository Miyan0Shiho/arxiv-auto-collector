# EHR-RAG: Bridging Long-Horizon Structured Electronic Health Records and Large Language Models via Enhanced Retrieval-Augmented Generation

**Authors**: Lang Cao, Qingyu Chen, Yue Guo

**Published**: 2026-01-29 07:06:34

**PDF URL**: [https://arxiv.org/pdf/2601.21340v1](https://arxiv.org/pdf/2601.21340v1)

## Abstract
Electronic Health Records (EHRs) provide rich longitudinal clinical evidence that is central to medical decision-making, motivating the use of retrieval-augmented generation (RAG) to ground large language model (LLM) predictions. However, long-horizon EHRs often exceed LLM context limits, and existing approaches commonly rely on truncation or vanilla retrieval strategies that discard clinically relevant events and temporal dependencies. To address these challenges, we propose EHR-RAG, a retrieval-augmented framework designed for accurate interpretation of long-horizon structured EHR data. EHR-RAG introduces three components tailored to longitudinal clinical prediction tasks: Event- and Time-Aware Hybrid EHR Retrieval to preserve clinical structure and temporal dynamics, Adaptive Iterative Retrieval to progressively refine queries in order to expand broad evidence coverage, and Dual-Path Evidence Retrieval and Reasoning to jointly retrieves and reasons over both factual and counterfactual evidence. Experiments across four long-horizon EHR prediction tasks show that EHR-RAG consistently outperforms the strongest LLM-based baselines, achieving an average Macro-F1 improvement of 10.76%. Overall, our work highlights the potential of retrieval-augmented LLMs to advance clinical prediction on structured EHR data in practice.

## Full Text


<!-- PDF content starts -->

EHR-RAG: Bridging Long-Horizon Structured Electronic Health Records and
Large Language Models via Enhanced Retrieval-Augmented Generation
Lang Cao1Qingyu Chen2Yue Guo1
Abstract
Electronic Health Records (EHRs) provide rich
longitudinal clinical evidence that is central to
medical decision-making, motivating the use of
retrieval-augmented generation (RAG) to ground
large language model (LLM) predictions. How-
ever, long-horizon EHRs often exceed LLM con-
text limits, and existing approaches commonly
rely on truncation or vanilla retrieval strategies
that discard clinically relevant events and tempo-
ral dependencies. To address these challenges, we
proposeEHR-RAG, a retrieval-augmented frame-
work designed for accurate interpretation of long-
horizon structured EHR data.EHR-RAGintro-
duces three components tailored to longitudinal
clinical prediction tasks:Event- and Time-Aware
Hybrid EHR Retrievalto preserve clinical struc-
ture and temporal dynamics,Adaptive Iterative
Retrievalto progressively refine queries in order
to expand broad evidence coverage, andDual-
Path Evidence Retrieval and Reasoningto jointly
retrieves and reasons over both factual and coun-
terfactual evidence. Experiments across four long-
horizon EHR prediction tasks show thatEHR-
RAGconsistently outperforms the strongest LLM-
based baselines, achieving an average Macro-F1
improvement of 10.76%. Overall, our work high-
lights the potential of retrieval-augmented LLMs
to advance clinical prediction on structured EHR
data in practice.
1. Introduction
Electronic Health Records (EHRs) are digital longitudinal
patient records composed of structured clinical events such
as diagnoses, medications, laboratory results, and proce-
dures. Unlike free-text clinical notes, structured EHR data
are systematically collected, standardized across care set-
1University of Illinois Urbana-Champaign, United States2Yale
University, United States. Correspondence to: Lang Cao <lang-
cao2@illinois.edu>, Yue Guo <yueg@illinois.edu>.
Preprint. January 30, 2026.tings, and less subjective in description, making them a
reliable and widely available foundation for clinical deci-
sion making (Rosenbloom et al., 2011; Ebbers et al., 2022).
Beyond individual events, structured EHRs encode rich tem-
poral and categorical information, including event types,
ordering, recurrence patterns, and long-term clinical trajec-
tories, that underpins clinical decision-making across the
full course of clinical care (Menachemi & Collum, 2011;
Cowie et al., 2017). For many non-acute or chronic con-
ditions, patients accumulate long-horizon EHRs spanning
years of irregular visits and thousands of heterogeneous
events. These trajectories are common in real-world health-
care but difficulty to interpret, even for clinicians, and recent
studies have highlighted both their modeling challenges and
clinical importance (Loh et al., 2025; Wornow et al., 2023;
Fries et al., 2025). Effective clinical prediction in this setting
requires reasoning over long temporal spans while preserv-
ing event-type structure and clinical dependencies.
Traditional machine learning (ML) methods have been
widely studied for EHR-based prediction tasks, including
length-of-stay estimation (Cai et al., 2016; Levin et al.,
2021), readmission prediction (Ashfaq et al., 2019; Xiao
et al., 2018), and medication recommendation (Bhoi et al.,
2021; Shang et al., 2019). More recently, large language
models (LLMs) (Achiam et al., 2023; Touvron et al., 2023)
have shown promise in clinical reasoning due to their strong
generalization capabilities to new task and their capability
to perform multi-step reasoning to provide rationale instead
of only final answer to make it more trustworthy for clini-
cian’s decision making (Bedi et al., 2025; Jiang et al., 2023;
2024; Kojima et al., 2022). However, LLMs cannot na-
tively process long-horizon structured EHRs: their fixed
context windows are quickly exceeded by multi-visit patient
histories, and naive event serialization obscures temporal
structure and discards clinically relevant context.
Existing approaches address this limitation primarily
through heuristic truncation, such as retaining only recent
events (Lin et al., 2025) or selecting events associated with
high-frequency codes (Liao et al., 2025). While retrieval-
augmented generation (RAG) (Gao et al., 2023) offers a
mechanism to selectively retrieve relevant events, existing
RAG-based methods typically retrieve isolated subsets of
1arXiv:2601.21340v1  [cs.AI]  29 Jan 2026

EHR-RAG
the patient history primarily to satisfy context-length con-
straints (Zhu et al., 2024b; Kruse et al., 2025; Zhu et al.,
2024a). As a result, vanilla RAG in clinical settings of-
ten suffers from low retrieval quality, incomplete evidence
coverage, and reduced robustness when reasoning over long-
horizon EHRs (Li et al., 2025b; Perçin et al., 2025). Despite
the prevalence and importance of such data, to the best of
our knowledge, no existing RAG framework is explicitly
designed to handle their longitudinal, multi-event nature.
In this work, we study clinical prediction with long-horizon
structured EHR data and proposeEHR-RAG, an enhanced
retrieval-augmented framework tailored for this setting.
EHR-RAGis built around three components designed to
improve retrievalquality,completeness, androbustnessun-
der strict context constraints: (a)Event- andTime-Aware
HybridEHRRetrieval(ETHER), which preserves event-
type structure and temporal dynamics during retrieval; (b)
AdaptiveIterativeRetrieval(AIR), which progressively re-
fines retrieval queries to expand evidence coverage while
respecting context limits; and (c)Dual-PathEvidence
Retrieval andReasoning(DER), which jointly retrieves
and reasons over both factual patient history and counterfac-
tual evidence to improve robustness. Together, these designs
enable reliable long-horizon clinical reasoning with LLMs.
We evaluateEHR-RAGagainst a range of baseline methods
on a long-horizon EHR benchmark. Experimental results
demonstrate consistent Macro-F1 improvements across all
four clinical prediction tasks, with gains of 3.63% onLong
Length of Stay, 11.28% on30-day Readmission, 16.46% on
Acute Myocardial Infarction, and 7.66% onAnemiarelative
to the strongest LLM-based baselines. Overall,EHR-RAG
achieves an average Macro-F1 improvement of 10.76%.
In summary, we make the following contributions:
•We identify and analyze the limitations of existing
truncation-based and vanilla RAG approaches on long-
horizon structured EHR data, highlighting challenges in
evidence completeness and temporal reasoning for reli-
able clinical prediction.
•We proposeEHR-RAG, a retrieval-augmented framework
explicitly designed for long-horizon EHR reasoning, inte-
grating event- and time-aware retrieval, adaptive iterative
retrieval, dual-path evidence retrieval.
•We conduct extensive experiments demonstrating consis-
tent improvements over strong baselines across multiple
long-horizon EHR prediction tasks.
2. Related Work
Clinical Predictive Models.Structured EHR data has sup-
ported the development of a wide range of machine learning
models for different clinical prediction tasks (Cai et al.,
2016; Levin et al., 2021; Ashfaq et al., 2019; Bhoi et al.,2021). Models such as RETAIN (Choi et al., 2016), GRAM
(Choi et al., 2017), and KerPrint (Yang et al., 2023) are
specifically designed to capture complex temporal and hier-
archical patterns within structured EHRs and have shown
strong performance across multiple predictive settings. In
parallel, recent efforts have focused on scaling clinical pre-
diction through large foundation models, including CLMBR-
T-Base (Wornow et al., 2023), Virchow (V orontsov et al.,
2024), and MIRA (Li et al., 2025a). However, traditional
predictive models—ranging from task-specific architectures
to supervised foundation model training—remain inflexible,
requiring labeled data for every downstream task and often
failing to generalize beyond the distribution on which they
were trained. These limitations are especially problematic
in the dynamic and heterogeneous healthcare environment.
LLM-based Clinical Prediction.Recently, the paradigm
of clinical prediction has begun to shift toward the use of
LLMs (Achiam et al., 2023). LLMs offer greater adaptabil-
ity and generalization than traditional supervised models,
with the capacity to interpret diverse medical information
and support more versatile clinical decision-making. Sev-
eral studies apply LLMs directly to clinical prediction tasks
(Lovon-Melgarejo et al., 2024; Zhu et al., 2024c; Chen et al.,
2024; Kruse et al., 2025), while others employ LLMs as
encoders for structured EHR data (Hegselmann et al., 2025).
Additional efforts explore training or adapting LLMs for
specific downstream healthcare tasks (Lin et al., 2025; Jiang
et al., 2023; 2024; Yang et al., 2022), and recent work intro-
duces LLM-driven agents to assist with clinical reasoning
and EHR interaction (Cui et al., 2025; Shi et al., 2024).
Despite these advances, few studies have addressed long-
horizon EHR prediction, leaving substantial challenges in
enabling LLMs to interpret complex, multi-visit longitudi-
nal patient histories. Our work extends this line of research
by developing an LLM-based approach specifically tailored
for long-horizon EHR prediction.
Retrieval-Augmented Generation with LLM.RAG has
emerged as an effective technique for enhancing LLMs with
external knowledge beyond their limited context window
(Gao et al., 2023). A variety of retrieval strategies have been
proposed to help LLMs accurately access additional rele-
vant information, leading to improved performance across
diverse tasks and settings (Asai et al., 2023; Jiang et al.,
2025b;a). In the healthcare domain, where clinical infor-
mation is often overwhelming in volume and complexity,
RAG has become increasingly popular for grounding LLM
outputs in accurate task-relevant evidence (Neha et al., 2025;
Kim et al., 2025). Some studies use RAG to inject medi-
cal knowledge and improve healthcare applications (Soman
et al., 2024; Niu et al., 2024; Cao et al., 2024a), while
others employ RAG to mitigate long-context limitations.
For example, REALM (Zhu et al., 2024b) enhances mul-
timodal EHR analysis using LLMs, and EMERGE (Zhu
2

EHR-RAG
EHR-RAGDirectGeneration
Vanilla RAG
Long-horizon EHR
QueryLLMPredictionLong-horizon EHR
FinalPredictionFactualEvidence
CounterfactualEvidence(a) ETHER: Event-and Time-AwareHybrid EHR Retrieval
Result
EvidenceFusionLimited Context Window
Incomplete Evidence RetrievalEnsure Retrieval Quality,Completeness, and Robustness
Initial/RecentEvidence
Numeric-valued Evidence
Text-based Evidence
Event-awareRetrieval
Indicator-wise AggregationNumerical Evidence Trajectory
U-shapeTime-awareRetrieval
Positive Outcome Hypothesis
Negative Outcome Hypothesis(c) DER: Dual-Path Evidence Retrieval and ReasoningDual-Hypothesis Reasoning
(b) AIR: Adaptive Iterative Retrieval for Evidence Refinement
Query
LLMCounterfactualQueryFactualQueryInfo-Seeking QueryConcept Verbalization
Integrated Evidence
Legend:
QueryLLMPrediction
Long-horizon EHRRetrievalRetrieved Clinical Evidence
Figure 1.Overview of theEHR-RAGframework for long-horizon clinical prediction. Compared withDirect GenerationandVanilla RAG,
our framework explicitly addresses context truncation and incomplete retrieval. It integrates(a) Event- and Time-aware Hybrid Retrieval
(ETHER),(b) Adaptive Iterative Retrieval (AIR), and(c) Dual-Path Factual and Counterfactual Reasoning (DER), ensuring retrieval
quality, robustness, and completeness for reliable clinical decision-making.
et al., 2024a) integrates RAG into multimodal EHR predic-
tive modeling. Additional work has investigated RAG-based
methods for clinical trial and patient matching (Tramontini
et al., 2025).However, most of these approaches focus on
unstructured EHR data such as clinical notes, rather than
structured tabular EHR, which exhibits distinct temporal
dynamics and event-type structure. Moreover, vanilla RAG
pipelines often suffer from incomplete evidence retrieval in
clinical settings. Our work aims to fill this gap by design-
ing a RAG framework specifically tailored for long-horizon
structured EHR data.
3. Methodology
Figure 1illustratesEHR-RAG, a retrieval-augmented frame-
work designed to enable LLMs to reason over long-horizon
structured EHR data under strict context constraints. The
framework comprises three core components:Event- and
Time-Aware Hybrid EHR Retrieval (ETHER),Adaptive It-
erative Retrieval (AIR), andDual-Path Evidence Retrieval
and Reasoning (DER).
3.1. Task Formulation
Given a patient p, we assume access to a longitudinal struc-
tured EHR Ep={e i}N
i=1, where each clinical event is rep-
resented as ei= (c i, vi, τi), with cidenoting a clinical
concept (e.g., diagnosis, procedure, laboratory test, or med-
ication), vithe associated value (numeric or textual), and
τithe timestamp at which the event occurred. Events are
ordered chronologically such thatτ 1≤τ 2≤ ··· ≤τ N.At a prediction timeτ∗, the available patient history is
E≤τ∗
p={e i∈ Ep|τi≤τ∗}.(1)
Our goal is to predict a clinical outcome y∈ Y (binary or
multiclass) based on this long-horizon EHR context. Exam-
ples include prolonged length of stay, laboratory abnormal-
ity severity, or risk of acute clinical events.
We assume access to a pretrained LLM Mcapable of clini-
cal reasoning but limited by a fixed context window, making
directly input of E≤τ∗
p infeasible when Nis large. There-
fore, we aim to construct a compact, task-relevant evidence
context C ⊂ E≤τ∗
psuch that ˆy=M(C) , is consistent with
the ground-truth outcomey.
3.2. Event- and Time-Aware Hybrid EHR Retrieval
Structured EHR E≤τ∗
p contain heterogeneous event types,
including numeric measurements (e.g., lab tests and vital
signs) and textual records (e.g., event descriptions and clin-
ical notes). Treating all clinical events uniformly during
retrieval is often ineffective. Dense text embeddings are pri-
marily optimized for capturing semantic similarity and are
known to be insensitive to precise numerical values and mag-
nitude differences (Wallace et al., 2019; Zhang et al., 2020),
which limits their effectiveness for reasoning over numeric
clinical measurements such as laboratory results and vital
signs. Moreover, raw numerical values alone do not fully
convey their clinical significance without appropriate tem-
poral context and longitudinal interpretation. Motivated by
these limitations, we propose an event- and time-aware hy-
brid retrieval strategy that explicitly handles numeric events
separately while preserving their temporal structure.
3

EHR-RAG
Indicator-wise aggregation of numeric events.All nu-
meric clinical events are first grouped by clinical indica-
tor (i.e., event name), forming a collection of indicator-
specific temporal trajectories. Formally, for each indi-
cator k, we define its numeric value trajectory as Vk=
{(vk,j, τk,j)|τk,j≤τ∗}, where vk,jdenotes the j-th ob-
served numeric value of indicator kandτk,jis its corre-
sponding timestamp. Collectively, the set {Vk}forms an
indicator-level numeric representation of the patient, where
eachVkcaptures the longitudinal evolution of a single clini-
cal measurement. This representation preserves clinically
meaningful temporal dynamics while enabling efficient and
semantically informed retrieval over numeric EHR data.
Coarse-to-fine indicator selection.Given a task-specific
query qtask, we first perform coarse-grained retrieval over
all indicator-level summaries {Vk}using dense similar-
ity search, where embeddings are computed from indi-
cator names. This step yields a candidate indicator set
Kcoarse ={k 1, . . . , k Ncoarse}, where Ncoarse denotes the num-
ber of indicators retained after coarse retrieval. We then ap-
ply LLM-based reranking to select a smaller, task-relevant
subset Kfine⊂ K coarse such that |Kfine|=N fine. For each
selected indicator k∈ K fine, we retain the Nrecent most re-
cent measurements occurring prior to the prediction time τ∗
to ensure temporal relevance, forming an indicator-specific
numeric evidence trajectory Ek={(v k,j, τk,j)∈ V k|
τk,j≤τ∗}. Collectively, these indicator-level evidence
trajectories constitute the final numeric evidence collection
Enum={E k|k∈ K fine}.
U-shape time-aware retrieval for textual events.Non-
numeric clinical events (e.g., diagnoses, procedures, and
clinical notes) are serialized at the event level and segmented
into overlapping temporal chunks, with each chunk embed-
ded and indexed in a vector store. Given a task-specific
query qtaskand prediction time τ∗, we first retrieve a can-
didate pool of Kcandtextual chunks using dense semantic
similarity search, and then jointly consider semantic rele-
vance and temporal proximity to select textual evidence.
Prior work has explored incorporating temporal informa-
tion to improve retrieval quality, for example by introducing
time-aware relevance weighting or decay functions (Cao
et al., 2024b; Abdallah et al., 2025). However, in clinical set-
tings, temporal importance is not strictly monotonic: while
recent events are often critical, early events corresponding
to disease onset or initial admission can be equally infor-
mative. This observation motivates a U-shaped time-aware
retrieval strategy that explicitly emphasizes both recent and
early clinical evidence.
For each candidate textual chunk cwith timestamp τc,
we compute a semantic similarity score ssem(qtask, c)us-
ing dense embeddings, together with a U-shaped temporal
relevance score. Let τfirstdenote the earliest timestamp inthe patient record. The temporal relevance score is defined
as
stime(τc) = max
exp
−τ∗−τc
τrecent
,exp
−τc−τ first
τearly
.
(2)
This U-shaped formulation assigns higher importance to
events occurring close to the prediction time τ∗or near the
beginning of the patient trajectory, while downweighting
mid-history events that are less informative for the current
prediction. The final retrieval score for each textual chunk
is computed as a convex combination of semantic relevance
and temporal importance,
s(c) =α s sem(qtask, c) + (1−α)s time(τc),(3)
where α∈[0,1] controls the trade-off between semantic
similarity and temporal relevance. Candidate chunks are
ranked by s(c), and the top- Kfinalchunks are selected and
temporally ordered to form the final textual evidence set
Etext={c 1, . . . , c Kfinal}.
By combining indicator-level numeric evidence Enumwith
U-shaped time-aware textual evidence Etext,ETHERensures
broad semantic coverage while preserving event-type dis-
tinctions and clinically meaningful temporal structure under
a compact evidence budget.
3.3. Adaptive Iterative Retrieval
Single-pass retrieval is often insufficient for long-horizon
EHRs, where relevant clinical evidence may be temporally
dispersed or only indirectly related to the initial query. To
address this limitation, we progressively expands evidence
coverage in a controlled and targeted manner.
Starting from an initial query q(0)=q task, we retrieve an
initial evidence set E(0). At each iteration t, the LLM as-
sesses whether the current evidence set E(t)is sufficient
to answer the clinical prediction task. If so, the retrieval
process terminates. Otherwise, the LLM generates a refined
query
q(t+1)=MR
q(t),E(t)
,(4)
where MR(·)denotes an LLM-based query refinement
module that identifies a single missing yet clinically salient
aspect of the current evidence. Each refined query is ex-
plicitly constrained to be concise, focused on one clinical
dimension, and non-redundant with previously retrieved
information.
Evidence retrieved using the refined query is then merged
with the existing context via deduplication and temporal
ordering:
E(t+1)= Merge
E(t),Retrieve(q(t+1))
.(5)
4

EHR-RAG
This iterative process continues until the evidence is deemed
sufficient or a predefined iteration limit is reached. By incre-
mentally expanding the evidence set in a targeted manner,
AIR improves retrieval recall while preventing uncontrolled
context growth, which is critical for effective long-horizon
clinical reasoning.
3.4. Dual-Path Evidence Retrieval and Reasoning
Clinical prediction often involves competing hypothe-
ses (Sox et al., 2024; Elstein et al., 1978), and reasoning
along a single evidence pathway can lead to biased or over-
confident conclusions (Graber et al., 2005; Croskerry, 2003).
To improve robustness, we propose a dual-path evidence
retrieval and reasoning strategy.
Specifically, we construct two complementary retrieval
queries: afactual (positive) query q+that seeks evidence
supporting the presence of the target outcome, and acoun-
terfactual (negative) query q−that seeks evidence sup-
porting its absence. Each query is processed independently
using the adaptive iterative retrieval mechanism, yielding
two textual evidence sets, E+
textandE−
text. For each path, the
retrieved textual evidence is combined with the shared nu-
meric evidence Enum, and the LLM is prompted to form an
explicit outcome hypothesis:
h+=M 
E+
text∪ E num
, h−=M 
E−
text∪ E num
,(6)
where h+andh−denote the positive and negative outcome
hypotheses, respectively. The corresponding evidence sets
are then fused into a unified evidence context:
Efuse=E+
text∪ E−
text∪ E num.(7)
Finally, the model is explicitly prompted to compare the
strength,directness, andclinical relevanceof evidence sup-
porting each outcome hypothesis, and to produce a final
prediction conditioned on both hypotheses and the fused
evidence:
ˆy=M dec 
Efuse, h+, h−
,(8)
where Mdec(·)denotes the LLM-based decision function
that performs comparative hypothesis evaluation and outputs
the final clinical prediction.
Overall, this dual-path design enables balanced hypothesis
evaluation, mitigates confirmation bias and spurious cor-
relations, and improves the robustness and reliability of
long-horizon clinical prediction from structured EHR data.
4. Experiments
4.1. Experimental Setup
To systematically evaluate our proposed method and com-
pare it against baseline approaches under long-horizon clin-
ical prediction settings, we conduct all experiments on theEHRSHOT benchmark (Wornow et al., 2023). Unlike com-
monly used EHR dataset, such as MIMIC-III/IV (Johnson
et al., 2016; 2023), which are largely restricted to ICU or
emergency department settings, EHRSHOT captures lon-
gitudinal patient records across general hospital care. As
a result, patient records in EHRSHOT often span multiple
decades and contain thousands of clinical events. On av-
erage, EHRSHOT includes 2.3× more clinical events and
95.2× more encounters per patient than MIMIC-IV , mak-
ing it substantially more challenging and better suited for
evaluating long-horizon clinical reasoning.
We select four representative prediction tasks from
EHRSHOT:Long Length of Stay,30-day Readmission,
Acute Myocardial Infarction (Acute MI), andAnemia. The
first two tasks fall under theOperational Outcomescategory,
Anemia belongs toAnticipating Lab Test Results, and Acute
MI is categorized asAssignment of New Diagnoses. The first
three tasks are binary classification problems, while Anemia
is a four-class classification task. Together, these tasks span
diverse clinical objectives and temporal reasoning scenar-
ios, enabling a comprehensive evaluation of long-horizon
reasoning on structured EHR data. We report Accuracy,
Macro-F1, and per-class F1 scores, as most clinical classifi-
cation tasks are inherently imbalanced.
We conduct experiments using three LLMs that span both
proprietary and open-source settings:GPT-5,Claude-Opus-
4.5, andLLaMA-3.1-8B. These models cover a broad range
of architectures, training scales, model sizes, and accessibil-
ity levels, allowing us to assess the robustness ofEHR-RAG
across different LLM backbones. Additional details of the
experimental setup are provided in Appendix A .
4.2. Baselines
We compareEHR-RAGagainst commonly used LLM-based
baselines for EHR data, including direct generation and
diverse RAG approaches.
•Direct Generation(Lin et al., 2025): The LLM directly
predicts the clinical outcome from the patient EHR
without retrieval. To satisfy context constraints, the
EHR is truncated by retaining the most recent events
up to the maximum context window, a common heuris-
tic in prior LLM-based EHR studies. This baseline
evaluates the performance of naive context truncation
without evidence selection.
•RAG(Gao et al., 2023): A vanilla retrieval approach
that retrieves relevant EHR evidence and conditions the
LLM on the retrieved context. This baseline reflects
the most common single-pass RAG formulation used
in recent clinical and non-clinical applications.
•Uniform RAG(Liu et al., 2024): A retrieval baseline
that uniformly samples EHR evidence without seman-
5

EHR-RAG
Table 1.Performance ofEHR-RAGand other LLM-based baselines onLong Length of Stayand30-day ReadmissionusingGPT-5. We
report Accuracy (%), Macro-F1 (%), and per-class F1 scores (%).Boldindicates the best performance, and improvements over the
strongest baseline are highlighted in green.EHR-RAGconsistently outperforms other methods on both tasks.
MethodLong Length of Stay 30-day Readmission
Accuracy Macro F1 F1 Short F1Long Accuracy Macro F1 F1 No Readmit F1Readmit
Direct Generation 69.41 66.52 76.36 56.67 46.56 44.37 55.41 33.33
RAG 68.24 64.77 75.82 53.71 45.80 43.40 55.06 31.73
Uniform RAG 65.10 61.50 73.27 49.72 52.67 49.63 62.65 35.42
Rule-based RAG 63.53 56.58 73.95 39.22 57.63 44.08 71.61 16.54
ReAct RAG 65.10 60.85 73.75 47.95 48.09 45.22 57.76 32.67
EHR-RAG (Ours) 74.12 +4.71 70.15 +3.63 81.03 +4.67 59.26 +2.59 71.76 +14.13 60.91 +11.28 81.50 +9.89 40.32 +4.90
Table 2.Performance ofEHR-RAGand other LLM-based baselines onAcute Myocardial InfarctionandAnemiausingGPT-5. We report
Accuracy (%), Macro-F1 (%), and per-class F1 scores (%).Boldindicates the best performance, and improvements over the strongest
baseline are highlighted in green.EHR-RAGconsistently outperforms other methods on both tasks.
MethodAcute MI Anemia
Accuracy Macro F1 F1 No MI F1MI Accuracy Macro F1 F1 Low F1Medium F1High F1Abnormal
Direct Generation 88.38 57.97 93.72 22.22 44.57 28.38 70.74 17.30 18.48 7.02
RAG 90.04 59.83 94.67 25.00 42.08 20.44 70.43 18.85 12.90 0.00
Uniform RAG 89.21 47.15 94.30 0.00 48.42 32.39 72.65 26.97 22.68 7.27
Rule-based RAG 91.29 52.06 95.42 8.70 44.34 26.21 67.61 17.33 16.30 3.57
ReAct RAG 89.21 56.49 94.22 18.75 46.15 31.64 69.37 27.23 22.68 7.27
EHR-RAG (Ours) 92.95 +1.66 76.29 +16.46 96.16 +0.74 56.41 +31.41 57.01 +8.59 44.07 +11.68 80.31 +7.66 43.00 +15.77 43.59 +20.91 9.38 +2.11
tic prioritization. This baseline controls for context
length and isolates the benefit of relevance-aware re-
trieval over random selection.
•Rule-based RAG(Liao et al., 2025): A heuristic re-
trieval approach that selects EHR evidence using pre-
defined rules rather than learned relevance signals. In
our implementation, events are ranked by occurrence
frequency, and the top frequently occurring events are
selected. This baseline reflects commonly adopted
heuristic filtering strategies for handling long EHR
sequences.
•ReAct RAG(Yao et al., 2022): A reasoning-and-acting
framework that interleaves reasoning steps with re-
trieval actions. The LLM iteratively generates reason-
ing traces and retrieval queries to obtain additional
EHR evidence, evaluating whether generic iterative re-
trieval improves long-horizon EHR reasoning without
domain-specific design.
All baselines use the same underlying LLM backbone and
context budget asEHR-RAGto ensure a fair and controlled
comparison.
Additionally, in Section 4.6, we compareEHR-RAGwith
classical ML baselines following the EHRSHOT benchmark
to assess whether LLM-based reasoning offers benefits be-
yond classical EHR models:
•Count-based LR(Wornow et al., 2023): Logistic regres-
sion trained on count-based clinical features extracted
from the EHR.•CLMBR-based LR(Wornow et al., 2023): Logis-
tic regression trained on patient representations pro-
duced by the pretrainedCLMBR-T-Basefoundation
model (Wornow et al., 2023; Steinberg et al., 2021).
4.3. Main Results
Tables 1and2present the main results on the four long-
horizon clinical prediction tasks from EHRSHOT, spanning
operational outcomes, diagnosis prediction, and laboratory
abnormality assessment.
(i) Operational outcome prediction.As shown in Table 1,
EHR-RAGconsistently outperforms all LLM-based base-
lines on bothLong Length of Stayand30-day Readmission.
For length of stay prediction,EHR-RAGachieves the high-
est accuracy (74.12) and macro-F1 (70.15), with substantial
improvements over the strongest baseline across both short-
stay and long-stay classes. On the highly imbalanced read-
mission task,EHR-RAGalso yields notable improvements
in macro-F1 (60.91).
(ii) Clinical diagnosis and lab prediction.Table 2presents
results onAcute MIandAnemia. For acute MI prediction,
EHR-RAGsubstantially improves macro-F1 (76.29) and F1
on the positive MI class (56.41), indicating more effective
identification of clinically meaningful risk signals from long-
horizon EHRs. On the multi-class anemia task,EHR-RAG
achieves the best overall accuracy (57.01) and macro-F1
(44.07), with consistent improvements across all severity
levels, including the rare abnormal class.
6

EHR-RAG
Table 3.Performance comparison ofEHR-RAGand other LLM-based baselines across different LLM backbones (GPT-5,Claude-Opus-
4.5, andLLaMA-3.1-8B) on theLong Length of Staytask. We report Accuracy (%), Macro-F1 (%), and per-class F1 scores (%).Bold
indicates the best performance, and improvements over the strongest baseline are highlighted in green. The results demonstrate that the
performance gains generalize across different LLM backbones.
MethodGPT-5 Claude-Opus-4.5 LLaMA-3.1-8B
Accuracy Macro-F1 F1 Short F1Long Accuracy Macro-F1 F1 Short F1Long Accuracy Macro-F1 F1 Short F1Long
Direct Generation 69.41 66.52 76.36 56.67 39.22 38.98 35.15 42.80 39.22 38.98 35.15 42.80
RAG 68.24 64.77 75.82 53.71 49.02 48.96 50.76 47.15 49.02 48.96 50.76 47.15
Uniform RAG 65.10 61.50 73.27 49.72 47.45 47.39 49.24 45.53 47.45 47.39 49.24 45.53
Rule-based RAG 63.53 56.58 73.95 39.22 45.10 45.03 46.97 43.09 45.10 45.03 46.97 43.09
ReAct RAG 65.10 60.85 73.75 47.95 47.84 47.76 45.71 49.81 47.84 47.76 45.71 49.81
EHR-RAG (Ours) 74.12 +4.71 70.15 +3.63 81.03 +4.67 59.26 +2.59 56.08 +7.06 55.58 +6.62 60.28 +9.52 50.88 +1.07 56.08 +7.06 55.58 +6.62 60.28 +9.52 50.88 +1.07
Table 4.Ablation results on theLong Length of StayandAnemia
tasks usingGPT-5. We report Accuracy (%) and Macro-F1 (%).
Red text indicates performance drops relative toEHR-RAG.
MethodLong Length of Stay Anemia
Accuracy Macro-F1 Accuracy Macro-F1
EHR-RAG (Ours) 74.12 70.15 57.01 44.07
w/oETHER72.94 -1.18 68.47 -1.68 53.39 -3.62 40.12 -3.95
w/oAIR72.16 -1.96 68.19 -1.96 54.07 -2.94 41.44 -2.63
w/oDER70.59 -3.53 67.55 -2.60 46.83 -10.18 38.58 -5.49
Vanilla RAG 68.24 -5.88 64.77 -5.38 42.08 -14.93 20.44 -23.63
(iii) Baseline analysis.We observe that vanilla RAG can
underperform direct generation in some cases, suggesting
that retrieving events solely based on semantic similarity,
without explicitly modeling recency and temporal context,
may omit clinically critical recent evidence and lead to
degraded performance. Uniform RAG exhibits unstable be-
havior, performing well in some cases but poorly in others,
suggesting that while information across the patient history
contains useful signals, indiscriminate retrieval lacks robust-
ness. Rule-based RAG yields moderate improvements but
introduces systematic bias by prioritizing high-frequency
events, which can skew predictions toward dominant classes.
ReAct RAG occasionally provides marginal gains over RAG
by enabling iterative retrieval and broader evidence cover-
age; however, it does not consistently translate additional
retrieval steps into reliable performance improvements.
(iv) Overall comparison.Across all four tasks,EHR-RAG
consistently outperforms direct generation, vanilla RAG,
and other retrieval baselines. These results highlight the
importance of jointly modeling event-type structure, tem-
poral dynamics, and evidence diversity through event- and
time-aware retrieval, adaptive iterative refinement, and dual-
path evidence retrieval and reasoning for accurate and reli-
able long-horizon clinical prediction from structured EHR
data. Overall, these findings demonstrate that effective
long-horizon EHR reasoning requires structured, tempo-
rally aware, and diversity-preserving retrieval mechanisms
rather than generic or heuristic RAG strategies. Appendix B
presents a qualitative case study comparingEHR-RAGwith
direct generation and vanilla RAG.4.4. Ablation Analysis of Key Components
Table 4reports the ablation results onLong Length of Stay
andAnemia, analyzing the contribution of each core com-
ponent ofEHR-RAG. Red subscripts indicate the absolute
performance drop relative to the full framework.
(i) Overall impact of each component.ExcludingEvent-
and Time-Aware Hybrid EHR Retrieval (ETHER)leads to
clear drops, particularly onAnemia, highlighting the im-
portance of preserving event-type distinctions and temporal
structure when retrieving long-horizon clinical evidence.
RemovingAdaptive Iterative Retrieval (AIR)results in uni-
form declines across all metrics, suggesting that iterative
query refinement is critical for recovering sparse but task-
relevant evidence distributed over time. Finally, ablating
Dual-Path Evidence Retrieval Reasoning (DER)causes the
largest Macro-F1 degradation onAnemia, underscoring the
value of explicitly modeling factual and counterfactual re-
trieval and reasoning in complex and multi-class clinical
prediction settings. Overall, the full framework achieves the
best performance across all metrics, while each ablated vari-
ant exhibits systematic performance drops, demonstrating
that all components ofEHR-RAGcontribute meaningfully
and synergistically to the overall gains.
(ii) Comparison with vanilla RAG.Vanilla RAG performs
substantially worse than allEHR-RAGvariants, with espe-
cially severe degradation onAnemia(14.93 accuracy and
23.63 Macro-F1 points). This confirms that naive retrieval
alone is insufficient for long-horizon structured EHR reason-
ing, and motivates the need for the proposed event-aware,
iterative, and dual-path design.
4.5. Performance Across Different LLM Backbones
Table 3compares the performance of different LLM back-
bones on theLong Length of Staytask, including both pro-
prietary and open-source models with varying capacities.
(i) Consistent gains across models.EHR-RAGyields con-
sistent performance improvements acrossGPT-5,Claude-
Opus-4.5, andLLaMA-3.1-8B, despite substantial differ-
ences in model architecture and training scale. This con-
7

EHR-RAG
1 2 4 816 32 64128 256 512All
Training Samples per Class0.450.500.550.600.650.70Macro F1
Long Length of Stay
Count-based LR
CLMBR-based LR
Vanilla RAG
EHR-RAG
1 2 4 816 32 64128 256 512All
Training Samples per Class0.400.450.500.550.600.650.70Macro F1
30-day Readmission
1 2 4 816 32 64128 256 512All
Training Samples per Class0.350.400.450.500.550.600.650.70Macro F1
Acute MI
1 2 4 816 32 64128 256 512All
Training Samples per Class0.150.200.250.300.350.400.45Macro F1
Anemia
Figure 2.Macro-F1 performance comparison betweenEHR-RAG, vanilla RAG, and traditional machine learning (ML) baselines under
varying amounts of labeled training data across four tasks. The x-axis denotes the number of training samples per class, while dashed
horizontal lines indicate the zero-shot performance of LLM-based methods.EHR-RAGconsistently matches or outperforms ML baselines,
particularly in low-resource settings.
sistency indicates that the proposed retrieval and reasoning
framework generalizes across LLM backbones, rather than
relying on the characteristics of any single model.
(ii) Larger gains on smaller models.WhileGPT-5
achieves the highest absolute performance, the relative im-
provements introduced byEHR-RAGare more pronounced
forLLaMA-3.1-8B. Compared to the strongest baseline,
EHR-RAGimproves Macro-F1 by over 6 points on both
models, suggesting that its design can partially mitigate the
limitations of smaller model capacity.
4.6. Comparison with Traditional ML Baselines
Figure 2compares the Macro-F1 performance ofEHR-RAG
with ML baselines across varying levels of labeled data
availability, ranging from few-shot to full-data regimes. The
x-axis denotes the number of training samples per class.
LLM-based methods are evaluated in a zero-shot setting and
are therefore shown as horizontal reference lines.
(i) Performance under low-data regimes.Across all tasks,
EHR-RAGoutperforms ML baselines in most low-shot set-
tings. When only a limited number of labeled samples per
class are available (e.g., 1 to 16 shots), both count-based
and CLMBR-based LR models experience substantial per-
formance degradation. In contrast,EHR-RAGmaintains
relatively stable performance by leveraging pretrained LLM
knowledge together with retrieved EHR evidence, highlight-
ing its strong data efficiency in low-resource settings.
(ii) Comparison with representation-based ML mod-
els.As expected, CLMBR-based LR generally outper-
forms count-based features, particularly as the number of
training samples increases, reflecting the advantage of pre-
trained foundation models in capturing general clinical rep-
resentations. However, even in medium- and full-data set-
tings,EHR-RAGachieves comparable or superior Macro-
F1 scores across all tasks. This suggests that retrieval-
augmented LLM reasoning can effectively integrate hetero-
geneous and temporally distributed clinical evidence beyondwhat is captured by fixed patient representations.
(iii) Robustness on long-horizon and imbalanced tasks.
The advantages ofEHR-RAGare most pronounced on long-
horizon and class-imbalanced tasks, such asAcute MIand
Anemia. In these settings, ML baselines struggle to capture
sparse yet clinically meaningful signals distributed over
extended patient histories. In contrast,EHR-RAGleverages
strong generalization and zero-shot reasoning capabilities
to surface and aggregate relevant evidence, leading to more
robust performance.
Overall, these results demonstrate thatEHR-RAGnot only
outperforms traditional ML baselines in low-resource set-
tings, but also remains competitive or superior even when
large amounts of labeled data are available, underscoring
its effectiveness as a general-purpose framework for long-
horizon EHR prediction.
5. Conclusion
In this paper, we introduceEHR-RAG, an enhanced retrieval-
augmented framework designed to bridge long-horizon
EHRs and LLMs.EHR-RAGaddresses key challenges
posed by long-horizon EHR data, including context win-
dow limitations, temporal fragmentation, and incomplete
evidence retrieval, through a combination of event- and time-
aware hybrid EHR retrieval, adaptive iterative refinement,
and dual-path evidence retrieval and reasoning. Extensive
experiments demonstrate thatEHR-RAGconsistently out-
performs other LLM-based baselines across diverse long-
horizon clinical prediction tasks. Overall, our results show
that carefully designed retrieval and reasoning mechanisms
can enable LLMs to effectively interpret long-horizon struc-
tured EHR data even under strict context constraints. We
believeEHR-RAGrepresents a practical step toward more
reliable and data-efficient clinical prediction systems, and
provides a foundation for future work on scalable, evidence-
grounded reasoning over longitudinal healthcare data.
8

EHR-RAG
Impact Statement
This work aims to advance large language model for rea-
soning over long-horizon electronic health records through
retrieval-augmented generation. While our framework is
developed using publicly accessible, de-identified data, we
acknowledge that work involving clinical information raises
considerations around privacy, fairness, and the appropri-
ate use of AI-assisted decision support. Our method is
designed as a research contribution rather than a deployed
clinical system, and it does not make autonomous treatment
recommendations. Instead, it seeks to improve technical
robustness in temporal retrieval and reasoning, which may
support future development of safer and more interpretable
clinical AI tools.
References
Abdallah, A., Piryani, B., Wallat, J., Anand, A., and Ja-
towt, A. Extending dense passage retrieval with temporal
information.arXiv e-prints, pp. arXiv–2502, 2025.
Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I.,
Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S.,
Anadkat, S., et al. Gpt-4 technical report.arXiv preprint
arXiv:2303.08774, 2023.
Asai, A., Wu, Z., Wang, Y ., Sil, A., and Hajishirzi, H. Self-
rag: Learning to retrieve, generate, and critique through
self-reflection.arXiv preprint arXiv:2310.11511, 2023.
Ashfaq, A., Sant’Anna, A., Lingman, M., and Nowaczyk, S.
Readmission prediction using deep learning on electronic
health records.Journal of biomedical informatics, 97:
103256, 2019.
Bedi, S., Liu, Y ., Orr-Ewing, L., Dash, D., Koyejo, S.,
Callahan, A., Fries, J. A., Wornow, M., Swaminathan, A.,
Lehmann, L. S., et al. Testing and evaluation of health
care applications of large language models: a systematic
review.Jama, 2025.
Bhoi, S., Lee, M. L., Hsu, W., Fang, H. S. A., and Tan,
N. C. Personalizing medication recommendation with a
graph-based approach.ACM Transactions on Information
Systems (TOIS), 40(3):1–23, 2021.
Cai, X., Perez-Concha, O., Coiera, E., Martin-Sanchez, F.,
Day, R., Roffe, D., and Gallego, B. Real-time predic-
tion of mortality, readmission, and length of stay using
electronic health record data.Journal of the American
Medical Informatics Association, 23(3):553–561, 2016.
Cao, L., Sun, J., and Cross, A. Autord: An automatic and
end-to-end system for rare disease knowledge graph con-
struction based on ontologies-enhanced large language
models.arXiv preprint arXiv:2403.00953, 2024a.Cao, L., Wang, Z., Xiao, C., and Sun, J. PILOT: Le-
gal case outcome prediction with case law. In Duh,
K., Gomez, H., and Bethard, S. (eds.),Proceedings of
the 2024 Conference of the North American Chapter of
the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers), pp.
609–621, Mexico City, Mexico, June 2024b. Associa-
tion for Computational Linguistics. doi: 10.18653/v1/
2024.naacl-long.34. URL https://aclanthology.
org/2024.naacl-long.34/.
Chen, C., Yu, J., Chen, S., Liu, C., Wan, Z., Bitterman,
D., Wang, F., and Shu, K. Clinicalbench: Can llms
beat traditional ml models in clinical prediction?arXiv
preprint arXiv:2411.06469, 2024.
Choi, E., Bahadori, M. T., Sun, J., Kulas, J., Schuetz, A., and
Stewart, W. Retain: An interpretable predictive model
for healthcare using reverse time attention mechanism.
Advances in neural information processing systems, 29,
2016.
Choi, E., Bahadori, M. T., Song, L., Stewart, W. F., and
Sun, J. Gram: graph-based attention model for health-
care representation learning. InProceedings of the 23rd
ACM SIGKDD international conference on knowledge
discovery and data mining, pp. 787–795, 2017.
Cowie, M. R., Blomster, J. I., Curtis, L. H., Duclaux, S.,
Ford, I., Fritz, F., Goldman, S., Janmohamed, S., Kreuzer,
J., Leenay, M., et al. Electronic health records to facilitate
clinical research.Clinical Research in Cardiology, 106
(1):1–9, 2017.
Croskerry, P. The importance of cognitive errors in diagnosis
and strategies to minimize them.Academic medicine, 78
(8):775–780, 2003.
Cui, H., Shen, Z., Zhang, J., Shao, H., Qin, L., Ho, J. C.,
and Yang, C. Llms-based few-shot disease predictions
using ehr: A novel approach combining predictive agent
reasoning and critical agent instruction. InAMIA Annual
Symposium Proceedings, volume 2024, pp. 319, 2025.
Ebbers, T., Kool, R. B., Smeele, L. E., Dirven, R., den
Besten, C. A., Karssemakers, L. H., Verhoeven, T., Her-
ruer, J. M., van den Broek, G. B., and Takes, R. P. The
impact of structured and standardized documentation on
documentation quality; a multicenter, retrospective study.
Journal of Medical Systems, 46(7):46, 2022.
Elstein, A. S., Shulman, L. S., and Sprafka, S. A.Medi-
cal problem solving: An analysis of clinical reasoning.
Harvard University Press, 1978.
9

EHR-RAG
Fries, J. A., Wornow, M., Steinberg, E., Huo, Z. F., Cui,
H., Bedi, S., Unell, A., and Shah, N. Advancing respon-
sible healthcare ai with longitudinal ehr datasets, 2025.
Stanford HAI News.
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y ., Dai, Y .,
Sun, J., Wang, H., and Wang, H. Retrieval-augmented
generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2(1), 2023.
Graber, M. L., Franklin, N., and Gordon, R. Diagnostic
error in internal medicine.Archives of internal medicine,
165(13):1493–1499, 2005.
Hegselmann, S., von Arnim, G., Rheude, T., Kronenberg,
N., Sontag, D., Hindricks, G., Eils, R., and Wild, B. Large
language models are powerful electronic health record
encoders.arXiv preprint arXiv:2502.17403, 2025.
Jiang, P., Xiao, C., Cross, A., and Sun, J. Graphcare: En-
hancing healthcare predictions with personalized knowl-
edge graphs.arXiv preprint arXiv:2305.12788, 2023.
Jiang, P., Xiao, C., Jiang, M., Bhatia, P., Kass-Hout, T., Sun,
J., and Han, J. Reasoning-enhanced healthcare predic-
tions with knowledge graph community retrieval.arXiv
preprint arXiv:2410.04585, 2024.
Jiang, P., Cao, L., Zhu, R., Jiang, M., Zhang, Y ., Sun,
J., and Han, J. Ras: Retrieval-and-structuring for
knowledge-intensive llm generation.arXiv preprint
arXiv:2502.10996, 2025a.
Jiang, P., Lin, J., Cao, L., Tian, R., Kang, S., Wang, Z.,
Sun, J., and Han, J. Deepretrieval: Hacking real search
engines and retrievers with large language models via re-
inforcement learning.arXiv preprint arXiv: 2503.00223,
2025b.
Johnson, A. E., Pollard, T. J., Shen, L., Lehman, L.-w. H.,
Feng, M., Ghassemi, M., Moody, B., Szolovits, P., An-
thony Celi, L., and Mark, R. G. Mimic-iii, a freely ac-
cessible critical care database.Scientific data, 3(1):1–9,
2016.
Johnson, A. E., Bulgarelli, L., Shen, L., Gayles, A., Sham-
mout, A., Horng, S., Pollard, T. J., Hao, S., Moody, B.,
Gow, B., et al. Mimic-iv, a freely accessible electronic
health record dataset.Scientific data, 10(1):1, 2023.
Kim, H., Sohn, J., Gilson, A., Cochran-Caggiano, N., Ap-
plebaum, S., Jin, H., Park, S., Park, Y ., Park, J., Choi,
S., et al. Rethinking retrieval-augmented generation for
medicine: A large-scale, systematic expert evaluation
and practical insights.arXiv preprint arXiv:2511.06738,
2025.Kojima, T., Gu, S. S., Reid, M., Matsuo, Y ., and Iwasawa,
Y . Large language models are zero-shot reasoners.Ad-
vances in neural information processing systems, 35:
22199–22213, 2022.
Kruse, M., Hu, S., Derby, N., Wu, Y ., Stonbraker, S., Yao,
B., Wang, D., Goldberg, E., and Gao, Y . Large language
models with temporal reasoning for longitudinal clinical
summarization and prediction. InFindings of the Associ-
ation for Computational Linguistics: EMNLP 2025, pp.
20715–20735, 2025.
Levin, S., Barnes, S., Toerper, M., Debraine, A., DeAn-
gelo, A., Hamrock, E., Hinson, J., Hoyer, E., Dungarani,
T., and Howell, E. Machine-learning-based hospital dis-
charge predictions can support multidisciplinary rounds
and decrease hospital length-of-stay.BMJ Innovations, 7
(2), 2021.
Li, H., Deng, B., Xu, C., Feng, Z., Schlegel, V ., Huang, Y .-
H., Sun, Y ., Sun, J., Yang, K., Yu, Y ., et al. Mira: Medical
time series foundation model for real-world health data.
arXiv preprint arXiv:2506.07584, 2025a.
Li, Z., Yu, H., Guo, G., Zhou, N., and Zhang, J. Muisqa:
Multi-intent retrieval-augmented generation for scientific
question answering.arXiv preprint arXiv:2511.16283,
2025b.
Liao, Y ., Wu, C., Liu, J., Jiang, S., Qiu, P., Wang, H., Yue,
Y ., Zhen, S., Wang, J., Fan, Q., et al. Ehr-r1: A reasoning-
enhanced foundational language model for electronic
health record analysis.arXiv preprint arXiv:2510.25628,
2025.
Lin, J., Wu, Z., and Sun, J. Training llms for ehr-based
reasoning tasks via reinforcement learning.arXiv preprint
arXiv:2505.24105, 2025.
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua,
M., Petroni, F., and Liang, P. Lost in the middle: How
language models use long contexts.Transactions of the
association for computational linguistics, 12:157–173,
2024.
Loh, D. R., Hill, E. D., Liu, N., Dawson, G., and Engel-
hard, M. M. Limitations of binary classification for long-
horizon diagnosis prediction and advantages of a discrete-
time time-to-event approach: Empirical analysis.JMIR
AI, 4:e62985, 2025.
Lovon-Melgarejo, J., Ben-Haddi, T., Di Scala, J., Moreno,
J. G., and Tamine, L. Revisiting the mimic-iv benchmark:
Experiments using language models for electronic health
records. InProceedings of the First Workshop on Patient-
Oriented Language Processing (CL4Health)@ LREC-
COLING 2024, pp. 189–196, 2024.
10

EHR-RAG
Menachemi, N. and Collum, T. H. Benefits and drawbacks
of electronic health record systems.Risk management
and healthcare policy, pp. 47–55, 2011.
Neha, F., Bhati, D., and Shukla, D. K. Retrieval-augmented
generation (rag) in healthcare: A comprehensive review.
AI, 6(9):226, 2025.
Niu, S., Ma, J., Bai, L., Wang, Z., Guo, L., and Yang, X. Ehr-
knowgen: Knowledge-enhanced multimodal learning for
disease diagnosis generation.Information Fusion, 102:
102069, 2024.
Perçin, S., Su, X., Syed, Q. S., Howard, P., Kuvshinov, A.,
Schwinn, L., and Scholl, K.-U. Investigating the robust-
ness of retrieval-augmented generation at the query level.
InProceedings of the Fourth Workshop on Generation,
Evaluation and Metrics (GEM2), pp. 439–457, 2025.
Rosenbloom, S. T., Denny, J. C., Xu, H., Lorenzi, N., Stead,
W. W., and Johnson, K. B. Data from clinical notes: a
perspective on the tension between structure and flexi-
ble documentation.Journal of the American Medical
Informatics Association, 18(2):181–186, 2011.
Shang, J., Xiao, C., Ma, T., Li, H., and Sun, J. Gamenet:
Graph augmented memory networks for recommending
medication combination. Inproceedings of the AAAI Con-
ference on Artificial Intelligence, volume 33, pp. 1126–
1133, 2019.
Shi, W., Xu, R., Zhuang, Y ., Yu, Y ., Zhang, J., Wu, H., Zhu,
Y ., Ho, J. C., Yang, C., and Wang, M. D. Ehragent: Code
empowers large language models for few-shot complex
tabular reasoning on electronic health records. InPro-
ceedings of the 2024 Conference on Empirical Methods in
Natural Language Processing, pp. 22315–22339, 2024.
Soman, K., Rose, P. W., Morris, J. H., Akbas, R. E., Smith,
B., Peetoom, B., Villouta-Reyes, C., Cerono, G., Shi, Y .,
Rizk-Jackson, A., et al. Biomedical knowledge graph-
optimized prompt generation for large language models.
Bioinformatics, 40(9):btae560, 2024.
Sox, H. C., Higgins, M. C., Owens, D. K., and Schmidler,
G. S.Medical decision making. John Wiley & Sons,
2024.
Steinberg, E., Jung, K., Fries, J. A., Corbin, C. K., Pfohl,
S. R., and Shah, N. H. Language models are an effective
representation learning technique for electronic health
record data.Journal of biomedical informatics, 113:
103637, 2021.
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux,
M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E.,
Azhar, F., et al. Llama: Open and efficient foundation lan-
guage models.arXiv preprint arXiv:2302.13971, 2023.Tramontini, D. L., Ghosh, S., and Eickhoff, C. Investi-
gating rag-based approaches in clinical trial and patient
matching.Machine Learning for Health 2025, 2025.
V orontsov, E., Bozkurt, A., Casson, A., Shaikovski, G.,
Zelechowski, M., Severson, K., Zimmermann, E., Hall,
J., Tenenholtz, N., Fusi, N., et al. A foundation model for
clinical-grade computational pathology and rare cancers
detection.Nature medicine, 30(10):2924–2935, 2024.
Wallace, E., Wang, Y ., Li, S., Singh, S., and Gardner, M.
Do nlp models know numbers? probing numeracy in
embeddings.arXiv preprint arXiv:1909.07940, 2019.
Wornow, M., Thapa, R., Steinberg, E., Fries, J., and Shah,
N. Ehrshot: An ehr benchmark for few-shot evaluation
of foundation models.Advances in Neural Information
Processing Systems, 36:67125–67137, 2023.
Xiao, C., Ma, T., Dieng, A. B., Blei, D. M., and Wang, F.
Readmission prediction via deep contextual embedding
of clinical concepts.PloS one, 13(4):e0195024, 2018.
Yang, K., Xu, Y ., Zou, P., Ding, H., Zhao, J., Wang, Y ., and
Xie, B. Kerprint: local-global knowledge graph enhanced
diagnosis prediction for retrospective and prospective
interpretations. InProceedings of the AAAI Conference
on Artificial Intelligence, volume 37, pp. 5357–5365,
2023.
Yang, X., Chen, A., PourNejatian, N., Shin, H. C., Smith,
K. E., Parisien, C., Compas, C., Martin, C., Costa, A. B.,
Flores, M. G., et al. A large language model for electronic
health records.NPJ digital medicine, 5(1):194, 2022.
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan,
K. R., and Cao, Y . React: Synergizing reasoning and
acting in language models. InThe eleventh international
conference on learning representations, 2022.
Zhang, X., Ramachandran, D., Tenney, I., Elazar, Y ., and
Roth, D. Do language embeddings capture scales?arXiv
preprint arXiv:2010.05345, 2020.
Zhu, Y ., Ren, C., Wang, Z., Zheng, X., Xie, S., Feng, J.,
Zhu, X., Li, Z., Ma, L., and Pan, C. Emerge: Enhancing
multimodal electronic health records predictive modeling
with retrieval-augmented generation. InProceedings of
the 33rd ACM International Conference on Information
and Knowledge Management, pp. 3549–3559, 2024a.
Zhu, Y ., Ren, C., Xie, S., Liu, S., Ji, H., Wang, Z., Sun, T.,
He, L., Li, Z., Zhu, X., et al. Realm: Rag-driven enhance-
ment of multimodal electronic health records analysis via
large language models.arXiv preprint arXiv:2402.07016,
2024b.
11

EHR-RAG
Zhu, Y ., Wang, Z., Gao, J., Tong, Y ., An, J., Liao, W., Harri-
son, E. M., Ma, L., and Pan, C. Prompting large language
models for zero-shot clinical prediction with structured
longitudinal electronic health record data.arXiv preprint
arXiv:2402.01713, 2024c.
12

EHR-RAG
A. Detailed Experimental Setup
Dataset Processing.We use the EHRSHOT benchmark (Wornow et al., 2023), which is available in a MEDS-compatible
format1. In the raw data, each clinical event is represented as a MEDS code. To make events interpretable to LLMs, we map
these codes to natural-language descriptions by constructing a lightweight medical ontology from the Athena vocabulary2
and a CPT subset via UMLS3. We evaluate four tasks,Long Length of Stay,30-day Readmission,Acute Myocardial
Infarction, andAnemia, whose metadata are summarized in Appendix C. For LLM inputs, we serialize each event
using a unified template: [time] type - description (value: value) ; for example, [2018-02-05
08:56:00] measurement - Blood Pressure (value: 120 mmHg).
LLM Generation Settings.We evaluate three LLMs:GPT-5( gpt-5-2025-08-07 ),Claude-Opus-4.5
(claude-opus-4-5-20251101 ), andLLaMA-3.1-8B( Meta-Llama-3.1-8B-Instruct ). All models are de-
ployed on the Azure AI platform, which complies with the PhysioNet Credentialed Data Use Agreement and responsible
use guidelines for clinical data4. For all LLM-based methods, we set the generation temperature to 0 to ensure stable
and deterministic outputs, except forGPT-5, which requires a temperature of 1. To ensure computational efficiency and
fair comparison across models, we fix the maximum context length to 128,000 tokens and keep all other generation
hyperparameters at their default values. All prompt templates used in our experiments are provided in Appendix D .
ML Baseline Training Settings.The training setups for bothCount-based LRandCLMBR-based LRstrictly follow prior
work (Wornow et al., 2023), including model architectures and hyperparameter settings. For each task, we train separate
models under varying data regimes with 1, 2, 4, 8, 16, 32, 64, 128, 256 shots per class, as well as using the full training set.
The pretrainedCLMBR-T-Basefoundation model (Wornow et al., 2023; Steinberg et al., 2021) is used to generate patient
representations for the CLMBR-based LR baseline. All machine learning baselines are trained and evaluated on a single
NVIDIA A800 GPU with 80GB of memory.
Baseline Retrieval Settings.ForDirect Generation, we truncate each patient EHR by retaining the most recent 1,000 events.
ForRule-based RAG, we first rank all events by the frequency of their associated event codes within the patient history, and
then select the top 1,000 most frequently occurring events as retrieved evidence. For all other RAG-based baselines, we
adopt dense retrieval using text-embedding-3-small from Azure OpenAI as a lightweight text embedding model.
Each patient EHR is segmented into chunks of 100 event rows with an overlap of 5 rows between adjacent chunks. For
vanilla RAG methods, we retrieve the top 10 chunks per query, while keeping the total number of retrieved event rows
comparable across methods.Uniform RAGrandomly samples chunks and concatenates them until the same event budget is
reached, serving as a relevance-agnostic control. ForReAct RAG, we retrieve the top 5 chunks per query over 3 iterative
retrieval steps, accounting for potential duplicate retrievals across iterations. All retrieval pipelines are implemented using
LangChain5, and cosine similarity between query and document embeddings is used as the retrieval scoring function.
EHR-RAG Settings.For our proposed method,EHR-RAG, we use the same dense text embedding model as the RAG
baselines to ensure a fair comparison. The basic retrieval configuration followsReAct RAG: at each iteration, we retrieve the
top-Kfinal=5textual chunks per query over three iterative retrieval steps. In the U-shaped time-aware retrieval component,
we set the semantic–temporal trade-off weight to α= 0.75 . The recent temporal window is defined as τrecent= 180 days,
while the early-history window spans up to τearly= 3,650 days (approximately 10 years). For textual retrieval, an initial
candidate pool of Kcand=100 chunks is retrieved before temporal re-ranking, from which the final evidence set Etextis
selected. For numeric event retrieval, we adopt a two-stage indicator selection strategy with a coarse-grained top- Ncoarse=30
followed by a fine-grained top- Nfine=10. For each selected indicator, we retain the Nrecent=5most recent measurements to
construct the numeric evidence set Enum. We observe that the overall performance ofEHR-RAGis not highly sensitive to
these hyperparameters; they can be tuned intuitively within a reasonable range without significantly affecting results.
1https://github.com/Medical-Event-Data-Standard
2https://athena.ohdsi.org/vocabulary/list
3https://uts.nlm.nih.gov/uts/
4https://physionet.org/news/post/gpt-responsible-use
5https://www.langchain.com
13

EHR-RAG
B. Case Study
We present a detailed representative case study to illustrate the effectiveness ofEHR-RAGin clinical prediction, with a
particular focus on its dual-path reasoning capability for balancing contradictory clinical evidence. This case demonstrates
howEHR-RAGavoids false alarms by systematically integrating both risk factors and stabilizing clinical signals, leading to
more calibrated and reliable predictions. These results are generated usingGPT-5on theLong Length of Staytask.
Ethical and privacy note.This case study is based on de-identified EHRSHOT data and does not include any personally
identifiable information. It is intended solely to qualitatively illustrate retrieval and reasoning behavior, rather than to provide
clinical guidance or diagnostic recommendations.
B.1. Clinical Scenario
Patient:Female, 56 years old; BMI 43.0
Admission Date:2014-12-05 20:00 (postoperative admission after intracranial aneurysm clipping)
Prediction Time:2014-12-05 23:59
Ground Truth Outcome:LOS = 1 day (discharged 2014-12-06)
Task:Predict whether long LOS (≥7 days)
Clinical Context.Initially admitted on 2014-11-27 for subarachnoid hemorrhage (SAH) due to ruptured intracranial
aneurysm; discharged 2014-11-28 after conservative management. Readmitted on 2014-12-04 for elective intracranial
aneurysm clipping. The microsurgical procedure was completed successfully, followed by routine postoperative monitoring
in the neurological intensive care unit.
Baseline Comorbidities.Type 2 diabetes mellitus / Hypertension / Hyperlipidemia / Class III obesity (BMI 43.0) / Tobacco
use
Model Predictions and Outcomes.
•EHR-RAG:Prediction = 0 (Correct).Rationale:Balances surgical risk factors with stabilizing postoperative signals,
including rapid neurologic recovery and early clinical stability.
•Direct Generation:Prediction = 1 (Incorrect).Rationale:Overemphasizes surgical complexity and baseline
comorbidities, overestimating prolonged hospitalization risk.
•Vanilla RAG:Prediction = 1 (Incorrect).Rationale:Retrieved evidence is dominated by high-risk features without
incorporating countervailing stabilization indicators, yielding a biased risk assessment.
B.2. Event- and Time-aware Hybrid EHR Retrieval inEHR-RAG
TheEvent- and Time-aware Hybrid EHR Retrieval(ETHER) module first performs knowledge-guided coarse retrieval to
collect candidate clinical indicators, then applies LLM-based reranking to select task-relevant indicators for downstream
reasoning.
B.2.1. COARSERETRIEVAL
Using medical priors for postoperative neurologic and respiratory monitoring, ETHER retrieves candidates spanning:
neurologic status (e.g., GCS), respiratory parameters (e.g., PEEP, FiO 2, airway pressure), arterial blood gases (pH, PaCO 2,
PaO 2), hemodynamics (blood pressure), and coagulation markers (INR, aPTT), as well as additional candidates including
vasopressor use, ventilation settings, pain, and sedation metrics.
B.2.2. LLM-BASEDINDICATORRERANKING
The LLM reranks these candidates and selects the top-10 indicators:
1. Glasgow Coma Scale (GCS)
2. Arterial blood gas measures (pH, PaCO 2, PaO 2)
3. Fraction of inspired oxygen (FiO 2)
14

EHR-RAG
4. Mean airway pressure and PEEP
5. Central venous pressure (CVP)
6. Diastolic blood pressure (DBP)
7. International Normalized Ratio (INR)
B.2.3. RETRIEVEDCLINICALEVIDENCE
Hemodynamics and Vital Signs.Stable SBP 98–130 mmHg, DBP 57–64 mmHg, and HR 81–89 bpm throughout the
postoperative period; no vasopressor support required. Mean arterial pressure is consistently maintained within a normal
range.
Neurologic Status.Rapid neurologic recovery: transient post-anesthetic decline (GCS 14) improves to full responsiveness
(GCS 15). Pain decreases from 6/10 to 0/10 within two hours postoperatively; spontaneous respiration is preserved.
Respiratory and Blood Gas Trends.Expected postoperative disturbances with rapid normalization: mild acidosis (pH
7.24) and hypercapnia (PaCO 252 mmHg) improve within two hours (pH 7.34, PaCO 246.5 mmHg). Oxygenation remains
adequate; FiO 2is weaned from 60% to 50%, indicating improving respiratory status with minimal ventilatory support.
Coagulation and Metabolic Status.Normal coagulation (INR 1.1; aPTT 21.9 s); preserved renal function (creatinine 0.9
mg/dL); electrolytes stable. Transient hyperglycemia (consistent with postoperative stress) is observed.
Monitoring and Support.Standard postoperative monitoring is in place, including arterial and central venous lines and
temporary airway protection, without evidence of escalating support requirements.
Summary.Overall, the retrieved evidence indicates rapid postoperative stabilization across neurologic, respiratory, and
hemodynamic domains, providing strong signals against prolonged hospitalization despite high baseline risk factors.
B.3. Adaptive Information Retrieval (AIR) inEHR-RAG
AIR refines retrieval by adaptively generating positive and negative queries to gather evidence supporting both hypotheses.
B.3.1. POSITIVEREASONINGPATH(EVIDENCE FORLONGSTAY)
Iteration 0:Query = “Cerebral vasospasm following SAH”
• Retrieved: Vasospasm monitoring protocols, CBF measurements, transcranial doppler assessments (5 documents).
• Rationale: Vasospasm is a major SAH complication requiring prolonged ICU stay; early monitoring is critical.
Iteration 1:Query = “Hydrocephalus following intracranial hemorrhage”
• Retrieved: Ventriculomegaly assessments, intracranial pressure trends, EVD indications (6 documents).
• Rationale: SAH frequently causes hydrocephalus; EVD insertion commonly prolongs ICU stay.
Iteration 2:Query = “External ventricular drain management post-aneurysm”
• Retrieved: EVD care protocols, CSF characteristics, EVD removal criteria, duration of drainage (3 documents).
• Rationale: EVD presence strongly predicts extended ICU stay (typically 5–7+ days).
B.3.2. NEGATIVEREASONINGPATH(EVIDENCE FORSHORTSTAY)
Iteration 0:Query = “Discharge to home post-neurosurgery without complications”
•Retrieved: Home discharge orders, absence of postoperative complications, successful emergence from anesthesia (4
documents).
• Rationale: Home discharge after neurosurgery indicates rapid, uncomplicated recovery.
Iteration 1:Query = “Social work assessment and discharge planning after ICU”
15

EHR-RAG
• Retrieved: Discharge readiness criteria, caregiver assessments, skilled nursing vs. home discharge (5 documents).
• Rationale: Early discharge planning and lack of complications facilitate short stay.
Iteration 2:Query = “Physical therapy: independent ambulation post-operative”
• Retrieved: PT clearance for discharge, functional independence measures, ICU mobility protocols (3 documents).
• Rationale: Independent ambulation signals discharge readiness.
B.4. Dual Evidence Retrieval and Reasoning (DER) inEHR-RAG
DER evaluates evidence for long- vs. short-stay hypotheses independently and then fuses contradictory evidence.
B.4.1. POSITIVEREASONING(SUPPORTINGLONGSTAY≥7 DAYS)
Severe Diagnosis and Procedure Complexity:
• Intracranial aneurysm clipping is a high-risk, complex microsurgical procedure.
• Requires precise dissection of carotid circulation and careful handling of fragile vessel.
• Typical post-op course involves 5–7 day minimum ICU stay for neuro monitoring.
• SAH increases risk of post-operative complications.
High-Intensity ICU Care Requirements:
• Mechanical ventilation for airway protection post-op.
• Continuous neuro monitoring (GCS, pupil reactivity, motor/sensory).
• Arterial line for beat-to-beat BP monitoring (neurosurgery standard).
• Central venous line for volume assessment and medication administration.
• Thermoregulation management (hypothermia induction protocol common in SAH).
• Vasoactive medications (nitroprusside documented) for hemodynamic optimization.
Acute Physiologic Derangements:
• Early metabolic acidosis (pH 7.24) indicating post-operative stress.
• Hypercapnia (PaCO 252 mmHg) suggesting ventilation/perfusion mismatch.
• Stress hyperglycemia (glucose 235 mg/dL) indicating systemic metabolic stress.
• These derangements typically persist for 3–5 days, prolonging ICU stay.
High-Risk Comorbidity Profile:
• Type 2 diabetes increases post-operative infection risk and impairs healing.
• Hypertension complicates post-op BP management in neurosurgery.
• Obesity increases anesthetic risk and mobility limitations.
• Tobacco use associated with respiratory complications.
• Combination of comorbidities predicts longer recovery trajectory.
Rehabilitation and Disposition Planning:
• Documented rehabilitation facility admission suggests anticipated prolonged stay.
• Comorbidities and mobility limitations often necessitate post-acute care facility.
• Home discharge typically requires minimal comorbidities and full functional independence.
16

EHR-RAG
B.4.2. NEGATIVEREASONING(SUPPORTINGSHORTSTAY<7 DAYS)
Rapid Anesthetic Emergence and Neurologic Recovery:
• By 20:42 (42 minutes post-op): MAC 0–0.1 indicates patient nearly fully awake.
• By 21:15 (75 minutes post-op): GCS 15 with full consciousness and appropriate responses.
• No requirement for prolonged sedation or neuromuscular blockade.
• Rapid emergence suggests excellent anesthetic recovery and minimal post-op CNS depression.
Stable Hemodynamics Without Vasopressor Dependence:
• SBP 98–130 mmHg without vasopressor infusions; mean BP 70–90s sufficient for cerebral perfusion.
• No hypotensive episodes requiring intervention.
• No documented inotropic support (dopamine, epinephrine).
• Hemodynamic stability by 1–2 hours post-op is a strong negative predictor.
Preserved Neurologic Integrity:
• GCS 15 at 21:15 demonstrates full consciousness.
• Temporary GCS 14 at 22:00 is mild and transient (likely opioid-related).
• No focal neurologic deficits; no stroke or intraoperative ischemic complications.
• Neurologic preservation suggests successful repair without complications.
Excellent Pain Control and Comfort:
• Pain reduced from 6/10 to 0/10 within 45 minutes.
• Adequate analgesia supports early mobilization and reduced ICU stay.
Laboratory Normalization and No Coagulopathy:
• Creatinine 0.9 mg/dL (normal renal function); electrolytes within normal limits.
• INR 1.1 and aPTT 21.9 indicate normal coagulation; no transfusions documented.
• Lack of metabolic/coagulation complications reduces ICU stay duration.
Physiologic Improvement and ABG Trend:
• pH improves from 7.24 to 7.34 within 2 hours.
• PaCO 2decreases from 52 to 46.5 mmHg; base deficit improves from 5 to 1 mmol/L.
• Rapid recovery suggests minimal complications and improving discharge readiness.
Absence of Documented Post-operative Complications:
• No rebleeding, hematoma expansion, new stroke/ischemia, ventilator failure, or respiratory distress.
• No infection/fever/sepsis; no escalation to vasopressors/ECMO.
Prior Rapid Recovery Pattern:
• Prior SAH admission (11/27) discharged within 1 day (11/28), indicating rapid recovery.
• Readmission on 12/04 for planned elective repair (not emergency).
• Planned procedures typically have better outcomes than emergency cases.
• Patient’s prior trajectory supports expectation of rapid current recovery.
17

EHR-RAG
B.4.3. EVIDENCEFUSION
The DER fusion module synthesizes both positive and negative evidence:
“While the procedure itself is complex and carries inherent risks typical of intracranial aneurysm repair, this particular patient
demonstrated exceptionally rapid post-operative recovery. Within 2 hours of ICU admission, she was fully conscious (GCS 15),
hemodynamically stable without vasopressor support, and showing rapid physiologic improvement (ABG normalization, pain
control achieved).
The presence of comorbidities (diabetes, obesity, HTN) and historical SAH are concerning for prolonged stay. However, the
complete absence of any documented post-operative complication, combined with the patient’s prior history of rapid discharge
after SAH and the elective (not emergency) nature of this procedure suggest excellent surgical repair with minimal morbidity.
By prediction time (23:59, 4 hours post-op), all indicators point to an uncomplicated trajectory: no vasospasm signs, no
hydrocephalus indicators, no coagulopathy, normal renal function, and preserved neurologic status. This contrasts with the
typical 5–7 day post-op course.
Conclusion:The balance of evidence strongly favors shorter length of stay ( <7 days). The rapid recovery trajectory and
absence of complications outweigh theoretical risks from procedure complexity and comorbidities. ”
Final Prediction:0 (Length of Stay<7 days)Confidence:High
Ground Truth:Discharged 2014-12-06 (1 day post-op)Outcome:✓Correct
B.5. Failure Analysis: Why Direct Generation and Vanilla RAG Failed
B.5.1. DIRECTGENERATIONFAILURE
Reasoning Pattern:
“Complex intracranial aneurysm clipping surgery in a patient with multiple comorbidities (diabetes, obesity, hypertension)
typically requires a lengthy ICU stay for monitoring of post-operative complications such as vasospasm and hydrocephalus.
The patient’s initial presentation with SAH suggests high-risk recovery. Expected LOS≥7 days. ”
Failure Mode – Confirmation Bias:
•Received Data:Accessed identical clinical data (GCS, ABGs, vital signs, etc.).
•Reasoning Error:Anchored on surgical complexity and comorbidities, underweighting recovery trajectory.
•Bias:Overweighted worst-case complications (vasospasm, hydrocephalus) without evidence.
•Missed Signal:Failed to treat rapid neurologic recovery (GCS 15 within hours) as countervailing evidence.
•Static Thinking:Evaluated a single timepoint rather than dynamic improvement.
B.5.2. VANILLARAG FAILURE
Reasoning Pattern:
“Retrieval augmented generation returned articles on post-operative complications of intracranial aneurysm surgery, typical
ICU stay durations (5–7 days), and management of vasospasm and hydrocephalus. Based on retrieved context, predicted LOS
≥7 days. ”
Failure Mode – Lack of Systematic Evaluation:
•Retrieval Bias:Query biased toward complications and prolonged monitoring.
•Missing Dual Perspective:Retrieved risk-confirming documents without systematic search for rapid recovery evidence.
•No Negative Evidence:No structured mechanism to identify/weight evidence against high-risk conclusion.
•Static Aggregation:Treated typical 5–7 day LOS as a rule, ignoring patient-specific trajectory.
•Confirmation Loop:Retrieval reinforced initial impression without critical reappraisal.
18

EHR-RAG
B.5.3. EHR-RAG SUCCESS
Key Advantages:
•Systematic Dual Evaluation:Explicitly considers both positive and negative evidence.
•Balanced Retrieval:AIR issues positive queries (vasospasm, hydrocephalus) and negative queries (recovery, discharge
planning).
•Trajectory Analysis:DER emphasizes dynamic trends (e.g., ABG improvement) rather than static risk factors.
•Evidence Fusion:Appropriately weights rapid improvement against theoretical risks.
•False-alarm Reduction:Prevents overconfident high-risk classification via structured dual reasoning.
B.6. Clinical Significance and Learning Points
Key Insight.Complex surgery does not automatically imply prolonged stay:
1.Procedure complexity̸=complication inevitability:Skilled repair can avoid major post-op complications.
2.Rapid recovery in high-risk patients:Despite obesity/diabetes/HTN, full neurologic recovery occurs within hours.
3.Trajectory over static risk:Improving vitals/labs/neuro status within 2 hours is a strong predictor of short stay.
4.No early complications:Absence of complications by 4 hours post-op increases short-stay likelihood.
Clinical Impact of Accurate Prediction:
•ICU resource allocation:Enables earlier step-down and frees beds for critical patients.
•Discharge planning:Supports planning for rapid discharge vs. prolonged facility stay.
•Patient/family expectations:Reduces unnecessary anxiety about prolonged recovery.
•Cost reduction:Avoids unnecessary ICU days (at ~$5000/day).
•Mobility/infection prevention:Earlier step-down supports mobilization and reduces ICU-related risks.
B.7. Summary
The structured dual reasoning of EHR-RAG mitigates false-alarm bias by preventing worst-case anchoring without propor-
tional evidence. By enforcing balanced evaluation of contradictory evidence, EHR-RAG maintains sensitivity to high-risk
cases while reducing excessive false positives.
19

EHR-RAG
C. Metadata for Clinical Prediction Tasks
Table 5presents the metadata of the EHRSHOT clinical prediction tasks used in our experiments, including core task
definitions, basic task queries, and task-specific instructions. This metadata is provided to the LLMs under a unified prompt
framework to support clinical prediction across different tasks.
Table 5.Metadata of EHRSHOT clinical prediction tasks used in our experiments. Each column corresponds to a task, and each row
reports a specific metadata field.
Field guo_los guo_readmission new_acutemi lab_anemia
Task name Long Length of Stay 30-day Readmission Acute Myocardial Infarction Anemia
Category Operational Outcomes Operational Outcomes Assignment of New Diagnoses Anticipating Lab Test Results
Description Predict whether a patient will
have a long hospital stay ( ≥7
days) based on their EHR data.Predict whether a patient will
be readmitted within 30 days
after hospital discharge based
on their EHR data.Predict whether a patient will
receive a new acute myocardial
infarction diagnosis within 1
year after discharge based on
their EHR data.Predict the severity category of
the next anemia-related labo-
ratory result based on the pa-
tient’s prior EHR data.
Factual query clinical factors and events asso-
ciated with prolonged hospital
stayclinical factors and events as-
sociated with 30-day hospital
readmissionclinical risk factors and events
associated with acute myocar-
dial infarctionclinical factors and events rele-
vant to predicting anemia sever-
ity
Counterfactual query clinical factors and events asso-
ciated with short hospital stayclinical factors and events asso-
ciated with no readmissionclinical risk factors and events
indicating absence of acute my-
ocardial infarctionclinical factors and events indi-
cating no anemia
Label type binary binary binary multiclass_4
Label values{0,1} {0,1} {0,1} {0,1,2,3}
Label description{0,1} aka {<7 days, ≥7 days} {0,1} aka {no readmission,
readmission}{0,1} aka {no diagnosis, diag-
nosis}{0,1,2,3} aka {low, medium,
high, abnormal}
Task-specific instructions No instructions provided. 30-day readmission is uncom-
mon. Default to 0 unless there
is clear, strong, patient-specific
evidence; do not predict 1 from
vague risk factors.Be sensitive to positives: if
there is any reasonable, patient-
specific evidence suggesting
acute MI, lean toward 1; if un-
certain, prefer 1.Choose among {0,1,2,3} with
calibrated preference for mild-
to-moderate (1 or 2) when
uncertain; use 0 only with
strong evidence of no ane-
mia; use 3 only with clear
severe/abnormal anemia evi-
dence.
20

EHR-RAG
D. Prompt Design
Figures 3present the prompt template used by the LLM-based clinical prediction baselines. Figures 4and5illustrate
the prompts employed in theEvent- and Time-Aware Hybrid EHR Retrieval (ETHER)andAdaptive Iterative Retrieval
(AIR)components ofEHR-RAG, respectively. Figures 6,7, and 8show the prompt templates used for factual reasoning,
counterfactual reasoning, and evidence fusion in theDual-Path Evidence Retrieval and Reasoning (DER)component.
# TaskYou are a reliable clinical expert.Based on the information below and the patient medical history,answer the clinical question for the task of "{task_name}", which belongs to the "{task_category}".# InformationClinical Question: {description}Prediction time: {prediction_time}Label Definition: {label_description}Patient History:{patient_history}# Instructions1. You must provide a clinical prediction based on the provided patient history at the end of your response.2. If the evidence is incomplete or uncertain, make the best-supported prediction based on the available information.3. This is a {label_type} classification task, so your final answer must be an integer chosen from {label_values}.# Response FormatPlease think step-by-step, showing your reasoning, and then provide the final answer strictly in this format (each on its own line):Reasoning: <your reasoning here>Answer: <your final answer of classification result>Now, give your response:Prompt of ClinicalPrediction Baselines
Figure 3.Prompt template used for LLM-based clinical prediction baselines. Blue text indicates input variables.
21

EHR-RAG
# TaskYou are a clinical expert selecting the most informative clinical indicators to support a downstream prediction task.The clinical question corresponds to the task "{task_name}", which belongs to the category "{task_category}".# InformationClinical Question: {description}Candidate Clinical Indicators: {indicator_list}# Instructions1. Select the top {topk} indicators that are most relevant and informative for answering the clinical question.2. Focus on indicators that are clinically meaningful, discriminative, and directly related to the task.3. Do NOT explain your choice.# Response FormatReturn ONLY a JSON list (no extra text).Each element must be the indicator name EXACTLY as it appears above.Example:["indicator A", "indicator B", "indicator C"]Prompt of Clinical Indicator Selection(ETHER)
Figure 4.Prompt template used for clinical indicator selection in theEvent- and Time-Aware Hybrid EHR Retrievalcomponent of
EHR-RAG. Blue text denotes input variables.
# TaskYou are refining a retrieval query to find missing EHR evidence that supports the objective.# InformationTask: {task_name}Clinical question: {description}Objective: {objective}Previously used queries: {other_queries}Evidence already retrieved:{evidence_snippets}# Instructions1. Only output ENOUGH if the retrieved evidence already covers all major clinical aspects that are typically required to answer the clinical question.2. Otherwise, write ONE refined query to retrieve an important missing piece of clinical evidence.3. Assume the retrieval source does NOT contain direct answers to the clinical question.4. The refined query must:-Be concise (<= 15 words)-Focus on a single clinical aspect (no lists, no multiple features)-Target evidence NOT already present above-Avoid dates, timestamps, and NOT mention prediction targets-Be different from all previous queries# Response FormatReturn ONLY one of the following:-ENOUGH-The refined query textPrompt of Query Refinement(AIR)
Figure 5.Prompt template used for query refinement in theAdaptive Iterative Retrievalcomponent ofEHR-RAG. Blue text denotes input
variables.
22

EHR-RAG
# TaskYou are a reliable clinical expert.Based on the information below and the patient medical history,answer the clinical question for the task of "{task_name}", which belongs to the "{task_category}".# InformationClinical Question: {description}Prediction time: {prediction_time}Label Definition: {label_description}Patient History:{patient_history}indicator_information:{indicator_information}# Instructions1. Assume the outcome is positive.2. Your role is to identify and reason over evidence that SUPPORTS the positive outcome (label = {positive_label}).3. Focus ONLY on strong, direct clinical evidence that would justify predicting label = {positive_label}.4. Do NOT consider evidence that argues against a positive outcome.5. If no strong supporting evidence exists, explicitly state that the evidence for label = {positive_label}is weak.Now, think step-by-step, showing your reasoning:Prompt of Factual Reasoning (DER)
Figure 6.Prompt template used for factual reasoning in theDual-Path Evidence Retrieval and Reasoningcomponent ofEHR-RAG. Blue
text denotes input variables.
# TaskYou are a reliable clinical expert.Based on the information below and the patient medical history,answer the clinical question for the task of "{task_name}", which belongs to the "{task_category}".# InformationClinical Question: {description}Prediction time: {prediction_time}Label Definition: {label_description}Patient History:{patient_history}indicator_information:{indicator_information}# Instructions1. Assume the outcome is negative.2. Your role is to identify and reason over evidence that SUPPORTS the negative outcome (label = {negative_label}).3. Focus ONLY on strong, direct clinical evidence that would justify predicting label = {negative_label}.4. Do NOT consider evidence that argues against a negative outcome.5. If no strong supporting evidence exists, explicitly state that the evidence for label = {negative_label}is weak.Now, think step-by-step, showing your reasoning:Prompt of Counterfactual Reasoning (DER)
Figure 7.Prompt template used for counterfactual reasoning in theDual-Path Evidence Retrieval and Reasoningcomponent ofEHR-RAG.
Blue text denotes input variables.
23

EHR-RAG
# TaskYou are a reliable clinical expert.Using the patient history and two complementary reasoning summaries(one arguing for the positive outcome and one for the negative outcome),answer the clinical question for the task "{task_name}" in the category "{task_category}".# InformationClinical Question: {description}Prediction time: {prediction_time}Label Definition: {label_description}Patient History:{patient_history}indicator_information:{indicator_information}Reasoning in support of the positive outcome:{factual_reasoning}Reasoning in support of the negative outcome:{counterfactual_reasoning}# Instructions1. Compare the strength, directness, and relevance of evidence on both sides. Prefer strong, direct evidence over weak or speculative evidence.2. Choose the label that is most consistent with the available evidence, and state why.3. If the evidence on both sides is roughly balanced in strength, or if both sides are weak, DEFAULT to the NEGATIVE outcome (0),since the default assumption is that the patient has no condition unless there is strong evidence to the contrary.4. This is a {label_type} classification task. Your final answer must be an integer chosen from {label_values}.{task_speicific_instructions}# Response FormatPlease think step-by-step, showing your reasoning, and then provide the final answer strictly in this format (each on its own line):Reasoning: <your reasoning here>Answer: <your final answer of classification result>Now, give your response:Prompt of Evidence Fusion Reasoning(DER)
Figure 8.Prompt template used for evidence fusion reasoning in theDual-Path Evidence Retrieval and Reasoningcomponent ofEHR-RAG.
Blue text denotes input variables.
24