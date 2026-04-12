# MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion Detection with an Experience Library

**Authors**: Md Shamimul Islam, Luis G. Jaimes, Ayesha S. Dina

**Published**: 2026-04-07 05:41:10

**PDF URL**: [https://arxiv.org/pdf/2604.05458v1](https://arxiv.org/pdf/2604.05458v1)

## Abstract
Network Intrusion Detection Systems (NIDS) face important limitations. Signature-based methods are effective for known attack patterns, but they struggle to detect zero-day attacks and often miss modified variants of previously known attacks, while many machine learning approaches offer limited interpretability. These challenges become even more severe in IoT environments because of resource constraints and heterogeneous protocols. To address these issues, we propose MA-IDS, a Multi-Agent Intrusion Detection System that combines Large Language Models (LLMs) with Retrieval Augmented Generation (RAG) for reasoning-driven intrusion detection. The proposed framework grounds LLM reasoning through a persistent, self-building Experience Library. Two specialized agents collaborate through a FAISS-based vector database: a Traffic Classification Agent that retrieves past error rules before each inference, and an Error Analysis Agent that converts misclassifications into human-readable detection rules stored for future retrieval, enabling continual learning through external knowledge accumulation, without modifying the underlying language model. Evaluated on NF-BoT-IoT and NF-ToN-IoT benchmark datasets, MA-IDS achieves Macro F1-Scores of 89.75% and 85.22%, improving over zero-shot baselines of 17% and 4.96% by more than 72 and 80 percentage points. These results are competitive with SVM while providing rule-level explanations for every classification decision, demonstrating that retrieval-augmented reasoning offers a principled path toward explainable, self-improving intrusion detection for IoT networks.

## Full Text


<!-- PDF content starts -->

MA-IDS: Multi-Agent RAG Framework for IoT Network Intrusion
Detection with an Experience Library
Md Shamimul Islam1, Luis G. Jaimes2and Ayesha S. Dina3
Abstract— Network Intrusion Detection Systems (NIDS) face
important limitations. Signature-based methods are effective
for known attack patterns, but they struggle to detect zero-
day attacks and often miss modified variants of previously
known attacks, while many machine learning approaches offer
limited interpretability. These challenges become even more
severe in IoT environments because of resource constraints
and heterogeneous protocols. To address these issues, we
propose MA-IDS, a Multi-Agent Intrusion Detection System
that combines Large Language Models (LLMs) with Retrieval
Augmented Generation (RAG) for reasoning-driven intrusion
detection. The proposed framework grounds LLM reasoning
through a persistent, self-building Experience Library. Two
specialized agents collaborate through a FAISS-based vector
database: a Traffic Classification Agent that retrieves past error
rules before each inference, and an Error Analysis Agent that
converts misclassifications into human-readable detection rules
stored for future retrieval, enabling continual learning through
external knowledge accumulation, without modifying the under-
lying language model. Evaluated on NF-BoT-IoT and NF-ToN-
IoT benchmark datasets, MA-IDS achieves Macro F1-Scores of
89.75% and 85.22%, improving over zero-shot baselines of 17%
and 4.96% by more than 72 and 80 percentage points. These
results are competitive with SVM while providing rule-level
explanations for every classification decision, demonstrating
that retrieval-augmented reasoning offers a principled path
toward explainable, self-improving intrusion detection for IoT
networks.
I. INTRODUCTION
The proliferation of Internet of Things (IoT) devices has
exponentially expanded the network attack surface, precip-
itating a rise in sophisticated, high-volume cyber threats.
Traditional Network Intrusion Detection Systems (NIDS)
primarily rely on signature-based methods, which are inher-
ently reactive; they struggle to identify zero-day exploits or
polymorphic variations of known attacks, leading to high
false-negative rates [1], [2].
To overcome these limitations, machine learning (ML) and
deep learning (DL) approaches have been widely adopted for
their ability to model complex patterns in high-dimensional
data. However, these models typically operate as “black
1Md Shamimul Islam is with the Department of Computer
Science, Florida Polytechnic University, Lakeland, Florida
mislam3051@floridapoly.edu
2Luis G. Jaimes is with the Department of Computer
Science, Florida Polytechnic University, Lakeland, Florida
ljaimes@floridapoly.edu
3Ayesha S. Dina is with the Department of Computer
Science, Florida Polytechnic University, Lakeland, Florida
adina@floridapoly.edu
This work has been submitted to the IEEE for possible publication.
Copyright may be transferred without notice, after which this version may
no longer be accessible.boxes,” providing high classification accuracy at the expense
of interpretability. This is a critical shortcoming in security-
critical environments where forensic justification is required
for incident response. Furthermore, their reliance on static
training distributions makes them brittle against evolving
threats, often necessitating computationally expensive re-
training to maintain efficacy [2], [4].
Large Language Models (LLMs) offer a paradigm shift
by enabling semantic reasoning, few-shot generalization, and
human-readable explanations. Yet, their direct application to
NIDS is hindered by a “domain gap”: pre-trained LLMs
struggle to interpret numerical NetFlow telemetry, which
lacks the natural language structure they were trained on. Our
empirical benchmarks highlight this failure: a zero-shot GPT-
4o baseline achieves macro F1-scores of only 17% on NF-
BoT-IoT and a negligible 4.96% on NF-ToN-IoT. These re-
sults confirm that without domain-specific grounding, LLMs
provide unstable and often misleading threat assessments [4],
[33].
To bridge this gap without the prohibitive overhead of
continuous fine-tuning, we propose MA-IDS, a Multi-Agent
Intrusion Detection System that transforms classification
errors into a persistent, self-evolving Experience Library.
Unlike traditional supervised models that suffer from “silent
failure,” MA-IDS employs a closed-loop architecture featur-
ing two specialized GPT-4o agents: (1) an Error Analysis
Agent that identifies discriminative features in misclassified
flows to formulate structured, semantic rules; and (2) a Traf-
fic Classification Agent that utilizes Retrieval-Augmented
Generation (RAG) to query a FAISS-based vector database
for relevant past experiences during inference.
Evaluated on the NF-BoT-IoT and NF-ToN-IoT datasets,
MA-IDS achieves macro F1-scores of 89.75% and 85.22%,
respectively, which constitutes a performance leap of up to 80
percentage points over zero-shot baselines. While supervised
classifiers like SVM may achieve higher peak accuracy on
static benchmarks, MA-IDS offers a critical trade-off: seman-
tic resilience. By providing rule-level forensic justifications
for every decision, it shifts NIDS from a pattern-matching
paradigm to an adaptive reasoning framework capable of
continuous self-improvement without parameter modifica-
tion.
The remainder of this paper is organized as follows.
Section II reviews the evolution of NIDS from traditional ML
to agentic AI. Section III details the MA-IDS architecture,
including its multi-agent workflow and Experience Library.
Section IV presents the experimental setup, performance
results, and comparative analysis. Finally, Section V summa-arXiv:2604.05458v1  [cs.CR]  7 Apr 2026

rizes the key findings and outlines future research directions.
II. RELATED WORK
Conventional IDS.Foundational research established two
primary paradigms: signature-based systems (SIDS), which
identify known threats through predefined patterns [7], [10]–
[12], and anomaly-based systems (AIDS), which flag de-
viations from established normal behavior [5], [6], [14],
[15]. While deployment guidelines for network and host-
based configurations are well-documented [8], [9], these
systems face significant maintenance burdens. Knowledge-
driven variants employing formal description languages [16]
or rule-based expert systems [17] often struggle in dy-
namic network environments. Critically, SIDS are ineffective
against zero-day exploits [13], and AIDS require frequent
manual recalibration to maintain accuracy. MA-IDS ad-
dresses these gaps by bypassing static signatures in favor
of autonomous, error-driven rule induction that eliminates
the need for manual profile updates.
Machine Learning-Based IDS.A variety of machine
learning (ML) algorithms, including multilayer percep-
trons [18], k-nearest neighbors [19], decision trees [20], and
SVMs [21], have been evaluated for anomaly-oriented detec-
tion. In IoT environments, ensemble methods like Random
Forest and AdaBoost have been extensively analyzed [22],
alongside lightweight models for resource-constrained de-
ployments [23] and quantum annealing for feature selec-
tion [24]. Despite their classification efficacy, these models
function as “black boxes” and require full retraining to
incorporate new attack vectors. MA-IDS maintains compet-
itive performance while providing human-readable semantic
explanations and adapting to new threats without model
retraining.
Deep Learning-Based IDS.Deep learning (DL) archi-
tectures, such as CNNs, LSTMs, and autoencoders, have
become central to modern NIDS research [25]–[28]. Recent
efforts have focused on IoT-specific threat detection [29],
hybrid CNN-BiLSTM models [30], and explainable DL
frameworks for Industrial IoT [31], [32]. However, DL mod-
els remain fundamentally static post-training and demand
substantial labeled data for retraining. In contrast, MA-
IDS generates interpretable rules during inference, operates
effectively without massive labeled datasets, and provides
real-time decision-time reasoning.
LLM and Agentic AI-Based IDS.The application of
Large Language Models (LLMs) to network intrusion detec-
tion has gained increasing attention, with techniques such as
in-context learning improving detection performance [33].
TrafficGPT [2] leverages decoder-only architectures for
open-set classification, while NetMoniAI [3] explores hi-
erarchical multi-agent coordination for distributed monitor-
ing. IDS-Agent [1] further incorporates reasoning–action
pipelines and long-term memory to integrate outputs from
multiple ML models.
However, existing approaches lack mechanisms for sys-
tematic self-correction based on deployment feedback. MA-
IDS addresses this limitation by introducing an Error Anal-ysis Agent that converts misclassifications into verifiable
rules stored in a persistent Experience Library. This en-
ables a self-improving feedback loop, reducing reliance on
ungrounded zero-shot reasoning and enhancing contextual
decision-making.
III. METHODOLOGY
The MA-IDS framework is designed to bridge the gap
between high-dimensional network telemetry and the se-
mantic reasoning capabilities of Large Language Models
(LLMs). As illustrated in Figure 1, the system integrates
two specialized agents that interact through a shared, FAISS-
based Experience Library. This closed-loop architecture en-
ables the system to transform transient classification errors
into persistent, human-readable knowledge, ensuring that the
detection engine becomes progressively more accurate while
avoiding the computational overhead of fine-tuning cycles.
A. Multi-Agent Architecture
The core architecture of MA-IDS follows a reasoning-
followed-by-action pipeline. This design decouples real-time
classification logic from diagnostic learning logic. Letx∈
R14represent the feature vector of an incoming network flow,
Cdenote the set of target traffic classes, andL trepresent
the state of the Experience Library at time stept. The final
classification decisionˆyis formally defined as:
ˆy=A cls
x∥ R(x;L t)
,(1)
where∥denotes prompt concatenation andR(x;L t)is the
retrieval function that provides the agent with contextually
relevant past experiences.
1) Traffic Classification Agent:The Traffic Classification
Agent serves as the primary inference engine operating in the
online inference phase. When a new network flow arrives,
the agent utilizes Retrieval-Augmented Generation (RAG) to
identify the most similar historical error stored in the library.
A 384-dimensional embedding is generated forxusing the
all-MiniLM-L6-v2 model, which is then used to query the
FAISS index via cosine similarity:
R(x;L t) =(
ρiifcos 
emb(x),emb(z i)
≥τ,
ϵotherwise,(2)
whereτis the similarity threshold andρ iis the retrieved
semantic rule. By injecting this rule into the structured
promptp(x, r), the agent gains a grounded memory of past
classification boundaries, significantly reducing the likeli-
hood of repeating previous errors.
2) Error Analysis and Rule Induction Agent:The Error
Analysis Agent constitutes the system’s learning mechanism,
operating in the offline refinement phase. It is activated
exclusively upon a misclassification (ˆy̸=y), whereyis
the ground-truth label. The agent performs a diagnostic
“compare-and-isolate” reasoning chain by examining the
misclassified flowxalongside the correct labelyand the

Fig. 1: Dual-Phase Agentic Workflow for MA-IDS. Phase 1 (Top) illustrates the offline library-building loop, where an Error
Analysis Agent evaluates misclassifications to generate human-readable rules for the Experience Library. Phase 2 (Bottom)
depicts the real-time classification process, utilizing RAG to inject past experiences into the LLM’s context for informed
decision-making.
erroneous predictionˆy. The agent then distills this analysis
into a concise, human-readable ruleρ:
ρ=A ind
x,ˆy, y,R(x;L t)
.(3)
For example, the agent may identify that a high average inter-
arrival time (IAT) is a primary indicator of a specific attack
type that was previously mislabeled. This induced rule is
committed to the Experience Library, effectively teaching the
system the discriminative features between traffic classes.
3) Experience Library and Vector Database:The Expe-
rience Library serves as the long-term memory of MA-IDS,
implemented as a persistent local FAISS vector database.
Unlike traditional machine learning models that store knowl-
edge in static mathematical weights, MA-IDS externalizes
knowledge into a retrievable library, allowing the system to
scale its expertise indefinitely. Each entry inLconsists of
a vector key (the flow embedding) and a metadata payload
containing the semantic rule. The update process is defined
as:
Lt+1=L t∪n 
emb(x), ρ,(ˆy, y)o
.(4)
This design choice ensures that learning is non-destructive,
protecting the system from the “catastrophic forgetting”
typically associated with deep learning fine-tuning.
B. Operational Workflow
As depicted in the Phase 1 and Phase 2 loops of Figure
1, the MA-IDS workflow operates in two interleaved stages
that separate knowledge acquisition from inference.
In Phase 1 (Offline Experience Creation), the system
identifies classification failures and updates the library. This
process functions as an automated forensic analysis thatgenerates a growing repository of “hard-negative” examples
derived from the system’s own errors.
In Phase 2 (Online Classification), the system uses the li-
brary to ground its real-time decisions. Upon flow arrival, the
Classification Agent performs high-speed retrieval from the
Experience Library. By combining raw numerical features
with high-level semantic rules, the agent performs context-
aware reasoning that addresses the limitations of standard
zero-shot LLM prompts.
IV. EXPERIMENTAL RESULTS
A. Experimental Setup
The experimental evaluation was conducted on a work-
station with an AMD Ryzen 5 2500U processor and 16GB
of RAM. Flow embeddings were generated locally using
the HuggingFaceall-MiniLM-L6-v2encoder, selected for its
balance between semantic representation quality and low-
latency CPU inference. The reasoning and decision-making
components were implemented using OpenAI GPT-4o ac-
cessed via API. To ensure deterministic and reproducible
results, the sampling temperature for both agents was fixed
at 0.0.
B. Dataset and Evaluation Protocol
We evaluate MA-IDS on two widely used benchmark
datasets for IoT intrusion detection, both derived from Net-
Flow V3 and curated by the University of Queensland [34].
The NF-BoT-IoT dataset comprises over 16.9 million labeled
flows, from which we consider four classes: Benign, DDoS,
DoS, and Reconnaissance. The NF-ToN-IoT dataset contains
27.5 million flows, from which we evaluate nine classes: Be-
nign, Scanning, DDoS, Backdoor, DoS, Injection, Password,
XSS, and MITM.

(a) NF-BoT-IoT
 (b) NF-ToN-IoT
Fig. 2: Macro-averaged Precision, Recall, and F1-Score for the Zero-Shot Baseline and MA-IDS on (a) NF-BoT-IoT and (b)
NF-ToN-IoT. The near-collapsed baseline polygon versus the substantially expanded MA-IDS polygon illustrates the critical
role of Experience Library retrieval in enabling reliable classification across all three metrics.
TABLE I: Performance comparison of MA-IDS against traditional ML baselines and the Zero-Shot GPT-4o baseline on
NF-BoT-IoT and NF-ToN-IoT test samples. MA-IDS results reflect the evaluation stage with the Experience Library fixed
and the Error Analysis Agent disabled.
Dataset Method Accuracy (%) Precision (%) Recall (%) F1-Score (%)
NF-BoT-IoT
AdaBoost 70.41 67.58 70.00 88.60
Na¨ıve Bayes 82.21 88.00 82.00 79.90
SVM 88.21 89.00 88.00 88.60
Zero-Shot (GPT-4o) 21.60 19.50 22.25 17.00
Our MA-IDS 90.00 90.00 90.00 89.75
Improvement (vs Zero-Shot) +68.40 +70.50 +67.75 +72.75
NF-ToN-IoT
AdaBoost 43.83 39.67 43.83 36.79
Na¨ıve Bayes 86.19 87.61 86.19 85.94
SVM 93.79 94.66 93.79 93.42
Zero-Shot (GPT-4o) 13.33 11.52 11.78 4.96
Our MA-IDS 84.00 85.56 85.00 85.22
Improvement (vs Zero-Shot) +70.67 +74.04 +73.22 +80.26
To ensure a balanced and unbiased evaluation, as well
as equitable rule generation across classes, uniform random
sampling with equal class quotas is employed. For each
dataset, 50,000 samples are used during the library construc-
tion phase, while a disjoint set of 20,000 samples is reserved
for evaluation. Both subsets are uniformly distributed across
all classes, ensuring equitable representation and enabling
direct comparison of per-class performance metrics while
preventing data leakage between phases.
From the original 53 NetFlow V3 attributes, 14 features
are selected based on discriminative relevance, computational
efficiency, and privacy preservation. These include contextual
identifiers (IP addresses, destination port, protocol), volu-
metric statistics (byte and packet counts), temporal features
(flow duration and inter-arrival times), throughput measures,
and TCP flag aggregates, which together capture structural,
behavioral, and directional characteristics of network flows.
Remaining features are excluded for three reasons: (i)
redundancy, as many attributes (e.g., min/max packet sizes,
TTL, and window statistics) provide overlapping information
with selected volumetric and temporal metrics; (ii) limiteddiscriminative contribution, particularly for protocol-specific
or rarely populated fields (e.g., DNS, FTP, ICMP); and (iii)
incompatibility with privacy-preserving and encrypted traffic
settings, where payload-dependent or fine-grained inspection
features are unreliable or unavailable.
During preprocessing, invalid values are normalized, pro-
tocol identifiers are mapped to categorical representations,
and all features are serialized into structured JSON format
to enable efficient and interpretable LLM-based reasoning.
C. Evaluation Metrics
Performance for all configurations was evaluated using
overall accuracy along with macro averaged precision, recall,
and F1 score. The macro averaging technique calculates each
metric independently for every individual class before com-
puting the arithmetic mean across the entire set of classes.
This approach assigns equal weight to each traffic category
irrespective of the specific sample distribution, which aligns
with the uniform sampling strategy employed in this study.

0%10%20%30%40%50%60%70%80%90%100%
0 10,000 20,000 30,000 40,000 50,000
SamplesMacro F1 ScoreWith Library Without Library(a) NF-BoT-IoT dataset
0%10%20%30%40%50%60%70%80%90%100%
10,000 20,000 30,000 40,000 50,000
SamplesMacro F1 ScoreWith Library Without Library (b) NF-ToN-IoT dataset
Fig. 3: Macro F1-score over cumulative samples during library construction for MA-IDS with and without the Experience
Library on (a) NF-BoT-IoT and (b) NF-ToN-IoT. Performance improves with library growth, while the no-library baseline
remains flat, highlighting the impact of accumulated rule context.
D. Performance Evaluation: NF-BoT-IoT
The performance of MA-IDS on the NF-BoT-IoT dataset
was assessed through a two-stage experimental protocol.
During the library construction stage, the complete MA-IDS
pipeline with the Error Analysis Agent active was executed
over 50,000 samples, yielding an accuracy of 85.75%, Macro
Precision of 85.66%, Macro Recall of 85.41%, and Macro
F1-Score of 85.43%. In the evaluation stage, the Error
Analysis Agent was deactivated and the fixed repository of
7,322 induced rules was used to classify an independent test
set of 20,000 samples, achieving an accuracy of 90.00% and
Macro F1-Score of 89.75% across all metrics, demonstrat-
ing strong generalisation of the constructed library without
further adaptation (Fig. 2a). As reported in Table I, MA-IDS
at 89.75% Macro F1 surpasses SVM (88.60%), Na ¨ıve Bayes
(79.90%), and AdaBoost (88.60%) on this dataset, while the
zero-shot GPT-4o baseline achieves only 17.00%, confirming
that the Experience Library is the decisive performance
driver rather than the underlying model capability.
E. Performance Evaluation: NF-ToN-IoT
On the NF-ToN-IoT dataset, the library construction stage
over 50,000 samples with the MA-IDS achieved an accuracy
of 80.60%, Macro Precision of 83.76%, Macro Recall of
79.91%, and a Macro F1-Score of 81.01%. In the evaluation
stage, with the MA-IDS disabled and the pre-built library
of 9,498 rules frozen, testing on 20,000 samples yielded
an accuracy of 84.00%, Macro Precision of 85.56%, Macro
Recall of 85.00%, and a Macro F1-Score of 85.22%, with
the performance contrast between the two configurations
clearly visible in Fig. 2b. As reported in Table I, MA-
IDS at 85.22% Macro F1 outperforms AdaBoost (36.79%)
and remains competitive with Na ¨ıve Bayes (85.94%) on
this significantly more complex nine-class task, while the
zero-shot GPT-4o baseline collapses to only 4.96% Macro
F1. Although SVM achieves a higher 93.42%, it offers no
interpretability, requires complete retraining. These resultsconfirm that the Experience Library is the critical enabler of
reliable LLM-based classification across both datasets.
F . Ablation Study
To isolate component contributions, we evaluate three
configurations: (1) Zero-Shot Baseline (no retrieval), (2)
Library Only (retrieval with fixed Experience Library), and
(3) Full MA-IDS (retrieval with continuous rule induction).
On NF-BoT-IoT, the zero-shot baseline achieves 17.00%
Macro F1, with DDoS detection failing (0% true positives),
indicating limited discrimination without contextual ground-
ing. The Library Only configuration improves performance
to 89.75% Macro F1 on 20,000 test samples, demonstrating
that retrieval is the primary driver. The Full MA-IDS system
attains 85.43% during the construction phase, where the rule
base is still evolving.
On NF-ToN-IoT, the zero-shot baseline degrades to 4.96%
Macro F1, while Library Only reaches 85.22%, again high-
lighting the impact of retrieval. Full MA-IDS achieves
81.01%, with slightly lower performance due to ongoing
rule induction. The larger gains on NF-ToN-IoT suggest that
retrieval becomes increasingly beneficial as task complexity
grows.
Learning curves (Fig. 3) show consistent improvement as
the Experience Library expands, while the zero-shot baseline
remains unchanged. Performance stabilizes earlier on NF-
BoT-IoT and more gradually on NF-ToN-IoT, reflecting
differences in class complexity. Overall, retrieval provides
the dominant performance gain, while the MA-IDS supports
continual adaptation through rule expansion.
G. Experience Library Rule Distribution
Table II summarizes the number of the rules generated per
class during the library construction phase. Rule counts re-
flect class-wise misclassification frequency, with overlapping
traffic patterns producing more rules. On NF-BoT-IoT, DDoS
and DoS generate the highest counts, while on NF-ToN-
IoT, Injection and XSS dominate due to greater semantic

complexity. In contrast, Backdoor yields few rules, indicating
clear separability. Overall, the library comprises 7,322 rules
for NF-BoT-IoT and 9,498 for NF-ToN-IoT, forming the
fixed knowledge base for subsequent evaluation.
TABLE II: Experience Library Rule Distribution
NF BoT IoT NF ToN IoT
Class Rules Class Rules
Benign 1,631 Benign 1,107
DDoS 2,384 Scanning 1,250
DoS 1,679 DDoS 708
Reconnaissance 1,624 Backdoor 69
Noise 4 DoS 745
Injection 2,010
Password 820
XSS 1,639
MITM 1,150
Total 7,322 Total 9,498
V. CONCLUSIONS
This paper presented MA-IDS, a Multi-Agent Intrusion
Detection System integrating LLM with Context-Grounded
Classification for reasoning-driven intrusion detection in IoT
networks. The framework couples a Traffic Classification
Agent with an MA-IDS over a persistent Experience Library,
enabling continual self-improvement without model retrain-
ing. Evaluation on NF-BoT-IoT and NF-ToN-IoT confirms
that the RAG-based Experience Library is the critical per-
formance driver. Without retrieved context, GPT-4o achieves
only 17% and 4.96% macro F1 respectively. With Experi-
ence Library active, MA-IDS reaches 89.75% and 85.22%,
accumulating 7,322 and 9,498 detection rules through error-
driven induction, competitive with SVM while offering
superior interpretability and adaptability. Future work will
investigate lightweight deployment on resource-constrained
IoT devices, extension to open-set zero-day detection, and
integration of additional network telemetry to further enrich
the Experience Library.
REFERENCES
[1] Y . Li, Z. Xiang, N. D. Bastian, D. Song, and B. Li, “IDS-Agent: An
LLM Agent for Explainable Intrusion Detection in IoT Networks,”
Under Review at ICLR, 2025.
[2] Y . Ginige and T. Dahanayaka, “TrafficGPT: An LLM Approach for
Open-Set Encrypted Traffic Classification,” inProc. ACM Workshop
Privacy Electron. Soc., 2023.
[3] P. Zambare, V . N. Thanikella, N. P. Kottur, S. A. Akula, and Y . Liu,
“NetMoniAI: An Agentic AI Framework for Network Security &
Monitoring,”arXiv preprintarXiv:2508.10052, 2025.
[4] S. Yang et al., “Large Language Models for Network Intrusion Detec-
tion Systems: Foundations, Implementations, and Future Directions,”
arXiv preprintarXiv:2507.04752, 2025.
[5] S. Axelsson, “Research in intrusion-detection systems: A survey,”
Technical Report, Chalmers University of Technology, 1998.
[6] R. Bace and P. Mell, “Intrusion Detection Systems,” NIST Special
Publication 800-31, Nov. 2001.
[7] D. E. Denning, “An Intrusion-Detection Model,”IEEE Trans. Softw.
Eng., vol. SE-13, no. 2, pp. 222–232, Feb. 1987.
[8] A. Khraisat, I. Gondal, P. Vamplew, and J. Kamruzzaman, “Survey
of intrusion detection systems: techniques, datasets and challenges,”
Cybersecurity, vol. 2, no. 1, pp. 1–22, 2019.
[9] K. Scarfone and P. Mell, “Guide to Intrusion Detection and Prevention
Systems (IDPS),” NIST Special Publication 800-94, Feb. 2007.[10] P. Ioulianou, V . Vasilakis, I. Moscholios, and M. Logothetis, “A
signature-based intrusion detection system for the internet of things,”
inInformation and Communication Technology Forum, Jun. 2018.
[11] S. Nagaraju, B. Shanmugham, and K. Baskaran, “High throughput
token driven fsm based regex pattern matching for network intrusion
detection system,”Materials Today: Proceedings, vol. 47, pp. 139–
143, 2021.
[12] M. Y . AlYousef and N. T. Abdelmajeed, “Dynamically detecting
security threats and updating a signature-based intrusion detection
system’s database,”Procedia Computer Science, vol. 159, pp. 1507–
1516, 2019.
[13] F. Ullah et al., “Cyber security threats detection in internet of things
using deep learning approach,”IEEE Access, vol. 7, pp. 124379–
124389, 2019.
[14] A. Khraisat, I. Gondal, and P. Vamplew, “An anomaly intrusion
detection system using c5 decision tree classifier,” inPAKDD 2018
Workshops, pp. 149–155, 2018.
[15] H. Jia, J. Liu, M. Zhang, X. He, and W. Sun, “Network intrusion
detection based on ie-dbn model,”Computer Communications, vol.
178, pp. 131–140, 2021.
[16] I. Studnia et al., “A language-based intrusion detection approach for
automotive embedded networks,”Int. J. Embedded Systems, vol. 10,
no. 1, pp. 1–12, 2018.
[17] G. Kim, S. Lee, and S. Kim, “A novel hybrid intrusion detection
method integrating anomaly detection with misuse detection,”Expert
Systems with Applications, vol. 41, no. 4, pp. 1690–1700, 2014.
[18] K. Bajaj and A. Arora, “Dimension reduction in intrusion detection
features using discriminative machine learning approach,”Int. J.
Computer Science Issues, vol. 10, no. 4, p. 324, 2013.
[19] W. Li, P. Yi, Y . Wu, L. Pan, and J. Li, “A new intrusion detection sys-
tem based on knn classification algorithm in wireless sensor network,”
J. Electrical and Computer Engineering, vol. 2014.
[20] A. Guezzaz, S. Benkirane, M. Azrour, and S. Khurram, “A reliable net-
work intrusion detection approach using decision tree with enhanced
data quality,”Security and Communication Networks, vol. 2021, 2021.
[21] M. Mohammadi et al., “A comprehensive survey and taxonomy of
the svm-based intrusion detection systems,”J. Network and Computer
Applications, vol. 178, p. 102983, 2021.
[22] A. Verma and V . Ranga, “Machine learning based intrusion detection
systems for iot applications,”Wireless Personal Communications, vol.
111, no. 4, pp. 2287–2310, 2020.
[23] S. Roy, J. Li, B. Choi, and Y . Bai, “A lightweight supervised intrusion
detection mechanism for iot networks,”Future Generation Computer
Systems, vol. 127, pp. 276–285, 2022.
[24] P. J. Davis, S. Coffey, L. Beshaj, and N. D. Bastian, “Quantum
machine learning for feature selection in internet of things network
intrusion detection,” inQuantum Information Science, Sensing, and
Computation XVI, vol. 13028, SPIE, 2024.
[25] Z. Xu et al., “Deep Learning-based Intrusion Detection Systems: A
Survey,” arXiv:2504.07839, 2025.
[26] F. M. Anis, M. Alabdullatif, S. Aljbli and M. Hammoudeh, ”A Survey
on the Applications of Deep Learning in Network Intrusion Detection
Systems to Enhance Network Security,” in IEEE Access, vol. 13, 2025.
[27] B. K. Sedraoui, A. Benmachiche, A. Makhlouf and C. Chemam,
”Intrusion Detection with deep learning: A literature review,” 6th PAIS,
Algeria, 2024. doi: 10.1109/PAIS62114.2024.10541191
[28] M. A. Ferrag et al., “Deep Learning for Cyber Security Intrusion
Detection,”J. Information Security and Applications, 2019.
[29] Awajan, Albara. 2023. ”A Novel Deep Learning-Based Intrusion
Detection System for IoT Networks” Computers 12, no. 2: 34.
[30] A. A. Jihado and A. S. Girsang, ”Hybrid deep learning network
intrusion detection system based on convolutional neural network and
bidirectional long short-term memory,”JAIT, vol. 15, 2024.
[31] M. Keshk, N. Koroniotis, N. Pham, N. Moustafa, B. Turnbull, and A.
Y . Zomaya, ”An explainable deep learning-enabled intrusion detection
framework in IoT networks,”Information Sciences, vol. 639, 2023,
[32] S. I. Popoola, Y . Tsado, A. A. Ogunjinmi, E. Sanchez-Velazquez,
Y . Peng, and D. B. Rawat, ”Multi-stage deep learning for intrusion
detection in industrial internet of things,”IEEE Access, 2025,
[33] H. Zhang, A. B. Sediq, A. Afana, and M. Erol-Kantarci, “Large
language models in wireless application design: In-context learning-
enhanced automatic network intrusion detection,” arXiv:2405.11002,
2024.
[34] M. Luay, S. Layeghy, S. Hosseininoorbin, M. Sarhan, N. Moustafa,

and M. Portmann, ”Temporal analysis of NetFlow datasets for network
intrusion detection systems,” arXiv:2503.04404, 2025. [Online].