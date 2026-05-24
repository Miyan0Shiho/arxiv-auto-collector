# From Detection to Response: A Deep Learning and Retrieval-Augmented Generation Framework for Network Intrusion Mitigation

**Authors**: Md Navid Bin Islam, Sajal Saha, Senior Member

**Published**: 2026-05-18 07:17:55

**PDF URL**: [https://arxiv.org/pdf/2605.17960v1](https://arxiv.org/pdf/2605.17960v1)

## Abstract
Machine-learning-based Intrusion Detection Systems (IDS) have achieved impressive accuracy in classifying network attacks, yet they consistently fall short on the question that matters most to a security analyst: what should I do next? This paper presents a unified, end-to-end framework that closes the gap between threat detection and actionable response. The system operates in two tightly coupled stages. First, an ensemble of three independently trained binary Deep Neural Networks (DNNs) classifies network traffic flows as Benign, Denial of Service (DoS), or Distributed Denial of Service (DDoS), achieving 99.84% accuracy on the CICIDS2018 dataset and 95.30% on the UNSW-NB15 dataset. Second, a Retrieval-Augmented Generation (RAG) pipeline constructs explanation-aware prompts from the top-5 anomalous features, retrieves the most semantically and lexically relevant guidance from a knowledge base derived from authorized sources and di- rects a locally deployed language model to synthesise structured, citation-grounded mitigation reports. The RAG-enhanced reports outperform vanilla LLM outputs across all automated evaluation metrics.

## Full Text


<!-- PDF content starts -->

From Detection to Response: A Deep Learning and
Retrieval-Augmented Generation Framework for
Network Intrusion Mitigation
Md Navid Bin Islam
University of Northern British Columbia
Prince George, Canada
navid.islam@unbc.caSajal Saha, Senior Member (IEEE)
University of Northern British Columbia
Prince George, Canada
sajal.saha@unbc.ca
Abstract—Machine-learning-based Intrusion Detection Sys-
tems (IDS) have achieved impressive accuracy in classifying
network attacks, yet they consistently fall short on the question
that matters most to a security analyst:what should I do
next?This paper presents a unified, end-to-end framework that
closes the gap between threat detection and actionable response.
The system operates in two tightly coupled stages. First, an
ensemble of three independently trained binary Deep Neural
Networks (DNNs) [1] classifies network traffic flows as Benign,
Denial of Service (DoS) [2], or Distributed Denial of Service
(DDoS) [3], achieving 99.84% accuracy on the CICIDS2018 [4]
dataset and 95.30% on the UNSW-NB15 [5] dataset. Second, a
Retrieval-Augmented Generation (RAG) [6] pipeline constructs
explanation-aware prompts from the top-5 anomalous features,
retrieves the most semantically and lexically relevant guidance
from a knowledge base derived from authorized sources and di-
rects a locally deployed language model to synthesise structured,
citation-grounded mitigation reports. The RAG-enhanced reports
outperform vanilla LLM outputs across all automated evaluation
metrics.
Index Terms—Intrusion Detection Systems, Deep Learning,
Retrieval-Augmented Generation (RAG), Cybersecurity, Large
Language Models (LLMs), Denial of Service, Explainable AI
(XAI), NIST, MITRE ATT&CK.
I. INTRODUCTION
Modern computer networks are experiencing a rapid in-
crease in cyber threats, especially Denial of Service (DOS)
[2] and Distributed Denial of Service (DDoS) [3] attacks.
Recent industry reports show that more than 7.9 million
DDoS incidents were recorded globally during the first half
of 2024 alone [7]. As cyber-attacks continue to grow in scale
and complexity, machine learning-based Intrusion Detection
Systems (IDS) have become widely used because of their
ability to detect attacks with very high accuracy. However,
high detection accuracy alone is not sufficient in real-world
cybersecurity operations.
In practice, detecting an attack is only the first step. Security
analysts in a Security Operations Center (SOC) also need
to understand the nature of the attack, identify the affected
systems, and decide which mitigation actions should be taken
immediately. A prediction label such as “DDoS” does not
provide enough information to support rapid decision-makingduring a live incident. The delay between attack detection and
effective response can itself become a security risk, giving
attackers additional time to disrupt systems and services.
Several important challenges still remain in current IDS
research. First, many deep learning-based IDS models operate
as black-box systems that provide only prediction labels and
confidence scores without explaining how the decision was
made [8]. This makes it difficult for security analysts to
fully trust or interpret the model outputs. Second, Explainable
Artificial Intelligence (XAI) [9] techniques such as SHAP
[10] and LIME [11] can identify which features influenced
a prediction, but they still do not provide practical mitigation
guidance for responding to the attack. Third, valuable cyber-
security knowledge sources such as NIST [12] guidelines,
MITRE ATT&CK [13], and ENISA [14] recommendations
contain detailed mitigation strategies, but these documents are
extensive and difficult to consult during real-time incident
response.
To address these limitations, this paper proposes a modular
end-to-end framework that combines deep learning-based in-
trusion detection, and RAG for both attack detection and mit-
igation generation. The framework not only identifies cyber-
attacks with high accuracy, but also generates structured and
evidence-based mitigation reports using authoritative cyberse-
curity knowledge sources. The main contributions of this work
are summarized as follows:
1) We design a structured prompting strategy that combines
the predicted attack class, important network features,
flow metadata, and security interpretations to provide
context-aware inputs for the language model. This helps
the model generate mitigation guidance based on the
actual attack evidence detected by the IDS.
2) We develop a hybrid retrieval pipeline that combines
BM25 [15] keyword search, FAISS-based [16] dense
vector retrieval, and cross-encoder reranking to retrieve
relevant information from a cybersecurity knowledge
base built from authoritative sources including NIST
[12] and MITRE [13] documents.
3) Using the retrieved cybersecurity knowledge, the frame-
work generates structured mitigation reports with sup-arXiv:2605.17960v1  [cs.CR]  18 May 2026

porting citations from trusted sources. The generated
reports provide actionable recommendations for security
analysts during incident response.
4) We evaluate the quality of the generated mitigation
reports against expert-written ground truth explanations
using BERTScore, ROUGE, and BLEU metrics. Exper-
imental results show that the RAG-enhanced framework
produces more accurate and reliable outputs compared
to standalone LLM-based generation.
The remainder of this paper is organized as follows. Sec-
tion II surveys related work and identifies the specific gaps
this paper addresses. Section III presents the full methodol-
ogy, including datasets, DNN architectures, prompt construc-
tion, knowledge base design, retrieval architecture, and LLM
configuration. Section IV reports and discusses experimental
results. Section V concludes and outlines future work.
II. RELATEDWORK
Recent advancements in cybersecurity have significantly
improved the performance of IDS through the use of machine
learning and deep learning models. Traditional IDS approaches
mostly depended on signature-based or rule-based techniques,
which often struggle to detect previously unseen attacks and
complex traffic patterns [17]. To overcome these limitations,
researchers introduced deep learning architectures capable of
learning hidden representations directly from network traffic
data.
A. Deep Learning-Based Intrusion Detection Systems
Deep learning models such as Convolutional Neural Net-
works (CNNs) [32], Recurrent Neural Networks (RNNs) [33],
Long Short-Term Memory (LSTM) networks [34], and Au-
toencoders [33] have shown strong performance in intru-
sion detection tasks [22]. These models can automatically
learn complex spatial and temporal relationships from high-
dimensional network traffic data, making them more effective
than traditional machine learning approaches.
Kim et al. [18] introduced an LSTM-based IDS framework
capable of learning sequential traffic behavior for anomaly
detection. Their work demonstrated that recurrent architectures
can effectively capture dependencies in network flows. Simi-
larly, Yin et al. [19] evaluated RNN-based IDS models and
showed significant improvements over traditional classifiers
such as Support Vector Machines (SVMs) [35] and Random
Forests [36]. Their study highlighted the importance of feature
learning in network intrusion detection.
Shone et al. [20] proposed a non-symmetric deep au-
toencoder architecture combined with Random Forest [36]
classification for feature extraction and intrusion detection.
Their work demonstrated that unsupervised feature learning
can improve detection performance while reducing feature en-
gineering complexity. Vinayakumar et al. [21] later presented
a comprehensive deep learning framework for cyber threat
detection using datasets such as UNSW-NB15 [5] and CI-
CIDS2017 [4]. Their results showed that deep neural networkscan achieve high classification accuracy under complex attack
scenarios.
Although these approaches achieved impressive detection
performance, most of them focused mainly on attack classifi-
cation accuracy. They generally lacked explainability, contex-
tual reasoning, and actionable mitigation support for security
analysts.
B. Explainable Artificial Intelligence in IDS
As deep learning models became more complex, researchers
started investigating XAI techniques to improve trust and inter-
pretability in IDS. Security analysts often require explanations
regarding why a network flow was classified as malicious
before taking operational decisions.
Lundberg and Lee [10] introduced SHAP (SHapley Additive
exPlanations), which provides feature-level importance scores
for machine learning predictions. Ribeiro et al. [11] proposed
LIME (Local Interpretable Model-Agnostic Explanations), a
model-agnostic explanation framework capable of interpreting
local predictions. Both SHAP and LIME have been widely ap-
plied in cybersecurity research to improve IDS interpretability.
Several recent IDS studies integrated explainability mech-
anisms into deep learning pipelines. For example, Tjhai et
al. [31] used SHAP explanations to identify influential net-
work traffic features contributing to attack detection decisions.
Similarly, Loi et al. [30] demonstrated that explainable IDS
frameworks can significantly improve analyst understanding
and trust during threat investigation.
Despite these improvements, most explainable IDS systems
still focus only on feature attribution and attack interpretation.
They generally do not provide contextual mitigation recom-
mendations or operational guidance after detection.
C. Transformer and LLM-Based Cybersecurity Systems
The success of transformer architectures and LLMs has
influenced cybersecurity research. Transformer-based models
can capture contextual relationships better in sequential data
compared to traditional CNN and RNN architectures [21].
Ferrag et al. [23] introduced SecurityBERT, a lightweight
transformer-based intrusion detection model designed for
IoT and IIoT environments. Their framework used Privacy-
Preserving Fixed Length Encoding (PPFLE) to convert net-
work traffic into a textual representation suitable for trans-
former processing while preserving sensitive information.
Other cybersecurity-focused language models such as
SecBERT [24] and CyBERT [25] further demonstrated the
effectiveness of transformer architectures for threat analysis,
malware detection, and vulnerability understanding. However,
many transformer-based IDS systems still operate mainly
as classification frameworks. While they improve contextual
understanding, they often lack reliable retrieval mechanisms
and grounded mitigation generation pipelines [37].
D. RAG in Cybersecurity
RAG has emerged as a promising approach for improving
the reliability and factual grounding of LLMs [6]. Instead

TABLE I
COMPARISON OFEXISTINGIDS, XAI,ANDRAG-BASEDCYBERSECURITYSYSTEMS
Work DL-Based IDS XAI LLM/RAG Trusted KB Mitigation Hybrid Retrieval Detection-to-Mitigation
Lippmann et al. [17] ✗ ✗ ✗ ✗ ✗ ✗ ✗
Kim et al. [18] ✓ ✗ ✗ ✗ ✗ ✗ ✗
Yin et al. [19] ✓ ✗ ✗ ✗ ✗ ✗ ✗
Shone et al. [20] ✓ ✗ ✗ ✗ ✗ ✗ ✗
Vinayakumar et al. [21] ✓ ✗ ✗ ✗ ✗ ✗ ✗
Lansky et al. [22] ✓ ✗ ✗ ✗ ✗ ✗ ✗
Lundberg and Lee [10] ✗ ✓ ✗ ✗ ✗ ✗ ✗
Ribeiro et al. [11] ✗ ✓ ✗ ✗ ✗ ✗ ✗
Ferrag et al. [23] ✓ Partial ✓ ✗ ✗ ✗ ✗
SecBERT [24] Partial ✗ ✓ ✗ ✗ ✗ ✗
CyBERT [25] Partial ✗ ✓ ✗ ✗ ✗ ✗
MoRSE [26] ✗ ✗ ✓ ✓ Advisory Partial ✗
CyberRAG [27] Partial ✓ ✓ Partial ✓ Partial Partial
Setiawan & Soewito [28] ✗ ✗ ✓ ✓ Partial ✗ ✗
Pinto et al. [29] Partial ✗ ✗ ✗ Limited ✗ ✗
Loi et al. [30] ✓ ✓ ✗ ✗ ✗ ✗ ✗
Tjhai et al. [31] ✓ ✓ ✗ ✗ ✗ ✗ ✗
Proposed Framework ✓ ✓ ✓ ✓ ✓ ✓ ✓
of relying only on the model’s pre-trained knowledge, RAG
systems retrieve relevant external documents during response
generation.
In cybersecurity, RAG has been utilized to reduce halluci-
nations and improve the accuracy of generated security recom-
mendations. Simoni et al. [26] introduced MoRSE, a chatbot
that combines structured and unstructured retrieval pipelines to
answer cybersecurity-related questions. Their results showed
that retrieval-enhanced responses were significantly more ac-
curate and context-aware than standard LLM outputs.
Blefari et al. [27] proposed CyberRAG, an agent-based RAG
framework that combines multiple attack-specific classifiers
with retrieval-enhanced explanation generation. Their system
achieved strong attack reporting performance while improving
interpretability and operational usability.
Setiawan and Soewito [28] explored a multi-agent RAG
framework for automatic Snort rule generation using cyber-
security knowledge sources such as Common Weakness Enu-
meration (CWE). Their system demonstrated that instruction-
based prompting combined with retrieval mechanisms can
improve threat response automation.
E. Automated Threat Mitigation and Incident Response
Automated mitigation and incident response systems have
also got the attention in recent years due to the increasing
volume of security alerts generated by modern networks.
Traditional Security Orchestration, Automation, and Response
(SOAR) platforms help automate incident handling workflows
but often depend heavily on manually crafted rules and pre-
defined rule-books.
Pinto et al. [29] discussed the growing cybersecurity risks
in smart grid infrastructures and highlighted the limitations of
current detection and mitigation frameworks under real-world
attack scenarios. Their work emphasized the importance of in-tegrating intelligent detection systems with adaptive response
mechanisms.
Recent studies also explored the integration of cybersecu-
rity knowledge bases such as NIST [12], MITRE ATT&CK
[13], ENISA [38] into automated incident response systems.
However, most existing automated mitigation frameworks ei-
ther operate independently from IDS systems or rely heavily
on generic responses without incorporating contextual attack
evidence derived from network traffic analysis.
F . Research Gap and Motivation
When analyzing existing literature, several important lim-
itations become clear. Deep learning-based IDS frameworks
achieve strong attack detection performance but usually lack
explainability and mitigation support. Explainable IDS sys-
tems improve interpretability but generally stop at feature at-
tribution without generating actionable responses. Transformer
and LLM-based cybersecurity systems improve contextual
reasoning but often lack authoritative grounding and retrieval
validation. Similarly, many RAG-based cybersecurity frame-
works focus on threat retrieval rather than tightly integrating
intrusion detection with mitigation generation.
Another important limitation is that most existing works
do not address real-world operational challenges such as
confidence-aware analysis, hybrid retrieval architectures,
reranking mechanisms, and grounded mitigation reporting.
Very few studies provide an end-to-end framework that com-
bines intrusion detection, explainability, authoritative retrieval,
and context-aware mitigation generation in a unified pipeline.
Motivated by these limitations, this paper proposes a com-
plete IDS framework that combines deep learning-based in-
trusion detection with a RAG pipeline for mitigation re-
port generation. Unlike existing approaches, the proposed
framework integrates explanation-aware prompt construction,

hybrid lexical-semantic retrieval, cross-encoder reranking,
confidence-aware classification, and grounded mitigation gen-
eration using trusted cybersecurity sources. The system not
only detects attacks but also generates structured, actionable,
and citation-grounded mitigation reports suitable for real-
world cybersecurity operations.
III. PROPOSEDMETHODOLOGY
This section presents the complete technical design of the
proposed IDS–RAG framework. The system follows a five-
stage modular pipeline: (1) data acquisition and preprocessing,
(2) confidence-calibrated DNN classification, (3) explanation-
aware prompt construction, (4) hybrid RAG retrieval and
reranking, and (5) structured LLM report generation. The
overall architecture is illustrated in Fig. 1.
A. System Architecture Overview
The proposed pipeline ingests raw network flow records,
preprocesses them into a unified 66-dimensional feature space,
classifies each flow using an ensemble of three binary DNNs,
extracts the most anomalous features for prompt construction,
retrieves relevant cybersecurity guidance through a hybrid
retrieval system, and generates a structured mitigation report
using a locally deployed LLM. Each stage is independently
modular: the classifier can be retrained on new traffic data
without modifying the RAG system, and the knowledge base
can be extended with new authorative sources without retrain-
ing the classifier.
B. Stage 1: Data Acquisition and Preprocessing
We have used two publicly available benchmark datasets.
The CICIDS2018 dataset [4] provides 1.15 million flow
records spanning three classes (Benign, DoS, DDoS) across
66 flow-level features. The UNSW-NB15 dataset [39] provides
a complementary evaluation environment with 49 original
features across ten classes. To enable cross-dataset evaluation,
all nine attack classes in UNSW-NB15 are remapped to
a three-class scheme: complex multi-stage attacks (Fuzzers,
Exploits, Generic, Reconnaissance) are combined under DDoS
due to their distributed, high-impact characteristics, while
direct volumetric attacks are assigned to DoS.
Numeric features such as packet counts, byte counts, TTL
values etc. are retained as-is. Categorical features like protocol,
service, connection state etc. are one-hot encoded. Zero-
padding features are constructed to bridge the 49-to-66 dimen-
sional gap, ensuring that both datasets share an identical 66-
dimensional feature space. A feature interpretation dictionary
is maintained that maps each technical feature name to a
human-readable description and a security implication string.
Table II shows ten representative entries from this dictionary.
A scikit-learn [40] pipeline appliesz-score normalization to
all numeric features:
ˆxj=xj−µj
σj(1)
whereµ jandσ jare the mean and standard deviation of
featurejcomputed over the training split. Stratified 70/15/15TABLE II
REPRESENTATIVEFEATUREINTERPRETATIONDICTIONARYENTRIES
Feature Description Security Implication
sttl Source Time-to-
LiveIrregular values may
indicate spoofed
packets
ctstat ttl Conn. state transi-
tionsAbnormal state cycles
signal protocol abuse
ctsrvdst Conn. to same ser-
viceRepeated targeting of
one endpoint
sload Source byte
throughputHigh values consis-
tent with flooding
dload Destination byte
loadAsymmetric loads
suggest amplification
Flow Bytes/s Bytes per second High rate indicates
volumetric activity
Flow Packet-
s/sPackets per second Scan DoS behaviour
at high values
Init Win
BytesInitial TCP window Anomalous values
flag scan tools
Avg Packet
SizeMean payload bytes Small uniform sizes
indicate flood
Protocol L4 protocol number UDP/ICMP often
used in amplification
train/validation/test splits are used across both datasets to
preserve class proportions and ensure unbiased evaluation
under severe class imbalance.
C. Stage 2: DNN Classification
Rather than training a single three-way classifier, we deploy
an ensemble of three independently trained binary DNNs [41],
one per class Benign-vs-Rest, DoS-vs-Rest, DDoS-vs-Rest.
This one-vs-rest strategy facilitates the learning objective of
each model by allowing it to focus on a single decision
boundary rather than simultaneously separating multiple attack
categories.
1) Network Architectures:Two different DNN architectures
are employed depending on the distribution of the dataset.
For the CICIDS2018 [4] dataset, the model consists of three
fully connected hidden layers with dimensions[128,64,32].
Each hidden layer uses the ReLU activation function followed
by dropout regularization [42] with a dropout probability of
p= 0.3to reduce overfitting and improve generalization.
For the UNSW-NB15 [39] dataset, the model con-
tains four fully connected hidden layers with dimensions
[256,128,64,32]. ReLU activation is applied after each layer,
while batch normalization [43] is incorporated to stabilize
training and reduce internal covariate shift. In addition, gradu-
ated dropout regularization is used with probabilities decreas-
ing from0.4to0.2across the deeper layers.
Both architectures terminate with a softmax output
layer [44]. The full forward pass for an input flowx∈R66
through thek-layer network is:
h(0)=x(2)

Fig. 1. Proposed end-to-end framework for intrusion detection and RAG-enhanced mitigation generation. The architecture integrates DNN-based attack
classification, explanation-aware prompt construction, hybrid retrieval using BM25 and FAISS with cross-encoder reranking, and LLM-based mitigation report
generation using authoritative cybersecurity knowledge sources.
h(l)= ReLU
BN
W(l)h(l−1)+b(l)
, l= 1, . . . , k−1
(3)
ˆy= Softmax
W(k)h(k−1)+b(k)
(4)
whereW(l)andb(l)represent the learnable weights and
bias parameters of layerl, and BN denotes the batch normal-
ization operation, which is applied only in the UNSW-NB15
architecture.
2) Class Imbalance Problem:In the UNSW-NB15 [39]
dataset, DoS traffic constitutes only 0.64% of total samples.
Under standard cross-entropy training, this causes the opti-mizer to achieve a slightly low loss by predicting the majority
class for nearly all inputs, yielding DoS recall close to 0.
3) Class-Weighted Binary Cross-Entropy:To address this,
the binary cross-entropy loss is modified with inverse-
frequency class weights. For a single sample with true label
y∈ {0,1}and predicted probabilityˆy, the weighted loss is:
LWBCE(y,ˆy) =−
w1ylog(ˆy) +w 0(1−y) log(1−ˆy)
(5)
where the class weightsw care computed as:
wc=N
2Nc(6)

TABLE III
ABLATIONSTUDY: EFFECT OFEACHBALANCINGINTERVENTION ON
UNSW-NB15 DOS RECALL
Configuration DoS Recall (%)
Baseline (no balancing) 0.30
+ Class-weighted BCE 28.4
+ Random oversampling (1:5) 71.2
+ Controlled undersampling 84.9
+ F1-based model selection (Full System)97.6
TABLE IV
CONFIDENCETIERMAPPING FORSOC TRIAGE
Confidence Tier SOC Action
≥0.95Very High Fully automated response
[0.70,0.95)High Analyst review within 15 min
[0.50,0.70)Medium Analyst review within 1 hr
<0.50Low Manual deep inspection
Nis the total number of training samples andN cis the
number of samples in classc. The total training objective over
a minibatch ofNsamples is:
Ltotal=1
NNX
i=1LWBCE(yi,ˆyi)(7)
This formulation implements empirical risk minimization un-
der class-balanced constraints, ensuring that minority-class
errors receive proportionally larger gradient contributions [45].
4) Additional Balancing Interventions:Beyond loss
weighting, three further strategies are applied specifically for
the UNSW-NB15 DoS classifier: (i) random oversampling
of the minority DoS class to a 1:5 ratio relative to the
Benign class [46], (ii) controlled undersampling of the
Benign majority, and (iii) model selection based on macro-
averaged F1-score rather than accuracy. The combined
interventions raise DoS recall from 0.30% to 97.6%—a
325-fold improvement (Table III).
5) Ensemble Prediction and Confidence Calibration:All
three classifiers receive the same input flow simultaneously.
Letp cdenote the softmax probability assigned by classifierc∈
{Benign,DoS,DDoS}. The final prediction and confidence
are given by:
ˆc= arg max
cpc,Confidence = max
cpc (8)
The scalar confidence score is mapped to a discrete tier for
SOC prioritization, as shown in Table IV. The full procedure
is presented as Algorithm 1.
All models are trained with the Adam [47] optimizer, an
initial learning rate of1×10−3, batch size 512, and early
stopping with patience of 5 epochs monitored on validation
F1-score. Training converges within 14–19 epochs across all
classifiers on both datasets.Algorithm 1Ensemble Confidence-Calibrated Classification
Require:Feature vectorx∈R66; trained classifiers
fB, fD, fDD
Ensure:Predicted classˆc, confidenceconf, tier labelt
1:pB←fB(x)[1]{Benign probability}
2:pD←fD(x)[1]{DoS probability}
3:pDD←fDD(x)[1]{DDoS probability}
4:ˆc←arg max{p B, pD, pDD}
5:conf←max{p B, pD, pDD}
6:ifconf≥0.95then
7:t←VERYHIGH
8:else ifconf≥0.70then
9:t←HIGH
10:else ifconf≥0.50then
11:t←MEDIUM
12:else
13:t←LOW
14:end if
15:returnˆc,conf, t
D. Stage 3: Explanation-Aware Prompt Construction
Conventional approaches query an LLM with a generic
question such as “Describe mitigation strategies for a DDoS
attack” which produces textbook-level advice contextually
detached from the specific event. Our approach constructs a
structured, evidence-rich prompt that binds the LLM’s reason-
ing to the concrete evidence produced by the classifier.
1) Top-KFeature Selection:Following classification, the
K= 5features most influential to the model’s prediction
are extracted. Feature importance scores are computed using
gradient-based attribution [48]. The gradient of the predicted
class score with respect to each input dimension is computed
and its absolute value taken as a proxy for feature influence:
Importance(j) =∂ˆyˆc
∂xj(9)
Features are sorted in descending order of importance
and the top-5 selected. For the experiments in this paper,
five security-relevant features per dataset were additionally
identified through domain knowledge:
•CICIDS2018:Flow Bytes/s, Flow Packets/s, Init Win
Bytes Forward, Protocol, Average Packet Size.
•UNSW-NB15:sttl(source TTL),ct_state_ttl
(connection state transitions),dload(destination byte
load),sload(source throughput),ct_srv_dst(re-
peated service connections).
2) Feature Description Mapping:Each selected feature is
mapped through the interpretation dictionary (Table II) to
produce a natural-language evidence string. For example, a
flow withsttl= 245becomes“Source TTL of 245 is
elevated and irregular; may indicate packet spoofing or TTL
manipulation consistent with DoS traffic engineering, ”and
Flow Bytes/s= 4.7×106becomes“Abnormally high byte

TABLE V
PROMPTBLOCKSTRUCTURE ANDVARIABLESOURCES
Block Variables Source Stage
Detection
Contextattack_class,
confidence_score,
confidence_tier,
datasetStage 2 (DNN)
Flow Metadataflow_id,timestamp,
src_ip,src_port,
dst_ip,dst_port,
protocolStage 1 (prepro-
cessing)
Anomalous Indi-
catorsfeature_{1..K}_name,
feature_{1..K}_value,
feature_{1..K}_interpretationStage 3 (XAI)
Retrieved Knowl-
edgeretrieved_chunksStage 4 (RAG)
rate suggests sustained bulk data flooding, consistent with
volumetric denial-of-service activity. ”
3) Prompt Template Construction:The structured prompt
is assembled using LangChain’sPromptTemplateframe-
work [49] and comprises four semantically distinct blocks,
each contributing different contextual information to guide
report generation. Table V summarizes the four blocks, their
constituent variables, and the pipeline stage responsible for
populating each. The complete template is shown in Box 1.
The system role instruction fixes the LLM persona as a
senior cybersecurity analyst and imposes an explicit cita-
tion requirement for NIST Special Publications and MITRE
ATT&CK techniques, ensuring that generated reports remain
grounded in authoritative sources rather than relying on the
model’s parametric knowledge alone. The detection context
block provides the predicted attack class, scalar confidence
score, and SOC triage tier from Algorithm 1, binding the
model’s reasoning to the specific classification outcome. The
flow metadata block supplies network-level identifiers that
enable traceability in automated reporting pipelines and allow
the LLM to tailor its language to the operational context of
the event. The anomalous indicators block injects the top-
Kfeature–interpretation pairs produced in Stage 3, providing
the concrete evidence from which the model constructs its
rationale. The retrieved knowledge block is populated by the
RAG pipeline with the top-k= 5chunks selected by the cross-
encoder; in the vanilla LLM baseline condition this field is
left empty, isolating the contribution of retrieval augmentation.
Finally, the report request block specifies the five-section
output structure required of the model.
4) Metadata Enrichment:The prompt incorporates flow
metadata [50] (IP addresses, ports, protocol, timestamp,
dataset provenance, confidence tier) to enable traceability in
automated reporting systems and to allow the LLM to tailor
its language to the specific operational context.
E. Stage 4: RAG Pipeline
1) Comprehensive Knowledge Base Construction:The
RAG system’s retrieval quality depends on the quality of the
knowledge base it searches. We construct a domain-specific,Explanation-Aware Prompt Template
SYSTEM:
You are a senior cybersecurity analyst.
Generate a structured incident response
report
based on the evidence provided. Cite NIST
Special
Publications and MITRE ATT&CK techniques
explicitly
where applicable.
USER:
[DETECTION CONTEXT]
Attack class{attack_class}
Confidence{confidence_score}
({confidence_tier})
Dataset{dataset}
[NETWORK FLOW METADATA]
Flow ID{flow_id}
Timestamp{timestamp}
Source{src_ip}:{src_port}
Destination{dst_ip}:{dst_port}
Protocol{protocol}
[KEY ANOMALOUS INDICATORS]
Top-{K}features most influential in this
decision:
Feature 1 :{feature_1_name}=
{feature_1_value}
,→ {feature_1_interpretation}
...
Feature K :{feature_K_name}=
{feature_K_value}
,→ {feature_K_interpretation}
[RETRIEVED KNOWLEDGE CONTEXT]
{retrieved_chunks}▷injected by RAG pipeline
[REPORT REQUEST]
Generate a structured report with sections:
1. Rationale 4. Threat Assessment
2. Key Indicators 5. Recommendations
3. Confidence Assessment
authoritative knowledge base from fourteen carefully selected
cybersecurity documents. Nine NIST Special Publications [12]
were included: SP 800-61 Rev. 2 (incident handling), SP 800-
94 (IDS guidelines), SP 800-83 Rev. 1 (malware incident
response), SP 800-115 (security testing), SP 800-137 (con-
tinuous monitoring), SP 800-92 (log management), SP 800-
123, SP 800-125, and SP 800-144. The MITRE ATT&CK
Enterprise Framework [13] was included with particular em-
phasis on T1498 (Network Denial of Service) and T1498.001
(Direct Network Flood). Additional sources include the CIS
Critical Security Controls, SOC incident response playbooks,
and curated attack signature databases [51].
2) Semantic Chunking:Documents are segmented using a
semantic chunking strategy [52] rather than naive fixed-size
splitting. Chunk boundaries are placed at semantic units with
a 200-token overlap between adjacent chunks, preserving con-
ceptual continuity across boundaries. Each chunk is annotated
with: (i) traffic relevance label (Benign / DoS / DDoS), (ii)

source document identifier, (iii) NIST/MITRE reference, and
(iv) section identifier. The final knowledge base contains 5,234
semantically coherent, metadata-tagged chunks.
3) Hybrid Retrieval Architecture:The retrieval system
combines two complementary paradigms lexical and semantic
search followed by a reranking stage.
a) Lexical Retrieval:The lexical retrieval stage uses the
BM25 ranking function [15] to measure the relevance between
the user queryQand each knowledge documentD. The BM25
relevance score is computed as:
Score BM25(D, Q) =X
iIDF(q i)
×f(qi, D)(k 1+ 1)
f(qi, D) +k 1
1−b+b|D|
avgdl(10)
wheref(q i, D)denotes the frequency of query termq iin
documentD,|D|represents the document length, andavgdlis
the average document length across the corpus. The parameter
k1controls term frequency saturation, whilebdetermines
the degree of document length normalization. In this paper,
standard BM25 hyper-parametersk 1= 1.5andb= 0.75are
used.
b) Semantic Retrieval:All knowledge chunks are en-
coded into 768-dimensional dense vectors using theall-mpnet-
base-v2sentence transformer [53] and indexed in a FAISS [54]
approximate nearest-neighbor index. Cosine similarity [55]
between query embeddingqand chunk embeddingdis:
Sim(q,d) =q·d
∥q∥2∥d∥2(11)
c) Query Expansion:To improve retrieval coverage, the
query is expanded using a cybersecurity-specific synonym
dictionary that incorporates semantically related attack ter-
minology. For example, the term “DoS” is expanded with
related expressions such as “denial of service,” “resource
exhaustion,” and “flooding,” while “DDoS” is associated with
terms including “distributed denial of service,” “botnet flood,”
and “amplification attack.” The expanded query representation
is applied to both the lexical BM25 retrieval stage and the
semantic embedding-based retrieval stage in order to improve
recall and increase the possibility of retrieving contextually
relevant cybersecurity knowledge.
d) Score Fusion:Normalised BM25 and cosine similar-
ity scores are combined as a weighted fusion:
Score fusion(D, Q) = 0.60·Sim sem(D, Q)+0.40·Score BM25(D, Q)
(12)
The 60/40 weighting was determined empirically and
favours semantic relevance while preserving sensitivity to ex-
act technical terminology. The top 20–30 chunks are forwarded
to reranking.
e) Cross-Encoder Reranking:Thems-marco-MiniLM-
L-6-v2cross-encoder [56] encodes each query–chunk pair,
enabling fine-grained relevance interactions that bi-encoders
cannot capture. Each of the 20–30 candidates receives aAlgorithm 2Hybrid RAG Retrieval Pipeline
Require:Queryq(attack class + features + metadata); knowl-
edge base KB;k= 5
Ensure:Top-kranked chunksC∗
1:q exp←QueryExpand(q,thesaurus)
2:C bm25←BM25(q exp,KB, n=30)
3:q emb←Encode(q exp)
4:C sem←FAISS(q emb,KB, n=30)
5:C cand←Dedup(C bm25∪ C sem)
6:for alld∈ C canddo
7:s bm25←NormBM25(d, q exp)
8:s sem←Cos(q emb,Encode(d))
9:s fuse←0.60·s sem+ 0.40·s bm25
10:end for
11:C 30←Top-30{s fuse}
12:for alld∈ C 30do
13:s rr←CrossEnc(q, d)
14:end for
15:C∗←Top-k{s rr}
16:returnC∗
relevance score in[0,1], and the top-k= 5chunks are selected
as final retrieval results. The complete pipeline is presented in
Algorithm 2.
F . Stage 5: LLM-Based Report Generation
LLaMA 3:8B [57], accessed through the Ollama [58] frame-
work for fully local inference, serves as the generation back-
bone. Network traffic data processed by an IDS is typically
sensitive, and routing it to external API services creates both
privacy and regulatory risks.
1) Structured Report Format:The LLM is instructed via
the prompt template (Listing 1) to produce a five-section
incident response report, each section serving a distinct op-
erational function for SOC analysts. TheRationalesection
links the observed anomalous network features to the classified
attack type, with explicit citations to the applicable NIST
Special Publication and MITRE ATT&CK technique. TheKey
Indicatorssection enumerates the top-Kfeatures selected
in Stage 3 alongside their security interpretations, providing
the evidentiary basis for the classification. TheConfidence
Assessmentsection evaluates detection reliability by contex-
tualizing the scalar confidence score within the SOC triage
tier produced by Algorithm 1, allowing analysts to calibrate
their response urgency accordingly. TheThreat Assessment
section characterizes the operational impact of the detected
attack using the risk assessment principles of NIST SP 800-30,
including affected assets and potential consequences. Finally,
theRecommendationssection presents specific, prioritized
mitigation actions drawn directly from the top-kretrieved
knowledge chunks, each accompanied by its source citation.
The five-section structure is deliberately aligned with standard
SOC incident response workflows, ensuring that generated
reports are immediately actionable without requiring analyst
reformatting or supplementary research.

2) Fallback Mechanism:If retrieval returns fewer than three
chunks with cross-encoder score above 0.5, a fallback knowl-
edge context containing general NIST SP 800-61 incident
response principles [59] is injected to guarantee consistent
output quality.
3) Vanilla LLM Baseline:To isolate the contribution
of RAG, a vanilla LLM baseline receives the
identical explanation-aware prompt but with the
{retrieved_chunks}field replaced by an empty
string. This controls for prompt design effects and isolates
the value of retrieval augmentation.
4) Expert Ground Truth Development:To enable rigor-
ous evaluation, expert-aligned ground truth explanations were
constructed from primary sources rather than researcher-
authored text. Ground truth texts were merged from NIST
SP 800-61, NIST SP 800-94 [60], MITRE ATT&CK
T1498/T1498.001 [13], and RFC 4732 [61]. Three ground
truth documents were produced, one per class:
•Benign:RFC-compliant protocol interactions, balanced
bidirectional traffic, human-paced timing, correct TCP
handshake sequences.
•DoS:single-source high-volume traffic, abnormal packet
rates, resource exhaustion patterns, NIST SP 800-61 [12]
incident handling procedures, MITRE T1498 [13] indi-
cators, rate limiting and SYN cookie recommendations.
•DDoS:distributed traffic sources, synchronized attack
timing, botnet coordination, large-scale bandwidth con-
sumption, DNS/NTP amplification, RFC 4732 [61] mit-
igation guidance, MITRE T1498.001 [62] countermea-
sures.
Each ground truth document is 300–600 words, matching
the target length of generated explanations to avoid artificial
length-mismatch penalties.
G. Implementation Details
The model has been developed using Python 3.11 with
PyTorch 2.1.0 for neural network training. Neural network
training has been done on an amd Ryzen-5 3600X workstation
with 32 GB RAM and NVIDIA RTX 2060 GPU running
Windows 11. The RAG pipeline integrates FAISS [54] 1.7.4
for vector search, LangChain [63] 0.1.0 for LLM management,
and LLaMA 3:8B [57] with 4-bit quantization using Ollama
[64]. The model also uses OpenAI’s text-embedding-ada-002
[65] model for all embeddings.
IV. RESULT& DISCUSSION
This section evaluates the proposed IDS–RAG framework
across both detection and mitigation tasks. The classifica-
tion performance of the ensemble DNN is analyzed on the
CICIDS2018 [66] and UNSW-NB15 [39] datasets, followed
by an evaluation of the hybrid retrieval pipeline and the
quality of the generated mitigation reports. Finally, qualitative
analysis and operational feasibility are discussed to assess the
framework’s practicality in real-world SOC environments.TABLE VI
CICIDS2018 CLASSIFICATIONPERFORMANCE
Class Precision Recall F1 Support
Benign 1.00 1.00 1.00 116,248
DoS 0.99 1.00 0.99 58,342
DDoS 1.00 1.00 1.00 75,460
Accuracy 1.00 (250,050 samples)
Macro Avg 1.00 1.00 1.00 —
Weighted Avg 1.00 1.00 1.00 —
TABLE VII
UNSW-NB15 CLASSIFICATIONPERFORMANCE(BALANCEDTESTSET)
Class Precision Recall F1 Support
Benign 1.0000 0.9940 0.9970 38,420
DoS 0.9673 0.8889 0.9264 38,115
DDoS 0.8978 0.9760 0.9353 38,465
Accuracy 0.9530 (115,000 samples)
Macro Avg 0.9550 0.9530 0.9529 —
Weighted Avg 0.9550 0.9530 0.9529 —
A. Classification Performance on CICIDS2018
Our proposed DNN architecture achieves an overall accu-
racy of99.84%on the CICIDS2018 test set (250,050 flows).
Table VI reports per-class precision, recall, and F1-score.
DDoS detection is near-perfect (F 1= 0.9999), with zero
missed DDoS flows. DoS detection achievesF 1= 0.9935.
The only meaningful error source is 150 misclassified Be-
nign flows out of 250,050 (0.06% false-positive rate), well
below thresholds that would cause alert fatigue in operational
systems. ROC-AUC values are≈1.0for all three classifiers
(Fig. 2), and confidence scores cluster tightly in[0.98,1.00]
for correct predictions.
Training dynamics confirm robust generalization: all clas-
sifiers converge within 14–19 epochs with a train/validation
accuracy gap below 0.0002 at convergence, and validation loss
plateaus at 0.0008–0.0010, confirming that dropout and batch
normalization effectively prevent overfitting (Figs. 4 and 5).
B. Classification Performance on UNSW-NB15
The UNSW-NB15 experiment is the more challenging eval-
uation: DoS traffic constitutes only 0.64% of total samples—a
156:1 class imbalance. Without balancing interventions, DoS
recall collapses to 0.30%. With the full four-part balancing
strategy, DoS recall rises to97.6%. Overall accuracy on a
balanced test set reaches95.30%(Table VII). ROC-AUC
values are 1.000 for Benign and 0.977 for both attack classes
(Fig. 3).
C. Cross-Dataset Generalization
The 4.54% accuracy difference between CICIDS2018
(99.84%) and UNSW-NB15 (95.30%) is mainly due to dif-

ferences in dataset characteristics, including feature distribu-
tions, traffic-generation environments rather than overfitting.
Despite these, the proposed system still achieves above 95%
accuracy on UNSW-NB15 [39] without requiring any major
architectural changes. Only the balancing strategies, such
as loss weighting and oversampling, were adjusted for the
dataset, demonstrating the strong generalization capability of
the ensemble architecture.
Fig. 2. ROC Curves for CSECICIDS2018 Dataset
Fig. 3. ROC Curves for UNSW-NB15 Dataset
D. Model Training and Validation Analysis
Figs. 4 and 5 show that all three classifiers converge
smoothly with minimal train/validation divergence. Final vali-TABLE VIII
RETRIEVALPERFORMANCE ANDKNOWLEDGEGROUNDING
Metric Value
Retrieval Performance
Precision@5 0.91
Recall@5 0.84
Mean Reciprocal Rank (MRR) 0.87
Retrieval Success Rate 0.97
Average Retrieval Latency 1.8 s
Knowledge Grounding
NIST Citation Rate 93.3%
MITRE Citation Rate 86.7%
Avg. Citations per Explanation 4.2
dation losses of 0.0008–0.0010 confirm successful regulariza-
tion through dropout, batch normalization, and early stopping.
The operating points achieved (TPR>95%, FPR<5%)
are critical for preventing alert fatigue in operational SOC
deployments [67].
E. Retrieval Performance and Knowledge Grounding
Table VIII summarizes the retrieval pipeline’s performance
on 33 evaluation queries. Precision@5 of 0.91 indicates that
91% of the five returned chunks are genuinely relevant;
Recall@5 of 0.84 indicates that 84% of all relevant chunks
appear in the top-5; MRR of 0.87 confirms that the most
relevant chunk typically ranks at or near position 1. The 97%
retrieval success rate (32/33 queries successfully resolved) at
an average latency of 1.8 s demonstrates both effectiveness
and practical feasibility.
Knowledge grounding metrics show that NIST publications
[12] are cited in 93.3% of reports and MITRE ATT&CK
[13] techniques in 86.7%, with an average of 4.2 authoritative
citations per report. This citation density is a key differentiator
from vanilla LLM reports and directly supports compliance
with security documentation standards such as ISO 27035 [68]
and NIST [60] incident response requirements.
F . RAG vs. Vanilla LLM: Quantitative Evaluation
Table IX presents the NLP evaluation results comparing
vanilla and RAG-enhanced reports against expert ground
truth, computed using BERTScore [69], ROUGE [70], and
BLEU [71]. The RAG pipeline outperforms the vanilla base-
line across every metric. BERTScore F1 improves by 4.4%,
reflecting better semantic alignment. ROUGE-1 improves by
32.5% and ROUGE-2 by 126.4%, indicating substantially
better coverage of critical unigrams and the technical bigram
phrases (e.g., “rate limiting”, “SYN cookie”, “upstream fil-
tering”) that characterize authoritative mitigation guidance.
BLEU improves by 244.1%, the largest relative gain, reflecting
the RAG pipeline’s ability to reproduce specific technical
phraseology directly from retrieved NIST [60] / MITRE [13]
sources.

Fig. 4. Validation Accuracy Progression for Benign, DoS, and DDoS Classifiers
Fig. 5. Training and Validation Loss for Benign, DoS, and DDoS Classifiers
TABLE IX
EXPLANATIONQUALITY: VANILLALLMVS. RAG-ENHANCEDLLM
(BOTH VS. EXPERTGROUNDTRUTH)
Metric Vanilla RAG-Enhanced∆
BERTScore F1 0.8659 0.9038 +4.4%
BERTScore Precision 0.8784 0.9080 +3.4%
BERTScore Recall 0.8558 0.9016 +5.4%
ROUGE-1 0.4733 0.6271 +32.5%
ROUGE-2 0.1387 0.3141 +126.4%
ROUGE-L 0.2219 0.3916 +76.5%
BLEU 0.0570 0.1961 +244.1%
All improvements are statistically significant atp <0.001
under a paired Wilcoxon signed-rank test [72] across 36 gener-
ated reports. The magnitude of the ROUGE-2 and BLEU gains
both measuring phrase-level precision rather than semantic
similarity which is particularly informative. These metrics
reward exactn-gram matches with the ground truth, andthe only way to achieve them is to use the same techni-
cal terminology as the expert references, which the RAG
pipeline does by retrieving and incorporating text directly
from those references. Fig. 6 confirms visually that RAG-
enhanced outputs dominate vanilla outputs on all five quality
dimensions: Technical Accuracy, Domain Knowledge, Detail
Level, Professional Terminology, and Actionability.
G. Qualitative Analysis of Generated Reports
Across the 36 mitigation reports reviewed for the CI-
CIDS2018 [66] and UNSW-NB15 [39] datasets, the sys-
tem consistently produces structured, operationally appropri-
ate output. For DDoS attacks, reports correctly escalate to
infrastructure-level responses rather than host-level counter-
measures. For DoS attacks, reports focus on rate limiting,
SYN cookie activation, and source IP blocking. For Benign
traffic, reports recommend continued monitoring and baseline
validation, appropriately avoiding false-alarm escalation. The
structural layout of a typical generated report is shown in
Fig. 7.

Fig. 6. Performance Quality
Core LLM Reasoning
Synthesizes classification output, feature
indicators, and retrieved knowledge chunks
Analytical Summary
Key indicators, anomaly pat-
terns, and supporting evidence
Feature Vector Summary
Feature values and interpre-
tation of security relevance
Conclusion
Final determination (Benign / DoS /
DDoS) with confidence-aligned assessment
Fig. 7. Structure of the explanation generated by the LLM.
Vanilla LLM reports are not factually incorrect; they identify
rate limiting and monitoring as relevant but remain at a generic
level that is not suitable for direct SOC use. They contain no
specific NIST [8] control references, no MITRE [13] technique
identifiers, and no tailored procedures. An analyst receiving a
vanilla report would need to independently research applica-
ble frameworks before acting; an analyst receiving a RAG-
enhanced report can proceed directly to implementation.
Overall, the experimental results demonstrate that the pro-
posed IDS–RAG framework effectively bridges the gap be-
tween attack detection and actionable response. The ensembleDNN provides reliable traffic classification, while the hybrid
retrieval pipeline grounds the generated reports in authoritative
cybersecurity knowledge. As a result, the system produces
mitigation reports that are more accurate, explainable, context-
aware, and operationally useful than those generated by vanilla
language models.
Key Summary
•The proposed ensemble DNN achieved strong detec-
tion performance with 95%—99% accuracy.
•The hybrid BM25 + FAISS retrieval pipeline
achieved 97% retrieval success while grounding re-
ports in trusted cybersecurity knowledge sources.
•RAG-enhanced reports substantially outperformed
vanilla LLM outputs across BERTScore, ROUGE,
and BLEU metrics.
•The framework successfully bridges the gap between
intrusion detection and actionable incident response
by generating explainable, context-aware, and opera-
tionally useful mitigation reports.
V. CONCLUSION
This paper has presented an end-to-end intrusion detec-
tion and mitigation framework that combines a confidence-
calibrated ensemble of binary DNNs with a hybrid Retrieval-
Augmented Generation pipeline grounded in authoritative cy-
bersecurity knowledge. The framework addresses a persistent
and important gap in the IDS literature: the absence of
actionable, explainable, and source-grounded guidance at the
point of detection. The DNN achieves 99.84% accuracy on
CICIDS2018 and 95.30% on UNSW-NB15; the RAG pipeline
retrieves relevant guidance with 97% success from a 5,234-
chunk knowledge base, and generates structured 400–500-
word mitigation reports. RAG-enhanced reports outperform
vanilla LLM outputs by 32.5% ROUGE-1, 126.4% ROUGE-
2, and 244.1% BLEU against expert ground truth, with all
differences significant atp <0.001.
In the future, we plan to extend the classifier to a broader
attack taxonomy; implementing multi-flow temporal analysis
for slow-rate and distributed attacks and continuously updating
the knowledge base as new NIST Publications and MITRE
ATT&CK entries are released.
REFERENCES
[1] K. Fukushima, “Neocognitron: A self-organizing neural network model
for a mechanism of pattern recognition unaffected by shift in position,”
Biological cybernetics, vol. 36, no. 4, pp. 193–202, 1980.
[2] A. Hussain, J. Heidemann, and C. Papadopoulos, “A framework for
classifying denial of service attacks,” inProceedings of the 2003
conference on Applications, technologies, architectures, and protocols
for computer communications, 2003, pp. 99–110.
[3] J. Mirkovic and P. Reiher, “A taxonomy of ddos attack and ddos
defense mechanisms,”ACM SIGCOMM Computer Communication Re-
view, vol. 34, no. 2, pp. 39–53, 2004.
[4] I. Sharafaldin, A. Habibi Lashkari, and A. A. Ghorbani, “Toward gener-
ating a new intrusion detection dataset and intrusion traffic characteriza-
tion,” inProceedings of the 4th International Conference on Information
Systems Security and Privacy (ICISSP 2018). SCITEPRESS - Science
and Technology Publications, 2018, pp. 108–116.

[5] N. Moustafa and J. Slay, “Unsw-nb15: A comprehensive data set for
network intrusion detection systems (unsw-nb15 network data set),” in
2015 Military Communications and Information Systems Conference
(MilCIS). IEEE, 2015, pp. 1–6.
[6] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschelet al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Advances in
neural information processing systems, vol. 33, pp. 9459–9474, 2020.
[7] J. John and E. A. Fraser, “Ddos attacks on cloud computing and iot
devices: Strategies for mitigation,”Kasu J. Comput. Sci., vol. 1, no. 4,
pp. 778–795, 2024.
[8] S. Neupane, J. Ables, W. Anderson, S. Mittal, S. Rahimi, I. Banicescu,
and M. Seale, “Explainable intrusion detection systems (x-ids): A survey
of current methods, challenges, and opportunities,”IEEE Access, vol. 10,
pp. 112 392–112 415, 2022.
[9] D. Gunning, “Explainable artificial intelligence (xai),”Defense advanced
research projects agency (DARPA), nd Web, vol. 2, no. 2, p. 1, 2017.
[10] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model
predictions,”Advances in neural information processing systems, vol. 30,
2017.
[11] M. T. Ribeiro, S. Singh, and C. Guestrin, “Lime: Local interpretable
model-agnostic explanations,” inProceedings of the 22nd ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining
(KDD), 2016, pp. 2145–2154.
[12] D. P. M ¨oller, “Nist cybersecurity framework and mitre cybersecurity
criteria,” inGuide to Cybersecurity in Digital Transformation: Trends,
Methods, Technologies, Applications and Best Practices. Springer,
2023, pp. 231–271.
[13] MITRE Corporation, “Enterprise ATT&CK matrix,”
https://attack.mitre.org/matrices/enterprise/, 2025, accessed: May
12, 2026.
[14] H. ˙Ipek and H. Y ¨uksel, “The institutional and structural transformation
of the european union agency for cybersecurity (enisa),”Journal of
International Relations and Political Science Studies, no. 14, pp. 28–
64.
[15] S. Robertson and H. Zaragoza,The probabilistic relevance framework:
BM25 and beyond. Now Publishers Inc, 2009, vol. 4.
[16] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with gpus,”arXiv, 2017.
[17] R. P. Lippmann, D. J. Fried, I. Graf, J. W. Haines, K. R. Kendall, D. Mc-
Clung, D. Weber, S. E. Webster, D. Wyschogrod, R. K. Cunningham
et al., “Evaluating intrusion detection systems: The 1998 darpa off-
line intrusion detection evaluation,” inProceedings DARPA Information
survivability conference and exposition. DISCEX’00, vol. 2. IEEE,
2000, pp. 12–26.
[18] G. Kim, H. Yi, J. Lee, Y . Paek, and S. Yoon, “Lstm-based system-
call language modeling and robust ensemble method for designing host-
based intrusion detection systems,”arXiv preprint arXiv:1611.01726,
2016.
[19] C. Yin, Y . Zhu, J. Fei, and X. He, “A deep learning approach for
intrusion detection using recurrent neural networks,”Ieee Access, vol. 5,
pp. 21 954–21 961, 2017.
[20] N. Shone, T. N. Ngoc, V . D. Phai, and Q. Shi, “A deep learning approach
to network intrusion detection,”IEEE transactions on emerging topics
in computational intelligence, vol. 2, no. 1, pp. 41–50, 2018.
[21] R. Vinayakumar, M. Alazab, K. Soman, P. Poornachandran, and A. Al-
Nemrat, “Deep learning approach for intelligent intrusion detection
system,”IEEE Access, vol. 7, pp. 41 525–41 550, 2019.
[22] J. Lansky, S. Ali, M. Mohammadi, M. K. Majeed, S. H. T. Karim,
S. Rashidi, M. Hosseinzadeh, and A. M. Rahmani, “Deep learning-based
intrusion detection systems: a systematic review,”IEEE Access, vol. 9,
pp. 101 574–101 599, 2021.
[23] M. A. Ferrag, M. Ndhlovu, N. Tihanyi, L. C. Cordeiro, M. Debbah,
T. Lestable, and N. S. Thandi, “Revolutionizing cyber threat detection
with large language models: A privacy-preserving bert-based lightweight
model for iot/iiot devices,”IEEe Access, vol. 12, pp. 23 733–23 750,
2024.
[24] H. Huang and Y . Wang, “Secbert: Privacy-preserving pre-training based
neural network inference system,”Neural Networks, vol. 172, p. 106135,
2024.
[25] P. Ranade, A. Piplai, A. Joshi, and T. Finin, “Cybert: Contextualized
embeddings for the cybersecurity domain,” in2021 IEEE international
conference on big data (Big Data). IEEE, 2021, pp. 3334–3342.[26] M. Simoni, A. Saracino, V . P, and M. Conti, “Morse: Bridging the gap
in cybersecurity expertise with retrieval augmented generation,” inPro-
ceedings of the 40th ACM/SIGAPP Symposium on Applied Computing,
2025, pp. 1213–1222.
[27] F. Blefari, C. Cosentino, F. A. Pironti, A. Furfaro, and F. Marozzo,
“Cyberrag: An agentic rag cyber attack classification and reporting tool,”
arXiv preprint arXiv:2507.02424, 2025.
[28] V . Setiawan and B. Soewito, “Instruction-based chain-of-thought for
multi-agent rag in snort rule generation.”International Journal of
Intelligent Engineering & Systems, vol. 18, no. 11, 2025.
[29] S. J. Pinto, P. Siano, and M. Parente, “Review of cybersecurity analysis
in smart distribution systems and future directions for using unsuper-
vised learning methods for cyber detection,”Energies, vol. 16, no. 4, p.
1651, 2023.
[30] P. Loi, D. Canavese, L. Regano, D. Maiorca, and G. Giacinto, “Shap
happens: an explainable ids for industrial iot networks,” in2025 IEEE
9th Forum on Research and Technologies for Society and Industry
(RTSI). IEEE, 2025, pp. 71–76.
[31] G. Tjhai, D. Papamartzivanos, S. Furnell, and M. Papadaki, “Explaining
the decisions of an intrusion detection system: A case study,”Journal
of Information Security and Applications, vol. 68, p. 103233, 2022.
[32] Y . LeCun and Y . Bengio, “Convolutional networks for images, speech,
and time series,”The handbook of brain theory and neural networks,
1998.
[33] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, “Learning repre-
sentations by back-propagating errors,”nature, vol. 323, no. 6088, pp.
533–536, 1986.
[34] S. Hochreiter and J. Schmidhuber, “Long short-term memory,”Neural
computation, vol. 9, no. 8, pp. 1735–1780, 1997.
[35] C. Cortes and V . Vapnik, “Support-vector networks,”Machine learning,
vol. 20, no. 3, pp. 273–297, 1995.
[36] L. Breiman, “Random forests,”Machine learning, vol. 45, no. 1, pp.
5–32, 2001.
[37] T. Sandaruwan, J. Wijayanayake, and J. Senanayake, “Leveraging large
language models in cybersecurity: A systematic review of emerging
methods and techniques,”DRC, p. 155, 2024.
[38] European Union Agency for Cybersecurity, “Soc good practices:
Operational controls and workflows for security operations centres,”
ENISA, Tech. Rep., 2022, accessed: 2026-05-14. [Online]. Available:
https://www.enisa.europa.eu/publications/soc-good-practices
[39] N. Moustafa and J. Slay, “Unsw-nb15: a comprehensive data set for
network intrusion detection systems (unsw-nb15 network data set),”
in2015 military communications and information systems conference
(MilCIS). IEEE, 2015, pp. 1–6.
[40] R. Garreta, G. Moncecchi, T. Hauck, and G. Hackeling,Scikit-learn:
machine learning simplified: implement scikit-learn into every step of
the data science pipeline. Packt Publishing Ltd, 2017.
[41] C. Yuan and S. S. Agaian, “A comprehensive review of binary neural
network,”Artificial Intelligence Review, vol. 56, no. 11, pp. 12 949–
13 013, 2023.
[42] Y .-D. Zhang, C. Pan, J. Sun, and C. Tang, “Multiple sclerosis identi-
fication by convolutional neural network with dropout and parametric
relu,”Journal of computational science, vol. 28, pp. 1–10, 2018.
[43] W. Jung, D. Jung, B. Kim, S. Lee, W. Rhee, and J. H. Ahn, “Restruc-
turing batch normalization to accelerate cnn training,”Proceedings of
machine learning and systems, vol. 1, pp. 14–26, 2019.
[44] X. Liang, X. Wang, Z. Lei, S. Liao, and S. Z. Li, “Soft-margin
softmax for deep classification,” inInternational Conference on Neural
Information Processing. Springer, 2017, pp. 413–421.
[45] Z. Xu, C. Dan, J. Khim, and P. Ravikumar, “Class-weighted classifica-
tion: Trade-offs and robust approaches,” inInternational conference on
machine learning. PMLR, 2020, pp. 10 544–10 554.
[46] R. Mohammed, J. Rawashdeh, and M. Abdullah, “Machine learning with
oversampling and undersampling techniques: overview study and exper-
imental results,” in2020 11th international conference on information
and communication systems (ICICS). IEEE, 2020, pp. 243–248.
[47] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,”
arXiv preprint arXiv:1412.6980, 2014.
[48] M. Ancona, E. Ceolini, C. ¨Oztireli, and M. Gross, “Gradient-based
attribution methods,” inExplainable AI: Interpreting, explaining and
visualizing deep learning. Springer, 2019, pp. 169–191.
[49] X. Liu, J. Wang, X. Yuan, J. Sun, G. Dong, P. Di, W. Wang, and
D. Wang, “Prompting frameworks for large language models: A survey,”
ACM Computing Surveys, 2023.

[50] L. Peel, D. B. Larremore, and A. Clauset, “The ground truth about
metadata and community detection in networks,”Science advances,
vol. 3, no. 5, p. e1602548, 2017.
[51] M. Y . AlYousef and N. T. Abdelmajeed, “Dynamically detecting security
threats and updating a signature-based intrusion detection system’s
database,”Procedia Computer Science, vol. 159, pp. 1507–1516, 2019.
[52] R. Qu, R. Tu, and F. Bao, “Is semantic chunking worth the computational
cost?” inFindings of the Association for Computational Linguistics:
NAACL 2025, 2025, pp. 2155–2177.
[53] C. Galli, N. Donos, and E. Calciolari, “Performance of 4 pre-trained
sentence transformer models in the semantic query of a systematic
review dataset on peri-implantitis,”Information, vol. 15, no. 2, p. 68,
2024.
[54] D. Danopoulos, C. Kachris, and D. Soudris, “Approximate similarity
search with faiss framework using fpgas on the cloud,” inInternational
Conference on Embedded Computer Systems. Springer, 2019, pp. 373–
386.
[55] D. Gunawan, C. Sembiring, and M. A. Budiman, “The implementation
of cosine similarity to calculate text relevance between two documents,”
inJournal of physics: conference series, vol. 978, no. 1. IOP Publishing,
2018, p. 012120.
[56] S. Ravishankar and P. Varalakshmi, “Novel hybrid retrieval and rerank-
ing with score fusion for advanced financial question answering using
large language models,” in2025 IEEE Silchar Subsection Conference
(SILCON). IEEE, 2025, pp. 1–6.
[57] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux,
T. Lacroix, B. Rozi `ere, N. Goyal, E. Hambro, F. Azharet al.,
“Llama: Open and efficient foundation language models,”arXiv preprint
arXiv:2302.13971, 2023.
[58] F. S. Marcondes, A. Gala, R. Magalh ˜aes, F. Perez de Britto, D. Dur ˜aes,
and P. Novais, “Using ollama,” inNatural Language Analytics with
Generative Large-Language Models: A Practical Approach with Ollama
and Open-Source LLMs. Springer, 2025, pp. 23–35.
[59] D. Bodeau and R. Graubart, “Cyber resiliency and nist special publica-
tion 800-53 rev. 4 controls,” 2013.
[60] National Institute of Standards and Technology, “Security and privacy
controls for information systems and organizations,” U.S. Department
of Commerce, Gaithersburg, MD, Tech. Rep. NIST SP 800-53, Revision
5, 2020. [Online]. Available: https://doi.org/10.6028/NIST.SP.800-53r5
[61] IAB, “Rfc 4732: Internet denial-of-service considerations,” 2006.
[62] The MITRE Corporation, “Common weakness enumeration (cwe) ver-
sion 4.19.1,” https://cwe.mitre.org/data/index.html, January 2026, ac-
cessed: April 29, 2026.
[63] H. Chase, “LangChain,” Oct. 2022. [Online]. Available:
https://github.com/langchain-ai/langchain
[64] Ollama Team, “Ollama,” 2024. [Online]. Available: https://ollama.com
[65] OpenAI, “text-embedding-ada-002: Openai embedding model,”
https://platform.openai.com/docs/guides/embeddings, 2022.
[66] Canadian Institute for Cybersecurity, “CICIDS2018 intrusion detection
dataset,” https://www.unb.ca/cic/datasets/ids-2018.html, 2018.
[67] J. L. Leevy, J. Hancock, R. Zuech, and T. M. Khoshgoftaar, “Detecting
cybersecurity attacks across different network features and learners,”
Journal of Big Data, vol. 8, no. 1, p. 38, 2021.
[68] ISO/IEC, “Information technology — information security
incident management — part 1: Principles and process,”
International Organization for Standardization, Geneva, Switzerland,
Tech. Rep. ISO/IEC 27035-1:2023, 2023. [Online]. Available:
https://www.iso.org/standard/78973.html
[69] T. Zhang, V . Kishore, F. Wu, K. Q. Weinberger, and Y . Artzi, “Bertscore:
Evaluating text generation with bert,”arXiv preprint arXiv:1904.09675,
2019.
[70] C.-Y . Lin, “Rouge: A package for automatic evaluation of summaries,”
inText summarization branches out, 2004, pp. 74–81.
[71] M. Post, “A call for clarity in reporting bleu scores,” inProceedings of
the third conference on machine translation: Research papers, 2018, pp.
186–191.
[72] F. Wilcoxon, “Individual comparisons by ranking methods,”Biometrics
Bulletin, vol. 1, no. 6, pp. 80–83, 1945.