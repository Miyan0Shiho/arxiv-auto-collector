# Memory-Augmented Log Analysis with Phi-4-mini: Enhancing Threat Detection in Structured Security Logs

**Authors**: Anbi Guo, Mahfuza Farooque

**Published**: 2025-10-01 05:23:05

**PDF URL**: [http://arxiv.org/pdf/2510.00529v1](http://arxiv.org/pdf/2510.00529v1)

## Abstract
Structured security logs are critical for detecting advanced persistent
threats (APTs). Large language models (LLMs) struggle in this domain due to
limited context and domain mismatch. We propose \textbf{DM-RAG}, a dual-memory
retrieval-augmented generation framework for structured log analysis. It
integrates a short-term memory buffer for recent summaries and a long-term
FAISS-indexed memory for historical patterns. An instruction-tuned Phi-4-mini
processes the combined context and outputs structured predictions. Bayesian
fusion promotes reliable persistence into memory. On the UNSW-NB15 dataset,
DM-RAG achieves 53.64% accuracy and 98.70% recall, surpassing fine-tuned and
RAG baselines in recall. The architecture is lightweight, interpretable, and
scalable, enabling real-time threat monitoring without extra corpora or heavy
tuning.

## Full Text


<!-- PDF content starts -->

Memory-Augmented Log Analysis with
Phi-4-mini: Enhancing Threat Detection in
Structured Security Logs
Anbi Guo, Mahfuza Farooque
School of Electrical Engineering and Computer Science,
Pennsylvania State University, University Park, PA, USA
{aqg6077, mff5187}@psu.edu
Abstract—Structured security logs are critical for detecting
advanced persistent threats (APTs). Large language models
(LLMs) struggle in this domain due to limited context and do-
main mismatch. We propose DM-RAG, a dual-memory retrieval-
augmented generation framework for structured log analysis. It
integrates a short-term memory buffer for recent summaries and
a long-term FAISS-indexed memory for historical patterns. An
instruction-tuned Phi-4-mini processes the combined context and
outputs structured predictions. Bayesian fusion promotes reliable
persistence into memory. On the UNSW-NB15 dataset, DM-
RAG achieves 53.64% accuracy and 98.70% recall, surpassing
fine-tuned and RAG baselines in recall. The architecture is
lightweight, interpretable, and scalable, enabling real-time threat
monitoring without extra corpora or heavy tuning.
Index Terms—Log anomaly detection, Security log analysis,
large language models, memory-augmented RAG, instruction
tuning, advanced persistent threats
I. INTRODUCTION
Structured security logs, such as those from network traf-
fic monitors, intrusion detection systems, or system audits,
form the basis for detecting and responding to cyber threats.
Despite the success of large language models (LLMs) in
natural language tasks, they often struggle to generalize to
specific domains such as Biomedicine [1], [2], Finance [3],
Medicine [4], and Security [5] without adaptation, due to
differences in syntax, semantics, and behavioral patterns.
This stems from the lack of interpretability in LLM out-
puts [6] and the gap between their pre-training corpora and
security logs [7]. Complex persistent threats, such as advanced
persistent threats (APTs), often manifest as multistage attacks.
Their key log events are dispersed across long periods [8], [9],
[27], challenging log analysis methods based on fixed context
windows. LLMs also face token context limits—information
beyond the window may be discarded or compressed. Reason-
ing over log entries across multiple windows and days thus
becomes difficult [10].
To address this, we introduce a “memory” mechanism into
log analysis, based on instruction tuning and adaptive real-time
RAG technology.
This paper uses Phi-4-mini [14], a lightweight, open-source,
decoder-only LLM as the foundation. We simulate working
memory and long-term memory from human cognition [11],
used respectively to retain recent behavior and persistent attackpatterns. Two adaptive memories are automatically maintained
to assist in analysis and reasoning on structured logs. To
enhance cross-temporal reasoning, instruction tuning is applied
to train the model for memory usage. A rolling summariza-
tion mechanism captures and maintains recent behaviors. In
addition, a long-term retrievable memory is built with RAG
to support integration of historical patterns. This architecture
strengthens the model’s ability to combine local context with
global historical signals.
Specifically, DM-RAG maintains two complementary mem-
ory streams. The first is a rolling summary to retain recent
behaviors, while the second is a real-time updated RAG that
stores high-confidence behaviors.
We evaluated our system on the UNSW-NB15 dataset [15],
split into training and test sets. We compared it with three
configurations: the original Phi-4-mini, a Phi-4-mini fine-
tuned with Low-Rank Adaptation (LoRA) [16], and a RAG-
enhanced Phi-4-mini incorporating external threat knowledge.
Experimental results show that DM-RAG achieves the highest
recall (98.70%) and F1 score (69.59%), validating its ef-
fectiveness for high-coverage, multistage threat detection in
structured logs.
II. RELATEDWORK
A. Language Modeling
Deep learning has become increasingly popular for log
anomaly detection recently. One of the earliest models is
DeepLog[17], which treats system logs as natural language
sequences and uses a Long Short-Term Memory (LSTM)
network to model log key sequences for anomaly detection.
With advances in natural language processing (NLP), re-
searchers explored transformer-based models for log analysis.
LogBERT[18] was the first to use a bidirectional transformer
encoder to learn contextual relationships between log keys.
It performsMasked Log Key Predictionto identify expected
log entries and appliesVolume of Hypersphere Minimization
to analyze log representations, where normal logs cluster near
the hypersphere center while anomalies deviate.
Building on this,LogGPT[19] introduced a reinforcement
learning (RL)-augmented transformer with a reward mecha-
nism for correct predictions.LogLLaMA[20] further refinedarXiv:2510.00529v1  [cs.CR]  1 Oct 2025

this approach by rewarding predictions based on the Top-K
most probable candidates.
Beyond architectural innovations,SecEncoder[21] takes a
different approach. It forgoes pre-trained models and trains
entirely from scratch on raw logs, focusing on log-specific
semantics without pretraining cost.
B. Retrieval-Augmented Generation (RAG)
RAG combines language models with external retrieval to
expand context and grounding. While effective in knowledge-
intensive tasks (e.g., QA, summarization), it assumes a static
knowledge base and cannot adapt to evolving log events
without frequent re-indexing. Performance of RAG-seq im-
proves as K increases, with the best effect for tokens when
K=10 [25]. More recently,RagLog[22] applied RAG for log
anomaly detection through a QA pipeline, leveraging external
log contexts to generate and evaluate expected log entries.
This improves interpretability and gives the model broader
contextual knowledge.
C. Memory-Augmented Language Models
To overcome fixed context limits, recent work augments
LLMs with external memory. MemGPT [12] introduces an
OS-inspired hierarchical memory architecture, enabling LLMs
to manage virtual context via function calls. It separates in-
context memory from out-of-context storage, and lets the
model retrieve, store, and update relevant information au-
tonomously. This supports long-term reasoning in tasks such
as multi-session dialogue and document analysis, improving
over fixed-context baselines.
D. Core Ideas in Log Anomaly Detection
Despite diverse approaches, the core idea is consistent:
modeling contextual dependencies of log sequences with
transformers and detecting anomalies based on prediction
confidence. If an observed log entry is not in the model’s Top-
K predictions given its context, it is considered anomalous due
to its low probability under the learned log model.
III. METHOD
A. Overview
We introduce a Dual-Memory Retrieval-Augmented Gen-
eration (DM-RAG) system for sequential log analysis. Our
system wraps a compact decoder-only language model (Phi-
4-mini) with two interacting memory modules supporting
continual reasoning and anomaly classification. Each inference
window processes log entries to generate a summary and
prediction.
Two memory structures are maintained:
Short-Term Memory (STM):A rolling buffer of sizeK=
10that retains recent summaries with confidence scores.
Long-Term Memory (LTM):A persistent vector-based
store indexed via semantic embeddings, storing high-
confidence summaries for retrieval and reuse.
These memories are injected into subsequent prompts to
enable retrieval-augmented reasoning with both recent and
accumulated context.
Fig. 1. Incoming network logs are analyzed by an instruction-tuned LLM with
dual memory. The short-term memory stores recent summaries and scores,
while the long-term memory retrieves relevant high-confidence examples via
FAISS. These jointly inform the prompt to Phi-4-mini for threat reasoning
and classification. STM’s confidence score are periodically compressed with
Bayesian fusion, and high-confidence results are promoted to LTM for future
retrieval.
B. Initial Confidence Generation via Logistic Regression
Logistic regression is widely used in anomaly detection. It
estimates the probability of normal or anomalous from the
input vector [30].
To estimate an initial confidence score for each log entry,
we train a logistic regression model on a large labeled dataset.
The process is as follows.
First, features are extracted, including flow-level metrics like
byte and packet counts, flow duration, traffic rate, TTL, TCP
indicators, jitter, and application-layer statistics. All continu-
ous features are normalized to[0,1]using min-max scaling
for consistency across dimensions:
xnorm
i=xi−min(x i)
max(x i)−min(x i)(1)
Next, the labeled UNSW-NB15 dataset is prepared. Each
instance consists of a normalized feature vectorx(i)and a
binary labely(i)∈ {0,1}indicating normal or anomalous:
D=n
x(i), y(i)oN
i=1(2)
We then train the model to map feature vectors to anomaly
probabilities. The posterior probability of anomaly is:
P(y= 1|x) =1
1 + exp(−w⊤x−b)(3)
For a new inputx, the model outputs a probabilityp∈[0,1],
interpreted as the initial anomaly confidence score:
score(x) =P(y= 1|x)(4)

C. Memory Generation
When a new log entry is received, the system performs
memory generation.
First, the log is encoded into a vector using SentenceTrans-
former MiniLM-L6-v2 [24] and retrieves relevant summaries
from long-term memory (LTM) via FAISS [23]. These serve
as background for reasoning.
Next, the log entry is embedded into a structured prompt and
passed to a language model, which outputs a natural language
summary, a confidence score from 0 to 1, and an attack label
such as “Normal,” “DoS,” or “Reconnaissance.”
The output is stored in short-term memory (STM) as an
object with log, analysis, confidence, and timestamp. STM
is a sliding window queue of the most recentK= 10
entries, maintaining localized temporal context without manual
management.
If the confidence score exceeds a threshold (c i≥0.9),
the summary is promoted to LTM, encoded and stored in the
FAISS index with metadata.
During future analysis, both STM contents and the top-
10 retrieved LTM summaries are included in the reasoning
prompt, providing context. This design lets the system learn
from past observations, incorporate new insights, and make
more accurate decisions over time.
D. Memory Compression and Promotion
When STM reaches full capacity (K= 10), compression is
triggered. The model is prompted to merge related summaries,
discard redundancies, and output a merged narrative summary.
To compute a confidence score for each log entry, we adopt
a Bayesian fusion framework over its structured features [28],
[29]. Each feature indicates anomalous behavior. We model the
normalized values of selected features as independent samples
drawn from Beta distributions conditioned on the underlying
class.
Let{x i}n
i=1denote the set ofncontinuous features ex-
tracted from a single network flow instance. Eachx iis
normalized to[0,1]and modeled as a class-conditional Beta
sample:
xi∼(
Beta(α 1, β1),ify= 1(anomalous)
Beta(α 0, β0),ify= 0(normal)(5)
The likelihood of observing the feature set{x i}under each
class is computed by assuming conditional independence:
P({x i} |y) =nY
i=1Beta(x i;αy, βy)(6)
Applying Bayes’ theorem, we compute the posterior prob-
ability of the log being anomalous as:
P(y= 1| {x i}) =P({x i}|y=1)P(y=1)
P({x i}|y=1)P(y=1)+P({x i}|y=0)P(y=0)(7)
This posterior is the fused anomaly confidence score for
the log entry. The Beta distribution parameters(α y, βy)areestimated from the training set using the ground-truth binary
labely∈ {0,1}.
Each log entry includes features such as basic flow meta-
data, packet and byte counts, traffic rate and load, TTL and
packet loss indicators, timing and jitter, TCP characteristics,
payload statistics, connection counters, application-level indi-
cators, a boolean flag, and labels.
This posterior is the overall anomaly confidence score,
enabling principled decision-making while accounting for con-
tributions of heterogeneous flow features.
E. Persistent Memory and Retrieval
The Long-Term Memory (LTM) stores compressed sum-
maries. Each summary is encoded into a 384-dimensional
embedding using the all-MiniLM-L6-v2 model and indexed
via FAISS. During inference, the top-10 most relevant entries
are retrieved by cosine similarity and incorporated into the
prompt. This provides continuity and helps detect patterns.
F . Prompt Construction
Each prompt is built from four components:
Task Description:Defines the objective, such as attack
detection and categorization.
Long-Term Memory (LTM):Retrieved summaries from
previous analyses.
Short-Term Memory (STM):Behavior summaries from
recent logs.
Current Log Entry:The active log entry for analysis.
This composite prompt format enables instruction-following
and supports few-shot reasoning with memory-based exam-
ples.
G. Comparison to Prior Strategies
Unlike traditional log parsers or static detectors, DM-RAG
supports:
Continual learning through dynamic memory updates.
Interpretable natural language summaries of behavior.
Probabilistic uncertainty aggregation via Bayesian methods.
H. Relation to Prior Work
Our confidence fusion mechanism is inspired by the
Bayesian ensemble model proposed in [28], first used for un-
supervised anomaly detection. We adapt it to fuse confidence
scores from multiple LLM-generated summaries and apply it
to memory compression and promotion in our architecture.
I. Algorithm
The proposed method is described in Algorithm 1.
IV. EXPERIMENTALSETUP
A. Dataset
We use the UNSW-NB15 dataset, containing network traffic
logs labeled as normal or attack behaviors across nine cate-
gories. NetFlow logs are preprocessed into token sequences
representing key attributes.

Algorithm 1:Log Analysis with Memory-Augmented
RAG and Bayesian Fusion
1:Input:Structured logs{l 1, ..., l n}; EncoderE;
Instruction-tuned LLMG; FAISS indexM LTM (initially
empty); Short-term memory bufferM STM (sizek);
Promotion thresholdτ(0.9)
2:Output:Analysis results with summaries and anomaly
labels
3:Step 1: Confidence Model Preparation
4:Load labeled datasetD={(x(i), y(i))}N
i=1
5:foreach feature vectorx(i)inDdo
6:Normalize features to[0,1]using min-max scaling
7:end for
8:Train logistic regression modelMto estimate
P(y= 1|x)
9:Define scoring function score(x)←M(x)
10:Step 2: Online Log Analysis
11:foreach logl tin{l 1, ..., l n}do
12:v t← E(l t){Encode current log}
13:S LTM←FAISS.Retrieve(v t, top-10){Top-k retrieval}
14:Build promptP t←Merge(Task,S LTM ,M STM ,lt)
15:output← G(P t){LLM inference}
16:Parse(summary t, conf t, label t)←output
17:M STM← M STM∪ {(summary t, conf t)}
18:if|M STM|=kthen
19:Build compression promptP STM←
Compress(M STM )
20:merged← G(P STM)
21:conf fused←BayesianFusion({conf i}inM STM )
22:ifconf fused > τthen
23:v← E(merged)
24:M LTM← M LTM∪ {v} {Add to FAISS}
25:end if
26:ResetM STM← {(merged, conf fused )}
27:end if
28:Save analysis result:{l t, summary t, label t}
29:end for
B. Data Preprocessing and Splitting
The dataset is split into training (157,806 logs), validation
(17,535), and test (82,332).
Each log entry is described by diverse features, including
flow metadata (e.g., source/destination IP and ports, protocol),
traffic statistics (e.g., byte and packet counts, rate, duration,
throughput), temporal measures (e.g., inter-arrival time, jitter,
TTL, packet loss), TCP-level indicators (e.g., round-trip time,
SYN/ACK delay), and application-level/service flags. All con-
tinuous features are normalized to[0,1].
Each sample has a binary label (attack or normal) and, if
anomalous, one of nine attack categories:Reconnaissance,
Backdoor,DoS,Exploits,Analysis,Fuzzers,Shellcode,
Worms, orGeneric. These labels support both binary and
multi-class classification.C. Model Configuration
Our system uses a dual-memory prompting architecture
based on Phi-4-mini. At each inference step, the model
receives a sliding window of prior log entries along with
recent and persistent summaries. It outputs a natural language
reasoning chain, a confidence estimate, and a classification
decision.
Two memory buffers are maintained: Short-Term Memory
(STM), a queue storing recent outputs; and Long-Term Mem-
ory (LTM), a persistent bank of high-confidence summaries
indexed with sentence embeddings and retrieved via FAISS.
When STM reaches capacity, its contents are summarized
and compressed using a fixed-sizedeque. High-confidence
entries are promoted to LTM, while the rest are merged
into one summary. The confidence score of the compressed
summary is computed via Bayesian fusion [28]. Both STM
and LTM are incorporated into subsequent prompts to ensure
temporal continuity and semantic grounding.
D. Prompt Design
To guide the language model in structured log analysis, we
design a composite prompt with four parts: the current log,
neighboring summaries, relevant long-term memory, and a task
instruction block. Each part is presented sequentially, forming
a structured input.
Fig. 2. Prompt template sent to the language model, composed of four parts:
current log, STM summaries, LTM retrievals, and task requirements.
Part 1 presents the current log in raw structured form. The
JSON-formatted input record contains all observed features of
a single network flow.
Part 2 includes summaries of neighboring logs retrieved
from the short-term memory (STM) buffer. These capture
recent context that may indicate correlated or evolving anoma-
lous behavior. Each STM entry has a natural language sum-
mary and a confidence score from earlier model output.
Part 3 adds high-confidence summaries retrieved from long-
term memory (LTM) using FAISS-based similarity search.
These entries represent verified anomalous cases resembling
the current log, allowing the model to generalize from past
examples.
Part 4 defines the task objective and enforces strict output
formatting.
The instruction asks the model to determine whether the cur-
rent log indicates an attack. If so, the model must identify the
attack category from a predefined set: Reconnaissance, Back-
door, DoS, Exploits, Analysis, Fuzzers, Shellcode, Worms, and
Generic. If no attack is detected, it should returnNormal.
The label is binary, where 1 indicates attack and 0 normal.

Fig. 3. Instruction block provided to the language model to define the anomaly
detection and classification task.
Additionally, the model must provide a brief description of
log behavior in fewer than 50 words.
The final output must be a valid JSON object with
exactly four keys:id,attack_cat,label, and
short_summary. No additional fields or commentary
are permitted.
Fig. 4. Instruction enforcing strict JSON output format from the LLM.
This constraint is enforced by a final instruction string,
ensuring output consistency across generations. The prompt
design provides rich multi-scale context while enforcing struc-
tured responses, enabling accurate and interpretable anomaly
detection.
E. Comparison Methods
We compare four methods for anomaly detection with
language models:
In thezero-shotsetting, the Phi-4-mini-instruct model is
directly applied without task-specific training. It receives the
log and a static prompt to assign labels, evaluating its pre-
trained reasoning ability.
WithLoRA fine-tuning, Phi-4-mini is adapted on labeled
logs. LoRA updates a small subset of parameters by inserting
trainable low-rank matrices into attention layers, reducing
training cost and memory while preserving general capabil-
ities. The fine-tuned model is then tested on unseen logs
without external retrieval or prompt engineering.
ForRAG, we adopt RAG-Sequence setup, retrieving a se-
quence of top-K passages per step to condition generation [25]
with MITRE ATT&CK v17.1 (enterprise-attack). Following Li
et al.[26], we emphasize semantic relevance over knowledge
base size. MiniLM-L6-v2 encodes both logs and ATT&CK
definitions into embeddings, which are searched with FAISS
to retrieve relevant entries. The retrieved snippets are injected
into prompts and passed to Phi-4 for structured output.
Finally, ourdual-memory promptingintegrates short- and
long-term memories into prompts, enabling reasoning over
both recent and recurring behaviors without relying on external
corpora.F . Evaluation Metrics
To evaluate our framework, we adopt standard classification
metrics used in intrusion detection:Precision,Recall,F1
Score, andAccuracy. These metrics provide a comprehensive
view of performance, especially under class imbalance and
multi-stage attacks.
We compute the metrics usingscikit-learn, after
aligning predictions and ground truth by a unique identifier
(id). This alignment ensures proper label correspondence.
a) Accuracy:reflects the overall proportion of correct
predictions but can be misleading in imbalanced datasets.
b) Precision:measures the proportion of correctly pre-
dicted attacks among all predicted attacks, showing the sys-
tem’s ability to reduce false positives.
c) Recall:measures how many actual attacks are de-
tected among all real attack instances, critical for stealthy or
staged threats.
d) F1 Score:combines precision and recall using their
harmonic mean, balancing false positives and false negatives.
G. Test Results
We evaluate DM-RAG and several baselines on the UNSW-
NB15 test set. Table I reports accuracy, precision, recall, and
F1 score.
Our method achieves the highest recall (98.70%) among
models, showing strong ability to detect true positives. This
indicates that DM-RAG, with its short-term memory and
Bayesian fusion strategy, is more sensitive to actual attacks.
In precision (53.74%), DM-RAG outperforms LoRA fine-
tuned (44.92%). However, the zero-shot Phi-4-mini has the
highest precision (98.91%), which is likely because the model
is extremely conservative: it only predicts positive in very rare
cases where it is highly confident. This yields very few false
positives, but at the cost of an extremely low recall (0.20%).
DM-RAG’s overall accuracy (53.64%) is slightly lower than
the MITRE-style RAG baseline (57.24%), but its F1 score
(69.59%) is higher than all other methods, indicating balanced
detection.
TABLE I
PERFORMANCE COMPARISON ON UNSW-NB15 TEST SET
Model Accuracy Precision Recall F1 Score
Phi-4-mini (Zero-shot) 45.05% 98.91% 0.20% 0.40%
LoRA Fine-tuned 37.97% 44.92% 46.89% 45.89%
Phi-4 + RAG (MITRE) 57.24% 68.38% 41.57% 51.70%
DM-RAG (Ours) 53.64% 53.74% 98.70% 69.59%
In summary, DM-RAG improves detection coverage (recall)
while maintaining reasonable precision, suitable when missing
attacks is costlier than false alarms.
V. LIMITATIONS ANDFUTUREWORK
While DM-RAG enhances reasoning and detection across
long-range log sequences, several limitations remain.
First, the heuristic promotion rule (confidence≥0.9) may
be sub-optimal, as mis-promoted summaries add noise to long-
term memory (LTM), degrading inference.

Second, the method does not use explicit temporal con-
straints or structured log schemas, reducing precision on
domain-specific data.
Third, semantic-embedding retrieval assumes latent sim-
ilarity reflects causal relevance, which may fail in noisy,
imbalanced, or adversarial contexts.
Fourth, the framework imposes no hard cap on LTM size.
Each promoted summary is appended, so the FAISS index can
grow indefinitely, increasing retrieval latency and GPU use.
During analysis this roughly doubles runtime due to index
rebuilds and queries.
Scalability is also constrained by token limits, even with
the compact Phi-4-mini backbone. This may hurt performance
under high-frequency events or deep behavior hierarchies. The
model assumes a stationary distribution and is untested under
distribution shift or continual deployment.
Future work includes schema-aware reasoning and log-
template clustering for better interpretability, continual learn-
ing to prevent forgetting, and hybrid designs that fuse MITRE-
style RAG with DM-RAG’s memory to improve multi-stage
attack detection.
VI. CONCLUSION
We presented DM-RAG, a dual-memory framework for
structured log anomaly detection, enhancing temporal reason-
ing of compact instruction-tuned LLMs. Inspired by cognitive
memory systems, DM-RAG integrates a short-term memory
buffer for recent context with a FAISS-indexed long-term
memory to retrieve historical patterns.
Through memory summarization, Bayesian confidence fu-
sion, and structured prompting, DM-RAG achieves high-recall
detection of multi-stage threats while remaining lightweight
and deployable.
On UNSW-NB15, DM-RAG attains the highest recall
(98.70%), with F1 of 69.59% and precision of 53.74%,
outperforming LoRA-tuned and MITRE-style RAG baselines.
The design requires no external corpora or large-scale LLMs,
making it suitable for real-time, resource-constrained environ-
ments.
Beyond security, the dual-memory design generalizes to
long-horizon reasoning in structured temporal data such as
medical audit trails, industrial telemetry, and financial transac-
tions. Future directions include adaptive memory management,
schema-aware prompting, and continual learning for robust
generalization under evolving conditions.
REFERENCES
[1] Yu Gu, Robert Tinn, Hao Cheng, Michael Lucas, Naoto Usuyama, Xi-
aodong Liu, Tristan Naumann, Jianfeng Gao, and Hoifung Poon. Domain-
specific language model pretraining for biomedical natural language
processing.ACM Transactions on Computing for Healthcare, 3(1):1–23,
October 2021.
[2] Renqian Luo, Liai Sun, Yingce Xia, Tao Qin, Sheng Zhang, Hoifung
Poon, and Tie-Yan Liu. BioGPT: Generative pre-trained transformer for
biomedical text generation and mining.Briefings in Bioinformatics, 23(6),
September 2022.
[3] S. Wu, et al., “Bloomberggpt: A large language model for finance,”arXiv
preprint arXiv:2303.17564, 2023.[4] K. Singhal, et al., “Toward expert-level medical question answering with
large language models,”Nature Medicine, vol. 31, no. 3, pp. 943–950,
2025.
[5] Google Cloud. Google Cloud, ”RSA: Google Cloud Security
AI Workbench and Generative AI,” 2024. [Online]. Avail-
able: https://cloud.google.com/blog/products/identity-security/
rsa-google-cloud-security-ai-workbench-generative-ai. Accessed:
Jul. 15, 2024.
[6] Gilpin, Leilani H., et al. “Explaining explanations: An overview of inter-
pretability of machine learning.”2018 IEEE 5th International Conference
on Data Science and Advanced Analytics (DSAA). IEEE, 2018, pp. 80–89.
[7] Arazzi, Marco, et al. “NLP-based techniques for cyber threat intelligence.”
Computer Science Review, vol. 58, 2025, p. 100765.
[8] Hutchins, Eric M., Michael J. Cloppert, and Rohan M. Amin.
“Intelligence-Driven Computer Network Defense Informed by Analysis
of Adversary Campaigns and Intrusion Kill Chains.”
Leading Issues in Information Warfare & Security Research, vol. 1, 2011,
pp. 80–106.
[9] Chen, Ping, Lieven Desmet, and Christophe Huygens. “A Study on
Advanced Persistent Threats.”Communications and Multimedia Security
(CMS 2014), Springer, 2014, pp. 63–72. doi:10.1007/978-3-662-44885-
45.
[10] Liu, Nelson F., et al. “Lost in the Middle: How Language Models Use
Long Contexts.”arXiv preprintarXiv:2307.03172, 2023.
[11] N. Cowan, ”What are the differences between long-term, short-term,
and working memory?”Prog. Brain Res., vol. 169, pp. 323–338, 2008.
doi:10.1016/S0079-6123(07)00020-9.
[12] Packer, Charles, et al. “MemGPT: Towards LLMs as Operating Sys-
tems.” 2023.arXiv preprintarXiv:2312.06635.
[13] T. Brownet al., “Language models are few-shot learners,” inAdv. Neural
Inf. Process. Syst. (NeurIPS), vol. 33, pp. 1877–1901, 2020.
[14] A. Aboueleninet al., “Phi-4-mini technical report: Compact yet powerful
multimodal language models via mixture-of-LoRAs,”arXiv preprint
arXiv:2503.01743, 2025.
[15] N. Moustafa and J. Slay, “UNSW-NB15: A comprehensive data set for
network intrusion detection systems (UNSW-NB15 network data set),” in
Proc. MilCIS, 2015, pp. 1–6.
[16] E. J. Hu et al., ”LoRA: Low-rank adaptation of large language models,”
arXiv, arXiv:2106.09685, 2021.
[17] M. Du, F. Li, G. Zheng, and V . Srikumar, “DeepLog: Anomaly detection
and diagnosis from system logs through deep learning,” inProc. 24th
ACM Conf. Comput. Commun. Secur., 2017.
[18] H. Guo, S. Yuan, and X. Wu, “LogBERT: Log anomaly detection via
BERT,” inProc. Int. Joint Conf. Neural Netw. (IJCNN), 2021, pp. 1–8.
[19] X. Han, S. Yuan, and M. Trabelsi, “LogGPT: Log anomaly detection
via GPT,” inProc. IEEE Int. Conf. on Big Data (BigData), 2023.
[20] Z. Yang and I. G. Harris, “LogLLaMA: Transformer-based log anomaly
detection with LLaMA,”arXiv preprint arXiv:2503.14849, 2025.
[21] M. F. Bulut, Y . Liu, N. Ahmad, M. Turner, S. A. Ouahmane, C. Andrews,
and L. Greenwald, “SecEncoder: Logs are All You Need in Security,”
arXiv preprint arXiv:2411.07528, 2024.
[22] J. Pan, S. L. Wong, and Y . Yuan, “RagLog: Log anomaly detection using
retrieval augmented generation,” inProc. IEEE World Forum Public Saf.
Technol. (WFPST), 2024, pp. 1–6.
[23] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with GPUs,”IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–547, 2019.
[24] W. Wanget al., “Minilm: Deep self-attention distillation for task-
agnostic compression of pre-trained transformers,” inAdv. Neural Inf.
Process. Syst. (NeurIPS), vol. 33, pp. 5776–5788, 2020.
[25] P. Lewiset al., “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inAdv. Neural Inf. Process. Syst. (NeurIPS), vol. 33,
pp. 9459–9474, 2020.
[26] S. Liet al., “Enhancing retrieval-augmented generation: a study of best
practices,”arXiv preprint arXiv:2501.07391, 2025.
[27] MITRE, “MITRE ATT&CK Framework,” 2024. https://attack.mitre.org
[28] E. Yu and R. Parekh,A Bayesian Ensemble for Unsupervised Anomaly
Detection, arXiv preprint arXiv:1610.07677, 2016.
[29] Dai, Hongsheng, Murray Pollock, and Gareth O. Roberts. Bayesian
Fusion: Scalable unification of distributed statistical analyses.Journal
of the Royal Statistical Society: Series B (Statistical Methodology),
85(1):84–107, 2023.
[30] Wu, Xingfang, Heng Li, and Foutse Khomh. ”On the effectiveness of
log representation for log-based anomaly detection.”Empirical Software
Engineering, vol. 28, no. 6, 2023, article 137.