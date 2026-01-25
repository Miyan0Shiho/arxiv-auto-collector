# Rethinking On-Device LLM Reasoning: Why Analogical Mapping Outperforms Abstract Thinking for IoT DDoS Detection

**Authors**: William Pan, Guiran Liu, Binrong Zhu, Qun Wang, Yingzhou Lu, Beiyu Lin, Rose Qingyang Hu

**Published**: 2026-01-20 15:18:56

**PDF URL**: [https://arxiv.org/pdf/2601.14343v1](https://arxiv.org/pdf/2601.14343v1)

## Abstract
The rapid expansion of IoT deployments has intensified cybersecurity threats, notably Distributed Denial of Service (DDoS) attacks, characterized by increasingly sophisticated patterns. Leveraging Generative AI through On-Device Large Language Models (ODLLMs) provides a viable solution for real-time threat detection at the network edge, though limited computational resources present challenges for smaller ODLLMs. This paper introduces a novel detection framework that integrates Chain-of-Thought (CoT) reasoning with Retrieval-Augmented Generation (RAG), tailored specifically for IoT edge environments. We systematically evaluate compact ODLLMs, including LLaMA 3.2 (1B, 3B) and Gemma 3 (1B, 4B), using structured prompting and exemplar-driven reasoning strategies. Experimental results demonstrate substantial performance improvements with few-shot prompting, achieving macro-average F1 scores as high as 0.85. Our findings highlight the significant advantages of incorporating exemplar-based reasoning, underscoring that CoT and RAG approaches markedly enhance small ODLLMs' capabilities in accurately classifying complex network attacks under stringent resource constraints.

## Full Text


<!-- PDF content starts -->

Rethinking On-Device LLM Reasoning: Why
Analogical Mapping Outperforms Abstract Thinking
for IoT DDoS Detection
William Pan§, Guiran Liu§, Binrong Zhu§, Qun Wang§, Yingzhou Lu†, Beiyu Lin‡, Rose Qingyang Hu£
§Department of Computer Science, San Francisco State University, San Francisco, CA, 94132
†School of Medicine, Stanford University, USA
‡Department of Computer Science, University of Texas Dallas, Dallas, USA
£Bradley Department of Electrical and Computer Engineering,
Virginia Polytechnic Institute and State University, Blacksburg, V A 24061
Emails: William910122@gmail.com, claudqunwang@ieee.org, gliu@sfsu.edu, bzhu2@sfsu.edu,
lyz66@stanford.edu, beiyu.lin@utdallas.edu, rosehu@vt.edu
Abstract—The rapid expansion of IoT deployments has in-
tensified cybersecurity threats, notably Distributed Denial of
Service (DDoS) attacks, characterized by increasingly sophis-
ticated patterns. Leveraging Generative AI through On-Device
Large Language Models (ODLLMs) provides a viable solution
for real-time threat detection at the network edge, though
limited computational resources present challenges for smaller
ODLLMs. This paper introduces a novel detection framework
that integrates Chain-of-Thought (CoT) reasoning with Retrieval-
Augmented Generation (RAG), tailored specifically for IoT edge
environments. We systematically evaluate compact ODLLMs,
including LLaMA 3.2 (1B, 3B) and Gemma 3 (1B, 4B), using
structured prompting and exemplar-driven reasoning strategies.
Experimental results demonstrate substantial performance im-
provements with few-shot prompting, achieving macro-average
F1 scores as high as 0.85. Our findings highlight the significant
advantages of incorporating exemplar-based reasoning, under-
scoring that CoT and RAG approaches markedly enhance small
ODLLMs’ capabilities in accurately classifying complex network
attacks under stringent resource constraints.
Index Terms—IoT Security, On-Device Large Language Mod-
els (ODLLMs), Chain-of-Thought (CoT), Retrieval-Augmented
Generation (RAG), Few-Shot Learning
I. INTRODUCTION
The proliferation of Internet of Things (IoT) sensors in res-
idential and industrial environments has significantly acceler-
ated digital transformation by timely and effective processing
of precise data for rapid decision-making [1]. However, the
extensive deployment of IoT devices also introduced network
risks, especially with the increasing frequency and complexity
of Distributed Denial of Service (DDoS) attacks and the
introduction of more sophisticated and adaptive attack combi-
nations [2]. These evolving threats pose substantial challenges
for conventional detection methodologies, underscoring the
necessity for advanced detection mechanisms.
The recent emergence of Generative Artificial Intelligence
(GenAI), particularly Large Language Models (LLMs), pro-
vides promising capabilities for enhancing cybersecurity by
identifying complex and dynamic attack patterns in real-time [3] [4]. However, employing cloud-based LLM solutions
introduces critical privacy concerns and latency issues, par-
ticularly pertinent in real-time threat detection scenarios and
sensitive data environments [5]. On the other hand, deploying
intelligence at the edge with On-Device Large Language
Models (ODLLMs) has been actively investigated [6], [7]. By
enabling effective decision-making directly on edge devices,
ODLLM-based solutions can alleviate privacy risks and la-
tency constraints [8] [9]. Nonetheless, the constraints on token
input length and the limited capabilities of intrinsic model un-
derstanding often result in suboptimal detection performance
without adequate background knowledge and tailored support
mechanisms.
Our previous work utilized a structured knowledge base
to increase ODLLM-enabled detectors’ performance, but the
smaller model is still struggling when handling multiple
attacks [10]. Recent efforts in enhancing LLM reasoning
capabilities have emphasized either structural prompting or
compute-adaptive inference. Weng [11] frames LLM reasoning
as a balance between fast heuristic-driven responses and slow
deliberative reasoning. Techniques such as Chain-of-Thought
(CoT) prompting and retrieval-augmented generation (RAG)
are presented as means to improving reliability on complex
tasks [12] [13]. Complementary to this cognitive framing,
Snell et al. study the trade-off between model size and test-
time compute, showing that iterative revision and verifier-
guided search can outperform naively scaling parameters [14].
However, how to integrate those methods to improve ODLLM
detection performance still needs to be solved.
To address these challenges, this paper proposes an innova-
tive detection framework integrating RAG and CoT method-
ologies to enhance ODLLM capabilities specifically for IoT
network security. Our contributions are as follows: (1) We
perform an in-depth analysis of DDoS attack characteristics
and their implications for ODLLM-based detection. (2) We
introduce a novel CoT-guided inference mechanism tailored
for resource-constrained edge models to improve reasoningarXiv:2601.14343v1  [cs.CR]  20 Jan 2026

and accuracy. (3) We compare and design a compact RAG
algorithm that is optimized for small-model constraints to
facilitate effective integration of structured knowledge bases.
(4) We perform comprehensive experimental evaluations for
different ODLLMs and validate the effectiveness and ro-
bustness of our proposed framework. It demonstrates that
our enhanced ODLLM system achieves comparable detection
accuracy to larger cloud-based models while maintaining the
operational advantages of edge-based deployment.
The subsequent sections are organized as follows. The sys-
tem model and problem formulation are presented in Section
II. The proposed CoT-assisted RAG design is developed in
Section III. The simulation results are presented in Section
IV . Finally, Section V provides the concluding remarks for
this paper.
II. SYSTEMMODEL ANDPROBLEMFORMULATION
A. System Model
As shown in Fig. 1, our cybersecurity attack detection
framework consists of two stages. In Stage 1, we utilize a
larger-sized LLM as a teacher model to extract key insights
from historical datasets of collected attacks offline, thereby
generating step-by-step reasoning outputs based on CoT
prompting and constructing a domain-specific knowledge base
(KB). This will lay the foundation for small-sized ODLLM to
perform its detection and overcome its input limitations and
constraint reasoning capabilities. Unlike standard distillation,
this process does not rely on soft labels. Instead, the teacher
model provides detailed CoT-based reports for each sample,
building new reasoning steps toward a well-formed conclusion.
These serve as demonstrations for the small-sized student
ODLLM model. ODLLM will learn to follow the reasoning
process and independently arrive at similar conclusions.
Based on Stage 1, Stage 2 will perform inference with
RAG-assisted ODLLM. Enlightened by paper [14], the raw
network traffic will first be transformed into feature embed-
dings generated by a pre-trained XGBoost classifier. We then
compute the Euclidean distance between samples and sort
the results in ascending order. For few-shot prompting, we
experimented with providing the ODLLM model with one
to three example samples. Notably, the objective is to utilize
ODLLM’s inference time (test time compute) capabilities for
slow, deliberate, and logically coherent step-by-step reasoning.
By performing the RAG search with XGBoost output, the
network traffic will be combined with retrieved examples
and the CoT prompt, and then fed into ODLLM for attack
detection.
With important feature ranking from our previous research
[10], incoming packets are aggregated into unidirectional
flows. Each flow is encoded as a nine-dimensional vector
corresponding to 9 top-ranking features:
x=
proto, r,IAT, s, α PSH, αACK, αSYN, αRST, αFIN
∈R9,
(1)
whereprotois the IP-layer protocol number,ris the packet-
rate (pps),IATthe mean inter-arrival time (ms),sthe av-erage payload length (bytes) and the fiveα i∈ {0,1}, i∈
{PSH,ACK,SYN,RST,FIN}are TCP flags.
The feature vector is normalized and passed to an XGBoost
classifierf ψ:R9→∆CwithCattack families labels. It
produces a softmax vector
p=f ψ(x),p∈∆C,(2)
which acts both as a coarse prediction and as a compact
semantic signature for retrieval.
Givenp, the retriever consults the KBM=
{(pi, xi, yi)}N
i=1and returns the indices of thekmost similar
prototypes under Euclidean distance:
Rk(p) = arg top-k
j≤Np−p j
2,(3)
wherex iis the archived flow andy iis the ground-truth label
of that flow.
The live flow descriptionD(§), the retrieved exemplar
block, and the fixed CoT template are concatenated into a
prompt presented to ODLLMg θ. With decoding temperature
T=0the model returns a deterministic labelˆyand a natural-
language rationaleπ:
(ˆy, π) =g θ 
Pη 
Rk(p),D(x)
,(4)
whereP ηdenotes the template-driven prompt builder.
B. Problem Formulation
Let the streaming data beD={(x t, yt)}t≥1withx t∈R9
and ground-truthy t∈ Y={1, . . . , C}. Define the composite
detector
hΦ(x) =g θ 
Pη 
Rk 
fψ(x)
,D(x)
,(5)
with the parameter setΦ ={θ, ψ, η,M}.
The design objective is to maximize the long-run detection
accuracy
Φ⋆= arg max
ΦE(x,y)∼D
1{h Φ(x) =y}
,(6)
By settingk= 0, the equation (5) reduces to a pure CoT
baseline without retrieval. In the next section, we will solve
the above problem via CoT design and RAG enhancement.
III. ODLLM DETECTORENHANCEMENT WITHCOT
REASONING
A. Chain-of-Thought Prompting
Chain-of-thought (CoT) prompting enhances the reasoning
capabilities of large language models (LLMs) by encouraging
them to generate intermediate reasoning steps before produc-
ing a final answer. Rather than jumping directly to conclusions,
the model is guided to break down the task into smaller,
logically connected steps. This structured decomposition mir-
rors the way cybersecurity professionals approach complex
incident analysis [12]. For example, when investigating a
DDoS attack, analysts typically examine traffic flows, rate
limits, source IP distributions, and attack durations step-by-
step to isolate the root cause and determine the appropriate
response. Similarly, CoT prompting enables LLMs to build a

Fig. 1: Proposed System Pipeline.
line of reasoning that mimics this analytical process. In this
paper, we implement a rule-based prompt structure that elicits
domain-specific CoT reasoning for cybersecurity reporting.
We apply the “Let’s think step by step” prefix strategy as
demonstrated by [15].
B. Chain-of-Thought (CoT) Baseline
Our starting point is a rule-guided CoT prompt that allows
ODLLM to emulate the deterministic logic traditionally en-
coded in signature-based intrusion systems. The prompt is
organized into three consecutive blocks:
1)Data description:A natural-language renderingD(x)
of the nine-dimensional feature vector in (1).
2)Instruction:a one-shot classification directive that (i)
constrains the output domain to the five DDoS flood
families and (ii) forces the model to emit the final an-
swer exactly once, in the pattern‘‘The answer is
<LABEL>’’.
3)Knowledge base (KB):A three-step reasoning scaffold
TCoTdistilled from traffic-engineering heuristics:
•Packet-size & rate gate.Reject benign flows whose
mean packet length is>60B or whose packet rate
is<1pps (empirically determined thresholds).
•Protocol branch.Distinguish ICMP and UDP floods
directly via theprotofield.
•TCP flag analysis.If the protocol is TCP, map
the tuple(α PSH, αACK, αRST, αFIN)to PSH/ACK,
RST/FIN or fallback to TCP flood.
The combined prompt is therefore Prompt(x) =
D(x)∥Instruction∥T CoT, where “∥” denotes string
concatenation.
IV. RAGENHANCEDODLLM DETECTIONDESIGN
A. Backbone of RAG
The effectiveness of Retrieval Augmented Generation
(RAG) depends strongly on the design of the retriever al-
gorithm. In our work, we evaluate multiple retriever models,
including BERT-based BGE (BAAI General Embedding), aMulti-Layer Perceptron (MLP), and XGBoost, to determine
which best identifies relevant documents for each query.
BGE Base.Our first retriever model is thebge-base-en-
v1.5, a fine-tuned BERT-base model from Hugging Face, to
generate 768-dimensional sentence embeddings. This model
is tailored for semantic similarity, retrieval, and clustering.
However, after projecting these embeddings using t-SNE into
2D space (Figure 2a), we observe that BGE fails to clearly
separate subcategories of TCP flood attacks (e.g., PSH/ACK
and RST/FIN). This indicates that our network traffic data is
not linearly separable in this embedding space, motivating the
exploration of alternative approaches.
MLP-based Embedding.To investigate whether a task-
specific embedding model can improve separability, we design
and train a Multi-Layer Perceptron (MLP) classifier. The MLP
also takes network traffic data from our knowledge database
and embeds it into a 16-dimensional latent space using a series
of linear layers with ReLU activations and early stopping
based on validation loss to avoid overfitting. The classifier is
trained using cross-entropy loss with label smoothing on strat-
ified splits of the data to ensure balanced class representation.
After training, we extract the 16-dimensional embeddings and
project them using t-SNE into 2D space for visualization. As
shown in Figure 2b, we observe a better separation between
different categories of attacks, unlike the general-purpose
BGE model. However, some regions still exhibit overlapping
clusters or include irrelevant types, suggesting room for further
improvement.
XGBoostRecognizing the limitations of purely embedding-
based approaches, we also investigate the use of XGBoost,
a gradient boosting framework well-suited for tabular data
[16]. Unlike embedding models, XGBoost directly builds an
ensemble of decision trees that iteratively correct previous
errors, which better capture complex patterns in network traffic
features. While XGBoost is primarily a classifier, we treat
the standardized 9 numerical input features as fixed embed-
dings and apply t-SNE to project them into a 2D space. As
shown in Figure 2c, the resulting layout reveals that the non-
linear decision boundaries can already capture the differences
between attack categories. For the training phase, we use
the extracted numerical features from network traffic data.
Then the model is configured for multi-class classification
with the objective functionmulti:softmax, optimized to
distinguish among the attack categories. Key hyper-parameters
include a maximum tree depth of 6, a learning rate of 0.1,
and 100 boosting iterations (estimators). This setup balances
model complexity and overfitting risk while enabling efficient
training on moderate-sized datasets.
B. Probability-Guided Retrieval–Augmented Generation
(Prob-RAG)
XGBoost constructs additive ensemble models consisting
of decision trees, optimized in a forward stage-wise manner.
The training procedure minimizes the following regularized

objective function:
L(t)=nX
i=1l 
yi,ˆy(t−1)
i +ft(xi)
+ Ω(f t),(7)
wherel(·)is a differentiable convex loss function, such as
logistic or multinomial log-loss,y iis the ground-truth label,
andˆy(t−1)
i is the prediction from the previous iteration.
The regularization termΩ(f)controls the complexity of the
tree model to prevent overfitting, defined as:
Ω(f) =γT+1
2λ∥w∥2,(8)
whereTdenotes the number of leaves in the tree,wrepre-
sents the vector of leaf weights, andγ, λare regularization
hyperparameters. Specifically,γpenalizes tree complexity
by restricting the number of leaves, whileλprovidesL 2-
regularization on leaf weights.
We represent each network flow as a nine-dimensional
numeric feature vector capturing protocol type, packet rate,
inter-arrival time, packet size, and TCP flags. Each vector is
standardised (zero mean, unit variance) and used directly as the
embedding signature for retrieval. Additionally, an XGBoost
modelf ψtrained to classify DDoS types outputs a class-
probability vectorp=f ψ(x)∈∆5, providing both a coarse-
grained prediction and an alternative semantic representation
for retrieval.
The exemplar databaseM Pstores the standardised feature
vectors of previously observed and labelled flows:
MP={(x i,D(x i), yi)}N
i=1,(9)
wherex iis the normalised numeric feature vector,D(x i)is
the textual description, andy iis the corresponding label.
At inference time, we retrieve the top-kexemplars closest
to the current flow’s normalised vectorxusing Euclidean
distance:
RP
k(x) = arg top-k
j∥x−x j∥2.(10)
Retrieved exemplars are concatenated with the live flow
description and a fixed Chain-of-Thought (CoT) instruction
template into a compact prompt:
PromptProb=⟨ERP
k(x)⟩ ∥ T CoT∥ D(x).(11)
This prompt is submitted to a local small language model to
yield a final classification.
V. EXPERIMENTPERFORMANCEEVALUATION
We use Ollama to retrieve ODLLMs with our constructed
KB on Desktop with NVIDIA RTX 4090 and Intel I9-
13900KF [17]. We consider the latest small-sized models as
follows:Llama 3.2 3B & 1Bis a compact variant in the
Llama 3 series with 3 billion parameters and 1 billion pa-
rameters, optimized for multilingual tasks and large-scale text
processing [18].Gemma 3 4B & 1BGemma is a lightweight,
family of models from Google built on Gemini technology.
The Gemma 3 models are multimodal—processing text and
images—and feature a 128K context window with support for
over 140 languages [19]. To demonstrate the effectiveness ofour proposed design, we consider the KB enhanced method
from our previous research [10] as a benchmark.
To evaluate our proposed intelligent IoT attack detection
design, we consider the dataset CICIOT 2023, which is a
real-time dataset and benchmark for large-scale attacks on
evaluating intrusion detection systems in IoT environments,
featuring diverse network traffic types, including normal and
malicious behaviors, across multiple IoT protocols and attack
scenarios [20]. We chose 500 high-quality samples each in
5 different types of DDoS attacks and Benign traffic.ICMP
Flood Attackoverwhelms the target with a high volume of
ICMP echo requests, causing the network to become congested
and unresponsive.UDP Flood Attacksends a large number of
UDP packets to random ports on the target server, forcing
it to process unnecessary requests.TCP SYN Flood Attack
exploits the TCP handshake mechanism by sending numerous
SYN packets without completing the handshake, consuming
server resources.TCP PSH+ACK Flood Attacksends a large
number of TCP packets with the PSH (Push) and ACK (Ac-
knowledgment) flags set to overwhelm the target’s processing
capabilities. Notably,PSHAKCandRSTFINattack type is
a subset ofDDOS TCP attack, which requires the model
to reason deeper and avoid just mimicking the response. For
instance, when only providing a knowledge base to ODLLM,
it might just give a short answer once it finds a DDoS TCP
attack and may not investigate further.
To evaluate the performance of our attack detection frame-
work, we adopt the macro-average F1 score as the primary
metric. Since our classification task involves six balanced
classes with 500 samples each, the macro-average F1 score
ensures equal weighting across all classes, providing a fair
assessment of the model’s overall performance. The F1 score
is the harmonic mean of precision and recall, capturing the
balance between false positives and false negatives: F1=
2×Precision×Recall
Precision+Recall. WherePrecisionmeasures how many of
the predicted positive instances are actually correct.Recall
measures how many of the actual positive instances were
correctly identified. To compute the macro-average F1 score,
the F1 score is calculated independently for each class and then
averaged: Macro-F1=1
NPN
i=1F1i, whereNis the number
of classes. This metric is particularly appropriate when classes
are evenly distributed, as it avoids bias toward any specific
class and provides a holistic view of model performance across
all types of network traffic.
The experimental results, summarized in Tables I and II,
reveal that model performance is critically dependent on the
chosen prompting strategy. We observe a distinct hierarchy of
effectiveness: Few-Shot RAG significantly outperforms One-
Shot RAG, which in turn is superior to knowledge base-
enhanced (Short KB), CoT, and baseline Zero-Shot (No KB)
methods. This finding holds true across both Gemma3 and
Llama 3.2 model families. In baseline zero-shot configurations,
all models struggled to process the traffic data, leading to
a state of ”mode collapse”. This was particularly severe for
smaller models; for example, Llama 3.2 1B (No KB) registered
a macro average F1-score of just 0.05, demonstrating a near-

(a) BGE base Embeddings
 (b) MLP base Embeddings
 (c) XGBoost base Embeddings
Fig. 2: 2D t-SNE projections of network traffic embeddings from three different base models.
Attack ICMP UDP TCP PSH/ACK RST/FIN Benign Macro avg
Gemma3 1BNo KB 0.42 1 0.49 0.33 0 0 0.37
Short KB 0.99 0.65 0.28 0 0 0 0.32
COT 0.2 0.62 0.31 0.3 1 0.13 0.43
One-Shot 0.77 0.98 0.35 0 0.25 0.74 0.51
Few-Shot 0.96 0.91 0.3 0.18 1 0.8 0.69
Gemma3 4BNo KB 0 1 0.12 0.82 0.98 0.08 0.5
Short KB 1 1 0.33 1 0.98 0 0.72
COT 0 0.5 0.24 0.83 0.95 0.4 0.49
One-Shot 1 0.95 0.68 0.95 0.98 0.8 0.89
Few-Shot 1 0.97 0.47 0.86 0.98 0.58 0.81
Llama 3.2 1BNo KB 0.22 0.72 0.2 0 0 0.38 0.25
Short KB 0.28 0.86 0.21 0 0 0.22 0.26
COT 0.49 0.55 0.2 0.19 0.29 0.15 0.31
One-Shot 0.97 0.99 0.29 0 0 0.85 0.52
Few-Shot 0.94 0.98 0.3 0 0 0.95 0.53
Llama 3.2 3BNo KB 0 1 0.02 1 1 0.21 0.54
Short KB 1 1 0.25 1 0 0.76 0.67
COT 0.91 0.97 0.34 0.89 0.99 0.26 0.73
One-Shot 0.91 0.89 0.42 0.93 0.98 0.69 0.8
Few-Shot 0.91 0.92 0.52 0.94 0.99 0.57 0.81
TABLE I: Precision Score of LLMs on DDoS Attack Types Using Different Reasoning Methods.
total inability to differentiate between attack classes.
Our analysis highlights a contrast between the failure of
abstract reasoning and the success of in-context learning. The
Inefficacy of CoT: The CoT strategy, intended to elicit step-
by-step reasoning, proved ineffective and often detrimental.
For instance, the Gemma3 1B with CoT experiment yielded
a macro average F1-score of only 0.13, one of the lowest
in our evaluation. Qualitative analysis of the model’s out-
puts confirmed that the generated ”reasoning” was frequently
filled with logical fallacies, such as ”protocol blindness” and
misinterpretation of numerical values, constituting a form of
confident hallucination. This indicates that forcing a model
to ”think” about a task it fundamentally does not understand
leads to fabricated logic rather than genuine insight.
The introduction of just one (One-Shot) and three (Few-
Shot) solved examples via RAG was the definitive factor in
unlocking high-performance detection. This approach provides
the model with a imitable template for analysis. The impact
was dramatic: the macro F1-score for Llama 3.2 1B surged
from 0.05 (No KB) to 0.50 (Few-Shot). Even smaller models
learned to achieve high accuracy on specific classes, such as
Llama 3.2 1B reaching 0.92 macro F1-score on ICMP floods
with a few-shot prompt. This demonstrates that for structured
tasks, LLMs excel not at abstract reasoning, but at analogicalmapping from clear exemplars.
Model scale acts as a crucial amplifier for the effectiveness
of a given prompting strategy. While a better strategy improves
all models, a larger model is better equipped to capitalize
on a sophisticated strategy. The benefit of scale is most
pronounced in the Few-Shot RAG setting, where larger models
are more adept at ”meta-learning” by abstracting a general
methodology from examples. The performance gap between
small and large models is widest here. The Gemma3 4B
model (Few-Shot) achieved an impressive macro F1-score of
0.77, substantially outperforming its 1B counterpart (0.48).
The most robust performance was delivered by the Llama
3.2 3B model under a few-shot regimen, which achieved the
highest macro average F1-score of 0.75 across its family, with
consistently high F1-scores on complex, nested classes like
PSH/ACK (0.85) and RST/FIN (0.79). This underscores the
synergistic effect of sufficient model scale and high-quality,
example-based prompting.
In conclusion, our results provide strong evidence that while
off-the-shelf ODLLMs are ill-suited for DDoS detection, their
performance can be dramatically enhanced through carefully
engineered, example-driven RAG prompts. The combination
of a sufficiently scaled model (3B+ parameters) and a few-
shot learning strategy creates a powerful and accurate sys-

Attack ICMP UDP TCP PSH/ACK RST/FIN Benign Macro avg
No KB 0.59 0.42 0.57 0.44 0 0 0.34
Short KB 0.96 0.77 0.43 0 0 0 0.36
COT 0.31 0.24 0.09 0.05 0.03 0.06 0.13
One-Shot 0.73 0.83 0.34 0 0 0.59 0.42Gemma3 1B
Few-Shot 0.85 0.8 0.38 0.01 0 0.820.48
No KB 0 0.98 0.11 0.85 0.99 0.1 0.51
Short KB 1 0.98 0.5 0 0.99 0 0.58
COT 0 0 0.03 0.11 0.14 0.01 0.05
One-Shot 0.83 0.94 0.78 0.96 0.98 0.640.85Gemma3 4B
Few-Shot 0.67 0.94 0.56 0.91 0.96 0.57 0.77
No KB 0.19 0.05 0.04 0 0 0.03 0.05
Short KB 0.36 0.16 0.22 0 0 0.04 0.13
COT 0.27 0.25 0.22 0.04 0.02 0.05 0.14
One-Shot 0.83 0.82 0.45 0 0 0.580.45Llama 3.2 1B
Few-Shot 0.92 0.89 0.45 0 0 0.77 0.5
No KB 0 0.63 0.02 0 0.69 0.34 0.28
Short KB 0.99 0.89 0.39 0.15 0 0.11 0.42
COT 0.8 0.82 0.47 0.68 0.75 0.17 0.62
One-Shot 0.7 0.73 0.57 0.84 0.82 0.6 0.71Llama 3.2 3B
Few-Shot 0.83 0.79 0.62 0.85 0.79 0.60.75
TABLE II: F1 Score of LLMs on DDoS Attack Types Using Different Reasoning Methods.
tem for classifying sophisticated network attacks in resource-
constrained environments.
VI. CONCLUSIONS
In this study, we presented a novel approach to enhancing
the capabilities of small-scale ODLLMs for accurate and
efficient detection of IoT-based DDoS attacks through CoT
prompting and RAG. Our work evaluates compact models like
LLaMA 3.2 and Gemma 3 across a spectrum of prompting
strategies. We found that abstract reasoning approaches, partic-
ularly standalone CoT prompting, fail spectacularly and suffer
from severe logical fallacies. In contrast, the RAG framework
providing a few concrete examples proved transformative. This
approach effectively re-purposes the ODLLM from a failed
analyst into a highly efficient and accurate pattern-matching
engine. This underscores that for ODLLMs operating on struc-
tured data, contextual grounding through relevant exemplars
is not just beneficial—it is essential for achieving reliable
performance. This paradigm shift from abstract reasoning
to guided, example-based mapping provides a robust and
replicable methodology for deploying ODLLMs in critical,
structured-data environments like network security.
REFERENCES
[1] J. Zhang, S. Liang, F. Ye, R. Q. Hu, and Y . Qian, “Towards detection
of zero-day botnet attack in iot networks using federated learning,” in
ICC 2023 - IEEE International Conference on Communications, 2023,
pp. 7–12.
[2] N. Jaton, S. Gyawali, and Y . Qian, “Distributed neural network-based
ddos detection in vehicular communication systems,” in2023 16th
International Conference on Signal Processing and Communication
System (ICSPCS), 2023, pp. 1–9.
[3] J. Zhu, S. Cai, F. Deng, B. C. Ooi, and J. Wu, “Do llms understand
visual anomalies? uncovering llm’s capabilities in zero-shot anomaly
detection,” inProceedings of the 32nd ACM International Conference
on Multimedia, ser. MM ’24. New York, NY , USA: Association
for Computing Machinery, 2024, p. 48–57. [Online]. Available:
https://doi.org/10.1145/3664647.3681190
[4] L. Li, J. Li, C. Chen, F. Gui, H. Yang, C. Yu, Z. Wang, J. Cai, J. A.
Zhou, B. Shenet al., “Political-llm: Large language models in political
science,”arXiv preprint arXiv:2412.06864, 2024.[5] W. Chen, Z. Li, and M. Ma, “Octopus: On-device language model for
function calling of software apis,”arXiv preprint arXiv:2404.01549,
2024.
[6] J. Xu, Z. Li, W. Chen, Q. Wang, X. Gao, Q. Cai, and Z. Ling,
“On-device language models: A comprehensive review,”arXiv preprint
arXiv:2409.00088, 2024.
[7] W. Chen, Z. Li, S. Xin, and Y . Wang, “Dolphin: Long context as a
new modality for energy-efficient on-device language models,”arXiv
e-prints, pp. arXiv–2408, 2024.
[8] J. Xu, Q. Wang, Y . Cao, B. Zeng, and S. Liu, “A general purpose device
for interaction with llms,” inProceedings of the Future Technologies
Conference. Springer, 2024, pp. 613–626.
[9] L. Gao, J. Sherwood, N. Aleisa, A. Damoah, Y . Lu, and X. Qu, “Human-
centered ai agents for healthcare and education: A systematic literature
review.”
[10] S. Verma, Q. Wang, and E. Bethel, “Intelligent iot attack detection design
via odllm with feature ranking-based knowledge base,”arXiv preprint
arXiv:2503.21674, 2025.
[11] L. Weng, “Why we think,”lilianweng.github.io, 2025. [Online].
Available: https://lilianweng.github.io/posts/2025-05-01-thinking/
[12] J. Wei, X. Wang, D. Schuurmanset al., “Chain of thought prompting
elicits reasoning in large language models,” inAdvances in Neural
Information Processing Systems (NeurIPS), 2022.
[13] P. Lewis, E. Perez, A. Piktuset al., “Retrieval-augmented generation
for knowledge-intensive nlp tasks,” inAdvances in Neural Information
Processing Systems (NeurIPS), 2020.
[14] C. Snell, J. Lee, K. Xu, and A. Kumar, “Scaling llm test-time compute
optimally can be more effective than scaling model parameters,”arXiv
preprint arXiv:2408.03314, 2024.
[15] T. Kojima, S. S. Gu, M. Reid, Y . Matsuo, and Y . Iwasawa, “Large lan-
guage models are zero-shot reasoners,”Advances in neural information
processing systems, vol. 35, pp. 22 199–22 213, 2022.
[16] L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based
models still outperform deep learning on typical tabular data?”Advances
in neural information processing systems, vol. 35, pp. 507–520, 2022.
[17] F. Liu, Z. Kang, and X. Han, “Optimizing rag techniques for automotive
industry pdf chatbots: A case study with locally deployed ollama
models,”arXiv preprint arXiv:2408.05933, 2024.
[18] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman,
A. Mathur, A. Schelten, A. Yang, A. Fanet al., “The llama 3 herd of
models,”arXiv preprint arXiv:2407.21783, 2024.
[19] G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej,
S. Perrin, T. Matejovicova, A. Ram ´e, M. Rivi `ereet al., “Gemma 3
technical report,”arXiv preprint arXiv:2503.19786, 2025.
[20] E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A.
Ghorbani, “Ciciot2023: A real-time dataset and benchmark for large-
scale attacks in iot environment,”Sensors, vol. 23, no. 13, p. 5941,
2023.