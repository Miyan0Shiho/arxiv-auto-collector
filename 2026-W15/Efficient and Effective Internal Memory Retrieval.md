# Efficient and Effective Internal Memory Retrieval for LLM-Based Healthcare Prediction

**Authors**: Mingchen Li, Jiatan Huang, Zonghai Yao, Hong yu

**Published**: 2026-04-08 23:56:05

**PDF URL**: [https://arxiv.org/pdf/2604.07659v1](https://arxiv.org/pdf/2604.07659v1)

## Abstract
Large language models (LLMs) hold significant promise for healthcare, yet their reliability in high-stakes clinical settings is often compromised by hallucinations and a lack of granular medical context. While Retrieval Augmented Generation (RAG) can mitigate these issues, standard supervised pipelines require computationally intensive searches over massive external knowledge bases, leading to high latency that is impractical for time-sensitive care. To address this, we introduce Keys to Knowledge (K2K), a novel framework that replaces external retrieval with internal, key-based knowledge access. By encoding essential clinical information directly into the model's parameter space, K2K enables rapid retrieval from internal key-value memory without inference-time overhead. We further enhance retrieval quality through activation-guided probe construction and cross-attention reranking. Experimental results demonstrate that K2K achieves state-of-the-art performance across four benchmark healthcare outcome prediction datasets.

## Full Text


<!-- PDF content starts -->

Efficient and Effective Internal Memory Retrieval for LLM-Based
Healthcare Prediction
Mingchen Li2, Jiatan Huang1, Zonghai Yao2, Hong Yu2,3,4
1University of Connecticut
2University of Massachusetts, Amherst
3University of Massachusetts, Lowell
4UMass Chan Medical School
Abstract
Large language models (LLMs) hold signifi-
cant promise for healthcare, yet their reliability
in high-stakes clinical settings is often compro-
mised by hallucinations and a lack of granular
medical context. While Retrieval-Augmented
Generation (RAG) can mitigate these issues,
standard supervised pipelines require compu-
tationally intensive searches over massive ex-
ternal knowledge bases, leading to high latency
that is impractical for time-sensitive care. To
address this, we introduce Keys-to-Knowledge
(K2K), a novel framework that replaces exter-
nal retrieval with internal, key-based knowl-
edge access. By encoding essential clinical
information directly into the model’s parameter
space, K2K enables rapid retrieval from inter-
nal key–value memory without inference-time
overhead. We further enhance retrieval quality
through activation-guided probe construction
and cross-attention reranking. Experimental
results demonstrate that K2K achieves state-
of-the-art performance across four benchmark
healthcare outcome prediction datasets.1
1 Introduction
Large language models (LLMs) have demonstrated
significant potential across diverse healthcare ap-
plications (Li and Ji, 2022; Achiam et al., 2023; Li
et al., 2024b; Guo et al., 2025; Li et al., 2024a,c).
However, their deployment in high-stakes clini-
cal environments is often hindered by hallucina-
tions and an inherent difficulty in accessing gran-
ular, patient-specific context. While Retrieval-
Augmented Generation (RAG) has emerged as a
primary strategy to ground these models in exter-
nal clinical knowledge (Lewis et al., 2020; Li et al.,
2025), existing approaches—which typically re-
trieve from structured knowledge graphs (Li and
Huang, 2023; Zhang et al., 2025), unstructured
1The code is available here: https://anonymous.4open.
science/r/K2K-2390/README.md
Figure 1: Overview of K2K. The input X consists of
longitudinal EHR diagnostic codes, and the output y
represents healthcare prediction results. K2K first con-
structs an internal memory within the LLM by inject-
ing external medical knowledge through the Internal
Memory Construction module. Given the input, K2K
then employs Activation-Guided Probe Construction to
build a probe query that retrieves relevant information
from the internal memory. Finally, the Cross-Attentive
Reranking module dynamically integrates diverse re-
trieved knowledge. The aggregated memory knowledge,
together with the learned input embeddings, is used for
the final healthcare prediction.
documents (Jin et al., 2025), or self-generated
knowledge- introduce a significant computational
burden.
Recent work (Su et al., 2025) identifies two criti-
cal bottlenecks in these traditional pipelines. First,
injecting external knowledge via input prompts ex-
pands context length, which escalates inference
costs and limits scalability. Second, construct-
ing high-quality retrievers remains a major hur-
dle; supervised retrieval requires extensive labeled
query–context pairs, while structured retrieval of-
ten relies on costly graph searches or oversimpli-
fied heuristics (Li et al., 2024b) that sacrifice ac-
curacy for coverage. These challenges create sub-
stantial overhead in data annotation and computa-
tion, particularly when navigating large-scale, het-
erogeneous medical knowledge bases. Ultimately,arXiv:2604.07659v1  [cs.CL]  8 Apr 2026

the requirement for exhaustive external searches
results in high end-to-end latency, a prohibitive
flaw in time-sensitive clinical settings where rapid
decision-making is vital.
To overcome these limitations, we introduce
Keys-to-Knowledge (K2K), a novel framework that
bypasses external retrieval by facilitating internal,
key-based knowledge access directly within the
model’s architecture. As prior work has demon-
strated (Geva et al., 2020), the keys within the
feed-forward network (FFN) layers of transformer-
based models implicitly store factual knowledge.
A promising research direction is to retrieve these
query-relevant FFN keys as internal knowledge,
enabling direct access to information without the
burden of long external contexts or the complexity
of structured knowledge base searches.
However, using the raw query alone to retrieve
top-kkeys, without incorporating contextual ac-
tivation signals, does not guarantee accuracy or
relevance. Our preliminary experiments indicate
that disparate queries often yield highly similar re-
trieved keys, suggesting that the resulting probe
representations lack discriminative power. Specif-
ically, these representations tend to obscure vital
semantic distinctions, a phenomenon also observed
in recent studies (Xiao et al., 2025), which ulti-
mately degrades retrieval effectiveness. Moreover,
internal key-space retrieval faces two significant
hurdles:Lack of Grounding: The retrieved key
vectors are latent and ungrounded, lacking the ex-
plicit provenance found in documents or knowl-
edge graphs.Static Retrieval: The process remains
non-adaptive, lacking explicit semantic signals to
guide the dynamic reweighting of retrieved knowl-
edge for specific downstream tasks.
To address these challenges, K2K retrieves key-
based knowledge from LLM purposefully infused
with external clinical information through three
core modules. As illustrated in Figure 1,Internal
Memory Constructiongrounds latent keys with ex-
ternal clinical knowledge,Activation-Guided Probe
Constructionenhances probe discriminability by
incorporating contextual activation signals, and
Cross-Attentive Rerankingenables adaptive, task-
aware reweighting of retrieved knowledge.
Our contributions are summarized as follows:
Internal Memory Construction: We transform
the pre-trained language model into a retrieval
memory by leveraging the keys stored in FFN lay-
ers. For domain-specific knowledge absent from
the pre-training corpus, we employ LoRA (Huet al., 2021) to inject new information into the pa-
rameter space. This approach mitigates reliance
on external retrievers and eliminates the latency
overhead of long input contexts.
Activation-Guided Probe Construction: To
ensure accurate retrieval from internal memory,
we design a probe-query mechanism that identi-
fies critical tokens and scarce outlier features dur-
ing inference. We utilize an activation bias, com-
puted via a diagonal approximation of the Maha-
lanobis distance, to emphasize query vectors with
high discriminative power, thereby balancing per-
dimension variance and improving retrieval preci-
sion.
Cross-Attentive Reranking: To account for
varying relevance and structural dependencies
across retrieved knowledge, we introduce a cross-
attentive reranking mechanism. This component
dynamically integrates and reweights multi-source
internal knowledge conditioned on the specific clin-
ical query, ensuring context-dependent integration
for downstream tasks.
2 Preliminaries
FFN Architecture: In transformer-based archi-
tectures (Vaswani et al., 2017), the feed-forward
network (FFN) operates in tandem with the self-
attention mechanism to transform hidden represen-
tations. Each FFN layer acts as a position-wise
function, processing each input vector indepen-
dently. Given an input vector x∈Rdtypically
sourced from a preceding attention layer, the out-
put of the FFN, denoted asFF(x), is defined as:
FF(x) =σ(xW 1+b 1)W2+b 2 (1)
where W1∈Rd×d ffandW2∈Rdff×dare
weight matrices, b1andb2are bias terms, and
σ(·) denotes a non-linear activation function (e.g.,
ReLU or GeLU).
Key-Value Memory Interpretation: Following
Geva et al. (Geva et al., 2020), the FFN can be
viewed as an associative key-value memory sys-
tem. In this view, the weight matrices represent the
stored knowledge: W1serves as the keys and W2
as the values. Specifically, the FFN operation can
be reformulated as:
FF(x) =dffX
i=1σ(x·k i)vi (2)
where ki(thei-th column of W1) is a key vector

Transformer layersTransformer layerTransformer layers
FFN layer
…  𝐊𝟐   𝐊𝟏   𝐊𝐝𝐦  𝐊𝟏∗ 𝐊𝟐∗ 𝐊𝐝𝐦∗
  𝐊𝐝𝐨𝐜   𝐊𝐠𝐫𝐚𝐩𝐡
Self attention layer
1) Internal Memory ConstructionHଵ
Hଶ
Hଷ
HHଵା 𝐟(.) 𝐂𝐀 𝐓𝐨𝐩−𝐤 𝐊𝐝𝐨𝐜
Hୢ୭ୡଵ
Hଶା 𝐟(.) 𝐂𝐀 𝐓𝐨𝐩−𝐤 𝐊𝐠𝐫𝐚𝐩𝐡
HଵାH୥୰ୟ୮୦ଵ 𝐩𝐨𝐨𝐥𝐢𝐧𝐠
2) Activation-Guided
Probe Query construction3) Cross-Attentive RerankingHOutput: y
Same steps as for HଵାQ1 ෪
Q2 ෪
input: X𝐌𝐋𝐏
 𝐋𝐋𝐌Figure 2: Overview of the K2K framework, consisting of three steps: (1) Retrieval Memory Construction builds
Kdoc= [K 1, K2,···, K dm]andKgraph= [K∗
1, K∗
2,···, K∗
dm]; (2) Activation-Guided Probe Query Construction
(Blue box function f(.)) enhances the query representation for key retrieval from KdocandKgraph; (3) Cross-
Attentive Reranking retrieves relevant document knowledge Hw
docand graph knowledge Hw
graph for the enhanced
queryQ+
w=f(H+
w), and integrates them with the original representationHfor final prediction. Here,t∈1,2,3.
that captures specific semantic patterns, and vi(the
i-th row of W2) is the corresponding value vector.
This interpretation suggests that a model’s factual
knowledge is stored within these parameter-based
keys, providing the theoretical foundation for K2K
to treat the model’s internal parameter space as a
retrievable knowledge base.
Knowledge Infusion via LoRA: To incorporate
domain-specific medical knowledge not present
in the pre-training corpus, we employ Low-Rank
Adaptation (LoRA) (Hu et al., 2021). LoRA rep-
resents weight updates as the product of two low-
rank matrices, AandB. Under this formulation,
the adapted FFN becomes:
FF(x) =σ(x(W 1+A 1B1)+b 1)(W 2+A 2B2)+b 2
(3)
In the context of K2K, the effective internal
memory consists of the updated key and value
matrices, defined as K⊤=W 1+A 1B1and
V=W 2+A 2B2, respectively. By freezing the
pre-trained weights and only updating the low-rank
adapters, we efficiently infuse clinical knowledge
into the model’s internal memory space while main-
taining its general reasoning capabilities.
3 Methodology
As illustrated in Figure 2, the K2K framework op-
erates in three distinct stages: (1)Retrieval Mem-
ory Construction, where domain-specific knowl-
edge is infused into the model’s internal param-
eter space; (2)Activation-Guided Probe Con-
struction, which identifies salient context features
for precise knowledge matching; and (3)Cross-Attentive Reranking, which dynamically inte-
grates the retrieved internal knowledge for the final
prediction.
3.1 Retrieval Memory Construction
The retrieval memory in K2K encompasses two
primary types of clinical information:
Document-Level Memory: We utilize a pre-
trained large language model ( Mbase) as our foun-
dational backbone. Rather than performing compu-
tationally expensive continued pre-training on the
entire model, we leverage a domain-adapted model
(Mdoc
domain ). Following the memory interpretation
of FFNs, we treat the Keys ( W1) within the l-th
Transformer layer of Mdoc
domain as the internal rep-
resentation of document-level knowledge, denoted
asKl
doc.
Graph-Level Memory: To incorporate struc-
tured knowledge, we linearize triples from a medi-
cal knowledge graph into textual descriptions (e.g.,
"The relationship between [head] and [tail] is [rela-
tion]"). We then apply LoRA-based adaptation to
Mdoc
domain using this organized triples dataset. The
resulting LoRA adapter matrices A1B1(per Equa-
tion 3) from the FFN layers are designated as the
structured knowledge source for layer l, denoted as
Kl
graph.
3.2 Activation-Guided Probe Query
Construction
As noted by (Xiao et al., 2025), standard probe
queries often rely on mean pooling, which dis-
perses attention across all tokens and fails to cap-
ture core semantics. To address this, we propose
aContextual Activation Weightto identify the

most informative query vectors within a context
window.
Given a sequence of query vectors2Hw=
[hw
1, hw
2, . . . , hw
L]for window wwith length Lin
the input X, we first compute the statistical mean
¯zw:
¯zw=1
LLX
j=1hw
j (4)
While prior work utilized Euclidean distance
to weight tokens, that approach assumes uniform
importance across all embedding dimensions. To
better account for per-dimension variance and in-
crease sensitivity to deviations in low-variance di-
rections, we employ a diagonal approximation of
the Mahalanobis distance:
ϕw
j≈vuutDX
d=1(hw
j,d−¯zw
d)2
σ2
d(5)
where σ2
ddenotes the variance computed across
all tokens within the window along the d-th embed-
ding dimension, ¯zw
ddenotes the d-th component
of the mean vector ¯zw. We then normalize these
scores to obtain a soft attention distributionαw
j:
αw
j=ϕw
jPL
j=1ϕw
j(6)
Finally, the enhanced probe vector Qwis con-
structed by aggregating token vectors hw
jusing
these weights, effectively emphasizing semanti-
cally grounded "anchor" tokens: then normalize
these
Qw=f(H w) =LX
j=1αw
j·hw
j (7)
3.3 Cross Attention Reranking
To perform cross-attention reranking, follow-
ing RETRO (Borgeaud et al., 2022), we first
split the representation Hof input sentence
into a sequence of w−1 windows, denoted as
{H+
1, H+
2, . . . , H+
w−1}.H+
wrepresents the query
embeddings constructed by concatenating the last
token of window wand the first L−1 tokens of
window w+ 1 . For each window w, we generate
an enhanced query representation Q+
w=f(H+
w).
We then retrieve the top- kmost relevant vectors
2The query vectors for window ware the embeddings
encoded by the LLM from the input sequenceX.from the internal document and graph memories
based on similarity scores:
Kw
doc=top-k 
sim(Q+
w, Kl
doc)
Kw
graph=top-k 
sim(Q+
w, Kl
graph) (8)
We apply a Cross-Attention (CA) mechanism
to rerank and refine these vectors, yielding the fi-
nal retrieved document knowledge Hw
docand graph
knowledgeHw
graph:
Hw
doc=CA(Q+
w, Kw
doc, Vw
doc)
Hw
graph=CA(Q+
w, Kw
graph, Vw
graph)(9)
The retrieved vectors are normalized via a pool-
ing functionP(·)and fused via concatenation:
˜Qw= [P(Hw
doc);P(Hw
graph)](10)
We then aggregate all window-level fused represen-
tations together with the input sentence representa-
tionHand feed the combined representation into
an MLP for final prediction. The loss is defined as:
Lcls=CrossEntropy
MLP([H; ˜Q1. . .˜Qw−1]), y
(11)
whereydenotes the ground truth label.
4 Experiments
4.1 Problem Formulation
We define the clinical history of a patient as a se-
quence of hospital visits V={v 1, v2, . . . , v |V|}.
Each visit viis associated with a set of Inter-
national Classification of Diseases (ICD) codes
Ci={c i,1, ci,2, . . .} , where each code c∈C irep-
resents a specific diagnosis or procedure. For each
code, we utilize its corresponding clinical descrip-
tors, provided as a short text snippet (e.g., "Acute
myocardial infarction").The goal of our model is
to predict a binary clinical outcome y∈ {0,1}
based on the historical context. We evaluate our
framework on two critical tasks:
Mortality Prediction: yiindicates whether the
patient’s death occurs during the subsequent visit
vi+1.
Readmission Prediction: Following the pro-
tocol in KARE (Jiang et al., 2024), yipredicts
whether the patient will be readmitted to the hospi-
tal withinα= 15days.
4.2 Dataset
We evaluate K2K on the publicly available MIMIC-
III (Johnson et al., 2016) and MIMIC-IV (Johnson
et al., 2020) datasets. The statistics for the pro-
cessed data are summarized in Table 1.

III-Mort III-Read IV-Mort IV-Read
Train 7,777 7,777 100,125 10,0125
Test 953 953 12,667 12,667
Dev 978 978 12,547 12,547
Table 1: Datasets Statistics, Mort refers to the Mortality.
III refers to the MIMIC-III. Read refers to readmission.
We employ a 0.8/0.1/0.1 split ratio for training,
validation, and testing, respectively. To prevent
data leakage, the split is grouped by patient ID,
ensuring that all visits from a single patient are
confined to a single subset. While prior work such
as KARE (Jiang et al., 2024) utilizes a random
subset of MIMIC-IV , we utilize the full dataset
to better simulate the scale and heterogeneity of
real-world clinical settings.
4.3 Baselines and Evaluation Metrics
We compare K2K against a diverse range of com-
petitive baselines:
Sequential Models: GRU (Chung et al., 2014),
RETAIN (Choi et al., 2016), Deepr (Nguyen et al.,
2016), AdaCare (Ma et al., 2020), StageNet (Gao
et al., 2020), and TCN (Bai et al., 2018).
Retrieval-Based Models: KARE (Jiang et al.,
2025), the current state-of-the-art for healthcare
prediction, and Standard RAG (Li et al., 2024c),
which uses Contriver (Izacard et al., 2021) to re-
trieve relevant patient examples.
Generative Knowledge Models: Prompt-Based
Retrieval (Frisoni et al., 2024), which leverages
in-context learning to instruct an LLM to generate
task-relevant medical knowledge.
Evaluation Metrics: Following standard prac-
tices in clinical predictive modeling (Jiang et al.,
2025, 2023), we report the F1-score, Jaccard Simi-
larity, AUPRC, and AUROC. Detailed implemen-
tation parameters and hardware configurations are
provided in Appendix A.2.
4.4 Main Results
Table 2 presents the main results and highlights
several key observations for two LLMs, BioMistral-
7B and Meditron3-Qwen2.5-7B: (1) K2K demon-
strates consistently strong performance across
datasets and tasks, achieving the highest average
scores among the evaluated methods. (2) Base-
line retrieval methods fail to capture the semantic
nuances of the input. Although KARE enhances
retrieval by combining relevant documents withthe shortest paths from the graph, such paths may
overlook critical relational information. In contrast,
our method retrieves key knowledge directly from
the language model’s internal knowledge store, en-
abling more comprehensive and context-aware re-
trieval. (3) We find that BioMistral-7B performs
worse than traditional machine learning models
when the input contains discontinuous or complex
diagnoses and suffers from class imbalance be-
tween positive and negative samples. This is also
observed by Gao et al. (2025). While Meditron3-
Qwen2.5-7B exhibits stronger performance. By
introducing document-level knowledge and graph-
based knowledge into the language model, our
method achieves improved performance. For ex-
ample, K2K outperforms LLMs without retrieval
mechanisms on the mortality prediction task using
the MIMIC-IV dataset. (4) Under BioMistral-7B,
we found that prompt-based retrieval outperforms
standard RAG by retrieving knowledge from ex-
ternal documents, enabling the language model to
generate more useful information that improves
the classification results, as evidenced by improve-
ments in AUPRC and AUROC on the Mortality-
MIMIC-III dataset. Additional efficiency results
are reported in Appendix A.7, showing that K2K
achieves competitive average performance while
being substantially more efficient in retrieval time
compared with KARE and prompt-based methods
on the test set of MIMIC-III.
5 Analysis
To further evaluate the effectiveness of our frame-
work, we conduct a series of analyses based on
different components of our model. First, we inves-
tigate the impact of different knowledge sources
by introducing two ablations: K2K without docu-
ment knowledge and K2K without graph knowl-
edge (Section 5.1). We also assess the perfor-
mance of directly using the LLM with its internal
knowledge to make predictions, in order to validate
the effectiveness of our key knowledge retrieval
framework, which leverages cross-window atten-
tion (Section 5.2). Next, we compare different
query representation strategies to demonstrate the
effectiveness of our proposed diagonal approxima-
tion of the Mahalanobis distance (Section 5.3). Fi-
nally, we analyze the effect of retrieving knowledge
from different LLM layers (Section 5.4). In this
section, we use BioMistral-7B as the base LLM.
For additional experiments on K2K, including

Mortality-MIMIC-III Readmission-MIMIC-III
Type Model F1 Jac. AUPRC AUROC Avg F1 Jac. AUPRC AUROC Avg
MLGRU (Chung et al., 2014) 13.87 7.45 8.03 53.50 20.71 68.28 51.84 52.94 50.00 55.77
RETAIN (Choi et al., 2016) 13.73 7.37 9.57 54.86 21.38 45.88 23.48 54.11 51.29 43.69
Deepr (Nguyen et al., 2016) 13.87 7.45 7.58 51.66 20.14 68.28 51.84 51.68 49.70 55.38
AdaCare (Ma et al., 2020) 12.90 6.89 7.80 50.69 19.57 63.49 46.51 52.83 52.27 53.77
StageNet (Gao et al., 2020) 9.97 5.25 7.10 47.14 17.37 51.56 34.74 50.38 48.27 46.24
TCN (Bai et al., 2018) 11.28 5.97 6.76 45.81 17.45 65.46 48.66 49.84 47.65 52.90
LLMFine-tuned (LLM: BioMistral-7B)
w/o retriever 16.00 8.69 11.61 59.40 23.92 69.17 52.87 59.07 54.61 58.93
KARE (Jiang et al., 2025) 18.01 9.90 9.72 56.65 23.57 61.64 44.55 56.67 50.97 53.46
Standard RAG (Li et al., 2024c) 11.94 6.34 9.34 54.19 20.45 69.73 53.52 57.09 52.99 58.33
Prompt-based (Frisoni et al., 2024) 15.05 8.13 10.78 58.72 23.17 66.51 49.82 54.19 49.71 55.06
K2K (Ours)18.55 10.22 15.22 61.0526.2669.31 53.03 62.49 56.6460.37
GRPO 16.90 9.10 13.80 58.40 24.55 66.80 51.20 58.90 54.70 57.90
GRPO——advance 18.91 10.05 15.74 61.54 26.56 69.12 53.97 63.18 57.20 60.87
Fine-tuned (LLM: Meditron3-Qwen2.5-7B)
w/o retriever 13.78 7.40 7.00 48.90 19.27 68.18 51.72 57.72 53.01 57.66
KARE (Jiang et al., 2025) 11.67 6.19 10.38 56.19 21.11 59.45 42.30 55.66 50.89 52.08
Standard RAG (Li et al., 2024c) 16.35 8.90 8.43 54.11 21.95 63.00 45.98 58.14 53.90 55.26
Prompt-based (Frisoni et al., 2024) 15.44 8.36 12.01 54.29 22.52 60.15 43.01 60.25 55.38 54.70
K2K (Ours)17.27 9.45 10.29 54.5622.8968.78 52.42 59.88 56.0859.29
Mortality-MIMIC-IV Readmission-MIMIC-IV
Type Model F1 Jac. AUPRC AUROC Avg F1 Jac. AUPRC AUROC Avg
MLGRU (Chung et al., 2014) 3.20 1.62 1.66 53.71 15.05 59.28 42.13 57.38 56.58 53.84
RETAIN (Choi et al., 2016) 2.78 1.41 1.43 47.18 13.20 66.77 50.12 51.44 49.61 54.48
Deepr (Nguyen et al., 2016) 2.86 1.46 1.57 51.48 14.34 68.13 51.66 52.27 50.44 55.62
AdaCare (Ma et al., 2020) 2.98 1.52 1.53 51.41 14.36 47.96 31.54 52.12 50.38 45.50
StageNet (Gao et al., 2020) 2.96 1.50 1.60 51.11 14.29 48.11 31.67 50.74 48.67 44.80
TCN (Bai et al., 2018) 2.92 1.48 1.63 54.17 15.05 53.32 36.35 51.33 49.62 47.66
LLMFine-tuned (LLM: BioMistral-7B)
w/o retriever 1.08 0.50 1.30 44.61 11.87 61.30 44.20 67.86 65.83 59.80
KARE (Jiang et al., 2025) 1.33 0.67 1.46 49.55 13.25 61.75 44.67 67.09 65.44 59.74
Standard RAG (Li et al., 2024c) 2.45 1.61 2.74 55.92 15.68 60.95 43.84 68.51 66.64 59.98
Prompt-based (Frisoni et al., 2024) 3.16 1.60 1.49 48.26 13.63 61.02 43.91 68.89 67.02 60.21
K2K (Ours)6.61 3.42 2.93 66.5019.8763.75 46.79 68.67 66.4761.42
Fine-tuned (LLM: Meditron3-Qwen2.5-7B)
w/o retriever 3.41 1.73 2.72 63.52 17.85 56.92 39.79 67.76 65.86 57.58
KARE (Jiang et al., 2025) 4.41 2.25 1.96 59.20 16.96 64.62 47.74 68.50 66.76 61.91
Standard RAG (Li et al., 2024c) 4.47 2.29 2.35 64.87 18.49 65.09 48.25 67.68 65.78 61.70
Prompt-based (Frisoni et al., 2024) 4.17 2.13 2.17 59.68 17.04 57.71 40.55 66.91 64.84 57.50
K2K (Ours)5.10 2.61 2.59 63.8518.5467.11 50.50 67.60 66.2862.87
Table 2: Comparative analysis of mortality and readmission prediction on MIMIC-III and MIMIC-IV .Avgdenotes
the arithmetic mean of F1, Jaccard, AUPRC, and AUROC for each model on the corresponding dataset. Boldface
indicates the highest average score under each LLM.
the effect of different chunk sizes, the impact
of the hyperparameter top- kin Equation 8, and
analyses of retrieval efficiency across various re-
trieval methods and pipelines, please refer to Ap-
pendix A.5,A.6,A.7.
5.1 Impact of Different Knowledge Source
Table 3 presents the results of K2K using different
knowledge sources. Specifically, K2K w/o docu-
ment refers to the variant of K2K that uses only the
retrieved graph knowledge Kt
graph, as described in
Section 3.3. To ensure a fair comparison, the only
difference between K2K and its ablated versions
(w/o document or w/o graph) is the type of knowl-
edge source used. From Table 3, we observe thatthe performance of K2K drops when either docu-
ment or graph knowledge is removed, especially on
the MIMIC-III dataset. Moreover, although K2K
w/o graph achieves a higher F1 score, its lower
AUPRC and AUROC suggest that it may overfit
to a specific threshold and lacks robustness in dis-
tinguishing positive cases across varying decision
boundaries. In contrast, K2K achieves more bal-
anced performance across all metrics, indicating
better generalization and retrieval effectiveness.
5.2 Direct Use vs. Retrieved Use of
Pre-trained Knowledge
Table 4 shows the results of the experiments of
different knowledge-enhanced models.We found

Model F1 Jaccard AUPRC AUROC
Mortality-IIIK2K 18.55 10.2215.22 61.05
K2K w/o graph 20.48 11.4013.18 60.54
K2K w/o document 16.66 9.09 10.52 55.72
Mortality-IVK2K Ours 6.61 3.42 2.93 66.50
K2K w/o graph 4.50 2.30 2.51 60.86
K2K w/o document 3.57 1.82 2.71 66.41
Readmission-IIIK2K 69.31 53.0362.49 56.64
K2K w/o graph 70.95 54.9860.87 54.55
K2K w/o document 69.74 53.54 61.93 56.36
Readmission-IVK2K Ours 63.75 46.79 68.67 66.47
K2K w/o graph 55.31 38.23 66.14 64.06
K2K w/o document 56.95 39.81 55.43 64.68
Table 3: Results of different knowledge sources in K2K
that leveraging windowed cross-attention and
Mahalanobis-guided query construction to retrieve
internal key knowledge from the LLM yields supe-
rior performance compared to directly employing a
knowledge-augmented LLM for downstream tasks.
We guess the reason is that although knowledge
augmented LLMs such as BioMixtral 7B encode
medical knowledge through pretraining, they may
not explicitly surface critical risk factors for spe-
cific knowledge. For instance, in the MIMIC-III
mortality task, the model might miss the implica-
tion of structured features likemechanical ventila-
tionorhigh SOFA scoreif not directly prompted.
In contrast, our method retrieves relevant internal
knowledge from the encoded medical graph, such
as the relations between symptoms, interventions,
and mortality and fuses it into the model input.
This structured retrieval improves the model’s abil-
ity to reason over clinical signals and enhances
prediction accuracy.
5.3 Comparison of Query Representation
Strategies
Table 5 presents various query representation strate-
gies for assessing the importance of each query vec-
tor within a window context. K2K (Euclidean) uses
Euclidean distance for token weighting, whereas
K2K (Mean Only) computes the window represen-
tation via simple mean pooling. Table 5 shows
that our Mahalanobis-guided query representation
consistently outperforms prior approaches. Unlike
Euclidean distance, which treats all dimensions
equally, our method accounts for per-dimension
variance and emphasizes informative low-variance
directions. This leads to more precise token weight-
ing and better contextual representations. The re-
sults validate the effectiveness of variance-aware
distance metrics in enhancing retrieval-informedModel F1 Jaccard AUPRC AUROC
Mortality-IIIK2K 18.55 10.22 15.22 61.05
LLM 4.49 2.29 8.67 55.62
LLM+Doc 16.00 8.69 11.61 59.40
LLM+Graph 4.50 2.29 8.67 55.62
LLM+Doc+Graph 16.00 8.70 11.61 59.41
Readmission-IIIK2K 69.31 53.0362.49 56.64
LLM 64.10 47.17 60.81 54.57
LLM+Doc 69.17 52.87 59.07 54.61
LLM+Graph 44.31 28.46 56.57 48.87
LLM+Doc+Graph 70.81 54.8161.51 54.70
Mortality-IVK2K 6.61 3.42 2.93 66.50
LLM 2.05 1.03 1.59 51.64
LLM+Doc 1.08 0.50 1.30 44.61
LLM+Graph 3.24 1.60 1.52 50.08
LLM+Doc+Graph 1.08 0.55 1.30 44.61
Readmission-IVK2K 63.75 46.79 68.67 66.47
LLM 60.06 42.92 66.15 64.64
LLM+Doc 61.30 44.20 67.86 65.83
LLM+Graph 48.97 32.43 50.80 48.30
LLM+Doc+Graph 54.86 37.80 51.57 49.93
Table 4: Comparison of Knowledge-Enhanced Mod-
els on Mortality and Readmission Prediction (MIMIC-
III/IV). LLM refers to Mixtral-7B. LLM+Doc denotes
BioMixtral-7B, which is obtained by further training
Mixtral-7B on a medical corpus. LLM+Graph refers
to Mixtral-7B adapted to graph-based knowledge using
LoRA. LLM+Doc+Graph represents BioMixtral-7B fur-
ther adapted to graph knowledge via LoRA.
reasoning.
5.4 Comparison of Knowledge from Different
LLM Layers
In this section, we conduct experiments on K2K
using knowledge (key) sources stored in differ-
ent transformer layers within the LLM. Both the
document-based knowledge and the graph-based
knowledge are extracted from the same correspond-
ing layer. Figure 3 reveals a nuanced deviation
from the conventional view that upper layers in
Transformers primarily encode semantic features
while lower layers capture shallow, surface-level
patterns. Although the final layers (e.g., Layer 30+)
do contribute positively to performance in both
Mortality-MIMIC-III and Readmission-MIMIC-III
tasks, this improvement is not strictly monotonic.
Notably, several shallow layers (e.g., Layers 5, 8,
and 10) also exhibit strong performance across mul-
tiple metrics, indicating that valuable structural or
entity-level knowledge resides in the lower layers
as well. Furthermore, the impact of each layer
varies across different evaluation metrics (F1, Jac-
card, AUROC), suggesting that knowledge is dis-
tributed in a non-linear fashion throughout the net-
work. These findings underscore the importance of

Model F1 Jaccard AUPRC AUROC
Mortality-IIIK2K 18.55 10.22 15.22 61.05
K2K w Euclidean 16.97 9.27 9.67 57.25
K2K (Mean Only) 12.06 6.42 8.45 52.51
Readmission-IIIK2K 69.31 53.03 62.49 56.64
K2K w Euclidean 63.27 46.28 58.26 53.25
K2K (Mean Only) 63.98 47.03 54.67 50.92
Mortality-IVK2K 6.61 3.42 2.93 66.50
K2K w Euclidean 4.79 2.45 2.19 61.81
K2K (Mean Only) 0.82 0.44 2.51 61.73
Readmission-IVK2K 63.75 46.79 68.67 66.47
K2K w Euclidean 63.56 46.59 67.87 66.41
K2K (Mean Only) 56.26 39.14 67.71 65.58
Table 5: Comparison of K2K with different query con-
struction methods.
5 10 15 20 25 30
Transformer Layer102030405060Score
Mortality-MIMIC-III
F1
Jaccard
AUPRC
AUROC
5 10 15 20 25 30
Transformer Layer5055606570
Readmission-MIMIC-III
F1
Jaccard
AUPRC
AUROC
Figure 3: K2K performance with different layer knowl-
edge. We used BioMistral-7B, which consists of 32
transformer layers.
considering both shallow and deep layers in knowl-
edge extraction and reasoning tasks.
6 Related Work
Existing research has extensively explored leverag-
ing retrieved information from diverse knowledge
stores to enhance text understanding and genera-
tion (Lewis et al., 2020; Guu et al., 2020; Li and
Huang, 2023; Li et al., 2025; Jiang et al., 2025).
These methods generally fall into two categories:
structured and unstructured retrieval.
Structured Knowledge Retrieval: Several stud-
ies focus on integrating relational data. For in-
stance, KIEST (Li and Huang, 2023) dynamically
injects entity and attribute knowledge from knowl-
edge graphs to improve generation in entity-state-
change tasks. Similarly, KARE (Jiang et al., 2025)
identifies relevant entities for each concept in a
query and constructs a subgraph using shortest-path
heuristics, providing structured relational context
for downstream reasoning.
Unstructured and Chunk-based Retrieval:
Other approaches utilize massive text corpora.
REALM (Guu et al., 2020) pioneered a gradient-
based method to reward the retriever, thereby im-
proving prediction accuracy through latent knowl-edge retrieval. More recently, BiomedRAG (Li
et al., 2025) introduced a dynamic mechanism
to rerank top- kchunks from diverse biomedical
databases. Similarly, RETRO (Borgeaud et al.,
2022) utilizes a chunk-based approach with cross-
attention mechanisms to integrate retrieved seg-
ments from trillion-token scales.
Despite these advancements, traditional RAG re-
mains burdened by the latency of querying massive,
heterogeneous sources and the computational cost
of processing lengthy external contexts. To miti-
gate these challenges, we propose a novel approach
that retrieves knowledge directly from the internal
key space of the LLM. By utilizing an activation-
guided top- kselection and a cross-window atten-
tion mechanism, our framework enables efficient,
grounded, and low-latency knowledge access with-
out the overhead of external search.
Recent work (Xiao et al., 2024; Liu et al., 2024;
Fountas et al., 2025) has explored retrieval mod-
ules that extract information from the key-value
(KV) cache using probe queries derived from cur-
rent context tokens. These methods typically treat
the sliding window as a probe to retrieve relevant
KV pairs. However, most overlook the critical
role of probe construction, as LLMs are not in-
herently optimized for internal retrieval tasks. To
date, few studies have addressed how to optimize
probe queries for internal key retrieval. One no-
table exception is ActQKV (Xiao et al., 2025),
which proposes an activation-aware mechanism
that selects key tokens based on activation magni-
tude and employs Euclidean distance for retrieval.
However, this approach assumes equal importance
across all embedding dimensions, thereby ignoring
per-dimension variance and reducing sensitivity to
meaningful deviations in low-variance directions.
7 Conclusion
In this paper, we introduce K2K, a framework that
retrieves knowledge directly from an LLM’s inter-
nal key space, bypassing context-heavy prompt-
ing and retrieval-intensive pipelines used in con-
ventional RAG. With Mahalanobis-guided query
representations and cross-attentive reranking for
multi-source integration, K2K improves reasoning
in knowledge-intensive clinical tasks. Our results
show that LLM parameters are not merely latent
carriers of knowledge but can be explicitly accessed
to enhance predictive performance.

8 Limitations
While the K2K framework demonstrates robust per-
formance in internal knowledge retrieval and inte-
gration, several limitations remain to be addressed
in future work.
Layer Selection and Granularity: The retrieval
memory is currently constructed from fixed layers
of the pre-trained model. Although LoRA-based in-
fusion facilitates domain adaptation, our approach
does not yet dynamically select which specific lay-
ers or representations (e.g., lower-level semantic
features vs. higher-level abstraction in FFN layers)
are most informative for a given query. Implement-
ing a layer-wise selection or multi-layer weighting
mechanism could further improve retrieval fidelity
and computational efficiency.
Domain Generalization: Our framework has
been primarily evaluated within the biomedi-
cal domain. While this provides a rigorous
testbed for high-stakes decision-making, the gen-
eralizability of K2K to other knowledge-intensive
fields—such as legal reasoning or financial forecast-
ing—remains to be explored. Furthermore, while
K2K performs well on benchmark datasets, future
iterations must address the pervasive issue of data
imbalance inherent in real-world clinical tasks to
ensure equitable performance across rare medical
conditions.
References
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-
cal report.arXiv preprint arXiv:2303.08774.
Shaojie Bai, J Zico Kolter, and Vladlen Koltun. 2018.
An empirical evaluation of generic convolutional and
recurrent networks for sequence modeling.arXiv
preprint arXiv:1803.01271.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoff-
mann, Trevor Cai, Eliza Rutherford, Katie Milli-
can, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, and 1 others.
2022. Improving language models by retrieving from
trillions of tokens. InInternational conference on
machine learning, pages 2206–2240. PMLR.
Edward Choi, Mohammad Taha Bahadori, Jimeng Sun,
Joshua Kulas, Andy Schuetz, and Walter Stewart.
2016. Retain: An interpretable predictive model for
healthcare using reverse time attention mechanism.
Advances in neural information processing systems,
29.Junyoung Chung, Caglar Gulcehre, KyungHyun Cho,
and Yoshua Bengio. 2014. Empirical evaluation of
gated recurrent neural networks on sequence model-
ing.arXiv preprint arXiv:1412.3555.
Zafeirios Fountas, Martin A Benfeghoul, Adnan Oomer-
jee, Fenia Christopoulou, Gerasimos Lampouras,
Haitham Bou-Ammar, and Jun Wang. 2025. Human-
like episodic memory for infinite context llms.arXiv
preprint arXiv:2407.09450.
Giacomo Frisoni, Alessio Cocchieri, Alex Presepi, Gi-
anluca Moro, and Zaiqiao Meng. 2024. To generate
or to retrieve? on the effectiveness of artificial con-
texts for medical open-domain question answering.
arXiv preprint arXiv:2403.01924.
Junyi Gao, Cao Xiao, Yasha Wang, Wen Tang, Lucas M
Glass, and Jimeng Sun. 2020. Stagenet: Stage-aware
neural networks for health risk prediction. InPro-
ceedings of the web conference 2020, pages 530–540.
Yanjun Gao, Skatje Myers, Shan Chen, Dmitriy Dligach,
Timothy Miller, Danielle S Bitterman, Guanhua
Chen, Anoop Mayampurath, Matthew M Churpek,
and Majid Afshar. 2025. Uncertainty estimation in di-
agnosis generation from large language models: next-
word probability is not pre-test probability.JAMIA
open, 8(1):ooae154.
Mor Geva, Roei Schuster, Jonathan Berant, and Omer
Levy. 2020. Transformer feed-forward layers are key-
value memories.arXiv preprint arXiv:2012.14913.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao
Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shi-
rong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025.
Deepseek-r1: Incentivizing reasoning capability in
llms via reinforcement learning.arXiv preprint
arXiv:2501.12948.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu
Chen. 2021. Lora: Low-rank adaptation of large
language models.arXiv preprint arXiv:2106.09685.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.
Peng Jiang, Chang Xiao, Meng Jiang, and 1 others.
2025. Reasoning-enhanced healthcare predictions
with knowledge graph community retrieval. InPro-
ceedings of the International Conference on Learning
Representations (ICLR).
Pengcheng Jiang, Cao Xiao, Adam Cross, and Jimeng
Sun. 2023. Graphcare: Enhancing healthcare pre-
dictions with personalized knowledge graphs.arXiv
preprint arXiv:2305.12788.

Pengcheng Jiang, Cao Xiao, Minhao Jiang, Parminder
Bhatia, Taha Kass-Hout, Jimeng Sun, and Jiawei Han.
2024. Reasoning-enhanced healthcare predictions
with knowledge graph community retrieval.arXiv
preprint arXiv:2410.04585.
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon,
Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei
Han. 2025. Search-r1: Training llms to reason and
leverage search engines with reinforcement learning.
arXiv preprint arXiv:2503.09516.
Alistair Johnson, Lucas Bulgarelli, Tom Pollard,
Steven Horng, Leo Anthony Celi, and Roger Mark.
2020. Mimic-iv.PhysioNet. Available online at:
https://physionet. org/content/mimiciv/1.0/(accessed
August 23, 2021), pages 49–55.
Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H
Lehman, Mengling Feng, Mohammad Ghassemi,
Benjamin Moody, Peter Szolovits, Leo Anthony Celi,
and Roger G Mark. 2016. Mimic-iii, a freely accessi-
ble critical care database.Scientific data, 3(1):1–9.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Ad-
vances in Neural Information Processing Systems.
Mingchen Li, Jiatan Huang, Jeremy Yeung, Anne Blaes,
Steven Johnson, Hongfang Liu, Hua Xu, and Rui
Zhang. 2024a. Cancerllm: A large language model
in cancer domain.arXiv preprint arXiv:2406.10459.
Mingchen Li and Lifu Huang. 2023. Understand the
dynamic world: An end-to-end knowledge informed
framework for open domain entity state tracking. Pro-
ceedings of the 46th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval.
Mingchen Li and Shihao Ji. 2022. Semantic struc-
ture based query graph prediction for question an-
swering over knowledge graph.arXiv preprint
arXiv:2204.10194.
Mingchen Li, Halil Kilicoglu, Hua Xu, and Rui Zhang.
2025. Biomedrag: A retrieval augmented large lan-
guage model for biomedicine.Journal of Biomedical
Informatics, 162:104769.
Mingchen Li, Chen Ling, Rui Zhang, and Liang Zhao.
2024b. Zero-shot link prediction in knowledge
graphs with large language models. In2024 IEEE
International Conference on Data Mining (ICDM),
pages 753–760. IEEE.
Mingchen Li, Zaifu Zhan, Han Yang, Yongkang Xiao,
Jiatan Huang, and Rui Zhang. 2024c. Benchmark-
ing retrieval-augmented large language models in
biomedical nlp: Application, robustness, and self-
awareness.arXiv preprint arXiv:2405.08151.Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang,
Zhenhua Han, Qianxi Zhang, Qi Chen, Chen-
gruidong Zhang, Bailu Ding, Kai Zhang, and 1 others.
2024. Retrievalattention: Accelerating long-context
llm inference via vector retrieval.arXiv preprint
arXiv:2409.10516.
Liantao Ma, Junyi Gao, Yasha Wang, Chaohe Zhang,
Jiangtao Wang, Wenjie Ruan, Wen Tang, Xin Gao,
and Xinyu Ma. 2020. Adacare: Explainable clin-
ical health status representation learning via scale-
adaptive feature extraction and recalibration. InPro-
ceedings of the AAAI Conference on Artificial Intelli-
gence, volume 34, pages 825–832.
Phuoc Nguyen, Truyen Tran, Nilmini Wickramasinghe,
and Svetha Venkatesh. 2016. Deepr: A convolutional
net for medical records. arxiv. org.
Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan,
Changyue Wang, Hongning Wang, Ziyi Ye, Yu-
jia Zhou, and Yiqun Liu. 2025. Parametric re-
trieval augmented generation.arXiv preprint
arXiv:2501.15915.
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob
Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz
Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in neural information processing
systems, 30.
Chaojun Xiao, Pengle Zhang, Xu Han, Guangxuan Xiao,
Yankai Lin, Zhengyan Zhang, Zhiyuan Liu, Song
Han, and Maosong Sun. 2024. Infllm: Unveiling the
intrinsic capacity of llms for understanding extremely
long sequences with training-free memory.arXiv e-
prints, pages arXiv–2402.
Qingfa Xiao, Jiachuan Wang, Haoyang Li, Cheng Deng,
Jiaqi Tang, Shuangyin Li, Yongqi Zhang, Jun Wang,
and Lei Chen. 2025. Activation-aware probe-query:
Effective key-value retrieval for long-context llms
inference.arXiv preprint arXiv:2502.13542.
Qinggang Zhang, Shengyuan Chen, Yuanchen Bei,
Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong,
Hao Chen, Yi Chang, and Xiao Huang. 2025. A
survey of graph retrieval-augmented generation for
customized large language models.arXiv preprint
arXiv:2501.13958.
A Appendices
A.1 Effect of LLM Backbone Choice
A.2 Implementation Detail
The chunk size is set to 64 throughout this work.
For the top- kvalues, we use k= 5 for Mortality-
MIMIC-III, k= 20 for Readmission-MIMIC-
III and Mortality-MIMIC-IV , and k= 10 for
Readmission-MIMIC-IV . The same LLM back-
bone is used during both the retrieval phase,
when keys are extracted, and the training/inference

phases, when those keys are utilized, ensuring
alignment in the representation space. We use
AdamW as our optimizer, with a learning rate of
2×10−5andϵset to 1×10−8. The batch size
is 16. For the cross-attention module, we set the
model dimension to 4096 and apply a dropout rate
of 0.3.
A.3 Separately retrieval
We intentionally use only the base component W1
from the final FFN layer of Mdoc
domain to represent
document knowledge. This design is motivated by
the need to preserve a clear and interpretable sepa-
ration between knowledge sources. Specifically, (1)
theoretically, unstructured document knowledge
(captured by W1) and structured graph knowledge
(injected via AB) differ fundamentally in format
and reasoning mechanisms, and thus should not be
merged directly in representation; (2)in practice,
combining them into a single matrix W1+AB
would entangle their contributions, making it diffi-
cult to analyze or attribute model behavior to spe-
cific knowledge types; and (3)from an engineer-
ing perspective, separating the two enables more
modular system design, facilitates ablation studies,
debugging, incremental updates, and future knowl-
edge extension.
A.4 Mahalanobis distance
Step 1: Compute the Covariance MatrixΣ
Σ =1
L−1LX
j=1(qw
j−¯zw)(qw
j−¯zw)T∈RD×D
Step 2: Compute the Mahalanobis Distance
(Activation Bias)ϕt
j
ϕw
j=q
(qw
j−¯zw)TΣ−1(qw
j−¯zw)∈R
Step 3: Construct the Probe-Query Vector
Qw
probe
Qw
probe=LX
j=1αw
j·qw
j,whereαw
j=ϕw
jPL
j=1ϕw
j
A.5 Comparison of K2K with different chunk
size
Figure 4 shows the K2K performance of different
chunk sizes on the dataset MIMIC-III Mortality.
We choose four chunk sizes: 16, 32, 64, and 128.
We observe that smaller chunk sizes (e.g., 16) lead
to higher F1 scores, indicating that finer granularity
benefits the identification of relevant knowledge
16 32 64 128
Chunk Size102030405060Score
F1
Jaccard
AUPRC
AUROCFigure 4: K2K performance with different chunk sizes.
5 10 15 20 25
Top-k Retrieved Knowledge204060F1
18.55
13.4217.44
14.1120.6162.8365.20 66.2469.3170.91F1 vs. Top-k
Mortality-MIMIC-III
Readmission-MIMIC-III
5 10 15 20 25
Top-k Retrieved Knowledge1020304050Jaccard
10.22
7.199.557.5911.4945.8048.3749.5253.0354.93Jaccard vs. Top-k
Mortality-MIMIC-III
Readmission-MIMIC-III
5 10 15 20 25
Top-k Retrieved Knowledge102030405060AUPRC
15.22
10.67 10.85 10.32 11.0155.6258.79 59.1862.49
55.50AUPRC vs. Top-k
Mortality-MIMIC-III
Readmission-MIMIC-III
5 10 15 20 25
Top-k Retrieved Knowledge505254565860AUROC
61.05
58.99
56.3859.00
56.88
50.3455.17
54.5856.64
51.35AUROC vs. Top-k
Mortality-MIMIC-III
Readmission-MIMIC-III
Figure 5: K2K Performance Across Different Top-k
Retrieved Knowledge Values on MIMIC-III.
segments. However, chunk size 64 achieves the
highest AUPRC and AUROC, suggesting it better
balances precision and recall for more robust clas-
sification. Larger chunk sizes may reduce retrieval
frequency but risk diluting critical signals. There-
fore, chunk size selection should consider both task
sensitivity and retrieval efficiency.
A.6 Ablation Study on Top-kRetrieval
Figures 5 and 6 demonstrate how the number of
retrieved knowledge entries (top-k) affects the per-
formance of K2K on both MIMIC-III and MIMIC-
IV datasets. For MIMIC-III, performance gener-
ally improves with increasing top-k, with the best
F1 (20.61) and Jaccard (11.49) observed at k=25
for the mortality task, while the readmission task
achieves optimal results at k=20–25. Notably, AU-
ROC and AUPRC peak at k=20, suggesting a bal-
ance between sufficient context and noise control.
In contrast, for MIMIC-IV , mortality prediction

5 10 15 20 25
Top-k Retrieved Knowledge0204060F1
4.642.216.17 6.615.2863.23 63.7561.51 61.87 60.79F1 vs. Top-k
Mortality-MIMIC-IV
Readmission-MIMIC-IV
5 10 15 20 25
Top-k Retrieved Knowledge010203040Jaccard
2.371.113.18 3.42 2.7146.23 46.79
44.41 44.7943.67Jaccard vs. Top-k
Mortality-MIMIC-IV
Readmission-MIMIC-IV
5 10 15 20 25
Top-k Retrieved Knowledge0204060AUPRC
2.32 2.45 2.50 2.93 2.4467.79 68.67 68.3866.9365.03AUPRC vs. Top-k
Mortality-MIMIC-IV
Readmission-MIMIC-IV
5 10 15 20 25
Top-k Retrieved Knowledge60626466AUROC
64.77
59.0163.0566.50
62.8566.2666.4766.62
64.94
63.15AUROC vs. Top-k
Mortality-MIMIC-IV
Readmission-MIMIC-IVFigure 6: K2K Performance Across Different Top-k
Retrieved Knowledge Values on MIMIC-IV .
Avg Retrieval Time
KARE 21.11 00:33:52
Prompt-based 22.52 3:26:00
K2K 22.89 0:0:5
Table 6: Performance comparison on the MIMIC-III
dataset in terms of averaged metrics (Avg), retrieval
time.
shows a performance peak at k=20 across all met-
rics, particularly for F1 and AUPRC, while read-
mission results are relatively stable across k values,
with the highest F1 (63.75) and AUPRC (68.67) at
k=10. However, large k values (e.g., k=25) tend
to hurt AUROC, especially for readmission. These
results indicate that task-specific tuning of top-k is
crucial, and that mortality prediction benefits more
from increasing top-k, while readmission may re-
quire a smaller, more focused knowledge set.
A.7 Retrieval Efficiency Comparison Across
different Retrieval methods
Compared to prior retrieval approaches, our K2K
method demonstrates substantially higher effi-
ciency. Specifically, KARE performs multi-stage
reasoning by first retrieving co-existing concepts
appear in each patient’s data and then computing
the shortest paths between the concepts and the
co-existing concepts over a large knowledge graph.
This results in a total complexity of O(k(|V|+
|E|)) , where kis the number of co-existing con-
cepts, and |V|,|E| are the number of nodes and
edges in the graph, respectively. Contriever, a
dense retriever, encodes the query and computes
similarities across the entire corpus, resulting ina time complexity of O(Nd) without approxima-
tion, where Nis the number of documents and
dis the embedding dimension. Prompt-based re-
trieval avoids external indexing but relies on LLM
generation conditioned on carefully designed in-
structions, which incurs substantial inference cost
atO(L·n2·h), where Lis the number of layers, n
is the token length, and his the number of attention
heads.
In contrast, K2K bypasses both external doc-
ument retrieval and graph traversal by directly
reusing the internal knowledge of the LLM. It re-
trieves relevant knowledge by comparing current
input representations with pre-trained FFN keys
and LoRA adapter keys from a specific transformer
layer. This enables fast memory access with a time
complexity of only O(m) orO(mk) (for top-k se-
lection), where mis the number of tokens. By
removing the need for external retrieval or gen-
eration, K2K achieves the fastest inference speed
among all retrievers while maintaining high accu-
racy, demonstrating the efficiency and practicality
of internal knowledge utilization.