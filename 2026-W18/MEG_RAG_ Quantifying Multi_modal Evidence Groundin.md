# MEG-RAG: Quantifying Multi-modal Evidence Grounding for Evidence Selection in RAG

**Authors**: Xihang Wang, Zihan Wang, Chengkai Huang, Quan Z. Sheng, Lina Yao

**Published**: 2026-04-27 14:51:00

**PDF URL**: [https://arxiv.org/pdf/2604.24564v2](https://arxiv.org/pdf/2604.24564v2)

## Abstract
Multimodal Retrieval-Augmented Generation (MRAG) addresses key limitations of Multimodal Large Language Models (MLLMs), such as hallucination and outdated knowledge. However, current MRAG systems struggle to distinguish whether retrieved multimodal data truly supports the semantic core of an answer or merely provides superficial relevance. Existing metrics often rely on heuristic position-based confidence, which fails to capture the informational density of multimodal entities. To address this, we propose Multi-modal Evidence Grounding (MEG), a semantic-aware metric that quantifies the contribution of retrieved evidence. Unlike standard confidence measures, MEG utilizes Semantic Certainty Anchoring, focusing on high-IDF information-bearing tokens that better capture the semantic core of the answer. Building on MEG, we introduce MEG-RAG, a framework that trains a multimodal reranker to align retrieved evidence with the semantic anchors of the ground truth. By prioritizing high-value content based on semantic grounding rather than token probability distributions, MEG-RAG improves the accuracy and multimodal consistency of generated outputs. Extensive experiments on the M$^2$RAG benchmark show that MEG-RAG consistently outperforms strong baselines and demonstrates robust generalization across different teacher models.

## Full Text


<!-- PDF content starts -->

MEG-RAG: Quantifying Multi-modal Evidence Grounding for
Evidence Selection in RAG
Xihang Wang∗
Zhejiang University
Hangzhou, China
xihangwang@163.comZihan Wang∗
Peking University
Beijing, China
1104976867@qq.comChengkai Huang†
University of New South Wales,
Macquarie University
Sydney, Australia
chengkai.huang1@unsw.edu.au
Quan Z. Sheng
Macquarie University
Sydney, Australia
michael.sheng@mq.edu.auLina Yao
University of New South Wales,
CSIRO’s Data61
Sydney, Australia
lina.yao@unsw.edu.au
Abstract
Multimodal Retrieval-Augmented Generation (MRAG) addresses
key limitations of Multimodal Large Language Models (MLLMs),
such as hallucination and outdated knowledge. However, current
MRAG systems struggle to distinguish whether retrieved multi-
modal data truly supports the semantic core of an answer or merely
provides superficial relevance. Existing metrics often rely on heuris-
tic position-based confidence, which fails to capture the informa-
tional density of multimodal entities. To address this, we propose
Multi-modal Evidence Grounding (MEG), a semantic-aware
metric that quantifies the contribution of retrieved evidence. Un-
like standard confidence measures, MEG utilizesSemantic Cer-
tainty Anchoring, which dynamically filters out high-frequency
stopwords via Inverse Document Frequency (IDF) to focus strictly
on information-bearing tokens. Building on MEG, we introduce
MEG-RAG, a framework that trains a multimodal reranker to align
retrieved evidence with the semantic anchors of the ground truth.
By prioritizing high-value content based on semantic grounding
rather than token probability distributions, MEG-RAG improves the
accuracy and multimodal consistency of generated outputs. Exten-
sive experiments on the 𝑀2RAG benchmark show that MEG-RAG
consistently outperforms strong baselines and demonstrates robust
generalization across different teacher models. The data and code
are available at here .
CCS Concepts
•Information systems→Information retrieval.
Keywords
Retrieval-Augmented Generation, Large Language Model, Multi-
modal RAG
∗These authors contributed equally.
†Corresponding author.
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3809947ACM Reference Format:
Xihang Wang, Zihan Wang, Chengkai Huang, Quan Z. Sheng, and Lina
Yao. 2026. MEG-RAG: Quantifying Multi-modal Evidence Grounding for
Evidence Selection in RAG. InProceedings of the 49th International ACM
SIGIR Conference on Research and Development in Information Retrieval
(SIGIR ’26), July 20–24, 2026, Melbourne, VIC, Australia.ACM, New York, NY,
USA, 6 pages. https://doi.org/10.1145/3805712.3809947
1 Introduction
In recent years, multimodal large language models (MLLMs) have
achieved rapid progress, demonstrating remarkable capabilities in
understanding and generating multimodal content such as text,
images, and audio [ 7,10,15,16,20]. However, MLLMs still suf-
fer from hallucination, outdated knowledge, and insufficient fac-
tual grounding [ 8,18]. To alleviate these issues, the Multimodal
Retrieval-Augmented Generation (MRAG) paradigm has emerged,
which retrieves external multimodal evidence (e.g., text snippets,
images, audio clips) to enhance the factuality and contextual ground-
ing of MLLMs’ outputs.
Despite the promise of MRAG, its effectiveness heavily depends
on the quality of the retrieved documents. Recent research high-
lights that irrelevant or contradictory information can mislead the
model and degrade generation quality [ 5,9,17]. However, exist-
ing multimodal retrievers and rerankers are ill-equipped to filter
out such documents, as they mainly rely on semantic similarity
rather than quantifying the information’s true contribution to accu-
rate answer generation. Several works have explored multimodal
rerankers to refine document selection [ 5,19], but these methods
primarily focus on improving relevance rather than measuring con-
tribution. Although benchmarks such as M2RAG [ 14] have been
proposed to standardize multimodal retrieval evaluation, they still
lack a principled mechanism to quantify the real informational
value of retrieved multimodal data [1, 16].
To address this limitation, we proposeMEG-RAG, a framework
for quantifyingMulti-modal Evidence Grounding (MEG)and
guiding evidence selection in MRAG. Specifically, MEG measures
the contribution of each retrieved multimodal sample by evaluating
the change in an MLLM’s semantic confidence with and without
that sample. We then train a multimodal reranker to predict MEG
scores, enabling effective re-ranking of multimodal evidence. ThisarXiv:2604.24564v2  [cs.CL]  30 Apr 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
reranker can be seamlessly integrated into various MRAG architec-
tures.
In this paper, our main contributions are threefold:
•We propose theMulti-modal Evidence Grounding (MEG),
which quantifies the contribution of each retrieved mul-
timodal sample by measuring the change in an MLLM’s
semantic confidence with and without that sample.
•Based on MEG, we developMEG-RAG, which leverages a
multi-task reranker that jointly optimizes binary discrim-
ination and pairwise ranking objectives, enabling precise
selection of high-value multimodal evidence.
•Experiments on the M2RAG benchmark, includingopen-
domain question answering,image captioning, andfact verifi-
cation, show that MEG-RAG consistently outperforms strong
baselines such as Jina Reranker, while achieving strong cross-
task generalization.
2 Methodology
In this section, we introduce the MEG-RAG to address the unique
challenges of evaluating multimodal documents by introducing
a new metric, Multi-modal Evidence Grounding (MEG). We then
detail a multi-task optimization strategy to train a reranker that
can effectively rank the multimodal documents through MEG pre-
diction.
2.1 Task Definition
We consider the task of MRAG, where a model generates an answer
gfor a queryxby leveraging a large-scale multimodal corpus C.
State-of-the-art MRAG frameworks employ a two-stageretrieve-
then-rerankpipeline to select the most relevant context. The end-
to-end process can be formally expressed as:
Dcand=Retriever(x,C),(1)
Dfinal=Reranker(x,D cand),(2)
g=MLLM(x,D final),(3)
First, an efficient initial retriever recalls a broad candidate set of
documentsDcand. Subsequently, a reranker model refines this set by
performing a fine-grained relevance assessment. It selects a small,
high-precision subset of documents Dfinal. This refined context
Dfinalis then prepended to the queryxand fed into a Multimodal
Large Language Model (MLLM) to produce the final answerg.
2.2 Multi-modal Evidence Grounding (MEG)
To train a high-performance reranker, we require a robust supervi-
sory signal that accurately reflects the semantic utility of a retrieved
document. To this end, we introduce a novel metric,Multi-modal
Evidence Grounding (MEG), designed to quantify the marginal
contribution of a document to the factual grounding of a correct
answer. MEG measures the change in a teacher MLLM’s semantic
confidence when a given document is added to the context. We
present our automated MEG data collection pipeline below.
2.2.1 MEG Definition.The core of our framework is to quantify
the grounding contribution of a retrievedmultimodal document.
Formally, given an MLLM 𝜙, a multimodal queryx, and its corre-
sponding ground truth textual answery, the Multi-modal EvidenceGrounding (MEG) for a retrieved documentd 𝑖is defined as:
MEG(d 𝑖|x)𝑑𝑒𝑓=𝑆 𝜙(y|x,d 𝑖)−𝑆 𝜙(y|x),(4)
where𝑆𝜙(y|·)represents the MLLM’s semantic confidence in gener-
ating the correct answerybased on its core informational content.
By measuring the change in semantic confidence, MEG captures
the marginal utility of each retrieved document and provides a rich,
continuous supervisory signal. A high MEG score indicates that
the document offers critical grounding for the entities, relations,
or actions in the answer; a near-zero score suggests that the docu-
ment is largely redundant with respect to the semantic core; and a
negative score indicates that the document introduces distracting
or contradictory information that reduces grounding confidence.
2.2.2 Semantic Certainty Anchoring.A key challenge is reliably
estimating the MLLM’s confidence 𝑆𝜙(y|·)without being biased
by syntactic filler tokens (e.g., "the", "is", "of") in the ground truth
answer. We proposeSemantic Certainty Anchoring. This mech-
anism dynamically identifies information-bearing tokens to serve
as anchors for confidence calculation based on their statistical sig-
nificance.
Specifically, for a ground truth answery =(𝑡 1,𝑡2,...,𝑡 𝐿), we
apply a semantic mask M∈{ 0,1}𝐿. Instead of relying on external
linguistic tools, we employ a frequency-based filtering strategy.
We calculate the inverse document frequency (IDF) for each word
𝑡𝑖across the corpus. Words with high frequency (low IDF) are
typically stopwords that contribute little to semantic distinction.
The mask is defined as:
M𝑖=(
1if IDF(𝑡 𝑖)>𝜏 freq
0otherwise(5)
where𝜏freqis a threshold determining the cut-off for high-frequency
words.M𝑖is set to 1 for rare, information-rich tokens and 0 for com-
mon, low-information tokens. The semantic confidence is defined
as the average log-probability over these anchors:
𝑆𝜙(y|·)def=1
Í𝐿
𝑖=1M𝑖𝐿∑︁
𝑖=1M𝑖·log𝑝 𝜙(𝑡𝑖|𝑡<𝑖,·)(6)
This data-driven approach ensures that the supervision signal fo-
cuses on the semantic essence of the multimodal task rather than
heuristic positional biases or linguistic priors. Notably, our MEG
formulation relies on the difference of log-probabilities rather than
pure probability subtraction or Kullback-Leibler (KL) divergence.
From an information-theoretic perspective, this log-probability
difference acts as an estimator of Pointwise Mutual Information
(PMI) [ 6], robustly capturing the information gain provided by
the evidence while avoiding numerical underflow. Furthermore,
recent empirical studies demonstrate that log-probabilities serve
as a highly reliable metric for evaluating semantic plausibility and
world knowledge in both base and instruction-tuned Large Lan-
guage Models [ 12]. Finally, unlike KL divergence, which requires
computing and storing the full vocabulary distribution at each
generation step, our approach only requires the probability of the
ground-truth semantic anchors, making it highly computationally
efficient for MLLMs. We collect a rich training dataset characterized
by the MEG of multimodal documents, providing a foundation for
reranker training.

MEG-RAG: Quantifying Multi-modal Evidence Grounding for Evidence Selection in RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
Figure 1: Overview of the MEG-RAG framework. (1) Dataset Construction computes MEG scores to quantify the utility of
multimodal documents. (2) Reranker Training learns a multimodal reranker using the constructed MEG dataset with a hybrid
objective combining document relevance classification and pairwise ranking. (3) MEG-RAG Inference retrieves candidate
documents, reranks them to select the top-𝐾results, and generates the final answer based on the selected evidence.
2.3 Multi-task Reranker Optimization
Building on the MEG-scored dataset, we train a multimodal reranker
𝑓𝜃to predict document utility via a hybrid-loss scheme that incor-
porates both absolute relevance (classification) and relative utility
(ranking) objectives. Our framework is model-agnostic and allows
for the utilization of any Transformer-based multimodal encoder
as the backbone. The reranker takes the query-document pair as
input and outputs a scalar utility score.
2.3.1 Document Relevance Classification.To enable the reranker
to learn the absolute relevance between the document and query,
we frame the task as a binary classification problem. We employ a
standard Cross-Entropy (CE) loss:
min
𝜃𝐿CE=1
𝑁𝑁∑︁
𝑖=1h
−𝑦𝑖log(𝑝 𝑖)−(1−𝑦 𝑖)log(1−𝑝 𝑖)i
, (7)
where𝑝 𝑖=𝑓𝜃(x𝑖,d𝑖)is the probability that documentd 𝑖is helpful
for queryx 𝑖. The binary label 𝑦𝑖is determined using two thresholds,
𝑏1and𝑏2, on the MEG score of the document: documents with
MEG>𝑏 1are positive ( 𝑦𝑖=1), while those with MEG<𝑏 2are
negative (𝑦 𝑖=0).2.3.2 Pairwise Ranking Optimization.While CE loss assesses abso-
lute relevance, effective RAG requires a nuanced relative ordering
of documents. To achieve this, we incorporate a pairwise ranking
objective based on RankNet loss [ 4]. For a given queryx, we sample
pairs of positive(d𝑖)and negative(d𝑗)documents based on their
MEG scores. The reranker 𝑓𝜃produces scalar scores 𝑠𝑖=𝑓𝜃(x,d𝑖)
and𝑠𝑗=𝑓𝜃(x,d𝑗). The RankNet loss minimizes the cross-entropy
of the "correct" ranking probability:
min
𝜃𝐿RankNet =∑︁
(x,d𝑖,d𝑗)∈Tlog 1+exp −𝜎(𝑠 𝑖−𝑠𝑗),(8)
whereTis the set of all training pairs (d𝑖,d𝑗)where MEG( d𝑖|x)>
MEG( d𝑗|x), and𝜎is a scaling factor that controls the steepness of
the logistic function. This loss directly encourages the reranker to
assign a higher score (𝑠 𝑖) to the document with higher MEG.
2.3.3 Final Training Objective.The complete training objective
for our multimodal reranker combines both the classification and
ranking losses, balanced by a hyperparameter𝛼:
𝐿hybrid =𝛼·𝐿 CE+(1−𝛼)·𝐿 RankNet .(9)

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
This dual capability is critical for selecting the optimal top-k multi-
modal evidence to pass to the MLLM, thereby improving the final
answer’s quality and factual grounding.
3 Experiments
In this section, we conduct experiments to validate the effectiveness
of our proposed MEG-RAG framework.
Tasks and Datasets.We evaluate MEG-RAG on three multi-
modal tasks from the M2RAG benchmark [ 14]: (1)MultiModal
QA(MMQA), an open-domain task requiring the MLLM to synthe-
size information from retrieved multimodal documents to answer
questions; (2)Image Captioning, where the MLLM generates a
descriptive caption for an image, using retrieved documents as con-
text; and (3)Fact Verification, which requires the model to verify
the truthfulness of a multimodal claim based on retrieved evidence.
Models and Metrics.Our evaluations are conducted on both
general MLLMs such asLLaVA-NeXT[ 13],Qwen2.5-VL[ 3] as
well asMM-RAIT-Qwen2.5-VL, a variant of Qwen2.5-VL specif-
ically fine-tuned for MRAG tasks [ 14]. We follow the M2RAG’s
standard metrics: BLEU, ROUGE, and CIDEr for MMQA and Image
Captioning tasks, and Acc and F1 for Fact Verification.
Baselines.We compare our method with three baselines: (1)
None-MRAG, (2) Naive-MRAG, where top- 𝑘retrieved documents
are directly fed to the MLLM; (3) Ranking-MRAG, where a reranker
is used to reorder retrieved documents such as Jina Reranker [11].
Implementation Details.We constructed the training dataset
based on the MMQA task. For the proposed Semantic Certainty
Anchoring, we set the frequency threshold 𝜏freq=0.15to exclude
the redundant words in the ground truth answer, identifying the
remaining words as semantic anchors. This threshold was empiri-
cally determined to maximize grounding quality on the validation
set. Using the calculated MEG scores, we labeled <query, answer,
document>triplets.
To ensure robust training and precise discrimination of evidence,
we established a rigorous data filtering pipeline. An analysis of the
initially generated raw triplets revealed a heavily skewed distribu-
tion: approximately6 .5%were useful samples ( MEG> 0.2, provid-
ing strong semantic grounding),12 .9%were misleading samples
(MEG<− 0.2, causing significant interference), and the vast major-
ity,80.6%, were irrelevant or neutral samples ( −0.2≤MEG≤ 0.2).
To prevent the model from being overwhelmed by noise and biased
toward neutral contexts, all irrelevant samples falling within the
[−0.2,0.2]interval were directly discarded. Finally, to address the
imbalance between the remaining valid positive and negative pairs,
we sampled exactly 31K positive triplets and 31K negative triplets.
This yielded a strictly balanced final training dataset containing
62K triplets (a perfect 1:1 ratio).
We instantiated our reranker using the Jina-m0 architecture [ 11].
The model was fine-tuned on four NVIDIA A100 (80GB) GPUs. We
used the AdamW optimizer with a learning rate of 2e-5 and a batch
size of 64 per GPU. The multi-task objective (9) was optimized
with the trade-off parameter 𝛼=0.74. The training was conducted
for 3 epochs with a linear warmup of the first 10% steps. During
inference, the reranker selects the top- 𝑘(𝑘∈{ 1,3}) documents for
generation.3.1 Main Result
As shown in Table 1, MEG-RAG consistently delivers significant
improvements across a wide range of tasks and MLLMs. On the MM-
RAIT-Qwen2.5-VL, our method achieves a top CIDEr score of 260.78
on the MMQA task under the top-1 setting, representing a relative
improvement of 7.0% over the Jina-reranker baseline (243.65). For
the general MLLM LLaVA-NeXT, MEG-RAG attains a CIDEr score
of 278.43 with the top-3 setting, marking a substantial 7.3% gain
compared to the Jina-reranker (259.41). Furthermore, MEG-RAG
delivers consistent gains on Image Captioning and Fact Verification.
For Image Captioning, MEG-RAG (top-3) with Qwen2.5-VL achieves
a CIDEr score of 73.66, substantially outperforming Jina Reranker
(45.73) with a relative improvement of61.1%. On Fact Verification,
it attains an F1 of 73.75, surpassing the baseline (65.96) by11.8%.
Crucially, since our reranker is trained solely on the MMQA dataset,
these cross-task gains strongly validate its robust generalization
capability.
3.2 Ablation Study
To validate the effectiveness of our design, we conduct two key
ablation studies.
Impact of Loss Function.We compare our hybrid-loss model
against the single-loss model in Table 2. The Hybrid-Loss model
consistently outperforms the single-loss variants across all MLLMs
and tasks. ForLLaVA-NeXTon the MMQA task, the Hybrid-Loss
method yielded a BLEU-4 of27.71, which exceeds the RankNet loss
(26.33) by5.2%and the CE loss (26.45) by4 .8%. This confirms that
the two loss components capture complementary learning signals,
namely absolute relevance (CE) and relative ordering (RankNet),
and that their synergistic combination is essential for achieving
optimal reranker performance.
Impact of Teacher Model Selection.A critical concern in
distillation-based approaches is whether the student model (reranker)
learns generalizable information utility or merely mimics the biases
of a specific teacher. To investigate this, we compared our default
teacher (Qwen2.5-VL) with Llama-3.2-11B-Vision [ 2], a model from
a different architectural family. As illustrated in Figure 2, the MEG-
RAG is remarkably robust to the choice of teacher model. In all
evaluated MLLMs and tasks, both teacher variants consistently and
significantly outperform the baseline. For instance, on the MMQA
task, the Llama-teacher variant (264.92 CIDEr) achieves perfor-
mance almost identical to the Qwen-teacher variant (264.78 CIDEr),
both surpassing the baseline by over 16 points. This consistent
improvement indicates that MEG primarily models evidence utility,
rather than merely reflecting the bias of a specific teacher model.
3.3 Sensitivity Analysis
Sensitivity of Loss Weight 𝛼.The hyperparameter 𝛼in Equation
(9) balances the CE loss and RankNet loss. To evaluate its impact,
we conducted a grid search for 𝛼∈{ 0.0,0.25,0.5,0.74,1.0}and
plotted the CIDEr performance of MEG-RAG (Top-3) with MM-
RAIT-Qwen2.5-VL on the validation set. As illustrated in Figure
3(a), relying solely on the RankNet loss ( 𝛼= 0.0) or the CE loss
(𝛼=1.0) yields sub-optimal results. The performance peaks around
𝛼= 0.74, demonstrating that a properly balanced hybrid loss is
critical for optimizing the reranker.

MEG-RAG: Quantifying Multi-modal Evidence Grounding for Evidence Selection in RAG SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
Table 1: Main comparison of MEG-RAG against baselines on the 𝑀2𝑅𝐴𝐺 benchmark tasks across multiple MLLMs. (B-N = Bleu-N,
R-N = Rouge-N, R-L = Rouge-L). Bold indicates the best performance, and underline indicates the second-best performance.
Model MethodMultimodal QA Image Captioning Fact Verification
B-1 B-2 B-3 B-4 R-1 R-2 R-L CIDEr B-1 B-2 B-3 B-4 R-1 R-2 R-L CIDEr ACC F1
LLaVA-NeXTNone-MRAG 34.85 30.00 27.04 24.49 48.36 37.00 45.61 250.82 6.42 3.08 2.02 1.63 19.17 5.34 17.33 18.98 39 35.76
Naive-MRAG-top1 35.73 30.77 27.51 24.86 49.17 37.44 46.28 251.26 8.99 5.32 3.78 2.82 22.17 8.43 20.01 40.34 50 33.18
Jina-Reranker-top1 35.94 30.97 27.73 25.10 49.47 37.40 46.97 254.57 9.55 5.65 3.91 2.80 26.26 9.94 23.44 42.08 52 34.93
MEG-RAG-top1 37.28 32.41 29.24 26.69 51.41 40.11 49.19 268.74 12.73 8.45 6.33 4.63 30.01 13.73 26.75 65.59 58 41.13
Naive-MRAG-top3 35.73 30.81 27.65 24.90 49.81 37.83 46.93 250.52 9.30 5.52 3.85 2.84 23.99 9.74 21.87 35.48 47 31.14
Jina-Reranker-top3 36.19 31.37 28.32 25.78 50.08 38.37 47.44 259.41 10.05 5.98 4.10 3.02 25.79 10.68 22.91 44.60 50 34.26
MEG-RAG-top3 37.89 33.16 30.21 27.71 52.66 41.01 50.22 278.43 13.17 8.18 5.95 4.43 29.85 12.84 25.96 57.78 55 38.25
Qwen2.5-VLNone-MRAG 25.41 20.64 17.61 14.91 39.30 27.78 34.97 105.33 8.97 4.46 2.84 2.03 20.05 6.64 18.20 19.51 51 50.67
Naive-MRAG-top1 30.04 25.15 22.16 19.90 43.43 32.11 40.43 183.07 11.63 6.10 3.58 2.53 25.21 8.68 22.64 40.29 56 56.63
Jina-Reranker-top1 30.76 25.73 22.51 19.70 45.23 33.56 41.39 179.37 11.90 6.86 4.41 3.08 25.10 9.39 22.68 43.27 58 59.23
MEG-RAG-top1 32.50 27.49 24.40 21.85 47.85 35.22 44.17 211.24 14.71 8.21 5.68 4.12 29.19 13.86 26.62 50.00 62 63.74
Naive-MRAG-top3 30.67 25.45 22.14 19.55 45.04 32.63 41.42 185.94 11.75 6.92 4.93 3.70 26.99 10.77 24.55 46.64 60 62.26
Jina-Reranker-top3 29.89 25.19 22.16 19.68 45.41 33.50 41.39 174.55 12.62 7.16 4.56 3.49 27.82 11.86 25.43 45.73 63 65.96
MEG-RAG-top3 33.37 28.11 24.79 22.02 47.94 36.03 44.73 218.62 15.31 9.77 7.10 5.58 31.24 16.29 28.50 73.66 70 73.75
MM-RAIT-
Qwen2.5-VLNone-MRAG 33.88 27.39 23.15 20.20 48.31 30.52 46.25 206.42 16.18 10.46 7.56 5.68 34.16 15.43 32.58 79.41 59 61.15
Naive-MRAG-top1 38.12 29.88 24.57 21.56 56.64 33.07 53.81 236.08 16.98 11.35 8.44 6.44 35.75 16.31 33.81 83.62 60 62.44
Jina-Reranker-top1 39.16 30.79 25.33 22.34 58.29 33.35 55.70 243.65 18.66 12.77 9.57 7.18 35.84 16.82 33.81 99.06 62 63.65
MEG-RAG-top1 41.84 33.26 27.43 24.00 61.34 36.70 58.86 260.78 22.82 17.14 13.32 10.95 42.45 23.02 40.49 134.05 66 67.43
Naive-MRAG-top3 40.67 31.39 26.08 23.07 56.56 33.37 54.09 247.13 17.15 11.41 7.90 5.89 36.09 16.89 34.57 84.41 62 62.95
Jina-Reranker-top3 40.46 31.61 26.29 23.41 56.34 34.17 54.05 248.63 19.39 13.05 9.47 7.42 38.05 18.31 35.74 96.66 63 64.31
MEG-RAG-top3 42.95 33.38 27.60 24.14 61.24 35.89 57.91 264.78 23.47 17.46 13.74 11.36 42.91 24.10 41.24 136.44 68 69.80
Table 2: Ablation study of loss functions (top-3).
Model Method (Loss)Multimodal QA Image Captioning Fact Verification
B-4 R-L CIDEr B-4 R-L CIDEr ACC F1
LLaVA-NeXTCE Loss 26.45 49.43 266.80 3.72 24.21 50.92 52 34.87
RankNet Loss 26.33 49.36 266.61 3.41 23.68 45.41 51 34.52
Hybrid-Loss 27.71 50.22 278.43 4.43 25.96 57.78 55 38.25
MM-RAIT-
Qwen2.5-VLCE Loss 23.37 56.78 257.34 9.74 37.59 122.12 65 67.41
RankNet Loss 23.00 56.04 254.93 8.53 38.04 106.55 64 65.01
Hybrid-Loss 24.14 57.91 264.78 11.36 41.24 136.44 68 69.80
Figure 2: Ablation study on Teacher Model Selection.
Sensitivity of MEG Threshold 𝜏.We further analyzed the sen-
sitivity of the positive/negative labeling threshold 𝜏used during
dataset construction. We tested varying the MEG boundary from
0.1 to 0.5 (i.e., assigning positive labels for MEG>𝜏 and negative
forMEG<−𝜏 ). As shown in Figure 3(b), a low threshold ( 𝜏=0.1)
introduces excessive noise from negligible documents, degrading
performance. Conversely, an overly strict threshold ( 𝜏≥0.4) se-
verely limits the number of available training samples, leading tooverfitting. The threshold of 𝜏=0.2provides the optimal balance
between dataset scale and signal purity.
Figure 3: Sensitivity analysis of (a) the loss weight 𝛼and (b)
the MEG labeling threshold𝜏.
4 Conclusion
In this paper, we present MEG-RAG, a unified framework for ev-
idence selection in Multimodal Retrieval-Augmented Generation.
MEG-RAG introduces Multi-modal Evidence Grounding (MEG), a
principled metric that quantifies the contribution of retrieved mul-
timodal evidence through a stabilized computation process. Based
on MEG, we develop a hybrid-loss training strategy that enables a
reranker to effectively distinguish and prioritize high-value multi-
modal documents. Experiments on the 𝑀2RAG benchmark show
that MEG-RAG consistently outperforms standard MRAG and exist-
ing reranking methods. Moreover, MEG-RAG is model-agnostic and
generalizes well across various multimodal tasks, demonstrating
its robustness and transferability in multimodal evidence selection.

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Xihang et al.
References
[1] Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Moham-
madali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soley-
mani Baghshah, and Ehsaneddin Asgari. 2025. Ask in any modality: A compre-
hensive survey on multimodal retrieval-augmented generation.arXiv preprint
arXiv:2502.08826(2025).
[2]Meta AI. 2024. Llama 3.2: Revolutionizing edge ai and vision with open, cus-
tomizable models.Meta AI Blog.20 (2024), 2024.
[3]Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai
Dang, Peng Wang, Shijie Wang, Jun Tang, et al .2025. Qwen2. 5-VL Technical
Report.eprint arXiv: 2502.13923(2025).
[4]Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton,
and Greg Hullender. 2005. Learning to rank using gradient descent. InProceedings
of the 22nd international conference on Machine learning. 89–96.
[5]Zhanpeng Chen, Chengjin Xu, Yiyan Qi, and Jian Guo. 2024. MLLM Is
a Strong Reranker: Advancing Multimodal Retrieval-augmented Generation
via Knowledge-enhanced Reranking and Noise-injected Training.CoRR
abs/2407.21439 (2024).
[6]Kenneth Church and Patrick Hanks. 1990. Word association norms, mutual
information, and lexicography.Computational linguistics16, 1 (1990), 22–29.
[7]Chengkai Huang, Junda Wu, Yu Xia, Zixu Yu, Ruhan Wang, Tong Yu, Ruiyi Zhang,
Ryan A Rossi, Branislav Kveton, Dongruo Zhou, et al .2025. Towards agentic
recommender systems in the era of multimodal large language models.arXiv
preprint arXiv:2503.16734(2025).
[8]Chengkai Huang, Yu Xia, Rui Wang, Kaige Xie, Tong Yu, Julian McAuley, and
Lina Yao. 2025. Embedding-informed adaptive retrieval-augmented generation
of large language models. InProceedings of the 31st International Conference on
Computational Linguistics. 1403–1412.
[9]Shuguang Jiao, Chengkai Huang, Shuhan Qi, Xuan Wang, Yifan Li, and Lina
Yao. 2026. Doctor-RAG: Failure-Aware Repair for Agentic Retrieval-Augmented
Generation.arXiv preprint arXiv:2604.00865(2026).
[10] Shuguang Jiao, Xinyu Xiao, Yunfan Wei, Shuhan Qi, Chengkai Huang, Quan Z
Sheng, and Lina Yao. 2026. PruneRAG: Confidence-Guided Query DecompositionTrees for Efficient Retrieval-Augmented Generation. InProceedings of the ACM
Web Conference 2026. 1923–1934.
[11] Jina AI. 2025. Jina Reranker M0: Multilingual & Multimodal Document Reranker.
[12] Carina Kauf, Emmanuele Chersoni, Alessandro Lenci, Evelina Fedorenko, and
Anna A Ivanova. 2024. Log probabilities are a reliable estimate of semantic
plausibility in base and instruction-tuned language models. InProceedings of the
7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP.
263–277.
[13] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma,
and Chunyuan Li. 2024. Llava-next-interleave: Tackling multi-image, video, and
3d in large multimodal models.arXiv preprint arXiv:2407.07895(2024).
[14] Zhenghao Liu, Xingsheng Zhu, Tianshuo Zhou, Xinyi Zhang, Xiaoyuan Yi, Yukun
Yan, Ge Yu, and Maosong Sun. 2025. Benchmarking retrieval-augmented gen-
eration in multi-modal contexts. InProceedings of the 33rd ACM International
Conference on Multimedia. 4817–4826.
[15] Haowei Lou, Chengkai Huang, Hye-young Paik, Yongquan Hu, Aaron Quigley,
Wen Hu, and Lina Yao. 2025. SpeechAgent: An End-to-End Mobile Infrastructure
for Speech Impairment Assistance.arXiv preprint arXiv:2510.20113(2025).
[16] Lang Mei, Siyu Mo, Zhihan Yang, and Chong Chen. 2025. A survey of multimodal
retrieval-augmented generation.arXiv preprint arXiv:2504.08748(2025).
[17] Matin Mortaheb, Mohammad A Amir Khojastepour, Srimat T Chakradhar, and
Sennur Ulukus. 2025. Re-ranking the context for multimodal retrieval augmented
generation.arXiv preprint arXiv:2501.04695(2025).
[18] Chuhan Wang, Xintong Li, Jennifer Yuntong Zhang, Junda Wu, Chengkai Huang,
Lina Yao, Julian McAuley, and Jingbo Shang. 2026. SceneAlign: Aligning Mul-
timodal Reasoning to Scene Graphs in Complex Visual Scenes.arXiv preprint
arXiv:2601.05600(2026).
[19] Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin
Zhong, and Hengxing Cai. 2025. MM-R5: MultiModal Reasoning-Enhanced
ReRanker via Reinforcement Learning for Document Retrieval.arXiv preprint
arXiv:2506.12364(2025).
[20] Juexiang Ye, Xue Li, Xinyu Yang, Chengkai Huang, Lanshun Nie, Lina Yao, and
Dechen Zhan. 2026. MemWeaver: Weaving Hybrid Memories for Traceable
Long-Horizon Agentic Reasoning.arXiv preprint arXiv:2601.18204(2026).