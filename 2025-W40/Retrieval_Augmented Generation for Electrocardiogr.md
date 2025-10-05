# Retrieval-Augmented Generation for Electrocardiogram-Language Models

**Authors**: Xiaoyu Song, William Han, Tony Chen, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao

**Published**: 2025-09-30 20:32:34

**PDF URL**: [http://arxiv.org/pdf/2510.00261v1](http://arxiv.org/pdf/2510.00261v1)

## Abstract
Interest in generative Electrocardiogram-Language Models (ELMs) is growing,
as they can produce textual responses conditioned on ECG signals and textual
queries. Unlike traditional classifiers that output label probabilities, ELMs
are more versatile, supporting domain-specific tasks (e.g., waveform analysis,
diagnosis, prognosis) as well as general tasks (e.g., open-ended questions,
dialogue). Retrieval-Augmented Generation (RAG), widely used in Large Language
Models (LLMs) to ground LLM outputs in retrieved knowledge, helps reduce
hallucinations and improve natural language generation (NLG). However, despite
its promise, no open-source implementation or systematic study of RAG pipeline
design for ELMs currently exists. To address this gap, we present the first
open-source RAG pipeline for ELMs, along with baselines and ablation studies
for NLG. Experiments on three public datasets show that ELMs with RAG
consistently improves performance over non-RAG baselines and highlights key ELM
design considerations. Our code is available at:
https://github.com/willxxy/ECG-Bench.

## Full Text


<!-- PDF content starts -->

RETRIEVAL-AUGMENTED GENERATION FOR
ELECTROCARDIOGRAM-LANGUAGE MODELS
Xiaoyu Song∗1, William Han∗1, Tony Chen2, Chaojing Duan3,
Michael A. Rosenberg4, Emerson Liu3, Ding Zhao1
1Carnegie Mellon University,2Columbia University,
3Allegheny Health Network,4University of Colorado
ABSTRACT
Interest in generative Electrocardiogram-Language Models
(ELMs) is growing, as they can produce textual responses
conditioned on ECG signals and textual queries. Unlike tra-
ditional classifiers that output label probabilities, ELMs are
more versatile, supporting domain-specific tasks (e.g., wave-
form analysis, diagnosis, prognosis) as well as general tasks
(e.g., open-ended questions, dialogue). Retrieval-Augmented
Generation (RAG), widely used in Large Language Models
(LLMs) to ground LLM outputs in retrieved knowledge, helps
reduce hallucinations and improve natural language genera-
tion (NLG). However, despite its promise, no open-source
implementation or systematic study of RAG pipeline design
for ELMs currently exists. To address this gap, we present
the first open-source RAG pipeline for ELMs, along with
baselines and ablation studies for NLG. Experiments on three
public datasets show that ELMs with RAG consistently im-
proves performance over non-RAG baselines and highlights
key ELM design considerations. Our code is available at:
https://github.com/willxxy/ECG-Bench.
Index Terms—Electrocardiograms, Retrieval-Augmented
Generation, Large Language Models, Multimodality, Natural
Language Processing
1. INTRODUCTION
Cardiovascular diseases (CVDs) are the leading global cause
of death, responsible for about 18 million deaths annually
[1]. Electrocardiograms (ECGs) are noninvasive, widely
available, and central to early CVD detection. However,
ECGs still require expert clinicians for accurate interpreta-
tions. This is a growing challenge given rising screening
demands and a nationwide shortage of cardiac specialists,
particularly in under-served regions [2]. To address this gap,
* Equal contribution
© 2026 IEEE. Personal use of this material is permitted. Permission
from IEEE must be obtained for all other uses, in any current or future media,
including reprinting/republishing this material for advertising or promotional
purposes, creating new collective works, for resale or redistribution to servers
or lists, or reuse of any copyrighted component of this work in other works.deep learning has been applied to automate ECG-based diag-
nosis centering around classification tasks [3]. More recently,
large language model (LLM)-based generative approaches,
termed Electrocardiogram-Language Models (ELMs), have
emerged. Unlike traditional classification models that pro-
vide only probability scores for fixed labels, ELMs incorpo-
rate knowledge from large-scale pretraining on internet data
and can be adapted to output not just CVD labels but also
textual explanations that justify the diagnostic decisions [4].
This expanded capability reduces the burden on cardiac elec-
trophysiologists, who are often tasked with examining vast
volumes of ECG data, and enables clinicians without special-
ized training to interact with sophisticated ECG interpretation
tools.
In natural language processing (NLP), Retrieval-Augmented
Generation (RAG) improves large language model (LLM)
outputs by retrieving relevant external documents. This re-
duces hallucinations and enables more accurate, context-
aware generation beyond the model’s parametric memory
[5].
Applying RAG to ECG deep learning has largely focused
on classification and retrieval tasks [6]. ECG-Chat [4] is
among the first to extend RAG to free-form NLG for ELMs,
but it (1) does not provide an open-source RAG pipeline
and (2) does not analyze how RAG design choices affect
performance. Another related work, Q-Heart [7], incorpo-
rates RAG content during instruction tuning to update input
prompts dynamically. However, this work currently lacks an
open-source implementation, hindering direct comparison,
and it treats RAG primarily as a tool for boosting question
answering without exploring design variations. Similarly,
[8] applies RAG with LLMs for ECG diagnosis, but their
approach (1) lacks an open-source implementation, (2) en-
codes handcrafted ECG features (e.g., heart rate variability,
QRS duration) as text instead of directly using ECG signals,
and (3) does not evaluate free-form NLG, despite producing
textual explanations.
To address these gaps, we present the first open-source
RAG framework for training and inference in ELMs tailored
to NLG. Additionally, we comprehensively evaluate the per-arXiv:2510.00261v1  [cs.CL]  30 Sep 2025

Fig. 1. Our RAG pipeline operates as follows: given an in-
put ECG, we optionally extract features and query a RAG
database of ECG signals, features, and diagnostic reports.
We retrieve the top-k similar diagnostic reports, construct a
prompt (system prompt + retrieved diagnostic reports + ECG
tokens + query), and use it to condition the ELM to generate
the response.
formance of including RAG during NLG across three publicly
available datasets and by ablating various components in the
pipeline. In summary, our contributions are the following:
1. To the best of our knowledge, we are the first to develop
an open source framework for RAG in NLG for ELMs.
2. We conduct baselines on three public datasets and demon-
strate strong performance gains when utilizing RAG across
different ELM architectures.
3. To inform researchers about RAG pipeline design trade-
offs, we conduct a comprehensive ablation study, varying
training–inference combinations with RAG, the number of
retrieved items k, the placement of retrieved content, and the
effect of noise injection in the retrieved content.
Our open-source implementation allows for direct com-
parisons, and the comprehensive ablation study on integrat-
ing RAG into ELMs provides valuable insights for future re-
searchers.
2. METHODS
2.1. Datasets and Preprocessing
This study leverages adapted versions of the MIMIC-IV-
ECG [9] and PTB-XL [10] datasets for NLG tasks. We
incorporate the ECG-Chat Instruct dataset curated by [4],
which includes single- and multi-turn conversational data
paired with a corresponding ECG. In addition, we utilize theECG-QA dataset [11], which contains ChatGPT-generated,
clinically relevant question-answer pairs derived from both
MIMIC-IV-ECG and PTB-XL.
We apply a unified preprocessing pipeline to all ECG sig-
nals, following ECG-Byte [12]. Signals are first standardized
to the PTB-XL lead configuration [I, II, III, aVL, aVR, aVF,
V1–V6]. Powerline noise at 50/60 Hz is removed with bidi-
rectional notch filters (Q=30), and clinically relevant compo-
nents are preserved using a fourth-order Butterworth band-
pass filter (0.5–100 Hz). Baseline drift is corrected with a
bidirectional high-pass filter (0.05 Hz). For denoising, we ap-
ply wavelet decomposition (Daubechies-6, level 4) with soft
thresholding based on the median absolute deviation of detail
coefficients. All signals are resampled to 250 Hz and seg-
mented into non-overlapping 5-second windows.
2.2. Retrieval-Augmented Generation Database
We curate a domain-specific RAG database composed of
ECG signal segments, derived features, and corresponding
diagnostic reports. This database is designed to support mul-
timodal similarity search using both ECG signals and ECG
features.
We extract features from the time, frequency, and time–
frequency domains for each lead of an ECG signal. The time-
domain features include maximum, minimum, heart rate,
heart rate variability, QRS duration, T-wave amplitude, ST
deviation, average absolute difference, and root mean square
difference. The frequency-domain features include total
power, peak frequency power, dominant frequency, and spec-
tral centroid. The time–frequency features include wavelet
coefficient approximation and detail levels (up to level 5).
These features are used to construct the RAG database. The
same set of features are derived from the input ECG and
computed on the fly during training and inference to serve as
the query to the RAG database.
We use the FAISS library’s [13] IndexIVFFlat structure
for efficient vector indexing. For each ECG signal and/or fea-
ture query, we search the corresponding index and retrieve
the top-knearest neighbors based on L2 distance. Each ECG
signal and feature index is linked to its associated diagnostic
report, allowing us to also retrieve the top-knearest diagnos-
tic reports. In our study, we query the RAG database utiliz-
ing both ECG features and signals unless specified otherwise.
During training or inference we query the RAG database and
insert the retrieved content (i.e., top-k diagnostic reports) in
the system prompt on the fly, as shown in Figure 1.
2.3. Electrocardiogram-Language Model
Several variants of ELMs have been explored [14], but
we mainly adopt ECG-Byte [12] due to its simplicity and
low computational overhead. Unlike neural network en-
coder–based ELMs, ECG-Byte does not compress the ECG

Table 1. Mean baseline comparisons over 3 random seeds across three datasets.
Dataset Method BLEU-4↑(%) ROUGE-L↑(%) METEOR↑(%) BERTScore F1↑(%) Accuracy↑(%)
ECG-Chat Instruct [4]ECG-Byte [12] 22.85 ± 0.18 66.82 ± 0.17 58.97 ± 0.16 96.24 ± 0.02 9.65 ± 0.29
ECG-Byte with RAG38.10 ± 0.05 75.61 ± 0.05 69.85 ± 0.02 97.49 ± 0.01 18.27 ± 0.05
ECG-QA PTB-XL [11]ECG-Byte [12] 21.07 ± 0.0548.47 ± 0.02 33.66 ± 0.0796.37 ± 0.00 37.03 ± 0.04
ECG-Byte with RAG21.46 ± 0.1348.00 ± 0.03 32.20 ± 0.0296.44 ± 0.01 38.30 ± 0.06
ECG-QA MIMIC-IV ECG [11]ECG-Byte [12] 15.52 ± 0.03 39.75 ± 0.11 26.62 ± 0.08 95.28 ± 0.01 27.21 ± 0.12
ECG-Byte with RAG49.07 ± 0.14 76.32 ± 0.00 59.51 ± 0.06 97.98 ± 0.00 57.77 ± 0.12
Fig. 2. Our RAG pipeline demonstrates flexibility across mul-
tiple ELM architectures while consistently improving BLEU-
4 and accuracy.
signal through a neural network encoder. Instead, it applies
the Byte Pair Encoding (BPE) algorithm [15] to create ECG
tokens. ECG-Byte has shown strong performance compared
with encoder-based approaches [14], making it our primary
method for all experiments unless otherwise specified. Build-
ing on ECG-Byte, we employ the Llama-3.2-1B-Instruct
checkpoint [16] via the HuggingFace Transformers API [17],
using default hyperparameters except for the addition of new
ECG signal tokens to the LLM’s embedding layer as de-
scribed in [12]. To demonstrate that our RAG pipeline is
not limited to ECG-Byte, we also report baselines with other
ELM architectures in Figure 2.
2.4. Learning Objective
Following previous work [14], we consider an autoregres-
sive objective that is compatible with general conversa-
tional input formats. Each input sequence is represented
asY= (y 1, y2, . . . , y T), constructed by concatenating a
system promptq sys, retrieved informationq rag, a signal token
sequenceX ID, the initial user queryq 1and assistant response
s1, and a sequence of query-response pairsq 2, s2, . . . , q n, sn.
To ensure that learning focuses exclusively on generating
the assistant responses, we define a target labeling function
ℓ:{1, . . . , T} → V ∪ {−100}, whereVis the model’s vo-
cabulary. The labeling function is
ℓ(t) =

yt,ifyt∈si
or end-of-turn token,
−100,otherwise
The value−100is used to mask out non-target tokens (q sys,
qrag,X ID,qn) during loss computation.LettingT={t∈ {1, . . . , T}:ℓ(t)̸=−100}, the loss
function is defined as
L(θ) =−X
t∈Tlogp θ(yt|y<t).
This formulation ensures that only assistant responses
S={s 1, . . . , s n}and end-of-turn tokens contribute to the
training loss. Retrieved content is treated purely as condition-
ing context and is excluded from supervision.
Table 2. Ablation study on different combinations of training
with RAG and inferencing with RAG.
Method Training w/ RAG Inference w/ RAG BLEU-4↑(%) Accuracy↑(%)
ECG-Byte [12] 22.85 ± 0.18 9.65 ± 0.29
ECG-Byte with RAG✓20.08 ± 0.11 7.17 ± 0.03
✓32.33 ± 0.10 14.96 ± 0.12
✓ ✓38.10 ± 0.05 18.27 ± 0.05
3. EXPERIMENTS
3.1. Experimental Settings
We utilize the Adam optimizer [18] with a learning rate of
1e-4 and weight decay of 1e-2. For all experiments, we train
the ELMs for 1 epoch with a batch size of 2 over 400,000
randomly sampled ECGs. We inference on a separate test set
of size 20,000 instances. For the input length of the ELM,
we pad/truncate inputs to a fixed size of 1024. We apply
Low-Rank Adaptation (LoRA) [19] with rank= 16. All ex-
periments were completed on NVIDIA RTX A6000 48 GB
GPUs.
We describe the experimental settings for each ELM con-
sidered. Following ECG-Byte [12], we train a tokenizer with
3,500 merges on 10-second unsegmented ECGs. We imple-
ment three neural network encoder-based ELMs from prior
work [14]: Merl, SigLIP (Signal), and SigLIP (Image). We
pretrain Merl on 800,000 ECG instances for 50 epochs using
a Res-Net101 backbone. For SigLIP-based ELMs, we use
the siglip-base-patch16-224 checkpoint from HuggingFace.
The two SigLIP variants differ only in input: SigLIP (Sig-
nal) uses stacked signal representations, while SigLIP (Im-
age) uses plotted ECG images [14]. We follow previous work
[14] and adopt a LLaV A-based approach [20], directly ap-
plying SigLIP without additional finetuning. For all encoder-

based ELMs, we freeze the encoder during LLM training and
add a projection layer jointly trained with the LLM. Lastly,
the three neural network encoder-based ELMs use the same
Llama-3.2-1B-Instruct checkpoint as the LLM.
4. RESULTS
4.1. Baselines
We present NLG results with and without RAG using ECG-
Byte, averaged across three random seeds and datasets, in Ta-
ble 1. Nearly all metrics show improvements when incorpo-
rating our RAG pipeline. To demonstrate that our pipeline is
not limited to ECG-Byte, we also report averaged results with
and without RAG integration for other ELM architectures (as
described in section 3.1) on the ECG-Instruct dataset in Fig-
ure 2. These results also show clear performance gains in both
BLEU-4 and accuracy when using our RAG pipeline.
4.2. Ablation Study
We ablate RAG usage across training and inference, top-kre-
trieval size, RAG placement in the input prompt, and noise
injection. Unless specified otherwise, experiments use the
ECG-Chat Instruct dataset with defaults of k=1, RAG in both
training and inference and system-prompt insertion. All re-
sults are averaged over 3 random seeds.
Table 3. Ablation study on varying the number of top k re-
trieved content.
k BLEU-4↑(%) Accuracy↑(%)
138.10 ± 0.05 18.27 ± 0.05
5 37.99 ± 0.04 18.07 ± 0.11
10 36.91 ± 0.09 17.20 ± 0.14
RAG During Training/InferenceIn Table 2, we present
results when introducing RAG during training and/or infer-
ence. Although RAG is often used only at inference time to
ground the LLM’s output, prior work has introduced RAG
during pretraining or finetuning to help the LLM learn the for-
mat of the inserted content [21]. We consider three settings
in Table 2: 1) using RAG only during training, 2) using RAG
only during inference, 3) using RAG during both training and
inference. We find that using RAG during both training and
inference yields the best performance, while using RAG only
during training and not at inference performs worse than the
baseline in Table 1 (without RAG). Using RAG only during
inference improves over the baseline but remains below the
performance achieved when RAG is used during both train-
ing and inference.
Varying Top-kWe evaluate model performance when al-
tering the number of retrieved items k (Table 3). Overall, the
differences are minor, with k=1 yielding slightly higher scores
than larger k values. While increasing k provides access to
more retrieved information, prior work [5] indicates that thisdoes not necessarily translate into better performance. As k
grows, the additional content may contain irrelevant or redun-
dant information, effectively acting as noise.
Table 4. Investigating the effect of injecting the RAG content
in the system prompt or user query.
RAG Location BLEU-4↑(%) Accuracy↑(%)
System prompt38.08 ± 0.1018.11 ± 0.10
User query 38.03 ± 0.0318.17 ± 0.23
RAG LocationWhere to place RAG content in the
prompt remains an open and underexplored question in NLP
[22]. In Table 4, we find that inserting the retrieved content
into either the system prompt or the user query yields com-
parable results. This similarity may arise because, in both
locations, the retrieved information becomes available to the
model before it generates its response, shaping the contextual
foundation of the interaction. However, subtle differences
in placement could influence how the model interprets the
retrieved content; either as background knowledge (system
prompt) or as part of the user’s intent (user query), which we
leave to future work for further exploration.
Table 5. The effect of injecting noise into RAG.
Method BLEU-4↑(%) Accuracy↑(%)
ECG-Byte [12] 22.85 ± 0.18 9.65 ± 0.29
ECG-Byte with RAG (Noise) 23.48 ± 0.05 8.19 ± 0.05
ECG-Byte with RAG38.10 ± 0.05 18.27 ± 0.05
Injecting noise into RAGTo test the effect of retrieval
quality, we inject noise into the RAG content by replacing di-
agnostic reports with “————————–” (Table 5). ECG-
Byte with and without noisy RAG show similar performance,
while incorporating correct reports yields clear gains, high-
lighting the importance of accurate retrieval.
5. DISCUSSION AND CONCLUSIONS
We present the first open-source RAG pipeline for ELMs,
showing consistent improvements in NLG across multiple
datasets and architectures. We also highlight four main find-
ings from our ablation studies: (1) using RAG during both
training and inference yields the highest performance, (2)
retrieving fewer items (e.g., top-1) marginally outperforms
larger retrievals, (3) RAG content placement (system prompt
vs. user query) produces similar outcomes, and (4) accurate
retrieval is necessary for performance improvements. These
insights provide practical guidance for the design of RAG-
enabled ELMs and establish a reproducible foundation for
future work.
AcknowledgmentsThis work is done in collaboration with
the Mario Lemieux Center for Heart Rhythm Care at Al-
legheny General Hospital.

6. REFERENCES
[1] World Health Organization, “Cardiovascular diseases,”
2024.
[2] Ryuichiro Yagi, Yuichiro Mori, Shinichi Goto, Taku
Iwami, and Kosuke Inoue, “Routine electrocardiogram
screening and cardiovascular disease events in adults,”
JAMA Internal Medicine, vol. 184, no. 9, pp. 1035–
1044, 09 2024.
[3] Jielin Qiu, William Han, Jiacheng Zhu, Mengdi Xu,
Michael Rosenberg, Emerson Liu, Douglas Weber, and
Ding Zhao, “Transfer knowledge from natural lan-
guage to electrocardiography: Can we detect cardiovas-
cular disease through language models?,” inFindings of
the Association for Computational Linguistics: EACL
2023, Andreas Vlachos and Isabelle Augenstein, Eds.,
Dubrovnik, Croatia, May 2023, pp. 442–453, Associa-
tion for Computational Linguistics.
[4] Yubao Zhao, Tian Zhang, Xu Wang, Puyu Han, Tong
Chen, Linlin Huang, Youzhu Jin, and Jiaju Kang, “Ecg-
chat: A large ecg-language model for cardiac disease
diagnosis,” 2024.
[5] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan
You, Chao Zhang, Mohammad Shoeybi, and Bryan
Catanzaro, “Rankrag: Unifying context ranking with
retrieval-augmented generation in llms,” 2024.
[6] Jielin Qiu, Jiacheng Zhu, Shiqi Liu, William Han, Jingqi
Zhang, Chaojing Duan, Michael A. Rosenberg, Emer-
son Liu, Douglas Weber, and Ding Zhao, “Automated
cardiovascular record retrieval by multimodal learning
between electrocardiogram and clinical report,” inPro-
ceedings of the 3rd Machine Learning for Health Sym-
posium. 2023, pp. 480–497, PMLR.
[7] Hung Manh Pham, Jialu Tang, Aaqib Saeed, and Dong
Ma, “Q-heart: Ecg question answering via knowledge-
informed multimodal llms,” 2025.
[8] Han Yu, Peikun Guo, and Akane Sano, “Zero-shot
ecg diagnosis with large language models and retrieval-
augmented generation,” inProceedings of the 3rd Ma-
chine Learning for Health Symposium. 10 Dec 2023,
vol. 225 ofProceedings of Machine Learning Research,
pp. 650–663, PMLR.
[9] Alistair E. W. Johnson, Lucas Bulgarelli, Lu Shen,
Alvin Gayles, Ayad Shammout, Steven Horng, Tom J.
Pollard, Benjamin Moody, Brian Gow, Li-wei H.
Lehman, Leo A. Celi, and Roger G. Mark, “Mimic-
iv, a freely accessible electronic health record dataset,”
Scientific Data, vol. 10, 01 2023.[10] Patrick Wagner, Nils Strodthoff, Ralf-Dieter Bousseljot,
Dieter Kreiseler, Fatima I. Lunze, Wojciech Samek, and
Tobias Schaeffter, “PTB-XL, a large publicly available
electrocardiography dataset,”Scientific Data, vol. 7, no.
1, pp. 154, May 2020, Number: 1 Publisher: Nature
Publishing Group.
[11] Jungwoo Oh, Gyubok Lee, Seongsu Bae, Joon myoung
Kwon, and Edward Choi, “Ecg-qa: A comprehensive
question answering dataset combined with electrocar-
diogram,” 2023.
[12] William Han, Chaojing Duan, Michael A. Rosenberg,
Emerson Liu, and Ding Zhao, “Ecg-byte: A tokenizer
for end-to-end generative electrocardiogram language
modeling,” 2024.
[13] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff
Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar ´e,
Maria Lomeli, Lucas Hosseini, and Herv ´e J´egou, “The
faiss library,”arXiv preprint arXiv:2401.08281, 2024.
[14] William Han, Chaojing Duan, Zhepeng Cen, Yihang
Yao, Xiaoyu Song, Atharva Mhaskar, Dylan Leong,
Michael A. Rosenberg, Emerson Liu, and Ding Zhao,
“Signal, image, or symbolic: Exploring the best input
representation for electrocardiogram-language models
through a unified framework,” 2025.
[15] Rico Sennrich, Barry Haddow, and Alexandra Birch,
“Neural machine translation of rare words with subword
units,” 2016.
[16] Grattafiori et al., “The llama 3 herd of models,” 2024.
[17] Wolf et al., “Huggingface’s transformers: State-of-the-
art natural language processing,” 2020.
[18] Diederik P. Kingma and Jimmy Ba, “Adam: A method
for stochastic optimization,” 2017.
[19] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan
Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and
Weizhu Chen, “Lora: Low-rank adaptation of large lan-
guage models,” 2021.
[20] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee, “Visual instruction tuning,” 2023.
[21] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang, “Retrieval-augmented generation for
large language models: A survey,” 2024.
[22] Joon Park, Kyohei Atarashi, Koh Takeuchi, and Hisashi
Kashima, “Emulating retrieval augmented generation
via prompt engineering for enhanced long context com-
prehension in llms,” 2025.