# GLiClass: Generalist Lightweight Model for Sequence Classification Tasks

**Authors**: Ihor Stepanov, Mykhailo Shtopko, Dmytro Vodianytskyi, Oleksandr Lukashov, Alexander Yavorskyi, Mykyta Yaroshenko

**Published**: 2025-08-11 06:22:25

**PDF URL**: [http://arxiv.org/pdf/2508.07662v1](http://arxiv.org/pdf/2508.07662v1)

## Abstract
Classification is one of the most widespread tasks in AI applications,
serving often as the first step in filtering, sorting, and categorizing data.
Since modern AI systems must handle large volumes of input data and early
pipeline stages can propagate errors downstream, achieving high efficiency and
accuracy is critical. Moreover, classification requirements can change
dynamically based on user needs, necessitating models with strong zero-shot
capabilities. While generative LLMs have become mainstream for zero-shot
classification due to their versatility, they suffer from inconsistent
instruction following and computational inefficiency. Cross-encoders, commonly
used as rerankers in RAG pipelines, face a different bottleneck: they must
process text-label pairs sequentially, significantly reducing efficiency with
large label sets. Embedding-based approaches offer good efficiency but struggle
with complex scenarios involving logical and semantic constraints. We propose
GLiClass, a novel method that adapts the GLiNER architecture for sequence
classification tasks. Our approach achieves strong accuracy and efficiency
comparable to embedding-based methods, while maintaining the flexibility needed
for zero-shot and few-shot learning scenarios. Additionally, we adapted
proximal policy optimization (PPO) for multi-label text classification,
enabling training classifiers in data-sparse conditions or from human feedback.

## Full Text


<!-- PDF content starts -->

GLICLASS : GENERALIST LIGHTWEIGHT MODEL FOR SEQUENCE
CLASSIFICATION TASKS
Ihor Stepanov1, Mykhailo Shtopko1, Dmytro Vodianytskyi1,
Oleksandr Lukashov1, Alexander Yavorskyi1, Mykyta Yaroshenko1
1Knowledgator Engineering, Kyiv, Ukraine
Correspondence: ingvarstep@knowledgator.com ,mykhailoshtopko@knowledgator.com
ABSTRACT
Classification is one of the most widespread tasks in AI applications, serving often as the first step
in filtering, sorting, and categorizing data. Since modern AI systems must handle large volumes of
input data and early pipeline stages can propagate errors downstream, achieving high efficiency and
accuracy is critical. Moreover, classification requirements can change dynamically based on user
needs, necessitating models with strong zero-shot capabilities. While generative LLMs have become
mainstream for zero-shot classification due to their versatility, they suffer from inconsistent instruction
following and computational inefficiency. Cross-encoders, commonly used as rerankers in RAG
pipelines, face a different bottleneck: they must process text-label pairs sequentially, significantly
reducing efficiency with large label sets. Embedding-based approaches offer good efficiency but
struggle with complex scenarios involving logical and semantic constraints. We propose GLiClass, a
novel method that adapts the GLiNER architecture for sequence classification tasks. Our approach
achieves strong accuracy and efficiency comparable to embedding-based methods, while maintaining
the flexibility needed for zero-shot and few-shot learning scenarios. Additionally, we adapted
proximal policy optimization (PPO) for multi-label text classification, enabling training classifiers in
data-sparse conditions or from human feedback.
Keywords Text classification ·Information Extraction ·NLP·RAG·GLiNER ·Zero-shot classification ·BERT
1 Introduction
Text classification is a fundamental task in machine learning with an extensive research history and significant practical
applications [Li et al., 2022]. It serves as a critical component in information extraction and analytical systems, powering
diverse applications from scientific article categorization [Lee et al., 2020] and support ticket classification [Revina
et al., 2020] to sentiment analysis [Giachanou and Crestani, 2016] and financial research [Felgueiras et al., 2020]. When
generalized to sequence classification, the impact extends further, including DNA sequence analysis [Helaly et al.,
2022] and RAG pipelines [Rosa et al., 2022], which have become essential for ensuring up-to-date, high-quality outputs
in modern chatbot systems.
Recent advances in auto-regressive language modeling have opened new possibilities for zero-shot classification tasks
[Brown et al., 2020, Raffel et al., 2023], including text classification [Puri and Catanzaro, 2019, Rasheed et al., 2024].
Although these models demonstrate impressive versatility, they often struggle with strict instruction adherence and
suffer from computational inefficiency in training and inference phases.
Cross-encoders operating as Natural Language Inference (NLI) models represent another popular approach for zero-shot
classification [Yin et al., 2019, Laurer et al., 2023] and RAG pipelines [Rosa et al., 2022]. These models treat the
input sequence as an NLI premise and construct hypotheses from candidate labels. Although more computationally
efficient than LLMs, they face scalability challenges with large label sets due to their pairwise processing requirement.
Furthermore, their limited ability to comprehend cross-label information can affect the quality of prediction in complex
scenarios.arXiv:2508.07662v1  [cs.LG]  11 Aug 2025

Knowledgator Engineering © 2025
Since the introduction of Word2Vec [Mikolov et al., 2013], embedding-based approaches have emerged as efficient
methods for text classification [Su et al., 2014], particularly in zero-shot settings [Dai et al., 2019]. The development
of sentence encoders improved semantic understanding [Perone et al., 2018], and Sentence Transformers [Reimers
and Gurevych, 2019] further enhanced embedding quality, enabling classification without fine-tuning [Piao, 2021].
SetFit extended this approach to achieve strong performance with minimal training examples [Tunstall et al., 2022].
Despite their efficiency, embedding-based methods often fall short in complex scenarios involving logical and semantic
constraints.
This work introduces GLiClass, a novel sequence classification model inspired by the GLiNER architecture [Zaratiana
et al., 2023] and explicitly adapted for text classification tasks. While developing the first multi-task GLiNER model
[Stepanov and Shtopko, 2024], text classification emerged as one of the evaluated tasks, exposing limitations that
highlighted the need for a more specialized solution. GLiClass addresses these limitations by combining the accuracy
of advanced architectures with the efficiency of embedding-based methods, while preserving strong zero-shot and
few-shot generalization capabilities. In this paper, we present updated architectural variants of GLiClass along with
an enhanced training methodology. The resulting models achieve performance comparable to or exceeding that of
cross-encoder baselines, with significantly improved computational efficiency. The development of other GLiNER
derivatives—such as GLiREL [Boylan et al., 2025] for zero-shot relation extraction and GLiDRE [Armingaud and
Besançon, 2025] for document-level relation extraction further demonstrates the flexibility of the GLiNER framework,
motivating the creation of task-specific generalist models like GLiClass.
2 Methods
2.1 Model Architecture
We developed several variants of the GLiClass architecture, with the main version built upon the GLiNER uni-encoder
design. Our primary models use the DeBERTa backbone [He et al., 2020], specifically DeBERTa v3 [He et al., 2021],
which incorporates Electra-style pretraining [Clark et al., 2020]—an approach shown to be particularly effective for text
classification tasks. In addition, we experimented with models based on the ModernBERT backbone [Warner et al.,
2024, Weller et al., 2025], which integrates several modern architectural enhancements, including support for Flash
Attention [Dao et al., 2022] and an extended context window. Despite these advancements, our experiments indicate
that DeBERTa-based models consistently outperform those based on ModernBERT. The GLiClass architecture was
designed to meet the following objectives:
• Perform multi-label classification in a single forward-pass, enabling efficient handling of multiple categories
without repeated computations;
•Achieve non-linear scaling with the number of classes provided, ensuring that inference time does not increase
proportionally with label count, which is crucial for large-scale applications;
•Enable inter-label information communication, allowing the model to capture relationships, hierarchies, and
dependencies between labels to improve prediction quality in complex scenarios.
At the time, while making GLiCLass more computationally efficient, our goal was to achieve performance at the level
of cross-encoders or even better accuracy, especially in the cases where inter-label communication can help.
To achieve this, a uni-encoder architecture was selected as the primary design, where text and labels are processed
jointly in a single encoder to facilitate rich interactions; however, we also developed and explored other variants, such as
bi-encoder (separate encoders for text and labels), fused bi-encoder (combining embeddings early), and encoder-decoder
(with cross-attention mechanisms), each offering trade-offs in efficiency, flexibility, and performance for different use
cases.
2

Knowledgator Engineering © 2025
2.1.1 Architecture Overview
Figure 1: GLiClass Uni-Encoder Architecture
GLiClass employs a sequence classification architecture that jointly processes label tokens with input text to enable rich
text-label interactions while maintaining computational efficiency. The pipeline consists of four main stages: (i) input
and label integration, (ii) contextual representation learning, (iii) representation pooling, and (iv) scoring.
2.1.2 Input Processing and Label Integration
Each class label is prepended with a special token «LABEL» and concatenated with the input text. This construction
allows the encoder to process text and labels jointly while preserving their distinct semantic roles.
2.1.3 Contextual Representation Learning
The concatenated sequence is processed through a bidirectional transformer encoder (e.g., BERT or DeBERTa), which
facilitates:
• Label-label interactions (capturing relations and hierarchies)
• Text-label interactions (text informing label representations)
• Label-text interactions (labels guiding text interpretation)
Unlike pairwise cross-encoders, this joint processing captures inter-label dependencies that would otherwise be missed,
leading to more informed predictions.
2.1.4 Representation Pooling
From the encoder outputs, we extract text and label representations separately using one of three pooling strategies:
• First-token pooling
• Mean pooling
3

Knowledgator Engineering © 2025
• Attention-weighted pooling
The pooling strategy can be selected based on task requirements.
2.1.5 Scoring Mechanism
Lett∈RB×Ddenote the pooled text embedding and c∈RB×C×Dthe pooled label embeddings for Cclasses. We
compute logits using either dot product or a learnable scorer:
Dot Product: sb,k=t⊤
bcb,k
τ(1)
NN Scorer: sb,k=g([tb;cb,k]) (2)
where τ >0is a temperature parameter and g(·)is a small MLP.
2.1.6 Layer-wise Attention Re-weighting
To optimize information flow across encoder layers, we implement a squeeze-excitation scheme. Let encoder layer
outputs be {U(k)}K
k=1, with U(k)∈RB×L×D. We compute layer weights as:
Z:,k=1
LLX
l=1Linear squeeze
U(k)
:,l,:
∈RB×K(3)
S=σ 
W2ReLU( W1Z⊤)⊤∈RB×K(4)
˜U=KX
k=1S:,k·U(k)∈RB×L×D(5)
O=Linear proj(˜U)∈RB×L×Dout(6)
where W1∈RK
2×KandW2∈RK×K
2.
2.1.7 Token-level Contrastive Loss
To enhance representation quality, we employ a token-level contrastive loss. Given embeddings E∈RB×L×Dand
token mask M∈ {0,1}B×L, let ˆE=E/∥E∥2be the L2-normalized embeddings along D. For each batch b, the
similarity matrix is:
S(b)=ˆEbˆE⊤
b∈RL×L(7)
The contrastive loss trains each valid token to identify itself among all tokens in its sequence:
L=1P
b,lMb,lBX
b=1LX
l=1Mb,l·CE 
S(b)
l,:, l
(8)
2.1.8 Architectural Variants
We explore four architectural configurations, each with specific advantages:
Uni-encoder: Processes text and labels jointly through a single encoder:
H=Encoder (X) (9)
C,Mc=ExtractClassFeatures (H,class_tokens ) (10)
T=Pooler (H) (11)
Logits =Scorer (T,C) (12)
4

Knowledgator Engineering © 2025
Bi-encoder: Uses separate encoders for text and labels:
T=TextEncoder (Xtext) (13)
C=ClassEncoder (Xclass) (14)
Logits =Scorer (T,C)/τ (15)
Fused bi-encoder: Combines class embeddings with text at the embedding layer:
Craw=ClassEncoder (Xclass) (16)
E=EmbeddingLayer (Xtext) (17)
E[class_token_pos ] =Craw[selected_classes ] (18)
H,Cfused =TextEncoder (E) (19)
Logits =Scorer (Pool(H),Cfused) (20)
Encoder-decoder: Employs an encoder-decoder architecture with cross-attention:
Henc=Encoder (Xtext) (21)
Hdec=Decoder (Xclass,Henc) (22)
C=ExtractClassFeatures (Hdec) (23)
Logits =Scorer (Pool(Henc),C) (24)
2.2 Data
Pre-training corpus: A 1.2M example general-purpose dataset covering text classification, sentiment analysis, and
natural language inference tasks.
Mid-training corpus: A representative subset of the pre-training corpus, used for intermediate fine-tuning.
Logic/NLI stream (post-training): Logical reasoning datasets including tau/CommonsenseQA and 2,000 synthetic
examples covering formal logic, sequent calculus, and NLI-style entailment/contradiction.
Pattern-focused stream (post-training): To address length and label density patterns, we created a dataset with texts
grouped by word count: [0,4,8,16,24,32,48,64,96,128,192,256,384,512,768,1024]
Short buckets (0-8 words) were populated with short ["title"] fields; buckets 8-48 with
fancyzhx/amazon_polarity using the ["content"] field; and buckets 48-1024 with samples from
m-a-p/FineFineWeb . All buckets were filled equally. Each text was annotated with GPT-4o to generate 50
true and 50 false candidate labels. For the final pattern-focused set, we sampled 2,000 texts in equal proportions from
all buckets; each example was duplicated, and for the duplicate, we varied the number of positive/negative labels using
random coefficients to diversify label density.
Additional NLI: Examples from nyu-mll/MultiNLI to strengthen classical NLI capabilities.
2.3 Model Training
2.3.1 Training Framework
Data Preparation: The dataset is loaded from JSON format, randomly shuffled, and split into 90% training and 10%
test partitions. Input sequences are tokenized with a maximum length of 1024 tokens using dynamic padding.
We implement two complementary training pipelines: standard supervised learning using focal loss and reinforcement
learning (RL), both extending the Hugging Face Trainer framework. The RL pipeline employs a modified Proximal
Policy Optimization (PPO) approach adapted for text classification.
2.3.2 Reinforcement Learning Loss Function
The total loss combines four components:
Ltotal=LPPO+Lvalue+LKL+Lentropy (25)
5

Knowledgator Engineering © 2025
1. PPO Loss:
LPPO=−1
NNX
i=1LX
j=1min
rijˆAij,clip(rij,1±ϵ)ˆAij
(26)
where rij=πθ(aij|si)
πθold(aij|si)is the probability ratio between current and previous policies, and ˆAij=Rij−V(si)is the
advantage estimate.
2. Value Loss:
Lvalue=|V(s)−R|2(27)
This measures the accuracy of the value model’s reward prediction.
3. KL-Divergence Penalty:
LKL=βKLDKL(πref∥πθ) (28)
Penalizes deviation from a reference policy, controlled by coefficient βKL.
4. Entropy Bonus:
Lentropy =βentH(πθ) (29)
Encourages policy exploration through prediction uncertainty, weighted by βent.
Key Hyperparameters:
•Clip range (ϵ): 0.2 (constrains policy updates)
•RL iterations : 3 (updates per batch)
•Entropy coefficient (βent): -1 (disabled by default)
•KL coefficient (βKL): -1 (disabled by default)
•Focal loss α: -1 (disabled)
•Focal loss γ: -1 (disabled)
•Label smoothing : -1 (disabled)
Special Modifications:
•Focal Loss Adaptation:
LPPO←1
NX
i,jαij(pt
ij)γLPPO (30)
•Label Smoothing:
asmooth
ij = (1−ϵsmooth )aij+ 0.5ϵsmooth (31)
•Action Sampling:
aij∼Bernoulli (pij)(stochastic)
I(pij>0.5) (deterministic)(32)
Training Execution:
1.Multi-iteration Updates : Each batch undergoes NRL-iters policy refinements
2.Separate Optimizers : Policy ( πθ) and value ( Vϕ) models have dedicated AdamW optimizers
3.Reference Integration : Frozen zero-shot pipeline provides baseline probabilities πref
4.Reward Composition :R=P
iwiri(s, a)with configurable components
5.Monitoring : Tracks Ltotal,E[ˆA], and individual reward metrics
6

Knowledgator Engineering © 2025
Shared Infrastructure:
•Layer-specific Optimization : Encoder layers use η= 10−5,δ= 0.01; classifier layers use ηother= 3×10−5,
δother= 0.01
•Gradient Handling : LayerNorm parameters excluded from weight decay
•Fault Tolerance : Per-batch exception handling with cache clearing
•Checkpointing : Snapshots every 1000 steps with 3-checkpoint retention
2.3.3 Training Stages
Pre-Training: Initial training on the 1.2M example corpus to learn general classification patterns and train custom
class tokens for pooling representations. Post hoc inspection of attention scores revealed two issues: (i) as the number
of labels increases, attention between tokens of labels and label tokens (prefixed with «LABEL» ) diminishes; (ii) under
extreme label-to-text token ratios (many labels and short texts), text representations degrade.
Mid-Training: Intermediate fine-tuning using the RL trainer on a subset of the pre-training corpus to refine decision
boundaries and improve label-text alignment. This bridge between large-scale pre-training and targeted post-training
yielded modest but consistent gains in macro F1 across diverse datasets.
Post-Training: Final stage combining logic/NLI and pattern-focused streams using Low-Rank Adaptation (LoRA) to
preserve prior knowledge while adapting to new patterns. We found that fine-tuning GLiClass on formal-logic tasks
formulated as question answering and classical NLI improves zero-shot text classification. The edge variant trained
more stably when using higher-rank (over-parameterized) LoRA adapters. Table 1 shows the LoRA configurations for
each model variant.
Table 1: LoRA configuration for GLiClass post-training
Model LoRA rank rLoRA α Focal loss α Target modules
gliclass-edge-v3.0 1536 3072 0.7 Wqkv, Wo, Wi,
linear_1, linear_2,
mlp.0, mlp.2, mlp.4
gliclass-modern-base-v3.0 512 1024 0.7 Wqkv, Wo, Wi,
linear_1, linear_2
gliclass-modern-large-v3.0 768 1536 0.7 Wqkv, Wo, Wi,
linear_1, linear_2
gliclass-base-v3.0 384 768 0.7 query_proj, key_proj,
value_proj, dense,
linear_1, linear_2,
mlp.0, mlp.2, mlp.4
gliclass-large-v3.0 384 768 0.7 query_proj, key_proj,
value_proj, dense,
linear_1, linear_2,
mlp.0, mlp.2, mlp.4
2.4 Evaluation
We evaluate GLiClass models against strong cross-encoder baselines on standard text classification benchmarks
including Rotten Tomatoes, CR, IMDB, and others (see Tables 3 and 4 for complete results). We also report few-shot
performance using 8 examples per label.
Inference speed is measured on a single NVIDIA A6000 GPU with batch size 1. We test across label counts
L∈ {1,2,4,8,16,32,64,128}and input lengths T∈ {64,256,512}tokens. For each (L, T)configuration, we
execute 10 forward passes and report average throughput in examples per second.
7

Knowledgator Engineering © 2025
3 Results
Table 2 summarizes model characteristics and performance. F1-score scales with model size within each family:
gliclass-large-v3.0 achieves the highest average (0.7193), followed by base (0.6764),modern-large (0.6197),
modern-base (0.5577), and edge (0.4900). Throughput shows an inverse relationship: edge is fastest (97.29 ex/s),
whilelarge is slowest among GLiClass models (25.22 ex/s).
Table 2: GLiClass v3.0 Model Overview
Model name Size Params Average Benchmark Avg. Inference Speed (ex/s)
gliclass-edge-v3.0 131 MB 32.7M 0.4900 97.29
gliclass-modern-base-v3.0 606 MB 151M 0.5577 54.46
gliclass-modern-large-v3.0 1.6 GB 399M 0.6197 43.80
gliclass-base-v3.0 746 MB 187M 0.6764 51.61
gliclass-large-v3.0 1.75 GB 439M 0.7193 25.22
Compared to cross-encoders (Table 4), GLiClass achieves superior accuracy-latency trade-offs.
gliclass-large-v3.0 surpasses the strongest cross-encoder baseline ( deberta-v3-large-zeroshot-v2.0 ,
0.6821) by +0.037 absolute (+5.5% relative), while gliclass-base-v3.0 remains within 0.006 absolute points.
gliclass-modern-large-v3.0 is comparable to roberta-large-zeroshot-v2.0-c (0.6197 vs. 0.6152).
Table 3: Performance Comparison of GLiClass Models
Dataset gliclass-
large
v3.0gliclass-base
v3.0gliclass-
modern-
large
v3.0gliclass-
modern-
base
v3.0gliclass-
edge
v3.0
CR 0.9398 0.9127 0.8952 0.8902 0.8215
sst2 0.9192 0.8959 0.9330 0.8959 0.8199
sst5 0.4606 0.3376 0.4619 0.2756 0.2823
20_news_groups 0.5958 0.4759 0.3905 0.3433 0.2217
spam 0.7584 0.6760 0.5813 0.6398 0.5623
financial_phrasebank 0.9000 0.8971 0.5929 0.4200 0.5004
imdb 0.9366 0.9251 0.9402 0.9158 0.8485
ag_news 0.7181 0.7279 0.7269 0.6663 0.6645
emotion 0.4506 0.4447 0.4517 0.4254 0.3851
cap_sotu 0.4589 0.4614 0.4072 0.3625 0.2583
rotten_tomatoes 0.8411 0.7943 0.7664 0.7070 0.7024
massive 0.5649 0.5040 0.3905 0.3442 0.2414
banking 0.5574 0.4698 0.3683 0.3561 0.0272
snips 0.9692 0.9474 0.7707 0.5663 0.5257
A VERAGE 0.7193 0.6764 0.6197 0.5577 0.4900
Few-shot adaptation with 8 examples per label consistently improves performance (Table 5). Average gains over zero-
shot are substantial: +0.1888 for edge (+50.0%), +0.2094 for modern-base (+47.1%), +0.1877 for modern-large
(+36.1%), +0.1067 for base (+18.4%), and +0.1063 for large (+17.1%). These results indicate that smaller variants
benefit disproportionately from limited supervision.
GLiClass demonstrates superior scalability with increasing label counts (Table 6, Figure 2). For gliclass-edge-v3.0 ,
throughput decreases modestly from 103.81 to 82.64 ex/s when scaling from 1 to 128 labels (-20%).
gliclass-base-v3.0 drops by 7% (49.42 →45.94 ex/s) and gliclass-large-v3.0 by 7.6% (19.05 →17.60
ex/s). In contrast, cross-encoders show dramatic degradation: deberta-v3-base-zeroshot-v2.0 drops from 24.55
to 0.47 ex/s ( ≈52×slower).
In aggregate, GLiClass delivers roughly 2.3×–16×higher average throughput than cross-encoders under our settings
(e.g.,large vs.deberta-v3-base :25.22/10.63 = 2 .37×;edge vs.deberta-v3-large :97.29/6.03 = 16 .1×;
base vs.roberta-large :51.61/16.12 = 3 .2×).
8

Knowledgator Engineering © 2025
Table 4: Cross-Encoders Performance Comparison
Dataset deberta-v3-
large
zeroshot-v2.0deberta-v3-
base
zeroshot-v2.0roberta-large
zeroshot-v2.0-ccomprehend_it
base
CR 0.9134 0.9051 0.9141 0.8936
sst2 0.9272 0.9176 0.8573 0.9006
sst5 0.3861 0.3848 0.4159 0.4140
enron_spam 0.5970 0.4640 0.5040 0.3637
financial_phrasebank 0.5820 0.6690 0.4550 0.4695
imdb 0.9180 0.8990 0.9040 0.4644
ag_news 0.7710 0.7420 0.7450 0.6016
emotion 0.4840 0.4950 0.4860 0.4165
cap_sotu 0.5020 0.4770 0.5230 0.3823
rotten_tomatoes 0.8680 0.8600 0.8410 0.4728
massive 0.5180 0.5200 0.5200 0.3314
banking77 0.5670 0.4460 0.2900 0.4972
snips 0.8340 0.7477 0.5430 0.7227
A VERAGE 0.6821 0.6559 0.6152 0.5331
Table 5: GLiClass Model Performance in Zero-shot and Few-shot Learning
Model
Examples
per label
sst5
financial_phrasebank
ag_news
emotion
cap_sotu
rotten_tomatoes
massive
banking77Avg
gliclass-edge-v3.0 0 0.2779 0.4986 0.6669 0.3854 0.2306 0.6955 0.2389 0.0255 0.3774
gliclass-edge-v3.0 8 0.3882 0.6998 0.7648 0.3989 0.3440 0.7344 0.5347 0.6644 0.5662
gliclass-modern-base-v3.0 0 0.2765 0.4199 0.6673 0.4237 0.3591 0.7070 0.3443 0.3581 0.4445
gliclass-modern-base-v3.0 8 0.3947 0.8675 0.7742 0.4700 0.4363 0.8264 0.6937 0.7683 0.6539
gliclass-modern-large-v3.0 0 0.4629 0.5940 0.7268 0.4506 0.4115 0.7653 0.3876 0.3653 0.5205
gliclass-modern-large-v3.0 8 0.5070 0.9066 0.8307 0.5337 0.4556 0.8638 0.7331 0.8354 0.7082
gliclass-base-v3.0 0 0.3377 0.8971 0.7279 0.4450 0.4681 0.7943 0.5041 0.4689 0.5804
gliclass-base-v3.0 8 0.4324 0.9116 0.8295 0.4931 0.4867 0.8450 0.7008 0.7975 0.6871
gliclass-large-v3.0 0 0.4627 0.9000 0.7183 0.4501 0.4666 0.8411 0.5651 0.5575 0.6202
gliclass-large-v3.0 8 0.5046 0.9042 0.8413 0.5303 0.5372 0.8827 0.7549 0.8563 0.7265
Dataset-level variability is present (Table 3). gliclass-large-v3.0 generally leads, while smaller or modern variants
occasionally match or exceed it on specific tasks (e.g., ag_news favorsbase ;sst5 is tight between modern-large
andlarge ). This suggests complementary inductive biases that can be further exploited in downstream selection.
4 Discussion
GLiClass effectively balances accuracy and speed, making it a versatile choice for sequence classification tasks.
As model size grows from edge tolarge , the average F1-score rises significantly (from 0.4900 to 0.7193), while
throughput decreases moderately (from 97.29 to 25.22 examples per second on an A6000 GPU). Unlike cross-encoders,
which experience severe slowdowns with more labels (e.g., 50 ×slower from 1 to 128 labels), GLiClass maintains
high efficiency, with only a slight throughput reduction (7–20% from 1 to 128 labels). This efficiency comes from
processing all labels in a single forward pass, ideal for production environments with large label sets. However, for
very large label sets (e.g., 1000+), efficiency may drop due to context length limits (around 1024 tokens), potentially
requiring techniques like truncation or batching. Additionally, performance can degrade with larger label sets, as seen
in datasets like banking77 , where accuracy slightly declines. Our tailored training approach has enabled GLiClass to
9

Knowledgator Engineering © 2025
Figure 2: Models Inference Speed Comparison
Table 6: Inference Speed: Samples per Second by Number of Labels (on A6000 GPU)
Model Name 1 2 4 8 16 32 64 128 Average
gliclass-edge-v3.0 103.81 101.01 103.50 103.50 98.36 96.77 88.76 82.64 97.29
gliclass-modern-base-v3.0 56.00 55.46 54.95 55.66 54.73 54.95 53.48 50.34 54.46
gliclass-modern-large-v3.0 46.30 46.82 46.66 46.30 43.93 44.73 42.77 32.89 43.80
gliclass-base-v3.0 49.42 50.25 40.05 57.69 57.14 56.39 55.97 45.94 51.61
gliclass-large-v3.0 19.05 26.86 23.64 29.27 29.04 28.79 27.55 17.60 25.22
deberta-v3-base-zeroshot-v2.0 24.55 30.40 15.38 7.62 3.77 1.87 0.94 0.47 10.63
deberta-v3-large-zeroshot-v2.0 16.82 15.82 7.93 3.98 1.99 0.99 0.49 0.25 6.03
roberta-large-zeroshot-v2.0-c 50.42 39.27 19.95 9.95 5.01 2.48 1.25 0.64 16.12
comprehend_it-base 21.79 27.32 13.60 7.58 3.80 1.90 0.97 0.49 9.72
match cross-encoder performance despite these challenges, though cross-encoders handle dense information better. We
attribute GLiClass’s limitations with extremely large label sets to current positional encoding and attention mechanisms,
which struggle to generalize across large contexts and effectively aggregate label information. These findings suggest
opportunities for future research into improved positional encoding and attention mechanisms to enhance scalability for
complex classification tasks.
The strong few-shot learning capabilities of GLiClass, particularly in smaller variants, underscore its adaptability to
new domains. With just 8 examples per label, the edge andmodern-base variants achieve substantial F1-score im-
provements (approximately 50% relative gain), making them ideal for resource-constrained scenarios. This adaptability
is driven by the joint text-label encoding strategy, which leverages contextual interactions to generalize from minimal
supervision.
Table 7 compares GLiClass with large language models (LLMs), cross-encoders, and embeddings-based models.
GLiClass achieves better scalability and efficiency than cross-encoders. Still, further increasing the label sets can
become more challenging for the model. We hypothesize it to the limitations of modern positional encoding and
attention mechanisms. In the case of GLiClass, the task becomes more complex with the model; it should be generalized
well to increase context size and aggregate information to label tokens. We believe that our work on GLiClass can
inspire further work on better positional encoding and attention mechanism approaches.
Post-training with Low-Rank Adaptation (LoRA) and specialized data streams (logic/NLI and pattern-focused) effec-
tively mitigates initial limitations, such as attention degradation at extreme label-to-text ratios. The layer-wise attention
re-weighting mechanism further enhances information flow, contributing to robust performance across diverse datasets.
10

Knowledgator Engineering © 2025
Table 7: Comparison of GLiClass, LLMs, Cross-encoders, and Embeddings Models for Classification Tasks
Aspect GLiClass LLMs Cross-encoders Embeddings Models
Scaling with Num-
ber of LabelsNon-linear; mild through-
put decrease (e.g., ∼7–
20% from 1 to 128 labels)
due to joint processing in
single forward passModerate; prompt length
increases with labels, but
generation time relatively
constant unless very large
setsPoor; linear decrease in
throughput as processes
text-label pairs sequen-
tially (e.g., 50 ×slower
from 1 to 128 labels)Excellent; constant time
for text encoding, sub-
linear for similarity com-
putations (very fast even
for large sets)
Performance Sta-
bility with Many
Labels (e.g., 100+)Moderate; feasible up to
context length limits (e.g.,
∼1024 tokens), with ef-
ficiency maintained via a
single pass, though trun-
cation or batching may be
required in extreme casesModerate; constrained
by context window size
(e.g., 8K–128K tokens),
requires prompt engineer-
ing; efficiency decreases
with very long promptsGood accuracy due to pair-
wise computations, but in-
ference time scales lin-
early with the number of
labelsExcellent; maintains both
high accuracy and compu-
tational efficiency
Computational Ef-
ficiencyHigh; single pass for
multi-label, comparable
to embeddings, optimized
for production (25–97
ex/s on A6000 GPU)Low; autoregressive gen-
eration is computationally
intensive, high latency for
inferenceMedium to Low; efficient
per pair but scales poorly
with labels due to repeated
forward passesHigh; fast encoding and
vector operations, mini-
mal compute per infer-
ence
Zero-Shot Capabil-
ityStrong; designed for flex-
ibility, outperforms cross-
encoders on benchmarks
(e.g., avg. F1 0.49–0.72)Strong but inconsistent;
versatile but struggles
with instruction adher-
enceStrong; good for NLI-
style classification but lim-
ited by lack of cross-label
infoModerate; effective for
semantic matching but
weaker on logical con-
straints
Few-Shot Capabil-
ityExcellent; significant
gains with minimal
examples (e.g., +17–50%
F1 with 8 examples/label),
especially for smaller
variantsStrong; in-context learn-
ing allows adaptation, but
requires careful prompt-
ingModerate; can fine-tune
but not optimized for few-
shot without additional
trainingModerate to Strong; meth-
ods like SetFit enable effi-
cient few-shot but may not
capture complex patterns
Handling Complex
Logical/Semantic
ConstraintsStrong; joint text-label
interactions capture rela-
tions, hierarchies, and de-
pendencies; enhanced by
logic/NLI post-trainingStrong; capable of com-
plex reasoning but may re-
quire large modelsModerate; pairwise pro-
cessing misses inter-label
dependencies, affecting
complex scenariosWeak; struggles with log-
ical constraints, relies on
semantic similarity
Overall Accuracy-
Efficiency Trade-
offSuperior; balances high
accuracy (surpasses cross-
encoders by ∼5.5%) with
embedding-like efficiency
and better scalabilityVersatile but inefficient;
high accuracy potential
offset by latency and in-
consistencyGood accuracy but poor
scalability limits practical
use for large label setsEfficient with good base-
line accuracy, but lower in
complex tasks compared
to others
Notably, higher-rank LoRA adapters improve training stability for the edge variant, suggesting that smaller models
benefit from over-parameterization during fine-tuning.
Consistent performance across datasets enables deployment-driven model selection: the large variant (0.7193 F1) suits
quality-critical applications, the base variant (0.6764 F1) offers a balanced trade-off, and the edge variant (0.4900 F1,
97.29 ex/s) excels in high-throughput scenarios. Dataset-specific variability highlights complementary inductive biases
among variants, which can be leveraged for task-specific optimization. Despite these strengths, challenges remain,
including calibration variability across datasets and sensitivity to extreme label-to-text ratios. These can be addressed
through targeted post-training, such as fine-tuning on diverse datasets or refining LoRA configurations. Our findings
on GLiClass suggest that limitations in scaling to large label sets may be linked to current positional encoding and
attention mechanisms, paving the way for future research into more robust approaches. Future work will also explore
optimizing attention mechanisms for extreme conditions and extending GLiClass to multilingual and domain-specific
settings to enhance its applicability.
11

Knowledgator Engineering © 2025
5 Conclusion
We introduced GLiClass, a label-conditioned encoder transformer-based family for sequence classification that success-
fully bridges the gap between accuracy and efficiency. The architecture achieves state-of-the-art results on standard
benchmarks while maintaining throughput that scales favorably with label count, which is a critical advantage over
pairwise cross-encoders.
Key contributions include:
• A novel uni-encoder architecture that jointly processes text and labels, enabling rich cross-label interactions;
•Superior accuracy-latency trade-offs, with the largest variant surpassing strong baselines by 5.5% while
maintaining practical inference speeds;
•Excellent few-shot learning capabilities, particularly for smaller models (up to 50% improvement with 8
examples);
• Robust scaling behavior with label count, maintaining 80% of single-label throughput even with 128 labels;
•Adaptation of proximal policy optimization to multi-label classification, which improves generalization and
enables training on data with limited label annotations or training with human feedback.
The GLiClass family offers flexible deployment options: large (0.7193 F1) for quality-critical scenarios, base
(0.6764 F1) for balanced deployments, modern variants for specific architectures, and edge (0.4900 F1, 97.29 ex/s)
for maximum throughput. Throughput degrades only mildly with the number of labels, contrasting with the sharp
slowdowns observed for pairwise cross-encoders.
Few-shot adaptation with 8 examples per label consistently improves performance, with the largest relative gains on
smaller models, enabling practical adaptation under tight annotation and latency budgets. Post-training with LoRA and
logic/pattern-focused streams stabilizes training and mitigates degradation under extreme label-text ratios.
Limitations include residual calibration differences across datasets, sensitivity under extreme label-text lengths, and
variability on fine-grained taxonomies. Future work will focus on improving calibration across datasets and extending
to multilingual settings and new domains.
6 Availability
Models are available through the GLiClass Python library at: https://github.com/Knowledgator/GLiClass
Pre-trained models can be downloaded from the Hugging Face repository at: https://huggingface.co/
collections/knowledgator/gliclass-v3-687a2d211b89659da1e3f34a
7 Acknowledgments
We sincerely thank Urchade Zaratiana, the creator of GLiNER, whose work and encouragement greatly inspired the
development of GLiClass.
References
Robin Armingaud and Romaric Besançon. Glidre: Generalist lightweight model for document-level relation extraction.
arXiv preprint arXiv:2508.00757 , 2025.
Jack Boylan, Chris Hokamp, and Demian Gholipour Ghalandari. Glirel–generalist model for zero-shot relation
extraction. arXiv preprint arXiv:2501.03172 , 2025.
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan,
Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-V oss, Gretchen Krueger, Tom
Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse,
Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam
McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL
https://arxiv.org/abs/2005.14165 .
Kevin Clark, Minh-Thang Luong, Quoc V Le, and Christopher D Manning. Electra: Pre-training text encoders as
discriminators rather than generators. arXiv preprint arXiv:2003.10555 , 2020.
12

Knowledgator Engineering © 2025
Yaoyao Dai, Abu Dhabi, and Benjamin J. Radford. Multilingual word embedding for zero-shot text classification. 2019.
URLhttps://api.semanticscholar.org/CorpusID:235194081 .
Tri Dao, Daniel Y . Fu, Stefano Ermon, Atri Rudra, and Christopher R’e. Flashattention: Fast and memory-efficient
exact attention with io-awareness. ArXiv , abs/2205.14135, 2022. URL https://api.semanticscholar.org/
CorpusID:249151871 .
Marco Felgueiras, Fernando Batista, and Joao Paulo Carvalho. Creating classification models from textual descriptions
of companies using crunchbase. Information Processing and Management of Uncertainty in Knowledge-Based
Systems , 1237:695 – 707, 2020. URL https://api.semanticscholar.org/CorpusID:219323803 .
Anastasia Giachanou and Fabio Crestani. Like it or not: A survey of twitter sentiment analysis methods. ACM
Computing Surveys (CSUR) , 49(2):1–41, 2016.
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled
attention. ArXiv , abs/2006.03654, 2020. URL https://api.semanticscholar.org/CorpusID:219531210 .
Pengcheng He, Jianfeng Gao, and Weizhu Chen. Debertav3: Improving deberta using electra-style pre-training with
gradient-disentangled embedding sharing. ArXiv , abs/2111.09543, 2021. URL https://api.semanticscholar.
org/CorpusID:244346093 .
Marwah A Helaly, Sherine Rady, and Mostafa M Aref. Bert contextual embeddings for taxonomic classification of
bacterial dna sequences. Expert Systems with Applications , 208:117972, 2022.
Moritz Laurer, Wouter van Atteveldt, Andreu Casas, and Kasper Welbers. Building efficient universal classifiers with
natural language inference. arXiv preprint arXiv:2312.17543 , 2023.
Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. Biobert: a
pre-trained biomedical language representation model for biomedical text mining. Bioinformatics , 36(4):1234–1240,
2020.
Qian Li, Hao Peng, Jianxin Li, Congying Xia, Renyu Yang, Lichao Sun, Philip S Yu, and Lifang He. A survey on text
classification: From traditional to deep learning. ACM Transactions on Intelligent Systems and Technology (TIST) ,
13(2):1–41, 2022.
Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector
space, 2013. URL https://arxiv.org/abs/1301.3781 .
Christian S. Perone, Roberto Silveira, and Thomas S. Paula. Evaluation of sentence embeddings in downstream and
linguistic probing tasks, 2018. URL https://arxiv.org/abs/1806.06259 .
Guangyuan Piao. Scholarly text classification with sentence bert and entity embeddings. In Pacific-Asia Confer-
ence on Knowledge Discovery and Data Mining , 2021. URL https://api.semanticscholar.org/CorpusID:
233873426 .
Raul Puri and Bryan Catanzaro. Zero-shot text classification with generative language models. arXiv preprint
arXiv:1912.10165 , 2019.
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li,
and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer, 2023. URL
https://arxiv.org/abs/1910.10683 .
Zeeshan Rasheed, Muhammad Waseem, Mika Saari, Kari Systä, and Pekka Abrahamsson. Codepori: Large scale
model for autonomous software development by using multi-agents. arXiv preprint arXiv:2402.01411 , 2024.
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings
of the 2019 Conference on Empirical Methods in Natural Language Processing . Association for Computational
Linguistics, 11 2019. URL https://arxiv.org/abs/1908.10084 .
Aleksandra Revina, Krisztian Buza, and Vera G Meister. It ticket classification: The simpler, the better. IEEE Access , 8:
193380–193395, 2020.
Guilherme Moraes Rosa, Luiz Henrique Bonifacio, Vitor Jeronymo, Hugo Abonizio, Marzieh Fadaee, Roberto
de Alencar Lotufo, and Rodrigo Nogueira. In defense of cross-encoders for zero-shot retrieval. ArXiv , abs/2212.06121,
2022. URL https://api.semanticscholar.org/CorpusID:254564419 .
Ihor Stepanov and Mykhailo Shtopko. Gliner multi-task: Generalist lightweight model for various information
extraction tasks. arXiv preprint arXiv:2406.12925 , 2024.
Zengcai Su, Hua Xu, Dongwen Zhang, and Yunfeng Xu. Chinese sentiment classification using a neural network
tool—word2vec. In 2014 International Conference on Multisensor Fusion and Information Integration for Intelligent
Systems (MFI) , pages 1–6. IEEE, 2014.
13

Knowledgator Engineering © 2025
Lewis Tunstall, Nils Reimers, Unso Eun Seo Jo, Luke Bates, Daniel Korat, Moshe Wasserblat, and Oren Pereg. Efficient
few-shot learning without prompts, 2022. URL https://arxiv.org/abs/2209.11055 .
Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis
Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Nathan Cooper, Griffin Adams, Jeremy Howard, and Iacopo
Poli. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context
finetuning and inference, 2024. URL https://arxiv.org/abs/2412.13663 .
Orion Weller, Kathryn Ricci, Marc Marone, Antoine Chaffin, Dawn Lawrie, and Benjamin Van Durme. Seq vs seq: An
open suite of paired encoders and decoders, 2025. URL https://arxiv.org/abs/2507.11412 .
Wenpeng Yin, Jamaal Hay, and Dan Roth. Benchmarking zero-shot text classification: Datasets, evaluation and
entailment approach, 2019. URL https://arxiv.org/abs/1909.00161 .
Urchade Zaratiana, Nadi Tomeh, Pierre Holat, and Thierry Charnois. Gliner: Generalist model for named entity
recognition using bidirectional transformer, 2023. URL https://arxiv.org/abs/2311.08526 .
14