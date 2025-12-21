# StutterFuse: Mitigating Modality Collapse in Stuttering Detection with Jaccard-Weighted Metric Learning and Gated Fusion

**Authors**: Guransh Singh, Md Shah Fahad

**Published**: 2025-12-15 18:28:39

**PDF URL**: [https://arxiv.org/pdf/2512.13632v1](https://arxiv.org/pdf/2512.13632v1)

## Abstract
Stuttering detection breaks down when disfluencies overlap. Existing parametric models struggle to distinguish complex, simultaneous disfluencies (e.g., a 'block' with a 'prolongation') due to the scarcity of these specific combinations in training data. While Retrieval-Augmented Generation (RAG) has revolutionized NLP by grounding models in external knowledge, this paradigm remains unexplored in pathological speech processing. To bridge this gap, we introduce StutterFuse, the first Retrieval-Augmented Classifier (RAC) for multi-label stuttering detection. By conditioning a Conformer encoder on a non-parametric memory bank of clinical examples, we allow the model to classify by reference rather than memorization. We further identify and solve "Modality Collapse", an "Echo Chamber" effect where naive retrieval boosts recall but degrades precision. We mitigate this using: (1) SetCon, a Jaccard-Weighted Metric Learning objective that optimizes for multi-label set similarity, and (2) a Gated Mixture-of-Experts fusion strategy that dynamically arbitrates between acoustic evidence and retrieved context. On the SEP-28k dataset, StutterFuse achieves a weighted F1-score of 0.65, outperforming strong baselines and demonstrating remarkable zero-shot cross-lingual generalization.

## Full Text


<!-- PDF content starts -->

1
StutterFuse: Mitigating Modality Collapse in
Stuttering Detection with Jaccard-Weighted Metric
Learning and Gated Fusion
Guransh Singh and Md. Shah Fahad
Abstract—Stuttering detection breaks down when disfluencies
overlap. Existing parametric models struggle to distinguish
complex, simultaneous disfluencies (e.g., a ’block’ with a ’pro-
longation’) due to the scarcity of these specific combinations in
training data. While Retrieval-Augmented Generation (RAG) has
revolutionized NLP by grounding models in external knowledge,
this paradigm remains unexplored in pathological speech pro-
cessing. To bridge this gap, we introduceStutterFuse, the first
Retrieval-Augmented Classifier (RAC) for multi-label stuttering
detection. By conditioning a Conformer encoder on a non-
parametric memory bank of clinical examples, we allow the model
to classify by reference rather than memorization. We further
identify and solve ”Modality Collapse”—an ”Echo Chamber”
effect where naive retrieval boosts recall but degrades precision.
We mitigate this using: (1)SetCon, a Jaccard-Weighted Metric
Learning objective that optimizes for multi-label set similarity, and
(2) aGated Mixture-of-Expertsfusion strategy that dynamically
arbitrates between acoustic evidence and retrieved context. On the
SEP-28k dataset, StutterFuse achieves a weighted F1-score of0.65,
outperforming strong baselines and demonstrating remarkable
zero-shot cross-lingual generalization.
Index Terms—stuttering detection ; multi-label classification
; Wav2Vec 2.0 ; Conformer ; retrieval-augmentation ; metric
learning ; contrastive loss ; cross-attention ; gated fusion
I. INTRODUCTION
Current speech recognition models fail when faced
with the irregular rhythmic interruptions typical of stutter-
ing—specifically blocks, prolongations, and repetitions. While
often dismissed as simple repetitions, stuttering is a complex
neurodevelopmental condition that breaks the flow and natural
rhythm of speech. It isn’t just one thing; it’s a collection
of disfluencies like sound repetitions (S-S-SoundRep), word
repetitions (Word-WordRep), and blocks, where airflow hits a
wall [1]. Beyond those things, stuttering carries a heavy weight,
often affecting communication confidence and personal life
from relationships to career paths [19].
Automated detection tools are critical for addressing these
issues. They provide objective metrics for speech-language
pathologists (SLPs) and real-time feedback for people who
stutter (PWS), while enabling more inclusive human-computer
interaction.
Building these systems is difficult because stuttering data
is messy. A single clip might contain multiple overlapping
disfluencies. This creates a combinatorial explosion of labels
The authors are with the Department of Computer Science, Birla
Institute of Technology, Mesra (e-mail: guransh766@gmail.com; fa-
had8siddiqui@bitmesra.ac.in).that standard classifiers can’t handle. Worse, the acoustic
signs are subtle and vary wildly between speakers, a problem
compounded by the scarcity of high-quality, annotated datasets.
Beacuse of these issues, simple classifiers often fail.
Prior work has often fallen short of addressing this com-
plexity. Many studies resort to binary (stuttered vs. fluent)
classification, use speaker-overlapping data splits that lead
to artificially inflated results, or treat the task as single-label,
forcing a ”winner-takes-all” choice that does not reflect clinical
reality.
In this work, we tackle the problem head-on by formulating
stuttering detection as a rigorous, speaker-independent, multi-
label classification task. We demonstrate that even high-capacity,
modern architectures like the Conformer [10], when trained
conventionally, struggle with the combinatorial explosion of
label sets and the imbalance of rare combinations. By leveraging
a retrieval-based approach, we aim to bypass the limitations
of ”parametric-only” learning, allowing the model to reference
specific, stored examples of these rare combinations during
inference.
Our central hypothesis is that a conventional classifier
can be significantly improved by explicitly providing it with
retrieved reference examplesfrom the most similar, already-
labeled instances in the training set. We proposeStutterFuse,
a retrieval-augmented pipeline that learns to ”classify by non-
local comparison.” Our contributions are:-
1)Retrieval-Augmented Architecture:We builtStutterFuse, a
system that pairs a standard classifier with a memory bank.
By adding aLate-Fusion (Gated Experts)mechanism, we
solved the ”Echo Chamber” problem—where the model
essentially copies its neighbors—allowing it to balance
acoustic evidence with retrieved context dynamically.
2)Jaccard-Weighted Metric Learning:Standard losses like
Multi-Similarity didn’t work well for our multi-label data.
So, we designedSet-Similarity Contrastive Loss (SetCon).
This loss function cares about the overlap of the entire
label set (Jaccard similarity), which bumped our retrieval
recall from 0.32 up to 0.47.
3)Cross-Attention Fusion:In our Mid-Fusion baseline, we
used a cross-attention module that doesn’t just look at
neighbor embeddings. It looks at a ”value” matrix contain-
ing their acoustics, labels, and similarity scores, letting
the model decide exactly which pieces of information to
trust.
4)Reproducible Results:We are releasing the full pipeline,
which achieves a weighted F1-score of 0.65. WhilearXiv:2512.13632v1  [cs.LG]  15 Dec 2025

2
perfectly clean detection remains elusive, this score is
approaching the limit of human agreement (Kappa ≈0.7)
[5], and we provide a full breakdown of why (Section
VI-F).
We are open-sourcing the embedder, Faiss index, and training
scripts to give the community a solid, reproducible starting
point: https://github.com/GS-GOAT/Stutter-Speech-Classifier/.
II. RELATEDWORK
Our work sits at the intersection of three primary research
areas: (1) automatic stuttering detection, (2) advances in deep
learning for speech, and (3) the emerging paradigm of retrieval-
augmented models.
A. Classical and Early Deep Learning for Stuttering Detection
The first generation of stuttering detection relied heavily on
manual feature engineering. Researchers would extract MFCCs,
LPC coefficients, or prosodic features (pitch, energy) and feed
them into standard classifiers like SVMs or GMMs [18]. These
methods were foundational but brittle; they often failed to
generalize across different speakers or capture the wide variety
of disfluency shapes.
Deep learning shifted the paradigm by automating feature
extraction. Early neural approaches—ranging from basic DNNs
[13] to CNNs [8] and RNNs [9]—showed they could learn
complex, hierarchical patterns directly from raw audio or
spectrograms. They comfortably outperformed the older feature-
based models [21]. But they hit a wall: deep models are data-
hungry, and in the niche field of pathological speech, labeled
data is scarce.
B. Self-Supervised Learning in Speech Processing
The recent breakthrough in speech processing has been
the development of self-supervised learning (SSL) models,
like Wav2Vec 2.0 [6], HuBERT [2], and data2vec [3] learn
high quality contextualized representations from large amounts
of unlabeled audio. These foundation models for speech can
then be fine-tuned on downstream tasks with limited labeled
data, or used as-is as high quality ”frozen” feature extractors.
The Wav2Vec 2.0 model, in particular, uses a contrastive loss
to learn discrete speech units, making it highly effective at
capturing phonetic and sub-phonetic details that are important
for identifying the subtle acoustic anomalies associated with
stuttering.
This approach has been successfully applied to stuttering.
[17] showed that fine-tuning a Wav2Vec 2.0 model yields state-
of-the-art results in stuttering detection. Our work builds on
this, using a frozen Wav2Vec 2.0 as a high-quality, reproducible
feature extractor. This allows us to decouple the feature
representation from the main contribution of our work: the
retrieval-augmented architecture built on top of these features.
By using frozen features, we also significantly reduce the
computational cost of training our subsequent metric learning
and classifier stages.C. Advanced Architectures: Attention and Conformers
Within the deep learning paradigm, attention mechanisms and
the Transformer architecture revolutionized sequence modeling.
For stuttering, attention-based BiLSTMs were explored in
models like StutterNet [16], which showed an ability to focus
on salient parts of the acoustic signal.
More recently, the Conformer architecture [10] has become
thede factostate-of-the-art for most speech tasks. By effectively
combining the local feature extraction of CNNs (via depthwise-
separable convolutions) with the global context modeling of
Transformers [24] (via multi-head self-attention), Conformers
provide an excellent inductive bias for speech. This motivates
our choice of a Conformer-based architecture as both our strong
baseline and the encoder for our final StutterFuse model.
D. Metric Learning and Retrieval-Augmentation
Standard parametric models are fundamentally limited by
their reliance on fixed weights to encode the entire variation
of stuttering phenomenology, often failing to capture the
long tail of rare and complex disfluencies. To overcome this
”memorization bottleneck”, we draw upon two complementary
paradigms: Metric Learning, which organizes the latent space
to reflect clinical similarity rather than just acoustic proxim-
ity, and Retrieval-Augmentation, which allows the model to
dynamically reference explicit examples from a memory bank,
thereby bridging the gap between specific instance recall and
general pattern matching.
a) Metric Learning:seeks to learn an embedding space
where a chosen similarity metric (e.g., Euclidean distance)
corresponds to a desired semantic similarity. This is often
achieved via siamese networks or, like, triplet loss [26]. Triplet
loss, famously used in FaceNet [15] and audio retrieval [28],
trains a model by optimizing for a “margin” where an anchor
sample is closer to positive samples (same class) than to
negative samples (different class) [27]. While common in
vision, its application to complex, multi-label audio events
like stuttering is less explored.
b) Retrieval-Augmentation:is a paradigm, primarily ap-
plied in large language models (e.g., RAG [12]), where a
model’s prediction is explicitly conditioned on information
retrieved from a large, external database. This has the effect
ofbolting ona massive memory to the model. This concept
is not new—the k-Nearest Neighbors (k-NN) algorithm is its
simplest form. Our work builds a modern, deep-learning-native
version of this: we use a learned metric space for retrieval (not
just raw features) and a deep-learning-based fusion mechanism
(cross-attention) to integrate the retrieved information, rather
than a simple majority-vote. This hybrid, ”parametic + non-
parametric” approach is what defines StutterFuse.
III. DATASET ANDPROBLEMFORMULATION
In this section, we detail the data and mathematical
framework used in our study. We first describe the primary
training corpus (SEP-28k) and the two out-of-domain evaluation
datasets (FluencyBank and KSOF) used to test robustness. We
then analyze the label distribution and our preprocessing strate-
gies for handling class imbalance, before formally defining the

3
multi-label classification task and the Jaccard-based similarity
metric.
A. SEP-28k Dataset
Our primary dataset is SEP-28k [5], one of the largest
publicly available corpora for stuttering detection. It con-
sists of approximately 28,000 audio clips, each roughly
3 seconds in duration, sourced from podcasts featuring
PWS. The clips are annotated with a multi-label ontology
derived from the Stuttering-Severity-Instrument 4 (SSI-4),
including Prolongation ,Block ,SoundRep ,WordRep ,
Interjection, andNoStutter.
A vital feature of this dataset is the availability of
speaker/show identifiers, which is essential for creating
clinically-relevant, speaker-independent evaluation splits. We
rigorously enforce this, ensuring that no speaker in the test set
is present in the training or validation sets.
B. FluencyBank Dataset
To assess the out-of-domain generalization of our models,
we use the FluencyBank dataset as a secondary test set [11].
FluencyBank consists of speech from PWS in more structured,
clinical, or conversational settings, differing significantly from
the podcast-style audio of SEP-28k. The labels were mapped
from FluencyBank’s scheme to the SEP-28k ontology by [7].
Evaluating on FluencyBank provides a robust test of whether
our model has learned fundamental acoustic properties of
disfluencies or has simply overfit to the acoustic environment
of SEP-28k.
C. KSOF Dataset
We also utilize the Kassel State of Fluency (KSoF) dataset
[23] to investigate cross-lingual generalization. The KSoF is a
German stuttering dataset sourced from various media formats.
Although the language differs, the physiological manifestations
of core stuttering events (like blocks and prolongations) share
acoustic similarities with English. We use the test split of
KSoF and map its labels to the SEP-28k ontology to perform
a zero-shot evaluation of our English-trained models.
D. Label Distribution and Data Preprocessing
As our goal is to characterizewhichdisfluencies are present
in a stuttered segment, we first filter the dataset to remove all
clips labeledonlyas NoStutter . We focused on careful data
augmentation strategies, as in a multi-label setting, naively
upsampling minority classes can inadvertently increase the
distribution of majority classes if they co-occur, necessitating
a nuanced approach.
The resulting label distribution in our speaker-independent
training split (post-filtering) is shown in Table I. The dataset
exhibits a severe long-tail imbalance: Block is present in
8,081 clips, whereas WordRep is present in only 2,759. This
imbalance poses a significant challenge, as a naive classifier
will be heavily biased towards the majority classes.
Furthermore, the multi-label nature of the data introduces
specific co-occurrence patterns, as shown in the Pearsoncorrelation heatmap in Figure 1. While most disfluency types
are statistically distinct (near-zero correlation), we observe
weak positive correlations between struggle behaviors (e.g.,
Block andSoundRep , r=0.19) and negative correlations
between distinct types (e.g., WordRep andProlongation ,
r=-0.10). This is a critical challenge: a simple ”instance-
balanced” augmentation (described in IV-A ) that duplicates
aWordRep clip may also duplicate a co-occurring Block ,
which can inadvertently worsen the majority-class bias. Our
metric learning stage (Section IV-B ) is designed specifically
to handle this combinatorial complexity.
Fig. 1: Pearson correlation matrix of disfluency labels in the
SEP-28k training set. The low correlation values indicate
that stuttering types are largely independent, supporting the
formulation of the task as a multi-label problem rather than
multi-class.
E. Problem Formulation
We formulate the task as multi-label classification.
•LetXbe a 3-second raw audio clip, which we represent
as a feature sequence FX∈RT×D, where T= 150
frames andD= 1024features (from Wav2Vec 2.0).
•LetC= 5be the number of disfluency classes.
•The target is a binary vector y∈ {0,1}C, where yi= 1
if thei-th disfluency is present, andy i= 0otherwise.
•The model Mmust learn a mapping M(F X)→ˆy , where
ˆy∈[0,1]Cis a vector of predicted probabilities, typically
from a sigmoid activation.
For our metric learning stage, we define a label similarity
metric. We use theJaccard distance dJ(ya, yb)between two
label vectorsy aandy b, which is defined as:
dJ(ya, yb) = 1−|ya∩yb|
|ya∪yb|=|(ya̸=yb)∧(y a∨yb)|
|ya∨yb|
This metric is 0 for identical label vectors and 1 for perfectly
disjoint, non-empty vectors. It is the ideal metric for our
similarity-based mining, as it directly measures the similarity
of the multi-label sets.

4
TABLE I: Label counts in the speaker-aware training set before and after instance-balanced augmentation.
Disfluency Class Original Count Augmented Count
Block 8081 10,848
Interjection 5934 10,824
Prolongation 5629 10,800
SoundRep 3486 10,576
WordRep 2759 10,569
IV. METHODOLOGY: THESTUTTERFUSEPIPELINE
Our proposed pipeline, StutterFuse, is a multi-stage process
designed to tackle the challenges of multi-label classification
and class imbalance through retrieval-augmentation. The full
pipeline is illustrated in Figure 2.
A. Stage 1: Feature Extraction and Augmentation
a) 4.1.1. Wav2Vec 2.0 Feature Extraction:We use the
facebook/wav2vec2-large-960h model as a frozen
feature extractor. Each 3-second audio clip is passed through
the model, and we extract the hidden states from its final
transformer layer. This sequence is center-truncated or padded
to a fixed length of T= 150 frames. This results in a feature
tensor FX∈R150×1024for each clip. Precomputing these
features Significantly reduces training time and ensures a
consistent, high-quality input for all subsequent experiments.
b) 4.1.2. Instance-Balanced Augmentation via Optimiza-
tion:As Table I highlights, the raw training set is heavily
skewed. Traditional class-wise oversampling doesn’t work well
here; if you duplicate a clip to boost a rare label like WordRep ,
you often unintentionally boost a co-occurring common label
likeBlock , which just distorts the marginal distributions
further.
To get around this, we treat augmentation as an optimization
problem. For every class c, we calculate a rarity score rc=
1/fc, based on raw frequency fc. For any clip xwith a set of
labelsL x, we define its instance rarity asr x= min c∈Lxrc.
Rather than deterministically augmenting every instance,
we formulate the selection of augmentation candidates as a
constrained optimization problem. Let ai,k∈ {0,1} indicate
whether instance ireceives its k-th augmentation copy. We
aim to minimize the weighted squared error between the final
class counts ˆFcand target countsT c:
min
ai,kX
cwc(ˆFc−Tc)2
subject to per-instance caps based on rarity (P
kai,k≤
⌈g(rx)⌉) and a total budget constraint. We solve this using
a greedy heuristic that iteratively selects the instance-copy
yielding the greatest reduction in the objective.
We use audiomentations (GaussianNoise, PitchShift,
TimeStretch) to generate the selected copies. This process
yields the near-equalized distribution detailed in Table I,
providing sufficient minority-class exposure without massive
co-occurrence distortion.
B. Stage 2: Metric Learning for Stuttering Similarity
A simple average-pooling of the Wav2Vec 2.0 features is
not optimized for retrieval based onlabel similarity. Our key
insight is to train a new, dedicated embedder for this purpose.a) 4.2.1. Embedding Architecture:We design a special-
ized ”BiGRU-Attention” embedder to transform the 150×1024
sequence into a single, fixed-size vector e∈R1024. The
architecture, detailed inembedr.py, consists of:
1)BiGRU Encoder:We employ a Bidirectional Gated Re-
current Unit (BiGRU) as the primary sequence encoder to
capture the temporal dynamics of disfluencies. With 256
hidden units per direction, it processes the 150-frame
Wav2Vec 2.0 sequence, producing a 512-dimensional
output at each time step. This bidirectional processing
ensures that the representation of any potential stuttering
event is informed by both preceding and succeeding
acoustic context, capturing the full envelope of the
disfluency.
2)Projection:A dense projection layer expands the en-
coder’s output to a higher-dimensional space of 1024
units, utilizing a ReLU activation. Intentionally avoiding
bottlenecks, this high-dimensional projection preserves
the rich information capacity needed to represent com-
plex, overlapping multi-label combinations. This allows
the network to ”untangle” the feature manifold before
the aggregation step, ensuring that subtle acoustic cues
distinguishing rare classes are not compressed away.
3)Attention Pooling:Instead of rigid global averaging,
we implement a learnable self-attention mechanism to
aggregate the temporal sequence. A dedicated scoring
head (Dense →Tanh→Softmax) assigns a relevance
weight to each frame, effectively learning to identify and
prioritize the moments where stuttering occurs. The final
pooled vector is computed as the weighted sum of the
projected features, allowing the model to suppress silence
or fluent segments and focus entirely on the pathological
event.
4)L2 Normalization:The final aggregated embedding is
passed through an L2 normalization layer, projecting the
vector onto the unit hypersphere. This step is mathemat-
ically essential for our metric learning framework, as it
makes the dot product equivalent to Cosine Similarity.
By removing magnitude variations, we ensure that the
subsequent neighbor retrieval is based purely on the
semantic orientation of the embeddings, stabilizing the
optimization of the contrastive loss.
b) 4.2.2. Set-Similarity Contrastive Loss (SetCon):We
train the embedder using aSetCon( τ= 0.1 ). Unlike standard
Triplet Loss, SetCon leverages the full batch for contrastive
learning and weighs positive pairs based on their label overlap.
L=X
i∈I−1
|P(i)|X
p∈P(i)wip·logexp(z i·zp/τ)P
a∈A(i)exp(z i·za/τ)(1)

5
Fig. 2: A high-level overview of the proposed StutterFuse pipeline. (1) A specialized embedder is trained with Multi-Similarity
Loss. (2) A Faiss index is built from these embeddings. (3) The final classifier uses cross-attention to fuse information from the
query clip and itsk-retrieved neighbors.
The weight wipis derived from the Jaccard similarity between
the label sets of anchor iand positive p. This approach resonates
with recent findings in the NLP domain, such as the Jaccard
Similarity Contrastive Loss (JSCL) proposed by [29] for multi-
label text classification. We extend this concept to the acoustic
domain, ensuring that the model learns to cluster samples with
similarmulti-label profiles, not just shared single labels.
c) Impact of SetCon and Failure of Standard Losses::
We found that this Jaccard-weighted objective was critical
for performance. We compared our approach against several
strong baselines. The simple averaging of Wav2Vec 2.0
features (Mean Pooling) yielded a Recall@5 of0.32. State-
of-the-art metric learning objectives provided moderate gains,
withMulti-Similarity (MS) Lossachieving0.39andStandard
Triplet Lossreaching0.42. However, our proposedSetCon
objective outperformed these methods significantly, achieving
a Recall@5 of0.47, confirming the importance of Jaccard-based
optimization for multi-label retrieval. Standard losses like MS
Loss and SupCon are designed formulti-classproblems: they
treat any shared label as a perfect match (binary positive)
and push everything else away. In the multi-label stuttering
domain, this is flawed. A ”Block” and a ”Block+WordRep” are
similar but not identical. SetCon’s continuous Jaccard weighting
captures these nuances, creating a semantically rich space where
partial overlaps are respected. This 16% absolute gain over the
baseline provides a far denser memory bank for the subsequent
fusion stages.
d) 4.2.3. Training and Validation:We trained the SetCon
model for 20 epochs using the Adam optimizer [25] with a
learning rate of 1e-4. The goal of this stage wasn’t just to lower
loss, but to improve retrieval quality. To track this, we built a
custom RecallAtKEvaluator . At the end of every epoch,
this callback would generate embeddings for the validation set,
build a temporary index, and check if the top 10 neighbors
were actually relevant (Set Jaccard Distance ≤0.5 ). The logs
showed steady progress, eventually hitting aRecall@5 of 0.47.
This metric confirmed that our embedding space was actuallylearning semantic similarity. We saved the weights from the
epoch that maximized this recall.
To double-check this visually, we projected the embeddings
using t-SNE (Figure 3). The difference is stark: while base-
line Wav2Vec 2.0 features are a tangled mess, our learned
embeddings form clear, distinct clusters for different stutter
types. We further analyze the structure of this space in Figure
4, which shows how multi-label combinations naturally cluster
between their constituent pure classes, and Figure 5, which
decomposes the space by class to highlight the compactness
of Blocks versus the variance of Interjections.
C. Stage 3: Retrieval-Augmented Classifier (RAC)
This is our final StutterFuse model, which uses the embedder
from Stage 2 to perform classification.
a) 4.3.1. ANN Index Construction:We first use the trained
and saved embedder model to compute the final, fixed-size
embeddings e∈R1024for all clips in the augmented training
set. These vectors are L2-normalized and stored in aFaiss
IndexFlatIP (Flat Inner Product) index. This index allows
for efficient MIPS (Maximum Inner Product Search), which
for L2-normalized vectors is equivalent to finding the nearest
neighbors by cosine similarity.
b) 4.3.2. Model Inputs:The StutterFuse classifier is a
multi-input model. For each batch sample, the data pipeline
retrieves its k= 5 nearest neighbors from the Faiss index and
provides the model with the following dictionary of tensors:
1)input_test_seq : The T×D Wav2Vec 2.0 sequence
of the target clip (query).
2)input_neighbor_vecs : The k×D pooled embed-
ding vectors of itskneighbors.
3)input_neighbor_labels : The k×C ground-truth
label vectors of thekneighbors.
4)input_neighbor_sims : The ksimilarity scores (dis-
tances) from the Faiss search.

6
Fig. 3: t-SNE visualization of the embedding space. Left: Raw Wav2Vec 2.0 features show poor separation. Right: Our
metric-learned embeddings show distinct clusters for different stuttering types, facilitating effective retrieval. Note the formation
of distinct clusters for Block and Prolongation, which were entangled in the Baseline space.
Fig. 4: Interaction Map of the learned embedding space. The
visualization reveals how multi-label combinations (e.g., Block
+ WordRep) cluster naturally between their constituent pure
classes, validating the model’s ability to capture compositional
semantics.
D. Phase 2: Fusion Strategies
We explore two distinct strategies for integrating the re-
trieved information. Note that in both strategies, the retrieved
neighbors are processed asvectors(pooled embeddings), not
full sequences, to maintain computational efficiency.
1) Strategy A: Mid-Fusion via Cross-Attention (RAC):In
this approach, we fuse the acoustic and retrieval streamsbefore
the final classification head. In this architecture, theQuery
(Q)is derived from the target audio, processed by a shared
Conformer encoder and pooled to a vector q∈R1024. The
Keys ( K) & Values ( V)are formed from the kretrieved
neighbor vectors, which are projected to 1024 dimensions to
create the Keys. The Values consist of a concatenation ofthese Keys, the neighbor label embeddings (Dense 16), and
their similarity scores. Finally, aCross-Attentionlayer (4
heads, key dim=256) attends to these neighbors, computing
the context as Context=Attention(Q=q, K=K neigh, V=
Vneigh). The context vector is concatenated with the query and
passed to the final MLP (Dense 512 →Dropout →Dense 256
→Output).
2) Strategy B: Late-Fusion via Gated Experts (StutterFuse):
To address the ”Echo Chamber” effect, we propose aLate-
Fusionarchitecture that treats the streams as independent
experts.
1)Expert A (Audio):A Conformer (2 blocks, ff dim=512)
processes the target audio. Output:z a∈R256.
2)Expert B (Retrieval):A lightweight MLP processes the k
neighbor vectors (Dense 256 →GlobalAvgPool →Dense
128). Output:z r∈R128.
3)Gated Fusion:A learned gate g∈[0,1] dynamically
weighs the retrieval expert:
g=σ(W g[za;zr]+bg), y final =MLP([z a;g·zr])(2)
ThisStutterFusemodel (Figure 6) allows the network to
suppress the retrieval stream when the acoustic signal is
unambiguous.
V. EXPERIMENTALSETUP
A. Implementation Details
All models were implemented in TensorFlow 2.x with Keras.
We used the faiss-cpu library [4] for nearest-neighbor
search. Due to the large model size and batch requirements,
all experiments were conducted on a Google Cloud TPU v5e-
8, using tf.distribute.TPUStrategy for distributed
training. The global batch size was set to 128 (16 per replica).

7
Fig. 5: Class Decomposition Grid. We decompose the embedding space by class to analyze the distribution of each disfluency
type. While ’Blocks’ and ’Prolongations’ form tight, distinct clusters, ’Interjections’ are more diffuse, highlighting the acoustic
variability of this class.
Fig. 6: The StutterFuse Architecture (Late-Fusion). The model processes the audio and retrieved neighbors via separate expert
streams. A learned gate dynamically weighs the retrieval expert’s contribution before the final classification.
a) Stage 2 (Embedder) Training::Adam optimizer,
LR=1e-4. Trained for 20 epochs, with the best model selected
byval_recall_at_5.
b) Stage 3 (RAC) Training::AdamW optimizer, LR=2e-5,
weight decay=5e-4. Loss was Binary Cross-Entropy with 0.1
label smoothing. We used val_auc_roc as the monitoring
metric for early stopping with a patience of 5, as it is a
threshold-independent metric suitable for imbalanced data.
Alternative losses like Focal Loss [30] were considered but
BCE with smoothing proved more stable. Figure 7 illustrates
the training progression, showing the convergence of the loss
and the steady improvement of the AUC-ROC score.
B. Baselines for Comparison
To quantify the benefit of our pipeline, we compare Stutter-
Fuse against a suite of strong baselines, all trained on the same
frozen Wav2Vec 2.0 features and instance-balanced augmented
data: We initially experimented with several standard deep
learning architectures, including aDNN,CNN, and a standard
Transformerencoder. However, preliminary observations in-
dicated that these models were less effective at capturing the
subtle, multi-scale dynamics of stuttering. Consequently, we
chose theConformer (No Retrieval)as our primary baseline,
as it provided the strongest parametric performance. Thismodel consists of a 2-block Conformer encoder (4 heads,
ffdim=1028). Unlike standard approaches that pool features
before classification, this model applies a TimeDistributed
Dense layer with sigmoid activation to the frame sequence,
followed by Global Average Pooling, ensuring it can detect
disfluencies occurring at any point in the clip.
C. Evaluation Metrics
Since this is a multi-label task with heavy imbalance,
accuracy is a meaningless metric. Instead, we look at: We
assess performance using aPer-Class Breakdownof Precision,
Recall, and F1-score for all 5 classes. Our primary aggregate
metric is theWeighted F1-score, which balances precision
and recall while accounting for the prevalence of common
classes like Blocks. We also reportMicroandMacroaverages
to distinguish between global hit/miss rates and class-equalized
performance. Finally, all classification reports utilize a sigmoid
Thresholdingof 0.3. As detailed in Section VII, this threshold
is not arbitrary but a necessary clinical trade-off to ensure
sensitivity to rare stuttering events.

8
Fig. 7: Training dynamics of the StutterFuse model. Left: Binary Cross-Entropy Loss over epochs. Right: Validation AUC-ROC
score, demonstrating stable convergence.
VI. RESULTS ANDANALYSIS
A. Performance on SEP-28k
Our main experimental results on the speaker-independent
SEP-28k test set are presented in Table II. We compare
the baseline Conformer (Audio-Only) against our two fusion
strategies: Mid-Fusion (RAC) and Late-Fusion (StutterFuse).
Table III provides a detailed breakdown of the class-wise
performance for our best model.
B. Analysis of Fusion Strategies
The results expose a clear trade-off between our two fusion
approaches:
•Mid-Fusion (RAC):This model is an aggressive retriever.
It hits a massive recall of0.82(up from 0.72 baseline),
but its precision suffers (0.52). We call this the “Echo
Chamber” effect: the model sees retrieved neighbors with
stutters and feels pressured to predict a stutter, even if the
audio is clean. It trusts the crowd too much and also is
overwhelmed by the retrived data which is k times more
than the actual clip info.
•Late-Fusion (StutterFuse):This architecture is more
balanced. By keeping the streams separate and using a
gate, it learns to be selective. It trusts the “Retrieval Expert”
when things are ambiguous (like WordRep) but sticks to
the “Audio Expert” when the signal is clear. This balance
allows it to recover precision (0.56) while keeping most
of the recall benefits, leading to the best overall F1 of
0.65.
With aWeighted Precisionof 0.60 andWeighted Recallof
0.72, coupled with our chosen threshold of 0.3, the model is
tuned to be more suitable for clinical screening. It catches
90% of Blocks and 84% of Interjections, which is great for
screening. The downside is it makes false guesses more often
(lower precision). This isn’t necessarily a failure of the model’s
intelligence (the AUC-ROC is a healthy 0.7670), but rather a
calibration choice to minimize missed diagnoses.C. Cross-Dataset Evaluation on FluencyBank
To test the model’s robustness, we evaluated the saved
StutterFuse model directly on the FluencyBank test set without
any fine-tuning. The results are shown in Table IV. Table IV
presents the detailed class-wise performance of our models
on FluencyBank. To quantify the specific contribution of
the retrieval mechanism in this domain-shifted setting, we
analyze the relative gain of the Fusion model over the Audio-
Only Expert (Expert A). As shown in Table V, retrieval
provides a significant boost to repetition classes, which are
often acoustically ambiguous and benefit from the ”consensus”
of retrieved neighbors.
The results indicate that while the overall weighted F1
is similar, the retrieval mechanism specifically targets and
improves the detection of repetition disfluencies, offering a
complementary signal to the acoustic encoder.
D. Qualitative Analysis of Retrieval and Attention
To better understand how StutterFuse resolves effectively or
fails in complex scenarios, we visualize the attention weights
and retrieved neighbors for three representative test cases.
a) Test Case 1: Successful Multi-Label Resolution:Figure
8 illustrates a complex clip containing three distinct disfluency
types: SoundRep ,WordRep , and Interjection . The
retrieval system accurately reflects this diversity, returning
neighbors that exhibit various combinations of these traits
(e.g., N1 contains SoundRep andWordRep ; N3 contains
SoundRep andInterjection ). The attention mechanism
assigns high, uniform weights ( ≈0.88 ) to all neighbors,
effectively deriving a ”consensus” from these partial matches to
correctly predict all three ground-truth labels. This demonstrates
the model’s ability to synthesize a correct multi-label prediction
from a heterogenous set of retrieved examples.
b) Test Case 2: The Echo Chamber Effect:In Figure 9, the
ground truth contains both WordRep andInterjection .
However, the retrieved neighbors are unanimously labeled
asWordRep , with high attention weights ( ≈0.89 ). Heavily

9
TABLE II: Performance Comparison on SEP-28k (Test Set). While Mid-Fusion achieves the highest recall, Late-Fusion offers
the best balance, achieving the highest overall F1-score.
Model Architecture Prec. Rec. F1 Gain (%) Insight
1. Audio-Only Baseline 0.66 0.56 0.60 - High precision, misses context
2. Mid-Fusion (RAC) 0.52 0.82 0.64 +4.9% High Recall, “Echo Chamber”
3.Late-Fusion (StutterFuse) 0.60 0.72 0.65 +6.6% Balanced Performance
TABLE III: Detailed Class-wise Performance of the Final StutterFuse Model (Late-Fusion).
Class Precision Recall F1-Score
Prolongation 0.53 0.73 0.61
Block 0.58 0.76 0.66
SoundRep 0.49 0.59 0.54
WordRep 0.50 0.62 0.55
Interjection 0.76 0.78 0.77
Weighted Avg 0.60 0.72 0.65
TABLE IV: Zero-Shot Cross-Dataset Performance onFluencyBank. Comparison of RAC and StutterFuse.
Class RAC (Mid-Fusion) F1 StutterFuse (Late-Fusion) F1
Prolongation 0.43 0.42
Block 0.52 0.51
SoundRep 0.54 0.54
WordRep 0.48 0.49
Interjection 0.67 0.70
Weighted Avg 0.55 0.55
TABLE V: Impact Analysis: Relative Gain of StutterFuse (Fusion) over Audio-Only Baseline (Expert A) on FluencyBank.
Class Audio Baseline F1 Fusion F1 Relative Gain (%)
SoundRep 0.504 0.542 +7.5%
WordRep 0.458 0.488 +6.6%
Fig. 8: Analysis of Test Case 1. The model successfully
aggregates information from neighbors with partial label
overlaps (e.g., SoundRep mixed with WordRep) to correctly
predict the full set of three disfluencies.
influenced by this uniform retrieval results, the model predicts
WordRep with high confidence but fails to detect the co-
occurring Interjection . This highlights the risk of an
”Echo Chamber,” where the lack of diversity in the retrieved
neighbors causes the model to miss secondary disfluencies that
are not represented in the top-kresults.
c) Test Case 3: Acoustic Confusion and Noise:Fig-
ure 10 shows a failure case involving a highly cluttered
Fig. 9: Analysis of Test Case 2. The retrieved neighbors are
unanimously ’WordRep’. Consequently, the model correctly
predicts ’WordRep’ but misses the co-occurring ’Interjection’,
illustrating how retrieval bias can suppress rare classes.
sample (True: Prolongation ,SoundRep ,WordRep ,
Interjection ). The retrieval system returns a mix of
Prolongation (4/5 neighbors) and Block (1 neighbor).
The model predicts Prolongation andBlock , missing the
repetition and interjection components. Here, the embedding
space likely collapsed the complex acoustic signature onto the
Prolongation prototype, and the single Block neighbor
(N3) triggered a false positive block prediction. This suggests

10
Fig. 10: Analysis of Test Case 3. A complex multi-label
sample results in noisy retrieval. The model latches onto
the dominant signal (Prolongation) and an acoustic imposter
(Block), differing from the complex ground truth.
that for extremely dense disfluency clusters, the metric space
serves as a ”simplification” filter, losing partial details.
E. Cross-Lingual Generalization (KSOF)
To evaluate the robustness of our approach beyond English,
we performed a zero-shot evaluation on the GermanKassel
State of Fluency (KSOF)dataset [23]. We filtered the KSOF test
set to include only clips with durations matching our training
window ( ≈3 seconds) and mapped the KSOF labels to the
SEP-28k ontology.
Table VI compares our zero-shot performance against two
baselines from [7]: the cross-corpus baseline (SEP-28k-E) and
the supervised topline (trained on KSOF).
The results are striking. Despite being trainedonly on
the English SEP-28k dataset, our zero-shot models match or
outperform theSupervised Topline (trained directly on KSOF)
on 4 out of 5 classes. Most notably, our RAC model achieves an
F1 of0.68on Blocks, surpassing the supervised model (0.60).
This suggests that our metric learning objective captures a
universal, language-independent representation of stuttering
blocks (e.g., airflow stoppage, tension) that generalizes better
than supervised training on a small target dataset. The only class
where the supervised model dominates is Interjection
(0.88 vs 0.62), which is expected as filler words are highly
language-specific (e.g., German “ ¨ah” vs. English “um”).
F . Ablation Study
To make sure our gains weren’t just noise, we systematically
dismantled the model. The Table VII tells the story and the
design choice’s effectiveness:
•Ablation 2 (No Retrieval):This is the critical component.
Removing the retrieval components drops F1 from 0.65
to 0.60. This 5 points drop shows that the memory bank
isn’t just a gimmick; it’s providing valuable signals and
information to the model.
•Ablation 3 (No Metric Learning):Using a generic index
(mean pooling the features across the 150 time steps to
get a vector) drops performance to 0.61. This confirmsthat our custom training stage (SetCon) is necessary to
build a meaningful semantic space.
•Ablation 4 (Acoustics Only):If we hide the neighbors’
logic (labels) from the model, it drops to 0.62. The model
needs to know the labels of its neighbors to make informed
decisions.
•Ablation 5 (End-to-End):We tried fine-tuning the whole
Wav2Vec 2.0 model. It scored 0.58. This shows that
on small, imbalanced datasets, massive fine-tuning often
leads to overfitting, whereas our approach of using frozen
features with a smart retrieval head is far more robust as
it implements seperation of concerns across the stages.
The ablation study shows that all three core components
of our design : 1) the retrieval mechanism itself, 2) the
organized metric-learning space, and 3) the multi-modal fusion
— contribute to the final performance.
VII. DISCUSSION
A. Principal Findings
Our experiments yielded three principal findings. Most
notably, the retrieval-augmented pipeline significantly out-
performs the Conformer baseline, proving that explicit non-
parametric memory is a viable strategy for combinatorial
tasks. Furthermore, the quality of this memory is crucial; a
generic acoustic index failed to match the performance of
our custom Jaccard-optimized metric space. Finally, the cross-
attention fusion mechanism proved vital, learning to weigh
the importance of the kneighbors to produce a context vector
more informative than simple averaging.
B. The High-Recall Phenomenon and Threshold Sensitivity
Our final model’s F1-score of 0.65 is a composite of 0.60
precision and 0.72 recall. This profile is a result of our empirical
threshold tuning. We evaluated thresholds from 0.1 to 0.9 on the
validation set and found that0.30provided the optimal balance
for our clinical objective: maximizing recall (sensitivity) while
maintaining acceptable precision. Higher thresholds (e.g., 0.5)
significantly degraded recall for rare classes like WordRep ,
which is unacceptable for a screening tool. Thus, 0.30 is not an
arbitrary default but a tuned hyperparameter chosen to prioritize
the detection of all potential disfluencies.
C. Cross-Lingual Generalization: Zero-Shot Evaluation on
KSOF
To assess the robustness of StutterFuse, we evaluated our
English-trained model (SEP-28k) directly on the German KSOF
dataset without any fine-tuning. We compare our results against
two key baselines from [23]: the direct zero-shot baseline and
the supervised topline. The results of this comparison are
detailed in Table VIII.
a) Matching the Supervised Upper Bound::Prior work
illustrates the difficulty of cross-lingual stuttering detection.
As shown in [23], a standard Wav2Vec2 model trained on
English (SEP-28k) and tested on German (KSOF) collapses to
an F1-score of0.10on Blocks, likely due to the domain shift
in acoustic environments and language phonetics.

11
TABLE VI: Zero-Shot Cross-Lingual Performance on KSOF (German). Comparison with Bayerl et al. [7] baselines.
Class Baseline (Trained on Sep28k-E) Supervised Topline (Trained on KSOF) RAC (Ours) StutterFuse (Ours)
(Bayerl et al.) (Bayerl et al.) (Mid-Fusion) (Late-Fusion)
Prolongation 0.44 0.57 0.57 0.56
Block 0.10 0.60 0.68 0.60
SoundRep 0.35 0.48 0.50 0.52
WordRep 0.23 0.18 0.16 0.20
Interjection 0.55 0.88 0.59 0.62
Weighted Avg - - 0.58 0.57
TABLE VII: the Ablation study on SEP-28k.
Configuration Weighted F1
1 Full Model (StutterFuse) 0.65
Ablations:
2 Conformer (No Retrieval) 0.60
3 RAC (No Metric Learning) 0.61
4 RAC (No Labels/Sims in Value) 0.62
TABLE VIII: Cross-Corpus Evaluation on KSOF (German). We compare StutterFuse against the direct baseline and the
supervised topline from [23].Key Result:StutterFuse not only outperforms the direct zero-shot baseline by 6 ×, but it also
matches the performance of the supervised model that was trained explicitly on KSOF data.
Model Direct Baseline StutterFuse (Ours) Supervised Topline
(Bayerl et al.) (Bayerl et al.)
Training Data SEP-28k (English) SEP-28k (English) KSOF (German)
Test Data KSOF (German) KSOF (German) KSOF (German)
Method Zero-Shot Zero-Shot Supervised
F1-Score (Blocks)
Block Detection 0.10 0.60 0.60
In contrast, StutterFuse bridges this gap entirely. Our
retrieval-augmented approach achieves an F1-score of0.60
on Blocks. Remarkably, thismatches the performance of the
fully supervised baseline(0.60), which was trained directly
on the German KSOF dataset. This suggests that retrieval-
augmentation effectively solves the domain adaptation problem
for physiological disfluencies, achieving supervised-level per-
formance without needing a single sample of German training
data.
b) Why Interjections Dropped::Performance on Inter-
jections was lower than the baseline ( 0.37 vs0.55). This is
expected: unlike blocks, interjections are strictly linguistic
fillers (e.g., English “um/uh” vs. German “ ¨ahm/also”). Since our
retrieval database (SEP-28k) contains only English fillers, the
model lacks the reference examples needed to retrieve German
fillers effectively. Similarly, WordRep (0.20) suffers because
identifying word repetitions often requires lexical or semantic
understanding that our purely acoustic model lacks. This
limitation essentially acts as a control experiment, confirming
that our model is indeed relying on acoustic similarity rather
than hallucinating.
D. Clinical Implications
From a clinical standpoint, StutterFuse’s ”high recall” be-
havior is a feature, not a bug. In a screening tool, a missed
diagnosis (false negative) is much worse than a false alarm. A
system that flags 90% of blocks allows a clinician to quickly
zoom in on potential problem areas rather than listening to the
whole recording.
Moreover, the retrieval mechanism adds a layer of trans-
parency. It doesn’t just say ”Block”; it can essentially say, ”Ithink this is a block because it sounds like these 10 other
confirmed blocks.” This ”explainability-by-example” can help
build trust with clinicians who might be skeptical of black-box
AI.
E. Limitations and Future Work
Despite its success, our model exhibits certain limita-
tions. First, theComputational Costis non-trivial; reasoning
with retrieval is significantly more expensive at inference
time than a simple classifier, as it requires a forward pass
through the embedder, a Faiss index search, and processing
by a larger fusion model. Second, regardingDataset Scale,
although we demonstrated strong cross-dataset and cross-
lingual generalization (FluencyBank, KSOF), the limited size
of available stuttering corpora remains a constraint. Unlike
general speech recognition, the scarcity of large-scale, high-
quality labeled data restricts the development of even more
robust, foundational stuttering models. Future work should
investigate identifying optimal retrieval thresholds, exploring
model distillation to create lighter, non-retrieval student models
that mimic StutterFuse’s performance, and adapting the pipeline
for low-latency streaming applications.
VIII. CONCLUSION
This study introducedStutterFuse, a retrieval-augmented
framework designed to tackle the twin challenges of multi-
label stuttering detection: combinatorial complexity and long-
tail class imbalance. We pinpointed a ”Modality Collapse” in
standard retrieval architectures—an ”Echo Chamber” where
early fusion spikes recall but hurts precision. By implementing
our Jaccard-Weighted Metric Learning objective (SetCon)

12
alongside a Gated Mixture-of-Experts fusion strategy, we
effectively solve this problem.
Our results on SEP-28k show that StutterFuse hits a weighted
F1-score of0.65, clearly outpacing strong Conformer baselines
and nearing the limit set by inter-annotator agreement. Even
more telling is our zero-shot evaluation on the German
KSOF dataset, where the model generalized surprisingly well,
matching fully supervised models. These findings suggest
that retrieval-augmentation isn’t just a performance hack; it’s
a viable path toward robust, interpretable systems that can
simplify their decisions by pointing to real-world examples.
Additionally, this architecture supports low-cost adaptability;
new examples can be added to the knowledge base store instead
of retraining the model, which is a less intensive way to see
on par performance gains. While StutterFuse pushes the state-
of-the-art, the computational cost of the retrieval step remains
a bottleneck for mobile deployment. Future iterations would
explore vector quantization or distillation to retain this accuracy
without the heavy inference penalty.
APPENDIXA
MODELARCHITECTUREDETAILS
We provide detailed layer-by-layer specifications of our
models to ensure reproducibility.
A. Hyperparameters and Training Details
Table IX lists the specific hyperparameters used for training
the Embedder (Stage 2) and the Classifier (Stage 3).
TABLE IX: Hyperparameters for StutterFuse training.
Parameter Stage 2 (Embedder) Stage 3 (Classifier)
Optimizer Adam AdamW
Learning Rate 1e−4 2e−5
Weight Decay 0.0 5e−4
Batch Size 4096 (Global) 128 (Global)
Epochs 20 50 (Early Stopping)
Loss Function SetCon (τ= 0.1) BCE (Label Smooth=0.1)
Dropout 0.0 0.3 (Conformer), 0.5 (Dense)
Hardware TPU v5e-8 TPU v5e-8
B. Stuttering Event Ontology
Table X defines the disfluency types used in this study,
derived from the SEP-28k ontology.
TABLE X: Definitions of Stuttering Events (SEP-28k Ontol-
ogy).
Label Definition
Block Inaudible or fixed articulatory posture; airflow is stopped.
Prolongation Unnatural lengthening of a sound (e.g., “Mmmm-my”).
SoundRep Repetition of a sound or syllable (e.g., “S-S-Sound”).
WordRep Repetition of a whole word (e.g., “Word Word”).
Interjection Filler words or sounds (e.g., “Um”, “Uh”, “Like”).
We include the model.summary() outputs from our logs
for full transparency and reproducibility.C. Stage 2: SetCon Embedder
This is the model (Table XI), which wraps the embedder in
a siamese structure. The key is the trainable parameter count
of the embedder itself.
TABLE XI: Stage 2: SetCon Embedder (BiGRU-Attention)
Layer (type) Output Shape Param # Connected to
input layer (Input) (None, 150, 1024) 0 -
input layer 1 (Input) (None, 150, 1024) 0 -
input layer 2 (Input) (None, 150, 1024) 0 -
embedder (None, 1024) 1,970,689 input layer[0][0]
(Functional) input layer 1[0][0]
input layer 2[0][0]
lambda 1 (Lambda) (None, 3, 1024) 0 embedder...
Total params:1,970,689 (Trainable: 1,970,689)
D. Stage 3: StutterFuse RAC Classifier
This is the final classification model (Table XII). Note the
large parameter counts from the shared conformer encoder and
the cross-attention layer.
TABLE XII: Stage 3: StutterFuse RAC Classifier (Mid-Fusion)
Layer (type) Output Shape Param # Connected to
input test seq (None, 150, 1024) 0 -
input neighbor vecs (None, 5, 1024) 0 -
input neighbor labels (None, 5, 5) 0 -
input neighbor sims (None, 5) 0 -
shared conformer enc (None, 1024) 21,051,400 input test seq[0]...
neighbor projection (None, 5, 1024) 1,049,600 input neighbor vecs
label embedder (None, 5, 16) 96 input neighbor labels
sim expander (None, 5, 1) 0 input neighbor sims
value concatenation (None, 5, 1041) 0 neighbor projection...
label embedder...
sim expander...
cross attention (None, 1, 1024) 4,215,808 neighbor projection...
(MultiHeadAttention) reshape...
value concatenation...
concatenate (None, 2048) 0 shared conformer...
reshape 1...
dense 12 (Dense) (None, 512) 1,049,088 concatenate
dense 13 (Dense) (None, 256) 131,328 dropout 9
output (Dense) (None, 5) 1,285 dense 13
Total params:27,498,605 (Trainable: 27,490,413)
E. Stage 3: StutterFuse Late-Fusion (Gated Experts)
We provide the architecture details for the individual experts
and the fusion mechanism: the Audio-Only Expert (Table XIII),
the Retrieval Expert (Table XIV), and the final Gated Fusion
System (Table XV).

13
TABLE XIII: Audio-Only Expert Architecture
Layer (type) Output Shape Param # Connected to
Input & Augmentation
input audio (Input) (None, 150, 1024) 0 -
spatial dropout1d (None, 150, 1024) 0 input audio
gaussian noise (None, 150, 1024) 0 spatial dropout1d
layer normalization (None, 150, 1024) 2,048 gaussian noise
Conformer Block 1
conv1d (Conv1D) (None, 150, 2048) 2,099,200 layer normalization
depthwise conv1d (None, 150, 2048) 8,192 conv1d
batch normalization (None, 150, 2048) 8,192 depthwise conv1d
activation (None, 150, 2048) 0 batch normalization
conv1d 1 (Conv1D) (None, 150, 1024) 2,098,176 activation
add (Add) (None, 150, 1024) 0 gaussian noise, conv1d 1
multi head attention (None, 150, 1024) 4,198,400 add
add 1 (Add) (None, 150, 1024) 0 add, multi head attention
dense (Dense) (None, 150, 512) 524,800 add 1
dense 1 (Dense) (None, 150, 1024) 525,312 dense
add 2 (Add) (None, 150, 1024) 0 add 1, dense 1
Conformer Block 2
conv1d 2 (Conv1D) (None, 150, 2048) 2,099,200 add 2
depthwise conv1d 1 (None, 150, 2048) 8,192 conv1d 2
batch normalization 1 (None, 150, 2048) 8,192 depthwise conv1d 1
activation 1 (None, 150, 2048) 0 batch normalization 1
conv1d 3 (Conv1D) (None, 150, 1024) 2,098,176 activation 1
add 3 (Add) (None, 150, 1024) 0 add 2, conv1d 3
multi head attention 1 (None, 150, 1024) 4,198,400 add 3
add 4 (Add) (None, 150, 1024) 0 add 3, multi head attention 1
dense 2 (Dense) (None, 150, 512) 524,800 add 4
dense 3 (Dense) (None, 150, 1024) 525,312 dense 2
add 5 (Add) (None, 150, 1024) 0 add 4, dense 3
Output Head
global average pooling (None, 1024) 0 add 5
audio features (Dense) (None, 256) 262,400 global average pooling
audio output (Dense) (None, 5) 1,285 audio features
Total params:19,200,517 (Trainable: 19,192,325)
TABLE XIV: Expert B (Retrieval Stream)
Layer (type) Output Shape Param # Connected to
input vecs (None, 10, 1024) 0 -
dense (Dense) (None, 10, 256) 262,400 input vecs
global average pool (None, 256) 0 dense
dense 1 (Dense) (None, 256) 65,792 global average pool
retrieval features (None, 128) 32,896 dropout
retrieval output (None, 5) 645 retrieval features
Total params:361,733 (Trainable: 361,733)
TABLE XV: Gated Fusion System
Layer (type) Output Shape Param # Connected to
Expert A (Functional) (None, 256) 21,051,400 input audio
Expert B (Functional) (None, 128) 361,088 input vecs
concatenate (None, 384) 0 Expert A, Expert B
trust gate (Dense) (None, 128) 49,280 concatenate
multiply (Multiply) (None, 128) 0 Expert B, trust gate
concatenate 1 (None, 384) 0 Expert A, multiply
dense 6 (Dense) (None, 128) 49,280 concatenate 1
final output (Dense) (None, 5) 645 dropout 14
Total params:21,511,693 (Trainable: 99,205)
REFERENCES
[1]American Psychiatric Association, 2013.Diagnostic and Statistical
Manual of Mental Disorders (DSM-5). American Psychiatric Publishing.
[2]Hsu, W.-N., Bolte, B., Tsai, Y .-H. H., Lakhotia, K., Salakhutdinov, R.,
Mohamed, A., 2021. HuBERT: Self-Supervised Speech Representation
Learning by Masked Prediction of Hidden Units.IEEE/ACM Trans.
Audio, Speech, Lang. Process., 29, 3451–3460.
[3]Baevski, A., Hsu, W.-N., Xu, Q., Babu, A., Gu, J., Auli, M., 2022.
Data2vec: A General Framework for Self-supervised Learning in Speech,
Vision and Language. In:Proc. ICML.
[4]Johnson, J., Douze, M., J ´egou, H., 2019. Billion-scale similarity search
with GPUs.IEEE Trans. Big Data, 7(3), 535–547.
[5]Lea, C., Mitra, V ., Joshi, A., Kajarekar, S., Bigham, J.P., 2021. SEP-28k:
A dataset for stuttering event detection from podcasts with people who
stutter. In:Proc. ICASSP.[6]Baevski, A., Zhou, Y ., Mohamed, A., Auli, M., 2020. Wav2vec 2.0: A
framework for self-supervised learning of speech representations. In:
Proc. NeurIPS.
[7]Bayerl, S. P., Wagner, D., Baumann, I., H ¨onig, F., Bocklet, T., N ¨oth, E.,
Riedhammer, K., 2023. A stutter seldom comes alone – Cross-corpus
stuttering detection as a multi-label problem. In:Proc. Interspeech.
[8]Chee, K. X., Tan, S.-Y ., Lee, T. H., 2016. Deep learning for automatic
detection of stuttering dysfluencies. In:Proc. ICASSP.
[9]Einarsdottir, J., Ingham, R., 2019. Automatic classification of stuttering
disfluencies using recurrent neural networks. In:Proc. SLPAT.
[10] Gulati, A., Qin, J., Chiu, C.-C., Parmar, N., Zhang, Y ., Yu, J., Han, W.,
Wang, S., Zhang, Z., Wu, Y ., 2020. Conformer: Convolution-augmented
transformer for speech recognition. In:Proc. Interspeech.
[11] Howell, P., Au-Yeung, J., Sackin, S., 2009. FluencyBank: A repository
for the study of fluency and disfluency across languages.Speech
Communication, 51(6), 484–496.
[12] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V ., Nogueira, G.,
... & Kiela, D., 2020. Retrieval-augmented generation for knowledge-
intensive nlp tasks. In:Proc. NeurIPS.
[13] Ma, J., Lee, A., Wen, C., Narayanan, S., 2018. Using Deep Neural
Networks to Detect Stuttering Events from Speech. In:Proc. Interspeech.
[14] Rudzicz, F., 2011. Perceptual and acoustic evidence for speaker-dependent
stuttering patterns.Journal of Fluency Disorders, 36(4), 298–318.
[15] Schroff, F., Kalenichenko, D., Philbin, J., 2015. FaceNet: A unified
embedding for face recognition and clustering. In:Proc. CVPR.
[16] Sheikh, S. A., Sahidullah, M., Hirsch, F., Ouni, S., 2021. StutterNet:
Stuttering Detection Using Time Delay Neural Network. In:Proc.
EUSIPCO.
[17] Takashima, Y ., Shibata, K., 2022. Fine-tuning wav2vec2 for stuttering
detection with limited data. In:Proc. Interspeech.
[18] Vinod, A., Sharma, D., Kumar, R., 2015. Automatic detection of stuttered
speech using MFCC features.International Journal of Speech Technology,
18(4), 495–502.
[19] Yairi, E., Ambrose, N. G., 2013. Epidemiology of stuttering: 21st century
advances.Journal of Fluency Disorders, 38(2), 66–87.
[20] Zayats, V ., Ostendorf, M., Hajishirzi, H., 2016. Disfluency detection
using recurrent neural networks. In:Proc. NAACL-HLT.
[21] Kourkounakis, T., Hajavi, A., Etemad, A., 2020. FluentNet: End-to-End
Detection of Speech Disfluency with Deep Learning.arXiv preprint
arXiv:2002.06649.
[22] Kourkounakis, T., Hajavi, A., Etemad, A., 2020. Detecting Multiple
Speech Disfluencies using a Deep Residual Network with Bidirectional
LSTM. In:Proc. ICASSP.
[23] Bayerl, S. P., et al., 2022. KSoF: The Kassel State of Fluency Dataset –
A Therapy Centered Dataset of Stuttering. In:Proc. LREC.
[24] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
A. N., Kaiser, Ł., Polosukhin, I., 2017. Attention is all you need. In:
Proc. NeurIPS.
[25] Kingma, D. P., Ba, J., 2014. Adam: A method for stochastic optimization.
arXiv preprint arXiv:1412.6980.
[26] Hoffer, E., Ailon, N., 2015. Deep metric learning using triplet network.
In:International Workshop on Similarity-Based Pattern Recognition.
[27] Hermans, A., Beyer, L., Leibe, B., 2017. In defense of the triplet loss
for person re-identification.arXiv preprint arXiv:1703.07737.
[28] Jansen, A., Plakal, M., Pandya, R., Ellis, D. P., Hershey, S., Liu, J.,
Moore, R. C., Saurous, R. A., 2018. Unsupervised learning of semantic
audio representations. In:Proc. ICASSP.
[29] Lin, N., Qin, G., Wang, G., Zhou, D., Yang, A., 2023. An Effective
Deployment of Contrastive Learning in Multi-label Text Classification.
In:Proc. ACL.
[30] Lin, T. Y ., Goyal, P., Girshick, R., He, K., Doll ´ar, P., 2017. Focal loss
for dense object detection. In:Proc. ICCV.