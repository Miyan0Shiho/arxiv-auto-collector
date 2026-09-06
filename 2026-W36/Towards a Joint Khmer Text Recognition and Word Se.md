# Towards a Joint Khmer Text Recognition and Word Segmentation

**Authors**: Marry Kong, Rina Buoy, Sovisal Chenda, Nguonly Taing, Masakazu Iwamura, Koichi Kise

**Published**: 2026-08-31 03:53:25

**PDF URL**: [https://arxiv.org/pdf/2608.30213v1](https://arxiv.org/pdf/2608.30213v1)

## Abstract
Text recognition, or extracting electronic text from document images, has been indispensable for knowledge retrieval tasks, such as retrieval-augmented generation (RAG). For Khmer, extracted text is subject to an extra word segmentation step, as Khmer does not use any visible word delimiters to denote word boundaries. Thus, a recognition-then-segmentation pipeline for Khmer requires two separate sequential models; this is not only error-prone but also adds significant latency for large-scale document processing. This paper proposes a novel joint Khmer text recognition and word segmentation framework in a unified model. The proposed model, using a connectionist-temporal-classification (CTC) decoder for fast, parallel decoding, can be instructed to recognize Khmer text with ($b=1$) and without ($b=0$) word segmentation. Experimental results on different benchmark datasets of different document modalities (document, scene, and handwritten images) show that the proposed model can not only recognize characters in document images but also locate word boundaries, removing the need for an extra word segmentation step in a conventional sequential pipeline.

## Full Text


<!-- PDF content starts -->

Towards a Joint Khmer Text Recognition and
Word Segmentation
Marry Kong1, Rina Buoy1,2[0000−0002−6960−4262], Sovisal Chenda1, Nguonly
Taing1, Masakazu Iwamura2, and Koichi Kise2
1Techo Startup Center, Ministry of Economy and Finance, Phnom Penh, Cambodia
2Osaka Metropolitan University, Osaka, Japan
Abstract.Text recognition, or extracting electronic text from docu-
ment images, has been indispensable for knowledge retrieval tasks, such
as retrieval-augmented generation (RAG). For Khmer, extracted text is
subject to an extra word segmentation step, as Khmer does not use any
visible word delimiters to denote word boundaries. Thus, a recognition-
then-segmentation pipeline for Khmer requires two separate sequential
models; this is not only error-prone but also adds significant latency
for large-scale document processing. This paper proposes a novel joint
Khmer text recognition and word segmentation framework in a unified
model.Theproposedmodel,usingaconnectionist-temporal-classification
(CTC) decoder for fast, parallel decoding, can be instructed to recognize
Khmer text with (b= 1) and without (b= 0) word segmentation. Ex-
perimental results on different benchmark datasets of different document
modalities (document, scene, and handwritten images) show that the
proposed model can not only recognize characters in document images
but also locate word boundaries, removing the need for an extra word
segmentation step in a conventional sequential pipeline.
Keywords:Text recognition·word segmentation·low-resource lan-
guage
1 Introduction
Central to knowledge retrieval tasks, such as retrieval-augmented generation
(RAG) [15], is text recognition, which is the process of extracting electronic
text from document images. For Khmer, an extra word segmentation step is
often needed to segment extracted text into words in order to facilitate a hybrid
lexical-vector search or apply any post-recognition spell-checking, as Khmer does
not use any visible word delimiters in its writing system [7].
A common pipeline for Khmer text recognition and word segmentation re-
quires two separate models applied in sequence. Separately, Khmer text recogni-
tion and word segmentation are relatively well-explored. Many text recognition
methods [3,4,13,17,18,23] have been proposed for the accurate recognition of
Khmer characters. Similarly, the task of Khmer word segmentation has also been
addressed in multiple previous works [1,7,11]. However, chaining both models
arXiv:2608.30213v1  [cs.CV]  31 Aug 2026

2 Kong et al.
Fig. 1:The multi-layered layout of Khmer text, highlighting complex character stack-
ingandligatures.Green:baseconsonant.Blue:consonantsubscript.Orange:dependent
vowel. Purple: diacritic. Best viewed in color.
together in a sequential pipeline is not only error-prone but also introduces sig-
nificant latency when processing a large batch of document images in industrial
settings. Furthermore, no previous research has focused on addressing joint text
recognition and word segmentation, at least for Khmer text.
In this paper, we propose the first jointKhmertextrecognition andword
segmentation (KTRWS) approach that performs both recognition and segmen-
tation of Khmer text in a single model. The proposed approach is fast, as it is
based on a connectionist-temporal-classification [10] (CTC) decoder for parallel
character recognition. In addition, the proposed model can be instructed to rec-
ognize Khmer text with (b= 1) and without (b= 0) word segmentation. Thus,
there is no need for an extra word segmentation step, making the proposed ap-
proach particularly suitable for real-world, low-latency, large-scale applications.
Since existing Khmer text recognition datasets do not have word boundary an-
notations, we synthetically generate a new open dataset3of Khmer text line
images with word boundary annotations for model training. Our contributions
can be summarized as follows:
1. We propose the KTRWS method, which performs joint Khmer text recog-
nition and word segmentation in a single model, along with the first Khmer
textline recognition dataset with word boundary annotations.
2. By performing joint text recognition and word segmentation with a parallel
CTC decoder, the proposed method achieves significantly lower latency for
this combined task.
3. Experimental results show that the proposed method can not only recognize
Khmer characters accurately but can also eliminate the need for an extra
word segmentation step.
2 Related Work
Khmer Text RecognitionKhmer script, as shown in Figure 1, is one of the
most complex writing systems, characterized by multi-layered character stack-
ing, absence of visible word delimiters, and a large inventory of characters, in-
cluding 33 consonants, 37 dependent and independent vowels, and eight diacrit-
ics [3,23]. Despite some recent research progress, Khmer text recognition is a
3Dataset repository - removed for review

Abbreviated paper title 3
relatively under-explored yet maturing field because of limited public training
and benchmark datasets. Early Khmer text recognition methods [8,14,21] relied
on classical feature extractors, such as wavelet descriptors, and classical machine
learning methods, such as support vector machines (SVM). Such methods often
require explicit character segmentation and are unable to generalize robustly.
Deep-learning-based approaches have been proven to achieve superior perfor-
mance across document modalities (i.e., scene vs. document). Keo et al. [12]
adopted two prominent Latin-based methods for the task of Khmer scene text
recognition on their published dataset using a convolutional feature extractor
and two different decoding mechanisms, including a CTC-based decoder and an
attention-based decoder. Similarly, Buoy et al. [5] addressed the task of Khmer
document text recognition using a sequence-to-sequence attentional network.
On the other hand, Valy et al. [23,24] proposed multiple deep-learning-based
approaches for recognizing Khmer characters and words on historical palm leaf
manuscript documents.
Variousrecentstate-of-the-art(SoTA)methods[2,3,13]forKhmertextrecog-
nition have incorporated Transformers [25] for both feature extraction and char-
acter decoding, and proposed a Khmer-native tokenization scheme called Khmer
characterclusters(KCC)insteadofconventionalcharacter-leveltokenization.As
a result, character error rates (CERs) across public benchmark datasets, includ-
ing scene, document, and handwritten images, have been significantly reduced.
Khmer Word SegmentationSince Khmer does not use any visible delimiters
to denote word boundaries, word segmentation is a necessary prior step for any
downstream natural language processing (NLP) tasks, such as part-of-speech
(POS) tagging, named entity recognition (NER), spell checking, lexical search,
and so on. Nonetheless, there is no standard definition of a word in Khmer, nor
a standard corpus for this task [11]. As a result, researchers often construct their
own corpora and train their own models.
The early work [1] on Khmer word segmentation was based on a dictionary-
based bidirectional maximal matching (BiMM) approach. Despite being simple
(requiring no training) and lightweight, the BiMM approach is unable to han-
dle out-of-vocabulary (OOV) words. To address this limitation, Chea et al. [7]
proposed a statistical machine learning approach using a conditional random
field (CRF) algorithm. The authors manually constructed training and evalu-
ation datasets using word-level and compound-level (i.e., prefixes and suffixes)
annotations and trained the CRF-based model. The CRF-based model achieved
an accuracy of 98.5% and has become a de facto baseline model.
In addition, Buoy et al. [6] proposed a neural Khmer word segmentation
network using a bidirectional long short-term memory (BiLSTM) architecture
to improve contextual information modeling. The authors proposed two variants
using character-level and character-cluster-level tokenization schemes. Recogniz-
ing that word segmentation and POS tagging can be jointly trained, Kaing et
al. [11] proposed a joint word segmentation and POS tagging method.

4 Kong et al.
CTC Decoder
Input T ext ImageDecoded Khmer T ext
A Binary Flag to Output
Word BoundaryBoundary ProjectorModality-A ware Feature
Selector
Visual Encoder
Text Decoder
Fig. 2:The proposed KTRWS framework. Best viewed in color.
In summary, while both word segmentation and text recognition for Khmer
have significantly evolved along their respective, seemingly different trajectories,
both tasks meet and complement each other in downstream applications, such as
knowledge retrieval, where electronic texts are recognized from document images
and words are segmented for ingestion and retrieval. Thus, unifying both tasks
within a single model can lead to significant latency reduction, which is of great
importance for practical applications. To this end, this paper proposes a novel
KTRWS approach.
3 The Proposed Method
As shown in Figure 2, our proposed KTRWS framework comprises four key
components: a word boundary projector, a visual encoder, a modality-aware
feature selector (MAFS), and a text decoder. The boundary projector embeds
andprojectsabinaryflag(b)indicatingwhethertooutputwordboundaries.The
visual encoder extracts both visual and sequential features from an input image.
The MAFS fuses word boundary projections and visual-temporal features, and
adapts according to a specific modality (i.e., scene, handwritten, and document
text). Finally, the text decoder uses a CTC decoder to output Khmer tokens
with (b= 1) or without word boundary tokens (b= 0).
Word Boundary Projector ModuleThis module embeds and projects a
binary flag indicating whether to output word boundaries, and returns two pro-
jection vectors,β∈Rdandγ∈Rd. These projection vectors are fused with the
adapted features from the encoder before Khmer character decoding. With this
design, the model can be instructed to output word boundaries or not during in-
ference. Mathematically, the word boundary projector module can be expressed
as

Abbreviated paper title 5
Eb= LINEARLAYER(b)(1)
β= LINEARLAYER(E b)(2)
γ= LINEARLAYER(E b), (3)
whereb∈ {0,1}is a binary variable indicating whether to output word bound-
aries,LINEARLAYERis a dense neural network layer,E b∈Rdis an embedding
vector ofb, andβandγare the resulting projection vectors.
Visual EncoderThe visual encoder is designed to extract the visual-temporal
features required for accurate character recognition. For a given RGB image (I),
our visual encoder consists of a base convolutional network for two-dimensional
(2D) visual features (F∈Rw′×h′×d) and a Transformer-based encoder network
for visual-temporal features (G∈Rw′×h′×d). Since the CTC decoder expects
a sequence of one-dimensional (1D) features (G 1D∈Rw′×d) along the width
direction, the extracted visual-temporal features are averaged over the height
direction. Mathematically, our visual encoder can be expressed as
F= CNN(I)(4)
G= TR ENC(F)(5)
G1D= HPOOLING(G), (6)
whereCNN,TR ENC,andHPOOLarethebaseconvolutionalnetwork,theTrans-
former encoder network, and the height-averaging operator, respectively;Iis an
input RGB image; andd(= 512) is the feature dimension.
Modality-Aware Feature SelectorTo enable the proposed framework to
robustly recognize Khmer text across different modalities (e.g., scene, handwrit-
ten, and document), we modify the modality-aware feature selector [13] (MAFS)
to incorporate the word boundary projections (i.e.,βandγ). Specifically, this
module computes an average vectorz∈Rdusing theGPOOLoperator, given
G1D. The module then usesROUTERto compute a distribution overnmodal-
ity sources (r∈Rn), from which modality-adapted features (H 1D∈Rd×n) from
ADAPTERare marginalized to yield the final features (U 1D∈Rw′×d), which
arethensubjecttofeature-wiselinearmodulation[20](FiLM),scaledandshifted
byγandβ, for output token generation. Mathematically, the modified MAFS
can be expressed as
z= GPOOLING(G 1D)(7)
r= ROUTER(z)(8)
H1D,k = ADAPTER k(G1D)(9)
U1D=γ⊙(H 1D·r) +β, (10)

6 Kong et al.
whereU 1Dis the input feature map for the CTC decoder to generate output
Khmer tokens. We adopt the same tokenization scheme as [2,13], which operates
at the character-cluster level rather than the character level.
Text DecoderAs shown in Figure 2, we utilize a CTC decoder for parallel
Khmer token generation. Given the input feature mapU 1D∈Rw′×dfrom the
MAFS module, the decoder projects each of thew′frames onto the augmented
vocabularyV′=V∪{∅}∪{u200b},whereVisthesetofcharacter-clustertokens,
∅is the blank symbol, andu200bis a word boundary marker token (invisible
when rendered). A softmax classifier produces a per-frame distribution as
pi= SOFTMAX(U 1D,i), (11)
wherep i(c)denotes the probability of emitting tokencat framei.
For a given target sequencey, aL CTCloss is given by
LCTC=−X
(I, y)∈Dlogp(y|U 1D), (14)
wherelogp(y|U 1D)is obtained by marginalizing over all alignments that col-
lapsetoy.Attesttime,weusegreedy(best-path)decoding,whichapproximates
the most probable sequence by selecting the most likely token independently at
each frame and collapsing repeated tokens and∅.
ˆπi= arg max
c∈V′pi(c), i= 1, . . . , w′(12)
AlternativeDesignforWordBoundaryProjectorModuleAlternatively,
the boundary embedding vector can be fused via gating. Mathematically, the
word boundary projector with gating can be expressed as
α=σ(LINEARLAYER(E b))(13)
p= LINEARLAYER(E b)(14)
U1D=α⊙(H 1D·r) + (1−α)⊙p, (15)
whereb∈ {0,1}is a binary variable indicating whether to output word bound-
aries,LINEARLAYERis a dense neural network layer,E b∈Rdis an embedding
vector ofb, andσis a sigmoid gating function applied element-wise to the em-
bedding vector.

Abbreviated paper title 7
Fig. 3:Sample preview images from the existing datasets.
4 Datasets & Experimental Setup
DatasetsPriorKhmertextrecognitionmethodshavereliedlargelyonsynthetic
training data, which is convenient to generate at scale, whereas real data for
the scene and handwritten modalities remains scarce. We adopt the publicly
available datasets listed below, comprising real and synthetic Khmer document,
scene,andhandwrittentextimages.Asummaryofalldatasetsandsomepreview
images are given in Table 1 and Figure 3, respectively.
Buoy et al.: [3]This dataset comprises about 2.8M synthetic textlines (1.5M
document, 1.3M scene) rendered with 11 Khmer fonts. This dataset features
document and scene modalities.
KHOB [9]:Thisdatasetcomprises1,318textlinesmanuallycroppedfromscanned
low-resolution Khmer PDF documents. This dataset is used for document-
modality evaluation.
SynthText [27]:This dataset comprises 70,000 textlines extracted from 10,000
syntheticKhmeridentitycardimages.Thisdatasetfeaturesdocumentmodal-
ity.
HierText [16]:This dataset comprises a large-scale Latin printed/handwritten
dataset yielding about 518,726 textline images after cropping and remov-
ing vertical texts. This dataset is used to learn Latin visual representations
(scene and handwritten modalities).
KhmerST [17]:This dataset comprises 3,022 real Khmer scene text images
(mostly logos and billboards) with diverse artistic fonts and challenging
imaging conditions; This dataset is used for scene-modality evaluation.
WildKhmerST [18]:This dataset comprises 29,601 textlines from 10,000 real
images across Cambodia, including artistic, blurred, low-light, curved, and
occluded text. This dataset is used for scene modality.
KH [26]:This dataset comprises about 4.2k Khmer and Latin textlines (3,991
train, 211 eval), both handwritten and printed. This dataset features docu-
ment and handwritten modalities.
GKST [13]:This dataset comprises 4,221 smartphone-captured Khmer scene
text images (4,009 train, 212 eval), annotated from general scenes rather
than focused close-ups. This dataset features scene modality.

8 Kong et al.
Table 1:Summary of datasets. Size: train/eval (– if none). Purpose: Tr = training
(D), Ad = adapting (S&H), Ev = evaluation. Modality: D = document, S = scene,
H = handwritten.
Dataset Size Purpose Modality
Buoy et al. 2.8M/– Tr D, S
KHOB –/1,318 Ad, Ev D
SynthText 70k/– Tr D
HierText 518,726/– Tr S, H
KhmerST –/3,022 Ad, Ev S
WildKhmerST 29,601/– Ad S
KH 3,991/211 Ad, Ev D, H
GKST 4,009/212 Ad, Ev S
KHT 13,457/711 Ad, Ev H
KHT [13]:Thisdatasetcomprises14,168Khmerhandwrittentextimages(13,457
train, 711 eval) from sources such as birth certificates, exam papers, and
notes. This dataset features handwritten modality.
Since existing datasets do not provide word boundary annotations, we syn-
thetically generate 60,000 text line images with word boundary annotations.
To achieve this, we begin by training a Khmer Transformer-based model for the
word segmentation task using the manually labelled dataset provided by Chea et
al. [7]. The textlines, sourced mainly from news articles and segmented by an in-
visible word boundary marker (i.e.,u200b), are rendered synthetically as images
for model training. Consequently, the rendered text images are not aesthetically
affected by the extra word boundary information. Some preview images from
this new dataset with word boundary annotations are provided in Figures 4.
(a)English translation:The leaders of the United States and Israel discussed while.
(b)English translation:The US President Joe Biden (left) and the Israeli Prime Minister
Benjamin Netanyahu (.
(c)English translation:The spread of conflict.
Fig. 4:Sample generated images with word boundary annotations (red bar) from our
new dataset.

Abbreviated paper title 9
Table 2:The visual encoder’s complexity and model size. Params: model parameters.
Flops: floating operations on an input of32×116pixels.
Name Params. Flops
ResNet backbone (visual features) 13.0M 3.2G
Transformers encoder (sequential features) 9.5M 2.2G
MAFS 0.80M 0.15G
Experimental Setup & Training StrategyThe baseCNNfeature extractor
is composed of six sequential ResNet blocks with channel dimensions progres-
sively increasing from 32 to 512, where each block comprises 1×1 and/or 3×3
convolutions repeated multiple times. Spatial downsampling by a factor of(2,2)
is performed at ResNet Blocks 1 and 4, while the remaining blocks maintain the
spatial resolution. TheTR ENCnetwork is configured with an embedding dimen-
sion (d) of 512, three stacked encoder layers, eight attention heads, a dropout
rate of 0.1, and a feed-forward network dimension of 2048 (4×d). The model
size and computational complexity of the visual encoder are reported in Table 2.
Although there are three discrete modalities, document, scene, and handwritten,
real images rarely belong to only one modality. Instead, they can be represented
as points within a simplex spanning these three modalities. For example, a text
image may be simultaneously both handwritten and scene text. By default, we
set this hyperparameter (n) to five, providing additional capacity to model hy-
brid or transitional cases [13].
Following [13], we group these datasets by primary modality:Document (D)
for large-scale printed data (Buoy et al., SynthText, HierText);Scene & Hand-
written (S&H)for real data (WildKhmerST, the KH training set, GKST, and
KHT); and anEvaluationgroup comprising KhmerST (S), KHOB (D), the KH
evaluation set (D&H), GKST (S) and KHT (H).
The KTRWS models were trained in two phases: a general training phase
followed by a modality-adapting phase. In thegeneral training phase, the mod-
els were first trained on the large-scale document (D group) datasets to learn
robust visual representations of Khmer and Latin characters. A cyclic learning
rate schedule was employed, with a minimum value of10−5and a maximum of
10−4. This phase lasted five full epochs with a batch size of 32 images. In the
modality-adapting phase, the trained models then underwent adaptation on the
real scene and handwritten (S&H group) datasets together with our newly an-
notated word-boundary dataset for 50 epochs, using a lower cyclic learning rate
schedule with a minimum of10−6and a maximum of10−5. In both phases, the
Adam optimizer was used with a gradient clipping value of 50. To prevent the
models from underperforming on printed document text during this phase, we
sampled an equal number of document images and mixed them with the S&H
images. As a result, the same model can both recognize Khmer text across the
document, scene, and handwritten modalities and, when conditioned onb= 1,
additionally output word boundaries.

10 Kong et al.
Table 3:Charactererrorrates(CER)ontheevaluationdatasetsofdifferentmodalities
(D,S,&H).*:model-specifictokenizer.Char.:character.KCC:Khmercharactercluster.
Seg.:jointcharacterrecognitionandwordsegmentation.Bold:best.Italic:secondbest.
Dec. Tok. Seg. KHOB(D) KhmerST(S) KH(D,H) GKST(S) KHT(H)
Surya-OCR [19] Tr. * No 17.69 43.21 – – –
Buoy et al. [3] Tr. Char. No 3.03 – – – –
Nom et al. [17] Tr. * No – 17.00 – – –
Soy et al. [26] Tr. * No – – 17.00 – –
Buoy et al. [2] Tr. KCC No 2.13 7.01 – – –
Tesseract-OCR [22] CTC Char. No 9.19 40.96 – – –
Buoy et al. [4] CTC KCC No 2.33 – – – –
UKTR [13] CTC KCC No 2.463.02 5.894.419.52
KTRWS(b= 0) CTC KCC No1.75 2.77 5.784.518.91
KTRWS(b= 1) CTC KCC Yes1.793.04 6.06 5.01 9.64
5 Results & Discussion
Recognition Performance of the KTRWS ModelWe begin by comparing
the recognition performance of the proposed method in terms of character error
rate (CER) across the evaluation datasets of different modalities. Since the eval-
uation datasets do not have word boundary annotations, CERs are computed
by excluding theu200bword boundary marker token.
Table 3 provides the recognition accuracy comparisons of the proposed mod-
els, with and without word boundary outputs, against the SoTA methods. Com-
pared with existing methods, the proposed models either outperform on the
KHOB dataset or are competitive on the other datasets (KhmerST, KH, GKST,
and KHT). Compared with the recent SoTA UKTR model using the same
CTC decoder, ourKTRWS(b= 0) model without outputting word bound-
aries achieves improved CERs on all datasets except GKST. Specifically, our
KTRWS(b= 0) model (i.e., no word boundary outputs) obtained CERs of
1.75%, 2.77%, 5.78%, 4.51%, and 8.91% on KHOB, KhmerST, KH, GKST, and
KHT, respectively, versus 2.46%, 3.02%, 5.89%, 4.41%, and 9.52% by the CTC-
based UKTR model. When outputting word boundaries, ourKTRWS(b= 1)
model remains competitive. Specifically, ourKTRWS(b= 1) model obtained
CERsof1.79%,3.04%,6.06%,5.01%,and9.64%.ThemarginalincreasesinCER
withb= 1are expected, as the model needs to generate both normal Khmer
tokens and word boundary tokens, which increases the probability of making
errors. On the other hand, the significant CER improvements on the KHOB
dataset can be attributed to the fact that the new dataset introduced in this
study is more aligned with the nature of the KHOB dataset (i.e., long document
textlines).
As shown in Figure 5, ourKTRWS(b= 1) model achieves significantly
lower combined recognition and segmentation latency on the KHOB evaluation

Abbreviated paper title 11
UKTR (CTC) KTRWS (b=0) KTRWS (b=1)020406050 50
40Latency (s)
Fig. 5:Latency (seconds) comparison on for the join recognition and segmentation
task on the KHOB evaluation set.
Table 4:F1 scores on the evaluation datasets of different modalities (D,S,&H). * :
model-specific tokenizer. Char.: character. KCC: Khmer character cluster.Bold: best.
Italic: second best.
KHOB(D) KhmerST(S) KH(D,H) GKST(S) KHT(H)
KTRWS(b= 0) 98.49 97.90 95.84 96.41 93.40
KTRWS(b= 1) 97.68 95.08 91.09 93.35 89.24
setthanthesequentialrecognition-then-segmentationapproaches(i.e.,UKTRor
KTRWS(b= 0)).Thisisattributedtoitsparallel,jointdecodingnature,which
eliminates the need to perform word segmentation as a separate step using an
external model such as the Transformer-based Khmer word segmentation model
described in Section 4. It should be noted that the degree of latency reduction
is directly proportional to the latency of the word segmentation model used in
the sequential pipeline.
Segmentation Accuracy of the KTRWS ModelsWe evaluate word seg-
mentation using F1 on the evaluation datasets. For each textline image, letR
andHdenote the reference and hypothesis word sequences, respectively. There
are two possible cases:
–WithoutOCRerrors:whentheconcatenatedcharactersequencesagree(SR=SH), we compare the sets of cumulative character-offset boundariesB Rand
BHdirectly and compute true positive (TP), false positive (FP), and false
negative (FN).
–With OCR errors: When OCR errors cause a character mismatch, boundary
offsets are no longer comparable. We fall back to word-level multiset overlap
TP =P
wmin(c R(w), c H(w)), wherec(·)denotes word counts, with FP and
FN defined analogously.
TP, FP, and FN are summed across all textline images and used to compute
micro-averaged F1 score.

12 Kong et al.
(a)English translation:Meditation helps to train the mind.
(b)English translation:Depending the present teachers.
(c)English translation:For small projects of humanity security.
(d)English translation:Tour vehicles for renting.
(e)English translation:For the vocal cords to muffle the painful groans.
(f)English translation:For the vocal cords to muffle the painful groans.
Fig. 6:Qualitative assessment of recognition and segmentation accuracy on some se-
lected cases. Red bar: missing word boundary. Strided red bar: extra word boundary.
Red: recognition mistake.
Since none of the evaluation datasets originally have word boundary annota-
tions, we used our trained word segmentor discussed in Section 4 to segment the
ground-truth texts. The same applies to the model predictions of ourKTRWS
(b= 0) model (i.e., without word boundaries). In contrast, the model predictions
of ourKTRWS(b= 1) model already include word boundaries. Table 4 shows
that the segmentation accuracy of ourKTRWS(b= 1) model degrades as the
document modality transitions from document to scene and handwritten text.
This is because the document dataset (KHOB) mainly comprises long text-line
imageswith completesentences,whicharenot hardtorecognize andprovide suf-
ficient contextual cues for word segmentation. On the other hand, the scene and
handwritten cases (KhmerST, KH, GKST, and KHT) mainly comprise single
words, short phrases, or non-textual content (e.g., numbers), which are out-of-
context and challenging for both recognition and word boundary identification.
Qualitative AssessmentFigure 6 provides a qualitative assessment of the
recognition and segmentation outputs from ourKTRWS(b= 1) model on se-
lected cases. Consistent with the above observations, while recognition errors
are infrequent, except in Figure 6f (i.e., a historically degraded image), segmen-
tation errors (mainly missing word boundaries) occur more frequently on scene
and handwritten text images, as in the cases of Figures 6b, 6c, 6e, and 6f.

Abbreviated paper title 13
VisualGrounding&AlternativeDesignforWordBoundaryProjector
ModuleAnother benefit of the proposed joint recognition and word segmen-
tation method is the visual grounding of recognized words. Thanks to the CTC
decoder predictingp i(c)for each feature frame along the width direction, it is
possible to locate the word boundary tokens and map them to their actual po-
sitions on the input image. As a result, each predicted word can be accurately
visually grounded.
Figure 7 shows the projections of the word boundary locations on the in-
put images. Except for highly curved text (e.g., Figure 7c), the projected word
boundary locations are reasonably accurate. It should be noted that the visual
encoder downsamples the features by a factor of four in the width direction,
which makes it inevitable that the predicted locations may be slightly off.
(a)English translation:Meditation helps to train the mind.
(b)English translation:Tour vehicles for renting.
(c)English translation:How about father and mother.
Fig. 7:Theprojectedwordboundarylocationsontheinputimages.Redbar:boundary
location.
In terms of the alternative design for the word boundary projector module,
Tables 5 and 6 show that the default design of the word boundary projector
module achieves better character recognition and word segmentation accuracies
across the evaluation datasets. Thus, fusing the word boundary instruction via
feature-wise linear modulation is more effective than simple gated feature fusion.
6 Limitations & Future Work
We identify the following limitations and future directions associated with this
study:
1. The training relies on the new synthetic dataset with word boundary anno-
tations. Although both recognition and segmentation performance are high
for document images, there is room for improvement for other modalities
(i.e., scene and handwritten). Thus, future work will focus on constructing
a diverse real dataset with word boundary annotations.

14 Kong et al.
Table 5:Charactererrorrates(CER)ontheevaluationdatasetsofdifferentmodalities
(D,S,&H).*:model-specifictokenizer.Char.:character.KCC:Khmercharactercluster.
Bold: best.Italic: second best.
KHOB(D) KhmerST(S) KH(D,H) GKST(S) KHT(H)
KTRWS(Gating;b= 0) 1.75 2.98 6.23 4.92 9.84
KTRWS(Gating;b= 1) 1.82 3.27 7.62 5.17 10.46
KTRWS(FiLM;b= 0) 1.75 2.77 5.78 4.51 8.91
KTRWS(FiLM;b= 1) 1.79 3.04 6.06 5.01 9.64
Table 6:F1 scores on the evaluation datasets of different modalities (D,S,&H). * :
model-specific tokenizer. Char.: character. KCC: Khmer character cluster.Bold: best.
Italic: second best.
KHOB(D) KhmerST(S) KH(D,H) GKST(S) KHT(H)
KTRWS(Gating;b= 0) 98.48 97.73 94.06 96.23 93.01
KTRWS(Gating;b= 1) 97.50 94.69 87.97 93.01 87.87
KTRWS(FiLM;b= 0) 98.49 97.90 95.84 96.41 93.40
KTRWS(FiLM ;b= 1) 97.68 95.08 91.09 93.35 89.24
2. ThewordboundaryisimplicitlyhandledbyminimizingtheL CTCloss.Thus,
future work will incorporate explicit word boundary supervision.
7 Conclusion
WeproposeanoveljointKhmertextrecognitionandwordsegmentation(KTRWS)
framework within a single model. With a CTC decoder for fast, parallel decod-
ing, the KTRWS model can be instructed to recognize Khmer text with (b= 1)
and without (b= 0) word segmentation. Experimental results on benchmark
datasets of different document modalities (document, scene, and handwritten
images) show that the proposed model not only recognizes characters in docu-
ment images but also locates word boundaries, removing the need for an extra
word segmentation step in a conventional sequential pipeline.
References
1. Bi, N., Taing, N.: Khmer word segmentation based on bi-directional maximal
matching for plaintext and microsoft word document. In: Signal and Informa-
tion Processing Association Annual Summit and Conference (APSIPA), 2014 Asia-
Pacific. pp. 1–9. IEEE (2014)
2. Buoy, R., Chenda, S., Taing, N., Kong, M., Iwamura, M., Kise, K.: Addressing
the attention drift problem for khmer long textline recognition: R. buoy et al.
International Journal on Document Analysis and Recognition (IJDAR) pp. 1–26
(2025)

Abbreviated paper title 15
3. Buoy, R., Iwamura, M., Srun, S., Kise, K.: Toward a low-resource non-latin-
complete baseline: an exploration of khmer optical character recognition. IEEE
Access11, 128044–128060 (2023)
4. Buoy, R., Iwamura, M., Srun, S., Kise, K.: Language-aware non-autoregressive
khmer textline recognition. In: International Conference on Pattern Recognition
and Artificial Intelligence. pp. 339–353. Springer (2024)
5. Buoy,R.,Taing,N.,Chenda,S.,Kor,S.:Khmerprintedcharacterrecognitionusing
attention-based seq2seq network. Ho Chi Minh City Open University Journal Of
Science-Engineering And Technology pp. 3–16 (2022)
6. Buoy, R., Taing, N., Kor, S.: Khmer word segmentation using bilstm networks. In:
4th Regional Conference on OCR and NLP for ASEAN Languages (ONA 2020),
Phnom Penh, Cambodia (2020)
7. Chea, V., Thu, Y.K., Ding, C., Utiyama, M., Finch, A., Sumita, E.: Khmer word
segmentationusingconditionalrandomfields.KhmerNaturalLanguageProcessing
pp. 62–69 (2015)
8. Chey, C., Kumhom, P., Chamnongthai, K.: Khmer printed character recognition
by using wavelet descriptors. International Journal of Uncertainty, Fuzziness and
Knowledge-Based Systems14(03), 337–350 (2006)
9. EKYC Solutions: Khmer ocr benchmark dataset.https : / / github . com /
EKYCSolutions/khmer-ocr-benchmark-dataset(2022), a standardized bench-
mark dataset for Khmer Optical Character Recognition (OCR), developed in col-
laboration with EKYC Solutions, Prudential Life Assurance PLC, and Paragon
International University
10. Graves, A., Fernández, S., Gomez, F., Schmidhuber, J.: Connectionist temporal
classification: labelling unsegmented sequence data with recurrent neural networks.
In: Proceedings of the 23rd international conference on Machine learning. pp. 369–
376 (2006)
11. Kaing, H.: Towards morphological and syntactic analyses for the khmer language
(2022)
12. Keo,S.,Coustaty,M.,Bakkali,S.,Rossinyol,M.:State-of-the-artkhmertextrecog-
nition using deep learning models. In: ASEAN Conference on Emerging Technolo-
gies 2024 (2024)
13. Kong, M., Buoy, R., Chenda, S., Taing, N., Iwamura, M., Kise, K.: Towards uni-
versal khmer text recognition. arXiv preprint arXiv:2603.00702 (2026)
14. Kruy, V., Kameyama, W.: Preliminary experiment on khmer ocr. FIT (2010)
15. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for
knowledge-intensive nlp tasks. Advances in neural information processing systems
33, 9459–9474 (2020)
16. Long, S., Qin, S., Panteleev, D., Bissacco, A., Fujii, Y., Raptis, M.: Icdar 2023
competition on hierarchical text detection and recognition. In: International Con-
ference on Document Analysis and Recognition. pp. 483–497. Springer (2023)
17. Nom, V., Bakkali, S., Luqman, M.M., Coustaty, M., Ogier, J.M.: Khmerst: a low-
resource khmer scene text detection and recognition benchmark. In: Proceedings
of the Asian Conference on Computer Vision. pp. 1777–1792 (2024)
18. Nom, V., Keo, S., Bakkali, S., Luqman, M.M., Coustaty, M., Rossinyol, M., Ogier,
J.M.: Wildkhmerst: a comprehensive dataset and benchmark for khmer scene text
detection and recognition in the wild. In: International Conference on Document
Analysis and Recognition. pp. 351–368. Springer (2025)
19. Paruchuri, V., Team, D.: Surya: A lightweight document ocr and analysis toolkit.
https://github.com/datalab-to/surya(2025), gitHub repository

16 Kong et al.
20. Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A.: Film: Visual rea-
soning with a general conditioning layer. In: Proceedings of the AAAI conference
on artificial intelligence. vol. 32 (2018)
21. Sok, P., Taing, N.: Support vector machine (svm) based classifier for khmer printed
character-set recognition. In: Signal and information processing association annual
summit and conference (APSIPA), 2014 Asia-Pacific. pp. 1–9. IEEE (2014)
22. Tesseract OCR: Tesseract open source ocr engine.https : / / github . com /
tesseract-ocr/tesseract(2024)
23. Valy, D., Verleysen, M., Chhun, S.: Data augmentation and text recognition on
khmer historical manuscripts. In: 2020 17th International Conference on Frontiers
in Handwriting Recognition (ICFHR). pp. 73–78. IEEE (2020)
24. Valy, D., Verleysen, M., Chhun, S., Burie, J.C.: A new khmer palm leaf manuscript
dataset for document analysis and recognition: Sleukrith set. In: Proceedings of
the 4th International Workshop on Historical Document Imaging and Processing.
pp. 1–6 (2017)
25. Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,L.,Gomez,A.N.,Kaiser,
Ł., Polosukhin, I.: Attention is all you need. Advances in neural information pro-
cessing systems30(2017)
26. Vitou, S., Y., K., Lany, M., Botum, C., Kor, S.: Khmer handwritten ocr using pre-
trained trocr architecture (10 2024).https://doi.org/10.13140/RG.2.2.22531.
92967
27. Yath, S.: Synthkhmer-10k.https : / / huggingface . co / datasets / seanghay /
SynthKhmer-10k(2024), synthetic Khmer text/OCR dataset