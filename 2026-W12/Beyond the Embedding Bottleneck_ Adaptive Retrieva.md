# Beyond the Embedding Bottleneck: Adaptive Retrieval-Augmented 3D CT Report Generation

**Authors**: Renjie Liang, Yiling Ma, Yang Xing, Zhengkang Fan, Jinqian Pan, Chengkun Sun, Li Li, Kuang Gong, Jie Xu

**Published**: 2026-03-16 18:56:14

**PDF URL**: [https://arxiv.org/pdf/2603.15822v1](https://arxiv.org/pdf/2603.15822v1)

## Abstract
Automated radiology report generation from 3D CT volumes often suffers from incomplete pathology coverage. We provide empirical evidence that this limitation stems from a representational bottleneck: contrastive 3D CT embeddings encode discriminative pathology signals, yet exhibit severe dimensional concentration, with as few as 2 effective dimensions out of 512. Corroborating this, scaling the language model yields no measurable improvement, suggesting that the bottleneck lies in the visual representation rather than the generator. This bottleneck limits both generation and retrieval; naive static retrieval fails to improve clinical efficacy and can even degrade performance. We propose \textbf{AdaRAG-CT}, an adaptive augmentation framework that compensates for this visual bottleneck by introducing supplementary textual information through controlled retrieval and selectively integrating it during generation. On the CT-RATE benchmark, AdaRAG-CT achieves state-of-the-art clinical efficacy, improving Clinical F1 from 0.420 (CT-Agent) to 0.480 (+6 points); ablation studies confirm that both the retrieval and generation components contribute to the improvement. Code is available at https://github.com/renjie-liang/Adaptive-RAG-for-3DCT-Report-Generation.

## Full Text


<!-- PDF content starts -->

Beyond the Embedding Bottleneck: Adaptive
Retrieval-Augmented 3D CT Report Generation
Renjie Liang1, Yiling Ma2, Yang Xing1, Zhengkang Fan1, Jinqian Pan1,
Chengkun Sun1, Li Li3, Kuang Gong1, and Jie Xu1
1University of Florida
2Yale University
3University of Southern California
Abstract.Automatedradiologyreportgenerationfrom3DCTvolumes
often suffers from incomplete pathology coverage. We provide empirical
evidence that this limitation stems from a representational bottleneck:
contrastive 3D CT embeddings encode discriminative pathology signals,
yet exhibit severe dimensional concentration, with as few as 2 effective
dimensions out of 512. Corroborating this, scaling the language model
yields no measurable improvement, suggesting that the bottleneck lies
in the visual representation rather than the generator. This bottleneck
limits both generation and retrieval; naive static retrieval fails to im-
prove clinical efficacy and can even degrade performance. We propose
AdaRAG-CT, an adaptive augmentation framework that compensates
for this visual bottleneck by introducing supplementary textual infor-
mation through controlled retrieval and selectively integrating it during
generation. On the CT-RATE benchmark, AdaRAG-CT achieves state-
of-the-art clinical efficacy, improving Clinical F1 from 0.420 (CT-Agent)
to 0.480 (+6 points); ablation studies confirm that both the retrieval and
generation components contribute to the improvement. Code is avail-
able athttps://github.com/renjie-liang/Adaptive-RAG-for-3DCT-
Report-Generation.
Keywords:3D CT report generation·retrieval-augmented generation
·contrastive representation analysis
1 Introduction
Three-dimensional computed tomography (CT) is central to thoracic diagno-
sis, yet translating volumetric findings into structured radiology reports re-
mains labor-intensive. Recent work has explored vision-language models for
this task: CT-CHAT [15] combines a contrastive 3D vision encoder with a
large language model; CT-Agent [11] incorporates anatomy-aware retrieval and
chain-of-thought reasoning; and BTB3D [17] replaces contrastive encoders with
reconstruction-based tokenization. Despite these advances, the best reported
Clinical F1 on CT-RATE remains below clinically acceptable levels, indicating
that many pathological findings remain unreported.arXiv:2603.15822v1  [cs.CV]  16 Mar 2026

2 R. Liang et al.
To understand this limitation, we analyze the representation space of con-
trastive 3D CT embeddings. Linear probes [2] achieve AUC scores between 0.59
and 0.97 across 18 pathological findings, confirming that discriminative signals
are present. However, principal component analysis shows that most variance is
concentrated in only a few dimensions [1,21], resulting in extremely low effective
dimensionality; organ-level encoders exhibit similar behavior. These embeddings
can therefore distinguish coarse pathological states, but they lack the capacity
to represent fine-grained report semantics. Scaling the language model from 8B
to 70B parameters yields no measurable improvement [17,39], indicating that
the bottleneck lies in the visual representation rather than the decoder.
When the visual channel cannot carry sufficient information, additional in-
formation must be introduced through another modality. Retrieval-augmented
generation [13,19,23] offers a mechanism for such supplementation: retrieving
reference sentences from similar cases and injecting them as context supplies the
generator with details not encoded in the visual embedding. However, the relia-
bility of this textual channel depends on the same representation that constrains
generation. If the embedding lacks capacity to separate fine-grained pathological
findings, image-based retrieval cannot reliably isolate relevant cases. In practice,
naive static retrieval does not improve clinical efficacy and can even degrade
performance.
An empirical analysis across four 3D contrastive encoders confirms this link:
pathology-level retrieval precision remains low, particularly for organs with di-
verse findings. As a result, image-based retrieval struggles to isolate relevant
findings when multiple pathologies co-occur within the same organ. Rather than
replacing the encoder, as in concurrent reconstruction-based approaches [17], we
retain existing contrastive models and design a controlled augmentation strategy
that maximizes the utility of the textual channel while mitigating the misalign-
ment inherited from the visual representation.
Ourframework,AdaRAG-CT,introducessupplementarytextualcontextthrough
controlled retrieval and integrates it selectively during report generation. At the
retrieval stage, we explore strategies that improve finding-level relevance beyond
naive image-based retrieval. At the decoding stage, inspired by Self-RAG [3], a
learned[RAG]token allows the model to trigger retrieval and condition on the
returned context only when it is beneficial, trained via an oracle-mixed regime
that exposes the model to both ground-truth and imperfect retrievals.
This work makes three contributions. First, we diagnose a representational
bottleneck in 3D medical contrastive embeddings: a PCA analysis across four
encoders reveals effective dimensionalities of only a few dimensions, far below
natural-image CLIP, providing empirical evidence that the visual channel alone
cannot support comprehensive report generation. Second, we propose AdaRAG-
CT, an adaptive augmentation framework that compensates for this bottle-
neck by introducing and controlling supplementary textual information: retrieval
strategies improve finding-level relevance, while a learned adaptive mechanism
enables the generator to selectively utilize retrieved context. Third, with adap-
tive retrieval augmentation, our model achieves state-of-the-art clinical efficacy

Beyond the Embedding Bottleneck 3
on CT-RATE; ablation studies confirm that both the retrieval and generation
components contribute to the improvement.
2 Related Work
2.1 3D Medical Report Generation
Generating free-text reports from volumetric CT data has advanced rapidly
alongside large language models. CT2Rep [16] introduced the first dedicated
3D CT report-generation framework using an autoregressive transformer with
relational memory; subsequent work scaled this recipe by pairing 3D CT en-
coders with powerful LLMs (CT-CHAT [15], Med3DVLM [42]). CT-Agent [11]
augments generation with anatomy-specific LoRA and memory-based exemplar
retrieval; other work explores multi-modal fusion [22,34] or memory-driven ev-
idence aggregation [12,40]. Several works directly recognise the visual repre-
sentation as a bottleneck and propose alternative tokenization or aggregation
schemes (BTB3D [17], SAMF [18]), or pursue broader 3D understanding via
multi-task pre-training (M3D [4], RadFM [41]). Our work complements this line
by providing a quantitative diagnosis ofwhyexisting contrastive embeddings
limit retrieval quality—the effective dimensionality is too low to encode fine-
grained pathology semantics—and proposing an adaptive RAG framework that
compensates through a textual channel rather than modifying the encoder.
2.2 Retrieval-Augmented Generation in Medical Imaging
RAGhasbeenwidelyadoptedinopen-domainNLP[13,24],wheredenseretrieval
over pre-trained embeddings provides grounding for generative models. In med-
ical imaging, retrieval-based approaches have been explored for 2D chest X-ray
report generation [8,19], typically retrieving full reports or report sections from
a database and using them as templates or auxiliary inputs. These methods gen-
erally assume that retrieval quality is sufficient and focus on how to integrate
retrieved content. Our analysis reveals that this assumption breaks down for
3D CT, where contrastive embeddings exhibit a systematic variance-semantic
misalignment that limits cosine retrieval precision. In the NLP literature, adap-
tive retrieval strategies have been explored: FLARE [20] generates a tentative
continuation, detects low-confidence tokens, backtracks, and re-generates with
retrieved context; Self-RAG [3] trains reflection tokens to critique and select
among retrieval-augmented candidates. Our approach differs in that the model
emits a dedicated[RAG]token proactively during generation, triggering retrieval
without backtracking or speculative rollback. The adaptive mechanism we pro-
pose, combining two-stage re-ranking with a learned retrieval trigger, addresses
a problem that, to our knowledge, has not been explicitly studied in the medical
RAG literature.
2.3 Representation Quality in Contrastive Vision-Language
Pre-training
Contrastive learning has become the dominant paradigm for aligning visual and
textual representations [25,26,29,36]. Several studies have identified patholo-

4 R. Liang et al.
gies in contrastive representation spaces: dimensional collapse, where embed-
dings occupy a low-dimensional subspace [21]; modality gap, where image and
text embeddings cluster separately [28,31]; and feature suppression, where task-
irrelevant variance dominates learned features [27,30,37,44]. In the medical
domain, CT-CLIP [15] and ViSD-Boost [9] have trained 3D CT encoders with
contrastive objectives on CT-RATE, but the retrieval-level consequences of their
representationgeometryhavereceivedlittleattention.Ouranalysiscomplements
these works by showing that the dominant variance directions in 3D medical con-
trastive embeddings fail to encode pathological semantics, a finding we term
variance-semantic misalignment, and that this property directly undermines
downstream RAG pipelines.
3 Diagnosing the Representational Bottleneck in 3D CT
Embeddings
We first diagnose the representational bottleneck of 3D CT contrastive embed-
dings and its consequences for retrieval. All experiments use the CT-RATE
dataset [15] (25,692 non-contrast chest CT studies with paired reports and 18 bi-
nary pathology labels) and three families of pre-trained 3D contrastive encoders:
CT-CLIP [15] and its fine-tuned variant CT-CLIP (VocabFine), FVLM [38], and
ViSD-Boost [9].
3.1 Encoded but Narrow
Following Alain and Bengio [2], we train linear probes (logistic regression) on
frozen representations to test whether pathology information is linearly accessi-
ble. Across the three encoders, probe AUC ranges from 0.59 to 0.97 for average-
pooled embeddings across 18 pathological findings (Tab. 2), confirming that the
contrastive objective successfully encodes diagnostic signals. However, high AUC
does not imply rich information content. A principal component analysis [1,21]
of the embedding spaces reveals that the effective dimensionality is remarkably
low (Tab. 1). CT-CLIP with average pooling concentrates 90% of its variance
in only 2 principal components (dim 90=2, PR=1.4); although max pooling in-
flates dim 90to 117, the participation ratio remains only 6.7, indicating that most
of this spread is long-tail noise rather than signal. Consistent with this, linear
probe AUC is actually lower for max pool (category-mean AUC 0.66–0.68) than
for avg pool (0.73–0.75; see Tab. 2). Organ-level embeddings from ViSD-Boost
and FVLM fare similarly (dim 90= 4–9 out of 256). An embedding can achieve
high binary AUC (e.g., “pleural effusion present vs. absent”) using just one or
two directions, yet the fine-grained semantic detail required for report generation
(laterality, severity, co-occurring findings) simply cannot be encoded in so few
effective dimensions. This distinction between discriminability (high AUC) and
information richness (low effective dimensionality) is central to our argument:
thebottleneckisnotthattheLLMfailstoextractavailableinformation,butthat
the visual channel carries too little informationto begin with. A natural-image
baselineconfirmstheseverityofthisconcentration:CLIPViT-B/32onImageNet

Beyond the Embedding Bottleneck 5
Table 1:Effective dimensionality of 3D CT contrastive embeddings. dim 90/dim 95:
number of principal components capturing 90%/95% of variance. PR: participation
ratio.
Model Embedding Total dim dim 90dim 95PR
CLIP ViT-B/32 (ImageNet) image emb. 512 243 322 64.4
CT-CLIP (Zero-shot) avg pool 512 2 4 1.4
CT-CLIP (Zero-shot) max pool 512 117 194 6.7
CT-CLIP (VocabFine) avg pool 512 1 2 1.2
CT-CLIP (VocabFine) max pool 512 3 43 1.3
ViSD-Boost lung 256 9 12 6.3
ViSD-Boost heart 256 5 6 3.6
FVLM lung 256 6 8 4.9
FVLM heart 256 4 6 2.6
achieves dim 90=243 and PR=64.4 on the same 512-dimensional space, nearly
50×higher than CT-CLIP (Tab. 1). Two factors likely explain the gap. First,
natural images span visually diverse categories, encouraging the encoder to use
many dimensions; chest CT volumes are structurally homogeneous, with patho-
logical differences confined to subtle local variations, so a contrastive objective
can separate most cases with very few directions. Second, existing 3D CT en-
coders are trained with a global contrastive loss that pairs an entire volume with
an entire report, a granularity too coarse to incentivize fine-grained, finding-level
representations. A scaling experiment corroborates this diagnosis: replacing the
8B LLM with a 70B variant yields Clinical F1 of 0.405 versus 0.455, showing no
improvement despite an 8.75×increase in generator capacity. If the bottleneck
were in the decoder, a larger model should improve performance; the absence of
any gain confirms that the visual channel is the limiting factor. This observa-
tion is consistent with prior findings: CT-CHAT reports near-identical metrics
at 8B and 70B scales [15], and BTB3D similarly concludes that improved visual
tokenization, rather than larger language backbones, is essential for 3D medical
VLMs [17].
3.2 Retrieval Under the Bottleneck
The dimensional concentration identified above has direct consequences for re-
trieval.Whenmostvarianceconcentratesinafewdimensions,cosinesimilarityis
dominated by these directions, collapsing fine-grained pathological distinctions.
We measure organ-level Jaccard@10, the intersection over union of pathology la-
bel sets between the query and each of the top-10 cosine-nearest neighbours av-
eraged over queries, and compare image-to-image, image-to-text (cross-modal),
andtext-to-text retrieval againstan upperboundconstructed fromground-truth
pathologylabeloverlap(Tab.3).Lungretrievalisweakest(Jaccard@10=0.351):
with 11 distinct findings, the Jaccard denominator grows quickly, requiring pre-
cise alignment across a combinatorially large label space. The upper bound of
0.992 for lung confirms that the database contains relevant matches—the bottle-

6 R. Liang et al.
Table 2:Linear probe AUC on frozen contrastive embeddings.*
CT-CLIP (Zero-shot) CT-CLIP (VocabFine)
Finding avg max avg max FVLM ViSD-Boost
Lung findings (11)
Lung nodule 0.619 0.582 0.588 0.516 0.664 0.713
Mosaic attenuation 0.794 0.710 0.711 0.578 0.887 0.892
Peribronchial thick. 0.717 0.631 0.730 0.645 0.777 0.814
Consolidation 0.732 0.659 0.630 0.543 0.859 0.920
Bronchiectasis 0.679 0.605 0.624 0.564 0.752 0.790
Interlob. septal thick. 0.762 0.700 0.725 0.583 0.863 0.874
Emphysema 0.712 0.629 0.695 0.641 0.792 0.799
Atelectasis 0.680 0.627 0.667 0.561 0.738 0.797
Lung opacity 0.653 0.610 0.634 0.551 0.805 0.871
Pulm. fibrotic seq. 0.621 0.582 0.614 0.542 0.688 0.744
Pleural effusion 0.911 0.849 0.850 0.718 0.956 0.966
Heart findings (2)
Cardiomegaly 0.891 0.794 0.836 0.666 0.943 0.952
Pericardial effusion 0.797 0.687 0.764 0.685 0.851 0.910
Aorta findings (2)
Arterial wall calc. 0.818 0.734 0.808 0.676 0.898 0.913
Coronary artery calc. 0.812 0.737 0.797 0.664 0.879 0.903
Esophagus findings (1)
Hiatal hernia 0.701 0.650 0.665 0.573 0.758 0.821
Other findings (2)
Lymphadenopathy 0.688 0.621 0.648 0.586 — —
Medical material 0.726 0.642 0.688 0.590 — —
Macro avg 0.739 0.669 0.704 0.604 0.823 0.853
*“—”: finding outside the organ encoder’s scope.
neck lies in the representation, not the coverage. By contrast, heart (2 findings)
and esophagus (1 finding) achieve much higher overlap simply because fewer la-
bels need to agree. The aorta is an exception (Img2Img 0.741>Txt2Txt 0.666):
its pathological vocabulary (arterial and coronary calcification) overlaps sub-
stantially with the heart section, reducing text-to-text discriminability, whereas
the organ-specific image embeddings remain well-separated by construction.
A clear asymmetry emerges across modalities. Text-to-text cosine similar-
ity outperforms image-to-image retrieval for lung (0.563 vs. 0.351), heart (0.925
vs. 0.825), and esophagus (0.935 vs. 0.796), indicating that the text encoder
preserves pathological semantics more effectively than the image encoder. Cross-
modalretrieval(Img2Txt)isworsethanimage-to-imageforallorgans,indicating
that direct cross-modal matching is unreliable in this setting. The gap between
Txt2Txt and Img2Img suggests that the bottleneck originates in the image rep-
resentation, not in cosine similarity per se. Conventional medical RAG systems

Beyond the Embedding Bottleneck 7
Table 3:Retrieval precision (Jaccard@10) by modality. Img2Img: image embedding
cosine similarity. Img2Txt: image query against text database (cross-modal). Txt2Txt:
text embedding cosine similarity. Upper Bound: ranking by ground-truth pathology
overlap.
Organ Img2Img Img2Txt Txt2Txt Upper Bound
Lung 0.351 0.313 0.563 0.992
Heart 0.825 0.638 0.925 1.000
Esophagus 0.796 0.617 0.935 1.000
Aorta 0.741 0.617 0.666 1.000
query a text database with the image embedding (Img2Txt), which is directly
constrained by this bottleneck. Text-to-text retrieval offers a way to reduce this
dependence, suggesting that retrieval strategies leveraging partial text may be
more robust than purely image-based approaches. These observations motivate
the adaptive framework introduced in Sec. 4.
4 Method: AdaRAG-CT
AdaRAG-CT operates on top of a base vision-language model and augments
it with adaptive retrieval-augmented generation. We describe each component
below and connect design choices to the diagnostic findings of Sec. 3.
4.1 Base Model
We adopt a LLaVA-style architecture [33] where a pre-trained LLM (Llama-
3.1-8B-Instruct and Llama-3.3-70B-Instruct [14]) receives visual input through
learned projectors. Each 3D CT volume is represented byK= 5embeddings:
a global max-pooled embeddingv global∈R512from CT-CLIP and four organ-
specific embeddingsv o∈R256(o∈ {lung, heart, esophagus, aorta}) from ViSD-
Boost. Each embedding is projected independently as
ho=MLP o(vo),h o∈RdLLM, (1)
where MLP ois a 2-layer projector specific to organo. The resulting visual tokens
{ho}K
o=1are incorporated into the input token sequence as visual context for
the LLM. This base model, without any retrieval augmentation, serves as our
primary baseline.
4.2 Organ-Indexed Sentence Database
We construct an organ-indexed sentence database from all training reports. Each
report is first split into individual sentences using NLTK sentence tokenisa-
tion. Every sentence is then annotated with its corresponding organ label (lung,
heart, esophagus, aorta) and associated pathology findings, yielding approxi-
mately 572K sentences in total across both Findings and Impression sections.

8 R. Liang et al.
Fig. 1:OverviewofAdaRAG-CT.(a)A3DCTvolumeisencodedintoaglobalembed-
ding(CT-CLIP)andfourorgan-specificembeddings(ViSD-Boost),whichareprojected
into the LLM input space as visual tokens. (b) An organ-indexed sentence database
is built from training-set reports and indexed with FAISS for efficient retrieval. (c)
During report generation, the LLM autonomously emits a[RAG]token when it needs
external evidence; retrieved sentences are then injected into the context before the
model continues generating.
To represent each sentence, we employ a text encoder based on BiomedVLP-
CXR-BERT [7], followed by a learned 256-dimensional projection layer. The text
encoder and projection layer are fine-tuned using a contrastive image–text objec-
tive, with frozen organ-level image embeddings serving as anchors. All sentence
embeddings are stored in per-organ FAISS indices to enable efficient nearest-
neighbour retrieval. Organ-level image embeddings from the training set are
similarly indexed for coarse candidate retrieval.
4.3 Adaptive Retrieval Training
Following Self-RAG [3], we introduce a special[RAG]token into the vocabulary.
Rather than injecting context at predetermined positions, the model learns to
generate[RAG]autonomously when it determines that external context would
be beneficial. To identify where[RAG]should appear during training, we run
the base model on ground-truth reports and compute per-sentence perplexity;
sentences exceeding a fixed percentile threshold are marked as retrieval targets.
Foreachtarget,contextsentencesareprecomputedofflineviaimage-basedcoarse
filtering followed by text-to-text re-ranking. Final candidates are selected using
Maximal Marginal Relevance (MMR) [10]:

Beyond the Embedding Bottleneck 9
MMR(d i) =λ·sim(d i)−(1−λ)·max
dj∈SBLEU-2(d i, dj), (2)
where sim(d i)is the cosine similarity score andSis the set of already-selected
sentences. Each retrieval target then receives either ground-truth reference sen-
tences (oracle context) with probabilityp oracle, or real retrieval results (noisy
context) otherwise. This oracle-mixed regime exposes the model to high-quality
signals while building robustness to imperfect retrieval. Oracle context is used
only during training; injected context tokens are masked from the generation
loss.
4.4 Adaptive Retrieval Inference
During autoregressive generation, whenever the model emits[RAG], it continues
to generate the next sentence, which is then used as the retrieval query. The
retrieved sentences are formatted and injected into the context, the initial gen-
eration is rolled back, and the model regenerates the sentence with the retrieved
context.Weexploretworetrievalstrategiesforsupplyingcandidatesentences.As
shown in Fig. 2, both yield comparable generation quality, reinforcing that the
adaptive[RAG]mechanismistheprimarydriverofimprovement.Bothstrategies
use MMR (Eq. (2)) for final candidate selection withK fine= 3.
Two-Stage retrieval.Given the organ currently being described, we first re-
trieve theK coarse = 20most similar training samples using image embedding
cosine similarity, collecting all their sentences for the same organ as a candi-
date pool. This vision-guided stage narrows the search from the full database
to patient-relevant candidates. Candidates are then re-ranked and selected via
MMR (Eq. (2)) against the current query sentence.
Text2Text retrieval.Weretrievedirectlyfromthefullper-organsentencedatabase
using text-to-text cosine similarity, with the model’s partial generation as query.
Candidates are selected via MMR (Eq. (2)).
Together,thesecomponentscompensateforthelimitedvisualchannelthrough
controlled textual augmentation.
5 Experiments
5.1 Setup
Dataset.All experiments use CT-RATE [15], comprising 25,692 non-contrast
chest CT studies from 21,304 patients, each paired with a free-text radiol-
ogy report containing both Findings and Impression sections. We use the con-
catenation of both sections as the generation target, and follow the official
train/validation/test split provided by CT-RATE.

10 R. Liang et al.
Table 4:Report generation performance on the CT-RATE validation set. Baseline
numbers are from original publications.
Clinical Efficacy NLP
Method Params F1 P R B-1 B-4 R-L MET LLaMA Score
CT2Rep [16] — 0.160 0.435 0.128 0.372 0.213 — 0.197 —
Merlin [6] 7B 0.160 0.295 0.112 0.231 0.099 — 0.148 —
CT-CHAT [15] 8B — — — 0.494 — 0.584 0.311 7.440
CT-CHAT [15] 70B 0.184 0.450 0.158 0.498 — 0.581 0.311 7.429
BTB3D [17] 8B 0.258 0.260 0.260 0.439 0.213 — 0.223 —
SAMF [18] 3.8B — — — 0.440 0.261 0.417 0.417 7.165
CT-Agent [11] — 0.420 0.423 0.477 0.502 0.231 0.490 0.425 —
Ours (base) 8B 0.455 0.474 0.469 0.463 0.205 0.315 0.206 7.30
AdaRAG-CT 8B 0.480 0.502 0.520 0.496 0.242 0.354 0.246 7.75
Ours (base) 70B 0.405 0.468 0.373 0.449 0.213 0.334 0.208 7.10
AdaRAG-CT 70B 0.426 0.483 0.413 0.497 0.250 0.361 0.232 7.53
Note: We also reproduce several baselines under our unified evaluation protocol in Tab. 10.
Metrics.Following the CT-RATE evaluation protocol [15], we adopt Clinical
Efficacy as the primary metric. A RadBERT-based classifier [43] extracts 18 bi-
narypathologicalfindinglabels(e.g.,lungnodule,pleuraleffusion,cardiomegaly)
from each generated report. These predictions are compared against the dataset-
providedground-truthannotationstocomputeper-findingPrecision,Recall,and
F1.Wereportthesupport-weightedmacroaverageasClinicalPrecision(Clin-P),
Clinical Recall (Clin-R), and Clinical F1 (Clin-F1). We also report BLEU [35],
ROUGE-L [32], and METEOR [5] for textual overlap.
Baselines.We compare against prior methods on CT-RATE report genera-
tion: CT2Rep [16], an encoder-decoder with relational memory; Merlin [6], a
7B VLM for 3D CT understanding; CT-CHAT [15], which pairs a CT-ViT en-
coder with Llama-3.1 (8B/70B) via attention pooling; BTB3D [17], which uses
reconstruction-based visual tokenization; SAMF [18], a 2D–3D feature fusion
approach; and CT-Agent [11], an agent-based pipeline with chain-of-thought
reasoning. All baseline numbers are taken from the original publications. We
also report our own base model (no RAG) at 8B and 70B scales, trained in two
stages: organ-specific projectors with frozen LLM weights, followed by full LoRA
fine-tuning.
5.2 Main Results
Table 4 presents the main comparison. Our 8B base model obtains a Clinical
F1 of 0.455 without retrieval. Scaling the LLM to 70B does not improve this
figure; in fact, the 70B base model scores only 0.405, consistent with the PCA
analysis showing that the visual channel carries only 2–9 effective dimensions
(Sec. 3.1). Both models read the same impoverished visual representations, so
the larger decoder has no additional signal to exploit—the bottleneck is repre-
sentational, not generative. Retrieval augmentation benefits both scales: the 8B
model improves from 0.455 to 0.480 (+0.025) and the 70B model from 0.405 to

Beyond the Embedding Bottleneck 11
1000 2000 3000 4000 5000 6000 7000
Training Step0.00.10.20.30.40.50.6
BLEU-1
1000 2000 3000 4000 5000 6000 7000
Training Step0.00.10.20.30.40.50.6
BLEU-4
1000 2000 3000 4000 5000 6000 7000
Training Step0.00.10.20.30.40.50.6
ROUGE-L
1000 2000 3000 4000 5000 6000 7000
Training Step0.300.350.400.450.500.550.60
Clinical Precision
1000 2000 3000 4000 5000 6000 7000
Training Step0.300.350.400.450.500.550.60
Clinical Recall
1000 2000 3000 4000 5000 6000 7000
Training Step0.300.350.400.450.500.550.60
Clinical F1T wo-Stage T ext2T ext
Fig. 2:Six evaluation metrics across training steps for Two-Stage and Text2Text re-
trieval pipelines on the 8B model, with bootstrap 95% confidence intervals.
0.426 (+0.021), confirming that the textual channel supplies information absent
from the visual input. The 8B AdaRAG-CT (0.480) surpasses CT-Agent (0.420)
and substantially outperforms CT-CHAT 70B (0.184) in Clinical F1, while also
improving NLP metrics and LLaMA Score across the board.
5.3 Retrieval Pipeline Comparison
Figure 2 shows that both pipelines yield stable performance throughout train-
ing, with Clinical F1 virtually identical at every step and bootstrap confidence
intervals consistently overlapping, confirming robustness to checkpoint selection.
A closer look at the precision–recall decomposition reveals a mild trade-off: Two-
Stage retrieval, which first narrows candidates via image similarity before text
re-ranking, tends to surface a broader set of organ-level findings and thus favours
recall; Text2Text retrieval, which matches directly against the model’s partial
generation, returns more topically precise context and favours precision. These
complementary biases cancel out in F1, reinforcing that the adaptive[RAG]
utilization mechanism—not the retrieval pipeline—is the primary driver of im-
provement. This pattern is consistent with the bottleneck hypothesis: because
any sufficiently relevant textual context substantially compensates for the im-
poverished visual signal, the marginal benefit of a more precise retrieval strategy
is small.
5.4 Context Utilization Ablation
Table 5 ablates the context injection strategy while holding the retrieval pipeline
fixed. The No RAG baseline (0.462) reflects the RAG-trained model’s own gener-

12 R. Liang et al.
Table 5:Context utilization ablation (8B model). All variants share the same retrieval
pipeline;∆is relative to the No RAG baseline.
Utilization Strategy Clin-F1∆BLEU-4 ROUGE-L
No RAG 0.462 — 0.228 0.349
Fixed-interval (N=3) 0.453−0.009 0.192 0.307
Fixed-interval (N=5)0.494 +0.0320.209 0.329
Fixed-interval (N=7) 0.482 +0.020 0.228 0.346
Adaptive RAG (ours) 0.480 +0.0180.242 0.354
−context (OOD) 0.402−0.060 0.122 0.228
ation capability without retrieval context at inference, which differs slightly from
the base model in Tab. 4 (0.455) that was trained without the RAG objective.
Fixed-interval injection is sensitive to the choice ofN: injecting too frequently
(N=3) hurts performance below baseline, whileN=5 maximises Clinical F1
(0.494) at the expense of NLP quality, andN=7 strikes a middle ground. This
sensitivity toNhighlights a practical limitation of fixed-interval strategies—the
optimal frequency is dataset-dependent and requires separate tuning. Adaptive
RAG resolves this by learning to trigger retrieval selectively: it achieves com-
petitive Clinical F1 (0.480, +0.018) without any frequency hyperparameter, and
consistently outperforms all fixed-interval variants on BLEU-4 and ROUGE-L.
On average, the model emits 1.5[RAG]triggers per report, confirming that re-
trieval is used sparingly and on demand rather than applied uniformly. Crucially,
N=5 is an ex-post optimum found by grid search on the same test distribution;
on an unseen dataset it may be suboptimal, whereas adaptive triggering requires
no such tuning. Evaluating the Adaptive RAG model without any context at in-
ference (last row) causes a sharp degradation to 0.402; this out-of-distribution
result confirms that the model has formed a genuine dependence on retrieved
context, and is included for completeness rather than as a fair comparison.
5.5 Qualitative Analysis
Figure 3 illustrates how the representational bottleneck manifests and how re-
trieval compensates. The base model, relying solely on the impoverished visual
embedding, hallucinates a long list of peripheral vascular structures (femoral,
popliteal, and tibial arteries) that are outside the chest-CT field of view and
implausible under non-contrast imaging—a direct consequence of the limited
effective dimensions available in the visual channel. When the adaptive[RAG]
trigger fires, the retrieved evidence supplies patient-specific context centred on
the aortic arch and coronary arteries; the model then regenerates the sentence,
correctly focusing on aortic arch calcification and coronary vascular involvement,
which aligns with the ground truth. This example illustrates that retrieval does
not merely re-rank information already present in the visual input; it introduces
clinically specific evidence that the visual embedding alone cannot provide.

Beyond the Embedding Bottleneck 13
There are calcified plaques 
in the descending aorta, 
abdominal aorta, splenic  
artery, celiac trunk, SMA,  
renal arteries, iliac arteries,  
common femoral arteries,  
deep femoral arteries,  
popliteal arteries, anterior  
tibial arteries and posterior 
tibial arteries. Base Model (without RAG) 
Hallucinated 
systemic 
vascular list 
➔Beyond chest-CT scope 
➔Implausible given non-contrast 
imaging Retrieved Context Correct 
Diffuse calcified atheroma  
plaques were observed in  
the aortic arch, supraoarrtic  
branches and coronary  
arteries.  Calcific plaques are 
observed in the descending 
aorta, coronary arteries, 
walls of the ascending and 
abdominal aorta. There are 
widespread calcific 
atheroma plaques on the 
walls of the arcus aorta, 
descending aorta and 
coronary arteries. 
Retrieved Evidence 
Ground Truth 
  regenerated 
GT reference: There are 
calcified atheromatous  
plaques  on the walls of the 
aortic arch  and coronary  
vascular structures. 
➔Remove: femoral / popliteal /  
tibial list 
➔Keep: aortic arch & coronary 
arteries 
➔Recovered: Coronary artery 
wall calcification, Pulmonary 
fibrotic sequela 
Fig. 3:Qualitative comparison for an aorta finding. Left: base model output; centre:
AdaRAG-CT output with the regenerated sentence highlighted; right: ground-truth
report. Red highlights indicate hallucinated content; green indicates correct clinical
findings.
6 Discussion
Our results converge on a central finding:the bottleneck in 3D CT report genera-
tion is representational, not generative.ThePCAanalysisshowsthatcontrastive
embeddings carry as few as 2–9 effective dimensions of information; the scale ex-
periment confirms that a 70B LLM cannot outperform an 8B model when both
read the same impoverished visual channel. These observations reframe the role
of RAG: rather than helping the LLM better utilise existing visual information,
retrievalopens a second, higher-bandwidth channel, text tokens, through which
fine-grained pathology semantics (laterality, severity, co-occurring findings) can
reach the generator. This distinguishes our framework from conventional RAG
systems designed to access external world knowledge or improve factual ground-
ing in general-domain tasks; here, retrieval functions as asurrogate visual chan-
nel, compensating for information that the embedding simply fails to encode
rather than supplementing an otherwise sufficient representation.
The framework design follows naturally from this perspective. The retrieval
module opens the textual channel, supplying candidate sentences from the train-
ing corpus. The adaptive[RAG]trigger then controls this channel, ensuring con-
text is consumed only when the model judges it useful; ablation studies confirm
that fixed-interval injection is sensitive to frequency and cannot match adaptive
triggering on NLP quality. This dynamic also explains the scale experiment: a
70B model equipped with stronger language priors produces more fluent text
but, given the same impoverished visual signal, generates more confident hal-
lucinations rather than better-calibrated outputs—retrieval corrects this by in-
jecting patient-specific evidence that overrides generic priors. A key enabler of
the adaptive mechanism is oracle-mixed training: by exposing the model to both
high-quality oracle context and noisy retrieved context during training, it acts
as implicit data augmentation, producing a generator that exploits retrieved ev-
idence when it helps and falls back gracefully to visual-only generation when it
does not.

14 R. Liang et al.
Limitations.All experiments use CT-RATE, a single-institution dataset of non-
contrast chest CT. While the cross-encoder analysis provides evidence of gen-
erality, validation on additional datasets and anatomical regions is needed. The
oracle-mixed training regime introducesp oracleas a hyperparameter whose opti-
mal value may be dataset-dependent.
Our retrieval database is drawn entirely from the training corpus and intro-
duces no external medical knowledge; its value lies in opening a supplementary
textual channel rather than supplying genuinely new clinical information. A
related limitation concerns context quality: many CT-RATE sentences follow
highly standardized templates (e.g., “No evidence of pleural effusion”), which
appear frequently in the database and may trigger the[RAG]mechanism with-
out adding patient-specific evidence. More broadly, because database sentences
share the same training distribution as the model’s parameters, retrieved con-
text can reinforce corpus language patterns, causing outputs to converge toward
common phrasing even when the visual evidence would support more specific
descriptions.
The primary evaluation metric, Clinical F1, measures binary presence or ab-
sence of 18 predefined pathological categories but cannot capture the clinical
richness of a well-formed radiology report. Critically, it does not assess lesion-
level attributes—size, location, laterality, morphology, or severity grade—that
are essential for clinical decision-making. A report that correctly flags “lung
nodule” but provides no information about its size (3mm vs. 15mm) or loca-
tion receives the same score as one that does. Temporal comparisons, follow-up
measurement recommendations, and incidental findings outside the 18-label tax-
onomy are likewise invisible to this metric. Developing evaluation frameworks
that measure the clinical completeness and quantitative accuracy of generated
reports remains an important open problem.
AdaRAG-CT compensates for the representational bottleneck by routing
supplementary information through a textual channel, but does not resolve the
underlyingcause.Thedimensionalcollapseofcontrastiveembeddingsarisesfrom
training on small medical datasets with limited pathological diversity; our re-
trievalframeworkprovidesapracticalworkaroundratherthanafundamentalfix.
True resolution would require richer visual representations through pathology-
awarepre-trainingordenseper-tokenvisualfeatures;untilsuchencodersbecome
available, the textual retrieval channel remains a necessary and effective surro-
gate.
7 Conclusion
3D CT contrastive embeddings, despite encoding pathology-discriminative sig-
nals, carry as few as 2 effective dimensions out of 512. This representational
poverty—not the generator’s capacity—is the fundamental bottleneck: scaling
the LLM from 8B to 70B parameters yields no improvement because both mod-
els read the same impoverished visual channel. AdaRAG-CT addresses this bot-
tleneck by opening a supplementary textual channel through adaptive retrieval.

Beyond the Embedding Bottleneck 15
A learned[RAG]token triggers patient-specific context injection during genera-
tion,achievingaClinicalF1of0.480onCT-RATEandsurpassingthepriorstate
of the art by 6.0 percentage points. Ablation studies confirm that the adaptive
utilization mechanism, rather than the choice of retrieval pipeline, drives this
improvement. Resolving the bottleneck at its source—through pathology-aware
visual pre-training or dense per-token visual representations—remains the most
important direction for future work. Until richer encoders become available, the
textual retrieval channel provides a necessary and effective surrogate.
References
1. Aghajanyan, A., Gupta, S., Zettlemoyer, L.: Intrinsic dimensionality explains the
effectiveness of language model fine-tuning. In: Annual Meeting of the Association
for Computational Linguistics. pp. 7319–7328 (2021) 2, 4
2. Alain, G., Bengio, Y.: Understanding intermediate layers using linear classifier
probes. arXiv preprint arXiv:1610.01644 (2017) 2, 4
3. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-RAG: Learning to re-
trieve, generate, and critique through self-reflection. In: International Conference
on Learning Representations (2024) 2, 3, 8
4. Bai, F., Du, Y., Huang, T., Meng, M.Q.H., Zhao, B.: M3D: Advancing 3D
medical image analysis with multi-modal large language models. arXiv preprint
arXiv:2404.00578 (2024) 3
5. Banerjee, S., Lavie, A.: METEOR: An automatic metric for MT evaluation with
improvedcorrelationwithhumanjudgments.In:ProceedingsoftheACLWorkshop
on Intrinsic and Extrinsic Evaluation Measures for Machine Translation and/or
Summarization. pp. 65–72 (2005) 10
6. Blankemeier, L., Cohen, J.P., Kumar, A., Van Veen, D., Gardezi, S.J.S., Paschali,
M., Chen, Z., Delbrouck, J.B., Reis, E.P., Truyts, C., et al.: Merlin: A vision
language foundation model for 3d computed tomography. Research Square (2024)
10
7. Boecking,B.,Usuyama,N.,Bannur,S.,Castro,D.C.,Schwaighofer,A.,Hyland,S.,
Wetscherek, M., Naumann, T., Nori, A., Alvarez-Valle, J., et al.: Making the most
of text semantics to improve biomedical vision–language processing. In: European
Conference on Computer Vision. pp. 1–21. Springer (2022) 8
8. Buckley,C.J.,Almansoori,M.,Azizi,A.,Oikonomou,A.,VafaeiSadr,A.,Apte,A.,
El Naqa, I.: LaB-RAG: Label boosted retrieval augmented generation for radiology
report generation. arXiv preprint arXiv:2411.16523 (2024) 3
9. Cao, W., Zhang, J., Shui, Z., Wang, S., Chen, Z., Li, X., Lu, L., Ye, X., Liang, T.,
Zhang, Q., Zhang, L.: Boosting vision semantic density with anatomy normality
modelingformedicalvision-languagepre-training.arXivpreprintarXiv:2508.03742
(2025) 4
10. Carbonell, J., Goldstein, J.: The use of MMR, diversity-based reranking for re-
ordering documents and producing summaries. In: Proceedings of the 21st Annual
International ACM SIGIR Conference on Research and Development in Informa-
tion Retrieval. pp. 335–336 (1998) 8
11. Chen, H., Zhao, W., Li, Y., Zhang, T., Wang, Z., Xu, D., Li, M., Ma, L.: CT-Agent:
A multimodal-LLM agent for 3D CT radiology question answering. arXiv preprint
arXiv:2505.16229 (2025) 1, 3, 10

16 R. Liang et al.
12. Chen, Z., Song, Y., Chang, T.H., Wan, X.: Generating radiology reports via
memory-driven transformer. In: Proceedings of the 2020 Conference on Empir-
ical Methods in Natural Language Processing (EMNLP). pp. 1439–1449 (2020)
3
13. Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang,
H.: Retrieval-augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 (2024) 2, 3
14. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Let-
man, A., Mathur, A., Schelten, A., et al.: The llama 3 herd of models. arXiv
preprint arXiv:2407.21783 (2024) 7
15. Hamamci, I.E., Er, S., Almas, F., Simsek, A.G., Esirgun, S.N., Dogan, I., Das-
delen, M.F., Durugol, O.F., Wittmann, B., Amiranashvili, T., et al.: Generalist
foundation models from a multimodal dataset for 3D computed tomography. Na-
ture Biomedical Engineering pp. 1–19 (2025) 1, 3, 4, 5, 9, 10, 21, 23
16. Hamamci,I.E.,Er,S.,Menze,B.:CT2Rep:Automatedradiologyreportgeneration
for 3D medical imaging. In: International Conference on Medical Image Computing
and Computer-Assisted Intervention (MICCAI). pp. 476–486. Springer (2024) 3,
10
17. Hamamci, I.E., Er, S., Shit, S., Reynaud, H., Yang, D., Guo, P., Edgar, M., Xu,
D., Kainz, B., Menze, B.: Better tokens for better 3D: Advancing vision-language
modeling in 3D medical imaging. In: Advances in Neural Information Processing
Systems (2025) 1, 2, 3, 5, 10
18. Hosseini, A., Ibrahim, A., Serag, A.: From slices to volumes: Multi-scale fusion
of 2D and 3D features for CT scan report generation. In: International Confer-
ence on Medical Image Computing and Computer-Assisted Intervention (MIC-
CAI). Springer (2025) 3, 10
19. Jeblick, K., Schachtner, B., Dexl, J., Mittermeier, A., Stüber, A.T., Topalis, J.,
Weber, T., Wesp, P., Sabel, B., Ricke, J., Ingrisch, M.: Retrieval-augmented gen-
eration with large language models in radiology. Radiology: Artificial Intelligence
7(2) (2025) 2, 3
20. Jiang, Z., Xu, F.F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan,
J., Neubig, G.: Active retrieval augmented generation. In: Proceedings of the 2023
Conference on Empirical Methods in Natural Language Processing. pp. 7969–7992
(2023) 3
21. Jing, L., Vincent, P., LeCun, Y., Tian, Y.: Understanding dimensional collapse
in contrastive self-supervised learning. In: International Conference on Learning
Representations (2022) 2, 4
22. Lee, C., Park, S., Shin, C.I., Choi, W.H., Park, H.J., Lee, J.E., Ye, J.C.: Read like a
radiologist: Efficient vision-language model for 3D medical imaging interpretation
(2024), arXiv preprint arXiv:2412.13558 3
23. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H.,
Lewis, M., Yih, W.t., Rocktäschel, T., Riedel, S., Kiela, D.: Retrieval-augmented
generation for knowledge-intensive NLP tasks. In: Advances in Neural Information
Processing Systems. vol. 33, pp. 9459–9474 (2020) 2
24. Li, L., Cai, P., Rossi, R.A., Dernoncourt, F., Kveton, B., Wu, J., Yu, T., Song, L.,
Yang, T., Qin, Y., Ahmed, N.K., Basu, S., Mukherjee, S., Zhang, R., Hu, Z., Ni,
B., Zhou, Y., Wang, Z., Huang, Y., Wang, Y., Zhang, X., Yu, P.S., Hu, X., Zhao,
Y.: A personalized conversational benchmark: Towards simulating personalized
conversations (2025),https://arxiv.org/abs/2505.141063

Beyond the Embedding Bottleneck 17
25. Li, L., Ji, W., Wu, Y., Li, M., Qin, Y., Wei, L., Zimmermann, R.: Panop-
tic scene graph generation with semantics-prototype learning. Proceedings of
the AAAI Conference on Artificial Intelligence38(4), 3145–3153 (Mar 2024).
https://doi.org/10.1609/aaai.v38i4.28098,https://ojs.aaai.org/index.
php/AAAI/article/view/280983
26. Li, L., Wang, C., Qin, Y., Ji, W., Liang, R.: Biased-predicate annotation identifica-
tion via unbiased visual predicate representation. In: Proceedings of the 31st ACM
International Conference on Multimedia. p. 4410–4420. MM ’23, Association for
Computing Machinery, New York, NY, USA (2023).https://doi.org/10.1145/
3581783.3611847,https://doi.org/10.1145/3581783.36118473
27. Li, S., Cai, P., Zhou, Y., Ni, Z., Liang, R., Qin, Y., Nian, Y., Tu, Z., Hu, X., Zhao,
Y.: Secure on-device video ood detection without backpropagation. In: Interna-
tional Conference on Computer Vision (ICCV) (October 2025) 4
28. Li, S., Gong, H., Dong, H., Yang, T., Tu, Z., Zhao, Y.: Dpu: Dynamic proto-
type updating for multimodal out-of-distribution detection. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
pp. 10193–10202 (June 2025) 4
29. Li,S.,Rossi,R.,Kim,S.,Choudhary,S.,Dernoncourt,F.,Mathur,P.,Tu,Z.,Zhao,
Y.: Charts are not images: On the challenges of scientific chart editing (2025),
https://arxiv.org/abs/2512.007523
30. Li, S., Yu, C., Ni, Z., Li, H., Peris, C., Xiao, C., Zhao, Y.: Defenses against prompt
attacks learn surface heuristics (2026),https://arxiv.org/abs/2601.071854
31. Liang, W., Zhang, Y., Kwon, Y., Yeung, S., Zou, J.: Mind the gap: Understanding
the modality gap in multi-modal contrastive representation learning. In: Advances
in Neural Information Processing Systems. vol. 35, pp. 17612–17625 (2022) 4
32. Lin, C.Y.: ROUGE: A package for automatic evaluation of summaries. In: Text
Summarization Branches Out. pp. 74–81 (2004) 10
33. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in Neural
Information Processing Systems36(2024) 7
34. Luo, L., Tang, B., Chen, X., Han, R., Chen, T.: VividMed: Vision language model
with versatile visual grounding for medicine. In: Chiruzzo, L., Ritter, A., Wang, L.
(eds.) Proceedings of the 2025 Conference of the Nations of the Americas Chapter
of the Association for Computational Linguistics: Human Language Technologies
(Volume 1: Long Papers). pp. 1800–1821. Association for Computational Linguis-
tics,Albuquerque,NewMexico(Apr2025).https://doi.org/10.18653/v1/2025.
naacl-long.89,https://aclanthology.org/2025.naacl-long.89/3
35. Papineni, K., Roukos, S., Ward, T., Zhu, W.J.: BLEU: A method for automatic
evaluation of machine translation. In: Proceedings of the 40th Annual Meeting of
the Association for Computational Linguistics. pp. 311–318 (2002) 10
36. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G.,
Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from
natural language supervision. In: International Conference on Machine Learning.
pp. 8748–8763. PMLR (2021) 3
37. Shawn, L., Qu, J., Song, L., Zhou, Y., Qin, Y., Yang, T., Zhao, Y.: Treble counter-
factual VLMs: A causal approach to hallucination. In: Association for Computa-
tional Linguistics: EMNLP 2025. pp. 18423–18434. Association for Computational
Linguistics, Suzhou, China (Nov 2025) 4
38. Shui, Z., Zhang, J., Cao, W., Wang, S., Guo, R., Lu, L., Yang, L., Ye, X., Liang,
T., Zhang, Q., Zhang, L.: Large-scale and fine-grained vision-language pre-training
for enhanced CT image understanding. arXiv preprint arXiv:2501.14548 (2025) 4

18 R. Liang et al.
39. Tong,S.,Liu,Z.,Zhai,Y.,Ma,Y.,LeCun,Y.,Xie,S.:Eyeswideshut?exploringthe
visual shortcomings of multimodal LLMs. In: IEEE/CVF Conference on Computer
Vision and Pattern Recognition (2024) 2
40. Wang, Z., Liu, L., Wang, L., Zhou, L.: METransformer: Radiology report genera-
tion by transformer with multiple learnable expert tokens. In: Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
pp. 11558–11567 (2023).https://doi.org/10.1109/CVPR52729.2023.011123
41. Wu, C., Zhang, X., Lin, W., Zhang, Y., Wang, Y., Xie, W.: Towards generalist
foundation model for radiology by leveraging medical vision-language datasets.
Nature Communications16(1), 5538 (2025) 3
42. Xin, Y., Ates, G.C., Gong, K., Shao, W.: Med3dvlm: An efficient vision-language
model for 3d medical image analysis. IEEE Journal of Biomedical and Health
Informatics (2025) 3
43. Yan, A., McAuley, J., Lu, X., Du, J., Chang, E.Y., Gentili, A., Hsu, C.N.: Radbert:
Adapting transformer-based language models to radiology. Radiology: Artificial
Intelligence4(4), e210258 (2022) 10
44. Zhang, J., Lan, X., Qu, X., Cheng, Y., Feng, M., Hooi, B.: Learning the unlearned:
Mitigating feature suppression in contrastive learning. In: European Conference on
Computer Vision. pp. 37–54. Springer (2024) 4

Beyond the Embedding Bottleneck 19
A Additional Evidence for the Embedding Bottleneck
A.1 PCA and linear probe protocol details
Weevaluaterepresentationalbottleneckseverityusingbothprincipalcomponent
analysis(PCA)andfrozenlinearprobes.PCAiscomputedviafullSVDonmean-
centred embeddings. We report three effective-dimensionality metrics: dim 90and
dim 95(theminimumnumberofprincipalcomponentsneededtoexplain90%and
95% of total variance, respectively), and the participation ratio
PR = P
iσ2
i2
P
iσ4
i, (3)
whereσ iare the singular values. PR equals the embedding dimension when all
dimensions carry equal variance and approaches 1 under complete collapse.
Linear probes are trained independently for each of the 18 CT-RATE pathol-
ogy labels. Each probe is a logistic regression (solver: L-BFGS,C=1.0, class-
balanced weighting, up to 1,000 iterations) trained on the training set and eval-
uated on the held-out validation set. Classification performance is reported as
AUC-ROC.
A.2 Projection test details
To quantify variance-semantic misalignment, we perform aprojection test: for
each class or pathology, we train a linear probe on only the top-kor tail-kprinci-
pal components of the embedding space and measure binary classification AUC.
If the dominant variance directions encode class-discriminative information, the
top-kprobe should outperform the tail-kprobe. A reversal (tail>top) indi-
cates that discriminative signals reside in low-variance dimensions that cosine
similarity effectively ignores.
Table 6 compares CLIP ViT-B/32 on ImageNet with the zero-shot CT-CLIP
embeddingusedinourpipeline.Bothdomainsexhibitthetail>toppattern,but
the severity differs substantially. For CLIP, the top-2 PCs remain informative
(AUC 0.82), whereas for CT-CLIP zero-shot the top-2 AUC is only 0.55, near
randomchance.Thissupportstheclaimthat3Dmedicalcontrastiveembeddings
suffer a much stronger form of variance-semantic misalignment than natural-
image CLIP.
Table7showstheprojectiontestforall18CT-RATEpathologicalfindingson
the zero-shot CT-CLIP embedding used in our pipeline. Every finding satisfies
tail 1/2>top2, confirming that the misalignment is not driven by only a few
outliers.
B Implementation Details
B.1 Dataset and report preprocessing
We use the official CT-RATE split. Each study contains a Findings section and
an Impression section; for report generation we concatenate both sections into

20 R. Liang et al.
Table 6:Cross-domain projection test (mean over all classes or findings). top2: AUC
using only the 2 highest-variance PCs. tail 1/2: AUC using the lower half of all PCs.∆:
tail1/2−top2.
Domain / Model top2 AUC tail 1/2AUC∆
CLIP ViT-B/32 (ImageNet, 30 classes) 0.821 1.000 +0.178
CT-CLIP zero-shot, avg pool (CT-RATE, 18 findings) 0.552 0.746 +0.193
CT-CLIP zero-shot, max pool (CT-RATE, 18 findings) 0.533 0.739 +0.206
Table 7:Per-findingprojectiontestonCT-CLIPzero-shot.All18pathologicalfindings
show tail 1/2>top2.
Finding top2 AUC tail 1/2AUC∆
Medical material 0.532 0.770 +0.238
Arterial wall calcification 0.514 0.708 +0.194
Coronary artery wall calc. 0.501 0.713 +0.212
Lymphadenopathy 0.519 0.677 +0.158
Lung nodule 0.545 0.651 +0.106
Pericardial effusion 0.514 0.817 +0.303
Hiatal hernia 0.525 0.748 +0.223
Mosaic attenuation 0.540 0.808 +0.268
Peribronchial thickening 0.571 0.755 +0.184
Consolidation 0.531 0.726 +0.195
Bronchiectasis 0.537 0.774 +0.237
Interlob. septal thickening 0.534 0.808 +0.274
Cardiomegaly 0.571 0.779 +0.208
Emphysema 0.521 0.711 +0.190
Atelectasis 0.536 0.709 +0.173
Lung opacity 0.542 0.672 +0.131
Pulm. fibrotic seq. 0.519 0.682 +0.163
Pleural effusion 0.535 0.787 +0.253
a single target sequence after basic text normalisation. The official training set
contains47,149CTstudies;weexclude742studiesforwhichpre-computedorgan
embeddings are unavailable, leaving 46,407 training studies used in all experi-
ments. The validation set contains 2,987 studies, all of which have corresponding
embeddings.
For retrieval database construction, organ-level report paragraphs (produced
by a rule-based organ-section parser) are sentence-split by a regex tokeniser
that segments on full stops and semicolons followed by whitespace. Fragments
consisting solely of punctuation or bare organ-name headers are discarded. Sen-
tences that do not map cleanly to the four target organs are assigned to another
category and excluded from the retrieval indices. The final per-organ sentence
database is built from training studies only; Tab. 8 summarises its composition.
Reassemblingtheorgan-splitparagraphsyieldsBLEU-1=0.992againsttheorig-

Beyond the Embedding Bottleneck 21
inal report, and per-organ Clinical F1 (Parser F1 in Tab. 8) exceeds 0.95 for all
four target organs, confirming faithful preservation of pathological content.
Table 8:Per-organ sentence database statistics (training set only).
Organ Sentences Unique studies Avg. sents/study Avg. words/sent Parser F1
Lung 398,538 46,380 8.59 13.3 0.974
Heart 82,025 42,777 1.92 8.4 0.957
Esophagus 50,096 39,977 1.25 11.8 0.990
Aorta 41,632 25,093 1.66 10.0 0.955
Total 572,291 — — — —
B.2 Training, evaluation, and checkpoint selection
Clinical efficacy follows the CT-RATE protocol [15]: a RadBERT classifier ex-
tracts 18 binary findings per report, and we report support-weighted Precision,
Recall, and F1. BLEU, ROUGE-L, and METEOR are computed after lower-
casing. LLaMA Score uses Llama-3.1-70B-Instruct as judge. Bootstrap 95% CIs
use 1,000 iterations. Both the 8B and 70B models follow the same two-stage
training recipe: (1) train organ-specific projectors with the LLM frozen, then
(2) LoRA fine-tune the language model jointly with the projectors. The base
visual input comprises one global CT-CLIP embedding and four organ-specific
ViSD-Boost embeddings, projected to the LLM hidden dimension by indepen-
dent two-layer MLPs. All reported validation results correspond to the manu-
ally selected checkpoint with the best Clin-F1 on the validation set. To isolate
whether the bottleneck lies in the generator or in the visual representation, the
8B and 70B variants are trained with the same visual interface, the same pro-
jector design, and the same evaluation protocol; both models receive the same
five visual tokens, differing only in LLM scale.
B.3 RAG token supervision and oracle-mixed training
A dedicated[RAG]token is appended to the tokenizer vocabulary and fine-tuned
jointly with the generation objective (LoRAr=32,α=64, lr=10−5, effective
batch size 16, cosine schedule with 200 warmup steps). During training, ora-
cle context is injected periodically (roughly every 2–3 sentences) before target
sentences: 1–2 ground-truth sentences drawn from later in the same report are
wrapped in<|ret_start|> ... <|ret_end|>delimiters and prepended to the
current generation position. Context tokens are masked from the language mod-
elling loss (labels set to−100) so that the model learns tocondition onretrieved
text rather than reproduce it. The oracle-mixed ratiop oraclecontrols how often
oracle (vs. real-retrieved) context is used. In our best model (AdaRAG-CT /
P10),p oracle =0.7and at mostK rag=4[RAG]triggers are allowed per training

22 R. Liang et al.
Table 9:Per-finding Clinical F1 breakdown: base model (8B, no RAG) vs. AdaRAG-
CT (8B, two-stage).∆: change due to adaptive retrieval augmentation.
Finding Base F1 AdaRAG F1∆
Lung findings (11)
Lung nodule 0.565 0.591 +0.025
Mosaic attenuation 0.349 0.379 +0.030
Peribronch. thick. 0.245 0.160−0.085
Consolidation 0.524 0.534 +0.009
Bronchiectasis 0.096 0.151 +0.054
Interlob. septal thick. 0.251 0.343 +0.092
Emphysema 0.433 0.410−0.023
Atelectasis 0.437 0.436−0.001
Lung opacity 0.651 0.677 +0.027
Pulm. fibrotic seq. 0.275 0.475+0.200
Pleural effusion 0.641 0.748+0.107
Heart findings (2)
Cardiomegaly 0.440 0.452 +0.012
Pericardial effusion 0.112 0.132 +0.020
Aorta findings (2)
Arterial wall calc. 0.654 0.688 +0.034
Coronary artery calc. 0.592 0.666 +0.073
Esophagus findings (1)
Hiatal hernia 0.261 0.358+0.097
Other findings (2)
Lymphadenopathy 0.329 0.207−0.122
Medical material 0.240 0.066−0.174
Weighted avg (Clin-F1) 0.455 0.480 +0.025
sample. This regime provides high-quality supervision signals while exposing the
model to realistic retrieved context in the remaining 30% of cases.
C Additional Results
C.1 Per-finding clinical F1 breakdown
Table9decomposestheaggregateClinicalF1improvementintoindividualpatho-
logical findings. AdaRAG-CT improves F1 on 13 of 18 findings. The largest gains
occur for findings that require fine-grained descriptive detail: pulmonary fibrotic
sequela (+0.200), pleural effusion (+0.107), and hiatal hernia (+0.097). These
are findings where the base model’s impoverished visual channel fails to encode
sufficientdetail,andtheretrievedtextualcontextsuppliesthemissingsemantics.
Thefewregressions(medicalmaterial−0.174,lymphadenopathy−0.122)involve
low-prevalence findings whose retrieval candidates are sparse or dominated by
negative templates.

Beyond the Embedding Bottleneck 23
Table 10:Reproduced baselines under our unified evaluation protocol (CT-RATE
validation set).
Clinical Efficacy NLP
Method Params F1 P R B-1 B-4 R-L MET LLaMA
CT-CHAT [15] 8B 0.224 0.331 0.366 0.421 0.188 0.303 0.193 6.73
CT-CHAT [15] 70B 0.161 0.334 0.131 0.356 0.182 0.321 0.190 6.02
Ours (base) 8B 0.455 0.474 0.469 0.463 0.205 0.315 0.206 7.30
AdaRAG-CT 8B 0.480 0.502 0.520 0.496 0.242 0.354 0.246 7.75
Ours (base) 70B 0.405 0.468 0.373 0.449 0.213 0.334 0.208 7.10
AdaRAG-CT 70B 0.426 0.483 0.413 0.497 0.250 0.361 0.232 7.53
C.2 Adaptive trigger statistics
On the CT-RATE validation set (2,987 reports), the 8B AdaRAG-CT model
emits an average of 1.48[RAG]triggers per report (median 1). 22% of reports
receive zero triggers, indicating that the model judges retrieval unnecessary for
straightforward cases, while 61% receive 1–2 triggers and 17% receive 3 or more.
Nearly all triggers fire during the lung paragraph, the organ with the lowest
retrieval precision and the highest number of distinct findings. This confirms
that the adaptive mechanism has learned to concentrate retrieval where the
visual bottleneck is most severe.
C.3 Reproduced baselines
Toensureafaircomparisonunderconsistentevaluationconditions,wereproduce
CT-CHATunderourevaluationpipelineandreportallmetricsalongsideourown
models in Tab. 10. The relative ranking is preserved under our protocol, while
our base and adaptive models remain substantially stronger on clinical efficacy.