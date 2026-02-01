# Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing

**Authors**: Zhuchenyang Liu, Ziyu Hu, Yao Zhang, Yu Xiao

**Published**: 2026-01-27 22:50:11

**PDF URL**: [https://arxiv.org/pdf/2601.20107v1](https://arxiv.org/pdf/2601.20107v1)

## Abstract
Recent Vision-Language Models (e.g., ColPali) enable fine-grained Visual Document Retrieval (VDR) but incur prohibitive index vector size overheads. Training-free pruning solutions (e.g., EOS-attention based methods) can reduce index vector size by approximately 60% without model adaptation, but often underperform random selection in high-compression scenarios (> 80%). Prior research (e.g., Light-ColPali) attributes this to the conclusion that visual token importance is inherently query-dependent, thereby questioning the feasibility of training-free pruning. In this work, we propose Structural Anchor Pruning (SAP), a training-free pruning method that identifies key visual patches from middle layers to achieve high performance compression. We also introduce Oracle Score Retention (OSR) protocol to evaluate how layer-wise information affects compression efficiency. Evaluations on the ViDoRe benchmark demonstrate that SAP reduces index vectors by over 90% while maintaining robust retrieval fidelity, providing a highly scalable solution for Visual RAG. Furthermore, our OSR-based analysis reveals that semantic structural anchor patches persist in the middle layers, unlike traditional pruning solutions that focus on the final layer where structural signals dissipate.

## Full Text


<!-- PDF content starts -->

Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG
Indexing
Zhuchenyang Liu, Ziyu Hu, Yao Zhang, Yu Xiao
Aalto University
Espoo, Finland
zhuchenyang.liu@aalto.fi
Abstract
Recent Vision-Language Models (e.g., Col-
Pali) enable fine-grained Visual Document Re-
trieval (VDR) but incur prohibitive index vec-
tor size overheads. Training-free pruning solu-
tions (e.g., EOS-attention based methods) can
reduce index vector size by approximately 60%
without model adaptation, but often underper-
form random selection in high-compression
scenarios ( ≥80% ). Prior research (e.g., Light-
ColPali) attributes this to the conclusion that
visual token importance is inherently query-
dependent, thereby questioning the feasibil-
ity of training-free pruning. In this work, we
propose Structural Anchor Pruning (SAP), a
training-free pruning method that identifies key
visual patches from middle layers to achieve
high performance compression. We also in-
troduce Oracle Score Retention (OSR) proto-
col to evaluate how layer-wise information af-
fects compression efficiency. Evaluations on
the ViDoRe benchmark demonstrate that SAP
reduces index vectors by over 90% while main-
taining robust retrieval fidelity, providing a
highly scalable solution for Visual RAG. Fur-
thermore, our OSR-based analysis reveals that
semantic structural anchor patches persist in
the middle layers, unlike traditional pruning
solutions that focus on the final layer where
structural signals dissipate.
1 Introduction
Visual Document Retrieval (VDR) has shifted
from traditional pipelines to end-to-end Vision-
Language Models (VLMs) (Zhang et al., 2024).
VLM-based retrievers like ColPali (Faysse et al.,
2024) achieve superior precision by representing
documents as bags of visual patch embeddings.
While this paradigm captures rich document struc-
tures through late interaction (Khattab and Zaharia,
2020), it suffers from massive index size overhead.
Addressing this scalability challenge through em-
bedding compression is essential for deploying Vi-
sual RAG in realistic, large-scale scenarios.
Figure 1:Comparison of Pruning Mechanisms.The
left panel illustrates Final Layer EOS Attention, often
fails to capture semantic structure. The right panel de-
picts ourStructural Anchor Pruning, which utilizes
In-Degree Centralitywithin theMiddle Layersof the
LLM backbone. This approach effectively identifies and
preserves semantic structural anchor patches.
To address this bottleneck, recent research has
diverged into two primary streams. One involves
training-based methods like Light-ColPali (Ma
et al., 2025) which are effective but require
fully re-training and additional model modifica-
tions. Alternatively, training-free methods like
DocPruner (Yan et al., 2025) offer a modular so-
lution, but these methods often suffer from perfor-
mance degradation in high-compression regimes
(≥80% reduction) (Ma et al., 2025; Yan et al.,
2025). Observing these challenges, (Ma et al.,
2025) conclude that visual token importance forarXiv:2601.20107v1  [cs.CV]  27 Jan 2026

Figure 2:The Mechanics of SAP.We illustrate theAlignment-Aggregation Divergence. Unlike final layers where
global signals decay due to MaxSim optimization, the middle layers naturally aggregate information into high-
centrality semantic structural anchor patches. SAP exploits this by measuring theIn-Degree Centralitybetween
visual tokens to identify key patches without query supervision.
pruning is inherently query-dependent, thereby ar-
guing that training-free pruning is insufficient for
high compression ratio.
In this work, we challenge this conclusion.
Firstly, we propose a training-free, query-agnostic
methodStructural Anchor Pruning (SAP). Un-
like prior methods, SAP identifies key tokens from
middle layers in the VLM backbone, preserving
the document’s intrinsic semantic structural anchor
patches (see Figure 1). Extensive evaluations on
the ViDoRe dataset (Faysse et al., 2024; Macé et al.,
2025) confirm that SAP consistently outperforms
EOS-Adaptive (Yan et al., 2025), Random (Ma
et al., 2025), and Semantic Clustering (Ma et al.,
2025) baselines. Notably, our method reduces the
number of stored vectors by over 90% compared
to the full-page index while maintaining robust re-
trieval fidelity. This offers a scalable, zero-shot
solution for efficient visual RAG.
We also introduce theOracle Score Retention
(OSR)protocol, a diagnostic metric designed to
isolate intrinsic information retention from cor-
pus information. Through this, we uncover the
Alignment-Aggregation Divergence, illustrated
in Figure 2. We observe that as the model ap-
proaches its final layers, it optimizes for sparse
query alignment, causing global structural informa-
tion to dissipate. Consequently, final-layer signals
become poor proxies for document structure. In
contrast, middle layers effectively provide informa-
tive structural signal for effective pruning.
2 Related Work and Preliminaries
2.1 VDR Multi-Vector Late Interaction
Visual Document Retrieval has recently shifted
from traditional OCR-based pipelines to end-to-end
Vision-Language Models (VLMs) (Zhang et al.,2024). Unlike dense retrievers that map a document
to a single vector, models such as ColPali (Faysse
et al., 2024) employ aMulti-Vector Late Interac-
tionmechanism (Khattab and Zaharia, 2020).
Formally, a document image Dis encoded
into a bag of visual patch embeddings ED=
{v1, . . . , v N} ∈RN×d, where Nrepresents the
sequence length (typically 1024 patches per im-
age). Given a text query Qencoded as tokens
{q1, . . . , q M}, the relevance score is computed via
the MaxSim operator:
S(Q, D) =MX
i=1Nmax
j=1(qi·vj)(1)
This mechanism bridges visual perception and se-
mantic retrieval by preserving fine-grained layout
details. However, it necessitates indexing the full
matrix ED, leading to index vector size that scales
linearly with N. For realistic corpora, this results in
terabytes of index vector storage (Xu et al., 2025),
creating the fundamental bottleneck that necessi-
tates the pruning strategies discussed below.
2.2 Efficient Visual Document Retrieval
To mitigate the index vector size overhead defined
in Section 2.1, recent research has diverged into
two primary streams: training-based adaptation and
training-free pruning.
Training-based Adaptation.Methods like Light-
ColPali (Ma et al., 2025) employ knowledge distil-
lation to train adapters that merge visual tokens into
a smaller set of latent representations. While effec-
tive at high compression ratios, these approaches
introduce significant operational overhead, requir-
ing large-scale of training datasets and full-model
fine-tuning, which limits their zero-shot applicabil-
ity to new architectures.

Figure 3:Overview of SAP.We compare three pruning paradigms on the ColPali architecture.Left:The shared
Vision-Language backbone processes the image.Middle:Conventional methods (Random Selection, Final Layer
EOS Attention) fail to identify critical tokens, resulting in lower retention.Right:Our proposed SAP method
identifies semantic structural anchor patches via In-Degree Centrality in the model’s middle layers, achieving high
retrieval performance retention on the ViDoRe v2 benchmark by preserving the document’s semantic structure.
Bottom:We illustrate theOracle Score Retentionprotocol, a white-box diagnostic used to validate our hypothesis.
This metric directly compares the MaxSim scores of pruned versus full embeddings, isolating intrinsic information
loss from corpus-dependent ranking noise.
Training-free Compression.Conversely, training-
free methods select a subset of informative patches
ˆED⊂E Dwithout model adaptation. Current
strategies typically rely on method signals: (1)
Random Pruning(Ma et al., 2025) assumes a
holographic distribution of visual information and
selects patches via uniform pruning; (2)Seman-
tic Clustering(Ma et al., 2025) aims to reduce
redundancy by grouping embeddings via K-Means
and indexing only the cluster centroids; (3)EOS-
Attention(Ma et al., 2025) selects patches based
on their cross-attention weights with the final
[EOS] token; and (4)EOS-Adaptive Pruning
(DocPruner)(Yan et al., 2025) extends this by
dynamically adjusting the pruning ratio based on
information density. However, these training-free
methods suffer from severe performance degrada-
tion when pushed to high compression ratios (e.g.,
>80% reduction) (Yan et al., 2025; Ma et al.,
2025). Consequently, they argue that static, query-
agnostic pruning strategies are insufficient for high
ratio compression (Ma et al., 2025).
3 Methodology
To address the limitations of existing training-free
pruning—specifically their degradation in high-
compression, we firstly introduceStructural An-chor Pruning. By shifting focus from the retrieval-
aligned final layers to the middle layers, SAP chal-
lenges the prevailing assumption that visual token
importance is inherently query-dependent. Sec-
ondly, to rigorously validate the theoretical basis
of our method, we establish theOracle Score Re-
tentionprotocol. Unlike standard ranking metrics
which confound information loss with corpus noise,
OSR provides a diagnostic metric to analyze in-
dependent retrieval score retention. The overall
framework of our approach is illustrated in Fig-
ure 3.
3.1 Structural Anchor Pruning
We propose SAP, a training-free strategy designed
to extract the intrinsic semantic structure of a doc-
ument image. We identify semantic structural an-
chor patches by measuring the visual In-Degree
Centrality of tokens within theLarge language
Model (LLM) backbone. We hypothesize that
semantic structural patches in middle layers acting
as information hubs, aggregating features from nu-
merous other regions, constitute the core semantic
representation of the document.
Visual In-Degree Centrality.We treat the self-
attention mechanism within the LLM layers at any

given layer las a directed graph, where nodes rep-
resent image patches and edges represent attention
weights. Mechanistically, the attention weight Aij
represents the importance score of token jfor to-
keni. Consequently, the summation over all query
indices,P
iAij, quantifies the total importance of
token jacross all visual tokens, serving as a direct
proxy for its global influence.
To isolate the visual structure, we restrict our
calculation to theVisual-to-Visualattention, mask-
ing out attention scores involving text tokens (e.g.,
system prompts). Let A(l,h)∈RT×Tbe the full
attention matrix for head hover sequence length T,
andVbe the set of indices corresponding to visual
patches. The importance of a visual patch j∈ V at
layerlis defined by its column-sum:
c(l,h)
j=X
i∈VA(l,h)
ij (2)
A high in-degree indicates that patch jacts as a
central aggregator within the visual modality.
Head Aggregation.To synthesize signals across
theHattention heads at layer l, we propose two
variants:
•SAP-Mean:Computes the average centrality,
prioritizing anchors consistently active across
the attention subspace.
S(l)
mean(j) =1
HHX
h=1c(l,h)
j (3)
•SAP-Max:Captures the peak prominence of
a token within its most dominant attention
head, preventing strong local signals from be-
ing diluted.
S(l)
max(j) = max
hc(l,h)
j (4)
Layer Integration.Standard pruning approaches
typically rely on the final layer ( l=L total) under
the assumption that it represents the most refined
semantic state. However, we hypothesize that mid-
dle layers function as anaggregation phase, where
tokens actively exchange information to build a co-
hesive structural understanding of the document.
Conversely, the final layers shift towards analign-
ment phase, where representations are implicitly
reorganized to optimize the contrastive retrieval
objective (MaxSim). This final alignment often
"sparsifies" the attention map to fit potential querydistributions, thereby degrading the intrinsic struc-
tural signals required for effective pruning.
To capture the robust structural core before this
degradation occurs, we introduceLayer Integra-
tion. We define the layer ensemble L∗as a function
of the model’s total depth Ltotal and relative depth
hyperparametersα, β∈[0,1]:
L∗(α, β) ={l∈N| ⌊α·L total⌋ ≤l≤ ⌊β·L total⌋}
(5)
where 0≤α < β≤1 define the boundaries of
the structural window (typically the middle block).
The final importance score SSAP(j)is obtained by
averaging the centrality scores across this window:
SSAP(j) =1
|L∗|X
l∈L∗S(l)(j)(6)
3.2 Oracle Score Retention
While standard evaluation metrics like Normalized
Discounted Cumulative Gain (NDCG) (Wang et al.,
2013) are essential for assessing retrieval effective-
ness, they are insufficient for diagnosing the intrin-
sic fidelity of pruned representations. Formally,
NDCG at positionkis defined as:
NDCG@k=1
IDCG kkX
i=12reli−1
log2(i+ 1)(7)
where reliis the relevance score, iis the ranking
position, and IDCG kis the Ideal Discounted Cu-
mulative Gain, acting as a normalization factor to
ensure the score lies in [0,1] . Crucially, the loga-
rithmic term log2(i+1) explicitly couples the eval-
uation to the relative rank i. This makes the metric
inherently corpus-dependent: a drop in NDCG may
result from the presence of hard negatives shift-
ing the rank i, rather than a loss of information in
the document representation itself. Consequently,
NDCG confounds pure information loss with the
model’s discriminative capacity by measuring prun-
ing performance.
To disentangle these factors and isolate the in-
trinsic visual information retained by a pruning
method, we introduce theOracle Evaluation Pro-
tocol. We define fidelity not by ranking position,
but by the preservation of the raw MaxSim score.
For a given query-document pair, we define the
Oracle Score Retentionas:
R(ˆED, ED) =PM
i=1maxv∈ˆED(qi·v)
PM
i=1max v∈ED(qi·v)(8)

This metric functions as a diagnostic indicator:
a retention of 1.0confirms that the pruned patches
ˆEDpreserve the exact visual features triggered by
the query, independent of the document’s ranking
relative to distractors.
4 Evaluation
In this section, we comprehensively benchmark
the performance of SAP on large-scale visual re-
trieval tasks. We evaluate SAP’s ability to maintain
high retrieval fidelity across diverse architectures
and datasets, compare it against state-of-the-art
training-free and training-based baselines, and as-
sess its computational efficiency.
4.1 Evaluation Setup
Diverse VLM Backbones.We employ three dis-
tinct VLM architectures to evaluate SAP:Col-
Pali(SigLIP + PaliGemma) (Beyer et al., 2024;
Faysse et al., 2024), representing the standard fixed-
patch retrieval paradigm, andColQwen2(NaViT
+ Qwen2-VL) (Wang et al., 2024; Faysse et al.,
2024), representing a dynamic-resolution architec-
ture with a deeper backbone. We also extend our
evaluation to a SOTA modelJina Embeddings
v4with Qwen2.5-VL backbone (Günther et al.,
2025; Bai et al., 2025). Incorporating these archi-
tectures allows us to assess the generalizability of
SAP across different VLM backbones and distinct
embedding optimization recipes. See Appendix A
for model architectural specifications.
Layer Integration Initialization.To ensure
zero-shot adaptability across varying backbone
depths, we instantiate the layer ensemble L∗(Eq.
6) using a simple fixed Geometric Central Win-
dow of 40%∼60% relative depth ( L∗={l|
⌊0.4L total⌋ ≤l≤ ⌊0.6L total⌋}). Specific layer
indices for each model are detailed inAppendix B.
Baselines.We compare ourSAP-Meanand
SAP-Maxagainst three distinct training-free prun-
ing paradigms (details in Appendix C):
(1) Adaptive-EOS: An EOS attention-based
method proposed by DocPruner (Yan et al., 2025),
which employs document-specific thresholding
based on final-layer global ( [EOS] ) attention scores.
To ensure fair comparison, we apply a quantile-
based calibration to align its global retention rate
strictly with our fixed-ratio methods; (2) Random:
The robust stochastic pruning baseline (Ma et al.,
2025); and (3) Semantic Cluster: K-Means cluster-
ing on final embeddings, identified by (Ma et al.,2025) as the state-of-the-art training-free compres-
sion approach.
Datasets.We utilize the fullViDoRe v1(Faysse
et al., 2024) andViDoRe v2(Macé et al., 2025)
benchmarks. These cover a wide spectrum of do-
mains. Detailed dataset statistics are provided in
Appendix D.
4.2 Main Results on ViDoRe
Table 1 presents the aggregated performance across
all datasets. Appendix E shows detailed evaluation
results for each sub-datasets and models.
Performance Consistency across High Compres-
sion Regimes.SAP demonstrates stability across
varying degrees of sparsity. As detailed in Table 1,
at the aggressive retention ratios of γ= 0.20 and
γ= 0.10 , our method consistently retains around
95% and 90% of the original retrieval performance
across benchmarks.
Universality across Architectures.SAP deliv-
ers consistent gains across different retrieval VLM
backbones. Notably, on the Jina v4 (Günther
et al., 2025) architecture, SAP achieves substan-
tial NDCG retention on both ViDoRe v1 and the
more challenging ViDoRe v2, outperforming other
baselines. Consequently, SAP offers a plug-and-
play compression solution that generalizes zero-
shot to diverse architectures without requiring any
model-specific tuning.
4.3 Efficiency-Fidelity Trade-off
Figure 4 illustrates the NDCG@5 retention per-
formance across a broad spectrum of compression
ratios. SAP demonstrates remarkable robustness
under a large range of compression regimes. While
methods like Cluster-Merge and Adaptive-EOS suf-
fer from rapid degradation as the keep ratio de-
creases, SAP maintains high fidelity from moderate
(γ= 0.9 ) down to aggressive ( γ= 0.1 ) sparsity
levels. This stability confirms that our structural
anchor identification is effective regardless of the
target storage constraint.
4.4 Computational Efficiency
Beyond retrieval fidelity, SAP maintains high oper-
ational throughput. Theoretical complexity analy-
sis and empirical benchmarks (Appendix F.2) show
that SAP variants add negligible overhead to the
total forward pass latency. In contrast, clustering-

Model MethodUpper Boundγ= 0.20γ= 0.10γ= 0.05
Full NDCG S.Ret NDCG % S.Ret NDCG % S.Ret NDCG %
Benchmark: ViDoRe v1 (Avg. across 10 datasets)
ColPaliEOS-Adaptive
0.850.86 0.76 89.03 0.79 0.69 80.60 0.72 0.62 71.32
Random 0.91 0.80 94.27 0.85 0.76 89.36 0.78 0.71 83.22
Cluster 0.89 0.82 96.75 0.80 0.78 91.05 0.69 0.67 78.35
SAP-Mean (Ours)0.94 0.83 97.43 0.89 0.79 93.08 0.83 0.75 87.47
SAP-Max (Ours) 0.94 0.83 98.05 0.89 0.80 93.52 0.83 0.75 87.59
ColQwen2EOS-Adaptive
0.880.84 0.77 87.44 0.75 0.70 78.98 0.65 0.63 70.05
Random 0.86 0.82 93.44 0.78 0.77 87.13 0.68 0.69 78.09
Cluster 0.82 0.84 95.55 0.70 0.79 90.20 0.59 0.72 81.99
SAP-Mean (Ours) 0.91 0.85 96.36 0.84 0.80 91.24 0.75 0.73 82.86
SAP-Max (Ours)0.91 0.84 96.14 0.84 0.80 90.55 0.75 0.73 82.26
Jina Embeddings v4EOS-Adaptive
0.900.86 0.85 94.49 0.79 0.77 85.39 0.70 0.65 71.93
Random 0.89 0.86 95.32 0.82 0.81 90.20 0.74 0.75 82.19
Cluster 0.83 0.86 96.01 0.73 0.81 90.40 0.62 0.75 83.20
SAP-Mean (Ours) 0.91 0.87 96.93 0.85 0.83 92.10 0.78 0.76 84.29
SAP-Max (Ours)0.91 0.87 96.93 0.85 0.82 91.43 0.76 0.74 81.74
Benchmark: ViDoRe v2 (Avg. across 4 datasets)
ColPaliEOS-Adaptive
0.560.86 0.44 78.60 0.79 0.38 68.96 0.70 0.32 56.94
Random 0.90 0.49 86.89 0.84 0.44 78.04 0.77 0.39 69.28
Cluster 0.88 0.47 84.00 0.78 0.39 69.41 0.67 0.29 52.04
SAP-Mean (Ours)0.930.52 92.57 0.88 0.48 86.150.810.44 78.32
SAP-Max (Ours) 0.940.51 91.09 0.88 0.48 86.130.820.44 78.26
ColQwen2EOS-Adaptive
0.540.84 0.47 87.27 0.75 0.41 76.50 0.66 0.33 62.35
Random 0.87 0.46 85.61 0.78 0.41 75.96 0.69 0.36 66.16
Cluster 0.80 0.45 84.05 0.70 0.39 72.16 0.58 0.32 58.59
SAP-Mean (Ours)0.89 0.48 88.61 0.820.44 81.440.740.40 75.18
SAP-Max (Ours) 0.89 0.49 90.79 0.820.44 81.130.740.40 74.82
Jina Embeddings v4EOS-Adaptive
0.580.89 0.53 90.91 0.82 0.47 81.47 0.73 0.38 66.41
Random 0.90 0.51 88.46 0.83 0.46 78.49 0.75 0.39 67.83
Cluster 0.83 0.47 80.81 0.73 0.40 67.98 0.63 0.33 57.07
SAP-Mean (Ours)0.91 0.55 95.60 0.85 0.51 88.490.78 0.45 77.14
SAP-Max (Ours) 0.91 0.56 97.18 0.85 0.52 89.240.78 0.43 73.33
Table 1:Main Results on ViDoRe Benchmarks.Comparative analysis against baselines across three architectures
and two benchmark suites.Upper Bounddenotes the full model performance ( γ= 1.0 ). The%column indicates
the relative NDCG retention (Pruned
Full×100).
Figure 4:Efficiency-Fidelity Trade-off.Impact of
pruning ratio on NDCG@5 Retention across ColPali,
ColQwen2, and JinaEmbeddingsV4 on ViDoRe v2.
SAP methods (green) exhibit exceptional stability, sig-
nificantly outperforming clustering and other pruning
baselines at low keep ratios.based method incurs a significant 6%computa-
tional overhead.
4.5 Comparison with Trained Method
We benchmark SAP againstLight-ColPali(Ma
et al., 2025), a state-of-the-art method that requires
supervised training to merge visual tokens.
As shown in Table 2, SAP demonstrates remark-
able efficiency. While the trained method excels
at extreme compression ( 25×) via feature fusion,
SAP remains robust at high compression regimes
(4×and9×). This highlights that semantic struc-
tural anchor patches naturally capture the majority
of retrieval signals, offering a compelling zero-cost
alternative that eliminates the need for architectural
modifications and full-model re-training.
4.6 OSR as a Reliable Proxy
As illustrated in Figure 5, we assess the efficacy
of our diagnostic protocol by observing a strong
linear correlation (Pearson r= 0.635 ) between

Table 2:Training-Free vs. Trained Compression.We
compare SAP against Light-ColPali (Faysse et al., 2024)
(Trained Merging method). We report NDCG@5 and
the relative retention percentage. Note that the Upper
Bounds (Full Model performance) differ slightly due to
evaluation environments.
Backbone MethodTraining Compression Factor
Req.4×9×25×
ColPaliLight-ColPali†Yes0.75 98.7 0.75 98.2 0.72 94.8
SAP-Mean (Ours)No0.76 98.2 0.71 92.8 0.63 81.5
SAP-Max (Ours)No0.76 98.7 0.71 92.5 0.62 79.9
ColQwen2Light-ColQwen2†Yes0.82 99.7 0.81 98.8 0.80 97.5
SAP-Mean (Ours)No0.79 96.9 0.75 91.9 0.64 77.9
SAP-Max (Ours)No0.80 97.3 0.75 91.7 0.65 78.8
†Results cited from Light-ColPali (Faysse et al., 2024). Subscript denotes % retention of full model.
Figure 5:Oracle Score Retention is a Strong Proxy
for Retrieval Performance.We observe a significant
positive correlation between intrinsic score preservation
and downstream ranking utility. Each data point corre-
sponds to a unique evaluation configuration defined by
the model architecture, compression ratio, and dataset
subset.
the intrinsic Oracle Score Retention and the final
NDCG performance. The plot reveals a distinct
separation of methods: while baseline approaches
like Adaptive-EOS and Random (hollow markers)
exhibit higher variance, our SAP variants (solid
markers) consistently occupy the top-right “High-
Fidelity Region”, where pruned representations
maintaining both high NDCG and Oracle Score
Retention.
5 Alignment-Aggregation Divergence
Having established the superior retrieval perfor-
mance of SAP and validated the OSR as a reliable
proxy, we now turn to the mechanistic question:
Why does the semantic signal necessary for prun-
ing decouple from the final retrieval embedding?
In this section, we utilize OSR as a "white-box"probe to scan the information retention capabilities
across the model’s depth. This diagnostic reveals
theAlignment-Aggregation Divergence—a phe-
nomenon where the model’s structural understand-
ing peaks in middle layers before degrading as it
aligns with the sparse retrieval objective.
To isolate the location of structural information,
we apply the OSR protocol to every layer of the
LLM backbone inColPali(Beyer et al., 2024;
Faysse et al., 2024) andColQwen2(Wang et al.,
2024; Faysse et al., 2024) architectures. We utilize
theViDoRebenchmark (Faysse et al., 2024), focus-
ing on five diverse subsets that categorize document
morphology: Text-Centric (ArxivQA (Li et al.,
2024), DocVQA (Mathew et al., 2021)), Structure-
Centric (TabFQuAD, TAT-DQA (Zhu et al., 2022)),
and Layout-Centric (InfoVQA (Mathew et al.,
2022)). This diversity allows us to test if seman-
tic structural anchor patches remain stable across
different visual modalities. As visualized in Figure
6, our analysis uncovers two distinct phases in the
VLM’s internal processing.
The Aggregation Phase (The Structural
Plateau).In the middle layers, we observe a sus-
tained peak in retention scores across all document
morphologies—a region of stability we identify
as theStructural Plateau(highlighted in blue in
Figure 6). Mechanistically, this corresponds to the
formation of structural anchor patches, where the
model aggregates local visual features into high-
centrality anchors to build a global understanding
of the document. This morphological robustness
indicates that the document’s "semantic core" is
naturally concentrated in these middle layers.
The Alignment Phase (Final Layers).Ap-
proaching the final layers, the retention metric
exhibits a pronounced decline. We attribute this
to theLate-Interaction MaxSim objective. We
hypothesize that to maximize contrastive separa-
bility, the model reorganizes its representation to
align strictly with potential query tokens, implic-
itly "sparsifying" the information. This suggests
that while beneficial for retrieval ranking, such op-
timization results in the loss of dense structural
context accumulated in the middle layers, render-
ing the final attention weights suboptimal proxies
for visual importance.
SAP vs. EOS AttentionThis layer-wise diver-
gence explains the limitations of prior pruning
methods. As shown in Figure 6, EOS-Attention

Figure 6:Morphological Robustness Analysis.Oracle Score Retention curves decomposed by document type
at a high-retention ratio ( γ= 0.1 ). We observe a universal "Structural Plateau" in the middle layers regardless of
document morphology (Text, Tables, or Layouts), in contrast to the information loss observed in final layers.
(which relies on the final layer) consistently yields
lower scores compared to SAP, and often falls be-
low the random pruning baseline. This empirical
gap serves as an indication that the MaxSim train-
ing objective decouples the global [EOS] token
from the document’s structural content.
6 Conclusion
In this work, we address the critical index scalabil-
ity bottleneck inherent to multi-vector VLM-based
VDR systems. Challenging the prevailing view
that training-free pruning is insufficient for high
compression due to query dependency, we propose
Structural Anchor Pruning (SAP). This zero-shot,
query-agnostic approach successfully reduces in-
dex storage by over 90% while maintaining robust
retrieval fidelity, consistently outperforming exist-
ing baselines on the ViDoRe benchmark. Further-
more, through our Oracle Score Retention (OSR)
protocol, we uncover the underlying Alignment-
Aggregation Divergence, demonstrating that un-
like the sparse, alignment-optimized final layers,
middle-layer representations retain essential seman-
tic structural signals, thereby offering a highly effi-
cient and scalable solution for Visual RAG.
7 Limitations
While SAP offers a scalable solution for Visual
RAG, our scope is currently confined to the multi-vector late-interaction paradigm, leaving its gen-
eralizability to broader image-text matching tasks
and large-scale industrial indices to be fully ex-
plored. Methodologically, the framework relies on
empirically fixed parameters for layer selection and
enforces a uniform token budget across all docu-
ments. Future research could address this rigidity
by developing dynamic mechanisms that adaptively
select LLM backbone layers and adjust index vec-
tor capacity based on instance-specific document
complexity, enabling flexible model adaptation and
variable compression rates across diverse document
types.
References
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wen-
bin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie
Wang, Jun Tang, and 1 others. 2025. Qwen2. 5-vl
technical report.arXiv preprint arXiv:2502.13923.
Lucas Beyer, Andreas Steiner, André Susano Pinto,
Alexander Kolesnikov, Xiao Wang, Daniel Salz,
Maxim Neumann, Ibrahim Alabdulmohsin, Michael
Tschannen, Emanuele Bugliarello, and 1 others. 2024.
Paligemma: A versatile 3b vlm for transfer.arXiv
preprint arXiv:2407.07726.
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Om-
rani, Gautier Viaud, Céline Hudelot, and Pierre
Colombo. 2024. Colpali: Efficient document re-
trieval with vision language models.arXiv preprint
arXiv:2407.01449.

Michael Günther, Saba Sturua, Mohammad Kalim
Akram, Isabelle Mohr, Andrei Ungureanu, Bo Wang,
Sedigheh Eslami, Scott Martens, Maximilian Werk,
Nan Wang, and 1 others. 2025. jina-embeddings-v4:
Universal embeddings for multimodal multilingual
retrieval. InProceedings of the 5th Workshop on
Multilingual Representation Learning (MRL 2025),
pages 531–550.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval, pages 39–
48.
Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong
Feng, Lingpeng Kong, and Qi Liu. 2024. Multimodal
arxiv: A dataset for improving scientific comprehen-
sion of large vision-language models.arXiv preprint
arXiv:2403.00231.
Yubo Ma, Jinsong Li, Yuhang Zang, Xiaobao Wu, Xi-
aoyi Dong, Pan Zhang, Yuhang Cao, Haodong Duan,
Jiaqi Wang, Yixin Cao, and 1 others. 2025. Towards
storage-efficient visual document retrieval: An empir-
ical study on reducing patch-level embeddings.arXiv
preprint arXiv:2506.04997.
Quentin Macé, António Loison, and Manuel Faysse.
2025. Vidore benchmark v2: Raising the bar for
visual retrieval.arXiv preprint arXiv:2505.17166.
Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthe-
nis Karatzas, Ernest Valveny, and CV Jawahar. 2022.
Infographicvqa. InProceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vi-
sion, pages 1697–1706.
Minesh Mathew, Dimosthenis Karatzas, and CV Jawa-
har. 2021. Docvqa: A dataset for vqa on document
images. InProceedings of the IEEE/CVF winter con-
ference on applications of computer vision, pages
2200–2209.
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhi-
hao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin
Wang, Wenbin Ge, and 1 others. 2024. Qwen2-
vl: Enhancing vision-language model’s perception
of the world at any resolution.arXiv preprint
arXiv:2409.12191.
Yining Wang, Liwei Wang, Yuanzhi Li, Di He, and Tie-
Yan Liu. 2013. A theoretical analysis of ndcg type
ranking measures. InConference on learning theory,
pages 25–54. PMLR.
Mengyao Xu, Gabriel Moreira, Ronay Ak, Radek Os-
mulski, Yauhen Babakhin, Zhiding Yu, Benedikt
Schifferer, and Even Oldridge. 2025. Llama nemore-
triever colembed: Top-performing text-image re-
trieval model.arXiv preprint arXiv:2507.05513.
Yibo Yan, Guangwei Xu, Xin Zou, Shuliang Liu, James
Kwok, and Xuming Hu. 2025. Docpruner: A storage-
efficient framework for multi-vector visual documentretrieval via adaptive patch-level embedding pruning.
arXiv preprint arXiv:2509.23883.
Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu.
2024. Vision-language models for vision tasks: A
survey.IEEE transactions on pattern analysis and
machine intelligence, 46(8):5625–5644.
Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang,
Haozhou Zhang, and Tat-Seng Chua. 2022. Towards
complex document understanding by discrete reason-
ing. InProceedings of the 30th ACM International
Conference on Multimedia, pages 4857–4866.

A Model Architectures
To ensure the universality of ourAlignment-
Aggregation Divergencehypothesis, we selected
three Vision-Language Models (VLMs) that repre-
sent distinct design paradigms in the current land-
scape. Table 3 summarizes their architectural spec-
ifications.
Model Backbone Layers Params
ColPali PaliGemma-3B 18 3B
ColQwen2 Qwen2-VL-2B 28 2B
Jina v4 Qwen2.5-VL-3B 36 3B
Table 3:Architectural Summary.The selected models
cover different LLM families (Gemma, Qwen, Llama-
style) and vision encoding strategies. Note the pro-
gression from the fixed-resolution SigLIP encoder in
PaliGemma to the dynamic-resolution capabilities in-
herent in the Qwen2-VL series.
A.1 ColPali
ColPali( ViDoRe/colpali-v1.31) represents the
pioneering architecture for late-interaction visual
retrieval, establishing the foundation for the Visual
Document Retrieval (ViDoRe) benchmark. It is
built upon thePaliGemma-3Bbackbone, which
uniquely combines aSigLIP-So400mvision en-
coder with the Gemma-2B language model. Un-
like traditional pipelines that rely on OCR to ex-
tract text, ColPali employs a Visual Large Lan-
guage Model (VLLM) approach to generate multi-
vector representations directly from document im-
ages. This allows it to effectively index complex
visual elements—such as figures, charts, and ta-
bles—thereby significantly outperforming standard
dense retrieval methods on visually rich documents.
A.2 ColQwen2
ColQwen2( ViDoRe/colqwen2-v1.02) is based
on the advancedQwen2-VL-2Barchitecture. This
model introduces significant complexity and ar-
chitectural improvements over ColPali, primarily
through its support fornative dynamic resolu-
tion. While ColPali typically resizes inputs to fixed
square patches (often distorting document aspect ra-
tios), ColQwen2 leverages the Naive Dynamic Res-
olution mechanism inherent to Qwen2-VL. This
allows the model to process images of varying di-
mensions and aspect ratios without information
1https://huggingface.co/ViDoRe/colpali-v1.3
2https://huggingface.co/ViDoRe/colqwen2-v1.0loss, resulting in superior visual fidelity and more
efficient visual token usage during the indexing of
high-resolution PDFs.
A.3 Jina Embeddings v4
TheJina Embeddings v4model
(jinaai/jina-embeddings-v43) represents
the state-of-the-art application of late-interaction
principles to the powerfulQwen2.5-VLar-
chitecture. By transitioning to the Qwen2.5
backbone, this iteration offers enhanced optical
character recognition (OCR) capabilities and
improved geometric reasoning for structured data.
Furthermore, it incorporates Jina AI’s signature
Matryoshka Representation Learning (MRL),
enabling flexible embedding dimensions that allow
users to trade off index vector size efficiency
against retrieval precision. This model aims to
unify multimodal retrieval by supporting extended
context windows and delivering high-performance
indexing for both textual and visual-heavy datasets.
B Detailed Layer Instantiation
To ensure SAP remains calibration-free and zero-
shot, we select the layer ensemble L∗using the
simple fixed Geometric Central Window:
L∗={l∈N| ⌊0.4·L total⌋ ≤l≤ ⌊0.6·L total⌋}
(9)
Table 4 details the specific layers selected for
each architecture used in our experiments. This
method effectively targets the "Structural Plateau"
where visual aggregation peaks, regardless of the
total depth of the backbone.
Parameters ColPali ColQwen2 Jina v4
Total Layers (L) 18 28 36
Range (0.4L∼0.6L)7.2∼10.8 11.2∼16.8 14.4∼21.6
SelectedL∗{7, . . . ,10} {11, . . . ,16} {14, . . . ,21}
Table 4:SAP Layer Ensemble Instantiation.The
specific middle layers used for In-Degree Centrality
calculation across different architectures, derived auto-
matically from the geometric method.
C Baseline Implementation Details
We provide the formal definitions for the baseline
pruning methods used in our comparative analysis.
LetE∈RN×ddenote the sequence of Nvisual
patch embeddings. We select a subset of size K
3https://huggingface.co/jinaai/
jina-embeddings-v4

based on the following importance scoring func-
tionsS(j)for thej-th patch.
1. Random.This method assumes visual infor-
mation is holographically distributed. The selection
is performed via uniform pruning without replace-
ment:
Srandom (j)∼ U(0,1)(10)
To mitigate the impact of randomness, we con-
ducted five independent runs initialized with dis-
tinct random seeds and reported the average perfor-
mance metrics.
2. EOS-Attention.This method assumes that
patches attended to during text generation are most
relevant. We define the score as the cross-attention
weight from the final token (representing [EOS] ) in
the last layerL last:
Seos(j) =1
HHX
h=1A(Llast,h)
eos,j (11)
3. Semantic Clustering (Post-Projector).This
method assumes that visual redundancy can be
reduced by grouping embeddings based on their
representation similarity. Following the architec-
tural insights from Light-ColPali (Ma et al., 2025),
we specifically perform this operation at thePost-
Projectorstage—immediately after the Vision-
LLM’s final linear projection layer.
The empirical study indicates that clustering is
significantly more effective in this low-dimensional
output space (e.g., 128 dimensions) compared to
high-dimensional intermediate representations, as
it enables more targeted feature aggregation with
minimal information loss. We apply K-Means clus-
tering to the set of projected embeddings E=
{v1, . . . , v N}to partition them into Kdisjoint sets
{C1, . . . , C K}. The objective is to minimize the
within-cluster sum of squares (WCSS):
min
{µ1,...,µK}KX
k=1X
vj∈Ck∥vj−µk∥2(12)
where µkis the centroid of cluster Ck. The pruned
representation consists of these Kcentroids, ef-
fectively merging redundant visual features into a
compact, representative set ready for indexing.
4. Adaptive EOS attention.While the standard
EOS-Attention method applies a fixed selection
ratio (Top-K) across all documents, this baseline
adopts a document-aware adaptive thresholdingstrategy inspired by DocPruner (Yan et al., 2025).
This method postulates that the information density
varies across documents, and thus the number of
retained tokens should be dynamic.
For a document d, we compute the mean µd
and standard deviation σdof its patch importance
scores Seos. A patch jis retained if its score ex-
ceeds a statistical threshold:
Seos(j)> µ d+k·σ d (13)
where kis an adaptation factor controlling the ag-
gressiveness of pruning.
Fairness Calibration.To ensure a rigorous com-
parison with fixed-ratio methods (like SAP) at a
specific target retention ratio γ(e.g., 10%), we
do not arbitrarily select k. Instead, we employ
a calibration process. We extract the EOS atten-
tion scores from a held-out calibration set of 128
randomly sampled documents. We convert these
scores into Z-scores zij= (S eos(j)−µ i)/σiand
compute the global empirical quantile:
k=Quantile({z ij}calib,1−γ)(14)
This ensures that the global average retention rate
of this adaptive baseline strictly aligns with the tar-
getγ, isolating the impact of the selection strategy
(Adaptive vs. Fixed) from the index vector size
budget.
D Dataset Details
Table 5 details the composition of the ViDoRe v1
and v2 benchmarks used in our comprehensive eval-
uation.
Benchmark Subset Name Primary Domain
ViDoRe v1ArxivQA Academic / STEM
DocVQA General Document
InfoVQA Infographics / Layout
Shift Project Environmental Reports
Artificial Intelligence Technical Reports
Energy Industry Reports
Government Reports Policy / Legal
Healthcare Industry Medical / Business
ViDoRe v2MIT Biomedical (Multi) Medical / Research
Economics Macro (Multi) Finance / Policy
ESG Restaurant (Multi) Business / Tables
ESG Restaurant (Human) Business / Tables
Table 5:Dataset Specifications.Overview of domains
covered in the ViDoRe benchmark suite.

E Detailed Evaluation Results
The following tables present the detailed perfor-
mance breakdown for ColPali (Table 6), ColQwen
(Table 7), and Jina Embeddings v4 (Table 8) on the
ViDoRe v1 benchmark, and Table 9 on the ViDoRe
v2 benchmark.

Dataset MethodUpper Boundγ= 0.20γ= 0.10γ= 0.05
Full NDCG S.Ret NDCG % S.Ret NDCG % S.Ret NDCG %
ArxivQAEOS-Adaptive
0.820.90 0.77 93.98 0.84 0.72 88.50 0.76 0.65 80.06
Random 0.93 0.80 98.20 0.89 0.78 95.94 0.83 0.76 92.53
Cluster 0.92 0.81 99.08 0.870.80 97.630.800.77 94.68
SAP-Mean 0.95 0.82 99.92 0.91 0.80 97.360.860.77 94.01
SAP-Max0.95 0.82 99.95 0.910.80 97.52 0.85 0.76 93.35
DocVQAEOS-Adaptive
0.580.91 0.50 86.99 0.86 0.43 73.34 0.81 0.34 58.79
Random 0.94 0.53 91.60 0.90 0.49 84.87 0.84 0.44 76.25
Cluster 0.92 0.56 96.02 0.86 0.51 87.27 0.77 0.44 75.71
SAP-Mean0.97 0.56 96.65 0.94 0.52 90.27 0.89 0.48 82.03
SAP-Max 0.97 0.56 96.41 0.93 0.52 90.08 0.89 0.47 81.04
InfoVQAEOS-Adaptive
0.850.86 0.77 90.95 0.79 0.73 85.37 0.73 0.68 80.15
Random 0.91 0.82 96.36 0.86 0.79 92.76 0.79 0.76 89.14
Cluster 0.90 0.84 98.19 0.82 0.81 94.61 0.71 0.72 85.01
SAP-Mean 0.94 0.84 98.50 0.88 0.80 93.780.82 0.76 89.31
SAP-Max0.94 0.84 99.07 0.88 0.81 94.810.82 0.76 89.07
Shift ProjectEOS-Adaptive
0.780.82 0.61 78.03 0.74 0.44 57.18 0.65 0.36 46.85
Random 0.90 0.66 84.67 0.83 0.59 76.49 0.76 0.52 66.92
Cluster 0.85 0.67 86.59 0.74 0.52 67.30 0.61 0.35 45.39
SAP-Mean 0.93 0.71 90.99 0.87 0.62 79.31 0.79 0.52 66.64
SAP-Max0.93 0.74 94.91 0.87 0.64 82.24 0.79 0.54 69.71
Artificial Intel.EOS-Adaptive
0.970.87 0.94 96.40 0.81 0.88 90.58 0.73 0.81 83.26
Random 0.90 0.95 97.83 0.84 0.92 94.36 0.770.88 90.46
Cluster 0.880.97 99.630.790.93 96.230.66 0.81 82.94
SAP-Mean 0.94 0.96 98.39 0.89 0.91 93.91 0.82 0.86 89.03
SAP-Max0.950.96 98.970.890.93 95.690.830.88 90.42
EnergyEOS-Adaptive
0.950.83 0.82 86.16 0.75 0.77 81.26 0.68 0.72 76.15
Random 0.90 0.92 96.98 0.84 0.88 92.94 0.76 0.81 85.12
Cluster 0.880.94 99.080.800.92 97.170.67 0.82 86.89
SAP-Mean0.950.93 98.40 0.88 0.89 94.170.81 0.87 91.42
SAP-Max 0.94 0.93 98.290.880.91 95.34 0.80 0.86 90.30
Gov. ReportsEOS-Adaptive
0.960.86 0.90 93.69 0.78 0.85 87.81 0.70 0.74 76.43
Random 0.90 0.92 95.95 0.84 0.88 91.63 0.76 0.83 85.73
Cluster 0.88 0.95 98.34 0.78 0.92 95.33 0.65 0.78 80.55
SAP-Mean 0.95 0.94 98.02 0.90 0.95 98.52 0.830.90 93.70
SAP-Max0.95 0.96 99.37 0.90 0.95 98.57 0.840.89 92.79
HealthcareEOS-Adaptive
0.970.86 0.91 93.91 0.80 0.88 90.83 0.73 0.81 83.59
Random 0.91 0.94 96.89 0.84 0.91 93.86 0.78 0.87 90.38
Cluster 0.88 0.94 97.66 0.78 0.91 93.84 0.64 0.70 72.10
SAP-Mean0.95 0.95 98.780.910.95 97.89 0.860.94 96.97
SAP-Max 0.95 0.95 98.650.910.93 96.29 0.850.94 97.31
TabFQuadEOS-Adaptive
0.870.82 0.81 93.56 0.74 0.77 88.04 0.66 0.68 77.91
Random 0.91 0.85 97.85 0.85 0.83 95.23 0.78 0.81 92.83
Cluster 0.91 0.87 100.45 0.850.86 99.040.780.85 98.23
SAP-Mean 0.94 0.87 100.16 0.88 0.85 97.47 0.79 0.80 92.00
SAP-Max0.94 0.88 100.71 0.890.85 97.850.800.82 94.26
TAT-DQAEOS-Adaptive
0.720.85 0.55 76.65 0.78 0.45 63.08 0.72 0.36 50.07
Random 0.89 0.62 86.33 0.81 0.54 75.50 0.73 0.45 62.85
Cluster 0.86 0.66 92.44 0.76 0.59 82.11 0.64 0.44 61.96
SAP-Mean0.93 0.68 94.44 0.88 0.63 88.16 0.82 0.57 79.60
SAP-Max 0.93 0.67 94.17 0.87 0.62 86.84 0.82 0.56 77.67
Table 6:Detailed Performance: ColPali on ViDoRe v1.Detailed metrics across all 10 datasets. The strongest
method in each column isbolded.

Dataset MethodUpper Boundγ= 0.20γ= 0.10γ= 0.05
Full NDCG S.Ret NDCG % S.Ret NDCG % S.Ret NDCG %
ArxivQAEOS-Adaptive
0.860.84 0.74 85.69 0.75 0.67 78.10 0.64 0.55 64.13
Random 0.91 0.83 95.93 0.84 0.80 92.94 0.76 0.75 87.17
Cluster 0.870.83 96.480.790.83 95.540.680.79 91.24
SAP-Mean0.920.82 95.06 0.86 0.79 91.37 0.77 0.73 83.93
SAP-Max 0.92 0.82 94.710.860.79 91.540.770.73 84.55
DocVQAEOS-Adaptive
0.590.87 0.49 83.85 0.79 0.39 66.94 0.71 0.29 48.66
Random 0.86 0.51 87.26 0.77 0.44 75.49 0.66 0.37 62.53
Cluster 0.82 0.56 94.29 0.71 0.51 86.52 0.590.46 77.71
SAP-Mean 0.93 0.56 94.70 0.87 0.51 86.790.790.44 74.29
SAP-Max0.93 0.57 96.08 0.87 0.51 87.440.79 0.43 73.46
InfoVQAEOS-Adaptive
0.910.83 0.82 89.67 0.74 0.77 84.81 0.65 0.71 77.74
Random 0.86 0.85 93.58 0.78 0.80 88.28 0.69 0.74 81.63
Cluster 0.810.87 95.240.69 0.81 89.03 0.57 0.72 78.74
SAP-Mean0.900.86 94.190.83 0.82 90.38 0.75 0.77 84.75
SAP-Max 0.90 0.85 93.87 0.82 0.81 88.73 0.74 0.76 83.37
Shift ProjectEOS-Adaptive
0.850.83 0.67 78.47 0.75 0.58 67.85 0.67 0.51 60.11
Random 0.84 0.74 87.33 0.76 0.64 75.65 0.66 0.52 61.47
Cluster 0.79 0.75 87.68 0.66 0.65 76.06 0.53 0.50 58.64
SAP-Mean0.91 0.85 99.41 0.83 0.78 91.92 0.73 0.64 75.39
SAP-Max 0.90 0.83 97.68 0.82 0.75 88.55 0.71 0.60 70.02
Artificial Intel.EOS-Adaptive
0.980.81 0.85 86.68 0.74 0.82 83.13 0.65 0.79 80.08
Random 0.86 0.96 97.36 0.78 0.92 93.62 0.68 0.84 85.45
Cluster 0.810.98 99.100.700.94 95.700.590.88 89.31
SAP-Mean 0.91 0.96 97.440.830.93 94.30 0.74 0.84 85.11
SAP-Max0.910.97 98.16 0.82 0.91 92.080.740.87 87.91
EnergyEOS-Adaptive
0.960.82 0.82 85.32 0.71 0.70 72.86 0.60 0.65 68.10
Random 0.86 0.89 93.07 0.78 0.85 89.06 0.68 0.77 80.66
Cluster 0.820.93 97.000.700.90 94.380.58 0.83 86.49
SAP-Mean0.920.91 94.870.850.85 88.92 0.75 0.82 86.05
SAP-Max 0.91 0.90 94.62 0.84 0.86 90.250.75 0.84 87.51
Gov. ReportsEOS-Adaptive
0.950.84 0.92 97.15 0.74 0.86 90.99 0.64 0.77 81.31
Random 0.87 0.93 98.01 0.78 0.90 94.88 0.69 0.83 87.38
Cluster 0.81 0.91 96.23 0.69 0.87 92.41 0.57 0.83 87.74
SAP-Mean0.92 0.94 98.91 0.84 0.92 97.02 0.76 0.87 92.49
SAP-Max 0.91 0.93 98.54 0.84 0.91 96.75 0.74 0.87 91.75
HealthcareEOS-Adaptive
0.980.84 0.94 96.25 0.76 0.91 92.64 0.67 0.83 85.26
Random 0.87 0.96 98.17 0.79 0.93 94.66 0.69 0.87 88.40
Cluster 0.810.98 100.330.69 0.93 94.66 0.57 0.85 87.06
SAP-Mean0.920.97 99.370.86 0.96 98.09 0.77 0.94 95.91
SAP-Max 0.92 0.97 99.16 0.85 0.96 97.98 0.77 0.93 95.12
TabFQuadEOS-Adaptive
0.880.87 0.82 92.97 0.78 0.76 85.79 0.68 0.70 79.32
Random 0.87 0.86 97.26 0.79 0.82 93.05 0.70 0.77 87.07
Cluster 0.85 0.86 97.80 0.75 0.85 96.60 0.64 0.83 93.81
SAP-Mean0.92 0.87 99.03 0.860.84 95.820.780.80 91.14
SAP-Max 0.92 0.87 98.44 0.850.85 96.690.77 0.80 90.94
TAT-DQAEOS-Adaptive
0.810.83 0.64 78.35 0.72 0.54 66.68 0.61 0.45 55.84
Random 0.82 0.70 86.41 0.72 0.60 73.66 0.61 0.48 59.17
Cluster 0.790.74 91.310.670.66 81.150.540.56 69.15
SAP-Mean0.900.74 90.610.800.63 77.840.670.48 59.50
SAP-Max 0.89 0.73 90.13 0.79 0.61 75.45 0.66 0.47 57.99
Table 7:Detailed Performance: ColQwen on ViDoRe v1.Detailed metrics across all 10 datasets. The strongest
method in each column isbolded.

Dataset MethodUpper Boundγ= 0.20γ= 0.10γ= 0.05
Full NDCG S.Ret NDCG % S.Ret NDCG % S.Ret NDCG %
ArxivQAEOS-Adaptive
0.880.90 0.84 94.94 0.82 0.76 85.73 0.72 0.62 70.50
Random 0.92 0.85 96.72 0.86 0.82 93.02 0.79 0.77 86.97
Cluster 0.880.87 98.800.810.85 95.830.720.82 92.86
SAP-Mean 0.93 0.85 96.12 0.87 0.81 91.690.800.75 84.56
SAP-Max0.930.86 97.520.870.81 91.42 0.79 0.71 80.10
DocVQAEOS-Adaptive
0.610.87 0.56 90.81 0.80 0.46 74.90 0.70 0.34 55.38
Random 0.87 0.54 87.91 0.79 0.49 79.23 0.70 0.40 65.91
Cluster 0.84 0.56 92.06 0.73 0.49 80.17 0.60 0.43 69.62
SAP-Mean0.930.59 96.430.88 0.56 91.06 0.80 0.49 79.21
SAP-Max 0.930.60 97.620.87 0.55 90.44 0.78 0.46 74.28
InfoVQAEOS-Adaptive
0.920.84 0.84 91.15 0.75 0.75 81.73 0.64 0.59 64.34
Random 0.87 0.88 95.92 0.800.84 90.890.720.77 83.62
Cluster 0.830.89 96.390.71 0.83 90.20 0.59 0.74 80.44
SAP-Mean0.900.88 95.560.830.83 90.720.750.76 82.18
SAP-Max 0.90 0.88 95.43 0.83 0.83 90.66 0.73 0.74 80.27
Shift ProjectEOS-Adaptive
0.890.85 0.87 96.77 0.78 0.77 85.72 0.72 0.67 75.24
Random 0.88 0.84 93.46 0.81 0.76 84.96 0.72 0.65 72.33
Cluster 0.81 0.82 91.32 0.70 0.73 81.37 0.59 0.58 64.90
SAP-Mean 0.92 0.90 100.730.86 0.86 95.92 0.78 0.79 87.77
SAP-Max0.92 0.91 101.700.85 0.84 94.08 0.77 0.75 84.32
Artificial Intel.EOS-Adaptive
0.990.84 0.97 98.03 0.77 0.88 88.94 0.69 0.76 76.94
Random 0.89 0.97 98.21 0.82 0.94 94.61 0.73 0.88 88.59
Cluster 0.81 0.97 97.90 0.71 0.92 93.11 0.60 0.90 90.43
SAP-Mean0.90 0.98 98.39 0.840.94 95.180.770.90 90.60
SAP-Max 0.90 0.97 98.19 0.840.95 96.160.760.91 91.31
EnergyEOS-Adaptive
0.960.85 0.90 93.77 0.78 0.83 86.57 0.69 0.72 75.31
Random 0.89 0.94 98.05 0.82 0.90 93.67 0.74 0.83 86.51
Cluster 0.810.95 98.590.710.91 94.280.600.84 87.17
SAP-Mean0.900.93 97.310.840.87 90.620.770.80 83.04
SAP-Max 0.90 0.91 94.22 0.84 0.85 88.49 0.76 0.78 81.04
Gov. ReportsEOS-Adaptive
0.970.85 0.94 97.11 0.78 0.90 92.61 0.70 0.81 83.35
Random 0.89 0.95 97.97 0.820.92 94.480.74 0.85 87.56
Cluster 0.820.95 98.090.71 0.92 94.41 0.60 0.87 89.15
SAP-Mean0.910.94 97.330.850.92 94.290.780.87 89.44
SAP-Max 0.91 0.95 97.56 0.84 0.91 93.89 0.770.87 89.68
HealthcareEOS-Adaptive
0.980.850.97 99.380.78 0.91 92.88 0.71 0.84 85.97
Random 0.89 0.97 98.78 0.82 0.95 96.39 0.74 0.90 91.53
Cluster 0.81 0.97 98.64 0.70 0.90 92.10 0.60 0.87 88.77
SAP-Mean 0.91 0.97 99.120.85 0.96 97.63 0.78 0.92 93.65
SAP-Max0.910.97 98.99 0.85 0.95 97.12 0.77 0.89 90.93
TabFQuadEOS-Adaptive
0.960.89 0.92 96.47 0.82 0.88 92.56 0.72 0.78 81.92
Random 0.90 0.94 98.11 0.84 0.92 96.010.760.88 92.09
Cluster 0.85 0.94 98.16 0.770.93 97.500.670.90 94.70
SAP-Mean 0.910.94 98.170.84 0.90 94.15 0.75 0.84 88.07
SAP-Max0.910.93 97.310.850.89 93.49 0.74 0.83 87.17
TAT-DQAEOS-Adaptive
0.790.89 0.68 86.50 0.81 0.57 72.21 0.71 0.40 50.38
Random 0.88 0.70 88.11 0.81 0.62 78.77 0.72 0.53 66.82
Cluster 0.81 0.71 90.13 0.710.67 85.060.610.58 73.93
SAP-Mean0.920.71 90.180.860.63 79.780.770.51 64.36
SAP-Max 0.920.72 90.710.85 0.62 78.51 0.75 0.46 58.33
Table 8:Detailed Performance: Jina Embeddings v4 on ViDoRe v1.Detailed metrics across all 10 datasets. The
strongest method in each column isbolded.

Dataset MethodUpper Boundγ= 0.20γ= 0.10γ= 0.05
Full NDCG S.Ret NDCG % S.Ret NDCG % S.Ret NDCG %
Model: ColPali
MIT BiomedicalEOS-Adaptive
0.570.87 0.48 83.58 0.80 0.42 72.62 0.73 0.36 63.28
Random 0.93 0.55 95.98 0.89 0.52 90.80 0.82 0.48 83.37
Cluster 0.92 0.55 95.54 0.86 0.51 89.29 0.78 0.46 79.61
SAP-Mean0.94 0.55 96.30 0.89 0.52 91.31 0.83 0.49 85.99
SAP-Max 0.94 0.55 96.29 0.89 0.52 90.92 0.83 0.49 85.34
Econ. MacroEOS-Adaptive
0.510.83 0.43 83.66 0.76 0.40 78.39 0.69 0.36 69.73
Random 0.88 0.45 87.83 0.81 0.41 79.75 0.73 0.39 75.75
Cluster 0.87 0.44 85.07 0.76 0.36 70.91 0.64 0.27 52.47
SAP-Mean 0.920.48 94.41 0.87 0.45 87.330.800.43 83.32
SAP-Max0.930.46 90.36 0.87 0.45 87.130.810.42 82.67
ESG Rest. (Multi)EOS-Adaptive
0.560.87 0.42 76.09 0.80 0.37 65.79 0.71 0.28 51.04
Random 0.90 0.47 84.43 0.83 0.42 74.43 0.75 0.35 63.27
Cluster 0.86 0.42 76.02 0.76 0.30 54.10 0.62 0.17 30.79
SAP-Mean 0.930.49 87.030.88 0.45 80.53 0.81 0.38 67.71
SAP-Max0.940.49 86.970.89 0.47 84.95 0.82 0.39 69.81
ESG Rest. (Human)EOS-Adaptive
0.600.87 0.43 71.07 0.79 0.35 59.04 0.69 0.26 43.70
Random 0.91 0.47 79.33 0.84 0.41 68.22 0.76 0.33 55.37
Cluster 0.86 0.47 79.36 0.76 0.38 63.35 0.64 0.27 45.29
SAP-Mean0.94 0.55 92.530.880.51 85.430.810.46 76.28
SAP-Max 0.94 0.54 90.740.890.49 81.540.820.45 75.20
Model: ColQwen2
MIT BiomedicalEOS-Adaptive
0.540.82 0.45 83.45 0.75 0.41 75.32 0.67 0.36 66.86
Random 0.90 0.51 94.81 0.83 0.48 89.18 0.76 0.45 83.03
Cluster 0.85 0.51 94.23 0.75 0.48 87.90 0.64 0.43 79.20
SAP-Mean 0.91 0.51 94.870.86 0.50 91.54 0.78 0.47 87.72
SAP-Max0.91 0.51 95.060.85 0.48 89.68 0.78 0.47 86.92
Econ. MacroEOS-Adaptive
0.480.80 0.43 89.22 0.71 0.39 80.70 0.62 0.36 74.41
Random 0.85 0.42 86.70 0.77 0.37 76.50 0.68 0.34 69.95
Cluster 0.79 0.41 85.31 0.67 0.34 71.11 0.55 0.27 57.17
SAP-Mean0.88 0.46 97.000.80 0.42 88.510.72 0.43 89.62
SAP-Max 0.88 0.45 93.560.80 0.43 89.320.72 0.41 86.46
ESG Rest. (Multi)EOS-Adaptive
0.570.89 0.57 100.630.810.49 85.200.70 0.36 62.74
Random 0.86 0.46 80.19 0.78 0.39 68.26 0.67 0.30 52.78
Cluster 0.79 0.45 79.06 0.68 0.37 64.06 0.58 0.27 48.12
SAP-Mean 0.88 0.50 88.21 0.81 0.46 80.60 0.72 0.38 65.79
SAP-Max 0.89 0.53 93.590.820.46 79.860.74 0.39 67.70
ESG Rest. (Human)EOS-Adaptive
0.570.84 0.43 75.76 0.74 0.37 64.78 0.64 0.26 45.38
Random 0.850.4680.76 0.760.40 69.900.660.34 58.86
Cluster 0.79 0.44 77.62 0.68 0.37 65.56 0.57 0.29 49.87
SAP-Mean 0.89 0.43 74.38 0.81 0.37 65.11 0.73 0.33 57.60
SAP-Max0.900.4680.94 0.820.38 65.670.730.33 58.20
Model: Jina Embeddings v4
MIT BiomedicalEOS-Adaptive
0.610.90 0.58 94.49 0.81 0.51 83.41 0.70 0.43 70.35
Random 0.91 0.58 95.83 0.86 0.55 90.55 0.790.51 84.40
Cluster 0.86 0.57 94.12 0.79 0.55 89.86 0.70 0.48 78.97
SAP-Mean 0.92 0.58 95.97 0.87 0.56 92.36 0.80 0.51 83.77
SAP-Max0.92 0.59 96.67 0.87 0.57 93.040.79 0.49 80.50
Econ. MacroEOS-Adaptive
0.550.86 0.51 93.57 0.80 0.50 90.14 0.71 0.42 77.19
Random 0.88 0.48 87.39 0.81 0.43 77.96 0.73 0.38 68.70
Cluster 0.81 0.43 77.96 0.70 0.35 62.81 0.60 0.30 54.01
SAP-Mean 0.90 0.55 99.440.84 0.52 94.34 0.770.44 80.50
SAP-Max0.90 0.56 102.730.83 0.51 91.97 0.760.45 81.42
ESG Rest. (Multi)EOS-Adaptive
0.530.90 0.44 84.04 0.84 0.40 75.34 0.75 0.35 66.78
Random 0.89 0.46 86.05 0.82 0.40 75.20 0.74 0.33 63.05
Cluster 0.82 0.41 78.30 0.72 0.34 63.48 0.63 0.27 51.72
SAP-Mean 0.91 0.51 95.81 0.85 0.47 89.650.78 0.39 73.79
SAP-Max0.91 0.52 97.93 0.85 0.48 90.690.77 0.34 64.66
ESG Rest. (Human)EOS-Adaptive
0.640.92 0.58 91.550.85 0.49 77.01 0.75 0.33 51.34
Random 0.89 0.54 84.56 0.83 0.45 70.26 0.74 0.35 55.18
Cluster 0.81 0.46 72.86 0.70 0.35 55.78 0.60 0.28 43.57
SAP-Mean 0.92 0.58 91.160.860.49 77.620.80 0.45 70.52
SAP-Max 0.92 0.58 91.39 0.860.52 81.280.79 0.42 66.75
Table 9:Detailed Performance on ViDoRe v2.SAP consistently outperforms baselines across most datasets and
models. The best performance in each column isbolded.

F Computational Complexity &
Efficiency Analysis
A primary concern for any indexing strategy is the
additional latency introduced during the document
processing phase. In this section, we formally ana-
lyze the computational overhead of Structural An-
chor Pruning (SAP) and provide empirical bench-
marks on the ViDoRe v2 dataset.
F.1 Theoretical Complexity
LetNbe the number of visual patches (e.g., 1024
for standard inputs), Lthe number of transformer
layers, anddthe hidden dimension.
Backbone Cost.The computational cost of the
standard forward pass is dominated by the self-
attention mechanism and feed-forward networks.
The complexity for the attention mechanism alone
across all layers isO(L·N2·d).
SAP Overhead.SAP operates by extracting at-
tention matrices A(l,h)from a subset of layers L∗.
The operations required are:
1.Extraction:Accessing attention logits (ef-
fectively zero FLOPs, bounded by memory
bandwidth).
2.Aggregation (In-Degree):Summing
columns of the attention matrix. For a
selected layer set |L∗|and heads H, the
complexity isC SAP=O(|L∗| ·H·N2).
Comparing the two, the ratio of SAP overhead
to the attention computation is approximately:
CSAP
CAttn≈|L∗| ·H·N2
L·H·N2·d=|L∗|
L·d(15)
For the jina-embeddings-v4 model used in our
experiments, with d= 1280 , this ratio is exceed-
ingly small ( <10−3), implying the theoretical cost
is negligible.
F.2 Empirical Benchmarks
To validate our theoretical analysis, we conducted
a rigorous latency benchmark using theViDoRe v2
dataset (subsets:esg_reports, biomedical_lectures,
economics_reports). Experiments were performed
on a singleNVIDIA H200 (141GB)GPU. The
backbone model is jina-embeddings-v4 (hidden
sized= 1280).
We measure thePruning Latency(time taken
to compute masks and select tokens) and compare
it against theFull Forward Passtime. The results
are summarized in Table 10.Method Avg Time (ms) Overhead (+ %)
Full Forward Pass206.05(Baseline)
Random 0.04 +0.02%
Adaptive-EOS 0.08 +0.04%
SAP-Max (Ours) 0.05 +0.03%
SAP-Mean (Ours) 0.06 +0.03%
Cluster-Merge 12.08 +5.86%
Table 10:Efficiency Benchmark on ViDoRe v2.La-
tency represents the time per page for the pruning op-
eration (mask generation) only.Overheadis calcu-
lated relative to the Full Forward Pass time ( 206.05 ms).
SAP variants introduce negligible overhead ( <0.03% ),
whereas clustering-based methods incur a significant
penalty (≈6%).
Results Analysis.As shown in Table 10, the Full
Forward Pass requires approximately 206ms per
page.
•Negligible Overhead:SAP-Mean and SAP-
Max require only 0.06ms and 0.05ms respec-
tively. This corresponds to an overhead of
approximately 0.03% relative to the model
inference. In a real-world pipeline, this is im-
perceptible.
•Comparison to Clustering:Iterative meth-
ods likeCluster-Merge(K-Means) are signifi-
cantly slower, taking ≈12 ms per page. While
feasible, this represents a ∼200× slowdown
compared to SAP and adds nearly 6%to the
total indexing time.
•Comparison to Random:SAP achieves com-
parable speed toRandomselection ( 0.04ms)
while providing the semantic benefits detailed
in Section 4.
These results confirm that SAP is a highly scal-
able solution suitable for high-throughput Visual
RAG systems processing millions of documents.
G Detailed Comparison with Trained
Methods
In Table 11, we provide a fine-grained break-
down of the comparison between SAP and Light-
ColPali/Light-ColQwen2 (Faysse et al., 2024)
across individual datasets.

Table 11:Detailed Dataset Breakdown: SAP vs. Light-Baselines.We compare the trained token merging
baselines (Top) with our training-free SAP variants (Bottom) across 6 datasets. We report NDCG@5 and the
percentage of the full model’s performance retained (subscript).
Model Method Factor InfoVQA DocVQA ArxivQA TabFQuAD TAT-DQA ShiftProj Average
Trained Baselines (Token Merging)
Light
ColPaliMerging4×0.83 98.1 0.53 97.4 0.84 98.8 0.87 101.4 0.73 100.7 0.73 96.0 0.75 98.7
Merging9×0.82 97.3 0.55 100.0 0.84 98.1 0.85 99.1 0.71 98.1 0.73 96.4 0.75 98.2
Merging25×0.81 96.2 0.51 92.2 0.83 97.1 0.83 97.0 0.67 92.9 0.71 93.6 0.72 94.8
Light
ColQwen2Merging4×0.90 97.8 0.57 102.2 0.89 100.7 0.90 99.7 0.81 99.3 0.87 98.4 0.82 99.7
Merging9×0.90 98.0 0.56 101.3 0.87 98.5 0.89 98.1 0.79 97.5 0.87 98.6 0.81 98.8
Merging25×0.89 97.2 0.55 98.6 0.86 98.2 0.89 98.7 0.79 97.0 0.84 95.4 0.80 97.5
Ours (Training-Free Structural Anchor Pruning)
ColPaliSAP-Mean4×0.84 98.8 0.57 98.0 0.82 99.8 0.87 100.0 0.69 96.4 0.75 96.5 0.76 98.2
9×0.81 94.6 0.53 92.0 0.80 98.4 0.85 97.4 0.64 89.5 0.66 84.6 0.71 92.8
25×0.75 88.0 0.46 79.9 0.74 90.8 0.79 90.5 0.54 75.1 0.50 64.5 0.63 81.5
SAP-Max4×0.85 99.2 0.57 97.8 0.82 99.7 0.87 100.3 0.69 96.2 0.77 98.7 0.76 98.7
9×0.81 95.2 0.53 91.5 0.81 98.5 0.86 98.8 0.63 87.7 0.65 83.4 0.71 92.5
25×0.74 87.2 0.45 76.8 0.74 91.0 0.78 89.8 0.53 73.5 0.48 61.3 0.62 79.9
ColQwen2SAP-Mean4×0.88 96.4 0.56 95.1 0.83 96.2 0.88 100.1 0.77 94.8 0.84 99.1 0.79 96.9
9×0.84 92.6 0.53 90.1 0.80 92.5 0.85 96.8 0.69 85.0 0.80 94.1 0.75 91.9
25×0.76 83.9 0.42 72.1 0.72 83.0 0.78 88.7 0.52 63.7 0.64 75.7 0.64 77.9
SAP-Max4×0.88 96.2 0.56 95.4 0.83 96.3 0.88 99.8 0.77 94.8 0.86 101.2 0.80 97.3
9×0.83 91.6 0.53 89.8 0.79 91.5 0.85 96.9 0.71 86.6 0.80 93.7 0.75 91.7
25×0.76 83.1 0.44 74.9 0.73 84.1 0.78 88.7 0.52 63.5 0.67 78.5 0.65 78.8