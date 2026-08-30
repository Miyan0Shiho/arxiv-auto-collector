# How AI Experiences Art: Emergent Aesthetic Structure in a Self-Supervised Multimodal Embedding Space

**Authors**: Corey D. C. Heath

**Published**: 2026-08-27 13:34:18

**PDF URL**: [https://arxiv.org/pdf/2608.27121v1](https://arxiv.org/pdf/2608.27121v1)

## Abstract
Aesthetics are an important part of the symbolism of artistic works. Although subjective, humans categorize art based on the emotion evoked regardless of modality. What remains under-explored is how AI models form their own aesthetic categorization of human-produced media without explicit labels or cross-modal supervision. We present a self-supervised framework that projects four modalities (text, audio, image and video) into a shared 256-dimensional embedding space and applies iterative clustering to discover aesthetic structure. We discuss the divergence between AI-generated cluster assignments and human affective register labels on a weakly supervised multimodal dataset. This work has applications in understanding how AI structures cross-modal similarity, organizing heterogeneous media collections for Retrieval-Augmented Generation (RAG), and automated data labeling.

## Full Text


<!-- PDF content starts -->

How AI Experiences Art: Emergent Aesthetic Structure in a
Self-Supervised Multimodal Embedding Space
Corey D.C. Heath
research@coreydheath.me
Independent Researcher
Phoenix, AZ, USA
Abstract
Aesthetics are an important part of the symbolism of artistic works.
Although subjective, humans categorize art based on the emotion
evoked regardless of modality. What remains underexplored is
how AI models form their own aesthetic categorization of human-
produced media without explicit labels or cross-modal supervision.
We present a self-supervised framework that projects four modali-
ties (text, audio, image and video) into a shared 256-dimensional
embedding space and applies iterative clustering to discover aes-
thetic structure. We discuss the divergence between AI-generated
cluster assignments and human affective register labels on a weakly
supervised multimodal dataset. This work has applications in un-
derstanding how AI structures cross-modal similarity, organizing
heterogeneous media collections for Retrieval-Augmented Genera-
tion (RAG), and automated data labeling.
CCS Concepts
•Computing methodologies →Cluster analysis;Dimension-
ality reduction and manifold learning;•Human-centered com-
puting→HCI theory, concepts and models;•Information
systems→Multimedia databases.
ACM Reference Format:
Corey D.C. Heath. 2026. How AI Experiences Art: Emergent Aesthetic
Structure in a Self-Supervised Multimodal Embedding Space. InCompanion
of the INTERNATIONAL CONFERENCE ON MULTIMODAL INTERACTION
(ICMI Companion ’26), October 05–09, 2026, Napoli, Italy.ACM, New York,
NY, USA, 5 pages. https://doi.org/10.1145/3776591.3837060
1 Introduction
Artistic works have the ability to elicit emotion and affect from
the humans who consume them. People’s tastes in art are highly
personalized; however, there is general agreement on how artistic
works can be categorized based on their aesthetic. Foundation AI
models have been trained using human content. Our research seeks
to determine how well AI understanding extends into aesthetics
across different modalities without explicit human labels, and if
these AI categorizations diverge from human categorization in a
meaningful way.
In order to explore AI aesthetic categorization, an architectural
framework was built utilizing AI models to embed multimodal
content featuring text, audio, images and video. These modalities
This work is licensed under a Creative Commons Attribution 4.0 International License.
ICMI Companion ’26, Napoli, Italy
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2319-3/2026/10
https://doi.org/10.1145/3776591.3837060are projected into a shared space and a clustering model is used
to categorize the samples. The architectural models are trained
and evaluated on a weakly supervised dataset of publicly available
media content. The primary contributions of this work are:
(1)A self-supervised framework for utilizing latent space repre-
sentations for AI aesthetic self-organization of multimodal
data.
(2)A divergence analysis between AI and human aesthetic topol-
ogy.
2 Related Works
Self-supervised learning aesthetics has been explored in previous
works [ 5,10,14,19] focusing on photographic quality; however, in
this work we are exploring affective-experiential representations
in media.
The aesthetic classification used in this project is based on the
valence-arousal scale [13]. Picard [11] established a relevant foun-
dation for machine understanding of human emotional states. Re-
search on emotion and affective response from images [ 9] and audio
[20] provide a theoretical backing of media classification based on
aesthetics. Image classification based on the emotional state was
implemented by Machajdik and Hanbury [ 7] using a Naive Bayes
model.
Unlike ImageBind [ 4], which aligns modalities through explicitly
paired training samples using images as a binding anchor, our archi-
tecture requires no-cross modal pairs. Alignment in shared space in
our architecture emerges from the self-supervised clustering signal
alone.
Deep clustering methods such as DeepCluster [ 3] and SCAN [ 15]
extend self-supervised representation learning to cluster discovery
within a single visual modality. Our framework generalizes this
paradigm to four modalities simultaneously without requiring cross-
modal paired sampling.
3 Methodology
A weakly-supervised dataset was created for this project based
on publicly available media. Data procurement was based on six
categories inspired by the valence-arousal scale. The categories
are elegiac, sublime, grotesque, uncanny, idyllic and pastoral. This
particular phrasing was chosen for its alignment with literary and
critical theory tradition. Four types of media were collected: text
(consisting of poems and prose), audio, images and video. For large
audio and video files, only the first 120 seconds are sampled. The
data split is available in Table 1.
Figure 1 shows a diagram of the framework architecture. The
first step was to extract embedding representations of each of the
samples using a specific model for each modality. The backbone
arXiv:2608.27121v1  [cs.MM]  27 Aug 2026

ICMI Companion ’26, October 05–09, 2026, Napoli, Italy Heath
Figure 1: Architectural flow diagram depicting training process.
Table 1: The number of samples in the weakly supervised
dataset by affective register (rows) and modality (columns).
Affective Register Text Audio Image Video
Elegiac840 389 523 380
Sublime789 553 469 521
Grotesque967 421 378 384
Uncanny706 333 408 380
Idyllic791 342 553 381
Pastoral725 268 345 164
models for extracting embeddings from each modality are presented
in Table 2. Each model was run locally on a system using an RTX
3090 GPU.
Table 2: Models utilized to extract embeddings from each
modality.
Modality Model Citation
TextE5-large-v2 Wang et al. [16]
AudioCLAP HTSAT Wu et al. [17]
ImageCLIP ViT-L/14 Radford et al. [12]
VideoV-JEPA 2 ViT-L Assran et al. [1]
After extraction, each modality embedding has its own dimen-
sional space representation. Multi-layer perceptrons (MLP) are used
to project each modality into a 256-d unit-norm representation.
Modality alignment happens through SupConLoss [ 6]. The training
loop has two phases. In Phase 1, The 256-d output vectors for each
sample are given a pseudo-label created by HDBSCAN [ 2]. In Phase
2, the pseudo-labels are held fixed while SupConLoss trains the
MLPs for 50 epochs, pulling same-cluster samples together and
pushing different-cluster samples apart. Gradients flow into the
MLP weights only. Backbone encoders remain frozen. The loop then
returns to Phase 1, re-projecting all items with updated weights
before the next HDBSCAN pass.
The training loop continues until the Adjusted Rand Index (ARI)
is greater than 0.9 between the same samples in sequential iterations
of the training loop. This indicates that the cluster assignments
have become consistent and the model is no longer changing the
organization of data in a meaningful way. After model convergence,
the data will be organized into K clusters. The value of K is deter-
mined by the final pseudo-labeling step performed by HDBSCAN.The centroid for each of the K clusters is computed and passed to
an LLM (Qwen 2.5-14B [18]) for a human-legible description.
For inference, a sample would be encoded by its modality back-
bone model, the embedding would then be passed through the MLP
for its 256-d representation and compared to the cluster centroids
using cosine similarity.
Cluster concept names are generated for human readability by
taking the six items nearest to each cluster center using cosine sim-
ilarity. Textual information for each item, the file name for image,
audio and video samples and a 60-word content excerpt for text sam-
ples, is passed to a Qwen 2.5-14B model to extract a concept label.
The human aesthetic register labels are withheld, allowing concept
name assignment independent of the human category scheme.
4 Results
The self-supervised multimodal clustering framework manages
to organize the data into meaningful partitions without human
supervision. Twenty-eight concept clusters are extracted from the
data. The embedding UMAP [ 8], Figure 2, shows a comparison of
the AI clusters to the embedding clustering based on the six weakly
assigned aesthetic labels. Both AI-discovered and human-aligned
registers show distinct clusters. The AI-discovered clusters, given
their finer granularity (28 vs six), are more homogeneous than the
human-aligned register groups.
Quantifying the divergence, we computed Normalized Mutual
Information (NMI) to be 0.40, Adjusted Rand Index (ARI) is 0.15 and
purity is 0.70 between AI-generated and human-aligned clusters.
The NMI score shows that there is a moderate amount of shared in-
formation between AI-generated clustering and the human-aligned
labels. The low ARI indicates that the AI is discovering a different
structuring of the sample clusters while the high purity indicates
the assignment is not arbitrary.
The far-left periphery contains a large cluster dominated by
two AI-generated concept-labeled samples, Timeless-Nostalgia and
Soulful-Contemplation, that stand out as cross-register exceptions.
In the human-aligned graphic, this cluster is the most heteroge-
neous with sample representations from each of the assignable
labels.
The heat map presented in Figure 3 associates the AI-generated
concepts with the human-aligned labels. The figure shows a tight
overlapping of concepts between the AI and human labels, with
the exception of Timeless-Nostalgia and Soulful-Contemplation.
Figure 4 shows the modal representation of samples assigned to
each of the AI-generated clusters. Of the 28 converged clusters, 25
are anchored predominantly to a single modality, consistent with
each medium carrying distinct perceptual properties, while three

How AI Experiences Art: Emergent Aesthetic Structure in a Self-Supervised Multimodal Embedding Space ICMI Companion ’26, October 05–09, 2026, Napoli, Italy
Figure 2: UMAP Comparison of AI-discovered and Human Registry Embeddings Clusters, with 12,010 samples, using cosine
metric.
show more balanced modality representation. The cross-register
exceptions are not cross-modal clusters but cross-human-register
ones: Timeless-Nostalgia and Soulful-Contemplation draw from all
six aesthetic registers while remaining text-dominated, revealing
aesthetic dimensions that cut across emotional categories rather
than across media boundaries.
At iteration 1, before the training loop has refined the embedding
space, HDBSCAN produces 22 modality-dominated clusters with a
mean dominant-modality purity of 0.72. By convergence, the model
reorganizes into 28 semantically coherent aesthetic clusters (mean
purity 0.77), demonstrating that the cross-register structure cap-
tured by Timeless-Nostalgia and Soulful-Contemplation emerges
through the iterative training rather than from backbone geometry.
Soulful-Contemplation is primarily associated with elegiac, grote-
sque and idyllic. If we consider these on a valence-arousal scale
we pass through three of the four quadrants with elegiac repre-
senting negative valence-low arousal, grotesque being negative
valence-high arousal, and idyllic as positive valence-low arousal.
This is interesting in comparison to the low representation of sub-
lime, uncanny and pastoral. What is potentially being illustrated is
the AI finding a different dimensionality of the data that was not
captured in the human labels. The labels are intended to represent
the emotions evoked by humans while consuming the media. Sub-
lime is typically meant to be grand and overwhelming. Pastoral is
focused on calm external and observational depictions. Uncanny
seeks to foster detachment and other-worldliness. What was cap-
tured by the AI was a more internally motivated and reflective
register, independent of the explicit emotion.
The AI-labeled, Timeless-Nostalgia samples represent a fairly
even distribution of human labels. Queries for text samples focused
on poems and prose, with sites like PoetryDB, Wikisource and
Gutenberg being a large representation of the source. These sources
are largely populated with public domain texts. This suggests thatthe utilization of older styles of language and writing is influencing
the categorization when considering the contemporary data the
large language models were trained on. What may be captured
here is literary temporality. Since the public domain is composed
of seminal and influential works, the AI labeled them as timeless.
This bias in the data may also be responsible for the poor clustering
on the human aesthetic assignments in Figure 2.
5 Conclusion
We presented a self-supervised framework for multimodal aes-
thetic clustering that requires no cross-modal pairs or human su-
pervision using a weakly labeled dataset spanning text, audio,
image and video. The framework converges on 28 semantically
coherent clusters, demonstrating that alignment across modali-
ties can emerge from clustering signal alone. Divergence analysis
(NMI=0.40, ARI=0.15, purity=0.70) reveals that AI-generated struc-
ture partially tracks human aesthetic registers while consistently
discovering finer-grained and orthogonal organization.
A key limitation is that human register labels were assigned at
the collection level rather than per item, introducing noise into the
divergence baseline. Additionally, collecting data from uncurated
sources likely introduced bias, particularly in text, where public
domain archives skew toward canonical Western literary works.
Future work will prioritize curated artistic media collections, item-
level human annotation to establish a cleaner ground truth, and
human validation of AI-generated concept names. We also plan to
evaluate the shared embedding space on downstream cross-modal
retrieval tasks and extend the dataset across cultural traditions and
at larger scale.

ICMI Companion ’26, October 05–09, 2026, Napoli, Italy Heath
Figure 3: AI concept composition by human aesthetic register.
Map is normalized by row and sorted by dominant register
and cluster purity.
Ethics and Privacy Statement
This study strictly utilized publicly available data collected via an
automated web tool designed for public data searches and sample
retrieval. Because the data collection process was limited exclu-
sively to publicly accessible archives and did not involve direct
intervention, interaction, or solicitation with living human subjects,
it did not require formal Institutional Review Board (IRB) oversight.
To ensure compliance with data privacy principles and ethical
research practices, the data collection methodology adhered strictly
to the terms of service, robots.txt protocols, and user agreements of
all sourced web platforms. No personally identifiable information
(PII), proprietary private records, or restricted credentials were
bypassed, scraped, or archived during the collection phase. All
downloaded samples were processed, stored, and analyzed in an
aggregated, de-identified format to fully mitigate any potential
re-identification or privacy risks.
References
[1]Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes,
Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zho-
lus, et al .2025. V-jepa 2: Self-supervised video models enable understanding,
prediction and planning.arXiv preprint arXiv:2506.09985(2025).
Figure 4: Modality composition of AI concepts clusters.
Dashed lines indicate the dataset baseline.
[2]Ricardo JGB Campello, Davoud Moulavi, and Jörg Sander. 2013. Density-based
clustering based on hierarchical density estimates. InPacific-Asia conference on
knowledge discovery and data mining. Springer, 160–172.
[3]Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. 2018.
Deep clustering for unsupervised learning of visual features. InProceedings of
the European conference on computer vision (ECCV). 132–149.
[4]Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev
Alwala, Armand Joulin, and Ishan Misra. 2023. Imagebind: One embedding space
to bind them all. InProceedings of the IEEE/CVF conference on computer vision and
pattern recognition. 15180–15190.
[5]Minrui Jia, Guangao Wang, Zibei Wang, Shuai Yang, Yongzhen Ke, and Kai
Wang. 2025. Self-Supervised image aesthetic assessment based on transformer.
International Journal of Computational Intelligence and Applications24, 01 (2025),
2450029.
[6]Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip
Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive
learning.Advances in neural information processing systems33 (2020), 18661–
18673.
[7]Jana Machajdik and Allan Hanbury. 2010. Affective image classification using
features inspired by psychology and art theory. InProceedings of the 18th ACM
international conference on Multimedia. 83–92.
[8]Leland McInnes, John Healy, and James Melville. 2018. Umap: Uniform man-
ifold approximation and projection for dimension reduction.arXiv preprint
arXiv:1802.03426(2018).
[9]Joseph A Mikels, Barbara L Fredrickson, Gregory R Larkin, Casey M Lindberg,
Sam J Maglio, and Patricia A Reuter-Lorenz. 2005. Emotional category data on
images from the international affective picture system.Behavior research methods
37, 4 (2005), 626–630.
[10] Jan Pfister, Konstantin Kobs, and Andreas Hotho. 2021. Self-supervised multi-task
pretraining improves image aesthetic assessment. InProceedings of the IEEE/CVF
conference on computer vision and pattern recognition. 816–825.

How AI Experiences Art: Emergent Aesthetic Structure in a Self-Supervised Multimodal Embedding Space ICMI Companion ’26, October 05–09, 2026, Napoli, Italy
[11] Rosalind Picard. 1997. W.,(1997). affective computing.Computer Science, Art,
Psychology. Semantic Scholar. DOI10 (1997).
[12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh,
Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
et al.2021. Learning transferable visual models from natural language supervision.
InInternational conference on machine learning. PmLR, 8748–8763.
[13] James A Russell. 1980. A circumplex model of affect.Journal of personality and
social psychology39, 6 (1980), 1161.
[14] Kekai Sheng, Weiming Dong, Menglei Chai, Guohui Wang, Peng Zhou, Feiyue
Huang, Bao-Gang Hu, Rongrong Ji, and Chongyang Ma. 2020. Revisiting image
aesthetic assessment via self-supervised feature learning. InProceedings of the
AAAI Conference on Artificial Intelligence, Vol. 34. 5709–5716.
[15] Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proes-
mans, and Luc Van Gool. 2020. Scan: Learning to classify images without labels.
InEuropean conference on computer vision. Springer, 268–285.
[16] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2022. Text embeddings by weakly-supervisedcontrastive pre-training.arXiv preprint arXiv:2212.03533(2022).
[17] Yusong Wu, Ke Chen, Tianyu Zhang, Yuchen Hui, Taylor Berg-Kirkpatrick, and
Shlomo Dubnov. 2023. Large-scale contrastive language-audio pretraining with
feature fusion and keyword-to-caption augmentation. InICASSP 2023-2023 IEEE
International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE,
1–5.
[18] An Yang, Baosong Yang, Bowen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengpeng Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al .2024. Qwen2.5
Technical Report.arXiv preprint arXiv:2412.15115(2024).
[19] Shuai Yang, Zibei Wang, Guangao Wang, Yongzhen Ke, Fan Qin, Jing Guo, and
Liming Chen. 2024. A self-supervised image aesthetic assessment combining
masked image modeling and contrastive learning.Journal of Visual Communica-
tion and Image Representation101 (2024), 104184.
[20] Marcel Zentner, Didier Grandjean, and Klaus R Scherer. 2008. Emotions evoked
by the sound of music: characterization, classification, and measurement.Emotion
8, 4 (2008), 494.