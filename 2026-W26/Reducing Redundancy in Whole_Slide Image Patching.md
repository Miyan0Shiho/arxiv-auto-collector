# Reducing Redundancy in Whole-Slide Image Patching for Scalable Indexing and Retrieval

**Authors**: Jialiang Geng, Ghazal Alabtah, Saghir Alfasly, Wataru Uegami, H. R. Tizhoosh

**Published**: 2026-06-23 20:57:22

**PDF URL**: [https://arxiv.org/pdf/2606.26157v1](https://arxiv.org/pdf/2606.26157v1)

## Abstract
The rapid growth of digital pathology has created an urgent need for efficient indexing and retrieval of whole slide images (WSIs). This need is intensified by emerging generative AI workflows, particularly retrieval-augmented generation (RAG), which require dependable similarity search to support high-stakes clinical decision-making. Yet the substantial cost of high-performance storage limits the scalability and accessibility of WSI indexing for many healthcare institutions. Consequently, methods that can reduce storage demands while preserving retrieval accuracy have become a critical research priority. We propose ARReST (Antithetical Redundancy Reduction Strategy), a principled oppositional framework that leverages redundancy across dissimilar tissue classes to markedly decrease the number of patches that must be indexed from each WSI. Instead of eliminating only within-class duplicates, ARReST identifies antithetical patches-those whose representations contribute minimally to cross-class discrimination-and prunes them from the searchable archive. This targeted reduction substantially compresses the index without sacrificing morphological diversity or retrieval fidelity. By minimizing superfluous patch representations, ARReST reduces storage footprint, lowers computational overhead, and accelerates similarity search across large pathology repositories. Extensive experiments on TCGA repository (The Cancer Genome Atlas with 21 organs) demonstrate that ARReST achieves significant index compression while maintaining competitive retrieval performance. The observed storage savings of 3% to 60% (14%$\pm$13%) can be reliably achieved without compromising retrieval performance for many organs. The proposed strategy enables scalable, cost-efficient WSI indexing and is well-suited for next-generation retrieval-driven clinical AI systems.

## Full Text


<!-- PDF content starts -->

Reducing Redundancy
in Whole-Slide Image Patching
for Scalable Indexing and Retrieval
Jialiang Geng, Ghazal Alabtah, Saghir Alfasly, Wataru Uegami, and H.R.Tizhoosh
KIMIA Lab, Dept. of Artificial Intelligence & Informatics,
Mayo Clinic, Rochester, MN, USA
June 26, 2026
Abstract
The rapid growth of digital pathology has created an urgent need for efficient indexing and retrieval
of whole slide images (WSIs). This need is intensified by emerging generative AI workflows, particularly
retrieval-augmented generation (RAG), which require dependable similarity search to support high-stakes
clinical decision-making. Yet the substantial cost of high-performance storage limits the scalability and
accessibility of WSI indexing for many healthcare institutions. Consequently, methods that can reduce
storage demands while preserving retrieval accuracy have become a critical research priority. We propose
ARReST (Antithetical Redundancy Reduction Strategy), a principled oppositional framework that lever-
ages redundancy across dissimilar tissue classes to markedly decrease the number of patches that must be
indexed from each WSI. Instead of eliminating only within-class duplicates, ARReST identifies antitheti-
cal patches—those whose representations contribute minimally to cross-class discrimination—and prunes
them from the searchable archive. This targeted reduction substantially compresses the index without
sacrificing morphological diversity or retrieval fidelity. By minimizing superfluous patch representations,
ARReST reduces storage footprint, lowers computational overhead, and accelerates similarity search
across large pathology repositories. Extensive experiments on TCGA repository (The Cancer Genome
Atlas with 21 organs) demonstrate that ARReST achieves significant index compression while maintain-
ing competitive retrieval performance. The observed storage savings of 3% to 60% (14%±13%) can be
reliably achieved without compromising retrieval performance for many organs. The proposed strategy
enables scalable, cost-efficient WSI indexing and is well-suited for next-generation retrieval-driven clinical
AI systems.
Whole Slide Image, Pathology, Patching, Image Retrieval, Redundancy
1 Introduction
With the increasing number of whole slide images (WSIs) in digital pathology and the growing need for
efficient search and retrieval of large WSI archives, effective indexing has become a critical necessity [3, 9].
After tissue segmentation, patch selection is the first major step in many computational pathology algorithms.
Self-supervision is less suitable for patch selection because it forces models to learn surrogate tasks—such
as predicting rotations or reconstructing masked regions—that do not directly align with identifying diagnos-
tically meaningful or representative patches. Consequently, the selected patches may optimize performance
on the pretext task rather than reflect true tissue relevance, leading to suboptimal indexing and retrieval.
Unsupervised methods are generally preferred in pathology because they eliminate the need for labor-
intensive expert annotations, which are costly and often inconsistent across institutions. They also better
capture the inherent variability of tissue without imposing predefined labels, enabling more generalizable
and scalable representation learning.
1arXiv:2606.26157v1  [cs.IR]  23 Jun 2026

Currently, all unsupervised patch selection techniques, such as mosaic [12], SPLICE [13], and SDM [14],
adopt a “divide” strategy to manage the gigapixel scale of each WSI. These methods break down WSIs
into a compact yet representative subset of patches to facilitate indexing and retrieval. However, significant
bottlenecks and challenges remain. A practical WSI search engine and indexing strategy—with acceptable
diagnostic accuracy—must balance storage requirements, cost efficiency, and the ability to perform rapid
comparisons during both indexing and retrieval.
Democratizationof AI has become a major concern in recent years [4, 5]. In this context, storage costs
and data management have emerged as critical challenges in digital pathology [7, 8]. Several studies have
highlighted the substantial expenses associated with handling and storing WSIs [18,20]. The size of each WSI
varies depending on magnification and resolution [11,19], typically ranging from 0.5 to 2 gigabytes per image.
With the ongoing digitization of slides, it is estimated that maintaining an indexed archive—often referred
to as anatlas—may require 50–150 petabytes of high-performance storage annually [10]. Such requirements
are highly inefficient and ultimately impede clinical deployment of large-scale digital pathology systems.
Current unsupervised methods extract a representative subset of patches from each WSI, reducing index-
ing storage to approximately 5–10% of the original size. Although this represents a substantial improvement,
the total storage demand—estimated at 5–15 petabytes—remains prohibitively large. Given that the cost
per petabyte varies across institutions, the resulting financial burden on hospitals can be significant. Any
further reduction in indexing storage would therefore be highly valuable, enabling broader adoption of digital
pathology and more scalable retrieval solutions.
Beyond storage, the computational burden of comparing embeddings is another limitation of automated
search systems. Most approaches compute distances between embeddings from the query WSI and those
stored in the archive, typically requiring exhaustive pairwise comparisons. If an incoming WSI producesm
embeddings and another WSI containsnembeddings, retrieval involvesm×npairwise distance calculations.
Consequently, every reduction in the number of stored embeddings directly decreases the number of required
comparisons by a factor ofn, accelerating retrieval. Even when “aggregation” methods [1, 2] are used to
produce slide-level representations, redundancy reduction remains essential. Aggregation still depends on the
underlying patch embeddings, and reducing their number lowers storage costs and accelerates computation
without compromising downstream performance.
The primary contribution of this paper is the introduction of a similarity-matching procedure that removes
redundancy by identifying highly similar patch-embedding pairs between WSIs of different classes. These
pairs are stored in a redundancy repository and subsequently used to prune redundant embeddings from
incoming WSIs before indexing or retrieval. This strategy not only reduces storage requirements but also
significantly accelerates the comparison process during search.
The remainder of the paper is organized as follows. Section 2 provides an overview of current search,
retrieval, and indexing systems. Section 3 describes the proposed similarity-matching approach in detail.
Section 4 presents experimental results on publicly available TCGA datasets.
2 Background Knowledge
One of the major challenges in digital histopathology is the lack of efficient methods for extracting information
from, and managing, the extremely large size of WSIs. Based on extensive benchmarking across public and
private datasets using multiple performance metrics, the current baseline for WSI search and retrieval is the
Yottixel engine [6,12]. The following subsections review the foundational components of the WSI processing
pipeline—patch selection, retrieval, and indexing—used in many popular systems, including the Yottixel
framework.
Given an atlas of WSIs with associated labels, slides are first divided into patches, after which a patch
selection algorithm identifies a representative subset for each WSI. Embedding extraction and barcoding
methods are then applied to enable fast retrieval. Incoming WSIs follow the same workflow, and their
predicted labels are obtained by leveraging the stored embeddings and metadata of the atlas.
The method proposed in this paper, described in detail in Section 3, further reduces storage by shrinking
the representation set of patches per WSI and decreases search time by removing redundant embeddings
from incoming slides as well. Even when aggregation approaches are used to produce slide-level features,
such redundancy reduction remains beneficial, since aggregation still depends on the underlying patch rep-
2

Figure 1: This schematic illustrations shows that patch selection methods select only extract a small number
of sub-images from the gigapixel whole slide image of size 87,584 pixels by 72,452 pixels (Source: TCGA file
TCGA-D8-A1XK-01Z-00-DX1.41EB1BBC-F230-4E3F-8C4E-CE331CAF1935.svs).
resentations.
2.1 Tissue Segmentation
The first preprocessing step identifies tissue regions and removes non-informative areas such as pen marks
or white background. Ann×ngrid is applied to each WSI, and tissue coverage is computed per grid cell.
All cells containing less than 30% tissue are discarded. Each remaining cell is treated as a patch of the WSI.
2.2 Patch Selection Algorithms
The most common strategy for extracting information from WSIs is to divide the slide into patches and
select only a compact, representative subset. This step, known aspatch selection, is central to reducing
computational and storage load. Numerous algorithms for this purpose have been introduced in recent years
(Figure 1).
Yottixel[12] was previously considered the state-of-the-art patch selection algorithm. Its core idea is a
two-step clustering strategy. The first step applies k-means clustering using color histograms, and the second
step incorporates spatial information to refine sampling. With a fixed sampling rate within each cluster, the
patch closest to the cluster centroid is selected as the representative.
SPLICE[13] (Sequential Patching Lattice for Image Classification and Enquiry) also relies on color
histograms but performs sequential sampling. After each selected patch, SPLICE removes similar patches
from the sampling pool, thereby encouraging diversity and reducing redundancy in the selected subset.
2.3 Foundation Model Embedding
After selecting the representative patches from each WSI, we extract feature embeddings using a pre-trained
or fine-tuned model—most commonly a foundation model. This step substantially reduces storage require-
ments and facilitates downstream processing such as BOB encoding, retrieval, and indexing. It also simplifies
similarity computation, since comparisons are performed in a compact embedding space.
3

2.4 Hashing and Barcoding
Binary encoding is widely used to reduce storage demands and accelerate retrieval. The MinMax algo-
rithm introduced in [17] converts feature embeddings into binary codes by computing the sign of each
one-dimensional deviation. This enables fast similarity analysis using Hamming distance instead of directly
computing Euclidean distances between high-dimensional embeddings.
2.5 Top-kMajority Voting Prediction
For label prediction of incoming WSIs, we retrieve information from the atlas embeddings using the same
processing workflow. After computing BOB barcodes for the incoming slide, we calculate its distance to each
atlas WSI, defined as the median of the minimum Hamming distances across all barcode pairs. We then
identify thekclosest atlas WSIs and obtain their labels. The final predicted label for the incoming WSI is
determined through majority voting among theseknearest neighbors.
3 Proposed Method
The main idea behind ARReST is based on the observation that when comparing representative patch
embeddings from two WSIs belonging to different classes, patches originating fromnormaltissue tend to
exhibit higher pairwise similarity than patches representing abnormal or cancerous regions. This motivates
a new redundancy-removal strategy: although labeled normal tissue is typically unavailable in WSI datasets,
the availability of cancer-type labels enables the use of cross-class pairwise comparisons to identify and isolate
highly similar (and therefore likely normal) patches. These patches can then be removed as redundant from
each WSI.
The first step of ARReST is to perform similarity matching and construct a redundancy repository.
Given labeled WSIs (W i, Li), whereW idenotes the slide andL iits class label, we apply a patch selection
algorithm to extract a representative set of patches,
Pi={P ij}j=1,2,...,n i,
whereP ijis thej-th selected patch from slideW i, andn iis the number of selected patches.
Figure 2 illustrates the similarity-matching process (see Algorithm 1). After patch selection (e.g., Yot-
tixel, SPLICE, SDM), each patch is passed through a foundation model to obtain its embedding. We then
repeatedly sample embedded patches from two WSIs of different classes and compute their pairwise similar-
ity. Patch pairs with similarity scores below theq-quantile threshold of the global similarity score distribution
are removed from both patch sets and stored separately in the redundancy repository.
For clarity, Figure 2 omits the embedding step and directly visualizes the removal of highly similar
cross-class patches—typically corresponding to normal tissue—highlighting how redundancy is identified
and eliminated within the ARReST framework.
Once the redundancy repositoryRis constructed and the atlas patch embeddings are updated to ( ˆPi, Li),
a common downstream step is to encode these embeddings as BOB barcodes ˆBi. This conversion further
reduces storage requirements and enables far more efficient distance computation, as Hamming distance
between binary codes is significantly faster and simpler to compute than distances in continuous embedding
space.
Algorithm 2 outlines the downstream processing and testing procedure. For each incoming WSIW i, we
first apply a patch selection algorithm to obtain a representative set of patches and then use a foundation
model to map these patches to embeddingsP i∈ P. For each patch embeddingP ik∈P i, we compute
its similarity score with every embeddingR jin the redundancy repositoryR. If the similarity score falls
below theq-quantile threshold of the similarity-score distributionS q, the corresponding patch embedding
is removed from the representation set. This yields the updated embedding set ˜Pi, which is subsequently
converted into its BOB barcode representation ˜Bi. We then use the atlas BOB barcodes, along with their
associated labels and patient identifiers ( ˆBi, Li, Gi), to predict the label of the incoming WSI via top-k
majority voting.
Figure 3 illustrates the complete pipeline of the proposed search engine and indexing framework.
4

Figure 2: Workflow to establish an atlas of antithetically redundant patches. Two WSIs from different
classes (e.g., two different cancer subtypes, hence the termantithetic) are first selected. An unsupervised
patch selection method is then applied to each slide to extract representative patches. These patches are
compared pairwise to identify highly similar regions shared across the two classes. The resulting cross-class
common patches—typically normal or non-discriminative tissue—are subsequently added to the Redundancy
Atlas.
5

Algorithm 1Create redundancy repository
Require:Patches embeddingsP i={P ij}j=i,2,...n i, labelsL i,i=1,2,...,n. Similarity Score function f, pre-
specified quantile q.
1:InitializeNormal repositoryR←[], Similar Score repositoryS←[]
2:fori←1 to ndo
3:forj←1 to ndo
4:ifi̸=jthen
5:forP ik∈Pido
6:forP jm∈Pjdo
7:Appendf(P ik, Pjm) to S
8:end for
9:end for
10:end if
11:end for
12:end for
13:get q quantile of S:S q
14:fori←1 to ndo
15:forj←1 to ndo
16:ifi̸=jthen
17:forP ik∈Pido
18:forP jm∈Pjdo
19:iff(P ik, Pjm)< S qthen
20:AppendP ik, Pjmto R
21:RemoveP ik, PjmfromP i,P j
22:end if
23:end for
24:end for
25:end if
26:end for
27:end for
28:returnRedundancy repository R, updated patch embeddings ˆPi,i=1,2,...,n.
Algorithm 2Remove redundancy and testing with topkmajority voting
Require:Incoming Patches embeddingsP= (P i) withP i={P ij}j=1,2,...,n i. Similarity Score function f, q
quantile of Similar Score repository S:S q, Redundancy repositoryR, patches embeddings repository (as
BOB barcodes) with labels and patients id ( ˆBi, Li, Gi),i=1,2,. . . ,n
1:Initialize
2:for P i∈ Pdo
3:forP ik∈Pido
4:forPatch embeddingR j∈Rdo
5:iff(P ik, Rj)< S qthen
6:RemoveP ikfromP i
7:Break
8:end if
9:end for
10:end for
11:Get the updated embeddings ˜Pi
12:Get BOB barcode ˜Bifrom updated embeddings ˜Pi
13:Do majority vote of the labelsL (1), . . . , L (k)from k closest BOB barcodes from repository
ˆB(1), . . . , ˆB(k)which do not share the same patient id with ˜Biand predict label ˆLi
14:end for
15:returnUpdated patch embeddings ˜Pi, i=1,2,. . . ,n and predicted labels ˆLi
6

Figure 3: The structure of the search engine, when using an Atlas enhanced with a Redundancy Atlas,
eliminates irrelevant patches after matching patch embeddings.
4 Results
In this section, we compare the search performance of the extracted embeddings before and after redundancy
removal via similarity matching. We use Yottixel as the baseline patch selection algorithm, with WSIs
processed at 20×magnification and a patch size of 224×224. For deep feature extraction, we employ the
UNI model, a publicly available pretrained vision encoder for histopathology (100M images, 100K WSIs)
developed on diverse neoplastic, infectious, inflammatory, and normal tissues [22].
We evaluate performance using 5-fold cross-validation under both conditions. In each fold, evaluation is
performed in a “leave-patient-out” manner to ensure that WSIs from the same patient never appear in both
the test set and the atlas. This requires all slides from a given patient to be assigned to a single fold, while
preserving the label distribution across folds. To satisfy these requirements, we implement Stratified Group
K-Fold cross-validation. For each fold, the held-out set serves as the test dataset and the remaining folds
form the atlas.
Each test WSI is patched, embedded, and compared against all embeddings in the redundancy repository
to compute similarity scores. Embeddings with similarity scores below the quantile threshold are removed.
The remaining embeddings are converted into BOB barcodes, and distances are computed between these
barcodes and those stored in the atlas. We then retrieve the Top-1, Top-3, and Top-5 most similar atlas
WSIs and determine the predicted label (i.e., cancer subtype for each organ) using majority voting.
Our experiments use The Cancer Genome Atlas (TCGA) dataset [21], consisting of 11,679 WSIs. We
evaluate performance separately for each organ. Table 1 reports the accuracy and macro-averaged F1 score
before and after similarity matching for the Top-1, MV@3, and MV@5 retrieval scenarios.
Analysis of Results –When measuring “accuracy,” results improved in 26 experiments after ARReST
was applied, remained unchanged in 20 cases, and decreased in 17 experiments. However, the averages, stan-
dard deviations, and highp-values indicate that the before/after accuracy values are essentially equivalent.
Because accuracy is typically reported forbalanceddatasets, this suggests that the observed storage savings
of 14%±13% can be reliably achieved without compromising retrieval performance in balanced settings.
7

Table 1: Comparison of performance before and after redundancy removal. Higher accuracy values and F1
scores are highlighted inbold. The storage savingSindicates the percentage reduction in required storage
after redundancy removal. MV@3: majority vote among the top-3 retrieved WSIs; MV@5: majority vote
among the top-5 retrieved WSIs.);S: storage saving.
Accuracy Macro Average F1 Score
Top-1 MV@3 MV@5 Top-1 MV@3 MV@5
Datasets Before After Before After Before After Before After Before After Before After S
Cervix 52 53 60 60 61 62 28 27 30 28 25 28 24%
Lung 63 64 66 66 67 68 23 22 24 21 22 20 17%
Testicles 51 56 56 59 60 60 35 39 39 41 44 42 33%
Colon 86 86 86 87 86 86 54 55 57 59 54 55 6%
Esophagus 91 92 90 92 91 90 62 62 62 62 62 62 8%
Eye 51 49 51 49 51 48 42 41 45 42 36 32 5%
Pancreas 82 81 87 88 88 88 41 43 47 49 45 48 12%
Bladder 85 83 87 87 87 88 68 65 69 69 70 70 3%
Rectum 84 84 85 86 87 88 62 64 57 59 53 55 15%
Thymus 34 34 38 36 40 36 27 28 35 33 35 32 12%
Stomach 44 44 49 49 50 48 31 32 34 34 36 34 7%
Skin 85 88 90 91 91 91 32 20 31 19 23 19 12%
Chest/abdom. 59 57 67 67 70 69 34 31 36 34 36 34 4%
Thyroid gland 71 72 78 76 79 77 48 48 49 43 48 43 4%
Head neck 82 84 85 85 84 85 45 37 39 25 29 25 16%
Soft Tissue 46 46 47 47 46 47 44 38 45 39 42 38 23%
Adrenal glands 79 81 79 82 80 81 41 34 39 33 34 29 60%
Uterus 75 74 76 75 74 75 53 50 52 52 47 46 9%
Breast 73 74 79 80 81 80 27 25 31 29 31 29 7%
Kidney 92 91 93 93 92 92 79 76 78 76 72 69 10%
Brain 63 63 65 65 65 66 41 42 44 44 44 45 13%
µ±σ 69±17 69±17 72±17 72±17 73±16 73±17 44±15 42±15 45±14 42±16 42±14 41±15
In contrast, when measuring “F1 Score,” performance improved in only 16 experiments, remained un-
changed in 9 cases, and decreased in 38 experiments. The averages, standard deviations, andp-values below
0.05 indicate that the before/after F1 values are statistically different. Since F1 score is generally used for
imbalanceddatasets, this suggests that the same level of storage saving may come at the cost of reduced
retrieval accuracy in imbalanced scenarios.
The organ-level analysis shows that the effectiveness of ARReST depends strongly on the underlying
anatomical and histopathological characteristics of each tissue type. ARReST performs most reliably in
organs withhomogeneous morphology and clear inter-class separability, such as the cervix, rectum,
and brain. In these cases, both accuracy and F1 score remain high despite a 14%±13% reduction in index
size, indicating that the pruned patches were largely redundant.
Several organs—including testis, colon, pancreas, skin, and kidney—exhibitedno loss in accuracy,
demonstrating that ARReST can safely reduce storage even when morphology is moderately variable but
consistent within dominant subtypes. Likewise, esophagus and bladder maintained their F1 scores, showing
resilience to pruning even under imbalanced subtype distributions.
In contrast, organs withhigh intraclass heterogeneity—such as lung, head and neck, soft tissue sar-
comas, adrenal gland, and uterus—were more sensitive to pruning. These tissues contain diverse or rare
subtypes that depend on a small number of discriminative patches; removing such patches can dispropor-
tionately impact retrieval in imbalanced datasets, leading to decreased F1 scores. Nevertheless, accuracy
often remained stable in these organs, suggesting that pruning primarily affects minority classes.
Overall, ARReST is most effective in organs withstable and distinctive histologic signaturesand
remains viable in moderately heterogeneous tissues where performance is unaffected. For highly imbalanced
or morphologically diverse organs, adaptive or subtype-aware pruning strategies may be needed to preserve
minority-class retrieval performance while still achieving meaningful storage reduction (Table 2).
5 Discussion and Conclusion
In this study, we proposed a novel approach—Antithetical Redundancy Reduction (ARReST)—designed
to improve the efficiency of indexing and retrieval in whole slide image (WSI) search engines for digital
pathology. By exploiting redundancy among dissimilar tissue classes, ARReST substantially reduces storage
requirements while maintaining, and in some cases improving, retrieval accuracy. Experiments on TCGA
datasets demonstrate that ARReST effectively minimizes the number of stored patches without compromising
overall search performance.
A key insight from this work is that a large portion of normal patches are inherently redundant and appear
8

Table 2: Organ Groups and Their Anatomical/Histopathological Characteristics Relevant to ARReST Per-
formance when analyzingMV@5
Storage Saving Organs Key Anatomi-
cal/Histopathological Features
Higher Accuracy & F1 Cervix, Rectum, Brain Homogeneous histology; strong
inter-class separability; pruning
removes mostly redundant patches.
Accuracy stays the
sameTestis, Colon, Pancreas, Skin,
KidneyModerate intraclass variability;
dominant morphologies are con-
sistent; clear subtype boundaries
support pruning without accuracy
loss.
Higher Accuracy only Lung, Bladder, Head & Neck,
Soft Tissue, Adrenal Gland,
UterusHigh intraclass heterogeneity; many
rare subtypes; pruning can remove
discriminative minority patches.
F1 stays the same Esophagus, Bladder Subtype imbalance moderate rather
than severe; preserved minority-
class features allow stable F1 after
pruning.
Higher F1 only Colon, Pancreas Dominant subtype morphology sta-
ble; ARReST preserves key discrim-
inative patterns despite imbalance.
across WSIs regardless of cancer type. Eliminating these redundant patches early in the pipeline significantly
reduces storage demands while preserving the discriminative content required for accurate retrieval.
Beyond storage reduction, ARReST also accelerates the retrieval process. With fewer patches to compare,
search time decreases, making the system more practical for large-scale deployments. This improvement is
particularly valuable in clinical contexts, where rapid and accurate WSI retrieval can support diagnostic
workflows and decision-making.
Nonetheless, the method has limitations. The effectiveness of redundancy removal depends on the chosen
similarity threshold, which may not be optimal for all organs or datasets. In tissues with high morphological
variability, overly aggressive pruning could risk discarding informative patches. Future work could explore
adaptive thresholding strategies or deep learning–based similarity estimation to refine redundancy detection.
In summary, ARReST offers a practical and effective strategy for improving WSI search engines by reduc-
ing redundant storage and accelerating retrieval. By addressing both storage constraints and computational
efficiency, ARReST provides a scalable solution for managing increasingly large WSI archives. Future exten-
sions may involve more adaptive similarity measures and applying the method to medical imaging domains
beyond histopathology.
Ethics and COI
This study exclusively utilized publicly available data from The Cancer Genome Atlas (TCGA), which is
fully de-identified and compliant with all relevant ethical and regulatory standards. No human subjects were
recruited, and no institutional review board (IRB) approval was required. The authors declare that they
have no competing interests related to this work.
9

References
[1] S. Chen et al., “Benchmarking embedding aggregation methods in computational pathology: A clinical
data perspective,”arXiv preprint arXiv:2407.07841, 2024.
[2] S. Hemati, K. R. Kalari, and H. R. Tizhoosh, “A Short Survey on Set-Based Aggregation Techniques
for Single-Vector WSI Representation in Digital Pathology,”arXiv preprint arXiv:2409.04615, 2024.
[3] K. K. Suresh, S. Sundaresan, R. Nishanth, and T. A. Kumar, “Optimization and Deep Learning–Based
Content Retrieval, Indexing, and Metric Learning Approach for Medical Images,” inComputational
Analysis and Deep Learning for Medical Care, Wiley, 2021, pp. 79–106.
[4] K. A. Castro, J. A. Siwady, E. Castillo, A. Alonzo, M. Cardona, and M. E. Perdomo, “Artificial
Intelligence for All: Challenges and Harnessing Opportunities in AI Democratization,” inProc. IEEE
ICMLANT, 2024, pp. 143–148.
[5] T. J. Flotte et al., “Democratizing Artificial Intelligence in Anatomic Pathology,”Archives of Pathology
& Laboratory Medicine, vol. 149, no. 1, pp. 55–59, 2025.
[6] I. Lahr et al., “Analysis and validation of image search engines in histopathology,”IEEE Reviews in
Biomedical Engineering, 2024.
[7] O. Ardon et al., “Digital pathology operations at a tertiary cancer center: Infrastructure requirements
and operational cost,”Journal of Pathology Informatics, vol. 14, p. 100318, 2023.
[8] M. G. Hanna et al., “Integrating digital pathology into clinical practice,”Modern Pathology, vol. 35,
no. 2, pp. 152–164, 2022.
[9] H. R. Tizhoosh and L. Pantanowitz, “On image search in histopathology,”Journal of Pathology Infor-
matics, p. 100375, 2024.
[10] O. Ardon et al., “Digital pathology operations at a tertiary cancer center: Infrastructure requirements
and operational cost,”Journal of Pathology Informatics, vol. 14, p. 100318, 2023.
[11] K. Ashman et al., “Whole slide image data utilization informed by digital diagnosis patterns,”Journal
of Pathology Informatics, vol. 13, p. 100113, 2022.
[12] S. Kalra, C. Choi, S. K. Shah, L. X. Pantanowitz, and H. R. Tizhoosh, “Yottixel – An Image Search
Engine for Large Archives of Histopathology Whole Slide Images,”arXiv preprint arXiv:1911.08748,
2019.
[13] A. Alsaafin et al., “SPLICE – Streamlining Digital Pathology Image Processing,”American Journal of
Pathology, vol. 194, 2024.
[14] A. Shafique, S. Alfasly, A. Alsaafin, P. Nejat, J. A. Khan, and H. R. Tizhoosh, “Selection of Distinct
Morphologies to Divide & Conquer Gigapixel Pathology Images,”arXiv preprint arXiv:2311.09902,
2023.
[15] C. Chen, M. Lu, D. Williamson, T. Chen, A. Schaumberg, and F. Mahmood, “Fast and scalable search
of whole-slide images via self-supervised deep learning,”Nature Biomedical Engineering, vol. 6, pp.
1–15, 2022.
[16] X. Peng, L. Wang, X. Wang, and Y. Qiao, “Bag of Visual Words and Fusion Methods for Action
Recognition: Comprehensive Study and Good Practice,”arXiv preprint arXiv:1405.4506, 2014.
[17] H. R. Tizhoosh, S. Zhu, H. Lo, V. Chaudhari, and T. Mehdi, “MinMax Radon Barcodes for Medical
Image Retrieval,”arXiv preprint arXiv:1610.00318, 2016.
[18] O. Ardon et al., “Understanding the financial aspects of digital pathology: A dynamic customizable
return on investment calculator for informed decision-making,”Journal of Pathology Informatics, vol.
15, p. 100376, 2024.
10

[19] T. Chong et al., “The California Telepathology Service: UCLA’s Experience in Deploying a Regional
Digital Pathology Subspecialty Consultation Network,”Journal of Pathology Informatics, vol. 10, no.
1, p. 31, 2019.
[20] M. G. Hanna et al., “Implementation of Digital Pathology Offers Clinical and Operational Increase
in Efficiency and Cost Savings,”Archives of Pathology & Laboratory Medicine, vol. 143, no. 12, pp.
1545–1555, 2019.
[21] T. Bloom et al., “Comprehensive genomic characterization defines human glioblastoma genes and core
pathways,” inCancer Genome Atlas Research Network, 2008.
[22] R. J. Chen et al., “Towards a General-Purpose Foundation Model for Computational Pathology,”Nature
Medicine, 2024.
11