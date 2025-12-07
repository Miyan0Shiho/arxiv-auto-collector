# Spatially-Grounded Document Retrieval via Patch-to-Region Relevance Propagation

**Authors**: Agathoklis Georgiou

**Published**: 2025-12-02 11:29:54

**PDF URL**: [https://arxiv.org/pdf/2512.02660v1](https://arxiv.org/pdf/2512.02660v1)

## Abstract
Vision-language models (VLMs) like ColPali achieve state-of-the-art document retrieval by embedding pages as images and computing fine-grained similarity between query tokens and visual patches. However, they return entire pages rather than specific regions, limiting utility for retrieval-augmented generation (RAG) where precise context is paramount. Conversely, OCR-based systems extract structured text with bounding box coordinates but lack semantic grounding for relevance assessment. We propose a hybrid architecture that unifies these paradigms: using ColPali's patch-level similarity scores as spatial relevance filters over OCR-extracted regions. We formalize the coordinate mapping between vision transformer patch grids and OCR bounding boxes, introduce intersection metrics for relevance propagation, and establish theoretical bounds on retrieval precision. Our approach operates at inference time without additional training. We release Snappy, an open-source implementation demonstrating practical applicability, with empirical evaluation ongoing.

## Full Text


<!-- PDF content starts -->

Spatially-Grounded Document Retrieval via
Patch-to-Region Relevance Propagation
Agathoklis Georgiou
Independent Researcher
athrael.soju@gmail.com
Abstract
Vision-language models (VLMs) like ColPali achieve state-of-the-art document retrieval by
embedding pages as images and computing fine-grained similarity between query tokens and
visual patches. However, they return entire pages rather than specific regions, limiting util-
ity for retrieval-augmented generation (RAG) where precise context is paramount. Conversely,
OCR-based systems extract structured text with bounding box coordinates but lack seman-
tic grounding for relevance assessment. We propose a hybrid architecture that unifies these
paradigms: using ColPali’s patch-level similarity scores as spatial relevance filters over OCR-
extracted regions. We formalize the coordinate mapping between vision transformer patch grids
and OCR bounding boxes, introduce intersection metrics for relevance propagation, and estab-
lish theoretical bounds on retrieval precision. Our approach operates at inference time without
additional training. We release Snappy, an open-source implementation demonstrating practical
applicability, with empirical evaluation ongoing.
1 Introduction
Retrieval-augmented generation (RAG) has emerged as the dominant paradigm for grounding large
language models in external knowledge, enabling factual responses without costly retraining. The
effectiveness of RAG systems hinges on a fundamental requirement: retrievingprecisely relevant
context while minimizing noise. For text corpora, this challenge is well-studied. Dense retrievers
identifysemanticallysimilarpassages, andchunkingstrategiescontrolcontextgranularity. However,
document collections present a fundamentally harder problem.
Documents are not sequences of tokens butspatially-organized visual artifacts. A single page
may contain heterogeneous elements, including tables, figures, equations, headers, and footnotes,
each carrying distinct semantic content at different spatial locations. When a user queries “What
was the Q3 revenue?”, the answer likely resides in a specific table cell, not spread across the entire
page. Yet current retrieval systems operate at the wrong granularity.
Vision-language models (VLMs) such as ColPali (Faysse et al., 2025) have achieved state-of-
the-art performance on document retrieval benchmarks by embedding document pages directly as
images. ColPali produces 1,024 patch embeddings (a32×32grid) per page, each projected to
128 dimensions. The model computes relevance through late interaction, specifically a MaxSim
operation that sums the maximum similarity between each query token and all document patches.
This approach elegantly sidesteps OCR errors and preserves layout semantics. However, ColPali
and its variants returnentire pagesas retrieval units. For RAG applications, this is problematic:
feeding a full page into a language model’s context window introduces irrelevant content, increases
latency, inflates costs, and, critically, dilutes the signal that the model must attend to. The retrieval
system knowswhich pagecontains the answer but notwhere on the page.
1arXiv:2512.02660v1  [cs.CV]  2 Dec 2025

Conversely, OCR-based pipelines extract text with precise bounding box coordinates, enabling
structured representations of document content. Tables become rows and columns; figures receive
captions; headers define hierarchy. This structural fidelity is invaluable for downstream processing.
Yet OCR systems lacksemantic grounding. They cannot assess which extracted regions are relevant
to a given query. A page with twenty OCR regions offers no ranking mechanism; all regions are
treated as equally plausible candidates.
We observe that these paradigms are complementary. ColPali’s patch-level similarity scores
encodewhereon a page the model attends when processing a query. This information is computed
but discarded when returning page-level results. OCR systems knowwhatcontent exists andwhere
it is located, but notwhyit matters. By unifying these signals through spatial coordinate mapping,
we achieve region-level retrieval: returning only the document regions that are both structurally
coherent (via OCR) and semantically relevant (via VLM attention).
Crucially, our approach operates atinference timewithout additional training. Unlike Region-
RAG (Li et al., 2025), which uses a hybrid training approach combining bounding box annotations
with weakly-supervised signals from unlabeled data, our method leverages ColPali’s emergent patch
attention as a post-hoc spatial filter. This provides flexibility: the same approach works with any
OCR system providing bounding boxes and any ColPali-family model.
1.1 Contributions
This paper presents a hybrid architecture for spatially-grounded document retrieval:
1.CoordinateMappingFormalism.Weformalizethemathematicalcorrespondencebetween
vision transformer patch grids and OCR bounding boxes, enabling spatial alignment between
heterogeneous representations (Section 3.2).
2.Relevance Propagation via Interpretability Maps.We repurpose ColPali’s late interac-
tion mechanism to generate per-query-token similarity heatmaps, then propagate these scores
to OCR regions through IoU-weighted patch-region intersection (Section 3.3).
3.Two-Stage Retrieval Architecture.We introduce a mean-pooling strategy that com-
presses patch embeddings along spatial axes, enabling efficient candidate retrieval before full-
resolution reranking (Section 3.4).
4.Theoretical Analysis.We establish bounds on localization precision as a function of patch
resolution, derive expected context reduction and precision amplification factors, and analyze
computational complexity tradeoffs (Section 4).
5.Open Implementation.We release Snappy, a complete system implementing this archi-
tecture with ColPali, DeepSeek OCR, and Qdrant vector search, demonstrating practical
applicability (Section 5).
We hypothesize that region-level retrieval will improve downstream RAG answer quality while
reducingcontextlength, validatingthepremisethatretrievalgranularitydirectlyimpactsgeneration
fidelity. Empirical evaluation is ongoing and will be reported in a subsequent revision.
2

2 Background and Related Work
2.1 Vision-Language Document Retrieval
ColPali and Late Interaction.ColPali (Faysse et al., 2025) represents the state-of-the-art in
visual document retrieval. Built on a SigLIP-So400m vision encoder, it produces 1,024 patch em-
beddings per page (32×32grid over448×448input resolution), each projected to 128 dimensions
via a language model projection layer. Unlike single-vector approaches that pool visual features
into a single embedding, ColPali preserves patch-level granularity and computes relevance through
MaxSim, summing the maximum similarity between each query token and all document patches.
This late interaction mechanism, inherited from ColBERT (Khattab and Zaharia, 2020), enables
fine-grained matching while remaining computationally tractable for retrieval at scale.
The ViDoRe benchmark (Faysse et al., 2025) evaluates visual document retrieval across diverse
domains, measuring NDCG@5 for page-level retrieval. ColPali achieves strong performance, but the
benchmark, like the model, operates at page granularity, leaving region-level retrieval unexplored.
ColModernVBERT.Recent work introduces ColModernVBERT (Teiletche et al., 2025), a
250M-parameter late-interaction retriever achieving within 0.6 NDCG@5 of ColPali at over 10×
smaller size. The architecture maintains patch-level embeddings, making our approach directly
applicable to this more efficient variant.
2.2 Layout-Aware Document Understanding
The LayoutLM family (Xu et al., 2020, 2021; Huang et al., 2022) pioneered joint modeling of
text, layout, and visual features for document understanding. LayoutLMv3 introduced Word-
Patch Alignment (WPA) pre-training, which predicts whether image patches corresponding to text
words are masked. While conceptually related to our patch-OCR alignment, WPA operates atpre-
training timeto improve representations, whereas our approach uses patch similarities atinference
timefor retrieval filtering. Critically, LayoutLM models are designed for documentunderstand-
ingtasks (NER, classification) rather than retrieval—they lack late interaction mechanisms and
query-conditioned relevance scoring.
OCR-free approaches including Donut (Kim et al., 2022) and Pix2Struct (Lee et al., 2023)
perform document understanding directly from pixels. UDoP (Tang et al., 2023) unifies vision,
text, and layout modalities. These models excel at understanding but do not address the retrieval
problem we target.
2.3 Region-Level Document Retrieval
RegionRAG.The closest existing work is RegionRAG (Li et al., 2025), which shifts retrieval from
document-level to semantic region-level granularity. However, RegionRAG uses a hybrid approach
combining bounding box annotations with weakly-supervised signals from unlabeled data and a
region-level contrastive loss. Our approach is fundamentally different: we achieve region-level re-
trieval atinference timeusing ColPali’s emergent patch attention, requiring no additional training
or annotation.
DocVLM.DocVLM (Shpigel Nacson et al., 2024) integrates OCR into VLMs by compressing
OCR features into learned queries. This represents theopposite directionof our approach. DocVLM
adds OCR to enhance VLM understanding, whereas we use VLM patches to filter and score OCR
output for retrieval.
Table 1 summarizes the positioning of our approach relative to prior work.
3

Table 1: Comparison with related approaches. Our approach is unique in achieving region-level
retrieval at inference time without additional training, by propagating VLM patch similarities to
OCR bounding boxes.
Method Granularity OCR Required Training
ColPali Page-level No Pre-trained
LayoutLM Understanding Yes Pre-trained
RegionRAG Region-level Yes Hybrid supervision
DocVLM Understanding Yes Fine-tuning
Ours Region-level Yes Inference-time only
3 Method
3.1 Problem Formulation
Given a queryqand a document corpusDwhere each documentd∈ Dconsists of one or more
pages, conventional visual document retrieval returns a ranked list of pages. We reformulate the
problem asregion-level retrieval: return a ranked list of (page, region) pairs where each region
corresponds to a semantically coherent text block (paragraph, table, figure caption, etc.) extracted
via OCR.
LetP={p 1, . . . , p n}denote the set of pages in the corpus, where each pagep ihas an associated
set of OCR regionsR(p i) ={r 1, . . . , r m}. Each regionr jis characterized by its bounding box
B(r j) = (x 1, y1, x2, y2)in pixel coordinates and its text contentT(r j). Our goal is to compute a
relevancescorerel(q, r j)foreachregionthatcapturesbothsemanticrelevanceandspatialgrounding.
3.2 Coordinate Mapping: Patches to Bounding Boxes
ColPali encodes each page as a grid ofG×Gpatch embeddings (G= 32for ColPali-v1.3) over an
input image of resolutionI×I(I= 448). Each patch corresponds to ans×spixel region where
s=I/G= 14pixels. Patches are indexed in raster scan order (left-to-right, top-to-bottom).
Definition 1(Patch Coordinate Mapping).For patch indexk∈ {0, . . . , G2−1}, the corresponding
bounding box in pixel coordinates is:
patch_bbox(k) = (col·s,row·s,(col+ 1)·s,(row+ 1)·s)(1)
where row=⌊k/G⌋and col=kmodG.
When the original document page has resolution(W, H)different from the model’s input reso-
lutionI×I, OCR bounding boxes must be scaled to the model’s coordinate space:
B′(r) =
x1·I
W, y1·I
H, x2·I
W, y2·I
H
(2)
3.3 Relevance Propagation via Patch Similarity
Given a queryqtokenized intontokens with embeddings{q 1, . . . , q n}and a page with patch
embeddings{d 1, . . . , d m}(m=G2= 1,024), we compute the similarity matrix:
S∈Rn×mwhereS ij=sim(q i, dj)(3)
4

where sim(·,·)is cosine similarity. Standard ColPali aggregates this into a page score via MaxSim:
Score page(q, p) =X
imax
jSij (4)
We instead extract the spatial distribution of relevance by computing a per-patch score:
score patch(j) = max
iSij (5)
This captures the maximum relevance of patchjto any query token, forming a spatial heatmap
over the page.
Definition 2(Region Relevance Score).For OCR regionrwith scaled bounding boxB′(r), we
propagate patch scores via IoU-weighted aggregation:
rel(q, r) =X
jIoU(B′(r),patch_bbox(j))·score patch(j)(6)
where the sum is over all patchesjwith non-zero intersection. This weights each patch’s contribu-
tion by its spatial overlap with the region, ensuring that patches fully contained within the region
contribute more than peripheral patches.
3.4 Two-Stage Retrieval Architecture
Computing full patch-level similarity for all pages in a large corpus is prohibitively expensive. We
introduce a two-stage architecture that balances efficiency and precision.
Stage 1: Candidate Retrieval.We compress patch embeddings via mean pooling along
spatial axes to obtain a single page-level embedding. This enables efficient approximate nearest
neighbor search (via Qdrant) to retrieve top-Kcandidate pages. Note that pooling discards spa-
tial information, which may impact recall for pages with small relevant regions (see Section 7 for
discussion of this limitation).
Stage 2: Region Reranking.For each candidate page, we compute full patch-level similarity
and propagate scores to OCR regions as described in Section 3.3. Regions are ranked by their
relevance scores, and top-kregions are returned.
3.5 Aggregation Strategies
We consider alternative aggregation strategies for propagating patch scores to regions:
Max Aggregation:
relmax(q, r) = max
j∈covered(r)score patch(j)(7)
Mean Aggregation:
relmean(q, r) =1
|covered(r)|X
j∈covered(r)score patch(j)(8)
IoU-Weighted (Default):Uses the region relevance score from Definition 2:
relIoU(q, r) =X
jIoU(B′(r),patch_bbox(j))·score patch(j)(9)
The choice of aggregation strategy may affect retrieval quality depending on region size and
content density. We plan to empirically compare these strategies in our evaluation.
5

4 Theoretical Analysis
4.1 Precision Bounds
The spatial precision of our approach is fundamentally bounded by patch resolution. We formalize
this tradeoff.
Theorem 1(Localization Precision Bound).For an OCR region with bounding box of widthwand
heighth(in model coordinates), and patch sizes, the maximum achievable localization precision is:
precision≤w·h
(w+s)·(h+s)(10)
Proof.Consider a region with bounding boxBof dimensionsw×h. The set of patches intersecting
Bdepends onB’s alignment with the patch grid. In the worst case,Bis positioned such that it
intersects partial patches on all four edges. Let the region’s top-left corner fall at position(x, y)
within a patch. The region then spans from patch column⌊x/s⌋to⌊(x+w)/s⌋and from patch row
⌊y/s⌋to⌊(y+h)/s⌋. The number of intersecting patches is at most⌈w/s+ 1⌉·⌈h/s+ 1⌉. The total
area covered by these patches is at most(w+s)·(h+s), achieved when the region is maximally
misaligned with patch boundaries. Since the region’s area isw·h, the precision (ratio of relevant
area to retrieved area) is bounded by(w·h)/((w+s)·(h+s)).
Corollary 1.For ColPali withs= 14pixels at448×448resolution:
•A typical paragraph region (200×50pixels): precision≤73%
•A table cell (100×30pixels): precision≤60%
•A small label (50×20pixels): precision≤46%
This analysis reveals that smaller regions suffer disproportionately from patch quantization.
Applications requiring fine-grained localization (e.g., form field extraction) may benefit from higher-
resolution patch grids or multi-scale approaches.
4.2 Computational Complexity
LetN= number of pages,M= average OCR regions per page,G2= patches per page,n= query
tokens,d= embedding dimension.
Page-level retrieval (baseline):O(N·n·G2·d)for full MaxSim over all pages.
Our two-stage approach:
•Stage 1:O(N·d)for ANN search with pooled embeddings
•Stage 2:O(K·n·G2·d+K·M·G2)for full similarity onKcandidates plus region scoring
ForK≪N, the two-stage approach provides substantial speedup. With typical values (N=
100,000pages,K= 100,G= 32,n= 20,d= 128,M= 15), Stage 1 reduces the search space by
1000×before the more expensive region-level computation.
4.3 Expected Performance Bounds
Beyond spatial precision, we analyze the expected improvements in retrieval quality and down-
stream efficiency. These bounds provide theoretical justification for our approach pending empirical
validation.
6

4.3.1 Context Reduction
LetA pdenote the total page area andA rthe area of the relevant region containing the answer. For
a page withMOCR regions of average area ¯A, we haveA p≈M· ¯A.
Theorem 2(Context Reduction Bound).Letkbe the number of top-scoring regions returned by
our hybrid approach. The expected context reduction factor relative to page-level retrieval is:
CRF=ApPk
i=1Ari≥M
k(11)
with equality when all regions have equal area.
Proof.Page-level retrieval returns context proportional toA p. Our approach returns context pro-
portional to the total area of the top-kregions,Pk
i=1Ari. Since each region has area at mostA p
and there areMregions total, selectingkregions yields area at mostk· ¯A=k·A p/M. The ratio
Ap/(k·A p/M) =M/kprovides the lower bound.
Corollary 2(Token Savings).For typical document parameters (M= 15regions per page,k= 3
returned regions), the expected context reduction factor is at least5×. This translates directly to
proportional reductions in:
•LLM inference cost (tokens processed)
•Response latency (context length)
•Attention dilution (irrelevant content in context window)
4.3.2 Precision Amplification
We model retrieval precision as the probability that a returned region contains relevant content.
Definition 3(Region Relevance).LetR j∈ {0,1}indicate whether regionjcontains content rele-
vant to queryq. For a typical factoid query, we assumeP
jRj∈ {1,2}(one or two relevant regions
per page).
Theorem 3(Precision Comparison).Letρ=P
jRj/Mdenote the fraction of relevant regions on
a page. Then:
(i) OCR-only (random selection ofkregions):
E[Precision OCR] =ρ(12)
(ii) ColPali page-level:
Precision ColPali =ρ(all regions returned implicitly)(13)
(iii) Hybrid with score-based ranking:If patch scoress jsatisfyE[s j|Rj= 1]>E[s j|Rj= 0]
(relevant regions score higher on average), then selecting top-kby score yields:
E[Precision Hybrid ]> ρ(14)
with the improvement depending on the score separation between relevant and irrelevant regions.
Proof.Parts (i) and (ii) follow directly from the definition ofρ. For (iii), score-based ranking is
equivalent to a classifier with ROC curve above the diagonal when relevant regions have higher
expected scores. By the Neyman-Pearson lemma, thresholding on any statistic positively correlated
with relevance improves precision over random selection.
7

4.3.3 Signal-to-Noise Ratio
We formalize the intuition that region-level retrieval improves the “signal” available to downstream
LLMs.
Definition 4(Retrieval SNR).For retrieved contextC, define:
SNR(C) =|C∩Relevant|
|C∩Irrelevant|(15)
where| · |denotes token count.
Theorem 4(SNR Improvement).Under the assumption that relevant and irrelevant regions have
similar average token counts:
(i)Page-level retrieval: SNR page=ρ/(1−ρ)
(ii)Hybrid top-kretrieval with precisionP k: SNR hybrid =P k/(1−P k)
The SNR improvement factor is:
SNR hybrid
SNR page=Pk(1−ρ)
ρ(1−P k)(16)
Proof.For page-level retrieval, the fraction of relevant tokens equalsρby definition. For hybrid
retrieval with precisionP k, the expected fraction of relevant tokens among returned regions isP k.
The SNR ratio follows from algebraic manipulation.
4.3.4 Summary of Expected Gains
Table 2 provides a unified view of the cost-quality tradeoff, showing that our hybrid approach
achieves the context efficiency of OCR-based selection while providing the relevance ranking that
OCR alone cannot offer.
Table 2: Combined efficiency-quality comparison. The hybrid approach uniquely achieves both low
context cost and high precision.
Method Context Cost Precision Best Use Case
ColPali (page-level) High (1.0×) Low (6.7%) Page identification
OCR + BM25 Low (0.2×) Low (6.7%) Keyword matching
OCR + Dense Low (0.2×) Low–Med Semantic text search
Hybrid (ours) Low (0.2×) High (25–60%) Precise RAG
5 Implementation: Snappy System
We implement our approach in Snappy, an open-source document retrieval system available at
https://github.com/athrael-soju/Snappy. The system demonstrates the practical applicability
of spatially-grounded hybrid retrieval.
8

5.1 Architecture Overview
Figure1illustratestheSnappysystemarchitecture, showingtheparalleldocumentindexingpipeline
and query processing pipeline that converge at region-level filtering.
Figure 1: Snappy system architecture. The document indexing pipeline processes uploads through
parallel OCR and ColPali embedding branches, storing regions in DuckDB and patch vectors in
Qdrant. The query processing pipeline retrieves top-K candidates, generates interpretability maps
from patch-level attention, and filters OCR regions by relevance to produce answers with spatial
context.
Snappy pairs a FastAPI backend with a ColPali embedding service, DeepSeek OCR for visual
grounding, DuckDB analytics for metadata storage, and a Next.js frontend. The system delivers
hybrid vision+text retrieval over PDFs, where each page is rasterized, embedded as multivectors,
and stored alongside images and OCR-extracted text in Qdrant.
The indexing pipeline processes documents as follows: (1) render each page as an image using
9

Poppler, (2) extract patch embeddings via the ColPali service, (3) optionally extract text regions
with bounding boxes via DeepSeek OCR with visual grounding enabled, (4) store patch embed-
dings as multivectors and OCR metadata in Qdrant with page-level pooled embeddings for efficient
retrieval, and (5) persist images to MinIO object storage for visualization.
At query time: (1) encode the query via ColPali’s text encoder, (2) retrieve top-Kcandidate
pages via ANN search on pooled embeddings in Qdrant, (3) compute full patch-level similarity for
candidates using the stored multivectors, (4) propagate scores to OCR regions via IoU-weighted
aggregation, and (5) return ranked regions with text content, bounding boxes, and optional visual
highlighting.
5.2 Component Integration
The ColPali embedding service runs as a dedicated container exposing patch-level embeddings over
HTTP, supporting both GPU and CPU deployment. Unlike standard ColPali deployments that
return only page-level scores, Snappy’s integration extracts the full(n×G2)similarity matrix
stored as multivectors in Qdrant for downstream region scoring.
DeepSeek OCR provides text extraction with visual grounding capabilities. When enabled, the
OCR service extracts markdown-formatted text alongside bounding box coordinates for each text
region. OCR results are stored in DuckDB for SQL-based querying.
Qdrant stores both page-level pooled embeddings (for Stage 1 candidate retrieval) and full patch
embeddings as multivectors (for Stage 2 region scoring).
6 Planned Evaluation
Empirical evaluation is ongoing using BBox-DocVQA (Yu et al., 2025) (3.6K documents with 32K
question-answerpairsannotatedwithevidenceboundingboxes)andViDoReforpage-levelbaselines.
We will measure retrieval quality (NDCG@k, Precision@k), spatial grounding accuracy (IoU@0.5,
IoU@0.7), and downstream RAG answer quality (ANLS). Baselines include ColPali page-level re-
trieval, BM25onOCRtext, anddenseretrievalonOCRchunks. Ablationswillcompareaggregation
strategies and threshold sensitivity. Results will be reported in a subsequent revision.
7 Discussion
7.1 Limitations
Patch Resolution Bound.As analyzed in Section 4, small regions suffer from limited localiza-
tion precision due to fixed patch granularity. For ColPali’s 14-pixel patches, regions smaller than
approximately35×35pixels achieve less than 50% precision. Future work could explore multi-scale
patch embeddings or dynamic resolution based on document density.
OCR Dependency.Region quality depends on OCR accuracy and segmentation. Poorly
segmented regions (merged paragraphs, split tables) degrade retrieval quality regardless of the
VLM’s spatial attention accuracy. Layout-aware OCR systems like DeepSeek OCR with visual
grounding mitigate this, but OCR errors propagate to retrieval.
Emergent Attention Limitations.ColPali’s patch attention is optimized for page-level re-
trieval through training, not explicitly for region-level localization. Attention may diffuse across
semantically related but spatially distant regions (e.g., a header and its corresponding paragraph).
We rely on OCR region boundaries to constrain this diffusion.
10

Two-Stage Information Loss.The mean-pooling in Stage 1 discards spatial information,
potentially missing pages where relevant content is concentrated in small regions. Relevant pages
could be filtered out before Stage 2 region scoring. Alternative pooling strategies (max pooling,
attention-weighted pooling) may mitigate this.
Theoretical Assumptions.The performance bounds in Section 4.3 assume that ColPali’s
patch scores correlate positively with region relevance. While this is plausible given ColPali’s
strong page-level retrieval performance, the degree of correlation at the region level remains to
be empirically validated.
7.2 Future Directions
Cross-Page Region Linking.Extending region-level retrieval to link semantically related regions
across pages could enable quantitative search capabilities. For example, tracking a financial metric
across quarterly reports by linking table cells containing that metric across documents.
Elimination of Image Storage.By storing only patch embeddings and OCR-extracted text
with bounding boxes, our approach potentially eliminates the need for raw image storage after
indexing. This could significantly reduce infrastructure costs for large document collections while
maintaining retrieval capability.
Region-Aware Fine-Tuning.While our approach operates without additional training, fine-
tuning ColPali with region-level supervision could improve spatial attention alignment. This would
sacrifice our training-free advantage but potentially yield higher localization accuracy.
Legal Document Retrieval and Citation Grounding.In legal contexts, retrieval precision
directly impacts downstream reliability. Recent empirical work demonstrates that even retrieval-
augmented legal AI tools hallucinate 17–33% of responses, with errors compounded by coarse re-
trieval granularity (Magesh et al., 2025). Region-level retrieval with bounding box coordinates
provides citation-bounded context—each retrieved region carries verifiable provenance, constrain-
ing generation to specific, locatable sources.
8 Conclusion
We presented a hybrid architecture for spatially-grounded document retrieval that unifies vision-
language late interaction with structured OCR extraction. By formalizing the coordinate mapping
between ColPali’s patch grid and OCR bounding boxes, we enable region-level retrieval without
additional training. Our approach operates entirely at inference time, providing flexibility across
OCR systems and VLM backends.
The theoretical analysis establishes precision bounds as a function of patch resolution, derives
expected context reduction factors of5×or more (Theorem 2), and predicts SNR improvements up
to14×under moderate assumptions about score-relevance correlation (Theorem 4). These bounds
provide practitioners with guidance on expected performance improvements pending empirical val-
idation.
The two-stage architecture balances computational efficiency with region-level granularity, mak-
ing the approach practical for large-scale document collections.
We release Snappy as an open-source implementation demonstrating practical applicability. Em-
pirical evaluation is ongoing; we hypothesize that region-level retrieval will improve downstream
RAG answer quality while reducing context length and associated costs. Results will be reported
in a subsequent revision of this manuscript.
11

Acknowledgments
The author thanks the ColPali team for their foundational work on vision-language document
retrieval and for maintaining an open research ecosystem that enabled this work. The Snappy
system implementation is available athttps://github.com/athrael-soju/Snappy.
References
Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre
Colombo. ColPali: Efficient Document Retrieval with Vision Language Models. InThe Thirteenth
International Conference on Learning Representations (ICLR), 2025. arXiv:2407.01449.
Omar Khattab and Matei Zaharia. ColBERT: Efficient and Effective Passage Search via Con-
textualized Late Interaction over BERT. InProceedings of the 43rd International ACM SIGIR
Conference on Research and Development in Information Retrieval, pages 39–48, 2020.
Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. LayoutLM: Pre-
training of Text and Layout for Document Image Understanding. InProceedings of the 26th
ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1192–
1200, 2020.
Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei, Guoxin Wang, Yijuan Lu, Dinei Florencio,
Cha Zhang, Wanxiang Che, Min Zhang, and Lidong Zhou. LayoutLMv2: Multi-modal Pre-
training for Visually-rich Document Understanding. InProceedings of the 59th Annual Meeting
of the Association for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing, pages 2579–2591, 2021.
Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. LayoutLMv3: Pre-training for
DocumentAIwithUnifiedTextandImageMasking. InProceedings of the 30th ACM International
Conference on Multimedia, pages 4083–4091, 2022.
Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon Nam, Jinyoung Park, Jinyeong Yim,
Wonseok Hwang, Sangdoo Yun, Dongyoon Han, and Seunghyun Park. OCR-free Document
Understanding Transformer. InProceedings of the European Conference on Computer Vision
(ECCV), pages 498–517, 2022.
Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu, Julian Eisenschlos, Urvashi Khan-
delwal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova. Pix2Struct: Screenshot Parsing
as Pretraining for Visual Language Understanding. InProceedings of the 40th International Con-
ference on Machine Learning, pages 18893–18912, 2023.
Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang Liu, Chenguang Zhu, Michael Zeng, Cha
Zhang, and Mohit Bansal. Unifying Vision, Text, and Layout for Universal Document Processing.
InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages
19254–19264, 2023.
Yinglu Li, Zhiying Lu, Zhihang Liu, Yiwei Sun, Chuanbin Liu, and Hongtao Xie. RegionRAG:
Region-level Retrieval-Augmented Generation for Visually-Rich Documents. arXiv:2510.27261,
2025.
12

Wenhan Yu, Wang Chen, Guanqiang Qi, Weikang Li, Yang Li, Lei Sha, Deguo Xia, and Jizhou
Huang. BBox-DocVQA:ALargeScaleBoundingBoxGroundedDatasetforEnhancingReasoning
in Document Visual Question Answering. arXiv:2511.15090, 2025.
Mor Shpigel Nacson, Aviad Aberdam, Roy Ganz, Elad Ben Avraham, Alona Golts, Yair Kittenplon,
Shai Mazor, and Ron Litman. DocVLM: Make Your VLM an Efficient Reader. arXiv:2412.08746,
2024.
Paul Teiletche, Quentin Macé, Max Conti, Antonio Loison, Gautier Viaud, Pierre Colombo, and
Manuel Faysse. ModernVBERT: Towards Smaller Visual Document Retrievers. arXiv:2510.01149,
2025.
Varun Magesh, Faiz Surani, Matthew Dahl, Mirac Suzgun, Christopher D. Manning, and Daniel E.
Ho. Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools.Journal
of Empirical Legal Studies, 22:216, 2025.
13