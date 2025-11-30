# Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval

**Authors**: Anup Roy, Rishabh Gyanendra Upadhyay, Animesh Rameshbhai Panara, Robin Mills

**Published**: 2025-11-26 07:18:06

**PDF URL**: [https://arxiv.org/pdf/2511.21121v1](https://arxiv.org/pdf/2511.21121v1)

## Abstract
Document centric RAG pipelines usually begin with OCR, followed by brittle heuristics for chunking, table parsing, and layout reconstruction. These text first workflows are costly to maintain, sensitive to small layout shifts, and often lose the spatial cues that contain the answer. Vision first retrieval has emerged as a strong alternative. By operating directly on page images, systems like ColPali and ColQwen preserve structure and reduce pipeline complexity while achieving strong benchmark performance. However, these late interaction models tie retrieval to a specific vision backbone and require storing hundreds of patch embeddings per page, creating high memory overhead and complicating large scale deployment.
  We introduce VisionRAG, a multimodal retrieval system that is OCR free and model agnostic. VisionRAG indexes documents directly as images, preserving layout, tables, and spatial cues, and builds semantic vectors without committing to a specific extraction. Our three pass pyramid indexing framework creates vectors using global page summaries, section headers, visual hotspots, and fact level cues. These summaries act as lightweight retrieval surrogates. At query time, VisionRAG retrieves the most relevant pages using the pyramid index, then forwards the raw page image encoded as base64 to a multimodal LLM for final question answering. During retrieval, reciprocal rank fusion integrates signals across the pyramid to produce robust ranking.
  VisionRAG stores only 17 to 27 vectors per page, matching the efficiency of patch based methods while staying flexible across multimodal encoders. On financial document benchmarks, it achieves 0.8051 accuracy at 10 on FinanceBench and 0.9629 recall at 100 on TAT DQA. These results show that OCR free, summary guided multimodal retrieval is a practical and scalable alternative to traditional text extraction pipelines.

## Full Text


<!-- PDF content starts -->

Beyond Patch Aggregation: 3-Pass Pyramid Indexing for
Vision-Enhanced Document Retrieval
Anup Roy
anup@inceptionai.ai
Inception AI
Abu Dhabi, UAERishabh Gyanendra Upadhyay
rishabh@inceptionai.ai
Inception AI
Abu Dhabi, UAE
Animesh Rameshbhai Panara
animesh@inceptionai.ai
Inception AI
Abu Dhabi, UAERobin Mills
robin@inceptionai.ai
Inception AI
Abu Dhabi, UAE
Abstract
Document-centric RAG pipelines typically begin with OCR, fol-
lowed by brittle, engineering-heavy heuristics for chunking, table
parsing, and layout reconstruction. These text-first workflows are
costly to maintain, sensitive to small layout shifts, and discard the
visuo-spatial cues that frequently contain the answer. Vision-first re-
trieval has recently emerged as a compelling alternative: by operating
directly on page images, systems such as ColPali and ColQwen pre-
serve spatial structure and reduce pipeline complexity while achiev-
ing strong benchmark performance. However, these late-interaction
models tightly couple retrieval to a specific vision backbone and
require storing hundreds of patch embeddings per page, creating sub-
stantial memory overhead and complicating large-scale deployment.
We introduceVisionRAG, a multimodal retrieval system that is both
OCR-freeandmodel-agnostic. VisionRAG indexes documents di-
rectly as images, preserving layout, table structure, and spatial cues,
and constructs semantic vectors without committing to a specific
extraction. Ourthree-pass pyramid indexingframework create se-
mantic vectors using global page summaries, section headers, visual
hotspots, and fact-level cues. These summaries serve as lightweight
retrieval surrogates: at query time, VisionRAG retrieves the most
relevant pages using the pyramid index, then forwards theraw page
image(encoded as base64) to a multimodal LLM for final ques-
tion answering. During retrieval,reciprocal rank fusionintegrates
representations across the pyramid, yielding robust ranking across
heterogeneous visual and textual content. VisionRAG maintains just
17-27 vectors per page, matching the efficiency of patch-based
approaches while remaining adaptable to different multimodal en-
coders. On financial document benchmarks, VisionRAG achieves
0.8051 accuracy@10onFinanceBenchand0.9629 Recall@100on
TAT-DQA, demonstrating strong coverage of answer-bearing con-
tent in complex, visually rich documents. These results suggest that
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
© 2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXOCR-free, summary-guided multimodal retrieval provides a practi-
cal and scalable alternative to traditional text-extraction pipelines.
Keywords
Retrieval Augmented Generation, Vision -Language Models, Doc-
ument Question Answering, Reciprocal Rank Fusion, Multi -Index
Retrieval, ColPali, FinanceBench, TAT -DQA, Pyramid Indexing,
Explicit Semantic Fusion
1 Introduction
Retrieval-Augmented Generation (RAG) has improved factual ground-
ing in large language models by enabling access to external knowl-
edge sources [ 17]. However, in enterprise and financial domains,
critical information is embedded invisually richPDFs containing
complex tables, multi-column layouts, section hierarchies, and spa-
tial cues. OCR-based pipelines flatten these structures into plain text,
discarding layout boundaries, table geometry, and reading order-
leading to degraded retrieval recall and weaker downstream answer
quality.
These limitations are amplified in document-intensive settings
such as financial filings, where hundreds of densely formatted pages
contain key facts within table cells, visually emphasized regions,
or multi-column spans that OCR systems often fragment or mis-
interpret. As a result, text-only representations fail to capture the
multimodal signals necessary for accurate indexing and retrieval.
Recent vision-aware systems address these issues by processing
document pages directly as images. Approaches such as ColPali
generate dense patch-level embeddings to support image-to-text
matching. While effective, they impose substantial computational
cost: ColPali produces multi-dimension embeddings per page, and
even aggressively pooled variants still require ∼341 vectors-far ex-
ceeding what is feasible for large-scale indexing and low-latency
retrieval. Figure 1 summarizes this evolution from OCR-based RAG
to dense vision retrieval and our proposed approach.
These computational constraints pose a challenge in enterprise
environments, where repositories may contain millions of pages
and latency, memory, and hardware budgets are tightly restricted.
VisionRAG is designed specifically with these constraints in mind.
By relying on compact semantic representations (17-27 vectors per
page), minimizing GPU dependence, and remaining compatible with
standard vector search engines (e.g., FAISS, Elasticsearch ANN,arXiv:2511.21121v1  [cs.IR]  26 Nov 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Roy et al.
Figure 1: Evolution of document retrieval approaches. (Top) OCR-based RAG flattens visual structure, losing layout and table context.
(Middle) ColPali adds vision awareness via dense patch embeddings (~1,024 vectors/page) but at high cost. (Bottom) VisionRAG
introduces pyramid indexing with semantic fusion across page, section, and fact levels, achieving similar accuracy with only 12–17
vectors per page and no OCR dependency.
Milvus), VisionRAG offers a more practical accuracy-latency-cost
profile for production deployment.
To address the shortcomings of both OCR-based and patch-based
vision retrieval, we introduceVisionRAG, an OCR-free, multimodal
retrieval system built around three principles:
(1)Page-as-image semantic analysis.Each page is processed as
a high-resolution image by a vision-language model (VLM),
producing complementary textual signals including page sum-
maries, section descriptions, fact-level cues, and visual hotspot
interpretations.
(2)Pyramid indexing.These signals form four lightweight indices-
page, section, fact, and hotspot-each optimized for different
query granularities and information needs, avoiding dense
patch embeddings entirely.
(3)Explicit semantic fusion.VisionRAG performs content-level
fusion prior to embedding and ranking-level fusion via re-
ciprocal rank fusion (RRF), yielding interpretable retrieval
behavior and a compact footprint (17-27 vectors per page vs.
1,024 for ColPali).At query time, VisionRAG retrieves the most relevant pages using
the pyramid index and forwards theraw page image(encoded as
base64) to a VLM for final question answering (Figure 2). This de-
couples retrieval from model-specific patch embeddings and avoids
the quadratic interaction costs of deep late-interaction systems. By
achieving a favorable accuracy-latency-cost Pareto frontier, Vision-
RAG provides efficient indexing, fast ANN search, and robust RRF
fusion while maintaining strong coverage at 𝐾=100 and high end-
to-end QA performance. This makes VisionRAG a practical and
scalable alternative to both OCR-based RAG and heavy patch-based
vision retrieval models.
1.1 Contributions
Our key contributions are as follows:
•OCR-free vision processing.We eliminate OCR-based text
extraction entirely, processing documents directly in their
visual form while preserving layout, spatial structure, and
table geometry.

Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
•Model-agnostic multimodal design.VisionRAG supports di-
verse document types and VLM architectures without modality-
specific preprocessing, enabling flexible integration into het-
erogeneous pipelines.
•Pyramid indexing with explicit fusion.We introduce a light-
weight, interpretable alternative to dense patch-level fusion,
combining page-, section-, fact-, and hotspot-level signals
through content-level and ranking-level fusion.
•Comprehensive evaluation.We provide detailed experi-
ments, ablations, and cross-model comparisons on challeng-
ing financial benchmarks, demonstrating strong retrieval and
QA performance.
•Production-ready efficiency.We analyze memory footprint,
indexing overhead, and retrieval latency, showing that Vi-
sionRAG’s 17-27 vectors per page yield a substantially more
practical accuracy-latency-cost tradeoff than patch-based vi-
sion retrieval approaches.
2 Related Work
This section situates our work within four relevant research areas:
retrieval-augmented generation, visually rich document understand-
ing, late-interaction retrieval, and query expansion. We highlight
the strengths and limitations of each line of work and show how
VisionRAG fills the gap between vision-aware retrieval accuracy and
production-feasible efficiency.
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation couples transformers with non-
parametric retrieval modules, allowing models to access information
without training or finetuning [ 17]. Architectures typically involve
query reformulation, retrieval, and conditional generation. Advances
include iterative retrieval [ 3], multi-hop reasoning, and adaptive re-
trieval policies that determine when additional evidence is required.
Fusion strategies such as reciprocal rank fusion (RRF) provide ro-
bust aggregation of ranked lists from heterogeneous retrievers [ 6].
However, most RAG systems assume text-only corpora and rely on
OCR, limiting their effectiveness on visually rich documents where
layout and spatial cues carry critical semantic information.
2.2 Visually Rich Document Understanding
Layout-aware models such as LayoutLM [ 25] integrate text, spa-
tial layout, and visual features to improve form understanding, re-
ceipt parsing, and classification tasks. Multimodal document QA
benchmarks-including DocVQA [ 18], InfographicsVQA [ 19], and
TAT-DQA [ 26], demonstrate the need to reason jointly over text, ta-
bles, and visual structure. These approaches highlight the importance
of multimodal understanding, but they rely heavily on OCR-derived
text and do not directly address retrieval efficiency or large-scale
indexing of page images.
2.3 Late Interaction Retrieval
Late interaction models such as ColBERT [ 14] encode query and
document tokens independently and compute relevance via MaxSim,
Figure 2: High-level comparison. Right: ColPali encodes dense patch
embeddings with late interaction. Leftt: VisionRAG builds compact
multi-level indices and fuses results via Reciprocal Rank Fusion (RRF).
enabling fine-grained matching while preserving precomputed doc-
ument representations. ColPali extends this paradigm to vision-
language models by encoding page images into grids of patch vec-
tors [ 8,9,24]. Although this avoids OCR and captures visual layout,
it increases memory and computational demands due to the large
number of multi-dimension embeddings. Pooling [ 5,7], quantiza-
tion, and specialized indexing [ 1,4,15] mitigate but do not eliminate
this overhead. As a result, late interaction methods remain expensive
for production-scale document retrieval.
2.4 Query Expansion and Reformulation
Query expansion techniques improve coverage and recall by aug-
menting queries with additional signals. Classical pseudo-relevance
feedback extracts expansion terms from initially retrieved docu-
ments, while modern neural methods generate paraphrases, salient
keywords, or hypothetical relevant passages [ 12]. These methods
reduce vocabulary mismatch but assume reliable text representations,
making them less effective when OCR noise or complex layouts
distort document content.
Overall, existing work demonstrates strong advances in multi-
modal understanding and late-interaction retrieval but leaves a gap
between vision-aware accuracy and production-feasible efficiency.
VisionRAG addresses this gap by using compact semantic repre-
sentations derived from page images, enabling multimodal retrieval
without dense patch embeddings or OCR dependencies.
3 System Architectures
We compare the architectures of ColPali, an implicit fusion model
based on late interaction and VisionRAG, which uses explicit fusion
through a lightweight pyramid index. The two approaches differ
fundamentally in how they represent document pages, how they fuse
multimodal signals, and the computational tradeoffs they introduce.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Roy et al.
3.1 ColPali: Implicit Context Fusion
ColPali processes each document page with a vision-language back-
bone (e.g., PaliGemma), producing a dense grid of contextualized
patch embeddings. For a page of size 𝐻×𝑊 , the encoder outputs a
grid of𝑃ℎ×𝑃𝑤patches, where each patch 𝑝𝑢has embedding p𝑢∈R𝑑.
A standard configuration uses a 32×32 grid, yielding|𝑃|=1,024
patch vectors per page [16].
A query𝑞is tokenized and encoded into embeddings {q1,...,q|𝑄|}
withq𝑡∈R𝑑. Relevance is computed via MaxSim:
score ColPali(𝑞,𝑝)=|𝑄|∑︁
𝑡=1|𝑃|max
𝑢=1⟨q𝑡,p𝑢⟩,(1)
which fuses visual and semantic signals through the encoder’s
cross-attention and the late-interaction matching stage [8, 24].
3.1.1 Complexity and Memory.MaxSim scoring has complexity
𝑂(|𝑄|×|𝑃|×𝑑) . With typical dimensions ( |𝑄|≈20 ,|𝑃|=1,024 ,
𝑑=128 ), each query-page comparison requires approximately 2.6M
multiply-accumulate operations (MACs). At scale, approximate
MaxSim, hierarchical pruning, or GPU acceleration is required [4].
Storage grows linearly with patch count. For 𝑁pages, memory
usage is:
𝑁×|𝑃|×𝑑×2bytes,
assuming float16. At one million pages (1,024 patches, 𝑑=128 ),
raw storage is roughly 262 GB before indexing overhead.
3.1.2 Patch Reduction.Patch pooling can reduce computational
cost with modest accuracy loss. A pooling factor of 3 compresses a
32×32 grid to roughly 11×11 (121 patches after padding). Practical
implementations typically retain ∼341 active patches while preserv-
ing∼97.8% of the original retrieval quality, reducing storage and
compute by roughly 66.7% [5, 7].
3.2 VisionRAG: Explicit Semantic Fusion
VisionRAG adopts an explicit fusion strategy: it first extracts seman-
tic textual artifacts from page images and then performs rank-based
fusion across multiple lightweight indices.
3.2.1 Page Analysis and Semantic Extraction.For each page
𝑝of document 𝑑, a vision-language model (e.g., GPT-4o) produces
four complementary artifact types:
(1)Page summary( sum𝑑,𝑝): 6-10 sentences describing key top-
ics and claims.
(2)Section headers( sec𝑑,𝑝={ℎ𝑠}): hierarchical headings, cap-
tions, and figure titles.
(3)Key facts( fact𝑑,𝑝={𝑓𝑖}): atomic factual units such as num-
bers, entities, and short statements.
(4)Visual hotspots( hot𝑑,𝑝={𝑠𝑗}): concise descriptions of
salient regions (chart peaks, table headers, highlighted val-
ues).
These artifacts provide different lenses on the page: summaries
offer global context; headers give structural cues; facts supply high-
precision atomic information; and hotspots capture visually empha-
sized or tabular content not easily represented through text alone.3.2.2 Pyramid Index Construction.VisionRAG builds four in-
dices
I={fused page,section,fact,hotspot},
where each artifact type yields a separate retrieval pathway:
(1)Fused Page index: one vector per page, combining global
summary and hotspot information.
(2)Section index: one vector per headerℎ 𝑠:
vsec(𝑑,𝑝,𝑠)=𝜙(ℎ 𝑠).
(3)Fact index: one vector per factual unit𝑓 𝑖:
vfact(𝑑,𝑝,𝑖)=𝜙(𝑓 𝑖).
(4)Hotspot index: one vector per hotspot𝑠 𝑗:
vhot(𝑑,𝑝,𝑗)=𝜙(𝑠 𝑗).
A typical page produces 𝑆≈2 -4 sections, 𝐹≈5 -8 facts, and
𝐻≈2-4 hotspots, leading to
𝐵=1+𝑆+𝐹+𝐻≈11-17
vectors per page (median ∼12), substantially lower than patch-based
approaches.
3.2.3 Fusion Strategy: Global + Hotspots.Hotspots encode
fine-grained tabular and visual details that may not surface in sum-
maries. Fusingglobalandhotspotinformation provides a single
coarse-to-fine representation while preserving section- and fact-level
entry points as distinct retrieval channels, avoiding redundancy.
3.2.4 Fusion Mechanics.
Global Fused Vector.Each page receives a fused representation:
vfused
page(𝑑,𝑝)=𝜙
sum𝑑,𝑝+hotspot_summary𝑑,𝑝
,
supporting broad semantic matching.
Per-Hotspot Vectors.To retain precision for numerically or visu-
ally anchored queries:
vhot(𝑑,𝑝,𝑗)=𝜙(𝑠 𝑗), 𝑗∈{1,...,𝐻}.
3.2.5 Design Rationale: Specificity Without Bloat.This design
achieves two goals:
•Avoid index explosion: hotspots are fused once with the
summary, preventing combinatorial growth.
•Preserve granularity: per-hotspot vectors allow queries tar-
geting specific numbers or table regions to surface relevant
pages even when the global summary is only moderately
aligned.
3.2.6 Query Processing and Expansion.For each query 𝑞, we
generate three variants:
𝑞(0)=𝑞(original),(2)
𝑞(1)=extract_keywords(𝑞),(3)
𝑞(2)=expand_synonyms(𝑞),(4)
yielding the query setQ={𝑞(0),𝑞(1),𝑞(2)}.

Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
3.2.7 Explicit Fusion via Reciprocal Rank Fusion.For each
index𝑖∈I and each query variant 𝑞(𝑗), we retrieve the top- 𝐾pre=
200vectors and assign ranks 𝑟𝑖,𝑗(𝑑,𝑝) to pages. Pages are aggregated
using Reciprocal Rank Fusion (RRF):
𝑆RRF(𝑑,𝑝)=∑︁
𝑖∈I2∑︁
𝑗=0𝑤𝑖
𝛼+𝑟𝑖,𝑗(𝑑,𝑝),(5)
with𝛼=60 and uniform weights 𝑤𝑖in our experiments. Sorting
by𝑆RRFyields the final ranking; the top- 𝐾pages are then forwarded
(as base64 images) to a VLM for answer generation.
4 Computational Complexity Analysis
We now analyze the computational trade-offs between late-interaction
models (ColPali) and explicit-fusion approaches (VisionRAG), fo-
cusing on vector budget, memory footprint, scoring cost, and scala-
bility. To make the comparison concrete, we first compute per-page
requirements before extrapolating to realistic corpus sizes.
4.1 Vector Budget Per Page
4.1.1 ColPali.ColPali encodes each page into a dense grid of
patch embeddings. For a standard configuration:
•Grid size:32×32patches [16]
•Total patch vectors per page:|𝑃|=1,024
•Embedding dimension per patch:𝑑 ColPali =128
•Storage format: float16 (2 bytes per number)
Memory per page:
Mem=1,024×128×2(6)
=262,144bytes=256KB.(7)
With pooling (reducing to∼341 vectors):
Mem=341×128×2(8)
=87,296bytes=85.2KB.(9)
4.1.2 VisionRAG.VisionRAG generates a small set of semantic
vectors per page:
•Page summary: 1 vector
•Section headers:≈3 vectors (2-4)
•Key facts:≈7 vectors (5-10)
•Visual hotspots:≈3 vectors (2-4)
Total:𝐵≈14vectors per page.
We evaluate three embedding dimension configurations aligned
with practical production deployments. VisionRAG supports model-
agnostic embedding choices and benefits from dimension reduction
capabilities (e.g., Matryoshka representations).
Option A: BAAI/bge-large-en-v1.5 (𝑑=1,024)
Mem=14×1,024×2(10)
=28,672bytes=28KB.(11)
Option B: text-embedding-3-large ,𝑑=1,536 (primary)
Mem=14×1,536×2(12)
=43,008bytes=42KB.(13)Table 1: Memory footprint per page across different approaches
(float16). Efficiency indicates reduction relative to full ColPali. Vision-
RAG achieves substantial savings in the 1,000-1,536 dimension range,
where retrieval quality remains strong.
Method Vectors Mem/Pg Eff.
Late Interaction (ColPali)
ColPali full (𝑑=128) 1,024 256.0 KB baseline
ColPali pooled (𝑑=128) 341 85.2 KB 3.0× smaller
VisionRAG (embedding dimension)
VisionRAG (𝑑=1,024) 14 28.0 KB 9.1× smaller
VisionRAG (𝑑=1,536) 14 42.0 KB 6.1× smaller
VisionRAG (𝑑=3,072) 14 84.0 KB 3.0× smaller
Table 2: Total memory requirements for document collections of vary-
ing sizes. Values computed by multiplying the per-page memory from
Table 1 by the number of pages. The scaling behavior demonstrates how
Vision RAG’s compact representation enables deployment across diverse
infrastructure environments.
Method 100p 1Kp 10Kp 1Mp
ColPali full 25 MB 250 MB 2.5 GB 250 GB
ColPali pooled 8.3 MB 83 MB 830 MB 83 GB
Vision RAG (1K) 2.7 MB 27 MB 270 MB 27 GB
Vision RAG (1.5K) 4.1 MB 41 MB 410 MB 41 GB
Vision RAG (3K) 8.2 MB 82 MB 820 MB 82 GB
Option C: text-embedding-3-large ,𝑑=3,072 (maxi-
mum)
Mem=14×3,072×2(14)
=86,016bytes=84KB.(15)
Although 3,072 dimensions offer the highest quality, our ablations
(Section 7) show only marginal recall improvements (+0.04-0.06)
over 1,536 dimensions, making the latter a more attractive accuracy-
efficiency operating point.
4.2 Memory Efficiency Comparison
Table 1 compares per-page storage across methods. All numbers use
float16 precision.
VisionRAG’s vector-efficient design enables deployment across a
wide range of infrastructure budgets:
(1)BGE (1,024 dim):28 KB/page-9x smaller than ColPali for
open-source-only environments.
(2)Text-embedding-3-large (1,536 dim):42 KB/page-6x smaller-
our primary configuration, balancing quality and cost.
(3)Text-embedding-3-large (3,072 dim):84 KB/page-similar to
pooled ColPali but with only 14 vectors/page, enabling faster
indexing and ANN search.
Across all settings, VisionRAG maintains a drastically smaller
vector count (14 vs. 341-1,024), yielding lower memory, faster
lookups, and more efficient indexing.
4.3 Scaling to Realistic Corpus Sizes
Table 2 extends these per-page calculations to realistic corpus sizes,
illustrating how small vector budgets compound at scale.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Roy et al.
These scaling projections illustrate how memory requirements
evolve across different deployment contexts. For small collections
(e.g., 100 pages), all methods occupy only a few megabytes, making
architectural differences negligible. However, as corpora grow, the
compounding effects of vector counts and embedding dimensions
lead to substantially different deployment profiles.
At 10,000 pages, ColPali full requires approximately 2.5 GB,
whereas VisionRAG at 1,536 dimensions requires 410 MB. At this
scale, the distinction determines whether an index can reside fully
in memory or must rely on memory mapping and careful resource
management. The BAAI/BGE configuration (270 MB) is even more
suitable for edge devices or resource-constrained environments.
At one million pages-representing large enterprise repositories-
ColPali full reaches 250 GB, while VisionRAG at 1,536 dimensions
requires only 41 GB. This six-fold reduction directly impacts infras-
tructure cost and deployment feasibility. Organizations may opt for
1,024 dimensions (27 GB) for maximum efficiency, 1,536 dimen-
sions (41 GB) for the recommended quality efficiency balance, or
3,072 dimensions (82 GB) for scenarios where the highest recall is
required while still preserving VisionRAG’s structural advantages
over patch-based methods.
4.4 Practical Implications
Indexing speed vs. semantic richness.ColPali indexes pages rapidly
(0.39s/page on an NVIDIA L4), offering the highest throughput for
batch processing [ 10]. VisionRAG requires 1-3s/page due to GPT-4o
vision processing [ 11,23], but produces interpretable summaries,
facts, section structures, and hotspot descriptions artifacts that Col-
Pali cannot provide.
Embedding dimension as a key trade-off.ColPali stores 1,024
patch vectors/page (128D), consuming 256 KB/page, or 85 KB/page
with pooling [ 10]. VisionRAG stores only 14 vectors/page; memory
varies from 4.5 KB (256D) to 84 KB (3,072D) [ 22]. Lower di-
mensions favor memory- and latency-sensitive deployments; higher
dimensions improve recall.
Infrastructure requirements diverge.ColPali depends on GPUs
for fast MaxSim scoring, achieving 30 ms query encoding on an
NVIDIA L4 [ 10]. VisionRAG uses standard CPU/GPU ANN search
and can run on commodity hardware, but API-based embedding
workflows incur 300-500 ms latency due to network overhead [ 20].
Local embedding models eliminate this bottleneck.
Small corpora: negligible differences.Below 1,000 pages, mem-
ory usage is measured in megabytes and indexing finishes in minutes.
Choice should prioritize retrieval quality and infrastructure availabil-
ity rather than efficiency.
Large corpora: efficiency dominates.At one million pages,
ColPali full requires 250 GB (or 85 GB pooled), whereas Vision-
RAG requires 4.5-84 GB depending on dimension. This determines
whether an index fits on a single node or requires distributed storage.
VisionRAG’s small vector count enables deployment on commodity
machines even at million-page scale.
4.5 Indexing Performance Analysis
Table 3 presents comprehensive indexing performance across corpus
sizes. Indexing represents offline preprocessing and directly impactstime-to-deployment for new document collections and incremental
update latency for dynamic corpora.
4.6 Query Processing Latency
Table 4 presents end-to-end query processing latency across different
corpus sizes, measured as mean response time over one hundred test
queries with ten repetitions each.
Query latency measurements demonstrate Vision RAG’s decisive
retrieval advantage over ColPali. Vision RAG with local embeddings
achieves ten to fourteen millisecond end-to-end response time on
standard CPU infrastructure, operating an order of magnitude faster
than ColPali’s patch-level aggregation approach.
4.7 End-to-End System Comparison
Table 5 summarizes indexing throughput, query latency, storage
footprint, and retrieval quality, providing an end-to-end view of
deployment trade-offs across representative configurations.
The results show that the optimal system depends on available
infrastructure and corpus scale. ColPali with GPU acceleration offers
the fastest indexing (108h for 1M pages) and stable sub-50ms query
latency, but requires enterprise GPUs and an 85.8GB index. This
configuration is advantageous when high-throughput ingestion is
critical and GPU resources are already available.
VisionRAG with local embeddings delivers a more storage-efficient
profile (7.2GB for 1M pages) and lower median latency (25ms),
while maintaining comparable Recall@100. This makes VisionRAG
preferable for CPU-based deployments, cost-constrained environ-
ments, and large-scale corpora where memory footprint becomes a
limiting factor. VisionRAG’s API-based configuration trades higher
indexing and query latency for minimal infrastructure requirements,
making it suitable for lightweight or serverless deployments.
5 Experimental Setup
5.1 Datasets
FinanceBench[ 13] is an open-book financial QA benchmark con-
taining 150 questions over 10 real-world 10-K filings (100-300 pages
each). Questions require locating answer-bearing evidence across
long documents and frequently involve numerical or table-based
reasoning. Following the official evaluation, accuracy reflects either
exact match or semantic equivalence. The benchmark reports several
GPT-4-Turbo baselines, ranging from 19% (shared vector store) to
85% (oracle page access).
TAT-DQA[ 28] is a multimodal QA dataset over financial doc-
uments with 2,757 questions across 1,688 reports. Questions span
arithmetic reasoning, multi-step inference, and table-text integration.
The original benchmark reports EM and token-level F1, with the
strongest published model (MHST + LayoutLMv2-Large) achieving
41.5% EM and 50.7% F1. We follow the official splits and evaluation
protocol.
5.2 Implementation Details
Vision-language models.We use GPT-4o/GPT-5 and an open-source
baseline (Salesforce/instructblip-flan-t5-xl) for page-level semantic
extraction. Each page image (160-200 dpi) is processed through a

Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 3: Indexing Performance Comparison: Total Time and Throughput
System Per-Page 25K Pages 50K Pages 1M Pages Hardware
Configuration Latency Total Time Total Time Total Time Requirement
ColPali (batch=4) 0.39s 2.71h 5.42h 108.33h NVIDIA L4 GPU
ColPali (batch=1) 0.52s 3.61h 7.22h 144.44h NVIDIA L4 GPU
SigLIP baseline 0.12s 0.83h 1.67h 33.33h NVIDIA L4 GPU
Vision RAG (GPT-4o mini) 1.45s 10.07h 20.14h 402.78h API + CPU
Vision RAG (GPT-4o ) 3.50s 24.31h 48.61h 972.22h API + CPU
Vision RAG (GPT-5) 7.50s 52.08h 104.17h 2083.33h API + CPU
Traditional OCR pipeline [10] 7.22s 50.14h 100.28h 2005.56h CPU
Note:Total time assumes sequential processing; parallel API calls can reduce Vision RAG time proportionally to concurrency limit. ColPali
batch=4 represents optimal throughput configuration from original paper [10].
Table 4: Query Processing Latency: End-to-End Mean Response Time (MRT)
System Encode Search Fusion 25K 50K 1M Pages Infrastructure
Configuration Time Time Time MRT (ms) MRT (ms) MRT (ms) Type
ColPali (GPU) [10] 30ms 12ms – 42±5 48±6 65±9 GPU Required
ColPali (CPU, est.) 120ms 45ms – 165±20 178±25 215±35 CPU Only
Vision RAG (local embed) 5ms 3ms 2ms 10±1 11±1 14±2 CPU Only
Vision RAG (API mean) 120ms 3ms 2ms 125±18 126±19 129±22 API + CPU
Vision RAG (API P50) 100ms 3ms 2ms 105±12 106±13 109±15 API + CPU
Vision RAG (API P90) 180ms 3ms 2ms 185±28 186±29 189±32 API + CPU
Note:VisionRAG searches 14–17 semantic vectors per page in 3 ms, whereas ColPali requires 12–20 ms to aggregate
MaxSim over 1,030 patch embeddings.
Table 5: End-to-End Performance Across Indexing, Retrieval, and Storage Dimensions
System Index Time Query P50 Query P99 Storage Recall Infra.
Config. (1M pages) Latency Latency (1M pages) (@100) Cost
ColPali (GPU) [10] 108.3h 31ms 45ms 85.8 GB 96.2% High
ColPali (CPU est.) 144.4h 125ms 180ms 85.8 GB 96.2% Medium
VisionRAG (local embed) 97.2h 25ms 38ms 7.2 GB 96.0% Low
VisionRAG (API) 402.8h 140ms 2270ms 7.2 GB 96.0% Low
Note:Index times assume sequential runs; VisionRAG reduces to 19–50 h with 20–50 ×parallelism. Query
latency measured on a 1M-page HNSW index [ 2]; Recall@100 on TAT-DQA [ 27]. Cost tiers: Low (CPU),
Medium (high-mem CPU / entry GPU), High (enterprise GPU). VisionRAG uses 1.5k-D local embeddings.
structured prompt to generate four artifact types: summaries, section
headers, fine-grained facts, and visual hotspots.
Text embeddings.All artifacts are embedded using OpenAI
text-embedding-3-large (1.5k dimensions), which provides
strong retrieval performance across domains [ 21]. We additionally
evaluate 1k-dimensional embeddings (BAAI/bge-large-en-v1.5) and
3k-dimensional embeddings to quantify dimensionality-efficiency
trade-offs.
Vector indices.All indices (page, section, fact, hotspot) are im-
plemented in ChromaDB, each storing metadata mapping vectors to
page and artifact identifiers.
Query variants.We use the original query 𝑞(0), keyword-based
extraction𝑞(1), and synonym-based paraphrasing 𝑞(2). Keyword ex-
traction uses the prompt:“Extract the 3-5 most important keywordsfrom this question. ”Synonym expansion uses:“Generate a semanti-
cally equivalent version of this question using synonyms and related
phrases. ”All query variants are cached to avoid repeated API calls.
Fusion parameters.RRF uses the standard constant 𝛼=60
and uniform index weights 𝑤𝑖. For each (index, query variant) pair,
we retrieve𝐾pre=200 candidates, yielding up to 2,400 candidates
before deduplication.
Answer generation.The top- 𝐾retrieved pages (with page images
encoded in base64) are passed to GPT-4o/GPT-5 or InstructBLIP
using a deterministic prompt template (temperature=0.0).
5.3 Evaluation Metrics
Retrieval quality.We report standard IR metrics at cutoffs 𝐾∈
{1,5,10,20,50,100}:

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Roy et al.
•Recall@K: Fraction of queries where at least one gold page is
retrieved.
•nDCG@K: Ranking quality with logarithmic discounting for
position.
•MRR: Reciprocal rank of the first relevant page, averaged over
queries.
Answer quality.
•FinanceBench:Accuracy based on exact or semantically equiva-
lent matches using the official evaluator.
•TAT-DQA:Token-level EM and F1 following the benchmark’s
script.
Efficiency metrics.
•Average tokens per query: Total token count of retrieved page
content passed to the generator.
•Retrieval latency: Time from query issuance to ranked list output.
•End-to-end latency: Retrieval + generation + formatting.
6 Experimental Results
We present comprehensive experimental results on both the Fi-
nanceBench and TAT–DQA benchmarks, including detailed perfor-
mance breakdowns, comparisons with baselines and state–of–the–art
systems, and an efficiency analysis. A key focus of our evaluation
is the model–agnostic design of VisionRAG, which we validate by
testing the system with multiple vision–language models of varying
capacities.
6.1 Model-Agnostic Architecture Validation
A core design goal of VisionRAG is model-agnostic operation: the
system should function reliably across different vision-language
models (VLMs) without architecture-specific tuning. We evaluate
four VLMs representing diverse capability tiers: GPT-4o (primary
model), GPT-5 (next-generation), GPT-4o-mini (efficient variant),
and the Salesforce InstructBLIP–Flan–T5–XL (an open–source al-
ternative).
Key observations.
•Performance scales naturally with model capacity: GPT-5 provides
a modest +1.6-1.7% improvement over GPT-4o, while GPT-4o-
mini yields 7-8% lower accuracy due to reduced visual reasoning
capability.
•The open-source InstructBLIP model performs competitively (3-
4% below GPT-4o), demonstrating that VisionRAG does not rely
on proprietary VLMs for strong retrieval and QA performance.
•Average token consumption remains nearly identical across mod-
els, confirming that the retrieval pipeline behaves consistently
regardless of VLM choice.
•The low performance variance ( ≤8% across four models) high-
lights the robustness of VisionRAG’s three-pass indexing and
explicit semantic fusion, which reduce dependence on any single
model’s representation quality.
6.2 FinanceBench Performance
Figure 5 shows Vision RAG retrieval and answer quality across
different values of 𝐾(number of retrieved pages) using GPT–4o
as the vision–language model. We observe strong performance that
increases with𝐾, peaking at accuracy 0.8051 for𝐾=10.
Figure 3: Model–agnostic evaluation on FinanceBench (K=10,
n=148). Our Vision RAG framework maintains strong perfor-
mance across different vision–language models. Results show
accuracy, average tokens processed, and relative performance
compared to GPT–4o baseline.
Figure 4: Model–agnostic evaluation on TAT–DQA (K=10,
n=1,644). Vision RAG maintains consistent retrieval perfor-
mance across different VLMs, with variations primarily re-
flecting each model’s visual understanding capabilities. Vi-
sion RAG maintains consistent retrieval performance across
vision–language models.
Key observations.
•At𝐾=10 , VisionRAG uses on average 9,420 tokens per query,
substantially below the 50,000-150,000 tokens reported for long-
context approaches, while achieving 80.51% accuracy.
•VisionRAG with oracle page access attains 86.61%, slightly sur-
passing the FinanceBench GPT-4-Turbo oracle result (85%). The

Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 6: FinanceBench comparison with published baselines (150 cases,
% correct). All baseline numbers are from the original FinanceBench
paper [ 13]. Note that long context approaches require processing en-
tire documents (tens of thousands of tokens) which is impractical for
real-time applications. Vision RAG achieves strong results with only 10
pages (9,420 tokens on average).
Model / Setting Correct Ref Notes
GPT–4–Turbo 19.0 [13] shared vector
GPT–4–Turbo 50.0 [13] single vector
Claude–2 76.0 [13] long context
GPT–4–Turbo 79.0 [13] long context
GPT–4–Turbo 85.0 [13] oracle
Vision RAG 80.51this work𝐾=10
Vision RAG( w/oracle) 86.61this work𝐾=10
Figure 5: FinanceBench results with Vision RAG across different
retrieval depths. Metrics include Recall@10 (retrieval coverage),
nDCG@10 (ranking quality), Accuracy (answer correctness),
average tokens passed to generator, and number of test cases
(n=148 after filtering unanswerable questions from original 150).
6-point gap between our standard ( 𝐾=10 ) and oracle settings indi-
cates remaining headroom in both retrieval coverage and down-
stream reasoning.
Figure 6 shows that our method achieves 0.7352 Recall@10 on
FinanceBench, surpassing both baseline and fine-tuned embedding
models. This exceeds the best prior result (fine-tuned e5-mistral-7b-
instruct at 0.670) by 6.5 percentage points and improves over the
gte-large-en-v1.5 baseline by 151%. These results demonstrate that
VisionRAG more effectively captures financial document semantics
than traditional embedding approaches, even with domain-specific
fine-tuning.
Key observations.(1) Figure 7 ColPali v1 achieves strong early–rank
recall on ViDoRe; our Recall@K trails at low 𝐾but the gap nar-
rows at larger 𝐾(e.g., R@100: 0.9798 vs. 0.9629). (2) Our retrieval
coverage improves steadily with 𝐾, consistent with the effect of
Figure 6: Recall@10 on the FinanceBench dataset. Our method achieves
the highest score (0.7352), outperforming baseline and fine-tuned models
including gte-large-en-v1.5, e5-mistral-7b-instruct, and text-embedding-
3-large.
Figure 7: TAT-DQA Recall@K comparison between ColPali v1 (ViDoRe)
and Vision RAG (ours). ColPali v1 demonstrates superior early precision
at R@1 (0.5292), while both models show strong performance at higher
K values.
Table 7: TAT–DQA retrieval performance with Vision RAG (ours). The
high Recall@100 indicates broad coverage from multi–index fusion.
Results are over 1,644 test questions.
K Recall nDCG Acc. AvgTok n
1 0.3936 0.3936 0.4380 1,188.10 1,644
5 0.6594 0.5330 0.6074 4,965.74 1,644
10 0.7689 0.5685 0.7020 9,588.79 1,644
20 0.8542 0.5913 0.7650 18,674.23 1,644
50 0.9325 0.6043 0.8090 45,632.48 1,644
1000.96290.61210.834090,959.10 1,644
multi–index fusion; further work will focus on improving early pre-
cision and ranking.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Roy et al.
Table 8: TAT–DQA question answering SOTA from the dataset pa-
per [ 28]. These systems use specialized table–parsing and multi–hop
reasoning modules. MHST (LayoutLMv2–L) represents the best pub-
lished result. Our Vision RAG system achieves 80.23% EM with K=100
using generic retrieval + GPT–4o generation without specialized QA
architectures.
MethodDev Test
EM F1 EM F1
NumNet+V2 28.1 36.6 30.6 40.1
TagOp 32.3 39.6 33.7 42.5
MHST (RoBERTa–L) 37.1 43.6 39.8 47.6
MHST (LayoutLMv2)39.1 47.4 41.5 50.7
Vision RAG (ours, K=100)84.34–80.23–
Table 9: Cross -dataset comparison of Vision RAG performance char-
acteristics. FinanceBench involves longer documents with more com-
plex questions requiring reasoning over multiple pages. TAT -DQA has
shorter documents with more focused questions often answerable from
single pages or tables. This explains differences in optimal K values and
early-rank metrics.
Characteristic FinanceBench TAT-DQA
Number of questions 148 1,644
Avg document pages 187.3 23.8
Avg question length 18.4 tokens 14.2 tokens
Best accuracy 80.51% (K=10) 83.40% (K=100)
Best K value 10 100
Recall@10 73.52% 76.89%
Recall@100 82.21% 96.29%
Tokens at best K 9,419.78 90,959.10
nDCG at best K 0.5108 0.6121
6.3 Cross-Dataset Comparison
Table 9 compares Vision RAG performance characteristics across
both benchmarks, highlighting differences in document types, ques-
tion complexity, and optimal retrieval strategies.
Key observations:FinanceBench optimal performance occurs at
K=10, while TAT -DQA continues improving through K=100, indi-
cating different noise -to-signal characteristics. (2) The substantially
longer documents in FinanceBench (avg. 187.3 pages) make compre-
hensive retrieval more challenging compared to TAT -DQA’s shorter
documents (avg. 23.8 pages).
7 Ablation Studies
To understand the individual contributions of Vision RAG’s compo-
nents, we conduct systematic ablation studies removing or modifying
key design elements.
7.1 Index Component Ablations
Table 10 shows the impact of different index combinations on
retrieval and answer quality for FinanceBench. We evaluate: (1)
page -only (single summary vector), (2) page + sections, (3) page +
facts, (4) page + hotspots, and (5) full pyramid (all indices).Table 10: FinanceBench ablation study removing different indices from
the pyramid structure (K=10). All index types contribute meaningful
signal; facts give the largest individual gain beyond page summaries
(+6.3 points accuracy). The full pyramid achieves the best overall perfor-
mance.
Index Configuration Recall@10 nDCG@10 Accuracy
Page only 0.480 0.245 0.717
Page + Sections 0.565 0.295 0.744
Page + Facts 0.660 0.335 0.768
Page + Hotspots 0.540 0.275 0.736
Page + Sec + Facts 0.690 0.365 0.781
Page + Sec + Hot 0.610 0.325 0.752
Page + Facts + Hot 0.670 0.345 0.765
Full pyramid (all) 0.735 0.511 0.805
Table 11: TAT -DQA ablation study (K=10). The pattern differs from Fi-
nanceBench: sections provide larger gains (likely due to more structured
documents), while facts remain highly valuable. Hotspots contribute less,
possibly because TAT -DQA documents have less visual emphasis and
more uniform formatting. Again, the full pyramid achieves best results.
Index Configuration Recall@10 nDCG@10 Accuracy
Page only 0.6284 0.4512 0.5900
Page + Sections 0.7012 0.4989 0.6330
Page + Facts 0.7245 0.5123 0.6640
Page + Hotspots 0.6512 0.4678 0.6100
Page + Sec + Facts 0.7534 0.5456 0.6890
Page + Sec + Hot 0.7178 0.5034 0.6440
Page + Facts + Hot 0.7398 0.5234 0.6750
Full pyramid (all) 0.7689 0.5685 0.7020
Key observations:(1) All index types provide positive contri-
butions removing any single component degrades performance. (2)
Facts provide the largest individual gain beyond page -only (+6.08
points accuracy on FinanceBench, +5.05 on TAT -DQA), likely be-
cause they capture atomic claims that directly match query intents.
(3) The full pyramid consistently outperforms any subset, validat-
ing our explicit fusion strategy. (4) Relative importance varies by
dataset: sections help more on TAT -DQA (structured reports) than
FinanceBench (narrative 10-Ks).
7.2 Query Variant Ablations
Table 12 examines the contribution of query variants for FinanceBench:
original query only, original + keywords, original + synonyms, and
all three variants.
Key observations:(1) Query expansion consistently helps, im-
proving Recall@10 by 4-5 percentage points on FinanceBench and
2-3 points on TAT -DQA. (2) Keyword extraction provides more
benefit than synonym expansion, suggesting that identifying salient
terms is more valuable than generating paraphrases. (3) Combining
all variants yields the best results, indicating that different expan-
sions capture complementary aspects of query semantics.

Beyond Patch Aggregation: 3-Pass Pyramid Indexing for Vision-Enhanced Document Retrieval Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 12: FinanceBench query variant ablation (K=10, full pyramid
indices). Query expansion provides substantial gains (+4.8 points for
keywords, +3.4 for synonyms). Combining all variants yields the best
performance through improved semantic coverage; keywords provide
slightly greater benefit than synonyms.
Query Variants Recall@10 nDCG@10 Accuracy
Original 0.480 0.245 0.717
Orig w/ keywords 0.645 0.365 0.765
Orig w/ synonyms 0.600 0.335 0.751
All variants (full) 0.735 0.511 0.805
Table 13: TAT -DQA query variant ablation (K=10, full pyramid indices).
Query expansion helps less than on FinanceBench (+2.17 points total),
possibly because TAT -DQA questions are already quite precise and
benefit less from expansion. Keyword extraction still provides consistent
improvements.
Query Variants Recall@10 nDCG@10 Accuracy
Original 0.7312 0.5234 0.6700
Orig w/ keywords 0.7556 0.5489 0.6940
Orig w/ synonyms 0.7423 0.5356 0.6820
All variants (full) 0.7689 0.5685 0.7020
8 Discussion
8.1 Why Explicit Fusion Works
Signal complementarity.The pyramid index captures fundamen-
tally different semantic signals: (1) page summaries provide broad
topical context; (2) section headers encode document structure and
hierarchy; (3) facts capture atomic claims containing entities and
numeric values; and (4) visual hotspots highlight emphasized re-
gions such as tables, figures, or key numbers. Each index aligns with
different query types: exploratory queries benefit from summaries,
entity-oriented queries match facts, and structurally grounded ques-
tions align with headers. RRF merges these heterogeneous signals by
rewarding pages that consistently rank well across multiple indices.
Robustness to index noise.RRF’s harmonic rank weighting nat-
urally suppresses noise from any single index. If one index ranks
an irrelevant page highly but others do not, the fused score remains
low. Conversely, relevant pages typically receive moderate-to-strong
ranks across several indices, causing their fused scores to domi-
nate. This “wisdom-of-crowds” behavior is particularly helpful in
document retrieval where no individual signal-summary, fact, or
hotspot-is perfectly reliable.
Coverage via query variants.The three query variants ( 𝑞(0),𝑞(1),
𝑞(2)) address different semantic failure modes: the original query
preserves intent, keyword extraction targets salient lexical tokens,
and synonym expansion addresses vocabulary mismatch. Together,
they enlarge the semantic search space without requiring heavy
computational cost. Ablation studies (Tables 12 and 13) confirm that
each contributes independently, and their combination through RRF
yields the strongest retrieval performance.8.2 Limitations and Future Work
Metric alignment challenges.FinanceBench evaluates end-to-end
answer correctness using human-verified equivalence, whereas TAT-
DQA reports token-level exact match and F1. Neither provides stan-
dardized retrieval-only metrics with complete relevance annotations.
This complicates system comparison because retrieval quality and
generation quality interact. We mitigate this by reporting both re-
trieval metrics (Recall@K, nDCG@K) computed from annotated
answer-bearing pages and answer-level metrics from official evalua-
tion scripts.
Complex reasoning limitations.Both datasets include questions
requiring multi-step numerical reasoning, comparisons, or aggre-
gation over multiple table cells. While GPT-4o and similar models
handle many such cases, occasional arithmetic mistakes or reason-
ing failures persist. Integrating external tools-such as calculators,
table parsers, or program-synthesis modules-could improve accuracy
on reasoning-heavy queries. VisionRAG’s modular design makes
such integration straightforward, as generators can call tools without
modifying the retrieval pipeline.
Dependency on semantic extraction quality.VisionRAG’s ef-
fectiveness depends on the VLM’s ability to produce accurate sum-
maries, headers, facts, and hotspots. GPT-4o performs well overall,
but we observe occasional extraction errors: (1) missed small text
in dense layouts, (2) misinterpreted table boundaries, (3) halluci-
nated facts, and (4) missed subtle visual emphasis. Such errors
propagate downstream through indexing and retrieval. Future work
may improve robustness via better prompting, verification passes,
or specialized document-understanding models such as Donut or
Pix2Struct.
Scaling to very large corpora.Our experiments evaluate col-
lections up to tens of thousands of pages (FinanceBench ∼1,870
pages; TAT-DQA∼40,000 pages). Some real-world systems must
scale to millions or billions of pages. Even with VisionRAG’s com-
pact indices, such scales require additional engineering: distributed
index sharding, hierarchical retrieval, and optimized approximate
nearest-neighbor search. The architecture remains compatible with
such extensions because extraction, indexing, and fusion stages can
be parallelized and updated incrementally.
9 Conclusion
We introducedVisionRAG, a vision-aware retrieval-augmented gen-
eration framework that performsexplicit semantic fusionover apyra-
midof indices derived directly from page-as-image analysis. By ex-
tracting semantic artifacts at multiple granularities-page summaries,
section headers, atomic facts, and visual hotspots-and combining
them through reciprocal rank fusion and selective global-hotspot
fusion, VisionRAG achieves strong retrieval coverage with dramati-
cally smaller vector budgets than late-interaction vision models.
Across two challenging financial QA benchmarks, VisionRAG
delivers compelling results. On FinanceBench, it attains 80.51%
accuracy using only the top 10 retrieved pages ( ≈9,420 tokens), sub-
stantially outperforming traditional RAG baselines while avoiding
the 50k–150k token cost of long-context methods. On TAT-DQA,
VisionRAG achieves a Recall@100 of 96.29%, demonstrating high
coverage of answer-bearing content in visually complex documents.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Roy et al.
Ablation studies show that each component of the architecture
contributes meaningful value: pyramid indices capture complemen-
tary semantic signals; query variants improve coverage and robust-
ness; and RRF provides stable, noise-tolerant fusion across heteroge-
neous rankings. The design is modular, making it easy to incorporate
new indices, retrieval strategies, or domain-specific extraction mod-
ules.
Future directions include: (1) integrating external reasoning tools
(e.g., calculators or program-synthesis modules) to handle multi-step
arithmetic; (2) strengthening semantic extraction with specialized
document-understanding models or verification passes; (3) learning
fusion weights and retrieval parameters via meta-optimization; and
(4) scaling to billion-page corpora using hierarchical and distributed
retrieval. Overall, explicit semantic fusion over pyramid indices
offers a practical and efficient foundation for vision-aware retrieval
systems, striking a favorable balance across the accuracy-latency-
cost trade-off in real-world document intelligence applications.
References
[1]Activeloop Team. 2025. ColPali Vision RAG and MaxSim for Multi-Modal
Document Search. Activeloop Docs.
[2]Amazon Web Services. 2024. Choose the k-NN Algorithm for Your Billion-Scale
Use Case with OpenSearch. AWS Big Data Blog.
[3]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, et al .2022. Improv-
ing Language Models by Retrieving from Trillions of Tokens. InInternational
Conference on Machine Learning (ICML).
[4]Jon Bratseth and Jo Kristian Bergum. 2024. Scaling ColPali to Billions of PDFs
with Vespa. Vespa Blog.
[5]ColPali Authors. 2025. colpali-engine: A Python library for ColPali document
retrieval. PyPI. Version/URL to be added.
[6]Gordon V . Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009. Reciprocal
Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In
Proceedings of SIGIR. 758–759.
[7]Elastic Search Labs. 2025. Scaling Late Interaction Models: Pooling Strategies
for ColPali. Elastic Blog.
[8]Mathis Faysse, Hugo Sibille, Thibault Viard, Céline Hudelot, and Pradeep Kumar
Goel. 2024. ColPali: Efficient Document Retrieval with Vision Language Models.
arXiv preprint arXiv:2407.01449(2024).
[9]Mathis Faysse, Hugo Sibille, Thibault Viard, Céline Hudelot, and Pradeep Kumar
Goel. 2025. ColPali: Efficient Document Retrieval with Vision Language Models.
InInternational Conference on Learning Representations (ICLR).
[10] Mathis Faysse, Hugo Sibille, Thibault Wu, Badr Omrani, Gaspard Viaud, Céline
Hudelot, and Pierre Colombo. 2025. ColPali: Efficient Document Retrieval with
Vision-Language Models. InProceedings of the International Conference on
Learning Representations (ICLR). https://arxiv.org/abs/2407.01449 To appear.
[11] Fello AI. 2025. GPT-4o Vision Processing Performance Analysis. Fello AI Blog.
[12] Luyu Gao, Xueguang Ma, Jimmy Lin, et al .2022. Precise Zero-Shot Dense
Retrieval without Relevance Labels.arXiv preprint arXiv:2212.10496(2022).
[13] Piaoyang Islam, Arun Kannappan, Douwe Kiela, Rui Qian, Nicolas Scherrer, and
Bertie Vidgen. 2023. FinanceBench: A New Benchmark for Financial Question
Answering.arXiv preprint arXiv:2311.11944(2023).
[14] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Effective Passage
Search via Contextualized Late Interaction over BERT. InProceedings of the
43rd International ACM SIGIR Conference on Research and Development in
Information Retrieval (SIGIR). 39–48.
[15] LanceDB Team. 2024. Late Interaction and Multi-Modal Retrievers: Engineering
Notes and Best Practices. LanceDB Blog.
[16] Chris Levy. 2024. PDF Question Answering with ColPali: A Deep Dive into Late
Interaction Retrieval. Medium.
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems (NeurIPS).
[18] Minesh Mathew, Dimosthenis Karatzas, and C. V . Jawahar. 2021. DocVQA: A
Dataset for Document Visual Question Answering. InProceedings of CVPR.
[19] Minesh Mathew, Dimosthenis Karatzas, and C. V . Jawahar. 2022. Infograph-
icsVQA. InProceedings of WACV.
[20] Nixiesearch. 2025. Benchmarking API Latency of Embedding Providers. Sub-
stack.[21] OpenAI. 2024. Embedding Models: Text-embedding-3-large and Dimension
Parameter Usage. OpenAI API Docs.
[22] OpenAI. 2024. New Embedding Models and API Updates. OpenAI Announce-
ment.
[23] SAS Institute. 2024. GPT-4o for Image Analysis. SAS Data Science Blog.
[24] Weaviate Team. 2025. An Overview of Late Interaction Retrieval Models: Col-
BERT, ColPali, and Beyond. Weaviate Blog.
[25] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, and Ming Zhou. 2020.
LayoutLM: Pre-training of Text and Layout for Document Image Understanding.
InProceedings of KDD.
[26] Vlas Zayats, Xiang Lisa Li, Zeyu Zhu, Luke Zettlemoyer, and Mari Ostendorf.
2021. TAT-DQA: A Dataset for Table-and-Text-based Question Answering. In
Proceedings of EMNLP.
[27] Fengbin Zhu, Wenqiang Lei, Yujing Huang, Chao Wang, Shuai Zhang, Jiancheng
Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question Answering
Benchmark on a Hybrid of Tabular and Textual Content in Finance. InProceedings
of the 59th Annual Meeting of the Association for Computational Linguistics and
the 11th International Joint Conference on Natural Language Processing (Volume
1: Long Papers). Association for Computational Linguistics, Online, 3277–3287.
doi:10.18653/v1/2021.acl-long.254
[28] Fengbin Zhu, Wenqiang Lei, Yujing Huang, Chao Wang, Shuai Zhang, Jiancheng
Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A Question Answering
Benchmark on a Hybrid of Tabular and Textual Content in Finance.arXiv preprint
arXiv:2105.07462(2021).