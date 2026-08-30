# PlanSightRAG: A Visual-First Multimodal RAG for Automating Question Answering and Compliance Checking for Civil Standard Plans

**Authors**: Nabaraj Subedi, Shuvo Dip Datta, Ahmed Abdelaty, Shivanand Venkanna Sheshappanavar

**Published**: 2026-08-26 17:54:24

**PDF URL**: [https://arxiv.org/pdf/2608.26091v1](https://arxiv.org/pdf/2608.26091v1)

## Abstract
Civil infrastructure compliance checking has long relied on engineers manually reading legacy 2D plans; however, OCR-based automation strips away the geometry and layout essential for interpreting these plans. We present a Visual-First Multimodal Retrieval-Augmented Generation (RAG) framework called PlanSightRAG. It indexes and reasons directly over plan imagery, integrates a ColNomic-3B multi-vector retrieval, an agentic Planner-Retriever-Auditor-Synthesizer, and MaxSim heatmaps as an evidence trail. We introduce a 4,056-pair benchmark from five state Departments of Transportation (DOT) standard plans (1,898 pages). PlanSightRAG achieves 91.47% Recall@5 on zero-shot retrieval, while on a held-out Michigan DOT corpus, it achieves 91.40%. On synthetic, parametrically-generated compliance drawings, our Qwen2.5-VL-72B pipeline reaches 100% verdict accuracy only when supplied a pre-resolved rule threshold, a controlled ceiling that a non-VLM OCR baseline already reaches at 76.4%. Finally, we demonstrate autonomous visual rule-grounding by extracting numeric limits directly from a specification corpus without any human-supplied rules.

## Full Text


<!-- PDF content starts -->

Highlights
PlanSightRAG: A Visual-First Multimodal RAG for Automating Question Answer-
ing and Compliance Checking for Civil Standard Plans
Nabaraj Subedi, Shuvo Dip Datta, Ahmed Abdelaty, Shivanand Venkanna Sheshappanavar
•Visual-first multimodal RAG audits civil standard plans without OCR.
•ColNomic-3B reaches 91.47% Recall@5 on a new 4,056-pair five-DOT benchmark.
•Zero-shot retrieval transfers to an unseen DOT at 91.40%; LoRA tuning adds nothing.
•Agentic pipeline: 100% verdicts with resolved thresholds; transfers to real plans.
•First visual rule-grounding: numeric limits extracted with no human-supplied rule.
arXiv:2608.26091v1  [cs.IR]  26 Aug 2026

PlanSightRAG: A Visual-First Multimodal RAG for Automating
Question Answering and Compliance Checking for Civil Standard Plans
Nabaraj Subedia, Shuvo Dip Dattab, Ahmed Abdelatyb,*and Shivanand
Venkanna Sheshappanavara
aDepartment of Electrical Engineering & Computer Science, University of Wyoming,
bDepartment of Civil & Architectural Engineering & Construction Management, University of Wyoming, 1000 E. University
Ave., Laramie, 82071, WY, USA
ARTICLE INFO
Keywords:
Multimodal RAG
Vision-Language Models
Engineering Drawing
Compliance Checking
Agentic AI
Visual Grounding
ExplainabilityABSTRACT
Civil infrastructure compliance checking has long relied on engineers manually reading
legacy 2D plans; however, OCR-based automation strips away the geometry and layout
essential for interpreting these plans. We present a Visual-First Multimodal Retrieval-
Augmented Generation (RAG) framework calledPlanSightRAG. It indexes and
reasons directly over plan imagery, integrates a ColNomic-3B multi-vector retrieval,
an agentic Planner–Retriever–Auditor–Synthesizer, and MaxSim heatmaps as an
evidence trail. We introduce a 4,056-pair benchmark from five state Departments of
Transportation (DOT) standard plans (1,898-pages). PlanSightRAG achieves91.47%
Recall@5 on zero-shot retrieval, while on a held-out Michigan DOT corpus, it achieves
91.40%. On synthetic, parametrically-generated compliance drawings, our Qwen2.5-
VL-72B pipeline reaches100%verdict accuracyonly when supplied a pre-resolved rule
threshold, a controlled ceiling that a non-VLM OCR baseline already reaches at 76.4%.
Finally, we demonstrate autonomous visual rule-grounding by extracting numeric limits
directly from a specification corpus without any human-supplied rules.
1. Introduction
Compliance checking underpins the safety and long-
term reliability of civil infrastructure, yet in most state
transportation agencies, it is still carried out by engi-
neers reading through dense 2D standard plans. The
plans are not obsolete—DOTs continue to maintain
thousands of legacy sheets that remain the authori-
tative reference for design review—and the workflow
is expensive, slow, and error-prone on multi-sheet sets
with complex spatial conventions [1]. A way to query
and audit these archives at scale while keeping humans
in the loop is missing.
RAG and Document Question Answering frame-
works have been natural candidates [2, 3, 4, 5], in
which documents are linearized via OCR into tokens
that an embedding model scores. On engineering plans,
these embeddings lack geometric properties, positional
layout,andcross-viewsymbols;hence,theylosethepri-
mary semantics of a standard plan. Once the standard
plans are reduced to a token bag, the retriever can no
longer locate the relevant plan reliably [6]. We therefore
treat this as the central gap and propose a multi-vector
retrieval framework in which the retrieval layer itself
sees the plan. Figure 1 makes this gap concrete on a
realistic WYDOT query.
*Corresponding author
nsubedi1@uwyo.edu(N. Subedi);sdatta1@uwyo.edu
(S.D. Datta);aahmed3@uwyo.edu(A. Abdelaty);
ssheshap@uwyo.edu(S.V. Sheshappanavar)
orcid(s):0009-0005-1596-8641(N. Subedi);
0000-0001-8604-3148(S.D. Datta);0000-0001-7329-452X(A.
Abdelaty);0000-0003-4039-2910(S.V. Sheshappanavar)Query:stake spacing & wire-tie config for
wire-enclosed riprap (WYDOT 511-1A)?
OCR+BM25 Text RAG×FAILS
WIRETYP
SPACING
FTSEERIPRAPToken bag⇒geometry lost
Recall@5: 58.02%
VisionRAG (Pyramid)×FAILS
blurredLow-res⇒fine details lost
Recall@5: 45.99%
ColNomic (Ours)✓SUCCEEDS
32×32Multi-vector MaxSim⇒grounded
Recall@5: 92.69%
Figure 1: Existing retrieval failure cases and patch-level
visual retrieval success on a realistic WYDOT query.
Two recent lines of work make a visual-first ap-
proach plausible. The Vision-centric retrieval model
ColPali [7] indexes a page as a grid of patch embed-
dings and scores queries via late-interaction MaxSim,
preserving layout without text conversion. In parallel,
vision-languagemodels(VLMs),includingQwen-VL[8]
and Gemini [9], have demonstrated fine-grained visual
grounding on diagrams. These systems are usually
evaluated in isolation; however, neither the retrieval
nor the reasoning layer has been integrated into a
pipeline capable of auditing multi-sheet engineering
archives end-to-end. As Figure 1 shows, a text-based
N. Subedi et al.:Preprint submitted to ElsevierPage 1 of 31

Visual-First Multimodal RAG for Civil Standard Plans
RAG pipeline and the hybrid VisionRAG baseline
both miss the dimension chain and wire-tie detail. On
the contrary, the patch-level visual indexing retrieves
the correct standard plan at Top-1 and grounds its
answer on the relevant region (Table 4 reports the full
quantitative comparison). In this paper, we close the
gap with a Visual-First Multimodal RAG framework
calledPlanSightRAGthat indexes plans directly as
images, reasons over the retrieved plans with a VLM,
and decomposes multi-plan compliance queries through
a dedicated agentic pipeline. The central contribution
is a single end-to-end, OCR-free system forautomated
compliance checkingof civil standard plans; visual
retrieval, MaxSim grounding, agentic auditing, and au-
tonomous rule-grounding are the system components,
and the five-DOT benchmark is the substrate on which
we validate it. Our contributions are threefold:
1.Visual-FirstRetrievalFrameworkandFive-
DOTBenchmark:Amulti-vectorlate-interaction
(ColNomic-3B) visual indexing pipeline with
high-resolutiontilingtoretrieveengineeringplans
without OCR.
2.Agentic Grounded Compliance Pipeline:A
Planner–Retriever–Auditor–Synthesizer pipeline
with sharpened MaxSim heatmaps that audit
designs against retrieved standards via a trans-
parent evidence trail.
3.Autonomous Visual Rule-Grounding:In-
stead of hand-injecting the rule, the agent re-
trieves the governing requirement from a speci-
fication corpus (Recall@5= 100%among 1,913
candidates), extracts its numeric limit (resolv-
ing symbolic𝑑/2), and audits against theself-
groundedthreshold, matching the 100% ceiling
with no human-supplied rule.
We ground the evaluation in four pre-declared,
falsifiable hypotheses (H1–H4) rather than qualita-
tive claims, so that each finding can be read as
supported, partially supported, or rejected against
an explicit numeric threshold declared with each
test.H1asks whether patch-level visual retrieval
surpasses OCR-based and hybrid alternatives.H2
asks whether the agentic Planner–Retriever–Auditor–
Synthesizer pipeline, armed with per-drawing rule
thresholds, audits dense multi-view sheets and multi-
plan designs.H3concerns domain adaptation: whether
LoRA fine-tuning of the adopted ColNomic-3B re-
trieveronthefive-DOTtrainingsplitimprovesheld-out
Recall@5 without degrading zero-shot transfer to the
unseen Michigan DOT corpus (Section 5.11). Finally,
H4tests whether high-resolution tiling improves judge
accuracy.2. Literature Review
2.1. Automated Compliance Checking in
Civil Infrastructure
Automated compliance checking has traditionally
relied on rule-based and text-driven approaches that
encode regulatory requirements as machine-executable
logic [10, 11]. These methods fundamentally assume
that compliance information can be represented as
structured data or linear text. However, for legacy 2D
standard plans, requirements are encoded implicitly
through geometry, layout, and cross-view symbols,
encouraging an error-prone manual visual inspection as
the dominant current practice [12].
2.2. OCR-Based Document QA and
Text-Centric RAG Systems
RAG [2] grounds language models in external
knowledge and has been deployed successfully for reg-
ulatory clause retrieval and general-purpose Document
Question Answering (QA) [6, 13, 14, 15, 16], where
rules and answers are explicitly stated in the text.
These successes are the reference point against which
visual-first retrieval must be compared; the failure
modes that motivate this paper have already been
discussed in Section 1. What is most relevant for
positioning is that layout-aware text encoders [13, 14]
and image-to-structure approaches [17, 18] recover part
of the spatial signal lost during OCR. However, they
do not fully restore this information. Recent diagnostic
evaluations further indicate that the remaining spatial
loss is sufficient to degrade engineering-compliance
tasks [1].
2.3. VLMs for Drawing Understanding
While early VLMs primarily targeted natural im-
ages and explicitly structured documents [19, 20],
recent advances have yielded state-of-the-art (SOTA)
transformer-based models capable of processing high-
resolution visual contexts. Some of the VLMs in-
clude robust open-source foundations such as Qwen-
VL [8], InternVL [21, 22], BLIP-2 [23], Flamingo [24],
and LLaVA [25], as well as closed-source alterna-
tives like Gemini [9] and GPT-4V [26, 27]. These
models achieve strong performance on general multi-
modal benchmarks; however, their application directly
to dense, domain-specific technical documents remains
underexplored.
In the civil engineering and construction domain,
several studies have explored VLM-based approaches
for technical document understanding [1, 28]. However,
multiple studies emphasize that generic VLMs face sig-
nificant limitations when applied to engineering draw-
ings[1,29].OCR-centricorregion-basedpipelinesoften
fail to capture implicit relationships encoded through
layout, geometry, symbols, and cross-view references,
which are central to technical drawings and standard
N. Subedi et al.:Preprint submitted to ElsevierPage 2 of 31

Visual-First Multimodal RAG for Civil Standard Plans
plans [28]. Chart and diagram understanding bench-
marks such as PlotQA [30], FigureQA [31], ChartQA
[32], and DVQA [33] further show the limits of current
VLMs. Their performance drops on tasks that require
precise spatial grounding, symbolic interpretation, or
multi-step visual reasoning, even when textual anno-
tations are available. Recent work on understanding
engineering drawings confirms that general-purpose
VLMs struggle with the visual conventions unique
to technical drafting. The HallusionBench diagnostic
suite [34] systematically demonstrates that VLMs are
prone to hallucinations when visual and textual cues
conflict, a critical concern for engineering compliance
applications.
Overall, existing literature suggests that while
VLMs provide a strong foundation for technical and di-
agram understanding, their effectiveness is constrained
by text-first preprocessing and limited spatial ground-
ing. Most applications discussed above remain focused
on single-image interpretation and do not support
scalable retrieval, evidence grounding, or compliance-
oriented reasoning across large engineering drawing
repositories. This motivates integrating vision-first
retrievalwithspatiallygroundedmultimodalreasoning.
2.4. Vision-First Document Retrieval and
Multimodal RAG
To overcome geometric loss, vision-first document
retrieval methods index content directly within the
visual embedding space. Models like ColPali [7] lever-
age multi-vector representations and late-interaction
paradigms—initially developed by ColBERT [35]—to
preserve spatial alignment globally without intermedi-
ate text extraction. These vision-first indexers confirm
that layout-aware representations significantly improve
retrieval on dense, visually complex documents [36].
While early Multimodal RAG pipelines [37] still
treated images merely as auxiliary grounding for tex-
tual retrieval, modern systems are beginning to capture
direct diagrammatic semantics. Nonetheless, existing
studies focus primarily on single-image understanding
and isolated QA tasks, failing to address the ro-
buststructurallinkingrequiredforcompliance-oriented
reasoning over large engineering archives. Addressing
these limitations, we introduce an integrated visual-
first RAG framework(PlanSightRAG). It pairs late-
interactionvisualindexingwithagenticVLMreasoning
tomeettherigorousdemandsofcross-sheetengineering
plan inspection.
3. Methodology
3.1. Visual-First Multimodal RAG
Framework Overview
We propose a Visual-First Multimodal RAG archi-
tecture for question answering and compliance check-
ing on civil engineering standard plans. As shownin Figure 2, the framework is organized into four
phases.Phase1(DocumentIngestion&Preprocessing
- section 3.2) rasterizes each PDF sheet for indexing.
We use 200 DPI for the deployed full-page index
used in all headline results, and 400 DPI for the
optional tiling path evaluated as an ablation. When
tiling is enabled, each sheet is split into overlapping
1024×1024tiles with a 256-px overlap. Finally, the
corpus is enriched with document-level metadata ex-
tracted by a vision–language model.Phase 2(Vi-
sual Indexing - section 3.3) encodes every tile with
ColNomic-3B into multi-vector late-interaction patch
embeddings at dynamic resolution and persists them
as a reusable visual vector store.Phase 3(Query
Processing, Retrieval & Visual Grounding - section
3.4) performs retrieval using MaxSim late-interaction
scoring. It can optionally apply VLM cross-encoder
re-ranking and BM25 hybrid fusion to refine the re-
sults. This phase also produces sharpened MaxSim
heatmaps, which highlight the specific plan regions
that drive each retrieval decision.Phase 4splits into
two downstream tasks: Phase 4a (Visual Question
Answering - section 3.5) generates grounded answers
with a multi-VLM pool of Qwen 2.5-VL-7B, Qwen 2.5-
VL-72B,andInternVL-2.5-8BovertheTop-𝐾retrieved
pages. Phase 4b (Automated Compliance Checking
section - 3.6) runs a prompted Qwen 2.5-VL-72B au-
ditor alongside an agentic Planner–Retriever–Auditor–
Synthesizer pipeline to produce structured, evidence-
grounded compliance reports. Qwen 2.5-VL-7B handles
high-resolution VQA; Qwen 2.5-VL-72B handles long-
context, multi-sheet compliance with enforced JSON
output; and the remaining models are evaluated as
alternative generators. The implementation details is
discussed in section 3.7 . A strict separation of concerns
among retrieval, reasoning, grounding, and auditing
reduces the risk of hallucinations and yields a trans-
parent, evidence-grounded decision path.
3.2. Document Ingestion and
Preprocessing
Standard Plans for Road and Bridge Construction
from five US state DOTs(Wyoming, California, Ari-
zona, Colorado and Florida) serve as the regulatory
ground truth in our work. Each PDF document is
rasterized page-by-page at 200 DPI, preserving line
weights, hatch patterns, dimension chains, symbols,
and annotations that are commonly lost in OCR.
Pages are treated as independent visual units; OCR
is intentionally excluded since the text-baseline results
in Section 5.1 (Table 4) confirm label fragmentation
in their geometric context(for example, the strongest
modern text retriever BGE-M3 + OCR plateaus at
36.79% Recall@5 against ColNomic’s 92.69% on the
424-pair test split). Preprocessing is limited to format
conversion and resolution standardization (no crop-
ping, segmentation, or manual annotation). Regarding
N. Subedi et al.:Preprint submitted to ElsevierPage 3 of 31

Visual-First Multimodal RAG for Civil Standard Plans
1○Document Ingestion & Preprocessing
PDF Corpus
5 DOTs: WY, CA, AZ, CO, FL
(1,898 pages)+MI held-out
Sample page: WYDOT 511-1ARasterization
200/400 DPI
(pdf2image/PyMuPDF)Sliding Window
1024×1024px
256 px overlapMetadata Enrichment
Gemini Flash
Strict JSONEnriched
Tiles + Meta
Stride768×768px|∼30 tiles/page|Grid(𝑖, 𝑗), BBox, PlanID, Date per tile
2○Visual Indexing (ColNomic)
ColNomic Encoder
Qwen2.5-VL backboneMulti-Vector Embeddings
Patch-level (32×32grid)Visual Index
Serialized.pt
(1,898 pages)batch=4, GPU
3○Query Processing, Retrieval & Visual Grounding
User Query
or Compliance
RequirementQuery Encoding
ColNomic tokens{𝑞 𝑖}MaxSim∑︀
𝑖max 𝑗𝑞𝑖·𝑝𝑗VLM Re-Ranker
Binary verdictBM25 Hybrid
𝛼^𝑠𝑑+(1−𝛼)^𝑠 𝑠opt.
Top-𝐾Evidence Pages
Sharpened MaxSim heatmap on WYDOT 511-1Avisual grounding
4a○Visual Question Answering
System: “You are a Senior WYDOT Engineer. You are looking
at{𝑁}standard plan sheets. Analyze these sheets to answer the
following query:{query}.”
Instructions:
(1)Cite which image (Image 1, Image 2, ...) contains the relevant
details.
(2)Extract specific values (dimensions, materials).
(3)If the images conflict, explain why (different view or context).
Input: Top-𝐾plan images + user query
Primary VLM
Qwen2.5-VL-7B (4-bit NF4)Alt. VLMs
Gemini / InternVL / Llama-3.2
Grounded Answer+ Evidence Gallery
VQA example evidence: hi-resolution retrieved
WYDOT 511-1A (Top-1from MaxSim, score=23.12)4b○Automated Compliance Auditing
ApproachA: Prompt-Based (Gemini2.5-Flash)
System: “You are a Senior Compliance Officer for WYDOT. Your task
is to audit a submitted Proposed Design against the provided Standard
Plans. Output strictly valid JSON.”
Schema:[{requirement_name, standard_source, standard_value,
design_value, verdict, citation_reason}]
Verdicts:pass/fail/unclearMIME:application/json
JSON Report: PASS / FAIL / UNCLEAR + structured reasoning
ApproachB: Agent-Based (Qwen2.5-VL-7B)
Planner: “Expert Civil Engineering Compliance Planner. Break the query
into𝑁logical verification steps; return JSON[{step, description,
search_query, category}].”
Auditor: “Analyze this engineering plan. Step Goal: {desc}. Original
Intent: {query}. Provide (1) Findings, (2) Verdict.”
Synthesizer: “Senior Civil Engineering Auditor. Aggregate findings into
(1) Summary of Evidence, (2) Compliance Verdict, (3) Engineer’s Note.”
Planner
𝑁sub-stepsRetriever
ColNomic
per-stepAuditor
Visual +
groundingSynthesizer
MD Reportiterate
Figure 2:Architecture of the Visual-First Multimodal RAG framework, organized into four phases: (1) document ingestion with
high-resolutiontilingandmetadataenrichment;(2)ColNomic-3Bvisualindexingwithmulti-vectorpatchembeddings;(3)MaxSim
late-interaction retrieval and grounding (optional VLM re-ranking and BM25 fusion); and (4) downstream tasks—(4a) multi-
VLM visual question answering with heatmap grounding and (4b) agentic compliance auditing via the Planner–Retriever–Auditor–
Synthesizer pipeline.
N. Subedi et al.:Preprint submitted to ElsevierPage 4 of 31

Visual-First Multimodal RAG for Civil Standard Plans
resolution choices, we used200 DPI full-pagefor
the visual index, all headline retrieval, and compliance-
pipeline retrieval, and the400 DPIsliding-window
tiling path is the only exception and is evaluated solely
as an ablation (Section 7). The combined index spans
1,898 pages: WYDOT (237), Caltrans 2025 (638), Ari-
zona DOT 2025 (181), Colorado DOT 2025 (62), and
Florida DOT 2026 (780), each deduplicated to a single
publication year to remove near-identical revisions. A
held-out Michigan DOT corpus (298 pages) is ingested
in the same way and used exclusively for zero-shot
cross-agency transfer evaluation.
Even with a higher-resolution backbone, full-page
indexing of very large multi-view sheets caps the
effective per-region resolution, downsampling high-DPI
engineering drawings until dimension text becomes
illegible, thin lines vanish, and hatching merges. We
therefore adopt a sliding-window tiling scheme (Fig-
ure 3) that preserves native-resolution detail by index-
ing at the tile level.
Tiling Procedure:Each page is rasterized at 400
DPI withPyMuPDFand decomposed into overlapping
tiles with a size1024×1024, an overlap of 256 px
(25%), and a stride of768×768, yielding⌈(𝑊−
256)/768⌉×⌈(𝐻−256)/768⌉tiles, where𝑊, 𝐻are the
full-resolution dimensions. A typical landscape sheet
(≈4400×3400) produces≈30 tiles.
Metadata and Retrieval:Each tile inherits its
document-level metadata (plan ID, revision date, cate-
gory, keywords) and adds tile-level fields: grid position
(𝑖, 𝑗), bounding box(𝑥 left, 𝑦top, 𝑥right, 𝑦bottom ), and a
reference to the full-page image. Tiles are encoded with
the same ColNomic retriever and scored individually,
enabling sub-page retrieval granularity and precise
spatial back-mapping for grounding.
3.3. Visual Indexing
Each page (and each tile) is encoded with a
pretrained ColNomic-3B retriever [38] (nomic-ai/
colnomic-embed-multimodal-3b),amulti-vectorlate-
interaction model on the Qwen2.5-VL backbone [8, 39].
Rather than compressing a page into a single vector,
it produces a multi-vector embedding over a dynamic-
resolution patch grid. This approach preserves spatial
locality, allowing tables, legends, callouts, and geo-
metric details to remain separately addressable. Such
granularity is critical for engineering drawings, where
compliance-relevantinformationisspatiallydistributed
and cannot be summarized holistically. Pages are
uniformly encoded in batches of four during inference.
Each index entry stores the CPU-resident embedding
tensor alongside a metadata record extracted once per
PDF by a vision-language model under strict JSON
output (with a filename-parsing fallback)—document-
level fields (plan ID, revision date, title, category,
keywords, source agency) and page-level fields (imagepath, page number, unique page ID). All 1,898 in-
distribution pages are serialized as a single joint index,
while the held-out Michigan DOT corpus (298 pages)
is encoded identically into a separate index used only
at zero-shot evaluation. The joint index supports cross-
agency retrieval without per-agency partitioning, and
decoupling indexing from downstream reasoning lets
the same embeddings be reused across retrieval, VQA,
and compliance without re-encoding.
3.4. Query Processing, Retrieval & Visual
Grounding
At query time, the ColNomic retriever embeds the
query into token-level vectors𝑄={𝑞 1, . . . , 𝑞 𝑚}; each
indexed page is represented by patch vectors𝑃=
{𝑝1, . . . , 𝑝 𝑛}. Retrieval uses a ColBERT-style [35, 40]
MaxSim late-interaction score computed in three steps:
𝑆=𝑄·𝑃⊤(1)
MaxSim(𝑞 𝑖, 𝑃) = max
𝑗=1,...,𝑛𝑞𝑖·𝑝𝑗 (2)
Score(𝑄, 𝑃) =𝑚∑︁
𝑖=1MaxSim(𝑞 𝑖, 𝑃)(3)
Each query token independently aligns with its
most relevant visual patch, which is essential when a
dimension value, a legend symbol, and a table entry
must all contribute to a page’s relevance—cues that
dense single-vector retrieval would average out. Pages
are ranked by score, and the Top-𝐾are returned as
visual evidence.
To optimize raw retrieval performance and adapt to
mixed visual-lexical queries, we evaluate two enhance-
ments layered on top of the base ColNomic retriever
without modifying the visual index.
VLM Cross-Encoder Re-Ranking:ColNomic
returns Top-𝐾candidates (Stage 1). VLM then re-
rankseachcandidatewithabinaryprompt—“Does this
engineering plan contain the specific data to answer the
query?”—retaining only positively verified candidates
(Stage 2). Images are processed sequentially to bound
GPU memory, with early stopping after two positive
verifications. This yields a more discriminative rele-
vance signal than bi-encoder similarity at the cost of
bounded additional latency (≤𝐾passes,≈0.3s each;
Appendix F).
Hybrid Dense–Sparse Fusion:To complement
visual similarity with lexical precision on alphanumeric
queries (plan IDs, note references), ColNomic dense
scores are fused with BM25 sparse scores over the
extracted metadata (plan ID, title, category, filename).
Both vectors are min-max normalized to[0,1]and
combined:
Score hybrid =𝛼·^𝑠 dense + (1−𝛼)·^𝑠 sparse (4)
with𝛼∈[0,1]controlling the visual/lexical balance.
N. Subedi et al.:Preprint submitted to ElsevierPage 5 of 31

Visual-First Multimodal RAG for Civil Standard Plans
1. Source PDF
WYDOT 511-1A at
400DPI (≈4400×3400px)
2. Sliding
Window (3D)
1024×1024tiles, 256px
overlap, stride=768PyMuPDF
3. Fine-Grained Tile
Single1024×1024
crop from the plan
4. Preserved
Annotations
+ Grid Position
Plan-ID, page#,
grid(𝑖, 𝑗), BBox per tilecrop
metadata
5. Dense Tile
Embedding
ColNomic: dynamic-
resolution patches,
multi-vector rep.ColNomic
+metadata
Parameters:Tile=1024×1024px|Overlap=256px|DPI=400|Stride=768×768px
Yield:∼30 tiles per landscape sheet|Metadata:PlanID, page#, grid(𝑖, 𝑗), BBox, revision date
Figure 3:Advanced Sliding Window Technique. Engineering sheets are rasterized at 400 DPI and decomposed into overlapping
1024×1024tiles with 256-pixel overlap, preserving fine-grained annotations and enabling sub-page retrieval granularity.
Beyondrawretrievalperformance,ColNomic’spatch-
level late interaction reveals which regions drove each
retrieval decision, making the architecture inherently
explainable (unlike opaque text-in/text-out RAG).
Sharpened MaxSim Heatmap:For each query
page pair, patch-level MaxSim scores are reshaped
into a 2D map aligned with the page layout and
sharpened in two stages. In the first stage, only the
top 5% of patches are retained (below-95th-percentile
scores are clamped to the minimum to suppress diffuse
activations), and in the second stage, normalized values
are raised to𝛾= 3.0to concentrate emphasis on the
strongest regions. The result is bicubic-upscaled to the
original resolution, colorized with JET, and blended at
𝛼= 0.4over the page.
Region-Level Grounding:High-activationpatches
are additionally scored against OCR-detected text
regions. The Top-𝐾with the highest mean activation
is returned as bounding-box evidence, pinpointing
specific annotations, dimension labels, notes, or graph-
ical elements behind the retrieval decision. The same
mechanism attaches to every VQA answer and to
each auditor step in the agentic pipeline, yielding a
persistent visual evidence trail.3.5. Visual Question Answering (VQA)
Pipeline
Given a query, the Top-𝐾plan pages returned
by Section 3.4 are passed to a vision–language model
(Qwen 2.5-VL [8]). The multimodal prompt consists
of (i) a strict system role (“You are an expert Civil
Engineer for WYDOT; answer only from the provided
Standard Plan images”), (ii) the Top-𝐾retrieved im-
ages as explicit visual inputs, (iii) lightweight text
anchors (plan ID, reference index), and (iv) the user
query:
^𝑦=VLM(︀
𝑥,{𝐼 1, 𝐼2, . . . , 𝐼 𝐾}, 𝐼user)︀
(5)
where𝑥is the query,{𝐼 𝑘}are the retrieved images, and
𝐼useris an optional user-supplied diagram. Decoding
uses a fixed token budget; the output pairs a natural-
language answer with an evidence gallery of supporting
plan sheets, allowing users to verify answers directly
against the source visuals.
3.6. Automated Compliance Checking
Workflow
For design-vs-standard auditing, a dedicated Qwen
2.5-VL-72B agent is configured as a WYDOT auditor.
Its multimodal prompt includes (i) Top-𝐾retrieved
standard-plan images as regulatory evidence, (ii) the
N. Subedi et al.:Preprint submitted to ElsevierPage 6 of 31

Visual-First Multimodal RAG for Civil Standard Plans
proposed design image, (iii) audit rules on dimensional
consistency, unit correctness, labeling, and geometric
logic, and (iv) an enforced JSON output schema. Each
audit item is classified underunit,dimension,label,
orlogic, reporting the applicable requirement, the
observed design condition, and a verdict in {pass,fail,
unclear,hallucination_suspected}. This yields
a structured, repeatable report that can be directly
exported to downstream review tools.
While this configuration is effective for localized
checks, single-pass RAG cannot handle compliance
queries that span multiple plans and regulatory rules.
We therefore introduce anAgentic Compliancepipeline
(Figure 4) with four specialized agents. While the
framework is model-agnostic and supports lightweight
backbones (such as Qwen 2.5-VL-7B for resource-
constrained local deployment), we scale the backbone
to Qwen 2.5-VL-72B for the primary high-accuracy
evaluations reported in Section 5.7:
1.Planner:Decomposes a high-level query (e.g.,
“Is the geotextile erosion control placement for
RC Culverts consistent with fill slope require-
ments?”) into𝑁structured JSON steps, eachspecifying a focused search query, the expected
plan category, and the information to verify.
2.Retriever:Executes MaxSim retrieval per step,
returning the most relevant plan page and its
metadata.
3.Auditor:Takes the step description, retrieves
theimage,andtheoriginalquery;performsvisual
analysis; and emits a structured finding with a
preliminary verdict.
4.Synthesizer:Aggregates step findings into a
consolidated report with a final verdict, a per-
step evidence summary, and an engineer’s note
flagging ambiguities.
EveryauditstepemitsasharpenedMaxSimheatmap
overlaid on the retrieved page. The pipeline out-
puts a machine-readable Markdown report interleaving
findings, verdicts, and grounding images (plan ID,
auditor findings, evidence paths) for direct ingestion
by engineering review workflows.
Sample Compliance Query:“Does the proposed horizontal wire stiffener connection detail (18-gauge ties
at 2-ft spacing) conform to WYDOT gabion wall standards, and is the 3:1 back-slope drainage acceptable?”
User
QueryPlanner
Qwen2.5-VL-72B
decompose→𝑁stepsRetriever
ColNomic MaxSim
per-step searchAuditor
Qwen2.5-VL-72B
visual analysisSynthesizer
Qwen2.5-VL-72B
aggregate findingsAudit
ReportLoop: Step𝑖= 1. . . 𝑁
visual
evidencenext step
Prompt:“You are an expert Civil Engineering Compliance Planner.
Given the high-level compliance query, break it down into𝑁logical
verification steps. Each step must identify (1) what information to
look for, (2) which standard-plan category it belongs to. Return
JSONonly.”
Sample Output: [{step:1, desc:“wire stiffener detail”,
search:“gabion wire tie”, category:“Retaining Wall”},
{step:2, desc:“back-slope drainage”, search:“3:1 slope
drain”, category:“Drainage”}]Search (Step1):“gabion wire tie”
Top-1:WYDOT 511-1A p.1
MaxSim score = 23.12
Search (Step2):“3:1 slope drain”
Top-1:WYDOT 203-2A p.1
MaxSim score = 19.87Prompt:“Analyze this engineering
plan. Step Goal: {description}. Original
Intent: {query}. Provide (1)Findings:
specific values/rules; (2)Verdict: does
it comply or provide enough info?”
Step1:“#9 wire ties @ 2-ft o.c. per
511-1A⇒Complies.”
Step2:“3:1 slope OK; toe drain
required per 203-2A.”Prompt:“Senior Civil Engineering
Auditor. Given per-step findings,
produce (1)Summary of Evidence,
(2)Compliance Verdict, (3)Engineer’s
Note.”
Output:Structured Markdown report
with per-plan citations, overall verdict,
and actionable engineer’s note.
Grounding Module
Sharpened MaxSim→Heatmap Overlay
Step1 evidence: WYDOT 511-1A
(gabion wire-stiffener detail)
Step2 evidence: WYDOT 203-2A
(back-slope / drainage)
Generated compliance
audit report (Markdown
+ citations)
Figure 4: Agentic Compliance Framework: Planner–Retriever–Auditor–Synthesizer pipeline for multi-step,
evidence-grounded compliance verification.
N. Subedi et al.:Preprint submitted to ElsevierPage 7 of 31

Visual-First Multimodal RAG for Civil Standard Plans
3.7. Implementation and Deployment
Details
The system runs on a single GPU to match
practical agency deployments. Indexing and retrieval
are executed offline; VQA and compliance inference
run at query time. Qwen 2.5-VL uses 4-bit BitsAnd-
Bytes quantization (NF4 weights, bfloat16 compute),
which reduces memory usage and allows several full-
resolution plan images to fit in a single prompt.
Models are loaded in inference mode only. The result-
ing efficiency–accuracy trade-off is managed through
strict, evidence-grounded prompting, bounded genera-
tion lengths, and a retrieval-first design, ensuring that
answers are anchored in retrieved evidence rather than
priors.
Deployment latency:On the single-GPU target,
indexing the1,898-page corpus is a one-time offline
cost (7.3min,4.34pages/s); at query time, ColNomic
retrieval over the full index is≈0.10s (p50). The
optional cross-encoder re-ranker adds≈0.3s per can-
didate, single-shot VQA is2–5s, and the agentic
compliance pipeline averages60.9s/query because the
Plannerexpandseachverdictinto∼8.6sequentialVLM
steps—the dominant cost and the target of the step-
pruning optimization in Section 11. Full timings are in
Appendix F (Tables A.8, A.7).
4. Evaluation Setup
4.1. Benchmark Dataset Construction
To rigorously evaluate the framework, we construct
(i) a 4,056-pair benchmark generated via an iterative
refine-loop QnA procedure across five state DOTs,
(ii) a held-out Michigan DOT zero-shot transfer set,
and (iii) a CAD-generated compliance test set used
to validate the agentic verdict pipeline against known
ground truth.
Five-DOT Visual Index (1,898 Pages):The
retrieval index aggregates standard plan PDFs from
five US state DOTs, deduplicated to a single year per
agency to remove near-identical revisions: WYDOT
(237 pages), Caltrans 2025 (638 pages), Arizona DOT
2025 (181 pages), Colorado DOT 2025 (62 pages),
and Florida DOT 2026 (780 pages), for a combined
1,898 pages. Each page is rendered at 200 DPI and
stored as a multi-vector record with metadata (agency,
plan ID, sheet title) extracted by a Qwen2.5-VL-7B
metadata extractor. During benchmarkconstruction
(the refine loop below), the curation index is built
with ColPali [7]; allreportedretrieval results re-embed
the identical pages with the adopted ColNomic-3B
backbone (Section 5.1).
Refine-Loop QnA Generation:Manually au-
thoring large QnA benchmarks for engineering plans
is cost-prohibitive—prior engineering plan QA sets aretypically limited to tens of manually verified pairs. We
replace manual authoring with an iterative procedure:
for each target page and each of four reasoning cat-
egories (Dimensional Accuracy, Visual Interpretation,
Logical Reasoning, Hallucination Rate), a drafter VLM
(Qwen2.5-VL-72B or Llama-3.2-90B-Vision in 4-bit
NF4) writes a question conditioned on the page image,
ColPali retrieves over the 1,898-page index, and if the
target page is outside top-5 the drafter rephrases with
a page-specific anchor (a plan ID, a labeled value, or
a quoted note phrase). Up to five rephrase attempts
are allowed per question (eight for the small 62-page
Colorado corpus, with a stricter two-anchor rephrase).
A separate Qwen-VL-7B verifier subsequently filters
out questions whose answers cannot be derived from
the page (yes/no rubric). The procedure yields4,056
verified question–answer pairsthat pass both the
refine-loop and the independent verifier. A further 93
verified pairs are generated in the same way over the
held-out Michigan DOT corpus for zero-shot transfer
evaluation.
Construct validity of the curation loop:Be-
cause the refine loop accepts or rephrases a question
according to whetherColPaliranks the target page
in the top-5, the benchmark could, in principle, favor
ColPali-retrievable phrasings and inflate any retriever
evaluated on it. Two facts rule this out as a ma-
terial confound. First,23.9%of the finalized pairs
(968/4,056;95/424on the page-disjoint test split) are
ColPalimisses—questions ColPali never retrieved even
after five rephrases—so the benchmark is not restricted
to ColPali-friendly items. Second, on exactly these
ColPali-miss test questions, the adopted ColNomic-3B
retrieverstillattains77.89%Recall@5(vs.96.66%on
the ColPali-hit subset), recovering the large majority
of ColPali’s own failures. A benchmark that merely en-
coded ColPali’s inductive biases could not be solved by
adifferentretrieveronColPali’sfailureset;theadopted
retriever’s advantage is therefore genuine rather than
an artifact of the curation retriever.
Manually-curated anchor set (78 WYDOT
pairs):Prior to the machine-generated benchmark,
we hand-authored and manually verified a set of78
WYDOT QnA pairs(29 Dimensional, 16 Visual, 17
Logical, 16 Hallucination) against the source standard
plans. Because these pairs carrynomachine-generation
provenance, they provide an independent check that
the framework’s headline behavior is not an artifact
of the generate-then-verify pipeline. On this human-
curated set—retrieving over the 237-page WYDOT-
onlysliceoftheindex—thesystemattains91.03%Re-
call@5(95% bootstrap CI[84.6,96.2]) and, with the
same local Qwen2.5-VL-7B answerer and Qwen2.5-VL-
72B judge used throughout,80.77%end-to-end judge
accuracy (CI[71.8,89.7]); the per-category breakdown
N. Subedi et al.:Preprint submitted to ElsevierPage 8 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table 1
Human-curated 78-pair WYDOT anchor set: per-category
Recall@5 and end-to-end judge accuracy with95%bootstrap
CIs(10,000resamples,seed42).Retrievalisoverthe237-page
WYDOT-only index; judge accuracy uses the local Qwen2.5-
VL-7B answerer with a Qwen2.5-VL-72B judge (no proprietary
API).
Category𝑁Recall@5 (%) Judge (%)
Dimensional 29 100.00 [100.0, 100.0] 79.31 [62.1, 93.1]
Visual 16 87.50 [68.8, 100.0] 68.75 [43.8, 87.5]
Logical 17 94.12 [82.4, 100.0] 94.12 [82.4, 100.0]
Hallucination 16 75.00 [50.0, 93.8] 81.25 [62.5, 100.0]
Overall 78 91.03 [84.6, 96.2] 80.77 [71.8, 89.7]
is reported in Table 1. The retrieval profile matches
the large machine-generated benchmark (Dimensional
strongest, Hallucination/Visual hardest), confirming
that the automatically-generated pairs reproduce the
difficulty structure of the human-authored ones.
The four reasoning categories are defined as follows:
•Dimensional Accuracy:Queries requiring pre-
cise extraction of numeric dimensions, spacing,
depths, gauges, and material quantities from
engineering annotations.
•Visual Interpretation:Queries testing recog-
nition of non-textual semantics, including geo-
metricconfigurations,layout-dependentmeaning,
symbol conventions, and spatial relationships.
•Logical Reasoning:Queries requiring multi-
stepdeductionbasedonretrievedvisualevidence,
such as reconciling notes, cross-referencing views,
or applying conditional specifications.
•Hallucination Rate:Adversarial queries where
the correct answer is “not specified,” testing
whethermodelsavoidassertingunsupportedfacts
when evidence is missing or ambiguous.
Hallucination-Rate category:The verifier ac-
cepts a pair when the page genuinely supports the
answer, which by construction retains predominantly
answerablelookups in this category; only a minority
of finalized pairs carry a literal “not specified” gold
answer. Forretrievalevaluation, this is immaterial—
everypairhasawell-definedgoldpage—sothecategory
should be read as a spec/material-lookup stress set
ratherthanapureabstentiontest. Abstentionbehavior
is instead measured directly on the CAD false-negative
study (Section 5.6). The per-agency and per-category
distribution of the 4,056-pair benchmark is reported in
Table 2.
CAD-Generated Compliance Test Set:To
evaluate the agentic verdict pipeline againstknownground truth (so that judge errors can be unambigu-
ously attributed to extraction, rule selection, or arith-
metic), we construct a parameterizedsyntheticdrawing
generator (matplotlib; schematic figures rather than
CAD/DWG output) that produces engineering-plan
drawings with title block, dimensioned section views,
material callouts, and notes. The generator covers five
archetypes (box culvert, guardrail post foundation,
beam rebar detail, drainage inlet, sign post founda-
tion) at three density tiers (single-view, multi-view
+ schedule table, and multi-component multi-plan),
with controlled parameter values. For non-compliant
instances, exactly one parameter violates a rule (e.g.,
cover= 1.0′′<2.0′′min, stirrup spacing= 18′′>
𝑑/2=15′′max). The full compliance ground-truth set
comprises(a)a 14-drawing pilot (𝑛=10single-doc
spanning the 5 archetypes+𝑛=4dense multi-view
withtheviolationembeddedinoneschedule-tablerow),
(b)a500-drawingsingle-doc scale set (5 archetypes
×100 instances, 250 compliant+250 non-compliant,
seven violation families), and(c)a100-drawing
multi-plan set requiring𝑁∈{2,3,4,5}separate stan-
dard plans for verdict, withnoplan IDs cited on the
drawings (pure visual inference). Section 5.7 reports
verdict accuracy on all three.
Dataset Standardization:All tracks are unified
intoasingleevaluationpipelinethatstandardizesinput
formats (question, ground truth, image file, category,
and applicable rule thresholds), enabling consistent
metric computation across all models and configura-
tions.
4.2. Evaluated Models
We evaluate vision–language models in two roles: as
draftersin the refine-loop QnA generation (Qwen2.5-
VL-72B and Llama-3.2-90B-Vision) and as thecompli-
ance judgein the agentic pipeline (Qwen2.5-VL-72B).
The ColNomic visual retrieval backend is held fixed
across all generator comparisons. Table 3 summarizes
the models and their roles.
All open-source models are deployed with 4-bit NF4
quantization via BitsAndBytes to enable single-GPU
inference. The evaluation pipeline is fully automated:
for each model, the system loads the ColNomic re-
triever, performs visual retrieval for each benchmark
question, generates an answer using the model under
test, and computes evaluation metrics.
4.3. Evaluation Metrics
Performance is reported using two complementary
KPIs:
•Recall@5 (Retrieval Hit):The percentage of
queries for which the correct standard-plan page
appears in the Top-5 retrieved results [41]. We
reportRecall@5ratherthanRecall@Kgenerically
because the end-to-end pipeline always passes the
N. Subedi et al.:Preprint submitted to ElsevierPage 9 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table 2
Distribution of the 4,056-pair five-DOT benchmark by agency and reasoning category. The held-out Michigan DOT zero-shot set
(93 pairs) is not included in this table.
Category WYDOT Caltrans AZDOT CDOT FDOT Total
Visual Interpretation 201 416 149 85 549 1,400
Dimensional Accuracy 170 287 144 87 327 1,015
Logical Reasoning 172 331 104 104 446 1,157
Hallucination Rate 15 192 17 2 258 484
Total 558 1,226 414 278 1,580 4,056
Table 3
Vision–language models used in the framework.
Model Type Params Tier
Qwen2.5-VL-72B-Instruct Open-source 72B Primary
Qwen2.5-VL-7B-Instruct Open-source 7B Primary
InternVL-2.5-8B Open-source 8B Primary
Top-5 candidates to the generator. This measures
the effectiveness of the visual retrieval layer inde-
pendent of the generator.
•Judge Accuracy:An LLM-as-judge score [42]
where a local Qwen2.5-VL-72B judge compares
the model’s predicted answer against the ground
truth and assigns a binary score (0 or 1). This
measures end-to-end correctness, including both
retrieval and reasoning.
5. Results
5.1. Comparison with Existing Solutions
To test H1, we pre-declare the acceptance threshold
as a≥15 pp Recall@5 margin over the strongest text-
based and hybrid baselines and compare patch-level
visual retrieval (ColPali) against a broad set of strong
modern retrieval baselines, including the 2025–2026 Vi-
DoRe state of the art: NVIDIA Nemotron-ColEmbed-
8B/4B [43] (Qwen3-VL backbones) and ColNomic-
3B/7B [38], along with ColQwen2.5-v0.2 and ColPali-
v1.3 [7], DSE-Qwen2-2B [44], VisRAG-Ret [45], and
BGE-M3 with OCR [46]—alongside seven legacy base-
lines (CLIP, LayoutLMv3, Nougat, Pix2Struct, UDOP,
OCR+MiniLM, VisionRAG-Pyramid) and a binary-
quantized HPC-ColPali variant. Every method is eval-
uated identically: the 424-pair page-disjoint test split
is queried against the full 1,898-page five-DOT visual
index. Table 4 reports Recall@5 for each method
alongside index-size and retrieval-latency figures; per-
agency and per-category bootstrap CIs are reported in
Appendix C.
On the choice of retriever:Several 2025–2026
retrieversoutperformzero-shotColPalionthisbenchma-
rk—-Nemotron-ColEmbed-8Breaches95.28%Recall@5and ColNomic-3B 92.69%, versus 76.89% for ColPali.
This reflects their newer, higher-resolution Qwen2.5/
Qwen3-VL backbones, whereas ColPali’s PaliGemma
encoder operates at a fixed448×448that coarsens
the fine dimensions and notes on large standard-
plan sheets.This is the same resolution limitation our
high-resolution tiling strategy targets (Section 7). The
framework is deliberately retriever-agnostic: the index-
ing, tiling, visual grounding, and agentic compliance
components are unchanged by the backbone, so any
of these retrievers can drop in directly, and the per-
agency gains transfer. Complementarily, a lightweight
Apache-2.0 visual reranker (MonoQwen2-VL) applied
to ColPali’s top-20 lifts Recall@5 from 76.89% to
85.14% (Table 5)—recovering most of the gap to the
stronger retrievers while keeping the stack fully open
and deployable. We thereforeadopt ColNomic-3B as
the retrieval backbone—the strongest openly licensed
visual retriever on our benchmark and fully deploy-
able—and report all main results with it; ColPali serves
as a companion backbone for our binary-quantization
study (HPC-ColPali,16×compression) and controlled
LoRA analysis. The NVIDIA Nemotron-ColEmbed
models reach even higher Recall@5 but are CC-BY-
NC (research-only). Crucially, H1 concerns the gap
totext-based and hybridpipelines, which every visual
retriever here clears by a wide margin. For clarity on
backbone attribution: allheadlineretrieval and the
rule-grounding and real-plan results use ColNomic-
3B, whereas the re-ranking (Table 5), high-resolution
tiling(Section7),HPCbinary-quantization,bootstrap-
CI (Appendix C), and grounding-pilot (Appendix G)
studies are companion analyses on the predecessor
ColPali backbone, and each is labeled accordingly.
Global vs. Local Vision:The comparison be-
tween CLIP and ColPali reveals a stark “granularity
gap.” CLIP, which compresses an entire document page
into a single global vector, achieved Recall@5 of only
1.89%,confirmingthatglobalembeddingsfailtoresolve
the fine-grained details of engineering schematics. In
contrast,patch-levelColPaliachieves76.89%Recall@5,
demonstrating that retaining spatial resolution is non-
negotiable for this domain.
N. Subedi et al.:Preprint submitted to ElsevierPage 10 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table 4
Retrieval comparison on the 424-pair page-disjoint test split over the 1,898-page five-DOT index. Strong (modern) baselines
on top, legacy/weak baselines below, and the predecessor ColPali backbone (zero-shot) at the bottom; the adopted retriever
is ColNomic-3B (92.69%, bolded). All numbers measured under an identical evaluation protocol; per-agency and per-category
bootstrap CIs are reported in Appendix C.
Method Modality Mechanism𝑁 idx Recall@5
Strong modern baselines (2024–2026)
Nemotron-ColEmbed-8B [43] Visual (Patch) Multi-vector late-int. 1898 95.28%
Nemotron-ColEmbed-4B [43] Visual (Patch) Multi-vector late-int. 1898 92.92%
ColNomic-7B [38] Visual (Patch) Multi-vector late-int. 1898 91.27%
ColQwen2.5-v0.2 [7] Visual (Patch) Multi-vector MaxSim 1898 87.26%
DSE-Qwen2-2B [44] Visual (Global) Single-vector dense 1898 22.17%
VisRAG-Ret [45] Visual (Global) Single-vector dense 1898 53.77%
BGE-M3 + OCR [46] Text (OCR) Dense Vector 1898 36.79%
ColNomic-3B (Ours, adopted) [38] Visual (Patch) Multi-vector late-int. 1898 92.69%
Legacy baselines
CLIP ViT-B/32 [20] Visual (Global) Global Embedding 1898 1.89%
LayoutLMv3 [47] Visual (Layout) Layout Embedding 1898 0.00%
Nougat-decode + MiniLM OCR-free (Decode) Decode + Dense 1898 0.47%
Pix2Struct-decode + MiniLM OCR-free (Decode) Decode + Dense 1898 7.31%
UDOP-decode + MiniLM OCR-free (Decode) Decode + Dense 1898 0.00%
OCR + MiniLM Text (OCR) Dense Vector 1898 25.24%
VisionRAG (Pyramid, RRF) Hybrid (Text) Structure-Aware RRF 1898 23.11%
ColPali (predecessor backbone) [7] Visual (Patch) Multi-vector MaxSim 1898 76.89%
HPC-ColPali (Binary Quantized) Visual (Patch) Multi-vector MaxSim 1898 65.09%
Table 5
Two-stage retrieve–rerank on the 424-pair test split. The pre-
decessor ColPali backbone retrieves the top-20; MonoQwen2-
VL-v0.1 (Apache-2.0 pointwise visual reranker) reorders them.
The top-20 recall is the reranker’s ceiling.
Configuration Recall@5 (%)
ColPali (Stage 1 only) 76.89
ColPali + MonoQwen2-VL rerank 85.14
Top-20 ceiling 86.79
Retrieval ceiling for text-centric baselines:
The strongest modern text retriever evaluated against
the 1,898-page index is BGE-M3 + OCR at 36.79%
Recall@5, the upper bound of what an OCR-then-
embed pipeline can deliver here, even after replacing
legacy baselines with 2024–2025 state-of-the-art en-
coders. The hybrid VisionRAG pyramid, which ex-
plicitly models document structure, raises this only
to 23.11%. ColNomic’s 92.69% Recall@5 sits 55.90 pp
above the strongest text retriever and 69.58 pp above
the strongest hybrid, indicating that the information
loss occurs at the OCR step itself rather than at the
embedder.
OCR-free document VLMs do not transfer:
Three OCR-free document VLMs used as page-to-text
encoders—Nougat [48], Pix2Struct [49], and UDOP[50], with text embedded by MiniLM, so only the page-
to-text step varies—also collapse on engineering draw-
ings (0.47%, 7.31%, and 0.00% Recall@5). Pretrained
on text-heavy corpora (papers, forms, screenshots),
they emit empty or layout-less output on dense plan
sheets, extending the granularity-gap argument to the
broader OCR-free family.
Efficiency vs. Accuracy:Thebinary-quantization
variant (HPC-ColPali) compresses the visual index by
16.0×(477.74 MB to 29.86 MB) and retains 65.09%
Recall@5—within 11.79 pp of full-precision ColPali.
H1 verdict:H1issupported.TheadoptedColNomic-
3Bretrieverreaches92.69%Recall@5,whichis55.90pp
above the strongest text retriever (BGE-M3 + OCR,
36.79%) and 69.58 pp above the strongest hybrid
baseline (VisionRAG (Pyramid, RRF), 23.11%). The
pre-declared 15-pp threshold is cleared on both axes.
5.2. Overall End-to-End Performance on
the Five-DOT Benchmark
Table 6 presents the zero-shot retrieval performance
of the adopted ColNomic-3B backbone on the full
4,056-pair five-DOT benchmark. This is thebaseline
retrieval configuration; the controlled LoRA ablation
in Section 5.11 indicates that it is also the strongest
deployment configuration among those we tested. End-
to-end judge accuracy is measured separately on the
N. Subedi et al.:Preprint submitted to ElsevierPage 11 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table 6
Zero-shot ColNomic-3B retrieval on the 4,056-pair five-DOT
benchmark (1,898-page index), used off-the-shelf with no
domain adaptation.
Agency𝑁Zero-shot Recall@5 (%)
WYDOT 55893.37
FDOT 1,580 91.27
Caltrans 1,226 91.92
AZDOT 414 89.13
CDOT 278 90.29
Overall 4,056 91.47
CAD-generated compliance test set (Section 5.7) be-
cause it has controlled ground-truth verdicts. Rank-
sensitive metrics (Recall@1, MRR, nDCG@10) on the
page-disjoint test split are reported in Appendix D
(Table A.3). For clarity, three Recall@5 figures recur
in this paper and each is reported against a distinct
evaluation set:91.47%on the full4,056-pairbench-
mark (this section, our headline zero-shot number),
92.69%on the424-pairpage-disjoint test split (the
controlled baseline comparison of Table 4 and the rank
metrics), and91.03%on the human-curated78-pair
WYDOT anchor set (Section 4.1, Figure 1); the held-
out Michigan transfer is91.40%over the joint index.
These are consistent measurements on different sets,
not revisions of a single result.
Zero-shot Recall@5 is 91.47% over the full 4,056-
pair benchmark—a deliberately demanding evaluation:
the benchmark is two orders of magnitude larger than
prior engineering-plan QA sets, spans five agencies
with heterogeneous drafting conventions, and is gener-
ated by an adversarial refine-loop that explicitly seeks
page-distinguishing anchors. Retrieval is strong and
balancedacross agencies (89–93%): WYDOT retrieves
best (93.37%), reflecting its sparse, high-contrast lay-
outs, and even CDOT—whose 62-page bridge book is
dominated by near-duplicate detail sheets that weaker
backbones conflate—reaches 90.29%, a marked im-
provement over the PaliGemma-class encoders. We
treat91.47%asthezero-shot baseline.Theper-category
breakdown is reported in Section 5.3.
5.3. Category-Wise Performance Analysis
Table 7 presents the zero-shot category-wise Re-
call@5 on the 4,056-pair five-DOT benchmark. All four
categories reach≥87% Recall@5, with Dimensional
Accuracy as the strongest (95.37%); Logical Reason-
ing, historically the hardest category for patch-level
retrievers (it asks the agent to infer a rule from a
note or table rather than match a page-unique numeric
anchor), remains the (mildly) weakest at 87.64% but is
now within 8 pp of the best—a substantial narrowing
relative to PaliGemma-class backbones.Table 7
Zero-shot category-wise Recall@5 on the 4,056-pair five-DOT
benchmark.Drafters:Qwen2.5-VL-72BforWYDOT,AZDOT,
CDOT; Llama-3.2-90B-Vision for Caltrans, FDOT. Verifier:
Qwen2.5-VL-7B (yes/no rubric).
Category𝑁Zero-shot Recall@5 (%)
Hallucination Rate 484 88.84
Dimensional Accuracy 1,01595.37
Visual Interpretation 1,400 92.71
Logical Reasoning 1,157 87.64
Overall 4,056 91.47
Table 8
End-to-end judge accuracy (%) for nine prompting and rea-
soning techniques on the 424-pair page-disjoint test split, with
Qwen2.5-VL-7B and Qwen2.5-VL-72B as the answerer. Re-
trieval, judge, and decoding are held fixed; only the answering-
stage prompting strategy varies.
Technique Qwen-7B Qwen-72B
Zero-shot78.3076.89
Chain-of-thought (CoT) 70.75 72.17
Self-consistency (5 samples) 69.58 69.10
Retrieval-augmented 62.50 70.75
Sliding-window tiling 64.39 72.17
OCR-hybrid 77.12 78.07
Question decomposition 75.71 81.60
Agentic (Planner–Auditor–Synth.) 75.71 75.00
Critic self-correction 77.3682.31
Union (any technique correct) 97.16
5.4. Prompting and Reasoning Technique
Ablation
Beyond the choice of VLM and the retrieval back-
bone, end-to-end judge accuracy depends onhowthe
VLM is prompted at the answering stage. We ablate
nine prompting and reasoning strategies on the page-
disjoint test split (424 pairs) with Qwen2.5-VL-7B and
Qwen2.5-VL-72B as the answerers, holding retrieval,
judge, and decoding fixed: zero-shot, chain-of-thought
(CoT), self-consistency (5 samples plus majority vote),
retrieval-augmented (top-3 distractor pages in the an-
swering prompt), sliding-window tiling, OCR-hybrid
(image plus extracted text), question decomposition,
the agentic Planner–Auditor–Synthesizer pipeline, and
critic self-correction. Each candidate answer is scored
against the ground-truth answer by anopen-source
Qwen2.5-VL-72Bjudge(binarycorrect/incorrect,same
rubric as Section 4.3), so the entire ablation is repro-
ducible without any proprietary judge API. Table 8
reports per-(technique, model) judge accuracy.
Three findings stand out. First, the best cell
across the 18-entry grid iscritic self-correction
with Qwen2.5-VL-72B at 82.31%, which surpasses
Qwen-72B zero-shot by 5.42 pp and Qwen-7B zero-
shot by 4.01 pp; the same critic loop on the 7B model
yields only a 0.94-pp drop relative to 7B zero-shot,
N. Subedi et al.:Preprint submitted to ElsevierPage 12 of 31

Visual-First Multimodal RAG for Civil Standard Plans
indicating that self-correction is a near-zero-cost option
at the smaller scale and a meaningful gain at the
larger scale. Second, retrieval-augmented prompting is
theworsttechnique on both models (62.50% on 7B,
70.75% on 72B): supplying additional distractor pages
in the answering prompt degrades the VLM’s attention
to the correctly-retrieved primary page, confirming
that the answering stage benefits from a tight visual
context rather than an expanded one. Third, clas-
sical text-side reasoning aids (CoT, self-consistency)
underperform zero-shot on both models—the chain-
of-thought tends to drift away from the dimension
actually visible in the drawing, and majority voting
over five samples amplifies rather than corrects shared
visual misreadings. The union of all nine techniques
reaches 97.16% accuracy, indicating that the residual
error budget is largelyprompting-recoverablerather
than fundamentally limited by the VLM’s perception.
5.5. Visual Grounding and Explainability
Results
To evaluate visual grounding, we generate sharp-
ened MaxSim heatmaps across benchmark queries.
Figure 5 demonstrates this on a representative query:
the heatmap correctly bounds both the regulatory note
and its associated visual representation, preserving the
spatial relationships lost in OCR.
Figure 5: Visual Grounding Analysis: The sharpened
MaxSim heatmap overlay (ColNomic-3B) correctly
highlights both the textual ‘Grading Notes’ and the
corresponding geometric detail.
Grounding Evaluation:In a 16-query qualitative
pilot across four plans, thresholding the heatmap at
the top 5% of activations consistently localizes the true
evidence (specific dimensions, note blocks, structural
symbols), and passing these high-intensity crops to
the generator improves response specificity. We treat
grounding as aqualitative diagnostic: a quantitative
IoU study with human-annotated evidence regions isscoped as a follow-up rather than claimed here (Ap-
pendix G). Figure 6 illustrates this grounding inte-
grated into the agentic compliance pipeline, in which
each audit step produces its own heatmap overlay,
forming a verifiable evidence trail.
5.6. Agentic Compliance Results
To test H2, we pre-declare three thresholds:≥90%
verdictaccuracyontheCAD-generatedsingle-doccom-
pliance set (Section 5.7),≥90% false-positive avoid-
ance on a PASS set, and≥60% sensitivity on a
FAIL set. We evaluate the Planner–Retriever–Auditor–
Synthesizer pipeline on two complementary corpora: an
8-query cross-plan PASS set (to measure false-positive
avoidance) and a 24-query FAIL set with synthetic
mutations (to measure sensitivity, Appendix E.1).
On the 8-query PASS set, the agentic pipeline
reaches100%verdictaccuracy.ThePlannersafelyover-
decomposes queries (averaging 8.6 steps per query),
while per-step retrieval, audit findings, and final syn-
thesis maintain 100% correctness. For example, when
checking geotextile consistency between Plans 511-
1A and 203-2A, the agent successfully retrieved both
sheets, recognized that 203-2A lacks explicit RC Cul-
vert details, and appropriately flagged the ambiguity
for engineering review rather than hallucinating a pass.
H2 verdict:Supported. The 8-query PASS set is
small and measures only false-positive avoidance (it
contains no FAIL cases); its 100% accuracy clears
the pre-declared 90% false-positive threshold, while
the pre-declared≥90% verdict accuracy on the CAD-
generated set is met (Section 5.7). On the 24-query
FAIL set, overall True Positive Rate is 79.17% (87.5%
for dimension/note mutations, 62.5% for symbol sub-
stitutions), satisfying the pre-declared 60% sensitivity
threshold. Symbol-substitution is the weakest class and
is flagged for future work.
5.7. Compliance Validation on a
CAD-Generated Test Set
The agentic compliance results in Section 5.6 use
real WYDOT plan content, where the “true” verdict
is determined by consistency across multiple sheets. To
isolatewherea judge VLM fails on numeric compliance
(e.g., is it the value extraction, the rule selection, or
the threshold arithmetic?), we additionally evaluate on
a CAD-generated compliance test set with controlled
ground truth. The generator is described in Section 4.1;
it produces engineering-plan drawings with a title
block,dimensionedsectionviews,materialcallouts,and
notes, and for non-compliant instances, injects exactly
one parameter violation.
A 100% verdict-accuracy claim on a single judge
configuration is unconvincing without a credible failure
mode. We therefore lead with an ablation across five
judge configurations on a controlled𝑛=10validation
set (Table 9) and show that onlytwoof the five reach
N. Subedi et al.:Preprint submitted to ElsevierPage 13 of 31

Visual-First Multimodal RAG for Civil Standard Plans
(a) Step 1: Geotextile placement details
 (b) Step 2: Fill slope requirements
Figure 6:Visual evidence trail from the agentic compliance pipeline. Each audit step generates a sharpened MaxSim heatmap
overlay on the retrieved standard plan page, creating a transparent chain of evidence.
Table 9
Verdict accuracy of five judge configurations on the𝑛=10single-doc CAD set; the final block reports the best single-judge
configuration on𝑛=4dense schedule-table drawings. Interpretation is in the text below.
Judge Configuration Compliant Non-compliant Overall
Qwen2.5-VL-7B (plain) 5/5 1/5 60%
Qwen2.5-VL-7B (CoT) 0/5 5/5 50%
Qwen2.5-VL-72B (plain) 5/5 3/5 80%
Qwen2.5-VL-72B (CoT + thresholds) 5/5 5/5 100%
Multi-agent (Plan–Audit–Synth, 72B + thresholds) 5/5 5/5 100%
Dense multi-view drawings (𝑛=4, best configuration):
Qwen2.5-VL-72B (CoT + thresholds) 2/2 2/2100%
100%—the other three sit at 50–80%, with diagnosable
failure modes that motivate the prompting recipe used
at scale. The five-configuration ablation is the headline
result; the scale-up numbers (Tables 10, 11) are evi-
dence that the recipe generalizes, not that compliance
verification is generically a solved problem.
Five judge configurations.We evaluate the
verdict accuracy of five judge configurations on the
𝑛=10single-doc CAD set: (a) Qwen2.5-VL-7B with
a plain JSON verdict prompt, (b) Qwen2.5-VL-7B
with chain-of-thought (CoT) reasoning, (c) Qwen2.5-
VL-72B (4-bit NF4) with the same plain prompt,
(d) Qwen2.5-VL-72B with CoT plus per-drawing pre-
resolved rule thresholds injected into the prompt (e.g.,
for a 30” beam,𝑠 max=𝑑/2=15”), and (e) the same
72B-CoT-thresholds judge wrapped inside the multi-
agent Planner–Auditor–Synthesizer pipeline. The same
configurations are evaluated on the𝑛=4dense multi-
view drawings, where the violation is embedded in one
row of a schedule table.
Failure-mode analysis:The 7B-class judge fails
in two distinct directions depending on the prompt.
Without CoT, it defaults toCOMPLIANTon all inputs
(high specificity, near-zero sensitivity). With CoT, itflips to flageverythingasNON_COMPLIANT; inspecting
the chain reveals that the 7B model selects visually
salient but irrelevant values and applies them to in-
correct rule checks (e.g., it compares a4′′anchor-
bolt embedment value against a30′′guardrail-footing-
depthrule).The72B-classjudgeeliminatestherandom
matching, but plain prompting still misses two non-
compliant cases:rebar_bad, where the rule’s threshold
is arithmetic-derived (𝑠 max=𝑑/2for a 30” beam), and
signpost_bad, where the violated value (a4′′anchor
embed) is visually less salient than a satisfied 42” foot-
ing depth on the same drawing. Both failure modes
are eliminated when the prompt supplies the pre-
resolved threshold (15” instead of𝑑/2) and explicitly
enumerates which value must be checked.
Multi-agent variant:ThePlanner–Retriever–Auditor–
Synthesizer pipeline reaches 100% in Table 9 by routing
the same 72B + thresholds judge through three lighter
agent turns rather than a single CoT pass. The Planner
emits 1–2 audit steps per drawing, and the Synthesizer
produces a structured JSON verdict for downstream
parsing. The Auditor sees only the design image (not
the retrieved standard image) to avoid out-of-memory
pressure when running 4-bit 72B on 2×A30 (48 GB
total); the retrieved standard’s metadata is supplied as
N. Subedi et al.:Preprint submitted to ElsevierPage 14 of 31

Visual-First Multimodal RAG for Civil Standard Plans
text context. This single-image variant is recommended
for deployment on GPUs with under 80 GB of memory.
OnasingleNVIDIAA10080GB(orRTXA6000)GPU,
the multi-agent pipeline executing an average of 8.6
sequential VLM steps takes≈60.9s per query (an
average of≈7.1s per step), yielding a throughput
of approximately 59 audits per hour. The planner
produces more steps than necessary, averaging 8.6
steps compared with the minimum of 2, representing a
latency-vs-accuracy trade-off: it ensures comprehensive
visual auditing across all standard plan components
at the expense of sequential VLM passes. That both
the single-judge and multi-agent routes reach 100%
with the same threshold-injection prompt isolates the
recipe as the cause—it is not an artifact of the agentic
decomposition.
Multi-plan compliance (𝑁∈{2,3,4,5}):A𝑛=10
pilot of multi-plan compliance drawings was con-
structed with𝑁∈{2,3,4,5}separate standard plans
applicable per design (4 drawings at𝑁=2, 2 each at
𝑁=3,4,5; 5 compliant + 5 non-compliant). Crucially,
no plan IDs are cited on the drawing; the Planner
must infer applicable standards from the visual con-
tent of each labeled component. The agentic pipeline
(Qwen2.5-VL-72B 4-bit backbone) achieves9/10 =
90%verdictaccuracyonthispilot.Thesinglemissis
onn5_pier_ok, where the rule for column stirrup spac-
ing is the arithmetic-derivedd/2(max spacing equals
half the effective beam depth), and the prompt initially
presentedd/2as the textual threshold rather than
the pre-resolved numeric value—the same arithmetic-
derived-threshold failure mode characterized for the
single-doc path A judge above.
Multi-plan compliance scale-up (𝑛=100):We
scalethepilotto𝑛=100multi-plandrawings(25per𝑁-
level, 48 compliant + 52 non-compliant) with parame-
ter randomization for visual diversity. A first scale-up
run with the original textuald/2threshold confirms
the pilot’s failure mode at scale: 88/100 = 88% verdict
accuracy, with all 12 misses concentrated in the𝑁=5
pier subset (13/25 at𝑁=5; perfect 25/25 at𝑁=2,3,4).
All 52 non-compliant subjects across all𝑁are correctly
classified (52/52). The 12 missed compliant pier draw-
ings are falsely flaggedNON_COMPLIANTbecause the
auditor’s reasoning at the stirrup-spacing step treats
the textuald/2as a literal symbolic threshold rather
than resolving it. Replacing the registry threshold with
the per-drawing pre-resolved numeric value (𝑠 max=12′′
for the standardized 24” column) transfers the path
A fix to the multi-plan pipeline and yields100/100
= 100% verdict accuracy(48/48 compliant +
52/52 non-compliant, 25/25 atevery𝑁∈{2,3,4,5}),
confirming that the verdict accuracy is invariant to𝑁
once arithmetic-derived thresholds are pre-resolved.Table 10
Multi-plan compliance verdict accuracy on the𝑛=100scale
set, before and after replacing the textuald/2threshold in the
rules registry with a per-drawing pre-resolved numeric value.
𝑁plans Befored/2fix Afterd/2fix
𝑁=225/25 25/25
𝑁=325/25 25/25
𝑁=425/25 25/25
𝑁=513/2525/25
Total88/100 (88%)100/100 (100%)
Single-doc compliance scale-up (𝑛=500):The
single-doc path A recipe (Qwen2.5-VL-72B 4-bit +
CoT + per-drawing pre-resolved thresholds) scales to
a 500-drawing test set (5 archetypes×100 instances
each, 250 compliant + 250 non-compliant). Each non-
compliant instance violates exactly one rule sampled
from seven violation families (concrete cover, foot-
ing depth, stirrup spacing, anchor bolt embedment,
drainage grate opening, rebar grade, material class).
Parameter randomization yields visually distinct in-
stances for each archetype (span, rise, wall thickness,
post height, etc.). The pipeline achieves500/500
= 100% verdict accuracy(250/250compliant+
250/250non-compliant) without per-archetype tuning
or additional supervision.
Adversarial near-threshold stress test (𝑛=50):
A perfect score on scale500 is suggestive but does
not establish discriminative tightness: scale500’s non-
compliant values miss the rule by 50%+ (e.g., cover of
1.0′′againsta2.0′′minimum).Wethereforeaddastress
set in which violations sit within 5–10% of the rule
cutoff and require the judge to read the dimensioned
value precisely. The set is 50 drawings (5 archetypes
×10, 25 compliant+25 non-compliant): culverts with
cover1.875′′vs.2.0′′min, guardrails with footing29′′
vs.30′′min, rebar with stirrup spacing exactly𝑑/2+1′′
over the limit, inlets with grate openings4.125′′vs.
4.0′′max, and signposts with anchor embed11.5′′vs.
12.0′′min. The same Qwen2.5-VL-72B + CoT + per-
drawing thresholds recipe reaches50/50 = 100%
verdict accuracy(25/25compliant+ 25/25non-
compliant,withnoarchetypefallingbelow10/10).This
confirms that the recipe discriminates at single-inch
resolution—the regime where prompt-level mistakes
would manifest—and that the scale500 perfection is
not an artifact of large compliance margins.
Is the synthetic verdict task simply OCR? A
non-VLM baseline:A perfect VLM score raises
the question of how much of the task is genuine
visual reasoning versus reading a printed dimension.
We therefore evaluate a non-VLMOCR+threshold-
comparebaseline on scale500: for each drawing we
OCR the page (multi-orientationtesseract), extract
N. Subedi et al.:Preprint submitted to ElsevierPage 15 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table 11
Verdict accuracy of the best judge recipe (Qwen2.5-VL-72B 4-bit + CoT + per-drawing pre-resolved thresholds) across six
CAD-generated compliance test sets (674 drawings spanning small pilots, large-scale runs, dense multi-view schedules, multi-plan
cross-document designs, and an adversarial near-threshold stress test).
Set Compliant Non-compliant Overall
𝑛=10single-doc (pilot) 5/5 5/5 100%
𝑛=4dense multi-view (pilot) 2/2 2/2 100%
𝑛=10multi-plan (pilot) 4/5 5/5 90%
𝑛=500single-doc (scale) 250/250 250/250 100%
𝑛=100multi-plan (scale, after𝑑/2fix) 48/48 52/52 100%
𝑛=50adversarial near-threshold (stress) 25/25 25/25 100%
Total across 6 sets 334/335 339/339 673/674 (99.85%)
Table 12
Non-VLM OCR+threshold-compare baseline on scale500, per
archetype.
Archetype Gov. dim. (layout) OCR found OCR acc VLM acc
Culvert cover, horiz. 100.0 100.0 100.0
Inlet grate, horiz. 98.0 99.0 100.0
Guardrail footing, rotated 32.0 82.0 100.0
Sign post anchor, rotated 3.0 51.0 100.0
Rebar stirrup, distractor 0.0 50.0 100.0
Overall 46.6 76.4 100.0
the governing dimension by its label, and compare it
to thesamepre-resolved threshold the VLM receives.
This pipeline reaches76.4% verdict accuracy, but
the per-archetype breakdown is the informative part:
it scores100%where the governing value is a clean
horizontal leader-line label (culvert cover, inlet grate
opening) and collapses to≈50%where the dimension
is rendered asrotated vertical text(guardrail footing
depth, sign-post anchor embedment) or sits among
distractor values (rebar stirrup spacing). The synthetic
task is therefore OCR-trivial wherever the value is
cleanly presented; the VLM’s marginal contribution
isrobust value localizationon rotated and distractor-
dense layouts—precisely the dense multi-view regime
real plan sheets present—rather than digit reading per
se. This bounds the compliance claim honestly: the
100%of Table 11 is aceiling given a resolved rule and a
localizable value, and the agentic reader earns its keep
on layout robustness, not arithmetic (Table 12).
5.8. Autonomous Rule-Grounding:
Retrieving the Governing Threshold
The 100% verdict accuracy in Section 5.7 is ob-
tained when the auditor ishandedthe pre-resolved
numeric threshold. This is the standard assumption
in LLM-based compliance work, which presupposes a
digitized rule base or a human-curated registry, and
it leaves the hardest part of real compliance—knowing
whichrule governs a component andwhatits numeric
limit is—outside the system. We therefore test whetherthe framework can close this loopvisually: infer the
component, retrieve the governing requirement from
a standard-specifications corpus, extract the numeric
threshold from the retrieved sheet, and audit against
it, with no injected threshold.
Setup:We render a 15-sheet Standard-Specifications
reference corpus (5 agencies×3 sections: Structural
Concrete & Reinforcement, Foundations & Anchorage,
Drainage Structures), each sheet being a requirement
table of 7–9 rows in which the seven governing limits
for our archetypes (e.g., min. cover2.0′′, min. guardrail
footing30′′, max. grate opening4.0′′, min. anchor em-
bedment12′′, max. stirrup spacing𝑑/2) are embedded
among distractor rows. For each of the 500 single-
doc compliance drawings, the agent (i) forms a rule-
retrieval query from the visually-inferred component,
(ii) retrieves using ColNomic-3B, (iii) a VLM extracts
the limit value from the top-ranked sheet, (iv) resolves
symbolic limits (stirrup𝑑/2) using a dimension read
from the design, and (v) audits the design value against
theself-groundedthreshold. To make retrieval realistic,
we index the spec sheetsamong the full 1,898-page
plan corpus(1,913 candidates), so the rule query must
surface the governing requirement against real plan-
sheet distractors.
Result:ColNomic retrieves a correct-section specifi-
cation sheet at80.0% Recall@1and100.0% Re-
call@5among 1,913 candidates, the VLM extracts
the correct numeric limit (including resolving symbolic
𝑑/2)at100%, andtheend-to-endverdictaccuracywith
self-grounded thresholds reaches100%—matching the
injected-threshold ceiling (Table 13). The20%top-1
misses are real plan pages that outrank the spec sheet
at rank 1, but a correct-section spec is always within
the top 5, so verdicts are unaffected. Removing the
human-supplied threshold therefore costs nothing on
this set: the system sources the governing rule on its
own. This is, to our knowledge, the first demonstration
ofvisualrule-groundingforcompliance—retrievingand
N. Subedi et al.:Preprint submitted to ElsevierPage 16 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table 13
Autonomous rule-grounding on the 500 single-doc compliance
drawings: the agent retrieves the governing specification sheet,
extracts its numeric limit, and audits the design against the
self-groundedthreshold (no injected rule).
Retrieval pool Cand. R@1 R@5 Extract Verdict
Spec only 15 100.0 100.0 100.0 100.0
+1,898 plan pages 1,913 80.0 100.0 100.0100.0
Injected-threshold ref. (human rule) 100.0
applying a numeric standard from rasterized specifica-
tionsheets(seeAppendixH,FigureA.3)withoutOCR,
a BIM model, or a hand-coded rule base.
Scope:The spec corpus here is clean and modest;
the contribution is theclosed-loop feasibilityof visual
rule-grounding, not a complete code-coverage system.
Scaling the reference corpus to full agency specification
books (hundreds of pages of dense, real code text) is
the decisive stress test; we take exactly this step in
Section5.9,indexingareal931-pageWYDOTstandard
corpus, where autonomous extraction falls from100%
to33%and the residual error concentrates in retrieving
the governing sentence and resolving exhibit tables.
5.9. External Validity: Real WYDOT
Project Plans
Thebenchmarksaboveusegenerateddrawingswith
controlledgroundtruth.Totestwhethertheframework
transfers torealsubmittals, we evaluate it on two pro-
duction WYDOT projects—the Chief Joseph Highway
reconstruction (#1507040) and project #N345107—
comprising cross-section, earthwork, and PS&E quan-
tity sheets. Crucially,the project plans are not added
to the retrieval index: they are theproposed design
being audited, while the Retriever continues to query
the unchanged 1,898-page standard-plan index. Each
project PDF is simply rasterized to page images; no
re-indexing is required.
Retrieval transfers:For 12 real design items read
from the two projects (cross slope, fill/cut slope ra-
tios, CMP/RCP/RCP-arch/structural-plate culverts,
geotextile, geogrid, fence), a VLM relevance judge
confirms that ColNomic retrieves a topically-governing
WYDOT standard within the top 5 for83.3%of items
(10/12). The two misses are niche items (a proprietary
fence type and biaxial geogrid) without a dedicated
standard sheet in the index. Retrieval thus generalizes
from synthetic queries to real design language.
Auditing transfers under the same recipe:
On a controlled compliant/non-compliant set built by
overlaying proposed-design value callouts onto the real
cross-section sheets, we evaluated 6 compliant and 6
violatingexamplesacrossbothprojects.Usingthesamejudge recipe that reaches 100% on the synthetic set—
Qwen2.5-VL-72B with CoT and the governing rule
supplied—the model reads the proposed value off the
real sheet and reaches100% verdict accuracy, cor-
rectlyjudgingeventherotatedslope-ratiocallouts.The
auditor’s read-and-judge capability, therefore, carries
over from generated to real plan imagery, as the final
verdict is entirely the model’s own decision rather than
a hard-coded comparison.
Autonomous rule-grounding on a real 931-
page standard corpus:We measured how effec-
tively the autonomous loop transfers from curated
specification sheets to production standards. To do
this, we ingested a real 931-page standard corpus. This
included the full WYDOT 2021Standard Specifications
for Road and Bridge Construction(771 pp) alongside
the governing WYDOT Road Design Manual chapters
(Cross Sectional Elements, Earthwork Design, Culvert
Design, and Typical Sections; 160 pp). We indexed
this document using ColNomic-3B. Next, we processed
12 real design items. For each item, we retrieved the
governing page and extracted the numeric limit using
Qwen2.5-VL-72B. We then audited the design value
against thisself-groundedthreshold without injecting
any rules. Nine of these items contained numeric limits.
The system extracted the correct governing value for3
out of 9 (33%). It returned the correct verdict for44%
of those items. The system succeeded when the limit
was stated as a clear value. For example, it correctly
read a cross slope of “0.02ft/ft” (= 2%) and a foreslope
of “1V:4H” (= 4:1). However, it failed when limits
were located in exhibit lookup tables or split across
multiple pages (e.g., culvert minimum size and cover).
In these cases, the visual retriever ranked the governing
sentences below other visually similar tables. As a
result, the necessary information was never surfaced.
Therefore, the primary failure point isretrieving the
governing sentence and resolving exhibit tables. The
issue is not threshold arithmetic. Autonomous rule
extraction thus dropped from100%on the clean speci-
fication corpus (Section 5.8) to33%on real production
standards.Wereportthisastheprincipalopenproblem
exposed by this external-validity test. Resolving this
will be the priority for future work (Section 11).
Consequently, real-plan auditing still benefits from a
supplied threshold to ensure reliability.
5.10. Cross-Agency Generalization and
Scaling
The 4,056-pair benchmark inherently evaluates
cross-agency scaling: retrieval stays high and balanced
across all five agencies (89–93% Recall@5; Section 5.2),
so the late-interaction embeddings retain discrimina-
tive power despite diverse drafting conventions.
Zero-shot transfer to an unseen agency:
Zero-shot ColNomic-3B retrieves a held-out Michigan
N. Subedi et al.:Preprint submitted to ElsevierPage 17 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table 14
Synthetic vs. real-plan compliance. The framework transfers
from generated drawings to two production WYDOT projects
for retrieval and supplied-threshold auditing; autonomous
threshold extraction from dense real standards is the remaining
gap.
Capability Synthetic Real plans
Standard retrieval (rel.@5) — 83.3%
Verdict acc. (supplied rule) 100%100%
Threshold extraction (auto.) 100% unreliable
DOT corpus (298 pages, 93 verified pairs) at91.40%
Recall@5over the combined1,898+298-page index—
essentially matching in-distribution performance (over-
all 91.47%) on an agency never seen during indexing,
drafting, or training (the 298-page Michigan-only pool
yields a higher 93.55%, but that smaller candidate set
is not directly comparable to the joint-index agency
numbers; Appendix D). This demonstrates strong zero-
shot generalization to unseen drafting conventions, en-
abling immediate deployment for new agencies without
adaptation.
H3 verdict:H3 isrejected. Across three LoRA con-
figurations, domain adaptation on the 3,211-pair in-
domain training set does not improve Recall@5 on the
5-DOT test split and significantly degrades unseen-
agency transfer (Table 15). Full-LM LoRA catastrophi-
callyreducesretrievalperformance(e.g.,in-distribution
test drops by37.03pp). The off-the-shelf ColNomic-3B
checkpoint therefore remains the optimal deployment
configuration, as its pretrained representations are
already robust for engineering drawings.
5.11. Domain-Adaptive Retrieval: LoRA
Fine-Tuning of the Adopted Retriever
Totestdomainadaptation,wefine-tunedColNomic-
3B using LoRA on a page-disjoint 3,211-pair training
split. Configurations includedhead-only LoRA(tuning
only the projection head) andfull-LM LoRA(tuning
all transformer layer projections).
As shown in Table 15, no configuration outper-
formed the zero-shot baseline. Thehead-LoRAvariants
yielded negligible differences (<0.5pp changes), while
full-LM LoRAseverely degraded both in-distribution
retrieval (92.69%→55.66%) and zero-shot Michigan
transfer (93.55%→75.27%). This highlights that
ColNomic-3B, strongly pretrained on diverse docu-
ments, already occupies a robust optimum for visual
engineering drawings. The small in-domain corpus is
insufficient to refine these embeddings under standard
contrastive loss without overwriting their generaliz-
able structure. Consequently, the off-the-shelf zero-shot
model remains the superior deployment choice.
The head-LoRA configurations restrict trainable
parameters tocustom_text_proj(the multi-vector
projection head,≈332k parameters).Gentleuses lr=1𝑒−5for 1 epoch with 10% linear warmup;standard
uses lr= 1𝑒−4for 5 epochs. The full-LM LoRA
configuration additionally inserts𝑟=32LoRA adapters
on every transformer layer’s𝑞/𝑘/𝑣/𝑜+gate/up/down
projections (37.2M parameters total for ColNomic-3B;
the comparable 39.3M canonical adapter for ColPali),
trained for 5 epochs with a unique-image batch sampler
that guarantees no positive page appears twice in a
batch.
Complementary retrieval enhancementsTwo
further enhancements are available as drop-in additions
and are orthogonal to LoRA: (i) aVLM cross-encoder
re-rank, where the Top-𝑘retrieved candidates are
each passed to a VLM with a binary “does this plan
answer the query?” prompt—this sharpens Recall@1
on visually-similar sheet pairs at a latency cost; and
(ii)hybrid BM25 fusion, combining dense MaxSim
scores with a sparse BM25 score over plan-ID/sheet-
title metadata for queries that target alphanumeric
identifiers. Pointwise VLM re-ranking (MonoQwen2-
VL-v0.1, 2B parameters) over the Top-20 retrieved
candidates adds≈5.91s of latency per query (av-
eraging≈295ms per pointwise candidate pass) on
a single NVIDIA A100 GPU, corresponding to a
throughput of approximately 610 queries per hour.
Both are reported as secondary options; the off-the-
shelf, zero-shot retriever (ColNomic-3B) is the primary
deployment configuration because it relies on the
pretrained embedding space alone without introducing
the latency of cross-encoder re-ranking or the metadata
dependencies of hybrid search.
6. Sample Cases
Figure A.1 (Appendix H) shows a representative
success case where the Qwen2.5-VL-7B generator cor-
rectly interprets a WYDOT standard plan (judge
score= 1.0); Figure A.2 shows a hard-failure case
(judge score= 0.0)—a component–dimension binding
error on the V-mesh end-strip height—that motivates
the failure-mode taxonomy in Section 7.
7. Failure Case Analysis
Although overall retrieval accuracy is high, several
queries received a judge score of 0.0. Reviewing these
zero-scorecasesrevealsfivesystematicerrorpatternsin
the model’s visual–textual reasoning over engineering
drawings.
(1) Component–dimension binding:The most
common failure was assigning a correctly detected
numeric value to the wrong physical entity. The model
repeatedly selected 4′-10′′[1.5 m]—a real, salient
dimension—to answer queries about steel-stake spac-
ing, V-mesh end-strip height, or scarification depth,
even though this value actually described enclosure
height or panel geometry. Such errors arise when
N. Subedi et al.:Preprint submitted to ElsevierPage 18 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table 15
Retrieval Recall@5: zero-shot vs. three LoRA fine-tuning configurations, on the page-disjoint 5-DOT test split (𝑁=424, in-
distribution) and the held-out Michigan DOT corpus (𝑁=93, zero-shot 6th-agency transfer).
Retriever Trainable params 5-DOT test R@5 (%) Michigan R@5 (%)
Adopted backbone
ColNomic-3B (zero-shot) 0 92.6993.55
ColNomic-3B + head-LoRA, gentle 332k (0.009%) 92.69 ( 0.00 ) 93.55 ( 0.00 )
ColNomic-3B + head-LoRA, standard 332k (0.009%)93.16(+0.47) 92.47 (−1.08)
ColNomic-3B + full-LM LoRA 37.2M (0.98%) 55.66 (−37.03) 75.27 (−18.28)
Predecessor backbone (companion)
ColPali (zero-shot) 0 76.89 88.17
ColPali + head-LoRA, gentle 332k (0.011%) 76.65 (−0.24) 88.17 ( 0.00 )
ColPali + head-LoRA, standard 332k (0.011%) 77.36 (+0.47) 88.17 ( 0.00 )
ColPali + full-LM LoRA 39.3M (1.33%) 60.85 (−16.04) 75.27 (−12.90)
multiple annotations are visually close yet refer to
different components.
(2)Cross-viewinference:Whentherequireddimen-
sion was not clearly visible in the specified view, the
model often inferred a value from another detail (e.g.,
“it can be inferred from the Typical Section”). This
violates the engineering convention that dimensions
must be taken from the explicitly referenced view,
producing hallucinated precision that yields a hard 0.0
under the rubric.
(3) Semantic misinterpretation:The model con-
flated terms such as “spacing” vs. “height” and “scar-
ification depth” vs. “fill thickness.” In one case, scar-
ification depth was reported as 4′-0′′(a fill-geometry
dimension) instead of the 6 in [150 mm] specified in
grading notes—reflecting a difficulty in distinguishing
process parameters from geometric dimensions.
(4) Symbol semantics:The model misassigned
meaning to plan-view iconography, e.g., conflating “+”
markers (wire-tie centers) with steel-stake symbols.
This indicates insufficient symbol-to-legend alignment
when symbols are reused across views.
(5) Spatial reasoning.Location-based questions pro-
duced plausible but incorrect placements (e.g., contour
ditches on the “downslope side of the right-of-way”
instead of “above the intersection of the backslope
and the original ground line”). The model defaulted
to generic civil-engineering heuristics rather than re-
producing the plan’s exact note language.
Impact of high-resolution tiling on failure
modes:To test H4, we pre-declare a≥20 pp judge-
accuracy gain as the threshold for tiling to count
as a material improvement and compare full-page
retrieval at 200 DPI against tile-level retrieval at
400 DPI (1024×1024crops with 256-pixel overlap) on
the 424-pair page-disjoint test split over the full 1,898-
page five-DOT visual index, holding the retriever fixed
across both granularities (this tiling ablation uses the
predecessor ColPali backbone). Table 16 reports the
comparison.Table 16
Full-page vs. tile-level retrieval on the 424-pair page-disjoint
test split over the 1,898-page five-DOT visual index.
Granularity R@5 (%) Tiles/Page
Full-page (200 DPI) 76.89 1
Tile-level (400 DPI) 82.08≈15
Table 17
Judge accuracy of full-page vs. tile-level retrieval on the 424-
pair test split (𝑁= 397queries whose target page yields
≥15 400 DPI tiles (about half the≈30-tile typical landscape
sheet)—in practice nearly all plan pages, which are uniformly
tile-dense).
Retrieval granularity Judge accuracy (%)
Full-page (200 DPI) 64.23 [59.4, 68.8]
Tile-level (400 DPI) 68.77 [64.2, 73.3]
Gain (tile−full) +4.53 pp
Tile-level Recall@5 (82.08%) exceeds full-page Re-
call@5(76.89%)by5.19pp,consistentwiththehypoth-
esis that high-DPI tiling preserves the fine-grained line
work and annotation density that 200 DPI page-level
rasterization loses. Focused crops remove competing
nearby annotations from the receptive field, largely
resolving component–dimension binding errors such as
scarification depth. However, crops introduce a new
failuremode:lossofcross-referencecontextwhenanote
and its referenced section fall in different tiles. This
trade-off between spatial precision and contextual com-
pleteness motivates the multi-scale retrieval strategy,
where full-page and tile-level embeddings are indexed
and retrieved jointly.
H4 verdict:H4isrejected.Onthe424-pairtestsplit,
tile-level retrieval liftsjudge accuracyfrom 64.23%
(full-page) to 68.77% (tile-level)—a+4.53pp gain (Ta-
ble 17)—well below the pre-declared 20 pp threshold;
the retrieval-level gain is similar (76.89%→82.08%
N. Subedi et al.:Preprint submitted to ElsevierPage 19 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Recall@5,+5.19pp). The much larger gain reported
on a narrow WYDOT-only dense subset does not
generalize to the heterogeneous five-DOT test set,
where nearly all 400 DPI plan pages are uniformly tile-
dense—tiling yields a consistent but modest lift rather
than a step change.
8. Discussion
The proposed Visual-First Multimodal RAG frame-
work represents a meaningful shift from OCR-centric
compliance pipelines by preserving layout, geometry,
and symbolic relationships. This leads to consistently
robust retrieval performance, maintaining over 91%
Recall@5 across five DOTs using off-the-shelf patch-
level embeddings. The robustness of these findings is
independently supported by a fully hand-curated and
manually-verified anchor set.
At the reasoning layer, model scale does not help
uniformly: at zero-shot, the 7B answerer is compet-
itive with—and marginally above—the 72B (78.30%
vs 76.89%, Table 8). The 72B’s scale advantage in
resolving dense engineering details emerges only under
structured reasoning, where question decomposition
and critic self-correction lift it to 81.60% and 82.31%.
Prompting strategies significantly influence end-to-end
outcomes; while zero-shot prompting suffices for 7B-
classmodels,larger70B-classjudgesbenefitprofoundly
from critic self-correction, capturing a large margin of
otherwise lost accuracy. Notably, the generator per-
forms best on a tight context—the single top-ranked
page—rather than an expanded set of retrieved pages,
which inject distractors and make retrieval-augmented
prompting the weakest technique on both models.
The agentic architecture fundamentally overcomes
the limitations of single-pass RAG by decomposing
complex cross-plan queries into structured verification
steps. This multi-step process, coupled with explicit
MaxSim heatmap grounding, provides the transparent,
auditable evidence trail necessary for regulatory and
safety-critical engineering reviews.
Finally, the framework exhibits exceptional zero-
shot cross-state generalization. Pretrained visual en-
coders like ColNomic-3B natively capture domain-
invariantstructuralpatterns,renderingdomain-specific
LoRA fine-tuning ineffective or even harmful on small
training corpora. The findings strongly suggest treating
pretrained multi-vector retrievers as fixed components
and focusing compute budgets on advanced reasoning
and prompting strategies.
9. Limitations
Several constraints should be considered regarding
these results. First, the 4,056-pair benchmark relies on
machine-generation and automated verification rather
than direct annotation by licensed engineers, although
the manually-curated anchor set (Section 4.1) andstructured human validation (Appendix B) mitigate
major failure modes. Second, the compliance test sets
are parameterized CAD generations. While this design
isolates VLM discrimination abilities, it does not fully
replicate real-world ambiguities such as overlapping
callouts or smudged scans; autonomous threshold ex-
traction from dense production standards remains an
open challenge—on a real 931-page WYDOT standard
corpus it succeeds on only33%of numeric items
(Section 5.9).
Furthermore, systematic reasoning failures such as
incorrect component–dimension binding and symbol
misalignment occasionally persist even with optimal re-
trieval. Finally, the system’s reliance on high-resolution
visuals makes it sensitive to low-quality scans, and
the agentic pipeline’s sequential VLM calls introduce
latency that requires further optimization for real-time
production use.
10. Conclusion
In this paper, we addressed a key limitation of
current compliance-checking systems: their reliance on
OCR-driven,text-centricpipelinesthatdiscardthelay-
out, geometry, and symbolic cues that encode meaning
in 2D engineering drawings. To overcome this, we intro-
duced a visual-first multimodal RAG(PlanSightRAG)
framework that retrieves and reasons directly over plan
imagery, combining ColNomic-3B patch-level retrieval,
an agentic compliance pipeline, and sharpened MaxSim
groundingtosupporttransparentandauditablereview.
On the 4,056-pair five-DOT benchmark, whose 424-
pair page-disjoint test split runs against the full 1,898-
page joint index, the adopted ColNomic-3B retriever
reaches92.69%Recall@5—55.90ppabovethestrongest
text retriever (BGE-M3 + OCR) and 69.58 pp above
the strongest hybrid baseline (VisionRAG (Pyramid,
RRF)).TheagenticPlanner–Auditor–Synthesizercom-
pliance pipeline, armed with per-drawing pre-resolved
rule thresholds, reaches 100% verdict accuracy on a
500-drawing single-doc CAD test set and on a 100-
drawing multi-plan set (𝑁∈{2,3,4,5}standards per
design), with both the compliant and non-compliant
directionsperfect.Thepatch-levelrepresentationtrans-
fers zero-shot to an unseen sixth agency at 91.40%
Recall@5 over the joint index, essentially matching in-
distribution performance (91.47%). A controlled LoRA
ablation of the adopted ColNomic-3B retriever spans
three configurations, 332k–37.2M trainable parame-
ters, and an order of magnitude in learning rate. No
fine-tuning recipe improves on the zero-shot baseline,
and full-LM LoRA catastrophically degrades it. High-
resolution tiling gives a modest, consistent lift: tile-
level retrieval reaches 82.08% Recall@5 versus 76.89%
for full-page (+5.19pp), and judge accuracy rises from
64.23%to68.77%(+4.53pp).Bothgainsfallwellbelow
H4’spre-declared20ppbar.Thebenchmark’sscaleand
N. Subedi et al.:Preprint submitted to ElsevierPage 20 of 31

Visual-First Multimodal RAG for Civil Standard Plans
manual anchor-set validation remove “small-sample” as
a threat to validity. The compliance results establish
that the bottleneck is no longer retrieval coverage; it is
fine-grained visual reading, entity–dimension binding,
and, decisively, supplying the judge with pre-resolved
rule thresholds.
Together, these findings suggest that visual-first
retrieval, multimodal reasoning, agentic verification,
and visual grounding are a viable foundation for au-
tomated engineering plan review, and point to symbol-
substitution robustness, multi-agency adaptation, and
scale-invariant retrieval as the next open problems.
11. Future Work
Future research should focus on expanding the
index to include historical archives and versioned stan-
dards from additional state DOTs to enable year-
aware comparisons and change tracking. Additionally,
improving perception robustness against low-quality or
handwritten legacy scans via image enhancement and
multi-resolution indexing is necessary. Extending the
visual grounding to provide precise, structured anno-
tations (e.g., dimension-entity links) that can be val-
idated against CAD-derived ground truth will further
solidify explainability. Finally, enhancing the agentic
compliance framework to autonomously handle cross-
agency regulatory conflicts, version-aware checks, and
variance documentation generation will be critical for
large-scale production deployments.
CRediT authorship contribution
statement
Nabaraj Subedi:Conceptualization, Methodol-
ogy, Software, Writing – Original Draft.Shuvo Dip
Datta:Investigation (literature review), Writing –
Original Draft.Ahmed Abdelaty:Conceptualiza-
tion, Supervision, Review & Editing.Shivanand
Venkanna Sheshappanavar:Supervision, Review &
Editing.
Declaration of Competing Interest
The authors declare that they have no known
competing financial interests or personal relationships
thatcouldhaveappearedtoinfluencetheworkreported
in this paper.
Data Availability
The code and sample datasets used in this study
will be made available upon publication. The full set
of DOT Standard Plans is publicly available through
their respective website.Funding
ThisworkwasfinanciallysupportedbytheWyoming
Department of Transportation (WYDOT) under the
knowledge-managementproject,grantnumberRS03225
(Principal Investigator: Ahmed Abdelaty).
Acknowledgement
The authors would like to acknowledge the various
state DOTs for providing the standard plans used as
theprimarydatasetforthisresearch.Wealsothankthe
open-source community for the development of ColPali
and Qwen 2.5-VL, which served as the foundational
models for this work. Computational resources were
provided by the Advanced Research Computing Center
(ARCC) at the University of Wyoming.
Declaration of generative AI used in the
manuscript preparation process
During the preparation of this work the authors
used generative AI tools like Claude to assist with
editing, consistency checking, writing and coding. After
using this tool, the authors reviewed and edited the
content as needed and take full responsibility for the
content of the published article.
References
[1] Cyril Picard, Kristen M. Edwards, Anna C. Doris, Bran-
don Man, Giorgio Giannone, Md Ferdous Alam, and
Faez Ahmed. From concept to manufacturing: evaluat-
ing vision-language models for engineering design.Ar-
tificial Intelligence Review, 58:288, 2025. doi: 10.1007/
s10462-025-11290-y.
[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küt-
tler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian
Riedel, and Douwe Kiela. Retrieval-augmented generation
for knowledge-intensive NLP tasks. InAdvances in Neural
Information Processing Systems (NeurIPS), volume 33,
pages 9459–9474, 2020.
[3] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann,
Trevor Cai, Eliza Rutherford, Katie Millican, George Bm
Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc,
Aidan Clark, et al. Improving language models by retrieving
from trillions of tokens. InProceedings of the 39th In-
ternational Conference on Machine Learning, volume 162
ofProceedings of Machine Learning Research, pages 2206–
2240. PMLR, 2022.
[4] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hos-
seini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Ar-
mand Joulin, Sebastian Riedel, and Edouard Grave. Atlas:
Few-shot learning with retrieval augmented language mod-
els.Journal of Machine Learning Research, 24(251):1–43,
2023.
[5] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Am-
non Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-
context retrieval-augmented language models.Transactions
of the Association for Computational Linguistics, 11:1316–
1331, 2023. doi: 10.1162/tacl_a_00605.
[6] Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu
Wei, and Ming Zhou. LayoutLM: Pre-training of text and
N. Subedi et al.:Preprint submitted to ElsevierPage 21 of 31

Visual-First Multimodal RAG for Civil Standard Plans
layout for document image understanding. InProceedings
of the 26th ACM SIGKDD International Conference on
Knowledge Discovery & Data Mining, pages 1192–1200,
2020. doi: 10.1145/3394486.3403172.
[7] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani,
Gautier Viaud, Céline Hudelot, and Pierre Colombo. Col-
Pali: Efficient document retrieval with vision language
models. InInternational Conference on Learning Repre-
sentations (ICLR), 2025. doi: 10.48550/arXiv.2407.01449.
[8] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan
Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren
Zhou. Qwen-vl: A versatile vision-language model for
understanding,localization,textreading,andbeyond.arXiv
preprint arXiv:2308.12966, 2023. doi: 10.48550/arXiv.2308.
12966. arXiv preprint arXiv:2308.12966.
[9] Gemini Team, Google. Gemini: A family of highly capable
multimodalmodels.arXiv preprint arXiv:2312.11805,2025.
doi: 10.48550/arXiv.2312.11805.
[10] Charles Eastman, Jae-min Lee, Yeon-suk Jeong, and Jin-
kook Lee. Automatic rule-based checking of building de-
signs.Automation in Construction, 18(8):1011–1033, 2009.
doi: 10.1016/j.autcon.2009.07.002.
[11] Wawan Solihin and Charles Eastman. Classification of rules
for automated BIM rule checking development.Automation
in Construction,53:69–82,2015. doi:10.1016/j.autcon.2015.
03.003.
[12] Pingbo Tang, Daniel Huber, Burcu Akinci, Robert Lipman,
and Alan Lytle. Automatic reconstruction of as-built build-
ing information models from laser-scanned point clouds: A
review of related techniques.Automation in Construction,
19(7):829–843, 2010. doi: 10.1016/j.autcon.2010.06.007.
[13] Yang Xu, Yiheng Xu, Tengchao Lv, Lei Cui, Furu Wei,
Guoxin Wang, Yijuan Lu, Dinei Florencio, Cha Zhang,
WanxiangChe,MinZhang,andLidongZhou. LayoutLMv2:
Multi-modal pre-training for visually-rich document un-
derstanding. InProceedings of the 59th Annual Meeting
of the Association for Computational Linguistics and the
11th International Joint Conference on Natural Language
Processing (Volume 1: Long Papers), pages 2579–2591.
Association for Computational Linguistics, 2021. doi: 10.
18653/v1/2021.acl-long.201.
[14] Srikar Appalaraju, Bhavan Jasani, Bhargava Urala Kota,
Yusheng Xie, and R. Manmatha. DocFormer: End-to-end
transformer for document understanding. InProceedings
of the IEEE/CVF International Conference on Computer
Vision (ICCV), pages 993–1003, 2021. doi: 10.1109/
ICCV48922.2021.00103.
[15] Chengke Wu, Xiao Li, Yuanjun Guo, Jun Wang, Zengle
Ren, Meng Wang, and Zhile Yang. Natural language
processingforsmartconstruction:Currentstatusandfuture
directions.Automation in Construction, 134:104059, 2022.
doi: 10.1016/j.autcon.2021.104059.
[16] Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar.
DocVQA: A dataset for VQA on document images. In
Proceedings of the IEEE/CVF Winter Conference on Ap-
plications of Computer Vision (WACV), pages 2199–2208,
2021. doi: 10.1109/WACV48630.2021.00225.
[17] Josselin Somerville Roberts, Tony Lee, Chi Heem Wong,
Michihiro Yasunaga, Yifan Mai, and Percy Liang. Im-
age2Struct: Benchmarking structure extraction for vision-
language models. InProceedings of the 38th International
Conference on Neural Information Processing Systems
(NeurIPS), pages 115058–115097, 2024. doi: 10.48550/
arXiv.2410.22456.
[18] Fangyu Liu, Francesco Piccinno, Syrine Krichene, Chenxi
Pang, Kenton Lee, Mandar Joshi, Yasemin Altun, Nigel
Collier, and Julian Martin Eisenschlos. MatCha: Enhancing
visual language pretraining with math reasoning and chartderendering. InProceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Volume 1:
Long Papers), pages 12756–12770. Association for Compu-
tational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.
714.
[19] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret
Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi
Parikh. VQA: Visual question answering. InProceedings
of the IEEE International Conference on Computer Vision
(ICCV), pages 2425–2433, 2015. doi: 10.1109/ICCV.2015.
279.
[20] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry,
Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen
Krueger, and Ilya Sutskever. Learning transferable visual
models from natural language supervision. InProceedings
of the 38th International Conference on Machine Learning,
volume 139 ofProceedings of Machine Learning Research,
pages 8748–8763. PMLR, 2021.
[21] ZheChen,WeiyunWang,YueCao,YangzhouLiu,Zhangwei
Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian,
Zhaoyang Liu, Lixin Gu, Xuehui Wang, et al. Expanding
performance boundaries of open-source multimodal models
with model, data, and test-time scaling.arXiv preprint
arXiv:2412.05271, 2024. doi: 10.48550/arXiv.2412.05271.
[22] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long
Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong
Ye, Jie Shao, et al. Internvl3.5: Advancing open-source
multimodal models in versatility, reasoning, and efficiency.
arXiv preprint arXiv:2508.18265, 2025. doi: 10.48550/
arXiv.2508.18265.
[23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.
BLIP-2: Bootstrapping language-image pre-training with
frozen image encoders and large language models. InPro-
ceedings of the 40th International Conference on Machine
Learning, volume 202 ofProceedings of Machine Learning
Research, pages 19730–19742. PMLR, 2023.
[24] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine
Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch,
Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual
languagemodelforfew-shotlearning. InAdvances in Neural
Information Processing Systems (NeurIPS), volume 35,
pages 23716–23736, 2022.
[25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae
Lee. Visual instruction tuning. InAdvances in Neural
Information Processing Systems (NeurIPS), volume 36,
pages 34892–34916, 2023. doi: 10.52202/075280-1516.
[26] OpenAI. GPT-4 technical report.arXiv preprint
arXiv:2303.08774, 2023. doi: 10.48550/arXiv.2303.08774.
[27] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ah-
mad, Ilge Akkaya, Florencia Leoni Aleman, et al. GPT-
4V(ision) system card.arXiv preprint arXiv:2309.17421,
2024. doi: 10.48550/arXiv.2309.17421. OpenAI Technical
Report.
[28] Vasil Shteriyanov, Rimma Dzhusupova, Jan Bosch, and
Helena Holmström Olsson. Blueprintsymvl: A discrimina-
tive benchmark for VLM symbol recognition in engineering
blueprints.Results in Engineering, 28:108171, 2025. doi:
10.1016/j.rineng.2025.108171.
[29] Leonhard Kunz, Mario Klostermeier, Kokulan Thanabalan,
Tatjana Legler, and Martin Ruskowski. Techmb: Exploring
the potential of vision language models for interpreting
technical drawings. InDS 140: Proceedings of the 36th
Symposium Design for X (DFX2025), pages 179–188. The
Design Society, 2025. doi: 10.35199/dfx2025.19.
[30] Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, and
Pratyush Kumar. PlotQA: Reasoning over scientific plots.
InProceedings of the IEEE/CVF Winter Conference on
N. Subedi et al.:Preprint submitted to ElsevierPage 22 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Applications of Computer Vision (WACV), pages 1527–
1536, 2020. doi: 10.1109/WACV45572.2020.9093523.
[31] Samira Ebrahimi Kahou, Vincent Michalski, Adam Atkin-
son, Akos Kadar, Adam Trischler, and Yoshua Bengio.
FigureQA: An annotated figure dataset for visual reasoning.
InInternational Conference on Learning Representations
(ICLR) Workshop, 2018.
[32] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty,
and Enamul Hoque. ChartQA: A benchmark for question
answering about charts with visual and logical reasoning. In
Findings of the Association for Computational Linguistics:
ACL 2022, pages 2263–2279. Association for Computational
Linguistics, 2022. doi: 10.18653/v1/2022.findings-acl.177.
[33] Kushal Kafle, Brian Price, Scott Cohen, and Christopher
Kanan. DVQA: Understanding data visualizations via
question answering. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), pages 5648–5656, 2018. doi: 10.1109/CVPR.2018.
00592.
[34] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia
Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang,
Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hal-
lusionBench: An advanced diagnostic suite for entangled
language hallucination and visual illusion in large vision-
language models. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition
(CVPR), 2024. doi: 10.48550/arXiv.2310.14566.
[35] Omar Khattab and Matei Zaharia. ColBERT: Efficient and
effective passage search via contextualized late interaction
over BERT. InProceedings of the 43rd International
ACM SIGIR Conference on Research and Development in
Information Retrieval, pages 39–48, 2020. doi: 10.1145/
3397271.3401075.
[36] Geewook Kim, Teakgyu Hong, Moonbin Yim, JeongYeon
Nam, Jinyoung Park, Jinyeong Yim, Wonseok Hwang,
Sangdoo Yun, Dongyoon Han, and Seunghyun Park. OCR-
free document understanding transformer. InEuropean
Conference on Computer Vision (ECCV), pages 498–517,
2022. doi: 10.1007/978-3-031-19815-1_29.
[37] Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and
William W. Cohen. MuRAG: Multimodal retrieval-
augmented generator for open question answering over
images and text. InProceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing
(EMNLP),pages5558–5570.AssociationforComputational
Linguistics, 2022. doi: 10.18653/v1/2022.emnlp-main.375.
[38] Nomic Team. Nomic embed multimodal: Interleaved text,
image,andscreenshotsforvisualdocumentretrieval.https:
//nomic.ai/blog/posts/nomic-embed-multimodal, 2025.
[39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszko-
reit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia
Polosukhin. Attentionisallyouneed. InAdvances in Neural
Information Processing Systems (NeurIPS), volume 30,
pages 6000–6010, 2017.
[40] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon,
Christopher Potts, and Matei Zaharia. ColBERTv2: Effec-
tive and efficient retrieval via lightweight late interaction. In
Proceedings of the 2022 Conference of the North American
Chapter of the Association for Computational Linguistics
(NAACL), pages 3715–3734, 2022. doi: 10.18653/v1/2022.
naacl-main.272.
[41] Yash Patel, Giorgos Tolias, and Jiří Matas. Recall@k surro-
gate loss with large batches and similarity mixup. In2022
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), pages 7492–7501, New Orleans, LA,
USA, 2022. IEEE. doi: 10.1109/CVPR52688.2022.00735.
[42] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan
Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, ZhuohanLi, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E.
Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with
MT-Bench and chatbot arena. InAdvances in Neural
Information Processing Systems (NeurIPS) Datasets and
Benchmarks Track, volume 36, pages 46595–46623, 2023.
doi: 10.48550/arXiv.2306.05685.
[43] Gabriel de Souza P. Moreira, Ronay Ak, Mengyao Xu,
Oliver Holworthy, Benedikt Schifferer, Zhiding Yu, Yauhen
Babakhin, Radek Osmulski, Jiarui Cai, Ryan Chesler,
Bo Liu, and Even Oldridge. Nemotron ColEmbed v2: Top-
performing late interaction embedding models for visual
document retrieval. InProceedings of the 1st Late Interac-
tion and Multi-Vector Retrieval Workshop (LIR) at ECIR,
2026. doi: 10.48550/arXiv.2602.03992.
[44] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen,
andJimmyLin. Unifyingmultimodalretrievalviadocument
screenshot embedding. InProceedings of the 2024 Confer-
ence on Empirical Methods in Natural Language Processing
(EMNLP), pages 6492–6505, 2024. doi: 10.18653/v1/2024.
emnlp-main.373.
[45] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao
Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han,
Zhiyuan Liu, and Maosong Sun. VisRAG: Vision-based
retrieval-augmented generation on multi-modality docu-
ments. InInternational Conference on Learning Represen-
tations (ICLR), 2025. doi: 10.48550/arXiv.2410.10594.
[46] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo,
Defu Lian, and Zheng Liu. M3-Embedding: Multi-
linguality, multi-functionality, multi-granularity text em-
beddings through self-knowledge distillation. InFindings of
the Association for Computational Linguistics: ACL 2024,
pages 2318–2335. Association for Computational Linguis-
tics, 2024. doi: 10.18653/v1/2024.findings-acl.137.
[47] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu
Wei. LayoutLMv3: Pre-training for document AI with
unified text and image masking. InProceedings of the 30th
ACM International Conference on Multimedia, pages 4083–
4091, 2022. doi: 10.1145/3503161.3548112.
[48] LukasBlecher,GuillemCucurullPreixens,ThomasScialom,
and Robert Stojnic. Nougat: Neural optical understanding
for academic documents. InInternational Conference
on Learning Representations (ICLR), pages 37646–37663,
2024.
[49] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu,
Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandel-
wal, Peter Shaw, Ming-Wei Chang, and Kristina Toutanova.
Pix2Struct: Screenshot parsing as pretraining for visual
language understanding. InProceedings of the 40th Inter-
national Conference on Machine Learning, volume 202 of
Proceedings of Machine Learning Research, pages 18893–
18912. PMLR, 2023.
[50] Zineng Tang, Ziyi Yang, Guoxin Wang, Yuwei Fang, Yang
Liu, Chenguang Zhu, Michael Zeng, Cha Zhang, and Mohit
Bansal. Unifying vision, text, and layout for universal
document processing. InProceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recogni-
tion (CVPR), pages 19254–19264, 2023. doi: 10.1109/
CVPR52729.2023.01845.
N. Subedi et al.:Preprint submitted to ElsevierPage 23 of 31

Visual-First Multimodal RAG for Civil Standard Plans
A. Supplementary data
Supplementary material related to this article—
including the 4,056-pair five-DOT benchmark, the
page-disjoint train/dev/test splits, the held-out Michi-
ganDOTtransferset,the500single-docand100multi-
plan CAD-generated compliance test sets, the 424-
pair page-disjoint test split used for the main-body
retrieval comparison (Table 4) and the per-category
VQA judge evaluation, the LoRA adapter checkpoints
(head-gentle, head-standard, full-LM) and the training
scripts, the complete set of visual grounding heatmaps,
and the specific prompt templates used for each agent
will be made available upon acceptance.
B. Human validation of the generated
datasets
Because both the QnA benchmark and the com-
pliance test set are model-generated, we complement
the automatic generate-then-verify pipeline with two
human-grounded checks: the fully hand-curated anchor
setdescribedinSection4.1,andalarge-scalestructured
web validation of the generated corpus.
Protocol.All4,656generated items—the4,056QnA
pairs (full five-DOT benchmark) together with the500
single-doc and100multi-plan compliance drawings—
arepresentedto humanreviewersthrougha lightweight
web interface. For each item the reviewer sees the
original drawing at full resolution alongside the com-
plete context needed for an independent judgement: for
QnA pairs, the model-generated question and answer
plus the source agency, plan identifier, sheet title,
and reasoning category; for compliance drawings, the
encoded design facts, the rule and threshold being
checked, the named governing standard (e.g., ACI 318,
AASHTO, WYDOT 606.05, PROWAG/ADA), the
model’s compliant/non-compliant verdict, and the
most relevant standard-plan page(s) retrieved by Col-
Pali for that check. Each reviewer records one of
three verdicts—correct,incorrect, orunsure—
with optional free-text notes. The study runs in
agreement mode: every item is shown to every re-
viewer independently, so items reviewed by two or
more people yield an inter-annotator agreement signal,
and disagreements localize the pairs most worth re-
examining. Per-reviewer verdicts are logged separately
and aggregated post hoc.
Status.This larger-scale validation over all4,656gen-
erated items isongoing: per-type confirmation rates,
inter-annotator agreement on doubly-reviewed items,
and flagged-item counts will be released together with
the code and data. In the interim, the hand-curated
78-pair anchor set provides a fully human-grounded
check, and the automatically generated-then-verified
pipeline (an independent Qwen-VL-7B verifier thatrejects roughly half of all drafted questions) guards
label quality at scale; pending completion, we scope
the benchmark’s label-quality guarantee to the verified
anchor set plus this automated two-stage filter.
C. Bootstrap CIs on the 424-pair test
split
This appendix reports bootstrap confidence inter-
vals on the full 424-pair page-disjoint test split over
the 1,898-page five-DOT visual index (no subset).
Table A.1 gives per-category Recall@5 (generator-
independent,cleanzero-shotColPali)andper-generator
Judge Accuracy for the open generators (Qwen2.5-
VL-7B, Qwen2.5-VL-72B, InternVL-2.5-8B), the latter
scored by a local Qwen2.5-VL-72B judge; Table A.2
gives per-baseline, per-agency Recall@5. All CIs are
percentile bootstrap with 10,000 resamples (seed 42).
D. Additional retrieval and judge
analyses
Rank-sensitive retrieval metrics.Table A.3 reports
Recall@1, Recall@5, MRR, and nDCG@10 for the
adopted ColNomic-3B retriever on the 424-pair page-
disjoint test split over the 1,898-page index. The≈26-
pp gap between Recall@5 (92.45%) and Recall@1
(66.75%) is material for the single-pass auditor, which
consumes the rank-1 page; it motivates the optional
cross-encoder re-ranker for rank-1-sensitive deploy-
ments.
Michigan transfer over the joint index.To re-
move the candidate-pool-size confound, we re-evaluate
Michigan retrieval over thecombined1,898+298-page
index rather than the298-page Michigan-only pool
(Table A.4). Over the joint index Michigan transfers
at91.40%Recall@5—essentially matching the in-
distribution benchmark (91.47%)—rather than the
93.55%obtained on the smaller, easier Michigan-only
pool. We therefore report the joint-index figure as the
comparabletransfernumberanddonotclaimMichigan
exceeds in-distribution agencies.
Cross-family judge agreement.Because Qwen2.5-VL
serves as drafter, verifier, answerer, and judge, we test
for same-family self-preference by re-judging the 424
Qwen2.5-VL-7B zero-shot answers with a non-Qwen
judge (InternVL2.5-8B) on identical inputs (gold page,
question, reference, model answer). The two judges
agree on92.92%of items with Cohen’s𝜅= 0.75(sub-
stantial); the same-family Qwen judge is in factstricter
(79.95%vs the cross-family85.61%), the opposite of a
self-preference bias. Judge verdicts are therefore not an
artifact of evaluator–generator family overlap.
N. Subedi et al.:Preprint submitted to ElsevierPage 24 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table A.1
Per-category Recall@5 (Hit) and Judge Accuracy (Judge) on the full 424-pair page-disjoint test split over the 1,898-page five-
DOT index, with 95% bootstrap CIs (10,000 resamples, seed 42). Recall@5 is a clean zero-shot ColPali and is generator-
independent. Judge Accuracy is scored by a local Qwen2.5-VL-72B judge (no proprietary API). Per-category𝑁: Dim=112,
Vis=145, Log=120, Hal=47. Logical-Reasoning Recall@5 is markedly lower here (52.50%) than in Table 7 (87.64%) because this
table uses the predecessor ColPali backbone, whose fixed448×448encoder is weakest on note/table-dense logical queries; the
adopted ColNomic-3B closes this gap.
Model Category Hit (%) Judge (%)
Qwen2.5-VL-72BDimensional 86.61 [80.4, 92.9] 62.16 [53.2, 71.2]
Visual 86.21 [80.0, 91.7] 78.47 [71.5, 84.7]
Logical 52.50 [43.3, 61.7] 69.17 [60.8, 77.5]
Hallucination 87.23 [76.6, 95.7] 70.21 [57.4, 83.0]
Overall 76.89 [72.9, 80.9] 70.62 [66.1, 74.9]
Qwen2.5-VL-7BDimensional 86.61 [80.4, 92.9] 60.36 [51.4, 69.4]
Visual 86.21 [80.0, 91.7] 68.06 [60.4, 75.7]
Logical 52.50 [43.3, 61.7] 59.17 [50.0, 68.3]
Hallucination 87.23 [76.6, 95.7] 57.45 [42.6, 70.2]
Overall 76.89 [72.9, 80.9] 62.32 [57.6, 66.8]
InternVL-2.5-8BDimensional 86.61 [80.4, 92.9] 43.75 [34.8, 52.7]
Visual 86.21 [80.0, 91.7] 60.00 [51.7, 68.3]
Logical 52.50 [43.3, 61.7] 48.33 [39.2, 57.5]
Hallucination 87.23 [76.6, 95.7] 31.91 [19.1, 44.7]
Overall 76.89 [72.9, 80.9] 49.29 [44.6, 54.0]
Table A.2
Per-baseline Recall@5 with 95% bootstrap CIs on the 424-pair page-disjoint test split (10,000 resamples, seed 42). Per-agency
𝑁: WYDOT= 53, Caltrans= 134, AZDOT= 46, CDOT= 31, FDOT= 160.
Method Overall (%) WYDOT Caltrans AZDOT CDOT FDOT
ColQwen2.5-v0.2 87.26 [84.0, 90.3] 92.5 [84.9, 98.1] 87.3 [81.3, 92.5] 82.6 [71.7, 93.5] 87.1 [74.2, 96.8] 86.9 [81.2, 91.9]
DSE-Qwen2-2B 22.17 [18.2, 26.2] 30.2 [18.9, 43.4] 20.1 [13.4, 26.9] 26.1 [13.0, 39.1] 25.8 [9.7, 41.9] 19.4 [13.1, 25.6]
VisRAG-Ret 53.77 [49.1, 58.5] 58.5 [45.3, 71.7] 46.3 [38.1, 54.5] 67.4 [54.3, 80.4] 48.4 [32.3, 64.5] 55.6 [47.5, 63.1]
BGE-M3 + OCR 36.79 [32.3, 41.5] 45.3 [32.1, 58.5] 26.1 [18.7, 33.6] 45.7 [32.6, 60.9] 41.9 [25.8, 58.1] 39.4 [31.9, 46.9]
CLIP ViT-B/32 1.89 [0.7, 3.3] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 8.7 [2.2, 17.4] 0.0 [0.0, 0.0] 2.5 [0.6, 5.0]
LayoutLMv3 0.00 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0]
Nougat-decode + MiniLM 0.47 [0.0, 1.2] 1.9 [0.0, 5.7] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.6 [0.0, 1.9]
Pix2Struct-decode + MiniLM 7.31 [5.0, 9.9] 11.3 [3.8, 20.8] 1.5 [0.0, 3.7] 13.0 [4.3, 23.9] 16.1 [3.2, 29.0] 7.5 [3.8, 11.9]
UDOP-decode + MiniLM 0.00 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0] 0.0 [0.0, 0.0]
OCR + MiniLM 25.24 [21.2, 29.5] 34.0 [20.8, 47.2] 18.7 [12.7, 25.4] 43.5 [30.4, 56.5] 16.1 [3.2, 29.0] 24.4 [17.5, 31.2]
VisionRAG (Pyramid, RRF) 23.11 [19.1, 27.1] 32.1 [18.9, 45.3] 18.7 [11.9, 25.4] 28.3 [15.2, 41.3] 16.1 [3.2, 29.0] 23.8 [17.5, 30.0]
HPC-ColPali (BQ) 65.09 [60.6, 69.6] 71.7 [58.5, 83.0] 62.7 [54.5, 70.9] 67.4 [54.3, 80.4] 41.9 [25.8, 58.1] 68.8 [61.2, 75.6]
Table A.3
Rank-sensitive retrieval metrics for zero-shot ColNomic-3B on
the 424-pair test split (1,898-page index).
Category𝑁R@1 R@5 MRR nDCG@10
Dimensional 112 67.86 97.32 0.803 0.847
Visual 145 68.28 93.10 0.788 0.828
Logical 120 69.17 88.33 0.778 0.810
Hallucination 47 53.19 89.36 0.682 0.741
Overall 424 66.75 92.45 0.777 0.819
E. Compliance pipeline confusion
matrix and decomposition analysis
This appendix presents the full confusion matrix for
the 8-query agentic compliance evaluation reported inTable A.4
Zero-shot Michigan transfer, joint vs. Michigan-only candidate
pool.
Candidate pool #Candidates R@1 R@5
Joint (1,898+298) 2,196 68.8291.40
Michigan-only 298 74.19 93.55
Section 5.6. Table A.5 gives the per-query predicted
verdict, ground-truth verdict, targetplans, andnumber
of verification steps executed. Ground-truth verdicts
are derived from the fact that the eight compliance
queries target internally consistent WYDOT standard-
plan configurations—no deliberately malformed inputs
were constructed—so the correct verdict is PASS in
every case.
N. Subedi et al.:Preprint submitted to ElsevierPage 25 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table A.5
Agenticcomplianceconfusionmatrixover8cross-planverifica-
tion queries. GT and Pred are the ground-truth and predicted
verdicts; Match indicates agreement; Steps is the number of
verification steps the Planner produced. Expected minimum
decomposition is 2 steps per query.
ID Plans GT Pred Match Steps
comp_01 511-1A, 203-2A PASS PASS✓8
comp_02 606-7C, 606-2B PASS PASS✓7
comp_03 511-1A PASS PASS✓8
comp_04 202-1 PASS PASS✓9
comp_05 511-1A PASS PASS✓10
comp_06 606-4B PASS PASS✓8
comp_07 203-2A PASS PASS✓10
comp_08 203-2A PASS PASS✓9
Over-decomposition.Expected minimum decomposi-
tion for each query is 2 steps (identify, then verify).
The Planner produces 7–10 steps per query, averaging
8.6 steps overall, i.e.,∼4.3×the minimum. Over-
decomposition does not compromise correctness in
this evaluation because every expanded sub-step still
retrieves a relevant plan region, but it does inflate end-
to-end latency linearly in the step count.
Accuracy.8/8 = 100.0%. The ground-truth construc-
tion (all-PASS) means this metric reflects the pipeline’s
ability to avoid false-positive rejections on valid plans;
itisnotadirectmeasureofsensitivitytocompliancevi-
olations.Acomplementaryevaluationwithdeliberately
malformed plans (insertion of inconsistent dimensions,
missing notes, conflicting cross-references) is listed as
follow-up work to characterise false-negative behavior,
and is beyond the current scope because no such
violation corpus exists in the WYDOT standard-plan
archive.
Known limitations.(i) The evidence-trail metric un-
derreports actual grounding quality because it treats
missing bounding-box metadata as missing evidence,
even though the Auditor’s rendered heatmaps and
textual citations are present. A post-hoc extension to
emit machine-parseable box coordinates would lift this
metric without any change to the underlying reasoning.
(ii) The 8-query scale is too small for per-domain
breakdowns; the eight queries are stratified across
compliance domains (geotextile, guardrail, drainage,
mailbox sight distance, gabion, silt fence, scarification,
contour ditch) but each domain contributes only one
query.
E.1. False-Negative Sensitivity Study
The 100% accuracy reported in Section 5.6 is
measured on an all-PASS ground-truth set and there-
fore characterises the pipeline’sfalse-positive rejec-
tion avoidancerather than itsfalse-negative violation-
detection sensitivity. Because the agentic pipeline isquery-driven—theAuditorreceivesaretrievedstandard-
plan page from ColPali and compares any proposed
design valuesstated in the query textagainst the
standard—we probe sensitivity by constructing a syn-
thetic corpus of queries that describe deliberately non-
compliant proposed designs, rather than by mutating
the underlying plan images. This matches the natural
deployment mode of the system (engineer submits
a proposed value; the pipeline checks it against the
relevant WYDOT standard) and isolates the Auditor’s
violation-detection behaviour from any retrieval-layer
artefacts.
Violation taxonomy.Twenty-foursynthetic-FAILquer-
ies are constructed, stratified evenly across three
violation types (eight queries per type):
•Dimension mutation (M dim): the query pro-
poses a numeric value that violates the WYDOT
specification for that component (e.g., “A con-
tractor proposed a steel stake vertical spacing of
15ft for silt fence installation per Plan 606-4B. Is
this compliant with the required spacing?”—the
true spacing is 4′-10′′).
•Note omission (M note): the query describes
a proposed construction procedure that skips a
required note-level constraint (e.g., omitting the
geotextile underlayment or the scarification step
referenced in the grading notes).
•Symbol / component swap (M sym): the
query substitutes a different physical component
for one depicted in the plan (e.g., wooden stakes
in place of steel stakes, or a concrete end anchor
in place of a cable anchor).
CombinedwiththeeightoriginalPASSqueriesfrom
Section 5.6, this yields an evaluation set of𝑁=32
queries (8PASS+ 24FAIL) that the pipeline processes
blind. The full query list is provided in the released
artefacts(agentic_compliance/run_fn_sensitivity.
py).
Detection rule.A query is counted asdetected
if the Synthesizer’s final verdict in the rendered
Markdown report matches any of the regexes (case-
insensitive):FAIL,NON-COMPLIANT,DOES NOT (MEET||
COMPLY||MATCH||SATISFY),VIOLAT(E||ES||ION),
EXCEEDS MAXIMUM, orBELOW MINIMUM. This rule is
conservative—it requires the pipeline to state the
violation verdict explicitly rather than imply it.
Metrics.We report the binary-classification confusion
matrix together with the true-positive rate (TPR,
equivalentlysensitivity), false-negative rate (FNR),
specificity (TNR), precision, F1, and Cohen’s𝜅against
N. Subedi et al.:Preprint submitted to ElsevierPage 26 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Table A.6
False-negative sensitivity on the𝑁=32synthetic query corpus.
Per-stratum verdict counts (PASS/FAIL); GT=FAIL rows
quantify detection sensitivity, and the GT=PASS row repeats
the Section 5.6 PASS accuracy for reference.
Stratum𝑁PASS / FAIL
GT=PASS (originals) 8 8 / 0
GT=FAIL, M dim 8 1 / 7
GT=FAIL, M note 8 1 / 7
GT=FAIL, M sym 8 3 / 5
Overall32 13 / 19
the synthetic ground truth:
TPR=TP
TP+FN,FNR=FN
FN+TP,
TNR=TN
TN+FP.(6)
Stratified TPR/FNR are additionally reported per
violation type to identify which classes the Auditor is
most likely to miss.
Results.The pipeline achieves perfect specificity
(TNR=100%) on the eight original PASS queries—
zero false positives on valid designs—and an overall
TPR of 79.17% on the twenty-four FAIL queries, for
a balanced-class𝜅of 0.655.
The per-type breakdown is revealing: dimension-
level (M dim) and note-omission (M note) violations are
caught at identical87.5% (7/8) sensitivity, whereas
symbol/component-swap(M sym)violationsfallto62.5%
(5/8). The three Msym symmisses are cases where the
swapped component is visually plausible in the drawing
(wooden vs. steel stakes, concrete vs. cable end anchor)
andtheAuditor’sMaxSimretrievalsurfacesthecorrect
plan page, but the downstream VQA step does not
explicitly contrast the proposed component against
the one depicted, instead issuing a hedged PASS.
Dimension mismatches—where the query states an
explicit numeric value—trigger a numeric compliance
check that the Auditor handles reliably. Precision is
100% by construction on this corpus because the
pipeline never rejects a valid design on the PASS
stratum, and the single overall metric most relevant for
compliance deployment (FNR=20.83%) is dominated
by the symbol-swap class.
Status.The query corpus, the runner script, the
slurm submission wrapper, and the scoring harness
(underagentic_compliance/andevaluation/fn_
sensitivity/)arereleasedalongsidetheprimaryeval-
uation artifacts. The numerical results in Table A.6 are
measured values from job53124015on themb-a6000
partition (24/24 FAIL queries completed successfully,
averagepipelinelatency≈62sperquery).ThisstudyisTable A.7
Retrieval-only latency distribution (ms) for full-page ColNomic
MaxSim retrieval on the 424-pair page-disjoint test split over
the combined 1,898-page five-DOT visual index. Wall-clock
timecollectedwithtime.perf_counter()onasingleNVIDIA
A100 80GB with PyTorch 2.x, one query at a time (batch size
1).
Percentile Full-page (424 queries, 1,898-page index)
min 98.72
p25 100.24
p50 101.26
p75 101.74
p95 102.48
p99 103.71
max 138.19
mean 101.24
Table A.8
Consolidated deployment latency on a single 80GB GPU
(batch 1). Indexing is a one-time offline cost; all other rows
are per query. The full retrieval percentile distribution is in
TableA.7;theagenticfigureistheaverageoverthecompliance
runs (Section 5.6).
Stage Cost Notes
Indexing (1,898 pages)7.3min (one-time)4.34pages/s, offline
Retrieval / query≈0.10s (p50) query enc.+MaxSim
Re-rank (optional)+≈0.3s/cand.≤𝐾passes; early-stop
VQA answer / query2–5s single-shot 72B
Agentic audit / query60.9s∼8.6sequential steps
citedastheimmediatefollow-upiteminSection11,and
the symbol-swap failure mode is flagged there as the
specific weakness that the next iteration should target
(e.g., an explicit component-comparison sub-step after
retrieval).
F. Latency distribution
This appendix reports end-to-end deployment la-
tency. Table A.8 consolidates the one-time indexing
costandtheper-querycostofeachpipelinestageonthe
single-GPU target; Table A.7 gives the full per-query
retrieval percentile distribution. All measurements are
wall-clock time (batch size 1) on a single 80GB GPU
(NVIDIA A100/H100).
Interpretation.The full-page retrieval latency distri-
bution over the 1,898-page five-DOT index is tightly
concentrated around the median (101 ms) with a heavy
right tail (p99=103.71 ms) driven by cold-cache
query encoding on the first queries of each run. The
mean of 101.24 ms/query reflects the linear corpus
scalingimpliedbytheMaxSimbrute-forcescoringloop.
The tile-level variant trades higher per-query cost for
N. Subedi et al.:Preprint submitted to ElsevierPage 27 of 31

Visual-First Multimodal RAG for Civil Standard Plans
finer spatial granularity; the binary-quantized HPC-
ColPali variant (Section 5.1) adds only millisecond-
scale unpacking overhead while compressing the index
by an order of magnitude.
End-to-end latency.Retrieval is one component of
end-to-end latency; VLM generation dominates the
wall-clock budget. As a first-order estimate, Qwen-
7B VQA on a1024×1024crop with 256 generation
tokens takes 2–4 s on the same A100 80GB, so total
interactive latency sits in the 2–5 s range per single-
shot query, excluding the iterated Planner–Auditor–
Synthesizer turns in the agentic compliance pipeline,
which typically multiplies this by 3–5×depending on
the number of plans cross-referenced.
G. Grounding IoU pilot
This appendix summarizes the grounding IoU pi-
lot study referenced in Section 5.5. The 16-query
pilot set comprises: 12 queries drawn from the suc-
cess/failure case studies in Section 6 (steel-stake spac-
ing, MDC depth, stake-spacing table, wire-tie loca-
tion, plus-symbol convention, diaphragm identification,
edge-draintrenchwidth,V-meshend-stripheight,scar-
ification depth, drainage fabric layout, contour ditch
placement) and 4 additional dimensional-accuracy and
visual-interpretation queries from the primary bench-
mark. Predicted bounding boxes are extracted from the
sharpened MaxSim heatmap by thresholding at the top
5% of patch activations and taking axis-aligned boxes
over the resulting connected components. Ground-
truth boxes are hand-drawn by a domain-informed
reviewer at full plan resolution.
Metric definition.For each query, we compute the
maximum IoU over all pairs of ground-truth and
predicted boxes associated with that query (many-to-
many max-IoU matching). Aggregate metrics include
mean IoU, IoU@0.3(fraction of queries with max IoU
≥0.3), IoU@0.5, and IoU@0.7. IoU@0.5is the primary
headline metric for grounding evaluation, following
common practice in object-detection pilots.
Pipeline status.The 16-query pipeline runs end-to-
end: ColPali produces three candidate boxes per query
from the sharpened MaxSim heatmap (48 predicted
boxes in total), annotation-helper overlays are ren-
dered for each query for reviewer inspection, and an
IoU scorer computes many-to-many max-IoU against
a ground-truth JSON. Pipeline artifacts—predicted
boxes, annotation helpers, GT template, and scoring
script—are released alongside the primary evaluation
code. The pilot is deliberately small: the primary
purpose is to establish that the MaxSim heatmap
produces spatially-localized boxes that overlap mean-
ingfully with human-identified answer regions, ratherthan to provide the sample size needed for per-category
grounding claims.
Ground-truth annotation status.At the time of sub-
mission, the GT-box JSON was seeded with the pre-
dicted boxes so that the IoU pipeline could be exercised
end-to-end; this configuration trivially yields IoU= 1.0
on every query and isnota meaningful measurement.
Human GT annotation by a domain-informed reviewer
is scoped as follow-up work, together with a larger-
scale IoU study on a fully annotated corpus. We report
this status explicitly rather than publishing a trivially
perfect IoU headline number, because the downstream
claim of interest—spatial localization quality—requires
an honest GT and is not materially supported by the
main results of this paper (which rely on Recall@5
and Judge Accuracy rather than pixel-level IoU). All
48 annotation-helper images are included in the sup-
plementary bundle to assist with visually assessing
localization quality prior to a formal IoU study.
H. Sample case figures
The following figures show high-resolution screen-
shots of the ColPali + Qwen2.5-VL-7B Visual-First
RAGpipelineappliedtorepresentativeWYDOTbench-
mark queries. Each card displays the input query (top-
left), the model’s response (top-right), and the top-3
retrievedstandard-planpageswiththeirMaxSimscores
(bottom). Success or failure is determined by direct
comparison of the model response to the benchmark
ground-truth answer.
Success Cases (Judge Score = 1.0)
Inthesecasestheretrieversurfacesthecorrectsheet
andtheVLMreadsthequerieddimensiondirectlyfrom
the drawing, matching the ground truth (judge score
1.0).
Failure Cases (Judge Score = 0.0)
Here retrieval succeeds but the VLM binds the
query to a neighboring dimension or conflates related
quantities—the two dominant error modes analyzed in
Section 7.
Rule Grounding and Real-Plan Auditing
Cases
This is the input to the autonomous rule-grounding
agent (Section 5.8): the governing limit sits in one row
among distractors, which the agent must retrieve and
extract before auditing the design value.
N. Subedi et al.:Preprint submitted to ElsevierPage 28 of 31

Visual-First Multimodal RAG for Civil Standard Plans
(a)Success– Dimensional accuracy. Q: vertical spacing of steel stakes at the centre of each enclosure (Plan 511-1A).
Predicted “4′-10′′[1.5 m]” matches the ground truth.
(b)Success– Dimensional accuracy. Q: maximum Moisture Density Control depth below the ditch bottom per Note H
(Plan 203-2A). Predicted “2 feet [600 mm]” matches the ground truth.
Fig. A.1:Representative dimensional-accuracy success cases on the WYDOT benchmark.
N. Subedi et al.:Preprint submitted to ElsevierPage 29 of 31

Visual-First Multimodal RAG for Civil Standard Plans
(a)Failure– Component–dimension binding. Q: height of the V-MESH END STRIPS in the EXPANDED VIEW (Plan
511-1A). Ground truth is 1′-4′′[0.4 m]; the model instead infers 4′-10′′[1.5 m] from a neighboring section, illustrating the
cross-view inference error described in Section 7.
(b)Failure–Semanticmisinterpretation.Q:specifieddepthforscarifyingthesurfaceintheFILLSECTION(Plan203-2A).
Ground truth is 6 inches [150 mm]; the model returns 4′-0′′[1200], a fill-geometry dimension, conflating “scarification depth”
with “fill thickness”.
Fig. A.2:Hard failures arising from component–dimension binding and semantic confusion.
N. Subedi et al.:Preprint submitted to ElsevierPage 30 of 31

Visual-First Multimodal RAG for Civil Standard Plans
Fig. A.3:A rendered Standard-Specifications reference sheet from the rule-grounding corpus. The seven governing limits for the
compliance archetypes (e.g., min. cover2.0′′, max. stirrup spacing𝑑/2) are embedded among distractor rows; the agent must
retrieve the correct sheet among 1,913 candidates and extract the governing limit from the table.
N. Subedi et al.:Preprint submitted to ElsevierPage 31 of 31