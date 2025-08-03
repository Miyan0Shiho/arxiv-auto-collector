# Zero-Shot Document Understanding using Pseudo Table of Contents-Guided Retrieval-Augmented Generation

**Authors**: Hyeon Seong Jeong, Sangwoo Jo, Byeong Hyun Yoon, Yoonseok Heo, Haedong Jeong, Taehoon Kim

**Published**: 2025-07-31 03:14:45

**PDF URL**: [http://arxiv.org/pdf/2507.23217v1](http://arxiv.org/pdf/2507.23217v1)

## Abstract
Understanding complex multimodal documents remains challenging due to their
structural inconsistencies and limited training data availability. We introduce
\textit{DocsRay}, a training-free document understanding system that integrates
pseudo Table of Contents (TOC) generation with hierarchical Retrieval-Augmented
Generation (RAG). Our approach leverages multimodal Large Language Models'
(LLMs) native capabilities to seamlessly process documents containing diverse
elements such as text, images, charts, and tables without requiring specialized
models or additional training. DocsRay's framework synergistically combines
three key techniques: (1) a semantic structuring module using prompt-based LLM
interactions to generate a hierarchical pseudo-TOC, (2) zero-shot multimodal
analysis that converts diverse document elements into unified, text-centric
representations using the inherent capabilities of multimodal LLMs, and (3) an
efficient two-stage hierarchical retrieval system that reduces retrieval
complexity from $O(N)$ to $O(S + k_1 \cdot N_s)$. Evaluated on documents
averaging 49.4 pages and 20,971 textual tokens, DocsRay reduced query latency
from 3.89 to 2.12 seconds, achieving a 45% efficiency improvement. On the
MMLongBench-Doc benchmark, DocsRay-Pro attains an accuracy of 64.7%,
substantially surpassing previous state-of-the-art results.

## Full Text


<!-- PDF content starts -->

Zero-Shot Document Understanding using Pseudo Table of Contents-Guided
Retrieval-Augmented Generation
Hyeon Seong Jeong Sangwoo Jo Byeong Hyun Yoon
Yoonseok Heo Haedong Jeong Taehoon Kim*
Sogang University
Abstract
Understanding complex multimodal documents remains chal-
lenging due to their structural inconsistencies and lim-
ited training data availability. We introduce DocsRay , a
training-free document understanding system that integrates
pseudo Table of Contents (TOC) generation with hierarchi-
cal Retrieval-Augmented Generation (RAG). Our approach
leverages multimodal Large Language Models’ (LLMs) na-
tive capabilities to seamlessly process documents containing
diverse elements such as text, images, charts, and tables with-
out requiring specialized models or additional training. Doc-
sRay’s framework synergistically combines three key tech-
niques: (1) a semantic structuring module using prompt-
based LLM interactions to generate a hierarchical pseudo-
TOC, (2) zero-shot multimodal analysis that converts diverse
document elements into unified, text-centric representations
using the inherent capabilities of multimodal LLMs, and (3)
an efficient two-stage hierarchical retrieval system that re-
duces retrieval complexity from O(N)toO(S+k1·Ns).
Evaluated on documents averaging 49.4 pages and 20,971
textual tokens, DocsRay reduced query latency from 3.89 to
2.12 seconds, achieving a 45% efficiency improvement. On
the MMLongBench-Doc benchmark, DocsRay-Pro attains an
accuracy of 64.7%, substantially surpassing previous state-of-
the-art results.
1 Introduction
Document understanding remains a critical challenge in nat-
ural language processing, particularly for complex docu-
ments containing heterogeneous content such as text, im-
ages, charts, tables, and technical diagrams. We address this
heterogeneity through text-centric representations generated
by multimodal LLMs rather than explicit spatial model-
ing. Consequently, tasks that hinge on absolute layout co-
ordinates or correlations between multiple simultaneous im-
ages fall beyond our current investigation. Recent advances
in multimodal large language models (LLMs) have shown
impressive capabilities in processing diverse content types,
yet most document retrieval systems fail to fully leverage
these capabilities, instead relying on traditional pipelines
with separate tools for OCR, table extraction, and image
analysis.
The fundamental challenge lies in the unstructured na-
ture of most real-world documents. Unlike well-formatted
*Correspondence to: taehoonkim@sogang.ac.krtextbooks or research papers with explicit tables of con-
tents, the majority of documents, from business reports
to technical manuals, lack clear structural markers. Ex-
isting retrieval-augmented generation (RAG) systems typi-
cally employ naive chunking strategies that split documents
into fixed-size segments, destroying semantic coherence and
leading to fragmented retrieval results. Moreover, traditional
approaches require extensive training on document-specific
datasets, making them impractical for the diverse document
types encountered in real applications.
We present DocsRay, a training-free document under-
standing system that demonstrates how the synergistic in-
tegration of existing techniques can create a novel, practical
solution exceeding the performance of its individual compo-
nents. While pseudo-TOC generation, multimodal process-
ing, and hierarchical retrieval are individually known, our
contribution lies in their careful orchestration into an end-
to-end system that requires zero training data or task-specific
fine-tuning. This training-free nature is crucial for practical
deployment, as it enables immediate application to diverse
document types without the resource-intensive data collec-
tion and model training required by existing approaches.
Our core innovation is a prompt-based pseudo Table of
Contents (TOC) generation algorithm that transforms un-
structured documents into intelligently organized hierar-
chies. Unlike traditional segmentation methods that rely on
formatting cues, fixed windows, or trained models, as shown
in Table 1, our approach leverages an LLM’s inherent se-
mantic understanding through carefully designed prompts.
This zero-shot structuring requires only two prompts: one
for boundary detection and another for title generation. De-
spite its simplicity, the method achieves document organi-
zations comparable to those produced by human annotators,
all without any need for model training.
DocsRay’s strength lies in the synergy of three compo-
nents: (1) prompt-based pseudo-TOC generation that se-
mantically structures documents without training; (2) zero-
shot multimodal understanding that unifies diverse content
through LLM-native processing; and (3) hierarchical re-
trieval leveraging the generated structure for efficient access.
Together, these components enable accurate understanding
and fast retrieval in a fully training-free system.
This design improves both accuracy and efficiency. Hi-
erarchical retrieval reduces computational complexity fromarXiv:2507.23217v1  [cs.LG]  31 Jul 2025

O(N)toO(S+k1·Ns), where S≪N, while se-
mantic structuring enhances precision by preserving topi-
cal coherence. Crucially, the system requires no training,
which allows immediate deployment across diverse lan-
guages, domains, and document types. This is a signifi-
cant advantage in scenarios where training data is scarce or
unavailable. DocsRay-Pro achieves 64.7% accuracy on the
MMLongBench-Doc (Ma et al. 2024) benchmark, marking
a 15.0 percentage point improvement over the strongest pre-
viously reported baseline, further validating the efficacy of
this training-free, structured retrieval approach.
2 Related Work
This section reviews five lines of research that collectively
address the challenges tackled by DocsRay: document visual
question answering, retrieval-augmented generation, docu-
ment structure inference, OCR-based and OCR-free under-
standing, and large multimodal foundation models.
2.1 Document Visual Question Answering
Early DocVQA systems assumed that a single page con-
tained all evidence. Recent benchmarks break this assump-
tion by requiring reasoning across multiple pages or slides.
SlideVQA introduces a 39 K-question dataset built from real
slide decks; each question is accompanied by ground-truth
evidence slides and answers (Tanaka et al. 2023). Models
must therefore retrieve relevant slides before answering, a
setting mirrored in other competitions such as the ICDAR
2023 Multi-Page DocVQA challenge (Tito, Karatzas, and
Valveny 2023). State-of-the-art DocVQA baselines typically
extract text with OCR, encode layout with transformers such
as LayoutLMv3 (Huang et al. 2022), then generate answers.
Yet they suffer when OCR is noisy or when information
spans long contexts, prompting the community to explore
retrieval and layout-aware reasoning.
2.2 Retrieval-Augmented Generation
RAG couples a retriever that selects external context with
a generator that conditions on that context (Lewis et al.
2020). For images and documents, several extensions have
emerged. RA VQA jointly trains a dense retriever and a VQA
decoder so that visual questions guide which passages to
fetch (Lin and Byrne 2022). REVEAL injects a learned
memory module and an attentive fusion layer, achieving new
state-of-the-art results on knowledge-based VQA bench-
marks (Hu et al. 2023). A line of work explores hierarchi-
calretrieval: Dense Hierarchical Retrieval first ranks docu-
ments, then passages within them, using heading informa-
tion where available (Liu et al. 2021); Hybrid Hierarchical
Retrieval combines sparse lexical features with dense em-
beddings to improve zero-shot robustness on open-domain
QA (Arivazhagan et al. 2023). The recent Multi-Level RAG
retrieves entities, expands queries, and finally retrieves pas-
sages, pushing knowledge-aware VQA performance on VI-
QuAE (Adjali et al. 2024). HiREC (Choe, Kim, and Jung
2025) demonstrates hierarchical retrieval in financial doc-
ument QA through document-level then passage-level re-
trieval on SEC filings.Method Semantic Hierarchical Multimodal
Aware Structure Support
Fixed-size Chunking × × ×
Sliding Window × × ×
Format-based ×√×
LayoutLM-based√× ×
LumberChunker√× ×
DocsRay (Ours)√ √ √
Table 1: Comparison of document segmentation techniques.
DocsRay uniquely combines semantic awareness, hierarchi-
cal structuring, and multimodal support in a training-free ap-
proach with automatic title generation.
2.3 Document Structure Extraction
When explicit headings exist (e.g. HTML or PDF book-
marks), two-stage retrieval greatly boosts recall (Liu et al.
2021). Real documents, however, are often scanned or lack
consistent metadata. LumberChunker shows that a large lan-
guage model (LLM) can infer semantic breakpoints, seg-
menting an entire novel into coherent chapters; plugging
these segments into a RAG pipeline raises answer recall by
more than 15 % (Duarte et al. 2024).
Table 1 compares document segmentation approaches,
highlighting how DocsRay’s pseudo-TOC generation ad-
vances beyond existing methods by combining semantic un-
derstanding with hierarchical structuring, all without requir-
ing any training data.
2.4 OCR-Based versus OCR-Free Approaches
The dominant paradigm extracts text with OCR, feeds the
tokens plus layout coordinates into a transformer (e.g. Lay-
outLMv3 (Huang et al. 2022), DocFormer (Appalaraju et al.
2021)), and applies task-specific heads. Accuracy hinges on
OCR quality; complex layouts and non-Latin scripts remain
challenging. OCR- free systems address these limitations.
Donut trains an encoder-decoder transformer end-to-end on
document images, directly predicting answers or structured
fields without explicit OCR extraction (Kim et al. 2022).
Google’s PaLI extends this concept to 70+ languages and di-
verse vision-language tasks (Chen et al. 2023). Microsoft’s
Kosmos-1 demonstrates ”OCR-free NLP” by prompting a
1.6 B parameter multimodal LLM to read text in the image
(Huang et al. 2023).
2.5 Large Multimodal Foundation Models
General-purpose multimodal LLMs have rapidly advanced.
Vision- language connectors such as BLIP-2 (Li et al. 2023),
encoder-decoder models like BEiT-3 (Wang et al. 2023), and
unified document transformers such as UDOP (Tang et al.
2023) achieve broad transfer across captioning, VQA, and
document AI tasks via prompting. These models provide the
backbone for zero-shot applications; yet their context win-
dows remain finite. Consequently, hierarchical retrieval is
critical when documents exceed several thousand tokens.

Multimodal LLM Input Documents
.pdf .doc
.ppt .jpg
.hwp⋯Boundary
DetectionSection 
RefinementTitle 
Generation
Chunking
StrategyMergeSummaryStage 1: Document Chunking and Indexing for Stage 2
Stage 2: Query Refinement and Retrieval Augmented Generation
Dual
Embedding
Sliding
Enc. 2Enc. 1
Concatenate & NormalizeUser Query
Section 
Vectors
Chunk 
VectorsPseudo -TOC Generation
Chunking and Dual Embedding Architecture Hierarchical Retrieval
Answer GenerationVisual Analysis
+
OCR
(optional,
max 2 times)
Fine SearchQuery
RefinementCoarse Search Multimodal LLMFigure 1: Simplified DocsRay architecture with two distinct stages. Stage 1 (Document Processing) handles input document
parsing and pseudo-TOC generation using a multimodal LLM. Stage 2 (Query Processing) performs hierarchical retrieval with
optional query refinement iterations. Query Refinement & Answer Generation perform Retrieval Augmented Generation (RAG)
based on retrieved chunks. Both stages utilize the same LLM for unified processing.
3 Methodology
3.1 Overview
DocsRay is a training-free document understanding system
integrating three key components: (1) multimodal analy-
sis and pseudo-TOC generation using LLMs, (2) semantic
chunking with dual embeddings, and (3) coarse-to-fine hi-
erarchical search with query refinement. Figure 1 illustrates
the architecture.
3.2 Semantic Document Analysis and Structuring
Modern documents typically include diverse content such
as plain text, tables, vector diagrams, and images. Instead
of relying on external tools or modality-specific models, we
utilize the native multimodal capabilities of large language
models (LLMs). This unified approach simultaneously pro-
cesses content and constructs hierarchical structure using
prompt-based methods.
LLM-Native Content Processing We prompt the LLM
to directly analyze each content type within the document.
Multi-column layouts are resolved via spatial clustering to
infer natural reading order. Tables are rendered as images to
maintain structure and semantics for layout-aware LLM pro-
cessing. Figures and charts are processed through descrip-
tive prompts, generating captions to support unified retrieval
across modalities. When standard text extraction fails due
to complex formatting or embedded text, the LLM’s vision
capability extracts structured text while preserving original
meaning. Further implementation details are provided in the
Appendix A.Prompt-Based Structure Generation Our approach gen-
erates a pseudo–Table of Contents (pseudo-TOC) based on
semantic understanding rather than formatting cues such as
headings or font sizes. This involves three phases: (1) seg-
menting the document by detecting semantic boundaries us-
ing LLM prompts; (2) merging smaller segments into co-
herent sections based on topical similarity and length con-
straints; and (3) generating section titles by sampling rep-
resentative content from each group. As this method relies
on semantic coherence rather than layout features, it adapts
flexibly across diverse document styles and languages. The
complete procedure, including prompts and segmentation
heuristics, is detailed in the Appendix C.
3.3 Chunking Strategy and Dual Embedding
After obtaining the document structure from the pseudo-
TOC, we segment each section into text chunks and encode
them using dual embeddings.
Chunking Strategy Our chunking strategy balances con-
textual completeness, retrieval efficiency, and structural
alignment. We apply a sliding window mechanism to pro-
duce chunks of roughly 500–600 tokens, providing sufficient
context while limiting computational costs. Adjacent chunks
slightly overlap to minimize information loss at boundaries.
Crucially, chunk boundaries align with pseudo-TOC sec-
tions for semantic coherence.
Dual Embedding Architecture To build a dual embed-
ding system, we evaluate several pre-trained sentence em-
bedding models on the CrossLingualSemanticDiscrimina-
tionWMT21 task (Muennighoff et al. 2022; Enevoldsen

Figure 2: 2D UMAP visualization of embeddings, where
each point corresponds to a sample from the cross-retrieval
task and colors indicate different embedding models.
et al. 2025). For dimensional consistency, we focus on can-
didate models producing embeddings in R1024. Specifically,
we compute embeddings for each candidate model as fol-
lows:
ei
model, i∈ {0,1,2, . . . , N −1}
where N= 8931 is the number of samples.
We then visualize these embeddings using UMAP for
qualitative comparison. As shown in Figure 2, BGE-M3
(Chen et al. 2024) exhibits a clearly separated and discrim-
inative distribution, indicating its suitability as the primary
embedding.
To complement this base model, we calculate pairwise
cosine similarities among candidate embeddings to iden-
tify a second embedding that is sufficiently distinct yet not
entirely orthogonal. We hypothesize that moderately corre-
lated embeddings produce more coherent fused represen-
tations compared to completely uncorrelated embeddings.
Figure 3 shows that BGE-M3 shares moderate similarity
with Multilingual-E5-Large (Wang et al. 2024), satisfying
this criterion. Thus, we select BGE-M3 and Multilingual-
E5-Large as components for our dual embedding system.
The combined embedding is constructed by concatenating
outputs from both models followed by L2 normalization:
ecombined =normalize (emodel1∥emodel2 )
This design preserves semantic richness from both em-
beddings and ensures consistent vector norms for similarity-
based retrieval. As our architecture is model-agnostic, it can
easily incorporate alternative embeddings depending on spe-
cific language, domain, or application requirements.
(a)
(b)
(c)
(d)
(e)
(f)
(g)
(a) (b) (c) (d) (e) (f) (g)
(a) bge -m3(b) e5 -large (c) e5 -large -v2(d) multilingual -e5-large
(e) gte -large -en-v1.5 (f) nli-bert-large (g) all -roberta -large -v1Figure 3: Averaged cosine similarity map of embedding
models on the cross-retrieval task. While most embeddings
are distinct, some pairs show moderate similarity.
3.4 Pseudo-TOC-Based Hierarchical Retrieval
Our hierarchical retrieval pipeline leverages the pseudo-
TOC for efficient and accurate retrieval through a two-stage
process. Figure 4 illustrates this coarse-to-fine approach.
Section Representation Each section is represented by
two embeddings: a title embedding capturing high-level se-
mantics of the section title,
etitle =DualEmbed (section.title )
and a content embedding obtained by averaging all chunk
embeddings within the section,
econtent =1
|Cs|X
c∈Csec,
where Csdenotes the chunks in section s. These represen-
tations are precomputed during indexing to enable efficient
retrieval.
Coarse Search In the first stage, the query is matched
against section representations. Similarities are computed
separately for title and content embeddings, then combined
using weighted interpolation:
ssection =β·cos(eq, etitle) + (1 −β)·cos(eq, econtent )
Here, eqis the query embedding, and βbalances the
two similarity sources. Higher βemphasizes title semantics,
beneficial for documents with descriptive headings, while
lower βallows content similarity to dominate when titles
are generic or uninformative.
Fine Search In the second stage, retrieval is performed
within chunks from top-ranked sections identified by the
coarse search. The query is compared against chunk embed-
dings, and the highest-scoring chunks are returned. By limit-
ing the search scope, this stage improves retrieval speed and
relevance while preserving the quality of retrieved content.

Document
(Nchunks)
with pseudo-TOC
(Ssections)Query: “Revenue
growth in Asia”
Coarse Search
(Sections)
Section 3 Section 7 Section 9
Fine Search
(Chunks)
Chunk 7.3 Chunk 9.1 Chunk 9.4
Retrieved Context
Refined Query:
+ “Asian
market metrics”k1Ns
Figure 4: Two-stage coarse-to-fine retrieval with query re-
finement. Given a query, the system first identifies relevant
sections (coarse search), then retrieves chunks within these
sections (fine search). As k1·Ns≪N, this reduces com-
plexity significantly. Retrieved contexts iteratively refine the
query, improving accuracy.
Efficiency Analysis LetNbe the total number of chunks,
Sthe number of sections, and Nsthe average number of
chunks per section. A naive flat retrieval requires O(N)
computations, while our hierarchical approach reduces this
toO(S+k1·Ns), where k1is the number of selected sec-
tions. Since S≪Nandk1·Ns≪N, computation is sig-
nificantly reduced. The memory overhead of storing O(S)
section embeddings is negligible compared to the retrieval
efficiency gained.
Iterative Query Refinement We improve retrieval
through iterative query refinement. After an initial retrieval:
R0=Retrieve (q0)
the LLM analyzes the results, identifies information gaps,
and generates a refined query. The augmented query is con-
structed as:
q1=q0+“: ‘’ +qrefined
This preserves the original intent while adding specificity.
We empirically limit iterations to two due to diminishing
returns.
Source Section Attribution DocsRay maintains metadata
about retrieved content throughout the retrieval pipeline, en-
abling source section attribution. The system lists sections
used as context for the final answer in a “References” sec-
tion, providing transparency about consulted document sec-
tions.Model Accuracy(%)
DocsRay Variants
DocsRay-Pro (27B) 64.7
DocsRay-Base (12B) 62.8
DocsRay-Lite (4B) 31.8
Large Visual Language Models (LVLMs)
GPT-4.1 (OpenAI 2025) 49.7
GPT-4o (OpenAI 2024) 46.3
GLM-4.1V-Thinking (Team et al. 2025b) 42.4
Kimi-VL-Thinking (Team et al. 2025c) 42.1
Qwen2.5-VL-72B (Bai et al. 2023) 35.2
Kimi-VL-A3B-Instruct (Team et al. 2025c) 35.1
MiniMax-VL-01 (MiniMax et al. 2025) 32.5
Aria (Li et al. 2024) 28.3
Gemini-1.5-Pro (Reid et al. 2024) 28.2
Qwen2.5-VL-7B (Bai et al. 2023) 25.1
OCR + Large Language Models (LLMs)
Gemini-1.5-Pro (Reid et al. 2024) 31.2
GPT-4o (OpenAI 2024) 30.1
GPT-4-turbo (OpenAI 2023) 27.6
Mixtral-Instruct-v0.1 (Jiang et al. 2024) 26.9
Claude-3 Opus (Anthropic 2024) 26.9
DeepSeek-V2 (DeepSeek 2024) 24.9
Human Performance
Human Expert 65.8
Table 2: Performance comparison on MMLongBench-Doc.
Accuracy comparison on MMLongBench-Doc. DocsRay-
Pro achieves the highest performance among automated sys-
tems, approaching human-level accuracy.
4 Experiments
We evaluate DocsRay through benchmark evaluation on
MMLongBench-Doc, qualitative case studies, and ablation
studies.
4.1 Results on MMLongBench-Doc
MMLongBench-Doc evaluates the ability of models to rea-
son over complex, multi-page documents containing visual
elements (Ma et al. 2024). Table 2 compares our results with
current state-of-the-art methods.
DocsRay-Pro achieves 64.7% accuracy, outperforming
the best LVLM baseline (GPT-4.1, 49.7%) by 15.0 points
and the strongest OCR+LLM pipeline (Gemini-1.5-Pro,
31.2%) by 33.5 points. This significant improvement high-
lights the effectiveness of hierarchical retrieval and pseudo-
TOC generation.
The performance differences across DocsRay variants il-
lustrate the impact of model scale. All variants use the
Gemma-3 family (Team et al. 2025a) without additional
training: Pro (27B), Base (12B), and Lite (4B). Notably,
even the Lite variant surpasses several leading LVLMs, con-
firming the strength of our system design.
Furthermore, DocsRay-Pro’s accuracy (64.7%) closely
approaches the human expert baseline (65.8%), demonstrat-
ing its capability to capture essential human document com-
prehension skills: structured organization, retrieval-based at-
tention, and multimodal understanding.

Pattern Description Example Key Model Behavior (DocsRay-Pro)
Single-Source Fact
RetrievalTasks where the answer is con-
tained entirely on a single pagemmlongbench 13 Correctly retrieves page 2 and extracts
the precise location
Multi-Page Evidence
SynthesisTasks requiring aggregation of facts
across multiple, non-contiguous
pagesmmlongbench 36 Correctly identifies all 5 quotes and
their respective page numbers
Evidence Attribution
FailuresTasks where the document lacks
sufficient information to form an
answermmlongbench 8 Correctly identifies that data for 2024 is
absent and avoids hallucination
Table 3: Analysis of Evidence Grounding and Scaling Behavior. Hierarchical search ensures systematic coverage, preventing
missed evidence common in flat retrieval.
4.2 Qualitative Analysis of Evidence Grounding
We conducted a detailed analysis of representative cases
from MMLongBench-Doc to understand DocsRay’s evi-
dence grounding capabilities and scaling behavior. Through-
out this section, we refer to specific test cases (e.g., mmlong-
bench 13), and the corresponding evidence summaries are
presented in Appendix F.
Single-Source Fact Retrieval For straightforward factual
questions like “Where was Gestalt psychology conceived?”
(mmlongbench 13), DocsRay effectively retrieves the spe-
cific page containing “Berlin School of Experimental Psy-
chology.” This demonstrates the system’s precision in local-
ized fact retrieval where evidence is unambiguous.
Multi-Page Evidence Synthesis More challenging cases
require aggregating information across multiple pages. The
question “How many human quotes with sources?” (mm-
longbench 36) necessitates scanning pages 14, 19, 20, 33,
and 37 to identify all quoted individuals. DocsRay’s hierar-
chical retrieval excels here by ensuring systematic section
coverage rather than relying on keyword matching alone,
preventing the missed evidence common in flat retrieval ap-
proaches.
Evidence Attribution Failures The most instructive
cases involve questions where evidence is absent. When
asked about “Democrats voting percentage in 2024?” (mm-
longbench 8), the document only contains data through
2018. DocsRay correctly identifies this limitation rather than
hallucinating an answer, demonstrating robustness against
unanswerable questions, a key feature for trustworthy AI.
Our scaling analysis across model sizes reveals that
while all models handle simple fact retrieval effectively,
complex multi-page synthesis and visual understanding ca-
pabilities strongly correlate with model scale. The Pro
model maintains coherent tracking across extended con-
texts, while smaller models struggle with evidence aggre-
gation tasks. For a comprehensive analysis of additional ev-
idence grounding patterns including statistical evidence re-
quirements (mmlongbench 10), visual evidence interpreta-
tion (mmlongbench 20, mmlongbench 26), and cases with
ambiguous evidence (mmlongbench 9, mmlongbench 18),
please refer to Table 7 in the Appendix F.Task Type Pro Base Lite
Tier 1: Simple Fact Retrieval
Where was Gestalt psychology con-
ceived?√ √ √
What year is the report for?√ √ √
Republican Hispanic vs no leans male√ √ √
Tier 2: Complex Reasoning
Define law of good gestalt (technical)√ √×
Count human quotes (multi-page)√ √×
5% support rate analysis√ √×
Tier 3: Advanced Synthesis
Count exterior photos (10+ pages)√× ×
Count hand-drawn cartoons (visual)√× ×
Identify closure principle shapes√× ×
Table 4: Performance across model scales reveals three ca-
pability tiers. All models handle simple retrieval, but only
larger models excel at complex multi-page synthesis and vi-
sual understanding tasks.
4.3 Comparative Analysis Across Model Scales
To better understand the impact of model scaling on doc-
ument understanding capabilities, we conducted a detailed
comparative analysis using three variants of DocsRay. By
examining performance differences across simple fact re-
trieval, multi-page synthesis, and visual understanding tasks,
we can identify which capabilities emerge or improve with
increased model scale.
Simple Fact Retrieval All model sizes successfully han-
dle straightforward factual questions requiring single-page
evidence. Even the Lite model correctly retrieves specific
facts like “Berlin School of Experimental Psychology,” val-
idating our hierarchical retrieval architecture’s effectiveness
across scales.
Complex Reasoning The Base and Pro models demon-
strate superior performance on technical comprehension and
multi-step reasoning. For instance, when defining “law of
good gestalt” (mmlongbench 16), larger models provide ac-
curate technical definitions while the Lite model produces
garbled responses mixing unrelated concepts. Additional ex-
amples of complex reasoning tasks, including the 5% sup-
port rate analysis (mmlongbench 9), are in Table 8 in Ap-
pendix F.

Advanced Multi-Page Synthesis Only the Pro model
consistently succeeds at comprehensive document analysis
across many pages. When counting exterior photos (mm-
longbench 26) scattered across pages 10-20, the Pro model
correctly identifies all 10 photos, the Base model finds only
6, and the Lite model identifies just 3. This suggests larger
models with longer context length maintain better working
memory for tracking evidence across extended contexts. The
Table 8 in Appendix F provides detailed case studies on ad-
vanced visual analysis tasks, including counting hand-drawn
cartoons (mmlongbench 48) and identifying shapes in dia-
grams (mmlongbench 19).
Our analysis also revealed several instructive edge cases
that demonstrate unexpected scaling behaviors and nu-
anced differences in model capabilities. We provide detailed
case studies in the Appendix F.3, including examples of
non-monotonic scaling where smaller models outperform
larger ones, hallucination patterns across model scales, and
the emergence of precise numerical reasoning capabilities.
These edge cases, presented with actual model responses in
the Appendix, have significant implications for deployment
strategies, suggesting a portfolio approach using different
model scales for different query types could provide optimal
cost-performance trade-offs.
4.4 Ablation Studies
We conduct ablation studies to analyze the individual con-
tributions of pseudo-TOC generation and dual embedding
architecture.
Impact of Pseudo-TOC Generation Table 5 evaluates
the tradeoff between retrieval accuracy and processing ef-
ficiency. While the pseudo-TOC approach shows a marginal
0.7 percentage point decrease in accuracy (62.8% vs.
63.5%), it significantly improves efficiency, reducing Stage
2 query time from 3.89 to 2.12 seconds (45.4% speedup).
This demonstrates that hierarchical retrieval successfully re-
duces the search space while maintaining competitive accu-
racy. Most errors stem from misidentified section boundaries
or generic titles like “Introduction” and “Results”, which
lack discriminative power during coarse retrieval.
The minor accuracy degradation occurs when the coarse
retrieval stage occasionally fails to identify sections contain-
ing relevant information. In these cases, even perfect fine-
grained retrieval cannot recover the pruned content. This
highlights a trade-off between efficiency and recall, which
may be mitigated through adaptive thresholds or query-
aware section selection.
Dual Embedding Architecture Analysis Table 6 reveals
striking performance differences across embedding configu-
rations. Individual models achieve moderate performance:
BGE-M3 at 54.0% and Multilingual-E5-Large at 54.7%.
Our analysis suggests BGE-M3 excels at keyword-based re-
trieval but may miss semantically related content using dif-
ferent terminology, while E5-Large provides stronger se-
mantic understanding but potentially weaker exact-match
capabilities.
The dual embedding with concatenation achieves 62.8%
accuracy, an 8–9 percentage point improvement over indi-Configuration Accuracy (%) Avg. Time (s)
with Pseudo-TOC 62.8 2.12
w/o Pseudo-TOC 63.5 3.89
Table 5: Ablation study on pseudo-TOC generation using
DocsRay-Base on MMLongBench-Doc. The pseudo-TOC
approach trades minimal accuracy for substantial process-
ing speed improvements. Avg. Time measures Stage 2 query
processing only (after document chunking and indexing are
complete).
Embedding Configuration Accuracy (%)
BGE-M3 only 54.0
Multilingual-E5-Large only 54.7
Dual Embedding (addition) 52.3
Dual Embedding (concatenation) 62.8
Table 6: Ablation study on embedding configurations us-
ing DocsRay-Base on MMLongBench-Doc. Dual embed-
ding with concatenation achieves the best performance, sig-
nificantly outperforming individual models.
vidual models. This supports our hypothesis that combining
complementary embeddings better captures lexical and se-
mantic cues. By preserving full dimensionality from both
models, concatenation allows the retrieval system to lever-
age keyword matching and semantic understanding simulta-
neously.
While concatenation increases the embedding dimension-
ality from 1024 to 2048, the accuracy improvement of 10.5
percentage points over addition and more than 8 points over
individual models makes the additional computational cost
worthwhile. In settings where memory is limited, using a
single embedding model is still a viable choice. However,
for applications that require higher accuracy, dual embed-
ding with concatenation is recommended.
5 Conclusion
DocsRay integrates pseudo-TOC generation, hierarchical re-
trieval, and multimodal reasoning into a training-free sys-
tem that achieves 64.7% accuracy on MMLongBench-Doc.
This performance approaches that of human experts and sur-
passes existing automated systems without requiring task-
specific training. Our results demonstrate that careful sys-
tem design and prompt-based structuring effectively lever-
age latent reasoning capabilities of large language mod-
els, suggesting a practical AI development approach that
emphasizes orchestration over scale. Future work includes
developing a multilingual document QA benchmark cov-
ering diverse languages and domains, exploring advanced
embedding fusion strategies to optimize accuracy and effi-
ciency, and integrating explicit evidence grounding and cita-
tion mechanisms to enhance system transparency and trust-
worthiness.

References
Adjali, O.; Ferret, O.; Ghannay, S.; and Le Borgne, H. 2024.
Multi-Level Information Retrieval Augmented Generation
for Knowledge-based Visual Question Answering. In Al-
Onaizan, Y .; Bansal, M.; and Chen, Y .-N., eds., Proceedings
of the 2024 Conference on Empirical Methods in Natural
Language Processing , 16499–16513. Miami, Florida, USA:
Association for Computational Linguistics.
Anthropic. 2024. The Claude 3 Model Family: Opus, Son-
net, Haiku. Anthropic Technical Report .
Appalaraju, S.; Jasani, B.; Kota, B. U.; Xie, Y .; and Man-
matha, R. 2021. DocFormer: End-to-End Transformer for
Document Understanding. In 2021 IEEE/CVF International
Conference on Computer Vision (ICCV) , 973–983.
Arivazhagan, M. G.; Liu, L.; Qi, P.; Chen, X.; Wang, W. Y .;
and Huang, Z. 2023. Hybrid Hierarchical Retrieval for
Open-Domain Question Answering. In Rogers, A.; Boyd-
Graber, J.; and Okazaki, N., eds., Findings of the Association
for Computational Linguistics: ACL 2023 , 10680–10689.
Toronto, Canada: Association for Computational Linguis-
tics.
Bai, J.; Bai, S.; Yang, S.; Wang, S.; Tan, S.; Wang, P.; Lin,
J.; Zhou, C.; and Zhou, J. 2023. Qwen-VL: A Versatile
Vision-Language Model for Understanding, Localization,
Text Reading, and Beyond. arXiv:2308.12966.
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and
Liu, Z. 2024. BGE M3-Embedding: Multi-Lingual, Multi-
Functionality, Multi-Granularity Text Embeddings Through
Self-Knowledge Distillation. arXiv:2402.03216.
Chen, X.; Wang, X.; Changpinyo, S.; Piergiovanni, A.;
Padlewski, P.; Salz, D.; Goodman, S.; Grycner, A.; Mustafa,
B.; Beyer, L.; Kolesnikov, A.; Puigcerver, J.; Ding, N.;
Rong, K.; Akbari, H.; Mishra, G.; Xue, L.; Thapliyal, A. V .;
Bradbury, J.; Kuo, W.; Seyedhosseini, M.; Jia, C.; Ayan,
B. K.; Ruiz, C. R.; Steiner, A. P.; Angelova, A.; Zhai, X.;
Houlsby, N.; and Soricut, R. 2023. PaLI: A Jointly-Scaled
Multilingual Language-Image Model. In The Eleventh In-
ternational Conference on Learning Representations .
Choe, J.; Kim, J.; and Jung, W. 2025. Hierarchical Retrieval
with Evidence Curation for Open-Domain Financial Ques-
tion Answering on Standardized Documents. In Findings of
the Association for Computational Linguistics: ACL 2025 ,
16663–16681. Vienna, Austria: Association for Computa-
tional Linguistics.
DeepSeek. 2024. DeepSeek-V2: A Strong, Economical,
and Efficient Mixture-of-Experts Language Model. arXiv
preprint arXiv:2405.04434 .
Duarte, A. V .; Marques, J. D.; Grac ¸a, M.; Freire, M.; Li, L.;
and Oliveira, A. L. 2024. LumberChunker: Long-Form Nar-
rative Document Segmentation. In Al-Onaizan, Y .; Bansal,
M.; and Chen, Y .-N., eds., Findings of the Association for
Computational Linguistics: EMNLP 2024 , 6473–6486. Mi-
ami, Florida, USA: Association for Computational Linguis-
tics.
Enevoldsen, K.; Chung, I.; Kerboua, I.; Kardos, M.; Mathur,
A.; Stap, D.; Gala, J.; Siblini, W.; Krzemi ´nski, D.; Winata,G. I.; et al. 2025. Mmteb: Massive multilingual text embed-
ding benchmark. arXiv preprint arXiv:2502.13595 .
Hu, Z.; Iscen, A.; Sun, C.; Wang, Z.; Chang, K.-W.; Sun,
Y .; Schmid, C.; Ross, D. A.; and Fathi, A. 2023. Re-
veal: Retrieval-Augmented Visual-Language Pre-Training
with Multi-Source Multimodal Knowledge Memory. In
2023 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR) , 23369–23379.
Huang, S.; Dong, L.; Wang, W.; Hao, Y .; Singhal, S.; Ma,
S.; Lv, T.; Cui, L.; Mohammed, O. K.; Patra, B.; Liu, Q.;
Aggarwal, K.; Chi, Z.; Bjorck, N.; Chaudhary, V .; Som, S.;
SONG, X.; and Wei, F. 2023. Language Is Not All You
Need: Aligning Perception with Language Models. In Oh,
A.; Naumann, T.; Globerson, A.; Saenko, K.; Hardt, M.; and
Levine, S., eds., Advances in Neural Information Processing
Systems , volume 36, 72096–72109. Curran Associates, Inc.
Huang, Y .; Lv, T.; Cui, L.; Lu, Y .; and Wei, F. 2022. Lay-
outLMv3: Pre-training for Document AI with Unified Text
and Image Masking. In Proceedings of the 30th ACM Inter-
national Conference on Multimedia , MM ’22, 4083–4091.
New York, NY , USA: Association for Computing Machin-
ery. ISBN 9781450392037.
Jiang, A. Q.; Sablayrolles, A.; Roux, A.; Mensch, A.;
Savary, B.; Bamford, C.; Chaplot, D. S.; de las Casas, D.;
Hanna, E. B.; Bressand, F.; Lengyel, G.; Bour, G.; Lample,
G.; Lavaud, L. R.; Saulnier, L.; Lachaux, M.-A.; Stock, P.;
Subramanian, S.; Yang, S.; Antoniak, S.; Scao, T. L.; Gervet,
T.; Lavril, T.; Wang, T.; Lacroix, T.; and Sayed, W. E. 2024.
Mixtral of Experts. arXiv:2401.04088.
Kim, G.; Hong, T.; Yim, M.; Nam, J.; Park, J.; Yim, J.;
Hwang, W.; Yun, S.; Han, D.; and Park, S. 2022. OCR-Free
Document Understanding Transformer. In Avidan, S.; Bros-
tow, G.; Ciss ´e, M.; Farinella, G. M.; and Hassner, T., eds.,
Computer Vision – ECCV 2022 , 498–517. Cham: Springer
Nature Switzerland. ISBN 978-3-031-19815-1.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V .;
Goyal, N.; K ¨uttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨aschel,
T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In
Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M.; and
Lin, H., eds., Advances in Neural Information Processing
Systems , volume 33, 9459–9474. Curran Associates, Inc.
Li, D.; Liu, Y .; Wu, H.; Wang, Y .; Shen, Z.; Qu, B.; Niu,
X.; Zhou, F.; Huang, C.; Li, Y .; Zhu, C.; Ren, X.; Li, C.;
Ye, Y .; Liu, P.; Zhang, L.; Yan, H.; Wang, G.; Chen, B.; and
Li, J. 2024. Aria: An Open Multimodal Native Mixture-of-
Experts Model. arXiv:2410.05993.
Li, J.; Li, D.; Savarese, S.; and Hoi, S. 2023. BLIP-2:
bootstrapping language-image pre-training with frozen im-
age encoders and large language models. In Proceedings
of the 40th International Conference on Machine Learning ,
ICML’23. JMLR.org.
Lin, W.; and Byrne, B. 2022. Retrieval Augmented Visual
Question Answering with Outside Knowledge. In Gold-
berg, Y .; Kozareva, Z.; and Zhang, Y ., eds., Proceedings of
the 2022 Conference on Empirical Methods in Natural Lan-

guage Processing , 11238–11254. Abu Dhabi, United Arab
Emirates: Association for Computational Linguistics.
Liu, Y .; Hashimoto, K.; Zhou, Y .; Yavuz, S.; Xiong, C.; and
Yu, P. S. 2021. Dense Hierarchical Retrieval for Open-
Domain Question Answering. arXiv:2110.15439.
Ma, Y .; Zang, Y .; Chen, L.; Chen, M.; Jiao, Y .; Li, X.; Lu, X.;
Liu, Z.; Ma, Y .; Dong, X.; Zhang, P.; Pan, L.; Jiang, Y .-G.;
Wang, J.; Cao, Y .; and Sun, A. 2024. MMLONGBENCH-
DOC: Benchmarking Long-context Document Understand-
ing with Visualizations. In Globerson, A.; Mackey, L.; Bel-
grave, D.; Fan, A.; Paquet, U.; Tomczak, J.; and Zhang, C.,
eds., Advances in Neural Information Processing Systems ,
volume 37, 95963–96010. Curran Associates, Inc.
MiniMax; Li, A.; Gong, B.; Yang, B.; Shan, B.; Liu, C.; Zhu,
C.; Zhang, C.; Guo, C.; Chen, D.; Li, D.; Jiao, E.; Li, G.;
Zhang, G.; Sun, H.; Dong, H.; Zhu, J.; Zhuang, J.; Song,
J.; Zhu, J.; Han, J.; Li, J.; Xie, J.; Xu, J.; Yan, J.; Zhang,
K.; Xiao, K.; Kang, K.; Han, L.; Wang, L.; Yu, L.; Feng,
L.; Zheng, L.; Chai, L.; Xing, L.; Ju, M.; Chi, M.; Zhang,
M.; Huang, P.; Niu, P.; Li, P.; Zhao, P.; Yang, Q.; Xu, Q.;
Wang, Q.; Wang, Q.; Li, Q.; Leng, R.; Shi, S.; Yu, S.; Li,
S.; Zhu, S.; Huang, T.; Liang, T.; Sun, W.; Sun, W.; Cheng,
W.; Li, W.; Song, X.; Su, X.; Han, X.; Zhang, X.; Hou, X.;
Min, X.; Zou, X.; Shen, X.; Gong, Y .; Zhu, Y .; Zhou, Y .;
Zhong, Y .; Hu, Y .; Fan, Y .; Yu, Y .; Yang, Y .; Li, Y .; Huang,
Y .; Li, Y .; Huang, Y .; Xu, Y .; Mao, Y .; Li, Z.; Li, Z.; Tao,
Z.; Ying, Z.; Cong, Z.; Qin, Z.; Fan, Z.; Yu, Z.; Jiang, Z.;
and Wu, Z. 2025. MiniMax-01: Scaling Foundation Models
with Lightning Attention. arXiv:2501.08313.
Muennighoff, N.; Tazi, N.; Magne, L.; and Reimers, N.
2022. Mteb: Massive text embedding benchmark. arXiv
preprint arXiv:2210.07316 .
OpenAI. 2023. GPT-4 Turbo: Enhanced Speed and Capabil-
ities. Technical Report .
OpenAI. 2024. GPT-4o System Card. arXiv preprint
arXiv:2410.21276 .
OpenAI. 2025. GPT-4.1 System Card.
Reid, M.; Savinov, N.; Teplyashin, D.; Lepikhin, D.; Lilli-
crap, T.; baptiste Alayrac, J.; Soricut, R.; Lazaridou, A.; Fi-
rat, O.; Schrittwieser, J.; Antonoglou, I.; Anil, R.; Borgeaud,
S.; Dai, A.; Millican, K.; Dyer, E.; Morris, M. R.; John-
son, M.; tau Yih, W.; Thoppilan, R.; Le, Q. V .; Wu, Y .;
Chen, Z.; and Dean, J. 2024. Gemini 1.5: Unlocking mul-
timodal understanding across millions of tokens of context.
arXiv:2403.05530.
Tanaka, R.; Nishida, K.; Nishida, K.; Hasegawa, T.; Saito,
I.; and Saito, K. 2023. SlideVQA: A Dataset for Doc-
ument Visual Question Answering on Multiple Images.
arXiv:2301.04883.
Tang, Z.; Yang, Z.; Wang, G.; Fang, Y .; Liu, Y .; Zhu, C.;
Zeng, M.; Zhang, C.; and Bansal, M. 2023. Unifying Vi-
sion, Text, and Layout for Universal Document Processing.
In2023 IEEE/CVF Conference on Computer Vision and Pat-
tern Recognition (CVPR) , 19254–19264.
Team, G.; Kamath, A.; Ferret, J.; Pathak, S.; Vieillard, N.;
Merhej, R.; Perrin, S.; Matejovicova, T.; Ram ´e, A.; Rivi `ere,
M.; Rouillard, L.; Mesnard, T.; Cideron, G.; bastien Grill,J.; Ramos, S.; Yvinec, E.; Casbon, M.; Pot, E.; Penchev, I.;
Liu, G.; Visin, F.; Kenealy, K.; Beyer, L.; Zhai, X.; Tsit-
sulin, A.; Busa-Fekete, R.; Feng, A.; Sachdeva, N.; Cole-
man, B.; Gao, Y .; Mustafa, B.; Barr, I.; Parisotto, E.; Tian,
D.; Eyal, M.; Cherry, C.; Peter, J.-T.; Sinopalnikov, D.; Bhu-
patiraju, S.; Agarwal, R.; Kazemi, M.; Malkin, D.; Kumar,
R.; Vilar, D.; Brusilovsky, I.; Luo, J.; Steiner, A.; Friesen,
A.; Sharma, A.; Sharma, A.; Gilady, A. M.; Goedeckemeyer,
A.; Saade, A.; Feng, A.; Kolesnikov, A.; Bendebury, A.;
Abdagic, A.; Vadi, A.; Gy ¨orgy, A.; Pinto, A. S.; Das, A.;
Bapna, A.; Miech, A.; Yang, A.; Paterson, A.; Shenoy, A.;
Chakrabarti, A.; Piot, B.; Wu, B.; Shahriari, B.; Petrini, B.;
Chen, C.; Lan, C. L.; Choquette-Choo, C. A.; Carey, C.;
Brick, C.; Deutsch, D.; Eisenbud, D.; Cattle, D.; Cheng,
D.; Paparas, D.; Sreepathihalli, D. S.; Reid, D.; Tran, D.;
Zelle, D.; Noland, E.; Huizenga, E.; Kharitonov, E.; Liu,
F.; Amirkhanyan, G.; Cameron, G.; Hashemi, H.; Klimczak-
Pluci ´nska, H.; Singh, H.; Mehta, H.; Lehri, H. T.; Hazimeh,
H.; Ballantyne, I.; Szpektor, I.; Nardini, I.; Pouget-Abadie,
J.; Chan, J.; Stanton, J.; Wieting, J.; Lai, J.; Orbay, J.; Fer-
nandez, J.; Newlan, J.; yeong Ji, J.; Singh, J.; Black, K.;
Yu, K.; Hui, K.; V odrahalli, K.; Greff, K.; Qiu, L.; Valen-
tine, M.; Coelho, M.; Ritter, M.; Hoffman, M.; Watson, M.;
Chaturvedi, M.; Moynihan, M.; Ma, M.; Babar, N.; Noy, N.;
Byrd, N.; Roy, N.; Momchev, N.; Chauhan, N.; Sachdeva,
N.; Bunyan, O.; Botarda, P.; Caron, P.; Rubenstein, P. K.;
Culliton, P.; Schmid, P.; Sessa, P. G.; Xu, P.; Stanczyk,
P.; Tafti, P.; Shivanna, R.; Wu, R.; Pan, R.; Rokni, R.;
Willoughby, R.; Vallu, R.; Mullins, R.; Jerome, S.; Smoot,
S.; Girgin, S.; Iqbal, S.; Reddy, S.; Sheth, S.; P ˜oder, S.; Bhat-
nagar, S.; Panyam, S. R.; Eiger, S.; Zhang, S.; Liu, T.; Ya-
covone, T.; Liechty, T.; Kalra, U.; Evci, U.; Misra, V .; Rose-
berry, V .; Feinberg, V .; Kolesnikov, V .; Han, W.; Kwon, W.;
Chen, X.; Chow, Y .; Zhu, Y .; Wei, Z.; Egyed, Z.; Cotruta,
V .; Giang, M.; Kirk, P.; Rao, A.; Black, K.; Babar, N.; Lo,
J.; Moreira, E.; Martins, L. G.; Sanseviero, O.; Gonzalez,
L.; Gleicher, Z.; Warkentin, T.; Mirrokni, V .; Senter, E.;
Collins, E.; Barral, J.; Ghahramani, Z.; Hadsell, R.; Matias,
Y .; Sculley, D.; Petrov, S.; Fiedel, N.; Shazeer, N.; Vinyals,
O.; Dean, J.; Hassabis, D.; Kavukcuoglu, K.; Farabet, C.;
Buchatskaya, E.; Alayrac, J.-B.; Anil, R.; Dmitry; Lepikhin;
Borgeaud, S.; Bachem, O.; Joulin, A.; Andreev, A.; Hardin,
C.; Dadashi, R.; and Hussenot, L. 2025a. Gemma 3 Techni-
cal Report. arXiv:2503.19786.
Team, G.-V .; Hong, W.; Yu, W.; Gu, X.; Wang, G.; Gan, G.;
Tang, H.; Cheng, J.; Qi, J.; Ji, J.; Pan, L.; Duan, S.; Wang,
W.; Wang, Y .; Cheng, Y .; He, Z.; Su, Z.; Yang, Z.; Pan, Z.;
Zeng, A.; Wang, B.; Shi, B.; Pang, C.; Zhang, C.; Yin, D.;
Yang, F.; Chen, G.; Xu, J.; Chen, J.; Chen, J.; Chen, J.; Lin,
J.; Wang, J.; Chen, J.; Lei, L.; Gong, L.; Pan, L.; Zhang, M.;
Zheng, Q.; Yang, S.; Zhong, S.; Huang, S.; Zhao, S.; Xue,
S.; Tu, S.; Meng, S.; Zhang, T.; Luo, T.; Hao, T.; Li, W.;
Jia, W.; Lyu, X.; Huang, X.; Wang, Y .; Xue, Y .; Wang, Y .;
An, Y .; Du, Y .; Shi, Y .; Huang, Y .; Niu, Y .; Wang, Y .; Yue,
Y .; Li, Y .; Zhang, Y .; Zhang, Y .; Du, Z.; Hou, Z.; Xue, Z.;
Du, Z.; Wang, Z.; Zhang, P.; Liu, D.; Xu, B.; Li, J.; Huang,
M.; Dong, Y .; and Tang, J. 2025b. GLM-4.1V-Thinking:
Towards Versatile Multimodal Reasoning with Scalable Re-
inforcement Learning. arXiv:2507.01006.

Team, K.; Du, A.; Yin, B.; Xing, B.; Qu, B.; Wang, B.; Chen,
C.; Zhang, C.; Du, C.; Wei, C.; Wang, C.; Zhang, D.; Du, D.;
Wang, D.; Yuan, E.; Lu, E.; Li, F.; Sung, F.; Wei, G.; Lai,
G.; Zhu, H.; Ding, H.; Hu, H.; Yang, H.; Zhang, H.; Wu, H.;
Yao, H.; Lu, H.; Wang, H.; Gao, H.; Zheng, H.; Li, J.; Su, J.;
Wang, J.; Deng, J.; Qiu, J.; Xie, J.; Wang, J.; Liu, J.; Yan, J.;
Ouyang, K.; Chen, L.; Sui, L.; Yu, L.; Dong, M.; Dong, M.;
Xu, N.; Cheng, P.; Gu, Q.; Zhou, R.; Liu, S.; Cao, S.; Yu,
T.; Song, T.; Bai, T.; Song, W.; He, W.; Huang, W.; Xu, W.;
Yuan, X.; Yao, X.; Wu, X.; Li, X.; Zu, X.; Zhou, X.; Wang,
X.; Charles, Y .; Zhong, Y .; Li, Y .; Hu, Y .; Chen, Y .; Wang,
Y .; Liu, Y .; Miao, Y .; Qin, Y .; Chen, Y .; Bao, Y .; Wang, Y .;
Kang, Y .; Liu, Y .; Dong, Y .; Du, Y .; Wu, Y .; Wang, Y .; Yan,
Y .; Zhou, Z.; Li, Z.; Jiang, Z.; Zhang, Z.; Yang, Z.; Huang,
Z.; Huang, Z.; Zhao, Z.; Chen, Z.; and Lin, Z. 2025c. Kimi-
VL Technical Report. arXiv:2504.07491.
Tito, R.; Karatzas, D.; and Valveny, E. 2023. Hierarchical
multimodal transformers for Multipage DocVQA. Pattern
Recognition , 144: 109834.
Wang, L.; Yang, N.; Huang, X.; Yang, L.; Majumder, R.; and
Wei, F. 2024. Multilingual E5 Text Embeddings: A Techni-
cal Report. arXiv:2402.05672.
Wang, W.; Bao, H.; Dong, L.; Bjorck, J.; Peng, Z.; Liu, Q.;
Aggarwal, K.; Mohammed, O. K.; Singhal, S.; Som, S.; and
Wei, F. 2023. Image as a Foreign Language: BEIT Pre-
training for Vision and Vision-Language Tasks. In 2023
IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR) , 19175–19186.

A Document Content Processing Strategies
We detail our strategies for handling various content types
within documents, providing specific prompts and the deci-
sion logic used for processing.
A.1 LLM-Native Document Analysis
Modern documents contain heterogeneous content types that
require specialized processing strategies. Rather than rely-
ing on external tools or modality-specific models, we lever-
age the native multimodal capabilities of LLMs to create a
unified processing pipeline. Our approach identifies and pro-
cesses four primary content types: text, tables, vector graph-
ics, and raster images.
Multi-Column Text Processing : Academic papers and
technical documents frequently employ multi-column lay-
outs that naive text extraction would linearize incorrectly.
We detect column structures through spatial clustering of
text bounding boxes. When multiple distinct x-coordinate
clusters are identified, we group text blocks by column and
merge them in reading order. This preserves the logical flow
of content while maintaining computational efficiency by
avoiding complex layout analysis models.
Table Detection and Extraction : Tables present unique
challenges as their semantic meaning derives from both con-
tent and structure. We identify tables through alignment pat-
terns in text blocks: consistent x-coordinates across multiple
rows indicate columnar structure. Rather than attempting to
reconstruct table semantics from positional data, we capture
detected tables as high-resolution images. The multimodal
LLM then interprets these visual representations, preserving
both structural relationships and content semantics that text-
based extraction would lose.
Vector Graphics Filtering : Documents often contain nu-
merous small vector graphics elements (logos, decorative
lines, page numbers) that provide minimal semantic value
while consuming processing resources. We filter these el-
ements through multiple heuristics: size thresholds, color
diversity metrics, and aspect ratios. Graphics smaller than
50x50 pixels or with extreme aspect ratios ( >10:1) are
excluded from processing. This selective filtering reduces
noise in our retrieval index while preserving semantically
rich diagrams and charts.
Visual Content Analysis : For images meeting our rel-
evance criteria, we apply content-aware processing strate-
gies. Charts and diagrams receive detailed analysis prompts
requesting data interpretation, while photographs and illus-
trations receive descriptive prompts. The multimodal LLM
generates textual descriptions that capture both the content
and purpose of visual elements. This text-centric represen-
tation enables unified retrieval across modalities: a query
about “revenue growth” can match both textual discussions
and chart descriptions.
Adaptive OCR Strategy : When standard text extrac-
tion fails or yields insufficient content (common in scanned
or image-heavy documents), we employ the LLM’s vision
capabilities for optical character recognition. Unlike tradi-
tional OCR tools that produce raw text, our approach gener-
ates properly formatted paragraphs with preserved semantic
structure.A.2 Content Type Detection and Processing
Our system categorizes document content into four main
types and applies specialized processing:
Vector Graphics Detection We identify vector graphics
components through multiple heuristics:
• Size threshold: Images smaller than 50x50 pixels
• Color diversity: Less than 10% unique colors in sampled
pixels
• White space ratio: More than 80% white pixels
• Aspect ratio: Elongation factor greater than 10:1
Vector graphics with more than 50 drawing commands
or 100 paths are rendered as complete images for holistic
interpretation.
Table Detection Tables are identified through text align-
ment patterns:
1. Group text blocks by vertical position (y-coordinate clus-
tering)
2. Identify rows with multiple text spans at consistent x-
coordinates
3. Require minimum 3 rows with similar column structure
4. Validate table dimensions exceed 100x50 pixels
Detected tables are captured as images at 2x zoom for vi-
sual analysis by the LLM.
Visual Content Analysis For standalone images meeting
size thresholds (100x100 pixels), we apply content-specific
prompts:
Single Image Prompt:
1Describe this visual content. If it’s a
chart, graph, or diagram, explain
what data or information it shows. If
it’s a photo orillustration,
describe what it depicts. Be concise
but informative.
Multiple Images Prompt:
1Describe these {n} visual elements in
order:
2
3Figure 1: [description]
4Figure 2: [description]
5...
6Figure N: [description]
7
8For each figure, identify ifit’s a
chart/graph/diagram (and what data it
shows) or a photo/illustration (and
what it depicts). Start immediately
with ‘‘Figure 1:’’.
OCR Processing When text extraction fails or images
contain embedded text, we use LLM-based OCR:
OCR Prompt:
1Extract text from this image and present
it as readable paragraphs. Start
directly with the content.

A.3 Multi-Column Layout Processing
For documents with multiple columns, we apply spatial clus-
tering:
1. Extract all text blocks with bounding boxes
2. Apply K-means clustering on x-coordinates (k=2 for
two-column)
3. Sort blocks within each cluster by y-coordinate
4. Merge columns in reading order (left-to-right for LTR
languages)
A.4 Adaptive Resolution Strategy
Visual processing resolution adapts to content complexity:
•Standard : Default resolution for simple images
•High (2x) : Tables and complex diagrams
•Maximum : Limited by available memory (800px
longest dimension)
A.5 Processing Pipeline Integration
The complete processing flow for each page:
1. Extract raw text using PyMuPDF
2. Detect and process tables as visual elements
3. Identify and filter vector graphics components
4. Extract standalone images above size thresholds
5. Apply OCR if text extraction yields insufficient content
6. Merge multi-column layouts if detected
7. Combine all extracted content preserving spatial relation-
ships
This unified approach ensures comprehensive content ex-
traction while maintaining computational efficiency through
selective processing.
B Prompt-Based Pseudo-TOC Generation
The core innovation of our approach is the generation
of pseudo-TOCs through carefully designed prompts. Un-
like traditional methods that rely on formatting cues, we
use semantic understanding to identify topic boundaries.
Our algorithm has three phases: initial segmentation, size-
constrained merging, and title generation. The full algorithm
and prompts are described in the Appendix.
Boundary Detection : Our semantic boundary detection
leverages the LLM’s understanding of topical coherence.
For each potential section boundary, we extract text excerpts
from the end of one segment and the beginning of the next.
The model analyzes these excerpts to determine whether
they represent a continuation of the same topic or a tran-
sition to a new subject. By relying on semantic understand-
ing rather than formatting cues, the method remains robust
across diverse document styles and languages.
Adaptive Section Refinement : The initial segmentation
offers a rough structural outline, but further refinement en-
sures practical usability. Sections that are too small to con-
vey meaningful content are merged with adjacent ones that
share similar topics. In contrast, large sections are preserved
if they exhibit coherent themes, as our hierarchical retrievalcan effectively handle varying section lengths. This adaptive
strategy maintains a balance between structural granularity
and semantic consistency.
Title Generation : Each identified section requires a de-
scriptive title for effective navigation and retrieval. We sam-
ple representative content from each section, with emphasis
on introductory passages where topics are typically estab-
lished. The LLM generates concise titles that capture the
essence of each section’s content. This process produces
human-readable titles that facilitate both automated retrieval
and user navigation.
C Pseudo-TOC Generation Algorithm
Algorithm 1: Pseudo-TOC Generation with Adaptive Seg-
mentation
0:Input: Document pages P={p1, p2, ..., p n}
0:Parameters: Initial chunk size k= 5, min pages m= 3,
max pages M= 15
0:Output: Sections S={s1, s2, ..., s j}
1:// Phase 1: Initial segmentation
2:Divide Pinto chunks of size k
3:boundaries ← { 0}
4:fori= 1to|chunks | −1do
5: Extract ending text from chunk i−1(last 500 chars)
6: Extract starting text from chunk i(first 500 chars)
7: Query LLM for topic boundary detection
8: isnewtopic←LLM response (0 or 1)
9: ifisnewtopic = 1then
10: boundaries .append( i×k)
11: end if
12:end for
13:// Phase 2: Size-constrained merging
14:Create initial sections S′from pages using boundaries
15:foreach section siinS′where |si|< m do
16: Compute content embeddings for si,si−1,si+1
17: simprev←cosine (esi, esi−1)
18: simnext←cosine (esi, esi+1)
19: ifsimprev> sim next then
20: Merge siwithsi−1
21: else
22: Merge siwithsi+1
23: end if
24:end for
25:// Phase 3: Title generation
26:foreach section sinsections do
27: Sample representative content from s
28: Query LLM for section title
29: s.title ←generated title
30:end for
31:return sections
The algorithm operates in three distinct phases. In Phase
1, we perform initial segmentation by dividing the docu-
ment into fixed-size chunks and detecting topic boundaries
between adjacent chunks using LLM-based semantic anal-
ysis. Phase 2 applies size constraints, merging sections that
are too small based on their semantic similarity with adja-

cent sections. Finally, Phase 3 generates descriptive titles for
each section by sampling representative content and prompt-
ing the LLM for concise summaries.
D Prompts for Pseudo-TOC Generation
The performance of our system depends on well-designed
prompts. We present the exact prompts used in our imple-
mentation:
Boundary Detection Prompt:
1Below are short excerpts from two
consecutive pages.
2If both excerpts discuss the same topic,
reply with ’0’.
3If the second excerpt introduces a new
topic, reply with ’1’.
4Reply with a single character only.
5
6[Page A]
7{first_page_text}
8
9[Page B]
10{second_page_text}
This minimalist prompt ensures consistent binary re-
sponses, enabling reliable parsing. The single-character con-
straint prevents verbose explanations that would complicate
post-processing.
Title Generation Prompt:
1Here isa passage from the document.
2Please propose ONE concise title that
captures its main topic.
3
4{section_sample}
5
6Return ONLY the title text, without any
additional commentary orformatting.
The title generation prompt emphasizes conciseness and
directness, producing clean titles without extraneous format-
ting or explanation.
E System Prompts and Query Processing
Our system employs additional prompts for various docu-
ment understanding tasks:
E.1 Chatbot System Prompt
The default system prompt establishes behavioral guidelines
for the conversational interface:
1Basic Principles
21) Check document context first, then
use reliable knowledge ifneeded.
32) Provide accurate information without
unnecessary disclaimers.
43) Always respond inthe same language
as the user’s question.
E.2 Query Improvement Prompts
To enhance retrieval accuracy, we employ query refinement:
Context-Based Query Improvement:
1The user question is: {query}
2
3The retrieved chunks are:
4{combined_answer}5
6Write ONE concise follow-up question
that would help retrieve even more
relevant information.
7Return ONLY the question text. Do not
include any additional text or
explanations.
Alternative Query Generation:
1Given the search query: ‘‘{query}’’
2
3Generate 3 alternative search queries
that might find relevant documents.
4Consider synonyms, related terms, and
different phrasings.
5Return only the queries, one per line.
6
7Alternative queries:
E.3 Document Summarization Prompts
For comprehensive document analysis, we use tiered sum-
marization:
Document Analysis System Prompt:
1You are a professional document analyst.
Your task isto create a
comprehensive summary of a PDF
document based on its sections.
2
3Guidelines:
4- Provide a structured summary that
follows the document’s table of
contents
5- For each section, include key points,
main arguments, and important details
6- Maintain the hierarchical structure of
the document
7- Use clear, concise language while
preserving technical accuracy
8- Include relevant quotes or specific
data points when they are crucial
9- Highlight connections between
different sections when relevant
Executive Summary Generation:
1Based on a document with these sections:
{section_titles}
2
3Provide a brief executive summary (2-3
paragraphs) highlighting the main
theme and key findings.
Section Summarization (Brief Mode):
1Summarize this section ‘‘{title}’’ in
2-3 sentences:
2{combined_content[:1500]}
3
4Summary:
Section Summarization (Detailed Mode):
1Based on the following content from
section ‘‘{title}’’, provide a
concise summary
2highlighting the main points, key
arguments, and important details:
3
4{combined_content}
5
6Summary (2-3 paragraphs):

F Retrieval Pattern Analysis
We analyzed representative cases from MMLongBench-Doc
to understand DocsRay’s retrieval patterns and identify com-
mon scenarios in document question answering. Table 7
presents 11 selected cases that illustrate different retrieval
scenarios and the source sections consulted by the system.
Our analysis reveals three distinct patterns in document
retrieval that highlight both the strengths and limitations of
current document QA systems.
The table presents retrieval patterns across 11 representa-
tive cases from MMLongBench-Doc illustrating three sce-
narios:
Pattern 1: Clear Single-Source Evidence. Simple factual
questions where answers reside on specific pages, such as
“What year is the report for?” (mmlongbench 21), demon-
strate DocsRay’s effectiveness in precise fact retrieval where
evidence localization is unambiguous.
Pattern 2: Multi-Page Evidence Synthesis. Complex
queries requiring information aggregation across multiple
pages, such as counting exterior photos of organizations
(mmlongbench 26) which requires examining pages 10-20.
These cases reveal the importance of complete document
coverage, as missing even one relevant page leads to incor-
rect counts.
Pattern 3: Insufficient Source Information. Questions
where the document lacks sufficient information, such as
mmlongbench 18 which asks about continent-wise partici-
pant distribution but the document only provides total counts
without geographic breakdown. These cases underscore the
importance of systems acknowledging when retrieved con-
tent is insufficient for answering queries.
Statistical Evidence Requirements. Questions involving
statistical comparisons pose unique challenges. Case mm-
longbench 10 asks which group is greater between “Repub-
lican Hispanic” (7%) and “no leans male” (55%), requiring
retrieval from pages 3 and 22. The distributed nature of sta-
tistical evidence, where denominators and percentages may
appear on different pages, demands sophisticated cross-page
reasoning. DocsRay’s pseudo-TOC organization helps by
grouping related statistical content within sections, reducing
the likelihood of missing critical context.
Visual Evidence Interpretation. Several cases involve vi-
sual content interpretation. The Indian Space Programme
map (mmlongbench 20) and organizational photos (mm-
longbench 26) require the system to process images and
generate textual descriptions. While DocsRay identifies and
describes these visual elements, the conversion to text-based
representations limits detailed visual analysis, a deliberate
trade-off prioritizing semantic retrieval over pixel-level pro-
cessing.
These patterns demonstrate that effective document QA
requires more than accurate retrieval; it benefits from trans-
parent source attribution allowing users to understand which
sections were consulted. The cases where DocsRay provides
section-level references (Pattern 1 and 2) offer transparency,
while acknowledgment of missing information (Pattern 3)
prevents misinformation. This analysis reinforces our rec-
ommendation for future systems to implement fine-grained
evidence grounding mechanisms that link specific claims toexact source passages, for high-stakes applications in legal,
medical, and financial domains where detailed answer veri-
fication is mandatory.
F.1 Comparative Analysis Across Model Scales
To better understand the impact of model scaling on doc-
ument understanding capabilities, we conducted a detailed
comparative analysis using three variants of DocsRay with
different backbone models: DocsRay-Pro, DocsRay-Base,
and DocsRay-Lite. This analysis reveals distinct perfor-
mance patterns that correlate with model scale and task com-
plexity.
The scaling analysis reveals three distinct tiers of docu-
ment understanding capability:
Tier 1: Simple Fact Retrieval All model sizes success-
fully handle straightforward factual questions that require
single-page evidence lookup. This suggests that basic re-
trieval and fact extraction capabilities are well-preserved
even at smaller scales, validating our hierarchical retrieval
architecture’s effectiveness across model sizes.
Tier 2: Complex Reasoning with Context The Base and
Pro models demonstrate superior performance on questions
requiring technical comprehension or multi-step reasoning.
Case mmlongbench 16 asks for the definition of “law of
good gestalt,” which requires understanding technical ter-
minology and synthesizing conceptual information. While
both larger models provide accurate definitions mentioning
“elements grouped together if they form regular, orderly pat-
terns,” the Lite model produces a garbled response mix-
ing unrelated concepts. This tier reveals the importance of
model scale for maintaining coherent reasoning chains.
Tier 3: Advanced Multi-Page Synthesis Only the Pro
model consistently succeeds at tasks requiring document
analysis across many pages. Case mmlongbench 26 de-
mands counting exterior photos of organizations scattered
across pages 10-20. The Pro model correctly identifies 10
photos with detailed descriptions, while the Base model
finds only 6, and the Lite model identifies just 3. This pattern
suggests that larger models maintain better working mem-
ory for tracking multiple pieces of evidence across extended
contexts.
F.2 Detailed Case Studies
To illustrate the differences in model capabilities, we present
analyses of representative cases from each performance tier.
These cases were selected to highlight where models suc-
ceed or fail and how the nature of their responses changes
with scale, revealing insights into the emergence of different
cognitive capabilities.
Case Study 1: Multi-Page Evidence Aggregation This
case asks “How many human quotes are provided with
source in this document?” requiring systematic scanning of a
40-page document to identify quoted individuals with proper
attribution.
DocsRay-Pro Response: “There are 5 human quotes pro-
vided with a source in this document: 1. John Holloway,

Case ID Question Evidence
PagesActual Reference Content
Cases with Clear Evidence Requirements
mmlongbench 10 Which group is greater: Republican
Hispanic or no leans male?[3, 22] Page 3: “31% of Americans identify as Democrats,
25% as Republicans”
Page 22: “7% of Americans are Hispanic Republi-
cans, 55% of no leans are male”
mmlongbench 13 Where was Gestalt psychology con-
ceived?[2] Page 2: “Gestalt psychology was conceived in the
Berlin School of Experimental Psychology”
mmlongbench 20 What does the map show? [17] Page 17: “Map displaying locations of various cen-
ters of Indian Space Programme including ISRO fa-
cilities, research laboratories, and launch sites”
mmlongbench 21 What year is the report for? [3] Page 3: “Annual Report 2015-2016”
Cases Requiring Multi-Page Evidence
mmlongbench 36 How many human quotes with
sources?[14, 19, 20,
33, 37]Page 14: “John Holloway, European Investment
Fund”
Page 19: “Andreas Ritter, Arico Investments”
Page 20: “Egbert Freiherr von Cramm, Deutsche
Bank”
Page 33: “Mark Thompson, Venture Capital Asso-
ciation”
Page 37: “Sarah Chen, Asian Development Bank”
mmlongbench 26 How many exterior photos of orga-
nizations?[10-12, 14-
16, 19-20]Pages show exterior views of: ISRO headquar-
ters, Vikram Sarabhai Space Centre, Satish Dhawan
Space Centre, Physical Research Laboratory, Space
Applications Centre, and 5 other facilities
mmlongbench 12 How many charts from Pew Re-
search data?[3, 6, 16,
18-20, 22]Charts sourced from “Annual totals of Pew Re-
search Center survey data” appear on voter demo-
graphics, party affiliation trends, generational vot-
ing patterns, and political ideology distributions
Cases Where Evidence Attribution Failed
mmlongbench 8 Democrats voting percentage in
2024?N/A No reference available: document only contains
data through 2018
mmlongbench 9 5% support increase comparison? N/A No specific 5% increase data found in presidential
approval ratings
mmlongbench 18 Which continent has most partici-
pants?[13]* Page 13: “425,105 registered participants from 135
countries” but no continent-wise breakdown pro-
vided
Table 7: Retrieval patterns in MMLongBench-Doc showing source pages consulted and actual reference content. Clear source
attribution helps understand which document sections were used, in multi-page scenarios. Cases marked with * indicate where
relevant sections were retrieved but answers were still incorrect due to interpretation issues.
European Investment Fund (Page 14) 2. Andreas Ritter,
Arico Investments Advisory (Page 19) 3. Egbert Freiherr
von Cramm, Deutsche Bank Private Equity (Page 20) 4.
Mark Thompson, Venture Capital Association (Page 33) 5.
Sarah Chen, Asian Development Bank (Page 37)”
DocsRay-Base Response: “There are five human quotes
provided with sources in the document: 1. Andreas Ritter
(Arico Investments Advisory) 2. John Holloway (European
Investment Fund) 3. Egbert Freiherr von Cramm (Deutsche
Bank) 4. Mark Thompson (mentioned but location unclear)
5. [One quote potentially missed]”
DocsRay-Lite Response: “There are 17 human quotes
with sources provided in the document.”
The Pro model demonstrates systematic document traver-
sal with precise page citations, while the Base model cap-
tures the correct count but shows less precise evidence lo-
calization. The Lite model overcounts, suggesting difficulty
distinguishing between quoted content and regular text.Case Study 2: Visual Content Analysis This challenging
case requires counting hand-drawn cartoon illustrations used
to explain A/B testing concepts across a technical presenta-
tion.
DocsRay-Pro identifies all 4 cartoons: “Four hand drawn
cartoons are included: 1) The cartoon depicting a meeting
room with a hippopotamus (Option B) - illustrating selection
bias 2) The ice cream shop cartoon - demonstrating customer
preference testing 3) The cartoon showing different testing
scenarios - explaining variance in results 4) The decision-
making process cartoon - showing statistical significance”
DocsRay-Base provides an ambiguous response: “The in-
formation is not specified in the document.”
DocsRay-Lite attempts counting but conflates different
visual elements: “The document describes several visual
content slides... Four panels in the first comic depicting A/B
testing, future trends, sentiment analysis...”
This case highlights how visual understanding capabil-

Case ID Question Type Pro Base Lite
Simple Fact Retrieval - All Models Succeed
mmlongbench 13 Where was Gestalt psychology conceived?√ √ √
mmlongbench 21 What year is the report for?√ √ √
mmlongbench 10 Republican Hispanic vs no leans male comparison√ √ √
Complex Reasoning - Larger Models Excel
mmlongbench 16 Define law of good gestalt (technical definition)√ √×
mmlongbench 36 Count human quotes with sources (multi-page)√ √×
mmlongbench 9 5% support rate analysis (inference required)√ √×
Advanced Multi-Page Synthesis - Only Pro Succeeds
mmlongbench 26 Count exterior photos across 10+ pages√× ×
mmlongbench 48 Count hand-drawn cartoons (visual analysis)√× ×
mmlongbench 19 Identify shapes in closure principle diagram√× ×
Table 8: Performance comparison across model scales on representative MMLongBench-Doc cases.√indicates correct answer,
×indicates incorrect or incomplete answer.
ities degrade with model scale. The Pro model not only
counts correctly but provides contextual understanding of
each cartoon’s purpose, while smaller models struggle to
distinguish hand-drawn illustrations from other visual ele-
ments.
F.3 Critical Case Studies and Edge Cases
Our analysis revealed several instructive edge cases that
demonstrate the nuanced differences in model capabilities
and unexpected scaling behaviors across the three DocsRay
variants.
Edge Case 1: Statistical Chart Interpretation This case
asks to count charts sourced from “Annual totals of Pew
Research Center survey data.” Surprisingly, smaller mod-
els performed better in this instance. The DocsRay-Pro vari-
ant incorrectly identified only 6 charts, while DocsRay-Base
correctly found all 7 charts with partial page citations. The
DocsRay-Lite model also performed well, identifying 6-7
charts though with different page references than the ground
truth.
DocsRay-Pro Response: “Six charts and tables are
sourced from Annual totals of Pew Research Center survey
data. Here’s a breakdown of where they appear: Page 22,
Page 4, Page 19, Page 3, Page 6, Page 1...”
DocsRay-Base Response: “The information is not speci-
fied in the document. However, several charts and tables are
sourced from Annual totals of Pew Research Center survey
data, including: Page 22, Page 19, Page 11...”
DocsRay-Lite Response: “There are six charts and tables
sourced from ’Annual totals of Pew Research Center survey
data’: Page 22, Page 4, Page 10, Page 14, Page 19, Page
23...”
The Pro model’s undercount reveals an interesting phe-
nomenon where increased model sophistication can some-
times lead to over-filtering of relevant evidence. Our analysis
suggests that the larger model may have applied stricter cri-
teria for what constitutes a “chart from Pew Research data,”
potentially excluding borderline cases where the chart attri-
bution was less explicit or where the data source was men-
tioned in adjacent text rather than directly within the chartcaption. This behavior highlights the importance of calibrat-
ing evidence selection thresholds across model scales and
suggests that larger models may develop more restrictive in-
terpretations of query requirements, which can be detrimen-
tal when complete coverage is needed.
Edge Case 2: Color Identification in Maps When pre-
sented with a question about Mali’s color in a geographic
visualization demonstrating GeoShapes using SPARQL and
OSM, the model responses revealed a striking pattern in hal-
lucination behavior across scales. Only the Lite model pro-
vided a specific answer claiming Mali was colored “dark
purple,” while both the Base and Pro models correctly stated
that the information was not specified in the document. Our
detailed investigation of the source document confirmed that
while a map of Africa was indeed present with countries
color-coded by land area, the document text only mentioned
that “lighter shades indicate smaller areas and darker shades
represent larger ones” without specifying the actual colors
used for any individual country.
DocsRay-Pro Response: “The document states that the
map of Africa with countries color-coded by land area has
lighter shades indicating smaller areas and darker shades
representing larger ones. However, it does not specify the
exact color for Mali.”
DocsRay-Base Response: “The information is not speci-
fied in the document.”
DocsRay-Lite Response: “The map used to demonstrate
GeoShapes using SPARQL and OSM colors Mali in dark
purple.”
This case provides valuable insight into how model scale
affects the propensity for hallucination. The Lite model’s
confident but incorrect response likely stems from pattern
matching against common geographic visualizations in its
training data, where African countries are frequently de-
picted in purple or violet hues. The fact that larger mod-
els resist this temptation and instead acknowledge the ab-
sence of specific color information demonstrates that scale
brings not just improved accuracy but also enhanced epis-
temic awareness: the ability to recognize and communicate
the boundaries of available information.

Edge Case 3: Intersection Point Analysis The question
asks for the specific value where the US, Europe, and China
trend lines intersect on an average exit cap chart. This
seemingly straightforward task of reading a numerical value
from a line chart produced notably different results across
model scales. The DocsRay-Pro model correctly identified
the intersection point at 300 million USD, while both the
DocsRay-Base model and DocsRay-Lite model incorrectly
reported 350 million USD.
DocsRay-Pro Response: “300”
DocsRay-Base Response: “350”
DocsRay-Lite Response: “$350M”
This case reveals the nuanced challenges involved in pre-
cise chart interpretation. The task requires multiple cognitive
steps: first identifying the three relevant trend lines among
potentially many lines on the chart, then locating their inter-
section point, and finally accurately reading the y-axis value
at that point. The consistent error of both smaller models
reporting 350 million suggests they may have identified a
nearby gridline or perhaps the value at a slightly different
point on the chart. The Pro model’s unique success indicates
that accurate chart interpretation (the ability to map visual
positions to numerical scales) is a capability that emerges
primarily at larger scales. This finding has implications for
applications requiring financial or scientific data extraction
from visualizations, where even small numerical errors can
have consequences.
F.4 Implications for Retrieval System Design
These case studies reveal several insights that should guide
the design of future document retrieval systems intended for
deployment across diverse computational environments and
use cases.
First, the universal effectiveness of hierarchical retrieval
across all model scales represents a finding. Even our small-
est 4B parameter model achieves reasonable performance
on simple factual queries when equipped with our pseudo-
TOC based hierarchical retrieval system. This validates a
key architectural decision: the structured document repre-
sentation created by our pseudo-TOC generation helps com-
pensate for limited model capacity by reducing the search
space to relevant sections. Rather than requiring the model
to search through potentially thousands of chunks, the hier-
archical structure allows even resource-constrained models
to focus their limited attention on the most promising doc-
ument regions. This finding suggests that investing in better
document structure inference may yield greater returns than
simply scaling model size, for deployment scenarios with
computational constraints.
Second, our analysis reveals that evidence aggregation ca-
pabilities scale strongly with model parameters. Tasks re-
quiring synthesis across multiple document pages show the
clearest performance stratification between model tiers. The
ability to maintain coherent tracking of evidence across ex-
tended contexts appears to be correlated with model param-
eters. For instance, when counting human quotes across a
40-page document, the Pro model tracks all five individu-
als with their precise page locations, while the Lite model
overcounts to 17, suggesting it loses track of which quotesit has already counted. This pattern indicates that applica-
tions requiring document analysis (such as due diligence re-
views, systematic literature surveys, or regulatory compli-
ance checking) should prioritize larger models despite their
increased computational cost.
Third, visual understanding capabilities demonstrate an
interesting graceful degradation pattern with model scale.
While all models can identify that visual content exists
within documents, the ability to accurately count, catego-
rize, and interpret specific visual elements clearly correlates
with model size. Our analysis of the hand-drawn cartoon
counting task exemplifies this: the Pro model not only counts
correctly but also provides semantic understanding of each
cartoon’s purpose, the Base model acknowledges its limi-
tations, and the Lite model attempts counting but conflates
different visual elements. This suggests a tiered deployment
strategy where basic visual awareness tasks can be handled
by smaller models, but applications requiring detailed visual
analysis (such as technical diagram interpretation or chart
data extraction) necessitate larger models.
Fourth, we observe that conservative behavior in han-
dling ambiguous or unanswerable questions emerges as a
function of scale. Larger models demonstrate more nuanced
responses, providing helpful context about available infor-
mation rather than attempting to guess. When asked about
2024 voting data in a document containing only information
through 2018, all models correctly identify the question as
unanswerable, but larger models additionally provide con-
text about what related information is available. This conser-
vative behavior proves valuable for high-stakes applications
in legal, medical, or financial domains where acknowledg-
ing uncertainty is preferable to confident errors that could
lead to costly mistakes.
Fifth, our edge cases reveal that model scaling does not
guarantee monotonic improvement across all task types.
Some specific capabilities may actually degrade or show
non-linear patterns with scale. The chart counting task where
smaller models outperformed the Pro model exemplifies this
phenomenon. The larger model’s more sophisticated under-
standing led it to apply stricter criteria for what constitutes
a relevant chart, resulting in undercounting. This finding ne-
cessitates careful evaluation for particular use cases and sug-
gests that blind trust in larger models may be misplaced for
certain task types.
These findings suggest that future document retrieval sys-
tems should adopt a more nuanced approach than maximiz-
ing model size. System designers should consider the spe-
cific types of questions and evidence patterns common in
their target domains. A portfolio approach using different
model scales for different query types may provide optimal
cost-performance trade-offs. For instance, a production sys-
tem might route simple factual queries to a Lite model for ef-
ficiency, escalate multi-page synthesis tasks to a Base model,
and reserve the Pro model for cases requiring precise visual
interpretation or complex reasoning. Such an adaptive rout-
ing strategy could significantly reduce computational costs
while maintaining high accuracy for challenging queries.

G Limitations and Future Work
While DocsRay demonstrates strong performance on
MMLongBench-Doc, we acknowledge several limitations in
our current evaluation scope and implementation choices.
G.1 Limitations of Docsray
Dependency on Backbone LLM Choice The quality and
structure of the generated pseudo-TOC depends on the spe-
cific choice of the backbone LLM. Different models may
produce varying segmentation boundaries and title qualities
based on their training data and inherent biases. While our
experiments with Gemini-1.5 Pro show robust performance,
the pseudo-TOC generation approach may require prompt
engineering adjustments when adapting to other LLMs. This
dependency underscores that our contribution lies not in a
universal algorithm but in demonstrating how to leverage
LLM capabilities for document structuring.
Performance on Documents with Multiple Images
SlideVQA and similar benchmarks requiring reasoning over
multiple images per page expose expected limitations given
our design choices. DocsRay achieves only 17.1% EM (Ex-
act Match) for Pro, 16.1% for Base, and 9.91% for Lite vari-
ants. The relatively low score on SlideVQA confirms that
quantitative reasoning over multiple concurrent images is a
distinct challenge, one that we deliberately exclude from the
primary research scope of DocsRay in order to concentrate
on document-level semantic retrieval. The core issue stems
from our text-based retrieval approach: we convert images
to textual descriptions (alt-text) for embedding and retrieval.
While this enables efficient text-based search, it cannot pre-
serve pixel-level visual relationships crucial for multi-image
comparison tasks. Our text-centric pipeline underperforms
on SlideVQA-style tasks that demand pixel-level compar-
ison across multiple images. We regard such tasks as fu-
ture extensions rather than shortcomings of the proposed re-
trieval architecture.
Absence of Semantic Retrieval Benchmarks The docu-
ment understanding community lacks benchmarks designed
to evaluate semantic retrieval quality in document contexts.
Existing benchmarks focus on end-to-end question answer-
ing accuracy but do not isolate retrieval performance. This
limitation prevented us from quantitatively validating our
core technical contribution, which lies in demonstrating the
superiority of hierarchical semantic retrieval over flat re-
trieval methods. We conducted qualitative evaluations with
domain experts who confirmed that our retrieved chunks
showed higher topical coherence and completeness com-
pared to baseline systems. However, the absence of stan-
dardized metrics and datasets for document-centric semantic
retrieval represents a significant gap. We are currently de-
veloping a benchmark dataset with manually annotated rel-
evance judgments to enable future quantitative evaluation of
retrieval quality.
Dependency on Multimodal LLMs for Document Pro-
cessing While leveraging Gemma-3’s (Team et al. 2025a)
multimodal capabilities enables state-of-the-art performance
on MMLongBench-Doc, this approach may not generalizeoptimally to all document understanding tasks. Documents
requiring precise layout understanding, such as forms, in-
voices, or complex tables with spatial relationships, may
benefit from specialized layout-aware models. Our current
approach treats layout implicitly through the multimodal
LLM’s understanding rather than explicitly modeling spatial
relationships. For instance, in documents where the relative
position of text blocks carries semantic meaning (e.g., or-
ganizational charts, complex forms), our text-extraction ap-
proach may miss crucial structural information. Future re-
search should investigate hybrid approaches that combine
our semantic understanding with explicit layout modeling
for documents where structure is paramount to meaning.
Lack of Fine-Grained Evidence Grounding While Doc-
sRay provides section-level source attribution by listing the
context sections used for generation (see Section 4.6), it cur-
rently lacks fine-grained, statement-level evidence ground-
ing. The generated “References” list indicates the general
source of information but does not link specific claims
within the answer to the exact sentences or visual elements
that support them. As demonstrated in our retrieval analysis
(see Table 7), fine-grained evidence grounding helps build
user trust, in high-stakes domains like financial reporting
and medical records where detailed audit trails are manda-
tory. The current pipeline maintains source metadata at the
section level but does not preserve paragraph or sentence-
level attribution in the output. Implementing this more gran-
ular form of citation to enhance verifiability is a key direc-
tion for future work, such as highlighting exact text spans,
bounding box identification for visual elements, or inline ci-
tations that map claims to specific source passages.
Limited Multilingual Evaluation Our quantitative eval-
uation focuses exclusively on English technical and busi-
ness documents, leaving the system’s multilingual capabil-
ities empirically unvalidated despite theoretical support. A
benchmark for multilingual, multi-page document question
answering remains, to our knowledge, absent from the re-
search landscape. This gap is concerning given the global
nature of document processing needs in multinational cor-
porations, international organizations, and cross-border col-
laborations.
While we selected the Multilingual-E5-Large and BGE-
M3 embedding combination for their advertised multilin-
gual capabilities, with BGE-M3 specifically trained on over
100 languages and Multilingual-E5-Large supporting 94
languages, we lack empirical evidence of their effective-
ness in our dual-embedding architecture across different lan-
guage families. Critical questions remain unanswered: Does
the concatenation approach maintain its advantages when
processing non-Latin scripts? How does the pseudo-TOC
generation perform with languages that have different dis-
course structures than English? Do our prompt-based bound-
ary detection methods generalize to languages with differ-
ent paragraph and section conventions? These questions de-
mand investigation across diverse language families includ-
ing East Asian languages with their unique character sys-
tems, right-to-left scripts like Arabic and Hebrew, and mor-
phologically rich languages like Finnish or Turkish.

Comprehensiveness of Architectural Exploration
While our experiments demonstrate the clear superiority of
concatenation over simple addition, we acknowledge that
the space of possible fusion mechanisms extends far beyond
these two approaches. We did not evaluate alternative
fusion strategies such as Hadamard product, learned linear
projections, gated fusion mechanisms, or attention-based
combination methods. This stems from our prioritization
of efficiency for production deployment over exhaustive
architectural search.
Our preliminary trials with more complex fusion mech-
anisms provide tantalizing hints of unexplored potential.
A learnable MLP layer for combining embeddings showed
marginal accuracy improvements of 1-2 percentage points
but at the cost of 3x higher inference latency due to the
additional neural network forward pass. Attention-based fu-
sion, where the query dynamically weights the contribution
of each embedding model, demonstrated promise for han-
dling domain-specific queries but required engineering to
integrate with our caching infrastructure. These results sug-
gest that while concatenation provides a balance of simplic-
ity and performance, task-specific applications might benefit
from more sophisticated fusion strategies. A comprehensive
benchmark comparing fusion mechanisms across different
document types, query categories, and computational bud-
gets remains valuable future work.
Out-of-Scope Tasks Complex Layout Understanding rep-
resents a fundamental boundary of our approach. Docu-
ments featuring intricate spatial layouts, where the relative
positioning of elements carries semantic meaning, require
capabilities beyond our text-centric architecture. Consider a
complex government form where checkboxes and their la-
bels are spatially distributed without explicit textual connec-
tions, or technical flowcharts and organizational diagrams
where relationships are expressed through alignment, direc-
tionality, or visual grouping.
Our system effectively extracts and understands textual
content but inevitably loses critical structural information
encoded in spatial relationships. Humans intuitively grasp
hierarchical and peer relationships from visual positioning
alone, whereas our method flattens two-dimensional layouts
into a linear text stream. This limitation arises from our de-
liberate emphasis on deep semantic understanding of textual
content rather than shallow capture of spatial relations.
Multi-Image Quantitative Comparison tasks, frequently
encountered in presentations and scientific documents, also
lie outside our scope. Such tasks involve multiple visualiza-
tions that require simultaneous visual retention and quantita-
tive reasoning. While our architecture successfully converts
individual images into textual descriptions, it fails when vi-
sual comparison across multiple images is essential. Textual
descriptions inherently omit the visual patterns vital for in-
tuitive comparisons, making cross-image reasoning imprac-
tical.
These exclusions are deliberate design choices reflect-
ing our primary research focus: effective semantic retrieval
from long-context documents. Our hypothesis, validated by
strong performance on MMLongBench-Doc, is that the ma-jority of practical document understanding tasks can be ef-
fectively addressed through deep textual comprehension,
even when that text originates from visual content.
Future research could explore hybrid architectures com-
bining our semantic retrieval strengths with specialized lay-
out analysis and visual comparison modules. Such modular
systems would route queries based on document type, effi-
ciently handling both text-heavy and visually complex doc-
uments without sacrificing computational efficiency.
G.2 Suggestions for Future Work
Development of a Multilingual Document QA Bench-
mark This benchmark should span at least 20 languages
across 5 language families, include documents from diverse
domains (legal, medical, technical, financial), and feature
questions requiring various levels of reasoning complexity.
Ground-truth relevance judgments at the section, paragraph,
and sentence levels would enable fine-grained retrieval eval-
uation and help identify language-specific optimal configu-
rations.
Systematic Exploration of Embedding Fusion Strate-
gies This includes investigating learned gates that dynam-
ically weight embeddings based on query characteristics,
late-interaction mechanisms inspired by ColBERT that pre-
serve token-level information, and cross-attention architec-
tures that allow embeddings to inform each other. An evalua-
tion framework measuring accuracy, latency, memory usage,
and cache efficiency would identify Pareto-optimal configu-
rations for different deployment scenarios.
Integration of Explicit Evidence Grounding and Cita-
tion Mechanisms This involves developing visual over-
lays that highlight relevant passages in original documents,
generating structured citations mapping answer components
to source locations, and creating confidence scores that com-
municate uncertainty at the claim level. An interactive inter-
face where users can trace any statement back to its support-
ing evidence would enable the trust and verifiability essen-
tial for high-stakes applications in legal, medical, and finan-
cial domains.
H Ethical Statement
Our work focuses on improving document understanding
and retrieval systems. While DocsRay processes document
content using LLMs, it does not generate new content that
could be used for misinformation. The system is designed
for legitimate document analysis tasks and includes no ca-
pabilities for document forgery or manipulation. All exper-
iments were conducted on publicly available datasets with
appropriate licenses.
We acknowledge potential dual-use concerns where doc-
ument understanding technology could be misused for
privacy violations or unauthorized information extraction.
However, DocsRay operates solely on documents provided
by users and includes no capabilities for web scraping, unau-
thorized access, or data exfiltration. The system processes
documents locally without transmitting content to external
services, preserving user privacy and data sovereignty.

I Reproducibility Statement
To ensure full reproducibility of our results, we provide im-
plementation details and release all necessary resources:
I.1 Code and Model Availability
We will release our complete codebase under the MIT li-
cense upon publication, including:
• Complete implementation of pseudo-TOC generation,
dual embedding, and hierarchical retrieval
• Exact prompts used for boundary detection and title gen-
eration
• Scripts for downloading and preparing MMLongBench-
Doc dataset
• Model configuration files for all DocsRay variants
(Pro/Base/Lite)
• Evaluation scripts with detailed logging of per-sample re-
sults
I.2 Model Specifications
All models used are publicly available on HuggingFace:
•Language Models : Gemma-3 family (4B/12B/27B pa-
rameters) from google/gemma-3
•Embedding Models :
–BGE-M3 from BAAI/bge-m3
–Multilingual-E5-Large from
intfloat/multilingual-e5-large
I.3 Hardware and Software Requirements
Hardware :
• Minimum: 16GB RAM/VRAM for DocsRay-Lite
• Recommended: 32GB RAM/VRAM for DocsRay-Base
• Optimal: 80GB VRAM (H100 GPU) for DocsRay-Pro
• Supports: CUDA (NVIDIA), MPS (Apple Silicon), CPU
fallback
Software Dependencies :
• Python 3.11.0
• PyTorch ≥2.0.0 (tested with 2.6.0)
• llama-cpp-python==0.3.9 (critical for GGUF support)
• tiktoken ≥0.5.0, numpy ≥1.24.0
• PyMuPDF ≥1.23.0 for PDF processing
• Complete requirements in requirements.txt
I.4 Hyperparameter Configuration
All hyperparameters are specified in config.yaml :
• Chunking: 550 tokens per chunk, 25 token overlap
• Coarse search: β= 0.3, top-5 sections
• Fine search: top-10 chunks
• Generation: temperature=0.7, top-p=0.95, repeat-
penalty=1.1
• Evaluation: All reported results are averaged over 5 in-
dependent runs with different random seeds, follow-
ing the official MMLongBench-Doc evaluation protocol
(https://github.com/mayubo2333/MMLongBench-Doc)
• Embedding: L2 normalization, concatenation fusionJ Evidence Grounding Output Format
To ensure trustworthy responses with verifiable sources,
DocsRay provides structured evidence grounding in all out-
puts. Here is an example demonstrating the system’s citation
capabilities:
1$ docsray analyze document.pdf ‘‘What
are the main contributions?’’
2
3Answer:
4This paper presents three main
contributions to document
understanding:
5(1) A training-free pseudo-TOC
generation algorithm that transforms
6unstructured documents into hierarchical
structures, achieving 92%
7accuracy inboundary detection without
requiring specialized training.
8(2) A novel dual embedding architecture
combining BGE-M3 and E5-Large
9that improves retrieval performance by
15% over single-model approaches.
10(3) A hierarchical retrieval system that
reduces computational complexity
11from O(N) to O(S + k_1 *N_s), providing
3.2x faster query processing.
12
13References:
14[Section 1: Introduction, Pages 1-3]
15[Section 3.2: Pseudo-TOC Generation,
Pages 12-15]
16[Section 4.2: Dual Embedding Analysis,
Pages 23-24]
17[Table 3: Retrieval Performance
Comparison, Page 25]
18[Figure 2: Query Processing Time
Analysis, Page 34]
This structured format enables users to verify each claim
by directly accessing the referenced sections, tables, and fig-
ures in the original document.