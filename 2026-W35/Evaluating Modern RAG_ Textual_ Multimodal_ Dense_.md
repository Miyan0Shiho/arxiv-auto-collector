# Evaluating Modern RAG: Textual, Multimodal, Dense, and Late Interaction Pipelines

**Authors**: Emre Kuru, Mehmet Onur Keskin

**Published**: 2026-08-24 12:25:52

**PDF URL**: [https://arxiv.org/pdf/2608.23176v1](https://arxiv.org/pdf/2608.23176v1)

## Abstract
Retrieval-augmented generation (RAG) systems have traditionally relied on text-based pipelines that extract and retrieve information from documents. While efficient and lightweight, these approaches often struggle with documents where meaning is conveyed through layout, tables, and visual elements. Recent advances in multimodal pipelines, powered by vision-language models (VLMs), improve retrieval quality by jointly encoding visual and textual signals, but at increased computational and memory cost. We propose a quantitative, data-driven selection methodology that guides practitioners in choosing the most appropriate RAG pipeline for a given document corpus based on empirical effectiveness and resource constraints. We evaluate contemporary textual and multimodal pipelines, including dense and late-interaction architectures, analyze their trade-offs, and provide actionable guidance for balancing retrieval performance with system efficiency.

## Full Text


<!-- PDF content starts -->

Evaluating Modern RAG: Textual, Multimodal, Dense, and
Late Interaction Pipelines
Emre Kuru1, Mehmet Onur Keskin1
1Özyeğin University, Istanbul, Türkiye
Abstract
Retrieval-augmented generation (RAG) systems have traditionally relied on text-based pipelines that extract
and retrieve information from documents. While efficient and lightweight, these approaches often struggle
with documents where meaning is conveyed through layout, tables, and visual elements. Recent advances in
multimodal pipelines, powered by vision-language models (VLMs), improve retrieval quality by jointly encoding
visual and textual signals, but at increased computational and memory cost. We propose a quantitative, data-
driven selection methodology that guides practitioners in choosing the most appropriate RAG pipeline for a
given document corpus based on empirical effectiveness and resource constraints. We evaluate contemporary
textual and multimodal pipelines, including dense and late-interaction architectures, analyze their trade-offs, and
provide actionable guidance for balancing retrieval performance with system efficiency.
Keywords
Retrieval-Augmented Generation, Multimodal Retrieval
1. Introduction
Retrieval-Augmented Generation (RAG) systems have emerged as a powerful solution for grounding
large language models in external document corpora, particularly in knowledge-intensive domains such
as finance, healthcare, and law [ 1,2,3]. Early RAG pipelines operated on purely textual representations
in such domains, extracting content from scanned documents using Optical Character Recognition
(OCR) [ 4]. Thesetext-only pipelinesproved effective for linear documents with clean structure, offering
fast and lightweight retrieval. Although widely adopted due to their simplicity and efficiency, traditional
text-only pipelines face significant limitations when applied to visually complex documents. OCR tends
to flatten structural and spatial information, such as table layouts, section headers, or figure references,
into unstructured text, often leading to the loss of semantic cues critical for accurate retrieval [ 5,6].
As a result, such pipelines frequently underperform tasks that require an understanding of document
layout, tabular reasoning, or visual context.
Several enhanced text-based methods have been proposed to mitigate these limitations, such as layout-
aware approaches. That incorporates structural information by applying document layout parsers [ 7,8]
to segment documents into semantically meaningful regions such as titles, captions, tables, and figures,
and using these annotations to guide chunking strategies [ 9], providing a partial structural understanding
of the document. Furthermore, image captioning techniques have been used to convert visual elements
(e.g., charts or figures) that cannot be captured by OCR into descriptive text [10], allowing them to be
indexed and retrieved within a traditional text-only framework. These enhancements extended the
reach of text-based pipelines into visually rich domains while maintaining relatively low computational
overhead. Nonetheless, even with such augmentations, text-based approaches still operate on symbolic
approximations of visual content. They remain constrained by the quality of the layout or captioning
models and often struggle to preserve fine-grained spatial relationships and formatting nuances. This
led to the emergence ofmultimodal pipelines[ 11,12], which operate directly on document images using
vision-language models (VLMs). By encoding full-page inputs, these models jointly capture textual
content, spatial layout, and visual features in a single embedding, dramatically improving retrieval
IR-RAG @ SIGIR 2025: Workshop on Information Retrieval for Retrieval-Augmented Generation, July 17, 2025, Padova, Italy
/envel⌢pe-⌢penemre.kuru@ozu.edu.tr (E. Kuru); onur.keskin@ozu.edu.tr (M. O. Keskin)
/orcid0009-0007-6130-6272 (E. Kuru); 0000-0002-8025-4304 (M. O. Keskin)
©2025 Copyright for this paper by its authors. Use permitted under Creative Commons License Attribution 4.0 International (CC BY 4.0).

in visually complex settings. However, they come with steep system-level costs, including increased
memory consumption, indexing, and retrieval latency.
In parallel to this modality shift, retrieval architectures have also evolved. Traditional retrieval
architectures have embedded these documents (text or image) into a single vector, enabling fast similarity
search. However, such representations compress the information, leading to diminished retrieval
precision, especially when fine-grained semantic or structural alignment is required. To address this,
late interactionarchitectures [ 13] have been proposed. Instead of encoding the input into a single
vector, these models preserve multiple sub-representations and perform more expressive similarity
computations at retrieval time. While this boosts performance, particularly in layout-sensitive queries,
it also introduces higher latency, increased memory usage, and larger index sizes.
Despite the growing variety of pipeline designs, there remains little guidance on selecting the best
approach for a specific task. This work addresses this gap by comprehensively evaluating state-of-the-
art RAG retrieval pipelines across the two key design axes discussed: modality (text vs. vision) and
architecture (dense vs. late interaction). Our study benchmarks these pipelines on retrieval performance
and across a range of Quality-of-Service (QoS) metrics on two challenging, real-world datasets that
exhibit diverse document structures and layout complexity. Building on these findings, we propose a
selection methodology for identifying the most suitable RAG pipeline for a given document corpus and
computational constraints. This is a deployment-oriented framework for selecting the most efficient
and accurate pipeline. The rest of the paper is organized as follows. Section 2 reviews the related work.
Section 3 describes the retrieval pipelines evaluated in this study. Section 4 presents the experimental
setup and results across effectiveness, latency, and memory usage. Finally, Section 5 concludes the
paper with key takeaways and future directions.
2. Related Work
Retrieval-augmented generation (RAG) has become a central paradigm for enhancing large language
models with access to external knowledge sources. Early RAG pipelines operated primarily over textual
inputs derived via Optical Character Recognition (OCR), with minimal attention paid to document
layout or visual structure. Lin and Byrne introduced one of the earliest RAG architectures based
on OCR outputs, integrating Dense Passage Retrieval (DPR) with a generative language model for
open-domain question answering over scanned texts [ 14]. Their work demonstrated that standard
dense retrieval methods could be extended to semi-structured documents with text extraction. To
better handle structurally rich documents, several works have incorporated layout-aware techniques.
Yepeset al.proposed a structure-guided chunking strategy that segments documents into semantically
distinct regions, such as titles, body text, and tables, using computer vision and NLP methods, thereby
preserving some spatial information during retrieval [ 9]. However, OCR-based pipelines, even with
such enhancements, remain constrained by transcription errors and the inherent flattening of spatial
layout. Zhanget al.showed that such limitations can significantly degrade both retrieval accuracy and
generation quality, especially in documents where meaning is intrinsically tied to structure [5].
Recognizing these persistent limitations, recent work has explored multimodal retrieval pipelines
that directly encode visual signals. Chenet al.introduced MuRAG, a memory-augmented architecture
capable of reasoning over both image and text embeddings [ 15]. Maet al.proposed a unified embedding
space for document screenshots, enabling dense retrieval over raw visual input [ 11]. Similarly, Riedler
et al.applied CLIP-based retrieval to visually complex documents, demonstrating the benefits of joint
vision-language representations [ 16]. Furthermore, Faysseet al.introduced ColPali, a visual late
interaction architecture that extends the ColBERT paradigm to full-page images [ 12]. By retaining
patch-level representations and applying token-wise similarity scoring at query time, ColPali achieves
state-of-the-art retrieval performance on layout-sensitive benchmarks, exemplifying the performance
gains achievable with sophisticated multimodal, late-interaction approaches.
While this body of work has significantly advanced retrieval performance across various modalities,
most evaluations have focused narrowly on effectiveness metrics such as nDCG and Recall. As RAG

systems become more powerful and complex, understanding their system-level implications—such as
indexing latency, retrieval latency, and memory footprint—becomes paramount for practical deployment,
especially at scale. In contrast to prior studies, our work offers a holistic evaluation of retrieval pipelines.
We systematically compare dense and late interaction methods across both text and visual modalities,
not only in terms of retrieval effectiveness but also with respect to these crucial Quality-of-Service
(QoS) metrics, which are essential for real-world deployment scenarios. To our knowledge, this is one
of the first comprehensive analyses directly comparing these four pipeline archetypes—Text-Dense,
Text-Late Interaction, Visual-Dense, and Visual-Late Interaction—across both retrieval quality and
detailed operational costs.
3. Proposed Approach
This work proposes a data-driven selection methodology for identifying the most suitable RAG retrieval
pipeline given a document corpus and its computational constraints. This approach is grounded in a
comprehensive evaluation of the state-of-the-art pipelines that represent key axes in modern retrieval
system design: input modality (textual vs. visual) and retrieval architecture (dense vs. late interaction).
This section outlines the design decisions behind these pipelines, detailing how they preprocess and
embed documents and how their underlying architectures handle retrieval. Figure 1 illustrates the
differences between text-based and multimodal pipelines. Text-based pipelines typically follow a
multi-stage preprocessing procedure. Raw document pages are first passed through Optical Character
Recognition (OCR) to extract textual content. Afterward, the layout analysis step is applied, segmenting
the document into semantically meaningful regions such as tables, titles, and figures. These regions
inform layout-aware chunking strategies that divide the extracted text into retrieval-ready text chunks.
An image captioning module is applied to generate natural language descriptions that can be indexed
alongside the text for chunks containing non-textual content, such as images or tables. Finally, each
chunk is embedded using a language model and stored in a vector database.
In contrast, multimodal pipelines operate directly on full-page document images. These systems
leverage a vision-language model (VLM) to process the entire page holistically, generating embeddings
that capture textual, visual, and spatial cues in a unified representation. This end-to-end approach
eliminates the need for OCR, chunking, or captioning, enabling a more direct and layout-aware document
encoding. The resulting embeddings are stored directly in the vector database.
Figure 1:Indexing Workflows of Text & Multimodal Pipelines
Figure 2 illustrates the architectural differences between dense and late interaction retrieval mecha-
nisms. In dense retrieval, the document and query are encoded into single vector representations using
a language or vision-language model. Retrieval is performed by computing the similarity between these
vectors using a distance metric such as cosine similarity. This highly efficient approach scales well,
making it well-suited for latency-sensitive applications. However, its compressed representation can
lose fine-grained alignment, particularly important in structured or layout-dependent documents. Late
interaction architectures address this limitation by preserving token-level or patch-level granularity

during indexing. Instead of compressing the input into a single embedding, these models retain multiple
embeddings per document, enabling more detailed and expressive similarity calculations at query time.
A common scoring mechanism is MaxSim, where each query token is compared with all document
tokens, and the maximum similarity scores are aggregated. While this approach improves retrieval
precision, it also increases storage requirements and retrieval latency.
Figure 2:Dense vs Late Interaction Retrieval Architectures
3.1. Proposed Pipelines
As discussed, we evaluate four retrieval pipelines that span the key design dimensions in modern RAG
systems: input modality (text vs. vision) and retrieval architecture (dense vs. late interaction).
Text-Dense.This pipeline implements a dense retrieval architecture of over-extracted text. As is
common practice, we rely on the Unstructured1off-the-shelf tool in the highest resolution settings
for OCR, layout analysis, and chunking (by-title). Afterward, for chunks that include non-textual
elements, we set up a full-fledged captioning strategy, in which we feed the visual elements to a state-of-
the-art Vision Language Model (Google-Gemini 2 0 Flash2) to obtain highly detailed textual descriptions
of the elements. Each resulting chunk is then embedded into a dense vector using a text encoder.
Text-Late Interaction.This pipeline retains the same preprocessing strategy as the dense variant but
uses a late interaction architecture to improve token-level alignment. Instead of embedding each chunk

into a single vector, it preserves multiple sub-token embeddings, allowing fine-grained comparisons
between query and document terms.
Visual-Dense.This pipeline skips OCR and processes full-page document images directly using a
vision-language model. Each image is encoded into a dense vector that jointly captures visual and textual
signals. This design allows for compact, layout-sensitive representations with minimal preprocessing.
Visual-Late Interaction.This pipeline operates on full-page images like the dense variant but uses a
late interaction retrieval mechanism. It produces multi-vector embeddings for each document patch and
performs expressive similarity scoring with patch-level granularity. This improves retrieval precision
for layout-sensitive queries at the cost of greater computational overhead.
4. Evaluation
This section evaluates the proposed pipelines regarding retrieval performance across various QoS
metrics, including memory consumption, indexing time, and retrieval latency. The remainder of this
section is organized as follows. Section 4.1 describes the experiment setting.Section 4.2 reports on
retrieval effectiveness using standard ranking metrics to compare the pipelines’ retrieval performances.
Finally, Section 4.3 presents our quality of service (QoS) evaluation, analyzing memory consumption,
indexing, and retrieval latency.
4.1. Expiriment Setting
All experiments were conducted on a single NVIDIA H100 GPU with 80GB of memory. Each retrieval
pipeline was evaluated independently using the same document corpus and query set to ensure a
fair comparison. We implemented each pipeline under the same codebase and hardware to eliminate
performance differences due to infrastructure. To ensure consistency across pipelines, all vector indices
were stored using identical Qdrant7clusters.
Datasets.
We evaluate our retrieval pipelines using the REAL-MM-RAG benchmark [ 17], a recent multimodal
dataset designed for realistic document retrieval in structured domains. REAL-MM-RAG focuses on long,
scanned documents with complex layouts. The queries are generated to reflect natural user questions
rather than direct surface-level matches, providing a more rigorous test of retrieval models. To assess
robustness under linguistic variation, each query in the dataset is rewritten up to three times using a
large language model, resulting in four levels of query phrasing. These rephrased versions introduce
increasing syntactic and lexical variation, allowing us to measure how well retrieval pipelines generalize
beyond superficial token overlap and capture true semantic intent. We focus on two challenging subsets
from this benchmark.
FinReportcontains annual financial reports from IBM between 2005 and 2023. These documents
include dense narrative content, structured financial tables, and visually segmented layouts. Queries
typically require extracting specific values, understanding table structures, and reasoning over multiple
modalities. This dataset is ideal for evaluating performance in layout-sensitive retrieval tasks.
TechReportconsists of IBM technical documentation across systems like FlashSystem and Power
Systems. These documents are primarily textual but include frequent visual components such as
block diagrams, formatted tables. The queries in this domain target procedural details and technical
descriptions, testing a model’s ability to retrieve relevant content embedded in text and images.
These datasets were chosen because they enable a fine-grained evaluation of how visual modality
influences retrieval performance. Specifically, each document’s proportion of visual content varies in
1www.unstructured.io
2https://ai.google.dev/gemini-api/docs/models
3https://qdrant.tech

(a) FinReport
(b) TechReport
Figure 3:Visual Component Rate Histogram across Datasets
both datasets, providing a balanced distribution of visually sparse and rich documents. This makes
it possible to observe how retrieval pipelines perform under different visual conditions. As shown in
Figure 5, both datasets exhibit a broad range of visual content levels.
Models
Each pipeline is built to represent the state-of-the-art, open-access models in their respective config-
urations. Specifically, text-based models were chosen from the top entries on the MTEB leaderboard
[18], while vision-based models were selected based on rankings in the ViDoRe leaderboard [ 12]. The
following describes the specific models used for each pipeline and their underlying architectures:
•Baseline: BM25 [ 19], a sparse retrieval baseline widely used in information retrieval. BM25
scores documents based on exact token matches, using term frequency and inverse document
frequency (TF-IDF) weighting. It does not use any learned parameters or embeddings.
•Text-single: Linq-Embed-Mistral4, is built upon the Mistral-7B architecture, fine-tuned for text
retrieval tasks. It generates dense embeddings of 768 dimensions and supports a context window
of 4096 tokens. The model size is approximately 7.11 billion parameters.
•Text-multi: GTE-ModernColBERT-v15, a ColBERT-style embedding model optimized for fine-
grained token matching. It produces 128-dimensional embeddings per token, resulting in multi-

vector representations. The model comprises around 110 million parameters and supports a
context window of 8192 tokens.
•Visual-single: nomic-embed-multimodal-7b6, a dense multimodal retriever based on the Bi-Qwen
2.5 architecture, an extension of the Qwen2.5-VL-7B vision-language model [ 20]. It generates
768-dimensional dense embeddings per page.
•Visual-multi: ColQwen2.5-7b-multilingual-v1.07, a ColPali-style retriever built on the ColQwen
2.5 architecture, another extension of the Qwen2.5-VL-7B vision-language model [ 20]. Unlike its
dense counterpart, ColQwen produces multi-vector representations for text and images, enabling
more fine-grained matching between queries and document regions.
4.2. Retrieval Performance
Retrieval pipelines must ultimately be judged by their ability to return relevant content in response
to a query. To quantify retrieval effectiveness, this section evaluates each pipeline using four widely
adopted top-𝑘ranking metrics (in our case, we chose𝑘= 5):
•nDCG@5 (Normalized Discounted Cumulative Gain):Captures graded relevance by reward-
ing highly ranked relevant documents more than later in the result list.
•MRR@5 (Mean Reciprocal Rank):Reflects how early the first relevant document is retrieved,
favoring pipelines that return relevant results near the top.
•Precision@5:Measures the proportion of retrieved documents within the top 5 that are relevant.
•Recall@5:Estimates the fraction of all relevant documents that appear in the top 5 results.
These metrics provide a comprehensive view of retrieval performance, balancing ranking position,
accuracy, and completeness. Table 1 presents retrieval performance across four rephrasing levels for
each pipeline and dataset. To assess the statistical significance of these results, we adopt a two-stage
testing procedure. First, we apply group-level significance tests to determine whether any pipeline
significantly differs from the rest. If the group contains three or more pipelines, we apply aRepeated
Measures ANOVAwhen normality is satisfied, and fall back to theFriedman testotherwise. For pairwise
comparisons, we apply either thePaired t-test(for normally distributed and homogeneous samples) or
theWilcoxon Signed-Rank Test(for non-normal or heterogeneous samples). Normality is tested using
theKolmogorov-Smirnov test, and homogeneity usingLevene’s test.
Thevisual-multipipeline significantly outperforms all other methods ( 𝑝 <0.01 ) across all metrics and
rephrasing levels, highlighting its robustness and superior generalization. Notably, even thevisual-single
(dense vision-language) pipeline significantly outperforms all text-based pipelines across all rephrasing
levels and metrics, confirming that multimodal pipelines; regardless of representation granularity,
offer a clear and statistically significant advantage over purely textual models. Furthermore, while
the baselineBM25performs competitively on the original query set (occasionally surpassing some
text-based pipelines) its performance degrades very sharply under rephrasing. This consistent drop
underscoresBM25’sreliance on exact token matches and its vulnerability to lexical variation. Conversely,
the proposed pipelines exhibit moderate performance drops as query rephrasing increases, reflecting
the increased difficulty of matching semantically similar but lexically varied queries. However, this
degradation is relatively consistent across architectures, with no single pipeline type disproportionately
affected. These results suggest that vision-based retrieval maintains a consistent advantage even under
paraphrasing stress, while all pipelines are sensitive to linguistic variation.
4https://huggingface.co/Linq-AI-Research/Linq-Embed-Mistral
5https://huggingface.co/lightonai/GTE-ModernColBERT-v1
6https://huggingface.co/nomic-ai/nomic-embed-multimodal-7b
7https://huggingface.co/Metric-AI/ColQwen2.5-7b-multilingual-v1.0

Table 1
Retrieval Results Across Rephrase Levels Per Dataset
Level Pipeline FinReport TechReport
nDCG@5 MRR@5 Precision@5 Recall@5 nDCG@5 MRR@5 Precision@5 Recall@5
0BM25 0.43±0.43 0.39±0.43 0.11±0.10 0.53±0.50 0.61±0.42 0.57±0.43 0.14±0.09 0.72±0.45
text-single 0.46±0.42 0.41±0.42 0.12±0.10 0.60±0.49 0.61±0.39 0.56±0.41 0.16±0.08 0.78±0.42
text-multi 0.32±0.43 0.30±0.42 0.08±0.10 0.40±0.49 0.53±0.45 0.50±0.46 0.12±0.10 0.62±0.48
visual-single 0.66±0.39 0.61±0.41 0.16±0.08 0.80±0.40 0.73±0.37 0.69±0.39 0.17±0.07 0.85±0.36
visual-multi0.74±0.33 0.69±0.37 0.18±0.06 0.89±0.31 0.85±0.28 0.82±0.32 0.19±0.05 0.94±0.24
1BM25 0.31±0.40 0.28±0.39 0.08±0.10 0.41±0.49 0.45±0.44 0.42±0.44 0.11±0.10 0.56±0.50
text-single 0.43±0.42 0.39±0.41 0.12±0.10 0.58±0.49 0.56±0.40 0.50±0.41 0.14±0.09 0.72±0.45
text-multi 0.28±0.41 0.26±0.41 0.07±0.09 0.34±0.47 0.48±0.45 0.45±0.45 0.12±0.10 0.58±0.49
visual-single 0.59±0.42 0.54±0.43 0.14±0.09 0.72±0.45 0.63±0.40 0.59±0.42 0.15±0.08 0.77±0.42
visual-multi0.68±0.37 0.63±0.39 0.17±0.07 0.84±0.37 0.78±0.33 0.75±0.37 0.18±0.06 0.89±0.31
2BM25 0.22±0.37 0.20±0.36 0.06±0.09 0.29±0.45 0.38±0.42 0.34±0.41 0.10±0.10 0.49±0.50
text-single 0.40±0.41 0.35±0.41 0.11±0.10 0.54±0.50 0.53±0.41 0.48±0.42 0.14±0.09 0.69±0.46
text-multi 0.27±0.40 0.24±0.39 0.07±0.09 0.34±0.47 0.46±0.44 0.42±0.44 0.11±0.10 0.56±0.50
visual-single 0.52±0.42 0.47±0.43 0.13±0.09 0.66±0.47 0.58±0.42 0.53±0.43 0.14±0.09 0.71±0.45
visual-multi0.61±0.40 0.55±0.42 0.15±0.09 0.76±0.43 0.75±0.35 0.72±0.38 0.17±0.07 0.87±0.34
3BM25 0.18±0.34 0.16±0.33 0.05±0.09 0.24±0.43 0.32±0.41 0.29±0.40 0.08±0.10 0.40±0.49
text-single 0.38±0.41 0.34±0.40 0.10±0.10 0.51±0.50 0.51±0.41 0.46±0.42 0.13±0.09 0.67±0.47
text-multi 0.23±0.38 0.21±0.37 0.06±0.09 0.30±0.46 0.42±0.44 0.39±0.43 0.11±0.10 0.53±0.50
visual-single 0.48±0.43 0.43±0.43 0.12±0.10 0.61±0.49 0.55±0.42 0.50±0.43 0.14±0.09 0.68±0.47
visual-multi0.57±0.41 0.52±0.42 0.14±0.09 0.72±0.45 0.73±0.36 0.68±0.39 0.17±0.07 0.85±0.36
All pipelines exhibit a moderate performance drop as query rephrasing increases, reflecting the
increased difficulty of matching semantically similar but lexically varied queries. However, this degra-
dation is relatively consistent across architectures, with no single pipeline type disproportionately
affected. These results suggest that vision-based retrieval maintains a consistent advantage even under
paraphrasing stress, while all pipelines are sensitive to linguistic variation.
To illustrate the strengths and limitations of different retrieval pipelines, two qualitative case studies
are examined. These examples highlight how performance is affected across modality (text vs. vision)
and retrieval architecture (dense vs. late interaction). The first case study comes from our FinRe-
port dataset and focuses on the impact of modality choice, particularly how OCR-induced errors can
compromise retrieval accuracy in text-based pipelines.
Case 1: Correct Context
Q: What changed IBM’s company debt from2004 to 2005?
Year2005 2004
Total company debt $22,641 $22,927
Non-Global Financing debt $2,142 $607
Non-Global Financing debt
capitalization 6.7% 2.1%
Case 1: OCR output
Total company debt $22,641 $22,927
Non-Global Financing debt $2,142 $607
Non-Global Financing debt
capitalization 6.7% 2.1%
In this example, the query asks about the change in IBM’s total company debt between 2004 and 2005.
However, the years in the corresponding financial table are entirely omitted during OCR preprocessing.
As a result, both text pipelines fail to locate the correct context, retrieving irrelevant passages instead.
In contrast, the visual pipelines operate directly on the document image and successfully place the

correct page at the top of the ranking. This example underscores how early-stage OCR failures can
cascade through text-based retrieval systems and highlights the robustness of vision-based models in
preserving layout-dependent information.
The second case study examines the effects of retrieval architecture, particularly the distinction
between dense and late interaction models in the face of linguistic variation. In this example, all pipelines
successfully retrieve the correct document when the query explicitly includes the term bandwidth,
which aligns closely with the vocabulary used in the document. However, when the query is rephrased
using the semantically equivalent phrase “maximum amount of data that can be transferred per unit of
time”, only the late interaction models retain the correct context at the top rank. The dense models, in
contrast, fail to match the rephrased version due to their reliance on surface-level term overlap in fixed
vector representations. This case highlights a key limitation of dense retrieval in real-world scenarios
where query phrasing is highly variable. It demonstrates how late interaction architectures offer greater
robustness by enabling finer-grained token-level comparisons that better capture semantic nuance.
Case 2: Relevant Document Context
With this setting, the Bronze customers can reach up to 1,000 Mbps, but they have to share the
maximumbandwidthwith all the other customers at the Bronze level. The Silver customers can
reach up to 1,000 Mbps and do not have to share their limit with other customers in their perfor-
mance class. The Gold customers are unlimited because they are not part of any performance
class.
Case 2: Original Query
What is thebandwidthlimit for Bronze customers in IBM FlashSystem A9000?
Case 2: Rephrased Query
In the IBM FlashSystem A9000, what is themaximum amount of data that can be transferred
per unit of timefor users with a Bronze subscription?
Figure 4 breaks down retrieval performance by answer modality, referring to the type of document
component in which the relevant information was embedded. Queries are labeled astext-onlywhen the
answer resides within textual elements such as paragraphs or titles, and asvisual-onlywhen the answer
is found within visual elements like tables or images. This figure further dissects the pipeline strengths.
For "text-only" queries (left column of subplots for each dataset), we anticipate that while all pipelines
might perform reasonably well, theVisual-multiandVisual-singlepipelines leveraging advanced
VLMs may still hold an edge due to their holistic page understanding, potentially capturing contextual
cues even from surrounding visual layout that text-only models might miss. TheText-singleand
Text-multipipelines are expected to be strong contenders here, with Text-multi potentially showing
greater robustness to rephrasing due to its late-interaction mechanism. Conversely, for "visual-only"
queries (right column of subplots), a more significant performance gap is expected.Visual-multi
should dominate, adeptly handling queries requiring interpretation of tables or diagrams directly.
Visual-singleis also expected to perform well, significantly outperforming text-based pipelines. The
text-based pipelines (Text-singleandText-multi) will likely struggle considerably on "visual-only"
queries, their success is highly dependent on the quality of OCR and any generated captions for those
visual elements. Performance degradation across rephrasing levels (0 to 3) is anticipated for all query
types and pipelines, but the visual pipelines, particularlyVisual-multi, might exhibit more graceful
degradation on "visual-only" queries due to their richer representations. Differences between FinReport
and TechReport would also be telling: the more table-intensive FinReport dataset is likely to highlight
the superiority of visual pipelines for "visual-only" queries, whereas in TechReport, with its mix of text
and diagrams, the distinctions might be nuanced but still favor visual approaches for visual queries.

(a) FinReport
(b) TechReport
Figure 4:Retrieval Performance across Rephrase Levels for Text-Only (left) and Visual-Only (right) Queries for
both Datasets
Figure 5 breaks down retrieval performance (nDCG@5) by document visual content proportion,
with subplots for rephrasing levels (0 to 3). Visual-multi leads, especially in TechReport. Text-single
is comparable to Visual-single on mostly textual documents but degrades sharply with more visual
content (especially in FinReport). Interestingly, Text-multi, despite underperforming Visual-single in
low-visual contexts, shows some resilience or even stronger results than Text-single in highly visual
documents (FinReport, Level 0-1), suggesting its finer-grained token interactions might better navigate
complex visual layouts indirectly via OCRed textual cues from those layouts.

(a) FinReport
(b) TechReport
Figure 5:Retrieval Performance across Visual Complexity for all Rephrase Levels (0-Top Left, 3-Bottom Right)

4.3. Quality of Service (QoS)
Deployment of RAG systems demands careful consideration of system-level efficiency. To capture these
concerns, each pipeline is evaluated across three key Quality of Service (QoS) dimensions: memory
consumption, indexing latency, and retrieval latency. Memory consumption refers to the total size of
the vector index stored in the database. Indexing latency measures the end-to-end time required to
preprocess, embed, and upsert documents into the retrieval system. Retrieval latency captures the time
needed to embed a query and retrieve the top- 𝑘most relevant documents. Together, these metrics
reflect real-world constraints such as hardware cost, scalability, and responsiveness, factors critical
when deploying retrieval systems over large document corpora.
Table 2 presents the memory consumption of each pipeline. Due to their multi-vector representations,
we can observe that late interaction models consume significantly more memory than their dense
counterparts, which store embeddings at the token or patch level. This granularity allows for improved
retrieval fidelity but comes at the cost of larger index sizes. Notably,visual-multiexceeds 5GB in memory
usage, more than 10 times that of the compactvisual-densemodel. Interestingly, despite operating at
higher embedding dimensionality, the visual-dense pipeline remains more storage-efficient overall. It
encodes entire pages into a single vector, avoiding the cumulative overhead introduced by chunk-wise
embeddings in text-based pipelines. Showcasing its scalability advantage in large-scale settings.
Table 2
Total Memory Consumption for both Datasets
Pipeline Memory (MB)
Text Single 390.00
Text Multi 1079.00
Visual Single340.00
Visual Multi 5068.80
Tables 3 and 4 show the average indexing workflows for both our datasets, broken down into
preprocessing, embedding, and upserting stages. As expected, text-based pipelines incur substantial
preprocessing overhead due to OCR, layout parsing, and captioning. Despite this, their embedding and
upsert times remain relatively lower, particularly for dense variants. In contrast, visual pipelines skip
preprocessing entirely, enabling faster end-to-end indexing, especially for the dense variant. However,
due to the higher dimensionality, the visual multi-pipeline requires significantly longer upsert durations.
These results highlight a key trade-off: text pipelines are preprocessing-heavy but compact, while visual
pipelines are plug-and-play but slower to upload.
Table 3
Average Indexing Workflow in Seconds -FinReport
Stage Text Single Text Multi Visual Single Visual Multi
Preprocess 10.23 10.230.00 0.00
Embedding0.020.18 0.72 0.71
Upsert1.243.26 5.69 6.04
Total 11.49 13.676.416.75
Table 5 reports the average retrieval latency across all pipelines, broken down into query embedding
and retrieval components. As expected, dense pipelines,text-singleandvisual-single, exhibit significantly
lower latency, with total times under one second. In contrast, late interaction pipelines introduce a
significant performance gap. Whiletext-multiremains within a negligible range,visual-multiincurs
exceptionally high retrieval times, exceeding 20 seconds per query. This discrepancy reflects the
computational overhead of multi-vector retrieval at scale, particularly for high-dimensional vision-
based embeddings.

Table 4
Average Indexing Workflow in Seconds -TechReport
Stage Text Single Text Multi Visual Single Visual Multi
Preprocess 10.70 10.700.00 0.00
Embedding 0.510.190.91 1.07
Upsert1.892.03 6.93 9.00
Total 13.10 12.927.8410.07
Table 5
Average Retrieval Latency in Seconds
Pipeline TechReport FinReport
Embedding Retrieval Total Embedding Retrieval Total
Text Single 0.040.50 0.540.030.40 0.43
Text Multi0.012.99 3.010.014.19 4.20
Visual Single 0.05 0.58 0.63 0.05 0.55 0.60
Visual Multi 0.05 22.53 22.58 0.03 27.35 27.39
5. Discussion
Our comprehensive evaluation of textual and multimodal RAG retrieval pipelines across dense and late
interaction architectures yields several notable trends and practical insights.
First, the Text-multi (late interaction) pipeline consistently underperformed not only its visual
counterparts but also the simpler Text-dense pipeline across most retrieval effectiveness metrics, while
still incurring higher memory usage and slower retrieval latency than Text-dense. For instance, it
achieved a mean nDCG@5 of only 0.32 on the FinReport dataset (Level 0 rephrase), significantly lower
than Text-single (0.46) and Visual-single (0.66). In contrast, among the visual pipelines, Visual-multi (late
interaction) achieved the highest retrieval performance overall, demonstrating the power of fine-grained,
patch-level visual understanding. However, this superior effectiveness comes at a steep operational
cost, particularly in terms of memory footprint (over 5GB) and retrieval latency (exceeding 20 seconds
per query), making it potentially less suitable for resource-constrained or latency-sensitive scenarios.
Interestingly, our results challenge the common assumption that vision-based pipelines are invariably
slower and more memory-intensive across the board. The Visual-single (dense) pipeline emerged as
a compelling alternative, delivering significantly better retrieval performance than both text-based
pipelines (e.g., 0.66 nDCG@5 on FinReport vs. 0.46 for Text-single and 0.32 for Text-multi) while also
offering comparable or superior indexing throughput and the lowest memory consumption of all tested
pipelines. This positions Visual-single as an excellent candidate for many real-world applications
seeking a balance of effectiveness and efficiency.
Furthermore from our QoS results; we can observe that, embedding time itself was nearly negligible
across all pipelines. The primary bottlenecks varied by modality: visual pipelines were predominantly
constrained by the embedding and vector database upsert phase (especially Visual-multi), whereas text
pipelines were bottlenecked by the extensive preprocessing phase (OCR, layout analysis, captioning).
6. Future Work
Our evaluation also revealed an important consideration for future benchmark development. Existing
retrieval benchmarks, even those labeled as multimodal, often predominantly emphasize the retrieval
of semantic answers that are textual, even if those answers are located within visual elements like
charts or tables. This can create a bias where the unique challenges and capabilities of retrieving
genuinely non-textual, visually grounded information are underrepresented with text retrieval from

images. Future work should aim to address this gap by developing more rigid evaluation methodologies
and datasets that specifically target the retrieval of diverse visual information, moving beyond text
embedded in images.
Complementing this direction, we also plan to explore a hybrid retrieval strategy that allocates
documents to the most suitable indexing pipeline based on their structure and retrieval needs. For
example, visually complex documents containing tables or figures may be better indexed using late-
interaction visual models to improve retrieval performance, while simpler documents may be handled
more efficiently with dense pipelines. At query time, the system retrieves results from all pipelines
and fuses the outputs, aiming to reduce memory usage and retrieval latency without significantly
compromising retrieval quality.
In summary, this study provides a data-driven foundation for selecting appropriate RAG retrieval
pipelines. Practitioners can use these findings—weighing the trade-offs between visual/textual modality,
dense/late-interaction architecture, retrieval effectiveness, and QoS metrics—to make informed decisions
tailored to their specific document corpora, task requirements, and operational constraints. These
insights contribute to the ongoing effort to build more efficient, effective, and practically deployable
RAG systems.
Declaration on Generative AI
The author(s) have not employed any Generative AI tools in the preparation of this work.
References
[1]S. Setty, H. Thakkar, A. Lee, E. Chung, N. Vidra, Improving retrieval for rag based question
answering models on financial documents, arXiv preprint arXiv:2404.07221 (2024).
[2]K. K. Y. Ng, I. Matsuba, P. C. Zhang, Rag in health care: a novel framework for improving
communication and decision-making by addressing llm limitations, NEJM AI 2 (2025) AIra2400380.
[3]N. Pipitone, G. H. Alami, Legalbench-rag: A benchmark for retrieval-augmented generation in the
legal domain, arXiv preprint arXiv:2408.10343 (2024).
[4]R. Smith, An overview of the tesseract ocr engine, in: Ninth international conference on document
analysis and recognition (ICDAR 2007), volume 2, IEEE, 2007, pp. 629–633.
[5]J. Zhang, Q. Zhang, B. Wang, L. Ouyang, Z. Wen, Y. Li, K.-H. Chow, C. He, W. Zhang, Ocr hinders
rag: Evaluating the cascading impact of ocr on retrieval-augmented generation, arXiv preprint
arXiv:2412.02592 (2024).
[6]W. Yimyam, M. Ketcham, T. Jensuttiwetchakult, S. Hiranchan, P. Pramkeaw, N. Chumuang,
Enhancing and evaluating an impact of ocr and ontology on financial document checking process,
in: 2020 15th International Joint Symposium on Artificial Intelligence and Natural Language
Processing (iSAI-NLP), IEEE, 2020, pp. 1–6.
[7]Y. Wu, A. Kirillov, F. Massa, W.-Y. Lo, R. Girshick, Detectron2, https://github.com/facebookresearch/
detectron2, 2019.
[8]B. Pfitzmann, C. Auer, M. Dolfi, A. S. Nassar, P. W. J. Staar, Doclaynet: A large human-annotated
dataset for document-layout analysis (2022). URL: https://arxiv.org/abs/2206.01062. doi: 10.1145/
3534678.353904.
[9]A. J. Yepes, Y. You, J. Milczek, S. Laverde, R. Li, Financial report chunking for effective retrieval
augmented generation, arXiv preprint arXiv:2402.05131 (2024).
[10] R. Zhao, H. Chen, W. Wang, F. Jiao, X. L. Do, C. Qin, B. Ding, X. Guo, M. Li, X. Li, et al., Retrieving
multimodal information for augmented generation: A survey, arXiv preprint arXiv:2303.10868
(2023).
[11] X. Ma, S.-C. Lin, M. Li, W. Chen, J. Lin, Unifying multimodal retrieval via document screenshot
embedding, arXiv preprint arXiv:2406.11251 (2024).

[12] M. Faysse, H. Sibille, T. Wu, B. Omrani, G. Viaud, C. Hudelot, P. Colombo, Colpali: Efficient
document retrieval with vision language models, arXiv preprint arXiv:2407.01449 (2024).
[13] O. Khattab, M. Zaharia, Colbert: Efficient and effective passage search via contextualized late
interaction over bert, in: Proceedings of the 43rd International ACM SIGIR conference on research
and development in Information Retrieval, 2020, pp. 39–48.
[14] W. Lin, B. Byrne, Retrieval augmented visual question answering with outside knowledge, arXiv
preprint arXiv:2210.03809 (2022).
[15] W. Chen, H. Hu, X. Chen, P. Verga, W. W. Cohen, Murag: Multimodal retrieval-augmented
generator for open question answering over images and text, arXiv preprint arXiv:2210.02928
(2022).
[16] M. Riedler, S. Langer, Beyond text: Optimizing rag with multimodal inputs for industrial applica-
tions, arXiv preprint arXiv:2410.21943 (2024).
[17] N. Wasserman, R. Pony, O. Naparstek, A. R. Goldfarb, E. Schwartz, U. Barzelay, L. Karlinsky,
Real-mm-rag: A real-world multi-modal retrieval benchmark, arXiv preprint arXiv:2502.12342
(2025).
[18] K. Enevoldsen, I. Chung, I. Kerboua, M. Kardos, A. Mathur, D. Stap, J. Gala, W. Siblini, D. Krzemiński,
G. I. Winata, et al., Mmteb: Massive multilingual text embedding benchmark, arXiv preprint
arXiv:2502.13595 (2025).
[19] S. Robertson, H. Zaragoza, et al., The probabilistic relevance framework: Bm25 and beyond,
Foundations and Trends®in Information Retrieval 3 (2009) 333–389.
[20] A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, et al., Qwen2.
5 technical report, arXiv preprint arXiv:2412.15115 (2024).