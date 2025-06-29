# MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering

**Authors**: Chinmay Gondhalekar, Urjitkumar Patel, Fang-Chun Yeh

**Published**: 2025-06-25 20:37:20

**PDF URL**: [http://arxiv.org/pdf/2506.20821v1](http://arxiv.org/pdf/2506.20821v1)

## Abstract
Financial documents--such as 10-Ks, 10-Qs, and investor presentations--span
hundreds of pages and combine diverse modalities, including dense narrative
text, structured tables, and complex figures. Answering questions over such
content often requires joint reasoning across modalities, which strains
traditional large language models (LLMs) and retrieval-augmented generation
(RAG) pipelines due to token limitations, layout loss, and fragmented
cross-modal context. We introduce MultiFinRAG, a retrieval-augmented generation
framework purpose-built for financial QA. MultiFinRAG first performs multimodal
extraction by grouping table and figure images into batches and sending them to
a lightweight, quantized open-source multimodal LLM, which produces both
structured JSON outputs and concise textual summaries. These outputs, along
with narrative text, are embedded and indexed with modality-aware similarity
thresholds for precise retrieval. A tiered fallback strategy then dynamically
escalates from text-only to text+table+image contexts when necessary, enabling
cross-modal reasoning while reducing irrelevant context. Despite running on
commodity hardware, MultiFinRAG achieves 19 percentage points higher accuracy
than ChatGPT-4o (free-tier) on complex financial QA tasks involving text,
tables, images, and combined multimodal reasoning.

## Full Text


<!-- PDF content starts -->

arXiv:2506.20821v1  [cs.CL]  25 Jun 2025MultiFinRAG: An Optimized Multimodal Retrieval-Augmented
Generation (RAG) Framework for Financial Question Answering
Preprint Copy
Chinmay Gondhalekar
S&P Global Ratings
New York, USA
chinmay.gondhalekar@spglobal.comUrjitkumar Patel
S&P Global Ratings
New York, USA
urjitkumar.patel@spglobal.comFang-Chun Yeh
S&P Global Ratings
New York, USA
jessie.yeh@spglobal.com
Abstract
Financial documents—such as 10-Ks, 10-Qs, and investor presenta-
tions—span hundreds of pages and combine diverse modalities, in-
cluding dense narrative text, structured tables, and complex figures.
Answering questions over such content often requires joint reason-
ing across modalities, which strains traditional large language mod-
els (LLMs) and retrieval-augmented generation (RAG)[ 10] pipelines
due to token limitations, layout loss, and fragmented cross-modal
context. We introduce MultiFinRAG , a retrieval-augmented genera-
tion framework purpose-built for financial QA. MultiFinRAG first
performs multimodal extraction by grouping table and figure images
into batches and sending them to a lightweight, quantized open-
source multimodal LLM, which produces both structured JSON
outputs and concise textual summaries. These outputs, along with
narrative text, are embedded and indexed with modality-aware
similarity thresholds for precise retrieval. A tiered fallback strat-
egythen dynamically escalates from text-only to text+table+image
contexts when necessary, enabling cross-modal reasoning while
reducing irrelevant context. Despite running on commodity hard-
ware, MultiFinRAG achieves 19 percentage points higher accuracy
than ChatGPT-4o[ 15] (free-tier) on complex financial QA tasks
involving text, tables, images, and combined multimodal reasoning.
Keywords
Retrieval-Augmented Generation (RAG), Multimodal Inference,
Large Language Models, Natural Language Processing, Financial
QA, PDF Document Understanding, Deep Learning
1 Introduction
Modern financial filings often span over a hundred pages and inte-
grate dense narrative text, structured tables, and complex graphical
elements. For instance, a recent Morgan Stanley 10-Q contains
approximately 120 pages, more than 275 tables, and nearly 200
figures. Accurate question answering (QA) on such filings is criti-
cal for analysts, auditors, and automated financial agents involved
in risk monitoring, compliance, and investment decision-making.
However, addressing queries over these documents poses signifi-
cant challenges for LLMs and conventional retrieval-augmented
generation (RAG) pipelines, with two core issues:
Length and cost: Document length far exceeds LLM token limits,
driving up API costs and making end-to-end processing infeasible.Mixed formats: Structured tables and visual figures lose their
inherent relationships when naively flattened into plain text, ob-
scuring the numerical context crucial for financial QA. Retrieval-
augmented generation (RAG) mitigates the length issue by retriev-
ing only relevant passages, but standard RAG pipelines typically:
•Use fixed-size, non-overlapping text chunks, often fragmenting
coherent explanations or splitting numeric context across bound-
aries.
•Treat tables and charts as unstructured text, sacrificing tabular
relationships and visual insights.
•Employ static top- 𝑘retrieval, which can return redundant or
marginally relevant snippets and dilute answer quality.
We introduce MultiFinRAG , a RAG framework tailored for financial
QA with three key advances:
•Batch multimodal extraction : Small groups of table and chart
images are fed to a lightweight multimodal LLM, which returns
structured JSON plus concise summaries, preserving numerical
relationships and visual insights for indexing.
•Semantic chunk merging & thresholded retrieval : Over-
segmented text chunks are recombined based on embedding sim-
ilarity and indexed in FAISS[ 4] with modality-specific thresholds
(e.g. 80% for text, 65% for images) to filter marginal contexts.
•Tiered fallback strategy : Queries first leverage high-similarity
text; if insufficient hits remain (below a preset count or similarity),
retrieval automatically escalates to table and image context and
combines text + table + image context, ensuring comprehensive
coverage.
2 Related Work
The integration of LLMs into real-world decision-making pipelines
has propelled the development of Retrieval-Augmented Genera-
tion (RAG) systems to address knowledge limitations of LLMs,
especially in domain-specific or constantly evolving contexts. Fi-
nancial documents, particularly regulatory filings like 10-Ks and
8-Ks, present unique challenges for RAG: their content is lengthy,
multimodal (text, tables, charts), and often semantically distributed
across sections. Existing RAG systems often fail to answer questions
that require synthesizing information from multiple modalities or
sources.
Open-Source Multimodal Capabilities Recent advancements in
open-source multimodal models—such as Meta’s Llama-3.2-11B-
Vision-Instruct [ 24], Google’s Gemma [ 23], and DeepSeek [ 12]-have
made it feasible to build robust, multimodal RAG pipelines without
relying on proprietary tools. RAG has matured significantly from its
early design as a simple retrieval-augmented QA pipeline. Modern
1

Chinmay Gondhalekar, Urjitkumar Patel, and Fang-Chun Yeh
Figure 1: MultiFinRAG pipeline: knowledge base construc-
tion from PDF text, tables, and figures
frameworks have introduced dynamic retrieval strategies, smarter
chunking mechanisms, and hierarchical or adaptive organization
of retrieved content.
Approaches like SELF-RAG [ 2] empower LLMs to assess and
refine their own retrievals through self-reflection. T-RAG [ 5] orga-
nizes retrieval hierarchically through tree-based entity structures,
while MoG (Mix-of-Granularity) [ 31] and DRAGIN [ 22] dynamically
control chunk sizes and retrieval timing based on query character-
istics and generation behavior. Late Chunking [ 7] further enhances
retrieval alignment by deferring chunk segmentation until after
document embedding.
Dense Passage Retrieval (DPR) [ 9], built on a dual-encoder ar-
chitecture, remains a cornerstone in RAG systems for open-domain
QA, outperforming sparse retrieval baselines. Evaluation frame-
works like eRAG [ 19] and DPA-RAG [ 3] assess retrieval relevance
in the context of downstream generation, while ClashEval [ 26] and
vRAG-Eval[ 25] benchmark LLM performance under conflicting or
noisy retrievals.
PDFTriage [ 18] shows that representing structured documents
like PDFs as plain text leads to loss of layout and context, and
proposes layout-aware retrieval to improve QA over figures, tables,
and multi-page content. In finance, Smith et al. [ 21] showed that
structural segmentation improves the accuracy of retrieval, while
Fin-RAG [ 8] improves the accuracy by using tree-based retrieval
methods with meta-data clustering.
Despite these advances, current financial RAG systems remain
limited in their ability to handle questions that require coordinated
reasoning across multiple modalities—text, tables, and figures. Most
approaches retrieve relevant snippets but lack mechanisms for
aligning and synthesizing information across formats.
To bridge the gap in answering complex, multimodal financial
questions, we introduce MultiFinRAG - designed for integrated
reasoning across text, tables, and figures. MultiFinRAG combines
approximate nearest-neighbor retrieval, modality-aware similar-
ity filtering, and a tiered fallback strategy to handle diverse query
contexts with precision. Unlike prior RAG systems, it is specifically
tailored for financial documents, emphasizing empirical fidelity and
structured reasoning. By preserving alignment between textual,
Figure 2: Generated table description and JSON
numerical, and visual information in long, layout-rich PDFs, Multi-
FinRAG addresses a key limitation in existing retrieval-augmented
approaches.
3 Methodology
3.1 Models and Tools Used
We utilize a combination of specialized and general-purpose models
to handle the multimodal nature of financial documents:
•Table Detection: Detectron2Layout [ 27], pre-trained on the
TableBank[ 11] dataset, identifies and extracts tabular structures.
•Image Detection: Pdfminer’s[ 20] layout analysis
(LTImage/LTFigure ) locates charts and diagrams within
PDFs.
•Multimodal Summarization: Quantized Gemma3: 12B [ 23]
and LLama 3.2:11B [ 1]vision models were tried for batch summa-
rization of tables and images into structured JSON and plain text
summaries, as these are state of the art open source multi-modal
capability models.
•Embedding Generation: BAAI/bge-base-en-v1.5 [ 29] model
using SentenceTransformer generates embeddings for semantic
text chunks, tables, and image summaries.
•Approximate Retrieval: FAISS (IVF-PQ index) provides scal-
able, efficient approximate nearest-neighbor searches.
2

MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering
•LLMs Used: We integrate two quantized open-source multi-
modal LLMs via the Ollama framework, chosen for their strong
fusion of text, tables, and images under constrained hardware:
•Gemma3:12B [23]: demonstrates best-in-class multimodal rea-
soning across narrative, tabular, and visual inputs. Quantiza-
tion reduces its 24 GB GPU footprint by ≈65% and lowers
inference latency, while preserving over 90% of its original
accuracy.
•LLaMA-3.2-11B (vision-Instruct) [1]: an open-source vision-
capable LLM with an integrated image encoder. When quan-
tized via Ollama, it achieves similar memory and speed im-
provements, enabling seamless single-GPU deployment.
3.2 Baseline Framework
To evaluate the performance and efficacy of our proposed method,
we establish a robust baseline using a conventional RAG pipeline:
•Basic RAG Setup: Standard retrieval-augmented generation
pipeline using fixed-size text chunks without semantic merging.
•Embedding & Retrieval: Uses the same BAAI/bge-base-en-v1.5
embeddings as MultiFinRAG, but retrieves from fixed chunks
via FAISS IVF-PQ without any tiered logic.
•No Multimodal Parsing: Charts and tables are not explicitly
summarized: Visual and tabular elements are flattened into raw
text or ignored, with no structured JSON conversion or image
captioning.
3.3 Proposed MultiFinRAG System
3.3.1 System Overview. Figure 1 illustrates our pipeline. Each PDF
𝐹𝑖is segmented into three sets of retrievable chunks:
𝐶𝑖=𝐶text
𝑖∪𝐶table
𝑖∪𝐶image
𝑖,
Figure 3: Image Summary Generation Flowchart
Figure 4: Example for type 2 questions
where𝐶text
𝑖are semantically coherent text passages, and 𝐶table
𝑖,
𝐶image
𝑖are table and figure regions converted via a multimodal
LLM. All chunks are embedded and stored in an approximate FAISS
index. A query 𝑄triggers a tiered retrieval (text-only then text
+ table and image), automatically escalating whenever context is
insufficient, before a final LLM answer generation.
3.3.2 Semantic Chunking & Indexing.
To avoid arbitrary splits and capture coherent semantic units, we:
(1)Sentence segmentation: split narrative into sentences 𝑆=
{𝑠1,...,𝑠 𝑛}.
(2)Sliding windows: with window size 𝑤and overlap 𝑜, form
overlapping blocks
𝐵𝑖={𝑠𝑖,...,𝑠 𝑖+𝑤−1}, 𝑖=1,1+(𝑤−𝑜), ...
(3)Embedding & breakpoints: embed each sentence 𝑒𝑗=𝐸(𝑠𝑗),
compute
𝑑𝑗=1−𝑒𝑗·𝑒𝑗+1
∥𝑒𝑗∥∥𝑒𝑗+1∥, 𝑗=1,...,𝑤−1,
and mark any 𝑑𝑗above the 95th percentile of {𝑑𝑗}as a split
point.
(4)Chunk formation: split each block 𝐵𝑖at its breakpoints into
semantic chunks, collecting all into 𝐶text.
(5)Chunk merging: compute pairwise cosine similarities among
the resulting chunks, then greedily merge any pairs whose simi-
larity exceeds a high threshold (e.g. 0.85) to reduce redundancy.
(6)Approximate indexing: embed the final set of chunks and
build a FAISS HNSW (or IVF-PQ) index, enabling sub-second
𝑘-NN lookups at scale |𝐶|>105.
Context-size reduction: By grouping semantically coherent sen-
tences and then merging near-duplicate chunks, we cut the total
number of chunks—and hence the total token count—sent to the
LLM by roughly 40–60% on average. This directly translates into
lower computational costs, reduced latency, and more efficient end-
to-end QA without sacrificing retrieval quality.
3.3.3 Batch Multimodal Extraction. Algorithm 1 details our unified
pipeline for ingesting tables and figures from each PDF. After initial
region detection (via Detectron2Layout and pdfminer) and semantic
3

Chinmay Gondhalekar, Urjitkumar Patel, and Fang-Chun Yeh
text chunking (lines 1–10), we process tables and images in two
analogous, batched passes to maximize LLM throughput and ensure
100% coverage:
Algorithm 1 Knowledge Base Construction
Require: PDF docsF, batch size 𝐵, embedder 𝐸, empty indexes
Itext,Itable,Iimage
Ensure: Populated indexes per modality
1:for all𝐹∈Fdo
2: Region detection:
3: Render pages→𝑃𝑇,𝑃𝑉
4: Detect tables 𝑇←detect_tables(𝑃𝑇)
5: Detect figures 𝑉←detect_figures(𝑃𝑉)
6: Text extraction & semantic chunking:
7:𝑋←extract_text(𝐹)
8:𝑆←segment_sentences (𝑋)
9:𝐶←semantic_chunk_merge (𝑆)
10: for all𝑐∈𝐶do
11:𝑒←𝐸(𝑐); normalize(𝑒)
12:Itext.insert(𝑒,{content :𝑐})
13: end for
14: Batch-parse tables:
15: Partition𝑇into⌈|𝑇|/𝐵⌉batches
16: for all batch𝑡𝑏do
17:{(𝑑𝑒𝑠𝑐 𝑖,𝑗𝑠𝑜𝑛 𝑖)}← batch_parse_one_batch (𝑡𝑏)
18: for all(𝑑𝑒𝑠𝑐 𝑖,𝑗𝑠𝑜𝑛 𝑖)do
19: 𝑒←𝐸(𝑑𝑒𝑠𝑐 𝑖); normalize(𝑒)
20:Itable.insert 𝑒,{summary :𝑑𝑒𝑠𝑐 𝑖,json :𝑗𝑠𝑜𝑛 𝑖}
21: end for
22: end for
23: Batch-summarize images:
24: Partition𝑉into⌈|𝑉|/𝐵⌉batches
25: for all batch𝑣𝑏do
26:{𝑠𝑢𝑚 𝑗}← batch_summarize(𝑣𝑏)
27: for all𝑠𝑢𝑚 𝑗do
28: 𝑒←𝐸(𝑠𝑢𝑚 𝑗); normalize(𝑒)
29:Iimage.insert(𝑒,{summary :𝑠𝑢𝑚 𝑗})
30: end for
31: end for
32:end for
Batch Table Parsing.
•Detected table regions 𝑇are first cropped with padding and saved
as individual images. We then partition 𝑇into⌈|𝑇|/𝐵⌉batches
of size at most 𝐵. For each batch 𝑡𝑏(lines 11–17):
•We construct a single multimodal prompt listing the exact file-
names in𝑡𝑏and invoke batch_parse_one_batch , which returns
(desc 𝑖,json𝑖)for each file.
•Each textual description desc 𝑖is embedded using the model 𝐸(·)
and inserted into the table index Itablealong with its parsed JSON
representation.
•If any filename fails to appear in the response—due to an LLM
omission or low confidence—we write a stub file in our dump di-
rectory and later retry with a single-image prompt, guaranteeing
no table goes unparsed.
Figure 5: Example for type 3 questions
Batch Image Summarization.
•Similarly, figure regions 𝑉(charts, diagrams, etc.) are batched
into⌈|𝑉|/𝐵⌉groups. For each batch 𝑣𝑏(lines 18–23):
•We send a batch prompt requesting a 3–6 sentence summary per
image, explicitly instructing the LLM to ignore non–data visuals
(e.g. logos, watermarks).
•The returned summaries {sum 𝑗}are embedded, normalized, and
inserted into the image index Iimage .
•Any missing summaries trigger a fallback to individual-image
summarization, ensuring complete coverage of all data-relevant
figures.
By batching up to 𝐵items per prompt, we amortize the LLM’s
per-call overhead and maintain exact filename ↔output alignment,
while the stub/fallback mechanism in each pass ensures no region is
left unprocessed. All embeddings are stored in independent FAISS
indexes for text, tables, and images, ready for downstream retrieval.
3.3.4 Tiered Retrieval & Decision Function. Let𝑛,𝑚, and𝑝denote
the minimum number of text chunks required, the number of table
chunks to fetch in the fallback, and the number of image summaries
to fetch on fallback, respectively (we use 𝑛=6,𝑚=4,𝑝=3);
these values were determined through trial and error on a holdout
development set.
(1)Text-only retrieval.
T=
𝑐∈𝐶textcosine 𝑒𝑄, 𝐸(𝑐)≥𝜃text	
.
If|T| ≥𝑛, issue a single LLM call with Tas context and
terminate.
(2)Table fallback.
Ttbl=Top𝑚
𝑐∈𝐶tablecosine 𝑒𝑄, 𝐸(𝑐)≥𝜃table	
.
(3)Image fallback.
Timg=Top𝑝
𝑐∈𝐶imagecosine 𝑒𝑄, 𝐸(𝑐)≥𝜃image	
.
(4)Combined prompt. Concatenate any non-empty sets among
T,Ttbl, andTimg(including each table’s JSON + summary), then
issue the final LLM call.
All retrievals use an approximate FAISS index, and each LLM
call carries a system prompt instructing it to explicitly defer (“insuf-
ficient information”) rather than hallucinate. We instrument every
stage with wall-clock timers and tqdm progress bars to monitor
end-to-end performance.
3.3.5 Threshold Calibration via Decision Function. To select opti-
mal similarity cut-offs for each modality, we ran a decision function
over a held-out set of sample queries and measured both retrieval
4

MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering
Figure 6: Examples for type 4 questions
quality (precision of retrieved contexts) and end-to-end QA accu-
racy:
(1)Text threshold sweep: Vary𝜃textfrom 0.55 to 0.85 in
steps of 0.05; at each value, retrieve all text chunks with
cos(𝐸(𝑄),𝐸(𝑐))≥𝜃text, feed the top- 𝑘to the LLM, and record
answer accuracy.
(2)Table & image threshold sweep: Keeping𝜃textfixed, vary
(𝜃table,𝜃image)independently from 0.55 to 0.75; at each pair,
retrieve table and image summaries above their thresholds,
query the LLM, and record accuracy.
(3)Decision criterion: Choose the triplet (𝜃text,𝜃table,𝜃image)
that maximized a combined metric of context relevance and QA
accuracy, while keeping per-query context size under budget.
The final thresholds selected by this procedure were
𝜃text=0.70, 𝜃 table=0.65, 𝜃 image =0.55.
These values balance precision and coverage, ensuring text-only
retrieval succeeds when possible, and gracefully falling back to
richer table/image contexts when needed.
4 Evaluation
Note on Retrieval Metrics. This study focuses on end-to-end QA
accuracy. Detailed information-retrieval statistics (precision, recall,etc.) for the retrieval components were collected during develop-
ment but are omitted here due to space constraints; we plan to
report them in a follow-up paper.
4.1 Dataset
For our evaluation, we collected financial documents from various
companies, including Form 10-Q (Quarterly Report), Form 10-K
(Annual Report), Form 8-K with EX-99.1 (Current Report), and
DEF 14A (Proxy Statement). These documents were obtained by
accessing the SEC’s EDGAR database at https://www.sec.gov/edgar/
search/ and utilizing the SEC API to download and convert the
forms into PDF files. We then manually crafted questions to ensure
they met our expected difficulty levels and quality standards.
To ensure that the pipeline accurately comprehends the files, we
developed four distinct types of questions with varying levels of
difficulty. The simplest are text-based questions, followed by image-
based and table-based questions. The most challenging category
involves questions that require the interpretation of both text and
images or tables. Details of each question type are provided below.
Question Distribution - We created a total of 300 evaluation
questions, distributed as follows (see Table 1):
•146 text-based
•42 image-based
•72 table-based
•40 requiring both text and image/table reasoning
5

Chinmay Gondhalekar, Urjitkumar Patel, and Fang-Chun Yeh
Table 1: MultiFinRAG Performance Comparison Across Question Types
Question Number of Baseline Baseline MultiFinRAG MultiFinRAG ChatGPT-4o
Type Questions with Llama with Gemma with Llama with Gemma (Free Tier)
1. Text-based 146 75.3% 76.7% 83.6% 90.4% 86.3%
2. Image-based 42 0% 0% 42.9% 66.7% 23.8%
3. Table-based 72 2.8% 5.6% 13.9% 69.4% 44.4%
4. Text+Image/Table-based 40 0% 0% 10.0% 40.0% 15.0%
Total 300 33.3% 36.7% 47.3% 75.3% 56.0%
4.1.1 Text-based Questions. This type of questions are straightfor-
ward and can be answered using the text in the financial documents.
For example, we use the sentence "A weaker U.S. dollar positively
impacted net sales by $106 million during the first quarter of fiscal
2024." from the Form 10-Q of The Home Depot, Inc. to create the
question: "How much did the weaker U.S. dollar positively impact
net sales during the first quarter of fiscal 2024 (in millions)?" The
answer is "106." We ensure that the answer is a value, term, or a
few words to reduce evaluation confusion.
4.1.2 Image-based questions. This type of questions require ana-
lyzing images and graphs in financial documents. These questions
test the framework’s ability to understand financial graphs. For
example, pie charts use different colors to represent various cate-
gories. We will create a question about categories to assess whether
the MultiFinRAG framework can capture such details. Figure 4 illus-
trates another example: using the bar chart from Morgan Stanley’s
Form 10-Q, we create the question, "During the current quarter,
within what range did the Daily 95% One-Day Total Management
Value at Risk (VaR) peak?" The correct answer is "$50 to $55." The
framework needs to capture the value for each bar and identify the
bar with the highest value to provide the correct answer. When
crafting the questions and answers, we ensure that the answers do
not appear elsewhere in the document.
4.1.3 Table-based questions. This type of questions involve inter-
preting tables in financial documents. These questions test the
framework’s ability to locate and understand tables. For instance,
in Figure 5, using a table from Morgan Stanley’s Form 10-Q, we
formulate the question: "How much restricted cash did Morgan
Stanley have as of March 31, 2025 (in millions)?" The answer is
"29,904." To correctly answer this question, the framework needs
to identify the appropriate row and column to retrieve the value.
Similar to image-based questions, we search for the value "29,904"
in the file and ensure it is not mentioned in the textual content.
4.1.4 Questions require both text and image/table. Questions re-
quire both text and image/table are the most challenging, as they
require information from both text and images or tables. For ex-
ample, in Figure 6, the first example is: "What were the net sales,
in millions, for the Lighting department’s product line as of April
28, 2024?" To answer this, one must first identify that the Lighting
department’s product line is "Decor" from the text, and then refer
to the related table to find that the value for the Decor product line
is "12,344."Another example from Figure 6 is: "What was Sasan K.
Goodarzi’s aggregate balance in the Non-Qualified Deferred Com-
pensation Plan as of July 31, 2023 (in dollars)?" This requires rec-
ognizing that the Non-Qualified Deferred Compensation Plan is
abbreviated as "NQDCP" in the table, and then locating that Sasan K.
Goodarzi’s NQDCP balance at the specified time was "$11,400,690."
4.2 Evaluation Strategy
Evaluating the output of LLM frameworks remains a significant
open research area. Traditional methods that assess whether the
generated answer exactly matches the reference answer are increas-
ingly unsuitable for evaluating LLM results. This is because it is
challenging to ensure that generated answers precisely replicate
reference answers. For instance, even when most answers in our
dataset are simple numeric values, discrepancies in units can arise,
such as a reference answer of "1 billion" versus a generated answer
of "1,000 million."
Another popular evaluation method, BERTScore [ 30], computes
a similarity score between the generated answer and the reference
answer. However, this approach is not suitable for our study. This
limitation arises because many of the questions we create pertain
to numerical values, making it illogical to embed numbers and
compare their semantic similarities.
Recently, there has been an increasing interest in evaluating
GenAI results using LLMs [ 13,14]. However, these studies all high-
light a downside: the inherent bias of the LLMs used for the eval-
uation tasks. While LLMs can be employed for evaluating large
volumes of QA pairs, the size of our dataset and resource constraints
lead us to opt for manual evaluations. This approach ensures more
accurate assessments and minimizes the risk of misjudgments.
4.3 Performance Comparison: Baselines vs.
MultiFinRAG
4.3.1 Text-based Questions. When examining the accuracy for text-
based questions in Table 1, two key observations emerge:
Firstly, there is a significant improvement in accuracy when
transitioning from the Baseline framework to MultiFinRAG. For
example, the accuracy for text-based questions using MultiFinRAG
with Gemma 3 is 90.4%, representing an improvement of 15.1%
compared to the Baseline framework with Gemma 3 (75.3%). This
increase indicates that embedding refined chunks into an IVF/PQ
FAISS index, which enables sub-second approximate k-NN lookup
at scale, is more effective than the baseline approach that retrieves
fragmented or redundant passages.
6

MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering
Secondly, Gemma 3 outperforms Llama-3.2-11B-Vision-Instruct
across both Baseline and MultiFinRAG frameworks. Specifically,
MultiFinRAG with Gemma 3 achieves 90.4% accuracy for text-based
questions, which is 6.8% higher than MultiFinRAG with Llama-3.2-
11B-Vision-Instruct (83.6%). This suggests that Gemma 3 is more
effective at identifying key answers from the retrieval trunks once
the LLMs receive them.
4.3.2 Image-based Questions. The accuracy difference for image-
based questions between the Baseline framework and MultiFinRAG
demonstrates that the Baseline framework cannot effectively extract
details from images, as anticipated. This limitation arises because
the Baseline framework only truncates and embeds text for retriev-
ing answers, resulting in weaker performance for both Baseline
with Llama-3.2-11B-Vision-Instruct and Baseline with Gemma 3 in
this question type.
In contrast, MultiFinRAG with Gemma 3 achieves higher accu-
racy (66.7%) compared to MultiFinRAG with Llama-3.2-11B-Vision-
Instruct (42.9%), indicating that Gemma 3 is more adept at describ-
ing images and graphs in the files in a detailed and comprehensive
manner.
4.3.3 Table-based Questions. Similar to image-based questions, the
Baseline framework performs poorly on table-based questions. Al-
though the Baseline framework may occasionally answer correctly
for questions based on relatively simple table structures, it generally
fails to provide accurate responses for this question type.
Moreover, when applying different LLMs to describe tables,
Gemma 3 provides significantly better descriptions than Llama-
3.2-11B-Vision-Instruct. This results in a 55.5% higher accuracy for
MultiFinRAG with Gemma 3 (69.4%) compared to MultiFinRAG
with Llama-3.2-11B-Vision-Instruct (13.9%).
4.3.4 Questions Requiring Both Text and Image/Table. Table 1 il-
lustrates that the accuracy for questions requiring both text and
image/table analysis varies significantly between MultiFinRAG
with Llama-3.2-11B-Vision-Instruct and MultiFinRAG with Gemma
3. This variance highlights the importance of the LLMs’ ability
to accurately describe graphs and tables within the MultiFinRAG
framework. Specifically, using MultiFinRAG with is Llama-3.2-11B-
Vision-Instruct too long to use? yields only 10% accuracy for these
complex questions, whereas using MultiFinRAG with Gemma 3
achieves 40% accuracy.
Additionally, the transition from the Baseline to the MultiFin-
RAG framework notably improves the accuracy rate. For instance,
accuracy increases from 0% with the Baseline framework using
Gemma 3 to 40% with MultiFinRAG using Gemma 3. This improve-
ment underscores the MultiFinRAG framework’s ability to retrieve
relevant trunks and integrate information effectively to generate
accurate final answers.
4.4 Benchmarking against ChatGPT-4o
We compare the performance of MultiFinRAG against ChatGPT-4o
across multiple question types. ChatGPT-4o [ 15] serves as a strong
benchmark due to its advanced general-purpose reasoning and
wide availability in the free tier, making it a relevant standard for
evaluating our domain-specific, multimodal QA system.
Figure 7: Accuracy Comparison by Question Type
4.4.1 Text-based Questions. In Figure 7, we observe that compared
to ChatGPT-4o, MultiFinRAG with Gemma 3 demonstrates similar
accuracy, being only 4.1% higher. However, we identified instances
where MultiFinRAG with Gemma 3 provided correct answers while
ChatGPT-4o did not. For example, consider the financial statement:
"As of March 31, 2025, and December 31, 2024, 98% of the Firm’s
portfolio of HTM securities were investment-grade U.S. agency
securities, U.S. Treasury securities, and Agency CMBS, which were
on accrual status and assumed to have zero credit losses."
The correct answer is "98%." In this case, ChatGPT-4o incorrectly
responded with "100%," likely due to the presence of multiple rele-
vant values within the document.
4.4.2 Image-based Questions. MultiFinRAG with Gemma 3
achieves over 40% higher accuracy than ChatGPT-4o for image-
based questions. For instance, in an image-based question depicted
in Figure 4, the correct answer is "$50 to $55," whereas ChatGPT-4o
inaccurately responded with "$63 to $98." This confusion likely
arises from the presence of multiple values ($63 and $98) in the text
and tables within the file, leading to incorrect information retrieval
by ChatGPT-4o.
4.4.3 Table-based Questions. When addressing table-based ques-
tions, MultiFinRAG with Gemma 3 outperforms ChatGPT-4o by
over 25% in accuracy. For example, in Figure 5, the ChatGPT-4o
incorrectly stated:
"As of March 31, 2025, Morgan Stanley had $0 million in restricted
cash. This is explicitly stated in the report: no amount is separately
listed or disclosed under restricted cash, indicating that it was not
material or not applicable at that reporting date."
However, this information is not expressed anywhere in the file,
suggesting that ChatGPT-4o’s response may be influenced by its
existing training data rather than the provided document.
4.4.4 Questions Requiring Both Text and Image/Table. Finally, for
questions requiring both text and image/table analysis, MultiFin-
RAG with Gemma 3 achieved an accuracy of 75.3%, compared to
ChatGPT-4o’s 56.0%, marking a 19.3% improvement. We observed
7

Chinmay Gondhalekar, Urjitkumar Patel, and Fang-Chun Yeh
that when the necessary information is already present in ChatGPT-
4o’s existing knowledge base, it can answer correctly. For example,
in Figure 6, Example 2, ChatGPT-4o correctly identified that the
"Non-Qualified Deferred Compensation Plan" is often abbreviated
as "NQDCP" and applied this knowledge to answer the question
accurately.
However, in cases where the required information is not familiar
or does not exist in its knowledge base, such as Example 1 in Figure
6, ChatGPT-4o struggled to integrate the information effectively,
resulting in incorrect or incomplete answers.
4.5 Efficiency and Cost
In this evaluation, we primarily assess accuracy rather than process-
ing time, as the latter is significantly influenced by the underlying
infrastructure’s capabilities, such as computational resources and
system configurations. The average processing time for MultiFin-
RAG with Gemma 3 is approximately 25 minutes when handling a
financial file consisting of a 200-page PDF with 200 tables and 150
images on Google Colab [6] T4 GPU which has 16 GB RAM [28].
Regarding cost, both the baseline and MultiFinRAG systems can
be operated using Google Colab’s free tier at the time of writing. Our
experiments were conducted using this free version, resulting in no
associated expenses. Similarly, the results obtained from ChatGPT-
4o, used as a comparative benchmark, were achieved using its free
tier, incurring no additional costs.
5 Discussion and Future Work
MultiFinRAG demonstrates strong performance and efficiency in
extracting insights from complex financial documents. However,
several avenues remain open for future enhancements, as men-
tioned below:
•Module-wise evaluation: Systematic ablations (e.g., disabling
batch extraction or tiered fallback) will help quantify each com-
ponent’s individual impact.
•User feedback scope: All 300 QA pairs were manually verified
by our team of domain experts, ensuring answer correctness;
future work will include broader user studies to assess usability
and workflow integration.
•Structured-Data Pipeline: To handle large tabular attachments
(e.g. CSV/Excel) beyond our current JSON-and-summary stage,
we are prototyping an ingestion-and-query system. It will nor-
malize heterogeneous table formats into a lightweight database,
expose a natural-language interface for precise data retrieval,
and feed the resulting rows and aggregates directly into our RAG
generator as structured context.
•Cross-Document & Longitudinal Analysis: Many financial
queries require comparisons over multiple time periods or across
related filings (e.g., 10-K vs. 10-Q). Future work will:
–Build a unified multi-document index supporting temporal
joins and trend detection;
–Develop specialized retrieval that pulls “paired” chunks (e.g.,
Q1 vs. Q2) and surfaces year-over-year deltas;
–Introduce timeline visualizations and automated narrative
summaries of longitudinal changes.•Robustness to Noise & Error Correction: PDF parsing and
OCR remain brittle in the face of low-quality scans or unusual
layouts. We intend to:
–Ensemble multiple OCR/layout engines to triangulate cell
boundaries and text;
–Apply consistency checks (e.g., row-sum invariants, unit-
sanity) to detect mis-parsed numbers;
–Use small fine-tuned correction LLMs to post-process and
validate extracted tables.
•Extended Domain Coverage: Beyond annual reports, our
methods can generalize to other financial and regulatory
documents—S-1 filings, prospectuses, and earnings-call tran-
scripts. Each domain brings new layout conventions, metadata
fields, and entity types that we plan to accommodate via modular
extraction components.
•Fine-Tuning & Domain Adaptation: Finally, we plan to fine-
tune both our retrieval and generation models on proprietary,
high-quality financial QA datasets, exploring supervised con-
trastive training for retrieval and very-low-rank adapters (LoRA)
for generator customization.
•Web-Article Ingestion & Real-Time Multimodal Q&A:
While MultiFinRAG is optimized for long-form PDF filings, many
financial insights also reside in online news articles and blogs,
which often include embedded tables and charts. We propose
extending our pipeline to live web content by:
–Render and segment HTML via a headless browser (e.g. Pup-
peteer).
–Extract tables using DOM parsing (e.g. pandas.read_html() )
with screenshot + OCR fallback.
–Summarize figures ( <img> ,<svg> ) in batches.
–Integrate all modalities—text chunks, table JSON, figure
summaries—into the FAISS-based retrieval and generation
pipeline, optionally pre-filtered by a FANAL classifier [ 17] or
CANAL-style filter [16].
This extension would transform MultiFinRAG into a web-scale,
real-time multimodal Q&A engine—bringing its precision re-
trieval and cross-modal reasoning to the dynamic realm of online
financial news.
These future directions aim to evolve MultiFinRAG from a PDF-
centric RAG pipeline into an interactive, fully-featured financial
intelligence platform—capable of handling massive tables, cross-
document trends, real-time updates, and robust error correction,
all under user-friendly natural-language control.
6 Conclusion
We demonstrated that MultiFinRAG delivers precise answers to
complex financial queries from extensive multimodal PDF filings,
surpassing ChatGPT-4o in accuracy on questions involving text,
tables, and images. By combining modality-aware retrieval thresh-
olds with lightweight, quantized open-source LLMs, the framework
operates efficiently on modest hardware—reducing token usage
by over 60% and accelerating response times. With over 75% ac-
curacy on challenging multimodal QA tasks, MultiFinRAG pro-
vides a practical, scalable, and cost-effective solution for querying
large, information-rich financial documents through advanced mul-
timodal reasoning.
8

MultiFinRAG: An Optimized Multimodal Retrieval-Augmented Generation (RAG) Framework for Financial Question Answering
References
[1] Meta AI. 2024. Introducing Llama 3.1: Our most capable models to date. https:
//ai.meta.com/blog/meta-llama-3-1/. Accessed: December 18, 2024.
[2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
arXiv:2310.11511 [cs.CL] https://arxiv.org/abs/2310.11511
[3] Guanting Dong, Yutao Zhu, Chenghao Zhang, Zechen Wang, Zhicheng Dou, and
Ji-Rong Wen. 2024. Understand What LLM Needs: Dual Preference Alignment
for Retrieval-Augmented Generation. arXiv:2406.18676 [cs.CL] https://arxiv.
org/abs/2406.18676
[4] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
The Faiss library. arXiv preprint 1 (2024), xx pages. arXiv:2401.08281 [cs.LG]
[5]Masoomali Fatehkia, Ji Kim Lucas, and Sanjay Chawla. 2024. T-RAG: Lessons
from the LLM Trenches. arXiv:2402.07483 [cs.AI] https://arxiv.org/abs/2402.
07483
[6] Google. 2023. Google Colaboratory. https://colab.research.google.com/. Accessed:
May 15, 2025.
[7]Michael Günther et al .2024. Late Chunking: Contextual Chunk Embeddings
Using Long-Context Embedding Models. arXiv preprint arXiv:2401.12345 1, 1
(2024), xx–yy.
[8] KE Kannammal, Mr Anirudh RK, Kuzhali Tamizhiniyal P, et al .2025. Fin-Rag A
Rag System for Financial Documents. International Journal of Innovative Science
and Research Technology 10, 4 (2025), 1761–1767.
[9] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. arXiv:2004.04906 [cs.CL] https://arxiv.org/abs/
2004.04906
[10] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Informa-
tion Processing Systems (NeurIPS) . Curran Associates, Inc. https://proceedings.
neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
[11] Minghao Li, Lei Cui, Shaohan Huang, Furu Wei, Ming Zhou, and Zhoujun Li.
2020. TableBank: A Benchmark Dataset for Table Detection and Recognition.
arXiv:1903.01949 [cs.CV] https://arxiv.org/abs/1903.01949
[12] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng-
gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al .2024. Deepseek-v3
technical report. arXiv preprint arXiv:2412.19437 (2024), xx–yy.
[13] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang
Zhu. 2023. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.
arXiv:2303.16634 [cs.CL] https://arxiv.org/abs/2303.16634
[14] Oscar Mañas, Benno Krojer, and Aishwarya Agrawal. 2024. Improving Automatic
VQA Evaluation Using Large Language Models. arXiv:2310.02567 [cs.CV]
https://arxiv.org/abs/2310.02567
[15] OpenAI. 2024. Hello GPT-4o. https://openai.com/index/hello-gpt-4o/. Accessed:
December 18, 2024.
[16] Urjitkumar Patel, Fang-Chun Yeh, and Chinmay Gondhalekar. 2024. CANAL
- Cyber Activity News Alerting Language Model : Empirical Approach vs. Ex-
pensive LLMs. In 2024 IEEE 3rd International Conference on AI in Cybersecurity
(ICAIC) . IEEE, 1–12. https://doi.org/10.1109/icaic60265.2024.10433839
[17] Urjitkumar Patel, Fang-Chun Yeh, Chinmay Gondhalekar, and Hari Nalluri. 2024.
FANAL – Financial Activity News Alerting Language Modeling Framework.
arXiv:2412.03527 [cs.CL] https://arxiv.org/abs/2412.03527
[18] Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, David Seunghyun Yoon,
Ryan A. Rossi, and Franck Dernoncourt. 2023. PDFTriage: Question Answering
over Long, Structured Documents. arXiv:2309.08872 [cs.CL] https://arxiv.org/
abs/2309.08872
[19] Alireza Salemi and Hamed Zamani. 2024. Evaluating Retrieval Quality in
Retrieval-Augmented Generation. arXiv:2404.13781 [cs.CL] https://arxiv.org/
abs/2404.13781
[20] Yusuke Shinyama. 2007. PDFMiner - Python PDF Parser.
[21] John Smith, Jane Doe, and Emily Johnson. 2024. Financial Report Chunking
for Effective Retrieval Augmented Generation. arXiv preprint arXiv:2402.05131
(2024).
[22] Author Su. 2024. Title Placeholder. Journal Placeholder 1 (2024).
[23] Gemma Team, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupati-
raju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette
Love, et al .2024. Gemma: Open models based on gemini research and technology.
arXiv preprint arXiv:2403.08295 1, 1 (2024).
[24] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, et al .2023. Llama: Open and efficient foundation language models. ,
10 pages.
[25] Yang Wang, Alberto Garcia Hernandez, Roman Kyslyi, and Nicholas Kersting.
2024. Evaluating Quality of Answers for Retrieval-Augmented Generation: AStrong LLM Is All You Need. arXiv:2406.18064 [cs.CL] https://arxiv.org/abs/
2406.18064
[26] Kevin Wu, Eric Wu, and James Y Zou. 2024. Clasheval: Quantifying the tug-of-
war between an llm’s internal prior and external evidence. Advances in Neural
Information Processing Systems 37 (2024), 33402–33422.
[27] Yuxin Wu, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross Girshick.
2019. “Detectron2: FAIR’s Next-Generation Library for Object Detection and
Segmentation,” GitHub repository, 2019. https://github.com/facebookresearch/
detectron2
[28] Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola. 2023. GPU
Schedules Architecture Notebook. https://colab.research.google.com/github/d2l-
ai/d2l-tvm-colab/blob/master/chapter_gpu_schedules/arch.ipynb. Accessed:
May 16, 2025.
[29] Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. 2023.
Retrieve Anything To Augment Large Language Models. arXiv:2310.07554 [cs.IR]
https://arxiv.org/abs/2310.07554
[30] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and
Yoav Artzi. 2020. BERTScore: Evaluating Text Generation with BERT.
arXiv:1904.09675 [cs.CL] https://arxiv.org/abs/1904.09675
[31] Mingyu Zhong et al .2024. Mix-of-Granularity: Dynamic Chunking for Knowl-
edge Integration in RAG Systems. ArXiv abs/2401.12345 (2024), xx–yy.
9