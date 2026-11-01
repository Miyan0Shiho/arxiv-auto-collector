# Tessera: Structure-Completed Evidence Units for Web Retrieval-Augmented Generation

**Authors**: Peichun Hua, He Sun, Zirui Zhuang, Qi Qi, Yiming Qiu, Shinan Liu, Yunming Xiao

**Published**: 2026-10-27 11:41:33

**PDF URL**: [https://arxiv.org/pdf/2610.37080v1](https://arxiv.org/pdf/2610.37080v1)

## Abstract
Web retrieval-augmented generation (RAG) splits a page into fixed-size text chunks or, in recent visual pipelines, screenshot tiles, and the retrieved item serves at once as the unit of indexing, ranking, and reading. We trace answer evidence from source HTML to reader input over the full English Wikipedia and find that this coupling loses evidence in text and screenshot pipelines alike: evidence present in the source page often fails to reach the reader intact, particularly structured content such as tables and infoboxes, which chunking separates from the headers and context that give it meaning. Stronger extraction, context expansion, and larger reader budgets leave the loss in place. We present Tessera, a training-free evidence delivery layer that decouples what a pipeline retrieves from what its reader reads. Given the pages behind any ranked list, Tessera rebuilds their HTML into fine-grained evidence units, each completed with its page title, section path, and table headers, selects the units relevant to the query, and packs them into the reader context at an unchanged token budget. The index, the retriever, and the reader stay as they are, so Tessera applies to text, screenshot, and agentic pipelines. Across three question answering benchmarks, Tessera improves answer accuracy by 5 to 6 points on text and screenshot pipelines, consistently across readers, retrievers, and reader budgets, and outperforms strong HTML-aware and structure-aware RAG baselines under matched budgets. The gains carry over to multi-step agentic search. Our code is publicly available at https://github.com/sarendis56/Tessera.

## Full Text


<!-- PDF content starts -->

Tessera: Structure-Completed Evidence Units for Web
Retrieval-Augmented Generation
Peichun Hua
The Chinese University of Hong
Kong, Shenzhen
Shenzhen, China
peichunhua@link.cuhk.edu.cnHe Sun
The University of Hong Kong
Hong Kong, China
hesun@hku.hkZirui Zhuang
Beijing University of Posts and
Telecommunications
Beijing, China
zhuangzirui@bupt.edu.cn
Qi Qi
Beijing University of Posts and
Telecommunications
Beijing, China
qiqi8266@bupt.edu.cnYiming Qiu
The University of Hong Kong
Hong Kong, China
yimingq@hku.hkShinan Liu
The University of Hong Kong
Hong Kong, China
shinan6@hku.hk
Yunming Xiao∗
The Chinese University of Hong
Kong, Shenzhen
Shenzhen, China
yunmingxiao@cuhk.edu.cn
Abstract
Web retrieval-augmented generation (RAG) splits a page into fixed-
size text chunks or, in recent visual pipelines, screenshot tiles, and
the retrieved item serves at once as the unit of indexing, rank-
ing, and reading. We trace answer evidence from source HTML
to reader input over the full English Wikipedia and find that this
coupling loses evidence in text and screenshot pipelines alike: ev-
idence present in the source page often fails to reach the reader
intact, particularly structured content such as tables and infoboxes,
which chunking separates from the headers and context that give it
meaning. Stronger extraction, context expansion, and larger reader
budgets leave the loss in place. We presentTessera, a training-free
evidence delivery layer that decouples what a pipeline retrieves
from what its reader reads. Given the pages behind any ranked
list,Tesserarebuilds their HTML into fine-grainedevidence units,
each completed with its page title, section path, and table headers,
selects the units relevant to the query, and packs them into the
reader context at an unchanged token budget. The index, the re-
triever, and the reader stay as they are, soTesseraapplies to text,
screenshot, and agentic pipelines. Across three question answering
benchmarks,Tesseraimproves answer accuracy by 5 to 6 points
on text and screenshot pipelines, consistently across readers, re-
trievers, and reader budgets, and outperforms strong HTML-aware
and structure-aware RAG baselines under matched budgets. The
gains carry over to multi-step agentic search. Our code is publicly
available at https://github.com/sarendis56/Tessera.
CCS Concepts
•Information systems →Retrieval models and ranking;Web
searching and information discovery;Question answering.
∗Corresponding author.Keywords
Retrieval-Augmented Generation, Content Extraction, Web Search,
HTML, Structured Evidence, Agentic Search
1 Introduction
Retrieval-augmented generation (RAG) grounds a language model
in an external corpus [ 15,18,23,28], and the Web is the corpus that
matters most in practice: search assistants, deep-research agents,
and browsing models all answer from retrieved webpages [ 17,20,
32,35,37,41,47,52]. A webpage is a structured document. Its
facts sit in infoboxes, tables, lists, and captions as often as in prose,
and each fact takes its meaning from the surrounding structure: a
number in a scorecard means nothing without its column header,
its row label, and the section that names the match. Carrying this
structure from page source to reader is a Web-specific challenge
for RAG and agentic search, which prior work largely overlooks by
retrieving from clean passage corpora that strip tables, infoboxes,
and lists [20, 21, 32].
The standard web RAG pipeline discards this structure early: an
extractor such as Trafilatura [ 5] flattens each page to text, a chunker
cuts the text into fixed windows, and the reader receives the top few
chunks after retrieval and reranking. HTML-centric methods keep
pruned markup in the reader’s context [ 44]. Vision-centric methods
skip parsing and index rendered pages as screenshot tiles [ 10,33,
56]; PixelRAG scales this design to the full English Wikipedia and
reports that screenshots beat text across six benchmarks [50].
We start from the text and screenshot pipelines released with
PixelRAG and ask where each loses its answers. Once both rerank
their candidates, the two modalities perform on par (§2.1), and
they share a ceiling that a four times larger reader budget lifts by
about two points (§2.2). Tracing quoted answer evidence from the
source HTML to the reader input (§2.2) shows that it disappears
the most at two places. For 9.7% of questions, a single stored chunk
arXiv:2610.37080v1  [cs.IR]  27 Oct 2026

Hua et al.
Question (NQ-Tables)
Who produced Willy Wonka and the Chocolate Factory? Reference:Stan
Margulies; David L. Wolper
(a) Source HTML, infobox of the page
<tr><th>Directed by</th><td>Mel Stuart</td></tr>
<tr><th>Produced by</th><td><ul>
<li>Stan Margulies</li>
<li>David L. Wolper</li>
</ul></td></tr>
(b) Top-ranked text chunk after extraction
| Directed by | Mel Stuart |
| Screenplay by | Roald Dahl[a] |
|Produced by |
|
| Starring | |
| Cinematography | Arthur Ibbetson |
Reader with top-3 chunks: “Mel Stuart”✗
(c) TESSERAunit block, same token budget
Willy Wonka & the Chocolate Factory › Theatrical release poster
Directed by: Mel Stuart
Screenplay by: Roald Dahl
Produced by: Stan Margulies; David L. Wolper
Distributed by: Paramount Pictures
Reader with top-2 chunks + units: “Stan Margulies and David L. Wolper”
✓
Figure 1: Evidence lost between a webpage and the reader:
the extractor drops a list-valued infobox cell, andTessera
delivers the completed row in place of the third chunk.
contains the complete evidence, yet it is missing from the top three
after ranking; placing that chunk in the third slot lifts accuracy on
these questions from 49.1% to 78.5%. For 7.6% of source-supported
questions, the evidence never enters the index intact, and81% of
these losses are table cells, table headers, or captions(Figure 1).
Both losses follow froma pipeline that fixes, at indexing time,
what its reader can see. Evidence that extraction drops is lost to
every later stage, and the chunk size is tuned for retrieval, where
1,024 tokens is a good choice: in our pilot study, 512-token chunks
enlarge the index by 73% and answer fewer questions (§4.4), and
the page that supports the answer is among the pages of the top
ten reranked chunks for 92.7% of source-supported questions. The
pipeline then hands the same chunks to the reader, anda chunk
sized for retrieval is a poor unit to read from. The answer
usually occupies one sentence or one table row, so only a few
chunks fit in the budget and the one that holds the evidence often
falls outside them, and a fixed window separates a table row from
the header and caption that label it.The pipeline uses a single
object, the chunk or the tile, as its unit of indexing, ranking,
and reading: it finds the right page and delivers the wrong part.
We presentTessera1, a training-free, query-time layer that de-
couples delivery from retrieval on top of any pipeline. It treats the
1Atesserais a single small tile of a mosaic.ranked list as a pointer toanchor pages, parses their HTML into
smallevidence unitssuch as a table row, a list item, or a paragraph,
and completes each unit with the structure that gives it meaning:
the page title, the section path, and the column headers resolved
through merged cells. A cross-encoder selects the units relevant
to the question, and a packer writes them into the token budget of
one evidence item, so the reader budget, the index, the retriever,
and the reader stay unchanged. Our contributions are as follows.
•A source-to-reader evidence trace of web RAG.We locate
where text and screenshot pipelines lose answer evidence, price
the loss with a controlled oracle intervention, and show that
stronger extraction, context expansion, structure-aware rerank-
ing, and larger budgets do not naturally recover it (§2).
•Tessera, a universal evidence delivery layer.We introduce
structure-completed evidence units, a faithful table-grid con-
struction, and budget-neutral packing; the layer plugs into text,
screenshot, and agentic pipelines without retraining (§3). We
show that this is a simple and effective add-on to all of traditional
RAG, multi-modal index, and agentic RAG (§4.2).
•Consistent gains across pipelines and settings.On 7,851
questions from Natural Questions [ 26], NQ-Tables [ 16], and Sim-
pleQA [ 51],Tesseraadds 5 to 6 points to text and screenshot
pipelines, and up to 5.2 points inside a multi-step search agent
on FRAMES [ 25] and MuSiQue [ 45]. Under a matched budget,
it outperforms all fourteen alternatives we test: four query-
expansion methods [ 1,11,27,49], two document-expansion
methods [ 13,36], late chunking [ 14], small-to-big retrieval [ 30],
RAPTOR [ 40], contextual retrieval [ 2], two variants of Html-
RAG [ 44], and two structure-aware chunkers [ 4,46]. The gain
persists across readers, retrievers, chunk sizes, and reader bud-
gets, and 3K tokens withTesserabeat 12K tokens of chunks or
tiles (§4).
2 Where Does Web Evidence Go Missing?
We first show that text and screenshot pipelines converge once both
rank their candidates well (§2.1). We then trace answer evidence
through the text pipeline and quantify what the reader never sees
(§2.2), test the obvious remedies, and derive the design principles
ofTessera(§2.3). All experiments use the Qwen3.5-4B reader and
August 2025 English Wikipedia snapshot, consistent with our main
experiment setup in §4.1.
2.1 The Modality Gap Is a Ranking Gap
PixelRAG compares a screenshot retriever against a text baseline
that sends the top three results of a single dense retriever to the
reader [ 50]. Table 1 reproduces this comparison on PixelRAG’s
released indexes and then gives each modality a reranker over its
top 50 candidates: BGE-reranker-v2-m3 [ 6] for text chunks and
Qwen3-VL-Reranker [29] for screenshot tiles.
Without reranking, screenshots lead text by 3.9 to 5.2 accuracy
points, in line with the PixelRAG report. Reranking adds 8.9 to 11.8
points to the text pipeline and 5.2 to 9.5 points to the screenshot
pipeline, and the two modalities meet: the difference is −0.4points
on NQ,+1.4on NQ-Tables, and +2.6on SimpleQA. A text pipeline
with our own index, which fuses dense and BM25 candidates [ 9,39]
before the same reranker, is the most accurate on every dataset.

Tessera: Structure-Completed Evidence Units for Web RAG
Table 1: Text and screenshot pipelines on the full Wikipedia:
answer accuracy / Recall@3 of the gold article (%).
Pipeline NQ NQ-Tables SimpleQA
Released PixelRAG indexes, first-stage retrieval only
Text, dense 53.2 / 47.1 37.1 / 37.7 68.3 / 80.1
Screenshot, tuned 57.1 / 58.1 42.2 / 51.0 73.3 / 85.3
Released PixelRAG indexes, reranked top 50
Text + BGE 62.7 / 66.5 46.0 / 59.2 80.1 / 87.5
Screenshot + Qwen3-VL 62.3 / 62.6 47.4 / 57.9 82.7 / 89.4
Our index
Hybrid text + BGE64.8/67.8 48.4/61.8 83.6/90.1
0 20 40 60 80 100SimpleQA (2,847)NQ-Tables (498)NQ (2,538)All (5,883)
8.0
17.231.321.620.3
8.818.924.716.5
70.541.850.759.5
Share of source-supported questions (%)Reaches the reader Stored, lost after indexing
Not verbatim in chunks Chunk not supporting
Figure 2: Where quoted answer evidence stops in the hybrid
text pipeline. “Chunk not supporting” means that the quotes
sit in a chunk that the grader does not judge supporting.
The comparison also exposes a shared ceiling. The best pipeline
of either modality retrieves the gold article in its top three for 61.8%
of NQ-Tables questions and answers 48.4% of them. Better ranking
has brought both modalities to the same place, and the remaining
errors call for a different explanation.
2.2 Tracing Evidence from Source to Reader
Protocol.We trace the hybrid text pipeline of Table 1 on 7,378
questions.2For each question, a grader model (GPT-5.6 Luna) reads
the Wikipedia page that the dataset cites and quotes the fragments
that support the reference answer. We then locate every quoted
fragment at each stage by string matching: in the extracted text, in
the stored 1,024-token chunks, in the top-100 retrieved candidates,
in the 50 candidates the reranker scores, and in the three chunks
the reader receives. The source page supports the reference answer
with valid quotes for 5,883 questions, and the trace assigns each of
them to the first stage at which its evidence disappears.
Stored evidence misses the reader.Figure 2 shows the outcome.
The quoted evidence reaches the reader for 59.5% of the source-
supported questions. For 16.5% the evidence sits verbatim in stored
chunks and disappears afterwards: the reader’s top three omit it
for 500 questions, the top-100 candidates omit it for 362, and the
reranker’s admission pool omits it for 110. The loss is largest on
NQ, where it affects one supported question in four.
2The trace covers the 7,851 evaluation questions minus 473 whose source pages we
used while developing the protocol.Table 2: Oracle intervention: the third reader slot is replaced
by a stored chunk that holds the complete evidence (71 tokens
longer on average). Accuracy in %.
Dataset 𝑛Top-3 TESSERA Oracle
NQ 478 51.9 61.975.3
NQ-Tables 62 48.4 54.871.0
SimpleQA 175 41.7 77.189.7
All 715 49.1 65.078.5
Oracle intervention.To measure what this evidence is worth,
we select the 715 questions (9.7% of traced questions) for which
one stored chunk contains every quoted fragment and the reader
input lacks the evidence. We keep the reader’s first two chunks
and replace the third with that chunk, so the reader still receives
three slots. Table 2 reports the result: accuracy rises from 49.1% to
78.5%, with 226 questions gained and 16 lost. Without any oracle
information,Tesseraalready closes more than half of this gap (§4.6).
As a control, another chunk from the same page moves accuracy
by only 2.1 points, so the answer-bearing passage carries the effect.
Extraction drops structured evidence.For a further 20.3% of
supported questions, the quoted fragments do not appear verbatim
in any stored chunk. Tolerant matching, which ignores case, punc-
tuation, whitespace, and word order, shows that most paragraph
fragments survive in reformatted form, while structured content
does not: extraction loses 16% of quoted table cells, 40% of figure
captions, and 44% of table captions, compared to 1% of paragraph
fragments. In total, 449 questions (7.6% of the supported ones) lose
at least one evidence fragment from every stored chunk, and for
362 of them (81%) the missing fragment is a table cell, a table header,
or a caption. No retriever or reranker can recover this evidence
because the index does not contain it.
Surviving structure loses its labels.Table evidence that does
survive often arrives stripped of meaning. Among quoted table cells
found verbatim in a chunk, 31% sit in a chunk that lacks the table’s
leading column headers. Later chunks of a page also lose its title.
Standard remedies fall short.Each loss suggests a familiar rem-
edy: recall-oriented re-extraction, neighboring chunks around the
hits, structure prefixed to the chunks, a four times larger reader
budget, or screenshots. Each adds at most 2.2 points with noticeable
increased budget or inference-time overhead, against the 29.4 points
that the oracle chunk shows to be available (details in Appendix C).
2.3 Design Principles
The measurements point to four principles.P1: Retrieve pages,
deliver units.The top ten reranked chunks reach the gold page
for 92.7% of supported questions but deliver the evidence for 59.5%,
so the delivery step should reopen the retrieved pages and choose
what to read at a finer grain than the index stores.P2: A unit must
carry its structure.A table row is useful only with its headers,
and a list item only with the heading above it (§4.6, Appendix D).
P3: Work from the page source at query time.Extraction loss
is invisible downstream of the index, whereas parsing the HTML
of a handful of pages per query sees the full page, costs little, and
leaves the index untouched, whether it holds text or screenshots.
P4: Stay within the reader budget.One well-chosen slot is worth

Hua et al.
Any Retrieval PipelineRanked list L: chunks, tiles, or search resultsTop k − 1 unchangedTESSERAQuery-time1Anchor PagesHTMLDistinct pagesof top m itemsFetch HTML2Build UnitsRows, lists, paragraphsFaithful table gridTitle + section pathHeaders + unit body3Select UnitsLexical prefilter to N unitsCross-encoder on (q, u)4Pack by ContextShared contextand headers once|U| ≤ |eₖ|Readere₁, …, eₖ₋₁+Unit block UReplaces eₖwithin the same budget B
Figure 3: Overview ofTessera: retrieved pages are rebuilt
into evidence units and packed into the last evidence slot.
29 points on the affected questions and many poorly chosen ones
are worth two, so the layer should reallocate the existing budget.
3Tessera
Tesserais a delivery layer between a retrieval pipeline and its
reader (Figure 3). It consumes the pipeline’s ranked list and produces
a reader context of the same size in which one evidence item is
replaced by a block of structure-completed evidence units. We first
state the problem (§3.1), then describe the four steps: anchoring
(§3.2), unit construction (§3.3), selection (§3.4), and packing (§3.5).
§3.6 explains how the layer attaches to text pipelines, screenshot
pipelines, and search agents.
3.1 Problem Setting
A retrieval pipeline answers a query 𝑞with a ranked list 𝐿=
(𝑒1,...,𝑒 𝑛)of evidence items. An item is a text chunk or a screen-
shot tile, and each item 𝑒𝑖comes from a webpage 𝜋(𝑒𝑖). The reader
receives the top 𝑘items, at a cost of 𝐵=Í
𝑖≤𝑘|𝑒𝑖|tokens under the
reader’s tokenizer or image processor. We call 𝐶=(𝑒 1,...,𝑒 𝑘)the
coupledcontext, because its elements were sized for indexing.
Tesserabuilds adecoupledcontext
𝐶′=(𝑒 1,...,𝑒 𝑘−1,𝑈(𝑞,𝐿)),|𝑈(𝑞,𝐿)|≤|𝑒 𝑘|,(1)
where𝑈(𝑞,𝐿) is a text block assembled from units of the pages be-
hind𝐿. The context keeps the pipeline’s best 𝑘−1items, which carry
continuous prose and the pipeline’s ranking signal, and spends the
last slot on units. The cost stays at most 𝐵. The layer is training-free
and reads only𝑞,𝐿, and the HTML of pages𝜋(𝑒 𝑖).
3.2 Anchor Pages
The ranked list is a noisy pointer to passages and a reliable pointer
to pages (§2.3).Tesseratakes the top 𝑚items of𝐿and collects
their distinct pages in rank order,
P(𝑞)=distinct 𝜋(𝑒 1),...,𝜋(𝑒 𝑚),(2)with𝑚= 10by default. Several top items often share a page, so
P(𝑞) holds at most 𝑚pages. The layer then fetches the HTML of
each anchor page from the page store, which is the crawl snapshot
that the index was built from. Units are query-independent, so a
deployment can cache them per page.
3.3 Structure-Completed Units
A unit is the smallest piece of a page that states a fact once its context
is attached. Each unit 𝑢has a context 𝑐(𝑢) (page title, section path,
table title, group label), optional labels ℎ(𝑢) (column headers), and
a body𝑏(𝑢). Itscompletedtext is 𝑐(𝑢) followed by 𝑏(𝑢), with each
row value paired with its label, and is self-contained: a reader, or
a reranker, can interpret it without the rest of the page.Tessera
walks the DOM of an anchor page in document order after removing
scripts, styles, navigation, and citation superscripts, tracking the
section headings above the current position. The walk emits two
kinds of units, about 150 per Wikipedia page.
Block units.Each paragraph, list item, definition term or value,
and figure caption becomes one unit. Its context 𝑐(𝑢) is the path
page title › section › subsectiongiven by the heading stack.
Row units.Each row of a table or an infobox becomes one unit.
The difficulty is to attach every cell to the right label. Web tables use
merged cells, multi-row headers, title rows, and group rows, and
the cells of a row in DOM order do not line up with the header cells
above them. On the anchor pages of our experiments, pairing cells
with the first header row in DOM order assigns a wrong header to
32.5% of the 2.9 million table rows, or drops one.Tesseratherefore
rebuilds each table as afaithful gridbefore reading rows from it
(Figure 4):
(1)Span expansion.Every cell is copied into all grid positions that
itsrowspan andcolspan cover, so each grid position holds the
cell that a reader of the rendered table would see there.
(2)Title rows.The caption and the leading rows that span the full
width become the table title and join𝑐(𝑢).
(3)Header rows.The following rows that consist of header cells
formℎ(𝑢). For multi-row headers, the labels of a column are
joined top-down, for exampleTerm of office › Took office.
(4)Group rows.A full-width row inside the body labels the rows
below it and joins their𝑐(𝑢)until the next group row.
(5)Key–value rows.In a table without column headers whose rows
start with a header cell (an infobox), the first cell is the key.
(6)Nested tablesinherit the title and headers of the table that wraps
them. Line breaks and list items inside a cell are kept as separa-
tors, preserving multi-valued cells (Figure 1).
3.4 Unit Selection
The anchor pages of a question hold about a thousand units, and
one slot holds a few dozen. Selection runs in two stages. A lexical
prefilter keeps the 𝑁= 100units whose completed text shares the
most content words with the question, which costs no model call. A
cross-encoder then scores each remaining pair (𝑞,completed(𝑢)) ;
we reuse the pipeline’s own text reranker, BGE-reranker-v2-m3, so
the layer adds no model to the deployment. Scoring the completed
text matters: the context states which match, season, or entity a

Tessera: Structure-Completed Evidence Units for Web RAG
(a) Rendered table with merged cells (illustrative)
Governors
NameTerm of officePartyTook office Left office
A. Reyes 1998 2004LiberalB. Okafor 2004 2010
(b) Cells paired with headers in DOM order
State › History: Name: B. Okafor |Term of office: 2004|Party:
2010
(c) Row unit from the faithful grid
State › History › Governors — Name: B. Okafor |Term of office ›
Took office: 2004|Term of office › Left office: 2010|Party:
Liberal
Figure 4: Faithful grid: span expansion, multi-row headers,
and title rows give every cell its correct label.
row belongs to, and the bare row does not reveal it (compare the
two scorecards in Figure 5).
3.5 Organized Budget-Neutral Packing
The packer fills the budget |𝑒𝑘|of the displaced item. Completion
serves the scorer, which judges each unit in isolation; the reader
sees the selected units together and needs each context only once,
soTesseragroups units by context instead of repeating the same
path and headers on every row. It visits units in descending score
order and admits a unit if its cost fits the remaining budget, where
the cost of a unit is its body plus, for the first unit of a group, the
group’s context line and header line. After selection it renders each
group once: the context path, then the column headers, then the
member rows as value lines in page order. Groups appear in the
order of their best unit. The result reads like an excerpt of the
original table (Figure 5), and it fits about 14 lines into a slot that
holds about 11 units when each one repeats its context.
The packer measures cost with the reader’s tokenizer, so the unit
block never exceeds the displaced item. In our runs the mean reader
prompt withTesserais within 12 tokens of the baseline prompt.
3.6 Plugging into Pipelines
Text pipelines.The anchor pages are the pages of the top reranked
chunks, and the unit block replaces the last chunk. Because units
come from the page HTML, they include content that the pipeline’s
extractor dropped before indexing.
Screenshot pipelines.A screenshot index stores each tile with
its URL and coordinates.Tesseraanchors on the pages of the top
tiles and replaces the last tile with the unit block, packed to the
reader tokens that the image processor charges for that tile. The
block works as text or as a rendered image (Appendix I).
Search agents.An agent issues several queries and reads a tool
result for each.Tesseraapplies to every search call independently
with the agent’s query in the role of 𝑞: the tool returns the top 𝑘−1items and the unit block as the last document. The agent’s prompt,
tool schema, and stopping rule stay unchanged.
Cost.Per query, the layer parses the anchor pages, scores 𝑁short
pairs, and tokenizes the selection. Building the units of a page takes
15 ms on one CPU core, and scoring takes 0.2 GPU-seconds per
question on an A100. The layer adds no index, no offline pass over
the corpus, and no reader token budget.
4 Experiments
We organize the evaluation around five questions. DoesTessera
help pipelines that differ in index, modality, retriever, and reader
(§4.2)? Does it beat existing ways of improving retrieved context
under a matched budget (§4.3)? Does the gain survive a larger reader
budget (§4.4)? Does it carry over to agents that search several times
(§4.5)? Which components produce it (§4.6)?
4.1 Setup
Datasets.All pipelines index the August 2025 English Wikipedia
snapshot [ 24] (7,060,072 articles) corpus. For evaluation, we use the
3,336 validation questions of Natural Questions (NQ) [ 26] similar to
PixelRAG [ 50], 1,068 development questions of NQ-Tables [ 16], and
the 3,447 SimpleQA questions [ 51] whose cited source is an Eng-
lish Wikipedia article, totaling 7,851 questions. NQ-Tables targets
tabular evidence, and SimpleQA targets long-tail facts. The agentic
evaluation adds two multi-hop benchmarks: all 824 questions of
FRAMES [ 25], which need 2 to 15 pages each, and 1,000 answerable
MuSiQue questions [45].
Pipelines.We evaluate three pipelines built by two groups.Hybrid
text + BGEis our index: we employ Trafilatura [ 5] for content extrac-
tion. Passages are cut into 15.5 million chunks of up to 1,024 tokens,
with Qwen3-Embedding-0.6B dense retrieval [ 22,57], BM25 [ 39,54]
fused by reciprocal rank fusion [ 9], and BGE-reranker-v2-m3 [ 6]
reranking over the top 50.PixelRAG text + BGEretrieves from the
text index released by PixelRAG authors, 15.7 million chunks em-
bedded with Qwen3-VL-Embedding-2B, and reranks using the same
reranker.PixelRAG screenshot + Qwen3-VLretrieves from the re-
leased fine-tuned screenshot index (26.3 million tiles) and reranks
with Qwen3-VL-Reranker-2B [ 29]. Each pipeline passes its top three
items to the reader.
Reader and grading.The default reader is Qwen3.5-4B [ 38] with
greedy decoding and thinking disabled; we also run Gemma 4 E4B
and Gemma 4 31B [ 12]. GPT-5.6 Luna grades each answer against
the reference with the SimpleQA grading template [ 51], and accu-
racy is the share of answers graded as correct.
Tesseraconfiguration.One configuration serves every experi-
ment: anchor pages from the top 𝑚= 10items, a lexical prefilter of
𝑁= 100units, BGE-reranker-v2-m3 as the scorer, table-grouped
packing, and one unit block in the budget of the third evidence item.
We fixed the anchor and prefilter sizes before the first evaluation
and never tuned the configuration per pipeline, reader, or agent;
Appendix D shows that accuracy is insensitive to these choices.
Statistics.Every comparison is paired on the same questions. We
report exact McNemar tests [ 34] and call a difference significant at
𝑝<0.05; confidence intervals in the appendix come from a boot-
strap that resamples source pages. Pooled results sum the paired

Hua et al.
Coupled context: top-3 chunks of the baseline (excerpts)
Q:What was the strike rate of Harbhajan Singh in the final match of IPL 2015?
Reference:200.00
[Chunk 1, same page] ... | Ravindra Jadeja | not out | 11 | 8 | 1 | 0 |
137.50 | |
| Mohit Sharma | not out | 21 | 7 | 1 | 2 | 300.00 | |
| Ashish Nehra | |||||||
| Extras | (lb 2, w 3, nb 2) | 7 | ||||| ...
[Chennai innings rows without their header row; the Mumbai batting rows
sit in a stored chunk outside the top three]
[Chunk 3, page “Harbhajan Singh”] ... The match referee Farokh Engineer
found Harbhajan guilty of a level 4.2 offence, banning him from the
remainder of the IPL ... Harbhajan had taken five wickets at 16.40 at
an economy rate of 8.20 and scored 30 runs at 15.00 in the three matches
before his ban ...
Reader:“Harbhajan Singh did not bat in the final match of IPL 2015 ... the strike
rate is not provided.”✗Decoupled context: TESSERAunit block in the third slot
2015 Indian Premier League final › Scorecard › Mumbai Indians
innings
Player | Status | Runs | Balls | 4s | 6s | Strike rate
Lendl Simmons | b Smith | 68 | 45 | 8 | 3 | 151.11
Rohit Sharma* | c Jadeja b Bravo | 50 | 26 | 6 | 2 | 192.30
Harbhajan Singh | not out | 6 | 3 | 0 | 1 | 200.00
2015 Indian Premier League final › Scorecard › Mumbai Indians
bowling
Bowler | Overs | Maidens | Runs | Wickets | Econ | Wides | NBs
Harbhajan Singh | 4 | 0 | 34 | 2 | 8.50 | 1 | 0
2013 Indian Premier League final › Scorecard › Mumbai Indians
innings
Player | Status | Runs | Balls | 4s | 6s | Strike rate
Harbhajan Singh | c Hussey b Bravo | 14 | 8 | 3 | 0 | 175.00
Reader:“Harbhajan Singh’s strike rate in the 2015 IPL final was 200.00 ...
listed with 6 runs off 3 balls.”✓
Figure 5: A packed unit block from SimpleQA (abridged). Rows are scored with their full context, but each group prints its
context path and column headers once (green), then its selected rows.
counts of the three datasets. The study comprises more than one
million graded answers, and each comparison reruns its reference
condition in the same software environment.
4.2 One Layer, Many Pipelines
Table 3 applies the same layer to twelve pipeline configurations.
Index and modality.Tesseraraises pooled accuracy by 5.0 points
on our hybrid text pipeline, 6.2 points on the released PixelRAG text
pipeline, and 5.6 points on the PixelRAG screenshot pipeline, under
the same context budget. The gain is significant on every dataset and
pipeline and stays between 4.6 and 5.8 points under four robustness
checks (Appendix H): a reader output limit of 16K instead of 200
tokens, a grader that also accepts answers updated since the dataset
was built, only questions whose source page supports the reference
answer, and only questions that our earlier pipeline experiments
did not use. It is largest on SimpleQA (6.9 to 8.2 points), where
long-tail facts often sit in infoboxes and tables, and it reaches 4.4 to
6.3 points on NQ-Tables. The pipelines start 3.0 points apart and
end 1.9 points apart:Tesseraadds the most where the pipeline’s
own chunks miss the most. Tiles already show tables as the page
renders them, but replacing the third tile with a unit block still adds
5.6 points, so the bottleneck thatTesseraremoves is the unit of
delivery in both modalities. We show in Appendix I that even when
rendered as an image tile, our unit block still has noticeable gains.
Retriever.Six alternative first stages change the baseline by −1.0
to+0.9points, andTesseraadds 4.9 to 5.5 points to each of them.
The gain does not shrink on the stronger retrievers: it is 5.1 points
on top of Query2doc, the best of them, against 5.0 on the default.
Retrieval quality and delivery quality improve the pipeline indepen-
dently, and the combination Query2doc +Tesserareaches 76.7%,
the highest single-shot accuracy in our study.
Reader.The gain transfers to a second reader family. Gemma 4
E4B, matched in size to the default reader, gains 4.9 points. Gemma
4 31B is eight times larger and improves the baseline by 1.3 points;
Tesseraadds 4.7 points on top of it for text and 5.9 for screenshots.4.3 Comparison with Alternatives
Table 4 comparesTesserawith fourteen methods on the hybrid
text pipeline, with the same questions, reader, and grader, and with
every reader context matched to the token count of the top-3 chunks
except where the table notes otherwise.
Retrieval-side methods.RM3 pseudo-relevance feedback [ 1], the
learned sparse retriever SPLADE-v3 [ 27], the LLM-based query
expansions HyDE [ 11] and Query2doc [ 49], late chunking with jina-
embeddings-v3 [ 14,42], and two expansions that append generated
queries to every chunk before BM25 indexing, Doc2Query [ 36] and
Doc2Query-- [ 13], each replace or expand one side of first-stage
retrieval. None moves pooled accuracy by more than about one
point, andTesserabeats all seven by 4.1 to 6.1 points: they reorder
chunks, and the reader still reads 1,024-token windows.
Context-enrichment methods.Small-to-big retrieval [ 30] ranks
512-token chunks and reads the 1,024-token windows around them,
and it ties the baseline. RAPTOR [ 40] builds a tree of LLM sum-
maries over each anchor page and fills the third slot with tree nodes
selected by our cross-encoder; it gains 1.2 points and trailsTessera
by 3.8. Contextual retrieval [ 2] prepends an LLM-written context
that situates each chunk in its page; applied to the retrieved can-
didates before reranking and reading, it gains 0.8 points, all on
SimpleQA, and trailsTesseraby 4.3. HtmlRAG [ 44] fills the third
slot with the HTML of the anchor pages, cleaned and pruned by
its released two-stage pruner. It ties the top-3 chunks, and with
our cross-encoder as its first-stage block ranker (HtmlRAG-CE) it
scores 0.7 points below them.Tesseraleads both variants by at
least 5.4 points, significantly on every dataset, and by 5.8 points at
HtmlRAG’s own design point of a 4K-token context (Appendix F).
HtmlRAG was designed for a handful of search results, and pruning
ten full Wikipedia pages into one context leaves fragments without
the headers and section paths thatTesseraattaches.
Structure-aware chunkers.We run the rule-based document
parsers Unstructured [ 46] and Docling [ 4] as drop-in unit builders
inside our delivery layer: the same anchor pages, prefilter, cross-
encoder, and slot, with their chunks in place of our units. Both
improve on the top-3 chunks, so the delivery layer is useful beyond
one parser, and their ranking follows principle P2: Unstructured

Tessera: Structure-Completed Evidence Units for Web RAG
Table 3: Accuracy (%) with the pipeline’s top-3 evidence→withTessera(top-2 + unit block, same budget), and the gain.
Pipeline Reader NQ(3,336) NQ-Tables(1,068) SimpleQA(3,447) Pooled
Retrieval pipeline and modality
Hybrid text + BGE Qwen3.5-4B 64.6→67.8+3.247.9→52.3+4.483.6→90.6+7.070.7→75.7+5.0
PixelRAG text + BGE Qwen3.5-4B 62.2→66.3+4.145.9→52.2+6.379.8→88.0+8.267.7→73.9+6.2
PixelRAG screenshot + VL Qwen3.5-4B 63.1→67.2+4.145.8→51.8+6.083.3→90.1+6.969.6→75.2+5.6
First-stage retriever of the hybrid text pipeline
BM25 + RM3 feedback Qwen3.5-4B 64.4→67.6+3.247.3→51.6+4.382.4→89.8+7.570.0→75.2+5.2
SPLADE-v3 for BM25 Qwen3.5-4B 65.0→68.6+3.549.0→52.6+3.780.6→88.5+7.969.7→75.2+5.5
HyDE Qwen3.5-4B 65.6→68.7+3.148.5→53.7+5.283.7→90.9+7.271.2→76.4+5.2
Query2doc Qwen3.5-4B 66.4→69.4+3.048.3→53.1+4.883.9→91.0+7.171.6→76.7+5.1
Doc2Query for BM25 Qwen3.5-4B 64.8→68.1+3.348.6→52.3+3.783.7→90.5+6.870.9→75.8+4.9
Doc2Query-- for BM25 Qwen3.5-4B 64.8→68.7+3.847.6→52.3+4.883.5→90.9+7.470.7→76.2+5.5
Reader model
Hybrid text + BGE Gemma 4 E4B 64.5→67.2+2.746.5→52.3+5.884.3→91.0+6.770.7→75.6+4.9
Hybrid text + BGE Gemma 4 31B 65.2→68.4+3.248.7→54.3+5.685.8→91.6+5.972.0→76.7+4.7
PixelRAG screenshot + VL Gemma 4 31B 65.8→70.2+4.449.3→55.4+6.183.8→91.0+7.271.5→77.4+5.9
Table 4: Matched-budget comparison on the hybrid text
pipeline (7,851 questions, 3K evidence tokens). Δ: pooled
change vs. the top-3 chunks; gray: not significant.‡Applied
to the retrieved candidates; about 170 more reader tokens.
Method NQ NQ-Tab. SimpleQA Pooled Δ
Top-3 chunks 64.63 47.94 83.64 70.70
Retrieval-side methods
BM25 + RM3 [1] 64.39 47.28 82.36 69.95−0.75
SPLADE-v3 [27] 65.02 48.97 80.62 69.69−1.02
HyDE [11] 65.62 48.50 83.72 71.24+0.53
Query2doc [49] 66.37 48.31 83.87 71.60+0.89
Late chunking [14] 64.96 47.85 83.46 70.76+0.05
Doc2Query [36] 64.84 48.60 83.70 70.91+0.20
Doc2Query-- [13] 64.84 47.57 83.49 70.68−0.03
Context-enrichment methods
Small-to-big [30] 64.33 47.38 83.72 70.54−0.17
RAPTOR [40] 65.92 48.41 85.00 71.91+1.21
Contextual retr.‡[2] 64.63 47.00 85.70 71.48+0.78
HtmlRAG [44] 65.08 47.57 82.39 70.30−0.41
HtmlRAG-CE 64.69 46.91 82.33 70.02−0.69
Structure-aware chunkers as our unit builder
Unstructured [46] 66.01 50.66 86.36 72.86+2.16
Docling [4]67.8451.03 89.30 74.97+4.27
TESSERA 67.84 52.34 90.63 75.74 +5.03
Query2doc + TESSERA 69.39 53.09 90.98 76.65 +5.95
flattens tables into cell text without headers and gains 2.2 points,
while Docling serializes each table row with its headers under the
heading path and gains 4.3 points.Tesserais the most accurate
builder, 0.8 points above Docling pooled ( 𝑝=0.013), and it parses a
page 26 times faster than Unstructured and over 500 times faster
than Docling (Appendix J). The gap between the builders widens
inside a search agent (§4.5).
4.4 Accuracy Under a Growing Reader Budget
A natural objection to any context-selection method is that modern
readers accept long inputs, so one could pass more chunks instead.Figure 6 varies evidence budgets from 2K to 12K tokens. For text,
we compare chunks only against three allocations: a 1K-token unit
block with chunks in the rest, half units and half chunks, and units
only. For screenshots, we compare the top 𝑘tiles against the top
𝑘−1tiles plus a unit block in the cost of one tile, for 𝑘from 2 to 14.
Units beat more context.Tesserabeats the chunks-only context
in all 51 pooled comparisons across three chunkings, six budgets,
and three allocations by 1.8 to 6.7 points. The two mixed allocations
are significant in every dataset in all 42 of their comparisons. On
screenshots, the unit block wins at every budget on every dataset.
Extra context helps the baselines slowly: quadrupling the budget
from 3K to 12K adds 2.1 points for 1,024-token chunks, 0.9 points
for 512-token chunks, and 2.2 points for tiles.
The smallestTesseracontext beats the largest baseline.With
1,024-token chunks, 3K tokens withTesserareach 76.1%, and 12K
tokens of chunks reach 73.2%. On the screenshot pipeline, one tile
plus a unit block at 2K tokens beats fourteen tiles at 12K by 3.4
points (𝑝< 10−16) with one sixth of the reader tokens.Tessera
therefore reduces reader cost and raises accuracy at once.
Chunk size stops mattering.Without units, 1,024-token chunks
beat 512-token chunks by 1.4 to 2.7 points from 3K upward. With
a unit block, the two chunk sizes stay within 0.6 points of each
other through 8K. Units provide the fine-grained evidence, and
the chunks only need to provide prose, so the pipeline no longer
depends on a carefully tuned chunk size.
Units complement, not replace.The mixed allocations beat units
only at every budget by 0.6 to 1.9 points, and at the three-tile budget,
one tile plus units is the best screenshot configuration (75.9%). The
pipeline’s top items carry running prose that rows and list items
cannot replace, which is whyTesserakeeps them.
4.5 Agentic Search
Agentic RAG lets the model write its own queries, read the results,
and search again [ 3,20,55]. Each search is a small RAG call, so
the delivery layer applies to each call. We evaluate an agent that
searches several times per question on multi-hop benchmarks; the
arms differ only in the last document of every tool result.

Hua et al.
2K 3K 4K 6K 8K 12K687072747678
3K with TESSERA
>12K of chunks
Reader evidence budget (tokens)Pooled accuracy (%)(a) Text, 1,024-token chunks
2K 3K 4K 6K 8K 12K687072747678
Reader evidence budget (tokens)(b) Text, 512-token chunks
2K 3K 4K 6K 8K 12K687072747678
1 tile + units
>14 tiles
Reader evidence budget (tokens)(c) Screenshot tiles (2 to 14 tiles)
Chunks or tiles only TESSERA: 1 unit block (1K tokens) TESSERA: half the budget Units only
Figure 6: Accuracy against reader budget on 7,851 questions. In (c), one tile costs 864 reader tokens, as does the unit block.
Table 5:Tesserainside a multi-step search agent (Qwen3.5-
4B with thinking, up to 8 searches).Srch.: mean searches per
question;Ctx.: mean final prompt length (K tokens).
Search result FRAMES MuSiQue Pooled Srch. Ctx.
Top-3 chunks 59.22 32.40 44.52 2.41 12.8
Top-4 chunks 59.22 32.30 44.46 2.18 14.5
Top-2 + Docling 57.65 32.10 43.64 2.65 13.6
Top-2 + TESSERA 64.44 33.50 47.48 2.19 11.7
gain vs. top-3 +5.22 +1.10 +2.96
Multi-hop reasoning agent.Qwen3.5-4B with thinking enabled
calls asearch tool up to eight times over the hybrid text pipeline
and averages 2.1 to 2.8 searches per question across arms on FRAMES
and MuSiQue. Table 5 shows thatTesseraraises FRAMES accu-
racy from 59.2% to 64.4%. The gain concentrates where structured
evidence decides the answer: +9.3points on tabular-reasoning ques-
tions,+7.8on numerical reasoning, and +13.1on post-processing
questions (details in Appendix G). MuSiQue composes its questions
from single paragraphs, mostly lead-section prose that the top-2
chunks already carry, and it gains 1.1 points.
We provide two more controls to isolate the cause.A fourth
chunk per search, about a third more tokens, does not bring benefits
(−0.05points).Docling chunksin the same slot lower accuracy by
0.9 points and trailTesseraby 3.8 points pooled and by 6.8 points
on FRAMES. The agent also behaves differently: withTesserait
searches less often than with chunks (2.19 against 2.41 searches) and
finishes with the shortest context (11.7K tokens vs. 12.8K), whereas
with Docling it issues a third search on 29% of FRAMES questions
against 16%. Compact, self-describing evidence lets the agent stop
earlier while achieving better performance.
4.6 Analysis
Ablations.We conduct ablations to show that completion brings
the gain: bare units, the same rows and paragraphs without titles,
sections, and headers add only 1.8 points, while completed units
add 4.6. We also test different row constructions and serializations,
which stay within 0.34 points of each other: for example, joint
ranking of chunks and units loses 0.6 points, units only lose 1.6, and
a second unit slot, other anchor counts, or prefilter sizes change
accuracy by at most 0.3 points (details in Appendix D).
Tesseragains where the evidence went missing.Table 6
groups the 5,883 source-supported questions of §2.2 by where theTable 6: Accuracy (%) on the source-supported questions,
grouped by where the top-3 pipeline loses the quoted ev-
idence. Net: questions gained minus questions lost.
Evidence in the top-3 pipeline 𝑛Top-3 TESSERA Net
Page not anchored 421 39.9 43.7 +16
Lost in extraction or chunking 390 60.882.3+84
Stored, not delivered 832 54.070.9+141
Delivered 4,240 90.7 91.9 +51
All 5,883 79.984.8+292
top-3 pipeline loses the quoted evidence, matched tolerantly against
the exact reader inputs: the source page is not among the ten an-
chor pages, extraction or chunking breaks the evidence, or a stored
chunk holds it but misses the reader.Tesseraraises evidence de-
livery from 72.1% to 81.9% of these questions. The two groups that
the delivery layer targets hold 21% of the questions and 225 of the
292 net gains (77%). Questions whose evidence sits only in tables or
infoboxes gain 13.8 points, and prose-only questions gain 2.4. On
the oracle questions (Table 2),Tesseraraises accuracy from 49.1%
to 65.0%, closing 54% of the oracle’s gain overall and 74% where the
evidence was stored but not delivered.
Remaining failures.OfTessera’s 893 errors on source-supported
questions, 425 occur with the quoted evidence already in the reader
input, which would improve with a stronger reader; an audit of 150
failures attributes most NQ and NQ-Tables failures to outdated ref-
erences and grading errors (Appendix H). On the oracle questions,
the oracle stays 13.4 points ahead (Appendix B). Of the 96 answers
in this gap, 61 lie on pages outside the anchor set, where a stronger
retriever adds toTessera(§4.2), and 35 lie within anchor pages,
split between the lexical prefilter (20) and selection and packing
(15). A larger prefilter changes accuracy by at most 0.3 points, so
recovering these units calls for semantic matching in the prefilter.
5 Related Work
We review prior work along the pipeline, from indexing to reading.
Web content and tables in RAG.Web RAG systems commonly
reduce a page to main-content text before indexing [ 5]. Learned
extractors [ 31,48] preserve more structure at a much higher corpus-
processing cost, and HtmlRAG [ 44] keeps pruned HTML in the
reader’s context. Table QA serializes a given table for a language
model [ 43] or retrieves tables from a table corpus [ 16], treating each
table as standalone.Tesserashares the premise that page structure

Tessera: Structure-Completed Evidence Units for Web RAG
carries information but converts it into explicit context on small
units, with no markup tokens or trained models, and selects table
rows, merged cells resolved from the HTML, under a budget shared
with the surrounding prose.
Visual document retrieval.DSE [ 33], ColPali [ 10], VisRAG [ 56],
and M3DocRAG [ 8] retrieve rendered pages with vision-language
encoders, and PixelRAG [ 50] scales the approach to a full Wikipedia
snapshot, motivated by the rich layout information present in web-
pages. A screenshot tile is a fixed window like a chunk, soTessera
complements this line: it uses the top-ranked tiles only to find
which pages to open, and builds its units from their HTML (§4.2).
Retrieval granularity and chunk context.Proposition-level
indexing [ 7], small-to-big retrieval [ 30], RAPTOR [ 40], late chunk-
ing [14], contextual retrieval [ 2], and structure-aware parsers [ 4,46]
all adjust what a chunk is when the corpus is indexed. They pay their
cost for every page, and most operate on extracted text, which has al-
ready lost the structured evidence.Tesseraacts at query time on the
HTML of the pages a query retrieves and is independent of the index
granularity. Query and document expansion [ 1,11,13,27,36,49]
and cross-encoder reranking [ 6] address earlier stages, and §4.2
shows that their gains add to ours.
Context compression.Post-retrieval refiners such as RECOMP [ 53]
and LongLLMLingua [ 19] shorten retrieved passages to reduce costs
and distractions.Tesserainstead selects complete structured units
from the pages behind them and is orthogonal to such refiners.
Agentic RAG.ReAct-style agents [ 55], Self-RAG [ 3], and RL-trained
agents such as Search-R1 [ 20] optimize when and what to retrieve
but take the tool’s output format as given; we show that this format
shapes the policy: with self-describing evidence, an agent searches
less and answers more questions correctly (§4.5).
6 Conclusion
Web RAG pipelines index, rank, and read the same object, and our
source-to-reader trace shows the cost of that coupling: answer ev-
idence that the corpus contains fails to reach the reader, in text
pipelines and screenshot pipelines alike.Tesseradecouples deliv-
ery from retrieval. It reopens the pages a pipeline retrieves, rebuilds
them into structure-completed evidence units, and spends one evi-
dence slot on the units the question needs. The layer is training-free
and index-free. It adds 5 to 6 accuracy points to three pipelines built
on two modalities by two groups, holds its gain across readers, re-
trievers, chunk sizes, and budgets, outperforms fourteen alternative
methods at matched cost, and makes a multi-step search agent up to
5.2 points more accurate while issuing fewer searches and reading
a shorter context.
Scope and future work.Our experiments use English Wikipedia,
whose pages are rich in tables and infoboxes; the open Web adds
layout tables, client-side rendering, and noisier markup. Extending
the unit builder to them requires a specific larger corpus and is our
next step for future work. The layer requires only access to page
HTML at query time, without additional training or models. Two
further directions follow from our analysis: wider anchoring and a
semantic prefilter, since most of the remaining oracle gap lies on
unanchored pages and in units that the lexical prefilter misses; and
search agents trained to issue queries that target unit blocks.References
[1] Nasreen Abdul-Jaleel, James Allan, W. Bruce Croft, Fernando Diaz, Leah Larkey,
Xiaoyan Li, Mark D. Smucker, and Courtney Wade. 2004. UMass at TREC 2004:
Novelty and HARD. InProc. TREC.
[2] Anthropic. 2024. Introducing Contextual Retrieval. Anthropic Engineering Blog.
https://www.anthropic.com/engineering/contextual-retrieval
[3] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
InProc. ICLR.
[4] Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele Dolfi, Nikolaos Livathi-
nos, Panos Vagenas, Cesar Berrospi Ramis, Matteo Omenetti, Fabian Lindlbauer,
Kasper Dinkla, et al. 2024. Docling Technical Report. arXiv:2408.09869 [cs.CL]
[5] Adrien Barbaresi. 2021. Trafilatura: A Web Scraping Library and Command-Line
Tool for Text Discovery and Extraction. InProc. ACL-IJCNLP: System Demonstra-
tions. 122–131. doi:10.18653/v1/2021.acl-demo.15
[6] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.
BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text
Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]
[7] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao,
Hongming Zhang, and Dong Yu. 2024. Dense X Retrieval: What Retrieval Gran-
ularity Should We Use?. InProc. EMNLP.
[8]Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024.
M3DocRAG: Multi-modal Retrieval is What You Need for Multi-page Multi-
document Understanding. arXiv:2411.04952 [cs.CV]
[9] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009. Reciprocal
Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In
Proc. SIGIR. 758–759. doi:10.1145/1571941.1572114
[10] Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline
Hudelot, and Pierre Colombo. 2025. ColPali: Efficient Document Retrieval with
Vision Language Models. InProc. ICLR.
[11] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot
Dense Retrieval without Relevance Labels. InProc. ACL. 1762–1777.
[12] Google DeepMind. 2026. Gemma 4 Model Card.
[13] Mitko Gospodinov, Sean MacAvaney, and Craig Macdonald. 2023. Doc2Query--:
When Less is More. InProc. ECIR.
[14] Michael Günther, Isabelle Mohr, Daniel James Williams, Bo Wang, and Han
Xiao. 2024. Late Chunking: Contextual Chunk Embeddings Using Long-Context
Embedding Models. arXiv:2409.04701 [cs.CL]
[15] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang.
2020. REALM: Retrieval-Augmented Language Model Pre-Training. InProc.
ICML. 3929–3938.
[16] Jonathan Herzig, Thomas Müller, Syrine Krichene, and Julian Eisenschlos. 2021.
Open Domain Question Answering over Tables via Dense Retrieval. InProc.
NAACL. 512–519. doi:10.18653/v1/2021.naacl-main.43
[17] Peichun Hua and Yunming Xiao. 2026. Semantics Delivery Network: Rethinking
Web Retrieval Infrastructure for LLM Agents. arXiv:2609.22486 [cs.NI]
[18] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. InProc. EACL. 874–
880. doi:10.18653/v1/2021.eacl-main.74
[19] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing
Yang, and Lili Qiu. 2024. LongLLMLingua: Accelerating and Enhancing LLMs in
Long Context Scenarios via Prompt Compression. InProc. ACL.
[20] Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang,
Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to Reason and
Leverage Search Engines with Reinforcement Learning. arXiv:2503.09516 [cs.CL]
[21] Jiajie Jin, Yutao Zhu, Zhicheng Dou, Guanting Dong, Xinyu Yang, Chenghao
Zhang, Tong Zhao, Zhao Yang, and Ji-Rong Wen. 2025. FlashRAG: A Modular
Toolkit for Efficient Retrieval-Augmented Generation Research. InCompanion
Proc. WWW. 737–740. doi:10.1145/3701716.3715313
[22] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-Scale Similarity
Search with GPUs.IEEE Trans. Big Data7, 3 (2021), 535–547. doi:10.1109/
TBDATA.2019.2921572
[23] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProc. EMNLP. 6769–6781. doi:10.18653/v1/2020.
emnlp-main.550
[24] Kiwix. 2025. Wikipedia ZIM Archive wikipedia_en_all_maxi_2025-08 . https:
//library.kiwix.org
[25] Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz,
Adam Stambler, Shyam Upadhyay, and Manaal Faruqui. 2025. Fact, Fetch, and Rea-
son: A Unified Evaluation of Retrieval-Augmented Generation. InProc. NAACL.
[26] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins,
Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob De-
vlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei
Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Nat-
ural Questions: A Benchmark for Question Answering Research.Trans. Assoc.
Comput. Linguistics7 (2019), 452–466. doi:10.1162/tacl_a_00276

Hua et al.
[27] Carlos Lassance, Hervé Déjean, Thibault Formal, and Stéphane Clinchant. 2024.
SPLADE-v3: New Baselines for SPLADE. arXiv:2403.06789 [cs.IR]
[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. InProc. NeurIPS. 9459–9474.
[29] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai,
Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, Jingren Zhou, and Junyang Lin.
2026. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework
for Multimodal Retrieval and Ranking. arXiv:2601.04720 [cs.CL]
[30] Jerry Liu. 2022. LlamaIndex. https://github.com/run-llama/llama_index
[31] Mengjie Liu, Jiahui Peng, Pei Chu, Jiantao Qiu, Ren Ma, He Zhu, Rui Min, Lindong
Lu, Wenchang Ning, Linfeng Hou, Kaiwen Liu, Yuan Qu, Zhenxiang Li, Chao Xu,
Zhongying Tu, Wentao Zhang, and Conghui He. 2025. Dripper: Token-Efficient
Main HTML Extraction with a Lightweight LM. arXiv:2511.23119 [cs.CL]
[32] Wenhan Liu, Xinyu Ma, Yutao Zhu, Yuchen Li, Daiting Shi, Dawei Yin, and
Zhicheng Dou. 2026. Agentic-R: Learning to Retrieve for Agentic Search.
arXiv:2601.11888
[33] Xueguang Ma, Sheng-Chieh Lin, Minghan Li, Wenhu Chen, and Jimmy Lin. 2024.
Unifying Multimodal Retrieval via Document Screenshot Embedding. InProc.
EMNLP. 6492–6505. doi:10.18653/v1/2024.emnlp-main.373
[34] Quinn McNemar. 1947. Note on the Sampling Error of the Difference Between
Correlated Proportions or Percentages.Psychometrika12, 2 (1947), 153–157.
[35] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina
Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al .
2021. WebGPT: Browser-Assisted Question-Answering with Human Feedback.
arXiv:2112.09332 [cs.CL]
[36] Rodrigo Nogueira and Jimmy Lin. 2019. From doc2query to docTTTTTquery.
Technical report, University of Waterloo.
[37] OpenAI. 2025. Introducing ChatGPT Agent: Bridging Research and Action.
OpenAI Blog. https://openai.com/index/introducing-chatgpt-agent/
[38] Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. Technical
report. https://qwen.ai/blog?id=qwen3.5
[39] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance
Framework: BM25 and Beyond.Found. Trends Inf. Retr.3, 4 (2009), 333–389.
doi:10.1561/1500000019
[40] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InProc. ICLR.
[41] Huatong Song, Jinhao Jiang, Wenqing Tian, Zhipeng Chen, Yuhuan Wu, Jiahao
Zhao, Yingqian Min, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-
Searcher++: Incentivizing the Dynamic Knowledge Acquisition of LLMs via
Reinforcement Learning. arXiv:2505.17005
[42] Saba Sturua, Isabelle Mohr, Mohammad Kalim Akram, Michael Günther, Bo
Wang, Markus Krimmel, Feng Wang, Georgios Mastrapas, Andreas Koukounas,
Nan Wang, and Han Xiao. 2024. jina-embeddings-v3: Multilingual Embeddings
With Task LoRA. arXiv:2409.10173 [cs.CL]
[43] Yuan Sui, Mengyu Zhou, Mingjie Zhou, Shi Han, and Dongmei Zhang. 2024.
Table Meets LLM: Can Large Language Models Understand Structured Table
Data? A Benchmark and Empirical Study. InProc. WSDM. 645–654.
[44] Jiejun Tan, Zhicheng Dou, Wen Wang, Mang Wang, Weipeng Chen, and Ji-Rong
Wen. 2025. HtmlRAG: HTML is Better Than Plain Text for Modeling Retrieved
Knowledge in RAG Systems. InProc. WWW. doi:10.1145/3696410.3714546
[45] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Trans. Assoc. Comput. Linguistics10 (2022), 539–554.
[46] Unstructured-IO. 2024. Unstructured: Open-Source Pre-Processing Tools for
Unstructured Data. https://github.com/Unstructured-IO/unstructured
[47] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry W. Wei, Jason Wei, Chris
Tar, Yun-Hsuan Sung, Denny Zhou, Quoc V. Le, and Thang Luong. 2024. Fresh-
LLMs: Refreshing Large Language Models with Search Engine Augmentation. In
Findings of ACL. 13697–13720. doi:10.18653/v1/2024.findings-acl.813
[48] Feng Wang, Zesheng Shi, Bo Wang, Nan Wang, and Han Xiao. 2025.
ReaderLM-v2: Small Language Model for HTML to Markdown and JSON.
arXiv:2503.01151 [cs.CL]
[49] Liang Wang, Nan Yang, and Furu Wei. 2023. Query2doc: Query Expansion with
Large Language Models. InProc. EMNLP. 9414–9423.
[50] Yichuan Wang, Zhifei Li, Zirui Wang, Paul Teiletche, Lesheng Jin, Matei Zaharia,
Joseph E. Gonzalez, and Sewon Min. 2026. PIXELRAG: Web Screenshots Beat
Text for Retrieval-Augmented Generation. arXiv:2606.28344 [cs.IR]
[51] Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay,
Amelia Glaese, John Schulman, and William Fedus. 2024. Measuring Short-Form
Factuality in Large Language Models. arXiv:2411.04368 [cs.CL]
[52] Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang,
Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. 2025. Web-
Walker: Benchmarking LLMs in Web Traversal. InProc. ACL. 10290–10305.
[53] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RECOMP: Improving Retrieval-
Augmented LMs with Context Compression and Selective Augmentation. InProc.ICLR.
[54] Peilin Yang, Hui Fang, and Jimmy Lin. 2017. Anserini: Enabling the Use of Lucene
for Information Retrieval Research. InProc. SIGIR. 1253–1256.
[55] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. InProc. ICLR.
[56] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao
Liu, Shuo Wang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. VisRAG: Vision-
Based Retrieval-Augmented Generation on Multi-Modality Documents. InProc.
ICLR.
[57] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.
2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through
Foundation Models. arXiv:2506.05176 [cs.CL]
A Implementation and Reproducibility
All runs use a consistent Python environment (vLLM 0.25.1) on
NVIDIA A100 GPUs. We will release the code, run manifests, unit
builder, and evaluation records.
Pipeline details.Beyond the setup of §4.1, the hybrid text pipeline
extracts text with Trafilatura 2.0.0, retrieves the top 100 candidates
each from a FAISS IVFFlat index (4,096 lists, 128 probes) and from
Anserini BM25 ( 𝑘1=1.2,𝑏=0.6), and fuses them by reciprocal rank
fusion with constant 60. The 512-token chunking yields 26.8 million
chunks. The reader’s output limit is 200 tokens, and theTessera
scorer truncates each question–unit pair at 512 tokens.
Baselines.HtmlRAG uses the released htmlrag package with its
HTML cleaning, block tree, BGE-large first-stage ranker over 256-
word blocks, and HTML-Pruner-Llama-1B over 128-word blocks,
applied to the same anchor pages. RAPTOR uses the reference set-
tings (100-token leaves, soft clustering, 100-token summaries) with
Qwen3.5-4B as the summarizer, one tree per anchor page. Late
chunking embeds all 15.5 million chunks with jina-embeddings-v3,
once alone and once inside 8,192-token article windows, and each in-
dex replaces the dense retriever. Query2doc and HyDE use Qwen3.5-
4B as a zero-shot generator with the instructions of the original
papers. RM3 uses the Anserini defaults. SPLADE-v3 replaces BM25.
Doc2Query samples 10 queries with doc2query-t5-base-msmarco
for each 511-token segment of every chunk and appends them be-
fore BM25 indexing; Doc2Query-- keeps the 30% of queries that
monoELECTRA scores highest against their segment. Contextual
retrieval uses the published prompt with Qwen3.5-4B over a win-
dow of at most 8,192 tokens of the chunk’s article; we contextualize
the 323,152 chunks among the top-50 candidates of the evaluation
questions and do not re-index the corpus. Docling uses its HTML
backend and hybrid chunker (256 tokens, contextualized headings);
Unstructured usespartition_htmlwithchunk_by_title.
B Evidence Trace Details
Table 7 accounts for the oracle gap of §4.6. Panel (a) groups the 715
oracle questions by where the top-3 pipeline loses the evidence:
Tesseracloses 98 of the 133 answers (74%) where the evidence was
stored but not delivered. Oracle eligibility uses exact matching, and
the stage analysis uses tolerant matching, under which 93 oracle
questions already have their evidence in the reader input. Panel
(b) groups all 5,883 source-supported questions by whereTessera
loses the evidence. Overall, the oracle leadsTesseraby 13.4 points
(95% CI 10.2 to 16.7, 𝑝=5.4×10−15). Table 8 adds the same-page
control to the oracle intervention: an arbitrary chunk of similar

Tessera: Structure-Completed Evidence Units for Web RAG
length from the page of the evidence chunk leaves accuracy at the
baseline level overall, and the evidence chunk beats it by 16.7 to
50.3 points on every dataset.
Table 7: Oracle gap accounting. Gap: oracle minusTessera
correct answers on the oracle questions.
(a) Oracle questions, by top-3 evidence stage
Evidence in the top-3 pipeline 𝑛Oracle gain Closed Gap
Page not anchored 173 69 9 60
Stored, not delivered 449 133 98 35
Delivered 93 8 7 1
All 715 210 114 96
(b) Source-supported questions, byTESSERAevidence stage
Evidence in TESSERA 𝑛 Errors Gap
Page outside top-100 182 107 27
Page outside reranker pool 79 51 18
Page below anchor cutoff 123 61 16
Not in any unit 143 24 1
Missed by lexical prefilter 321 152 20
Missed by selection and packing 217 73 15
Delivered 4,818 425−1
All 5,883 893 96
Table 8: Oracle intervention with a same-page control (ques-
tions that have a control chunk within 20% of the evidence
chunk’s length). Entries are correct answers.
Dataset 𝑛Top-3 Control Evidence vs. control
NQ 454 242 265 341+16.7(92/16)
NQ-Tables 56 28 27 41+25.0(15/1)
SimpleQA 157 69 61 140+50.3(79/0)
All 667 339 353 522+25.3(186/17)
C Standard Remedies in Detail
Table 9 gives the accuracy change of each remedy of §2.2, all applied
to the hybrid text pipeline.
Better extraction.Re-extracting the corpus with Trafilatura’s
recall-oriented mode and rebuilding the index doubles the coverage
of the 449 lost-evidence questions, yet accuracy changes by at most
0.8 points. The recovered text still has to win one of three reader
slots as part of a 1,024-token window.
More context around the hits.Adding the stored neighbors of
every hit on the ten best pages to the candidate pool and reranking
the enlarged pool brings at most 1.5 points.
Structure as a ranking signal.Prefixing each chunk with its page
title moves the evidence chunk into the reranker’s top three for a
net 66 of 3,718 questions; the heading path adds three more, and
table headers push out more evidence chunks than they bring in.
Showing the reader the title-prefixed chunks moves accuracy by
less than a point in either direction.
A larger reader budget.Raising the evidence budget from 3K to
12K tokens lifts pooled accuracy from 71.1% to 73.2%: 2.1 points for
four times the reader cost.
Screenshots.Tiles show the tables that the extractor drops, yet the
screenshot pipeline shares the ceiling of the text pipeline (Table 1),
and raising its context from 3 to 14 tiles adds only 2.2 points.Learned extraction.Beyond Trafilatura’s recall-oriented mode,
we also tested the LLM extractor MinerU-HTML [ 31]. It covers
275 of the 449 lost-evidence questions, yet it classifies infoboxes as
boilerplate (97 of its 156 missed table cells are infobox cells) and
would take about 80 A100 GPU-days to process the corpus.
Table 9: Standard remedies on the hybrid text pipeline, with
the oracle of Table 2 for scale. Δ: accuracy change in points
on NQ/ NQ-Tables/ SimpleQA, or pooled.
Remedy Δaccuracy
Recall-oriented extraction, full re-index +0.5 / +0.8 / +0.4
Neighbor chunks of the top pages, reranked +0.4 / +1.5 / +0.2
Page title prefixed to each chunk +0.6 /−0.4 / +0.7
Reader budget 3K→12K tokens of chunks +2.1 pooled
Screenshot tiles, 3→14 tiles (12K tokens) +2.2 pooled
Oracle evidence chunk (715 affected questions)+29.4
D Ablations
Completion brings the gain.Panel (a) of Table 10 varies the
content of the third slot. Bare units, which are the same rows and
paragraphs without titles, sections, and headers, add 1.8 points
pooled and nothing on NQ-Tables. Completed units add 4.6 points,
and they significantly outperform bare units on every dataset. Pre-
fixing the page title to every chunk, the common “contextual header”
practice, adds 0.5 points. Adding the unit block as a fourth item, at
about 900 more tokens, improves the budget-neutral replacement
by 0.2 points:the content of the slot matters, but its extra
tokens do not.
Faithful grids help tables; serialization is robust.Panel (b)
holds the content fixed and varies construction and packing. The
faithful grid corrects the labels of a third of all table rows, and its
largest effect is on NQ-Tables ( +1.4points over DOM-order pairing).
Grouped packing fits a quarter more lines into the slot. The four
variants stay within 0.34 points pooled, so the method does not
hinge on one serialization; we adopt the grid with table grouping
because it is faithful to the source table and the most compact.
The fixed split is the right allocation.Panel (c) varies the bud-
get split and the pool. Ranking chunks and units jointly by cross-
encoder score loses 0.6 points because scores favor long chunks
over one-row units and squeeze the unit block out on 13% of Sim-
pleQA questions. Replacing all chunks with units loses 1.6 points.
A second unit slot, half or double the anchor pages, and half or
double the prefilter change accuracy by at most 0.3 points. Adding
the sibling rows of the best-scoring row and removing units whose
text the chunks already contain do not help either.The default
configuration sits on a wide plateau.
E Complete Budget Results
Table 11 lists every cell of the text budget sweep of §4.4, including
512-token chunks with a 64-token overlap, and Table 12 gives the
screenshot sweep per dataset. Every gain in Table 12 is significant.
F HtmlRAG at Its Design Point
HtmlRAG targets a 4K-token context. Table 13 evaluates it there
on all 7,851 questions with the same anchor pages asTessera.

Hua et al.
Table 10: Ablations on the hybrid text pipeline (accuracy, %).
Shaded rows are the defaults; panels (b) and (c) share one.
†differs significantly from the default. Panel (a) ran in an
earlier software environment (top-3 chunks: 70.82 pooled).
Variant NQ NQ-Tab. SimpleQA Pooled
(a) What the third slot contains
Third chunk 64.78†48.41†83.61†70.82
Chunks with title prefix 65.38†48.03†84.31†71.33
Bare units, no context 66.49†48.31†85.99†72.58
Completed units 68.17 51.69 89.85 75.44
+ units as a 4th item68.5651.4090.08 75.67
(b) Row construction and packing
DOM-order rows, flat68.3851.31 89.73 75.43
Faithful grid, flat 67.9952.7290.11 75.62
Grid, context grouped 68.11 52.43 90.4375.77
Grid, table grouped 67.84 52.34 90.63 75.74
(c) Budget allocation and pool size
Joint ranking 67.63 53.00 89.32†75.16†
Two unit slots 67.96 51.78 90.22 75.53
Units only 67.60 50.75 87.61†74.09†
Anchors from top 5 67.99 52.06 90.28 75.61
Anchors from top 20 67.78 51.50 90.89 75.71
Prefilter𝑁=5067.81 52.25 89.99†75.43
Prefilter𝑁=20068.35 52.43 90.48 75.90
+ sibling rows 67.81 53.00 90.48 75.75
−redundant units 68.11 52.34 89.41†75.32
Table 11: Pooled accuracy (%) by chunking, evidence budget,
and allocation. “+1K units” gives 1,024 tokens to the unit
block and the rest to chunks.
Chunks Budget Chunks +1K units Half Units only
1,0242K 69.4375.42– 74.39
3K 71.09 75.8476.1274.51
4K 71.39 76.1476.1875.02
6K 72.2776.6076.21 74.75
8K 73.0676.2676.22 74.89
12K 73.20 76.6877.0175.30
5122K 69.0175.75– 74.44
3K 69.60 75.8475.9374.54
4K 70.03 75.9976.1374.81
6K 70.70 76.0476.1474.70
8K 70.50 75.8975.9174.36
12K 70.48 75.6076.1775.02
512,
overlap2K 69.5175.44– 74.37
3K 70.19 75.3875.8574.58
4K 70.58 75.8976.2374.79
6K 70.96 75.9376.0774.82
8K 71.28 76.0076.2774.65
12K 71.28 76.2276.3375.18
Pruned HTML alone loses most answer-bearing text when ten
Wikipedia pages are pruned into 4K tokens, and the reader declines
to answer 1,305 questions. Placed after the top-2 chunks, either
HtmlRAG variant scores below chunks alone.Tesseraleads theTable 12: Screenshot pipeline: top- 𝑘tiles→top-(𝑘−1)tiles +
unit block at the same reader cost (accuracy, %).
Budget 𝑘 NQ NQ-Tables SimpleQA Pooled
2K 2 61.0→67.4 45.5→50.6 82.3→90.4+6.95
3K 3 63.1→67.2 45.8→51.8 83.3→90.1+5.59
4K 4 63.7→67.0 47.4→50.7 83.5→90.7+5.04
6K 7 64.8→67.5 48.1→51.0 84.1→90.8+4.41
8K 9 65.3→67.3 48.6→52.1 84.5→90.8+4.10
12K 14 65.4→67.1 49.3→52.0 84.8→90.5+3.55
best HtmlRAG configuration by 5.8 points ( 𝑝<10−49), andTessera
units alone, without any chunk, reach 74.6%.
Table 13: HtmlRAG andTesseraat a 4,096-token evidence
budget (accuracy, %).Decl.: questions the reader declined.
Context NQ NQ-Tab. SimpleQA Pooled Decl.
Chunks 65.83 48.03 84.31 71.52 118
HtmlRAG 41.28 25.84 17.52 28.75 1,305
HtmlRAG-CE 42.33 26.87 21.85 31.23 1,283
Top-2 + HtmlRAG 64.36 46.72 82.56 69.95 172
Top-2 + HtmlRAG-CE 64.45 48.03 83.14 70.42 174
TESSERAunits only 68.08 50.66 88.22 74.55 96
Top-2 + TESSERA 68.59 53.37 90.60 76.18 80
G Agentic Search Details
Protocol.The reasoning agent is Qwen3.5-4B served with its own
chat template and function-calling format, thinking enabled, the
model card’s sampling settings with a fixed seed, and at most eight
calls to a single search(query) tool. Each call runs the hybrid text
pipeline on the agent’s query and returns three documents. Table 14
breaks down the FRAMES result by annotated reasoning type.
Table 14: FRAMES accuracy (%) of the reasoning agent by
reasoning type; a question can carry several. G/L: questions
gained and lost byTesseraagainst the top-3 chunks.
Type 𝑛Chunks Docling TESSERA G/L 𝑝
Tabular 236 50.8 51.760.244/22 0.009
Numerical 293 53.2 52.961.151/28 0.013
Post-proc. 107 47.7 49.560.721/7 0.013
Temporal 278 57.6 55.063.747/30 0.068
Multi-constraint 549 58.7 56.863.495/69 0.051
All 824 59.2 57.764.4137/94 0.006
H Robustness of the Main Comparison
Table 15 repeats the comparison between the top-3 evidence and
Tesseraunder four robustness checks. The longer output limit
raises the reader’s limit from 200 to 16,384 tokens, the PixelRAG
setting. The updated-answer grader receives the Wikipedia page
that the question was written from and also accepts an answer that
this page states as the current one, which removes the penalty for
references that have gone out of date since the datasets were built.
The supported-reference subset keeps only the questions whose
source page supports the reference answer. The held-out subset

Tessera: Structure-Completed Evidence Units for Web RAG
keeps the 4,851 questions outside the first 1,000 of each dataset,
which our earlier pipeline experiments had used.
We also sampled 50 remainingTesserafailures per dataset and
labeled their cause. An outdated or invalid reference or a grading
error accounts for 72%, 50%, and 12% of the NQ, NQ-Tables, and
SimpleQA samples. Among the pipeline failures, reader errors on
delivered evidence outnumber missing evidence on every dataset
(63 against 20 in total).
Table 15: TheTesseragain under four robustness checks
(pooled accuracy, %; all significant with𝑝<10−12).
Condition Top-3 TESSERA Gain
Default (text pipeline) 70.70 75.74+5.03
Output limit 16,384 instead of 200 tokens 70.95 76.03+5.08
Updated-answer grader, text (7,378) 73.50 78.06+4.55
Updated-answer grader, screenshot (7,378) 71.70 77.46+5.76
Source page supports reference (6,245) 79.34 84.48+5.14
Held-out questions, screenshot (4,851) – –+5.50
I Unit Blocks in a Screenshot Context
Table 16 varies how the unit block enters the screenshot pipeline at
the reader cost of three tiles. The text and image forms of the block
give the same accuracy ( 𝑝=0.89), and a unit image re-packed to
the token cost of the tile it replaces keeps the gain. Allocating two
of the three tile budgets to units is the best configuration, and units
without any tile still beat three tiles by 4.9 points. With Gemma
4 31B as the reader, the text form of the block is 1.9 points more
accurate than its image form.
Table 16: Screenshot pipeline at the three-tile reader cost
(pooled accuracy, %, Qwen3.5-4B reader).
Reader evidence Pooled vs. 3 tiles
Top-3 tiles 69.58
Top-2 tiles + unit block as text 75.18+5.59
Top-2 tiles + unit block as an image 75.23+5.64
Top-2 tiles + token-matched unit image 74.70+5.12
Top-1 tile + units in two tile budgets 75.91 +6.33
Units in three tile budgets 74.51+4.93
J Processing Cost
Table 17 compares the processing that each structure-aware method
adds.Tesseranever times out; HtmlRAG and RAPTOR add GPU
passes over every page.
Table 17: Cost of bringing page structure to the reader. Times
are per page on one CPU core unless noted.
Method Page processing Models added
TESSERA 0.015 s none
Unstructured 0.39 s none
Docling 7.9 s (1.4% time out) none
HtmlRAG 68 GPU-s / question embedder, pruner
RAPTOR 1.0 GPU-s (summaries) LLM, embedder
MinerU-HTML 5.2 GPU-s (extraction) extractor LMK Additional Examples
Figure 7 shows one more gain and one loss from the SimpleQA
run. In the first case, the retrieved chunks discuss the Nobel Peace
Prize, and the unit block delivers the list item that names the Ig
Nobel winner under its year heading. In the second case, the third
chunk thatTesseradisplaced contained the answer, and the unit
block did not: a fixed split occasionally trades away a useful chunk.
Across the 7,851 questions,Tesseragains 620 answers and loses
225.
Gain: evidence in a list under a year heading
Q:Who won the 1991 Ig Nobel Prize for Peace? Reference:Edward Teller
List of Ig Nobel Prize winners › 1991:
Literature: Erich von Däniken, visionary raconteur and author of
Chariots of the Gods?, ...
Peace: Edward Teller, father of the hydrogen bomb and first champion
of the Star Wars weapons system, "for his lifelong efforts to change
the meaning of peace as we know it".
Top-3 chunks: “... there is no mention of an Ig Nobel Prize for 1991 ...”✗
TESSERA: “Edward Teller won the 1991 Ig Nobel Prize for Peace.”✓
Loss: the displaced chunk held the answer
Q:What object did Helly use to attempt suicide in Season 1, Episode 4 of
Severance? Reference:extension cord
Helly R. › Overview:
... She makes several attempts to permanently leave the floor,
including a suicide attempt. ...
Top-3 chunks: “Helly used an extension cord ...”✓
TESSERA: “... hanging herself in an elevator ... It does not specify a partic-
ular object ...”✗
Figure 7: A gain and a loss ofTesseraon SimpleQA.