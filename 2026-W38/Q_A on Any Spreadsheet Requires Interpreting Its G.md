# Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure

**Authors**: Zofia Smoleń

**Published**: 2026-09-17 17:22:43

**PDF URL**: [https://arxiv.org/pdf/2609.20732v1](https://arxiv.org/pdf/2609.20732v1)

## Abstract
Semantic cell annotation improves chunking interpretability for spreadsheets in LLM-driven RAG systems, aiding answer generation through enriched context rather than improved retrieval accuracy. We propose a novel framework of splitting any spreadsheet into interpretable chunks using cell role annotation. Our framework beats the state of the art, yet it faces a hard ceiling. Spreadsheets are fundamentally two-dimensional unstructured data with continuous relationships and infinite potential cell roles. Because classification models are restricted to finite, pre-defined classes, they cannot perfectly capture this structural nuance, even with human-level annotation. We show that addressing the spreadsheet-to-LLM bottleneck requires moving beyond discrete cell classification. Instead, the field must develop dimensionality-reduction techniques to directly flatten 2D unstructured spreadsheets into 1D unstructured text. Text chunks would be easier for downstream RAG to interpret and generate from.

## Full Text


<!-- PDF content starts -->

Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure
Zofia Smoleń
zofsmolen@gmail.com
Systems Research Institute, Polish Academy of Sciences
Warsaw, Poland
Figure 1: The same answer cell (highlighted) as it reaches the LLM in chunks built by three methods, on a real nested-header
sheet from our corpus (a CFTC swaps report). Unstructured emits bare rows with the headers detached; STC maps values to
generic first-row column labels that name nothing. Our grid-aware Row chunk spells out the full context—sheet title and the
complete header path (Total Credit→Buyside, Feb 1→Short)—so the generator can interpret the number without guessing.
Abstract
Semantic cell annotation improves chunking interpretability for
spreadsheets in LLM-driven RAG systems, aiding answer genera-
tion through enriched context rather than improved retrieval accu-
racy. We propose a novel framework of splitting any spreadsheet
into interpretable chunks using cell role annotation. Our framework
beats the state of the art, yet it faces a hard ceiling. Spreadsheets are
fundamentally two-dimensional unstructured data with continuous
relationships and infinite potential cell roles. Because classifica-
tion models are restricted to finite, pre-defined classes, they cannot
perfectly capture this structural nuance—even with human-level
annotation. We show that addressing the spreadsheet-to-LLM bot-
tleneck requires moving beyond discrete cell classification. Instead,
the field must develop dimensionality-reduction techniques to di-
rectly flatten 2D unstructured spreadsheets into 1D unstructured
text. Text chunks would be easier for downstream RAG to interpret
and generate from.
Keywords
spreadsheets, chunking, retrieval-augmented generation, cell clas-
sification, table understanding
1 Introduction
One of the most common applications of language models is answer-
ing questions. Very often the question asked cannot be answered
just based on data the model was trained on—for example when
the information requested isn’t public (like undisclosed financial
data or company secrets). In such situations the user can provide
context to the language model by putting it in the same prompt as
the question. If the source with the information requested is smallenough, it can simply be stuffed into the prompt as a whole. How-
ever, given limited context windows of language models, putting a
5-year history of customer claims against Walmart in the prompt
would be difficult if not impossible. A standard workaround is
Retrieval-Augmented Generation (RAG): first dividing the context
into small pieces (chunking), then searching for the pieces most rel-
evant to the query (retrieval), and at the end generating the answer
based on the question asked and the chunks retrieved (Figure 2).
In this paper we focus on creating those chunks of knowledge,
called chunking. It is comparatively easy to chunk text or other
sequential data: it is usually enough to split on patterns such as a
period, a comma, a formatting type (headline, paragraph), or even a
fixed number of characters. However, chunking unstructured data
organised in some way other than sequentially is still a research
gap. In particular, very little has been done to efficiently chunk real-
life Excel spreadsheets. Tables are often associated with structured
data sources, but in this paper we look broadly at spreadsheets:
real-life sheets (often created in MS Excel or Google Sheets) can
contain tables in any format and quantity on a single tab. Extracting
data fromanyspreadsheet—not a specific spreadsheet created in a
known format—requires upfront interpretation of its structure. We
propose a novel framework where we first semantically annotate
cell roles and then use these roles to extract meaningful, grid-aware
chunks.
Frameworks currently most used in industry mostly just cut
XML into pieces. This strips extracted values of their structural
context: column, row, and table headers among others. Chunks
become blurbs of text and values that happen to be visually close
to each other, and such chunks are usually useless. The difficulty
is confirmed in practice: across 59 active practitioner discussions
on machine-learning forums (r/Rag, r/LangChain) we scraped, the
arXiv:2609.20732v1  [cs.AI]  17 Sep 2026

Zofia Smoleń
Figure 2: Retrieval-Augmented Generation. This paper is
about the first step—chunking—for spreadsheets: how the
sheet is cut determines what context each retrieved value
carries into generation.
most-cited failure mode was loss of context during chunking (59%
of threads), followed by complex or nested headers (37%) and messy,
sparse layouts (29%). Practitioners report a fragmented toolchain—
49% iterate rows with Pandas, 32% use Text-to-SQL agents (which
translate a question into a database query), 25% resort to vision
LLMs to find relationships in the data, 22% use Docling [ 12] (an
open-source document-parsing toolkit whose Excel backend detects
the table regions on a sheet), 17% use Unstructured.io,1and 12%
parse HTML with BeautifulSoup.2None of these reliably preserves
the header→value hierarchy that carries most of the context.
Research on efficiently extracting spreadsheet data for RAG is
limited, but two works stand out: SpreadsheetLLM with its Chain of
Spreadsheet [ 3], and STC [ 7]. SpreadsheetLLM starts from a simple
problem: spreadsheets are too big for an AI to read in one go. So it
shrinks them—about 25 times smaller. It does this by keeping only
the rows and columns where something changes (like where a table
or its headers begin), by writing each repeated value once instead
of fifty times, and by replacing long runs of numbers with a short
note like “these cells are all whole numbers”. What is left is a small
map of the sheet: you can see where everything is, but the actual
numbers have been thrown out.
A map with no numbers cannot answer a question, and that
is why the authors provide a retrieval method called Chain of
Spreadsheet. An LLM reads the map and the question, and instead
of answering, it points: “the answer lives around B2:D40”. That
small piece is then cut out of the original sheet—with the real
numbers still in it—and given to the LLM again, and now it answers.
The weak point is that this assumes you already know which file
to look in. When there are hundreds of sheets, something has to
find the right one first—and all there is to search through are those
1https://unstructured.io
2https://www.crummy.com/software/BeautifulSoup/shrunken maps, which are hard to match against a question because
the numbers are gone.
STC then proved that guiding chunk extraction with cell roles
can improve the quality of answers generated with RAG, though it
does not explain why. It slices a spreadsheet into retrieval chunks
with a fixed rule: it treats the first populated row as the header,
emits one key–value block per data row, and merges those blocks
up to a token budget. It assumes every sheet has the same shape—
header on top, data below—and stamping that template onto a sheet
of that shape yields context-rich chunks. Unfortunately, not every
table looks like that; but such an approach, if generalized, could be
more exact than using an LLM-generated map.
We build on the idea of using cell roles for chunk extraction, but
we do not assume a fixed spreadsheet structure. Such generalization
requireslearningcell roles and the relationships between them—a
cell annotation task with a long history. The earliest attempts relied
on hand-written rules: if a cell is bold, sits in the first row, or is
followed by a column of numbers, call it a header—that kind of
logic, encoded by hand (e.g., the DeExcelerator pipeline [ 5]). Rules
like these work on tidy sheets and break on everything else, much
the same weakness that STC’s fixed template runs into today. The
next wave let the machine learn the rules instead: Fang et al. [ 6]
trained a feature-based classifier to detect and classify table headers
in document tables; Chen and Cafarella [ 1] taught a statistical
model to label whole rows of web-published spreadsheets as titles,
headers or data; Koci and colleagues [ 10] pushed the granularity
down to the single cell, feeding a classifier dozens of hand-picked
clues—the cell’s formatting, its content type, where it sits in the grid.
Their DECO corpus [ 9] of real, manually annotated spreadsheets
gave the field shared training data, though with only a handful
of coarse roles along the lines of data, header, derived value and
note. The third wave dropped the hand-picked clues too and let
neural networks read the grid directly: learned cell embeddings,
multi-task networks that extract header structure straight from the
sheet, TabularNet [ 4] combining a recurrent network with a graph
network over neighbouring cells, and large-scale pretraining on
millions of tables (TUTA [ 17]). The trend line is clear—from rules,
to learned decisions over designed features, to learned features over
the raw grid, with each step handling messier spreadsheets than
the last.
What all of these share, however, is that the cell label is the finish
line: models are scored on classification accuracy and the story ends
there. For chunking we need two things more. First, finer labels—it
is not enough to know that a cell is “a header”; we need to know
how deep it sits in the header hierarchy, because a level-two header
describes a different slice of the data than a level-one header, and
a chunk must carry the right one. Second, we need the labels to
prove their worth downstream: the test of a cell role is not whether
it matches an annotation, but whether the chunk built from it lets
a model answer a question. Those two requirements—depth-aware
roles, judged by the answers they enable—are where our work picks
up.
To learn cell roles we train six neural networks from two different
families. Three of them look at each cell on its own—its text and its
formatting—and guess the role. The other three also learn how cells
are connected, treating the sheet as a graph where neighbouring
cells share information. The two families read a spreadsheet in very

Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure
different ways, and that is on purpose: if both make RAG better,
then the credit goes to the idea itself, not to one lucky network.
That is what happens. On our benchmark of 480 questions, chunks
built from learned roles beat every baseline we test: the best model
scores 3.88 out of 5 on human-rated answer quality; the strongest
competitor gets 3.43.
We can also say why it works. Surprisingly, good roles do not
help the search step much—a chunk matches a question through
its words, whether the roles are right or not. The roles pay off after
retrieval, when the model writes the answer. A model that sees
“51.5” next to the words that describe it can use the number. A model
that sees a bare “51.5” cannot. In short: structure detection does
not make the right chunk easier to find, it makes the found chunk
easier to understand.
Finally, we check how far this approach can go. We rebuild
the chunks using roles annotated by a human—perfect structure
understanding—and the score rises only a little, to 4.01. So the
approach has a ceiling, and even perfect roles sit far from a perfect
score. Some spreadsheets are simply hard. And on the simplest
ones, our machinery adds nothing: a plain first-row-header rule
does just as well. The real bottleneck is no longer recognising the
structure. It is deciding how to cut the sheet into chunks—because
no single cutting rule works for every sheet. That is why we end by
suggesting a less structured way of building chunks: one that adapts
to each sheet, possibly even turning records into plain sentences,
instead of forcing one shape on everything.
2 Methodology
Our methodology proceeds in three steps. First, we train six cell-
role annotation models from two families—node classifiers and
graph learners—on human-labelled spreadsheets. Second, we test
whether the learned roles help downstream Q&A. We build chunks
from each model’s predictions and compare them, on retrieval
(recall@1/@5) and on a 1–5 answer rating from a human judge,
against five baselines: the two most popular Python ingestion paths
(BeautifulSoup and Unstructured) and three academic approaches
(STC, STC with Docling’s table split, and SpreadsheetLLM’s Sheet-
Compressor with Chain-of-Spreadsheet adapted to multiple sheets).
We also compare against chunks built from human-annotated gold
roles, which give the ceiling of the approach. Our models beat all
baselines, but even the gold-role ceiling stays well short of a 5/5
rating. Third, we ask why the approach wins and where its limit
lies. Regressing both metrics on role-model F1 across sixteen de-
ployed checkpoints shows that role quality drives generation, not
retrieval. An ablation that removes one role at a time from the
gold chunks shows the value is carried by the header roles. And a
breakdown by sheet structure shows the gains concentrate on com-
plicated layouts—nested headers, cross-tabs, multiple tables—while
on simple flat tables, where a first-row header already names every
column, all methods converge to a tie.
2.1 Dataset
For training, we used 505 spreadsheet tabs from the Sheetpedia
corpus [ 14], each with cell roles annotated by two human labelers
(with peer review) directly in native Excel. Together these sheetscontain about 1.0M annotated cells, roughly 550K of them non-
empty. Of the 505 tabs, 496 pass minimal-size filtering and form
the training and evaluation pool for the node classifiers; the graph
learners use the 419-sheet subset with multi-table structure. Since
the two families are evaluated on slightly different pools, their
macro-F1 scores are not perfectly cross-comparable.
For RAG evaluation, we used 480 questions targeting 80 held-out
sheets, embedded in a corpus with 302 additional distractor sheets
(382 in total), so retrieval has to find the right sheet among many
plausible ones. The 80 answer sheets also carry full human role
annotations, which serve as a ceiling estimate: from these gold
roles we can build perfect chunks and measure the best a role-based
method could do if it recognized every cell role correctly. Every
answer generated from chunks—ours and every baseline’s—was
rated 1–5 by a human judge.
Each answer sheet was also tagged with one or more structural
categories, which we use to break results down by layout difficulty.
Asimple flat tablehas one header row at the top and plain value
rows below—the layout most tools assume.Nested headersmeans
the headers form a hierarchy (e.g., Year split into Q1–Q4), so a
value only makes sense with its full header path. Amatrix / cross-
tabindexes each value by a row header and a column header at once
(e.g., regions down the side, years across the top). Amulti-table
sheet holds several separate tables, so their boundaries must be
found before anything else can be read. The categories can overlap.
Of the 480 questions, 81 target nested headers, 200 matrix/cross-
tabs, 86 multi-table sheets, and 99 simple flat tables.
Annotation uses 13 cell-role classes, designed around one ques-
tion: what does a cell contribute to a chunk? The two dominant
classes arevalue(observed data, 49% of cells) andempty(45%). Cells
whose content would only add noise to any chunk—placeholder
text, decorative fragments—are labelledjunk(0.2%) and excluded
from chunking altogether.Aggregation(0.9%) separates formula-
derived totals from raw values, since mistaking one for the other
misleads numeric answers. The rest is structure: column and row
headers, each at three nesting depths (4.0% combined), plus sheet-
levelheader,metadata, andcommentcells (0.4%). Header depth
matters because real sheets nest their headers; a single flat “header”
class would collapse exactly the hierarchy that gives values their
meaning. The distribution is highly imbalanced, which motivates
the focal loss [11] used in training.
Annotators also segment each sheet into logical tables (T0 for
sheet-wide scope, T1+ for each distinct table), where a table is a
contiguous region meant to be read as one set of records. This
is what stops the chunker from ever mixing two unrelated tables
in one chunk. Importantly, table membership is not a prediction
target: the models are trained only on the 13 cell roles (the graph
learners’ auxiliary edge losses supervise cell adjacency and header–
value links, never table ids), and the human table ids serve only for
gold-role chunks and the oracle ceiling. At inference time, tables
are recovered by a region detector that finds connected blocks of
non-empty cells; we use Docling’s Excel segmentation [ 12], but the
pipeline is agnostic to the segmenter—any table-detection method
can be plugged in, since the learned component is needed only
for roles. Together, roles and table structure compactly encode the
sheet’s structure graph: cells are nodes, role labels type the nodes,

Zofia Smoleń
MLP GCN GAT AdjTransf. DualMod. SpatialET0.50.60.70.8T est macro-F1
0.5840.6290.672 0.7180.7490.726
Node classifiers
Graph learners
Deployed checkpoints
Figure 3: Test macro-F1 across 15 runs (5 folds ×3 seeds) per
architecture: per-run pooled 13-class macro-F1, with means
as data labels.
and table membership plus header depth encode the header-owns-
value edges that chunk assembly later follows.
2.2 Learning cell roles: graph learners beat node
classifiers
To test whether recognizing cell roles improves spreadsheet RAG,
we trained six role-annotation models belonging to two families.
Node classifiers take table membership as given (from Docling’s
segmentation) and predict the 13 cell roles within tables: an MLP
(three linear blocks, no message passing—the non-relational base-
line), a GCN [ 8] (three GCNConv layers with LayerNorm residu-
als), and a GAT [ 15] (three GATConv layers with a learned edge
encoder). Graph learners additionally learn the sheet’s structure
itself—from value cells through multi-level row and column headers
up to tables—by predicting cell-to-cell links alongside the roles: an
AdjTransformer (a TransformerEncoder with four bilinear adja-
cency heads), a DualModalityGNN (separate content and format
encoders fused over a learned 𝑘-NN graph), and a SpatialEdgeTrans-
former (spatial-bias attention with a typed edge scorer). All six
consume the same 857-dimensional per-cell feature vector built
from content, formatting, formula, and neighborhood signals; gold
labels are never used as inputs. Full architecture details are in the
supplementary material.
Every architecture was trained under an identical grid on its
training pool from §2.1: five-fold cross-validation at the sheet level,
three seeds (42, 2137, 10042010), 100 epochs with AdamW and
a cosine learning-rate schedule—90 runs in total. The loss is fo-
cal loss (𝛾=2) over the 13 classes with inverse-frequency weights,
countering the heavy class imbalance; graph learners add a binary
cross-entropy term on predicted adjacency, with gold header–value
edges used only as auxiliary targets, never as inputs. We report
per-run pooled macro-F1, i.e., the per-sheet confusion matrices of
a run’s test fold pooled into one matrix and macro-averaged over
all 13 classes. As Figure 3 shows, every graph learner outperforms
every node classifier (macro-F1 0.72–0.75 vs. 0.58–0.67, DualModal-
ityGNN best at 0.75). Since the composition of tables varies from
sheet to sheet, we hypothesize that learning the relationships be-
tween cells transfers across layouts better than classifying each cell
in isolation, though we do not test this mechanism directly.
Table 1 breaks the same scores down by class, with the best
model per class in bold and fold-to-fold variance reported alongside
the mean—for the rare classes (deeper header levels, comments,sheet headers, each under 1% of cells) a handful of sheets can swing
the score, so the variance is as informative as the mean. Rare classes
are difficult for every model, which means the predicted structure
our chunks are built from is never perfect and any downstream
method must tolerate this noise; graph learners are nevertheless
the best and most stable family, leading on the majority of classes
with low to moderate variance.
2.3 Cell role annotation helps RAG
We evaluated RAG quality end-to-end on the Q&A dataset described
above: 480 questions over 80 answer sheets, retrieved from the full
382-sheet corpus (answer sheets plus 302 distractors). For our ap-
proach, we deployed one trained checkpoint per architecture and
built chunks from each checkpoint’s predicted roles; for four of
the six architectures the deployed checkpoint is the best cross-
validation fold, and for GAT and DualModalityGNN a high-scoring
fold (Figure 3 marks every deployed checkpoint as a star, includ-
ing the additional low- and mid-quality checkpoints used in §2.4).
We compared them against five baselines. Two are the most pop-
ular Python paths for ingesting spreadsheets in industry practice:
BeautifulSoup, where the sheet is converted to an HTML table and
parsed with bs4 (a general-purpose HTML parser widely used as the
quick default for tabular ingestion), with each parsed row becom-
ing a chunk; and Unstructured, a document-ingestion framework
with native spreadsheet support. Three are academic approaches:
STC [ 7], the strongest structure-aware baseline chunker; STC pre-
ceded by Docling’s table split—so this variant runs STC within each
detected table rather than on the whole sheet, using the same seg-
menter as our own pipeline and thereby isolating the effect of the
roles from the effect of segmentation; and SpreadsheetLLM’s Sheet-
Compressor with Chain-of-Spreadsheet [ 3], modified to work over
multiple sheets. We compared all methods on retrieval—recall@ 𝑘,
the share of questions for which a chunk containing the answer
cells appears among the top 𝑘retrieved results ( 𝑘=1and5)—and
on answer quality, a final RAG rating from 1 to 5 given by a human
judge to every generated answer.
Table 2 shows the outcome. Every architecture scores at or above
every baseline. The strongest checkpoint (GAT) reaches a human
rating of 3.88 against STC’s 3.43 (Wilcoxon 𝑝= 1.5×10−6) and dom-
inates on retrieval as well (recall@1 0.465 vs. 0.356); DualModal-
ityGNN, AdjTransformer, and GCN also beat STC significantly
(𝑝≤ 3×10−3), SpatialET sits at the significance threshold ( 𝑝= 0.05),
and even the non-relational MLP matches it (3.52 vs. 3.43, 𝑝= 0.43).
Additionally, to see how good RAG can get if the cell roles are
known perfectly, we used the human role annotations of all 80 an-
swer sheets to build gold-role chunks. The last row of Table 2 shows
this ceiling: 4.01—clearly above every learned checkpoint, yet still
far from a perfect 5.0. So we know that interpreting spreadsheet
structure through cell role annotation helps RAG; we do not yet
know why, and we can also see that the approach has a limit even
with perfect roles.
Role predictions by themselves do not dictate a chunk shape,
and we had no a-priori reason to prefer one, so we treated the
assembly geometry as an open choice and tested three over the
same predicted roles (from the deployed GAT checkpoint of Ta-
ble 2):Row, which emits one chunk per record row and attaches

Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure
Table 1: Per-class F1 (mean±SD over 15 runs = 5 folds×3 seeds). Best mean per class in bold.
node classifiers graph learners
Class %cells MLP GCN GAT DualMod. AdjTr. SpatET
empty47.5.970 ±.024.966±.033.975±.031.979±.008.975±.013.973±.012
value39.4.889 ±.050.905±.053.928±.054.942±.015.939±.013.934±.018
row_header_15.47.730 ±.066.773±.081.799±.099.890±.032.887±.023.887±.026
col_header_12.54.880 ±.024.895±.031.913±.034.924±.018.916±.011.914±.017
aggregation1.53.456 ±.201.535±.234.616±.225.710±.092.617±.113.614±.114
metadata1.46.589 ±.157.656±.149.735±.162.849±.027.833±.030.827±.035
row_header_20.61.295 ±.131.325±.148.396±.187.574±.115.486±.149.511±.142
junk0.57.419 ±.134.459±.119.465±.129.480±.156.455±.153.453±.137
header0.27.708 ±.060.761±.080.765±.088.775±.051.749±.064.759±.058
row_header_30.27.369 ±.287.356±.323.456±.320.791±.238.761±.298.815±.212
col_header_20.22.340 ±.132.430±.169.440±.172.572±.166.521±.171.532±.179
comment0.09.655 ±.095.734±.098.771±.102.740±.114.742±.104.765±.100
col_header_30.03.272 ±.283.371±.337.459±.392.485±.383.418±.343.434±.360
Macro-F1.584 ±.075.629±.081.672±.091.749±.057.718±.064.726±.049
Table 2: End-to-end RAG results on 480 questions (human
ratings, 1–5). Best per column in bold (oracle excluded).
Method Recall@1 Recall@5 Human
Baselines (no learned roles)
BeautifulSoup 0.219 0.400 1.91
Unstructured 0.273 0.496 2.60
SheetCompr.+CoS 0.210 0.285 3.00
STC+Docling 0.325 0.492 3.34
STC 0.356 0.515 3.43
Learned per-cell roles (ours; all six architectures)
MLP 0.467 0.613 3.52
GCN 0.450 0.610 3.71
DualModalityGNN0.4920.623 3.82
GAT 0.4650.640 3.88
SpatialET 0.433 0.585 3.62
AdjTransformer 0.458 0.625 3.75
Oracle (gold roles) 0.502 0.608 4.01
to every value the full nesting path of its column and row head-
ers;Flat, which serializes each table region into a single chunk;
andKG, which decomposes tables into subject–predicate–object
triples. In all three, chunks never cross the detected table boundaries,
junk cells are dropped, and aggregation cells stay distinguishable
from raw values. Each assembly was evaluated with the identi-
cal protocol: all 382 corpus sheets are chunked and indexed both
densely (intfloat/multilingual-e5-base [16], 768-d) and lexi-
cally (BM25 [ 13]); for each question the top ten chunks are retrieved
by reciprocal rank fusion [ 2] (𝑘=60), the top five go to the generator
(gemini-2.5-flash , temperature 0), and every generated answer
is rated 1–5 by a human judge.
Table 3 reports the outcome by table type: no single geometry
is best for all table types. Row wins overall and on every struc-
tured type—nested headers, matrix, and multi-table sheets—but
on simple flat tables the strategies converge to a tie, and the win-
ning margin depends significantly on the table type (strategy ×typeTable 3: Human rating (1–5) by chunk-assembly strategy and
sheet structure (𝑛=480).
Sheet structure𝑛STC Row Flat KG
Nested (multi-level) headers 81 3.124.143.35 2.43
Matrix / cross-tab 200 3.714.133.81 3.20
Multi-table 86 4.034.233.67 1.66
Simple flat table 99 3.82 3.813.842.11
All queries 480 3.433.883.60 2.51
interaction, 𝑝=1.0×10−3). We therefore adopt Row as the default
geometry throughout the paper while explicitly not claiming it
is optimal: a single rigid assembly rule leaves value on the table,
which is itself evidence that structure-conditioned, ideally learned,
chunk assembly is the natural next step.
2.4 Cell role annotation helps generation, not
retrieval
Regardless of architecture, better role recognition makes for better
RAG answers. The next logical step is to see why. To isolate the ef-
fect of role quality, we deploy sixteen checkpoints spanning a wide
range of role quality (pooled macro-F1 0.50–0.82): low/mid/high
triples within five architectures, plus a single checkpoint for the
sixth (SpatialEdgeTransformer, whose memory limits made fur-
ther runs unreliable). Each checkpoint runs the identical down-
stream pipeline—same Row chunk assembly, same corpus, same
480 questions—so the only thing that varies is how well the cell
roles are recognized. For every checkpoint we measure retrieval (re-
call@1, recall@5) and human-rated answer quality (480 ratings per
checkpoint), and fit an OLS regression of each metric on macro-F1
(Figure 4).
As the figure shows, the two effects split cleanly. Answer quality
rises with role quality: the fit gains +0.56rating points per unit of
macro-F1 (𝑝=0.031), and extrapolating it to perfect roles (F1 =1.0)
predicts 3.9—consistent with the gold-role oracle, which actually

Zofia Smoleń
0.5 0.6 0.7 0.80.400.420.440.460.480.500.52Recall@1y = 0.45 + 0.01·F1
p = 0.863 (not significant)
0.5 0.6 0.7 0.80.580.600.620.640.66Recall@5y = 0.61 + 0.01·F1
p = 0.897 (not significant)
0.5 0.6 0.7 0.83.43.53.63.73.83.94.0Human rating (1-5)y = 3.32 + 0.56·F1
p = 0.031 (significant)
checkpoint macro-F1node classifier checkpoint graph learner checkpoint OLS regression
Figure 4: Sixteen deployed checkpoints, identical downstream pipeline: checkpoint macro-F1 against Recall@1, Recall@5, and
the human answer rating, with OLS fits.
attains 4.01: the ceiling sits on the same curve. Retrieval, in contrast,
does not depend on role quality at all: both regression lines are flat
(slopes≈0,𝑝=0.86and0.90).
The reason is simple. A chunk matches a query lexically and
semantically whether or not its cell roles were recognized correctly,
so retrieval is blind to role quality. But for the LLM to interpret a
retrieved value, the raw value alone is not enough—it also needs
to know what the value means and where it sits: which column
names it, which row it belongs to, which header hierarchy gives it
units and scope. That context can only be packed into a chunk if
the cell roles—and with them the table’s structure—are recognized
correctly. So good role recognition pays off when the answer is
generated, not when the chunk is retrieved.
This is exactly what the opening example (Figure 1) illustrates.
The three chunks shown there carry the same answer cell from the
same nested-header sheet, and the sheet is findable for the retriever
in every case—yet the chunks read very differently. Unstructured
emits bare rows with the headers detached from their values; STC
maps the values to generic first-row column labels that name noth-
ing. Our Row chunk spells out the full context: the sheet title (the
headerrole marks sheet-level titles), the record’s row header, and
the complete column-header path above the value. Only the last
one lets the generator tell which counterparty, which date, and
which position a number belongs to without guessing—and that
difference, invisible to retrieval, is precisely the gap the regression
in Figure 4 measures.
2.5 Non-standard tables gain the most from
predicted roles
Knowing that roles help generation raises two follow-up questions:
which roles carry the benefit, and on which sheets does it material-
ize? We answer the first with an ablation: starting from the gold-role
chunks of the 80 answer sheets—the same human annotations that
define the oracle ceiling, so no prediction error is involved—we
remove one role class at a time by demoting every cell carrying it
to plainvalue, rebuild the chunks with the identical Row assembly,Table 4: Ablation on gold chunks: judge loss after removing
one role class.
Role removed judge loss𝑝(Wilcoxon) interpretation
col_header_11.402×10−33names the values
col_header_20.301×10−8deeper column header
row_header_20.229×10−6deeper row header
row_header_10.130.018primary row header
header0.120.023sheet-level title
aggregation0.060.038formula-derived totals
metadata0.06 n.s. no measurable effect
and rerun the identical retrieval and generation pipeline on all 480
questions. Thejudge lossof a role is the drop in mean answer score
relative to the unmodified gold chunks (baseline 3.66; panel scores
sit systematically lower in level than the human columns of Ta-
ble 2, but every judge loss is a panel-minus-panel difference, so the
levels cancel), with significance from a Wilcoxon signed-rank test
on the per-question paired differences. Because this is an auxiliary
mechanism analysis with seven ablation arms (7 ×480answers), it
is the one evaluation scored not by the human judge but by a panel
of five versioned open-weight LLM judges using the identical 1–5
rubric; on answer sets carrying both panel and human ratings, the
panel agrees with the human judge within one point on ≥96%of
questions, so the substitution is safe for this auxiliary analysis.
Table 4 shows the result: chunk usefulness lives in the headers,
and overwhelmingly in one of them. Demoting primary column
headers (col_header_1 ) costs 1.40 rating points ( 𝑝=2×10−33)—an
order of magnitude more than any other role—because a value
whose column is unnamed is just a number. The rest of the header
hierarchy follows at a distance ( col_header_2 0.30,row_header_2
0.22,row_header_1 0.13, sheet-level header 0.12), and it is exactly
the part that only exists on sheets with nested structure: a simple
table has no second header level to lose. Aggregation costs little
(0.06) and metadata nothing measurable.

Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure
Table 5: Human rating (1–5) by sheet structure: the strongest
baseline (STC), our two best checkpoints, and the gold-role
oracle.
Sheet structure𝑛STC GAT DualMod. OracleΔ
Nested headers 81 3.12 4.14 3.744.27+1.15
Matrix / cross-tab 200 3.714.133.98 4.08+0.36
Multi-table 86 4.03 4.23 4.124.40+0.36
Simple flat table 99 3.82 3.81 3.913.92+0.10
If the value of roles lies in recovering header hierarchy, the payoff
should concentrate on sheets that have one—and it does. Table 5
breaks the human-rated advantage over the strongest baseline (STC)
down by table type, alongside the gold-role ceiling. On nested-
header sheets, perfect roles are worth +1.15rating points over
STC, and the best learned checkpoint already realizes +1.01of that;
matrix and multi-table sheets gain +0.36at the ceiling. On simple
flat tables the entire effect disappears: the ceiling shrinks to +0.10
and the best learned method to−0.01—a tie.
This is no surprise. On a simple flat table, the first row already
names every column, so a plain first-row heuristic recovers all the
structure there is, and role annotation has nothing left to add. On
non-standard layouts—nested headers, cross-tabs, multiple tables
per sheet—that heuristic breaks down, and recovering the header
hierarchy is worth up to a full rating point. Cell role annotation, in
other words, is not a general-purpose booster: it is a targeted fix
for exactly the tables that existing tools read wrong.
2.6 Why our method wins—and why room for
improvement remains
We argue that the reason for this may be subtle: everyone who
ever created a spreadsheet did it using their own preferences. Our
semi-structured approach may not be enough for subtle differences
between layouts and the infinite possibilities of table construction
and interpretation. Even if we, as humans, read the table and assign
roles, they may not exactly match the original author’s idea.
We see that higher F1 reflects in better RAG quality, and that
cell roles related to header order and relationships between values
matter most for RAG quality. We see that the tables where we
most outperform STC are the complicated, nested and messy ones.
We think that pushing the approach further will require chunk
assembly that adapts to each sheet’s layout—up to rendering records
as plain natural-language sentences—rather than committing to
any fixed geometry in advance.
3 Conclusion
We showed that semantic cell-role annotation substantially im-
proves spreadsheet RAG: chunks built from learned roles beat the
strongest state-of-the-art chunker by +0.45points on human-rated
answer quality (3.88 vs. 3.43, 𝑝=1.5×10−6). The reason is not better
retrievability but better interpretability of the chunks—each value
arrives with the headers that explain it—and the benefit concen-
trates exactly where layouts are non-standard. We also measured
the ceiling of the approach: even chunks built from perfect, human-
annotated roles reach only 4.01 of 5. We argue this is because afixed set of cell-role classes cannot capture the full variety of rela-
tionships that authors express in real-life spreadsheets, and because
no single chunk geometry suits every table type—which together
point toward learned, structure-conditioned rendering of 2D sheets
into text as the way past the ceiling. To make the finding usable, we
release our framework, which is modular in both the role annotator
and the chunk geometry; in our experiments, the best configura-
tions paired the GAT or DualModalityGNN annotator (statistically
indistinguishable downstream) with row-based chunk assembly.
The benchmark behind our evaluation—480 questions, gold role
annotations, and roughly 18,000 human answer ratings—will be
released separately as a dataset publication.
References
[1] Zhe Chen and Michael Cafarella. 2013. Automatic Web Spreadsheet Data Extrac-
tion. InProceedings of the 3rd International Workshop on Semantic Search over the
Web. doi:10.1145/2509908.2509909
[2] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. 2009. Reciprocal
Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods.
InProceedings of the 32nd International ACM SIGIR Conference on Research and
Development in Information Retrieval. 758–759. doi:10.1145/1571941.1572114
[3]Haoyu Dong, Jianbo Zhao, Yuzhang Tian, Junyu Xiong, Mengyu Zhou, Yun
Lin, José Cambronero, Yeye He, Shi Han, and Dongmei Zhang. 2024. Encoding
Spreadsheets for Large Language Models. InProceedings of the 2024 Conference
on Empirical Methods in Natural Language Processing (EMNLP). Association for
Computational Linguistics, 20728–20748. doi:10.18653/v1/2024.emnlp-main.1154
System name: SpreadsheetLLM. arXiv:2407.09025.
[4] Lun Du, Fei Gao, Xu Chen, Ran Jia, Junshan Wang, Jiang Zhang, Shi Han, and
Dongmei Zhang. 2021. TabularNet: A Neural Network Architecture for Under-
standing Semantic Structures of Tabular Data. InProceedings of the 27th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining (KDD). 322–331.
doi:10.1145/3447548.3467228
[5] Julian Eberius, Christopher Werner, Maik Thiele, Katrin Braunschweig, Lars Dan-
necker, and Wolfgang Lehner. 2013. DeExcelerator: A Framework for Extracting
Relational Data from Partially Structured Documents. InProceedings of the 22nd
ACM International Conference on Information and Knowledge Management (CIKM).
2477–2480. doi:10.1145/2505515.2508210
[6]Jing Fang, Prasenjit Mitra, Zhi Tang, and C. Lee Giles. 2012. Table Header
Detection and Classification. InProceedings of the Twenty-Sixth AAAI Conference
on Artificial Intelligence. 599–605.
[7] Pooja Guttal, Varun Magotra, Vasudeva Mahavishnu, Natasha Chanto, Sidharth
Sivaprasad, and Manas Gaur. 2026. Structure-Aware Chunking for Tabular Data
in Retrieval-Augmented Generation. arXiv:2605.00318 arXiv:2605.00318.
[8]Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with
Graph Convolutional Networks. InInternational Conference on Learning Repre-
sentations (ICLR).
[9] Elvis Koci, Maik Thiele, Josephine Rehak, Oscar Romero, and Wolfgang Lehner.
2019. DECO: A Dataset of Annotated Spreadsheets for Layout and Table Recog-
nition. InProceedings of the International Conference on Document Analysis and
Recognition (ICDAR). 1280–1285. doi:10.1109/ICDAR.2019.00207
[10] Elvis Koci, Maik Thiele, Oscar Romero, and Wolfgang Lehner. 2016. A Machine
Learning Approach for Layout Inference in Spreadsheets. InProceedings of the
8th International Joint Conference on Knowledge Discovery, Knowledge Engineering
and Knowledge Management (KDIR). 77–88. doi:10.5220/0006052200770088
[11] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017.
Focal Loss for Dense Object Detection. InProceedings of the IEEE International
Conference on Computer Vision (ICCV).
[12] Nikolaos Livathinos, Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele
Dolfi, et al .2025. Docling: An Efficient Open-Source Toolkit for AI-driven
Document Conversion.arXiv preprint arXiv:2501.17887(2025). Accepted at the
AAAI 2025 Workshop on Open-Source AI for Mainstream Use. https://github.
com/docling-project/docling.
[13] Stephen Robertson and Hugo Zaragoza. 2009. The Probabilistic Relevance Frame-
work: BM25 and Beyond.Foundations and Trends in Information Retrieval3, 4
(2009), 333–389. doi:10.1561/1500000019
[14] Zailong Tian, Zhuoheng Han, Houfeng Wang, and Lizi Liao. 2025. Sheetpedia:
A 300K-Spreadsheet Corpus for Spreadsheet Intelligence and LLM Fine-Tuning.
InAdvances in Neural Information Processing Systems (Datasets and Benchmarks
Track). https://huggingface.co/datasets/tianzl66/Sheetpedia_xlsx.
[15] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro
Liò, and Yoshua Bengio. 2018. Graph Attention Networks. InInternational
Conference on Learning Representations (ICLR).

Zofia Smoleń
[16] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang,
Rangan Majumder, and Furu Wei. 2022. Text Embeddings by Weakly-Supervised
Contrastive Pre-training.arXiv preprint arXiv:2212.03533(2022).
[17] Zhiruo Wang, Haoyu Dong, Ran Jia, Jia Li, Zhiyi Fu, Shi Han, and Dongmei
Zhang. 2021. TUTA: Tree-based Transformers for Generally Structured Table
Pre-training. InProceedings of the 27th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining. 1780–1790. doi:10.1145/3447548.3467434

Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure
A Architecture details
All models share a common interface: forward produces node embeddings ℎ,classify_nodes mapsℎto 13-class logits, and predict_edges
produces hierarchy and table-boundary edge logits. Table 6 summarises backbone layers, total learned depth (including encoder, classification
head, edge heads, and auxiliary modules), and parameter counts.
Table 6: Architecture specifications. Hidden dim is 128 for node classifiers, 256 for graph learners. All models receive the same
857-d input feature vector. “Depth” counts backbone layers plus input encoder, classification head, edge heads, and (for graph
learners) adjacency heads and optional refinement layers.
Family Architecture Backbone Depth Params Key components
Node
classifiersMLP 3 7 718 K Linear→LayerNorm→ReLU→Dropout; no
message passing
GCN 3 8 784 K GCNConv with symmetric normalisation, residual
+ LayerNorm per layer
GAT 3 9 2.18 M GATConv (4 heads, averaged), learned edge en-
coder (23-d→hidden), residual + LayerNorm
Graph
learnersAdjTransformer 4∼15 3.57 M Row/col pos. embeddings, TransformerEncoder (4
heads, GELU), 4 bilinear adjacency heads, optional
StructureRefinementGNN
DualModalityGNN 2∼17 1.93 M Dual content/format encoders, bilinear fusion,
learned𝑘-NN graph, graph reasoning layers with
edge MLPs
SpatialEdgeTransformer 4∼14 3.62 M Row/col pos. embeddings, TransformerEncoder (4
heads), spatial-bias attention (7-d pair features),
typed MLP edge scorer
The six architectures split into two families that differ inwhere the graph comes from. Node classifiers consume a fixed, hand-engineered
adjacency (dashed gray in the diagrams); graph learners predict the graph topology via learned adjacency heads (yellow blocks) and optionally
refine it through a second-pass GNN (cyan). Figure 5 contrasts the two patterns side by side; the same colour coding is used in all six
per-architecture diagrams that follow.
857-d input
Encoder
Backbone
conv / linearEngineered
adjacency
(fixed)
Node head
→13 logitsEdge heads
(a) Node classifier pattern857-d input
Projection
Backbone
transformerAdjacency
heads
(learned)
Refinement
GNN
(optional)
Node head
→13 logitsEdge headslearned adjacency
(b) Graph learner pattern
Figure 5: The defining architectural difference. Left: node classifiers receive a fixed, hand-engineered adjacency graph (dashed
gray); the model only learns node representations over this static wiring. Right: graph learners predict the graph topology
via learned adjacency heads (yellow); predicted edges feed an optional refinement GNN (cyan) before classification. The bold
“learned adjacency” arrow is the feedback path absent from node classifiers.
Colour key.All architecture diagrams use the same block colours: input features, encoder / projection, backbone (conv or
transformer), fixed engineered graph (node classifiers only; dashed border), learned adjacency heads(graph learners only), optional
refinement GNN, classification head, edge prediction heads.
A.1 Node classifiers
Node classifiers operate on the fixed engineered graph (grid neighbours, row/column strips, sheet-to-table bridges) and do not modify graph
topology. All three share a common structure: an input encoder (Linear + LayerNorm + ReLU + Dropout) projects the 857-d feature vector

Zofia Smoleń
to the hidden dimension, a backbone processes node representations, and a 2-layer MLP classification head maps to 13 role logits. Two
additional 2-layer MLP heads predict hierarchy and table-boundary edges. Figures 6–8 show each architecture individually.
MLP (Figure 6).The simplest baseline: three fully-connected blocks (Linear →LayerNorm→ReLU→Dropout) classify each cell
independently from its feature vector, with no message passing. Despite ignoring graph structure entirely, MLP provides a strong feature-only
baseline that isolates the contribution of the 857-d cell representation. As the diagram shows, the backbone is purely feed-forward — no
neighbour information enters the computation at any stage.
857-d input
Input encoder
Linear + LN + ReLU + Drop
Linear blocks×3
LN + ReLU + Drop each
no message passing
Node head
2-layer MLP→13 logitsEdge heads×2
hierarchy + table bdyNo adjacency
(no graph)×
Figure 6: MLP (718 K params). Each cell is classified independently from its 857-d feature vector; three linear blocks replace
convolution. The dashed gray box with “×” emphasises that no graph structure is used — this is a non-relational baseline.
GCN (Figure 7).Three GCNConv layers with symmetric normalisation propagate information along the engineered graph. Each layer
applies a residual connection followed by LayerNorm, enabling gradient flow through the message-passing stack. The key difference from
MLP is that GCN aggregates features from spatial neighbours via the fixed adjacency, so each cell’s representation reflects its local context.
857-d input
Input encoder
Linear + LN + ReLU + Drop
GCNConv×3
symmetric norm
residual + LN each
Node head
2-layer MLP→13 logitsEdge heads×2
hierarchy + table bdyEngineered
adjacency
(fixed)
Figure 7: GCN (784 K params). Three GCNConv layers propagate features over the fixed engineered adjacency (grid, row/col
strips, sheet-to-table bridges). The dashed box indicates the static graph that is not learned.
GAT (Figure 8).Three GATConv layers with 4 attention heads (averaged) learn to weight neighbour messages. A learned edge encoder
maps the 23-d edge feature vector (type, relative position) to the hidden dimension, providing edge-type-aware attention. Each layer includes
residual connections and LayerNorm. GAT is the largest node classifier (2.18 M parameters) due to the multi-head attention and edge
encoding parameters. The dedicated edge encoder, shown in the diagram as a separate block feeding into each GATConv layer, is the
distinguishing feature over GCN.

Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure
857-d input
Input encoder
Linear + LN + ReLU + Drop
GATConv×3
4 heads, averaged
residual + LN eachEdge encoder
23-d→hidden
Node head
2-layer MLP→13 logitsEdge heads×2
hierarchy + table bdyEngineered
adjacency
(fixed)
Figure 8: GAT (2.18 M params). Three GATConv layers with 4 attention heads learn to weight neighbour messages over the fixed
adjacency. A learned edge encoder maps each edge’s 23-d feature vector (type identifier + relative position) into the hidden
dimension, enabling edge-type-aware attention — the key addition over GCN.
A.2 Graph learners
Graph learners additionally predict soft adjacency matrices and receive edge-level supervision for four edge types (table membership,
hierarchy, same-row, same-column). They produce 𝑁×𝑁 logit matrices via learned heads, optionally refine predictions through a Struc-
tureRefinementGNN, and jointly optimise node classification and edge prediction losses. Figures 9–11 show each architecture; the same
colour coding as the node classifiers applies, with two additional colours: yellow for adjacency prediction heads and cyan for the optional
refinement GNN.
AdjTransformer (Figure 9).An input projection maps the 857-d features to the hidden dimension, augmented with learned row and column
positional embeddings. Four TransformerEncoder layers (4 heads, GELU activation) produce contextualised node representations. Four
bilinear adjacency heads compute 𝜎(𝑄𝑘𝐾⊤
𝑘)matrices, one per edge type. When struct_refine_active is set, predicted edges above a
learned threshold are fed into a 2-layer StructureRefinementGNN for second-pass message passing before the classification head. The bilinear
heads (right branch in the diagram) are the distinguishing mechanism: each head learns a separate query–key space for one edge type.
857-d input
Input projection
+ row/col pos. embed
TransformerEncoder×4
4 heads, GELUBilinear adj. heads×4
𝜎(𝑄𝑘𝐾⊤
𝑘)per edge type
StructureRefinementGNN
(optional, 2 layers)
Node head
2-layer MLP→13 logitsEdge heads×2
hierarchy + table bdylearned adjacency
Figure 9: AdjTransformer (3.57 M params). A 4-layer TransformerEncoder produces node representations; four bilinear adjacency
heads predict edge types via 𝜎(𝑄𝐾⊤). The bold “learned adjacency” arrow feeds predicted edges into an optional 2-layer
StructureRefinementGNN before the classification head — no fixed graph is provided.
DualModalityGNN (Figure 10).Content statistics and formatting features are encoded through separate 2-layer pathways and merged
via a bilinear fusion block. A learned𝑘-nearest-neighbour graph is constructed from fused embeddings. Two graph reasoning layers (each
containing an edge MLP, attention-weighted aggregation, and node update) propagate information over the combined topology. Four

Zofia Smoleń
adjacency heads (same bilinear 𝜎(𝑄𝐾⊤)pattern) predict edge types. DualModalityGNN has the most complex input processing ( ∼17 total
learned layers) but the fewest parameters among graph learners (1.93 M) because it uses a smaller hidden dimension for the dual encoders.
The dual-path input stage, shown as two parallel encoder blocks in the diagram, is the defining design choice: it forces the model to learn
modality-specific representations before fusion.
857-d input
Content encoder
2-layer MLPFormat encoder
2-layer MLP
Bilinear fusion
Learned𝑘-NN graph
+ 2 graph reasoning layers
edge MLP + attn agg.Adjacency heads×4
𝜎(𝑄𝐾⊤)per edge type
Node head
2-layer MLP→13 logitsEdge heads×2
hierarchy + table bdylearned adjacency
Figure 10: DualModalityGNN (1.93 M params). Content and format features are encoded through separate pathways, fused via a
bilinear block, and processed by graph reasoning layers over a learned 𝑘-NN topology. Adjacency heads predict edge types and
feed learned adjacency back into the graph reasoning layers — no fixed graph is provided. The dual-encoder input stage is the
distinguishing design.
SpatialEdgeTransformer (Figure 11).An input projection plus learned row/column positional embeddings feed into four TransformerEncoder
layers (4 heads). For edge prediction, 7-dimensional spatial pair features (relative row, relative column, Manhattan distance, Chebyshev distance,
same-row flag, same-column flag, log-area ratio) are computed for all candidate pairs within a configurable grid radius (default 50). A typed
pairwise MLP scores each candidate edge, replacing the bilinear heads used by AdjTransformer. An optional 2-layer StructureRefinementGNN
post-processes the predicted graph before the classification head. The 7-d spatial pair features, shown as a dedicated input to the MLP edge
scorer in the diagram, make this architecture the most spatially explicit among graph learners.
857-d input
Input projection
+ row/col pos. embed
TransformerEncoder×4
4 heads, spatial-bias attnTyped MLP edge scorer
per edge type7-d spatial pair features
Δrow,Δcol, Manhattan,
Chebyshev, flags, log-area
StructureRefinementGNN
(optional, 2 layers)
Node head
2-layer MLP→13 logitsEdge heads×2
hierarchy + table bdylearned adjacency
Figure 11: SpatialEdgeTransformer (3.62 M params). Shares the Transformer backbone with AdjTransformer but replaces
bilinear adjacency heads with a typed MLP edge scorer that consumes 7-d spatial pair features (relative position, distances,
flags). The bold “learned adjacency” arrow feeds predicted edges into the refinement GNN — no fixed graph is provided.

Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure
B Training hyperparameters
All 90 runs (6 architectures×5 folds×3 seeds) share one grid:
•Cross-validation: 5 folds, split at the sheet level; seeds 42, 2137, 10042010.
•Epochs: 100; optimizer AdamW; cosine learning-rate schedule.
•Sheets capped at 800 nodes during training.
•Loss: focal loss ( 𝛾=2) over the 13 classes with inverse-frequency 𝛼weights. Graph learners add binary cross-entropy on predicted
adjacency with warmup scaling; gold header–value hierarchy edges serve only as auxiliary loss targets and are excluded from the
convolution graph.
•Hidden dimension: 128 (node classifiers), 256 (graph learners); dropout in the input encoder and backbone blocks.
•Reported metric: per-run pooled macro-F1 — the per-sheet 13-class confusion matrices of the run’s test fold are pooled into a single
matrix, per-class F1 is computed over classes with gold support, and averaged.
Figure 12 disaggregates performance by fold: fold 2 is consistently the hardest, and the architecture ranking is stable across folds.
Fold 0 Fold 1 Fold 2 Fold 3 Fold 40.680.700.720.740.760.780.800.820.84T est macro-F1
Fold 0 Fold 1 Fold 2 Fold 3 Fold 40.820.840.860.880.900.920.940.96Best validation accuracy
MLP GCN GAT AdjTransf. DualMod. SpatialET
Figure 12: Per-fold test macro-F1 (left) and best validation accuracy (right), averaged over 3 seeds. Blue = node classifiers, orange
= graph learners.
C Full 13-class frequency table
Label Meaning Share (%)
value Observed data cell (non-formula) 48.99
aggregation Formula-derived (totals, subtotals) 0.91
header Section/sheet titles and captions 0.07
metadata Footnotes, sources, disclaimers 0.33
comment Cell-stored note 0.04
empty Intentional blank for layout 45.44
junk Placeholder text to exclude 0.23
col_header_1 Column header, innermost 0.70
col_header_2 Column header, mid level 0.07
col_header_3 Column header, outer level 0.01
row_header_1 Row header, innermost 2.66
row_header_2 Row header, mid level 0.33
row_header_3 Row header, outer level 0.24
Table 7: Empirical class distribution (∼1M labelled cells).
D Annotation protocol
Every sheet used for training or evaluation was structured by two human labelers working in native Excel, following a published bilingual
labelling guide; the workflow includes peer review and a self-review pass. Annotators could mark a tab “skipped” when they could not
interpret how to read it coherently; skipped tabs ( ≈24%of decided tabs) are excluded from training, evaluation, and RAG indexing. Labels
attach per-sheet table ids (T0 = sheet-wide scope; T1+ = distinct logical tables) and header depth levels for row and column headers. Table
membership is never used as a model input or prediction target; it serves only for building gold-role chunks and the oracle ceiling.

Zofia Smoleń
E Deployed checkpoints (relationship study)
Table 8 lists all sixteen deployed checkpoints used in the role-quality regression (Section 2.4), with their pooled macro-F1 and downstream
results under the identical Row pipeline. MLP, GCN, GAT, AdjTransformer, and DualModalityGNN contribute low/mid/high triples;
SpatialEdgeTransformer contributes a single checkpoint (its 𝑂(𝑁2)edge scorer required a stricter node cap: 138/382 corpus sheets fell back
to naive chunks vs. 33/382 for all other checkpoints, so its scores are partly fallback-driven).
Table 8: The sixteen deployed checkpoints, sorted by pooled macro-F1. Human = mean human rating over 480 questions.
Checkpoint Family macro-F1 Recall@1 Recall@5 Human
MLP-low node 0.496 0.440 0.596 3.45
GCN-low node 0.516 0.452 0.619 3.57
GAT-low node 0.518 0.502 0.640 3.66
AdjTr-low graph 0.589 0.421 0.598 3.66
MLP-mid node 0.612 0.475 0.642 3.64
Dual-low graph 0.625 0.452 0.602 3.74
GCN-mid node 0.648 0.442 0.608 3.71
GAT-mid node 0.660 0.442 0.617 3.79
AdjTr-mid graph 0.702 0.458 0.627 3.75
Dual-mid graph 0.712 0.465 0.629 3.81
MLP-high node 0.743 0.467 0.613 3.52
GCN-high node 0.781 0.450 0.610 3.71
Dual-high graph 0.787 0.492 0.623 3.82
GAT-high node 0.810 0.465 0.640 3.88
SpatialET graph 0.816 0.433 0.585 3.62
AdjTr-high graph 0.818 0.458 0.625 3.75
F Prompts
F.1 Answer generation prompt
The generator (gemini-2.5-flash, thinking disabled, temperature 0) receives the top five retrieved chunks joined with—separators:
You are answering questions about data in spreadsheets.
Given the following spreadsheet excerpts, answer the question.
Include the specific values and numbers from the data that
support your answer.
If you cannot find the answer in the provided excerpts, say
"NOT FOUND".
Spreadsheet excerpts:
{chunks}
Question: {question}
Answer:
F.2 Judge prompt (role-ablation panel)
The role ablation (Table 4) is the one evaluation scored by a panel of five open-weight LLM judges instead of the human judge, for three
reasons. First, scale: the ablation adds7 ×480=3,360generated answers for a single auxiliary table — an extra rating round of about a fifth of
the entire human-rated volume, disproportionate to the weight of the result. Second, the analysis is purelyrelative: what matters is the
difference between ablation arms on identical questions and identical gold chunks, so a consistent, repeatable judge suffices and no absolute
calibration is needed — and the ablation baseline (unmodified gold chunks) is scored by the same panel, so every reported judge loss is a
panel-minus-panel difference, free of any human-vs-panel offset. Third, the substitution is validated: on answer sets carrying both panel and
human ratings, the panel agrees with the human judge within one point on ≥96%of questions and runs ∼0.1points lower on average. Each
judge receives the same rubric that the human judge uses:
You are evaluating whether a generated answer correctly answers
the same question as an expected answer.
Both answers refer to data extracted from a spreadsheet.

Q&A on Any Spreadsheet Requires Interpreting Its Grid Structure
Expected answer: {expected}
Generated answer: {generated}
Rate the generated answer on a scale of 1-5:
1 = Completely wrong, contradicts the expected answer, or
unrelated
2 = Partially related but key facts are wrong or misleading
3 = Answers a different aspect of the question, or gives only
tangentially related information
4 = Correctly answers the core question but with less detail or
supporting data than the expected answer
5 = Correctly answers the question with equivalent or sufficient
detail
Important: if the question is yes/no or asks "which one", and the
generated answer gives the correct yes/no/choice, that is at
least a 4 even if it omits supporting numbers. The core answer
matters most.
Respond with ONLY a single integer (1-5).
G Human evaluation protocol
Every generated answer in every reported human column — ours, every baseline’s, every geometry variant’s, and the oracle’s — was rated on
the same 1–5 rubric shown above by a human judge, blind to which method produced the chunks. The only exception is the seven-arm role
ablation (Table 4), which is scored by a panel of five versioned open-weight LLM judges (DeepSeek-v4-flash, Llama-3.3-70B, Qwen3.7-flash,
Gemma-3-27B, Mistral-Small-24B) using the identical rubric; on answer sets carrying both panel and human ratings, the panel agrees with
the human judge within one point on ≥96%of questions and runs ∼0.1points lower on average, so the substitution is safe for this auxiliary
analysis.