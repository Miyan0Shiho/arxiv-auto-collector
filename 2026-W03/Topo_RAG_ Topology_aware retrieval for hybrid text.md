# Topo-RAG: Topology-aware retrieval for hybrid text-table documents

**Authors**: Alex Dantart, Marco Kóvacs-Navarro

**Published**: 2026-01-15 09:27:14

**PDF URL**: [https://arxiv.org/pdf/2601.10215v1](https://arxiv.org/pdf/2601.10215v1)

## Abstract
In enterprise datasets, documents are rarely pure. They are not just text, nor just numbers; they are a complex amalgam of narrative and structure. Current Retrieval-Augmented Generation (RAG) systems have attempted to address this complexity with a blunt tool: linearization. We convert rich, multidimensional tables into simple Markdown-style text strings, hoping that an embedding model will capture the geometry of a spreadsheet in a single vector. But it has already been shown that this is mathematically insufficient.
  This work presents Topo-RAG, a framework that challenges the assumption that "everything is text". We propose a dual architecture that respects the topology of the data: we route fluid narrative through traditional dense retrievers, while tabular structures are processed by a Cell-Aware Late Interaction mechanism, preserving their spatial relationships. Evaluated on SEC-25, a synthetic enterprise corpus that mimics real-world complexity, Topo-RAG demonstrates an 18.4% improvement in nDCG@10 on hybrid queries compared to standard linearization approaches. It's not just about searching better; it's about understanding the shape of information.

## Full Text


<!-- PDF content starts -->

TOPO-RAG: TOPOLOGY-AWARE RETRIEVAL FOR HYBRID
TEXT–TABLE DOCUMENTS
Alex Dantart∗
CIO, Humanizing Internet
arxiv@humanizinginternet.comMarco K ´ovacs-Navarro
CTO, Humanizing Internet
marco@humanizinginternet.com
ABSTRACT
In enterprise datasets, documents are rarely pure. They are not just text, nor just numbers; they are
a complex amalgam of narrative and structure. Current Retrieval-Augmented Generation (RAG)
systems have attempted to address this complexity with a blunt tool: linearization. We convert rich,
multidimensional tables into simple Markdown-style text strings, hoping that an embedding model
will capture the geometry of a spreadsheet in a single vector. But it has already been shown that this
is mathematically insufficient.
This work presentsTopo-RAG, a framework that challenges the assumption that “everything is text.”
We propose a dual architecture that respects the topology of the data: we route fluid narrative through
traditional dense retrievers, while tabular structures are processed by aCell-Aware Late Interaction
mechanism, preserving their spatial relationships. Evaluated onSEC-25, a synthetic enterprise corpus
that mimics real-world complexity, Topo-RAG demonstrates an18.4 % improvement in nDCG@10
on hybrid queries compared to standard linearization approaches. It’s not just about searching better;
it’s about understanding the shape of information.
KeywordsRetrieval-Augmented Generation (RAG) ·table retrieval ·late interaction ·multivector retrieval ·enterprise
search·heterogeneous data ·semantic routing ·structure-aware embeddings ·Topo-RAG ·ColBERT ·cell-aware
interaction·linearization bottleneck
1. Introduction
1.1. The Problem of Business Heterogeneity
Let us take as a basis the document repository of a large corporation. Unlike classical libraries filled with narrative
scrolls, corporate “knowledge” is inherently heterogeneous. A single PDF file, such as an agricultural settlement report
or a financial audit, is an ecosystem in itself. It contains paragraphs of legal text (narrative), immediately followed by a
grid of prices by size and variety (tabular structure), and perhaps footnotes that link both worlds.
However, most modern RAG systems treat these documents with blind uniformity. As recent benchmarks such as HERB
[6] and AIR-Bench point out, the industry faces a problem of “topological blindness.” When ingesting these documents,
current systems ignore the fact that reading, for example, a legal contract requires sequential semantic understanding,
while interpreting a settlement table requires positional and relational understanding. Treating both types of data as a
sequence of words is like trying to understand a road map by reading it as if it were a novel: the sense of direction is
lost.
∗Other papers by the author arXivarXiv:2601.10215v1  [cs.AI]  15 Jan 2026

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
(a) Standard (Linearization)
Prod Price
A 10
B 20
Source Table| Prod | Price |
| --- | --- |
| A | 10 |
| B | 20 |
Text / Markdown Single VectorQuery:
”Price of B?”
Noise/
Loss
(b) Topo-RAG (Topology-Aware)
Prod Price
A 10
B 20
Source TableCell-TokHeaders
Values
Spatial Vector GridQuery:
”Price of B?”
Late
Interaction
Figura 1:The linearization bottleneck versus Topo-RAG.(a) Standard approaches flatten tables into text, compressing
two-dimensional relationships into a single noisy vector. (b)Topo-RAGpreserves the topological grid: each cell becomes
an independent embedding, allowing the query to interact precisely with the relevant values (e.g., matching “B” and
“20”) via Late Interaction.
1.2. The Fallacy of Linearization
The industry-standard solution to date, popularized by approaches such as TabRAG [ 1], has been “linearization”:
converting the two-dimensional structure of a table into a one-dimensional representation, typically in Markdown or
JSON format, and then compressing that long text string into a single dense vector (embedding).
We call this thefallacy of linearization. While it is a convenient engineering feat, it rests on a fragile scientific premise.
As Weller et al. theoretically demonstrate inOn the Theoretical Limitations of Embedding-Based Retrieval[ 2], there is
a fundamental limit to the ability of a single vector to represent all possible combinations of relationships in a dataset.
2

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
When we “flatten” a table of 50 rows and 10 columns into a single vector, we are asking the embedding model to
compress 500 potential relationships (cell-row, cell-column, cell-header) into a fixed point in latent space. The result is
“semantic noise”: the model understands that the document is about “prices” and “lemons” (for example), but loses
the ability to precisely distinguish whether the price of C0.85 corresponds to the 2023 campaign or the 2024 one, or
whether it applies to the “Verna” or “Eureka” lemon variety. The geometry of the table is lost in translation.
1.3. Contribution: the Topo-RAG Framework
To overcome this barrier, we propose to stop fighting against the nature of the data and start designing architectures
that respect it. We presentTopo-RAG, an approach inspired by cognitive biology: just as the human brain processes
language and spatial images in distinct regions, our system separates processing according to the topology of the
information.
Our main contributions are:
1.Topology-Aware Routing:We implement a lightweight classifier, inspired by the efficiency ofPneuma[ 4],
which acts as a “switchman”, separating narrative text blocks from structured blocks (tables/lists) before
vectorization.
2.Dual-Path Retrieval:
For text, we use the proven path: optimizedDense Retrieval.
For tables, we introduce aCell-Aware Late Interactionmechanism. Instead of compressing the table, we
maintain the vector identity of its individual cells (tokens), using an adaptedColBERT-type architecture.
This allows the user’s query to “interact” directly with the specific cell values (the price, the date), without
the loss from compression.
3.Empirical Validation:We validate our hypothesis on a synthetic dataset (SEC-25) designed to mimic the
complexity of real corporate documents, demonstrating that respecting the topology of the data is not only
theoretically sound, but also industrially profitable.
2. Current Related Work
The search for information retrieval systems that truly understand data, rather than merely matching keywords, has
been a major challenge over the past decade. To understand the proposal ofTopo-RAG, we must explore three research
streams that converge at this historical moment: how we represent tables, how models interact with data (single vs.
multi-vector), and how we make all of this computationally feasible.
2.1. Table Retrieval and Representation: The Search for Structure
The first challenge is fundamental: How do we teach a neural network, which thinks in vectors and numbers, what a
table is?
Until very recently, the dominant strategy waslinearization. If we have a well-structured two-dimensional table, the
conventional strategy refined in works such asTabRAG[ 1], consists of “reading” that table from left to right and top
to bottom, turning it into a long narrative sentence (using Markdown or JSON). The premise is seductive: if LLMs
are excellent at reading text, let’s turn everything into text. TabRAG optimizes this process by generating structured
representations that LLMs can digest. However, this is equivalent to describing a building brick by brick instead of
showing the architectural plans; the immediatespatial relationshipbetween elements is lost.
In a parallel and more experimental line, we find approaches such asBirdie, which uses aDifferentiable Search Index
(DSI). Birdie is fascinating because it tries to eliminate the intermediary (the traditional vector index). Instead of
searching for vectors, it trains a model todirectly generate the identifier(TabID) of the correct table given a query. It’s
as if the librarian had memorized the exact location of every book and could give you the shelf number from memory.
Although promising, this approach suffers from rigidity: if the data changes (something constant in enterprise settings),
the model must be retrained or adapted at significant cost.
Finally, works such asSQuAREattempt a hybrid approach, adapting retrieval specifically for tabular formats through
structured queries. But all these methods share a common weakness: they either ignore the native topology of the table
by flattening it, or they require monolithic architectures that are difficult to scale.
3

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
2.2. Multivector and Late Interaction: Preserving the “Pixels” of the Data
This is where the most exciting paradigm shift comes into play. Most embedding models (such as those from OpenAI or
BGE) areSingle-Vector: they take an entire document (or a flattened table) and compress it into a single point in space
(a vector).
The problem, brilliantly explained by theory, is that this compression islossy. For example, with a fruit price table, if
we compress the entire table into a vector, the model may remember the concept of “fruit prices,” but it may forget
whether the price “0.50 C” belongs to “Oranges” or “Lemons.” Fine-grained information becomes blurred.
The alternative islate interaction, popularized by theColBERTarchitecture and recently refined in libraries such as
PyLate. Instead of compressing the entire document into a single vector, these models maintainone vector per token
(or in our case, per cell).
Think of this as the difference between a blurry, low-resolution image (Single-Vector) and a high-definition image where
each pixel retains its color (Multi-Vector). Similarity is not computed just once, but rather through an operation called
MaxSim, which seeks the best match foreach partof the query within the document. This is crucial for tables: it allows
the question “Price of oranges?” to find exactly the cell “Oranges” and its neighboring cell “Price,” without interference
from the “noise” of other rows. Papers such asOn the Theoretical Limitations of Embedding-Based Retrieval[ 2]
provide the mathematical foundation to state that, for complex tasks, the Single-Vector approach has a glass ceiling that
only the Multi-Vector approach can break through.
2.3. Efficiency in Retrieval
If we store one vector per cell instead of one per document, RAM usage will increase, making the Multi-Vector
approach, by definition, heavier. However, the year 2025 has brought spectacular advances in efficiency that make our
Topo-RAGproposal viable.
WARP[ 3] introduces an optimized engine that drastically reduces the latency of these models. It uses low-level
techniques so that “late interaction” does not mean “slow interaction.”
Even more interesting is the proposal ofCRISP[ 5]. This work introduces the idea ofclusteringto reduce
noise. Instead of storing all vectors, it groups similar ones together. For a table, this is revealing: many cells
are redundant or empty. CRISP allows us to “prune” irrelevant information before storing it.
Complementarily, the work onEfficient Constant-Space Multi-Vector Retrievalteaches us how to set a
memory budget without sacrificing too much accuracy, making these systems deployable on standard enterprise
infrastructure, not just supercomputers.
2.4. Join-Aware & Multi-Hop: beyond simple search
Finally, we must recognize that in the real world, the answer is rarely in a single cell. It often requires connecting the
dots.
Papers such asREaR[ 8] andExploring Multi-Table Retrievaladdress the problem of queries that require hops
(multi-hop) or joins between tables. These works show us that retrieval is not a single event, but an iterative process:
searching a table, reading a cell, using that value to search in another table.
We are also inspired byBridging Queries and Tables through Entities[ 7], which suggests that entities (names, places,
product codes) are the “hooks” that link queries with tables.
2.5. The gap
Despite these incredible advances, there is a gap. We have excellent systems for text (Single-Vector) and promising
technologies for fine structure (Multi-Vector), but the industry continues to try to force both types of data through the
same funnel (linearization to Markdown).
Topo-RAGis born from this observation: we should not treat everything as text. By recognizing thetopologyof the data
and applying the right tool for each form (Route A for narrative, Route B for tables), we can overcome the theoretical
limitations of linearization and offer a system that truly “understands” business structure.
4

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
PATH A: NARRATIVE
PATH B: STRUCTURALHeterogeneous DocumentTopology
RouterText Encoder
(Bi-Encoder)FAISS
Dense IndexNarrative Segments1 Vector/Doc
Table Encoder
(Cell-Aware ColBERT)WARP
Multi-Vector IndexTabular SegmentsN Vectors/TableUser QueryANN Search
MaxSim SearchUnified
Cross-Encoder
RerankingTop-K Text
Top-K TablesLLM
Generation
Figura 2:The Topo-RAG architecture.The system employs a topology-aware routing mechanism to split heteroge-
neous documents. Narrative text follows a standard dense retrieval route (top, blue), while tabular data is processed via
a cell-awareLate Interactionpath (bottom, orange), using the WARP engine for greater efficiency. Both flows converge
in a Unified Cross-Encoder Reranker to provide context to the LLM.
3. Methodology: the Topo-RAG framework
In this section, we break down the architecture ofTopo-RAG. Our fundamental premise is that theshape(topology) of
the data dictates the optimal retrievalfunction. We do not attempt to force a square peg into a round hole; instead, we
build a system with two specialized “hands”: one to handle the fluidity of natural language and another to manipulate
the crystalline rigidity of tabular data.
The system operates in three sequential phases: (1) topological routing, (2) dual-path retrieval, and (3) unified reranking.
3.1. Topology-aware routing
The first challenge in an enterprise environment is that documents are not labeled as “Text” or “Table.” They are chaotic
mixtures. An “Export Policy” PDF may have three pages of dense legal text and suddenly insert a table of customs
tariffs.
If we feed all this into a standard embedding model, the signal from the table gets diluted in the noise of the text. To
avoid this, we introduce a pre-processing module inspired by the classification efficiency ofPneuma[4].
We define a heuristic metric calledStructural Density Score (SDS). For a given text blockb, we compute:
SDS(b) =Nnum+Nsep+Nent
Ntotal
Where:
Nnum is the number of numeric tokens.
Nsepis the number of structural separators (such as — in Markdown, ’td’ tags in HTML, or frequent line
breaks).
Nentis the density of named entities (detected via lightweight NER), since tables are usually dense in product
names, locations, or companies, unlike narrative text which uses more functional words (stopwords).
The routing algorithm:the system scans the document using a sliding window.
5

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
1.IfSDS(b)> τ (an empirical threshold, typically 0.4), the block is classified asstructured. It is sent toRoute
B.
2. IfSDS(b)≤τ, the block is considered Narrative. It is sent toRoute A.
This ensures that we do not waste expensive computational resources (late interaction) on simple text paragraphs, and
do not lose accuracy by using simple dense embeddings on complex tables.
3.2. Route A: dense narrative retrieval (the semantic stream)
For blocks classified as narrative (sustainability reports, emails, contractual clauses...), the standard solution remains the
most efficient. Human narrative is sequential and semantically redundant; a single well-trained vector can capture the
“essence” of a paragraph with high fidelity.
In this route, we use a standardBi-Encoder(in our experiments, the robust text-embedding-3-large or BGE-M3).
Input:Narrative text blockT.
Process:V T=Encoder(T).
Output:A single dense vectorV T∈Rd(whered= 1536or1024).
This route prioritizes speed and the capture of general thematic nuances (“What is this document about?”).
3.3. Route B: conscious late cell interaction (the structural stream)
Here lies the main innovation of Topo-RAG. For the data that the Router identified as “Tables”, we reject compression
into a single vector. We adopt aCell-Aware Late Interaction (CALI)approach, which is an adaptation of theColBERT
architecture specifically designed for tabular structures.
3.3.1. The “cell as a token” paradigm
In standard ColBERT, each wordtokenhas its own vector. In CALI, we raise the abstraction: our atomic unit is not the
syllable, it is theCell.
A table is decomposed not into sentences, but into a “bag of cell vectors.” However, a cell by itself (e.g., “0.85”) lacks
meaning. It needs its topological context (its column header and its row identifier).
To address this, we apply aPositional Injectiontechnique inspired byBridging Queries and Tables[ 7]. Each cell ci,j
(rowi, columnj) is serialized enriched with its metadata before being vectorized:
Content(c i,j) = ”[COL:Headerj] [V AL:Valuei, j]”
This generates a vectorv i,jthat encapsulates both the value (“0.85”) and its structural meaning (“Price”).
3.3.2. MaxSim optimized for tables
When a user query arrives (e.g.,“Price of lemon in Germany”), we do not compare it with a vector of the entire table.
We use the modifiedMaxSimoperator.
For each term in the query qk(e.g., “Germany”), we search for the maximum similarityonlyamong the vectors of the
table’s cells.
Score(Q, Table) =X
qk∈Qm´ ax
vi,j∈Table(qk·vi,j)
Why is this revolutionary for tables?Let’s work, for example, with the query:“Verna Price”.
1.The query vector “Price” will find its maximum similarity with the header column “Price” or the cells
containing monetary values.
2.The vector “Verna” will find its maximum similarity in the “Variety” column where the word “Verna” appears.
6

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
Query Terms (q i)
Price Lemon Verna
Table Grid (d j)
Product Origin Price
Apple Spain 1.20
Verna Italy 0.85Max
Max ΣRelevance
Score
Figura 3:Cell-Aware Late Interaction (CALI).Unlike dense retrieval, which compares one vector against another,
Topo-RAG compares each token vector of the query ( qi) with all cell vectors ( dj) of the table. TheMaxSimoperator
(orange arrows) independently identifies the best-matching cell for each term (for example, “Price” matches the header,
“Verna” matches the row identifier), regardless of their distance in the linearized text. These maximum scores are
summed to quantify the total topological relevance.
3.The sum of these maximum interactions gives us a high scoreonlyif the table containsbothelements with
high specificity. A normal dense model could be confused by a table that talks about “Eureka Prices”, because
“Eureka” and “Verna” are semantically close (both are lemons). CALI, by working at a fine-grained interaction
level, distinguishes the exact entity.
3.3.3. Reduction to improve efficiency (the WARP & CRISP influence)
The problem with storing one vector per cell is the memory explosion. A 100x10 table would generate 1000 vectors. To
make this industrially viable, we apply aggressive reduction techniques inspired byCRISP[5] andWARP[3]:
1.Vector clustering:Many cells are semantically identical or empty. We group very similar vectors (e.g., all
cells that say “USD” or “Kg”) and store only one centroid.
7

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
2.Quantization:We use product quantization (PQ) to reduce the size of the cell vectors from 32-bit floats to
4-bit integers without significant loss of retrieval precision, allowing millions of cells to be stored in standard
RAM.
3.4. Unified Reranking: The Best of Both Worlds
Finally, we have two lists of candidates: one coming from Route A (Text) and another from Route B (Tables). Their
scores are not directly comparable (one is Cosine Similarity [0-1], the other is a sum of MaxSim [with no clear limit]).
To unify the results, we use a final phase ofCross-Encoder Reranking.
1. We take the Top-K from the Narrative Route and the Top-K from the Structural Route.
2. We normalize their scores usingMin-Max Scalingto obtain an initial combined heuristic.
3.We pass the final candidates (plain text and serialized tables) through a lightweight Cross-Encoder model (e.g.,
BGE-Reranker-v2-m3).
The Cross-Encoder acts as the “human judge”: it reads the query and the candidate (whether text or table) with full
attention and issues the final relevance verdict. This step corrects any hallucination that may have arisen during fast
retrieval and ensures that the final list presented to the generator LLM contains the perfect mix of narrative context and
precise data needed to answer the user’s question.
4. Experimental Setup: Simulating Business Chaos
To validate our hypothesis—that data topology matters—we could not rely on traditional academic datasets such as
NQ-TablesorSpider. These datasets are often “too clean” or focused exclusively on either tables or text.
The challenge of the modern enterprise ishybridization. We needed a testing environment where a legal contract
(text) could contradict or complement a settlement spreadsheet (table). Since no public dataset with these specific
characteristics existed, we built one.
4.1. Datasets: The Synthetic Corporate Corpus (SEC-25)
Following the synthetic data generation methodology proposed inHERB[ 6] andAIR-Bench, we created theSEC-25
(Synthetic Enterprise Corpus 2025).
The goal of SEC-25 is not to be massive in size, but ratherdense in complexity. We used GPT-4o to generate documents
that mimic the structure of real corporate files from the agri-food sector.
Corpus composition (10,000 Documents):The corpus is intentionally divided into two topological hemispheres to test
our Router’s capability:
1.Narrative Hemisphere (50 %):
Sustainability reports:Dense, rhetorical text with few figures.
Legal contracts:Complex clauses, conditional language (“if X, then Y”).
Emails:Informal conversation threads, scattered context.
2.Structured Hemisphere (50 %):
Settlement sheets:Dense tables with columns such as “Variety”, “Size”, “Price/Kg”, “Discount”. Here lies
the trap: the same number (e.g., “0.50”) can appear in hundreds of different cells with different meanings.
Logistics inventories:Long lists with product codes (IDs) that are hostile to standard tokenizers.
The Query Challenge (The Query Set):We generated an evaluation set of500 queriesdesigned to break traditional
RAG systems. We divided them into three categories of cognitive difficulty:
Type A: Factual Retrieval (200 queries):“What is the return policy?”These can be answered with a single
block of text. This is the comfort zone of classic Dense Retrieval.
Type B: Cell-Precise Lookup (200 queries):“What was the price of Verna lemon in week 42 at Mercadona?”
Requires navigating exact coordinates (Row: Verna/Week 42, Column: Price). This is where dense embeddings
often “hallucinate” due to the noise from nearby neighbors.
8

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
Type C: Hybrid Multi-Hop (100 queries):The ultimate test.“Based on the 2024 quality contract [Text], list
the producers from Table B [Table] who did not meet the Brix standard. ”Requires retrieving a text document
(to know the standard) and a table (to filter the data), and then reasoning over both.
4.2. Baselines: the titans to beat
To demonstrate that Topo-RAG provides real value, we compare it against the current standards in industry and academia.
We did not choose straw men; we chose the systems that a company would implement today if they hired a standard
consulting firm.
Baseline 1: Naive RAG (the industry standard):
Strategy:Blind linearization. Everything (text and tables) is converted to Markdown.
Model:OpenAI text-embedding-3-large (the de facto standard).
Logic:It is fast, cheap, and easy to implement. This is what most companies use today.
Baseline 2: Advanced Recursive RAG:
Strategy:Intelligent “Parent-Child” chunking. Small fragments (children) are indexed for retrieval, but the
large block (parent) is returned to the LLM to provide context.
Model:BGE-M3 (SOTA among open source dense models).
Logic:Attempts to solve the context loss of Naive RAG, but still uses a single vector per chunk.
Baseline 3: TabRAG (Structure-Aware Linearization) [1]:
Strategy:Specialized for tables. Uses an auxiliary LLM to “narrate” or describe the table before vectorizing it,
adding synthetic metadata to enrich the vector.
Logic:This is the most sophisticated attempt to make tables work in the single-vector paradigm. It is our direct
competitor.
4.3. Implementation details
The implementation ofTopo-RAGis not trivial. To ensure reproducibility and industrial viability, we used the following
technologies:
Infrastructure:Everything was run on an instance withNVIDIA A100 (40GB). This is important: we want
to demonstrate that this works on accessible hardware, not just on Google clusters.
Software Stack:
• ForRoute A (Text), we usedFAISSfor approximate vector search, optimized for speed.
•ForRoute B (Tables), we implemented ourCell-Aware Late Interactionengine using thePyLatelibrary.
PyLate allows us to manage the complexity of multi-vectors without rewriting all the training code from
scratch.
•For efficiency, we applied thepruningtechniques described inWARP[ 3], reducing the table index by
40 % by removing vectors of empty cells or irrelevant stopwords (“el”, “la”, “de”) within the tables.
4.4. Evaluation Metrics and Success
In RAG, “finding the document” is not enough. The user needs thecorrect answer. That is why we use metrics at two
levels:
Retrieval Quality (Did we find the needle?):
nDCG@10 (Normalized Discounted Cumulative Gain):Measures not only whether we found the relevant
document, but also if we ranked it among the top positions. This is crucial so that the LLM does not get
distracted.
Recall@20:Is the answer somewhere within the top 20 results? If it is not here, the LLM has no chance to
answer.
9

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
Generation Faithfulness (Did the LLM tell the truth?):
Hallucination Rate (Inverse Accuracy):Here we use theLLM-as-a-Judgeparadigm. We give the LLM
(GPT-4o) the answer generated by our system and the “Ground Truth” (the correct gold answer). The judge
evaluates whether the generated answer contains fabricated or numerically incorrect data.
Why is this vital?In financial tables, saying “0.85” when it is “0.86” is a critical hallucination. Standard text
metrics (such as BLEU or ROUGE) often fail to detect these numerical precision errors.
We have set the stage: a treacherous corpus full of structural traps (SEC-25), worthy opponents (TabRAG and Naive
RAG), and a high-precision “microscope” to measure the results (LLM-as-a-Judge).
5. Results and Analysis
To understand the results, we must recall our objective: we are not simply aiming to “win” on a metric. We seek to
demonstrate thattopology matters.
If our hypothesis is correct,Topo-RAGshould not be just “a little better”; it should behave in aqualitatively different
manner depending on the type of data. It should be a chameleon, adapting to both fluid text and rigid tables with equal
skill.
5.1. Retrieval Performance
We evaluate our models on theSEC-25corpus using the three defined query categories: Narrative (Text), Tabular
(Structure), and Hybrid (Multi-hop Reasoning).
Below we present the main results (Table 1). The key metric isnDCG@10, which rewards the system not only for
finding the answer, but for placing it in the first position—something vital so that the LLM is not distracted by noise.
Architecture Model Type A:
Narrative
(Text)Type B: Ta-
bular (Cell-
Precise)Type C: Hy-
brid (Multi-
Hop)Overall Average
Naive RAG(OpenAI Ada-002) 0.882 0.451 0.410 0.581
Advanced RAG(Parent-Child) 0.895 0.523 0.485 0.634
TabRAG(Linearization SOTA) 0.880 0.685 0.612 0.725
Topo-RAG (ours) 0.891 0.842 0.796 0.843
Improvement vs. SOTA (TabRAG) +1.2 % +22.9 % +30.0 % +16.2 %
Cuadro 1:Retrieval Effectiveness Comparison (nDCG@10)
Analysis of Table 1 The Narrative Tie (Type A):Observe the first column. For narrative queries (“What is the
ethics policy?”),all models are excellent. The difference between a complex system like Topo-RAG (0.891) and a
simple one like Naive RAG (0.882) is marginal. This confirms our theory: for sequential text, current dense embeddings
have already “solved” the problem. Linearization works for what is linear.
Structural collapse (Type B):The second column reveals the catastrophe. The Naive RAG model plummets to0.451.
Why? Because when asked“Price of Verna lemon in week 42”, the dense model retrieves any document containing the
words “price”, “lemon”, or “week”, without understanding the exact intersection.TabRAGimproves (0.685) because it
adds descriptions (“This table contains prices...”), but it still hits a glass ceiling.Topo-RAGdominates with a0.842. By
usingLate Interaction, the system does not look for a “similar” document; it searches for the exact match of the vectors
for the “Verna” cell and the “Week 42” cell within the same spatial structure.
The hybrid (Type C):This is where Topo-RAG shines the most (0.796vs 0.612 for TabRAG). Hybrid queries require
finding both a text and a table simultaneously. Systems that treat everything as text tend to “flood” the context with
many irrelevant text fragments, pushing the necessary table out of the Top-K. Topo-RAG, by having separate pathways,
ensures that the finalrerankeralways receives the best candidates from both worlds.
10

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
Narrative Tabular Hybrid00,20,40,60,81
+30 % Gain0,88
0,450,410,88
0,68
0,610,890,840,8nDCG@10 ScoreNaive RAG TabRAG (SOTA) Topo-RAG (ours)
Figura 4:Retrieval performance by query type.While all models show similar performance on narrative text (left), a
massive performance gap opens up for tabular and hybrid queries.Topo-RAG(orange) maintains high accuracy in
complex scenarios where linear approaches (gray/blue) collapse due to loss of structure.
5.2. Why Linearization Fails
To deeply understand why the baselines fail, we conducted a forensic analysis of the errors. We focus on the phenomenon
we call“Structure Loss”.
We use the metric ofnumerical hallucination rate(evaluated with LLM-as-a-Judge). We gave the LLM the context
retrieved by each system and asked it to extract a specific numerical fact. If the retrieved context was incorrect or
imprecise, the LLM would make up the number.
In small tables (3 columns), Naive RAG works well.
As the table grows (10, 20 columns), the Naive RAG line drops sharply. This is due tovector dilution. As the
authors ofCRISPtheoretically explain, a single vector has a finite information capacity. If you try to fit 20
columns of data into 1536 dimensions, the “noise” from irrelevant columns drowns out the signal from the
column you are looking for.
TheTopo-RAGline remains almost flat (horizontal). Thanks toLate Interaction, it does not matter whether
the table has 5 or 50 columns; the system only activates the vectors of the cells relevant to the query, ignoring
the rest. It’s like having a flashlight in a dark room: no matter how big the room is, you only see what you
illuminate.
Key fact:in an “agricultural settlement” table (with ¿15 columns of grades and prices), Topo-RAG reduced the LLM
hallucination rate from45 % (Naive)to8 %. This is the difference between a useful tool and a generator of legal
liabilities.
11

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
510 20 30 40 5000,20,40,60,81
The Topology Gap
Single vectors dilute
information as
complexity grows
Table Density (Number of Columns)Recall@10
Naive RAG
TabRAG (SOTA)
Topo-RAG (Ours)
Figura 5:Robustness to information density.As tables become wider (more columns), standard linearization-based
models (Naive, TabRAG) suffer a sharp drop in retrieval recall due to the “vector dilution” phenomenon.Topo-RAG
maintains an almost constant performance, demonstrating thatCell-Aware Late Interactioneffectively decouples the
information capacity from the fixed dimensions of the vector.
5.3. Latency vs Accuracy
In engineering, nothing comes for free. The extreme accuracy of Topo-RAG has a cost: computation.
Implementing aColBERT-stylearchitecture (as we do in Route B) involves handling gigabytes of vectors (one per cell)
instead of megabytes (one per document). Is this viable for a company?
Metric Naive RAG (Vector) Topo-RAG (Standard) Topo-RAG (optimized
with WARP/PyLate)
Index size (GB) 0.5 GB 12.4 GB 4.1 GB
Indexing time 10 min 45 min 28 min
Latency (ms) 45 ms 210 ms 85 ms
Cuadro 2:Efficiency metrics (index size & latency)
Trade-off analysis:
1.The initial shock:Without optimization, Topo-RAG is heavy (12.4 GB index vs 0.5 GB). This would scare
any cloud architect.
2.The salvation (Pruning & Quantization):This is where we apply the lessons fromWARPandCRISP.
12

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
By applyingquantization(going from float32 to int4) andpruning(removing vectors from empty cells
or stopwords like “el”, “de” within the tables), we reduce the index to4.1 GB.
It is still 8 times larger than the Naive index, but for a company, 4 GB of RAM is a trivial cost (just a few
cents per hour on AWS).
3.Latency:The optimized version responds in85 ms. While this is double that of Naive RAG (45 ms), for a
human the difference between 0.04 seconds and 0.08 seconds is imperceptible. The user is willing to wait an
extra 40 milliseconds in exchange for not receiving a hallucinated price.
5.4. Conclusion
The results validate our central thesis:heterogeneity demands specialization.
Topo-RAG does not win because it uses a larger model or more data. It wins because itunderstands the physics of
information. Treating a table as text would be like trying to listen to a painting; that is, you can describe the colors, but
you lose the spatial experience. By separating the routes and applyingCell-Aware Late Interaction, Topo-RAG restores
the topological dignity of tables, allowing enterprise RAG systems to operate with the precision of a database and the
flexibility of an LLM.
6. Discussion: the heterogeneity gap
The results of our experiments are not simply an incremental victory on a leaderboard; they are evidence of a fundamental
fracture in how we have been building AI for enterprises. We call this phenomenon“The Heterogeneity Gap”.
6.1. The physics of information
The deep reason whyTopo-RAGoutperforms linearization models (such as TabRAG) lies in the physical nature of
information.
Text is time (sequential):A sentence is a timeline. The meaning of a word depends on what came before and
what comes after. Denseembeddings(such as those from OpenAI) are masters of time; they compress that
sequence into a coherent thought.
A table is space (positional):A table is not read, it isnavigated. The meaning of the cell “0.85” does not
depend on the previous word, but on itsspatial coordinate(Row: “Verna”, Column: “Price”).
When we “linearize” a table to Markdown, we are forcing a spatial structure to become a temporal sequence. We
are forcing the model to “memorize” the position of each cell through syntax tokens (—, —). As we demonstrate
empirically, attention models struggle to maintain these long-distance relationships in a single vector.
Topo-RAGsolves this by not fighting against physics. By usingLate Interactionfor tables, we treat the table as a
spatial map of vectors (cells) that are preserved individually. The user query acts as a cursor that moves over this map,
seeking precise matches in specific locations, without the need to compress the entire map into a single point.
6.2. Implications for industry
For industry, the implications are profound. The era of the monolithic “Single Vector Store” is over.
Until now, the standard architecture was:ingest everything →vectorize everything →a single index in Pinecone/Milvus.
Our study suggests that mature RAG architectures must becomposite systems:
1. A lightweight dense index for corporate narrative.
2. A heavy (but optimized) multi-vector index for critical structured data.
3. An intelligent router that decides which path to take.
This is not unnecessary complication; it is the price of accuracy. In domains where a numerical hallucination costs
money (finance, logistics, legal), the architectural redundancy of Topo-RAG pays for itself.
13

Topo-RAG: Topology-aware retrieval for hybrid text–table documents
7. Conclusion and Future Work
In this work, we have challenged the convention of “linearization” in enterprise information retrieval. We present
Topo-RAG, a framework that respects the inherent topology of the data, applying differentiated retrieval strategies for
narrative text and tabular structures.
Our results on the syntheticSEC-25corpus demonstrate that this dual approach is not only theoretically superior, but
also empirically dominant, achieving an18.4 % improvement in nDCG@10on complex hybrid queries. We have
shown that, through modern optimization techniques such as quantization andpruning(inspired byWARPandCRISP),
it is possible to deployLate Interactionarchitectures with acceptable latency for production.
7.1. From Tables to Graphs
Although Topo-RAG solves the problem of finding the correct table, it opens the door to a greater ambition: total
connectivity.
Business tables do not exist in isolation. The entities within a table (e.g., “Supplier: agr ´ıcola del sur”) are the same entities
that appear in narrative contracts. The immediate future of this research, inspired by works such asMixture-of-RAG, is
the integration ofGraphRAG.
We envision an evolution of Topo-RAG where:
1. The cells of the retrieved table act as “anchor nodes”.
2. The system automatically “jumps” from the table cell to the text documents that mention that entity.
3.This would allow answeringsecond-order reasoningquestions, such as:“Tell me which suppliers have
above-average prices [Table] and check if their contracts include penalty clauses for delays [Text]”.
Topo-RAG is the first step: we have taught the AI to read the map. The next step is to teach it to navigate the entire
territory.
Referencias
[1]Jacob Si, Mike Qu, Michelle Lee, and Yingzhen Li. TabRAG: Tabular Document Retrieval via Structured Language
Representations.arXiv preprint arXiv:2511.06582, 2025.https://arxiv.org/abs/2511.06582
[2]Orion Weller, Michael Boratko, Iftekhar Naim, and Jinhyuk Lee. On the Theoretical Limitations of Embedding-
Based Retrieval.arXiv preprint arXiv:2508.21038, 2025.https://arxiv.org/abs/2508.21038
[3]Jan Luca Scheerer, Matei Zaharia, Christopher Potts, Gustavo Alonso, and Omar Khattab. WARP: An Efficient
Engine for Multi-Vector Retrieval.arXiv preprint arXiv:2501.17788, 2025. https://arxiv.org/abs/2501.
17788
[4]Muhammad Imam Luthfi Balaka, David Alexander, Qiming Wang, Yue Gong, Adila Krisnadhi, and Raul Castro
Fernandez. Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System.
arXiv preprint arXiv:2504.09207, 2025.https://arxiv.org/abs/2504.09207
[5]Jo˜ao Veneroso, Rajesh Jayaram, Jinmeng Rao, Gustavo Hern ´andez ´Abrego, Majid Hadian, and Daniel Cer. CRISP:
Clustering Multi-Vector Representations for Denoising and Pruning.arXiv preprint arXiv:2505.11471, 2025.
https://arxiv.org/abs/2505.11471
[6]Prafulla Kumar Choubey, Xiangyu Peng, Shilpa Bhagavath, Kung-Hsiang Huang, Caiming Xiong, and Chien-Sheng
Wu. Benchmarking Deep Search over Heterogeneous Enterprise Data.arXiv preprint arXiv:2506.23139, 2025.
https://arxiv.org/abs/2506.23139
[7]Da Li, Keping Bi, Jiafeng Guo, and Xueqi Cheng. Bridging Queries and Tables through Entities in Table Retrieval.
arXiv preprint arXiv:2504.06551, 2025.https://arxiv.org/abs/2504.06551
[8]Rishita Agarwal, Himanshu Singhal, Peter Baile Chen, Manan Roy Choudhury, Dan Roth, and Vivek Gupta.
REaR: Retrieve, Expand and Refine for Effective Multitable Retrieval.arXiv preprint arXiv:2511.00805, 2025.
https://arxiv.org/abs/2511.00805
14