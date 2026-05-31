# Query Symbolically or Retrieve Semantically? A Dataset and Method for Semi-Structured Question Answering

**Authors**: Mateusz Czyżnikiewicz, Ryszard Tuora, Adam Kozakiewicz, Tomasz Ziętkiewicz, Mateusz Galiński, Michał Godziszewski, Michał Karpowicz, Timothy Hospedales, Cristina Cornelio

**Published**: 2026-05-26 15:22:42

**PDF URL**: [https://arxiv.org/pdf/2605.27164v1](https://arxiv.org/pdf/2605.27164v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems for question answering typically retrieve evidence by semantic similarity between the query and document chunks. While effective for unstructured text, this approach is less reliable on semi-structured corpora where answering may require exact filtering, aggregation, or exhaustive retrieval over structured attributes across multiple documents. Symbolic approaches support such operations, but they are often brittle on noisy natural-language corpora. We address this gap with DualGraph, a RAG framework that represents documents through two complementary views: a Textual Knowledge Graph for semantic retrieval and a Symbolic Knowledge Graph for symbolic querying over typed subject--predicate--object triples. Building on these two components, we provide multiple strategies for selecting or combining semantic and symbolic evidence.We also introduce SpecsQA, a benchmark from a commercial shopping website with semi-structured product documents and manually curated questions spanning open-ended and specification-oriented retrieval. Experiments show that DualGraph consistently outperforms state-of-the-art dense-retrieval, GraphRAG, symbolic, and table-oriented baselines across question types.Code and data are available at https://github.com/corneliocristina/DualGraphRAG.

## Full Text


<!-- PDF content starts -->

Query Symbolically or Retrieve Semantically?
A Dataset and Method for Semi-Structured Question Answering
Mateusz Czy ˙znikiewicz1, Ryszard Tuora1, Adam Kozakiewicz1,
Tomasz Zi˛ etkiewicz1,Mateusz Gali ´nski1,Michał T. Godziszewski1,
Michał Karpowicz1,Timothy Hospedales2,Cristina Cornelio2
1Samsung AI Warsaw,2Samsung AI Cambridge,
Correspondence:c.cornelio@samsung.com
Abstract
Retrieval-Augmented Generation (RAG) sys-
tems for question answering typically retrieve
evidence by semantic similarity between the
query and document chunks. While effective
for unstructured text, this approach is less re-
liable on semi-structured corpora where an-
swering may require exact filtering, aggrega-
tion, or exhaustive retrieval over structured at-
tributes across multiple documents. Symbolic
approaches support such operations, but they
are often brittle on noisy natural-language cor-
pora. We address this gap withDualGraph,
a RAG framework that represents documents
through two complementary views: aTex-
tual Knowledge Graphfor semantic retrieval
and aSymbolic Knowledge Graphfor sym-
bolic querying over typed subject–predicate–
object triples. Building on these two com-
ponents, we provide multiple strategies for
selecting or combining semantic and sym-
bolic evidence. We also introduceSpecsQA,
a benchmark from a commercial shopping
website with semi-structured product docu-
ments and manually curated questions span-
ning open-ended and specification-oriented re-
trieval. Experiments show that DualGraph
consistently outperforms state-of-the-art dense-
retrieval, GraphRAG, symbolic, and table-
oriented baselines across question types. Code
and data are available at https://github.
com/corneliocristina/DualGraphRAG.
1 Introduction
Retrieval-Augmented Generation (RAG) has be-
come a standard approach for question answering
by grounding language models on external evi-
dence retrieved from large document collections.
Most existing RAG systems (Lewis et al., 2020)
retrieve evidence according to semantic similar-
ity between the query and document chunks in
an embedding space. This works well when rele-
vant information is expressed in localized natural-
language (NL) passages, but becomes less reliablefor semi-structured corpora that combine free-form
text with structured specifications, tables, or at-
tributes distributed across multiple documents.
Many real-world questions over such corpora
require operations beyond what semantic similarity
alone can provide. Queries such asWhich Sam-
sung phones support both wireless charging and
eSIM?orList all TVs with HDMI 2.1 and refresh
rates above 120Hzrequire exact filtering, aggrega-
tion, or exhaustive listing over structured attributes.
While recent graph-based RAG systems (Gutiérrez
et al., 2025; Zhuang et al., 2025) improve multi-
hop and cross-document retrieval, they still lack
formal guarantees for precise operations such as
exhaustive enumeration or arithmetic aggregation.
Symbolic approaches (Chepurova et al., 2025) can
support exact querying, but they are often not ro-
bust to noisy natural-language questions and het-
erogeneous document structures.
This gap appears in many real-world settings,
including enterprise documentation, local-file as-
sistants, and customer-support systems over prod-
uct catalogs or manuals, where users ask both
open-ended questions and precise specification-
oriented queries. To address this limitation, we
introduceDualGraph, a retrieval framework for
question answering over semi-structured corpora.
DualGraph represents the same documents collec-
tion through two complementary graph views: a
Textual Knowledge Graph(TKG), optimized for
semantic retrieval over natural-language descrip-
tions, and aSymbolic Knowledge Graph(SKG),
designed for structured querying over normalized
subject–predicate–object triples. Our framework
supports multiple retrieval strategies that combine
semantic and symbolic evidence in different ways,
including direct retrieval over either graph, fallback
mechanisms, concatenation of retrieved contexts,
and query-dependent routing.
We also introduceSpecsQA, a benchmark built
from a time-specific snapshot of the Samsung UK
1arXiv:2605.27164v1  [cs.AI]  26 May 2026

website. The benchmark contains semi-structured
product documents combining natural-language de-
scriptions with specification tables. Existing QA
benchmarks (Yang et al., 2018; Trivedi et al., 2022)
rarely combine structured and unstructured infor-
mation, and mixed text-table datasets typically fo-
cus on questions grounded within a single docu-
ment or table instance (Strich et al., 2026). In
contrast, SpecsQA evaluates corpus-level retrieval
and reasoning, requiring systems to identify and
combine evidence across multiple documents and
content types before answering. Because the bench-
mark is built from a website snapshot that evolves
over time, it also reduces the likelihood that ques-
tions can be answered through memorization by
pre-trained language models alone.
Experiments show that DualGraph consistently
outperforms state-of-the-art dense-retrieval, graph-
based, symbolic, and table-oriented baselines
across evaluation metrics. The improvement is es-
pecially pronounced on questions like identifying
all products satisfying a set of constraints, a setting
where current RAG systems are particularly limited.
These results show that the textual and symbolic
graph views provide complementary strengths: the
SKG improves precise filtering and exhaustive list
retrieval, while the TKG provides robustness for
more open-ended or underspecified questions.
To summarize, our contributions are:1)We
introduceDualGraph, a RAG architecture that
combines semantic and symbolic retrieval through
complementary graph representations;2)We re-
leaseSpecsQA, a benchmark for question answer-
ing over semi-structured corpora with both natural-
language and symbolic answers;3)We empirically
show that current RAG systems remain limited on
semi-structured QA, and that combining semantic
and symbolic retrieval improves performance.
1.1 Related Work
Retrieval-Augmented Generation (RAG) (Lewis
et al., 2020; Gao et al., 2023) augments language
models with external evidence retrieved at infer-
ence time. Standard index-based RAG systems
retrieve text chunks according to semantic sim-
ilarity to the user query and have shown strong
performance on open-domain question answering
tasks. However, semantic retrieval becomes less
reliable in large-scale semi-structured corpora, es-
pecially when answering requires exact filtering,
aggregation, or exhaustive retrieval over structured
attributes distributed across multiple documents.To improve retrieval over complex corpora, re-
cent GraphRAG approaches (Edge et al., 2025;
Peng et al., 2025; Zhuang et al., 2025; Hu et al.,
2025; Gutiérrez et al., 2025; Sarthi et al., 2024) aug-
ment retrieval with graph representations that cap-
ture entities, relations, and cross-document struc-
ture. These methods improve multi-hop and global
retrieval, but they do not provide formal guarantees
for operations such as exhaustive listing or exact fil-
tering. Complementary symbolic and logic-based
systems (Chepurova et al., 2025; Mo et al., 2025)
support exact inference, but typically require struc-
tured inputs and are less robust to noisy natural-
language text. More recent agentic RAG systems
(Chen et al., 2026; Du et al., 2026; Maragheh et al.,
2025; Besrour et al., 2025; Singh et al., 2025) fur-
ther extend retrieval pipelines through iterative ex-
ploration and reasoning steps.
Question answering benchmarks have tradition-
ally focused either on unstructured text or on struc-
tured information such as tables and knowledge
graphs. Popular text-only datasets include Nat-
uralQuestions (Kwiatkowski et al., 2019), Trivi-
aQA (Joshi et al., 2017), HotpotQA (Yang et al.,
2018), and MuSiQue (Trivedi et al., 2022). Struc-
tured and semi-structured QA benchmarks include
knowledge-graph datasets such as WebQuestions
(Berant et al., 2013), ComplexWebQuestions (Tal-
mor and Berant, 2018), 2WikiMultiHopQA (Ho
et al., 2020), and BMW-KG (Roy et al., 2025), as
well as table-oriented datasets such as WikiTable-
Questions (Pasupat and Liang, 2015), HybridQA
(Chen et al., 2020), FinQA (Chen et al., 2021), and
TAT-QA (Zhu et al., 2021).
More recent benchmarks such as RAGBench
(Friel et al., 2024) and T2-RAGBench (Strich et al.,
2026) explicitly evaluate retrieval-augmented gen-
eration in context-independent settings. However,
they still primarily focus on questions grounded
within a single document or table instance, even
when multiple knowledge types are involved. In
contrast, SpecsQA evaluates corpus-level retrieval
and reasoning over semi-structured documents, re-
quiring systems to retrieve and combine evidence
across multiple documents and content types before
answering.
Additional related work appears in Appendix A.
2 SpecsQA Dataset
SpecsQA evaluates question answering over a cor-
pus of semi-structured product documents, where
2

relevant evidence may appear in natural-language
text, specification tables, or across multiple docu-
ments. Existing QA benchmarks rarely combine
structured and unstructured evidence, and mixed
text-table datasets typically focus on questions an-
swerable within a single document. In contrast,
SpecsQA targets a corpus-level retrieval setting
and it also provides, both natural-language answers
and canonical product lists, allowing more reliable
evaluation. Moreover, because the benchmark is
built from a time-specific snapshot of a commer-
cial website that continuously evolves, it reduces
the likelihood that answers can be solved through
memorization by pre-trained LLMs alone.
2.1 Dataset Construction
The dataset was created from a snapshot of the Sam-
sung UK online shop ( https://www.samsung.
com/uk/ ) collected on November 14th, 2025. We
scraped 2162 webpages spanning 26 categories of
products. Product pages combine free-form textual
descriptions with structured specification tables,
although layouts varied across categories and re-
quired dedicated parsing pipelines (more details
and examples in Appendix C.1).
Many products are variants of a broader prod-
uct family, differing only in features such as color,
storage capacity, or screen size. Since multiple
variants with different prices are often mapped to
the same URL, we additionally stored, besides
the raw HTML pages, the variant configuration
used when visiting each webpage: we extracted
structured metadata including product names, cate-
gories, prices, model identifiers, and specification
attributes into a unified JSON representation.
2.2 Questions Design
We manually wrote 117 questions designed to
evaluate both retrieval and reasoning over semi-
structured product data. Questions were grouped
into four categories summarized in Table 1:In-
verse queriesrequire retrieving all products satis-
fying a given property or relation;Multi-condition
queriescombine several structured constraints si-
multaneously;Group comparison queriesrequire
comparing properties across multiple product fami-
lies; whileReasoning queriesinvolve more open-
ended recommendations and user-preferences.
Questions were designed around realistic
consumer-oriented criteria such as price ranges,
battery capacity, connectivity features, display tech-
nology, or device categories. Constraints were in-Table 1: Distribution of SpecsQA questions by category,
and percentage of objective and list-based answers.
Question Category Proportion Count Obj. List
Inverse35.0% 41 100% 100%
Multi-condition22.2% 26 100% 100%
Group Comparison20.5% 24 100% 0%
Reasoning22.2% 26 7.7% 96.2%
Total117 79.5% 78.6%
tentionally formulated to require either exact filter-
ing or aggregation over structured attributes. For
example, a query may ask for all smartphones un-
der a certain price threshold that simultaneously
support 5G connectivity, AMOLED displays, and
a minimum battery capacity. Additional examples
are provided in Appendix C.4.
2.3 Answers Annotation
Ground-truth answers were manually annotated
using the information available on the website at
the time of scraping. Since product availability
and prices evolve over time, we release the original
scraped data snapshot together with the benchmark
to ensure reproducibility.
Depending on the question type, answers were
represented either as natural-language text, product
lists, or both. Product lists enable deterministic
evaluation for questions requiring exact filtering
or exhaustive retrieval, while natural-language an-
swers support evaluation of more open-ended rea-
soning tasks. To allow for NL-based evaluation we
also automatically generated a NL version of prod-
uct lists answers using an LLM. We additionally
annotate whether a question is objective or sub-
jective, distinguishing factual specification queries
(e.g., which products meet given specifications)
from recommendation-oriented questions (e.g., a
phone for an elderly person) where multiple an-
swers may be acceptable.
3 DualGraphRAG
We consider question answering over semi-
structured corpora containing both natural-
language descriptions and structured content
such as specification tables. Given a corpus of
documents C={d i}N
i=1and a question q∈ Q , the
goal is to generate an answer agrounded in the
corpus. For questions with a well-defined symbolic
target (i.e., a canonical set of entities), the answer
may also be converted into a symbolic form asym
(e.g., using an LLM to extract entities from a). We
3

PreprocessingEntity extraction and
deﬁnition generation
Symbolic-based 
entity extraction
Logical predicates
extraction and
deﬁnitionTextual
Knowledge
Graph
Symbolic
Knowledge
GraphReﬁnement
& Alignment
DocumentsDualGraphLLM modelInformation extractionFigure 1: Overview of Dualgraph indexing process. Legend: Blue - TKG processing; Orange - SKG processing.
therefore model the task as f: (q,C)7→a where
the system must retrieve a supporting evidence set
e(q,C)⊆ C sufficient to generate a complete and
grounded answer.
To solve this class of problems, we proposeDu-
alGraph(see Figure 1 for an overview) a retrieval
framework that converts a corpus Cinto an inter-
mediate representation composed of two comple-
mentary graph views: aTextual Knowledge Graph
(TKG)and aSymbolic Knowledge Graph (SKG).
TheTKG= (V T, ET)encodes natural language
information as text-annotated nodes and relations,
while the SKG= (V S, ES)encodes structured
information as typed entities and directed predicate
edges formingsubject-predicate-objecttriples.
3.1 Textual and Symbolic Graph Views
TheTextual Knowledge Graphis an undirected
graph TKG= (V T, ET)that captures natural
language information in the corpus. Each node
v∈V Tis associated with a canonical name and a
textual description generated from all the passages
supporting the entity, while each edge e∈E Trep-
resents mutual semantic relationships between two
entities and is also annotated with a textual descrip-
tion. However, in our implementation we dropped
explicit TKG edges and retain only entity nodes
for efficiency. In contrast to the SKG, the TKG
preserves linguistic context and ambiguity, making
it more suitable for semantic matching, paraphrase
robustness, and underspecified questions.
TheSymbolic Knowledge Graphis a directed
graph SKG= (V S, ES)that represents structured
facts extracted from the corpus in the form of
typed subject–predicate–object triples. We assume
an ontology-based representation, where a node
v∈V Scorresponds to an individual, a class or a
numeric attribute, while each edge e∈E Srepre-
sent a logic predicate p∈ P mapping a subject
node vsto an object node vo. As any symbolicknowledge graph, SKG supports exact operations
including filtering, aggregation, comparison, arith-
metic constraints, and exhaustive listing. Questions
admitting a symbolic interpretation can therefore
be mapped to formal queries (e.g., SPARQL) that,
executed directly on the graph, return a set of bind-
ings, triples, or subgraphs, which can be used for
answer generation.
3.2 Information Extraction and Graph
Construction
Given a corpus C={d i}N
i=1of documents in semi-
structured formats, we first decompose them into
a sequence of text chunks tjand a set of semi-
structured components sj(e.g., specification ta-
bles), where the chunking granularity is treated as
a hyper-parameter. The text chunks are processed
into MD files to construct the TKG, while the struc-
tured components are processed into JSON files
and used to populate the SKG (see Appendix B.2
for more details).
Textual Knowledge Graph.The TKG is in-
duced through open information extraction and
does not assume a predefined schema. We follow
the method proposed in UnWeaver (Tuora et al.,
2026) which uses an LLM to map each chunk tj
to a set of entity mentions. Entity descriptions are
then aggregated across supporting chunks and em-
bedded to form an entity-centric retrieval index (see
Appendix B.5 for more details).
Symbolic Knowledge Graph.SKG con-
struction assumes an input ontology O(i.e., a
description-logic schema specifying classes and
predicates). In our implementation, we manu-
ally designed Oas a lightweight schema (see Ap-
pendix B.4) intended to minimize domain-specific
engineering while remaining simple enough to
adapt when the underlying data format changes.
The SKG is populated by converting each semi-
structured component sjinto ontology-consistent
4

triples (vs, p, vo), where subject vsand object so
nodes represent typed entities, attributes, or values,
and edges represent predicates from the ontology
O. Each specification row extracted from a product
table is modeled as a Spec entity connected to its
product, specification section, feature name, and
value node, following the mapping in Appendix Ta-
bles 5 and 6, Figure 3. This preserves the origi-
nal tabular organization while remaining generic
enough to accommodate diverse webpage layouts
and product categories. Product variants are rep-
resented as distinct but related entities connected
through shared product-range and category nodes.
Numeric values undergo additional symbolic
processing. When possible, numeric quantities are
detected within the raw text and converted into
typed symbolic representations with normalized
units, including the special case of multidimen-
sional quantities where several numerical compo-
nents must be represented jointly (e.g., “resolution”
which contains both width and height). This en-
ables exact numerical operations (e.g, filtering with
arithmetic constraints) directly in SPARQL. This
processing is necessary due to noise in the scraped
data, and for the same reason we preserve the orig-
inal textual value to prevent information loss.
Beyond specification rows, the SKG also models
product hierarchies and shared features. Products
are grouped into ranges and categories, while com-
mon features are represented as reusable entities
linked to multiple products, reducing duplication
and producing more consistent graph patterns for
SPARQL generation.
Entity and value normalization is central to SKG
construction: product identifiers, feature names,
section labels, and categorical values are canon-
icalized through lightweight heuristics including
lowercasing, stemming, stop-word removal, and
string normalization, reducing duplication due to
inconsistent webpages or table structures.
The SKG is then enriched via lightweight rule-
based reasoning. We defined a set of Datalog rules
(see Appendix B.3) on top of the schema Oto
derive higher-level features implicitly expressed
in the specifications (e.g., inferring “5G Support”
from the presence of specific 5G standards).
Finally, the two graph views are refined through
alignment: We first apply joint canonicalization
heuristics that normalize identifiers, names, and
types, and merge duplicate nodes. This simple
step already aligns many entities across VTand
VS, since many identifiers in e-commerce corporaare already partially standardized. We also experi-
mented with a learned contrastive alignment model
based on textual and graph embeddings, but ob-
served no measurable improvement.
3.3 Retrieval and Question Answering
Given a question q, DualGraph supports two
main retrieval functions: a symbolic SKG-based
retriever, implemented through a LLM-based
SPARQL query generation, and a semantic TKG-
based retriever, following the entity-centric re-
trieval procedure of UnWeaver (Tuora et al., 2026).
Semantic Retrieval.The retrieval function used
in UnWeaver is similar to VectorRAG, but uses
entities as an intermediate layer between questions
and chunks. Given a question q, the system first
retrieves the entities whose descriptions are most
similar to q. Each retrieved entity then votes for
its originating chunks, with votes weighted by the
entity’s similarity rank. This produces a final chunk
ranking, from which the top-kare selected.
Symbolic retrieval.Symbolic retrieval maps a
natural-language question qto a SPARQL query ex-
ecuted over the SKG. The NL-to-SPARQL transla-
tion is performed by an LLM prompted with qand
two context components: acommon component
shared across all questions, and aquestion-specific
componentretrieved at inference time.
Thecommon componentspecifies: (1) a persona
defining the model as an expert in SPARQL, RDF,
OWL, and related standards; (2) query-generation
rules for selecting query types, applying filters, and
incorporating constraints; (3) a domain description
of online-shop product data represented as a knowl-
edge graph; (4) the SKG schema, including rules
for handling numerical values and prices; and (5)
the expected output format.
Thequestion-specific componentconsists of re-
trieved graph patterns. After graph construction,
predefined schema-based patterns are linearized au-
tomatically into natural language, embedded, and
stored in a vector database. At inference time, the
patterns most similar to qare retrieved and used
to ground SPARQL generation. We use four pat-
tern types:Spec patterns, describing product at-
tributes (e.g.,In the product specification, the S
Pen Support entry in the Specifications section has
the value Yes);Feature patterns, describing product
capabilities (e.g.,The product has Samsung Dex
Support feature);Category patterns, representing
product-family information (e.g.,Galaxy S); and
Singular Node patterns, representing individual en-
5

tities appearing in triples.
In our experiments, we retrieve 5instances of
each pattern type and convert them into syntacti-
cally valid SPARQL snippets, which are included
in the LLM prompt. The LLM then generates can-
didate SPARQL queries, which are executed over
the SKG. Results are formatted as Markdown ta-
bles together with the query that produced them.
We discard queries that fail execution, return more
than 100results, or return no results when all other
candidates also fail or are discarded.
To improve robustness, we generate multiple
candidate SPARQL queries ( 3in our experiments)
through controlled sampling, execute them on the
SKG, and use all the resulting outputs as context
for answer generation.
We also experimented with an agentic sym-
bolic retrieval, where an LLM iteratively refines
SPARQL queries based on execution feedback.
However, since it did not improve retrieval per-
formance, we do not report these results.
3.4 Orchestration and Answer Generation
DualGraph supports multiple orchestration strate-
gies combining symbolic and semantic retrieval
(see Appendix-Figure 4). We evaluate seven vari-
ants:(1)TKG only, which applies semantic re-
trieval only;(2)SKG only, which applies symbolic
retrieval only;(3)TKG + SKG, which concate-
nates the contexts returned by both retrievers;(4)
SKG + TKG fallback, which applies semantic re-
trieval when symbolic retrieval returns no context;
(5)Router, which uses an LLM binary classifier
to select between symbolic and semantic retrieval
conditioned on the user question, descriptions of
the retrieval methods, and few-shot examples;(6)
Router + TKG fallback, which applies semantic
retrieval as fallback when routed symbolic retrieval
fails; and(7)Agentic, which uses an LLM agent
with access to both retrieval functions. The agentic
orchestrator, implemented with Pydantic-AI (Py-
dantic, 2023), iteratively decides whether to invoke
a retriever, reformulate the query, generate an an-
swer, or continue retrieval based on intermediate
outputs. After generating an answer, it self-reflects
and either returns the answer, revises it, or contin-
ues the loop until a stopping criterion is reached.
Conditioned on the retrieved context and q, a
generator LLM produces a natural-language an-
swera. When symbolic output is required for eval-
uation, an additional LLM-generation step extracts
a canonical symbolic answera symfroma.4 Results
We performed several experiments of which the
main results can be summarized as follows:1)
DualGraph outperforms state-of-the-art baselines,
with the largest gains on specification-heavy
queries requiring exact filtering and exhaustive
lists;2)The SKG–TKG duality is key for han-
dling diverse question types: symbolic retrieval
improves precision over structured attributes while
textual retrieval improves coverage and robustness;
3)SpecsQA serves as a diagnostic benchmark for
semi-structured QA, revealing the strengths and
weaknesses of different retrieval strategies. The
full set of results, together with additional experi-
ments, can be found in Appendix D.
4.1 Baselines and Metrics
We compared our method with state-of-the-art
baselines including: pure LLM calls (‘LLM-
only’), standard vector-based RAG systems (‘Vec-
tor RAG’) and state-of-the-art graph-based (‘Mi-
crosoft GraphRAG’ (Edge et al., 2025) – both stan-
dard and fast version, ‘RAPTOR’ (Sarthi et al.,
2024), ‘LinearRAG’ (Zhuang et al., 2025), ‘Ari-
Graph’ (Anokhin et al., 2025), ‘HippoRAG 2’
(Gutiérrez et al., 2025)), agentic (‘A-RAG’ (Du
et al., 2026)), symbolic (‘Wikontic’ (Chepurova
et al., 2025) ) and table-oriented (‘TableRAG’ (Yu
et al., 2025a)) RAG methods.
We evaluate all systems with four metrics:Fac-
tual Correctnessfrom RAGAS (VibrantLabs, 2024)
measures whether generated statements are sup-
ported by the ground truth, after decomposing
both into atomic claims;List Matchevaluates list-
answer questions by extracting (with an LLM) a
predicted symbolic list from the generated answer
and computing set-based precision, recall, and F1
against the ground-truth list (when available);Pair-
wise LLM-as-a-judge (LaaJ)compares system out-
puts in pairs using a fixed prompt and reports each
system’s aggregate win rate; andComputational
cost, measured as input and output token usage.
For all experiments, we used GPT-OSS-120B as
the underlying LLM, Qwen3-Embedding-4B as em-
bedding model and report results averaged over 5
indexing runs, 3 query-generation runs, and 3 eval-
uation runs to reduce stochastic variability.
4.2 DualGraph vs Baselines
Table 2 compares DualGraph with the considered
RAG baselines. We report our two most represen-
6

Table 2: Comparison between DualGraph and state-of-the-art baselines on SpecsQA. We report answer quality using
factual correctness (FC), list matching (LM), pairwise LLM-as-a-judge (LaaJ), and objective-only LM, together
with indexing and query-time token usage.
Answer Quality Token Usage
FC (F1) LM (F1) LaaJ Obj. LM (F1) Indexing (·106) Query (·103)
LLM only 0.107 0.028 0.559 0.0360 0.565
Vector RAG 0.092 0.118 0.438 0.1580 2.436
Microsoft GraphRAG 0.153 0.203 0.575 0.243181.213 11.120
Microsoft GraphRAG (fast) 0.083 0.140 0.420 0.15829.420 12.236
RAPTOR 0.207 0.216 0.568 0.2485.615 2.378
LinearRAG 0.138 0.085 0.526 0.0930 7.541
AriGraph 0.072 0.124 0.378 0.14016.516 5.097
HippoRAG 2 0.149 0.152 0.540 0.18216.084 22.217
Wikontic 0.135 0.290 0.523 0.349420.325 138.627
A-RAG 0.057 0.129 0.331 0.1590 36.384
TableRAG 0.091 0.118 0.453 0.1390 20.527
DualGraph (R+TKG_fb) (our) 0.2980.357 0.6400.44116.058 7.678
DualGraph (SKG+TKG_fb) (our)0.2930.372 0.6440.43416.058 6.802
Table 3: Ablation of our different DualGraph variants.
Answer Quality Token Usage
DualGraph FC (F1) LM (F1) LaaJ Query (·103)
SKG only 0.273 0.321 0.5075.471
TKG only 0.140 0.136 0.3523.095
SKG concat TKG0.3060.367 0.5618.186
SKG+TKG_fb0.2930.3720.5236.802
Router 0.268 0.303 0.4856.503
Router+TKG_fb0.298 0.357 0.5287.678
Agentic 0.240 0.3410.76455.454
tative variants:Router+TKG fallbackachieves the
highest factual correctness, andSKG+TKG fall-
backachieves the best list matching and LLM-
as-a-judge scores. The strongest non-DualGraph
baselines are RAPTOR for factual correctness and
Wikontic for list matching.
Overall, the results show that combining sym-
bolic querying over structured specifications with
semantic retrieval substantially improves QA qual-
ity on semi-structured corpora. Text-only methods
retrieve locally relevant passages but struggle with
global questions, while graph-based baselines im-
prove retrieval structure but still lack support for
exhaustive lists and precise filtering. Wikontic re-
mains competitive on symbolic questions, but at
significantly higher indexing and query cost.
DualGraph provides a favorable quality–cost
trade-off: its indexing cost is comparable to Hip-
poRAG 2 and AriGraph, and substantially lower
than Microsoft GraphRAG and Wikontic, while
maintaining moderate query-time cost. Finally, the
relatively low list-matching scores across systems
indicate the difficulty of SpecsQA, where answers
are sensitive to indexing noise and product avail-ability. Moreover, we observe consistently higher
performance on objective questions, since subjec-
tive recommendation-oriented queries may admit
multiple valid answers that may differ from the
ground-truth annotations.
4.3 Impact of SKG and retrieval strategies
Table 3 reports an ablation over DualGraph re-
trieval strategies. TKG-only retrieval performs
substantially worse than SKG-based variants, espe-
cially on list matching, confirming that semantic re-
trieval alone is insufficient for semi-structured QA.
Hybrid variants consistently outperform single-
view retrieval, showing that symbolic and textual
representations provide complementary strengths.
Among the variants,SKG+TKG fallback
achieves the best list-matching performance, while
Router+TKG fallbackoffers the best general-
purpose trade-off between performance and token
usage by selecting retrieval strategies dynamically
and falling back to textual retrieval when sym-
bolic querying fails. The former is particularly
effective on SpecsQA because many questions are
specification-heavy, whereas the latter is likely
preferable for more balanced datasets. Concate-
nating both contexts slightly improves factual cor-
rectness but increases query cost, while the agentic
router is substantially more expensive due to iter-
ative retrieval steps. However, it also achieves the
highest LLM-as-a-judge score, possibly reflecting
known verbosity and superficial-quality biases in
LLM-based evaluators (Zhou et al., 2024).
Overall, the results support the central design of
DualGraph: symbolic retrieval improves precision
7

Inverse Multi-Condition Reasoning
Category0.00.10.20.30.4LM (F1)LLM only
Vector RAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our)Figure 2: List matching (F1) results by question category for DualGraph and baseline retrieval methods.
on structured constraints, while textual retrieval
increases robustness on open-ended and underspec-
ified questions.
4.4 Impact of Question Categories
Figure 2 reports performance by question category.
DualGraph performs particularly well on inverse
and multi-condition queries, where answers usu-
ally require exact filtering and exhaustive product
retrieval. These gains mainly derive from the SKG
component and its support for structured querying.
These results demonstrate the value of the SKG
component for specification-heavy questions.
Performance is lower across all systems on
group-comparison (see Appendix Fig 12 and 13)
and reasoning questions, especially for subjective
recommendation-style queries. In these cases, the
TKG provides complementary natural-language
context that improves robustness beyond purely
symbolic retrieval. The category-level breakdown
also shows the diagnostic value of SpecsQA, mak-
ing it possible to identify which retrieval capabili-
ties are required by different classes of questions.
4.5 Impact of Patterns
Table 4 reports an ablation on the contribution of
graph patterns to SPARQL generation quality.Spec
appears to be the most critical pattern: removing
it leads to the largest drop in successful SPARQL
retrievals and substantially lowers both factual cor-
rectness and list matching. This shows that de-
tailed specification-level descriptions are essential
for constructing accurate symbolic queries. The
fact that using theSpecpattern alone remains com-
petitive further confirms its foundational role in
symbolic retrieval.FeatureandSingular Nodepat-
terns provide smaller but consistent improvements,
suggesting that they offer complementary context
for some question types. TheCategorypatternTable 4: Ablation of graph patterns [Spec, Feature, Cat-
egory, Singular Node] used for SPARQL generation
in DualGraph (SKG with TKG fallback). We report
the percentage of successful SKG SPARQL retrievals,
factual correctness (FC), list matching (LM), and LLM-
as-a-judge (LaaJ) scores.
Pattern % SKG FC(F1) LM(F1) LaaJ
Y Y Y Y50.20.293 0.372 0.534
N Y Y Y 37.7 0.247 0.292 0.461
Y N Y Y 48.4 0.299 0.361 0.521
Y Y N Y 48.80.317 0.3800.538
Y Y Y N 48.8 0.296 0.367 0.536
Y N N N 44.7 0.301 0.3620.545
N Y N N 31.3 0.246 0.290 0.478
N N Y N 34.7 0.228 0.259 0.449
N N N Y 37.0 0.246 0.299 0.472
N N N N 29.4 0.230 0.253 0.459
appears less beneficial and may even degrade per-
formance: the best overall performance is achieved
when it is removed while all other patterns are re-
tained. However, we keep all patterns in the main
experiments because the full configuration achieves
the highest SPARQL retrieval success rate and ro-
bust performance across metrics. The marginal
gains from removing Category are outweighed by
the benefit of a more general default that can adapt
to diverse question types beyond SpecsQA.
4.6 Conclusions
To conclude, we introduced DualGraph, a RAG
architecture that leverages the strengths of both
semantic and symbolic retrieval, combining flexi-
ble language-based retrieval with exact structured
querying, and outperforming state-of-the-art base-
lines particularly on questions requiring precise fil-
tering and exhaustive retrieval. We also presented
SpecsQA, a useful new benchmark for question
answering over semi-structured documents, a com-
mon scenario in real-world applications.
8

Limitations
Our work opens several directions for further
improvement. In the current implementation,
the alignment between the TKG and SKG is
lightweight: because symbolic nodes do not have
textual descriptions, SKG and TKG entities are
mainly aligned through normalized names. How-
ever, even this simple alignment allows the two
graph views to provide complementary retrieval
functions. Richer cross-graph alignment could fur-
ther exploit this complementarity, enabling gen-
uinely joint SKG–TKG retrieval in which sym-
bolic results are expanded with textual evidence,
or semantic retrieval is constrained by symbolic
structure. Improving this alignment through richer
node descriptions, an improved learned cross-graph
matching, or retrieval functions that explicitly tra-
verse both views is therefore a promising direction
for future work.
The SKG is currently constructed mainly from
the structured portions of the corpus, which in
SpecsQA correspond to product specification ta-
bles. This choice provides a clean and reliable sym-
bolic representation for attributes that are naturally
tabular, and is well suited to exact filtering, compar-
ison, and list retrieval. Some product information,
however, may appear only in natural-language de-
scriptions, marketing text, or FAQs. Extending
symbolic extraction to these textual sections, or us-
ing stronger SKG–TKG alignment to combine par-
tial symbolic evidence with textual context, would
allow DualGraph to cover an even broader range
of semi-structured questions.
Several symbolic components, including the ini-
tial ontology, SPARQL retrieval patterns, and Dat-
alog rules, are currently specified by a domain ex-
pert. This manual design makes the pipeline in-
terpretable, controllable, and robust in our setting,
while keeping the symbolic representation compact.
Future work could reduce this domain-specific ef-
fort by automatically inducing schemas and rules,
learning derived features from data, or extracting
useful SPARQL patterns from the graph schema
and query distribution.
Finally, our orchestration strategies are deliber-
ately simple and general-purpose. We use LLM-
based routing and agentic retrieval variants to study
how symbolic and textual retrieval can be com-
bined without task-specific training. More adaptive
routing policies could further improve performance,
especially for ambiguous questions or cases requir-ing close interaction between both graph views. For
example, a router trained on task-specific feedback
could learn when to use symbolic retrieval, textual
retrieval, or a joint strategy.
References
Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry
Evseev, Andrey Kravchenko, Mikhail Burtsev, and
Evgeny Burnaev. 2025. Arigraph: Learning knowl-
edge graph world models with episodic memory for
llm agents. InProceedings of the Thirty-Fourth Inter-
national Joint Conference on Artificial Intelligence,
IJCAI-25, pages 12–20. International Joint Confer-
ences on Artificial Intelligence Organization. Main
Track.
Jonathan Berant, Andrew Chou, Roy Frostig, and Percy
Liang. 2013. Semantic parsing on Freebase from
question-answer pairs. InProceedings of the 2013
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 1533–1544, Seattle, Wash-
ington, USA. Association for Computational Linguis-
tics.
Ines Besrour, Jingbo He, Tobias Schreieder, and
Michael Färber. 2025. RAGentA: Multi-agent
retrieval-augmented generation for attributed ques-
tion answering. InSIGIR 2025 LiveRAG Challenge
(Workshop).
Chia-Yuan Chang, Zhimeng Jiang, Vineeth Rakesh,
Menghai Pan, Chin-Chia Michael Yeh, Guanchu
Wang, Mingzhi Hu, Zhichao Xu, Yan Zheng, Ma-
hashweta Das, and Na Zou. 2025. MAIN-RAG:
Multi-agent filtering retrieval-augmented generation.
InProceedings of the 63rd Annual Meeting of the As-
sociation for Computational Linguistics (ACL 2025),
pages 2607–2622.
Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong,
Hong Wang, and William Yang Wang. 2020. Hy-
bridQA: A dataset of multi-hop question answering
over tabular and textual data. InFindings of EMNLP
2020, pages 1026–1036.
Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena
Shah, Iana Borova, Dylan Langdon, Reema Moussa,
Matt Beane, Ting-Hao Huang, Bryan Routledge, and
William Yang Wang. 2021. FinQA: A dataset of nu-
merical reasoning over financial data. InProceedings
of EMNLP 2021, pages 3697–3711.
Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma,
Sameena Shah, and William Yang Wang. 2022. Con-
vFinQA: Exploring the chain of numerical reasoning
in conversational finance question answering. InPro-
ceedings of the 2022 Conference on Empirical Meth-
ods in Natural Language Processing, pages 6279–
6292, Abu Dhabi, United Arab Emirates. Association
for Computational Linguistics.
Zihan Chen, Lei Zheng, and Di Zhu. 2026. A survey of
agentic graphrag: From retrieval-augmented genera-
tion to graph-native agents. (6713979).
9

Alla Chepurova, Aydar Bulatov, Yuri Kuratov, and
Mikhail Burtsev. 2025. Wikontic: Construct-
ing wikidata-aligned, ontology-aware knowledge
graphs with large language models.Preprint,
arXiv:2512.00590.
Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan
Wang, Pengyu Wang, Xiaorui Wang, and Zhen-
dong Mao. 2026. A-rag: Scaling agentic retrieval-
augmented generation via hierarchical retrieval inter-
faces.arXiv preprint arXiv:2602.03442.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2025. From local to global: A
graph rag approach to query-focused summarization.
Preprint, arXiv:2404.16130.
Vincent Emonet, Jerven Bolleman, Severine Duvaud,
Tarcisio Mendes de Farias, and Ana Claudia Sima.
2025. Llm-based sparql query generation from nat-
ural language over federated knowledge graphs. In
ISWC 2024 Special Session on Harmonising Gen-
erative AI and Semantic Web Technologies, Novem-
ber 13, 2024, Baltimore, Maryland, volume 3953 of
CEUR Workshop Proceedings. CEUR-WS.org, on-
linehttps://ceur-ws.org/Vol-3953/355.pdf.
Robert Friel, Masha Belyi, and Atindriyo Sanyal. 2024.
RAGBench: Explainable benchmark for retrieval-
augmented generation systems.arXiv preprint
arXiv:2407.11005.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, Haofen Wang, and 1 others. 2023. Retrieval-
augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1):32.
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. 2024. Hipporag: Neu-
robiologically inspired long-term memory for large
language models. InThe Thirty-eighth Annual Con-
ference on Neural Information Processing Systems.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From rag to memory:
Non-parametric continual learning for large language
models.Preprint, arXiv:2502.14802.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-hop
qa dataset for comprehensive evaluation of reasoning
steps.Preprint, arXiv:2011.01060.
Mengkang Hu, Haoyu Dong, Ping Luo, Shi Han, and
Dongmei Zhang. 2024. KET-QA: A dataset for
knowledge enhanced table question answering.arXiv
preprint arXiv:2405.08099.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen
Ling, and Liang Zhao. 2025. Grag: Graph retrieval-
augmented generation.Preprint, arXiv:2405.16506.Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke
Zettlemoyer. 2017. TriviaQA: A large scale distantly
supervised challenge dataset for reading comprehen-
sion. InProceedings of the 55th Annual Meeting of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 1601–1611, Vancouver,
Canada. Association for Computational Linguistics.
Kezhi Kong, Jiani Zhang, Zhengyuan Shen, Bal-
asubramaniam Srinivasan, Chuan Lei, Chris-
tos Faloutsos, Huzefa Rangwala, and George
Karypis. 2024. Opentab: Advancing large lan-
guage models as open-domain table reasoners.
arXiv preprint arXiv:2402.14361. ICLR 2024,
Code: https://github.com/amazon-science/
llm-open-domain-table-reasoner.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–
9474. Curran Associates, Inc.
Feiyang Li, Peng Fang, Zhan Shi, Arijit Khan, Fang
Wang, Weihao Wang, Zhangxin-hw, and Yongjian
Cui. 2025a. CoT-RAG: Integrating chain of thought
and retrieval-augmented generation to enhance rea-
soning in large language models. InFindings of the
Association for Computational Linguistics: EMNLP
2025, pages 3119–3171.
Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh
Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei,
Henry Peng Zou, Xiao Luo, Yusheng Zhao, and 1
others. 2025b. A survey of rag-reasoning systems in
large language models. InFindings of the Associa-
tion for Computational Linguistics: EMNLP 2025,
pages 12120–12145.
Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan
Meng, Ding Wang, and Jun Ma. 2025. HM-
RAG: Hierarchical multi-agent multimodal re-
trieval augmented generation.arXiv preprint
arXiv:2504.12330.
Reza Yousefi Maragheh, Pratheek Vadla, Priyank Gupta,
Kai Zhao, Aysenur Inan, Kehui Yao, Jianpeng Xu,
Praveen Kanumala, Jason Cho, and Sushant Kumar.
2025. ARAG: Agentic retrieval augmented gen-
eration for personalized recommendation. InPro-
ceedings of the 48th ACM SIGIR Conference (SIGIR
2025).
10

Belinda Mo, Kyssen Yu, Joshua Kazdan, Proud Mpala,
Lisa Yu, Chris Cundy, Charilaos I. Kanatsoulis, and
Sanmi Koyejo. 2025. Kggen: Extracting knowledge
graphs from plain text with language models.CoRR,
abs/2502.09956.
Thang Nguyen, Peter Chin, and Yu-Wing Tai. 2025.
MA-RAG: Multi-agent retrieval-augmented gener-
ation via collaborative chain-of-thought reasoning.
arXiv preprint arXiv:2505.20096.
Faheem Nizar, Elias Lumer, Anmol Gulati, Pradeep
Basavaraju, and Vamse Kumar Subbiah. 2025.
Agent-as-a-Graph: Knowledge graph-based tool and
agent retrieval for llm multi-agent systems.arXiv
preprint arXiv:2511.18194.
OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card.
Preprint, arXiv:2508.10925.
Panupong Pasupat and Percy Liang. 2015. Composi-
tional semantic parsing on semi-structured tables. In
Proceedings of ACL 2015, pages 1470–1480.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2025. Graph retrieval-augmented generation:
A survey.ACM Trans. Inf. Syst., 44(2).
Pydantic. 2023. pydantic/pydantic-ai: Genai agent
framework, the pydantic way. [Online; accessed
2026-01-30].
Rishiraj Saha Roy, Chris Hinze, Joel Schlotthauer,
Farzad Naderi, Viktor Hangya, Andreas Foltyn,
Luzian Hahn, and Fabian Küch. 2025. RAGO-
NITE: iterative retrieval on induced databases and
verbalized RDF for conversational QA over kgs with
RAG. InDatenbanksysteme für Business, Technolo-
gie und Web (BTW 2025), 21. Fachtagung des GI-
Fachbereichs „Datenbanken und Informationssys-
teme" (DBIS), 03.-07, März 2025, Bamberg, Ger-
many, Proceedings, volume P-361 ofLNI, pages 787–
794. Gesellschaft für Informatik e.V .
Rishiraj Saha Roy, Chris Hinze, Joel Schlotthauer,
Farzad Naderi, Viktor Hangya, Andreas Foltyn,
Luzian Hahn, and Fabian Kuech. 2024. Ragonite: It-
erative retrieval on induced databases and verbalized
rdf for conversational qa over kgs with rag.Preprint,
arXiv:2412.17690.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D. Manning.
2024. RAPTOR: recursive abstractive processing for
tree-organized retrieval. InThe Twelfth International
Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024. OpenReview.net.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented gen-
eration: A survey on agentic rag.arXiv preprint
arXiv:2501.09136.Panayiotis Smeros, Vincent Emonet, Ruijie Wang,
Ana-Claudia Sima, and Tarcisio Mendes de Farias.
2025. Sparql-llm: Real-time sparql query gener-
ation from natural language questions.Preprint,
arXiv:2512.14277.
Jan Strich, Enes Kutay Isgorur, Maximilian Trescher,
Chris Biemann, and Martin Semmann. 2026. T2-
RAGBench: Text-and-table benchmark for evaluat-
ing retrieval-augmented generation. InProceedings
of the 19th Conference of the European Chapter of
the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pages 165–191, Rabat, Mo-
rocco. Association for Computational Linguistics.
Alon Talmor and Jonathan Berant. 2018. The web as
a knowledge-base for answering complex questions.
InProceedings of the 2018 Conference of the North
American Chapter of the Association for Computa-
tional Linguistics: Human Language Technologies,
Volume 1 (Long Papers), pages 641–651, New Or-
leans, Louisiana. Association for Computational Lin-
guistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
TACL, 10:539–554.
Ryszard Tuora, Mateusz Gali ´nski, Michał
Godziszewski, Michał Karpowicz, Mateusz
Czy˙znikiewicz, Adam Kozakiewicz, and Tomasz
Zi˛ etkiewicz. 2026. Unweaving the knots of graphrag
– turns out vectorrag is almost enough.Preprint,
arXiv:2603.29875.
VibrantLabs. 2024. Ragas: Supercharge your llm
application evaluations. https://github.com/
vibrantlabsai/ragas.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of EMNLP 2018, pages 2369–2380.
Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga,
Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingn-
ing Yao, Shanelle Roman, Zilin Zhang, and Dragomir
Radev. 2018. Spider: A large-scale human-labeled
dataset for complex and cross-domain semantic pars-
ing and text-to-SQL task. InProceedings of the 2018
Conference on Empirical Methods in Natural Lan-
guage Processing, pages 3911–3921, Brussels, Bel-
gium. Association for Computational Linguistics.
Xiaohan Yu, Pu Jian, and Chong Chen. 2025a. Tablerag:
A retrieval augmented generation framework for
heterogeneous document reasoning.Preprint,
arXiv:2506.10380.
Xiaohan Yu, Pu Jian, and Chong Chen. 2025b.
TableRAG: A retrieval augmented generation frame-
work for heterogeneous document reasoning. InPro-
ceedings of the 2025 Conference on Empirical Meth-
ods in Natural Language Processing, pages 14063–
11

14082, Suzhou, China. Association for Computa-
tional Linguistics.
Xiaohan Yu, Pu Jian, and Chong Chen. 2025c. Tablerag:
A retrieval augmented generation framework for
heterogeneous document reasoning. InProceed-
ings of EMNLP 2025, pages 14063–14082. Code:
https://github.com/yxh-y/TableRAG.
Xiaohan Yu, Zhihan Yang, and Chong Chen. 2025d.
CogPlanner: Unveiling the potential of agentic multi-
modal retrieval augmented generation with planning.
InProceedings of the 48th International ACM SI-
GIR Conference on Research and Development in
Information Retrieval (SIGIR 2025).
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Victor Zhong, Caiming Xiong, and Richard Socher.
2017. Seq2sql: Generating structured queries
from natural language using reinforcement learning.
Preprint, arXiv:1709.00103.
Hongli Zhou, Hui Huang, Yunfei Long, Bing Xu, Con-
ghui Zhu, Hailong Cao, Muyun Yang, and Tiejun
Zhao. 2024. Mitigating the bias of large language
model evaluation. InProceedings of the 23rd Chi-
nese National Conference on Computational Linguis-
tics (Volume 1: Main Conference), pages 1310–1319.
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao
Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-
Seng Chua. 2021. TAT-QA: A question answering
benchmark on a hybrid of tabular and textual content
in finance. InProceedings of ACL-IJCNLP 2021,
pages 3277–3287.
Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi
Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang,
and Xiao Huang. 2025. Linearrag: Linear graph re-
trieval augmented generation on large-scale corpora.
Preprint, arXiv:2510.10114.
Jiaru Zou, Dongqi Fu, Sirui Chen, Xinrui He, Zihao
Li, Yada Zhu, Jiawei Han, and Jingrui He. 2025.
Rag over tables: Hierarchical memory index, multi-
stage retrieval, and benchmarking.arXiv preprint
arXiv:2504.01346. Code: https://github.com/
jiaruzouu/T-RAG.
12

A Extended Literature Review
A.1 Graph RAG
GraphRAG methods extend standard retrieval-augmented generation by converting a document corpus
into a knowledge graph structure, where entities are represented as nodes and relations as edges, with
links to supporting passages. At inference time, relevant subgraphs are selected using semantic similarity
or graph traversal, enabling the language model to synthesize answers from connected evidence across
multiple documents.
Microsoft GraphRAG (Edge et al., 2025) constructs a knowledge graph from source documents by
extracting entities, relationships, and claims using large language models. The method partitions the
graph into hierarchical communities and generates community-level summaries that enable both global
sensemaking queries and local retrieval operations. GraphRAG offers two indexing variants:standard,
which uses LLM-based extraction for rich entity and relationship descriptions, andfast, which employs
traditional NLP techniques for faster and cheaper indexing. While GraphRAG effectively supports multi-
document queries through hierarchical summarization, its graph representation is primarily textual and
does not support exact symbolic operations or formal querying over structured attributes.
RAPTOR (Sarthi et al., 2024) builds a hierarchical index through recursive embedding, clustering, and
summarization of text chunks using Gaussian Mixture Models. The method constructs a multi-layered
tree structure from the bottom up, where each level represents increasingly abstract summaries of the
underlying content, enabling retrieval at varying granularities. RAPTOR’s collapsed tree querying method
allows simultaneous evaluation across abstraction levels. However, RAPTOR operates on hierarchical
text summaries rather than a structured graph, and thus cannot support precise filtering or aggregation
over typed attributes.
LinearRAG (Zhuang et al., 2025) addresses the instability of traditional relation extraction by construct-
ing a relation-free hierarchical graph comprising entity, sentence, and passage nodes connected through
contain and mention adjacency matrices. The framework employs a two-stage retrieval mechanism
with local semantic bridging followed by global importance aggregation using personalized PageRank.
While LinearRAG achieves efficient linear-time complexity, its relation-free representation prevents exact
symbolic querying and structured reasoning over predicates.
AriGraph (Anokhin et al., 2025) integrates semantic and episodic memories within a unified graph
structure for LLM agents. The system continuously extracts semantic triplets from textual observations
and links them with episodic vertices storing raw observations. While AriGraph demonstrates competitive
performance on multi-hop question-answering benchmarks at lower computational cost than dedicated
knowledge graph methods, it remains focused on semantic retrieval rather than supporting formal symbolic
operations over structured data.
RAGonite (Roy et al., 2024) is a retrieval-augmented generation system designed for conversational
question answering over RDF knowledge graphs. The method employs a two-pronged retrieval approach
that fuses SQL query results over a database automatically derived from the KG with text-search results
over verbalizations of KG facts. RAGonite supports iterative retrieval, enabling multiple rounds of
evidence gathering when initial results are unsatisfactory, and integrates both retrieval branches through
an LLM to generate coherent responses. This design shares conceptual similarities with DualGraph
in maintaining complementary symbolic and textual representations. However, RAGonite assumes a
pre-existing structured knowledge graph as input and operates over a database representation, whereas
DualGraph constructs both aligned graph views from unstructured and semi-structured documents. This
enables DualGraph’s entity-centric retrieval to leverage graph connectivity and expand evidence through
neighborhoods over the aligned graphs, while RAGonite’s SQL-based approach is limited to relational
queries without access to the underlying graph structure.
HippoRAG 2 (Gutiérrez et al., 2024, 2025) is an advanced retrieval-augmented generation framework
inspired by the neural mechanisms of human long-term memory. An LLM is presented as an artificial
neocortex, a knowledge graph (KG) with Personalized PageRank (PPR) algorithms is shown as analogous
to the hippocampus, and a retrieval encoder maps to the temporal lobe. Unlike traditional RAG methods
that rely solely on vector-based retrieval, HippoRAG 2 enhances the search process by incorporating KG
13

triples, enabling deeper contextualization and memory-aware filtering. During offline indexing, the LLM
converts documents into KG triples, while the retrieval encoder detects synonyms. At inference time,
the system extracts entities from queries, performs context-aware searches using PPR, and filters results
through a two-stage recall-and-recognition process. HippoRAG 2 introduces dense-sparse integration to
minimize information loss, and deeper contextualization to improve search accuracy. While the framework
excels in factual memory, sensemaking, and associative tasks (outperforming RAG baselines by 7% in
associative tasks) it retains the limitations of graph-based approaches, such as the inability to support
exact symbolic operations over structured attributes. Its computational efficiency and robust performance
across diverse benchmarks position it as a significant advancement in non-parametric continual learning
for LLMs. The concept of embedding whole triples is similar to our embedding of patterns, but the
application is completely different, as in our system the patterns provide anchors for querying the SKG,
while in HippoRAG they drive textual retrieval. Our patterns are also more complex and linearized in a
more natural way.
A.2 Symbolic RAG
Symbolic RAG is a subset of GraphRAG methods that encode the corpus with formal semantics, typi-
cally using description logics, ontologies, and typed predicates, by mapping documents into a logical
knowledge graph where nodes denote typed entities and edges denote predicates between them. In many
systems (Chepurova et al., 2025; Mo et al., 2025), retrieval follows the standard GraphRAG pipeline,
selecting relevant subgraphs via semantic similarity and optionally expanding them through graph traversal
to collect more evidence. In other approaches (Smeros et al., 2025; Emonet et al., 2025; Roy et al., 2025),
inference relies on translating the question into a formal query, such as SPARQL or an equivalent logic
form, to retrieve a set of triples or a subgraph, which is then returned directly or provided as context to a
language model for natural language generation.
Symbolic RAG includes a subclass of RAG systems suited especially for answering questions on
data in tabular format. Recent RAG pipelines target multi-table QA by combining retrieval of relevant
tables with structured reasoning. For example T-RAG (Zou et al., 2025), a table-corpora-aware RAG
framework builds a hierarchical index over a large collection of tables and uses multi-stage retrieval
with a graph-aware prompt to aggregate evidence across tables. Similarly, OpenTab (Kong et al., 2024)
retrieves relevant tables and uses LLM-generated SQL programs to parse these tables, yielding more
accurate answers. TableRAG (Yu et al., 2025b) loads tables into a database and iteratively performs query
decomposition, text retrieval, SQL execution, and answer generation. It achieves state-of-the-art results
on the new HeteQA benchmark, confirming the value of SQL-based, global reasoning over tables. All
these methods explicitly preserve table semantics, addressing issues with simply flattening tables into text
that often “compromises” structure and hinders multi-hop reasoning (Yu et al., 2025c).
A.3 Agentic RAG
Agentic RAG systems augment standard retrieval-plus-generation pipelines with planning, modular
reasoning, tool usage and self-reflection (Li et al., 2025b; Singh et al., 2025). For instance, CogPlanner
(Yu et al., 2025d) adds an iterative planning module for multimodal RAG, significantly improving accuracy
and efficiency. A-RAG (Du et al., 2026) features hierarchical retrieval interfaces that enable LLMs to
autonomously access corpus information at keyword, sentence, and chunk levels enabling the agent to
adaptively search and retrieve information at different levels of granularity. ARAG (Maragheh et al.,
2025) employs specialized agents (user understanding, NLI inference, context summarization, and item
ranking) to refine retrieval for personalized recommendation, achieving large improvements. MAIN-RAG
(Chang et al., 2025) introduces a training-free multi-agent pipeline that dynamically filters and scores
retrieved documents via agent consensus. RAGentA (Besrour et al., 2025) similarly uses a multi-agent
workflow with hybrid sparse+dense retrieval and inline citations to enhance answer faithfulness. MA-RAG
(Nguyen et al., 2025) decomposes multi-hop QA via distinct planner, extractor, and answer-generation
agents (with chain-of-thought prompting), enabling even small LLMs to match larger models and setting
new SOTA on complex QA. CoT-RAG (Li et al., 2025a) integrates knowledge-graph-driven CoT and
pseudo-program prompts within a RAG pipeline, yielding substantial accuracy gains across diverse
14

reasoning benchmarks. For multimodal QA, HM-RAG (Liu et al., 2025) uses a hierarchical agent stack
(decomposition, multi-source retrieval, decision) over text, graph, and web data. Agent-as-a-Graph (Nizar
et al., 2025) frames agent/tool selection as a knowledge-graph retrieval problem, improving relevant-agent
recall in multi-agent RAG settings.
A.4 RAG Datasets.
RAG QA systems have been evaluated on a range of datasets spanning pure text, knowledge graphs, and
tables. Popular text-only QA benchmarks include open-domain tasks like TriviaQA (Joshi et al., 2017), as
well as multi-hop textual QA such as HotpotQA (Yang et al., 2018) and MuSiQue (Trivedi et al., 2022).
All of the above contain questions which require reasoning across multiple paragraphs, with answers
found in multiple documents (global retrieval). Note that these datasets share a common weakness – are
based on Wikipedia, which means that LLMs were pre-trained on their content.
Separate group of datasets evaluate answering questions about data contained in knowledge graphs.
Spider (Yu et al., 2018) is a cross-domain text-to-SQL data set containing 10,181 questions on 200
databases. Questions require generating SQL against the provided database schema (local context).
WebQuestions (Berant et al., 2013) is a QA dataset with 6 thousand entries using the Freebase knowledge
graph. Questions are answered by querying the entire KG (global context). ComplexWebQuestions
(Talmor and Berant, 2018) is another QA corpus with ∼34K questions combining Web search and
Freebase. Each question is decomposed into subquestions using the full web/KG (global retrieval). The
2WikiMultiHopQA corpus (Ho et al., 2020) integrates structured knowledge by using Wikidata: questions
require combining information from a Wikidata subgraph and associated Wikipedia text.
Another group of datasets focuses on tabular data. Many of them are also based on Wikipedia contents.
KET-QA (Hu et al., 2024) is a knowledge-enhanced table QA dataset containing 9.4 thousand questions
that link each Wikipedia table to a subgraph of Wikidata. Questions require combining table values
with graph facts, with local context per question. Similarly, HybridQA (Chen et al., 2020) is a larger,
hybrid table-text QA dataset (70K examples) where each question is paired with a Wikipedia table and
linked passages. Answers require combining table and text, so each query uses the provided local context.
WikiTableQuestions (Pasupat and Liang, 2015) takes a simpler approach – it is a table QA dataset (22K)
where each question is answered using a single HTML table. It contains no extra text; each query is local
to its table. WikiSQL (Zhong et al., 2017) is another text-to-SQL dataset, containing 80,654 questions
over 24,241 Wikipedia tables. Questions are answered by generating SQL on the local table, no extra text
is provided.
While Wikipedia is by far the dominant source of knowledge for such datasets, it is not the only one.
Quite naturally, the tabular structure of data is most common in financial contexts, resulting in another
class of datasets. For example, FinQA (Chen et al., 2021) is a financial QA dataset with 8.3K questions on
financial reports. Each question comes with a PDF-derived table and accompanying text. Answers require
numeric reasoning over this local context, making this dataset more specialized. TAT-QA (Zhu et al.,
2021) is a financial hybrid QA dataset of 16.5K questions on annual reports. Questions involve both table
and text content from the report and each question uses its associated table and/or text (local context).
In such benchmarks, questions generally require integrating information drawn from a particular table
or graph instance. These datasets are not well suited for evaluating retrieval-augmented generation (RAG)
systems because the questions are inherently context-dependent: they are unambiguously answerable
only given the pre-specified table or graph. Their primary purpose is to assess reasoning and answer
generation conditioned on a provided context, rather than to assess the quality or effectiveness of retrieval
mechanisms.
The gap is not absolute though – recently, some dedicated RAG benchmarks have been introduced.
T2-RAGBench (Strich et al., 2026) is a hybrid financial benchmark (text+tables) of 23,088 QA pairs
designed for RAG evaluation on real-world documents and created by concatenating FinQA, TAT-QA and
ConvFinQA (Chen et al., 2022). While in the source datasets each question is tied to its specific table and
text (local context), here that oracle context isn’t explicitly provided; all examples have been transformed
to require retrieval of that context. In fact, 91.3% of the dataset consists of expert-verified context-
independent questions. However, the dataset does not include questions that draw on multiple sources
15

or demand combining or summarizing information across several documents. Outside of the financial
niche, RAGBench (Friel et al., 2024) offers ∼100K examples of industrial text QA (user manuals), which
explicitly evaluate end-to-end retrieval+generation. It spans multiple domains and assumes a global corpus
(no per-question context).
B Additional Implementation Details
For all experiments, both for our and the baselines methods, we used GPT-OSS-120B (OpenAI, 2025)
as the underlying LLM, Qwen3-Embedding-4B (Zhang et al., 2025) as embedding model and report
results averaged over 5 indexing runs, 3 query-generation runs, and 3 evaluation runs to reduce stochastic
variability.
B.1 Baselines methods
We maintained the default hyperparameter configurations for each baseline method unless otherwise
specified, with modifications documented only when necessary to ensure stable execution or to align with
official implementation guidelines.
Microsoft GraphRAG.(Edge et al., 2025) For the fast variant of GraphRAG, we followed the official
documentation recommendations and configured the chunk size to 100 tokens with a 15-token overlap1.
This configuration ensures optimal performance while maintaining the system’s graph-based retrieval
capabilities.
RAPTOR.(Sarthi et al., 2024) We encountered numerical instability during the Gaussian Mixture
Model clustering phase, which resulted in ill-defined empirical covariance due to singleton or collapsed
samples. Despite converting input data to float64 precision as recommended, the issue persisted. To
address this, we modified several hyperparameters: we increased the chunk size from100to200tokens,
raised the generated summary token limit from 100to150tokens, and expanded the total cluster length
limit from 3500 to10000 tokens. These adjustments provided more robust data for clustering by ensuring
larger, more informative text segments. Additionally, we modified the summarization prompt to"Write a
short (up to {max_tokens} words, can be less, do not count words) summary of the following, including as
many key details as possible: {context}."This change was necessary because the original token-based
limit conflicted with the thinking tokens generated by the language model, causing unpredictable output
lengths. These modifications maintained RAPTOR’s core hierarchical clustering approach while ensuring
stable execution in our experimental setup.
Wikontic.(Chepurova et al., 2025) Wikontic processes the datasets at the level of individual files,
while we study RAG in a “corpus setting”, where evidence can be distributed across documents and
retrieval is performed over the whole corpus at the same time. This distinction matters because file-level
question answering implicitly provides part of the solution, namely which file contains the answer, and is
therefore a much simpler task.
Running WikOntic on our full corpus as a single file is not practical, since it splits text into sen-
tences/chunks and processes them sequentially, which would be time-wise prohibitive at our scale. Instead,
we extract graphs independently for each document (after chunking, as in the other baselines) and then
merge them into a single graph. Merging uses lightweight heuristics for entity matching, such as stemming
and stop-word removal. We do not apply additional canonicalization beyond these heuristics.
Also, the retrieved triples generate a context that goes beyond the limit for gpt-oss-120B of 131k tokens.
Therefore we set a cap on this context at 130k.
LinearRAG(Zhuang et al., 2025),AriGraph(Anokhin et al., 2025),HippoRAG 2(Gutiérrez et al.,
2024, 2025),A-RAG(Du et al., 2026), andTableRAG(Yu et al., 2025b) required no modifications.
B.2 Dataset Preprocessing
The raw dataset we released consists of a set of scraped HTML files and metadata in JSON format. While
this raw representation can support different downstream uses, in our experiments we first applied two
preprocessing stages to convert it into the textualt jand structureds jinputs required by our pipeline.
1https://microsoft.github.io/graphrag/index/methods/#fastgraphrag
16

The first phase ( parse.py ) extracts structured data from HTML. The parsed tables, corre-
sponding to 2327 individual products in total, are stored in a single specs.json file as a list
of dictionaries of attribute:value pairs – note that attributes were often structured, e.g.
‘Display.Resolution(MainDisplay) : 2160×1856(QXGA+)’.
The second pre-processing phase ( dataset.py ) adapts the data to the format required by Dual-
GraphRAG. The resulting processed dataset is organized as a set of separate directories, one per original
HTML file. Each directory contains the corresponding HTML file, a markdown file storing the extracted
textual content, a JSON file for the extracted structured data and a metadata JSON file with file- prefix.
This latter file stores the metadata associated with the original HTML file and lists the derived files
available for that page. In particular, section content points to the extracted text of the page, stored in the
markdown file, while section prescience points to the JSON file containing the extracted structural data.
Either section may be absent if the corresponding data type is unavailable.
Thestructuredpart of the data (JSON files) collects the specifications of all products the specific HTML
page describes, as separate objects. The information is taken from the specs.json file, with simple
processing ensuring a consistent structure of the object, parsing the structured attributes where necessary.
The name, product range name and list of categories are individual top-level fields, while any other entries
are converted into additional rows in a single dictionary collecting the available specifications. Each row
in the dictionary has a key consisting of table section name and entry label separated by a comma. This
consistent structure makes importing the data into any database easier.
Thetextualcontent (markdown files) is extracted heuristically from the HTML, ignoring scripts, CSS
and content-less structural elements, which account for the majority of the original file’s size. Extraction
focuses on the sections that contain useful information in the form of product or feature descriptions, FAQ
entries, etc. The remaining text is mostly irrelevant to the actual products, often repeats on many or all
pages, and includes for example various disclaimers, additional menus, financing options. On average
only about 20% of visible text and meaningful metadata (like title or alt attributes) on each page is relevant
(see Table 7). Some predefined headers, e.g. containing the section of the page a given portion of text is
from, are also inserted in the markdown. The structured information, already stripped of formatting and
other noise is also attached in a plain text form to the markdown file, giving text-only based systems (e.g.
TKG, Microsoft GraphRAG, RAPTOR, etc.) access to all available information.
B.3 SKG: Datalog Preprocessing Rules
The manually defined rules shown in Listing 1 are applied to the SKG as a means of making certain useful
features easily accessible and adjusting data structure. The purpose of each line is shortly explained in the
comments.
B.4 SKG: Schema
The SKG’s schema is presented in Figure 3, with the individual entity types described in Table 5 and
predicates listed in Table 6. This schema is sufficient to describe the imported data, however, to facilitate
SPARQL generation and higher level understanding, additional custom types and predicates can be added
with Datalog rules (see previous section).
B.5 UnWeaver TKG
We generate the TKG as in UnWeaver (Tuora et al., 2026), as it is conceptually simple and efficient
in terms of token usage. UnWeaver uses an LLM to extract a list of untyped entities out of each text
chunk. Relations are not extracted, as this additional layer contributes little to the actual end-to-end QA
performance. Each entity is fully characterized by its LLM-generated name, and a short description. After
the extraction, all the entities are merged by name to reduce cross-chunk information redundancy, which
yields canonicalized entities, with aggregate descriptions. The final result is a bipartite graph index linking
entities with chunks in which they are mentioned. The full architecture of UnWeaver can be seen in Figure
5. In the query phase entities (represented by their descriptions) are used as an intermediate layer for
the purpose of identifying question-relevant chunks, with no LLM-generated content being shown in the
context provided to the LLM for generating the final answer.
17

% Common type for SKG nodes
[?s, a, skgt : SKG_Entity ] :- [?s, a, ?c], FILTER (?c != skgt : UTKG_Entity ) .
% skg : hasFeature maps to Entry in table if entry has value " yes "
[?p, skg : hasFeature , ?f], [?f, rdf :type , skgt : Feature ] :-
[?p, skg : hasSpec , ?s], [?s, skg : inEntry , ?f], [?s, skg : hasValue , skg : yes ] .
% skg : hasFeature maps to skg :8 k_recording_support ( created if not exists )
% if some feature ID in skg : video_recording_resolution contains "8k"
[?p, skg : hasFeature , skg :8 k_recording_support ],
[ skg :8 k_recording_support , rdf :type , skgt : Feature ],
[ skg :8 k_recording_support , skg : hasName , "8K Recording Support "] :- [?p, skg : hasSpec , ?s],
[?s, skg : inEntry , skg : video_recording_resolution ], [?s, skg : hasValue , ?f],
FILTER ( REGEX ( str (?f), "8k")) .
% skg : hasFeature maps to skg :5 g_support ( created if not exists )
% if at least one of the specific 5G features is present
[?p, skg : hasFeature , skg :5 g_support ], [ skg :5 g_support , rdf :type , skgt : Feature ],
[ skg :5 g_support , skg : hasName , "5G Support "] :- [?p, skg : hasSpec , ?s], [?s, skg : hasValue , ? f5g ],
FILTER (? f5g IN ( skg :5 g_sub6_fdd , skg :5 g_sub6_tdd , skg :5 g_sub6_sdl ) ) .
% skg : hasFeature maps to skg :4 g_support ( created if not exists )
% if at least one of the specific 4G features is present
[?p, skg : hasFeature , skg :4 g_support ], [ skg :4 g_support , rdf :type , skgt : Feature ],
[ skg :4 g_support , skg : hasName , "4G Support "] :- [?p, skg : hasSpec , ?s], [?s, skg : hasValue , ? f4g ],
FILTER (? f4g IN ( skg :4 g_lte_fdd , skg :4 g_lte_tdd ) ) .
% Price is a special property and should be easily found
[? product , skg : hasPrice , ? price ] :- [? product , skg : hasSpec , ? spec ],
[? spec , skg : inEntry , ? entry ], [? entry , skg : hasName , " Price "], [? spec , skg : hasValue , ? price ] .
% Products also belong in categories
[?p, skg : belongs , ?c] :- [?p, skg : variantOf , ?pr], [?pr , skg : belongs , ?c] .
Listing 1: Default Datalog rules
Table 5: Description of classes used in the schema.
Class Description
skgt:Category Wide category of products, e.g. skg:smartphones, skg:projectors or
skg:galaxy_z, extracted from URL
skgt:ProductRange and skgt:Product Specific range of products or individual product, e.g. Galaxy S22 5G or
Galaxy S25 FE. Product is used for differentiating variants, e.g. each color
version is an individual product. Where product has no variants there is one
node with both types
skgt:Spec Single specification, an auxiliary node for representing given entry in specifi-
cation table, it is connected to nodes of type Product. The only type without
a name (skg:hasName)
skgt:Section Always connected to specification node, represents name of specification
table section, directly extracted from the header, e.g. skg:storage_memory,
skg:processor, skg:camera. Sectionless entries are grouped in an artificial
default section "Specifications"
skgt:Entry Always connected to specification node, represents name of specification
table entry, e.g. skg:multi_view, skg:mount, skg:battery_type
skgt:Value Always connected to specification node, represents value in given specifica-
tion table section and entry, e.g. skg:40_w, skg:glare_free, skg:included. The
name (skg:hasName) is the literal string from the page, or a fragment of it
resulting from splitting on commas into multiple entries where necessary
skgt:Feature Additional features of a product not directly present in the specification table
but deduced from it using expert-provided Datalog rules, e.g. skg:5g_support,
skg:8k_recording_support. Connected to Product, but not connected to Spec
as it is often deduced from multiple rows
skgt:UTKG_Entity Entity extracted in TKG. May coincide with any other type (except Spec) if
name matches exactly
18

variantOf
hasSpec
hasNamehasFeaturehasPrice
ProductbelongshasName
hasNamehasName
hasV alue
inSectioninEntryhasDim1Spec
hasNameSmartphonesGalaxy S
Galaxy S25 FE
Galaxy S25 FE
Navy 128GB SM
1080 x 2340
(FHD+)hasDim2 Value Entry
Section
Resolution Main
DisplayhasName
DisplayFeature
hasName
5G Support23401080
hasNamehasSpecinEntryhasUnit
hasV alueinSection
SpechasName
hasNumericV alueValue EntryhasName
SectionWeight
122hasName
g122 gGeneral
Feature
hasFeature
Automatically generated
with expert Datalog rulesbelongsCategory
Category
ProductRangehasName hasNumericV alue hasUnit
Value235.00 GBP GBP 235.00
hasPriceFigure 3: Schema of the Symbolic Knowledge Graph (SKG) used in DualGraph.
B.6 Refinement Details
After constructing both the SKG and the TKG, we align their nodes using normalized name matching
(stemming). This relies on a simple heuristic where we cast all the names to lower cases, remove any
leading or trailing whitespace as well as any prepositions like "a" or "the" (stop words removal). Then we
align them using exact name matching.
We then apply an additional refinement step to improve alignment quality by using the aligned entities
from previous step as seed to train a self supervised contrastive aligner. For that we calculate the
embeddings of SKG by finding the largest connected component and running node2vec on it to obtain
the embeddings. For TKG we rely on dense textual embeddings from a pre-trained embedder such as
Qwen3-Embedding-4B. Then we take batches from common nodes of SKG and TKG and minimize
the information noise-contrastive estimation (InfoNCE) loss to train the aligner. Once the aligner is
trained we apply it on the non-common nodes of SKG and TKG to obtain the most likely candidates for
alignment. Those candidates are then evaluated in two independent passes by LLM judge to determine if
the two nodes should be aligned. The unfortunate limitation of this method is that LLM judge can only
make the decision based on the names of proposed nodes since for the SKG nodes we can not offer a
textual description of the node and so to keep the selection process symmetric we chose not to offer the
description for the TKG nodes either.
B.7 LLM as a Judge
TheLaaJevaluation uses gpt-oss-120b as a judge to choose the better out of two competitors. Each pair
of systems is tested across all examples in both permutations of placement. The final score of a system is
the average winrate across all settings and all competitors.
TheLaaJshows a consistent preference for longer answers, as shown in Figure 6.
19

QueryOrchestratorTKG
Symbolic
retrieverSemantic
retriever
SPARQL  queryFinal answer
Generator
SKG TriplesEntities
Text chunks
UserContext
Feedback
LLM modelFigure 4: Overview of the DualGraph querying pipeline. Blue components denote operation modules, green
components denote retrieval functions, yellow components denote data storage, and LLM is depicted in red.
C Additional details about the data
C.1 Layout types
The product pages conform to three rough layouts, which we label A, B, and C (see Fig 7). The A type
is the most general one, occurring across categories, whereas B is reserved for premium smartphones,
and C is mostly used for accessories. B layout contains rich multimedia content, but most importantly
uses a separate webpage (as opposed to a tab) to list specifications, which we use to generate triples for
injection into the knowledge graph. Moreover the pages containing specifications in the B layout do not
conform to the 1 : 1 mapping between products and pages, instead listing multiple variants, and even
multiple product ranges. This requires dedicated scraping logic: dynamically loading all the possible
combinations of parameters on the specification page, to obtain all the information. For C layout there is
no table containing specifications.
It should also be observed that some information (most importantly the price) is often not present on
the product page itself, but instead must be carried from elsewhere (the product list).
C.2 Extracted Data
The documents obtained from the shop page are multimodal, and a full representation of their content
would have to include visual, and video processing. In this work we focus only on textual information,
and structured information contained in the specification. In Table 7 we present the quantitative statistics
about the length of the raw documents, their extracted textual content, and final markdown files (without
the structural parts contained in the tables). It is clear that the pure markdown text, which serves as the
basis of TKG creation is rather limited, and most information is presented using other means (including
the specification tables).
C.3 Question Generation
The process of question generation consisted of several steps: 1) Feature Identification: We first identified
key smartphone attributes commonly considered in purchase decisions. Prior research and market analyses
highlight features such as price, brand, network connectivity (e.g., 5G capability), battery capacity, display
technology, camera quality, memory, and storage as critical factors for consumers. These attributes
formed the foundation for our question criteria. 2) Constraint Definition: For each selected attribute, we
defined concrete constraint values or categories to ensure measurability. Numeric features were given
20

Table 6: Description of predicates used in the schema.
Predicate Description
rdf:type Alias for ’a’, means assigning a given node to a type
skg:hasName Each node type (except for Spec) has a name in natural language, retrieved
directly from the page
skg:variantOf Means that a given Product is a variant of given ProductRange. For products
with no variants this relation is cyclical, as the same node is both Produc-
tRange and Product
skg:belongs Means that a given ProductRange is in a given Category
skg:hasDescription UTKG_Entities have longer, a LLM-provided description in natural language
skg:hasSpec Connects nodes of type skgt:Product and skgt:Spec
skg:inSection Connects nodes of type skgt:Spec and skgt:Section
skg:inEntry Connects nodes of type skgt:Spec and skgt:Entry
skg:hasValue Connects nodes of type skgt:Spec and skgt:Value
skg:hasFeature Connects nodes of type skgt:Product and skgt:Feature
skg:hasNumericValue If a Value could be meaningfully parsed as a number, links the Spec to a
numeric literal containing the result
skg:hasDim# If a Value specifies dimensions (2D or 3D) these point to individual numeric
literals for each dimension
skg:hasUnit If a unit could be deduced or parsed out of the Value (or potentially Entry –
not currently done), this relation points to a string literal specifying the unit.
Units are normalized, eg. ’l’, ’L’ all point to ’l’
Table 7: Textual information in HTML data – statistics for the full dataset.
Processing phase Avg. character count Avg. token count
(min/max) (min/max)
Raw HTML 1,600,592 650,714
(1,248,179/2,728,986) (557,184/991,785)
Textual content 48,735 10,691
(21,599/104,381) (5,124/21,344)
Final extracted MD 8,101 2,035
(364/43,154) (108/10,090)
21

TEXTCHUNKINGENTITY
EXTRACTION
e1 : [...]
e2 : [...]
e3 : [...]
e4 : [...]
e5 : [...]c1c2
CHUNKc3m1m2
name: e3
description: d1
chunk: c2m3
SYNT ACTIC
CANONICALIZA TIONe1e2
description: d1 + d2
chunk: c2 ,c3e3 DESCRIPTION
EMBEDDINGINDEXING
PHASE
LEGEND
DATA
OPERA TION
MODELQUER YING
PHASE
QUESTION EMBEDDING COSINE SIMILARITYe3 : c2 ,c3
e5 : c3
e2 : c2TOP K ENTITIES
CHUNK SELECTION
VIA ENTITY  VOTING
c3 : ...
c2 : ...RAG CONTEXTLLM
QUESTION
ANSWERINGANSWERFigure 5: Architecture of UnWeaver (Figure 1 from (Tuora et al., 2026))
threshold values (e.g., ’under GBP 300’ for price, ’>4000 mAh’ for battery capacity) while categorical
features were specified by exact values (e.g. ’AMOLED display’, ’5G connectivity = Yes’). The constraint
levels were chosen to reflect meaningful consumer thresholds: for instance, price caps correspond to
common budget segments (budget vs. mid-range) and battery capacity cutoffs align with above-average
battery life expectations. 3) Query Formulation: We then composed a set of evaluation questions. Each
query explicitly enumerates all conditions to avoid ambiguity. For example, one formulated question
asks for ’Samsung smartphones under GBP 300 that support 5G, have at least a 4000 mAh battery, and
feature an AMOLED display.’ This multi-faceted structure ensures the question is highly specific—only
devices meeting all stated criteria qualify as answers. Such questions effectively mimic a faceted search
query a consumer might use on an e-commerce site, filtering by brand, price range, and specific features.
The specificity of the constraints makes the expected results clear-cut and reproducible: any researcher
applying the same filters to the same dataset should retrieve an identical set of smartphones.
C.4 More about question categories
To better illustrate the diversity of the benchmark and the complementary strengths of the SKG and TKG,
we provide several representative examples below.
Inverse Queriesrequire retrieving all entities satisfying a given property or relation. Instances of
these are: "Which Samsung phones support wireless charging?" "Which Galaxy models feature a 200MP
wide-angle camera?" "Which Samsung phones support expandable storage via microSD card?"
These questions primarily involve structured attribute lookup and entity filtering over relatively stable
specifications. More nuanced inverse queries of these type are e.g.,: "Which Samsung phones currently
have a removable battery?" "Which Samsung phones are considered water-resistant or waterproof?"
"Which phones are marketed as rugged or MIL-STD compliant?"
Multi-Condition Queriescombine several constraints simultaneously, often requiring compositional
retrieval. Examples of these queries include: "Which Samsung phones under GBP 500 offer a 120Hz
display, 5G, and at least a 5000mAh battery?" "What phones come with 8GB RAM, 256GB storage, and
cost under GBP 600?" "Which Samsung phones support Samsung DeX, have 5G, and are under GBP
1000?"
22

Figure 6: LaaJ decision frequency is dependent on A vs B answer length log-ratio. No contest correspond to cases
where the LLM response cannot be parsed as either A or B.
Some constraints are implicit, fuzzy, or context-dependent: "Which Samsung phones are durable
enough for outdoor work and also support dual SIM?" "Which foldable Samsung phones are still practical
for everyday use and long battery life?".
Here, the system must combine structured specifications with textual or contextual knowledge dis-
tributed across documentation.
Group Comparison Queriesrequire aggregating and contrasting properties across multiple device
families or products, as seen in the following examples: "How do the Galaxy A series and Galaxy S series
differ?" "Compare the features of Galaxy Z Fold vs. Galaxy Z Flip phones." "What changed from Galaxy
Z Flip6 to Galaxy Z Flip7?"
Answers to these typically require synthesizing multiple heterogeneous relations into coherent compar-
ative narratives rather than returning isolated facts. Sometimes, the comparison is grounded in explicit
specifications such as battery size, camera resolution, RAM, or display dimensions.
Reasoning Queriesare the most open-ended category and require combining factual retrieval with
user-oriented inference, e.g., "What’s a good phone for someone who only uses basic apps like YouTube,
Tinder, and WhatsApp?" "Recommend a compact Samsung phone for easy one-handed use." "Which
Samsung phone is best for multitasking and productivity?" "What Samsung phone would you suggest for
someone who values a great display for movies?"
These questions often cannot be answered via direct attribute matching alone. Instead, they require
synthesizing multiple signals (display quality, battery life, ergonomics, portability, productivity features,
etc.) into a recommendation aligned with user intent.
D Full set of results
This appendix reports the complete experimental results supporting the analysis in Section 4. Tables 8–12
provide the full numerical results, while Figures 8–14 visualize the corresponding trends, trade-offs, and
category-level breakdowns.
Table 8 reports the full comparison between DualGraph and the state-of-the-art baselines in terms of
factual correctness, list matching, and pairwise LLM-as-a-judge scores. Table 9 reports the corresponding
full results in terms of token usage, separating indexing and querying costs into prompt, completion, and
total tokens. Together, these tables provide the complete version of the main comparison summarized in
Table 2.
Table 10 reports the full ablation study over different DualGraph variants. It includes both answer-
quality metrics and query-time token usage, complementing the discussion in Section 4.3 (Table 3).
Table 11 separates results on objective and subjective questions, showing how performance changes when
23

(a) Example of an A-layout
 (b) Example of a B-layout
(c) Example of a C-layout
Figure 7: Specification page layout examples
evaluation is restricted to questions with factual ground truth versus recommendation-style questions.
Table 12 reports the pattern ablation for both Router+TKG fallback and SKG+TKG fallback variants,
measuring the contribution of Spec, Feature, Category, and Singular Node patterns to SKG retrieval and
downstream answer quality.
Figures 8–10 show quality–cost trade-offs for the main comparison with baselines. Figure 8 plots
list-matching F1 against querying and indexing token usage, Figure 9 shows the same analysis for factual
correctness, and Figure 10 reports the trade-off for LLM-as-a-judge scores. These figures illustrate the
Pareto behavior of DualGraph relative to other RAG systems.
24

Table 8: Main results: comparison between DualGraph and state-of-the-art baselines in terms of answer quality.
Factual Correctness List Matching LaaJ
F1 Precision Recall F1 Precision Recall
LLM only 0.107 0.136 0.111 0.028 0.046 0.026 0.559
Vector RAG 0.092 0.161 0.085 0.118 0.244 0.106 0.438
Microsoft GraphRAG (Edge et al., 2025) 0.153 0.240 0.152 0.203 0.375 0.188 0.575
Microsoft GraphRAG (fast) (Edge et al., 2025) 0.083 0.130 0.082 0.140 0.205 0.139 0.420
RAPTOR (Sarthi et al., 2024) 0.207 0.301 0.200 0.216 0.373 0.200 0.568
LinearRAG (Zhuang et al., 2025) 0.138 0.197 0.163 0.085 0.172 0.079 0.526
AriGraph (Anokhin et al., 2025) 0.072 0.155 0.069 0.124 0.199 0.119 0.378
HippoRAG 2 (Gutiérrez et al., 2024, 2025) 0.149 0.267 0.166 0.152 0.336 0.125 0.540
Wikontic (Chepurova et al., 2025) 0.1350.3510.107 0.2900.5220.253 0.523
A-RAG (Du et al., 2026) 0.057 0.047 0.102 0.129 0.164 0.124 0.331
TableRAG (Yu et al., 2025a) 0.091 0.123 0.093 0.118 0.237 0.110 0.453
Router with TKG fallback (our) 0.2980.3500.3120.357 0.490 0.344 0.640
SKG with TKG fallback (our)0.293 0.346 0.3090.3720.5180.350 0.644
Table 9: Main results: comparison between DualGraph and state-of-the-art baselines in terms of token usage.
Indexing Querying
Prompt Completion Total Prompt Completion Total
LLM-only0 0 0 128 437 565
VectorRAG0 0 0 2215 221 2436
Microsoft GraphRAG (Edge et al., 2025)120 886 386 60 326 338 181 212 724 10 486 634 11 120
Microsoft GraphRAG (fast) (Edge et al., 2025)22 438 336 6 981 861 29 420 197 11 777 459 12 236
RAPTOR (Sarthi et al., 2024)4 795 069 819 578 5 614 647 2078 300 2378
LinearRAG (Zhuang et al., 2025)0 0 0 7030 511 7541
AriGraph (Anokhin et al., 2025)7 564 910 8 951 382 16 516 292 4397 700 5097
HippoRAG 2 (Gutiérrez et al., 2024, 2025)10 293 782 5 790 019 16 083 801 21 469 748 22 217
Wikontic(Chepurova et al., 2025)364 334 817 55 989 918 420 324 735 133 701 4926 138 627
A-RAG (Du et al., 2026)0 0 0 35 712 672 36 384
TableRAG (Yu et al., 2025a)0 0 0 18 526 2001 20 527
Router with TKG fallback (our)7 257 303 8 800 795 16 058 098 4898 2780 7678
SKG with TKG fallback (our)7 257 303 8 800 795 16 058 098 3813 2989 6802
Figure 11 provides the same quality–cost analysis between the different DualGraph variants, comparing
list matching, factual correctness, and LLM-as-a-judge scores against query-time token usage. Figures 12
and 13 break down performance by question category: Figure 12 compares DualGraph with the baselines,
while Figure 13 compares the internal DualGraph variants. Finally, Figure 14 reports performance
separately for objective and subjective questions, highlighting the additional difficulty of evaluating
subjective recommendation-style queries.
25

0 20 40 60 80 100 120 140
Query Tokens (×103)0.000.050.100.150.200.250.300.35LM (F1)LLM-only
VectorRAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our)(a) Querying token count.
0 100 200 300 400
Index Tokens (×106)0.000.050.100.150.200.250.300.35LM (F1)LLM-only
VectorRAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our) (b) Indexing token count.
Figure 8: Pareto-front comparison between DualGraph and state-of-the-art baselines in terms of list matching (F1)
and querying/indexing token usage.
0 20 40 60 80 100 120 140
Query Tokens (×103)0.000.050.100.150.200.250.30FC (F1)LLM-only
VectorRAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our)
(a) Querying token count.
0 100 200 300 400
Index Tokens (×106)0.000.050.100.150.200.250.30FC (F1)LLM-only
VectorRAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our) (b) Indexing token count.
Figure 9: Pareto-front comparison between DualGraph and state-of-the-art baselines in terms of factual correctness
(F1) and querying/indexing token usage.
0 20 40 60 80 100 120 140
Query Tokens (×103)0.00.10.20.30.40.50.6LaaJLLM-only
VectorRAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our)
(a) Querying token count.
0 100 200 300 400
Index Tokens (×106)0.00.10.20.30.40.50.6LaaJLLM-only
VectorRAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our) (b) Indexing token count.
Figure 10: Pareto-front comparison between DualGraph and state-of-the-art baselines in terms of LLM-as-a-judge
score and querying/indexing token usage.
26

Table 10: Ablation study of DualGraph variants in terms of answer quality and token usage.
Answer Quality Token Usage
Factual Correctness List Matching LaaJ Querying
F1 Precision Recall F1 Precision Recall Prompt Completions Total
SKG only 0.273 0.306 0.288 0.321 0.421 0.300 0.5072431 3040 5471
TKG only 0.140 0.232 0.129 0.136 0.304 0.120 0.3522853 242 3095
SKG concat TKG0.306 0.3540.324 0.367 0.514 0.344 0.5615166 3020 8186
SKG + TKG_fb0.293 0.346 0.3090.372 0.518 0.3500.5233813 2989 6802
Router 0.268 0.301 0.286 0.303 0.389 0.288 0.4853720 2783 6503
Router + TKG_fb0.298 0.350 0.312 0.357 0.490 0.344 0.5284898 2780 7678
Agentic router 0.240 0.219 0.362 0.341 0.485 0.3330.76452 920 2534 55 454
5 10 20 50
Query Tokens (×103)0.000.050.100.150.200.250.300.35LM (F1)
SKG only
TKG only
SKG concat TKG
SKG+TKG_fb
Router
Router+TKG_fb
Agentic router
(a) List matching F1 vs querying token count.
5 10 20 50
Query Tokens (×103)0.000.050.100.150.200.250.30FC (F1)
SKG only
TKG only
SKG concat TKG
SKG+TKG_fb
Router
Router+TKG_fb
Agentic router (b) Factual correctness (F1) vs querying token count.
5 10 20 50
Query Tokens (×103)0.00.10.20.30.40.50.60.7LaaJSKG only
TKG only
SKG concat TKG
SKG+TKG_fb
Router
Router+TKG_fb
Agentic router
(c) LLM-as-a-Judge score vs querying token count.
Figure 11: Pareto-front comparison of different DualGraph variants across answer-quality metrics and query-time
token usage.
27

Group Comparison Inverse Multi-Condition Reasoning
Category0.00.10.20.30.4FC (F1)LLM only
Vector RAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our)
Group Comparison Inverse Multi-Condition Reasoning
Category0.00.10.20.30.40.50.60.7LaaJLLM only
Vector RAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our)
Inverse Multi-Condition Reasoning
Category0.00.10.20.30.4LM (F1)LLM only
Vector RAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our)Figure 12: Factual Correctness F1, List Matching F1, and LLM-as-a-judge scores by question category for
DualGraph and state-of-the-art baselines.
28

Group Comparison Inverse Multi-Condition Reasoning
Category0.00.10.20.30.4FC (F1)SKG only
TKG only
SKG concat TKG
SKG+TKG_fb
Router
Router+TKG_fb
Agentic router
Group Comparison Inverse Multi-Condition Reasoning
Category0.00.20.40.60.8LaaJSKG only
TKG only
SKG concat TKG
SKG+TKG_fb
Router
Router+TKG_fb
Agentic router
Inverse Multi-Condition Reasoning
Category0.00.10.20.30.4LM (F1)SKG only
TKG only
SKG concat TKG
SKG+TKG_fb
Router
Router+TKG_fb
Agentic routerFigure 13: Factual Correctness F1, List Matching F1, and LLM-as-a-judge scores by question category for the
ablation study of different DualGraph variants.
29

Objective Subjective0.00.10.20.3FC (F1)LLM only
Vector RAG
Microsoft GraphRAG
Microsoft GraphRAG (fast)
RAPTOR
LinearRAG
AriGraph
HippoRAG 2
Wikontic
A-RAG
T ableRAG
DualGraph (R+TKG_fb) (our)
DualGraph (SKG+TKG_fb) (our)
Objective Subjective0.00.10.20.30.4LM (F1)
Objective Subjective0.00.20.40.6LaaJFigure 14: Factual Correctness F1, List Matching F1, and LLM-as-a-judge scores separated by question type
(objective vs. subjective).
30

Table 11: Performance on objective and subjective questions for DualGraph and state-of-the-art baselines.
Factual Correctness List Matching LaaJ
F1 Precision Recall F1 Precision Recall
Objective questions (93)
LLM only 0.129 0.160 0.137 0.036 0.049 0.034 0.531
Vector RAG 0.115 0.200 0.106 0.158 0.326 0.142 0.451
Microsoft GraphRAG (Edge et al., 2025) 0.178 0.283 0.174 0.243 0.438 0.223 0.573
Microsoft GraphRAG (fast) (Edge et al., 2025) 0.088 0.139 0.085 0.158 0.216 0.161 0.420
RAPTOR (Sarthi et al., 2024) 0.229 0.344 0.222 0.248 0.432 0.232 0.557
LinearRAG (Zhuang et al., 2025) 0.162 0.232 0.188 0.093 0.183 0.086 0.505
AriGraph (Anokhin et al., 2025) 0.086 0.167 0.083 0.140 0.214 0.135 0.378
HippoRAG 2 (Gutiérrez et al., 2024, 2025) 0.168 0.313 0.184 0.182 0.406 0.148 0.526
Wikontic(Chepurova et al., 2025) 0.158 0.399 0.129 0.349 0.621 0.301 0.554
A-RAG (Du et al., 2026) 0.067 0.055 0.122 0.159 0.184 0.156 0.329
TableRAG (Yu et al., 2025a) 0.104 0.147 0.101 0.139 0.278 0.132 0.441
Router with TKG fallback (our)0.354 0.412 0.370 0.441 0.580 0.4270.653
SKG with TKG fallback (our)0.337 0.395 0.356 0.434 0.573 0.416 0.643
Subjective questions (24)
LLM only 0.022 0.043 0.012 0.006 0.036 0.0030.671
Vector RAG 0.003 0.009 0.001 0.000 0.001 0.000 0.391
Microsoft GraphRAG (Edge et al., 2025) 0.057 0.075 0.068 0.086 0.187 0.084 0.586
Microsoft GraphRAG (fast) (Edge et al., 2025) 0.061 0.096 0.074 0.088 0.172 0.077 0.418
RAPTOR (Sarthi et al., 2024) 0.120 0.137 0.113 0.120 0.197 0.107 0.611
LinearRAG (Zhuang et al., 2025) 0.042 0.060 0.066 0.064 0.139 0.059 0.608
AriGraph (Anokhin et al., 2025) 0.020 0.109 0.013 0.075 0.154 0.069 0.378
HippoRAG 2 (Gutiérrez et al., 2024, 2025) 0.075 0.090 0.096 0.064 0.129 0.056 0.596
Wikontic(Chepurova et al., 2025) 0.048 0.173 0.027 0.118 0.231 0.110 0.403
A-RAG (Du et al., 2026) 0.017 0.016 0.026 0.043 0.110 0.035 0.337
TableRAG (Yu et al., 2025a) 0.039 0.036 0.062 0.053 0.115 0.045 0.497
Router with TKG fallback (our)0.082 0.109 0.086 0.111 0.226 0.099 0.589
SKG with TKG fallback (our)0.124 0.154 0.129 0.188 0.355 0.157 0.649
Table 12: Impact of graph patterns on DualGraph retrieval performance for the Router+TKG fallback and SKG+TKG
fallback variants.
Pattern % SKG Factual Correctness List Matching LaaJ
Spec Feature Category Singular Node Router Actual F1 Precision Recall F1 Precision Recall
DualGraph version: Router + TKG fallback
Y Y Y Y 84.4 41.90.298 0.350 0.3120.357 0.490 0.3440.540
N Y Y Y 83.1 28.9 0.225 0.300 0.232 0.274 0.439 0.256 0.470
Y N Y Y 83.0 40.6 0.274 0.339 0.286 0.346 0.480 0.329 0.534
Y Y N Y 82.9 42.2 0.284 0.339 0.3020.364 0.491 0.3500.531
Y Y Y N 83.5 40.2 0.266 0.324 0.282 0.339 0.475 0.325 0.527
Y N N N 82.6 38.5 0.271 0.331 0.288 0.349 0.484 0.334 0.532
N Y N N 80.9 25.4 0.228 0.303 0.236 0.266 0.422 0.251 0.473
N N Y N 78.3 25.8 0.206 0.283 0.211 0.243 0.399 0.226 0.454
N N N Y 79.2 28.0 0.230 0.296 0.237 0.265 0.407 0.253 0.468
N N N N 76.6 24.1 0.214 0.279 0.223 0.242 0.385 0.229 0.463
DualGraph version: SKG + TKG fallback
Y Y Y Y - 50.2 0.293 0.346 0.309 0.3720.5180.350 0.534
N Y Y Y - 37.7 0.247 0.318 0.254 0.292 0.460 0.272 0.461
Y N Y Y - 48.4 0.299 0.361 0.307 0.361 0.508 0.340 0.521
Y Y N Y - 48.80.317 0.364 0.331 0.3800.5120.3600.538
Y Y Y N - 48.8 0.296 0.354 0.303 0.367 0.516 0.344 0.536
Y N N N - 44.7 0.301 0.351 0.314 0.362 0.493 0.3450.545
N Y N N - 31.3 0.246 0.316 0.254 0.290 0.446 0.272 0.478
N N Y N - 34.7 0.228 0.300 0.234 0.259 0.419 0.243 0.449
N N N Y - 37.0 0.246 0.316 0.257 0.299 0.462 0.287 0.472
N N N N - 29.4 0.230 0.304 0.234 0.253 0.409 0.238 0.459
31