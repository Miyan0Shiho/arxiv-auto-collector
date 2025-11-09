# Relational Deep Dive: Error-Aware Queries Over Unstructured Data

**Authors**: Daren Chao, Kaiwen Chen, Naiqing Guan, Nick Koudas

**Published**: 2025-11-04 16:30:55

**PDF URL**: [http://arxiv.org/pdf/2511.02711v1](http://arxiv.org/pdf/2511.02711v1)

## Abstract
Unstructured data is pervasive, but analytical queries demand structured
representations, creating a significant extraction challenge. Existing methods
like RAG lack schema awareness and struggle with cross-document alignment,
leading to high error rates. We propose ReDD (Relational Deep Dive), a
framework that dynamically discovers query-specific schemas, populates
relational tables, and ensures error-aware extraction with provable guarantees.
ReDD features a two-stage pipeline: (1) Iterative Schema Discovery (ISD)
identifies minimal, joinable schemas tailored to each query, and (2) Tabular
Data Population (TDP) extracts and corrects data using lightweight classifiers
trained on LLM hidden states. A main contribution of ReDD is SCAPE, a
statistically calibrated method for error detection with coverage guarantees,
and SCAPE-HYB, a hybrid approach that optimizes the trade-off between accuracy
and human correction costs. Experiments across diverse datasets demonstrate
ReDD's effectiveness, reducing data extraction errors from up to 30% to below
1% while maintaining high schema completeness (100% recall) and precision.
ReDD's modular design enables fine-grained control over accuracy-cost
trade-offs, making it a robust solution for high-stakes analytical queries over
unstructured corpora.

## Full Text


<!-- PDF content starts -->

Relational Deep Dive: Error-Aware Queries Over Unstructured
Data
Daren Chao
University of Toronto
Toronto, Canada
drchao@cs.toronto.eduKaiwen Chen
University of Toronto
Toronto, Canada
kckevin.chen@mail.utoronto.ca
Naiqing Guan
University of Toronto
Toronto, Canada
naiqing.guan@mail.utoronto.caNick Koudas
University of Toronto
Toronto, Canada
koudas@cs.toronto.edu
ABSTRACT
Unstructured data is pervasive, but analytical queries demand struc-
tured representations, creating a significant extraction challenge.
Existing methods like RAG lack schema awareness and struggle
with cross-document alignment, leading to high error rates. We pro-
pose ReDD (Relational Deep Dive), a framework that dynamically
discovers query-specific schemas, populates relational tables, and
ensures error-aware extraction with provable guarantees. ReDD
features a two-stage pipeline: (1) Iterative Schema Discovery (ISD)
identifies minimal, joinable schemas tailored to each query, and
(2) Tabular Data Population (TDP) extracts and corrects data using
lightweight classifiers trained on LLM hidden states. A main contri-
bution of ReDD is SCAPE, a statistically calibrated method for error
detection with coverage guarantees, and SCAPE-HYB, a hybrid
approach that optimizes the trade-off between accuracy and human
correction costs. Experiments across diverse datasets demonstrate
ReDD’s effectiveness, reducing data extraction errors from up to 30%
to below 1% while maintaining high schema completeness (100%
recall) and precision. ReDD’s modular design enables fine-grained
control over accuracy-cost trade-offs, making it a robust solution
for high-stakes analytical queries over unstructured corpora.
1 INTRODUCTION
In many applications, including healthcare, finance, and engineer-
ing, the vast majority of the data produced is unstructured (in the
form of reports, surveys, clinical trials, etc). Data analytics applica-
tions, however, in these domains require the data to be in a struc-
tured format. Consider, for example, analytical queries on the results
of clinical trials for drug side effects or related queries in a financial
domain to report on the top reasons identified for missed earnings
of public companies in certain sectors. These queries may involve
entity alignment, multi-hop reasoning, and statistical aggregation—
tasks that are particularly difficult in the absence of structured
representations. Structured data is also mandated by the strict accu-
racy requirements in these applications. As a result, unstructured
data has to be processed to yield structured information for further
downstream analytical processing.
Motivating Example.Consider the example in Figure 1, which
depicts a natural language query asking for the average treatment
cost by disease for hospitals in New York in January 2024, alongside
a collection of documents, depicted as document chunks for illustra-
tion purposes. These document chunks vary in focus: some describetreatment details (e.g., costs and diseases) like D1, others provide
hospital metadata (e.g., location) like D2, and some contain patient
profiles, irrelevant to the query, like D3. Answering this query
requires aligning hospital names across document chunks and ag-
gregating costs, a process complicated by the heterogeneous and
fragmented nature of the data. Moreover, the absence of schemas
or explicit semantic information (e.g., entity relationships) further
hinders query answering. These challenges extend across domains-
beyond medical reports to financial filings, legal documents, and
more. Answering query 𝑄in Figure 1 directly from the data in the
document collection is challenging.
Current methods, such as retrieval-augmented generation (RAG)
[13], based on large language models (LLMs), attempt to answer
queries over unstructured data by retrieving top-ranked documents
based on query similarity and generating a response conditioned on
a limited context. This design makes RAG optimized for precision,
but offers limited control over recall—a property that is often more
critical in database-style queries, such as those involving statistical
aggregation and other analytical tasks, across documents as in
Figure 1. Moreover, RAG lacks schema awareness and struggles with
cross-document entity alignment, such as linking hospital names
across documents [ 19,28]. State-of-the-art approaches for text-to-
SQL [ 8] or traditional information extraction techniques [ 20], rely
on predefined schemas or explicit semantic information, which is
generally unavailable or undefined in unstructured corpora.
In extracting value out of unstructured data, frameworks such
asDeepResearch[ 10,12,23] andDeepSearch[ 39] are gaining pop-
ularity. These are agentic frameworks that scan web pages (or
documents) in a query-driven manner, producing detailed sum-
maries and analysis in response to aspecificuser query. Motivated
by such frameworks, we seek to analyze semantic information from
unstructured documents and extractquery-specific structured data.
Moreover, given that our focus is on running analytical queries on
extracted structured data, we are interested in providingerror-aware
query processingwith controllable accuracy—such as the ability to
specify query-time error bounds or adjust accuracy-cost trade-offs.
Relational Deep Dive (ReDD) Overview.In this paper, we present
ReDD, (pronounced ‘ready’) our proposedRelationalDeepDive
framework. We aim for fine-grained accuracy control for the speci-
fied query, bridging the gap between raw documents and structured
query execution.ReDDintegrates schema discovery (if applicable),arXiv:2511.02711v1  [cs.DB]  4 Nov 2025

Daren Chao, Kaiwen Chen, Naiqing Guan, and Nick Koudas
Q   Find the avg treatment cost by disease 
for each hospital in New York, based on 
treatments conducted during January 2024.
D               Example Document Chunks
…D1:Patient P1003 was admitted to the 
Central Hospital on Jan 12, 2024, ...
D2:The Central Hospital situated in the 
bustling heart of New York, is Grade…
D3:Ms. Lee (PT1003) is a 45 -year -old 
female without notable chronic conditi…Tabular Data Population  (TDP)
Treatments (excerpt)
Hospitals (excerpt)Disease Hospital Cost StartDate
Mild C… Central Ho… 3420 2024 -01-12
Ligam… Orthopedi… 7650 2024 -01-04
Name Location
Central Hospital New York
Orthopedic Hospital BostonSQL Generation & Execution
 SELECT AVG(t.Cost) … 
 FROM Treatments t JOIN Hospitals h
 WHERE … GROUP BY …
Query ResultsQuery -specific Schema Discovery  (ISD)
Treatments( Disease, Hospital, Cost, StartDate )
Hospitals( Name, Location )A B
C
Hospital Disease AvgCost
Central Hospital Mild Concussion 4002.37
… … … Error Correction Mechanism
Figure 1: Overview of the query processing pipeline inReDD.The left side of the dashed line shows the raw input, consisting of a natural language
query and a collection of unstructured document chunks. The right side illustrates the core system workflow ofReDD, comprising: (A) schema discovery; (B)
data population; and (C) SQL query generation and execution (not the focus of this work). Within the data population component (B), an error correction
mechanism is integrated to automatically detect and rectify low-confidence extractions, enabling controllable accuracy.
structured data population, with error guarantees, and result syn-
thesis in a cohesive pipelinedriven by the input query.
The pipeline begins with Iterative Schema Discovery (ISD), which
processes the collection of documents at a suitable granularity (re-
ferred to as document chunks), iteratively refines a candidate set of
tables and attributes, given a query 𝑄, as more document chunks
are processed. This procedure identifies the minimal schema re-
quired to answer 𝑄, and uncovers latent semantic structures such as
shared entities (join keys) across document chunks. For example, in
Figure 1(A), ISD discovers two tables—Treatments and Hospitals—
along with attributes both directly relevant to the query (e.g., treat-
ment cost and disease) and indirectly necessary for alignment (e.g.,
hospital name), even if not mentioned in the query itself. Once
the schema is in place,ReDDproceeds to Tabular Data Popula-
tion (TDP). Each document chunk is parsed and converted into
one or more rows across the discovered tables, depending on its
content and relevance to the query 𝑄. As exemplified in Figure 1(B),
chunk D1 populates the first row of the Treatments table, while D2
contributes a row to Hospitals. In contrast, D3 is irrelevant to the
query and therefore does not populate any table. However, due to
the ambiguity of natural language and the inherent variability of
LLM outputs, extraction errors remain common—especially when
attributes are implicit, phrased inconsistently, or missing altogether.
To combat these challenges, we introduce a downstream correction
module that proactively detects and corrects extraction errors, and
supports human-in-the-loop intervention when required.
To enable this functionality,ReDDsupports controlled accuracy.
Rather than treating the LLM as an infallible oracle,ReDDquanti-
fies extraction uncertainty and selectively corrects low-confidence
outputs through lightweight classifiers trained on LLM hidden
states and ensemble strategies. Specifically,ReDDexploits the hid-
den representations (i.e., intermediate layer states) of LLMs, which
encode rich contextual signals. These hidden representations are
used as features for a suite of classifiers that predict extraction
correctness—e.g., incorrect attribute values, wrongly assigned ta-
bles, or missing entries. These classifiers underpin a set of correctionstrategies, including ensemble agreement checks, conformal pre-
diction, and fallback re-extraction. When classifier disagreement
or uncertainty is high, we either reprocess the document chunk or
flag it for human-in-the-loop review. This modular design enables
fine-grained control over the accuracy-efficiency trade-off at query
time as we detail in §4.
In our experiments across multiple domains, we observe that
direct data extraction by LLMs can yield error rates (measured
as the proportion of incorrectly populated rows) of up to 30% on
certain challenging datasets. WithReDD’s correction techniques,
these error rates consistently drop below 1%, without requiring
schema supervision or domain-specific rule engineering. These
gains underscore the effectiveness of our end-to-end framework in
achieving not only high accuracy but also controllable extraction
quality, even over large-scale unstructured document collections.
This paper makes the following contributions:
•We presentReDD, a query-specific framework for structured
query execution over unstructured text. The framework bridges
the gap between raw documents and structured query processing
by dynamically discovering schemas, populating tables, and cor-
recting extraction errors.ReDDachieves this through a cohesive
pipeline comprising Iterative Schema Discovery (ISD), Tabular
Data Population (TDP), and error detection and control, enabling
accurate query answering with structured representations even
in the absence of predefined schemas or annotations.
•We develop a two-stage schema discovery pipeline that con-
structs minimal, joinable tables tailored to each query. Empirical
results demonstrate that this two-phase approach yields more
accurate and query-complete schemas compared to single-phase
alternatives.
•We introduceSCAPE(Spatial Conformal Activation Partitioning
for Errors) a statistically calibrated method that guarantees er-
ror coverage (Theorem 4.1) while asymptotically being optimal
in minimizing human correction costs (Theorem 4.2) by parti-
tioning high-dimensional non-conformity scores (Algorithm 1).
Moreover, we introduceSCAPE-Hyb, a hybrid approach that
integrates conflict-aware signals withSCAPE, maintaining its

Relational Deep Dive: Error-Aware Queries Over Unstructured Data
well calibrated properties (Theorem 4.1) while enabling flexible
trade-offs between error detection recall and correction costs
with provable guarantees (Theorem 4.4).
•We validateReDDon several real-world datasets, showing that
it scales to large unstructured document collections and reduces
query error rates from up to 30% to below 1%. The results un-
derscoreReDD’s effectiveness in transforming unstructured text
into query-ready structured data with provable guarantees.
In §2 we introduce the core components of theReDDframework.
§3-4 present our error management module for data extraction,
followed by §5 presenting our schema discovery methodology. §6
reports experimental results across multiple datasets1. §7 reviews
related work and we present our closing remarks in §8.
2 THEREDDFRAMEWORK
ReDDproceeds in two stages and transforms unstructured textual
data into query-ready structured tables. The system accepts as in-
put a natural language query 𝑞and a collection of unstructured
documents. These documents are segmented into semantically co-
herentchunks, denoted as 𝐷={𝑑 1,𝑑2,...,𝑑𝑛}, where each chunk
𝑑𝑖is a contiguous span of text bounded by semantic discontinuities
(e.g., paragraph breaks). We treat each chunk as the minimal se-
mantic unit of processing. Depending on its content, a chunk may
yield one or multiple rows2distributed across one or more (initially
unknown) tables. The goal ofReDDis to (i) discover the latent
schema𝑆𝑞
𝐷required to answer 𝑞, and (ii) populate that schema
with tuples extracted from 𝐷, supporting accurate and error-aware
query execution.
To answer complex analytical queries over unstructured text,
it is often necessary to recover latent structured representations—
namely, relational tables and their schemas—that are not explicitly
present in the input. In some cases (as exemplified by systems such
as Galois [ 29] and Palimpzest [ 17]), the schema may be known
or provided (see §7 for details). AlthoughReDDhandles this case
naturally, we take a more general and principled approach by en-
abling automated schema discovery. This process is complicated
by several factors:
•The number of tables, their schemas, and the mapping from
document chunks to tables are unknown a priori.
•The query𝑞may require aggregating information distributed
across multiple latent tables to produce a complete answer.
This setting reflects real-world scenarios where no external schema,
entity linking, or domain-specific annotations are available, and the
query-specific structured tables must be discovered dynamically
from text.
ReDDaddresses these challenges through its two-stage pipeline:
Iterative Schema Discovery (ISD) and Tabular Data Population
(TDP), which includes an error correction module (see §3-4). The
entire pipeline is driven by the input query 𝑞, while also mining
latent semantics across chunks to construct accurate, structured
1Our code and datasets are available at: https://github.com/daren996/ReDD.
2For brevity and to ease presentation, in the remainder of this section we assume
that each chunk yields one row for illustration; extending to other scenarios such
as one-to-many and many-to-one mappings is straightforward (§3) and imposes no
changes to the downstream modules. We empirically validate these along with their
trade-offs, in §6.4.1 (one-to-many) and §6.2.1 (many-to-one).outputs. Given a query specific schema 𝑆𝑞
𝐷at hand (a process de-
scribed in §5) we next detail the TDP phase (§3-4) followed by
ISD.
3 TABULAR DATA POPULATION
Given a query-specific schema 𝑆𝑞
𝐷, the Tabular Data Population
(TDP) stage inserts tuples into the schema by extracting structured
data from document chunks. Each chunk 𝑑𝑘is independently pro-
cessed to yield structured rows aligned with the schema semantics.
The TDP stage follows a fixed, two-step pipeline, in which light-
weight, LLM-guided prompt functions extract and format relevant
data3.
•Table ResolverX𝑇. For every chunk 𝑑𝑘, the resolver selects the
most semantically compatible table𝑇 𝑘∈𝑆𝑞
𝐷:
𝑇𝑘=X𝑇 𝑑𝑘,𝑆𝑞
𝐷,for𝑘=1..𝑛(1)
where𝑇𝑘denotes the identifier of the selected target table.
•Attribute Extractor X𝐴. Given the pair(𝑑𝑘,𝑇𝑘), the extractor
iteratively fills each attribute𝑎 𝑖∈𝑇𝑘:
𝑣𝑘,𝑖=X𝐴(𝑑𝑘,𝑎𝑖,𝑇𝑘),for𝑘=1..𝑛, 𝑎 𝑖∈𝑇𝑘 (2)
where𝑣𝑘,𝑖denotes the value of attribute 𝑎𝑖extracted from chunk
𝑑𝑘. The resulting tuple (𝑣𝑘,1,...,𝑣𝑘,𝑚)forms a single row to be
inserted into table𝑇 𝑘.
This procedure extracts, for each chunk, a semantically aligned table
and a complete attribute-value tuple, yielding one structured row
per chunk as the final output of data population. It also generalizes
to one-to-many settings: the table resolver can return multiple
candidate tables per chunk, and the attribute extractor can extract
multiple tuples per table accordingly.
Due to the inherent ambiguity of natural language and the sto-
chastic behavior of LLM outputs, the chunk-wise extraction of at-
tribute values can lead to extraction inconsistencies and errors, such
as incorrect or incomplete attribute values, or even mis-assigned
rows. Relying solely on an LLM in TDP during table population
yields error rates (measured as the proportion of incorrectly popu-
lated rows) of up to 30% on challenging datasets (see §6.2). The TDP
stage, makes local decisions: it processes each document chunk
during data population, one at a time, mapping it to one or more
tables and extracting tuple(s). Lacking a holistic view, TDP cannot
revise earlier decisions based on broader context, making tuple
extraction inherently error-prone. This limitation instigates the
introduction of additional mechanisms to mitigate such errors.
3.1 Error Detection Approaches
While Tabular Data Population (TDP) extracts tuples and populates
query table(s) whose schema is discovered for a query 𝑞during ISD,
it may introduce extraction errors due to the inherent uncertainty
of LLM outputs. In high-stakes settings, even a small number of
erroneous entries may lead to incorrect conclusions in downstream
analyses.ReDDoperation is geared towards a specific query 𝑞and
it includes mechanisms to identify and correct data extraction er-
rors, ensuring that each relational table entry is grounded in the
source document. Central to our approach towards error-free data
extraction is assessing the trustworthiness of each extracted value,
3Concrete prompt templates and implementation details are provided in Appendix A.3.

Daren Chao, Kaiwen Chen, Naiqing Guan, and Nick Koudas
utilizing small, open-weight LLMs. Their compact size allows for
local deployment, while being open-weight enables the develop-
ment of specialized, tunable algorithms around them. Based on this
assessment, we decide when to trigger corrective actions and/or
abstain from extracting the value and trigger a human review to
assist our algorithms with extraction. Thus,ReDDincorporates a
human-in-the-loop as a first-class primitive.
Before TDP commences, we construct a small labeled dataset,
denoted asDcls, which serves as training data for lightweight clas-
sifiers that predict the correctness of extracted values during TDP.
Each entry inDclsis generated by applying the LLM prompts
and using the same LLM model (denoted as 𝑀TDP) as in the TDP
stage (as defined in Equations (1)-(2)) to a limited set of document
chunks, producing candidate table rows. The ground-truth labels
for these entries can be obtained through one of the following two
approaches:
(1)Human-Verified Labeling: A user manually verifies each
extracted value against the source text, annotating whether it
is correct or incorrect.
(2)LLM-Committee-Based Labeling: A committee of powerful
LLMs (e.g., OpenAI-o3 [ 25] and Claude-4 [ 3]) independently
extracts and evaluates the same tuples. The correctness of a
candidate extraction is determined by the consensus of all com-
mittee members: if the initial LLM’s output disagrees with the
committee’s assessment, the committee’s label is taken as the
ground truth, and the extraction is marked as incorrect.
Formally, letM={𝑀 1,...,𝑀𝐾}denote the committee of 𝐾powerful
LLMs. For a candidate tuple 𝑡extracted by the initial LLM 𝑀TDP,
the label𝑦 𝑡is assigned as:
𝑦𝑡=(
1if𝑡matches majority ofMor human confirms,
0otherwise.(3)
This hybrid approach ensures high-quality labeled data while mini-
mizing reliance on manual annotation. The required size of Dclsis
small, as we demonstrate in §6.2 by varying its size and measuring
its impact on extraction accuracy. These classifiers (§3.2) enable
error detection during the extraction process.
Following error detection,ReDDapplies error resolution mea-
sures, such as committee-based inference using more powerful
LLMs, or escalation to human verification for low-confidence cases.
Such error resolution methods incur additional cost (e.g., mone-
tary, time burden, etc), and need to be minimized. We refer to this
overhead ashuman correction costorcostinterchangeably. This
cost is proportional to the number of entries flagged as erroneous,
i.e., those with predicted error label ˆ𝑦=1. Notice that in case error
detection instigatesfalse positives( ˆ𝑦=1but the true label 𝑦=0), this
adds extra human correction cost (such as human validation of
already correct extractions). Thus, although correct error detection
is important (to minimize false negatives), reducing false positives
is a major design requirement as well.
Our approach deploys a collection of lightweight classifiers utiliz-
ing the hidden states of the LLM ( 𝑀TDP) to quantify the correctness
of each token output. We will start by introducing notation and
two baseline methods. The first,MV, utilizes majority voting over
the binary outputs of individual classifiers to quantify token-level
correctness. The second,CF, is a conservative refinement ofMVdesigned to identify strong disagreement among classifier predic-
tions, thereby reducing false positives. We then introduce our main
proposalsSCAPEandSCAPE-Hybin §4.
3.2 Error Detection via Latent Representations
A cornerstone of our approach is a lightweight binary classifier
that determines whether each extracted attribute value is correct,
using the LLM’s own hidden representations. Recent studies have
demonstrated that LLMs’ internal states encode rich information
about the truthfulness of their outputs [ 26], which we exploit to
identify potential extraction errors inReDD.
Leveraging LLM Hidden States.As described in §3, for each
document chunk 𝑑𝑘, the TDP stage first assigns a target table 𝑇𝑘
via the table resolver X𝑇(Equation (1)), and then extracts attribute
values𝑣𝑘,𝑖using the attribute extractor X𝐴(Equation (2)). During
these steps, we have full access to the hidden states of the LLM
𝑀TDP. All LLM outputs (including the assigned table name 𝑇𝑘and
each extracted attribute value 𝑣𝑘,𝑖) are generated as sequences of
output tokens. Letw 𝑘,table =(𝑤1
𝑘,table,...,𝑤𝑚
𝑘,table)be the token
sequence corresponding to the assigned table name 𝑇𝑘, and let
w𝑘,attr-𝑖 =(𝑤1
𝑘,attr-𝑖,...,𝑤𝑚𝑖
𝑘,attr-𝑖)be the token sequence correspond-
ing to the extracted value 𝑣𝑘,𝑖for attribute 𝑎𝑖(the𝑖-th attribute) in
𝑇𝑘. Letℎ(𝑙)(𝑤)denote the hidden state at layer 𝑙corresponding to a
token𝑤. To obtain a compact representation of the LLM extraction
outputs (for both table names and attribute values), we apply mean-
max pooling across the token-level hidden states at each layer, then
concatenate the pooled vectors:
ℎ(𝑙)
𝑘,table=concat
mean
𝑗ℎ(𝑙)(𝑤𝑗
𝑘,table),max
𝑗ℎ(𝑙)(𝑤𝑗
𝑘,table)
,
ℎ(𝑙)
𝑘,attr-𝑖=concat
mean
𝑗ℎ(𝑙)(𝑤𝑗
𝑘,attr-𝑖),max
𝑗ℎ(𝑙)(𝑤𝑗
𝑘,attr-𝑖)
,(4)
whereℎ(𝑙)
𝑘,table∈R2𝑑andℎ(𝑙)
𝑘,attr-𝑖∈R2𝑑denote the layer- 𝑙hidden rep-
resentations for table name 𝑇𝑘and attribute value 𝑣𝑘,𝑖, respectively.
Here,𝑑is the size of the hidden state of the underlying LLM 𝑀TDP.
Since both vectors are computed through identical mean-max pool-
ing operations, share the same dimensionality, and follow the same
procedure to train the per-layer classifiers (introduced in §3.3), we
adopt a unified notation for simplicity. Specifically, we denote each
concatenated hidden representation as ℎ(𝑙)
𝑘,◦, where◦stands for ei-
ther extracted table names or attribute values. This simplification
relaxes notation without loss of generality. In addition, we use 𝑦𝑘,◦
to refer to the ground truth label of each extraction, indicating
whether the extracted item (table or attribute value) is erroneous
or not. A value of 𝑦𝑘,◦=1denotes an error, while 𝑦𝑘,◦=0indicates
correctness.
LetLdenote the set of LLM layers from which hidden states are
extracted. This set may include all layers or a selected subset4, as
recent studies have shown that certain layers encode richer and
more informative signals than others [ 11,21]. The aggregated hid-
den representations obtained using Equation (4) are subsequently
used as input features for binary classifiers that predict whether
the corresponding table assignment or extracted attribute value is
likely to be incorrect. These hidden states capture rich contextual
4When a subset of layers is used to train classifiers for error prediction, the index 𝑙
refers to the𝑙-th element inL, not the original layer number in the LLM.

Relational Deep Dive: Error-Aware Queries Over Unstructured Data
and semantic cues from both the document and the query, making
them an informative signal for spotting inconsistencies or mistakes
in the LLM’s own outputs. The classifiers are trained usingD cls.
Example.Consider the value “Central Hospital” extracted for the
attributeNamein the tableHospitals. The LLM may tokenize this
value into multiple subword tokens as the output, such as “Central”
𝑤1
𝑘,attr-𝑖and “Hospital” 𝑤2
𝑘,attr-𝑖. For each LLM layer 𝑙, we obtain the
corresponding hidden states ℎ(𝑙)(𝑤1
𝑘,attr-𝑖)andℎ(𝑙)(𝑤2
𝑘,attr-𝑖)for all
tokens in the value. We then compute both the mean and max over
these token representations and concatenate the results to obtain
a fixed-length vector ℎ(𝑙)
𝑘,attr-𝑖following Equation (4). The resulting
vectorsℎ(𝑙)
𝑘,attr-𝑖|𝑙∈L are used as input to the per-layer binary
classifiers that estimate whether the extracted value is erroneous.
3.3 Voting Based Methods:MVandCF
To detect extraction errors from TDP, we begin with a straightfor-
ward but effective baseline: performing majority voting on predic-
tions from classifiers trained on hidden states from each layer.
Layer-wise Error Classifiers.For each LLM layer𝑙∈L, we train
a lightweight binary classifier5𝑓(𝑙):R2𝑑→[0,1]to predict if the
extraction associated with hidden state ℎ(𝑙)
𝑘,◦(as in §3.2) is erroneous:
𝜋(𝑙)
𝑘,◦=𝑓(𝑙)
ℎ(𝑙)
𝑘,◦
,(5)
where𝜋(𝑙)
𝑘,◦is the predicted probability of an error. The classifier
per layer is trained independently using binary cross-entropy loss.
To convert probabilities into binary decisions, a fixed threshold 𝜃
is applied:
ˆ𝑦(𝑙)
𝑘,◦=1h
𝜋(𝑙)
𝑘,◦>𝜃i
.(6)
Here,𝜃determines the decision boundary of each classifier. Its
choice is typically set arbitrarily (e.g., 0.5) and ignores classifier-
specific mis-calibrations, as well as classifier dependencies.
Majority Voting (MV).MVaggregates the binary predictions ˆ𝑦(𝑙)
𝑘,◦
(𝑙∈L ) via majority voting. The idea is that if most classifiers agree
an extraction is wrong, it’s likely to be so:
ˆ𝑦MV
𝑘,◦=1"∑︁
𝑙∈Lˆ𝑦(𝑙)
𝑘,◦>|L|
2#
.(7)
This simple rule effectively denoises isolated errors from individual
classifiers by requiring consensus across layers. However, majority
voting offers no explicit control over the false positive and false
negative rates, making it hard to balance detection accuracy and
correction cost as application demands vary.
Conflict Filtering (CF).WhileMVrelies on agreement, classifier
disagreement can also be informative.CFbuilds on this idea by mea-
suring how much conflict exists among the layer-wise predictions.
Intuitively, if different layers disagree about whether an extraction
is erroneous, it likely reflects ambiguity or model hesitation. We
define the conflict score 𝜅as the number of classifiers that disagree
with the majority vote:
𝜅=∑︁
𝑙∈L1n
ˆ𝑦(𝑙)
𝑘,◦≠ˆ𝑦MV
𝑘,◦o
,(8)
5Each classifier is a multilayer perceptron (MLP) with a sigmoid output.An extraction is flagged as potentially erroneous if the conflict score
exceeds a tunable threshold𝜏CF:
ˆ𝑦CF
𝑘,◦=(
ˆ𝑦MV
𝑘,◦,if𝜅<𝜏CF
1,if𝜅≥𝜏CF
Empirically, a higher 𝜅value signifies a high-conflict case and often
corresponds to true errors, makingCFan effective strategy for
reducing false negatives. In contrast, a small 𝜅value signifies low
disagreement and points to more confident decisions. The threshold
𝜏CFcontrols the sensitivity of theCFstrategy: a smaller value flags
potential errors even in low-conflict cases, thereby improving recall
by capturing more true errors thanMV, but at the cost of increased
false positives and correction overhead.
Limitations.Both methods lack a principled way to control the
trade-off between detection accuracy and correction cost. WhileCF
introduces a tunable conflict threshold 𝜏CF, its impact on accuracy
and cost is heuristic, with no calibrated semantics or statistical
guarantees. This motivates the development of a more controlled
algorithm with formal error-rate guarantees to be introduced next.
4ENABLING ERROR DETECTION TRADEOFFS
To address the limitations of the previous methods and enhance
control over the accuracy of error detection and associated error
correction costs, we present two proposals,SCAPEand a hybrid
approachSCAPE-Hyb. Instead of relying on binary classifier pre-
dictions and aggregations thereof to quantify the accuracy of a
prediction, our proposals leverage the continuous activation proba-
bilities of classifiers jointly to quantify uncertainty more precisely.
•SCAPE(§4.1): is a statistically calibrated method that quantifies
prediction uncertainty. Its aim is to reduce false negatives (and
thus undetected errors) but may increase false positives in the
process and thus increase cost (i.e., imposing extra human labour
to check the result of the extraction). It allows adjustment of
correction aggressiveness via a coverage parameter𝛼.
•SCAPE-Hyb(§4.2): enhancesSCAPEwithCF, allowing a more
flexible balance between accuracy and human correction cost.
In essence, these algorithms enable a tradeoff between prediction
accuracy for erroneous data extractions by the LLM and (human)
correction cost, by introducing two key parameters:
•Coverage Threshold 𝛼: Higher values reduce false negatives
(undetected errors) but may increase false positives, leading to
higher correction costs.
•Conflict Weight 𝜆: Controls how strongly the conflict-aware
algorithmCFinfluences correction decisions. A higher 𝜆gives
greater weight toCF, encourages more conservative decisions,
increasing the chance of flagging potential errors (thus reducing
false negatives), but may also lead to more cautious behavior
and a rise in false positives.
These techniques enable precise and cost-aware control during data
extraction. This is essential in applications where undetected errors
are unacceptable.
4.1SCAPE: Spatial Conformal Activation
Partitioning for Errors
TheSCAPEframework introduces a novel approach to uncertainty
quantification by leveraging a high-dimensional non-conformity

Daren Chao, Kaiwen Chen, Naiqing Guan, and Nick Koudas
score space [ 5,35], contrasting with the previously introduced
methods that rely on independent thresholds for binary classifiers
[2,6,30]. Instead of treating each classifier’s decision boundary in
isolation, this technique constructs a multi-dimensional (spatial)
score by combining outputs from classifiers trained on different
layers. The key innovation lies in partitioning this high-dimensional
space into adaptive cells centered around calibration data, which
are then ranked by the empirical ratio of correct to incorrect labels
observed among the calibration samples contained in each cell.
By selecting regions with low concentrations of incorrect labels
(where the classifiers historically perform well), the method gener-
ates more efficient and precise prediction sets while maintaining
guaranteed coverage. This approach avoids rigid thresholding or
weighted aggregation, instead exploiting the richer geometric struc-
ture of the multi-dimensional space to better separate true from
false predictions, yielding smaller and more informative uncertainty
sets. The framework’s flexibility allows it to outperform traditional
conformal prediction, particularly in scenarios where classifiers
provide complementary information across input or output regions.
SCAPEenables coverage (or recall) control of error detection under
mild distributional assumptions, ensuring a specified proportion of
true errors is identified with high probability.
Non-Conformity Vectors.For each extracted item (either a table
name or an attribute value) with hidden representations {ℎ(𝑙)
𝑘,◦}𝑙∈L
(as per §3.2), we first obtain the sigmoid outputs 𝜋(𝑙)
𝑘,◦=𝑓(𝑙)(ℎ(𝑙)
𝑘,◦)∈
[0,1]from each layer-wise classifier 𝑓(𝑙)at layer𝑙∈L (as detailed
in Equation (5)). We then define amulti-dimensional non-conformity
vectors(𝑐)∈R|L|for each candidate label 𝑐∈{0,1}(where𝑐=1de-
notesw erroneous extraction and 𝑐=0denotes correct extraction)
as:
s(𝑐)=h
𝑠𝑙(𝑐)iL
𝑙=1,where𝑠𝑙(𝑐)=(
1−𝜋(𝑙)
𝑘,◦if𝑐=1,
𝜋(𝑙)
𝑘,◦if𝑐=0.(9)
which reflects how atypical the outputs of the layer-specific error
classifiers are, under the assumption that the extraction is either
erroneous or correct.
Cell Construction and Selection.We leverage a labeled dataset
Dcal-base ={(𝑥𝑖,𝑦𝑖)}𝑁cal-base
𝑖=1, constructed using the same procedure
as in §3.1, where each 𝑥𝑖denotes an extracted item and 𝑦𝑖∈{0,1}
indicates whether the extraction is erroneous (1) or correct (0).6
We randomly split this dataset into two disjoint subsets,D celland
Dre-cal, which serve distinct purposes in the calibration procedure.
• D cell={(𝑥𝑖,𝑦𝑖)}𝑁cell
𝑖=1used to construct cells in the non-conformity
score space by applying 𝑘-means clustering to the score vectors
s(ˆ𝑦𝑖)𝑁cell
𝑖=1, producing 𝐾clusters. Each cluster defines a cell, yield-
ing a partition of the score space into non-overlapping regions
𝐶1,...,𝐶𝐾⊂R|L|, where any new score vector can be assigned
to one of the cells by finding its nearest cluster centroid in Eu-
clidean space.
• D re-cal={(𝑥𝑗,𝑦𝑗)}𝑁re-cal
𝑗=1used to select cells for coverage.
6Dcal-base is a small, user-curated labeled dataset, generated by applying the LLM
prompts used in TDP stage (as Equations (1)-(2)) to a small number of document
chunks. The dataset size is varied in the experiments to study its impact on overall
accuracy in §6.2.Each cell𝐶𝑚groups similar non-conformity patterns. To identify
the most reliable regions of the score space for detecting true ex-
traction errors, we rank cells based on theirfalse-to-true ratioon
Dre-cal, that is, for each cell 𝐶𝑚, and for𝑐∈{0,1}the number of
examples with 𝑦𝑗=𝑐whose score vectorss (1−𝑐)fall into the cell
𝐶𝑚(i.e., false examples), divided by the number of examples 𝑦𝑗=𝑐
whoses(𝑐)also fall into the cell (i.e., true examples):
𝜌𝑚=Í
𝑐∈{0,1}|(𝑥𝑗,𝑐)∈D re-cal :s(1−𝑐)∈𝐶 𝑚|
Í
𝑐∈{0,1}|(𝑥𝑗,𝑐)∈D re-cal :s(𝑐)∈𝐶 𝑚|.(10)
We rank the cells in ascending order of 𝜌𝑚, prioritizing those where
non-error examples are least likely to be mistaken as errors.
To guarantee the desired coverage level, we select the small-
est set of top-ranked cells such that the score vectors of the true
labels for at least ⌈(1−𝛼)(𝑁 re-cal+1)⌉examples inDre-cal fall
within the selected cells. Formally, let all cells bere-indexedas
𝐶(1),𝐶(2),...,𝐶(𝐾), sorted in order of increasing false-to-true ratio.
We define the selected region asC 𝛼⊂R|L|:
C𝛼=𝜂∗Ø
𝑗=1𝐶(𝑗),where𝜂∗=minn
𝜂∈1..𝐾|
(𝑥𝑗,𝑦𝑗)∈D re-cal :
s(𝑦𝑗)∈C𝛼	≥
(1−𝛼)(𝑁 re-cal+1)o (11)
where𝛼∈(0,1)is the user-specified miscoverage tolerance.
Test-Time Inference.At test time, for each new extraction with
hidden representations {ℎ(𝑙)
𝑘,◦}𝑙∈L(as per §3.2), we compute the
non-conformity vectorss (𝑦)for both possible labels 𝑦∈{0,1}(via
Equation (9)), and define the conformal prediction set as:
ˆ𝑦SCAPE
𝑘,◦={𝑦∈{0,1}|s(𝑦)∈C 𝛼}.(12)
By evaluating both candidate labels, we construct a prediction set
that includes multiple labels only when necessary to satisfy the
desired𝛼-coverage for error detection, while keeping the set as
small as possible to reduce correction cost. If ˆ𝑦SCAPE
𝑘,◦={0}, we accept
the extraction as correct. Otherwise, if the prediction set contains 1
(i.e.,{1}or{0,1}), the extraction is flagged for potential error and
triggers a correction step, typically by routing the value for human
verification and correction.
SCAPEis presented as Algorithm 1, where line 1 corresponds
to the computation of non-conformity vectors as defined in Equa-
tion (9), lines 2-3 implement the clustering and ranking procedure
described in Equation (10), line 4 selects cells according to the cov-
erage constraint in Equation (11), and line 5 defines the prediction
setˆ𝑦SCAPE
𝑘,◦for each new extraction as Equation (12).
Coverage Guarantee.For any erroneous extraction (i.e.,𝑦 𝑘,◦=1),
Theorem4.1 (Coverage Guarantee under Exchangeability).Un-
der the assumption that calibration and test examples are exchange-
able, the conformal prediction set determined bySCAPEsatisfies:
P
𝑦𝑘,◦∈ˆ𝑦SCAPE
𝑘,◦
≥1−𝛼,
where ˆ𝑦SCAPE
𝑘,◦={𝑦∈{0,1}|s(𝑦)∈C 𝛼}.
Proof.Assume the test example is exchangeable with the ele-
ments inDerr
re-cal. Each example is assigned a non-conformity score
s(1)(per Equation (9)) and mapped into a cell 𝐶(𝑗)among𝐾pre-
defined, ranked cells. Let 𝑅𝑖be the rank index of the lowest-ranked
cell containings 𝑖(1), and𝑅testbe the corresponding rank for the test
point.

Relational Deep Dive: Error-Aware Queries Over Unstructured Data
Algorithm 1:Spatial Conformal Activation Partitioning
for Errors (SCAPE)
Require:Calibration datasetD cal-base ={(𝑥𝑖,𝑦𝑖)}𝑁cal-base
𝑖=1, split
intoD cellsandD re-cal; Coverage level𝛼∈(0,1);
Ensure:Error prediction set ˆ𝑦SCAPE
𝑘,◦⊆{0,1};
1:Compute non-conformity vectors using Eq. (9), for all
(𝑥𝑖,𝑦𝑖)∈D cal-base .
2:Partition non-conformity space using𝑘-means clustering onD cellto
obtain cells𝐶 1,...,𝐶𝐾.
3:Rank cells in ascending order of false-to-true ratio 𝜌𝑚computed using
Eq. (10):𝐶(1),𝐶(2),...,𝐶(𝐾).
4:Select top-ranked cellsC 𝛼through Eq. (11).
5:For new extraction: ˆ𝑦SCAPE
𝑘,◦={𝑦∈{0,1}:s(𝑦)∈C 𝛼}.
The regionC𝛼is constructed by selecting the top 𝜂∗cells such that
at least⌈(1−𝛼)(𝑁 re-cal+1)⌉calibration errors fall within them, as
per Equation (11),
𝑁re-cal∑︁
𝑖=11{𝑅𝑖≤𝜂∗}≥⌈(1−𝛼)(𝑁 re-cal+1)⌉.
By exchangeability, 𝑅testis uniformly distributed among { 𝑅1,...,𝑅𝑁,
𝑅test}, so
P(𝑅 test≤𝜂∗)≥1−𝛼.
Hence,P(s test(1)∈C𝛼)≥1−𝛼, which implies
P
𝑦test∈ˆ𝑦SCAPE
test|𝑦test=1
≥1−𝛼.
□
Set Size Optimality.Theorem 4.2 below establishes thatSCAPE
achievesasymptotic optimalityin prediction set size under a mixture
model assumption. This guarantees that, as the calibration data
grows (|D cal-base|→∞), the method:
•Minimizes the expected number of extractions flagged for human
review (E[| ˆ𝑦SCAPE
𝑘,◦|]),
•While maintaining the desired error coverage (1−𝛼).
The full proof, which leverages the Neyman-Pearson lemma
[22,30] to show that ranking cells by false-to-true ratio 𝜌𝑚(as
Equation (10)) is equivalent to optimizing the likelihood ratio Λ(s),
is provided below.
Theorem4.2 (Optimal Set Size ofSCAPE).Assume that for each
𝑐∈{0,1}the label-conditional densities 𝑝(s(𝑐)|𝑦= 0)and𝑝(s(𝑐)|
𝑦=1)exist and are continuous ( 𝑦represents the true label). Then,
SCAPEasymptotically minimizes the expected prediction set size
E[|ˆ𝑦SCAPE
𝑘,◦|]subject to coverage≥1−𝛼.
Proof.We treat error detection as a binary hypothesis test 𝐻0:𝑦=0
vs.𝐻1:𝑦=1, based on the observations (0),s(1)∈R|L|. For each
extracted item, the prediction set ˆ𝑦SCAPE
𝑘,◦is constructed by checking
whether each candidate score vectors(𝑦)lies in a selected regionC 𝛼
of the score space.
To constructC𝛼, we partition the score space into disjoint cells
{𝐶𝑚}𝐾
𝑚=1and compute the empirical false-to-true ratio 𝜌𝑚in each
cell using the definition in Equation 10. As the calibration set sizegrows,𝜌𝑚converges to a population-level quantity:
lim
|Dre-cal|→∞𝜌𝑚→∫
𝐶𝑚[𝑝(s(0)|𝑦=1)+𝑝(s(1)|𝑦=0) ]𝑑s
∫
𝐶𝑚[𝑝(s(0)|𝑦=0)+𝑝(s(1)|𝑦=1) ]𝑑s.
Motivated by the Neyman-Pearson principle [ 22,30], we define the
likelihood ratio for cell𝐶 𝑚as:
Λ(𝐶𝑚):=∫
𝐶𝑚[𝑝(s(0)|𝑦=0)+𝑝(s(1)|𝑦=1) ]𝑑s
∫
𝐶𝑚[𝑝(s(0)|𝑦=1)+𝑝(s(1)|𝑦=0) ]𝑑s=1
𝜌𝑚.
This likelihood ratio captures the total correct classification mass
over misclassification mass in each cell, across both classes. Rank-
ing cells by increasing 𝜌𝑚is therefore asymptotically equivalent to
ranking them by decreasingΛ(𝐶 𝑚).
SCAPEselects the smallest set of top-ranked cells C𝛼such that the
score vectors of at least ⌈(1−𝛼)(𝑁 re-cal+1)⌉true errors are covered.
The prediction set for a new point includes label 𝑦if and only if
s(𝑦)∈C𝛼.
The expected prediction set size is:
Ehˆ𝑦SCAPE
𝑘,◦i
=1+P(s(1)∈C𝛼|𝑦=0),
which is minimized when C𝛼contains cells with the highest Λ, i.e.,
lowest𝜌𝑚, while satisfying the coverage constraint. Hence, under
mild assumptions,SCAPEasymptotically minimizes expected set size
subject to valid coverage.□
Limitations.Theorem 4.2 establishes optimality under the assump-
tion of a monotonic likelihood ratio between Λ(s)and𝜌. In practice,
this requires the classifier outputs to be well-calibrated, which is
challenging as we wish to keep | Dcal-base | very small. Moreover,
finite-sample effects may lead to marginal under-coverage when
|Dcells|is small. Our empirical results demonstrate that extractions
with high disagreement among layer-wise classifiers are more likely
to be erroneous. The raw non-conformity scores inSCAPE(based
on classifier probabilities) may not fully capture this disagreement.
Metrics like 𝜅—the number of layers disagreeing with the majority
vote—provide an orthogonal signal that improves error detection.
We thus extendSCAPEbelow to capture such conflict.
4.2SCAPE-Hyb: Hybrid Method
We now introduceSCAPE-Hyb, a unified method that incorporates
the inter-layer conflict signal (§3.3) into the conformal prediction
framework (§4.1). The key idea is to augment the multi-dimensional
non-conformity vector with a scaled conflict term, allowing the
conformal predictor to respond not only to probabilistic uncertainty
but also to internal disagreement among classifiers.
Conflict-Augmented Non-Conformity.The conflict score 𝜅used
in CF (Equation (8)) captures binary disagreement among classifiers,
but tends to be, after probability thresholding per classifier, small
or even zero; this happens even when the probability outputs of
the classifiers vary significantly as they are “smoothed” by thresh-
olding. Thus the value of 𝜅itself is not a very informative signal.
To address this, we replace 𝜅with a more granular, real-valued
disagreement score Δ. Let𝜋(𝑙)denote the predicted probability of
erroneous extraction from layer 𝑙, and let ¯𝜋=1
|L|Í
𝑙∈L𝜋(𝑙)be the
average probability across layers. We define the disagreement score
as:Δ=max𝑙∈L|𝜋(𝑙)−¯𝜋|, which reflects the maximum deviation

Daren Chao, Kaiwen Chen, Naiqing Guan, and Nick Koudas
from consensus among the classifiers. We then embed it as an ad-
ditional dimension into the non-conformity vector by defining a
conflict-calibrated representation:
s𝜆(𝑐)=h
(1−𝜆)·𝑠 1(𝑐),...,(1−𝜆)·𝑠 |L|(𝑐),𝜆·Δi
,(13)
where each 𝑠𝑙(𝑐)is the layer-wise non-conformity score defined in
§4.1, and𝜆∈[0,1]is a tunable parameter that adjusts the relative
weight of the disagreement signal inside the conformal prediction
pipeline. When 𝜆=0, the method falls back toSCAPE(§4.1); on
the other hand, when 𝜆=1, the method ignores all layer-wise non-
conformity scores and relies exclusively on the disagreement score
Δ, effectively acting as a calibrated variant of conflict filtering (§3.3).
Calibration and Prediction.Following the calibration protocol de-
scribed in §4.1, we compute the conflict-augmented non-conformity
vectors at the given 𝜆-weighting for all examples in the calibration
dataset and partition the non-conformity space into cells. We rank
the cells through the conflict-augmented false-to-true ratio 𝜌𝜆
𝑚and
retain only the cellsC𝜆
𝛼that satisfy the conformal coverage condi-
tion at level1−𝛼(see Equation (11)). At test time, the conformal
prediction set is defined as:
ˆ𝑦Hyb
𝑘,◦=
𝑦∈{0,1}|s 𝜆(𝑦)∈C𝜆
𝛼	
,
As before, we treat the extraction as correct if ˆ𝑦Hyb
𝑘,◦=0, and trigger
correction if the set includes 1 or both labels. It is easy to see that
the conformal guarantee (Theorem 4.1) still holds. This is because
the augmented scores 𝜆(𝑐)retains exchangeability, and the cell
selection process is still thresholding a well-defined statistic.
Conflict-Aware Optimality.SCAPE-Hybextends the optimality
guarantee ofSCAPE(Theorem 4.2) to conflict-augmented non-
conformity scoress 𝜆. Following the same principle asSCAPE, it
ranks cells in the score space by their empirical false-to-true ratio,
which remains proportional to the inverse of the likelihood ratio. By
selecting the smallest set of cells C𝜆
𝛼that satisfies the user-specified
coverage constraint (1 −𝛼),SCAPE-Hybasymptotically minimizes
the expected prediction set size.
Theorem4.3 (Optimality ofSCAPE-Hyb).Lets 𝜆(𝑐)∈R|L|+1
denote the conflict-augmented non-conformity score defined in Equa-
tion (13). Suppose the class-conditional densities 𝑝(s𝜆(𝑐)|𝑦) exist and
are continuous for 𝑦∈{ 0,1}. Then, under the exchangeability and reg-
ularity assumptions of Theorem 4.2, theSCAPE-Hybmethod asymp-
totically minimizes the expected prediction set size E[|ˆ𝑦SCAPE-Hyb
𝑘,◦|]
among all predictors satisfying class-conditional coverage ≥1−𝛼for
erroneous extractions (𝑦=1).
Proof.SCAPE-Hybaugments the original non-conformity vector
s(𝑐)∈R|L|with a real-valued conflict signal Δ∈R to form the
augmented vectors 𝜆(𝑐)∈R|L|+1, as defined in Equation (13).
As inSCAPE, theSCAPE-Hybmethod partitions the augmented
score space into disjoint cells {𝐶𝜆
𝑚}𝐾
𝑚=1and ranks them using the
empirical symmetric false-to-true ratio 𝜌𝜆
𝑚, defined analogously to
Equation (10) but over the extended representation.
As the calibration size grows, this ratio converges to:
lim
|Dre-cal|→∞𝜌𝜆
𝑚→∫
𝐶𝜆𝑚[𝑝(s𝜆(0)|𝑦=1)+𝑝(s 𝜆(1)|𝑦=0)]𝑑s
∫
𝐶𝜆𝑚[𝑝(s𝜆(0)|𝑦=0)+𝑝(s 𝜆(1)|𝑦=1)]𝑑s.We define the symmetric likelihood ratio for cell𝐶𝜆
𝑚as:
Λ𝜆(𝐶𝜆
𝑚):=∫
𝐶𝜆𝑚[𝑝(s𝜆(0)|𝑦=0)+𝑝(s 𝜆(1)|𝑦=1)]𝑑s
∫
𝐶𝜆𝑚[𝑝(s𝜆(0)|𝑦=1)+𝑝(s 𝜆(1)|𝑦=0)]𝑑s=1
𝜌𝜆𝑚.
This generalizes theSCAPElikelihood ratio to the conflict-augmented
score space. Ranking cells by 𝜌𝜆
𝑚thus corresponds asymptotically to
ranking by decreasingΛ𝜆(𝐶𝜆
𝑚).
SCAPE-Hybselects the smallest set of top-ranked cells C𝜆
𝛼to satisfy
the desired coverage level. As in Theorem 4.2, the expected prediction
set size is minimized by including only the most reliable cells (with
highestΛ𝜆or lowest𝜌𝜆
𝑚), while ensuring coverage.
Hence,SCAPE-Hybasymptotically minimizes expected prediction
set size under the coverage constraint, completing the proof.□
Advantage ofSCAPE-Hyb.As long as the conflict score pro-
vides an additional discriminatory signal—specifically, if erroneous
outputs tend to have higher conflict than correct ones—the hybrid
methodSCAPE-Hybcan yield smaller expected prediction sets (thus
reducing human correction cost in expectation), while preserving
the same coverage.
Theorem4.4 (Optimality ofSCAPE-HyboverSCAPE).Assume
that the conflict score provides an additional signal for distinguishing
erroneous from correct extractions, i.e., erroneous examples ( 𝑦=1) are
more likely to have a higher conflict score than correct ones. Then,
under the same(1−𝛼) coverage constraint,SCAPE-Hybproduces a
prediction set with equal or smaller expected size compared toSCAPE:
Ehˆ𝑦Hyb
𝑘,◦i
≤Ehˆ𝑦SCAPE
𝑘,◦i
.
Assumption4.5 (Disagreement Signal Consistency).The dis-
agreement score Δis more likely to take high values under erroneous
extractions (true label 𝑦=1) than under correct ones. In other words,
the conditional density ofΔis higher under𝑦=1than under𝑦=0:
𝑝(Δ|𝑦=1)
𝑝(Δ|𝑦=0)≥1.
A detailed formal proof of Theorem 4.4 is provided below.
Proof.We begin with the following definitions. Let 𝑑=|L| , and
define the original non-conformity vector space used bySCAPEas
V=R𝑑. Define the augmented space used bySCAPE-Hybas V′=R𝑑+1.
Define a projection operator𝛿:V′→Vas:
𝛿([𝑠1,...,𝑠𝑑,𝑠𝑑+1])=[𝑠1,...,𝑠𝑑].
1. Neyman-Pearson Optimality (SCAPE). Based on the Neyman-Pearson
lemma [ 22], the optimal acceptance region forSCAPEin Vis given
by:
𝑅SCAPE={s∈V:Λ(s)>𝑡 SCAPE},
where the likelihood ratio is Λ(s)=𝑝( s(1)|𝑦=1)/𝑝(s(1)|𝑦=0). The
threshold𝑡 SCAPE is chosen to satisfy:
Pr
𝑦=1[s(1)∈𝑅 SCAPE]=1−𝛼.
2. Extended Acceptance Region (SCAPE). Define the inverse projection
of𝑅 SCAPE in the extended spaceV′:
e𝑅SCAPE=𝛿−1(𝑅SCAPE)={s′∈V′:𝛿(s′)∈𝑅 SCAPE},

Relational Deep Dive: Error-Aware Queries Over Unstructured Data
This set does not differentiate values in the added (𝑑+1)-th dimension,
thus preserving coverage and false-positive rate:
Pr
𝑦=1[s′(1)∈e𝑅SCAPE]=Pr
𝑦=1[s(1)∈𝑅 SCAPE]=1−𝛼,
Pr
𝑦=0[s′(0)∈e𝑅SCAPE]=Pr
𝑦=0[s(0)∈𝑅 SCAPE].
3. Likelihood Ratio Advantage (SCAPE-Hyb). In the extended space
V′, under the Assumption 4.5, the likelihood ratio function Λ𝜆of
SCAPE-Hybsatisfies:
Λ𝜆(s𝜆)=𝑝(s𝜆(1)|𝑦=1)+𝑝(s 𝜆(0)|𝑦=0)
𝑝(s𝜆(1)|𝑦=0)+𝑝(s 𝜆(0)|𝑦=1)
=𝑝(s(1)|𝑦=1)·𝑝(Δ|𝑦=1)+𝑝(s(0)|𝑦=0)·𝑝(Δ|𝑦=0)
𝑝(s(1)|𝑦=0)·𝑝(Δ|𝑦=0)+𝑝(s(0)|𝑦=1)·𝑝(Δ|𝑦=1)
≥𝑝(s(1)|𝑦=1)+𝑝(s(0)|𝑦=0)
𝑝(s(1)|𝑦=0)+𝑝(s(0)|𝑦=1)=Λ(𝛿(s𝜆)),
Thus, using the same threshold𝑡 SCAPE , we have:
Pr
𝑦=1[Λ𝜆(s𝜆)>𝑡 SCAPE]≥Pr
𝑦=1[Λ(𝛿(s𝜆))>𝑡 SCAPE]=1−𝛼.
4. Coverage Matching (SCAPE-Hyb). To achieve exact (1−𝛼) coverage,
we select𝑡 Hyb≥𝑡SCAPE such that:
Pr
𝑦=1
Λ𝜆(s𝜆)>𝑡 Hyb
=1−𝛼.
Since increasing the threshold reduces the coverage set and the cover-
age rate decreases monotonically, there must exist such a𝑡 Hyb.
5. Acceptance Region Comparison. Define the acceptance region for
SCAPE-Hybwith threshold𝑡 Hyb:
𝑅Hyb={s′∈S′:Λ𝜆(s′)>𝑡 Hyb}.
Given𝑡 Hyb≥𝑡SCAPE andΛ𝜆≥Λ◦𝛿, for anys′∈V′:
Λ𝜆(s′)>𝑡 Hyb=⇒Λ𝜆(s′)>𝑡 SCAPE =⇒Λ(𝛿(s′))>𝑡 SCAPE,
Thus,
𝑅Hyb⊆{s′:Λ(𝛿(s′))>𝑡 SCAPE}=𝛿−1(𝑅SCAPE).
6. Expected Set Size Advantage. The originalSCAPEmethod defines
its acceptance region 𝑅SCAPE⊆V in the original non-conformity
space. Its corresponding inverse projection (or cylindrical lift) in the
extended space is:
𝛿−1(𝑅SCAPE)={s′∈V′:𝛿(s′)∈𝑅 SCAPE},
which retains the same false positive rate as 𝑅SCAPE , since the addi-
tional coordinate has no effect on the classifier outputs:
Pr
𝑦=0[s′∈𝛿−1(𝑅SCAPE)]=Pr
𝑦=0[s∈𝑅 SCAPE].
In contrast, the acceptance region 𝑅Hybused bySCAPE-Hybis a strict
subset of this lifted region, as established in Step 5:
𝑅Hyb⊆𝛿−1(𝑅SCAPE).
Therefore, it must hold that the false positive rate ofSCAPE-Hybis at
most that ofSCAPE:
Pr
𝑦=0[s′∈𝑅 Hyb]≤Pr
𝑦=0[s∈𝑅 SCAPE],
which implies directly:
Ehˆ𝑦Hyb
𝑘,◦i
≤Ehˆ𝑦SCAPE
𝑘,◦i
.
□Our experimental results (§6) empirically corroborate this theo-
retical advantage.
Besides,SCAPE-Hybcan be viewed as asoftcompatible general-
ization of conflict filtering: instead of enforcing hard thresholds, the
conflict score is smoothly embedded into the non-conformity space
and calibrated within the conformal framework. In this way, the
hybrid method preserves a key advantage ofCF—its ability to im-
prove recall at relatively low cost—while avoiding brittle thresholds
and retaining the formal coverage guarantees ofSCAPE. The con-
tinuous weighting of conflict also gives practitioners finer control
over the accuracy-cost trade-off.
Summary.The parameter 𝜆controls the relative influence of con-
flict: higher values give more weight to conflict, approachingCF
behavior as 𝜆→ 1; lower values emphasize the original confor-
mal score. Low values in 𝜆prioritize probabilistic uncertainty and
are better for well-calibrated classifiers (when | Dcls| is large). High
values of𝜆prioritize conflict, which is better when layer disagree-
ments correlate with errors and as we empirically demonstrate in
§6.2 when |Dcls| is small. The optimal setting can be selected via
grid search on a validation set. In practice, intermediate values (e.g.,
𝜆=0.5) often yield a good trade-off between error detection recall
and extra correction cost, as demonstrated in §6.2.
5 ITERATIVE SCHEMA DISCOVERY
The goal of this stage is to derive a relational schema 𝑆𝑞
𝐷over a
collection of document chunks 𝐷that contains exactly the entities
(attributes, such as names and locations) and relationships required
to answer the query 𝑞. Schema discovery occurs dynamically in
two phases. Phase I induces a general (query-agnostic) schema
𝑆gen
𝐷, that captures all salient attributes and relationships present in
the documents, independently of any specific query. Phase II then
adapts this into a query-specific schema 𝑆𝑞
𝐷tailored explicitly to
the requirements of query𝑞.
Phase I: General Schema Discovery.We treat schema discovery
as an iterative process of reading and abstraction. It begins with an
empty schema and processes the document chunks in a sequential,
one-pass manner. At each step, it incrementally revises the current
schema state. As new information becomes available, each step
may refine earlier decisions by revising the previously constructed
schema state. The final schema is the result of this sequence of in-
cremental updates across all chunks. We next describe the structure
of the schema state as maintained during this process, i.e., how the
algorithm represents and updates it at each step.
The schema state is organized as a collection of relational ta-
bles. Each table corresponds to either an entity type (e.g., Person ,
Hospital ) or a relationship (e.g., Admission ,Treatment ). To help the
algorithm maintain and use the schema state effectively, each table
is annotated with the following information.
•A canonical table name and a concise natural language descrip-
tion, both generated by the LLM based on the semantics of rele-
vant document chunks and prior schema context;
•A small set of example document chunks for each table, selected
by the LLM to motivate the creation of the table and provide
grounding for its semantics;

Daren Chao, Kaiwen Chen, Naiqing Guan, and Nick Koudas
•A list of attributes for each table, each annotated with a name
and a usage-based explanation derived by the LLM from the
context in which it appeared.
Schema updates are performed by a prompt-based function S𝐺,
which uses a fixed prompt template to invoke an LLM7. Given the
current schema state 𝑆gen
𝑘−1and the𝑘-th document chunk 𝑑𝑘, the
algorithm computes the next state as:
𝑆gen
𝑘=S𝐺
𝑆gen
𝑘−1,𝑑𝑘
,for𝑘=1..𝑛(14)
Updates may involve introducing new tables, adding attributes
to existing tables, or refining existing descriptions. Crucially, the
current schema state remains in memory at every step, enabling
the functionS𝐺to leverage prior schema structure and annotations
to extract additional structure information from each new chunk.
As the process continues, earlier schema elements that were
ambiguous or incomplete may be clarified by later chunks, support-
ing limited self-correction without retroactive reconstruction. We
empirically validate the effectiveness of our method in §6.3. After
all chunks are processed, the final schema 𝑆gen
𝑛constitutes𝑆gen
𝐷. It
serves as a comprehensive, query-agnostic abstraction over the
document collection and provides the structural basis for the next
stage of query-specific schema adaptation.
Phase II: Query-Specific Schema Discovery.In the second phase,
the goal is to transform the general schema 𝑆gen
𝐷into a query-
specific schema 𝑆𝑞
𝐷that contains only the tables and attributes
necessary to answer the input query 𝑞. The process is again it-
erative, following the same schema structure and update pattern
as in Phase I. Schema updates are now performed by a different
prompt-based function S𝑄, which also uses a fixed LLM prompt8,
but incorporates the query 𝑞as an additional input to guide the
refinement.
𝑆𝑞
𝑘=S𝑄
𝑆𝑞
𝑘−1,𝑑𝑘,𝑞,𝑆gen
𝐷
,for𝑘=1..𝑛(15)
In this phase,S𝑄selectively removes irrelevant schema elements,
adds previously overlooked attributes, if any, and restructures at-
tributes to precisely match the intent of query 𝑞. Such refinement
includes accommodating explicit query constraints (e.g., filters,
group-by keys) and implicit query requirements (e.g., join paths,
derived attributes). Similar to Phase I, schema decisions can be
iteratively revised if errors are introduced in earlier steps.
The result is a minimal and query-complete schema 𝑆𝑞
𝐷, essential
for accurate data population and effective query execution. Without
this targeted refinement, critical attributes might be omitted or
extraneous attributes retained, impairing query effectiveness. We
empirically demonstrate in §6.3 that omitting this step leads to
lower schema completeness and reduced query accuracy.
Repair Step.As an optional enhancement, afallback repairstep
can be applied after Phase II, using a powerful LLM (e.g., GPT-5 [ 24])
to verify whether the extracted schema 𝑆𝑞
𝐷suffices to answer the
query𝑞; if not, the system re-invokes schema discovery with addi-
tional iterations to repair and complete the schema. We empirically
show in §6.3 that this repair mechanism discovers any attributes,
achieving perfect attribute-level recall across all datasets.
7Details of the prompt design and its implementation are available in Appendix A.1.
8Details of the prompt design and its implementation are available in Appendix A.2.Performance Considerations.Our schema discovery pipeline
performs two sequential passes over the document collection: Phase
I builds a general schema from scratch, and Phase II refines it to
match the specific query intent. While techniques such as sampling,
chunk clustering, or selective analysis could significantly reduce
computational cost, they are orthogonal to the focus of this work.
We aim to understand whether one can indeed design anaccu-
rateschema discovery and data extraction strategy for a specific
query without hard computational considerations (e.g., token usage
[17,29]). Answering this question first, instigates future research
directions or engineering optimizations to derive performance effi-
ciency. A detailed exploration of efficiency-oriented enhancements
is left to future work and falls outside the scope of this paper.
6 EXPERIMENTAL EVALUATION
6.1 Experimental Setup
6.1.1 Datasets.We evaluateReDDonfive datasetsthat simulate
realistic information extraction and analytical query scenarios over
unstructured or weakly structured document collections. Table 1
summarizes these datasets, detailing the number of queries per
set and the average number of expected output table entries per
query. The first two datasets,SpiderandBird, are derived from the
Spider [ 40] and Bird [ 14] benchmarks. Using their original schemas
and tabular data, following [ 19] we convert tabular rows into natu-
ral language document chunks using a state-of-the-art LLM (GPT-
5 [24]). This ensures that the model used inReDD(Qwen3-30B-A3B
[32]) has not seen these documents during training, thereby miti-
gating data leakage concerns. For each benchmark query, we know
the precise result (ground truth) to evaluate correctness.ReDDis
evaluated on the original natural language queries from Spider and
Bird datasets, as well as on newly introduced queries that involve
multi-table joins, aggregation, and multi-hop reasoning. In our eval-
uation, we do not account for natural language to SQL translation;
instead, we provide the correct SQL query to execute on the ex-
tracted tables. We do this to isolate our evaluation from text-to-SQL
translation errors; naturally any state-of-the-art text-to-SQL tech-
nique can be adopted. While generating the datasets, we randomly
shuffle the chunks to eliminate any semantic correlations. We also
introduce varying degrees of information density in the document
collection (ratio of relevant vs irrelevant chunks to the query) in
§6.4. The prompts used to generate the documents, due to space
constraints, are provided in Appendix A.4. The datasetGalois
is sourced from theFortuneandPremierdatasets as described
in [29]. The datasetFDAis sourced from FDA 510(k) regulatory
filings [ 4,38], which are long-form and heterogeneous, containing
narrative summaries, tabular sections, and metadata. The dataset
CUADis a legal-contract benchmark [ 31], whose lengthy docu-
ments make single-pass LLM processing infeasible due to context
limitations [ 31]. For these datasets, we use both the original bench-
mark queries (typically involving a limited set of documents and
attributes) and additional new queries proposed in this paper. The
additional queries are designed to include more attributes and to
incorporate cross-document aggregation as well as multi-hop rea-
soning9.
9We plan to make available all artifacts associated with this paper.

Relational Deep Dive: Error-Aware Queries Over Unstructured Data
Table 1: Evaluation Datasets Used for AssessingReDD.
Dataset Dataset Source # Queries Avg. Result Rows
SpiderSpider [40] 86 1733
BirdBird [14] 36 2435
GaloisGalois [29] 10 497
FDAFDA 510(k) [38] 6 100
CUADCUAD [31] 15 501
6.1.2 Measurements.We evaluateReDDin two stages: schema
discovery (ISD) and data population (TDP). For data population, we
evaluate the accuracy of the extracted tabular data by comparing it
with the ground truth at thecell(attribute value) level. Specifically,
each ground-truth cell (i.e., each populated value in the ground-
truth table) is checked against the extracted result. We then compute
the following accuracy metric [31]:
ACC pop=1−# missing cells+# incorrect cells
# ground-truth cells(16)
Here,missingcells are those that should have been extracted but
were not, andincorrectcells are those that were extracted but con-
tain erroneous values.
SCAPEandSCAPE-Hyb, identify potentially erroneous cells and
use additional review steps (e.g., human inspection) to validate
and fix extraction errors. If correctly extracted cells are flagged
for inspection, this results in unnecessary correction efforts. To
quantify this, we compute the false positive rate:
FPR pop=FP
FP+TN,(17)
where FP(false positives) denotes correctly extracted cells flagged
for inspection and TN(true negatives) denotes correctly extracted
cells not flagged. For a fixed ACC pop, a higher FPR popimplies addi-
tional wasted effort inspecting accurate extractions, while a lower
FPR popreflects a more efficient process, focusing inspections primar-
ily on erroneous extractions. Thus, FPR popquantifies the inefficiency
(unnecessary inspections) in the extraction pipeline.
For schema discovery, we first assess whether the discovered
schema for each query is sufficient to answer the query, resulting in
an accuracy metric denoted as ACC sch. Additionally, by comparing
the discovered schema with the ground-truth schema in terms
of their attributes, we compute attribute-level recall ( Recattr
sch) and
precision ( Preattr
sch), which measure the completeness and redundancy
of the discovered schema.
6.1.3 Experimental Settings.ReDDis implemented in Python. For
schema discovery, it employs GPT-5 [ 24], a state-of-the-art LLM.
For data population, it employs Qwen3-30B-A3B [ 32], a leading
open-source model at the 30B scale, deployed on an A100 (80GB)
GPU. Unless otherwise specified, all experiments use the follow-
ing default settings: the conflict weight parameter is set to 𝜆=0.5
(see Equation (13)); the classifiers are trained with 50 entries and
calibrated on 150 entries randomly sampled per query.
6.2 Results of Data Population and Correction
6.2.1 Accuracy and Correction Cost.Table 2 reports the data pop-
ulation accuracy ( ACC pop, defined in Equation 16) under different
configurations proposed in this paper: (i)ReDDwithout Correction,Table 2: Summary of Data Population Accuracy ( ACC pop).All values
are averages over queries in Tb. 1.ACC popis defined in Eq. (16).
MethodSpider Bird Galois FDA CUAD
EVAPORATE [4] - - 0.475 0.516 0.209
Palimpzest [17] - - 0.867 0.924 0.613
ReDD(No Correction) 0.938 0.949 0.873 0.965 0.661
ReDD(SCAPE) 0.991 0.992 0.989 0.988 0.724
ReDD(SCAPE-Hyb)0.993 0.994 0.989 0.990 0.983
Table 3: Correction Overhead in Data Population.All values are av-
erages over queries in Tb. 1. FPR popquantifies the cost of unnecessary
corrections, defined in Eq. (17).
MethodSpider Bird Galois FDA CUAD
ReDD(SCAPE) 0.063 0.054 0.044 0.051 0.114
ReDD(SCAPE-Hyb)0.038 0.039 0.027 0.032 0.072
which executes without attempting to detect errors (outputs the
extraction directly from the LLM), (ii)SCAPE, and (iii)SCAPE-Hyb,
which integrates the proposed algorithms with default parameters
(𝛼=0.15), as well as prior baselines, EVAPORATE [ 4] and Palimpzest
[17, 18]10. All metrics are averaged over all queries in Table 1.
Without correction,ReDDachieves reasonably high accuracy
(e.g., 0.938 onSpiderand 0.949 onBird), but still produces a sub-
stantial number of error cells (e.g., 516 remaining error cells on
Spiderand 491 onBird).Since EVAPORATE and Palimpzest are
not designed to handle schemas involving multiple tables, we present
their evaluation only onGalois,FDA, andCUADdatasets (which do
not involve multiple tables). Using the same base model (Qwen3-
30B-A3B [ 32]), EVAPORATE performs considerably worse than
other approaches, while Palimpzest achieves accuracy comparable
(though slightly lower) thanReDDwithout correction(e.g., 0.867 vs.
0.873 onGalois).
By contrast, applying our correction algorithms yields substan-
tial improvements. With 𝛼=0.15, bothSCAPEandSCAPE-Hyb
raise accuracy toabove or near 0.99onSpider,Bird,Galois, and
FDA, withSCAPE-Hybconsistently achieving the highest accuracy
across all datasets.
The CUAD dataset poses unique challenges due to its long-
document nature: a single legal contract (corresponding to one
row in the ground-truth table) can be nearly 100,000 tokens, far
exceeding the context window of Qwen3-30B-A3B [ 32]. To address
this, we adapt a map-reduce style document chunking strategy
inspired by DocETL [ 31]. Each long contract is divided into smaller
chunks (e.g., sections or pages) that can fit within the LLM’s context
window. Instead of assuming that one document chunk corresponds
to one complete table row (as in the default setting), we allow multi-
ple chunks (from the same legal contract) to collectively provide the
attribute values required for a single row. Attribute values are first
extracted independently from each chunk, and then consolidated
into a complete row. Combined with our correction algorithm, this
10[29] in their recent evaluation, demonstrated that Palimpzest has superior accuracy
over other prior baselines. For this reason we compare with Palimpzest, without
including the baselines that [ 29] demonstrated inferior to Palimpzest. We do this as
ccuracy is the focus of our work. We run experiments with the same configuration
parameters as in [29].

Daren Chao, Kaiwen Chen, Naiqing Guan, and Nick Koudas
0.961.00
:0.3
:0.2
:0.1
:0.05
:0.01
:0.005
0.0 0.2 0.4 0.60.870.88 ReDD (No Correction)
FPRpopACCpop
MV
CF
IndivConformal
SCAPE
SCAPE-Hyb
Figure 2: Trade-off between data population accuracy (ACC pop) and
correction cost measured by the false positive rate ( FPR pop).For the
SCAPE-Hybcurve, labels above each point indicate the corresponding 𝛼
value used to produce that accuracy–cost trade-off.
0.1 0.3 0.5 0.7 0.9
1.00
0.96ACCpop
Ncalbase=30
Ncalbase=150
Ncalbase=300
Figure 3: Data population accuracy ACC popforSCAPE-Hybwith
different calibration dataset size 𝑁cal-base on datasetSpider, varying
conflict weight𝜆, underFPR pop=0.2.
chunk-merge (map-reduce) pipeline enablesSCAPE-Hybto reach
0.983 accuracy on CUAD—substantially higher thanSCAPE(0.724,
without chunk merge) and Palimpzest (0.583).
Table 3 reports on the corresponding correction overhead in
terms of false positive rate ( FPR pop). Here,SCAPE-Hybconsistently
maintains lower false positive rates thanSCAPEacross all datasets
(e.g., 0.038 vs. 0.063 onSpiderand 0.039 vs. 0.054 onBird), in-
dicating fewer correctly extracted cells are unnecessarily flagged
for review. The overhead remains modest overall, particularly on
large-scale benchmarks whereSCAPE-Hybachieves both higher
accuracy and lower correction cost. These results demonstrate the
effectiveness of our correction module in improving extraction
quality while incurring minimal verification overhead.
6.2.2 Ablation: Accuracy-Cost Tradeoff.We conduct ablation ex-
periments to compareSCAPE-Hybagainst several error detection
methods proposed in this paper: MV (§3.3), CF (§3.3), andSCAPE.
We also evaluate an alternative method that calibrates each base
classifier independently using conformal prediction [ 2], followed
by majority voting over the calibrated outputs (denoted asIndi-
vConformal). The results are presented in Figure 2, which plots
data population accuracy ( ACC pop) against unnecessary correction
cost measured by the false positive rate ( FPR pop) averaged over all
datasets. The gray dashed line marks the accuracy ofReDDwithout
correction. All methods yield notable improvements over the base-
line. The MV approach, lacks tunable parameters and provides only
a fixed trade-off point. The curve for IndivConformal consistently
lies below others, indicating weaker performance. CF supports lim-
ited tuning but remains inferior in accuracy—its curve consistently
lies belowSCAPE, indicating higher correction cost for the same
level of accuracy. In contrast,SCAPEandSCAPE-Hyboffer sig-
nificantly better accuracy-correction cost trade-offs.SCAPE-Hyb
consistently outperforms all baselines. It achieves the highest data
50 100 150 200 250 300 Ncal-base0.961.00ACCpop
 SCAPE on Spider
SCAPE-Hyb on Spider
SCAPE on Bird
SCAPE-Hyb on BirdFigure 4: Data population accuracy ACC popofSCAPEandSCAPE-
Hybvarying calibration dataset size𝑁 cal-base , underFPR pop=0.2.
population accuracy while incurring the lowest rate of unnecessary
corrections, demonstrating the effectiveness of combiningSCAPE
with conflict information.
6.2.3 Effect of Coverage Threshold 𝛼.The results in Figure 2, aver-
aged across all datasets, reveal a key trend:SCAPE-Hybachieves
high initial accuracy ( ≥0.974across all datasets) even at large 𝛼,
with more accuracy gains as 𝛼decreases. For small 𝛼, accuracy can
reach 1, but at the cost of a higher false positive rate (FPR pop).
Using theSpiderdataset as an example (avg. 1,733 rows per
query), setting 𝛼=0.5yields 0.947 accuracy with 4.9% of the data re-
viewed (including 8 false positives). Reducing the threshold to 𝛼=0.3
improves accuracy to 0.975, with 9.2% of rows reviewed (37 false
positives). At 𝛼=0.01, accuracy further increases to 0.998, but 31%
of the data must be reviewed (608 false positives). As 𝛼decreases,
the proposed framework becomes more conservative—fewer pre-
dictions are accepted as confident, and more data are flagged for
review. This generally improves accuracy, as more uncertain predic-
tions are reviewed and corrected. However, it also increases false
positives, reflected in higher FPR pop. Conversely, larger 𝛼values
yield cheaper but relatively less accurate results.
The results suggest that minimal user effort (relative to output
size) suffices for high accuracy ( >0.97), while perfect accuracy (1.0)
demands significantly more verification effort. This finding opens
several research directions, as discussed in §8.
6.2.4 Effect of Conflict Weight 𝜆.Figure 3 illustrates the impact
of the conflict weight ( 𝜆) on data population accuracy ( ACC pop)
forSCAPE-Hybon datasetSpider(under FPR pop=0.2). According
to the blue curve under default setting calibration dataset size
𝑁cal-base =150the results reveal a clear trend: accuracy peaks at 𝜆≈0.5
and declines when 𝜆is either too low or too high. This indicates
that an optimal balance in weighting conflicts is critical—under
weighting (𝜆≪0.5) fails to leverage disagreement signals effectively,
while over weighting ( 𝜆≫0.5) can suppress correct but less frequent
outputs. These findings validate our choice of 𝜆=0.5as the default
inSCAPE-Hyb. This trend also holds well across other datasets.
However, when the size of the calibration dataset 𝑁cal-base varies,
the peak of the curve shifts accordingly. For instance, when 𝑁cal-base
is relatively small (=30),SCAPEis significantly weaker than the con-
flict signal, resulting in the curve peaking at a larger 𝜆, around 0.9.
In contrast, when 𝑁cal-base is large (=300), the peak shifts leftward
to a𝜆value around 0.3-0.4.
6.2.5 Effect of Calibration Dataset Size 𝑁cal-base .Figure 4 illustrates
the impact of calibration dataset size ( 𝑁cal-base , defined in §4.1) on
data population accuracy ( ACC pop) for bothSCAPEandSCAPE-
Hyb(measured at FPR pop=0.2). We report results only onSpider
andBirdfor brevity. Across the reported datasets, we observe a

Relational Deep Dive: Error-Aware Queries Over Unstructured Data
50 100 200 250 300 Ncal-base0.951.00ACCpop
SCAPE-Hyb on Spider
SCAPE-Hyb on Bird
Figure 5: Data population accuracy ACC popvarying training dataset
size𝑁 cls, underFPR pop=0.2.
0.851.00
Spider Bird Galois FDA CUAD0.650.70
No Correction SCAPE-Hyb (Committee) SCAPE-Hyb (Human)ACCpop
Figure 6: Data population accuracy ACC popusing human-annotated
vs. LLM committee-generated training data.
Table 4: Evaluation of Schema Discovery over All Datasets.# Invalids
refers to the number of schema discoveries missing essential attributes, such
that the query cannot be answered. This is across all queries in Table 1.
Method # InvalidsRecattr
schPreattr
sch
ISD (Phase I Only) 2 0.989 0.522
ISD (Phase II Only) 12 0.951 0.968
ISD (Phase I & II) 1 0.991 0.956
ISD (Phase I & II + Repair) 0 1.000 0.956
steady improvement in accuracy as 𝑁cal-base increases, with perfor-
mance plateauing once the calibration set exceeds roughly 100-150
examples.
This finding demonstrates that near-optimal accuracy can be at-
tained with only a modest calibration set. For example,SCAPE-Hyb
achieves accuracy above 0.99 onSpiderwith just 150 calibration ex-
amples. These results demonstrate the data efficiency of our method,
as high accuracy is attainable even with limited calibration data,
an advantage in low-resource or high-cost settings.
Furthermore,SCAPE-Hybconsistently surpassesSCAPEwhen
calibration data is scarce. By leveraging conflict information,SCAPE-
Hybcompensates for the reduced supervision in small calibration
sets, reinforcing its robustness in such scenarios.
6.2.6 Effect of Training Dataset Size 𝑁cls.Figure 5 demonstrates that
data population accuracy ( ACC pop) improves with larger training
set size (𝑁cls), but plateaus quickly—for example, reaching >0.99
accuracy with just 50 examples. These results suggest that our
approach is label-efficient: high accuracy can be achieved with
limited training data and scales well as more data becomes available.
6.2.7 Impact of Label Source: Human vs. LLM-Synthetic.Figure 6
compares data population accuracy ( ACC pop) when the training
labels used in correction are sourced from either human annotations
or LLM committee decisions.SCAPE-Hybperforms comparably
under both label sources, with <1%difference in accuracy. This
suggests that high-quality synthetic labels from LLM committees
can be a viable alternative to human annotations in training error
detection classifiers. Compared to data sourced from humans, LLM-
generated labels are relatively cheaper, easier to obtain, and more
practical for scaling to large datasets.
ACCschACCpop0.961.00
one chunk  multiple rows
one chunk  one row
Figure 7: Schema discovery ac-
curacy ACC schand data popula-
tion accuracy ACC popunder one-
to-many chunk-to-table setting.
0.2 0.4 0.6 0.8 1.0
Information Density1.00
0.96
ACCpop
DocDensity
DatasetDensityFigure 8: Data population accu-
racy ( ACC pop) ofSCAPE-Hyb
under diff information density
levels, underFPR pop=0.2.
6.3 Results of Schema Discovery
We evaluate the schema discovery stage (ISD) over all datasets. As
shown in Table 4, the final pipeline (Phase I & II + Repair, see §5)
achieves perfect sufficiency (identifies all required attributes) on all
queries, with zero invalid cases—i.e., every extracted schema con-
tains all necessary attributes required to answer the query, result-
ing in an average recall of Recattr
sch=1.000. In addition, the extracted
schemas remain concise, with an average precision of Preattr
sch=0.956,
meaning only a few redundant attributes are occasionally included—
improving the efficiency of downstream data population.
6.3.1 Ablation Study of ISD Components.We conduct an ablation
study to assess the contribution of each ISD component. Table 4
compares four configurations derived from Section 5:
•Phase I Only: A query-independent document-based schema
extraction approach that achieves high recall (0.989)—slightly
below 1.0, as some query-relevant attributes are deeply embed-
ded in the text and not easily surfaced by a general approach.
However, it suffers from very low precision (0.522), as it tends
to include many irrelevant attributes not required by the query.
•Phase II Only: A query-specific strategy that improves precision
(0.968) but often misses necessary attributes due to the lack of
contextual information, resulting in many invalid schemas.
•Phase I & II: Combining both phases significantly improves
overall effectiveness, achieving high recall (0.991) and precision
(0.952), and greatly reducing invalid extractions.
•Phase I & II + Repair: A repair step further eliminates remaining
invalid cases, achieving 100% valid extractions.
These results highlight the effectiveness of the two-phase design in
achieving both schema completeness and compactness, while the
repair step ensures full robustness across diverse query scenarios.
6.4 Additional Analyses
6.4.1 Results under One-to-Many Chunk-to-Table Mapping.We cre-
ated a modifiedSpiderdataset, merging two or three document
chunks with different schema to simulate one-to-many mappings,
where each chunk contributes to multiple schemas, contrasting
with the default one-to-one setup. As shown in Figure 7, schema
discovery accuracy ( ACC sch) remains unchanged, while data popu-
lation accuracy ( ACC pop) drops slightly by 0.13% due to increased
ambiguity when mapping chunks to multiple rows. This trend holds
across datasets. Since our system processes attributes iteratively,
performance stays robust if schema descriptions are clear, emphasiz-
ing the need for precise schema discovery. The results confirm our
method’s ability to handle one-to-many mappings when schemas
are well-defined.

Daren Chao, Kaiwen Chen, Naiqing Guan, and Nick Koudas
6.4.2 Impact of Information Density.Figure 8 testsReDD’s perfor-
mance under different information density levels, examining two
aspects: DocDensity, which quantifies how much of a document
chunk’s text is relevant to the extracted data, and DatasetDensity
which asseses how many document chunks in the dataset are query-
relevant (both presented as fractions with 1 being the most relevant
and lower values signifying increased presence of irrelevant infor-
mation aiming to deter correct inference).ReDDmaintains high
accuracy across all scenarios. Our key findings are that DocDensity
has minimal effect becauseReDDprocesses attributes indepen-
dently during TDP, avoiding confusion from irrelevant content.
In addition DatasetDensity has little impact asReDD’s ISD stage
performs global schema discovery, easily filtering irrelevant chunks.
These results showReDDworks well with real-world data where
relevant information may be sparse or unevenly distributed.
6.4.3 Token and Performance Considerations.This study demon-
strates that near-perfect accuracy for unstructured data query pro-
cessing is achievable. While monetary cost modeling and query
performance remain critical considerations, they are beyond the
scope of this work. They do represent key directions for future
research however.
In all cases in our experiments total per query execution for
SCAPE-Hyb(including schema discovery, data extraction, auto-
mated training and calibration data acquisition, classifier training,
calibration and query execution) is under 4 hours. This time is
roughly broken down as follows: approximately 40 minutes for
schema discovery on average, which is not optimized currently
and sequentially processes all documents two times (via GPT-5 API
[24]); on average 3 hours per query to extract the data using a multi-
threaded setting to parallelize the process without any inference
optimizations (on a 8 A100 GPU cluster) and around 20 minutes for
data correction, which is a manual process in our implementation
currently. Token consumption is not currently optimized. For the
Galoisdataset, average token consumption per query execution is
approximately 18M.
There is ample scope for optimizations. This could involve de-
ploying techniques from the literature, such as sampling to reduce
the number of documents processed for schema discovery, leverag-
ing LLMs to automatically generate code for efficient data extrac-
tion, thereby minimizing token usage as well as using advanced
LLMs for correction. In all cases, however, formally quantifying
the trade-offs between these optimizations and accuracy remains a
primary focus.
7 RELATED WORK
LLMs have enabled novel lines of research to realize the vision of
query processing on unstructured data. [ 4] presents EVAPORATE,
a system that uses Large Language Models (LLMs) to automat-
ically generate structured, queryable tables from heterogeneous
semi-structured documents. It explores direct extraction and a more
cost-effective code synthesis approach (EVAPORATE-CODE+) that
generates and aggregates multiple code snippets using weak su-
pervision to achieve higher quality. TWIX is a tool that extracts
structured data from templatized documents by first inferring the
underlying visual template [ 15]. ZENDB, is a document analyticssystem designed to answer ad-hoc SQL queries on collections of
templatized documents [16].
[27] introduces semantic operators, a novel declarative model
for AI-based data transformations that extend traditional relational
operators with natural language specifications. Dai et. al., [ 9] pro-
posed UQE (Universal Query Engine), a new system designed for
flexible and efficient analytics on unstructured data using a dialect
of SQL called UQL (Universal Query Language). Liu et. al., [ 17,18]
introduce a declarative system for optimizing AI workloads, partic-
ularly "Semantic Analytics Applications" (SAPPs), which interleave
traditional data processing with AI-driven semantic reasoning. [ 34]
introduces ELEET, a novel execution engine that enables seamless
querying and processing of text as a first-class citizen alongside
tables by leveraging learned multi-modal operators (MMOps). [ 7]
proposes Table-Augmented Generation (TAG) as a unified para-
digm for answering natural language questions over databases,
addressing the limitations of existing Text2SQL and RAG methods
by integrating language model reasoning with database capabili-
ties. Anderson et. al., [ 1] introduce infrastructure for unstructured
data analytics. Wang [ 36] presents Unify, an innovative system that
leverages large language models (LLMs) to automatically generate,
optimize, and execute query plans for unstructured data analytics
queries expressed in natural language. [ 33] proposes CAESURA,
a novel query planner that utilizes large language models (LLMs)
to translate natural language queries into executable multi-modal
query plans, which can include complex operators for various data
modalities beyond traditional relational data. [ 37] is a novel LLM-
powered analytics system designed to handle data analytics queries
over multi-modal data lakes by taking natural language queries
as input, orchestrating a pipeline, and outputting results. Recently
[29], presented Galois a system for logical and physical optimiza-
tion for SQL execution over LLMs and demonstrated optimization
strategies to reduce token costs with small accuracy implications
compared to other approaches. Our work is the first to propose
a formal framework to quantify query accuracy execution over
unstructured data with guarantees.
8 CONCLUSION AND FUTURE WORK
This paper introducesReDD, a novel framework for executing error-
aware analytical queries over unstructured data.ReDDfeatures a
two-stage pipeline that first discovers a query-specific schema and
then populates relational tables. A key contribution is the introduc-
tion ofSCAPEandSCAPE-Hyb, statistically calibrated methods
for error detection that provide coverage guarantees. Experimental
results demonstrate thatReDDsignificantly reduces data extraction
errors from as high as 30% to less than 1%, while ensuring high
schema completeness with 100% recall. The core reliability frame-
work presented raises numerous promising avenues for future work,
ranging from incorporatingReDDto other unstructured data query
processing frameworks [ 33], to designing novel query processing
techniques to enhance performance, optimizing for monetary cost,
and mitigating bounded errors in extraction results.
REFERENCES
[1]Eric Anderson, Jonathan Fritz, Austin Lee, Bohou Li, Mark Lindblad, Henry
Lindeman, Alex Meyer, Parth Parmar, Tanvi Ranade, Mehul A. Shah, Benjamin
Sowell, Dan Tecuci, Vinayak Thapliyal, and Matt Welsh. 2024. The Design of

Relational Deep Dive: Error-Aware Queries Over Unstructured Data
an LLM-powered Unstructured Analytics System. arXiv:2409.00847 [cs.DB]
https://arxiv.org/abs/2409.00847
[2]Anastasios N. Angelopoulos and Stephen Bates. 2022. A Gentle Introduc-
tion to Conformal Prediction and Distribution-Free Uncertainty Quantification.
arXiv:2107.07511 [cs.LG] https://arxiv.org/abs/2107.07511
[3]Anthropic. 2025. Introducing Claude 4. https://www.anthropic.com/index/
claude-4
[4] Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel,
Immanuel Trummer, and Christopher Ré. 2023. Language Models Enable Simple
Systems for Generating Structured Views of Heterogeneous Data Lakes.arXiv
preprint arXiv:2304.09433(2023). https://arxiv.org/abs/2304.09433
[5] Rina Foygel Barber, Emmanuel J Candes, Aaditya Ramdas, and Ryan J Tibshirani.
2023. Conformal prediction beyond exchangeability.The Annals of Statistics51,
2 (2023), 816–845.
[6] Rina Foygel Barber, Emmanuel J. Candes, Aaditya Ramdas, and Ryan J. Tibshirani.
2023. Conformal prediction beyond exchangeability. arXiv:2202.13415 [stat.ME]
https://arxiv.org/abs/2202.13415
[7]Asim Biswal, Liana Patel, Siddarth Jha, Amog Kamsetty, Shu Liu, Joseph E.
Gonzalez, Carlos Guestrin, and Matei Zaharia. 2024. Text2SQL is Not Enough:
Unifying AI and Databases with TAG. arXiv:2408.14717 [cs.DB] https://arxiv.
org/abs/2408.14717
[8] Kaiwen Chen, Yueting Chen, Nick Koudas, and Xiaohui Yu. 2025. Reliable Text-
to-SQL with Adaptive Abstention.Proc. ACM Manag. Data3, 1, Article 69 (Feb.
2025), 30 pages. doi:10.1145/3709719
[9] Hanjun Dai, Bethany Yixin Wang, Xingchen Wan, Bo Dai, Sherry Yang, Azade
Nova, Pengcheng Yin, Phitchaya Mangpo Phothilimthana, Charles Sutton, and
Dale Schuurmans. 2024. UQE: A Query Engine for Unstructured Databases.
arXiv:2407.09522 [cs.DB] https://arxiv.org/abs/2407.09522
[10] Google. 2025. Gemini Deep Research. https://gemini.google/overview/deep-
research.
[11] Ganesh Jawahar, Benoît Sagot, and Djamé Seddah. 2019. What does BERT
learn about the structure of language?. InACL 2019-57th Annual Meeting of the
Association for Computational Linguistics.
[12] Nicola Jones. 2025. OpenAI’s ’deep research’ tool: is it useful for scientists?
Nature(2025). doi:10.1038/d41586-025-00377-9
[13] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. [n. d.]. Retrieval-Augmented Gener-
ation for Knowledge-Intensive NLP Tasks. https://arxiv.org/abs/2005.11401
[14] Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang,
Bowen Qin, Ruiying Geng, Nan Huo, et al .2023. Can llm already serve as a
database interface? a big bench for large-scale database grounded text-to-sqls.
Advances in Neural Information Processing Systems36 (2023), 42330–42357.
[15] Yiming Lin and et al. 2025. TWIX: Automatically Reconstructing Structured
Data from Templatized Documents.arXiv preprint arXiv:2501.06659(2025).
arXiv:2501.06659 [cs.CL]
[16] Yiming Lin, Madelon Hulsebos, Ruiying Ma, Shreya Shankar, Sepanta Zeigham,
Aditya G. Parameswaran, and Eugene Wu. 2024. Towards Accurate and Efficient
Document Analytics with Large Language Models. arXiv:2405.04674 [cs.DB]
https://arxiv.org/abs/2405.04674
[17] Chunwei Liu, Matthew Russo, Michael Cafarella, Lei Cao, Peter Baile Chen, Zui
Chen, Michael Franklin, Tim Kraska, Samuel Madden, Rana Shahout, et al .2025.
Palimpzest: Optimizing ai-powered analytics with declarative query processing.
InProceedings of the Conference on Innovative Database Research (CIDR). 2.
[18] Chunwei Liu, Gerardo Vitagliano, Brandon Rose, Matthew Printz, David Andrew
Samson, and Michael Cafarella. 2025. PalimpChat: Declarative and Interactive
AI analytics. InCompanion of the 2025 International Conference on Management
of Data. 183–186.
[19] Seiji Maekawa, Hayate Iso, and Nikita Bhutani. 2025. Holistic Reasoning with
Long-Context LMs: A Benchmark for Database Operations on Massive Textual
Data. InThe Thirteenth International Conference on Learning Representations.
https://openreview.net/forum?id=5LXcoDtNyq
[20] Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. 2008.Intro-
duction to Information Retrieval. Cambridge University Press. Web publication
at informationretrieval.org..
[21] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating
and editing factual associations in gpt.Advances in neural information processing
systems35 (2022), 17359–17372.
[22] Jerzy Neyman and Egon Sharpe Pearson. 1933. IX. On the problem of the most
efficient tests of statistical hypotheses.Philosophical Transactions of the Royal
Society of London. Series A, Containing Papers of a Mathematical or Physical
Character231, 694-706 (1933), 289–337.
[23] OpenAI. 2025. Introducing Deep Research. https://openai.com/index/
introducing-deep-research.
[24] OpenAI. 2025. Introducing GPT-5. https://openai.com/index/introducing-gpt-5/
[25] OpenAI. 2025. Introducing OpenAI o3 and o4-mini. https://openai.com/index/
introducing-o3-and-o4-mini/[26] Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas
Kotek, and Yonatan Belinkov. 2024. Llms know more than they show: On the
intrinsic representation of llm hallucinations.arXiv preprint arXiv:2410.02707
(2024).
[27] Liana Patel, Siddharth Jha, Melissa Pan, Harshit Gupta, Parth Asawa, Carlos
Guestrin, and Matei Zaharia. 2025. Semantic Operators: A Declarative Model for
Rich, AI-based Data Processing. arXiv:2407.11418 [cs.DB] https://arxiv.org/abs/
2407.11418
[28] Jonathan Roberts, Kai Han, and Samuel Albanie. 2025. Needle Threading: Can
LLMs Follow Threads Through Near-Million-Scale Haystacks?. InThe Thirteenth
International Conference on Learning Representations. https://openreview.net/
forum?id=wHLMsM1SrP
[29] Dario Satriani, Enzo Veltri, Donatello Santoro, Sara Rosato, Simone Varriale,
and Paolo Papotti. 2025. Logical and Physical Optimizations for SQL Query
Execution over Large Language Models.Proceedings of the ACM on Management
of Data3, 3 (2025), 1–28.
[30] Glenn Shafer and Vladimir Vovk. 2008. A tutorial on conformal prediction.
Journal of Machine Learning Research9, 3 (2008).
[31] Shreya Shankar, Tristan Chambers, Tarak Shah, Aditya G Parameswaran, and
Eugene Wu. 2024. Docetl: Agentic query rewriting and evaluation for complex
document processing.arXiv preprint arXiv:2410.12189(2024).
[32] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https:
//arxiv.org/abs/2505.09388
[33] Matthias Urban and Carsten Binnig. 2024. Demonstrating CAESURA: Lan-
guage Models as Multi-Modal Query Planners. InCompanion of the 2024 Inter-
national Conference on Management of Data(Santiago AA, Chile)(SIGMOD
’24). Association for Computing Machinery, New York, NY, USA, 472–475.
doi:10.1145/3626246.3654732
[34] Matthias Urban and Carsten Binnig. 2024. Efficient Learned Query Execution
over Text and Tables [Technical Report]. arXiv:2410.22522 [cs.DB] https://arxiv.
org/abs/2410.22522
[35] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. 2005.Algorithmic
learning in a random world. Vol. 29. Springer.
[36] Jiayi Wang and Jianhua Feng. 2025. Unify: An Unstructured Data Analytics Sys-
tem . In2025 IEEE 41st International Conference on Data Engineering (ICDE). IEEE
Computer Society, Los Alamitos, CA, USA, 4662–4674. doi:10.1109/ICDE65448.
2025.00374
[37] Jiayi Wang, Guoliang Li, and Jianhua Feng. 2025. iDataLake: An LLM-Powered
Analytics System on Data Lakes.IEEE Data Eng. Bull.49, 1 (2025), 57–69. http:
//sites.computer.org/debull/A25mar/p57.pdf
[38] Eric Wu, Kevin Wu, Roxana Daneshjou, David Ouyang, Daniel E Ho, and James
Zou. 2021. How medical AI devices are evaluated: limitations and recommenda-
tions from an analysis of FDA approvals.Nature Medicine27, 4 (2021), 582–584.
[39] xAI. 2024. Introducing Grok-3: xAI’s Next-Generation Language Model. https:
//x.ai/news/grok-3
[40] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James
Ma, Irene Li, Qingning Yao, Shanelle Roman, et al .2018. Spider: A large-scale
human-labeled dataset for complex and cross-domain semantic parsing and
text-to-sql task.arXiv preprint arXiv:1809.08887(2018).
AAPPENDIX: PROMPT TEMPLATES USED IN
REDD
We include the prompt templates used in different components of
ReDDfor reproducibility and future work reference.
A.1 General Schema Discovery Prompt
This prompt is used to iteratively construct a query-agnostic schema
(ISD Phase I, §5) over the document chunks. The actual prompt
includes a one-shot example of input and output to guide the model.
For brevity, the example is omitted here.
You are a database expert specializing in relational schema design
for heterogeneous, natural-language documents. Given a document
and the current schema state, your goal is to iteratively construct
and refine a schema that accurately captures the document’s struc-
ture and semantics.
Instructions:
•Identify all salient attributes from the document.
•Determine whether the current schema can accommo-
date the document; create a new table if more than two
attributes are missing from any suitable table.

Daren Chao, Kaiwen Chen, Naiqing Guan, and Nick Koudas
•Assign the document to the most appropriate table in the
schema.
•For each attribute, provide a 1–2 sentence context expla-
nation derived from its usage in the document.
Rules:
•Only include attributes that can be supported by explicit
evidence in the document.
•Avoid adding placeholders like IDunless directly men-
tioned or inferable.
•Provide clear step-by-step reasoning before outputting
the revised schema and document assignment.
Format:
•Input:{ "Document": <text>, "Record of Schema": <schema
state> }
•Output:{ "Reasoning": <text>, "Updated Record of Schema":
<tables>, "Assignment": <table name> }
A.2 Query-Specific Schema Discovery Prompt
This prompt is used to iteratively construct a query-specific schema
(ISD Phase II, §5) tailored for the given query. The actual prompt
includes a one-shot example to guide behavior. The example is
omitted here for brevity.
You are a database expert focusing on schema design. Your task is
to iteratively construct a query-specific relational schema over a
collection of natural-language documents. In each iteration, you
are given:
•a natural language query,
•a new document,
•a general schema derived from the entire corpus, and
•the current query-specific schema state.
Instructions:
•Determine which table (from the general schema) the
document maps to.
•Identify any query-relevant attributes from the document
that are necessary to answer the query, including those
needed for joins.
•Update the query-specific schema only by:
(a)adding a new table (reusing the general schema name),
or
(b) adding new attributes to an existing table.
•You may applyat most oneof these actions per iteration.
•Assign the document to a table, or return "Assignment":
Noneif the document is irrelevant.
Rules:
•Do not assess whether the data satisfies the query condi-
tions—only whether it contains schema-relevant attributes.
•Do not include unnecessary attributes.
•For aggregate queries, include raw attributes necessary
for computing the aggregate.
Format:
•Input:{ "Document", "Query", "Record of Query-specific
Schema", "General Schema" }
•Output:{ "Reasoning", "Updated Record of Query-specific
Schema", "Assignment" }
A.3 Tabular Data Population Prompt (TDP)
Two prompts are used: one to select the target table for a document
chunk, and another to extract values for each attribute.
(1) Table Resolver PromptYou are a database expert. Your task is to determine which table a
given document belongs to, based on a provided set of table schemas.
Each document can be assigned to only one table.
Instructions:
•Read the document and compare it with the attribute
descriptions in each table schema.
•Assign the document to the table whose schema best
matches its content.
Format:
•Input:{ "Document", "Schema" }
•Output:{ "Table Assignment": <Schema Name> }
(2) Attribute Extractor Prompt
You are a database expert. Your task is to extract a specific attribute
value from a natural language document, given a table schema and
a target attribute.
Instructions:
•Examine the document and the schema.
•Locate the value in the document corresponding to the
target attribute.
•If the attribute value is found, return it; otherwise, return
None.
Format:
•Input:{ "Document", "Schema", "Target Attribute" }
•Output:{ <Target Attribute>: <Extracted Value or None>
}
A.4 Dataset Document Generation Prompt
In ReDD-S and ReDD-B (§6), document chunks were generated by
transforming structured data (e.g., table rows) into natural language
using the following prompt template.
You are a technical writer tasked with converting structured table
rows into natural, paragraph-style descriptions suitable for a pro-
fessional document corpus (e.g., medical reports, financial filings,
legal records).
Instructions:
•Given a structured data row and a schema, rewrite
the row as a coherent, grammatically correct para-
graph.
•Preserve factual accuracy and semantics.
•Implicitly include all attributes without listing them
as key–value pairs.
•The generated text should resemble real-world docu-
ment language, using varied sentence structure and
terminology.
Format:
•Input:{ "Schema": <table schema>, "Row": <struc-
tured data> }
•Output:Natural language paragraph describing the
row.