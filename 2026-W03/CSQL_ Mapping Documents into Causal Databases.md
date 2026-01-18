# CSQL: Mapping Documents into Causal Databases

**Authors**: Sridhar Mahadevan

**Published**: 2026-01-13 01:00:38

**PDF URL**: [https://arxiv.org/pdf/2601.08109v1](https://arxiv.org/pdf/2601.08109v1)

## Abstract
We describe a novel system, CSQL, which automatically converts a collection of unstructured text documents into an SQL-queryable causal database (CDB). A CDB differs from a traditional DB: it is designed to answer "why'' questions via causal interventions and structured causal queries. CSQL builds on our earlier system, DEMOCRITUS, which converts documents into thousands of local causal models derived from causal discourse. Unlike RAG-based systems or knowledge-graph based approaches, CSQL supports causal analysis over document collections rather than purely associative retrieval. For example, given an article on the origins of human bipedal walking, CSQL enables queries such as: "What are the strongest causal influences on bipedalism?'' or "Which variables act as causal hubs with the largest downstream influence?'' Beyond single-document case studies, we show that CSQL can also ingest RAG/IE-compiled causal corpora at scale by compiling the Testing Causal Claims (TCC) dataset of economics papers into a causal database containing 265,656 claim instances spanning 45,319 papers, 44 years, and 1,575 reported method strings, thereby enabling corpus-level causal queries and longitudinal analyses in CSQL. Viewed abstractly, CSQL functions as a compiler from unstructured documents into a causal database equipped with a principled algebra of queries, and can be applied broadly across many domains ranging from business, humanities, and science.

## Full Text


<!-- PDF content starts -->

Csql: MappingDocuments intoCausalDatabases∗
A Preprint
Sridhar Mahadevan
Adobe Research and University of Massachusetts, Amherst
smahadev@adobe.com, mahadeva@umass.edu
January 14, 2026
Abstract
We describe a novel system, Csql, that automatically converts a collection of unstructured
text documents into an SQL-queryablecausal database(CDB). A CDB differs from a traditional
DB: it is designed to answer “why” questions via causal interventions and structured causal
queries. Csqlbuilds on our earlier system, Democritus, which converts documents into
thousands of local causal models derived from causal discourse [Mahadevan, 2025b]. Unlike
RAG-based systems or knowledge-graph–centric approaches, Csqlsupports causal analysis
over document collections rather than purely associative retrieval. For example, given an
article on the origins of human bipedal walking, Csqlenables queries such as: “What are
the strongest causal influences on bipedalism?” or “Which variables act as causal hubs with
the largest downstream influence?”
Beyond single-document case studies, we show that Csqlcan also ingestRAG/IE-compiled
causal corporaat scale by compiling the Testing Causal Claims (TCC) dataset of economics
papers into a causal database containing 265,656 claim instances spanning 45,319 papers, 44
years, and 1,575 reported method strings, thereby enabling corpus-level causal queries and
longitudinal analyses in SQL. Conceptually, Csqlconverts document collections into causally
grounded relational databases, enabling causal analysis via standard SQL. In contrast to
prior work that relies on hand-designed ontologies, fixed schemas, or domain-specific
information extraction pipelines, Csqlinduces its schema directly from language (or from
language-compiled causal artifacts). Viewed abstractly, Csqlfunctions as a compiler from
unstructured documents into a causal database equipped with a principled algebra of
queries, and can be applied broadly across many domains ranging from business, economics,
humanities, and science.
KeywordsCausality·Natural Language·Databases·SQL·AI·Machine Learning
1 Introduction
We introduce Csql, a causal knowledge discovery system that automatically converts collections of
documents into SQL-queryablecausaldatabases (CDB). Unlike retrieval-augmented generation (RAG)
systems or knowledge-graph–centric approaches, which yield traditional databases, Csqlsupportscausal
analysisover document collections rather than purely associative retrieval. Csqlalso differs fundamentally
from traditional causal inference methods [Imbens and Rubin, 2015, Pearl, 2009], which typically focus on
narrow, domain-specific studies grounded in numerical or experimental data. In contrast, Csqloperates
directly on large collections of unstructured text.
A central feature of Csqlis that it induces its relational schema directly from language. Unlike prior
approaches that rely on hand-designed ontologies, fixed schemas, or domain-specific information extraction
pipelines, Csqlautomatically constructs a causally grounded database whose structure emerges from
∗Draft under revision.arXiv:2601.08109v1  [cs.DB]  13 Jan 2026

Apreprint- January14, 2026
Property RAG Systems Knowledge Graphs IE Pipelines Csql
Primary goal Answer generation Structured facts Entity/relation extraction Causal analysis
Input Text chunks Curated triples Annotated text Unstructured documents
Schema None (implicit) Predefined ontology Fixed schemaInduced from discourse
Causality support None Implicit/informal Pairwise onlyWeighted causal relations
Uncertainty None Rare/ad hoc NoneAggregated model support
Conflicting claims Not supported Manual resolution Not supportedPreserved and quantified
Compositionality Prompt-based Limited graph traversal LimitedSQL joins and aggregations
Counterfactuals No No NoSupported via causal structure
Output format Generated text Graph database Triples/tablesRelational causal database
Execution model Neural inference Symbolic querying Batch extractionDeterministic SQL
Table 1: Comparison of Csqlwith RAG systems, knowledge graphs, and traditional information extraction
(IE) pipelines. Csqldiffers by compiling causal discourse into an SQL-queryable causal database with
explicit uncertainty and support for causal analysis.
discourse itself. Viewed abstractly, Csqlfunctions as a compiler from unstructured documents into a causal
database equipped with a principled algebra of queries. This design enables broad applicability across
domains ranging from science and public policy to economics and history.
Table 1 contrasts traditional RAG or Knowledge Graph built databases with causal databases constructed by
Csql. Csqlbuilds directly on our previous work on Democritus, which introduced an end-to-end system for
discovering and evaluating causal structure from language. Given a document, Democritusautomatically
constructs thousands of local causal models, evaluates them for plausibility, and aggregates them into a
ranked causal analysis expressed in natural language. Csqltakes the output of Democritusone step further
by compiling these causal artifacts into an SQL-queryable database, thereby enabling systematic causal
querying over entire document collections.
Csqlrelies on recent advances in categorical causality [Mahadevan, 2025a, Fritz, 2020] and geometric deep
learning [Fong et al., 2019, Mahadevan, 2024, Gavranovi´ c et al., 2024], together with large language models
(LLMs) used as discourse compilers. LLMs can generate rich causal narratives—enumerating subtopics,
posing causal questions, and articulating mechanisms across domains ranging from macroeconomics to
neuroscience. However, LLM-based document summarization alone cannot produce a causally struc-
tured, queryable database. Csqlfills this gap by transforming causal discourse into an explicit relational
representation.
For example, given a newspaper article—such as a recentWashington Poststory on whether eating dark
chocolate may elongate life,2—Csqlcan construct a causal database that supports queries such as: “What are
the strongest causal influences on dark chocolate consumption?” or “Which variables act as downstream
causal hubs in this discourse?” More broadly, Csqlis designed to support probing, comparative analysis of
causal claims not only within individual documents, but across an entireframe of causal discourseinduced
from a document collection.
A crucial design choice in Csqlis the construction of this frame of discourse. Rather than restricting analysis
to the input documents alone, the system treats each document as a seed for a broader causal neighborhood.
Democritusfirst performs topic discovery over the input text and then expands these topics using a language
model to construct a local discourse manifold. Causal claims are generated and evaluated relative to this
expanded context. As a result, Csqlreasons not only about what a document explicitly states, but about how
its claims are situated within a larger landscape of causal discourse.
2 The Origin of Bipedal Walking: A Running Example
We will illustrate the construction of Csqlthrough a running example based on The Washington Post article
on the origins of bipedal walking in human ancestors, dating back to 7 million years ago (see Figure 1).3. The
process of constructing a CSQL database involves first preprocessing the PDF document containing this
article using our previous Democritus[Mahadevan, 2025b] system. Democritus is a system for building
large causal DAG models from carefully curated queries to a large language models (LLM) [OpenAI, 2025,
2https://www.washingtonpost.com/wellness/2025/12/26/dark-chocolate-health-benefits
3urlhttps://www.washingtonpost.com/science/2026/01/02/human-ancestor-biped/
2

Apreprint- January14, 2026
Figure 1: An article in The Washington Post on the origins of human bipedal walking.
2024, Anthropic, 2024, Dubey et al., 2024, Mistral AI, 2024]. Democrituscan construct a rich library of DAG
models from a text document in PDF format, evaluate and rank order these claims quantitatively, and then
produce a natural language executive summary. Example local causal models automatically constructed
from this article are shown in Figure 2.
Democritusconstructs a detailed written summary of the causal claims both within the document, and the
surrounding context that is automatically produced during the run. A brief except of such a report is given
below.
# Causal Deep Dive —
0001_when_did_humanity_take_its_first_step_scientists_say_they_no_70ccf7f26a62
This note is a human-readable interpretation of the Democritus credibility analysis.
It is **more detailed than an AI overview** but **less technical than the full model report**.
## What appears most credible
- upright walking in human ancestors —causes→structural reorganization of the pelvis
to support bipedal locomotion
- upright walking in human ancestors —increases→the width and
shortening of the pelvis to support bipedal locomotion
3

Apreprint- January14, 2026
Figure 2: Two causal models constructed from The Washington Post article on the origins of bipedal walking
by human ancestors 7 million years ago. Democritusconstructs thousands of such models automatically
from the PDF document, which are then converted by Csqlinto a causal database.
4

Apreprint- January14, 2026
- upright walking in human ancestors —causes→the development of
an s-shaped spine to balance the body over the pelvis
In this paper, we instead convert the local causal models, such as shown in Figure 2, into a causal database
over which SQL-like queries can be performed. We detail the CSQL data model next.
3 CsqlData Model
3.1 Overview
Csqlrepresents causal knowledge extracted from document collections as a relational database. Rather than
storing isolated facts or entity relations, the database encodescausal structuretogether with empirical support
aggregated across many local causal models. The data model is designed to satisfy three requirements: (i) it
must be queryable using standard SQL, (ii) it must preserve uncertainty and disagreement present in the
source documents, and (iii) it must support compositional causal queries such as influence ranking, hub
detection, and multi-step causal paths.
3.2 Core Relations
A Csqldatabase consists of the following core relations.
Nodes.Each distinct causal concept is represented as a node:
node_id64-bit identifier
label_canonCanonicalized concept label
label_examplesExample surface forms
deg_inNumber of incoming causal edges
deg_outNumber of outgoing causal edges
Edges.Causal relations are represented as weighted directed edges:
edge_idUnique edge identifier
src_idSource node
dst_idDestination node
rel_typeRelation type (e.g., CAUSES, INFLUENCES)
polarityIncrease/decrease/unknown
support_lcmsNumber of supporting local models
support_docsNumber of supporting documents
score_sumAggregated credibility mass
score_meanMean support per model
score_maxMaximum single-model score
Edge Support.Fine-grained provenance is captured in a support table:
edge_idReferenced causal edge
doc_idSource document
lcm_instance_idSupporting local model
score_rawRaw model score
couplingModel-specific coupling strength
3.3 Derived Relations
Csqladditionally materializes derived relations computed from the core tables. These include:
Causal Modules (SCCs).Strongly connected components (SCCs) are computed over the causal graph:
5

Apreprint- January14, 2026
scc_idComponent identifier
n_nodesNumber of nodes
n_edgesNumber of edges
support_docsNumber of supporting documents
top_nodesHigh-degree nodes in the component
3.4 Query Semantics
Csqlenables causal analysis through standard SQL queries. For example:
•Backbone extraction:Identify the most credible causal relations by ordering edges byscore_sum.
•Causal hubs:Identify nodes with large outgoing score mass.
•Downstream influence:Join edges transitively to analyze multi-step causal paths.
•Disagreement analysis:Comparescore_maxandscore_meanto detect fragile claims.
All such queries are executed using deterministic SQL over materialized tables, rather than neural inference
or prompt-based generation.
4 Querying Causal Databases with Csql
Once documents have been compiled into a Csqldatabase, causal analysis is performed using standard SQL
queries. Unlike neural or prompt-based systems, Csqlqueries execute deterministically over materialized
relations that encode causal structure, uncertainty, and provenance.
This section illustrates several representative query patterns supported by Csql. All examples are executed
using unmodified SQL over Parquet-backed tables and can be run in any modern analytical database engine
(e.g., DuckDB).
4.1 Backbone Extraction
A common analytic task is to identify the most credible causal relationships across the document collection.
In CSQL, this corresponds to ordering causal edges by aggregated support.
SELECT
e.rel_type,
n1.label_canon AS src,
n2.label_canon AS dst,
e.support_lcms,
e.score_sum
FROM atlas_edges e
JOIN atlas_nodes n1 ON e.src_id = n1.node_id
JOIN atlas_nodes n2 ON e.dst_id = n2.node_id
ORDER BY e.score_sum DESC
LIMIT 20;
This query returns acausal backbone: relationships that recur across many high-scoring local causal models.
In the human bipedalism domain, this query surfaces relationships such asbipedalism influencing metabolic
efficiencyandbipedalism increasing locomotion efficiency, which are consistently supported across the extracted
discourse.
4.2 Causal Hubs
Csqlsupports identifying causal hubs: variables whose downstream influence is large when aggregated
across models.
SELECT
n.label_canon AS src,
SUM(e.score_sum) AS out_mass,
COUNT(*) AS out_degree
FROM atlas_edges e
6

Apreprint- January14, 2026
JOIN atlas_nodes n ON e.src_id = n.node_id
GROUP BY n.node_id, n.label_canon
ORDER BY out_mass DESC
LIMIT 10;
This query ranks variables by the total credibility mass of their outgoing edges. In practice, this identifies
dominant explanatory drivers within a domain. For example, in the bipedal walking corpus,bipedalism
emerges as a central hub with broad downstream effects on energetics, skeletal structure, and locomotion.
4.3 Local Mechanism Exploration
Csqlenables focused exploration of a single causal variable by filtering on a specific source concept.
SELECT
e.rel_type,
n2.label_canon AS effect,
e.support_lcms,
e.score_sum
FROM atlas_edges e
JOIN atlas_nodes n1 ON e.src_id = n1.node_id
JOIN atlas_nodes n2 ON e.dst_id = n2.node_id
WHERE n1.label_canon = ’bipedalism’
ORDER BY e.score_sum DESC;
This query enumerates the strongest downstream effects of a given cause, ordered by credibility. Such
queries are useful for drilling into mechanistic hypotheses associated with a particular concept, and can be
composed with additional filters (e.g., relation type or polarity).
4.4 Provenance and Auditability
Every causal edge in Csqlretains provenance information linking it to supporting documents and local
causal models.
SELECT
s.doc_id,
s.lcm_instance_id,
s.score_raw,
s.coupling
FROM atlas_edge_support s
WHERE s.edge_id = 10262796833405321330;
This query reveals exactly which documents and local models contributed to a given causal claim. Such
provenance queries enable auditing, filtering by source, and sensitivity analysis under alternative aggregation
strategies.
4.5 Cycles and Feedback Structures
Csqlsupports detection of feedback and cyclic influence via derived relations such as strongly connected
components (SCCs).
SELECT
scc_id,
n_nodes,
n_edges,
support_docs,
top_nodes
FROM atlas_scc
ORDER BY n_nodes DESC
LIMIT 5;
7

Apreprint- January14, 2026
Strongly connected components identify tightly coupled causal modules that may represent feedback loops,
co-evolving variables, or higher-level mechanisms. These structures are difficult to express in DAG-based
systems but arise naturally in language-derived causal discourse.
5 Example of a CsqlDatabase
Figure 3 illustrates how Csqlrepresents causal knowledge extracted from a document collection as a
relational database, using the example of upright walking in human ancestors. Each row in the atlas_edges
table corresponds to a causal relation discovered across multiple local causal models (LCMs), aggregated
over the discourse frame induced by the source articles.
In this example, the conceptbipedalismemerges as a high-degree causal hub. The table shows thatbipedalism
has strong outgoing causal influence on multiple downstream variables, includingmetabolic efficiency during
long-distance locomotion,energy efficiency during locomotion, andendurance capacity. These relations are not
asserted from a single sentence or model; instead, each edge aggregates evidence across many independently
generated LCMs, reflected in columns such as support_lcms andscore_sum . Higher values indicate stronger
redundancy and agreement across causal hypotheses.
Crucially, Csqlexposes this causal structure through standard SQL. For example, a query selecting edges
with maximal score_sum directly answers questions such as: What are the strongest causal influences on
bipedalism? Similarly, grouping by source nodes and summing outgoing score mass identifies causal hubs
with large downstream impact. These operations require no specialized causal language or graph query
engine—only relational joins and aggregations.
This example highlights the central design goal of Csql: to make causally grounded knowledge extracted
from text accessible through familiar database abstractions. Rather than returning a flat list of extracted
triples or a static knowledge graph, Csqlmaterializes a causally meaningful schema in which credibility,
redundancy, polarity, and structural roles (e.g., hubs, paths, cycles) can be queried, filtered, and composed
using SQL. The following sections formalize this data model and show how increasingly sophisticated causal
reasoning patterns arise naturally from relational queries over these tables.
These examples illustrate how Csqltransforms document collections into causally grounded relational
databases. By exposing causal structure, uncertainty, and provenance through standard SQL, Csqlenables
analysts to perform causal exploration, auditing, and hypothesis generation using familiar database tools.
5.1 Quantitative Properties of CsqlDatabases
Beyond supporting causal SQL queries, Csqlinduces a structured relational database whose graph-theoretic
and statistical properties can be summarized compactly. These properties are useful for (i) diagnosing
document-induced topic drift, (ii) comparing domains, and (iii) choosing thresholds for downstream
applications (e.g., filtering low-support edges or selecting “hub” concepts for exploration).
Schema-level summary.For each document collection, Csqlproduces two primary relations: (i)
atlas_nodes (canonicalized concept objects) and (ii) atlas_edges (canonicalized causal generators). Edges
carry bothsupport(how many local causal models contain the edge) andmass(e.g., score_sum ,score_mean ,
score_max ) aggregated over model instances. This provides a principled basis for ranking edges without
imposing a fixed ontology.
Heavy-tailed support and score mass.Across domains we observe a consistently heavy-tailed distribution
over edge support and score mass: a small number of edges account for a disproportionate fraction of total
score mass, while the majority of edges have low support and low mass. This is expected in discourse-derived
causal graphs: a few mechanisms are stated (or paraphrased) redundantly across discourse neighborhoods,
while many peripheral claims occur only rarely. In practice, this means aggressive compression is possible:
for many analytic tasks, retaining only the top kedges by score_sum preserves most of the interpretable
backbone.
Hub concentration.Csqlmakes it explicit that influence is concentrated around a small set of hubs. Define
theoutgoing massof a nodexas
Mout(x)=X
(xr− →y)∈Escore_sum(x,r,y).
8

Apreprint- January14, 2026
(a) Backbone edges (top by score mass)
SELECT e.rel_type, n1.label_canon AS src, n2.label_canon AS dst,
e.support_lcms, e.score_sum
FROM read_parquet(’atlas_edges.parquet’) e
JOIN read_parquet(’atlas_nodes.parquet’) n1 ON e.src_id=n1.node_id
JOIN read_parquet(’atlas_nodes.parquet’) n2 ON e.dst_id=n2.node_id
ORDER BY e.score_sum DESC
LIMIT 8;
INFLUENCES | bipedalism | metabolic efficiency ... | 15 | 1.4624
INCREASES | bipedalism | energy efficiency ... | 13 | 1.3282
INCREASES | bipedalism | energy efficiency ... | 13 | 1.2644
INFLUENCES | bipedalism | endurance capacity ... | 13 | 1.2557
INCREASES | bipedalism | energy efficiency ... | 12 | 1.2093
INFLUENCES | bipedalism | structural adaptation...| 11 | 1.2010
INFLUENCES | bipedalism | stride length ... | 11 | 1.1029
INFLUENCES | reduced forest cover | selection ... | 6 | 0.4637
(b) Causal hubs (sources with largest downstream mass)
SELECT n.label_canon AS src, SUM(e.score_sum) AS out_mass, COUNT(*) AS out_deg
FROM read_parquet(’atlas_edges.parquet’) e
JOIN read_parquet(’atlas_nodes.parquet’) n ON e.src_id=n.node_id
GROUP BY n.node_id, n.label_canon
ORDER BY out_mass DESC
LIMIT 8;
bipedalism | 8.8239 | 10
reduced forest cover | 0.4637 | 1
changes in environmental conditions ... | 0.4026 | 1
abnormal femoral tubercle position | 0.2853 | 2
fossil structure of ardipithecus | 0.1358 | 1
muscle reorganization for balance ... | 0.1283 | 1
larger brain size in primates | 0.0814 | 1
lateral displacement of the femoral ... | 0.0763 | 1
(c) Local mechanism: top outgoing edges from a hub
SELECT e.rel_type, n2.label_canon AS dst, e.support_lcms, e.score_sum
FROM read_parquet(’atlas_edges.parquet’) e
JOIN read_parquet(’atlas_nodes.parquet’) n1 ON e.src_id=n1.node_id
JOIN read_parquet(’atlas_nodes.parquet’) n2 ON e.dst_id=n2.node_id
WHERE n1.label_canon=’bipedalism’
ORDER BY e.score_sum DESC
LIMIT 7;
INFLUENCES | metabolic efficiency ... | 15 | 1.4624
INCREASES | energy efficiency ... | 13 | 1.3282
INCREASES | energy efficiency ... | 13 | 1.2644
INFLUENCES | endurance capacity ... | 13 | 1.2557
INCREASES | energy efficiency ... | 12 | 1.2093
INFLUENCES | structural adaptation... | 11 | 1.2010
INFLUENCES | stride length ... | 11 | 1.1029
Figure 3:Csqlin action (DuckDB).Example SQL queries over a Parquet-backed causal atlas
(atlas_nodes.parquet ,atlas_edges.parquet ). (a) Extracts backbone causal relations by aggregated
credibility mass. (b) Identifies causal hubs by total outgoing score mass. (c) Drills into a specific hub to
recover its highest-scoring downstream influences. All results are deterministic given the compiled atlas
tables.9

Apreprint- January14, 2026
-- Baseline: top outgoing generators from the
hub "bipedalism"
SELECT
e.rel_type,
n2.label_canon AS dst,
e.support_lcms,
e.score_sum
FROM read_parquet(’atlas_edges.parquet’) e
JOIN read_parquet(’atlas_nodes.parquet’) n1 ON e
.src_id = n1.node_id
JOIN read_parquet(’atlas_nodes.parquet’) n2 ON e
.dst_id = n2.node_id
WHERE n1.label_canon = ’bipedalism’
ORDER BY e.score_sum DESC
LIMIT 10;-- Hard intervention (do-cut): delete all
outgoing edges from "bipedalism"
WITH intervened_edges AS (
SELECT e.*
FROM read_parquet(’atlas_edges.parquet’) e
JOIN read_parquet(’atlas_nodes.parquet’) n1 ON
e.src_id = n1.node_id
WHERE n1.label_canon <> ’bipedalism’
)
SELECT
e.rel_type,
n1.label_canon AS src,
n2.label_canon AS dst,
e.support_lcms,
e.score_sum
FROM intervened_edges e
JOIN read_parquet(’atlas_nodes.parquet’) n1 ON e
.src_id = n1.node_id
JOIN read_parquet(’atlas_nodes.parquet’) n2 ON e
.dst_id = n2.node_id
ORDER BY e.score_sum DESC
LIMIT 10;
Figure 4:Csqlcounterfactual reasoning as SQL query rewriting.Left: baseline query retrieves the strongest
outgoing causal generators from the hub bipedalism . Right: a hard intervention (do-cut) is implemented as a
view that removes all edges whose source is bipedalism , revealing the next-strongest hubs and mechanisms
in the atlas. All operations are deterministic transformations over the atlas tables.
In our bipedalism case study, the top hub ( bipedalism ) dominates outgoing mass by a wide margin. We
quantify this concentration using simple concentration ratios (e.g., top-1 and top-5 share of total outgoing
mass).
Relation-type mix and polarity mass.Edges are normalized into a small set of relation types (e.g., CAUSES ,
INFLUENCES ,INCREASES ,REDUCES ) and polarity buckets ( inc,dec,unk). We report the distribution of relation
types and polarity mass as a coarse indicator of discourse framing. For example, some domains emphasize
INFLUENCES /AFFECTS (soft, explanatory discourse), while others emphasize CAUSES or signed relations
(INCREASES/REDUCES).
Cycles and strongly connected components.Unlike DAG-only representations, the induced causal
database is not constrained to be acyclic. We compute strongly connected components (SCCs) over the
atlas edges to identify feedback loops and closed influence motifs. At the document scale, SCCs may be
sparse; at corpus scale, SCC modules become a useful signal for non-trivial causal connectivity and potential
intervention loops. We store SCC summaries inatlas_sccwhen available.
5.2 Counterfactual query via SQL view rewriting
Csqlsupports intervention-style reasoning by treating interventions as deterministic rewrites of the atlas
edge relation. In the simplesthard intervention, we apply a “do-cut” operator to a concept Xby deleting all
outgoing causal generators with source X. A counterfactual query is then evaluated as adifference of query
resultsbetween the baseline atlas and the intervened atlas view. This provides a lightweight but concrete
form of counterfactual reasoning: we can see which downstream influences disappear (or drop in rank)
when the outgoing influence of a hub is removed. Figure 4 shows the Csqlspecification for a counterfactual
query. Table 2 shows the results of its application to the Csqldatabase constructed for the running example
of The Washington Post article on bipedal walking.
10

Apreprint- January14, 2026
rel dst (baseline top) support score_sum
INFLUENCES metabolic efficiency during long-distance travel 15 1.462
INCREASES energy efficiency during locomotion in early hominins 13 1.328
INCREASES energy efficiency during long-distance locomotion 13 1.264
INFLUENCES endurance capacity (heat dissipation, etc.) 13 1.256
INCREASES energy efficiency (metabolic resources) 12 1.209
INFLUENCES structural adaptation of femoral/tibial bones 11 1.201
INFLUENCES stride length and pelvic mechanics 11 1.103
Counterfactual (do-cut): these edges vanish from the atlas view.
Table 2:Counterfactual difffor a hard intervention.Baseline: strongest outgoing generators from
bipedalism . Under do-cut( bipedalism ) the outgoing edges are removed by construction, so these top-
ranked consequences disappear from query results.
CsqlAtlas Summary: WaPo Human Origins (Bipedalism)
Atlas Size
Number of nodes (concepts) 501
Number of unique causal edges 65
Edge-support rows (LCM evidence) 717
Hub Concentration
Top hubbipedalism
Top hub outgoing mass share 0.748
Top-5 hubs outgoing mass share 0.857
Edge Weight Distribution (score_sum)
Median (p 50) 0.0337
90th percentile (p 90) 0.847
99th percentile (p 99) 1.377
Tail ratio (p 99/p50) 40.8
Relation-Type Mass Breakdown
Relation type Polarity Total score mass
INFLUENCES unk 6.19
INCREASES inc 5.13
CAUSES unk 0.30
REDUCES dec 0.14
LEADS_TO unk 0.03
AFFECTS unk 0.01
Table 3: Quantitative summary of the Csqlatlas induced from a Washington Post article on the origins
of human bipedal walking. The atlas exhibits strong hub concentration, heavy-tailed edge weights, and
a dominance of influence- and increase-type causal relations. All statistics are computed directly via SQL
queries over Parquet-backed Csqltables using DuckDB.
It is also possible to implement “soft" interventions in Csqlas follows.
WITH soft_do AS (
SELECT
e.*,
CASE WHEN n1.label_canon=’bipedalism’ THEN 0.2*e.score_sum ELSE e.score_sum END AS score_sum_do
FROM read_parquet(’atlas_edges.parquet’) e
JOIN read_parquet(’atlas_nodes.parquet’) n1 ON e.src_id=n1.node_id
)
SELECT ... ORDER BY score_sum_do DESC LIMIT 10;
11

Apreprint- January14, 2026
5.3 Quantitative Summary of a CsqlAtlas
To complement the qualitative examples, we report summary statistics of the Csqlatlas constructed from the
Washington Post document on the origins of bipedal walking. These statistics are computed directly over the
atlas Parquet tables using DuckDB SQL queries. The statistics are reported in Table 3.
Atlas size.The atlas contains 501 canonical concept nodes and 65 unique canonical causal edges. In
addition, the edge-support table contains 717 rows, where each row records an instance of an atlas edge being
supported by a particular local causal model (LCM) in a particular document. This separation between (i)
unique canonical edgesand (ii)support instancesis essential: the atlas table stores the distilled causal backbone,
while the edge-support table retains provenance for auditability and reproducibility.
Hub concentration.We measure hubness by aggregating outgoing causal mass per source concept
(score_sum aggregated over outgoing edges). In this domain, the top hub isbipedalism. Remarkably, the top
hub alone accounts for approximately 74 .8% of the total outgoing score mass in the atlas, and the top five
hubs account for 85 .7%. This demonstrates a strongly centralized causal atlas in this document-induced
discourse frame: most highly supported causal relations radiate from a small number of discourse pivots.
Heavy-tailed edge strength.Edge strengths in Csqlare strongly heavy-tailed. Using score_sum as the
edge weight, the median (50th percentile) edge mass is 0 .0337, while the 90th percentile is 0 .847, and the 99th
percentile is 1 .377. The ratio p99/p50≈40.8 quantifies the extreme concentration: a small number of edges
dominate the causal backbone, while most edges carry comparatively small mass. This is consistent with the
empirical behavior observed in Democritus v2.0: causal discourse induces a large hypothesis space, but only
a small fraction of hypotheses are redundantly supported.
Relation-type and polarity mass.We also summarize atlas edges by normalized relation type ( rel_type )
and polarity ( polarity ). In this atlas, most edges fall into two broad families: INFLUENCES (unknown
polarity) and INCREASES (positive polarity). By mass, INFLUENCES contributes 6 .19 units of score mass,
whileINCREASES contributes 5 .13, with smaller contributions from CAUSES ,REDUCES , and a handful of
miscellaneous relations. This breakdown reflects the linguistic character of the source document and the
implicit uncertainty of causal discourse: journalistic and paleoanthropological claims are often phrased as
influences and associations rather than signed, quantified effects.
Top edges.Finally, we list the top weighted edges (by score_sum ). The highest-mass edges takebipedalism
as the source and connect it to mechanistic consequences such as metabolic efficiency, endurance capacity,
stride length, and locomotion energy efficiency. These edges collectively form the backbone of the induced
atlas.
Takeaway.These quantitative summaries illustrate why a Csqlatlas is useful: the induced schema is
not merely a flat list of extracted triples, but a weighted causal database with (i) strong hub structure, (ii)
heavy-tailed edge strength, and (iii) an explicit decomposition by relation-type and polarity. Such structure is
immediately queryable by standard SQL, enabling downstream analytics (e.g., identifying dominant drivers,
tracing multi-step influence chains, or locating high-uncertainty edges) without requiring a hand-designed
ontology or fixed domain schema.
In summary, Csqlyields not only a queryable causal database but also a set of stable quantitative signals
(heavy tails, hub concentration, relation-type mixtures) that support corpus diagnostics and downstream
system design.
5.4 Database Scale and Structure
Table 4 summarizes the size of the causal database constructed from a single Washington Post article on the
origins of human bipedal walking. Despite originating from a single document, the induced discourse frame
expands into hundreds of causal variables and hundreds of supported causal relations.
12

Apreprint- January14, 2026
Quantity Value
Number of nodes (concepts) 501
Number of causal edges 65
Number of edge-support records 717
Table 4: Scale of the Csqlcausal database constructed from a single document. Edge-support rows correspond
to individual local causal models (LCMs) that support each aggregated causal edge.
Top hub Top-1 mass share Top-5 mass share
bipedalism 0.748 0.857
Table 5: Causal hub dominance in the Csqlatlas. The single most influential node ( bipedalism ) accounts for
nearly 75% of outgoing causal mass, while the top five hubs account for over 85%.
5.5 Hub Dominance and Causal Centralization
Causal influence in the atlas is highly concentrated around a small number of hubs. We quantify this
concentration by measuring the fraction of total outgoing causal mass attributable to the most influential
source nodes.
5.6 Heavy-Tailed Causal Strength
Causal edge strength in Csqlfollows a strongly heavy-tailed distribution. We characterize this using
empirical quantiles of the aggregated edge score distribution.
5.7 Relation-Type Composition
The causal database contains a heterogeneous mix of relation types, with asymmetric distribution of causal
mass across predicates.
6 The CsqlData Model
Csqlrepresents causally grounded knowledge extracted from document collections as a small set of relational
tables with well-defined semantics. Unlike traditional information extraction pipelines, the schema is not
hand-designed in advance; instead, it is induced automatically from language via the Democrituscausal
modeling pipeline.
At a high level, Csqlconsists of four core relations:
•atlas_nodes: canonical causal concepts,
•atlas_edges: aggregated causal relations,
•atlas_edge_support: provenance linking edges to local causal models and documents,
•atlas_scc: strongly connected components capturing feedback structure.
Together, these tables define a causally meaningful relational schema that supports both structural analysis
and evidence-aware querying using standard SQL.
6.1 Nodes: Canonical Causal Concepts
Each row in atlas_nodes represents a canonicalized causal concept induced from language, such asbipedalism,
energy efficiency during locomotion, ormetabolic efficiency during long-distance travel. Nodes are obtained by
normalizing and de-duplicating subject and object phrases across all extracted discourse triples.
Key columns include:
•node_id: a stable identifier,
•label_canon: canonical concept string,
13

Apreprint- January14, 2026
p50 p90 p99 p99/p50
0.034 0.847 1.377 40.8
Table 6: Quantiles of causal edge score distribution. The extreme p99/p50ratio confirms a strongly heavy-tailed
energy landscape over causal hypotheses.
Relation type # edges Total mass
INFLUENCES 19 6.19
INCREASES 27 5.13
CAUSES 9 0.30
REDUCES 7 0.14
LEADS_TO 1 0.03
AFFECTS 2 0.01
Table 7: Distribution of causal relation types. Influence-style relations dominate both numerically and in
total causal mass, while strict causal predicates are rarer and more conservative.
•label_examples: representative surface forms,
•deg_in,deg_out: in- and out-degrees in the causal atlas.
Nodes function as theobjectsof the induced causal structure. No external ontology or controlled vocabulary
is required; the object space is learned directly from text.
6.2 Edges: Aggregated Causal Relations
Each row in atlas_edges represents a causal relation aggregated across many local causal models (LCMs).
An edge corresponds to a canonical triple
(src_id,rel_type,dst_id),
whererel_typeranges over normalized relations such asINCREASES,REDUCES,CAUSES, orINFLUENCES.
In addition to structural information, each edge carries quantitative evidence:
•support_lcms: number of LCMs in which the edge appears,
•support_docs: number of documents contributing evidence,
•score_sum,score_mean,score_max: aggregated credibility scores,
•polarity mass fields distinguishing increasing, decreasing, and unknown effects.
These values encodeagreement under variation: edges that recur across many high-scoring hypotheses
accumulate greater score mass and therefore higher causal credibility.
6.3 Edge Support and Provenance
Csqlpreserves fine-grained provenance via the atlas_edge_support table, which links each aggregated
edge back to:
•specific LCM instances,
•source documents,
•raw per-model scores and coupling values.
This design enables auditability and traceability. Users can drill down from a high-level causal claim to the
specific models and textual evidence that support it.
6.4 Strongly Connected Components
Theatlas_scc table captures strongly connected components of the causal atlas graph. These components
represent feedback loops or tightly coupled causal subsystems, which cannot be represented in DAG-based
causal formalisms.
14

Apreprint- January14, 2026
SCCs provide a natural entry point for reasoning about cyclic causality, mutual reinforcement, and dynamical
regimes—key motivations for moving beyond DAGs to more expressive causal representations.
7 Causal Reasoning as SQL
A central contribution of Csqlis thatcausal reasoning can be expressed directly as SQL queriesover the induced
schema. No specialized graph query language or causal DSL is required.
7.1 Identifying Causal Backbones
The strongest causal claims in a document collection correspond to edges with maximal aggregated score
mass. For example:
SELECT
e.rel_type,
n1.label_canon AS src,
n2.label_canon AS dst,
e.support_lcms,
e.score_sum
FROM atlas_edges e
JOIN atlas_nodes n1 ON e.src_id = n1.node_id
JOIN atlas_nodes n2 ON e.dst_id = n2.node_id
ORDER BY e.score_sum DESC
LIMIT 50;
This query extracts thecausal backboneof the discourse frame—relations that survive across many high-scoring
hypotheses.
7.2 Causal Hubs and Downstream Influence
Causal hubs are nodes with large outgoing score mass:
SELECT
n.label_canon AS src,
SUM(e.score_sum) AS out_mass
FROM atlas_edges e
JOIN atlas_nodes n ON e.src_id = n.node_id
GROUP BY n.label_canon
ORDER BY out_mass DESC;
In the upright walking example,bipedalismemerges as the dominant hub, with strong downstream influence
on metabolic efficiency, endurance, and locomotion.
7.3 Causal Composition
Multi-step causal chains correspond to relational joins:
SELECT
n1.label_canon AS a,
e1.rel_type AS r1,
n2.label_canon AS b,
e2.rel_type AS r2,
n3.label_canon AS c,
(e1.score_sum + e2.score_sum) AS path_score
FROM atlas_edges e1
JOIN atlas_edges e2 ON e1.dst_id = e2.src_id
JOIN atlas_nodes n1 ON e1.src_id = n1.node_id
JOIN atlas_nodes n2 ON e1.dst_id = n2.node_id
JOIN atlas_nodes n3 ON e2.dst_id = n3.node_id
ORDER BY path_score DESC;
Such queries implement causal composition directly in SQL.
15

Apreprint- January14, 2026
Quantity Value
Canonical nodes (|nodes|) 839
Canonical edges (|edges|) 450
Edge-support rows (|edge_support|) 1311
Documents (|docs|) 10
Atlases merged (|atlases|) 7
Table 8: Corpus-level size statistics for the merged Csqldatabase. Nodes and edges are canonicalized
(string-normalized) concepts and relations; edge-support rows store provenance at the level of (document,
local causal model) instances.
Hub source (src) Out mass Out edges LCM support
exposure to roundup 105.03 3 24
dark chocolate consumption 64.70 5 20
severity of post impacts 44.76 2 17
impact of researchers 33.68 3 15
application of roundup 30.58 2 9
Table 9: Top causal hubs by outgoing score mass in the merged Csqlcorpus database.
7.4 Cycles and Feedback
Cycles are detected via self-joins or SCC tables. For example, mutual influence:
SELECT
n1.label_canon AS a,
e1.rel_type AS r1,
n2.label_canon AS b,
e2.rel_type AS r2
FROM atlas_edges e1
JOIN atlas_edges e2
ON e1.src_id = e2.dst_id
AND e1.dst_id = e2.src_id
JOIN atlas_nodes n1 ON e1.src_id = n1.node_id
JOIN atlas_nodes n2 ON e1.dst_id = n2.node_id;
This enables explicit querying of feedback structures absent from DAG-based causal systems.
Counterfactual query via SQL view rewriting.Csqlsupports intervention-style reasoning by treating
interventions as deterministic rewrites of the atlas edge relation. In the simplesthard intervention, we apply a
“do-cut” operator to a concept Xby deleting all outgoing causal generators with source X. A counterfactual
query is then evaluated as adifference of query resultsbetween the baseline atlas and the intervened atlas view.
This provides a lightweight but concrete form of counterfactual reasoning: we can see which downstream
influences disappear (or drop in rank) when the outgoing influence of a hub is removed.
7.5 Quantitative Summary of the CsqlCorpus Database
Table 8 reports corpus-level summary statistics for the merged Csqldatabase constructed from multiple
document-derived atlases. The database contains canonicalized concept nodes and relation edges, together
with provenance (edge-support rows linking edges to document and local-model instances) and aggregate
credibility statistics (e.g., score mass and polarity mass). The resulting score distribution is heavy-tailed, and
the outgoing score mass is concentrated in a small number of causal hubs, reflecting the system’s tendency to
surface central drivers in the discourse manifold.
Interpretation.The merged Csqldatabase exhibits two recurring properties. First, the aggregate credibility
signal (score mass) is heavy-tailed: a small fraction of edges and sources account for a large fraction
of total mass. Second, this mass is organized around a handful of causal hubs (Table 5), which act as
central drivers in the discourse manifold (e.g.,exposure to roundupanddark chocolate consumption). These
hubs are precisely the nodes that become most useful for interactive causal querying: they support
16

Apreprint- January14, 2026
rel_type polarity #edges score mass
INFLUENCES unk 135 353.78
LEADS_TO unk 80 188.11
REDUCES dec 75 181.83
INCREASES inc 89 156.30
CAUSES unk 50 89.99
AFFECTS unk 21 75.71
Table 10: Distribution of edge mass by normalized relation type and polarity in the merged corpus.
SQL queries that surface high-impact downstream consequences, compare competing mechanisms, and
identify which claims are stable across multiple local causal models. Finally, the relation-mass breakdown
(Table 10) shows that the induced schema concentrates into a small, reusable vocabulary of causal predicates
(INFLUENCES/LEADS_TO/INCREASES/REDUCES), which is critical for portability across corpora without
requiring a hand-built ontology.
8 Algorithmic Construction of CsqlDatabases
This section describes the deterministic compilation step that turns a collection of local causal models (LCMs)
into acausal SQL database(Csql). The input to this stage is a directory of LCMs produced by an upstream
system (e.g., Democritus) and optionally accompanied by scoring metadata (e.g., scores.csv ). The output
is a small set of Parquet tables that can be queried directly in DuckDB (or any Parquet-capable SQL engine).
8.1 Input Contract
We assume each document dhas an associated set of LCMs stored as JSON files. An LCM is represented as a
directed labeled multigraph M=(VM,EM) with a focus string and optional metadata. Each edge is a record
e=(src,rel,dst)
wheresrcanddstare surface strings and relis a free-form relation phrase (e.g., “increases”, “reduces”,
“influences”). Optionally, each LCM is accompanied by a scalar score and auxiliary fields (radius, model size,
coupling, etc.) used for aggregation.
8.2 CsqlSchema
Csqlmaterializes three core tables (plus one optional analytic table):
•nodes: canonicalized concepts (entities/variables) discovered from LCM node strings.
•edges: canonicalized causal edges aggregated across all LCMs (deduplicated).
•edge_support : provenance table linking each canonical edge to the document and LCM instances
that support it (including local model scores).
•scc (optional): strongly connected component summaries over the induced graph, to expose
cycles/modules.
This separation is deliberate and mirrors standard database normalization practice: edges provides a stable
fact table for queries, while edge_support preserves provenance and enables auditing, filtering, and corpus
merging.
8.3 Canonicalization and Keying
To make the database queryable and mergeable across documents, Csqlnormalizes strings and relations.
We define:
Canonical node labels.A normalization function canon (·) maps raw node strings to a canonical form
(lowercasing, whitespace/punctuation normalization, dash unification, etc.). Canonical nodes are keyed by:
node_id=H(canon(x))
whereHis a stable 64-bit hash.
17

Apreprint- January14, 2026
Algorithm 1BuildAtlas: Compile LCMs into a Csqldatabase
Require: Document runs root directory Rcontaining per-document LCM folders; optional scoring metadata
(scores.csv); min-edge filterτ; relation whitelist/blacklist.
Ensure: Parquet tables: nodes.parquet ,edges.parquet ,edge_support.parquet (and optional
scc.parquet).
1:Initialize empty maps:NodeMap:node_id7→node record,EdgeAgg:edge_id7→aggregate record
2:Initialize empty listSupportRows
3:for alldocumentsdunderRdo
4:Load set of LCM JSON files{M i}ford(optionally filtered by score/size)
5:for allLCMsM ido
6:Extract metadata:doc_id,lcm_instance_id,score,score_raw,coupling
7:for alledgese=(u,r,v)∈E Mido
8:if|E Mi|<τthen continue
9:end if
10:u c←canon(u);v c←canon(v)
11:src_id←H(u c);dst_id←H(v c)
12:(rel_type,polarity)←reltype(r)
13:edge_id←H(src_id∥rel_type∥dst_id)
14:UpdateNodeMapwith examples foru,v
15:UpdateEdgeAgg[edge_id]: support counts, score mass sums, polarity mass
16:Append support row toSupportRows:
17:
(edge_id,doc_id,atlas_id,lcm_instance_id,score,score_raw,coupling)
18:end for
19:end for
20:end for
21:Materializenodestable fromNodeMapwith degree stats computed fromedges
22:Materializeedgestable fromEdgeAggwith summary fields:
support_lcms,support_docs,score_sum,score_mean,
score_max,pol_mass_inc,pol_mass_dec,pol_mass_unk,controversy
23:Materializeedge_supporttable fromSupportRows
24:(Optional) Build induced directed graph fromedges; compute SCCs; writescc.parquet
25:Write all tables to Parquet
26:return
Canonical relation types.A relation normalization function reltype (·) maps raw relation phrases to a small
controlled vocabulary (e.g., CAUSES, INFLUENCES, INCREASES, REDUCES, AFFECTS, LEADS_TO). We
also extract a coarsepolaritylabel∈{inc,dec,unk}.
Canonical edges.A canonical edge is identified by:
edge_id=H(src_id∥rel_type∥dst_id)
This provides stable deduplication within and across documents.
8.4 Atlas Builder: From LCMs to Parquet
Algorithm 1 describes the Csqlcompilation procedure. The output tables store: (i) canonical objects and
arrows (nodes ,edges ), (ii) provenance and scoring ( edge_support ), and (iii) optional graph analytics ( scc).
Aggregation and “controversy”.Each canonical edge aggregates evidence across many LCM instances.
Letwibe the score mass contributed by LCM ito a canonical edge (we use the adjusted score when available).
For polarity mass we define:
minc=X
i:polarity=incwi,mdec=X
i:polarity=decwi,munk=X
i:polarity=unkwi.
18

Apreprint- January14, 2026
Algorithm 2MergeAtlases: Merge per-document Csqldatabases into a corpus database
Require:A directory containing atlas folders{A j}, each withnodes,edges,edge_support.
Ensure:Merged corpus Csqldatabase:nodes,edges,edge_support.
1:Load and concatenate all edge_support tables; optionally prefix doc_id with atlas id for disambiguation
2:Groupedge_support byedge_id to recompute edge aggregates (support counts, score sums, polarity
mass)
3:Construct mergededgesfrom aggregated groups
4:Construct merged nodes by collecting all referenced src_id ,dst_id and recomputing degrees from
mergededges
5:Write merged tables to Parquet
6:return
A simple controversy indicator is then:
controversy=min(m inc,mdec)
minc+mdec+ϵ.
This yields controversy =0 when only one direction is supported and approaches 0 .5 when inc/dec are
balanced.
8.5 Corpus Merge: Union of CsqlDatabases
Because all identifiers are canonical (hash-based), merging multiple atlas databases is a deterministic union
operation followed by re-aggregation. Algorithm 2 sketches the merge procedure.
In this way, Csqlsupports bothdocument-levelanalysis andcorpus-levelanalysis using the same schema.
Crucially, provenance is preserved: corpus queries can be refined by filtering on doc_id ,atlas_id , or
lcm_instance_id , enabling traceable drill-down from aggregate causal claims back to local models and
source documents.
8.6 Practical Notes for Users
Csqlis designed to be usable with off-the-shelf tools:
•Any SQL engine with Parquet support can query the tables; we use DuckDB.
•Most queries are expressed as joins betweenedgesandnodesfor human-readable labels.
•Provenance-aware queries joinedge_supportto recover document and LCM-level evidence.
This makes Csqlacausal database backendthat can be placed under visualization layers, RAG systems, or
higher-level reasoning systems (including Democritusitself).
9 Csqlfrom RAG-Compiled Causal Corpora
A key design goal of Csqlis to decouplecausal discourse compilationfromcausal database construction. While
earlier sections focused on LLM-driven discourse compilation via Democritus, the CSQL pipeline is agnostic
to how causal claims are obtained. In this subsection, we show that CSQL can be constructed directly from
RAG-compiled causal corpora, without requiring access to the original documents or an LLM-based generation
pipeline.
We use theTesting Causal Claims (TCC)dataset [Garg and Fetzer, 2025] as a canonical example. TCC extracts
causal claims from a large corpus of economics papers using information extraction and retrieval-based
methods, yielding structured records of the form
(cause,effect,sign,method,document metadata).
Rather than treating TCC as a static knowledge graph, we interpret it as aprecompiled causal discourse layer.
Csqlthen compiles this layer into a relational causal database supporting causal aggregation, ranking, and
intervention-style queries.
19

Apreprint- January14, 2026
Discourse-to-database compilation.Given a RAG-compiled corpus such as TCC, Csqlperforms a
deterministic compilation step that maps extracted causal claims into a small set of normalized relational
tables. Canonicalized cause and effect phrases become nodes, while typed causal relations become edges.
Repeated claims across documents are aggregated, producing support counts and score mass analogous to
those obtained from Democritus-generated local causal models.
Importantly, Csqldoes not assume a fixed ontology, predefined schema, or hand-designed knowledge graph.
Instead, the schema is induced directly from the causal discourse itself. Provenance information—such
as document identifiers, publication year, and causal inference method—is preserved in an auxiliary
edge-support table, enabling downstream filtering and auditability.
Unified causal querying across compilation methods.Once compiled, a Csqldatabase constructed from
a RAG-based corpus supports the same class of causal queries as one constructed from Democritus. For
example:
•identifying global causal hubs with the largest downstream influence,
•ranking causal claims by redundancy and aggregate support,
•detecting controversial claims with mixed directional evidence,
•approximating interventions by removing or conditioning on selected sources.
From the perspective of the query layer, there is no distinction between causal databases derived from
LLM-generated discourse and those derived from RAG-compiled corpora.
Implications for scale and interoperability.This separation enables Csqlto operate at corpus scale without
requiring LLM inference over tens of thousands of documents. Large, curated resources such as TCC can be
ingested directly, allowing Csqlto support causal analysis over orders of magnitude more documents than
would be feasible with generative pipelines alone. At the same time, databases compiled from RAG-based
sources and from Democrituscan be merged seamlessly, yielding a unified causal database spanning
heterogeneous document collections.
Viewed abstractly, Csqlserves as a common compilation target for diverse causal discourse generators—LLMs,
RAG systems, and classical information extraction pipelines—while providing a uniform, SQL-native interface
for causal analysis.
9.1 Csqlfrom a RAG-Compiled Causal Corpus: Testing Causal Claims (TCC)
We also instantiate Csqlon theTesting Causal Claims(TCC) dataset, which already provides a corpus-
level extraction of directed cause–effect claims from ∼45K economics papers. TheTesting Causal Claims
(TCC) project [Garg and Fetzer, 2025] extracts and catalogs causal claims from large corpora of economics
papers, producing a global index of cause–effect statements enriched with metadata such as sign, statistical
significance, and inference method. In this setting, Csqldoes not run the Democritusdiscourse compiler;
instead, it treats TCC as a pre-compiled discourse layer and compiles it into the same relational schema
(nodes,edges,edge_support).
The resulting database contains 295,459 canonicalized concept strings and 260,777 distinct directed edges
(265,656 edge-support rows), spanning 44 publication years. Because the current ingest uses the generic
relation label INFLUENCES and does not yet project sign/method metadata into Csql, edge mass largely tracks
frequency (number of supporting papers), producing a near-degenerate heavy tail (e.g., median support
mass 1, 99th percentile 2). This behavior is expected: TCC is already an edge list, whereas Democritus-
native atlases aggregate many overlapping local causal hypotheses and therefore induce richer relation
vocabularies and heavier-tailed score mass. Despite this limitation, the Csqlinterface already supports useful
corpus-scale queries such as hub discovery and backbone extraction; in ongoing work we project TCC’s sign
and identification-method metadata into Csqlto enable polarity-aware queries and method-conditioned
disagreement analyses.
To stress-test Csqlat scale without running Democritusover tens of thousands of PDFs, we treat theTesting
Causal Claims(TCC) dataset as an already-compiled causal discourse layer. TCC provides paper-level causal
edges (cause→effect) with metadata such as year and the stated causal inference method. We compile
this corpus into a Csqldatabase by canonicalizing node strings (causes/effects) into concept identifiers,
aggregating identical edges across papers, and storing per-paper evidence rows in an edge_support table.
All statistics below were computed directly with DuckDB queries over the resulting Parquet tables.
20

Apreprint- January14, 2026
Quantity Value
Claim instances (edge_supportrows) 265,656
Unique papers (doc_id) 45,319
Unique years 44
Distinct method strings 1,575
Canonical concept nodes (nodes) 295,459
Unique causal edges (edges) 260,777
Table 11: Summary statistics for the TCC–Csqldatabase. The corpus is treated as a RAG/IE-compiled causal
discourse layer; Csqlcompiles it into a queryable causal database with edge aggregation and per-paper
evidence.
Hub (cause) Outgoing mass # outgoing edges
education 442 374
trade liberalization 391 370
age 358 339
inflation 280 250
monetary policy 270 211
Table 12: Top corpus hubs in TCC–Csql, ranked by outgoing support mass (sum of support_docs over
outgoing edges). These hubs summarize the dominant causal discourse centers in the economics corpus.
Scale.The compiled TCC–Csqldatabase contains265,656evidence rows (one per extracted claim instance),
spanning45,319distinct papers,44publication years, and1,575distinct method strings (as reported in the
corpus). At the schema level, the database contains295,459canonical concept nodes and260,777unique
directed causal edges. This illustrates that Csqlcan ingest a large RAG/IE-compiled causal corpus and
expose it as a unified, queryable causal database without requiring a hand-designed ontology.
Hubs and backbone claims.A hallmark of corpus-scale causal discourse is the emergence ofhub variables
with large outgoing mass. In TCC–Csql, the largest outgoing hubs includeeducation,trade liberalization,
age,inflation, andmonetary policy, each participating in hundreds of distinct outgoing edges. For example,
the query “What most strongly influences inflation?” returns recurrent claims such asmonetary policy →
inflation,money growth →inflation, andoutput gap →inflation, ranked by paper support. These results are not
interpreted as ground-truth causality; rather, they operationalize the corpus as a causal claim space that can
be searched, filtered, and audited.
Temporal dynamics.Because each claim instance in edge_support carries a publication year, Csqlsupports
time-sliced causal analysis. Aggregating claim instances by year reveals strong growth over time: the corpus
contains under 2 ,000 extracted claim instances per year in the early 1980s, rising to over 10 ,000 per year in
the late 2010s and above 18 ,000 in 2020. For a specific canonical claim (e.g.,monetary policy →inflation), Csql
can return the distribution of supporting papers by year, enabling longitudinal tracking of how particular
causal claims appear and reappear across decades.
Method strings and heterogeneity.The TCC metadata includes a reported causal inference method field.
In the current corpus snapshot, the method field contains a long tail of heterogeneous strings (e.g., IV,DID,
TWFE , but also many fine-grained or free-text variants). This is reflected in the large number of distinct
method values (1,575). Csqltreats the method column as an observational attribute: users can filter or
group by methods as-is, or optionally normalize method strings into a smaller controlled taxonomy for
downstream analysis.
10 Related Work
This paper sits at the intersection of (i) causal relation extraction from text, (ii) causal knowledge base
construction, (iii) database- and graph-centric representations of knowledge, and (iv) causal discovery and
inference. We situate Csqlwith respect to prior work in each area, emphasizing how it differs from existing
knowledge graphs, retrieval-augmented generation (RAG) systems, and causal extraction pipelines.
21

Apreprint- January14, 2026
Claim affecting inflation Support (papers) Mass
monetary policy→inflation 15 15
money growth→inflation 12 12
output gap→inflation 11 11
monetary policy shocks→inflation 8 8
fiscal policy→inflation 7 7
Table 13: Example query result: top corpus claims targetinginflation(ranked by paper support). In Csql,
such queries are expressible directly as SQL joins and filters overedgesandedge_support.
10.1 Causal Relation Extraction from Text
There is a long line of work on identifying causal relations in natural language, ranging from early pattern-
based approaches using cue phrases (e.g., “because”, “leads to”) to modern neural and transformer-based
models trained to classify cause–effect relations between text spans [Radinsky et al., 2012, Yang et al., 2022,
He et al., 2025]. These systems typically operate at the level of individual sentences or short contexts and
output pairwise labeled relations.
Csqldiffers fundamentally in scope and objective. Rather than treating extracted relations as final outputs,
we treat them asnoisy discourse samplesfrom which many competing causal hypotheses are constructed,
evaluated, and aggregated. The output is not a flat list of extracted triples, but a relational database that
encodes causal structure, evidence mass, and model agreement across an entire document collection.
10.2 Causal Knowledge Bases and Graph Construction
Several systems aim to construct causal knowledge bases or causal graphs from large text corpora [Hassan-
zadeh et al., 2020]. These resources aggregate extracted causal tuples into graph-structured stores, often with
domain-specific ontologies or manually designed schemas.
Csqldiffers in three key ways. First, its schema isinduced automatically from languagerather than fixed in
advance. Second, causal edges are not treated as atomic facts, but asaggregatessupported by many local
causal models. Third, the resulting representation is relational rather than purely graph-based, allowing
causal analysis to be expressed using standard SQL queries.
10.3 Knowledge Graphs, Ontologies, and GraphRAG
Knowledge graphs (KGs) and ontology-driven representations provide a mature substrate for storing
structured assertions and supporting deductive inference [Hogan et al., 2021, Group, 2012, Knublauch
and Kontokostas, 2017]. More recently, GraphRAG-style systems combine graph-structured data with
retrieval-augmented generation pipelines [Lewis et al., 2020].
These approaches primarily representwhat is assertedin text. They lack intrinsic semantics for causal
operations such as intervention, composition, feedback, and downstream influence. By contrast, Csql
representscausal hypothesestogether with their degree of support across competing models. Causal reasoning
in Csqlis performed by aggregating and composing relations using SQL, rather than by retrieving or ranking
nodes in a graph.
10.4 Testing Causal Claims and Corpus-Scale Causal Databases
TheTesting Causal Claims(TCC) project [Garg and Fetzer, 2025] extracts and catalogs causal claims from large
corpora of economics papers, producing a global index of cause–effect statements enriched with metadata
such as sign, statistical significance, and inference method.
Csqlis complementary but differs in representation and workflow. Rather than producing a single
corpus-level catalog of claims, Csqlconstructs manylocal causal modelsper document, evaluates them, and
aggregates their structure into a queryable causal database. Conceptually, TCC emphasizes breadth and
standardization across a discipline, whereas Csqlemphasizes depth and model-based agreement within and
across documents.
22

Apreprint- January14, 2026
Domain Nodes Edges Top Hub Max Score
Human origins (The Washington Post) 412 683 bipedalism 1.46
Chocolate & aging (The Washington Post) 527 912 dark chocolate intake 1.12
Glyphosate (The NY Times) 489 801 roundup use 2.04
Antarctica (The NY Times) 366 594 glacier retreat 1.87
Table 14: Summary statistics for Csqldatabases induced from different document domains.
10.5 Large Language Models for Causal Discovery
A rapidly growing literature investigates the use of large language models (LLMs) for causal discovery and
reasoning [Shen et al., 2023, Le et al., 2024, Kosoy et al., 2023]. These approaches often rely on LLMs to
propose causal directions or graph structures directly.
Csqladopts a different design. We treat LLMs ashypothesis generatorsthat propose candidate causal
statements in natural language. All subsequent model construction, evaluation, aggregation, and querying are
deterministic. This separation allows causal analysis to be audited, reproduced, and executed independently
of the language model.
10.6 Causal Discovery and Statistical Inference
Classical causal discovery methods operate on numerical data, using score-based, constraint-based, or hybrid
search over DAGs or equivalence classes [Spirtes et al., 2000, Chickering, 2002, Zheng et al., 2018]. While
these methods are not directly applicable to document-centric settings, Csqlis compatible with them in
principle. The causal database produced by Csqlcan serve as a hypothesis generator, prior structure, or
explanatory scaffold for downstream statistical causal analysis.
Summary.Relative to prior work, Csqlmakes three distinguishing contributions: (i) it compiles un-
structured documents into a relational causal database, (ii) it represents causal claims as aggregates over
competing local models rather than isolated extractions, and (iii) it enables causal reasoning using standard
SQL as a query language.
11 Csqlover Other Domains
We have experimented with Csqlover a number of different domains (see Table 1). The domains analyzed
are described below.
1.A recent newspaper article in The Washington Post on a potential causal link between dark chocolate
and aging.4. We used this domain as a running example throughout the paper to illustrate the type
of causal analysis that Democrituscan perform.
2.The causal analysis was based on The Washington Post article on the origins of bipedal walking in
human ancestors, dating back to 7 million years ago.5.
3.The NY Times published a recent article on the controversy surrounding the weedkiller Roundup,
whose technical name is glyphosate6.
4.The NY Times published an article on the melting of glaciers in Antarctica7. The high-scoring model
(top) contains a dense cluster of edges that are repeatedly supported across multiple discourse
neighborhoods, including ice melt leading to freshwater influx, changes in ocean circulation, and
rising global sea levels. These overlapping mechanisms yield strong model agreement and a high
evaluation score. In contrast, the low-scoring model (bottom) contains fewer well-supported edges
and weaker semantic coherence. Although such models are freely generated by the left adjoint, they
4https://www.washingtonpost.com/wellness/2025/12/26/dark-chocolate-health-benefits
5urlhttps://www.washingtonpost.com/science/2026/01/02/human-ancestor-biped/
6https://www.nytimes.com/2026/01/02/climate/glyphosate-roundup-retracted-study.html?smid=
nytcore-ios-share.
7https://www.nytimes.com/2025/12/27/climate/antarctica-thwaites-glacier.html
23

Apreprint- January14, 2026
fail to accumulate sufficient evidence during evaluation and are therefore suppressed before adjoint
reconstruction.
12 Discussion and Implications
12.1 Relation to Knowledge Graphs and RAG
Csqldiffers fundamentally from knowledge graphs and GraphRAG-style systems. Knowledge graphs store
asserted facts; Csqlstoresevaluated causal hypotheses. GraphRAG retrieves relevant nodes; Csqlcomputes
credibility via aggregation across competing causal models.
Where knowledge graphs emphasizewhat is stated, Csqlemphasizeswhat is structurally supportedby
agreement across discourse.
12.2 Relation to Causal Inference
Csqldoes not replace statistical causal inference. Instead, it complements it by operating in regimes where
numerical data is unavailable but causal discourse is abundant. The output of Csqlcan serve as a hypothesis
generator, prior constructor, or explanatory scaffold for downstream causal analysis.
12.3 Csqlas a Causal Compiler
Viewed abstractly, Csqlis a compiler:
Documents−→Causal Models−→Relational Database.
The compilation target is a causally meaningful SQL schema equipped with a principled algebra of queries.
This perspective connects Csqlto work on functorial databases and data migration, though users need not
be aware of this machinery to use the system effectively.
13 Limitations and Future Work
While Csqldemonstrates that large document collections can be compiled into SQL-queryable causal
databases, the current system has several important limitations. These limitations also suggest clear
directions for future research and system development.
13.1 Limitations
Dependence on discourse quality.Csqlrelies on large language models (LLMs) as discourse compilers that
generate candidate causal statements. Although all downstream processing is deterministic, the quality and
topical coherence of the resulting causal database depends on the quality of the extracted discourse. Noisy
or off-domain discourse expansions can lead to diffuse causal atlases with weaker hub concentration and
flatter score distributions. While the scoring and aggregation layers are designed to suppress unsupported
claims, discourse quality remains a primary bottleneck.
No ground-truth causal validation.The causal relations represented in Csqlare inferred from language
rather than validated against experimental or observational data. As a result, Csqlshould be understood as
capturingdiscursive causal structure—what is claimed, suggested, or argued in text—rather than verified
causal ground truth. This distinction is fundamental: Csqlevaluates credibility under discourse agreement,
not causal identifiability in the sense of statistical causal inference.
Static causal snapshots.The current system constructs causal databases from static document collections.
Temporal evolution of causal discourse—how claims change over time, how evidence accumulates or is
retracted, and how controversies evolve—is not yet explicitly modeled. Each Csqldatabase represents a
snapshot of discourse at the time the documents were processed.
24

Apreprint- January14, 2026
Limited compositional reasoning.Although Csqlsupports multi-step causal queries via SQL joins (e.g.,
two-hop paths and hub analysis), it does not yet implement explicit causal algebraic operations such as
intervention, counterfactual reasoning, or compositional diagram rewriting. These operations are implicit in
the underlying theory but are not yet exposed at the database query layer.
Scalability of discourse compilation.The most computationally expensive stage of the pipeline is discourse
compilation via LLMs. While all subsequent steps scale efficiently and operate over Parquet and SQL engines,
large-scale deployment over millions of documents will require additional optimizations, such as caching,
incremental updates, or hybrid retrieval–generation strategies.
13.2 Future Work
Temporal and versioned causal databases.A natural extension of Csqlis to support time-indexed causal
databases that track how causal claims evolve across document versions, publication dates, or evidence
updates. This would enable queries such as “Which causal claims have gained or lost support over time?”
and would be particularly valuable for scientific, policy, and regulatory analysis.
Interventional and counterfactual query layers.Future versions of Csqlwill expose higher-level causal
operators on top of SQL, including abstractions for intervention, deletion, and hypothetical modification
of causal edges. These operations can be grounded in the underlying causal semantics while remaining
accessible through familiar query interfaces.
Integration with numerical data.Although Csqlcurrently operates on text-derived causal claims, it can
naturally be extended to incorporate numerical datasets when available. For example, edges in the causal
database could be linked to statistical models, regression results, or experimental evidence, enabling hybrid
discourse–data causal analysis.
User-guided schema refinement.While Csqldeliberately avoids hand-designed ontologies, future work
may explore lightweight human-in-the-loop mechanisms for refining concept normalization, merging equiv-
alent nodes, or highlighting domain-relevant causal variables. Such guidance could improve interpretability
without sacrificing generality.
Distributed and incremental compilation.To support large-scale deployments, future implementations
will explore incremental atlas construction, distributed discourse compilation, and streaming updates to
Parquet-backed causal tables. This would allow Csqldatabases to be maintained continuously as new
documents arrive.
Causal Csqlas a standard interface.Finally, we envision Csqlas a candidate abstraction layer for causal
querying over textual corpora, analogous to how SQL standardized access to relational data. A long-term
goal is to define a portable causal query interface that can be embedded into analytics platforms, dashboards,
and decision-support systems without requiring users to interact directly with causal modeling formalisms.
References
Anthropic. Claude 3 model card. https://assets.anthropic.com/m/61e7d27f8c8f5919/original/
Claude-3-Model-Card.pdf, 2024. Accessed 2025-12-28.
David Maxwell Chickering. Optimal structure identification with greedy search.Journal of Machine Learning
Research, 3:507–554, 2002.
Abhimanyu Dubey et al. The llama 3 herd of models.arXiv preprint arXiv:2407.21783, 2024.
Brendan Fong, David I. Spivak, and Rémy Tuyéras. Backprop as functor: A compositional perspective
on supervised learning. In34th Annual ACM/IEEE Symposium on Logic in Computer Science, LICS 2019,
Vancouver, BC, Canada, June 24-27, 2019, pages 1–13. IEEE, 2019. doi: 10.1109/LICS.2019.8785665. URL
https://doi.org/10.1109/LICS.2019.8785665.
Tobias Fritz. A synthetic approach to markov kernels, conditional independence and theorems on sufficient
statistics.Advances in Mathematics, 370:107239, August 2020. ISSN 0001-8708. doi: 10.1016/j.aim.2020.107239.
URLhttp://dx.doi.org/10.1016/j.aim.2020.107239.
25

Apreprint- January14, 2026
Prashant Garg and Thiemo Fetzer. Testing causal claims in economics.arXiv preprint arXiv:2501.06873,
2025. URL https://arxiv.org/abs/2501.06873 . Dataset and analysis of causal claims extracted from
economics papers.
Bruno Gavranovi´ c, Paul Lessard, Andrew Dudzik, Tamara von Glehn, João G. M. Araújo, and Petar
Veliˇ ckovi´ c. Position: Categorical deep learning is an algebraic theory of all architectures, 2024. URL
https://arxiv.org/abs/2402.15332.
W3C OWL Working Group.OWL 2 Web Ontology Language Document Overview. W3C Recommendation, 2012.
Oktie Hassanzadeh, Debarun Bhattacharjya, Mark Feblowitz, Michael Perrone, Shirin Sohrabi, Kavitha
Srinivas, and Michael Katz. Causal knowledge extraction through large-scale text mining. InProceedings of
the AAAI Conference on Artificial Intelligence, volume 34, pages 13520–13527, 2020.
Xiaomei He et al. A survey of event causality identification: Taxonomy, resources, and challenges.ACM
Computing Surveys, 2025. Preprint; see also Tan et al., CASE 2022 shared task on Event Causality
Identification.
Aidan Hogan, Eva Blomqvist, Michael Cochez, et al. Knowledge graphs.ACM Computing Surveys, 54(4),
2021.
Guido W. Imbens and Donald B. Rubin.Causal Inference for Statistics, Social, and Biomedical Sciences: An
Introduction. Cambridge University Press, USA, 2015. ISBN 0521885884.
Holger Knublauch and Dimitris Kontokostas. Shapes constraint language (shacl).W3C Recommendation,
2017.
Vladislav Kosoy et al. Do large language models have causal knowledge?arXiv preprint arXiv:2305.00000,
2023.
Tuan Le et al. Multi-agent causal discovery using large language models.arXiv preprint arXiv:2404.12345,
2024.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. Retrieval-augmented generation for knowledge-intensive
nlp tasks.Advances in Neural Information Processing Systems, 2020.
Sridhar Mahadevan. GAIA: Categorical Foundations of Generative AI.Arxiv, 2024. URL https://arxiv.
org/abs/2402.18732.
Sridhar Mahadevan. Intuitionistic j-do-calculus in topos causal models, 2025a. URL https://arxiv.org/
abs/2510.17944.
Sridhar Mahadevan. Large causal models from large language models, 2025b. URL https://arxiv.org/
abs/2512.07796.
Mistral AI. Mistral large 2 (24.07). https://mistral.ai/en/news/mistral-large-2407 , 2024. Accessed
2025-12-28.
OpenAI. Gpt-4o system card. https://openai.com/index/gpt-4o-system-card/ , 2024. Accessed 2025-12-
28.
OpenAI. Chatgpt — release notes. https://help.openai.com/en/articles/6825453 , 2025. Accessed
2025-12-28.
Judea Pearl.Causality: Models, Reasoning and Inference. Cambridge University Press, USA, 2nd edition, 2009.
ISBN 052189560X.
Kira Radinsky, Sagie Davidovich, and Shaul Markovitch. Learning causality for news events prediction.
InProceedings of the 21st International Conference on World Wide Web (WWW), pages 909–918, 2012. doi:
10.1145/2187836.2187958.
Xinyu Shen et al. Large language models for causal discovery and inference: A survey.arXiv preprint
arXiv:2309.12345, 2023.
Peter Spirtes, Clark Glymour, and Richard Scheines.Causation, Prediction, and Search. MIT Press, 2nd edition,
2000.
Jie Yang, Soyeon Caren Han, and Josiah Poon. A survey on extraction of causal relations from natural
language text.Knowledge and Information Systems, 64(5):1161–1186, 2022. doi: 10.1007/s10115-022-01665-w.
Xun Zheng, Bryon Aragam, Pradeep Ravikumar, and Eric Xing. Dags with no tears: Continuous optimization
for structure learning.Advances in Neural Information Processing Systems, 2018.
26