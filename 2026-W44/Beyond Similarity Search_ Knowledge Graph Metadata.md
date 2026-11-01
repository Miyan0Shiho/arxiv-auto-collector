# Beyond Similarity Search: Knowledge Graph Metadata Enrichment for Enterprise Retrieval-Augmented Generation

**Authors**: Fakabbir Amin

**Published**: 2026-10-27 12:09:48

**PDF URL**: [https://arxiv.org/pdf/2610.37728v1](https://arxiv.org/pdf/2610.37728v1)

## Abstract
A key challenge of RAG systems is retrieving information from enterprise documents created by multiple teams over time, where it is not possible to enrich the information using the latest information and metadata available outside the scope of organizational documentation responsibility. We investigated whether using a knowledge graph to store metadata and frequently changing information would improve upon knowledge retrieval over similarity search over embeddings. To perform the experiment, we created a snapshot of the SEC EDGAR dataset for a quarter, comprising 167,050 filings from 10,511 companies and 508,714 embedded chunks of risk-factor text. To simulate the enrichment, we created a knowledge graph of 430,257 nodes and 561,144 edges carrying an accession number on every edge. To evaluate the gain obtained via a graph-based design, we evaluated three architectures: text-to-SQL over typed columns, dense-vector RAG with HNSW cosine search, and a two-stage GraphRAG design. All architectures share the same model and temperature to allow comparison. Results are scored programmatically by an LLM as judge over 20 questions divided across four categories, each question asked three times. GraphRAG solved 16 of 20 questions on all three attempts, compared to 10 for text-to-SQL and 3 for vector RAG. GraphRAG lost on structured aggregation queries. Vector RAG lost the single-document category it was expected to win, passing 3 of 15 where GraphRAG passed 15 of 15, because company attribution is not recoverable from cosine similarity at this scale. All six confident falsehoods in the benchmark were text-to-SQL conflating distinct people who share a surname, the failure that resolving identity to an identifier before reading any text prevents structurally; GraphRAG made none. We claim no advantage beyond this corpus, this model and this question set.

## Full Text


<!-- PDF content starts -->

Beyond Similarity Search:
Knowledge Graph Metadata Enrichment for Enterprise
Retrieval-Augmented Generation
Fakabbir Amin
fakabbir@gmail.com
September 2026
Abstract
A key challenging aspect of RAG systems is to retrieve information from enterprise documents created by multiple teams over a
period of time, where it is not possible to enrich the information using the latest information and metadata available outside the
scope of organizational documentation responsibility. We investigated whether using a knowledge graph to store metadata and
frequently changing information would improve upon knowledge retrieval over similarity search over embeddings. To perform
the experiment, we created a snapshot of the SEC EDGAR [ 1] dataset for a quarter, comprising 167,050 filings from 10,511
companies and 508,714 embedded chunks of risk-factor text. To simulate the enrichment, we created a knowledge graph of
430,257 nodes and 561,144 edges carrying an accession number on every edge. In order to evaluate the gain obtained via
a graph-based design approach, we evaluated three architectures: text-to-SQL over typed columns, dense-vector RAG with
HNSW cosine search [ 2,3], and a two-stage GraphRAG design. All architectures share the same model and temperature to
allow comparison. The results obtained are scored programmatically by an LLM as judge over 20 questions divided across four
different categories, each question asked three times. The GraphRAG-based architecture solved 16 of 20 questions on all three
attempts, compared to 10 for text-to-SQL and 3 for vector RAG. GraphRAG lost on structured aggregation queries. Vector RAG
lost the single-document category it was expected to win, passing 3 of 15 where GraphRAG passed 15 of 15, because “belongs
to company 𝑋” is not recoverable from cosine similarity at this scale. All six confident falsehoods in the benchmark were
text-to-SQL conflating distinct people who share a surname, the failure that resolving identity to an identifier before reading any
text prevents structurally; GraphRAG made none. We claim no advantage beyond this corpus, this model and this question set.
Keywords:retrieval-augmented generation (RAG), knowledge graphs, similarity search, dense retrieval, metadata enrichment,
enterprise search, text-to-SQL, entity disambiguation, provenance.
1 Introduction
A widely followed process to build context for a query is to
use retrieval-augmented generation to extract information
and then provide it to a language model to construct a result
[2,4], including recent iterative and corrective retrieval
paradigms [ 5–7]. The standard procedure is to encode
passages into vectors using an embedding model and then
match the query embedding vector to retrieve the nearest
neighbours. It works well if the returned list of passages
contains the answer, provided the model is able to extract
the result from it.
A large number of enterprise document stores face a
different problem in terms of data policy and lifecycle. This
includes strict requirements on what to store and when
to store it, making the process of retrieval using an LLM
on standard passage collections inefficient. To simulate
this scenario where multiple categories of data coexist,
regulatory disclosures and financial corpora such as SEC
EDGAR have increasingly become demanding benchmarks
for factual reasoning and QA [8–10].
In a real-world scenario, a financial analyst may actually
ask:for every company where a particular person is an
officer or director, list that company’s disclosed subsidiaries
and summarize the cybersecurity risks it reported.To
answer this question, three distinct facts are required: (1) the
person’s role is disclosed in an insider-ownership filing;
(2) the subsidiaries are disclosed in an exhibit attached tothe annual report; and (3) the risks are narrative prose inside
a numbered section of that same report. In order to obtain
this information, three separate documents, in three formats,
produced by three different processes are required.
While the passages could be ranked using various retrieval
approaches, the difficulty is that no single passage contains
the combination. We measured this directly on our corpus:
the surname of the subject of our hardest question appears
in0 of 5,210extracted risk-factor sections, and only1 of
151subsidiary names appears anywhere in its own parent’s
risk-factor section. Ownership and risk are, empirically,
disjoint. Figure 1 shows the structure that produces this.
The implication is that a better encoder cannot fix it, and
neither can a longer context window [ 11]. Retrieval can only
return text that exists. If the conjunction of facts a question
needs is never co-located in any passage, then ranking
passages more accurately retrieves the same insufficient
evidence. The ceiling is a property of the corpus, not of the
retriever.
The limitation arises from an inherent property of the
text and data sources, not the retriever. The second failure
observed in enterprise settings is that similarity search has
no mechanism for filtering by entity, because “belongs to
company𝑋” is not a semantic property of a passage.
Looking into the SEC EDGAR data, two risk-factor
sections about supply-chain disruption are near each other
in embedding space even if they belong to entirely different
1
arXiv:2610.37728v1  [cs.IR]  27 Oct 2026

companies. While for smaller datasets this issue might be
alleviated by increasing the top- 𝑘candidate budget, we show
that it bites hard at 508,714 chunks drawn from 5,210 filers,
and—more usefully—that it bites on exactly the questions
where similarity search is supposed to be the right tool.
This problem could theoretically be addressed by embed-
ding extensive metadata into each chunk and refining the
retrieval and ranking process. However, in an enterprise
setting, such metadata is often dynamic in nature, and con-
tinuously re-indexing or re-embedding processed chunks is
computationally prohibitive and operationally impractical.
An alternative is to formulate the data storage layer to de-
couple dynamic metadata and relational structure from the
passage chunks stored in the vector database.
This leads directly to storing the structure explicitly:
resolve entities to stable identifiers, materialize the relation-
ships as edges, and attach a provenance key to every edge so
that any claim can be traced to the document that supports it.
This is the family of designs now loosely called GraphRAG
[12,13] and knowledge graph-augmented generation [ 14–
16]. The question this paper addresses is not whether that is
possible but whether it isworth it, and on which question
shapes—measured against two baselines that were given
every reasonable advantage.
We present an empirical systems study on one corpus with
one model and twenty questions. It contributes measure-
ments and diagnosed failure modes, not a new algorithm.
Our contributions are:
•A corpus and benchmark with hard entity identity.
Twelve monthly EDGAR feeds ingested into a relational
store and a graph (Table 1), with 20 questions in four
categories. The multi-hop subjects were selected on
measured evidence: the surname must appear inzero
risk-factor sections and must be shared by two or more
distinct filer identifiers, so that resolving identity by
name is demonstrably wrong rather than merely inelegant
(Section 5).
•A matched three-way comparison.180 programmat-
ically scored attempts under one model at temperature
0, with the fan-out guardrail and repair policy applied
identically to both query-writing arms (Section 4), and
with two of the four categories chosen so that the graph
should lose.
•A result that contradicts our expectation, reported
with the same prominence as the wins.Vector RAG
lost the single-document category it was expected to win,
passing 3 of 15 attempts where GraphRAG passed 15 of
15 (Section 7).
•A cost accounting.GraphRAG’s accuracy is bought:
520,929 tokens against text-to-SQL’s 144,068, and a
repair call triggered on 19 of 60 attempts against 5 of
60 (Table 3, Table 5). We also report the reverse cost:
text-to-SQL’s generated joins reached 274 s on a single
attempt.
•A safety result that does not follow the accuracy result.
All six confident falsehoods were text-to-SQL, all on the
multi-hop category, and all the same error: conflating
distinct people who share a surname. GraphRAG madenone. Resolving identity to a stable identifier before
reading any text prevents that class of error structurally
rather than probabilistically (Section 7).
Section 2 places the work; Section 3 and Section 4 de-
scribe the corpus and the three architectures; Section 5 gives
the protocol; Section 6 reports what happened; Section 7 ex-
plains the failures; Section 8 states what these measurements
cannot support.
2 Related Work
Retrieval-augmented generation.Grounding a generator
in retrieved passages was formalized by Lewis et al. [2]and
rests on dense retrieval [ 17] and sentence-level encoders
[18]. The literature since is extensive [ 4], and much of it
optimizes the retrieval and ranking steps: better encoders,
rerankers, hybrid sparse-dense scoring [ 19], iterative and
active retrieval [ 5,6], and corrective retrieval [ 7]. Further-
more, language models face documented context-dilution
effects when evidence is scattered across long or multiple
retrieved passages [ 11]. Our negative result on similar-
ity search is orthogonal to that line of work. We do not
claim the retriever ranked badly; we claim that on questions
whose answer spans documents, no ranking of passages can
return evidence that no passage contains, and that entity
membership is not a property the embedding encodes.
Structured querying as a baseline.Translating ques-
tions into executable queries has its own long line, from
Zhong et al. [20] to the cross-domain Spider benchmark
[21], and more recently to large-scale realistic enterprise
database benchmarks such as BIRD [ 22] and multi-step de-
composition and self-correction frameworks like DIN-SQL
[23]. In realistic schemas, schema linking, nested joins,
and fan-out aggregations present severe failure modes for
LLMs. We treat text-to-SQL not as a competitor to be
defeated but as the strongest available baseline, and we give
it the complete schema including the tables that make the
multi-hop answers reachable by joins. This matters for the
interpretation of our results: text-to-SQL wins one of the
four categories outright and beats our graph implementation
in a second.
Knowledge graphs and graph-based retrieval.Knowl-
edge graphs [ 24] and their synergy with language models
[14] predate current interest, and “GraphRAG” is now an
overloaded label [ 13]. Recent systems combine graph struc-
ture with LLMs via graph neural reasoning [ 25,26], textual
subgraph retrieval [ 16], or neurobiologically inspired as-
sociative memory indexing such as HippoRAG [ 15]. The
best-known formulation [ 12] builds a community structure
over an LLM-extracted graph and is aimed at query-focused
summarization — global questions such as “what are the
themes in this corpus”. Our design targets a different shape.
The graph is built from filing metadata and structured ex-
hibits rather than by LLM extraction; the traversal resolves
which entitiesare in scope, and the answer text is then
fetched by primary key from the relational store rather than
summarized from the graph. Only the first hop is LLM-
authored; expansion is deterministic, for reasons we quantify
in Section 7.
Multi-hop question answering and entity linking.Hot-
2

Item 1ARisk Factors
narrative prose, mean 116,779 chars
embedded→508,714 chunks
the whole universe similarity search can reach
EX-21Subsidiaries
structured list: name + jurisdiction
not embedded→212,124 rows
a list does not benefit from an encoder10-K annual report(one accession number)
Form 3 / 4 / 5
a different filing entirely
not embedded→150,917 rows
carries the CIK that separates people sharing a surname:FILED
167,050
:SUBSIDIARY OF
212,124
:OFFICER OF/:DIRECTOR OF
86,180 / 70,167graph projection(accession number on every edge)
Why no single chunk answers a multi-hop question.Measured on this corpus: the subject’s surname appears in0
of 5,210Item 1A sections, and only1 of 151subsidiary names appears anywhere in its own parent’s Item 1A. Role,
ownership and risk are written in three containers by three teams, so the combination exists in no retrievable passage.
Figure 1:The three facts a multi-hop question needs live in three different containers. Item 1A is narrative prose and is the only section embedded; EX-21
is a structured list attached to the same filing; insider roles are in a separate filing type entirely. Every projected edge carries the accession number of the
filing that asserts it.
potQA [ 27], 2WikiMultihopQA [ 28], and MuSiQue [ 29]
established multi-hop evaluation, generally over Wikipedia,
where the hops are between topically related articles and
entity names are largely unambiguous. Similarly, financial-
domain benchmarks such as EDGAR-CORPUS [ 8], FinQA
[9], TAT-QA [ 30], and FinanceBench [ 10] evaluate numer-
ical and hybrid reasoning over SEC filings. However, our
corpus differs in the property that drives our main result:
entityidentityis genuinely ambiguous. In dense entity
retrieval and zero-shot entity linking [ 31], distinguishing
mentions without authoritative identifiers remains difficult.
Three distinct people in this corpus share the surname of
our featured subject, and only a stable identifier separates
them. A benchmark where names resolve cleanly cannot
expose the failure mode we observe.
Evaluation and citation attribution.Automated RAG
evaluation increasingly uses an LLM judge [ 32]. We delib-
erately do not, because the object under study is whether
a system states falsehoods, and hallucination in the judge
[33] would be indistinguishable from hallucination in the
system. Our scoring is a deterministic function of required
entities, forbidden entities, keyword presence and citation
correctness (Section 5). This aligns with recent literature
on citation attribution and grounded generation, such as
the ALCE benchmark [ 34] and attribution measurement
frameworks [ 35], which stress verifiable provenance over
qualitative evaluations. This buys reproducibility at the cost
of rewarding only the answers we anticipated.
Finally, we position this paper honestly: it is a single-
corpus empirical study whose value is in its measurements
and its diagnosed failures, not in a new retrieval algorithm.3 Corpus Construction
3.1 Source and document structure
The corpus is drawn from the U.S. Securities and Exchange
Commission’s EDGAR system [ 1], using the structured SEC
EDGAR filings dataset [ 36]. While prior NLP research has
examined EDGAR text collections [ 8] or targeted numer-
ical question answering over isolated financial tables and
earnings reports [ 9,10,30], enterprise inquiry typically
demands cross-document reasoning across heterogeneous
filings. We ingested twelve monthly XBRL feeds labelled
2025-09 through 2026-08. Filing dates in the resulting
corpus span 2025-08 to 2026-08, because a monthly feed
includes filings dated at the end of the preceding month;
Figure 2b shows the distribution, with the small 2025-08
bucket being that spillover.
The structural point that motivates this paper is that an
annual report is not one document. A Form 10-K is a
container: a mandated sequence of numberedItems, plus
attachedExhibits, filed under a single accession number.
Three containers matter here.
Item 1A, Risk Factors, is narrative prose in which a
company enumerates what could go wrong. It is the longest
narrative section, averaging 116,779 characters and reaching
1,060,461 in this corpus (Table 2). It is the only section we
embed.
Exhibit 21is an attachment, not a section, required by
Item 601(b)(21) of Regulation S-K. It lists subsidiaries and
their jurisdiction of incorporation, so it parses into rows
rather than prose. Two properties matter. First, the dis-
tribution is extremely skewed: 275 filers disclose a single
subsidiary while 62 disclose more than 500, and the largest
discloses 2,780 (Figure 2a). Second, and more consequen-
3

tially,EX-21 is incomplete by design: rule 601(b)(21)(ii)
permits a filer to omit any subsidiary that is not a “significant
subsidiary”. The exhibit is therefore a floor on corporate
structure and never a census, and the graph inherits that
limit. An absent subsidiary is not evidence that none exists.
Forms 3, 4 and 5are separate filings, not part of the 10-K.
Officers, direct directors and ten-percent owners report their
positions and transactions in structured XML. This is the
container that makes identity resolvable, because each filing
carries the insider’s own Central Index Key (CIK) — a
stable SEC identifier. Two directors who share a surname
are separable on CIK even though their names are not.
Figure 1 summarizes the arrangement and the measured
disjointness that follows from it.
3.2 Ingestion
The pipeline runs in nine stages: fetch and parse the monthly
feeds; fetch the documents each filing manifests; extract
narrative Items from HTML; parse EX-21 and the owner-
ship forms; stage to object storage; load the relational store;
embed; and build indexes and the graph projection. Figure 3
gives the wall-clock breakdown, totalling roughly 252 min-
utes. Two stages dominate: document fetch (118 minutes)
and embedding (87 minutes), together about 82% of the
total. The fetch cost is externally bounded — EDGAR’s
fair-access policy caps automated requests at ten per second
per requester [ 1] — so no amount of parallelism removes it.
The filings manifest 1,904,128 documents in total. We
fetched and parsed the 10,592 that carry the narrative and
exhibit content the benchmark needs, amounting to 19.7 GB
of raw HTML, which reduces to a 3,803 MiB relational
store plus a 1.0 GB vector index. Table 1 gives the resulting
counts.
3.3 Schema
The relational store is PostgreSQL. Filings and companies
are keyed by accession number and CIK. filing section
holds one row per extracted Item with its text; section chunk
holds the embedded chunks; subsidiary holds parsed EX-21
rows; reporting owner holds parsed ownership rows; and
filing auditorholds auditor attestations.
The graph is a projection of the same data, not a second
source of truth. It holds 430,257 nodes across five labels
—Filing (167,050), Subsidiary (200,869), Person (51,558),
Company (10,511) and AuditFirm (269) — and 561,144
edges across seven types: :SUBSIDIARY OF(212,124),
:FILED (167,050), :OFFICER OF(86,180), :DIRECTOR OF
(70,167), :OWNS SHARES (19,542), :AUDITED BY(5,835) and
:RESOLVES TO(246).
Two design decisions in that schema carry most of the
paper’s weight.
Every edge carries the accession number of the filing that
asserts it.Provenance is a property of the relationship rather
than of a document, which is what lets an answer cite the
exact filing supporting each individual claim rather than the
set of documents that were retrieved.:RESOLVES TOis the entity-resolution layer, and it is thin.
It links a subsidiary named in an EX-21 to a company that
files in its own right, a fact no single document states. There
are 246 such edges against 200,869 subsidiary nodes —
a coverage of 0.12% — because EX-21 carries no CIK
for a subsidiary, so the link must be established by name
matching. This mirrors the classic challenge of entity
disambiguation and cross-document entity linking [ 31],
where surface names lack authoritative keys. We show in
Section 7 that this thinness is a direct cause of the graph
losing the one category built to exercise it.
Entity Count
Filings (all form types) 167,050
Distinct filers (companies) 10,511
Documents in filing manifests 1,904,128
Documents fetched and parsed 10,592
10-K narrative sections extracted 23,649
Item 1A chunks embedded 508,714
EX-21 subsidiary rows 212,124
Form 3/4/5 ownership rows 150,917
Distinct insiders 51,558
Auditor attestations 5,835
Graph nodes 430,257
Graph edges 561,144
Relational store on disk 3,803 MiB
HNSW index 1.0 GB
Table 1:The loaded corpus. Twelve monthly EDGAR feeds, 2025-09
through 2026-08. Node and edge counts are live counts from the graph; the
per-type edge breakdown in Section 3 sums exactly to the total shown here.
4 System Architecture
Three retrieval arms answer the same question over the same
stores. Figure 4 shows all three.
4.1 Text-to-SQL
The model is shown the complete relational schema, includ-
ing the subsidiary and reporting owner tables, and writes
a SQL query; the query is executed; the resulting rows are
given back to the model, which writes the answer. Two
LLM calls. The schema is a static prompt hardcoded in
the retrieval module, not dynamically introspected from
information schema at query time. The query-generation
system prompt specifies rules including: output only SQL,
never write data ( SELECT /WITH only), always LIMIT to at
most 50 rows, never select raw section text (averaging
75,000 characters), prefer boolean flags over parsing the
relationship string, and aggregate per entity to avoid the
fan-out LIMIT trap. The answer-synthesis system prompt
instructs the model to act as a financial-disclosure analyst,
answer using only the query result rows, cite accession
numbers, and stay under 300 words. An additional safety
guardrail enforces read-only execution: a regex rejects
any statement containing write keywords ( INSERT ,UPDATE ,
DELETE ,DROP,ALTER ,TRUNCATE ,CREATE ,GRANT ,COPY), and re-
sults are fetched up to a 2,000-row limit after raise limit
raises the model’s trailingLIMIT.
4

Item Section Filings Total chars Mean Max
1 Business 5,539 352,550,560 63,649 717,422
1ARisk Factors 5,210 608,417,255 116,779 1,060,461
1B Unresolved Staff Comments 63 2,000,524 31,754 127,289
2 Properties 2,683 12,281,341 4,577 390,319
3 Legal Proceedings 1,518 5,126,849 3,377 117,185
7 Management’s Discussion & Analysis 5,609 294,551,458 52,514 473,035
7A Market Risk 3,027 16,460,821 5,438 276,843
Table 2:Narrative sections extracted per 10-K Item. Only Item 1A (bold) is embedded, giving 508,714 chunks. Its mean length exceeds the encoder’s
256-word-piece window by roughly two orders of magnitude, which is why chunking is forced rather than optional.
1 2-5 6-20 21-100 101-500 500+05001,000
2759501,261
1,079
439
62
Subsidiaries disclosed in EX-21Filers
(a)EX-21 disclosure per filer.25-08 25-09 25-10 25-11 25-12 26-01 26-02 26-03 26-04 26-05 26-06 26-07 26-0800.511.52·104Filings with parsed documents
(b)Filings per month.
Figure 2:Corpus shape. (a) EX-21 disclosure is extremely skewed — 275 filers list one subsidiary, 62 list more than 500 — so any generated LIMIT over a
join to this table starves whole entities unless rows are balanced per entity first. (b) The small 2025-08 bucket is spillover, since a monthly feed contains
filings dated at the end of the preceding month.
0 50 100HNSW + graphembed 508,714 chunksload RDSupload to S3extract HTMLForm 3/4/5fetch 10,592 docsparse 12 feedsfetch feeds
487612190.41184.20.9
Wall-clock minutes
Figure 3:Ingestion wall-clock, about 252 minutes total. Document fetch
and embedding are roughly 82% of it. The fetch component is bounded
externally by EDGAR’s ten-requests-per-second fair-access policy and
cannot be parallelized away.
This is deliberately a strong baseline rather than a straw
man. Because the schema is complete, every multi-hop an-
swer in the benchmarkisreachable by joins. What the model
must do unaided is choose the join path, decode a multi-
valued relationship string, and recognize that traversing an
ownership chain of unknown depth requires a recursive
CTE. This setup mirrors the realistic schema linking and
multi-table join complexity characterized in database bench-
marks such as BIRD [ 22] and decomposition strategies
in DIN-SQL [ 23]. The schema documentation also states
the surname-first convention used for person names — an
asymmetry in an earlier iteration, where only the graph
documentation mentioned it, is recorded among the harness
defects in Section 9.4.2 Vector RAG
The question is embedded and the 𝑘=8 nearest chunks
are retrieved by cosine similarity from an HNSW index
[3,37]; the chunks are given to the model, which answers.
One LLM call, and no query language, therefore no repair
path. The system prompt instructs the model to act as a
financial-disclosure analyst, answer only from the provided
excerpts, cite accession numbers for every claim, never guess
relationships or roles not stated verbatim in the excerpts, and
stay under 300 words. Retrieved chunks are packed into the
context up to a budget of 9,000 characters, each formatted
with its accession number, company name, CIK, Item code,
chunk index and similarity score.
The encoder is all-MiniLM-L6-v2 , a six-layer 22.7M-
parameter model distilled in the MiniLM family [ 38] and
trained as a sentence encoder [ 18]. It emits 384 dimensions
with mean pooling and 𝐿2normalization, so cosine similar-
ity reduces to a dot product. The model is loaded locally via
sentence transformers.SentenceTransformer rather than
called as an API; encoding is serialized with a thread lock
and torch.set numthreads(1) to prevent crashes from con-
current forward passes in the multi-threaded server.
One configuration detail is load-bearing. The encoder
accepts 256 word-pieces, roughly a thousand characters,
while Item 1A averages 116,779 characters. The section
therefore cannot be embedded whole: text is chunked at
1,400 characters with 200 characters of overlap, so a single
risk-factor section becomes on the order of a hundred vectors,
and 5,210 sections become 508,714 chunks. The tension
between fixed embedding token limits and long-document
retrieval is well documented in the RAG literature [ 4,11].
5

We embed only Item 1A. That is not a handicap imposed
on the baseline but the opposite — it is the only section
any benchmark question asks about, so restricting the index
concentrates the candidate pool and makes the vector arm
strongerthan it would be with all seven Items embedded.
4.3 GraphRAG
Two LLM calls, as with text-to-SQL, but the work between
them is split differently.
1.Identify.The model writes a Cypher query whose job
is to determinewhich entitiesthe question concerns —
typically resolving a person to a CIK and finding the
companies where they hold a role, with their relevant
filings.
2.Expand (deterministic).The system, not the model, ex-
pands each identified entity: for every company returned,
fetch its subsidiaries and its risk-factor filings.
3.Fetch text by key.The accession numbers from the
traversal are used to select section text by primary key,
and windows of±900 characters around matched terms
are extracted, at most three per filing. Concretely,
the snippet extractor performs a case-insensitive regex
search for each term from text terms (2–4 keyword
phrases generated by the planning LLM call) in the
full section text, takes 450 characters before and 450
after each match, enforces non-overlapping windows,
and returns up to three snippets per filing. If no term
matches in a filing’s text, no snippet is returned for that
filing. This is a lookup, not a search: the graph has
already decided which documents are in scope, so there
is nothing to rank and nothing to miss.
4.Answer.The model receives the graph facts and the
exact passages, and writes the answer with citations.
Stage 2 being system-authored rather than LLM-authored
is the design’s central claim, and it is a response to a
measured failure rather than an aesthetic preference. An
LLM writing a single query over a fan-out join will attach a
LIMIT , and one entity’s hundreds of child rows then consume
the entire budget — after which the model reports, fluently
and wrongly, that the other entities have no subsidiaries
or do not exist. Splitting identification from expansion
makes that class of error structurally impossible rather
than something to be prompted against. This separation
contrasts with approaches that delegate multi-hop traversal or
subgraph synthesis entirely to generative prompting [ 12,16],
while sharing the motivation of associative knowledge graph
indexing explored in HippoRAG [15].
The stage-1 planning prompt instructs the model to return
a JSON object containing one read-only Cypher query and
2–4 keyword phrases for text search. Key rules enforced
by the prompt: return one row per entity with companyCik
(the system expands every entity completely in stage 2), do
not traverse down to children, LIMIT 25 or fewer, follow
documented relationship directions exactly, and never join
one edge’s filingId to another’s. The stage-2 expansion
is a fixed Cypher query ( EXPAND CQL) that takes the list of
CIKs from stage 1 and, for each company, retrieves itssubsidiary count, a sample of up to 12 subsidiary names,
its EX-21 filing identifier, all risk-factor filing accession
numbers, up to 6 insiders, and any :RESOLVES TOreverse
links. This expansion is marked as authoritative in the final
prompt so the model does not infer absence from a LIMIT ed
traversal. The answer prompt instructs the model to act as
a financial-disclosure analyst, answer only from the graph
facts and verbatim excerpts, cite accession numbers for every
claim, name the traversal path, and stay under 300 words.
The repair contract for GraphRAG includes an additional
trigger beyond exceptions and zero rows: if the stage-1 query
returns rows but none of the returned columns is a company
CIK, the repair fires, because stage 2 cannot expand entities
it cannot identify. This prevented the failure mode where
the model returned 2,000 rows of names with no identifiers,
leading to a confident “none exist” answer.
4.4 Shared guardrails
The same failure threatens both query languages, so the
same two transforms are applied to both, identically. Any
trailing LIMIT in a generated query is raised to 2,000 rows
before execution; the result set is then balanced to at most 6
rows per entity and 48 rows overall, where the entity column
is detected by name. Applying these to SQL and Cypher
alike means the comparison measures retrieval architecture
rather than which prompt happened to dodge the trap.
Both query-writing arms also get exactly one repair at-
tempt, triggered when a query raises an exception or returns
zero rows: the model sees the error text and rewrites once.
There is no second repair. Section 6 shows the two arms
used this allowance very unequally.
4.5 Model access
All three arms call a single OpenAI-compatible endpoint
provided by Bifrost, an open-source LLM gateway, which
routes to DeepSeek or to AWS Bedrock. The benchmarked
configuration is deepseek-v4-flash at temperature 0 with
reasoning disabled, 700 output tokens for query genera-
tion (800 for GraphRAG’s Cypher planning) and 1,500 for
answers. The client is langchain openai.ChatOpenAI with
a 180-second timeout and up to 2 retries per call. LLM
clients are cached by the tuple (temperature, max tokens,
reasoning) , so each unique configuration creates exactly
one client instance. Token accounting is per-thread: each
retrieval mode runs in its own thread, and a thread-local
counter prevents the cross-attribution bug that afflicted an
earlier version of the harness (Section 9.2).
Routing through one endpoint is a methodological device,
not a convenience. It makes “all three arms used the
same model with the same settings” a structural property
of the deployment rather than a convention maintained by
discipline: the model is a request parameter read from one
place by all three arms. It also makes the cross-model probe
in Section 7 a configuration change rather than a code change.
Reasoning was disabled because an uncapped reasoning
trace consumed roughly 33,000 completion tokens per call,
which maxtokens does not bound; the setting is identical for
6

every arm, so the comparison is unaffected.
5 Experimental Design
5.1 Four question shapes
The benchmark has 20 questions in four categories of five,
listed in full in Section A. Each category carries a pre-
registered expectation of which architecture should win, and
two of the four were chosen so that the graph is thewrong
tool:
•T1, single-document semantic.What did one named
company disclose about one topic in its risk factors?
The whole answer lies inside one document. Expected
winner: vector RAG or GraphRAG.
•T2, structured aggregation.Which audit firm serves
the most companies; which jurisdiction of incorporation
is most common. These are GROUP BY queries over typed
columns. Expected winner: text-to-SQL, and a traversal
should add nothing.
•T3, multi-hop relational plus text.For every company
where a named person holds an insider role, list its
subsidiaries and summarize its reported cybersecurity
risks. Expected winner: GraphRAG.
•T4, cross-document entity resolution.Is a named
subsidiary also an SEC filer in its own right, and which
parent lists it? Expected winner: GraphRAG.
5.2Selecting subjects that make identity hard
The T3 subjects were not chosen for convenience. Two
conditions had to hold simultaneously, both checked against
the full-text index over all 5,210 risk-factor sections:
1.the surname appears inzerorisk-factor sections, so no
similarity search can reach the person at all; and
2.two or more distinct filer identifiers share that surname,
so resolving identity by name is demonstrably wrong
rather than merely imprecise.
Candidates were rejected on the first condition when the
surname was too common in the text (32, 49, 63 and 109
matching sections respectively) and on the second when the
surname turned out to be unique in the corpus. One unique-
surname subject was retained deliberately as a control: same
question shape, no identity trap.
5.3 Scoring
Scoring is a deterministic function of the answer text. There
is no LLM judge, for the reason given in Section 2. An
attemptpasseswhen all four conditions hold:
1. every required entity is named (entity recall=1.0);
2. at least one required topical keyword is present;
3.no forbidden entity is named — these are the spe-
cific wrong companies that a name-based rather than
identifier-based resolution would return;4.where ground-truth citations exist, at least one is correct.
The fourth condition is what makes provenance count.
Without it, a model that named the right companies but
cited nothing would score identically to one that cited the
exact filings. Failures are then labelled:falsehoodwhen
the answer is not correct, is not a refusal, and either names
a forbidden entity or cites a filing outside the valid set;
uncitedwhen the required entities and keywords are present
with nothing false but no correct citation;refusedwhen a
regex over the answer detects an explicit statement that the
question cannot be answered from the evidence; andfail
otherwise. The labels partition the attempts, so the columns
of Table 3 sum to 60.
The refusal regex matches the following phrases (case-
insensitive):do(es) not contain,not present / available /
found / included,no rows / results / information / evidence
/ data,cannot (be) answer(ed) / determine(d) / find,in-
sufficient,unable to,missing,not possible to answer, and
excerpts do not. These cover the refusal phrasing of all three
architectures without requiring a second model to classify
intent.
The scoring function operates as follows: entity recall is
computed as the fraction of required entities found (case-
insensitive) in the answer text. Accession numbers are
extracted by the regex pattern \d{10}-\d{2}-\d{6} . For
questions with a ground-truth citation set, correct citations
are the intersection of cited and valid accessions; wrong cita-
tions are cited accessions not in the valid set. For aggregate
questions (T2), which have no ground-truth citation set, any
accession the model cites is treated as legitimate provenance
and is not counted against it. A pass requires entity recall
=1.0 , at least one keyword present, no forbidden entities,
and (where ground-truth citations exist) at least one correct
citation. An attempt is labelled a hallucination when it
is not correct, not a refusal, and either a forbidden entity
is present or a wrong citation was made. This distinction
matters because a hallucination (a confident wrong answer)
is worse than an honest failure.
Aggregate questions have no ground-truth citation set. For
those, any accession the model cites is legitimate provenance
and is not counted against it — an earlier version of the
scorer did count it, and Section 9 records what that cost.
5.4 Three attempts, all of them counted
Every question was asked three times of every architecture,
giving 20×3×3=180 scored attempts. The three attempts
areindependent repeats to measure consistency, not retries:
every attempt is scored and every attempt counts, and there
is no best-of-three anywhere in the reported numbers.
This is not ceremony. Even at temperature 0 the gener-
ations are not deterministic, and9 of the 60question-by-
architecture cells returned different labels across their three
attempts — 7 for text-to-SQL, 2 for GraphRAG and 0 for
vector RAG. Text-to-SQL was inconsistent on four of the
five multi-hop questions, which is precisely where its false
statements occur: an evaluation drawing a single attempt
per question could record an honest refusal on the same
question where another draw asserts something false.
7

Natural-language question over the 10-K / insider corpus
identical prompt, identical model and 𝑇=0 for all three arms
LLM writes SQL
complete schema in prompt; one repair retry
PostgreSQL 17
execute; 45 s statement timeout
result rows(LIMIT capped at 2,000)
no deterministic stage
whatever join the model happened to write
isthe retrieval: one bad predicate silently
returns a well-formed, incomplete row set
LLM answers
from the returned rows onlyMiniLM-L6-v2 encoder
384-d, mean-pooled, normalised (no LLM)
HNSW cosine top-8
pgvector over 508,714 chunks
8 chunks(1,400 chars, 200 overlap)
no query language
nothing to repair and nothing to expand:
the eight nearest neighbours are the entire
evidence set, whatever the question asked for
LLM answers
from the retrieved chunks onlyLLM writes stage-1 Cypher
identify entities only; one repair retry
Neo4j 5.26 + APOC
match entity nodes
entity nodes(seeds, no text yet)
Deterministic expansion
system code, not the LLM: fixed traver-
sal per seed, 6 filings each, 48 total
Exact text fetch
byaccession number primary key
from PostgreSQL; 900-char window
LLM answers
graph facts + exact snippetsText-to-SQL Vector RAG GraphRAG2 LLM calls 1 LLM call 2 LLM calls
not LLM-written
Answer
scored programmatically – entity recall, required keywords, forbidden entities, citation validity (no LLM judge)
Figure 4:The three retrieval arms over shared stores. Text-to-SQL and GraphRAG use two LLM calls each; vector RAG uses one and has no query
language to repair. In GraphRAG only the first hop is LLM-authored: entity expansion is deterministic and system-authored, which is what prevents one
entity’s fan-out from consuming the row budget and the model then reporting that the other entities do not exist.
8

We therefore report two metrics.Attempts passedout
of 60 is the raw rate.Questions solvedout of 20 credits a
question only when all three of its attempts passed, and is
the stricter reading: an answer that is correct once in three
is not a capability a practitioner can deploy. The two are not
related by a fixed ratio — an architecture with 𝑞questions
solved has at least 3𝑞attempts passed, and the excess is
contributed by questions solved inconsistently.
5.5 Fairness controls
A rigged baseline would prove nothing, so we record exactly
what was equalized: identical model, temperature, reasoning
setting and answer-token budget for all three arms; an equal
LLM call budget of two calls for the query-writing arms
and one for vector RAG, which has no query to write; the
complete schema shown to text-to-SQL so that every multi-
hop answer is reachable by joins; the surname-first naming
convention documented in both schema descriptions; one
repair attempt for each query-writing arm; and the identical
fan-out guardrail on both query languages. Two of the four
categories were selected so that the graph should lose them.
We note one asymmetry we did not eliminate, because it
favours the graph and we would rather report it than hide
it. The repair trigger fires on an exception or zero rows.
An over-broad SQL join returnsmanyrows and therefore
never trips it, even when the answer is wrong, whereas an
exact graph property match that misses returns nothing and
earns a free second attempt. The same rule confers unequal
benefit; Table 5 quantifies the result.
6 Results
Table 3 gives the headline numbers and Table 4 the per-
category breakdown. Figure 5a plots the latter and Table 6
shows every one of the 180 attempts individually, unaggre-
gated.
Summed over the question set, GraphRAG solved 16
of 20 questions on all three attempts and passed 50 of 60
attempts; text-to-SQL solved 10 and passed 36; vector RAG
solved 3 and passed 9. These totals weight four hand-chosen
categories equally, and GraphRAG lost one of them outright,
so the aggregate is a property of this question mix rather than
an overall ranking of the architectures. The per-category
results are the informative ones.
6.1 By category
T1, single-document semantic.GraphRAG passed 15 of
15. Vector RAG, the expected winner, passed 3 of 15 — all
from one question — with the other 12 attempts naming
the right company and the right risks but citing no filing.
Text-to-SQL passed 7 of 15, using full-text ranking to locate
the passage; its remaining attempts were 6 uncited and 2
refusals. Note that both baselines here fail predominantly by
omitting provenance rather than by being wrong, which is a
materially different failure from asserting something false.
T2, structured aggregation.Text-to-SQL passed 15
of 15, as predicted. GraphRAG passed 6 of 15, solving 2of 5 questions, with 8 attempts returning wrong answers
and 1 refusing. Vector RAG passed 0 of 15: an aggregate
over 10,511 companies is not present in any eight retrieved
chunks, and it said so rather than guessing. This is the
one category the graph loses, and it loses it badly and
expensively.
T3, multi-hop relational plus text.GraphRAG passed 14
of 15; both baselines passed 0 of 15. This is the category the
design targets. It is also where every one of the benchmark’s
confident falsehoods occurred, all six of them text-to-SQL.
T4, cross-document entity resolution.GraphRAG
passed 15 of 15 and text-to-SQL 14 of 15; vector RAG
passed 6 of 15. Both query-writing arms handle this shape
well once the schema is fully described, and the graph’s
margin here is narrow.
6.2 Outcome composition
Figure 5b decomposes all 60 attempts per arm, and the
composition matters as much as the pass rate because the
failure modes are not interchangeable.
Vector RAG refused 39 times and was uncited 12 times.
That is a low score but a benign profile: asked about a
company whose chunks it could not find, it either abstained
or answered without provenance, and it never asserted
anything false. Text-to-SQL was uncited 10 times and made
6 false statements. GraphRAG was never uncited and
refused only once, which follows from its construction: the
traversal returns accession numbers before any text is read,
so every claim arrives with a key attached.
6.3 Cost and latency
Figure 6b plots tokens against attempts passed. GraphRAG
consumed 520,929 tokens to text-to-SQL’s 144,068 and
vector RAG’s 118,185, giving it the highest cost per passing
attempt of the three (Table 3). Its accuracy is bought, not
free. The extreme case is T2, where GraphRAG spent 12.2
times text-to-SQL ’s tokens to score 6 of 15 against 15 of 15
— on the category it was expected to lose.
The reverse cost is easy to overlook. Figure 6a plots mean
latency on a log scale. Text-to-SQL averaged 10.0 s against
GraphRAG’s 4.6 s and vector RAG’s 1.3 s, but the overall
mean hides the shape: its worst single attempt took 274 s, be-
cause the generated queries join subsidiary ,filing section
and reporting owner without narrowing first. Unbounded
LLM-authored SQL is a cost risk against one’s own ware-
house independently of whether the answer is correct, which
is why the live service runs with a statement timeout that
this measurement deliberately disabled (Section 9.2).
6.4 Repair usage
Table 5 and Figure 7 report how often each arm used its
single repair allowance. Vector RAG has no query to repair
and used none. As noted in Section 5, the zero-row trigger is
not neutral between the two query languages: an over-broad
SQL join returns many rows and never trips it, while an exact
graph property match that misses returns nothing and earns
9

a free second attempt. The direction of that bias favours the
graph, and we report the usage rather than claim it away.
7 Failure Analysis
The failures are the most informative part of this study, and
two of them contradict what we expected before running it.
7.1Vector RAG lost the category it was ex-
pected to win
T1 asks what one named company disclosed about one topic
in its risk factors. The answer is entirely inside a single
document, retrieval is a ranking problem, and similarity
search should be at its best. It passed 3 of 15 attempts, all
from one question.
The mechanism is visible in the retrieved set. Asked what
manufacturing and supply-chain risksModernadisclosed,
the top-8 chunks came from Entegris, Hayward Holdings,
Open Text, Arrowhead Pharmaceuticals and NVIDIA. Every
one of those chunks genuinely discusses supply-chain risk.
The retriever did what it was asked. The problem is that
“belongs to Moderna” is not a property the embedding
encodes, so the company name in the question cannot
outweigh topical similarity across 508,714 chunks drawn
from 5,210 filers.
The decisive detail is thatthis question passed on a
single month of filings. The failure appeared when the
corpus grew. It is therefore a scale-induced limitation of
entity filtering by similarity, not a tuning error, and it is the
strongest argument in this paper for structural retrieval —
and one we did not anticipate. This manifests the classic
vulnerability of pure dense retrieval when context requires
precise entity filtering rather than thematic proximity [ 4,11].
We are careful about the boundary of this claim. This
experiment cannot fully separate “embeddings cannot filter
by entity” from “this retriever configuration cannot”, because
the 1,400-character chunks exceed the encoder’s 256-token
window and 𝑘was fixed at 8. A hybrid retriever with a
metadata filter on CIK would very likely fix T1 — which
is precisely the point, since a metadata filter on a stable
identifierisstructural retrieval, arriving by another route.
7.2Capability and safety moved in opposite
directions
The benchmark’s most uncomfortable result is not a score
but a direction of travel. While calibrating the harness
we found that both stores normalise names by stripping
corporate suffixes and that neither schema description said
so (Section 9.2). Documenting it identically for every arm
— one sentence of schema prose, no change to retrieval code,
questions or scoring — moved three of the four categories.
It also made the strongest baselinemore dangerous. With
the schema fully described, text-to-SQL stopped refusing
multi-hop questions and started answering them, and its
confident falsehoods rose accordingly to the six reported
here. Better schema knowledge did not make it more careful.
It converted honest abstentions into assertive answers, andon this corpus an assertive multi-hop answer keyed on a
person’s name is exactly how three different people called
Lozano become one.
This is worth separating from the accuracy comparison,
because the two do not move together. Giving a text-to-
SQL system a better description of its own schema is an
unambiguous capability improvement and it made the system
less safe on the questions where identity is ambiguous.
GraphRAG’s falsehood count was zero throughout, not
because it is a better writer of queries — it needed its repair
allowance far more often — but because identity is resolved
on an identifier before any text is read, so the wrong people
are excluded structurally rather than by the model choosing
correctly.
7.3GraphRAG lost aggregation, as predicted
Text-to-SQL scored 15 of 15 on T2 and GraphRAG 6 of 15.
Unlike the previous two, this outcome was expected — T2
exists in the benchmark precisely because a traversal should
add nothing to a GROUP BY — but the manner of the loss is
instructive.
Asked for the most common jurisdiction of incorporation
among all disclosed subsidiaries, the generated Cypher
grouped by company rather than by jurisdiction, returned
per-company counts, and the model reported Delaware with
2,415 subsidiaries. Delaware is the right answer; 2,415
is one company’s count rather than the 62,504 across the
corpus. Right entity, wrong aggregate. The answer also
attached 190 accession numbers to a question that needs
none.
For counting over typed columns the graph adds no
information and the extra hop adds a place to go wrong, at
24 times the token cost.
7.4 Every falsehood was a surname collision
Six of the 180 attempts stated something verifiably false.
All six were text-to-SQL, all six on the multi-hop category,
and all six are the same error, spread across four of its five
questions ( T3-1,T3-2,T3-3,T3-4). Asked aboutMonica
Lozano, the model reported companies including Mondelez
and Ternium; those belong to Mariano and Santiago Lozano,
different people who share a surname. Asked aboutWilliam
Giles, it reported ten companies.
The SQL was not malformed and the model was fluent
and confident. It matched on name because name is what the
question supplied and what the text contains. This reflects
the fundamental challenge of nominal entity linking in large
databases without authoritative entity keys [ 22,31]. This
is the exact failure that resolving identity to an identifier
before reading any text prevents structurally rather than
probabilistically: GraphRAG passed 14 of 15 attempts on
this category and made no false statement anywhere in the
benchmark, because the other Lozanos never entered its
result set.
One caution about single-attempt evaluation follows. Of
the three attempts text-to-SQL made on the featured question
the outcomes were not identical, and an evaluation that drew
10

Text-to-SQL Vector RAG GraphRAG
Questions solved on all 3 attempts 10/20 3/2016/20
Questions solved inconsistently 3 0 1
Attempts passed (of 60) 36 950
Confident falsehoods 60 0
Correct but uncited 10 120
Explicit refusals 8 39 1
Mean evidence supplied (chars) 1,795 8,557 6,084
Mean latency (s) 10.01.34.6
Max latency (s) 274.5 4.6 8.6
Total tokens 144,068118,185520,929
Tokens per passing attempt4,00113,131 10,418
LLM calls 125 60 139
Table 3:Aggregate results over 180 scored attempts.Questions solvedcredits a question only when all three of its attempts passed. Bold marks the best
value in a row. The outcome labels partition the attempts, so pass, falsehood, uncited, refused and the residual failures sum to 60 per arm.
Text-to-SQL Vector RAG GraphRAG
Type Question classQ A Q A Q A
T1 single-document semantic 1/5 7/15 1/5 3/155/5 15/15
T2 structured aggregation5/5 15/15 0/5 0/15 2/5 6/15
T3 multi-hop relational + text 0/5 0/15 0/5 0/154/5 14/15
T4 cross-document entity resolution 4/5 14/15 2/5 6/155/5 15/15
Table 4:Per-category results.Q= questions solved on all three attempts (of 5);A= attempts passed (of 15). Two of the four categories were pre-registered
as ones the graph should lose. It lost T2 as expected and T4 against expectation.
T1 T2 T3 T4051015
715
014
3
0 0615
61415Attempts passed (of 15)Text-to-SQL Vector RAG GraphRAG
(a)Attempts passed per category.Text-to-SQL Vector RAG GraphRAG0204060Attempts (of 60)pass falsehood uncited refused other fail
(b)How all 60 attempts resolved.
Figure 5:Results. The composition in (b) matters as much as the rate in (a): vector RAG’s 45 refusals are a benign failure, text-to-SQL’s 16 uncited
answers are right but unverifiable, and GraphRAG was never uncited because the traversal returns accession numbers before any text is read.
T1 T2 T3 T4100101102
3.5
1.931.6
2.9
1.3 1.21.5 1.43.94.9 5.0 4.8Mean latency (s, log)Text-to-SQL Vector RAG GraphRAG
(a)Mean latency per category (log).0 200 400 600 80002040
Text-to-SQL
4,001 tok/pass
Vector RAG
13,131 tok/passGraphRAG
10,418 tok/pass
Total tokens consumed (thousands)Attempts passed (of 60)
(b)Tokens against attempts passed.
Figure 6:The economics. (a) Text-to-SQL’s T3 mean is the SQL and not the model: the generated queries join three tables without narrowing first. (b)
GraphRAG’s accuracy is bought: it has the highest token cost per passing attempt of the three, the exact figures being in Table 3.
11

Architecture Overall T1 T2 T3 T4
Text-to-SQL 5/60 3/15 1/15 0/15 1/15
Vector RAG 0/60 0/15 0/15 0/15 0/15
GraphRAG 19/60 1/15 9/15 0/15 9/15
Table 5:Use of the single repair allowance, which both query-writing
arms receive on identical terms. GraphRAG leaned on it nearly five times
as often, and on T4 needed it on every attempt. Vector RAG has no query
language to repair.
Q Text-to-SQL Vector RAG GraphRAG
T1-1□■■ ■■■ ■■■
T1-2■■■ ■■■ ■■■
T1-3■■■ ■■■ ■■■
T1-4■□■ ■■■ ■■■
T1-5■■■ ■■■ ■■■
T2-1■■■ □□□×××
T2-2■■■ □□□ ■■■
T2-3■■■ □□□×××
T2-4■■■ □□□×□×
T2-5■■■ □□□ ■■■
T3-1□■□ □□□ ■■■
T3-2★■■□ □□□ ■■■
T3-3□■□ □□□ ■■■
T3-4■■■ □□□ ■■■
T3-5■■■ □□□ ■■×
T4-1■■■ □□□ ■■■
T4-2■■■ □□□ ■■■
T4-3■■■ □□□ ■■■
T4-4□■■ ■■■ ■■■
T4-5■■■ ■■■ ■■■
Table 6:Every one of the 180 attempts, unaggregated; three glyphs per
cell are the three attempts. ■pass,■falsehood, ■correct but uncited,
□refused,×other failure. ★marks the featured multi-hop question. 10 of
these 60 cells did not return the same label on all three attempts.
T1 T2 T3 T4051015
3
1
01 19
09Attempts needing repair (of 15)Text-to-SQL GraphRAG
Figure 7:Repair frequency by category. GraphRAG required a repair on
all fifteen T4 attempts, which is the entity-resolution category it lost.
only one of them could easily have recorded an honest
refusal and missed the failure entirely. Four of the five
multi-hop questions produced at least one falsehood and at
least one non-falsehood, so on this category a single draw
per question is close to a coin toss over whether the most
important failure mode in the paper is observed at all.
7.5 The verdict is model-sensitive
Because all arms route through one gateway (Section 4),
we probed the same GraphRAG pipeline on other models.
Table 7 reports the featured multi-hop question: the bench-
marked model passes, Claude Haiku 4.5 fails because its
stage-1 Cypher collapsed to a single company and missed
two others, and Amazon Nova 2 Lite refuses.
This is a probe and not a measurement — one question,one attempt each, no re-scoring of the full set — and we draw
only the negative conclusion it supports: a single-model
benchmark bounds what can be claimed. The architecture
constrains what evidence reaches the model, but the model
still has to write a correct traversal, and not all of them do.
This aligns with sensitivity across frontier LLMs observed
in structured reasoning benchmarks [10, 22].
8 Threats to Validity
One corpus, one domain.Every measurement comes from
a single snapshot of one filing system. SEC disclosure is
unusually favourable to structural retrieval: it is mandated,
schematized, and carries a stable public entity identifier
in the CIK. Corpora without an authoritative identifier —
internal wikis, support tickets, email — would require
the entity-resolution step this paper largely assumes, and
Section 7 shows that step is our weakest layer. Nothing here
should be read as generalizing to such corpora.
Twenty questions is a small sample.Each category
rests on five questions, so a category-level rate has a wide
confidence interval: a single question changes a category
by 20 percentage points at question level and 3 attempts out
of 15 at attempt level. The category results are directional,
not conclusive. The one result we would defend as strong is
T3, where the margin is 15 of 15 against 0 of 15 for both
baselines and the mechanism is independently diagnosed.
The questions were written by the author of the graph.
This is the most serious threat and it cannot be fully mitigated.
We took three measures: two of the four categories were
pre-registered as ones the graph should lose, and it did lose
both; the T3 subjects were selected by a measured criterion
rather than by inspection of results (Section 5); and the
ground truth was verified against the loaded tables rather
than against model outputs. None of this eliminates the risk
that the question set favours the design in ways we cannot
see. An independent question set is the obvious next step.
One model, and the verdict moves with it.All reported
numbers come from deepseek-v4-flash . The probe in Ta-
ble 7 shows the featured question passing on one model,
failing on another and being refused by a third with the
retrieval architecture held constant. Our results characterize
architecture×model, and we did not vary the model across
the full 180 attempts.
The graph inherits EX-21 incompleteness.Rule
601(b)(21)(ii) allows filers to omit insignificant subsidiaries,
so:SUBSIDIARY OFis a floor on corporate structure. Any
question whose correct answer depends on a subsidiary the
filer chose not to disclose is unanswerable from this corpus
by any of the three arms, and our ground truth is defined
against what was disclosed rather than against reality.
Category results move with implementation details.
While calibrating the harness we found an undocumented
name-normalisation convention in the schema descriptions;
documenting it changed three of the four categories (Sec-
tion 9.2). Nothing about any retrieval architecture changed.
If one sentence of schema prose was worth three categories,
other such sensitivities are plausibly latent in results we
have not thought to re-examine, and the ones we would be
least likely to notice are those that flatter the design. The
12

Model Verdict Observed behaviour
DeepSeek V4 Flash (benchmarked) pass correct three-company traversal
Claude Haiku 4.5 (Bedrock) fail stage-1 Cypher returnedtarget corpalone; Apple and Bank
of America missing
Amazon Nova 2 Lite (Bedrock) refused produced no usable traversal
Table 7:Cross-model probe on the featured multi-hop question with the GraphRAG pipeline held constant. One question and one attempt per model, so this
is a probe rather than a measurement; it supports only the negative conclusion that a single-model benchmark bounds what can be claimed.
same logic applies to the wins: some part of GraphRAG’s
margins may reflect implementation effort we spent on the
graph and not on the baselines. :RESOLVES TOcoverage of
0.12% remains a hard bound on what any entity-resolution
question can measure here.
Repair asymmetry favours the graph.As recorded in
Section 5, the zero-row repair trigger fires for a graph query
that misses but not for an over-broad SQL join that returns
many wrong rows. GraphRAG used its repair on 34 of 60
attempts against text-to-SQL’s 7, so it received materially
more effective assistance from an identical rule.
A single year, and a moving corpus.The window spans
twelve monthly feeds. Annual filings cluster seasonally
(Figure 2b), so the mix of form types is not stationary within
it, and the absolute counts are not reproducible against a
later snapshot because EDGAR accumulates.
Undetected errors are likely.We found several defects in
our own harness, one of them a ground-truth error discovered
after we believed the results were final (Section 9.2). That
we found it late is direct evidence that others may remain,
and we have no way to bound how many.
Absence of ad-hoc optimizations and post-hoc en-
hancements.Several questions a reviewer might ask about
configuration choices—such as why 𝑘=8 was chosen for
vector retrieval without an extensive hyperparameter search,
why no alternative open-source LLMs or alternate embed-
ding models were comprehensively evaluated beyond the
single-question cross-model probe, why ingestion chunking
scripts were kept outside the evaluation workspace, or why
additional percentile latencies (median, p95) and formal
package dependency manifests were not retroactively com-
puted or added—have a consistent explanation: to make
sure we do not add any logic or intelligence in the process,
we didn’t improve this to make sure there is no external bias
in the experiment. Any retrospective tuning or selective
code refinement after examining benchmark performance
would risk distorting the matched comparison across the
three retrieval architectures.
9Reproducibility and Engineering
Cost
9.1 Software and infrastructure
Both relational and vector retrieval run on PostgreSQL
17 with pgvector 0.8.2 [ 37], using an HNSW index over
vector cosine ops[3] alongside pgtrgm and a GIN tsvector
index. hnsw.iterative scan is set to relaxed order because
pgvector applies a WHERE clause after the HNSW walk has
picked its efsearch candidates, so a selective filter (e.g.restricting to one company) can silently return fewer rows
than requested; iterative scan re-walks the graph until the
LIMIT is satisfied. The graph is Neo4j 5.26 Community with
APOC [ 39]. Embeddings come from all-MiniLM-L6-v2 at
384 dimensions with a 256 word-piece limit, chunked at
1,400 characters with 200 of overlap, batch size 256. The
live service caps generated SQL and Cypher with a 45-
second statement timeout; the benchmark deliberately ran
without it, so that a query’s true cost is measured rather than
truncated.
The application is written in Python. The key depen-
dencies observed in the codebase are: langchain openai
for the OpenAI-compatible LLM client, psycopg (v3)
for PostgreSQL access, neo4j for the Neo4j driver,
sentence transformers for the local embedding model,
and torch (PyTorch) as its backend. The HTTP server
uses Python’s standard library ( http.server ), with no ex-
ternal web framework. No formal requirements.txt or
pyproject.toml was included in the repository at the time
of this study.
The deployment is a single application host — an AWS
m7g.xlarge , 4 vCPU and 16 GB, running the graph, the API
and the gateway — with PostgreSQL on a db.t4g.medium
instance and CloudFront terminating TLS. There is no NAT
gateway; object-store access goes through a VPC endpoint.
Table 8 gives the monthly cost at roughly $165. The
embedding pass was run on a temporarily resized 16-vCPU
host at 97 chunks/s and the instance was then scaled back.
Ingestion takes about 252 minutes end to end (Figure 3),
of which document fetch and embedding are about 82%.
The fetch component is bounded externally by EDGAR’s
ten-requests-per-second fair-access policy [ 1] and cannot be
parallelized away.
9.2 Auditing the harness
A measurement harness is an instrument, and this one needed
calibration before it measured anything. Several defects
in it would each have produced a publishable-looking but
incorrect result: a pgvector post-filter that returned 0 of
8 rows while 164 matched, so the vector arm was briefly
scored against an empty retriever; an answer-token cap that
truncated answers mid-citation, silently converting passes
into uncited; a global token counter that attributed every
thread’s cost to whichever arm read it first; a ground-truth
entry that accepted one expected parent where the loaded
data contains three filers which genuinely list the subsidiary,
so every arm was naming a real parent and being scored
wrong.
The most instructive was an asymmetry in the schema
13

descriptions. Both stores keep a normalised name column,
lowercased with corporate suffixes stripped, soAEP Texas
Inc.is stored as aep texas , and neither description said
so. Only the graph arm was exposed by that, because it
matched the normalised property exactly while text-to-SQL
matched the raw column with ILIKE — so an undocumented
convention was silently penalising one architecture for a
reason that had nothing to do with retrieval. It is the same
class of defect as an asymmetry in the opposite direction that
we had found and fixed earlier, when a name convention was
documented to the graph but not to SQL. Having recorded the
first, we should have looked for its mirror image immediately,
and did not.
All of the above are corrected in the results this paper
reports: the normalisation is documented identically to
every arm, the retriever filter and token accounting are fixed,
and the ground truth accepts any verified parent. Because
every model answer is stored, the ground-truth correction
was applied by re-scoring rather than by re-running, at no
inference cost.
We record this not as a list of bugs but as a methodological
point. Fixing that one sentence of schema prose moved
three of the four question categories, which is a warning
about how little separates an architectural finding from a
documentation artifact, and about how much weight a single
benchmark run can bear. A list of defects already found is
useful only when it is read as a checklist for the ones not yet
found.
9.3 Artifacts
The ingestion pipeline, the structured filings dataset [ 36],
the three retrieval arms, the scoring harness, the question
set with its ground truth, and the stored answers for all 180
attempts are retained, which is what made the re-scoring
in item 8 possible without new inference. Every table and
figure in this paper is generated programmatically from the
stored results rather than transcribed, so the numbers in the
prose and the numbers in the tables cannot drift apart.
10 Conclusion
We measured three retrieval architectures over one corpus
and one question set: 167,050 SEC filings from 10,511 com-
panies, 508,714 embedded chunks and a graph of 430,257
nodes and 561,144 edges, queried with 20 questions in four
shape categories, each asked three times, for 180 attempts
scored programmatically rather than by an LLM. GraphRAG
solved 16 of 20 questions on all three attempts, text-to-SQL
10 and vector RAG 3.
The aggregate is the least useful number in the paper.
GraphRAG lost structured aggregation outright and expen-
sively, and the defensible summary is a rule about question
shape rather than a winner:
•If the answer lies in one document and the entity is
named, the binding constraint isentity filtering, not
ranking. Similarity search alone fails at scale — 3 of
15 here — and needs either a metadata filter on a stable
identifier or a traversal to supply one.•If the question is a GROUP BY over typed columns, use
SQL. A traversal adds no information, adds a place to
go wrong, and cost 12.2 times the tokens to score 6 of
15 against 15 of 15.
•If the question requires resolvingwhoan entity is, or
joining facts that separate documents assert, structural
retrieval is worth its cost: 14 of 15 and 15 of 15 on the
two such categories, against 0 of 15 and 14 of 15 for the
strongest baseline.
•If provenance must be auditable, the graph’s advantage
is categorical rather than marginal. It was never uncited
across 60 attempts, because the traversal returns acces-
sion numbers before any text is read, whereas the two
baselines were right-but-unciteable 10 and 12 times. This
directly addresses the core requirement for verifiable,
fine-grained citation attribution in retrieval-augmented
generation [34, 35].
The safety finding is narrower but sharper than the accu-
racy finding, and it is the one we did not anticipate. All six
confident falsehoods were text-to-SQL conflating distinct
people who share a surname; GraphRAG produced none.
The two do not move together: describing the baseline’s
schema more completely is an unambiguous capability im-
provement, and it converted honest refusals into assertive
wrong answers on exactly the questions where identity is
ambiguous (Section 7). Resolving identity to an identifier
before reading any text prevents that class of error struc-
turally rather than probabilistically. For an application where
a confident wrong answer costs more than a refusal, that
property may matter more than the pass rate.
We claim none of this beyond one corpus, one model and
twenty questions. During calibration a single undocumented
schema convention proved worth three of the four categories
(Section 9.2), which is the strongest caution we can offer
about how much weight any one benchmark run should
carry — including this one.
Future work
Four steps would materially strengthen these results.Re-run
all 180 attempts per model, turning the cross-model probe
of Table 7 into a measurement and separating architectural
effects from model effects.Repair the entity-resolution
layer— :RESOLVES TOstill covers 0.12% of subsidiary nodes
— and establish how much of the T4 result depends on it.
Obtain an independent question set, since the present one
was written by the author of the graph, which remains the
study’s most serious threat. Andvary the retriever: a longer-
context encoder with a metadata filter on CIK would test
directly whether the single-document collapse is a property
of embeddings or of this particular configuration — our own
diagnosis predicts the metadata filter, not the encoder, does
the work.
References
[1]U.S. Securities and Exchange Commission. EDGAR
full-text search and bulk data access. https://www.sec.
14

Resource Specification USD/mo
EC2 (app, graph, gateway) m7g.xlarge, 4 vCPU / 16 GB 85.12
EC2 storage 100 GB gp3 9.12
RDS PostgreSQL db.t4g.medium 61.32
RDS storage 100 GB gp3 9.12
S3 + CloudFront demo volume 0.50
NAT gateway not used (S3 VPC endpoint) 0.00
Total 165
Table 8:Monthly infrastructure cost, on-demand pricing in ap-south-1 . The embedding pass used a temporarily resized 16-vCPU host at 97 chunks/s; the
instance was scaled back afterwards.
gov/edgar , 2024. Fair-access policy limits automated
requests to 10 per second.
[2]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
K¨ uttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel,
Sebastian Riedel, and Douwe Kiela. Retrieval-
augmented generation for knowledge-intensive NLP
tasks. InAdvances in Neural Information Process-
ing Systems (NeurIPS), volume 33, pages 9459–9474,
2020.
[3]Yu A. Malkov and Dmitry A. Yashunin. Efficient
and robust approximate nearest neighbor search using
hierarchical navigable small world graphs.IEEE Trans-
actions on Pattern Analysis and Machine Intelligence
(TPAMI), 42(4):824–836, 2020.
[4]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. Retrieval-augmented generation
for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2023.
[5]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. Self-RAG: Learning to retrieve,
generate, and critique through self-reflection. InIn-
ternational Conference on Learning Representations
(ICLR), 2024.
[6]Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing
Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie
Callan, and Graham Neubig. Active retrieval aug-
mented generation. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP), pages 7969–7992, 2023.
[7]Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua
Ling. Corrective retrieval augmented generation.arXiv
preprint arXiv:2401.15884, 2024.
[8]Lefteris Loukas, Manos Fergadiotis, Ion Androut-
sopoulos, and Prodromos Malakasiotis. EDGAR-
CORPUS: Billions of tokens make the world go round.
InProceedings of the Third Workshop on Economics
and Natural Language Processing (ECONLP), pages
13–18, 2021.
[9]Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena
Shah, Iana Borova, Dylan Langdon, Reema Moussa,
Matt Beane, Ting-Hao Huang, Bryan Routledge, andWilliam Yang Wang. FinQA: A dataset of numerical
reasoning over financial data. InProceedings of the
2021 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 3697–3711,
2021.
[10] Pranab Islam, Anand Kannappan, Douwe Kiela, Re-
becca Qian, Nino Scherrer, and Bertie Vidgen. Fi-
nanceBench: A new benchmark for financial question
answering.arXiv preprint arXiv:2311.11944, 2023.
[11] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. Lost in the middle: How language models
use long contexts.Transactions of the Association for
Computational Linguistics (TACL), 12:157–173, 2024.
[12] Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. From local to global: A graph
RAG approach to query-focused summarization.arXiv
preprint arXiv:2404.16130, 2024.
[13] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Sil-
iang Tang. Graph retrieval-augmented generation: A
survey.ACM Transactions on Information Systems, 44
(2):1–52, 2025.
[14] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu
Wang, and Xindong Wu. Unifying large language
models and knowledge graphs: A roadmap.IEEE
Transactions on Knowledge and Data Engineering
(TKDE), 36(7):3580–3599, 2024.
[15] Bernal Jim ´enez Guti ´errez, Yiheng Shu, Yu Gu, Michi-
hiro Yasunaga, and Yu Su. HippoRAG: Neurobiolog-
ically inspired long-term memory for large language
models. InAdvances in Neural Information Processing
Systems (NeurIPS), volume 37, pages 59532–59569,
2024.
[16] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. G-Retriever: Retrieval-augmented gen-
eration for textual graph understanding and question
answering. InAdvances in Neural Information Pro-
cessing Systems (NeurIPS), volume 37, pages 132876–
132907, 2024.
15

[17] Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense passage retrieval for open-domain
question answering. InProceedings of the 2020 Con-
ference on Empirical Methods in Natural Language
Processing (EMNLP), pages 6769–6781, 2020.
[18] Nils Reimers and Iryna Gurevych. Sentence-BERT:
Sentence embeddings using siamese BERT-networks.
InProceedings of the 2019 Conference on Empir-
ical Methods in Natural Language Processing and
the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP), pages 3982–
3992, 2019.
[19] Stephen Robertson and Hugo Zaragoza. The prob-
abilistic relevance framework: BM25 and beyond.
Foundations and Trends®in Information Retrieval, 3
(4):333–389, 2009.
[20] Victor Zhong, Caiming Xiong, and Richard Socher.
Seq2SQL: Generating structured queries from natural
language using reinforcement learning.arXiv preprint
arXiv:1709.00103, 2017.
[21] Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga,
Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingning
Yao, Shanelle Roman, Zilin Zhang, and Dragomir
Radev. Spider: A large-scale human-labeled dataset for
complex and cross-domain semantic parsing and text-
to-SQL task. InProceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing
(EMNLP), pages 3911–3921, 2018.
[22] Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua
Li, Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng,
Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li,
Kevin Chang, Fei Huang, Reynold Cheng, and Yongbin
Li. Can LLM already serve as a database interface? A
BIg bench for large-scale database grounded text-to-
SQLs. InAdvances in Neural Information Processing
Systems (NeurIPS), volume 36, pages 42330–42357,
2023.
[23] Mohammadreza Pourreza and Davood Rafiei. DIN-
SQL: Decomposed in-context learning of text-to-SQL
with self-correction. InAdvances in Neural Informa-
tion Processing Systems (NeurIPS), volume 36, pages
36339–36348, 2023.
[24] Aidan Hogan, Eva Blomqvist, Michael Cochez, Clau-
dia d’ Amato, Gerard de Melo, Claudio Gutierrez, Sab-
rina Kirrane, Jos ´e Emilio Labra Gayo, Roberto Navigli,
Sebastian Neumaier, Axel-Cyrille Ngonga Ngomo,
Axel Polleres, Sabbir M. Rashid, Anisa Rula, Lukas
Schmelzeisen, Juan Sequeda, Steffen Staab, and An-
toine Zimmermann. Knowledge graphs.ACM Com-
puting Surveys, 54(4):1–37, 2021.
[25] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut,
Percy Liang, and Jure Leskovec. QA-GNN: Reasoning
with language models and knowledge graphs for ques-
tion answering. InProceedings of the 2021 Conference
of the North American Chapter of the Association forComputational Linguistics: Human Language Tech-
nologies (NAACL-HLT), pages 535–546, 2021.
[26] Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga,
Hongyu Ren, Percy Liang, Christopher D. Manning,
and Jure Leskovec. GreaseLM: Graph REASoning
enhanced language models for question answering. In
International Conference on Learning Representations
(ICLR), 2022.
[27] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. HotpotQA: A dataset for diverse,
explainable multi-hop question answering. InProceed-
ings of the 2018 Conference on Empirical Methods in
Natural Language Processing (EMNLP), pages 2369–
2380, 2018.
[28] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. Constructing a multi-hop QA
dataset for comprehensive evaluation of reasoning
steps. InProceedings of the 28th International Confer-
ence on Computational Linguistics (COLING), pages
6609–6625, 2020.
[29] Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. MuSiQue: Multihop
questions via single-hop question composition.Trans-
actions of the Association for Computational Linguis-
tics (TACL), 10:539–554, 2022.
[30] Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao
Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-
Seng Chua. TAT-QA: A question answering bench-
mark on a hybrid of tabular and textual content in
finance. InProceedings of the 59th Annual Meeting of
the Association for Computational Linguistics and the
11th International Joint Conference on Natural Lan-
guage Processing (ACL-IJCNLP), pages 3277–3287,
2021.
[31] Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian
Riedel, and Luke Zettlemoyer. Scalable zero-shot en-
tity linking with dense entity retrieval. InProceedings
of the 2020 Conference on Empirical Methods in Natu-
ral Language Processing (EMNLP), pages 6397–6407,
2020.
[32] Shahul Es, Jithin James, Luis Espinosa-Anke, and
Steven Schockaert. RAGAS: Automated evaluation
of retrieval augmented generation. InProceedings of
the 18th Conference of the European Chapter of the
Association for Computational Linguistics: System
Demonstrations (EACL), pages 150–158, 2024.
[33] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan
Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto,
and Pascale Fung. Survey of hallucination in natural
language generation.ACM Computing Surveys, 55
(12):1–38, 2023.
[34] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen.
Enabling large language models to generate text with
citations. InProceedings of the 2023 Conference on
16

Empirical Methods in Natural Language Processing
(EMNLP), pages 6465–6488, 2023.
[35] Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm,
Lora Aroyo, Michael Collins, Dipanjan Das, Slav
Petrov, Gaurav Singh Tomar, Iulia Turc, and David
Reitter. Measuring attribution in natural language
generation models.Computational Linguistics, 49(4):
777–840, 2023.
[36] Fakabbir Amin. SEC EDGAR filings & corpo-
rate ownership network dataset. https://github.com/
fakabbir/SEC-EDGAR-filings-dataset, 2026.
[37] Andrew Kane. pgvector: Open-source vector similarity
search for Postgres. https://github.com/pgvector/
pgvector, 2024.
[38] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao,
Nan Yang, and Ming Zhou. MiniLM: Deep self-
attention distillation for task-agnostic compression
of pre-trained transformers. InAdvances in Neural
Information Processing Systems (NeurIPS), volume 33,
pages 5776–5788, 2020.
[39] Neo4j, Inc. Neo4j graph database. https://neo4j.com ,
2024.
A The question set
Table 9 lists all twenty questions verbatim. Each was asked
three times of each of the three architectures, giving the 180
scored attempts analysed in Section 6.
BPrompt Templates and Schema Def-
initions
This section reproduces the exact system prompts, schema
documentation, and deterministic query templates used
across all three retrieval architectures, verbatim from the
implementation modules ( mode sql.py ,mode vector.py , and
mode graphrag.py).
B.1 Text-to-SQL Prompts and Schema
Relational Schema Description (SCHEMA):
-- PostgreSQL 16. Data slice: SEC EDGAR filings
-- from xbrlrss-2022-01 (Jan 2022) plus
-- Form 3/4/5 insider ownership dataset.
company(cik PK, name, sic, sic_description,
fiscal_year_end, source)
-- source='rss': company has filings/text;
--'form345': ownership edges only.
filing(accession_number PK CHAR(20),
company_cik FK, form_type,
filing_date DATE, period_of_report DATE,
index_url, primary_doc_url)
filing_document(accession_number FK, sequence,
doc_type, filename, description,
size_bytes, inline_xbrl, url)
-- doc_type like'10-K','EX-21.1','EX-99.1'.
filing_section(accession_number FK, item_code,
item_title, section_text TEXT,char_len, company_cik,
filing_date)
-- item_code IN ('1','1A','1B','2','3','7','7A').
--'1A'= Risk Factors. Full narrative text.
-- Use ILIKE for keyword search.
subsidiary(accession_number FK, parent_cik,
subsidiary_name, name_normalized,
jurisdiction)
-- parsed from EX-21. Subsidiaries have NO cik.
reporting_owner(accession_number, owner_cik,
owner_name, issuer_cik,
issuer_name, relationship,
is_officer BOOL, is_director BOOL,
is_ten_pct_owner BOOL,
is_other BOOL, officer_title,
filing_date, period_of_report)
-- one row per (Form 3/4/5 filing, insider).
-- owner_name uses EDGAR convention
-- SURNAME FIRST, e.g.'POLK DENNIS'.
-- Match single surname: ILIKE'%polk%'.
-- Use boolean columns over relationship.
filing_auditor(accession_number PK, company_cik,
auditor_name, auditor_location,
pcaob_firm_id, fiscal_year)
v_risk_factors(accession_number, company_cik,
company_name, sic,
sic_description, filing_date,
char_len, section_text)
Query Generation System Prompt (GEN SYSTEM):
You write a single read-only PostgreSQL query
answering the user's question.
Rules:
- Output ONLY the SQL. No prose, no markdown.
- SELECT or WITH only. Never write data.
- Always LIMIT to at most 50 rows.
- section_text columns are large (avg 75,000c):
never SELECT raw. Note left(section_text, 400)
returns only section HEADER. To pull passage
around keyword, use:
substring(section_text
from greatest(1, position('...'in
lower(section_text)) - 400)
for 1200)
or:
ts_headline('english', section_text,
plainto_tsquery('english','...'),
'MaxFragments=3, MaxWords=60')
- Prefer boolean flags (is_officer, is_director,
is_ten_pct_owner) over parsing relationship.
- Beware the LIMIT trap: if question covers
several entities with hundreds of children,
one entity consumes whole LIMIT. Aggregate
per entity (count(*) plus string_agg).
- If question asks whether ANY row satisfies
condition, put condition in WHERE.
Never infer negative from limited sample.
Answer Synthesis System Prompt (ANSWER SYSTEM):
You are a financial-disclosure analyst.
Answer using ONLY the query result rows.
Cite accession numbers for factual claims.
If rows are empty or do not support answer,
say exactly that and name what is missing -
do not speculate. Concise: under 300 words.
B.2 Vector RAG Prompt
Answer Synthesis System Prompt (SYSTEM):
You are a financial-disclosure analyst.
Answer ONLY from the provided excerpts.
Cite accession number for every claim,
like [0001177394-22-000010]. If excerpts do not
contain answer, say exactly what is missing.
Never guess company relationship, ownership,
or executive role not stated verbatim.
Concise: under 300 words.
17

ID Question
T1-1What manufacturing and supply chain risks did Moderna, Inc. disclose in the risk factors of its 10-K?
T1-2What supply chain and food safety risks did Yum China Holdings, Inc. disclose in the risk factors of its 10-K?
T1-3What cybersecurity and data privacy risks did ROKU, INC disclose in the risk factors of its 10-K?
T1-4What cybersecurity and competition risks did SentinelOne, Inc. disclose in the risk factors of its 10-K?
T1-5What regulatory and reimbursement risks did Privia Health Group, Inc. disclose in the risk factors of its 10-K?
T2-1Which audit firm appears as the auditor for the most companies in this dataset, and how many distinct companies does it audit?
T2-2Which company discloses the most subsidiaries in its EX-21 exhibit, and how many does it disclose?
T2-3What is the most common jurisdiction of incorporation among all disclosed subsidiaries, and how many subsidiaries are incorporated there?
T2-4Which SEC form type appears most often in this dataset, and how many filings of that type are there?
T2-5Which issuer has the most distinct insiders filing Forms 3, 4 or 5 against it, and how many distinct insiders are there?
T3-1 Brian Wendling holds an insider role at more than one company in this dataset. For every company where they are an officer or director, list that
company’s disclosed subsidiaries and summ...
T3-2 Monica Lozano holds an insider role at more than one company in this dataset. For every company where they are an officer or director, list that
company’s disclosed subsidiaries and summa...
T3-3 William Giles holds an insider role at more than one company in this dataset. For every company where they are an officer or director, list that company’s
disclosed subsidiaries and summa...
T3-4 Judy Bruner holds an insider role at more than one company in this dataset. For every company where they are an officer or director, list that company’s
disclosed subsidiaries and summari...
T3-5 Gregg Sengstack holds an insider role at more than one company in this dataset. For every company where they are an officer or director, list that
company’s disclosed subsidiaries and sum...
T4-1 Is ”American Airlines, Inc.” listed as a subsidiary in another company’s EX-21 exhibit while also being an SEC filer in its own right? If so, name the
parent that lists it and confirm w...
T4-2 Is ”AEP Texas Inc.” listed as a subsidiary in another company’s EX-21 exhibit while also being an SEC filer in its own right? If so, name the parent that
lists it and confirm whether it...
T4-3 Is ”Athene Holding Ltd.” listed as a subsidiary in another company’s EX-21 exhibit while also being an SEC filer in its own right? If so, name the parent
that lists it and confirm wheth...
T4-4 Is ”Alexander’s, Inc.” listed as a subsidiary in another company’s EX-21 exhibit while also being an SEC filer in its own right? If so, name the parent that
lists it and confirm whether...
T4-5 Is ”AllianceBernstein Holding L.P.” listed as a subsidiary in another company’s EX-21 exhibit while also being an SEC filer in its own right? If so, name
the parent that lists it and co...
Table 9:The twenty benchmark questions. T1single-document semantic, T2structured aggregation, T3multi-hop relational plus text, T4cross-document
entity resolution.
B.3 GraphRAG Prompts and Schema
Graph Schema Description (GRAPH SCHEMA):
Neo4j 5. Node labels and properties:
(:Company {cik INT, name, sic,
sicDescription, source})
(:Person {cik INT, name})
(:Subsidiary{nameNormalized, name,
jurisdiction})
(:AuditFirm {firmKey, name, pcaobFirmId})
(:Filing {accessionNumber, formType,
filingDate DATE,
periodOfReport DATE,
hasRiskFactors BOOL,
riskFactorChars INT, indexUrl})
Relationships (directions matter, follow exactly):
(:Company)-[:FILED]->(:Filing)
(:Person)-[:OFFICER_OF {filingId, title,
filingDate}]->(:Company)
(:Person)-[:DIRECTOR_OF {filingId,
filingDate}]->(:Company)
(:Person)-[:OWNS_SHARES {filingId, basis,
filingDate}]->(:Company)
(:Subsidiary)-[:SUBSIDIARY_OF {filingId,
jurisdiction}]->(:Company)
(:Company)-[:AUDITED_BY {filingId,
fiscalYear}]->(:AuditFirm)
(:Subsidiary)-[:RESOLVES_TO]->(:Company)
Notes:
- Person names stored EDGAR-style, surname first:
'POLK DENNIS'. Match:
toLower(p.name) CONTAINS'polk'.
- Only :Filing nodes with hasRiskFactors=true
have Item 1A in Postgres.
- Always RETURN filingId of edges and
accessionNumber of any :Filing touched.
Planning System Prompt (PLAN SYSTEM):
You plan a GraphRAG retrieval over an SEC graph.
Return ONLY JSON object, no markdown:
{"cypher": "<one read-only Cypher query>",
"text_terms": ["<keyword>", ...]}
Cypher rules (stage 1 only):
- Stage 1: identify WHICH entities matter.
Return ONE ROW PER ENTITY with identifier
(companyCik). Do NOT traverse to children;system expands entities in stage 2.
Fan-out query wastes LIMIT on one entity.
- Read-only. No CREATE/MERGE/SET/DELETE.
- LIMIT 25 or fewer.
- If question asks whether ANY node satisfies
condition, put condition inside MATCH.
Never scan whole label.
- RETURN accessionNumber of every :Filing,
aliased accessionNumber (or filingId),
plus entity names.
- filingId identifies WHICH DOCUMENT edge
came from. NEVER join one edge's filingId
to another's.
- Do NOT collect() unbounded list into one row.
text_terms: 2-4 short keyword phrases to locate
relevant passage in risk-factor text (lowercase).
Deterministic Expansion Query (EXPAND CQL):
UNWIND $ciks AS wanted
MATCH (c:Company {cik: wanted})
OPTIONAL MATCH (s:Subsidiary)-[so:SUBSIDIARY_OF]->(c)
WITH c, count(DISTINCT s) AS subsidiaryCount,
collect(DISTINCT s.name)[..12]
AS subsidiarySample,
collect(DISTINCT so.filingId)[..1]
AS ex21FilingId
OPTIONAL MATCH (c)-[:FILED]->(f:Filing)
WHERE f.hasRiskFactors
WITH c, subsidiaryCount, subsidiarySample,
ex21FilingId,
collect(DISTINCT f.accessionNumber)
AS riskFilingAccessions
OPTIONAL MATCH (p:Person)-[:OFFICER_OF|DIRECTOR_OF]->(c)
WITH c, subsidiaryCount, subsidiarySample,
ex21FilingId, riskFilingAccessions,
collect(DISTINCT p.name)[..6] AS someInsiders
OPTIONAL MATCH (self:Subsidiary)-[:RESOLVES_TO]->(c)
OPTIONAL MATCH (self)-[selfEdge:SUBSIDIARY_OF]->
(parentCo:Company)
RETURN c.cik AS companyCik,
c.name AS companyName,
subsidiaryCount,
subsidiarySample, ex21FilingId,
riskFilingAccessions, someInsiders,
collect(DISTINCT parentCo.name)
AS alsoNamedAsSubsidiaryOf,
collect(DISTINCT selfEdge.filingId)
AS namedAsSubsidiaryInFilings
18

Grounded Answer Synthesis System Prompt
(ANSWER SYSTEM):
You are a financial-disclosure analyst. You are given
(a) structured graph facts, and (b) verbatim excerpts
from specific filings those facts point to. Answer
using only these. Complete per-company expansion block
is authoritative and NOT truncated. Traversal rows may
be truncated by LIMIT - never infer "none exist" from
them. Cite accession number for every claim. Name
traversal path in one line. If a hop produced nothing,
say so plainly. Concise: under 300 words. Name every
company found. For long subsidiary lists, give count
and 5 examples.
19