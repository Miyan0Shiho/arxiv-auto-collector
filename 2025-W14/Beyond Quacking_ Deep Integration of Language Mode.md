# Beyond Quacking: Deep Integration of Language Models and RAG into DuckDB

**Authors**: Anas Dorbani, Sunny Yasser, Jimmy Lin, Amine Mhedhbi

**Published**: 2025-04-01 19:48:17

**PDF URL**: [http://arxiv.org/pdf/2504.01157v1](http://arxiv.org/pdf/2504.01157v1)

## Abstract
Knowledge-intensive analytical applications retrieve context from both
structured tabular data and unstructured, text-free documents for effective
decision-making. Large language models (LLMs) have made it significantly easier
to prototype such retrieval and reasoning data pipelines. However, implementing
these pipelines efficiently still demands significant effort and has several
challenges. This often involves orchestrating heterogeneous data systems,
managing data movement, and handling low-level implementation details, e.g.,
LLM context management.
  To address these challenges, we introduce FlockMTL: an extension for DBMSs
that deeply integrates LLM capabilities and retrieval-augmented generation
(RAG). FlockMTL includes model-driven scalar and aggregate functions, enabling
chained predictions through tuple-level mappings and reductions. Drawing
inspiration from the relational model, FlockMTL incorporates: (i) cost-based
optimizations, which seamlessly apply techniques such as batching and caching;
and (ii) resource independence, enabled through novel SQL DDL abstractions:
PROMPT and MODEL, introduced as first-class schema objects alongside TABLE.
FlockMTL streamlines the development of knowledge-intensive analytical
applications, and its optimizations ease the implementation burden.

## Full Text


<!-- PDF content starts -->

Beyond Quacking: Deep Integration of Language Models and
RAG into DuckDB
Anas Dorbani
Polytechnique Montréal
anas.dorbani@polymtl.caSunny Yasser
Polytechnique Montréal
sunny.yasser@polymtl.ca
Jimmy Lin
University of Waterloo
jimmylin@uwaterloo.caAmine Mhedhbi
Polytechnique Montréal
amine.mhedhbi@polymtl.ca
ABSTRACT
Knowledge-intensive analytical applications retrieve context from
both structured tabular data and unstructured, text-free documents
for effective decision-making. Large language models (LLMs) have
made it significantly easier to prototype such retrieval and rea-
soning data pipelines. However, implementing these pipelines effi-
ciently still demands significant effort and has several challenges.
This often involves orchestrating heterogeneous data systems, man-
aging data movement, and handling low-level implementation de-
tails, e.g., LLM context management.
To address these challenges, we introduce FlockMTL : an ex-
tension for DBMSs that deeply integrates LLM capabilities and
retrieval-augmented generation (RAG). FlockMTL includes model-
driven scalar and aggregate functions, enabling chained predictions
through tuple-level mappings and reductions. Drawing inspiration
from the relational model, FlockMTL incorporates: (i) cost-based
optimizations, which seamlessly apply techniques such as batching
and caching; and (ii) resource independence, enabled through novel
SQL DDL abstractions: PROMPT andMODEL , introduced as first-class
schema objects alongside TABLE .FlockMTL streamlines the de-
velopment of knowledge-intensive analytical application and its
optimizations ease the implementation burden.
1 INTRODUCTION
Complexity of Workflows. A variety of workflows are in the
form of knowledge-intensive analytical applications, i.e., they rely
on integrating relevant context from structured and unstructured
datasets to support data-driven decision-making. They further rely
on analytics, semantic analysis, or a combination of both to take
effective action. For example, consider an investigative analyst
reporting on inquiries regarding a new vessel offence. The ana-
lyst might: i) consult tabular data to obtain specific vessel details;
ii) interpret legal documents to assess the severity of the reported of-
fence; iii) aggregate the vessel’s history with similar prior offences;
and iv) identify and rank potential interventions.
Novel Data Pipelines. The advent of LLMs has led to a techno-
logical step change. Their commoditization since the release of
GPT-3 [ 2] made it possible to implement pipelines that interleave:
(i) querying of tables; (ii) retrieval of relevant tuples; and (iii) gen-
eration and reasoning using LLM-predictions. These pipelines use
disparate systems, e.g., DBMSs and search engines, follow the RAG
approach [5] and use possibly tool calling [4].Implementation Challenges. The development of such pipelines
is reminiscent of the early data management era prior to the rela-
tional model [ 3]. Data engineers currently make many low-level
execution decisions, e.g., choosing specific models for predictions,
adapting prompts, managing LLM context, caching of predictions
for reuse, and incorporating novel optimizations as they are re-
leased publically, and deciding when to use them. To make mat-
ters worse, any changes to the application requirements in terms
of expected quality, latency, dollar cost, or scale require a major
re-implementation. Beyond these execution decisions, the use of
disparate systems results in significant data shuffling and missed
co-optimization opportunities. This often pushes users to rely on
DBMSs for initial simple querying and on re-implementing more
complex operations within an orchestration layer.
Our Approach. We propose FlockMTL , an open-source extension
forDuckDB [8] that enables the use of LLMs with scalar and aggre-
gate functions. Using SQL’s common table expressions (CTEs) with
these functions leads to powerful pipelines that can interleave ana-
lytics with LLM-chained predictions. Following in the tradition of
declarative relational model, FlockMTL uses cost-based optimiza-
tion to alleviate the burden of implementing low-level execution
details from developers and non-expert users.
Core Insights. Our insights on important features designing and
implementing FlockMTL can be summarized as follows:
•Flexible paradigm :FlockMTL supports a broad range of semantic
operations, including classification, summarization, and rerank-
ing, through the use of scalar and aggregate functions, all of
which are summarized in Table 1). Additionally, FlockMTL in-
troduces some specialized built-in functions, e.g., reranking (first
/ last) and fusion (rrf, combanz, combmed, combmnz, combsum)
that enable full hybrid search.
•Resource-independence : Functions accept both model and prompt
specifications as inputs. FlockMTL introduces two new DDL
resource types: MODEL s and PROMPT s, treated as first-class schema
objects akin to TABLE s. This abstraction allows SQL queries to
remain fixed while enabling model and prompt updates adminis-
tratively, without requiring changes to application logic.
•Seamless optimizations : Lower-level implementation like LLM
context management, batching on input tuples in a single LLM
prediction, caching and reusing of results, and predictions on
unique, i.e., deduplicated input values are handled seamlessly by
FlockMTL . This reduces the complexity of integrating semantic
operations and allows developers and data engineers alike to
focus on higher-level application logic.arXiv:2504.01157v1  [cs.DB]  1 Apr 2025

Anas Dorbani, Sunny Yasser, Jimmy Lin, and Amine Mhedhbi
Scalar Functions: Map an input tuple to an output value using chat completion API or embedding API.
Genericllm_complete{_json?} uses an LLM and a user prompt to generate textor structured JSON output from an input tuple.
llm_embedding uses an LLM to generate an embedding vector (fixed-length array) from an input text value.
fusion fuses𝑁scores from 𝑁retrievers — choices: rrf/combanz /combmed /combmnz /combsum .
Specialized llm_filter uses an LLM and a prompt to return True/False given an input tuple.
Aggregate Functions: Reduce multiple input tuples to a single output value using the chat completion API.
Generic llm_reduce{_json?} uses an LLM and a prompt to generate textor structured JSON output from multiple input tuples.
Specializedllm_rerank uses an LLM and a prompt to rank input tuples based on relevance.
llm_first/last uses an LLM and a prompt to return the most or least relevant tuple from multiple input tuples.
Table 1: Summary of the scalar and aggregate functions supported by FlockMTL .
2 SYSTEM OVERVIEW
LLMs and prompt engineering align well with SQL’s original goal
of making data querying accessible to non-experts. FlockMTL ’s
core feature is introducing semantic operations and hybrid search
within SQL. Given that the majority of enterprise data is stored in
RDBMSs, FlockMTL relies on DuckDB [8]’s extension module.
DuckDB implements a state-of-the-art analytics engine and
makes the DBMS internals easily extensible. Its extension mod-
ule allows changes to SQL parsing, to the optimizer and execution
engine, as well as the addition of new data types. DuckDB has
already a rich ecosystem of extensions that can be complemen-
tary. For instance, its extensions enable the querying of file formats
such as Parquet andCSV as well as attaching to DBMSs such as
PostgreSQL andMySQL . As such, users can write federated queries
over multiple data formats and databases. As a DBMS of choice, the
capabilities we add within FlockMTL become instantly available
across a variety of file formats and databases. FlockMTL also pro-
vides an ASKfunctionality to turn a natural language queries into
SQL augmented with FlockMTL ’s functions. It is easy to INSTALL
andLOAD FlockMTL as a community extension using:
INSTALL flockmtl FROM community; LOAD flockmtl;
In the remainder of this section, we provide an overview of
FlockMTL ’s new schema object resources ( MODEL and PROMPT ),
functions, and optimizations. To illustrate resources and functions,
we consider a simple use case involving the following table:
research_papers: (id, title, abstract, content )
In this scenario, the user is a researcher aiming to identify relevant
papers, extract key insights, and generate summaries using SQL.
2.1 Models and Prompts
Users can define reusable resources in the form of MODEL s and
PROMPT s. These resources can be scoped to the current database
using the Local setting, which is the default, or configured as
Global ,i.e., accessible across all databases on a given machine.
Query 1 below shows how to define a global model named model-
relevance-check , configured to point to GPT-4o-mini with the server
provider OpenAI. It also includes a local user-defined prompt for
identifying papers relevant to join algorithms.
-- 1Define a model to use
CREATE GLOBAL MODEL( 'model-relevance-check ','gpt-4o-mini ','openai ')
-- 2Define a prompt to check if the paper is a join algorithm
CREATE PROMPT( 'joins-prompt ','is related to join algos given abstract ')
Query 1: Prompt and Model DefinitionsUsers have the flexibility to modify or delete these resources.
When a resource is modified, FlockMTL automatically creates a
new version. Previous versions remain available for inspection
and use, while the most recent version is applied by default unless
specified otherwise.
2.2 Functions
FlockMTL enables semantic analysis through a combination of
generic and specialized functions. The generic functions allow users
to define operations that map or reduce tuples to text or JSON
outputs using LLMs available through Azure and OpenAI cloud ser-
vices or locally through Ollama. These are summarized in Table 1.
Building on these generic functions, FlockMTL also provides spe-
cialized functions that wrap the generic ones for common use cases.
For example, llm_filter converts LLM predictions into boolean
values, and llm_first /llm_last use the output of llm_rerank to
select the most or least relevant tuple based on a ranking.
Scalar Functions. We showcase in the next query an example of se-
mantic filtering, summarization, and extraction using FlockMTL ’s
scalar functions. A scalar function maps each input tuple to a new
attribute. Query 2 demonstrates this by identifying papers relevant
to join algorithms. Then, for each relevant paper, it extracts key-
words, determines whether the paper is empirical or theoretical,
and summarizes its abstract in a sentence. The query uses three
semantic operations: llm_filter , which produces a boolean value;
andllm_complete andllm_complete_json , which return a string
and a JSON object, respectively, via chat completion APIs.
WITH
-- 1Select papers related to join algorithms
relevant_xpapers AS (
SELECT id, title, abstract, content
FROM research_papers P
WHERE llm_filter({ 'model_name ':'model-relevance-check '},
{'prompt_name ':'joins-prompt '},
{'title ': P.title, 'abstract ': P.abstract})
),
-- 2Summarize the paper 's abstract
summarized_Papers AS (
SELECT RP.id, RP.title, llm_complete({ 'model ':'gpt-4o '},
{'prompt ':'Summarize the abstract in 1 sentence '},
{'abstract ': RP.abstract}) AS summarized_abstract,
llm_complete_json({ 'model ':'gpt-4o '},
{'prompt ':'Based on the provided paper title, abstract, and content
extract the following as JSON: {
"keywords": <three relevant keywords>,
"type": <specify if empirical or theoretical> } '},
{'title ': p.title, 'abstract ': p.abstract})
FROM relevant_papers RP
)
SELECT * FROM summarized_Papers
Query 2: Finding relevant join algo papers.

Beyond Quacking: Deep Integration of LMs and RAG into DuckDB
Figure 1: Prompt construction example for llm_complete .
More specifically, llm_filter filters out non-relevant papers by
applying a scalar function to each input tuple using the predefined
model-relevance-check and an associated prompt. The result-
ing subset is then passed to llm_complete , which summarizes
each abstract using GPT-4o , and to llm_complete_json , which
extracts keywords and the paper type in structured JSON form.
While the use of CTEs in Query 2 is not required, it illustrates how
FlockMTL supports the chaining of LLM predictions through com-
posable functions. Notably, only llm_filter relies on a predefined
model and prompt resources; the remaining functions specify these
parameters directly within the query.
Aggregate Functions and Full Hybrid Search. We showcase in
the next query an example of a full hybrid search pipeline within
FlockMTL . To our knowledge, this is the first such implementation
within SQL. Query 3 aims to find passages from research papers
relevant to join algorithms in databases . Among those, it further
reranks the results to prioritize passages related specifically to cyclic
join queries . We consider a table containing previously extracted
passages from publications: research_passages: (idx, content ).
The query breaks down the hybrid search into distinct steps.
First, it computes the embedding for the user intent join algorithms
in databases using FlockMTL ’sllm_embedding . It then performs a
vector similarity scan, selecting the top 100most relevant passages
from research. While this example could alternatively use the Vec-
tor Search Extension1, the intention is to highlight the flexibility
and direct use of llm_embedding . It also uses the Full-Text Search
extension2to retrieve the top 100passages based on BM25 scores.
The two result sets are then fused using a FULL OUTER JOIN and
are normalized in this case using the max score for each retriever.
Following fusion, the top-10 candidate passages are reranked us-
ing an LLM-based list-wise aggregation. This is achieved through
llm_rerank , a specialized function built on top of the generic aggre-
gate interface. It applies a learned reranking model to the candidate
list, inspired by the approach of Xueguang Ma et al. , [7], to assess
their relevance to cyclic join queries.
1https://duckdb.org/docs/extensions/vss.html
2https://duckdb.org/docs/extensions/full_text_search.html-- 1Compute the embedding for the input query
WITH Query AS (
SELECT llm_embedding({ 'model ':'text-embedding-3-small '}, { 'query ':'join
algorithms in databases '})::DOUBLE[1536] AS embedding),
-- 2Scan vectors for similar search based on array_distance
VS AS (
SELECT idx, content, array_cosine_similarity(Query.embedding,
llm_embedding({ 'model ':'text-embedding-3-small '},
{'content ': content})::DOUBLE[1536]) AS score
FROM research_passages
ORDER BY vs_score DESC
LIMIT 100)
-- 3BM25 retriever over chunked text contents of papers
BM25 AS (
SELECT idx, content, fts_main_research_chunks.match_bm25(index_column,
'join algorithms in databases ', fields:= 'content ') AS score
FROM research_passages
ORDER BY bm25_score DESC
LIMIT 100),
-- 4Combine chunks with a fusion algorithm assuming the same score scale
top_10 AS (
SELECT bm.content AS doc1, vs.content AS doc2
FROM BM25 FULL OUTER JOIN VS ON BM25.idx = VS.idx
ORDER BY fusion(BM25.score::DOUBLE / (MAX(BM25.score) OVER ()),
VS.score::DOUBLE / (MAX(VS.score) OVER ()))
LIMIT 10)
-- 5Rerank top 10 elements if they are relevant to cyclic join queries
SELECT llm_rerank({ 'model ':'gpt-4o '},{'prompt ':'mentions cyclic joins '},
{'doc1 ': doc1, 'doc2 ': doc2})
FROM top_10;
Query 3: Hybrid search to find top 10 passages on cyclic joins
2.3 Optimizations
FlockMTL introduces several key optimizations to improve ef-
ficiency and usability: (i) meta-prompting for robust predictions
and simpler user queries; (ii) batching tuples into a single request
to improve latency; (iii) caching predictions for reuse within and
across queries; and (iv) predicting over deduplicated values to avoid
redundancy. Due to space limitations, we focus on (i) and (ii).
Meta-prompt. InFlockMTL , users provide prompts intended for
a single tuple (in the case of map functions) or for multiple tuples
(in reduce functions). The system then composes a full prompt
using the structured meta-prompt template shown in Fig. 1. This
meta-prompt includes the user-specified content and augments it
with formatting instructions, output expectations, and serialization
of tabular input tuples. It is implemented to be KV-cache friendly.

Anas Dorbani, Sunny Yasser, Jimmy Lin, and Amine Mhedhbi
(a). Data application with NL interface.
 (b). The plan inspection interface.
Figure 2: Screens of Prepared Demonstration for Users to Get Started.
Batching. When using FlockMTL ’s scalar functions, users write
prompts for a single tuple. However, making an API call per tuple is
inefficient, so FlockMTL automatically applies batching to optimize
inference. The system dynamically determines the batch size based
on the input attribute values and the LLM’s context window size. It
fills the prompt with as many tuples as possible until the context
limit is reached, then sends a single batched request. If the LLM
returns an error due to the output exceeding the context window,
FlockMTL automatically reduces the batch size by 10% iteratively
until a successful prediction is obtained. If a single tuple exceeds
the output context size, the result is set to NULL .
On the Kaggle Bank Review dataset, batching on a table scan
query with a single scalar FlockMTL function yields significant
performance gains–achieving up to 7 ×speedup for chat completion
map functions and 48 ×for embedding-based functions.
3 DEMONSTRATION
Goal. Our goals with FlockMTL is two fold. First, we aim to show-
case how easily users can build data applications using the ASK
functionality, combining analytics and semantic analysis within
an embedded RDBMS without the need to orchestrate multiple
external systems. Second, we aim to highlight the importance of
FlockMTL ’s low-level optimizations by involving the audience
in an interactive challenge. We added to our repo on github3a
demonstration for users to get started.
Interaction. The landing page of the demonstration presents a
data application where users can explore and interact with multi-
ple tabular datasets sourced from Kaggle and spanning domains
such as biomedical, academic, and product reviews. Users begin by
viewing a preview of the dataset made of one or more tables. To ex-
plore the dataset, attendees can issue a natural language query, e.g.,
“list reviews mentioning technical issues and assign a
severity score to each issue ” as done in Fig. 2a on a bank-
ing services review dataset, and FlockMTL ’sASKfunctionality
automatically generates a SQL query augmented with FlockMTL ’s
functions. Users can inspect the generated SQL query and its output.
This part of the demonstration illustrates the power of integrating
semantic operations directly into SQL and its ease of use.
3https://github.com/dsg-polymtl/flockmtl/Following this, by clicking on Inspect Plan for the generated
query shown in Fig. 2a, users are taken to a separate interface
for plan debugging and analysis. The separate interface shows
the plan of our example query in Fig. 2b. This query plan in-
cludes: (i) standard SQL operations such as scans and filters, and
(ii)FlockMTL specific functions, such as llm_filter as well as
llm_complete_json . The FlockMTL function box on the UI con-
tains additional system-level details, such as access to the full meta-
prompt used, the serialization format, and the batch size chosen
automatically by the system.
Users are first presented with the default setting where batch
size is set to Auto , hiding the one FlockMTL used. They can change
it manually and select a different batch size that might match the
system’s performance and accuracy. The default serialization for-
mat shown by default is XML, but users may modify it to JSON or
Markdown . For instance, if a user sets the batch size to 30and reruns
the query, they might observe a latency increase. We believe that
the plan inspection highlights trade-offs in latency and prediction
accuracy with different parameters. Finally, users can replace the
full prompt using a Jinja template, and then compare it in both
structure and output with the full one generated by FlockMTL .
We believe that this hands-on demonstration shows both the
accessibility of building semantic and analytical data applications in
SQL when compared with alternative systems [ 1,6] and the value
ofFlockMTL ’s optimizations in reducing developer burden.
REFERENCES
[1] A. Biswal and et al. Text2sql is not enough: Unifying ai and databases with tag.
InCIDR , 2025.
[2] T. B. Brown and et al. Language models are few-shot learners. NeurIPS , 2020.
[3] E. F. Codd. A relational model of data for large shared data banks. CACM , 1970.
[4] S. G. P. et al. Gorilla: Large language model connected with massive apis. CoRR ,
abs/2305.15334, 2023.
[5] P. S. H. Lewis and et al. Retrieval-augmented generation for knowledge-intensive
NLP tasks. In NeurIPS , 2020.
[6]C. Liu and et al. Palimpzest: Optimizing ai-powered analytics with declarative
query processing. In CIDR , 2025.
[7]X. Ma and et al. Zero-shot listwise document reranking with a large language
model. CoRR , abs/2305.02156, 2023.
[8]M. Raasveldt and H. Mühleisen. Duckdb: an embeddable analytical database.
SIGMOD , 2019.