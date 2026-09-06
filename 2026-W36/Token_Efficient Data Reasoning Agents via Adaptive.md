# Token-Efficient Data Reasoning Agents via Adaptive Structuring of Unstructured Data

**Authors**: Milad Rezaei Hajidehi, Qitong Wang, Stratos Idreos

**Published**: 2026-08-31 16:53:45

**PDF URL**: [https://arxiv.org/pdf/2608.31082v1](https://arxiv.org/pdf/2608.31082v1)

## Abstract
Valuable data remains embedded in unstructured sources: web pages, reports, contracts, filings, earnings calls, and PDFs. The big bet in enterprise AI is deploying LLM agents that reason over this data to answer complex questions for every knowledge worker. Agents can do this today, but at prohibitive cost. Each question repeatedly opens large documents to recover scattered evidence, consuming up to a million tokens. However, if the data were already structured, the same question would reduce to a cheap database lookup. For example, on FanOutQA benchmark, reasoning over an ideal pre-structured store is 28X cheaper, and the gap grows to orders of magnitude as questions fan out over more documents. Yet structuring everything in advance is not viable: documents hold vastly more possible structure than any workload will use, and the useful structure and documents are unknown until queries arrive. We propose agentic data cracking, a method that structures unstructured data adaptively and speculatively as a byproduct of reasoning itself. Structuring is adaptive because observed queries decide when it happens and what matters, and speculative because it goes beyond the current question. Whenever the agent opens a document to answer, a cracking sub-agent forks from the already-loaded context at marginal cost and extracts grounded structure likely to serve related future queries. Over time, an increasing share of queries is fully covered by structured data and answered without opening a document, keeping agentic accuracy at close to RAG cost. On FanOutQA, extended with merely one related question per test question, cracking cuts cost by 53% while preserving accuracy. Agentic data cracking is a first step toward next-generation data infrastructure for agentic reasoning over unstructured data: a shared substrate beneath the model where knowledge that reasoning already paid to uncover accumulates.

## Full Text


<!-- PDF content starts -->

Token-Efficient Data Reasoning Agents via
Adaptive Structuring of Unstructured Data
Milad Rezaei Hajidehi
Harvard University
Cambridge, Massachusetts, USA
milad@seas.harvard.eduQitong Wang
Harvard University
Cambridge, Massachusetts, USA
qitong@seas.harvard.eduStratos Idreos
Harvard University
Cambridge, Massachusetts, USA
stratos@seas.harvard.edu
Abstract
Across open and enterprise settings, valuable data remains embed-
ded in unstructured sources: web pages, reports, contracts, filings,
earnings calls, and PDFs. The big bet in enterprise AI is deploying
LLM agents that reason over this data to answer complex questions
for every knowledge worker. Agents can do this today, but at pro-
hibitive cost. Each question repeatedly opens large documents to
recover scattered evidence, consuming up to a million tokens. How-
ever, if the data were already structured, the same question would
reduce to a cheap database lookup. For example, on the FanOutQA
benchmark, reasoning over an ideal pre-structured store is 28 ×
cheaper, and the gap grows to orders of magnitude as questions fan
out over more documents. Yet structuring everything in advance
is not viable: documents hold vastly more possible structure than
any workload will use, and the useful structure and documents are
unknown until queries arrive.
We proposeagentic data cracking, an approach that structures
unstructured data adaptively and speculatively as a byproduct of
reasoning itself, in the spirit of database cracking and adaptive data
systems. Structuring is adaptive because observed queries decide
when it happens and what matters, and speculative because it goes
beyond the current question. Whenever the agent opens a docu-
ment to answer, a cracking sub-agent forks from the already-loaded
context at marginal cost and extracts grounded structure likely
to serve related future queries. Over time, an increasing share of
queries is fully covered by structured data and answered without
opening a document, keeping agentic accuracy at close to RAG
cost. On FanOutQA, extended with merely one related question per
test question, cracking cuts cost by 53% while preserving accuracy,
and already runs 9×cheaper at the 10th percentile of per-question
savings. Agentic data cracking is a first step toward next-generation
data infrastructure for agentic reasoning over unstructured data: a
shared substrate beneath the model where knowledge that reason-
ing already paid to uncover accumulates, so the system improves
with use for every agent and user that queries it.
1 Introduction
Knowledge and Insights Are Encased in Unstructured Data.
Much of the world’s knowledge is not stored in clean tables. It
lives in unstructured sources such as books, enterprise and legal
documents, web pages, SEC filings, earnings calls, and PDFs. The
evidence required to answer questions over these sources is scat-
tered across many documents. Consider this question: which cast
members worked with both Alfred Hitchcock and Martin Scorsese?
Over Wikipedia, answering it requires enumerating each director’s
films from their pages, extracting cast members from each film’s
page, connecting the resulting director-film and film-cast relations,
0 20 40 60 80 100
Accuracy (%)0.00.10.20.3Cost per Query (USD)RAG
(top-5)RAG
(top-10)Agentic
Reasoning
↓53%
Cheaper
Agentic
ReasoningIdeal
(Oracle Pre
Structured)0.00.10.20.328×Lower Cost
OpportunityRAG Agentic Reasoning IdealFigure 1: Left: On the full FanOutQA [ 36] benchmark over
Wikipedia, agentic reasoning far outperforms fixed top- 𝑘
RAG but at high cost. Our approach,agentic data cracking,
preserves almost all of this accuracy advantage while re-
ducing cost by 53%. Right: On ten sampled questions, we
extracted every fact required to answer each question from
the unstructured documents and stored it in a database. Rea-
soning over this ideal store is 28 ×cheaper. This gap is the
cost reduction available to adaptive structuring systems.
and intersecting the two cast sets. This flow is general: at each step,
evidence is gathered from documents, assembled into intermediate
entities, attributes, and relations, and passed to later steps. The
answer is not a set of retrieved passages. It is a latent structured
dataset, constructed step by step from sparse evidence inside many
large documents. We call this patterndata reasoning.
Data Reasoning over Unstructured Data is Already Every-
where and Still Growing.More than eighty commercial and open-
source systems have publicly launched to answer complex ques-
tions over the web and other large document collections [ 18,22,30],
and many more run privately inside enterprises [ 9,10]. Many of
the questions real users ask follow exactly the data reasoning pat-
tern [ 29]. Personal questions dominate in volume, and millions of
people ask them over the open web. Professional questions carry
higher stakes and target enterprise archives and public government
records [ 19]. A financial analyst might ask which companies blamed
supply chains on their earnings calls yet reported rising inventory
in their annual reports, matching a single remark in an hour-long
transcript against one line in a long filing. A health department
might ask which suppliers appear in both food recall notices and
restaurant inspection reports across multiple cities, with supplier
names scattered through lengthy inspection narratives. Answering
either means fanning out over hundreds of documents in multiple
steps.
arXiv:2608.31082v1  [cs.AI]  31 Aug 2026

Milad Rezaei Hajidehi, Qitong Wang, and Stratos Idreos
Agentic Data Reasoning Extracts Encased Knowledge but
at Prohibitive Cost.Agentic systems can execute the multi-step
flow needed to answer data reasoning questions. An agent plans,
searches, opens documents, extracts facts, executes code, maintains
intermediate state, and selects its next action based on the evidence
gathered so far [ 1,4]. These capabilities bring far higher answer
quality than models relying on parametric knowledge or fixed top-
𝑘retrieval [ 19,36]. Figure 1 (left) also shows this. This quality
comes at high cost. Data reasoning is largely prefill-intensive: the
agent repeatedly loads large documents into context to recover
scattered facts, while the final answer is often short. Cost grows
with dependent steps, entities, and documents. In our FanOutQA
experiments, one question can consume up to one million tokens
and cost nearly one dollar. Real-world data reasoning questions can
have greater fan-out, evidence requirements, and reasoning depth,
making them substantially more expensive [29].
Read it Once, Pay it Once: Queries Reveal Useful Structure.
If the data were magically structured, many of these questions,
including our Hitchcock example, would reduce to a simple and
low-cost Text2SQL query. We demonstrate this phenomenon in
Figure 1 (right). We sampled questions from FanOutQA, manually
extracted every fact their answers require into a database, and reran
the agent over the database instead of the raw documents. Each task
reduced to data discovery and a short query, without opening a sin-
gle document. This ideal pre-structured store makes reasoning28 ×
cheaper. This factor is specific to FanOutQA, whose questions span
seven documents on average [ 36]. Once structure covers a question,
the agent skips every document open, so the saving reaches or-
ders of magnitude for real questions that fan out over hundreds of
documents. While LLMs are excellent at capturing such structure,
structuring everything in advance is infeasible. Documents contain
orders of magnitude more entities, attributes, and relations than
any workload will use, and extracting them all is an unbounded
decode. Moreover, we do not know in advance which structure a
workload wants, or how often it will be reused. Only the workload
reveals which structure is worth extracting.
Two insights turn this dead end into an opportunity. First, at the
workload level, related queries reopen the same documents and
repeat much of the same work, while each query reveals which
structure and relations matter. This semantic locality is the norm,
not the exception. From the user’s side, conversations become in-
vestigations and users hold similar interests. From the data’s side,
the same documents get opened over and over because they hold
many related facts and evidence that queries need. Second, at the
inference level, once the agent loads a document, its prefill has
already been computed and its KV-cache entries already exist. With
careful prompt structure, shared-prefix serving [ 12,34], and API
prompt caching, a second generation over the same context avoids
or sharply reduces repeated prefill cost, leaving decoded tokens as
the main additional work. We give this decode a fixed budget that is
small relative to the prefill it saves, yet sufficient to produce reusable
structure. Together, the two insights suggest a move stronger than
caching. Cached answers help only when the exact question returns.
Speculating about useful structures goes much further: While the
document is loaded and paid for, extract grounded evidence that
related but distinct future queries will need.Agentic Data Cracking for Token/Cost-Efficient Data Rea-
soning.We introduceagentic data cracking, an approach that brings
the query-driven philosophy of database cracking and adaptive data
systems to AI reasoning [ 11,26]. The system never opens docu-
ments solely to extract structure. Instead, whenever the reasoning
agent opens one to answer, a cracking sub-agent forks from the
already-loaded context at low marginal cost. It speculates about
grounded, evidence-backed entities, attributes, and relations likely
to serve future queries. The system validates and catalogs these
cracked objects. Later queries reuse them via structured reads, avoid-
ing document opens entirely, and fall back to original documents
on a miss. Unlike database cracking, which reorganizes relations,
agentic data cracking creates useful structure where none exists.
This structuring is adaptive because observed queries guide what
is worth extracting, and agentic because a reasoning sub-agent, not
a fixed extractor or schema, uses semantic understanding to read
each document and speculate about what to extract.
The savings available to adaptive structuring are substantial.
Longer-running workloads and smarter cracking unlock them. The
ideal store is 28×cheaper on the FanOutQA benchmark, and this
gap widens with fan-out. In our case study, which models an evolv-
ing investigation, cracking cuts cost by 3 ×. On FanOutQA extended
with only a single reuse chance per test question, one earlier related
question, cracking still cuts cost by 53% while preserving accuracy.
At the tenth percentile of savings, cracking already runs 9 ×cheaper.
Reuse separates the average from this best case, and reuse chances
accumulate as the workload runs. The store an enterprise or AI
service accumulates through use becomes a durable asset, a data
moat that, unlike KV caches, transfers across models.
2 Agentic Reasoning: Capable but Expensive
Latent Data Reasoning Tasks.A major advance in language
modeling has been the ability to decompose complex tasks into
sequences of smaller reasoning steps. Reasoning tasks take dis-
tinct forms: solving IOI/ICPC problems, constructing data science
pipelines, and resolving knowledge-intensive questions over large
unstructured corpora (e.g., the web, Wikipedia [ 36], and enterprise
document collections [ 19]). In this paper, we focus on reasoning
tasks that require an agent to extract knowledge from unstructured
documents, construct intermediate data and supporting evidence at
each step, and pass those artifacts to later steps. In these tasks, the
answer comes from reasoning over the constructed intermediate
data rather than retrieving a single fact. We call this classlatent
structured data reasoning, ordata reasoningfor brevity.
Two Running Examples.Throughout the paper, we use two
questions to explain the problem, intuitions, and solution:
(1) What is the career average rebounds per game for each NBA
player who has won the IBM Award?
(2) Who are all the cast members who ever worked with both
Alfred Hitchcock and Martin Scorsese?
The first question comes from FanOutQA [ 36], our primary
benchmark. The second question is a harder data reasoning task
that we use in our case study. As illustrated in the introduction, an-
swering it requires a multi-step construction over many Wikipedia
pages. Although both questions appear casual, they follow the same
pattern as the high-stakes questions in the introduction.

Token-Efficient Data Reasoning Agents via Adaptive Structuring of Unstructured Data
Agentic Solutions: Capable but Expensive.LLMs that rely
only on parametric knowledge or conventional retrieval achieve low
accuracy on data reasoning tasks [ 19,36]. Agentic systems instead
imitate the data reasoning process by planning a sequence of steps
and adaptively choosing when to invoke tools [ 1,4]. The agent can
search the corpus, open documents, use filesystem search/navigation
API, extract facts, run python code, and store intermediate data for
later steps. Rather than following a fixed pipeline, the agent selects
its next action based on the question, the documents read so far,
and the intermediate state. The agent can also manage its context
by loading documents, extracting useful facts, and removing the
documents after use. These capabilities substantially improve accu-
racy and grounding, but processing large documents makes each
query expensive. For example, in our experiments with FanOutQA,
answering just a single question with the Haiku model consumes
up to one million tokens and costs nearly a dollar, using Haiku
Model and FanOutQA Agents [ 36]. These costs limit the scalability
and accessibility of agentic data reasoning.
3 Agentic Data Cracking: Adaptive Structuring
of Unstructured Data
We show that data systems techniques can solve the cost bottleneck
of data reasoning, substantially reducing token consumption. We
show that the high cost of AI reasoning creates a pressing new class
of problems for database research, requiring data systems and algo-
rithms designed around data reasoning workloads. Prior work has
likewise shown that data-management operators and abstractions
improve unstructured-data processing [3, 15, 16, 21, 27].
Documents and Prefill Dominate Cost.Reasoning workloads
have different inference profiles. In mathematical reasoning, little
task-specific evidence is available, the models work over their own
generated tokens; such workloads are often decode-intensive. Data
reasoning has the opposite profile. Under an accuracy-first policy,
the agent cannot speculate about missing facts. It must repeatedly
open documents, load them into context, extract needed facts, and
potentially remove them. Data reasoning is therefore largely prefill-
intensive. This bottleneck intensifies withdepth, as reasoning spans
dependent steps;width, as each step fans out over many entities
and documents; andsparsity, as a few relevant tokens are scattered
across much larger documents. In our cast-member example, the
agent opens one page per film although scattered cast names occupy
only a small fraction of each page.
Avoiding Document Prefill Through Structure and Its Chal-
lenges.One way to avoid document opens is to magically structure
document knowledge in a database. The prefill-heavy workload
becomes a short SQL query without opening a document; for exam-
ple, retrieving all cast members in Hitchcock films. Figure 1 (right)
demonstrates the performance gap between raw documents and a
pre-structured database. Yet extraction is difficult. Hand-designed
algorithms over unstructured data are task-specific and potentially
lossy. LLMs are more flexible but still require document prefill.
Documents also contain many entities, attributes, and relations,
making useful structure and its representation unclear. Extracting
everything would require an unbounded decode. Unlike a database
index, which organizes structure that already exists, exhaustive
corpus extraction is infeasible and may never amortize.Adaptive Agentic Data Cracking: Queries Guide What to
Extract.We build our system to extract structure adaptively as
queries reveal what is useful over time. When the answer agent
opens a document, we fork a parallel cracking branch over it. The
cracking sub-agent sees current and prior queries over the doc-
ument. Using semantic reasoning, our instructions, and our data
model, it speculates about reusable entity sets, attributes, and re-
lations. Crucially, it is not limited to the attribute or entity set
requested by the query. It also extracts nearby structure likely to
support related queries. Auxiliary algorithms normalize and val-
idate the output before storage. Extraction is semantic, selective,
and guided by observed demand. This design makes sense from
both application and low-level LLM inference perspectives.
From the application perspective, the approach is effective when
queries seek related objectives over overlapping entity sets. When
a query opens NBA player pages to obtain career points per game
for all-time scorers, the cracking sub-agent can also extract career
rebounds per game. Our running question can then reuse those
values for overlapping IBM Award winners without reopening their
pages. Such locality grows as conversations become investigations
and organizations form recurring categories and document hotspots
around the same companies, products, cases, and people. From the
performance perspective, the document’s KV cache already exists
when cracking begins. We reuse the cached prefix and fork a second
generation under a cracking instruction (Figure 2(a)), avoiding
another prefill through shared-prefix KV-cache reuse [ 12,34]. The
cracking branch runs in parallel and outside the answer path, so it
does not delay the current query. Its only additional model work is
the bounded decode that produces the structured output.
We call this approachagentic data cracking(Adc) because it
shares database cracking’s central principle: observed queries drive
adaptive, incremental organization for future reuse. However, un-
like database cracking, which reorganizes existing relations, agentic
data cracking adaptively creates useful structure where none ex-
ists. Adaptation is therefore more consequential here: exhaustive
extraction is infeasible, KV-cache reuse keeps overhead minor and
justifiable, and the payoff is larger LLM inference savings. It remains
durable and usable across models.
Cracked Objects & Data Model.Our system operates over facts
extracted by a cracking sub-agent running on an inference branch
forked from the answer agent after a document enters context. We
call these factscracked objects. We model each cracked object as
an extended RDF-style edge annotated with cardinality, unit, and
evidence:
𝑐=⟨𝑠, 𝑟, 𝑜, 𝜅,𝑢, 𝜀⟩∈C, 𝑠∈E, 𝑟∈R, 𝑜∈E∪V,
V≜ {str}×Σ∗∪ {int}×Z∪ {date}×D.
Here,EandRare open domains populated by the cracking sub-
agent. An object ( 𝑜) is either an entity or a tagged scalar, which
pairs an explicit type with a value. The current schema supports
strings, integers, and sortable dates. A cracked object therefore
represents an entity-to-entity or entity-to-value relation.
Each cracked object carries evidence 𝜀=⟨𝑑, 𝜌⟩, where 𝑑∈D
identifies its source and 𝜌identifies the supporting region within
that source. The cardinality 𝜅∈K={singular,list} distinguishes
single-object relations from relations that yield complete sets. It
defines completeness during reuse: one edge completes a singular

Milad Rezaei Hajidehi, Qitong Wang, and Stratos Idreos
<cracking>
Bounded
Cracking over
Last Document<task> <reasoning>
Thinking Logs,
Intermediate Data,
Actions<document>
Last Opened Document
After Prefill
Cached Shared Prefix(a) Write Behaviour: KV-Cache & Token Layout 
Cracked Object
Storage ()(b) Read Behaviour: Reasoning Agent Protocol<reasoning>
Thinking Logs,
Intermediate Data,
Actions...<system>
Shared Instructions
(Reasoning &
Cracking)
Cracked Object
Storage ()
Adaptive Structuring:
Write speculated useful objects,
when a document is "forced" to open.Cracking
Sub-Agent
Documents and
Catalogue View
(4) Return document tokens for (1) Request catalogue for document 
(2) Return 
(3) Request , if desired objects missing
Lookup by subject, relation, or
subject–relation pair ( )Reasoning Agent
Figure 2: (a) The cracking branch decodes over a shared prefix
to populate cracked-object store for later queries. (b) Reason-
ing protocol with catalogue lookup and document fallback.
relation, whereas a list relation is reusable only after all members are
extracted. We store list members as separate edges so each remains
independently indexable and joinable. The optional 𝑢∈U∪{⊥}
records a canonical unit. In our model, documents are logical units
that may represent a file, revision, or chunk.
Read/Reasoning Behaviour: Using Cracked Objects.Like
the conventional baseline agent, our main reasoning agent has
filesystem-style tools for searching, navigating, and opening docu-
ments. However, it also benefits from new constrained tool-calls
that support structured reads by subject, relation, and subject–
relation pair from the cracked-object store. These calls let the agent
reuse compact cracked objects without opening or prefilling raw
documents. When entity-name or relation-label mismatches occur,
they prevent retrieval of relevant stored objects. We therefore in-
troduce thecatalogue, a logical view over the cracked-object store.
For each document 𝑑, the catalogue lists available subject–relation
pairs with their canonical labels and cardinalities:
Cat(𝑑)≜𝜋 𝑠,𝑟,𝜅 𝜎doc(𝜀)=𝑑(C).
The catalogue helps the agent resolve available subjects and re-
lations before issuing a structured read (Figure 2(b)). The system
surfaces it with the document search result before the agent decides
whether to open the document. If the desired structure is present,
the agent issues the corresponding structured read and receives
compact values with evidence, avoiding document prefill. Other-
wise, it opens and prefills the raw document. A missing structured
result means that the required structure is unavailable or unresolved.
Raw-document access remains the conservative fallback.
Write/Cracking Behaviour: Producing Cracked Objects.
When cracked-object reads cannot satisfy an information need,
the main reasoning agent resolves the resulting “miss” by opening
the raw document. The system never opens documents solely for
cracking. Instead, after the document enters context, a cracking sub-
agent forks from the reasoning branch and targets that document.
In local serving, this fork can reuse the document’s KV-cache prefix,
avoiding a second prefill. For API models with prompt caching, theshared prefix hits the cache and is billed at a lower rate. Given
the document and missed information need, the sub-agent does
not merely cache the requested fact or extract predefined entities
and attributes. It uses semantic reasoning to speculate about which
additional entities, attributes, and relations may support future
queries, while extracting only grounded facts.
In our NBA example, a miss on one player statistic may prompt
extraction of other reusable player statistics. Cracking thus re-
sembles prefetching after a cache miss, but replaces fixed address-
based locality with agent-discovered semantic locality. Under a
fixed output-token budget, the sub-agent emits schema-constrained
JSON. It emits a complete list-valued relation or omits it. Post-
processing validates and normalizes the output, expands lists into
separate edges, and inserts only grounded cracked objects. Because
cracking remains outside the answer dependency path, it neither
blocks the current answer nor prevents the reasoning agent from
using the raw document if cracking fails. The read and write be-
haviour capture the core mechanism. The implementation also
includes cracking instructions and invariant checks. For instance,
the system canonicalizes numbers, units, and dates and requires
each invocation to crack the entire logical document, avoiding the
complexity and ambiguity of partially cracked data.
4 Evaluation
We test if agentic data cracking captures reusable structure across
related questions with related but distinct objectives and overlap-
ping but distinct entity sets. We measure how much this reuse
reduces inference cost and whether it preserves answer quality.
Evaluation Setup.The baseline and our system are built on
Claude-Haiku-4.5 with FanOutQA’s specifications, instructions,
prompts, and tool calls to access Wikipedia and the filesystem.
Our system differs from the baseline only in the cracking interface.
It adds cracking instructions and cracked-object read/write calls.
Prompt caching was enabled during all experiments. We measure
prefill and decode tokens, API cost, and answer accuracy.
Our benchmarks are FanOutQA [ 36] and a cinema case study
centered on Alfred Hitchcock. FanOutQA contains wide, multi-hop
questions over Wikipedia, ranging from Supreme Court justices to
NBA players. Its questions almost never revisit the same documents
and therefore lack the locality our system targets. This locality is
absent from reasoning benchmarks by design, since they assess the
quality of individual queries. Real-world workloads instead revisit
documents through related queries.
To recover this locality without altering the test questions, we
double the workload: for each test question, an LLM (claude-opus-5)
generates one related but distinct question, and a human verifies
it. Each generated question targets a different attribute over an
overlapping entity set. The generator sees only the target question
and our criteria: answerable from Wikipedia, overlapping entities,
and a distinct attribute. It is unaware of the cracking system, its
design, implementation, or outputs. For our running NBA example,
the generated question asks for the career points per game of the
top ten all-time NBA scorers. Caching answers cannot help here.
No generated question repeats or paraphrases its test question, and
the two never share the requested attribute. Any saving comes from
semantic speculation about related attributes. In both FanOutQA

Token-Efficient Data Reasoning Agents via Adaptive Structuring of Unstructured Data
FanOutQA Case Study0200K400K600KMean Tokens(a) Token Usage
pre/f_ill: 189K
decode: 0.86K
pre/f_ill: 87K
decode: 0.94Kpre/f_ill: 565K
decode: 2.5K
pre/f_ill: 161K
decode: 2.4K
FanOutQA Case Study0.00.20.40.60.8Cost per Query (USD)(b) Total Cost (LLM API)
$0.26$0.81
$0.12$0.27
LLM-as-Judge String
Accuracy0255075100Accuracy (%)(c) FanOutQA Accuracy
43.0%76.1%
42.0%74.2%
$.001 $.01 $.10 $10255075100% of Questions(d) Cost Distribution (CDF)
Median cost
Baseline: $0.246
A/d.sc/c.sc: $0.072
3.4×cheaperBaseline (Agentic Reasoning) A/d.sc/c.sc (Agentic Reasoning with Cracking)
Figure 3: Performance of Agentic Data Reasoning with and without Data Cracking.
and the case study, all primary metrics are reported on the origi-
nal test questions only. First, this keeps the FanOutQA accuracy
evaluation well defined because the generated related questions
have no gold answers. Second, the cracked-object store is populated
entirely by the preceding related questions, allowing us to evaluate
a warmed store without the noise of an empty-store cold start. We
report the additional cracking cost separately.
The case study models a longer investigation, closer to cracking
in long-running systems. It has fewer test questions than FanOutQA
but more related questions, around a single interest. We first ask 20
questions about Hitchcock’s films, career, and life. We evaluate 10
related test questions that reuse these cracked objects. The sequence
represents an evolving film investigation.
Results.Figure 1 (left) is our headline. Cracking keeps agen-
tic accuracy at close to RAG cost. All systems, including the RAG
baselines, use the same model. Figure 1 (right) shows the opportu-
nity for adaptive structuring systems. With either ideal cracking
or longer-running systems, cost converges toward the ideal struc-
tured store. Figure 3 breaks the headline down. Figure 3(a) shows
token usage. Cracking cuts mean prefill from 189K to 87K tokens
on FanOutQA and from 565K to 161K on the case study. Decode
stays below 3K tokens in all settings. We also report cost as a com-
prehensive metric and the most important one for AI applications.
Figure 3(b) shows that mean cost per question drops from $0.26
to $0.12 on FanOutQA and from $0.81 to $0.27 on the case study.
The case study is more expensive overall because its questions are
harder, with two to four times the fan-out. Its improvement is also
larger because twenty related questions come first, so test questions
are more likely to find cracked objects to reuse. Figure 3(c) shows
accuracy. Both metrics come from FanOutQA. String accuracy is
the fraction of gold-answer facts matched at the word level. LLM-as-
judge scores the answer against the gold answer and instructions
using GPT, as specified by the benchmark. The judge is harsher
on partially correct answers. LLM-judged accuracy differs by one
point from the baseline (42% vs. 43%, 𝑝-value =0.39), with no sta-
tistically significant difference and word-match accuracy drops by
two points, mostly due to answer canonicalization. The similar ac-
curacy is rooted in our design where we prioritize a robust fallback.
When reuse misses, the agent reads the original document. A more
aggressive policy could trade this robustness for larger savings.
Figure 3(d) shows the independently sorted per-question cost
distributions. The median question costs $0.246 under the baseline
and $0.072 under cracking, a 3.4 ×ratio of medians. The gain is notuniform. For paired question-by-question cost ratios, cracking is
9×cheaper at the 10th percentile and costs 1.24 ×the baseline at the
90th percentile. These are questions with no reuse, where cracking
and data discovery add overhead without savings. Cracking pays
off on three quarters of the questions and wins clearly in aggregate.
We set a 4K-token cracking decode budget per question, spent
across cracking forks. On FanOutQA, this adds 12% overhead per
question. It includes the decode cost and cache read that vendors
charge for previous context when forking. This budget produces
roughly 150 cracked objects. Larger budgets yield proportionally
more objects, but the additional objects tend to be narrower and less
likely to be reused. This cost amortizes once a question’s cracked
objects let later questions avoid one or two document opens. A few
avoided opens repay the budget, and every open beyond that saves a
document prefill. Smarter budget allocation (e.g., per specific query
or document) remains future work. Even with our simple allocation
and cracking overhead, the gains in Figure 3 hold by a wide margin.
5 Related Work
Deep-research and reasoning agents answer complex questions over
large corpora with multi-step search, document opens, and evidence
synthesis [ 1,4]. Prior work introduces data-management operators,
abstractions, and execution strategies for processing unstructured
data [ 3,15,16,21,25,27]. Agentic data cracking complements both
lines by speculating and accumulating reusable, evidence-backed
structure that these agents and operators can run over, so later
queries reuse the document opens that earlier queries paid for.
OpenIE mines relational tuples from text without a predefined
schema [ 7]. Knowledge-base construction, knowledge graphs, and
graph-based RAG build representations before queries arrive [ 6,
28,33]. Query-driven document analytics extracts only values
needed by the current query [ 14]. Dataspaces add integration struc-
ture incrementally as queries reveal what matters [ 26]. Our ap-
proach sits between upfront construction and per-query extraction.
The data model is open, queries guide extraction, and cracking
builds structure incrementally from documents already opened by
queries. Semantic caches return answers to near-duplicate ques-
tions [ 2]. Cracked objects remain compact, evidence-backed, and
reusable across distinct queries. Serving systems persist document
and shared-prefix KV state to skip prefill [ 8,17,23,32]. Cached KV
state stays bound to one model, is far larger than its source text,
occupies GPU or host memory, and adds transfer and cache-read
costs. Cracking complements KV-cache reuse by reading the cached

Milad Rezaei Hajidehi, Qitong Wang, and Stratos Idreos
prefix. Cracked structures are plain text, so they survive model up-
grades, transfer across LLMs, and compound into an organizational
asset that continues saving tokens and cost.
Agent memory systems persist information across sessions [ 5,
13,20,24,31,35]. One can also view agentic data cracking as a
memory system, but a memory withspeculationandprefetching:
it memorizes and speculates structure over the corpus rather than
a user’s preferences and past answers, and it writes that struc-
ture, amortizing extraction cost across future queries. MemGPT
pages conversation history between a bounded context window
and external storage [ 20], and Letta extends that design into an
agent framework [ 13]. Mem0 extracts and consolidates salient facts
from ongoing conversations into a long-term store [ 5], Memory-
Bank decays and reinforces stored memories over time [ 35], Zep
maintains a temporal knowledge graph over conversational and
business data [ 24], and A-Mem organizes memories as dynamically
linked notes [ 31]. Memory writes in these systems are triggered by
conversation turns and scoped to what was discussed.
6 Discussion
Design Decisions, Limitations, and Future Work.We expose
constrained read functions rather than the full SQL space. This
keeps the tool interface small, fallbacks explicit, and generated
queries reliable. However, SQL supports counting, aggregation,
joins, and direct multi-hop queries. Safely exposing these capabili-
ties remains interesting future work. Our current system assumes
a static corpus. Edits could be handled by dropping the affected
cracked objects or incrementally refining them from the changes.
Our benchmarks use open, commonly trained-on Wikipedia
content. Private office documents are unseen by the model and
organization-specific. They are also repeatedly queried across users
and workflows, making them an even more interesting workload
for cross-query reuse. To reflect real-world workloads, we need
benchmarks that capture the long history of queries over a doc-
ument, ideally drawn from deployed systems. Such benchmarks
would expose a large and practical opportunity to improve both
individual queries and long-running real-world systems.
7 Conclusion
Agentic data reasoning can answer complex questions over unstruc-
tured corpora, but repeatedly reading documents makes it expen-
sive. We introduced agentic data cracking. It uses observed queries
to speculate about useful structure and extract it from documents
already opened during reasoning. Future queries can then reuse
this structure. The design exploits locality twice: related queries
reuse extracted structure, and cracking itself reuses the document’s
KV cache. We showed that agentic data cracking reduces token
usage and API cost while preserving answer quality. The remaining
gap to an ideal fully structured store suggests a broad opportunity
for data systems to improve agentic reasoning, especially in work-
loads where useful structure emerges across queries. This direction
connects decades of work on the Semantic Web, entity resolution,
and adaptive data systems with the reasoning ability of modern
agents. Combining these foundations can make agentic reasoning
more scalable and reliable. Agentic data cracking is an initial step
toward such adaptive data systems for AI reasoning. Beyond oursystem, the idea applies wherever systems repeatedly reason over
large unstructured documents.
References
[1] Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee
Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’arcy, David Wadden,
Matt Latzke, Minyang Tian, Pan Ji, Shengyan Liu, Hao Tong, Bohao Wu, Yanyu
Xiong, Luke Zettlemoyer, Graham Neubig, Daniel S. Weld, Doug Downey, Wen-
tau Yih, Pang Wei Koh, and Hannaneh Hajishirzi. 2026. Synthesizing Scientific
Literature with Retrieval-Augmented Language Models.Nature(2026).
[2] Fu Bang. 2023. GPTCache: An Open-Source Semantic Cache for LLM Applica-
tions Enabling Faster Answers and Cost Savings. InNLP-OSS Workshop 2023.
[3] Ugur Cetintemel, Shu Chen, Alexander W. Lee, Deepti Raghavan, Duo Lu, and
Andrew Crotty. 2026. Making Prompts First-Class Citizens for Adaptive LLM
Pipelines. InCIDR 2026.
[4] Zijian Chen, Xueguang Ma, Shengyao Zhuang, Jimmy Lin, Akari Asai, and Victor
Zhong. 2026. AgentIR: Reasoning-Aware Retrieval for Deep Research Agents.
arXiv:2603.04384(2026).
[5] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav.
2025. Mem0: Building Production-Ready AI Agents with Scalable Long-Term
Memory.arXiv:2504.19413(2025).
[6]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From Local to Global: A Graph RAG Approach to Query-Focused
Summarization.arXiv:2404.16130(2024).
[7] Oren Etzioni, Michele Banko, Stephen Soderland, and Daniel S. Weld. 2008. Open
Information Extraction from the Web.Commun. ACM51, 12 (2008).
[8] In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and
Lin Zhong. 2024. Prompt Cache: Modular Attention Reuse for Low-Latency
Inference. InMLSys 2024.
[9] Glean. 2025. Glean: Work AI for Enterprises. https://www.glean.com (2025).
[10] Hebbia. [n. d.]. Introducing Matrix. https://www.hebbia.com.
[11] Stratos Idreos, Martin L. Kersten, and Stefan Manegold. 2007. Database Cracking.
InCIDR 2007.
[12] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAttention.
InSOSP 2023.
[13] Letta. 2025. Letta: Stateful Agents with Persistent Memory. https://www.letta.
com (2025).
[14] Yiming Lin, Madelon Hulsebos, Ruiying Ma, Shreya Shankar, Sepanta Zeighami,
Aditya G. Parameswaran, and Eugene Wu. 2025. Querying Templatized Docu-
ment Collections with Large Language Models. InICDE 2025.
[15] Chunwei Liu, Matthew Russo, Michael Cafarella, Lei Cao, Peter Baile Chen,
Zui Chen, Michael Franklin, Tim Kraska, Samuel Madden, Rana Shahout, and
Gerardo Vitagliano. 2025. Palimpzest: Optimizing AI-Powered Analytics with
Declarative Query Processing. InCIDR 2025.
[16] Shu Liu, Soujanya Ponnapalli, Shreya Shankar, Sepanta Zeighami, Alan Zhu,
Shubham Agarwal, Ruiqi Chen, Samion Suwito, Shuo Yuan, Ion Stoica, Matei
Zaharia, Alvin Cheung, Natacha Crooks, Joseph E. Gonzalez, and Aditya G.
Parameswaran. 2026. Supporting Our AI Overlords: Redesigning Data Systems
to be Agent-First. InCIDR 2026.
[17] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng
Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, Michael Maire,
Henry Hoffmann, and Junchen Jiang. 2024. CacheGen: KV Cache Compression
and Streaming for Fast Large Language Model Serving. InSIGCOMM 2024.
[18] OpenAI. 2025. Introducing Deep Research. https://openai.com/index/
introducing-deep-research/ (2025).
[19] Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang,
Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, Michael
Bendersky, Matei Zaharia, and Xing Chen. 2026. OfficeQA Pro: An Enterprise
Benchmark for End-to-End Grounded Reasoning.arXiv:2603.08655(2026).
[20] Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion
Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as Operating
Systems.arXiv:2310.08560(2023).
[21] Liana Patel, Siddharth Jha, Melissa Pan, Harshit Gupta, Parth Asawa, Carlos
Guestrin, and Matei Zaharia. 2025. Semantic Operators and Their Optimization:
Enabling LLM-Based Data Processing with Accuracy Guarantees in LOTUS.
PVLDB18, 11 (2025).
[22] Perplexity. 2025. Introducing Perplexity Deep Research. https://www.perplexity.
ai/hub/blog/introducing-perplexity-deep-research (2025).
[23] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Mingxing Zhang,
Yongwei Wu, Weimin Zheng, and Xinran Xu. 2025. Mooncake: Trading More
Storage for Less Computation — A KVCache-centric Architecture for Serving
LLM Chatbot. InFAST 2025.

Token-Efficient Data Reasoning Agents via Adaptive Structuring of Unstructured Data
[24] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel
Chalef. 2025. Zep: A Temporal Knowledge Graph Architecture for Agent Memory.
arXiv:2501.13956(2025).
[25] Matthew Russo and Tim Kraska. 2026. Deep Research is the New Analytics
System: Towards Building the Runtime for AI-Driven Analytics. InCIDR 2026.
[26] Marcos Antonio Vaz Salles, Jens-Peter Dittrich, Shant Kirakos Karakashian,
Olivier René Girard, and Lukas Blunschi. 2007. iTrails: Pay-as-you-go Information
Integration in Dataspaces. InVLDB (2007).
[27] Shreya Shankar, Tristan Chambers, Tarak Shah, Aditya G. Parameswaran, and
Eugene Wu. 2025. DocETL: Agentic Query Rewriting and Evaluation for Complex
Document Processing.PVLDB18, 9 (2025).
[28] Denny Vrandečić and Markus Krötzsch. 2014. Wikidata: A Free Collaborative
Knowledgebase.Commun. ACM57, 10 (2014).
[29] Tomer Wolfson, Harsh Trivedi, Mor Geva, Yoav Goldberg, Dan Roth, Tushar
Khot, Ashish Sabharwal, and Reut Tsarfaty. 2026. MoNaCo: More Natural and
Complex Questions for Reasoning Across Dozens of Documents.TACL14 (2026).
[30] Renjun Xu and Jingwen Peng. 2025. A Comprehensive Survey of Deep Research:
Systems, Methodologies, and Applications. arXiv:2506.12594 (2025).[31] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang.
2025. A-Mem: Agentic Memory for LLM Agents. InNeurIPS 2025.
[32] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang,
Kuntai Du, Shan Lu, and Junchen Jiang. 2025. CacheBlend: Fast Large Language
Model Serving for RAG with Cached Knowledge Fusion. InEuroSys 2025.
[33] Ce Zhang, Christopher Ré, Michael J. Cafarella, Jaeho Shin, Feiran Wang, and
Sen Wu. 2017. DeepDive: Declarative Knowledge Base Construction.CACM
(2017).
[34] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang,
Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez,
Clark Barrett, and Ying Sheng. 2024. SGLang: Efficient Execution of Structured
Language Model Programs. InNeurIPS 2024.
[35] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Mem-
oryBank: Enhancing Large Language Models with Long-Term Memory. InAAAI
2024.
[36] Andrew Zhu, Alyssa Hwang, Liam Dugan, and Chris Callison-Burch. 2024.
FanOutQA: A Multi-Hop, Multi-Document Question Answering Benchmark
for Large Language Models. InACL 2024.