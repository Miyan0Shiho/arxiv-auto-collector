# ScalableRAG: High-Quality RAG at Zero Ingestion Cost

**Authors**: Hilaf Hasson, Aditya Chakravarty, Jayant Thomas, Krishna Gogineni

**Published**: 2026-07-27 23:03:30

**PDF URL**: [https://arxiv.org/pdf/2607.25135v1](https://arxiv.org/pdf/2607.25135v1)

## Abstract
Recent advances in RAG aim to optimize for performance by paying high ingestion costs for knowledge ingestion: building knowledge graphs or extracting SQL tables. In this work we show that the operations that such knowledge bases allow can be replicated with zero ingestion costs (not even a vector database); in fact our solution, Zero-Ingestion ScalableRAG, handily out-performs all baselines (including knowledge graph approaches) in three out of the six corpora considered here, and only marginally missing maximum performance on the other three, with average accuracy across all six datasets 7.36% above the next most competitive baseline. It achieves this by keeping a workspace of document sets and values sets that it can write into and read from, allowing for on-the-fly aggregative reasoning in all situations where grouping is required on a primary key that is in one to one correspondence with a subset of the total document set.
  Capping the number of LLM calls by a constant independent of the corpus size, we also introduce Limited-Ingestion ScalableRAG, which does use a minimal vector database as well as an automated pattern discovery from a sample of documents, to further improve accuracy at scale. Our code is available at https://github.com/cohesity/ScalableRAG .

## Full Text


<!-- PDF content starts -->

ScalableRAG: High-Quality RAG at Zero Ingestion Cost
Hilaf Hasson1, Aditya Chakravarty1, Jayant Thomas1, Krishna Gogineni1
1Cohesity
1{hilaf.hasson,aditya.chakravarty,jayant.thomas,krishna.gogineni}@cohesity.com
Abstract
Recent advances in RAG aim to optimize
for performance by paying high ingestion
costs for knowledge ingestion: building
knowledge graphs or extracting SQL tables.
In this work we show that the operations
that such knowledge bases allow can be
replicated with zero ingestion costs (not
even a vector database); in fact our solu-
tion, Zero-Ingestion ScalableRAG, handily
out-performs all baselines (including knowl-
edge graph approaches) in three out of
the six corpora considered here, and only
marginally missing maximum performance
on the other three, with average accuracy
across all six datasets 7.36% above the next
most competitive baseline. It achieves this
by keeping a workspace of document sets
and values sets that it can write into and
read from, allowing for on-the-fly aggrega-
tive reasoning in all situations where group-
ing is required on a primary key that is in
one to one correspondence with a subset of
the total document set.
Capping the number of LLM calls by a
constant independent of the corpus size,
we also introduce Limited-Ingestion Scal-
ableRAG, which does use a minimal vector
database as well as an automated pattern
discovery from a sample of documents, to
further improve accuracy at scale. Our
code is available at https://github.com/
cohesity/ScalableRAG.
1 Introduction
Retrieval-augmented generation (Karpukhin
et al., 2020; Lewis et al., 2020; Douze et al.,
2025) was first introduced as creating a vector
database from chunks of a corpus of documents,
and then using nearest neighbors between the
embedding of the question and the embeddings
of the corpus chunks. Since then there have
been two main advances in RAG: The first isto create a knowledge data structure by having
an LLM read the entire corpus and indexing
in either a knowledge graph (Edge et al., 2024;
Bai et al., 2025; Gutiérrez et al., 2024, 2025)
or a schema (Koshorek et al., 2025). Such
methods have the benefit that they make it
possible for the retrieval agent to group by any
key, and create sets on which some aggregative
operation (count, average, etc.) produces the
correct answer. A second nascent approach
is to keep ingestion to be vector embeddings
only, and to add onto it an agentic retrieval
(Du et al., 2026; Hui et al., 2025).
In this paper we aim to replicate the success
of creating a knowledge data structure, but
with minimal, even zero, ingestion costs. The
main insight we use is that frequently the key
on which we want the system to group by is in
one to one correspondence with a subset of the
set of documents in the corpus, and thus can
efficiently be generated at inference.
Zero-Ingestion ScalableRAGrequires no
ingestion costs whatsoever, including no vector
database creation. Using regex tools, it can
create and persist subsets of filenames, sets
numbers, sets of dates, and sets of lists. Each
such set gets named, and can be used in future
steps using set operations, filtering, and ag-
gregative tools. Each time a set is created, the
previous sets are used in order to give the agent
context about the differences between the sets,
so that it can make better decisions on whether
the filtering worked as anticipated. Already
this base case, with zero ingestion, outperforms
most ingestion-heavy methods in most of our
experiments.
In order to further improve performance
while keep ingestion costs minimal, we build on
this solution to introduceLimited-Ingestion
ScalableRAGby adding two more capabili-
ties: Vector embedding (with large chunks), al-
1
arXiv:2607.25135v1  [cs.AI]  27 Jul 2026

lowing for cosine similarity, as well as inference-
time classifier training and inference on chunks;
and a thorough pattern discovery and valida-
tion preprocessing step that uses only a sample
of the documents. The latter decomposes as: 1.
Deterministic discovery for regex patterns, 2.
LLM-generated regex discovery, and 3. Discov-
ery of extraction hints for values that cannot
easily be extracted through regex. The three
patterns get thoroughly vetted at ingestion,
and then exposed strategically to the retrieval
agent.
To summarize, our contributions are:
1.IntroducingZero-Ingestion Scal-
ableRAG(Section 3.1, a RAG system
that does not even require vector em-
beddings, and uses inference-time set
persistence to achieve state of the art
performance compared to even heavy-
ingestion RAG systems. This solution
handily out-performs all baselines (includ-
ing knowledge graph approaches) in three
out of the six corpora considered here,
and only marginally missing maximum
performance on the other three, with aver-
age accuracy across all six datasets about
7% above the next most competitive
baseline. (See Section 4.)
2.IntroducingLimited-Ingestion Scal-
ableRAG(Section 3.2), which builds on
Zero-Ingestion ScalableRAG but adds vec-
tor embeddings, and a thorough pattern
and extraction discovery and validation
pre-processing so as to improve perfor-
mance with aconstantnumber of LLM
calls. This leads to moderate accuracy
improvements in those datasets where ex-
traction is more challenging.
Our code is available at https://github.
com/cohesity/ScalableRAG.
2 Related Work
Baseline RAG.RAG based on dense re-
trieval (Karpukhin et al., 2020; Lewis et al.,
2020; Douze et al., 2025) remains the standard
reference for dense passage retrieval. There
havebeensomeimprovementsthatretaindense
retrieval as the core primitive. To name a
few: HyDE (Gao et al., 2023) augments doc-
uments with synthetic queries at ingestion toimproveperformance; HyQE(Zhouetal.,2024)
augments questions with synthetic documents
at inference to improve performance; FLARE
(Jiang et al., 2023) retrieves at generation when
the LLM is “uncertain”; Iter-RetGen (Shao
et al., 2023) iterates retrieval and generation;
and IRCoT (Trivedi et al., 2023) interleaves
retrieval with chain-of-thought.
Ingestion-Heavy Knowledge Representa-
tionGraphRAG (Edge et al., 2024) popular-
ized the idea of indexing a corpus as an entity-
relation graph and retrieving community-level
summaries. HippoRAG (Gutiérrez et al., 2024)
and HippoRAG2 (Gutiérrez et al., 2025) ex-
tend this with personalized PageRank over an
openIE-derived graph, motivated by hippocam-
pal indexing theory. AutoSchemaKG (Bai
et al., 2025) pushes the scale further by au-
tomatically inducing schemas (conceptualizing
entities into abstract types) and building a
triple index across 50M+ documents. For a
good survery on knowledge graph solutions for
RAG see (Peng et al., 2025).
SRAG (Koshorek et al., 2025) takes a dif-
ferent approach by ingesting the knowledge
into forming a SQL table with one row per
document, populated via LLM extraction. We
remark that while SRAG follows a completely
different algorithm to ours, by allowing only
one row per document (and having a single
table rather than multiple tables joined by for-
eign keys) it also makes the bet we are making:
that most questions group by a primary key
that is in one-to-one correspondence with a
subset of the documents.
All of these methods require heavy ingestion:
typically at least one LLM call per document,
often much more.
Agentic RAG.A-RAG (Du et al., 2026)
is an agent equipped with three retrieval
tools:keyword_search for exact lexical
matching that returns snippet-level evidence,
semantic_search for dense sentence-level re-
trieval (using a vector database) grouped by
chunks, and chunk_read for reading the full
text of selected chunks. Interact-RAG (Hui
et al., 2025), in contrast, exposes the retrieval
process itself as an interactive environment, en-
abling an agent to explicitly control retrieval
strategies (semantic vs. exact search and their
fusion), enforce entity-anchored matching, and
2

dynamically shape context through inclusion,
exclusion, and scale adjustment actions. We
remark that neither solution creates persistent
artifacts for future steps.
3 Proposed Method
We describe ScalableRAG in two stages. In Sec-
tion 3.1 we present thezero-ingestionsystem: a
stateful agent whose sole input is a directory of
plain-text files. In Section 3.2 we layer on the
two optional ingestion modules that together
constitutelimited-ingestionScalableRAG: vec-
tor embeddings, automatic pattern discovery.
3.1 Zero-Ingestion ScalableRAG
The zero-ingestion system requires no pre-
processing: not only no knowledge graph or
schema extraction, but also no vector embed-
dings. The agent is given only the corpus and
a question.
3.1.1 Agent Architecture
LetC={d1,...,d N}be the corpus of Ndoc-
uments.
ScalableRAG is at its base a ReAct-style
agent (Yao et al., 2022) that interacts with the
corpus through tool calls. Given a question q,
the agent produces a sequence of( at,ot)pairs,
whereatis a JSON-formatted tool invocation
andotis the tool observation, until it emits a
final answer.
However, unlike a ReAct-style agent, at turn
t, the agent maintains aset registry S(t)=
{(si,Di)}i, where for each iDiis a subset ofC,
and its name is si. The registry is initialized to
S(0)={(all,C )}and is updated in-place after
every tool invocation. In addition, the agent
maintains a set ofvalue sets V(t)={(vj,fj)}j,
where thevjis the name of the function fj:
D→R(scalar),fj:D→Rk(date), orfj:
D→ 2Σ∗(list of strings), where Dis one of
the sets inS(t).
At each step the agent selects a tool τtand
arguments αt; the tool reads from and writes
to the shared state:
(S(t),V(t),ot) =τ t(αt;S(t−1),V(t−1),C)(1)
This is inherently different from all prior
agentic RAG solutions, and allows the agent
to gain significant helpful knowledge at each
turn by automatizing computations that refer
to existing sets; see Section 3.1.2 for details.Algorithm 1ScalableRAG Agent Loop
Require: CorpusC, questionq, LLMM, max steps
T, context budgetB
1:S(0)←{(all,C)};V(0)←∅
2:ℓ←QuestionLandscape(q,C){keyword
statistics}
3: msgs←[SysPrompt, q⊕ℓ]
4:fort= 1,...,Tdo
5: msgs←RedactAndCompress(msgs,B){con-
text management}
6: response←M(msgs)
7:ifresponse containsanswerthen
8:returnresponse.answer
9:end if
10: Parse(τ t,αt)←response.tool, response.args
11:(S(t),V(t),ot)←τ t(αt;S(t−1),V(t−1),C)
12: Append assistant response ando tto msgs
13:end for
14:
15:returnM(msgs⊕“give your best answer”)
The full system prompt is given verbatim in
Appendix A; and context management details
(per-tool output budgets, observation redac-
tion after consumption) are described in Ap-
pendix B.
Lexical pre-analysis (used only for guid-
ance).Before the first LLM turn, Scal-
ableRAG performs a zero-cost scan to give
the agent a coarse map of the search space.
Concretely, it extracts up to five distinctques-
tion keywords(maximal alphanumeric spans of
length≥4) and, for each keyword, counts how
many documents contain it by case-insensitive
substring inclusion. It also reports filename hit
counts and up to three pairwise intersections
among the keywords that matches at least one
but fewer than all of the documents. These
statistics are not retrieval: they are a cheap
prior that tells the agent whether it should
start from filenames, from a single broad text
filter, or from a narrower combination. The
question keywords above get sorted based on
a lightweight logic (see Appendix B), and are
henceforth called “anchor tokens”.
Algorithm 1 formalizes the agent loop.
Producing document sets.Every filtering
operation takes an existing named set as its
target. This means filtering is alwaysrelative:
the agent refines incrementally, and all observa-
tions are computed with respect to the target
set, not the full corpus.
•apply_filter (ρ,S)→(S+,S−): applies
regexρto every document in the named
3

target setS, partitioning it into two new
named sets: a positive set S+⊆S(doc-
uments matching ρ) and a negative set
S−=S\S+. Both are registered in the
workspace. Because the partition is rela-
tive toS, not to the full corpus, the agent
sees snippets from both sides ofits cur-
rent working set: positive snippets show
why a document matched, negative snip-
pets are centered on a question-derived an-
chor token (see Appendix B) so the agent
can inspect format variants that the regex
missed within the same scope.
•search_filenames (p,S)→D: filters by
filename pattern, optionally restricted to
a target set.
•set_operation (A,B,op )→D: explicit
∩,∪,\. The registry also resolveslazy
compound names: the agent may write
A_and_B in any tool argument and the
system computes A∩Bon the fly without
a separate call.
•filter_values (v,op,θ )→D: promotes
a value set back into a document set
by thresholding extracted values. Oper-
ationally, the workspace stores a value
setvjas a partial mapping from docu-
ment IDs to typed values (numbers or
dates) for the subset of documents where
extraction succeeded. The tool returns
D={d∈dom (vj) :vj(d)opθ}(e.g.,
date≥2020-01-01, revenue >1B) and reg-
isters it as a new named set; documents
with missing or non-numeric values are
simply absent fromdom(v j).
Producing value sets.Value sets lift a doc-
ument set into structured per-document data.
These tools performregex extraction: the regex
is used to capture a value (via a capture group)
and populate a per-document mapping, not to
decide whether the document belongs in the
set.
•extract_field (ρ,S)→v: runs a regex
with a capture group over every document
inS; stores the first match per doc as a
scalar (number, text, or date).
•extract_list (ρ,S)→v: storesall
matches per doc as a list, for repeating
items (participants, line items, references).•extract_from_filename (ρ,S)→v: cap-
tures metadata from filenames (e.g., ticker,
quarter, category) via a capture group.
Reducing to answers. aggregate (v,op)
reduces a value set: sum,avg,min,max,count
for scalars; unique_values ,count_unique ,
value_counts , group_by_value ,
per_doc_count for lists. Notably, min/max
can additionally produce a new document set
containing the extremal documents, letting
the agent read them. count_set (S)returns
the exact cardinality |S|.calculate (expr)
evaluates arithmetic safely.
Observation and planning. read_docs
reads up to kdocuments from a target
set, optionally sampling from the end via
offset =−kto expose format variants. For
long documents, it supportscontiguous paging
(chunk_idx ) and precise slices ( start_char ,
max_chars ) so the agent can zoom into
specific regions without dumping entire fil-
ings/transcripts. find_in_doc is a deter-
ministic in-document navigator that returns
character offsets for exact match (substring
or regex) and a query-aware ranked-window
mode; usedwith read_docs(start_char=...,
max_chars=...) , it enables reliable “jump-to-
section” behavior in huge files. table_lookup
locates a row in aligned/table-like text and re-
turns the row plus nearby header lines (often
containingyears/columns)andparsednumbers.
explore_patterns evaluates a small batch of
candidate regexes against a target set and re-
turns match counts and samples without creat-
ing new sets, enabling cheap comparison before
committing. create_regex registers a regex
for reuse (or asks the LLM to propose one
given a small sample set). list_sets exposes
the current workspace state (set names, sizes,
and provenance), which the agent uses as a
planning view.
Branch-and-Select for deterministic
tools.Many tool calls are deterministic
given their JSON arguments (e.g., lexical
search, in-document navigation, and table
lookup). ScalableRAG canspeculatively pre-
viewa small set of alternative tool invocations,
generated automatically from the agent’s
proposed action, and then ask the LLM to
select which branch to execute. Only the
4

selected action is committed to the workspace;
previews are discarded.
3.1.2 Diagnostic Feedback: Using Set
Persistence to Compute
Intermediate Validation Hints
A key reason ScalableRAG outperforms state-
less retrieval agents is that every tool observa-
tion is enriched withautomatically computed
diagnosticsthat guide the agent’s next action.
We enumerate these for the primary tools.
apply_filter diagnostics.When the agent
partitions a target set Sby regexρ, the obser-
vation contains far more than just the resulting
set names and sizes. All statistics below are
computedrelative to S, so when the agent fil-
ters amdocument subset, it sees token hit
counts out of m, not out of the full corpus.
This relative scoping is what makes iterative
refinement informative:
•Token decomposition: for each token ( ≥4
chars) extracted from theregex string(via
[^\W_]{4,} ), the number of documents
in the target set whose normalized text
matches that token as a standalone, case-
insensitiveregexprobe. Thisrevealswhich
token is the bottleneck (too strict) or
which is vacuous (matches everything).
•Keyword intersection: the number of doc-
uments containingboththe top two key-
words extracted from ρsimultaneously, re-
vealing how much the combined pattern
over-constrains.
•Selectivity gap: when the keyword inter-
section is larger than the regex match
count, the system flags: “ kdocs likely use
a different format. Read negatives.” This
prompts the agent to investigate format
variants it is missing.
•Anchor-token coverage( entity_coverage
in logs): for the question’s primary anchor
token (see Appendix B), how many docu-
ments containing that token fall inside vs.
outside the current set.
•Negative anchor-token check: how many
negative (excluded) documents still con-
tainthetopquestion-derivedanchortoken,
alerting the agent to potential false nega-
tives.•Scope check: when all docs in a narrow
subsetmatch, thesystemcheckshowmany
docsoutsidethat subset also match the
same regex, preventing premature narrow-
ing.
•Positive and negative snippets: text ex-
cerpts from both sides of the partition,
centered on the regex match (positive) or
the top anchor token (negative), so the
agent can seewhya document was in-
cluded or excluded.
count_set diagnostics.Beyond the count
itself:
•Provenance warning: if the set was never
refined by a condition-specific regex, the
system warns “unrefined set—count may
include false positives.”
•Anchor-token coverage: how many docs
containing the top anchor token are out-
side the counted set.
•Filename range: the lexicographically first
and last filenames, showing the scope of
the set at a glance.
extract_field /extract_list diagnostics.
The observation reports: how many documents
were scanned, how many yielded a value, and
a sample of extracted values, letting the agent
immediately see whether its regex captures the
intended structure or misses most documents.
3.2 Limited-Ingestion ScalableRAG
The zero-ingestion system described above al-
ready achieves strong performance, see Section
4. Two optional modules, each requiring only
modest, one-time preprocessing, extend the
workspace with additional tools for further per-
formance boost improvement.
3.2.1 Module 1: Vector Embeddings
Pre-computing document embeddings adds
cosine_search (q,k)→Dto the workspace:
dense retrieval producing a named document
set from the documents that contain the top k
chunks.In our experiments we assume that the
budget for vector embedding is minimal, and
our chunks are as large as the encoder allows.
Additionally, the embeddings enable an
optional label_docs→train_classifier
pipeline: the agent can label a small sample of
5

Example: multi-hop question over
MuSiQue.“What county is Erik Hort’s
birthplace a part of?”
Step 1: apply_filter("Erik", all)
→has_Erik_pos(30 docs).
Step 2: apply_filter("Hort",
has_Erik_pos)→ 2 docs, including
Erik_Hort.txt. Observation snippets reveal
birthplace = Montebello, NY.
Step 3: apply_filter("Montebello",
all)→ 3 docs. Observation includes
Montebello_New_York.txt: “incorporated
village in... Rockland County.”
Answer:Rockland County. (3 tool calls,
correct.)
Figure 1: Set-algebraic multi-hop reasoning. Each
hop narrows the set, and the agent uses the inter-
mediate set as both evidence and anchor for the
next hop.
documents, train a lightweight classifier over
their embeddings (XGBoost by default; logis-
tic regression and random forest are also sup-
ported), and apply it corpus-wide. In practice,
the agent rarely invokes this pipeline, but it
provides a principled fallback for corpora where
relevant documents share no lexical signal.
3.2.2 Module 2: Automatic Pattern
Discovery
Overview.We distinguish three preprocess-
ing outputs surfaced to the agent:
Stage 1: Pattern Discovery.Determin-
istic regexes.A scanner identifies recurring
“Label: Value” lines at the start of a line (la-
bel up to 80 characters; value up to 300). We
normalize the label key (whitespace collapsing;
case-folding) and count document frequency.
A label becomes a pattern if it appears in at
least 3 documents with at least 2 distinct val-
ues. Each discovered field yields a regex tem-
plate anchored to the label and a capture group
(?P<value>...).
LLM-generated regexes.Independently, an
LLM reads a small, diverse sample (default 20
documents) and proposes multi-line patterns:
named sections that span multiple lines and
cannot be captured as a single “Label: Value”
field. Each pattern includes a name, descrip-tion, parameter list, and a parameterized regex
template with named capture groups.
Semantic patternsIn addition to regex pat-
terns, thepreprocessingstepdiscoverssemantic
patterns: prose-embedded concepts not cap-
tured by any regex. Each semantic pattern
includes a short description, an extraction hint,
and a small list of search keywords; we esti-
mate its coverage by keyword matching over
the corpus.
Each of the three types of patterns also gets
associated keywords, and get exposed to the
agent if these keywords are in the user ques-
tion. (For deterministic patterns it is simply
the label.)
Stage2: Validationandrefinement(LLM
patterns only).Each LLM-generated regex
is executed over the corpus in a sandbox with
a 2-second per-document timeout (guarding
against catastrophic backtracking). We esti-
mate precision from an LLM-judged sample of
matches, and recall from an LLM-judged sam-
ple of non-matches plus partial matches (regex
fired but capture groups empty). If either is
below threshold (defaults 0.90/0.85), the LLM
is asked to refine the regex using false-negative
evidence, iterating up to 8 rounds and keeping
the best-scoring template.
A curation step assigns each pattern (regexes
from the deterministic field scanner, LLM-
generated regexes, and the semantic patterns)
a 0–10 utility score of “question-answering util-
ity” via LLM judgment, keeping only patterns
scoring≥5.
Stage 3: Gap clusters.All patterns above
a minimum validation score (0.5 by default)
are indexed by executing their regex over each
document (using a 12,000-character head+tail
window). We then computegap clustersas fol-
lows: for each keyword token, find documents
where the token appears in the scannable text
butnotinside any captured group; extract the
linecontaining the token (truncated), lower-
case it, replace the token by {}, then cluster
identical templates and count them.
# (1) Deterministic labeled-field pattern (Transcripts;
field scanner)
name: field_present
regex_template: (?:^|[\n])[ \t]*Present\s*:\s*(?P<value>[^\
n]{1,300})
# (2) LLM-generated structural section (Hotels; multi-line
regex template)
name: facilities_amenities_section
6

Algorithm2Automaticpatterndiscoveryand
extraction indexing
Require: CorpusC, LLMM, sample size m(default
20), max refinement roundsR(default 8)
1:P field←DeterministicRegexes(C){determinis-
tic label patterns}
2:P llm←M (LLMGeneratedRegexes (Sample (C,m)))
{multi-line regex templates}
3:foreachp∈P llmdo
4:Validate(p,C)→(/hatwidestprec,/hatwiderrec)
5:forr= 1toRwhile below thresholdsdo
6:p←M (RefinePattern (p,FalseNegatives ))
7:Validate(p,C)
8:end for
9:end for
10:P←{p∈(P field∪Pllm) : LLMScore(p)≥0.5}
11:I←BuildIndex(P,C){matches, values, value
keywords}
12:ComputeGapClusters(I,C){surrounding-text
templates}
13:P sem←M (SemanticPatterns (P,Sample (C,m)))
{prose-embedded concepts + extraction hints}
14:P sem←EstimateCoverage(P sem,C){keyword-
based coverage estimates}
15:returnI⊕P sem
parameters: ["facilities_amenities_block"]
regex_template: (?:^|\n)##\s*Facilities\s*&\s*Amenities\n(?
P<facilities_amenities_block>(?:.{0,200}\n)+?)(?=\n
##\s*Pricing|\n#|\Z)
# (2b) Pattern usefulness in practice: answer in one tool
call (Hotels)
question: "How many hotel pages have a swimming pool
facility?"
tool: pattern_search
args: {"pattern": "facilities_amenities_section", "params":
{"facilities_amenities_block": "swimming"}, "
create_set": "has_swimming_pool"}
confirmed_matches: 24
capture_sample: "Swimming Pool\n\nFitness Center\n\nBar\n\
nSpa\n\nSauna\n\nConcierge ..."
# (3) Gap cluster returned at query time: keyword occurs
outside capture (FinanceBench)
question: "Which debt securities are registered to trade
on a national securities exchange under 3M's name as
of Q2 of 2023?"
tool: pattern_search
args: {"pattern": "Exhibit List/Table", "params": {"
content": "registered"}, "create_set": "
registered_exhibits"}
confirmed_matches: 6
coverage.keyword_in_pattern_not_captured: 71
coverage.gap_clusters[0]: {template: "securities {}
pursuant to section 12(b) of the act:", count: 54}
coverage.gap_clusters[1]: {template: "securities {}
pursuant to section 12(b) of the securities exchange
act of 1934:", count: 2}
Listing 1: Examples for extracted patterns and gap
clusters.
Agent Exposure to Patterns.A compact
corpus profileis prepended to the agent’s input:
a bulleted list of each indexed pattern with its
coverage percentage and sample values. The
profile also includes a compact summary of fre-
quently occurring labeled fields and semantic
patterns (as extraction hints). Additionally, apattern expert sectionin the question landscape
matches the question’s keywords against the
index’s value-keyword catalog, suggesting spe-
cific patterns. This “pattern routing” happens
before the first LLM calland costs zero tokens
of generation.
A new tool called pattern_search is ex-
posed for using regex-based patterns. LLM-
generated structural regex patterns are listed
fully; deterministic labeled-field regexes, of
which there can be hundreds, only show a
few based on prevalence, and the rest based
on whether the label appeared in the ques-
tion; LLM-generated non-regex semantic pat-
terns are shown only as extraction hints with
keyword-based coverage estimates, but not as
pattern_search targets. We also pre-register
named document sets pat_{name} for the cu-
rated regex patterns, so the agent can immedi-
ately intersect structural sets with other con-
straints.
4 Experiments
4.1 Experimental Setup
Datasets.We evaluate on six corpora:
MuSiQue (Trivedi et al., 2022), with 11,656
docs and 1,000 questions; 2WikiMultiHopQA
(Ho et al., 2020), with 6,118 docs and 1,000
questions; Transcripts (a new dataset we con-
struct from public U.S. Government Publishing
Office (GPO) records of the 117th Congress;1
see Appendix D for construction details), with
75 U.S. Congressional hearing transcripts and
100 questions; FinanceBench (Islam et al.,
2023), with 84 financial filing documents and
150 questions; ComplexTR (Tan et al., 2024),
with 872 documents and 200 temporal multi-
hop questions; and Hotels (Koshorek et al.,
2025) (only the train split), with 50 hotel page
documents and 138 questions.
4.2 Results
Zero-Ingestion ScalableRAG already beats
all other baselines, including the high cost
ingestion baselines (HippoRAG2, SRAG,
GraphRAG, and AutoSchemaKG) in 3 out of
the 6 datasets: MuSiQue, Transcripts, and
Hotels. We remark that Hotels was first in-
troduced in the SRAG paper (Koshorek et al.,
1Source documents from https://www.govinfo.
gov/app/collection/chrg/117.
7

System MuSiQue 2Wiki Transcripts FinanceBench ComplexTR Hotels All Datasets
Limited-Ingestion ScalableRAG58.91
33.70/49.9386.76
59.70/73.6983.50
31.00/59.4175.58
20.67/38.1075.57
60.00/75.5673.15
26.09/35.2875.58
38.53/55.33
Zero-Ingestion ScalableRAG58.09
34.00/48.6485.74
59.60/72.5582.00
23.00/52.7676.31
19.33/37.5578.84
60.00/77.3771.74
23.19/32.6075.45
36.52/53.58
HippoRAG250.43
27.40/42.4873.99
58.50/65.6976.00
23.00/53.5261.08
13.33/29.1081.34
68.50/82.2032.97
6.52/10.5462.63
32.88/47.26
SRAG3.33
1.40/2.4114.75
11.40/13.0074.00
26.00/41.0118.25
1.33/7.278.33
5.50/7.1135.51
16.67/17.4725.70
10.38/14.71
Vanilla RAG (top-100)38.49
21.30/31.3043.80
25.90/35.0371.00
19.00/47.2332.09
4.00/14.9677.59
64.50/77.8830.80
2.90/10.3548.96
22.93/36.12
A-RAG57.00
31.80/48.0487.05
61.10/74.6478.50
18.00/45.9777.41
18.00/34.8477.09
60.00/75.8831.52
7.25/12.1268.09
32.69/48.58
GraphRAG15.47
7.00/12.8735.55
30.00/34.6935.50
1.00/25.2223.54
4.00/10.4913.75
8.50/15.011.45
0.00/0.0020.88
8.42/16.38
AutoSchemaKGNaN
NaN/NanNaN
NaN/NanNaN
NaN/NanNaN
NaN/Nan80.40
67.50/82.4124.64
2.17/4.97NaN
NaN/Nan
Table 1: LLM-as-judge accuracy (top, %) and SQuAD-style exact match / token-overlap F1 (bottom, %;
EM/F1). All runs used GPT4.1 and run on a n1-standard-8 instance on GCP. NaN values represent runs
that either ran out of memory or took more than 24 hours. For each question, before the LLM-as-a-judge
and EM and F1 computation, there is first an LLM call (GPT4.1) to ensure that all reasoning is removed
from the answer. The LLM-as-a-judge then assigns a numeric score based on a list of rules for judging
different types of questions; we report the mean over questions. See Appendix E for a full account.
Dataset Zero-Ingestion ScalableRAG Limited-Ingestion ScalableRAG A-RAG
steps per Q / tokens per Q steps per Q / tokens per Q steps per Q / tokens per Q
MuSiQue 8.05 / 624 8.32 / 847 7.63 / 2,030
2Wiki 5.05 / 402 5.94 / 821 5.49 / 1,650
Transcripts 4.47 / 26,100 4.71 / 30,100 3.26 / 318,001
FinanceBench 8.07 / 26,137 6.69 / 27,516 4.87 / 147,828
ComplexTR 3.42 / 8,059 3.46 / 8,461 3.52 / 9,442
Hotels 5.99 / 1,359 5.89 / 5,237 4.59 / 4,064
Table 2: Steps and tokens per question for ScalableRAG and A-RAG
2025) as a dataset that is challenging for requir-
ing aggregative reasoning, yet we beat SRAG
by 36.23% with no ingestion whatsoever. On
the3remainingdatasets, 2Wiki, FinanceBench,
and ComplexTR; it had lost by an average of
only 1.63% compared to the best performing
baseline for each dataset. Only A-RAG wins
across all 3 of these datasets compared Zero-
Ingestion ScalableRAG. We remark that the
reason that A-RAG loses significantly in Hotels
and Transcripts specifically, despite having a
vector database at its disposal, is exactly be-
cause it does not write and read persistent sets
as ScalableRAG does, and therefore cannot an-
swer aggregative questions. Note also in Table
2 that the number of steps in ScalableRAG
is similar to A-RAG, but that A-RAG uses
significantly more tokens at inference.
The lift of Limited-Ingestion ScalableRAG
compared to Zero-Ingesetion ScalableRAG is
in exactly those datasets that have more inter-
esting structure: Hotels, and Transcripts. We
remark that in Limited-Ingestion ScalableRAG
we used most coarse chunking possible follow-
ing the philosophy of limited ingestion (us-
ingtext-embedding-3-small ); perhaps finergranularity such as in A-RAG would improve
results.
Vanilla RAG is using all-MiniLM-L6-v2
with 900 character windows and 50% overlap.
5 Conclusion
We presentedScalableRAG, a stateful agentic
RAG framework that replicates the aggrega-
tive reasoning of ingestion-heavy knowledge-
base methods at a fraction of their prepro-
cessing cost. Its two variants,Zero-Ingestion
(no preprocessing) andLimited-Ingestion(vec-
tor embeddings plus a one-time, sample-based
pattern discovery whose cost is constant in
the corpus size), share the same set-algebraic
workspace: every tool reads from and writes
back to a typed registry of named document
sets and value sets, and every observation is
enriched with diagnostics that reference the
workspace state. On six diverse corpora both
ScalableRAG variants beat all ingestion-heavy
and agentic baselines on average, with Limited-
Ingestion ScalableRAG providing an additional
lift on corpora with rich structure amenable to
pattern discovery.
8

6 Limitations
A core design choice of our algorithm is that
every set in the workspace is in one-to-one
correspondence with a subset of the set of doc-
uments. This allows for easy aggregations so
long as the primary key one aggregates by is it-
self in one-to-one correspondence with a subset
of the set of documents. We found in practice
that questions that violate this hypothesis are
rare in datasets for question-answering on a
corpus of documents; but we expect knowledge
graph approaches to perform better on such
questions.
7 Ethical Considerations
All corpora used in our experiments are
publicly available under permissive licenses,
with the exception of the Transcripts dataset.
The Transcripts dataset is constructed from
U.S. Government Publishing Office releases
(govinfo.gov ), which place the underlying
hearing transcripts in the public domain. Our
experiments do not involve human subjects,
and we are not publishing ScalableRAG as a
pre-trained model, and it therefore makes no
use of personally identifying information. As
with all LLM-based systems, the outputs de-
pend on the underlying language model; down-
stream users should treat answers as derived
evidence rather than authoritative claims. Any
artifacts generated by AI (Cursor/ChatGPT/-
Claude) as part of this paper’s code generation,
ideation, or paper editing had been under man-
ual supervision and verification by the authors.
References
Jiaxin Bai, Wei Fan, Qi Hu, Qing Zong, Chun-
yang Li, Hong Ting Tsang, Hongyu Luo, Yauwai
Yim, Haoyu Huang, Xiao Zhou, and 1 others.
2025. Autoschemakg: Autonomous knowledge
graph construction through dynamic schema in-
duction from web-scale corpora.arXiv preprint
arXiv:2505.23628.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and
Hervé Jégou. 2025. The faiss library.IEEE
Transactions on Big Data.
Mingxuan Du, Benfeng Xu, Chiwei Zhu, Shaohan
Wang, Pengyu Wang, Xiaorui Wang, and Zhen-
dong Mao. 2026. A-rag: Scaling agentic retrieval-augmented generation via hierarchical retrieval
interfaces.arXiv preprint arXiv:2602.03442.
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, AlexChao, ApurvaMody, StevenTruitt,
Dasha Metropolitansky, Robert Osazuwa Ness,
and Jonathan Larson. 2024. From local to global:
A graph rag approach to query-focused summa-
rization.arXiv preprint arXiv:2404.16130.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie
Callan. 2023. Precise zero-shot dense retrieval
without relevance labels. InProceedings of the
61st Annual Meeting of the Association for Com-
putational Linguistics (Volume 1: Long Papers),
pages 1762–1777.
Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro
Yasunaga, and Yu Su. 2024. Hipporag: Neuro-
biologically inspired long-term memory for large
language models.Advances in neural informa-
tion processing systems, 37:59532–59569.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian
Qi, Sizhe Zhou, and Yu Su. 2025. From rag
to memory: Non-parametric continual learn-
ing for large language models.arXiv preprint
arXiv:2502.14802.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sug-
awara, and Akiko Aizawa. 2020. Constructing
a multi-hop qa dataset for comprehensive eval-
uation of reasoning steps. InProceedings of the
28th International Conference on Computational
Linguistics, pages 6609–6625.
Yulong Hui, Chao Chen, Zhihang Fu, Yihao
Liu, Jieping Ye, and Huanchen Zhang. 2025.
Interact-rag: Reason and interact with the cor-
pus, beyond black-box retrieval.arXiv preprint
arXiv:2510.27566.
Pranab Islam, Anand Kannappan, Douwe Kiela,
Rebecca Qian, Nino Scherrer, and Bertie Vid-
gen. 2023. Financebench: A new benchmark
for financial question answering.arXiv preprint
arXiv:2311.11944.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing
Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. 2023. Active
retrieval augmented generation. InProceedings
of the 2023 conference on empirical methods in
natural language processing, pages 7969–7992.
Vladimir Karpukhin, Barlas Oguz, Sewon Min,
Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen-tau Yih. 2020. Dense passage
retrieval for open-domain question answering.
InProceedings of the 2020 conference on em-
pirical methods in natural language processing
(EMNLP), pages 6769–6781.
Omri Koshorek, Niv Granot, Aviv Alloni, Shahar
Admati, Roee Hendel, Ido Weiss, Alan Arazi,
Shay-Nitzan Cohen, and Yonatan Belinkov. 2025.
9

Structured rag for answering aggregative ques-
tions.arXiv preprint arXiv:2511.08505.
Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-
tau Yih, Tim Rocktäschel, and 1 others. 2020.
Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural informa-
tion processing systems, 33:9459–9474.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and
Siliang Tang. 2025. Graph retrieval-augmented
generation: A survey.ACM Transactions on
Information Systems, 44(2):1–52.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing retrieval-augmented large language mod-
els with iterative retrieval-generation synergy. In
Findings of the Association for Computational
Linguistics: EMNLP 2023, pages 9248–9274.
Qingyu Tan, Hwee Tou Ng, and Lidong Bing. 2024.
Towards robust temporal reasoning of large lan-
guage models via a multi-hop qa dataset and
pseudo-instruction tuning. InFindings of the
Association for Computational Linguistics: ACL
2024, pages 6272–6286.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2022. Musique:
Multihop questions via single-hop question com-
position.Transactions of the Association for
Computational Linguistics, 10:539–554.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2023. Interleav-
ing retrieval with chain-of-thought reasoning for
knowledge-intensivemulti-stepquestions. InPro-
ceedings of the 61st annual meeting of the asso-
ciation for computational linguistics (volume 1:
long papers), pages 10014–10037.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du,
Izhak Shafran, Karthik Narasimhan, and Yuan
Cao. 2022. React: Synergizing reasoning and
acting in language models.arXiv preprint
arXiv:2210.03629.
Weichao Zhou, Jiaxin Zhang, Hilaf Hasson, Anu
Singh, and Wenchao Li. 2024. Hyqe: Ranking
contexts with hypothetical query embeddings. In
Findings of the Association for Computational
Linguistics: EMNLP 2024, pages 13014–13032.
A System Prompt
The following is the complete system prompt
provided to the ScalableRAG agent, copied
verbatim from the source code. Sections
between «IF:X»and«ENDIF:X» are condi-
tionally included depending on which mod-
ules are enabled (embeddings, pattern_search,agg_tools). Template variables ( {n_docs} ,
{total_chars}, etc.) are filled at runtime.
You are a document retrieval agent. You have a corpus of {
n_docs} documents ({total_chars:,} characters).
Answer the question using ONLY evidence from the
documents. Never guess.
Context budget: ~{context_budget} tokens. Previous
observations may be redacted to save space –-your
thought is preserved, so always record key findings
in your thought.
==========================================================
=============
STEP 0 –-DECOMPOSE THE QUESTION
==========================================================
=============
Before doing ANYTHING, analyze the question for complexity:
•Does the question reference entities INDIRECTLY ("the X
who...", "the Y where...")? Each indirect reference
is a HOP –-a fact you must resolve before you can
answer.
•Count the hops. If there are N hops, decompose the
question into N+1 numbered sub-questions in your
FIRST thought.
•After resolving each sub-question, state: "Sub-Q K
resolved: [result]. Remaining: Sub-Q [K+1, ...]."
•Budget at least 2 tool calls per hop.
*** Your final answer must address the OUTERMOST question,
not an intermediate sub-question. If you resolved an
intermediate fact but the question asks about a
PROPERTY of that fact, you are NOT done –-keep
searching. ***
==========================================================
=============
APPROACH –-for each sub-question (or the whole question if
single-hop)
==========================================================
=============
1. START BROAD –-one keyword at a time. Your first
apply_filter MUST use a SINGLE entity keyword, never
a compound pattern with multiple terms. For
structural constraints (dates, categories, document
types), prefer search_filenames –-filenames encode
metadata reliably while text mentions can be
incidental. Combine sets with set_operation(intersect
). False positives are fine –-false negatives are
hard to recover.
2. EXPLORE from BOTH ENDS. read_docs from the beginning
AND end of your set (use offset=-3). Documents are
sorted by filename, so beginning/end often show
different document types. Note ALL format variants
you observe –-a general category may appear in
several distinct formats. Build your refinement to
cover ALL of them.
3. REFINE (mandatory for counting/aggregation). "Contains
keyword"̸="Satisfies the condition." Write a more
specific regex targeting the actual format you
observed –-anchor to structural markers (section
headings, field labels), not bare keywords. Apply
WITHIN your working set. If the negative set has docs
in a different format, build a separate filter and
UNION.
<<IF:agg_tools>>
4. NUMERICAL AGGREGATION. For questions asking for average,
sum, min, max, or total of a value: first narrow to
the exact relevant subset of docs (not "all" unless
the question is about all docs). Then read a few docs
to learn the exact format of the value. Then use
extract_field with a regex capture group to scan
EVERY doc in that subset –-never compute extrema from
a sample or from search results alone. Check the "
extracted" count matches your expectation. Finally,
aggregate with the right operation. For min/max
questions that ask about a PROPERTY of the extremal
doc, use create_set to get the doc, then read_docs.
For dates, use as_type="date" so min/max give
10

earliest/latest.
4b. LIST EXTRACTION. When a question asks "how many X per
doc", "who are the X", "list all X", or requires
collecting repeating items (names, line items,
participants, references) from each document, use
extract_list –-NOT extract_field. extract_field
returns ONE value per doc; extract_list returns ALL
matches as a list per doc. Read a few docs first to
learn the exact line format, write a regex with a
capture group that matches each item, then run
extract_list on the target set. Follow with aggregate
using a list operation: unique_values (deduplicated
list), count_unique (number of distinct items),
value_counts (frequency table), or group_by_value (
which docs share each value). Use these for cross-
document questions like "which X appear in both A and
B" or "how many unique X across all docs".
<<ENDIF:agg_tools>>
5. VERIFY & ANSWER. For counts: read 2-3 docs from your
NEGATIVE (excluded) set to verify your refinement
regex did not miss valid variants. If you find a
different format, build a second filter and UNION.
Only then count_set on the refined set (not the broad
set –-a broad count is an upper bound, never the
answer). For facts: quote exact wording from the
source. Include ALL relevant structured fields. Use
the document's own words –-do not paraphrase.
==========================================================
=============
TOOLS
==========================================================
=============
apply_filter –-Apply regex to a set. Creates _pos and _neg
sets. Text is section-normalized: lines within a
paragraph are joined (use .* to span within a section,
\n for section boundaries). The result includes a
keyword_decomposition showing how many docs match
each individual keyword –-use this to gauge
selectivity and find a better strategy when your
combined pattern is too strict.
{{"filter_type": "regex", "pattern": "term", "target_set
": "all", "output_prefix": "has_term"}}
{{"filter_type": "regex", "pattern": "precise", "
target_set": "broad_pos", "output_prefix": "refined
"}}
<<IF:embeddings>>
{{"filter_type": "classifier", "filter_id": "clf_1", "
target_set": "working", "output_prefix": "refined"}}
<<ENDIF:embeddings>>
explore_patterns –-Dry-run 2-5 regex patterns WITHOUT
creating sets. Returns match counts and samples.
Compare ideas cheaply.
{{"patterns": ["pat_a", "pat_b"], "target_set": "all"}}
set_operation –-Combine two named sets (intersect, union,
difference).
{{"operation": "intersect", "set_a": "A", "set_b": "B", "
result_name": "A_and_B"}}
read_docs –-Read documents by set or IDs. Use offset to
sample different positions. Set full_text=true for
the complete document (no excerpting) –-free, zero
LLM cost. Use chunk_idx to page through a large
document in contiguous chunks.
{{"set_name": "my_set", "max_docs": 5, "query": "keywords
"}}
{{"set_name": "my_set", "max_docs": 3, "offset": -3}}
{{"doc_ids": ["id"], "full_text": true}}
{{"doc_ids": ["id"], "chunk_idx": 0}}
# For huge docs, page through a contiguous slice:
{{"doc_ids": ["id"], "start_char": 0, "max_chars":
50000}}
find_in_doc –-Navigate inside ONE document by returning
char offsets. Two modes: (1) exact match: provide
pattern (+use_regex); (2) lexical passage ranking:
provide query to surface high-overlap windows. In
query mode, compare multiple matches using their
previews and hit_tokens –-don’t blindly take only the
top-scored window if it doesn’t contain the right
section/table. Use with read_docs(start_char=...,max_chars=...) to jump directly in huge docs.
{{"doc_id": "id", "pattern": "Balance Sheets", "
max_matches": 10}}
{{"doc_id": "id", "pattern": "cash and cash equivalents",
"case_sensitive": false}}
{{"doc_id": "id", "pattern": "Item\s+7\.?\s+Management",
"use_regex": true}}
{{"doc_id": "id", "query": "business segments net income
2022 Q2", "max_matches": 5}}
table_lookup –-Find a specific row in aligned/table-like
text and return the row + nearby header lines (often
containing years/columns), plus parsed numbers. Use
this to avoid mis-reading the wrong year/column in
statements.
{{"doc_id": "id", "row": "Total current liabilities", "
max_matches": 5}}
{{"doc_id": "id", "row": "Net property, plant, and
equipment", "use_regex": true}}
count_set –-Count docs in a set. Shows verification
snippets.
{{"set_name": "my_set"}}
search_filenames –-Find docs by filename pattern. Supports
target_set to restrict to an existing set. Combine
two filename searches with set_operation(intersect)
for precise counts.
{{"pattern": "value", "create_set": "matched"}}
{{"pattern": "value", "target_set": "existing_set", "
create_set": "refined"}}
create_regex –-Register a regex or have the LLM create one
from sample documents.
{{"pattern": "your_regex"}}
{{"condition": "what to detect", "sample_set_name": "
working"}}
<<IF:embeddings>>
label_docs –-LLM reads each doc and labels it 0/1 for
conditions.
{{"set_name": "working", "labels": ["condition"], "
max_docs": 20}}
train_classifier –-Train on embeddings + labels from
label_docs.
{{"label_batch_id": "label_0", "label_key": "condition"}}
cosine_search –-Semantic search. Creates a named set (
default "cosine_top_k", or specify create_set). Use
when keyword search returns 0 for a key entity. Read
from the created set, learn corpus structure, then
build regex filters for the final answer.
{{"query": "description", "top_k": 10, "create_set": "
my_cosine"}}
<<ENDIF:embeddings>>
<<IF:pattern_search>>
pattern_search –-Query a pre-computed extraction index.
Use simple single-keyword params (e.g. "Smith" not "
John.*Smith") for best recall. Scope with target_set
FIRST when the question constrains by year/category.
The result includes: (1) confirmed_matches –-keyword
inside the captured structural region; (2)
gap_clusters –-pre-computed structural templates
showing where the keyword appears OUTSIDE the
captured region, each with a template (keyword
replaced by {{}}) and a count. Compare each cluster
template to the capture_sample and to your question –-
read a few gap docs from the explore_set when a
cluster may be relevant to the answer.
{{"pattern": "name", "params": {{"key": "value"}}, "
target_set": "some_set", "create_set": "result"}}
<<ENDIF:pattern_search>>
<<IF:agg_tools>>
extract_field –-Extract ONE value from EACH doc using a
regex capture group. Stores results as a scalar value
set (doc→value). Use group=1 for the first capture
group or group="name" for a named group (?P<name
>...). as_type: "number" (default, parse as float), "
text" (raw string), or "date" (parse date→enables
min/max for earliest/latest). First read_docs to
learn the exact value format, then write a regex that
captures it reliably.
{{"pattern": "your_regex_with_(capture)", "target_set": "
filtered_set", "as_type": "number", "save_as": "
11

my_values"}}
{{"pattern": "(?:established|founded).*?(\w+ \d{{1,2}},?
\d{{4}})", "target_set": "all", "as_type": "date",
"save_as": "est_dates"}}
extract_list –-Extract ALL matches of a regex from EACH
doc as a list. Unlike extract_field (one value per
doc), this returns every match per doc. Use for
repeating items: names, line items, references,
participants, etc. Runs on raw document text (
preserves newlines). Follow with aggregate using list
operations (unique_values, count_unique,
value_counts, group_by_value).
{{"pattern": "^([A-Z][a-z]+ [A-Z][a-z]+)", "target_set":
"filtered_set", "save_as": "names"}}
{{"pattern": "Item\s+(\d+)", "target_set": "all", "group
": 1, "save_as": "item_numbers"}}
aggregate –-Compute statistics over a named value set. For
scalar sets (from extract_field): sum, avg, min, max,
count. For min/max, returns the matching doc(s) and
optionally creates a doc set for further inspection.
For list sets (from extract_list): unique_values (
deduplicated list), count_unique (number of distinct
items), value_counts (frequency table), flatten (all
items), group_by_value (which docs share each value –-
useful for cross-doc overlap questions),
per_doc_count (total items per doc→scalar value set
), per_doc_count_unique (unique items per doc→
scalar value set). per_doc_count_unique is essential
for "average unique X per doc" questions: first
extract_list, then aggregate(per_doc_count_unique,
create_set="counts"), then aggregate(value_set="
counts", operation="avg").
{{"value_set": "my_values", "operation": "avg"}}
{{"value_set": "my_values", "operation": "min", "
create_set": "lowest_docs"}}
{{"value_set": "names", "operation": "unique_values"}}
{{"value_set": "names", "operation": "
per_doc_count_unique", "create_set": "name_counts"}}
extract_from_filename –-Extract metadata from each
document's FILENAME using a regex capture group.
Unlike extract_field (which runs on document text),
this runs on the filename string. Use for metadata
encoded in filenames: company tickers, dates,
quarters, categories. Stores results as a scalar
value set (doc→captured string).
{{"pattern": "^([A-Z]+)_", "target_set": "all", "save_as
": "company_ticker"}}
{{"pattern": "(\d{{4}}-Q\d)", "target_set": "all", "
save_as": "quarter"}}
filter_values –-Filter docs by comparing their extracted
value to a threshold. Takes a value set from
extract_field, an operator (<, <=, >, >=, ==, !=),
and a threshold. Creates a new doc set containing
only the docs that pass. For dates, the threshold can
be a date string (e.g. "2020-01-01").
{{"value_set": "ratings", "operator": "<", "threshold":
8.23, "create_set": "low_rated"}}
{{"value_set": "est_dates", "operator": ">=", "threshold
": "2020-01-01", "create_set": "recent"}}
<<ENDIF:agg_tools>>
calculate –-Evaluate math expressions safely. Supports +,
-, *, /, //, %, ** (including fractional exponents
like x**0.5), parentheses, and functions: sqrt, cbrt,
log, log2, log10, ln, exp, abs, round, ceil, floor,
pow, min, max, sum, factorial, gcd. Constants: pi, e.
Use for ANY arithmetic –-never do mental math. For
multi-step calculations, pass a list of named
expressions; each step can reference earlier step
names as variables.
{{"expr": "1234.5 / 67.8"}}
{{"expr": "(revenue - cost) / revenue * 100", "variables
": {{"revenue": 5400, "cost": 3200}}}}
{{"expressions": [{{"name": "margin", "expr": "revenue -
cost"}}, {{"name": "pct", "expr": "margin / revenue
* 100"}}], "variables": {{"revenue": 5400, "cost":
3200}}}}
list_sets<<IF:embeddings>> / list_classifiers<<ENDIF:
embeddings>> –-Inspect current state.
=======================================================================
GUARDRAILS
==========================================================
=============
•"Contains X"̸="Is about X." Always refine before
answering.
•Validate from different positions –-beginning, middle,
and end.
•Watch alternation: "A|B:.*X" means "A" OR "B:.*X". Use
"(?:A|B:).*X" for what you probably intend.
•Different format̸=different answer. UNION format variants
.
•Record findings in your thought –-observations get
redacted.
•SCOPE FIDELITY. Do not narrow beyond what the question
asks. If the question uses a general term, your
answer must cover the full category, not a single sub-
type.
•BE VERY SKEPTICAL OF ZERO RESULTS. 0 matches usually
means your pattern is wrong, not the data. Try:
shorter substrings, different spellings, broader
regex<<IF:embeddings>>, cosine_search with a natural
language description<<ENDIF:embeddings>>. Exhaust at
least 3 different strategies before concluding
information is absent.
•NEVER STOP AT AN INTERMEDIATE FINDING. If you resolved "A
is associated with B" but the question asks about a
property OF B, you must keep searching. Re-read your
sub-question decomposition.
•INSUFFICIENT_EVIDENCE is a claim that requires evidence.
You may only conclude "insufficient evidence" if you
searched thoroughly and can state what you searched
for and why nothing matched. If you have partial
findings, give your best answer and state what is
uncertain.
•COUNTING: count_set gives an EXACT doc count. Answer "how
many" with a single integer from count_set. NEVER
say "at least N" or "approximately N" –-hedging is
always wrong. If uncertain about filter quality,
refine the set and re-count instead of hedging.
<<IF:pattern_search>>
•CONFIRMED vs. GAP: pattern_search reports
confirmed_matches (keyword inside the captured
structural region) and gap_clusters (keyword OUTSIDE
it, in a different structural position). The captured
region defines what the pattern was designed to
extract. A gap cluster with a DIFFERENT surrounding-
text pattern than the capture_sample means the
keyword serves a different function there –-do NOT
count those docs. Only read gap docs (from the
explore_set) when a gap cluster template is genuinely
ambiguous and you cannot determine from the template
alone whether it satisfies the question. The
confirmed count is high-confidence; gap additions
need clear evidence of equivalence.
<<ENDIF:pattern_search>>
•CORPUS ONLY. Your pre-trained knowledge may conflict with
this corpus. An abbreviation, name, or term may have
a corpus-specific meaning that differs from the real
world. Always derive meaning FROM the documents,
never from memory.
•read_docs FOCUS. When you need a specific section of a
large document (e.g., a clause, a date, a specific
term), pass query="the specific terms" to focus
excerpts on what you need. If excerpts still miss the
section, page contiguously with chunk_idx (coarse)
or start_char/max_chars (precise). Use find_in_doc to
get offsets for start_char: use pattern/use_regex
when you know the exact phrase/section title; if you
only have a natural-language description, use
find_in_doc(query=...) to surface likely windows (
compare previews + hit_tokens), then page
contiguously with start_char/max_chars. Without query,
excerpts show the filter-match context.
•{max_steps} tool calls. Typical: 5-15 steps.
==========================================================
=============
SELF-CHECK BEFORE ANSWERING
==========================================================
=============
□Did I resolve ALL sub-questions, or am I answering an
intermediate hop?
□Is my answer based on text I actually read (read_docs),
not just keyword counts or snippet previews?
12

□For counting: is my count a single integer from count_set
–-not hedged with "at least" or "approximately"?
<<IF:agg_tools>>
□For aggregation (avg/sum/min/max): did I use
extract_field + aggregate –-not manual addition or
estimation?
□For list/enumeration questions: did I use extract_list +
aggregate(unique_values/count_unique) –-not manual
counting?
<<ENDIF:agg_tools>>
□For any arithmetic (division, percentages, ratios,
differences): did I use calculate –-not mental math?
□Percent vs ratio: ONLY multiply by 100 or add a % sign if
the question explicitly asks for a percentage /
percent / %. Otherwise, report ratios as decimals (e.
g., 0.83), not percent (83%).
□Does my answer satisfy EVERY constraint in the original
question?
□Am I using the CORPUS meaning of terms, not my own world
knowledge?
==========================================================
=============
RESPONSE FORMAT
==========================================================
=============
Each turn: exactly ONE JSON object, nothing else.
Tool call:
{{"thought": "what I plan to do and why", "tool": "name",
"args": {{...}}}}
Final answer:
{{"thought": "summary of evidence", "answer": "precise
answer"}}
B Context Management Details
Question landscape.Before the agent’s
first turn, we extract all maximal alphanumeric
tokens of length≥4from the question ("key-
words"). For the first five keywords (in order
of appearance), the system scans the full cor-
pus via case-insensitive substring match and
records: (i) the number of documents whose
body text contains the keyword, and (ii) the
number of filenames containing it. For four key-
words that match at least1but less than all
of the documents (sorted by ascending match
count), we additionally compute and report
pairwise intersection sizes, i.e., the number of
documents containing both keywords simulta-
neously. A 600-character snippet from the top
keyword’s first match is appended, showing
the agent one formatted example. The entire
landscape is prepended to the user message.
Tokenization and matching rules (for
disambiguation).The code uses similar-
looking but distinct lexical operations in dif-
ferent places; we list them explicitly to avoid
ambiguity.
•Question landscape tokens (“key-
words”).Tokens are extracted from
the question via [^\W_]{4,} . A tokenis counted as present in a document
if it matches by case-insensitive sub-
string inclusion ( tok.lower() in
text.lower() ). These counts provide a
cheap prior only.
•Anchor tokens (“entities”).The sys-
tem sorts a subset the keywords with
0<count (tok)< N, preferring tokens
that appear capitalized in the original
question and then by increasing document
frequency. These anchor tokens are used
only for entity-aware diagnostics (scope
checks; negative-entity checks), not as a
learned NER component.
•apply_filter decomposition tokens.
For diagnostics, tokens are extracted from
theregex stringwith the same tokenizer
[^\W_]{4,} . Eachtokenisthenprobedas
a single-token case-insensitive regex over
normalized text to estimate per-token se-
lectivity and token intersections.
•Extraction-index value tokens (gap
clusters).During indexing, tokens
are extracted from captured values via
[^\W\d_]{3,} (letter-only, length ≥3),
lowercased, and used to pre-compute
keyword_gaps : cases where a value to-
ken appears in a document but outside
the captured region. pattern_search sur-
faces these as gap clusters (surrounding-
text templates with the token replaced by
{}).
Context management overview.The
agent’s context window must accommodate the
system prompt, the question, thequestion land-
scape, and a growing sequence of tool observa-
tions and agent responses. Because individual
documents can be hundreds of thousands of
characters and trajectories can span 25 steps,
two mechanisms prevent overflow while pre-
serving the set-algebraic state that the agent
needs for planning.
Per-tool output budgets.Every tool ob-
servation is hard-capped at Btoolcharacters
(set to 200,000 for models with ≥500K context,
proportionally less for smaller models). For
read_docs specifically, the budget is further
dividedevenlyamongtherequesteddocuments:
when reading kdocuments, each gets ⌊Bcall/k⌋
characters, where Bcall=min(Cmax/6,Btool).
13

Documents exceeding their per-doc budget are
presented viakeyword-guided excerpting: the
system identifies text windows around query-
relevant terms and concatenates only those
passages. For long-document navigation with-
out relying on excerpting, the agent can also
page contiguously (chunked reads) or request
explicitstart_char/max_charsslices.
Observation redaction.After the agent re-
sponds to an observation (i.e., the observation
is “consumed"), the full text is replaced by a
structural summary. The summary retains a
whitelisted set of JSON keys:
•Set identity: positive_set ,
negative_set , result_set ,
created_set.
•Cardinalities: positive_count ,
negative_count ,total_filtered ,
count.
•Diagnostics: keyword_decomposition
(per-term match counts),
entity_coverage ,selectivity_gap ,
zero_result_hint ,scope_check ,
negative_entity_check.
•Extraction state: value_set ,extracted ,
scanned,n_values,matching_docs.
Everything else (raw document text, full
match listings) is discarded. The agent’s own
thought field (which the system prompt in-
structs it to use as a scratchpad) preserves
factual findings across turns.
Last-resort compression.If the total con-
versation still exceeds the budget B= 0.6×
Cmaxcharacters after redaction, messages in
the middle of the conversation (excluding the
system prompt, the initial question, and the
four most recent messages) are further trun-
cated to their first 150 characters. If still over
budget, middle messages are dropped entirely.
The net effect is that theworkspace state
(the names, sizes, provenance, and diagnostic
metadata of all sets) is always visible to the
agent, while raw document content is available
only for the most recent observation and must
be captured in the agent’s thought before it
becomes redacted.C Trace Examples
The following are representative agent traces
copied verbatim from logs. They illustrate how
diagnostic feedback drives the agent’s reason-
ing.
C.1 Aggregation: Transcripts Corpus
This trace illustrates ScalableRAG answering
a corpus-wide aggregation using two filename
filters and a difference computation.
Question: How many more hearings did one chamber hold than
the other in this corpus?
Golden answer: 5.
Agent answer: The House held 5 more hearings than the
Senate in this corpus.
Step 0: search_filenames({
"pattern": "hhrg",
"create_set": "house_hearings"
})
Agent thought: To answer how many more hearings one
chamber held than the other, I need to (1) identify
which hearings were held by each chamber (House vs
Senate), (2) count the number of hearings for each,
and (3) compute the difference. The best way to
distinguish chambers is by filename (e.g.,'hhrg'
for House,'shrg'for Senate), as this is a
standard convention in congressional hearing
transcripts. I will first search filenames for'
hhrg'(House) and'shrg'(Senate), count each, and
then calculate the difference.
Observation: {
"count": 39,
"searched": 75,
"matches": [
{
"doc_id": "CHRG-117hhrg44411__396ab7bdb162",
"filename": "CHRG-117hhrg44411.txt",
"chars": 87030
},
{
"doc_id": "CHRG-117hhrg47493__dadb2f96711a",
"filename": "CHRG-117hhrg47493.txt",
"chars": 180244
},
{
"doc_id": "CHRG-117hhrg47530__281e009dab4d",
"filename": "CHRG-117hhrg47530.txt",
"chars": 85073
},
{
"doc_id": "CHRG-117hhrg47569__dcfab7dcae2c",
"filename": "CHRG-117hhrg47569.txt",
"chars": 445619
},
{
"doc_id": "CHRG-117hhrg47691__58319a4c1cfd",
"filename": "CHRG-117hhrg47691.txt",
"chars": 168012
},
{
"doc_id": "CHRG-117hhrg47737__9026617797e9",
"filename": "CHRG-117hhrg47737.txt",
"chars": 65819
},
{
"doc_id": "CHRG-117hhrg47840__2ee4adedbbb6",
"filename": "CHRG-117hhrg47840.txt",
"chars": 96050
},
{
"doc_id": "CHRG-117hhrg47882__e93645a43e27",
"filename": "CHRG-117hhrg47882.txt",
"chars": 84991
},
{
"doc_id": "CHRG-117hhrg47929__d773214aa378",
"filename": "CHRG-117hhrg47929.txt",
"chars": 114632
14

},
{
"doc_id": "CHRG-117hhrg48013__a4aeb68a5f5f",
"filename": "CHRG-117hhrg48013.txt",
"chars": 78597
},
{
"note": "... (24 more) ..."
},
{
"doc_id": "CHRG-117hhrg59789__4c57831bcc67",
"filename": "CHRG-117hhrg59789.txt",
"chars": 150905
},
{
"doc_id": "CHRG-117hhrg60497__4d86153ef05b",
"filename": "CHRG-117hhrg60497.txt",
"chars": 96107
},
{
"doc_id": "CHRG-117hhrg61455__fa67ee74f3ce",
"filename": "CHRG-117hhrg61455.txt",
"chars": 158615
},
{
"doc_id": "CHRG-117hhrg61990__8cbf8eef8f8b",
"filename": "CHRG-117hhrg61990.txt",
"chars": 159716
},
{
"doc_id": "CHRG-117hhrg62185__7d975bb7125d",
"filename": "CHRG-117hhrg62185.txt",
"chars": 159174
}
],
"set_created": "house_hearings",
"not_matching": 36,
"filename_range": "CHRG-117hhrg44411.txt ... CHRG-117
hhrg62185.txt"
}
[Auto-sampled snippets (house_hearings) –-3 of 39 docs,
explore further if needed]
[CHRG-117hhrg44411.txt] - [H.A.S.C. No. 117-12]
DEPARTMENT OF DEFENSE ELECTROMAGNETIC SPECTRUM
OPERATIONS: CHALLENGES AND OPPORTUNITIES IN THE
INVISIBLE BATTLESPACE
[House Hearing, 117 Congress]
[From the U.S. Government Publishing Office]
.
[H.A.S.C. No. 117-12]
DEPARTMENT OF DEFENSE
ELECTROMAGNETIC SPECTRUM
OPERATIONS: CHALLENGES AND
OPPORTUNITIES IN THE
INVISIBLE BATTLESPACE
__________
HEARING
BEFORE THE
SUBCOMMITTEE ON CYBER, INNOVATIVE
TECHNOLOGIES, AND INFORMATION SYSTEMS
OF THE
COMMITTEE ON ARMED SERVICES
HOUSE OF REPRESENTATIVES
ONE HUNDRED SEVENTEENTH CONGRESS
FIRST SESSION
__________
HEARING HELD
MARCH 19, 2021[GRAPHIC NOT AVAILABLE IN TIFF FORMAT]
__________
U.S. GOVERNMENT PUBLISHING OFFICE
44-411 WASHINGTON : 2021
----------------------------------...
[CHRG-117hhrg48949.txt] ...rmer and the founder of Crack
Open the Door and Get Clemency Now. I have
contributed to nearly a dozen individuals being
released early from prison, nine of which were
serving life without parole. I am an author who has
written a book called Clemency Now, Get Clemency
Now, which teaches people incarcerated and their
families who to put together powerful clemency
petitions. I am the 2017 Black Chamber of Commerce
Trailblazer and Community Civic Leader of the Year.
I am the 2018 McKinney Volunteer of the Year. I am
the 2021 Leadership McKinney Alumnus of the Year.
I'm also the 2022 NACDL's Champion of Justice of
the Year. I am also the founder and Executive
Director of At Last, which is a leadership program
for Latino high school students which teaches them
how to become leaders in their sc...
[CHRG-117hhrg62185.txt] - NO TIME TO WASTE: SOLUTIONS FOR
AMERICA'S BROKEN RECYCLING SYSTEM
[House Hearing, 117 Congress]
[From the U.S. Government Publishing Office]
NO TIME TO WASTE: SOLUTIONS FOR AMERICA'S
BROKEN RECYCLING SYSTEM
=====================================
HYBIRD HEARING
BEFORE THE
SUBCOMMITTEE ON ENVIRONMENT AND CLIMATE
CHANGE
OF THE
COMMITTEE ON ENERGY AND COMMERCE
HOUSE OF REPRESENTATIVES
ONE HUNDRED SEVENTEENTH CONGRESS
SECOND SESSION
__________
JUNE 30, 2022
__________
Serial No. 117-92
[GRAPHIC NOT AVAILABLE IN TIFF FORMAT]
Published for the use of the Committee on Energy and
Commerce
govinfo.gov/committee/house-energy
energycommerce.house.gov
__________
U.S. GOVERNMENT PUBLISHING OFFICE
62-185 PDF WASHINGTON : 2026
-...
Step 1: search_filenames({
"pattern": "shrg",
"create_set": "senate_hearings"
})
Agent thought: I have identified 39 House hearings by
searching for'hhrg'in filenames. Now, I will
search for'shrg'in filenames to identify Senate
hearings, count them, and then calculate the
difference between the two chambers.
Observation: {
"count": 34,
"searched": 75,
"matches": [
15

{
"doc_id": "CHRG-117shrg19104915__a98345dc64a0",
"filename": "CHRG-117shrg19104915.txt",
"chars": 824153
},
{
"doc_id": "CHRG-117shrg39104013__e0f425d86cc4",
"filename": "CHRG-117shrg39104013.txt",
"chars": 256863
},
{
"doc_id": "CHRG-117shrg39104020__c38508a0247b",
"filename": "CHRG-117shrg39104020.txt",
"chars": 161017
},
{
"doc_id": "CHRG-117shrg39104988__ee0e35c33612",
"filename": "CHRG-117shrg39104988.txt",
"chars": 149038
},
{
"doc_id": "CHRG-117shrg39104991__6a10d5809ff3",
"filename": "CHRG-117shrg39104991.txt",
"chars": 174014
},
{
"doc_id": "CHRG-117shrg47818__daec5e8ab066",
"filename": "CHRG-117shrg47818.txt",
"chars": 117084
},
{
"doc_id": "CHRG-117shrg47983__c74f8b4a35f8",
"filename": "CHRG-117shrg47983.txt",
"chars": 110745
},
{
"doc_id": "CHRG-117shrg48137__6073c99ce3d1",
"filename": "CHRG-117shrg48137.txt",
"chars": 586346
},
{
"doc_id": "CHRG-117shrg48287__522b12a1d919",
"filename": "CHRG-117shrg48287.txt",
"chars": 126257
},
{
"doc_id": "CHRG-117shrg48907__23b1cd53b6a3",
"filename": "CHRG-117shrg48907.txt",
"chars": 183568
},
{
"note": "... (19 more) ..."
},
{
"doc_id": "CHRG-117shrg54763__e3aec0d0e287",
"filename": "CHRG-117shrg54763.txt",
"chars": 309122
},
{
"doc_id": "CHRG-117shrg55777__06ae1ae00374",
"filename": "CHRG-117shrg55777.txt",
"chars": 154286
},
{
"doc_id": "CHRG-117shrg55818__63675610a12d",
"filename": "CHRG-117shrg55818.txt",
"chars": 130758
},
{
"doc_id": "CHRG-117shrg59763__055014dfc194",
"filename": "CHRG-117shrg59763.txt",
"chars": 169810
},
{
"doc_id": "CHRG-117shrg59770__0237ff3bf0d9",
"filename": "CHRG-117shrg59770.txt",
"chars": 158404
}
],
"set_created": "senate_hearings",
"not_matching": 41,
"filename_range": "CHRG-117shrg19104915.txt ... CHRG-117
shrg59770.txt"
}
[Auto-sampled snippets (senate_hearings) –-3 of 34 docs,
explore further if needed]
[CHRG-117shrg19104915.txt] - NONDEPARTMENTAL WITNESSES[Senate Hearing 117-]
[From the U.S. Government Publishing Office]
STATE, FOREIGN OPERATIONS, AND RELATED PROGRAMS
APPROPRIATIONS FOR
FISCAL YEAR 2022
----------
U.S. Senate,
Subcommittee of the Committee on Appropriations,
Washington, DC.
NONDEPARTMENTAL WITNESSES
[Clerk's note.--The subcommittee was unable to hold
hearings on nondepartmental witnesses. The statements and
letters of those submitting written testimony are as
follows:]
Prepared Statement of the Accountability Counsel
Dear Chairman Coons, Ranking Member Graham, and members of
the
subcommittee:
On behalf of Accountability Counsel, thank you for this
opportunity
to provide input on the FY 2022 State, Foreign Operations,
and Related
Programs (SFOPs) appropriations process. I...
[CHRG-117shrg51394.txt] - FARM BILL 2023: RESEARCH
PROGRAMS
[Senate Hearing 117-613]
[From the U.S. Government Publishing Office]
S. Hrg. 117-613
FARM BILL 2023: RESEARCH PROGRAMS
=====================================
HEARING
BEFORE THE
COMMITTEE ON AGRICULTURE,
NUTRITION, AND FORESTRY
UNITED STATES SENATE
ONE HUNDRED SEVENTEENTH CONGRESS
SECOND SESSION
__________
December 6, 2022
__________
Printed for the use of the
Committee on Agriculture, Nutrition, and Forestry
[GRAPHIC NOT AVAILABLE IN TIFF FORMAT]
Available on http://www.govinfo.gov/
__________
U.S. GOVERNMENT PUBLISHING OFFICE
51-394 PDF WASHINGTON : 2024
-------------------------------------
COMMITTEE ON AGRICULTURE, NUTRITIO...
[CHRG-117shrg59770.txt] - WORLDWIDE THREATS
[Senate Hearing 117-960]
[From the U.S. Government Publishing Office]
S. Hrg. 117-960
WORLDWIDE THREATS
16

=====================================
HEARING
BEFORE THE
COMMITTEE ON ARMED SERVICES
UNITED STATES SENATE
ONE HUNDRED SEVENTEENTH CONGRESS
SECOND SESSION
__________
May 10, 2022
__________
Printed for the use of the Committee on Armed Services
[GRAPHC NOT AVAILABLE IN TIFF FORMAT]
Available via: http: //www.govinfo.gov
__________
U.S. GOVERNMENT PUBLISHING OFFICE
59-770 PDF WASHINGTON : 2025
-------------------------------------
COMMITTEE ON ARMED SERVICES
JACK REED, Rhode Island, Chairman JAMES M. INHOFE,
Oklahoma
JEANNE SHAHEEN, New Hampshire...
===Automatically computed ===
Set compositions –-house_hearings (39)×senate_hearings
(34):
intersect→0
union→73
house_hearings \ senate_hearings→39
senate_hearings \ house_hearings→34
Step 2: calculate({
"expr": "39 - 34"
})
Agent thought: I have identified 39 House hearings and 34
Senate hearings by filename pattern. The
difference is 39 - 34 = 5. I will now calculate
this difference to answer how many more hearings
one chamber held than the other.
Observation: {
"name": "result",
"expr": "39 - 34",
"value": 5,
"display": "5"
}
C.2 Multi-Hop with Diagnostic
Recovery: 2WikiMultiHopQA
This trace illustrates a two-hop question solved
by constructing and comparing named doc-
ument sets. The agent starts with a broad
lexical filter, then refines by intersecting con-
straints (song title, composer mention), and
uses diagnostics such as scope_check and
negative_entity_check plus auto-sampled
snippets to confirm the correct entity before
extracting the birthplace.
Question: Where was the composer of song Back In The U.S.A.
born?
Golden answer: St. Louis, MissouriAgent answer: St. Louis, Missouri
Step 0: apply_filter({
"filter_type": "regex",
"pattern": "composer",
"target_set": "all",
"output_prefix": "has_composer"
})
Agent thought: The question is: Where was the composer of
the song'Back In The U.S.A.'born? This is a two-
hop question: (1) Who is the composer of'Back In
The U.S.A.'? (2) Where was that composer born? I
will first search for documents mentioning both'
composer'and'Back'to identify the composer, then
search for the composer's birthplace.
Observation: {
"positive_set": "has_composer_pos",
"positive_count": 116,
"negative_set": "has_composer_neg",
"negative_count": 6002,
"total_filtered": 6118,
"sample_positive_docs": [
{
"doc_id": "A_Time_of_Destiny__1b00ec88a396",
"filename": "A_Time_of_Destiny.txt"
},
{
"doc_id": "Abd\u00fclaziz__6cfec66ec8bc",
"filename": "Abd\u00fclaziz.txt"
},
{
"doc_id": "Keith_Emerson__f2d70a357c06",
"filename": "Keith_Emerson.txt"
},
{
"doc_id": "Kurt_Schr\u00f6der__4830d105778e",
"filename": "Kurt_Schr\u00f6der.txt"
},
{
"doc_id": "Youve_Got_Possibilities__e9a4993a16da",
"filename": "Youve_Got_Possibilities.txt"
},
{
"doc_id": "\u00c1ngel_Recasens__6497d74babaa",
"filename": "\u00c1ngel_Recasens.txt"
}
],
"sample_negative_docs": [
{
"doc_id": "11_Harrowhouse__08cc6c95c2f9",
"filename": "11_Harrowhouse.txt"
},
{
"doc_id": "1971_Copa_Libertadores__8369d1b96cbe",
"filename": "1971_Copa_Libertadores.txt"
},
{
"doc_id": "Khalid_Abdel_Nasser__b9f61f804f4b",
"filename": "Khalid_Abdel_Nasser.txt"
},
{
"doc_id": "Khalid_al-Habib__ca209876aaad",
"filename": "Khalid_al-Habib.txt"
},
{
"doc_id": "\u021aerova__10c94b229856",
"filename": "\u021aerova.txt"
},
{
"doc_id": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku__3e99f95742e5",
"filename": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku.txt"
}
],
"scope_check": "apply_filter searches DOCUMENT TEXT. 116
docs matched in text, but 4 filenames match the
same pattern. If you intended to filter by filename
/date, use search_filenames instead.",
"negative_entity_check": "139/6002 negatives contain'
Back'(keyword only)."
}
[Auto-sampled snippets (positive) –-3 of 116 docs, explore
further if needed]
[A_Time_of_Destiny.txt] A Time of Destiny
A Time of Destiny is a 1988 American drama film directed
17

by Gregory Nava and written by Nava and Anna Thomas.
The story is based on the opera" La forza del destino
" by Giuseppe Verdi. The motion picture was executive
produced by Shep Gordon and Carolyn Pfeiffer. It
features original music by veteran composer Ennio
Morricone. Set during World War II in Italy and San
Diego, the film tells of two friends who become
enemies during the war.
[Kurt_Schröder.txt] Kurt Schröder
Kurt Schröder( 1888–1962) was a German composer. Schröder
composed a number of film scores. During the 1930s he
worked in Britain for Alexander Korda's London Film
Productions, and scored the company's breakthrough
hit" The Private Life of Henry VIII" in 1933.
[Ángel_Recasens.txt] Ángel Recasens
Ángel Recasens( 4 March 1938 in Cambrils –2 August 2007)
was a Catalan organist, teacher, composer and
musicologist best known as a choral conductor. From
1975 to 1986, he was director of the Coro de Sant
Esteve of Vila- seca. There, he performed music from
the romantics Schumann and Mendelssohn, to the
contemporary music of Ligeti and Schnittke. He also
directed the Coral Verge del Camí in his home town
Cambrils. In his later years, he dedicated himself to
reviving lost Spanish choral works from the 16th to
18th centuries; first with la Capilla Príncipe de
Viana, then with the expanded La Grande Chapelle and
his own label Lauda Records. Both the ensemble and
the label were continued by his son Albert Recasens.
[Auto-sampled snippets (negative) –-3 of 6002 docs,
explore further if needed]
[11_Harrowhouse.txt] 11 Harrowhouse
11 Harrowhouse is a 1974 British DeLuxe Color film
directed by Aram Avakian and starring Charles Grodin,
Candice Bergen, James Mason, Trevor Howard and John
Gielgud in Panavision. It was adapted by Charles
Grodin based upon the novel by Gerald A. Browne with
the screenplay by Jeffrey Bloom.
[Khalid_al-Habib.txt] Khalid al-Habib
Khalid al- Habib also believed to use the nom de guerre
Khalid al Harbi is an Al- Qaeda leader who has
overseen Al- Qaeda operations in Afghanistan. In 2005
it was reported that al- Habib died in a US missile
strike on the Pakistani village of Damadola though it
is unclear whether this is the case as Pakistani
officials have claimed that no Al- Qaeda leaders died
in the attack. al- Habib's nationality is not clear
with sources claiming he may be Egyptian or Moroccan.
[’Elisiva_Fusipala_Tauki’onetuku.txt] ’Elisiva Fusipala
Tauki’onetuku
’Elisiva Fusipala Tauki’onetuku (18 May 1850 –September
1889) was the mother of King George Tupou II.
===Automatically computed ===
Filename search counts:
search_filenames("composer")→4 docs
Step 1: apply_filter({
"filter_type": "regex",
"pattern": "Back",
"target_set": "has_composer_pos",
"output_prefix": "composer_and_Back"
})
Agent thought: Now I need to narrow to docs that mention
both'composer'and'Back'to find the composer of
'Back In The U.S.A.'. This will help identify the
composer before searching for their birthplace.
Observation: {
"positive_set": "composer_and_Back_pos",
"positive_count": 7,
"negative_set": "composer_and_Back_neg",
"negative_count": 109,
"total_filtered": 116,
"sample_positive_docs": [
{
"doc_id": "Deva_composer__371eabd0fe49",
"filename": "Deva_composer.txt"
},
{
"doc_id": "Johann_Sebastian_Bach__d44cb4c567c7",
"filename": "Johann_Sebastian_Bach.txt"
},{
"doc_id": "Mario_Bonnard__be124b2927fc",
"filename": "Mario_Bonnard.txt"
},
{
"doc_id": "Michael_Corcoran_musician__f988e03f8c59",
"filename": "Michael_Corcoran_musician.txt"
},
{
"doc_id": "Raffy_Shart__4b2c2178b7ca",
"filename": "Raffy_Shart.txt"
},
{
"doc_id": "Youve_Got_Possibilities__e9a4993a16da",
"filename": "Youve_Got_Possibilities.txt"
}
],
"sample_negative_docs": [
{
"doc_id": "A_Time_of_Destiny__1b00ec88a396",
"filename": "A_Time_of_Destiny.txt"
},
{
"doc_id": "Abd\u00fclaziz__6cfec66ec8bc",
"filename": "Abd\u00fclaziz.txt"
},
{
"doc_id": "Karel_Zich__efc220f3e985",
"filename": "Karel_Zich.txt"
},
{
"doc_id": "Karl_Wilhelm_conductor__fdea54ab7eab",
"filename": "Karl_Wilhelm_conductor.txt"
},
{
"doc_id": "William_Axt__f2e5b44b3b94",
"filename": "William_Axt.txt"
},
{
"doc_id": "\u00c1ngel_Recasens__6497d74babaa",
"filename": "\u00c1ngel_Recasens.txt"
}
],
"scope_check": "apply_filter searches DOCUMENT TEXT. 7
docs matched in text, but 0 filenames match the
same pattern. If you intended to filter by filename
/date, use search_filenames instead."
}
[Auto-sampled snippets (positive) –-3 of 7 docs, explore
further if needed]
[Deva_composer.txt] Deva (composer)
Devanesan Chokkalingam, popularly known as Deva, is an
Indian film composer and singer. He has composed
songs and provided background music for Tamil, Telugu,
Malayalam and Kannada films in a career spanning
about 20 years. Many know his gaana songs, written
mostly using Madras Tamil. He is known as the "Father
of Gaana Genre" in the Tamil film industry. Deva has
composed music for many films. He debuted as a film
music director in the film "Manasukkeththa Maharaasa"
in 1989. In the intervening years he has composed
music for a total of more than 400 films.
[Michael_Corcoran_musician.txt] Michael Corcoran (
musician)
Michael Thomas Corcoran( born December 9, 1972), known
professionally as Backhouse Mike or Ken Lofkoll, is
an American musician, singer- songwriter, composer,
sound designer, musical arranger, orchestrator, and
record producer.
[Youve_Got_Possibilities.txt] ...e pry you from your
shell ... You've got possibilities ... you do n't
even know you've got"). " It's a Bird ... It's a
Plane ... It's Superman" was not a big hit, but"
You've Got Possibilities", generally considered the
show's most memorable tune, became something of a
cabaret standard. Peggy Lee recorded the song on
her 1966 album" Big$ pender" and released it as the
B-side of the single" Come Back To Me". Joanie
Sommers released" You've Got Possibilities" as the
B-side of her single" Never Throw Your Dreams Away",
also in 1966, while Carol Ventura released it also
in 1966 as an A- side single. Linda Lavin included
the song on her 2011 album" Possibilities"( she
had earlier sung it on the 1966 original cast album
for" It's a Bird ... It's a Plane ... It's
18

Superman"). Matt Monro's version...
[Auto-sampled snippets (negative) –-3 of 109 docs, explore
further if needed]
[A_Time_of_Destiny.txt] A Time of Destiny
A Time of Destiny is a 1988 American drama film directed
by Gregory Nava and written by Nava and Anna Thomas.
The story is based on the opera" La forza del destino
" by Giuseppe Verdi. The motion picture was executive
produced by Shep Gordon and Carolyn Pfeiffer. It
features original music by veteran composer Ennio
Morricone. Set during World War II in Italy and San
Diego, the film tells of two friends who become
enemies during the war.
[Karl_Wilhelm_conductor.txt] Karl Wilhelm (conductor)
Karl Wilhelm, also Carl Wilhelm( 5 September 1815,
Schmalkalden –26 August 1873, Schmalkalden) was a
German choral director. He is best known as the
composer of the music of the song “Die Wacht am Rhein.
”
[Ángel_Recasens.txt] Ángel Recasens
Ángel Recasens( 4 March 1938 in Cambrils –2 August 2007)
was a Catalan organist, teacher, composer and
musicologist best known as a choral conductor. From
1975 to 1986, he was director of the Coro de Sant
Esteve of Vila- seca. There, he performed music from
the romantics Schumann and Mendelssohn, to the
contemporary music of Ligeti and Schnittke. He also
directed the Coral Verge del Camí in his home town
Cambrils. In his later years, he dedicated himself to
reviving lost Spanish choral works from the 16th to
18th centuries; first with la Capilla Príncipe de
Viana, then with the expanded La Grande Chapelle and
his own label Lauda Records. Both the ensemble and
the label were continued by his son Albert Recasens.
[Context audit –-'Back'in 5 of 7 positive docs:]
...ger. He has composed songs and provided background
music for Tamil, Telugu, Malayala...
...many in films directed by Luis Trenker. Back in Italy
in 1932, he became a prolific...
...ember 9, 1972), known professionally as Backhouse Mike
or Ken Lofkoll, is an America...
...king at the Überbrettl cabaret. He went back to Vienna
and began writing operettas,...
...ed it as the B-side of the single" Come Back To Me".
Joanie Sommers released" You've...
===Automatically computed ===
Filename search counts:
search_filenames("Back")→11 docs
Set compositions –-has_composer_pos (116)×
composer_and_Back_pos (7):
intersect→7
union→116
has_composer_pos \ composer_and_Back_pos→109
composer_and_Back_pos \ has_composer_pos→0
Step 2: apply_filter({
"filter_type": "regex",
"pattern": "Back In The U\\.S\\.A\\.",
"target_set": "all",
"output_prefix": "has_BackInUSA"
})
Agent thought: None of the 7 docs in
composer_and_Back_pos appear to be about the song'
Back In The U.S.A.'–-the matches are for'Backhouse
Mike','Come Back To Me', and other unrelated uses
of'Back'. I need to try a broader search for the
song title'Back In The U.S.A.'to identify the
composer directly. I will search for'Back In The U.
S.A.'in all docs.
Observation: {
"positive_set": "has_BackInUSA_pos",
"positive_count": 1,
"negative_set": "has_BackInUSA_neg",
"negative_count": 6117,
"total_filtered": 6118,
"sample_positive_docs": [
{
"doc_id": "Back_in_the_USA__a505370ce874",
"filename": "Back_in_the_USA.txt"
}
],
"sample_negative_docs": [
{"doc_id": "11_Harrowhouse__08cc6c95c2f9",
"filename": "11_Harrowhouse.txt"
},
{
"doc_id": "1971_Copa_Libertadores__8369d1b96cbe",
"filename": "1971_Copa_Libertadores.txt"
},
{
"doc_id": "Khalid_Abdel_Nasser__b9f61f804f4b",
"filename": "Khalid_Abdel_Nasser.txt"
},
{
"doc_id": "Khalid_al-Habib__ca209876aaad",
"filename": "Khalid_al-Habib.txt"
},
{
"doc_id": "\u021aerova__10c94b229856",
"filename": "\u021aerova.txt"
},
{
"doc_id": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku__3e99f95742e5",
"filename": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku.txt"
}
],
"scope_check": "apply_filter searches DOCUMENT TEXT. 1
docs matched in text, but 0 filenames match the
same pattern. If you intended to filter by filename
/date, use search_filenames instead.",
"negative_entity_check": "145/6117 negatives contain'
Back'(keyword only)."
}
[Auto-sampled snippets (positive) –-3 of 1 docs, explore
further if needed]
[Back_in_the_USA.txt] Back in the U.S.A.
"Back in the U.S.A." is a song written by Chuck Berry that
was released in 1959 and was a top 40 hit. A cover
version in 1978 by Linda Ronstadt was also a hit.
[Auto-sampled snippets (negative) –-3 of 6117 docs,
explore further if needed]
[11_Harrowhouse.txt] 11 Harrowhouse
11 Harrowhouse is a 1974 British DeLuxe Color film
directed by Aram Avakian and starring Charles Grodin,
Candice Bergen, James Mason, Trevor Howard and John
Gielgud in Panavision. It was adapted by Charles
Grodin based upon the novel by Gerald A. Browne with
the screenplay by Jeffrey Bloom.
[Khalid_al-Habib.txt] Khalid al-Habib
Khalid al- Habib also believed to use the nom de guerre
Khalid al Harbi is an Al- Qaeda leader who has
overseen Al- Qaeda operations in Afghanistan. In 2005
it was reported that al- Habib died in a US missile
strike on the Pakistani village of Damadola though it
is unclear whether this is the case as Pakistani
officials have claimed that no Al- Qaeda leaders died
in the attack. al- Habib's nationality is not clear
with sources claiming he may be Egyptian or Moroccan.
[’Elisiva_Fusipala_Tauki’onetuku.txt] ’Elisiva Fusipala
Tauki’onetuku
’Elisiva Fusipala Tauki’onetuku (18 May 1850 –September
1889) was the mother of King George Tupou II.
===Automatically computed ===
Filename search counts:
search_filenames("Back")→11 docs
Set compositions –-composer_and_Back_pos (7)×
has_BackInUSA_pos (1):
intersect→0
union→8
composer_and_Back_pos \ has_BackInUSA_pos→7
has_BackInUSA_pos \ composer_and_Back_pos→1
Step 3: apply_filter({
"filter_type": "regex",
"pattern": "Chuck Berry",
"target_set": "all",
"output_prefix": "has_ChuckBerry"
})
Agent thought: The document'Back_in_the_USA.txt'states:
'Back in the U.S.A.'is a song written by Chuck
Berry. This answers sub-question 1: the composer is
Chuck Berry. Now, I need to find where Chuck Berry
19

was born. I will search for'Chuck Berry'and'
born'to find his birthplace.
Observation: {
"positive_set": "has_ChuckBerry_pos",
"positive_count": 3,
"negative_set": "has_ChuckBerry_neg",
"negative_count": 6115,
"total_filtered": 6118,
"sample_positive_docs": [
{
"doc_id": "Alan_Reeves_composer__89c0d8e6240e",
"filename": "Alan_Reeves_composer.txt"
},
{
"doc_id": "Back_in_the_USA__a505370ce874",
"filename": "Back_in_the_USA.txt"
},
{
"doc_id": "Chuck_Berry__ff7ec8ef49a2",
"filename": "Chuck_Berry.txt"
}
],
"sample_negative_docs": [
{
"doc_id": "11_Harrowhouse__08cc6c95c2f9",
"filename": "11_Harrowhouse.txt"
},
{
"doc_id": "1971_Copa_Libertadores__8369d1b96cbe",
"filename": "1971_Copa_Libertadores.txt"
},
{
"doc_id": "Khalid_al-Habib__ca209876aaad",
"filename": "Khalid_al-Habib.txt"
},
{
"doc_id": "
Khalid_bin_Abdullah_bin_Abdulaziz_Al_Saud__
b63dd78ee252",
"filename": "
Khalid_bin_Abdullah_bin_Abdulaziz_Al_Saud.txt"
},
{
"doc_id": "\u021aerova__10c94b229856",
"filename": "\u021aerova.txt"
},
{
"doc_id": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku__3e99f95742e5",
"filename": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku.txt"
}
],
"scope_check": "apply_filter searches DOCUMENT TEXT. 3
docs matched in text, but 0 filenames match the
same pattern. If you intended to filter by filename
/date, use search_filenames instead.",
"keyword_decomposition": [
{
"keyword": "Chuck",
"matches": 10,
"pct": 0.2
},
{
"keyword": "Berry",
"matches": 21,
"pct": 0.3
}
],
"selectivity_note": "Combined pattern matched 3;
individual keywords match 10\u201321 each. Consider
starting with one keyword.",
"keyword_intersection": {
"keywords": [
"Chuck",
"Berry"
],
"both_match": 3
},
"negative_entity_check": "144/6115 negatives contain'
Back'(keyword only)."
}
[Auto-sampled snippets (positive) –-3 of 3 docs, explore
further if needed]
[Alan_Reeves_composer.txt] Alan Reeves (composer)
Alan David Reeves is a British film composer, musicproducer, and Hammond B3 virtuoso. In the course of
his career he has received 35 international awards,
including a Goldene Schallplatte. He became known for
his work with the bands The Showtimers and Clinic as
well as for the music for the films To Walk with
Lions,, and Kill Bill Vol. 2. He has appeared played
or recorded with/ for among others, the Rolling
Stones, Jimi Hendrix, Chuck Berry and David Bowie and
David Gilmore
[Back_in_the_USA.txt] Back in the U.S.A.
"Back in the U.S.A." is a song written by Chuck Berry that
was released in 1959 and was a top 40 hit. A cover
version in 1978 by Linda Ronstadt was also a hit.
[Chuck_Berry.txt] Chuck Berry
Charles Edward Anderson Berry (October 18, 1926 –March 18,
2017) was an American singer and songwriter, and one
of the pioneers of rock and roll music. Nicknamed
the "Father of Rock and Roll", Berry refined and
developed rhythm and blues into the major elements
that made rock and roll distinctive with songs such
as "Maybellene" (1955), " Roll Over Beethoven" (1956),
"Rock and Roll Music" (1957) and "Johnny B. Goode"
(1958). Writing lyrics that focused on teen life and
consumerism, and developing a music style that
included guitar solos and showmanship, Berry was a
major influence on subsequent rock music. Born into a
middle-class African-American family in St. Louis,
Missouri, Berry had an interest in music from an
early age and gave his first public performance at
Sumner Hig...
[Auto-sampled snippets (negative) –-3 of 6115 docs,
explore further if needed]
[11_Harrowhouse.txt] 11 Harrowhouse
11 Harrowhouse is a 1974 British DeLuxe Color film
directed by Aram Avakian and starring Charles Grodin,
Candice Bergen, James Mason, Trevor Howard and John
Gielgud in Panavision. It was adapted by Charles
Grodin based upon the novel by Gerald A. Browne with
the screenplay by Jeffrey Bloom.
[Khalid_bin_Abdullah_bin_Abdulaziz_Al_Saud.txt] Khalid
bin Abdullah bin Abdulaziz Al Saud
Khalid bin Abdullah( born 1950, Riyadh) is the eldest
child of King Abdullah of Saudi Arabia. He is a
member of the Allegiance Council.
[’Elisiva_Fusipala_Tauki’onetuku.txt] ’Elisiva Fusipala
Tauki’onetuku
’Elisiva Fusipala Tauki’onetuku (18 May 1850 –September
1889) was the mother of King George Tupou II.
[Context audit –-'Back'in 2 of 3 positive docs:]
Back in the U.S.A. "Back in the U.S.A." is...
...ormer, playing his past hits with local backup bands
of variable quality. However, i...
===Automatically computed ===
Filename search counts:
search_filenames("Chuck")→2 docs
search_filenames("Berry")→3 docs
Set compositions –-has_BackInUSA_pos (1)×
has_ChuckBerry_pos (3):
intersect→1
union→3
has_BackInUSA_pos \ has_ChuckBerry_pos→0
has_ChuckBerry_pos \ has_BackInUSA_pos→2
C.3 Zero-Result Diagnostics and
Recovery: 2WikiMultiHopQA
Here the agent’s initial detention-related
regexes yield zero results within its already-
narrowedworkingset. The zero_result_hint
diagnosticpromptsittotrybroaderlexicalvari-
ants (e.g., detain→arrest→prison|jail ),
and the agent recovers without widening the
search back to the full corpus.
20

Question: Where was the performer of song B Boy (Song)
detained?
Golden answer: State Correctional Institution –Chester
Agent answer: State Correctional Institution –Chester in
Chester, Pennsylvania.
Step 0: apply_filter({
"filter_type": "regex",
"pattern": "B Boy",
"target_set": "all",
"output_prefix": "has_b_boy"
})
Agent thought: This is a multi-hop question: (1) Identify
the performer of'B Boy (Song)'; (2) Determine
where that performer was detained. I will start by
searching for'B Boy'to identify the performer,
then search for detention information about that
person.
Observation: {
"positive_set": "has_b_boy_pos",
"positive_count": 1,
"negative_set": "has_b_boy_neg",
"negative_count": 6117,
"total_filtered": 6118,
"sample_positive_docs": [
{
"doc_id": "B_Boy_song__638a840ec141",
"filename": "B_Boy_song.txt"
}
],
"sample_negative_docs": [
{
"doc_id": "11_Harrowhouse__08cc6c95c2f9",
"filename": "11_Harrowhouse.txt"
},
{
"doc_id": "1971_Copa_Libertadores__8369d1b96cbe",
"filename": "1971_Copa_Libertadores.txt"
},
{
"doc_id": "Khalid_Abdel_Nasser__b9f61f804f4b",
"filename": "Khalid_Abdel_Nasser.txt"
},
{
"doc_id": "Khalid_al-Habib__ca209876aaad",
"filename": "Khalid_al-Habib.txt"
},
{
"doc_id": "\u021aerova__10c94b229856",
"filename": "\u021aerova.txt"
},
{
"doc_id": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku__3e99f95742e5",
"filename": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku.txt"
}
],
"scope_check": "apply_filter searches DOCUMENT TEXT. 1
docs matched in text, but 0 filenames match the
same pattern. If you intended to filter by filename
/date, use search_filenames instead.",
"negative_entity_check": "279/6117 negatives contain'
Where'(keyword only)."
}
[Auto-sampled snippets (positive) –-3 of 1 docs, explore
further if needed]
[B_Boy_song.txt] B Boy (song)
"B Boy" is a song by American hip hop recording artist
Meek Mill. It was released as a third single from "
Dreams Worth More Than Money" on January 3, 2015, by
Maybach Music Group and Atlantic Records. The song,
which was produced by Sap, features guest appearances
from Big Sean and ASAP Ferg.
[Auto-sampled snippets (negative) –-3 of 6117 docs,
explore further if needed]
[11_Harrowhouse.txt] 11 Harrowhouse
11 Harrowhouse is a 1974 British DeLuxe Color film
directed by Aram Avakian and starring Charles Grodin,
Candice Bergen, James Mason, Trevor Howard and John
Gielgud in Panavision. It was adapted by Charles
Grodin based upon the novel by Gerald A. Browne with
the screenplay by Jeffrey Bloom.
[Khalid_al-Habib.txt] Khalid al-HabibKhalid al- Habib also believed to use the nom de guerre
Khalid al Harbi is an Al- Qaeda leader who has
overseen Al- Qaeda operations in Afghanistan. In 2005
it was reported that al- Habib died in a US missile
strike on the Pakistani village of Damadola though it
is unclear whether this is the case as Pakistani
officials have claimed that no Al- Qaeda leaders died
in the attack. al- Habib's nationality is not clear
with sources claiming he may be Egyptian or Moroccan.
[’Elisiva_Fusipala_Tauki’onetuku.txt] ’Elisiva Fusipala
Tauki’onetuku
’Elisiva Fusipala Tauki’onetuku (18 May 1850 –September
1889) was the mother of King George Tupou II.
Step 1: apply_filter({
"filter_type": "regex",
"pattern": "Meek Mill",
"target_set": "all",
"output_prefix": "has_meek_mill"
})
Agent thought: The performer of'B Boy (song)'is Meek
Mill, with guest appearances from Big Sean and ASAP
Ferg. The question asks where the performer was
detained. I will now search for information about
Meek Mill being detained, focusing on location. I
will start by searching for'Meek Mill'in the
corpus.
Observation: {
"positive_set": "has_meek_mill_pos",
"positive_count": 2,
"negative_set": "has_meek_mill_neg",
"negative_count": 6116,
"total_filtered": 6118,
"sample_positive_docs": [
{
"doc_id": "B_Boy_song__638a840ec141",
"filename": "B_Boy_song.txt"
},
{
"doc_id": "Meek_Mill__52e01b469e2a",
"filename": "Meek_Mill.txt"
}
],
"sample_negative_docs": [
{
"doc_id": "11_Harrowhouse__08cc6c95c2f9",
"filename": "11_Harrowhouse.txt"
},
{
"doc_id": "1971_Copa_Libertadores__8369d1b96cbe",
"filename": "1971_Copa_Libertadores.txt"
},
{
"doc_id": "Khalid_Abdel_Nasser__b9f61f804f4b",
"filename": "Khalid_Abdel_Nasser.txt"
},
{
"doc_id": "Khalid_al-Habib__ca209876aaad",
"filename": "Khalid_al-Habib.txt"
},
{
"doc_id": "\u021aerova__10c94b229856",
"filename": "\u021aerova.txt"
},
{
"doc_id": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku__3e99f95742e5",
"filename": "\u02bbElisiva_Fusipala_Tauki\
u02bbonetuku.txt"
}
],
"scope_check": "apply_filter searches DOCUMENT TEXT. 2
docs matched in text, but 0 filenames match the
same pattern. If you intended to filter by filename
/date, use search_filenames instead.",
"keyword_decomposition": [
{
"keyword": "Meek",
"matches": 3,
"pct": 0.0
},
{
"keyword": "Mill",
"matches": 157,
"pct": 2.6
}
21

],
"selectivity_note": "Combined pattern matched 2;
individual keywords match 3\u2013157 each. Consider
starting with one keyword.",
"keyword_intersection": {
"keywords": [
"Meek",
"Mill"
],
"both_match": 2
},
"negative_entity_check": "279/6116 negatives contain'
Where'(keyword only)."
}
[Auto-sampled snippets (positive) –-3 of 2 docs, explore
further if needed]
[B_Boy_song.txt] B Boy (song)
"B Boy" is a song by American hip hop recording artist
Meek Mill. It was released as a third single from "
Dreams Worth More Than Money" on January 3, 2015, by
Maybach Music Group and Atlantic Records. The song,
which was produced by Sap, features guest appearances
from Big Sean and ASAP Ferg.
[Meek_Mill.txt] Meek Mill
Robert Rihmeek Williams (born May 6, 1987), known
professionally as Meek Mill, is an American rapper,
songwriter, and activist. Born and raised in
Philadelphia, he embarked on his music career as a
battle rapper, and later formed a short-lived rap
group, The Bloodhoundz. In 2008, Atlanta-based rapper
T.I. signed Meek Mill to his first record deal. In
February 2011, after leaving Grand Hustle Records,
Mill signed with Miami-based rapper Rick Ross's
Maybach Music Group (MMG). Mill's debut album, "
Dreams and Nightmares", was released in 2012 under
MMG and Warner Bros. Records. The album, preceded by
the lead single "Amen" (featuring Drake), debuted at
number two on the U.S. "Billboard" 200. In October
2012, Mill announced the launch of his own label
imprint, Dream Chasers Records,...
[Auto-sampled snippets (negative) –-3 of 6116 docs,
explore further if needed]
[11_Harrowhouse.txt] 11 Harrowhouse
11 Harrowhouse is a 1974 British DeLuxe Color film
directed by Aram Avakian and starring Charles Grodin,
Candice Bergen, James Mason, Trevor Howard and John
Gielgud in Panavision. It was adapted by Charles
Grodin based upon the novel by Gerald A. Browne with
the screenplay by Jeffrey Bloom.
[Khalid_al-Habib.txt] Khalid al-Habib
Khalid al- Habib also believed to use the nom de guerre
Khalid al Harbi is an Al- Qaeda leader who has
overseen Al- Qaeda operations in Afghanistan. In 2005
it was reported that al- Habib died in a US missile
strike on the Pakistani village of Damadola though it
is unclear whether this is the case as Pakistani
officials have claimed that no Al- Qaeda leaders died
in the attack. al- Habib's nationality is not clear
with sources claiming he may be Egyptian or Moroccan.
[’Elisiva_Fusipala_Tauki’onetuku.txt] ’Elisiva Fusipala
Tauki’onetuku
’Elisiva Fusipala Tauki’onetuku (18 May 1850 –September
1889) was the mother of King George Tupou II.
===Automatically computed ===
Filename search counts:
search_filenames("Meek")→1 docs
search_filenames("Mill")→12 docs
Set compositions –-has_b_boy_pos (1)×has_meek_mill_pos
(2):
intersect→1
union→2
has_b_boy_pos \ has_meek_mill_pos→0
has_meek_mill_pos \ has_b_boy_pos→1
Step 2: apply_filter({
"filter_type": "regex",
"pattern": "detain",
"target_set": "has_meek_mill_pos",
"output_prefix": "meek_mill_detain"
})
Agent thought: The performer of'B Boy (song)'is MeekMill, as confirmed in the B_Boy_song.txt and
Meek_Mill.txt documents. Now, I need to determine
where Meek Mill was detained. I will search for the
term'detain'(to cover'detained','detainment',
etc.) in the Meek_Mill.txt document to find any
mention of his detention and its location.
Observation: {
"positive_set": "meek_mill_detain_pos",
"positive_count": 0,
"negative_set": "meek_mill_detain_neg",
"negative_count": 2,
"total_filtered": 2,
"sample_positive_docs": [],
"sample_negative_docs": [
{
"doc_id": "B_Boy_song__638a840ec141",
"filename": "B_Boy_song.txt"
},
{
"doc_id": "Meek_Mill__52e01b469e2a",
"filename": "Meek_Mill.txt"
}
],
"zero_result_hint": "Zero matches on 2 docs. Try: (1) a
simpler/shorter regex, (2) search_filenames if the
condition is about document names."
}
===Automatically computed ===
Set compositions –-has_meek_mill_pos (2)×
meek_mill_detain_pos (0):
intersect→0
union→2
has_meek_mill_pos \ meek_mill_detain_pos→2
meek_mill_detain_pos \ has_meek_mill_pos→0
Step 3: apply_filter({
"filter_type": "regex",
"pattern": "arrest",
"target_set": "has_meek_mill_pos",
"output_prefix": "meek_mill_arrest"
})
Agent thought: No matches for'detain'or its variants in
the Meek Mill-related documents. To ensure
coverage, I will try searching for'arrest'(a
common synonym for'detain') in the same set, as
the corpus may use different terminology.
Observation: {
"positive_set": "meek_mill_arrest_pos",
"positive_count": 0,
"negative_set": "meek_mill_arrest_neg",
"negative_count": 2,
"total_filtered": 2,
"sample_positive_docs": [],
"sample_negative_docs": [
{
"doc_id": "B_Boy_song__638a840ec141",
"filename": "B_Boy_song.txt"
},
{
"doc_id": "Meek_Mill__52e01b469e2a",
"filename": "Meek_Mill.txt"
}
],
"zero_result_hint": "Zero matches on 2 docs. Try: (1) a
simpler/shorter regex, (2) search_filenames if the
condition is about document names."
}
Step 4: apply_filter({
"filter_type": "regex",
"pattern": "prison|jail",
"target_set": "has_meek_mill_pos",
"output_prefix": "meek_mill_prison_jail"
})
Agent thought: No matches for'arrest'or'detain'in the
Meek Mill documents. To ensure I am not missing
alternate terminology, I will search for'prison'
and'jail'in the Meek Mill-related documents, as
these are common terms used to describe where
someone was detained.
Observation: {
"positive_set": "meek_mill_prison_jail_pos",
"positive_count": 1,
"negative_set": "meek_mill_prison_jail_neg",
"negative_count": 1,
"total_filtered": 2,
22

"sample_positive_docs": [
{
"doc_id": "Meek_Mill__52e01b469e2a",
"filename": "Meek_Mill.txt"
}
],
"sample_negative_docs": [
{
"doc_id": "B_Boy_song__638a840ec141",
"filename": "B_Boy_song.txt"
}
],
"scope_check": "apply_filter searches DOCUMENT TEXT. 1
docs matched in text, but 0 filenames match the
same pattern. If you intended to filter by filename
/date, use search_filenames instead.",
"keyword_decomposition": [
{
"keyword": "prison",
"matches": 1,
"pct": 50.0
},
{
"keyword": "jail",
"matches": 0,
"pct": 0.0
}
]
}
[Auto-sampled snippets (positive) –-3 of 1 docs, explore
further if needed]
[Meek_Mill.txt] ...and "Ima Boss", being included on
volume one. He released his second album, "Dreams
Worth More Than Money", in 2015 and his third album,
"Wins & Losses", in 2016. His fourth studio album,
"Championships", was released in November 2018 and
debuted at number one on the "Billboard" 200. Its
lead single, "Going Bad" (featuring Drake), peaked
at number 6 on the "Billboard" Hot 100. In November
2017, he was sentenced to two to four years in
prison for violating parole and served five months
at the State Correctional Institution –Chester in
Chester, Pennsylvania, before being released while
his trial continues. In August 2019, a documentary
series about his battle with the criminal justice
system, "Free Meek", was released on Amazon's Prime
Video. Mill served as executive producer on the
series.
[Auto-sampled snippets (negative) –-3 of 1 docs, explore
further if needed]
[B_Boy_song.txt] B Boy (song)
"B Boy" is a song by American hip hop recording artist
Meek Mill. It was released as a third single from "
Dreams Worth More Than Money" on January 3, 2015, by
Maybach Music Group and Atlantic Records. The song,
which was produced by Sap, features guest appearances
from Big Sean and ASAP Ferg.
===Automatically computed ===
Filename search counts:
search_filenames("prison")→3 docs
Set compositions –-meek_mill_arrest_pos (0)×
meek_mill_prison_jail_pos (1):
intersect→0
union→1
meek_mill_arrest_pos \ meek_mill_prison_jail_pos→0
meek_mill_prison_jail_pos \ meek_mill_arrest_pos→1
DTranscripts Dataset Construction
This appendix describes the construction of the
Transcripts corpus and question set used in our
experiments (Section 4). The goal is to pro-
duce questions where (i) all gold answers are
derivable from the corpus text, (ii) a substan-
tial fraction of questions requirecorpus-wide
aggregationthat a top- kretrieval system struc-
turally cannot answer, and (iii) the questions
are phrased as a domain analyst would natu-rally phrase them, with no leading references
to formatting features.
Source data.The corpus is drawn from the
U.S. Government Publishing Office (GPO) col-
lection of 117th-Congress hearing transcripts,
distributed publicly via govinfo.gov . We
sample 75 transcripts, balanced across the
House (39), the Senate (34), and joint House–
Senate sessions (2), and spanning calendar
2021–2022. Each transcript is converted to
UTF-8 plain text from the GPO HTML release;
document filenames follow GPO’s package-id
convention: CHRG-117hhrg....txt for House
hearings, CHRG-117shrg....txt for Senate
hearings, and CHRG-117jhrg....txt for joint
House-Senate sessions (chamber is encoded by
thehhrg/shrg/jhrgsubstring).
Structural features used as gold.CHRG
transcripts have a strongly conventional struc-
turethatmakesdeterministicgoldconstruction
tractable. We rely on two families of features:
1.Cover-page metadata.The first
∼160 lines of each transcript contain
the chamber identifier (e.g., U.S. HOUSE
OF REPRESENTATIVES ,UNITED STATES
SENATE, or aJOINT HEARING marker), the
main committee heading ( COMMITTEE ON
...), an optional subcommittee heading
(SUBCOMMITTEE ON ... ), the hearing
date in the form Weekday, Month Day,
Year, and the hearing title. A parser
extracts these four fields per document.
2.Labeled-field markers in the body.
CHRG transcripts use a small set of
conventional line-anchored labels that
recur across hearings: Present: and
Members present: (formal attendance
roll);Staff Present: ;Also present:
(visiting members); Available via
the World Wide Web: (publication
URL); the closing parliamentary no-
tation [Whereupon, at HH:MM ...] ;
andResponses to written questions
of ... (post-hearing QFR section
header). Each marker is detected by a
line-anchored regex over the full tran-
script text; coverage counts (e.g., 50/75
transcripts contain an attendance roll
underPresent:∪Members present: ;
23

26/75 adjourned during the 11 AM hour)
are taken directly from these regex hits.
Question types.All questions and gold an-
swers are produced deterministically from the
parsedcorpusbyagenerationscript(werelease
the script and the independent validator). No
hand-editing is applied. The resulting evalua-
tion file has 100 questions across four groups:
1.Single-document lookups(69/100):
per-hearing questions about the commit-
tee, the chamber, the hearing date, and
the number of witnesses listed in the Con-
tents block.
2.Chamber-level aggregations(6/100):
corpus-wide counts and comparisons
across chambers, e.g., the total num-
ber of House versus Senate hearings, the
size of the House–Senate difference, and
which chamber predominates corpus-wide
or within a year.
3.Chamber×year intersections and
year totals(3/100): e.g., how many
House hearings occurred in 2022; the total
number of hearings in a given year.
4.Labeled-field3-wayconcept-majority
questions(22/100): given three corpus-
level features (e.g., an attendance roll, a
published online transcript link, and an
11 AM adjournment notation), the ques-
tion asks which of the three appears in the
mosthearings; chamber-restrictedvariants
ask the same within House- or Senate-only
subsets.
Natural-language phrasing without lead-
ing hints.Both the question text and the
gold answers are phrased in natural language,
without quoted regex markers or formatting
hints. The two semantically equivalent at-
tendance markers ( Present: andMembers
present: ) are merged into a single concept
so that a natural reading of the question is
unambiguous.
E LLM-as-a-Judge Details
WeevaluateallsystemsusingthesameLLM-as-
a-judge pipeline to avoid system-specific eval-
uators. For each question, we run two judge-
model calls:1.Answer-only extraction:transform
a possibly verbose model output into
a single “final answer” string (or
INSUFFICIENT_EVIDENCE).
2.Judging:compare the extracted final an-
swer against the gold answer and return a
JSON verdict and score.
Prompt for answer-only extraction (reason-
ing removal):
You are a QA assistant.
You will be given:
- the question
- a model answer that may contain reasoning
Your task:
- Output ONLY the final answer to the question.
- Do NOT include any reasoning, explanation, preamble, or
extra text.
- Do NOT use markdown code fences.
If the answer cannot be determined from the provided model
answer, output exactly:
INSUFFICIENT_EVIDENCE
SQuAD-style exact-match and token-overlap
F1 are computed on the output of the above
normalization.
Prompt for judging rules and scoring:
You are an accurate evaluator for QA.
You will be given:
- the question
- the gold answer
- the model answer (final answer only; no reasoning)
Decide if the model answer matches the gold answer.
Be strict about factual correctness, but allow paraphrases
and equivalent numeric formats.
If the gold answer is empty, mark as incorrect unless the
model explicitly says it cannot be determined (e.g.,
INSUFFICIENT_EVIDENCE).
INSUFFICIENT_EVIDENCE rule:
- If the model answer is exactly the string "
INSUFFICIENT_EVIDENCE" AND the gold answer is non-
empty, verdict MUST be "incorrect" and score MUST be
0.
Special scoring rules:
1) Count questions:
- If the question is asking for a count/number, the answer
is ONLY correct if the predicted number exactly
matches the gold number.
- If it is off by even 1, score MUST be 0 and verdict MUST
be "incorrect".
2) List/set questions:
- Treat the gold answer and model answer as sets of items.
- Let L be the number of distinct gold items.
- Start with score=1.0.
- Apply a penalty of (1/L) for each missing gold item AND
for each extra predicted item.
- Treat possible aliases as correct. Do not give partial
score per element.
- Score is floored at 0.
- If the model includes additional explanatory text,
ignore it and focus on the item set.
When the answer is a proper name / named entity:
- Treat clear aliases or alternative spellings as correct
(score=1).
- If the model answer is ambiguous (could refer to
multiple entities) but not clearly wrong, allow
partial credit with a short rationale.
24

- Give score=0 only if it is clearly the wrong entity.
Return a JSON object with:
- verdict: one of ["correct","incorrect","partial"]
- score: number in [0,1] where correct=1, incorrect=0,
partial is between
- rationale: short explanation (1-3 sentences)
- abs_diff: OPTIONAL. Include ONLY for count/number
questions. It must be a non-negative integer equal to
|gold_number - predicted_number| if you can extract
both numbers; otherwise null.
Rules:
- Output must be valid JSON (no markdown fences, no extra
text).
25