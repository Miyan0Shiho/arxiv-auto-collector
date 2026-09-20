# When Is Graph Structure Worth Its Cost? The Case for Structure Pricing in Retrieval-Augmented Generation

**Authors**: Yuzhong Zhang, Haoyang Ma, Chao Peng, Lionel Briand, Boxi Yu, Jialun Cao

**Published**: 2026-09-16 04:08:15

**PDF URL**: [https://arxiv.org/pdf/2609.18099v1](https://arxiv.org/pdf/2609.18099v1)

## Abstract
Graph-based retrieval-augmented generation (RAG) can help answer questions that require information from many documents. However, building a graph often requires many language-model calls during ingestion. It is therefore important to ask whether its quality gains justify the additional cost.
  We present EffiRAG, a graph-based RAG system designed to reduce this cost. It uses the graph to locate relevant passages and generates answers from the original text. This design preserves source information while keeping graph construction and query processing lightweight.
  We evaluate EffiRAG on UltraDomain, which contains 120 open-ended questions from four domains. Compared with LightRAG-hybrid, EffiRAG produces the preferred answer on 93 questions. LightRAG is preferred on 7, and the remaining 20 are splits. EffiRAG also reduces total system cost by 57 percent, from USD 0.952 to USD 0.408. The cost includes language-model calls during ingestion and querying.
  The advantage remains as the corpus grows. At 10 and 20 documents per domain, EffiRAG uses a lightweight, non-LLM filter to skip low-salience chunks. It remains preferred over LightRAG-hybrid. It costs 4.2 times and 4.5 times less, respectively.
  The comparisons identify different quality-cost trade-offs. Graph-based RAG systems should therefore be evaluated by both answer quality and cost. The results favor graph structure that locates and preserves source evidence.

## Full Text


<!-- PDF content starts -->

When Is Graph Structure Worth Its Cost?
The Case for Structure Pricing in Retrieval-Augmented
Generation
Yuzhong Zhang
The Chinese University of Hong
Kong, Shenzhen
Shenzhen, ChinaHaoyang Ma
The Hong Kong University of
Science and Technology
Hong Kong, ChinaChao Peng
University of Edinburgh
Edinburgh, United Kingdom
Lionel Briand
Lero, the Research Ireland Centre for
Software, University of Limerick
Limerick, Ireland
University of Ottawa
Ottawa, Canada
lionel.briand@lero.ieBoxi Yu∗
Lero, the Research Ireland Centre for
Software, University of Limerick
Limerick, Ireland
boxi.yu@lero.ieJialun Cao
The Hong Kong University of
Science and Technology
Hong Kong, China
Abstract
Graph-based retrieval-augmented generation (RAG) can help an-
swer questions that require information from many documents.
However, building a graph often requires many language-model
calls during ingestion. It is therefore important to ask whether its
quality gains justify the additional cost.
We present EffiRAG, a graph-based RAG system designed to re-
duce this cost. It uses the graph to locate relevant passages and
generates answers from the original text. This design preserves
source information while keeping graph construction and query
processing lightweight.
We evaluate EffiRAG on UltraDomain, which contains 120
open-ended questions from four domains. Compared with LightRAG-
hybrid, EffiRAG produces the preferred answer on 93 questions.
LightRAG is preferred on 7, and the remaining 20 are splits.
EffiRAG also reduces total system cost by 57%, from $0.952 to
$0.408. The cost includes language-model calls during ingestion
and querying.
The advantage remains as the corpus grows. At 10 and 20 doc-
uments per domain, EffiRAG uses a lightweight, non-LLM filter
to skip low-salience chunks. It remains preferred over LightRAG-
hybrid. It costs 4.2×and 4.5×less, respectively.
The comparisons identify different quality–cost trade-offs.
Graph-based RAG systems should therefore be evaluated by both
answer quality and cost. The results favor graph structure that
locates and preserves source evidence.
1 Introduction
Retrieval-augmented generation (RAG) [1] allows large language
models to answer questions over private or recently updated cor-
pora without additional fine-tuning. A standard RAG system splits
documents into chunks, embeds them, retrieves the chunks most
relevant to a query, and generates an answer from the retrieved
text. This pipeline is simple and relatively inexpensive. However, it
∗Corresponding author.can miss information distributed across several chunks, especially
for comparison, cross-document reasoning, or corpus synthesis.
Graph-based RAG systems address this limitation by represent-
ing entities and relations explicitly. GraphRAG organizes graph
elements into communities and generates community summaries
for corpus-level sensemaking [3]. LightRAG combines graph-
based retrieval with vector retrieval [2]. These structures can
connect information that chunk similarity alone may miss, but
extraction, relation discovery, summarization, and graph traversal
add language-model calls.
This paper therefore asks a practical question:when does
graph structure improve answer quality enough to justify
the cost of building and using it?We study this trade-off
by measuring graph-construction and query cost together with
answer quality.
We introduce EffiRAG, a graph-based RAG system built around
a simple principle: the graph locates evidence, while source chunks
supply the answer material. During ingestion, EffiRAG extracts en-
tities and relations from source chunks. It links each extraction
back to the chunk that supports it. During querying, these extrac-
tions help locate relevant source chunks under a fixed context bud-
get. The generator answers from the original source text, with the
extractions serving as retrieval handles. We call this designsource-
chunk grounding.
EffiRAG controls ingestion cost through bounded graph extrac-
tion. It extracts each source chunk once and omits community
summarization, reducing ingestion cost relative to LightRAG-
hybrid.
We reportsystem costas the input- and output-token charges
for provider-LLM calls during ingestion and querying. Evaluation
judging is excluded. Local embeddings are treated as unbilled.
In our main comparison,EffiRAGapplies this bounded extrac-
tion policy to all source chunks and uses a fixed context budget.
We evaluate it on UltraDomain, which contains 120 open-ended
queries across four domains. EffiRAG is preferred over LightRAG-
hybrid on 93 queries. LightRAG-hybrid is preferred on 7, and the
remaining 20 produce no consistent preference. Additional checks
arXiv:2609.18099v1  [cs.AI]  16 Sep 2026

Yuzhong Zhang, Haoyang Ma, Chao Peng, Lionel Briand, Boxi Yu, and Jialun Cao
show robustness to judging and answer-length effects. Across in-
gestion and the 120 queries, EffiRAG costs $0.408, compared with
$0.952 for LightRAG-hybrid, a reduction of 57%.
We also study larger corpora containing 10 and 20 documents
per domain. EffiRAG retains the same fixed-budget query policy.
Before extraction, a lightweight, non-LLM salience filter selects
chunks likely to contain useful relations. EffiRAG remains pre-
ferred over LightRAG-hybrid while reducing system cost by fac-
tors of 4.2 and 4.5 in the 10- and 20-document-per-domain settings.
Under tight cost constraints, lightweight non-graph retrievers
such as BM25-vector [13], NaiveRAG, and HyDE [6] may be more
practical, although they produce lower-quality answers on Ultra-
Domain.
This paper makes three contributions:
(1) We introduce EffiRAG, a source-grounded graph-RAG
system. Extracted entities and relations guide retrieval.
The original source chunks remain available to the answer
generator. On UltraDomain, EffiRAG is preferred over
LightRAG-hybrid at substantially lower system cost.
(2) We provide a cost evaluation that jointly reports ingestion
and query-time provider-LLM usage.
(3) We provide an empirical analysis of when graph con-
struction is worthwhile. The analysis covers non-graph
baselines, extraction-free graphs, larger corpora, and gold-
answer multi-hop question answering.
2 Related Work
Graph construction.GraphRAG uses LLM-based entity and
relation extraction, community detection, and generated commu-
nity summaries [3]. LightRAG constructs an entity–relation graph
alongside vector indexes [2], while LazyGraphRAG lowers index-
ing cost through non-LLM graph construction [7]. EffiRAG retains
bounded LLM extraction but omits community summarization
and links every extracted entity and relation to its supporting
source chunks.
Graph use.GraphRAG retrieves community reports for
corpus-level questions [3]. LightRAG combines graph and vec-
tor retrieval [2], while PathRAG retrieves selected relational paths
to reduce irrelevant graph context [4]. EffiRAG uses extracted
entities and relations to locate their supporting source chunks.
The generator receives the retrieved relations and original text, so
the graph guides retrieval while retaining source evidence.
Cost–quality evaluation.GraphRAG-Bench studies whether
the quality gains of graph-based RAG justify its construction
cost [11], while LazyGraphRAG targets lower indexing cost [7].
EffiRAG jointly measures provider-LLM token cost during inges-
tion and querying and compares it with judged answer quality.
It also includes lower-cost non-graph retrievers to identify when
graph construction provides sufficient benefit.
Positioning.GraphRAG relies on community summaries for
corpus-level synthesis. LightRAG combines graph and vector re-
trieval. LazyGraphRAG prioritizes low indexing cost via non-LLM
construction. EffiRAG takes a different route: it uses bounded LLM
extraction to locate original source chunks, and reports ingestion
and querying cost jointly.3 Method
EffiRAG has two stages (Figure 1).
Ingestion divides the corpus into source chunks. Each chunk
is embedded locally. An LLM extracts entities and relations from
each chunk.
Querying retrieves relevant entities, relations, and source
chunks. It selects evidence under a fixed context budget. The
generator then answers from the selected evidence.
The central design principle issource-chunk grounding: every
extracted entity or relation retains a link to the chunk that supports
it. The extracted entities and relations serve as retrieval handles,
while the source text remains the answer evidence. This matters
because a compact relation may omit information such as dates,
negation, conditions, or uncertainty. EffiRAG gives the generator
both the retrieved relations and their supporting source chunks.
Ingestion.Asource chunkis a contiguous unit of original docu-
ment text used for extraction and retrieval. The graph store con-
tains a source-chunk record for each chunk and two types ofgraph
record: entity records and relation records. A source-chunk record
stores the original text and its embedding. An entity record stores
a normalized key, name, type, description, mention count, and ref-
erences to its supporting source chunks. A relation record stores
the source entity, target entity, relation keywords, description, and
a reference to its supporting source chunk.
For retrieval, each graph record is converted torecord text. En-
tity record text has the form “name (type): description.” Relation
record text has the form “source→target (keywords): descrip-
tion.” The graph store holds the records and their source links.
Separate embedding collections for graph-record text and source
chunks support cosine-similarity search.
Retrieval and generation.Given a query, EffiRAG embeds it and
compares the query embedding with the graph-record and source-
chunk embeddings. Graph records are ranked by the cosine simi-
larity of their record-text embeddings. Source chunks are ranked
in the same way using their stored embeddings. EffiRAG follows
the source links of the retrieved graph records. It combines the re-
sulting supporting chunks with those retrieved directly. Duplicates
are removed.
EffiRAG formats the selected graph records as a textualgraph
view. The final context contains this graph view together with the
selected source chunks. The graph view exposes entity and relation
information, while the source chunks provide the original answer
evidence.
Query-time evidence selection.At query time, Coverage-Budgeted
Evidence Selection (CBES) assembles graph records and source
chunks under acontext budget𝐵.𝐵is the maximum number of
tokens placed in the final context.
Query aspects are named entities and noun phrases extracted
from the query. CBES uses a submodular coverage objective. An
item receives less additional value when its aspects are already
covered by selected evidence. It receives more value when it covers
new aspects. It greedily selects the item with the largest marginal
coverage gain per token. It stops when no remaining item fits the
budget. This reduces redundant context.

When Is Graph Structure Worth Its Cost?
The Case for Structure Pricing in Retrieval-Augmented Generation
Ingest
Corpuschunk + embed
local · unbilled
source chunks + embeddingsLLM extraction
$ ingest costGraph store
Query
Query qretrieve
fixed budget
graph records + chunksgraph finds the evidence
assemble
Context: graph view + source chunksgenerate
$ query cost
Answer
graph record — retrieval handle source chunk — answer evidence
Figure 1: EffiRAG pipeline. Extracted entities and relations locate evidence; their supporting source chunks provide the orig-
inal answer text.
Reported settings.EffiRAG combines source-chunk grounding,
bounded graph extraction, and fixed-budget evidence selection.
Unless stated otherwise,EffiRAGrefers to the default setting used
in the five-document-per-domain comparison.
EffiRAG extracts every source chunk once and omits commu-
nity summarization. Before CBES, EffiRAG retrieves up to 16 en-
tity records, 16 relation records, and 8 source chunks. It then adds
at most 4 supporting chunks.
These are candidate-pool caps, not the final context size. The fi-
nal context is separately bounded by𝐵=3000. The caps only need
to be large enough to give CBES a diverse candidate set. We chose
16/16/8/4 as a conservative setting that is well above the typical
number of items CBES selects. We did not tune these values on
UltraDomain.
EffiRAG w/ salience filteringskips chunks before extraction us-
ing a non-LLM score based on lexical content, entity cues, and em-
bedding novelty. It is used in the larger-corpus experiments and
keeps the same query policy as EffiRAG.
4 Evaluation
We first measure the quality–cost trade-off of EffiRAG and the ef-
fect of reducing ingestion work. We then compare EffiRAG with
graph and lower-cost non-graph retrievers. Finally, we examine
individual design choices and test the robustness of the judged-
quality conclusion.
4.1 Evaluation Design and Rationale
We evaluate on a four-domain sample of UltraDomain [21]: agri-
culture, computer science, legal, and mixed. Each domain has five
documents and 30 open-ended questions, for 120 queries in total.The questions require cross-document comparison and corpus-
level synthesis. This makes the benchmark suited to testing
whether graph structure helps connect evidence across docu-
ments. We follow LightRAG’s public dataset-construction proto-
col. Concretely, for each domain we select a fixed document set
and generate open-ended questions from the corpus using an LLM.
The questions target cross-document comparison and synthesis.
We use LightRAG’s released prompt and filtering rules without
modification.
We organize the comparison by retrieval role. LightRAG-hybrid
is the primary baseline because it combines graph and vector re-
trieval. PathRAG represents an alternative graph-retrieval design.
NaiveRAG, HyDE, and BM25-vector are lower-cost non-graph
controls. HippoRAG2 is evaluated separately on gold-answer
multi-hop QA, outside the main UltraDomain comparison.
We use DeepSeek-V4-Flash as the answer model (deepseek-v4-flash,
non-thinking mode) [15]. It provides a competitive balance of gen-
eration quality and inference cost, and was the model available in
our deployment environment.
We useall-MiniLM-L6-v2for embeddings [14, 16]. It runs lo-
cally, which keeps embedding cost separate from the provider-
LLM cost accounting. It is also publicly available and stable across
runs. Holding these models fixed isolates the effects of retrieval
and graph construction. System cost includes provider-LLM calls
during ingestion and querying. Judge calls are excluded, and local
embeddings are treated as unbilled.
We evaluated context budgets of𝐵∈ {1500,2000,3000,4500}
over all 120 queries (Table 1). Increasing𝐵from 3000 to 4500 added
705 selected tokens on average, while the win rate relative to full-
context generation increased by only 0.008. We therefore use𝐵=

Yuzhong Zhang, Haoyang Ma, Chao Peng, Lionel Briand, Boxi Yu, and Jialun Cao
Table 1: Context-budget sensitivity over all 120 queries. Win
rate compares each budgeted setting with full-context gen-
eration.
𝐵Avg. tokens Win rate
1500 1231 0.208
2000 1694 0.317
3000 2442 0.367
4500 3147 0.375
0.10.20.30.40.6
0.5 tie
$0 $0.2 $0.4 $0.6 $0.8 $1
System cost (USD)Baseline query-level scorebelow 0.5 = EffiRAG preferredEffiRAG
EffiRAG w/o LLM extractionBM25
HyDE
NaiveRAG
LightRAGPathRAG
Figure 2: Cost and judged quality on UltraDomain.
3000as a practical fixed budget rather than claiming that it is op-
timal.
We use pairwise LLM judging. Given the same question and two
candidate answers, GPT-4o-mini [17] selects the better answer. It
judges correctness, relevance, and completeness.
Presentation order can affect pairwise judgments [8]. Each pair
is therefore judged twice with the answer order reversed. A query
is a win for a system when both orders prefer it. We call a query
with no consistent preference asplit. A split happens when the two
orders disagree, or the judge returns a tie. We report query-level
wins, losses, and splits. When a single score is needed, splits count
as half:
score=wins+0.5 splits
𝑁.
4.2 Main Quality–Cost Result
No measured baseline is both cheaper than EffiRAG and preferred
over it on UltraDomain (Figure 2 and Table 2).
Against LightRAG-hybrid, EffiRAG wins 93 queries, loses 7,
and splits 20. The result is consistent across all four domains:
the win/loss margins are 20/2 in agriculture, 27/1 in computer
science, 26/1 in legal, and 20/3 in mixed. Across ingestion and
the 120 queries, EffiRAG costs $0.408, compared with $0.952 forLightRAG-hybrid, a reduction of 57.1%. In this experiment, Ef-
fiRAG has higher total query wall-clock time (1163 versus 888
seconds).
4.3 Ingestion Cost Control
EffiRAG extracts every source chunk in the main five-document
setting. We ask whether extracting fewer chunks can reduce cost
while keeping the same quality conclusion.
Random removal could confound extraction volume with con-
tent coverage. We therefore construct a series of extraction set-
tings with progressively smaller extracted corpora while preserv-
ing broad corpus coverage. To do so, we use a fixed non-LLM se-
lector that estimates the coverage contributed by each chunk.
The selector combines four complementary signals: noun-
phrase coverage, cross-document coverage, corpus representa-
tiveness, and expected query-workload coverage. It uses these
signals to estimate the additional coverage contributed by each
chunk per token. The selector is fixed across all settings and is
used only for this sensitivity analysis.
We evaluated𝜏∈{0,0.03,0.05,0.07,0.10}, which retained 68, 45,
24, 4, and 1 of the 68 chunks, respectively. Each setting was eval-
uated on the same UltraDomain queries to measure its quality–
cost trade-off. As extraction volume decreased, measured system
cost also decreased, but judged answer quality consistently de-
clined. We therefore retain all-chunk extraction in the main five-
document EffiRAG configuration.
As the boundary of this analysis, we also remove LLM extrac-
tion entirely.EffiRAG w/o LLM extractionuses non-LLM noun-
phrase extraction while retaining the same query pipeline. It costs
$0.226, 44.7% less than EffiRAG, but EffiRAG wins 44 queries, the
zero-extraction setting wins 26, and 50 are splits.
These results support all-chunk extraction at the five-document
scale. As the corpus grows, however, extracting every chunk be-
comes increasingly expensive. The selector above is used only to
study the effect of reducing extraction volume. For larger corpora,
we instead evaluate a lightweight online filtering strategy intended
to control ingestion cost. This configuration isEffiRAG w/ salience
filtering.
The filter combines lexical cues, entity cues, and embedding
novelty to estimate chunk salience. Chunks with low salience are
skipped before graph extraction.
EffiRAG w/ salience filtering remains preferred over LightRAG-
hybrid. Its query-level scores are 0.800 and 0.771 at 10 and 20
documents per domain, respectively, while costing 4.2×and 4.5×
less ($2.316 versus $9.799 and $3.238 versus $14.608). EffiRAG w/
salience filtering remains effective at both measured corpus sizes.
4.4 Comparison Scope
Against PathRAG, EffiRAG reduces cost from $0.954 to $0.408.
Judged quality stays close: EffiRAG wins 39 queries, PathRAG
wins 26, and 55 are splits.
NaiveRAG, HyDE, and BM25-vector cost less than EffiRAG.
Their query-level pairwise scores against EffiRAG are 0.321, 0.371,
and 0.408, respectively (score = (baseline wins+0.5×splits)/120;
below 0.5 means EffiRAG is preferred).

When Is Graph Structure Worth Its Cost?
The Case for Structure Pricing in Retrieval-Augmented Generation
Table 2: UltraDomain quality and system cost (120 queries). Quality reports query-level EffiRAG wins / baseline wins / splits.
Cost includes provider-LLM ingestion and querying. Cost delta is relative to EffiRAG. Query time is cumulative wall-clock
time for all 120 queries.
System EffiRAG wins / baseline wins / splits Cost Cost delta Query time (s)
EffiRAG (baseline) $0.408 +0.0% 1162.5
LightRAG-hybrid 93/7/20 $0.952 +133.1% 888.1
PathRAG 39/26/55 $0.954 +133.5% 1565.3
NaiveRAG 58/15/47 $0.139 -66.0% 1011.6
HyDE 51/20/49 $0.194 -52.4% 1419.4
BM25-vector 43/21/56 $0.145 -64.6% 1059.8
EffiRAG w/o LLM extraction 44/26/50 $0.226 -44.7% 1069.1
Table 3: EffiRAG w/ salience filtering versus LightRAG-
hybrid at 10 and 20 documents per domain (120 queries
per scale): query-level wins, losses, splits, and score.
†Agriculture has 12 unique documents at the 20-document
point.
Domain EffiRAG LightRAG Split Score
10 docs per domain
Agriculture 21 1 8 0.833
CS 16 3 11 0.717
Legal 26 1 3 0.917
Mixed 19 5 6 0.733
All 82 10 28 0.800
20 docs per domain
Agriculture†18 2 10 0.767
CS 16 4 10 0.700
Legal 23 0 7 0.883
Mixed 19 5 6 0.733
All 76 11 33 0.771
Pooled system cost: EffiRAG w/ salience filtering $2.316 vs. LightRAG-hybrid $9.799
at 10 docs (4.2×lower), and $3.238 vs. $14.608 at 20 docs (4.5×lower).
UltraDomain emphasizes retrieving and synthesizing evidence
distributed across documents. Gold-answer multi-hop QA instead
emphasizes reasoning over linked facts to produce an exact an-
swer. We therefore include a complementary multi-hop evaluation
using exact-match (EM) and token-level F1.
We use fixed 100-question development slices of HotpotQA [18],
2WikiMultihopQA [19], and MuSiQue [20]. The pairwise judge
prefers HippoRAG2 in 312 of 600 order-level decisions, and thus
EffiRAG’s multi-hop configuration in 288. Their aggregate EM/F1
scores are 0.460/0.583 for EffiRAG and 0.447/0.570 for HippoRAG2.
The two systems achieve similar aggregate quality under different
metrics, while EffiRAG has substantially lower measured answer-
generation cost.
4.5 Design Evidence
The testbed includes several mechanisms (salience skipping,
extractor-tier routing, entity-repeat hints, query expansion, query-
class-dependent retrieval, and context compression).
Source-chunk grounding has the clearest positive effect. Re-
moving linked source chunks reduces cost by 49.9%. The testbed
is preferred in 195 of 240 order-level decisions, compared with 44
for the ablated system and one tie (Table 4).Context compression also reduces cost but lowers quality. The
testbed is preferred over the compressed variant by 170/70/0. The
remaining rows show small or inconsistent quality differences
within the testbed.
4.6 Reliability
Finally, we test whether the main EffiRAG–LightRAG-hybrid con-
clusion depends on answer order, ambiguous questions, answer
length, or the judge model.
Among the 120 queries, 20 are splits. A two-sided sign test on
the remaining 100 queries gives𝑝<0.001. The query-level score
is 0.858. The bootstrap 95% confidence interval is[0.804,0.908].
Two length controls preserve the conclusion. A re-judge explic-
itly instructed not to reward answer length gives EffiRAG a query-
level score of 0.883. On the subset whose answer lengths differ by
at most 20%, EffiRAG retains a query-level score of 0.682.
As an independent-model check, Gemini reviewed 80 answer
pairs. We selected pairs for which GPT-4o-mini’s judgments were
close or order-sensitive, preferred the baseline, or favored the
shorter answer.
GPT-4o-mini selects the same answer in both orders for 23 of
them. The other 57 have no single direction for agreement. Gemini
agrees on 17 of the 23 comparable pairs (74%). This check is consis-
tent with the conclusion that EffiRAG is preferred over LightRAG-
hybrid.
5 Discussion and Threats to Validity
The results suggest that graph structure is most valuable when
it helps retrieve original evidence, with source chunks remain-
ing the answer material. Removing source chunks substantially
lowers judged quality, whereas additional routing, budgeting,
and compression mechanisms provide no clear improvement.
Different workloads nevertheless favor different quality–cost
trade-offs: lightweight retrievers minimize cost, while EffiRAG
and HippoRAG2 show comparable multi-hop QA quality under
different metrics. The larger-corpus results provide two measured
cost-control points for EffiRAG w/ salience filtering.
The evaluation uses 120 LLM-generated queries from four do-
mains and LLM-based pairwise judgments. Reversed answer or-
ders, length controls, statistical tests, and an independent-model
check reduce judging effects, but other datasets, human judgments,

Yuzhong Zhang, Haoyang Ma, Chao Peng, Lionel Briand, Boxi Yu, and Jialun Cao
Table 4: Mechanism analysis using the separate ablation testbed on UltraDomain (120 queries). The testbed costs $0.469. Qual-
ity is testbed wins / variant wins / ties across 240 order-level decisions. Each row changes one testbed component and is not a
direct variant of the main EffiRAG setting.
Ablation Quality Cost Cost delta (vs. testbed) Avg. context chars
w/o source chunks 195/44/1 $0.235 -49.9% 3846
w/ compression 170/70/0 $0.259 -44.8% 31209
w/o salience filtering 116/124/0 $0.451 -3.9% 31756
lightweight extraction only 115/125/0 $0.453 -3.4% 32106
w/o entity-repeat hints 105/135/0 $0.468 -0.4% 31622
w/ fixed retrieval counts 115/125/0 $0.430 -8.5% 26411
w/o query expansion 119/121/0 $0.471 +0.3% 31594
and model stacks may produce different results. Dollar costs also
depend on provider prices and exclude local computation.
6 Conclusion
This paper asked when graph structure improves retrieval-
augmented generation enough to justify its cost. EffiRAG ad-
dresses this question with a source-grounded design: extracted
entities and relations locate relevant evidence, while the gener-
ator answers from the linked source chunks. On UltraDomain,
EffiRAG is preferred over LightRAG-hybrid on 93 of 120 queries
while reducing measured system cost by 57%. EffiRAG w/ salience
filtering remains preferred at lower measured cost on the larger
corpora studied here.
The broader result is that graph-RAG systems require joint eval-
uation of answer quality, graph complexity, ingestion cost, query
cost, and the evidence that reaches the generator. Our results fa-
vor using the minimum graph structure needed to connect relevant
source evidence, then selecting a configuration appropriate to the
workload and cost budget.
Acknowledgements
This work has emanated from research jointly funded by Taighde
Éireann–Research Ireland under Grant Number 13/RC/2094_2, and
by Genesys Cloud Services, Inc.
References
[1] P. Lewis et al., “Retrieval-augmented generation for knowledge-intensive NLP
tasks,” inNeurIPS, 2020.
[2] Z. Guo et al., “LightRAG: Simple and fast retrieval-augmented generation,”
arXiv:2410.05779, 2024.
[3] D. Edge et al., “From local to global: A graph RAG approach to query-focused
summarization,” arXiv:2404.16130, 2024.
[4] B. Chen et al., “PathRAG: Pruning graph-based retrieval augmented generation
with relational paths,” arXiv:2502.14902, 2025.
[5] P. Sarthi et al., “RAPTOR: Recursive abstractive processing for tree-organized
retrieval,” inICLR, 2024.
[6] L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot dense retrieval without
relevance labels,” inACL, 2023.
[7] D. Edge, H. Trinh, and J. Larson, “LazyGraphRAG: Setting a new standard for
quality and cost,” Microsoft Research Blog, November 2024.
[8] L. Zheng et al., “Judging LLM-as-a-judge with MT-Bench and Chatbot Arena,”
inNeurIPS, 2023.
[9] G. L. Nemhauser et al., “An analysis of approximations for maximizing submod-
ular set functions—I,”Math. Prog., 1978.
[10] S. Khuller, A. Moss, and J. Naor, “The budgeted maximum coverage problem,”
Inf. Process. Lett., 1999.
[11] Z. Xiang et al., “When to use graphs in RAG: A comprehensive analysis for
graph retrieval-augmented generation,” arXiv:2506.05690, 2025.[12] B. J. Gutiérrez et al., “From RAG to memory: Non-parametric continual learning
for large language models,” arXiv:2502.14802, 2025.
[13] S. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25
and beyond,”Foundations and Trends in Information Retrieval, vol. 3, no. 4,
pp. 333–389, 2009.
[14] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using
Siamese BERT-networks,” inEMNLP, 2019.
[15] DeepSeek-AI, “DeepSeek-V4-Flash,” Hugging Face model card, 2026. [Online].
Available: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash.
[16] Sentence Transformers, “all-MiniLM-L6-v2,” Hugging Face model card. [On-
line]. Available: https://huggingface.co/sentence-transformers/all-MiniLM-L6-
v2.
[17] OpenAI, “GPT-4o mini,” model documentation. [Online]. Available: https://
platform.openai.com/docs/models/gpt-4o-mini.
[18] Z. Yang et al., “HotpotQA: A dataset for diverse, explainable multi-hop question
answering,” inEMNLP, 2018.
[19] X. Ho et al., “Constructing a multi-hop QA dataset for comprehensive evaluation
of reasoning steps,” inCOLING, 2020.
[20] H. Trivedi et al., “MuSiQue: Multihop questions via single-hop question com-
position,”TACL, 2022.
[21] H. Qian et al., “MemoRAG: Boosting long context processing with global
memory-enhanced retrieval augmentation,” arXiv:2409.05591, 2024.