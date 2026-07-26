# Semantically Similar, Logically Distinct: Diagnosing the Semantic-Answerability Gap in Table RAG

**Authors**: Jiaming Tian, Liyao Li, Wentao Ye, Haobo Wang, Lihua Yu, Zujie Ren, Gang Chen, Junbo Zhao

**Published**: 2026-07-20 09:36:10

**PDF URL**: [https://arxiv.org/pdf/2607.17742v1](https://arxiv.org/pdf/2607.17742v1)

## Abstract
Tables are a critical knowledge source in retrieval-augmented generation (RAG), but a retrieved table may lack sufficient evidence to answer a query, a property we call answerability. While answerability broadly concerns whether a source or collection of sources contains sufficient evidence, retrieval models optimized for semantic relevance do not guarantee it even in the single-source case, creating a fundamental mismatch. To study this, we introduce TCR-Bench, a diagnostic benchmark for Table Content-level Answerability in RAG, built around sibling tables, i.e., tables with highly similar schemas but subtle content differences. On TCR-Bench, the dense retrievers we evaluate persistently exhibit a Semantic-Answerability Gap: they often retrieve the correct sibling group yet struggle to pinpoint the uniquely answerable table within it, dropping QA performance from 0.755 (oracle) to 0.330 (top-5 retrieved). Our analysis suggests this gap is associated with semantic accumulation, schema-level cue dependence, and weak row-column binding. As a diagnostic probe into the source of this gap, we test whether a lightweight two-stage pipeline, Answerability-Aware Reranking (AAR), applying direct query-table answerability judgment, can recover performance: it raises top-1 target retrieval from 18.2% to 57.4%, and this large gain is itself evidence that much of the observed failure reflects a missing answerability verification step, rather than an inherent limitation of model capacity alone.

## Full Text


<!-- PDF content starts -->

Semantically Similar, Logically Distinct:
Diagnosing the Semantic-Answerability Gap in Table RAG
Jiaming Tian1, Liyao Li1, Wentao Ye1, Haobo Wang1, Lihua Yu2, Zujie Ren1,3*, Gang Chen1, Junbo Zhao1
1Zhejiang University
2Bank of Hangzhou Co., Ltd.
3Zhejiang Lab
hsxz2@zju.edu.cn, renzju@zju.edu.cn
Abstract
Tables are a critical knowledge source in
retrieval-augmented generation (RAG), but a
retrieved table may lack sufficient evidence to
answer a query, a property we callanswer-
ability. While answerability broadly concerns
whether a source or collection of sources con-
tains sufficient evidence, retrieval models op-
timized for semantic relevance do not guaran-
tee it even in the single-source case, creating
a fundamental mismatch. To study this, we
introduceTCR-Bench1, a diagnostic bench-
mark forTableContent-level Answerability in
RAG, built aroundsibling tables, i.e., tables
with highly similar schemas but subtle con-
tent differences. On TCR-Bench, the dense
retrievers we evaluate persistently exhibit a
Semantic-Answerability Gap: they often re-
trieve the correct sibling group yet struggle to
pinpoint the uniquely answerable table within
it, dropping QA performance from 0.755 (or-
acle) to 0.330 (top-5 retrieved). Our analysis
suggests this gap is associated with semantic
accumulation, schema-level cue dependence,
and weak row-column binding. As a diag-
nostic probe into the source of this gap, we
test whether a lightweight two-stage pipeline,
Answerability-Aware Reranking (AAR), ap-
plying direct query-table answerability judg-
ment, can recover performance: it raises top-1
target retrieval from 18.2% to 57.4%, and this
large gain is itself evidence that much of the ob-
served failure reflects a missing answerability
verification step, rather than an inherent limita-
tion of model capacity alone.
1 Introduction
Tables in enterprise databases, data lakes, and web
resources contain rich structured information (Ca-
farella et al., 2008; Stonebraker et al., 2013; Nar-
gesian et al., 2018). In RAG systems (Lewis et al.,
*Corresponding author.
1Code and data are available at https://github.com/
minger-hsxz/TCR-Bench-Open.
Country Year Gold
USA 2016 46
GBR 2016 27
··· ··· ···Country Year Gold
GBR 2012 29
GBR 2016 27
··· ··· ···Query: Who won more than 29 gold medals at the 2016 Olympics?
Retrieved Table A Retrieved Table B
Dense Retriever Score = 0.88
AnswerableDense Retriever Score = 0.89
Not AnswerableFigure 1: Two sibling tables with nearly identical
schemas yield similar retrieval signals but differ in an-
swerability: Table A contains the answer, while Table B
scores slightly higher because it more directly matches
salient query tokens such as 2016 and 29.
2020; Huang and Huang, 2024), retrieved sources
must provide not only topical relevance but suffi-
cient evidence to answer a query, a property we
termanswerability. Semantic relevance does not
guarantee this: two sources may appear equally
relevant while differing critically in whether they
contain the required evidence. Table retrieval offers
a clean setting for studying this problem, as pre-
cise row-column grounding and numerical match-
ing (Herzig et al., 2020; Deng et al., 2022) make
answerability failures easier to isolate.
Figure 1 illustrates the core challenge: two sib-
ling tables with nearly identical schemas receive
similar retrieval signals, yet only one contains suf-
ficient evidence to answer the query. Such sib-
lings are common in real data lakes (Lou et al.,
2024; Yang et al., 2021; Shraga and Miller, 2023),
arising from filtering, truncation, or minor revi-
sions. We term this failure mode theSemantic-
Answerability Gap: retrievers capture coarse sim-
ilarity but fail to distinguish answerability. On our
benchmark, dense retrievers we test identify the
answerable table only18.2%of the time, causing
QA performance to fall from0.755to0.330.
Existing table QA benchmarks such as Wik-
iTableQuestions and HybridQA (Pasupat and
Liang, 2015; Chen et al., 2020) assume the rel-
1
arXiv:2607.17742v1  [cs.AI]  20 Jul 2026

evant table is given, bypassing retrieval entirely.
Recent Table RAG work (Pan et al., 2022; Balaka
et al., 2025; Yu et al., 2025) improves retrieval and
reasoning, yet evaluation still emphasizes semantic
relevance over evidence sufficiency.
To study this problem systematically, we in-
troduceTCR-Bench, a diagnostic benchmark for
TableContent-level Answerability Benchmark in
RAG. Built around sibling tables, TCR-Bench iso-
lates answerability from coarse semantic relevance
and enables controlled analysis of retrieval be-
havior. Our experiments show that the Semantic-
Answerability Gap stems from semantic accumula-
tion, schema-level cue dependence, and weak row-
column binding, rather than superficial artifacts
such as query phrasing or serialization format.
We further adoptAnswerability-Aware
Reranking (AAR), a lightweight two-stage
pipeline that reintroduces explicit answerability
judgment, improving top-1 retrieval from18.2%
to57.4%. This result suggests that a substantial
portion of the failure stems from missing answer-
ability verification rather than insufficient model
capacity alone.
Our contributions are threefold.(1) We iden-
tify and formalize the Semantic-Answerability
Gap:to our knowledge, the first work to define
answerability as distinct from semantic relevance,
and to show that, on our sibling-table setting, the
dense retrievers we evaluate frequently conflate the
two. Within our controlled sibling-table setting,
we trace this gap to three mechanisms: seman-
tic accumulation, schema-level cue dependence,
and weak row-column binding.(2) We introduce
TCR-Bench:a sibling-table benchmark purpose-
built to measure content-level answerability, isolat-
ing it from coarse semantic relevance and enabling
controlled analysis of retrieval behavior.(3)We
introduce AAR as an answerability-oriented di-
agnostic tool:showing that explicit answerability
verification substantially closes the gap, suggest-
ing content-aware retrieval as a promising direction
worth further investigation.
2 Related Work
2.1 Beyond Semantic Relevance in Retrieval
Recent work has identified intrinsic limitations of
embedding-based retrieval beyond coarse seman-
tic similarity. Theoretical analyses show that sin-
gle vector representations have fundamental ex-
pressivity limits (Weller et al., 2025; Vangara andGopinath, 2026), with diminishing returns from
scaling (Killingback et al., 2026) and degeneration
effects in practice (Guo et al., 2024).
Semantic similarity alone is insufficient in cer-
tain settings, with (Belfathi et al., 2026) empirically
revealing performance gaps on difficult instances
without explicitly attributing them to answerabil-
ity. Other works incorporate robustness (Liu et al.,
2026), QA accuracy (Nian et al., 2025), or multi-
hop reasoning (Trivedi et al., 2023; Asai et al.,
2024), yet none define answerability as a dimension
separate from semantic relevance. Our work for-
malizes this as theSemantic-Answerability Gap:
retrieved documents may be semantically similar
to the query while only one contains the specific
evidence needed.
2.2 Table RAG Methods and Benchmarks
Table retrieval has progressed from table-specific
dense retrievers (Herzig et al., 2021) to generic
ones like DPR (Karpukhin et al., 2020), which
(Wang et al., 2022) show can match specialized
architectures. Thus, structural bias alone is in-
sufficient for fine-grained content distinction, and
retrieval performance on existing benchmarks is
largely driven by semantic relevance. Simpler
schema-aware and title-aware representations (Kim
et al., 2024; Khanna and Subedi, 2025) prove ef-
fective, confirming that coarse relevance signals
dominate current settings. Hybrid and multi-table
pipelines (Chen et al., 2024b; Zhang et al., 2025a;
Zhang and Chen, 2025) similarly optimize for rele-
vance matching.
Existing benchmarks (Herzig et al., 2021; Zhang
and Balog, 2018; Cui et al., 2025; Ji et al., 2024;
Strich et al., 2026; Xu et al., 2025; Zou et al., 2025)
increase difficulty through multi-hop reasoning,
mixed modalities, or heterogeneous corpora, yet
consistently frame retrieval as semantic matching.
None requires distinguishing the uniquely answer-
able table among semantically similar candidates.
TCR-Bench targets this gap directly throughsib-
ling tables, isolating fine-grained answerability se-
lection from coarse semantic retrieval.
2

Datasets:
Spider
BIRDQuery
Original TableTarget Table
Sibling Distractor Construction Query ConstructionDrop colsFull resampling
Drop & resample
Target Cell
Other Cell in Target Table
Other Cell in Original
···
Embedding 
ModelsReranker 
Models
···top-k
Target Table Sibling Distractor Table
 Non-Sibling Table
high costTCR-Bench Construction Retrieval Performance on TCR-BenchFigure 2: Overview of the TCR-Bench pipeline and retrieval performance. The left panel shows its construction,
and the right panel shows that the embedding model can distinguish Sibling Distractor Tables from Non-Sibling
Tables but confuses Target with Sibling Distractor Tables, while the reranker correctly separates all three at a higher
computational cost.
3 TCR-Bench: A Diagnostic Benchmark
for Table Content-Level Answerability
in RAG
3.1 From Semantic Relevance to
Answerability
Existing Table RAG retrievers optimize for seman-
tic relevance, i.e., topical similarity between query
and table, but retrieval for generation requires
answerability. Broadly, answerability means a
source or collection of sources contains sufficient
evidence to answer a query. In table retrieval, we
operationalize it in the single-source setting as: a
table is answerable if it provides all required rows,
values, and attribute bindings to produce a non-
empty, constraint-consistent result.
We define three table types:Target Tables,
which satisfy all answerability criteria;Sibling
Distractor Tables, which are semantically simi-
lar but remove or violate answer-bearing evidence;
andNon-Sibling Tables, which are unrelated in
schema and semantics.
3.2 Operationalizing Answerability with
Controlled Sibling Tables
TCR-Bench (TableContent-level Answerability
Benchmark inRAG) is acontrolled diagnos-
tic benchmarkdesigned to isolate answerability-
aware retrieval from confounding factors such as
multi-table reasoning and generation-stage effects.
By design, it prioritizes diagnostic precision over
corpus breadth, providing a focused setting for iso-
lating whether retrievers can distinguish answer-
bearing evidence from semantically similar but
non-answerable content.
Each query is associated with one Target Ta-ble and multiple Sibling Distractor Tables derived
from the same source table. As shown in Figure 2,
Sibling Distractor Tables preserve strong semantic
overlap while removing answerability under the
query constraints, ensuring exactly one valid Tar-
get Table per query. While controlled, this design
reflects realistic scenarios such as partial exports,
schema projections, and filtered table versions.
Answerability-aware retrieval requires schema
grounding, condition matching, value coverage,
and row-column consistency. We use relatively
large tables (up to ∼8K tokens) to expose the ten-
sion between semantic similarity and content-level
answerability under compression pressure.
3.3 Benchmark Construction
3.3.1 Data Source and Task Design
TCR-Bench is built from two well-known Text-
to-SQL benchmarks, Spider (Yu et al., 2018) and
BIRD (Li et al., 2024), whose databases are derived
from real-world relational tables across diverse do-
mains. We consider TableQA with three query
types:Exact Match (EM),Selective Filtering
(SF), andSelective Aggregation (SA)(detailed
in Appendix A.3), with 1-3 conjunctive conditions
over target columns. Tables are rendered inMark-
down,HTML,CSV, andMixedformats.
Diagnostic subset.EM queries form a diagnos-
tic subset where retrieval is restricted to sibling
groups from the same source tables, isolating re-
trieval fidelity under minimal reasoning complexity
and enabling controlled analysis of schema and
query perturbations.
3

Qwen3-8B Qwen3-4B Qwen3-0.6B Stella Jina BGE-M3 GTE01020304050607080R / GR Score (%)
18.246.966.567.0
17.246.963.268.4
18.241.155.562.7
9.624.434.052.6
16.342.654.164.1
8.621.532.537.3
7.218.226.327.8R@1
R@3
R@5GR@1
DS@1
DS@1 random baseline
10%15%20%25%30%35%40%
DS@1 (%)
Random baseline = 29.8%Figure 3: Retrieval performance on TCR-Bench using Mixed Format. Abbreviated model names are used.
3.3.2 Sibling Distractor Construction
For each query, Sibling Distractor Tables are gen-
erated from the same source table by selectively
removing query-condition or target columns and
deleting or resampling rows violating query con-
straints. Query logic is then verified across the cor-
pus to confirm that each query has exactly one an-
swerable Target Table. To prevent shortcut match-
ing via surface overlap, we subsequently apply
paraphrased queries and schemas via GPT-OSS-
120B (OpenAI, 2025), along with numerical and
date perturbations. We further perform targeted
manual inspection over the most likely semanti-
cally ambiguous candidates outside each sibling
group, and remove cases that may introduce unin-
tended alternative answer tables; full details are in
Appendix A.
3.4 Benchmark Statistics
TCR-Bench contains 209 queries (98 EM, 55 SF,
56 SA) and 637 tables, with tables shared across
query types. Despite its moderate scale, difficulty
stems from dense, content-aware Sibling Distrac-
tor Tables and controlled perturbations rather than
corpus size. Full statistics are in Appendix A. Fu-
ture work will extend the framework to multi-table
and heterogeneous-source settings, where answer-
ability is compositional and evidence may span
modalities beyond structured tables.4 Primary Observations from Main
Experiments
4.1 Retrieval Evaluation
4.1.1 Embedding Models
We evaluate representative embedding models span-
ning different architectural families and parameter
scales, including Qwen3-Embedding-0.6B, Qwen3-
Embedding-4B, Qwen3-Embedding-8B (Zhang
et al., 2025c), stella_en_1.5B_v5 (Zhang et al.,
2025b), jina-embeddings-v4 (Günther et al., 2025),
bge-m3 (Chen et al., 2024a), and gte_Qwen2-7B-
instruct (Li et al., 2023), as shown in Figure3; their
abbreviations are used hereafter when unambigu-
ous. These models cover both general-purpose and
instruction-tuned embeddings, ranging from sub-
billion to multi-billion parameter scales. Detailed
descriptions are provided in Appendix B.
4.1.2 Evaluation Metrics
We evaluate retrieval from two complementary per-
spectives:semantic relevanceandanswerability.
Letyibe the ground-truth target and Rk
ithe top- k
retrieved items for queryi.
Top-kRecall ( R@k )(Ji et al., 2024; Zou
et al., 2025; Strich et al., 2026) measures re-
trieval performance at a fixed granularity: R@k=
1
NPN
i=1 I(yi∈Rk
i). In our inter-table retrieval
context, R@k serves as the indicator ofanswer-
ability retrieval, assessing whether the model pin-
points the exact evidence-bearing table.
Top-kGroup Recall ( GR@k )applies to
4

inter-table retrieval. Let Gidenote the Sib-
ling Table Group for query i. Inspired by
CR@k (Thakur et al., 2021), we define: GR@k=
1
NPN
i=1|Rk
i∩Gi|
min(|G i|,k). At the inter-table level,
GR@k maps tocoarse semantic relevance, mea-
suring whether the retriever lands in the correct top-
ical neighborhood; we focus primarily onGR@1.
Top-1 Discriminative Score ( DS@1 ):
DS@1 =R@1/GR@1 . This metric quantifies the
gap between semantic relevance and answerability
by measuring how well the model differentiates
the target within the retrieved semantic group.
Under random selection within a Sibling Table
Group, the expected DS@1 equals the reciprocal
of the average group size, which is 0.221 on the
original dataset and 0.298 excluding column-
deletion variants. A DS@1 near these bounds
indicates the retriever struggles with content-aware
answerability.
4.1.3 Main Results
Figure 3 reports retrieval performance under the
Mixedsetting. Full results are in Appendix D.1.
Across all models, a substantialSemantic-
Answerability Gapemerges: semantic relevance
is far easier to achieve than answerability. Even the
strongest model reaches GR@1 = 0.670 , yet its an-
swerability retrieval remains at only R@1 = 0.182 ,
revealing that successful semantic matching rarely
translates into reliably identifying the truly answer-
able table. Scaling model size improves consis-
tency but only partially addresses this bottleneck:
increasing Qwen3 from 0.6B to 8B raises R@5
by19.8% , yet top-1 improvements remain modest
relative to group-level semantic matching.
Most models score near or below the random-
selection baseline of 0.298 in DS@1 (with signifi-
cance analysis in Appendix O confirming that these
methods do not perform significantly better than
chance), indicating that fine-grained discrimination
among semantically matched sibling tables is not
significantly better than chance. As an additional
observation, in column-deletion cases that disrupt
schema structure, top-1 rates drop to 0.0718 for
Qwen3-4B and 0.0574 for Stella, indicating that
models do remain sensitive to explicit structural
mismatches. However, they struggle to distinguish
answerability differences within schema-consistent
tables. This suggests that the dense retrieval repre-
sentations we evaluate capture topical and schema-
level similarity effectively, but show limited sen-
sitivity to fine-grained evidence verification suchas precise row-column bindings, value constraints,
and condition consistency.
This pattern is also observed beyond stan-
dard dense embeddings: the alternative first-stage
paradigms we test, including DTR-style, ColBERT-
style, sparse, and hybrid retrieval, likewise do
not substantially close the answerability gap (Ap-
pendix D.2 and D.3)
4.2 Downstream QA Evaluation
To examine how retrieval quality propagates to gen-
eration, we evaluate downstream QA performance
using retrieved tables.
4.2.1 Experimental Setup
We evaluate the QA pipeline of
LongTableBench (Li et al., 2025) using Qwen3-
30B-A3B-Thinking-2507 (Team, 2025) as the
downstream model, reporting average F1 under
three primary settings: (1) Oracle (Target Table
only), (2) No-Table, and (3) Retrieved Top- k
Tables with k∈ {1,3,5} , where retrieved tables
are concatenated in retrieval order. This yields a
total of five experimental configurations.
To analyze how effectively downstream QA
utilizes retrieved evidence, we define Eff@k=
(QA@k/QA@Oracle)/R@k , where QA@k de-
notes QA F1 using Top- kretrieved tables. Intu-
itively, Eff@k measures whether the QA model
can effectively exploit the Target Table once it ap-
pears in the retrieved set, with values near 1 indi-
cating minimal interference from additional tables.
4.2.2 Downstream QA Results
Table 1 summarizes downstream QA performance.
The Oracle setting achieves F1 = 0.755 , while No-
Table drops to 0.014 , confirming that successful
answering depends heavily on retrieved tabular evi-
dence rather than parametric memorization. This
near-zero No-Table result also makes the efficiency
analysis more reliable, since downstream perfor-
mance is overwhelmingly determined by retrieval
quality.
Method QA@1 QA@3 QA@5 Eff@1 Eff@3 Eff@5
Qwen3-0.6B 0.1440.244 0.276 1.046 0.786 0.659
Qwen3-4B 0.1130.2690.329 0.871 0.760 0.690
Qwen3-8B 0.139 0.2660.330 1.014 0.751 0.657
Table 1: QA@k andEff@k using Top- kretrieved
tables from different embedding retrievers. The Oracle
(Target Table only) and No-Table settings achieve F1
scores of 0.755 and 0.014, respectively.
5

Although recall increases with larger k, down-
stream gains remain limited: even Top-5 retrieval
reaches only F1 = 0.330 . Meanwhile, Eff@1
remains close to 1, indicating near-optimal QA
once the Target Table is ranked first. In contrast,
Eff@3 andEff@5 exhibit a clear decreasing
trend as kincreases, with this pattern consistently
observed across all three model sizes. These sub-
stantial drops are consistent with prior observa-
tions that additional semantically related context
can introduce interference in RAG systems (Yoran
et al., 2024; Kim and Lee, 2024; Iratni et al., 2025;
Ouyang et al., 2025).
4.3 Summary and Reflections on the
Answerability Bottleneck
Our experiments reveal a systemic gap between
coarse semantic relevance and fine-grained an-
swerability in retrieval. While embedding mod-
els reliably locate the correct topical neighbor-
hood, DS@1 scores near or below the random-
selection baseline confirm that discriminating the
exact evidence-bearing table within a sibling group
remains highly unsolved. Downstream QA results
reinforce this picture: Eff@k degrades sharply
beyond k=1, underscoring that retrieval precision
at the top rank is the decisive factor for genera-
tion quality. Together, these results establish the
Semantic-Answerability Gap as a fundamental bot-
tleneck in Table RAG, which we investigate further
in subsequent experiments. We first ask whether
surface-level factors can account for this failure,
before probing deeper into the retrieval objective
itself.
The Answerability Bottleneck in Table RAG
The dense retrievers we evaluate achieve strong coarse
semantic relevance but exhibit near-chance fine-grained
answerability. ThisSemantic-Answerability Gapper-
sists across all evaluated models and scales.
5 Investigation I: Surface Variations Do
Not Explain Answerability Failures
The results in Section 4 reveal a large gap between
group-level retrieval and exact target identification.
A natural question is whether this failure reflects
sensitivity to surface-level artifacts, such as serial-
ization choices or query phrasing, or a more per-
sistent mismatch between the semantic retrieval
objective and answerability discrimination. We ex-
amine both factors on theDiagnostic Subsetusing
Qwen3-4B and Stella.5.1 Effect of Table Serialization Format
We evaluate six table formats:Markdown,CSV,
HTML,Mixed,Sen(Sentence), andSenS(Sen-
tence_Shuffle).Senconverts each row into a declar-
ative sentence while preserving row order;SenS
applies the same conversion but independently shuf-
fles the attribute order within each sentence, moti-
vated by prior evidence that models are sensitive to
column order (Cong et al., 2023).
As shown in Table 2, performance remains rel-
atively stable across all formats: for Qwen3-4B,
R@1 ranges from 0.186 to 0.257 and GR@1 from
0.757 to 0.857. Structured formats and sentence-
based representations yield comparable results, and
the similar performance between Sen and SenS
further indicates that attribute order within rows
contributes little to retrieval decisions. Consistency
analysis (Appendix E.2) confirms that models re-
trieve highly overlapping candidate sets across for-
mats, though internal ranking may vary.
Model FormatR@kGR@1 DS@1
@1 @3 @5
Qwen3-4BCSV 0.186 0.5710.771 0.814 0.228
Mixed 0.186 0.586 0.729 0.814 0.228
HTML 0.200 0.543 0.757 0.857 0.233
Markdown 0.2570.557 0.729 0.814 0.316
Sen 0.200 0.529 0.743 0.800 0.250
SenS 0.2430.6140.714 0.757 0.321
StellaCSV 0.171 0.414 0.543 0.671 0.255
Mixed 0.143 0.357 0.500 0.657 0.217
HTML 0.086 0.257 0.414 0.629 0.136
Markdown 0.171 0.386 0.514 0.629 0.273
Sen 0.314 0.5000.643 0.786 0.400
SenS 0.286 0.4710.671 0.771 0.370
Table 2: Retrieval performance under different table se-
rialization formats, showing largely stable performance
across formats.
5.2 Effect of Query Paraphrasing
We evaluate three query variants:Final Query
(NL), the natural-language query used in the main
benchmark;Sentential Query, which explicitly
expresses the same condition in sentence form; and
Template Query, a structured template emphasiz-
ing attribute-value constraints.
As shown in Table 3, retrieval performance
shows only moderate variation across formulations.
Consistency analysis (Appendix E.3) shows that
candidate sets remain largely stable under para-
phrasing even when the Target Table’s ranking
shifts, suggesting models capture logical intent but
lack the granularity to discriminate among struc-
turally similar tables.
6

Model Query TypeR@kGR@1 DS@1
@1 @3 @5
Qwen3-4BFinal (NL) 0.1860.586 0.729 0.814 0.228
Sentential 0.229 0.4710.729 0.800 0.286
Template 0.2570.571 0.714 0.786 0.327
StellaFinal (NL) 0.143 0.357 0.500 0.657 0.217
Sentential 0.100 0.3430.500 0.629 0.159
Template 0.100 0.329 0.471 0.714 0.140
Table 3: Retrieval performance under different query
formulations, showing largely stable performance across
query types.
Answerability Gap Transcends Surface Variations
Neither serialization format nor query phrasing explains
the observed retrieval failures on this diagnostic sub-
set. The stability of candidate sets across perturbations
suggests that the bottleneck lies in an objective-level mis-
match: the semantic retrieval objective handles topical
filtering well but is not well aligned with the fine-grained
answerability discrimination required to identify a unique
Target Table.
6 Investigation II: Semantic Retrieval
Objectives Are Associated with
Collapsed Answerability Distinctions
Having ruled out surface artifacts, we investigate
why the dense semantic retrievers we evaluate are
poorly aligned with answerability despite strong
group-level semantic matching. We identify three
patterns suggesting that the contrastive alignment
objective is associated with retrievers accumulat-
ing topical signal rather than verifying the precise
evidence required to answer a query.
6.1 Semantic Volume Bias: Evidence from
Sentential Queries
We probe whether retrieval is driven by semantic
coverage or logical sufficiency by converting table
rows into natural language sentences under three
transformations:(1) Sen: full row in original or-
der;(2) SenS: all column-value pairs with random-
ized phrase order;(3) SenSS: a shuffled subset of
column-value pairs sufficient to uniquely identify
the row. We report inter-table (Target Table se-
lection) and intra-table (row localization) retrieval
using Qwen3-4B and Stella on the Diagnostic Sub-
set under Mixed format.
Table 4 and Figure 4 reveal three patterns. First,
retrieval scales with semantic volume rather than
sufficiency: reducing queries to subsets degrades
accuracy if the subset uniquely identifies the row
(e.g., Qwen3-4B row-level R@1 drops 0.871 to
0.619). This suggests that contrastive training re-
wards broad topical coverage, not constraint com-pleteness. Second, the evaluated models are largely
insensitive to phrase order (Sen vs. SenS ≤0.02 ),
treating sentences as unordered token pools. Third,
row-level retrieval outperforms 10-line chunk re-
trieval (0.871 vs. 0.387 for Qwen3-4B), consis-
tent with length-induced embedding collapse (Zhou
et al., 2025). Despite high row-level R@1 (>0.85),
inter-table R@1 remains <0.26, indicating that lo-
cal key-value associations are captured but do not
translate into global answerability discrimination.
Model TypeR@kGR@1 DS@1
@1 @3 @5
Qwen3-4BSen 0.232 0.529 0.800 0.987 0.235
SenS 0.252 0.548 0.806 0.987 0.255
SenSS 0.226 0.490 0.703 0.865 0.261
StellaSen 0.213 0.5610.768 0.994 0.214
SenS 0.206 0.5550.774 0.987 0.209
SenSS 0.148 0.439 0.626 0.794 0.187
Table 4: Inter-table retrieval by query type: Sen/SenS
consistently outperform SenSS while phrase order has
minimal impact.
0% 20% 40% 60% 80% 100%
R@1 (%)Sen
SenS
SenSS
Sen
SenS
SenSS
Sen
SenS
SenSS
Sen
SenS
SenSS87.1
84.5
61.9
38.7
33.5
28.4
85.2
80.0
61.9
49.7
47.1
40.0Qwen3-
Row
Qwen3-
Chunk
Stella-
Row
Stella-
ChunkRow
Chunk
Figure 4: Intra-table retrieval by granularity and query
type: row-level outperforms 10-line chunks; SenSS de-
grades accuracy despite logical sufficiency. Qwen3:
Qwen3-4B. Full results in Appendix G.
6.2 Failure of Row-Column Binding Under
Column Shuffling
Using theDiagnostic Subset, we construct aMini-
mal Contrastive Setfor each table, consisting of
the Target Table and a column-shuffled distractor.
This setup follows the column-wise shuffling pro-
tocol in (Wang et al., 2022). We evaluate three con-
figurations (details in Appendix J): (1) Ori (original
table), (2) RID (each cell prefixed with its row in-
dex), and (3) RID+Shuf (row-indexed with column
shuffling applied).
As shown in Table 5, Ori achieves the strongest
discrimination (Qwen3-4B Mixed: R@1 = 0.443 ),
while RID+Shuf reduces accuracy to 0.314 and
7

row-ID augmentation provides only a modest im-
provement (0.371), indicating that explicit posi-
tional signals do not restore binding between values
and their column semantics. That column-shuffled
tables remain sufficiently similar to confuse re-
trieval suggests that the semantic retrieval objective
does not require row-column consistency to be pre-
served: aggregate topical overlap is sufficient to
produce near-identical retrieval scores. Full results
across all formats appear in Appendix I.
Model Config Format R@1 GR@1 DS@1
Qwen3-4B Ori Mixed 0.443 0.671 0.660
Qwen3-4B Ori Markdown 0.343 0.643 0.533
Stella Ori Mixed 0.229 0.471 0.485
Qwen3-4B RID Mixed 0.371 0.671 0.553
Stella RID Mixed 0.243 0.471 0.515
Qwen3-4B RID+Shuf Mixed 0.314 0.643 0.489
Stella RID+Shuf Mixed 0.200 0.429 0.467
Table 5: Retrieval on Minimal Contrastive Set. Col-
umn shuffling incurs only a modest penalty, indicating
that the retrieval objective does not enforce row-column
binding.
6.3 Conclusion: Systemic Failure Modes
Experiments on the models we evaluate reveal three
failure modes that point to potential limitations of
the LLM-based dense embeddings tested here for
tabular data.
Retrieval reward depends on token density
Performance scales with token density rather than
logical sufficiency, suggesting the embeddings we
test favor semantic volume over complete reason-
ing cues.
Dependence on schema-level cuesThe evalu-
ated models over-rely on high-level schema signals,
limiting discrimination among structurally similar
tables and consistent with absent relational super-
vision.
Weak row-column bindingsRow-column asso-
ciations are weak, as column-shuffled distractors
incur little penalty, suggesting these embeddings
act more like bag-of-phrases aggregators than struc-
tured relational representations.
Additional analyses of other factors (e.g.,
schema sensitivity and positional effects) are pro-
vided in Appendix K, L, and M.7 Answerability-Aware Reranking for
Table Retrieval
To examine the importance of explicit answerabil-
ity modeling at the retrieval stage, we introduce
Answerability-Aware Reranking(AAR), which
evaluates each candidate jointly with the query. Un-
like single-vector retrieval, cross-encoder interac-
tion enables direct assessment of whether a table
contains sufficient structured evidence to satisfy
the query, making answerability an explicit scor-
ing objective rather than an implicit byproduct of
semantic similarity.
We instantiate AAR with two variants: (1)AAR-
CE, a cross-encoder initialized from the same
Qwen3 family as the embedding retriever (Qwen3-
Embedding-8B with Qwen3-Reranker-8B), ensur-
ing comparable model capacity and identical initial-
ization lineage under different scoring paradigms;
and (2)AAR-Judge, an LLM-based binary answer-
ability judge (Balaka et al., 2025) using Qwen3-
30B-A3B-Thinking-2507 (Temperature=0).
Method R@1 R@5 GR@1 DS@1
AAR-CE 0.574 0.766 0.818 0.702
AAR-Judge 0.536 0.737 0.761 0.704
Table 6: Answerability-aware reranking on the Qwen3-
Embedding-8B backbone ( k=10 ). Adding query-
conditioned interaction substantially improves exact ta-
ble selection on TCR-Bench.
Both variants substantially improve retrieval
over the dense baseline. AAR-CE raises R@1
from 0.182 to 0.574 and GR@1 from 0.670 to
0.818; AAR-Judge achieves R@1 = 0.536 and
the best DS@1 = 0.704 . Because the reranker
and retriever share the same model family and
comparable initialization, these gains suggest that
explicit interaction-based answerability modeling
rather than differences in model capacity. Full re-
sults and significance analysis are in Appendix N
and O.
Answerability Modeling as a Core Retrieval Objective
Explicit answerability modeling substantially improves
exact target identification, suggesting that a major bottle-
neck in Table RAG retrieval lies in the lack of interaction-
based answerability assessment during the initial retrieval
phase. While AAR provides a practical mitigation, na-
tively incorporating answerability into the retrieval objec-
tive, rather than deferring it to a reranking stage, remains
a promising direction for future work.
8

8 Conclusion
TheSemantic-Answerability Gapidentifies a fun-
damental retrieval failure mode: high semantic rel-
evance does not guarantee that a retrieved source
contains sufficient evidence to answer a query. We
introduceTCR-Bench, a diagnostic benchmark
built around sibling tables, to study this gap in a
controlled setting where answerability can be pre-
cisely evaluated. Experiments show that the dense
retrievers we evaluate reliably identify semantically
relevant candidate groups yet struggle to identify
the uniquely answerable table within them, leading
to a sharp drop from oracle QA performance of
0.755 to 0.330 with top-5 retrieved tables. Con-
trolled analyses attribute this gap to three patterns
(Semantic Accumulation,Weak Row-Column Bind-
ing, andSchema-Level Cue Dependence), reflect-
ing the mismatch between contrastive alignment
objectives and fine-grained content verification. As
a diagnostic probe,AARsubstantially closes this
gap via explicit query-table interaction, suggesting
that much of the bottleneck reflects missing answer-
ability modeling rather than model scale or serial-
ization alone. This points to answerability-aware
retrieval as a direction worth further investigation,
beyond coarse semantic matching.
Limitations
To the best of our knowledge, this work is the first
to explicitly define and systematically study the
Semantic-Answerability Gap as a distinct retrieval
problem. The gap is defined under a controlled
setting where answerability can be isolated from
semantic relevance, but the concept extends beyond
tabular retrieval. In broader retrieval environments,
answerability may become less tractable: passages
may provide partial evidence, multi-source settings
distribute supporting facts across documents, and
verification tasks may admit no fully satisfying
source. These settings call for richer notions of
partial or compositional answerability, an open di-
rection for the field.
Due to resource constraints, our evaluation fo-
cuses on mainstream open-source embedding mod-
els spanning several architectural families and pa-
rameter scales, rather than exhaustively covering
all dense retrieval systems, including proprietary
and closed-source ones. The consistency of the
Semantic-Answerability Gap across these architec-
turally diverse models, together with its persistence
across the Qwen3 embedding family at scales from0.6B to 8B, leads us to expect the underlying find-
ings to generalize reasonably well, though confirm-
ing this across a wider range of retrievers remains
an important direction for future work.
At the benchmark level, TCR-Bench intention-
ally balances realism and controllability, preserv-
ing practically motivated properties such as schema
overlap and incomplete exports while constraining
the retrieval space for precise diagnostic analysis.
Real-world table ecosystems remain substantially
more complex, with noisier schemas, evolving ta-
bles, and heterogeneous evidence sources.
AAR mitigates rather than closes the Semantic-
Answerability Gap. Reranking improves target
identification via explicit query-table interaction,
suggesting answerability can be recovered with bet-
ter evidence sufficiency modeling. However, this
adds cost and depends on initial retrieval quality.
Overall, future systems should embed content- and
structure-aware representations earlier, balancing
efficiency and precise answerability.
Ethical Considerations
TCR-Bench is constructed entirely from publicly
available datasets (Spider and BIRD) and syntheti-
cally generated data, containing no personally iden-
tifiable or sensitive information. Semantic pertur-
bations and query templates were generated au-
tomatically and manually verified by the authors
to ensure quality and appropriateness. Large lan-
guage models were used solely for paraphrasing
queries and schemas, minor code generation as-
sistance, and minor manuscript polishing. To pro-
mote transparency and reproducibility, we plan to
publicly release all data, code, and evaluation pro-
tocols upon publication. While TCR-Bench is in-
tended to advance retrieval research in Table RAG,
practitioners applying it to sensitive domains (e.g.,
healthcare, finance) should incorporate appropriate
domain-specific safeguards.
References
Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to re-
trieve, generate, and critique through self-reflection.
InInternational conference on learning representa-
tions, volume 2024, pages 9112–9141.
Muhammad Imam Luthfi Balaka, David Alexander,
Qiming Wang, Yue Gong, Adila Krisnadhi, and Raul
Castro Fernandez. 2025. Pneuma: Leveraging llms
for tabular data representation and retrieval in an
9

end-to-end system.Proceedings of the ACM on Man-
agement of Data, 3(3):1–28.
Anas Belfathi, Nicolas Hernandez, Laura Monceaux,
Warren Bonnard, and Richard Dufour. 2026. Se-
mantic reranking at inference time for hard exam-
ples in rhetorical role labeling.arXiv preprint
arXiv:2605.18007.
Michael J Cafarella, Alon Y Halevy, Daisy Zhe Wang,
Eugene Wu, and Yang Zhang. 2008. Webtables: Ex-
ploring the power of tables on the web.Proc. VLDB
Endow., 1(1):538–549.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024a. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
Preprint, arXiv:2402.03216.
Peter Baile Chen, Yi Zhang, and Dan Roth. 2024b. Is ta-
ble retrieval a solved problem? exploring join-aware
multi-table retrieval. InProceedings of the 62nd an-
nual meeting of the association for computational lin-
guistics (volume 1: long papers), pages 2687–2699.
Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong,
Hong Wang, and William Yang Wang. 2020. Hy-
bridqa: A dataset of multi-hop question answering
over tabular and textual data. InFindings of the Asso-
ciation for Computational Linguistics: EMNLP 2020,
pages 1026–1036.
Tianji Cong, Madelon Hulsebos, Zhenjie Sun, Paul
Groth, and HV Jagadish. 2023. Observatory: Charac-
terizing embeddings of relational tables.Proceedings
of the VLDB Endowment, 17(4):849–862.
Lingxi Cui, Guanyu Jiang, Huan Li, Ke Chen, Lidan
Shou, and Gang Chen. 2025. Tablecopilot: A table
assistant empowered by natural language conditional
table discovery.Proceedings of the VLDB Endow-
ment, 18(12):5399–5402.
Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong
Yu. 2022. Turl: Table understanding through repre-
sentation learning.ACM SIGMOD Record, 51(1):33–
40.
Thibault Formal, Carlos Lassance, Benjamin Pi-
wowarski, and Stéphane Clinchant. 2021a. Splade
v2: Sparse lexical and expansion model for informa-
tion retrieval.arXiv preprint arXiv:2109.10086.
Thibault Formal, Benjamin Piwowarski, and Stéphane
Clinchant. 2021b. Splade: Sparse lexical and expan-
sion model for first stage ranking. InProceedings
of the 44th International ACM SIGIR Conference on
Research and Development in Information Retrieval,
pages 2288–2292.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.Xingzhuo Guo, Junwei Pan, Ximei Wang, Baixu Chen,
Jie Jiang, and Mingsheng Long. 2024. On the em-
bedding collapse when scaling up recommendation
models. InInternational Conference on Machine
Learning, pages 16891–16909. PMLR.
Michael Günther, Saba Sturua, Mohammad Kalim
Akram, Isabelle Mohr, Andrei Ungureanu, Sedigheh
Eslami, Scott Martens, Bo Wang, Nan Wang, and
Han Xiao. 2025. jina-embeddings-v4: Universal
embeddings for multimodal multilingual retrieval.
Preprint, arXiv:2506.18902.
Jonathan Herzig, Thomas Müller, Syrine Krichene, and
Julian Eisenschlos. 2021. Open domain question
answering over tables via dense retrieval. InPro-
ceedings of the 2021 Conference of the North Amer-
ican Chapter of the Association for Computational
Linguistics: Human Language Technologies, pages
512–519.
Jonathan Herzig, Pawel Krzysztof Nowak, Thomas
Müller, Francesco Piccinno, and Julian Eisenschlos.
2020. Tapas: Weakly supervised table parsing via
pre-training. InProceedings of the 58th annual meet-
ing of the association for computational linguistics,
pages 4320–4333.
Yizheng Huang and Jimmy Huang. 2024. A survey
on retrieval-augmented text generation for large lan-
guage models.arXiv preprint arXiv:2404.10981.
Malika Iratni, Mohand Boughanem, and Taoufiq Dkaki.
2025. Dynamic context selection for retrieval-
augmented generation: Mitigating distractors and
positional bias.arXiv preprint arXiv:2512.14313.
Xingyu Ji, Aditya Parameswaran, and Madelon Hulse-
bos. 2024. Target: Benchmarking table retrieval for
generative tasks. InNeurIPS 2024 Third Table Rep-
resentation Learning Workshop.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 conference on empirical methods in natural
language processing (EMNLP), pages 6769–6781.
Sujit Khanna and Shishir Subedi. 2025. Tabular embed-
ding model (tem): Finetuning embedding models for
tabular rag applications. InIntelligent Computing-
Proceedings of the Computing Conference, pages
448–460. Springer.
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR conference on research
and development in Information Retrieval, pages 39–
48.
Julian Killingback, Mahta Rafiee, Madine Manas, and
Hamed Zamani. 2026. Scaling laws for embedding
dimension in information retrieval.arXiv preprint
arXiv:2602.05062.
10

Kihun Kim, Mintae Kim, Hokyung Lee, Seong Ik Park,
Youngsub Han, and Byoung-Ki Jeon. 2024. Thorr:
Complex table retrieval and refinement for rag. In
IR-RAG@ SIGIR, pages 50–55.
Kiseung Kim and Jay-Yoon Lee. 2024. Re-rag: Improv-
ing open-domain qa performance and interpretability
with relevance estimator in retrieval-augmented gen-
eration. InProceedings of the 2024 Conference on
Empirical Methods in Natural Language Processing,
pages 22149–22161.
Carlos Lassance, Hervé Déjean, Thibault Formal, and
Stéphane Clinchant. 2024. Splade-v3: New baselines
for splade.arXiv preprint arXiv:2403.06789.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li,
Bowen Li, Bailin Wang, Bowen Qin, Ruiying Geng,
Nan Huo, and 1 others. 2024. Can llm already serve
as a database interface? a big bench for large-scale
database grounded text-to-sqls.Advances in Neural
Information Processing Systems, 36.
Liyao Li, Jiaming Tian, Hao Chen, Wentao Ye, Chao Ye,
Haobo Wang, Ningtao Wang, Xing Fu, Gang Chen,
and Junbo Zhao. 2025. Longtablebench: benchmark-
ing long-context table reasoning across real-world
formats and domains.Findings of the Association for
Computational Linguistics: EMNLP, 2025.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023. Towards
general text embeddings with multi-stage contrastive
learning.arXiv preprint arXiv:2308.03281.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024a. Lost in the middle: How language
models use long contexts.Transactions of the associ-
ation for computational linguistics, 12:157–173.
Peiyang Liu, Qiang Yan, Ziqiang Cui, Di Liang,
Xi Wang, and Wei Ye. 2026. Beyond semantic
relevance: Counterfactual risk minimization for ro-
bust retrieval-augmented generation.arXiv preprint
arXiv:2605.01302.
Tianyang Liu, Fei Wang, and Muhao Chen. 2024b. Re-
thinking tabular data understanding with large lan-
guage models. InProceedings of the 2024 Confer-
ence of the North American Chapter of the Associ-
ation for Computational Linguistics: Human Lan-
guage Technologies (Volume 1: Long Papers), pages
450–482.
Yuze Lou, Chuan Lei, Xiao Qin, Zichen Wang, Christos
Faloutsos, Rishita Anubhai, and Huzefa Rangwala.
2024. Datalore: Can a large language model findall lost scrolls in a data repository? In2024 IEEE
40th International Conference on Data Engineering
(ICDE), pages 5170–5176. IEEE.
Fatemeh Nargesian, Erkang Zhu, Ken Q Pu, and Renée J
Miller. 2018. Table union search on open data.Pro-
ceedings of the VLDB Endowment, 11(7):813–825.
Jinming Nian, Zhiyuan Peng, Qifan Wang, and Yi Fang.
2025. W-rag: Weakly supervised dense retrieval in
rag for open-domain question answering. InProceed-
ings of the 2025 International ACM SIGIR conference
on innovative concepts and theories in information
retrieval (ICTIR), pages 136–146.
OpenAI. 2025. gpt-oss-120b & gpt-oss-20b model card.
Preprint, arXiv:2508.10925.
Jie Ouyang, Tingyue Pan, Mingyue Cheng, Ruiran Yan,
Yucong Luo, Jiaying Lin, and Qi Liu. 2025. Hoh: A
dynamic benchmark for evaluating the impact of out-
dated information on retrieval-augmented generation.
InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume
1: Long Papers), pages 6036–6063.
Feifei Pan, Mustafa Canim, Michael Glass, Alfio
Gliozzo, and James Hendler. 2022. End-to-end table
question answering via retrieval-augmented genera-
tion.arXiv preprint arXiv:2203.16714.
Panupong Pasupat and Percy Liang. 2015. Composi-
tional semantic parsing on semi-structured tables. In
Proceedings of the 53rd Annual Meeting of the As-
sociation for Computational Linguistics and the 7th
International Joint Conference on Natural Language
Processing (Volume 1: Long Papers), pages 1470–
1480.
Stephen E. Robertson and Hugo Zaragoza. 2009.The
Probabilistic Relevance Framework: BM25 and Be-
yond. Now Publishers Inc. Foundations and Trends
in Information Retrieval.
Tao Shen, Guodong Long, Xiubo Geng, Chongyang
Tao, Tianyi Zhou, and Daxin Jiang. 2023. Large
language models are strong zero-shot retriever.arXiv
preprint arXiv:2304.14233.
Roee Shraga and Renée J Miller. 2023. Explaining
dataset changes for semantic data versioning with
explain-da-v.Proceedings of the VLDB Endowment,
16(6).
Ananya Singha, José Cambronero, Sumit Gulwani,
Vu Le, and Chris Parnin. 2023. Tabular representa-
tion, noisy operators, and impacts on table structure
understanding tasks in llms. InNeurIPS 2023 Second
Table Representation Learning Workshop.
Michael Stonebraker, Daniel Bruckner, Ihab F Ilyas,
George Beskales, Mitch Cherniack, Stanley B
Zdonik, Alexander Pagan, and Shan Xu. 2013. Data
curation at scale: the data tamer system. InCidr,
volume 2013.
11

Jan Strich, Enes Kutay Isgorur, Maximilian Trescher,
Chris Biemann, and Martin Semmann. 2026. T2-
ragbench: Text-and-table aware retrieval-augmented
generation. InProceedings of the 19th Conference of
the European Chapter of the Association for Compu-
tational Linguistics (Volume 1: Long Papers), pages
165–191.
Yuan Sui, Mengyu Zhou, Mingjie Zhou, Shi Han, and
Dongmei Zhang. 2024. Table meets llm: Can large
language models understand structured table data?
a benchmark and empirical study. InProceedings
of the 17th ACM International Conference on Web
Search and Data Mining, pages 645–654.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. Beir:
A heterogeneous benchmark for zero-shot evaluation
of information retrieval models. InThirty-fifth Con-
ference on Neural Information Processing Systems
Datasets and Benchmarks Track (Round 2).
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2023. Interleaving retrieval
with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of
the 61st annual meeting of the association for com-
putational linguistics (volume 1: long papers), pages
10014–10037.
Anirudh Bharadwaj Vangara and Ashwin Gopinath.
2026. The geometry of consolidation. Submitted
to NeurIPS 2026. arXiv version in this repository.
Blerta Veseli, Julian Chibane, Mariya Toneva, and
Alexander Koller. 2025. Positional biases shift as in-
puts approach context window limits.arXiv preprint
arXiv:2508.07479.
Zhiruo Wang, Zhengbao Jiang, Eric Nyberg, and Gra-
ham Neubig. 2022. Table retrieval may not necessi-
tate table-specific model design. InProceedings of
the workshop on structured and unstructured knowl-
edge integration (SUKI), pages 36–46.
William Webber, Alistair Moffat, and Justin Zobel. 2010.
A similarity measure for indefinite rankings.ACM
Transactions on Information Systems (TOIS), 28(4):1–
38.
Orion Weller, Michael Boratko, Iftekhar Naim, and
Jinhyuk Lee. 2025. On the theoretical limita-
tions of embedding-based retrieval.arXiv preprint
arXiv:2508.21038.
Chuan Xu, Qiaosheng Chen, Yutong Feng, and Gong
Cheng. 2025. mmrag: A modular benchmark for
retrieval-augmented generation over text, tables, and
knowledge graphs. InInternational Semantic Web
Conference, pages 3–21. Springer.Junwen Yang, Yeye He, and Surajit Chaudhuri. 2021.
Auto-pipeline: synthesizing complex data pipelines
by-target using reinforcement learning and search.
Proceedings of the VLDB Endowment, 14(11):2563–
2575.
Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan
Berant. 2024. Making retrieval-augmented language
models robust to irrelevant context. InICLR 2024
Workshop on Large Language Model (LLM) Agents.
Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga,
Dongxu Wang, Zifan Li, James Ma, Irene Li, Qingn-
ing Yao, Shanelle Roman, and 1 others. 2018. Spider:
A large-scale human-labeled dataset for complex and
cross-domain semantic parsing and text-to-sql task.
InProceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing.
Xiaohan Yu, Pu Jian, and Chong Chen. 2025. Tablerag:
A retrieval augmented generation framework for het-
erogeneous document reasoning. InProceedings of
the 2025 Conference on Empirical Methods in Natu-
ral Language Processing, pages 14074–14093.
Yijiong Yu, Huiqiang Jiang, Xufang Luo, Qianhui
Wu, Chin-Yew Lin, Dongsheng Li, Yuqing Yang,
Yongfeng Huang, and Lili Qiu. 2024. Mitigate posi-
tion bias in large language models via scaling a single
dimension.arXiv preprint arXiv:2406.02536.
Chi Zhang and Qiyang Chen. 2025. Hd-rag: Retrieval-
augmented generation for hybrid documents contain-
ing text and hierarchical tables.arXiv e-prints, pages
arXiv–2504.
Chi Zhang, Qiyang Chen, and Mengqi Zhang. 2025a.
Mixture-of-rag: Integrating text and tables with large
language models.arXiv preprint arXiv:2504.09554.
Dun Zhang, Jiacheng Li, Ziyang Zeng, and Fulong
Wang. 2025b. Jasper and stella: distillation of sota
embedding models.Preprint, arXiv:2412.19048.
Shuo Zhang and Krisztian Balog. 2018. Ad hoc table
retrieval using semantic similarity. InProceedings
of the 2018 world wide web conference, pages 1553–
1562.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025c. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Yuqi Zhou, Sunhao Dai, Zhanshuo Cao, Xiao Zhang,
and Jun Xu. 2025. Length-induced embedding col-
lapse in transformer-based models.
Jiaru Zou, Dongqi Fu, Sirui Chen, Xinrui He, Zihao
Li, Yada Zhu, Jiawei Han, and Jingrui He. 2025.
Rag over tables: Hierarchical memory index, multi-
stage retrieval, and benchmarking.arXiv preprint
arXiv:2504.01346.
12

A Benchmark Construction,
Augmentation, and Representation
A.1 Distractor Sampling Parameters
For the generation of sibling distractors, we adhere
to the following constraints:
•Length Constraint:All sub-tables are sam-
pled to a length of 6,500-7,800 tokens (via
tiktoken2).
•Augmentation Strategies:We employ three
augmentation strategies to construct distractor
tables:
–Removing Condition or Target
Columns:We remove either the
condition column or the target column
from the original table (only one column
is removed at a time) to prevent direct
retrieval of the answer while preserving
the remaining table structure.
–Row Deletion:We randomly delete
30%-70% of the rows from the original
table and then resample additional rows
from the remaining corpus until the re-
sulting table falls within the predefined
token length range.
–Row Resampling:We completely re-
sample rows from the table corpus to con-
struct a new table of the required length.
•Answer Filtering:For both the row deletion
and row resampling strategies, we explicitly
filter out any rows that would allow the query
to be answered correctly. Furthermore, we
conduct a global verification step across all
table groups to ensure that no distractor table,
including those sampled from other groups,
contains rows that could answer the current
query.
•Verification:Human experts corrected ap-
proximately 5% of the GPT-generated para-
phrases where semantic shifts occurred.
A.2 Representation Transformation Rules
To rigorously evaluate whether embedding mod-
els rely on surface-level token matching, we apply
three distinct types of transformations:
2https://github.com/openai/tiktokenA.2.1 Date Format Augmentation
Standard Y Y Y Y−MM−DD dates are trans-
formed into multiple representations:
•English Abbreviations:e.g.,15/Jan/2023,
Jan. 15, 2023.
•Full English Spellings:e.g.,January 15th,
2023.
•Roman Numerals:e.g.,2023-I-15, where
year, month, and day are in Roman numerals.
•Numeric Words:e.g.,two thousand twenty-
three. one. fifteen.
A.2.2 Numeric and Scale Perturbations
Numeric columns are augmented to create alterna-
tive representations:
•Scaling Large Numbers:Numbers in the
thousands are converted to compact forms
withkorKsuffix, e.g.,15000→15k.
•Multiplicative and Additive Shifts:Small
numeric perturbations such as multiplying/-
dividing by 10 or 100, or adding/subtracting
small constants.
•Numeral Systems Conversion:Numbers
are converted to Roman numerals or English
words, e.g., 102→ CorOne hundred and
two.
A.3 Task Definitions
We formalize three query-answer tasks of increas-
ing semantic complexity: Exact Match (EM), Se-
lective Filtering (SF), and Selective Aggregation
(SA). Each task is defined over a relational table
Twith schema S= (A 1, . . . , A k), where each
attribute Aitakes values from a discrete or con-
tinuous domain. A query Qcomprises a set of
selection predicates Pand, depending on the task
type, may additionally specify a target attribute At
and an aggregation functionf.
LetR⊆ T denote the set of rows that satisfy all
predicates inP, i.e.,
R={r∈ T | ∀p∈ P:p(r) =true}.
Exact Match (EM)In the EM setting, every
predicate in Pis a conjunction of equality con-
ditions of the form Ai=vi, where viis a literal
value from the domain of Ai. The query requests
the value(s) of a designated target column Atfrom
13

the rows in R. Since all predicates are equalities, R
is either empty or a singleton under the assumption
of a key-based uniqueness; in practice, the answer
is the unique value of Atin the matching row.Ex-
ample.For the table below, the query “What is C
when A = 5 and B = 3?” yields the unique answer
4.
Selective Filtering (SF)The SF task relaxes the
equality restriction: at least one predicate employs
a comparison operator from the set {<, >,≤,≥
,̸=}. Predicates may still include equalities, but
the filter condition Pis no longer constrained to
exact key lookups. The answer is the set of values
(or a single value, if the filter yields a singleton)
of the target column Atover the filtered rows R.
Example.The query “What is C when A = 5 and
B < 5?” filters rows by an equality on Aand an
inequality onB, returning4.
Selective Aggregation (SA)The SA task extends
SF by requiring a downstream aggregation oper-
ation. The query first applies the selection predi-
catesP(which may include comparisons) to obtain
the filtered set R, and then computes an aggregate
statistic over the values of a target column AtinR.
The aggregation function fis drawn from the stan-
dard set {SUM,A VG,MIN,MAX} . The answer
is the scalar result of f({r[A t]|r∈R}) .Exam-
ple.The query “What is the average of C when A ≥
5 and B≥3?” filters rows satisfying both inequal-
ities and returns the mean of the corresponding C
values, which is5.
ABC
534
686
Table 7: Illustrative relational table for the task defini-
tions.
These three tasks form a natural progression:
EM tests exact symbolic retrieval, SF evaluates
conditional reasoning with numerical comparisons,
and SA additionally assesses compositional reason-
ing over sets via aggregation. They are designed
to probe distinct cognitive and computational skills
required for table-based question answering.
A.4 Choice of GPT-OSS-120B for Semantic
Augmentation
We use GPT-OSS-120B (OpenAI, 2025), an open-
weight model released by OpenAI under theApache 2.0 license, for query and schema para-
phrasing. At the time of benchmark construc-
tion, it was among the strongest locally deployable
open-weight models, offering full reproducibility
without reliance on proprietary API endpoints and
strong instruction-following capability, which is
well-suited for controlled paraphrase generation.
To mitigate potential biases that this GPT-based
model might introduce, all subsequent components,
including dense retrievers, rerankers, and down-
stream QA models, are explicitly chosen from non-
GPT families.
A.5 Manual Validation and False-Negative
Reduction
To improve annotation quality and reduce false
negatives, we perform a multi-stage manual valida-
tion procedure over both query answerability and
schema transformations.
First, uniqueness is guaranteed within each sib-
ling group by construction: only the Target Table
satisfies all query constraints. We then apply the
query logic globally to the full corpus to eliminate
surface-level duplicate answerability.
To further reduce semantic false negatives, we
conduct targeted manual inspection over the most
likely ambiguous candidates. Specifically, for each
query, we retrieve the top- kcandidate tables out-
side the corresponding sibling group using multi-
ple similarity measures. The candidate sets from
different retrieval signals are merged, and the re-
sulting tables are manually inspected for semantic
equivalence. This process focuses on cases where
different schemas or column names may express
the same underlying concept, potentially leading to
unintended alternative answer tables. Ambiguous
cases are removed from the benchmark.
In addition, we manually verify the validity of all
queries and schema transformations introduced dur-
ing data construction. This includes checking that
rewritten queries preserve the original intent and
constraints, and that schema-level transformations
remain semantically consistent with the source ta-
bles without introducing annotation artifacts or un-
intended shortcuts.
While it is impossible to completely eliminate
every potential false negative in the benchmark,
this validation pipeline substantially reduces the
risk to a controllable level. More importantly, our
main experimental conclusions remain robust to the
residual noise: the performance gap between first-
stage retrieval and answerability-oriented reranking
14

Model Formatd eff¯d
Qwen3-Embedding-4B Mixed 55.947 0.5257
Qwen3-Embedding-4B CSV 45.154 0.4818
Qwen3-Embedding-4B HTML 58.698 0.5357
Qwen3-Embedding-4B Markdown 56.038 0.5394
Qwen3-Embedding-0.6B Mixed 50.795 0.4943
stella_en_1.5B_v5 Mixed 43.447 0.5491
jina-embeddings-v4 Mixed 49.654 0.3958
Table 8: Geometric indicators of the embedding space across models and formats
is still large and statistically significant across eval-
uations (see Appendix O for detailed significance
analysis).
A.6 Geometric Characteristics of the
Embedding Space
We analyze the geometric properties of table-level
embeddings produced by different models, follow-
ing the diagnostic framework introduced in (Van-
gara and Gopinath, 2026). The analysis focuses
on two key indicators of the embedding space:ef-
fective dimensionality( deff) andmean pairwise
cosine distance( ¯d).
Effective dimensionality, defined as the partici-
pation ratio of the covariance spectrum, measures
how many independent directions carry meaningful
variance in the embedding space. A lower value in-
dicates more concentrated information. The mean
pairwise cosine distance captures the overall sep-
aration between table embeddings in the vector
space, with higher values reflecting greater seman-
tic distinctiveness.
We computed these two metrics for 637 data
tables across six model/format combinations, with
embedding dimensions ranging from 1024 to 2560.
Results are shown in Table 8.
Two observations emerge from the results:
1.The embedding space exhibits clear inter-
table separation.Mean pairwise cosine dis-
tances range from 0.3958 to 0.5491 across
all models, indicating that even structurally
similar tables remain well-separated in the
embedding space.
2.Effective dimensionality is consistently
high.All models exhibit deffvalues between
43 and 59, substantially higher than the typi-
cal values reported for natural language text
(≈16) in (Vangara and Gopinath, 2026). Thissuggests that table-level embeddings encode
semantically rich and diverse information that
cannot be fully captured by a small number of
principal directions.
Taken together, these two indicators character-
ize our embedding space as well-separated and
information-diverse. This geometric profile pro-
vides a useful reference for interpreting retrieval
behavior.
B Detailed Description of Embedding
Models
This appendix summarizes the embedding model
families evaluated in the main experiments. Since
many table-specific embedding models, like
DTR (Herzig et al., 2021), often use very limited
context, and studies such as Wang et al. (Wang
et al., 2022) have shown that table-specific embed-
dings do not necessarily outperform general text
embeddings in the Table RAG domain, we focus
only on commonly used, open-source, and rela-
tively strong embedding models, all of which are
capable of handling 8K context windows.
•Qwen3 Embedding Family (Zhang et al.,
2025c).TheQwen3embedding series in-
cludesQwen3-0.6B,Qwen3-4B, andQwen3-
8B. These instruction-aware text embedding
models support long contexts (up to 32k to-
kens), multilingual input (100+ languages),
and Matryoshka Representation Learning
(MRL) for flexible embedding dimensions.
Larger variants generally provide stronger se-
mantic representation quality.
•Stella (Zhang et al., 2025b).
stella_en_1.5B_v5is a 1.5B-parameter
English sentence embedding model optimized
for semantic similarity and dense retrieval. It
15

Retrieval Instructions
Given a TableQA query, retrieve a relevant table that can answer the query. Note that the retrieved
table should contain sufficient information to provide an answer, rather than resulting in an empty
answer. [Context-Specific Task Description]
Table 9: Instruction prepended to user queries for the retrieval stage.
produces fixed-length embeddings and serves
as a competitive mid-scale dense baseline.
•Jina (Günther et al., 2025). jina-
embeddings-v4is a ∼3.8B-parameter mul-
tilingual and multimodal embedding model.
It supports unified text-image representations,
as well as both single-vector dense and multi-
vector late-interaction retrieval.
•BGE-M3 (Chen et al., 2024a). bge-m3is a
multilingual embedding model designed for
dense, sparse (lexical), and multi-vector re-
trieval within a unified architecture. It sup-
ports long inputs (up to 8192 tokens) and en-
ables hybrid retrieval pipelines.
•GTE (Li et al., 2023). gte_Qwen2-7B-
instructis a 7B-parameter instruction-tuned
embedding model from the GTE series. It is
optimized for semantic matching and large-
scale dense retrieval.
Model Diversity.The selected models differ
in parameter scale (0.6B–8B+), modality support
(text-only vs. multimodal), and retrieval design
(dense-only vs. hybrid/multi-vector); however, we
utilize only text-only dense retrieval in this work.
C Prompts Used in Our Pipeline
C.1 Retrieval Prompt
Since we utilize the Qwen3-Embedding series,
which is an instruction-aware retrieval model, we
prepend a specific instruction to the user query to
guide the embedding space towards TableQA rel-
evance. The exact instruction used is detailed in
Table 9. Note that traditional lexical methods, such
as BM25 (Robertson and Zaragoza, 2009), do not
utilize this instruction.
C.2 Downstream Question Answering Prompt
For the RAG-based generation stage, we adopt
a structured prompt following the style ofLongTableBench. This prompt enforces strict for-
matting constraints, including numerical normal-
ization (Roman to Arabic), handling of derived
columns (e.g., reversing mathematical operations),
and date standardization.
As shown in Table 10, the system prompt ensures
that the model outputs are parsable and normalized
across diverse table schemas.
D Comprehensive Retrieval Results
D.1 Dense Embedding Retrieval Results
Table 11 presents the complete benchmark results
for all evaluated models across all supported for-
mats.
D.2 Alternative Retrieval Paradigms
Beyond standard dense embedding retrieval, we fur-
ther evaluate several alternative retrieval paradigms,
including table-specific retrievers (DTR) (Herzig
et al., 2021), ColBERT-style late-interaction re-
trieval (Khattab and Zaharia, 2020), sparse lexi-
cal retrieval (BM25 and SPLADE (Formal et al.,
2021b,a; Lassance et al., 2024)), symbolic-neural
hybrid retrieval, and row-level multi-vector re-
trieval.
Model DescriptionsFor DTR, we evaluate
two variants: tapas_nq_retriever_large and
tapas_nq_hn_retriever_large3. The “nq”
suffix indicates the model is fine-tuned on Natural
Questions (Herzig et al., 2021), while “hn”
denotes training with hard negative sampling. For
ColBERT-style retrieval, we note that the original
ColBERT architecture (Khattab and Zaharia, 2020)
is designed for passages and does not natively
support 8K-token contexts. We therefore adopt
three ColBERT-style models from HuggingFace
that have been extended or adapted for longer
sequences, namely Reason-ModernColBERT4,
3We use official DTR weights with our PyTorch adaptation;
other public adaptations (e.g., deepset/tapas-large-nq-reader)
yielded inferior results.
4https://huggingface.co/lightonai/
Reason-ModernColBERT
16

QA System Prompt
System Prompt:
### Requirements:
Please read the following table and then answer the questions based on the table. Organize your answers into a list of
strings, with each element being an answer item (The number of answer items is less than 11). If the question includes a
special requirement, such as outputting a dictionary, please fulfill that specific request. Otherwise, always output a list of
strings, even if there is only one answer item. Please place your answers between the```and```.
### Notes:
The table may include non-standard formats for numbers or dates. For numbers, formats may include Roman numerals,
English words, or scientific notation. Whenever possible, please convert these into Arabic numerals in your response.
Additionally, some columns are derived through mathematical operations. For example, total_add_10 indicates that the
values in this column are obtained by adding 10 to the original values. You should return the original values (i.e., subtract 10
from the current values). Furthermore, for any values ending with “k” or “K”, convert them into standard Arabic numerals.
Please convert all dates to the format %Y-%m-%d <other time elements> (in Arabic numerals) in your response.
Additionally, even if there are duplicate answer items, you need to output all of them.
# Example Output Format:
```[‘answer_item_1’, ‘answer_item_2’, ..., ‘answer_item_n’]```
Please do not generate any text after outputting the final answer.
User Prompt:
table information:
{table_infos}
Question:{query}
Answer:
Table 10: The structured prompt used for the downstream TableQA generation stage.
SauerkrautLM-Multi-ModernColBERT5, and
reason-colBERT-150M-GTE-ModernColBERT6,
as available open-source checkpoints without
further fine-tuning. For sparse lexical retrieval,
we consider both the classical term-matching
approach BM25 and its learned counterpart
splade-v3 (Lassance et al., 2024)7. For
row-level multi-vector retrieval, each table row
(concatenated with its header) is independently
embedded, and its similarity score with the
query is computed separately; row-level scores
are then aggregated into a single table-level
score. We experimented with several aggregation
strategies (mean, max, top- kaveraging) and
found max aggregation to perform best; we
therefore report only the max-aggregation results.
For hybrid retrieval, we perform a grid search
over dense-BM25 interpolation weights using
Qwen3-Embedding-4B as the dense retriever; the
best-performing configuration uses a dense:BM25
5https://huggingface.co/VAGOsolutions/
SauerkrautLM-Multi-ModernColBERT
6https://huggingface.co/fjmgAI/
reason-colBERT-150M-GTE-ModernColBERT
7https://huggingface.co/naver/splade-v3ratio of 9:1.
ResultsTable 12 summarizes the retrieval perfor-
mance across paradigms.
AnalysisAcross paradigms, DTR models exhibit
extremely low retrieval accuracy, with R@1 re-
maining below 1%. ColBERT-style late-interaction
retrievers substantially improve recall metrics but
still fail to reliably distinguish answerable tables
from structurally similar distractors. Between the
two sparse lexical methods, SPLADE consider-
ably outperforms BM25, achieving recall com-
parable to the ColBERT-style models, though its
DS@1 remains similarly limited. The symbolic-
neural hybrid setup achieves marginal improve-
ments over pure dense retrieval, but increasing the
BM25 contribution consistently degrades perfor-
mance. Row-level multi-vector retrieval emerges
as our strongest first-stage retrieval method over-
all, achieving the highest recall and GR@1 scores
among all paradigms; however, its DS@1 remains
low, showing that even the best first-stage retriever
struggles to isolate the truly answerable table from
structurally similar distractors.
17

Model FormatR@k GR@kDS@1
@1 @2 @3 @4 @5 @1 @2 @3 @4 @5
Qwen3-Embedding-0.6BCSV 0.139 0.273 0.378 0.469 0.526 0.608 0.605 0.579 0.561 0.539 0.228
Mixed 0.182 0.297 0.411 0.502 0.555 0.627 0.617 0.593 0.574 0.556 0.290
HTML 0.182 0.282 0.402 0.478 0.574 0.651 0.636 0.616 0.590 0.583 0.279
Markdown 0.153 0.306 0.383 0.455 0.555 0.603 0.596 0.573 0.548 0.532 0.254
Sen 0.144 0.297 0.397 0.478 0.545 0.641 0.624 0.596 0.573 0.560 0.224
SenS 0.124 0.287 0.373 0.483 0.541 0.641 0.622 0.585 0.567 0.559 0.194
Qwen3-Embedding-4BCSV 0.158 0.311 0.464 0.579 0.646 0.656 0.682 0.681 0.663 0.653 0.241
Mixed 0.172 0.321 0.469 0.560 0.632 0.684 0.708 0.686 0.670 0.662 0.252
HTML 0.134 0.287 0.474 0.584 0.679 0.713 0.725 0.708 0.696 0.687 0.188
Markdown 0.187 0.3400.5120.569 0.656 0.703 0.706 0.691 0.682 0.666 0.265
Sen 0.139 0.306 0.440 0.545 0.632 0.684 0.682 0.667 0.656 0.653 0.203
SenS 0.158 0.321 0.488 0.579 0.646 0.684 0.667 0.663 0.652 0.644 0.231
Qwen3-Embedding-8BCSV 0.124 0.301 0.445 0.545 0.627 0.646 0.636 0.632 0.611 0.610 0.193
Mixed 0.182 0.344 0.4690.6080.665 0.670 0.658 0.641 0.628 0.622 0.271
HTML 0.153 0.335 0.488 0.5980.684 0.699 0.689 0.683 0.679 0.674 0.219
Markdown 0.177 0.349 0.493 0.565 0.632 0.660 0.648 0.646 0.634 0.629 0.268
Sen 0.148 0.297 0.431 0.531 0.593 0.636 0.639 0.611 0.603 0.596 0.233
SenS 0.177 0.306 0.407 0.507 0.560 0.632 0.632 0.619 0.604 0.597 0.280
bge-m3CSV 0.067 0.177 0.273 0.354 0.388 0.459 0.419 0.402 0.384 0.376 0.146
Mixed 0.086 0.177 0.215 0.282 0.325 0.373 0.352 0.338 0.334 0.316 0.231
HTML 0.043 0.139 0.225 0.263 0.292 0.349 0.347 0.346 0.341 0.330 0.123
Markdown 0.081 0.148 0.244 0.297 0.344 0.364 0.356 0.365 0.349 0.347 0.224
Sen 0.120 0.258 0.368 0.426 0.483 0.517 0.510 0.499 0.477 0.466 0.231
SenS 0.115 0.234 0.325 0.388 0.450 0.545 0.510 0.496 0.488 0.470 0.211
gte_Qwen2-7B-instructCSV 0.072 0.134 0.196 0.234 0.297 0.330 0.311 0.301 0.291 0.290 0.217
Mixed 0.072 0.139 0.182 0.211 0.263 0.278 0.263 0.252 0.236 0.240 0.259
HTML 0.038 0.086 0.124 0.172 0.211 0.244 0.239 0.231 0.227 0.228 0.157
Markdown 0.077 0.134 0.196 0.249 0.278 0.282 0.280 0.279 0.282 0.282 0.271
Sen 0.096 0.163 0.225 0.311 0.349 0.354 0.328 0.330 0.341 0.329 0.270
SenS 0.091 0.167 0.215 0.278 0.321 0.388 0.342 0.343 0.333 0.331 0.235
jina-embeddings-v4CSV 0.129 0.282 0.368 0.416 0.498 0.612 0.603 0.581 0.560 0.541 0.211
Mixed 0.163 0.306 0.426 0.483 0.541 0.641 0.634 0.609 0.573 0.547 0.254
HTML 0.1580.3680.474 0.584 0.646 0.689 0.687 0.667 0.644 0.634 0.229
Markdown 0.139 0.268 0.373 0.445 0.522 0.598 0.591 0.576 0.560 0.562 0.232
Sen 0.177 0.273 0.368 0.474 0.522 0.550 0.548 0.530 0.513 0.503 0.322
SenS 0.158 0.239 0.364 0.459 0.522 0.560 0.560 0.544 0.533 0.521 0.282
stella_en_1.5B_v5CSV 0.129 0.244 0.335 0.407 0.464 0.541 0.531 0.515 0.505 0.501 0.239
Mixed 0.096 0.172 0.244 0.287 0.340 0.526 0.476 0.448 0.417 0.405 0.182
HTML 0.048 0.105 0.172 0.220 0.258 0.349 0.330 0.309 0.297 0.292 0.137
Markdown 0.139 0.244 0.325 0.407 0.455 0.517 0.514 0.494 0.493 0.479 0.269
Sen 0.2060.306 0.397 0.502 0.550 0.641 0.615 0.609 0.598 0.582 0.321
SenS 0.163 0.292 0.397 0.493 0.560 0.632 0.632 0.617 0.602 0.588 0.258
Table 11: Comprehensive evaluation results across different input formats.
Model Type R@1 R@3 R@5 GR@1 DS@1
tapas_nq_hn_retriever_large DTR 0.010 0.014 0.019 0.038 0.250
tapas_nq_retriever_large DTR 0.005 0.010 0.010 0.005 1.000
Reason-ModernColBERT ColBERT-style 0.086 0.196 0.282 0.411 0.209
SauerkrautLM-Multi-ModernColBERT ColBERT-style 0.172 0.474 0.632 0.766 0.225
reason-colBERT-150M-GTE-ModernColBERT ColBERT-style 0.086 0.239 0.340 0.435 0.198
BM25 Sparse 0.005 0.005 0.014 0.005 1.000
splade-v3 Sparse 0.091 0.244 0.368 0.392 0.232
Qwen3-4B + BM25 hybrid Hybrid 0.215 0.455 0.612 0.689 0.312
Qwen3-0.6B Row-level multi-vector 0.263 0.488 0.627 0.699 0.377
Qwen3-4B Row-level multi-vector 0.249 0.507 0.679 0.732 0.340
Qwen3-8B Row-level multi-vector 0.282 0.512 0.660 0.694 0.407
Table 12: Retrieval performance across alternative retrieval paradigms.
18

Method Retrieval Model R@1 R@3 R@5 GR@1 GR@3 GR@5 DS@1
Meta-AugmentationQwen3-Embedding-0.6B 0.144 0.426 0.584 0.646 0.770 0.833 0.222
Qwen3-Embedding-4B 0.196 0.512 0.670 0.746 0.833 0.880 0.263
Qwen3-Embedding-8B 0.153 0.498 0.660 0.742 0.785 0.837 0.206
Query-to-TableQwen3-Embedding-0.6B 0.153 0.388 0.478 0.550 0.651 0.679 0.278
Qwen3-Embedding-4B 0.144 0.368 0.545 0.541 0.660 0.713 0.265
Qwen3-Embedding-8B 0.153 0.402 0.502 0.565 0.632 0.684 0.271
Table 13: Retrieval performance of augmentation strategies across different embedding models.
These results suggest that, in this setting, aug-
menting dense retrieval with symbolic lexical sig-
nals provides only limited complementary benefit.
More importantly, the same general failure pat-
tern appears across dense, late-interaction, sparse,
hybrid, and row-level paradigms, which points to
a bottleneck that may not be purely architecture-
specific. Instead, a substantial part of the difficulty
likely arises from the intrinsic ambiguity among
structurally similar tables.
D.3 LLM-Augmented Dense Retrieval
We further evaluate several augmentation strategies
commonly used in retrieval pipelines to determine
whether the performance gap observed in the main
experiments can be mitigated by adding auxiliary
semantic signals.
Experimental SetupInspired by prior
work (Gao et al., 2023; Shen et al., 2023),
we evaluate two semantic augmentation strategies
using Qwen3-30B-A3B-Thinking-2507:
•Meta-Info Generation.For each table, we
generate descriptive metadata summarizing
table themes, column semantics, and value
distributions. The generated metadata is con-
catenated with the table content before embed-
ding.
•Query-to-Table (Q2T).Given a query, the
model first generates a hypothetical “answer
table”. Retrieval is then performed by match-
ing this generated table against candidate ta-
bles in the corpus.
ResultsTable 13 summarizes the results under
the Qwen3-Embedding-8B retrieval backbone.
AnalysisBoth semantic augmentation methods
provide only limited improvements over the base-
line dense retrievers. Meta-information genera-
tion slightly improves recall metrics but does notsubstantially improve the ranking of Target Ta-
bles. The Query-to-Table strategy frequently in-
troduces hallucinated structures in the generated
tables, which inject additional noise into the re-
trieval process.
This observation suggest that simply enriching
the semantic representation of tables is insufficient
to resolve ambiguity among structurally similar
tables. In contrast, the reranking approaches pre-
sented in the main text (AAR) are more effective
because they enable direct query-table interaction
during scoring.
D.4 Retrieval and Downstream QA
Performance by Query Type
To further characterize where retrieval and reason-
ing difficulties arise, we report a breakdown of re-
trieval and downstream QA performance by query
type using Qwen3-Embedding-8B as the retriever.
The breakdown reveals three main patterns.
First, EM is the easiest query type overall, achiev-
ing the highest group recall, exact retrieval, and
downstream QA performance among the three
types. Even so, its DS@1 remains below or near
a random baseline, indicating that a persistent Se-
mantic Answerability Gap is present even for the
easiest queries. Second, SA is the hardest type for
exact answerability discrimination, obtaining the
lowest R@1 and DS@1 of all three types. Third,
coarse retrieval and within group discrimination
behave as distinct sources of difficulty: SF has the
lowest GR@1 among the three types, yet its DS@1
is higher than that of both EM and SA, showing that
difficulty in locating the correct semantic group is
not identical to difficulty in identifying the target
within that group once it has been located.
These results suggest that retrieval difficulty and
reasoning difficulty are related but not interchange-
able. Given SA’s low retrieval performance and
low Oracle score, the non monotonic Eff@k pattern
observed for this type should be interpreted with
caution, since the small effective sample size lim-
19

Table 14: Retrieval performance by query type using Qwen3-Embedding-8B.
Type R@1 R@3 R@5 GR@1 DS@1
All 0.182 0.469 0.665 0.670 0.271
EM 0.224 0.531 0.724 0.755 0.297
SF 0.182 0.418 0.600 0.564 0.323
SA 0.107 0.411 0.625 0.625 0.171
Table 15: Downstream QA performance by query type using Qwen3-Embedding-8B.
Type Oracle QA@1 QA@3 QA@5 Eff@1 Eff@3 Eff@5
All 0.755 0.139 0.266 0.330 1.014 0.751 0.657
EM 0.884 0.194 0.415 0.486 0.977 0.885 0.759
SF 0.711 0.129 0.180 0.204 0.998 0.605 0.479
SA 0.571 0.054 0.089 0.179 0.876 0.381 0.500
its the stability of downstream QA estimates. We
therefore report these figures together with the un-
derlying sample counts rather than drawing strong
conclusions from minor numeric variations.
E Consistency Analysis Across Surface
Variations
To better understand retrieval stability under
surface-level perturbations, we conduct a detailed
consistency analysis across both table serialization
formats and query formulations. The goal is to
determine whether retrieval errors stem from sensi-
tivity to superficial variations or from deeper repre-
sentational limitations.
E.1 Consistency Metrics
Beyond absolute accuracy, we evaluate four com-
plementary consistency metrics comparing re-
trieval outputs under different configurations:
•Hit Consistency: the proportion of instances
where top-1 retrieval outcomes (hit vs. miss)
are identical across two configurations.
•Partial Consistency: the proportion of cases
where the top-1 table from the base configu-
ration appears within the top-5 results of the
comparison configuration.
•Ranking Consistency: measured using Rank-
Biased Overlap (RBO, p= 0.9 ) (Webber
et al., 2010) between Top-5 ranking lists.
RBO emphasizes higher-ranked items while
accounting for list overlap.•Top-1 Identity: the percentage of cases where
the exact same table ID appears at rank 1 in
both configurations.
These metrics allow us to distinguish between
three possible scenarios: (1) identical retrieval be-
havior, (2) stable candidate sets with different rank-
ings, and (3) completely divergent retrieval outputs.
E.2 Serialization Format Consistency
Table 16 reports consistency metrics across serial-
ization formats using theMixedformat as the base
configuration.
Several observations emerge. First,Hit Consis-
tencyandPartial Consistencyremain high across
most format pairs, indicating that models retrieve
largely overlapping candidate sets regardless of se-
rialization conventions. Second, the particularly
strong consistency betweenSentenceandSen-
tence_Shufflesuggests that retrieval embeddings
are largely insensitive to row ordering.
However,Ranking Consistencyremains com-
paratively low (RBO around 20-30%), indicating
that while candidate tables remain similar, their
internal ranking positions frequently shift. This
pattern implies that serialization changes do not
disrupt coarse semantic matching but can influence
fine-grained similarity scoring.
E.3 Query Formulation Consistency
We also analyze retrieval stability under different
query formulations. Table 17 reports consistency
metrics using theFinal Queryas the base configu-
ration.
20

Model Config 1 vs. 2 Hit Consis. Ranking (RBO) Partial Consis. Top-1 Id.
Qwen3-4BMixed vs. CSV 85.71% 29.22% 97.14% 58.57%
Mixed vs. HTML 87.14% 27.76% 94.29% 45.71%
Mixed vs. Markdown 81.43% 28.47% 97.14% 48.57%
Mixed vs. Sentence 75.71% 25.88% 92.86% 38.57%
Mixed vs. Sentence_shuf 82.86% 25.06% 88.57% 41.43%
Sent. vs. Sent_shuf 84.29% 30.14% 94.29% 58.57%
StellaMixed vs. CSV 88.57% 24.55% 88.57% 52.86%
Mixed vs. HTML 82.86% 19.98% 71.43% 42.86%
Mixed vs. Markdown 88.57% 25.87% 92.86% 55.71%
Mixed vs. Sentence 77.14% 20.43% 82.86% 34.29%
Mixed vs. Sentence_shuf 74.29% 20.05% 78.57% 35.71%
Sent. vs. Sent_shuf 91.43% 34.43% 95.71% 77.14%
Table 16: Consistency analysis across serialization formats and narrative structures.
Model vs. Query Hit Consis. Ranking (RBO) Partial Consis. Top-1 Id.
Qwen3-4BTemplate 84.29% 31.67% 97.14% 58.57%
Sentential 84.29% 27.65% 90.00% 48.57%
StellaTemplate 95.71% 30.92% 94.29% 70.00%
Sentential 90.00% 28.46% 90.00% 60.00%
Table 17: Consistency analysis across query variations (Base: Final Query).
The results show that retrieval behavior remains
highly stable under query paraphrasing. In partic-
ular, the highPartial Consistencyindicates that
candidate tables remain largely invariant across
query formulations. Meanwhile, the moderateTop-
1 Identityscores suggest that ranking differences
arise primarily from subtle scoring shifts rather
than changes in semantic understanding.
Overall, these results confirm that query phras-
ing does not substantially alter the semantic inter-
pretation captured by the retrievers.
F Query Variant Definitions
We evaluate three query variants, as summarized in
Table 18.
For template queries, we use three sets of fixed
patterns, each targeting a different query intention:
•EM (Exact-Match Lookup):Retrieve target
values for rows satisfying conditions.Pat-
terns: "What is the {target} when
{conds}?" ,"Please find the {target}
for rows where {conds}." ,"List
the {target} for the entries with
{conds}." ,"Which {target} values
correspond to {conds}?"
•SF (Simple Filtering):Similar to EM, with
consistent emphasis on row-based filtering.
Patterns: "What is the {target} when
{conds}?" ,"List the {target} forrows where {conds}." ,"Which {target}
values correspond to rows where
{conds}?" ,"Please find the {target}
for entries with {conds}."
•SA (Simple Aggregation):Apply aggrega-
tion functions (e.g., COUNT, SUM, A VG)
over the target.Patterns: "What is the
{agg} of {target} when {conds}?" ,
"Compute the {agg} of {target} for
rows where {conds}." ,"Find the
{agg} value of {target} corresponding
to rows where {conds}." ,"Please
calculate the {agg} of {target} for
entries with {conds}."
The condition placeholder {conds} is instan-
tiated as conjunctions of attribute–value compar-
isons using symbolic operators (e.g., Year = 2016
AND Gold > 29).
G Detailed Intra-table Retrieval Results
For completeness, we report the full numerical re-
sults corresponding to Figure 4 in Table 19. The
table provides exact R@k values for both row-level
and chunk-level retrieval across different query
types.
21

Variant Format Example
Final NL Natural-language question “Which country won more than
29 gold medals in the 2016
Olympics?”
Sentential Conditions expressed as separate
sentences; target requested at the
end“The year is 2016. The num-
ber_of_gold_medals is greater
than 29. Country is?”
Template Structured templates with ex-
plicit placeholders for attributes,
values, and conditions“What is the Country when Year
= 2016 AND Gold > 29?”
Table 18: Summary of the three query variants.
Model Config Type R@1 R@3 R@5
Qwen3-4BRowSen 0.8710.9030.929
SenS 0.8450.916 0.929
SenSS 0.619 0.735 0.800
ChunkSen 0.387 0.587 0.742
SenS 0.335 0.561 0.723
SenSS 0.284 0.490 0.665
StellaRowSen 0.852 0.890 0.903
SenS 0.800 0.858 0.877
SenSS 0.619 0.729 0.755
ChunkSen 0.497 0.710 0.832
SenS 0.471 0.677 0.819
SenSS 0.400 0.645 0.768
Table 19: Intra-table retrieval by granularity and query
type: row-level achieves higher accuracy than chunk-
level (“Chunk” = 10-line segments); SenSS reduces
accuracy despite logical sufficiency.
H Comparison of 10-line and 20-line
Chunk Retrieval
Table 20 summarizes performance differentials be-
tween 10-line and 20-line chunk settings. The re-
sults indicate that increasing chunk size does not
yield consistent improvements. For Qwen3-4B,
R@1 rises from 0.387 (10-line) to 0.432 (20-line)
undersentence_full, yet shuffle variants show neg-
ligible or negative gains. Stella exhibits a simi-
lar pattern, with modest improvements undersen-
tence_fullbut instability across other conditions.
Empirical evidence further reinforces the granu-
larity paradox: retrieval accuracy saturates at fine
granularity and is weakened by contextual noise in
larger chunks.
I Full Retrieval Results Across All Table
Formats
Table 21 reports the complete retrieval results on
the Minimal Contrastive Set across all table for-
mats (CSV , HTML, Markdown, and Mixed) andconfigurations. These results extend the summa-
rized comparison in Table 5 and provide a more
detailed view of how format variations interact with
row-identifier augmentation and shuffled distrac-
tors.
J Row-Indexing and Column-Shuffling
Configurations
For each table, we generate a column-shuffled dis-
tractor table by independently permuting each col-
umn’s values. This preserves the topical semantics
of each column while disrupting row-level entity
alignment. We then evaluate three configurations:
1.Ori (Original Table):

A B
1 2
3 4

2.RID (Row-Indexed Table):each cell is pre-
fixed with its row index

A B
r0 : 1r0 : 2
r1 : 3r1 : 4

3.RID+Shuf (Row-Indexed with Column
Shuffling):column-shuffled on top of row-
indexing
A B
r0 : 1r1 : 4
r1 : 3r0 : 2

This setup allows us to systematically test
whether the model embeddings preserve relational
information at the row level while controlling for
column semantics.
22

Model Type 10-line R@1 20-line R@1∆ 10-line R@3 20-line R@3∆
Qwen3-4BSen 0.387 0.432 +0.045 0.587 0.619 +0.032
SenS 0.335 0.387 +0.052 0.561 0.619 +0.058
SenSS 0.284 0.271 -0.013 0.490 0.445 -0.045
StellaSen 0.497 0.516 +0.019 0.710 0.742 +0.032
SenS 0.471 0.516 +0.045 0.677 0.703 +0.026
SenSS 0.400 0.413 +0.013 0.645 0.561 -0.084
Table 20: 10-line vs 20-line Chunk Retrieval Comparison.
Model Config Format R@1 R@2 R@3 GR@1 DS@1
Qwen3-4BOri CSV 0.400 0.671 0.714 0.686 0.583
Ori Mixed 0.443 0.700 0.800 0.671 0.660
Ori HTML 0.386 0.686 0.800 0.671 0.574
Ori Markdown 0.343 0.671 0.729 0.643 0.533
StellaOri CSV 0.229 0.529 0.571 0.557 0.410
Ori Mixed 0.229 0.443 0.529 0.471 0.485
Ori HTML 0.257 0.414 0.514 0.443 0.581
Ori Markdown 0.214 0.514 0.586 0.500 0.429
Qwen3-4BRID CSV 0.357 0.729 0.757 0.629 0.568
RID Mixed 0.371 0.700 0.743 0.671 0.553
RID HTML 0.314 0.700 0.786 0.686 0.458
RID Markdown 0.400 0.657 0.743 0.657 0.609
StellaRID CSV 0.300 0.543 0.600 0.543 0.553
RID Mixed 0.243 0.429 0.500 0.471 0.515
RID HTML 0.200 0.329 0.343 0.329 0.609
RID Markdown 0.243 0.429 0.500 0.471 0.515
Qwen3-4BRID+Shuf CSV 0.271 0.629 0.686 0.629 0.432
RID+Shuf Mixed 0.314 0.686 0.729 0.643 0.489
RID+Shuf HTML 0.371 0.657 0.743 0.671 0.553
RID+Shuf Markdown 0.343 0.686 0.743 0.714 0.480
StellaRID+Shuf CSV 0.200 0.471 0.529 0.500 0.400
RID+Shuf Mixed 0.200 0.400 0.457 0.429 0.467
RID+Shuf HTML 0.100 0.286 0.329 0.329 0.304
RID+Shuf Markdown 0.186 0.414 0.514 0.429 0.433
Table 21: Full retrieval performance across all table formats and configurations.
23

K Sensitivity Analysis: Schema vs.
Cell-Level Signals
This experiment provides a preliminary observa-
tion that the retrieval models we test appear more
sensitive to schema-level modifications than to cell-
level changes. However, because column headers
consistently appear at the beginning of serialized
tables, this effect may partially reflect positional
bias rather than genuine schema sensitivity.
K.1 Experimental Design
From the Diagnostic Subset, we generate one
variant per table using three controlled pertur-
bations: (1)Content Perturbation (Cell-level),
where answer-relevant cells are replaced by other
valid entries from the same column while preserv-
ing schema and structure; (2)Schema Perturba-
tion (Header Rename), which renames a query-
relevant column header to a generic token (e.g.,
“Price” →“Column_2”) without modifying cell
values; and (3)Schema Perturbation (Column
Deletion), removing the column referenced by the
query to test logical dimension dependence. These
interventions preserve minimal semantic alignment
while altering either factual content or structural
anchors.
K.2 Results and Analysis
Table 22 (Mixed format) reports retrieval per-
formance under perturbation. The data reveal
pronounced schema dominance: for Qwen3-4B,
DS@1 rises from 0.578 under content perturbation
to 0.810 when a header is renamed. Higher DS@1
indicates a stronger preference toward the original
Target Table over its perturbed counterpart. Sen-
sitivity to schema changes substantially exceeds
sensitivity to cell-level modifications, indicating
that, for the models tested, retrieval similarity ap-
pears substantially more sensitive to schema-level
modifications than to factual cell-level changes.
Cell-level perturbation demonstrates markedly
reduced sensitivity to factual changes for the
evaluated models. Although modified tables no
longer satisfy query conditions, they are frequently
ranked at top-1 because the unchanged schema
preserves high semantic similarity. The logical
breakdown is further evidenced by column dele-
tion results (Stella DS@1=0.629), which exceed
content-perturbation sensitivity (0.611) but remain
dominated by schema effects.K.3 Key Conclusion
The embedding representations we evaluate exhibit
a schema-over-content bias: tabular understand-
ing appears anchored by headers rather than rela-
tional cell structures. Consequently, these mod-
els struggle to discriminate between tables sharing
identical schemas but differing in factual values.
Fine-grained cell-level discrimination appears lim-
ited for the embedding-based retrieval mechanisms
tested here, which may help explain persistent er-
rors in Table RAG scenarios that require structural
and factual reasoning.
We acknowledge that positional bias may par-
tially contribute to this observation, given that
schema tokens appear at leading positions in most
serialized formats. Appendix L confirms that front-
loaded content does confer a measurable retrieval
advantage. Nevertheless, positional effects alone
are unlikely to fully account for the observed phe-
nomenon, suggesting that schema dominance may
also reflect deeper representational limitations. Im-
portantly, both schema-over-content bias and po-
sitional sensitivity, as observed in the models we
test, point to deficiencies worth further attention in
embedding-based retrieval more broadly.
L Position Bias and the “Header
Preference”
This experiment investigates whether embedding
models exhibitposition biaswhen encoding tables.
Prior work on long-context language models has
shown that model performance can vary signifi-
cantly depending on where relevant information
appears in the input sequence, a phenomenon com-
monly referred to as thelost-in-the-middleeffect
(Liu et al., 2024a). Specifically, we test whether
the physical placement of a relevant row affects
retrieval similarity, even when the table content
remains unchanged.
L.1 Experimental Setup
We construct pairs of tables that aresemantically
identical and both target. The only difference
is theposition of the relevant row. The row is
placed in one of three positions:
•Top: immediately after the header
•Middle: within the body of the table
•Bottom: near the end of the table
We evaluate four comparison settings:
24

Configuration Model R@1 R@2 R@3 GR@1 DS@1
Content Perturb.Qwen3-4B 0.371 0.671 0.757 0.643 0.578
Stella 0.314 0.486 0.529 0.514 0.611
Rename HeaderQwen3-4B 0.486 0.671 0.814 0.600 0.810
Stella 0.386 0.514 0.557 0.500 0.771
Delete ColumnQwen3-4B 0.443 0.714 0.800 0.600 0.738
Stella 0.314 0.429 0.486 0.500 0.629
Table 22: Retrieval performance on Minimal Contrastive Sets under small-scale perturbations (Mixed format).
•Top vs Original (original row order in the
dataset)
• Top vs Middle
• Bottom vs Middle
• Top vs Bottom
For each comparison, the table listed first is
treated as the formal “Target” Table, although in
realityboth tables are valid.
The metricDS@1is used to measure model pref-
erence between the two tables. A valuegreater
than 0.5indicates that the model tends to rank
thefirst tablehigher, while a valuebelow 0.5indi-
cates preference for thesecond table. Thus, DS@1
directly reflects positional preference in these pair-
wise comparisons.
L.2 Results
Setting Model R@1 GR@1 DS@1
Top vs Original Qwen3-4B 0.557 0.686 0.813
Top vs Original Stella 0.314 0.514 0.611
Top vs Middle Qwen3-4B 0.571 0.686 0.833
Top vs Middle Stella 0.314 0.514 0.611
Bottom vs Middle Qwen3-4B 0.371 0.629 0.591
Bottom vs Middle Stella 0.300 0.514 0.583
Top vs Bottom Qwen3-4B 0.557 0.671 0.830
Top vs Bottom Stella 0.314 0.500 0.629
Table 23: Position bias evaluation under different row
placements. DS@1 >0.5 indicates preference for the
first table.
The results reveal a clear and consistentheader/-
top preference. When the relevant row is moved
to the top of the table, both models tend to assign
higher similarity scores to that table.
ForQwen3-4B, the effect is particularly strong.
In theTop vs OriginalandTop vs Middlecompar-
isons, DS@1 reaches0.813and0.833, indicating
a strong preference for tables where relevant evi-
dence appears immediately after the header. TheTop vs Bottomsetting further reinforces this ob-
servation (DS@1 =0.830), suggesting that early
placement significantly increases retrieval prefer-
ence.
TheBottom vs Middlecomparison provides ad-
ditional insight. When neither table places the rele-
vant row near the header, the preference becomes
weaker but still present (DS@1 =0.591). This sug-
gests that earlier placement within the table body
can still influence similarity scores, though less
strongly than header-adjacent placement.
ForStella, the pattern is more moderate but still
consistent. Across all four settings, DS@1 remains
between0.58and0.63, indicating a stable prefer-
ence toward the first table in each comparison. This
implies that the model also exhibits a positional
bias favoring earlier occurrences of relevant rows,
although the magnitude of the effect is smaller than
that observed in Qwen3-4B.
L.3 Discussion
Our analysis indicates that the embedding mod-
els we evaluate exhibit a noticeableposition bias
when encoding tables. This observation is consis-
tent with prior studies on long-context language
models, which show that models often favor in-
formation appearing near the beginning or end of
the input sequence while struggling to utilize in-
formation located in the middle (Liu et al., 2024a).
Rows placed closer to the header are more likely
to influence the embedding representation and thus
receive higher similarity scores during retrieval.
Notably, the preference for top-positioned rows
persists even when the alternative placement is only
moved to the middle of the table. This suggests
that positional signals play a meaningful role in
similarity computation. Similar positional effects
have been widely observed in transformer-based
models, where attention patterns and positional
encodings can introduce systematic biases toward
certain regions of the input sequence (Yu et al.,
2024).
25

At the same time, the effect remains weaker than
structural signals such as column-name alignment
observed in other perturbation experiments. In
other words, while schema-level cues dominate
retrieval similarity, row-level evidence is still influ-
enced by its physical placement within the table.
L.4 Implication
The experiment highlights that the embedding mod-
els we evaluate implicitly prioritize information
appearing near the beginning of a table. This be-
havior resembles the way models process textual
documents, where early content often receives dis-
proportionate influence in the final representation.
Such primacy effects have been reported in long-
context evaluation studies, where models tend to
prioritize information appearing at the beginning
of the context window (Liu et al., 2024a; Veseli
et al., 2025).
Consequently, the embeddings we test may rely
on positional heuristics rather than explicitly rea-
soning about whether the table contains the infor-
mation required to answer a query. This further
suggests that the table embedding models evaluated
here do not fully capture the relational structure of
tables and instead encode them in a manner closer
to structured text.
M Retrieval under Table Transposition
Prior studies in TableQA and table representation
learning have shown that table orientation can sub-
stantially affect model behavior, especially for ar-
chitectures relying on serialized table representa-
tions and positional encoding schemes (Liu et al.,
2024b; Sui et al., 2024; Singha et al., 2023). Moti-
vated by these observations, we investigate whether
table transposition influences retrieval performance
in our setting.
Specifically, we transpose each table by swap-
ping rows and columns while preserving all orig-
inal cell contents. The experiment is conducted
exclusively under the Mixed format setting.
Experimental SetupWe evaluate representative
dense embedding models on the transposed-table
corpus using the same retrieval protocol and evalua-
tion metrics as in the main experiments. To further
understand whether the observed sensitivity is spe-
cific to single-vector retrieval, we additionally eval-
uate the multi-vector retrieval approach introduced
in Appendix D.2 under the same transposed-table
setting.ResultsTable 24 reports the retrieval perfor-
mance on transposed tables.
Model R@1 R@3 R@5 GR@1 DS@1
Qwen3-0.6B 0.072 0.211 0.311 0.311 0.231
Qwen3-4B 0.081 0.273 0.392 0.450 0.181
Stella 0.038 0.163 0.258 0.301 0.127
Multi-vector (Qwen3-0.6B) 0.153 0.388 0.550 0.632 0.242
Multi-vector (Qwen3-4B) 0.201 0.455 0.603 0.689 0.292
Table 24: Retrieval performance on transposed tables
under the Mixed format setting.
AnalysisAll evaluated single-vector embedding
models show substantial performance degradation
after table transposition.
We attribute this decline to two factors. First,
original tables are largely row-oriented, where at-
tributes of the same entity remain locally grouped
in serialized sequences. Since most benchmark
queries are entity-centric, this structure helps em-
bedding models capture entity-level semantics. Af-
ter transposition, related attributes become dis-
persed, weakening local semantic coherence.
Second, transposition scatters schema informa-
tion across the sequence and disrupts the align-
ment between entity attributes and their surround-
ing context. This observation is consistent with
the “Header Preference” phenomenon discussed in
Appendix L.
Interestingly, although the multi-vector retriever
achieved strong results in Appendix D.2, its per-
formance also drops noticeably after transposition.
More importantly, its DS@1 score is no longer
higher than the random baseline. This result sug-
gests that the gains of the evaluated multi-vector
retrieval approach depend, at least in part, on the
row-oriented organization of information in the
original tables.
Overall, the orientation stress test indicates that
retrieval performance is influenced not only by ta-
ble semantics but also by structural layout and
serialization order. While multi-vector retrieval
remains more robust than single-vector retrieval
under transposition, the substantial performance
drop observed in our experiments suggests that
orientation sensitivity may extend beyond single-
vector representations and can also affect row-level
retrieval approaches.
26

Method Retrieval ModelR@k GR@kDS@1
@1 @2 @3 @4 @5 @1 @2 @3 @4 @5
Qwen3-0.6BARR-CE (T5) 0.297 0.435 0.507 0.545 0.555 0.737 0.703 0.641 0.596 0.556 0.403
ARR-CE (T10) 0.359 0.541 0.617 0.665 0.689 0.789 0.761 0.719 0.695 0.664 0.455
ARR-CE (T∞) 0.507 0.694 0.804 0.866 0.900 0.895 0.861 0.818 0.794 0.771 0.567
ARR-Judge (T5) 0.383 0.455 0.512 0.545 0.555 0.689 0.639 0.600 0.574 0.556 0.556
ARR-Judge (T10) 0.469 0.589 0.641 0.675 0.684 0.751 0.694 0.633 0.603 0.589 0.624
Qwen3-4BARR-CE (T5) 0.426 0.522 0.589 0.617 0.632 0.789 0.749 0.711 0.682 0.662 0.539
ARR-CE (T10) 0.498 0.598 0.651 0.694 0.722 0.828 0.801 0.777 0.757 0.735 0.601
ARR-Judge (T5) 0.445 0.550 0.598 0.612 0.632 0.746 0.725 0.692 0.673 0.662 0.596
ARR-Judge (T10) 0.517 0.622 0.689 0.703 0.718 0.804 0.766 0.721 0.688 0.676 0.643
Qwen3-8BARR-CE (T5) 0.507 0.608 0.636 0.660 0.665 0.770 0.730 0.684 0.653 0.622 0.658
ARR-CE (T10) 0.574 0.675 0.722 0.761 0.766 0.818 0.787 0.754 0.732 0.709 0.702
ARR-Judge (T5) 0.498 0.608 0.641 0.660 0.665 0.732 0.694 0.652 0.629 0.622 0.680
ARR-Judge (T10) 0.536 0.651 0.684 0.722 0.737 0.761 0.720 0.681 0.650 0.637 0.704
Table 25: Complete experimental results across retrieval configurations. Here, T iindicates that reranking is applied
to the topiresults returned by the embedding-based retrieval model.
N Full Experimental Results for AAR
In the following, T kdenotes that reranking is
applied to the top- kcandidates returned by the
embedding-based retrieval stage, while T ∞indi-
cates reranking over all retrieved samples. T ∞
serves as an empirical performance upper bound
for the reranking pipeline; due to its higher com-
putational cost, we report it only for the ARR-CE
configuration with Qwen3-0.6B. Table 25 provides
the complete experimental results across retrieval
backbones.
O Statistical Significance Analysis
Unless otherwise stated, all reported 95% confi-
dence intervals for DS@1 and p-values are esti-
mated via bootstrap resampling over queries ( N=
209, 1,000 samples, percentile method). To eval-
uate the statistical reliability of the observed im-
provements, we further conduct paired bootstrap
significance tests comparing reranking-based AAR
variants against the embedding-only Qwen3-8B
retrieval baseline.
ResultsTable 26 summarizes the statistical es-
timates for representative retrieval and reranking
models, while Table 27 reports the paired boot-
strap comparison between reranking-based AAR
variants and the Qwen3-8B baseline.
AnalysisThe statistical analysis supports the
main conclusions of the paper. Although several
first-stage retrieval paradigms achieve moderate
gains in recall-based metrics, their DS@1 improve-
ments remain statistically indistinguishable from
Random retrieval under paired bootstrap testing.In contrast, both AAR reranking variants pro-
duce large and statistically significant improve-
ments, with substantial effect sizes and highly sig-
nificant p-values. Compared with the embedding-
only Qwen3-8B baseline, reranking substantially
improves both R@1 and DS@1, demonstrating that
the primary limitation lies not in coarse retrieval re-
call itself, but in the inability of first-stage retrievers
to distinguish answerable tables from structurally
similar distractors. Query-aware reranking substan-
tially alleviates this issue.
P Summary of Experimental
Observations
Table 28 summarizes the major experimental ob-
servations regarding answerability-aware table re-
trieval and the Semantic-Answerability Gap, as
observed for the models evaluated in this study.
Q Benchmark Availability and
Open-Source Plan
The TCR-Bench has been publicly released
and is available at: https://github.com/
minger-hsxz/TCR-Bench-Open . The benchmark
is open-sourced under the CC BY-SA 4.0 license.
The current repository includes the core re-
sources required for reproducing the main experi-
ments in this paper, including:
• table data files,
• complete query files,
• files for the Diagnostic subset,
•and the full implementation of the RAG
pipeline used in this paper, including dense
27

Model Type R@1 DS@1 DS@1 (95% CI) p-value vs. Random
tapas_nq_retriever_large DTR 0.005 1.000 [0.000, 1.000] 0.350
tapas_nq_hn_retriever_large DTR 0.010 0.250 [0.000, 0.600] 0.628
SauerkrautLM-Multi-ModernColBERT ColBERT-style 0.173 0.225 [0.164, 0.291] 0.985
reason-colBERT-150M-GTE-ModernColBERT ColBERT-style 0.086 0.198 [0.116, 0.286] 0.982
Qwen3-4B + BM25 hybrid Hybrid 0.215 0.312 [0.239, 0.389] 0.369
Qwen3-Embedding-8B Dense 0.182 0.271 [0.200, 0.346] 0.748
Qwen3-Embedding-4B Dense 0.172 0.252 [0.185, 0.322] 0.904
stella_en_1.5B_v5 Dense 0.096 0.182 [0.117, 0.257] 0.999
AAR-CE Reranker 0.574 0.702 [0.636, 0.767]<0.001
AAR-Judge Reranker 0.536 0.704 [0.636, 0.774]<0.001
Table 26: Bootstrap confidence intervals and significance tests for representative retrieval methods.
Comparison∆R@195% CI∆DS@195% CIp-value
AAR-CE(T10)−Qwen3-8B +0.394 [0.321, 0.474] +0.431 [0.330, 0.529]<0.001
AAR-Judge(T10)−Qwen3-8B +0.353 [0.273, 0.435] +0.434 [0.336, 0.539]<0.001
Table 27: Paired bootstrap significance tests comparing reranking-based AAR variants with the embedding-only
Qwen3-8B retrieval baseline. Confidence intervals are computed from 1,000 bootstrap samples.
embedding retrieval, AAR reranking, and
downstream TableQA.
We will continue to expand and refine the repos-
itory with additional resources, including:
•intermediate files used during benchmark con-
struction,
• other benchmark variants used in the paper,
•and implementations based on additional re-
trieval architectures, such as DTR and Col-
BERT.
28

Observations Implications for Answerability Retrieval §
Best embedding retriever achieves only 18.2% Top-1 Tar-
get Table retrieval despite consistently retrieving tables
from the correct sibling groupDense retrieval captures coarse semantic neighborhoods but
fails to reliably identify the uniquely answerable table4.1
QA performance remains near oracle when the Target
Table is ranked first, but degrades rapidly as additional
retrieved tables are introducedAnswerability precision at top ranks is more important than
broad semantic recall; semantically related distractors inter-
fere with downstream reasoning4.2
Retrieval performance remains relatively stable across
Markdown, CSV , HTML, and sentence-based serializa-
tionsThe Semantic-Answerability Gap is not caused by surface
formatting or serialization artifacts5.1
Query paraphrasing changes ranking but preserves highly
overlapping candidate setsRetrievers capture general query intent but lack the fine-
grained discrimination required for answerability verification5.2
Reducing a query to a logically sufficient subset substan-
tially lowers retrieval accuracyRetrieval rewards semantic volume and token coverage rather
than minimal answer-bearing evidence6.1
Row-level retrieval exceeds 0.87 R@1, while table-level
Target Table selection remains below0.26Local key-value associations are captured, but do not scale
to reliable table-level answerability discrimination6.1
Column shuffling only moderately reduces retrieval accu-
racy despite breaking row-column semanticsEmbeddings weakly encode relational structure and fail to
robustly preserve row-column bindings required for answer-
ability6.2
Interaction-based reranking substantially improves exact
Target Table identificationExplicit query-table interaction restores answerability ver-
ification that is difficult to capture through single-vector
semantic retrieval alone7
Table 28: Summary of experimental observations on answerability-aware table retrieval and the Semantic-
Answerability Gap.
29