# Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model Selection

**Authors**: Max Nelson, Hanoz Bhathena, Aviral Joshi, Saket Sharma

**Published**: 2026-09-02 15:47:49

**PDF URL**: [https://arxiv.org/pdf/2609.02745v1](https://arxiv.org/pdf/2609.02745v1)

## Abstract
Selecting a retrieval model for a production RAG system requires reliable comparative evaluation, but obtaining relevance judgments at scale is expensive and difficult to repeat as new candidate systems arrive. We study pooled LLM evaluation, in which an LLM judges the union of documents retrieved by the current set of candidate systems, and the pool is then expanded incrementally as new systems are introduced by judging only the new documents they contribute. These judgments are reused to evaluate all systems on a common basis. We validate this approach on four retrieval benchmarks with 11 systems spanning dense, sparse, and hybrid configurations, and deploy it to compare 62 retrieval configurations for a financial news QA system. Pooled LLM rankings correlate strongly with gold-standard evaluation across datasets, and 97% of pairwise system orderings are preserved once bootstrap uncertainty in the qrels is taken into account. In production, document overlap yields 65-80% judgment reuse and up to 4.9x lower evaluation cost, allowing teams to benchmark new retrieval candidates without re-judging previously assessed documents. These results suggest pooled LLM evaluation is a practical and cost-effective workflow for incremental retrieval model selection in deployed systems.

## Full Text


<!-- PDF content starts -->

Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model
Selection
Max Nelson, Hanoz Bhathena, Aviral Joshi, Saket Sharma
JPMorganChase
max.nelson@jpmchase.com
Abstract
Selecting a retrieval model for a production
RAG system requires reliable comparative eval-
uation, but obtaining relevance judgments at
scale is expensive and difficult to repeat as
new candidate systems arrive. We study pooled
LLM evaluation, in which an LLM judges the
union of documents retrieved by the current set
of candidate systems, and the pool is then ex-
panded incrementally as new systems are intro-
duced by judging only the new documents they
contribute. These judgments are reused to eval-
uate all systems on a common basis. We vali-
date this approach on four retrieval benchmarks
with 11 systems spanning dense, sparse, and
hybrid configurations, and deploy it to compare
62 retrieval configurations for a financial news
QA system. Pooled LLM rankings correlate
strongly with gold-standard evaluation across
datasets, and 97% of pairwise system orderings
are preserved once bootstrap uncertainty in the
qrels is taken into account. In production, doc-
ument overlap yields 65–80% judgment reuse
and up to 4.9x lower evaluation cost, allowing
teams to benchmark new retrieval candidates
without re-judging previously assessed docu-
ments. These results suggest pooled LLM eval-
uation is a practical and cost-effective workflow
for incremental retrieval model selection in de-
ployed systems.
1 Introduction
Retrieval-augmented generation (RAG) pipelines
depend critically on the quality of the underlying
retrieval model. As new embedding models are re-
leased (OpenAI text-embedding-3 , Cohere Em-
bed v4, Nomic, Titan, etc.), practitioners face a
recurring decision:which model gives the best re-
trieval quality for my domain?
The gold standard for retrieval evaluation,
pooled human relevance judgments, as in
TREC (V oorhees, 1998), is expensive, slow, and
difficult to repeat as new candidate systems arrive.Evaluating Nsystems independently with an LLM
judge costs O(N·Q·K) judgments (queries ×
top-kdocuments per system), with substantial re-
dundancy since different retrieval systems often
surface overlapping document sets.
We evaluatepooled LLM evaluation. We judge
theunionof documents retrieved by all systems
under test, then reuse these judgments to compute
standard IR metrics for any system in the pool. As
new systems are added to the set, pools grow based
on only the new documents that those systems re-
trieve. This approach offers four advantages:
1.Cost amortization.Shared documents are
judged only once. Each additional system
added to the pool incurs only marginal cost
for its uniquely retrieved documents.
2.Ranking stability.By definition, system-
level P@ kand DCG@ kwill always be invari-
ant under pool expansion, and per-query pair-
wise orderings are preserved for recall@ k, AP,
and nDCG@ k. Empirically, macro-averaged
benchmark rankings remain highly stable as
new systems are added.
3.Reduced pool bias.Unlike classical TREC
pooling, where unjudged documents are
treated as non-relevant and systems with
novel retrieval behavior are penalised (Buck-
ley et al., 2007), pooled LLM judging can
assess every newly retrieved document con-
tributed to the pool. Teams can begin bench-
marking with as few as two or three systems
and add new ones over time, without system-
atic penalties for novel retrieval behavior due
to unjudged documents.
4.Accuracy.On four public IR benchmarks,
pooled LLM rankings correlate strongly with
gold-standard human judgments (nDCG@10
ρ= 0.69 –0.95), and 74% of pairwise ranking
arXiv:2609.02745v1  [cs.IR]  2 Sep 2026

disagreements fall within the gold standard’s
own bootstrap sampling noise.
2 Related Work
LLM-as-Judge for IR Evaluation.Recent stud-
ies have demonstrated that LLM judges (GPT-4,
Claude, etc.) produce relevance assessments that
agree with human annotators at rates comparable
to inter-annotator agreement (Faggioli et al., 2023;
Thomas et al., 2024).
Recent TREC-related studies have explored
LLM judges as part of the evaluation process and
found that LLM-based relevance or support la-
bels can yield system-level rankings that corre-
late well with rankings derived from human judg-
ments (Upadhyay et al., 2024b,a).
Our work builds on this foundation but addresses
concerns of efficiency, bias, and ranking stability
specifically as they pertain to industry practitioners.
We focus on how to structure LLM evaluation to
minimize cost while characterizing which aspects
of ranking stability follow from metric definitions
and which must be established empirically.
Test Collection Reusability.The reusability of
pooled judgments is a classical topic in IR eval-
uation. Zobel (1998) and V oorhees (1998) estab-
lished that TREC-style pools produce stable sys-
tem rankings even when new, unjudged systems are
evaluated. Buckley et al. (2007) analysed the con-
ditions under whichpool biasaffects system com-
parison: when a new system retrieves documents
absent from the original pool, those documents are
unjudged and effectively treated as non-relevant,
systematically disadvantaging novel retrieval strate-
gies.
Our LLM-based approach provides a structural
safeguard against this concern: because LLM judg-
ing is inexpensive, we judgealldocuments re-
trieved by any system, ensuring that no relevant
document goes unassessed.
Cost-Effective Evaluation.Prior work has ex-
plored reducing annotation cost through active
selection (Carterette et al., 2006), stratified sam-
pling (Pavlu and Aslam, 2007), and assessment
budgeting. Our approach is complementary: rather
than selectingwhichdocuments to judge, we judge
all pooled documents but reuse judgments across
systems, exploiting the natural overlap in retrieval
results.3 Method
3.1 Pooled LLM Evaluation
Given a set of Nretrieval systems {S1, . . . , S N}
and a query setQ, we:
1.Retrieve:For each system Siand query q∈
Q, obtain the top-kranked documentsR i(q).
2.Pool:Form the document pool P(q) =SN
i=1Ri(q).
3.Judge:For each unique (q, d) pair where
d∈ P(q) , obtain a graded relevance judg-
ment from an LLM judge on a 3-point scale
(0: not relevant, 1: partially relevant, 2: highly
relevant).
4.Evaluate:Compute standard IR metrics for
each system Siusing the shared pool judg-
ments restricted to documents inR i(q).
Critically, this process is incremental. A team
begins with an initial set of systems, builds the
pool, and later adds system SN+1 by retriev-
ing with it and judging only the new documents
RN+1(q)\ P(q) . All previously computed judg-
ments are reused exactly, and the new system is
immediately comparable to all existing ones on an
equal footing.
3.2 Ranking Stability Under Pool Expansion
We note the following stability properties, which
follow directly from metric definitions under the
assumption that previously assigned labels are not
revised and each system’s ranked list remains fixed.
P@kand DCG@ kdepend only on each system’s
own top- kdocuments and their (frozen) relevance
labels, so they are exactly invariant under pool ex-
pansion. Recall@ k, AP, and nDCG@ keach have
a system-specific numerator (fixed) divided by a
query-specific denominator shared across all sys-
tems and because all systems’ scores for a given
query are scaled by the same factor, pairwise order-
ings are preserved within each query. When scores
are macro-averaged across queries, the per-query
scaling factors may differ, so benchmark-level in-
variance is not guaranteed in general. We evaluate
this empirically in Section 5.4.
3.3 Cost Analysis
LetKdenote the top- kretrieval depth. Inde-
pendent evaluation of Nsystems requires up to
N·|Q|·K judgments. Under pooling, we judge

FiQA T-COVID NQ FinRAG-V
Queries 648 50 500*536
Corpus size 57K 171K 2.68M 10.9K
Pool size 249,865 18,870 188,903 192,783
nqrels 1,706 66,336 3,452 606
Rel/query (avg)∼2.6∼1,327∼1.0∼1.1
Relevance scale Binary 0-2 Binary Binary
Systems 11 11 11 11
Table 1: Dataset characteristics.*First 500 of 3,452 test
queries used for NQ.
only the unique (q, d) pairs in the union pool:
Cpool=P
q|S
iRi(q)|. Thereuse rate, 1−
Cpool/(N·|Q|·K) , measures the fraction of judg-
ments shared across systems. The marginal cost of
adding system N+1 is∆CN+1=P
q|RN+1(q)\
P(q)|, only its uniquely retrieved documents.
4 Experimental Setup
4.1 Datasets
We evaluate on four benchmarks with varying char-
acteristics (Details in Table 1). FiQA (Maia et al.,
2018), TREC-COVID (V oorhees et al., 2021), and
Natural Questions (NQ) (Kwiatkowski et al., 2019)
are sourced from the BEIR zero-shot retrieval
benchmark (Thakur et al., 2021); FinRAGBench-
V (Zhao et al., 2025) is a recent financial-domain
multimodal RAG benchmark. We employ a layout-
aware parser to extract text, tables, and figures
from each document page. Extracted tables and fig-
ures are independently processed by a multimodal
large language model (MLLM) to generate natu-
ral language descriptions, following the approach
described in Bhathena et al. (2026). The resulting
textual descriptions are then interleaved with the
extracted text spans in natural reading order, pre-
serving the original spatial layout of the document.
To retain modality provenance, table and figure de-
scriptions are enclosed in XML tags that explicitly
mark their source modality.
4.2 Retrieval Systems
We evaluate systems spanning five embedding
model families in three retrieval configurations:
dense(embedding-only retrieval with cosine simi-
larity),sparse(BM25 keyword retrieval), andhy-
brid(reciprocal rank fusion of dense and sparse
results). Embedding models: OpenAI text-
embedding-3-large , Cohere Embed v4, Amazon
Titan Embed v2, Nomic embed-text-v1.5 , and
Nomic embed-text-v2-moe . All embeddings are
projected to 256 dimensions via Matryoshka trunca-
tion where supported. This yields 5 dense systems,5 hybrid systems, and 1 sparse (BM25) baseline
per dataset.
4.3 LLM Judge
We use GPT-4.1 ( gpt-4.1-2025-04-14 ) as the rel-
evance judge, called at temperature 0.0 for near de-
terministic outputs. Each (q, d) pair is judged inde-
pendently in a single-turn prompt (no in-context ex-
amples, no system-level calibration), ensuring that
judgments are reproducible and order-independent.
Prompt Design.The prompt (Appendix C) in-
structs the judge to assign a graded relevance score
on a 3-point scale:0(not relevant - nothing in the
document addresses the query),1(partially rele-
vant - peripheral information is present but the core
intent remains unanswered), and2(fully relevant
- the document provides a complete answer). We
adopt a 3-point scale rather than binary to capture
the distinction between documents that are tangen-
tially related versus directly responsive, which is
critical for metrics like nDCG that weight graded
relevance. The prompt requires a brief chain-of-
thought (“thought” field, ≤2 lines) followed by
the numeric score in a structured JSON output, en-
couraging the judge to reason before scoring and
enabling post-hoc error analysis. Documents are
presented in a Title: . . . / Content: . . .
format, giving the judge a consistent, structured
view of each passage.
Judge Consistency.To assess intra-judge relia-
bility, we re-judged 1,000 randomly sampled (q, d)
pairs per dataset under identical conditions. Exact
3-point agreement was 93–96% across the BEIR
datasets ( κ= 0.84 –0.90), with ≥99.9% agreement
within ±1 grade. Under binarization (scores 1–2
= relevant), agreement rises to 96–98%. These re-
sults indicate that the judge is sufficiently stable to
support reuse of pooled judgments.
Cross-Judge Agreement.To assess sensitivity to
judge choice, we repeated the analysis with Claude
Sonnet 4.6 using the same prompt. On 1,000 ran-
domly sampled (q, d) pairs per dataset, binarized
agreement with GPT-4.1 was 84–92% ( κ= 0.65 –
0.77). To test whether these item-level differences
affect system comparison, we re-judged the full
TREC-COVID pool (18,870 documents) and re-
computed rankings. Despite 27% item-level dis-
agreement, system rankings remained highly cor-
related between judges ( ρ= 0.89 for nDCG@10,
0.92 for recall@100, and 0.88 for MAP), with 89%

pairwise agreement for both nDCG@10 and re-
call@100. Both judges identified the same top
system on the two most important metrics. Preci-
sion@10 was less stable, consistent with its sen-
sitivity to a small number of judgments per query.
Overall, these results suggest that pooled evalua-
tion is more robust at the system-ranking level than
raw item-level agreement alone would imply.
4.4 Evaluation Protocol
For each dataset, all systems retrieve top-100 doc-
uments per query. Documents are pooled and
judged once. We compute P@ k, R@k, nDCG@ k,
and MAP against bothpseudolabels(pooled LLM
judgments, available for all datasets) andgold qrels
(published human judgments, available for all four
datasets; NQ and FinRAGBench-V qrels are very
sparse with∼1 relevant document per query).
5 Results
5.1 System-Level Rank Correlation
Table 2 reports Spearman and Kendall rank cor-
relations between system rankings under pooled
LLM judgments and gold-standard qrels. Correla-
tions are strongest on FiQA and NQ (nDCG@10
ρ≥0.91 ), where the LLM judge’s broader rel-
evance notion aligns well with the sparse qrels.
FinRAGBench-V shows similarly strong agree-
ment ( ρ= 0.91 for nDCG@10, ρ= 0.94 for
R@100), consistent with its sparse qrels. TREC-
COVID yields lower but still significant correla-
tions ( ρ= 0.64 –0.82), reflecting the gap between
a general-purpose LLM and specialist assessors.
5.2 Pairwise Rank Agreement
The primary question for practitioners is whether
pooled LLM evaluation preserves relative system
orderings, i.e., if metrics calculated from qrel
ground truths say system A is better than B, do
pseudolabels agree? We measure this over all 11
2
= 55pairs per dataset:
FiQA and NQ show the strongest agreement
(up to 91%), while TREC-COVID, with its dense
expert-annotated qrels and specialised biomedical
domain, shows somewhat lower agreement (76–
84%).
Critically, many of these disagreements fall
within measurement noise. To quantify measure-
ment uncertainty in the gold standard itself, we
computebootstrap pairwise swap probabilities: in
each of B= 10,000 iterations we resample the Q
Figure 1: Bootstrap 95% rank confidence intervals for
nDCG@10 across four datasets. Blue circles show qrels
point ranks with CIs; orange diamonds show pseudola-
bel ranks with CIs. Most pseudolabel ranks fall within
the qrels CI, indicating that ranking disagreements are
within measurement noise.
queries with replacement, recompute the macro-
averaged metric for every system, and re-rank (full
procedure in Appendix A). The swap probabil-
ity for a pair (A, B) is the fraction of iterations
in which their ordering reverses. A pair isqrels-
uncertainif swaps occur in >5% of resamplings;
a pairwise disagreement between qrels and pseu-
dolabels is classified aswithin qrels noiseif the
qrels themselves are uncertain about that pair. For
nDCG@10, 20 of 27 disagreements across all four
datasets (74%) fall within qrels noise; only 7 re-
flect genuine ranking differences. Across all four
metrics combined, 74 of 112 disagreements (66%)
are within noise on the three BEIR benchmarks;
FinRAGBench-V shows the same pattern, with 4
of 6 nDCG@10 disagreements falling within qrels
uncertainty. Figure 1 visualizes this: most pseu-
dolabel ranks fall within the bootstrap 95% rank
confidence intervals of the qrels. In practical terms,
a developer using pooled LLM evaluation to decide
between two retrieval models will reach the same
conclusion as human evaluation in all but the most
marginal cases. Moreover, pooled LLM evalua-
tion correctly identifies the rank-1 system on FiQA
across all four metrics; the few disagreements on
NQ and TREC-COVID occur only where the qrels
margin between the top two systems is negligible
(bootstrap swap probability>40%).
5.3 Cost Amortization
Document overlap between retrieval systems pro-
duces substantial cost savings (Table 4). The
marginal cost of adding a new system decreases
as the pool grows, with each successive system

Metric FiQA TREC-COVID NQ FinRAG-V
ρ τ ρ τ ρ τ ρ τ
nDCG@10 .909 .782 .691 .564 .945 .818 .909 .782
P@10 .918 .818 .636 .527 .864 .745 .936 .818
R@100 .864 .709 .818 .673 .736 .600 .936 .855
MAP .855 .673 .736 .527 .718 .600 .882 .745
Table 2: Spearman ( ρ) and Kendall ( τ) rank correlation between pooled LLM rankings and gold-standard qrels
rankings. All correlations are statistically significant (p <0.05).
Metric FiQA T-COVID NQ FinRAG
nDCG@10 49 (89%) 43 (78%) 50 (91%) 49 (89%)
P@10 50 (91%) 42 (76%) 48 (87%) 50 (91%)
R@100 47 (85%) 46 (84%) 44 (80%) 51 (93%)
MAP 46 (84%) 42 (76%) 44 (80%) 48 (87%)
Table 3: Pairwise rank agreement (count and %) be-
tween qrels and pseudolabel system rankings (55 pairs
per dataset).
Dataset Indep. Pooled Saved Reuse
FiQA 712,800 249,865 462,935 65%
TREC-COVID 55,000 18,870 36,130 66%
NQ 550,000 188,903 361,097 66%
FinRAG-V 589,600 192,783 396,817 67%
Total1,907,400 650,421 1,256,979 66%
Table 4: Judgment cost: independent
(N×Q×K=11×Q×100 ) vs. pooled unique (q, d)
pairs. Pooling eliminates ∼1.26M redundant judgments
across four datasets.
requiring fewer novel judgments.
5.4 Ranking Stability Under Permuted Order
As discussed in Section 3.2, pool expansion
guarantees per-query order preservation but not
benchmark-level invariance for macro-averaged
metrics like nDCG@ k. To stress-test empirical
stability, we simulate 500 random system-addition
orderings per dataset: for each permutation we
grow the pool from 1 to 11 systems and check
whether any pairwise nDCG@10 ranking reverses
at any step (Table 5).
In this test NQ and TREC-COVID exhibitzero
reversals across all 500 orderings. On FiQA, 269
orderings (53.8%) trigger a reversal, but this is
confined to a single pair whose final nDCG@10
scores differ by only 0.000014 (dense-nomic-1.5
vs. hybrid-titan-v2); a second pair (hybrid-nomic-
1.5 vs. hybrid-nomic-v2) reverses in only 2.4% of
orderings. On FinRAGBench-V , 7 orderings (1.4%)
reverse the dense-emb3-large vs. dense-titan-v2
pair (∆ = 0.0006 ). In every case, the fragile pairs
are the same ones that bootstrap rank CIs identify
as statistically indistinguishable. No pair with a
meaningful score gap ( ∆>0.001 ) reverses underDataset % rev Fragile pair∆
FiQA 53.8 nomic-1.5 / hyb-titan 1.4e-5
hyb-nomic-1.5 / hyb-v2 1.2e-4
NQ 0 — —
TREC-COVID 0 — —
FinRAG-V 1.4 emb3-large / titan-v2 5.9e-4
Table 5: Permutation reversal search (nDCG@10, 500
random orderings). % rev = orderings with ≥1 pairwise
reversal during pool growth. ∆= final-pool nDCG@10
gap of the fragile pair.
any ordering on any dataset.
These results confirm that benchmark-level rank-
ing instability under pool growth, while theoreti-
cally possible, is in practice confined to pairs whose
scores are too close for any evaluation methodology
to reliably distinguish.
6 Analysis
6.1 Calibration Bias
Pseudolabel scores differ in absolute magnitude
from qrels scores, the LLM’s broader relevance
notion inflates nDCG (mean +0.1% to +51% de-
pending on dataset) while deflating recall ( −17%
to−39%), but this bias isapproximately uniform
across systemson all four datasets (within-column
std.≤12.8 pp for nDCG, ≤11.1 pp for recall; Ta-
ble 6 in Appendix B). Because the calibration offset
does not vary meaningfully between retrieval con-
figurations, it preserves ordinal system rankings.
6.2 Sources of Disagreement
Pairwise ranking disagreements are concentrated
among systems with very small score differences.
TREC-COVID shows the weakest agreement, con-
sistent with its dense expert biomedical qrels and
domain-specific relevance criteria. On the other
three datasets, most disagreements occur among
near-tied systems, and on FinRAGBench-V the re-
maining meaningful differences are concentrated
in the emb3-large family. The same fragile pairs
also appear in the pool-growth and bootstrap anal-
yses, indicating that most disagreements reflect
measurement-noise ties rather than substantive

ranking errors.
6.3 Pool Bias
Classical TREC pooling can penalize systems that
retrieve documents absent from the pool, because
unjudged documents are effectively treated as non-
relevant (Buckley et al., 2007). To quantify this
risk, we ran a leave-one-family-out analysis: for
each of six model families (five embedding fami-
lies, plus sparse), we removed that family’s unique
contributions and treated them as non-relevant.
Pool bias is dataset-dependent. FiQA and TREC-
COVID were most affected (8 and 10 pairwise
swaps, with maximum nDCG@10 penalties of
0.033 and 0.055, respectively), FinRAGBench-
V showed limited impact (1 swap, max penalty
0.012), and NQ was effectively unaffected (0
swaps, max penalty 0.008), consistent with higher
cross-family overlap.
By judging newly contributed documents when-
ever systems are added, pooled LLM evaluation
avoids this classical failure mode. In our experi-
ments, this removed all 19 pairwise ranking errors
introduced by classical pooling across the three
affected datasets.
7 Production Application
We applied pooled LLM evaluation to benchmark
a set of embedding models for a production news
question-answering system. This section reports
deployment experience, concrete cost figures, and
practical lessons.
7.1 Setting
The production corpus consists of ∼300k news arti-
cle pages indexed in OpenSearch, representing one
month of news data that is licensed for use in inter-
nal generative AI applications. We constructed
a benchmark of 303 queries from four sources:
LLM-generated questions based on semantic article
clusters (135), LLM-generated questions based on
metadata-clustered articles by company, industry,
and topic (91), templatic “latest news” questions
for frequently mentioned entities (50), and real
user queries from production API traffic, downsam-
pled to remove near-duplicates (27). Queries range
from factual lookups (Who were the newly elected
directors at TD Bank Group’s 2025 annual meet-
ing?) to complex multi-hop multi-source questions
(How is Stellantis addressing sales declines under
its new CEO?). We further perturb these queriesby converting them to keyword searches and to
“framed" queries, which specify output structure or
formatting e.g.How is Stellantis addressing sales
declines under its new CEO? →Summarize the
recent changes at Stellantis under its new CEO
and explain the company’s strategies for address-
ing sales declines. Organize your response using
section headings: Leadership Changes, Strategic
Initiatives, and Sales Recovery Measures.
7.2 Systems Evaluated
Using the pooled evaluation methodology, we
benchmarked 62 retrieval configurations compris-
ing 5 embedding models (OpenAI emb3-large ,
Cohere Embed v4, Nomic v1.5, Nomic v2-moe,
MiniLM-L12-v2); 3 retrieval modes (dense, hybrid,
sparse); 2–3 dimensionalities per model (256, 768,
1536); 3 query formulations per system (original,
keyword-reduced, instruction-framed); and vari-
ants of the top-performing system with and without
query expansion.
7.3 Cost Savings
The final pool contains 766,350 unique (q, d) judg-
ments (all annotated by GPT-4.1). Without pooling,
evaluating the same set of experiments indepen-
dently would have required 3,765,643 judgments,
a 79.6% reuse rate yielding a 4.9 ×cost reduction.
The pool was built incrementally over four weeks
as new model candidates and query variants were
added; at no point did previously computed judg-
ments need to be re-run.
7.4 Key Findings and Model Selection
Production rankings.The top-5 systems by
MAP@100 were all hybrid configurations: (1)
hybrid-emb3-large-768 (MAP 0.461), (2) hybrid-
emb3-large-256 (0.459), (3) hybrid-nomic-1.5-768
(0.454), (4) hybrid-nomic-v2-moe-768 (0.454), (5)
hybrid-cohere-v4-1536 (0.454). Based on these
results combined with latency and embedding cost
considerations, we selected hybrid-emb3-large-256
for production deployment.
Additional findings.Higher dimensions im-
prove dense retrieval (mean ∆MAP = +0.029 for
256→768/1536), but the gap narrows under hybrid
retrieval (+0.012), suggesting BM25 fusion com-
pensates for reduced embedding capacity. Hybrid
retrieval is also more robust to query formulation:
keyword-reduced queries hurt dense models more
(∆MAP = −0.013), and instruction-framed queries

degrade all systems ( ∆MAP = −0.039) by diluting
the retrieval signal. Query expansion does not help
on well-formed queries (MAP 0.424 vs. 0.459) but
acts as a robustness mechanism against verbose for-
mulations, nearly eliminating the framing-induced
recall@100 penalty (−4.8pp→ −1.1pp).
7.5 Practical Lessons
Incremental pool growth works.We began with
5 systems and grew to 62 over four weeks. In
our deployment, early conclusions (e.g., “hybrid
>dense”) were not invalidated by later additions.
This is qualitatively different from re-running an
entire benchmark each time.
Cost is manageable.At GPT-4.1 pricing ($2/M
input tokens, $8/M output tokens), the 766K judg-
ments cost approximately $800 total.
Beyond model selection.The same pool an-
swered questions the team had not originally
planned: “Does dimensionality matter under hy-
brid?” “Are keyword queries harder?” “Does query
expansion help on framed queries?” Because all
judgments are reusable, these analyses required
zero additional LLM cost.
Judge stability.Using a pinned model ver-
sion ( gpt-4.1-2025-04-14 ) at temperature 0 pre-
cludes drift by design within an evaluation cam-
paign. If the judge model is later updated, selective
re-judging of a sample can validate continuity.
8 Discussion
Practical Implications.For production teams
maintaining retrieval systems, pooled LLM evalu-
ation enables a reusable incremental benchmark-
ing workflow: (1) establish an initial pool with
3–5 candidate systems; (2) when a new model be-
comes available, retrieve with it and judge only the
newdocuments it contributes; (3) compare against
all previously evaluated systems, with exact score
reuse for previously judged documents and strong
empirical ranking stability as the pool grows.
When This Method is Most Valuable.The ap-
proach is particularly suited to domains where: (a)
gold-standard relevance judgments do not exist
or are prohibitively expensive to create; (b) new
embedding models are regularly evaluated (e.g.,
quarterly vendor updates); (c) maintaining a stable
historical benchmark is important for governance.9 Conclusion
We presented pooled LLM evaluation as a practical
framework for ongoing retrieval model selection.
In a production deployment, the method evaluated
62 retrieval configurations over four weeks at a
total annotation cost of $800, directly informing
model selection for a financial news QA system.
Across four public benchmarks with 11 systems
each, pooled LLM rankings correlate strongly with
human judgments (nDCG@10 ρ= 0.69 –0.95),
and bootstrap analysis shows that 74% of pairwise
disagreements fall within the qrels’ own sampling
uncertainty, leaving only 7 meaningful nDCG@10
ranking differences across 220 system pairs. Our
empirical stress tests confirm that macro-averaged
rankings remain stable as new systems are added
to the pool. Document overlap yields reuse rates of
65–80%, making incremental evaluation economi-
cally viable as new embedding models arrive.
10 Limitations
•Tight clusters:When systems are separated
by<0.001 nDCG (e.g., the NQ top-2), nei-
ther qrels nor pseudolabels can reliably distin-
guish them, bootstrap swap probabilities ex-
ceed 40% for both label sources. The method
is designed for comparative model selection,
not for resolving effectively tied systems.
•Limited judge diversity:System rankings
are derived from a single primary judge (GPT-
4.1). Full pool re-judging with Sonnet 4.6
on TREC-COVID confirms high system-level
agreement ( ρ= 0.89 nDCG@10; §4.3), but
this validation covers only two frontier-class
LLMs on one dataset. Correlation and bias
characteristics may differ for smaller, open-
weight, or domain-specialized judges. Addi-
tionally, the frozen-model assumption means
that if either provider deprecates the pinned
checkpoint, re-judging a validation sample is
necessary to confirm continuity.
•System architecture scope:All evaluated
systems are single-stage retrievers (dense,
sparse, or hybrid). We do not test rerank-
ing pipelines or cross-encoder architectures,
where document overlap patterns and score
distributions may differ substantially.
•Benchmark-level stability:Per-query pair-
wise invariance under pool expansion is exact

for recall@ k, AP, and nDCG@ k(Section 3.2),
but macro-averaged benchmark rankings can
in principle shift when new systems change
the relevance denominator. Our permutation
stress tests (Section 5.4) show this is confined
to pairs separated by <0.001, yet the guaran-
tee remains empirical rather than universal.
References
Hanoz Bhathena, Parin Rajesh Jhaveri, Rohan Mit-
tal, Prateek Singh, Aymen Kallala, Rachneet Kaur,
Yiqiao Jin, Zhen Zeng, Adwait Ratnaparkhi, and De-
nis Kochedykov. 2026. Mm-bizrag: Rethinking mul-
timodal retrieval-augmented generation for general
purpose enterprise q&a.Preprint, arXiv:2606.04231.
Chris Buckley, Darrin Dimmick, Ian Soboroff, and Ellen
V oorhees. 2007. Bias and the limits of pooling for
large collections.Information Retrieval, 10:491–
508.
Ben Carterette, James Allan, and Ramesh Sitaraman.
2006. Minimal test collections for retrieval evalua-
tion. InProceedings of the 29th Annual International
ACM SIGIR Conference, pages 268–275.
Guglielmo Faggioli, Laura Dietz, Charles Clarke, Gi-
anluca Demartini, Matthias Hagen, Claudia Hauff,
Noriko Kando, Evangelos Kanoulas, Martin Potthast,
Benno Stein, and Henning Wachsmuth. 2023. Per-
spectives on large language models for relevance
judgment. InProceedings of the 2023 ACM SIGIR
International Conference on the Theory of Informa-
tion Retrieval.
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red-
field, Michael Collins, Ankur P. Parikh, Chris Alberti,
Danielle Epstein, Illia Polosukhin, Jacob Devlin, Ken-
ton Lee, Kristina Toutanova, Llion Jones, Matthew
Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob
Uszkoreit, Quoc V . Le, and Slav Petrov. 2019. Natu-
ral questions: A benchmark for question answering
research.Transactions of the Association for Compu-
tational Linguistics, 7:452–466.
Macedo Maia, Siegfried Handschuh, André Freitas,
Brian Davis, Ross McDermott, Manel Zarrouk, and
Alexandra Balahur. 2018. WWW’18 open challenge:
Financial opinion mining and question answering.
InCompanion Proceedings of The Web Conference
2018, pages 1941–1942.
Virgil Pavlu and Javed A. Aslam. 2007. A practical
sampling strategy for efficient retrieval evaluation.
Technical report, College of Computer and Informa-
tion Science, Northeastern University. Manuscript /
technical report.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. BEIR:A heterogeneous benchmark for zero-shot evalua-
tion of information retrieval models.arXiv preprint
arXiv:2104.08663.
Paul Thomas, Seth Spielman, Nick Craswell, and
Bhaskar Mitra. 2024. Large language models can ac-
curately predict searcher preferences. InProceedings
of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval.
Shivani Upadhyay, Ronak Pradeep, Nandan Thakur,
Daniel Campos, Nick Craswell, Ian Soboroff,
Hoa Trang Dang, and Jimmy Lin. 2024a. A
large-scale study of relevance assessments with
large language models: An initial look.Preprint,
arXiv:2411.08275.
Shivani Upadhyay, Ronak Pradeep, Nandan Thakur,
Nick Craswell, and Jimmy Lin. 2024b. Umbrela:
Umbrela is the (open-source reproduction of the)
bing relevance assessor.Preprint, arXiv:2406.06519.
Ellen V oorhees, Tasmeer Alam, Steven Bedrick, Dina
Demner-Fushman, William R. Hersh, Kyle Lo, Kirk
Roberts, Ian Soboroff, and Lucy Lu Wang. 2021.
TREC-COVID: Constructing a pandemic informa-
tion retrieval test collection.ACM SIGIR Forum,
54(1):1–12.
Ellen M V oorhees. 1998. Variations in relevance judg-
ments and the measurement of retrieval effectiveness.
InProceedings of the 21st annual international ACM
SIGIR conference on Research and development in
information retrieval, pages 315–323.
Suifeng Zhao, Zhuoran Jin, Sujian Li, and Jun Gao.
2025. Finragbench-v: A benchmark for multimodal
RAG with visual citation in the financial domain.
arXiv preprint arXiv:2505.17471.
Justin Zobel. 1998. How reliable are the results of
large-scale information retrieval experiments? In
Proceedings of the 21st Annual International ACM
SIGIR Conference on Research and Development in
Information Retrieval, pages 307–314.
A Bootstrap Swap-Probability Details
This appendix gives the full procedure underly-
ing thewithin qrels noiseclassification reported
in Section 5. To quantify measurement uncer-
tainty in the gold standard itself, we computeboot-
strap pairwise swap probabilities. For each label
source (qrels or pseudolabels), we construct a per-
query score matrix M∈RQ×Nwhere Mq,iis the
nDCG@10 score of system ion query q. In each
ofB= 10,000 bootstrap iterations, we resample
Qqueries with replacement, compute the macro-
averaged metric ¯si=1
QP
qMq,iover the resam-
pled set, and rank all Nsystems (rank 1 = highest
score; ties broken by minimum rank). For each
system pair (A, B) , the swap probability pA≻B

is the fraction of iterations in which Ais ranked
above B. A pair is classified asqrels-uncertain
ifmin(p A≻B,1−p A≻B)>0.05 , i.e., the less-
favoured system wins in more than 5% of query
resamplings. A pairwise disagreement between
qrels and pseudolabels is then classified aswithin
qrels noiseif the qrels themselves are uncertain
about that pair under this criterion. Under this cri-
terion, pooled LLM evaluation correctly identifies
the rank-1 system on FiQA across all four metrics;
the few disagreements on NQ and TREC-COVID
occur only where the qrels margin between the top
two systems is negligible (bootstrap swap probabil-
ity>40%).
B Calibration Bias Details
Pooled LLM judgments exhibit consistent but
dataset-dependent calibration bias relative to hu-
man qrels. The LLM judge assigns graded rele-
vance to a broader set of documents per query than
human annotators. This inflates nDCG numerators
(more documents contribute gain) while simultane-
ously expanding the denominator of recall-based
metrics (more documents are “relevant,” so any
fixed-depth retrieval covers a smaller fraction).
FiQA.nDCG@10 pseudolabels score 29–74%
higher than qrels (mean +51%), while recall@100
scores 33–45% lower.
TREC-COVID.nDCG@10 bias is near zero
(mean +0.1%, range −17.5% to +8.5%). However,
recall@100 is 145–211% higher under pseudola-
bels because TREC-COVID’s exhaustive expert
qrels identify ∼1,327 relevant documents per query,
far more than the pool’s 256 documents can cover.
NQ.nDCG@10 pseudolabels score 2–46%
higher (mean +17%), while recall@100 scores 31–
43% lower.
FinRAGBench-V .nDCG@10 pseudolabels
score 29–76% higher (mean +43%), while
recall@100 scores 13–23% lower (mean −17%).
The pattern mirrors FiQA: sparse qrels ( ∼1.1
rel/query) mean the LLM identifies many
additional relevant documents.
C LLM Judge Prompt
The full prompt template used for graded relevance
annotation is shown below. Placeholders {pas-
sage}and{query}are filled at inference time.### Task Description: Determine the relevance
of the "DOCUMENT" to the "QUERY" provided below.
The goal is to assess the relevance of the
document with respect to a query based on the
document's "Title" and "Content".
### Relevance Scores:
0: Not Relevant - the document does not answer
the query
1: Partially Relevant - the document contains
peripheral information but the core intent
remains unanswered
2: Fully Relevant - the document provides a
full and complete answer to the query
### Output Guidelines:
1. Understand the Context
2. Apply the Relevance Criteria
3. Determine the Relevance Score (0-2)
4. Output as JSON:
{"thought": "<2 lines max>",
"relevance_score": <0, 1, or 2>}
DOCUMENT: {passage}
QUERY: {query}

FiQA TREC-COVID NQ FinRAG-V
System nDCG Recall nDCG Recall nDCG Recall nDCG Recall
dense-cohere-v4+29.1−43.1−17.5 +183.1 +11.0−42.8 +29.9−22.6
dense-emb3-large+36.9−33.1 +1.8 +171.2 +14.1−32.9 +75.8−16.5
dense-nomic-1.5+56.7−40.2 +1.0 +159.3 +5.5−40.6 +37.0−20.3
dense-nomic-v2+55.1−39.5 +8.5 +171.0 +1.9−42.4 +52.3−20.4
dense-titan-v2+38.8−44.5−4.9 +152.4 +12.8−42.1 +42.9−18.8
hybrid-cohere-v4+43.6−40.8−0.4 +153.4 +19.5−36.1 +29.2−14.4
hybrid-emb3-large+55.6−35.3 +0.9 +153.3 +21.6−30.7 +53.9−12.6
hybrid-nomic-1.5+58.9−38.1 +1.3 +151.8 +21.9−36.5 +33.0−16.6
hybrid-nomic-v2+61.4−37.4 +4.1 +160.1 +19.7−37.2 +41.5−15.4
hybrid-titan-v2+50.2−41.0 +1.9 +145.4 +17.0−36.0 +39.0−14.2
sparse (BM25)+74.1−36.4 +4.1 +172.7 +46.4−32.2 +39.4−12.8
Mean+50.9−39.0 +0.1 +161.2 +17.4−37.2 +43.1−16.8
Std.±12.3±3.2±6.4±11.1±11.0±4.1±12.8±3.2
Table 6: Per-system calibration bias (% ∆: pseudolabel vs. qrels) for nDCG@10 and recall@100. The low standard
deviation within each column confirms that bias is approximately uniform across systems, it differs in magnitude
between datasets and metrics, but not between retrieval configurations. This uniformity preserves ordinal system
rankings under pseudolabel evaluation.