# Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval Evaluation

**Authors**: Andrew Klearman, Radu Revutchi, Rohin Garg, Rishav Chakravarti, Samuel Marc Denton, Yuan Xue

**Published**: 2026-04-22 16:49:30

**PDF URL**: [https://arxiv.org/pdf/2604.20763v1](https://arxiv.org/pdf/2604.20763v1)

## Abstract
Retrieval quality is the primary bottleneck for accuracy and robustness in retrieval-augmented generation (RAG). Current evaluation relies on heuristically constructed query sets, which introduce a hidden intrinsic bias. We formalize retrieval evaluation as a statistical estimation problem, showing that metric reliability is fundamentally limited by the evaluation-set construction. We further introduce \emph{semantic stratification}, which grounds evaluation in corpus structure by organizing documents into an interpretable global space of entity-based clusters and systematically generating queries for missing strata. This yields (1) formal semantic coverage guarantees across retrieval regimes and (2) interpretable visibility into retrieval failure modes.
  Experiments across multiple benchmarks and retrieval methods validate our framework. The results expose systematic coverage gaps, identify structural signals that explain variance in retrieval performance, and show that stratified evaluation yields more stable and transparent assessments while supporting more trustworthy decision-making than aggregate metrics.

## Full Text


<!-- PDF content starts -->

Coverage, Not Averages: Semantic Stratification for Trustworthy Retrieval
Evaluation
Andrew Klearman1Radu Revutchi1Rohin Garg1Rishav Chakravarti1Samuel Marc Denton1Yuan Xue1
Abstract
Retrieval quality is the primary bottleneck for ac-
curacy and robustness in retrieval-augmented gen-
eration (RAG). Current evaluation relies on heuris-
tically constructed query sets, which introduce a
hidden intrinsic bias. We formalize retrieval evalu-
ation as a statistical estimation problem, showing
that metric reliability is fundamentally limited by
the evaluation-set construction. We further intro-
ducesemantic stratification, which grounds evalu-
ation in corpus structure by organizing documents
into an interpretable global space of entity-based
clusters and systematically generating queries for
missing strata. This yields (1) formal semantic
coverage guarantees across retrieval regimes and
(2) interpretable visibility into retrieval failure
modes.
Experiments across multiple benchmarks and re-
trieval methods validate our framework. The
results expose systematic coverage gaps, iden-
tify structural signals that explain variance in
retrieval performance, and show that stratified
evaluation yields more stable and transparent
assessments while supporting more trustworthy
decision-making than aggregate metrics.
1. Introduction
Retrieval-augmented generation (RAG) has emerged as a
central paradigm for grounding large language models in
external knowledge, enabling applications ranging from
open-domain question answering to scientific search and
enterprise assistants. Retrieval quality is a primary factor
influencing the effectiveness of RAG systems. By mediating
access to external knowledge, retrieval determines which
information is available for generation and thus strongly
affects downstream performance. Consequently, careful
and reliable evaluation of retrieval systems is essential for
understanding and improving RAG behavior.
1Scale AI. Correspondence to: Andrew Klearman <an-
drew.klearman@scale.ai>, Yuan Xue<yuan.xue@scale.ai>.
Preprint. April 23, 2026.Despite its importance, the construction of retrieval eval-
uation datasets remains comparatively under-investigated.
Standard benchmarks rely on heuristically constructed query
sets that attempt to approximate a “natural” query distribu-
tion. In offline and cold-start settings, however, such a
distribution is ill-defined or unavailable.
Through an empirical analysis of public benchmarks from
BEIR (Thakur et al., 2021), we show that evaluation queries
often fail to adequately cover the semantic space of the
underlying document corpus. Large semantic regions that
appear frequently in documents are rarely or never exer-
cised by evaluation queries. Crucially, these under-covered
regions are precisely where retrieval performance is weak-
est. As a result, standard aggregate metrics can substantially
overestimate true retrieval quality.
To address this issue, we formalize retrieval evaluation as
a statistical estimation problem. We show when evaluation
queries are drawn from a population that isstructurally
heterogeneouswith respect to retrieval regimes, ignoring
this structure leads to biased estimates and misleading con-
fidence. These limitations echo broader concerns about
benchmark reliability in NLP, where aggregate scores ob-
scure systematic failure modes.
Motivated by this analysis, we introducesemantic strati-
fication: a framework for grounding retrieval evaluation
dataset construction in the structure of the document corpus.
Our approach organizes documents into an interpretable
global semantic structure via entity-based clustering, de-
fines retrieval regimes along both semantic and structural
dimensions, and curates evaluation queries to ensure explicit
coverage guarantees. This enables unbiased performance es-
timation across retrieval regimes and exposes failure modes
invisible to aggregate metrics.
We evaluate the proposed framework across multiple public
datasets, retrieval methods, and evaluation metrics. These
experiments illustrate how coverage-aware, stratified eval-
uation exposes retrieval behavior obscured by standard ag-
gregate metrics and motivates more reliable comparison
between retrieval systems.
Contributions.This paper makes three contributions:
1arXiv:2604.20763v1  [cs.IR]  22 Apr 2026

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Problem identification.We identify a critical and underex-
plored issue in retrieval evaluation: evaluation datasets im-
plicitly assume query homogeneity, while real-world query
exhibitsstructural heterogeneityacross retrieval regimes.
We show that ignoring this heterogeneity leads to biased
metrics and misleading conclusions.
Methodological framework.We formulate retrieval evalua-
tion as a stratified statistical estimation problem and formal-
ize the conditions required for reliable regime construction.
Building on this analysis, we propose a semantic stratifica-
tion framework for evaluation dataset construction that is
coverage-aware and grounded in corpus structure.
Empirical validation and insights.We validate our frame-
work across multiple benchmarks and retrieval methods.
Our results expose systematic coverage gaps, identify struc-
tural signals explaining variance in retrieval performance,
and show that stratified evaluation yields more stable, trans-
parent and trustworthy decision-making than aggregate met-
rics.
2. Empirical Motivation
Using NFCorpus from BEIR (Thakur et al., 2021), a widely
adopted retrieval benchmark, we show that aggregate met-
rics can mischaracterize the system performance. Figure 1
compares the distribution of evaluation queries against the
underlying document corpus. We organize the corpus into
326semantic clusters (details in Sec. 4) and map each query
to the cluster of its ground-truth relevant document. Our
analysis reveals two key observations.
First, there is a compositional mismatch between evaluation
queries and the document corpus. Many high-volume se-
mantic clusters receive little or no query coverage, appearing
in the bottom-right region of Figure 1. In particular, we iden-
tify 26 clusters that together contain 17.3% of the corpus but
are covered by only 1.1% of queries, including medically
central topics such asimmune cells and their functions(122
documents, 1 query). As a result, these prevalent semantic
regions are effectively invisible to benchmark evaluation.
Second, retrieval performance varies systematically across
semantic clusters. The color-coded mean nDCG@10 in
Figure 1 shows that underrepresented clusters exhibit lower
mean nDCG@10 when evaluated, but contribute little to
aggregate scores due to their low query frequency.
Together, these effects induce a structural bias in benchmark
evaluation with aggregate metrics dominated by overrepre-
sented semantic regions and masking failures in prevalent
but sparsely evaluated domains.
Figure 1.Semantic coverage in NFCorpus. Each point corresponds
to a semantic cluster. The x-axis denotes the number of documents
in the cluster, and the y-axis denotes the number of evaluation
queries mapped to that cluster. The diagonal indicates proportional
coverage. Point color encodes mean nDCG@10 for queries in the
cluster; gray points indicate clusters with no query coverage.
3. Retrieval Evaluation as a Statistical
Estimation Problem
We view retrieval evaluation as a statistical estimation prob-
lem. Let D={d 1, . . . , d N}denote a document corpus
and let Qdenote the space of information-seeking queries
supported by the corpus. A retrieval system is modeled as
a policy π:Q → DN, which maps each query q∈ Q to a
ranked list of documents. This policy may be instantiated
using different retrieval paradigms, such as dense or sparse
retrieval, alternative embedding models, or document chunk-
ing strategies. Let ϕ(q;π) denote an evaluation functional,
such as nDCG@k, Recall@k, or MRR@k.
Evaluation is typically performed by computing the empiri-
cal average of ϕ(q;π) over a finite set of evaluation queries.
This procedure implicitly assumes that queries are sampled
from the canonical distribution D. Under this assumption,
the empirical average is treated as an unbiased estimator of
the system performance.
As shown in Sec. 2, this assumption is violated in realistic
retrieval settings. Evaluation queries are drawn from a pop-
ulation that isstructuredly heterogeneouswith respect to
retrieval behavior: queries fall into distinct retrieval regimes
for which retrieval metrics differ systematically.
Estimation under structured heterogeneity.We model
this structure by partitioning the query space into retrieval
regimes. Let {Sk}K
k=1denote a partition of Q, where each
regime corresponds to a coherent semantic or structural
region of the corpus. These regimes may reflect differences
in semantic intent, entity specificity, relevance dispersion,
or document connectivity.
Within each regime Sk, retrieval performance is character-
2

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
ized by the conditional mean µk=E[ϕ(q;π)|q∈ S k]
and conditional varianceσ2
k= Var[ϕ(q;π)|q∈ S k].
Letwk=Pq∼Q(q∈ S k)denote the population mass of
regimek. The population-level performance ofπis then
µ=KX
k=1wkµk.(1)
Given an evaluation set Qevalof size n, standard retrieval
evaluation techniques compute the empirical mean
ˆµnaive =1
nnX
i=1ϕ(qi;π).(2)
Under the assumption that evaluation queries are i.i.d. sam-
ples from the true query distribution Q,ˆµnaive is an unbi-
ased estimator of µwith variance Var(ϕ(q;π))/n . This
assumption underlies common practices such as reporting
single-number metrics and confidence intervals computed
via bootstrapping over evaluation queries.
Bias from coverage gaps.Let ˆwk=nk
ndenote the em-
pirical fraction of evaluation queries falling into regime Sk.
Conditioned on the regime composition of the evaluation
set, the expected value of the naive estimator is
E
ˆµnaive| {ˆwk}K
k=1
=KX
k=1ˆwkµk.(3)
Comparing Eq. (3)with the population mean in Eq. (1), the
conditional bias of the naive estimator is
Bias(ˆµ naive) =KX
k=1( ˆwk−wk)µk.(4)
In particular, if a retrieval regime Skis entirely missing from
the evaluation set (i.e., ˆwk= 0butwk>0), its contribution
to Eq. (1)is systematically excluded, yielding irreducible
bias regardless of evaluation set size.
Variance mis-estimation.Structured heterogeneity af-
fects not only the bias of aggregate estimation, but also
the uncertainty of reported metrics. Let S(q)∈[K] denote
the regime index ofq. By the law of total variance,
Var(ϕ(q;π)) =E[Var(ϕ(q;π)|S(q))]
+ Var(E[ϕ(q;π)|S(q)]),(5)
where the first term captures within-regime variability and
the second term captures between-regime variability, i.e.,
differences in regime-level means{µ k}K
k=1.
For i.i.d. queries q1, . . . , q n∼Q , the naive estimator sat-
isfies Var(ˆµ naive) = Var(ϕ(q;π))/n . When regimes withextreme values of µkare underrepresented or missing from
Qeval, the between-regime term in Eq. (5)is systematically
attenuated, yielding overly optimistic uncertainty estimates
for aggregate performance.
Regime-level performance as the primary evaluation
target.Equations (1)and(4)show that aggregate per-
formance µdepends on the (typically unknown) regime
weights wk. In many retrieval settings, the deployed query
distribution is ill-defined, making µan unstable evaluation
target.
In contrast, the conditional means µk=E[ϕ(q;π)|q∈
Sk]characterize intrinsic system behavior within coherent
regions of the query space and do not depend on the global
mixture weights. Whenever a regime Skis represented in
the evaluation set, µkcan be estimated by the within-regime
empirical mean
ˆµk=1
nknX
i=1I{qi∈ Sk}ϕ(q i;π),(6)
This suggests that reliable evaluation should prioritize
regime-level coverage, in order to accurately estimate per-
formance profiles {ˆµk}K
k=1, rather than relying solely on a
single aggregate metric.
Practical conditions for regime construction.Regime
definitions should satisfy two practical requirements to sup-
port fair and actionable evaluation.
•Method-independent regimes.Regime construction
should be independent of the particular retrieval policy
being evaluated, so that the partition does not “favor”
a specific method.
•Human-interpretable descriptors.It is desirable that
each regime admits an interpretable descriptor that
enables systematic diagnosis and comparison across
systems.
4. Stratified Evaluation Dataset Construction
This section presents a practical framework for constructing
evaluation datasets that satisfy the conditions introduced in
Sec. 3. In particular, our goal is to ensurewithin-stratum ho-
mogeneityandcoverage awarenessby explicitly grounding
evaluation set construction in the semantic and structural
organization of the document corpus. We follow a three-step
process: (1) corpus-level semantic structure construction,
(2) query stratification, and (3) evaluation dataset curation.
4.1. Semantic Structure Construction
We first construct an explicit and interpretable semantic
representation of the document corpus independent of any
retrieval system as shown in Fig. 2
3

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Figure 2.Semantic structure of a document corpus constructed via entity-based clustering. Query stratification converts query text into a
set of entities. Cluster assignments,J, and∆form a single query coverage vector.
Semantic entity extraction.We extractsemantic entities,
which are atomic and interpretable semantic units, from
each document using a LLM. Specifically, we prompt the
model to identify salient entities and return a pair (name,
description) that captures both surface form and semantic
meaning. To improve consistency and reduce redundancy,
entities are normalized and de-duplicated. Details on model
selection, prompt design, example entities, and dataset-wide
statistics are provided in Appendix A.1.
Semantic graph construction.We construct a semantic
graph in which nodes correspond to normalized semantic
entities. Each entity is represented by concatenating its
name and description and embedding the resulting text into
a fixed-dimensional vector space.1
Edges encode semantic relatedness. For each node, we
connect a fixed number of nearest neighbors whose cosine
similarity exceeds a minimum threshold, using FAISS (John-
son et al., 2019)-based approximate nearest neighbor search.
Edges are undirected and weighted by cosine similarity,
yielding a sparse graph that captures strong local semantic
affinity. Details of the parameter choice are provided in
Appendix A.2.
Corpus-level semantic clustering.We apply the Leiden
community detection algorithm (Traag et al., 2019) to the
semantic graph to identify coherent semantic neighborhoods.
The resolution parameter γcontrols cluster granularity and
1The embedding model is used solely to represent fine-grained
semantic entities, rather than full queries, documents or chucks,
and therefore does not reflect or depend on the retrieval system un-
der evaluation. At this granularity, we observe minimal sensitivity
to the specific choice of embedding model.is tuned on a per-dataset basis to balance interpretability and
coverage.
Each document is associated with the semantic communi-
ties containing its entities, yielding a corpus-level semantic
structure. Details of the clustering procedure, resolution
selection for each dataset, and sensitivity to γare provided
in Appendix A.3 and Appendix B.2.
4.2. Query Stratification
Based on the corpus semantic structure, we define retrieval
regimes along two categories. Together, these strata opera-
tionalize the structured heterogeneity model introduced in
Sec. 3.
Semantic strata.Semantic strata are defined directly by
the corpus-level semantic clusters. Each stratum corre-
sponds to a distinct region of information-seeking intent
supported by the corpus. This construction ensures that
evaluation queries are distributed across the full semantic
scope of the dataset, rather than concentrating on a narrow
subset of topics.
Structural strata.Retrieval difficulty varies substantially
due to structural properties of the query and document rela-
tionship. We define structural strata using two complemen-
tary signals that characterize the structural complexity of
retrieval regimes:
•Relevance dispersion ( ∆).Let C(d) denote the set
of semantic clusters associated with document d. We
4

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
define the dispersion of a queryqas
∆(q) =S
d∈D(q)C(d)
P
d∈D(q)|C(d)|.(7)
This quantity measures how widely the relevant docu-
ments for qare spread across semantic clusters. High
dispersion indicates that relevance mass is distributed
across multiple semantic regions, corresponding to
broad or ambiguous information needs that are typ-
ically harder for retrieval systems to rank accurately.
•Query-document semantic alignment ( J).Let C(q)
denote the set of semantic clusters associated with
entities extracted from the query, and let C(D(q)) =S
d∈D(q)C(d) denote the clusters touched by relevant
documents. We measure semantic alignment using the
Jaccard similarity:
J(q) =|C(q)∩C(D(q))|
|C(q)∪C(D(q))|.(8)
High alignment indicates strong correspondence be-
tween the semantic intent expressed in the query and
the semantic regions covered by relevant documents,
while low alignment reflects semantic mismatch that
makes retrieval more error-prone.
These two signals induce a two-dimensional partitioning
of queries into structural regimes with systematically dif-
ferent retrieval behavior. We empirically validate that the
resulting 3×3 dispersion/Jaccard grid partitions queries
into distinct difficulty regimes (Table 3, see also Tables 5
and 6 in Appendix B.1).
Unlike semantic strata, which are fixed by corpus structure,
structural strata are defined by discretizing continuous struc-
tural signals. Given a target number of strata per signal, we
compute regime boundaries from the empirical distributions
of∆(q) andJ(q) over an initial set of benchmark queries.
These boundaries partition queries into ordered categories
(e.g., low, medium, and high dispersion or alignment), yield-
ing balanced coverage across structural regimes.
As new evaluation queries are generated or incorporated,
regime boundaries may be recomputed to reflect updated
dataset statistics and the desired granularity of stratification.
This adaptive construction ensures that evaluation sets main-
tain coverage across the full spectrum of structural regimes,
preventing over-concentration in narrow subsets of queries
as the evaluation set evolves.
4.3. Evaluation Dataset Curation
The process begins by identifying coverage gaps across en-
tity cluster Jaccard, dispersion, and cluster coverage. Basedon these gaps, the algorithm selects a generation strategy-
single-cluster or multi-cluster-and samples seed documents
accordingly. An LLM generates a query from the sampled
document, then candidate entities are retrieved via KNN
and filtered by the LLM to determine the query’s seman-
tic scope. Relevant documents are identified by pooling
candidates from dense (KNN) and sparse (BM25) retrieval,
verified through LLM relevance judgments. The query and
its qrels are added to the evaluation set, updating coverage
state. This loop repeats until the target size |D|=N is
reached, producing an evaluation set with guarantees on se-
mantic coverage, structural coverage, and difficulty balance.
As the final step, we construct the evaluation dataset through
an iterative query curation process that explicitly controls
dataset quality. Starting from an initial query set (which may
be empty), our goal is to generate additional high-quality
queries to balance coverage across semantic and structural
regimes. The process consists of three stages.
First, given the existing semantic and structural strata, we
assess evaluation set quality using explicitcoverage cri-
teria. Semantic coverage measures whether semantically
prevalent regions of the corpus are exercised by sufficient
queries, while structural coverage measures whether queries
within each semantic region span the full range of structural
regimes. These metrics expose missing or underrepresented
strata that are obscured by aggregate statistics.
Second, we iteratively generate candidate queries to address
identified coverage gaps. At each iteration, query generation
is enabled by prompting a LLM to produce multiple candi-
dates, followed by joint LLM-based filtering and a scoring
function to ensure candidate quality.
Definitions of coverage criteria, algorithmic implementation
details, document sampling procedures, and query genera-
tion prompts are provided in Appendix C.
5. Experiments
Our experiments address three questions: (1) Do seman-
tic strata capture meaningful structure, exhibiting internal
homogeneity while revealing performance heterogeneity
across retrieval regimes? (2) Do standard benchmarks ex-
hibit coverage gaps that stratified evaluation can expose and
remedy? (3) Does stratum-level analysis yield actionable
diagnostics invisible under aggregate evaluation? Across
multiple datasets and retrieval systems, we answer all three
questions affirmatively. Strata form coherent evaluation
units with up to 3 ×performance variation across regimes,
existing benchmarks leave 40–50% of semantic regions
untested, and stratified analysis reveals both retriever failure
modes and corpus-level annotation issues that aggregate
metrics mask entirely.
5

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
5.1. Experimental Setup
Datasets and Evaluation Protocols.We conduct experi-
ments on three datasets from the BEIR benchmark (Thakur
et al., 2021): NFCorpus (Boteva et al., 2016), FiQA (Maia
et al., 2018), and Scidocs (Cohan et al., 2020). These
datasets cover specialized domains in which semantic cover-
age is particularly important for reliable evaluation. Table 1
summarizes their key statistics.
Dataset Docs Queries Relevant
NFCorpus (Medical) 3.6K 323 38.2
FiQA (Finance) 57K 648 2.6
SciDocs (Scientific) 25K 1,000 4.9
Table 1.Dataset statistics for Selective BEIR corpora (Relevant
indicates average relevant documents per query)
We compare two evaluation protocols: (1)Aggregate eval-
uation, where queries are sampled directly from benchmark
query sets and performance is reported as the aggregate
metrics with bootstrap confidence intervals; and (2)Strat-
ified evaluation(ours), where queries are organized into
semantic strata derived from corpus structure, ensuring ex-
plicit coverage across strata. Stratum-level metrics {µk}are
computed without assuming stratum prevalence, and scalar
summaries are reported using statistics across strata (e.g.,
medians or quantiles).
Retrieval Systems.We evaluate three represen-
tative retrieval paradigms: (1)Denseretrieval via
text-embedding-3-large , an embedding model
widely adopted in production systems (OpenAI, 2024);
(2)BM25(Robertson & Zaragoza, 2009), a standard
term-based sparse retrieval baseline; (3)Hybridretrieval
using Reciprocal Rank Fusion (Cormack et al., 2009) with
k= 60to combine dense and sparse rankings.
Across all retrieval paradigms, we observe consistent
stratum-level behavior, demonstrating that our analysis is
not specific to a particular retrieval paradigm.
Metrics.Retrieval performance is evaluated using: (1)
nDCG@10, for both rank position and graded relevance; (2)
Recall@100, measuring retrieval coverage; (3)MAP@10,
averaging precision over the top-ranked results.
All metrics show consistent trends across our analyses. For
brevity, we report results primarily on theNFCorpusdataset
and usenDCG@10as the main metric in figures. Re-
sults for additional datasets and metrics are provided in
Appendix E and exhibit the same patterns.Dataset Metric Dense BM25 Hybrid
NFCorpus nDCG@10 .31 / .29 .29 / .27 .30 / .28
Recall@100 .29 / .27 .25 / .22 .29 / .26
MAP@10 .21 / .16 .17 / .12 .18 / .14
SciDocs nDCG@10 .23 / .22 .20 / .17 .22 / .20
Recall@100 .30 / .28 .27 / .25 .29 / .27
MAP@10 .17 / .15 .14 / .11 .16 / .13
FiQA nDCG@10 .36 / .35 .27 / .26 .33 / .32
Recall@100 .28 / .25 .39 / .38 .31 / .28
MAP@10 .38 / .36 .23 / .23 .31 / .30
Table 2.Standard deviation (σ overall /σwithin) of retrieval metrics.
nDCG@10 Recall@100
LoJMedJHiJ LoJMedJHiJ
Hi∆.25 .31 .38 .22 .33 .55
Med∆.34 .42 .51 .25 .36 .60
Lo∆.45 .54 .62 .28 .40 .65
Table 3.Mean nDCG@10 / Recall@100 by relevance dispersion
(∆) and semantic alignment ( J) tertiles. Dispersion dominates
ranking; alignment dominates coverage.
5.2. Stratum Validation
We validate that our stratification framework captures mean-
ingful structure at both the semantic and structural levels:
semantic strata exhibit internal homogeneity while structural
signals partition queries into distinct difficulty regimes.
Semantic Stratum Homogeneity.Table 2 shows that
the standard deviation within strata ( σwithin) is consistently
lower than the standard deviation across the overall dataset
(σoverall ), across all retrieval metrics, systems, and datasets.
Performance Heterogeneity Across Strata.Figure 3
reveals substantial performance variation across semantic
strata, with different retrievers exhibiting distinct strengths
and weaknesses. For example,Nutrition Expertis the
lowest-performing stratum for all three retriever types, in-
dicating systematic retrieval difficulty rather than model-
specific failure. In contrast,Prostate Enlargementranks
among the top strata for dense retrieval but not for sparse or
hybrid methods—retriever-specific advantages masked by
aggregate metrics.
Structural Signal Effects.The structural signals (rele-
vance dispersion ∆and semantic alignment J) partition
queries into difficulty regimes (Table 3). Dispersion primar-
ily affects ranking quality (nDCG ranges from 0.25-0.62),
while alignment primarily affects coverage (Recall ranges
from 0.22-0.65). The diagonal from high ∆/lowJto low
∆/high Jspans a nearly 3 ×performance difference, val-
idating that structural strata represent genuinely distinct
retrieval regimes. Tables 5 and 6 in Appendix B.1 confirm
6

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Figure 3.Retrieval performance (nDCG@10) across semantic clus-
ters for different retrievers. Each box shows per-query score distri-
bution within a cluster; dashed line indicates overall nDCG@10.
these patterns hold across additional datasets.
5.3. Coverage Analysis
We assess semantic space coverage using three metrics:
Minimum Semantic Coverage (MSC): the fraction of se-
mantic clusters “touched” by at least one evaluation query.
A cluster is touched if any query contains any entity belong-
ing to that cluster. This captures whether a semantic region
is tested at all.
Sufficient Corpus Coverage (SCC): the fraction of docu-
ments in clusters with ≥5 queries, capturing if clusters have
sufficient queries for reliable per-stratum estimation.
Zero-Query Clusters (ZQC): the count of semantic clus-
ters with no evaluation queries, indicating untested semantic
regions.
Table 4 reveals substantial coverage gaps: NFCorpus
achieves only 51% MSC and 11% SCC. Stratified queries
generated with coverage-aware sampling achieve dramati-
cally higher coverage (90% MSC, 53% SCC) with compa-
rable query counts.
These gaps are systematic: ZQC in NFCorpus are pre-
dominantly methodological topics (statistical methods, clin-
ical trial designs) spanning 1,700+ documents (see Ap-Aggregate Stratified
MSC SCC ZQC MSC SCC ZQC
NFCorpus 50.9% 11% 182 90.3% 53% 36
FIQA 60.9% 38% 232 100% 56% 0
SciDocs 89.2% 48% 69 99.7% 65% 2
Table 4.Coverage metrics for aggregate (original BEIR) vs strati-
fied (coverage-aware generated) evaluation queries.
pendix D.4 for details). Note that documents in these clus-
ters may still be retrieved for other queries, but the concepts
themselves are never directly tested. A retrieval system
could fail completely on such queries and the benchmark
would not detect it.
We empirically confirm this by evaluating retrieval perfor-
mance on queries generated within zero-query clusters, re-
vealing both unobserved strengths and systematic failure
modes that are untested in the NFCorpus benchmark (Ap-
pendix D.5).
5.4. Interpretable Diagnostics
Stratified evaluation enables diagnosis ofwhyretrieval fails
within specific semantic regions, including corpus-level,
retrieval-agnosticfailure modes where success is funda-
mentally impossible due to limitations in corpus structure.
By isolating these cases, stratified evaluation reveals fail-
ure patterns that are obscured by aggregate metrics. We
present results on NFCorpus, with additional failure modes
for FiQA and SciDocs reported in Appendix E
Failure Mode 1: Semantic-Lexical Disconnect.Several
clusters reveal mismatches between query semantics and
document content due to how relevance was annotated. For
example, queries like “Arkansas” or “Iowa Women’s Health
Study” (nDCG ≈0.14) are judged relevant to documents
discussing researchconducted inthese locations or reporting
findingsfromthese studies, but the names never appear in
document. Similarly, researcher names and organizations
are judged relevant to documents that never mention them.
These queries are unanswerable from document content
alone, a corpus construction issue, not a retriever limitation.
Failure Mode 2: Register Mismatch.In NFCorpus,
health misinformation queries (nDCG = 0.14) like “Raw
Food Diet Myths” use colloquial language, while relevant
documents use scientific terms. This linguistic register gap
makes retrieval structurally difficult regardless of method.
Contrast: Well-Aligned Strata.In contrast, strata with
strong query-document alignment (endocrine disruptors
(nDCG = 0.76), food additives (0.69), respiratory diseases
(0.65)) benefit from consistent vocabulary, enabling effec-
7

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Figure 4.Comparing text-embedding-3-small (x-axis) and embed-
english-v3.0 (y-axis). Each point represents one semantic cluster,
sized by query count. Points above the diagonal indicate Cohere
wins; below OpenAI wins.
tive semantic matching.
5.5. Implications on Retrieval System Decisions
In practice, retrieval evaluation guides system selection
by comparing aggregate statistics. Stratified evaluation of-
fers more intentional approaches for such decisions. We
demonstrate this by comparing OpenAI’s text-embedding-
3-small (OpenAI, 2024) and Cohere’s embed-english-
v3.0 (Cohere, 2023) on NFCorpus (Figure 4).
When neither model dominates across all clusters, practi-
tioners must still choose one. The conventional approach
of comparing overall mean implicitly weights clusters by
query count, allowing large clusters to dominate. However,
this may not reflect the practitioner’s actual priorities.
Figure 5 demonstrates this sensitivity through 1,000 boot-
strapped evaluation sets. Using overall mean, text-
embedding-3-small wins only 32.2% of samples; using
macro-average (equal weight per cluster), the same model
wins 63.3%. If a practitioner cares equally about all seman-
tic regions regardless of query volume, macro-averaging
better reflects their intent. Stratified evaluation makes these
trade-offs explicit rather than leaving them as implicit arti-
facts of dataset composition.
6. Related Work
RAG Benchmarks. The evaluation of RAG systems inher-
its methodologies from information retrieval research, most
notably the Text REtrieval Conference (TREC) (Chowd-
hury, 2007). These efforts led to unified question answering
(QA) benchmarks such as BEIR, as well as foundational
datasets including Natural Questions (NQ) (Kwiatkowski
et al., 2019), HotpotQA (Yang et al., 2018), TriviaQA (Joshi
Figure 5.Bootstrap win rates for text-embedding-3-small vs.
embed-english-v3.0 under different aggregation protocols (1,000
resamples). Overall Mean weights by query count; Macro Average
weights clusters equally; Median and Worst Case capture central
tendency and tail performance.
et al., 2017), and MS MARCO (Nguyen et al., 2016) . Build-
ing on these foundations, more recent benchmarks have
been developed to target specific failure modes of RAG
systems, including MultiHop-RAG (Tang & Yang, 2024),
CRAG (Yang et al., 2024), GlobalQA (Luo et al., 2025),
Long2RAG (Qi et al., 2024), MT-RAG (IBM Research,
2025), mmRAG (Xu et al., 2025), and QE-RAG (Zhang
et al., 2025). However, these benchmarks largely rely on
static or narrowly defined evaluation settings, leaving the
limitation targeted in this paper unexamined.
Query Taxonomies and Curation.Prior work has pro-
posed principled benchmark curation via query taxonomies
that classify queries by task type (Teixeira de Lima et al.,
2025) or operational category (Lyu et al., 2025). However,
these approaches focus on query-level properties and do not
address corpus-level semantic coverage gaps.
Corpus Stratification.Methods such as GraphRAG (Edge
et al., 2025), RAPTOR (Sarthi et al., 2024), and Hip-
poRAG (Gutierrez et al., 2024) explore corpus stratifica-
tion to improve retrieval efficiency and quality. GraphRAG
is most closely related, but our approach detects relation-
ships directly in semantic space and uses stratification for
benchmark diagnosis rather than retrieval optimization.
Synthetic Data Generation.To reduce manual annotation
costs, RAGEval (Zhu et al., 2025), Long2RAG (Qi et al.,
2024), and DataMorgana (Filice et al., 2025) use schema-
driven pipelines to generate grounded QA data. While ef-
fective for scaling evaluation, these methods do not address
systematic coverage gaps across corpus semantic regions.
See Appendix F for further discussion on related works.
7. Limitations
This work has several limitations that suggest directions
for future research. First, our evaluation is conducted on
8

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
a limited set of datasets and embedding models. Broader
validation across more diverse domain, other languages,
domains, and benchmark suites is necessary. Second, the
queries used in our experiments are synthetically generated
and verified by LLMs. Although this enables scalable eval-
uation, it may introduce biases or overlook subtle errors.
Incorporating human evaluation would provide a stronger
calibration and improve the reliability of the assessment.
Finally, the extraction performance depends on the design
of prompts, which were not exhaustively optimized. Prompt
engineering could further improve extraction quality.
8. Conclusion
By formalizing retrieval evaluation as a statistical estimation
problem under structured heterogeneity, we explain why
aggregate metrics can yield biased and decision-unstable
assessments. Our semantic stratification framework grounds
evaluation in corpus structure, exposes previously hidden
failure modes, and enables more transparent and trustworthy
system comparison. We hope this work encourages a shift
toward coverage-aware evaluation protocols for retrieval
and RAG systems. To support reproducibility and future
research, we will publicly release the stratified datasets
introduced in this work.
Impact Statement
This paper examines the reliability of retrieval evaluation
benchmarks and proposes a coverage-aware, stratified eval-
uation framework to improve the trustworthiness of sys-
tem comparison and selection. The goal of this work is
to advance evaluation methodology in machine learning,
particularly for retrieval and retrieval-augmented generation
systems.
The primary societal impact of this work is indirect. By
improving the reliability and transparency of evaluation, our
framework may help practitioners make better-informed de-
cisions when selecting retrieval systems for downstream
applications, including search, question answering, and
knowledge-intensive AI systems. More trustworthy evalua-
tion may reduce the risk of deploying systems that perform
well on benchmarks but fail systematically in underrepre-
sented or high-risk information domains.
This work does not introduce new models, training pro-
cedures, or data sources, nor does it directly enable new
deployment capabilities. As such, we do not anticipate
immediate negative societal impacts beyond those already
associated with retrieval-based machine learning systems.
Ethical considerations primarily concern responsible evalu-
ation practice, and we view improved coverage and diagnos-
tic transparency as a positive step toward more accountable
system assessment.References
Boteva, V ., Gholipour, D., Sokolov, A., and Riezler, S. A
full-text learning to rank dataset for medical information
retrieval. InProceedings of the 38th European Confer-
ence on Information Retrieval, 2016.
Carterette, B., Allan, J., and Sitaraman, R. Minimal test
collections for retrieval evaluation. InProceedings of
the 29th Annual International ACM SIGIR Conference
on Research and Development in Information Retrieval,
SIGIR ’06, pp. 268–275, 2006. doi: 10.1145/1148170.
1148219.
Chowdhury, G. Trec: Experiment and evaluation in in-
formation retrieval.Online Information Review, 31(5):
717–718, 10 2007. ISSN 1468-4527. doi: 10.1108/
14684520710832478. URL https://doi.org/10.
1108/14684520710832478.
Cohan, A., Feldman, S., Beltagy, I., Downey, D., and
Weld, D. SPECTER: Document-level representation
learning using citation-informed transformers. In Juraf-
sky, D., Chai, J., Schluter, N., and Tetreault, J. (eds.),
Proceedings of the 58th Annual Meeting of the Associa-
tion for Computational Linguistics, pp. 2270–2282, On-
line, July 2020. Association for Computational Linguis-
tics. doi: 10.18653/v1/2020.acl-main.207. URL https:
//aclanthology.org/2020.acl-main.207/.
Cohere. Introducing embed v3. https://cohere.
com/blog/introducing-embed-v3 , November
2023. Accessed: 2026.
Cormack, G. V ., Clarke, C. L. A., and Buettcher, S. Re-
ciprocal rank fusion outperforms condorcet and indi-
vidual rank learning methods. InProceedings of the
32nd International ACM SIGIR Conference on Research
and Development in Information Retrieval, SIGIR ’09,
pp. 758–759, New York, NY , USA, 2009. Associa-
tion for Computing Machinery. ISBN 9781605584836.
doi: 10.1145/1571941.1572114. URL https://doi.
org/10.1145/1571941.1572114.
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A.,
Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O.,
and Larson, J. From local to global: A graph rag ap-
proach to query-focused summarization, 2025. URL
https://arxiv.org/abs/2404.16130.
Filice, S., Horowitz, G., Carmel, D., Karnin, Z., Lewin-
Eytan, L., and Maarek, Y . Generating Q&A benchmarks
for RAG evaluation in enterprise settings. In Rehm, G.
and Li, Y . (eds.),Proceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Volume
6: Industry Track), pp. 469–484, Vienna, Austria, July
2025. Association for Computational Linguistics. ISBN
9

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
979-8-89176-288-6. doi: 10.18653/v1/2025.acl-industry.
33. URL https://aclanthology.org/2025.
acl-industry.33/.
Guti´errez, B. J., Shu, Y ., Gu, Y ., Yasunaga, M., and Su, Y .
Hipporag: Neurobiologically inspired long-term memory
for large language models. In Globerson, A., Mackey,
L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., and
Zhang, C. (eds.),Advances in Neural Information Pro-
cessing Systems, volume 37, pp. 59532–59569. Curran
Associates, Inc., 2024. doi: 10.52202/079017-1902.
URL https://proceedings.neurips.
cc/paper_files/paper/2024/file/
6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.
pdf.
Gutierrez, B. J., Shu, Y ., Gu, Y ., Yasunaga, M., and Su,
Y . HippoRAG: Neurobiologically inspired long-term
memory for large language models. InThe Thirty-eighth
Annual Conference on Neural Information Processing
Systems, 2024. URL https://openreview.net/
forum?id=hkujvAPVsg.
IBM Research. A benchmark for evaluating conver-
sational RAG. https://research.ibm.com/
blog/conversational-RAG-benchmark , 2025.
Accessed: 2025-12-06.
Johnson, J., Douze, M., and J ´egou, H. Billion-scale simi-
larity search with gpus.IEEE Transactions on Big Data,
2019.
Joshi, M., Choi, E., Weld, D., and Zettlemoyer, L. Trivi-
aQA: A large scale distantly supervised challenge dataset
for reading comprehension. In Barzilay, R. and Kan,
M.-Y . (eds.),Proceedings of the 55th Annual Meet-
ing of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 1601–1611, Vancouver,
Canada, July 2017. Association for Computational Lin-
guistics. doi: 10.18653/v1/P17-1147. URL https:
//aclanthology.org/P17-1147/.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., De-
vlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M.,
Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., and
Petrov, S. Natural questions: A benchmark for question
answering research.Transactions of the Association for
Computational Linguistics, 7:452–466, 2019. doi: 10.
1162/tacl a00276. URL https://aclanthology.
org/Q19-1026/.
Luo, Q., Li, X., Fan, T., Chen, X., and Qiu, X. Towards
global retrieval augmented generation: A benchmark for
corpus-level reasoning, 2025. URL https://arxiv.
org/abs/2510.26205.Lyu, Y ., Li, Z., Niu, S., Xiong, F., Tang, B., Wang,
W., Wu, H., Liu, H., Xu, T., and Chen, E. Crud-
rag: A comprehensive chinese benchmark for retrieval-
augmented generation of large language models.ACM
Trans. Inf. Syst., 43(2), January 2025. ISSN 1046-8188.
doi: 10.1145/3701228. URL https://doi.org/10.
1145/3701228.
Maia, M., Handschuh, S., Freitas, A., Davis, B., McDermott,
R., Zarrouk, M., and Balahur, A. Www’18 open chal-
lenge: Financial opinion mining and question answering.
InCompanion Proceedings of the The Web Conference
2018, WWW ’18, pp. 1941–1942, Republic and Canton
of Geneva, CHE, 2018. International World Wide Web
Conferences Steering Committee. ISBN 9781450356404.
doi: 10.1145/3184558.3192301. URL https://doi.
org/10.1145/3184558.3192301.
Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S.,
Majumder, R., and Deng, L. MS MARCO: A human
generated machine reading comprehension dataset. In
Proceedings of the Workshop on Cognitive Computation:
Integrating Neural and Symbolic Approaches (NeSy) at
NIPS, 2016.
OpenAI. New embedding models and API up-
dates. https://openai.com/index/
new-embedding-models-and-api-updates ,
January 2024. Accessed: 2026.
Polo, F. M., Weber, L., Choshen, L., Sun, Y ., Xu, G., and
Yurochkin, M. tinybenchmarks: evaluating llms with
fewer examples. InProceedings of the 41st International
Conference on Machine Learning, ICML’24. JMLR.org,
2024.
Qi, Z., Xu, R., Guo, Z., Wang, C., Zhang, H., and Xu,
W.long2rag: Evaluating long-context & long-form
retrieval-augmented generation with key point recall.
In Al-Onaizan, Y ., Bansal, M., and Chen, Y .-N. (eds.),
Findings of the Association for Computational Linguis-
tics: EMNLP 2024, pp. 4852–4872, Miami, Florida,
USA, November 2024. Association for Computational
Linguistics. doi: 10.18653/v1/2024.findings-emnlp.
279. URL https://aclanthology.org/2024.
findings-emnlp.279/.
Rahman, M. M., Kutlu, M., Elsayed, T., and Lease, M. Effi-
cient test collection construction via active learning. In
Proceedings of the 2020 ACM SIGIR on International
Conference on Theory of Information Retrieval, ICTIR
’20, pp. 177–184, New York, NY , USA, 2020. Associa-
tion for Computing Machinery. ISBN 9781450380676.
doi: 10.1145/3409256.3409837. URL https://doi.
org/10.1145/3409256.3409837.
10

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Robertson, S. and Zaragoza, H. The probabilistic relevance
framework: Bm25 and beyond.Foundations and Trends
in Information Retrieval, 3:333–389, 01 2009. doi: 10.
1561/1500000019.
Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie,
A., and Manning, C. D. Raptor: Recursive abstrac-
tive processing for tree-organized retrieval, 2024. URL
https://arxiv.org/abs/2401.18059.
Tang, Y . and Yang, Y . Multihop-RAG: Bench-
marking retrieval-augmented generation for multi-hop
queries. InFirst Conference on Language Modeling,
2024. URL https://openreview.net/forum?
id=t4eB3zYWBK.
Teixeira de Lima, R., Gupta, S., Berrospi Ramis, C., Mishra,
L., Dolfi, M., Staar, P., and Vagenas, P. Know your
RAG: Dataset taxonomy and generation strategies for
evaluating RAG systems. In Rambow, O., Wanner, L.,
Apidianaki, M., Al-Khalifa, H., Eugenio, B. D., Schock-
aert, S., Darwish, K., and Agarwal, A. (eds.),Proceedings
of the 31st International Conference on Computational
Linguistics: Industry Track, pp. 39–57, Abu Dhabi, UAE,
January 2025. Association for Computational Linguis-
tics. URL https://aclanthology.org/2025.
coling-industry.4/.
Thakur, N., Reimers, N., R ¨uckl´e, A., Srivastava, A., and
Gurevych, I. BEIR: A heterogeneous benchmark for
zero-shot evaluation of information retrieval models. In
Proceedings of the 44th International ACM SIGIR Con-
ference on Research and Development in Information
Retrieval, 2021.
Traag, V . A., Waltman, L., and Van Eck, N. J. From Louvain
to Leiden: guaranteeing well-connected communities.
Scientific Reports, 9(1):1–12, 2019.
Xu, C., Chen, Q., Feng, Y ., and Cheng, G. mmrag: A
modular benchmark for retrieval-augmented generation
over text, tables, and knowledge graphs, 2025. URL
https://arxiv.org/abs/2505.11180.
Yang, Y . et al. Comprehensive RAG benchmark (CRAG).
InAdvances in Neural Information Processing Systems
(NeurIPS) 37, 2024.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W.,
Salakhutdinov, R., and Manning, C. D. HotpotQA: A
dataset for diverse, explainable multi-hop question an-
swering. In Riloff, E., Chiang, D., Hockenmaier, J.,
and Tsujii, J. (eds.),Proceedings of the 2018 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, pp. 2369–2380, Brussels, Belgium, October-
November 2018. Association for Computational Lin-
guistics. doi: 10.18653/v1/D18-1259. URL https:
//aclanthology.org/D18-1259/.Zhang, K., Sun, Z., Yu, W., Zang, X., Zheng, K., Song, Y .,
Li, H., and Xu, J. Qe-rag: A robust retrieval-augmented
generation benchmark for query entry errors. InPro-
ceedings of the International ACM SIGIR Conference on
Research and Development in Information Retrieval, 04
2025.
Zhu, K., Luo, Y ., Xu, D., Yan, Y ., Liu, Z., Yu, S., Wang,
R., Wang, S., Li, Y ., Zhang, N., Han, X., Liu, Z., and
Sun, M. RAGEval: Scenario specific RAG evaluation
dataset generation framework. In Che, W., Nabende,
J., Shutova, E., and Pilehvar, M. T. (eds.),Proceed-
ings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pp. 8520–8544, Vienna, Austria, July 2025. Association
for Computational Linguistics. ISBN 979-8-89176-251-
0. doi: 10.18653/v1/2025.acl-long.418. URL https:
//aclanthology.org/2025.acl-long.418/.
11

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
A. Semantic Structure Construction
A.1. Entity Extraction
Model: GPT-4o-mini
EXTRACTION_PROMPT = """Extract key entities from this document.
For each entity provide:
- name: the entity name (normalized, lowercase, no hyphens or special characters)
- description: a brief, context-independent description (10-20 words) that would help cluster this entity with
semantically similar entities
Include: people, organizations, places, concepts, technical terms, methods, algorithms, systems.
Exclude: generic words, common verbs, articles.
Document:
{text}
Examples of extracted entities.Extracted entities include technical concepts (e.g., “neural network,” “transformer
architecture”), domain-specific terminology (e.g., “Cox regression,” “quantitative easing”), named entities (e.g., “Glaxo-
SmithKline,” “PubMed”), and methodological terms (e.g., “randomized controlled trial”).
Entity normalization.Entities undergo string normalization and semantic merging (e.g., “neural-networks” →“neural
network”) to reduce redundancy. For NFCorpus, this yields approximately 24,000 unique entities across 3,633 documents;
for FiQA, over 109,000 entities from 57,000 documents.
A.2. Graph Construction
Entity embedding.Each semantic entity is represented by concatenating its extracted name and description into a short
text span, which is embedded using text-embedding-3-large text embedding model with 256 dimensions. We find
that dimensionality reduction improves graph sparsity and clustering stability without materially affecting downstream
analyses.
Graph sparsification.We construct a semantic graph by connecting each entity node to its nearest neighbors in the
embedding space, using cosine similarity as the affinity measure. Approximate nearest neighbor search is performed using
FAISS (Johnson et al., 2019) for scalability. For each node, we retain a fixed number of neighbors subject to a minimum
similarity threshold, yielding a sparse, undirected graph with edge weights given by cosine similarity.
In all experiments, we set the number of neighbors to k= 50 and the similarity threshold to 0.5. These values were selected
to balance graph connectivity and sparsity, ensuring that semantic neighborhoods are well-formed while avoiding spurious
long-range connections. We observe that moderate variations in these parameters do not qualitatively affect the resulting
semantic communities, coverage statistics, or downstream evaluation conclusions.
A.3. Corpus-Level Semantic Clustering
We apply the Leiden community detection algorithm (Traag et al., 2019) to the semantic graph to identify coherent semantic
neighborhoods. The resolution parameter γcontrols cluster granularity and is tuned per dataset to balance interpretability
and coverage (γ= 25for NFCorpus,γ= 40for FIQA,γ= 10for SciDocs).
Each document is associated with all clusters containing its entities, yielding a corpus-level semantic taxonomy. Manual
inspection confirms that clusters correspond to interpretable regions of the information space (e.g., diseases, methods,
financial instruments).
This representation directly enables semantic coverage analysis. For NFCorpus, 42 clusters (11.3%), containing 13.5% of
documents, have zero evaluation queries (systematic gaps that standard evaluation would not reveal).
12

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
B. Query Stratification Details
B.1. Structural Validation
Dispersion explanation power.On NFCorpus, relevance dispersion ( ∆) explains 12.3% of variance in nDCG@10 (VRR
= 0.123,p <10−80), with low-∆queries achieving mean nDCG of 0.56 versus 0.30 for high-∆queries.
Alignment explanation power.On NFCorpus, semantic alignment ( J) explains 23.5% of variance in Recall@100 (VRR
= 0.235,p <10−100), with high-Jqueries achieving mean Recall of 0.60 versus 0.25 for low-Jqueries.
Structural stratum partition.We partition each signal into tertiles (Low, Medium, High), yielding a 3×3 structural
grid within each semantic stratum. A complete stratum is thus identified by three components: the semantic cluster, the
dispersion tertile (∆: Low, Medium, or High), and the alignment tertile (J: Low, Medium, or High).
Structural signal validation.Tables 5 and 6 extend the structural signal analysis from NFCorpus (main paper) to
FiQA and SciDocs. The same pattern holds: dispersion ( ∆) primarily affects ranking quality (nDCG), while alignment
(J) primarily affects coverage (Recall). The diagonal from high ∆/lowJto low ∆/high Jconsistently represents the
hardest-to-easiest progression across all datasets.
Table 5.FiQA: Mean nDCG@10 (left) and Recall@100 (right) by relevance dispersion ( ∆) and semantic alignment ( J) tertiles (n=6,148
queries).
nDCG@10 Recall@100
LoJMedJHiJ LoJMedJHiJ
Hi∆0.32 0.47 0.58 0.65 0.79 0.85
Med∆0.41 0.49 0.58 0.70 0.77 0.83
Lo∆0.46 0.54 0.59 0.72 0.80 0.84
Table 6.SciDocs: Mean nDCG@10 (left) and Recall@100 (right) by relevance dispersion ( ∆) and semantic alignment ( J) tertiles
(n=1,000 queries).
nDCG@10 Recall@100
LoJMedJHiJ LoJMedJHiJ
Hi∆0.15 0.21 0.25 0.39 0.48 0.50
Med∆0.17 0.25 0.27 0.41 0.58 0.55
Lo∆0.19 0.28 0.31 0.45 0.58 0.62
In FiQA, the performance range is narrower than NFCorpus (nDCG: 0.32-0.59 vs 0.25-0.62), suggesting more uniform
retrieval difficulty across structural strata. SciDocs exhibits the lowest overall performance (nDCG: 0.15-0.31), consistent
with its challenging scientific citation retrieval task, but the structural patterns remain consistent: low ∆and high J
consistently yield better retrieval outcomes.
B.2. Resolution Sensitivity Analysis
We conduct a resolution sensitivity analysis on NFCorpus by varying the clustering granularity parameter γ∈
{10,15,25,40,60}.
γClustersσ within ∆VRRJVRR nDCG (min / max)
10 155 0.310 0.104 0.219 0.436 / 0.632
15 210 0.305 0.131 0.228 0.438 / 0.643
25 313 0.306 0.125 0.237 0.432 / 0.739
40 474 0.305 0.133 0.286 0.435 / 0.856
60 669 0.304 0.137 0.326 0.401 / 0.856
Table 7.Sensitivity of clustering resolution γ. Core structural metrics and retrieval behavior remain stable across a 4×increase in cluster
count.
13

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Across this 4×range in cluster count, all core findings remain stable. Within-stratum homogeneity ( σwithin) is consistent,
and variance reduction ratios (VRR) for dispersion and Jaccard—measuring how well structural signals predict nDCG@10
and Recall@100—remain stable across resolutions.
At finer granularities, the spread in per-cluster performance increases (e.g., nDCG ranges from 0.40 to 0.86 at γ= 60 ),
revealing increasingly differentiated retrieval behavior across semantic regions.
Qualitatively, clusters remain coherent at all resolutions; the primary difference is granularity rather than quality. For
example, at γ= 10 , a single cluster captures statistical modeling (285 entities), while at γ= 25 this separates into survival
analysis, meta-analysis, and significance testing. At γ= 60 , further refinement isolates more specific subdomains such as
Cox regression and age-adjusted models.
C. Dataset Curation
Algorithm 1 presents the coverage-aware query generation procedure.
Algorithm 1Coverage-Aware Query Generation
Require:Entity graphGwith clustersC, corpusDwith entity mappings
Require:Target query countN, candidates per iterationB
Q← ∅ {generated queries}
nσ,nJ←0{bucket counts for dispersion/jaccard (low, mid, high)}
Ccovered← ∅ {clusters with at least one query}
while|Q|< Ndo
// Step 1: Adaptive Strategy Selection
(s, m)←ChooseStrategy(n σ,nJ){sampling strategy, entity mode}
// Step 2: Generate Candidate Queries
candidates← ∅
fori= 1TOBdo
dq←SampleDoc(s)
qtext←GenerateQuery(d q)
Dcand←RetrieveRelevantDocs(q text, D, s)
Dq← {d q} ∪LLMVerifyRelevance(q text, Dcand)
Eq←AssignEntities(q text, m, D q)
σq←ComputeDispersion(E q)
Jq←ComputeJaccard(E q, Dq)
candidates←candidates∪ {(q text, Dq, σq, Jq)}
end for
// Step 3: Select Best Candidate
(q∗, D∗, σ∗, J∗)←arg max candidates Score(σ, J,n σ,nJ, Ccovered )
// Step 4: Update Coverage State
Q←Q∪ {(q∗, D∗)}
nσ[bucket(σ∗)] += 1
nJ[bucket(J∗)] += 1
Ccovered←C covered∪clusters(D∗)
end while
returnQ
C.1. Prompt Specifications
Query generation.The following prompt is used to synthesize search queries from source documents.
SYSTEM:
You are an expert search query generator for information retrieval
evaluation datasets.
Your task is to generate realistic search queries that a user might
type when looking for information contained in a given document.
14

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
These queries will be used to evaluate search systems, so quality
and naturalness are critical.
CORE PRINCIPLES:
1. AUTHENTICITY: Generate queries that real users would actually
type, not artificial constructs
2. SPECIFICITY CONTROL: Match the requested query type precisely
(short/long, ambiguous/specific)
3. ANSWERABILITY: The query MUST be answerable by the provided
document
4. NO LEAKAGE: Do not copy exact phrases from the document title
5. INFORMATION NEED: Express a genuine information need, not just
keywords
WHAT TO AVOID:
- Queries that are too generic to meaningfully match the document
- Queries that require information NOT in the document
- Forced or unnatural combinations of keywords
---
USER:
Generate a search query that this document would answer.
Query Type: {query_type_description}
{query_type_instructions}
[RANDOMLY SAMPLED STYLE CONSTRAINTS - see Query Diversity Grid]
{style_constraints}
Document:
Title: {doc_title}
Content: {doc_text}
Generate ONE search query matching the specified type and style
constraints (just the query text, nothing else):
Query diversity dimensions.Style constraints are randomly sampled from this 3x2 grid to ensure diverse query character-
istics.
QUERY DIVERSITY DIMENSIONS
==========================
For each query, we randomly sample from two orthogonal
dimensions to ensure diverse query characteristics:
DIMENSION 1: LENGTH
-------------------
- SHORT (1-2 words): "BERT embeddings", "cancer treatment"
- MEDIUM (3-4 words): "GPT-4 reasoning capabilities"
- UNRESTRICTED: Natural question length
DIMENSION 2: SPECIFICITY
------------------------
- PROPER NOUNS: Use specific named entities
Examples: "BERT", "Pfizer", "COVID-19", "Tesla"
- GENERIC TERMS: Use category-level terms
Examples: "language models", "vaccines", "electric vehicles"
SAMPLED STYLE CONSTRAINTS
=========================
The selected (Length, Specificity) pair injects instructions:
SHORT + PROPER NOUNS:
LENGTH: Your query MUST be exactly 1-2 words only.
ENTITY STYLE: Use SPECIFIC proper nouns and named entities.
-> "BERT embeddings", "COVID-19 transmission"
SHORT + GENERIC TERMS:
LENGTH: Your query MUST be exactly 1-2 words only.
ENTITY STYLE: Use GENERIC category terms, NOT specific names.
-> "language models", "cancer treatment"
MEDIUM + PROPER NOUNS:
LENGTH: Your query MUST be exactly 3-4 words only.
ENTITY STYLE: Use SPECIFIC proper nouns and named entities.
-> "GPT-4 reasoning capabilities", "CRISPR Cas9 delivery"
MEDIUM + GENERIC TERMS:
LENGTH: Your query MUST be exactly 3-4 words only.
15

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
ENTITY STYLE: Use GENERIC category terms, NOT specific names.
-> "deep learning applications", "gene therapy risks"
UNRESTRICTED + PROPER NOUNS:
ENTITY STYLE: Use SPECIFIC proper nouns and named entities.
-> "How does BERT handle contextual word representations?"
UNRESTRICTED + GENERIC TERMS:
ENTITY STYLE: Use GENERIC category terms, NOT specific names.
-> "How do transformer models improve understanding?"
This 3x2 sampling ensures diverse query formulations matching
real-world search patterns.
Entity assignment.The following prompt identifies which semantic entities are relevant to a generated query.
SYSTEM:
You are an expert entity-to-query alignment system for information
retrieval research.
Your task is to identify which entities from a candidate list are
semantically relevant to a given search query. This mapping is used
to understand the semantic structure of queries.
WHAT COUNTS AS "RELEVANT":
1. DIRECT MENTION: The entity or a synonym appears in the query
2. IMPLICIT REFERENCE: The query is about this entity even if not
named explicitly
Example: "heart disease prevention" -> relevant: "cardiovascular
disease", "heart health", "coronary artery"
3. REQUIRED KNOWLEDGE: Answering the query requires knowing about
this entity
Example: "CRISPR applications" -> relevant: "gene editing",
"Cas9", "genetic engineering"
4. SEMANTIC OVERLAP: The entity’s domain strongly overlaps with
the query’s information need
WHAT IS NOT RELEVANT:
- Entities that are merely topically adjacent but not needed to
answer the query
- Entities from the same broad field but addressing different
specific aspects
- Entities that might appear in the same document but aren’t
query-relevant
PRECISION OVER RECALL:
- It’s better to miss a marginally relevant entity than to include
an irrelevant one
- Only include entities you are confident help characterize the
query’s information need
- When uncertain, lean toward exclusion
OUTPUT: Return only the indices of relevant entities.
---
USER:
Identify which entities from the candidate list are relevant to
the search query.
An entity is RELEVANT if:
1. It appears in or is directly referenced by the query
2. It represents a key concept the query is asking about
3. Understanding this entity is necessary to answer the query
4. It’s a person, organization, place, or technical term central
to the query
An entity is NOT RELEVANT if:
- It’s from the same general field but addresses a different aspect
- It might co-occur with query terms but isn’t about the query topic
- It’s too broad or too specific relative to the query’s need
Search Query: "{query_text}"
Candidate Entities:
{entity_list}
For each candidate, ask: Would a user asking this query want/need
information about this entity? Is it central to answering the query?
16

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Be SELECTIVE. Only include entities that genuinely characterize
the query’s semantic content.
Return ONLY the entity indices that are relevant:
Relevance filtering.The following prompt identifies which retrieved documents are relevant to the query.
SYSTEM:
You are an expert relevance assessor for information retrieval
evaluation.
Your task is to judge whether candidate documents are RELEVANT to
a given search query. Your judgments will be used as ground truth
for evaluating search systems, so accuracy is critical.
RELEVANCE DEFINITION:
A document is RELEVANT if it contains information that would
SATISFY the user’s information need expressed in the query. The
document should directly help answer or address the query.
RELEVANCE LEVELS (for your internal reasoning):
- HIGHLY RELEVANT: Document directly and completely addresses query
- RELEVANT: Document provides useful information that partially
addresses the query
- MARGINALLY RELEVANT: Document has some topical overlap but
limited utility
- NOT RELEVANT: Document doesn’t help answer the query despite
surface similarity
For this task, return documents that are RELEVANT or HIGHLY
RELEVANT only.
COMMON PITFALLS TO AVOID:
1. TOPICAL SIMILARITY != RELEVANCE
Two documents about "machine learning" may address completely
different aspects. The query might be about "ML for medical
imaging" but doc is about "ML for NLP"
2. KEYWORD MATCHING != RELEVANCE
Matching keywords doesn’t mean the document answers the query.
Context and the specific information need matter.
3. DOCUMENT QUALITY != RELEVANCE
A poorly written document can still be relevant. A high-quality
document on a different topic is not relevant.
WHEN IN DOUBT:
- Re-read the query and identify the core information need
- Ask: "If I were the user, would this document satisfy my search?"
- Lean toward precision: it’s better to miss a marginally relevant
doc than include an irrelevant one
---
USER:
Judge which documents are RELEVANT to the given search query.
A document is RELEVANT if it:
- Contains information that directly helps answer the query
- Would satisfy a user who submitted this search
- Provides substantive content on the query topic
A document is NOT RELEVANT if it:
- Only shares keywords but addresses a different information need
- Is about a related but distinct topic
- Mentions the query concepts tangentially without substantive
coverage
Search Query: "{query_text}"
Candidate Documents:
{documents_text}
For each document, determine if it would SATISFY a user searching
for this query. Consider:
1. Does this document contain information the user is looking for?
2. Would returning this document be helpful for this specific query?
3. Is the relevance substantive or merely superficial keyword overlap?
Respond with ONLY the document numbers that are RELEVANT,
17

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
comma-separated (e.g., "0, 2, 5"). If none are relevant,
respond with "NONE".
D. Examples and Coverage Analysis
D.1. Example Semantic Cluster
Table 8 shows an example cluster capturing the semantic region of “fruit juices and their health effects,” with entities
spanning beverage types, specific juice varieties, and associated medical conditions.
D.2. Cluster Visualization
Figure 6 visualizes the semantic clusters across the NFCorpus dataset.
Figure 6.NFCorpus semantic cluster graph. Nodes represent entity clusters, node size indicates document count, and edges represent
semantic similarity between clusters. Colors indicate avg in-cluster nDCG score.
D.3. Example Generated Queries
Sample generated NFCorpus queries with strata values and relevant documents. Example queries show increase in
Zero-Query Cluster coverage.
[Q1] "renal function differences vegans non-vegans"
Docs (7): MED-5340, MED-3230, MED-3236, MED-5338, MED-934, MED-1458, MED-1456
Delta: 0.530 | J: 0.159 | Clusters: 7 (6 new)
[Q2] "How do different vegetarian diets affect cholesterol and triglyceride levels compared to omnivorous diets in
adults?"
Docs (18): MED-1430, MED-4632, MED-4831, MED-5329, MED-1551, MED-2945, MED-4088, MED-4515, MED-1253, MED-1250,
MED-5118, MED-4353, MED-4509, MED-5113, MED-2008, MED-1249, MED-4510, MED-1258
Delta: 0.384 | J: 0.227 | Clusters: 15 (7 new)
[Q3] "glycemic response of different date varieties in healthy versus type 2 diabetic adults"
Docs (2): MED-3897, MED-3907
Delta: 0.429 | J: 0.444 | Clusters: 4 (4 new)
18

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Entity Description
apple juice A beverage made from the pressing and
processing of apples
cranberry juice A beverage made from cranberries, of-
ten studied for health benefits
fruit juice Beverage made by extracting liquid
from fruits
cranberries A fruit known for its health benefits, par-
ticularly for urinary health
grapefruit A citrus fruit known for its potential in-
teractions with medications
pomegranate juice Juice extracted from pomegranate,
known for its health benefits
red wine A type of wine made from dark-colored
grape varieties
white wine A type of wine made from green grapes
noni juice Juice derived from noni fruit, marketed
for health benefits
mangosteen juice Juice extracted from mangosteen, asso-
ciated with health claims
hibiscus tea A herbal tea made from dried petals of
hibiscus flowers
cherry juice A beverage made from cherries, noted
for health benefits
smoothie A blended beverage typically made
from fruits and vegetables
bladder infection An infection of the urinary bladder (fre-
quently studied in relation to cranberry
juice)
... and 53 more entities
Table 8.Example Semantic Cluster: Fruit Juices & Beverages (Cluster 118, NFCorpus)
Sample generated FIQA queries with strata values and relevant documents. Example queries show increase in Zero-Query
Cluster coverage.
[Q1] "reverse stock split investor impact"
Docs (16): 579056, 507021, 229707, 477951, 388065, 304085, 502970, 513620, 237390, 581848, 159439, 129481,
524940, 512914, 245654, 352130
Delta: 0.587 | J: 0.031 | Clusters: 2 (2 new)
[Q2] "Chinese public opinion of Donald Trump during 2016-2020 presidential elections"
Docs (4): 541283, 279360, 291836, 106268
Delta: 0.800 | J: 0.333 | Clusters: 4 (3 new)
[Q3] "economic confirmation bias in academic research debates"
Docs (8): 303978, 498180, 187090, 457235, 394096, 382709, 118092, 394357
Delta: 0.707 | J: 0.069 | Clusters: 2 (1 new)
Sample generated SciDocs queries with strata values and relevant documents. Example queries show increase in Zero-Query
Cluster coverage.
[Q1] "neural encoding auditory stimuli"
Docs (10): 620b7d0d5e2ceeba58d808dc3d7b09a9fb57831c, 4934ac57123efbfbde139ec846452acd95520b40, 3
c203e4687fab9bacc2adcff2cbcab2b6d7937dd, 6d3f38ea64c84d5ca1569fd73497e34525baa215, 8
bfedec4b58a5339c7bd96e367a2ed3662996c84, 6042669265c201041c7523a21c00a2ab3360f6d1,
ca579dbe53b33424abb13b35c5b7597ef19600ad, f18821d47b2bee62a7b3047ed7b71a5895214619,
bdfb20d112069c106a9535407d107a43ebe6517f, 8fc062ac5cded5cbacb164e2b128f9914eb04727
Delta: 0.327 | J: 0.361 | Clusters: 13 (1 new)
[Q2] "information technology strategic impact"
19

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Docs (16): 14316b885f65d2197ce8c6d4ab3ee61fdab052b8, d1909ee7543ab92a8d8aad81af6c589278654b48, 42540
e75173dea14c343a567f9fc722db8609e7e, 65e58981f966a6d1b62c1f7d889ae3e1c0a864a1, 5
f09cb313b6fb14877c6b5be79294faf1f4f7f02, 4c341889abc15b296c85dd5b4e1356c664f633f4, 7
bad7050e0f447ea68c1ff5838cd5392726ca388, c13ee933b7ddccb4a1ccc7820c09fda8f32d1fb5, 588
b9fae704c1964d639a5f87c3793a1ad354e69, d60fcb9b0ff0484151975c86c70c0cb314468ffb, 37
c998ede7ec9eeef7016a206308081bce0fc414, 8326f86876adc98c72031de6c3d3d3fac0403175,
eec44862b2d58434ca7706224bc0e9437a2bc791, ac8877b0e87625e26f52ab75e84c534a576b1e77, 6
e07fcf8327a3f53f90f86ea86ca084d6733fb88, 176d486ccd79bdced91d1d42dbd3311a95e449de
Delta: 0.386 | J: 0.078 | Clusters: 4 (1 new)
[Q3] "What are the TG13 guideline recommendations for biliary drainage timing and antimicrobial use in Grade II and
III acute cholangitis patients?"
Docs (1): e65881a89633b8d4955e9314e84b943e155da6a9
Delta: 0.250 | J: 1.000 | Clusters: 3 (1 new)
D.4. Zero-Query Cluster Analysis
Table 9 provides detailed examples of zero-query clusters identified in NFCorpus. These clusters represent methodological
topics (statistical methods, clinical trial designs, and survey techniques) that collectively span over 1,700 documents but
receive no evaluation queries in the original benchmark. This systematic gap means retrieval performance on methodological
queries (e.g., “What is a Cox proportional hazards model?”) remains entirely untested.
Cluster Description Ent. Docs
Statistical methods (Cox regression, hazard ratios) 104 470
Multivariate analysis methods 135 325
Clinical trial methods (RCTs, double-blind) 141 450
Dietary assessment (food frequency questionnaire) 81 192
Survey methodologies 70 89
Healthy human participants 79 155
Table 9.Zero-query clusters in NFCorpus span 1,700+ documents but have no evaluation queries.
D.5. Retrieval Performance on Zero-Query Clusters
To directly evaluate retrieval behavior in untested regions of the corpus, we generate queries targeting two zero-query
clusters in NFCorpus and evaluate three retrieval methods using nDCG@10.
Zero-Query Cluster Dense BM25 Hybrid
Cancer cell lines 0.7950.8540.872
Statistical methods 0.365 0.3360.483
Table 10.Retrieval performance (nDCG@10) on queries generated for zero-query clusters.
These results illustrate why coverage is critical for trustworthy evaluation. Thecancer cell linescluster achieves strong
retrieval performance across all methods, yet this signal is entirely absent from the benchmark’s aggregate score due to the
lack of evaluation queries. In contrast, thestatistical methodscluster represents a genuine blind spot, with substantially
lower performance across all retrievers.
The two clusters further reveal distinct retriever dynamics. BM25 outperforms dense retrieval oncancer cell lines, while the
hybrid approach provides the largest improvement onstatistical methods, suggesting that different regions of the corpus
favor different retrieval strategies.
E. Additional Experiment Results
E.1. Results on Additional Datasets
Figures 7-12 present performance distributions and system comparisons for FiQA and SciDocs datasets.
20

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Figure 7.FiQA: Distribution of nDCG@10 scores across semantic clusters, showing per-cluster performance variance.
21

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Figure 8.SciDocs: Distribution of nDCG@10 scores across semantic clusters, showing per-cluster performance variance.
22

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Figure 9.FiQA: Per-query performance comparison between retrieval systems, colored by semantic cluster.
23

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Figure 10.SciDocs: Per-query performance comparison between retrieval systems, colored by semantic cluster.
24

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
Figure 11.FiQA: Win rate comparison between retrieval systems across bootstrap samples (n=300).
Figure 12.SciDocs: Win rate comparison between retrieval systems across bootstrap samples (n=300).
25

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
E.2. Additional Interpretable Diagnostics
Beyond the corpus-level failure modes identified in the main paper (semantic-lexical disconnect, register mismatch), stratified
evaluation reveals additional retriever-dependent failure patterns across datasets.
Named Entity Mismatch (NFCorpus).ThePharmaceuticals and drug developmentcluster (39 entities) exhibits a 75%
zero-score rate across 8 queries. Sample entities include company names like “GlaxoSmithKline” and technical terms
like “pharmaceuticalization.” Queries in this cluster ask about drug development processes, but the named pharmaceutical
companies mentioned in corpus documents rarely appear in natural user queries. This represents a retriever failure where
proper noun specificity in documents does not match the more generic query formulations.
High-Frequency Practical Topics (FiQA).In FiQA, clusters representing common personal finance topics achieve
disproportionately poor retrieval performance:
•Retirement investment strategies(62 entities, nDCG = 0.23): 66.7% zero-score rate despite targeting common
concepts like “target date retirement funds.”
•Language and terminology in finance(144 entities, nDCG = 0.17): 62.5% zero-score rate for queries about contextual
meaning and financial terminology.
•Customization and personalization(nDCG = 0.11): Abstract concepts like “personalized gifts” and “features” that
are too generic to embed discriminatively.
These high-frequency, practical questions fail because common financial terminology may be too generic in the embedding
space to discriminate between relevant and irrelevant documents.
Technical Algorithm Names (SciDocs).In SciDocs, clusters containing highly specialized algorithm names achieve
near-zero performance:
•Optimization algorithms(424 entities, nDCG = 0.07): 75% zero-score rate with entities like “NelderMeadAlgorithm”
and “alternating direction method of multipliers.”
•Sentiment analysis(308 entities, nDCG = 0.07): 77.8% zero-score rate despite covering a well-established NLP
subfield.
•Digital healthcare innovations(268 entities, nDCG = 0.02): 80% zero-score rate for “quantified self” and “healthcare
innovation” topics.
The failure despite exact terminology matches suggests that these technical terms may be underrepresented in the embedding
model’s training data, or that the citation-based relevance judgments in SciDocs create a mismatch between lexical similarity
and annotated relevance.
High-Performing Clusters.For contrast, we identify consistently high-performing clusters:
•NFCorpus: Viruses linked to obesity (nDCG = 0.81), MRSA strains (nDCG = 0.77), gluten-related disorders (nDCG
= 0.76)
•FiQA: Tax incentives (nDCG = 0.97), corporate scandals (nDCG = 0.96), aviation industry (nDCG = 0.96)
•SciDocs: Algebraic structures in programming (nDCG = 0.72), amplifier design (nDCG = 0.72), wearable health
monitoring (nDCG = 0.57)
These clusters succeed because queries and documents use consistent domain terminology with distinctive named entities,
validating that retrieval works well when lexical-semantic alignment holds.
26

Coverage, Not Averages Semantic Stratification for Trustworthy Retrieval Evaluation
F. Additional Related Work
The evaluation of Retrieval-Augmented Generation (RAG) has evolved from lexical n-gram matching to principled
methodologies that isolate the performance of retrieval and generation components.
IR Evaluation.Modern RAG evaluation inherits methodology from information retrieval research such as the Text
REtrieval Conference (TREC) which established standardized test collections, relevance judgments, and evaluation metrics
(Chowdhury, 2007). These efforts culminated in unified Question Answering (QA) benchmarks like BEIR (Thakur et al.,
2021), which aggregates 18 diverse and foundational sets like NQ, HotpotQA, TriviaQA, and MS MARCO (Kwiatkowski
et al., 2019; Joshi et al., 2017; Nguyen et al., 2016; Yang et al., 2018).
RAG Benchmarks.Modern RAG-specific benchmarks extend these foundations to target specific RAG failure modes.
MultiHop-RAG requires synthesis across multiple documents (Tang & Yang, 2024). CRAG provides question-answer
pairs spanning five domains with varying entity popularity, temporal dynamics, and complexity (Yang et al., 2024). For
corpus-level reasoning, GlobalQA introduces aggregation tasks such as counting, sorting, and top-k extraction over entire
collections (Luo et al., 2025). Long2RAG addresses long-context scenarios with the Key Point Recall metric for evaluating
generation (Qi et al., 2024), MT-RAG benchmarks multi-turn conversational RAG with human-generated dialogues (IBM
Research, 2025), mmRAG (Xu et al., 2025) tackles heterogeneous knowledge sources, and QE-RAG (Zhang et al., 2025)
evaluates robustness to query entry errors such as typos and keyboard proximity mistakes.
Query Taxonomies and Curation.Principled curation by defining taxonomies that reflect real-world difficulty. Teixeira de
Lima et al. (2025) proposed a taxonomy classifying context-query pairs into predefined categories such as fact single ,
summary ,reasoning , and unanswerable categories to identify specific retrieval challenges. The CRUD-RAG
benchmark further organizes tasks into Create, Read, Update, and Delete operations to evaluate RAG systems in scenarios
beyond standard question-answering, such as text continuation and hallucination modification (Lyu et al., 2025).
Corpus Stratification.Works like GraphRAG (Edge et al., 2025), RAPTOR (Sarthi et al., 2024), and HippoRAG (Guti ´errez
et al., 2024) have explored corpus stratification to improve retrieval efficiency and quality. GraphRAG’s stratification is
most similar to ours as it constructs clusters using the Leiden algorithm (Traag et al., 2019) after LLM-prompt based entity
and relationship extraction. Our approach differs in that relationships are detected directly in the semantic space and the
stratification is used to inform benchmark diagnosis and repair rather than retrieval optimization.
Synthetic Data Generation (SDG).To overcome the bottleneck of manual annotation, frameworks like RAGEval and
Long2RAG employ schema-based pipelines to generate grounded documents, questions, and answers (Zhu et al., 2025; Qi
et al., 2024). Similarly, DataMorgana provides a customizable framework to control the lexical, syntactic, and semantic
diversity of generated benchmarks, ensuring they reflect the expected traffic of enterprise applications (Filice et al., 2025).
Efficient Evaluation under Budget Constraints.Complementary work reduces annotation costs via strategic item selection:
active learning for IR test collections (Rahman et al., 2020), minimal test collections (Carterette et al., 2006), and compact
LLM benchmarks (Polo et al., 2024). These approaches select items based on model uncertainty or statistical informativeness;
in contrast, our framework prioritizes corpus-informed coverage independent of the systems being evaluated.
27