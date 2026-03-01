# Revisiting RAG Retrievers: An Information Theoretic Benchmark

**Authors**: Wenqing Zheng, Dmitri Kalaev, Noah Fatsi, Daniel Barcklow, Owen Reinert, Igor Melnyk, Senthil Kumar, C. Bayan Bruss

**Published**: 2026-02-25 04:19:06

**PDF URL**: [https://arxiv.org/pdf/2602.21553v1](https://arxiv.org/pdf/2602.21553v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems rely critically on the retriever module to surface relevant context for large language models. Although numerous retrievers have recently been proposed, each built on different ranking principles such as lexical matching, dense embeddings, or graph citations, there remains a lack of systematic understanding of how these mechanisms differ and overlap. Existing benchmarks primarily compare entire RAG pipelines or introduce new datasets, providing little guidance on selecting or combining retrievers themselves. Those that do compare retrievers directly use a limited set of evaluation tools which fail to capture complementary and overlapping strengths. This work presents MIGRASCOPE, a Mutual Information based RAG Retriever Analysis Scope. We revisit state-of-the-art retrievers and introduce principled metrics grounded in information and statistical estimation theory to quantify retrieval quality, redundancy, synergy, and marginal contribution. We further show that if chosen carefully, an ensemble of retrievers outperforms any single retriever. We leverage the developed tools over major RAG corpora to provide unique insights on contribution levels of the state-of-the-art retrievers. Our findings provide a fresh perspective on the structure of modern retrieval techniques and actionable guidance for designing robust and efficient RAG systems.

## Full Text


<!-- PDF content starts -->

Revisiting RAG Retrievers: An Information Theoretic Benchmark
Wenqing Zheng1Dmitri Kalaev1Noah Fatsi1Daniel Barcklow1Owen Reinert1Igor Melnyk1
Senthil Kumar1C. Bayan Bruss1
Abstract
Retrieval-Augmented Generation (RAG) systems
rely critically on the retriever module to surface
relevant context for large language models. Al-
though numerous retrievers have recently been
proposed, each built on different ranking princi-
ples such as lexical matching, dense embeddings,
or graph citations, there remains a lack of sys-
tematic understanding of how these mechanisms
differ and overlap. Existing benchmarks primarily
compare entire RAG pipelines or introduce new
datasets, providing little guidance on selecting
or combining retrievers themselves. Those that
do compare retrievers directly use a limited set
of evaluation tools which fail to capture comple-
mentary and overlapping strengths. This work
presents MIGRASCOPE, a Mutual Information
based RAG Retriever Analysis Scope. We revisit
state-of-the-art retrievers and introduce principled
metrics grounded in information and statistical es-
timation theory to quantify retrieval quality, redun-
dancy, synergy, and marginal contribution. We
further show that if chosen carefully, an ensem-
ble of retrievers outperforms any single retriever.
We leverage the developed tools over major RAG
corpora to provide unique insights on contribu-
tion levels of the state-of-the-art retrievers. Our
findings provide a fresh perspective on the struc-
ture of modern retrieval techniques and actionable
guidance for designing robust and efficient RAG
systems.
1. Introduction
Retrieval-Augmented Generation (RAG) has emerged as
a dominant paradigm for knowledge-intensive NLP tasks
(Lewis et al., 2020). By retrieving relevant external context
to inform large language models (LLMs), RAG systems
dramatically improve factual accuracy and reduce hallu-
1CapitalOne. Correspondence to: Wenqing Zheng <wen-
qing.zheng@capitalone.com>.
Preprint. February 26, 2026.cination (Shuster et al., 2021). The retriever module is
the core component responsible for selecting which evi-
dence is passed to the generator, and its effectiveness di-
rectly limits downstream performance. A rich ecosystem of
retrievers has flourished, including lexical-based methods
such as BM25 (Robertson et al., 2009), dense embedding
models trained for semantic similarity (Karpukhin et al.,
2020), graph-based retrievers designed for multi-hop and
global reasoning (Edge et al., 2024), and increasingly hy-
brid retrieval systems (Sarmah et al., 2024) . Each intro-
duces distinct assumptions and ranking mechanisms, yet
the field lacks a unified framework to compare them rigor-
ously. Many existing benchmarks focus on new datasets
or evaluation metrics applied to full RAG pipelines. While
these studies provide valuable insights into overall system
quality, they rarely isolate retrieval performance. Some
retriever-specific evaluation work has been done, covering
heterogeneous domain generalization (Thakur et al., 2021),
massive embedding leaderboards (Muennighoff et al., 2023),
and complex-objective tasks (Wang et al., 2024b). How-
ever these frameworks focus on aggregate ranking-based
metrics such as recall, MRR, and nDCG which assume in-
dependence between retrieved items and thus fail to explain
how different retrievers complement or supersede each other.
Recent ensemble-RAG efforts mostly concentrate on hierar-
chical integration of the full pipeline without disentangling
the unique contributions of the retriever itself. As a result,
practitioners face uncertainty when selecting the best re-
triever for a given application or dataset, and new retrievers
are introduced without clarity about the retrieval behaviors
they replace or replicate. We argue that revisiting the re-
triever module with dedicated evaluation tools is essential
to unlocking the next phase of RAG progress.
To address these gaps, we introduce MIGRASCOPE, an
information-theoretic framework for analyzing and bench-
marking retrievers in RAG. We model each retriever’s rank-
ing signal as a noisy view of an underlying chunk probability
distribution conditioned on ground-truth answers. Building
on Mutual Information (MI), we define a retriever quality
score that computes the diverging level from the presumed
ideal chunk distribution. This score generalizes across lexi-
cal, dense, and graph-based retrieval mechanisms, enabling
principled comparison and calibration.
1arXiv:2602.21553v1  [cs.IR]  25 Feb 2026

Revisiting RAG Retrievers: An Information Theoretic Benchmark
We further develop MI-grounded synergy and redun-
dancy metrics to quantify how multiple retrievers inter-
act—whether they contribute complementary evidence or
duplicate signals—and provide visual diagnostics to reveal
overlap structure. These tools support marginal contribution
estimation and guide retriever selection and combination.
Finally, we translate these analyses into practice with a
lightweight ensemble that uses MI-based attribution to select
and weight retrievers with minimal supervision. Applied
across major RAG corpora and a broad suite of state-of-the-
art (SOTA) retrievers, the framework yields new insights
into contribution patterns, redundancy, and robustness, and
the ensemble consistently outperforms strong single retriev-
ers. This paper makes the following contributions:
•Introduce MIGRASCOPE, a mutual-
information–based retriever quality score that
normalizes across ranking paradigms (lexical, dense,
graph) to quantify retrieval relevance and utility.
•Define and visualize MI-grounded synergy and redun-
dancy metrics for multiple retrievers, revealing com-
plementary and overlapping signal structure within
ensembles.
•Develop a retriever ensemble framework with MI-
based attribution to estimate each retriever’s marginal
contribution to overall retrieval performance.
•Benchmark a broad set of graph-RAG and standard
RAG retrievers using the proposed methodology, yield-
ing new insights on contribution patterns, robustness,
and design guidance for RAG systems.
Our results suggest a need to shift community focus from
endlessly inventing new retrievers toward understanding
how existing ones differ, overlap, and cooperate. Through
an information-theoretic lens, we uncover structure and
simplicity hidden beneath the diversity of modern retrieval
methods, paving the way for more reliable and interpretable
RAG systems.
2. Related Work
Our work is positioned at the intersection of three active re-
search areas: RAG benchmarking, RAG ensemble methods,
and the application of information-theoretic principles to
large language models.
2.1. RAG Benchmarking
Recent work on RAG evaluation has primarily focused on
establishing new datasets, defining system-level metrics,
or analyzing operational costs, rather than systematically
comparing the retrievers themselves. Several benchmarkshave been introduced to test RAG systems on complex
tasks, such as multi-hop reasoning over structured docu-
ments (Khang et al., 2025) or domain-specific GraphRAG
applications (Xiao et al., 2025). Others have focused on
creating datasets for specific industrial domains, such as
finance (Wang et al., 2024a), or on proposing new evalu-
ation methodologies, like assessing LLM robustness and
information integration (Chen et al., 2024a) or developing
explainable metrics for industrial settings (Friel et al., 2024).
To evaluate retrievers in isolation, the Information Retrieval
community has developed rigorous benchmarks covering
heterogeneous domain generalization (Thakur et al., 2021),
massive embedding leaderboards (Muennighoff et al., 2023),
and complex user objectives (Wang et al., 2024b). However,
these frameworks typically rely on aggregate ranking met-
rics such as nDCG and Recall. While effective for static
leaderboards, these metrics treat document utility as addi-
tive, implicitly assuming independence between retrieved
items. Consequently, they fail to quantify redundancy or
measure the conditional information gain required to under-
stand how distinct retrievers complement or supersede one
another in an ensemble.
While valuable, these contributions often treat the retriever
as a component within a larger pipeline. The focus remains
on dataset design or system-level assessment. Other studies
analyze important, but orthogonal, aspects. For instance,
Lin (2024) provide practical guidance on the operational
trade-offs of different retrieval infrastructures (e.g., HNSW
vs. inverted indexing) but do not offer prescriptive insights
on retriever algorithm selection. Similarly, Zhang et al.
(2025a) compare different LLMs as dense retrievers, but
limit their analysis to this single family.
Across these studies, there remains a significant gap in un-
derstanding the fundamental differences, redundancies, and
complementary strengths of the various retriever families
(e.g., dense, sparse, hybrid, and graph-based). Our work
directly addresses this gap by proposing a framework to
benchmark the retrievers themselves.
2.2. RAG Ensemble Methods
As RAG systems have matured, ensemble and multi-
retriever architectures have become increasingly common.
Many of these efforts focus on sophisticated pipeline or-
chestration. For example, Zhang et al. (2025b) propose a
hierarchical system with a planner that orchestrates multiple
low-level searchers, while Wu et al. (2025) introduce a mod-
ular framework that composes distinct steps like question
decomposition and retrieval. Other approaches integrate
heterogeneous data sources, such as combining web search
with knowledge graphs (Xie et al., 2024) or creating adap-
tive hybrid systems for specific domains like law (Kalra
et al., 2024).
2

Revisiting RAG Retrievers: An Information Theoretic Benchmark
(b) Pseudo ground-truth &  Retriever 
Divergence(a) Pipeline: Multi-retriever Eval
BM25
RAG
GraphRAG
QRAGCandidate Chunks C
LLM Generator
for Cross Entropy MI
...QA Pairs 
/ CorpusMulti DatasetsJSD
Retriever
Ensemble
🎖 Ranker Fusion
🎼 Synergym Retrievers;
chunk idx aligned
Ground TruthX∈Rd×m Y∈Rd×1Matrix obs. based 
MI estimation
Stack d 
chunk 
scores
across
 all 
questionschunk 
probs  
CP*(c) Retriever Ensemble  & 
MI Computation
ChunkScore / Prob.
Ground T ruth RetrieverChunk(d) Analyzing T ools
Prob(a|c1)
=0.2chunk 1query answer
chunk 2
chunk 3
chunk 4MI=0.0031
MI=0.0024
MI=0.0013
MI=0.0013Prob(a|c1)
=0.6
Prob(a|c1)
=0.1
Prob(a|c1)
=0.1
NormalizeProb(a|c1)
=0.1Label = 1
Label = 0
Label = 0
Label = 0Prob(a|c1)
=0.8
Prob(a|c1)
=0.05
Prob(a|c1)
=0.05
Label ReinforcementDataset
BenchmarkingPerformance 
AttributionBM25
RAG
GraphRAG
QRAG
...
RecallDivergence
Metric 
CorrelationRetrieveri @ datasetj
Redundancy
Synergy
DatasetMI / Recall m etrics
Retriever
3Retriever
1
Retriever
2
Figure A.The proposed MIGRASCOPE analyzing framework
A separate line of work seeks to optimize the information
passed to the generator. This includes methods for decou-
pling retrieval and generation representations (Yang et al.,
2025) or explicitly balancing relevance and semantic diver-
sity in the retrieved document set (Rezaei & Dieng, 2025).
Chen et al. (2025) explicitly analyze ensembles through an
information-theoretic lens, arguing that they increase the
total useful information. However, their work stops short of
providing an operational recipe for selecting or weighting
retrievers. These studies primarily offer architectural solu-
tions or in-domain systems, but do not provide a principled,
data-driven method for understandingwhichretrievers to
ensemble orhowto combine them based on their quantified
marginal contribution.
2.3. Information-Theoretic Analysis in RAG
Information theory provides a powerful, formal toolkit for
quantifying concepts like uncertainty, relevance, and redun-
dancy, which are central to RAG. Recently, these tools have
been applied to various parts of the RAG pipeline. For
instance, Zhu et al. (2024) use an information bottleneck
formulation to filter noise from retrieved contexts. Liu et al.
(2024) propose using Pointwise Mutual Information (PMI)
as a probabilistic gauge for RAG performance, while oth-
ers have used divergence metrics to measure attribution (Li
et al., 2025) or detect hallucinations (Halperin, 2025). These
methods successfully apply information-theoretic concepts
todiagnoseoroptimizea specific RAG component, such
as context pruning (Deng et al., 2025) or detecting data
memorization (Anh et al., 2025).
However, this body of work has not yet been directed at
the problem ofretriever evaluation. While Yumu¸ sak (2025)
outline a theoretical framework for RAG information flow,
it lacks the empirical benchmarking to reveal how different
SOTA retrievers actually perform under these metrics. To
our knowledge, no existing work uses mutual information
and information divergence to create a comparative frame-
work that quantifies the quality, redundancy, and marginal
contribution of distinct retriever families.
In summary, the field currently lacks a unified framework for
comparing retriever modules directly, a principled methodfor designing retriever ensembles based on redundancy, and
an empirical application of information theory to bench-
mark retriever contributions. Our work, MIGRASCOPE, is
designed to fill these precise gaps.
3. Analytical Framework
3.1. Preliminaries: GraphRAG Formulations
Let a query be q∈ Q , a candidate chunk set be C=
{c1, . . . , c N}, and an answer be a∈ A . A retriever rmaps
qto a scored or ranked subset Cq⊆C as the context. The
indices of chunks subset Cqare denoted as Iq. The re-
triever gives each chunk a score, combining into a scored
list{(cj, sj)}j∈Iq. A generator then produces the answer a
conditioned on(q,C q).
We formulate a simplified version of RAG and GraphRAGs
here. More details are described in Appendix A.
Vanilla RAG.The retrieval Cqrelies solely on vector simi-
larity to select the top-kchunks:
Cq=Topk({cj}N
j=1,Sim(Embed(q),Embed(c j)))(1)
GraphRAG (General Form).GraphRAG introduces a
knowledge graph Gconstructed from the corpus. Retrieval
is a two-stage process: graph anchoring and context map-
ping.
G=Construct(C)(2)
Cq=MapG→C(Anchor(G, q))(3)
3.2. Information-Theoretic View of RAG
Mutual information (MI).For random variables X, Y
with jointp(x, y)and marginalsp(x), p(y),
I(X;Y) =E p(x,y)
logp(x, y)
p(x)p(y)
=H(X)−H(X|Y) =H(Y)−H(Y|X).(4)
MI measures the reduction in uncertainty of one variable
given the other; it is zero iffXandYare independent.
3

Revisiting RAG Retrievers: An Information Theoretic Benchmark
Divergence.In this work we apply Jensen–Shannon di-
vergence ( JSD) which is symmetric, bounded in [0,log 2] ,
and well-defined even with disjoint supports.
KL(P∥Q) =E p
logp
q
,
JSD(P, Q) =1
2KL 
P∥M
+1
2KL 
Q∥M
,
M=1
2(P+Q).(5)
Pointwise mutual information in RAG.Given query q
and a random multiset of retrieved chunks Cq, define the
pointwise mutual information w.r.t. the answeraas
PMI(a;C q|q) = logp(a| C q, q)
p(a|q).(6)
Exact evaluation is intractable because the answer ais gen-
erative with a combinatorially large support. In practice we
approximate with a fixed ground-truth answer afrom the
dataset:
dPMI(a;C q|q)≈logp(a| C q, q)−logp(a|q).(7)
We estimate logp(a| ·) with the LLM token-level
cross-entropy (log perplexity). For a tokenization a=
(t1, . . . , t T)and model conditionZ,
logp(a|Z) =TX
t=1logp θ(tt|t<t, Z)
⇐⇒CE(a|Z) =−1
TTX
t=1logp θ(tt|t<t, Z).(8)
where θrepresents the generative LLM parameters. Large
PMI between the answer aand context Cqcorresponds to
large perplexity drop under retrieved context Z= (q,C q)
relative to no context Z= (q) . PMI, then, quantifies how
much a retrieved set of chunks Cq, for a given question q,
reduces uncertainty in the ground-truth answera.
3.3. Pseudo Ground-Truth Chunk Probability
We define a pseudo ground truth probability over the re-
trieved chunks that reflects how well each chunk supports
the known answer. Related chunk attribution techniques uti-
lize LLM cross-entropy to gauge chunk importance, either
by calculating Information Bottleneck scores (Zhu et al.,
2024) or by measuring the performance degradation when
a specific chunk is removed from the context (Deng et al.,
2025). However, such methods often introduce undesirable
complexity. Ablation-based approaches suffer from compu-
tational overhead and bias induced by chunk order, as LLMs
have been shown to disproportionately favor the first and last
chunks in a sequence (Liu et al., 2024). Meanwhile, Infor-
mation Bottleneck methods require tuning sensitive hyperpa-
rameters and optimizing auxiliary reconstruction objectives.To avoid these pitfalls, we apply a simple, independent scor-
ing approach to measure the chunk self-attribution score.
Thus we have the pseudo ground-truth chunk probability
CP∗:
CP∗(c|q, a) = Softmax Cq(logp θ(a|q, c))(9)
where Softmax Cqnormalizes the chunk probabilities based
on the logits over the subset of retrieved chunksC q.
Golden chunk reinforcement.If the dataset provides
golden supporting chunks Gq⊆ Iq, we optionally reinforce
them by a scalar γ >1 , i.e. amplify the dataset labeled
ground truth chunks then re-normalize1:
gCP∗(c|q, a) =1[c∈G q]γ·CP∗(c|q, a) +1[c /∈G q]·CP∗(c|q, a)
P
j∈Iq 
1[cj∈Gq]γ+1[c j/∈Gq]
CP∗(cj|q, a).
(10)
We use fCP∗as the default pseudo target distribution to
produce the following retriever analysis.
3.4. Retriever Divergence Score
Once we have the pseudo ground-truth chunk probability
CP∗as the standard to compare with, we can easily mea-
sure the quality of any retriever rby comparing the scores
produced by this retriever and CP∗. We first convert the
retriever scoress r(c)to probabilities:
pr(c|q) =exp(s r(c)/τ)P
j∈Iqexp(s r(cj)/τ)(11)
where τ >0 be a temperature-softmax normalization. The
quality of the retriever rcan be defined by the divergence
ofrfrom the pseudo target:
Div(r) =1
|Q|X
q∈QJSD
pr(· |q)fCP∗(· |q, a q)
.(12)
3.5. Ensembling Multiple Retrievers
Assume mretrievers {ri}m
i=1produce scores {si(c)}m
i=1.
There are multiple ways that these retrievers can be fused.
We list and compare 12 ways to merge the rankers and show
results in Section 4. We do not aim to claim novelty on the
ranker fusion approaches; we leverage existing approaches
as detailed in Appendix B.
3.6. Marginal Contributions and Redundancy Analysis
In the attempt to benchmark and analyze different retrievers,
we utilize well-established tools from statistical signal pro-
cessing to quantify the contribution and redundancy levels.
1When golden chunk support exists, the chunk distribution is
idealy one-hot. The CP∗softens the hard distribution. For better
alignment with dataset labels, we empirically pick largeγ.
4

Revisiting RAG Retrievers: An Information Theoretic Benchmark
Setup.Given a batch of queries {qi}|Q|
i=1, we stack the
retriever’s scores and CP∗to obtain observation Xand
estimation targetY.
For each query, we use fCP∗to produce a list of top Kchunk
scores, and concatenate them into target vector Y∈RK|Q|
On the other hand, each of the mretrievers also produces
scores for these top Kchunks. We stack them into the
observation matrixX∈RK|Q|×m.
Utility.Define the set function
F(S) =I 
Y;X S
,(13)
which measures predictive information about Ycontained
in the subset Sof retrievers. This is distinct from Fisher
information.
Independent marginal contribution.Denote the collec-
tion of mretrievers as R. For retriever i, its independent
marginal contribution is:
Cidef=I 
Y;X i|XR/{i}
=H 
Y|X R/{i}
−H 
Y|X R
,(14)
where XR/{i} denotes all columns of Xexcluding the col-
umn corresponding to retriever i,XRdenotes the full ob-
servation matrix. Large Cithen captures the unique infor-
mation conveyed by retriever i. Values near zero indicate
redundancy given the others.
Shapley value for fair attribution.Define Shapley values
over the coalition game({1, . . . , m}, F):
ϕi=X
S⊆{1,...,m}\{i}|S|! (m− |S| −1)!
m![F(S∪ {i})−F(S)].(15)
Shapley values fairly allocate the total utility F({1:m})−
F(∅) across retrievers by Σm
i=1ϕi=F({1:m})−F(∅) .
They average marginal information gains over all insertion
orders, which resolves order dependence.
Redundancy vs synergy.For any pair (i, j) , define the
interaction information
II(Y;X i;Xj) =I(Y;X i)−I 
Y;X i|Xj
.(16)
Positive IIindicates redundancy; negative IIindicates syn-
ergy, i.e., complementary information that emerges only
jointly. Accordingly, we define a surrogate distance between
retrieversr iandr jas
d(ri, rj) = exp 
−II(Y;X i;Xj)
.(17)We further leverage this score to position each retriever on a
redundancy–synergy spectrum, and perform clustering anal-
ysis to reveal the structural grouping of retrievers. Results
are presented in Section 4.6.
Retriever Selection Principle.We apply the tractable re-
dundancy/synergy scores in Eq. (16) to decide whether to
include certain RAG retriever for ensemble. We follow the
principle that multicollinearity harms estimation stability
but independence helps. For any sources X1, X2and tar-
getY,I(Y;X 1, X2) =I(Y;X 1) +I(Y;X 2|X1). IfX2
is redundant given X1, then I(Y;X 2|X1) = 0 . IfX2is
complementary, this conditional term is positive, so the pair
contains strictly more information than either alone.
3.7. Estimating Entropy and Mutual Information from
Vector Observations
Assuming X and Y are jointly Gaussian yields a closed-
form mutual information (MI) estimator (see Appendix C).
However, this assumption is theoretically tenuous in our
setting and, empirically, leads to systematic underestimation
of MI by failing to capture nonlinear dependencies between
the variables.
Therefore, we accept the distributions of XandYas un-
known, then train a predictive model ˆp(y|x) and compute
bH(Y|X) =1
nP
iH(ˆp(· |x i))for discrete Y, or ap-
proximate continuous Ywith a Gaussian residual model
where bH(Y|X) =1
2log(2πebσ2
res). We use XGB as the
regression model.
4. Results
4.1. Benchmarking RAG Retrievers
In this work, we focus on the effect of retriever mechanisms
rather than encoder, hence the encoder remains fixed and
same for all RAG approaches in our work. We use BGE-M3
(Chen et al., 2024b) indiscriminatively. We benchmark a
dozen of SOTA RAG retrievers, including chunk similarity
based (vanilla) retriever, lexical retrievers, graph retrievers,
and chunk decomposition based retrievers, etc. Especially,
for graph based retrievers, we unify the graph construction
and indexing approaches into seven types for benchmark-
ing and additionally compare with HippoRAG (Gutiérrez
et al., 2025) and LightRAG (Guo et al., 2024). Detailed
formulations of these retrievers are in Appendix A.2
4.2. Datasets and Split
We evaluate the retrievers on four multi-hop QA corpora:
HotpotQA, MuSiQue, 2WikiMultiHopQA, TriviaQA. For
2Our codes are available at https://github.com/
CapitalOne-Research/Migrascope
5

Revisiting RAG Retrievers: An Information Theoretic Benchmark
Method2Wiki HotpotQA MuSiQue TriviaQA
Rec MRR Div Rec MRR Div Rec MRR Div Rec MRR Div
G-window0.935 0.959 0.110 0.871 0.916 0.118 0.822 0.806 0.125 0.891 0.922 0.102
G-threshold0.936 0.960 0.120 0.921 0.924 0.121 0.725 0.805 0.122 0.944 0.922 0.101
G-hierarchical0.934 0.958 0.117 0.872 0.916 0.120 0.724 0.803 0.129 0.881 0.914 0.098
G-naive0.957 0.934 0.122 0.876 0.920 0.125 0.729 0.808 0.134 0.892 0.933 0.144
G-community0.935 0.959 0.125 0.873 0.917 0.123 0.727 0.807 0.130 0.891 0.919 0.119
G-relationship0.934 0.958 0.128 0.873 0.917 0.124 0.728 0.807 0.129 0.845 0.922 0.127
G-global0.937 0.961 0.125 0.880 0.922 0.128 0.726 0.805 0.131 0.835 0.955 0.121
HippoRAG0.931 0.960 0.142 0.892 0.921 0.128 0.833 0.810 0.120 0.921 0.920 0.119
RAG0.911 0.921 0.133 0.851 0.889 0.130 0.711 0.768 0.159 0.860 0.891 0.125
LLM0.913 0.944 0.131 0.824 0.908 0.138 0.807 0.801 0.155 0.882 0.914 0.129
QRAG0.889 0.919 0.147 0.713 0.792 0.133 0.681 0.753 0.166 0.829 0.881 0.144
LightRAG0.785 0.863 0.155 0.507 0.644 0.162 0.563 0.659 0.199 0.877 0.820 0.157
BM250.486 0.605 0.252 0.620 0.739 0.238 0.415 0.507 0.302 0.420 0.569 0.289
Table 1.RAG Retriever Benchmarking with MI metric
each corpus, we uniformly sample1,000QA pairs.
Benchmarking results for these retrievers are presented in
Table. 1, which shows the traditional metrics (Recall@1,
MRR) as well as the newly derived divergence metric from
Eq.(12). Next, we leverage the proposed tools to address
four key research questions.
•RQ1: How does the new MI based metric align with
previous MRR/Recall metrics?
•RQ2: How does the selection of hyper-parameters im-
pact MI?
•RQ3: Is it possible to combine multiple weak retrievers
into a stronger retriever, and how are they contributing
individually?
•RQ4: Among the SOTA retrievers, how redundant are
they?
4.3. Correlation Analysis (RQ1)
This experiment aims to check the level of alignment be-
tween the proposed divergence metric and traditional met-
rics. For each (retriever, dataset) pair, we plot the divergence
score against standard evaluation metrics (Recall@ 1, MRR)
in Figure 1.
The first observation is that recall andDivscores are nega-
tively correlated in general (strong retrievers tend to have
lower divergence). There are also systematic deviations,
as multiple retrievers achieve similar recall but differ sig-
nificantly in divergence. This confirms the complementary
nature of our divergence metric and its ability to diagnose
failure modes not captured by ranking accuracy alone.
4.4. Sensitivity of MI Hyperparameters (RQ2)
To address RQ2, we investigate the variation of the proposed
MI based metric under different hyperparameter configura-tions. Specifically, we examine three key hyperparameters:
(1)MI candidate depth: the top- Kto compute MI for
each question, which is the length of Cq. We visual-
ize a line sweep over the top- K, with value varies in
{3,5,10,20,50,100}.
(2)Anchor retriever: the basis of supporting chunk list
(Cq) that the divergence score is computed on top of. We
consider two choices under this setting. (i) The union of
all retrievers’ chunk sets, which means that MI scores are
computed on top of the union of all retriever chunks. (ii) One
single retriever. In practice we implemented HippoRAG as
the anchor of Cq, so that top- Kof HippoRAG’s retrieved
chunks are pulled as the basis for MI scores. Missing chunks
from other retrievers are filled with zero values.
(3)The reinforce strength coefficient( γin Eq. (10)). It
takes values from{2,10,100}.
We quantify the alignment of the recall and the negative
divergence score using Pearson Correlation. The results on
HotPotQA are plotted in Figure 2, where the X-axis is the
top-Kline sweep, and the legends are anchor retriever and
γ.
3 5 10 20 50 1000.00.10.20.30.40.50.60.70.8Pearson CorrelationGRAG-Window
3 5 10 20 50 100HippoRAG
ALL, =2
ALL, =10
ALL, =100
HippoRAG, =2
HippoRAG, =10
HippoRAG, =100
Figure 2.Hparam robustness visualization. Using the first bar of
the first subfigure (RAG) to interpret the meaning: in HotPotQA
dataset and for GRAG-Window retriever, the Pearson correlation
between theDivmetric and the Recall is 0.54 if theDivis computed
at top 3 chunks retrieved by all retrievers (otherwise, top 3 chunks
retrieved only by HippoRAG) andγis set to 2 for theDiv.
The first observation is that large γbrings up the correlation
6

Revisiting RAG Retrievers: An Information Theoretic Benchmark
0.4 0.5 0.6 0.7 0.8 0.9
Recall@K0.100.150.200.250.30DivergenceRecall vs Divergence
G-window
G-threshold
G-hierarchical
G-naive
G-community
G-relationship
G-globalBM25
RAG
LightRAG
QRAG
HippoRAG
LLM
0.4 0.5 0.6 0.7 0.8 0.9
Recall@K0.50.60.70.80.9MRRRecall vs MRR
0.5 0.6 0.7 0.8 0.9
MRR0.100.150.200.250.30DivergenceMRR vs Divergence
Figure 1.Metric Correlations Across Datasets, Legend by Retriever
values. This is because large γwill degenerate the pseudo
ground truth into one hot (or few-hot for multi-hop QA)
labels. In that case, the divergence is mainly capturing the
entropy on the golden chunks, bringing up its correlation
with recall.
We also observe that as top- Kincreases, the Pearson corre-
lation generally decreases, as reflected by all lines showing a
consistently decreasing trend. This suggests that the retriev-
ers exhibit different distributions than the pseudo-ground
truth distributions. Otherwise if the retriever distribution
is similar to the pseudo-ground truth distribution except by
picking the wrong peak (golden chunk), their distribution
shift should gradually close as top-Kincreases.
4.5. Retriever Ensemble and Shapley Contribution
Values (RQ3)
We conduct a series of experiments in the attempt to find
a subset of combined retrievers outperforming any single
strongest retriever, as well as visualizing the synergy and
redundancy across different retrievers.
Our retriever ensemble is based on the divergence metric.
For every given dataset, we first run retrieval procedure with
every retriever to get a large top-K chunk list for 1000 QA
pairs. Then we select a list of retriever subset to perform
ensemble search in the unified search space, then the best re-
triever ensemble algorithm is found within that space, using
the 20% training QA pairs. Finally, the ensembled retriever
is evaluated on the test 80% QA pairs. The detailed options
for the retreiver ensemble algorithm space are specified in
Appendix B.
We break down this research question into several key points
for a better comprehensive understanding.
•RQ3.1How much are these retrievers contributing to
the ensembled results?
•RQ3.2How much information does the best ensemble
strategy provide? What if it deviates from the best
configuration?
•RQ3.3Ensemble stability: if more retrievers are added
to the best ensemble configuration, or some retrievers
are dropped, how does it impact the performance?4.5.1. BESTRECALL BYSHAPLEYSHARES(RQ3.1)
We compute the Shapley attribution values for every re-
triever within the best retriever set in Figure 3.1. The height
of the bar plot means the overall recall performance relative
to 1.0. From this figure we can see that the strongest retriever
also tends to contribute the most. However, the winner is
not the ensemble of the top-performing GraphRAG retriev-
ers, but it often includes worse retrievers such as BM25.
This hints that different GraphRAG retrievers may contain
redundant components that harm synergy.
2WikipediaQA HotpotQA MuSiQue TriviaQA0.00.20.40.60.81.0Best Recall per shareG-threshold
G-hierarchical
G-naive
HippoRAG
RAG
LLM
QRAG
BM25
Figure 3.1.Shapley share for the best ensemble configs.
4.5.2. PERTURBATIONIMPACT ONMI (RQ3.2)
We evaluate RQ3.2 by computing I(fCP∗;XS)from
Eq.(13) for the best retriever subset and its perturbed varia-
tions, where Sis some retriever ensemble set. The results
are plotted in Figure 3.3. In this figure, the y-axis labels are
different retriever subsets, ranked by the MI. As can be seen
in the figure, the few top performing retriever subsets show
similar performance. However, when the perturbation of
retrievers are too large, the performance drops drastically.
0.00 0.02 0.04
MIGna+RAG+Gth+QRA
Gna+RAG+Gth+Hip
Gna+LLM+Gth+QRA+RAG
Gwi+RAG+Gth+QRA
Gna+RAG+Gre+QRA+Gth
QRA+Gth+RAG
BM2+RAG+QRA
Gna+RAG+Gth+Lit2WikipediaQA
0.00 0.02 0.04
MIGth+Hip+RAG+QRA
Gth+Hip+RAG+LLM
Gre+Hip+RAG+QRA
Gth+Hip+QRA+RAG
Gth+Hip+QRA+LLM+Lit
Gth+Gwi+QRA
Gth+Hip+RAG+Lit+Gre
Gth+BM2+QRAHotpotQA
Figure 3.3.MI under ensemble perturbation.
7

Revisiting RAG Retrievers: An Information Theoretic Benchmark
4.5.3. PERTURBATIONIMPACT ONRECALL(RQ3.3)
We answer RQ3.3 by first identifying the best retriever sub-
set for ensemble, then gradually add or drop retrievers on
top of this best subset, and finally concatenate all results
into a continuous line. We plot the results in Figure 3.2.
Gna+RAG+QRA+BM2+Ggl+Ghi+Gre+Gth+Gco+Gwi+Hip+BM2+LLM
Retrievers added0.40.60.8Recall
2WikipediaQA
Gth+Hip+RAG+QRA+Gre+Gwi+Ghi+Gco+BM2+Ggl+Gna+LLM+Lit
Retrievers added
HotpotQA
Figure 3.2.Recall under ensemble perturbation.
As can be seen in Figure 3.2, if the retriever subset is chosen
wisely, it performs better than single best retriever. As more
retrievers are added, redundancy dominates over synergy,
hence the performance dropped drastically.
4.6. Retriever Redundancy Spectrum (RQ4)
We formulate the visualization of the mretrievers as a geo-
metric embedding problem. We first pre-compute the pair-
wise redundancy for all retrievers. By interpreting the pair-
wise redundancy as a measure of similarity, we derive a
distance metric where highly redundant pairs correspond
to short distances as in Eq. 17. Using Classical Multidi-
mensional Scaling (MDS), we map the high-dimensional
relationship matrix into a 2D Euclidean space. This map-
ping preserves the global relational structure, allowing us
to visualize retriever clusters based on their informational
synergy, as shown in Figure 4.
2
 1
 0 1 2 32
1
0
1
22WikipediaQA
2
 1
 0 1 2 3HotpotQA
Gwi
GthGhi
GnaGco
GreGgl
HipRAG
LLMQRA
LitBM2
Figure 4.Retriever redundancy/synergy spectrum.
As can be observed in the figure, GraphRAG approaches
are similar to each other and further away from vanilla RAG
or lexical BM25 retriever. This implies that redundancy
might be clustered by approach types (graph vs. lexical vs.
vanilla).5. Conclusion And Limitations
We introduced MIGRASCOPE, an information-theoretic
framework for evaluating RAG retrievers using principled
metrics that quantify quality, redundancy, synergy, and
marginal contribution. By modeling rankings as noisy
views of an answer-conditioned chunk distribution, our ap-
proach provides actionable insights for retriever selection
and demonstrates that carefully designed ensembles can
outperform individual retrievers. This work offers a fresh
perspective on retrieval evaluation and contributes tools for
building more robust and efficient RAG systems.
As a pioneering effort in applying mutual information to
retriever evaluation, MIGRASCOPE has certain limitations
that suggest opportunities for refinement. The MI-based
metrics, while theoretically grounded, are less familiar than
conventional metrics like Recall@K or MRR, and their in-
terpretation may require additional context for practitioners.
The framework depends on hyperparameters and estima-
tor choices, which can influence numerical stability and
relative rankings, though sensitivity analysis can mitigate
these effects. Additionally, the reported MI values are ap-
proximations influenced by model assumptions, estimator
errors, and practical truncations, and should be viewed as
consistent estimates under specific settings.
Despite these limitations, MIGRASCOPE reveals new in-
sights into retriever interactions and highlights the potential
of ensemble methods. Future work could extend this frame-
work to integrate generation-side metrics, explore retriever-
specific internal mechanisms, and optimize computational
efficiency for large-scale deployments. By addressing these
directions, the community can build on this foundation to
further advance retrieval-augmented generation systems.
References
Anh, L. V ., Anh, N. V ., Dik, M., and Van Nghia, L. Repcs:
Diagnosing data memorization in llm-powered retrieval-
augmented generation.arXiv preprint arXiv:2506.15513,
2025.
Chen, J., Lin, H., Han, X., and Sun, L. Benchmarking large
language models in retrieval-augmented generation. In
Proceedings of the AAAI Conference on Artificial Intelli-
gence, volume 38, pp. 17754–17762, 2024a.
Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D.,
and Liu, Z. Bge m3-embedding: Multi-lingual,
multi-functionality, multi-granularity text embeddings
through self-knowledge distillation.arXiv preprint
arXiv:2402.03216, 2024b.
Chen, Y ., Dong, G., Zhu, Y ., and Dou, Z. Revisiting
rag ensemble: A theoretical and mechanistic analy-
8

Revisiting RAG Retrievers: An Information Theoretic Benchmark
sis of multi-rag system collaboration.arXiv preprint
arXiv:2508.13828, 2025.
Deng, J., Shen, Y ., Pei, Z., Chen, Y ., and Huang, L. Influence
guided context selection for effective retrieval-augmented
generation.arXiv preprint arXiv:2509.21359, 2025.
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A.,
Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O.,
and Larson, J. From local to global: A graph rag ap-
proach to query-focused summarization.arXiv preprint
arXiv:2404.16130, 2024.
Friel, R., Belyi, M., and Sanyal, A. Ragbench: Explainable
benchmark for retrieval-augmented generation systems.
arXiv preprint arXiv:2407.11005, 2024.
Guo, Z., Xia, L., Yu, Y ., Ao, T., and Huang, C. Lightrag:
Simple and fast retrieval-augmented generation.arXiv
preprint arXiv:2410.05779, 2024.
Gutiérrez, B. J., Shu, Y ., Qi, W., Zhou, S., and Su, Y . From
rag to memory: Non-parametric continual learning for
large language models.arXiv preprint arXiv:2502.14802,
2025.
Halperin, I. Prompt-response semantic divergence met-
rics for faithfulness hallucination and misalignment
detection in large language models.arXiv preprint
arXiv:2508.10192, 2025.
Kalra, R., Wu, Z., Gulley, A., Hilliard, A., Guan, X.,
Koshiyama, A., and Treleaven, P. Hypa-rag: A hybrid
parameter adaptive retrieval-augmented generation sys-
tem for ai legal and policy applications.arXiv preprint
arXiv:2409.09046, 2024.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P. S., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense passage
retrieval for open-domain question answering. InEMNLP
(1), pp. 6769–6781, 2020.
Khang, M., Park, S., Hong, T., and Jung, D. Crest: A
comprehensive benchmark for retrieval-augmented gener-
ation with complex reasoning over structured documents.
arXiv preprint arXiv:2505.17503, 2025.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information pro-
cessing systems, 33:9459–9474, 2020.
Li, R., Chen, C., Hu, Y ., Gao, Y ., Wang, X., and Yilmaz,
E. Attributing response to context: A jensen-shannon
divergence driven mechanistic study of context attribu-
tion in retrieval-augmented generation.arXiv preprint
arXiv:2505.16415, 2025.Lin, J. Operational advice for dense and sparse retriev-
ers: Hnsw, flat, or inverted indexes?arXiv preprint
arXiv:2409.06464, 2024.
Liu, T., Qi, J., He, P., Bisazza, A., Sachan, M., and Cot-
terell, R. Pointwise mutual information as a performance
gauge for retrieval-augmented generation.arXiv preprint
arXiv:2411.07773, 2024.
Muennighoff, N., Tazi, N., Magne, L., and Reimers, N.
Mteb: Massive text embedding benchmark. InProceed-
ings of the 17th Conference of the European Chapter of
the Association for Computational Linguistics, pp. 2014–
2037, 2023.
Rezaei, M. R. and Dieng, A. B. Vendi-rag: Adaptively
trading-off diversity and quality significantly improves
retrieval augmented generation with llms.arXiv preprint
arXiv:2502.11228, 2025.
Robertson, S., Zaragoza, H., et al. The probabilistic rele-
vance framework: BM25 and beyond.Foundations and
Trends® in Information Retrieval, 3(4):333–389, 2009.
Sarmah, B., Mehta, D., Hall, B., Rao, R., Patel, S., and
Pasquali, S. Hybridrag: Integrating knowledge graphs
and vector retrieval augmented generation for efficient
information extraction. InProceedings of the 5th ACM
International Conference on AI in Finance, pp. 608–616,
2024.
Shuster, K., Poff, S., Chen, M., Kiela, D., and Weston, J.
Retrieval augmentation reduces hallucination in conver-
sation.arXiv preprint arXiv:2104.07567, 2021.
Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and
Gurevych, I. Beir: A heterogenous benchmark for zero-
shot evaluation of information retrieval models.arXiv
preprint arXiv:2104.08663, 2021.
Wang, S., Tan, J., Dou, Z., and Wen, J.-R. Omnieval: An
omnidirectional and automatic rag evaluation benchmark
in financial domain.arXiv preprint arXiv:2412.13018,
2024a.
Wang, X., Wang, J., Cao, W., Wang, K., Paturi, R., and
Bergen, L. Birco: A benchmark of information re-
trieval tasks with complex objectives.arXiv preprint
arXiv:2402.14151, 2024b.
Wu, R., Lee, Y ., Shu, F., Xu, D., Hwang, S.-w., Yao, Z., He,
Y ., and Yan, F. Composerag: A modular and composable
rag for corpus-grounded multi-hop question answering.
arXiv preprint arXiv:2506.00232, 2025.
Xiao, Y ., Dong, J., Zhou, C., Dong, S., Zhang, Q.-w., Yin,
D., Sun, X., and Huang, X. Graphrag-bench: Challenging
9

Revisiting RAG Retrievers: An Information Theoretic Benchmark
domain-specific reasoning for evaluating graph retrieval-
augmented generation.arXiv preprint arXiv:2506.02404,
2025.
Xie, W., Liang, X., Liu, Y ., Ni, K., Cheng, H., and Hu,
Z. Weknow-rag: An adaptive approach for retrieval-
augmented generation integrating web search and knowl-
edge graphs.arXiv preprint arXiv:2408.07611, 2024.
Yang, P., Li, X., Hu, Z., Wang, J., Yin, J., Wang, H., He, L.,
Yang, S., Wang, S., Huang, Y ., et al. Heterag: A hetero-
geneous retrieval-augmented generation framework with
decoupled knowledge representations.arXiv preprint
arXiv:2504.10529, 2025.
Yumu¸ sak, S. An information-theoretic framework for
retrieval-augmented generation systems.Electronics, 14
(15):2925, 2025.
Zhang, H., Bi, K., and Guo, J. A comparative study
of specialized llms as dense retrievers.arXiv preprint
arXiv:2507.03958, 2025a.
Zhang, Z., Feng, Y ., and Zhang, M. Levelrag: Enhancing
retrieval-augmented generation with multi-hop logic plan-
ning over rewriting augmented searchers.arXiv preprint
arXiv:2502.18139, 2025b.
Zhu, K., Feng, X., Du, X., Gu, Y ., Yu, W., Wang, H., Chen,
Q., Chu, Z., Chen, J., and Qin, B. An information bottle-
neck perspective for effective noise filtering on retrieval-
augmented generation.arXiv preprint arXiv:2406.01549,
2024.A. Detailed RAG/GraphRAG Formulations
This section formalizes the graph construction and retrieval
variants used in our evaluation, emphasizing succinct defini-
tions and avoiding redundant statements.
A.1. Method 1: Windowed Co-occurrence Graph
TheG-Windowmethod limits co-occurrence edges to enti-
ties within a fixed offset in the ordered entity list Ecfor any
chunkc.
EWindow =n
{ei, ej} |e i, ej∈ E, i̸=j,∃c∈C,∃k, l:
(Ec,k=ei∧ Ec,l=ej∧ |k−l| ≤window_size)o
(18)
A.2. Method 2: Thresholded Co-occurrence Graph
TheG-Thresholdmethod filters incidental co-occurrences
by requiring a minimum count across the corpus:
EThreshold ={{e i, ej} |e i, ej∈ E, i̸=j,
X
c∈CI({e i, ej} ⊆ E c)≥min_cooc}(19)
A.3. Method 3: Hierarchical and Algorithmic Graph
Construction Methods
G-Hierarchicalconstructs a heterogeneous directed graph
G= (V, E) with document, chunk, and entity nodes, en-
abling explicit provenance.
•(doc:d,chunk:c)with type: CONTAINS.
•(chunk:c,entity:e) with type: MENTIONS fore∈
Ec.
•undirected {ei, ej}with type: CO_OCCURS for
ei, ej∈ Ec.
A.4. Method 4: Naive Co-occurrence Graph
G-Naiveestablishes an undirected edge between any two
entities that co-occur in at least one chunk:
ENaive=
{ei, ej} |ei, ej∈ E, i̸=j,∃c∈C:{e i, ej} ⊆ E c	
(20)
A.5. Method 5: Community Graph
G-Communityfirst forms Gtemp= (E, E Threshold ), then
applies Louvain to obtain clustersS k:
{S1,S2, . . .}=Louvain(G temp,weight: cooccurrence)(21)
Community nodes vkare added with edges EComm =
{(e, v k)|e∈ S k,type: BELONGS_TO}.
10

Revisiting RAG Retrievers: An Information Theoretic Benchmark
A.6. Method 6: Relationship Graph
G-Relationship-Globalfocuses on relation-
ships among significant entities Esig (frequency
≥MIN_ENTITY_OCCURRENCE ), aggregating their
global context:
Ce={c∈C|e∈ E c}(22)
Directed relationships are extracted by an LLM over
Context(C e):
ERelGlobal =[
e∈E sigLLM extract_global (Context(C e), e,E cand)(23)
A.7. Method 7: Global Co-occurrence Graph
G-Globalderives the corpus-level unique set of co-
occurring entity pairs (i.e., the non-attributed, de-duplicated
union of chunk-level co-occurrences; equivalent in edge set
to Method 4 but without per-chunk metadata).
A.8. Method 8: HippoRAG
HippoRAG(Gutiérrez et al., 2025) models an heteroge-
neous memory graphG mem= (V, E):
1.Graph Construction: V=E ∪ F ∪C , where F
are OpenIE triples. Edges include containment and
synonymy (viaEmbedsimilarity).
2.Fact Reranking:Retrieve fact candidates Fcand=
VDB F(q)and refine with an LLM reranker:
Fq=LLM Rerank(Fcand, q)(24)
3.PPR Initialization:Extract query entities from Fq
and formp 0:
p0=InitDistribution(MapF→E(Fq))(25)
4.PPR Graph Search:Run PPR onG mem:
PPR scores=PPR(G mem,p0)(26)
5.Context Selection:Select top-scoring chunks:
Cq=TopkC({cj}j∈C,PPR scores[cj])(27)
A.9. Method 9: RAG
Standard RAG using embedding-based retrieval over chunks
followed by generation.
A.10. Method 10: LLM Reranking
LLMreranking orders candidate chunks Ccandwith the
LLM:
RLLM =LLM rank(q,C cand)(28)
The final context Cqis the top kCbyRLLM . In our tests,
candidates were produced via the windowed co-occurrence
graph, but the reranker is graph-agnostic.A.11. Method 11: Question as Sub-chunk Summary
(QRAG)
Graph Construction (Synthetic Questions).For each
chunkc i, generate a set of synthetic questions:
Qi={q i,1, . . . , q i,Mi}=LLM Gen(ci)(29)
Retrieval Scoring.Score ciby the maximum similarity
between the query and its synthetic questions:
si= max
qi,j∈QiSim(Embed(q),Embed(q i,j))(30)
Final Context.Select the topkchunks:
Cq=Topk({ci}N
i=1, si)(31)
A.12. Method 12: LightRAG
Graph Construction.Build a directed, labeled graph
GLightRAG = (ˆE,ˆR)from LLM-extracted triples, applying
deduplication and refinement:
GLightRAG =Dedupe◦Prof(V init,Einit)(32)
Retrieval.
1.Search & Neighborhood Expansion:Retrieve Esim=
VDB E(q),Rsim=VDB R(q), then expand to Ehood,
Rhoodvia graph traversal.
2.Token Truncation:Filter and truncate to a token bud-
getTB:
Efinal,R final=
Truncate(Filter(E sim∪ E hood,R sim∪ R hood),TB)
(33)
3.Chunk Merging:Map Efinal,Rfinalback to source
chunks and merge.
4.Context Formatting:Produce the final context:
Cq=Format(E final,R final,MergeER→C (Efinal,R final))
(34)
A.13. Method 13: BM25
Unmodified lexical BM25 retrieval.
B. Ranker fusion
We formulate the ranker fusion search space as follows.
Score standardization.We apply per-retriever z-score
normalization
˜si(c) =si(c)−µ i
σi+ϵ, µ i= mean csi(c),
σi= std csi(c).(35)
11

Revisiting RAG Retrievers: An Information Theoretic Benchmark
Weighted linear fusion.
ˆs(c) =mX
i=1wi˜si(c), w i≥0,X
iwi= 1.(36)
Weights can be set from divergence wi∝exp(−λDiv(r i)),
or learned by minimizing a surrogate loss such as cross-
entropy to fCP∗:
min
wX
qCE
fCP∗(· |q, a q),softmax(ˆs(·)/τ)
.(37)
Temperature fusion of probabilities.Convert each re-
triever to a calibrated distribution
pi(c|q) = softmax˜si(c)
τi
,
ˆp(c|q) =Qm
i=1pi(c|q)wi
P
jQm
i=1pi(cj|q)wi(38)
which is a log-opinion pool.
Z-score fusion
ϕr(c) =sr(c)−µ r
σr+ϵ, S(c) =mX
r=1wrϕr(c)(39)
Z-score fusion (39) aligns scales using per-list mean µrand
stdσ r.
Logit pooling
S(c) =mX
r=1wrlogit 
ˆpr(c)
,logit(p) = logp
1−p
(40)
Logit pooling (40) adds log-odds, equivalent to naive
Bayesian evidence combination under independence.
Noisy-or
ˆp(c) = 1−mY
r=1 
1−ˆp r(c)wr(41)
Noisy-or (41) estimates the probability that at least one
retriever deemscrelevant.
Adaptive weights from divergence.
wr(q, a) = 
ϵ+ Div(r|q, a)−1
Pm
j=1 
ϵ+ Div(j|q, a)−1(42)
Inverse-divergence weights (42) emphasize better-aligned
retrievers.Reciprocal Rank Fusion (RRF).Let rank i(c)be the
rank ofcunderr i, then
ˆs(c) =mX
i=11
k+ rank i(c),(43)
withk >0controlling tail influence.
Borda count with weights.
ˆs(c) =mX
i=1wi 
|Iq| −rank i(c) + 1
.(44)
Robust Rank Aggregation (RRA).For each c, compute
a one-sidedp-value under the null of random rankings,
pi(c) =rank i(c)
|Iq|,
pagg(c) = min
t∈{1,...,m}BetaCDF
p(t)(c);t, m−t+1
,
(45)
and rank by−logp agg(c).
Bayesian model averaging (BMA).Assuming per-
retriever likelihoodsp i(a|q, c)and model priorsπ i, form
ˆp(c|q)∝mX
i=1πipi(a|q, c)
orˆs(c) =X
ilog 
πipi(a|q, c)
.(46)
Ifpi(a|q, c) is unavailable, use surrogates based on fCP∗
or calibratedp i(c|q).
Markov-chain rank aggregation (Rank Centrality).
Construct a transition matrix
Muv=1
ZmX
i=11[u̸=v]·σ 
˜si(v)−˜s i(u)
,(47)
withσ(x) = 1/(1 +e−x)and row-normalization Z. The
stationary distribution πofMdefines the fused ranking by
π(c).
C. Gaussian assumption.
This section present an alternative way to estimate MI under
vector observations in Section 3.7. While the true distribu-
tions of the retriever scores Xand the pseudo ground truth
scores Yare unknown, adopting a Gaussian assumption
facilitates computational tractability by making possible a
closed-form solution.
12

Revisiting RAG Retrievers: An Information Theoretic Benchmark
Let(Y, X S)be jointly Gaussian with covariance Σ =ΣY Y ΣY X
ΣXY ΣXX
.
H(Y) =1
2log 
(2πe)dYdet Σ Y Y
.(48)
ΣY|XS= ΣY Y−ΣY XΣ−1
XXΣXY.(49)
H(Y|X S) =1
2log 
(2πe)dYdet Σ Y|XS
.(50)
I(Y;X S) =1
2logdet Σ Y Y
det Σ Y|XS.(51)
Equations (48)–(51) use Schur complements to yield closed-
form entropy and mutual information values.
Empirically, we observed that Gaussian assumption leads
to MI values about one to two magnitude smaller, and
the correlation with Recall is also lower, indicating over-
simplification. We leverage the non-Gaussian assumption
for MI estimation throughout this research.
13