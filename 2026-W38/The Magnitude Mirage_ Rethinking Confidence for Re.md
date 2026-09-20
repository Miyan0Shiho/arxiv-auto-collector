# The Magnitude Mirage: Rethinking Confidence for Reasoning-Intensive Retrieval

**Authors**: Jamie Holdcroft, Abdelrahman Abdallah, Adam Jatowt

**Published**: 2026-09-14 13:49:24

**PDF URL**: [https://arxiv.org/pdf/2609.15578v1](https://arxiv.org/pdf/2609.15578v1)

## Abstract
Many production RAG systems implement retrieval abstention by thresholding raw similarity scores, implicitly treating score magnitude as a confidence signal. We demonstrate that this practice degrades systematically as queries require reasoning beyond semantic matching. Across 11 retrieval architectures and 28 datasets, neural retrievers consistently assign high similarity scores to semantically related but constraint-violating documents, causing magnitude-based thresholds to collapse toward near-random abstention performance on logical and temporal reasoning tasks---a failure we term the Magnitude Mirage. To address this without computationally expensive alternatives, we conduct a large-scale empirical study of six zero-cost Query Performance Prediction (QPP) metrics across three cognitive tiers: semantic matching (BEIR), logical reasoning (BRIGHT), and temporal reasoning (TEMPO). Our central finding is that the key improvement comes from abandoning magnitude in favor of score-distribution signals: the gain from this shift exceeds the differences among distributional alternatives by a factor of 5-10$\times$. In particular, Score Gap ($s_1 - s_k$) and a practical adaptation of Score Magnitude and Variance (LSMV) improve abstention AUROC by up to 0.16 in settings where magnitude-based confidence provides little discriminative power. These methods require no additional inference, retraining, or latency, making them a practical zero-cost replacement for magnitude thresholding in deployed RAG systems.

## Full Text


<!-- PDF content starts -->

The Magnitude Mirage: Rethinking Confidence for Reasoning-Intensive
Retrieval
Jamie Holdcroft1, Abdelrahman Abdallah2, Adam Jatowt2
1UNSW Sydney2University of Innsbruck
j.holdcroft@student.unsw.edu.au
{abdelrahman.abdallah,adam.jatowt}@uibk.ac.at
Abstract
Many production RAG systems implement re-
trieval abstention by thresholding raw simi-
larity scores, implicitly treating score mag-
nitude as a confidence signal. We demon-
strate that this practice degrades systematically
as queries require reasoning beyond semantic
matching. Across 11 retrieval architectures and
28 datasets, neural retrievers consistently as-
sign high similarity scores to semantically re-
lated but constraint-violating documents, caus-
ing magnitude-based thresholds to collapse to-
ward near-random abstention performance on
logical and temporal reasoning tasks—a failure
we term the Magnitude Mirage. To address this
without computationally expensive alternatives,
we conduct a large-scale empirical study of six
zero-cost Query Performance Prediction (QPP)
metrics across three cognitive tiers: semantic
matching (BEIR), logical reasoning (BRIGHT),
and temporal reasoning (TEMPO). Our cen-
tral finding is that the key improvement comes
from abandoning magnitude in favor of score-
distribution signals: the gain from this shift
exceeds the differences among distributional
alternatives by a factor of 5–10 ×. In particular,
Score Gap ( s1−sk) and a practical adaptation
of Score Magnitude and Variance (LSMV) im-
prove abstention AUROC by up to 0.16 in set-
tings where magnitude-based confidence pro-
vides little discriminative power. These meth-
ods require no additional inference, retraining,
or latency, making them a practical zero-cost
replacement for magnitude thresholding in de-
ployed RAG systems.
1 Introduction
Retrieval-Augmented Generation (RAG) systems
rely on external retrieval components to provide
factual context to large language models (Lewis
et al., 2020; Guu et al., 2020; Izacard et al., 2023).
The reliability of such systems depends critically
on their ability to detect retrieval failures. When
the retrieved context does not contain relevant in-formation, downstream generation models may hal-
lucinate or produce incorrect answers (Asai et al.,
2024; Lee et al., 2025; Sun et al., 2025b). Con-
sequently, practical RAG pipelines require mecha-
nisms that can predict retrieval success and abstain
when confidence is low (Thakur et al., 2023; Asai
et al., 2024).
In many industrial RAG frameworks, this deci-
sion is often implemented using simple threshold-
ing of raw retrieval scores, a design reflected in
widely used toolkits such as LangChain and Lla-
maIndex (Chase, 2022; Liu, 2022). While com-
putationally inexpensive, this approach implicitly
assumes that absolute score magnitudes are com-
parable across queries and can reliably distinguish
successful from unsuccessful retrieval.
Modern dense retrievers are typically trained to
optimize semantic similarity between query and
document embeddings (Karpukhin et al., 2020;
Thakur et al., 2021). Consequently, a retriever may
assign a high similarity score to a document that
matches the general semantics of a query while
still violating key logical or temporal constraints
necessary for correctness. In such cases, the raw
retrieval score may appear highly confident even
when the retrieved document is irrelevant to the
true task requirement. We refer to this behavior
as theMagnitude Mirage: the misleading confi-
dence conveyed by high absolute retrieval scores
under reasoning-intensive retrieval constraints. As
illustrated in Figure 1, two queries with identical
maximum scores can exhibit fundamentally differ-
ent score distributions.
While recent work has explored LLM-based
approaches for evaluating retrieved context, such
methods introduce substantial latency and compu-
tational cost. This motivates revisiting classical
Query Performance Prediction (QPP) techniques,
which estimate retrieval success directly from the
distribution of retrieval scores.
In this work, we conduct a large-scale empirical
arXiv:2609.15578v1  [cs.IR]  14 Sep 2026

10 8 0e
0 2 4 6
𝑺𝟏 = 0.85 𝑺𝟏 = 0.85Figure 1: The Magnitude Mirage vs. Reality.Section A:Two queries sharing identical top scores ( s1= 0.85 )
exhibit fundamentally different score distributions. Failed retrieval produces a flat distribution; successful retrieval
isolates the correct document via a large Score Gap.Section B:MaxScore thresholding cannot reliably distinguish
successful from failed retrievals (AUROC: 0.583), while Score Gap thresholding achieves meaningful discrimination
(AUROC: 0.724)—AUROC@25 averaged across the evaluated BEIR datasets for Instructor-Large.
study of zero-cost QPP methods in modern neural
retrieval pipelines. To assess robustness across task
complexity, we evaluate six QPP metrics across
eleven retrieval architectures and twenty-eight
datasets covering three cognitive tiers: semantic
retrieval (BEIR) (Thakur et al., 2021), logical rea-
soning (BRIGHT) (Su et al., 2025), and temporal
reasoning (TEMPO) (Abdallah et al., 2026). We
measure predictive performance using AUROC
for binary query success ( NDCG@k >0 ), which
directly corresponds to the abstention decision
in RAG systems. In open-domain QA and RAG
pipelines, successful retrieval often requires only a
single relevant passage within the retrieved context
window to enable correct downstream generation
(Lewis et al., 2020; Karpukhin et al., 2020). We
additionally report Pearson and Spearman corre-
lations with NDCG@k to evaluate continuous
prediction quality.
Our contributions are as follows:
•We provide a large-scale study of zero-cost
QPP under reasoning-intensive retrieval, eval-
uating 6 metrics across 11 retrievers and 28
datasets spanning semantic, logical, and tem-
poral retrieval.
•We demonstrate that magnitude-based confi-
dence collapses systematically on reasoning-
intensive benchmarks across all architecturestested, a regime largely unexplored in prior
QPP work.
•We show that the primary gain comes from
abandoning magnitude in favor of distribu-
tional estimation: all variance-based metrics
substantially outperform MaxScore, while dif-
fering only marginally from one another.
•We provide end-to-end validation, show-
ing that variance-based abstention improves
downstream RAG correctness prediction rela-
tive to magnitude thresholding.
2 Related Work and Background
2.1 Retrieval Reliability and RAG Abstention
Irrelevant or misleading documents in a retrieved
context can substantially degrade downstream gen-
eration, even when only a single distracting pas-
sage is included (Cuconasu et al., 2024). While a
growing body of work leverages generative models
to evaluate retrieved documents (LLM-as-a-judge)
(Gu et al., 2025; Liu et al., 2025), these approaches
incur substantial latency. Other methods improve
retrieval reliability through retriever alignment or
reinforcement learning (Zhou and Chen, 2026), or
cross-encoder reranking, but require retraining or
additional inference and do not provide lightweight
abstention signals. In contrast, our work inves-
tigateszero-costconfidence signals that operate

directly on the retriever’s score distribution at in-
ference time, requiring no additional model calls
or retraining.
2.2 Query Performance Prediction for Neural
Retrieval
Query Performance Prediction (QPP) estimates re-
trieval success without human relevance judgments.
QPP has been extensively studied in classical infor-
mation retrieval (Tao and Wu, 2014; He and Ounis,
2004; Carmel and Yom-Tov, 2010). Post-retrieval
variance-based methods quantify the dispersion of
top-ranked scores based on the intuition that confi-
dent retrieval produces strong separation between
relevant documents and the background distribu-
tion (He and Ounis, 2004; Tao and Wu, 2014).
Recent work has extended QPP to neural dense
retrieval. Datta et al. (2025a) show that classical
QPP signals can differ substantially from sparse
settings and that magnitude-based signals often de-
grade in dense retrieval (Datta et al., 2025a). Dense-
specific adaptations include DenseQPP (Arabzadeh
et al., 2023), QPP-PRP (Datta et al., 2025a), and
PDQPP (Datta et al., 2025b); complementary work
has explored calibrating cosine similarity directly
(Rossi et al., 2024). Chifu et al. (2025) provide
a cross-paradigm evaluation of QPP robustness,
finding that predictive power varies substantially
across collections and rankers and that QPP-driven
selective query processing yields only marginal
gains. However, prior evaluations largely focus on
standard semantic benchmarks. We argue the limi-
tation lies not merely in calibration, but in relying
on magnitude itself as a confidence signal—a fail-
ure that becomes substantially more severe under
reasoning-intensive retrieval, a regime prior QPP
work has not examined.
2.3 Reasoning-Intensive Retrieval
Benchmarks such as BRIGHT and TEMPO intro-
duce tasks requiring logical or temporal reasoning
beyond semantic similarity, where documents may
be semantically close to a query while violating its
key constraints. How retrieval confidence signals
behave in such settings remains largely unexplored.
Our work addresses this gap by evaluating QPP
metrics across semantic, logical, and temporal re-
trieval tasks.3 Methodology
3.1 Problem Formulation
We formalize the standard Retrieval-Augmented
Generation (RAG) first-stage retrieval pro-
cess. Given a query qand a corpus
C={d 1, d2, . . . , d N}, a retrieval model R
maps the query and corpus to a ranked list of the
top-kdocuments
Dk= [d (1), d(2), . . . , d (k)],
where each retrieved document d(i)is associated
with a retrieval scores i∈R, such that
s1≥s2≥ ··· ≥s k.
LetSk={s 1, s2, . . . , s k}denote the set of top- k
retrieval scores.
Among the various classes of QPP methods, we
focus specifically onzero-costQPP predictors that
operate solely on the score distribution Sk. For-
mally, a QPP function
f(Sk)→c q
maps the score distribution to a scalar confidence
estimate cqthat approximates the true retrieval ef-
fectiveness of the query.
3.2 The Magnitude Mirage
An increasingly common heuristic for zero-cost
RAG abstention relies on absolute magnitude
thresholding. In practice, RAG frameworks such as
LangChain and LlamaIndex apply fixed similarity
thresholds to filter retrieved documents, passing
only those where si> τto the generator. This de-
sign implicitly assumes that raw retrieval scores are
calibrated confidence signals and that a high score
reliably indicates a relevant document regardless
of query type. Under this assumption, the maxi-
mum score s1serves as a natural proxy for overall
retrieval confidence: if the top-ranked document
scores highly, the retrieval is considered successful.
As retrieval tasks shift from semantic matching
to reasoning-oriented queries, the calibration of
absolute retrieval scores degrades. As shown in
Figure 1, even when two queries share an identi-
cal top score, MaxScore thresholding cannot reli-
ably distinguish successful from failed retrieval —
achieving AUROC of only .583 compared to .724
for Score Gap on the same queries. This failure re-
flects a fundamental mismatch between what mag-
nitude measures (semantic proximity) and what
abstention requires (constraint satisfaction).

3.3 Variance-Based QPP Metrics
To address the potential miscalibration of raw score
magnitudes, we evaluate six zero-cost QPP metrics
computed directly from the top- kretrieval score
distribution Sk={s 1, . . . , s k}. Distribution-based
metrics are motivated by the intuition that success-
ful retrieval produces greater separation between
relevant documents and the background score dis-
tribution.
Absolute Magnitude (Baseline):
MaxScore=s 1
Score Gap:
Gap@k=s 1−sk
Top-kStandard Deviation:Let µ=1
kPk
i=1si.
We define
σ@k=vuut1
kkX
i=1(si−µ)2
Normalized Query Commitment (NQC):
Adapting the classical NQC formulation of Shtok
et al. (2012) to dense retrieval, we normalize the
score standard deviation by the mean absolute
top-kscore:
NQC@k=σ(Sk)
1
kPk
i=1|si|
Maximum Iterative Standard Deviation ( σmax)
(Pérez-Iglesias and Araujo, 2010):
σmax@k= max
i∈[2,k]σ(Si)
Linearized Score Magnitude and Variance
(LSMV):A practical adaptation of SMV (Tao
and Wu, 2014), removing the logarithmic scaling
used for sparse lexical retrieval, we evaluate the
linearized dense-retrieval variant:
LSMV@k=s 1×σ(S k)
4 Experimental Setup
4.1 Datasets and Cognitive Tiers
To evaluate the robustness of QPP signals across
varying retrieval difficulty, we benchmark metrics
across 28 datasets grouped into three cognitive tiers
capturing different forms of reasoning constraints:•Semantic Matching (BEIR):We utilize four
representative datasets from the BEIR bench-
mark: SciFact, FiQA, NFCorpus, SciDocs.
These datasets evaluate standard semantic re-
trieval tasks.
•Logical Reasoning (BRIGHT):We evaluate
on the BRIGHT benchmark, which introduces
queries requiring logical constraints, multi-
hop reasoning, and structured reasoning con-
ditions across twelve datasets. In these tasks,
simple semantic similarity is often insufficient
to identify correct documents.
•Temporal Reasoning (TEMPO):We include
the TEMPO benchmark consisting of twelve
datasets to evaluate retrieval under explicit
and implicit temporal reasoning constraints.
In this setting, documents may share high se-
mantic similarity with a query while still vio-
lating required time conditions.
4.2 Retrieval Models
To ensure our findings generalize across retrieval
paradigms, we evaluate QPP metrics across a di-
verse set of retrieval architectures.
(1) Sparse Lexical Models:We include
BM25 (Robertson and Zaragoza, 2009) as a clas-
sical lexical retrieval baseline.(2) Dense Bi-
Encoders:We evaluate several widely used dense
retrieval models, including BGE (Xiao et al.,
2024), E5 (Wang et al., 2022), Contriever (Izac-
ard et al., 2021), SBERT (Reimers and Gurevych,
2019), SFR (Meng et al., 2024), Qwen (Li et al.,
2023), and Instructor-Large (Su et al., 2023).(3)
Reasoning-Augmented Encoders:We addition-
ally include models explicitly trained or fine-tuned
to handle structured or reasoning-intensive re-
trieval tasks, including Diver (Sun et al., 2025a),
RaDeR (Das et al., 2025), and ReasonIR (Shao
et al., 2025).
This diverse model suite enables evaluation
across both classical and modern neural retrieval
architectures.
4.3 Evaluation Protocol
For each query, we extract the raw retrieval score
distribution Skfrom each model over the top- k
retrieved documents.
We evaluate QPP metrics across multiple context
window sizes commonly used in RAG pipelines:
k∈ {5,10,25,50}.

Binary Retrieval SuccessWe define retrieval
success as the presence of at least one relevant
document within the retrieved set:
NDCG@k >0.
This criterion reflects the practical requirement that
downstream generation typically depends on re-
trieving at least one relevant supporting passage
within the context window (Lewis et al., 2020;
Karpukhin et al., 2020). Because this definition
becomes less strict as kincreases, results at smaller
cutoffs (e.g. k= 5 ) serve as a stricter success cri-
terion: retrieving a relevant document within only
5 candidates is substantially harder than within 25
or 50. Full results across all cutoffs are reported in
Appendix Tables D.3–D.5; the relative ordering of
QPP methods is consistent throughout.
Evaluation MetricsWe evaluate the predictive
quality of QPP signals using three complementary
statistical measures:(1) AUROC (Area Under
the Receiver Operating Characteristic Curve).
This serves as our primary evaluation metric and
simulates the RAG abstention decision. AUROC
measures the ability of a QPP signal to distinguish
between successful and failed retrieval instances
across all possible decision thresholds, rather than
relying on a fixed confidence threshold.(2) Pear-
son Correlation Coefficient ( r).Measures the
linear correlation between the predicted QPP con-
fidence value and the observed retrieval effective-
ness ( NDCG@k ).(3) Spearman Rank Corre-
lation ( ρ).Measures the monotonic relationship
between predicted confidence and retrieval effec-
tiveness, providing robustness to non-linear scaling
effects. All metrics are computed per dataset and
reported as macro-averages across datasets within
each benchmark.
4.4 Implementation Details
Experiments were conducted on a compute node
equipped with 4 ×NVIDIA H100-80GB GPUs,
512GB system memory, and AMD EPYC proces-
sors using PyTorch and HuggingFace Transform-
ers. All dense models use FP16 inference with offi-
cial Huggingface checkpoints. We release all code,
evaluation scripts, dataset preprocessing pipelines,
and experiment configurations to support full re-
producibility.1
1https://github.com/JamieHoldcroft/
the-magnitude-mirage
Semantic
(BEIR)Temporal
(TEMPO)Logical
(BRIGHT)0.500.550.600.65AUROC@25
random MaxScore (s1)
LSMVFigure 2: LSMV vs. MaxScore AUROC @25 illustrating
systematic degradation of magnitude-based confidence
as reasoning difficulty increases.
5 Results
5.1 The Magnitude Mirage Across Cognitive
Tiers
Table 1 presents AUROC@25 scores for binary re-
trieval success across all three cognitive tiers. The
results reveal a consistent pattern across all mod-
els: raw score magnitude (MaxScore) becomes an
unreliable abstention signal for reasoning-intensive
retrieval tasks, and its failure is systematic rather
than incidental. Figure 2 illustrates this collapse for
ReasonIR: MaxScore AUROC degrades steadily
across cognitive tiers, approaching random perfor-
mance on logical retrieval, while LSMV maintains
stable discrimination.
On standard semantic retrieval (BEIR), MaxS-
core performs adequately for some models—
BGE (.749) and SFR (.754) achieve reasonable
discrimination—but already show instability for
reasoning-augmented encoders such as ReasonIR
(.621) and RaDeR (.666). The degradation be-
comes dramatic on reasoning-intensive tasks. On
TEMPO, MaxScore collapses to near-random per-
formance for Qwen (.536) and ReasonIR (.544).
On BRIGHT, the failure is universal:everymodel
tested produces MaxScore AUROC values in the
range .520–.611, providing essentially no action-
able abstention signal. This confirms the core claim
of the Magnitude Mirage: neural retrievers confi-
dently assign high scores to logically or temporally
incorrect documents, and the raw magnitude of
those scores carries little reliable information about
retrieval success.
5.2 Variance-Based Metrics Restore
Calibration
In contrast, variance-based QPP signals—
particularly Score Gap and LSMV—consistently

recover meaningful abstention performance across
all three tiers. On BEIR, Score Gap and LSMV
improve over MaxScore for nearly every model,
with gains of up to .158 AUROC (Instructor-Large:
.583→.741). On TEMPO, where MaxScore
degrades most severely, Score Gap achieves
gains of up to .125 AUROC (Qwen: .536 →
.661). On BRIGHT, Score Gap and LSMV
improve over MaxScore by margins ranging from
.060 to .091 across dense models. These gains
are consistent across both dense bi-encoders
and reasoning-augmented encoders, suggesting
that variance-based metrics capture a structural
property of confident retrieval that persists across
architectures: when a retriever correctly identifies
a relevant document under complex constraints,
that document statistically separates from the
background distribution, producing a measurable
score gap.
5.3 Comparison Across All Six QPP Metrics
Full results for all six QPP metrics are provided
in Appendix Table D.1. First, all variance-based
alternatives uniformly and substantially outperform
MaxScore across all settings. Across dense models
on BEIR, the improvement from switching to any
variance-based metric ranges from .048 (SFR) to
.158 (Instructor-Large), while the spreadamong
variance-based metrics for the same models is at
most .017. On BRIGHT, where MaxScore AU-
ROC is compressed into the narrow band .525–.571
across dense models, variance-based metrics ex-
pand this to .590–.653—a recovery of meaningful
discrimination from near-random baselines. The
pattern is consistent: the gain from abandoning
magnitude exceeds the differences among distribu-
tional alternatives by a factor of roughly 5–10×,
confirming that the core contribution is the shift
from magnitude to distributional estimation, rather
than the specific choice of distributional statistic.
Second, Score Gap and LSMV emerge as the
most consistent pair across tiers, though no sin-
gle metric dominates universally. LSMV achieves
the best or joint-best AUROC across the major-
ity of BEIR configurations, while Score Gap pro-
vides stronger results on several BRIGHT and
TEMPO settings. NQC offers complementary
strength for sparse retrieval: BM25 on TEMPO
shows NQC (.602) substantially outperforming
Score Gap (.526), reflecting its stability under the
heavier-tailed score distributions of term-frequency
models. The key practical implication is that anyvariance-based signal provides more reliable ab-
stention where magnitude fails—the specific choice
is secondary. Per-subset and per-cutoff breakdowns
are provided in Appendix Section B and Tables D.3–
D.5 where the results remain consistent.
5.4 Correlation with Continuous Retrieval
Effectiveness
Pearson and Spearman correlations with
NDCG@ k, as comprehensively reported in the
Appendix tables, confirm the same ordering. MaxS-
core shows weak or near-zero linear correlation
with retrieval effectiveness on reasoning-intensive
tasks (e.g., Pearson r < .03 for Qwen and BM25
on TEMPO at k= 25 ), while Score Gap and
LSMV achieve correlations up to .40 on the
same benchmarks. Spearman correlations exhibit
the same pattern, confirming the relationship
is monotonic rather than an artifact of linear
scaling assumptions. For example, Score Gap
achieves Pearson r=.467 vs. MaxScore r=.328
for Qwen on BEIR, r=.401 vs.r=.193 on
BRIGHT andr=.303vsr=.028on TEMPO.
5.5 End-to-End RAG Abstention
We evaluate whether retrieval score geometry pre-
dicts downstream RAG correctness across 11 re-
trievers and 19,376 BRIGHT query–retriever pairs
(11 retrievers ×1384 queries). For each query,
the top-5 retrieved documents are provided to
Qwen3.6-27B , while GPT-4o independently judges
answer correctness against BRIGHT gold evidence
documents to avoid self-evaluation bias. We simu-
late abstention by ranking query instances by con-
fidence and answering only the top- p%of highest-
confidence queries. Table 2 reports accuracy at
representative coverage levels; Figure 3 shows the
full continuous trade-off curve. Per-retriever AU-
ROC with 95% bootstrap confidence intervals and
per-retriever operating points are reported in Ap-
pendix C (Tables C.1–C.2).
Signal @10% @25% @50%
MaxScore (s 1) 65.8 65.6 64.5
Gap@25 (s 1−s25)74.1 71.0 68.8
Baseline (no abstention) 63.5
Table 2: Answer accuracy (%) at selective coverage
levels, macro-averaged across 11 retrievers on 1,384
BRIGHT queries. All signals converge to the baseline
at full coverage.
As shown in Table 2, MaxScore provides al-

Model BEIR (Semantic) TEMPO (Temporal) BRIGHT (Logical)
Max Gap LSMV Max Gap LSMV Max Gap LSMV
Sparse Models
BM25 .701.695 .700 .472.526.510 .611 .622.624
Dense Models
BGE .749 .805.810 .626 .701.705 .538.608 .608
E5 .736 .777.787 .648 .709.718 .552 .628.633
Contriever .683 .727.733 .591 .705.707 .560 .624.626
SBERT .718 .766.772 .638 .691.720 .525.605.590
SFR .754 .790.802 .649 .671.690 .553.623.622
Qwen .698 .787.788 .536.661.654 .571.653.650
Instructor-Large .583 .724.741 .563 .664.669 .534.619.608
Reasoning Encoders
Diver-Retriever .665 .788.789 .582 .666.684 .543 .605.606
RaDeR .666 .740.752 .570 .646.665 .560.645.640
ReasonIR .621 .626.650 .544 .638.642 .520.611.606
Table 1: Comparison of zero-cost QPP metrics across three cognitive tiers. We report AUROC@25 scores for
binary retrieval success ( NDCG@25>0 ). Max = s1, Gap = s1−s25, and LSMV represents the Linearized Score
Magnitude and Variance metric. The baseline magnitude column is shaded gray; best result per row–tier isbolded.
most no advantage over the no-abstention baseline:
even at 10% coverage it reaches only 65.8%, barely
above the 63.5% baseline, and for several retrievers
falls below it.
0.0 0.2 0.4 0.6 0.8 1.0
Coverage (fraction of queries answered)0.620.640.660.680.700.720.740.76Answered-query accuracybaseline 0.635MaxScore (s1)
Gap@25 (s1s25)
Figure 3: Coverage–accuracy trade-off macro-averaged
across 14 retrievers on BRIGHT.
Figure 3 shows the full coverage–accuracy trade-
off. MaxScore degrades rapidly toward baseline
as coverage increases, whereas Gap@25 main-
tains a consistent advantage throughout the oper-
ating range. These results confirm that the Magni-
tude Mirage propagates directly into downstream
answer quality: replacing s1thresholding with
Gap@25 yields meaningfully higher precision at se-
lective operating points at zero additional cost. The
per-retriever breakdown in Appendix C confirms
this pattern is broadly consistent: all 11 retrievers
show Gap@25 ≥MaxScore AUROC for answer
correctness, and three (Qwen2,Nomic,Contriever)exhibit MaxScore AUROCbelow 0.5; forQwen2
the 95% bootstrap CI [0.398, 0.459] lies strictly be-
low chance—raw similarity isanti-correlatedwith
answer success—whileNomicandContrieverfall
at the chance boundary (CIs include 0.5).
6 Analysis and Discussion
6.1 Why Magnitude Fails in Reasoning
Retrieval
Modern dense retrievers are trained with con-
trastive objectives that reward semantic proxim-
ity, without explicitly encoding logical, mathe-
matical, or temporal constraints. Consequently,
the top-ranked score s1primarily reflects learned
query–document compatibility and does not explic-
itly guarantee constraint satisfaction. Multiple doc-
uments may achieve similarly high scores while vi-
olating critical task conditions, causing magnitude-
based thresholding to fail to distinguish correct
from incorrect retrieval. This explains the system-
atic MaxScore collapse observed on BRIGHT and
TEMPO.
6.2 Why Variance Signals Capture Retrieval
Confidence
Variance-based metrics operate on score distri-
butionstructurerather than absolute magnitude.
When a retriever correctly identifies a relevant doc-
ument under complex constraints, that document of-
ten separates from the corpus background, produc-
ing a steeply declining score distribution with mea-
surable gaps. As shown in Figure 4, on BRIGHT’s

0.4 0.5 0.6 0.7 0.8 0.9
MaxScore (s1)0.000.020.040.060.080.100.120.14Top-10 std 10
Failed retrieval
Successful retrieval
Similarity threshold
Variance thresholdFigure 4: SBERT on BRIGHT TheoremQA.
TheoremQA subset, a vertical MaxScore threshold
produces near-random class separation, while a hor-
izontal Score Gap threshold partitions successful
from failed retrievals. This separation is not an arti-
fact of corpus sparsity: reasoning-intensive bench-
marks contain comparable or higher relevant docu-
ment density per query than semantic retrieval tasks
(BRIGHT: 4–5, TEMPO: 2–4, vs. BEIR: 1–5), sug-
gesting that the observed score separation reflects
retrieval confidence rather than a statistical property
of the relevance annotations. Two observations ar-
gue against a benchmark-specific confound. First,
MaxScore degradation is consistent across both
BRIGHT and TEMPO, which differ substantially
in reasoning type, annotation methodology, and cor-
pus construction. Second, the pattern holds across
all 11 retrieval architectures and all four cutoffs
tested. Per-subset results in Appendix Section B
further confirm this: Gap@25 outperforms MaxS-
core across the large majority of subsets in both
BRIGHT and TEMPO independently. Failed re-
trieval produces flatter distributions where many se-
mantically similar documents receive comparable
scores. This structural separation is architecture-
agnostic: it emerges consistently across dense bi-
encoders and reasoning-augmented encoders alike,
explaining the uniform gains of Score Gap and
LSMV .
6.3 Implications for RAG System Design
Our results suggest a simple revision to stan-
dard RAG abstention: replace absolute magnitude
thresholding with a variance-based confidence sig-
nal. Across dense retrieval architectures and all
three cognitive tiers, variance-based signals consis-
tently outperform MaxScore, with improvementslarger than the differences among the variance met-
rics themselves. This indicates that the primary
benefit comes from abandoning magnitude-based
confidence estimation rather than from the specific
choice of variance metric. Because variance-based
QPP metrics produce scalar confidence estimates
analogous to existing similarity thresholds, they
can directly replace magnitude-based filtering in de-
ployed RAG pipelines without modifying retrieval
or generation components. Score Gap and LSMV
provide strong default choices given their consis-
tent performance. Because these metrics operate
solely on the top- kscore distribution already pro-
duced during retrieval, they can be integrated into
existing RAG pipelines atzero cost, without addi-
tional model calls, retraining, or latency. In prac-
tice, abstention thresholds can be calibrated on a
held-out validation set to achieve a desired operat-
ing point, such as target answer precision or absten-
tion rate.
7 Conclusion
We characterize theMagnitude Mirage—the sys-
tematic failure of absolute retrieval score thresh-
olds as confidence signals under logical and tem-
poral reasoning constraints—demonstrating that
this behavior consistently affects neural retrievers
across diverse architectures. Our findings indicate
that variance-based QPP metrics, specifically Score
Gap and adaptations like LSMV , restore meaning-
ful, albeit moderate, abstention signals at zero ad-
ditional cost by measuring score distribution struc-
ture rather than raw magnitude.
Our broader finding is that the shift from mag-
nitude to distributional estimation matters more
than which distributional statistic is chosen—the
distribution-based metrics substantially outperform
MaxScore. This suggests that the RAG commu-
nity’s reliance on magnitude thresholding is not a
calibration problem to be tuned, but a structural
mismatch between semantic retrieval objectives
and the requirements for constraint-based absten-
tion. As retrieval tasks continue to grow in reason-
ing complexity, this mismatch will likely widen,
making distribution-aware confidence estimation
an increasingly critical component of reliable RAG
system design.
Limitations
First, while our end-to-end experiment demon-
strates that variance-based abstention improves

downstream answer quality on BRIGHT, this eval-
uation uses a single generator ( Qwen3.6-27B ) and
judge (GPT-4o). Whether these gains generalize
across generator architectures, prompting strate-
gies, or answer evaluation protocols remains an
open question.
Second, our experiments are limited to publicly
available benchmarks. Real-world retrieval sys-
tems may exhibit different score calibration prop-
erties due to domain-specific corpora, proprietary
query distributions, or custom retriever training
procedures not represented here.
Third, we define binary retrieval success as
NDCG@k >0 , reflecting the presence of at least
one relevant document. For queries with a sin-
gle gold document—common in both BRIGHT
and TEMPO—this is the only meaningful binary
criterion, as stricter grade thresholds collapse to
the same condition. Results at k= 5 serve as
a naturally stricter operating point; the consistent
ordering of QPP signals across all cutoffs in the
appendix tables confirms that our conclusions are
robust to this choice.
Finally, while we establish that variance-based
signals substantially outperform magnitude thresh-
olding, we do not characterize the absolute perfor-
mance ceiling for zero-cost abstention. Whether
the remaining gap relative to computationally ex-
pensive approaches such as cross-encoder rerank-
ing or LLM-based verification can be closed with-
out additional inference remains an open question.
References
Abdelrahman Abdallah, Mohammed Ali, Muham-
mad Abdul-Mageed, and Adam Jatowt. 2026.
Tempo: A realistic multi-domain benchmark for
temporal reasoning-intensive retrieval.Preprint,
arXiv:2601.09523.
Negar Arabzadeh, Radin Hamidi Rad, Maryam Khod-
abakhsh, and Ebrahim Bagheri. 2023. Noisy per-
turbations for estimating query difficulty in dense
retrievers. InCIKM ’23: Proceedings of the 32nd
ACM International Conference on Information and
Knowledge Management, CIKM ’23, pages 3722 –
3727, Birmingham, UK. Association for Computing
Machinery.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and
Hannaneh Hajishirzi. 2024. Self-rag: Learning to
retrieve, generate, and critique through self-reflection.
InThe Twelfth International Conference on Learning
Representations, ICLR’24.
David Carmel and Elad Yom-Tov. 2010. Estimating the
query difficulty for information retrieval. InProceed-ings of the 33rd International ACM SIGIR Confer-
ence on Research and Development in Information
Retrieval, SIGIR ’10, page 911, New York, NY , USA.
Association for Computing Machinery.
Harrison Chase. 2022. Langchain. Software, released
2022-10-17.
Adrian-Gabriel Chifu, Sébastien Déjean, Moncef
Garouani, Josiane Mothe, Diégo Ortiz, and
Md Zia Ullah. 2025. Uncovering the limitations of
query performance prediction: Failures, insights, and
implications for selective query processing.ACM
Transactions on Information Systems.
Florin Cuconasu, Giovanni Trappolini, Federico Sicil-
iano, Simone Filice, Cesare Campagnano, Yoelle
Maarek, Nicola Tonellotto, and Fabrizio Silvestri.
2024. The power of noise: Redefining retrieval for
rag systems. InProceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and De-
velopment in Information Retrieval, SIGIR ’24, page
719–729, New York, NY , USA. Association for Com-
puting Machinery.
Debrup Das, Sam O’Nuallain, and Razieh Rahimi. 2025.
Rader: Reasoning-aware dense retrieval models. In
Proceedings of the 2025 Conference on Empirical
Methods in Natural Language Processing, pages
19970–19997, Suzhou, China. Association for Com-
putational Linguistics.
Suchana Datta, Guglielmo Faggioli, Nicola Ferro, De-
basis Ganguly, Cristina Ioana Muntean, Raffaele
Perego, and Nicola Tonellotto. 2025a. Projection-
displacement-based query performance prediction
for embedded space of dense retrievers.ACM Trans.
Inf. Syst., 44(1).
Suchana Datta, Guglielmo Faggioli, Nicola Ferro, De-
basis Ganguly, Cristina Ioana Muntean, Raffaele
Perego, and Nicola Tonellotto. 2025b. Projection-
displacement-based query performance prediction
for embedded space of dense retrievers.ACM Trans.
Inf. Syst., 44(1).
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan,
Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen,
Shengjie Ma, Honghao Liu, and 1 others. 2025. A
survey on llm-as-a-judge.The Innovation.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Ming-Wei Chang. 2020. Realm: Retrieval-
augmented language model pre-training. InProceed-
ings of the 37th International Conference on Machine
Learning, ICML’20. JMLR.org.
Ben He and Iadh Ounis. 2004. Inferring query per-
formance using pre-retrieval predictors. InString
Processing and Information Retrieval, pages 43–54,
Berlin, Heidelberg. Springer Berlin Heidelberg.
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Se-
bastian Riedel, Piotr Bojanowski, Armand Joulin,
and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv
preprint arXiv:2112.09118.

Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas
Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-
Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval
augmented language models.Journal of Machine
Learning Research, 24(251):1–43.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 6769–6781,
Online. Association for Computational Linguistics.
Zhan Peng Lee, Andre Lin, and Calvin Tan. 2025.
Finetune-rag: Fine-tuning language models to re-
sist hallucination in retrieval-augmented generation.
Preprint, arXiv:2505.10792.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Ad-
vances in Neural Information Processing Systems,
33:9459–9474.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long,
Pengjun Xie, and Meishan Zhang. 2023. Towards
general text embeddings with multi-stage contrastive
learning.Preprint, arXiv:2308.03281.
Jerry Liu. 2022. Llamaindex: Data framework for rag
applications. Retrieved from GitHub.
Shuliang Liu, Xinze Li, Zhenghao Liu, Yukun Yan,
Cheng Yang, Zheni Zeng, Zhiyuan Liu, Maosong
Sun, and Ge Yu. 2025. Judge as a judge: Improv-
ing the evaluation of retrieval-augmented generation
through the judge-consistency of large language mod-
els. InFindings of the Association for Computational
Linguistics: ACL 2025, pages 5788–5807.
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming
Xiong, Yingbo Zhou, and Semih Yavuz. 2024. Sfr
embedding-mistral: Enhance text retrieval with trans-
fer learning.Salesforce AI Research Blog, 3(6).
Joaquín Pérez-Iglesias and Lourdes Araujo. 2010. Stan-
dard deviation as a query hardness estimator. In
String Processing and Information Retrieval, pages
207–212, Berlin, Heidelberg. Springer Berlin Heidel-
berg.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empirical
Methods in Natural Language Processing and the 9th
International Joint Conference on Natural Language
Processing (EMNLP-IJCNLP), pages 3982–3992.
Stephen Robertson and Hugo Zaragoza. 2009.The
Probabilistic Relevance Framework: BM25 and Be-
yond, volume 3. Now Publishers Inc.Nicholas Rossi, Juexin Lin, Feng Liu, Zhen Yang, Tony
Lee, Alessandro Magnani, and Ciya Liao. 2024. Rel-
evance filtering for embedding-based retrieval. In
Proceedings of the 33rd ACM International Confer-
ence on Information and Knowledge Management,
CIKM ’24, page 4828–4835, New York, NY , USA.
Association for Computing Machinery.
Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muen-
nighoff, Xi Victoria Lin, Daniela Rus, Bryan
Kian Hsiang Low, Sewon Min, Wen-tau Yih,
Pang Wei Koh, and 1 others. 2025. Reasonir: Train-
ing retrievers for reasoning tasks.arXiv preprint
arXiv:2504.20595.
Anna Shtok, Oren Kurland, David Carmel, Fiana Raiber,
and Gad Markovits. 2012. Predicting query perfor-
mance by query-drift estimation.ACM Trans. Inf.
Syst., 30(2).
Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang,
Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A
Smith, Luke Zettlemoyer, and Tao Yu. 2023. One
embedder, any task: Instruction-finetuned text em-
beddings. InFindings of the Association for Compu-
tational Linguistics: ACL 2023, pages 1102–1121.
Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi,
Niklas Muennighoff, Han-yu Wang, Liu Haisu, Quan
Shi, Zachary Siegel, Michael Tang, Ruoxi Sun, Jin-
sung Yoon, Sercan Arik, Danqi Chen, and Tao Yu.
2025. Bright: A realistic and challenging benchmark
for reasoning-intensive retrieval. InInternational
Conference on Learning Representations, volume
2025, pages 48941–48991.
Duolin Sun, Meixiu Long, Dan Yang, Junjie Wang,
Yecheng Luo, Yue Shen, Jian Wang, Hualei Zhou,
Chunxiao Guo, Peng Wei, and 1 others. 2025a. Diver:
A multi-stage approach for reasoning-intensive infor-
mation retrieval.arXiv preprint arXiv:2508.07995.
Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Yang Song,
Jun Xu, Xiao Zhang, Weijie Yu, Yang Song, and
Han Li. 2025b. Redeep: Detecting hallucination
in retrieval-augmented generation via mechanistic
interpretability.Preprint, arXiv:2410.11414.
Yongquan Tao and Shengli Wu. 2014. Query perfor-
mance prediction by considering score magnitude
and variance together. InProceedings of the 23rd
ACM International Conference on Conference on In-
formation and Knowledge Management, CIKM ’14,
page 1891–1894, New York, NY , USA. Association
for Computing Machinery.
Nandan Thakur, Luiz Bonifacio, Xinyu Zhang,
Odunayo Ogundepo, Ehsan Kamalloo, David
Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Boxing
Chen, Mehdi Rezagholizadeh, and 1 others. 2023.
Nomiracl: Knowing when you don’t know for robust
multilingual retrieval-augmented generation.arXiv
preprint arXiv:2312.11361.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. 2021. Beir:

A heterogenous benchmark for zero-shot evaluation
of information retrieval models. InAdvances in Neu-
ral Information Processing Systems 34.
Liang Wang, Nan Yang, Xiaolong Huang, Binxing
Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder,
and Furu Wei. 2022. Text embeddings by weakly-
supervised contrastive pre-training.arXiv preprint
arXiv:2212.03533.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, pages 641–649.
Jiawei Zhou and Lei Chen. 2026. Optimizing re-
trieval for rag via reinforcement learning.Preprint,
arXiv:2510.24652.

Appendices
This appendix provides supplementary material for
The Magnitude Mirage. It is organized as follows:
•Appendix A provides comprehensive QPP
evaluation results that complement the head-
line numbers reported in the main paper, in-
cluding all six metrics at k= 25 , Pear-
son correlations across all benchmarks and
cutoffs, and full per-benchmark tables at
k∈ {5,10,25,50} for BEIR, BRIGHT, and
TEMPO.
•Appendix B reports per-subset AUROC@25
for MaxScore and Score Gap across every
individual BEIR (4), BRIGHT (12), and
TEMPO (12) subset, addressing reviewer re-
quests for breakdowns of where the variance-
based gains concentrate.
•Appendix C provides the per-retriever break-
down of the end-to-end RAG abstention exper-
iment (Section 5.5) with 95% bootstrap confi-
dence intervals across 11 retrievers, and per-
retriever selective-prediction operating points
at coverage levels 10/25/50%.
•Appendix D reproduces verbatim the two
prompt templates used in the end-to-end RAG
experiment (generator and judge), with full
decoding parameters and parsing rules to sup-
port exact reproduction.
Key Tables
•Table D.1: AUROC@25 for all six zero-cost
QPP metrics (MaxScore, Score Gap, Top- k
Std, NQC, Maximum Iterative Std, LSMV)
across BEIR, BRIGHT, and TEMPO.
•Table D.2: Pearson correlations between
MaxScore / Score Gap and NDCG@k for
every model, benchmark, and cutoff k∈
{5,10,25,50}.
•Tables D.3, D.4, D.5: full six-metric evalua-
tion jointly reporting AUROC, Pearson r, and
Spearman ρfor BEIR, BRIGHT, and TEMPO
at all four cutoffs.
•Tables D.6, D.7, D.8: per-subset AUROC@25
for MaxScore and Gap@25 with the per-
subset ∆, macro-averaged across 11 retriev-
ers.•Table C.1: per-retriever AUROC for down-
stream answer correctness in the end-to-end
RAG experiment, with 95% bootstrap CIs
(N=1000).
•Table C.2: per-retriever selective-prediction
answered-accuracy at abstention coverages
10/25/50%, highlighting the 38-point Qwen2
coverage swing.
Key Figures
•Figure D.1: generator prompt template used
byQwen3.6-27B (top-5 retrieved BRIGHT
documents as context).
•Figure D.2: judge prompt template used by
GPT-4o (gold BRIGHT evidence as refer-
ence), deliberately chosen from a different
model family to avoid self-evaluation bias.
A Comprehensive Per-Benchmark
Results
This appendix collects the comprehensive evalua-
tion tables behind the headline numbers in the main
paper. Subsection A.1 reports all six QPP metrics
at the headline cutoff k= 25 ; Subsection A.2 re-
ports Pearson correlations across all four cutoffs;
Subsection A.3 provides the full per-benchmark
tables atk∈ {5,10,25,50}.
A.1 All Six QPP Metrics atk= 25
Table D.1 expands the three-metric main-paper ta-
ble (MaxScore, Gap, LSMV) to all six zero-cost
QPP signals (additionally: Top- kStd, NQC, Max-
imum Iterative Std) at k= 25 , broken down by
cognitive tier. The key observation is that every
variance-based metric beats MaxScore on every
dense retriever on BRIGHT and TEMPO. The
spreadamong variance-based metrics is small (typ-
ically≤0.02 AUROC) relative to the gap between
any of them and MaxScore (often 0.05–0.16), con-
firming that the operative axis of improvement is
magnitude →distribution rather than the choice of
specific distributional statistic.
A.2 Pearson Correlations Across Benchmarks
and Cutoffs
Table D.2 provides a compact cross-benchmark
view of the continuous prediction quality of
MaxScore versus Score Gap, reporting Pearson
rbetween each signal and NDCG@k for every
model, benchmark, and cutoff k∈ {5,10,25,50} .

MaxScore correlations are near-zero or weak on
reasoning-intensive benchmarks (e.g., r≈0 for
Qwen and BM25 on TEMPO at most cutoffs),
while Score Gap reaches r=.40 + on the same
benchmarks. The pattern is monotone across cut-
offs, confirming that the distinction is not an artifact
of any specific choice ofk.
A.3 Full Per-Benchmark Tables at All Cutoffs
Tables D.3, D.4, and D.5 provide the full six-metric
evaluation across all k∈ {5,10,25,50} for BEIR,
BRIGHT, and TEMPO respectively, jointly report-
ing AUROC, Pearson ( r), and Spearman ( ρ). The
relative ordering of QPP metrics is stable across
all cutoffs, confirming that conclusions drawn at
k= 25generalize across context window sizes.
B Per-Subset Results
Tables D.6, D.7, and D.8 report AUROC@25 for
MaxScore and Score Gap at the subset level, macro-
averaged across all 11 retrievers. Gap@25 outper-
forms MaxScore on 3 of 4 BEIR subsets, 10 of
12 BRIGHT subsets, and 10 of 12 TEMPO sub-
sets. Exceptions are limited to a small number of
outlier subsets where corpus structure or query dis-
tribution gives magnitude residual discriminative
value; in all cases the gap is small ( |∆| ≤0.055 ).
These results confirm that the gains reported in the
main paper are broadly consistent across individual
datasets rather than driven by a small number of
outlier subsets.
C Per-Retriever End-to-End RAG
Abstention with Bootstrap CIs
This appendix provides the per-retriever breakdown
of the end-to-end RAG abstention experiment sum-
marized in Section 5.5 (Table 2, Figure 3). Confi-
dence intervals are computed by bootstrap resam-
pling of the per-query (signal, judgment) pairs with
N=1000resamples.
Table C.1 reports per-retriever AUROC for pre-
dicting downstream answer correctness, using each
of the three confidence signals (MaxScore s1,
Score Gap at the RAG context window s1−s5,
and the paper-headline Score Gap s1−s25).All
11 retrievers exhibit Gap@25 AUROC ≥MaxS-
core AUROC, with non-overlapping 95% CIs for
the majority of dense retrievers. For three retrievers
(Contriever,Nomic, and most strikinglyQwen2)
the MaxScore AUROC liesbelow 0.5; forQwen2
the CI [0.398,0.459] lies strictly below chance—raw similarity is statisticallyanti-correlatedwith
downstream answer success—whileNomicand
Contrieverfall at the chance boundary (CIs in-
clude 0.5). These cases provide the cleanest single-
retriever evidence of the Magnitude Mirage in the
entire study, and the Gap@25 signal recovers AU-
ROC of 0.55–0.60 with statistical significance.
Table C.2 reports selective-prediction answered-
accuracy at three abstention coverage levels (10%,
25%, 50%) per retriever. The most striking case
isQwen2at coverage 10%: MaxScore-based ab-
stention retains queries with only 39.1% answered-
accuracy—26 pointsbelowthe no-abstention base-
line of 65.8%—while Gap@25 retains queries with
76.8% accuracy, an 11-pointimprovementover
baseline. The same set of queries, the same re-
triever, but a 38-point swing in downstream answer
quality depending on which confidence signal is
used for abstention. The macro-averaged ∆of
+8.3 points at 10% coverage (Table C.2, last row)
matches the macro figure in Section 5.5.
D Prompts Used in the End-to-End RAG
Experiment
For full transparency and reproducibility, Fig-
ures D.1–D.2 reproduce verbatim the two prompt
templates used in the end-to-end RAG experi-
ment (Section 5.5 and Appendix C). The generator
prompt is sent to Qwen3.6-27B (served locally via
vLLM) together with the top-5 retrieved BRIGHT
documents as context. The judge prompt is sent
to GPT-4o (Azure OpenAI deployment gpt-4o-2 )
with the BRIGHT gold documents as reference
evidence. Both prompts use temperature 0.0; the
generator caps generation at 300 tokens and the
judge at 10 tokens (since the expected output is
a single CORRECT/INCORRECT token). The
generator runs with the Qwen reasoning “thinking”
mode disabled ( enable_thinking=False ) so that
the recorded latency reflects the practical RAG-
pipeline setting and the judge sees a final answer
rather than chain-of-thought.

RetrieverNMaxScore Gap@5 (ctx) Gap@25
BM25 1384 0.573[0.542, 0.603]0.565[0.535, 0.598]0.586[0.554, 0.618]
SBERT 1384 0.517[0.486, 0.548]0.556[0.525, 0.589]0.562[0.530, 0.593]
Contriever 1384 0.488[0.457, 0.519]0.551[0.522, 0.583]0.558[0.531, 0.591]
E5 1384 0.510[0.479, 0.542]0.524[0.493, 0.557]0.552[0.521, 0.584]
BGE 1384 0.514[0.483, 0.545]0.571[0.542, 0.602]0.577[0.546, 0.606]
SFR 1384 0.511[0.480, 0.543]0.552[0.519, 0.584]0.576[0.543, 0.607]
Instructor-L 1384 0.513[0.482, 0.543]0.555[0.525, 0.585]0.574[0.547, 0.604]
Instructor-XL 1384 0.525[0.490, 0.557]0.568[0.535, 0.598]0.576[0.546, 0.607]
Qwen 1384 0.528[0.497, 0.558]0.539[0.504, 0.569]0.551[0.520, 0.580]
Qwen2 1384 0.429[0.398, 0.459]0.558[0.526, 0.588]0.585[0.551, 0.615]
Nomic 1384 0.481[0.450, 0.512]0.594[0.565, 0.626]0.603[0.572, 0.634]
RaDeR 1384 0.537[0.507, 0.570]0.524[0.490, 0.557]0.538[0.503, 0.568]
ReasonIR 1384 0.530[0.498, 0.562]0.550[0.519, 0.583]0.589[0.559, 0.618]
Diver 1384 0.527[0.495, 0.557]0.563[0.533, 0.597]0.594[0.562, 0.628]
Macro mean— 0.513 0.5550.573
Table C.1: Per-retriever AUROC (with 95% bootstrap CI, N=1000 resamples) for predicting downstream answer
correctness on BRIGHT in the end-to-end RAG experiment (Section 5.5). Generator: Qwen3.6-27B via vLLM;
judge: GPT-4o (Azure). Bold = best signal per row.All 14 retrievers show Gap@25 ≥MaxScore.ForQwen2,
Nomic, andContriever, MaxScore AUROC isbelow 0.5, indicating that raw similarity is anti-correlated with answer
success—the clearest signature of the Magnitude Mirage in the downstream pipeline.
RetrieverNBaselineCoverage 10% Coverage 25% Coverage 50%
Max Gap@25∆Max Gap@25∆Max Gap@25∆
BM25 1384 61.3 75.474.6-0.7 71.772.0+0.3 66.268.9+2.7
SBERT 1384 62.6 68.173.2+5.1 65.368.2+2.9 64.367.9+3.6
Contriever 1384 59.9 54.370.3+15.9 57.865.3+7.5 60.364.3+4.0
E5 1384 63.6 65.268.1+2.9 65.368.5+3.2 64.567.1+2.6
BGE 1384 62.5 68.874.6+5.8 66.270.8+4.6 62.068.1+6.1
SFR 1384 63.9 69.672.5+2.9 65.972.5+6.6 64.369.2+4.9
Instructor-L 1384 62.9 64.576.1+11.6 67.373.4+6.1 63.668.2+4.6
Instructor-XL 1384 64.2 66.776.1+9.4 68.273.1+4.9 66.069.7+3.6
Qwen 1384 64.9 71.777.5+5.8 65.671.7+6.1 66.368.9+2.6
Qwen2 1384 65.8 39.176.8+37.7 55.876.0+20.2 62.671.4+8.8
Nomic 1384 62.6 66.778.3+11.6 63.672.0+8.4 61.669.7+8.1
RaDeR 1384 64.7 72.566.7-5.8 68.566.8-1.7 66.967.6+0.7
ReasonIR 1384 65.0 69.673.9+4.3 69.471.1+1.7 67.270.8+3.6
Diver 1384 65.2 69.679.0+9.4 67.673.1+5.5 67.371.1+3.8
Macro mean19376 63.5 65.874.1+8.3 65.671.0+5.5 64.568.8+4.3
Table C.2: Per-retriever selective-prediction accuracy (%) at three abstention coverage levels. Baseline = answered-
accuracy with no abstention. ∆= Gap@25 −MaxScore (positive favors variance-based abstention).Most striking
case:Qwen2at coverage 10%: MaxScore retains 39.1%—26 points below baseline—while Gap@25 retains 76.8%
(11 points above baseline), a 38-point swing on the same retriever and same queries.

Generator prompt (Qwen3.6-27B, top-5 BRIGHT context)
System Prompt:You are an expert QA assistant. You must answer the question based on the provided context. Provide
the best possible answer. Do not apologize or refuse.
User Prompt Template:
Context:
Document 1:
{retrieved_doc_1}
Document 2:
{retrieved_doc_2}
...
Document 5:
{retrieved_doc_5}
Question: {query_text}
Answer:
Decoding parameters.
• Model:Qwen/Qwen3.6-27Bserved via vLLM (tensor-parallel size 4 on4×H100-80GB)
• Temperature:0.0
•max_tokens:300
•enable_thinking:False(reasoning-mode disabled to reflect the deployed-RAG operating point)
•Each retrieved document is truncated to at most 4,000 characters to fit within the context window across the
longest BRIGHT subsets.
Figure D.1: Generator prompt used in the end-to-end RAG abstention experiment (Section 5.5). The model is
shown the top-5 documents retrieved by the active retriever and is required to answer the BRIGHT query without
refusing or abstaining; abstention is delegated to the QPP-signal threshold appliedaftergeneration.
ModelBEIR (Semantic) TEMPO (Temporal) BRIGHT (Logical)
Max Gap Std NQC MIS LSMV Max Gap Std NQC MIS LSMV Max Gap Std NQC MIS LSMV
Sparse
BM25 .701.695 .694 .663 .688 .700 .472 .526 .529.602.530 .510 .611 .622.626.611 .623 .624
Dense
BGE .749 .805 .805 .795 .805.810 .626 .701 .698 .696 .703.705 .538 .608 .611.615.608 .608
E5 .736 .777 .780 .770 .778.787 .648 .709.719 .719.708 .718 .552 .628 .636.638.630 .633
Contriever .683 .727 .728 .701 .730.733 .591 .705.715.708 .701 .707 .560 .624 .625 .618 .619.626
SBERT .718 .766 .770 .713 .769.772 .638 .691 .716 .696 .694.720 .525.605.600 .608 .597 .590
SFR .754 .790 .795 .782 .791.802 .649 .671 .688 .684 .675.690 .553 .623 .625.626.624 .622
Qwen .698 .787 .785 .771 .783.788 .536.661 .661.660 .656 .654 .571.653.652 .650 .646 .650
Instructor-L .583 .724 .740 .738 .718.741 .563 .664.671.670 .660 .669 .534.619.609 .610 .611 .608
Reasoning
Diver-Retriever .665 .788.790.789 .786 .789 .582 .666 .681 .681 .663.684 .543 .605.606.605 .599.606
RaDeR .666 .740 .748 .743 .744.752 .570 .646 .668.670.658 .665 .560.645.640 .640 .644 .640
ReasonIR .621 .626 .624 .583 .627.650 .544 .638 .652.653.641 .642 .520 .611 .616.626.616 .606
Table D.1: AUROC@25 for all six zero-cost QPP metrics across three cognitive tiers.Max= MaxScore ( s1),Gap=
Score Gap ( s1−s25),Std= Top- kStandard Deviation,NQC= Normalized Query Commitment,MIS= Maximum
Iterative Standard Deviation,LSMV= Linearized Score Magnitude and Variance. Baseline magnitude column is
shaded gray; best result per row–tier isbolded.

Judge prompt (GPT-4o, evaluating answer correctness against BRIGHT gold evidence)
System Prompt:You are an expert evaluator grading whether an AI system correctly answered a question.
A reference evidence section is provided to help determine the expected answer, but the AI prediction does not need to
exactly match the wording or structure of the reference.
Mark CORRECT if the prediction:
• correctly answers the question,
• is semantically consistent with the reference evidence,
• and would reasonably be accepted by a human expert.
Mark INCORRECT if the answer is wrong, unsupported, incomplete for the task, or contradicts the reference evidence.
Output ONLY: CORRECT or INCORRECT
User Prompt Template:
Question:
{query_text}
Reference Evidence:
{gold_doc_1}
{gold_doc_2}
...
AI Prediction:
{prediction}
Is the AI prediction correct?
Decoding parameters.
• Model: GPT-4o (Azure OpenAI deploymentgpt-4o-2, API version2025-01-01-preview)
• Temperature:0.0
•max_tokens:10(output is a single CORRECT/INCORRECT token)
•Up to four BRIGHT gold documents per query are included as reference evidence; each is truncated to at most
4,000 characters to keep the judge prompt within Azure’s per-call token budget.
•Parsing rule: an output whose first non-whitespace token (case-insensitive) starts with CORRECT is recorded as
CORRECT; anything else—including refusals, empty replies, and the literal string INCORRECT —is recorded as
INCORRECT.
Figure D.2: Judge prompt used in the end-to-end RAG abstention experiment. The judge model (GPT-4o) is
deliberately chosen from a different model family than the generator ( Qwen3.6-27B ) to avoid self-evaluation bias,
and is grounded by the BRIGHT gold reference passages so that judgments approximate human-expert relevance
rather than free-form LLM opinion.

Model k BEIR BRIGHT TEMPO
Max Gap Max Gap Max Gap
BM25 5 .330 .307 .149 .177 .029 .095
10 .344 .334 .175 .195 .019 .106
25 .347 .353 .191 .208 -.000 .070
50 .360 .366 .206 .224 .017 .095
BGE 5 .391 .460 .131 .256 .198 .210
10 .391 .505 .131 .270 .198 .272
25 .391 .538 .131 .274 .198 .302
50 .391 .541 .131 .256 .198 .314
E5 5 .350 .443 .142 .233 .192 .227
10 .357 .470 .159 .284 .201 .270
25 .363 .495 .157 .302 .211 .311
50 .364 .495 .153 .285 .218 .323
Contriever 5 .286 .416 .119 .275 .101 .240
10 .296 .430 .127 .298 .105 .281
25 .297 .442 .129 .308 .129 .312
50 .301 .439 .136 .297 .147 .328
SBERT 5 .356 .448 .102 .244 .193 .225
10 .367 .483 .121 .266 .185 .276
25 .366 .495 .114 .273 .216 .326
50 .365 .493 .119 .281 .241 .366
SFR 5 .374 .440 .171 .266 .162 .200
10 .375 .467 .169 .300 .181 .255
25 .377 .492 .173 .312 .194 .300
50 .376 .493 .172 .298 .195 .315
Qwen 5 .329 .410 .190 .325 .013 .217
10 .335 .451 .188 .366 .018 .257
25 .328 .467 .193 .401 .028 .303
50 .327 .470 .197 .404 .023 .328
Instructor-Large 5 .203 .346 .144 .274 .046 .181
10 .201 .399 .139 .298 .069 .219
25 .193 .437 .139 .314 .086 .261
50 .191 .453 .138 .292 .090 .260
Diver-Retriever 5 .261 .445 .139 .295 .098 .209
10 .262 .468 .150 .342 .084 .239
25 .257 .482 .159 .347 .106 .274
50 .255 .476 .164 .339 .118 .285
RaDeR 5 .227 .402 .130 .312 .052 .136
10 .242 .446 .131 .355 .075 .223
25 .247 .462 .136 .358 .087 .261
50 .246 .456 .139 .353 .093 .280
ReasonIR 5 .170 .287 .117 .268 .003 .142
10 .177 .294 .126 .298 .001 .174
25 .186 .297 .120 .309 .019 .198
50 .187 .283 .123 .303 .025 .211
Table D.2: Pearson correlation between QPP signals and retrieval effectiveness across benchmarks and cutoffs. Max
denotes MaxScore (s 1) and Gap denotes Score Gap (s 1−sk).

ModelkMaxScore ScoreGap Top-kStd NQC MaxIterStd LSMV
AUCr ρAUCr ρAUCr ρAUCr ρAUCr ρAUCr ρ
Sparse Models
BM255 .696 .330 .340 .694 .307 .341 .694 .305 .341 .671 .268 .307 .694 .189 .343 .698 .240 .345
10 .712 .344 .351 .709 .334 .361 .705 .323 .350 .676 .282 .322 .703 .194 .355 .711 .245 .354
25 .701 .347 .348 .695 .353 .357 .694 .329 .338 .663 .284 .307 .688 .209 .349 .700 .252 .349
50 .700 .360 .355 .691 .366 .358 .687 .327 .331 .646 .272 .287 .680 .220 .346 .696 .263 .349
Dense Models
BGE5 .749 .391 .397 .755 .460 .487 .754 .456 .487 .742 .444 .470 .753 .349 .487 .763 .455 .498
10 .749 .391 .397 .786 .505 .534 .787 .500 .526 .776 .488 .514 .783 .388 .529 .793 .497 .533
25 .749 .391 .397 .805 .538 .558 .805 .514 .530 .795 .510 .524 .805 .423 .557 .810 .510 .534
50 .749 .391 .397 .808 .541 .561 .793 .482 .500 .783 .483 .496 .805 .425 .556 .800 .483 .507
E55 .701 .350 .354 .754 .443 .472 .755 .440 .471 .748 .429 .461 .753 .355 .470 .759 .440 .479
10 .722 .357 .361 .759 .470 .487 .766 .465 .481 .758 .455 .472 .763 .386 .488 .770 .466 .487
25 .736 .363 .367 .777 .495 .501 .780 .456 .459 .770 .452 .456 .778 .412 .492 .787 .459 .466
50 .739 .364 .367 .787 .495 .498 .776 .408 .417 .766 .410 .418 .786 .413 .482 .784 .416 .427
Contriever5 .663 .286 .279 .705 .416 .414 .704 .414 .413 .686 .394 .389 .703 .353 .413 .713 .412 .422
10 .672 .296 .293 .714 .430 .435 .717 .421 .431 .693 .398 .404 .716 .365 .434 .724 .418 .435
25 .683 .297 .303 .727 .442 .453 .728 .408 .431 .701 .392 .409 .730 .376 .453 .733 .402 .429
50 .694 .301 .308 .728 .439 .452 .717 .365 .399 .688 .360 .383 .726 .375 .447 .726 .364 .401
SBERT5 .701 .356 .355 .724 .448 .456 .727 .447 .458 .686 .379 .394 .725 .374 .457 .744 .449 .481
10 .713 .367 .366 .741 .483 .496 .740 .474 .488 .690 .397 .423 .738 .404 .492 .753 .469 .497
25 .718 .366 .371 .766 .495 .513 .770 .454 .478 .713 .399 .421 .769 .418 .506 .772 .443 .477
50 .726 .365 .373 .782 .493 .513 .779 .402 .441 .699 .363 .378 .789 .406 .503 .781 .400 .447
SFR5 .722 .374 .375 .761 .440 .478 .761 .432 .477 .751 .418 .462 .759 .329 .476 .767 .432 .487
10 .738 .375 .375 .773 .467 .493 .775 .458 .479 .763 .444 .467 .771 .363 .486 .782 .458 .489
25 .754 .377 .379 .790 .492 .506 .795 .451 .458 .782 .445 .455 .791 .392 .494 .802 .453 .467
50 .744 .376 .378 .796 .493 .503 .786 .409 .416 .774 .411 .418 .796 .393 .487 .793 .417 .428
Qwen5 .691 .329 .312 .730 .410 .423 .727 .403 .420 .712 .379 .399 .723 .327 .416 .738 .408 .434
10 .711 .335 .314 .764 .451 .462 .764 .444 .453 .747 .421 .433 .759 .371 .456 .771 .447 .462
25 .698 .328 .316 .787 .467 .472 .785 .434 .435 .771 .420 .422 .783 .392 .468 .788 .435 .442
50 .716 .327 .315 .802 .470 .472 .787 .397 .400 .769 .392 .389 .803 .395 .466 .794 .403 .412
Instructor-Large5 .593 .203 .191 .669 .346 .309 .668 .343 .306 .667 .340 .303 .662 .298 .297 .669 .345 .308
10 .592 .201 .189 .695 .399 .364 .704 .413 .378 .703 .411 .376 .690 .356 .358 .705 .416 .380
25 .583 .193 .187 .724 .437 .418 .740 .453 .435 .738 .450 .431 .718 .399 .410 .741 .455 .438
50 .582 .191 .186 .724 .453 .435 .737 .451 .438 .734 .448 .432 .723 .415 .433 .739 .454 .444
Reasoning Encoders
Diver-Retriever5 .667 .261 .267 .756 .445 .470 .755 .439 .464 .752 .434 .463 .753 .341 .464 .756 .435 .465
10 .666 .262 .265 .775 .468 .489 .772 .455 .474 .771 .453 .475 .769 .371 .479 .773 .450 .472
25 .665 .257 .267 .788 .482 .502 .790 .442 .458 .789 .445 .462 .786 .397 .490 .789 .437 .458
50 .675 .255 .265 .805 .476 .494 .810 .396 .420 .808 .404 .426 .815 .395 .486 .810 .393 .422
RaDeR5 .634 .227 .227 .707 .402 .405 .707 .399 .404 .704 .396 .401 .705 .319 .402 .708 .396 .405
10 .641 .242 .241 .724 .446 .444 .732 .450 .448 .729 .449 .446 .727 .372 .448 .733 .444 .447
25 .666 .247 .252 .740 .462 .461 .748 .437 .439 .743 .442 .441 .744 .390 .462 .752 .432 .438
50 .683 .246 .253 .758 .456 .461 .761 .387 .403 .753 .397 .409 .760 .382 .458 .765 .384 .405
ReasonIR5 .601 .170 .177 .628 .287 .248 .625 .280 .241 .607 .244 .212 .625 .258 .244 .634 .288 .255
10 .605 .177 .193 .619 .294 .253 .615 .284 .240 .583 .232 .190 .614 .263 .245 .630 .295 .261
25 .621 .186 .207 .626 .297 .262 .624 .268 .239 .583 .202 .184 .627 .260 .258 .650 .288 .271
50 .619 .187 .205 .621 .283 .245 .614 .230 .202 .569 .161 .143 .621 .247 .239 .641 .260 .246
Table D.3: Comprehensive zero-cost QPP evaluation across all target cutoffs ( k∈ {5,10,25,50} ) for theBEIR
benchmark. Predictive power is evaluated using AUROC (AUC), Pearson (r), and Spearman (ρ).

ModelkMaxScore ScoreGap Top-kStd NQC MaxIterStd LSMV
AUCr ρAUCr ρAUCr ρAUCr ρAUCr ρAUCr ρ
Sparse Models
BM255 .597 .149 .153 .613 .177 .180 .615 .178 .182 .601 .177 .163 .612 .132 .178 .617 .153 .184
10 .602 .175 .177 .612 .195 .209 .618 .199 .217 .598 .206 .191 .612 .138 .207 .615 .164 .209
25 .611 .191 .206 .622 .208 .238 .626 .197 .244 .611 .214 .229 .623 .127 .241 .624 .161 .237
50 .604 .206 .220 .615 .224 .250 .612 .206 .247 .600 .218 .235 .611 .136 .255 .611 .177 .241
Dense Models
BGE5 .538 .131 .088 .583 .256 .167 .584 .260 .167 .584 .261 .168 .586 .240 .170 .583 .258 .166
10 .538 .131 .088 .602 .270 .195 .603 .266 .196 .606 .270 .201 .603 .248 .197 .600 .261 .192
25 .538 .131 .088 .608 .274 .202 .611 .247 .201 .615 .256 .208 .608 .244 .199 .608 .243 .197
50 .538 .131 .088 .604 .256 .193 .596 .204 .173 .601 .216 .182 .602 .227 .187 .593 .203 .170
E55 .537 .142 .090 .608 .233 .202 .606 .237 .200 .607 .234 .200 .604 .195 .196 .605 .236 .199
10 .546 .159 .110 .621 .284 .240 .617 .291 .235 .617 .290 .236 .613 .243 .227 .617 .288 .234
25 .552 .157 .122 .628 .302 .259 .636 .305 .275 .638 .310 .279 .630 .260 .263 .633 .301 .269
50 .544 .153 .115 .625 .285 .253 .616 .256 .245 .621 .263 .253 .621 .237 .251 .614 .254 .242
Contriever5 .565 .119 .101 .644 .275 .204 .643 .269 .201 .641 .259 .199 .643 .236 .202 .643 .271 .202
10 .575 .127 .119 .654 .298 .246 .652 .298 .243 .648 .295 .239 .651 .255 .242 .654 .294 .244
25 .560 .129 .120 .624 .308 .251 .625 .287 .257 .618 .295 .252 .619 .262 .248 .626 .279 .254
50 .526 .136 .118 .629 .297 .241 .634 .240 .240 .641 .252 .245 .629 .243 .240 .624 .239 .236
SBERT5 .536 .102 .075 .617 .244 .189 .616 .236 .188 .617 .231 .187 .615 .215 .186 .612 .233 .184
10 .544 .121 .095 .619 .266 .218 .615 .255 .213 .617 .250 .213 .615 .228 .213 .610 .249 .206
25 .525 .114 .079 .605 .273 .228 .600 .256 .224 .608 .258 .234 .597 .230 .221 .590 .245 .208
50 .519 .119 .087 .601 .281 .242 .596 .243 .233 .605 .253 .245 .591 .231 .233 .584 .231 .216
SFR5 .568 .171 .130 .615 .266 .218 .616 .268 .217 .615 .262 .216 .612 .227 .212 .614 .270 .214
10 .550 .169 .119 .623 .300 .248 .623 .307 .249 .626 .305 .252 .619 .260 .242 .623 .304 .249
25 .553 .173 .136 .623 .312 .272 .625 .310 .275 .626 .315 .279 .624 .263 .272 .622 .305 .269
50 .551 .172 .135 .628 .298 .269 .614 .263 .249 .618 .272 .256 .622 .240 .257 .611 .260 .244
Qwen5 .598 .190 .180 .663 .325 .305 .659 .321 .298 .654 .310 .290 .658 .277 .296 .662 .324 .304
10 .594 .188 .183 .667 .366 .341 .665 .375 .340 .661 .363 .333 .663 .329 .335 .667 .378 .342
25 .571 .193 .180 .653 .401 .360 .652 .407 .356 .650 .396 .353 .646 .363 .353 .650 .407 .353
50 .583 .197 .190 .637 .404 .362 .630 .380 .345 .626 .375 .344 .628 .362 .352 .632 .382 .343
Instructor-Large5 .581 .144 .121 .649 .274 .235 .651 .273 .238 .651 .270 .237 .650 .241 .236 .652 .274 .238
10 .545 .139 .095 .639 .298 .247 .638 .288 .241 .638 .286 .241 .635 .258 .240 .637 .289 .240
25 .534 .139 .093 .619 .314 .253 .609 .296 .231 .610 .297 .233 .611 .266 .237 .608 .296 .231
50 .525 .138 .090 .594 .292 .233 .584 .248 .207 .586 .251 .210 .588 .240 .222 .584 .249 .207
Reasoning Encoders
Diver-Retriever5 .567 .139 .135 .625 .295 .271 .622 .288 .265 .619 .288 .260 .622 .234 .264 .623 .285 .267
10 .572 .150 .154 .633 .342 .316 .629 .337 .313 .628 .338 .312 .627 .283 .307 .631 .331 .313
25 .543 .159 .159 .605 .347 .316 .606 .336 .308 .605 .341 .309 .599 .298 .303 .606 .329 .308
50 .554 .164 .163 .628 .339 .313 .624 .294 .278 .624 .301 .281 .631 .285 .298 .622 .290 .277
RaDeR5 .566 .130 .125 .655 .312 .295 .654 .309 .293 .652 .306 .292 .651 .253 .290 .653 .307 .294
10 .550 .131 .117 .653 .355 .329 .651 .357 .328 .651 .359 .327 .648 .294 .322 .650 .352 .326
25 .560 .136 .133 .645 .358 .334 .640 .349 .324 .640 .355 .327 .644 .294 .331 .640 .342 .323
50 .548 .139 .142 .660 .353 .356 .650 .309 .322 .653 .318 .327 .650 .266 .335 .647 .303 .319
ReasonIR5 .564 .117 .117 .627 .268 .237 .628 .268 .240 .627 .269 .239 .626 .227 .237 .627 .260 .237
10 .559 .126 .124 .624 .298 .265 .625 .303 .269 .623 .304 .267 .623 .254 .262 .622 .290 .261
25 .520 .120 .101 .611 .309 .273 .616 .308 .281 .626 .321 .293 .616 .255 .281 .606 .288 .266
50 .534 .123 .118 .616 .303 .291 .620 .272 .271 .623 .295 .285 .618 .243 .280 .614 .255 .260
Table D.4: Comprehensive zero-cost QPP evaluation across all target cutoffs ( k∈ {5,10,25,50} ) for theBRIGHT
benchmark. Predictive power is evaluated using AUROC (AUC), Pearson (r), and Spearman (ρ).

ModelkMaxScore ScoreGap Top-kStd NQC MaxIterStd LSMV
AUCr ρAUCr ρAUCr ρAUCr ρAUCr ρAUCr ρ
Sparse Models
BM255 .508 .029 .040 .563 .095 .115 .561 .086 .112 .573 .119 .126 .559 .039 .110 .555 .045 .103
10 .504 .019 .030 .581 .106 .148 .571 .105 .130 .608 .175 .190 .563 .052 .124 .548 .055 .097
25 .472 -.000 -.015 .526 .070 .091 .529 .062 .090 .602 .174 .209 .530 .026 .096 .510 .014 .059
50 .505 .017 .024 .559 .095 .126 .560 .071 .118 .642 .209 .258 .560 .048 .128 .526 .028 .061
Dense Models
BGE5 .626 .198 .226 .629 .210 .260 .633 .204 .260 .628 .206 .252 .632 .116 .251 .638 .195 .269
10 .626 .198 .226 .673 .272 .326 .670 .266 .320 .664 .266 .310 .670 .156 .320 .672 .253 .323
25 .626 .198 .226 .701 .302 .364 .698 .317 .349 .696 .320 .347 .703 .187 .357 .705 .303 .361
50 .626 .198 .226 .708 .314 .371 .704 .311 .361 .704 .316 .360 .714 .190 .366 .705 .299 .368
E55 .619 .192 .217 .668 .227 .306 .663 .217 .301 .660 .218 .296 .662 .109 .300 .667 .206 .306
10 .645 .201 .237 .708 .270 .362 .709 .272 .356 .706 .273 .353 .702 .129 .341 .710 .257 .354
25 .648 .211 .242 .709 .311 .382 .719 .344 .394 .719 .348 .399 .708 .145 .377 .718 .327 .380
50 .625 .218 .251 .670 .323 .384 .681 .358 .387 .677 .363 .388 .661 .147 .373 .686 .342 .386
Contriever5 .585 .101 .121 .649 .240 .264 .638 .234 .250 .633 .243 .244 .636 .148 .248 .639 .213 .251
10 .588 .105 .127 .692 .281 .322 .685 .279 .306 .671 .285 .296 .683 .171 .306 .688 .256 .310
25 .591 .129 .159 .705 .312 .377 .715 .317 .368 .708 .323 .359 .701 .191 .358 .707 .295 .358
50 .596 .147 .176 .694 .328 .381 .698 .327 .357 .689 .333 .357 .690 .198 .350 .694 .306 .354
SBERT5 .636 .193 .209 .657 .225 .270 .651 .223 .260 .641 .209 .250 .649 .130 .256 .657 .212 .269
10 .624 .185 .199 .673 .276 .316 .673 .279 .313 .659 .263 .299 .665 .148 .302 .677 .259 .326
25 .638 .216 .236 .691 .326 .371 .716 .370 .408 .696 .343 .369 .694 .202 .367 .720 .344 .408
50 .654 .241 .261 .708 .366 .412 .743 .403 .437 .717 .380 .405 .705 .232 .400 .747 .381 .428
SFR5 .625 .162 .215 .627 .200 .264 .616 .184 .244 .612 .186 .238 .615 .090 .242 .623 .172 .257
10 .647 .181 .239 .675 .255 .338 .675 .256 .340 .667 .257 .332 .664 .111 .311 .680 .239 .344
25 .649 .194 .236 .671 .300 .362 .688 .331 .376 .684 .333 .372 .675 .127 .344 .690 .311 .368
50 .622 .195 .237 .660 .315 .369 .698 .359 .385 .692 .365 .390 .679 .137 .363 .694 .340 .373
Qwen5 .483 .013 -.005 .643 .217 .268 .643 .216 .270 .640 .235 .260 .641 .127 .265 .640 .186 .267
10 .503 .018 .023 .648 .257 .303 .646 .266 .288 .650 .286 .288 .645 .157 .285 .641 .231 .281
25 .536 .028 .022 .661 .303 .302 .662 .304 .289 .660 .318 .297 .656 .182 .277 .654 .270 .273
50 .510 .023 .002 .648 .328 .320 .654 .331 .299 .659 .348 .328 .643 .215 .282 .635 .293 .259
Instructor-Large5 .556 .046 .097 .624 .181 .253 .619 .177 .251 .623 .180 .252 .623 .092 .258 .620 .170 .251
10 .553 .069 .127 .617 .219 .282 .611 .224 .269 .615 .227 .276 .611 .116 .273 .611 .216 .267
25 .563 .086 .125 .664 .261 .332 .671 .280 .324 .670 .284 .320 .660 .129 .319 .669 .271 .320
50 .557 .090 .140 .664 .260 .335 .705 .294 .358 .708 .298 .361 .664 .117 .343 .698 .285 .349
Reasoning Encoders
Diver-Retriever5 .563 .098 .124 .619 .209 .282 .619 .210 .279 .617 .213 .270 .618 .110 .276 .618 .199 .277
10 .558 .084 .104 .650 .239 .325 .657 .257 .323 .656 .262 .318 .648 .122 .319 .656 .242 .323
25 .582 .106 .125 .666 .274 .351 .681 .309 .349 .681 .314 .353 .663 .143 .350 .684 .293 .357
50 .557 .118 .135 .626 .285 .342 .643 .304 .323 .640 .309 .327 .625 .147 .330 .643 .291 .330
RaDeR5 .553 .052 .078 .601 .136 .197 .600 .139 .193 .597 .142 .190 .600 .068 .195 .602 .133 .195
10 .543 .075 .083 .630 .223 .267 .631 .234 .256 .632 .237 .257 .625 .139 .248 .631 .225 .258
25 .570 .087 .100 .646 .261 .307 .668 .299 .311 .670 .304 .315 .658 .175 .299 .665 .287 .309
50 .619 .093 .102 .650 .280 .345 .653 .315 .334 .635 .322 .328 .648 .180 .315 .659 .303 .320
ReasonIR5 .531 .003 .041 .630 .142 .260 .629 .142 .255 .632 .161 .258 .630 .040 .255 .627 .103 .249
10 .547 .001 .052 .638 .174 .281 .633 .190 .275 .641 .211 .295 .632 .042 .262 .635 .138 .263
25 .544 .019 .064 .638 .198 .311 .652 .246 .322 .653 .268 .338 .641 .048 .301 .642 .183 .298
50 .581 .025 .072 .685 .211 .338 .698 .270 .350 .685 .291 .361 .675 .048 .328 .698 .206 .304
Table D.5: Comprehensive zero-cost QPP evaluation across all target cutoffs ( k∈ {5,10,25,50} ) for theTEMPO
benchmark. Predictive power is evaluated using AUROC (AUC), Pearson (r), and Spearman (ρ).

Subset MaxScore Gap@25∆
FiQA 0.643 0.761 +0.118
NFCorpus 0.749 0.736 -0.013
SciDocs 0.606 0.678 +0.072
SciFact 0.756 0.816 +0.060
Macro Avg.0.689 0.748 +0.059
Table D.6: Per-subset AUROC@25 for MaxScore ( s1)
and Score Gap ( s1−s25), macro-averaged across 11 re-
trievers onBEIR. ∆= Gap@25 −MaxScore. Gap@25
outperforms MaxScore on 3 of 4 subsets. NFCorpus,
a known outlier in BEIR with very short queries and
an atypical corpus structure, is the single exception
(∆ =−0.013).
Subset MaxScore Gap@25∆
Biology 0.452 0.629 +0.177
AoPS 0.528 0.610 +0.082
Earth Sci. 0.595 0.617 +0.022
Economics 0.512 0.537 +0.025
LeetCode 0.670 0.679 +0.009
Pony 0.528 0.526 -0.002
Psychology 0.511 0.613 +0.102
Robotics 0.624 0.610 -0.014
StackOverflow 0.543 0.601 +0.058
Sust. Living 0.618 0.666 +0.048
TheoremQA-Q 0.523 0.770 +0.247
TheoremQA-T 0.515 0.607 +0.092
Macro Avg.0.552 0.622 +0.071
Table D.7: Per-subset AUROC@25 for MaxScore ( s1)
and Score Gap ( s1−s 25), macro-averaged across 11
retrievers onBRIGHT. ∆= Gap@25 −MaxScore.
Gap@25 outperforms MaxScore on 10 of 12 subsets.
The two exceptions (Pony, Robotics) show differences
below 0.015 and are subsets with structured, domain-
specific queries where semantic similarity is more infor-
mative.Subset MaxScore Gap@25∆
Cardano 0.631 0.623 -0.008
Economics 0.553 0.639 +0.086
Genealogy 0.649 0.766 +0.117
History 0.644 0.755 +0.111
HSM 0.556 0.663 +0.107
Iota 0.689 0.833 +0.144
Law 0.404 0.511 +0.107
Monero 0.516 0.584 +0.068
Politics 0.606 0.728 +0.122
Quant 0.615 0.560 -0.055
Travel 0.479 0.608 +0.129
Workplace 0.662 0.670 +0.008
Macro Avg.0.584 0.662 +0.078
Table D.8: Per-subset AUROC@25 for MaxScore ( s1)
and Score Gap ( s1−s 25), macro-averaged across 11
retrievers onTEMPO. ∆= Gap@25 −MaxScore.
Gap@25 outperforms MaxScore on 10 of 12 subsets.
The two exceptions (Cardano, Quant) show differences
of≤0.05; Quant involves specialized financial queries
with peaked score distributions even under failed re-
trieval.