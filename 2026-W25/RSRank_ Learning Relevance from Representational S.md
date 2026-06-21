# RSRank: Learning Relevance from Representational Shifts

**Authors**: Archit Gupta, Sai Sundaresan, Debabrata Mahapatra

**Published**: 2026-06-16 03:29:23

**PDF URL**: [https://arxiv.org/pdf/2606.17468v1](https://arxiv.org/pdf/2606.17468v1)

## Abstract
As enterprises deploy RAG-based systems to provide grounded responses to user queries, reranking has become a critical component for the final filtering step that separates relevant from distracting or irrelevant documents. Existing rerankers often rely on heuristic thresholds to achieve optimal filtering. Moreover, for relevance scoring, state-of-the-art methods use a language model's logit signals, which are designed for next-token prediction, not for assessing relevance. To address these limitations, we identify a principled signal for relevance: the representational shift (RS) induced in a query's internal state when conditioned on a document. We observe that the alignment between (a) RS induced by a candidate document and (b) RS induced by an oracle document-set provides a robust indicator of relevance. Building on this insight, we introduce a lightweight training framework that learns projections mapping RS to calibrated relevance scores. Our training objectives naturally filter irrelevant content at a zero threshold, reducing dependence on heuristic tuning. Across diverse retrieval datasets, our method delivers gains over SOTA rerankers.

## Full Text


<!-- PDF content starts -->

RSRank: Learning Relevance from Representational Shifts
Archit Gupta1Sai Sundaresan1Debabrata Mahapatra1*
1Adobe Research, India
Abstract
As enterprises deploy RAG-based systems to
provide grounded responses to user queries,
reranking has become a critical component for
the final filtering step that separates relevant
from distracting or irrelevant documents. Ex-
isting rerankers often rely on heuristic thresh-
olds to achieve optimal filtering. Moreover,
for relevance scoring, state-of-the-art methods
use a language model’s logit signals, which
are designed for next-token prediction, not for
assessing relevance. To address these limita-
tions, we identify a principled signal for rel-
evance: therepresentational shift (RS)in-
duced in a query’s internal state when condi-
tioned on a document. We observe that the
alignment between (a) RS induced by a candi-
date document and (b) RS induced by an oracle
document-set provides a robust indicator of rel-
evance. Building on this insight, we introduce
a lightweight training framework that learns
projections mapping RS to calibrated relevance
scores. Our training objectives naturally filter
irrelevant content at a zero threshold, reducing
dependence on heuristic tuning. Across diverse
retrieval datasets, our method delivers gains
over SOTA rerankers.
1 Introduction
1.1 Role of Rerankers in Enterprise Systems
Information retrieval (IR) systems form founda-
tional infrastructure for enterprises that serve users
at scale. Search engines, knowledge bases, and AI
assistants increasingly depend on their ability to
identify relevant information from a large corpus
for a user query. A fundamental tradeoff that gov-
erns the design of these systems isefficiencyvs.ac-
curacy. Traditional methods like BM25 (Robertson
and Zaragoza, 2009) and modern dense embedding
approaches (Khattab and Zaharia, 2020; Nogueira
et al., 2020) are efficient, as documents are encoded
*Corresponding author: dmahapatra@adobe.comonce and indexed for fast lookup, but suffer from
an information bottleneck, being unable to repre-
sent query-specific information when constrained
to a single document representation (Luan et al.,
2021).
The introduction of reranking through a two-
stage retrieval pipeline (Wang et al., 2011) ad-
dresses this representational limitation. The first
stage retrieves a broad candidate set using efficient
methods (accepting some loss in precision), and
the second stage applies more expensive models
torerankthese candidates. Rerankers jointly rep-
resent query and document tokens, capturing se-
mantic relationships that independent encodings
miss, improving retrieval quality (Nogueira and
Cho, 2020). More recently, LLM-based rerankers
have extended this paradigm, achieving SOTA re-
sults (Zhang et al., 2025). This two-stage archi-
tecture is now standard in enterprise systems (Liu
et al., 2017; Microsoft Research, 2021).
1.2 Reranking in Retrieval Systems
Retrieval systems (Lewis et al., 2020) have widely
adopted rerankers, with vector database platforms
providing native reranker support (Pinecone, 2024;
Weaviate, 2024), and cloud providers offering
reranking through APIs (Amazon Web Services,
2024; Google Cloud, 2025). In these systems, re-
trieved documents enter the LM’s context window,
and irrelevant documents degrade response qual-
ity (Liu et al., 2024; Wu et al., 2024), increase
latency, and raise API costs (Pinecone, 2024). Ac-
curate selection is therefore critical, and the trade-
off between accuracy and efficiency becomes even
more consequential in this setting.
In practice, the standard approach is to retrieve a
broad candidate set (e.g., top-100) via embedding
search, rerank, and then select a subset for the LM’s
context. The selection step is typically performed
via fixed top-kselection or score thresholding.
1arXiv:2606.17468v1  [cs.IR]  16 Jun 2026

1.3 Limitations of Existing Rerankers
Neither approach discussed above adequately ad-
dresses the efficiency–accuracy tradeoff. A fixed
top-kselection ignores that queries differ in the
number of relevant documents, often selecting
too many or too few documents per query. De-
termining a score cutoff is also difficult because
rerankers are not calibrated for absolute relevance;
optimal values vary across domains and even across
queries. We quantify these inefficiencies empiri-
cally in Sec. 2.
These limitations stem, in part, from how
rerankers derive their signal. Current approaches
rely on signals tuned for next-token prediction (in-
ternal states, attention maps, logits), rather than rel-
evance assessment (Zhang et al., 2025; Chen et al.,
2024, 2025a). The resulting scores are effective for
ranking—ordering documents by relevance—but
poorly calibrated forselection—deciding which
documents are relevant.
1.4 Toward a Calibrated Relevance Signal
The limitations above motivate a search for a dif-
ferent relevance signal—one that is inherently cali-
brated rather than repurposed from the next-token
prediction objective. We observe that relevance
fundamentally concerns how a document changes
the model’s internal representation of a query: a
relevant document should shift the model’s inter-
nal representation characteristically. This obser-
vation leads us to formalize and studyrepresen-
tational shifts, the change in the model’s repre-
sentation of the query induced by a document in
context. In Sec. 4, we show that the geometry of
RS encodes relevance information that, when trans-
formed through a learned projection can output
scores that are calibrated towards a natural deci-
sion boundary.
1.5 Contributions
Our key contributions are as follows: We highlight
thethreshold inconsistencyproblem in current
SOTA rerankers and provide a means to quantify
its impact. We identifyrepresentational shiftsas
a relevance signal: changes in the query’s value
vectors induced by conditioning on a document
in context. We introduce alightweight learning
frameworkthat maps the representational shift
space to calibrated scores, yielding a consistent de-
cision boundary across datasets. We demonstrate
competitive performanceacross six diverse re-Table 1: Paired t-test: per-query optimal F1 vs. dataset-
optimal threshold F1 for Qwen3-Reranker-8B.
DatasetNQ-Opt D-Opt Gapt p-value
2WikiMQA 500 73.0 55.0 18.1 24.22<10−99
Fever 500 100.0 99.5 0.5 2.736.5×10−3
FiQA 500 99.1 95.0 4.0 8.056.0×10−15
HotpotQA 500 79.7 61.0 18.8 23.94<10−99
MuSiQue 500 77.7 61.0 16.7 22.93<10−99
NFCorpus 323 82.1 64.9 17.2 15.40<10−99
trieval datasets, achieving 2.0 and 7.2-point gains
in Recall@5 and F1 at the natural threshold rel-
ative to baselines, while using only 2.3M trained
parameters on top of frozen LLM representations.
2 Threshold Inconsistency in Rerankers
We evaluate Qwen-Reranker-8B (Zhang et al.,
2025) on six retrieval datasets (Sec. 5) and ana-
lyze the resulting score distributions to show the
threshold inconsistency problem.
2.1 Optimal Thresholds Vary Across Datasets
For each dataset, we compute theoptimal threshold,
the threshold achieving the highest mean F1, across
500 queries. We plot this against the per-dataset
score range after applying a global min-max nor-
malization (mapping the score range to [0,1] ) so
that models with different native scales can be com-
pared directly. To quantify dataset-level calibra-
tion we report two metrics:Bias, the absolute off-
set between the optimal threshold and the model’s
natural decision boundary ( τ), andVariance, the
spread of per-dataset optimal thresholds around
their mean. Fig. 1 shows the results for Qwen3-
Reranker-8B: bias=0.379, variance=0.023. This
reveals that the natural threshold ( τ=0.5 ) consis-
tently overshoots the true decision boundary, while
the optimal threshold varies substantially across
domains. Consequently, effective deployment re-
quires labeled data for calibration, limiting out-of-
the-box performance.
2.2 Fixed Thresholds Hurt Individual Queries
Even within a single dataset, the optimal threshold
varies substantially across queries. Fig. 2 illustrates
this effect on HotpotQA by showing the fraction
of queries for which the F1 score obtained using
the dataset-level optimal threshold falls short of the
per-query optimal F1. For Qwen3-Reranker-8B,
63% of queries incur an F1 loss greater than 0.1,
and 30% incur a loss greater than 0.3.
2

0.0 0.2 0.4 0.6 0.8 1.0
Reranker Score (globally normalized)HotpotQA
2WikiMQA
MuSiQue
Fever
FiQA
NFCorpus0.42
0.20
0.08
0.01
0.01
0.00 = 0.5
Figure 1: Optimal threshold for Qwen3-Reranker-8B
for F1 across datasets. The x-axis shows the range of
scores (globally normalized); the optimal threshold is
indicated by the red dot.
0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
Per-Query F1 Gap020406080100% of Queries Exceeding Gap63%
40%
30%
Figure 2: F1 gap CDF for Qwen3-Reranker-8B on
HotpotQA. The x-axis shows the F1 gap between the
dataset-level optimal threshold and the per-query opti-
mal threshold; the y-axis shows the fraction of queries
exceeding that gap. 63% of queries lose ≥0.1 F1 from
using a fixed threshold.
To quantify performance loss attributable specifi-
cally to poor calibration rather than reranking qual-
ity, we conduct a paired t-test comparing dataset-
level optimal F1 scores with per-query optimal
F1 scores. Table 1 reports the results for Qwen3-
Reranker-8B across datasets. In all cases, the differ-
ence between the two scores is statistically signifi-
cant ( p <0.05 ), indicating that a large portion of
the observed performance gap arises from calibra-
tion error rather than limitations in ranking ability.
3 Related Work
Ranking paradigms.Reranking methods can be
broadly categorized into three paradigms.Point-
wisemethods score each query–document pair in-
dependently, enabling efficient threshold-based se-
lection (Nogueira et al., 2020; Ma et al., 2024;
Zhang et al., 2025).Listwisemethods condition
on the entire candidate set and directly optimize
or generate ranked permutations (Pradeep et al.,
2023; Sun et al., 2023).Setwisemethods iteratively
identify the most relevant document from subsetsof candidates (Chen et al., 2025b; Zhuang et al.,
2024). Since listwise and setwise approaches in-
cur substantially higher computational and latency
overheads for large candidate sets, we primarily
compare against pointwise rerankers.
Model architectures.Reranking models fall into
three architectural families.Cross-encodersjointly
encode the query and document for relevance pre-
diction (Nogueira et al., 2020; Pradeep et al., 2021;
Khattab and Zaharia, 2020).Open-source LLMs
have been adapted for pointwise, listwise, and set-
wise reranking (Ma et al., 2024; Pradeep et al.,
2023; Zhang et al., 2025; Meng et al., 2025; Sun
et al., 2026; BehnamGhader et al., 2026).Closed-
source LLMsare often used as zero-shot rerankers
via prompting (Sun et al., 2023). Across these fam-
ilies, relevance is typically inferred from language
modeling objectives rather than explicit relevance
supervision. We primarily compare against cross-
encoder and open-source LLM rerankers of equiva-
lent scale.
Threshold Calibration.Prior work studies cali-
bration techniques for making reranker scores com-
parable across queries and suitable for threshold-
based decisions. Methods based on Platt scaling
and related calibration approaches (Platt, 1999;
Posokhov et al., 2025; Ren et al., 2025; Yu et al.,
2025) convert raw scores into calibrated confi-
dence estimates, but still require task-specific cal-
ibration and externally defined thresholds. Other
approaches derive statistically grounded thresh-
olds (Li et al., 2022) or use predictive uncertainty
for selective acceptance (Yoon and Sael, 2025;
Yoon et al., 2025), yet they also depend on aux-
iliary decision rules. RSRank learns a consistent
relevance boundary directly during training, while
remaining compatible with post-hoc calibration and
thresholding techniques.
Probing LLM Representations.Recent work
shows that intermediate LLM representations en-
code task-relevant signals. Intermediate layers im-
prove embedding quality (Skean et al., 2025), con-
trastive layer analysis enhances factuality (Zhang
et al., 2024), hidden states support document attri-
bution (Phukan et al., 2024), and attention patterns
have been used for reranking (Chen et al., 2025a).
These findings highlight the importance of leverag-
ing model internals beyond next-token prediction.
In contrast to prior work on hidden states or atten-
tion weights, our approach uses value vector shifts
3

to capture how a document updates the model’s
internal representation of a query, directly aligning
the representation with the relevance decision.
Intrinsic Geometry in LLMs.Prior work shows
that neural representations are highly anisotropic,
often forming a cone-shaped geometry in embed-
ding space (Ait-Saada and Nadif, 2023), a property
observed consistently across layers and architec-
tures (Razzhigaev et al., 2024; Skean et al., 2025).
Rather than being a training artifact, this geometry
has been argued to encode meaningful semantic
and structural information (Godey et al., 2024; Ku-
drjashov et al., 2025). Although some methods
attempt to suppress anisotropy through normaliza-
tion or whitening, recent evidence suggests that
doing so can degrade generation and downstream
performance, implying that anisotropy itself car-
ries useful signals (Godey et al., 2024; Kudrjashov
et al., 2025). These observations motivate our ap-
proach: we leverage the anisotropic structure of RS
to extract a relevance-discriminative signal.
4 Methodology
4.1 Finite Difference as Representational Shift
We consider a decoder-only Transformer with L
layers and Hattention heads per layer. Let Σde-
note the vocabulary and let a document and query
be token sequences d= (d 1, . . . , d n)∈Σnand
q= (q 1, . . . , q m)∈Σm, respectively. We study
how the document prefix alters internal representa-
tions of query tokens duringprefill.
Pre-attention value vectors.Fix a layer ℓ∈[L]
and head h∈[H] with head dimension dh. Let
r(ℓ−1)
i(x)∈RDdenote the residual stream vector
entering layer ℓat position ifor input sequence
x, and let W(ℓ,h)
V∈Rdh×Dbe the value projec-
tion matrix for head (ℓ, h) . The pre-attention value
vector at positioniis
v(ℓ,h)
i(x) :=W(ℓ,h)
Vr(ℓ−1)
i(x)∈Rdh.(1)
In our reranking setup, we focus on value vectors of
the query tokens in the concatenated input x=d·q .
Controlling for prefix length.Prepending a doc-
ument changes both (i) thecontentavailable for
attention and (ii) theabsolute positionsof query to-
kens. Since our goal is to isolate onlyprefix content
effects, we need to compare the document prefix to
a length-matchednull prefix ∅n∈Σnthat carries
minimal semantic content (e.g., padding or benignfiller tokens), yielding a controlled finite-difference
feature. Specifically, for each query token position
t∈[m] , we define the document-induced delta of
the value vector at head(ℓ, h)as
δv(ℓ,h)
t(d;q) :=v(ℓ,h)
n+t(d·q)−v(ℓ,h)
n+t(∅n·q)∈Rdh
(2)
a standard construction in discrete/finite-
difference calculus (Appendix A). However, specif-
ically for value vector based signals, we can sim-
plify this construction to Eq. (3) when our base
model uses RoPE (Su et al., 2024), which encodes
relative positions. Under RoPE, the QK attention
between tokens of the query remains the same re-
gardless of how far the query is shifted, and since
the value vectors themselves are not subject to
RoPE, they are not affected by their position.
δv(ℓ,h)
t(d;q) :=v(ℓ,h)
n+t(d·q)−v(ℓ,h)
n+t(q)∈Rdh(3)
Representational shift tensor.Let I ⊆[L]×
[H]be a selected set of Llayers and Hheads.
Collecting the per-head deltas across query token
positions t∈[m] , we define the representational
shift tensor∆(d, q)∈RL×H×T×Das
∆(d, q) ℓ,h,t:=δv(ℓ,h)
t(d;q)(4)
where D≡d his the head dimension. Intu-
itively, ∆(d, q) captures how the document prefix
re-contextualizeseach query token in value space,
at a resolution indexed by (ℓ, h, t) . Given a set
of documents D, we write ∆(D, q) for the shift
induced by conditioning on all documents inD.
4.2 Representational Shift Models Relevance
Notation.For a query qwith candidate document
setDq, letD+
q⊆ D qdenote the set of relevant and
D−
q=D q\ D+
qthe irrelevant documents.
Oracle shift.We define theoracle shift
∆∗(q) :=∆(D+
q, q)as the representational shift
induced by conditioning onallrelevant documents
simultaneously. This shift encodes the aggregate
effect that the complete relevant context has on the
model’s internal representations of the query.
Oracle-similarity ranking.For each candidate
document di∈ D q, we compute the cosine sim-
ilarity between its individual shift ∆(d i, q)and
the oracle shift ∆∗(q), and rank documents ac-
cordingly. On the 2WikiMQA validation set,
this oracle-similarity ranking achieves R@5 = 89.5,
4

P@5 = 42.8, and F1@5 = 57.9—demonstrating that
alignment with oracle is effective in separating rel-
evant and irrelevant documents, and that the geom-
etry of the shift space encodes relevance structure.
From oracle similarity to learned projection.
The oracle shift is unavailable at inference time.
However, the experiment above suggests that if we
can learn a projection Bthat transforms the RS
space such that B ∆∗(q)falls in a chosen orthant
specifically aligning to 1Pacross queries, then
the oracle similarity cos 
B ∆(d, q),B ∆∗(q)
re-
duces to cos 
B ∆(d, q),1 P
, applicable at infer-
ence. This motivates our learning framework.
4.3 Learning Calibrated Projections
4.3.1 Projection Matrix
We learn a projection matrix B∈RL×H×P×D,
where Pis the projection dimension. For each layer
ℓ, head h, the submatrix Bℓ,h∈RP×Dprojects the
D-dimensional shift into aP-dimensional space.
Scoring Function.Given a candidate document
d, we compute its relevance score as:
zℓ,h,t=B ℓ,h∆(d, q) ℓ,h,t∈RP(5)
s(d|q) =X
ℓ,h1
TTX
t=1cos(z ℓ,h,t,1P)(6)
where 1Pdenotes the all-ones vector in RP. In-
tuitively, Blearns to extract a “relevance direc-
tion” from representational shift vectors: docu-
ments whose projected shifts have high cosine sim-
ilarity with1 Preceive high scores.
4.3.2 Training Objectives
Our training objective consists of five terms de-
signed to achieve calibrated separation of relevant
and irrelevant documents at a fixed threshold.
Calibration Loss.The core objective pushes rel-
evant documents to have positive scores and irrele-
vant documents to have negative scores:
Lcal=1
|D+|X
d∈D+[-s(d)] ++1
|D−|X
d∈D−[s(d)] +
(7)
where [·]+= max(·,0) denotes the ReLU function
andD+,D−are the relevant and irrelevant docu-
ment sets as defined in Sec. 4.2. This loss creates a
natural decision boundary ats= 0.Margin Loss.To ensure robust separation, we
enforce a marginmbetween classes:
Lmargin =1
|D+|X
d∈D+[m−s(d)] +
+1
|D−|X
d∈D−[s(d) +m] +
+h
m− 
min
d∈D+s(d)−max
d∈D−s(d)i
+
(8)
The first two terms push relevant scores above +m
and irrelevant scores below −m. The third term
ensures that the lowest-scoring relevant document
exceeds the highest-scoring irrelevant document by
at leastm. We usem= 0.5in all experiments.
Orthogonality Regularization.To prevent di-
mension collapse in the projection, we regularize
eachB ℓ,hto have orthonormal rows:
Lortho=1
LHX
ℓ,h∥Bℓ,hB⊤
ℓ,h−IP∥2
F
P2(9)
This ensures all Pdimensions are effectively uti-
lized and prevents the projection from degenerating
to a lower-rank mapping.
Oracle Alignment.We provide explicit supervi-
sion on the direction of the oracle shift projection:
Lalign= 1−E ℓ,h,t
cos(B ℓ,h∆∗(q)ℓ,h,t,1P)
(10)
where ∆∗(q) =∆(D+, q)is the oracle represen-
tational shift induced by the full set of relevant
documents (Sec. 4.2). This anchors the “ideal rele-
vance direction” to the ones vector.
Magnitude Constraint.For training stability, we
bound the Frobenius norm ofB:
Lfrob=
∥B∥ F−c
+, c=1
2√
LHPD(11)
Total Objective.The complete training loss is:
L=λ 1Lcal+L margin+λ 2Lortho+λ 3Lalign+λ 4Lfrob
(12)
5 Experimental Setup
5.1 Baseline
We compare against Qwen3-Reranker-8B (Zhang
et al., 2025), a state-of-the-art LLM reranker built
on Qwen3-8B (Team, 2025) that produces cal-
ibrated binary relevance scores. For fairness,
5

RSRank uses frozen representations from the same
backbone. We focus on Qwen3-Reranker-8B be-
cause it substantially outperforms earlier rerankers
such as BGE-Reranker (Chen et al., 2024) and Jina
Reranker v2 (JinaAI, 2025). We additionally com-
pare against MonoT5 (Nogueira et al., 2020), a
standard cross-encoder baseline, and LLM2Vec-
Gen (BehnamGhader et al., 2026) (Qwen3-8B) to
isolate gains beyond representation quality alone.
5.2 Datasets
We evaluate on six retrieval datasets spanning multi-
hop reasoning and domain specific retrieval.
Multi-hop QA.2WikiMultihopQA(Ho et al.,
2020) focuses on compositional reasoning across
pairs of Wikipedia articles with sentence-level
supporting facts.HotpotQA(Yang et al., 2018)
emphasizes multi-hop reasoning in a distractor
setting, where each question is paired with 2
supporting and 8 TF-IDF-retrieved paragraphs.
MuSiQue(Trivedi et al., 2022) increases reason-
ing complexity by composing multiple single-hop
questions into multi-hop questions and includes
adversarial unanswerable examples.
Domain-Specific Retrieval.FiQA(Maia et al.,
2018) contains opinion-based financial question an-
swering data.FEVER(Thorne et al., 2018) is a
fact verification dataset where claims must be sup-
ported or refuted using evidence from Wikipedia.
NFCorpus(Boteva et al., 2016) is a biomedical
retrieval dataset linking natural-language nutrition
and medical queries to scientific documents.
Document granularity.The multi-hop QA
datasets (2WikiMQA, HotpotQA) operate at
sentence-level granularity, while BEIR datasets and
MuSiQue use paragraph-level documents. In prac-
tice, RAG pipelines operate on chunked passages
granularities, which is the primary setting we tar-
get. Our method is trained and evaluated jointly on
these mixed chunk lengths, demonstrating general-
isation across the passage sizes typical of chunked
retrieval. Reranking over longer-form documents
(e.g., entire articles) is an interesting direction but
falls outside the scope of this work.
5.3 Training
Training protocol.RS features are pre-
computed by running a forward pass per
query-document pair. We train the projection
matrix B(2.3M parameters; Sec. 4.3) on just2000samples from the datasets listed above
(excluding Fever, which serves as a zero-shot test).
Layer 0 is excluded from training since its value
vectors are raw token embeddings rather than
contextualized representations. Complete training
hyperparameters are provided in Appendix B.4.
Training cost.RSRank has a distinct advantage
in the training phase compared to the regular fine-
tuning carried out in Qwen3-Reranker-8B because
it works on top of frozen LLM representations,
training an independent projection matrix. Because
of this, the expensive forward pass of the LLM has
to be done only once and the backpropagation does
not need to update the parameters of the LLM.
5.4 Evaluation
Evaluation protocol.For each dataset, we evalu-
ate on up to 500 sampled queries from the test split.
For multi-hop datasets, each query comes with its
original set of gold and distractor documents. For
BEIR datasets, we pair each query with its relevant
documents plus 15 randomly sampled negatives
from the corpus.
Metrics.We report the following metrics:
1.NDCG@5: Normalized Discounted Cumula-
tive Gain at rank 5, measuring ranking quality.
2.Recall@5: Fraction of relevant documents ap-
pearing in the top 5 ranked positions, measuring
retrieval coverage.
3.F1@τ: F1 score computed by thresholding
scores at each method’snaturaldecision bound-
ary. This metric measures how well a method
separates relevant from irrelevant documents
without dataset-specific tuning, and is our pri-
mary metric for evaluating calibration quality.
Inference cost.RSRank requires an additional
query-only forward pass over baselines to compute
the null-prefix baseline, followed by a lightweight
projection and cosine similarity step. However, be-
cause the query-only pass is amortized across all
documents for a query, the effective per-document
overhead is negligible. As shown in Table 2, the
full RSRank pipeline adds only 15.4 ms (+0.88%)
over standard query+document forwards on an
A100-80GB GPU.
6 Results
6.1 Ranking Quality
As shown in Table 3, RSRank achieves the best av-
erage NDCG@5 (87.3) and Recall@5 (81.1) across
6

Table 2: Time to rerank 100 docs on A100-80GB
GPU (mean ±std over 30 runs) for Qwen3-8B.100 fwd:
query+document prompt batched.101 fwd: + one query-
only prompt.101 fwd + score: full RSRank pipeline.
Stage Prompt length Time (ms) Overhead
100 fwd100×229 1747.2±1.3—
101 fwd+1×82 1749.9±1.2 +2.7(+0.15%)
101 fwd + scoring same1762.6±6.4 +15.4(+0.88%)
0.0 0.2 0.4 0.6 0.8 1.0
Reranker Score (globally normalized)Fever
HotpotQA
2WikiMQA
FiQA
MuSiQue
NFCorpus0.43
0.41
0.39
0.38
0.38
0.36 = 0
Figure 3: Optimal threshold for RSRank for best mean
F1 across datasets. The x-axis shows the range of scores
(globally normalized) given by the reranker; the optimal
threshold is indicated by the red dot. Optimal threshold
Bias: 0.0221, Variance: 0.0005.
all six datasets. The largest gains appear on the
multi-hop benchmarks 2WikiMQA and MuSiQue,
where RSRank outperforms Qwen3-Reranker-8B
by 8.6 and 5.2pp in NDCG@5, respectively. On
HotpotQA and the BEIR benchmarks, performance
is largely comparable, with Qwen3-Reranker-8B
holding a slight edge on HotpotQA and NFCor-
pus. Both methods achieve near-perfect results
on FEVER and FiQA, while NFCorpus remains
challenging for both. These results demonstrate
that representational shifts provide a competitive
alternative to fully trained rerankers.
6.2 Selection Quality
RSRank, evaluated at its designed threshold of
τ=0, achieves the highest average F1 score (67.5),
outperforming Qwen3-Reranker-8B at its default
threshold of τ=0.5 (60.3) by 7.2pp. The largest
gap appears on NFCorpus, where Qwen3-Reranker-
8B is severely miscalibrated: its default thresh-
old yields an F1 of only 13.5, despite the dataset-
optimal threshold lying near zero. RSRank also
shows consistent gains on 2WikiMQA, MuSiQue,
and FiQA, suggesting that representational shifts
provide a more robust relevance signal under fixed-
threshold evaluation.
6.3 Threshold Stabilization Across Datasets
RSRank produces substantially more consistent
thresholds across datasets than Qwen3-Reranker-
Figure 4: UMAP visualization of representational shifts
on 2WikiMQA.Left:Raw shifts, where relevant (red)
and irrelevant (blue) documents overlap.Right:After
projection with B, relevant documents form a clearly
separated cluster.
8B, reducing threshold bias from 0.379 to 0.022
(17×lower) and variance from 0.023 to 0.0005
(47×lower). Fig. 3 shows that RS training aligns
per-dataset optimal thresholds under a shared
global normalization, improving calibration robust-
ness across datasets.
7 Analysis
We conduct ablation studies on the 2WikiMQA
validation set to understand which design choices
drive RSRank’s performance. Ablations on ar-
chitectural choices, loss components and compar-
isons with analytical baselines are presented in Ap-
pendix. B
7.1 Separability of Representational Shift
Fig. 4 visualizes RS vectors via UMAP for 100
queries (3115 irrelevant, 243 relevant documents).
Raw shifts (left) show relevant and irrelevant doc-
uments thoroughly intermixed—the shift signal
alone does not linearly separate classes. After the
learned projection B(right), relevant documents
consolidate into a distinct cluster, confirming that
Bextracts a relevance-discriminative subspace.
7.2 Sample Efficiency
The stability of the RS space allows for the learned
projection to converge with remarkably few exam-
ples. Table 4 shows that just 50 samples achieve
89.2% R@5 ( <5pp of the 800-sample model), and
performance saturates around 400 samples.
7.3 Per-Query Calibration
As established in Sec. 2, threshold calibration op-
erates at two levels:dataset-levelandquery-level.
Sec. 6.3 showed that RSRank effectively addresses
dataset-level calibration. We now examine the
query-level picture.
7

Table 3: Results across six retrieval datasets. Best result per column inbold; ties within 1 point are co-bolded.
F1@τuses each method’s default decision boundary without dataset-specific tuning.
Method 2WikiMQA HotpotQA MuSiQue FiQA Fever NFCorpus Avg
NDCG@5↑
MonoT5-base (0.2B) 61.8 68.6 66.7 96.5 99.9 80.0 78.9
MonoT5-3B 64.4 73.7 73.5 98.5100.084.9 82.5
LLM2Vec-Gen (Qwen3-8B) 55.7 60.0 62.9 97.0 99.7 83.6 76.5
Qwen3-Reranker-8B 71.681.178.899.4 100.086.9 86.3
RSRank (Ours)80.279.684.097.5 99.2 83.487.3
Recall@5↑
MonoT5-base (0.2B) 63.0 71.6 68.6 94.4 99.8 31.0 71.4
MonoT5-3B 66.0 76.4 76.7 96.3 99.9 34.6 75.0
LLM2Vec-Gen (Qwen3-8B) 60.5 63.9 69.8 95.199.835.1 70.7
Qwen3-Reranker-8B 74.084.384.096.7 99.835.8 79.1
RSRank (Ours)85.183.088.8 95.699.7 34.381.1
F1@τ(natural threshold)↑
MonoT5-base(τ=0.5)45.9 47.3 49.0 53.9 91.3 11.5 49.8
MonoT5-3B(τ=0.5)48.1 53.3 55.4 62.2 91.5 9.1 53.3
LLM2Vec-Gen (Qwen3-8B)(τ=0)16.7 11.9 18.2 23.8 13.161.524.2
Qwen3-Reranker-8B(τ=0.5)51.960.657.1 80.698.213.5 60.3
RSRank (Ours)(τ=0)60.551.861.2 85.983.462.2 67.5
Table 4: Sample efficiency on 2WikiMQA (10 epochs,
Qwen3-8B). Performance saturates by 400 samples.
Samples0 50 100 200 400 800
R@528.7 89.2 90.7 91.1 93.8 93.9
F1@013.2 74.5 76.9 77.7 80.8 82.1
Fig. 5 decomposes each model’s F1 into two
components: the dataset-optimal F1 and the
additional headroom to per-query optimal F1.
RSRank achieves a higher per-query optimal F1
than Qwen3-Reranker-8B on average, indicating
stronger underlying ranking quality. The headroom
from dataset-optimal to query-optimal is larger for
RSRank (+15.9 vs. +12.6 on average). These re-
sults indicate that RSRank provides a strong foun-
dation with better dataset-level calibration. The
superior per-query ranking of RSRank indicates
that future work on per-query calibration can fur-
ther improve the performance.
8 Conclusion
We present RSRank, a reranking method that uses
representational shifts (RS) of value vectors to pro-
duce relevance scores calibrated at a natural deci-
sion boundary. Our motivation identifies two levels
of threshold calibration—dataset-level and query-
level—and shows how existing rerankers suffer in
both areas. RSRank addresses dataset-level cali-
bration by learning a lightweight projection (2.3M
params) that maps RS to scores, reducing dataset-
level threshold bias by 17× and variance by 47×
2WikiMQA HotpotQA MuSiQue FiQA Fever NFCorpus Avg020406080100F1 Score (%)+18.1+16.7 +18.8 +22.7+16.7+20.8+4.0+10.3+0.5 +7.8
+17.2+17.3+12.6 +15.9Qwen3-Reranker-8B  (dataset-optimal F1)
Qwen3-Reranker-8B  (+ headroom to query-optimal)RSRank (Ours)  (dataset-optimal F1)
RSRank (Ours)  (+ headroom to query-optimal)Figure 5: Dataset-optimal F1 and headroom to per-
query optimal F1 for Qwen3-Reranker-8B and RSRank.
RSRank achieves higher per-query optimal F1 on aver-
age (86.3 vs. 85.3), indicating better ranking quality
compared to baselines. Across six diverse retrieval
benchmarks, RSRank achieves the highest average
NDCG@5, Recall@5, and F1 at its natural thresh-
old, outperforming SOTA Qwen3-Reranker-8B. A
headroom analysis further shows that RSRank at-
tains higher per-query optimal F1 than the baseline,
confirming stronger underlying ranking quality.
Future Work.The remaining headroom from
dataset to per-query optimal F1 (+15.9pp on av-
erage) shows that query-level threshold selection
could unlock further gains without retraining. An-
other direction is end-to-end evaluation within a
full RAG pipeline, integrating RSRank with up-
stream embedding retrieval and downstream gener-
ation to measure its impact on final answer quality.
8

Limitations
Random negatives vs. hard negatives.Our
BEIR evaluation pairs queries with randomly sam-
pled corpus negatives rather than hard negatives
from a retrieval stage. Random negatives could
be topically unrelated and easier for any model
to distinguish. While this reflects a realistic RAG
context filteringscenario—where a first-stage re-
triever has already narrowed the candidate set—it
overestimates absolute performance compared to
full-corpus retrieval benchmarks. Evaluation with
hard negatives from a BM25 or dense retrieval first
stage is needed to assess robustness in more adver-
sarial settings.
Per-query calibration.While RSRank’s zero-
threshold design provides better cross-dataset sta-
bility than baselines (Sec. 2), query-level calibra-
tion remains imperfect. On HotpotQA, the gap be-
tween F1@ τand F1@ τ∗indicates that many indi-
vidual queries would benefit from a query-specific
threshold. Improving per-query calibration is an
important direction for future work.
Ethical Considerations
We used AI assistants for support tasks such as im-
proving writing clarity, grammar, and formatting.
All technical content, experimental design, analy-
ses, and conclusions were developed, and verified
by the authors.
References
Mira Ait-Saada and Mohamed Nadif. 2023. Is
anisotropy truly harmful? a case study on text cluster-
ing. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume
2: Short Papers), pages 1194–1203, Toronto, Canada.
Association for Computational Linguistics.
Amazon Web Services. 2024. Amazon Bedrock now
supports Rerank API to improve accuracy of RAG
applications. Accessed: 2025-01-26.
Parishad BehnamGhader, Vaibhav Adlakha,
Fabian David Schmidt, Nicolas Chapados, Marius
Mosbach, and Siva Reddy. 2026. Llm2vec-gen:
Generative embeddings from large language models.
Preprint, arXiv:2603.10913.
Vera Boteva, Demian Gholipour, Artem Sokolov, and
Stefan Riezler. 2016. A full-text learning to rank
dataset for medical information retrieval. InProceed-
ings of the 38th European Conference on Information
Retrieval (ECIR), Padova, Italy.Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun
Luo, Defu Lian, and Zheng Liu. 2024. M3-
embedding: Multi-linguality, multi-functionality,
multi-granularity text embeddings through self-
knowledge distillation. InFindings of the Asso-
ciation for Computational Linguistics: ACL 2024,
pages 2318–2335, Bangkok, Thailand. Association
for Computational Linguistics.
Shijie Chen, Bernal Jimenez Gutierrez, and Yu Su.
2025a. Attention in large language models yields
efficient zero-shot re-rankers. InThe Thirteenth In-
ternational Conference on Learning Representations.
Yiqun Chen, Qi Liu, Yi Zhang, Weiwei Sun, Xinyu Ma,
Weiwei Yang, Daiting Shi, Jiaxin Mao, and Dawei
Yin. 2025b. Tourrank: Utilizing large language mod-
els for documents ranking with a tournament-inspired
strategy. InProceedings of the ACM Web Conference
2025, WWW ’25. Association for Computing Ma-
chinery.
Nathan Godey, Éric Clergerie, and Benoît Sagot. 2024.
Anisotropy is inherent to self-attention in transform-
ers. InProceedings of the 18th Conference of the
European Chapter of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pages
35–48, St. Julian’s, Malta. Association for Computa-
tional Linguistics.
Google Cloud. 2025. Boost your search and RAG
agents with Vertex AI’s new state-of-the-art Ranking
API. Accessed: 2025-01-26.
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara,
and Akiko Aizawa. 2020. Constructing a multi-
hop QA dataset for comprehensive evaluation of
reasoning steps. InProceedings of the 28th Inter-
national Conference on Computational Linguistics,
pages 6609–6625, Barcelona, Spain (Online). Inter-
national Committee on Computational Linguistics.
JinaAI. 2025. jinaai/jina-reranker-v2-base-multilingual
· hugging face. [Online; accessed 2026-02-09].
Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR Conference on Research
and Development in Information Retrieval, SIGIR
’20, page 39–48, New York, NY , USA. Association
for Computing Machinery.
Sergej Kudrjashov, Olesya Karpik, and Eduard Klyshin-
sky. 2025. Shrink the longest: Improving latent space
isotropy with simplicial geometry. InAnalysis of
Images, Social Networks and Texts, pages 120–130,
Cham. Springer Nature Switzerland.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, Sebastian Riedel, and Douwe Kiela. 2020.
9

Retrieval-augmented generation for knowledge-
intensive nlp tasks. InProceedings of the 34th Inter-
national Conference on Neural Information Process-
ing Systems, NIPS ’20, Red Hook, NY , USA. Curran
Associates Inc.
Minghan Li, Xinyu Zhang, Ji Xin, Hongyang Zhang,
and Jimmy Lin. 2022. Certified error control of can-
didate set pruning for two-stage relevance ranking.
InProceedings of the 2022 Conference on Empirical
Methods in Natural Language Processing, pages 333–
345, Abu Dhabi, United Arab Emirates. Association
for Computational Linguistics.
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the Association
for Computational Linguistics, 12:157–173.
Shichen Liu, Fei Xiao, Wenwu Ou, and Luo Si. 2017.
Cascade ranking for operational e-commerce search.
InProceedings of the 23rd ACM SIGKDD Interna-
tional Conference on Knowledge Discovery and Data
Mining, KDD ’17, page 1557–1565, New York, NY ,
USA. Association for Computing Machinery.
Yi Luan, Jacob Eisenstein, Kristina Toutanova, and
Michael Collins. 2021. Sparse, dense, and attentional
representations for text retrieval.Transactions of the
Association for Computational Linguistics, 9:329–
345.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei, and
Jimmy Lin. 2024. Fine-tuning llama for multi-stage
text retrieval. InProceedings of the 47th Inter-
national ACM SIGIR Conference on Research and
Development in Information Retrieval, pages 2421–
2425.
Macedo Maia, Siegfried Handschuh, André Freitas,
Brian Davis, Ross McDermott, Manel Zarrouk, and
Alexandra Balahur. 2018. Www’18 open challenge:
Financial opinion mining and question answering. In
Companion Proceedings of the The Web Conference
2018, WWW ’18, page 1941–1942, Republic and
Canton of Geneva, CHE. International World Wide
Web Conferences Steering Committee.
Siyuan Meng, Junming Liu, Yirong Chen, Song Mao,
Pinlong Cai, Guohang Yan, Botian Shi, and Ding
Wang. 2025. From ranking to selection: A simple
but efficient dynamic passage selector for retrieval
augmented generation.CoRR, abs/2508.09497.
Microsoft Research. 2021. The science behind seman-
tic search: How AI from Bing is powering Azure
Cognitive Search.
Rodrigo Nogueira and Kyunghyun Cho. 2020. Passage
re-ranking with bert.Preprint, arXiv:1901.04085.
Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and
Jimmy Lin. 2020. Document ranking with a pre-
trained sequence-to-sequence model. InFindings
of the Association for Computational Linguistics:EMNLP 2020, pages 708–718, Online. Association
for Computational Linguistics.
Anirudh Phukan, Shwetha Somasundaram, Apoorv Sax-
ena, Koustava Goswami, and Balaji Vasan Srinivasan.
2024. Peering into the mind of language models: An
approach for attribution in contextual question an-
swering. InFindings of the Association for Compu-
tational Linguistics: ACL 2024, pages 11481–11495,
Bangkok, Thailand. Association for Computational
Linguistics.
Pinecone. 2024. Introducing reranking to pinecone
inference. https://www.pinecone.io/blog/
introducing-reranking-to-pinecone-inference/ .
Accessed: 2026-02-06.
Pinecone. 2024. Rerankers and two-stage re-
trieval. https://pinecone.io/learn/series/
rag/rerankers. Accessed: 2025-01-26.
John C. Platt. 1999. Probabilistic outputs for support
vector machines and comparisons to regularized like-
lihood methods. In Alex J. Smola, Peter Bartlett,
Bernhard Schölkopf, and Dale Schuurmans, editors,
Advances in Large Margin Classifiers, pages 61–74.
MIT Press.
Pavel Posokhov, Sergei Masliukhin, Skrylnikov Stepan,
Danil Tirskikh, and Olesia Makhnytkina. 2025. Rele-
vance scores calibration for ranked list truncation via
TMP adapter. InFindings of the Association for Com-
putational Linguistics: ACL 2025, pages 7728–7734,
Vienna, Austria. Association for Computational Lin-
guistics.
Ronak Pradeep, Rodrigo Frassetto Nogueira, and Jimmy
Lin. 2021. The expando-mono-duo design pattern for
text ranking with pretrained sequence-to-sequence
models.CoRR, abs/2101.05667.
Ronak Pradeep, Sahel Sharifymoghaddam, and Jimmy
Lin. 2023. Rankzephyr: Effective and robust zero-
shot listwise reranking is a breeze!arXiv preprint
arXiv:2312.02724.
Anton Razzhigaev, Matvey Mikhalchuk, Elizaveta Gon-
charova, Ivan Oseledets, Denis Dimitrov, and Andrey
Kuznetsov. 2024. The shape of learning: Anisotropy
and intrinsic dimensions in transformer-based mod-
els. InFindings of the Association for Computational
Linguistics: EACL 2024, pages 868–874, St. Julian’s,
Malta. Association for Computational Linguistics.
Ruiyang Ren, Yuhao Wang, Kun Zhou, Wayne Xin
Zhao, Wenjie Wang, Jing Liu, Ji-Rong Wen, and Tat-
Seng Chua. 2025. Self-calibrated listwise reranking
with large language models. InProceedings of the
ACM on Web Conference 2025, WWW ’25, page
3692–3701, New York, NY , USA. Association for
Computing Machinery.
Stephen Robertson and Hugo Zaragoza. 2009. The
probabilistic relevance framework: Bm25 and be-
yond.Found. Trends Inf. Retr., 3(4):333–389.
10

Oscar Skean, Md Rifat Arefin, Dan Zhao, Niket Nikul
Patel, Jalal Naghiyev, Yann LeCun, and Ravid
Shwartz-Ziv. 2025. Layer by layer: Uncovering hid-
den representations in language models. InForty-
second International Conference on Machine Learn-
ing.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan,
Wen Bo, and Yunfeng Liu. 2024. Roformer: En-
hanced transformer with rotary position embedding.
Neurocomput., 568(C).
Jiashuo Sun, Pengcheng Jiang, Saizhuo Wang, Jia-
jun Fan, Heng Wang, Siru Ouyang, Ming Zhong,
Yizhu Jiao, Chengsong Huang, Xueqiang Xu,
Pengrui Han, Peiran Li, Jiaxin Huang, Ge Liu,
Heng Ji, and Jiawei Han. 2026. Rethinking the
reranker: Boundary-aware evidence selection for
robust retrieval-augmented generation.Preprint,
arXiv:2602.03689.
Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang
Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and
Zhaochun Ren. 2023. Is ChatGPT good at search?
investigating large language models as re-ranking
agents. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Process-
ing, pages 14918–14937, Singapore. Association for
Computational Linguistics.
Qwen Team. 2025. Qwen3 technical report.Preprint,
arXiv:2505.09388.
James Thorne, Andreas Vlachos, Christos
Christodoulopoulos, and Arpit Mittal. 2018.
FEVER: a large-scale dataset for fact extraction
and VERification. InProceedings of the 2018
Conference of the North American Chapter of
the Association for Computational Linguistics:
Human Language Technologies, Volume 1 (Long
Papers), pages 809–819, New Orleans, Louisiana.
Association for Computational Linguistics.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot,
and Ashish Sabharwal. 2022. MuSiQue: Multi-
hop questions via single-hop question composition.
Transactions of the Association for Computational
Linguistics, 10:539–554.
Lidan Wang, Jimmy Lin, and Donald Metzler. 2011. A
cascade ranking model for efficient ranked retrieval.
InProceedings of the 34th International ACM SIGIR
Conference on Research and Development in Infor-
mation Retrieval, SIGIR ’11, page 105–114, New
York, NY , USA. Association for Computing Machin-
ery.
Weaviate. 2024. Cohere reranker models with Weaviate.
https://weaviate.io/developers/weaviate/
model-providers/cohere/reranker . Accessed:
2025-01-26.
Siye Wu, Jian Xie, Jiangjie Chen, Tinghui Zhu, Kai
Zhang, and Yanghua Xiao. 2024. How easily do
irrelevant inputs skew the responses of large language
models? InFirst Conference on Language Modeling.Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2369–2380, Brussels, Belgium. Association for Com-
putational Linguistics.
Jeongnoh Yoon and Lee Sael. 2025. Document re-
ranking with evidential neural networks.IEEE Ac-
cess, 13:161964–161972.
Soyoung Yoon, Gyuwan Kim, GYU-HWUNG CHO,
and seung-won hwang. 2025. Acurank: Uncertainty-
aware adaptive computation for listwise reranking.
InThe Thirty-ninth Annual Conference on Neural
Information Processing Systems.
Puxuan Yu, Daniel Cohen, Hemank Lamba, Joel R.
Tetreault, and Alejandro Jaimes. 2025. Explain then
rank: Scale calibration of neural rankers using natu-
ral language explanations from LLMs. InFindings of
the Association for Computational Linguistics: ACL
2025, pages 22716–22730, Vienna, Austria. Associa-
tion for Computational Linguistics.
Jianyi Zhang, Da-Cheng Juan, Cyrus Rashtchian, Chun-
Sung Ferng, Heinrich Jiang, and Yiran Chen. 2024.
SLED: Self logits evolution decoding for improving
factuality in large language models. InThe Thirty-
eighth Annual Conference on Neural Information
Processing Systems.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang,
Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren
Zhou. 2025. Qwen3 embedding: Advancing text
embedding and reranking through foundation models.
arXiv preprint arXiv:2506.05176.
Shengyao Zhuang, Honglei Zhuang, Bevan Koopman,
and Guido Zuccon. 2024. A setwise approach for
effective and highly efficient zero-shot ranking with
large language models. InProceedings of the 47th
International ACM SIGIR Conference on Research
and Development in Information Retrieval, SIGIR
’24, page 38–47, New York, NY , USA. Association
for Computing Machinery.
11

A Discrete Difference Calculus on Prefix
Space
This appendix provides a formal viewpoint for the
feature construction in Sec. 4.1. The main paper
uses the first-order, length-matched difference fea-
tures (Equations (2)–(4)). The material below jus-
tifies terminology such as “finite difference” and
clarifies how deltas behave under sequences of pre-
fix edits.
A.1 Shift Operators on Prefixes
LetΣbe the vocabulary and Σ∗the set of all finite
token sequences. We viewΣ∗as a rooted directed
graph (a |Σ|-ary tree) whose vertices are prefixes
s∈Σ∗and whose edges correspond to appending
a token:s→s·afor anya∈Σ.
For a function F: Σ∗→Rk, define theshift
operatorS a(append-a) by
(SaF)(s) :=F(s·a).(13)
The associatedforward differenceoperator is
(∆aF)(s) := (S aF)(s)−F(s) =F(s·a)−F(s).
(14)
Equation (14) is the standard discrete/finite-
difference construction “difference = shift minus
identity” in a non-numeric domain.
A.2 Telescoping Identity
Finite differences compose along a path in the pre-
fix graph. Let u= (u 1, . . . , u m)∈Σmbe a
suffix and u<i= (u 1, . . . , u i−1). Then for any
F: Σ∗→Rk,
F(s·u)−F(s) =mX
i=1
F(s·u <i·ui)−F(s·u <i)
=mX
i=1(∆uiF)(s·u <i)
(15)
Equation (15) is the discrete analogue of the funda-
mental theorem of calculus: the total change equals
the sum of incremental changes along a path.
A.3 Instantiation for Decoder-Only
Transformers
Fix layer ℓand head h. For an input sequence x, let
v(ℓ,h)
i(x)∈Rdhdenote the pre-attention value vec-
tor at position i. For a fixed query q∈Σmand doc-
ument d∈Σn, define the document-conditionedvalue vector for query positiont:
F(ℓ,h)
t(d;q) :=v(ℓ,h)
n+t(d·q).(16)
The main paper’s controlled finite difference is ex-
actly the first-order difference between dand a
length-matched null prefix∅ n:
δv(ℓ,h)
t(d;q) =F(ℓ,h)
t(d;q)−F(ℓ,h)
t(∅n;q),(17)
and the representational shift tensor ∆(d, q) col-
lects these deltas across(ℓ, h, t).
B Ablations
We conducted ablation studies on our techniques to
identify the most effective variations and determine
which configurations yield the best results.
B.1 Representation and Optimization
Methods
Here we analyze alternative representation and op-
timization methods. Table 5 compares our learned
approach on shifts against: (a) learning on raw rep-
resentations without subtraction to see the effect of
“shifting”, and (b) three closed-form projections on
shifts to compare analytical methods against our
learned optimisation.
Shift vs. raw representations.The shift is
critical for calibration:while the direct approach
achieves comparable R@5(98.3vs97.8), its F1@0
drops by 6.4pp because raw representations lack
the centering that makes zero a natural threshold.
Closed-form baselines.We consider three ana-
lytical solutions:
1.Oracle alignment: Bℓ,h=1 Pµ⊤
ℓ,h/∥µℓ,h∥2,
whereµis the mean shift, so thatBµ=1 P.
2.Separation:solves a ridge regression B⊤=
(X⊤X+λI)−1X⊤Ywith targets 1P,0Pfor
relevant, irrelevant shifts.
3.Combined:adds an oracle constraint to the
objective: min∥BS irrel∥2subject to Bµ=1 P,
yielding a covariance-weighted projection.
All three preserve reasonable ranking (R@5 up
to 90.8) but fail at calibration (F1@0 ≤30.8).
These methods optimize alignment and separation
targets in projection space, but scoring uses cosine
similarity with 1P, which introduces a non-linear
normalization. As a result, pushing irrelevant pro-
jections toward 0Pdoes not yield negative scores—
it yields near-zero-norm vectors with noisy cosine
similarities. Our learned objective optimizes di-
rectly on thescores(cosine similarities), explicitly
12

Table 5: Shift vs. direct representations on 2WikiMQA.
The shift is essential for threshold calibration (F1@τ).
Method R@5 F1@τ
Learned on shifts (ours) 97.882.2
Learned on raw repr. (no shift)98.375.8
Closed-form oracle 75.3 17.9
Closed-form separation 90.8 23.1
Closed-form combined 86.8 30.8
Table 6: Extraction type ablation on 2WikiMQA. Value
vectors provide the best signal.
Signal R@5 F1@5
Value vectors (ours)98.3 66.3
Key vectors 97.8 66.0
Hidden states 96.5 65.0
pushing relevant scores above zero and irrelevant
scores below zero with a margin.
B.2 Choice of Internal Signal
We compare three internal LLM signals: value vec-
tors (our choice), key vectors, and hidden states.
Table 6 shows that value vectors achieve the best
performance, closely followed by key vectors. Hid-
den states perform worst, suggesting that the head-
level decomposition provides useful structure for
relevance detection.
B.3 Loss Components Ablation
Table 7 shows the effect of removing each loss com-
ponent. The calibration loss is the most critical: re-
moving it drops F1@ τby 8.3pp, as the model loses
its ability to anchor the decision boundary at zero.
Removing the margin loss causes a moderate 1.9-
point F1@ τdrop by weakening class separation.
Orthogonality regularization and oracle alignment
have minimal individual impact.
Table 7: Loss component ablation on 2WikiMQA vali-
dation set.∆is relative to the full model.
Configuration R@5 F1@τ∆R@5∆F1@τ
Full model 94.0 80.2 — —
−Calibration loss 88.2 71.9−5.8−8.3
−Margin loss 92.9 78.3−1.1−1.9
−Ortho. reg. 93.4 79.9−0.6−0.3
−Oracle alignment 93.3 80.0−0.7−0.2
−Norm constraint 93.6 79.4−0.4−0.8B.4 Training Details
Table 8: Training hyperparameters used across all ex-
periments.
Hyperparameter Value
Optimizer AdamW
Learning rate schedule Cosine annealing
Initial learning rate0.05
Final learning rate0.001
Weight decay10−3
Training precision Mixed precision
Maximum epochs120
Early stopping patience35
λcal 0.8
λalign 0.5
λortho 0.05
λfrob 0.1
Margin(m) 0.5
Projection dimension(P) 64
13