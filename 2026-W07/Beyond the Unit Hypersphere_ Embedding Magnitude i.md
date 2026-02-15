# Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning

**Authors**: Xincan Feng, Taro Watanabe

**Published**: 2026-02-09 21:53:23

**PDF URL**: [https://arxiv.org/pdf/2602.09229v1](https://arxiv.org/pdf/2602.09229v1)

## Abstract
Cosine similarity is prevalent in contrastive learning, yet it makes an implicit assumption: embedding magnitude is noise. Prior work occasionally found dot product and cosine similarity comparable, but left unanswered WHAT information magnitude carries, WHEN it helps, and HOW to leverage it. We conduct a systematic study through a $2 \times 2$ ablation that independently controls input-side and output-side normalization across text and vision models. Our findings reveal three key insights. First, in text retrieval, output (document) magnitude strongly correlates with relevance (Cohen's $d$ up to 1.80), yielding the largest gains on reasoning-intensive tasks. Second, input and output magnitudes serve asymmetric roles: output magnitude directly scales similarity scores while input magnitude modulates training dynamics. Third, magnitude learning benefits asymmetric tasks (text retrieval, RAG) but harms symmetric tasks (STS, text-image alignment). These findings establish a task symmetry principle: the choice between cosine and dot product depends on whether the task has distinct input roles, enabling cost-free improvements by simply removing an unnecessary constraint.

## Full Text


<!-- PDF content starts -->

Beyond the Unit Hypersphere:
On the Role of Embedding Magnitude in Contrastive Learning
Xincan Feng1Taro Watanabe1
Abstract
Cosine similarity is prevalent in contrastive learn-
ing, yet it makes an implicit assumption: embed-
ding magnitude is noise. Prior work occasionally
found dot product and cosine similarity compara-
ble, but left unansweredwhatinformation mag-
nitude carries,whenit helps, andhowto lever-
age it. We conduct a systematic study through a
2×2 ablation that independently controls input-
side and output-side normalization across text and
vision models. Our findings reveal three key in-
sights. First, in text retrieval, output (document)
magnitude strongly correlates withrelevance(Co-
hen’s dup to 1.80), yielding the largest gains on
reasoning-intensive tasks. Second, input and out-
put magnitudes serve asymmetric roles: output
magnitude directly scales similarity scores while
input magnitude modulates training dynamics.
Third, magnitude learning benefitsasymmetric
tasks (text retrieval, RAG) but harmssymmetric
tasks (STS, text-image alignment). These findings
establish atask symmetry principle: the choice be-
tween cosine and dot product depends on whether
the task has distinct input roles, enabling cost-free
improvements by simply removing an unneces-
sary constraint.
1. Introduction
Contrastive learning has become a foundational technique
for learning representations, with applications spanning im-
age representation (Chen et al., 2020; He et al., 2020),
sentence embeddings (Gao et al., 2021), text retrieval
(Karpukhin et al., 2020; Izacard et al., 2022), and text-image
alignment (Radford et al., 2021). A prevalent design choice
is cosine similarity, which normalizes embeddings to unit
Preliminary work. Under review by the International
Conference on Machine Learning (ICML). Do not distribute.
1Natural Language Processing Laboratory, Nara Institute
of Science and Technology, Japan.AUTHORERR: Missing
\icmlcorrespondingauthor.
Preprint. February 11, 2026.length before computing their inner product. This normal-
ization is often considered benign, as it stabilizes training
and bounds similarity scores.
However, cosine similarity makes a strongimplicit assump-
tion: by projecting all representations onto the unit hyper-
sphere Sn−1, it presumes that embedding magnitude carries
no task-relevant information. Geometrically, this reduces
representational capacity from nton−1 degrees of free-
dom, discarding magnitude entirely. The assumption that
“magnitude is noise” may be reasonable in some contexts,
but is itnecessary? Or is it merely a historical default?
Prior work touched on this question but left key gaps. DPR
(Karpukhin et al., 2020) found dot product comparable to
cosine and used it for simplicity but did not investigate
why; Steck et al. (2024) showed that cosine similarity of
learned embeddings can yield arbitrary results due to nor-
malization ambiguity; Oyama et al. (2023) showed word
embedding magnitude encodes information gain, but only
at the word level. Most recently, Liao et al. (2025) demon-
strated that CLIP’s pre-normalization image features exhibit
magnitude correlated with perceptual quality. However, this
information is discarded by L2 normalization, and the cor-
relation is emergent rather than learned. This raises our
central question:if we allow training to access magnitude,
can models learn to encode relevance in this degree of free-
dom?Critically, no prior work has isolated input-side versus
output-side magnitudes, nor established when magnitude
learning helps versus harms.
Our approach is deliberately minimal. We simply replace
cosine similarity with unnormalized dot product during fine-
tuning:
scos(q,d) =q⊤d
∥q∥ · ∥d∥−→s dot(q,d) =q⊤d.(1)
This introduces no new parameters and no new loss terms;
it merely removes an implicit constraint. This minimality is
intentional: it isolates the effect of the magnitude assump-
tion from confounding factors. For models with built-in
normalization layers, such as E5, the normalization layer
must be removed; this case reveals important architectural
prerequisites for magnitude learning.
We use text retrieval and RAG as the primary testbed, con-
1arXiv:2602.09229v1  [cs.LG]  9 Feb 2026

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
ducting systematic experiments across three BERT-based
retrievers (Contriever, RetroMAE, E5) on 30+ evaluation
datasets. We focus on fine-tuning (the dominant paradigm
in modern NLP) while also comparing with training from
scratch. From these experiments, we derive atask symmetry
principleand validate its generalization on STS and CLIP.
Our contributions are:
1.What information magnitude carries?While
prior work found dot product and cosine comparable
(Karpukhin et al., 2020) and saw no clear advantage,
we discover that magnitude carries task-relevant infor-
mation. Measuring Cohen’s dacross 30+ datasets, we
find that document magnitude strongly correlates with
relevance(up to d= 1.80 , averaging d= 1.22 on
reasoning-intensive benchmarks), directly improving
retrieval and RAG performance. Furthermore, input
and output magnitudes serve asymmetric roles: output
magnitude affects inference-time ranking while input
magnitude modulates training dynamics.
2.When magnitude helps?The largest gains occur
on challenging out-of-domain tasks (BRIGHT +5.33,
Multi-hop +6.79) where angular similarity alone is
insufficient. We find that magnitude learning is
architecture-dependent regardless of fine-tuning or
training from scratch: different models favor dif-
ferent normalization strategies. Furthermore, pre-
trained models facilitate magnitude learning: fine-
tuning yields larger and more consistent gains than
training from scratch, suggesting that pre-training pro-
vides representations that can be effectively reorga-
nized to exploit magnitude.
3.How to leverage magnitude?We propose a learnable
normalization framework ( γq, γd∈[0,1] ) that contin-
uously interpolates between cosine and dot product,
allowing models to discover their optimal normaliza-
tion strategy through gradient descent. Furthermore,
we establish atask symmetry principle: magnitude
learning benefitsasymmetrictasks (retrieval, RAG)
where inputs have distinct roles, but harmssymmetric
tasks (STS, text-image alignment) where inputs are
interchangeable. We validate this principle through
controlled experiments on STS and CLIP.
2. The Geometry of Similarity Functions
2.1. Background: Contrastive Learning with InfoNCE
Contrastive learning trains encoders by pulling together rep-
resentations of similar pairs while pushing apart dissimilar
ones. The InfoNCE loss (van den Oord et al., 2018) formal-izes this objective:
L=−logexp(s(q,d+)/τ)
exp(s(q,d+)/τ) +Pk
i=1exp(s(q,d−
i)/τ),
(2)
where qis the query embedding, dis the document embed-
ding, s(·,·) is a similarity function, and τis the temperature.
The choice of sdetermines what geometric structure the
model can learn.
2.2. Cosine Similarity: The Unit Hypersphere
Constraint
The standard approach uses cosine similarity:
scos(q,d) =q⊤d
∥q∥ · ∥d∥=ˆq⊤ˆd,(3)
where ˆq=q/∥q∥ andˆd=d/∥d∥ are unit-normalized vec-
tors. Throughout this paper, we use∥q∥and∥d∥to denote
the Euclidean norms (magnitudes) of query and document
embeddings respectively.
Geometric interpretation.Cosine similarity constrains
all representations to lie on the unit hypersphere Sn−1=
{x∈Rn:∥x∥= 1} . This imposes a strong geometric
prior: the only learnable structure isangularrelationships.
All points are equidistant from the origin, and similarity is
purely directional.
Implicit assumption.This constraint encodes an assump-
tion that is rarely stated explicitly:embedding magnitude
carries no task-relevant information. A confident repre-
sentation and an ambiguous one are treated identically, as
long as their directions match. Whether this assumption
is appropriate depends on the task, but it is an assumption
nonetheless.
2.3. Dot Product Similarity: Releasing the Constraint
We consider the unnormalized dot product:
sdot(q,d) =q⊤d=∥q∥ · ∥d∥ ·cosθ.(4)
Geometric interpretation.Dot product similarity per-
mits representations to occupy the full Rn, restoring magni-
tude as a learnable degree of freedom. The decomposition
above reveals that dot product can be viewed asmagnitude-
weighted angular similarity:
•A high-magnitude document boosts similarity with all
queries
•A high-magnitude query boosts similarity with all docu-
ments
Intuitively, magnitude may encode the confidence or
strength of a representation’s contribution to similarity.
2

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Document MagnitudeQuery Magnitudediscard preservediscard preserveˆq
ˆd
(a) Cosineθˆq
d
(b) QNorm
q
ˆd
(c) DNormq
d
(d) Dot
Figure 1.The 2×2 ablation framework. The dashed circle rep-
resents the unit sphere. Normalized vectors ( ˆv) lie on the sphere;
unnormalized vectors extend beyond. Rows: query magnitude
preserved or discarded. Columns: document magnitude preserved
or discarded.
2.4. A2×2Ablation Framework
As shown in Equation (3), cosine similarity involves two
independent normalization operations: one for the input
query ( q→ ˆq) and one for the output document ( d→ ˆd).
These operations can be applied independently, allowing
us to isolate their individual effects. We introduce two
intermediate variants:
Query-Only Normalization (QNorm).Normalize only
the query, preserving document magnitude:
sqnorm(q,d) = ˆq⊤d=∥d∥ ·cosθ.(5)
Document-Only Normalization (DNorm).Normalize
only the document, preserving query magnitude:
sdnorm(q,d) =q⊤ˆd=∥q∥ ·cosθ.(6)
Together with Cosine (both normalized) and Dot (neither
normalized), these form a complete 2×2 ablation (Figure 1).
This design allows us to isolate the contribution of each
magnitude component.
Inductive bias trade-off.Cosine imposes the strongest
constraint (unit sphere), partial normalization releases one
degree of freedom, and dot product permits full magni-
tude use. Our hypothesis is that pre-trained models already
possess sufficient inductive bias, making the unit norm con-
straint unnecessarily restrictive during fine-tuning.
Learnable Normalization: A Continuous Generalization.
The four variants above are discrete points in a continuousspace. We unify them through a learnable normalization
framework:
slearn(q,d) =q⊤
∥q∥γq·d
∥d∥γd, γ=σ(ˆγ)∈[0,1],(7)
whereˆγ q,ˆγdare learnable parameters andσis the sigmoid
function, ensuring γ∈[0,1] . This subsumes all four dis-
crete variants as special cases: (γq, γd) = (1,1) yields
Cosine, (0,0) yields Dot, (1,0) yields QNorm, and (0,1)
yields DNorm. By initializing at γ= 0.5 (the geometric
midpoint), we let the modeldiscoverthe optimal normaliza-
tion level through gradient descent, providing an empirical
test of which strategy the data prefers.
3. Experiments
3.1. Experimental Setup
Training Retrievers.We conduct experiments using three
pre-trained dense retrievers: Contriever (Izacard et al.,
2022), RetroMAE (Xiao et al., 2022), and E5 (Wang et al.,
2022). All three are BERT-based models that produce 768-
dimensional embeddings (see Appendix A for pre-training
details). We focus on fine-tuning (the dominant paradigm
in practice) rather than pre-training from scratch. E5 is
presented separately in Appendix J due to architectural con-
straints.
Training Data.We fine-tune on MS MARCO v1.1 QA
(Nguyen et al., 2016) (81,795 samples) (see Appendix Q for
details).
Evaluation Benchmarks.We evaluate on four bench-
mark categories using NDCG@10:MS MARCO Devand
TREC-DL 2019/2020(in-domain),BEIR(Thakur et al.,
2021) (14 diverse domains),BRIGHT(Su et al., 2025) (12
reasoning-intensive datasets), andMulti-hop(4 multi-hop
QA datasets). See Appendix Q for statistics.
Hyperparameters.We train for up to 100 epochs
with batch size 128, selecting checkpoints by validation
NDCG@10. Learning rates ( 2e-6 for Contriever, 5e-6 for
RetroMAE/E5) are determined via grid search on cosine
similarity training, as shown in Table 9 in Appendix, and
fixed across all similarity variants for fair comparison. Fol-
lowing standard practice (Karpukhin et al., 2020), we apply
a scaling factor α= 20 to all similarity variants to ensure
logits are in a reasonable range for softmax. All experiments
use deterministic settings; Contriever and RetroMAE report
mean±std across 3 seeds. See Appendix B and C for full
configuration and statistical significance analysis.
3

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 1.NDCG@10 for fine-tuning on MS MARCO 80K (mean ±std across 3 seeds). PT = pretrained (before fine-tuning). Numbers in
parentheses indicate the number of subsets averaged.Bold= best, underline = second best per model.
Contriever RetroMAE
Dataset PT Cosine Dot QNorm DNorm Learn PT Cosine Dot QNorm DNorm Learn
In-Domain
MSM-Dev 19.45 30.86±0.11 31.07±0.07 31.35±0.03 31.40±0.05 32.17±0.06 2.87 30.68±0.08 31.24±0.18 32.15±0.0932.92±0.1832.84±0.11
DL-19 42.85 56.65±0.34 56.69±0.2358.02±0.3855.41±0.40 57.76±0.10 14.05 56.11±0.27 56.82±0.84 57.51±0.42 59.95±2.01 60.43±1.04
DL-20 40.52 57.25±0.74 56.32±0.18 56.87±0.59 57.17±0.0458.48±0.32 9.00 56.85±0.61 57.83±0.86 58.11±0.73 59.20±0.82 59.69±0.90
Out-of-Domain
BEIR (14) 28.73 40.96±0.50 43.48±0.1344.01±0.2742.45±0.16 43.58±0.19 14.27 38.28±0.17 40.09±0.19 41.05±0.28 41.15±0.16 41.22±0.27
BRIGHT (12) 3.84 7.41±0.26 12.53±0.17 12.74±0.209.83±0.26 11.66±0.19 5.20 6.11±0.20 8.71±0.20 9.44±0.27 9.46±0.16 9.75±0.07
MHop (4) 39.70 51.37±0.8458.16±0.2057.38±0.15 57.54±0.16 57.17±0.07 18.45 50.67±0.44 52.93±0.37 54.77±0.3355.47±0.6055.16±0.44
0 5k 10k 15k 20k 25k 30k 35k0.880.90.920.94
Training StepsVal NDCG@10Contriever
0 5k 10k 15k 20k 25k 30k 35k
Training StepsRetroMAE
Cosine Dot QNorm DNorm Learnable
Figure 2.Step-matched val NDCG@10 comparison during training. For both models, the Learnable variant consistently performs near the
top, with validation curves closely following DNorm throughout training.
3.2. Results
Table 1 presents the retrieval performance (NDCG@10) of
different similarity functions when fine-tuning Contriever
and RetroMAE on MS MARCO v1.1 QA. Complete re-
sults including Recall@100 and MRR@10 are provided in
Appendix D.
Across both models, all magnitude-aware variants outper-
form Cosine on all benchmarks, with substantial gains on
reasoning-intensive tasks: BRIGHT (+72% for Contriever
QNorm) and Multi-hop (+13% for Contriever Dot). In-
terestingly, the optimal strategy is model-dependent: Con-
triever favors QNorm (preserving document magnitude),
while RetroMAE favors DNorm (preserving query magni-
tude). The learnable normalization provides a robust default,
achieving competitive performance without requiring prior
knowledge about model characteristics. These results raise
three questions that guide our analysis:whatinformation
magnitude carries,whenit helps, andhowto leverage it.
4. What information magnitude carries?
We analyze what information magnitude carries, revealing
that document and query magnitudes serve fundamentally
different roles.4.1. Document Magnitude Encodes Relevance
Measuring the magnitude-relevance association.We
use Cohen’s dto quantify the magnitude difference between
relevant and irrelevant documents:
d=µrel−µ irrel
σpooled,(8)
where µrelandµirrelare the mean magnitudes of relevant and
irrelevant documents respectively, and σpooled is the pooled
standard deviation. Positive dindicates relevant documents
have larger magnitudes.
Theoretical analysis.The 2×2 framework (Section 2.4)
decomposes the dot product sdot(q,d) =∥q∥ · ∥d∥ ·cosθ
into three learnable components. Each similarity function
retains a different subset. The following proposition shows
that atinference time, only one component, document mag-
nitude, can alter ranking.
Proposition 4.1(Ranking Equivalence).For a fixed query
q, onlydocumentmagnitude can alter document rankings.
Formally, let πsdenote the ranking (permutation) of a docu-
ment setDsorted by decreasings(q,d). Then:
1.π cos=π dnorm : both rank documents bycosθalone.
4

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning∆%Contriever:r= 0.57,p <0.001, significant
Cohen’sd(mean±std across 3 seeds)∆%RetroMAE:r= 0.68,p <0.001(excl. pony)0100200
−1−0.5 0 0.5 1 1.5 20100200
In-Domain BEIR BRIGHT MultiHop Regression
Figure 3.Per-dataset Cohen’s dvs.∆% for Contriever (top) and
RetroMAE (bottom, excluding bright-pony). Cohen’s dvalues are
averaged across 3 seeds (0, 42, 1337) with horizontal error bars
showing ±std. Each point represents one dataset: blue squares
= In-Domain (TREC-DL), green circles = BEIR, red triangles =
BRIGHT, orange pentagons = Multi-hop. Background shading
indicates Cohen’s deffect size interpretation: large (|d| ≥0.8 ),
medium (0.5≤ |d|<0.8 ),small (0.2≤ |d|<0.5 ), and
negligible ( |d|<0.2 , white). Dashed lines show regression fits:
Contriever ( r= 0.57 ,p <0.001 ), RetroMAE ( r= 0.68 ,p <
0.001 ). The bright-pony dataset is excluded from the RetroMAE
plot due to its anomalous ∆% = 512.5 caused by an extremely
low Cosine baseline (see text for analysis). See Table 19 for per-
dataset values.
2.π qnorm =π dot: both rank documents by∥d∥cosθ.
Query magnitude ∥q∥is a positive constant for a fixed query
that scales all scores uniformly, thus never affecting rank-
ings.
The proof follows from the observation that ∥q∥is a positive
constant for fixed query (see Appendix H.1).
Empirical verification.Figure 3 presents per-dataset cor-
relation analysis across evaluation datasets.1For Contriever,
per-dataset analysis reveals a significant positive correlation
(r= 0.57 ,p <0.001 ), with BRIGHT datasets clustering in
the upper-right (high Cohen’s dand high ∆%), consistent
withmagnitude-aware training provides larger bene-
1NovelHopQA is excluded because its corpus only contains
relevant documents.fits. RetroMAE initially shows no correlation ( r= 0.08 ,
p= 0.62 ), but the bright-pony dataset is a clear outlier
(∆% = 512.5 ). Excluding it reveals a significant positive
correlation ( r= 0.68 ,p <0.001 ), comparable to Con-
triever. The outlier arises from dataset-specific character-
istics (see Appendix L). See Table 19 in the Appendix for
complete values.
4.2. Query vs Document: Asymmetric Roles
Comparing QNorm (preserves document magnitude) and
DNorm (preserves query magnitude) reveals model-
dependent patterns:Contrieverfavors QNorm, while
RetroMAEfavors DNorm. The source of this difference re-
mains unclear, though it may relate to different pre-training
objectives.
Proposition 4.1 established that query magnitude cannot
affect inference-time ranking. Yet RetroMAE’s best variant
is DNorm. How can ∥q∥help if it does not affect ranking?
The answer lies in training-time gradient structure.
Proposition 4.2(Gradient Asymmetry).Under InfoNCE
loss with DNorm similarity sdnorm(q,d) =∥q∥cosθ : (1)
effective temperature becomes τeff=τ/∥q∥ , where high
∥q∥ sharpens the distribution; (2) gradient magnitude is
proportional to ∥q∥; (3) at inference, πdnorm =π cos, so
benefits are realized during training. See Appendix H.4 for
derivations.
Empirical observations.DNorm increases query CV 6 ×
for RetroMAE compared to Dot (Appendix P), consistent
with Proposition 4.2’s prediction that query magnitude mod-
ulates training dynamics. For RetroMAE, DNorm outper-
forms QNorm consistently (+0.77 MSM-Dev, +0.10 BEIR,
+0.02 BRIGHT, +0.70 MHop), while Contriever shows the
opposite pattern. The optimal choice between QNorm and
DNorm is model-dependent.
5. When magnitude helps?
Not all settings benefit equally from magnitude learning. We
identify three key conditions: task difficulty, pre-training,
and architectural compatibility.
5.1. Benefits Scale with Task Difficulty
The striking difference between BRIGHT (mean Cohen’s
d= 1.22 , reaching d= 1.80 ) and in-domain tasks (mean
d≈0 ) suggests that standard retrieval shows minimal
magnitude–relevance correlation, while reasoning tasks
show strong correlation. This pattern aligns with the obser-
vation that the largest performance gains occur on reasoning-
intensive benchmarks where angular similarity alone may
be insufficient.
5

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 2.NDCG@10 for training from scratch (Contriever archi-
tecture, MS MARCO 80K, mean ±std across 3 seeds). Cosine
dominates on most benchmarks, but DNorm matches or exceeds
Cosine on reasoning tasks (BRIGHT). Dot and QNorm consistently
perform worst.Bold= best, underline = second best.
Dataset Cosine Dot QNorm DNorm Learn
In-Domain
MSM-Dev17.66±0.0711.07±0.02 9.58±0.02 15.19±0.12 11.97±0.06
DL-1938.45±2.2224.27±1.42 20.58±1.27 29.78±0.56 24.59±2.35
DL-2037.98±1.7824.93±0.72 22.31±0.57 33.55±0.87 27.43±0.82
Out-of-Domain
BEIR (14)24.38±0.5413.37±0.51 15.46±0.53 21.05±0.37 17.91±0.27
BRIGHT (12) 3.09±0.18 0.31±0.08 0.32±0.083.77±0.231.15±0.04
MHop (4)27.87±1.1018.29±0.11 20.26±0.51 27.72±0.66 22.71±0.27
Table 3.Cohen’s dfor document magnitude: Scratch vs Fine-tuned
(Contriever architecture, 3-seed average). Positive dindicates rele-
vant documents have larger magnitude. Background shading indi-
cates effect size: large (|d| ≥0.8 ),medium (0.5≤ |d|<0.8 ),
small (0.2≤ |d|<0.5 ).†Raw embeddings before inference-
time normalization.∗Excludes NovelHopQA (all documents are
relevant).
Scratch Fine-tuned
Category Cos†Dot QNorm Learn Cos†Dot QNorm Learn
TREC-DL (2) +0.15−0.16−0.13−0.17 +0.42−0.22−0.12−0.07
BEIR (14) −0.03+0.06−0.02−0.02 −0.01−0.00+0.01+0.00
BRIGHT (12) −1.00 −0.40 −0.71 −0.81 +0.36 +1.01 +1.02 +1.06
Multi-Hop (3)∗−0.55+0.02 −0.37 −0.51 −0.08 +0.50 +0.50 +0.46
All (31) −0.33 −0.09 −0.24 −0.28 +0.11 +0.30 +0.32 +0.32
5.2. Pre-training Facilitates Magnitude Learning
To isolate the role of pre-trained representations, we train
randomly initializedmodels from scratch using identical
settings. Table 2 reveals again thatquery and document
magnitudes serve fundamentally different roles: DNorm
outperforms Cosine on BRIGHT (3.77 vs 3.09) by preserv-
ing query magnitude for training dynamics, while Dot and
QNorm fail (0.31, 0.32). In addition,pre-training facil-
itates magnitude learning: Cohen’s danalysis (Table 3)
shows that scratch training yieldsnegative dfor Dot/QNorm
on BRIGHT (irrelevant documents have larger magnitude),
opposite to fine-tuning where d >1.0 (relevant documents
have larger magnitude). This sign reversal explains why
Dot/QNorm fail from scratch but succeed with pre-training.
5.3. Architectural Compatibility: E5
E5’s built-in normalization layer makes it incompatible with
magnitude-aware training without architectural modifica-
tion. Removing this layer enables magnitude learning but
causesmagnitude collapse: when trained with dot product
loss, E5’s cosine similarity collapses from 0.94 to 0.23 while
dot product reaches 0.94, indicating complete abandonment
of directional information (Appendix N). This demonstrates
thatarchitectural compatibility is essentialfor magnitude
learning. Details and extended experiments (500K samples)
in Appendix E.0 5k 10k 15k 20k 25k 30k 35k0.4980.4990.50.5010.5020.503
Init=0.5
Training Stepsγ
Contr.γ q Retro.γ q
Contr.γ d Retro.γ d
Figure 4.Learned normalization strengths γq, γdover training.
Contriever drifts toward Dot ( γ <0.5 ), while RetroMAE drifts
toward Cosine (γ >0.5).
6. How to leverage magnitude?
We now translate our findings into practical guidance, first
exploring learnable normalization and RAG evaluation
within text tasks, then establishing a task symmetry principle
and validating it on STS and CLIP.
6.1. Learnable Normalization as Safe Default
Can the model discover its optimal normalization strategy?
Using Equation (7) with γq, γdinitialized at 0.5, Figure 4
shows that Contriever drifts toward Dot ( γ∗≈0.499 ) while
RetroMAE drifts toward Cosine ( γ∗≈0.502 ) (see Ap-
pendix T.3 for gradient dynamics). Figure 2 shows that the
learnable variant consistently performs near the top for both
models, providing a safe default without requiring prior
knowledge about model characteristics.
CV as a diagnostic signal.Figure 5 suggests that query
magnitude CV may predict optimal normalization choice.
We plot ∆CV (query CV ratio: DNorm/Dot) against ∆Perf
(DNorm −QNorm). The two models formcompletely sep-
arated clusters: RetroMAE ( ∆CV: 4–9 ×) exploits query
magnitude and prefers DNorm ( ∆Perf>0), while Con-
triever ( ∆CV: 0.5–2 ×) does not and prefers QNorm ( ∆Perf
<0). See Appendix M for details.
6.2. End-to-End RAG Evaluation
To verify retrieval improvements transfer to downstream
tasks, we conduct RAG experiments on Natural Questions,
HotpotQA, and TriviaQA using Contriever with Flan-T5-
Large ( k= 5 ). Table 4 shows that QNorm consistently
outperforms Cosine (+13.5% on NQ, +20.2% on HotpotQA,
+24.5% on TriviaQA), confirming retrieval gains translate
to QA accuracy. Learnable Normalization performs com-
parably to Dot but does not match QNorm, demonstrating
6

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
0 1 2 3 4 5 6 7 8−15−10−50510
∆CV (DNorm CV / Dot CV)∆Perf (DNorm−QNorm)
Contriever (∆CV: 0.5–2) RetroMAE (∆CV: 4–7)
Contriever mean RetroMAE mean
Figure 5. ∆CV vs. ∆Perf for Contriever and RetroMAE on 39
datasets (3-seed averaged). The two models form clearly separated
clusters: Contriever (blue, ∆CV<2) benefits from QNorm ( ∆Perf
<0), while RetroMAE (red, ∆CV>4) benefits from DNorm
(∆Perf>0). Large markers indicate cluster means; error bars
show∆CV std across datasets.
Table 4.End-to-end RAG evaluation on open-domain QA. Re-
triever: Contriever models from Table 1 (fine-tuned on MS
MARCO 80K with Cosine, Dot, QNorm, DNorm, and Learn).
Reader: Flan-T5-Large. QNorm achieves the largest gains over Co-
sine, with ∆up to +7.9 EM (+24%) on TriviaQA. Learn performs
comparably to Dot, demonstrating that learned normalization can
match but not exceed asymmetric normalization.
NQ(3.5K) HotpotQA(7.4K) TriviaQA(11.3K)
Similarity EM F1 EM F1 EM F1
Cosine 23.0 31.9 27.2 35.9 32.3 38.7
Dot 24.5 33.8 32.9 42.7 39.0 46.3
QNorm 26.1 35.4 32.7 42.4 40.2 47.5
DNorm 22.8 31.5 31.5 41.0 37.7 44.6
Learn 25.0 34.0 32.6 42.4 39.4 46.5
∆(QNorm−Cos) +3.1 +3.5 +5.5 +6.5 +7.9 +8.8
Number in parentheses: test set size. Bold: best result per dataset.
∆row: performance gain of QNorm over Cosine.
robust performance without prior knowledge of the optimal
strategy. Details in Appendix I.
6.3. The Task Symmetry Principle
From the above explorations, we establish atask symmetry
principle: asymmetric tasks (retrieval, RAG) benefit from
magnitude learning, while symmetric tasks (STS, clustering)
require sim(a, b) =sim(b, a) , making partial normalization
mathematically incompatible.
Corollary 6.1(Task Symmetry Constraint).For tasks re-
quiring symmetric similarity s(a,b) =s(b,a) , partial nor-
malization is incompatible: sqnorm(a,b) =∥b∥cosθ̸=
∥a∥cosθ=s qnorm(b,a) unless ∥a∥=∥b∥ for all pairs.
Only Cosine and Dot preserve symmetry; QNorm and
DNorm do not.Table 5.STS-B test set Spearman correlation. Unlike retrieval,
magnitude provides no benefit; asymmetric normalization severely
degrades performance.
Model Cosine Dot S1Norm S2Norm
Contriever0.7880.784 0.413 0.371
RetroMAE 0.7490.7500.314 0.328 NormalizationLoss Direction
Symmetric I2T only T2I only
Image Norm
Text Norm
No NormNorm
d≈0
NoNorm
d≈0ImgNorm
d=1.83
—
NoNorm
d=0.58—
TxtNorm
d=2.11
NoNorm
d=1.31
Figure 6.CLIP pre-training framework. Gray: symmetric loss
yields d≈0 . Blue/red: asymmetric loss enables magnitude
learning on non-query side.
Table 6.CLIP pre-training results on MS-COCO. Shaded cells
indicate the direction(s) trained by the loss function. Symmetric
loss yields d≈0 ; asymmetric loss enables magnitude learning but
sacrifices bidirectional capability.
Cohen’sdCosine R@1 Dot R@1∆
ConfigImage Text I→T T→I I→T T→I(Dot−Cos)
Norm (Sym) −0.55 −0.30 38.4 28.0 32.8 26.3−5.6
NoNorm (Sym) −0.65 −0.06 23.4 19.1 25.1 19.5+1.7
ImgNorm-I2T 5.72 1.83 30.5 0.3 32.10.3 +1.6
NoNorm-I2T−0.49 0.58 18.0 12.4 21.29.6 +3.2
TxtNorm-T2I 2.1110.89 1.9 17.6 0.2 23.4+5.8
NoNorm-T2I 1.311.21 8.6 13.5 6.9 16.7+3.2
6.4. Verification on STS
We test whether magnitude benefits persist in symmetric
tasks using STS Benchmark. Table 5 shows that unlike re-
trieval where magnitude-aware variants outperform Cosine,
symmetric STS shows Cosine ≈Dot, and asymmetric nor-
malizationcatastrophically fails(40–45 point degradation),
exactly as Corollary 6.1 predicts.
6.5. Verification on CLIP
To test generalization beyond text, we first analyze the pre-
trained CLIP (ViT-B/32) on MS-COCO at inference time:
Cohen’s d≈0 for both modalities, and switching to Dot
hurts performance because CLIP was trained with normal-
ized embeddings. To understand why, we conduct controlled
CLIP pre-training experiments from scratch (Figure 6, Ta-
ble 6). The results reveal thatsymmetric loss prevents
magnitude learning( d≈0 even without normalization),
whileasymmetric loss enables it(positive don output
side). However, this comes at a cost: magnitude learning
requires sacrificing bidirectional retrieval capability. Details
in Appendix F.
7

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
7. Related Work
Dense Retrieval and Contrastive Learning.DPR
(Karpukhin et al., 2020) compared similarity functions on its
training domain (Natural Questions) and related Wikipedia-
based QA datasets, finding dot product and cosine similarity
comparable and choosing dot product only for simplicity.
Most subsequent dense retrievers (Izacard et al., 2022; Xiao
et al., 2022; Wang et al., 2022) and contrastive methods
(Chen et al., 2020; He et al., 2020; Gao et al., 2021; Radford
et al., 2021) adopted cosine similarity, treating magnitude
as noise. LaBSE (Feng et al., 2022) uses dot product with
additive margin softmax but applies L2 normalization, mak-
ing it equivalent to cosine. We show that magnitude-aware
methods can outperform cosine on both in-domain and out-
of-domain tasks, with particularly large gains on diverse
out-of-domain benchmarks (BEIR, BRIGHT, Multi-hop;
Section 5.1).
Magnitude in Neural Representations.Beyond embed-
dings, magnitude carries information across neural network
components. At the word level, Oyama et al. (2023) showed
embedding norms encode information gain. In attention
mechanisms, Kobayashi et al. (2020) showed that vector
norms reveal patterns invisible to attention weights alone,
and Guo et al. (2024) demonstrated that value vector norms
are critical for token importance in KV cache reduction. For
vision-language models, Liao et al. (2025) found CLIP im-
age embedding magnitude correlates with perceptual quality.
These works analyze magnitude post-hoc; we show models
canlearnto encode relevance in magnitude when training
allows it.
Similarity Functions in Metric Learning.The choice
between cosine and Euclidean distance has been studied ex-
tensively in metric learning (Musgrave et al., 2020). Wang
et al. (2017) showed that L2 normalization improves face
verification by removing intra-class variance in magnitude.
Ranjan et al. (2017) proposed L2-constrained softmax for
better-calibrated features. However, these works focus on
symmetric classification tasks where normalization is ben-
eficial. Our work reveals that the optimal choice depends
ontask symmetry: normalization helps symmetric tasks but
limits representation learning for asymmetric ones where
magnitude can encode task-relevant information.
8. Conclusion
We investigated the assumption that embedding magnitude
is noise. By replacing cosine with dot product, we discov-
ered that models learn to encode relevance in magnitude
(Cohen’s dup to 1.80), with input and output magnitudes
serving asymmetric roles. The task symmetry principle
provides actionable guidance: magnitude benefits asym-metric tasks but harms symmetric ones. RAG experiments
confirm practical value (+24% QA accuracy), while STS
and CLIP experiments validate the generalization of our
findings beyond text retrieval. Limitations are discussed in
Appendix K.
Impact Statement
This paper presents work whose goal is to advance the field
of Machine Learning. There are many potential societal
consequences of our work, none which we feel must be
specifically highlighted here.
References
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A
simple framework for contrastive learning of visual rep-
resentations. InInternational Conference on Machine
Learning, pp. 1597–1607, 2020.
Feng, F., Yang, Y ., Cer, D., Arivazhagan, N., and Wang, W.
Language-agnostic BERT sentence embedding. InPro-
ceedings of the 60th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers),
pp. 878–891, 2022.
Gao, T., Yao, X., and Chen, D. SimCSE: Simple contrastive
learning of sentence embeddings. InProceedings of the
2021 Conference on Empirical Methods in Natural Lan-
guage Processing, pp. 6894–6910, 2021.
Guo, Z., Kamigaito, H., and Watanabe, T. Attention score
is not all you need for token importance indicator in KV
cache reduction: Value also matters. InProceedings of
the 2024 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pp. 21131–21146, 2024.
He, K., Fan, H., Wu, Y ., Xie, S., and Girshick, R. Mo-
mentum contrast for unsupervised visual representation
learning. InProceedings of the IEEE/CVF Conference
on Computer Vision and Pattern Recognition, pp. 9729–
9738, 2020.
Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski,
P., Grave, A., and Joulin, A. Unsupervised dense infor-
mation retrieval with contrastive learning.Transactions
on Machine Learning Research, 2022.
Karpukhin, V ., O ˘guz, B., Min, S., Lewis, P., Wu, L., Edunov,
S., Chen, D., and Yih, W.-t. Dense passage retrieval
for open-domain question answering. InProceedings of
the 2020 Conference on Empirical Methods in Natural
Language Processing, pp. 6769–6781, 2020.
Kobayashi, G., Kuribayashi, T., Yokoi, S., and Inui, K.
Attention is not only a weight: Analyzing transformers
with vector norms. InProceedings of the 2020 Conference
8

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
on Empirical Methods in Natural Language Processing
(EMNLP), pp. 7057–7075, 2020.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive NLP tasks. InAdvances in Neural Information
Processing Systems, volume 33, pp. 9459–9474, 2020.
Liao, Z., Wu, D., Shi, Z., Mai, S., Zhu, H., Zhu, L., Jiang,
Y ., and Chen, B. Beyond cosine similarity: Magnitude-
aware CLIP for no-reference image quality assessment.
InProceedings of the AAAI Conference on Artificial In-
telligence, 2025.
Musgrave, K., Belongie, S., and Lim, S.-N. A metric learn-
ing reality check. InEuropean Conference on Computer
Vision, pp. 681–699, 2020.
Nguyen, T., Rosenberg, M., Song, X., Gao, J., Tiwary, S.,
Majumder, R., and Deng, L. MS MARCO: A human
generated machine reading comprehension dataset. In
Proceedings of the Workshop on Cognitive Computation:
Integrating Neural and Symbolic Approaches, 2016.
Oyama, M., Yokoi, S., and Shimodaira, H. Norm of word
embedding encodes information gain. InProceedings of
the 2023 Conference on Empirical Methods in Natural
Language Processing, pp. 2120–2141, 2023.
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G.,
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J.,
et al. Learning transferable visual models from natural
language supervision. InInternational Conference on
Machine Learning, pp. 8748–8763, 2021.
Ranjan, R., Castillo, C. D., and Chellappa, R. L2-
constrained softmax loss for discriminative face verifica-
tion.arXiv preprint arXiv:1703.09507, 2017.
Steck, H., Ekanadham, C., and Kallus, N. Is cosine-
similarity of embeddings really about similarity? In
Companion Proceedings of the ACM on Web Conference
2024, pp. 887–890, 2024.
Su, H., Yen, H., Xia, M., Shi, W., Muennighoff, N., Wang,
H.-y., Liu, H., Shi, Q., Siegel, Z. S., Tang, M., Sun, R.,
Yoon, J., Arik, S. O., Chen, D., and Yu, T. BRIGHT: A re-
alistic and challenging benchmark for reasoning-intensive
retrieval. InInternational Conference on Learning Rep-
resentations, 2025.
Thakur, N., Reimers, N., R ¨uckl´e, A., Srivastava, A., and
Gurevych, I. BEIR: A heterogeneous benchmark for
zero-shot evaluation of information retrieval models. In
Thirty-fifth Conference on Neural Information Process-
ing Systems Datasets and Benchmarks Track (Round 2),
2021.van den Oord, A., Li, Y ., and Vinyals, O. Representation
learning with contrastive predictive coding. InAdvances
in Neural Information Processing Systems, 2018.
Wang, F., Xiang, X., Cheng, J., and Yuille, A. L. NormFace:
L2 hypersphere embedding for face verification. InPro-
ceedings of the 25th ACM International Conference on
Multimedia, pp. 1041–1049, 2017.
Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L.,
Jiang, D., Majumder, R., and Wei, F. Text embeddings
by weakly-supervised contrastive pre-training.arXiv
preprint arXiv:2212.03533, 2022.
Wei, H., Xie, R., Cheng, H., Feng, L., An, B., and Li,
Y . Mitigating neural network overconfidence with logit
normalization. InInternational Conference on Machine
Learning, pp. 23631–23644, 2022.
Xiao, S., Liu, Z., Shao, Y ., and Cao, Z. RetroMAE: Pre-
training retrieval-oriented language models via masked
auto-encoder.arXiv preprint arXiv:2205.12035, 2022.
9

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
A. Pre-trained Model Details
Table 7 summarizes the pre-training details of the three retrievers used in our experiments.
Table 7.Pre-training Details of Dense Retrievers Used in This Study. MAE = Masked Auto-Encoder. InfoNCE = Noise Contrastive
Estimation loss for contrastive learning. Table information is collected from original papers and official sources (e.g., Hugging Face). For
unsupervised/weakly supervised stages, only document sources are shown; query synthesis methods are omitted.
Contriever RetroMAE E5
Checkpoint facebook/contriever Shitao/RetroMAE intfloat/e5-base-unsupervised
Backbone google-bert/bert-base-uncased google-bert/bert-base-uncased google-bert/bert-base-uncased
Model Size 439M 438M 877M
Pre-training Loss InfoNCE MAE InfoNCE
Pre-training Data Wikipedia, CCNet Wikipedia, BookCorpus CCPairs, including Wikipedia, Reddit, Common Crawl (web pages
from MS-MARCO document ranking corpus included), Stackexchange,
S2ORC, News, SimpleWiki, GooAQ, WikiHow, Yahoo Answers
A key distinction among these models is the composition of their pre-training data. Contriever’s pre-training uses Wikipedia
and CCNet, while RetroMAE uses Wikipedia and BookCorpus. In contrast, E5’s pre-training corpus (CCPairs) explicitly
includes web pages from the MS-MARCO document ranking corpus through Common Crawl. This data overlap has
important implications for our fine-tuning experiments, as discussed in Section 3.
B. Training Configuration
To ensure fair comparison and reproducibility, all BERT-based retriever models in our experiments were trained using a
unified configuration. This section details the hyperparameters and settings used.
B.1. General Training Hyperparameters
Table 8 summarizes the training hyperparameters for BERT-based models.
Table 8.Training Hyperparameters for BERT-based Models (Contriever, RetroMAE, E5)
Hyperparameter Value
Training data MS MARCO 80K
Training samples 81,795
Number of GPUs 2 (data parallel)
Per-device batch size 128
Effective batch size 256 (128×2 GPUs)
Number of epochs 100
Learning rate (Contriever)2×10−6
Learning rate (RetroMAE, E5)5×10−6
Warmup ratio 0
Weight decay 0.01
Max gradient norm 1.0
LR scheduler Cosine decay
AdamW Optimizer
β1 0.9
β2 0.98
ϵ1×10−8
Multi-GPU Training.All experiments use distributed data parallel training across 2 GPUs via Hugging Face Accelerate.
The effective batch size is 128×2 = 256 samples per optimization step. Training steps are computed based on the effective
batch size: steps per epoch =⌈81795/256⌉= 320 , and 100 epochs corresponds to approximately 32,000 steps. Checkpoints
are saved and evaluated every 500 steps (approximately 1.6 epochs).
10

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Learning Rate Selection.We conduct learning rate sweeps using cosine similarity training to determine the optimal
learning rate for each model. Learning rates are selected from {2e-7,5e-7,1e-6,2e-6,5e-6,1e-5} . Table 9 shows the results.
The selected learning rates are then fixed across all similarity function variants for fair comparison.
Table 9.Learning Rate Selection via Grid Search (Cosine Similarity Training)
Model Best LR Val NDCG@10
Contriever2×10−692.87
RetroMAE5×10−691.87
E55×10−693.97
B.2. Evaluation Configuration
Table 10 summarizes the evaluation settings used during training and final evaluation.
Table 10.Evaluation Configuration
Setting Value
Evaluation batch size 64
Similarity function Corresponding to loss function
Primary metric NDCG@10
Checkpoint save interval 500 steps
Evaluation interval 500 steps
Note that each model variant is evaluated using its corresponding similarity function: Cosine-trained models use cosine
similarity, Dot-trained models use dot product, QNorm-trained models use query-normalized dot product, DNorm-trained
models use document-normalized dot product, and Learnable-trained models use the learnedγ-interpolated similarity.
C. Training Convergence Analysis
To ensure reliable evaluation results, we train all models for sufficient epochs until convergence and select the best checkpoint
based on validation performance.
C.1. Convergence Criteria
We consider a model converged when both conditions are met: (1) the standard deviation of NDCG@10 over the last 5
evaluation points is less than 0.002, and (2) the difference between the best and final NDCG@10 is less than 0.02. We
evaluate validation performance every 500 training steps.
C.2. Convergence Results
Table 11 summarizes the convergence dynamics for all three retrievers trained on MS MARCO 80K. We report both the first
convergence point (when validation performance first stabilizes) and the best checkpoint performance, showing that models
continue to improve after initial convergence.
All fifteen model-loss combinations achieved stable convergence. Models first satisfy convergence criteria within 8–20
epochs (2,500–6,500 steps), but continue to improve by 0.13–1.49 percentage points before reaching their best performance.
This justifies our choice of 100 epochs as a conservative upper bound.
C.3. Convergence Curves
Figure 7 shows the NDCG@10 progression during training for all model-loss combinations.
The convergence curves reveal consistent patterns across all three retrievers. All variants show rapid initial improvement
within the first 5,000–10,000 steps, followed by gradual convergence with performance stabilizing after approximately
11

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 11.Convergence Analysis for All Retrievers on MS MARCO 80K (seed=0). “First Conv.” indicates when the model first satisfies
convergence criteria (std of last 5 eval points <0.002 and best-current diff <0.02 ). “Gain” shows NDCG@10 improvement from first
convergence to best checkpoint. Steps per epoch≈320.
First Convergence Best Checkpoint
Model Loss Step Epoch NDCG@10 NDCG@10 Epoch Gain
ContrieverCosine 3,500 11.0 92.00 92.82 56.3 +0.82
Dot 3,500 11.0 92.49 93.50 75.1 +1.01
QNorm 5,000 15.6 90.63 91.70 98.6 +1.07
DNorm 4,000 12.5 92.57 93.57 84.5 +1.00
Learnable 5,000 15.6 92.97 93.58 82.8 +0.61
RetroMAECosine 3,500 11.0 91.07 91.88 42.3 +0.81
Dot 6,500 20.3 91.88 93.05 89.2 +1.17
QNorm 3,000 9.4 90.11 91.26 53.2 +1.15
DNorm 4,000 12.5 91.90 93.08 68.9 +1.18
Learnable 4,500 14.1 92.04 93.09 85.9 +1.05
E5Cosine 2,500 7.8 93.91 94.04 48.5 +0.13
Dot 2,500 7.8 92.94 94.43 90.8 +1.49
QNorm 4,000 12.5 93.95 94.62 67.3 +0.67
DNorm 3,000 9.4 94.34 94.80 70.4 +0.46
Learnable 4,000 12.5 94.23 94.77 68.8 +0.54
05k10k 15k 20k 25k 30k 35k0.880.90.920.940.96
Training StepsVal NDCG@10Contriever
05k10k 15k 20k 25k 30k 35k
Training StepsRetroMAE
05k10k 15k 20k 25k 30k 35k
Training StepsE5
Cosine Dot QNorm DNorm Learnable
Figure 7.Validation NDCG@10 during training for Contriever, RetroMAE, and E5 with different similarity functions (seed=0). All
models demonstrate stable convergence across all loss function variants.
15,000–25,000 steps. For Contriever and RetroMAE, the Dot, DNorm, and Learnable variants consistently track above the
Cosine baseline throughout training, while QNorm converges to lower performance. For E5, all five variants converge to
similar high performance on the validation set, with magnitude-aware variants slightly outperforming Cosine.
Key observations.
•Early convergence: All models first satisfy convergence criteria within 8–20 epochs (approximately 2,500–6,500 steps),
indicating rapid stabilization of training dynamics.
•Continued improvement: Despite meeting convergence criteria early, models continue to improve by 0.13–1.49
percentage points in NDCG@10 before reaching their best performance. This “post-convergence gain” represents
meaningful improvement that would be missed by early stopping at first convergence.
•Model-specific patterns: E5 converges fastest (2,500–4,000 steps, 8–12 epochs) but shows variable post-convergence
gains. Notably, E5-Dot shows the largest gain (+1.49%), suggesting that magnitude-aware training requires more
optimization for E5’s architecture. Contriever and RetroMAE show more consistent post-convergence gains around
+1%.
•Implications for training budget: These results suggest that training for approximately 60–90 epochs (rather than
stopping at 10–20 epochs when first convergence is detected) captures important gradual improvements. This justifies
our choice of 100 epochs as a conservative upper bound that ensures all models reach their optimal performance.
12

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
D. Complete Evaluation Results
This section presents the complete evaluation results with all three metrics: NDCG@10 (N), Recall@100 (R), and MRR@10
(M). We compare six model variants: Pretrained (no fine-tuning), and five fine-tuned methods (Cosine, Dot, QNorm, DNorm,
and Learnable). The best result among the five fine-tuned models is shown inbold, and the second-best is underlined . These
tables complement the main results (Tables 1–16) which only report NDCG@10.
D.1. Contriever Results
Table 12 presents the complete results for Contriever across all evaluation benchmarks.
Table 12.Complete Evaluation Results for Contriever (mean of seeds 0, 42, 1337). Comparing Pretrained and five fine-tuned methods
(Cosine, Dot, QNorm, DNorm, Learnable). N = NDCG@10, R = Recall@100, M = MRR@10.Bold= best among fine-tuned, underline
= second best.
Pretrained Cosine Dot QNorm DNorm Learnable
Dataset N R M N R M N R M N R M N R M N R M
In-Domain
MSM-Dev 19.45 64.61 15.29 30.96 81.45 24.95 31.12 83.46 24.77 31.33 83.51 25.00 31.35 83.59 25.00 32.17 83.82 25.83
DL-19 42.85 38.18 72.54 56.99 45.1287.16 56.5550.1581.05 57.8549.69 81.56 55.23 49.07 80.04 57.76 50.04 82.30
DL-20 40.52 41.22 71.32 58.04 52.23 87.35 56.31 54.81 80.31 57.54 55.31 83.10 57.19 54.3088.02 58.48 55.7986.47
BEIR (14 datasets)
ArguAna 44.94 95.16 36.49 49.04 97.94 40.69 45.62 96.73 36.59 43.37 95.31 34.71 54.04 98.65 44.62 49.08 97.73 40.08
Climate-FEVER 7.16 23.60 9.37 18.84 47.24 25.83 19.11 48.99 25.79 19.41 48.95 26.20 20.97 51.24 28.34 19.59 48.87 26.69
CQADupStack 20.41 49.19 19.79 29.51 61.64 28.61 34.64 67.40 33.76 34.10 66.78 33.14 34.07 67.31 33.16 34.40 67.08 33.61
DBPedia 27.03 41.70 56.80 34.62 46.77 68.18 39.40 54.00 72.08 39.22 53.93 71.92 37.58 52.48 68.69 39.4453.70 71.81
FEVER 27.22 64.22 24.08 66.01 92.10 63.62 73.08 94.17 71.15 72.18 94.14 70.15 62.66 93.33 58.55 69.93 93.90 67.34
FiQA 12.41 36.34 17.26 26.34 60.09 32.83 29.83 64.80 36.56 29.76 63.35 36.58 30.17 66.57 36.76 29.23 64.41 35.86
HotpotQA 41.01 62.82 54.23 52.01 67.11 69.54 63.11 79.54 80.42 62.62 79.22 79.91 60.80 78.02 77.62 62.78 79.16 80.04
NFCorpus 27.11 26.11 45.92 31.38 29.24 51.26 33.25 29.95 54.65 32.91 29.81 53.46 33.10 29.89 54.28 32.87 29.43 53.27
NQ 18.05 67.08 14.60 34.50 82.06 29.22 36.34 88.24 30.81 38.7488.1333.43 35.30 88.13 29.62 37.12 88.3831.60
Quora 83.37 98.53 82.66 84.45 98.47 83.64 86.12 99.1985.30 85.74 99.10 84.93 85.88 99.13 85.18 86.5299.18 85.76
SCIDOCS 10.97 32.37 19.74 15.10 35.03 27.24 16.22 38.40 28.84 16.40 38.46 28.75 17.10 39.64 29.97 16.84 39.22 29.36
SciFact 57.14 89.66 53.19 60.06 90.39 55.38 68.5894.6064.84 68.26 93.93 64.04 68.8892.9365.14 68.73 94.28 64.86
TREC-COVID 18.16 1.91 39.04 45.53 7.90 74.81 47.48 5.99 75.74 55.06 8.54 86.06 40.51 4.77 65.23 46.45 5.80 75.03
Touche-2020 7.25 15.66 21.26 18.01 39.54 34.32 15.71 38.82 29.78 19.57 39.72 39.51 15.82 38.90 27.35 17.18 38.93 30.79
BRIGHT (12 datasets)
AOPS 3.46 9.00 6.31 4.22 8.94 7.63 6.13 16.3511.91 5.3317.189.96 5.63 14.58 9.78 6.9617.09 11.52
Biology 1.58 11.09 1.96 5.92 22.25 8.97 15.74 46.30 21.25 17.73 49.07 22.94 11.47 43.84 16.04 13.16 42.47 17.59
Earth Sci. 3.44 12.26 5.02 11.11 36.47 14.40 20.35 53.36 27.59 23.31 56.43 31.50 16.34 47.93 21.41 17.87 51.13 23.37
Economics 2.27 9.03 3.93 9.55 31.34 12.58 16.04 41.79 21.52 17.27 45.27 22.61 12.50 36.85 17.65 15.68 42.05 22.43
LeetCode 13.34 48.31 12.33 12.35 29.86 12.34 13.74 48.13 12.47 12.93 45.73 12.17 14.43 50.01 13.43 13.29 47.65 12.02
Pony 0.07 7.25 0.13 1.66 10.00 3.54 6.7222.87 17.66 5.11 23.1612.28 3.05 15.95 7.56 5.09 19.68 12.12
Psychology 2.43 11.95 3.00 8.74 32.20 10.11 15.80 44.20 20.03 17.13 51.56 21.53 12.40 42.19 14.42 14.54 45.71 18.40
Robotics 5.21 14.60 7.30 6.59 21.00 9.27 14.50 38.0917.97 14.6836.40 18.32 9.59 29.31 12.64 12.39 34.80 15.85
StackOverflow 4.42 16.64 4.51 7.31 26.27 8.35 11.68 43.20 14.08 10.42 42.13 11.99 9.55 37.43 10.66 11.9939.44 13.53
Sust. Living 1.43 14.98 2.23 8.40 34.67 9.71 14.25 52.07 18.30 16.76 53.70 21.15 9.42 45.23 11.46 14.18 51.26 16.42
TheoremQA-Q 7.66 18.56 8.20 6.92 15.62 7.64 11.30 22.67 12.53 11.31 23.7312.70 10.99 22.2512.73 11.21 22.55 12.44
TheoremQA-T 0.74 5.92 0.40 2.53 8.99 2.63 4.66 15.79 4.79 3.59 13.60 3.95 3.82 11.84 4.08 3.60 13.75 3.46
Multi-hop (4 datasets)
2WikiMHopQA 47.85 66.49 65.94 61.39 71.41 85.93 66.85 76.91 90.69 66.35 76.72 89.90 66.95 77.3090.63 66.02 76.83 89.50
MuSiQue 32.11 60.58 47.66 33.72 65.64 49.80 35.7168.1051.03 35.70 68.12 50.93 34.87 68.41 49.85 35.5168.4750.76
NovelHopQA 37.84 86.70 32.19 54.53 93.90 48.69 66.29 96.62 60.86 64.31 96.57 58.78 68.25 96.92 63.08 64.38 96.54 58.82
HotpotQA 41.01 62.82 54.23 52.01 67.11 69.54 63.11 79.54 80.42 62.62 79.22 79.91 60.80 78.02 77.62 62.78 79.16 80.04
Averages
DL Avg (2) 41.68 39.70 71.93 57.52 48.6887.26 56.43 52.48 80.68 57.70 52.50 82.33 56.21 51.68 84.03 58.12 52.9284.38
BEIR Avg (14) 28.73 50.31 35.32 40.39 61.11 48.94 43.46 64.34 51.88 44.1064.2453.06 42.6364.3650.32 43.58 64.29 51.86
BRIGHT Avg (12) 3.84 14.97 4.61 7.11 23.13 8.93 12.58 37.07 16.68 12.96 38.16 16.76 9.93 33.12 12.66 11.66 35.63 14.93
MHop Avg (4) 39.70 69.15 50.00 50.41 74.51 63.49 57.99 80.29 70.75 57.24 80.16 69.88 57.72 80.16 70.30 57.17 80.25 69.78
D.2. RetroMAE Results
Table 13 presents the complete results for RetroMAE across all evaluation benchmarks.
D.3. E5 Results
Table 14 presents the complete results for E5 across all evaluation benchmarks. Note that magnitude-aware variants
(Dot, QNorm, DNorm, Learnable) require removing E5’s built-in normalization layer, which disrupts the pre-trained
13

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 13.Complete Evaluation Results for RetroMAE (mean of seeds 0, 42, 1337). Comparing Pretrained and five fine-tuned methods
(Cosine, Dot, QNorm, DNorm, Learnable). N = NDCG@10, R = Recall@100, M = MRR@10.Bold= best among fine-tuned, underline
= second best.
Pretrained Cosine Dot QNorm DNorm Learnable
Dataset N R M N R M N R M N R M N R M N R M
In-Domain
MSM-Dev 2.87 14.14 2.25 30.76 78.84 25.12 31.41 81.76 25.38 32.11 81.84 25.95 33.01 82.67 26.79 32.84 82.60 26.67
DL-19 14.05 10.35 37.44 56.38 44.36 88.57 57.73 46.44 85.93 57.6447.1790.33 58.04 47.02 91.16 60.4346.8291.73
DL-20 9.00 8.30 23.30 56.14 48.76 83.33 58.62 52.21 85.76 57.34 52.35 86.16 59.26 52.87 86.73 59.69 53.07 87.08
BEIR (14 datasets)
ArguAna 32.70 78.66 26.64 44.49 96.09 36.39 48.57 97.37 39.93 52.25 98.29 43.44 48.89 97.72 40.20 52.41 98.36 43.71
Climate-FEVER 5.95 20.03 8.30 19.73 46.75 27.09 17.13 44.93 23.40 20.2148.94 27.56 19.4149.1226.18 19.53 48.87 26.64
CQADupStack 10.22 27.20 10.04 26.40 56.67 25.71 30.79 61.30 30.09 31.51 61.63 30.85 32.15 62.41 31.53 31.96 62.07 31.43
DBPedia 4.78 9.79 13.78 33.01 41.2668.09 29.90 45.41 58.98 33.38 46.05 64.24 33.56 47.2964.36 33.56 46.95 64.76
FEVER 4.72 13.70 4.23 62.9791.5959.99 61.89 90.92 59.19 62.58 91.83 59.59 62.60 92.3559.02 61.69 91.60 58.45
FiQA 2.75 11.21 3.60 22.73 51.21 27.93 24.94 55.33 30.67 25.19 56.06 30.31 26.1456.67 32.15 25.98 56.8831.37
HotpotQA 20.88 38.21 28.44 48.92 62.24 66.37 55.14 71.29 72.95 56.34 71.47 74.40 57.33 73.02 74.92 56.50 72.78 74.04
NFCorpus 6.22 11.25 13.73 27.48 24.68 48.21 28.30 25.64 48.46 29.08 25.95 49.17 28.96 25.12 47.59 28.57 25.13 47.48
NQ 3.73 21.80 2.86 33.46 79.94 28.97 35.41 84.34 30.14 34.87 84.13 29.68 36.49 85.11 31.12 36.11 84.72 30.85
Quora 56.76 82.95 56.47 82.86 97.99 82.22 82.48 97.90 81.86 84.47 98.72 83.81 85.27 98.8684.53 85.3498.86 84.68
SCIDOCS 4.07 16.74 7.76 12.94 30.40 23.95 14.79 33.28 26.91 15.07 32.98 27.22 15.7433.93 28.96 15.34 33.9927.72
SciFact 30.93 67.40 27.74 51.23 85.60 48.11 57.40 89.53 53.26 57.28 89.77 54.00 59.07 90.1055.23 58.79 89.7755.29
TREC-COVID 15.72 1.68 31.36 54.44 8.72 77.74 60.49 10.72 87.17 52.93 9.72 80.18 53.48 9.24 73.17 53.87 9.38 79.84
Touche-2020 0.41 4.00 1.14 17.91 39.35 31.65 16.91 38.4232.20 16.92 38.54 29.88 19.55 40.8730.42 17.39 38.88 30.49
BRIGHT (12 datasets)
AOPS 4.00 12.44 8.54 1.62 7.49 2.79 3.90 14.29 7.39 4.33 15.55 8.04 5.1414.519.10 4.38 15.708.09
Biology 6.10 23.57 8.54 5.68 22.37 8.81 8.75 31.83 13.16 8.58 30.41 13.42 10.09 32.07 13.22 10.72 32.64 16.30
Earth Sci. 7.80 23.86 11.36 11.87 35.57 17.19 14.84 38.48 17.84 14.59 37.89 18.49 17.30 41.27 21.64 15.49 40.05 19.09
Economics 5.81 28.04 9.68 9.46 30.56 12.15 11.38 33.74 12.64 13.02 36.77 16.22 10.98 33.20 11.98 13.13 37.1914.98
LeetCode 11.44 40.01 10.60 11.65 26.69 11.56 14.21 43.77 13.15 13.63 41.48 13.30 14.67 44.41 13.92 14.08 42.69 13.46
Pony 3.17 8.06 8.83 0.56 3.70 1.28 0.92 6.97 2.51 3.43 13.66 9.24 4.36 13.68 11.80 3.67 13.62 9.54
Psychology 3.89 16.35 4.98 10.13 30.88 12.78 10.59 31.66 12.11 13.08 36.0514.82 13.1834.6916.43 12.97 35.63 15.22
Robotics 6.71 21.02 9.06 7.47 20.32 11.23 12.9537.40 16.28 10.42 35.09 13.70 11.4238.5315.58 12.45 35.7116.44
StackOverflow 3.13 13.61 3.68 4.40 20.00 5.75 6.40 29.39 9.12 7.11 32.06 8.49 8.29 31.92 10.22 9.18 32.76 11.24
Sust. Living 6.20 25.16 8.73 7.11 33.60 8.21 11.1843.6613.77 11.76 40.34 13.22 12.3842.20 16.02 12.25 41.27 14.17
TheoremQA-Q 3.90 8.44 4.46 4.49 11.46 5.13 7.42 13.86 8.79 7.47 16.91 9.72 6.24 13.26 7.75 7.20 15.32 9.12
TheoremQA-T 0.27 1.97 0.19 1.55 5.92 1.17 2.749.212.82 2.32 11.622.08 1.63 7.02 1.58 1.45 9.25 1.17
Multi-hop (4 datasets)
2WikiMHopQA 19.18 36.86 27.38 63.65 70.74 89.95 60.46 71.17 85.09 64.41 73.14 90.33 64.49 73.5590.01 62.69 72.78 88.10
MuSiQue 12.55 38.20 17.22 33.4260.4850.00 31.79 62.17 45.97 33.12 63.57 47.84 32.53 63.42 46.90 32.5163.6346.83
NovelHopQA 21.20 60.32 17.96 58.61 93.49 52.85 65.11 95.33 60.04 66.28 95.83 61.13 70.23 96.48 65.46 68.93 96.30 63.95
HotpotQA 20.88 38.21 28.44 48.92 62.24 66.37 55.14 71.29 72.95 56.34 71.47 74.40 57.33 73.02 74.92 56.50 72.78 74.04
Averages
DL Avg (2) 11.52 9.32 30.37 56.26 46.56 85.95 58.18 49.32 85.84 57.49 49.76 88.24 58.65 49.9488.94 60.0649.94 89.40
BEIR Avg (14) 14.27 28.90 16.86 38.47 58.04 46.60 40.30 60.46 48.23 40.86 61.01 48.88 41.33 61.5648.53 41.22 61.30 49.05
BRIGHT Avg (12) 5.20 18.54 7.39 6.33 20.71 8.17 8.77 27.86 10.80 9.14 28.99 11.73 9.64 28.9012.44 9.75 29.3212.40
MHop Avg (4) 18.45 43.40 22.75 51.15 71.74 64.79 53.12 74.99 66.01 55.04 76.00 68.43 56.14 76.62 69.32 55.16 76.37 68.23
representations.
D.4. E5 Results (MS MARCO 500K)
Table 15 presents the complete results for E5 trained on MS MARCO 500K (compared to 80K above). With more training
data, DNorm overcomes E5’s architectural constraint and achieves the best performance on reasoning-intensive tasks
(BRIGHT and Multi-hop).
E. E5 Experiments and Analysis
E5 (Wang et al., 2022) presents a unique case study for magnitude-aware training due to its architectural constraint: unlike
Contriever and RetroMAE, E5 includes a built-in normalization layer as its final module. This section provides detailed
analysis of E5’s behavior when magnitude-aware training is enabled by removing this normalization constraint.
14

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 14.Complete Evaluation Results for E5 (seed=0). Comparing Pretrained and five fine-tuned methods (Cosine, Dot, QNorm, DNorm,
Learnable). N = NDCG@10, R = Recall@100, M = MRR@10.Bold= best among fine-tuned, underline = second best. Magnitude-aware
variants require removing E5’s normalization layer.
Pretrained Cosine Dot QNorm DNorm Learnable
Dataset N R M N R M N R M N R M N R M N R M
In-Domain
MSM-Dev 18.13 60.11 13.98 33.6384.68 27.14 16.74 71.83 12.16 9.53 48.39 6.87 26.01 75.07 21.02 28.56 85.6621.96
DL-19 33.03 27.84 53.57 62.2750.41 92.64 22.80 35.58 30.51 10.36 18.27 15.50 52.31 40.07 82.99 48.8150.8364.95
DL-20 31.97 35.81 58.36 63.9257.18 90.74 28.00 41.45 38.73 12.79 26.80 17.12 56.95 45.60 86.07 51.3257.8764.30
BEIR (14 datasets)
ArguAna 35.19 91.39 27.80 50.11 98.08 41.74 1.04 13.23 0.52 18.38 67.71 13.78 16.67 63.02 12.69 31.96 88.48 24.65
Climate-FEVER 15.86 40.72 22.78 20.70 51.11 27.55 6.33 19.81 8.02 26.07 54.11 38.77 3.11 11.79 4.52 20.57 51.07 28.03
CQADupStack 25.64 61.07 24.15 40.5074.38 39.52 35.23 70.58 33.85 26.17 72.36 23.32 31.19 64.41 30.15 35.31 74.7633.28
DBPedia 18.19 32.00 38.00 36.23 47.43 70.21 14.20 28.93 24.68 11.91 28.46 20.39 31.33 41.43 62.56 24.35 44.99 41.22
FEVER 49.48 84.29 46.46 67.50 93.05 65.19 52.48 81.53 49.98 61.26 90.69 58.16 13.02 38.67 11.33 60.60 91.14 57.12
FiQA 26.18 62.98 29.91 34.30 67.4241.55 30.19 68.36 35.95 27.03 70.51 28.66 23.68 54.19 30.26 34.72 71.5940.28
HotpotQA 51.06 67.29 66.70 52.17 66.16 69.38 49.19 64.84 63.28 47.78 68.26 60.36 29.41 48.21 38.20 56.62 73.20 72.33
NFCorpus 30.40 30.99 48.97 34.6333.6855.63 35.35 33.34 56.46 33.77 33.23 50.85 26.86 25.67 45.90 35.8533.36 56.63
NQ 25.33 73.02 21.32 40.74 86.32 35.52 5.38 22.52 4.28 2.43 12.65 1.94 33.14 79.44 28.37 17.46 64.81 14.08
Quora 83.08 97.06 82.66 87.16 99.22 86.38 59.57 88.45 58.00 77.52 94.66 77.10 74.44 94.15 73.86 81.57 96.45 81.15
SCIDOCS 1.94 5.36 4.44 19.99 46.36 33.81 3.53 18.95 6.39 3.74 10.26 7.25 5.88 25.72 10.64 6.12 21.09 11.15
SciFact 41.13 85.79 35.62 68.76 95.33 64.74 45.47 75.06 41.68 57.12 83.96 53.14 23.65 58.78 21.03 59.44 87.23 55.74
TREC-COVID 31.82 3.21 60.38 64.80 11.87 91.00 25.71 2.86 44.03 31.64 2.85 56.92 29.81 3.47 52.27 37.41 3.64 62.16
Touche-2020 1.52 3.69 4.66 20.28 43.74 36.80 0.00 10.71 0.00 0.28 0.64 1.02 4.06 23.96 10.66 0.37 1.60 1.43
BRIGHT (12 datasets)
AOPS 2.58 7.47 4.98 2.68 13.34 4.90 0.00 0.00 0.00 0.12 1.36 0.30 0.00 0.00 0.00 1.41 4.55 2.70
Biology 15.29 50.94 20.72 7.58 34.50 11.32 0.91 7.43 1.46 17.44 50.43 25.14 5.69 20.79 8.48 12.68 40.29 17.57
Earth Sci. 17.69 46.08 22.11 13.77 40.94 18.79 0.73 11.32 0.64 21.04 47.26 28.34 6.28 17.43 8.34 16.58 41.08 22.11
Economics 12.24 39.33 16.18 11.13 36.90 12.78 0.00 7.09 0.00 9.30 29.35 10.91 3.64 9.20 4.53 9.77 33.08 11.16
LeetCode 14.83 48.49 14.77 14.63 39.68 14.67 0.48 1.82 0.78 9.75 25.09 10.46 5.55 12.10 6.81 12.15 34.88 12.51
Pony 1.72 10.84 5.08 0.40 3.71 0.64 10.37 30.07 23.10 29.32 35.33 65.36 7.04 27.08 14.41 13.83 24.17 37.71
Psychology 16.66 43.57 18.79 10.92 36.73 13.14 0.26 7.81 0.25 13.8842.02 14.57 2.38 11.31 2.48 11.64 43.1512.72
Robotics 10.21 26.93 15.14 8.75 26.24 11.89 1.06 4.42 1.24 8.24 27.97 13.10 2.06 12.13 3.24 10.76 30.80 13.90
StackOverflow 9.99 42.63 12.27 10.1243.3213.14 1.11 9.50 1.50 10.6224.1514.26 2.38 12.19 2.39 10.19 32.27 13.05
Sust. Living 10.36 40.36 13.50 8.48 39.4410.56 0.45 5.07 1.06 8.03 31.22 10.38 1.26 12.83 2.81 8.25 35.30 11.81
TheoremQA-Q 11.24 24.57 12.77 12.81 25.31 15.46 0.52 2.32 0.52 6.30 14.13 7.92 2.50 8.98 2.55 9.14 17.90 11.72
TheoremQA-T 4.96 16.23 4.62 5.96 28.295.49 0.00 4.06 0.00 3.86 9.43 3.87 1.61 3.07 2.63 5.77 12.94 5.92
Multi-hop (4 datasets)
2WikiMHopQA 62.43 73.30 83.71 68.77 74.83 93.84 68.24 75.37 91.57 70.19 76.77 93.31 38.54 58.90 52.99 71.12 77.01 94.44
MuSiQue 30.34 60.26 43.74 34.61 64.95 50.57 30.86 59.52 43.22 33.70 62.94 47.83 21.04 46.21 29.50 34.16 64.01 48.32
NovelHopQA 61.33 94.98 55.50 54.6694.2748.69 47.75 83.04 43.57 60.3492.2955.30 23.00 64.63 20.14 60.24 92.34 55.08
HotpotQA 51.06 67.29 66.70 52.17 66.16 69.38 49.19 64.84 63.28 47.78 68.26 60.36 29.41 48.21 38.20 56.62 73.20 72.33
Averages
DL Avg (2) 32.50 31.83 55.96 63.1053.80 91.69 25.40 38.52 34.62 11.58 22.54 16.31 54.63 42.84 84.53 50.0654.3564.62
BEIR Avg (14) 31.20 52.78 36.70 45.56 65.30 54.22 25.98 42.80 30.51 30.36 49.31 35.12 24.73 45.21 30.89 35.88 57.39 41.37
BRIGHT Avg (12) 10.65 33.12 13.41 8.9430.7011.07 1.32 7.58 2.55 11.4928.1417.05 3.37 12.26 4.89 10.18 29.20 14.41
MHop Avg (4) 51.29 73.96 62.41 52.55 75.05 65.62 49.01 70.69 60.41 53.00 75.06 64.20 28.00 54.49 35.21 55.54 76.64 67.54
E.1. Main Results
E.2. Architectural Constraints and Performance Trade-offs
To enable magnitude-aware training for E5, we programmatically detect and remove its built-in normalization layer.2
Performance trade-offs.Removing E5’s normalization layer during fine-tuning creates severe performance degradation.
On in-domain and BEIR benchmarks, magnitude-aware variants dramatically underperform Cosine (Table 16), with ∆%
ranging from −81.7% to−30.0% . This occurs because E5’s representations were fundamentally optimized for normalized
embeddings during pre-training, and removing this constraint destabilizes learned representations. However, on reasoning-
intensive benchmarks, QNorm shows modest improvement over Cosine (BRIGHT: +28.6%; Multi-hop: +0.9%), suggesting
that the strong magnitude-relevance signal on these tasks partially compensates for the architectural mismatch.
Implications.The E5 results provide a critical lesson:architectural compatibility is essential. Magnitude-aware training
requires matching the training configuration with the original model architecture. Models pre-trained with normalization
2Our training code iterates through the model’s modules and removes any instance of
sentence transformers.models.Normalize , which would otherwise discard all magnitude information regardless
of the loss function used.
15

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
have learned representations that fundamentally depend on unit-norm constraints, and removing this constraint destabilizes
the learned structure.
E.3. Extended Training with 500K Data
A natural hypothesis is that E5’s in-domain performance degradation with magnitude-aware training stems from insufficient
training data, as 80K samples may be too few to adapt representations optimized over 1 billion pre-training pairs. To test this,
we conduct additional experiments using the MS MARCO Passage Ranking training set (Nguyen et al., 2016), specifically
the “judged” subset containing 502,939 queries with relevance annotations, the standard training corpus used by mainstream
retrievers including Sentence-Transformers and Contriever. This provides 6×more training signal than MS MARCO v1.1
QA (80K).
Experimental setup.We train E5 for 20 epochs using the same hyperparameters as the 80K experiments (learning rate
5×10−6, effective batch size 256). The 20-epoch duration is derived from our convergence analysis (Appendix C.2): 80K
models reach optimal performance at 40–90 epochs, so proportionally scaling by the 6.15 ×data ratio suggests 7–15 epochs,
with 20 epochs providing a conservative margin to ensure convergence.
Results.Table 16b shows E5’s performance with 6 ×more training data. The key finding is thatDNorm becomes the
best-performing variant on reasoning-intensive benchmarks:
•BRIGHT: DNorm achieves 11.71 vs. Cosine’s 9.47 (+23.7%), reversing the 80K pattern where Cosine outperformed
DNorm (8.94 vs. 3.37).
•Multi-hop: DNorm achieves 58.82 vs. Cosine’s 55.52 (+5.9%), compared to 80K where DNorm severely underperformed
(28.00 vs. 52.55).
•In-domain: Cosine remains best, but Dot becomes competitive (52.95–53.68 on TREC-DL) vs. the 80K collapse
(22.80–28.00).
Analysis.The dramatic improvement in DNorm performance (from 28.00 to 58.82 on Multi-hop, a +110% gain) suggests
that E5’s architectural constraint (pre-training with normalized embeddings) can be partially overcome with sufficient
fine-tuning data. With 80K samples, the model lacks the capacity to restructure its representations to exploit document
magnitude; with 500K samples, meaningful magnitude-relevance patterns emerge. However, Cosine remains optimal for
in-domain evaluation, indicating that E5’s pre-trained representations are fundamentally optimized for angular similarity,
and magnitude learning provides complementary benefits primarily on out-of-distribution tasks.
In-Domain BEIR BRIGHT Multi-hop02040
1.4 0.2 0.5327.9
3.92.3810.8
1.6
−1.13.8
−8.212.2
8.330.8∆NDCG@10
Cosine Dot QNorm DNorm
Figure 8.Performance improvement ( ∆NDCG@10) when scaling E5 training data from 80K to 500K. On out-of-domain benchmarks
(BEIR, BRIGHT, Multi-hop), DNorm (red) shows dramatically larger gains, especially on Multi-hop (+30.83). On In-Domain, the pattern
reverses: Dot shows the largest gain (+27.91) while DNorm decreases ( −8.20 ), reflecting E5’s pre-training bias toward normalized
embeddings for in-distribution data.
Data scaling analysis.Figure 8 visualizes the performance improvement when scaling training data from 80K to 500K.
The key observation is thatmagnitude-aware methods benefit disproportionately from additional training data. DNorm
shows the largest gains across all out-of-domain benchmarks: +30.83 on Multi-hop, +12.18 on BEIR, and +8.34 on BRIGHT.
In stark contrast, Cosine shows minimal improvement ( ∆<3 NDCG@10), indicating that magnitude-unaware training
cannot leverage additional data to improve out-of-domain generalization. This asymmetry demonstrates that learning to
16

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
exploit document magnitude requires sufficient training signal to restructure representations; E5’s architectural constraint
(pre-training with normalized embeddings) can be overcome given enough fine-tuning data.
F. Verification Experiments: STS and CLIP
This appendix provides detailed verification experiments on symmetric tasks (STS) and vision-language models (CLIP) to
validate our task symmetry hypothesis.
F.1. STS Experiments: Extended Details
This section provides additional details for the STS experiments in Table 5 (main text).
Experimental Setup.We train Contriever and RetroMAE on STS Benchmark combined with AllNLI ( ∼950K samples),
implementing the same 2×2 ablation: Cosine (both normalized), Dot (neither normalized), and asymmetric variants
Sent1Norm/Sent2Norm (analogous to QNorm/DNorm).
Key Findings.
•Cosine≈Dot: Unlike retrieval, magnitude provides no benefit in symmetric tasks
•Asymmetric normalization fails catastrophically: 40–45 point degradation
•Sent1Norm ≈Sent2Norm: As expected in symmetric tasks, the choice of which sentence to normalize does not matter
F.2. CLIP Experiments: Extended Details
This section provides extended details for the CLIP experiments in Table 6 and Figure 6 (main text).
Zero-shot analysis.We first analyze CLIP (ViT-B/32) on MS-COCO without fine-tuning. Cohen’s d≈0 for both
modalities (Image: −0.0002 , Text: −0.0009 ), and Dot hurts performance (Text →Image R@1: 25.8 vs 30.5 for Cosine).
This occurs because CLIP normalizes embeddings during both pre-training and inference.
Pre-training setup.We train CLIP models on COCO Captions with the following configurations:
•Norm: Standard CLIP with L2 normalization and symmetric loss (I2T + T2I)
•NoNorm-Sym: No normalization, symmetric loss
•ImageNorm-I2T / TextNorm-T2I: Single-direction with partial normalization
Conclusion.These experiments establish that magnitude learning requiresbothremoving normalizationandasymmetric
task structure. CLIP’s symmetric loss prevents magnitude learning even without normalization, explaining why our retrieval
findings do not transfer to standard vision-language models.
G. Discussion Details
This appendix provides detailed theoretical analysis and RAG experimental setup moved from the main text.
G.1. Task Symmetry Framework
This section extends Corollary 6.1 (main text) with a taxonomy and detailed implications.
Implications for similarity function choice.
•Asymmetric tasks: Magnitude can encode role-specific semantics. Document magnitude encodes “relevance strength”
while query magnitude modulates “matching confidence” (Propositions 4.1–4.2).
17

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
•Symmetric tasks: Corollary 6.1 shows that partial normalizationmathematicallybreaks the symmetry requirement.
While Dot preserves symmetry, it offers no benefit over Cosine because the symmetric task structure prevents magnitude
from encoding role-specific information.
G.2. The Asymmetry Principle: Detailed Analysis
Our analysis reveals a fundamental asymmetry in how query and document magnitudes function:
•Document magnitudeoperates atinference time: it directly modulates relevance scores, allowing the model to express
“this document is especially relevant.”
•Query magnitudeoperates attraining time: it modulates gradient strength via the effective softmax temperature,
allowing the model to express “I am confident about this query.”
This asymmetry helps explain why different pre-trained models benefit from different magnitude strategies. Contriever and
E5, both pre-trained with contrastive learning, have learned to encode semantic information in document representations,
and hence QNorm (preserving document magnitude) helps most; their query magnitude CV does not increase (or even
decreases for E5) under DNorm because they do not exploit query magnitude. For RetroMAE, pre-trained with masked
auto-encoding, the pattern reverses: DNorm yields better performance and query magnitude CV increases 6 ×, suggesting
RetroMAE learns to exploit query magnitude variation when given the opportunity.
More broadly, these findings align with the geometric view from Section 2: magnitude acts as asoft reliability weightthat
becomes most valuable when the model encounters inputs where angular similarity alone is insufficient. The model can
effectively encode “this document is especially relevant despite indirect semantic match,” a capability that cosine similarity’s
unit sphere constraint explicitly prevents.
G.3. Magnitude as Per-Example Temperature
In standard contrastive learning, temperature τis a global hyperparameter that uniformly scales all similarity scores.
Our analysis reveals that embedding magnitude can serve as alearned, per-examplescaling factor. When document
magnitude ∥d∥varies, the effective similarity becomes s=∥d∥ ·cos(q,d) , allowing the model to adaptively emphasize or
de-emphasize specific examples. This is analogous to learning a separate temperature for each example, but without explicit
parameterization.
G.4. Connection to Logit Scaling in Classification
LogitNorm (Wei et al., 2022) treats logit magnitude as a source of overconfidence to be removed via normalization. The
key evidence for LogitNorm is that high logit magnitude does not correspond to high accuracy, meaning magnitude is
miscalibratedwith respect to the task.
Our finding presents the complementary perspective with equally strong evidence: in retrieval, magnitude iscalibratedwith
relevance. On reasoning-intensive benchmarks, we observe Cohen’s daveraging 1.22 (with individual datasets reaching
d= 1.80 ), which are large effect sizes indicating that magnitude reliably distinguishes relevant from irrelevant documents.
This is precisely the opposite of miscalibration: the model has learned to use magnitude as a meaningful relevance signal.
The key difference is thus task-dependent: in classification, unconstrained magnitude reflects training dynamics rather than
true confidence (miscalibration), while in retrieval, magnitude reflects relevance strength (calibration). This suggests a
general principle:whether to normalize depends on whether magnitude carries task-relevant information that is calibrated
with the objective.
G.5. Implications for Other Domains
The choice between cosine and dot product similarity in contrastive learning is not merely about bounded vs. unbounded
scores, but about whether the task benefits from per-example adaptive scaling:
• Inretrieval, document magnitude can encode relevance strength, which directly benefits ranking.
•Insymmetric contrastive learning(SimCLR, MoCo), magnitude could encode example “informativeness,” where
harder or more distinctive examples might benefit from larger magnitudes.
18

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
•Invision-language learning(CLIP), image and text magnitudes could encode modality-specific confidence about the
match.
These hypotheses remain to be tested, but our retrieval results provide existence proof that magnitude can carry task-relevant
information in contrastive learning.
G.6. Scope of Experimental Verification
In this paper, we provide experimental evidence for the following tasks: (1)dense text retrievalwith three pre-trained
models across 30+ evaluation datasets (Section 3), (2)RAG-based question answeringshowing retrieval gains transfer to
downstream QA accuracy (Section 6.2), (3)semantic textual similaritydemonstrating that asymmetric normalization harms
symmetric tasks (Section 6.4), and (4)vision-language retrievalwith CLIP confirming the task symmetry principle extends
to cross-modal settings (Section 6.5). Other applications in Table 17 (e.g., recommendation systems, code search) share
similar asymmetric structure and may benefit from magnitude-aware training, but we have not directly verified them.
H. Detailed Proofs and Analysis
H.1. Proof of Proposition 1 (Ranking Equivalence)
For a fixed q, the quantities ∥q∥and1/∥q∥ are positive constants across all d∈ D . Multiplying all scores by a positive
constant preserves their ordering. We have scos= cosθ ,sdnorm =∥q∥cosθ ,sqnorm =∥d∥cosθ , and sdot=∥q∥∥d∥cosθ .
Since∥q∥>0is constant across documents,π cos=π dnorm andπ qnorm =π dot.
H.2. Optimization Dynamics Mechanisms
Even when Cohen’s dis small or negative, releasing the unit-norm constraint can improve learning through several
mechanisms:
1.Gradient flow: L2 normalization introduces a Jacobian term (I− ˆvˆv⊤)/∥v∥ during backpropagation, which projects
gradients onto the tangent space of the hypersphere. Removing normalization eliminates this projection, allowing
gradients to flow more directly and potentially enabling faster convergence to better minima.
2.Representation capacity: Constraining representations to the unit hypersphere Sn−1reduces the effective dimensional-
ity from nton−1 . Releasing this constraint restores the full Rnspace, providing additional capacity that may help the
model learn better angular structures even if the magnitude dimension itself is not used for relevance encoding.
3.Loss landscape smoothing: The normalization operation creates a non-convex mapping that can introduce sharp
curvature in the loss landscape. Dot product similarity, being a simple linear operation, may yield a smoother landscape
that is easier to optimize.
H.3. What Does Magnitude Encode Whend <0?
The negative Cohen’s don in-domain tasks raises an interesting question: when magnitude shows weak or negative
correlation with relevance, whatelsemight it encode? We hypothesize that on well-matched training domains, magnitude
may capturedocument-level propertiessuch as specificity or information density. Short, generic documents may receive
smaller magnitudes while detailed, specific documents receive larger magnitudes. However, in MS MARCO, relevant
documents are often short answer passages rather than detailed expositions, leading to negative d. This interpretation
suggests that magnitude is a flexible channel that models use to encode whatever information is most useful for the task at
hand.
H.4. Proposition 2: Gradient Asymmetry Details
Under InfoNCE loss with DNorm similaritys dnorm(q,d) =∥q∥cosθ:
Effective temperature.The softmax distribution over documents has per-query effective temperatureτ eff(q) =τ/∥q∥:
pj=exp(∥q∥cosθ j/τ)P
kexp(∥q∥cosθ k/τ)(9)
19

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
High∥q∥sharpens the distribution (confident query); low∥q∥smooths it (ambiguous query).
Gradient modulation.The gradient w.r.t. the positive document is:
∇ˆd+L=−∥q∥
τ(1−p+)ˆq(10)
The gradient magnitude is proportional to ∥q∥: the model can learn to upweight “confident” queries and downweight
“ambiguous” ones.
I. RAG Experimental Setup
I.1. Evaluation Metrics
We report two standard metrics for open-domain QA: (1)Exact Match (EM): the percentage of predictions that exactly
match the ground-truth answer after normalization (lowercasing, removing articles and punctuation); (2)Token-level F1:
the harmonic mean of precision and recall computed over individual tokens, providing a softer measure that gives partial
credit for partially correct answers.
I.2. Choice ofk
We use k= 5 retrieved documents based on two considerations: (1)Context length constraints: Flan-T5-Large has
an effective context window of 512–1024 tokens, and each retrieved document contains approximately 100–200 tokens;
withk= 5 , the total context (500–1000 tokens) fits within model capacity without truncation. (2)Information density:
Retrieving more documents risks introducing noise that dilutes relevant information; k= 5 balances coverage and precision,
consistent with the original RAG paper (Lewis et al., 2020).
I.3. Prompt Format
We structure the input to Flan-T5-Large with the questionbeforethe retrieved context:
Question:{question}
Answer the question based on the following context:
[1]{doc 1}[2]{doc 2}... [k]{doc k}
Answer:
This ordering ensures that when the total input exceeds the model’s context window, truncation occurs at theendof
the retrieved documents rather than the question itself. Since the question is essential for answer generation while later
documents contribute diminishing marginal value, this design prevents catastrophic failures from question truncation.
J. Why Fine-tuning Pre-trained Retrievers?
Our choice to fine-tune pre-trained retrievers, rather than train from scratch, reflects the dominant paradigm in practical
retrieval systems, where practitioners typically adapt existing checkpoints to new domains. Importantly, this setting ismore
challengingfor magnitude learning than pre-training from scratch: models pre-trained with cosine similarity have learned
representations optimized for unit-norm embeddings, and the magnitude information that dot product can exploit must be
reorganizedfrom these constrained representations during fine-tuning. Thus, our main experiments represent a conservative
test that does not reveal the full potential of magnitude learning.
Among our three models, E5 presents an especially adversarial case: (1) it was pre-trained exclusively with cosine similarity,
(2) it includes a built-in normalization layer that must be removed to enable magnitude-aware training, and (3) it was already
fine-tuned on MS MARCO during its original training, meaning our fine-tuning is essentially asecondadaptation on the
same dataset. These conditions collectively minimize E5’s capacity to learn meaningful magnitude representations. Due to
these unique constraints, we present E5 experiments separately in Appendix E. Our main results focus on Contriever and
20

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
RetroMAE, where magnitude learning provides substantial benefits, particularly on out-of-domain and reasoning-intensive
benchmarks.
K. Limitations and Future Work
Our study focuses on fine-tuning pre-trained models; generalization to pre-training from scratch and other domains (speech,
graphs) requires further investigation. Future directions include: (1) designing asymmetric objectives for vision-language
models that preserve bidirectional retrieval capability while enabling magnitude learning, (2) applying magnitude-aware
training to recommendation systems where item magnitude could encode popularity or quality, and (3) exploring whether
the task symmetry principle extends to other contrastive learning settings beyond those we tested.
L. Bright-Pony Outlier Analysis
The pony dataset has an exceptionally high Queries/Doc ratio of 51.6 (see Table 21), meaning each document is relevant
to∼52 queries on average, far exceeding other datasets (typically 1–5). This “hub” structure causes Cosine similarity to
achieve only 0.56% NDCG@10, an extremely low baseline. The absolute improvement from magnitude-aware methods is
modest (+3–4 NDCG points), but the relative improvement ( ∆%) is amplified by the low baseline. This does not indicate
that document magnitude is ineffective for RetroMAE; rather, the significant correlation on other datasets ( r= 0.68 )
confirms that magnitude encodes relevance. The pony outlier reflects dataset-specific characteristics rather than a failure of
the magnitude mechanism.
M. CV Prediction: Extended Analysis
This section extends the CV analysis in Figure 5 (main text) with additional details.
Measuring CV .Compute the coefficient of variation (CV = std/mean) of document embedding magnitudes from the
pre-trained model on a held-out corpus. High document CV ( ≳0.1 ) indicates the model uses magnitude meaningfully; low
CV (≲0.05) indicates magnitude is noise.
In-domain vs out-of-domain behavior.An important nuance: learnable normalization (Section 2.4) tracks closest to
DNorm for both Contriever and RetroMAE onin-domainvalidation, suggesting query-side magnitude learning is preferred
during fine-tuning. However, QNorm’s advantage for Contriever emerges onout-of-domaingeneralization (BEIR, BRIGHT).
This suggests that document magnitude encodes domain-general relevance signals that transfer better across tasks.
(a) Cohen’sd: Document Magnitude Effect Size
Smalld
∥d∥µirµrd
Irrel. Rel.Larged
∥d∥µir µrd
Irrel. Rel.(b) CV: Query Magnitude Variation
Low CV
∥q∥µsmallσHigh CV
∥q∥µlargeσ
Figure 9.Concept illustrations for magnitude-based metrics.(a) Cohen’s d: Measures the standardized difference between relevant (red)
and irrelevant (blue) document magnitudes. Small dindicates overlapping distributions; large d(≥0.8 ) indicates clear separation where
QNorm can exploit magnitude for ranking.(b) CV: Coefficient of variation ( σ/µ) of query magnitudes. Low CV indicates uniform
magnitudes; high CV indicates the model differentiates queries via magnitude.
∆CV as a model-level predictor.Figure 5 (main text) shows the ∆CV vs. ∆Perf scatter plot for Contriever and RetroMAE
on 39 datasets (3-seed averaged). To test whether ∆CV predicts optimal training configuration, we analyzed per-dataset ∆CV
and∆Perf across models. Note that E5’s fine-tuning exhibited instability (with |∆Perf|>10 on 40% of datasets), likely
due to architectural constraints when removing its normalization layer; we therefore focus on Contriever and RetroMAE.
Key finding: As shown in Figure 5, the two models formcompletely separated clusterswith no ∆CV overlap. Contriever’s
∆CV ranges from 0.5 to 2.0 ×with mean ∆Perf=−1.50 (QNorm better), while RetroMAE’s ∆CV ranges from 4 to
21

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
7×with mean ∆Perf= +0.26 (DNorm better). This clear separation demonstrates that ∆CV reliably distinguishes
which training configuration benefits a given model. Within each cluster, ∆CV does not predict per-dataset performance
(Contriever:r= 0.14; RetroMAE:r=−0.16), confirming that∆CV operates at themodel levelrather than dataset level.
This suggests ∆CV serves as amodel-level indicatorfor selecting training configuration: models with high ∆CV (indicating
they actively learn to differentiate query magnitudes) benefit from DNorm, while models with low ∆CV benefit from
QNorm. RetroMAE consistently shows high ∆CV (4–7 ×) and positive ∆Perf, indicating it actively exploits query
magnitude variation. In contrast, Contriever shows low ∆CV (0.5–2 ×) and negative ∆Perf, indicating it primarily benefits
from document magnitude (QNorm) rather than query magnitude (DNorm).
N. Dot vs. Cosine During Training
To directly observe how magnitude learning affects the angular component of embeddings, we record both dot product and
cosine similarity performance on the validation set throughout training for models trained with DotProductRankingLoss.
This dual evaluation uses thesame embeddingsbut computes retrieval scores with different similarity functions, isolating
the contribution of magnitude from direction.
Table 18 shows that Contriever and RetroMAE maintain a modest but consistent gap (∆≈2–3%) between dot and cosine
performance, indicating that these models learn to utilize magnitude while preserving directional information. In contrast,
E5 with the normalization layer removed exhibits a dramatic divergence: cosine similarity performance collapses from
0.92 to 0.23 while dot performance continues improving to 0.94. This indicates that E5completely abandons directional
informationand relies solely on magnitude for relevance encoding; the angular component becomes essentially random.
This finding has important implications: (1) the normalization layer is necessary not just for cosine similarity computation,
but for maintaining directional information during training; (2) simply removing the normalization layer and using dot
product loss can lead tomagnitude collapse, a pathological state where the model ignores the angular component entirely;
(3) healthy magnitude learning, as exhibited by Contriever and RetroMAE, requires architectural or training mechanisms
that encourage the model to encode information inbothmagnitude and direction. Note that E5 with the normalization
layer retained shows ∆ = 0 because all embeddings have unit norm, making dot product identical to cosine similarity by
definition.
O. Learnable Normalization: Step-Matched Comparison
The learned γvalues translate to strong performance. Figure 2 (main text) shows the step-matched comparison: at each
evaluation step during training, we compare learnable normalization’s validation NDCG@10 against all four discrete
methods trained under identical conditions:
•Contriever: Learnable normalization consistently achieves top-tier validation NDCG@10, with its trajectory closely
following DNorm throughout training.
•RetroMAE: Learnable normalization shows similar behavior, with validation performance trending close to DNorm
across training steps.
These results provide important validation of the discrete framework:
1.DNorm emerges as the preferred strategy: When free to choose any normalization level, both models gravitate toward
DNorm-like performance, confirming that preserving query magnitude while normalizing documents is the preferred
asymmetric strategy.
2.Continuous interpolation can match or exceed discrete variants: For Contriever, the learnable variant achieves the
best performance at every evaluation step, suggesting that even a small continuous adjustment can improve upon any
fixed discrete strategy.
3.Fine-grained control matters: The learned γvalues, though close to the midpoint, represent precise balance points that
the model discovers through optimization. This fine-grained control enables performance that matches or exceeds the
best discrete variants.
22

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
P. Per-Dataset Analysis: Cohen’sdand Query CV
Table 19 presents per-dataset analysis for Contriever and RetroMAE: (a) Cohen’s d(document magnitude effect size) and
∆% (QNorm improvement over Cosine); (b) query magnitude coefficient of variation (CV = σ/µ) under Dot and DNorm
training.
Key observations:
•Cohen’s d: Contriever shows consistently positive Cohen’s dfor BRIGHT datasets ( d= 0.46 –1.80), indicating
relevant documents have substantially larger magnitudes. RetroMAE shows mixed effect sizes. The mean Cohen’s d
differs substantially: Contriever (+0.38) vs. RetroMAE (−0.08).
•Query CV: RetroMAE shows consistently high ∆CV (3.9–8.9 ×) across all datasets, with Dot CV uniformly low
(<1% ). Contriever shows modest ∆CV (0.5–2.0 ×), but BRIGHT shows ∆CV<1 (CV decreases). The non-
overlapping∆CV ranges support∆CV as a model-level predictor.
Q. Dataset Statistics
Q.1. Training Data
Table 20 provides statistics for the training datasets used in our experiments. MS MARCO v1.1 QA is the question-answering
version containing 82K query-answer pairs, which we use for main experiments. MS MARCO Passage Ranking (judged) is
the passage retrieval version containing 503K queries with relevance annotations, which we use for extended E5 experiments
(Section E.3). All statistics are computed on the training split.
Q.2. Evaluation Data
Table 21 provides detailed statistics for all evaluation datasets. Note that TREC-DL 2019 and 2020 contain only 43 and 54
queries respectively, which may lead to high variance in evaluation results. We therefore include MS MARCO Dev (6,980
queries) as a complementary in-domain benchmark to provide more stable estimates.
R. Seed Sensitivity Analysis
To verify the robustness of our findings, we conduct experiments with three different random seeds (0, 42, 1337) for both
Contriever and RetroMAE on MS MARCO 80K. Table 22 reports the validation NDCG@10 statistics across seeds.
Key observations.
•Stable performance: For both models, Cosine, Dot, DNorm, and Learnable show very low variance (std ≤0.06 ),
indicating that our main conclusions are robust to random seed selection.
•Consistent ranking: The relative ordering of methods (Learnable ≈DNorm ≈Dot>Cosine for in-domain validation)
is preserved across all seeds for both models.
•QNorm variance: Both models show higher QNorm variance due to one seed exhibiting different early-training
dynamics. This variance does not affect our main conclusions.
•Best step variation: While the optimal checkpoint step varies across seeds (reflecting different convergence speeds), all
seeds reach similar final performance, validating our early stopping strategy based on validation NDCG@10.
These results demonstrate that our findings, particularly that magnitude-aware training (Dot, DNorm, Learnable) outperforms
Cosine similarity, are reproducible across different random initializations for both Contriever and RetroMAE.
R.1. Statistical Significance on Evaluation Benchmarks
Beyond validation performance stability, we verify that the improvements on evaluation benchmarks are statistically
significant. Table 23 reports mean ±std and paired t-test results across the 3 seeds for Contriever (RetroMAE shows
consistent patterns with DNorm outperforming across all benchmarks).
23

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Statistical significance.Magnitude-aware methods show statistically significant improvements ( p <0.05 ) over Cosine on
BEIR, BRIGHT, and MultiHop benchmarks. The improvements are most pronounced on reasoning-intensive benchmarks,
where all magnitude-aware methods significantly outperform Cosine ( p <0.01 ). TREC-DL shows no significant differences,
likely due to smaller absolute improvements and the in-domain nature of this benchmark. These results confirm that the
benefits of magnitude-aware scoring are robust across random initializations and not due to random variation.
S. STS Verification Experiment Configuration
This appendix provides detailed hyperparameter rationale for the STS verification experiments described in Section 6.4. The
key differences from retrieval experiments arise from task and loss function differences.
S.1. Hyperparameter Comparison
Table 24 compares the hyperparameters used in retrieval and STS experiments.
S.2. Rationale for Key Differences
Learning rate ( 2×10−5vs.2–5×10−6).The STS experiments use 10 ×higher learning rate because: (1) STS training
uses only 4 epochs compared to retrieval’s 100 epochs, requiring faster learning; (2) 2×10−5is the standard learning rate
in SentenceTransformers for BERT-based models.
Scale factor (λ= 0.01vs.20).This is the most critical difference, arising from fundamentally different loss functions:
•Retrieval (InfoNCE): The scale factor α= 20 acts as inverse temperature ( τ= 1/20 = 0.05 ), sharpening the softmax
distribution over candidates. Higher values increase the penalty for ranking errors.
•STS (MSE): The scale factor λ= 0.01 maps dot product values (typically in the range 100–1000 for 768-dimensional
embeddings) to the target similarity range [0,1] . Since dot product ⟨s1,s2⟩ ≈ ∥s 1∥∥s2∥cosθ , and embedding norms
are typically 10–30, the dot product can reach several hundred. Multiplying by 0.01 brings this into the [0,1] range for
MSE loss.
Batch size (64 vs. 128).STS uses smaller batch size because: (1) the STS-B dataset is smaller ( ∼6K samples without
AllNLI); (2) 64 is the default in SentenceTransformers.
Epochs (4 vs. 100).STS converges much faster because: (1) CosineSimilarityLoss in SentenceTransformers is well-
optimized for this task; (2) the combined AllNLI + STS-B dataset is larger ( ∼950K samples), so fewer epochs are needed;
(3) 4 epochs is standard practice in SentenceTransformers tutorials.
T. Detailed Derivations
T.1. Relationship Between Similarity Functions
The four similarity functions can be understood through a unified framework. Let q,d∈Rnbe the query and document
embeddings. We can write:
sdot(q,d) =q⊤d=∥q∥∥d∥cosθ(11)
scos(q,d) = cosθ(12)
sq-norm(q,d) =∥d∥cosθ(13)
sd-norm(q,d) =∥q∥cosθ(14)
whereθis the angle betweenqandd.
This decomposition reveals that:
24

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
• Cosine similarity captures only the angular component
• Query-only normalization additionally captures document magnitude
• Document-only normalization additionally captures query magnitude
• Dot product captures both magnitudes and angle
T.2. Gradient Analysis
We provide a detailed analysis of the gradient dynamics for cosine and dot product similarity, explaining why the normaliza-
tion constraint fundamentally limits magnitude learning.
Dot Product Gradient.For the InfoNCE loss with dot product similarity, the gradient with respect to the query embedding
qis:
∂Ldot
∂q=−d++X
jpjdj,(15)
where pj=exp(αq⊤dj/τ)P
kexp(αq⊤dk/τ)is the softmax probability, and d+is the positive document. This gradient operates in the full
Rnspace and can adjust both the direction and magnitude ofq.
Cosine Similarity Gradient and Tangent Space Projection.For cosine similarity scos(q,d) = ˆq⊤ˆdwhere ˆv=v/∥v∥ ,
the gradient computation involves the Jacobian of the normalization operation. We derive this Jacobian explicitly.
Derivation of the Normalization Jacobian.Let ˆv=v/∥v∥. The(i, j)-th element of the Jacobian matrixJ=∂ˆv
∂vis:
Jij=∂
∂vjvi
∥v∥
=δij
∥v∥−vivj
∥v∥3=1
∥v∥(δij−ˆviˆvj),(16)
whereδ ijis the Kronecker delta. In matrix form:
J=∂ˆv
∂v=1
∥v∥ 
I−ˆvˆv⊤
=1
∥v∥Pv.(17)
For a functionf( ˆv)composed with L2 normalization, the chain rule gives:
∂f
∂v=∂f
∂ˆv·∂ˆv
∂v=1
∥v∥Pv∂f
∂ˆv,(18)
whereP v=I− ˆvˆv⊤is theprojection matrix onto the tangent spaceof the unit sphere at ˆv.
Spectral Properties ofP v.The projection matrixP vhas important spectral properties:
•Idempotent:P2
v=P v(applying the projection twice yields the same result)
•Symmetric:P⊤
v=P v
•Eigenvalues:λ= 0with eigenvector ˆv(radial direction), andλ= 1with multiplicityn−1(tangent directions)
•Rank: rank(P v) =n−1, confirming the loss of one degree of freedom
The zero eigenvalue in the radial direction mathematically guarantees that no gradient signal can flow in that direction,
regardless of how the loss function depends on magnitude.
This projection has a crucial geometric interpretation: Pvremoves the component of the gradient parallel to v(the radial
direction), leaving only the component perpendicular tov(the tangential direction). Formally, for any vectorg:
Pvg=g−( ˆv⊤g)ˆv=g−projˆv(g).(19)
25

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Why Tangent Space Projection Prevents Magnitude Learning.The tangent space projection fundamentally constrains
what the model can learn:
1.Radial gradients are eliminated: Any gradient component that would change ∥v∥is projected out. Even if the loss
function would benefit from adjusting magnitude, the optimizer cannot act on this signal.
2.Optimization is confined to Sn−1: The effective optimization landscape is the (n−1) -dimensional unit hypersphere,
not the fulln-dimensional space. This reduces representational capacity by one degree of freedom per embedding.
3.Magnitude information is noise: Since gradients cannot systematically adjust magnitude, any magnitude variation in
cosine-trained models reflects initialization artifacts or numerical effects, not learned representations. This explains why
CLIP (trained with normalized cosine) shows Cohen’sd≈0.
Dot Product: Unconstrained Optimization.In contrast, the dot product gradient (Eq. 15) operates in the full Rnspace:
1.Radial gradients are preserved: The gradient can increase or decrease∥q∥and∥d∥based on the loss signal.
2.Magnitude encodes relevance: When the positive document d+has large magnitude, the gradient −d+pullsqtoward
high-magnitude regions. Over training, this creates the magnitude-relevance correlation we observe (Cohen’sd >0).
3.Effective temperature modulation: The softmax probabilities pj∝exp(αq⊤dj/τ)depend on both direction and
magnitude. High-magnitude queries produce sharper distributions, effectively lowering the temperature for “confident”
queries.
Geometric Interpretation.Geometrically, cosine similarity constrains optimization to move along the surface of the
unit sphere (geodesics), while dot product allows optimization to move freely in the ambient space, including radially
inward/outward. This difference, illustrated in Figure 1, is why dot product enables magnitude learning while cosine blocks
it.
From Observations to Insights.Combining the gradient analysis (established theory) with our experimental findings
yields two insights about when and how magnitude can be useful:
Insight 1: Task Symmetry Constrains Whether Magnitude Can Be Exploited.
It is well known that cosine similarity enforces sim(a, b) =sim(b, a) , while partial normalization (QNorm, DNorm) breaks
this symmetry. Our experiments reveal thepractical consequence: for symmetric tasks (e.g., STS), breaking this symmetry
causes catastrophic performance degradation (Appendix F.1), while for asymmetric tasks (e.g., retrieval), relaxing the
unit-norm constraint allows magnitude to serve as an additional degree of freedom that encodes task-relevant information.
This suggests a practical guideline:
•Symmetric tasks(paraphrase detection, duplicate detection, clustering): Cosine similarity is appropriate; magnitude
freedom provides no benefit and partial normalization is harmful.
•Asymmetric tasks(retrieval, recommendation, QA): Magnitude can be exploited as an additional learning signal.
Whether to use Dot, QNorm, or DNorm depends on the specific task characteristics and should be determined empirically.
Insight 2: In Asymmetric Tasks, Candidate-Side Magnitude Accumulates Relevance Signal.
The InfoNCE gradient w.r.t. the positive document is ∇d+L=−(1−p+)q, which is proportional to the query vector. Over
training, a document relevant tomany diverse queriesreceives gradient contributions from multiple directions, causing its
magnitude to grow. We term this the “relevance counter” effect.
This intuition is supported by the theoretical framework in Section 4: Proposition 4.1 shows that only document magnitude
affects inference-time ranking, and Proposition 4.2 shows that query magnitude modulates training dynamics via effective
temperature. Our retrieval experiments confirm this mechanism: documents exhibit Cohen’s d >0 (magnitude correlates
with relevance), while queries show weaker or inconsistent magnitude-relevance correlation.
26

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Practical Recommendations.Based on these insights, we offer the following recommendations:
1.For symmetric tasks: Use cosine similarity. Do not use partial normalization (QNorm/DNorm).
2.For asymmetric tasks: Consider removing the unit-norm constraint to allow magnitude learning. The choice among
Dot, QNorm, and DNorm should be based on validation performance, as the optimal choice varies by model and task.
3.Diagnostic tool: Compute Cohen’s dbetween relevant and irrelevant candidate magnitudes. If d≈0 , magnitude is not
being utilized; ifd >0.5, magnitude carries significant relevance signal.
4.Testable prediction: The “relevance counter” mechanism predicts that in recommendation systems, item embeddings
trained without normalization should exhibit magnitude correlated with item popularity; we leave this verification to
future work.
Connection to Representational Flexibility.The tangent space constraint has implications beyond magnitude learning:
•Loss landscape: The cosine loss landscape is defined on a compact manifold ( Sn−1), while the dot product landscape
extends to infinity. This affects optimization dynamics, including the possibility of escaping local minima by moving
radially.
•Implicit regularization: Cosine similarity implicitly regularizes representations to unit norm, which may help or hurt
depending on whether magnitude carries task-relevant information. Our experiments suggest this regularization is
harmful for asymmetric tasks (retrieval) but neutral for symmetric tasks (STS).
•Scale invariance: Cosine similarity is scale-invariant ( scos(cq,d) =s cos(q,d) forc >0 ), which can aid optimization
stability but prevents the model from using scale to encode information. Dot product’s scale-sensitivity is precisely what
enables magnitude-based relevance encoding.
T.3. Learnable Normalization: Gradient Analysis
We extend the gradient analysis to the learnable normalization framework introduced in Section 2.4. Recall that the
parameterized similarity is:
sγq,γd(q,d) =q⊤
∥q∥γq·d
∥d∥γd=∥q∥1−γq· ∥d∥1−γd·cosθ,(20)
whereγ q, γd∈[0,1]control the degree of query-side and document-side normalization, respectively.
Proposition T.1(Normalization Gradient).The partial derivatives of sγq,γdwith respect to the normalization exponents
are:
∂ sγq,γd
∂γq=−ln∥q∥ ·s γq,γd(q,d)(21)
∂ sγq,γd
∂γd=−ln∥d∥ ·s γq,γd(q,d)(22)
Proof.From Eq. (20),s=∥q∥1−γq· ∥d∥1−γd·cosθ. Taking the derivative with respect toγ q:
∂s
∂γq=∂
∂γq∥q∥1−γq· ∥d∥1−γdcosθ=−ln∥q∥ · ∥q∥1−γq· ∥d∥1−γdcosθ=−ln∥q∥ ·s
The derivation forγ dis analogous.
Interpretation.These gradients have a natural interpretation:
•Self-regulating normalization: When magnitude is large ( ∥q∥>1 , hence ln∥q∥>0 ), the gradient pushes γqin the
direction that reduces the influence of that magnitude. The model learns the degree of normalization that yields the best
trade-off between retaining magnitude information and suppressing noise.
27

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
•Asymmetric by construction: γqandγdreceive gradients modulated by ln∥q∥ andln∥d∥ respectively, which have
different distributions. This means the model canindependentlylearn different normalization degrees for each side,
which is exactly the asymmetry our ablation framework was designed to study.
•Connection to Proposition 4.2: At the QNorm vertex ( γq= 1, γ d= 0), the effective temperature becomes τeff(d) =
τ/∥d∥ , and at the DNorm vertex ( γq= 0, γ d= 1),τeff(q) =τ/∥q∥ . The learnable framework interpolates smoothly
between these regimes.
Proposition T.2(Generalized Ranking Equivalence).For a fixed query q, the ranking πγq,γdinduced by sγq,γddepends
only onγ d:
πγq,γd=πγ′q,γd∀γq, γ′
q∈[0,1].(23)
That is, query-side normalization degree does not affect inference-time ranking. The ranking continuously interpolates
between Cosine-type (γ d= 1, ranking bycosθalone) and Dot-type (γ d= 0, ranking by∥d∥cosθ), weighted by∥d∥1−γd.
Proof. For a fixed query, ∥q∥1−γqis a positive constant that scales all document scores uniformly, and therefore does not
affect their relative ordering.
Implications.Proposition T.2 generalizes Proposition 4.1 from discrete to continuous: while γqdoes not affect inference,
it shapes training dynamics (through effective temperature modulation). Thus the two exponents serve fundamentally
different roles: γdcontrolswhatmagnitude information is used at inference, while γqcontrolshowtraining gradients are
modulated.
28

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 15.Complete Evaluation Results for E5 (MS MARCO 500K, seed=0). N = NDCG@10, R = Recall@100, M = MRR@10.Bold=
best among fine-tuned models, underline = second best. With 6 ×more training data, DNorm becomes the best method on BRIGHT and
Multi-hop.
Pretrained Cosine Dot QNorm DNorm
Dataset N R M N R M N R M N R M N R M
In-Domain
MSM-Dev 18.13 60.11 13.98 34.32 86.76 27.65 29.28 85.22 22.66 14.94 61.57 11.21 28.17 85.95 21.64
DL-19 33.03 27.84 53.57 64.4851.42 90.50 52.95 51.6871.43 23.39 30.05 32.27 46.83 51.11 64.72
DL-20 31.97 35.81 58.36 64.42 58.64 90.74 53.68 56.97 71.58 21.38 39.65 29.44 46.04 57.72 57.78
BEIR (14 datasets)
ArguAna 35.19 91.39 27.80 47.55 97.72 38.94 4.78 31.22 3.47 18.79 72.76 13.79 42.61 95.45 34.20
Climate-FEVER 15.86 40.72 22.78 23.75 55.4631.36 13.48 39.03 17.85 24.1254.04 33.40 18.95 51.87 25.47
CQADupStack 25.64 61.07 24.15 41.04 74.92 40.02 37.55 72.68 36.36 32.96 73.53 30.70 34.22 74.83 31.88
DBPedia 18.19 32.00 38.00 38.10 49.38 72.07 19.03 37.74 33.02 15.73 37.79 26.31 25.94 49.22 44.03
FEVER 49.48 84.29 46.46 71.23 94.10 69.24 42.48 73.42 39.81 60.71 89.07 58.00 54.07 89.95 49.82
FiQA 26.18 62.98 29.91 32.5967.0239.33 30.53 67.84 36.67 27.27 69.24 29.80 31.39 70.2935.21
HotpotQA 51.06 67.29 66.70 56.68 69.3074.84 56.13 72.69 72.45 56.56 73.98 71.79 58.86 75.7074.65
NFCorpus 30.40 30.99 48.97 36.0734.0555.87 36.09 33.04 57.17 37.3233.3059.75 37.05 33.95 59.12
NQ 25.33 73.02 21.32 41.41 89.26 35.91 14.18 53.17 11.26 6.85 31.35 5.29 18.22 68.49 14.30
Quora 83.08 97.06 82.66 87.33 99.30 86.52 77.83 96.14 76.71 78.94 95.69 78.36 86.28 98.15 85.80
SCIDOCS 1.94 5.36 4.44 19.31 45.65 32.65 6.66 31.08 10.71 9.10 30.96 15.27 10.09 33.42 16.54
SciFact 41.13 85.79 35.62 70.68 96.00 66.56 47.30 78.92 43.57 49.59 83.20 45.81 58.34 87.20 54.57
TREC-COVID 31.82 3.21 60.38 58.30 10.86 86.17 31.51 3.94 53.94 29.66 2.50 53.70 40.32 3.61 71.25
Touche-2020 1.52 3.69 4.66 16.21 37.20 30.77 0.32 6.56 0.84 0.00 0.08 0.00 0.44 1.29 1.65
BRIGHT (12 datasets)
AOPS 2.58 7.47 4.98 1.41 14.163.20 1.06 4.56 1.66 0.39 2.24 1.03 2.149.00 3.82
Biology 15.29 50.94 20.72 9.21 37.82 12.15 9.76 41.60 14.84 17.38 54.11 22.93 10.59 36.76 15.08
Earth Sci. 17.69 46.08 22.11 16.76 47.30 22.30 9.45 39.16 10.05 23.06 49.91 30.75 17.08 42.79 23.06
Economics 12.24 39.33 16.18 11.05 42.10 13.30 0.33 23.00 0.38 7.06 26.75 10.09 9.56 35.45 12.40
LeetCode 14.83 48.49 14.77 15.22 40.09 15.41 2.28 10.33 2.93 8.77 20.50 10.02 16.39 47.69 16.35
Pony 1.72 10.84 5.08 0.42 4.57 0.70 5.51 24.79 8.12 22.0927.09 52.83 15.94 30.2143.80
Psychology 16.66 43.57 18.79 11.93 40.01 13.43 2.35 23.84 1.62 11.94 40.20 13.11 16.05 42.14 17.23
Robotics 10.21 26.93 15.14 8.12 29.77 11.86 2.91 20.34 4.93 7.90 30.93 10.15 12.81 37.81 18.10
StackOverflow 9.99 42.63 12.27 11.03 44.02 13.41 2.45 16.44 3.34 9.58 31.99 12.72 11.72 46.51 13.81
Sust. Living 10.36 40.36 13.50 9.72 41.2211.81 2.50 17.78 3.12 7.65 29.96 10.37 11.0038.72 14.42
TheoremQA-Q 11.24 24.57 12.77 13.5223.03 15.67 3.26 11.75 3.65 7.49 20.61 8.90 10.84 26.6113.39
TheoremQA-T 4.96 16.23 4.62 5.24 26.434.44 1.15 7.57 1.32 1.79 12.17 2.41 6.3720.29 5.96
Multi-hop (4 datasets)
2WikiMHopQA 62.43 73.30 83.71 71.73 75.9496.40 71.05 76.32 94.99 72.58 76.97 95.99 72.67 76.9896.29
MuSiQue 30.34 60.26 43.74 35.67 66.77 51.60 34.43 65.32 48.52 36.5666.78 52.32 36.16 68.5752.03
NovelHopQA 61.33 94.98 55.50 58.01 94.59 52.17 66.47 95.28 61.43 61.67 94.29 56.48 67.60 96.20 62.36
HotpotQA 51.06 67.29 66.70 56.68 69.3074.84 56.13 72.69 72.45 56.56 73.98 71.79 58.86 75.7074.65
Averages
DL Avg (2) 32.50 31.83 55.96 64.45 55.03 90.62 53.32 54.32 71.50 22.38 34.85 30.86 46.44 54.42 61.25
BEIR Avg (14) 31.20 52.78 36.70 45.73 65.73 54.30 29.85 49.82 35.27 31.97 53.39 37.28 36.91 59.53 42.75
BRIGHT Avg (12) 10.65 33.12 13.41 9.47 32.54 11.47 3.58 20.10 4.66 10.42 28.87 15.44 11.71 34.50 16.45
MHop Avg (4) 51.29 73.96 62.41 55.52 76.65 68.75 57.02 77.40 69.35 56.84 78.00 69.14 58.82 79.36 71.33
29

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 16.NDCG@10 forE5on MS MARCO 80K and 500K. PT = pretrained (before fine-tuning). Numbers in parentheses indicate the
number of subsets averaged.Bold= best, underline = second best. Shaded columns : magnitude-aware variants after removing E5’s
built-in normalization layer. With 80K training data (a), QNorm performs best on reasoning-intensive tasks; with 6 ×more data (b),
DNorm becomes the best method.
(a)MS MARCO 80K
Dataset PT Cosine Dot QNorm DNorm
In-Domain
MSM-Dev 18.1333.63 16.74 9.53 26.01
DL-19 33.0362.27 22.80 10.36 52.31
DL-20 31.9763.92 28.00 12.79 56.95
Out-of-Domain
BEIR (14) 31.2045.56 25.98 30.36 24.73
BRIGHT (12) 10.65 8.94 1.32 11.49 3.37
MHop (4) 51.29 52.55 49.01 53.00 28.00
(b)MS MARCO 500K
Dataset PT Cosine Dot QNorm DNorm
In-Domain
MSM-Dev 18.1334.32 29.28 14.94 28.17
DL-19 33.0364.48 52.95 23.39 46.83
DL-20 31.9764.42 53.68 21.38 46.04
Out-of-Domain
BEIR (14) 31.2045.73 29.85 31.97 36.91
BRIGHT (12) 10.65 9.47 3.58 10.42 11.71
MHop (4) 51.29 55.52 57.02 56.84 58.82
Table 17.Taxonomy of contrastive learning tasks by symmetry.
Type Characteristics Examples
Asymmetric Distinct roles; sim(a, b)̸=sim(b, a)Retrieval, QA, Recommendation
Symmetric Interchangeable; sim(a, b) =sim(b, a)STS, Paraphrase, Clustering
Table 18.Dot vs. Cosine similarity on validation set during training (NDCG@10, seed=0). All models trained with DotProductRank-
ingLoss.∆= Dot−Cosine at best dot step. E5 (w/ norm) retains the normalization layer; E5 (w/o norm) removes it.
Model Dot Cosine∆
Contriever 0.935 0.915 +0.020
RetroMAE 0.931 0.899 +0.031
E5 (w/ norm) 0.940 0.940 0.000
E5 (w/o norm) 0.944 0.227 +0.717
30

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 19.Per-dataset analysis for Contriever and RetroMAE. (a) Cohen’s dand∆% = (QNorm −Cosine) / Cosine ×100. Background
shading indicates Cohen’s deffect size: large (|d| ≥0.8 ),medium (0.5≤ |d|<0.8 ),small (0.2≤ |d|<0.5 ), white = negligible
(|d|<0.2). (b) Query magnitude CV (%).∆CV = DNorm CV / Dot CV .
(a) Cohen’sdand∆%
DatasetContr. Retro.
d∆% d∆%
In-Domain
trec-dl-2019 −0.06+1.5 −0.24+2.2
trec-dl-2020 −0.15−0.9 −0.39+2.1
Mean −0.11 +0.3 −0.32 +2.2
BEIR
arguana +0.02−11.6 +0.32+17.4
climate-fever +1.01+3.0 −0.56+2.4
cqadupstack-android −0.24+8.9 −0.35+14.9
cqadupstack-english +0.13+17.3 +0.34+13.5
cqadupstack-gaming −0.05+16.0 −0.04+13.1
cqadupstack-gis −0.26+20.3 −0.25+29.6
cqadupstack-mathematica −0.28+12.7 −0.15+25.3
cqadupstack-physics −0.59+8.9 −0.55+9.7
cqadupstack-programmers −0.27+16.6 −0.40+20.0
cqadupstack-stats −0.35+17.4 −0.22+27.5
cqadupstack-tex −0.41+27.9 −0.37+32.6
cqadupstack-unix −0.23+17.0 −0.16+25.0
cqadupstack-webmasters −0.14+14.4 −0.22+23.5
cqadupstack-wordpress −0.06+19.3 −0.08+19.0
dbpedia-entity +0.00+13.3 −0.17+1.1
fever +1.12+9.3 −0.44−0.6
fiqa −0.17+13.0 −0.07+10.8
hotpotqa +0.75+20.4 +0.05+15.2
nfcorpus −0.06+4.9 −0.05+5.8
nq +0.15+12.3 −0.27+4.2
quora −0.34+1.5 −0.22+1.9
scidocs −0.15+8.6 −0.07+16.5
scifact +0.08+13.7 +0.08+11.8
trec-covid +0.34+20.9 +0.06−2.8
webis-touche2020 +0.22+8.7 −0.67−5.5
Mean +0.00 +13.0 −0.18 +13.3
BRIGHT
bright-aops +1.50+26.3 +1.60+167
bright-biology +1.19+200 −0.24+51.1
bright-earth science +0.87+110 −0.99+22.9
bright-economics +1.71+80.8 +0.43+37.6
bright-leetcode +1.42+4.7 −0.75+17.0
bright-pony +1.04+208 −0.69+513
bright-psychology +1.57+96.0 +0.18+29.1
bright-robotics +1.80+123 +0.72+39.5
bright-stackoverflow +1.43+42.5 +0.57+61.6
bright-sustainable living +1.11+99.5 −0.04+65.4
bright-theoremqa q +0.46+63.4 +0.63+66.4
bright-theoremqa t +0.50+41.9 +0.26+49.7
Mean +1.22 +91.3 +0.14 +93.3
Multi-Hop
multihop-2wiki +0.64+8.1 +0.11+1.2
multihop-musique +0.13+5.9 −0.01−0.9
Mean +0.39 +7.0 +0.05 +0.2
Overall +0.38 +34.7 −0.08 +35.5(b) Query magnitude CV (%)
DatasetContriever RetroMAE
Dot DN∆ Dot DN∆
In-Domain
trec-dl-2019 5.48 10.8 2.0 0.63 3.28 5.2
trec-dl-2020 5.76 10.9 1.9 0.65 3.24 5.0
Mean 5.62 10.9 1.9 0.64 3.26 5.1
BEIR
arguana 5.48 4.86 0.9 0.28 1.96 7.0
climate-fever 5.03 7.65 1.5 0.46 3.50 7.6
cqadupstack-android 4.66 8.29 1.8 0.58 3.24 5.6
cqadupstack-english 4.96 9.03 1.8 0.59 3.77 6.4
cqadupstack-gaming 5.26 8.99 1.7 0.64 3.28 5.1
cqadupstack-gis 5.49 8.80 1.6 0.60 3.26 5.4
cqadupstack-mathematica 5.53 8.96 1.6 0.56 3.79 6.8
cqadupstack-physics 5.24 8.39 1.6 0.56 4.25 7.6
cqadupstack-programmers 5.20 8.42 1.6 0.58 3.41 5.9
cqadupstack-stats 5.78 9.04 1.6 0.54 3.53 6.5
cqadupstack-tex 5.68 9.49 1.7 0.57 3.78 6.6
cqadupstack-unix 5.17 8.96 1.7 0.64 3.46 5.4
cqadupstack-webmasters 4.87 7.72 1.6 0.60 3.66 6.1
cqadupstack-wordpress 5.05 8.49 1.7 0.57 3.44 6.0
dbpedia-entity 6.55 10.4 1.6 0.71 5.87 8.3
fever 5.67 9.27 1.6 0.51 4.39 8.6
fiqa 4.82 7.42 1.5 0.54 3.21 5.9
hotpotqa 4.90 8.66 1.8 0.45 2.44 5.4
nfcorpus 8.03 11.2 1.4 0.89 7.89 8.9
nq 5.37 9.38 1.7 0.60 2.98 5.0
quora 5.03 8.84 1.8 0.60 3.32 5.5
scidocs 4.85 7.71 1.6 0.49 2.95 6.0
scifact 5.36 7.16 1.3 0.50 3.68 7.4
trec-covid 4.47 6.78 1.5 0.42 2.10 5.0
webis-touche2020 4.39 7.32 1.7 0.47 3.05 6.5
Mean 5.37 8.24 1.5 0.54 3.64 6.7
BRIGHT
bright-aops 7.03 6.12 0.9 0.46 2.69 5.8
bright-biology 7.58 6.23 0.8 0.38 2.16 5.7
bright-earth science 7.38 6.41 0.9 0.43 2.06 4.8
bright-economics 8.18 5.41 0.7 0.37 2.02 5.5
bright-leetcode 5.79 5.70 1.0 0.28 1.70 6.1
bright-pony 4.10 3.51 0.9 0.28 1.42 5.1
bright-psychology 7.21 4.90 0.7 0.33 2.24 6.8
bright-robotics 9.26 4.79 0.5 0.43 1.69 3.9
bright-stackoverflow 9.17 5.42 0.6 0.43 1.95 4.5
bright-sustainable living 6.97 5.04 0.7 0.29 1.97 6.8
bright-theoremqa q 5.82 5.38 0.9 0.33 2.07 6.3
bright-theoremqa t 5.39 5.14 1.0 0.33 1.88 5.7
Mean 6.99 5.34 0.8 0.36 1.99 5.5
Multi-Hop
multihop-2wiki 4.53 7.98 1.8 0.54 2.52 4.7
multihop-musique 5.04 7.88 1.6 0.48 2.42 5.0
Mean 4.79 7.93 1.7 0.51 2.47 4.9
Overall 5.77 7.60 1.3 0.50 3.04 6.1
Table 20.Statistics of training datasets. Length statistics are measured in characters (Ave = Average, Min = Minimum, Max = Maximum).
“—” indicates not applicable (MS MARCO Passage Ranking has no answer annotations as it is a retrieval task).
Dataset #Samples #Queries #DocsQuery Length Doc Length Answer Length
Ave Min Max Ave Min Max Ave Min Max
MS MARCO v1.1 QA 82,326 82,326 80,304 34.0 8 144 478 38 3,570 82.2 0 990
MS MARCO Passage Ranking 502,939 502,939 489,288 33.2 5 215 370 25 3,602 — — —
31

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Type Dataset #Queries #DocsQuery Length Doc Length Docs/ Queries/
Ave Min Max Ave Min Max Query Doc
In-Domainmsmarco-dev 6,980 8,841,823 33 9 186 335 3 1,665 1.07 1.00
trec-dl-2019 43 8,841,823 33 16 55 335 3 1,665 215 1.01
trec-dl-2020 54 8,841,823 34 12 70 335 3 1,665 211 1.01
BEIRarguana 1,406 8,674 1,193 251 5,500 1,030 2 6,673 1 1.00
climate-fever 1,535 5,416,593 123 26 406 539 1 374,597 3.05 3.48
cqadupstack 13,145 457,199 50 15 149 932 41 43,874 1.8 1.00
dbpedia-entity 400 4,635,922 34 6 88 310 8 42,899 109 1.07
fever 6,666 5,416,568 50 14 189 539 1 374,597 1.19 5.29
fiqa 648 57,638 63 16 147 767 0 16,990 2.63 1.00
hotpotqa 7,405 5,233,329 92 32 288 289 8 8,276 2 1.07
nfcorpus 323 3,633 22 3 72 1,591 123 10,090 38 3.94
nq 3,452 2,681,468 48 25 100 493 4 17,008 1.22 1.00
quora 10,000 522,931 52 2 258 62 1 1,169 1.57 1.00
scidocs 1,000 25,657 72 16 206 1,204 10 10,169 30 1.17
scifact 300 5,183 90 28 204 1,499 221 10,127 1.13 1.20
trec-covid 50 171,332 69 30 165 1,118 0 122,459 1,327 1.87
webis-touche2020 49 382,545 43 16 83 1,720 3 106,072 45 1.05
BRIGHTaops 111 188,002 320 85 1,167 754 58 7,334 4.72 4.72
biology 103 57,359 523 89 2,195 330 1 31,131 3.61 1.00
earth science 116 121,249 477 83 1,565 338 2 233,623 5.04 1.00
economics 103 50,220 740 164 2,223 395 3 39,672 7.77 1.00
leetcode 142 413,932 1,459 422 3,964 1,059 75 103,665 1.85 1.21
pony 112 7,894 389 182 946 260 8 2,583 19.851.6
psychology 101 52,835 693 166 2,334 384 3 226,941 6.85 1.01
robotics 101 61,961 2,180 165 19,341 291 3 28,640 5.15 1.00
stackoverflow 117 107,081 1,293 185 12,432 1,715 1 4,000 4.09 1.01
sustainable living 108 60,792 683 158 2,843 344 1 158,299 5.33 1.00
theoremqa questions 194 188,002 426 84 1,255 754 58 7,334 3.18 1.90
theoremqa theorems 76 23,839 416 159 846 874 74 19,106 1.99 1.59
Multi-hop2wikimultihopqa 12,576 125,237 68 23 179 377 18 9,780 2.44 1.00
musique 2,417 48,315 102 26 275 524 116 2,044 2.65 1.00
novelhopqa 4,345 4,345 138 43 334 2,336 152 42,658 1 1.00
hotpotqa 7,405 5,233,329 92 32 288 289 8 8,276 2 1.07
Table 21.Statistics of evaluation benchmarks. Length statistics are measured in characters (Ave = Average, Min = Minimum, Max =
Maximum). Docs/Query indicates the average number of relevant documents per query; Queries/Doc indicates the average number of
queries each document is relevant to. The pony dataset has an exceptionally high Queries/Doc ratio (51.6), indicating that each document
is relevant to many queries on average. This “hub” structure may affect magnitude learning patterns and explains its outlier behavior in
Figure 3. HotpotQA appears in both BEIR and Multi-hop categories.
Table 22.Seed sensitivity analysis (3 random seeds). We report mean and standard deviation of validation NDCG@10. Low standard
deviations ( <0.06 ) for Cosine, Dot, DNorm, and Learnable indicate stable convergence. QNorm exhibits higher variance due to one seed
showing different early-training dynamics.
(a)Contriever
Method Val NDCG@10 Std
Cosine 92.84±0.03
Dot 93.49±0.02
QNorm 92.87±1.02
DNorm 93.55±0.02
Learnable 93.59±0.01(b)RetroMAE
Method Val NDCG@10 Std
Cosine 91.88±0.02
Dot 93.02±0.06
QNorm 92.45±1.03
DNorm 93.06±0.04
Learnable 93.08±0.02
32

Beyond the Unit Hypersphere: Embedding Magnitude in Contrastive Learning
Table 23.Statistical analysis of Contriever results across benchmark categories (3 random seeds). We report mean ±standard deviation of
NDCG@10 and paired t-test p-values comparing magnitude-aware methods against Cosine. Bold indicates the best mean;∗indicates
p <0.05;∗∗indicatesp <0.01.
Benchmark Cosine Dot QNorm DNorm
In-Domain
TREC-DL(2) 56.95±0.54 56.50±0.0657.45±0.3056.29±0.18
Out-of-Domain
BEIR(14) 40.96±0.50 43.48±0.13∗44.01±0.27∗42.45±0.16
BRIGHT(12) 7.41±0.26 12.53±0.17∗∗12.74±0.20∗∗9.83±0.26∗∗
MultiHop(4) 51.37±0.8458.16±0.20∗∗57.38±0.15∗∗57.54±0.16∗∗
Paired t-test p-values (vs. Cosine):
TREC-DL(2) –p= 0.324p= 0.240p= 0.238
BEIR(14) –p= 0.012p= 0.012p= 0.059
BRIGHT(12) –p= 0.001p= 0.002p= 0.010
MultiHop(4) –p= 0.003p= 0.005p= 0.009
Table 24.Hyperparameter comparison between retrieval and STS experiments. Differences arise from task characteristics and loss
function requirements.
Hyperparameter Retrieval STS
Loss function InfoNCE MSE
Learning rate2–5×10−62×10−5
Batch size 128 64
Epochs 100 4
Warmup ratio 0 0.1
Scale factor (λ) 20 0.01
Training data MS MARCO (80K) AllNLI + STS-B (∼950K)
Evaluation NDCG@10 Spearman correlation
33