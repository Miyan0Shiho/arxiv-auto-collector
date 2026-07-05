# Relevance Is Not Permission: Warranted Attention for Value Contributions

**Authors**: Minwoo Yu, Young-guk Ha

**Published**: 2026-06-29 11:19:13

**PDF URL**: [https://arxiv.org/pdf/2606.30139v2](https://arxiv.org/pdf/2606.30139v2)

## Abstract
Relevance is not permission. Attention lets a model read key-value items related to the current query, but it does not guarantee that the value contribution of such an item becomes prediction evidence. A retrieved passage may be relevant to a question without being supporting evidence, and a historical fact or temporal neighbor may even blur true-tail ranking or the current edge score. This paper formalizes this gap as a permission problem for the weighted value term alpha_ij * v_j that is actually added to the prediction path. We propose Warrant, a path-localized interface that preserves attention relevance alpha_ij, exposes the value path leading to the primary metric, and, in the full model, turns alpha_ij * v_j into alpha_ij * g_ij * v_j through learned query-item permission g_ij. We place the same operator on the metric-defining value paths of CTDG link prediction, MTPP next-mark ranking, RAG supporting evidence selection, STPP next-location forecasting, and TKG tail prediction. Across 32 paired comparisons, 3 seeds, and 192 total runs, Warrant improves the primary metric in 27 comparisons; practical tiers consist of 10 substantial effects, 1 marginal effect, 8 positive but uncertain effects, 8 tie/negligible effects, and 5 drops. In the path-localization check, correct-path placement outperforms direction-aware Base performance in every domain and exceeds generic attention placement by +0.1076 AUC in CTDG and +0.0683 MRR in TKG. Ablations show that most TKG gains come from historical-tail value path exposure, whereas the core CTDG gain comes from edge-conditioned query-item permission. In conclusion, prediction evidence is not attention mass. A weighted value term becomes evidence only when it is warranted on the path to the metric.

## Full Text


<!-- PDF content starts -->

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 1
Relevance Is Not Permission: Warranted
Attention for Value Contributions
Minwoo Yu and Y oung-guk Ha
Abstract—Relevance is not permission. Attention lets a model read key-value items related to the current query, but it does not ensure
that the value contribution of the item is valid prediction evidence. A retrieved passage can be relevant to a question without being
supporting evidence, and a historical fact or temporal neighbor can even obscure true-tail ranking or the current edge score. This paper
formalizes this gap as a permission problem for the weighted value termα ijvjthat is actually added to the prediction path. We propose
Warrant, a path-localized interface that preserves attention relevanceα ij, exposes the value path leading to the primary metric, and, in
the full model, replacesα ijvjwithα ijgijvjusing learned query-item permissiong ij. We place the same operator on metric-defining
value paths for CTDG link prediction, MTPP next-mark ranking, RAG supporting evidence selection, STPP next-location forecasting,
and TKG tail prediction. Across 32 paired comparisons with three seeds, for a total of 192 runs, Warrant improves the primary metric in
27 comparisons; practical tiers are 10 substantial, 1 marginal, 8 positive but uncertain, 8 tie/negligible, and 5 drops. In path-localization
checks, correct-path placement exceeds direction-aware Base performance in every domain and outperforms generic attention
placement by +0.1076 AUC in CTDG and +0.0683 MRR in TKG. Ablations show that most of the TKG gain comes from exposing the
historical-tail value path, whereas the core CTDG gain comes from edge-conditioned query-item permission. Overall, prediction
evidence is not attention mass. A weighted value term becomes evidence only when it is warranted on the path to the metric.
Index Terms—Attention weighted value term, path-localized contribution, path exposure, query-conditioned permission, evidence
contribution.
✦
1 INTRODUCTION
Attention computes relevance between a query and key-
value items, then incorporates the values of highly relevant
items into a prediction representation. Since the Trans-
former, attention has become a common computational unit
beyond sequence modeling, appearing in temporal graph
learning, retrieval-augmented reasoning, temporal point
processes, and temporal knowledge graph forecasting [1],
[2], [3], [4]. Across domains, the item read by attention may
look different: a temporal neighbor, a past marked event, a
retrieved passage, a spatio-temporal history element, or a
historical temporal fact. In all cases, however, it is a key-
value item read for the current prediction.
This work starts from the observation that attention
relevance and prediction evidence are not the same con-
cept. Attention estimates whether an item is lexically, se-
mantically, temporally, structurally, or historically related
to the current query. Prediction loss, however, requires not
relevance itself but an effective contribution that supports
the final prediction. In RAG, a retrieved passage can be se-
mantically related to the question without being supporting
evidence. In temporal knowledge graphs, a historical fact
can be structurally connected to the current temporal query
without supporting the true-tail ranking. In continuous-
time dynamic graphs, a past interaction can be related to
the current edge while being stale or reinforcing a hard
negative. In marked temporal point processes, a past event
can be useful for one candidate mark but become a shortcut
•M. Yu and Y.-g. Ha are with the Smart Computing Laboratory, De-
partment of Computer Science & Engineering, Konkuk University, Seoul
05029, Republic of Korea. E-mail:{snowypainter, ygha}@konkuk.ac.kr.
•Y.-g. Ha is the corresponding author.for another. Thus, a relevant key-value item is not always
entitled to affect the prediction.
To address this problem, we propose Warrant. War-
rant starts from the read operation produced by existing
attention, retrieval, memory access, or temporal-neighbor
aggregation, and explicitly models the remaining metric-
facing contribution path. Specifically, it exposes the item-
wise value contribution already formed by attention on the
path that leads to the primary metric. In the full instan-
tiation, it controls whether the contribution may enter the
prediction-facing path through query-conditioned permis-
sion. In other words, attention decides what the model can
read, whereas Warrant decides on which metric path the
read value contribution is exposed and how much it should
affect the current prediction. This separation is important
when an item is relevant but can produce a destructive
or noisy contribution, such as stale, shifted, shortcut, or
ambiguous items.
Figure 1 summarizes the central distinction. Attention
decides which item a query can access, whereas Warrant
re-decides whether the read weighted value contribution
is qualified to enter the metric-facing prediction object as
evidence through path-localized permission.
The key idea of Warrant is not the mere presence of
a gate, but the identification and exposure of the metric-
defining value path. Changing attention scores or the re-
trieved candidate set mixes relevance estimation with the
contribution decision again. In contrast, Warrant preserves
the value contribution formed by attention and separates
path exposure from learned permission on the value path
leading to the primary metric. We define this placement
principle as path localization. Rather than attaching a gate
to an arbitrary attention layer, we find the path througharXiv:2606.30139v2  [cs.AI]  1 Jul 2026

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 2
a Attention Access
query
qi✓
!
▷ ◁item 1:
support / true evidence
item 2:
relevant distractor
item 3:
stale history
item 4:
weak but useful signalαi1
αi2
αi3
αi4
hi=X
jαijvj
αij: relevance / access
cij=αijvj: read-derived contribution
Relevant̸=evidentialb Warrant Admission
read-derived
contribution
cij=αijvjquery-item
permission
gij=
g(qi, kj)warranted
evidence
contribution
cW
ij=
αijgijvjmetric-
facing
prediction
object
score / logit /
state /
copy masspath
exposure
hW
i=X
jαijgijvj
•••item-wise ♦query-conditioned⊘no attention
renormalization
PathExposureOnly:expose path,g= 1
Full Warrant:expose path + learnedg ijc Formulation Comparison
Post-attention GLU
yi=F
X
jαijvj

after aggregation
item identity already mixed
Attention-logit gate
˜αij= softmax j(sij+ logr ij)
changes relevance
redistributes mass
Warrant
yi=X
jαijg(qi, kj)vj
before aggregation
gates contribution, not relevance
F X
jαijvj!
̸=X
jαijg(qi, kj)vj
d Path Localization Across Domains
αijvj−→α ijg(qi, kj)vj
CTDG:
history term→
edge scoreMTPP:
past event term→
mark logitRAG:
passage term→
support logitSTPP:
spatial history→
location stateTKG:
historical fact→
tail score / copy path
Apply Warrant only where the weighted value term reaches the reported metric.
Fig. 1. Main message of Warrant. Attention gives access to relevant key-value items, while Warrant admits read-derived weighted value contributions
on metric-facing paths through query-item permission. The same operation is localized across CTDG, MTPP , RAG, STPP , and TKG.
which the actual value contribution flows into the prediction
object that determines the task metric, and place a value-
contribution interface on that path.
Warrant also differs from generic gating, modulation,
and routing in its computational target. Highway and recur-
rent gates control information flow through hidden states
or layer transformations, FiLM modulates feature channels
condition-wise, and MoE routes examples to expert paths
[5], [6], [7], [8], [9]. The target controlled by Warrant is
not the entire hidden state, a feature channel, or an expert
path, but the prediction-facing value contribution produced
item by item by attention. Therefore, the effect of Warrant
must be evaluated as a specific placement of a contribution
interface on a metric-defining value path, rather than as an
effect of extra parameters or generic feature modulation.
We evaluate this idea in five task families: CTDG, MTPP ,
RAG, STPP , and TKG. The main benchmark compares
each existing model as a baseline with a paired variant
that adds Warrant to the same model’s metric-facing value
path. The full evaluation consists of 32 paired comparisons,
three seeds, and 192 completed runs; Warrant improves the
primary metric in 27 of the 32 comparisons. This count
is a direction-level summary rather than evidence of a
uniformly strong effect, and the practical tiering separately
distinguishes substantial effects, marginal effects, positive
but uncertain effects, tie/negligible effects, and drops. Path-
localization experiments show that placing Warrant on the
value path leading to the task metric outperforms the base-
line in every domain and produces much larger gains than
placing it at a generic attention location, especially in CTDGand TKG. Operator ablations, mass diagnostics, counter-
factual interventions, and CTDG controls provide evidence
that the improvement comes from query-item permission on
the metric-facing value path, not from a simple parameter
increase or attention-mass redistribution.
The contributions of this paper are as follows.
•We define the weighted value contribution produced
item by item by attention as the basic computational
unit of prediction evidence, and propose a path-
localized value-contribution interface that separates
path exposure from learned permission.
•We present a value-path adaptation methodology
that separates attention relevance from evidence con-
tribution.
•We apply the same operator to CTDG, MTPP , RAG,
STPP , and TKG, and propose a path-localization pro-
cedure that identifies and targets each task’s metric-
defining value path.
•Through a 192-run main benchmark, path localiza-
tion, CTDG edge-query ablation, TKG operator ab-
lation, and mass diagnostics, we report both per-
formance improvements and evidence for how the
path-localized value-contribution interface operates
internally.
2 RELATEDWORK
This section distinguishes Warrant from five lines of prior
work. Attention and memory access select information; gat-
ing and modulation control hidden states, feature channels,

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 3
activations, or expert paths; retrieval and reranking handle
candidate context before or after reading; and calibration
and selective prediction adjust final confidence or rejection.
Warrant targets a different computational object. After at-
tention has already estimated relevance, Warrant decides
whether the attention-weighted value contribution may af-
fect the current prediction before the prediction is formed.
2.1 Attention and Memory Access
Bahdanau attention alleviates the fixed-length context bot-
tleneck by allowing a decoder to softly search relevant parts
of a source sentence, and the Transformer formalizes atten-
tion as a query-key-value weighted-sum operation [1], [10].
End-to-End Memory Networks apply recurrent attention
over a large external memory [11]. The central question
in this family is which item or memory slot the current
query should read. Warrant does not choose the read target
again. Instead, it separately asks whether the item-wise
value contribution produced by attention or memory access
may enter the prediction path.
2.2 Gating and Modulation
Gating is a general principle for controlling information
flow in neural computation. Highway Networks and recur-
rent gates control flows through transformed inputs, car-
ried inputs, memory-cell updates, and hidden-state updates,
while GLU gates convolutional activations [5], [6], [7], [12].
FiLM applies feature-wise affine modulation from a con-
ditioning signal, squeeze-and-excitation reweights feature
channels, and MoE selects expert computation paths with
a gating network [8], [9], [13]. Warrant also uses a gate.
However, its gated object is an attention-derived item-wise
value contribution, which is distinct from hidden states,
channels, activations, and expert routes.
2.3 Gated Attention Variants
The Gated-Attention Reader builds query-specific token
representations through multiplicative interactions between
query and document representations; GTrXL introduces
gating into Transformer architectures for training stability;
and Mega combines moving averages with gated attention
for long-sequence modeling [14], [15], [16]. Sparse or hard
attention changes the attention support or probability dis-
tribution itself, and pointer-generator models control the
mixture between generation and copy branches [17], [18],
[19]. These methods improve the expressiveness, stability,
or sequence-modeling capability of attention architectures.
Warrant does not replace the attention architecture. It pre-
serves the attention weight as a relevance estimate and adds
a permission variable to the subsequent value contribution.
Table 1 summarizes representative gate families by their
gated objects. Prior work has used gates widely inside and
around attention models. The difference of Warrant lies in
both the computational unit to which the gate is applied
and the path on which that unit resides. Warrant does not
renormalize the attention distribution; it gates the item-wise
attention-weighted value term on the metric-facing value
path.2.4 Retrieval, Reranking, and Calibration
Dense retrieval fetches candidate passages, RAG and FiD
integrate retrieved passages into generation, and long-
context models such as Longformer enable processing of
extended contexts [4], [20], [21], [22]. Passage reranking
reorders retrieved candidates using query-passage scores
[23]. Calibration and selective prediction adjust confidence
after prediction or reject low-confidence cases [24], [25].
Retrieval and reranking mainly change the candidate set or
order before or after reading, whereas calibration handles
post-prediction confidence. Warrant is placed between these
stages. After attention has formed a weighted contribution,
Warrant computes permission before that contribution en-
ters the metric-facing prediction path.
2.5 Positioning of Warrant
Prior work has decided what to retrieve, what to attend to,
which hidden feature to modulate, which expert to route to,
or how much to trust the final prediction. Warrant decides
whether an already attention-weighted value contribution
may affect the current prediction. Thus, the distinction of
Warrant is not a redefinition of attention relevance, but a
separation of relevance estimation and contribution permis-
sion into different computational decisions.
3 PROBLEMDEFINITION
Attention-based models repeat the same form of compu-
tation across many domains. A queryq irepresents the
current prediction request, and the model reads candidate
history, retrieved context, memory items, or historical facts
as key-value items(k j, vj). Standard attention computes
compatibility between the query and the key, then produces
a softmax-normalized relevance weight.
αij= softmax j(s(qi, kj)).
It then gathers values into a prediction representation with
a weighted sum.
hi=X
jαijvj.
In this paper, a key-value item is the basic unit of attention
aggregation. Depending on the domain, the item may be a
temporal neighbor, a past marked event, a retrieved pas-
sage, a past spatio-temporal event, or a historical temporal
fact. Table 2 summarizes each task’s key-value item and
metric-defining value path.
The object whose contribution must be decided.: In
standard attention, the term that itemjactually adds to the
prediction representation is neither the key nor the value
alone, but
αijvj.
We define this term as theweighted value term. The attention
weightα ijindicates how relevant the item is to the current
query, but what enters the final prediction is not scalar
relevance itself; it is the value contribution multiplied by
relevance. Therefore, from the perspective of prediction, the
important question does not end with “which item is rele-
vant?” A more precise question is “may the already formed
weighted value termα ijvjenter the current prediction state
or score?”

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 4
TABLE 1
Taxonomy of prior work by gated object. The distinction of Warrant lies in the combination of an item-wise weighted value term and a metric-facing
value path.
Family Gated object Gate unit Attention renormalized Difference from Warrant
Highway / recurrent layer, carry, or memory update vector / channel no layer-level flow
GLU / gated FFN hidden activation token / channel no activation-level gate
SE / channel attention feature channel channel no channel-level gate
Attention-logit / sparse attention relevance score or support item logit / item set yes changesα ijand redistributes mass
MoE gate expert output mixture expert mixture-normalized expert-routing gate
Copy / pointer gate generation-copy branch branch / candidate branch-normalized task branch gate
Post-attention gateh i=P
jαijvj aggregate vector no item identity is already mixed
Warrantα ijvj query-item weighted value term no gates metric-facing contribution directly
TABLE 2
Key-value items and metric-defining value paths by task. Warrant uses the same permission operator across domains and targets the weighted
value path that reaches the primary metric.
Domain Key-value item Prediction target Metric-defining value path
CTDG temporal neighbor / interaction history current source-destination edge edge score
MTPP past marked event next event mark candidate mark logit
RAG retrieved passage supporting evidence passage support ranking logit
STPP past spatio-temporal event next event location next-location dynamics state
TKG historical temporal fact tail entity tail ranking score / copy path
Relevance is not permission.: Existing attention binds
relevance and contribution permission into the same scalar
αij. Once an item receives high relevance, its value term en-
ters the representation through the same weight. However,
the task loss requires not relevance itself but an evidence
contribution that supports the current label, ranking, loca-
tion, event mark, or edge score. These two notions diverge
across domains.
•In RAG, a retrieved passage can be relevant because
it has lexical or semantic overlap with the question,
but it may not be supporting evidence for the answer.
•In temporal knowledge graphs, a historical fact can
be connected to the current(head, relation, time)
query without being evidence that raises the current
true tail.
•In continuous-time dynamic graphs, a past interac-
tion can be related to the source or destination while
being stale, or while reinforcing a hard negative or
outdated preference for the current edge prediction.
•In marked temporal point processes, the same past
event can be transition evidence for one candidate
mark but a spurious shortcut for another.
•In spatio-temporal forecasting, a nearby past location
can be relevant history, but if reflected too strongly
it can create local copy bias rather than next-location
dynamics.
Thus, relevance explains why an item should be read, but it
does not guarantee that the value of the item is entitled to
affect the current prediction.
Why this problem matters.: This distinction matters
because the attention output is a computational path that
directly reaches the downstream metric. Onceα ijvjenters
a representation, candidate score, copy mass, support logit,
or dynamics state, the term functions as prediction evidence
rather than merely retrieved context. If a relevant but non-
evidential item enters this path, the model can rank adistractor passage as support, use stale temporal history
as evidence for the current edge, copy a repeated historical
tail as if it were the current answer, or reinforce local bias
in next-location prediction. Conversely, if a valid evidence
term does not pass through sufficiently, the model can find
the item with attention yet fail to improve the primary
metric.
Therefore, the problem is not to normalize attention
better, but to decide contribution qualification on the metric-
facing value path. Merely changing attention mass entangles
the effect of using one item less with the effect of redistribut-
ing that mass to other items. Changing the retrieved set,
memory access, temporal-neighbor sampling, or historical-
fact retrieval itself also entangles relevance estimation with
evidence contribution. What is needed is a separate inter-
face that preserves the weighted value term produced by
attention while deciding, conditioned on the query, whether
that term may enter the current prediction-facing path.
Problem statement.: For each prediction requesti
and key-value itemj, base attention forms a weighted value
termα ijvj. The problem addressed in this paper is the
following permission decision. Given(q i, kj, αijvj), should
this weighted value term affect the current prediction? The
solution is not to hand-code domain-specific rules, but to
learn contribution permission between the current query
and item from the task loss. The method that follows there-
fore keepsα ijas relevance, places a separate permission
variable on the weighted value term, and learns which
contributions become prediction evidence on the value path
leading to the primary metric.
4 METHOD
Warrant uses the item-wise attention-weighted value term
produced by standard attention as its computational target.

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 5
This choice distinguishes it from methods that gate attention
logits, hidden states, or post-attention aggregates.
cij=αijvj.(1)
The full Warrant replaces this term before aggregation, on
the value path leading to the primary metric, as follows.
cW
ij=αijgijvj, hW
i=X
jcW
ij.(2)
Thus, the distinction of Warrant comes from the object
and path to which the gate is applied. Attention relevance
αijis not softmax-renormalized, and the empirical effect
decomposes into metric-facing path exposure and learned
query-item permission. This section describes the interface,
theoretical properties, path localization, and domain adap-
tation. Figure 2 summarizes the full interface.
4.1 Path-Localized Warrant Interface
For each query-item pair, the full Warrant interface com-
putes a scalar contribution score as in Eq. (3).
ψ(qi, kj) =w⊤GELU(W qqi+W kkj) +b.(3)
This score is transformed into the leak-sigmoid contribution
scaling variable in Eq. (4).
gij=λ+ (1−λ)σ(ψ(q i, kj)).(4)
Thusg ij∈[λ,1], whereλis a leak factor that prevents any
valid key-value item from being removed by a hard zero.
The default value throughout this paper is0.05. Warranted
aggregation is given by Eq. (5).
hW
i=X
jαijgijvj.(5)
The scalarg ijis the learned-permission instantiation of
the interface. For decomposition, thePathExposureOnlydi-
agnostic opens the same localized path but fixesg= 1,
isolating path exposure. In the full interface,g ijis learned
from the query-key pair. Through the task loss, the model
adjusts its own query and key representations and forms
features that allow it to infer contribution scaling for the
corresponding weighted value term. No domain-specific
rule is injected directly intog ij. Warrant controls the post-
attention term without renormalizing attention logits. This
placement separates the decision to read a relevant item, the
decision to expose a metric-facing path, and the decision to
scale the contribution on that path; it also avoids forcing the
attention mass of a down-weighted item to be redistributed
as relevance to other items.
Difference from generic gates.: The mathematical
distinction of Warrant comes less from the sigmoid or mul-
tiplicative gate itself than from the computational unit and
index to which the gate is applied. A Highway gate [5] typ-
ically mixes layer-wise information flow between a trans-
formed path and a carry path,
y=T(x)⊙H(x) + (1−T(x))⊙x,
and LSTM-style recurrent gates [6] control the memory-cell
update at timet,
ct=ft⊙ct−1+it⊙˜ct.If a simple GLU [12] is attached after attention, the aggregate
hi=P
jαijvjhas already been formed, and the gate
controls an aggregate representation or channel activation,
GLU(h i) =A(h i)⊙σ(B(h i)).
In all three cases, the gate targets hidden dimensions, layer
transformations, recurrent states, or already mixed attention
outputs; it does not separate whether each item-specific
αijvjis prediction evidence.
In contrast, Warrant places the gate on the item-wise
weighted value term before aggregation.
cW
ij=αijgijvj, hW
i=X
jcW
ij, mW
ij=αijgij.
Thus,g ijcreates different permissions for different item
indicesjwithin the same queryi, without softmax-
renormalizing the relevance distribution defined byα ij. In
general, the post-attention GLUA(h i)⊙σ(B(h i))is not
equal toP
jαijgijvj. The former is common channel scaling
after item contributions have been mixed, whereas the latter
is evidence scaling for each weighted value term before
mixing. Because of this distinction, Warrant can separately
measure path exposure from Base toPathExposureOnlyand
learned query-item permission fromPathExposureOnlyto
the full interface.
For stability,g ijuses leak-sigmoid scaling rather than
a hard mask. Every valid item retains at least aλ-sized
contribution path, and the bias is initialized so thatg ij
starts close to one. Training therefore starts near the base
attention model and gradually lowers the contribution of
specific query-item pairs only when required by the task
loss.
4.2 Theoretical Properties and Optimization Scope
Warrant reparameterizes the metric-facing weighted value
termc j=αjvjwith item-wise permission while preserving
attention relevanceα ij. This operation is not equivalent
to either a post-attention gate or an attention-logit gate.
A post-attention gateF(P
jαijvj)cannot distinguish dif-
ferent item decompositions that produce the same aggre-
gate. Attention-logit reweighting redistributes mass to the
remaining items through softmax normalization, whereas
Warrant usesmW
ij=αijgijand does not enforceP
jmW
ij=
1. Appendix A.9 and Appendix A.10 summarize this differ-
ence with a constructive check.
Contribution-level permission signal.: Let the loss be
L(hW). The gate logit is learned from the inner product be-
tween the weighted value term and the metric-loss gradient
(Appendix A.1).
∂L
∂ψj= (1−λ)σ′(ψj)⟨∇ hWL, cj⟩.(6)
Therefore, a contribution that increases loss receives a signal
under gradient descent that lowers its permission, whereas
a contribution that reduces loss receives a signal that pre-
serves its permission. IfLisβ-smooth and only one contri-
bution is scaled, the ideal permission that minimizes a local
smoothness upper bound is as follows (Appendix A.2).
g⋆
j= Π [λ,1]
1−⟨∇L(h), c j⟩
β∥cj∥2+ϵ
.(7)

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 6
αijvj × αijgijvjhW
i=X
jαijgijvjgij=g(q i, kj)
scaling
aggregate overj
attention-weighted
value termvalue-contribution
interface
warranted
termprediction
state
Fig. 2. Warrant interface. Attention computes item relevanceα ij, and Warrant exposes the post-attention weighted value term to the metric-defining
path before computing the contribution scaling variableg ijin the full instantiation. The prediction-facing contribution becomesα ijgijvj.
Here,ϵis a numerical stabilizer used when∥c j∥is small.
This local rule shows why a gate that lowers harmful
weighted value terms and preserves helpful weighted
value terms arises naturally. The same view appears in
the signal-to-noise ratio analysis. Appendix A.6 derives
SNR W/SNR B=R S/RN, so the condition for Warrant to
improve SNR isR S> R N. That is, SNR improvement de-
pends not on perfect preservation of support contributions,
but on the relative size of retained support signal versus
retained noise standard deviation. Figure 3 visualizes this
boundary and shows that operating points fall in the im-
provement region under an evidence-aligned gate regime;
the simulation setting is given in Appendix B.
Manifold-local interpretation and entangled paths.:
The derivation above is written in the ambient represen-
tation space, but Appendix A.3 shows that the same rule
also holds on a local representation manifold. The key is
to replace ambient contributions and gradients with their
tangent-space counterparts:c jbecomes˜c j=P hcj, and
∇LbecomesgradML. This extension is a local descent
interpretation, not a global convergence guarantee.
When multiple terms are controlled together, cou-
pling among tangent contributions becomes important.
Appendix A.4 derives the following bound fora j=
⟨gradML(h),˜c j⟩andG jk=⟨˜c j,˜ck⟩.
L(R h(ξ(g)))≤ L(h)−δ⊤a+βM
2δ⊤Gδ.(8)
IfGis nearly diagonal, item-wise Warrant is a good lo-
cal shrinkage approximation. Conversely, if off-diagonal
tangent overlap is large, helpful contributions and short-
cut/noisy contributions can share the same local subspace,
and a scalar permission variable can become conservative
or miscalibrated. The synthetic simulation in Appendix C
also shows that the cross term increases as tangent Gram
off-diagonal mass grows and can reverse the diagonal
shrinkage benefit under high curvature. This perspective is
connected to the path-sensitive but entangled or saturated
negative-row interpretation in Section 5.8.
Optimization scope.: The leak-sigmoid parameteri-
zation guaranteesg j∈[λ,1],|∂g j/∂ψj| ≤(1−λ)/4, and
∥∆(g)∥ ≤(1−λ)P
j∥cj∥(Appendix A.5). Thus, Warrant
is a smooth bounded perturbation of the base value path.Under standard smooth nonconvex SGD assumptions, the
Warrant-augmented model inherits the usual convergence-
to-stationarity guarantee. This does not imply global con-
vergence or guaranteed improvement over the base model.
It only states that Warrant preserves smooth trainability
while providing a local descent-aligned evidence-admission
signal.
4.3 Path Localization
Warrant targets the metric-defining value path that leads
to the primary metric. Operationally, a metric-defining path
is the last item-wise bottleneck through which a key-value
item’s weighted value term actually changes the input to the
prediction object from which the reported primary metric is
computed. The mere presence of an attention block, or the
fact that a hidden representation changes, is not sufficient
for a path to be metric-defining.
Path localization is not black-box automatic discovery.
The procedure assumes access to the model computation
graph and to the tensor from which the reported primary
metric is computed. Under this condition, we use the au-
ditable tracing protocol in Algorithm 1. We expose candi-
date attention-derived weighted value terms, test whether
perturbing each term actually changes the metric object,
and then select the last item-wise bottleneck before that
object. Therefore, path selection is constrained by metric-
object perturbation and the item-wise bottleneck criterion
rather than by arbitrary attention placement.
Algorithm 1 defines Warrant as a value-path adaptation
protocol rather than a fully automatic black-box search pro-
cedure. Generic attention placement can attach an interface
to an attention module, but it is not metric-defining unless
perturbing its item-wise weighted value term changes the
reported metric object. Table 3 summarizes the resulting
implementation-level tensor paths for the evaluated do-
mains.
4.4 Domain Adaptation
In this benchmark, CTDG, MTPP , RAG, STPP , and TKG refer
to concrete prediction tasks. Therefore, Warrant adaptation
focuses less on designing domain-specific auxiliary features
and more on identifying the key-value aggregation path that

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 7
0.0 0.2 0.4 0.6 0.8 1.0
retained support signal RS0.00.20.40.60.81.0retained noise std. RN
RS>RN
SNR improvesRS≤RN
SNR degrades
RS=RNSNR improvement condition
example: RS=0.9,RN=0.5
SNRW/SNRB=1.8
0.0 0.2 0.4 0.6 0.8 1.0
retained support signal RSRS>RN
SNR improvesRS≤RN
SNR degrades
RS=RN
Illustrative Monte Carlo under evidence-aligned gates
P(SNR improves) = 0.982
median ratio = 1.19x
Fig. 3. SNR improvement condition under evidence-aligned permission. Left: Warrant improves SNR when retained support signalR Sexceeds
retained noise standard deviationR N. The diagonal line marks the boundaryR S=RN. Right: In an evidence-aligned gate simulation, sampled
operating points concentrate in theR S> RNregion.
Algorithm 1. Metric-facing value-path tracing for Warrant
Input:base modelF, evaluation batchB, reported primary metricm, candidate attention modulesA, candidate prediction
objectsZ.
Output:metric-facing weighted value pathp⋆.
1) Identify the prediction objectz∈ Zfrom whichmis computed, e.g., edge score, candidate mark logit, support
logit, next-location state, tail score, or copy mass.
2) Instrument candidate attention modules. For each module, expose attention weightsα ij, valuesv j, and weighted
value termsc ij=αijvj.
3) Enumerate candidate item-wise pathsp. Keep only paths wherec(p)
ijhas an item axisjand lies upstream ofzin
the computation graph.
4) Estimate path reach by Jacobian or intervention:
Reach(p) =E B∥Jz←c(p)c(p)∥
∥z∥+ϵ,Reach 0(p) =E Bd 
z, z[c(p)←0]
.
Discard paths whose perturbation changes hidden states but not the metric objectz.
5) Select the last item-wise bottleneck beforezamong paths with non-negligible reach. Prefer the path closest toz
that still preservesc ij=αijvj.
6) Construct Warrant onp⋆:αijvj7→α ijg(qi, kj)vj.
7) Validate the placement with Base vs. PathExposureOnly, Correct-path vs. Generic q-k Warrant, and Correct-path
vs. Shuffled-pairing Warrant diagnostics.
TABLE 3
Summary of tensor-level path localization traced by Algorithm 1. Each row denotes the last item-wise weighted-value bottleneck whose
perturbation actually changes the reported metric object.
Domain Gated tensor Query interface Insertion point Gradient path to metric
CTDG source/destination history term current edge query edge-score history aggregation AUC edge score
MTPP past event value term candidate mark query candidate mark residual logit Mark MRR candidate score
RAG retrieved passage value term question-passage query support ranking logit Support MRR passage score
STPP spatio-temporal history term next-event dynamics query next-location state update Location RMSE dynamics state
TKG historical fact tail term(h, r, t)query tail score / copy path Tail MRR ranking score
leads to each task’s primary metric and attaching the same
value-contribution interface to that path.
The full Warrant interface uses the same mapping in
Eq. (9) for every domain.
Warrant:α ijvj7→α ijg(qi, kj)vj (9)
The operator is fixed; only the weighted value path lead-ing to the task metric and the learned query-key repre-
sentations supplied by the base model to Warrant differ
across domains. The application criterion is simple. The
path through which the key-value item’s weighted value
term is transmitted to the task metric must be exposed, and
exactly that term must be used as the scaling target. We
do not insert Warrant at arbitrary locations merely because

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 8
an attention layer exists. We first find the value path that
determines the metric, then place the interface at the point
where unchecked weighted value terms from stale, shifted,
shortcut, or ambiguous items pass through that path.
CTDG link prediction.: The CTDG task is
continuous-time dynamic graph link prediction, where
the goal is to predict whether the current source-destination
edge exists from a time-ordered interaction history. Since
the prediction target is the current edge score rather
than a token or node representation, the Warrant query
must also represent an edge-level prediction request.
Warrant constructs the current edge query through an
edge-query adapter and gates source-side and destination-
side temporal-history weighted terms. This adapter does
not provide manual permission features such as recency
thresholds, stale indicators, hard-negative labels, or overlap
heuristics. It is an interface construction that exposes the
model-internal representations already used by the base
temporal encoder to the query-key interface of Warrant.
MTPP next-mark ranking.: The MTPP task is marked
temporal point process next-mark ranking, and the primary
metric is Mark MRR computed over candidate mark rank-
ings. The same past event can act as support, a shortcut,
or an ambiguous item depending on the candidate mark. If
a shared sequence state scores every candidate, candidate-
specific permission differences can become weak. Warrant
builds a candidate query from the candidate mark embed-
ding and the current sequence state, reads event-history
values to form a candidate-specific warranted value, and
provides a residual to the candidate mark logit. At this
application point, Warrant asks whether a past marked
event value may contribute to the current candidate mark
score.
RAG supporting evidence selection.: The RAG task
is formulated as retrieval-augmented supporting evidence
selection. This experiment does not call an LLM or generate
free-form answers; because there is no generated answer
string, answer F1/EM is not evaluated. Instead, using Hot-
potQA sentence-level supporting facts, we define a retrieved
passage that contains an annotated supporting fact as a
supporting evidence passage, and use Support MRR to
measure how highly that passage is ranked as the primary
metric. A distractor passage can receive attention because it
has lexical overlap with the question, but it is not thereby
qualified to contribute to the supporting-evidence ranking
score. Warrant forms warranted passage mass between the
question query and retrieved passage representations and
connects this mass directly to the support logit. It therefore
gates the path by which weighted value terms from sup-
porting evidence passages and distractor passages directly
affect the ranking metric.
STPP next-location forecasting.: The STPP task is
spatio-temporal point process next-location forecasting. The
model predicts the time and continuous location of the next
event, and the primary metric is Location RMSE. Unlike
settings such as CTDG and TKG, where bias is added
directly to a discrete candidate score, STPP determines the
final metric through a continuous next-location dynamics
state. Strongly copying past coordinates can create local
cluster bias, and time likelihood and Location RMSE can
move in different directions. Therefore, Warrant is not usedas a coordinate-copy device; it gates the path by which the
prediction query reads history values to construct the next-
event dynamics state.
TKG tail prediction.: The TKG task is tem-
poral knowledge graph tail prediction. The query is
(head, relation, time), the key-value item is a historical
temporal fact, and the primary metric is tail-entity ranking
MRR. A historical fact can be strong support, but a tail
repeated in the past is not guaranteed to be correct for the
current query. The bottleneck is therefore the path through
which the historical fact value moves into the tail score
or copy path. In RE-NET and xERTE, warranted mass is
added to the historical tail-entity score; in CyGNet, Warrant
is attached to the path on which copy mass is formed. This
placement directly gates whether the tail weighted term of
a historical fact may contribute to the current tail ranking.
Diagnostic criteria.: Correct adaptation should be
evaluated with both performance numbers and internal sig-
nals. The difference between attention mass and warranted
mass shows whether relevance and permission are sepa-
rated, and candidate-wise gate differences show whether
the same history is permitted differently across candidates.
Support/distractor mass checks whether distractor contri-
butions are suppressed in RAG, while stale history terms
check whether old items in CTDG, TKG, and MTPP avoid
excessively raising scores. In STPP and MTPP , we also
examine residual scale to verify that Warrant does not overly
perturb the base dynamics.
5 EXPERIMENTS
This section evaluates Warrant along several experimental
axes. We first use a cross-domain main benchmark to test
whether primary-metric improvements recur across task
families. We then separate path exposure from learned
permission through an operator ablation, and use mass and
counterfactual diagnostics to show that a relevant item is not
always prediction evidence. Next, a HotpotQA/RoBERTa
candidate-marker control examines the effect of the gated
object and path. Finally, path localization and the CTDG
edge-query adapter ablation validate the interpretation of
Warrant as an adaptation method for metric-facing value
paths, and the Warrant Need Score (WNS) with a negative-
row audit analyzes heterogeneity and drop rows.
5.1 Main Benchmark Settings
Every comparison uses the same reference implementation,
seed set, and dataset/model grid, changing only the Base
model versus the Warrant variant. Table 4 summarizes the
training, Warrant, and model settings used in the reported
benchmark.
5.2 Main Benchmark: Cross-Domain Performance
The main benchmark covers five task families: CTDG,
MTPP , RAG, STPP , and TKG. We conduct 32 paired Base-
versus-Warrant comparisons, and each comparison is re-
ported as the mean and standard deviation over three seeds.
The total number of completed runs is 192. Table 5 reports
the paired results. Each row compares Base and Warrant for
the same domain, dataset, and model, and improvement is

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 9
TABLE 4
Main benchmark settings. The benchmark runs Base and Warrant variants with the same reference implementation, seed set, and model grid.
Group Setting Value
Experiment seeds7,17,37
Training epochs / max examples 30; 65,536
Training optimizer learning rate 0.001; weight decay 0.0
Training history / evaluation history length 50; evaluation ratio 0.2
Warrant context / auxiliary loss no context schema or fields; auxiliary weight 0.0
Warrant gate initialization / leak gate init 0.95; gate leak 0.05
Warrant domain loss weights STPP gate loss 0.0; TKG gate loss 0.0
Warrant RAG exception FiD/LED use gate leak 0.7 and support loss weight 1.0
Models CTDG Wikipedia, MOOC, LastFM with TGAT, DyGFormer, GraphMixer
Models MTPP StackOverflow, Retweets with SAHP , THP , AttNHP
Models STPP Earthquake, Gowalla with Transformer-STPP , NSTPP , DeepSTPP
Models TKG ICEWS14, ICEWS18, GDELT with RE-NET, xERTE, CyGNet
Models RAG HotpotQA with FiD, LED
computed with the metric direction taken into account. The
±values in Table 5 are seed-wise standard deviations; they
are not row-level 95% confidence intervals or pairedt-tests.
Row-level paired CIs andp-values should be interpreted
only when seed-wise paired deltas are reported, so the main
text claims only aggregate directionality and practical tiers.
The main benchmark directly shows cross-domain per-
formance improvements. Warrant improves the primary
metric in 27 of the 32 comparisons, and the gains are not iso-
lated to a single domain or architecture. An exact two-sided
sign test provides aggregate evidence that 27/32 positive
directions exceed chance-level symmetry (p= 0.000113).
This value does not imply individual row significance,
nor does it imply that every positive comparison has the
same effect size. Under practical tiering, 10 comparisons are
substantial, 1 is marginal, 8 are positive but uncertain, 8 are
tie/negligible, and 5 are drops. In CTDG, AUC increases in
all 9 combinations, and in TKG, MRR increases in 8 of 9
combinations. In STPP , Location RMSE decreases in 5 of 6
combinations, and in RAG, Support MRR increases for both
HotpotQA supporting evidence selection readers. MTPP
improves on Retweets but is mixed on StackOverflow.
When the value path through which attention-formed
weighted value terms reach the primary metric is found
and Warrant is placed on that path, direction-aware im-
provements recur across task families. The mean domain-
level improvements are +0.0261 AUC for CTDG, +0.0002
Mark MRR for MTPP , +0.0010 Support MRR for RAG, a
+0.0290 reduction in Location RMSE for STPP , and +0.0062
MRR for TKG, with an overall mean of +0.0146. These
results are consistent with the interpretation of Warrant as
an adaptation operator for metric-defining value paths.
Drop rows are revisited in Section 5.8 through Warrant
Need Score (WNS) component decomposition and row-level
failure diagnostics. The key point is that WNS is not a guar-
anteed performance-improvement score, but a diagnostic of
path leverage or sensitivity. A row can be highly sensitive to
the path, yet have zero positive PathReach and a negative
main-row delta if the opened path is entangled with thedynamics or already saturated by a useful prior path.
5.3 Separating Path Exposure from Learned Permis-
sion
In standard attention, relevance and contribution are tied to
the same scalarα ij.
relevance(q i, kj)⇒α ij,contribution(j) =α ijvj.
Changing an attention weight changes the item relevance
distribution itself. In contrast, Warrant does not redefine
the attention distribution; it places a separate contribution
scaling variable on the weighted value term produced by
attention.
contribution(j) =α ijgijvj.
Thus,α ijdenotes key-value item relevance,g ijdenotes
weighted-value-term permission, andα ijgijvjdenotes the
warranted contribution entering prediction-facing aggrega-
tion.
The TKG operator ablation demonstrates this separation
at the level of the computational path. Table 6 reports
GDELT-xERTE variant results. This path-localization setting
uses a controlled diagnostic configuration separate from the
full main-benchmark configuration in Table 5. Therefore,
the absolute MRR values differ from the main benchmark,
and the comparison is between path exposure and learned
permission under the same conditions.PathExposureOnlyis
a diagnostic that connects the historical-tail weighted value
term to the metric-facing tail score while fixing every item-
wise scaling variable tog= 1. Base→PathExposureOnly
measures the effect of exposing a missing contribution path
to the score, and PathExposureOnly→Full measures the
refinement added by learned query-item permission on the
opened path.
On GDELT-xERTE, moving from Base to PathExposure-
Only increases MRR by +0.0677. Moving from PathExpo-
sureOnly to the Full value gate adds +0.0010 MRR, and
the Shuffled gate, which breaks the pairing between the

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 10
TABLE 5
Full main benchmark results by domain. Each row compares Base and Warrant on the same dataset and model, reporting the mean and standard
deviation over three seeds. Paired∆is the direction-aware mean difference. The±values are seed-wise standard deviations, not row-level 95%
CIs or pairedt-tests. Tier indicates the practical interpretation.
Dataset Model Metric Base Warrant Paired∆Relative Tier
CTDGcontinuous-time dynamic graph link prediction, AUC↑
LastFM DyGFormer AUC0.8482±0.0045 0.9023±0.0022 +0.0541+6.38%substantial
LastFM GraphMixer AUC0.8806±0.0031 0.9139±0.0009 +0.0334+3.79%substantial
LastFM TGAT AUC0.8598±0.0044 0.8897±0.0024 +0.0298+3.47%substantial
MOOC TGAT AUC0.9639±0.0019 0.9715±0.0007 +0.0076+0.79%positive uncertain
MOOC DyGFormer AUC0.9763±0.0003 0.9773±0.0016 +0.0010+0.11%tie/negligible
MOOC GraphMixer AUC0.9770±0.0000 0.9774±0.0015 +0.0004+0.04%tie/negligible
Wikipedia DyGFormer AUC0.9440±0.0015 0.9836±0.0005 +0.0396+4.20%substantial
Wikipedia TGAT AUC0.9437±0.0027 0.9824±0.0010 +0.0387+4.10%substantial
Wikipedia GraphMixer AUC0.9529±0.0040 0.9832±0.0003 +0.0303+3.18%substantial
MTPPmarked temporal point process next-mark ranking, Mark MRR↑
Retweets AttNHP Mark MRR0.7574±0.0130 0.7653±0.0049 +0.0079+1.06%positive uncertain
Retweets SAHP Mark MRR0.7541±0.0092 0.7578±0.0106 +0.0037+0.49%positive uncertain
Retweets THP Mark MRR0.7584±0.0029 0.7497±0.0115−0.0086−1.14%drop
StackOverflow THP Mark MRR0.6066±0.0017 0.6076±0.0013 +0.0009+0.15%tie/negligible
StackOverflow AttNHP Mark MRR0.6051±0.0020 0.6041±0.0045−0.0010−0.17%drop
StackOverflow SAHP Mark MRR0.6076±0.0017 0.6062±0.0017−0.0013−0.22%drop
RAGretrieval-augmented supporting evidence selection, Support MRR↑
HotpotQA FiD Support MRR0.6555±0.0017 0.6574±0.0030 +0.0019+0.29%positive uncertain
HotpotQA LED Support MRR0.6669±0.0018 0.6670±0.0002 +0.0002+0.02%tie/negligible
STPPspatio-temporal point process next-location forecasting, Location RMSE↓
Earthquake DeepSTPP Location RMSE1.1383±0.0288 1.0502±0.0926 +0.0881+7.72%substantial
Earthquake NSTPP Location RMSE1.1589±0.0612 1.1159±0.1006 +0.0430+3.84%substantial
Earthquake Transformer-STPP Location RMSE1.1147±0.0893 1.0748±0.0948 +0.0399+3.56%substantial
Gowalla Transformer-STPP Location RMSE0.0483±0.0047 0.0460±0.0024 +0.0023+4.38%positive uncertain
Gowalla NSTPP Location RMSE0.0486±0.0033 0.0474±0.0017 +0.0012+2.26%tie/negligible
Gowalla DeepSTPP Location RMSE0.0481±0.0012 0.0484±0.0046−0.0002−0.42%drop
TKGtemporal knowledge graph tail prediction, MRR↑
GDELT xERTE MRR0.1277±0.0022 0.1429±0.0008 +0.0151+11.85%substantial
GDELT RE-NET MRR0.1280±0.0011 0.1395±0.0010 +0.0115+9.01%marginal
GDELT CyGNet MRR0.1478±0.0006 0.1478±0.0012 +0.0000+0.02%tie/negligible
ICEWS14 xERTE MRR0.2096±0.0024 0.2176±0.0032 +0.0080+3.81%positive uncertain
ICEWS14 RE-NET MRR0.2076±0.0008 0.2127±0.0046 +0.0052+2.50%positive uncertain
ICEWS14 CyGNet MRR0.2195±0.0039 0.2196±0.0031 +0.0000+0.02%tie/negligible
ICEWS18 RE-NET MRR0.1423±0.0002 0.1550±0.0025 +0.0127+8.95%positive uncertain
ICEWS18 xERTE MRR0.1515±0.0017 0.1580±0.0022 +0.0065+4.31%tie/negligible
ICEWS18 CyGNet MRR0.1536±0.0007 0.1501±0.0054−0.0035−2.24%drop
TABLE 6
Controlled TKG path-exposure diagnostic on GDELT -xERTE. PathExposureOnly opens the historical-tail contribution path but fixes item-wise
scaling tog= 1, removing learned permission.
Variant MRR Delta MRR vs Base Full–Variant MRR Interpretation
Base 0.0488 0.0000 +0.0687 no localized tail contribution path
Generic q-k gate 0.0492 +0.0004 +0.0683 generic attention placement
PathExposureOnly,g= 10.1165 +0.0677 +0.0010 path exposure without learned permission
Shuffled gate 0.1144 +0.0655 +0.0032 path exposure with broken query-item pairing
Full value gate 0.1175 +0.0687 0.0000 path exposure plus learned permission
query and fact-tail item, is 0.0032 below Full. Most of the
improvement arises when the historical-tail contribution
path directly enters the tail score; learned permission then
provides a finer adjustment of fact-tail terms to the current
query within the opened path.
This pattern matches the structure of TKG tail pre-diction. Event sequences in GDELT and ICEWS often re-
peat the same entities, relation patterns, and tail entities
across timestamps. Since the tail-ranking metric is deter-
mined by candidate-entity scores, recurrence signals become
strong when the tail value inside a historical fact is trans-
mitted to the current candidate score. Under conditions

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 11
where representation-only paths do not transmit this sig-
nal sufficiently to the score, simply opening the historical-
tail path can yield a large gain. Learned permission then
organizes this recurrence prior according to the current
(head, relation, time)query. The TKG result therefore pro-
vides evidence for which metric path the path-localized
interface should open.
5.4 Evidence Mass and Counterfactual Diagnostics
Permission is needed because a relevant key-value item is
not identical to prediction evidence. In retrieval, a passage
with high lexical overlap with the question can receive
attention without being supporting evidence. In TKG, a
historical fact can be temporally and structurally connected
to the query without being true-tail evidence for the current
tail ranking. In CTDG, temporal history can be related to the
current edge while counterpart evidence and surrounding
context differ in contribution strength. Therefore, Warrant
focuses on deciding whether the weighted value term pro-
duced by attention is qualified to enter the evidence path of
the current prediction.
To examine this, neural dissection is run as a restricted
diagnostic experiment separate from the full main bench-
mark. For each domain, we select one dataset/model, use
seed 7, train for 10 epochs, and record gate regimes, war-
ranted mass, and contribution paths on at most 8192 ex-
amples. This setting is intended to test whether attention
mass and warranted mass separate, and whether the local-
ized contribution path has a counterfactual dependency on
labeled evidence.
Figure 4 summarizes the relevant observations. In RAG,
support attention share changes little, but warranted sup-
port mass increases and distractor mass decreases. This
shows that question-passage relevance and supporting-
evidence permission are not the same variable. The TKG
panel shows learned gate strength together with path ex-
posure. Even when the gate remains nearly identity in the
dissection run, the localized interface can greatly improve
tail ranking by opening the historical-tail contribution path.
Thus, a high gate mean does not imply learning failure, and
a low gate mean is not automatically better. The decisive
factor is whether the contribution interface is placed on the
metric-defining value path.
Table 7 reports the corresponding intervention values.
In RAG, removing supporting passage mass lowers Sup-
port MRR by 0.0030. Conversely, removing non-evidence
mass raises Support MRR by 0.0034 in the oracle counterfac-
tual. In TKG, removing true-tail evidence mass lowers MRR
by 0.1162, and removing non-evidence mass raises MRR
by 0.2245 in the oracle counterfactual. In CTDG, removing
counterpart history mass lowers AUC by 0.0575, and re-
moving non-evidence mass also lowers AUC by 0.0203. This
indicates that surrounding temporal context in CTDG is
also useful for edge prediction, while the larger dependency
lies on the counterpart evidence path. Thus, permission is
closer to current-query-dependent contribution scaling than
to binary evidence/non-evidence masking.
RAG case study.: Figure 5 is a HotpotQA case study
that visualizes raw attention, Warrant permission, and ef-
fective mass side by side on a passage-query grid. Rawattentionα ijcan assign mass to both supporting passages
and distractor passages. Permissiong ijin the middle panel
remains high on annotated supporting-passage cells and
becomes lower on distractor cells that received attention
because of lexical or contextual relevance. The effective
massα ijgijin the right panel combines the two signals:
supporting cells remain in the prediction-facing support
score, while distractor cells contribute less even when they
receive similar raw attention. This example shows that the
attention map is not the evidence map in RAG, and that
the mass entering supporting-evidence ranking is re-formed
through permission.
5.5 Pretrained Transformer Encoder: HotpotQA Sen-
tence Evidence Selection
The preceding RAG diagnostic showed that attention mass
and warranted mass separate for retrieval-style passage
evidence. This section moves the same question into the
candidate-marker readout path of a pretrained Transformer
encoder, testing whether the effect of Warrant can be ex-
plained by simply adding a gate, adding parameters, ap-
plying a post-attention gate, using an attention readout, or
exposing a path. The main text summarizes the result with
one control table, and Appendix E provides detailed settings
and diagnostics.
The experiment fine-tunesroberta-base[26] for Hot-
potQA supporting sentence selection. Each input consists
of a question, an instruction, up to 8 candidate evidence
sentences, and candidate markers. The candidate set in-
cludes annotated supporting sentences, hard distractors,
and random distractors, and the primary metric is Support
MRR.openpath_nogateopens the same metric-facing
candidate-marker value path while fixingg= 1, and
full_warrantreplaces the item-wise termα ijvjon that
path withα ijgijvj. The full setup and permission diagnos-
tics are provided in Appendix E.
Table 8 shows that Warrant is not explained by parame-
ter count or aggregate-level gating alone.full_warrant
is +0.0098 MRR aboveparam_mlp, +0.0083 MRR
abovepost_attention_glu, and +0.0088 MRR above
shuffled_warrant, which breaks query-item pairing.
At the same time,openpath_nogateis strong, so in
the HotpotQA/RoBERTa setting, opening the metric-facing
candidate-marker value path itself explains an important
part of the result. On top of this path,full_warrant
adds a small learned-permission refinement and obtains
the best values for MRR, R@1, Evidence F1, and Unsup-
ported@GoldCount. AUPRC is lower thanbase. Therefore,
the interpretation of this control focuses on improvements in
the ranking top end and unsupported evidence attribution
rather than uniform improvement across all metrics.
Unsupported evidence attribution.: Unsup-
ported@GoldCount measures the fraction of unsupported
candidates included when selecting as many evidence
sentences as the number of gold supports. In the controlled
candidate-evidence setting, this metric captures an
evidence-attribution failure in which an unsupported
sentence is selected as if it were supporting evidence; such
failures can lead to downstream answer hallucination.
full_warrantlowers Unsupported@GoldCount from

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 12
Support
shareDistractor
mass0.00.20.40.60.8Mass / ratio0.2590.741
0.2610.634
+0.002-0.107RAG
Attention relevance is nearly unchanged;
permission reduces distractor contribution.Attention Warranted
1 3 5 7 10
Epoch0.000.020.040.060.08MRR
0.073
0.013TKG
Permission opens the historical tail contribution path
while ranking quality separates from the base model.Base MRR Warrant MRR T ail permission mass
CTDG RAG TKG0.00.20.40.60.81.01.21.4Warrant mass0.609
0.242
0.0821.379
0.6960.879Mass diagnostic
Localized Warrant paths carry labeled evidence
while retaining domain-specific context mass.Evidence Non-evidence
CTDG RAG TKG−0.2−0.10.00.1Metric drop+0.057
+0.003+0.116
+0.020
-0.003
-0.225Counterfactual intervention
Positive evidence drop means useful evidence mass;
negative non-evidence drop means distractor contribution.Zero evidence Zero non-evidence0.00.20.40.60.81.0
T ail permission mass1.000
Fig. 4. Mass separation and counterfactual mass diagnostic. The upper panels show RAG relevance-permission separation and TKG path exposure,
and the lower panels show CTDG/RAG/TKG zeroing interventions.
TABLE 7
Mass diagnostic using evaluation-time counterfactual interventions. A positive zero-evidence drop means that evidence mass is useful, and a
negative zero-non-evidence drop means that non-evidence contribution harms the oracle counterfactual.
Domain Metric Primary Evidence mass Non-evidence mass Drop: zero evidence Drop: zero non-evidence
CTDG AUC0.8916±0.0073 0.6086±0.0234 1.3791±0.0236 +0.0575±0.0068 +0.0203±0.0012
RAG Support MRR0.5527±0.0113 0.2419±0.0008 0.6960±0.0082 +0.0030±0.0015−0.0034±0.0008
TKG MRR0.1173±0.0022 0.0815±0.0029 0.8787±0.0308 +0.1162±0.0022−0.2245±0.0027
TABLE 8
HotpotQA/RoBERTa candidate-marker control results. All variants use the same candidate-marker input, split, metric, and three seeds. Lower
Unsupported@GoldCount is better.
Variant MRR↑R@1↑Evidence F1↑AUPRC↑Unsupported@GoldCount↓
base0.9111±0.0045 0.8371±0.0073 0.7751±0.0049 0.8110±0.0065 0.2249±0.0049
param_mlp0.9036±0.0057 0.8245±0.0106 0.7688±0.0033 0.8103±0.0082 0.2312±0.0033
post_attention_glu0.9051±0.0039 0.8295±0.0098 0.7629±0.0084 0.8034±0.0116 0.2371±0.0084
attention_readout0.9125±0.0042 0.8405±0.0072 0.7730±0.00760.8129±0.0040 0.2270±0.0076
query_only_gate0.9078±0.0021 0.8312±0.0048 0.7726±0.0057 0.8104±0.0053 0.2274±0.0057
shuffled_warrant0.9046±0.0102 0.8270±0.0166 0.7671±0.0160 0.7979±0.0086 0.2329±0.0160
openpath_nogate0.9129±0.0092 0.8405±0.0165 0.7755±0.0110 0.8102±0.0127 0.2245±0.0110
full_warrant0.9134±0.00100.8414±0.00520.7772±0.0058 0.8042±0.00270.2228±0.0058
0.2249 to 0.2228, corresponding to an unsupported-
attribution error reduction of about 0.94%. Since MRR, R@1,
and Evidence F1 also increase, Warrant acts in this setting
to reduce unsupported evidence selection.5.6 Path Localization as a General Adaptation Method
The methodology of Warrant is defined as a path-
localization procedure that finds a metric-facing weighted
value path. As defined in Section 4.3 and Algorithm 1, the

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 13
Item 1 Item 2 Item 3 Item 4 Item 5Item 1
Item 2
Item 3
Item 4
Item 5
Item 6
Item 7
Item 8
Item 9
Item 10
Item 11
Item 12★ ★
★
★ ★
★ ★
★ ★
★ ★
★ ★
★
★
★ ★
★
★ ★Raw attention αij
Item 1 Item 2 Item 3 Item 4 Item 5Item 1
Item 2
Item 3
Item 4
Item 5
Item 6
Item 7
Item 8
Item 9
Item 10
Item 11
Item 12★ ★
★
★ ★
★ ★
★ ★
★ ★
★ ★
★
★
★ ★
★
★ ★Warrant permission gij
Item 1 Item 2 Item 3 Item 4 Item 5Item 1
Item 2
Item 3
Item 4
Item 5
Item 6
Item 7
Item 8
Item 9
Item 10
Item 11
Item 12★ ★
★
★ ★
★ ★
★ ★
★ ★
★ ★
★
★
★ ★
★
★ ★Effective mass αijgij
0.000.050.100.150.200.25
0.700.750.800.850.900.951.00
0.000.050.100.150.200.25
Outlined ★ cells are annotated supporting passages. Raw attention and effective mass use the same color scale.
Fig. 5. RAG case study on HotpotQA. The left panel shows raw attentionα ij, the middle panel shows Warrant permissiong ij, and the right panel
shows effective massα ijgij. Outlines and stars mark annotated supporting-passage cells.
procedure first fixes the prediction object from which the
primary metric is computed, then traces whether perturbing
a candidate attention-derived weighted value term actually
changes that object, and finally places the value-contribution
interface at the last item-wise bottleneck. The experiments
test whether this tracing protocol applies unchanged across
domains. Figure 6 compares the path selected by Algo-
rithm 1 with generic q-k placement and shuffled query-item
pairing.
Correct-path Warrant achieves higher direction-aware
performance than Base in every domain. In CTDG and TKG,
Generic q-k Warrant is nearly flat, whereas Correct-path
Warrant yields large improvements. In CTDG, Generic gives
only +0.0043 AUC, while Correct-path obtains +0.1119 AUC.
In TKG, Generic gives only +0.0004 MRR, while Correct-
path obtains +0.0687 MRR. Therefore, performance gains are
more closely tied to placement on the metric-defining value
path than to the mere presence of a Warrant block.
Shuffled pairing tests the second condition of the
method. The improvement depends jointly on the path
itself and on the permission relation between the current
query and the key-value item. In CTDG, MTPP , and STPP ,
shuffled pairing is much lower than Correct-path. In RAG
and TKG, the shuffled degradation is smaller, but Correct-
path still has the highest score. Thus, the general method
consists of path localization, which finds the path where the
weighted value term reaches the primary metric, and query-
item permission, which computes whether the current query
authorizes the key-value item’s contribution on that path.5.7 CTDG Edge-Query Adapter Ablation
In CTDG, the prediction target is the current source-
destination edge score rather than a node representation.
Therefore, the Warrant query must also represent an edge-
level prediction request. The edge-query adapter ablation
separates whether this interface can be explained by in-
creased parameters or hand-crafted temporal features. Fig-
ure 7 summarizes the ablation results.
Edge-conditioned Warrant obtains an AUC of 0.8980,
which is +0.1119 above Base. The parameter control and
hand-crafted-only control also provide some improvement,
but neither approaches the gain of Edge-conditioned War-
rant. When the pairing between the edge query and history
item is broken at evaluation time, AUC collapses to 0.6376.
This result indicates that the value-contribution interface
between the current edge query and source/destination
history weighted terms plays a central role in CTDG.
5.8 Explaining Heterogeneity: WNS and Negative-Row
Audit
The main benchmark in Section 5.2 shows that the effect
of Warrant is not uniform across domains. Improvements
are relatively large in CTDG, STPP , and TKG, but small or
mixed in MTPP and RAG. Warrant Need Score (WNS) is
not a scalar success predictor; it is an operational diagnostic
that measures the leverage and localization sensitivity of
the metric-facing value path. WNS is computed by directly
training and evaluating control variants with the same bud-
get in each representative domain setting; it is not a post hoc

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 14
Correct
> BaseCorrect
> GenericCorrect
> ShuffledCTDG
MTPP
RAG
STPP
TKG✓
+0.112 (3/3)✓
+0.108 (3/3)✓
+0.260 (3/3)
✓
+0.016 (1/1)×
-0.000 (0/1)✓
+0.017 (1/1)
✓
+0.002 (1/1)✓
+0.001 (1/1)✓
+0.000 (1/1)
✓
+0.020 (1/1)×
-0.007 (0/1)✓
+0.132 (1/1)
✓
+0.069 (1/1)✓
+0.068 (1/1)✓
+0.003 (1/1)
Fig. 6. Path localization check. The figure compares Base, Generic q-k Warrant, Correct-path Warrant, and Shuffled pairing for each domain.
Base Generic
q-kParam
controlHand-crafted
controlEdge-conditioned
WarrantShuffled
edge query0.60.70.80.9AUCBase mean0.786 0.790
+0.0040.807
+0.0210.820
+0.0340.898
+0.112
0.638
-0.148
Fig. 7. CTDG edge-query adapter ablation. Edge-conditioned Warrant outperforms generic Warrant, the parameter control, and the hand-crafted
control, while a shuffled edge query causes a large performance drop.
combination of the main benchmark, path localization, and
mass-diagnostic results in Section 5.2.
For each domain, we compare Base, Generic q-k War-
rant, Correct-path Warrant, and Shuffled-pairing Warrant.
PathReach asks whether Correct-path Warrant improves
over Base; Correct>Generic asks whether metric-facing
placement matters; and Correct>Shuffled asks whether
query-item pairing matters. Every component is computed
as a paired delta with the metric direction taken into ac-
count. For higher-is-better metrics,∆(A, B) =A−B; for
lower-is-better metrics,∆(A, B) =B−A. Raw WNS is
a weighted average of the positive parts of the PathReach
relative, Correct>Generic relative, and Correct>Shuffled
relative components, each divided by the absolute basemetric value. Therefore, a WNS component should be in-
terpreted as a direction-aware delta normalized by the base
metric value; it is distinct from an absolute AUC, MRR, or
RMSE delta.
Domain-level WNS: Where Does Warrant Have Leverage?
Table 9 reports WNS results together with descriptive cor-
relations to the main-benchmark relative gain from Sec-
tion 5.2. Because there are only five domains, the correlation
p-values should not be interpreted as independent statistical
proof; checking the ordering and component pattern is suf-
ficient. WNS and the main-benchmark domain-level relative
gain from Section 5.2 show rank consistency, with Pearson
r= 0.765and Spearmanρ= 0.800. Here, RAG WNS is

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 15
a diagnostic for the reference FiD support-passage ranking
path, whereas the RoBERTa result in Section 5.5 is a separate
evidence-attribution setting that uses candidate sentence
marker readout and auxiliary permission supervision.
Domain-level WNS explains why the effect of War-
rant is heterogeneous across task families. TKG shows
high PathReach and Correct>Generic, indicating a high-
leverage historical-tail path. CTDG also shows strong path
sensitivity and query-item pairing sensitivity. STPP has
strong sensitivity but low PathReach, suggesting that correct
query-history pairing matters even though it can interact
with continuous dynamics. MTPP and RAG remain at weak
or negligible WNS, which is consistent with mixed or tie-
level gains.
Negative-row Audit: Sensitivity Without Positive PathReach
Table 10 reports the WNS component audit for the negative
rows.
The two negative rows share a PathReach of zero. They
therefore do not contradict the domain-level WNS interpre-
tation. Instead, they show cases where the localized path is
sensitive or saturated but is not safely improved by a scalar
permission interface.
Gowalla / DeepSTPP is not a simple false-suppression
case. Under a useful-local-history proxy, Correct-path War-
rant preserves target-proximal history more strongly than
remaining history. The drop is therefore difficult to explain
as suppression of useful local evidence. This row is a case
where query-history pairing sensitivity is large without
positive PathReach. A more appropriate interpretation is
local-prior/dynamics entanglement: Warrant changes the
local-history path, but the latent state of DeepSTPP already
compresses user-specific revisits, place priors, and mobility
dynamics, so a scalar permission perturbation can conflict
with short-move or revisit dynamics.
ICEWS18 / CyGNet is better interpreted as a copy-
saturated case. PathReach and Correct>Generic are zero,
OpenPath-NoGate is almost identical to Base, and the
learned copy gate is also close to identity. This suggests that
CyGNet already exposes a strong copy path, leaving little
room for an additional scalar permission interface to im-
prove performance. In the diagnostic, false-over-true copy
imbalance slightly increases after Warrant scaling, which is
consistent with copy-calibration risk rather than a missing-
path failure.
5.9 Synthesis: Path-Localized Evidence Contributions
The overall experimental flow converges on a single inter-
pretation. The main benchmark in Section 5.2 shows broad
positive directionality but also includes mixed and drop
rows. Path-exposure and mass diagnostics show that perfor-
mance depends on the metric-facing contribution path, not
on attention relevance itself. Path localization and the CTDG
edge-query ablation show that correct placement and query-
item pairing matter. Finally, WNS and the negative-row au-
dit explain why the effect of Warrant is heterogeneous across
domains and rows. Even when path sensitivity is high, it
may not lead to positive reach if the path is entangled with
dynamics or already saturated by a prior path.
Therefore, Warrant is best interpreted as a localized inter-
face that exposes and controls weighted value contributionsentering prediction. Edge history in CTDG, event history
in MTPP , retrieved passages in RAG, spatio-temporal his-
tory in STPP , and historical facts in TKG can all receive
attention, but whether their contributions are qualified to
enter the metric-facing score or state must be re-decided by
the current query and metric path. When this interface is
explicitly placed on the value path, relevance is converted
into prediction evidence.
6 DISCUSSION, LIMITATION,ANDCONCLUSION
Discussion.: The experiments in this paper show
that identifying the value path through which attention-
formed weighted value terms reach the final metric is
the starting point of Warrant adaptation. We first find the
path of the weighted value termα ijvjthat flows through
attention aggregation into the actual prediction object and
determines the primary metric, and then place a path-
localized value-contribution interface on that path. When
this placement is correct,α ijvjis replaced byα ijgijvj, and
metric-facing contributions are exposed or controlled more
accurately across tasks. The main benchmark, TKG path-
exposure ablation, mass diagnostics, HotpotQA/RoBERTa
control, path localization, CTDG edge-query ablation, and
WNS/negative-row audit all point to the same conclusion.
When the interface lies on the metric-defining value path,
the weighted value term is exposed as the unit of prediction
evidence and, when needed, becomes the target of permis-
sion.
In standard attention,α ijsimultaneously represents rel-
evance between a query and key and determines the size of
the value contribution. Under this structure, once an item is
judged relevant, its value term enters the prediction state.
Relevance, path exposure, and contribution permission are
tied inside a single scalar. Warrant preserves the relevance
decision of attention, places the already formed weighted
value term on the metric path through which it can affect
the current prediction, and, when needed, decides its qual-
ification with a separate gateg ij. The attention output is
thereby treated as a path-localized evidence unit.
This decomposition is especially important in domains
where the contribution of a relevant key-value item varies
with the query and metric path. A retrieved passage can be
related to a question without being supporting evidence, a
historical fact can be connected to a temporal query without
being current true-tail evidence, and temporal history can be
related to the current edge while being stale or reinforcing
a hard negative. Such items can receive attention, but if
their weighted value terms enter the metric-facing path
unchanged, they can act as distractors, shortcuts, or stale
signals. Evidence is formed when a weighted value term is
exposed on the path leading to the primary metric and its
contribution is permitted under the current query.
Limitation.: The performance of Warrant is sensitive
to how accurately the interface captures the value path of
the primary metric. In MTPP , Mark MRR is computed from
candidate mark ranking, but the shared state jointly contains
mark transitions, temporal intensity, and inter-event time. In
data such as Retweets, where the number of marks is small
and repeated propagation patterns are strong, past events
are more easily used as candidate-specific support. In data

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 16
TABLE 9
Dedicated Warrant Need Score (WNS). The upper block summarizes raw WNS, relative WNS, and base-normalized component scores for one
representative setting per domain. Component columns are direction-aware differences normalized by the base metric value, not absolute AUC,
MRR, or RMSE differences. The lower block reports descriptive correlations with main-benchmark domain-level relative gain; with only five
domains, these correlations are used as rank-consistency diagnostics rather than standalone statistical evidence.
Domain-level WNS components
Domain Setting TierSummary Component: direction-aware∆/|Base|
WNS Rel. WNSMain rel.
gainPathReach
rel.Correct>
Generic rel.Correct>
Shuffled rel.
TKG GDELT / xERTEvery high0.5758 1.0000 4.248% 0.9385 0.9351 0.0437
CTDG LastFM / DyGFormerstrong0.1322 0.2290 2.896% 0.1143 0.1159 0.1734
STPP Earthquake / DeepSTPPstrong0.1157 0.2004 3.557% 0.0074 0.0008 0.3822
MTPP Retweets / AttNHPweak0.0146 0.0247 0.028% 0.0000 0.0000 0.0496
RAG HotpotQA / FiDnegligible0.0004 0.0000 0.157% 0.0007 0.0002 0.0000
Descriptive correlation with main-benchmark relative gain
Predictor PearsonrSpearmanρExactp n
WNS 0.765 0.800 0.133 5
PathReach rel. 0.634 0.900 0.083 5
Correct>Generic rel. 0.630 0.900 0.083 5
Correct>Shuffled rel. 0.522 0.200 0.783 5
TABLE 10
Negative-row WNS component audit. Both rows have zero PathReach, so their WNS reflects sensitivity rather than positive reach over Base.
Negative row Main∆WNS PathReach Correct>Generic Correct>Shuffled
Gowalla / DeepSTPP−0.42%12.36 0.000 0.022 37.43
ICEWS18 / CyGNet−2.24%0.130 0.000 0.000 0.393
such as StackOverflow, where transitions are more diffuse,
the required weak transition signal and shortcut signal can
be mixed within the same history term, causing the gate to
behave conservatively.
In STPP , the reported metric is Location RMSE, whereas
the training objective fits time dynamics and spatial dynam-
ics together. In Earthquake, down-weighting stale spatial
history is transmitted relatively stably to the continuous
location head. In Gowalla, user-specific check-ins, long-tail
place priors, and sparse mobility patterns overlap; when
latent dynamics strongly compress history contributions,
as in DeepSTPP , residual-path alignment becomes more
important. The RAG results are limited to Support MRR
for supporting evidence selection and do not directly eval-
uate answer generation quality. In TKG, historical-tail path
exposure can account for a large share of the gain, as in
the GDELT-xERTE diagnostic, and copy-heavy models can
show small or negative additional gains because a copy
path is already open. These conditions motivate future
localization methods that identify metric-facing weighted
value paths more finely.
Conclusion.: Relevance is not permission. Attention
decides what the model can read, but it does not guaran-
tee that the read value contribution becomes evidence for
the current prediction. Warrant finds the weighted value
term that is actually transmitted to the primary metric in
each task and decides, through path exposure and query-conditioned permission, whether that term is qualified to
enter the prediction under the current query. From this
perspective, evidence is a value contribution permitted on
the metric path, and it is distinct from attention mass. A
weighted value term becomes evidence only when it is
warranted on the path to the metric.
DECLARATION OF COMPETING INTEREST
The authors declare that they have no known competing
financial interests or personal relationships that could have
appeared to influence the work reported in this paper.
ACKNOWLEDGEMENTS
The authors have no acknowledgements to declare.
DATA AVAILABILITY
All datasets used in this study are publicly available. The
exact data sources are listed in Table 11.
CODE AVAILABILITY
The implementation, experiment configurations, aggrega-
tion scripts, and plotting scripts used in this study are avail-
able at https://github.com/SnowyPainter/warrant-public.

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 17
TABLE 11
Public dataset sources used in this study.
Domain Dataset Source
CTDG Wikipedia https://snap.stanford.edu/jodie/wikipedia.csv
CTDG MOOC https://snap.stanford.edu/jodie/mooc.csv
CTDG LastFM https://snap.stanford.edu/jodie/lastfm.csv
MTPP StackOverflow HuggingFace: https://huggingface.co/datasets/easytpp/stackoverflow
MTPP Retweets HuggingFace: https://huggingface.co/datasets/easytpp/retweet
STPP Earthquake GitHub: https://github.com/ss15859/EarthquakeNPP
STPP Gowalla https://snap.stanford.edu/data/loc-gowalla totalCheckins.txt.gz
TKG ICEWS14 GitHub RE-Net data: https://github.com/INK-USC/RE-Net
TKG ICEWS18 GitHub RE-Net data: https://github.com/INK-USC/RE-Net
TKG GDELT GitHub RE-Net data: https://github.com/INK-USC/RE-Net
RAG HotpotQA HuggingFace: https://huggingface.co/datasets/hotpotqa/hotpot qa
REFERENCES
[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones,
A. N. Gomez, L. Kaiser, and I. Polosukhin, “Attention is
all you need,” inAdvances in Neural Information Processing
Systems, 2017. [Online]. Available: https://papers.nips.cc/paper/
7181-attention-is-all-you-need
[2] D. Xu, C. Ruan, E. K ¨orpeo ˘glu, S. Kumar, and K. Achan,
“Inductive representation learning on temporal graphs,” in
International Conference on Learning Representations, 2020. [Online].
Available: https://openreview.net/forum?id=rJeW1yHYwH
[3] L. Yu, L. Sun, B. Du, and W. Lv, “Towards better dynamic graph
learning: New architecture and unified library,” inAdvances in
Neural Information Processing Systems, 2023. [Online]. Available:
https://proceedings.neurips.cc/paper files/paper/2023/hash/
d611019afba70d547bd595e8a4158f55-Abstract-Conference.html
[4] P . Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin,
N. Goyal, H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel,
S. Riedel, and D. Kiela, “Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks,” inAdvances
in Neural Information Processing Systems, 2020. [On-
line]. Available: https://proceedings.neurips.cc/paper/2020/
hash/6b493230205f780e1bc26945df7481e5-Abstract.html
[5] R. K. Srivastava, K. Greff, and J. Schmidhuber, “Highway
networks,” inICML Deep Learning Workshop, 2015. [Online].
Available: https://arxiv.org/abs/1505.00387
[6] S. Hochreiter and J. Schmidhuber, “Long short-term memory,”
Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.
[7] K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau,
F. Bougares, H. Schwenk, and Y. Bengio, “Learning phrase
representations using rnn encoder–decoder for statistical machine
translation,”arXiv preprint arXiv:1406.1078, 2014. [Online].
Available: https://arxiv.org/abs/1406.1078
[8] E. Perez, F. Strub, H. de Vries, V . Dumoulin, and A. Courville,
“Film: Visual reasoning with a general conditioning layer,” in
Proceedings of the AAAI Conference on Artificial Intelligence, 2018.
[Online]. Available: https://arxiv.org/abs/1709.07871
[9] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le,
G. Hinton, and J. Dean, “Outrageously large neural networks:
The sparsely-gated mixture-of-experts layer,” inInternational
Conference on Learning Representations, 2017. [Online]. Available:
https://openreview.net/forum?id=B1ckMDqlg
[10] D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation
by jointly learning to align and translate,” inInternational
Conference on Learning Representations, 2015. [Online]. Available:
https://arxiv.org/abs/1409.0473
[11] S. Sukhbaatar, J. Weston, R. Ferguset al., “End-to-end
memory networks,” inAdvances in Neural Information Processing
Systems, 2015. [Online]. Available: https://papers.nips.cc/paper/
5846-end-to-end-memory-networks
[12] Y. N. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language
modeling with gated convolutional networks,” inProceedings of
the 34th International Conference on Machine Learning, 2017, pp. 933–
941.[13] J. Hu, L. Shen, and G. Sun, “Squeeze-and-excitation networks,” in
Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition, 2018, pp. 7132–7141.
[14] B. Dhingra, H. Liu, Z. Yang, W. W. Cohen, and R. Salakhutdinov,
“Gated-attention readers for text comprehension,” inProceedings
of the 55th Annual Meeting of the Association for Computational
Linguistics, 2017, pp. 1832–1846. [Online]. Available: https:
//aclanthology.org/P17-1168/
[15] E. Parisotto, H. F. Song, J. W. Rae, R. Pascanu, C. Gulcehre,
S. M. Jayakumar, M. Jaderberg, R. L. Kaufman, A. Clark,
S. Noury, M. Botvinick, N. Heess, and R. Hadsell, “Stabilizing
transformers for reinforcement learning,”Proceedings of the 37th
International Conference on Machine Learning, 2020. [Online].
Available: https://proceedings.mlr.press/v119/parisotto20a.html
[16] X. Ma, C. Zhou, X. Kong, J. He, L. Gui, G. Neubig, J. May, and
L. Zettlemoyer, “Mega: Moving average equipped gated atten-
tion,” inInternational Conference on Learning Representations, 2023.
[17] K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhudinov,
R. Zemel, and Y. Bengio, “Show, attend and tell: Neural image
caption generation with visual attention,” inProceedings of the 32nd
International Conference on Machine Learning, 2015, pp. 2048–2057.
[18] A. F. T. Martins and R. F. Astudillo, “From softmax to sparsemax:
A sparse model of attention and multi-label classification,” in
Proceedings of the 33rd International Conference on Machine Learning,
2016, pp. 1614–1623.
[19] A. See, P . J. Liu, and C. D. Manning, “Get to the point: Summariza-
tion with pointer-generator networks,” inProceedings of the 55th
Annual Meeting of the Association for Computational Linguistics, 2017,
pp. 1073–1083.
[20] V . Karpukhin, B. Oguz, S. Min, P . Lewis, L. Wu, S. Edunov,
D. Chen, and W.-t. Yih, “Dense passage retrieval for open-domain
question answering,” inProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing, 2020, pp.
6769–6781. [Online]. Available: https://aclanthology.org/2020.
emnlp-main.550/
[21] G. Izacard and E. Grave, “Leveraging passage retrieval with
generative models for open domain question answering,” in
Proceedings of the 16th Conference of the European Chapter of
the Association for Computational Linguistics, 2021, pp. 874–880.
[Online]. Available: https://aclanthology.org/2021.eacl-main.74/
[22] I. Beltagy, M. E. Peters, and A. Cohan, “Longformer: The
long-document transformer,”arXiv preprint arXiv:2004.05150,
2020. [Online]. Available: https://arxiv.org/abs/2004.05150
[23] R. Nogueira and K. Cho, “Passage re-ranking with bert,”
arXiv preprint arXiv:1901.04085, 2019. [Online]. Available: https:
//arxiv.org/abs/1901.04085
[24] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of
modern neural networks,” inProceedings of the 34th International
Conference on Machine Learning, 2017, pp. 1321–1330. [Online].
Available: https://proceedings.mlr.press/v70/guo17a.html
[25] Y. Geifman and R. El-Yaniv, “Selective classification for deep
neural networks,” inAdvances in Neural Information Processing
Systems, 2017. [Online]. Available: https://papers.nips.cc/paper/
7073-selective-classification-for-deep-neural-networks

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 18
[26] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy,
M. Lewis, L. Zettlemoyer, and V . Stoyanov, “Roberta: A
robustly optimized bert pretraining approach,”arXiv preprint
arXiv:1907.11692, 2019. [Online]. Available: https://arxiv.org/
abs/1907.11692
APPENDIXA
THEORETICALDERIVATIONS
This appendix summarizes the mathematical structure by
which Warrant operates as a permission operator on metric-
facing attention-weighted value terms. For a single query,
write
h=X
jαjvj, c j=αjvj,
hW=X
jαjgjvj=X
jgjcj, gj=λ+ (1−λ)σ(ψ j).
Here,λ >0is the leak factor. Thus, Warrant is not
a hard mask, and every valid item retains a minimal
value/gradient path.
A.1 Gate Gradient
Let the metric-facing loss beL(hW). Sinceψ jdirectly affects
onlyg j,
∂hW
∂gj=αjvj=cj,
∂gj
∂ψj= (1−λ)σ′(ψj),
∂L
∂ψj= (1−λ)σ′(ψj)⟨∇ hWL, cj⟩.
By the chain rule, the gate receives an item-wise loss-aligned
signal for the weighted value term itself. If the inner product
is positive, the contribution points in a loss-increasing direc-
tion, and gradient descent lowers the permission logit. If the
inner product is negative, the contribution points in a loss-
reducing direction, and the gate moves toward preservation.
A.2 Local Loss Bound
Assume thatLisβ-smooth with respect to the prediction
representation. Taking the base state ash=P
jcjand
scaling only itemj,
h(gj) =h+ (g j−1)c j.
The smoothness condition gives the following bound.
L(h(g j))≤ L(h) + (g j−1)⟨∇L(h), c j⟩+β
2(gj−1)2∥cj∥2,
γj=⟨∇L(h), c j⟩,
g⋆
j= Π [λ,1]
1−γj
β∥cj∥2
.
Up to constants, this is a one-dimensional quadratic. There-
fore, a harmful contribution with positive gradient align-
ment is reduced, whereas a helpful contribution with nega-
tive alignment is preserved atg j= 1by projection.A.3 Manifold-Local Permission Rule
The Euclidean local rule in the main text is written in the
ambient representation space. Suppose that the prediction
state lies near a localC2representation manifoldM ⊂Rd,
with tangent spaceT hMand tangent projectionP h. Writing
gj= 1−δ j, theWarrantperturbation is
∆(g) =X
j(gj−1)c j=−X
jδjcj
and the component that directly enters first-order loss
change on the manifold is
ξ(g) =P h∆(g) =−X
jδjPhcj.
Let˜c j=P hcj. Riemannian smoothness with respect to a
local retractionR hgives
L(R h(ξ))≤ L(h) +⟨gradML(h), ξ⟩+βM
2∥ξ∥2.
If only a single contribution is controlled, thenξ=−δ j˜cj,
and the upper bound excluding constants is
−δj⟨gradML(h),˜c j⟩+βM
2δ2
j∥˜cj∥2.
Thus, whenϵ= 0, the unconstrained minimizer is
δj=⟨gradML(h),˜c j⟩
βM∥˜cj∥2.
Projecting this value to the feasible interval[0,1−λ]yields
the rule in the main text. Theϵin the main-text expression
is a numerical stabilizer used when∥˜c j∥is close to zero; it
is not part of the exact minimizer itself. This result is a local
tangent-space descent interpretation, not a global optimality
statement.
A.4 Tangent Gram Matrix and Entangled Paths
When multiple contributions are controlled simultaneously,
ξ(g) =−P
jδj˜cj. Define
aj=⟨gradML(h),˜c j⟩, G jk=⟨˜c j,˜ck⟩.
Substituting into the Riemannian smoothness bound gives
L(R h(ξ(g)))≤ L(h)−X
jδjaj+βM
2X
jδj˜cj2
=L(h)−δ⊤a+βM
2δ⊤Gδ.
Here,Gis the Gram matrix of tangent contributions. It
is not precise to always call off-diagonal entries penalties.
SinceG jkcan be negative, they are more safely inter-
preted as off-diagonal tangent coupling or cross terms.
However, large positive off-diagonal coupling, or large-
magnitude off-diagonal coupling, can destabilize diagonal
item-wise shrinkage. In this case, useful contributions and
shortcut/noisy contributions share the same local tangent
subspace, so scalar item-wise permission can become con-
servative or miscalibrated.

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 19
A.5 Smooth Bounded Parameterization and Stationary
Scope
The leak-sigmoid gate is defined as
gj=λ+ (1−λ)σ(ψ j),
so
gj∈[λ,1],∂gj
∂ψj= (1−λ)σ(ψ j)(1−σ(ψ j))≤1−λ
4.
Also, becauseδ j= 1−g j∈[0,1−λ],
∥∆(g)∥=X
j(gj−1)c j≤X
jδj∥cj∥ ≤(1−λ)X
j∥cj∥.
Therefore, Warrant is not a hard discontinuous mask but a
smooth bounded perturbation of the base value path. Under
standard smooth nonconvex SGD assumptions that the base
network andψ jnetwork are smooth and stochastic gradi-
ents are unbiased with bounded variance, the augmented
objectiveF(θ)also has a finite smoothness constant. For
step sizeη≤1/L F, the standard descent lemma gives a
convergence-to-stationarity bound of the form
1
TT−1X
t=0E∥∇F(θ t)∥2≤2(F(θ 0)−F ⋆)
ηT+LFησ2
g.
This is not a guarantee of global convergence specific to
Warrant or of performance improvement over the base
model. It is a scope statement that the bounded smooth pa-
rameterization does not break ordinary smooth trainability.
A.6 SNR Condition
Decompose the metric-facing contribution into support sig-
nal and noisy non-support terms.
hB=X
j∈Sαjµj+X
j∈Nαjξj,E[ξ j] = 0,Var(ξ j) =σ2
j,
SNR B=P
j∈SαjµjqP
j∈Nα2
jσ2
j,
SNR W=P
j∈SαjgjµjqP
j∈Nα2
jg2
jσ2
j.
The SNRs of Base and Warrant areSNR BandSNR W,
respectively. Support signal retention and noise standard-
deviation retention are
RS=P
j∈SαjgjµjP
j∈Sαjµj,
RN=vuutP
j∈Nα2
jg2
jσ2
jP
j∈Nα2
jσ2
j,
SNR W
SNR B=RS
RN,
SNR W>SNR B⇐⇒R S> R N.
This condition does not require perfect preservation of the
support signal. SNR increases when the support signal is
retained more strongly than the noise standard deviation.A.7 High-Probability Interpretation
The values appearing in the SNR condition are computed as
weighted averages. Define the weights
wS
j=αjµjP
l∈Sαlµl,
wN
j=α2
jσ2
jP
l∈Nα2
lσ2
l,
RS=X
j∈SwS
jgj, R2
N=X
j∈NwN
jg2
j.
Becauseg j∈[λ,1], both quantities are bounded weighted
averages. If evidence-aligned permission creates a pos-
itive expectation margin between the weighted support
gate and the weighted noisy gate RMS, concentration of
bounded weighted averages makes the probability of the
false-suppression eventR S≤R Ndecrease as the effective
number of support/noise items grows. Thus, the alignment
margin controls the probability of SNR degradation.
A.8 Gradient Noise and Effective Curvature
Letξ jbe a noisy value-gradient component. The base path
transmitsα jξj, whereas the Warrant path transmitsα jgjξj.
Therefore,
Var[α jgjξj] =α2
jg2
jVar[ξ j]≤α2
jVar[ξ j],
∥∇2
vjL∥base≤α2
jβ,
∥∇2
vjL∥Warrant ≤α2
jg2
jβ≤α2
jβ.
Likewise, ifLisβ-smooth with respect toh, the second
line holds along the item value directionv j. This bound
is path-local. A down-weighted metric-facing value path
reduces gradient-noise and effective-curvature exposure on
that path.
A.9 Diagonal Permission vs. Softmax Reweighting
LetmW
j=α jgj. Since Warrant does not recompute the
attention softmax,
∂mW
j
∂ψl=αj(1−λ)σ′(ψj)1[j=l],
˜αj= softmax j(sj+ logg j),
∂˜αj
∂logg l= ˜αj(1[j=l]−˜α l).
The first derivative has an item-wise diagonal structure.
In contrast, attention-logit gating couples all items through
softmax normalization. Therefore, Warrant scales contribu-
tions while preserving relevanceα j, whereas logit gating
changes the relevance distribution itself.
A.10 Post-Attention Gates and Path Localization
A post-attention gateF(h)sees only the aggregateh=P
jαjvj. If different item decompositions produce the same
h, thenF(h)cannot assign different permissions to the
underlying items. For example, ifv 1=u,v 2=−u, and
α1=α 2= 0.5, then the aggregate ish= 0. A post-
attention gate sees onlyh= 0, so it cannot permitv 1
andv 2differently, whereas Warrant can expressg 1̸=g 2
inhW= 0.5g 1u+ 0.5g 2(−u).

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 20
An attention-logit gate can preserve item identity, but
softmax normalization redistributes attention mass. Lower-
ing one item automatically increases the mass of other items.
Warrant instead usesmW
j=α jgjand does not enforceP
jmW
j= 1. This operation corresponds to attenuation of
weighted value contributions rather than redistribution of
relevance.
Table 12 summarizes the same-aggregate constructive
check.
In contrast, Warrant is applied before item identity dis-
appears.
hW=X
jαjg(q, k j)vj,
∂L
∂ψj= (1−λ)σ′(ψj)D
J⊤
p∇zL, α jvjE
, J p=∂z
∂hp.
Here, the gated path ish p=P
jαjgjvj, and the metric
object isz=f(h p). The gate learning signal is proportional
to the path-to-metric Jacobian. This is the mathematical
reason for path localization. If Warrant is placed on a path
weakly connected to the metric, the permission signal is also
weak; correct-path placement exposes the weighted value
term that actually changes the reported metric.
APPENDIXB
MONTECARLODETAILS
Figure 3 visualizes the SNR condition derived in Ap-
pendix A.6. The left panel shows the deterministic bound-
aryR S=R N, whereR SandR Nare the support-signal
retention and noise-standard-deviation retention defined in
Appendix A.6. The right panel samples the same quanti-
ties under an evidence-aligned gate regime. The weighted-
average interpretation follows Appendix A.7.
For each trial, we sample a support item setSand
a noisy item setN. The simulation uses|S|= 8sup-
port items and|N|= 32noisy items. Attention weights
for support and noisy items are independently sampled
from normalized Gamma variables to form random simplex
weights. Support signal magnitudesµ jand noise standard
deviationsσ jare sampled from log-normal distributions to
avoid a degenerate equal-weight setting.
µj∼LogNormal(0,0.252), σ j∼LogNormal(0,0.252).
Gate variables are sampled from Beta distributions. In the
evidence-aligned regime of Figure 3, the support gate mean
is set to0.72, the noisy gate mean to0.60, and both distri-
butions use concentration parameter30.
gS
j∼Beta(0.72·30,(1−0.72)·30),
gN
j∼Beta(0.60·30,(1−0.60)·30).
This is a weak alignment setting rather than a near-perfect
separator. Support gates are only moderately larger than
noisy gates, and some sampled points remain in the degra-
dation region.
For each trial,R S,RN, andSNR W/SNR B=R S/RN
are computed according to the definitions in Appendix A.6.
A trial is counted as improved whenR S> R N. The number
of trials is 2,500, with a fixed random seed. Under this
weak evidence-aligned regime,98.2%of sampled operatingpoints lie in theR S> R Nregion, and the median SNR ratio
is1.19×. If the support gate becomes lower than the noisy
gate, sampled points move toward the degradation region
described in Appendix D.
APPENDIXC
MANIFOLD-LOCALSIMULATION
This appendix provides a synthetic bound decomposition
that illustrates the tangent Gram bound in Appendix A.4. It
is an illustrative instantiation showing how the cross term
δ⊤Gδcan reduce the benefit of diagonal shrinkage. With
two tangent contributions, seta= (1,1),δ= (0.5,0.5), and
G=1ρ
ρ1
.
Then the descent term isδ⊤a= 1.0, and the bound change
is
∆bound =−δ⊤a+βM
2δ⊤Gδ.
Table 13 shows that the cross term grows asρand curvature
βMincrease, and that the bound can become positive in a
high-curvature entangled case.
This simulation aligns with the negative-row interpreta-
tion in Section 5.8. Even when path sensitivity is high, scalar
item-wise permission may not lead to safe positive reach if
useful contributions and shortcut/noisy contributions are
strongly entangled in the tangent space.
APPENDIXD
FAILUREREGIMES
The same analysis identifies three failure regimes. First,false
suppressionoccurs when support items receive low gates
andR S≤R N. Second,unaligned permissionoccurs when
noisy or shortcut items receive gates as high as support
items, so noise retention does not decrease. Third,weak path
localizationoccurs when Warrant is placed on a path with
a small path-to-metric Jacobian, causing gates to receive
only weak metric-facing learning signals. Therefore, the
experiments jointly evaluate diagnostics such as the primary
metric, support/distractor mass, gate ratios, and correct-
path versus generic placement.
APPENDIXE
ADDITIONALHOTPOTQA/ROBERTADETAILS
This appendix provides detailed settings, full per-variant
results, and permission diagnostics for the pretrained
RoBERTa supporting sentence selection control in Sec-
tion 5.5. The main text presents one control table and the
core interpretation, while the appendix supplements it with
input construction, training settings, readout variants, and
seed-wise mean/std results.
Table 14 describes the candidate-marker control setting,
Table 15 reports full per-variant results, and Table 16 reports
the permission diagnostics.

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 21
TABLE 12
Same-aggregate constructive check. This toy check is a constructive experiment that isolates function-class differences between gated objects.
Variant Item identity available Attention renormalized Expected accuracy Interpretation
post_attention_gateno no 0.5 aggregate gate sees the same zero vector for both labels
attention_logit_gateyes yes 1.0 separates items by redefining attention mass
warrant_value_term_gateyes no 1.0 keepsαfixed and changes item-wise weighted value contribution
TABLE 13
Synthetic manifold-local bound decomposition. The table is an illustrative instantiation of Appendix A.4, not a benchmark result. Larger
off-diagonal tangent coupling increases the cross term and can overturn the diagonal shrinkage benefit under high curvature.
Regimeβ M ρ−δ⊤adiagonal term cross term∆ bound
Low entanglement 1.0 0.05−1.0000.250 0.013−0.738
Moderate entanglement 1.0 0.40−1.0000.250 0.100−0.650
High entanglement 1.0 0.80−1.0000.250 0.200−0.550
High-curvature entangled 3.0 0.80−1.0000.750 0.600+0.350
TABLE 14
HotpotQA/RoBERTa candidate-marker control setting. This experiment does not evaluate answer generation; it is an encoder-side
evidence-attribution task that ranks whether a candidate sentence is annotated supporting evidence.
Group Component Setting
Encoder base modelroberta-base
Data dataset HotpotQA processed supporting sentence candidates
Input format question + instruction +⟨C0⟩, . . . ,⟨C7⟩candidate evidence sentences
Input candidate count maximum 8 candidates per question
Input candidate composition supporting candidates + hard distractors + random distractors
Training questions / seeds / epochs maximum 2,000 questions; seeds7,17,37; 2 epochs
Training main loss candidate-wise binary cross entropy
Training permission auxiliary weak pairwise gate-alignment auxiliary loss
Evaluation primary metric Support MRR over candidate ranking
Evaluation unsupported attribution unsupported candidate selected within gold-count budget
Controlbasefinal candidate marker readout baseline
Controlparam_mlpparameter-count control without value-term permission
Controlpost_attention_gluaggregate/readout gate after item identity is mixed
Controlattention_readoutungated candidate-marker value-path exposure
Controlquery_only_gategeneric permission without item key conditioning
Controlshuffled_warrantquery-item pairing is broken
Controlopenpath_nogatesame metric-facing path withg= 1
Controlfull_warrantα ijvj7→α ijgijvjon the candidate-marker value path
TABLE 15
HotpotQA/RoBERTa supporting sentence selection full control results. Values are means and standard deviations over three seeds. Bold indicates
the best value for each metric.
Variant MRR↑R@1↑Evidence F1↑AUPRC↑Unsupported@GoldCount↓
base0.9111±0.0045 0.8371±0.0073 0.7751±0.0049 0.8110±0.0065 0.2249±0.0049
param_mlp0.9036±0.0057 0.8245±0.0106 0.7688±0.0033 0.8103±0.0082 0.2312±0.0033
post_attention_glu0.9051±0.0039 0.8295±0.0098 0.7629±0.0084 0.8034±0.0116 0.2371±0.0084
attention_readout0.9125±0.0042 0.8405±0.0072 0.7730±0.00760.8129±0.0040 0.2270±0.0076
query_only_gate0.9078±0.0021 0.8312±0.0048 0.7726±0.0057 0.8104±0.0053 0.2274±0.0057
shuffled_warrant0.9046±0.0102 0.8270±0.0166 0.7671±0.0160 0.7979±0.0086 0.2329±0.0160
openpath_nogate0.9129±0.0092 0.8405±0.0165 0.7755±0.0110 0.8102±0.0127 0.2245±0.0110
full_warrant0.9134±0.00100.8414±0.00520.7772±0.0058 0.8042±0.00270.2228±0.0058

IEEE TRANSACTIONS ON PATTERN ANAL YSIS AND MACHINE INTELLIGENCE 22
TABLE 16
HotpotQA/RoBERTa permission diagnostic. The Gold/Random ratio is the average mass ratio between annotated supporting candidates and
random distractors.
Variant Gate mean Gold gate Random gate Gold/RandomαGold/Randomαg
attention_readout0.3671 0.8688 0.1745 0.3811 2.2536
query_only_gate0.5416 0.8610 0.4161 0.99875.0091
shuffled_warrant0.9994 0.9994 0.9994 0.9013 0.9015
openpath_nogate1.0000 1.0000 1.0000 0.6265 0.6265
full_warrant0.5319 0.8647 0.4049 0.8179 3.8522