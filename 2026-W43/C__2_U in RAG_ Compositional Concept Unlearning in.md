# C$^2$U in RAG: Compositional Concept Unlearning in Retrieval-Augmented Generation

**Authors**: Yiyang Duan, Fan Wu, Tiantong Wu, Che Wang, Zhiren Gong, Wei Guo, Yang Cao, Wei Yang Bryan Lim

**Published**: 2026-10-21 05:30:06

**PDF URL**: [https://arxiv.org/pdf/2610.29223v1](https://arxiv.org/pdf/2610.29223v1)

## Abstract
Retrieval-augmented generation can expose unwanted concepts through retrieved evidence even when the generator is moderated. We study query-side concept suppression with a fixed encoder and document index. Compositional Concept Unlearning (C$^2$U) routes a profile-derived direction through an augmented atom bank, adding only a bounded edited-minus-original delta to the raw query. Across five datasets, four encoders, and three seeds, the validation-selected configuration reduces pooled binary target Hit@5 by 4.25 percentage points with non-positive control degradation and target-exposure changes. Under the evaluated query transforms, it suppresses target density without increasing budgeted control-query exposure. Separately, ArXiv four-concept composition improves over an averaged-direction comparator after independent validation gates and first-use test evaluation.

## Full Text


<!-- PDF content starts -->

C2U in RAG: Compositional Concept Unlearning
in Retrieval-Augmented Generation
Yiyang Duan1, Fan Wu1, Tiantong Wu1, Che Wang1, Zhiren Gong1, Wei Guo3,
Yang Cao2, Wei Yang Bryan Lim1
1Midori Lab, Nanyang Technological University,2Institute of Science Tokyo (formerly Tokyo Institute
of Technology),3Beihang University
Abstract
Retrieval-augmented generation can expose unwanted concepts through retrieved
evidence even when the generator is moderated. We study query-side concept
suppression with a fixed encoder and document index. Compositional Concept
Unlearning (C2U) routes a profile-derived direction through an augmented atom
bank, adding only a bounded edited-minus-original delta to the raw query. Across
five datasets, four encoders, and three seeds, the validation-selected configuration
reduces pooled binary target Hit@5 by 4.25 percentage points with non-positive
control degradation and target-exposure changes. Under the evaluated query
transforms, it suppresses target density without increasing budgeted control-query
exposure. Separately, ArXiv four-concept composition improves over an averaged-
direction comparator after independent validation gates and first-use test evalua-
tion.
1 Introduction
Retrieval-Augmented Generation (RAG) grounds answers in external documents, but also
creates a pathway for private, unwanted, or policy-restricted evidence to enter a model’s
context. Output moderation can reject a response without changing that evidence pathway.
Conversely, deleting whole documents can remove useful content alongside an unwanted
concept. We study an intermediate intervention: suppress concept-bearing retrieval through
query-side representation editing, leaving the encoder and indexed corpus unchanged.
Here, “unlearning” means retrieval-stage suppression, not certified parameter deletion or a
guarantee about generated answers.
The problem has four aspects: intervention happens on a fixed-index retriever, a small profile
must guide bounded query edits, suppression–utility trade-offs must hold under ordinary
and transformed queries, and several concepts must share an editing budget. Figure 1 locates
the intervention within RAG. Linear erasure provides useful comparisons, but reducing
target retrieval need not reduce exposure under context packing. Access also matters: a
query transform cannot inspect and remove target documents like a membership-aware
filter. We therefore evaluate suppression, utility, and exposure together, distinguishing the
information available to each method.
C2U combines an Atomic Compositional Representation (ACR) with a raw-query bypass.
Target profiles and calibration controls determine a negative, diagonally preconditioned
direction and a calibrated gate. Least-squares coefficients and an appended residual atom
1
arXiv:2610.29223v1  [cs.CL]  21 Oct 2026

C2U in RAG
UnlearningLearningRetained Data
RAG 
Query
ResponseQuery VectorsEncoderRetrievalGeneratorSplitDatabase
(a) Retraining-Based(b) Linear ErasureEncoder r
DatabaseRetained VectorsVectorsForgotten Data
Erasure Matrix r
MapRep.SpaceRep. : RepresentationTrainable
(c)OurproposedC  U2
Original Rep.Erased Rep.…
(Instance-time trained)𝑎!𝑎"…𝑎|			|
Atom WeightsAtom Dictionary
(One-time trained)(Instance-time trained)
Figure 1 Intervention locations in RAG.Retraining changes the representation model; linear erasure
transforms representations; C2U edits atom coordinates and bounds the resulting raw-query update.
Its encoder, document index, and downstream generator remain fixed.
carry that direction through the frozen atom bank; only the weight difference returns to the
raw query. A trust region limits the update. For multiple concepts, separate gates allocate a
shared angular budget before their deltas are added.
The experiments assess both useful suppression and its limits. The single-concept compar-
ison includes stronger-access methods and methods that suppress more at larger utility
costs; query-transform evaluation tests whether suppression coexists with increased context
exposure. Separately, ArXiv four-concept composition passes selection, independent con-
firmation, and first-use test evaluation, while the two-concept design fails selection. Our
contributions are threefold:
•Bounded query editing.An atom-coordinate operator with calibrated gating,
residual completion, and a raw-query trust region (Section 3) with gate-off exact
identity.
•Suppression and utility evidence.Joint retrieval and exposure comparisons across
datasets, encoders, and query conditions, with explicit access classes and evidence
scope.
•Concept composition.Separate-gate, shared-budget editing evaluated against
averaged-direction steering.
2 Related Work and Problem
Retrieval and safety.RAG research improves retrieval and context use through filtering,
ranking, and corrective generation (Hofstätter et al., 2023; Asai et al., 2024; Yu et al., 2024b;
Yan et al., 2024; Liu et al., 2024). Retrieved sources can nevertheless introduce unsafe content
or poisoning (An et al., 2025; Zhao et al., 2026). Moderation and safety frameworks (Inan
et al., 2023; Liang et al., 2025) address complementary parts of this problem; adversarial
suffixes and latent instructions illustrate why output defenses alone need evaluation (Zou
et al., 2023; Qiu et al., 2023). Unlike database redaction or retriever optimization (Wang
et al., 2026), our intervention preserves the raw document index that retrieval searches.
Unlearning and representation control.Parameter-updating unlearning (Chen et al.,
2023; Tarun et al., 2023; Fan et al., 2023) and diffusion concept erasure (Orgad et al., 2023;
Gandikota et al., 2023; Wu et al., 2025; Bui et al., 2025) motivate selective interventions but
do not establish fixed-index retrieval behavior. INLP (Ravfogel et al., 2020) and LEACE (Bel-
rose et al., 2023) remove linearly accessible concepts; CURE (Kocak et al., 2025) and RAG
Embedder Control (Kim et al., 2025) provide recent representation-control comparisons. Our
focus is a bounded query-side operator and its composition rule. Training and adaptation
differences are described in Sections A.3 and C.
2

C2U in RAG
Input Embeddings
Encoder
 Generation
 ACR
 Retrieval
: Frozen
 : Inference Process
C  U2
 
 
Atom bank
 Retrieved
 Texts
…
 
r
 
 ……
 
… ……
Atom editing
wq → w′q
: Trainable : Training Process
 : Concept-related texts
 
 
 
Initialize
Freeze
bβ
concept profileraw query q
q
Bounded update
qq′
θ ≤ θmaxq′ = normalize( q + ηδq)
 δqq′
Figure 2 Deployed C2U pipeline.The frozen encoder’s raw query qfeeds ACR and, via the bypass,
the bounded update. Profile-guided editing adds the scaled coefficients kqbβto the weights wq; the
difference δqand the update q′=normalize(q+ηδ q),θ≤θ max=0.2 rad, return one edited query to
the index.
Problem.Let q∈Sd−1be a normalized frozen-encoder query and uj∈Sd−1the un-
changed indexed document vectors. Retrieval ranks q⊤uj. Given a target concept profile, we
produce q′to reduce target retrieval while preserving control-label retrieval and avoiding
extra target exposure on control queries. Query-role and target-document membership
labels are evaluation inputs, not inputs to the deployed C2U operator.
A profile identifies what to suppress through examples, not a runtime list of forbidden
document IDs. Calibration controls characterize non-target geometry and set the gate before
evaluation. At retrieval time the same operator applies to target and control queries without
observing their roles. Suppressing target-seeking queries can still introduce unwanted
evidence for other users, so target suppression and control-query exposure are separate
outcomes throughout the evaluation.
3 Deployed Query Editing
Atomic representation.The atom bank A ∈RN×dis a learned dictionary whose rows ai
are unlabeled unit directions in the encoder’s representation space; Nvaries by dataset and
encoder (Section A.1). The frozen encoder maps text xtohx, and a lightweight projection
head scores hxagainst every row; a temperature-scaled softmax turns these scores into atom
weights, whose bank combination reconstructs the representation:
wx=softmax (s(hx,A)/τ ),
zx=normalize
A⊤wx
.(1)
Training optimizes only the bank and head—the encoder stays frozen throughout—with
a retrieval objective on zxand an entropy penalty that keeps wxpeaked; both are frozen
afterward (Section A.3). Editing then happens in atom coordinates: a direction in repre-
sentation space becomes a change of weight coefficients rather than a free-form vector edit.
The reconstruction itself never enters deployment. The index stores raw encoder vectors,
and online editing returns only the weight-difference delta to the raw query rather than
replacing the query by zx; substituting the reconstruction would define a different method
and break identity with the unchanged index deployment searches.
Direction and calibrated gate.Let µt,µcbe means of normalized target-profile and
calibration-control vectors. With v=normalize(µ t−µ c), the negative, diagonally
preconditioned direction is g=normalize(−D−1v), where Dis a ridge-regularized,
shrunk diagonal control second-moment matrix. A separate raw-space score z(q) =
3

C2U in RAG
q⊤normalize(µ t)−q⊤normalize(µ c)determines the continuous gate:
s(q) =clipz(q)−z 0
z1−z 0, 0, 1
. (2)
Here z0,z1are the calibration-control median and 95th percentile. The gate uses neither
ranks nor candidate labels; degenerate calibration uses the conservative fallback in Sec-
tion A.6.
Residual completion.The bank A ∈RN×dand weight-generating head are frozen after
ACR training (Section A.3); Nvaries by dataset and encoder (Section A.1). Solve a=
lstsq(A⊤,g)and setp=A⊤a,e=g−p. Above the residual threshold, appendr=e/∥e∥
and use bβ= [a;β∥e∥] inA′= [A;r] . For zero or small residuals, append r=p/∥p∥ with
coefficient zero; a degenerate pfails closed. Thus βscalesonly the appended coefficient. At the
selected β=1 , projected and residual components recover full gin exact arithmetic; the
small-residual branch instead truncateseto its in-span projection.
Angular allocation and delta transfer.Let h=A′⊤bβandϕ=∠(q,h) . The requested
angle is θreq=0.3s(q) ; angular feasibility gives θ=max(0, min(θ req,ϕ−ϵ plane)). For
active, feasible rows set
kq=sinθ
sin(ϕ−θ),w′
q= [w q; 0] +k qbβ. (3)
Edited weights are neither clipped nor renormalized. Normalizing hdefines its angle
only; bβretains its magnitude. The requested, feasible, and realized angles therefore differ
in general, particularly when β̸=1 . Transfer the actual edited-minus-padded weight
difference:
δq= (w′
q−[w q; 0])⊤A′. (4)
Training uses reconstruction, whereas online editing uses this difference; precision, cancella-
tion, and threshold details are in Section A.6.
Raw-query trust-region update.For active rows, return q′=normalize(q+ηδ q), taking
the full step if feasible and otherwise reducing η∈[0, 1] to satisfy the numerical counterpart
of
⟨q′,q⟩ ≥max(cos(0.2), 0.98). (5)
The raw query, not its reconstruction, anchors the update (Figure 2). Gate-off and zero-delta
rows return the input directly. The 0.2-rad cap is tighter than the 0.98 cosine floor and can
further reduce the realized angle. Numerical checks are detailed in Section A.6.
The trust region bounds motion, not retrieval damage. For ideal unit queries satisfying a
capθ max, each unit document’s score changes by at most 2 sin(θ max/2)(Theorem J.4).
Design rationale.Each component answers one deployment question. The calibrated gate
decides whether a query is edited at all, from the profile and unlabeled calibration controls
alone: it needs neither candidate labels nor ranks, scales the requested angle continuously,
and passes concept-distant queries through as an exact identity. The raw-query bypass
fixes what is edited: the deployed object is the encoder’s own vector, the index is never
touched, and the reconstruction zxstays a training device. The residual atom makes the
profile direction expressible in bank coordinates even when the frozen atom bank does not
span it, so the weight-difference transfer carries the full direction rather than its projection,
and the trust region bounds the resulting rotation. In composition, one shared budget caps
the summed rotation across concepts, so several active gates cannot compound into a large
motion, keeping the deployed operator bounded, query-side, and index-preserving.
Why not profile subtraction.The classical atom-coordinate edit subtracts profile mass
from query weights (Section A.5). Under the locked bank and deployed runtime, every
measured subtraction variant is an exact inactive identity: frozen query weights overlap
4

C2U in RAG
Table 1 Task overview: what each dataset suppresses and preserves, and its query source (Section B).
gRelaxed global controls.dOne synthetic template per role.
Dataset Target Control Queries
ArXiv topic categorygtitle/abstract
TOFU sensitive topic author identity questions
Amazon sentiment product category review titles
IMDB movie genre sentiment templated
MNLI text genre logic label hypotheses
the top- 30atom profile by only ≈0.0011 , and the applied-angle diagnostic stays near
2×10−5rad. Two validation-only revival probes—knob-only bank retraining and an
off-diagonal orthogonality loss—fail to revive subtraction and leave the deployed route
unchanged (Section A.6). Atom coordinates are what make this failure measurable: query
and profile mass occupy almost disjoint weight coordinates, so a subtractive edit has nothing
to act on. Suppression must instead add a bounded rotation along the completed direction—
the deployed operator. These bounded interventions do not rule out every subtraction
method under other banks or objectives.
4 Evaluation
4.1 Setup and Evidence Scope
We evaluate Amazon, ArXiv, IMDB, MNLI, and TOFU (McAuley & Leskovec, 2013; Bahram-
inasr & Vafaei-Sadr, 2020; Williams et al., 2018; Maini et al., 2024), using MiniLM, bge, e5,
and Qwen3 frozen encoders and three seeds. Single-concept stages contain 60units, each
with 100target and 300control queries, a shared 100-target/ 300-control raw index, and 20
target-profile documents. The final comparison contains 6,000 target and 18,000 control row
contributions. Profile, selection, confirmation, and final-evaluation allocations are disjoint;
selection and independent confirmation precede the fixed configuration.
Table 1 summarizes the task each dataset poses: a target concept whose documents should
leave the top ranks is paired with a control attribute whose retrieval must survive the edit—a
scientific topic against its ArXiv category, a sensitive biography topic against author identity,
review sentiment against product category, movie genre against review sentiment, and
source genre against the entailment label. Queries are titles or abstracts, natural questions,
review titles, or hypotheses; IMDB uses one synthetic template per role. ArXiv controls
come from the whole corpus rather than matched categories, so its control statistics describe
a relaxed, global comparison.
Single-concept and attack evaluations are posthoc extensions of existing test units, without
test tuning or an untouched-test claim. Multi-concept experiments use separate budgets
and acceptance gates; only ArXiv K=4 reaches a separately gated first-use test. IMDB’s
row-bootstrap intervals are duplicate-template diagnostics, not strong semantic-sample
evidence.
Access and metrics.C2U uses profiles and unlabeled calibration controls without can-
didate access. Raw, Delta-Edit, PISCES, Random, and the portable DARE-GRL mapper
are query-side comparisons; INLP , LEACE, CURE, and RAG-Emb-Ctrl additionally fit
on400labeled calibration examples per unit. Filter-and-Refill uses candidate/document
membership; Rank-Router uses target-document membership as a label-assisted benchmark
router. PISCES, DARE-GRL, CURE, and RAG Embedder Control are in-house diagnostic
adaptations, not official reproductions (Gur-Arieh et al., 2025; Ganin & Lempitsky, 2015; Yu
et al., 2024a; Kocak et al., 2025; Kim et al., 2025). Their access classes preclude a cross-class
dominance interpretation.
Binary H5 Drop is the reduction inany-target-in-top-fiveretrieval on target queries. CN is
control own-label Hit@5 loss; TE is the change in any-target-in-top-five retrieval on control
5

C2U in RAG
Table 2 Single-concept suppression and utility by dataset.Percentage-point changes from Raw,
averaged over four encoders and three seeds; Pooled weights datasets equally. Tints mark access
classes; full results are in Appendix E.
Method H5 Drop↑CN↓TE↓
Amazon
C2U +6.33 -0.64 -12.44
Delta-Edit 0.00 0.00 -0.08
PISCES -0.33 -0.25 +2.69
Random +1.17 -0.17 +3.00
INLP +12.17 -0.50 +18.86
LEACE +36.58 -0.61 -0.83
RAG-Emb-Ctrl +68.83 -0.69 -33.53
CURE 0.00 0.00 0.00
DARE-GRL +22.50 -0.11 +38.53
Filter-and-
Refill‡+0.92 -0.56 -4.14
Rank-Router§+7.00 -0.64 -13.83
ArXivr
C2U +5.67 -1.25 -7.64
Delta-Edit 0.00 +0.03 0.00
PISCES -0.17 -0.25 +1.33
Random +1.08 +4.92 +6.31
INLP +5.92 +5.00 +8.89
LEACE +1.00 +2.19 +16.19
RAG-Emb-Ctrl +69.50 +5.56 -21.03
CURE 0.00 0.00 0.00
DARE-GRL +22.67 +68.58 +58.61
Filter-and-
Refill‡+38.33 -1.97 -18.06
Rank-Router§+5.67 -1.08 -7.64
IMDBd
C2U +2.58 0.00 0.00
Delta-Edit 0.00 0.00 0.00
PISCES +1.83 0.00 0.00
Random +2.42 +2.03 +23.11
INLP 0.00 0.00 +16.67
LEACE +1.83 0.00 +8.33
RAG-Emb-Ctrl +30.75 0.00 +25.00
CURE 0.00 0.00 0.00
DARE-GRL +34.08 +25.00 +58.33
Filter-and-
Refill‡0.00 0.00 0.00
Rank-Router§+2.58 0.00 0.00Method H5 Drop↑CN↓TE↓
MNLI
C2U +3.42 -1.08 -6.03
Delta-Edit 0.00 +0.03 +0.03
PISCES -0.25 -0.08 +0.42
Random +1.58 +2.67 +3.67
INLP +4.00 +0.22 +5.83
LEACE +1.25 +0.06 +7.94
RAG-Emb-Ctrl +42.58 +1.33 -34.25
CURE 0.00 0.00 0.00
DARE-GRL +35.42 +27.72 +14.50
Filter-and-
Refill‡+10.50 -1.94 -11.83
Rank-Router§+3.83 -1.22 -6.36
TOFU
C2U +3.25 -0.44 -2.14
Delta-Edit 0.00 0.00 +0.03
PISCES +0.42 +0.08 +0.53
Random -0.92 +0.94 +0.92
INLP +6.42 +6.47 +9.00
LEACE +4.92 -0.08 +0.58
RAG-Emb-Ctrl +28.92 -1.19 -23.78
CURE 0.00 0.00 0.00
DARE-GRL +11.33 +24.58 +11.72
Filter-and-
Refill‡+4.00 -1.19 -7.31
Rank-Router§+4.58 -0.42 -2.67
Pooled
C2U +4.25 -0.68 -5.65
Delta-Edit 0.00 +0.01 -0.01
PISCES +0.30 -0.10 +0.99
Random +1.07 +2.08 +7.40
INLP +5.70 +2.24 +11.85
LEACE +9.12 +0.31 +6.44
RAG-Emb-Ctrl +48.12 +1.00 -17.52
CURE 0.00 0.00 0.00
DARE-GRL +25.20 +29.16 +36.34
Filter-and-
Refill‡+10.75 -1.13 -8.27
Rank-Router§+4.73 -0.67 -6.10
Raw (zero differences) is omitted; positive CN or TE is a cost.‡Candidate/document access;
§membership-assisted;rrelaxed global control;dduplicate-template diagnostic.
queries. Positive CN or TE is a cost; utility-clean means both rate changes are at most 10−9.
We report rate differences in percentage points (pp), with target-cosine changes in the full
results.
Dataset summaries average four encoders and three seeds; pooled summaries weight the
five datasets equally. Absolute levels and per-dataset intervals are in Table 4; statistical
definitions and resampling scope are in Section A.
4.2 Single-Concept Results
Suppression and utility.Table 2 shows larger C2U suppression on Amazon and ArXiv
than on MNLI and TOFU. Aggregate-derived pooled binary target Hit@5 decreases from
94.7333% to90.4833% , a4.25 pp drop rather than near-complete removal. H5 Drop is
6

C2U in RAG
0 25 50 75 100
share of queries (%)Target
Control5.3%
18.2% 76.5%
28.5% 27.3% 44.3%(a) Edit status
Unedited Below cap At cap
0.25
 0.00 0.25 0.50 0.75 1.00
Cosine to target-document centroid012density(b) Target-centroid similarity
Target before
Target after
Control (raw)
Figure 3 Bounded editing behavior.(a) Share of target queries ( n=6,000 ) and control queries
(n=18,000 ) that are unedited, edited below the 0.2rad cap, or edited at the cap; each bar sums to its
group. (b) Cosine to the target-document centroid for target queries before and after editing;Control
(raw)is the unedited control state.
positive with non-positive CN and TE on every dataset; pooled CN is −0.68 pp and TE
−5.65 pp.
Suppression–utility trade-offs.RAG-Emb-Ctrl reaches pooled H5 Drop +48.12 pp, with
utility/exposure costs including IMDB TE +25.00 pp; CURE is a utility-gated no-op and
does not suppress. DARE-GRL reaches +25.20 pp with CN +29.16 pp and TE +36.34
pp. Filter-and-Refill ( +10.75 pp) and Rank-Router ( +4.73 pp) exceed C2U with utility-
clean outcomes under stronger access. Full results and the Pareto view are in Section E
and Figure 5.
Encoder heterogeneity.The dataset means hide a wide encoder-level spread (Tables 9,
11 and 12). On Amazon, C2U’s H5 Drop ranges from 2.0pp with bge to 14.0 pp with e5;
ArXiv is the most uniform ( 4.3–8.0pp), MNLI spans 1.7–6.7pp and TOFU 1.3–5.0pp; on
IMDB only bge shows a drop ( 10.3 pp). e5 is the most responsive encoder on three datasets
and Qwen3 the least on three. CN is non-positive in 19of20dataset–encoder cells (MNLI
with Qwen3:+0.67 pp) and TE in all 20. The pooled 4.25 pp thus summarizes behaviours
that differ sevenfold across encoders.
Operating point.The deployed configuration is the validation-selected residual coefficient
β=1 with the 0.2-rad cap and 0.98 cosine floor; the validation sweep over larger caps and
smallerβfound no eligible alternative (Section A.6).
4.3 Analysis of Bounded Editing
Figure 3 summarizes the bounded operator on the posthoc rows. The gate admits most
queries: 94.7% of target and 71.5% of control queries receive an edit. The cap then binds:
76.5% of target and 44.3% of control queries sit at the 0.2rad boundary, so realized motion
is usually below the requested angle. Panel (b) shows after-edit target queries at lower
cosine to the target-document centroid. This view is consistent with the Hit@5 drops in
Table 2; the retrieval effect is measured there, not inferred from the shift. Two cautions
remain:Control (raw)is unedited while most control queries are edited, and centroid cosine
is one direction, not full semantic alignment. Full angle distributions and retrieval examples:
Figure 6b and Table 6.
4.4 Suppression and Exposure under Query Transforms
We compare each method with Raw under the same condition: standard queries, instruction
injection, or Zou-noise suffixes (Zou et al., 2023).Target density ∆is the change in the fraction
of top-five documents carrying the target label, averaged over target queries; it is not binary
H5 Drop.CER ∆records whether any target document enters the packed context ( B=512
7

C2U in RAG
Table 3 Suppression and control exposure under query transforms.Paired percentage-point
changes from Raw, averaged over the three query conditions (Standard, Injection, Zou-noise) per
dataset; Pooled weights the five datasets equally. Cell counts and signed worst values: Table 14; full
per-condition matrix: Table 15.
Amazon
Method Density∆↓CER∆↓Utility loss↓
C2U -16.4 -13.7 0.0
PISCES -1.2 +4.3 0.0
Random -0.4 +6.4 0.0
INLP -15.8 +17.6 0.0
LEACE -21.9 +2.4 0.0
RAG-Emb-Ctrl -34.7 -31.4 0.0
CURE 0.0 0.0 0.0
DARE-GRL -9.2 +46.0 0.0MNLI
Method Density∆↓CER∆↓Utility loss↓
C2U -14.9 -6.8 +0.3
PISCES -2.3 +0.4 -0.2
Random -3.9 +1.2 +1.2
INLP -17.7 +4.8 -0.9
LEACE -8.4 +3.3 -0.3
RAG-Emb-Ctrl -40.6 -47.5 +5.7
CURE 0.0 0.0 0.0
DARE-GRL -29.8 +6.6 +6.9
ArXivr
Method Density∆↓CER∆↓Utility loss↓
C2U -17.7 -11.7 -2.2
PISCES -2.0 +0.6 -0.2
Random -5.6 +4.6 +4.7
INLP -34.1 +2.0 +4.1
LEACE -14.0 +11.8 +2.2
RAG-Emb-Ctrl -72.4 -21.9 +5.3
CURE 0.0 0.0 0.0
DARE-GRL -50.7 +33.8 +63.6TOFU
Method Density∆↓CER∆↓Utility loss↓
C2U -4.4 -4.5 -0.1
PISCES -0.6 +1.3 +0.1
Random -0.1 +1.5 +0.9
INLP -5.6 +14.0 +2.2
LEACE -5.1 +3.2 0.0
RAG-Emb-Ctrl -20.0 -30.6 +0.4
CURE 0.0 0.0 0.0
DARE-GRL -0.3 +18.4 +13.8
IMDBd
Method Density∆↓CER∆↓Utility loss↓
C2U -11.1 -0.8 0.0
PISCES -0.6 +0.2 0.0
Random -5.4 +14.7 +0.7
INLP -16.7 +10.1 -0.1
LEACE -5.6 +5.7 +0.1
RAG-Emb-Ctrl -34.3 +8.9 -0.1
CURE 0.0 0.0 0.0
DARE-GRL -29.4 +27.9 0.0Pooledp
Method Density∆↓CER∆↓Utility loss↓
C2U -12.9 -7.5 -0.4
PISCES -1.4 +1.4 -0.1
Random -3.1 +5.7 +1.5
INLP -18.0 +9.7 +1.1
LEACE -11.0 +5.3 +0.4
RAG-Emb-Ctrl -40.4 -24.5 +2.2
CURE 0.0 0.0 0.0
DARE-GRL -23.9 +26.5 +16.9
Density: target-label fraction in the target-query top-5; CER: any-target occurrence in the packed
control context (budget B=512 , at most 50documents); utility: control own-label Hit@10 loss. CURE
never suppresses.rArXiv: relaxed global control.dIMDB: duplicate-template diagnostic.pPooled:
equal-weight mean over the five datasets (fifteen dataset–condition cells). Metric definitions and
posthoc scope: §4.4.
tokens, at most 50documents), averaged overcontrol queries. Attack utility is control own-
label Hit@10 loss, distinct from CN and TE. The two describe different populations, so a
decrease in one need not entail a decrease in the other; both are reported together.
Table 3 shows why the two must be read together. INLP lowers Amazon target density by
15.8 pp while raising control exposure by 17.6 pp: target documents leave the target queries’
top five but enter other users’ packed contexts. RAG-Emb-Ctrl lowers both on four datasets,
most strongly on ArXiv ( −72.4 and−21.9 pp), but raises IMDB exposure ( +8.9 pp) and
costs control utility (+5.3 pp on ArXiv,+5.7 on MNLI). C2U’s density and CER means are
negative on every dataset with utility changes between−2.2 and+0.3 pp.
Table 3 reports per-dataset means over the three conditions and an equal-weight pooled
block. Table 14 counts the 15dataset–condition cells on unrounded values (per-condition
matrix: Table 15), since an average can hide a condition-level increase: suppression requires
density ∆<0 ; divergence additionally requires CER ∆>0 . Counts are descriptive signs,
not significance tests; standard is a no-attack condition. C2U suppresses in all 15cells with
0/15 divergent. Its attack utility change still ranges from −5.2 to+2.6 pp, so this does not
establish universal utility preservation under attack.
INLP and LEACE diverge in 13/15 cells; PISCES, DARE-GRL, and Random in 11/15 .
RAG-Emb-Ctrl suppresses in all cells but diverges in the three IMDB cells (CER up to
+16.7 pp); the duplicate-template limitation applies. CURE’s 0/15 divergence is trivial: it
8

C2U in RAG
Software
eng.Artificial
intel.Neurons &
cognitionNumber
theory01020H5 Drop difference (pp)
ArXiv K=4: per-concept paired lift
Compose  averaged-direction ablation
Bars: means; dots: all 12 encoder seed units
Figure 4 ArXiv K=4 per-concept H5 lift.Compose minus the averaged-direction comparator:
means and all twelve encoder–seed unit differences; these conceptwise differences are not the min-
concept endpoint reported in the text.
suppresses in zero cells. Among evaluated suppressing profile-only methods, only C2U
avoids density–exposure divergence under these query transforms. Stronger-access filters
and routers are excluded by the attack contract. The full matrices and both posthoc arms
also appear in Section H.
4.5 Multi-Concept Composition
Each concept chas its own direction gc, gate sc(q), and augmented-bank coefficients. An-
gular requests start at 0.3s c(q)and shrink proportionally whenever their sum exceeds
min(0.3 max csc(q), 0.2) . Each allocated angle converts to a feasible coefficient scale; the
atom-coordinate deltas are added and pass through the shared trust-region update. The
comparator instead steers along the normalized average of unit concept directions under a
single max gate—a different gating and allocation rule.
The capacity-expanded ArXiv K=4 arm passes selection and independent confirmation
before first access to fresh test rows. Its endpoint is
1
12∑
uh
mincHcompose
u,c −mincHavg
u,ci
, (6)
where His per-concept H5 Drop. Test lift is +0.04125 , CI95 [+0.0325,+0.050625] , from
4,000 bootstraps of the twelve paired unit differences. All four compose concept cells are
positive; Figure 4 separately shows paired conceptwise differences. Filter-and-Refill-multi
remains a non-gating document-membership oracle reference (+0.7181).
Independent gates decide, per concept, whether a query carries evidence for that concept
and how much rotation it requests, so a query tied to one concept is edited mainly toward
that concept. The shared budget rescales all requests so their sum stays within the single-
concept budget: when several gates fire, each receives a smaller share. The per-concept lifts
in Figure 4 are all positive but uneven: +1.4 pp for Artificial Intelligence (positive in 10
of12units), +5.7 for Software Engineering, +6.9 for Neurons and Cognition, and +14.2
for Number Theory. The spread is consistent with concepts differing in gate activity and
direction separation, not with properties of individual atoms.
9

C2U in RAG
The expansion uses larger budgets than earlier arms, so the ladder is not a controlled
experiment on sample size alone. It is the only untouched-test arm here: its K=2 design
fails selection, and the earlier arms are validation-only or fail acceptance, as Section G
and Figure 8 document.
5 Conclusion
C2U edits the query, not the encoder or the index: a calibrated gate admits concept-evidenced
queries, the profile direction enters atom coordinates via a residual atom, and the difference
rotates the raw query by at most 0.2rad before retrieval. The selected configuration lowers
target Hit@5 by 4.25 pp pooled with non-positive control costs on every dataset mean, keeps
density and control exposure changes non-positive in every dataset–condition cell under
query transforms, and composes four ArXiv concepts with positive lift on each. It suits
fixed-index retrievers tolerating partial, bounded suppression rather than removal.
Limitations
Three limitations bound the claims. First, suppression is partial: the aggregate-derived after-
edit target Hit@5 stays near 90% pooled, so C2U reduces exposure rather than removing
a concept. Second, the data and evaluation scope is narrow: five English datasets, four
encoders, and 100/300-query units; IMDB has one template per role, ArXiv uses relaxed
global controls, single-concept and attack results are posthoc extensions of frozen test
units, and the untouched multi-concept claim covers ArXiv K=4 only, with capacity and
granularity limiting other datasets. Third, we evaluate retrieval only: generated-output
safety, multilingual or open-domain corpora, and defense-aware adaptive attackers are
untested, and low runtime access does not establish a measured cost advantage.
Ethical considerations
Retrieval-side suppression may reduce unwanted exposure but can also hide legitimate
evidence or enforce biased policies. Concept definitions, access to profiles, and effects
on non-target retrieval require transparent governance and auditing. This method is not
a substitute for privacy deletion, consent, output safeguards, or monitoring. The work
evaluates retrieval rather than safety of generated content.
References
Bang An, Shiyue Zhang, and Mark Dredze. Rag llms are not safer: A safety analysis of
retrieval-augmented generation for large language models.arXiv preprint arXiv:2504.18041,
2025.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag:
Learning to retrieve, generate, and critique through self-reflection. 2024.
M Bahraminasr and A Vafaei-Sadr. Imdb data from two generations (1979 to 2019). part one:
Dataset.CoRR, abs/2005.14147, 2020.
Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and
Stella Biderman. Leace: Perfect linear concept erasure in closed form.Advances in Neural
Information Processing Systems, 36:66044–66063, 2023.
Anh Bui, Trang Vu, Long Vuong, Trung Le, Paul Montague, Tamas Abraham, Junae Kim,
and Dinh Phung. Fantastic targets for concept erasure in diffusion models and where to
find them.arXiv preprint arXiv:2501.18950, 2025.
Min Chen, Weizhuo Gao, Gaoyang Liu, Kai Peng, and Chen Wang. Boundary unlearning:
Rapid forgetting of deep networks via shifting the decision boundary. InProceedings of the
IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7766–7775, 2023.
10

C2U in RAG
Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David
Stap, Jay Gala, Wissam Siblini, Dominik Krzemi´ nski, Genta Indra Winata, et al. Mmteb:
Massive multilingual text embedding benchmark.arXiv preprint arXiv:2502.13595, 2025.
Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, and Sijia Liu. Salun:
Empowering machine unlearning via gradient-based weight saliency in both image
classification and generation.arXiv preprint arXiv:2310.12508, 2023.
Rohit Gandikota, Joanna Materzynska, Jaden Fiotto-Kaufman, and David Bau. Erasing
concepts from diffusion models. InProceedings of the IEEE/CVF international conference on
computer vision, pp. 2426–2436, 2023.
Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropaga-
tion. InInternational conference on machine learning, pp. 1180–1189. PMLR, 2015.
Yoav Gur-Arieh, Clara Haya Suslik, Yihuai Hong, Fazl Barez, and Mor Geva. Precise in-
parameter concept erasure in large language models. InProceedings of the 2025 Conference
on Empirical Methods in Natural Language Processing, pp. 18986–19006, 2025.
Sebastian Hofstätter, Jiecao Chen, Karthik Raman, and Hamed Zamani. Fid-light: Efficient
and effective retrieval-augmented text generation. InProceedings of the 46th International
ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1437–1447,
2023.
Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao,
Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-
based input-output safeguard for human-ai conversations.arXiv preprint arXiv:2312.06674,
2023.
Taeyoun Kim, Jacob Mitchell Springer, Aditi Raghunathan, and Maarten Sap. Mitigating
Bias in RAG: Controlling the Embedder. InFindings of the Association for Computational
Linguistics: ACL 2025, pp. 18999–19024, 2025. doi: 10.18653/v1/2025.findings-acl.974.
Aysenur Kocak, Shuo Yang, Bardh Prenkaj, and Gjergji Kasneci. CURE: Controlled Unlearn-
ing for Robust Embeddings — Mitigating Conceptual Shortcuts in Pre-Trained Language
Models. InFindings of the Association for Computational Linguistics: EMNLP 2025, pp.
14283–14297, 2025. doi: 10.18653/v1/2025.findings-emnlp.771.
Xun Liang, Simin Niu, Zhiyu Li, Sensen Zhang, Hanyu Wang, Feiyu Xiong, Zhaoxin Fan,
Bo Tang, Jihao Zhao, Jiawei Yang, et al. Saferag: Benchmarking security in retrieval-
augmented generation of large language model. InProceedings of the 63rd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4609–4631, 2025.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni,
and Percy Liang. Lost in the middle: How language models use long contexts.Transactions
of the association for computational linguistics, 12:157–173, 2024.
Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C Lipton, and J Zico Kolter. Tofu:
A task of fictitious unlearning for llms.arXiv preprint arXiv:2401.06121, 2024.
Julian McAuley and Jure Leskovec. Hidden factors and hidden topics: understanding rating
dimensions with review text. InProceedings of the 7th ACM conference on Recommender
systems, pp. 165–172, 2013.
Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. Mteb: Massive text
embedding benchmark. InProceedings of the 17th Conference of the European Chapter of the
Association for Computational Linguistics, pp. 2014–2037, 2023.
Hadas Orgad, Bahjat Kawar, and Yonatan Belinkov. Editing implicit assumptions in text-
to-image diffusion models. InProceedings of the IEEE/CVF International Conference on
Computer Vision, pp. 7053–7061, 2023.
11

C2U in RAG
Huachuan Qiu, Shuai Zhang, Anqi Li, Hongliang He, and Zhenzhong Lan. Latent jailbreak:
A benchmark for evaluating text safety and output robustness of large language models.
arXiv preprint arXiv:2307.08487, 2023.
Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. Null
it out: Guarding protected attributes by iterative nullspace projection.arXiv preprint
arXiv:2004.07667, 2020.
Ayush Kumar Tarun, Vikram Singh Chundawat, Murari Mandal, and Mohan Kankanhalli.
Deep regression unlearning. InInternational Conference on Machine Learning, pp. 33921–
33939. PMLR, 2023.
Shang Wang, Tianqing Zhu, Dayong Ye, and Wanlei Zhou. When machine unlearning meets
retrieval-augmented generation (rag): Keep secret or forget knowledge?IEEE Transactions
on Dependable and Secure Computing, 23(2):2030–2045, 2026.
Adina Williams, Nikita Nangia, and Samuel R Bowman. A broad-coverage challenge corpus
for sentence understanding through inference. InProceedings of the 2018 conference of
the North American chapter of the association for computational linguistics: human language
technologies, volume 1 (long papers), pp. 1112–1122, 2018.
Jing Wu, Trung Le, Munawar Hayat, and Mehrtash Harandi. Erasing undesirable influence
in diffusion models. InProceedings of the Computer Vision and Pattern Recognition Conference,
pp. 28263–28273, 2025.
Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented
generation. 2024.
Le Yu, Bowen Yu, Haiyang Yu, Fei Huang, and Yongbin Li. Language models are su-
per mario: Absorbing abilities from homologous models as a free lunch. InForty-first
International Conference on Machine Learning, 2024a.
Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi,
and Bryan Catanzaro. Rankrag: Unifying context ranking with retrieval-augmented
generation in llms.Advances in Neural Information Processing Systems, 37:121156–121184,
2024b.
Tianzhe Zhao, Jiaoyan Chen, Yanchi Ru, Haiping Zhu, Nan Hu, Jun Liu, and Qika Lin.
Exploring knowledge poisoning attacks to retrieval-augmented generation.Information
Fusion, 127:103900, 2026.
Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson.
Universal and transferable adversarial attacks on aligned language models.arXiv preprint
arXiv:2307.15043, 2023.
12

C2U in RAG
Appendix
A Experimental Details
A.1 Implementation and Data Protocol
Retrieval framework.All locked experiments share one evaluation pipeline per unit:
the raw frozen-encoder document embeddings are encoded once per unit, shared bit-
identically across methods, and never mutated; retrieval similarity is the dot product of ℓ2-
normalized embeddings. The four encoders are sentence-transformers/all-MiniLM-L6-v2 ,
BAAI/bge-base-en-v1.5 ,intfloat/e5-base-v2 , and Qwen/Qwen3-Embedding-0.6B , each
pinned to a specific revision with a local snapshot manifest.
Index construction and budgets.The single-concept and attack cohorts use 100
target/ 300 control index documents and 100 target/ 300 control queries per unit, with
20target-profile documents. These budgets do not describe all multi-concept stages. Within
the single-concept allocation, profile, selection, confirmation, and test rows are disjoint.
Multi-concept budgets and the known v2 overlap defect are reported in Section G; blocked
capacity rows are not zero-filled or replenished from another split.
Lock ladder and reuse.Single-concept selection and independent confirmation precede
its method lock. Its final evaluation and the attacks extend previously frozen test units
without test tuning or an untouched-test claim. Recent baselines have their own selection,
confirmation, and posthoc-final records; their provenance-complete replay occurred after
final outcomes were observed and closes provenance, not prospective selection. Multi-
concept test access requires its separately specified acceptance lock. Only expanded ArXiv
K=4 first accesses its allocated test rows after passing that gate. Common infrastructure
does not make these one protocol or imply identical bootstrap populations across arms.
Hyperparameters and bank dimensions.The deployed single-concept setting appends
one row to each frozen N-atom bank, with 20target-profile documents, residual-coefficient
multiplier β=1 , cap 0.2rad, and floor 0.98. The frozen validation assets have N=64 for
ArXiv under every encoder. For Amazon, IMDB, MNLI, and TOFU, N=768 for MiniLM,
1,536 for bge/e5, and 2,048 for Qwen3; embedding dimensions are respectively 384,768,
and 1,024 . These are asset dimensions, not a newly selected bank-size sweep. The historical
top-30atom-profile truncation is not the deployed raw-space gate. Configuration selection
uses validation only; the posthoc evaluation does not retune it.
A.2 Evaluation Metrics
LetQtandQcdenote the sets of target queries and control queries, respectively. For a
given query q,TopK(q) represents the set of labels or categories associated with the top- K
retrieved documents. H5, TC, and CN are reductions (Base minus Method), while TE is an
exposure change (Method minus Base); the equations below specify each sign. The locked
metric names map as follows: Hit@ KDrop =hit_at_5_drop (H5), Target Cos Drop =TC,
Control/Neighbor Delta=CN, Target Exposure∆=TE.
Erasure and utility in the standard setting.
13

C2U in RAG
1.Target Hit@K Drop ( ↑)measures the reduction in retrieval success for target queries:
Hit@K Drop=1
|Qt|∑
q∈Q t
1[t∈TopKBase(q)]
−1[t∈TopKMethod(q)]
.(7)
2.Target Cos Drop ( ↑)quantifies the reduction in cosine similarity between target-
query representations and the target concept centroid. Let µtbe the centroid of all
target document embeddings in the base index:
∆cos=1
|Qt|∑
q∈Q t
cos(v base(q),µ t)
−cos(v method (q),µ t)
.(8)
3.Control/Neighbor Delta ( ↓)evaluates the impact on non-target retrieval utility. Let
yqbe the ground-truth control label forq∈ Q c:
Utility Delta=1
|Qc|∑
q∈Q c
1[yq∈TopKBase(q)]
−1[y q∈TopKMethod(q)]
.(9)
4.Target Exposure ∆(↓)measures the change in false-positive retrieval of target
documents by control queries:
Exposure∆=1
|Qc|∑
q∈Q c
1[t∈TopKMethod(q)]
−1[t∈TopKBase(q)]
.(10)
All three locked main hit metrics use K=5 : H5 and TE concern target-label presence, while
CN concerns control own-label presence. Attack utility instead uses own-label Hit@ 10.
Rate differences in percentage points multiply the above quantities by 100; target-cosine
differences are not rates.
Absolute levels and intervals.The main table’s Raw and C2U after-rates are aggregate-
derived. Per dataset, Raw H5 is reconstructed as a comparator’s after-hit rate plus its locked
H5 Drop; all seven reconstructions agree to floating-point precision. C2U’s own drop is then
subtracted. The five Raw/after pairs are 96.8333/90.5000% (Amazon), 97.8333/92.1667%
(ArXiv), 100.0000/97.4167% (IMDB), 94.5833/91.1667% (MNLI), and 84.4167/81.1667%
(TOFU). Equal-weight pooling gives 94.7333/90.4833% . These are not new measurements.
The per-dataset binary H5 Drop intervals reuse the existing standard/no-attack matched
replay and its 4,000 paired-row bootstrap samples. There is no native pooled H5 CI in the
final summary; we neither average interval endpoints nor substitute validation intervals.
IMDB row resampling does not create independent templates.
Robustness under query transforms.We evaluate Standard (no attack), Injection, and
Zou-noise. Changes are relative to Raw under the same condition, paired by query.
14

C2U in RAG
Table 4 Absolute target Hit@5 and dataset uncertainty.Raw and C2U levels are aggregate-
derived; intervals describe C2U’s binary H5 Drop.
Dataset Raw H5 (%) C2U H5 (%) Drop (pp) 95% CI (pp)
Amazon 96.83 90.50 +6.33 [4.83, 7.92]
ArXivr97.83 92.17 +5.67 [4.42, 7.00]
IMDBd100.00 97.42 +2.58 [1.75, 3.50]
MNLI 94.58 91.17 +3.42 [2.25, 4.50]
TOFU 84.42 81.17 +3.25 [1.92, 4.58]
Intervals reuse the existing standard/no-attack paired-row replay (4,000 bootstrap samples; 1,200
target-row contributions per dataset). No pooled H5 interval is reported.rRelaxed global control.
dIMDB has one unique template per target/control role; its row-bootstrap interval is descriptive, not
uncertainty over independent semantic examples.
•Target density ∆(pp) ( ↓): for D5(q) =1
5∑j∈Top5(q) 1[j∈ D t], this is
100E Qt[DMethod
5−DRaw
5]. It is a target-document fraction, not binary H5.
•CER∆(pp) (↓): let Docs(q; 512, 50) be the ranked context packed under the
locked token accounting, with budget 512, at most 50documents, and partial
last-document inclusion where allowed. Then
CER(q) =1[Docs(q; 512, 50)∩ D t̸=∅]. (11)
The reported delta is 100E Qc[CERMethod−CERRaw]: the population iscontrol
queries.
•Utility Drop (pp) ( ↓): the decrease in control own-label Hit@ 10under the same
condition. CN instead uses own-label Hit@5; TE uses target Hit@5.
Suppression and divergence counts use unrounded dataset–condition signs, not significance
tests.
A.3 ACR Training and Reconstruction
ACR trains a dictionary and lightweight projection head atop a frozen encoder. Atoms have
no pre-assigned semantic labels and start from standard-normal samples. Given frozen
representationh x, a bilinear compatibility score and temperature-scaled softmax give
wx,i=exp(sim(h x,ai)/τ)
∑jexp(sim(h x,aj)/τ). (12)
The weights are peaked but not literally sparse. With rows aiinA, training reconstruction is
zx=A⊤wx
∥A⊤wx∥2=∑iwx,iai
∥∑iwx,iai∥2. (13)
This reconstruction is used by the training objective. At deployed retrieval the index contains
raw encoder vectors, and the raw query receives only equation 4; replacing it by zxwould
define a different method. Entropy concentration does not prove that the deployed additive
edit has small support. The full training objective follows.
A.4 Optimization Setup
Training dynamics.To preserve the retrieval geometry of the pre-trained dense retriever,
we freeze the encoder parameters throughout ACR training. Optimization is restricted to
the atom bank Aand the lightweight projection head. We use AdamW with a batch size of
32 and an initial learning rate of 10−3, aReduceLROnPlateauscheduler (factor 0.5, patience
one epoch), a 90/10 train-validation split, and early stopping within a small epoch budget.
15

C2U in RAG
Hyperparameter configuration.The reported training setup uses alignment penalty
weight λalign=1.0 , entropy regularization weight λent=0.02 , retrieval margin mr=0.3 ,
and scoring temperature τ=0.1 . Bank dimensions vary as specified in Section A.1; the
locked reproduction probe establishes bit-identical recovery of its MNLI/MiniLM/seed-1
checkpoint, not one universal checkpoint or bank size for all units.
Training objective (ACR).We freeze the transformer encoder and optimize only the
atom bank and a lightweight projection head. For each triplet (q,d+,d−), ACR outputs a
composed Euclidean embedding eacr
xand atom weights wx∈∆|A|−1forx∈ {q,d+,d−}. We
L2-normalize the composed embeddings and compute cosine similarities by dot product:
zx=eacr
x
∥eacrx∥2+ε, cos(z a,zb) =z⊤
azb.
Let ebase
x denote the frozen dense retriever embedding for input x, and let ¯ebase
x =
ebase
x/(∥ebase
x∥2+ε). The total loss is:
L=L tri+λ alignLalign+λ entLent,
where:
(i) Cosine triplet retrieval loss.
Ltri=max 
0,m r−cos(z q,zd+)
+cos(z q,zd−)
.
(ii) Cosine alignment loss.We encourage each ACR-composed embedding to remain close
to the corresponding frozen retriever embedding:
Lalign=∑
x∈{q,d+,d−}
1−cos(z x,¯ebase
x)
.
(iii) Entropy penalty on atom weights.To obtain sharper training-weight distributions, we
penalize entropy; this does not imply sparsity of the deployed least-squares coefficient edit:
Lent=∑
x∈{q,d+,d−}
−|A|
∑
k=1wx,klog(w x,k+ϵ)
.
Here,m r>0 is the triplet margin, andλ align,λent>0 are scalar weights.
A.5 Historical Profile Subtraction (Not the Deployed Edit)
The historical formulation averages the weights of 20profile documents and normalizes
their non-negative mass:
˜πc=ReLU(E x∈D c[wx]),π c=˜πc/∑i˜πc[i]. (14)
It retains the top Katom=30 coordinates and renormalizes. The original exposition used an
atom-cosine activity
αx=max(0, cos( ˆwx,ˆπc)), (15)
followed by
w′
x∝ReLU(w x−λα xπc). (16)
The tested subtraction family includes native/hybrid weight normalizations; these historical
expressions do not specify the selected additive operator. In particular, αxis not an input to
the deployed gate, and its vanishing is not the deployed identity condition. Under the locked
bank and deployed runtime the measured subtraction variants are exact inactive identities,
with approximately 2×10−5rad in the diagnostic angular readout. That mechanism finding
does not prove every subtraction method must fail.
16

C2U in RAG
A.6 Numerical Details and Validation-Only Mechanism Probes
Diagonal-O2 and G0.The runtime names map onto the main-text terms: the negative
diagonal-O2 direction is the diagonally preconditioned direction, the continuous G1 ramp
with its conservative G0 fallback is the calibrated gate, and the delta exit is the raw-query
trust-region update. For normalized controls x, let mi=E c[x2
i]and ¯m=d−1∑imi. The
diagonal uses Dii=0.9m i+0.1 ¯m+0.001 , with an implementation denominator floor. The
runtime first forms v=normalize(µ t−µ c)and−0.1D−1v/(v⊤D−1v), then normalizes
this negative vector for least-squares routing. The means are not separately normalized
for this direction; they are separately normalized only for the gate score. Non-finite or
degenerate directions fail closed. When G1 quantiles are non-finite or not increasing, G0 uses
a conservative calibration order-statistic threshold with declared control false-positive rate
0.02; equality is ineligible. The median-to- 95th-percentile G1 ramp is otherwise continuous.
Least squares and numerical branches.The solve uses true least squares on A⊤in float64.
Above the residual threshold, hβ=p+βe ; below 10−6the appended row is normalized
pand its coefficient is zero. A degenerate projected vector also fails closed rather than
dividing by zero. The fallback row is bookkeeping, not a new independent direction. Exact
in-span gis recovered by its projected coefficients; a small nonzero omitted residual causes
truncation, and coefficient/bank rounding adds numerical error. The angular scale uses the
unit direction of hβ, but multiplies the original coefficient vector. Gate-off, zero-angle, and
non-positive-denominator branches produce zero scale. FP32 subtraction is performed after
constructing and casting padded and steered weights. The exit checks cos(float32(θ cap))
against the floor, tries η=1 , and otherwise keeps the feasible lower bisection endpoint
in[0, 1] . Inactive rows are copied from the original query array. Finite bisection does not
guarantee the exact real-valued boundary point.
Coverage and selected residual coefficient.Across 120 diagnostic validation units,
true least-squares projection ratio averages 0.895 , with 96/120 directions recorded as fully
covered. Coverage is not suppression and is not evidence of atom necessity. The residual-
coefficient ladder β∈ {1, 0.5, 0.25} selects β=1 : confirmation primary H5 +0.0644 , CI95
[+0.0467,+0.0864] , with 3/3 seeds and 4/4 encoders positive. This validation endpoint
is distinct from the posthoc test pooled value. Lowering βchanges only the residual
contribution, not the whole edit. The runtime rejects β=0 ; algebraically setting it to zero
would still retain the projected component and would not generally yield identity.
Operating-point sweep: negative result.Default cos(0.2) =0.9800665778>0.98 , so
cap0.2is tighter than the floor. Only caps above arccos(0.98) =0.2003348423 . . . rad are
floor-limited. The β=1 settings at caps 0.3, 0.5, 1.0 tie reference primary H5 +0.049444
(five-dataset pooled +0.033000 ), while target-cosine values still change slightly. This ob-
served metric tie is not geometric identity with cap 0.2. Atβ=0.5 , cap 0.5, primary H5
falls to +0.043056 (five-dataset pooled +0.029167 ). Every setting, including the reference,
is ineligible under this sweep because TOFU CN is +0.000278>0+10−9. A reference-only
confirmation replay reproduces the locked 480unit-metrics exactly. The existing selected
configuration stands; the sweep authorizes no new successful operating point. Reference
cap saturation is 0.5813 in validation and 80.7% /61.9% of active target/control edits at test;
default saturation is at the nominal cap, not at arccos(0.98).
Subtraction revival probes: both NO-GO.Frozen query-weight overlap with the top-
30profile is approximately 0.0011 under the v1 operationalization. Four knob-only bank
configurations (reproduction, 1536 atoms, ten epochs, and their combination, with the repro-
duction baseline corrected to one epoch) fail the revival criterion: subtraction’s applied-angle
diagnostic stays near 2×10−5rad and its runtime output stays inactive. The reproduction
checkpoint is bitwise identical; augmented H5 is unchanged. The probe’s mean-weight
cosine near 0.998 is a different statistic, not a substitute for applied-angle evidence. A
second probe adds an off-diagonal orthogonality loss with weights 1and 10: the bank
moves, but augmented H5/CN/TE remain unchanged and subtraction stays inactive. Both
17

C2U in RAG
Dataset Domain Target Concept (t) Control Attribute (y) Query Source
ArXivScientific Papers Specific Topic ArXiv Category Title / Abstract
TOFUFictional QA Sensitive Topic Author Identity Natural Question
AmazonProduct Reviews Sentiment Product Category Review Title
IMDBMovie Reviews Movie Genre Sentiment Synthetic Template
MNLILogic Inference Text Genre Logic Label Hypothesis
Table 5Overview of datasets used in evaluation. The Target Concept represents the semantic
information to be suppressed, while the Control Attribute represents the utility signal to be maintained.
decisions are NO-GO; neither conditional full arm triggers. These bounded interventions
do not exhaust possible losses or establish a general impossibility theorem.
A.7 Compute Environment
The original training and evaluation ran on a single GPU of a workstation with 4 NVIDIA
RTX 6000 Ada Generation GPUs (no distributed training); the locked re-evaluation reported
here ran on a single NVIDIA L20 GPU. C2U does not train a large language model, fine-tune
a generator, or require retriever-index rebuilding during concept unlearning; optimization
is restricted to the atom bank and lightweight projection/composition modules. Legacy
wall-clock latency, peak-memory, and checkpoint-size figures from the original pipeline are
not artifact-proven under the locked protocol and are therefore not reported.
B Dataset Details
We evaluate across five datasets covering scientific literature, fictional biographies, product
reviews, movie reviews, and logical inference. For each dataset, we define theTarget
Conceptto be suppressed and theControl Attributeto be preserved. Table 5 summarizes the
definitions.
Why these datasets.We select these datasets to stress-test concept suppression under
heterogeneous retrieval conditions: (i) domain-specific terminology and hierarchical labels
(ArXiv ), (ii) identity-linked fictional knowledge where topic-level suppression can interact
with author identity ( TOFU ), (iii) real-world sentiment versus topical content confounding
(Amazon ), (iv) an adversarial query formulation that explicitly mentions the target concept
(IMDB ), and (v) logic-conditioned matching where the control signal is relational rather than
topical ( MNLI ). Together, they cover both natural and synthetic queries, multiple domains,
and distinct target/control factorization patterns.
B.1 Hierarchical Scientific Knowledge (ArXiv)
We construct a hierarchical dataset using the Cornell-University/arxiv corpus (Kaggle:
https://www.kaggle.com/dsv/7548853 , accessed Jan. 28, 2026) to simulate concept suppres-
sion in specialized domains. The dataset uses the arXiv taxonomy, where theControl
Attributeis defined as the primary category (L1, e.g., cs.CR ,cs.CV ). TheTarget Concept
(L2) is derived via heuristic keyword matching within the abstract, such as mapping terms
like “attack” or “malware” to the conceptSystem Attackswithin thecs.CRdomain.
For the retrieval task, each document dconsists of the concatenated paper title and abstract.
Each query qis generated using a stochastic strategy: 50% of queries use the exact paper
title to simulate navigational search, while the remaining 50% use the first sentence of the
abstract to simulate semantic search. The locked protocol evaluates the L2 conceptSystem
Attacksunder a relaxed global control (the matched-L1 control pool is capacity-deficient),
and this label is preserved in every table.
18

C2U in RAG
B.2 Fictional Knowledge Unlearning (TOFU)
The TOFU (Task of Fictional Unlearning) dataset (Maini et al., 2024) consists of question-
answer pairs about fictional authors, designed for unlearning tasks. We adapt this dataset
for retrieval by treating the answer as the document source. TheControl Attributeis the
Author ID, representing the identity that should be preserved. TheTarget Conceptis the
semantic topic of the fact, such asFamily,Location, orWork, which is classified via keyword
matching on the question text.
The retrieval setup involves documents formatted as biography records and queries taken
directly from the dataset questions. This structure tests whether the retrieval system can
suppress topic-level evidence while preserving author-linked retrieval utility. TOFU remains
an entity-centric regime: target and non-target facts are tightly coupled around author
identifiers, which is why conservative, trust-region-bounded steering is appropriate here.
B.3 Classification and Inference Tasks
Product Reviews ( Amazon ).Adapted from the amazon_polarity dataset, which is derived
from the Amazon Reviews corpus (McAuley & Leskovec, 2013) and is also included in
text-embedding benchmarks such as MTEB/MMTEB (Muennighoff et al., 2023; Enevoldsen
et al., 2025), this task aims to suppress sentiment information while preserving product
categorization. TheTarget Conceptis sentiment (Positive/Negative), and theControl
Attributeis product category, such asBooksorElectronics, extracted via keyword filtering.
Queries are derived from review titles, while documents contain full review content.
Movie Reviews ( IMDB ).In contrast to the Amazon dataset, the IMDB dataset (Bahraminasr
& Vafaei-Sadr, 2020) inverts the roles of sentiment and content. Here, theTarget Conceptis
movie genre, such asSci-FiorRomance, and theControl Attributeis sentiment. Since the
original dataset lacks titles, we generate synthetic queries using templates that explicitly
mention the target concept, such as “Find a Positive review about a Sci-Fi movie”. This
presents a challenging setting where the query explicitly seeks the concept intended for
suppression. Under the locked split, the IMDB evaluation rows collapse to one unique target
and one unique control query text; we retain the row with itsduplicate-template diagnostic
label and treat its zero-variance metrics as a diagnostic, not as high-confidence evidence.
Logical Inference ( MNLI ).Adapted from the multi_nli dataset (Williams et al., 2018), this
task evaluates concept suppression in logical inference contexts. TheTarget Conceptis the
source text genre, such asGovernmentorFiction, and theControl Attributeis the NLI label
(Entailment,Neutral, orContradiction). The retrieval task is formulated such that the query is
the hypothesis and the document is the premise.
C Baseline Implementation Details
All baselines are evaluated under the same locked pipeline: the same per-unit raw-base
index, the same query IDs and seeds, and the same metric definitions. Access-class labels
are part of each method’s identity and appear in every table.
C.1 Linear Concept Erasure Methods
INLP (Iterative Nullspace Projection).Proposed by Ravfogel et al. (2020), INLP iteratively
trains linear probes to predict the target concept and projects representations onto the
nullspace of the probe weights. In the locked protocol INLP fits on the 400labeled calibration
rows of each unit, placing it in theprofile-only + labeled calibration fitaccess class.
LEACE (Least-Squares Concept Erasure).Belrose et al. (2023) introduced LEACE as
a closed-form erasure transform minimizing dependence between representations and
concept labels. It uses the same labeled calibration fit as INLP , and the runner fails closed
when the erasure library is missing, unversioned, fails fitting, or returns an empty eraser.
19

C2U in RAG
C.2 Model Editing Baselines (Diagnostic Adaptations)
PISCES.PISCES (Gur-Arieh et al., 2025) was proposed for generative safety in LLMs.
Our in-house adaptation treats the output projection layer of the retriever as the editable
component, identifies directions with high cosine similarity to the target concept centroid,
and masks them.
DARE-GRL.We construct a training-based baseline combining Gradient Reversal Lay-
ers (Ganin & Lempitsky, 2015) with DARE-style model merging (Yu et al., 2024a): a
lightweight mapper is trained with a GRL objective against a concept adversary, and the
delta weights are sparsified and rescaled before merging. The port to the retrieval encoder
is our in-house adaptation.
C.3 Recent Embedding-Space Baselines (Diagnostic Adaptations)
CURE.CURE (Kocak et al., 2025) extracts concept-irrelevant content through a frozen
pretrained encoder with GPT-4o concept labels and a task classifier, then reconstructs a
controlled, debiased representation. The faithful pipeline requires raw text, concept/task
supervision, and a downstream classifier, none of which the frozen query-only retrieval
contract exposes; our in-house adaptation fits a content/reversal-inspired projection on the
frozen calibration vectors of each unit, using the same 400labeled calibration examples
as INLP and LEACE, transforms queries only, and preserves the raw document index.
Under the preregistered utility gate the adapted adapter is locked to strength zero, so every
reported CURE cell is an exact identity operation.
RAG Embedder Control.Kim et al. (2025) debias a RAG embedder with SimCSE-style
contrastive reverse-bias training (PEFT/WiSE-FT). Full fidelity requires trainable embedder
weights and corpus-side re-encoding, which the frozen offline vector cache does not autho-
rize; our portable adaptation fits a deterministic reverse-bias linear adapter on the frozen
calibration vectors and labels of each unit ( 400labeled calibration examples), transforms
queries only, and keeps the raw fixed document index. It is a portable adaptation, not an
official full-embedder reproduction.
C.4 Other Query-Side and Stronger-Access References
Raw.The identity (no-edit) transform; the per-query paired reference for the attack arm
(its relative metrics are exactly zero by construction).
Delta-Edit.A fidelity-preserving delta variant that adds an FP32 edited-minus-original
atom delta to the raw query; at the locked operating point it is a near-identity control.
Filter-and-Refill.Removes target documents from the candidate set and refills the top- K
from the remainder; it consumes candidate/document membership and is reported only as
a stronger-access reference.
Rank-Router.A relaxed boundary router that consumes target-document membership
labels; it is a label-assisted benchmark router, not an unlabeled query-only deployment
method, and is reported only as a reference.
Random Perturbation.Isotropic Gaussian noise ϵ∼ N(0,σ2I)added to query embed-
dings, with σcalibrated to match the average edit norm of C2U, controlling for generic
representation degradation.
20

C2U in RAG
Table 6 Case study: target-suppression flips under C2U.One locked test-stage target query per
dataset, selected by the pre-registered rule (base Hit@5 =1→ method Hit@5 =0flips only; maximum
target-cosine drop; deterministic tie-breaks), frozen before any query text was read. Top-5 columns
list the retrieved document labels. Illustrative only; drawn from the locked posthoc extension of the
frozen test units (evaluated exactly once, no test-time tuning).
Dataset Query Target Base top-5 labels Method top-5 la-
belsTC drop Angle
AmazonWORTH THE MONEY Positive Negative, Positive,
Negative, Negative,
PositiveNegative, Negative,
Negative, Negative,
Negative+0.101 0.196
ArXiv Research about AI-based code generators
have gained a fundamental role in assist-
ing developers in writing software starting
from natural language (NL)System Attacks Software Engi-
neering, Software
Engineering, Soft-
ware Engineering,
System Attacks, Ar-
tificial IntelligenceSoftware Engi-
neering, Software
Engineering, Arti-
ficial Intelligence,
Software Engi-
neering, Machine
Translation+0.166 0.200
IMDBPositive feedback on Horror cinema Horror Romance, Action,
Romance, Sci-Fi,
HorrorAction, Sci-Fi, Ro-
mance, Sci-Fi, Sci-Fi+0.103 0.200
MNLIThe marauders rounded up the villagers. fiction travel, travel, travel,
travel, fictiontravel, travel, travel,
travel, travel+0.114 0.200
TOFU Who is the person described in the text:
’In Luka Khachidze’s works, readers can
expect a diverse range of characters, but
predominantly they are male leads explor-
ing their love interests and overcom. . .Work General, General,
General, Work,
WorkGeneral, General,
General, General,
Family+0.123 0.200
D Case Study: Target-Suppression Flips
Table 6 shows one locked test-stage target query per dataset where the base retriever
returns the target in the top-5 and C2U removes it entirely. The selection rule was frozen
before any query text was read: among target rows with a base hit and a method miss
and non-zero applied angle, choose the maximum target-cosine drop with deterministic
tie-breaks. Query texts were joined from the source parquets only after the rule was frozen.
These success-selected examples illustrate target-label removal under a small, trust-region-
bounded rotation ( ≈0.2 rad, at the cap). They are not representative-frequency estimates
and do not establish unchanged control neighborhoods; aggregate CN and TE measure
those separate outcomes.
Selection mechanics.The rule behind Table 6 is a deterministic function of locked per-
query rows: among posthoc-extension target rows with a base top-five target hit, a method
miss, and a non-zero applied angle, it takes the maximum target-cosine drop under de-
terministic tie-breaks. Ties depend on row content alone, never on judgments made after
reading the queries. Re-running it on the same locked rows therefore returns the same five
queries: reproducible selections, not hand-picked anecdotes.
Reading the rows.In each row the base top five carries exactly one target-labeled docu-
ment and the edited list carries none—the flip the rule selects for. The vacated slot is refilled
by the next non-target candidates, so the list stays full while the target label disappears.
The rotation behind each flip stays inside the trust region ( ≈0.2 rad, at the cap), and the
target-cosine drop is positive on all five datasets, matching the pooled Hit@5 Drop of Table 2.
The five rows span one query per dataset, so the illustration covers every corpus rather than
a single favorable one.
Scope of the examples.The flips give the pooled effect a concrete shape: C2U can
remove a target document from the top five through one bounded, query-side rotation while
the raw index and the control population stay fixed. They say nothing about how often
such removals occur—frequency is the pooled tables’ question—and nothing about control
neighborhoods, which aggregate CN and TE measure separately (Tables 11 and 12). Nor do
they isolate why a particular document leaves the list: the edited query moves every score
within the drift bound, and the flip is an outcome of the full ranking, not of a per-document
operation. Drawn from the same locked test units as those tables and evaluated exactly
21

C2U in RAG
once with no test tuning, the five examples carry the posthoc-extension label of Section H,
not an untouched-test claim.
D.1 Legacy Qualitative Atom Examples
The following table from the original appendix is retained as qualitative context for the flips
above; its numbers are paper-reported from the original evaluation pipeline and have no
locked counterpart.
[Paper-reported, original evaluation pipeline]
Table 7 Representative high-usage atoms on ArXiv (paper-reported, original pipeline).Labels
are post-hoc summaries of high-activation documents and are used only for qualitative interpretation.
Atom ID Post-hoc Label Purity Representative Keywords
492 Genomics 0.60 assembly, datasets, single
485 Biomolecules 0.50 protein, network, security
901 GR and Cosmology 0.50 structures, field, nucleat
564 Biomolecules 0.40 domain, learning, neto2
475 Crypto and Security 0.30 vulnerabilities, software
E Full Dataset–Method and Dataset–Encoder Results
Table 8 retains the complete dataset–method rendering.
Tables 9 to 12 report the complete dataset ×encoder decomposition behind the pooled cells
of Table 2, one grid per metric; every cell is the mean over the three locked seeds, and the
per-seed rows are retained in the locked artifact. Utility-clean status and the marking rule
follow the pre-registered Pareto criterion of Section 4.2: CN ≤10−9and TE ≤10−9, H5
Drop compared only inside the exact profile-only class, and no cross-class dominance; bold
and underline mark the best and second-best values inside the profile-only access class
of each row. ArXiv carries therelaxed global controllabel and IMDB theduplicate-template
diagnosticlabel throughout.
F Visual Diagnostics
The five panels in this section are descriptive renderings of locked outcomes under the
unified figure standard of this paper: they recompute or replot rows that the locked chains
already contain, and they add no new evidence. Each caption states its populations, denom-
inators, and provenance. The paragraphs below connect each panel to the main-text claim it
supports and state what the panel does not show.
Pareto view.Figure 5 plots every displayed method’s pooled suppression against its
pooled utility cost, one point per method and dataset, pooled over four encoders and three
seeds, with colors marking the locked access classes. CURE coincides with Raw because
its utility gate locks it to a zero-strength identity, and DARE-GRL’s larger suppression sits
together with its large positive CN and TE costs.
Target-cosine shift.Figure 6a shows per-query cosines to the target-document centroid on
locked test rows, base versus steered, one panel per dataset. The steered target distribution
shifts left of base in every dataset: queries move away from the target centroid, which
is the geometric reading of the Hit@5 drops in Table 2. The figure shows target rows
only; Figure 7b adds the control distribution, and the main-text Figure 3 is the pooled
three-population companion.
Applied-angle readout.Figure 6b is the trust region’s readout: the angle the deployed
edit actually applies to each locked test query. Gate-off rows are exactly zero, since a closed
22

C2U in RAG
Table 8 RAG retrieval unlearning under the locked unified protocol.Test-stage behaviour of
the validation-selected configuration on the 60 frozen units (24,000 queries). Bold and underline are
computed only inside the profile-only access group; no cross-access-group dominance is claimed. The
Pooled block pools all twenty dataset×encoder cells per method.
MethodHit@5
Drop↑Target
Cos.↑Ctrl./Neigh.
∆↓Target Exp.
∆↓
Amazon
C2U 0.0633 0.0565 -0.0064 -0.1244
Raw 0.0000 0.0000 0.0000 0.0000
Delta-Edit 0.0000 0.0015 0.0000 -0.0008
PISCES -0.0033 0.0014 -0.0025 0.0269
Random 0.0117 0.2226 -0.0017 0.0300
INLP 0.1217 0.0394 -0.0050 0.1886
LEACE 0.3658 0.0386 -0.0061 -0.0083
RAG-Emb-Ctrl 0.6883 0.0874 -0.0069 -0.3353
CURE 0.0000 0.0000 0.0000 0.0000
DARE-GRL 0.2250 0.5455 -0.0011 0.3853
Filter-and-Refill‡0.0092 0.0000 -0.0056 -0.0414
Rank-Router§0.0700 0.0541 -0.0064 -0.1383
ArXiv
C2U 0.0567 0.0859 -0.0125 -0.0764
Raw 0.0000 0.0000 0.0000 0.0000
Delta-Edit 0.0000 0.0025 0.0003 0.0000
PISCES -0.0017 0.0152 -0.0025 0.0133
Random 0.0108 0.2434 0.0492 0.0631
INLP 0.0592 0.1387 0.0500 0.0889
LEACE 0.0100 0.0627 0.0219 0.1619
RAG-Emb-Ctrl 0.6950 0.2217 0.0556 -0.2103
CURE 0.0000 0.0000 0.0000 0.0000
DARE-GRL 0.2267 0.5746 0.6858 0.5861
Filter-and-Refill‡0.3833 0.0000 -0.0197 -0.1806
Rank-Router§0.0567 0.0838 -0.0108 -0.0764
IMDB
C2U 0.0258 0.0766 0.0000 0.0000
Raw 0.0000 0.0000 0.0000 0.0000
Delta-Edit 0.0000 0.0016 0.0000 0.0000
PISCES 0.0183 0.0017 0.0000 0.0000
Random 0.0242 0.2643 0.0203 0.2311
INLP 0.0000 0.0805 0.0000 0.1667
LEACE 0.0183 0.0146 0.0000 0.0833
RAG-Emb-Ctrl 0.3075 0.0695 0.0000 0.2500
CURE 0.0000 0.0000 0.0000 0.0000
DARE-GRL 0.3408 0.6527 0.2500 0.5833
Filter-and-Refill‡0.0000 0.0000 0.0000 0.0000
Rank-Router§0.0258 0.0749 0.0000 0.0000MethodHit@5
Drop↑Target
Cos.↑Ctrl./Neigh.
∆↓Target Exp.
∆↓
MNLI
C2U 0.0342 0.0531 -0.0108 -0.0603
Raw 0.0000 0.0000 0.0000 0.0000
Delta-Edit 0.0000 0.0007 0.0003 0.0003
PISCES -0.0025 0.0099 -0.0008 0.0042
Random 0.0158 0.2260 0.0267 0.0367
INLP 0.0400 0.1060 0.0022 0.0583
LEACE 0.0125 0.0301 0.0006 0.0794
RAG-Emb-Ctrl 0.4258 0.1680 0.0133 -0.3425
CURE 0.0000 0.0000 0.0000 0.0000
DARE-GRL 0.3542 0.5446 0.2772 0.1450
Filter-and-Refill‡0.1050 0.0000 -0.0194 -0.1183
Rank-Router§0.0383 0.0506 -0.0122 -0.0636
TOFU
C2U 0.0325 0.0592 -0.0044 -0.0214
Raw 0.0000 0.0000 0.0000 0.0000
Delta-Edit 0.0000 0.0003 0.0000 0.0003
PISCES 0.0042 0.0016 0.0008 0.0053
Random -0.0092 0.2293 0.0094 0.0092
INLP 0.0642 0.0431 0.0647 0.0900
LEACE 0.0492 0.0048 -0.0008 0.0058
RAG-Emb-Ctrl 0.2892 0.0315 -0.0119 -0.2378
CURE 0.0000 0.0000 0.0000 0.0000
DARE-GRL 0.1133 0.5547 0.2458 0.1172
Filter-and-Refill‡0.0400 0.0000 -0.0119 -0.0731
Rank-Router§0.0458 0.0497 -0.0042 -0.0267
Pooled
C2U 0.0425 0.0662 -0.0068 -0.0565
Raw 0.0000 0.0000 0.0000 0.0000
Delta-Edit 0.0000 0.0013 0.0001 -0.0001
PISCES 0.0030 0.0060 -0.0010 0.0099
Random 0.0107 0.2371 0.0208 0.0740
INLP 0.0570 0.0816 0.0224 0.1185
LEACE 0.0912 0.0302 0.0031 0.0644
RAG-Emb-Ctrl 0.4812 0.1156 0.0100 -0.1752
CURE 0.0000 0.0000 0.0000 0.0000
DARE-GRL 0.2520 0.5744 0.2916 0.3634
Filter-and-Refill‡0.1075 0.0000 -0.0113 -0.0827
Rank-Router§0.0473 0.0626 -0.0067 -0.0610
Note:Values pool the four encoder cells per dataset.‡candidate/document-access method.§
target-membership label-assisted relaxed boundary router. Raw is the identity (no-edit) reference.
CURE and RAG-Emb-Ctrl are adapted recent baselines evaluated as a posthoc extension under the
same locked protocol and frozen units. ArXiv uses a relaxed global control; IMDB carries the
duplicate-template diagnostic label. Row tints encode the locked access class (same colors as Figure 5);
the C2U row is highlighted.
gate is an exact identity. Active edits concentrate at the 0.2-rad cap, slightly tighter than
arccos(0.98) ; among edited queries, 80.7% of the 5,681 target edits and 61.9% of the 12,876
control edits sit at the cap, whereas the main-text Figure 3 uses all queries of each group as
denominator ( 76.5% and 44.3% ). Saturation at the cap is what the operating-point sweep of
Section A.6 stress-tests.
Profile sharpness.Figure 7a measures how concentrated each dataset’s concept profiles
are in atom coordinates: each curve is the median over a dataset’s 12locked validation
units of the sorted, max-normalized mean profile-weight vector, plotted against atom-rank
fraction. ArXiv uses 64atoms for every encoder; the other datasets use 768/1,536 /2,048
for MiniLM/bge+e5/Qwen3. Sharper drop-offs mean that profile mass concentrates on
a few atoms, which is why one per-concept residual row appended at deployment can
23

C2U in RAG
Table 9 Hit@5 Drop for every dataset ×encoder cell(locked test stage; per-cell mean over seeds
1–3). Higher values indicate stronger target suppression. Bold and underline mark the best and second-
best values inside the exact profile-only access class of each row; no cross-access-class dominance is
claimed.
Dataset Encoder C2U Raw INLP LEACERAG-Emb-
CtrlCURE PISCESDARE-
GRLRandomDelta-
EditFilter-and-
Refill‡Rank-
Router§
Amazonminilm 0.0633 0.0000 0.0733 0.1367 0.4867 0.0000 -0.0100 -0.0167 -0.0100 0.0000 0.0033 0.0900
bge 0.0200 0.0000 0.0100 0.6300 0.9800 0.0000 0.0000 0.4700 0.0000 0.0000 0.0233 0.0200
e5 0.1400 0.0000 0.1567 0.4867 0.4767 0.0000 0.0000 0.3333 0.0333 0.0000 0.0000 0.1400
qwen3 0.0300 0.0000 0.2467 0.2100 0.8100 0.0000 -0.0033 0.1133 0.0233 0.0000 0.0100 0.0300
ArXivminilm 0.0500 0.0000 0.0467 0.0033 0.9600 0.0000 0.0000 0.2133 0.0200 0.0000 0.5833 0.0500
bge 0.0800 0.0000 0.0900 0.0067 0.5000 0.0000 0.0000 0.3167 0.0100 0.0000 0.2567 0.0800
e5 0.0533 0.0000 0.0733 0.0200 0.4400 0.0000 -0.0033 -0.0300 0.0100 0.0000 0.1800 0.0533
qwen3 0.0433 0.0000 0.0267 0.0100 0.8800 0.0000 -0.0033 0.4067 0.0033 0.0000 0.5133 0.0433
IMDBminilm 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0033 0.0000 0.0000 0.0000
bge 0.1033 0.0000 0.0000 0.0000 0.5767 0.0000 0.0000 0.3333 0.0433 0.0000 0.0000 0.1033
e5 0.0000 0.0000 0.0000 0.0733 0.3933 0.0000 0.0733 0.6667 0.0267 0.0000 0.0000 0.0000
qwen3 0.0000 0.0000 0.0000 0.0000 0.2600 0.0000 0.0000 0.3633 0.0233 0.0000 0.0000 0.0000
MNLIminilm 0.0267 0.0000 0.0667 0.0267 0.7300 0.0000 0.0033 0.5333 0.0100 0.0000 0.1833 0.0333
bge 0.0267 0.0000 0.0400 0.0300 0.3133 0.0000 0.0000 0.3000 0.0033 0.0000 0.0800 0.0300
e5 0.0667 0.0000 -0.0067 0.0033 0.2767 0.0000 -0.0133 0.2433 0.0433 0.0000 0.1400 0.0733
qwen3 0.0167 0.0000 0.0600 -0.0100 0.3833 0.0000 0.0000 0.3400 0.0067 0.0000 0.0167 0.0167
TOFUminilm 0.0433 0.0000 0.0467 0.0200 0.1967 0.0000 0.0000 0.3933 -0.0067 0.0000 0.0567 0.0500
bge 0.0233 0.0000 0.0667 0.0600 0.2733 0.0000 -0.0033 0.1833 0.0133 0.0000 0.0267 0.0467
e5 0.0500 0.0000 0.0500 0.1000 0.2933 0.0000 0.0100 -0.1133 -0.0133 0.0000 0.0433 0.0600
qwen3 0.0133 0.0000 0.0933 0.0167 0.3933 0.0000 0.0100 -0.0100 -0.0300 0.0000 0.0333 0.0267
Notes:Each cell is the mean over the three locked seeds (1–3). Encoders: minilm = all-MiniLM-L6-v2,
bge = bge-base-en-v1.5, e5 = e5-base-v2, qwen3 = Qwen3-Embedding-0.6B (pinned revisions). Raw is
the identity (no-edit) reference. Marking scope per row is the exact profile-only access class (C2U,
Raw, PISCES, Random, Delta-Edit); INLP and LEACE (400 labeled calibration examples per unit),
CURE and RAG-Emb-Ctrl (400 labeled calibration examples per unit; posthoc adapted-baseline
extensions under the same locked protocol), DARE-GRL (training-based),‡(candidate/document
access), and§(target-membership label-assisted) are shown unmarked. Utility-clean requires CN
≤10−9and TE ≤10−9, and no cross-access-class dominance is claimed. ArXiv uses a relaxed global
control; IMDB carries the duplicate-template diagnostic label. Column tints encode the locked access
class (same colors as Figure 5); the C2U column is highlighted.
represent the profile direction in bank coordinates (Section 3). Concentrated profiles do not
by themselves imply sparse deployed edits.
Projection separability.Figure 7b plots kernel densities of each query’s cosine to the
target centroid of the unchanged base index for three populations: target at base, target after
steering, and control, recomputed from the locked per-unit tensors (the same recomputation
reproduces the locked target rows to 2.8×10−8). A steered target distribution that moves
toward the control distribution reads as suppressed concept expression. Cosine to a single
centroid is one direction only, so the panel corroborates the hit-based outcomes rather than
certifying suppression by itself.
G Multi-Concept Composition Ladder
Table 13 retains the four-arm history. Only the expanded ArXiv K=4 arm is a pass; its
K=2 design and the earlier arms are validation-only or failures and cannot be presented
as passes. Each concept has a separate raw-space G1/G0 gate and coefficient vector. If sc
are gates, start with tc=0.3s cand set Bq=min(0.3 max csc, 0.2) . For positive total request
allocate ˜tc=t cmin(1,B q/∑ℓtℓ); if all gates vanish all allocations are zero. Per-concept
angular feasibility and sine-ratio scaling precede additive delta composition and the shared
trust exit. The comparator uses a normalized average of unit O2 directions with a single
max gate. The endpoint takes minima of per-concept H5 within each method and unit,
then differences them and averages units, as in equation 6; it does not minimize paired
conceptwise differences or use a min gate.
24

C2U in RAG
Table 10 Target Cosine Drop for every dataset ×encoder cell(locked test stage; per-cell mean
over seeds 1–3). Higher values indicate stronger reduction of the target-centroid cosine. Marking as in
Table 9.
Dataset Encoder C2U Raw INLP LEACERAG-Emb-
CtrlCURE PISCESDARE-
GRLRandomDelta-
EditFilter-and-
Refill‡Rank-
Router§
Amazonminilm 0.0279 0.0000 0.0604 0.0717 0.0861 0.0000 0.0003 0.3298 0.0818 -0.0000 0.0000 0.0245
bge 0.0555 0.0000 0.0384 0.0525 0.1702 0.0000 0.0030 0.6610 0.2786 0.0025 0.0000 0.0555
e5 0.0505 0.0000 0.0174 0.0057 0.0099 0.0000 -0.0023 0.8229 0.3486 0.0009 0.0000 0.0481
qwen3 0.0921 0.0000 0.0415 0.0246 0.0832 0.0000 0.0047 0.3683 0.1812 0.0025 0.0000 0.0884
ArXivminilm 0.1317 0.0000 0.2284 0.1124 0.4639 0.0000 0.0300 0.3187 0.0994 0.0036 0.0000 0.1286
bge 0.0710 0.0000 0.0922 0.0340 0.0862 0.0000 0.0081 0.6618 0.2913 0.0031 0.0000 0.0681
e5 0.0498 0.0000 0.0639 0.0124 0.0330 0.0000 0.0012 0.8220 0.3500 0.0007 0.0000 0.0490
qwen3 0.0910 0.0000 0.1705 0.0920 0.3038 0.0000 0.0216 0.4959 0.2331 0.0025 0.0000 0.0896
IMDBminilm 0.0808 0.0000 0.1724 0.0518 0.0435 0.0000 0.0036 0.6319 0.1797 0.0012 0.0000 0.0808
bge 0.0795 0.0000 0.0438 -0.0065 0.1467 0.0000 0.0012 0.6263 0.2738 0.0024 0.0000 0.0726
e5 0.0620 0.0000 0.0240 0.0029 0.0154 0.0000 0.0021 0.8481 0.3581 0.0013 0.0000 0.0620
qwen3 0.0841 0.0000 0.0817 0.0104 0.0725 0.0000 -0.0000 0.5046 0.2458 0.0015 0.0000 0.0841
MNLIminilm 0.0850 0.0000 0.1581 0.0730 0.4172 0.0000 0.0204 0.2980 0.0869 0.0012 0.0000 0.0810
bge 0.0299 0.0000 0.0899 0.0230 0.0890 0.0000 0.0068 0.6297 0.2628 0.0007 0.0000 0.0284
e5 0.0247 0.0000 0.0317 0.0046 0.0364 0.0000 0.0004 0.8094 0.3459 0.0003 0.0000 0.0226
qwen3 0.0726 0.0000 0.1444 0.0200 0.1295 0.0000 0.0118 0.4413 0.2083 0.0007 0.0000 0.0703
TOFUminilm 0.0677 0.0000 0.0291 0.0071 0.0325 0.0000 0.0013 0.4427 0.1372 0.0003 0.0000 0.0529
bge 0.0441 0.0000 0.0373 -0.0001 0.0181 0.0000 0.0002 0.6042 0.2539 0.0002 0.0000 0.0372
e5 0.0366 0.0000 0.0679 0.0024 0.0093 0.0000 0.0006 0.7699 0.3352 0.0000 0.0000 0.0332
qwen3 0.0883 0.0000 0.0380 0.0099 0.0659 0.0000 0.0040 0.4019 0.1909 0.0007 0.0000 0.0757
Notes:Each cell is the mean over the three locked seeds (1–3). Encoders: minilm = all-MiniLM-L6-v2,
bge = bge-base-en-v1.5, e5 = e5-base-v2, qwen3 = Qwen3-Embedding-0.6B (pinned revisions). Raw is
the identity (no-edit) reference. Marking scope per row is the exact profile-only access class (C2U,
Raw, PISCES, Random, Delta-Edit); INLP and LEACE (400 labeled calibration examples per unit),
CURE and RAG-Emb-Ctrl (400 labeled calibration examples per unit; posthoc adapted-baseline
extensions under the same locked protocol), DARE-GRL (training-based),‡(candidate/document
access), and§(target-membership label-assisted) are shown unmarked. Utility-clean requires CN
≤10−9and TE ≤10−9, and no cross-access-class dominance is claimed. ArXiv uses a relaxed global
control; IMDB carries the duplicate-template diagnostic label. Column tints encode the locked access
class (same colors as Figure 5); the C2U column is highlighted.
Original pools (v1): validation-only pass.Pooled over MNLI and ArXiv K=4 , the paired
acceptance claim passes selection ( +0.0300 , CI95 [+0.0167,+0.0450] ) and independent
confirmation ( +0.0250 , CI95 [+0.0050,+0.0467] ). All eight dataset–concept compose H5
cells are positive. Other datasets are capacity-blocked, not zeros; this pass authorizes no test
by itself.
New pools (v2): failed acceptance with a registered defect.Amazon L1 K=4 , ArXiv
L2K=8 , and MNLI K=3 have a positive selection paired endpoint ( +0.0244 ), but
selection acceptance fails: Amazon Books and Electronics have compose H5 −0.0133 and
−0.0267 . Confirmation does not replicate ( −0.0044 ). The builder restarts allocation cursors
across role kinds, contaminating selection with profile rows and creating corpus/query
overlap. Confirmation is disjoint from profiles, and its independent failure supports the
negative verdict; test rows are disjoint from other slices and never accessed. The overlap
remains an additional blocker, not an excused positive result. IMDB/TOFU control capacity
is insufficient. The diagnostic ArXiv K=2/4/8 sweep is also negative and carries the
defect label.
Cursor-disjoint rebuild (v3): failed selection.The repaired builder uses advancing cross-
role cursors and true overlap checks. Pre-outcome capacity/mechanism rules restrict it to
ArXiv: MNLI/IMDB/TOFU lack controls, while Amazon is excluded after ineffective L1
retrieval suppression and lacks suitable L2 controls. Both pre-registered K=2 and K=4
designs fail selection. Confirmation never runs and test refusal is exercised; no test rows are
touched.
25

C2U in RAG
Table 11 Control/Neighbor ∆for every dataset ×encoder cell(locked test stage; per-cell mean
over seeds 1–3). Lower values indicate less non-target degradation; the utility-clean gate requires CN
≤10−9. Marking as in Table 9.
Dataset Encoder C2U Raw INLP LEACERAG-Emb-
CtrlCURE PISCESDARE-
GRLRandomDelta-
EditFilter-and-
Refill‡Rank-
Router§
Amazonminilm -0.0011 0.0000 0.0000 -0.0011 -0.0011 0.0000 0.0000 -0.0011 0.0000 0.0000 0.0011 -0.0011
bge -0.0167 0.0000 -0.0144 -0.0178 -0.0178 0.0000 -0.0067 0.0056 -0.0056 0.0000 -0.0167 -0.0167
e5 -0.0033 0.0000 -0.0011 -0.0022 -0.0033 0.0000 -0.0011 -0.0033 0.0000 0.0000 -0.0033 -0.0033
qwen3 -0.0044 0.0000 -0.0044 -0.0033 -0.0056 0.0000 -0.0022 -0.0056 -0.0011 0.0000 -0.0033 -0.0044
ArXivminilm -0.0122 0.0000 0.0178 0.0233 0.0978 0.0000 -0.0044 0.6544 0.0044 0.0000 -0.0167 -0.0089
bge -0.0078 0.0000 0.0422 0.0256 0.0100 0.0000 -0.0011 0.7244 0.0156 0.0000 -0.0189 -0.0111
e5 -0.0144 0.0000 0.1189 0.0178 0.0478 0.0000 -0.0022 0.6211 0.1467 0.0000 -0.0200 -0.0067
qwen3 -0.0156 0.0000 0.0211 0.0211 0.0667 0.0000 -0.0022 0.7433 0.0300 0.0011 -0.0233 -0.0167
IMDBminilm 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3333 0.0578 0.0000 0.0000 0.0000
bge 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3333 0.0022 0.0000 0.0000 0.0000
e5 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0211 0.0000 0.0000 0.0000
qwen3 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.3333 0.0000 0.0000 0.0000 0.0000
MNLIminilm -0.0244 0.0000 -0.0256 -0.0067 0.0389 0.0000 -0.0078 0.2322 0.0167 0.0011 -0.0422 -0.0256
bge -0.0100 0.0000 -0.0211 -0.0033 0.0078 0.0000 -0.0022 0.2578 0.0256 0.0000 -0.0211 -0.0078
e5 -0.0156 0.0000 0.0356 0.0000 -0.0100 0.0000 0.0000 0.3278 0.0356 0.0000 -0.0178 -0.0200
qwen3 0.0067 0.0000 0.0200 0.0122 0.0167 0.0000 0.0067 0.2911 0.0289 0.0000 0.0033 0.0044
TOFUminilm -0.0067 0.0000 0.0278 -0.0000 -0.0133 0.0000 0.0011 0.1833 -0.0033 0.0000 -0.0078 -0.0067
bge -0.0011 0.0000 0.0467 0.0000 -0.0067 0.0000 0.0011 0.2400 0.0122 0.0000 -0.0067 -0.0011
e5 -0.0011 0.0000 0.1122 -0.0067 -0.0156 0.0000 -0.0011 0.2567 0.0233 0.0000 -0.0089 -0.0011
qwen3 -0.0089 0.0000 0.0722 0.0033 -0.0122 0.0000 0.0022 0.3033 0.0056 0.0000 -0.0244 -0.0078
Notes:Each cell is the mean over the three locked seeds (1–3). Encoders: minilm = all-MiniLM-L6-v2,
bge = bge-base-en-v1.5, e5 = e5-base-v2, qwen3 = Qwen3-Embedding-0.6B (pinned revisions). Raw is
the identity (no-edit) reference. Marking scope per row is the exact profile-only access class (C2U,
Raw, PISCES, Random, Delta-Edit); INLP and LEACE (400 labeled calibration examples per unit),
CURE and RAG-Emb-Ctrl (400 labeled calibration examples per unit; posthoc adapted-baseline
extensions under the same locked protocol), DARE-GRL (training-based),‡(candidate/document
access), and§(target-membership label-assisted) are shown unmarked. Utility-clean requires CN
≤10−9and TE ≤10−9, and no cross-access-class dominance is claimed. ArXiv uses a relaxed global
control; IMDB carries the duplicate-template diagnostic label. Column tints encode the locked access
class (same colors as Figure 5); the C2U column is highlighted.
Capacity expansion (v4): ArXiv K=4 passes.Earlier locked validation differences inform
a budget design, not a causal diagnosis of quantization noise. The T1/T2/T3 ladder is
frozen before v4 outcomes; all tiers fail the predeclared 0.8power criterion. The adjudicated
fallback selects the largest feasible T3: 200selection, 200confirmation, and 400test target
queries per concept, with 1,200 controls per split. Profiles remain 20documents per concept;
the index uses 25target documents per concept and 300controls. Cursor-disjoint allocation
requires 845rows per concept and 4,800 control rows overall. Method configuration and
the atom bank are unchanged, but evaluation pools and budgets differ.
Concepts are frozen by free-row ranking after prior consumption: Software Engineer-
ing, Artificial Intelligence, Neurons and Cognition, Number Theory; K=2 is the prefix.
K=2 fails selection ( −0.0008 , CI95 [−0.0075,+0.0058] , only one of three seeds positive).
K=4 passes selection ( +0.0425 , CI95 [+0.0279,+0.0575] ) and independent confirma-
tion ( +0.0517 , CI95 [+0.0375,+0.0663] ). Its hard-gated test endpoint is +0.04125 , CI95
[+0.0325,+0.050625] from 4,000 bootstraps of twelve paired unit differences; all four con-
cept cells, three seeds, and four encoders are positive. The retained table rounds these
to+0.0412 and[+0.0325,+0.0506] . These are previously unused multi-concept test rows,
first evaluated without tuning after the passing lock; refusal was exercised before that
lock existed. This untouched-test scope is separate from the posthoc single-concept/attack
cohorts. The document-membership oracle Filter-and-Refill-multi is a non-gating upper
reference (+0.7181), not an access-matched competitor.
The ladder supports a scoped improvement over the averaged-direction comparator on
ArXiv K=4 . It does not prove that larger budgets remove interference, that noise caused
earlier failures, or that atom representation alone causes the gain.
26

C2U in RAG
Table 12 Target Exposure ∆for every dataset ×encoder cell(locked test stage; per-cell mean over
seeds 1–3). Lower values indicate less false-positive retrieval of target documents by control queries;
the utility-clean gate requires TE≤10−9. Marking as in Table 9.
Dataset Encoder C2U Raw INLP LEACERAG-Emb-
CtrlCURE PISCESDARE-
GRLRandomDelta-
EditFilter-and-
Refill‡Rank-
Router§
Amazonminilm -0.0700 0.0000 0.0811 0.0178 -0.3533 0.0000 0.0067 0.3678 0.0189 -0.0011 -0.0100 -0.1011
bge -0.1356 0.0000 0.2922 -0.0611 -0.2644 0.0000 0.0111 0.3956 0.0200 0.0000 -0.0933 -0.1411
e5 -0.1800 0.0000 0.2733 -0.0267 -0.2556 0.0000 0.0644 0.3778 0.0600 0.0000 -0.0256 -0.1833
qwen3 -0.1122 0.0000 0.1078 0.0367 -0.4678 0.0000 0.0256 0.4000 0.0211 -0.0022 -0.0367 -0.1278
ArXivminilm -0.0389 0.0000 0.0567 0.1922 -0.1756 0.0000 -0.0033 0.5067 0.0211 0.0000 -0.1467 -0.0389
bge -0.0878 0.0000 0.0500 0.1789 -0.1789 0.0000 0.0078 0.4700 0.0333 0.0000 -0.1478 -0.0878
e5 -0.1089 0.0000 0.1900 0.1422 -0.2578 0.0000 0.0400 0.7056 0.1489 0.0000 -0.2156 -0.1089
qwen3 -0.0700 0.0000 0.0589 0.1344 -0.2289 0.0000 0.0089 0.6622 0.0489 0.0000 -0.2122 -0.0700
IMDBminilm 0.0000 0.0000 0.6667 0.3333 0.0000 0.0000 0.0000 1.0000 0.2822 0.0000 0.0000 0.0000
bge 0.0000 0.0000 0.0000 0.0000 0.3333 0.0000 0.0000 0.6667 0.0944 0.0000 0.0000 0.0000
e5 0.0000 0.0000 0.0000 0.0000 0.3333 0.0000 0.0000 0.3333 0.3167 0.0000 0.0000 0.0000
qwen3 0.0000 0.0000 0.0000 0.0000 0.3333 0.0000 0.0000 0.3333 0.2311 0.0000 0.0000 0.0000
MNLIminilm -0.0556 0.0000 0.0378 0.0433 -0.3789 0.0000 -0.0078 -0.1211 -0.0044 0.0000 -0.1411 -0.0578
bge -0.0611 0.0000 0.0111 0.0433 -0.3833 0.0000 -0.0011 0.0722 0.0167 0.0000 -0.1544 -0.0644
e5 -0.0911 0.0000 0.1400 0.1333 -0.3011 0.0000 0.0211 0.2500 0.0756 0.0000 -0.1456 -0.0978
qwen3 -0.0333 0.0000 0.0444 0.0978 -0.3067 0.0000 0.0044 0.3789 0.0589 0.0011 -0.0322 -0.0344
TOFUminilm -0.0200 0.0000 0.0633 0.0211 -0.1844 0.0000 0.0022 -0.2378 0.0100 0.0000 -0.0611 -0.0278
bge -0.0244 0.0000 0.1111 0.0167 -0.2433 0.0000 0.0144 0.0811 0.0100 0.0011 -0.0433 -0.0267
e5 -0.0167 0.0000 0.1122 -0.0233 -0.2133 0.0000 0.0056 0.3356 0.0222 0.0000 -0.0811 -0.0244
qwen3 -0.0244 0.0000 0.0733 0.0089 -0.3100 0.0000 -0.0011 0.2900 -0.0056 0.0000 -0.1067 -0.0278
Notes:Each cell is the mean over the three locked seeds (1–3). Encoders: minilm = all-MiniLM-L6-v2,
bge = bge-base-en-v1.5, e5 = e5-base-v2, qwen3 = Qwen3-Embedding-0.6B (pinned revisions). Raw is
the identity (no-edit) reference. Marking scope per row is the exact profile-only access class (C2U,
Raw, PISCES, Random, Delta-Edit); INLP and LEACE (400 labeled calibration examples per unit),
CURE and RAG-Emb-Ctrl (400 labeled calibration examples per unit; posthoc adapted-baseline
extensions under the same locked protocol), DARE-GRL (training-based),‡(candidate/document
access), and§(target-membership label-assisted) are shown unmarked. Utility-clean requires CN
≤10−9and TE ≤10−9, and no cross-access-class dominance is claimed. ArXiv uses a relaxed global
control; IMDB carries the duplicate-template diagnostic label. Column tints encode the locked access
class (same colors as Figure 5); the C2U column is highlighted.
Composition view.Figure 8 is a descriptive view of the locked ArXiv K=4 outcomes,
not new evidence. Panels (a,b) place raw and compose-edited queries in one deterministic
PCA basis on the pre-registered bge/seed-1 illustration (variance share 0.157 ): control
centroids shift by 0.022 , target centroids by 0.058 –0.118 . Panel (c) reports residual-atom
cosines—SE–NT 0.022 , NC–NT 0.015 , SE–AI 0.560 —not the Gram matrix of full O2 edit
directions. Panel (d) shows paired per-concept H5 differences, with Artificial Intelligence
smallest ( +0.0142 ): every concept contributes positively, which is what the min-concept
endpoint of equation 6 rewards.K=2 failed selection and never entered test.
H Query-Transform Evaluation and Full Results
Table 15 retains every dataset–condition value and the Mean columns; Table 14 summarizes
the same cells as counts and signed worst values. The original arm covers 5datasets ×4
encoders ×3seeds ×3conditions ×7methods, including Raw: 504,000 per-query rows.
A separate additive arm supplies the two recent adaptations; the original count is not the
combined-arm count. Both are posthoc extensions of frozen single-concept units, with no
test tuning and no untouched-test claim. The main table omits Raw’s construction-zero
row, not its role as the paired reference. Candidate/document-access Filter-and-Refill,
membership-assisted Rank-Router, and Delta-Edit are predeclared attack exclusions.
Transforms are bitwise-equal reimplementations of the original pure functions; the recent-
baseline arm replays its standard-condition runs unchanged, and CURE’s all-zero cells are
its locked zero-strength identity. Aggregates use 4,000 paired bootstrap samples. CER is
averaged over control queries with the 512-token/ 50-document packing rule; the legacy
attack “Hit” heading is clarified as target density without changing values. Main Table 3
27

C2U in RAG
0.005
 0.0000.00.20.40.6Amazon
0.0 0.50.00.20.40.6ArXiv
0.0 0.20.00.10.20.3IMDB
0.00 0.250.00.20.4MNLI
0.0 0.20.00.10.20.3TOFU
0.0 0.50.00.20.40.6
0.0 0.50.00.20.40.6
0.0 0.50.00.10.20.3
0.25
 0.000.00.20.4
0.25
 0.000.00.10.20.3Hit@5 Drop (higher = better)Ctrl./Neigh. 
Target Exp. 
C2U
RawINLP
LEACERAG-Emb-Ctrl
CUREPISCES
DARE-GRLRandom
Delta-EditF&R
Rank-Router
profile-only
profile-only + labeled calibration fitprofile-only (adapted baselines)
candidate/document accessmembership label-assisted
Figure 5 Full suppression–utility Pareto view.All twelve displayed methods on every dataset,
pooled over four encoders and three seeds, with access-class colors. CURE coincides with Raw because
its utility gate locks it to identity. Stronger suppression and its costs remain visible.
summarizes equal-weight cell means, and Table 14 the unrounded suppression/divergence
signs and signed worst values; neither is significance or adaptive-attack certification.
Reading the full matrix.Table 15 expands every summary cell into per-condition values.
Each dataset block lists the eight evaluated methods in the main table’s order with locked
access-class tints and the C2U row highlighted; each condition group pairs target density
∆on target queries with CER ∆and own-label Hit@10 loss on control queries, and the
Mean group reproduces the main table’s per-dataset cells. Marks compare only the exact
profile-only group (C2U, PISCES, Random).
Condition-level patterns.Read per condition, the counts in Table 14 are not artifacts of
Mean pooling. C2U lowers target density in all 15cells and never raises control exposure,
with attack utility between −5.2 and+2.6 pp; its remaining variation is condition mix, not
sign. INLP’s Amazon pair—density down 15.8 pp, exposure up 17.6 pp on Mean—repeats
under every condition, the displacement pattern behind the 13/15 divergent cells it shares
with LEACE. RAG-Emb-Ctrl suppresses most strongly on ArXiv ( −72.4 density, −21.9
exposure) yet diverges in the three IMDB cells, worst +16.7 pp under injection, and costs
utility on ArXiv ( +5.3 pp) and MNLI ( +5.7 ); the duplicate-template limitation applies.
CURE’s all-zero matrix is its locked zero-strength identity, so its 0/15 divergence count is
trivial. PISCES, Random, and DARE-GRL diverge in 11/15 cells.
28

C2U in RAG
0.25 0.50 0.7501020densityAmazon
0.0 0.5ArXiv
0.50 0.75IMDB
0.0 0.5MNLI
0.25 0.50 0.75TOFU
base query steered query
(a) Target-cosine distributions before and after steering(locked test rows). The steered distribution
shifts left of base in every dataset; Figure 7b adds the control distribution.
05001000target
queries1200 active
sat 0.93Amazon
1178 active
sat 0.82ArXiv
1200 active
sat 1.00IMDB
1135 active
sat 0.72MNLI
968 active
sat 0.50TOFU
0.0 0.1 0.202000control
queries3570 active
sat 0.77
0.0 0.1 0.21929 active
sat 0.24
0.0 0.1 0.2
applied steering angle (rad)3600 active
sat 1.00
0.0 0.1 0.22012 active
sat 0.32
0.0 0.1 0.21765 active
sat 0.30
trust cap 0.2 rad cos floor arccos(0.98)
(b) Applied-angle distribution under the delta-exit trust region(locked test rows). Active edits
concentrate at the default 0.2-rad cap, slightly tighter than arccos(0.98) ; among edited queries, 80.7%
of the 5,681 target edits and 61.9% of the 12,876 control edits sit at the cap, whereas Figure 3 uses all
queries of each group as denominator. Gate-off rows are zero. Rows denote query role and columns
dataset. These are descriptive locked-outcome distributions.
29

C2U in RAG
103
102
101
100
atom rank fraction (sorted by mean profile weight)0.20.40.60.81.0normalized mean profile weightAmazon
ArXiv
IMDB
MNLI
TOFU
(a) Concept-profile sharpness across datasets(locked validation assets). Each curve is the me-
dian over a dataset’s 12 locked validation units of the sorted, max-normalized mean profile-weight
vector, against atom-rank fraction. ArXiv uses 64atoms for every encoder; the other datasets
use 768/1,536 /2,048 for MiniLM/bge+e5/Qwen3. Sharper drop-offs indicate concentrated pro-
file weights, not sparse deployed edits. One per-concept row is appended at deployment (Section 3).
0.0 0.5024densityAmazon
0 1ArXiv
0.50 0.75IMDB
0.0 0.5MNLI
0.25 0.50 0.75TOFU
control (recomputed) target base target steered
(b) Projection-score separability before and after steering(locked test outcomes). Per-dataset
kernel densities of each query’s cosine to the target centroid of the unchanged base index: target
at base, target after steering, and control (recomputed from the locked per-unit tensors; the same
recomputation reproduces the locked target rows to 2.8×10−8). A steered target distribution shifting
toward the control distribution indicates suppression of concept expression.
30

C2U in RAG
Table 13 Multi-concept composition ladder.Primary endpoint: paired per-unit difference of min-
concept Hit@5 Drop, GR-C2U-ACR-compose minus the internal ablation GR-C2U-Fixed-m3-multi
(positive favors compose). v1 (original pools) passes validation only; v2 (new pools) fails selection
and carries a registered split-overlap construction defect; v3 (cursor-disjoint) is scope-limited to ArXiv
and fails selection on both designs; v1–v3 touch no test rows and their test stages stay hard-closed. v4
(cursor-disjoint tier-T3 capacity expansion on fresh rows) fails selection at K=2 but passes selection,
confirmation, and the hard-gated test stage at K=4 . F&R-multi is a document-membership oracle
upper reference, non-gating.
Arm Pools DesignSelection∆
[CI95]Confirmation∆
[CI95]Test∆
[CI95]Verdict
v1 originalMNLIK=4;
ArXivK=4+0.0300
[+0.0167, +0.0450]+0.0250
[+0.0050, +0.0467]—PASSv
v2 newdAmazonK=4;
ArXivK=8;
MNLIK=3+0.0244
[+0.0122, +0.0389]-0.0044c— FAILd
v3 cursor-disjoint ArXivK=2e+0.0100g—f— FAILd
v3 cursor-disjoint ArXivK=4e+0.0067g—f— FAILd
v4 cursor-disjoint ArXivK=2e -0.0008
[-0.0075, +0.0058]—f— FAILd
v4 cursor-disjoint ArXivK=4e +0.0425
[+0.0279, +0.0575]+0.0517
[+0.0375, +0.0663]+0.0412
[+0.0325, +0.0506]PASSt
Notes:vvalidation-only acceptance; not a test claim.ddiagnostic-only; the test runner refuses unless
the method lock is PASS.cv2 confirmation CI95 straddles zero per dataset: Amazon [-0.0233, +0.0033];
ArXiv [-0.0100, +0.0100]; MNLI [-0.0100, +0.0000]; v2 selection per-concept compose Hit@5 Drop on
Amazon L1 is negative for Books (-0.0133) and Electronics (-0.0267), failing the per-concept gate.
Granularity: the v2 Amazon cells use L1; all ArXiv cells (v2, v3, v4) use L2.eConcepts (pre-registered
free-row ranking; v4 re-ranked after v3 consumption): v3K=2: {Artificial Intelligence, Software
Engineering}; K=4 : {Artificial Intelligence, Genomics, Neurons and Cognition, Software Engineering};
v4K=2: {Software Engineering, Artificial Intelligence};K=4: {Software Engineering, Artificial
Intelligence, Neurons and Cognition, Number Theory}.fconfirmation never ran: no design passed
selection.gv3 gates failed (CI95 lower bound≤0;K=2: seeds 2/3, encoders 3/4;K=4: seeds 2/3,
encoders 1/4); v4K=2 fails the CI-lower-bound, all-seeds, and encoder gates (seeds 1/3, encoders
2/4 positive).ttest-stage acceptance on freshly allocated, never-before-used test rows (the first use of
these test rows in any experiment, evaluated exactly once with no test-time tuning); the refusal path
was exercised and evidenced before the PASS method lock existed. v4 tier T3 (pre-registered before
any v4 outcome): 200/200 selection/confirmation and 400 test target queries per concept, 1,200
control rows per split; v4K=4 test gates: seeds 3/3, encoders 4/4 positive. Filter-and-Refill-multi
(document-membership oracle, non-gating upper reference) min-concept Hit@5 Drop on the same
units: v2 confirmation Amazon +0.6967, ArXiv +0.9333, MNLI +0.8867; v3 selectionK=2 +0.9433,
K=4 +0.8967; v4 selectionK=2 +0.7683,K=4 +0.7662; v4 testK=4 +0.7181. v3/v4 scope:
MNLI/IMDB/TOFU excluded by capacity (best-case free control rows MNLI 395, IMDB 565, TOFU
428 vs 1,200 required), Amazon by mechanism (v2-measured L1-granularity retrieval-level
ineffectiveness; L2 offers no non-target control).
Table 14 Cross-condition behavior over the fifteen dataset–condition cells.Suppressing: density
∆<0 ; divergent: suppressing with CER ∆>0 ; worst values are signed maxima over the fifteen cells
(pp). Counts use unrounded values; the per-condition cells are in Table 15.
MethodSuppressing
/15Divergent
/15Worst
CER∆Worst
utility loss
C2U 15/15 0/15 0.0 +2.6
PISCES 14/15 11/15 +4.9 +0.1
Random 12/15 11/15 +22.3 +5.7
INLP 15/15 13/15 +25.0 +5.4
LEACE 15/15 13/15 +12.5 +3.4
RAG-Emb-Ctrl 15/15 3/15 +16.7 +14.0
CURE 0/15 0/15 0.0 0.0
DARE-GRL 11/15 11/15 +50.9 +66.3
31

C2U in RAG
0.4
0.2
0.0
0.2
0.4 PC10.2
0.00.20.4
PC20.2
0.00.20.4
PC3var share 0.157(a) raw base queries
0.4
0.2
0.0
0.2PC10.2
0.0
0.2
PC20.4
0.2
0.00.2
PC3var share 0.157(b) compose edited queries
Software Engineering Artificial IntelligenceNeurons and CognitionNumber TheorySoftware Engineering
Artificial Intelligence
Neurons and Cognition
Number Theory1.000 0.560 0.170 0.022
0.560 1.000 0.352 0.231
0.170 0.352 1.000 0.015
0.022 0.231 0.015 1.000(c) residual-atom pairwise cosine
1.00
0.75
0.50
0.25
0.000.250.500.751.00
Software Engineering Artificial IntelligenceNeurons and CognitionNumber Theory0.000.050.100.150.20paired Hit@5 Drop diff (compose - Fixed-m3-multi)
(d) per-concept paired differences
Software Engineering Artificial Intelligence Neurons and Cognition Number Theory control
Figure 8 Descriptive composition view of locked ArXiv K=4 outcomes.(a,b) Raw and edited
queries in one deterministic PCA basis on the pre-registered bge/seed-1 illustration (variance share
0.157 ); centroid shifts are 0.022 for controls and 0.058 –0.118 for targets. (c) Residual-atom cosines,
not the Gram matrix of full O2 edit directions: SE–NT 0.022 , NC–NT 0.015 , SE–AI 0.560 . (d) Paired
per-concept H5 differences; AI is smallest ( +0.0142 ). Selection/confirmation/test endpoints are
+0.0425/0.0517/0.0412 (test CI95 [+0.0325,+0.0506] ); oracle refill is +0.7181 .K=2 failed selection
and never entered test. This preserves the first-use, no-test-tuning scope of the lockedK=4 test.
32

C2U in RAG
Table 15 Retrieval under standard and transformed queries.Per-query paired percentage-point
changes relative to the Raw reference under thesamecondition;Target density ∆,CER∆on control
queries (context budget B=512 , at most 50documents), andcontrol own-label Hit@10 lossare all
lower-is-better. Bold/underline are computed only inside the exact profile-only group (C2U, PISCES,
Random), excluding the Raw reference; no cross-access-group dominance is claimed. The Mean
columns pool the three query conditions per dataset.
Standard
(no attack) Injection Zou-noise Mean
MethodTarget
density∆Control
CER∆Hit@10
loss↓Target
density∆Control
CER∆Hit@10
loss↓Target
density∆Control
CER∆Hit@10
loss↓Target
density∆Control
CER∆Hit@10
loss↓
Amazon
C2U -18.0 -12.2 -0.1 -14.3 -16.7 0.0 -16.7 -12.4 0.0 -16.4 -13.7 0.0
PISCES -3.5 +3.4 -0.1 +0.2 +4.9 0.0 -0.5 +4.6 0.0 -1.2 +4.3 0.0
Random -5.7 +2.2 0.0 +2.5 +11.2 0.0 +2.1 +5.7 0.0 -0.4 +6.4 0.0
INLP -31.2 +20.1 -0.1 -5.4 +16.9 0.0 -10.7 +15.8 0.0 -15.8 +17.6 0.0
LEACE -45.2 0.0 -0.1 -6.2 +3.2 0.0 -14.3 +4.1 0.0 -21.9 +2.4 0.0
RAG-Emb-Ctrl -59.7 -34.6 -0.1 -19.3 -28.6 0.0 -25.1 -30.8 0.0 -34.7 -31.4 0.0
CURE 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
DARE-GRL -38.4 +36.3 -0.1 +9.5 +50.7 0.0 +1.3 +50.9 0.0 -9.2 +46.0 0.0
ArXiv
C2U -14.6 -6.1 -0.8 -20.1 -21.9 -5.2 -18.4 -7.1 -0.8 -17.7 -11.7 -2.2
PISCES -2.1 +1.1 -0.2 -2.0 -0.4 -0.2 -2.0 +1.0 -0.2 -2.0 +0.6 -0.2
Random -4.8 +4.8 +3.6 -6.0 +3.9 +5.7 -6.0 +5.1 +4.7 -5.6 +4.6 +4.7
INLP -33.0 +3.7 +4.6 -35.0 -3.3 +2.3 -34.4 +5.6 +5.4 -34.1 +2.0 +4.1
LEACE -14.8 +10.6 +1.6 -11.8 +12.5 +3.4 -15.4 +12.3 +1.7 -14.0 +11.8 +2.2
RAG-Emb-Ctrl -68.2 -14.0 +5.6 -76.3 -34.9 +3.3 -72.7 -16.8 +6.9 -72.4 -21.9 +5.3
CURE 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
DARE-GRL -51.4 +41.5 +66.3 -50.5 +19.6 +60.0 -50.4 +40.2 +64.6 -50.7 +33.8 +63.6
IMDB
C2U -10.9 0.0 0.0 -14.5 0.0 0.0 -7.9 -2.3 -0.1 -11.1 -0.8 0.0
PISCES 0.0 0.0 0.0 -0.4 0.0 0.0 -1.4 +0.7 +0.1 -0.6 +0.2 0.0
Random -8.1 +8.2 +0.1 -1.1 +22.3 +1.7 -7.0 +13.6 +0.3 -5.4 +14.7 +0.7
INLP -18.6 0.0 0.0 -11.8 +25.0 0.0 -19.6 +5.3 -0.2 -16.7 +10.1 -0.1
LEACE -4.7 +8.3 0.0 -7.2 0.0 0.0 -5.0 +8.9 +0.4 -5.6 +5.7 +0.1
RAG-Emb-Ctrl -35.5 +8.3 0.0 -31.0 +16.7 0.0 -36.3 +1.6 -0.2 -34.3 +8.9 -0.1
CURE 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
DARE-GRL -38.3 +25.0 0.0 -15.4 +33.3 0.0 -34.4 +25.3 0.0 -29.4 +27.9 0.0
MNLI
C2U -11.4 -3.4 -0.5 -19.8 -8.8 +2.6 -13.5 -8.1 -1.1 -14.9 -6.8 +0.3
PISCES -2.0 +0.9 -0.2 -2.0 +0.4 -0.2 -2.8 +0.1 -0.3 -2.3 +0.4 -0.2
Random -4.8 +2.9 +1.0 -3.5 -0.3 +1.4 -3.2 +0.9 +1.3 -3.9 +1.2 +1.2
INLP -16.9 +9.0 -0.9 -14.8 +3.0 -1.5 -21.2 +2.3 -0.3 -17.7 +4.8 -0.9
LEACE -8.0 +6.6 -0.4 -7.5 +1.1 +0.1 -9.6 +2.2 -0.7 -8.4 +3.3 -0.3
RAG-Emb-Ctrl -43.1 -45.1 +0.6 -37.8 -51.8 +14.0 -40.7 -45.5 +2.6 -40.6 -47.5 +5.7
CURE 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
DARE-GRL -37.5 +13.1 +10.0 -21.9 +2.3 +0.3 -30.0 +4.3 +10.4 -29.8 +6.6 +6.9
TOFU
C2U -2.9 -1.6 -0.1 -5.7 -6.1 0.0 -4.4 -5.8 -0.3 -4.4 -4.5 -0.1
PISCES -0.9 +1.1 +0.1 -0.8 +1.3 -0.1 -0.3 +1.7 +0.1 -0.6 +1.3 +0.1
Random -0.1 +1.1 +0.9 -0.5 +2.2 +1.0 +0.2 +1.4 +0.8 -0.1 +1.5 +0.9
INLP -7.0 +11.4 +1.8 -5.7 +14.4 +2.6 -4.2 +16.3 +2.1 -5.6 +14.0 +2.2
LEACE -5.7 +1.4 -0.3 -5.5 +3.9 +0.2 -4.2 +4.4 +0.1 -5.1 +3.2 0.0
RAG-Emb-Ctrl -18.3 -27.4 -0.4 -22.8 -34.7 +1.9 -18.9 -29.8 -0.4 -20.0 -30.6 +0.4
CURE 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
DARE-GRL -1.2 +15.3 +13.3 +0.3 +20.2 +13.6 +0.2 +19.7 +14.6 -0.3 +18.4 +13.8
Note:Filter-and-Refill and Rank-Router are excluded on access-class mismatch (candidate/document
and membership-label access versus this arm’s query-only contract). Raw (ref) is exactly zero by
construction (method minus itself); its all-zero reference rows are omitted from the display. This arm
is a posthoc extension of the frozen test units, evaluated exactly once with no test-time tuning; no
untouched-test claim is made. CURE’s all-zero cells are its locked zero-strength utility-gate identity,
verified by execution. Row tints encode the locked access class (same colors as Figure 5); the C2U row
is highlighted.
33

C2U in RAG
I Reproducibility and Source Record
Frozen experimental packages retain per-arm protocols, configuration/input/method locks,
encoder/environment records, per-query outputs, aggregation rules, and independent
checks. Bootstrap units differ: main per-dataset H5 intervals reuse the standard replay’s
paired-row bootstrap, whereas the multi-concept test resamples twelve unit differences.
Recent-baseline provenance replay is explicitly retrospective. Unknown legacy resource
measurements remain unavailable, not estimated.
The ARR manuscript uses the official ACL template and a lightweight single-JSON con-
version/source record, CONVERSION_PROVENANCE.json . That record is distinct from frozen
experimental audit chains; ARR has no ICLR-style closure-manifest claim. The restructur-
ing introduces no experiments or test-time selection. Full matrices, negative probes, and
descriptive views distinguish evidence reuse from new inference. Generated summaries
derive from existing locked aggregates rather than newly accessed raw test queries.
J Theoretical Analysis of the Deployed Edit Operator
We distinguish exact real-arithmetic identities from the finite-precision, thresholded im-
plementation in Section 3. No theorem below proves retrieval benefit, atom necessity, or
absence of multi-concept interference. Throughout, Sd−1is the unit sphere of Rd,gdenotes
the negative diagonal-O2 unit direction, and ddenotes ambient dimension. Unit queries
and documents are assumed where stated. Numerical branches are specified separately in
Section A.6.
J.1 Trust-Region Geometry of the Delta Exit
Definition J.1(Edit segment and ideal trust region).Forq∈Sd−1and finiteδ̸=0, define
Eη(q) =q+ηδ
∥q+ηδ∥ 2,η∈[0, 1], (17)
where the denominator is nonzero. For 0<γ floor<1 and 0<θ cap≤π, define
γ=max(cosθ cap,γfloor),
Tγ(q) ={u∈Sd−1:⟨u,q⟩ ≥γ}.(18)
Then θmax=arccosγ=min(θ cap, arccosγ floor). The angular request is already incorpo-
rated into δ; the exit never increases its step above one. On a nonsingular segment the ideal
exit chooses the largest feasible step in [0, 1] . Finite bisection need not return that exact
real-valued endpoint.
Lemma J.2(Monotone exit on a bounded segment).If δis not parallel to q, set a=⟨δ,q⟩
andb=∥δ∥2. Then φ(η) =⟨E η(q),q⟩ is continuous on [0, 1] and strictly decreasing for positive
steps. If φ(1)≥γ , the full step is feasible. Otherwise a unique η∗∈(0, 1) satisfies φ(η∗) =γ ;
exact-arithmetic bisection brackets it, with feasible lower endpoints converging toη ∗.
Proof. Non-parallelism gives b−a2>0, so q+ηδ cannot vanish. Direct differentiation
gives
φ(η) =1+ηap
1+2ηa+η2b,
φ′(η) =−η(b−a2)
(1+2ηa+η2b)3/2.(19)
The derivative is negative for η>0 , and φ(0) =1>γ . Continuity gives the unique
crossing when φ(1)<γ . Maintaining feasible lower and infeasible upper endpoints gives
the bisection claim.
34

C2U in RAG
RemarkJ.3 (Degeneracy and numerical scope).If δ=tq , the normalized point stays qwhile
1+ηt>0 ; an antiparallel segment may reach a singular zero or flip sign. Such a segment is
not covered by the lemma. Finite FP32 checks and denominator safeguards do not establish
exact real-number boundary attainment or monotonicity of rounded cosines. The drift
result below applies to a normalized output satisfying its stated cap condition; numerical
tolerances must be accounted for when applying it to computed outputs.
Proposition J.4(Uniform worst-case drift bound).Assume q,q′∈Sd−1andq′∈ Tγ(q). For
every unit vector u∈Sd−1, in particular every frozen document embedding,
⟨q′,u⟩ − ⟨q,u⟩≤ ∥q′−q∥ 2
=2 sin∠(q′,q)
2≤2 sinθmax
2.(20)
Proof. The inequality |⟨q′−q,u⟩| ≤ ∥q′−q∥ 2∥u∥ 2=∥q′−q∥ 2is Cauchy–Schwarz. For two
unit vectors separated by angle θ, the chord length is ∥q′−q∥ 2=√
2−2 cosθ=2 sin(θ/2) ,
andθ≤θ maxholds because q′∈ Tγ(q)by construction; sine is increasing on [0,π/2] , which
containsθ maxsinceγ>0.
Corollary J.5(Margin stability of the top-ranked document).Under the hypotheses of The-
orem J.4, let the fixed index contain at least two unit document vectors, with u1,u2ranked first
and second for q. Set m=⟨q,u 1⟩ − ⟨q,u 2⟩. Ifm>4 sin(θ max/2), then u1remains uniquely
top-ranked after editing.
Proof. By Theorem J.4 each score moves by at most 2 sin(θ max/2), so the post-edit gap is at
leastm−4 sin(θ max/2)>0.
RemarkJ.6 (Floor and cap).If θcap>arccosγ floor, the floor is tighter; at equality they
coincide. Below that angle the nominal cap is tighter. Equal floors across raised caps do
not imply equality with a smaller-cap edit, nor identical rankings or H5. The observed
operating-point ties and the residual-coefficient counterexample are empirical results in
Section A.6.
J.2 Exact Direction Routing through the Residual Atom
Definition J.7(Bank projection and completed coordinates).For A ∈RN×dand nonzero g,
let
a= (A⊤)+g,P=A⊤(A⊤)+,
p=Pg,e= (I−P)g.(21)
Here M+is the Moore–Penrose pseudoinverse. If e̸=0 , append r=e/∥e∥ and use
bβ= [a;β∥e∥] . Ife=0 , then p=g̸=0 : append r=p/∥p∥ and use bβ= [a; 0] . In both
casesA′= [A;r]is defined.
Proposition J.8(Projected plus residual recovers the full direction).In exact arithmetic, eis
orthogonal to the row span ofAand
A′⊤bβ=p+βe. (22)
Forβ=1 this is full g, including when gis in span. The projected coefficients aare the minimum-
norm preimage of p. Appending rraises rank by one exactly when e̸=0 ; in the in-span branch it
adds no rank.
Proof. P is the orthogonal projector onto range(A⊤), hence p⊥e andA⊤a=p . Ife̸=0 ,
the last row contributes β∥e∥r=βe , so the sum is p+βe ; atβ=1 ,p+e=g . Ife=0 ,
the last coefficient is zero and p=g . Pseudoinverse identities give the minimum-norm
preimage property. A nonzero orthogonal row raises rank by one, whereas normalized pis
already in the row span.
35

C2U in RAG
RemarkJ.9 (What residual completion does not say).A pure last-coordinate edit produces
only a multiple of ewhen e̸=0 . That is not the deployed full-direction route, which
also changes projected coefficients a. At a positive numerical threshold εr, the fallback for
0<∥e∥<ε ryields p, not exactly g, even before rounding; its exact-arithmetic truncation
error is ∥e∥. FP32 bank/weight subtraction adds further error. The theorem describes ideal
routing, not bitwise equality or a retrieval advantage over denseg.
Angular scale.For unit direction h,q⊤h=cosϕ with 0<ϕ<π , and 0≤θ<ϕ ,
solving for a turn of θalong q+kh gives k=sinθ/ sin(ϕ−θ) . Write h=cosϕq+sinϕv ⊥
and resolve the angle to obtain ksin(ϕ−θ) =sinθ . The runtime uses this scale with
unnormalized routed coefficients. At β=1 their ideal dense direction has unit norm; for
general β,∥p+βe∥ need not be one, so the same coefficient scale need not realize precisely
the requested angle. The trust exit acts afterward in either case.
J.3 Coherence Bounds for Multi-Concept Composition
For simultaneous suppression of K≥2 target concepts (Section 4.5), let δ1, . . . ,δ K∈Sd−1be
the unit full per-concept edit directions and write ∆=∑K
k=1ηkδkfor nonnegative coefficients.
Hereη kdenotes each pre-exit dense magnitude after angular allocation, not the exit stepη
or the allocated angle itself. The following is an exact-arithmetic bound on this sum.
Definition J.10(Mutual coherence).TheGram matrixof the concept directions is G∈RK×K,
Gkℓ=⟨δ k,δℓ⟩, and themutual coherenceisµ=max k̸=ℓ|Gkℓ|.
Proposition J.11(Cross-talk and spectral sandwich).With the notation above:
(a) each concept direction retains alignment⟨∆,δ k⟩ ≥η k−µ∑ℓ̸=kηℓ;
(b) every eigenvalueλof G lies in
1−(K−1)µ, 1+ (K−1)µ
, and therefore
max 
0, 1−(K−1)µ∥η∥2
2
≤ ∥∆∥2
2=η⊤Gη
≤ 
1+ (K−1)µ∥η∥2
2;(23)
(c)in the equal-strength case ηk=η>0 , the condition µ<1
K−1suffices for the composite edit to
keep positive alignment with every concept direction.
Proof. (a)⟨∆,δ k⟩=η k+∑ℓ̸=kηℓ⟨δℓ,δk⟩ ≥η k−µ∑ℓ̸=kηℓ, using ⟨δℓ,δk⟩ ≥ −µ andηℓ≥
0. (b) Ghas unit diagonal and off-diagonal entries bounded by µin absolute value, so
each Gershgorin disc of Gis centered at 1with radius ∑ℓ̸=k|Gkℓ| ≤(K−1)µ ; since Gis
symmetric, its eigenvalues are real and lie in the union of these discs. The Rayleigh-quotient
sandwich λmin(G)∥η∥2
2≤η⊤Gη≤λ max(G)∥η∥2
2with λmin(G)≥max(0, 1−(K−1)µ)
(the Gram matrix is positive semidefinite) and η⊤Gη=∑kηkδk2
2gives the claim. (c) Plug
ηk=ηinto (a):⟨∆,δ k⟩ ≥η 
1−(K−1)µ>0 for everyk.
RemarkJ.12 (Geometry is not retrieval non-interference).The norm bound depends on
both coherence and coefficient magnitudes. Positive alignment of a pre-normalization
sum does not guarantee improved per-concept retrieval after normalization and trust
clipping. Nor can the residual-atom heatmap in Figure 8 instantiate G: atβ=1 the full
directions contain projected and residual components. With a common exact projector,
⟨gk,gℓ⟩=⟨p k,pℓ⟩+⟨e k,eℓ⟩, not simply ⟨rk,rℓ⟩. Low residual coherence therefore does not
establish full-direction incoherence, causal isolation of dictionary benefits, or absence of
interference.
36

C2U in RAG
J.4 Weight Concentration and Gate-Off Identity
The weight generator assigns wx,i∝exp(sim(h x,ai)/τ) over the Natoms (Section A.3).
The next lemma describes input-weight concentration, not locality of the deployed edit:
least-squares coefficients can be dense regardless of the support or peakedness ofw x.
Lemma J.13(Temperature concentration of the atom weights).For N≥2 finite real scores
si=sim(h x,ai)and temperature τ>0 , order s(1)≥s(2)≥ · · · ≥s(N). Let∆i=s(1)−s(i)and
∆=min i≥2∆ibe the top-gap. Then
w(1)≥1
1+∑i≥2e−∆ i/τ
≥1
1+ (N−1)e−∆/τ,
∑
i≥2w(i)≤(N−1)e−∆/τ.(24)
In particular, at fixed positive top-gap the weights converge to the one-hot argmax at an exponential
rate asτ→0+.
Proof. Divide numerator and denominator of w(1)byes(1)/τto get w(1)= 
1+
∑i≥2e−∆ i/τ−1, then bound each ∆i≥∆ and sum the N−1 identical terms; the tail mass is
∑i≥2w(i)=1−w(1)≤(N−1)e−∆/τ
1+(N−1)e−∆/τ≤(N−1)e−∆/τ.
Proposition J.14(Gate-off exactness of the implementation branch).Under the validated
finite-input contract, if the deployed single-concept gate is zero, or its resulting dense delta is zero,
the runtime returns the original query row without modification. For multi-concept composition, all
per-concept gates zero likewise imply an unchanged row.
Proof. A zero gate gives zero angle and coefficient scale; identical padded and steered
weights produce zero delta. The exit’s active mask requires both a positive gate and a delta
norm above its numerical inactivity threshold. For inactive rows it explicitly copies the
original source query, bypassing normalization. With every multi-concept gate zero, all
allocations and scales vanish and the same copy branch applies. This is bit-exact copying of
the input representation, not a consequence of ideal normalization algebra. Historical αxis
absent from this argument and from the deployed operator.
RemarkJ.15.The ideal cap bound and explicit inactive branch explain bounded motion
and exact inactivity under their respective hypotheses. They do not establish dataset-level
suppression or a dictionary advantage; empirical comparisons remain necessary.
37