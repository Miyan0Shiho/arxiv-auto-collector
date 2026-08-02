# ScoreShield: Differentially Private Release of Similarity Scores

**Authors**: Behrooz Razeghi, Parsa Rahimi

**Published**: 2026-07-27 20:02:28

**PDF URL**: [https://arxiv.org/pdf/2607.25041v1](https://arxiv.org/pdf/2607.25041v1)

## Abstract
A growing number of applications, such as biometrics and retrieval-augmented generation (RAG), rely on cosine similarity scores computed between vector embeddings of text, images, or audio. These systems return similarity scores through their APIs for ranking and verification. However, such releases can leak information about individual records and enable membership inference attacks. While differential privacy (DP) provides a principled metric for quantifying attack risks, naïve application of DP mechanisms---such as adding i.i.d. Gaussian noise to vector entries---leads to excessive distortion (i.e., low utility) at a given privacy constraint that scales poorly with the number of released scores. We propose \textsc{ScoreShield}, a perturb-then-project mechanism that adds Gaussian noise calibrated to global sensitivity of the chosen score release regime and then projects the result onto the feasibility set of valid cosine objects. \textsc{ScoreShield} satisfies $(\varepsilon,δ)$-DP for releasing similarity score vectors and Gram matrices. We provide utility guarantees for the exact Frobenius metric projection used in the risk analysis, and prove convergence to feasibility for the practical averaged alternating-projection solver used for large-scale Gram releases. For full pairwise cosine Gram release under record-level replacement adjacency, the exact-projection bound improves the $n$-dependence of squared Frobenius risk from $Θ(n^3)$ for the naïve Gaussian baseline to $\mathcal{O}(n^2)$ for fixed privacy parameters, with sharper local bounds at low-rank Grams. We evaluate the mechanism across RAG, face recognition, semantic retrieval, image similarity, and recommender-system tasks.

## Full Text


<!-- PDF content starts -->

arXiv preprint, ScoreShield
ScoreShield: Differentially Private Release of Similarity Scores
Behrooz Razeghi
1,Parsa Rahimi
2
1School of Engineering and Applied Sciences, Harvard University,2School of Engineering, École
Polytechnique Fédérale de Lausanne (EPFL)
A growing number of applications, such as biometrics and retrieval-augmented generation (RAG), rely
on cosine similarity scores computed between vector embeddings of text, images, or audio. These
systems return similarity scores through their APIs for ranking and verification. However, such
releases can leak information about individual records and enable membership inference attacks. While
differential privacy (DP) provides a principled metric for quantifying attack risks, naïve application of
DP mechanisms—such as adding i.i.d. Gaussian noise to vector entries—leads to excessive distortion
(i.e., low utility) at a given privacy constraint that scales poorly with the number of released scores.
We proposeScoreShield, a perturb-then-project mechanism that adds Gaussian noise calibrated to
global sensitivity of the chosen score release regime and then projects the result onto the feasibility
set of valid cosine objects.ScoreShieldsatisfies( ε,δ)–DP for releasing similarity score vectors and
Gram matrices. We provide utility guarantees for the exact Frobenius metric projection used in the risk
analysis, and prove convergence to feasibility for the practical averaged alternating-projection solver
used for large-scale Gram releases. For full pairwise cosine Gram release under record-level replacement
adjacency, the exact-projection bound improves the n-dependence of squared Frobenius risk fromΘ( n3)
for the naïve Gaussian baseline to O(n2)for fixed privacy parameters, with sharper local bounds at
low-rank Grams. We evaluate the mechanism across RAG, face recognition, semantic retrieval, image
similarity, and recommender-system tasks.
Date:July 29, 2026
Correspondence:behroozrazeghi@seas.harvard.edu
Code:https://github.com/BehroozRazeghi/scoreshield
1 Introduction
Similarity scores drive retrieval, verification and search [28, 31, 32]. In RAG and enterprise semantic search,
a service that maintains an indexed corpus returns, for each query, a ranked list of the top- kmatched
records together with a per-record similarity (or relevance) score computed using a similarity metric [ 18].
In biometric verification, standard matcher interfaces likewise return a scalar match score to the calling
application. Recent work demonstrates black-box membership inference against RAG datastores, in which
an adversary decides whether a target passage is contained in the retrieval database by issuing queries and
analyzing the observable outputs of the system (retrieved context and/or generated responses) [ 2,19,20,29].
Moreover, similarity-score statistics can be leveraged to detect and localize membership-inference attempts,
since they re-use the similarity scores already computed during top- kselection [ 6]. The biometrics literature
has long noted that exposing match scores enables score-guided hill-climbing attacks [11, 22].
A direct way to protect these score releases is to treat them as real-valued query outputs and apply a standard
DP mechanism, for example by adding Gaussian noise calibrated to the global sensitivity of the released
vector or matrix. This baseline is model-agnostic and gives a formal( ε,δ)-DP guarantee. However, it treats
cosine-score vectors and cosine Gram matrices as unconstrained Euclidean arrays. For a fixed unit-norm
queryqand stored unit-norm embeddingse 1,...,en, the released score vector( ⟨e1,q⟩,...,⟨ en,q⟩)must lie
in[−1,1]n. For full pairwise release, the score matrixS=EE⊤must be symmetric, positive semidefinite, and
have unit diagonal. Entrywise Gaussian perturbation does not preserve these constraints. Thus, the naïve
Gaussian mechanism is formally private but geometrically mismatched: it can add noise in infeasible directions
and produce unnecessarily distorted inputs for ranking, thresholding, retrieval, clustering, verification, or
recommendation.
1
arXiv:2607.25041v1  [cs.IR]  27 Jul 2026

arXiv preprint, ScoreShield
Figure 1.ScoreShieldis a general, modality-agnostic post-processing DP wrapper for releasing similarity scores
under central-model differential privacy. Given a downstream task Tthat consumes similarity scores, the curator
serves a non-private model and applies a calibrated Gaussian perturbation followed by projection onto the feasibility
set, yielding a released vector/matrix that satisfies( ε,δ)-DP with a quantified privacy–utility tradeoff. Red dotted
arrow denotes the standard (non-private) release of similarity scores, while green dotted arrow denotes releasing the
same score objects through ScoreShield mechanism.
We study one-shot release of similarity scores under record-level( ε,δ)-DP in the trusted-curator model [ 10].
The curator does not release embeddings; it releases either (i) a privatized score vector between one external
query/probe embedding and all stored embeddings, used in verification, nearest-neighbor search, or re-ranking
[7,9,15,24,26,35,40], or (ii) a privatized pairwise cosine Gram matrix of the stored collection, used in non-
interactive analytics such as de-duplication, clustering, or bias auditing [ 1,25,30,33,34,38].ScoreShield
applies the Gaussian mechanism to the chosen score statistic, with variance σ2
ε,δ=cε,δ∆2calibrated to its
globalℓ2sensitivity, where cε,δ= 2log(2/δ)/ε2, and then applies a privacy-preserving post-processing map
that enforces the corresponding cosine feasibility constraints; see Figure 1. Figure 2 summarizes the main
risk behavior ofScoreShield. The stated Gram-risk bounds are for the exact Frobenius metric projection,
which is DP-preserving post-processing and cannot increase squared distance to the non-private feasible Gram
matrix.
Main Contributions.
1.Quantitative Improvement over Naïve Gaussian DP:For Gram-matrix release, the naïve Gaussian
mechanism has expected squared Frobenius errorΘ( n3cε,δ)under record-level replacement adjacency and
Θ(n2∆2
Gcε,δ)under∆ G-Gram adjacency. For the exact Frobenius metric projection onto the cosine-Gram
feasibleset, theglobalGaussian-complexityboundforScoreShieldgives O(n2√cε,δ)andO(n3/2∆G√cε,δ),
respectively. At rank- rGram matrices satisfying the local Gram-smoothness condition, the local tangent-
cone bound gives O(n2rcε,δ)andO(nr∆2
Gcε,δ), respectively. For vector releases, projection never increases
squared-error risk and preserves theΘ( ncε,δ)worst-case scaling of naïve Gaussian DP, with possible
constant-factor gains.
2.Utility Guarantees for Regime (i):For query-to-collection vector releases calibrated to( ε,δ)-DP: (i)
We characterize the probability that perturbation noise flips the accept/reject decision for any interior
thresholdτ∈(−1,1). (ii) We prove existence and uniqueness of a threshold re-calibration that restores
any attainable operating point( FPR,TPR )achievable by thresholding the privatized scores, and show that
in the small-noise regime σ→ 0the required offset scales asΘ( σ2)with a coefficient determined by the
score density at τ(closed form for Gaussian impostors). (iii) We establish uniform ROC/AUC stability:
underL-Lipschitz score CDFs, both ROC coordinates and AUC deviate by at most O(σ). ThisO(σ)rate
is first-order tight. The same O(σ)scaling holds for EER and partial-AUC. (iv) For small ε, we bound the
sensitivity of the false match rate (FMR) function across neighboring galleries by O(ε+δ)under one-shot
(ε,δ)-DP release.
2

arXiv preprint, ScoreShield
0 500 1000 1500 2000 2500 3000
n(gallerysize)101102103104MechanismMSE/c,
c,=1fixed
Regime(i):VectorRelease
Gaptonaiveupperbound
Naive
Boxbound(a=n)
Pointwisemin.upperbound
0 500 1000 1500 2000 2500 3000
n(gallerysize)1021041061081010MechanismMSE/c,
c,=1fixed
Regime(ii):GramReleaseunder(R)/(O)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)
Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)NormalizedMSEupperboundsforfixedc,=1
Figure 2. Normalized MSE Scaling of Naïve Gaussian vs. ScoreShield Mechanisms.Left (regime (i), vector release):
releasing a query-to-collection cosine score vector.Right (regime (ii), matrix release):releasing the full pairwise cosine
Gram matrix.
3.A Fast, Scalable Projection Method with Guarantees:For regime (ii), we use an averaged alternating-
projection (AAP) algorithm for cosine Grams that alternates between the PSD cone K+
nand the entrywise
constraint setCn
unit. We prove R-linear convergence to feasibility under bounded linear regularity; the
AAP limit is not, in general, the exact Frobenius metric projection.
4.Evaluation:We evaluate regime (i) for two applications: (a) differentially private retrieval-augmented
generation (DP-RAG); (b) differentially private face-recognition (DP-FR). For DP-RAG, we evaluate
regime (i) on the Google FRAMES benchmark for multi-hop RAG, employing multiple LLM generators
and an LLM-based evaluator, and multiple text embedders for retrieval. For DP-FR, we evaluate regime (i)
on seven public FR benchmarks (e.g.,LFW, IJB-C ) using three backbones across various privacy settings
on an(ε,δ)grid and report several utility metrics. We evaluate regime (ii) onvision/NLP/recommender
tasks that consume only the private cosine Gram (CIFAR-10/100 and Oxford-IIIT Pets, STS-B, and
MovieLens-100K).
Positioning.To our knowledge, we provide the first systematic central-model( ε,δ)-DP analysis of match-
score releases in deep face recognition systems that connects threshold recalibration to decision-level guarantees
and validates on standard public FR benchmarks.
Notation.Fore ∈Rd, the Euclidean norm is denoted by ∥e∥2=/parenleftbig/summationtextd
i=1e2
i/parenrightbig1/2, and the unit sphere in Rdis
the set Sd−1:={e∈Rd:∥e∥2= 1}. For a matrixA ∈Rn×d, the Frobenius norm is ∥A∥F=/radicalbig
tr(A⊤A). The
spectral norm ofA, denoted as ∥A∥2, is the supremum supe∈Sd−1∥Ae∥2and equals its largest singular value.
The nuclear norm ∥A∥∗is the sum of the singular values ofA, i.e.,/summationtextmin (n,d)
k=1σk(A). The maximum entry
norm ofAis defined as ∥A∥max=max 1≤i≤n,1≤j≤d|Aij|. A symmetric matrixA ∈Rn×nis PSD, denoted
A⪰0, ifv⊤Av≥0,∀v∈Rn. For any integer n≥1,In×ndenotes the n×nidentity matrix. The standard
deviation parameter in the Gaussian mechanism is denoted by σ. The Euclidean projection operator onto a
closed convex setCis written asprojC(·).
Related Work.Releasing pairwise statistics with central DP has a long history [ 4,39]. Direct output
perturbationaddsnoisecalibratedtotheglobalsensitivityofthereleasedobject; underrecord-levelreplacement,
releasing a full n×nGram modifies an entire row/column, so the expected squared Frobenius error can scale as
Θ(n3)(up to log factors in δ) unless additional structure is used [ 14]. These approaches do not guarantee that
a released cosine Gram remains PSD with unit diagonal and bounded entries, nor do they provide guarantees
for decision-level verification metrics. Perturb-and-Project (PnP) [ 8] improves utility for cosine-similarity
release by adding Gaussian noise and then projecting onto an admissible set of Gram matrices. Their analysis
controls error via the Gaussian complexity of the feasibility set, yielding tighter Frobenius-risk bounds than
naïve perturbation. We differ in four respects: (i) we calibrate to record-level replacement for two disclosure
regimes used in practice, and we also analyze the alternative∆ G-Gram (output-space) adjacency used in prior
3

arXiv preprint, ScoreShield
work; (ii) we enforce cosine-specific feasibility (PSD with unit diagonal and entrywise bounds) to guarantee
validity of the released object; (iii) we provide decision-level utility guarantees (threshold recalibration, flip
probability, ROC/AUC stability) that are not captured by norm-only bounds on ℓ2or Frobenius error, and
(iv) we prove convergence guarantees for a scalable alternating-projection algorithm.
A complementary line of work privatizes embeddings rather than scores. Johnson–Lindenstrauss (JL)
transforms with calibrated Gaussian noise achieve vector-level DP while approximately preserving distances
[16]. Under suitable spectral assumptions, the Gaussian JL transform itself can satisfy( ε,δ)-DP without an
additional additive-noise step [ 4]. These methods protect the published vectors, but they do not by themselves
bound leakage when the release object is an entire similarity vector or the full Gram matrix, which is our
focus.
DP-FR:Regime (i) matches the standard deep FR pipelines that operate on cosine similarity scores. Prior
privacy-preserving FR research focuses on (i) local-DP or image/feature obfuscation [ 5,13] and (ii) crypto-
graphic inference (HE/MPC) [ 3]. To our knowledge, prior work has not studied central-model( ε,δ)-DP for the
non-interactive publication of semantically valid FR similarity vectors together with decision-level verification
guarantees. Perturb–and–project releases for cosine similarities do not provide verification decision-metric
guarantees [8]. Regime (i) therefore addresses a gap in the FR literature.
DP-RAG:Recent work studies differentially private RAG, where the goal is to protect an indexed corpus
against leakage under (possibly adaptive) querying and generation. Existing approaches vary by what is
privatized: some privatize retrieval outcomes (e.g., DP selection of IDs/ranks/scores) and treat subsequent
processing as DP post-processing of the privatized retrieval output, while others spend privacy budget in
the generation stage to make the final answer text DP, or privatize the corpus once via a DP synthetic
proxy dataset [ 12,17,27,37]. The multi-query regime is nontrivial because per-query privacy costs compose;
recent mechanisms exploit relevance sparsity via screening and per-record privacy accounting to answer many
queries under a fixed total budget [ 36]. Our regime (i) provides a one-shot central-DP primitive for releasing
a bounded cosine score vector, and hence supports DP rankings/top- k/thresholding via post-processing, when
the corpus is accessed only through these privatized scores. This is complementary to multi-query accountants
and to end-to-end DP generation mechanisms.
For extended discussion, see Appendix A and Appendix B.
2 Preliminaries
For completeness, Appendix C collects extended preliminaries and proofs. The proofs of all lemmas and
corollaries in this section are deferred to the appendix.
Setup.LetD={x1,...,xn}be a dataset of nrecords (e.g., images, text chunks, user profiles), and let
ϕ:X →Rdbe a (fixed) feature encoder. We assumee i:=ϕ(xi)and∥ei∥2= 1,∀i∈ [n]. Stacking row
vectors givesE= [e⊤
1,...,e⊤
n]⊤∈Rn×d. We study the differentially private release of cosine-similarity
statistics derived fromEunder record-level replacement adjacency on D. Our two disclosure regimes are:(i)
a query-to-collection score vectors :=Eq∈[−1,1]nfor a public query embeddingq ∈Rdwith∥q∥2= 1;
and(ii)the full pairwise cosine Gram matrixS :=EE⊤∈Rn×nwith entries Sij=⟨ei,ej⟩∈[−1,1].
Accordingly, the score-vector feasibility set is Cquery :={s∈Rn:|si|≤1,∀i∈ [n]}= [−1,1]n. Since
∥ei∥2= 1,Sis a correlation matrix:S ⪰0and diag(S) =1; accordingly we define the feasible Gram set
Ccoll:={S∈Rn×n:S⪰0,diag (S) =1,|Sij|≤1 (i̸=j)}, and note rank(S)≤min{n,d} . We instantiate
this setup in face recognition, retrieval, and other similarity-based tasks.
Differential Privacy.
Definition 2.1(Record-level Adjacency).Datasets D,D′(or equivalently, their embedding matricesE ,E′)
areadjacent, denoted by D∼D′, if they differ in at most one record (and thus in one embedding), i.e.,
|{i∈[n] :x i̸=x′
i}|= 1.
Remark2.2.Under a fixed backbone model ϕθ, adjacency implies that the embedding matricesEandE′
differ in at most one row. Therefore, there exists at most one index isuch thate j=e′
j,∀j̸=i, and by our
normalization assumptione i̸=e′
i∈Rdsatisfy∥e i∥2=∥e′
i∥2= 1.
4

arXiv preprint, ScoreShield
Definition 2.3(Output–space (Gram-matrix) Adjacency).Two collections with embeddingsE ,E′are
adjacent at radius∆ G>0if∥EE⊤−E′E′⊤∥F≤∆ G.
Definition 2.4(( ε,δ)–DP).A (possibly randomized) mechanism M:E→Osatisfies(ε,δ)–DP if for all
measurableT ⊆Oand all adjacentE ∼E′,Pr[M(E)∈T]≤eεPr[M(E′)∈T] +δ, whereε >0and
δ∈(0,1)are privacy parameters.
Definition 2.5( ℓ2-Sensitivity).For a function f:E→Rn, theℓ2-sensitivity is∆ f,2=supE∼E′∥f(E)−
f(E′)∥2. We use the notation∆for brevity.
Lemma 2.6(Gaussian Mechanism).Let f:E→Rnhaveℓ2–sensitivity∆ f,2=supE∼E′∥f(E)−f(E′)∥2.
DefineM(E)=f(E) +w,w∼N/parenleftbig
0,σ2In/parenrightbig
,σ2≥cε,δ∆2
f,2withcε,δ:=2 log(2/δ)
ε2. ThenMsatisfies(ε,δ)-DP.
Corollary 2.7(Gaussian Mechanism for Matrix-Valued Outputs).Let f:E→Rn×nand define the Frobenius
sensitivity∆ f,F:=supE∼E′∥f(E)−f(E′)∥F. LetW∈Rn×nhave i.i.d. entries Wij∼N(0,σ2), and define
M(E) :=f(E) +W. Ifσ2≥cε,δ∆2
f,F, withcε,δ= 2 log(2/δ)/ε2, thenMis(ε,δ)-DP.
Lemma 2.8(Post-processing).Let M:E→Obe a mechanism that satisfies( ε,δ)–differential privacy. For
any measurable function g:O→O′, the composed mechanism g◦M :E→O′also satisfies( ε,δ)–differential
privacy.
Convex-Geometry.
Definition 2.9(Euclidean Projection Onto a Closed Convex Set).Let m≥ 1and letC⊂Rmbe non–empty,
closed and convex. The Euclidean projection operator projC:Rm→Cis defined for everys ∈Rmby
projC(s):= arg miny∈C∥s−y∥ 2.
Definition 2.10(Tangent Cone).Let C⊂Rmbe non-empty, closed, and convex and fix a points ∈C. The
tangent cone toCatsisT s(C):=cl/braceleftbig
λ(y−s) :λ≥0,y∈C/bracerightbig
⊂Rm, where “ cl” denotes the Euclidean
closure. Geometrically,T s(C)contains all velocity directions of feasible curves that start atsand remain
insideC.
Definition 2.11(Gaussian Complexity of a Bounded Set).Let C⊂Rmbe bounded. Its Gaussian complexity
isGC(C) :=Ew∼N(0,Im)[ sups∈C⟨w,s⟩]. IfCis unbounded we setGC(C) := +∞by convention.
Lemma 2.12(Gaussian Complexity of Cquery).LetCquery :={s∈Rn:|si|≤1,∀i∈ [n]}= [−1,1]nand
z∼N(0,I n). Consider the Gaussian complexity definition C.16. ThenGC(C query) =n/radicalig
2
π= Θ(n).
Lemma 2.13(Gaussian Complexity of Ccoll).LetCcoll:={S∈Rn×n:S⪰0,diag (S) =1,|Sij|≤1 (i̸=j)}.
ThenGC(C coll) = Θ(n3/2).
Lemma 2.14(Firm Non -expansiveness of Euclidean Projections).Let C⊂Rnbe non-empty, closed and
convex and define the Euclidean projector projC(s) = arg mins′∈C∥s−s′∥2. Then for alls ,s′∈Rnwe have
∥projC(s)−projC(s′)∥2
2≤ ∥s−s′∥2
2.
Design Objective.Given private unit-norm embeddings, the task is to release cosine similarities without
releasing the embeddings. The released object is either a vector of similarities between a public query and
the database, or a matrix of pairwise similarities among database records. The mechanism must satisfy
record-level differential privacy and output an element of the corresponding cosine-feasible set. The objective
is to minimize distortion relative to the non-private similarities.
Threat Model.A curator holds the database (equivalently, the embedding matrixE) and makes a single,
non-interactive release of a privatized similarity object (either a score vector in regime (i) or a Gram matrix in
regime (ii)). The adversary is an external analyst who observes only the released output, knows the mechanism
and all public parameters, including( ε,δ), and has unbounded computational power. We allow arbitrary
auxiliary information, including knowledge of all records except the one that may differ under the adjacency
relation.
5

arXiv preprint, ScoreShield
Algorithm 1DP Query-to-Collection Similarity Score Vector Release (regime (i))
1:Input:E∈Rn×d,q∈Rd,ε>0,δ∈(0,1)
2:Output:/hatwides∈Rn
3:Computes=E q
4:Setσε,δ←∆ query/radicalbig
2 log(2/δ)/ε
5:Sample noisew∼N/parenleftbig
0,σ2
ε,δIn/parenrightbig
6:Computes′=s+w
7:Compute/hatwides=projC(s′),
where(projC(s′))i= max/parenleftbig
−1,min(1,s′
i)/parenrightbig
8:Return/hatwides
3 ScoreShield Framework
3.1 Query-to-Collection Similarity Score Vector Release
Operational Scenario.At run time, a client (or sensor) produces a fresh ℓ2-normalized probe embedding
q∈Rdand submits it for matching. The server stores a fixed gallery of unit-norm enrollment embeddings
E= [e⊤
1;...;e⊤
n]∈Rn×dand computes the cosine-similarity vectors=E q= (e⊤
1q,...,e⊤
nq)∈[−1,1]n.
This vector drives the downstream decisions (e.g., thresholding for1 :1verification or top- kranking for open-set
search). Formally, the function to be privatized is fquery(E,q) =E q∈Rn. We omit the subscript in fquery
when the context is clear.
ss′
/hatwides
C
Figure 3.ScoreShieldVisu-
alization.Sensitivity.Fix any publicq ∈Sd−1. Denote δ:=ei−e′
iand note that
∥δ∥2≤2(e.g.,e i=q,e′
i=−q). Then underE ∼E′differing in row i,
f(E,q)−f(E′,q) =Eq−E′q=/parenleftbig
0,..., 0,δ⊤q,0,..., 0/parenrightbig
, hence, the inner
product is bounded by Cauchy–Schwarz as follows ∥f(E,q)−f(E′,q)∥2=
|δ⊤q|≤∥δ∥ 2∥q∥2≤2. The global ℓ2-sensitivity is therefore the constant
∆f,2=:∆query= 2.
DP Mechanism.The DP query-to-collection similarity vector release al-
gorithm is provided in Algorithm 1. A conceptual illustration is shown in
Figure 3.
Theorem 3.1(Privacy Guarantee of Query-to-Collection Similarity Score
Vector Release).Let Mquerybe the mechanism returned by Algorithm 1 with
parameters ε>0, δ∈ (0,1). Then for every pair of neighboring embedding
matricesE∼E′and every measurable set S⊆ [−1,1]n,Pr/bracketleftbig
Mquery(E)∈
S/bracketrightbig
≤eεPr/bracketleftbig
Mquery(E′)∈S/bracketrightbig
+δ. HenceM queryis(ε,δ)–DP.
Proof.See Appendix Theorem G.1 in Appendix G.
Risk Scaling: Naïve Gaussian vs. ScoreShield Mechanism
Lemma 3.2(Global ScoreShield Bound via Gaussian Complexity).Let m≥ 1and letCquery⊂Rmbe
nonempty, closed, convex, and bounded. Fixs ∈C queryand letw∼N (0,σ2Im). Consider ScoreShield
projector/hatwides=projCquery(s+w). Then
E∥/hatwides−s∥2
2≤4σGC(C query),(1)
where the expectation is overwand for any boundedA⊂Rm,GC(A) :=E w∼N(0,Im)/bracketleftbig
supa∈A⟨w,a⟩/bracketrightbig
.
Proof.See Lemma F.1 in Appendix F.
6

arXiv preprint, ScoreShield
Lemma 3.3(Local ScoreShield Bound via Tangent Cones).Let m≥ 1and letCquery⊂Rmbe nonempty,
closed, and convex. Fixs∈C queryand letw∼N(0,σ2Im). Let/hatwides=projCquery(s+w). Then
E∥/hatwides−s∥2
2≤σ2δ(Ts(Cquery)),(2)
whereT s(Cquery)is the tangent cone ofC queryatsand
δ(K) :=E/bracketleftbig
∥projK(z)∥2
2/bracketrightbig
,z∼N(0,I m),(3)
denotes the statistical dimension of a closed convex coneK⊂Rm.
Proof.See Lemma F.3 in Appendix F.
Naïve Gaussian Mechanism.Releases′=s+w,w∼N(0,σ2In)withσ2=cε,δ∆2
query= 4cε,δ. Then we have
E∥s′−s∥2
2=E∥w∥2
2=nσ2= 4cε,δn (4)
ScoreShield Mechanism.Let /hatwides=projCquery(s+w). We have:
Local (Pointwise) Risk Bound via the Tangent Cone.For the constraint set Cquery= [−1,1]n, the tangent cone
atsis a product cone whose statistical dimension depends only on the active set. Let a(s)denote the number
of coordinates ofson the boundary {±1}. Thenδ(Ts(Cquery)) =n−1
2a(s)(see Lemma G.9 for a proof), and
the local conic denoising bound (Lemma 3.3) yields
E∥/hatwides−s∥2
2≤σ2/parenleftig
n−1
2a(s)/parenrightig
= 4cε,δ/parenleftig
n−1
2a(s)/parenrightig
. (5)
If no coordinate is active, a(s) = 0and δ(Ts) =n, i.e., the bound matches the naïve risk. If many coordinates
are saturated (large a(s)), projection can reduce the bound by up to a factor1
2on those coordinates. Since
∆queryis constant, the per-coordinate MSE isO(1)for both mechanisms.
Global Risk Bound via the Gaussian Complexity.Letz ∼ N (0,In). For a universal constant C > 0,
Lemma C.21 and Lemma F.1) give the uniform estimate
E∥/hatwides−s∥2
2≤CσGC(C query) =Cσn/radicalbigg
2
π. (6)
Adversary Gain.We analyze attacker reconstruction under three knowledge regimes: (K0) no side information
aboutE; (K1) full knowledge ofE(sos=Eq ∈col (E)); (K2) DP-admissible side information revealing
E−i(all rows except the single row ithat differs under adjacency), sos −i∈col (E−i). In (K1), under the
no-clipping Gaussian model the optimal linear denoiser projects onto col(E), yielding average per-entry MSE
σ2
ε,δr/nwherer=rank(E). In (K2), the attacker can denoise the unchanged coordinates via projection onto
col(E−i), but the changed coordinate remains at MSEσ2
ε,δ.
Extended discussion and additional results are provided in Appendix F.4.
3.2 Full Pairwise Similarity Score Matrix Release
Operational Scenario.Batch analytics workflows such as unsupervised clustering, multi-camera trajectory
association, or demographic bias auditing require the complete pairwise similarity structure of the embeddings.
To support these non-interactive tasks the curator makes a one-shot disclosure of the cosine Gram matrix
S=E E⊤∈[−1,1]n×n, whereSij=e⊤
iej. The statistic to be privatized is therefore ffull(E) =EE⊤∈Rn×n.
We drop subscript for when the context is clear.
7

arXiv preprint, ScoreShield
Algorithm 2DP All-Pairs Similarity Matrix Release (regime (ii))
1:Input:E∈Rn×d,ε>0,δ∈(0,1), sensitivity∆>0
2:Output:/hatwideS∈C coll⊆Rn×n
3:Construct:S←EE⊤.
4:Gaussian calibration: setσ2
ε,δ←cε,δ∆2
5:Sample noiseW∼N/parenleftbig
0,σ2In×n/parenrightbig
6:Perturb with noise:S′=S+W
7:Project:/hatwideS=projCcoll(S′)=arg min A∈C coll∥A−S′∥2
F
8:Return/hatwideS
Sensitivity.Under output-space (Gram-matrix) adjacency, the global Frobenius sensitivity is∆ f,F= ∆ Gby
definition (for a fixed radius∆ G); see Appendix E.1Under record-level adjacency,EandE′differ in one row
i, soE′=E+∆where∆has a single nonzero row δ⊤with∥δ∥2≤2. LetS=EE⊤andS′=E′E′⊤. Then
S′−Sis supported only on row/column i, and the diagonal satisfies(S′−S)ii= 0since∥ei∥2=∥e′
i∥2= 1.
Writingvj:=δ⊤ej, we obtain∥S′−S∥2
F= 2/summationtext
j̸=iv2
j≤2/summationtext
j̸=i∥δ∥2
2∥ej∥2
2= 2(n−1)∥δ∥2
2≤8(n−1), and
therefore the global Frobenius sensitivity is∆ f,F=:supE∼E′∥S′−S∥ F= 2/radicalbig
2(n−1) = :∆full.
DP Mechanism.The DP full pairwise similarity score matrix release algorithm is provided in Algorithm 2.
Theorem 3.4(Privacy Guarantee of Full Pairwise Similarity Score Matrix Release).Let ffull:E→Rn×nbe
ffull(E) =S=EE⊤. Fix an adjacency relation ∼onEand assume that for some∆ >0,∥ffull(E)−ffull(E′)∥F≤
∆,∀E∼E′. Letε >0,δ∈(0,1),cε,δ:= 2log(2/δ)/ε2, and setσ2:=cε,δ∆2. LetW∈Rn×nhave i.i.d.
entriesWij∼N(0,σ2). Let projCbe any (possibly randomized) measurable post-processing that depends on
Eonly through its input argument. Define the release /hatwideS=Mcoll(E) :=projC(ffull(E) +W) . Then the
mechanismE∝⇕⊣√∫⊔≀→ /hatwideSis(ε,δ)-DP with respect to∼, i.e.,
Pr[M coll(E)∈S]≤eεPr[M coll(E′)∈S] +δ,∀E∼E′.(7)
Proof.See Theorem J.1 in Appendix J.
Fast Projection via Averaged Alternating Projections.LetS′=S+W∈Rn×nbe a perturbed Gram
matrix, whereS=EE⊤is the clean cosine Gram matrix and ∥ei∥2= 1,∀i. Our goal is to projectS′onto the
cosine-Gram feasibility set Ccoll:=/braceleftbig
S∈Rn×n:S⪰0, Sii= 1 (1≤i≤n ),|Sij|≤1 (i̸=j)/bracerightbig
⊂Rn×n. The
exact projection onto the cosine-Gram feasible set Ccollunder the Frobenius norm is the metric projection onto
the elliptope and in general requires solving an SDP. Instead we compute an approximately feasible point
by iterating a Krasnosel’ski˘ ı-Mann averaged projector [ 21,23], and hence replace the direct projection by
alternating projections onto two closed convex sets, each admitting a closed-form projector. We decompose
Ccoll=Kn
+∩Cn
unitwhere
Kn
+={S∈Rn×n|S⪰0},Cn
unit={S∈Rn×n|Sii= 1,|Sij|≤1,(i̸=j)}.(8)
Starting from /hatwideS0:=S′, we iterate the equal-weights averaged map
/hatwideSt+1=1
2/parenleftig
projKn
+/parenleftbig
sym(/hatwideSt)/parenrightbig
+projCn
unit/parenleftbig
sym(/hatwideSt)/parenrightbig
,sym(Y) :=1
2(Y+Y⊤).(9)
This feasibility map depends only on the privatized matrixS′(and any auxiliary randomness used internally),
hence it is post-processing and does not affect the privacy guarantee. The fast projection method is provided
in Algorithm 3.
1If∆ Gis instead chosen to dominate record-level replacement, then necessarily∆ G≥2/radicalbig
2(n−1), hence∆ G= Θ(√n); see
Appendix E.
8

arXiv preprint, ScoreShield
Algorithm 3Fast Alternating Projection ontoC coll
1:Input:E∈Rn×d,ε>0,δ∈(0,1), sensitivity∆>0, toleranceτ >0ormaximum iterationsT∈N
2:Output:/hatwideS∈C coll⊆Rn×nwith(ε,δ)–DP guarantee s.t./vextenddouble/vextenddouble/hatwideS−projCcoll(S+W)/vextenddouble/vextenddouble
F≤τ
3:Construct:S←EE⊤.
4:DP noise:setσ2←cε,δ∆2
5:Sample noiseW∼N/parenleftbig
0,σ2In×n/parenrightbig
6:S′←S+1
2/parenleftbig
W+W⊤/parenrightbig
7:Initialize/hatwideS(0)←S′
8:fort= 0toT−1do
9:/* symmetrize current iterate */
10:Y t=1
2/parenleftbig/hatwideSt+/hatwideS⊤
t/parenrightbig
11:/* projection onto PSD ConeKn
+*/
12:Compute eigen-decompositionY t=Udiag(λ 1,...,λn)U⊤
13:λ+
k←max{0,λ k},∀k∈[n]
14:Pt←projKn
+(Yt) =Udiag(λ+
1,...,λ+
n)U⊤
15:ifFrobenius-ball constraint is enforced and∥P t∥F>nthen
16:P t←(n/∥P t∥F)Pt % radial projection ontoKn
+∩Bn
F
17:end if
18:/* projection onto Unit-Hyper-CubeCn
unit*/
19:Q t←projCn
unit(Yt)%(Q t)ii= 1;(Qt)ij= clip((Y t)ij,−1,1)fori̸=j
20:/* averaged update */
21:/hatwideSt+1←1
2/parenleftbig
Pt+Qt/parenrightbig
22:/* stopping tests */
23:r chg←∥/hatwideSt+1−/hatwideSt∥F
24:r psd←∥Yt−Pt∥F;r box←∥Yt−Qt∥F
25:ifr chg≤τandmax{r psd,rbox}≤τthen
26:break
27:end if
28:end for
29:/* PSD on return */
30:S avg←/hatwideSt+1
31:/hatwideS←projKn
+(1
2(Savg+S⊤
avg))
32:ifmin i/hatwideSii≤0then/hatwideS←/hatwideS+µIwith smallµ>0(e.g.,µ= 10−8∥/hatwideS∥F/n)
33:D←diag( /hatwideS)1/2,/hatwideS←D−1/hatwideS D−1
34:Output:/hatwideS
Projection onto the PSD Cone.Given a symmetric iterateY=Y⊤=1
2/parenleftbig/hatwideSt+/hatwideS⊤
t/parenrightbig
∈Sn,/hatwideS0:=S+W, we
compute its spectral decompositionY=U diag(λ1,...,λn)U⊤, costingO(n3). All subsequent steps are
performed in the eigen-basisU.
The Frobenius–orthogonal projector onto the PSD cone solves minS⪰0∥S−Y∥2
Fand is obtained as follows.
Defineλ+
k:=max{ 0,λk}, k= 1,...,n, and let λ+:= (λ+
1,...,λ+
n). The orthogonal projector onto the PSD
cone is
projKn
+(Y) =Udiag(λ+
1,...,λ+
n)U⊤.(10)
This map is a firmly non-expansive projector (see Lemma C.27).
ProjectionOntotheUnit-DiagonalBox.Theset Cn
unit:=/braceleftbig
S∈Rn×n:|Sij|≤1 (i̸=j), Sii= 1 (1≤i≤n )/bracerightbig
is an axis-aligned hyper-box with fixed diagonal. Because the Frobenius norm decouples over coordinates, the
orthogonal projector is entry-wise. For anyY∈Rn×nthe projector ontoCn
unitdecouples entry-wise as
/bracketleftbig
projCn
unit(Y)/bracketrightbig
ij=

1, i=j,
clip(Yij,−1,1), i̸=j,(11)
9

arXiv preprint, ScoreShield
where clip(y,−1,1):=max{− 1,min{ 1,y}}. Note that for each off-diagonal coordinate the convex problem
min|z|≤1 (z−Yij)2yields the clip operator. Moreover, the diagonal constraint is enforced exactly. This
operationisfirmlynon-expansive(seeLemmaC.27), costs O(n2)arithmeticoperationsandcanbeimplemented
in-place (withΘ(n2)storage for the matrix itself).
Risk Scaling: Naïve Gaussian vs. ScoreShield MechanismConsider the ScoreShield projector
/hatwideS=projCcoll(S+W), whereprojCcollis the Frobenius (metric) projection ontoC coll. Then
E∥/hatwideS−S∥2
F≤4σGC(C coll),(12)
whereGC(A) =E/bracketleftbig
supA∈A⟨Z,A⟩ F/bracketrightbig
(see Corollary F.2 for more details).
Under record-level adjacency (R) the exact Frobenius sensitivity is∆ f,F= 2/radicalbig
2(n−1) = Θ(√n), hence
σ2= Θ(n). Under output-space (Gram) adjacency (O) we have ∥S−S′∥F≤∆Gwith∆ G= Θ(1)independent
ofn, henceσ2= Θ(1).
Naïve Gaussian Mechanism.In our practical algorithm, we sampleWwith i.i.d. entries Wij∼N(0,σ2)
(Gaussian mechanism), and then apply the deterministic symmetrization post-processingG :=1
2(W+W⊤).
SinceSis symmetric, the symmetrized releaseS′:=1
2/parenleftbig
(S+W) + (S+W)⊤/parenrightbig
=S+Gis a post-processing
ofS+Wand therefore preserves( ε,δ)-DP2(withσcalibrated to the sensitivity ofSunder the chosen
adjacency). We report utility for this symmetric pre-projection matrixS′. A direct variance calculation gives
E∥S′−S∥2
F=E∥G∥2
F=nσ2
/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
diagonal+n(n−1)σ2
2/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
off-diagonal=n2+n
2σ2= Θ(n2σ2).(13)
Therefore, for the two adjacency definitions we have:
(a) Record-level adjacency (R): Usingσ2=cε,δ∆2
f,F= Θ/parenleftbignlog(2/δ)
ε2/parenrightbig
,
naïve + (R):E∥S′−S∥2
F= Θ/parenleftig
n2cε,δ∆2
F,rec/parenrightig
= Θ/parenleftign3log(2/δ)
ε2/parenrightig
. (14)
(b) Output-space adjacency (O): Withσ2=cε,δ∆2
G,
naïve + (O):E∥S′−S∥2
F= Θ/parenleftig
n2cε,δ∆2
G/parenrightig
= Θ/parenleftign2∆2
Glog(2/δ)
ε2/parenrightig
. (15)
Global Risk Bound via the Gaussian Complexity.Let /hatwideS=projCcoll(S+G)for the regime (ii) Algorithm 5.
Using Corollary F.2 we have
E∥/hatwideS−S∥2
F≤C σGC(C coll)≤/tildewideCσn3/2,(16)
where we usedGC(C coll) = Θ(n3/2).
(a) Record-level adjacency (R): Withσ= Θ/parenleftbig√
nlog(2/δ)
ε/parenrightbig
,
ScoreShield + (R):E∥ /hatwideS−S∥2
F≤/tildewideCn2/radicalbig
log(2/δ)
ε=O/parenleftigg
n2/radicalbig
log(2/δ)
ε/parenrightigg
. (17)
(b) Output-space adjacency (O): Withσ=√
2 log(2/δ)
ε∆G,
ScoreShield + (O):E∥ /hatwideS−S∥2
F≤/tildewideCn3/2∆G/radicalbig
log(2/δ)
ε=O/parenleftigg
n3/2∆G/radicalbig
log(2/δ)
ε/parenrightigg
.(18)
2Equivalently, samplingGdirectly as a symmetric Gaussian with Var(Gii) =σ2andVar(Gij) =σ2/2,∀i̸=j, yields the
same distribution as1
2(W+W⊤).
10

arXiv preprint, ScoreShield
We refer the reader to Appendix F.5 for extended discussion.
Adversary Gain.We analyze attacker reconstruction under two knowledge regimes: (i) no side information and
(ii) side information where the attacker knows the gallery embeddings {ej}j̸=i(one-row unknown). Extended
discussion and additional results are provided in Appendix F.6.
4 Experiments
WeevaluateScoreShieldinthetworeleaseregimesstudiedinthepaper. Inregime(i), themechanismreleases
a single privatized score vector. We instantiate this regime in face recognition and in single-query retrieval-
augmented generation. In regime (ii), the mechanism releases a single privatized cosine Gram matrix. We
evaluate downstream tasks that consume only the released similarity matrix or deterministic transformations
of it. Across both regimes, we compare the utility of the non-private release, the Gaussian mechanism applied
directly to the score object, and the proposed perturb-then-project release. Full experimental details, extended
grids, and ablations are deferred to App. I, App. H, and App. K.
For regime (i), all guarantees are for a single released score vector under central-model record-level replacement.
The face-recognition study uses three representative LFW score sets for operating-point analysis and seven
public FR benchmarks for aggregate evaluation. The DP-RAG study uses theFRAMESbenchmark, EG300M
retrieval embeddings, and Gemma-family generators. For regime (ii), we evaluate CIFAR-10/100, Oxford-IIIT
Pets, STS-B, and MovieLens-100K. In every matrix benchmark we compare the clean (non-private) Gram
matrixS, the symmetrized noisy releaseS′, and the projected release /hatwideS; an SDP projector is included only
where it is computationally tractable.
4.1 Regime (i): Similarity Score Vector Release
Regime (i): DP-FR.For face recognition, average score distortion does not by itself determine deployment
utility. The relevant question is whether a target false-match operating point remains attainable after
privatization. We therefore evaluate utility at the decision level. We use seven public benchmarks: LFW,
CFP-FP, CALFW, CPLFW, AgeDB, IJB-B, and IJB-C. For LFW, CFP-FP, CALFW, CPLFW, and AgeDB,
we report verification accuracy. For IJB-B and IJB-C, we report TAR at FPR∈{ 10−6,10−5,10−4}. We
evaluate WebFace4M-trained IR101 and ViT-Base backbones. Full benchmark tables, calibration sweeps,
synthetic score experiments, and operating-point analyses are reported in App. H. Figure 4 illustrates how
privacy noise affects the verification operating point. At a fixed privacy budget, analytic Gaussian calibration
injects less noise than the conservative sufficient calibration, and therefore gives a lower post-privacy FMR
near the target threshold. The public-margin curves show the possible gain under the condition cmin=−0.5;
the worst-case formal calibration remains∆ = 2. Panels (b)–(c) show the same effect through the required
threshold correction: as the noise scale increases, the target FMR may become infeasible under strict endpoint
semantics.
Table 1 reports a fixed- δslice of the full DP-FR benchmark grid. At δ= 10−4, utility improves as εincreases,
but the recovery depends strongly on the operating point. For IR101, increasing εfrom70to100raises LFW
accuracy from86 .72%to92.05%, while IJB-B TAR at FPR10−6increases from53 .82to73.18. At the less
stringent IJB-B FPR10−4, the same change increases TAR from78 .60to89.87. The ViT-Base rows show the
same qualitative pattern. At ε∈{ 20,35}, the strict10−6and10−5IJB operating points are not attained
in several cases under the strict endpoint convention, although LFW accuracy remains nonzero. The added
Gaussian noise first affects the most stringent low-FPR decisions, whereas verification accuracy on LFW,
CFP-FP, CALFW, CPLFW, and AgeDB, and less stringent IJB thresholds, degrade more gradually.
Regime (i): DP-RAG.We evaluate score-vector release as a retrieval primitive onFRAMES. For each
query, the retriever computes cosine scores over a Wikipedia chunk index. Clean RAG applies thresholded
top-kretrieval tos(q), whereas DP-RAG applies the same rule to /hatwides(q). The generator, prompt format,
decoding parameters, and LLM judge are fixed across all conditions. The privacy guarantee applies only to
the released score vector and its post-processing. Here the source corpus is public Wikipedia text, so retrieved
identifiers, thresholded retrieval sets, and answers generated from retrieved text are treated as post-processing
of the privatized scores. If the corpus text were private under the same adjacency relation, releasing chunks
or generated answers would require an additional privacy mechanism.
11

arXiv preprint, ScoreShield
(a)FMR curves atε= 10
 (b)∆τversusσ
 (c)∆τversusε
Figure 4. DP-FR operating-point behavior under score-vector release.All panels use LFW ArcFace-101/WebFace4M
scores with target FMR α= 10−2under strict endpoint semantics. (a) Post-privacy FMR curves at δ= 10−6
andε= 10, comparing conservative and analytic Gaussian calibration under worst-case sensitivity∆ = 2, and the
public-margin model∆ = 1 −cmin= 1.5withcmin=−0.5. (b) Threshold correction∆ τ(σ)versus Gaussian noise scale;
shading marks strict-endpoint infeasibility, where no thresholdτ <1attains the target FMR. (c) The corresponding
∆τ–εcurve for analytic Gaussian calibration with∆ = 2and δ= 10−6; filled markers denote feasible thresholds and
inverted markers infeasible budgets.
Table 1. Representative DP face-recognition results at δ= 10−4.For LFW we report verification accuracy. B-10−6,
B-10−5, and B-10−4denote TAR on IJB-B at the corresponding FPR; C-10−6, C-10−5, and C-10−4are defined
analogously for IJB-C. “Avg.” is the macro-average over IJB-B and IJB-C TAR at FPR∈{ 10−6,10−5,10−4}, and
verification accuracy on LFW, AgeDB, CFP-FP, CALFW, and CPLFW. The clean row is non-private. Entries equal
to0.00in the IJB columns indicate that the requested operating point is not attained under the strict endpoint
convention used in the appendix. Full grids overε,δ, backbones, and benchmarks are reported in App. H.
Backbone(ε,δ)LFW↑B-10−6↑B-10−5↑B-10−4↑C-10−6↑C-10−5↑C-10−4↑Avg.↑
IR101/WebFace4M
IR101 clean 99.70 89.46 93.07 95.52 43.49 89.07 93.72 89.74
IR101(100,10−4)92.05 73.18 83.68 89.87 29.36 76.48 86.85 78.23
IR101(70,10−4)86.72 53.82 66.66 78.60 20.35 60.17 75.10 67.81
IR101(35,10−4)71.52 0.00 0.00 18.01 0.00 0.00 17.60 33.43
IR101(20,10−4)61.67 0.00 0.00 0.00 0.00 0.00 0.00 26.69
ViT-Base/WebFace4M
ViT-Base clean 99.80 87.12 94.54 96.89 38.62 90.51 95.39 90.01
ViT-Base(100,10−4)93.43 74.98 86.14 91.97 29.72 78.45 88.94 80.34
ViT-Base(70,10−4)87.07 52.67 67.86 80.93 35.75 60.49 77.05 70.72
ViT-Base(35,10−4)71.82 0.00 0.00 18.26 0.00 0.00 17.05 34.10
ViT-Base(20,10−4)62.30 0.00 0.00 0.00 0.00 0.00 0.00 27.05
Table 2 reports representative small- δDP-RAG settings onFRAMES. Clean retrieval improves accuracy in all
displayed settings, with gains from4 .73to20.75points over the no-context baseline. Under DP score release,
accuracy depends on whether the privatized scores preserve the thresholded top- kretrieval set. The k= 50
G3-12B/EG300M row shows that larger retrieval depth alone does not make RAG robust to score perturbation.
The G3-27B/EG300M rows remain close to the no-context baseline at ε= 1, while Q3-8B/Q3VL-E2B gives
the strongest displayed small-δresult, slightly exceeding the baseline at(ε,δ) = (10,10−6).
4.2 Regime (ii): Similarity Score Matrix Release
Given normalized embeddingsE ∈Rn×d, the non-private score object is the cosine Gram matrixS=EE⊤.
We compare the clean matrixS, the symmetrized Gaussian perturbationS′, and the projected release /hatwideS. The
projected matrix is obtained by projecting the perturbed matrix onto the cosine-Gram feasible set Ccoll. Since
this projection is post-processing of the noisy release, it preserves the same( ε,δ)-DP guarantee. For every
benchmark, the downstream method receives onlyM ∈{S,S′,/hatwideS}, or a fixed deterministic transformation of
M. The encoder, sample, split, graph construction, prediction rule, and evaluation metric are fixed within
12

arXiv preprint, ScoreShield
Table 2. Representative small δDP-RAG results onFRAMES.“Base” denotes generation without retrieved context.
“Full” uses the linked Wikipedia pages and serves as an evidence upper bound. “RAG” uses thresholded top- kretrieval.
Clean rows retrieve froms(q), while DP rows retrieve from /hatwides(q). “Gain” is RAG accuracy minus the corresponding
no-context baseline. The protocol is shown in App. Fig. I.1; full grids are reported in App. I.
Generator Embedding Scores(ε,δ) (τ,k)Base Acc. Full Acc. RAG Acc. Gain
Gemma3-12B EG300M clean –(0.25,50)45.87 72.09 55.46+9.59
Gemma3-12B EG300M DP(1,10−5) (0.25,50)46.36 71.48 39.44−6.92
Gemma3-12B Q3VL-E2B clean –(0.25,10)46.36 71.48 56.92+10.56
Gemma3-12B Q3VL-E2B DP(1,10−6) (0.25,10)46.36 71.48 42.23−4.13
Gemma3-12B Q3VL-E2B DP(10,10−6) (0.25,10)46.36 71.48 41.75−4.61
Gemma3-27B EG300M clean –(0.35,20)40.17 72.82 60.92+20.75
Gemma3-27B EG300M DP(1,10−5) (0.35,20)40.17 72.82 39.68−0.49
Gemma3-27B EG300M DP(1,10−6) (0.35,20)40.17 72.82 37.86−2.31
Gemma3-8B Q3VL-E2B clean –(0.25,10)75.85 91.02 80.58+4.73
Gemma3-8B Q3VL-E2B DP(1,10−5) (0.25,10)75.85 91.02 74.27−1.58
Gemma3-8B Q3VL-E2B DP(10,10−6) (0.25,10)75.85 91.02 76.46+0.61
each experiment. Thus, utility differences are attributable to the matrix supplied to the downstream method.
Tasks.We evaluateScoreShieldon three classes of matrix-based downstream tasks. For image similarity,
we use frozen DINOv2-B/14 embeddings and form cosine Gram matrices on CIFAR-10/100 and Oxford-IIIT
Pets. Pairwise verification uses Mijas the score for the label 1{yi=yj}and reports ROC–AUC. Nearest-
neighbor classification assigns each image the majority label among the5largest off-diagonal entries in its
row ofMand reports top-1 accuracy. Instance retrieval ranks candidates j̸=iby decreasing Mijand
reports Recall@1. Spectral clustering forms the affinityA= (M+1) /2with zero diagonal and reports NMI
against the ground-truth labels. For semantic textual similarity, we embed the unique STS-B sentences with
SBERT; each labeled pair( u,v)is scored by the released entry Muv, and utility is Spearman correlation with
human similarity scores. For recommendation systems, we form a MovieLens-100K user–user cosine Gram
matrix from mean-centered rating vectors and predict held-out ratings by a similarity-weighted neighborhood
estimator; utility is RMSE.
Representative results.Figure 5 reports representative results at δ= 10−8and∆ = 2, the main calibration
setting used for the displayed panels. The panels cover image verification on Oxford-IIIT Pets and CIFAR-10,
image retrieval on Oxford-IIIT Pets, spectral clustering on CIFAR-100, semantic textual similarity on STS-B,
and collaborative filtering on MovieLens-100K. Full extended results over∆, δ, datasets, and additional
metrics are reported in App. K. Gaussian perturbation alone can reduce utility becauseS′is not guaranteed
to remain a valid cosine Gram matrix: it can be indefinite, its diagonal can differ from one, and its entries
can lie outside[−1,1]. These violations affect the row-wise rankings, neighborhoods, and graph spectra used
by the downstream methods. Projecting onto Ccollenforces the PSD, unit-diagonal, and entrywise cosine
constraints, and improves utility overS′at the same privacy level in the displayed settings. The improvement
is most visible in tasks that depend on relative similarity structure, including retrieval, clustering, semantic
similarity, and recommendation. For very small ε, the injected noise dominates and post-processing cannot in
general recover the clean ordering. As εincreases,/hatwideSmoves toward the non-private baseline while preserving
the same DP guarantee as the noisy release.
Projection scalability.We also compare AAP with an SDP projector for the nearest feasible cosine-Gram
problem. The SDP baseline is useful at small matrix sizes, but becomes time- or memory-limited for the
larger matrices used in STS-B, MovieLens-100K, and Oxford-IIIT Pets. AAP remains tractable at these sizes
because each iteration uses a closed-form unit-diagonal box projection and one PSD projection. In the small- n
cases where both projectors are run, AAP gives comparable downstream utility; at the larger scales used in
the main experiments, AAP is the practical projection method.
13

arXiv preprint, ScoreShield
(a)Pets verification AUC
 (b)Pets Recall@1
 (c)CIFAR-10 verification AUC
(d)CIFAR-100 clustering NMI
 (e)STS-B Spearmanρ
 (f)MovieLens RMSE
Figure 5. Regime (ii): representative utilities for DP cosine-Gram release.Each downstream task uses only the
released matrixM ∈{S,S′,/hatwideS}or a fixed deterministic transformation of it. The panels show representative results
for image verification, image retrieval, spectral clustering, semantic textual similarity, and collaborative filtering. All
displayed panels use δ= 10−8and∆ = 2, the main setting used in the paper; complete grids over∆, δ, datasets, and
metrics are reported in App. K. Higher is better for AUC, Recall@1, NMI, and Spearman correlation; lower is better
for RMSE. Projection improves the raw Gaussian release by restoring the PSD, unit-diagonal, and entrywise cosine
constraints required of a valid cosine Gram matrix.
5 Conclusion
We introducedScoreShield, a perturb–then–project framework for the central-model( ε,δ)-differentially
private release of cosine similarity score vectors and cosine Gram matrices. The mechanism adds Gaussian
noise calibrated to the sensitivity of the chosen release regime and then projects the perturbed output onto the
corresponding feasibility set of valid cosine objects. This projection is privacy-preserving by post-processing
and enforces the structural constraints required by the released object. For vector release,ScoreShield
preserves the privacy guarantee, does not increase squared error relative to the unprojected Gaussian release,
and allows downstream thresholding, ranking, and top- kselection rules applied to the privatized score vector
to inherit the same differential-privacy guarantee by post-processing. For full Gram release,ScoreShield
exploits cosine-Gram geometry to improve Frobenius mean-squared error scaling over the unprojected Gaussian
baseline, and we introduced a scalable averaged alternating-projection solver with convergence guarantees for
feasibility projection. Experiments on retrieval-augmented generation and face recognition evaluate the vector-
release regime, while Gram-only evaluations on semantic textual similarity, image similarity and clustering,
and recommender-system tasks evaluate the Gram-release regime; across these settings, the results support the
predicted privacy–utility behavior across modalities and quantify the benefit of feasibility enforcement. These
results show that constraint-aware privatization of similarity scores is a useful primitive for similarity-based
systems.
6 Limitations and Future Works
Our results characterize a single release of cosine scores for fixed normalized embeddings under the stated
central-DP adjacency relation. They do not address repeated or adaptive score queries, for which privacy
losses compose and tight composition accounting is required. Nor do they characterize utility-optimal noise
distributions for cosine-score release under joint( ε,δ)-DP and cosine-feasibility constraints; the Gaussian
14

arXiv preprint, ScoreShield
mechanisms analyzed here satisfy the privacy guarantee but are not proved minimax- or instance-optimal.
Acknowledgments
This work was supported by the Swiss National Science Foundation (SNSF) under Grant No. 222339. The
authors thank Dr. Flavio P. Calmon, Dr. Sébastien Marcel, Dr. Shahab Asoodeh, and Dr. Hatef Otroshi
Shahreza for insightful discussions and helpful suggestions that helped improve this work.
References
[1]Amro Kamal Mohamed Abbas, Kushal Tirumala, Daniel Simig, Surya Ganguli, and Ari S Morcos. SemDeDup:
Data-efficient learning at web-scale through semantic deduplication. InICLR 2023 Workshop on Mathematical
and Empirical Understanding of Foundation Models, 2023.
[2]Maya Anderson, Guy Amit, and Abigail Goldsteen. Is my data in your retrieval database? membership inference
attacks against retrieval augmented generation. InInternational Conference on Information Systems Security and
Privacy, volume 2, pp. 474–485. Science and Technology Publications, Lda, 2025.
[3]Jianli Bai, Xiaowu Zhang, Xiangfu Song, Hang Shao, Qifan Wang, Shujie Cui, and Giovanni Russello. Cryptomask:
Privacy-preserving face recognition. InInternational Conference on Information and Communications Security,
pp. 333–350. Springer, 2023.
[4]Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. The johnson-lindenstrauss transform itself preserves
differential privacy. In2012 IEEE 53rd Annual Symposium on Foundations of Computer Science, pp. 410–419.
IEEE, 2012.
[5]Mahawaga Arachchige Pathum Chamikara, Peter Bertok, Ibrahim Khalil, Dongxi Liu, and Seyit Camtepe. Privacy
preserving face recognition utilizing differential privacy.Computers & Security, 97, 2020.
[6]Yujin Choi, Youngjoo Park, Junyoung Byun, Jaewook Lee, and Jinseong Park. Safeguarding privacy of retrieval
data against membership inference attacks: Is this query too close to home?arXiv preprint arXiv:2505.22061,
2025.
[7]Ondrej Chum, James Philbin, Josef Sivic, Michael Isard, and Andrew Zisserman. Total recall: Automatic query
expansion with a generative feature model for object retrieval. In2007 IEEE 11th international conference on
computer vision, pp. 1–8. IEEE, 2007.
[8]Vincent Cohen-Addad, Tommaso d’Orsi, Alessandro Epasto, Vahab Mirrokni, and Peilin Zhong. Perturb-and-
project: differentially private similarities and marginals. InProceedings of the 41st International Conference on
Machine Learning, pp. 9161–9179, 2024.
[9]Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep
face recognition. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp.
4690–4699, 2019.
[10]Cynthia Dwork, Aaron Roth, et al. The algorithmic foundations of differential privacy.Foundations and trends®
in theoretical computer science, 9(3–4):211–407, 2014.
[11]Javier Galbally, Chris McCool, Julian Fierrez, Sebastien Marcel, and Javier Ortega-Garcia. On the vulnerability
of face verification systems to hill-climbing attacks.Pattern Recognition, 43(3):1027–1038, 2010.
[12]Nicolas Grislain. Rag with differential privacy. In2025 IEEE Conference on Artificial Intelligence (CAI), pp.
847–852. IEEE, 2025.
[13]Jiazhen Ji, Huan Wang, Yuge Huang, Jiaxiang Wu, Xingkun Xu, Shouhong Ding, ShengChuan Zhang, Liujuan
Cao, and Rongrong Ji. Privacy-preserving face recognition with learnable privacy budgets in frequency domain.
InEuropean Conference on Computer Vision, pp. 475–491. Springer, 2022.
[14]Tianxi Ji and Pan Li. Less is more: Revisiting the gaussian mechanism for differential privacy. In33rd USENIX
Security Symposium (USENIX Security 24), pp. 937–954, 2024.
[15]Jeff Johnson, Matthijs Douze, and Hervé Jégou. Billion-scale similarity search with gpus.IEEE Transactions on
Big Data, 7(3):535–547, 2019.
15

arXiv preprint, ScoreShield
[16]Krishnaram Kenthapadi, Aleksandra Korolova, Ilya Mironov, and Nina Mishra. Privacy via the johnson-
lindenstrauss transform.Journal of Privacy and Confidentiality, 5(1):39–71, 2013.
[17]Tatsuki Koga, Ruihan Wu, Zhiyuan Zhang, and Kamalika Chaudhuri. Privacy-preserving retrieval-augmented
generation with differential privacy.arXiv preprint arXiv:2412.04697, 2024.
[18]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive
nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[19]Hao Li, Jiajun He, Guangshuo Wang, Dengguo Feng, Zheng Li, and Min Zhang. Budgetleak: Membership
inference attacks on rag systems via the generation budget side channel.arXiv preprint arXiv:2511.12043, 2025.
[20]Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang. Generating is believing: Membership inference attacks
against retrieval-augmented generation. InICASSP 2025-2025 IEEE International Conference on Acoustics,
Speech and Signal Processing (ICASSP), pp. 1–5. IEEE, 2025.
[21]KRASNOSEL’SKII MA. Two comments on the method of successive approximations.Usp. Math. Nauk, 10:
123–127, 1955.
[22]Emanuele Maiorana, Gabriel Emile Hine, and Patrizio Campisi. Hill-climbing attacks on multibiometrics
recognition systems.IEEE Transactions on Information Forensics and Security, 10(5):900–915, 2014.
[23]W Robert Mann. Mean value methods in iteration.Proceedings of the American Mathematical Society, 4(3):
506–510, 1953.
[24]Brianna Maze, Jocelyn Adams, James A Duncan, Nathan Kalka, Tim Miller, Charles Otto, Anil K Jain, W Tyler
Niggel, Janet Anderson, Jordan Cheney, et al. Iarpa janus benchmark-c: Face dataset and protocol. In2018
international conference on biometrics (ICB), pp. 158–165. IEEE, 2018.
[25]Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and
fairness in machine learning.ACM computing surveys (CSUR), 54(6):1–35, 2021.
[26]Qiang Meng, Shichao Zhao, Zhida Huang, and Feng Zhou. Magface: A universal representation for face recognition
and quality assessment. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition,
pp. 14225–14234, 2021.
[27]Junki Mori, Kazuya Kakizaki, Taiki Miyagawa, and Jun Sakuma. Differentially private synthetic text generation
for retrieval-augmented generation (rag).arXiv preprint arXiv:2510.06719, 2025.
[28]Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. MTEB: Massive text embedding benchmark.
InProceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics,
pp. 2014–2037, 2023.
[29]Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, and Amir Houmansadr. Riddle me
this! stealthy membership inference for retrieval-augmented generation. InProceedings of the 2025 ACM SIGSAC
Conference on Computer and Communications Security, pp. 1245–1259, 2025.
[30]Andrew Ng, Michael Jordan, and Yair Weiss. On spectral clustering: Analysis and an algorithm.Advances in
neural information processing systems, 14, 2001.
[31]Hai Phan and Anh Nguyen. Deepface-emd: Re-ranking using patch-wise earth mover’s distance improves out-of-
distribution face identification. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pp. 20259–20269, 2022.
[32]Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese BERT-networks. In
Proceedings of EMNLP-IJCNLP, 2019.
[33]Jianbo Shi and Jitendra Malik. Normalized cuts and image segmentation.IEEE Transactions on pattern analysis
and machine intelligence, 22(8):888–905, 2000.
[34]Eric Slyman, Stefan Lee, Scott Cohen, and Kushal Kafle. FairDeDup: Detecting and mitigating vision-language
fairness disparities in semantic dataset deduplication. InProceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition, pp. 13905–13916, 2024.
[35]Cameron Whitelam, Emma Taborsky, Austin Blanton, Brianna Maze, Jocelyn Adams, Tim Miller, Nathan Kalka,
Anil K Jain, James A Duncan, Kristen Allen, et al. Iarpa janus benchmark-b face dataset. Inproceedings of the
IEEE conference on computer vision and pattern recognition workshops, pp. 90–98, 2017.
16

arXiv preprint, ScoreShield
[36]Ruihan Wu, Erchi Wang, and Yu-Xiang Wang. Beyond per-question privacy: Multi-query differential privacy for
rag systems. InNeurIPS 2025 Workshop: Reliable ML from Unreliable Data, 2025.
[37]Ruihan Wu, Erchi Wang, Zhiyuan Zhang, and Yu-Xiang Wang. Private-rag: Answering multiple queries with
llms while keeping your data private.arXiv preprint arXiv:2511.07637, 2025.
[38]Yang Wu, Zhiwei Ge, Yuhao Luo, Lin Liu, and Sulong Xu. Face clustering via graph convolutional networks
with confidence edges. InProceedings of the IEEE/CVF International Conference on Computer Vision, pp.
20990–20999, 2023.
[39]Mengmeng Yang, Tianqing Zhu, Lichuan Ma, Yang Xiang, and Wanlei Zhou. Privacy preserving collaborative
filtering via the johnson-lindenstrauss transform. In2017 IEEE Trustcom/BigDataSE/ICESS, pp. 417–424. IEEE,
2017.
[40]Zhun Zhong, Liang Zheng, Donglin Cao, and Shaozi Li. Re-ranking person re-identification with k-reciprocal
encoding. InProceedings of the IEEE conference on computer vision and pattern recognition, pp. 1318–1327, 2017.
17

arXiv preprint, ScoreShield
Appendix Contents
A Extended Introduction: Differentially Private Deep Face Recognition 19
B Extended Related Work 20
B.1 Differential Privacy for Vector, Covariance, and Gram Release Mechanisms . . . . . . . . . . . . . . . . . . . . . 20
B.2 Differentially Private Retrieval-Augmented Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
B.3 Differentially Private Face Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 22
C Extended Preliminaries 23
C.1 Notation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
C.2 Face Recognition . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 23
C.3 Differential Privacy . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
C.4 Convex Geometry . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 26
C.5 Non-Expansive Operators: Clipping and Euclidean Projection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
D Regime (iii): Per-record Similarity Score Vector Release 33
E Output–Space Adjacency for DP Face-Recognition 35
E.1 Calibrating∆ Gto Operational Scenarios . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
E.2 Bridging to Image Adjacency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 36
F Naïve Gaussian vs. ScoreShield Mechanism Under Different Release Regimes 37
F.1 ScoreShield Projection Risk Bounds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37
F.2 Regime (i): Query-to-Collection Similarity Score Vector . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 40
F.3 Regime (iii): Per-Record Similarity Score Vector . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
F.4 Attacker Reconstruction Error for Regime (i) & (iii) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
F.5 Regime (ii): Full Pairwise Similarity Score Matrix . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 45
F.6 Attacker Reconstruction Error for Regime (ii) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 47
F.7 Visual Comparison of MSE Scaling Bounds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 49
G Supplementary Details for Regime (i): Omitted Theorems, Propositions, Proofs and Lemmas 55
G.1 Privacy Guarantee and Stability of Projections . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55
G.2 Impact on Verification Thresholds . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
G.3 Effect on the ROC Curve and AUC . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
G.4 Rate-Optimal Upper and Matching Bound . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 70
H Supplementary Details for Regime (i): DP-FR 74
H.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 74
H.2 Performance Analysis . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 76
H.3 Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 85
I Supplementary Details for Regime (i): DP-RAG 89
I.1 Privacy Object, Adjacency, and Retrieval Scores . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 89
I.2 DP-RAG Mechanism and Privacy Calibration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 89
I.3 Experimental Protocol . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 90
I.4 Empirical Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 91
J Supplementary Details for Regime (ii): Omitted Theorems, Propositions, Proofs and Lemmas 95
J.1 Global Frobenius Sensitivity Under Record-Level Adjacency . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
J.2 DP Release Mechanism for the Full Pairwise Similarity Score Matrix . . . . . . . . . . . . . . . . . . . . . . . . . 95
J.3 Privacy Guarantee of ScoreShield for Regime (ii) . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 95
J.4 Fast Projection via Averaged Alternating Projections . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 96
J.5 Averaged Alternating Projection: Theoretical Guarantees . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 99
J.6 Symmetrization Prior to PSD Projection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 103
J.7 Dykstra’s Algorithm for the Metric Projection ontoKn
+∩Cn
unit. . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
J.8 Feasible Sets and Projection Decompositions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 105
K Supplementary Details for Regime (ii): Extended Experiments 106
K.1 Experimental Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 106
K.2 Benchmark Suite and Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 107
K.3 How the Gram is Consumed . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
K.4 Aggregation and Visualization . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
K.6 Benchmarks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 108
18

arXiv preprint, ScoreShield
A Extended Introduction: Differentially Private Deep Face Recognition
Deep face-recognition (FR) systems are deployed in consumer devices, border-control gates, CCTV networks
and large-scale photo grouping services. In these settings decision are made from cosine similarities: an image
is mapped to an ℓ2-normalized embedding, and the dot product with reference embeddings is used for ranking
and threshold-based match decisions. Although raw images often remain on device, similarity scores are
often stored (e.g., for auditing and analytics) or shared across services. Releasing similarity scores without
protection can enable membership inference about whether a specific enrolled record is in the gallery.
Motivation.In regime (i), we study central-model( ε,δ)-DP for releasing FR cosine-similarity scores. Due
to three obstacles, this problem not directly treated in prior work:
(i)What is released and how it scales.The common release objects in FR are eitherembeddingsor their
pairwise similarities. Under standard record-level adjacency, the global sensitivity of the matrix releases
grows with the gallery size, so naïvely adding Gaussian noise degrades accuracy at strict operating points
(very small FMR).
(ii)Accuracy in the low–false-match regime.FR is tuned at thresholds that target very small impostor
rates. Norm-based DP error bounds do not translate into decision-level guarantees (e.g., threshold shift,
false-match/false-non-match changes, ROC/AUC), making it unclear how to translate Euclidean error
bounds into guarantees on FMR/FNMR at a target threshold.
(iii)Geometry and computation.Valid similarity vectors/matrices form a structured set (PSD, unit diagonal,
bounded off-diagonals). Exploiting this geometry via projection can improve utility relative to uncon-
strained entrywise noise, but proving guarantees and obtaining scalable algorithms is nontrivial. Choose
of adjacency (image-level, identity-level, or Gram-bounded) and privacy accounting under repeated probe
releases also require explicit composition analysis.
Consequently, prior work either applies local/instance-level perturbations to inputs/features or relies on
HE/MPC without DP for released scores, leaving a gap: central-model( ϵ,δ)-DP guarantees for released
similarity scores are rarely provided.
Training-time DP (DP-SGD).A natural alternative is to train the encoder with training-time DP
guarantee (e.g., DP-SGD). We view this as orthogonal to our goal: training-time DP protects membership
with respect to thetraining corpus, whereas our risk concerns the privacy ofreleased similarity scoresover
galleries and probes that may include individuals never seen during training. In practice, deploying DP-SGD
within modern FR pipelines (large-batch schedules, margin-based losses) requires per-example clipping, careful
noise calibration, and full retraining. Accordingly, we treat training-time DP as complementary: if similarity
scores are released, release-time mechanisms remain necessary to certify central-model( ε,δ)-DP for the scores
themselves, regardless of how the encoder was trained.
Our resolution.Two ingredients make this setting analyzable. First, projection-based mechanisms for
pairwisesimilarities[ 14]showthataddingoneGaussianperturbationperreleaseandprojectingontothefeasible
set can improve utility (in Frobenius/MSE scaling) relative to unconstrained noise by exploiting structure.
Second, modern FR backbones exhibit calibrated score distributions that allow threshold-level (decision-level)
analysis at strict FMR targets. Building on these, we (a) formalize central, record-level adjacencies for
three practical FR-release settings; (b) derive closed-form global sensitivities and the corresponding Gaussian
standard deviations; and (c) couple projection with decision-level utility bounds at low FMR. This connects
generic central-DP releases of pairwise statistics to the needs of FR deployments.
Adjacency for central model DP deep FR.Prior FR papers labeled “DP” typically act in the local
model and perturb inputs or early features, without defining central-model neighbors for score releases.
Prior work on DP release of cosine similarity matrices [ 14] defines a∆ G-bounded Gram adjacency that is
task-agnostic. We define FR-specific central-model adjacencies at the image and identity levels and tie them
to enrollment and verification procedures, enabling sensitivity-calibrated Gaussian noise and operating-point
analysis for query (probe) vectors and Gram matrices of similarities.
19

arXiv preprint, ScoreShield
B Extended Related Work
B.1 Differential Privacy for Vector, Covariance, and Gram Release Mechanisms
Early vector-level random projections for privacy.A classical approach uses Johnson-Lindenstrauss
(JL) embeddings to releasevectorswith small distortion while approximately preserving pairwise distances.
Blockiet al.[ 9] showed that under specific spectral/conditioning assumptions (rank-1 neighbor change; all
singular values above a threshold), a randomized JL transformitselfcan satisfy differential privacy (“an old
dog performs new tricks”) and applied this to edge-DP graph cut queries and directional variance/covariance
release with distortion/estimation error bounds that does not scale with ambient dimension (with high
probability). Building on this tool, JL with calibrated noise was then specialized to collaborative filtering:
Yanget al.[ 45] proposed JLCF, proving ε–DP for a linear JL transform that approximately preserves
user–user distances and, under margin conditions, the neighbors ( k-NN structure), hence giving strong utility
fork-NN recommenders. While effective for vector publication, these methods do not, by themselves, bound
leakage when full similarity vectors or the entire Gram are disclosed.
Private ERM without heavy projections.A concurrent line in differentially private learning focuses on
optimization rather than projecting outputs. Talwaret al.[ 39] gave a nearly-optimal( ε,δ)-DP LASSO via a
private Frank-Wolfe method with the exponential mechanism, achieving excess risk ˜O((nε)−2/3)with only
logarithmic dependence on dimension and is optimal up to logarithmic factors under standard assumptions.
This direction is distinct from releasing pairwise similarities: it privatizes model training, whereas our setting
privatizes a structured output (a similarity vector or matrix).
Covariance release revisited.Recent central-model work highlights how clipping and trace-dependent
calibration affect utility for matrix-valued queries. Donget al.proposeAdaptiveCov[ 16], which DP-selects
clipping thresholds to balance bias and variance. On heavy-tailed data, AdaptiveCov achieves lower estimation
error than both isotropic Gaussian and coordinatewise baselines, motivating structure-aware calibration that
we later adapt to Gram release.
Structured output noise beyond full-rank Gaussians.Jiet al.[ 23] identify a “curse of full-rank
covariance”: for Gaussian mechanisms in their setting, the expected squared error depends on tr(Σnoise),
yielding matching lower bounds at fixed( ε,δ). They introduce a rank-1singular multivariate Gaussian
(R1SMG) mechanism whose covariance is random and rank-1, achieving( ε,δ)-DP with a different risk scaling
in the number of released coordinates. This is complementary to projection-based post-processing: one can
shapethe noise before projecting.
Perturb–and–Project (PnP) for structured releases.Cohen–Addadet al.[ 14] formalize a central-
model perturb-and-project template for releasing structured objects: add Gaussian noise calibrated to the
globalℓ2sensitivity∆, then perform Euclidean projection onto a compact convex feasibility set. By post-
processing, the mechanism is( ε,δ)-DP. Their utility guarantees are governed by the Gaussian complexity of
the feasibility set (rather than directly by the ambient dimension) and bound the expected squared error
to the true projection of the input onto that set. For releasing a cosine-similarity matrix, they apply this
template to the dataset’s Gram matrix and assume a Gram-space adjacency condition: neighboring datasets
are those whose induced Gram matrices differ by at most a prescribed Frobenius-norm radius, i.e., the total
squared change across all pairwise similarities is bounded.
B.2 Differentially Private Retrieval-Augmented Generation
Setup and Protection Goal.In retrieval-augmented generation (RAG), a server answers prompts by
retrieving relevant corpus items and conditioning a generator on the retrieved context. DP-RAG is commonly
formalized as corpus-level( ε,δ)-DP for the (possibly interactive) transcript produced under adaptively chosen
prompts. Across this literature, the privacy unit is often add/remove (document-deletion) at the document
level (or a specified privacy unit), motivated by membership-inference and data extraction attacks in external
knowledge bases [ 24,32,43].ScoreShieldregime (i) instead adopts chunk-level replacement adjacency and
treats prompts as public input in the curator model.
20

arXiv preprint, ScoreShield
Privatized Objects.DP-RAG proposals differ primarily in (i) the adjacency relation and (ii) which
intermediate or final objects are made private:
1.Retrieval-stage privatization(IDs/ranks/thresholds/scores): the mechanism releases a DP version of the
retrieval outcome (e.g., a DP top- kset, a DP threshold, or a DP score signal), and subsequent steps are
treated as post-processingof that released retrieval signal.3
2.Generation-stage privatization(DP decoding / private prediction): the system uses a DP mechanism during
token generation (often via prompting an LLM on multiple documents and aggregating with a DP rule), so
that the final text output satisfies DP w.r.t. the corpus.
3.One-time dataset privatization(DP synthetic corpus): the corpus is privatized once into a DP proxy dataset,
enabling unlimited downstream (non-private) retrieval and generation by post-processing.
These design points target different release goals and are therefore not generally interchangeable. That is
privatizing a retrieval signal is often sufficient when thepublic releaseis restricted to retrieval metadata
(IDs/ranks/scores), whereas end-to-end DP for thegenerated answer textgenerally requires additional structure
(generation-stage DP or dataset privatization) if the generator can access sensitive corpus text.
Single-query DP-RAG.A first line of work targets the single-query setting ( T= 1), where one prompt is
answered under a fixed privacy budget. Representative mechanisms achieve end-to-end DP for the answer
text by spending privacy budget in the generation stage (e.g., private-prediction / aggregated-generation style
approaches such asDPSparseVoteRAG) [ 24]. A complementary single-query strategy privatizes retrieval
identifiers rather than releasing raw retrieved text. For example, in [ 20] the DP-RAG system designs a DP
procedure to obtain a top- kretrieval set while preserving the feasibility of a downstream DP generation
step. Concretely, it (i) DP-selects a similarity threshold θ(via an exponential-mechanism step), (ii) forms a
candidate set above θ, and (iii) samples document indices under a DP distribution before invoking generation.
On the generation side, the same work follows a DP aggregated generation / DP in-context learning template:
it queries the LLM separately on each retrieved document and then aggregates token distributions with a DP
mechanism to produce the final output. This class of methods targets end-to-end DP for the answer text by
spending privacy budget during token selection.
Multi-Query DP-RAG is Nontrivial.In realistic deployments, an adversary can issue many (adaptive)
prompts against the same corpus. If one applies a single-query DP-RAG mechanism independently to each
query, under basic sequential composition, the privacy cost accumulates by sequential composition and can
become very large after modest T(e.g.,ε≈1000forT= 100queries at εq≈10) [42]. This motivates
mechanisms that exploitsparsity of relevance, the empirical observation that each query typically touches
only a small subset of the corpus, rather than paying as if all records participate in every query [42].
Multi-Query DP-RAG via Relevance Screening and Individual Privacy Filters (MURAG).The
“Beyond per-question privacy” framework develops DP-RAG algorithms explicitly designed for the multi-
query regime, introducingMURAGandMURAG-ADA[ 42]. Their core mechanism combines (i)relevance
screening, which restricts which records are eligible to be used for a given query, with (ii)individual privacy
accountingvia Rényi privacy filters, which track and halt a record’s participation once its ex-ante privacy
budget is exhausted [ 42]. At a high level,MURAGmaintains an active subset by screening records whose
relevance score exceeds a fixed threshold τ, answers the query using a single-query DP-RAG subroutine on the
screened set, and decrements per-record budgets after each use [ 42].MURAG-ADAreplaces a static τby a
query-adaptive private threshold calibrated to the (private) top- Kboundary: it discretizes similarity scores
into bins and releases noisy prefix sums with Laplace noise until the cumulative count exceeds K, spending a
dedicated budget εthrfor this threshold-release step [ 42]. They prove an overall corpus-level DP guarantee by
ensuring each record participates only while its per-record privacy filter remains below a target budget [ 42],
and empirically they report answering many queries (e.g., T= 100) under total budgets such as ε= 10[42].
3In this category, post-processing applies only insofar as downstream computation depends on the corpus exclusively through
the released DP retrieval signal (and public inputs/internal randomness), i.e., without additional access to sensitive record
contents.
21

arXiv preprint, ScoreShield
This line of research addressesmulti-query accountingrather than proposing a new one-shot privatization of
retrieval scores.
One-time dataset privatization for RAG via DP synthetic corpora (DP-SynRAG).A different
strategy avoids additional per-query privacy expenditure by privatizing the corpusonceand then running
standard (non-private) retrieval and generation over a DP proxy dataset. DP-SynRAG explicitly targets this
regime, noting that enforcing DP directly on LLM outputs in RAG consumes privacy budget per query and
can degrade rapidly as queries accumulate [ 32]. It adopts a private-prediction (subsample-and-aggregate)
paradigm to generate synthetic text, emphasizing that RAG requireslocality preservation(query-relevant fine
structure) rather than only global distributional similarity [ 32]. Provided downstream retrieval/generation
accesses only the DP proxy corpus, by producing a DP synthetic corpus once, subsequent retrieval and
answering incur no additional privacy cost by post-processing, yielding a fixed total privacy budget that scales
favorably with the number of queries [ 32], so the total privacy budget is fixed (independent of the number
of queriesT). Their experiments report lower attack success under repeated querying (under their threat
model/metrics) than per-query DP baselines [32].
Relation toScoreShield.ScoreShieldregime (i) can be viewed as a DP retrieval primitive that
privatizes the score interface itself. Given a normalized prompt embeddingqand chunk embeddingsx i, it forms
the cosine score vectors(q) :=/parenleftbig
⟨q,x1⟩,...,⟨ q,xn⟩/parenrightbig
∈[−1,1]n, and releases/hatwides(q) = proj[−1,1]n/parenleftbig
s(q)+w/parenrightbig
,w∼
N(0,σ2In). Under chunk-level replacement adjacency and clipping, at most one coordinate can change and
its magnitude is bounded, yielding global ℓ2-sensitivity∆ 2= 2and hence closed-form Gaussian calibration
for(ε,δ)-DP. Importantly, the DP guarantee applies directly to the released retrieval signal /hatwides(q)and to any
objects that are deterministic/randomized functions of /hatwides(q)and internal randomness only, such as rankings,
top-ksets, or thresholded retrievalidentifiers(by post-processing). This places regime (i) closest in spirit
to retrieval-stage privatization methods (selection/IDs/thresholds), and it is complementary to multi-query
layers (e.g.,MURAG/MURAG-ADA) that manage privacy over T≫ 1queries [ 42]. By contrast, providing
end-to-end DP for thegenerated answer textwhen the generator can accesssensitive corpus contenttypically
requires additional mechanisms beyond score release (e.g., generation-stage DP aggregation or one-time corpus
privatization), and is the explicit target of private-prediction DP-RAG and DP-SynRAG-style approaches. In
our experiments the retrieved corpus is public (Wikipedia), so returning retrieved text is not treated as a
private release; the DP claim concerns the retrieval chunk IDs/ranks induced by it. Empirically,ScoreShield
evaluates regime (i) as DP retrieval for multi-hop RAG on Google FRAMES using EmbeddingGemma for
retrieval and Gemma 3–12B as generator/judge [19, 25, 40].
B.3 Differentially Private Face Recognition
Local/instance-level FR perturbations.In FR, several prior works adopt a local/instance adjacency
and perturb inputs or early features [ 12,22]. For example, [ 22] maps images to frequency space (block-DCT),
removes the DC (zero-frequency) component, and injects DP noise with learnable per-frequency budgets
before feeding a standard FR backbone. This choice is motivated by the observation that visualization-critical
and identification-critical information separate in frequency. The adjacency is also reframed in a learned
representation (“secret”) space and target instance-level protection, rather than central-model DP guarantees
for public score releases.
22

arXiv preprint, ScoreShield
C Extended Preliminaries
C.1 Notation
We use three dimension parameters: (i) ndenotes the number of enrolled records, (ii) ddenotes the embedding
dimension, (iii) N:=dim(Sn) =n(n+1)
2denotes the ambient Euclidean dimension of symmetric n×nmatrices
Snunder Frobenius vectorization. For any integer m≥ 1, the unit sphere in RmisSm−1:={u∈Rm:∥u∥2=
1}. Fore∈Rd, the Euclidean norm is ∥e∥2=/parenleftbig/summationtextd
i=1e2
i/parenrightbig1/2. For any integer m≥ 1,Imdenotes the m×m
identity matrix. For a matrixA ∈Rn×d, the Frobenius norm is ∥A∥F=/parenleftbig/summationtextn
i=1/summationtextd
j=1A2
ij/parenrightbig1/2=tr/parenleftbig
A⊤A/parenrightbig1/2.
The spectral norm ofA, denoted as ∥A∥2, is the supremum supe∈Sd−1∥Ae∥2and equals its largest singular
value. The nuclear norm ∥A∥∗is the sum of the singular values ofA, i.e.,/summationtextmin (n,d)
k=1σk(A). The maximum
entry norm ofAis defined as ∥A∥max=max 1≤i≤n,1≤j≤d|Aij|. A symmetric matrixA ∈Rn×nis positive
semi-definite (PSD), denotedA ⪰0, ife⊤Ae≥0,∀e∈Rn. The standard deviation parameter in the
Gaussian mechanism is denoted by σ, and the Euclidean projection operator onto a closed convex set Cis
written as projC(·). For sequences an,bn>0, we write an=O(bn)if there exist constants C <∞andn0
such thatan≤Cbnfor alln≥n 0. We write an= Ω(bn)ifbn=O(an), andan= Θ(bn)if bothan=O(bn)
andan= Ω(bn)hold. Unless stated otherwise, asymptotics are withn→∞and(ε,δ)treated as fixed.
C.2 Face Recognition
Setup.LetD={x1,...,xn}denote a dataset of face images. A pre-trained backbone model ϕθmaps
each imagex ito anℓ2-normalized embedding vectore i=ϕθ(xi)∈Rdwith∥ei∥2= 1. The embeddings are
aggregated into a dataset matrixE ∈Rn×d, where each row corresponds to the transpose of an embedding,
i.e.,E= [e⊤
1,...,e⊤
n]⊤∈Rn×d. For fixed( n,d)define the admissible embedding space E:=/braceleftbig
E∈Rn×d:
∥ei∥2= 1,∀i∈ [n]/bracerightbig
, wheree⊤
idenotes row iofE. The Gram matrix (cosine similarity matrix) is defined
asS :=EE⊤∈Rn×n, with entries Sij=⟨ei,ej⟩∈[−1,1]. By construction (i)Sis PSD (S ⪰0), as Gram
matrices satisfyv⊤Sv=∥E⊤v∥2
2≥0for allv∈Rn, (ii) the diagonal entries satisfy Sii= 1,∀idue to
∥ei∥2= 1, (iii) the off-diagonal entries satisfy |Sij|≤1,∀i̸=j. Hence, we define the set of valid cosine Gram
matrices (correlation matrices) as
Ccoll:=/braceleftbig
S∈Rn×n:S⪰0, S ii= 1 (1≤i≤n),|S ij|≤1 (j̸=i)/bracerightbig
.(19)
The rank ofSis bounded by rank(S)≤min (n,d), since the rank of a matrix product cannot exceed the rank
of either factor.
Similarity Scores in Face-Recognition Systems.FR systems perform tasks like verification, iden-
tification, and clustering by measuring how similar face embeddings are to one another. For any pair
(i,j)∈[n]2
Sij:=⟨ei,ej⟩= 1−1
2∥ei−ej∥2
2,(20)
so cosine similarity and squared Euclidean distance are affine transforms of each other, hence, they induce
identical rankings of pairs. Decision thresholds translate via τeuc= 2(1−τcos)orτcos= 1−1
2τeuc, where
τcos∈[−1,1], andτ euc∈[0,4].
Verification, open-set identification, closed-set search, and even graph-based clustering all consume either
(i) cosine similarity scoresS ij=⟨ei,ej⟩, or (ii) their monotone surrogate∥e i−ej∥2
2. For verification (1:1) a
single score Sijis compared with a threshold τto decide amatchvsnon-match. For closed-set identification
(1:N) the probe isknownto correspond to one of the nenrolled identities. Writing the noisy probe as
qi=ei+z(for some unknown index i), wherezis nuisance noise, the backend must compare it with every
gallery item, i.e., consume the entire i-th rows i= (Si1,...,Sin) =E ei, in order to return arg maxjSij.
For open-set identification the probe may stem from an unseen identity, no row ofSis predetermined. The
system instead forms the similarity vectors=E q ∈Rnand declares either the nearest neighbor or “no
match” according to an open-set rule. Clustering/multi-target tracking algorithms, such as spectral clustering,
single-linkage, or Louvain, typically require the full pairwise affinity matrixSitself or a monotone transform
such asexp(αS), to define the graph’s edge weights.
23

arXiv preprint, ScoreShield
C.3 Differential Privacy
Definition C.1(Record-level Adjacency).Datasets D,D′(or equivalently, their embedding matricesE ,E′)
areadjacent, denoted by D∼D′, if they differ in at most one record (and thus in one embedding), i.e.,
|{i∈[n] :x i̸=x′
i}|= 1.
RemarkC.2.Under a fixed backbone model ϕθ, adjacency implies that the embedding matricesEandE′
differ in at most one row. Therefore, there exists at most one index isuch thate j=e′
j,∀j̸=i, and by our
normalization assumptione i̸=e′
i∈Rdsatisfy∥e i∥2=∥e′
i∥2= 1.
Definition C.3(Output–space (Gram-matrix) Adjacency).Two collections with embeddingsE ,E′are
adjacent at radius∆ G>0if∥EE⊤−E′E′⊤∥F≤∆ G.
Definition C.4(( ε,δ)–DP).A (possibly randomized) mechanism M:E→Osatisfies(ε,δ)–DP if for all
measurableT ⊆Oand all adjacentE ∼E′,Pr[M(E)∈T]≤eεPr[M(E′)∈T] +δ, whereε >0and
δ∈(0,1)are privacy parameters.
Definition C.5( ℓ2-Sensitivity).For a function f:E→Rn, theℓ2-sensitivity is∆ f,2=supE∼E′∥f(E)−
f(E′)∥2. We use the notation∆for brevity.
Lemma C.6(Gaussian Mechanism).Let f:E→Rnhaveℓ2–sensitivity∆ f,2=supE∼E′∥f(E)−f(E′)∥2.
DefineM(E)=f(E) +w,w∼N/parenleftbig
0,σ2In/parenrightbig
,σ2≥cε,δ∆2
f,2withcε,δ:=2 log(2/δ)
ε2. ThenMsatisfies(ε,δ)-DP.
Lemma C.7(Post-processing).Let M:E→Obe a mechanism that satisfies( ε,δ)–differential privacy. For
any measurable function g:O→O′, the composed mechanism g◦M :E→O′also satisfies( ε,δ)–differential
privacy.
RemarkC.8.After adding Gaussian noise to the similarity scores to achieve( ε,δ)-DP, we project the
noisy output onto the closed convex set of valid similarity scores. Since projection is independent of the
private dataset, the post-processing lemma ensures that this step does not compromise the privacy guarantee
established by the perturbation.
Corollary C.9(Gaussian mechanism for matrix-valued outputs).Let f:E→Rn×nand define the Frobenius
sensitivity
∆f,F:= sup
E∼E′∥f(E)−f(E′)∥F.(21)
LetW∈Rn×nhave i.i.d. entriesW ij∼N(0,σ2), and define
M(E) :=f(E) +W.(22)
Ifσ2=cε,δ∆2
f,F(withcε,δ= 2 log(2/δ)/ε2), thenMis(ε,δ)–differentially private.
Proof.Equip Rn×nwith the Frobenius norm ∥·∥ Fand identify it with Rn2via the vectorization map vec,
which is a linear isometry: ∥A∥F=∥vec(A)∥2. Therefore vec◦fhasℓ2-sensitivity∆ f,F. Moreover, since
Wiji.i.d.∼ N (0,σ2), we have vec(W)∼N (0,σ2In2). Applying Lemma C.6 to the Rn2-valued mechanism
vec(M(E)) = vec(f(E)) + vec(W)yields(ε,δ)–DP forM.
Corollary C.10(Symmetric Gaussian noise via averaging).In the setting of Corollary C.9, assume f(E)∈Sn
for allEand define
Msym(E):=f(E) +1
2(W+W⊤).(23)
Ifσ2=cε,δ∆2
f,F, thenMsymis(ε,δ)–DP. Moreover, writingG :=1
2(W+W⊤), the collection of upper-
triangular entries {Gii:i∈[n]}∪{Gij: 1≤i<j≤n} is mutually independent with Gii∼N(0,σ2)and
Gij∼N(0,σ2/2)fori<j.
Proof.Defineg:Rn×n→Snbyg(A):=1
2(A+A⊤). LetM(E):=f(E) +Wwith i.i.d. Wij∼N(0,σ2). By
Corollary C.9, ifσ2=cε,δ∆2
f,FthenMis(ε,δ)-DP. Sincef(E)∈Sn, we haveg(f(E)) =f(E)and hence
(g◦M)(E) =g(f(E) +W) =f(E) +g(W) =f(E) +1
2(W+W⊤) =M sym(E).(24)
24

arXiv preprint, ScoreShield
ThusM symis a deterministic post-processing ofM, and is(ε,δ)–DP by Lemma C.7.
For the distributional claim, note that Gii=Wii∼N(0,σ2). Fori<j,Gij=1
2(Wij+Wji)is the average
of two independent N(0,σ2)variables, hence Gij∼N(0,σ2/2). Independence across the upper-triangular
collection follows because each Gijdepends only on the disjoint set {Wij,Wji}(and eachGiionly onWii),
and the entries ofWare mutually independent.
RemarkC.11 (Orthonormal-basis view).Let {Bk}N
k=1be any Frobenius-orthonormal basis of Sn(e.g.,eie⊤
i
and(eie⊤
j+eje⊤
i)/√
2fori<j). Then/parenleftbig
⟨G,Bk⟩F/parenrightbigN
k=1are i.i.d.N(0,σ2).
Theorem C.12(Analytic Gaussian Calibration [3]).Letε>0,δ∈(0,1), and writeρ := ∆/σ. Define
δAG(ε,ρ) = Φ/parenleftig
−ε
ρ+ρ
2/parenrightig
−eεΦ/parenleftig
−ε
ρ−ρ
2/parenrightig
,(25)
whereΦis the standard normal cdf. ThenM σis(ε,δ)–DP if and only if
δ≥δ AG(ε,∆/σ).(26)
Equivalently, the minimal noise is obtained by the uniqueρ⋆>0solvingδ AG(ε,ρ⋆) =δ, with
σ⋆=∆
ρ⋆.(27)
Moreover, any measurable post–processing g(e.g., projection onto[ −1,1]n) preserves( ε,δ)by DP post–processing.
Proof.A proof is beyond the scope of this paper; see [3] for a complete derivation and proof.
Identity–Adjacency.We used the classical one–row/column notionE ∼imgE′⇐⇒ ∃i∈ [n] :ei̸=
e′
i,ej=e′
j∀j̸=i. With unit–norm embeddings this gives∆query
img= 2for the probe vector fquery(E,q) =Eq.
Now suppose each image (embedding) carries an identity label yi∈{1,...,K}. LetIk={i:yi=k}for the
index set of identity kand cardinality gk:=|Ik|. We sayE∼idE′iff there exists exactly one identity k⋆such
thatei̸=e′
i⇐⇒i∈I k⋆. We say
E∼ idE′⇐⇒ ∃k⋆:ei̸=e′
i⇐⇒i∈I k⋆.(28)
Thus all embeddings belonging to at most one identity k⋆may change (any or all of them can be added,
removed, replaced); every other identity’s rows stay fixed.
Let aggregate each identity to a unit–norm centroid
ck:=1
gk/summationdisplay
i∈Ikei,∥ck∥2≤1,(29)
and publish theKdimensional vector
˜fquery(E,q) =/parenleftbig
⟨c1,q⟩,...,⟨c K,q⟩/parenrightbig⊤∈RK.(30)
In this case, only the centroidk⋆may move
/vextenddouble/vextenddouble/tildewidefquery(E,q)−/tildewidefquery(E′,q)/vextenddouble/vextenddouble
2=|⟨ck⋆−c′
k⋆,q⟩|≤2.(31)
Hence∆centroid
id = 2. That is we have the same constant as image-adjacency definition.
Publishing the originaln-vectorf query(E,q) =Equnder identity-adjacency changes up tog k⋆coordinates:
∆query
id= 2√gk⋆>2.(32)
Sensitivity grows only with/radicalbig
|Ik⋆|, but is larger than the centroid case.
Therefore, identity-level privacy is free (∆ = 2)iffthe mechanism aggregates all images of a person into one
coordinate before noise is added. Otherwise the cost is the factor√gk⋆.
25

arXiv preprint, ScoreShield
C.4 Convex Geometry
Definition C.13(Euclidean Projection Onto a Closed Convex Set).Let m≥ 1and letC⊂Rmbe non–empty,
closed and convex. The Euclidean projection operatorprojC:Rm→Cis defined for everys∈Rmby
projC(s):= arg min
y∈C∥s−y∥ 2.(33)
Definition C.14(Tangent Cone).Let C⊂Rmbe non-empty, closed, and convex and fix a points ∈C. The
tangent cone toCatsis
Ts(C):=cl/braceleftig
λ(y−s) :λ≥0,y∈C/bracerightig
⊂Rm,(34)
where “ cl” denotes the Euclidean closure. Geometrically,T s(C)contains all velocity directions of feasible
curves that start atsand remain insideC.
Definition C.15(Gaussian Width of a Cone).LetK⊂Rmbe a non–empty, closed cone (i.e.,λK=Kfor
allλ≥0). Its Gaussian width is
GW(K) :=Ew∼N(0,Im)/bracketleftbigg
sup
s∈K∩Sm−1⟨w,s⟩/bracketrightbigg
.(35)
Definition C.16(Gaussian Complexity of a Bounded Set).Let C⊂Rmbe bounded. Its Gaussian complexity
is
GC(C) :=Ew∼N(0,Im)/bracketleftbigg
sup
s∈C⟨w,s⟩/bracketrightbigg
.(36)
IfCis unbounded we set GC(C):= +∞by convention. When Cis itself a cone intersected with the unit
sphere,GC(C)coincides with the Gaussian width of that cone, i.e.,GC(C) =GW/parenleftbig
cone(C)/parenrightbig
.
RemarkC.17 (Statistical Dimension vs. Gaussian Width).For a closed convex cone K⊂Rm, define its
statistical dimension by
δ(K) :=E/bracketleftbig
∥projK(z)∥2
2/bracketrightbig
,z∼N(0,I m).(37)
Thenδ(K)is tightly comparable to the Gaussian width (see Lemma G.8 for a proof):
GW(K)2≤δ(K)≤GW(K)2+ 1.(38)
Therefore any risk bound stated in terms of δ(Ts(C))can equivalently be expressed using GW(Ts(C))(or
GC(T s(C)∩Sm−1)).
RemarkC.18.In the convex-geometry definitions above, mdenotes the ambient Euclidean dimension. In
our applications: (i) for score vectorss ∈Rnwe havem=nand the sphere is Sn−1; (ii) for Gram matrices
S∈Snwe identify SnwithRNunder the Frobenius inner product, where N=dim(Sn) =n(n+ 1)/2, and
the sphere isSN−1.
RemarkC.19 (Matrix Gaussian Complexity under Frobenius Geometry).Definition C.16 is stated for subsets
ofRm. WhenC⊆Snis a set of symmetric matrices, we view Snas a Euclidean space with Frobenius inner
product⟨A,B⟩:=Tr(A⊤B)and dimension N=dim(Sn) =n(n+ 1)/2. Equivalently, one may identify Sn
withRNvia any fixed linear isometry (e.g., vectorization of the upper triangle with the appropriate√
2
scaling on off-diagonal entries). Accordingly, forC⊆Snwe write
GC(C) = E
W∼N(0,I n×n)/bracketleftig
sup
S∈C⟨W,S⟩/bracketrightig
,(39)
whereWis a standard Gaussian in this ambient Euclidean space.
Lemma C.20(Symmetrization Invariance of Matrix Gaussian Complexity).LetW ∈Rn×nhave i.i.d.
N(0,1)entries. For anyC⊆Sn,
GC(C) = E
W∼N(0,I n×n)/bracketleftig
sup
S∈C⟨W,S⟩/bracketrightig
= E
W∼N(0,I n×n)/bracketleftig
sup
S∈C/angbracketleftigW+W⊤
2,S/angbracketrightig/bracketrightig
.(40)
Moreover,G := (W+W⊤)/2satisfiesG ii∼N(0,1)andG ij∼N(0,1
2)fori<j.
26

arXiv preprint, ScoreShield
Proof.For any symmetricS, ⟨W,S⟩=Tr/parenleftbig
W⊤S/parenrightbig
=Tr/parenleftig
W+W⊤
2S/parenrightig
. The distributional claims follow by
direct computation.
Lemma C.21(Gaussian Complexity of Cquery).LetCquery :={s∈Rn:|si|≤1,∀i∈ [n]}= [−1,1]nand
z∼N(0,I n). Consider the Gaussian complexity definition C.16. Then
GC(C query) =E/bracketleftig
sup
x∈[−1,1]n⟨z,x⟩/bracketrightig
=E/bracketleftign/summationdisplay
i=1|zi|/bracketrightig
=n/radicalbigg
2
π= Θ(n).(41)
Proof.Fixz∈Rn. For anyx∈[−1,1]n,
⟨z,x⟩=n/summationdisplay
i=1zixi≤n/summationdisplay
i=1|zi||xi|≤n/summationdisplay
i=1|zi|.(42)
Equality is achieved by choosingx i= sign(zi)(with any value in[−1,1]whenz i= 0). Hence
sup
x∈[−1,1]n⟨z,x⟩=n/summationdisplay
i=1|zi|.(43)
Taking expectations gives the first two equalities in Eq. 41. Since the coordinates ofzare i.i.d. N(0,1)and
E|Z|=/radicalbig
2/πforZ∼N(0,1),
E/bracketleftign/summationdisplay
i=1|zi|/bracketrightig
=n/summationdisplay
i=1E|zi|=n/radicalbigg
2
π.(44)
Lemma C.22(Elliptope Equivalence).Let En:={S∈Sn:S⪰0,diag (S) =1}. Then everyS∈Ensatisfies
|Sij|≤1for alli̸=j. Consequently,C coll=/braceleftbig
S∈Rn×n:S⪰0,diag(S) =1,|S ij|≤1 (i̸=j)/bracerightbig
=En.
Proof.FixS∈En. SinceS⪰0, there exist vectors {ei}n
i=1such thatSij=⟨ei,ej⟩andSii=∥ei∥2
2. Because
diag(S) =1, we have∥e i∥2= 1for alli. Thus fori̸=j,|S ij|=|⟨ei,ej⟩|≤∥ei∥2∥ej∥2= 1.
Lemma C.23(Gaussian Complexity of Ccoll).LetEn:={S∈Sn:S⪰0,diag (S) =1}. There exist
universal constants0<c≤C <∞such that
cn3/2≤GC(En)≤Cn3/2.(45)
Equivalently,GC(C coll) = Θ(n3/2).
Proof.By Lemma C.20, withG= (W+W⊤)/2,GC(En) =E/bracketleftig
supS∈En⟨G,S⟩/bracketrightig
, where the expectation is over
the randomness ofW(equivalently,G).
Upper bound:ForS∈E n,Tr(S) =n, hence
sup
S∈En⟨G,S⟩≤sup
S⪰0
Tr(S)=n⟨G,S⟩=nλ max(G),(46)
where the equality holds because the right-hand side is achieved byS= nvv⊤for any unit top-eigenvectorv
ofG. Therefore, taking expectations gives
GC(En)≤n E[λmax(G)].(47)
Next, note thatGhas independent Gaussian entries (up to symmetry) with Gii∼N(0,1)andGij∼N(0,1
2)
fori<j. Standard spectral-norm bounds for such Wigner matrices imply E/bracketleftbig
λmax(G)/bracketrightbig
≤E/bracketleftbig
∥G∥op/bracketrightbig
≤C 0√n
for a universal constantC 0<∞, and therefore Eq. 47 yieldsGC(E n)≤Cn3/2.
27

arXiv preprint, ScoreShield
Lower bound:For anys ∈{± 1}n, the rank-one matrixS=ss⊤satisfiesS⪰0and diag(S) =1, hence
ss⊤∈En. Thus
sup
S∈En⟨G,S⟩≥max
s∈{±1}n⟨G,ss⊤⟩= max
s∈{±1}ns⊤Gs.(48)
Define the centered Gaussian processX s:=s⊤Gsindexed bys∈{±1}n.
We first choose a subset T⊆{± 1}nwhose pairwise Hamming distances are bounded both from below and
from above. Specifically, there exists a universal constantc 0>0and a setT⊆{±1}nsuch that
|T|≥exp(c 0n),n
4≤dH(s,t)≤3n
4,∀s̸=t∈T.(49)
To see this, draw Mindependent codewordss(1),...,s(M)uniformly from{±1}n. For any fixed pair a̸=b,
the Hamming distance dH(s(a),s(b))has distribution Binomial (n,1/2). By Chernoff’s bound, for a universal
constantc 1>0,
Pr/bracketleftbigg
dH(s(a),s(b))<n
4ord H(s(a),s(b))>3n
4/bracketrightbigg
≤2e−c1n.(50)
TakingM=⌊ec0n⌋with2c 0<c1, the union bound gives
Pr/bracketleftig
∃a<b:d H(s(a),s(b))/∈[n/4,3n/4]/bracketrightig
≤M22e−c1n<1(51)
for all sufficiently large n. Hence a deterministic set Tsatisfying Eq. 49 exists. In particular, log|T| = Ω(n);
finite values ofncan be absorbed into the universal constants.
RecallGis symmetric with independent entries {Gij:i≤j}, where Var(Gii) = 1and Var(Gij) = 1/2for
i<j. For anys∈{±1}n, we have
Xs=n/summationdisplay
i=1Giis2
i+ 2/summationdisplay
1≤i<j≤nGijsisj=n/summationdisplay
i=1Gii+ 2/summationdisplay
i<jGijsisj.(52)
Hence the diagonal term cancels in differences, and fors ,t∈{± 1}n, we haveXs−Xt= 2/summationtext
i<jGij/parenleftbig
sisj−titj/parenrightbig
.
Using independence andVar(G ij) = 1/2fori<j,
E/bracketleftbig
(Xs−Xt)2/bracketrightbig
= 4/summationdisplay
i<jVar(Gij) (sisj−titj)2(53a)
= 4·1
2/summationdisplay
i<j(sisj−titj)2= 2/summationdisplay
i<j(sisj−titj)2.(53b)
Now(sisj−titj)∈{0,±2}, so(sisj−titj)2= 4·1{sisj̸=titj}. LetD={i:si̸=ti}withd=|D|=dH(s,t).
Thensisj̸=titjiff exactly one of {i,j}lies inD, so the number of such pairs is |{(i,j) :i < j, sisj̸=
titj}|=d(n−d ). Therefore E/bracketleftbig
(Xs−Xt)2/bracketrightbig
= 8d(n−d ). For distincts ,t∈T, Eq. 49 gives d∈[n/4,3n/4].
Consequently,d(n−d)≥n
4·3n
4, and Eq. 53 yields
E/bracketleftbig
(Xs−Xt)2/bracketrightbig
≥8·n
4·3n
4=3
2n2.(54)
Thus the canonical metric [1]d X(s,t) :=/radicalbig
E[(X s−Xt)2]satisfies
inf
s̸=t∈TdX(s,t)≥/radicalbigg
3
2n.(55)
Sudakov’s minoration inequality [13, 38] for Gaussian processes yields
E/bracketleftig
max
s∈TXs/bracketrightig
≥c/parenleftig
inf
s̸=t∈TdX(s,t)/parenrightig/radicalbig
log|T| ≥c′n√n=c′n3/2,(56)
for universal constantsc,c′>0(usinglog|T|= Ω(n)). Sincemax s∈{±1}nXs≥max s∈TXs,
E/bracketleftig
max
s∈{±1}ns⊤Gs/bracketrightig
≥c′n3/2,(57)
and thereforeGC(E n)≥c′n3/2.
28

arXiv preprint, ScoreShield
Lemma C.24(Rank- rGram-manifold Tangent Space and its relation to the Elliptope Tangent Cone).
LetE= [e⊤
1;...;e⊤
n]∈Rn×dhave unit-norm rows ∥ei∥2= 1, and letS=EE⊤∈Snbe the associated
(cosine) Gram matrix with rank r:=rank(S)≤min{n,d} . Consider the elliptope (correlation-matrix set)
En=/braceleftbig
Y∈Sn:Y⪰0,diag (Y) =1/bracerightbig
. Equivalently,Ccoll=Ensince|Yij|≤1fori̸=jis implied byY⪰0
anddiag(Y) =1. LetF∈Rn×rbe any rank factor such thatS=FF⊤, and writef⊤
ifor thei-th row ofF
(so∥fi∥2
2=Sii= 1). Define the row-orthogonality constraint set
M:=/braceleftig
∆∈Rn×r:⟨fi,∆i⟩= 0,∀i∈[n]/bracerightig
,(58)
and the associated linear image (the rank-rGram-manifold tangent space atS)
Tman(S):=/braceleftig
F∆⊤+ ∆F⊤: ∆∈M/bracerightig
⊆Sn.(59)
Then:
(i)T man(S)is a linear subspace and admits the representation
Tman(S) =/braceleftig
F∆⊤+ ∆F⊤: ∆∈Rn×r,⟨fi,∆i⟩= 0,∀i∈[n]/bracerightig
,(60)
with
dim/parenleftbig
Tman(S)/parenrightbig
=n(r−1)−r(r−1)
2.(61)
(ii) LetT S(En)denote the contingent tangent cone ofE natS. Then
Tman(S)⊆T S(En)⊆/braceleftbig
H∈Sn: diag(H) =0/bracerightbig
.(62)
Consequently, for the statistical dimensionδ(·),
δ/parenleftbig
TS(En)/parenrightbig
≥δ/parenleftbig
Tman(S)/parenrightbig
= dim/parenleftbig
Tman(S)/parenrightbig
=n(r−1)−r(r−1)
2.(63)
(iii) Ifr=n(equivalently,S≻0), thenSis an interior point of the PSD constraint and
TS(En) =/braceleftbig
H∈Sn: diag(H) =0/bracerightbig
, δ/parenleftbig
TS(En)/parenrightbig
=n(n−1)
2= Θ(n2).(64)
In particular, whenr=none hasT man(S) =T S(En).
Proof.Note that ifY⪰0and diag(Y) =1, then for all i̸=j,|Yij|≤/radicalbig
YiiYjj= 1by Cauchy–Schwarz for
PSD matrices. HenceC coll=En.
(i)Consider a smooth perturbation of the factorF(t) :=F+t∆with∆∈Rn×r. Then
F(t)F(t)⊤=S+t(F∆⊤+ ∆F⊤) +t2∆∆⊤,(65)
so the first-order variation ofSinduced by∆isH=F∆⊤+ ∆F⊤. Moreover, for eachi∈[n],
d
dt/vextendsingle/vextendsingle/vextendsingle
t=0∥fi(t)∥2
2=d
dt/vextendsingle/vextendsingle/vextendsingle
t=0∥fi+t∆i∥2
2= 2⟨fi,∆i⟩.(66)
Thus the unit-diagonal constraint diag(F(t)F(t)⊤) =1holds to first order if and only if ⟨fi,∆i⟩= 0for all
i, yielding Eq. 60 and showing that Tman(S)is a linear subspace. To compute its dimension, note that the
constraints⟨fi,∆i⟩= 0arenindependent row-wise linear constraints, each reducing the rdegrees of freedom
in∆iby one because∥f i∥2= 1. Hence
dim(M) =n(r−1).(67)
Define the linear mapL:M→SnbyL(∆) :=F∆⊤+ ∆F⊤, soIm(L) =T man(S). Its kernel is
ker(L) =/braceleftbig
FΩ:Ω⊤=−Ω/bracerightbig
.(68)
29

arXiv preprint, ScoreShield
Indeed, if∆ =F ΩwithΩ⊤=−Ω, thenL(∆) =F( Ω+Ω⊤)F⊤=0. Conversely, if L(∆) =0, decompose
∆ =FΩ+ ∆⊥where the columns of∆ ⊥lie in range (F)⊥. ThenF∆⊤
⊥+ ∆⊥F⊤=0forces∆⊥=0, and
F(Ω+Ω⊤)F⊤=0impliesΩ⊤=−ΩsinceFhas full column rank. Therefore
dim ker(L) = dim{Ω∈Rr×r:Ω⊤=−Ω}=r(r−1)/2,(69)
and rank–nullity gives Eq. 61.
(ii)Fix∆∈Mand letH :=F∆⊤+ ∆F⊤∈Tman(S). Define the PSD curve
X(t) := (F+t∆)(F+t∆)⊤⪰0.(70)
As above,X(t) =S+tH+O(t2)in Frobenius norm. Its diagonal satisfies, for eachi,
diag(X(t)) i=∥fi+t∆i∥2
2= 1 +t2∥∆i∥2
2,(71)
using⟨fi,∆i⟩= 0. Thusdiag(X(t))is entrywise positive for allt, and we may define the diagonal scaling
D(t) := Diag/parenleftbig
diag(X(t))/parenrightbig−1/2.(72)
Set the diagonally normalized curve
/tildewideX(t) :=D(t)X(t)D(t).(73)
Then/tildewideX(t)⪰0and diag(/tildewideX(t)) =1for all t, hence/tildewideX(t)∈Enfor allt. Moreover, sinceD( t) =I+O(t2)
entrywise andX( t) =S+tH+O(t2), we have/tildewideX(t) =S+tH+O(t2). By the definition of the contingent
tangent cone, this impliesH ∈TS(En), proving the left inclusion in Eq. 62. For the right inclusion in Eq. 62,
note that diag(Y) =1is an affine constraint onY; hence any feasible first-order velocity atSmust satisfy
diag(H) =0. Finally, Eq. 63 follows from (a) monotonicity of statistical dimension under set inclusion and
(b)δ(U) = dim(U)for any linear subspaceU.
(iii)Ifr=n, thenS≻0is an interior point of the PSD cone, soT S(Sn
+) =Sn. Intersecting with the affine
constraint diag(Y) =1yieldsT S(En) ={H∈Sn:diag(H) =0}. This is a linear subspace of dimension
n(n−1)/2, hence its statistical dimension equalsn(n−1)/2, establishing Eq. 64.
Lemma C.25(Rank-aware Upper Bound on δ/parenleftbig
TS(En)/parenrightbig
Under Gram-Smoothness).LetS ∈Enbe a correlation
matrix with rank r≤n, whereEn:={X∈Sn:X⪰0,diag (X) =1}. LetS=FF⊤for someF∈Rn×r
with rowsf⊤
isatisfying∥fi∥2
2=Sii= 1. Consider the rank- rGram-manifold tangent space in Eq. 60. Assume
that the Bouligand tangent cone of EnatScoincides with rank- rGram manifold tangent space (i.e., assume
local Gram-smoothness atS):
TS(En) =T man(S).(74)
Then
δ/parenleftbig
TS(En)/parenrightbig
=n(r−1)−r(r−1)
2≤nr.(75)
Proof.Under Eq. 74, we have δ(TS(En)) =δ(Tman(S)). By Lemma C.24(i), Tman(S)is a linear subspace with
dim/parenleftbig
Tman(S)/parenrightbig
=n(r−1)−r(r−1)
2.(76)
For any linear subspace U,δ(U) =dim(U), hence the first equality in Eq. 75 follows. For the upper bound,
note that
n(r−1)−r(r−1)
2= (r−1)(n−r
2)≤rn,(77)
which completes the proof.
RemarkC.26 (Scope of the rank-aware bound).Lemma C.25 is conditional on the local Gram-smoothness
assumption in Eq. 74. It is not a general upper bound on δ(TS(En))for every rank- rpoint of the elliptope.
At singular boundary points, the elliptope tangent cone can be strictly larger than the rank- rGram-manifold
tangent space. For example, at a rank-one extreme pointS=vv⊤withv∈{± 1}n,Tman(S) ={0}, while
TS(En)contains nontrivial zero-diagonal feasible directions. Thus the rank-aware risk bound used in our
analysis should be interpreted only on the local smooth stratum where Eq. 74 holds.
30

arXiv preprint, ScoreShield
C.5 Non-Expansive Operators: Clipping and Euclidean Projection
Lemma C.27(Firm Non -expansiveness of Euclidean Projections).Let C⊂Rnbe non-empty, closed and
convex and define the Euclidean projector
projC(s) = arg min
s′∈C∥s−s′∥2.(78)
Then for alls,s′∈Rnwe have
∥projC(s)−projC(s′)∥2
2≤/angbracketleftbig
projC(s)−projC(s′),s−s′/angbracketrightbig
≤ ∥s−s′∥2
2.(79)
Consequences:
(a) 1-Lipschitzness. Taking square-roots in Eq. 79 gives∥projC(s)−projC(s′)∥2≤∥s−s′∥2.
(b) Difference bound. For everyw∈Rn,∥projC(s+w)−projC(s)∥ 2≤∥w∥ 2.
(c) Second moment bound. Ifw∼N(0,σ2In)thenE∥projC(s+w)−projC(s)∥2
2≤nσ2.
Proof.Optimality of the projection implies
⟨s−projC(s),v−projC(s)⟩≤0,⟨s′−projC(s′),v−projC(s′)⟩≤0,∀v∈C.(80)
Choosev=projC(s′)in the first andv=projC(s)in the second. Adding the two gives
⟨s−projC(s),projC(s′)−projC(s)⟩+⟨s′−projC(s′),projC(s)−projC(s′)⟩ ≤0.(81)
Hence
⟨projC(s)−projC(s′),s−s′⟩ ≥ ∥projC(s)−projC(s′)∥2
2,(82)
which is the first inequality in Eq. 79. The second follows by Cauchy–Schwarz.
Part (a) is immediate; (b) is (a) withs′=s+w; (c) squares (b) and usesE∥w∥2
2=nσ2.
Lemma C.28(Euclidean Projection Onto the Hypercube).Let C= [−1,1]n. The Euclidean projector onto
Csatisfies/parenleftbig
projC(s)/parenrightbig
i= max{−1,min{1,s i}}, i= 1,...,n.(83)
Proof.SinceCis a Cartesian product of intervals, the minimization minx∈C∥s−x∥2
2=minx∈C/summationtextn
i=1(si−xi)2
separates across coordinates. Thus each xiis the minimizer of minx∈[−1,1] (si−x)2, which is the scalar clip of
sionto[−1,1].
Corollary C.29(Pythagorean Inequality for Euclidean Projections).Fixs∈Rn. For everyy∈C,
∥s−y∥2
2=∥s−projC(s)∥2
2+∥projC(s)−y∥2
2+ 2/angbracketleftbig
s−projC(s),projC(s)−y/angbracketrightbig
.(84)
Moreover,/angbracketleftbig
s−projC(s),projC(s)−y/angbracketrightbig
≥0.(85)
Consequently,
∥projC(s)−y∥2
2≤∥s−y∥2
2−∥s−projC(s)∥2
2,∀y∈C.(86)
Proof.The identity Eq. 84 is the elementary expansion∥a+b∥2
2=∥a∥2
2+∥b∥2
2+ 2⟨a,b⟩with
a:=s−projC(s),b :=projC(s)−y.(87)
Also,a+b=s−y. The variational inequality for Euclidean projections onto a closed convex set gives
/angbracketleftbig
s−projC(s),v−projC(s)/angbracketrightbig
≤0,∀v∈C.(88)
31

arXiv preprint, ScoreShield
Choosingv=yin Eq. 88 gives/angbracketleftbig
s−projC(s),y−projC(s)/angbracketrightbig
≤0,(89)
or equivalently Eq. 85. Substituting Eq. 85 into Eq. 84 gives
∥s−y∥2
2≥∥s−projC(s)∥2
2+∥projC(s)−y∥2
2,(90)
which is equivalent to Eq. 86.
32

arXiv preprint, ScoreShield
D Regime (iii): Per-record Similarity Score Vector Release
Operational Scenario.In closed-set1 :Nidentification at e-gates or corporate turnstiles, duplicate-
enrollment audits in civil-ID databases, or when a watch-list entry must be shipped to an air-gapped or mobile
device, the back-end needs theentire similarity profileof one enrolled identity. Fix a public index i∈[n]and
define
fref,i(E):=Eei= (e⊤
1ei,...,e⊤
nei)∈[−1,1]n,(91)
where all embeddings satisfy ∥ej∥2= 1. Publishing fref,i(E)once lets downstream systems run top- ksearches,
threshold checks, or quality audits without further access to the raw embeddings. We drop the subscript i
when the reference index is clear.
Threat Model.We consider one-shot, non-interactive release of the i-th similarity rowr= fref,i(E) =
Eei∈[−1,1]nwithri= 1. We consider central( ε,δ)-DP under record-level adjacency where identity imay
change, so the n−1off-diagonal similarities involving identity imay change while the diagonal entry remains
equal to one. The adversary sees only the noisy vector /hatwider, knows the mechanism and all hyperparameters
(ε,δ,σ,n,i ), and is computationally unbounded. We consider: (i)no side informationaboutE; and (ii)full
gallery known, enabling linear subspace denoising; the diagonal constraint is public ( ri= 1). See App. F.3
and App. F.4.
Sensitivity.LetE′differ fromEonly in row i, replacinge ibye′
i. Write δ:=ei−e′
i,∥δ∥2≤2. The
neighboring outputs are fref,i(E) =Ee i,fref,i(E′) =E′e′
i. Forj̸=i, thej-th component of their difference is
e⊤
jei−e⊤
je′
i=e⊤
jδ. For the diagonal component,
/parenleftbig
fref,i(E)/parenrightbig
i−/parenleftbig
fref,i(E′)/parenrightbig
i=e⊤
iei−e′
i⊤e′
i= 1−1 = 0.(92)
Therefore,
fref,i(E)−f ref,i(E′) = (e⊤
1δ,...,e⊤
i−1δ,0,e⊤
i+1δ,...,e⊤
nδ).(93)
Hence
∥fref,i(E)−f ref,i(E′)∥2
2=/summationdisplay
j̸=i(e⊤
jδ)2≤/summationdisplay
j̸=i∥ej∥2
2∥δ∥2
2≤4(n−1).(94)
Thus the globalℓ 2-sensitivity satisfies
∆f,2:= ∆ ref≤2√
n−1.(95)
This bound is tight in the worst case, so∆ ref= 2√n−1.
Adversary Gain & Risk Scaling.With∆ ref= 2√n−1, the Gaussian mechanism uses per–coordinate
noiseσ≥∆ ref√cε,δ), withcε,δ= 2 log(2/δ)/ε2, henceσ= Θ(√n). For any off–diagonal entryj̸=i,
SNRj:=|e⊤
iej|
σ≤1
σ≤1
2/radicalbig
(n−1)cε,δ=O/parenleftigε/radicalbig
nlog(1/δ)/parenrightig
.(96)
That is each off-diagonal coordinate is perturbed by N(0,σ2)with per–coordinate SNR at most O(1/√n)
(since|e⊤
iej|≤1). IfEis unknown to the attacker, a naïve entrywise estimator would then incur per–entry
MSEσ2= Θ(n).
Jointly optimal denoising under auxiliary knowledge.GivenE, the clean row lies in the rank- rsubspace
col(E)⊆Rn. Letr′
i=ri+widenote the noisy row. The affine-unbiased oracle subspace estimator is
/hatwider⋆
i=Pr′
iwithP :=E(E⊤E)†E⊤∈Rn×n. Its error law is /hatwider⋆
i−ri∼N (0,σ2P), so thej-th entry has
MSEj=σ2Pjj=σ2ℓj, whereℓj∈[0,1]are statistical leverage scores of column j, with/summationtextn
j=1ℓj=tr(P) =r.
Hence, noting thatσ2= Θ(n), the average per–entry MSE equals
1
nn/summationdisplay
j=1MSEj=σ2
ntr(P) =σ2r
n= Θ(r),(97)
33

arXiv preprint, ScoreShield
which is constant in nfor fixed embedding dimension d(full rankr=d), while at mostO(r)high–leverage
indices can reach the naïveΘ( n)level. Note that each MSEj=σ2ℓjcan be as large as σ2= Θ(n)ifℓj≈1(a
high-leverage index), but only O(r)entries can be large because/summationtext
jℓj=r. In particular, the median per-entry
MSE isO(σ2r/n) = Θ(r). If face embeddings ofEare approximately isotropic, which is typical in practice
for largenwith unit-norm embeddings, then ℓj≈r/n,∀j (concentration of leverage scores). In that case,
MSEj≈σ2r
n= Θ(r), uniformly over j, hence, again independent of nfor fixedr. Therefore, given knowledge
of embeddings datasetE, the attacker can denoise jointly and recover the whole row to constant per-entry
accuracy (scaling with r, notn). The naïveΘ( n)per-entry MSE only applies if the attacker ignores structure.
The indistinguishability guarantee for membership inference remains governed by(ε,δ)independently ofn.
Lemma D.1(Orthogonal subspace denoising).Letr′=r+w,w∼N (0,σ2Im), where the unknown
deterministic vector satisfiesr ∈Sfor a known linear subspace S⊆Rm. LetPdenote the orthogonal projector
ontoS, and define/hatwider⋆:=Pr′. Then, for every realizationr′,/hatwider⋆is the unique solution of minz∈S∥z−r′∥2
2
. Moreover, /hatwider⋆−r=Pw, and therefore E∥/hatwider⋆−r∥2
2=σ2tr(P) =σ2dim(S). For each coordinate j,
E/bracketleftbig
(/hatwider⋆
j−rj)2/bracketrightbig
=σ2Pjj. Among all affine unbiased estimators /hatwider=Ar′+bthat take values in Sand satisfy
Er[/hatwider] =rfor everyr∈S, /hatwider⋆=Pr′is the unique minimizer of the mean squared error.
Proof.SinceSis a closed linear subspace of Rm, everyr′∈Rmadmits the orthogonal decomposition
r′=Pr′+(I−P)r′, wherePr′∈Sand(I−P)r′∈S⊥. Thus, foranyz∈S,∥z−r′∥2
2=∥z−Pr′∥2
2+∥(I−P)r′∥2
2.
The second term is independent ofz, and the first term is uniquely minimized atz=Pr′. Hence/hatwider⋆=Pr′is
the unique Euclidean projection ofr′ontoS.
Sincer∈S, we havePr=r. Therefore, /hatwider⋆−r=P(r+w)−r=Pw. Consequently,
E∥/hatwider⋆−r∥2
2=E∥Pw∥2
2=σ2tr(P2) =σ2tr(P) =σ2dim(S),(98)
where we usedP2=Pand tr(P) = dim(S). Also, Cov(Pw) =σ2P, which gives E/bracketleftbig
(/hatwider⋆
j−rj)2/bracketrightbig
=σ2Pjj.
It remains to prove the affine-unbiased optimality statement. Let /hatwider=Ar′+bbe affine, take values in S,
and be unbiased for everyr ∈S. Unbiasedness atr=0givesb=0. Unbiasedness for allr ∈Sgives
Ar=r,∀r∈S, equivalentlyAP=P. Since the estimator takes values in S, we also havePA=A.
HenceA=AP+A(I −P) =P+A(I−P). The two terms are orthogonal in Frobenius inner product,
so∥A∥2
F=∥P∥2
F+∥A(I−P)∥2
F≥∥P∥2
F, with equality if and only ifA(I −P) =0, i.e.,A=P. For any
such affine unbiased estimator, E∥/hatwider−r∥2
2=σ2∥A∥2
F. Therefore the unique minimizer isA=P, which gives
/hatwider⋆=Pr′.
Algorithm 4DP Reference-Identity Similarity Row Release (Regime (iii))
Input:E∈Rn×d,e∈Rd,ε>0,δ∈(0,1)
Output:/hatwides∈Rn
Computes=E e
DP noise:setσ2
ε,δ←cε,δ∆2
Samplew∼N/parenleftig
0, σ2
ε,δIn/parenrightig
Computes′=s+w
Compute/hatwides=projCref(s′), where/parenleftbig
proj[−1,1]n(s′)/parenrightbig
i= max(−1,min(1,s′
i))
Return/hatwides
We do not analyze regime (iii) further and focus on the more general regime (ii) (Sec. 3.2).
34

arXiv preprint, ScoreShield
E Output–Space Adjacency for DP Face-Recognition
This appendix formalizes anoutput–space(Gram-level) adjacency as an alternative to theimage-adjacency
used in our framework4. We show what privacy it delivers, how to calibrate the adjacency radius∆to
operational threat models, and how it compares, case by case, to our record–level adjacency guarantees and
to the naïve Gaussian mechanism.
Setting and Notation.Let D={x1,...,xn}be the enrollment (collection) set, fθa fixed backbone,
andE= [e⊤
1,...,e⊤
n]⊤∈Rn×dthe unit-normalized embeddings ( ∥ei∥2= 1). LetS=EE⊤∈Rn×nbe the
cosine Gram. Define the feasible set Ccoll:=/braceleftbig
X∈Rn×n:X⪰0, Xii= 1,|Xij|≤1 (i̸=j)/bracerightbig
. We write
cε,δ:= 2log(2/δ)/ε2. In our paper adjacency wasrecord-adjacency: D∼D′iff the embeddingsE ,E′differ in
at most one row.
Definition E.1(Output–Space (Gram-Matrix) Adjacency).Two collections with embeddingsE ,E′are
adjacent at radius∆ G>0if
∥EE⊤−E′E′⊤∥F≤∆ G.(99)
A randomized mechanism MmappingEto an output in Rn×nis(ε,δ)-DP at Gram radius∆ Gif for allE,E′
satisfying Eq. 99 and all measurable setsA,Pr[M(E)∈A]≤eεPr[M(E′)∈A] +δ.
RemarkE.2.By the Gaussian mechanism on the Euclidean space (Sn,∥·∥F), adding i.i.d. Gaussian noise
calibrated to Frobenius sensitivity∆ Gyields(ε,δ)-DP. We therefore set σ=∆ G/radicalbig
2 log(2/δ)/ε (equivalently,
σ2=cε,δ∆2
G). Post-processing (projection) preserves DP.
ScoreShield under∆ G-Gram Adjacency.ScoreShield performsperturb–then–project:
/hatwideS=projCcoll/parenleftbig
S+1
2(W+W⊤)/parenrightbig
, Wiji.i.d.∼ N(0,σ2), σ2=cε,δ∆2.(100)
Semantic Appropriateness of∆ G-Gram DP.∆ G-Gram DP is natural for non-interactive analytics
whose object is thepairwise similarity structure(e.g., clustering, link-mining, bias auditing). This provides
(ε,δ)-DP for any two collections whose Gram matrices are within Frobenius distance∆ G, i.e.,∥S−S′∥F≤∆G.
It is not a person-/image-level guarantee unless∆ Gis chosen to upper bound the Gram change induced by
any single-image replacement (see Sec. E.2).
E.1 Calibrating∆ Gto Operational Scenarios
We provide simple sufficient calibrations with explicit n,k,mdependence, linking∆ Gto interpretable threat
parameters.
Lemma E.3(Sparse pairwise perturbations).Suppose at most mundirected pairs( i,j),i<jchange and
for each such pair|∆S ij|=|S′
ij−Sij|≤τ. Then∥S′−S∥2
F≤2mτ2and hence a sufficient choice is
∆G≥τ√
2m.(101)
Proof.Each undirected pair contributes two symmetric entries, so∥S′−S∥2
F=/summationtext
i̸=j(∆Sij)2≤2mτ2.
Lemma E.4(At most kimages drift by at most ηinℓ2).Let∆E=E′−Ehave at most knonzero rows
and∥∆ei∥2≤ηfor each changed row. Then
∥S′−S∥ F≤2η√
nk+η2k,(102)
hence a sufficient choice is
∆G≥2η√
nk+η2k.(103)
4Note that one record corresponds to one image (equivalently, one embedding), so record-level replacement adjacency coincides
with image-level adjacency.
35

arXiv preprint, ScoreShield
Proof. S′−S=E∆E⊤+∆E E⊤+ ∆E∆E⊤. Using∥AB⊤∥F≤∥A∥2∥B∥F,∥E∥2≤∥E∥F=√n, and
∥∆E∥ F≤η√
kwe have
∥S′−S∥ F≤2∥E∥ 2∥∆E∥ F+∥∆E∥2
F≤2η√
nk+η2k,(104)
which yields Eq. 102. Note that for smallη, the linear term dominates.
Lemma E.5(Row-sparse single-image effect).If a single image changes but only mof its similarities move
by at mostτ, then∥S′−S∥ F≤τ√
2mand a sufficient choice is∆ G≥τ√
2m.
Proof.This is immediate from Lemma E.3 by taking the set of changed undirected pairs.
These bounds express the Gram-adjacency radius∆ Gas an explicit function of interpretable threat parameters
(either(m,τ)for sparse score changes or(k,η)fork-record embedding drift).
E.2 Bridging to Image Adjacency
Our main setup adopts image-adjacency :E,E′differ in a single row. In that case the Gram sensitivity is
exact:
∆F,img := sup
E∼E′∥EE⊤−E′E′⊤∥F= 2/radicalbig
2(n−1) = Θ(√n).(105)
Proposition E.6(Dominating image-adjacency with Gram-adjacency).If one wishes∆ G-Gram DP to imply
image-level DP, it is necessary and sufficient to take
∆G≥∆F,img = 2/radicalbig
2 (n−1).(106)
Consequently, the minimum Gaussian variance required to dominate image-adjacency is σ2
min=cε,δ∆2
F,img =
8cε,δ(n−1) = Θ(n), and any larger∆ Gyields proportionally larger variance.
Conversely, fixing∆ G=O(1)yields a distinct guarantee that does not cover arbitrary single-image changes.
It protects only changes that only pairs(E,E′)satisfying∥S−S′∥F≤∆ G(see Lemmas E.3–E.5).
36

arXiv preprint, ScoreShield
FNaïve Gaussian vs. ScoreShield Mechanism Under Different Release Regimes
We compare mechanisms for privatizing similarity statistics derived from unit-normalized face embeddings
E= [e⊤
1,...,e⊤
n]⊤∈Rn×dwith∥ei∥2= 1. We use the central model( ε,δ)-DP and calibrate the Gaussian
mechanism to the global ℓ2-sensitivity of the (vectorized) released statistic. We denote cε,δ:= 2log(2/δ)/ε2,
and choose the noise variance as σ2=cε,δ∆2(anyσ2≥cε,δ∆2is valid). Unless stated otherwise, all
asymptotics are asn→∞with(ε,δ)fixed. In particular,c ε,δ= 2 log(2/δ)/ε2is treated asΘ(1)inn.
F.1 ScoreShield Projection Risk Bounds
Global Projection Risk via Gaussian Complexity.Consider ScoreShield with isotropic Gaussian noise
in the ambient Euclidean space. A uniform (set-level) bound controls the squared projection error by the
Gaussian complexity of the feasible set as
E∥/hatwideS−S∥2
F≤CσGC(C),(107)
for a universal constant C > 0; see Lemma F.1.5This guarantee is global (uniform over allS ∈C), does not
rely on a small-noise regime, and depends only on the set geometry throughGC(C)(and boundedness ofC).
Lemma F.1(Global ScoreShield Bound via Gaussian Complexity).Let m≥ 1and letC⊂Rmbe nonempty,
closed, convex, and bounded. Fixs ∈Cand letw∼N (0,σ2Im). Consider ScoreShield projector /hatwides=
projC(s+w). Then
E∥/hatwides−s∥2
2≤2σGC(C−C)≤4σGC(C),(108)
where the expectation is overw, C−C :={u−v:u,v∈C}, and for any bounded A⊂Rm,GC(A):=
Ez∼N(0,Im)/bracketleftbig
supa∈A⟨z,a⟩/bracketrightbig
.
Proof.Let/hatwides=projC(s+w)and set∆ :=/hatwides−s. By optimality of Euclidean projection, for everyy∈C,
∥s+w−/hatwides∥2
2≤ ∥s+w−y∥2
2.(109)
Choosingy=s∈Cyields
∥s+w−/hatwides∥2
2≤ ∥w∥2
2.(110)
Expanding the left-hand side,
∥s+w−/hatwides∥2
2=∥(s−/hatwides) +w∥2
2=∥∆∥2
2+∥w∥2
2−2⟨w,∆⟩.(111)
Cancel∥w∥2
2from both sides to obtain the deterministic inequality
∥∆∥2
2≤2⟨w,∆⟩.(112)
Since/hatwides,s∈C, we have∆∈C−C, hence
⟨w,∆⟩ ≤sup
u∈C−C⟨w,u⟩.(113)
Combining with Eq. 112 and taking expectations gives
E∥/hatwides−s∥2
2≤2E/bracketleftig
sup
u∈C−C⟨w,u⟩/bracketrightig
.(114)
Writew=σzwithz∼N(0,I m). Then
E/bracketleftig
sup
u∈C−C⟨w,u⟩/bracketrightig
=σE/bracketleftig
sup
u∈C−C⟨z,u⟩/bracketrightig
=σGC(C−C),(115)
which together with Eq. 114 proves the first inequality in Eq. 108.
For the second inequality, for any fixedz,
sup
u∈C−C⟨z,u⟩= sup
a,b∈C⟨z,a−b⟩≤sup
a∈C⟨z,a⟩+ sup
b∈C⟨−z,b⟩.(116)
Taking expectations and using−zd=zyieldsGC(C−C)≤2GC(C).
5Usingσ2=cε,δ∆2givesσ=√cε,δ∆, henceE∥/hatwideS−S∥2
F≤C√cε,δ∆GC(C) =C√
2 log(2/δ)
ε∆GC(C).
37

arXiv preprint, ScoreShield
Corollary F.2(Global ScoreShield Bound for Gram Matrices).Let H:= (Rn×n,⟨·,·⟩F)with Frobenius norm
∥·∥F, and letC⊂Rn×nbe nonempty, closed, convex, and bounded. FixS ∈Cand letW∈Rn×nsatisfy
vec(W)∼N(0,σ2In2). Consider the ScoreShield projector /hatwideS=projC(S+W), where projCis the Frobenius
(metric) projection ontoC. Then
E∥/hatwideS−S∥2
F≤2σGC(C−C)≤4σGC(C),(117)
whereC−C :={U−V:U,V∈C}and, for any boundedA⊂Rn×n, we have
GC(A) :=E/bracketleftig
sup
A∈A⟨Z,A⟩F/bracketrightig
,vec(Z)∼N(0,I n2).(118)
Proof.Define the linear isometry vec:(Rn×n,∥·∥F)→/parenleftig
Rn2,∥·∥ 2/parenrightig
. Let/tildewideC:=vec(C)⊂Rn2and/tildewides:=vec(S).
Sincevecis an isometry andCis closed convex, we have
vec(projC(Y)) =proj/tildewideC(vec(Y)),∀Y∈Rn×n.(119)
Moreover, by assumption /tildewidew:= vec(W)∼N(0,σ2In2). Thereforevec( /hatwideS) =proj/tildewideC(/tildewides+/tildewidew).
Apply Lemma F.1 with m=n2to the convex set /tildewideC⊂Rn2. Finally, use∥/hatwideS−S∥F=∥vec(/hatwideS)−vec (S)∥2and
note that gaussian complexities are equal by the identity ⟨vec(Z),vec(A)⟩=⟨Z,A⟩F. This yields the stated
bound.
Projection Risk via Tangent Cones.We next state a point-dependent (instance-dependent) bound in
terms of the tangent cone geometry at the true point. Let m≥ 1and letC⊂Rmbe nonempty, closed, and
convex. Fixs∈Cand letw∼N(0,σ2Im). Let/hatwides=projC(s+w). Then the squared error is controlled by the
statistical dimension of the tangent coneT s(C):
E∥/hatwides−s∥2
2≤σ2δ(Ts(C)),(120)
as formalized in Lemma F.3. Unlike the global Gaussian-complexity bound, Eq. 120 is local and can be
substantially smaller whenT s(C)is low-dimensional (e.g., due to rank structure).
Lemma F.3(Local ScoreShield Bound via Tangent Cones).Let m≥ 1and letC⊂Rmbe nonempty, closed,
and convex. Fixs∈Cand letw∼N(0,σ2Im). Let/hatwides=projC(s+w). Then
E∥/hatwides−s∥2
2≤σ2δ(Ts(C)),(121)
whereT s(C)is the tangent cone ofCats(see Definition C.14) and
δ(K) :=E/bracketleftbig
∥projK(z)∥2
2/bracketrightbig
,z∼N(0,I m),(122)
denotes the statistical dimension of a closed convex coneK⊂Rm.
Proof.Let∆:=/hatwides−sand define the translated set D:=C−s={y−s:y∈C}. ThenDis closed and
convex,0∈D, and by translation invariance of Euclidean projection, ∆=projD(w). LetK:=T0(D). Since
Dis convex and contains0, its tangent cone at the origin is
K=T 0(D) =cl/parenleftbig
cone(D)/parenrightbig
,(123)
and henceD⊆K. Moreover,K=T s(C)by translation invariance of tangent cones.
We next prove the deterministic inequality
∥projD(w)∥ 2≤∥projK(w)∥ 2.(124)
Set∆=projD(w). Since0∈D, the variational inequality for Euclidean projection gives
⟨w−∆,0−∆⟩≤0.(125)
38

arXiv preprint, ScoreShield
Equivalently,
∥∆∥2
2≤⟨w,∆⟩.(126)
Because∆∈D⊆K, if∆̸=0, then∆/∥∆∥ 2∈K∩Sm−1. Therefore
⟨w,∆⟩≤∥∆∥ 2 sup
v∈K,∥v∥ 2≤1⟨w,v⟩.(127)
We now identify the support function ofK∩B 2. Let
K◦:={q∈Rm:⟨q,v⟩≤0,∀v∈K}(128)
denote the polar cone. By Moreau’s decomposition for closed convex cones [7, 31],
w=projK(w) +projK◦(w),⟨projK(w),projK◦(w)⟩= 0.(129)
SinceprojK◦(w)∈K◦, for everyv∈Kwith∥v∥ 2≤1we have
⟨w,v⟩=⟨projK(w),v⟩+⟨projK◦(w),v⟩≤⟨projK(w),v⟩≤∥projK(w)∥ 2.(130)
IfprojK(w)̸=0, equality is attained byv= projK(w)/∥projK(w)∥2. IfprojK(w) =0, the supremum is zero by
the preceding inequality. Hence
sup
v∈K,∥v∥ 2≤1⟨w,v⟩=∥projK(w)∥ 2.(131)
Combining Eqs. equation 126, equation 127, and equation 131 gives
∥∆∥2
2≤∥∆∥ 2∥projK(w)∥ 2.(132)
Thus
∥∆∥ 2≤∥projK(w)∥ 2,(133)
with the conclusion trivial when∆=0. Hence Eq. equation 124 holds, and
∥/hatwides−s∥2
2=∥∆∥2
2≤∥projK(w)∥2
2.(134)
Taking expectations overw∼N(0,σ2Im)gives
E∥/hatwides−s∥2
2≤E∥projK(w)∥2
2.(135)
Writingw= σzwithz∼N(0,Im)and using positive homogeneity of projection onto a cone, projK(σz) =
σprojK(z)forσ≥0, we obtain
E∥projK(w)∥2
2=σ2E∥projK(z)∥2
2=σ2δ(K).(136)
Finally, sinceK=T s(C), we get
E∥/hatwides−s∥2
2≤σ2δ/parenleftbig
Ts(C)/parenrightbig
.(137)
Corollary F.4(Local ScoreShield bound for matrix inputs under i.i.d. Gaussian noise).Let H:=
(Rn×n,⟨·,·⟩F)with Frobenius norm ∥·∥F, and letC⊂Rn×nbe nonempty, closed, and convex. FixS ∈Cand
letW∈Rn×nsatisfy vec(W)∼N(0,σ2In2). Consider the ScoreShield projector /hatwideS:=projC(S+W), where
projCdenotes the metric projection in Frobenius norm. Then
E∥/hatwideS−S∥2
F≤σ2δ(TS(C)),(138)
whereT S(C)is the tangent cone ofCatSinH, and for any closed convex coneK⊂Rn×n,
δ(K) =E/bracketleftbig
∥projK(Z)∥2
F/bracketrightbig
,vec(Z)∼N(0,I n2).(139)
39

arXiv preprint, ScoreShield
Proof.LetL:=vec: (Rn×n,∥·∥F)→(Rn2,∥·∥ 2)be the canonical linear isometry. Set /tildewideC:=L(C)⊂Rn2,
/tildewides:=L(S), and/tildewidew:=L(W). By assumption, /tildewidew∼N (0,σ2In2). SinceLis an isometry and Cis closed convex,
metric projections commute withL:
L(projC(Y)) =proj/tildewideC(L(Y)),∀Y∈Rn×n.(140)
Hence
L(/hatwideS) =proj/tildewideC(/tildewides+/tildewidew).(141)
Apply Lemma F.3 withm=n2to the convex set /tildewideC⊂Rn2:
E∥L(/hatwideS)−L(S)∥2
2≤σ2δ/parenleftig
T/tildewides(/tildewideC)/parenrightig
.(142)
Using∥L(/hatwideS)−L(S)∥2=∥/hatwideS−S∥F, it remains to relate tangent cones and statistical dimensions under L. By
the definitionT S(C) = cl cone(C−S)and linearity ofL,
L(TS(C)) =TL(S)(L(C)) =T/tildewides(/tildewideC).(143)
Moreover, forZwithvec(Z)∼N(0,I n2), we have
δ/parenleftig
T/tildewides(/tildewideC)/parenrightig
=E/bracketleftig/vextenddouble/vextenddoubleprojL(TS(C))(L(Z))/vextenddouble/vextenddouble2
2/bracketrightig
=E/bracketleftig/vextenddouble/vextenddoubleprojTS(C)(Z)/vextenddouble/vextenddouble2
F/bracketrightig
=δ(T S(C)),(144)
where the middle equality uses that Lis an isometry and projections commute with Lon closed convex cones.
Substituting completes the proof.
We now use these tools in our three regimes.
F.2 Regime (i): Query-to-Collection Similarity Score Vector
Given a fixed queryq ∈Rdwith∥q∥2= 1, the released vector iss= fquery(E,q) =Eq∈[−1,1]n.
Under record–level replacement adjacency onE), only the corresponding coordinate may change, and the
globalℓ2-sensitivity is constant∆ query= 2. We therefore calibrate the Gaussian mechanism with variance
σ2=cε,δ∆2
query= 4cε,δ.
Naïve Gaussian Mechanism.Releases′=s+w,w∼N(0,σ2In)withσ2=cε,δ∆2
query= 4cε,δ. Then we
have
E∥s′−s∥2
2=E∥w∥2
2=nσ2= 4cε,δn.(145)
ScoreShield Mechanism.Let /hatwides=projCquery(s+w). We have:
Local (Pointwise) Risk Bound via the Tangent Cone.For the constraint set Cquery= [−1,1]n, the tangent
cone atsis a product cone whose statistical dimension depends only on the active set. Let a(s)denote the
number of coordinates ofson the boundary {±1}. Thenδ(Ts((C))) =n−1
2a(s)(see Lemma G.9 for a proof),
and the local conic denoising bound (Lemma F.3) yields
E∥/hatwides−s∥2
2≤σ2/parenleftig
n−1
2a(s)/parenrightig
= 4cε,δ/parenleftig
n−1
2a(s)/parenrightig
.(146)
If no coordinate is active, a(s) = 0and δ(Ts) =n, i.e., the bound matches the naïve risk in Eq. 145. If
many coordinates are saturated (large a(s)), projection can reduce the bound by up to a factor1
2on those
coordinates. Since∆ queryis constant, the per-coordinate MSE isO(1)for both mechanisms.
Global Risk Bound via the Gaussian Complexity.Letz∼N(0,I n), then using Lemma C.21, we have
GC(C query) =E/bracketleftig
sup
x∈[−1,1]n⟨z,x⟩/bracketrightig
=n/radicalbigg
2
π.(147)
40

arXiv preprint, ScoreShield
Applying the global ScoreShield bound (Lemma F.1) gives the uniform estimate
E∥/hatwides−s∥2
2≤CσGC(C query) =Cσn/radicalbigg
2
π,(148)
for a universal constant C > 0. This bound is uniform ins ∈[−1,1]nbut does not exploit the local active-set
geometry captured by the tangent-cone bound in Eq. 146. Because GC(Cquery) = Θ(n), the global bound scales
asΘ(σn), whereas the local tangent-cone bound scales asΘ( σ2n). Sinceσis fixed by privacy calibration,
either inequality may be numerically smaller depending on the regime. In particular, for the common
high-privacy regime where σis not small, the two bounds are of comparable order up to constants, but only
the tangent-cone bound captures the reduction bya(s).
F.3 Regime (iii): Per-Record Similarity Score Vector
Fixi∈[n]and released the similarity vectorr= fref(E,ei) =Eei∈[−1,1]n, which satisfies ri= 1. Under
record-levelreplacementadjacency, onlytheoff-diagonalcoordinatescanchange, and∆ ref= 2√n−1 = Θ(√n).
See the exact derivation in Sec. D. Thereforeσ2=cε,δ∆2
ref= 4cε,δ(n−1).
Naïve Gaussian Mechanism.Releaser′=r+w,w∼N(0,σ2In),σ2=cε,δ∆2
ref= 4cε,δ(n−1). Then
E∥r′−r∥2
2=nσ2= 4cε,δn(n−1) = Θ/parenleftbig
cε,δn2/parenrightbig
.(149)
ScoreShield Mechanism.Let /hatwider=projCref(r+w). We have:
Local (Pointwise) Risk Bound via the Tangent Cone.Project onto Cref:={x∈[−1,1]n:xi= 1}. The
equality constraint fixes the ith (self) coordinate and removes one free direction. If a¬i(r)of the remaining
coordinates are on the boundary {±1}, thenδ(Tr(Cref)) = (n−1)−1
2a¬i(r), and therefore Lemma F.3 gives
E/bracketleftbig
∥/hatwider−r∥2
2/bracketrightbig
≤σ2/parenleftig
(n−1)−1
2a¬i(r)/parenrightig
= 4cε,δ(n−1)/parenleftig
(n−1)−1
2a¬i(r)/parenrightig
,(150)
wherea¬i(r)countsboundaryactivationsamongthecoordinates j̸=i. ThusScoreShieldyieldsdata-dependent
constant-factor improvement (up to roughly a factor1
2on activated coordinates) relative to the fully interior
case, while the overall scaling remains O(n2)becauseσ2= Θ(n). Equivalently, over the( n−1)free coordinates,
the average per-coordinate MSE scales asΘ( n)(reduced by up to a factor2on boundary-active coordinates)6,
and the fixed coordinate incurs zero error after projection due to hard constraintr i= 1.
Global Risk Bound via the Gaussian Complexity.The feasible set is the affine slice of the box Cref:=/braceleftbig
x∈
[−1,1]n:xi= 1/bracerightbig
. SinceCrefdiffers fromCqueryonly by fixing one coordinate, it follows immediately that
GC(C ref) = Θ(n)as in Eq. 147. Below we compute the exact constant.
Applying Lemma F.1 withC=C refyields the uniform bound
E∥/hatwider−r∥2
2≤2σGC/parenleftbig
Cref−C ref/parenrightbig
≤4σGC(C ref),/hatwider=projCref(r+w),w∼N(0,σ2In).(151)
We can computeGC(C ref)in closed form. Letz∼N(0,I n). Then
sup
x∈C ref⟨z,x⟩=z i·1 +/summationdisplay
j̸=isup
xj∈[−1,1]zjxj=zi+/summationdisplay
j̸=i|zj|.(152)
Taking expectation and usingE[z i] = 0andE|z j|=/radicalbig
2/πgives
GC(C ref) =E/bracketleftig
sup
x∈C ref⟨z,x⟩/bracketrightig
= (n−1)/radicalbigg
2
π= Θ(n).(153)
6Compared to the fully interior case a¬i(r) = 0, boundary-active coordinates reduce the tangent-cone statistical dimension by
1
2each, so the bound drops from σ2(n−1)toσ2((n−1)−a¬i(r)/2). In the extreme case a¬i(r) =n−1, this is a2×reduction,
while the overall order remainsΘ(n2)sinceσ2= Θ(n).
41

arXiv preprint, ScoreShield
Moreover,
Cref−C ref=/braceleftbig
u∈Rn:ui= 0,|uj|≤2∀j̸=i/bracerightbig
= 2/braceleftig
u∈Rn:ui= 0,|uj|≤1∀j̸=i/bracerightig
,(154)
hence by the scaling lawGC(aA) =aGC(A),
GC/parenleftbig
Cref−C ref/parenrightbig
= 2(n−1)/radicalbigg
2
π= Θ(n).(155)
Substituting Eq. 155 into Eq. 151 yields the explicit global bound
E∥/hatwider−r∥2
2≤2σ·2(n−1)/radicalbigg
2
π= 4σ(n−1)/radicalbigg
2
π=O(σn).(156)
Usingσ2= 4cε,δ(n−1), we haveσ= 2√cε,δ√n−1, so
E∥/hatwider−r∥2
2≤8/radicalbigg
2
π√cε,δ(n−1)3/2=O/parenleftbig√cε,δn3/2/parenrightbig
.(157)
This global bound is uniform overr ∈C refbut does not exploit the local active-set geometry captured by the
tangent-cone bound in Eq. 150. Since the local bound scales asΘ( σ2n) = Θ(cε,δn2)while the global bound
scales asΘ( σn) = Θ(√cε,δn3/2), either inequality can be numerically smaller depending on the privacy noise
level (viaσ) and the active set sizea ¬i(r).
Summary for Regimes (i) & (iii).Under the same record-level adjacency model and Gaussian calibration,
ScoreShield projection is non-expansive and therefore cannot increase the squared error relative to the
unprojected Gaussian release. Moreover, it can yield data-dependent constant-factor improvements governed
by the number of active (saturated) coordinates. In particular, the local tangent-cone bounds give:
Regime Naïve Mechanism MSE ScoreShield Mechanism MSE
Regime (i) (∆ query= 2)4c ε,δn≤4c ε,δ/parenleftig
n−1
2a(s)/parenrightig
Regime (iii) (∆ ref= 2√n−1)4c ε,δn(n−1)≤4c ε,δ(n−1)/parenleftig
(n−1)−1
2a¬i(r)/parenrightig
In regimes (i) and (iii), the feasible sets are (products of) intervals (with an additional equality constraint
in regime (iii)), so projection yields constant-factor improvements and does not change the exponent in the
leadingn-scaling under the local tangent-cone bound. In the following, we address the attacker’s reconstruction
error.
F.4 Attacker Reconstruction Error for Regime (i) & (iii)
ScoreShield release model.For any released score vector we uses′=s+w, w∼N (0,σ2In),
/hatwides=projC(s′). For regime (i),C=C query={s∈[−1,1]n}and the projection is coordinate-wise clipping
/parenleftbig
proj[−1,1]n(s′)/parenrightbig
j= max(−1,min(1,s′
j)).
For regime (iii),C=Cref:={s∈[−1,1]n:si= 1}and the projection onto Crefsets the reference coordinate
to one and clips the remaining coordinates:
/parenleftbig
projCref(s′)/parenrightbig
i= 1,/parenleftbig
projCref(s′)/parenrightbig
j= max(−1,min(1,s′
j)), j̸=i.
For the box-clipping coordinates, the scalar clipping map is1-Lipschitz, so E[(/hatwidesj−sj)2]≤σ2. For the fixed
reference coordinate in regime (iii),/hatwides i=si= 1, so the MSE is zero.
Adjacency and altered index set.Under record-level replacement,EandE′differ in exactly one row i.
LetIdenote the set of altered indices under adjacency. For regime (i) (query-to-collection), only coordinate
iofs=Eqmay change, hence I1:={i}. For regime (iii) (reference-to-gallery), the altered set is the
off-diagonalsI 2(i):={j̸=i}of sizen−1.
42

arXiv preprint, ScoreShield
Risk Functionals.For any indexj, define the coordinate MSE
MSEj:=E/bracketleftbig
(/hatwidesj−sj)2/bracketrightbig
,(158)
and for an altered index setIdefine the restricted risk
R(I) :=E/bracketleftbig
∥(/hatwides−s)I∥2
2/bracketrightbig
=/summationdisplay
j∈IMSEj.(159)
Auxiliary-information regimes.The reconstruction calculations below concern estimation error from
the noisy released score vector; they are separate from the differential-privacy guarantee. We distinguish
knowledge of the score-generating values from knowledge of the subspace in which the clean score vector lies.
If an attacker knows all quantities that determine the clean score vector, for example bothEand a public
queryqin regime (i), thens=Eqis computable exactly and the reconstruction MSE is zero. The linear
denoising bounds below are therefore stated for a subspace-knowledge model, in which the attacker knows the
relevant column space but not the latent coefficient vector generating the released scores.
We consider three regimes:
(K1)No side information:the attacker knows neitherEnor the relevant column spacecol(E).
(K2)Knows col(E), but not the latent score generator:the attacker knows the column space col(E),
equivalently the orthogonal projector
P:=E(E⊤E)†E⊤∈Rn×n,(160)
but does not know the vector that generates the clean score vector within this subspace. Let r:=rank(E).
(K3)KnowsE−i:the attacker knows all rows except the single differing row i, i.e.,E−i∈R(n−1)×d. Let
r−i:= rank(E−i)and let
P−i:=E−i(E⊤
−iE−i)†E⊤
−i∈R(n−1)×(n−1)(161)
be the orthogonal projector ontocol(E −i).
Baseline (coordinatewise upper bounds).Without using any structure beyond the released values, a
natural estimator fors jis/hatwidesj. By non-expansiveness,
MSEj≤σ2,∀j∈[n],(162)
with equality on coordinates that do not clip (in particular, exactly under the no-clipping Gaussian model
/hatwides=s′=s+w).
Linear denoising under subspace knowledge.Assume the no-clipping Gaussian model /hatwides=s′=s+w.
The following formulas apply when the clean score vector is an unknown deterministic vector in the known
subspace. They do not apply to the case where the attacker knows all score-generating inputs; in particular,
in regime (i), if bothEand the public queryqare known to the attacker, thens=Eqis known exactly and
the reconstruction MSE is zero.
(K2) Knows col(E), but not the latent generator.In the subspace-knowledge model, the clean vector is an
unknown element of col(E); for example, in regime (i),s=Eq ∈col (E)whenqis not disclosed to the attacker.
Under additive Gaussian noise, the affine-unbiased least-squares estimator is the orthogonal projection
/tildewidesLS=P/hatwides.(163)
Its error covariance is
Cov(/tildewidesLS−s) =σ2P,(164)
and hence
MSEj=σ2Pjj,1
nn/summationdisplay
j=1MSEj=σ2
ntr(P) =σ2r
n.(165)
43

arXiv preprint, ScoreShield
The coordinatewise risk is governed by the leverage scoreP jj∈[0,1].
(K3) KnowsE −i.In the subspace-knowledge model, the attacker can denoise the subvector indexed by j̸=i
using col(E−i), but cannot useE −ito denoise the altered coordinate i. A natural affine-unbiased estimator is
/tildewidesi=/hatwidesi,/tildewides−i=P−i/hatwides−i.(166)
Under no clipping, forj̸=i,
MSEj=σ2(P−i)jj,(167)
and/summationdisplay
j̸=iMSEj=σ2tr(P−i) =σ2r−i,1
n−1/summationdisplay
j̸=iMSEj=σ2r−i
n−1.(168)
The altered coordinate remains at the no-denoising level, MSEi=σ2. If, in regime (i), the queryqis public
andE−iis known, then the unchanged coordinatess −i=E−iqare known exactly; in that case the above
subspace-denoising bound is not the relevant reconstruction model for those coordinates.
Attacker’s reconstruction error for regime (i) (query vector).In regime (i), σ2= 4cε,δandI1={i}.
Therefore, the restricted risk equals the altered-coordinate risk, R(I1) = MSEi. Under the no-clipping
Gaussian model, in the subspace-knowledge setting described above,
(K1) No side information:MSE i≤σ2,R(I 1)≤σ2.
(K2) Knowscol(E):MSE i=σ2Pii∈[0,σ2],R(I 1) =σ2Pii.
(K3) KnowsE −i:MSEi=σ2,R(I 1) =σ2.
In (K2), the average per-entry MSE over all coordinates equals σ2r/nunder no clipping. This is an average
overj∈[n]and is not the risk on the single altered coordinate unless the altered index is randomized. If the
queryqis public and the attacker knowsE, thens=Eqis known exactly and the reconstruction MSE is
zero; the K2 formulas above apply only to the subspace-knowledge model.
Attacker’s reconstruction error for regime (iii) (reference vector).In regime (iii), σ2= 4cε,δ(n−1)
andI2(i) ={j̸=i}. Under the no-clipping Gaussian model, in the subspace-knowledge setting described
above,
(K1) No side information:MSE j≤σ2(j̸=i),R(I 2(i))≤(n−1)σ2.
(K2) Knowscol(E):MSE j=σ2Pjj(j̸=i),R(I 2(i)) =σ2/summationdisplay
j̸=iPjj=σ2(r−Pii)≤σ2r.
(K3) KnowsE −i:MSEj=σ2(P−i)jj(j̸=i),R(I 2(i)) =σ2r−i.
If the attacker knows the full galleryE, then the reference rowr i=Eeiis computable exactly and the
reconstruction MSE is zero. Therefore the K2 formulas above should be read as subspace-denoising benchmarks,
not as full-gallery-knowledge risks.
The equalities involving σ2Pjjandσ2(P−i)jjare no-clipping Gaussian benchmarks. With clipping, the
universal coordinate-wise bound MSEj≤σ2still holds for box-clipped coordinates, and the total nonexpansive
bound
E∥/hatwides−s∥2
2≤nσ2
continues to hold. Clipping may reduce the error on saturated coordinates, but the exact leverage-score
covariance identities need not hold after clipping. Moreover, note that these bounds are attacker estimation
errors given the one-shot release /hatwidesand the specified auxiliary-information regime. They do not alter the
(ε,δ)indistinguishability guarantee, which is defined at the dataset level under the adopted adjacency. We
summarize the attacker’s reconstruction error bounds in the following table.
44

arXiv preprint, ScoreShield
Regime Quantity (K1) none (K2) knowscol(E)(K3) knowsE −i
(i)I 1={i}MSEi ≤σ2σ2Pii σ2
R(I 1)≤σ2σ2Pii σ2
(iii)I 2(i) ={j̸=i}1
n−1/summationtext
j̸=iMSEj≤σ2 σ2(r−Pii)
n−1σ2r−i
n−1
R(I 2(i))≤(n−1)σ2σ2(r−Pii)σ2r−i
Note.K2 is a subspace-knowledge model. If all score-generating values are known, such as bothEand a
publicqin regime (i), or the full galleryEin regime (iii), then the corresponding clean score vector is exactly
computable and the reconstruction MSE is zero.
F.5 Regime (ii): Full Pairwise Similarity Score Matrix
We releaseS=EE⊤∈[−1,1]n×nwithS∈C coll:=/braceleftbig
S∈Rn×n:S⪰0, Sii= 1 (1≤i≤n ),|Sij|≤1 (i̸=
j)/bracerightbig
⊂Rn×n. Our ScoreShield mechanism underrecord-level adjacencyonEhas the exact Frobenius sensitivity
∆f,F=:∆F,rec
∆F,rec = 2/radicalbig
2(n−1) = Θ(√n),(169)
since changing recordichanges only the2(n−1)off-diagonal entries in row/columni.
We analyze both adjacency notions. In each regime we setσ2=cε,δ∆2,cε,δ:=2 log(2/δ)
ε2.
•(R)Record-level adjacency onE: exact Frobenius sensitivity∆ f,F= 2/radicalbig
2(n−1) = Θ(√n), hence
σ2= Θ(n).
•(O)Output-space adjacency onS[ 14]:∥S−S′∥F≤∆Gwith∆ G= Θ(1)independent of n, henceσ2= Θ(1).
Naïve Gaussian Mechanism.In our practical algorithm, we sampleWwith i.i.d. entries Wij∼N(0,σ2)
(Gaussian mechanism), and then apply the deterministic symmetrization post-processingG :=1
2(W+W⊤).
SinceSis symmetric, the symmetrized releaseS′:=1
2/parenleftbig
(S+W) + (S+W)⊤/parenrightbig
=S+Gis a post-processing
ofS+Wand therefore preserves( ε,δ)-DP7(withσcalibrated to the sensitivity ofSunder the chosen
adjacency). We report utility for this symmetric pre-projection matrixS′. A direct variance calculation gives
E∥S′−S∥2
F=E∥G∥2
F=nσ2
/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
diagonal+n(n−1)σ2
2/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
off-diagonal=n2+n
2σ2= Θ(n2σ2).(170)
Therefore, for the two adjacency definitions we have:
(a) Record-level adjacency(R): Usingσ2=cε,δ∆2
f,F= Θ/parenleftbignlog(2/δ)
ε2/parenrightbig
,
naïve + (R):E∥S′−S∥2
F= Θ/parenleftig
n2cε,δ∆2
F,rec/parenrightig
= Θ/parenleftign3log(2/δ)
ε2/parenrightig
.(171)
(b) Output-space adjacency(O): Withσ2=cε,δ∆2
G,
naïve + (O):E∥S′−S∥2
F= Θ/parenleftig
n2cε,δ∆2
G/parenrightig
= Θ/parenleftign2∆2
Glog(2/δ)
ε2/parenrightig
.(172)
RemarkF.5.If instead we perturb only the upper triangle and reflect, Eq. 170 still yieldsΘ( n2σ2), while
constants differ but the exponent is unchanged.
7Equivalently, samplingGdirectly as a symmetric Gaussian with Var(Gii) =σ2andVar(Gij) =σ2/2,∀i̸=j, yields the
same distribution as1
2(W+W⊤).
45

arXiv preprint, ScoreShield
ScoreShield Mechanism.Let /hatwideS=projCcoll(S+G)denote the exact Frobenius metric-projection release.
The risk bounds in this paragraph apply to /hatwideS. The AAP feasibility solver used in large-scale experiments is a
different post-processing map and is not the object of these exact-projection risk bounds.
Global Risk Bound via the Gaussian Complexity (uniform inS).Using Corollary F.2 we have
E∥/hatwideS−S∥2
F≤C σGC(C coll)≤/tildewideCσn3/2,(173)
where we usedGC(C coll) = Θ(n3/2).
(a) Record-level adjacency(R): Withσ= Θ/parenleftbigg√
nlog(2/δ)
ε/parenrightbigg
,
ScoreShield + (R):E∥ /hatwideS−S∥2
F≤/tildewideCn2/radicalbig
log(2/δ)
ε=O/parenleftigg
n2/radicalbig
log(2/δ)
ε/parenrightigg
.(174)
(b) Output-space adjacency(O): Withσ=√
2 log(2/δ)
ε∆G,
ScoreShield + (O):E∥ /hatwideS−S∥2
F≤/tildewideCn3/2∆G/radicalbig
log(2/δ)
ε=O/parenleftigg
n3/2∆G/radicalbig
log(2/δ)
ε/parenrightigg
.(175)
Local Risk Bound via Rank-Aware Tangent-Cone (under local Gram-smoothness).Let r:=rank(S)≤d
andT S:=TS(Ccoll)denote the contingent tangent cone of the elliptope atS. For the projected Gaussian
estimator/hatwideS=projCcoll(S+G), the conic denoising bound gives
E∥/hatwideS−S∥2
F≤σ2δ(TS).(176)
By the rank-aware upper bound (Lemma C.25), under local Gram-smoothnessT S(En) =Tman(S), we have
δ(TS)≤nr. The conic bound gives
E∥/hatwideS−S∥2
F≤σ2δ(TS)≤/tildewideCσ2nr=O(σ2nr).(177)
(a) Record-level adjacency(R): Withσ2= Θ/parenleftbignlog(2/δ)
ε2/parenrightbig
,
ScoreShield + (R):E∥ /hatwideS−S∥2
F≤ O/parenleftign2rlog(2/δ)
ε2/parenrightig
.(178)
(b) Output-space adjacency(O): Withσ2=2 log(2/δ)
ε2 ∆2
G,
ScoreShield + (O):E∥ /hatwideS−S∥2
F≤ O/parenleftignr∆2
Glog(2/δ)
ε2/parenrightig
.(179)
Summary.We summarize the mechanism reconstruction error bounds in the following table.
Mechanism Adjacency Mechanism MSEE∥ /hatwideS−S∥2
F
Naïve Gaussian (R): replace one rowΘ/parenleftbig
n3cε,δ/parenrightbig
ScoreShield(global) (R): replace one rowO/parenleftbig
n2√cε,δ/parenrightbig
ScoreShield(rank-aware) (R): replace one rowO/parenleftbig
n2rcε,δ/parenrightbig
Naïve Gaussian (O):∥S−S′∥F≤∆ G Θ/parenleftbig
n2∆2
Gcε,δ/parenrightbig
ScoreShield(global) (O):∥S−S′∥F≤∆ GO/parenleftbig
n3/2∆G√cε,δ/parenrightbig
ScoreShield(rank-aware) (O):∥S−S′∥F≤∆ GO/parenleftbig
nr∆2
Gcε,δ/parenrightbig
46

arXiv preprint, ScoreShield
F.6 Attacker Reconstruction Error for Regime (ii)
ScoreShield release model.LetW ∈Rn×nhave i.i.d. entries Wij∼N(0,σ2)and apply the deterministic
symmetrization post-processingG :=1
2(W+W⊤). ThenGii∼N (0,σ2)and fori < j,Gij=Gji∼
N(0,σ2/2). We release /hatwideS=projCcoll(S+G)whereS=EE⊤∈C coll:={X∈Sn:X⪰0,diag (X) =1,|Xij|≤
1 (i̸=j)}. Note that our practical AAP feasibility solver used in large-scale experiments is a different
post-processing map and is not analyzed by the coordinatewise formulas below. Euclidean projection in
Frobenius norm is firmly non-expansive (and therefore1-Lipschitz). Hence
∥/hatwideS−S∥ F≤∥G∥ F,E/bracketleftig
∥/hatwideS−S∥2
F/bracketrightig
≤E/bracketleftbig
∥G∥2
F/bracketrightbig
=n2+n
2σ2≤n2σ2.(180)
Per–coordinate inequalities like E[(/hatwideSij−Sij)2]≤Var (Gij)need not hold for the elliptope due to PSD coupling
(unlike coordinatewise clipping). They do hold for coordinate–separable clipping.
Risk functionals.Under record–level replacement of row iinE, the affected Gram entries are the
off–diagonal row/column Jrow
i:={(i,j) :j̸=i}, and, if counted, the symmetric strip Jstrip
i:={(i,j) :
j̸=i}∪{ (j,i) :j̸=i}. Because Sij=SjiandGij≡Gji8, the two strips are redundant. Therefore,
Rstrip(i) = 2Rrow(i)holds exactly. Define the coordinate risk MSEij:=E[(/hatwideSij−Sij)2]and the restricted risks
Rrow(i):=/summationdisplay
j̸=iMSEij,R strip(i):= 2R row(i).(181)
Knowledge regimes.We consider two auxiliary-information regimes: (i) no side information about the
embeddings; and (ii) one-row-unknown side information, where the attacker knows the gallery embeddings
{ej}j̸=ibut not the changed rowe i.
(i)No side information on embeddings.For the symmetric additive pre-projection modelS′=S+G, we
have, for everyj̸=i,
MSEij=E[(S′
ij−Sij)2] = Var(G ij) =σ2
2.(182)
Hence
Rrow(i) =(n−1)σ2
2,R strip(i) = (n−1)σ2(183)
for the symmetric additive pre-projection model.
For the exact metric-projection release /hatwideS=projCcoll(S+G), PSD coupling can redistribute error across
entries. Therefore, coordinatewise inequalities such as E[(/hatwideSij−Sij)2]≤Var (Gij)are not guaranteed.
However, the row-restricted risk is bounded by the total Frobenius risk:
Rrow(i) =E/summationdisplay
j̸=i(/hatwideSij−Sij)2≤E∥/hatwideS−S∥2
F.(184)
Since diag(/hatwideS) =diag(S) =1, the diagonal error is zero. Moreover, the diagonal part ofGcontributes only
an additive constant to the Frobenius projection objective over Ccoll, because every feasible matrix has unit
diagonal. Thus the exact projection depends only on the off-diagonal perturbation for the purpose of the
minimizer, and non-expansiveness gives
E∥/hatwideS−S∥2
F≤E∥G off∥2
F=n(n−1)
2σ2,(185)
whereG offdenotes the off-diagonal part ofG. Consequently,
Rrow(i)≤n(n−1)
2σ2,R strip(i)≤n(n−1)σ2.(186)
8SinceS=EE⊤is symmetric and we useS′=S+1
2(W+W⊤), both the pre- and post-projection matrices are symmetric.
The projector also preserves symmetry.
47

arXiv preprint, ScoreShield
(ii)Knows embeddings {ej}j̸=i(one-row unknown).LetE −i∈R(n−1)×dstack the rows{e⊤
j}j̸=i, let
r−i:= rank(E−i), and let
P−i:=E−i(E⊤
−iE−i)†E⊤
−i∈R(n−1)×(n−1)(187)
be the orthogonal projector ontocol(E −i). Under the symmetric additive pre-projection row model,
y=E−iei+η,η∼N/parenleftbigg
0,σ2
2In−1/parenrightbigg
,(188)
the clean off-diagonal score vectorE −ieilies in col(E−i). The affine-unbiased least-squares estimator of
this score vector is
/tildewideyLS=P−iy.(189)
Its error covariance is
Cov(/tildewideyLS−E−iei) =σ2
2P−i.(190)
Therefore, forj̸=i,
MSELS
ij=σ2
2(P−i)jj,(191)
and/summationdisplay
j̸=iMSELS
ij=σ2
2tr(P−i) =σ2r−i
2,1
n−1/summationdisplay
j̸=iMSELS
ij=σ2r−i
2(n−1).(192)
Moreover, sinceP −iis an orthogonal projector,
σ2r−i
2(n−1)≤max
j̸=iMSELS
ij≤σ2
2.(193)
For the released projected matrix /hatwideS, these LS formulas are pre-projection additive benchmarks. They are
not coordinatewise guarantees after projection, because projection onto Ccollcan redistribute error across
entries.
Calibration under (R) and (O): benchmarks vs. guarantees.Let σ2=cε,δ∆2withcε,δ=
2 log(2/δ)/ε2.
•(R) Record–level adjacency:∆ = ∆ F,rec = 2/radicalbig
2(n−1) = Θ(√n)soσ2= Θ/parenleftbignlog(2/δ)
ε2/parenrightbig
.
No side info: avg. per off–diagonal= Θ/parenleftignlog(2/δ)
ε2/parenrightig
,R row(i) = Θ/parenleftign2log(2/δ)
ε2/parenrightig
.
KnowsE−i:avg. per off–diagonal= Θ/parenleftbiggr−ilog(2/δ)
ε2/parenrightbigg
,R row(i) = Θ/parenleftbiggnr−ilog(2/δ)
ε2/parenrightbigg
.
With fixed r, the adversary’saverageper–entry error is O(1)inn, while a few high-leverage coordinates
may be larger.
•(O) Output–space adjacency:∆ G= Θ(1)andσ2=2 log(2/δ)
ε2 ∆2
G= Θ(1).
No side info: avg. per off–diagonal= Θ/parenleftig∆2
Glog(2/δ)
ε2/parenrightig
,R row(i) = Θ/parenleftign∆2
Glog(2/δ)
ε2/parenrightig
.
KnowsE−i:avg. per off–diagonal= Θ/parenleftigr−i∆2
Glog(2/δ)
nε2/parenrightig
,R row(i) = Θ/parenleftigr−i∆2
Glog(2/δ)
ε2/parenrightig
.
Summary of pre-projection row-risk benchmarks.Let r−i=rank(E−i). The following rates are
for the symmetric additive pre-projection modelS′=S+G. They are not asserted as coordinatewise or
48

arXiv preprint, ScoreShield
row-restricted guarantees for the projected release /hatwideS, because projection onto Ccollcan redistribute error across
entries.
Knowledge regime (R):R row(i)(O):R row(i)
No side infoΘ/parenleftign2log(2/δ)
ε2/parenrightig
Θ/parenleftign∆2
Glog(2/δ)
ε2/parenrightig
KnowsE−i Θ/parenleftignr−ilog(2/δ)
ε2/parenrightig
Θ/parenleftigr−i∆2
Glog(2/δ)
ε2/parenrightig
For the symmetric two-strip count in the same pre-projection model, the row risks are multiplied by2.
Remarks.(i) If one perturbs only the upper triangle and mirrors, constants change but the rates above are
unaffected. (ii) If the attacker knows all ofE, thenS=EE⊤is already known and reconstruction is trivial.
The one-row-unknown model matches record-level adjacency (R). (iii) Exact metric projection onto Ccoll
cannot increase the overall Frobenius error relative to the symmetric additive inputS+G. However, it can
redistribute error across entries. Therefore, the pre-projection strip formulas are exact for the purely additive
model, while post-projection row or strip risks should be interpreted through the global Frobenius bound
unless additional structure is imposed.
F.7 Visual Comparison of MSE Scaling Bounds
This subsection reports the analytical MSE bounds used to compare the naïve Gaussian mechanism with
ScoreShield. We use two scalings of the same bounds. The first set of figures reports unnormalized MSE
bounds at fixed privacy parameters( ε,δ). The second set reports the same bounds divided by cε,δ:=2 log(2/δ)
ε2,
σ2=cε,δ∆2. Dividing by cε,δremoves the privacy-calibration factor only for bounds that are linear in σ2.
This applies to the naïve Gaussian risk and to the local rank-aware tangent-cone bound. It does not apply
to the global Gaussian-complexity bound, which is linear in σ; after division by cε,δ, that bound retains the
factorc−1/2
ε,δ.
Regime (i): vector release.For query-to-collection vector release, the global ℓ2-sensitivity is∆ query= 2.
The naïve Gaussian release satisfies B(i)
naive(n):=E∥s′−s∥2
2=nσ2=ncε,δ∆2
query. For projection onto[ −1,1]n,
ifa(s)coordinates are boundary-active, the local tangent-cone bound gives B(i)
box(n,a):=cε,δ∆2
query/parenleftig
n−a(s)
2/parenrightig
.
Thus the projection changes the leading constant through boundary activity, but it does not change the
Θ(n)dependence on n. In the vector-release panels, the curve labeled by a=ncorresponds to the maximal
boundary-active reduction; the casea= 0coincides with the naïve Gaussian risk.
Regime (ii): Gram release.For full pairwise Gram release, the symmetrized Gaussian perturbation is
G=1
2(W+W⊤),Wiji.i.d.∼ N (0,σ2). ThenGii∼N(0,σ2)andGij=Gji∼N(0,σ2/2)fori<j. Hence the
naïve symmetrized Gaussian risk isB(ii)
naive(n,∆) :=E∥S′−S∥2
F=E∥G∥2
F=n(n+1)
2σ2=n(n+1)
2cε,δ∆2.
Under record-level adjacency(R), the exact Frobenius sensitivity satisfies∆2= ∆2
F,rec = 8(n−1), and therefore
B(ii,R)
naive (n) = Θ(c ε,δn3). Under output-space adjacency(O),∆ = ∆ G= Θ(1), soB(ii,O)
naive (n) = Θ(c ε,δ∆2
Gn2).
For the exact Frobenius metric-projection release /hatwideS=projCcoll(S+G), the global Gaussian-complexity bound
givesB(ii)
glob(n,∆) :=/tildewideCglobσn3/2=/tildewideCglob√cε,δ∆n3/2. Consequently,
B(ii,R)
glob(n) =O(√cε,δn2), B(ii,O)
glob(n) =O(√cε,δ∆Gn3/2).
After division byc ε,δ, these bounds become
B(ii,R)
glob(n)
cε,δ=O/parenleftbiggn2
√cε,δ/parenrightbigg
,B(ii,O)
glob(n)
cε,δ=O/parenleftbigg∆Gn3/2
√cε,δ/parenrightbigg
.
The conditional rank-aware tangent-cone bound is
B(ii)
rank(n,r,∆) :=/tildewideCrankσ2nr=/tildewideCrankcε,δ∆2nr.
49

arXiv preprint, ScoreShield
This bound assumes the local Gram-smoothness conditionT S(En) =T man(S). Under(R),
B(ii,R)
rank(n,r) =O(c ε,δn2r),
whereas under(O),
B(ii,O)
rank(n,r) =O(c ε,δ∆2
Gnr).
Pointwise minimum of valid upper bounds.Individual analytical upper bounds can be looser than
the naïve Gaussian risk for some values of n. For example, the rank-aware upper bound may lie above
the naïve curve when the rank-aware estimate is not active. This should not be interpreted as an increase
in the exact Frobenius risk after projection. SinceS ∈C colland Euclidean projection is non-expansive,/vextenddouble/vextenddoubleprojCcoll(S+G)−S/vextenddouble/vextenddouble
F≤∥G∥F. Therefore the naïve Gaussian risk is also a valid upper bound on the exact
metric-projection risk. Hence the pointwise minimum of the displayed valid upper bounds is itself a valid
upper bound:
Bmin(n):= min{B naive(n),B glob(n),B rank(n)}.
The curve Bminis not a lower bound and is not an empirical risk estimate. It is the smallest among the
displayed valid analytical upper bounds at each n. If the local Gram-smoothness assumption required for Brank
is not invoked, then the corresponding pointwise minimum is min{B naive(n),Bglob(n)}. The shaded regions in
the figures indicate the difference between the naïve Gaussian upper bound and the pointwise minimum of
the displayed valid upper bounds.
Global–rank crossover.The crossover between the global and rank-aware bounds is obtained from
/tildewideCrankcε,δ∆2nr≤/tildewideCglob√cε,δ∆n3/2.
Equivalently,
r≤/tildewideCglob
/tildewideCrank√n
∆√cε,δ.
Under record-level adjacency(R),∆ = ∆ F,rec =/radicalbig
8(n−1), so the condition becomes
r≤/tildewideCglob
/tildewideCrank√n/radicalbig
8(n−1)c ε,δ.
For largen, this is a constant-rank condition:
r=O/parenleftig
c−1/2
ε,δ/parenrightig
.
Under output-space adjacency(O),∆ = ∆ Gis independent ofn, and the condition becomes
r=O/parenleftbigg√n
∆G√cε,δ/parenrightbigg
.
Thus the crossover r=O(√n)applies only under output-space adjacency with fixed∆ Gand fixed privacy
parameters.
50

arXiv preprint, ScoreShield
101102103
n(gallerysize)102103104105MechanismMSE
=1,=1e06,c,=29
Regime(i):VectorRelease
Gaptonaiveupperbound
Naive
Boxbound(a=n)
Pointwisemin.upperbound
101102103
n(gallerysize)10210410610810101012MechanismMSE
=1,=1e06,c,=29
Regime(ii):GramReleaseunder(R)/(O)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)
Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)MSEupperboundsforfixed=1,=1e06
Figure F.1. MSE upper bounds at fixed( ε,δ)on a log–log scale.The left panel reports regime (i), where the naïve
vector-release risk is ncε,δ∆2
queryand the box-projection bound with a=nboundary-active coordinates changes only
the leading constant. The right panel reports regime (ii) under record-level adjacency(R)and output-space adjacency
(O). The global and conditional rank-aware bounds are shown separately. The dash-dotted curve reports the pointwise
minimum of the displayed valid upper bounds, min{B naive,Bglob,Brank}. The shaded region indicates the difference
betweenB naiveand this pointwise minimum.
0 500 1000 1500 2000 2500 3000
n(gallerysize)102103104105MechanismMSE
=1,=1e06,c,=29
Regime(i):VectorRelease
Gaptonaiveupperbound
Naive
Boxbound(a=n)
Pointwisemin.upperbound
0 500 1000 1500 2000 2500 3000
n(gallerysize)10210410610810101012MechanismMSE
=1,=1e06,c,=29
Regime(ii):GramReleaseunder(R)/(O)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)
Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)MSEupperboundsforfixed=1,=1e06
Figure F.2. MSE upper bounds at fixed( ε,δ)on a semi-log scale.The quantities are the same as in Fig. F.1. The
semi-log scale separates the finite- nvalues of the displayed upper bounds. The rank-aware curve can lie above the
naïve curve when the rank-aware upper bound is loose. The dash-dotted curve reports the pointwise minimum of the
displayed valid upper bounds.
51

arXiv preprint, ScoreShield
0 500 1000 1500 2000 2500 3000
n(gallerysize)050000100000150000200000250000300000350000MechanismMSE
=1,=1e06,c,=29
Regime(i):VectorRelease
Gaptonaiveupperbound
Naive
Boxbound(a=n)
Pointwisemin.upperbound
0 500 1000 1500 2000 2500 3000
n(gallerysize)0.00.51.01.52.02.53.0MechanismMSE
=1,=1e06,c,=29
1e12Regime(ii):GramReleaseunder(R)/(O)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)
Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)MSEupperboundsforfixed=1,=1e06
Figure F.3. MSE upper bounds at fixed( ε,δ)on a linear scale.This panel reports the same bounds as Figs. F.1–F.2
using a linear vertical scale.
2 3 4 5 6 7 8 9 10
n102103104105MechanismMSE
=1,=1e06,c,=29
Regime(ii):GramRelease(zoom)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)
Figure F.4. Finite- nview of unnormalized regime (ii) MSE upper bounds at fixed( ε,δ).This panel restricts the
horizontal axis to the finite- nrange shown in the figure. The rank-aware curves are conditional local bounds and can
exceed the naïve curve when they are not active. In such ranges, the pointwise minimum of the displayed valid upper
bounds coincides with either the naïve bound or the global bound.
52

arXiv preprint, ScoreShield
101102103
n(gallerysize)101102103104MechanismMSE/c,
c,=1fixed
Regime(i):VectorRelease
Gaptonaiveupperbound
Naive
Boxbound(a=n)
Pointwisemin.upperbound
101102103
n(gallerysize)1021041061081010MechanismMSE/c,
c,=1fixed
Regime(ii):GramReleaseunder(R)/(O)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)
Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)NormalizedMSEupperboundsforfixedc,=1
Figure F.5. MSE upper bounds divided by cε,δon a log–log scale.The displayed quantities are the bounds in Fig. F.1
divided by a fixed value of cε,δ. This division removes the privacy-calibration factor from the naïve and rank-aware
bounds, which are linear in σ2. It leaves a factor c−1/2
ε,δin the global bound, which is linear in σ. The dash-dotted
curve reports the pointwise minimum of the displayed valid upper bounds after the same division byc ε,δ.
0 500 1000 1500 2000 2500 3000
n(gallerysize)101102103104MechanismMSE/c,
c,=1fixed
Regime(i):VectorRelease
Gaptonaiveupperbound
Naive
Boxbound(a=n)
Pointwisemin.upperbound
0 500 1000 1500 2000 2500 3000
n(gallerysize)1021041061081010MechanismMSE/c,
c,=1fixed
Regime(ii):GramReleaseunder(R)/(O)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)
Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)NormalizedMSEupperboundsforfixedc,=1
Figure F.6. MSE upper bounds divided byc ε,δon a semi-log scale.The quantities are the same as in Fig. F.5. The
rank-aware curves are shown even when they are not the smallest valid upper bound.
53

arXiv preprint, ScoreShield
0 500 1000 1500 2000 2500 3000
n(gallerysize)020004000600080001000012000MechanismMSE/c,
c,=1fixed
Regime(i):VectorRelease
Gaptonaiveupperbound
Naive
Boxbound(a=n)
Pointwisemin.upperbound
0 500 1000 1500 2000 2500 3000
n(gallerysize)0.00.20.40.60.81.0MechanismMSE/c,
c,=1fixed
1e11Regime(ii):GramReleaseunder(R)/(O)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)
Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)NormalizedMSEupperboundsforfixedc,=1
Figure F.7. MSE upper bounds divided by cε,δon a linear scale.This panel reports the same divided bounds as
Figs. F.5–F.6 using a linear vertical scale. The pointwise minimum curve is an analytical upper bound and is not an
empirical MSE curve.
2 3 4 5 6 7 8 9 10
n101102103104MechanismMSE/c,
c,=1fixed
Regime(ii):GramRelease(zoom)
Naive(R)
Globalupperbound(R)
Rankawareupperbound(R),conditional
Pointwisemin.upperbound(R)
Naive(O)
Globalupperbound(O)
Rankawareupperbound(O),conditional
Pointwisemin.upperbound(O)Gaptonaiveupperbound(R)
Gaptonaiveupperbound(O)
Figure F.8. Finite- nview of regime (ii) MSE upper bounds divided by cε,δ.This panel restricts the horizontal axis to
the finite-nrange shown in the figure. Under record-level adjacency(R), the global–rank crossover is a constant-rank
condition up to the constants in the two bounds and the factor c−1/2
ε,δ. Under output-space adjacency(O), the crossover
satisfiesr=O(√n/(∆ G√cε,δ)).
54

arXiv preprint, ScoreShield
GSupplementary Details for Regime (i): Omitted Theorems, Propositions,
Proofs and Lemmas
G.1 Privacy Guarantee and Stability of Projections
Theorem G.1(Privacy Guarantee of Query-to-Collection Similarity Score Vector Release).Let Mquerybe
the mechanism returned by Algorithm 1 with parameters ε>0, δ∈ (0,1). Then for every pair of neighboring
embedding matricesE∼E′(Def. C.5) and every measurable setS⊆[−1,1]n,
Pr/bracketleftbig
Mquery(E)∈S/bracketrightbig
≤eεPr/bracketleftbig
Mquery(E′)∈S/bracketrightbig
+δ.(194)
HenceM queryis(ε,δ)–DP.
Proof.Section 3.1 established the global ℓ2-sensitivity∆ query = 2. The Gaussian mechanism (Lemma C.6)
with scaleσ= ∆ query/radicalbig
2 log(2/δ)/ε = 2/radicalbig
2 log(2/δ)/ε is therefore( ε,δ)–DP. Projection onto the fixed convex
setC= [−1,1]ndepends only on the noisy output, so by the post-processing lemma (Lemma C.7) the
composite mechanism remains(ε,δ)–DP.
Definition G.2(Normal cone).Let C⊂Rnbe non-empty, closed, and convex, and letx ∈C. The normal
cone toCatxis
Nx(C):=/braceleftbig
v∈Rn:⟨v,y−x⟩≤0,∀y∈C/bracerightbig
.(195)
Definition G.3(Critical cone).Let C⊂Rnbe non-empty, closed, and convex. Fixs ∈Rnand define
x:=projC(s)andu :=s−x∈N x(C)(see Lemma G.4). The critical cone at(x,u)is
K(x,u) :=T x(C)∩u⊥,u⊥:={v∈Rn:⟨v,u⟩= 0}.(196)
Lemma G.4(Variational Inequality for Euclidean Projection).Let C⊂Rnbe non-empty, closed, and convex.
For anyz∈Rnandp :=projC(z), one has
⟨z−p,y−p⟩≤0,∀y∈C.(197)
Equivalently,z−p∈N p(C).
Proof.The pointpminimizes the convex function ϕ(y):=1
2∥z−y∥2
2over the closed convex set C. For any
y∈Candτ∈(0,1), the pointp τ:= (1−τ)p+τy∈C, henceϕ(p)≤ϕ(pτ). Expanding ϕ(pτ)and dividing
byτthen lettingτ↓0yields Eq. 197. The normal-cone equivalence is exactly Definition G.2.
Lemma G.5(Monotonicity of the Normal Cone Mapping).Let C⊂Rnbe non-empty, closed, and convex. If
v∈N x(C)andv′∈Nx′(C), then
⟨v−v′,x−x′⟩≥0.(198)
Proof.Sincev∈Nx(C)andx′∈C, we have⟨v,x′−x⟩≤0. Similarly,v′∈Nx′(C)andx∈Cimply
⟨v′,x−x′⟩≤0. Adding the two inequalities gives Eq. 198.
Lemma G.6(Directional derivative of the projector onto a polyhedral convex set).Let C⊂Rmbe a nonempty
closed convex polyhedron. Fixs ∈Rmand definex :=projC(s),u :=s−x∈Nx(C). LetK:=Tx(C)∩u⊥be
the critical cone. Then, for every directionh∈Rm, the one-sided directional derivative
DprojC(s)(h) := lim
t↓0projC(s+th)−projC(s)
t(199)
exists and satisfies
DprojC(s)(h) =projK(h).(200)
Equivalently,
projC(s+th) =x+tprojK(h) +o(t), t↓0.(201)
55

arXiv preprint, ScoreShield
Proof.SinceCis polyhedral, write C={z∈Rm:Az≤b,Bz=c}. LetI(x):={i:a⊤
ix=bi}be the active
inequality set atx. Then the tangent cone is
Tx(C) ={d:a⊤
id≤0∀i∈I(x),Bd= 0}.(202)
Fort>0, setx t:=projC(s+th),d t:=xt−x
t.
By nonexpansiveness of the projector,
∥dt∥2=∥projC(s+th)−projC(s)∥ 2
t≤∥h∥ 2,(203)
so(dt)is bounded.
Because inactive inequalities atxhave a positive slack, for every bounded set of directions and all sufficiently
smallt, the conditionx+ td∈Cis equivalent tod ∈Tx(C). Hence, for all sufficiently small t,dtis the unique
minimizer overT x(C)of1
2∥x+td−(s+th)∥2
2.
Sinces=x+u, this is equivalent, after removing constants and dividing byt>0, to
dt= arg min
d∈T x(C)/braceleftbigg
−⟨u,d⟩+t
2∥d−h∥2
2/bracerightbigg
.(204)
Becauseu∈N x(C), we have⟨u,d⟩≤0,∀d∈T x(C).
Thus the first term is nonnegative and vanishes exactly on K=Tx(C)∩u⊥. Letv∈K. By optimality ofd t,
−⟨u,dt⟩+t
2∥dt−h∥2
2≤t
2∥v−h∥2
2.(205)
Since−⟨u,d t⟩≥0, Eq. 205 implies
∥dt−h∥2
2≤∥v−h∥2
2,∀v∈K.(206)
The same inequality also implies
0≤−⟨u,d t⟩≤t
2∥v−h∥2
2,∀v∈K.(207)
Lettk↓0be any sequence such thatd tk→d. SinceT x(C)is closed,d∈Tx(C). Taking the limit in Eq. 207
gives⟨u,d⟩= 0; henced∈K. Taking the limit in Eq. 206 gives
∥d−h∥2
2≤∥v−h∥2
2,∀v∈K.(208)
Therefored= projK(h). Every cluster point of(d t)is the same vector, sod t→projK(h)ast↓0. This proves
the claimed directional derivative.
Lemma G.7(Directional derivative of the PSD-cone projector).LetA∈Snand let
A=U
Λ+0 0
0 0 0
0 0Λ−
U⊤(209)
be an eigendecomposition, where Λ+≻0contains the positive eigenvalues, Λ−≺0contains the negative
eigenvalues, and the middle block corresponds to the zero eigenspace. ForH∈Sn, write
/tildewideH=U⊤HU=
/tildewideH++/tildewideH+0/tildewideH+−
/tildewideH0+/tildewideH00/tildewideH0−
/tildewideH−+/tildewideH−0/tildewideH−−
.(210)
Define the matrixΓ∈R|+|×|−|by
Γij=λi
λi−λj, λi>0, λj<0.(211)
56

arXiv preprint, ScoreShield
Then the projectorprojSn
+is directionally differentiable atA, and
DprojSn
+(A)(H) =U
/tildewideH++/tildewideH+0 Γ◦/tildewideH+−
/tildewideH0+ projS|0|
+(/tildewideH00) 0
Γ⊤◦/tildewideH−+ 0 0
U⊤,(212)
where◦denotes the Hadamard product. Empty blocks are omitted.
Proof.The PSD projection is the spectral operator associated with the scalar functionf(λ) = max{λ,0}:
projSn
+(A) =Udiag(f(λ 1),...,f(λ n))U⊤.(213)
For eigenvalue pairs away from zero, the directional derivative of a spectral operator is governed by the first
divided differences off:
f[1](λi,λj) =

f(λi)−f(λj)
λi−λj, λi̸=λj,
f′(λi), λ i=λj, λi̸= 0.(214)
Thus the positive-positive block has coefficient1, the negative-negative block has coefficient0, the positive-zero
block has coefficient1, the zero-negative block has coefficient0, and the positive-negative block has coefficient
λi
λi−λj, λi>0, λj<0.(215)
On the zero eigenspace, fis not differentiable as a scalar function. The directional derivative of the spectral
operator restricted to this block is therefore the spectral operator generated by the one-sided directional
derivative offat zero, namelyf′
+(0;µ) = max{µ,0}.
Applied to the zero-eigenspace compression /tildewideH00, this gives projS|0|
+(/tildewideH00). Combining these block contributions
gives Eq. 212, which is the standard directional-derivative formula for the spectral projection onto the PSD
cone.
Lemma G.8(Stability of ScoreShield Projections).Let C⊂Rnbe non-empty, closed, and convex, and let
projCdenote the Euclidean projector as defined in Definition C.13. Fix σ>0and letw∼N(0,σ2In). For
anys∈Rn, definex :=projC(s),∆ :=projC(s+w)−projC(s). Then:
(a)Coarse Global Bound.
E∥∆∥2
2≤E∥w∥2
2=nσ2.(216)
(b)Global Gaussian-Complexity Bound (non-asymptotic). IfCis bounded, then
E∥∆∥2
2≤4σGC(C),(217)
(c)Small-noise Local Limit (geometry-aware). Letz ∼N (0,In)and defineu :=s−x∈Nx(C). Let
K:=K(x,u)be the critical cone (Definition G.3). Then
lim
t↓01
t2E/vextenddouble/vextenddoubleprojC(s+tσz)−x/vextenddouble/vextenddouble2
2=σ2δ(K), δ(K) :=E∥projK(z)∥2
2.(218)
Moreover,δ(K)≤nand
GW(K)2≤δ(K)≤GW(K)2+ 1.(219)
Proof.
(a)Byfirmnon-expansivenessof projC(LemmaC.27), onehas ∥∆∥ 2≤∥w∥2. Squaringandtakingexpectations
givesE∥∆∥2
2≤E∥w∥2
2=nσ2.
(b)See Lemma F.1 for a complete proof.
57

arXiv preprint, ScoreShield
(c)Definex= projC(s)andu=s−x, and letK=K(x,u). For each fixed realization ofz, Lemma G.6 yields
projC(s+tσz)−x
t−−→
t↓0σprojK(z).(220)
Moreover, by1-Lipschitzness ofprojC,
/vextenddouble/vextenddouble/vextenddoubleprojC(s+tσz)−x
t/vextenddouble/vextenddouble/vextenddouble
2≤σ∥z∥ 2,∀t>0.(221)
Since E∥z∥2
2<∞, the family/vextenddouble/vextenddouble/parenleftbig
projC(s+tσz)−x/parenrightbig
/t/vextenddouble/vextenddouble2
2is dominated by σ2∥z∥2
2. Applying dominated
convergence to Eq. 220 gives
lim
t↓01
t2E/vextenddouble/vextenddoubleprojC(s+tσz)−x/vextenddouble/vextenddouble2
2=σ2E∥projK(z)∥2
2=σ2δ(K).(222)
The bound δ(K)≤nfollows from∥projK(z)∥2≤∥z∥2andE∥z∥2
2=n. The inequalities GW(K)2≤δ(K)≤
GW(K)2+ 1are the standard relationship between statistical dimension and squared Gaussian width for
closed convex cones.
Corollary G.9(Piecewise-affine structure of polyhedral projections inScoreShield).Let C ⊂Rnbe
a nonempty closed convex polyhedron. Fixs ∈Rnand definex :=projC(s),u :=s−x∈Nx(C). Let
K:=Tx(C)∩u⊥denote the critical cone at(x,u).
(a)Locally exact first-order expansion. There exist finitely many polyhedral cones {Qj}J
j=1whose union is
Rnand whose relative interiors are disjoint such that, for every jand everyh∈ri(Qj), there exists t0(h)>0
for which
projC(s+th) =x+tDprojC(s)(h),∀t∈(0,t 0(h)).(223)
Moreover, by Lemma G.6,DprojC(s)(h) =projK(h).
Thus the first-order expansion has zero remainder along all directions in the relative interiors of the cones of
this partition. In particular, ifhhas an absolutely continuous distribution, for exampleh ∼N(0,In), then
this locally exact regime holds with probability one.
(b)Explicit critical cone and statistical dimension for the box. Let C= [−1,1]nand letx= projC(s)be the
componentwise clipping ofs. Define
I0:={i:|xi|<1}, I +:={i:xi= 1}, I−:={i:xi=−1}.(224)
Refine the active sets according to complementarity:
I>0
+:={i∈I +:ui>0}, I0
+:={i∈I +:ui= 0},(225)
and
I<0
−:={i∈I−:ui<0}, I0
−:={i∈I−:ui= 0}.(226)
Then
K=/braceleftig
v∈Rn:vi∈R∀i∈I 0, vi≤0∀i∈I0
+, vi≥0∀i∈I0
−, vi= 0∀i∈I>0
+∪I<0
−/bracerightig
.(227)
Consequently, forz∼N(0,I n),
δ(K) =E∥projK(z)∥2
2=|I 0|+1
2/parenleftbig
|I0
+|+|I0
−|/parenrightbig
.(228)
In particular, ifI0
+=I0
−=∅,δ(K) =|I 0|and, ifu=0, equivalentlys∈[−1,1]nandx=s, then
δ(K) =|I 0|+1
2(|I+|+|I−|) =n−1
2(|I+|+|I−|).(229)
58

arXiv preprint, ScoreShield
Proof.
(a)SinceCis polyhedral, the Euclidean projector projCis piecewise affine. Hence there exist finitely many
polyhedra{Rj}J
j=1covering Rnand affine mapsz ∝⇕⊣√∫⊔≀→Ajz+bjsuch that projC(z) =Ajz+bj,∀z∈Rj. Fix
sand define the corresponding cones of directions
Qj:={h∈Rn:∃t0>0such thats+th∈R j,∀t∈(0,t 0)}.(230)
After discarding empty cones and refining overlaps if necessary, these sets form a finite polyhedral conic
partition ofRn.
Ifh∈ri(Q j), then there existst 0(h)>0such thats+th∈R jfor allt∈(0,t 0(h)). Therefore
projC(s+th) =A j(s+th) +b j (231a)
=projC(s) +tAjh,∀t∈(0,t 0(h)).(231b)
By Lemma G.6,
Ajh= DprojC(s)(h) =projK(h).(232)
This proves the locally exact expansion. The boundary of a finite polyhedral conic partition is a finite union of
lower-dimensional polyhedral cones and therefore has Lebesgue measure zero. Hence any absolutely continuous
direction belongs to the union of the relative interiors with probability one.
(b)ForC= [−1,1]n, the tangent cone atxis coordinatewise:
vi∈R(i∈I 0), vi≤0 (i∈I +), vi≥0 (i∈I−).(233)
The normal vectoru=s−xhas the corresponding signs
ui= 0 (i∈I 0), ui≥0 (i∈I +), ui≤0 (i∈I−).(234)
Therefore, for everyv∈T x(C),uivi≤0,i= 1,...,n. The conditionv∈u⊥is/summationtextn
i=1uivi= 0.
Since every summand is nonpositive, the sum can equal zero only if uivi= 0,i= 1,...,n. Thus, on an active
coordinate with strict complementarity, namely ui>0onI+orui<0onI−, we must have vi= 0. On
active coordinates with ui= 0, the one-sided tangent restriction remains. This gives the stated product-form
description ofK.
SinceKis a Cartesian product of coordinate cones, its Euclidean projection acts coordinatewise. A free
coordinate contributes E[z2
i] = 1. A half-line coordinate contributes E[(zi)2
+] =E[(zi)2
−] =1
2, by symmetry
of the standard normal distribution. A fixed-zero coordinate contributes zero. Summing the coordinate
contributions yields
δ(K) =|I 0|+1
2/parenleftbig
|I0
+|+|I0
−|/parenrightbig
.(235)
If strict complementarity holds on all active coordinates, then I0
+=I0
−=∅, and Eq. 235 gives δ(K) =|I0|. If
u=0, equivalentlys∈[−1,1]nandx=s, thenI>0
+=I<0
−=∅, and Eq. 235 gives
δ(K) =|I 0|+1
2(|I+|+|I−|) =n−1
2(|I+|+|I−|).
G.2 Impact on Verification Thresholds
Fix an operating verification threshold τ∈ [−1,1]. Thecleanverification decision for a pair( i,j)is
accept⇐⇒S ij> τ, whereSij=⟨ei,ej⟩is the cosine-similarity score. Lets ∈[−1,1]nbe the vector of
query-to-collection similarities for a given probe, one entry per gallery identity.ScoreShieldperturbs and
then projects scores coordinatewise:
s′=s+w,w∼N(0,σ2In),
/hatwides=projCquery(s′),C query= [−1,1]n,/parenleftbig
projCquery(s′)/parenrightbig
i= max/parenleftbig
−1,min(1,s′
i)/parenrightbig
.
59

arXiv preprint, ScoreShield
Projection monotonicity at interior thresholds.The scalar projection proj[−1,1]is nondecreasing and
fixes every point in(−1,1). Hence, for any scalar scoreSand anyτ∈[−1,1),
1/braceleftig
proj[−1,1] (S+W)>τ/bracerightig
=1{S+W >τ}.(236)
Indeed, ifS+W≤τ then proj(S+W )≤τ, ifτ <S+W≤ 1then proj(S+W ) =S+W >τ , and ifS+W > 1
then proj(S+W ) = 1>τsinceτ <1. The only exceptional case is τ= 1, where1{proj (S+W )>1}≡0but
1{S+W >1}may be1.
Therefore, for any interior threshold τ∈(−1,1), clipping does not affect the decision rule:1 {/hatwideS > τ} =
1{S+W >τ}. Hence post-privacy FMR depend only on the additive Gaussian mechanism noise Wand hold
for an arbitrary distribution of the clean scoreS.
Definition G.10(Complementary Gaussian Tail).For a standard normal variableZ∼N(0,1)we denote
Q(x) =Pr[Z≥x] =1√
2π/integraldisplay∞
xe−u2/2du.(237)
The lower-tail CDF isΦ(x) = 1−Q(x).
Definition G.11(Decision Flip).Let Sij∈[−1,1]be the clean cosine-similarity score for a pair( i,j), let
/hatwideSijbe its perturbed-and-projected version, and let τ∈[−1,1]be a fixed verification threshold. Adecision
flipoccurs when the private verdict disagrees with the clean verdict, i.e.,
(Sij−τ) (/hatwideSij−τ)<0.(238)
Ties are not counted as a flip.
Proposition G.12(Exact flip probability at interior thresholds).Let S∈[−1,1],τ∈(−1,1)andW∼
N(0,σ2). Then
Pr[flip|S=s] =Q/parenleftbigg|s−τ|
σ/parenrightbigg
.(239)
Proof.By Eq. 236, for every τ <1we have1{/hatwideS >τ} =1{S+W >τ}, so a flip occurs iff SandS+Wlie on
different sides of τ, i.e., iffW <−d whend:=s−τ > 0, orW >−d whend<0. Each case has probability
Q(|d|/σ)by symmetry.
RemarkG.13 (Endpoint behavior at τ= 1).With strict rule “ >”,1{/hatwideS >1}≡0for allS≤1, so flips cannot
occur atτ= 1. The flip probability equals0even thoughPr[S+W >1]may be positive. That is
Pr[flip|S=s] = 0andPr[ /hatwideS >1] = 0.
This is the only threshold at which clipping alters the decision rule. For every interior threshold τ∈(−1,1),
clipping does not affect decisions and the flip law is given by Proposition G.12. This endpoint effect does not
preclude threshold re-calibration. Under W∼N (0,σ2), the random variable S+Wadmits a continuous
density. Hence, for any fixed (possibly random) S∈[−1,1], the map τ∝⇕⊣√∫⊔≀→Pr [/hatwideS > τ ] =Pr[S+W > τ ]is
continuous and strictly decreasing on( −1,1), and with the strict rule “ >” we have Pr[/hatwideS > 1] = 0. One
can potentially raise the threshold from any τα∈(−1,1)to a unique τα+ ∆τ≤1that restores any target
α∈(0,1). We formalize this in the following.
RemarkG.14 (Exponential decay and Mills’ bound).Fork>0,
Pr[flip|S=s] =Q(k)≤e−k2/2
k√
2π,(240)
so the flip probability in Proposition G.12 decays like exp(−k2/2)in the normalized verification margin
k=|s−τ|/σ. The bound is conservative but tightens askgrows.
60

arXiv preprint, ScoreShield
Acceptance probabilities.For anyτ <1, Eq. 236 yields
Pr[/hatwideS >τ] =E/bracketleftbig
1{S+W >τ}/bracketrightbig
=E/bracketleftig
Q/parenleftbiggτ−S
σ/parenrightbigg/bracketrightig
,(241)
where the expectation is taken over the (arbitrary) distribution of the clean score SandW∼N (0,σ2)is
independent. At the endpoint, Pr[/hatwideS >1] = 0under the strict rule “ >” (Remark G.13). Because Whas a
continuous density, Pr[S+W=τ] = 0for every τ <1, so the strict “ >” and non-strict “ ≥” rules coincide on
(−1,1).
Endpoint behavior atτ= 1.With the strict rule “ /hatwideS >τ”, projection saturates at1but1̸>1, hence
Pr[/hatwideS >1] = 0,∀S≤1,
which is the only threshold where clipping alters the decision rule. For every τ <1the projection is monotone
and fixes(−1,1), hence “>” and “≥” coincide and1{/hatwideS >τ} =1{S+W >τ}. If the upper endpoint rule is
instead non-strict (i.e., “ /hatwideS≥τ”), then
Pr[/hatwideS≥1] =Pr[S+W≥1] =E/bracketleftig
Q/parenleftbigg1−S
σ/parenrightbigg/bracketrightig
>0,(242)
so the minimum achievable acceptance over τ≤1is strictly positive in that semantics. (For every τ <1, “>”
and “≥” remain identical.)
Recalibration guarantee and feasibility conditions.The map τ∝⇕⊣√∫⊔≀→Pr [/hatwideS >τ ]is continuous and strictly
decreasing on(−1,1). From Eq. 241 we have
d
dτPr[/hatwideS >τ] =−1
σE/bracketleftbigg
φ/parenleftbiggτ−S
σ/parenrightbigg/bracketrightbigg
<0,(243)
Thusd
dτPr[/hatwideS >τ ]<0forτ∈(−1,1). Atτ= 1the strict rule induces a jump discontinuity, so this derivative
statement applies only to the interior. Consequently, starting from any interior operating point τ0∈(−1,1)
there exists auniqueoffset∆τ∈[0,1−τ 0]such that
Pr[/hatwideS >τ 0+ ∆τ] =α,
whereαis a desired target utility level. In the following sections we will instantiate this with impostor scores
(yielding a recalibration that restores a desired operating rate). By contrast, under the non-strict endpoint
rule one encounters a genuinefeasibility frontier: targets below E[Q((1−S)/σ) ]are unattainable unless σis
reduced (equivalently,εincreased). We defer the precise statement and computation to the next section.
Definition G.15(Clean False–Match Rate ( FMR)).LetSijdenote animpostorcosine–similarity score
(i̸=j). For any thresholdτ∈[−1,1]define thecleanfalse–match rate9
FMR clean(τ):=Pr/bracketleftbig
Sij>τ|i̸=j/bracketrightbig
.(244)
An operational FR system fixes a targetFMRlevelα∈(0,1)(e.g.,α= 10−2or10−3) and chooses
τα:= inf/braceleftbig
τ:FMR clean(τ)≤α/bracerightbig
.(245)
When the impostor CDF is continuous at ταone has FMR clean(τα) =α. Otherwise FMR clean(τα)≤αby
construction.
RemarkG.16 (Endpoint behavior of FMR clean).ForS∈[−1,1],FMR clean(τ) =Pr[S >τ ]is right–continuous
and non–increasing on[−1,1], with
FMR clean(1) = 0,lim
τ↑1FMR clean(τ) =Pr[S= 1].
Thus there is no jump atτ= 1iffPr[S= 1] = 0(e.g., when the impostor score law is continuous at1).
9For impostor comparisons (negative class), the false match rate used in biometrics coincides with the false positive rate used
in ROC analysis:
FMR(τ) =Pr/bracketleftbig
Sij>τ|i̸=j/bracketrightbig
= 1−Fi(τ) =FPR(τ).
We use FMR when discussing verification operating points and FPR when parameterizing ROC curves.
61

arXiv preprint, ScoreShield
Exact post-privacy FMR at interior thresholds.For τ∈(−1,1), projection monotonicity (Eq. 236)
implies
FMR priv(τ):=Pr/bracketleftig
/hatwideSij>τ|i̸=j/bracketrightig
=E/bracketleftbigg
Φ/parenleftbiggSij−τ
σ/parenrightbigg
|i̸=j/bracketrightbigg
=E/bracketleftbigg
Q/parenleftbiggτ−Sij
σ/parenrightbigg
|i̸=j/bracketrightbigg
,(246)
i.e., the clean survivor function smoothed by a Gaussian kernel. Evaluating at τ=τα∈(−1,1)gives the
exact post-privacy FMR at the clean operating point.
RemarkG.17 (Endpoint behavior of FMR privatτ= 1).With the strict rule “ >”,FMR priv(τ) =E[1{S+W >
τ}]is continuous and strictly decreasing on(−1,1), and
lim
τ↑1FMR priv(τ) =E/bracketleftbigg
Q/parenleftbigg1−S
σ/parenrightbigg/bracketrightbigg
=:L(σ)>0.(247)
At the endpoint the strict rule forces FMR priv(1) = 0because1 ̸>1, so there is a jump discontinuity of size
L(σ)atτ= 1. Consequently, a target level α∈(0,1)can be attained by thresholding below1iff α≥L (σ).
The targetsα∈(0,L(σ))are unattainable in strict semantics (the endpoint itself attains onlyα= 0).
In contrast, with the non–strict rule “ ≥” we have FMR priv(1) =L(σ)and the map is continuous at τ= 1. The
minimum achievable acceptance overτ≤1is exactlyL(σ)>0, which induces a genuine feasibility frontier.
Corollary G.18(Feasible threshold shifts and uniqueness).Fix σ > 0and an interior operating point
τ0∈(−1,1)with target levelα∈(0,1).
Strict endpoint (“>”).The mapτ∝⇕⊣√∫⊔≀→FMR priv(τ)is continuous and strictly decreasing on(−1,1)with
lim
τ↑1FMR priv(τ) =L(σ)>0andFMR priv(1) = 0.
Hence the equationFMR priv(τ0+ ∆τ) =αhas a unique interior solution∆τ∈[0,1−τ 0)iff
α∈/bracketleftbig
L(σ),FMR priv(τ0)/bracketrightbig
.
The boundary valueτ 0+ ∆τ= 1attainsα= 0. Targetsα∈(0,L(σ))are unattainable by anyτ≤1.
Non–strict endpoint (“ ≥”).The map is continuous on[ −1,1]with minimum FMR priv(1) =L(σ)>0, so a
(unique) solution exists iffα∈/bracketleftbig
L(σ),FMR priv(τ0)/bracketrightbig
.
Proposition G.19(Threshold-crossing identity and small–noise expansion at τα).Assume the additive
mechanism noise is W=σZwithZ∼N (0,1)independent of the clean impostor score Sij. Letτα∈(−1,1)
satisfyFMR clean(τα) = Pr[Sij>τα|i̸=j] =α. Forτ∈(−1,1)define
FMR priv(τ):= Pr[Sij+W >τ|i̸=j] =E/bracketleftbigg
Q/parenleftbiggτ−Sij
σ/parenrightbigg/vextendsingle/vextendsingle/vextendsingle/vextendsinglei̸=j/bracketrightbigg
.(248)
Then, atτ=τ α,
FMR priv(τα) =α−E/bracketleftbigg
1{Sij>τα}Q/parenleftbiggSij−τα
σ/parenrightbigg/vextendsingle/vextendsingle/vextendsingle/vextendsinglei̸=j/bracketrightbigg
+E/bracketleftbigg
1{Sij≤τα}Q/parenleftbiggτα−Sij
σ/parenrightbigg/vextendsingle/vextendsingle/vextendsingle/vextendsinglei̸=j/bracketrightbigg
.(249)
Moreover, suppose the conditional (impostor) law of Sij|(i̸=j)admits a density fiin a neighborhood of τα,
andf iis continuously differentiable atτ α. Then, asσ→0,
FMR priv(τα) =α−σ2
2f′
i(τα) +o(σ2).(250)
Equivalently, sinceF′
i=fi, ifF′′
i(τα)exists then
FMR priv(τα) =α−σ2
2F′′
i(τα) +o(σ2).(251)
62

arXiv preprint, ScoreShield
Proof. Flip decomposition.Condition onS ij=s. Forτ∈(−1,1),
Pr[Sij+W >τ|S ij=s] = Pr[W >τ−s] =Q/parenleftbiggτ−s
σ/parenrightbigg
.(252)
Ifs>τ, thenQ((τ−s)/σ) = 1−Q((s−τ)/σ), so
1{s>τ}Q/parenleftbiggτ−s
σ/parenrightbigg
=1{s>τ}−1{s>τ}Q/parenleftbiggs−τ
σ/parenrightbigg
.(253)
Ifs≤τ, thenQ((τ−s )/σ) =Q((τ−s )/σ)as written. Taking expectations (conditional on i̸=j) and using
Pr[Sij>τα|i̸=j] =αgives Eq. 249.
Rewrite the smoothing bias.Fixτ∈(−1,1)and write (still underi̸=j)
FMR priv(τ) =/integraldisplay
Q/parenleftbiggτ−s
σ/parenrightbigg
fi(s) ds.(254)
Introduce the odd correction kernel
q(u) :=Q(u)−1{u<0}.(255)
ThenQ(u) =1{u<0}+q(u), hence
FMR priv(τ) =/integraldisplay
1{s>τ}f i(s) ds
/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
=FMR clean(τ)+/integraldisplay
q/parenleftbiggτ−s
σ/parenrightbigg
fi(s) ds.(256)
Atτ=ταthe first term equalsα. It remains to expand the second term.
Change variables and use a first-order Taylor remainder.Let u= (τ−s )/σsos=τ−σuandds=−σdu.
Then /integraldisplay
q/parenleftbiggτ−s
σ/parenrightbigg
fi(s) ds=σ/integraldisplay
Rq(u)f i(τ−σu) du.(257)
Becausef iis differentiable atτ, write
fi(τ−σu) =f i(τ)−σuf′
i(τ) +σuε σ(u),(258)
whereεσ(u)→0pointwise asσ→0(by the definition of derivative). Plugging in:
σ/integraldisplay
q(u)f i(τ−σu) du=σf i(τ)/integraldisplay
q(u) du−σ2f′
i(τ)/integraldisplay
uq(u) du+σ2/integraldisplay
uq(u)εσ(u) du.(259)
Evaluate the kernel moments and bound the remainder.First, qis odd because for u>0,q(u) =Q(u)while
foru<0,q(u) =Q(u)−1 =−(1−Q(u)) =−Q(−u), henceq(−u) =−q(u). Therefore/integraltext
q(u)du= 0.
Next,uq(u)is even and integrable, and
/integraldisplay
Ruq(u) du= 2/integraldisplay∞
0uQ(u) du.(260)
Using the identity forZ∼N(0,1),
E[(Z +)2] =/integraldisplay∞
0Pr(Z2
+>t) dt=/integraldisplay∞
0Pr(Z >√
t) dt=/integraldisplay∞
02uQ(u) du,(261)
andE[(Z +)2] =1
2E[Z2] =1
2, we obtain/integraltext∞
0uQ(u) du=1
4, hence
/integraldisplay
Ruq(u) du=1
2.
63

arXiv preprint, ScoreShield
Finally, since/integraltext
|uq(u)|du <∞andεσ(u)→0pointwise with |εσ(u)|bounded for small σ(from local
boundedness off′
inearτ), dominated convergence gives
/integraldisplay
uq(u)εσ(u) du=o(1) (σ→0).(262)
Combining the above,/integraldisplay
q/parenleftbiggτ−s
σ/parenrightbigg
fi(s) ds=−σ2f′
i(τ)·1
2+o(σ2).(263)
Settingτ=ταand adding the clean term αyields Eq. 250. The equivalence with F′′
i(τα)follows from
F′
i=fi.
Corollary G.20(One-sided flip bounds (interior thresholds)).Forτ α∈(−1,1),
α−E/bracketleftbigg
1{Sij>τα}Q/parenleftbiggSij−τα
σ/parenrightbigg/bracketrightbigg
≤FMR priv(τα)≤α+E/bracketleftbigg
1{Sij≤τα}Q/parenleftbiggτα−Sij
σ/parenrightbigg/bracketrightbigg
.(264)
Applying Mills’ inequalityQ(z)≤e−z2/2/(z√
2π)yields closed-form, data-dependent upper/lower bounds.
Definition G.21(Feasibility frontier under non-strict endpoint).For fixed( α,δ)letσε,δbe the Gaussian
DP scale. Define
εmin(α,δ) = inf/braceleftbigg
ε>0 :E/bracketleftbigg
Q/parenleftbigg1−Sij
σε,δ/parenrightbigg/bracketrightbigg
≤α/bracerightbigg
.(265)
Because the integrand is strictly decreasing inε,ε minis well-defined and can be found by 1-D root finding.
Threshold re-calibration.Let Fibe the CDF of thecleanimpostor scores and fi=F′
iits density. Let τ0
be the operating threshold that achieves the desired pre-privacy target FMR1 −Fi(τ0) =FMR 0. By adding
isotropic Gaussian noise with standard deviation σ, theScoreShieldmechanism smooths the (impostor) score
distribution by convolution, /tildewideFi=Fi∗gσ, wheregσ(x) =σ−1φ/parenleftbig
x/σ/parenrightbig
. To preserve the target FMR after
smoothing, raise the threshold by∆τso that
/tildewideFi(τ0+ ∆τ) =F i(τ0).(266)
Small-noise regime ( σ≪ 1).Assume Fi∈C2in a neighborhood of τ0(i.e.,Fiis twice continuously
differentiable atτ 0) with densityf i(τ0)>0. Gaussian smoothing admits the local expansion
(Fi∗gσ)(t) =F i(t) +σ2
2F′′
i(t) +rσ(t), r σ(t) =o(σ2)asσ→0,(267)
uniformly fortnearτ 0. Evaluate att=τ 0+ ∆τand Taylor–expand in∆τ:
Fi(τ0+ ∆τ) =F i(τ0) +f i(τ0)∆τ+1
2F′′
i(τ0)∆τ2+o(∆τ2),(268a)
F′′
i(τ0+ ∆τ) =F′′
i(τ0) +o(1).(268b)
Plug in to Eq. 266 gives
Fi(τ0) +f i(τ0) ∆τ+1
2F′′
i(τ0) ∆τ2+σ2
2F′′
i(τ0) +o(σ2) +o(∆τ2) =F i(τ0).(269)
Solving for∆τyields
∆τ=−σ2
2F′′
i(τ0)
fi(τ0)+o(σ2) =−σ2
2f′
i(τ0)
fi(τ0)+o(σ2),(270)
and hence∆τ= Θ(σ2). If, additionally,F i∈C4nearτ 0, the remainder sharpens toO(σ4).
64

arXiv preprint, ScoreShield
RemarkG.22.Define the log-pdf slope ℓ(t) =d
dtlogf i(t) =f′
i(t)/fi(t). For right-tail operating thresholds of a
unimodal density we have ℓ(τα)<0, so∆τ >0, matching the intuition that the operating point moves to the
right.10
RemarkG.23 (Gaussian impostor example (exact calibration)).Assume the clean impostor score is standard
normal,Sij∼N (0,1), and the added noise is W∼N (0,σ2). For any interior threshold τ < 1under
the strict rule “ >”, clipping does not affect the decision. The post-privacy impostor tail at threshold τis
Pr[S+W >τ ] =Q/parenleftbig
τ/√
1 +σ2/parenrightbig
. If the pre-privacy operating point satisfies α=Q(τ0), preserving this tail
after noise requires
Q/parenleftigτ0+ ∆τ√
1 +σ2/parenrightig
=Q(τ 0).(271)
SinceQis strictly decreasing, it follows that
∆τ=τ 0/parenleftbig/radicalbig
1 +σ2−1/parenrightbig
.(272)
Expanding√
1 +σ2= 1 +σ2
2−σ4
8+O(σ6)yields
∆τ=σ2
2τ0−σ4
8τ0+O(σ6) =σ2
2τ0+O(σ4).(273)
This matches the small-noise regime∆ τ=−(σ2/2)f′
i(τ0)/fi(τ0) +o(σ2)from Eq. 270, because for the
standard normalf′
i/fi=−τ 0.
RemarkG.24 (Extreme–FMR scaling and tail shape).Recall the small-noise calibration formula∆ τ=
−σ2
2f′
i(τ0)
fi(τ0)+o(σ2)from Eq. 270. For afixedtarget α(hence fixed τ0∈(−1,1)), the coefficient |f′
i(τ0)/fi(τ0)|
is a finite constant and∆ τ= Θ(σ2). Whenα↓0(extreme right tail), the coefficient’s growth is determined
entirely by the impostor tail:
•Quadratic log–tail (Gaussian/sub-Gaussian) regime.If logf i(t) =−ψ(t) +O(1)withψ′(t) = Θ(t)as
t→+∞(e.g., the standard normal, where ψ(t) =t2/2andf′
i/fi=−t), then/vextendsingle/vextendsinglef′
i(τ0)/fi(τ0)/vextendsingle/vextendsingle= Θ(|τ0|)and
∆τ= Θ/parenleftbig
σ2|τ0|/parenrightbig
.(274)
For the standard normal,τ 0= Φ−1(1−α)∼/radicalbig
2 log(1/α), hence
∆τ= Θ/parenleftbig
σ2/radicalbig
log(1/α)/parenrightbig
.(275)
•Bounded support with algebraic vanishing near the endpoint.If scores live in[ −1,1]andfi(t)≍C(1−t)β
ast↑1for someβ >0, thenf′
i(t)/f i(t)∼−β/(1−t)and1−F i(τ0) =α≍C′(1−τ 0)β+1. Consequently,
∆τ∼σ2
2β
1−τ 0= Θ/parenleftbig
σ2α−1/(β+1)/parenrightbig
.(276)
In all cases the dependence on the privacy noise scale remains quadratic in σ. Only the multiplicative constant
changes with how extreme the operating point is, via the local tail shape encoded byf′
i(τ0)/fi(τ0).
Proposition G.25(Exact re-calibration: existence and uniqueness).Let τα∈(−1,1)satisfy FMR clean(τα) =
αas in Definition G.15. Forη∈[0,1−τ α]define
G(η) :=FMR priv(τα+η),(277)
where, forτ∈(−1,1),
FMR priv(τ) =E/bracketleftbigg
Q/parenleftbiggτ−Sij
σ/parenrightbigg/vextendsingle/vextendsingle/vextendsingle/vextendsinglei̸=j/bracketrightbigg
.(278)
Then for anyσ>0:
10The threshold must moveto the right, i.e, become stricter, as noise spreads the scores.
65

arXiv preprint, ScoreShield
(a)Strict endpoint (“ >”).Gis continuous and strictly decreasing on[0 ,1−τα)with limη↑1−ταG(η) =
L(σ)>0andG(1−τα) = 0(at the endpoint). There exists a unique interior solution η⋆∈[0,1−τα)to
G(η) =αiffα∈[L(σ),G(0)]. In other words,
∃∆τ∈[0,1−τ α]withFMR priv(τα+ ∆τ) =α⇐⇒G(0)≥α.(279)
In particular, if G(0)> αthen∆τ∈(0,1−τα). Ifα= 0, the unique solution is η⋆= 1−τα(i.e.,
τ= 1). Ifα<L(σ)there is no solution withτ≤1in strict semantics.
(b)Non–strict endpoint (“ ≥”).Gis continuous and strictly decreasing on[0 ,1−τα]withG(1−τα) =L(σ)>0.
There exists a unique solutionη⋆∈[0,1−τ α]iffα∈[L(σ),G(0)], whereα=L(σ)is attained atτ= 1.
Proof.Forτ∈(−1,1), projection monotonicity and independence of W∼N (0,σ2)implies Pr[/hatwideSij>τ|i̸ =
j] =Pr[Sij+W >τ|i̸ =j] =E[Q((τ−Sij)/σ)|i̸=j]. Fixσ>0. Since∂
∂τQ((τ−s)/σ) =−σ−1φ((τ−s)/σ)
andSij∈[−1,1]a.s., dominated convergence yields
d
dτFMR priv(τ) =−1
σE/bracketleftbigg
φ/parenleftbiggτ−Sij
σ/parenrightbigg/vextendsingle/vextendsingle/vextendsingle/vextendsinglei̸=j/bracketrightbigg
<0, τ∈(−1,1),(280)
soFMR privis continuous and strictly decreasing on( −1,1)and continuous up to τ= 1. With the strict “ >”
rule andSij≤1,FMR priv(1) = 0, hence Gis continuous up to τ= 1and strictly decreasing on[0 ,1−τα)
withG(1−τα) = 0. By the intermediate value theorem, the equation G(η) =αhas a unique solution in
[0,1−τα]iffG(0)≥α. Uniqueness follows from strict decrease.
Definition G.26(Recalibration offset).The recalibration offset∆τis the uniqueη∈[0,1−τ α]satisfying
FMR priv(τα+η) =α.(281)
By Proposition G.25, this is well-defined wheneverG(0)≥α.
RemarkG.27 (Small-noise Law).Under the regularity assumptions stated in the re-calibration section (twice
differentiable impostor CDF withf i(τα)>0), the exact shift obeys the previously derived expansion
∆τ=−σ2
2f′
i(τα)
fi(τα)+o(σ2),(282)
see Eq. 270. Thus exact re-calibration cancels theΘ(σ2)inflation atτ αwith a shift onlyΘ(σ2).
RemarkG.28 (Small-noise Regime (When is G(0)>α?)).Iffi(τα)>0andFiisC2near the clean operating
pointτα, the small-noise expansion
FMR priv(τα) =α+σfi(τα)√
2π+o(σ),(σ→0),(283)
impliesG(0)>αfor all sufficiently smallσ>0. In that case the unique solution satisfies∆τ∈(0,1−τ α).
RemarkG.29 (Conservative sufficient offset).Define, forη≥0,
Iup(η,σ) :=E/bracketleftbigg
1{Sij≤τα+η}Q/parenleftbiggτα+η−Sij
σ/parenrightbigg/vextendsingle/vextendsingle/vextendsingle/vextendsinglei̸=j/bracketrightbigg
.(284)
Applying the upper bound from Corollary G.20 atτ=τ α+ηgives
FMR priv(τα+η)≤FMR clean(τα+η) +I up(η,σ).(285)
Hence anyη≥0satisfying FMR clean(τα+η) +Iup(η,σ)≤αis asufficient(conservative) recalibration offset.
RemarkG.30 (Operational Interpretation).Keeping τ=ταinduces additive inflation of orderΘ( σ2)(Propo-
sition G.19). Exact re-calibration finds the unique∆ τthat restores the target and, by Remark G.27, requires
a shift onlyΘ( σ2). Whenσfi(τα)/√
2πis already below tolerance, re-calibration can be skipped, otherwise
compute∆τvia the root condition Eq. 281 (or use the small-noise approximation Eq. 270 when appropriate.
See Remark G.27).
66

arXiv preprint, ScoreShield
G.3 Effect on the ROC Curve and AUC
LetFgandFidenote the CDFs of the genuine and impostor scores prior to perturbation. We view them as
CDFs on Rby extending constantly outside[ −1,1](i.e.,F(t) = 0fort<− 1andF(t) = 1fort>1). Adding
zero-mean Gaussian noise with varianceσ2smooths both distributions by convolution11:
/tildewideFg=F g∗gσ,/tildewideFi=F i∗gσ, gσ(x) =1
σφ(x/σ), φ(x) =e−x2/2
√
2π.
For a decision thresholdτ∈[−1,1]we write
TPRFg(τ) = 1−F g(τ),FPR Fi(τ) = 1−F i(τ),(286)
and analogouslyTPR/tildewideFg,FPR/tildewideFi.
Theorem G.31(Uniform perturbation bound on the ROC).Assume that for ℓ∈{ g,i}the CDFFℓis
L-Lipschitz on R(equivalently, its density fℓexists a.e. and satisfies0 ≤fℓ≤L), and let/tildewideFℓ=Fℓ∗gσbe its
Gaussian smoothing. Then for every thresholdτ∈[−1,1],
/vextendsingle/vextendsingleTPR/tildewideFg(τ)−TPR Fg(τ)/vextendsingle/vextendsingle≤Lσ/radicalig
2
π,(287a)
/vextendsingle/vextendsingleFPR/tildewideFi(τ)−FPR Fi(τ)/vextendsingle/vextendsingle≤Lσ/radicalig
2
π.(287b)
Consequently, for the Hausdorff distance induced by theℓ ∞metric on[0,1]2,
dH/parenleftbig
ROC(/tildewideFg,/tildewideFi),ROC(F g,Fi)/parenrightbig
≤Lσ/radicalig
2
π.(288)
Proof.For anyL-Lipschitz CDFFonR,
sup
τ∈R/vextendsingle/vextendsingleF(τ)−(F∗g σ)(τ)/vextendsingle/vextendsingle= sup
τ/vextendsingle/vextendsingleE[F(τ−σZ)−F(τ)]/vextendsingle/vextendsingle≤LE[|σZ|] =Lσ/radicalig
2
π,(289)
Z∼N (0,1). Since TPR = 1−Fgand FPR = 1−Fi, the same bound holds for both coordinates. For the
ROC Hausdorff bound, at each τthe two ROC points differ by at most this amount in each coordinate, so
theℓ∞distance between the two curves is at most the same bound.
RemarkG.32 (Lipschitz on [-1, 1]via constant extension).If FisL-Lipschitz on[−1,1](equivalently, its
density exists a.e. on[ −1,1]with0≤f≤ 1) and we extend Fconstantly outside[ −1,1]by setting F(t) = 0
fort <− 1andF(t) = 1fort >1, then the extension is globally L-Lipschitz on R. Hence the proof of
Theorem G.31 (which applies convolution onR) goes through unchanged.
RemarkG.33 (Hausdorff metric choice).The theorem states the bound using the Hausdorff distance induced
by theℓ∞metric on [0,1]2. If we instead measure Hausdorff distance with the Euclidean metric, we need to
multiply the constant by√
2.
AUC Functional.For CDFsF g,Fisupported on[−1,1],
AUC(F g,Fi) =/integraldisplay1
−1/parenleftbig
1−F g(τ)/parenrightbig
dFi(τ) =/integraldisplay1
−1/parenleftbig
1−F g(τ)/parenrightbig
fi(τ) dτ,(290)
wheneverF iis absolutely continuous with densityf i. Equivalently, if(S g,Si)has a continuous joint law (no
ties),AUC=Pr[S g>Si]. In generalAUC=Pr[S g>Si] +1
2Pr[S g=Si][4, 33].
11Because1{proj (S+W )>τ} =1{S+W >τ} forτ∈(−1,1), both genuine and impostor CDFs are perturbed by Gaussian
convolution.
67

arXiv preprint, ScoreShield
Theorem G.34(AUC Perturbation Bound).Let Fg,Fibe CDFs supported on[ −1,1]. Extend FgtoRby
settingFg(t) = 0fort <− 1andFg(t) = 1fort >1, and assume that this extension is L-Lipschitz. Let
/tildewideFg=F g∗gσand/tildewideFi=F i∗gσwithgσ(x) =σ−1φ(x/σ),φ(x) =e−x2/2/√
2π. Defineη :=Lσ/radicalbig
2/π. Then
/vextendsingle/vextendsingleAUC(/tildewideFg,/tildewideFi)−AUC(F g,Fi)/vextendsingle/vextendsingle≤2η.(291)
In particular,/vextendsingle/vextendsingleAUC(/tildewideFg,/tildewideFi)−AUC(F g,Fi)/vextendsingle/vextendsingle=O(σ),(292)
with constant2L/radicalbig
2/π.
Proof.LetY∼F iand couple/tildewideY∼/tildewideFias/tildewideY=Y+σZ, whereZ∼N (0,1)is independent of Y. Using the
Stieltjes form,
AUC(F g,Fi) =E/bracketleftbig
1−F g(Y)/bracketrightbig
,AUC(/tildewideFg,/tildewideFi) =E/bracketleftbig
1−/tildewideFg(/tildewideY)/bracketrightbig
.(293)
Set
∆:=AUC(F g,Fi)−AUC(/tildewideFg,/tildewideFi).(294)
Adding and subtractingE[1−F g(/tildewideY)]gives
∆AUC=E/bracketleftig
/tildewideFg(/tildewideY)−F g(/tildewideY)/bracketrightig
/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
(A)+/parenleftig
E[1−F g(Y)]−E[1−F g(/tildewideY)]/parenrightig
/bracehtipupleft /bracehtipdownright/bracehtipdownleft /bracehtipupright
(B).(295)
For term(A), the globalL-Lipschitz property ofF ggives
∥Fg−/tildewideFg∥∞= sup
t|Fg(t)−EF g(t−σZ)|≤LσE|Z|=η,(296)
and hence|(A)|≤η. For term(B), the functionh(t) = 1−F g(t)is alsoL-Lipschitz, so
|(B)|=/vextendsingle/vextendsingle/vextendsingleEh(Y)−Eh(/tildewideY)/vextendsingle/vextendsingle/vextendsingle≤LE|Y−/tildewideY|=LσE|Z|=η.(297)
Therefore|∆ AUC|≤|(A)|+|(B)|≤2η, which proves the claim.
Corollary G.35(Privacy–Utility Rate).Let AUC 0:=AUC(Fg,Fi)and AUCσ:=AUC(/tildewideFg,/tildewideFi), and set
η=Lσ/radicalbig
2/π. Then
/vextendsingle/vextendsingleAUCσ−AUC 0/vextendsingle/vextendsingle≤2η= 2Lσ/radicalig
2
π= Θ(σ).(298)
If the(ε,δ)–DP Gaussian mechanism uses σ=c(δ)/ε(e.g.,c(δ) = 2/radicalbig
2 log(2/δ) in our setting), then/vextendsingle/vextendsingleAUCσ−AUC 0/vextendsingle/vextendsingle=O(1/ε). Moreover, Theorem G.31 implies the sameΘ( σ)rate holds pointwise along the
ROC: for everyτ∈[−1,1],
/vextendsingle/vextendsingleTPR/tildewideFg(τ)−TPR Fg(τ)/vextendsingle/vextendsingle≤η,/vextendsingle/vextendsingleFPR/tildewideFi(τ)−FPR Fi(τ)/vextendsingle/vextendsingle≤η.(299)
Corollary G.36(EER Perturbation).Let
EER 0:= inf
τ∈[−1,1]max/braceleftbig
FPRFi(τ),1−TPR Fg(τ)/bracerightbig
,(300)
EERσ:= inf
τ∈[−1,1]max/braceleftbig
FPR/tildewideFi(τ),1−TPR/tildewideFg(τ)/bracerightbig
,(301)
be the equal–error rates before and after smoothing. Under the hypotheses of Theorem G.31, with η=Lσ/radicalbig
2/π,
/vextendsingle/vextendsingleEERσ−EER 0/vextendsingle/vextendsingle≤η.(302)
68

arXiv preprint, ScoreShield
Proof.By Theorem G.31, for every τ∈[−1,1],|FPR/tildewideFi(τ)−FPRFi(τ)|≤ηand|TPR/tildewideFg(τ)−TPRFg(τ)|≤η.
Hence, setting e0(τ) =max{FPR Fi(τ),1−TPRFg(τ)}andeσ(τ) =max{FPR/tildewideFi(τ),1−TPR/tildewideFg(τ)}, we have
for eachτ:
|eσ(τ)−e 0(τ)| ≤η,(303)
because the max is1-Lipschitz in the ℓ∞norm. Taking infima over τon both sides gives EERσ=infτeσ(τ)≤
infτ(e0(τ) +η) =EER 0+η, and symmetrically EER 0≤EERσ+η. Combining the two inequalities complete
the proof.
In face recognition, utility is often evaluated at extremely low false-positive rates (FPR), where the global
AUC can mask performance in the left tail of the ROC. We therefore study the ROC over and FPR-restricted
range[0,α]forα∈(0,1). For a threshold τ∈[−1,1], define TPRFg(τ)and FPRFi(τ)as in Eq. 286. Since the
set-valued ROC may contain vertical segments, we work with the standard ROC upper envelope
R(u;F g,Fi):= sup/braceleftig
TPRFg(τ) :FPR Fi(τ)≤u/bracerightig
, u∈[0,1].(304)
The (unnormalized) partial area under the ROC curve [15, 44, 46] up toαis
pAUC(α;F g,Fi):= AUC [0,α](Fg,Fi) =/integraldisplayα
0R(u;F g,Fi) du.(305)
Corollary G.37(Partial-AUC perturbation).Assume the hypotheses of Theorem G.31 and set η:=Lσ/radicalbig
2/π.
Then, for everyα∈(0,1),
/vextendsingle/vextendsingle/vextendsinglepAUC(α;/tildewideFg,/tildewideFi)−pAUC(α;F g,Fi)/vextendsingle/vextendsingle/vextendsingle≤αη+ min{α,η} ≤(α+ 1)η.(306)
In particular, /vextendsingle/vextendsingle/vextendsinglepAUC(α;/tildewideFg,/tildewideFi)−pAUC(α;F g,Fi)/vextendsingle/vextendsingle/vextendsingle=O(σ),(307)
with constant(α+ 1)L/radicalbig
2/π.
Proof.WriteR 0(u):=R(u;F g,Fi)andRσ(u):=R(u;/tildewideFg,/tildewideFi). By Theorem G.31, for everyτ∈[−1,1],
/vextendsingle/vextendsingleTPR/tildewideFg(τ)−TPR Fg(τ)/vextendsingle/vextendsingle≤η,/vextendsingle/vextendsingleFPR/tildewideFi(τ)−FPR Fi(τ)/vextendsingle/vextendsingle≤η.(308)
We claim that for allu∈[0,1],
Rσ(u)≤R 0(min{1,u+η}) +η, R 0(u)≤R σ(min{1,u+η}) +η.(309)
Toprovethefirstinequality, fix uandanyτsuchthat FPR/tildewideFi(τ)≤u. ThenbyEq.308wehave FPRFi(τ)≤u+η,
and also TPR/tildewideFg(τ)≤TPRFg(τ) +η. Taking the supremum over such τyieldsRσ(u)≤R 0(min{ 1,u+η}) +η.
The second inequality follows symmetrically by swapping(F g,Fi)with(/tildewideFg,/tildewideFi).
Integrating Eq. 309 overu∈[0,α]gives
/integraldisplayα
0Rσ(u)du≤/integraldisplayα
0R0(min{1,u+η})du+αη.(310)
Define the clipped extension ¯R0(v):=R 0(min{1,v})forv≥0. Then
/integraldisplayα
0R0(min{1,u+η})du=/integraldisplayα+η
η¯R0(v)dv.(311)
Since0≤ ¯R0≤1, we have/integraldisplayα+η
η¯R0(v)dv≤/integraldisplayα
0R0(v)dv+ min{α,η}.(312)
69

arXiv preprint, ScoreShield
Indeed, ifη≤α, then
/integraldisplayα+η
η¯R0(v)dv−/integraldisplayα
0R0(v)dv=/integraldisplayα+η
α¯R0(v)dv−/integraldisplayη
0R0(v)dv(313)
≤η.(314)
Ifη>α, then the left integral is at mostα, and hence
/integraldisplayα+η
η¯R0(v)dv≤/integraldisplayα
0R0(v)dv+α.(315)
Combining the two cases gives
/integraldisplayα
0R0(min{1,u+η})du≤pAUC(α;F g,Fi) + min{α,η}.(316)
Therefore,
pAUC(α;/tildewideFg,/tildewideFi)≤pAUC(α;F g,Fi) +αη+ min{α,η}.(317)
The reverse inequality follows by swapping(F g,Fi)and(/tildewideFg,/tildewideFi).
G.4 Rate-Optimal Upper and Matching Bound
Consider the clean impostor false–match rate FMR clean(τ):=Pr/bracketleftbig
Sij> τ|i̸ =j/bracketrightbig
. For a one-shot release
/hatwides=M(E,q)∈[−1,1]non a fixed probeqand galleryEof size n(with the probe not enrolled, hence all n
comparisons are impostors), the corresponding gallery–conditional FMR is
FMRM
E(τ):=Pr/bracketleftig
/hatwideSIJ>τ|I̸=J,E/bracketrightig
=1
nn/summationdisplay
j=1Pr/bracketleftbig
/hatwidesj>τ/vextendsingle/vextendsingleE/bracketrightbig
,(318)
whereτ∈[−1,1]is a verification threshold. We keep the strict rule “ >” as we discussed before. At τ= 1the
strict rule gives Pr/bracketleftig
/hatwideS >1/bracketrightig
= 0. WhenE∼E′differ in exactly one enrollment (neighbors), consider FMRM
E′(t)
for the corresponding FMR underE′. Also note that according to the regime (i) setup in Sec. 3.1, we assume
that the probe (query) is not enrolled (so allncoordinates are impostor comparisons).
Theorem G.38(Universal FMR perturbation bounds under DP).Let Mbe any one-shot( ε,δ)–DP
mechanism that releases, for a fixed probeq, the full vector of clipped similarity scores /hatwides=M(E,q)∈[−1,1]n
for a galleryEof size n. For neighboring galleriesE ∼E′, define the gallery-conditional FMR of Mby
Eq. 318. For neighboring galleriesE∼E′the following hold.
(a) Universal bound (no structural condition).
sup
τ∈[−1,1]/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle≤TV(P vec,Qvec)≤tanh/parenleftbig
ε/2/parenrightbig
+δ≤ε
2+δ(ε≤1),(319)
wherePvec:=L(/hatwides|E)andQvec:=L(/hatwides|E′), and TV(P,Q) =supA|P(A)−Q(A)|is total variation
distance.
(b)Sharpened bound under marginal stability. Assume the following marginal–stability condition: IfE ∼E′
differ only at index i, then for every j̸=i,L(/hatwidesj|E) =L(/hatwidesj|E′). LetPi:=L(/hatwidesi|E)andQi:=L(/hatwidesi|E′)
are the one-dimensional marginals of the changed coordinate /hatwidesiunderEandE′, respectively. Then
sup
τ∈[−1,1]/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle≤TV(Pi,Qi)
n≤tanh(ε/2) +δ
n≤ε/2 +δ
n(ε≤1).(320)
Proof.Forτ∈[−1,1]define the coordinate exceedance sets Aτ
j:={x∈ [−1,1]n:xj>τ},j= 1,...,n. By
definition in Eq. 318,
FMRM
E(τ) =1
nn/summationdisplay
j=1Pvec(Aτ
j),FMRM
E′(τ) =1
nn/summationdisplay
j=1Qvec(Aτ
j).(321)
70

arXiv preprint, ScoreShield
Hence
/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle=/vextendsingle/vextendsingle/vextendsingle1
nn/summationdisplay
j=1/parenleftbig
Pvec(Aτ
j)−Q vec(Aτ
j)/parenrightbig/vextendsingle/vextendsingle/vextendsingle (322a)
≤1
nn/summationdisplay
j=1/vextendsingle/vextendsinglePvec(Aτ
j)−Q vec(Aτ
j)/vextendsingle/vextendsingle≤1
nn/summationdisplay
j=1TV(P vec,Qvec)(322b)
=TV(P vec,Qvec).(322c)
Taking supτ∈[−1,1]preserves the bound, proving the first inequality in Eq. 319. Because Mis(ε,δ)–DP
on neighboring inputs, the standard DP ⇒TV conversion gives TV(Pvec,Qvec)≤tanh (ε/2) +δ12. Finally,
tanh(x)≤xforx≥0yields the small–εsimplification.
For part (B), let ibe the unique index whereEandE′differ. By marginal stability, Pvec(Aτ
j) =Qvec(Aτ
j)for
allj̸=i, hence
FMRM
E(τ)−FMRM
E′(τ) =1
n/parenleftig
Pvec(Aτ
i)−Q vec(Aτ
i)/parenrightig
=1
n/parenleftig
Pi/parenleftbig
(τ,1]/parenrightbig
−Qi/parenleftbig
(τ,1]/parenrightbig/parenrightig
,(323)
where we used that /hatwidesi∈[−1,1]implies{/hatwidesi>τ} ={/hatwidesi∈(τ,1]}={/hatwidesi∈(τ,∞)}. Also note total variation is
the supremum over all Borel sets, so taking the supremum over this particular subfamily of half-lines can only
underestimateTV. Therefore
sup
τ∈[−1,1]/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle≤1
nsup
τ∈R/vextendsingle/vextendsinglePi/parenleftbig
(τ,∞)/parenrightbig
−Qi/parenleftbig
(τ,∞)/parenrightbig/vextendsingle/vextendsingle≤TV(Pi,Qi)
n.(324)
Since/hatwidesiis a measurable post–processing of the DP output /hatwides, the pair(P i,Qi)also satisfies(ε,δ)–DP, hence
TV(Pi,Qi)≤tanh(ε/2) +δand Eq. 320 follows.
RemarkG.39 (Marginal Stability).The condition doesnotrequire independence across coordinates and
allows shared internal randomness. A sufficient formulation is: there exist measurable maps Kjand a
data–independent random seed Usuch that/hatwidesj=Kj(sj,U)for eachj. Then changing one enrollment (hence
only one clean sifor a non–enrolled probe) leaves the marginals of /hatwidesjunchanged for all j̸=i. Mechanisms
such as per–coordinate additive noise followed by clipping (e.g., Gaussian noise) satisfy this property.
RemarkG.40 (Half-lines).Since outputs are clipped to[ −1,1], the exceedance event {/hatwidesj> τ}equals
{/hatwidesj∈(τ,1]}, which is the same as {/hatwidesj∈(τ,∞)}because there is no mass above1. Thus using( τ,∞)is a
notational convenience for survivor sets and does not affect probabilities or suprema.
RemarkG.41 (ROC-level Stability).The bounds holduniformlyover τ∈[−1,1], so the entire impostor ROC
(equivalently, the false–match rate curve across thresholds) changes by at most O(ε+δ)between neighboring
gallerieswithoutstructure, and by at mostO((ε+δ)/n)under marginal stability.
Proposition G.42(Achievability by the Gaussian ScoreShield mechanism).Consider regime (i) with a fixed
(non-enrolled) probeq ∈Rdand galleryE∈Rn×d. Lets=Eq∈[−1,1]nbe the clean probe-to-enrollment
similarity vector, and let the DP mechanism bes′=s+w,w∼N(0,σ2In),/hatwides=proj[−1,1]n(s′). Assume
the Gaussian calibration for( ε,δ)–DP with regime (i) sensitivity∆ f,2= 2, namely σ=2√
2 log(2/δ)
ε. Fix
neighboring galleriesE∼E′that differ only at indexi= 1and satisfy
s(E)
1= +1, s(E′)
1=−1, s(E)
j=s(E′)
j=aj∈[−1,1] (j= 2,...,n),(325)
with arbitrary constants aj. Denote the gallery–conditional impostor FMR of this mechanism by FMRM
E(τ).
Then the FMR gap admits the exact identity
sup
τ∈[−1,1]/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle=1
n/parenleftig
2 Φ(1/σ)−1/parenrightig
=1
nerf/parenleftig1
σ√
2/parenrightig
.(326)
12Forδ= 0(pure DP) the inequalityTV≤tanh(ε/2)is tight. For(ε,δ)–DP a standard extension givesTV≤tanh(ε/2) +δ.
71

arXiv preprint, ScoreShield
Moreover, forε≤1andδ∈(0,0.1]there exist absolute constants0<c<C <∞such that
c
n·ε/radicalbig
log(2/δ)≤sup
τ∈[−1,1]/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle≤C
n·ε/radicalbig
log(2/δ),(327)
and hence, for any fixed δ∈(0,0.1]the gap isΘ( ε/n), matching the DP upper bound in Theorem G.38 up to
constants.
Proof.BecauseEandE′differ only at index1and the mechanism iscoordinatewiseperturbation plus
coordinatewiseprojection, the coordinates j≥2have identical distributions underEandE′and thus cancel
in the FMR difference, irrespective of the valuesa j. Therefore, for every thresholdτ∈[−1,1],
FMRM
E(τ)−FMRM
E′(τ) =1
nn/summationdisplay
j=1/parenleftig
Pr[/hatwides(E)
j>τ]−Pr[/hatwides(E′)
j>τ]/parenrightig
(328a)
=1
n/parenleftig
Pr[/hatwides(E)
1>τ]−Pr[/hatwides(E′)
1>τ]/parenrightig
.(328b)
By projection monotonicity (see Eq. 236), for anyinteriorthreshold τ < 1and any scalar S∈ [−1,1],
1{proj[−1,1] (S+W)>τ} =1{S+W >τ}. The optimizer we will obtain satisfies τ⋆= 0<1, hence clipping
does not alter the exceedance events at the optimizer and we may analyze theunclippedGaussian convolutions.
UnderEandE′, the first coordinate laws (before projection) are N(+1,σ2)andN(−1,σ2), respectively.
Thus, for anyτ,
∆1(τ):=Pr[/hatwides(E)
1>τ]−Pr[/hatwides(E′)
1>τ] = Φ/parenleftig1−τ
σ/parenrightig
−Φ/parenleftig−1−τ
σ/parenrightig
.(329)
By differentiation we have:
∆′
1(τ) =1
σ/parenleftig
ϕ/parenleftbig−1−τ
σ/parenrightbig
−ϕ/parenleftbig1−τ
σ/parenrightbig/parenrightig
,(330)
withϕthe standard normal pdf (even and strictly decreasing on[0 ,∞)). Hence∆′
1(0) = 0and∆′
1(τ)>0for
τ <0while∆′
1(τ)<0forτ >0, so∆ 1(τ)is maximized atτ⋆= 0. Therefore
sup
τ∈[−1,1]∆1(τ) = ∆ 1(0) = Φ(1/σ)−Φ(−1/σ) = 2 Φ(1/σ)−1 = erf/parenleftig1
σ√
2/parenrightig
.(331)
Dividing byngives the exact identity Eq. 326. Forx≥0,
2√πxe−x2≤erf(x)≤2√πx.(332)
Withx= 1/(σ√
2)andε≤1,δ≤0.1implyσ≥1, yielding
√
2√π·e−1/(2σ2)
σ≤erf/parenleftig1
σ√
2/parenrightig
≤1√
2π·2
σ.(333)
Usinge−1/(2σ2)≥e−1/2forσ≥1and substitutingσ= 2/radicalbig
2 log(2/δ)/εgives
e−1/2
2√π·1
n·ε/radicalbig
log(2/δ)≤sup
τ∈[−1,1]/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle≤1
2√π·1
n·ε/radicalbig
log(2/δ).(334)
For any fixedδ∈(0,0.1], the factor1//radicalbig
log(2/δ)is a constant, hence the rate isΘ(ε/n).
RemarkG.43 (Consistency with the DP upper bound and marginal stability).The perturb–then–project
release in Proposition G.42 iscoordinatewise(add noise and project per coordinate), hence it satisfies the
72

arXiv preprint, ScoreShield
marginal–stability assumption of Theorem G.38 (b). Combining Eq. 320 with the exact identity Eq. 326 yields
the sandwich
1
nerf/parenleftig1
σ√
2/parenrightig
/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Prop. G.42≤sup
τ∈[−1,1]/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle≤tanh(ε/2) +δ
n/bracehtipupleft/bracehtipdownright/bracehtipdownleft/bracehtipupright
Thm. G.38 (b).(335)
Withσ=2√
2 log(2/δ)
εandε≤ 1,δ∈ (0,0.1], the left-hand side equalsΘ/parenleftbig
ε/(n/radicalbig
log(2/δ) )/parenrightbig
while the
right-hand side is O(ε/n). Thus, for fixed δ, the marginal–stability upper bound in Theorem G.38 (b) is
rate-tightup to constants. By contrast, without marginal stability, Theorem G.38 (a) gives the universal
mechanism–independent bound O(ε+δ)(no1/nfactor), which the Gaussian mechanism trivially satisfies
but does not saturate.
RemarkG.44 (Ratetightnessandtheminimaxviewpoint).Foraone–shotmechanism M, definetheworst–case
FMR disturbance
Gap(M) := sup
E∼E′sup
τ∈[−1,1]/vextendsingle/vextendsingleFMRM
E(τ)−FMRM
E′(τ)/vextendsingle/vextendsingle.(336)
Theorem G.38 (a) gives the universal mechanism–independent bound Gap(M)≤tanh (ε/2) +δ=O(ε+δ)
forε≤1, and under marginal stability Theorem G.38 (b) sharpens this to Gap(M)≤(tanh(ε/2) +δ)/n=
O((ε+δ)/n). Proposition G.42 showsexistentialrate tightness:
Gap(M) =1
nerf/parenleftig1
σ√
2/parenrightig
= Θ/parenleftig1
n·ε/radicalbig
log(2/δ)/parenrightig
, σ=2/radicalbig
2 log(2/δ)
ε,(337)
hence Gap(M) = Θ(ε/n)for fixedδ. This shows that the marginal–stability upper bound in Theorem G.38 (b)
israte-tightup to constants. Moreover, a per–coordinate randomized–response release (one bit per coordinate
with(ε,0)–DP) also satisfies marginal stability and achieves Gap(M) =1
ntanh(ε/2) = Θ(ε/n)on a suitable
neighboring pair, i.e., without the√logfactor. In contrast, the universal part (a) without marginal stability
provides only the mechanism–independent bound O(ε+δ)(no1/nfactor), which the above mechanisms
trivially satisfy but do not saturate.
RemarkG.45 (Absence of a mechanism–uniform lower bound).There is no positive lower bound that holds
uniformly over all(ε,δ)–DP one–shot mechanisms. In particular,
inf
M(ε,δ)–DPGap(M) = 0.(338)
Consider a data–independent mechanism that outputs a random vector with a fixed law (e.g., i.i.d. N(0,1)),
regardless of the input. Such a mechanism is(0 ,0)–DP and therefore( ε,δ)–DP for all parameters. Its
gallery–conditional FMRs coincide under any neighboring galleries, whence Gap(M) = 0. Consequently,
statements asserting Gap(M)≥cε/nforeveryDP mechanism do not hold. By contrast, Proposition G.42
and the randomized–response example show that thereexistnatural mechanisms and neighboring pairs
achieving Gap(M) = Θ(ε/n)(for fixed δ), while Theorem G.38 (b) provides the matching O(ε/n)upper
bound under marginal stability. These results characterize the achievable rate up to constant factors, but not
via a mechanism–uniform lower bound.
73

arXiv preprint, ScoreShield
H Supplementary Details for Regime (i): DP-FR
H.1 Experimental Setup
Domain–restricted sensitivity.Assume that all admissible record–query pairs satisfy the public margin
constraint⟨e,q⟩∈[cmin,1],cmin∈(−1,1). Consider the score-vector release fquery(E,q) =Eq, underrow-
replacement adjacency, where one row ofEmay change and ∥e∥2=∥q∥2= 1. Then neighboring databases
differ in exactly one score coordinate, and the corresponding effectiveℓ 2sensitivity satisfies
∆eff
query≤sup
a,b∈[c min,1]|a−b|= 1−c min.(339)
In our audits across several face-recognition backbones and datasets, off-diagonal cosine similarities seldom took
highly negative values: the minimum observed negative similarity was typically in the range[ −0.45,−0.40],
and the mean over negative entries was approximately −0.06, with about one quarter of off-diagonal pairs
being negative.13Motivated by this empirical structure, we also study the reduced sensitivity
∆eff
query = 1−c min (340)
as a utility model under the stated public margin restriction. For example, the conservative choice cmin=−0.5
gives∆eff
query = 1.5, which is often more consistent with the observed privacy–utility degradation curves
than the worst-case value∆ = 2. Unless the margin restriction is enforced as a public precondition of the
mechanism, however, all formal DP guarantees in this paper remain calibrated with the worst-case global
sensitivity∆ = 2.
Configurations compared (calibration ×sensitivity).To separate the effect of the calibration rule
from the effect of the sensitivity choice, we evaluate the following four configurations:
•Conservative sufficient calibration with worst-case sensitivity: thestandardsufficientbound σ= ∆/radicalbig
2 ln(2/δ)/ε
with∆ = 2(worst-case for our FR mechanism).
•Conservative sufficient calibration with domain-restricted sensitivity: the same sufficient bound, but with
∆ = 1−c minunder the public margin constraint⟨e,q⟩≥c min.
•Analytic calibration [ 3] with worst-case sensitivity: σis the smallest solution of δAG(ε,∆/σ) =δwith
∆ = 2.
•Analytic calibration [ 3] with domain-restricted sensitivity: the same analytic calibration, but with∆ =
1−c minunder the margin restriction.
In the two domain-restricted configurations, both methods use the same reduced sensitivity∆ = 1 −cmin, so
any performance difference isolates the effect of the calibration rule itself. Note that all formal privacy claims
are calibrated with the worst-case value∆ = 2; the reduced-sensitivity results are reported to interpret utility
under the stated margin model. For some figures, we additionally plot a classical reference calibration with
constantc= 1.25, alongside the conservative sufficient calibration and the analytic calibration.
Impostor–genuine score sets.We evaluate two types of impostor–genuine (imp–gen) score sets.
(i) Real FR scores.We compute cosine-similarity scores on LFW using three off-the-shelf FR pipelines and
use all available genuine and impostor comparisons from each run (about3 ,000pairs per class in our setup):
(a)lfw_arc101_webface4m: ArcFace with an IR-101 backbone pretrained onWebFace4M.
(b)lfw_arc50_casia: ArcFace with an IR-50 backbone pretrained onCasia.
(c)lfw_ada_rdigi1mcodeformer : AdaFace with an IR-101 backbone pretrained onDigiFace-1M, with
CodeFormerrestoration before embedding.
13For a representative100k ×100ksimilarity matrix, we observed approximately25%negative off-diagonal entries, mean over
negatives approximately −0.066, and minimum approximately −0.424. Exact values vary across backbones and datasets. We
therefore treat these numbers as descriptive rather than universal.
74

arXiv preprint, ScoreShield
(a)
 (b)
 (c)
Figure H.1.Real impostor—genuine score distributions on LFW. Each panel overlays the impostor (blue) and
genuine (orange) cosine-similarity histograms for one FR pipeline. (a) ArcFace with an IR-101 backbone pretrained on
WebFace4M( lfw_arc101_webface4m ). (b) ArcFace with an IR-50 backbone pretrained onCasia( lfw_arc50_casia ).
(c) AdaFace with an IR-101 backbone pretrained onDigiFace-1M, with inputs restored byCodeFormerbefore embedding
(lfw_ada_rdigi1mcodeformer). Each run yields roughly3,000impostor and3,000genuine pairs.
Histograms of these real imp–gen score distributions are shown in Fig. H.1.
(ii) Synthetic FR-like scores.To probe behavior under controlled distributional shapes, we generate matched
imp–gen score sets with N= 200,000i.i.d. samples per side, using independent random seeds. To avoid
boundary atoms, all scores are constrained to the open interval( −1,1). Hence if Xis drawn from a base law
F, we retain only draws in( −1,1), i.e.,Xtr∼F(·|−1<X < 1 ), implemented by accept–reject sampling.
We report the rejection rate
rrej:= 1−n
Nprop,
that is, the fraction of proposals discarded by truncation. This quantifies the mass that the untruncated law
assigns outside(−1,1)and ensures that the synthetic histograms have no spikes at±1.
Impostor families.We use four synthetic impostor families, each truncated to(−1,1):
(a) Gaussian:X∼N(0,σ).
(b) Mixture:X∼(1−p)N(0,σ 1) +pN(µ 2,σ2).
(c)Student-t:X=sTν,s=σ/radicalbig
(ν−2)/ν whereTνis standard tν, so that sd(X) =σbefore truncation (for
ν >2).
(d)Symmetric Beta: draw Z∼Beta (a,a)on[0,1]and map X= 2Z−1. This law already lies in[ −1,1]; we
numerically nudge the samples into(−1,1)to avoid exact endpoints.
Matching genuine families.For each impostor file, we generate a matched genuine file (same N) with mass
shifted toward higher similarity while preserving the basic family shape:
•Right-shifted Gaussian:G∼N(µ,σ)withµ∈[0.6,0.7].
•Right-shifted mixture:a dominant high-mean component plus a small low-mean tail to mimic difficult
genuine pairs.
•Shifted Student-t:G=µ+sT νwith the mass centered in the genuine region.
•Skewed Beta:drawZ∼Beta(a,b)witha≥b, then mapG= 2Z−1, concentrating mass near+1.
Representative synthetic imp–gen histograms are shown in Figure H.2.
Operating points and endpoint semantics.A pre-privacy target FMR =α∈{ 10−2,10−3}determines
the clean operating threshold τα(Definition G.15). We then applyScoreShieldwith Gaussian scale σ,
calibrated from( ε,δ)and the chosen sensitivity∆, and evaluate either strict endpoint semantics (‘ >’) or
non-strict endpoint semantics (‘≥’) atτ= 1.
Privacy grid and reproducibility.We use δ∈{ 10−8,10−6,10−5},ε∈{ 0.5,1,2,3,5,8,10,15,20,30,40}.
All experiments use fixed random seeds, single-threaded numerical routines, and non-interactive rendering.
75

arXiv preprint, ScoreShield
(a)
 (b)
(c)
 (d)
Figure H.2.Synthetic impostor–genuine score distributions. Each panel overlays impostor (blue) and genuine (orange)
histograms for N= 200,000samples per side, using independent random seeds. (a) Impostor: Gaussian with σ= 0.20;
Genuine: right-shifted Gaussian (default µ≈0.65,σ≈0.20). (b) Impostor: symmetric Beta with a= 5mapped via
x= 2z−1; Genuine: skewed Beta (default a= 9,b= 2) mapped to concentrate mass near+1. (c) Impostor: mixture
(1−p)N(0,σ1) +pN(µ2,σ2)withp= 0.10,µ2= 0.30,σ1= 0.25,σ2= 0.20; Genuine: right-shifted mixture with a
dominant high-mean component and a small low-mean tail. (d) Impostor: Student-twithν= 10and pre-truncation
standard deviation0 .25; Genuine: shifted Student- t. The generator reports the rejection rate rrej:= 1−n
Npropfor each
file. In all displayed settingsr rej≪1, indicating negligible boundary mass at±1.
H.2 Performance Analysis
DiscussionofFigureH.3.FigureH.3compares FMR (τ)underseveralprivacycalibrationsatfixed δ= 10−6
and target level α= 10−2, using strict endpoint semantics. For fixed( ε,δ,∆), the analytic calibration chooses
the smallest Gaussian noise scale σthat satisfies differential privacy, whereas the conservative sufficient
bound uses a larger σ. Consequently, in the threshold range relevant to small- αoperation, the analytic
curves lie below the corresponding conservative curves. Replacing the worst-case sensitivity∆ = 2by the
domain-restricted value∆ = 1 −cmin= 1.5withcmin=−0.5reduces the required noise for both calibration
rules and further improves the right-tail behavior.
Asεincreases from5to10to15, the noise scale decreases and the private curves approach the clean curve.
The qualitative ordering remains the same across all displayed score families. Under strict endpoint semantics,
feasibility is governed by the left limit L(σ):=limτ↑1FMR priv(τ). IfL(σ)≥α, then no threshold τ <1
attains the target level α. Thus, for fixed δandα, feasibility is lost first under conservative calibration with
worst-case sensitivity and retained longest under analytic calibration with reduced sensitivity. This effect
is especially visible for the Student- timpostor family, whose right tail is harder to suppress after Gaussian
perturbation.
For a fixed privacy budget(ε,δ), analytic calibration and, when justified, domain-restricted sensitivity both
improve the attainable operating region relative to the conservative worst-case baseline.
Discussion of Figs. H.4–H.6.Across the three figure sets we fix the target impostor rate at α= 10−2
withδ= 10−5and sweepε∈{5,10,20,30}. The blue curves show the cleanFMR(τ), and the orange curves
show the post-privacy FMR priv(τ). The dashed vertical line marks the clean operating threshold τα, while the
dash-dotted line marks the re-calibrated threshold τα+ ∆τwhen an interior solution exists. All panels use
strict endpoint semantics atτ= 1.
Increasingεreduces the Gaussian noise scale σ, so the private curve contracts monotonically toward the clean
76

arXiv preprint, ScoreShield
curve. Feasibility is controlled by L(σ):=limτ↑1FMR priv(τ). Under strict endpoint semantics, an interior
threshold achievingαexists only whenL(σ)<α. IfL(σ)≥α, then no thresholdτ <1attains the target.
Sensitivity modeling shifts this feasibility boundary. In Figs. H.4 and H.6, the analytic calibration uses
the domain-restricted sensitivity∆ = 1 −cmin= 1.5,cmin=−0.5, whereas Figure H.5 uses the worst-case
sensitivity∆ = 2. The smaller∆produces a smaller σat the same( ε,δ), and therefore reaches feasibility at
a lower privacy cost. The backbone and training set mainly affect the clean operating threshold and the local
shape of the impostor tail nearτ α; the primary determinant of feasibility remains the noise scaleσ(ε,δ,∆).
For ArcFace-101/WebFace4M (Figure H.4), the target is infeasible for ε∈{ 5,10,20}and becomes feasible
atε= 30. AdaFace/RDigi1M/CodeFormer (Figure H.6) shows the same feasibility pattern. For ArcFace-
50/Casia under worst-case sensitivity (Figure H.5), the noise inflation is larger, and even at ε= 30the left
limit remains slightly above α, so the target is still infeasible. In that case, feasibility would require a larger ε,
a largerδ, or a less stringent target levelα.
When feasibility holds, the required correction τα∝⇕⊣√∫⊔≀→τα+ ∆τremains away from the saturation boundary
τ= 1, which helps preserve genuine-match utility.
Discussion of Figure H.7.For fixed target level αand fixed endpoint semantics, the threshold correction
∆τ(σ) =τpriv(α;σ)−τclean(α)depends only on the clean score distribution and the Gaussian noise scale σ.
Hence different privacy calibrations agree whenever they induce the same σ; they differ only through the map
(ε,δ,∆)∝⇕⊣√∫⊔≀→σ.
Under strict endpoint semantics, FMR priv(τ)is continuous and strictly decreasing on( −1,1), withL(σ):=
limτ↑1FMR priv(τ)>0. An interior threshold achieving αexists if and only if α > L (σ). This yields the
feasibility boundary σcrit(α):=inf{σ :L(σ)≥α}. Forσ≥σ crit(α), the target αis infeasible under strict
endpoint semantics.
In the small-noise regime, Proposition G.19 gives the local expansion∆ τ(σ) =−σ2
2f′
i(τα)
fi(τα)+o(σ2), so the
leading correction is quadratic in σ. Its magnitude is governed by the local log-derivative −f′
i(τα)
fi(τα)of the
impostor density at the clean operating point, rather than by tail class alone. Across datasets, the qualitative
shape is similar, but the vertical offset and the feasibility boundary are distribution-dependent.
Practically, we therefore report∆ τ(σ)using (i) a shared σ-grid across panels, (ii) the clean threshold ταas a
reference, (iii) a shaded infeasible region where α≤L (σ), and, when useful, (iv) a secondary top axis mapping
σtoεunder a selected calibration. That secondary axis is only interpretive; it does not alter the∆ τ(σ)curve
itself.
Discussion of Figure H.8 ( ε-sweep of∆ τ).Figure H.8 reports the threshold correction∆ τas a function
of the privacy budget εfor two sensitivity models. For each backbone, the top row uses the worst-case
score-vector sensitivity∆ = 2, whereas the bottom row uses the domain-restricted sensitivity induced by
cmin=−0.5, namely∆ = 1−c min= 1.5.
Across the three LFW-based score sets,∆ τdecreases as εincreases. This follows from the monotonic decrease
of the Gaussian noise scale σ(ε,δ,∆)withεat fixed(δ,∆). Among feasible operating points at the same
(ε,δ,∆), the analytic Gaussian calibration gives the smallest noise scale among the displayed calibrations and
therefore the smallest required threshold correction. The conservative sufficient calibration gives the largest
displayed correction, while the intermediate classical reference curve lies between these two calibrations.
Under strict endpoint semantics, feasibility at a target false-match rate αis determined by the limiting
privatized false-match rate at the upper endpoint. If limτ↑1FMR priv(τ)≥α, then no admissible threshold
τ <1attains the target. In Fig. H.8, such cases are marked by inverted triangles. The critical noise level for
feasibility depends on the clean impostor-score distribution and on α; the calibration rule determines which
noise level corresponds to a given(ε,δ,∆).
Changing the sensitivity from∆ = 2to∆ = 1 .5reduces the Gaussian noise scale σ(ε,δ,∆)by the factor
1.5/2 = 0.75for each displayed calibration rule, at fixed( ε,δ). Therefore, for the same backbone, calibration
rule, and feasible value of ε, the domain-restricted model requires no larger threshold correction than the
77

arXiv preprint, ScoreShield
worst-case model. The bottom-row curves can nevertheless span a wider range of εvalues, because the
smaller sensitivity makes some lower privacy budgets feasible that are infeasible when∆ = 2. These newly
feasible points occur at smaller εand can still require relatively large threshold corrections. Differences across
backbones are reflected in the level and curvature of the curves, which depend on the local impostor-score
distribution near the clean operating thresholdτ α.
78

arXiv preprint, ScoreShield
(a) Real(LFW, ArcFace-101, Web-
Face4M),ε= 5
(b) Real,ε= 10
 (c) Real,ε= 15
(d) SyntheticGaussian,ε= 5
 (e) SyntheticGaussian,ε= 10
 (f) SyntheticGaussian,ε= 15
(g) SyntheticBeta,ε= 5
 (h) SyntheticBeta,ε= 10
 (i) SyntheticBeta,ε= 15
(j) SyntheticStudent-t,ε= 5
 (k) SyntheticStudent-t,ε= 10
 (l) SyntheticStudent-t,ε= 15
Figure H.3. FMR vs threshold under different calibrations.Each panel plots FMR(τ)for fixedδ= 10−6and target
levelα= 10−2under the strict endpoint rule. Solid curves use the worst-case sensitivity∆ = 2; dashed curves use
the domain-restricted sensitivity∆ = 1 −C minwithCmin=−0.5, hence∆ = 1 .5;Classic(orange/red) uses the
sufficient Gaussian calibrationσ2=cε,δ∆2withcε,δ= 2 log(2/δ)/ε2;Analytic(green/purple) uses the exact analytic
Gaussian calibration ( σ⋆= ∆/ρ⋆). Columns sweep ε∈{ 5,10,15}; rows vary the underlying score distribution (one
real dataset and three synthetic families). In the threshold range relevant to α= 10−2, analytic calibration yields
lower post-privacy FMR than the conservative sufficient bound, and the reduced-sensitivity model further improves
the right-tail behavior. Asεincreases, all private curves move closer to the FMR curve.
79

arXiv preprint, ScoreShield
(a)ε=5
 (b)ε=10
(c)ε=20
 (d)ε=30
Figure H.4. LFW (ArcFace-101 trained on WebFace4M): clean vs. post-privacy FMR under analytic calibration with
domain-restricted sensitivity.Each panel shows the clean and private FMRcurves (log-scale) for a fixed δ= 10−5and
varyingε∈{ 5,10,20,30}, using the analytic Gaussian calibration with∆ = 1 −C min,Cmin=−0.5, hence∆ = 1 .5.
The dotted horizontal line marks the target level α. The dashed vertical line marks the clean operating threshold τα.
The dash-dotted line marks the re-calibrated threshold τα+ ∆τan interior solution exists. The shaded band indicates
a±20%tolerance region aroundα. Endpoint semantics are strict (‘>’) atτ= 1.
80

arXiv preprint, ScoreShield
(a)ε=5
 (b)ε=10
(c)ε=20
 (d)ε=30
Figure H.5. LFW (ArcFace-50 trained on Casia): clean and post-privacy FMR under analytic calibration with
worst-case sensitivity.Same setup as Figure H.4, except that the analytic calibration uses the worst-case sensitivity
∆ = 2. We fix δ= 10−5, varyε∈{ 5,10,20,30}, and annotate the target α(dotted), the clean threshold τα(dashed),
and the re-calibrated threshold τα+ ∆τ(dash-dotted, when feasible). The shaded band indicates a ±20%tolerance
region aroundα. Endpoint semantics are strict (‘>’) atτ= 1.
81

arXiv preprint, ScoreShield
(a)ε=5
 (b)ε=10
(c)ε=20
 (d)ε=30
Figure H.6. LFW (AdaFace + RDigi1M/CodeFormer): clean and post-privacy FMR under analytic calibration with
domain-restricted sensitivity.Same setup as Figure H.4, but using the AdaFace-101 pipeline with RDigi1M pretraining
and CodeFormer restoration. The analytic calibration uses∆ = 1−c min= 1.5,c min=−0.5.
82

arXiv preprint, ScoreShield
(a)ArcFace-R101 (WebFace4M)
 (b)Synthetic Gaussian
(c)ArcFace-R50 (Casia)
 (d)Synthetic Symmetric Beta
(e)AdaFace + RDigi1M/CodeFormer
 (f)Synthetic Student-t
Figure H.7. Sigma sweep: threshold shift∆ τversus noise scale σ.Each panel fixes the target impostor rate α= 0.01
and plots the calibration-agnostic threshold correction∆ τ(σ)for the indicated score model. Panels (a,c,e) use real face-
recognition score sets; panels (b,d,f) use synthetic score families. Shaded regions indicate strict-endpoint infeasibility:
ifL(σ) =limτ↑1FMR priv(τ)≥α, then no threshold τ <1attains the target. For small σ,∆τ(σ) =−σ2
2f′
i(τα)
fi(τα)+o(σ2),
so the leading curvature is determined by the impostor density near the clean operating point τα. A secondary top
axis, when shown, mapsσtoεunder a selected Gaussian calibration; it is included only for interpretation.
83

arXiv preprint, ScoreShield
(a) LFW, AdaFace
RDIGI1M+CodeFormer,∆ = 2
(b) LFW, ArcFace-101 WebFace4M,
∆ = 2
(c) LFW, ArcFace-50 CASIA,∆ = 2
(d) LFW, AdaFace
RDIGI1M+CodeFormer,∆ = 1.5
(e) LFW, ArcFace-101 WebFace4M,
∆ = 1.5
(f) LFW, ArcFace-50 CASIA,∆ = 1 .5
Figure H.8. Threshold correction∆ τversus privacy budget εon LFW-based score sets at fixed δ= 10−6under strict
endpoint semantics.Each panel reports the threshold correction∆ τrequired to attain α= 10−2. The curves inside
each panel correspond to the Gaussian calibration rules shown in the panel legend, using the same sensitivity value for
that panel. The top row uses the worst-case score-vector sensitivity∆ = 2. The bottom row uses the domain-restricted
sensitivity with cmin=−0.5, hence∆ = 1−cmin= 1.5. Filled circles mark feasible operating points; inverted triangles
mark infeasible budgets for whichlim τ↑1FMR priv(τ)≥α.
84

arXiv preprint, ScoreShield
H.3 Benchmarks
Dataset availability.IJB-B and IJB-C provide large-scale, template-based face-verification protocols,
including evaluation at stringent false-positive rates [ 30]. NIST discontinued official distribution of the IJB
challenges on March 14, 2023. We retain IJB-B/C because the publicly obtainable identity-style benchmarks
used in our evaluation—LFW, AgeDB, CFP-FP, CALFW, and CPLFW—primarily use pairwise verification
protocols and do not provide scientifically equivalent large-scale, template-based evaluation at false-positive
rates of10−6,10−5, and10−4. Reproducing the IJB-B/C results therefore requires previously authorized
access to these datasets.
Before discussing the aggregate benchmark tables, we note that very large values of δ(e.g.,δ≥10−1) are
included only asutility ablationsto trace the dependence of performance on the Gaussian noise scale. They
should not be interpreted as standard operational DP regimes.
Privacy-utility tradeoff.Figure H.9 and Tables H.1–H.2 show monotonic recovery of utility as the privacy
budgetεincreases for fixed δ. The curves are steepest for εin the range roughly[15 ,35]and begin to
saturate around ε≈70–100. For fixed ε, increasing δimproves utility because the Gaussian calibration
σ(ε,δ) =∆√
2 log(2/δ)
εdecreases as δincreases. For example, at ε= 70, the IR101 macro average increases
from54.65%atδ= 10−7to86.40%atδ= 0.4, while the ViT-Base macro average increases from55 .35%to
87.69%. Beyondε≈70, the gains become modest, indicating diminishing returns.
Most of the degradation is concentrated at the strict IJB-B/C operating points, especially at FPR = 10−6. The
identity-style sets (LFW, AgeDB, CFP-FP, CALFW, CPLFW) remain much closer to their clean baselines,
with LFW the most robust. For IR101 at( ε,δ) = (70,0.4), LFW drops by1 .88percentage points (from99 .70%
to97.82%), whereas IJB-B at10−6drops by3.49percentage points (from89 .46%to85.97%). ViT-Base
shows the same qualitative pattern: LFW drops by1 .35percentage points (from99 .80%to98.45%), while
IJB-B at10−6drops by2.55percentage points (from87 .12%to84.57%). This matches the decision-flip profile
under Gaussian perturbation: the flip probability is smallest far from the decision threshold and largest in the
extreme right tail.
Infeasible endpoints.Rows with zeros in the IJB-B/C@10−6columns correspond to infeasible operating
points under the calibrated Gaussian noise scale, i.e.,lim τ↑1FMR priv(τ)>10−6. Feasibility can be restored
by increasing ε, increasing δ, or moving to a less stringent operating point such as10−5or10−4. For IR101,
the10−6columns become nonzero around( ε,δ)≈(35,10−2)or(20,0.4), while the10−5and10−4targets are
attainable at stricter privacy budgets.
Backbone comparison.Across most of the grid, ViT-Base is more robust than IR101 at matched( ε,δ),
typically by about0 .8–2.5percentage points in macro average. For example, at(35 ,0.6), ViT-Base attains
80.83%versus78 .43%for IR101, and at(100 ,0.6)it attains89 .18%versus87 .68%. A plausible explanation is
that the ViT-Base score distribution places less impostor mass near the operational thresholds, which lowers
the decision-flip probability under the same Gaussian noise scale.
Practical regimes.The benchmark sweeps suggest three coarse regimes:
1.Small-budget regime( ε≤30): utility is strongly degraded, and the IJB@10−6operating points are often
infeasible.
2.Transition regime(35≤ε<70): utility improves rapidly with either increasingεor relaxingδ.
3.Near-saturation regime( ε≥70): performance is close to the clean baseline, with diminishing gains from
further increases inε.
Thus, for moderate δ, the practical knee of the privacy–utility trade-off occurs around ε≈ 70in these
experiments. If the operating point can be relaxed from10−6to10−5or10−4, useful accuracy extends to
smallerε.
85

arXiv preprint, ScoreShield
Effect ofδ.At fixed ε, increasing δimproves utility because the Gaussian noise scale shrinks with δ. For
IR101 atε= 100, the macro average rises from71 .29%atδ= 10−7to80.70%atδ= 10−3and to88.44%at
δ= 0.8. ViT-Base shows the same pattern, increasing from75 .07%to83.72%and then to89 .10%. When
analytic calibration or domain-restricted sensitivity reduces the effective σ, the same( ε,δ)pair moves closer
to the low-noise regime, shifting both the feasibility frontier and the practical knee toward stricter privacy.
Table H.1.Results for the IR101 backbone trained on WebFace4M. For LFW, CFP-FP, CPLFW, AgeDB, and
CALFW, we report average verification accuracy. Columns labeled B-1e-6, B-1e-5, and B-1e-4 report TAR on IJB-B
at the corresponding FPR; columns C-1e-6, C-1e-5, and C-1e-4 are defined analogously for IJB-C. For each( ε,δ)pair,
the table reports performance after adding Gaussian noise to the released score vector in Algorithm 1. The row with
ε= N/Aandδ= N/Ais the clean, non-private baseline.
εδ B-1e-6 B-1e-5 B-1e-4 C-1e-6 C-1e-5 C-1e-4 AgeDB CALFW CFPFP CPLFW LFW Avg
N/A N/A 89.46 93.07 95.52 43.49 89.07 93.72 96.35 95.57 98.37 92.82 99.70 89.74
100.00 1e-7 58.54 72.81 83.41 25.60 65.13 79.26 75.72 80.02 79.34 76.77 87.57 71.29
100.00 1e-6 68.08 77.37 85.49 28.39 70.83 82.23 77.15 81.85 81.24 76.53 88.55 74.34
100.00 1e-5 70.31 80.71 87.98 32.37 73.99 84.81 78.47 82.77 82.41 79.25 90.37 76.68
100.00 1e-4 73.18 83.68 89.87 29.36 76.48 86.85 79.78 84.47 84.73 80.10 92.05 78.23
100.00 1e-3 79.99 87.05 91.82 26.29 81.09 89.16 82.65 86.13 87.01 82.63 93.87 80.70
100.00 1e-2 80.47 88.73 93.08 31.80 83.32 90.57 85.72 89.20 89.56 84.35 96.03 82.98
100.00 0.10 86.35 90.88 94.32 29.66 85.65 92.20 88.70 91.77 92.71 87.48 98.12 85.26
100.00 0.20 86.26 91.42 94.71 39.36 86.00 92.57 90.43 92.63 93.91 88.40 98.70 86.76
100.00 0.40 88.17 92.07 94.86 43.25 87.72 92.92 91.83 93.33 95.27 89.92 99.17 88.05
100.00 0.60 87.92 92.33 95.15 35.69 87.37 93.13 93.40 93.65 96.23 90.32 99.33 87.68
100.00 0.80 88.81 92.25 95.26 39.64 87.97 93.35 93.88 94.38 96.59 91.12 99.53 88.44
70.00 1e-7 24.47 40.48 58.17 16.29 37.13 55.07 70.10 73.08 73.59 71.38 81.37 54.65
70.00 1e-6 33.72 49.64 65.33 23.10 42.62 61.51 70.98 75.93 74.09 72.12 82.67 59.25
70.00 1e-5 39.52 57.38 72.30 25.80 50.95 68.16 72.95 76.43 75.71 73.35 83.90 63.31
70.00 1e-4 53.82 66.66 78.60 20.35 60.17 75.10 73.92 77.53 77.93 75.17 86.72 67.81
70.00 1e-3 61.19 74.96 84.41 36.37 69.71 81.37 76.78 80.58 80.97 76.47 88.75 73.78
70.00 1e-2 73.44 82.93 89.36 34.15 77.06 86.20 80.32 84.25 83.01 79.53 91.27 78.32
70.00 0.10 81.22 88.55 92.73 27.91 82.78 90.05 84.72 87.87 88.57 83.72 95.55 82.15
70.00 0.20 84.54 89.52 93.44 38.26 84.02 91.09 86.85 88.87 90.23 84.93 96.78 84.41
70.00 0.40 85.97 90.91 94.30 44.22 85.58 91.85 89.35 90.98 92.73 86.72 97.82 86.40
70.00 0.60 86.96 91.32 94.69 36.18 86.85 92.32 91.05 92.00 93.60 88.48 98.83 86.57
70.00 0.80 87.40 91.92 94.87 38.26 86.93 92.78 91.95 93.53 95.29 89.25 99.20 87.40
35.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 60.63 62.92 61.44 60.47 67.75 28.47
35.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 61.07 63.68 62.96 61.85 67.95 28.86
35.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 61.77 65.50 64.74 63.07 69.00 29.46
35.00 1e-4 0.00 0.00 18.01 0.00 0.00 17.60 63.75 66.28 65.81 64.75 71.52 33.43
35.00 1e-3 0.00 14.64 28.11 0.00 13.23 26.80 65.13 68.45 67.64 65.70 75.08 38.62
35.00 1e-2 17.57 28.35 46.40 14.28 25.59 43.35 68.45 72.18 70.77 69.07 78.00 48.55
35.00 0.10 42.13 58.61 73.00 26.07 52.52 69.58 72.73 77.13 76.34 73.43 84.23 64.16
35.00 0.20 55.13 69.57 80.58 37.11 62.14 76.80 75.17 79.05 78.46 75.77 86.53 70.57
35.00 0.40 69.21 79.14 87.12 30.30 73.36 84.23 78.65 82.52 81.96 78.17 89.68 75.85
35.00 0.60 76.71 83.74 89.88 24.63 77.26 86.93 80.87 84.92 84.51 80.67 92.62 78.43
35.00 0.80 77.07 86.53 91.85 35.02 81.17 88.79 83.07 86.52 86.64 82.63 94.22 81.23
30.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 57.70 59.53 60.13 58.52 63.83 27.25
30.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 59.40 61.07 60.91 59.02 66.20 27.87
30.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 61.13 63.50 62.81 60.82 67.27 28.68
30.00 1e-4 0.00 0.00 0.00 0.00 0.00 0.00 62.55 64.95 63.41 62.62 69.08 29.33
30.00 1e-3 0.00 0.00 17.05 0.00 0.00 16.14 63.73 66.18 65.56 64.17 71.50 33.12
30.00 1e-2 0.00 16.10 30.90 0.00 14.29 28.84 66.05 68.55 67.50 67.22 75.65 39.55
30.00 0.10 27.65 41.93 60.11 19.44 38.86 56.23 70.57 73.98 71.77 70.80 81.65 55.73
30.00 0.20 39.44 55.93 71.33 28.02 51.97 67.58 72.63 77.35 76.27 73.13 83.07 63.34
30.00 0.40 54.00 70.91 81.70 36.80 65.96 78.37 74.38 79.47 78.23 75.15 87.50 71.13
30.00 0.60 67.55 78.67 87.02 20.44 72.94 83.25 78.00 82.17 81.81 78.48 90.08 74.58
30.00 0.80 74.49 83.36 90.06 31.31 77.54 86.76 79.72 84.47 84.71 80.62 92.18 78.66
25.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 55.95 57.77 57.86 55.93 61.02 26.23
25.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 56.75 59.52 58.89 58.60 62.83 26.96
25.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 59.10 59.42 60.66 58.65 64.77 27.51
25.00 1e-4 0.00 0.00 0.00 0.00 0.00 0.00 58.72 61.97 60.99 59.03 65.63 27.85
25.00 1e-3 0.00 0.00 0.00 0.00 0.00 0.00 60.23 63.73 63.74 61.83 68.70 28.93
25.00 1e-2 0.00 0.00 17.00 0.00 0.00 16.34 63.92 66.30 65.41 64.47 72.63 33.28
25.00 0.10 12.80 24.78 41.07 11.58 21.43 38.88 67.53 70.53 70.04 67.85 78.02 45.86
25.00 0.20 19.23 35.74 54.72 18.66 33.56 51.78 68.37 71.75 72.41 70.17 79.48 52.35
25.00 0.40 40.68 55.13 71.30 25.35 50.67 66.89 72.15 75.65 75.61 72.50 83.57 62.68
25.00 0.60 53.25 67.51 80.25 26.58 63.02 76.10 75.37 78.88 78.04 74.73 87.02 69.16
25.00 0.80 64.33 76.95 85.64 34.54 69.76 82.26 77.58 81.00 80.40 77.17 89.08 74.43
20.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 54.30 54.50 55.39 56.37 57.33 25.26
20.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 56.28 57.10 56.27 56.30 59.07 25.91
20.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 57.72 58.18 56.81 55.93 60.25 26.26
20.00 1e-4 0.00 0.00 0.00 0.00 0.00 0.00 56.63 59.50 58.83 56.95 61.67 26.69
20.00 1e-3 0.00 0.00 0.00 0.00 0.00 0.00 58.90 60.92 59.79 59.30 64.55 27.59
20.00 1e-2 0.00 0.00 0.00 0.00 0.00 0.00 60.78 62.62 61.53 60.35 67.95 28.48
20.00 0.10 0.00 0.00 20.15 0.00 0.00 19.95 63.87 67.82 67.16 64.87 72.97 34.25
20.00 0.20 0.00 16.98 32.03 0.00 14.99 29.16 65.07 68.87 68.40 66.38 74.93 39.71
20.00 0.40 18.99 31.37 50.28 13.84 29.79 46.65 68.88 71.35 71.83 70.13 79.78 50.26
20.00 0.60 32.45 48.07 64.77 22.68 42.97 60.03 71.65 75.40 73.56 71.53 81.62 58.61
20.00 0.80 45.84 61.56 75.12 29.11 56.03 72.10 73.05 77.35 75.96 73.90 85.17 65.93
15.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 53.05 54.38 53.40 52.85 55.07 24.43
15.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 52.82 55.17 53.06 52.95 56.47 24.59
15.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 53.15 53.98 53.67 54.38 57.30 24.77
15.00 1e-4 0.00 0.00 0.00 0.00 0.00 0.00 54.72 55.13 54.90 56.37 57.43 25.32
15.00 1e-3 0.00 0.00 0.00 0.00 0.00 0.00 55.27 56.90 57.37 56.53 59.53 25.96
15.00 1e-2 0.00 0.00 0.00 0.00 0.00 0.00 57.40 58.90 58.87 57.72 62.20 26.83
15.00 0.10 0.00 0.00 0.00 0.00 0.00 0.00 61.37 62.82 61.96 60.37 67.02 28.50
15.00 0.20 0.00 0.00 0.00 0.00 0.00 0.00 62.47 64.13 64.03 62.17 68.73 29.23
15.00 0.40 0.00 0.00 22.47 0.00 0.00 21.60 64.25 67.48 67.36 64.82 73.43 34.67
15.00 0.60 0.00 19.90 35.46 11.92 18.34 32.94 67.43 69.07 68.90 67.97 77.03 42.63
15.00 0.80 16.10 31.52 49.03 11.43 28.21 46.55 68.78 71.92 71.81 70.37 79.12 49.53
86

arXiv preprint, ScoreShield
Table H.2.Results for the ViT-Base backbone trained on WebFace4M. For LFW, CFP-FP, CPLFW, AgeDB, and
CALFW, we report verification accuracy. Columns labeled B-1e-6, B-1e-5, and B-1e-4 report TAR on IJB-B at the
corresponding FPR; columns C-1e-6, C-1e-5, and C-1e-4 are defined analogously for IJB-C. For each( ε,δ)pair, the
table reports performance after adding Gaussian noise to the released score vector in Algorithm 1. The row with
ε= N/Aandδ= N/Ais the clean, non-private baseline.
εδ B-1e-6 B-1e-5 B-1e-4 C-1e-6 C-1e-5 C-1e-4 AgeDB CALFW CFPFP CPLFW LFW Avg
N/A N/A 87.12 94.54 96.89 38.62 90.51 95.39 97.28 96.07 98.99 94.88 99.80 90.01
100.00 1e-7 60.84 74.62 84.93 43.50 68.40 82.13 78.92 81.62 81.90 79.40 89.50 75.07
100.00 1e-6 67.04 79.05 87.57 21.67 72.85 84.29 79.32 83.58 83.33 81.08 90.08 75.44
100.00 1e-5 71.69 82.05 89.73 34.94 77.93 86.71 81.13 84.92 85.61 82.23 92.10 79.00
100.00 1e-4 74.98 86.14 91.97 29.72 78.45 88.94 83.82 85.22 87.04 84.07 93.43 80.34
100.00 1e-3 79.64 88.92 93.72 40.61 83.51 91.08 84.98 87.40 89.74 86.20 95.15 83.72
100.00 1e-2 81.42 91.16 94.83 28.18 85.53 92.46 88.17 89.93 92.29 88.27 97.02 84.48
100.00 0.10 82.59 92.89 95.78 32.69 87.83 94.01 91.30 92.58 95.46 90.82 98.78 86.79
100.00 0.20 86.81 93.16 95.91 33.00 88.86 94.35 92.60 93.28 96.10 91.83 99.08 87.73
100.00 0.40 86.18 93.60 96.33 38.83 89.40 94.72 93.83 94.52 97.31 92.95 99.42 88.83
100.00 0.60 85.99 93.78 96.49 39.81 89.60 94.89 94.75 94.88 97.61 93.60 99.55 89.18
100.00 0.80 84.34 94.06 96.56 36.83 90.22 95.15 95.68 95.33 98.10 94.10 99.70 89.10
70.00 1e-7 25.13 39.46 59.06 16.29 35.87 55.32 72.38 74.48 75.47 73.92 81.42 55.35
70.00 1e-6 29.96 47.21 66.67 26.94 44.26 61.70 73.67 76.38 77.04 74.52 83.63 60.18
70.00 1e-5 37.29 56.93 73.36 30.94 52.83 70.12 74.73 76.95 78.31 76.37 86.17 64.91
70.00 1e-4 52.67 67.86 80.93 35.75 60.49 77.05 76.82 79.87 80.27 79.12 87.07 70.72
70.00 1e-3 58.04 76.45 86.15 28.89 69.46 83.19 79.35 82.68 83.14 79.62 90.03 74.27
70.00 1e-2 71.37 84.52 91.01 37.24 78.48 88.21 82.23 84.83 86.67 83.58 92.90 80.10
70.00 0.10 81.20 90.44 94.31 27.32 85.10 92.04 86.47 89.50 90.69 87.67 95.85 83.69
70.00 0.20 81.98 91.31 95.22 39.81 87.16 92.89 89.13 90.80 93.13 88.87 97.25 86.14
70.00 0.40 84.57 92.43 95.76 43.55 87.83 93.57 90.72 92.38 94.51 90.80 98.45 87.69
70.00 0.60 85.98 93.09 96.05 44.46 88.53 94.15 92.82 93.15 95.96 91.82 99.02 88.64
70.00 0.80 85.95 93.50 96.30 36.43 89.23 94.59 93.55 94.20 96.97 92.65 99.18 88.41
35.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 61.77 62.90 63.66 62.80 67.53 28.97
35.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 61.88 65.47 62.86 63.23 69.05 29.32
35.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 62.98 65.78 65.73 65.77 72.38 30.24
35.00 1e-4 0.00 0.00 18.26 0.00 0.00 17.05 65.17 68.28 67.66 66.87 71.82 34.10
35.00 1e-3 0.00 15.92 28.03 0.00 15.03 25.80 67.77 69.98 70.50 68.40 75.33 39.71
35.00 1e-2 17.05 28.95 46.23 11.94 24.58 43.20 70.03 73.35 73.51 71.85 79.95 49.15
35.00 0.10 43.42 57.97 74.08 21.27 53.43 71.02 75.63 77.98 79.03 76.55 86.35 65.16
35.00 0.20 53.35 69.76 82.13 35.91 63.36 78.75 76.92 80.73 81.24 78.98 88.00 71.74
35.00 0.40 66.49 81.48 88.96 33.61 74.53 85.95 80.80 83.77 84.81 81.88 91.02 77.57
35.00 0.60 74.62 86.18 91.84 35.38 79.31 89.05 82.83 85.53 87.81 83.53 93.00 80.83
35.00 0.80 76.83 89.27 93.57 28.70 83.15 91.11 86.10 88.02 89.14 86.02 94.98 82.44
30.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 58.53 60.90 60.01 59.88 65.15 27.68
30.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 59.98 62.13 62.17 62.30 66.40 28.45
30.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 61.05 62.30 63.80 62.22 66.22 28.69
30.00 1e-4 0.00 0.00 0.00 0.00 0.00 0.00 62.53 66.40 65.77 64.20 69.13 29.82
30.00 1e-3 0.00 0.00 19.02 0.00 0.00 18.45 65.42 67.35 66.91 66.03 72.70 34.17
30.00 1e-2 0.00 15.56 30.67 0.00 14.62 28.80 66.80 69.33 70.10 69.48 76.43 40.16
30.00 0.10 25.75 41.70 60.18 16.35 36.54 56.38 71.95 75.13 75.69 73.88 82.95 56.05
30.00 0.20 38.42 55.71 72.13 21.90 51.15 68.57 75.65 77.73 78.61 75.55 84.77 63.65
30.00 0.40 59.83 72.27 83.74 30.84 66.40 80.69 78.22 81.27 81.97 79.60 88.68 73.05
30.00 0.60 66.94 80.96 88.84 28.65 73.95 85.82 80.67 84.07 85.20 82.95 90.68 77.16
30.00 0.80 73.60 85.86 91.46 22.94 79.55 88.96 82.27 86.55 87.44 83.52 93.28 79.58
25.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 57.07 58.17 58.10 58.27 61.85 26.68
25.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 58.38 59.48 59.61 58.90 62.43 27.16
25.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 59.57 60.43 60.44 59.35 64.57 27.67
25.00 1e-4 0.00 0.00 0.00 0.00 0.00 0.00 60.83 62.28 61.84 61.65 66.67 28.48
25.00 1e-3 0.00 0.00 0.00 0.00 0.00 0.00 63.37 64.32 63.20 62.57 68.33 29.25
25.00 1e-2 0.00 0.00 18.45 0.00 0.00 17.88 65.35 67.45 68.09 65.48 72.75 34.13
25.00 0.10 14.24 23.08 40.62 12.96 20.93 38.62 69.53 72.35 72.54 70.85 78.50 46.75
25.00 0.20 22.15 36.31 54.91 16.75 32.40 50.93 71.02 74.47 75.67 72.68 81.53 53.53
25.00 0.40 41.10 56.44 72.11 27.59 50.54 68.34 74.12 77.57 76.59 76.27 85.32 64.18
25.00 0.60 50.97 69.69 81.44 33.58 63.65 77.88 77.55 81.08 81.23 79.17 88.30 71.32
25.00 0.80 65.19 78.83 87.55 25.03 72.02 84.03 79.85 82.85 83.23 80.92 90.25 75.43
20.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 55.65 56.50 55.63 54.63 57.98 25.49
20.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 56.48 56.65 56.80 55.88 60.85 26.06
20.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 58.45 58.95 56.91 57.40 60.05 26.52
20.00 1e-4 0.00 0.00 0.00 0.00 0.00 0.00 58.22 59.73 58.71 58.60 62.30 27.05
20.00 1e-3 0.00 0.00 0.00 0.00 0.00 0.00 59.73 60.23 61.29 60.93 64.37 27.87
20.00 1e-2 0.00 0.00 0.00 0.00 0.00 0.00 62.47 64.55 64.11 63.50 66.87 29.23
20.00 0.10 0.00 0.00 20.40 0.00 0.00 19.77 65.18 69.18 68.61 67.00 73.47 34.87
20.00 0.20 0.00 16.54 31.76 0.00 14.90 29.18 67.50 70.62 70.89 69.93 76.05 40.67
20.00 0.40 18.79 32.46 50.77 15.71 29.11 47.32 70.87 72.52 73.14 72.72 80.30 51.25
20.00 0.60 29.03 47.20 65.23 22.12 44.73 62.32 72.43 77.15 76.93 75.07 83.72 59.63
20.00 0.80 44.87 62.17 76.83 28.01 56.69 72.87 76.42 78.83 79.03 76.63 86.67 67.18
15.00 1e-7 0.00 0.00 0.00 0.00 0.00 0.00 52.90 53.72 54.21 53.53 55.03 24.49
15.00 1e-6 0.00 0.00 0.00 0.00 0.00 0.00 54.80 54.78 54.76 52.73 56.12 24.84
15.00 1e-5 0.00 0.00 0.00 0.00 0.00 0.00 53.28 54.90 54.26 56.48 55.60 24.96
15.00 1e-4 0.00 0.00 0.00 0.00 0.00 0.00 55.27 57.28 55.64 55.75 57.97 25.63
15.00 1e-3 0.00 0.00 0.00 0.00 0.00 0.00 55.78 57.53 57.21 56.88 59.33 26.07
15.00 1e-2 0.00 0.00 0.00 0.00 0.00 0.00 58.13 60.35 59.59 59.15 62.07 27.21
15.00 0.10 0.00 0.00 0.00 0.00 0.00 0.00 61.95 62.48 63.50 63.95 67.78 29.06
15.00 0.20 0.00 0.00 0.00 0.00 0.00 0.00 64.67 65.00 65.56 64.52 70.27 30.00
15.00 0.40 0.00 0.00 21.86 0.00 0.00 20.35 66.25 69.23 69.03 66.90 74.27 35.26
15.00 0.60 0.00 18.03 33.69 0.00 16.59 32.91 67.80 71.97 70.96 69.78 77.80 41.78
15.00 0.80 18.00 31.12 49.75 14.04 28.11 46.88 70.48 74.10 74.24 72.33 79.65 50.79
87

arXiv preprint, ScoreShield
(a)
 (b)
Figure H.9.Average performance across seven benchmarks when applying Algorithm 1 at different( δ,ε)values to (a)
an IR101 backbone and (b) a ViT-Base backbone, both trained on WebFace4M.
88

arXiv preprint, ScoreShield
I Supplementary Details for Regime (i): DP-RAG
We evaluate a single-query ( T= 1) DP-RAG retrieval primitive. For each query, the mechanism releases a
differentially private query-to-collection score vector and then applies standard thresholded top- kretrieval to
the privatized scores.
I.1 Privacy Object, Adjacency, and Retrieval Scores
In our experiments, the source corpus is public Wikipedia text. The DP guarantee therefore concerns the
released similarity-score vector and index-level functions of that vector, such as rankings, top-kindices, and
thresholded retrieval decisions. It does not claim secrecy of the Wikipedia text itself.
Setting.We work in the central model, where a trusted server holds an indexed retrieval corpus and releases
only theretrieval output(a set of top- kchunks) and thegenerated answer. For each evaluation instance, we
construct an instance-specific knowledge base KB={d1,...,dn}, where each record di∈Dis one indexed
document chunk (raw text, i.e., a finite token/string sequence) and nis the number of chunks produced
by the scrape–chunk–index pipeline described in Sec. I.3.1. The protected object is the indexed retrieval
representation used to compute query–corpus similarity scores.
Adjacency.We use record-level replacement adjacency on chunks. Two knowledge bases KB∼KB′are
neighboring if they differ in exactly one chunk, i.e., dj̸=d′
jfor onej∈[n],di=d′
i, for alli̸=j. The user
query is treated as public input. Protecting the query itself would require a different threat model, such as
local or distributed privacy, and is outside the scope of this experiment.
Embedding map and score vector.Let ϕ:(text)→Rddenote the document embedding map. Each
chunk diis embedded asx i:=ϕ(di)∈Rdand then clipped to the unit ball asx i←xi/max{ 1,∥xi∥2}, so
∥xi∥2≤1,∀i. Given a public prompt, we form a retrieval query embeddingq :=ϕqry(prompt )and apply the
same clipping rule asq ←q/max{ 1,∥q∥2}, so that∥q∥2≤1. We define the retrieval score for chunk i∈[n]
assi(q):=⟨q,xi⟩∈[−1,1], and writes(q) = (s 1(q),...,s n(q))∈[−1,1]n.
I.2 DP-RAG Mechanism and Privacy Calibration
ScoreShieldMechanism.We privatize the complete query-to-collection score vector using the Gaussian
mechanism followed by entrywise clipping:
/hatwides(q) :=clip[−1,1]n/parenleftbig
s(q) +w/parenrightbig
,w∼N(0,σ2
ε,δIn),(341)
where clip[−1,1]ndenotes entrywise clamping to the interval[ −1,1]. Retrieval is then performed on /hatwides(q)using
the same thresholded top- krule as in the non-private pipeline. Finally, the generator receives the retrieved
text and produces an answer. Any index-level output computed from /hatwides(q), including rankings, top- kindex
lists, and thresholded index sets, is an(ε,δ)-DP post-processing of the privatized score vector.
If the retrieved text is not itself part of the private dataset, then the retrieved indices/text and the generated
answer are measurable functions of /hatwides(q), the public prompt, and the mechanism’s internal randomness, they
are post-processings of /hatwides(q)and inherit the same(ε,δ)-DP guarantee14.
RemarkI.1 (ScoreShield Scope of the DP guarantee).ScoreShieldguarantees DP for the released score
vector and for index-level functions of that vector. In our DP-RAG experiments, this is sufficient because
the underlying corpus text is public. If the corpus text itself is private under the adjacency relation, then
releasing raw retrieved chunks (or an answer generated from those chunks) generally requires additional
DP mechanisms beyond score release (e.g., DP generation/aggregation or one-time corpus privatization).
ScoreShield alone guarantees privacy for the released score object and index-level functions of it, but does not
provide end-to-end DP for revealed chunk contents or for generated answers conditioned on those contents.
14DP post-processing allows arbitrary functions of the DP output that do not additionally access the private dataset. If
the corpus text is private and differs across neighboring KBs, then index∝⇕⊣√∫⊔≀→chunk text is an additional dataset access, so
ScoreShieldlone does not imply end-to-end DP for revealed chunk text or for the final answer.
89

arXiv preprint, ScoreShield
Globalℓ2-sensitivity under record replacement.Fix a public queryqwith ∥q∥2≤1. For neighboring
KBs differing only in record j, the score vector differs in exactly one coordinate si(q) =s′
i(q) (i̸=j),
sj(q)−s′
j(q) =⟨q,x j−x′
j⟩. Hence
∥s(q)−s′(q)∥ 2=|⟨q,xj−x′
j⟩|≤∥q∥ 2∥xj−x′
j∥2≤2,(342)
Thus the globalℓ 2-sensitivity is at most∆ query= 2.
Gaussian calibration.Using Lemma C.6, we set σ2
ε,δ=cε,δ∆2
query,cε,δ:=2 log(2/δ)
ε2, with∆ query= 2. This
givesσ2= 4cε,δandσε,δ= 2/radicalbig
2 log(2/δ)/ε.
I.3 Experimental Protocol
We evaluate a single-query DP-RAG retrieval primitive on the FRAMES benchmark [ 25]. For each query, the
retrieval module computes query–corpus similarity scores, optionally privatizes these scores, and then applies
the same thresholded top-kretrieval rule to either the clean or privatized score vector.
I.3.1 Run-Level Corpus Construction
Each FRAMES example contains a set of Wikipedia pages in its wiki_links field. For a fixed evaluation run,
we collect the union of all linked Wikipedia pages appearing in the selected examples and construct a single
run-level retrieval corpus. Thus, RAG retrieves from one global chunk index for the run, rather than from a
question-specific corpus.
For each page, we scrape the article text using the wikipedia API and convert the content to plain text
using BeautifulSoup . The text is split into overlapping token windows of approximately 2000 tokens with
200-token overlap, where token counts are computed using the embedding tokenizer. Each chunk is embedded
together with its page title using the retrieval encoder and the document instruction "Retrieval-document" .
We denote the resulting chunk embeddings byx 1,...,xn∈Rd, wherenis the number of chunks in the
run-level corpus.
I.3.2 Retrieval Scores and Thresholded Top-kRule
Given a queryq, we embed it using the retrieval encoder and the query instruction "Retrieval-query" .
The retrieval score for chunk iis the cosine similarity si(q) =⟨q,xi⟩,i∈[n], where the query and chunk
embeddings are normalized before computing the inner product. We writes(q) = ( s1(q),...,sn(q))∈[−1,1]n
.
For a generic score vectorz∈[−1,1]n, define the eligible index set
Iτ(z):={i∈[n] :z i≥τ}.(343)
Let≺zdenote the strict total order on Iτ(z)that sorts indices by decreasing score and breaks ties by smaller
index. That is, fori̸=j,
i≺zj⇐⇒/parenleftbig
zi>zj/parenrightbig
or/parenleftbig
zi=zjandi<j/parenrightbig
.(344)
Leti(1)(z),...,i (|Iτ(z)|)(z)be the elements of Iτ(z)listed according to ≺z. The thresholded top- kretrieval
operator returns
TopKk,τ(z):=/parenleftbig
i(1)(z),...,i (m)(z)/parenrightbig
, m= min{k,|I τ(z)|}.(345)
We usek= 20andτ= 0.35.
I.3.3 Evaluation Conditions
We compare the following conditions.
•No-context baseline.The generator receives only the user question. No retrieved document text is included
in the prompt.
90

arXiv preprint, ScoreShield
•Oracle linked-page context.The generator receives text from the Wikipedia pages listed in the current
example’s wiki_links field. The concatenated article text is truncated to 10,000 generator tokens before
being inserted into the prompt. This condition measures performance when the model is given the
benchmark linked evidence, subject to the same context-length constraint.
•RAG with clean retrieval.The retrieval module computes the clean score vectors(q)over the run-level
corpus and returns TopKk,τ(s(q)). The corresponding chunks are inserted into the generator prompt in
retrieved order. If no chunk satisfies the threshold, the prompt states that no document was retrieved.
•DP-RAG with noisy retrieval.The retrieval module first privatizes the score vector using
/hatwides(q) =clip[−1,1]n/parenleftbig
s(q) +w/parenrightbig
,w∼N/parenleftbig
0,σ2
ε,δIn/parenrightbig
,(346)
whereσε,δ=2√
2 log(2/δ)
ε. Retrieval then returns TopKk,τ(/hatwides(q)). Thus, clean RAG and DP-RAG differ
only in whether the retrieval rule is applied tos(q)or to /hatwides(q).
I.3.4 Models and Evaluation
We use vLLMfor generation and sentence-transformers utilities for embedding and dense retrieval. The
retrieval backbone encoder is Embedding Gemma [40] (i.e.,EG300M) orQwen3-VL-Embedding[ 27] (i.e.,
Q3VL-E2B). The evaluated generators are listed in Table I.1. Prompts are rendered using the chat template
associated with each generator. Decoding hyperparameters, including temperature, top- p, and maximum
generation length, are fixed across conditions for a given generator.
Because FRAMES contains open-ended answers, exact string matching is not reliable. We therefore use
an LLM judge. For each example, the judge receives the question, the reference answer, and the generated
answer, and returns a binary match decision. We report answer accuracy as the percentage of examples
marked correct. Latency is measured only for answer generation; it excludes corpus construction, embedding,
retrieval-index construction, and judging.
Figure I.1 summarizes the experimental protocol used for the FRAMES evaluation.
I.4 Empirical Results
Evaluation protocol and retrieval database size.For each question in FRAMES, we extract the
associated Wiki links and chunk the corresponding documents with a overlap based on the maximum context
length supported by the embedding model. The resulting chunks define the retrieval database queried at
inference time. Therefore, the number of database entries depends on the embedding extractor: EG300M
produces 22,880 entries, while Q3VL-E2B produces 6,892 entries (with a fixed number number of token
overlaps), due to its larger supported input length. The privacy parameters( ϵ,δ)only affect the noisy retrieval
mechanism. They do not change the underlying language model, the embedding extractor, the question set,
or the oracle/no-context baselines. Consequently, oracle and baseline accuracies remain fixed for a given
model–embedding configuration, up to small variation due to repeated runs or generation/evaluation seeds.
Retrieval improves utility in the non-private setting.Across all configurations where non-noisy
RAG is evaluated, adding retrieved context improves accuracy over the no-context baseline. For example,
Gemma3-27B improves from40 .17to60.92, Gemma4-26B-A4B improves from26 .09to62.26at threshold
0.25and top- k= 50, and Q3-8B improves from75 .85to80.58. In all cases, non-noisy RAG remains below
the oracle setting, which provides only the relevant evidence. This gap indicates that retrieval is useful, but
also that the retrieved context still contains irrelevant or incomplete information compared with the oracle
evidence.
Effect of generation and evaluation seed.Small differences in oracle and baseline accuracy across
otherwise similar settings are caused by run-level variation rather than by the privacy parameters. For instance,
the first two Gemma3-12B EG300M rows show very similar oracle and baseline accuracies,72 .09/45.87versus
71.48/46.36. This suggests that the effect of the generation/evaluation seed is minor compared with the effect
of adding retrieval or injecting noise into the retrieval scores.
91

arXiv preprint, ScoreShield
FRAMES sample
? question
ground-truth
answer
linked
Wikipedia
pages
(wiki_links)
Run-level retrieval corpus construction
Collect the union of
linked Wikipedia
pages
across the
evaluation
examples
Scrape and
clean ar-
ticle text
Chunk into
overlapping
windows
(about 2000
tokens,
200-token
overlap)
Embed
each chunk
with retrieval
encoder
instruction:
“Retrieval-
document”
Run-level chunk index
x1, . . . ,xn∈Rd
Public retrieval query
derived from
the question text
Query embedding
Retrieval encoder
Instruction: “Retrieval-
query”
q∈Rd
Score computation (against indexed chunks)
s(q) =/parenleftbig⟨q,x1⟩, . . . ,⟨q,xn⟩/parenrightbig∈[−1,1]n
cosine similarities between qand
indexed chunks x1, . . . ,xn
Clean RAG
Thresholded top- kretrieval
TopKk,τ(s(q))
Default: k= 20, τ= 0.35
Retrieved chunks
DP-RAG
Gaussian score privatization
ˆs(q) = clip[−1,1]n/parenleftbig
s(q) +w/parenrightbig
w∼ N (0, σ2
ϵ,δIn)
σϵ,δ= 2/radicalbig
2 log(2 /δ)/ϵ
Thresholded top- kretrieval
TopKk,τ(ˆs(q))
Default: k= 20, τ= 0.35
Retrieved chunks
No-context baseline
Generator receives
the question only
Oracle linked-page context
Generator receives
the linked
Wikipedia pages
for the current example
Text is truncated to
10,000 generator tokens
Generator LLM
Gemma-3 and Gemma-4 models
Same decoding hyperparameters
across conditions
LLM judge
Input: question, ground-truth answer,
generated answer
Output: JSON {“match”: true/false}
Reported metric
Accuracy = percentage of
examples with match = true
question + ground-truth answer
Figure I.1.Experimental protocol for the FRAMES DP-RAG evaluation. A run-level retrieval corpus is constructed
from the union of linked Wikipedia pages across the evaluation examples and embedded into a global chunk index
x1,...,xn.For each public retrieval queryq, clean RAG applies thresholded top- kretrieval to the clean score vector
s(q), whereas DP-RAG first applies Gaussian score privatization and retrieves from /hatwides(q). The no-context baseline,
clean RAG, DP-RAG, and oracle linked-page condition are evaluated with the same generator and LLM-judge protocol.
Noisy retrieval exhibits a sharp privacy–utility tradeoff.Injecting noise into the retrieval scores can
substantially degrade utility, especially at stronger privacy settings. For Gemma4-26B-A4B with EG300M,
noisy RAG at ϵ∈{ 1,10}achieves only4 .61–6.07accuracy, far below the26 .09baseline. Similarly, Gemma4-
E4B remains below2accuracy for several noisy settings at ϵ∈{ 1,10}. These results show that, when the
noise is too large, the retrieved context can become misleading enough that RAG performs worse than using
no retrieved context at all.
Privacy–utility operating points.As ϵandδincrease, the amount of injected noise decreases. The noisy
retrieval mechanism therefore approaches the behavior of standard non-private RAG, but with weaker privacy
guarantees. We mark in green the noisy retrieval settings whose RAG accuracy exceeds the corresponding
no-context baseline. For example, Gemma3-12B with EG300M reaches54 .98accuracy at( ϵ,δ) = (100,0.01),
compared with a46 .36baseline and55 .46non-noisy RAG accuracy. Gemma3-4B improves from a49 .51
baseline to52 .43under(100 ,0.01). Gemma4-26B-A4B improves from26 .09to50.24at(100,0.01), and
Gemma4-E4B improves from10 .19to39.68under the same setting. These operating points demonstrate
that noisy retrieval can preserve part of the utility gain from RAG, but only under relatively weak privacy
parameters in the present runs.
92

arXiv preprint, ScoreShield
Main empirical takeaway.The results support three observations. First, retrieval is consistently beneficial
when no noise is added. Second, strong privacy settings can destroy retrieval quality and may reduce accuracy
below the no-context baseline. Third, weaker privacy settings can recover a substantial fraction of the
non-private RAG gain, indicating a clear privacy–utility tradeoff. We therefore do not interpret the highlighted
rows as evidence that private RAG always improves performance, but rather as empirical operating points
where the proposed noisy retrieval mechanism remains useful.
93

arXiv preprint, ScoreShield
Table I.1.Accuracy summary by model and noise setting. Highlighted rows have noise enabled and RAG accuracy
above baseline accuracy.
Model Embedding δ ϵTop-kThreshold DB Entries Oracle Acc. Baseline Acc. RAG Acc.
Gemma3-12B EG300M N/A N/A 50 0.25 22880 72.09 45.87 55.46
Gemma3-12B EG300M 1e-05 1.0 50 0.25 22880 71.48 46.36 39.44
Gemma3-12B EG300M 0.001 10.0 50 0.25 22880 71.48 46.36 40.41
Gemma3-12B EG300M 0.01 10.0 50 0.25 22880 71.48 46.36 39.68
Gemma3-12B EG300M 0.001100.0 50 0.25 22880 71.48 46.36 52.31
Gemma3-12B EG300M 0.01100.0 50 0.25 22880 71.48 46.36 54.98
Gemma3-12B Q3VL-E2B N/A N/A 10 0.25 6892 71.48 46.36 56.92
Gemma3-12B Q3VL-E2B 1e-05 1.0 10 0.25 6892 71.48 46.36 41.63
Gemma3-12B Q3VL-E2B 1e-06 1.0 10 0.25 6892 71.48 46.36 42.23
Gemma3-12B Q3VL-E2B 1e-05 10.0 10 0.25 6892 71.48 46.36 41.63
Gemma3-12B Q3VL-E2B 1e-06 10.0 10 0.25 6892 71.48 46.36 41.75
Gemma3-27B EG300M N/A N/A 20 0.35 22880 72.82 40.17 60.92
Gemma3-27B EG300M 1e-05 1.0 20 0.35 22880 72.82 40.17 39.68
Gemma3-27B EG300M 1e-06 1.0 20 0.35 22880 72.82 40.17 37.86
Gemma3-4B EG300M 0.001 10.0 50 0.25 22880 65.66 49.51 46.60
Gemma3-4B EG300M 0.01 10.0 50 0.25 22880 65.66 49.51 47.94
Gemma3-4B EG300M 0.001100.0 50 0.25 22880 65.66 49.51 52.18
Gemma3-4B EG300M 0.01100.0 50 0.25 22880 65.66 49.51 52.43
Gemma4-26B-A4B EG300M N/A N/A 20 0.35 22880 73.30 26.09 52.55
Gemma4-26B-A4B EG300M 1e-05 1.0 20 0.35 22880 73.30 26.09 4.61
Gemma4-26B-A4B EG300M 1e-06 1.0 20 0.35 22880 73.30 26.09 4.85
Gemma4-26B-A4B EG300M 1e-05 10.0 20 0.35 22880 73.30 26.09 4.61
Gemma4-26B-A4B EG300M 1e-06 10.0 20 0.35 22880 73.30 26.09 4.61
Gemma4-26B-A4B EG300M N/A N/A 50 0.25 22880 73.30 26.09 62.26
Gemma4-26B-A4B EG300M 1e-05 1.0 50 0.25 22880 73.30 26.09 5.34
Gemma4-26B-A4B EG300M 1e-06 1.0 50 0.25 22880 73.30 26.09 4.73
Gemma4-26B-A4B EG300M 0.01 10.0 50 0.25 22880 73.30 26.09 4.85
Gemma4-26B-A4B EG300M 1e-06 10.0 50 0.25 22880 73.30 26.09 6.07
Gemma4-26B-A4B EG300M 0.01100.0 50 0.25 22880 73.30 26.09 50.24
Gemma4-26B-A4B EG300M 1e-06100.0 50 0.25 22880 73.30 26.09 36.04
Gemma4-31B EG300M 1e-05 1.0 20 0.35 22880 80.58 29.98 0.85
Gemma4-31B EG300M N/A N/A 50 0.25 22880 80.58 29.98 72.57
Gemma4-E4B EG300M N/A N/A 20 0.35 22880 61.04 9.83 41.38
Gemma4-E4B EG300M 1e-05 1.0 20 0.35 22880 61.04 9.83 0.97
Gemma4-E4B EG300M 1e-06 1.0 20 0.35 22880 61.04 9.83 0.61
Gemma4-E4B EG300M 1e-05 10.0 20 0.35 22880 61.04 9.83 1.46
Gemma4-E4B EG300M 1e-06 10.0 20 0.35 22880 61.04 9.83 1.21
Gemma4-E4B EG300M N/A N/A 50 0.25 22880 62.01 10.19 47.82
Gemma4-E4B EG300M 0.01 10.0 50 0.25 22880 62.01 10.19 1.82
Gemma4-E4B EG300M 1e-06 10.0 50 0.25 22880 62.01 10.19 1.70
Gemma4-E4B EG300M 0.01100.0 50 0.25 22880 62.01 10.19 39.68
Gemma4-E4B EG300M 1e-06100.0 50 0.25 22880 62.01 10.19 28.03
Q3-8B Q3VL-E2B N/A N/A 10 0.25 6892 91.02 75.85 80.58
Q3-8B Q3VL-E2B 1e-05 1.0 10 0.25 6892 91.02 75.85 74.27
Q3-8B Q3VL-E2B 1e-06 1.0 10 0.25 6892 91.02 75.85 73.54
Q3-8B Q3VL-E2B 1e-05 10.0 10 0.25 6892 91.02 75.85 74.76
Q3-8B Q3VL-E2B 1e-06 10.0 10 0.25 6892 91.02 75.85 76.46
94

arXiv preprint, ScoreShield
JSupplementary Details for Regime (ii): Omitted Theorems, Propositions,
Proofs and Lemmas
This appendix complements Sec. 3.2. It first derives the sensitivity for regime (ii) and provides an extended
presentation of the fast projection via alternating steps algorithm. Next, we provide the full statement and
proof of the averaged–alternating-projection theorem used in Sec. 3.2. We then contrast our projection scheme
with the perturb-and-project method of Cohen-Addadet al.[ 14], detailing (i) the respective feasible sets and
adjacency models, (ii) the decomposition choices that make the projection steps cheap, (iii) the role of the
unit-diagonal constraint, and (iv) complexity and convergence rates.
J.1 Global Frobenius Sensitivity Under Record-Level Adjacency
Under record-level adjacency model,EandE′differ in at most a single row i. Let δ=e′
i−eidenote the
row–difference, with ∥δ∥2≤2, and embed it into an n×dmatrix ∆:=/bracketleftbig
0;...;δ⊤;...;0/bracketrightbig
. Hence, we have
E′=E+∆. Then
ffull(E)−f full(E′) =EE⊤−(E+∆)(E+∆)⊤=−E∆⊤−∆E⊤−∆∆⊤.(347)
Equivalently,S′−S=E′E′⊤−EE⊤=E∆⊤+∆E⊤+∆∆⊤. Because ∆has exactly one non-zero row,
above decomposition populates only row iand column iof then×ndifference matrix. Writing vj:=δ⊤ej
andv := (v1,...,vn)⊤, the diagonal cancels via(S′−S)ii= 2e⊤
iδ+∥δ∥2
2= 0(since∥e′
i∥2
2=∥ei∥2
2= 1).
HenceS′−S=v e⊤
(i)+e(i)v⊤, wheree (i)∈Rndenotes the i-th canonical basis vector. Only the off-diagonal
pairs(i,j)and(j,i),j̸=i, survive. Hence
∥S′−S∥2
F= 2/summationdisplay
j̸=iv2
j= 2/summationdisplay
j̸=i/parenleftbig
δ⊤ej/parenrightbig2(348a)
= 2δ⊤/parenleftig/summationdisplay
j̸=ieje⊤
j/parenrightig
δ≤2(n−1)∥δ∥2
2≤8(n−1).(348b)
Therefore the global Frobenius sensitivity is∆ f,F=:∆full= 2/radicalbig
2(n−1).
J.2 DP Release Mechanism for the Full Pairwise Similarity Score Matrix
Algorithm 5DP All-Pairs Similarity Matrix Release (regime (ii))
1:Input:E∈Rn×d,ε>0,δ∈(0,1), sensitivity∆>0
2:Output:/hatwideS∈C coll⊆Rn×n
3:Construct:S←EE⊤.
4:Gaussian calibration: setσ2
ε,δ←cε,δ∆2
5:Sample noiseW∼N/parenleftbig
0,σ2In×n/parenrightbig
6:Perturb with noise:S′=S+W
7:Project:/hatwideS=projCcoll(S′)=arg min A∈C coll∥A−S′∥2
F
8:Return/hatwideS
J.3 Privacy Guarantee of ScoreShield for Regime (ii)
Theorem J.1(Privacy Guarantee of Full Pairwise Similarity Score Matrix Release).Let ffull:E→Rn×nbe
ffull(E) =S=EE⊤. Fix an adjacency relation∼onEand assume that for some∆>0,
∥ffull(E)−f full(E′)∥F≤∆,∀E∼E′.(349)
Letε > 0,δ∈ (0,1),cε,δ:= 2log(2/δ)/ε2, and setσ2:=cε,δ∆2. LetW∈Rn×nhave i.i.d. entries
Wij∼N(0,σ2). Let projCbe any (possibly randomized) measurable post-processing that depends onEonly
through its input argument. Define the release /hatwideS=Mcoll(E) :=projC(ffull(E) +W) . Then the mechanism
E∝⇕⊣√∫⊔≀→/hatwideSis(ε,δ)-DP with respect to∼.
95

arXiv preprint, ScoreShield
Proof.Byassumption,thestatistic ffull:E→Rn×nhasglobalFrobeniussensitivity∆ f,F:=supE∼E′∥ffull(E)−
ffull(E′)∥F≤∆. Consider the additive Gaussian mechanism M0(E) :=ffull(E) +W,Wiji.i.d.∼ N (0,σ2),
σ2=cε,δ∆2. Viewing Rn×nas a Euclidean space equipped with the Frobenius norm, M0is exactly the
Gaussian mechanism for matrix-valued outputs (Corollary C.9). Therefore, for every pairE ∼E′and every
measurable setT⊆Rn×n,
Pr[M 0(E)∈T]≤eεPr[M 0(E′)∈T] +δ.(350)
Now define the released mechanism /hatwideS=Mcoll(E) :=proj(M 0(E)) =proj(f full(E) +W) , where projis any
measurable map that depends onEonly through its input argument (possibly using additional randomness
independent ofE). By the post-processing property (Lemma C.7), for every measurable setS⊆Y,
Pr[M coll(E)∈S]≤eεPr[M coll(E′)∈S] +δ,∀E∼E′.(351)
Hence the mechanismE∝⇕⊣√∫⊔≀→ /hatwideSis(ε,δ)-DP with respect to∼.
In particular, under output-space (Gram) adjacency (Def. 2.3), the global sensitivity bound is∆ = ∆ G
by definition of the adjacency radius. Under record-level (single-record replacement) adjacency, the global
Frobenius sensitivity is∆ = ∆ full= 2/radicalbig
2(n−1).
J.4 Fast Projection via Averaged Alternating Projections
LetS′=S+W∈Rn×nbe a perturbed Gram matrix, whereS=EE⊤is the clean cosine Gram matrix and
∥ei∥2= 1,∀i. Our goal is to projectS′onto the cosine-Gram feasibility set Ccoll:=/braceleftbig
S∈Rn×n:S⪰0, Sii=
1 (1≤i≤n ),|Sij|≤1 (i̸=j)/bracerightbig
⊂Rn×n. The exact projection onto the cosine-Gram feasible set Ccollunder
the Frobenius norm is the metric projection onto the elliptope and in general requires solving an SDP. Instead
we compute an approximately feasible point by iterating a Krasnosel’ski˘ ı-Mann averaged projector [ 28,29],
and hence replace the direct projection by alternating projections onto two closed convex sets, each admitting
a closed-form projector15. We decomposeC coll=Kn
+∩Cn
unitwhere
Kn
+={S∈Rn×n|S⪰0},Cn
unit={S∈Rn×n|Sii= 1,|Sij|≤1,(i̸=j)}.(352)
Starting from /hatwideS0:=S′, we iterate the equal-weights averaged map
/hatwideSt+1=1
2/parenleftig
projKn
+/parenleftbig
sym(/hatwideSt)/parenrightbig
+projCn
unit/parenleftbig
sym(/hatwideSt)/parenrightig
,sym(Y) :=1
2(Y+Y⊤).(353)
Under bounded linear regularity of( Kn
+,Cn
unit)on a ball containing the iterates, Theorem J.13 yields a
geometric contraction of the feasibility gap: dist(/hatwideSt,Ccoll)≤ρtdist(/hatwideS0,Ccoll)for someρ∈(0,1). Consequently,
to reach dist(/hatwideSt,Ccoll)≤τit suffices that t≥log (dist(/hatwideS0,Ccoll)/τ)/log(1/ρ) =O(log(1/τ))(see Corollary J.14).
Each iteration is dominated by an n×neigendecomposition for the PSD projection ( O(n3)), so the total cost
isO(n3log(1/τ))operations to tolerance.
Symmetrization.Before projecting onto Kn
+, we need to replace /hatwideStbyY= sym(/hatwideSt). By Lemma J.16, the
PSD projection depends only on the symmetric part under∥·∥ F, soprojKn
+(/hatwideSt) =projKn
+(Y). This removes
antisymmetric numerical artifacts that would otherwise inflate∥ /hatwideSt∥F.
J.4.1 Projection onto the PSD Cone
Step (i): Spectral Decomposition.Given a symmetric iterateY=Y⊤=1
2/parenleftbig/hatwideSt+/hatwideS⊤
t/parenrightbig
∈Sn,/hatwideS0:=S+W,
we compute its spectral decompositionY=U diag(λ1,...,λn)U⊤, costingO(n3). All subsequent steps are
performed in the eigenbasisU.
15In our mechanism definition and in the geometric analysis, projCcolldenotes the Euclidean projector (metric projection) onto
Ccoll. In implementation, however, we enforce feasibility using a Krasnosel’ski˘ ı–Mann [ 28,29] averaged-projection iteration that
returns anapproximately feasiblepoint in Ccollbut is not, in general, the metric projection of the noisy input onto Ccoll. Since
this feasibility map depends only on the perturbed matrix (and any internal randomness is independent of the data), it is
post-processing and therefore does not affect the privacy guarantee.
96

arXiv preprint, ScoreShield
Step (ii): PSD Truncation.The Frobenius–orthogonal projector onto the PSD cone solves minS⪰0∥S−Y∥2
F
and is obtained as follows. Define λ+
k:=max{ 0,λk}, k= 1,...,n, and let λ+:= (λ+
1,...,λ+
n). The orthogonal
projector onto the PSD cone is
projKn
+(Y) =Udiag(λ+
1,...,λ+
n)U⊤.(354)
This map is a firmly non-expansive projector (see Lemma C.27).
J.4.2 Projection Onto the Unit-Diagonal Box
The setCn
unit:=/braceleftbig
S∈Rn×n:|Sij|≤1 (i̸=j), Sii= 1 (1≤i≤n )/bracerightbig
is an axis-aligned hyper-box with fixed
diagonal. Because the Frobenius norm decouples over coordinates, the orthogonal projector is entry-wise. For
anyY∈Rn×nthe projector ontoCn
unitdecouples entry-wise as:
/bracketleftbig
projCn
unit(Y)/bracketrightbig
ij=

1, i=j,
clip(Yij,−1,1), i̸=j,(355)
where clip(y,−1,1):=max{− 1,min{ 1,y}}. Note that for each off-diagonal coordinate the convex problem
min|z|≤1 (z−Yij)2yields the clip operator. Moreover, the diagonal constraint is enforced exactly. This
operationisfirmlynon-expansive(seeLemmaC.27), costs O(n2)arithmeticoperationsandcanbeimplemented
in-place (withΘ(n2)storage for the matrix itself).
Both projectors are firmly non-expansive; their averaged composition is an averaged non-expansive operator.
Alternating these maps yields a Fejér -monotone sequence with respect to the intersection, guaranteeing
convergence (see App. J.5 for details).
RemarkJ.2.If one omits entrywise clipping (i.e., uses only diag(S) =1as the second constraint), it can
be useful to additionally enforce ∥S∥F≤nto keep iterates uniformly bounded. When projCn
unitincludes
|Sij|≤1, the bound∥S∥F≤nholds automatically. Therefore, in order to project onto Kn
+∩Bn
F, where
Bn
F:={S∈Rn×n|∥S∥F≤n},we need an additional Frobenius-ball rescaling step. To enforce this additional
constraint, define
t:=∥λ+∥2=/parenleftign/summationdisplay
k=1(λ+
k)2/parenrightig1/2
, α := min/braceleftig
1,n
t/bracerightig
.(356)
The convex program minS⪰0,∥S∥F≤n∥S−projKn
+(Y)∥2
Fdecouples in the eigen-basis and yields aradial
projection. Because the eigenvectors are orthogonal, the joint projection onto the intersection Kn
+∩Bn
Fis
obtained by scaling the positive eigenvalues:
projKn
+∩Bn
F(Y) =Udiag(αλ+
1,...,αλ+
n)U⊤.(357)
Theorem J.3(DP guarantee for fast AAP all-pairs Gram release).Let ffull(E) =S=EE⊤∈Snbe
the all-pairs cosine Gram statistic. Fix an adjacency relation ∼on embedding matrices and assume the
global Frobenius sensitivity∆ full:=supE∼E′/vextenddouble/vextenddoubleffull(E)−f full(E′)/vextenddouble/vextenddouble
Fis finite. Let ε > 0,δ∈ (0,1), set
cε,δ:= 2log(2/δ)/ε2andσ2:=cε,δ∆2
full. SampleW∈Rn×nwith i.i.d. entries Wij∼N(0,σ2)and define
the symmetric noiseG :=1
2(W+W⊤)∈Sn. Let projCcoll:Sn→Sndenote the (deterministic) output of
Algorithm 6 run for either (i) a fixed number of iterations T, or (ii) any stopping time that is measurable with
respect to the noisy inputS+G(equivalently, depends only on the iterates /S+G, not on the raw data).
Release/hatwideS:=projCcoll/parenleftbig
ffull(E) +G/parenrightbig
. ThenE∝⇕⊣√∫⊔≀→/hatwideSis(ε,δ)-DP.
Proof.ByCorollaryC.10(Gaussianmechanismwithsymmetricaveraging), theintermediaterelease ffull(E)+G
is(ε,δ)–DP when σ2=cε,δ∆2
full. The mapping projCcoll(including any stopping rule measurable w.r.t. its input)
is a measurable post-processing off full(E) +G, hence the output /hatwideSremains(ε,δ)-DP by Lemma C.7.
97

arXiv preprint, ScoreShield
J.4.3 Fast Averaged Alternating Projection Algorithm
Our algorithm is provided in Algorithm 6.
Algorithm 6Fast Alternating Projection ontoC coll
1:Input:E∈Rn×d,ε>0,δ∈(0,1), sensitivity∆>0, toleranceτ >0ormaximum iterationsT∈N
2:Output:/hatwideS∈C coll⊆Rn×nwith(ε,δ)–DP guarantee s.t./vextenddouble/vextenddouble/hatwideS−projCcoll(S+W)/vextenddouble/vextenddouble
F≤τ
3:Construct:S←EE⊤.
4:DP noise:setσ2←cε,δ∆2
5:Sample noiseW∼N/parenleftbig
0,σ2In×n/parenrightbig
6:S′←S+1
2/parenleftbig
W+W⊤/parenrightbig
7:Initialize/hatwideS(0)←S′
8:fort= 0toT−1do
9:/* symmetrize current iterate */
10:Y t=1
2/parenleftbig/hatwideSt+/hatwideS⊤
t/parenrightbig
11:/* projection onto PSD ConeKn
+*/
12:Compute eigen-decompositionY t=Udiag(λ 1,...,λn)U⊤
13:λ+
k←max{0,λ k},∀k∈[n]
14:Pt←projKn
+(Yt) =Udiag(λ+
1,...,λ+
n)U⊤
15:ifFrobenius-ball constraint is enforced and∥P t∥F>nthen
16:P t←(n/∥P t∥F)Pt % radial projection ontoKn
+∩Bn
F
17:end if
18:/* projection onto Unit-Hyper-CubeCn
unit*/
19:Q t←projCn
unit(Yt)%(Q t)ii= 1;(Qt)ij= clip((Y t)ij,−1,1)fori̸=j
20:/* averaged update */
21:/hatwideSt+1←1
2/parenleftbig
Pt+Qt/parenrightbig
22:/* stopping tests */
23:r chg←∥/hatwideSt+1−/hatwideSt∥F
24:r psd←∥Yt−Pt∥F;r box←∥Yt−Qt∥F
25:ifr chg≤τandmax{r psd,rbox}≤τthen
26:break
27:end if
28:end for
29:/* PSD on return */
30:S avg←/hatwideSt+1
31:/hatwideS←projKn
+(1
2(Savg+S⊤
avg))
32:ifmin i/hatwideSii≤0then/hatwideS←/hatwideS+µIwith smallµ>0(e.g.,µ= 10−8∥/hatwideS∥F/n)
33:D←diag( /hatwideS)1/2,/hatwideS←D−1/hatwideS D−1
34:Output:/hatwideS
98

arXiv preprint, ScoreShield
J.5 Averaged Alternating Projection: Theoretical Guarantees
J.5.1 Foundational Definitions and Lemmas
Definition J.4(Bounded Linear Regularity).Let Hbe a (finite-dimensional) Hilbert space with norm
∥·∥. For a nonempty set A ⊂ H, define the point–set distance dist(x,A):=infa∈A∥x−a∥. Let
C1,C2⊂Hbe nonempty, closed, and convex, and set C:=C1∩C2̸=∅. ForR> 0, denote the closed ball
BR:={x∈H :∥x∥≤R} . The pair(C1,C2)is calledboundedly linearly regularif for every R> 0there exists
a constantκ R≥1such that
dist(x,C)≤κ Rmax/braceleftbig
dist(x,C 1),dist(x,C 2)/bracerightbig
,∀x∈B R.(358)
In our ScoreShield application,H= (Rn×n,⟨·,·⟩F)and∥·∥=∥·∥ F.
Definition J.5(Fejér Monotone Sequence).Let( X,⟨·,·⟩ )be a real Hilbert space, let ∥·∥be its induced
norm, and letC⊂Xbe non–empty and closed16. A sequence(S t)t≥0⊂Xis calledFejér monotone with
respect toCif
∥St+1−Z∥ ≤ ∥S t−Z∥,∀Z∈C,∀t≥0.(359)
Lemma J.6(Basic Properties of Fejér Monotone Sequences).Let(S t)be Fejér monotone w.r.t. a non–empty,
closed and convex setCin a Hilbert space. Then
(a)/parenleftbig
∥St−Z∥/parenrightbig
t≥0is non-increasing for eachZ∈Cand therefore convergent;
(b)The sequence(S t)is bounded;
(c)The distance sequenced t:=dist(St,C) = inf Z∈C∥St−Z∥is decreasing and convergent;
(d)If there exists a subsequence(S tk)that converges in norm to someZ ∞∈C, thenS t→Z∞in norm.
Proof.Properties (a)–(c) follow directly from Definition J.5. For (d), assumeS tk→Z∞∈Cin norm. By
Fejér monotonicity, the sequencea t:=∥St−Z∞∥is nonincreasing, hencelim t→∞atexists. Buta tk→0, so
limt→∞at= 0, i.e.,S t→Z∞(see also Lemma J.12).
Lemma J.7(Strong Quasi -Nonexpansive for Averaged Operator Tλ).LetH=(Rn×n,⟨·,·⟩F)be a real Hilbert
space (in our case, Rn×nwith the Frobenius inner product), and let C1,C2⊂Hbe nonempty, closed, convex sets.
Denote by projCithe orthogonal projector onto Ci,i= 1,2. Forλ∈(0,1)defineTλ:=λprojC1+ (1−λ)projC2.
Then the following hold.
(i)Firm nonexpansiveness (averagedness).Each projCiis firmly nonexpansive, hence1 /2–averaged. A
convex combination of1 /2–averaged (resp. firmly nonexpansive) operators is again1 /2–averaged (resp.
firmly nonexpansive). Consequently,
∥TλS−TλY∥2≤ ⟨TλS−TλY,S−Y⟩,∀S,Y∈H,(360)
andTλis1/2–averaged.
(ii)Strong quasi–nonexpansiveness (SQNE).For everyZ∈Fix(T λ)and everyS∈H,
∥TλS−Z∥2≤ ∥S−Z∥2− ∥TλS−S∥2.(361)
In particular, the Picard iteratesS t+1=TλStare Fejér monotone w.r.t. Fix(Tλ)and satisfy/summationtext
t≥0∥St+1−
St∥2
F<∞(hence∥S t+1−St∥F→0).
Proof.
(i)Each projectorprojCiis firmly nonexpansive:
∥projCi(S)−projCi(Y)∥2≤ ⟨projCi(S)−projCi(Y),S−Y⟩, i= 1,2.(362)
16Convexity ofCis not required for the definition itself, but almost every convergence theorem that uses Fejér monotonicity
assumesCis closed and convex.
99

arXiv preprint, ScoreShield
LetA :=projC1(S)−projC1(Y)andB :=projC2(S)−projC2(Y). Then TλS−TλY=λA+ (1−λ)B. Using
the convexity of the norm-square and dropping a nonpositive variance term,
∥λA+ (1−λ)B∥2≤λ∥A∥2+ (1−λ)∥B∥2.(363)
Applying firm nonexpansiveness ofprojC1andprojC2to the RHS gives
λ∥A∥2+ (1−λ)∥B∥2≤λ⟨A,S−Y⟩+ (1−λ)⟨B,S−Y⟩=⟨λA+ (1−λ)B,S−Y⟩.(364)
This is exactly Eq. 360. ThusT λis firmly nonexpansive, hence1
2–averaged.
(ii)LetZ∈Fix(T λ)so thatT λZ=Z. By (i),T λis firmly nonexpansive, so
∥TλS−TλZ∥2≤⟨TλS−TλZ,S−Z⟩.(365)
ButTλZ=Z, hence
∥TλS−Z∥2≤⟨TλS−Z,S−Z⟩.(366)
Expand the right-hand side using the polarization identity:
⟨TλS−Z,S−Z⟩=1
2/parenleftbig
∥TλS−Z∥2+∥S−Z∥2−∥TλS−S∥2/parenrightbig
.(367)
Rearranging gives
∥TλS−Z∥2≤∥S−Z∥2−∥TλS−S∥2,(368)
which is Eq. 361. This is the strong quasi–nonexpansiveness with parameter 1. Summing Eq. 361 over tgives/summationtext
t∥St+1−St∥2
F<∞.
Lemma J.8(Fixed Points ofT λvia a Convex Potential).Define the weighted proximity functional
Jλ(S):=λ
2dist(S,C 1)2+1−λ
2dist(S,C 2)2,S∈H,(369)
wheredist(S,C) :=∥S−projC(S)∥F. ThenJλis convex and Fréchet differentiable with
∇Jλ(S) =S−T λ(S).(370)
Consequently,
Fix(Tλ) = arg min
S∈HJλ(S).(371)
If moreoverC 1∩C2̸=∅, thenminJ λ= 0and
Fix(Tλ) = arg minJ λ=C1∩C2.(372)
Proof.For any closed convex C, the function f(S):=1
2dist(S,C)2=minZ∈C1
2∥S−Z∥2
Fis convex and
differentiable with∇f(S) =S−projC(S). Therefore
∇Jλ(S) =λ(S−projC1(S)) + (1−λ)(S−projC2(S)) =S−(λprojC1(S) + (1−λ)projC2(S)) =S−T λS.(373)
SinceJλis convex differentiable, ∇Jλ(S) = 0is equivalent toS ∈arg minJ λ. Thus Fix(Tλ) =arg minJλ. If
C1∩C2̸=∅, thenJ λ(S) = 0iffS∈C 1andS∈C 2, i.e.,S∈C 1∩C2.
Lemma J.9(Bounded BLR from relative-interior intersection).Let Hbe a finite-dimensional Hilbert space
with norm∥·∥, and letC1,C2⊂Hbe nonempty, closed, and convex with C:=C1∩C2̸=∅. Assume
ri(C1)∩ri(C2)̸=∅. Then(C1,C2)is boundedly linearly regular in the sense of Definition J.4; i.e., for every
R>0there existsκ R≥1such that
dist(x,C)≤κ Rmax{dist(x,C 1),dist(x,C 2)},∀x∈B R.(374)
100

arXiv preprint, ScoreShield
Proof.FixR> 0and consider the closed ball BR={x∈H :∥x∥≤R} . Under the qualification condition
ri(C1)∩ri(C2)̸=∅, the pair(C1,C2)satisfies a standard Slater-type constraint qualification for the convex
feasibility problem C1∩C2. A classical consequence is ametric inequality(a.k.a. bounded linear regularity/error
bound) on bounded sets: there exists a constantγ R>0such that
dist(x,C)2≤γR/parenleftbig
dist(x,C 1)2+dist(x,C 2)2/parenrightbig
,∀x∈B R,(375)
see, e.g., [8, Cor. 6] (and also [6, Sec. 5]). Now for anyx∈B R,
dist(x,C 1)2+dist(x,C 2)2≤2 max{dist(x,C 1),dist(x,C 2)}2.(376)
Combine this with Eq. 375 and take square-roots to obtain
dist(x,C)≤/radicalbig
2γRmax{dist(x,C 1),dist(x,C 2)},∀x∈B R.(377)
Thus Definition J.4 holds withκ R:=√2γR.
RemarkJ.10 (Restricting BLR to the iterate region).Suppose the iterates satisfyS t∈Bfor allt, where
B⊂His bounded (e.g., a Fejér ball). Pick any R> 0such thatB⊆BR. If BLR holds on BRwith constant
κR(for instance, by Lemma J.9), then the same inequality holds for allx∈Bwith thesameconstantκ R.
Lemma J.11(BLR for PSD cone and unit–diagonal box).Consider H= (Rn×n,⟨·,·⟩F)and setC1:=Kn
+
andC 2:=Cn
unit. Then(C 1,C2)is boundedly linearly regular: for everyR>0there existsκ R≥1such that
dist(S,C 1∩C2)≤κRmax{dist(S,C 1),dist(S,C 2)},∀S∈B R.(378)
Proof.In∈ri(C 1)∩ri(C 2), hence Lemma J.9 applies.
Lemma J.12(Opial’s Lemma in Finite Dimension [ 34]17).LetHbe a finite-dimensional Hilbert space and
letC⊂Hbe nonempty, closed, and convex. Let(S t)t≥0⊂Hbe a sequence such that:
(O1)For everyZ∈C, the limitlim t→∞∥St−Z∥exists.
(O2)Every norm-convergent subsequence(S tk)has its limit inC; i.e., ifS tk→S∞thenS∞∈C.
Then(St)converges in norm to some pointS ⋆∈C.
Proof.Pick anyZ 0∈C. By(O1), the real sequence rt:=∥St−Z0∥has a finite limit, hence is bounded.
Therefore(S t)is bounded. Since His finite-dimensional, every bounded sequence has a norm-convergent
subsequence [ 7,37]. Thus there exist indices tk↑∞and a pointS ∞∈Hsuch thatS tk→S∞in norm. By
(O2), we haveS ∞∈C. Apply(O1)withZ=S ∞∈Cto conclude that the limit L:=limt→∞∥St−S∞∥
exists. Along the subsequence tk,∥Stk−S∞∥ −→ 0, so necessarily L= 0. Hence∥St−S∞∥→ 0, i.e.,
St→S∞in norm. SettingS ⋆:=S∞∈Ccompletes the proof.
J.5.2 Convergence and Linear Rates for AAP in ScoreShield Regime (ii)
Theorem J.13(Convergence and Linear Feasibility Rates of AAP Under BLR).LetH= (Rn×n,⟨·,·⟩F)be
the Hilbert space of real n×nmatrices with Frobenius inner product. Let C1,C2⊂Hbe non-empty, closed,
convex withC1∩C2̸=∅. For a relaxation parameter λ∈ (0,1)define the averaged projection operator
Tλ=λprojC1+ (1−λ)projC218, where the orthogonal projector is defined as projC(S):=arg min Z∈C∥S−Z∥F.
Define point–set distance as dist(S,C):=∥S−projC(S)∥F. Given an initial matrixS 0∈H, iterateS t+1=
TλSt, t= 0,1,....
17See also the streamlined presentation in [2].
18Note that our iteration uses a convex combination of projectors Tλ=λprojC1+ (1−λ)projC2. This is a Krasnosel’ski˘ ı–Mann
[28,29] averaged projector scheme, distinct from the classical von Neumann alternating projections projC2(projC1(·))(composition)
[41].
101

arXiv preprint, ScoreShield
(i)Convergence and Optimality.The sequence(S t)t≥0converges in the Frobenius norm to someS ⋆∈
C1∩C2. Moreover,
Fix(Tλ) =C 1∩C2= arg min
S∈HJ(S),J(S) :=λ
2dist2(S,C 1) +1−λ
2dist2(S,C 2)(379)
andJ(S) = 0onC1∩C2. IfC1∩C2contains more than one element, the particular minimizer reached
may depend onS 0.
(ii)Linear Feasibility Decay on a Bounded Region dist(St,C1∩C2)under BLR.Assume the pair( C1,C2)
is BLR. There existsR>0such thatS t∈BR,∀tand that BLR hold onB Rwith constantκ R
dist/parenleftbig
S,C1∩C2/parenrightbig
≤κRmax/braceleftbig
dist(S,C 1),dist(S,C 2)/bracerightbig
,∀S∈B R.(380)
Letm := min{λ,1−λ}. Then
dist(St,C1∩C2)≤ρt
Rdist(S 0,C1∩C2), t= 0,1,... , ρ R:=/radicalbigg
1−m
κ2
R∈(0,1).(381)
(iii)Linear Decay in Norm under a Local Metric Error Bound.Let B:=/braceleftbig
S:∥S−Z∥F≤∥S0−Z∥F,∀Z∈
C/bracerightbig
be the Fejér ball containing all iterates. If there exist σ>0and a (necessarily unique)S ⋆∈C1∩C2
such that
∥S−S⋆∥F≤σdist(S,C 1∩C2),∀S∈B,(382)
then
∥St−S⋆∥F≤σρt
Rdist(S 0,C1∩C2)≤σρt
R∥S0−S⋆∥F.(383)
Thus∥St−S⋆∥Falso decaysR-linearly.
Proof.
(i)
By Lemma J.7, Tλis firmly nonexpansive and the Picard iterates are Fejér monotone w.r.t. Fix(Tλ), with/summationtext
t≥0∥St+1−St∥2
F<∞, hence∥St+1−St∥F→0. We verify Opial’s conditions from Lemma J.12 with
C=Fix(Tλ). For anyZ∈Fix (Tλ), Fejér monotonicity gives that ∥St−Z∥Fis nonincreasing, hence has a limit
(O1). For (O2), letS tk→S∞in norm. Then∥TλStk−Stk∥F=∥Stk+1−Stk∥F→0. SinceTλis continuous
(Lipschitz), TλStk→TλS∞, soTλS∞=S∞, i.e.,S∞∈Fix (Tλ). Opial’s lemma yieldsS t→S⋆∈Fix (Tλ).
Finally, Lemma J.8 gives Fix(Tλ) =arg minJ , and sinceC1∩C2̸=∅,minJ = 0and arg minJ =C1∩C2,
proving (i).
(ii)FixS∈BRand note that dist(TλS,C)2=minZ∈C∥TλS−Z∥2
F≤∥TλS−projC(S)∥2
F. By firm nonexpan-
siveness of each projector, convexity of ∥·∥2
F, and the projector Pythagorean inequality (see Lemma C.29), we
have
∥TλS−projC1∩C2(S)∥2
F (384a)
=∥λ(projC1(S)−projC1∩C2(S)) + (1−λ)(projC2(S)−projC1∩C2(S))∥2
F (384b)
≤λ∥projC1(S)−projC1∩C2(S)∥2
F+ (1−λ)∥projC2(S)−projC1∩C2(S)∥2
F (384c)
≤∥S−projC1∩C2(S)∥2
F−/bracketleftig
λ∥S−projC1(S)∥2
F+ (1−λ)∥S−projC2(S)∥2
F/bracketrightig
.(384d)
That is
dist(TλS,C1∩C2)2≤dist(S,C 1∩C2)2−/bracketleftig
λdist(S,C 1)2+ (1−λ)dist(S,C 2)2/bracketrightig
.(385)
Usingm= min{λ,1−λ}we have
λdist(S,C 1)2+ (1−λ)dist(S,C 2)2≥m(dist(S,C 1)2+dist(S,C 2)2)(386a)
≥mmax{dist(S,C 1)2,dist(S,C 2)2}(386b)
≥m
κ2
Rdist(S,C 1∩C2)2,(386c)
102

arXiv preprint, ScoreShield
where the last step uses BLR Eq. 380. Insert this lower bound in Eq. 384 to obtain
dist(TλS,C1∩C2)2≤dist(S,C 1∩C2)2−m
κ2
Rdist(S,C 1∩C2)2=/parenleftig
1−m
κ2
R/parenrightig
dist(S,C 1∩C2)2.(387)
Hence
dist(TλS,C1∩C2)≤ρRdist(S,C 1∩C2), ρ :=/radicalbigg
1−m
κ2
R∈(0,1).(388)
Iterating yields the geometric decrease Eq. 381.
(iii)If Eq. 382 holds on B, then for all t∥St−S⋆∥F≤σdist (St,C1∩C2)≤σρtdist(S0,C1∩C2), which gives
Eq. 383. The last inequality in the statement follows sincedist(S 0,C1∩C2)≤∥S 0−S⋆∥F.
Corollary J.14(Iteration Complexity).LetS t+1=TλStand letτ >0be a target tolerance (i.e., dist(St,C)≤
τis desired). Under the assumptions of Theorem J.13,
dist(St,C)≤ρtdist(S 0,C)⇒t≥log/parenleftbig
dist(S 0,C)/τ/parenrightbig
log(1/ρ)=O/parenleftbig
log(1/τ)/parenrightbig
, ρ=/radicalbigg
1−m
κ2.(389)
RemarkJ.15 (Practical Expected Iteration).SinceS 0=S+1
2/parenleftbig
W+W⊤/parenrightbig
andS∈C, we have dist(S 0,C)≤
∥1
2/parenleftbig
W+W⊤/parenrightbig
∥F. IfWiji.i.d.∼ N(0,σ2), we have
E/bracketleftig/vextenddouble/vextenddouble1
2/parenleftbig
W+W⊤/parenrightbig/vextenddouble/vextenddouble
F/bracketrightig
≤σ/radicalbigg
n(n+ 1)
2.(390)
With Gaussian mechanism σ= ∆/radicalbig
2 log(2/δ)/ε we have E/bracketleftbig
∥1
2/parenleftbig
W+W⊤/parenrightbig
∥F/bracketrightbig
≤∆
ε/radicalbig
n(n+ 1) log(2/δ) .
Hence, a practical expected iteration bound is
t≳log/parenleftig
∆
ετ/radicalbig
n(n+ 1) log(2/δ)/parenrightig
log(1/ρ).(391)
Therefore decreasing εorδ(more noise) increases the initial distance but does not change the geometric rate
ρ, which depends only on the geometry.
J.6 Symmetrization Prior to PSD Projection
BeforeeachPSDprojectionwereplacetheiterate /hatwideStbyitssymmetricpartY=1
2(/hatwideSt+/hatwideS⊤
t). Thistransformation
serves three purposes. (i) It preserves the inherent symmetry of the target Gram matrixS=EE⊤, ensuring
structural consistency throughout the alternating-projection procedure. (ii) It guarantees that the subsequent
eigen-decomposition used to realize projKn
+(·)is well-posed, thereby removes the skew-symmetric component
which cannot be reduced by PSD projection and would otherwise inflate the distance. (iii) Because for
any matrix the closest PSD point in Frobenius norm is obtained from its symmetric part, our explicit
symmetrization merely makes an implicit step in the original perturb-and-project algorithm of Cohen-Addad
et al.[14] transparent without altering privacy or utility guarantees. The operation costs only O(n2)flops,
which is negligible relative to theO(n3)eigen-decomposition that follows.
Lemma J.16(Symmetrization Invariance of the PSD Projection).Let /hatwideSt∈Rn×nbe any (not necessarily
symmetric) matrix and define its symmetric and skew-symmetric parts
Y:=1
2/parenleftbig/hatwideSt+/hatwideS⊤
t/parenrightbig
,K :=1
2/parenleftbig/hatwideSt−/hatwideS⊤
t/parenrightbig
.(392)
Then
min
X⪰0∥X−/hatwideSt∥2
F= min
X⪰0∥X−Y∥2
F+1
4∥/hatwideSt−/hatwideS⊤
t∥2
F,(393)
and every minimizer of the left-hand side is symmetric and coincides with a minimizer of the first term on the
right-hand side. In particular,
projKn
+(/hatwideSt) =projKn
+/parenleftbig
Y/parenrightbig
.(394)
103

arXiv preprint, ScoreShield
Algorithm 7Dykstra’s algorithm forprojC1∩C2(S0)
1:Input:S 0∈Sn, setsC 1,C2⊂Sn
2:InitializeS(0)←S 0,U(0)←0,V(0)←0
3:fork= 0,1,2,...,K−1do
4:Y(k)←projC1/parenleftbig
S(k)+U(k)/parenrightbig
5:U(k+1)←S(k)+U(k)−Y(k)
6:S(k+1)←projC2/parenleftbig
Y(k)+V(k)/parenrightbig
7:V(k+1)←Y(k)+V(k)−S(k+1)
8:end for
9:Return:S(K)
Proof.Write/hatwideSt=Y+KwithY⊤=YandK⊤=−K. Any feasibleX⪰0is symmetric, so
∥X−/hatwideSt∥2
F=∥X−(Y+K)∥2
F=∥X−Y∥2
F+∥K∥2
F,(395)
because⟨X−Y,K⟩F= 0(symmetric vs. skew–symmetric orthogonality). Since ∥K∥2
F=1
4∥/hatwideSt−/hatwideS⊤
t∥2
Fis
independent ofX, minimizing overX ⪰0yields Eq. 393. The independence also shows any minimizer must
minimize∥X−Y∥2
Fsubject toX⪰0, hence is symmetric and equal toprojKn
+(Y).
Corollary J.17(Distance Decomposition).Under our setup addressed in Theorem J.13, for any /hatwideStwe have
dist/parenleftbig/hatwideSt,Kn
+/parenrightbig2=dist/parenleftig
1
2(/hatwideSt+/hatwideS⊤
t),Kn
+/parenrightig2
+1
4∥/hatwideSt−/hatwideS⊤
t∥2
F.(396)
Proof.Immediate from Lemma J.16 by recognizing each distance as the objective value at the respective
minimizer.
J.7 Dykstra’s Algorithm for the Metric Projection ontoKn
+∩Cn
unit
Let consider the Hilbert space H:= (Sn,⟨·,·⟩ F)of real symmetric n×nmatrices equipped with the Frobenius
inner product. Define the closed convex sets
C1:=Kn
+={S∈Sn:S⪰0},C 2:=Cn
unit={S∈Sn:Sii= 1,|Sij|≤1 (i̸=j)}.
Given a (possibly non-symmetric) noisy matrix /tildewideS0∈Rn×narising from the DP perturbation step, we first
symmetrizeS 0:=1
2/parenleftbig/tildewideS0+/tildewideS⊤
0/parenrightbig
∈Sn. This symmetrization is a deterministic post-processing and, for the
PSD projection, does not change the result (Lemma J.16). The Euclidean (Frobenius) metric projection onto
the cosine-Gram feasibility set is
projC1∩C2(S0) = arg min
S∈C 1∩C21
2∥S−S 0∥2
F.(397)
Feasibility Enforcement versus Metric Projection.Our fast projection method based on AAP is a feasibility
method, i.e., it generates iterates whose distance to C1∩C2decreases (and is R-linear under bounded linear
regularity on the Fejér ball), but it does not, in general, return the metric projection ofS 0ontoC1∩C2. When
the best-approximation property is required, one may instead use Dykstra’s algorithm (see Algorithm 7),
which alternates the same two projectors with correction terms and converges to projC1∩C2(S0)[5,10,17,18].
Per outer iteration, both methods apply (i) one PSD-cone projection (dominated by an eigendecomposition)
and (ii) one entrywise projection onto C2. Dykstra additionally stores and updates two dense correction
matrices.
Theorem J.18(Convergence of Dykstra’s algorithm to the metric projection).Let Hbe a finite-dimensional
Hilbert space and let C1,C2⊂Hbe nonempty, closed, convex sets with C1∩C2̸=∅. LetS 0∈H, and let
(S(k))k≥0be the primal sequence produced by Algorithm 7. Then
S(k)−→projC1∩C2(S0)
in norm ask→∞.
104

arXiv preprint, ScoreShield
Proof.This is the classical convergence guarantee for Dykstra’s algorithm for projecting onto the intersection
of finitely many closed convex sets in finite-dimensional Hilbert spaces; see [5, 10, 17, 18].
J.8 Feasible Sets and Projection Decompositions
In this subsection we clarify the feasible set and projection decomposition used for regime (ii), and contrasts it
with the perturb-and-project construction of Cohen-Addadet al.[ 14]. Our projection uses the decomposition
A=Kn
+andB=Cn
unit, where
Kn
+={M∈Sn:M⪰0},Cn
unit={M∈Sn:Mii= 1,|Mij|≤1, i̸=j}.(398)
Projection onto Ais eigenvalue clipping, and projection onto Bsets the diagonal to one and clips the
off-diagonal entries to[−1,1].
Cohen-Addadet al.instead use a decomposition of the formA′=Kn
+∩Bn
FandB′=Cn
max, where
Bn
F={M∈Sn:∥M∥F≤n},Cn
max={M∈Sn:∥M∥ max≤1}.(399)
This decomposition does not enforce Mii= 1. Its Frobenius-ball step can radially rescale the positive spectrum,
and this rescaling may change the diagonal entries. Such a step is incompatible with exact preservation of the
cosine self-similarity constraintM ii= 1.
In our setting the radius- nFrobenius ball is unnecessary. IfM ∈C coll, then|Mij|≤1for all entries and
therefore
∥M∥2
F=/summationdisplay
i,jM2
ij≤n2,∥M∥ F≤n.(400)
Hence every feasible cosine Gram matrix already lies in Bn
F. The unit-diagonal constraint gives compactness
without adding a separate Frobenius-ball constraint.
105

arXiv preprint, ScoreShield
K Supplementary Details for Regime (ii): Extended Experiments
Evaluation overview.We evaluate settings in which downstream algorithms access only a symmetric
similarity matrix, or deterministic transformations of it. The experiments assess three questions:
(i)How much utility is retained after enforcing the cosine-Gram feasibility constraints on a privatized
Gram matrix?
(ii) How does the projected release compare with the corresponding non-private Gram baseline?
(iii)How does the proposed feasibility solver scale compared with an SDP baseline when both are tractable?
Some experiments also report the intermediate noisy Gram matrix before projection. When this baseline is
reported, it is labeledNoisy. For CIFAR-10 and CIFAR-100, the reported plots compare only the non-private
Gram and the projectedScoreShieldGram; the unprojected noisy baseline is not included in those figures.
K.1 Experimental Setup
Implementation.ExperimentsareimplementedinPythonusingPyTorch. Featureencodersareinstantiated
with timmand evaluated with deterministic preprocessing; no stochastic data augmentations are used.
Computation uses a single CUDA GPU when available and otherwise runs on CPU. The AAP solver runs for
at most500iterations and terminates early when the relative Frobenius-norm change falls below10−7. For
benchmarks involving random subsampling, each configuration is repeated R= 5times with distinct seeds.
We report mean±standard deviation and plot mean curves with shaded±standard-deviation bands.
Privacy parameters.We fix δ= 10−5and sweepεon a logarithmic grid (e.g., {0.1,1,10,100}) to cover
the privacy–utility range. We perturb the Gram entrywise via the Gaussian mechanism S′
ij=Sij+Zij,
Ziji.i.d.∼ N (0,σ2), with scale σ=σ(ε,δ,∆)calibrated from an ℓ2-sensitivity bound∆for a single entry (classic
sufficient vs. analytic calibration). We then enforce symmetry byS′←1
2(S′+S′⊤). The same calibration is
used across benchmarks.
Reported evaluation variants.Depending on the benchmark, we consider the following matrix variants:
1.Non-private(S): construct the cosine Gram from ℓ2-normalized embeddings, or from the task-specific
construction noted below. This serves as the non-private reference.
2.Noisy(S′): apply the Gaussian mechanism toSwith scale σ=σ(ε,δ,∆)calibrated for the adopted
adjacency model, and enforce symmetry byS′←1
2(S′+S′⊤). The resulting matrix need not be positive
semidefinite, need not have unit diagonal, and may have entries outside[−1,1].
3.ScoreShield( /hatwideS): apply a deterministic post-processing map to the noisy matrix to enforce the cosine-
Gram constraints. In the large-scale experiments, this step is implemented by AAP over the PSD cone,
the unit-diagonal affine set, and the entrywise box[−1,1].
The ‘Noisy’ variant is not reported for every benchmark.
Stability of graph-based methods.From the cosine GramSwe form an affinityA=1
2(S+1), clip to
[0,1], and set diag(A) =0. To control degrees and avoid isolates, we optionally sparsify with a symmetric
m-nearest-neighbor graph (retain edge( i,j)iffi∈NNm(j)orj∈NNm(i)). Spectral clustering is applied to
the symmetric normalized LaplacianL sym=I−D−1/2AD−1/2. Unless otherwise stated, we use an oracle- k
setting where kequals the number of ground-truth classes in the subsample (capped at10for CIFAR-10); a
fixed-kvariant is reported in ablations.
Backbones and datasets.Unless otherwise noted, we extract frozen image embeddings with DINOv2-
B/14 [35] (ImageNet-pretrained) using standard evaluation preprocessing (resize and center-crop). Vision
benchmarks are CIFAR-10/100 [ 26], and Oxford-IIIT Pets [ 36]. Non-vision benchmarks are STS-B [ 11] and
MovieLens-100K [21].
106

arXiv preprint, ScoreShield
K.2 Benchmark Suite and Metrics
All tasks consume only the released Gram matrix Sand therefore directly assess the usability of a DP Gram
release. Metrics are defined precisely below.
Unsupervised clustering (vision).
Goal:recover semantic clusters fromSalone.
Construction:for a random subsample of nimages, compute cosine similarities Sij=x⊤
ixj
∥xi∥∥xj∥fromℓ2-
normalized embeddings and form an affinityA=1
2(S+1)with diag(A) =0. Apply spectral clustering
(optionally with a symmetricm-NN sparsification).
Metric:Normalized Mutual Information (NMI) between predicted /hatwideYand labelsY,
NMI(Y,/hatwideY) =2I(Y;/hatwideY)
H(Y) +H(/hatwideY)∈[0,1].(401)
Report mean±std overR= 5subsamples per(ε,n).
k-NN classification (vision).
Goal:assign labels using neighborhoods induced byS.
Construction:with labelsy ∈{1,...,C}n, predict/hatwideyiby majority vote over the klargest off-diagonal entries
in rowiofS(excludeS ii).
Metric:Top-1 accuracy1
n/summationtextn
i=11{/hatwideyi=yi}. Ties are broken by class frequency, then index.
Instance retrieval (vision).
Goal:retrieve same-class instances using onlyS.
Construction:for each queryi, rankj̸=iby descendingS ij.
Metrics:Recall@K and mAP. Withr i(j)thej-th ranked neighbor andM ithe number of relevant items,
R@K =1
nn/summationdisplay
i=11{∃j≤K:y ri(j)=yi},mAP =1
nn/summationdisplay
i=11
Min−1/summationdisplay
j=1Prec@j·1{y ri(j)=yi}.(402)
Same/different verification (vision).
Goal:decide whether(i,j)share a label usingS ijas the score.
Metric:ROC–AUC computed from{(S ij,1{yi=yj})}i<j(diagonal excluded).
Semantic textual similarity (NLP).
Goal:preserve sentence-pair similarity.
Construction:embed STS-B sentences with SBERT and form a cosine GramS; for each labeled pair( u,v),
useSuv.
Metric:Spearman’s rank correlationρbetween{S uv}and human scores.
User-based collaborative filtering (recommender).
Goal:predict ratings from user–user similarity.
Construction:represent each user by a centered rating vector;Sis the user–user Gram. Predict /hatwideruiby a
similarity-weighted average over neighbors who rated itemi.
Metric:RMSE on a held-out test set,
RMSE =/radicaltp/radicalvertex/radicalvertex/radicalbt1
|T|/summationdisplay
(u,i)∈T/parenleftbig
/hatwiderui−rui/parenrightbig2.(403)
107

arXiv preprint, ScoreShield
K.3 How the Gram is Consumed
All downstream methods use only the Gram matrixS ∈Rn×n(or deterministic transforms thereof); no raw
features are accessed at inference time. For vision tasks, embeddings are ℓ2-normalized prior to forming cosine
similarities, soSis a cosine Gram.
-Clustering.We construct an affinityA=1
2(S+1), set diag(A) = 0, optionally apply a symmetric m-nearest-
neighbor sparsification for graph stability, and run spectral clustering with kchosen either by oracle ( k=
number of unique labels in the subsample) or fixed.
-k-NN classification.For each index i, we exclude self-similarity and assign /hatwideyiby majority vote over the k
largest off-diagonal entries in rowiofS; ties are resolved deterministically (class frequency, then index).
-Retrieval and verification.Retrieval ranks candidates j̸=iby descending Sij; metrics (e.g., Recall@K, mAP)
are computed from these rankings. Verification uses Sijas the decision statistic to produce ROC–AUC or
TMR@FMR.
-Semantic similarity (STS-B).For each labeled sentence pair( u,v), the predicted similarity is the single
Gram entryS uv; utility is measured via Spearman’sρagainst human scores.
-Recommender (MovieLens).A user–user Gram is formed from centered rating vectors; predicted ratings are
similarity-weighted averages over neighbors who rated the target item.
-Link prediction (if used).Candidate edges(u,v)are scored directly byS uvand ranked accordingly.
In all cases, the identical pipeline is applied to non-privateS, noisyS′, and ScoreShield projected /hatwideS.
K.4 Aggregation and Visualization
In stochastic settings with random n-subsampling, each configuration is repeated R= 5times; we report
µ±σacross repeats and plot mean curves with translucent ±1σbands. Deterministic settings with fixed
nreport single values. Runtime summaries report measured wall-clock time for the AAP feasibility solver
and, when the optimization terminates within the prescribed wall-time budget, for the SDP metric-projection
baseline at the same( ε,δ,∆,n). If the SDP solver does not terminate within the budget, the corresponding
entry is omitted rather than extrapolated.
K.5 Runtime Comparison with an SDP Metric-Projection Baseline
We compare AAP with an SDP metric-projection baseline for the nearest feasible cosine-Gram problem
min
S∥S−S′∥2
Fs.t.S⪰0,diag(S) =1,|S ij|≤1 (i̸=j).(404)
The SDP baseline is implemented in CVXPY using MOSEK, Clarabel, or SCS backends, subject to a fixed
wall-time budget. AAP is not an exact metric-projection solver; it is the feasibility-enforcement post-processing
used in the large-scale experiments. Its dominant per-iteration cost is the PSD projection, implemented
through an eigen-decomposition. The SDP baseline solves the Frobenius metric-projection problem above and
is included only for problem sizes where it terminates within the wall-time budget.
Figure K.1 reports a representative CIFAR-100 clustering runtime comparison at matched( ε,δ,∆). The
vertical axis is logarithmic. In the displayed configurations, AAP has lower wall-clock runtime than the SDP
metric-projection baseline whenever both are reported. These measurements are used only to document the
runtime behavior of the implementation, not to claim that AAP computes the exact metric projection.
K.6 Benchmarks
Each grid fixes the sensitivity∆by row and increases the failure probability δby column. The horizontal
axis is the privacy budget ε(log scale). Curves compare: (i) the non-private GramS, (ii) the noisy GramS′
obtained by the Gaussian mechanism calibrated for( ε,δ,∆), and (iii) the projected Gram /hatwideSproduced by our
ScoreShield fast alternating projection. An SDP projection baseline is included when numerically tractable.
Shaded ribbons show mean±1σoverR=5repeats (when subsampling is used).
108

arXiv preprint, ScoreShield
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 2
 (e)δ= 10−5,∆ = 2
 (f)δ= 10−2,∆ = 2
Figure K.1.CIFAR-100 clustering runtime comparison. Wall-clock execution time is reported in seconds on a
logarithmic scale for the AAP feasibility solver and the SDP metric-projection baseline when the latter terminates
within the prescribed wall-time budget.
K.6.1 CIFAR-10
Dataset and protocol.CIFAR-10 has 10 classes and 60K images [ 26]. For each run we uniformly subsample
n=64images, extract frozen DINOv2-B/14 embeddings, form a cosine Gram S, and evaluate two tasks that
consume only S: (i) pairwise same/different verification via ROC–AUC computed from {(Sij,1{yi=yj})}i<j;
and (ii) spectral clustering scored by NMI with oracle k=10. This setting isolates the small- nvision regime.
Results are in Figs. K.2–K.3.
109

arXiv preprint, ScoreShield
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 2
 (e)δ= 10−5,∆ = 2
 (f)δ= 10−2,∆ = 2
Figure K.2.CIFAR-10 verification AUC ( n= 64). Each row fixes∆; δincreases from left to right. The x-axis is ε(log
scale). Shaded regions indicate variability across runs.
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 2
 (e)δ= 10−5,∆ = 2
 (f)δ= 10−2,∆ = 2
Figure K.3.CIFAR-10 clustering NMI ( n= 64). Each row fixes the sensitivity parameter∆; δ; columns correspond to
increasingδ. The horizontal axis isεon a logarithmic scale. Shaded regions indicate variability across runs.
110

arXiv preprint, ScoreShield
K.6.2 CIFAR-100
Dataset and protocol.CIFAR-100 has 100 classes and 60K images. To enable comparison with the
SDP projection baseline (which becomes memory-bound for larger n) we report spectral clustering NMI at
n∈{ 64,100}. For clustering, kis set to the number of distinct labels present in the subsample (oracle k).
We omitk-NN and verification on CIFAR-100, as stable estimates in this setting require on the order of103
images (i.e.,≳10 per class), which is outside the feasible range for the SDP baseline. See Figs. K.4 and K.5.
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 1
 (e)δ= 10−5,∆ = 1
 (f)δ= 10−2,∆ = 1
(g)δ= 10−8,∆ = 2
 (h)δ= 10−5,∆ = 2
 (i)δ= 10−2,∆ = 2
Figure K.4.Results for CIFAR-100 utility NMI ( n= 64). Each row represents a fixed∆with δincreasing from left to
right. The horizontal axis isεon a logarithmic scale. Shaded bands show variability across repeated runs.
111

arXiv preprint, ScoreShield
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 1
 (e)δ= 10−5,∆ = 1
 (f)δ= 10−2,∆ = 1
(g)δ= 10−8,∆ = 2
 (h)δ= 10−5,∆ = 2
 (i)δ= 10−2,∆ = 2
Figure K.5.Results for CIFAR-100 utility NMI ( n= 100). Each row represents a fixed∆with δincreasing from left
to right. The horizontal axis isεon a logarithmic scale. Shaded bands show variability across repeated runs.
112

arXiv preprint, ScoreShield
K.6.3 STSBench
Semantic Textual Similarity (STS-B)[ 11]. We embed the n=2,552unique sentences with SBERT, con-
struct the sentence–sentence cosine Gram S, and evaluate utility as Spearman’s rank correlation ρbetween
{Suv}(u,v)∈Land the human similarity scores on the labeled pairsL(diagonal excluded; higher is better).
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 1
 (e)δ= 10−5,∆ = 1
 (f)δ= 10−2,∆ = 1
(g)δ= 10−8,∆ = 2
 (h)δ= 10−5,∆ = 2
 (i)δ= 10−2,∆ = 2
Figure K.6.Results for Spearman correlation ( ↑) for STSBench. Each row corresponds to a fixed∆, with δincreasing
from left to right. The x-axis shows ε(log scale), and the color hues indicate variance across repeated runs. For large n,
solving the SDP with linear system solvers (e.g., MOSEK) was computationally infeasible within the time or memory
budget.
113

arXiv preprint, ScoreShield
K.6.4 MovieLens
MovieLens–100K[ 21]. We form a user–user cosine Gram Sover then=943users using user–mean–centered
rating vectors, and evaluate a standard similarity-weighted neighborhood predictor on the official test split.
Utility is reported as RMSE on held-out ratings. (The SDP baseline is omitted at this scale due to solver
limitations.)
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 1
 (e)δ= 10−5,∆ = 1
 (f)δ= 10−2,∆ = 1
(g)δ= 10−8,∆ = 2
 (h)δ= 10−5,∆ = 2
 (i)δ= 10−2,∆ = 2
Figure K.7.Results for RMSE ( ↓) for MovieLens Benchmark. Each row corresponds to a fixed∆, with δincreasing
from left to right. The x-axis shows ε(log scale), and the color hues indicate variance across repeated runs. For large n,
solving the SDP with linear system solvers (e.g., MOSEK) was computationally infeasible within the time or memory
budget.
114

arXiv preprint, ScoreShield
K.6.5 Oxford Pets
Oxford–IIIT Pets[ 36]: 37 breeds,∼7.4K images. For each run, we sample n=2048images, extract frozen
DINOv2-B/14 embeddings, ℓ2-normalize, and form the cosine Gram S. We report three utilities: (i) ROC–AUC
for same/different verification from pair scores {Sij}(diagonal excluded); (ii)5-NN top-1 accuracy using each
Gram row as the neighborhood (self excluded); and (iii) instance retrieval Recall@1 from rankings induced by
S.
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 1
 (e)δ= 10−5,∆ = 1
 (f)δ= 10−2,∆ = 1
(g)δ= 10−8,∆ = 2
 (h)δ= 10−5,∆ = 2
 (i)δ= 10−2,∆ = 2
Figure K.8.Oxford-IIIT Pets same/different verification ROC–AUC ( ↑) atn= 2048. Each row fixes the sensitivity
parameter∆; columns correspond to increasing δ. The horizontal axis is εon a logarithmic scale. Shaded bands show
variability across repeated runs.
115

arXiv preprint, ScoreShield
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 1
 (e)δ= 10−5,∆ = 1
 (f)δ= 10−2,∆ = 1
(g)δ= 10−8,∆ = 2
 (h)δ= 10−5,∆ = 2
 (i)δ= 10−2,∆ = 2
Figure K.9.Results for KNN Accuracy ( ↑) (n= 2048). Each row represents a fixed∆with δincreasing from left to
right. The horizontal axis isεon a logarithmic scale. Shaded bands show variability across repeated runs.
116

arXiv preprint, ScoreShield
(a)δ= 10−8,∆ = 0.5
 (b)δ= 10−5,∆ = 0.5
 (c)δ= 10−2,∆ = 0.5
(d)δ= 10−8,∆ = 1
 (e)δ= 10−5,∆ = 1
 (f)δ= 10−2,∆ = 1
(g)δ= 10−8,∆ = 2
 (h)δ= 10−5,∆ = 2
 (i)δ= 10−2,∆ = 2
Figure K.10.Results for Retrieval Recall@1 ( ↑) (n= 2048). Each row represents a fixed∆with δincreasing from left
to right. The horizontal axis isεon a logarithmic scale. Shaded bands show variability across repeated runs.
117

arXiv preprint, ScoreShield
Appendix References
[1] Robert J Adler and Jonathan E Taylor.Random fields and geometry. Springer, 2007.
[2] Aleksandr Arakcheev and Heinz H Bauschke. On opial’s lemma.arXiv preprint arXiv:2503.22004, 2025.
[3]Borja Balle and Yu-Xiang Wang. Improving the gaussian mechanism for differential privacy: Analytical calibration
and optimal denoising. InInternational conference on machine learning, pp. 394–403. PMLR, 2018.
[4]Donald Bamber. The area above the ordinal dominance graph and the area below the receiver operating
characteristic graph.Journal of mathematical psychology, 12(4):387–415, 1975.
[5]Heinz H Bauschke and Jonathan M Borwein. Dykstra’ s alternating projection algorithm for two sets.Journal of
Approximation Theory, 79(3):418–443, 1994.
[6]Heinz H Bauschke and Jonathan M Borwein. On projection algorithms for solving convex feasibility problems.
SIAM review, 38(3):367–426, 1996.
[7]Heinz H Bauschke and Patrick L Combettes. Correction to: convex analysis and monotone operator theory in
hilbert spaces. InConvex analysis and monotone operator theory in Hilbert spaces, pp. C1–C4. Springer, 2020.
[8]Heinz H Bauschke, Jonathan M Borwein, and Wu Li. Strong conical hull intersection property, bounded linear
regularity, jameson’s property (g), and error bounds in convex optimization.Mathematical Programming, 86(1):
135–160, 1999.
[9]Jeremiah Blocki, Avrim Blum, Anupam Datta, and Or Sheffet. The johnson-lindenstrauss transform itself preserves
differential privacy. In2012 IEEE 53rd Annual Symposium on Foundations of Computer Science, pp. 410–419.
IEEE, 2012.
[10] James P Boyle and Richard L Dykstra. A method for finding projections onto the intersection of convex sets in
hilbert spaces. InAdvances in Order Restricted Statistical Inference: Proceedings of the Symposium on Order
Restricted Statistical Inference held in Iowa City, Iowa, September 11–13, 1985, pp. 28–47. Springer, 1986.
[11]Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. SemEval-2017 task 1: Semantic
textual similarity multilingual and cross-lingual focused evaluation. InProceedings of the 11th International
Workshop on Semantic Evaluation (SemEval-2017), pp. 1–14, 2017.
[12]Mahawaga Arachchige Pathum Chamikara, Peter Bertok, Ibrahim Khalil, Dongxi Liu, and Seyit Camtepe. Privacy
preserving face recognition utilizing differential privacy.Computers & Security, 97, 2020.
[13]Yifeng Chu and Maxim Raginsky. Talagrand meets talagrand: Upper and lower bounds on expected soft maxima
of gaussian processes with finite index sets.arXiv preprint arXiv:2502.06709, 2025.
[14]Vincent Cohen-Addad, Tommaso d’Orsi, Alessandro Epasto, Vahab Mirrokni, and Peilin Zhong. Perturb-and-
project: differentially private similarities and marginals. InProceedings of the 41st International Conference on
Machine Learning, pp. 9161–9179, 2024.
[15] Lori E Dodd and Margaret S Pepe. Partial auc estimation and regression.Biometrics, 59(3):614–623, 2003.
[16]Wei Dong, Yuting Liang, and Ke Yi. Differentially private covariance revisited.Advances in Neural Information
Processing Systems, 35:850–861, 2022.
[17]Richard L Dykstra. An algorithm for restricted least squares regression.Journal of the American Statistical
Association, 78(384):837–842, 1983.
[18]Richard L Dykstra. An iterative procedure for obtaining i-projections onto the intersection of convex sets.The
annals of Probability, pp. 975–984, 1985.
[19]Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin,
Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron,
Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu, Francesco
Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen
Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin
Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi, Dan
Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu
Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander
Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur
Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu,
118

arXiv preprint, ScoreShield
BobakShahriari, BrycePetrini, CharlieChen, CharlineLeLan, ChristopherA.Choquette-Choo, CJCarey, Cormac
Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar
Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick
Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Plucińska, Harman Singh, Harsh
Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie,
Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji,
Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine,
Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma,
Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva, Oskar
Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe
Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob
Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy,
Shruti Sheth, Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu,
Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad
Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed,
Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira,
Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan
Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov,
Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet,
Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem,
Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard Hussenot. Gemma 3 technical
report, 2025. URLhttps://arxiv.org/abs/2503.19786.
[20]Nicolas Grislain. Rag with differential privacy. In2025 IEEE Conference on Artificial Intelligence (CAI), pp.
847–852. IEEE, 2025.
[21]F Maxwell Harper and Joseph A Konstan. The movielens datasets: History and context.ACM transactions on
interactive intelligent systems (TIIS), 5(4):1–19, 2015.
[22]Jiazhen Ji, Huan Wang, Yuge Huang, Jiaxiang Wu, Xingkun Xu, Shouhong Ding, ShengChuan Zhang, Liujuan
Cao, and Rongrong Ji. Privacy-preserving face recognition with learnable privacy budgets in frequency domain.
InEuropean Conference on Computer Vision, pp. 475–491. Springer, 2022.
[23]Tianxi Ji and Pan Li. Less is more: Revisiting the gaussian mechanism for differential privacy. In33rd USENIX
Security Symposium (USENIX Security 24), pp. 937–954, 2024.
[24]Tatsuki Koga, Ruihan Wu, Zhiyuan Zhang, and Kamalika Chaudhuri. Privacy-preserving retrieval-augmented
generation with differential privacy.arXiv preprint arXiv:2412.04697, 2024.
[25]Satyapriya Krishna, Kalpesh Krishna, Anhad Mohananey, Steven Schwarcz, Adam Stambler, Shyam Upadhyay,
and Manaal Faruqui. Fact, fetch, and reason: A unified evaluation of retrieval-augmented generation. InProceedings
of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers), pp. 4745–4759, 2025. Dataset: google/frames-benchmark.
[26]Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, University of Toronto,
2009. Tech Report.
[27]Mingxin Li, Yanzhao Zhang, Dingkun Long, Chen Keqin, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie,
An Yang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Qwen3-vl-embedding and qwen3-vl-reranker: A unified
framework for state-of-the-art multimodal retrieval and ranking.arXiv preprint arXiv:2601.04720, 2026.
[28]KRASNOSEL’SKII MA. Two comments on the method of successive approximations.Usp. Math. Nauk, 10:
123–127, 1955.
[29]W Robert Mann. Mean value methods in iteration.Proceedings of the American Mathematical Society, 4(3):
506–510, 1953.
[30]Brianna Maze, Jocelyn Adams, James A Duncan, Nathan Kalka, Tim Miller, Charles Otto, Anil K Jain, W Tyler
Niggel, Janet Anderson, Jordan Cheney, et al. Iarpa janus benchmark-c: Face dataset and protocol. In2018
international conference on biometrics (ICB), pp. 158–165. IEEE, 2018.
[31]Jean Jacques Moreau. Décomposition orthogonale d’un espace hilbertien selon deux cônes mutuellement polaires.
Comptes rendus hebdomadaires des séances de l’Académie des sciences, 255:238–240, 1962.
119

arXiv preprint, ScoreShield
[32]Junki Mori, Kazuya Kakizaki, Taiki Miyagawa, and Jun Sakuma. Differentially private synthetic text generation
for retrieval-augmented generation (rag).arXiv preprint arXiv:2510.06719, 2025.
[33]John Muschelli III. Roc and auc with a binary predictor: a potentially misleading metric.Journal of classification,
37(3):696–708, 2020.
[34]Zdzisław Opial. Weak convergence of the sequence of successive approximations for nonexpansive mappings.
Bulletin of the American Mathematical Society, 73(4):591–597, 1967.
[35]Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre
Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech
Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel
Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2:
Learning robust visual features without supervision.Transactions on Machine Learning Research, 2024. Featured
Certification.
[36]Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In2012 IEEE conference
on computer vision and pattern recognition, pp. 3498–3505. IEEE, 2012.
[37] Walter Rudin. Principles of mathematical analysis.3rd ed., 1976.
[38]Michel Talagrand. Sudakov-type minoration for gaussian chaos processes.Israel Journal of Mathematics, 79(2):
207–224, 1992.
[39]Kunal Talwar, Abhradeep Guha Thakurta, and Li Zhang. Nearly optimal private lasso.Advances in Neural
Information Processing Systems, 28, 2015.
[40]Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, Sindhu Raghuram Panyam,
Sara Smoot, Iftekhar Naim, Joe Zou, Feiyang Chen, et al. Embeddinggemma: Powerful and lightweight text
representations.arXiv preprint arXiv:2509.20354, 2025. Model: google/embeddinggemma-300M.
[41] John Von Neumann. On rings of operators. reduction theory.Annals of Mathematics, 50(2):401–485, 1949.
[42]Ruihan Wu, Erchi Wang, and Yu-Xiang Wang. Beyond per-question privacy: Multi-query differential privacy for
rag systems. InNeurIPS 2025 Workshop: Reliable ML from Unreliable Data, 2025.
[43]Ruihan Wu, Erchi Wang, Zhiyuan Zhang, and Yu-Xiang Wang. Private-rag: Answering multiple queries with
llms while keeping your data private.arXiv preprint arXiv:2511.07637, 2025.
[44]Hanfang Yang, Kun Lu, Xiang Lyu, and Feifang Hu. Two-way partial auc and its properties.Statistical methods
in medical research, 28(1):184–195, 2019.
[45]Mengmeng Yang, Tianqing Zhu, Lichuan Ma, Yang Xiang, and Wanlei Zhou. Privacy preserving collaborative
filtering via the johnson-lindenstrauss transform. In2017 IEEE Trustcom/BigDataSE/ICESS, pp. 417–424. IEEE,
2017.
[46]Zhiyong Yang, Qianqian Xu, Shilong Bao, Yuan He, Xiaochun Cao, and Qingming Huang. When all we need
is a piece of the pie: A generic framework for optimizing two-way partial auc. InInternational Conference on
Machine Learning, pp. 11820–11829. PMLR, 2021.
120