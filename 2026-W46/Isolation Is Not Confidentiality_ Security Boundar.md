# Isolation Is Not Confidentiality: Security Boundaries of Geometric Access Control in RAG

**Authors**: Steve Nouyep, Maxime Puys, Sébastien Salva

**Published**: 2026-11-10 13:25:54

**PDF URL**: [https://arxiv.org/pdf/2611.14061v1](https://arxiv.org/pdf/2611.14061v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems increasingly rely on vector databases containing documents governed by different access-control policies. Recent search-preserving geometric approaches enforce such access control by structuring the embedding space so that transformed embeddings, which we call protected representations, remain retrievable without exposing information across policy boundaries.
  We provide a unified security analysis of this paradigm, covering policy isolation, representation confidentiality, query forgery resistance, and policy-membership hiding. We consider both a curious cloud service provider observing the protected representations stored in the vector database and a curious user holding aligned pairs of original embeddings and protected representations. Evaluating the available implementations of the works proposing these geometric transformation mechanisms across three encoders, we find that geometric transformations preserve retrieval utility and provide strong policy isolation, but do not provide the other security properties. A curious user can recover the original embeddings from a few hundred pairs and forge policy-compatible queries at a higher cost that grows with encoder dimension for secret per-policy transformations. Moreover, policy-dependent geometry can leak policy membership even when unauthorized retrieval is prevented. These results show that policy isolation, confidentiality, forgery resistance, and membership hiding are distinct security objectives.

## Full Text


<!-- PDF content starts -->

Isolation Is Not Confidentiality: Security
Boundaries of Geometric Access Control in RAG
Steve Nouyep, Maxime Puys, and Sébastien Salva
Université Clermont Auvergne, CNRS, Clermont Auvergne INP, LIMOS, 63000
Clermont-Ferrand, France
{steve_durand.nouyep_noundjio, maxime.puys, sebastien.salva}@uca.fr
Abstract. Retrieval-Augmented Generation (RAG) systems increasingly
rely on vector databases containing documents governed by different
access-control policies. Recent search-preserving geometric approaches
enforce such access control by structuring the embedding space so that
transformed embeddings, which we call protected representations, remain
retrievable without exposing information across policy boundaries.
We provide a unified security analysis of this paradigm, covering policy
isolation, representation confidentiality, query forgery resistance, and
policy-membership hiding. We consider both a curious cloud service
provider observing the protected representations stored in the vector
database and a curious user holding aligned pairs of original embeddings
and protected representations. Evaluating the available implementations
of the works proposing these geometric transformation mechanisms across
three encoders, we find that geometric transformations preserve retrieval
utility and provide strong policy isolation, but do not provide the other
security properties. A curious user can recover the original embeddings
from a few hundred pairs and forge policy-compatible queries at a higher
cost that grows with encoder dimension for secret per-policy transforma-
tions. Moreover, policy-dependent geometry can leak policy membership
even when unauthorized retrieval is prevented. These results show that
policy isolation, confidentiality, forgery resistance, and membership hiding
are distinct security objectives.
Keywords:Retrieval-Augmented Generation·Access Control·Embed-
ding Security·Geometric Transformations·Vector Retrieval·Privacy
1 Introduction
Retrieval-Augmented Generation (RAG) is more and more considered to ground
large language models (LLMs) with external knowledge, which is encoded as
embeddings by an encoder (a model that maps text to numeric vectors) and
stored in a latent embedding space to enable similarity search between queries
and knowledge [1]. In enterprise deployments, a vector database may contain
documents governed by different access-control policies [2]. This creates a mis-
match: embedding similarity search determines which documents are relevant
to a query, but not whether the user is authorized to access them. As a result,
arXiv:2611.14061v1  [cs.CR]  10 Nov 2026

2 S. Nouyep et al.
semantically similar documents governed by different policies may be retrieved
together unless authorization is enforced separately.
Existing approaches address this problem using metadata-based filtering [21],
policy-specific indexes [23], or cryptographic protection [26,29]. However, these
approaches either enforce authorization outside the embedding space, require
separate storage structures, or apply additional mechanisms to perform similarity
search over encrypted embeddings. A different approach is to encode authorization
directly into embeddings by means of geometric transformations that preserve
similarity. We refer to this paradigm assearch-preserving geometric access control.
Each policy is associated with a secret transformation of the embedding space; the
transformed embedding, which we call theprotected representation, is what the
vector database stores, and authorized queries undergo the same transformation.
Recent systems implement this approach using policy-specific transformations
of stored embeddings [8 –10]. With policy separation built into the embedding
space, these approaches can enforce access control directly through similarity
search while preserving retrieval utility.
However, preserving retrieval does not guarantee other security properties.
Policy isolation and representation confidentiality are distinct objectives. A
transformation may prevent cross-policy retrieval during normal operation while
still being learnable from informative observations. Moreover, the same structure
used to separate policies may reveal policy membership. This raises a broader
question: what security properties does search-preserving geometric access control
actually provide?
We study this question through similarity preservation and four security
properties: policyisolation, representationconfidentiality, queryforgeryresistance,
and policy-membership hiding. We consider two adversaries. A curious cloud
service provider has access to protected representations but no policy keys or
original embeddings for the target policy. A curious user additionally holds aligned
pairs of original embeddings and protected representations for a target policy.
We use this case to evaluate the residual protection once enough observations
are available to learn a policy-specific transformation. Our goal is therefore not
to claim a cryptographic break of a particular system, but to characterize the
security boundary of the search-preserving geometric access-control paradigm.
Our evaluation leads to three main findings. First, policy-specific geomet-
ric transformations can preserve retrieval utility while providing strong policy
isolation. Second, isolation does not imply confidentiality or forgery resistance:
aligned observations make the evaluated transformations learnable, so a curious
user can recover the original embeddings from a few hundred pairs and forge
policy-compatible queries at a higher, dimension-dependent cost. Third, geometric
separation can leak policy membership, even when policy isolation is preserved.
Contributions.Our contributions are threefold:
–We provide a unified security characterization of geometric access control
for multi-policy RAG, treating retrieval utility and four security properties

Isolation Is Not Confidentiality: Geometric Access Control in RAG 3
(policy isolation, representation confidentiality, query forgery resistance, and
policy-membership hiding) as distinct objectives.
–We analyze the learnability of policy transformations from aligned pairs
of original embeddings and protected representations, and distinguish rep-
resentation confidentiality (recovering the original embedding) from query
forgery resistance (constructing a policy-compatible query), two objectives
that target different objects and need not hold together.
–We evaluate representative published mechanisms (policy-specific rotations,
key-parametrized dimension augmentation, and distance-comparison-pre-
serving encryption) and their fusion, on the authors’ own implementations
across three encoders. A curious user recovers the original embeddings from
a few hundred aligned pairs and forges policy-compatible queries at a higher,
dimension-dependentcost,andpolicy-dependentgeometricseparationexposes
policy membership, including for the cryptographic scheme.
Outline.Section 2 reviews existing approaches to access control in RAG and
situates search-preserving geometric access control among them. Section 3 defines
the system and adversary models. Section 4 presents the security analysis of
the four properties under a generic transformation model. Section 5 reports
the experimental evaluation of the three mechanisms and their fusion across
three encoders. Section 6 discusses implications and limitations, and Section 7
concludes.
2 Background and Related Work
2.1 Access Control in RAG
Access control in RAG systems is commonly enforced outside the embedding
space. Metadata-based filtering associates documents or chunks with authoriza-
tion attributes and restricts retrieval to those accessible to the user [21,22,27].
Policy-specific indexes provide a similar separation by maintaining separate re-
trieval structures for different authorization domains [23], and LLM-driven or
multitenant gateways enforce authorization outside the vector database [24,25].
These approaches provide explicit access-control semantics but require authoriza-
tion information to remain available during retrieval.
Recent work has explored integrating access policies directly into the em-
bedding space [8 –10]. Trans-RAG applies policy-dependent transformations to
embeddings so that queries and documents under the same policy remain com-
patible while different policies are separated [8]; other schemes use policy-specific
rotations [9] or key-parametrized dimension augmentation [10]. This design re-
duces reliance on metadata filtering during vector search and allows multiple
policies to coexist within the same retrieval infrastructure. Our work focuses on
the security properties of search-preserving geometric access control rather than
on access-control enforcement alone.

4 S. Nouyep et al.
2.2 Search-Preserving Protection of Embeddings
Related work studies transformations that protect vector representations while
preserving the similarity relationships needed for retrieval. Distance-comparison-
preserving encryption (DCPE) [11] and related constructions [13] aim to hide
the underlying representation while allowing a server to perform similarity-based
operations. Other approaches obscure embeddings while preserving retrieval
utility, using embedding-space alignment [14] or cell-keyed residual splitting [15].
This creates an important tension. A transformation must preserve enough
structure for similarity search, but hiding that structure is precisely what makes
the original embedding difficult to recover. In particular, deterministic transfor-
mations can become learnable from aligned pairs of original embeddings and
protected representations. Classical results already show that distance- and
property-preserving transformations are vulnerable to known-plaintext and infer-
ence attacks that recover the transformation or the underlying data [16,17,20,30].
Unsupervised methods can even align embedding spaces without paired supervi-
sion [18,19], so aligned observations are a sufficient but not a necessary condition
for learning such a mapping. We extend this idea to the policy-specific RAG
setting and, beyond reconstruction, to query forgery and policy-membership expo-
sure. We therefore treat similarity preservation and representation confidentiality
as separate properties rather than assuming that one implies the other.
2.3 Representation Leakage and Inversion Attacks
Learned embeddings and intermediate representations can leak information about
their inputs. Prior work has demonstrated reconstruction and inversion attacks
that recover inputs from such vectors, including attacks based on corresponding
input–representation pairs [32 –34]. Related attacks recover data from the access
and query-leakage patterns of encrypted nearest-neighbor search [31]. These
results motivate evaluating what an adversary can learn beyond the intended
retrieval operation.
Our setting differs from conventional embedding inversion because the pro-
tected representation is generated by a policy-specific transformation. This leads
to two related but distinct attack objectives. An adversary may either (i) recover
the original embedding from a protected representation, or (ii) learn the transfor-
mation well enough to construct a policy-compatible query. Query forgery does
not require recovering the original embedding and therefore cannot be inferred
from reconstruction performance alone.
2.4 Positioning of This Work
Prior work has studied policy-aware retrieval transformations and embedding
inversion attacks separately. RAG privacy surveys [35] and provably secure de-
signs [28] discuss related risks but do not examine search-preserving geometric
access control. We bridge these gaps by jointly analyzing policy isolation, repre-
sentation confidentiality, query forgery resistance, and policy-membership hiding.
Table 1 summarizes this positioning.

Isolation Is Not Confidentiality: Geometric Access Control in RAG 5
Trusted server holds secret transformsf pUntrusted vector databaseDocuments
policiesp 1, p2, p3
EncoderΦ
publicv′
i=fp(ei)
policy-specific transformci ei v′
i
User
authorized forpEncoderΦ
publicv′
q=fp(eq)
same keyq eqv′
q
top-kprotected representation space
p1 p2
p3v′
q
same-policy representations rank first
Curious CSP: allv′, no key
Curious user: also(e i, v′
i)
Fig. 1.System and threat model. Document and query embeddings from the public
encoder Φare transformed by a policy-specific secret transformation fpinside the trusted
server; only the protected representations v′reach the untrusted vector database, where
same-policy representations rank first. The two adversaries differ in the information
available to them: the CSP observes protected representations, whereas the curious user
additionally holds aligned pairs of original embeddings and protected representations.
Table 1.Positioning of this work. For each property, a circle indicates whether it is
a stated design objective of the approach ( full,partial, absent), while ✓marks
the properties evaluated in this work;∗indicates an open-source implementation. The
circles and checkmarks do not indicate whether a property is actually satisfied; this is
determined by our security analysis and experiments. DCPE is included as a search-
preserving cryptographic reference, not as an access-control mechanism.
Policy Representation Forgery Membership
Approach isolation confidentiality resistance hiding
Metadata filtering [21,22]
Policy-specific indexes [23]∗
Cryptographic retrieval [26,29]∗
DCPE [11,13]∗
Geometric access control [8,10]∗, [9]
This work (evaluation)✓ ✓ ✓ ✓
3 System and Threat Model
3.1 System Model
Figure 1 summarizes the system and threat model. We consider a multi-policy
RAG system in which documents governed by different access-control policies
are stored in a vector database. A public encoder Φmaps each document chunk
cito an embedding ei=Φ(ci)∈Rd. Each policy pis associated with a secret
transformation fp. The transformations are applied by atrusted serverthat
holds the policy keys, whereas the vector database that stores the results is

6 S. Nouyep et al.
untrusted. We assume that embeddings are L2-normalized and that retrieval
ranks candidates by cosine similarity, i.e., the inner product of unit-norm vectors.
During ingestion, the server applies the corresponding transformation to each
embedding:
v′
i=fp(ei),(1)
and stores the resulting protected representationv′
iin the vector database.
At query time, an authenticated user is associated with one or more autho-
rized policies. For a query q, the server computes eq=Φ(q)and applies the
transformation for an authorized policy before submitting the transformed query
to the vector database. The retrieved chunks are then returned to a trusted LLM
for response generation.
Policy keys never leave the server. The vector database stores only protected
representations, and similarity search is performed by a nearest-neighbor engine
(e.g., FAISS [7]). The provider can access these representations but never the
policy keys.
This work focuses on the security of the protected representations used
for retrieval. Document confidentiality at rest and security of the generation
component are therefore outside the scope of our analysis.
3.2 Adversary Model
We consider honest-but-curious adversaries that follow the retrieval protocol but
may analyze protected representations available to them. All adversaries know
the public encoder Φbut do not possess the secret transformation key of the
target policy. We distinguish two levels of adversarial knowledge:
–Curious Cloud Service Provider (CSP).The provider operating the vector
database has access to the protected representations stored in the vector
database but no corresponding original embeddings for the target policy and
no policy key. This models the information available to an untrusted provider.
–Curious User.The adversary is a legitimately enrolled user that is authorized
under one policy but not under the target policy. For its own authorized
policy, it obtains pairs of original embeddings and protected representations
through the normal system interface. For the target policy, we assume that it
additionally obtains a bounded set of such pairs through collusion with the
CSP; this is the most capable adversary considered in this work. Because the
encoder Φis public, each such pair, which we call analigned pair, provides
an observation:
(ei, v′
i) = (Φ(c i), fp(ei)).(2)
These observations allow the user to estimate the policy-specific transforma-
tion without obtaining its secret key. This learnability from aligned pairs is a
known weakness of distance-preserving transformations [16,17]; we examine
its implications for both the confidentiality of policy-specific transformations
(Section 4.3) and their resistance to query forgery (Section 4.4).

Isolation Is Not Confidentiality: Geometric Access Control in RAG 7
For forgery evaluation, we additionally assume that the curious user can
submit a self-constructed protected representation of a query for similarity search
under the target policy, without possessing the target policy key or going through
the trusted server’s normal query transformation. The user therefore attempts to
construct a target-policy-compatible query using the transformation estimated
from the available aligned observations. This models a compromised or colluding
client and is distinct from the passive CSP capability. This capability is not
part of normal operation; instead, it characterizes the residual protection of the
transformation once informative observations become available.
We do not consider compromise of the trusted server or disclosure of policy
keys. Active attacks on the retrieval infrastructure, such as index tampering,
denial of service, or malicious document insertion, are outside the scope of this
work. Query forgery is evaluated separately as an authorization attack against
similarity search.
4 Security Analysis
We analyze the security properties of search-preserving geometric access con-
trol through a component model of the underlying geometric transformations
(Section 4.1) and under the threat model defined in Section 3. We treatsimilar-
ity preservationas a functional requirement: the transformation must preserve
the similarity relationships needed for retrieval, which we evaluate through re-
trieval quality relative to the unprotected embeddings (Section 5). Beyond this
requirement,westudyfoursecurityproperties:policyisolation,representationcon-
fidentiality, query forgery resistance, and policy-membership hiding. A document
containing the complete derivations of this section ( docs/derivations.pdf ) is
available in the anonymized artifact.1
4.1 A Component Model of the Geometric Family
We represent the considered geometric mechanisms through a common component
model. The policy-dependent components may include an orthogonal map Rp(a
key-dependent coordinate permutation is a particular case, since a permutation
matrix is orthogonal), a geometric anchor ap∈Rm, where mis the anchor-block
dimension and mmust be at least the number of policies for mutually orthogonal
anchors, a bounded perturbation n, and a possibly nonlinear map g. The resulting
protected representation is:
v′=N([g(R pe+n) ;Ma p]),(3)
where Mscales the anchor and N(·)renormalizes the result to unit length.
Here gdenotes a possibly nonlinear map that some constructions add to hinder
inversionoftheprotectedrepresentation.Itistheidentitymap,denoted g=id,for
the linear constructions, whereas Trans-RAG applies a bounded nonlinearity [8].
1https://anonymous.4open.science/r/geoac-rag-repro

8 S. Nouyep et al.
Table 2.Mapping of geometric components to the evaluated mechanisms ( Rp: policy-
specific orthogonal map; Map: policy anchor; Noise: additive noise or bounded pertur-
bationn; Nonlin.: nonlinear mapg).
MechanismR pMapNoise Nonlin.
Trans-RAG [8]✓–✓ ✓†
Subspace rotation [9] ✓– – –
Key-param. [10] –‡✓– –
DCPE [11,12]✓–✓–
†Trans-RAG appliesg(x) = tanh(βx)/βelementwise withβ= 0.1; the complete
mechanism is evaluated in Section 5.
‡Key-param. [10] applies a rotation shared across all policies (not policy-specific,
henceR pmarked absent); only its anchor is policy-dependent.
Only Rpis orthogonal: once n,g, and the renormalization are applied, the overall
transformation fpis no longer an isometry. The isolation analysis of Section 4.2
therefore takes g=idand is first stated without n; the effect of gandnis
evaluated empirically in Section 5.
An authorized query is transformed with the same policy-specific components,
without the document-side perturbation:
v′
q=N([g(R peq) ;Ma p]).(4)
We denote the noise-free linear signal-and-anchor component by up(e) =
[Rpe;Ma p]; it is the object of the isolation analysis below.
Relation to existing approaches.The component model is not a new access-
control mechanism. It is an analytical abstraction of the published constructions
(Table 2); all empirical attacks are performed on the published mechanisms and
their fusion (Section 5).
4.2 Policy Isolation
A query authorized under policy pshould retrieve documents governed by prather
than documents belonging to another policy. To quantify residual cross-policy
retrieval, we define two complementary rates: thequery violation rate(QVR),
the fraction of queries whose top- kcontains at least one unauthorized document
(how manyqueries leak), and theunauthorized chunk rate(UCR), the fraction of
retrieved chunks that are unauthorized (how muchleaks). Both are zero under
perfect isolation.
Under the cosine-similarity retrieval setting of Section 3.1, we derive the
conditions for isolation between policies. When a policy anchor is used, as
in [10], policy separation follows from its geometric contribution, and we seek
the condition on Mthat ensures every same-policy document ranks above every

Isolation Is Not Confidentiality: Geometric Access Control in RAG 9
cross-policy document, regardless of content. For an authorized query equnder
policy pand a document egoverned by policy por by another policy p′̸=p, we
assume unit-norm embeddings ( ∥e∥=∥eq∥= 1), an orthogonal map ( R⊤
pRp=I),
unit-norm and mutually orthogonal anchors ( ∥ap∥= 1and ⟨ap, ap′⟩= 0for
p̸=p′), andg= id. The same-policy inner product before normalization is:
⟨up(eq), up(e)⟩=⟨R peq, Rpe⟩+M2⟨ap, ap⟩
=⟨e q, e⟩+M2,(5)
where we useR⊤
pRp=Iand∥a p∥= 1, whereas for a cross-policy pair it is:
⟨up(eq), up′(e)⟩=⟨R peq, Rp′e⟩+M2⟨ap, ap′⟩
=⟨R peq, Rp′e⟩,(6)
since⟨a p, ap′⟩= 0forp̸=p′.
For the noise-free idealization, in which each protected representation is up(e),
the squared norm is:
∥up(e)∥2=∥R pe∥2+M2∥ap∥2= 1 +M2.(7)
Similarly, ∥up(eq)∥2= 1 + M2, so both cosine similarities have the same
denominator1 +M2:
simsame =⟨eq, e⟩+M2
1 +M2,(8)
and,
simcross=⟨Rpeq, Rp′e⟩
1 +M2.(9)
Because the embeddings are normalized, −1≤ ⟨e q, e⟩ ≤ 1, while by the
Cauchy–Schwarz inequality,
|⟨Rpeq, Rp′e⟩| ≤1.(10)
The common denominator therefore does not affect the relative ranking. The
same-policy numerator is at least M2−1, and the cross-policy numerator is
at most1by Eq. (10). Therefore, every same-policy pair ranks above every
cross-policy pair ifM2−1>1, which gives:
M2>2.(11)
The anchor is not strictly necessary for probabilistic separation. With inde-
pendently keyed, uniformly random orthogonal maps, cross-policy inner products
concentrate around zero in high dimension, with fluctuations of order O(1/√
d),
where dis the embedding dimension. The anchor instead provides the determin-
istic separation guarantee of Eq. (11). The effect of the bounded perturbation
is evaluated empirically in Section 5. This establishes the intended retrieval
property, but does not imply that the transformation or policy membership is
hidden from an observer.

10 S. Nouyep et al.
4.3 Representation Confidentiality
The original embedding should not be readily recoverable from the protected
representation available to an adversary. We measure this property through
functional reconstruction resistance: the number of aligned pairs required to
reproduce the retrieval behavior of the original embedding (Section 5).
The component model shows that a search-preserving transformation may pre-
vent cross-policy retrieval while remaining learnable from sufficiently informative
observations. For the curious user, the aligned observations have the form:
(ei, v′
i) = (e i, fp(ei)),(12)
where eiis the original embedding and v′
iis its protected representation. For a
linear content-dependent transformation, these pairs constrain the policy-specific
mapping. In particular, for an orthogonal transformation Rp, sufficiently many
linearly independent pairs allow the mapping to be estimated by orthogonal
Procrustesalignment[6].Thisisanalogoustoknown-plaintextattacksondistance-
preserving transformations [16,17], which we extend here to the RAG access-
control setting. Thus, keeping the policy transformation secret does not by itself
prevent it from being learned. We therefore evaluate reconstruction cost under
different observation budgets in Section 5.
4.4 Query Forgery Resistance
Using the policy transformation estimated from aligned pairs (Section 3), the
curious user should not be able to construct a query accepted under a policy it
is not authorized for. We considerconstructive forgery: the adversary estimates
the target transformation from prior observations and applies it to a fresh query,
rather than replaying a captured legitimate query.
Forgery is evaluated separately from reconstruction because it targets a
differentobject:apolicy-compatiblequeryratherthananoriginalembedding.The
forged query competes against the genuine protected representations of the target
policy, so its success depends on the accuracy of the estimated transformation,
not on the fidelity of any single reconstructed embedding.
4.5 Policy-Membership Hiding
An observer of the vector database should not be able to reliably infer which
policy a protected representation belongs to. This property is distinct from
policy isolation: a system may prevent unauthorized retrieval while still revealing
policy membership through the geometry of its representations. Policy-dependent
geometric separation can make membership observable. In particular, policy-
specific anchors give representations from the same policy a common geometric
component, while policy-specific transformations may create additional patterns
that distinguish policies. We therefore evaluate membership inference separately
from retrieval isolation, by measuring how reliably an observer can infer the
policy of a protected representation.

Isolation Is Not Confidentiality: Geometric Access Control in RAG 11
5 Experimental Evaluation
We evaluate the security properties introduced in Section 4 on published search-
preserving geometric mechanisms and their fusion, using the authors’ own im-
plementations. The evaluation addresses four questions: (i) do the mechanisms
preserve retrieval utility while enforcing policy isolation? (ii) how difficult is it to
recover original embeddings from protected representations? (iii) how difficult is
it to forge a query for a policy the adversary is not authorized for? and (iv) can
an observer infer policy membership from the vector database?
5.1 Experimental Setup
We evaluate the mechanisms on the datasets and encoders summarized in Ta-
ble 3. We run three published mechanisms with public implementations: the
query-centric transformation of Trans-RAG [8], the key-parametrized dimension
augmentation of [10], and DCPE through the IronCore SDK [12]. Using the
authors’ own implementations ensures that the measurements characterize the
published mechanisms rather than a reinterpretation.
We additionally evaluate afusionthat applies Trans-RAG’s transformation
followed by the key-parametrized augmentation of [10], to test whether composing
the two search-preserving geometric access-control mechanisms strengthens their
security properties. DCPE is evaluated separately as a distance-comparison-
preserving encryption scheme and is not part of the fusion.
Policy isolation is evaluated on a balanced MedQuAD sample with attribute-
based access-control (ABAC) metadata based on document type, project, and
region, defining nine policies. This corpus provides the multi-policy setting needed
to measure unauthorized retrieval. Retrieval utility is evaluated separately on
MS MARCO [5], a passage-ranking benchmark with real-world search queries,
to measure whether the transformations preserve semantic retrieval quality. We
additionally evaluate retrieval utility on three BEIR datasets [3] to test whether
this behavior extends across different retrieval corpora.
Reconstruction and forgery are evaluated on a designated target-policy subset
of MedQuAD containing 3,500 stored embeddings and 300 held-out queries.
This controlled single-policy setting isolates the cost of learning the target
transformation and using it for reconstruction or query forgery.
Retrieval utility is measured using Recall@10 and nDCG@10, while policy
isolation is measured using QVR and UCR (Section 4.2). Attack success is
measured functionally, by the overlap between the top-10results obtained with
the estimated and true policy transformations. We denote by n⋆
τthe smallest
number of aligned pairs required to reach an overlap of τ. We use τ= 0.8as
the primary threshold and τ= 0.9as a stricter secondary threshold. Thus, n⋆
τ
measures the empirical number of observations required to reproduce at least τ
of the reference top-10results; it is not a cryptographic security guarantee.
Reconstruction and forgery use the same linear least-squares estimator. We
fit the policy transformation from the aligned pairs using the Moore–Penrose
pseudo-inverse; for the anchor-based mechanism, the constant policy anchor is

12 S. Nouyep et al.
Table 3.Evaluation corpora, tasks, and sizes. Reported sizes are the indexed corpora
and query sets. The MedQuAD ABAC sample is balanced across nine policies (400
documents per policy); the full corpus contains 54,269 documents.
Corpus Purpose Docs Queries Policies
MS MARCO [5] Retrieval utility3,047 306none
BEIR NFCorpus [3] Retrieval utility3,633 323none
BEIR SciFact [3] Retrieval utility5,183 300none
BEIR FiQA [3] Retrieval utility57,638 648none
MedQuAD ABAC sample [4] Policy isolation3,600 3609
MedQuAD target-policy subset Reconstruction, forgery3,500 3001
Encoders: MiniLM (d=384, primary), MPNet (d=768), GTE-large (d=1024).
recovered from a single pair. We use the unconstrained least-squares solution
rather than an orthogonality-constrained one, so that it also accommodates
DCPE’s global scaling while remaining robust to its additive perturbation. The
attackcost n⋆
τis measured ona single witness sampleusinga coarse pairgrid, with
100-pair increments from 100 to 600 and larger steps up to 3,500. Consequently,
differences such as 200 versus 300 pairs should not be interpreted as precise one-
step estimates. Isolation experiments are repeated with three random seeds. All
experiments run on a Linux server (Ubuntu 24.04) with an Intel Xeon Gold 5416S
CPU and an NVIDIA L40 GPU. Embeddings are encoded on the GPU, while
attack-cost measurements use a single CPU thread.
5.2Does Geometric Transformation Preserve Retrieval and
Isolation?
Without policy-specific geometry, the baseline provides no policy isolation. Every
query returns at least one unauthorized document (QVR= 1 .00), while about8–9
of the10returned chunks are unauthorized (UCR ≈0.86). Across the evaluated
geometric configurations, both QVR and UCR are zero, with zero variance across
seeds.
On the policy-isolation corpus, geometric isolation raises authorized within-
document Recall@10 from 0.28 for the unfiltered baseline to 0.44, matching an
oracle pre-filter that removes unauthorized documents before ranking (0.44). Pol-
icy isolation is evaluated only on the policy-labeled MedQuAD corpus. Retrieval
quality is likewise preserved across the three BEIR collections and the three
encoders: Recall@10 stays within 0.01 of the unprotected reference in all nine
dataset–encoder combinations.
Consistent with the analysis of Section 4.2, independently keyed per-policy
orthogonalmaps(therotationofTrans-RAG[8]andthekey-dependentcoordinate
permutation of DCPE [11]) already produce zero measured cross-policy retrieval,
while the anchor of the key-parametrized augmentation [10] yields the same
isolation (Table 4).

Isolation Is Not Confidentiality: Geometric Access Control in RAG 13
For retrieval utility on MS MARCO, the unprotected reference system achieves
Recall@10= 0 .986and nDCG@10= 0 .699. The geometric transforms preserve
retrieval: a noise-free rotation or anchor is order-preserving, while mechanisms
that add a per-vector perturbation (Trans-RAG’s blinding noise and DCPE’s
perturbation) affect the ranking only marginally, keeping the top-10agreement
with the unprotected ranking at R@10 ≥0.91across the three encoders (Table 4).
Metric sensitivity of dimension augmentation.The key-parametrized augmen-
tation of [10] adds a large-valued coordinate per policy, whose magnitude is
calibrated for a specific similarity metric. With the article’s α= 103under
cosine similarity, or the code’s α= 109under the Euclidean metric it targets,
retrieval is preserved (Recall@10 ≥0.98across the three encoders). Porting
the code’s α= 109to a cosine-based store, the common RAG configuration,
collapses retrieval (Recall@10= 0 .02) while isolation still holds, because, with
M=αin Eq.(8), a difference ∆cosbetween two underlying cosine scores be-
comes ∆cos/ (1 +M2)and can fall below floating-point resolution, making the
resulting similarities numerically indistinguishable. The mechanism is therefore
metric-dependent, which is a portability caveat rather than a flaw.
5.3 Can Protected Representations Be Reconstructed?
Table 4 reports the reconstruction cost for the curious user on the three published
mechanisms and their fusion, run on the authors’ own code across three encoders
of increasing dimension. The same table also reports forgery and membership
results, discussed in Sections 5.4 and 5.5.
The reconstruction results show two main observations. First, across the
three encoders, increasing the encoder dimension from d= 384to d= 1024
does not increase the observed reconstruction cost: all evaluated transformations
are reconstructible within200–400aligned pairs (Figure 2). This is consistent
with the relatively low effective dimensionality of the embedding spaces: the
participation ratio, which summarizes how many directions carry most of the
variance, is 54, 53, and 64 for the three encoders, while 90% of the variance
is captured by 159, 182, and 291 directions, respectively. The larger encoder
therefore does not raise the observed reconstruction budget; the unprotected
embeddings are likewise recovered within300pairs.
Second, added noise or nonlinearity limits thefidelityof reconstruction but
not itscost. Trans-RAG’s blinding noise and nonlinearity [8], and the bounded
perturbation of DCPE [11], cap the top-10overlap below one (0 .86–0.94), whereas
the key-parametrized augmentation [10], being a constant per-policy coordinate,
leaves reconstruction of the rotated content unimpeded (ceiling1 .00). Yet all reach
the operational success threshold within a comparable pair budget. Policy-specific
rotations and anchors therefore do not increase the observed reconstruction cost.
5.4 Can the Same Observations Forge Policy-Compatible Queries?
The aligned pairs used for reconstruction also let the curious user estimate the
policy transformation and forge a query under a policy it is not authorized for.

14 S. Nouyep et al.
Table 4.Published mechanisms and fusion across encoder dimensions. Reconstruction
and forgery report n⋆
0.8and the overlap ceiling at the largest budget. R@10 is the same-
policy retrieval overlap between protected and unprotected top-10results; ARI measures
policy-membership clustering by the curious CSP. Isolation is ✓when QVR =UCR≈0.
Shaded cells mark a failed property: isolation if QVR >0; confidentiality and forgery
resistance if the attack ceiling reaches τ= 0.8; membership hiding if ARI ≥0.5. Pair
countsn⋆
0.8are costs, not verdicts, and are left unshaded.
Reconstruction Forgery
MechanismdIso.n⋆
0.8ceil.n⋆
0.8ceil. R@10 ARI
No access control384 ✗300 1.00n/a n/a1.00 0.00
Trans-RAG [8]384✓300 0.86 500 0.86 0.93 1.00
768✓300 0.93 1,000 0.90 0.95 1.00
1,024✓400 0.87 1,800 0.84 0.91 1.00
Key-param. aug. [10]384✓300 1.00 300 0.98 1.00 1.00
768✓200 1.00 200 1.00 1.00 1.00
1,024✓300 1.00 300 1.00 1.00 1.00
DCPE [11,12]384✓300 0.90 500 0.90 0.97 0.97
768✓300 0.92 1,000 0.91 0.97 1.00
1,024✓400 0.91 1,200 0.90 0.96 1.00
Fusion (TR⊕aug.)384✓200 0.86 500 0.86 0.93 1.00
768✓200 0.94 1,000 0.90 0.95 1.00
1,024✓400 0.88 1,800 0.84 0.91 1.00
Table 4 reports forgery cost alongside reconstruction. Forgery reaches0 .84–1.00
of the legitimate top-10overlap across the three published mechanisms and their
fusion, so none resists the attack. Its cost, however, differs from reconstruction
in how it scales with encoder dimension. For mechanisms whose target-policy
transformation is secret (Trans-RAG, DCPE, and the Trans-RAG ⊕augmentation
fusion), the attacker must estimate the full target transformation from the aligned
pairs. Forgery cost therefore grows with encoder dimension, from500to1 ,800
pairs asdincreases from384to1024(Figure 2).
The key-parametrized augmentation [10] behaves differently: it applies a
fixed rotation shared across all policies and separates policies only by a constant
per-policy anchor. Because the same rotation appears in every policy’s pairs, an
adversary enrolled in any policy can estimate it from its own authorized pairs,
while only the low-dimensional anchor remains target-specific. The reported
n⋆therefore counts the pairs needed from the adversary’s own policy; for the
target policy, a single pair suffices to recover the target anchor (Section 5.1).
Forgery thus does not scale with dand stays at200–300pairs, the same cost as
reconstruction. For the mechanisms with a secret target-policy transformation,
forgery costs more than reconstruction (500versus300pairs at d= 384). This
higher cost is consistent with the stricter requirement imposed by forgery: the

Isolation Is Not Confidentiality: Geometric Access Control in RAG 15
384 768 1,02403006009001,2001,5001,800
Encoder dimensiondAttack costn⋆
0.8(aligned pairs)(a) Reconstructionv′→e
Trans-RAG
Key-param. aug.
DCPE
Fusion (TR⊕aug.)
384 768 1,02403006009001,2001,5001,800
Encoder dimensiond(b) Query forgerye→v′
q
Fig. 2.Attack cost n⋆
0.8versus encoder dimension, with one line per mechanism (values
from Table 4; the same symbol denotes a mechanism in both panels). (a) Reconstruction
stays within 200–400 aligned pairs as dgrows from 384 to 1024. (b) Query forgery grows
with dfor mechanisms whose per-policy transformation is secret (Trans-RAG, DCPE,
and their fusion), reaching 1,200–1,800 pairs, while the key-parametrized augmentation,
whose rotation is shared across policies, stays at 200–300. Markers are slightly horizon-
tally offset where values coincide.
estimated transformation must produce a query that retrieves the target policy’s
genuine protected representations.
5.5 Does Geometric Isolation Leak Policy Membership?
The curious CSP clusters the stored protected representations using k-means,
without any aligned pair or policy key. The Adjusted Rand Index (ARI) measures
agreement between the inferred clusters and the true policy partition. Across the
three encoders, Trans-RAG, the key-parametrized augmentation [10], and their
fusion reach ARI= 1 .00, while DCPE reaches ARI= 0 .97–1.00, compared with
ARI≈0.001for the unprotected baseline (Table 4). Thus, policy membership is
highly recoverable even without aligned pairs or policy keys, including for DCPE,
a cryptographic distance-comparison-preserving scheme.
This exposure is consistent with embedding anisotropy. Learned embeddings
concentrate around a common mean, so applying a distinct per-policy key pro-
duces distinct per-policy geometry, whether through a rotated or permuted mean
for the orthogonal-map mechanisms or through an added coordinate for the
anchor. These differences remain separable by unsupervised clustering. The same
geometric separation that supports policy isolation can therefore expose policy
membership even when unauthorized retrieval is prevented.

16 S. Nouyep et al.
5.6 Summary of Findings
Across the evaluated mechanisms, policy isolation does not imply reconstruc-
tion resistance, forgery resistance, or policy-membership hiding. All geometric
mechanisms achieve the measured isolation, yet policy membership remains re-
coverable from the protected representations. Increasing encoder dimension does
not increase the observed reconstruction cost, while forgery cost increases with
dimension for mechanisms with a target-policy-specific transformation. Adding
noise or nonlinearity limits reconstruction fidelity but not the observed pair
budget, and the evaluated fusion does not increase the attack costs.
The unprotected baseline provides a complementary reference: it fails isolation
(QVR =1.00/UCR =0.86) but does not expose policy membership (ARI =0.00).
The geometric mechanisms reverse these two properties: they provide measured
isolation while exposing policy membership.
6 Discussion
The results clarify the security boundary of search-preserving geometric access
control. Geometric transformations can enforce policy separation directly in the
embedding space, but policy isolation does not imply confidentiality, forgery
resistance, or policy-membership hiding. Once the curious user holds aligned
pairs, the evaluated mechanisms remain learnable for both reconstruction and
query forgery, while their policy-dependent geometry can expose membership
information.
6.1 Limitations and Security Boundary
The reconstruction and forgery results assume that the curious user can obtain
aligned pairs of original embeddings and protected representations for the target
policy. This provides more information than is normally available to the CSP and
is therefore best understood as a test of the residual protection once informative
observations are available, rather than as an assumption that such observations
are always available in practice. The evaluation covers the published mechanisms
considered in this work and their fusion, not the full space of search-preserving
constructions. The measured attack costs therefore characterize these mechanisms
under the evaluated threat model and do not constitute cryptographic security
guarantees for the broader paradigm. Our formal analysis of isolation takes g=id;
the nonlinearity of Trans-RAG is retained in the empirical evaluation but is not
modeled analytically. Overall, the results support treating policy isolation as
a distinct security property rather than as a proxy for broader confidentiality
or anti-forgery guarantees. The required protection therefore depends on the
security objective and the adversarial capabilities being considered.
6.2 Towards Mitigations
The results also suggest several directions for mitigation. First, query authoriza-
tion should not rely solely on the geometric transformation: a trusted server

Isolation Is Not Confidentiality: Geometric Access Control in RAG 17
could authenticate transformed queries before they reach the vector database,
preventing a user who has learned the transformation from directly submitting
forged queries. Second, limiting the number of observations available under a
single policy transformation, for example through periodic re-keying or stricter
separation of protected representations, may reduce the effectiveness of attacks
based on aligned pairs. These measures address different parts of the problem
and should be evaluated explicitly in future work. When representation confiden-
tiality is a primary requirement, cryptographic retrieval mechanisms may also be
preferable to relying on geometric separation alone.
7 Conclusion
Search-preserving geometric access control can provide effective policy isolation
by embedding policy separation into the embedding space. Our analysis and
experiments show that this isolation does not imply representation confidentiality,
query forgery resistance, or policy-membership hiding. With aligned pairs of
original embeddings and protected representations, a curious user can recover
those embeddings from a few hundred observations and forge policy-compatible
queries at a higher, dimension-dependent cost for secret per-policy transforma-
tions. Composing published mechanisms does not substantially increase these
costs. These results show that search-preserving geometric access control should
be evaluated beyond retrieval effectiveness and policy isolation. Future work
will evaluate mitigation strategies and extend the analysis to broader classes of
search-preserving constructions.
Ethical statement.This work evaluates search-preserving geometric access-control
mechanisms using controlled experiments, publicly available data, and implemen-
tations released as part of scientific publications. Our evaluation is limited to
research implementations intended as proofs of concept; to our knowledge, none
of the implementations evaluated in this work is known to be deployed as part
of a commercial product. The attacks target experimental representations and
retrieval interfaces in these research implementations. No real-world system was
compromised and no sensitive user data were accessed. In the event of accep-
tance, we plan to contact the authors of the evaluated implementations following
a responsible-disclosure process before the final publication of the work. The
reproducibility artifact is described below.
Open Science.All code, dataset-preparation scripts, and configurations required
to reproduce the reported results, including the corpus statistics of Table 3 and
the reconstruction, forgery, and membership costs of Table 4, are available as an
anonymized artifact: https://anonymous.4open.science/r/geoac-rag-rep
ro. The repository provides scripts for reproducing the main experiments, gener-
ating the reported tables and figures, and evaluating the considered geometric
mechanisms and adversarial procedures. The artifact also contains a document
with the complete derivations of Section 4. Long-term archival storage is provided
through Zenodo.

18 S. Nouyep et al.
Acknowledgement.ThisworkwassupportedbytheIndustrialChaironReliable
and Confident Use of LLMs2and by the MIAI Cluster, France 2030 (ANR-23-
IACL-0006).
References
1.P. Lewiset al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP
Tasks,” inAdvances in Neural Information Processing Systems (NeurIPS), 2020.
2.V. Karpukhinet al., “Dense Passage Retrieval for Open-Domain Question Answer-
ing,” inProc. Conf. Empirical Methods in Natural Language Processing (EMNLP),
2020.
3.N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, and I. Gurevych, “BEIR: A Het-
erogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models,”
inProc. NeurIPS Datasets and Benchmarks Track, 2021.
4.A. Ben Abacha and D. Demner-Fushman, “A question-entailment approach to ques-
tion answering,”BMC Bioinformatics, vol. 20, art. 511, 2019. doi: 10.1186/s12859-
019-3119-4.
5.T. Nguyen, M. Rosenberg, X. Song, J. Gao, S. Tiwary, R. Majumder, and L. Deng,
“MS MARCO: A Human Generated MAchine Reading Comprehension Dataset,” in
Proc. Workshop on Cognitive Computation (CoCo@NeurIPS), 2016.
6.P. H. Schönemann, “A generalized solution of the orthogonal Procrustes problem,”
Psychometrika, vol. 31, no. 1, pp. 1–10, 1966.
7.J. Johnson, M. Douze, and H. Jégou, “Billion-Scale Similarity Search with GPUs,”
IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2021.
8.Y. Liu, K. Peng, W. Zhang, F. Yuan, C. Cao, W. Lu, and Y. Liu, “Trans-RAG:
Query-Centric Vector Transformation for Secure Cross-Organizational Retrieval,”
inProc. 31st Int. Conf. on Database Systems for Advanced Applications (DASFAA),
LNCS, vol. 16538, Springer, 2026, pp. 571–587. doi: 10.1007/978-981-92-0372-7_35.
9.T. K. Sharma, “Geometric Access Control: Securing Vector Retrieval in RAG
Systems via Isomorphic Subspace Projections,” TechRxiv preprint, February 2026.
doi: 10.36227/techrxiv.177161986.64628984/v1.
10.T. Thuillier, L. Lavaur, J. François, R. State, and W. Ferguson, “Key-Parametrized
Embeddings for Access Control in Retrieval-Augmented Generation,” inProc. IEEE
COMPSAC Workshops (20th Int. Workshop on Security, Trust, and Privacy for
Software Applications, STPSA), 2026. doi: 10.1109/COMPSAC69091.2026.00386.
11.G. Fuchsbauer, R. Ghosal, N. Hauke, and A. O’Neill, “Approximate Distance-
Comparison-Preserving Symmetric Encryption,” inProc. 13th Conf. on Secu-
rity and Cryptography for Networks (SCN), LNCS, Springer, 2022, pp. 117–144.
doi: 10.1007/978-3-031-14791-3_6. Full version: IACR Cryptology ePrint Archive,
Report 2021/1666.
12.IronCore Labs, “Cloaked AI: Encrypted Vector Embeddings,” product documenta-
tion and ironcore-alloy SDK, https://ironcorelabs.com/docs/cloaked-ai/ ,
https://github.com/IronCoreLabs/ironcore-alloy, accessed September 2026.
13.H. Ye, J. Guo, Z. Liu, and K.-Y. Lam, “Efficient Privacy-Preserving Retrieval
Augmented Generation with Distance-Preserving Encryption,” inProc. 3rd Int.
Conf. on Foundation and Large Language Models (FLLM), 2025, pp. 668–676.
doi: 10.1109/FLLM67465.2025.11391120.
2https://uca-fondation.fr/chaires/chaire-sur-lutilisation-fiable-et-d
e-confiance-des-grands-modeles-de-langage-llms/

Isolation Is Not Confidentiality: Geometric Access Control in RAG 19
14.R. Heet al., “Transform Before You Query: A Privacy-Preserving Approach for Vec-
tor Retrieval with Embedding Space Alignment,” arXiv preprint arXiv:2507.18518,
2025.
15.S. Kurilenko, “SHARD: Cell-Keyed Residual Splitting for Alignment-Resistant
Private Dense Retrieval,” arXiv preprint arXiv:2606.27976, 2026.
16.K. Liu, C. Giannella, and H. Kargupta, “An Attacker’s View of Distance Preserving
Maps for Privacy Preserving Data Mining,” inProc. 10th European Conf. on
Principles and Practice of Knowledge Discovery in Databases (PKDD), LNCS,
vol. 4213, Springer, 2006, pp. 297–308.
17.E. O. Turgay, T. B. Pedersen, Y. Saygın, E. Savaş, and A. Levi, “Disclosure Risks of
Distance Preserving Data Transformations,” inProc. 20th Int. Conf. on Scientific
and Statistical Database Management (SSDBM), LNCS, vol. 5069, Springer, 2008,
pp. 79–94.
18.A. Conneau, G. Lample, M. Ranzato, L. Denoyer, and H. Jégou, “Word Translation
Without Parallel Data,” inProc. Int. Conf. on Learning Representations (ICLR),
2018.
19.E. Grave, A. Joulin, and Q. Berthet, “Unsupervised Alignment of Embeddings with
Wasserstein Procrustes,” inProc. 22nd Int. Conf. on Artificial Intelligence and
Statistics (AISTATS), 2019.
20.W. Lin, K. Wang, Z. Zhang, and H. Chen, “Revisiting Security Risks of Asymmetric
Scalar Product Preserving Encryption and Its Variants,” inProc. 37th IEEE Int.
Conf. on Distributed Computing Systems (ICDCS), 2017.
21.B. Chen and T. Taipalus, “Metadata-Aware RAG for Enforcing Access Control and
Metadata-Based Filtering: Proof of Concept and Evaluation,” inProc. Int. Conf.
on Research in Adaptive and Convergent Systems (RACS), 2025, pp. 20:1–20:7.
doi: 10.1145/3769002.3769952.
22.I. Martinelli and M. A. Ponti, “Towards Secure Retrieval-Augmented Generation:
Preventing LLM Data Leaks with RBAC,” inProc. Brazilian Conf. on Intelligent
Systems (BRACIS), LNAI, vol. 16182, Springer, 2025, pp. 500–515. doi: 10.1007/978-
3-032-15993-9_34.
23.H. Zhonget al., “HONEYBEE: Efficient Role-Based Access Control for Vector
Databases,” inProc. ACM SIGMOD, 2026. arXiv:2505.01538.
24.M. Lorenzoet al., “ARBITER: LLM-Driven Access Control for Retrieval-Augmented
Generation,” arXiv preprint arXiv:2512.20535, 2025.
25.F. J. Arceo and V. P. Narsing, “Securing the Agent: Vendor-Neutral, Multitenant
Enterprise Retrieval and Tool Use,” arXiv preprint arXiv:2605.05287, 2026.
26.J. Bethencourt, A. Sahai, and B. Waters, “Ciphertext-Policy Attribute-Based
Encryption,” inProc. IEEE Symp. on Security and Privacy (S&P), 2007.
27.P.Zhou,Y.Feng,andZ.Yang,“Privacy-AwareRAG:SecureandIsolatedKnowledge
Retrieval,” arXiv preprint arXiv:2503.15548, 2025.
28.P. Zhou, Y. Feng, and Z. Yang, “Provably Secure Retrieval-Augmented Generation,”
arXiv preprint arXiv:2508.01084, 2025.
29.J. Zhuet al., “Compass: Encrypted Semantic Search with High Accuracy,” inProc.
USENIX Symp. on Operating Systems Design and Implementation (OSDI), 2025.
30.M. Naveed, S. Kamara, and C. V. Wright, “Inference Attacks on Property-Preserving
Encrypted Databases,” inProc. ACM SIGSAC Conf. on Computer and Communi-
cations Security (CCS), 2015.
31.E. M. Kornaropoulos, C. Papamanthou, and R. Tamassia, “Data Recovery on
Encrypted Databases with k-Nearest Neighbor Query Leakage,” inProc. IEEE
Symp. on Security and Privacy (S&P), 2019.

20 S. Nouyep et al.
32.C. Song and A. Raghunathan, “Information Leakage in Embedding Models,” in
Proc. ACM SIGSAC Conf. on Computer and Communications Security (CCS),
2020, pp. 377–390.
33.J. X. Morris, V. Kuleshov, V. Shmatikov, and A. M. Rush, “Text Embeddings
Reveal (Almost) As Much As Text,” inProc. Conf. Empirical Methods in Natural
Language Processing (EMNLP), 2023.
34.Y.-H. Huang, Y. Tsai, H. Hsiao, H.-Y. Lin, and S.-D. Lin, “Transferable Embedding
Inversion Attack: Uncovering Privacy Risks in Text Embeddings without Model
Queries,” inProc. 62nd Annual Meeting of the Association for Computational
Linguistics (ACL), 2024, pp. 4193–4205.
35.A.-E. Bodea, S. Meisenbacher, A. Klymenko, and F. Matthes, “SoK: Privacy Risks
and Mitigations in Retrieval-Augmented Generation Systems,” arXiv preprint
arXiv:2601.03979, 2026.