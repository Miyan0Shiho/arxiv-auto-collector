# Privacy Without Losing Place: A Paradigm for Private Retrieval in Spatial RAGs

**Authors**: Kennedy Edemacu, Mohammad Mahdi Shokri, Vinay M. Shashidhar, Jong Wook Kim

**Published**: 2026-05-06 21:33:43

**PDF URL**: [https://arxiv.org/pdf/2605.05459v1](https://arxiv.org/pdf/2605.05459v1)

## Abstract
This work introduces PAS -- Privacy Anchor Substitution, a structured mechanism for enabling user location privacy in spatial retrieval-augmented generation (RAG) systems. Unlike conventional differential privacy methods that directly perturb user locations, PAS represents location with relative anchor encoding consisting of an anchor, direction bin, and distance bin, allowing seamless integration with modern RAG pipelines. We evaluate PAS on a synthetic urban dataset and show that it achieves impressive coarse privacy guarantees, with approximately 370-400m adversarial location error, while retaining more than half of the baseline retrieval performance. Despite the slight drop in retrieval performance, the downstream generation quality under PAS remains comparatively robust, indicating that large language models can compensate for imperfect spatial retrieval. Furthermore, we provide empirical analysis showing that PAS exhibits non-monotonic privacy-utility relationship with respect to privacy parameters. We attribute this to geometric bias induced by anchor discretization, making it different from continuous noise mechanisms such as geo-indistinguishability. Our results show that structured spatial representations offer a practical approach to privacy in location based reasoning in RAG systems.

## Full Text


<!-- PDF content starts -->

Privacy Without Losing Place: A Paradigm for Private
Retrieval in Spatial RAGs
Kennedy Edemacu∗
The City University of New York
New York, NY 10314
kennedy.edemacu@csi.cuny.eduMohammad Mahdi Shokri†
The City University of New York
New York, NY 10016
mshokri@gradcenter.cuny.edu
Vinay M. Shashidhar
Northern Michigan University
Marquette, Mi 49855
vmadanbh@nmu.eduJong Wook Kim
Sangmyung University
Seoul, Korea
jkim@smu.ac.kr
Abstract
This work introduces PAS –PrivacyAnchorSubstitution, a structured mechanism
for enabling user location privacy in spatial retrieval-augmented generation (RAG)
systems. Unlike conventional differential privacy methods that directly perturb user
locations, PAS represents location with relative anchor encoding consisting of an
anchor, direction bin, and distance bin, allowing seamless integration with modern
RAG pipelines. We evaluate PAS on a synthetic urban dataset and show that
it achieves impressive coarse privacy guarantees, with approximately 370-400m
adversarial location error, while retaining more than half of the baseline retrieval
performance. Despite the slight drop in retrieval performance, the downstream
generation quality under PAS remains comparatively robust, indicating that large
language models can compensate for imperfect spatial retrieval. Furthermore, we
provide empirical analysis showing that PAS exhibits non-monotonic privacy-utility
relationship with respect to privacy parameters. We attribute this to geometric
bias induced by anchor discretization, making it different from continuous noise
mechanisms such as geo-indistinguishability. Our results show that structured
spatial representations offer a practical approach to privacy in location based
reasoning in RAG systems.
1 Introduction
The emergence of retrieval-augmented generation (RAG) has transformed the performance of large
language models (LLMs), especially for knowledge-intensive applications [ 1–3]. LLMs are known to
exhibit critical limitations in applications that require access to vast, specialized, or up-to-date sources
of factual information. This inherent struggle stems from their static parametric memory and their
inability to incorporate information beyond their context window [ 4,5]. RAG directly addresses this
limitation during inference by integrating relevant external knowledge dynamically retrieved from
external sources [ 6–8]. Thus, significantly improving performance on tasks such as domain-specific
question answering, fact-checking, and few-shot learning [9, 10].
Recent advances have extended this idea to geographical contexts, leading to the coinage of the term
Spatial RAG. In Spatial RAG, retrieval is conditioned on user location and spatial metadata [ 11–
18]. The importance of this approach is far reaching, spanning multiple location-based application
∗The City University of New York, College of Staten Island and Graduate Center
†The City University of New York, Graduate Center
Preprint.arXiv:2605.05459v1  [cs.CR]  6 May 2026

scenarios such as location-aware tourism, urban navigation and mobility guides, and disaster and
emergency response systems. For example, by leveraging its ability to interpret natural language
questions, a Spatial RAG system can accurately answer location-based queries such aswhat attraction
features are near me within a reasonable walking distance?andfind restaurants north of my current
location within 1 mile range.
However, current Spatial RAG studies [ 11–18] assume access to the exact user location as a query
feature. This creates a privacy bottleneck, as the system can expose fine-grained location coordinates
of users to the retrieval system. Consequently, this introduces significant risks, including leakage,
logging, and possible secondary use outside the scope of the intended application. Crucially, revealing
the exact location coordinates can unintentionally expose sensitive and personal information, such
as a user’s home address, workplace, daily routines, or religious/medical affiliations, thus creating
potential for surveillance, profiling, and targeted attacks. Established location privacy methods such
as geo-indistinguishability, spatial cloaking, k-anonymity, and grid-based tokenization [ 19,20] are
not tailored for RAGs and may suffer from a critical dichotomy. They either over-protect location
data, thereby harming utility, owing to introduction of too much noise or overly large cloaks, or
under-protect location data, resulting in the leakage of fine-grained geometrical information that
adversaries can triangulate. For example, to protect location privacy in the query,find restaurants
north of my current location within 1 mile rangeusing geo-indistinguishability, the direct injection of
noise to the location coordinates shifts locations globally and may result in relevant point of interests
falling outside the query radius.
A key insight for Spatial RAGs is that the retrieval process does not depend on exact location
coordinates. Rather, it often relies on establishing semantic proximity, such asnear a hospital
orwithin 1 km south of the university. This finding opens the door for the development of task-
aware privacy methods. Instead of perturbing the raw location coordinates, the system can release
a customized spatial representation tailored for the retrieval task, thereby reducing the privacy risk.
This work attempts to achieve this goal. To our knowledge, this effort represents the first attempt to
protect location privacy in Spatial RAGs.
We propose a novel approach,PublicAnchorSubstitution (PAS), which is architected upon the
aforementioned principle. PAS avoids the release of exact location coordinates by implementing a
three-stage transformation.1)the user location is probabilistically mapped to a public anchor point
(such as a landmark or transit hub).2)the spatial relationship between the true location and this
anchor is encoded into coarse bins of distance and direction. And3)the system generates a query
for the retrieval mechanism that exclusively leverages the resulting triplets (anchor, distance bin,
direction bin), thereby successfully protecting the location information in the input as shown in Figure
1. Thus, in summary, PAS provides the following benefits:
•Anchors the privacy noise in semantically meaningful locations, successfully preventing
unrealistic and unhelpful perturbed points.
•Produces features that are sufficient for effective retrieval and are aligned with the corpus
indexing strategy, thus minimizing utility loss.
•Enables formal privacy guarantees via the exponential mechanism over public anchors,
ensuring a coarse provable secure system.
2 Methodology
We propose PAS, a location-obfuscation mechanism designed specifically for Spatial RAG tasks as
shown in Figure 1. Unlike traditional obfuscation techniques that release noisy location information,
PAS maps a user’s location in a semantically meaningful manner, referred to asanchor triplets, which
align with the corpus index while providing coarsely provable differential privacy guarantees.
2.1 Problem Formalization
Spatial and Semantic Definitions: Given a user query, PAS first decomposes the query into spatial
(qsp) and semantic ( qsem) components. For example, for the queryfind restaurants north of my
current location within 1 mile range,restaurantis a semantic component, andcurrent location,north
and1 mileform the spatial component. It is only the spatial part that gets privatized. PAS transforms
2

qsp ”north, 1 mile , current location ”
“Find restaurants  north  within  1 mile of my location.”
Query Decomposition
Semantic Query
qsem ”restaurants ”Spatial Query
Semantic Component
qsem
Spatial Score  𝑆𝑠𝑝(t)
Semantic Score  𝑆𝑠𝑒𝑚(t)
Final Score 𝑆t=λ 𝑆𝑠𝑒𝑚 +1−𝜆𝑆𝑠𝑝Hybrid RetrievalSpatial Component & PAS Mapping
✓Anchor Sampling   𝑎∗
✓Direction Binning   𝐵𝜃∗
✓Distance Binning   𝐵r∗
𝒛=(𝒂∗,𝑩𝜽∗,𝑩𝒓∗)
Relevant POIs and Documents
RAG Grounding & 
Response Generation
“Here are restaurant 
north within 1 mile …”Figure 1: Illustration of PAS-based RAG framework. The main phases of the pipeline are: (1)
query decomposition, where a user query is decomposed into semantic and spatial components. The
spatial component undergoes a privacy mapping process, in which the true location is transformed
into an obfuscated triplet (anchor, direction bin, distance bin). (2) the privatized spatial token is
then combined with the semantic component to for a hybrid retrieval. (3) the retrieved evidence is
presented to RAG grounding model to the generate the final response.
the spatial part into (anchor, distance bin, direction bin) triplets. Formally, we let Urepresent a user’s
location space and define the privacy mapping components of PAS as follows.
•Public Anchors ( A): A finite set A={a 1,···, a m} ⊂ U of publicly known landmarks
such as campus buildings, transit hubs, and civic hubs.
•Directional Partition ( Bθ): A finite partition of directional bearings, such as the 8 campus
sectors.
•Distance Partition ( Br): A finite partition of non-negative distances into discrete rings (e.g.,
[0,500),[500,2000),···meters).
Corpus Indexing: We assume each document t∈ D in a Spatial RAG corpus Dis indexed by
Φ(t)∈A×B θ×Br. In other words, each document in the corpus is tagged with the nearest or most
relevant public anchor and coarse direction and distance bin tags relative to that anchor. This indexing
strategy allows for pruning of relevant point of interests and ensures that retrieval is performed over
the same discrete space as our obfuscated queries.
2.2 PAS Privacy Mapping Mechanism
Given a true location u∈ U , a privacy parameter ε >0 , and a scaling factor s, for every location-
based query, instead of exposing u, thePAS ε,smechanism releases a sanitized query token z=
(a∗, B∗
θ, B∗
r)∈A×B θ×Brto the RAG for retrieval. The mechanism comprises two stages:
(1) Probabilistic Anchor Selection: We take advantage of the Exponential Mechanism [ 21] to prob-
abilistically select a representative public anchor a∗∈A. The selection probability is proportional to
the distance between the true location and the candidate anchor location, as shown in Eq. 1.
Pr(a∗|u)∝exp
−εd(u, a∗)
s
,∀a∗∈A(1)
3

This ensures that the true location is likely to be mapped to a nearby public anchor but maintains a
strictly defined probability of being mapped to any public anchor in the set.
(2) Deterministic Post-Processing: Once a∗is identified and sampled, we generate the coarse
semantics: distance and direction bins relative to the true locationu.
1.Directional Binning: Let θ(a∗, u)be the bearing from the true location uto the sampled
public anchora∗. We set the directional binB∗
θ=dir_bin(θ(a∗, u))∈B θ.
2.Distance Binning: Similarly, let r(u, a∗) =d(u, a∗)be the distance between the true
location uand the sampled public anchor a∗. We set the distance bin as B∗
r=
dist_bin(r(u, a∗))∈B r.
The triplet z= (a∗, B∗
θ, B∗
r)is what the RAG backend will gain access to, and the raw location u
never leaves the user’s site. Most importantly, the token z= (a∗, B∗
θ, B∗
r)defines an uncertainty
region, i.e., the user lies in the set:
U(z) ={x:θ(a∗, x)∈B∗
θ, d(a∗, x)∈B∗
r}(2)
For example, supposeB∗
θ=north andB∗
r=0-1mi, thenU(z)is:
U(z) = (north wedge)∩(0−1mile ring).
This allows the retriever to treatuas unknown, but is constrained toU(z).
2.3 Point of Interest Retrieval
To retrieve a point of interest (POI), t, we assume a distribution, π(x|z) over possible user locations
(latent user locations) inside the uncertainty region. Then:
R(z) =top-nt∈C(z) Ssp(t|z)(3)
where:
Ssp(t|z) = Pr
x∼π(.|z)[d(x, t)≤R∧θ(x, t)∈B q],
C(z) is a candidate set, RandBqare the distance and direction constraints specified in the query.
This implies that we retrieve POIs with high probability of meeting the user-relative constraints.
In implementation, integration over all x∈U(z) is challenging. Instead, we approximate with
sampling using the Monte Carlo approximation [ 22]. We sample Klatent user points, x1,···, x K∈
U(z), then we estimate:
Ssp(t|z) =1
KKX
k=11[d(x k, t)≤R]·1[θ(x k, t)∈B q](4)
For textual relevance between the user query and the retrieved documents, we compute the semantic
score for each retrieved item t. We define the semantic score, Ssem(t)as a cosine similarity between
qsemandt:
Ssem(t) =cosine(q sem, t)(5)
This formulation follows the standard dense retrieval in RAG pipelines. We then perform a hybrid
ranking for each POI,tas:
S(t) =λS sem(t) + (1−λ)S sp(t|z)(6)
where 0≤λ≤1 is a weighting parameter. Any item twhose final score S(t) falls within the top- k,
is selected for retrieval.
To improve retrieval efficiency, we first employ the distance tags to generate a superset of plausible
POIs before performing the above ranking. A good candidate set is defined as:
C(z) ={t:d(a∗, t)≤R+r max}(7)
where rmax is the maximum radius in Br. This follows the principle of triangle inequality [ 23]. Since
d(u, a∗)≤r max, any POI within the distance Rfrom the user lies within the distance R+r max
from the anchor. Such a pruning can also be done with direction bins.
4

2.4 Privacy Analysis
We measure the privacy of PAS using the differential privacy framework of geo-indistinguishability
(geo-DP).
Definition 1 (ε-geo-DP over Anchors) A mechanism M:U →A×B θ×Brsatisfies ε-geo-DP
with scalesif for allu, u′∈ Uand all measurableO⊆supp(M)[24]:
Pr(M(u)∈O)
Pr(M(u′)∈O)≤exp 
εd(u, u′)
s
(8)
Lemma 1The mechanism PAS ε,ssatisfiesε-geo-DP.
Proof Sketch: The sampling of a∗follows the standard exponential mechanism where the score
function q(u, a∗) =−d(u, a∗)/shas the metric sensitivity d(u, u′)/s. By the standard exponential
bound and the closure of differential privacy under post-processing, the subsequent deterministic
computations ofB∗
θandB∗
rpreserve this bound.
3 Experiment
3.1 Dataset
To rigorously evaluate PAS, we construct a synthetic yet realistic Spatial RAG benchmark dataset
based on New York City. The dataset is designed to simulate location-aware queries under privacy
constraints. The dataset includes a diverse set of urban point-of-interests (POIs) such as restaurants,
hotels, hospitals, parks, etc, reflecting diversity across the five boroughs of New York City. The
dataset has three main components: anchors, chunks (POIs), and queries.Anchorscontains a set
of 30 spatial anchors distributed across the city. Each anchor is defined by a geographic coordinate
and semantic identifiers such as neighborhood, name, etc. The anchors serve as a basis for spatial
abstraction in PAS through probabilistic spatial representation.Chunksforms the corpus, which
contains 1010 POIs. Each POI represents a retrievable document. Each POI includes semantic
attributes (such as name, category, and description), a geographic coordinate, and multi-anchor
tags. Multi-anchor tagging allows each chunk to be pre-indexed with spatial relations relative to
at least two anchors. Each anchor tag specifies a direction bin and a distance bin relative to the
anchor. This multi-anchor indexing enables the probabilistic spatial matching required to evaluate
PAS, and it is vital for candidate pruning.Queriesforms a set of 423 evaluation queries of varying
complexity. Each query is equipped with a true location (for baseline evaluations only), spatial
constraints (distance and direction bins), ground-truth POIs, and a semantic category. The dataset
supports two retrieval modes: (i) the deterministic baseline retrieval, where the true user location is
used, and (ii) PAS retrieval, where probabilistic tokens are generated for spatial representation. We
used ChatGPT [ 25] during the dataset construction, specifically to assist in generating components of
synthetic queries and metadata. All generated content was reviewed and validated.
3.2 Metrics
We evaluate across three dimensions, with each dimension having its own metric(s) as follows.
Retrieval Metrics: To assess the retrieval ability, we employ two metrics:Recall@kandnDCG@k.
Recall@k measures the fraction of relevant POIs that appear in the top-k retrieved results. Meanwhile,
nDCG@k [ 26] measures how well the retrieval identifies the relevant POIs and how high they are
ranked amongst the retrieved items.
Generation Metrics: We assess the generation quality using two metrics:F1 scoreandCitation
Correctness. F1 score measures the overlap between the generated answer and the ground truth
answer. We employ it to reflect if degradation in retrieval due to privacy integration propagates to
the generation phase. Citation correctness (Citation) evaluates whether the generated answers are
properly grounded in the retrieved context.
Security Metrics: We introduce a security metric,Expected Localization Error (ALE), which
measures the expected distance between the true user location uand an adversary’s location estimate
ˆugiven the PAS tokenz:
ALE(z) =E[d(u,ˆu)|z](9)
5

Retriever Model Setting Recall@k nDCG@k F1 Citation ALE (m)
bge-baseMeta-Llama-3.3-70BBaseline0.8456 0.77780.24280.80170.00
PAS (ϵ= 1.0) 0.5260 0.4623 0.2577 0.5667 367.57
gemma-3-12b-itBaseline0.8456 0.7778 0.46860.7380 0.00
PAS (ϵ= 1.0) 0.5190 0.4533 0.4168 0.5310389.65
gpt-oss-120bBaseline0.8456 0.77780.3977 0.7840 0.00
PAS (ϵ= 1.0) 0.4508 0.4270 0.3625 0.5567 370.61
MiniLMMeta-Llama-3.3-70BBaseline0.87190.8089 0.2726 0.7451 0.00
PAS (ϵ= 1.0) 0.5807 0.5163 0.2673 0.6559 372.98
gemma-3-12b-itBaseline0.87190.80890.47790.6690 0.00
PAS (ϵ= 1.0) 0.5504 0.4896 0.4300 0.5112397.88
gpt-oss-120bBaseline0.8719 0.80910.35660.75890.00
PAS (ϵ= 1.0) 0.5648 0.5322 0.3657 0.6012 369.12
Table 1: Comparison of PAS and baseline (non-private) performance across retrievers. PAS introduces
strong privacy (ALE≈370–400m) while reducing retrieval performance. Best values arebold.
where ˆuis defined as the centroid of the uncertainty region U(z) , and it is computed using the Monte
Carlo sampling approximation as:
ˆu=1
KKX
k=1xk, x k∼U(z)(10)
We use ALE to capture the expected error of an adversary attempting to infer the user’s true location
from the privatized spatial token.
3.3 RAG Settings
We evaluate our proposed PAS within the standard RAG pipeline, where a query is first decomposed
into a semantic and a spatial component. The decomposed components are then used to retrieve
relevant items from the grounded corpus before passing them to the generation model to generate the
final answer. We employ the hybrid scoring approach in Eq. 6 during the retrieval. Unless stated
otherwise, we use aλvalue of 0.8, giving priority to semantic relevance while incorporating spatial
constraints.
For semantic retrieval, we perform dense vector retrieval with two widely used models: MiniLM
(sentence-transformers/all-MiniLM-L6-v2) [27] and bge-base (bge-base-en) [28]. Spatial relevance
is computed using the true location for the baseline (non-private). However, for PAS, the true location
is replaced with the triplets of a sampled anchor, and discretized direction and distance bins relative
to the sampled anchor. The triplets define an uncertainity region for the user location. Thus, spatial
relevance is estimated over this region instead of a single point. For each query, we generate a set of
latent user locations from the uncertainity region, and employ Monte Carlo sampling over the latent
locations to estimate the spatial relevance for corpus items. The selected top-k items are based on the
hybrid score comprising the semantic and spatial relevance scores. We use a top-5 in our experiments.
To improve efficiency, we reduce the search space by first pruning the documents using anchor-based
constraints. For a DP sampled anchor, we retain documents that lie within an expanded radius relative
to the anchor. The retrieved documents (POIs) are presented to generation models to output the final
answer. We evaluate with three generation models: Meta-Llama-3.3-70B [ 29], gemma-3-12b-it [ 30],
and gpt-oss-120b [ 31]. For other settings, we computed distances using the Haversin formula [ 32]
and we discretize bearings into 8 direction bins. Further details are provide in Appendix C.
4 Results
We evaluate the impact of PAS framework on retrieval and generation quality across two retrievers:
bge-base and MiniLM, and three generation models: Meta-Llama-3.3-70B, gemma-3-12b-it and
gpt-oss-120b. We summarize the results in comparison to the baseline (non-private) in Table 1, with
additional graphical comparisons in Appendix A. In general, across all retriever-generation model
configurations, PAS achieves strong privacy guarantees, with ALE consistently ranging between 367
6

MiniLM
Meta-Llama-3.3-70B-InstructMiniLM
gemma-3-12b-itbge-base
Meta-Llama-3.3-70B-Instructbge-base
gemma-3-12b-itbge-base
gpt-oss-120b0.00.20.40.6Recall RetentionRelative Recall Retention under PAS(a) Recall@k Retention
MiniLM
Meta-Llama-3.3-70B-InstructMiniLM
gemma-3-12b-itbge-base
Meta-Llama-3.3-70B-Instructbge-base
gemma-3-12b-itbge-base
gpt-oss-120b0.00.20.40.6nDCG RetentionRelative nDCG Retention under PAS (b) nDCG@k Retention
bge-base
Meta-Llama-3.3-70B-InstructMiniLM
Meta-Llama-3.3-70B-Instructbge-base
gpt-oss-120bMiniLM
gemma-3-12b-itbge-base
gemma-3-12b-it0.000.250.500.751.00F1 Score RetentionRelative F1 Score Retention under PAS
(c) F1 Score Retention
MiniLM
gemma-3-12b-itMiniLM
Meta-Llama-3.3-70B-Instructbge-base
gemma-3-12b-itbge-base
gpt-oss-120bbge-base
Meta-Llama-3.3-70B-Instruct0.00.20.40.60.8Citation RetentionRelative Citation Retention under PAS (d) Citation Correctness Retention
Figure 2: Performance Retention under PAS
370 375 380 385 390 395 400
ALE (meters)0.460.480.500.520.540.56Recall@k
Privacy Utility Trade-off: ALE vs Recall@k
MiniLM / Meta-Llama-3.3-70B-Instruct
MiniLM / gemma-3-12b-it
MiniLM / gpt-oss-120b
bge-base / Meta-Llama-3.3-70B-Instruct
bge-base / gemma-3-12b-it
bge-base / gpt-oss-120b
(a) Privacy vs Recall
370 375 380 385 390 395 400
ALE (meters)0.420.440.460.480.500.52nDCG@k
Privacy Utility Trade-off: ALE vs nDCG@k
MiniLM / Meta-Llama-3.3-70B-Instruct
MiniLM / gemma-3-12b-it
MiniLM / gpt-oss-120b
bge-base / Meta-Llama-3.3-70B-Instruct
bge-base / gemma-3-12b-it
bge-base / gpt-oss-120b (b) Privacy vs nDCG
Figure 3: Privacy-Utility Trade-off Curves for PAS
and 398 meters, corresponding to neighborhood-level obfuscation. However, this coarse privacy
guaranty comes at a measurable cost in retrieval performance. We further illustrate this through the
privacy-utility trade-off curves presented in Figure 3.
4.1 Retrieval Performance Analysis
We observe that increasing privacy (high ALE) results in decrease in retrieval performance. Specif-
ically, from Table 1 we observe that Recall@k decreases from ∼0.85-0.87 to ∼0.45-0.58, and
nDCG@k drops from ∼0.78-0.81 to ∼0.43-0.53. Furthermore, we observe that the drop in retrieval
performance is worse for bge-base compared to MiniLM, suggesting that MiniLM is more robust to
spatial perturbations, possibly due to stronger semantic generalization. However, this degradation in
retrieval performance is rather gradual than catastrophic, indicating that PAS preserves meaningful
portion of spatial signals despite location obfuscation. To this effect, we report the retention of the
retrieval performance in Figures 2a and 2b. Recall@k and nDCG@k retentions are measured as:
PAS Recall@k
Baseline Recall@kandPAS Recall@k
Baseline Recall@k, respectively. In Figure 2a, we observe that in most cases, PAS
retains at least 60% of the baseline Recall@k. Meanwhile in Figure 2b, we observe that PAS retains
at least 55% of the baseline nDCG@k. This shows that PAS maintains a substantial fraction of
retrieval capabilities under privacy constraints. Despite the drop in retrieval, the fairly high nDCG@k
retention indicates that PAS preserves the relative ordering of the relevant documents even when
some are missing. Primarily this is because PAS affects coverage rather than ranking fidelity.
7

1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
Privacy Parameter 
0.420.440.460.480.500.520.540.56Recall@k
Recall Sensitivity to  and 
MiniLM/Meta-Llama-3.3-70B-Instruct, =0.7
MiniLM/Meta-Llama-3.3-70B-Instruct, =0.8
MiniLM/gemma-3-12b-it, =0.7
MiniLM/gemma-3-12b-it, =0.8
MiniLM/gpt-oss-120b, =0.7
MiniLM/gpt-oss-120b, =0.8
bge-base/Meta-Llama-3.3-70B-Instruct, =0.7
bge-base/Meta-Llama-3.3-70B-Instruct, =0.8
bge-base/gemma-3-12b-it, =0.7
bge-base/gemma-3-12b-it, =0.8
bge-base/gpt-oss-120b, =0.7
bge-base/gpt-oss-120b, =0.8
(a) Recall Sensitivity toεandλ
1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0
Privacy Parameter 
365370375380385390395400ALE (m)
ALE Sensitivity to  and 
MiniLM/Meta-Llama-3.3-70B-Instruct, =0.7
MiniLM/Meta-Llama-3.3-70B-Instruct, =0.8
MiniLM/gemma-3-12b-it, =0.7
MiniLM/gemma-3-12b-it, =0.8
MiniLM/gpt-oss-120b, =0.7
MiniLM/gpt-oss-120b, =0.8
bge-base/Meta-Llama-3.3-70B-Instruct, =0.7
bge-base/Meta-Llama-3.3-70B-Instruct, =0.8
bge-base/gemma-3-12b-it, =0.7
bge-base/gemma-3-12b-it, =0.8
bge-base/gpt-oss-120b, =0.7
bge-base/gpt-oss-120b, =0.8
 (b) ALE Sensitivity toεandλ
Figure 4: Privacy and Utility Sensitivity toεandλ
4.2 Generation Performance Analysis
Despite the decrease in retrieval performance, the generation quality remains fairly stable as shown
in Table 1, and Figures 2c and 2d. F1 score exhibits moderate drop and even improves in some cases.
For example, for the bge-Llama configuration, F1 slightly increases from 0.2428 to 0.2577. While
for MiniLM-Llama configuration, F1 slightly drops from 0.2726 to 0.2673. This suggests that PAS
noise may likely act as a regularizer, minimizing overfitting to precise spatial cues, thus encouraging
reliance on semantic signals. Citation correctness exhibits degradation but remains largely at usable
levels. Overall, the citation correctness drops from ∼0.74-0.80 in baseline to ∼0.51-0.66 in PAS. In
general, PAS highly retains the generational performances of the baseline as shown in Figures 2c
and 2d. This together with the retrieval vs generation results presented in Appendix B, indicate that
models can still produce quality answers even with imperfect retrieval. However, the overall final
answer quality is dependent on the generation model used. Meta-Llama-3.3-70B achieves the most
stable performance under PAS, while gemma-3-12b-it achieves the highest baseline F1 score but
experiences the greatest degradation. This suggests that larger or better instruction-tuned models can
compensate for the noisy retrieved inputs.
4.3 Privacy and Utility Sensivity toεandλAnalysis
We also report privacy and utility sensitivity to varying privacy parameters, as shown in Figure 4.
Interestingly, we observe that an increase in εdoes not monotonically result in improved utility.
Instead, we observe that Recall@k decreases or remains flat as εincreases and ALE increases
asεincreases, which contradicts the typical differential privacy intuition. Unlike the traditional
geo-indistinguishability mechanism, PAS demonstrates a non-monotonic relationship between ε
and utility. This is mainly due to anchor discretization and spatial encoding. Although larger ε
reduces noise in continuous differential privacy mechanisms, PAS operates under discrete anchor-
relative representations. Thus, ALE is governed not only by the anchor distance, but also by the
discretization error introduced by the direction and distance bins. In this case, the nearest anchor does
not necessarily reduce the reconstruction error, thus increasing εmay concentrate probability mass
on anchors whose induced uncertainty region is geometrically biased (suboptimal anchors). This
can increase ALE and reduce Recall@k, resulting in non-monotonic privacy-utility behavior in our
experiments.
In general, the results show a clear trade-off. PAS sacrifices utility to achieve impressive coarse
location privacy. However, more than half of the retrieval signal and ranking quality are maintained,
while the generation performance is stable. These findings suggest that PAS enables practical privacy-
preserving Spatial RAG, where strong privacy guarantees can be achieved without horrendous
degradation in downstream tasks.
5 Related Work
Retrieval-Augmented Generation (RAG): Due to the parametric nature of LLMs, RAG has emerged
as the dominant paradigm for grounding LLMs on external, up-to-date, and factual knowledge sources.
8

Early studies [ 1–3] have established improved LLM performance in terms of factual accuracy and
interpretability under the paradigm. Subsequent efforts focus on improving retrieval quality through
methods such as dense representations [ 33] and late interaction model [ 34]. Furthermore, [ 35,36]
integrate retrieval into pre-training and generation pipelines. The RAG frameworks are increasingly
being adopted in diverse application domains such as healthcare [ 37], agriculture [ 38], and finance
[39]. Importantly, due to the consistent increase in the ability of LLMs to reason over various types
of data, including spatial data, RAGs are being integrated into georgraphical application domains,
leading to the termSpatial RAG. A growing body of work [ 11–18] explores various strategies
to improve spatial RAG in various geographical application domains. Despite these advances,
these works assume access to the users’ true location, a vulnerability that presents significant risks
and exposes users to location-based inference attacks. In the literature, several works have been
proposed to protect user locations. Geo-indistinguishability [ 24] extends the differential privacy
[21,40,41] to location information by adding noise to geographical coordinates. Other location
protection approaches are spatial cloaking [42, 43] and k-anonymity [19, 20, 44] that aim to protect
user locations by expanding spatial regions. A more recent work [ 45] protects sequences of user
movements instead of single points. However, these methods are not tailored for RAGs and are likely
to degrade utility in the downstream tasks, due to precise spatial constraints associated with Spatial
RAGs.
Privacy in Retrieval and LLM Systems: Recent research advances [ 46,47] have begun to address
privacy concerns in LLM-based systems. Zhang et al. [ 48] proposes a framework that operates
at the entity and relation granularity level using knowledge graph representations to protect the
privacy of the retrieved documents during the generation phase. [ 49] proposes using synthetic data
as an alternative to sensitive KB data in RAGs. Duan et al. [ 50] uses an ensembling approach by
aggregating different versions of a prompted model to hide data. A significant portion of these studies
have specifically focused on protecting sensitive user inputs. Carey et al. [ 51] privatizes tabular data
with differential privacy before serializing them into a prompt. Liu et al. [ 52] employs differential
privacy for token- and sentence-level sanitization on sensitive and labeled data in prompts. Ngong et
al. [53] uses a locally deployable framework between users and LLM to identify and reformulate
sensitive information in the user prompt before being sent to the LLM. However, all of these studies
do not address spatial privacy.
Our proposed PAS framework bridges previously disconnected areas. Unlike previous approaches,
the PAS framework introduces a structured probabilistic representation that enables retrieval under
uncertainty. It integrates differential privacy with Spatial RAG, allowing retrieval without access
to the true user location coordinates. By combining probabilistic anchor-based representations and
hybrid retrieval, PAS enables privacy-preserving spatial reasoning while maintaining compatibility
with Spatial RAG pipelines. To our knowledge, this is the first work to address privacy in Spatial
RAGs.
6 Conclusion
This work presented PAS, a structured mechanism to preserve location privacy in spatial retrieval
within RAG systems. Unlike the conventional differential privacy based approaches that inject
continuous noise, PAS encodes user location as a discrete anchor-relative representation comprising
an anchor, and direction and distance bins. This design enables seamless integration into modern RAG
pipelines while providing coarsely guaranteed location privacy. Through a comprehensive evaluation
on a synthetic urban dataset, we showed that PAS achieves impressive location obfuscation while
retaining over half of the baseline retrieval performance and a stable generation quality. A key insight
from our analysis is that PAS exhibits a non-monotonic privacy-utility relationship with respect to the
varying privacy parameter. Empirically, we showed that this behavior is due to geometrical biases
induced by anchor discretization and binning, instead of purely stochastic noise. In general, our
results demonstrate that PAS provides ingredients for a practical spatial privacy in RAG systems.
Limitations: We acknowledge that this work has few limitations that we shall aim to address in
the future. (i) PAS relies on anchor-based discretizations, which introduce geometric bias that can
move the inferred region away from the user’s true location in a structured manner. This can lead
to the non-monotonic behavior discussed earlier, and it may affect robustness in sparse or irregular
anchor regions. (ii) the use of Monte Carlo sampling for estimating the spatial relevance introduces
computational overhead and approximation variance. Although suitable for lab experiments, it may
9

not scale well for larger datasets and real-time environments. (iii) we evaluate PAS on synthetic urban
data, even though it is controlled and diverse, it may not fully reflect the real-world distributions and
noisy metadata. Thus, further validations are required to generalize the results for production-scale
services.
References
[1]P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih,
T. Rocktäschelet al., “Retrieval-augmented generation for knowledge-intensive NLP tasks,”Advances in
Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
[2]N. Kandpal, H. Deng, A. Roberts, E. Wallace, and C. Raffel, “Large language models struggle to learn
long-tail knowledge,” inInternational conference on machine learning. PMLR, 2023, pp. 15 696–15 707.
[3]Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, H. Wang, and H. Wang, “Retrieval-augmented
generation for large language models: A survey,”arXiv preprint arXiv:2312.10997, vol. 2, no. 1, 2023.
[4]A. Roberts, C. Raffel, and N. Shazeer, “How much knowledge can you pack into the parameters of a
language model?”arXiv preprint arXiv:2002.08910, 2020.
[5]F. Petroni, T. Rocktäschel, P. Lewis, A. Bakhtin, Y . Wu, A. H. Miller, and S. Riedel, “Language models as
knowledge bases?”arXiv preprint arXiv:1909.01066, 2019.
[6]K. Zhu, Y . Luo, D. Xu, Y . Yan, Z. Liu, S. Yu, R. Wang, S. Wang, Y . Li, N. Zhanget al., “Rageval: Scenario
specific rag evaluation dataset generation framework,”arXiv preprint arXiv:2408.01262, 2024.
[7]G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open domain question
answering,”arXiv preprint arXiv:2007.01282, 2020.
[8]S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. B. Van Den Driessche, J.-B.
Lespiau, B. Damoc, A. Clarket al., “Improving language models by retrieving from trillions of tokens,” in
International conference on machine learning. PMLR, 2022, pp. 2206–2240.
[9]O. Ram, Y . Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-Brown, and Y . Shoham, “In-context
retrieval-augmented language models,”Transactions of the Association for Computational Linguistics,
vol. 11, pp. 1316–1331, 2023.
[10] W. Shi, S. Min, M. Yasunaga, M. Seo, R. James, M. Lewis, L. Zettlemoyer, and W.-t. Yih, “Replug:
Retrieval-augmented black-box language models,”arXiv preprint arXiv:2301.12652, 2023.
[11] D. Yu, R. Bao, G. Mai, and L. Zhao, “Spatial-rag: Spatial retrieval augmented generation for real-world
spatial reasoning questions,”arXiv preprint arXiv:2502.18470, 2025.
[12] B. Martins, P. Gramackiet al., “A vision for geo-temporal deep research systems: Towards comprehensive,
transparent, and reproducible geo-temporal information synthesis,”arXiv preprint arXiv:2506.14345, 2025.
[13] T. Jung and I. Joe, “An intelligent docent system with a small large language model (sllm) based on
retrieval-augmented generation (rag),”Applied Sciences, vol. 15, no. 17, p. 9398, 2025.
[14] H. Ni, F. Liu, X. Ma, L. Su, S. Wang, D. Yin, H. Xiong, and H. Liu, “Tp-rag: Benchmarking retrieval-
augmented large language model agents for spatiotemporal-aware travel planning,”arXiv preprint
arXiv:2504.08694, 2025.
[15] M. Zajac, C. Kulawiak, S. Li, C. Erickson, N. Hubbell, and J. Gong, “Unifying flood-risk communication:
Empowering community leaders through ai-enhanced, contextualized storytelling,”Hydrology, vol. 12,
no. 8, p. 204, 2025.
[16] Z. Liu, S. Wu, and L. Hurni, “Geospatial question answering on historical maps using spatio-temporal
knowledge graphs and large language models,”arXiv preprint arXiv:2508.21491, 2025.
[17] N. R. Schneider, N. Ramachandran, K. O’Sullivan, and H. Samet, “Distrag: Towards distance-based spatial
reasoning in llms,”arXiv preprint arXiv:2506.03424, 2025.
[18] W. Ruan, X. Dang, Z. Zhou, S. Lyu, and Y . Liang, “A retrieval augmented spatio-temporal framework for
traffic prediction,”arXiv preprint arXiv:2508.16623, 2025.
[19] J. W. Kim, K. Edemacu, and B. Jang, “Privacy-preserving mechanisms for location privacy in mobile
crowdsensing: A survey,”Journal of Network and Computer Applications, vol. 200, p. 103315, 2022.
10

[20] J. W. Kim, K. Edemacu, J. S. Kim, Y . D. Chung, and B. Jang, “A survey of differential privacy-based
techniques and their applicability to location-based services,”Computers & Security, vol. 111, p. 102464,
2021.
[21] F. McSherry and K. Talwar, “Mechanism design via differential privacy,” in48th Annual IEEE Symposium
on Foundations of Computer Science (FOCS’07). IEEE, 2007, pp. 94–103.
[22] N. Metropolis and S. Ulam, “The monte carlo method,”Journal of the American statistical association,
vol. 44, no. 247, pp. 335–341, 1949.
[23] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein,Introduction to algorithms. MIT press, 2022.
[24] M. E. Andrés, N. E. Bordenabe, K. Chatzikokolakis, and C. Palamidessi, “Geo-indistinguishability:
Differential privacy for location-based systems,” inProceedings of the 2013 ACM SIGSAC conference on
Computer & communications security, 2013, pp. 901–914.
[25] OpenAI, “Chatgpt,” https://chat.openai.com, 2026, accessed: 2026-03-28.
[26] Y . Wang, L. Wang, Y . Li, D. He, and T.-Y . Liu, “A theoretical analysis of ndcg type ranking measures,” in
Conference on learning theory. PMLR, 2013, pp. 25–54.
[27] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association
for Computational Linguistics, 11 2019. [Online]. Available: https://arxiv.org/abs/1908.10084
[28] S. Xiao, Z. Liu, P. Zhang, N. Muennighoff, D. Lian, and J.-Y . Nie, “C-pack: Packed resources for general
chinese embeddings,” 2023, model: BAAI/bge-base-en, available at https://huggingface.co/BAAI/bge-
base-en.
[29] M. L. Team, “The llama 3 herd of models,” 2024, technical Report. [Online]. Available:
https://ai.meta.com/research/publications/the-llama-3-herd-of-models/
[30] G. Team, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrinet al., “Gemma 3 technical
report,” 2025. [Online]. Available: https://arxiv.org/abs/2503.19786
[31] OpenAI, :, S. Agarwal, L. Ahmad, J. Ai, S. Altman, A. Applebaumet al., “gpt-oss-120b & gpt-oss-20b
model card,” 2025. [Online]. Available: https://arxiv.org/abs/2508.10925
[32] J. Inman,Navigation and Nautical Astronomy: For the Use of British Seamen. Portsea, England: W.
Woodward, 1835.
[33] V . Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, “Dense passage
retrieval for open-domain question answering,” inProceedings of the 2020 conference on empirical methods
in natural language processing (EMNLP), 2020, pp. 6769–6781.
[34] O. Khattab and M. Zaharia, “Colbert: Efficient and effective passage search via contextualized late
interaction over bert,” inProceedings of the 43rd International ACM SIGIR conference on research and
development in Information Retrieval, 2020, pp. 39–48.
[35] K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval augmented language model pre-training,”
inInternational conference on machine learning. PMLR, 2020, pp. 3929–3938.
[36] G. Izacard and E. Grave, “Leveraging passage retrieval with generative models for open domain ques-
tion answering,” inProceedings of the 16th conference of the european chapter of the association for
computational linguistics: main volume, 2021, pp. 874–880.
[37] L. M. Amugongo, P. Mascheroni, S. Brooks, S. Doering, and J. Seidel, “Retrieval augmented generation
for large language models in healthcare: A systematic review,”PLOS Digital Health, vol. 4, no. 6, p.
e0000877, 2025.
[38] S. S. Kumar, A. K. M. A. Khan, I. A. Banday, M. Gada, and V . V . Shanbhag, “Overcoming llm challenges
using rag-driven precision in coffee leaf disease remediation,” in2024 International Conference on
Emerging Technologies in Computer Science for Interdisciplinary Applications (ICETCS). IEEE, 2024,
pp. 1–6.
[39] S. Setty, H. Thakkar, A. Lee, E. Chung, and N. Vidra, “Improving retrieval for rag based question answering
models on financial documents,”arXiv preprint arXiv:2404.07221, 2024.
[40] C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating noise to sensitivity in private data analysis,”
inTheory of cryptography conference. Springer, 2006, pp. 265–284.
11

[41] C. Dwork and A. Roth, “The algorithmic foundations of differential privacy,”Foundations and trends® in
theoretical computer science, vol. 9, no. 3-4, pp. 211–487, 2014.
[42] C.-Y . Chow, M. F. Mokbel, and X. Liu, “Spatial cloaking for anonymous location-based services in mobile
peer-to-peer environments,”GeoInformatica, vol. 15, no. 2, pp. 351–380, 2011.
[43] M. Gruteser and D. Grunwald, “Anonymous usage of location-based services through spatial and temporal
cloaking,” inProceedings of the 1st international conference on Mobile systems, applications and services,
2003, pp. 31–42.
[44] B. Gedik and L. Liu, “Location privacy in mobile systems: A personalized anonymization model,” in
25th IEEE International Conference on Distributed Computing Systems (ICDCS’05). IEEE, 2005, pp.
620–629.
[45] W. LAN, Y . LIN, L. BAO, T. LI, M. CHEN, and J. SHAN, “Trajectory-differential privacy-protection
method with interest region,”Journal of Frontiers of Computer Science and Technology, vol. 14, no. 1, pp.
59–72, 2020.
[46] S. Zeng, J. Zhang, P. He, Y . Liu, Y . Xing, H. Xu, J. Ren, Y . Chang, S. Wang, D. Yinet al., “The good and
the bad: Exploring privacy issues in retrieval-augmented generation (rag),” inFindings of the Association
for Computational Linguistics: ACL 2024, 2024, pp. 4505–4524.
[47] K. Edemacu and X. Wu, “Privacy preserving prompt engineering: A survey,”ACM Computing Surveys,
vol. 57, no. 10, pp. 1–36, 2025.
[48] Y . Zhang, J. Wu, R. Li, T. Zhang, Y . Song, C. Li, S. Wang, H. Shen, J. Yin, J. Geet al., “Privacy protection
in rag: A novel method and evaluation framework,”Information Processing & Management, vol. 63, no. 3,
p. 104505, 2026.
[49] S. Zeng, J. Zhang, P. He, J. Ren, T. Zheng, H. Lu, H. Xu, H. Liu, Y . Xing, and J. Tang, “Mitigating the
privacy issues in retrieval-augmented generation (rag) via pure synthetic data,” inProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing, 2025, pp. 24 538–24 569.
[50] H. Duan, A. Dziedzic, M. Yaghini, N. Papernot, and F. Boenisch, “On the privacy risk of in-context
learning,”arXiv preprint arXiv:2411.10512, 2024.
[51] A. N. Carey, K. Bhaila, K. Edemacu, and X. Wu, “Dp-tabicl: In-context learning with differentially private
tabular data,” in2024 IEEE International Conference on Big Data (BigData). IEEE, 2024, pp. 1552–1557.
[52] Z. Liu, Z. Wang, T. Zhou, Y . Tang, Y . Luo, and Z. Cai, “Risk-aware privacy preservation for llm inference,”
IEEE Transactions on Information Forensics and Security, 2026.
[53] I. C. Ngong, S. R. Kadhe, H. Wang, K. Murugesan, J. D. Weisz, A. Dhurandhar, and K. N. Ramamurthy,
“Protecting users from themselves: Safeguarding contextual privacy in interactions with conversational
agents,” inFindings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 26 196–26 220.
[54] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in
Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP),
2019.
[55] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro,
F. Azharet al., “Llama: Open and efficient foundation language models,”arXiv preprint arXiv:2302.13971,
2023.
[56] G. Team, “Gemma: Open models based on gemini research and technology,”arXiv preprint
arXiv:2403.08295, 2024.
Appendices
A PAS vs Baseline Graphical Comparison
We also provide a graphical retrieval and generation performance comparison between the baseline
and PAS mechanism as presented in 5. We observe that PAS induces drop in retrieval and generation
performance. However, the drop is more for retrieval compared to downstream generation. This
visually confirms the utility results presented in Table 1.
12

MiniLM
Meta-Llama-3.3-70B-InstructMiniLM
gemma-3-12b-itbge-base
Meta-Llama-3.3-70B-Instructbge-base
gemma-3-12b-itbge-base
gpt-oss-120b0.00.20.40.60.8Recall@kBaseline vs PAS Recall@k
Baseline
PAS(a) Recall
MiniLM
Meta-Llama-3.3-70B-InstructMiniLM
gemma-3-12b-itbge-base
Meta-Llama-3.3-70B-Instructbge-base
gemma-3-12b-itbge-base
gpt-oss-120b0.00.20.40.60.8nDCG@kBaseline vs nDCG@k
Baseline
PAS (b) nDCG
MiniLM
Meta-Llama-3.3-70B-InstructMiniLM
gemma-3-12b-itbge-base
Meta-Llama-3.3-70B-Instructbge-base
gemma-3-12b-itbge-base
gpt-oss-120b0.00.20.40.60.8Citation CorrectnessBaseline vs PAS Citation Correctness
Baseline
PAS
(c) Citation
MiniLM
Meta-Llama-3.3-70B-InstructMiniLM
gemma-3-12b-itbge-base
Meta-Llama-3.3-70B-Instructbge-base
gemma-3-12b-itbge-base
gpt-oss-120b0.00.10.20.30.40.5F1 ScoreBaseline vs PAS F1 Score
Baseline
PAS (d) F1 Score
Figure 5: PAS Performance Comparison with the Baseline
B Retrieval vs Generation
Similarly, we evaluate how retrieval affects generation. We perform this by computing generation
metrics (citation correctness and F1 score) under varying retrieval metrics (Recall@k and nDCG@k)
for different retriever-generation model configurations as presented in Figure 6. In general, we
observe that the bge-base retriever performs worse in comparison to MiniLM, and the retrieval
performance has slight impact on citation correctness but close to negligible performance on F1
scores. This confirms our earlier claim that the model can still produce quality results with imperfect
retrievals.
C Experimental Setup
In this section, we present additional details on our experimental setup. For reproducibility, we
will release our implementation codes and dataset through the GitHub repository once the work is
accepted for publication.
C.1 Retriever Configuration
We employ dense retrieval models to capture semantic relevance between queries and
candidate documents. Specifically, we use two widely adopted embedding models:
sentence-transformers/all-MiniLM-L6-v2 [54] and BAAI/bge-base-en-v1.5 [28]. For
each query–document pair, semantic similarity is computed using cosine similarity in the embedding
space.
To incorporate spatial relevance, we use the hybrid scoring function defined in Equation 6, which
combines semantic and spatial signals. In the main experiments, we retrieve the top- k= 5 candidates
based on this combined score.
C.2 Privacy-Aware Spatial (PAS) Mechanism
We implement the PAS mechanism described in Section 2.2 by instantiating the probabilistic anchor
selection in Eq. 1 and the resulting uncertainty region U(z) . In practice, this mechanism is realized
through discrete anchor sets and binning functions for direction and distance, as defined earlier.
13

0.46 0.48 0.50 0.52 0.54 0.56
Recall@k0.550.560.570.580.590.600.610.62Citation CorrectnessRetrieval Coverage vs Grounding
MiniLM / Meta-Llama-3.3-70B-Instruct
MiniLM / gemma-3-12b-it
MiniLM / gpt-oss-120b
bge-base / Meta-Llama-3.3-70B-Instruct
bge-base / gemma-3-12b-it
bge-base / gpt-oss-120b(a) Recall vs Citation
0.46 0.48 0.50 0.52 0.54 0.56
Recall@k0.2500.2750.3000.3250.3500.3750.4000.425F1 ScoreRetrieval Coverage vs F1 Score
MiniLM / Meta-Llama-3.3-70B-Instruct
MiniLM / gemma-3-12b-it
MiniLM / gpt-oss-120b
bge-base / Meta-Llama-3.3-70B-Instruct
bge-base / gemma-3-12b-it
bge-base / gpt-oss-120b (b) Recall vs F1
0.42 0.44 0.46 0.48 0.50 0.52
nDCG@k0.550.560.570.580.590.600.610.62Citation CorrectnessnDCG Coverage vs Grounding
MiniLM / Meta-Llama-3.3-70B-Instruct
MiniLM / gemma-3-12b-it
MiniLM / gpt-oss-120b
bge-base / Meta-Llama-3.3-70B-Instruct
bge-base / gemma-3-12b-it
bge-base / gpt-oss-120b
(c) nDCG vs Citation
0.42 0.44 0.46 0.48 0.50 0.52
nDCG@k0.2500.2750.3000.3250.3500.3750.4000.425F1 ScorenDCG Coverage vs F1 Score
MiniLM / Meta-Llama-3.3-70B-Instruct
MiniLM / gemma-3-12b-it
MiniLM / gpt-oss-120b
bge-base / Meta-Llama-3.3-70B-Instruct
bge-base / gemma-3-12b-it
bge-base / gpt-oss-120b (d) nDCG vs F1
Figure 6: Retrieval vs Generation Performance
For all experiments, we use a fixed set of public anchors. The exponential mechanism in Equation 1
is applied withε∈ {1.0,2.0,5.0}and a scaling factors= 500.
Directional and distance discretization follows the binning scheme defined in Section 2.2, with
8-way directional bins ( N, NE, E, SE, S, SW, W, NW ) and distance bins ( 0-0.5mi, 0.5-1mi,
1-2mi, 2mi+ ). Given a sampled anchor a∗, we compute the corresponding token z= (a∗, B∗
θ, B∗
r)
using the deterministic post-processing procedure described previously.
At retrieval time, instead of operating on a single estimated user location, we approximate the
distribution π(x|z) over the uncertainty region U(z) using Monte Carlo sampling, as defined in
Eq. 4. Specifically, we draw K= 1000 latent user samples within U(z) and use these samples to
estimate the spatial scoreS sp(t|z)via the Monte Carlo formulation in Eq. 4.
In implementation, the PAS token also carries auxiliary metadata, including anchor identifiers and
sampling probabilities, to ensure consistent filtering during retrieval (e.g., restricting candidate
anchors when required). These components collectively enable the retriever to operate over a privacy-
preserving representation of user location while remaining compatible with the hybrid scoring
framework in Eq. 6.
C.3 Baseline Configuration
As a point of comparison, we consider a non-private baseline that assumes access to the user’s exact
location. The baseline uses the same dense retrievers and identical score fusion formulation, differing
only in the absence of PAS perturbation. This ensures that performance differences can be attributed
directly to the privacy mechanism.
We evaluate retrieval and generation performance using multiple metrics, including Recall@k,
nDCG@k, citation correctness, and F1 score.
14

System Prompt
You are a careful RAG answering assistant.
Answer the user’s query using ONLY the retrieved context.
Do not use outside knowledge.
If the evidence is weak or partially mismatched, say so briefly.
Prefer concise answers that directly name the best matching places.
Only cite documents that actually support your answer.
Return STRICT JSON with exactly these keys:
{
"answer": string,
"citations": ["title": string, "doc_id": string],
"faithfulness_notes": [string]
}
Do not include markdown fences.
User Prompt Template
User query: {raw_query}
Semantic target:
- entity_type: {entity}
- must_have_tags: {must_have_tags}
Spatial intent:
- direction_constraint: {direction_constraint}
- radius_miles: {radius_miles}
Retrieved context:
{context_text}
Instructions:
1. Write a grounded answer that directly answers the user query.
2. Mention strongest matching places first.
3. Exclude unsupported claims.
4. Mention uncertainty for weak spatial matches.
5. Cite only docs used in the answer.
6. Output STRICT JSON only.
Figure 7: Our prompt template used for grounded generation.
C.4 LLM-Based Generation
For answer generation, we use multiple large language models (LLMs), namely gpt-oss-120b ,
Meta-Llama-3.3-70B-Instruct [55], and gemma-3-12b-it [56]. Generation is guided by a
structured prompt template shown in Figure 7, in which a system prompt enforces strict grounding
in the retrieved context, and a user prompt provides the query, semantic targets (e.g., entities or
tags), spatial intent (direction and radius), and retrieved documents. All models are decoded using
low-temperature sampling (T= 0.1) to reduce variability and improve determinism.
C.5 Compute and Hardware Details
We provide details of the computational resources used to perform our experiments. All embedding-
based retrieval tasks using the retrieval models were conducted on a single NVIDIA A40 GPU. The
GPU was sufficient to perform all the retrieval tasks for the dataset scale used in the work. The
generator models were not hosted locally, but were accessed via an API-based interface provided
by SambaNova3. All the generation results were obtained through these API calls. This introduced
network latency, but it does not affect the correctness of the results.
3https://sambanova.ai/
15

NeurIPS Paper Checklist
1.Claims
Question: Do the main claims made in the abstract and introduction accurately reflect the
paper’s contributions and scope?
Answer: [Yes]
Justification: The abstract clearly states the core contributions of the PAS framework,
including, the structure achor-based spatial privacy mechanism, experimental evaluation
indicating ∼370-400m adversarial localization error with retention of over half of the
baseline utility performance and experimental analysis of the non-monotonic privacy-utility
relationship. These claims are consistent with the experimental results presented later in the
paper.
Guidelines:
•The answer [N/A] means that the abstract and introduction do not include the claims
made in the paper.
•The abstract and/or introduction should clearly state the claims made, including the
contributions made in the paper and important assumptions and limitations. A [No] or
[N/A] answer to this question will not be perceived well by the reviewers.
•The claims made should match theoretical and experimental results, and reflect how
much the results can be expected to generalize to other settings.
•It is fine to include aspirational goals as motivation as long as it is clear that these goals
are not attained by the paper.
2.Limitations
Question: Does the paper discuss the limitations of the work performed by the authors?
Answer: [Yes]
Justification: The paper explicitly indicates the limitations within Section 6 (conclusion
section), acknowledging the limitations of the proposed PAS framework. These limitations
include: geometric bias due to anchor discretizations, computational overhead and variance
due to Monte Carlo sampling, and use of synthetic data. The discussions offer potential
implications and future directions.
Guidelines:
•The answer [N/A] means that the paper has no limitation while the answer [No] means
that the paper has limitations, but those are not discussed in the paper.
• The authors are encouraged to create a separate “Limitations” section in their paper.
•The paper should point out any strong assumptions and how robust the results are to
violations of these assumptions (e.g., independence assumptions, noiseless settings,
model well-specification, asymptotic approximations only holding locally). The authors
should reflect on how these assumptions might be violated in practice and what the
implications would be.
•The authors should reflect on the scope of the claims made, e.g., if the approach was
only tested on a few datasets or with a few runs. In general, empirical results often
depend on implicit assumptions, which should be articulated.
•The authors should reflect on the factors that influence the performance of the approach.
For example, a facial recognition algorithm may perform poorly when image resolution
is low or images are taken in low lighting. Or a speech-to-text system might not be
used reliably to provide closed captions for online lectures because it fails to handle
technical jargon.
•The authors should discuss the computational efficiency of the proposed algorithms
and how they scale with dataset size.
•If applicable, the authors should discuss possible limitations of their approach to
address problems of privacy and fairness.
•While the authors might fear that complete honesty about limitations might be used by
reviewers as grounds for rejection, a worse outcome might be that reviewers discover
limitations that aren’t acknowledged in the paper. The authors should use their best
16

judgment and recognize that individual actions in favor of transparency play an impor-
tant role in developing norms that preserve the integrity of the community. Reviewers
will be specifically instructed to not penalize honesty concerning limitations.
3.Theory assumptions and proofs
Question: For each theoretical result, does the paper provide the full set of assumptions and
a complete (and correct) proof?
Answer: [No]
Justification: The paper presents a theoretical result (Lemma 1), stating that PAS satisfies
ε-geo-indistinguishability. While the paper presents the mechanism, assumptions and an
intuitive argument, it only contains a proof sketch. This is because, the proof relies on the
standard properties of exponential mechanism and post-processing invariance, and thus, no
step-by-step derivation for complete proof.
Guidelines:
• The answer [N/A] means that the paper does not include theoretical results.
•All the theorems, formulas, and proofs in the paper should be numbered and cross-
referenced.
•All assumptions should be clearly stated or referenced in the statement of any theorems.
•The proofs can either appear in the main paper or the supplemental material, but if
they appear in the supplemental material, the authors are encouraged to provide a short
proof sketch to provide intuition.
•Inversely, any informal proof provided in the core of the paper should be complemented
by formal proofs provided in appendix or supplemental material.
• Theorems and Lemmas that the proof relies upon should be properly referenced.
4.Experimental result reproducibility
Question: Does the paper fully disclose all the information needed to reproduce the main ex-
perimental results of the paper to the extent that it affects the main claims and/or conclusions
of the paper (regardless of whether the code and data are provided or not)?
Answer: [Yes]
Justification: The paper provides detailed information to reproduce the main experimental
results. The dataset construction is fully described. The PAS mechanism is fully specified
from the probabilistic anchor selection to latent user sampling. The retrieval pipeline is
clearly defined including the retrievers and generators used. The evaluation metrics are
explicitly defined, and the used hyperparameters are reported.
Guidelines:
• The answer [N/A] means that the paper does not include experiments.
•If the paper includes experiments, a [No] answer to this question will not be perceived
well by the reviewers: Making the paper reproducible is important, regardless of
whether the code and data are provided or not.
•If the contribution is a dataset and/or model, the authors should describe the steps taken
to make their results reproducible or verifiable.
•Depending on the contribution, reproducibility can be accomplished in various ways.
For example, if the contribution is a novel architecture, describing the architecture fully
might suffice, or if the contribution is a specific model and empirical evaluation, it may
be necessary to either make it possible for others to replicate the model with the same
dataset, or provide access to the model. In general. releasing code and data is often
one good way to accomplish this, but reproducibility can also be provided via detailed
instructions for how to replicate the results, access to a hosted model (e.g., in the case
of a large language model), releasing of a model checkpoint, or other means that are
appropriate to the research performed.
•While NeurIPS does not require releasing code, the conference does require all submis-
sions to provide some reasonable avenue for reproducibility, which may depend on the
nature of the contribution. For example
(a)If the contribution is primarily a new algorithm, the paper should make it clear how
to reproduce that algorithm.
17

(b)If the contribution is primarily a new model architecture, the paper should describe
the architecture clearly and fully.
(c)If the contribution is a new model (e.g., a large language model), then there should
either be a way to access this model for reproducing the results or a way to reproduce
the model (e.g., with an open-source dataset or instructions for how to construct
the dataset).
(d)We recognize that reproducibility may be tricky in some cases, in which case
authors are welcome to describe the particular way they provide for reproducibility.
In the case of closed-source models, it may be that access to the model is limited in
some way (e.g., to registered users), but it should be possible for other researchers
to have some path to reproducing or verifying the results.
5.Open access to data and code
Question: Does the paper provide open access to the data and code, with sufficient instruc-
tions to faithfully reproduce the main experimental results, as described in supplemental
material?
Answer: [Yes]
Justification: All the resource will be shared publicly via a repository once the paper
is accepted for publication. The repository will contain the dataset and implementation
codes used during the experiment. Particularly, it will include: the dataset files used for
benchmarking, the end-to-end PAS pipeline implementations, the configurations that enable
reproduction of the main results and an example illustration of the PAS mechanism at work.
Together, these resources will allow a reader to reproduce the reported experimental results.
Guidelines:
• The answer [N/A] means that paper does not include experiments requiring code.
•Please see the NeurIPS code and data submission guidelines ( https://neurips.cc/
public/guides/CodeSubmissionPolicy) for more details.
•While we encourage the release of code and data, we understand that this might not
be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not
including code, unless this is central to the contribution (e.g., for a new open-source
benchmark).
•The instructions should contain the exact command and environment needed to run to
reproduce the results. See the NeurIPS code and data submission guidelines ( https:
//neurips.cc/public/guides/CodeSubmissionPolicy) for more details.
•The authors should provide instructions on data access and preparation, including how
to access the raw data, preprocessed data, intermediate data, and generated data, etc.
•The authors should provide scripts to reproduce all experimental results for the new
proposed method and baselines. If only a subset of experiments are reproducible, they
should state which ones are omitted from the script and why.
•At submission time, to preserve anonymity, the authors should release anonymized
versions (if applicable).
•Providing as much information as possible in supplemental material (appended to the
paper) is recommended, but including URLs to data and code is permitted.
6.Experimental setting/details
Question: Does the paper specify all the training and test details (e.g., data splits, hyperpa-
rameters, how they were chosen, type of optimizer) necessary to understand the results?
Answer: [No]
Justification: The paper provides significant amount of details about the experimental setup,
including dataset construction, retrieval pipeline, PAS mechanism and key parameters.
However, the work does not include training a new model, but it uses existing retrievers and
generation models. The configuration details will be included in the source code once the
work is accepted for publication.
Guidelines:
• The answer [N/A] means that the paper does not include experiments.
18

•The experimental setting should be presented in the core of the paper to a level of detail
that is necessary to appreciate the results and make sense of them.
•The full details can be provided either with the code, in appendix, or as supplemental
material.
7.Experiment statistical significance
Question: Does the paper report error bars suitably and correctly defined or other appropriate
information about the statistical significance of the experiments?
Answer: [No]
Justification: The paper reports mean performance metrics across evaluation queries. It does
not include error bars, confidence intervals, or statistical tests such as standard deviation,
bootstrap intervals, and hypothesis testing. The experiments are conducted over large evalu-
ation sets (423 queries) and report aggregate metrics that exhibit consistent and substantial
differences between the baseline and the PAS. Such effect sizes are significantly larger than
typical variances introduced by sampling noise.
Guidelines:
• The answer [N/A] means that the paper does not include experiments.
•The authors should answer [Yes] if the results are accompanied by error bars, confidence
intervals, or statistical significance tests, at least for the experiments that support the
main claims of the paper.
•The factors of variability that the error bars are capturing should be clearly stated (for
example, train/test split, initialization, random drawing of some parameter, or overall
run with given experimental conditions).
•The method for calculating the error bars should be explained (closed form formula,
call to a library function, bootstrap, etc.)
• The assumptions made should be given (e.g., Normally distributed errors).
•It should be clear whether the error bar is the standard deviation or the standard error
of the mean.
•It is OK to report 1-sigma error bars, but one should state it. The authors should
preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis
of Normality of errors is not verified.
•For asymmetric distributions, the authors should be careful not to show in tables or
figures symmetric error bars that would yield results that are out of range (e.g., negative
error rates).
•If error bars are reported in tables or plots, the authors should explain in the text how
they were calculated and reference the corresponding figures or tables in the text.
8.Experiments compute resources
Question: For each experiment, does the paper provide sufficient information on the com-
puter resources (type of compute workers, memory, time of execution) needed to reproduce
the experiments?
Answer: [Yes]
Justification: Much as the paper focuses on the methodology, dataset, and evaluation
pipelines. It specifies the retrievers and generation models used, it provides detailed infor-
mation on the computational resources used in Section C.5.
Guidelines:
• The answer [N/A] means that the paper does not include experiments.
•The paper should indicate the type of compute workers CPU or GPU, internal cluster,
or cloud provider, including relevant memory and storage.
•The paper should provide the amount of compute required for each of the individual
experimental runs as well as estimate the total compute.
•The paper should disclose whether the full research project required more compute
than the experiments reported in the paper (e.g., preliminary or failed experiments that
didn’t make it into the paper).
9.Code of ethics
19

Question: Does the research conducted in the paper conform, in every respect, with the
NeurIPS Code of Ethicshttps://neurips.cc/public/EthicsGuidelines?
Answer: [Yes]
Justification: The research complies with NeurIPS Code of Ethics in all important aspect.
The dataset used is synthetic, thus avoids the use of real user location data and thereby
eliminating the concerns associated with personal data misuse and privacy violations. The
methodology does not involve use of human subjects, sensitive attribute and potentially
harmful application. Finally, no deceptive and unsafe practices are employed.
Guidelines:
•The answer [N/A] means that the authors have not reviewed the NeurIPS Code of
Ethics.
•If the authors answer [No], they should explain the special circumstances that require a
deviation from the Code of Ethics.
•The authors should make sure to preserve anonymity (e.g., if there is a special consid-
eration due to laws or regulations in their jurisdiction).
10.Broader impacts
Question: Does the paper discuss both potential positive societal impacts and negative
societal impacts of the work performed?
Answer: [Yes]
Justification: The paper explicitly emphasizes the societal impact of PAS framework, par-
ticularly, it enables privacy-preserving retrieval, thus reducing the risks associated with
location tracking and data exposure.
Guidelines:
• The answer [N/A] means that there is no societal impact of the work performed.
•If the authors answer [N/A] or [No], they should explain why their work has no societal
impact or why the paper does not address societal impact.
•Examples of negative societal impacts include potential malicious or unintended uses
(e.g., disinformation, generating fake profiles, surveillance), fairness considerations
(e.g., deployment of technologies that could make decisions that unfairly impact specific
groups), privacy considerations, and security considerations.
•The conference expects that many papers will be foundational research and not tied
to particular applications, let alone deployments. However, if there is a direct path to
any negative applications, the authors should point it out. For example, it is legitimate
to point out that an improvement in the quality of generative models could be used to
generate Deepfakes for disinformation. On the other hand, it is not needed to point out
that a generic algorithm for optimizing neural networks could enable people to train
models that generate Deepfakes faster.
•The authors should consider possible harms that could arise when the technology is
being used as intended and functioning correctly, harms that could arise when the
technology is being used as intended but gives incorrect results, and harms following
from (intentional or unintentional) misuse of the technology.
•If there are negative societal impacts, the authors could also discuss possible mitigation
strategies (e.g., gated release of models, providing defenses in addition to attacks,
mechanisms for monitoring misuse, mechanisms to monitor how a system learns from
feedback over time, improving the efficiency and accessibility of ML).
11.Safeguards
Question: Does the paper describe safeguards that have been put in place for responsible
release of data or models that have a high risk for misuse (e.g., pre-trained language models,
image generators, or scraped datasets)?
Answer: [N/A]
Justification: The paper does not introduce or release any high-risk models or datasets.
Guidelines:
20

• The answer [N/A] means that the paper poses no such risks.
•Released models that have a high risk for misuse or dual-use should be released with
necessary safeguards to allow for controlled use of the model, for example by requiring
that users adhere to usage guidelines or restrictions to access the model or implementing
safety filters.
•Datasets that have been scraped from the Internet could pose safety risks. The authors
should describe how they avoided releasing unsafe images.
•We recognize that providing effective safeguards is challenging, and many papers do
not require this, but we encourage authors to take this into account and make a best
faith effort.
12.Licenses for existing assets
Question: Are the creators or original owners of assets (e.g., code, data, models), used in
the paper, properly credited and are the license and terms of use explicitly mentioned and
properly respected?
Answer: [Yes]
Justification: The paper appropriately credits all assets used in the work. The pretrained
models e.g., bge-base, MiniLM, Meta-Llama-3.3-70B, gemma-3-12b-it, and gpt-oss-120b
are all identified and cited. The dataset is synthetically generated, avoiding reliance on third
party proprietory data and associated license issues. The code has proper attributions to
dependencies and libraries. No restricted asset is used without acknowledgment.
Guidelines:
• The answer [N/A] means that the paper does not use existing assets.
• The authors should cite the original paper that produced the code package or dataset.
•The authors should state which version of the asset is used and, if possible, include a
URL.
• The name of the license (e.g., CC-BY 4.0) should be included for each asset.
•For scraped data from a particular source (e.g., website), the copyright and terms of
service of that source should be provided.
•If assets are released, the license, copyright information, and terms of use in the
package should be provided. For popular datasets, paperswithcode.com/datasets
has curated licenses for some datasets. Their licensing guide can help determine the
license of a dataset.
•For existing datasets that are re-packaged, both the original license and the license of
the derived asset (if it has changed) should be provided.
•If this information is not available online, the authors are encouraged to reach out to
the asset’s creators.
13.New assets
Question: Are new assets introduced in the paper well documented and is the documentation
provided alongside the assets?
Answer: [Yes]
Justification: The paper introduces new assets such as the synthetic spatial dataset and the
PAS implementation. These assets will be released via public repositories once the work is
accepted for publication. They will be accompanied by descriptive metadata and schema.
An illustrative example will also be added to demonstrate PAS at work.
Guidelines:
• The answer [N/A] means that the paper does not release new assets.
•Researchers should communicate the details of the dataset/code/model as part of their
submissions via structured templates. This includes details about training, license,
limitations, etc.
•The paper should discuss whether and how consent was obtained from people whose
asset is used.
•At submission time, remember to anonymize your assets (if applicable). You can either
create an anonymized URL or include an anonymized zip file.
21

14.Crowdsourcing and research with human subjects
Question: For crowdsourcing experiments and research with human subjects, does the paper
include the full text of instructions given to participants and screenshots, if applicable, as
well as details about compensation (if any)?
Answer: [N/A]
Justification: The paper does not involve crowdsourcing experiments or human subjects. All
the experiments are conducted over a synthetic dataset.
Guidelines:
•The answer [N/A] means that the paper does not involve crowdsourcing nor research
with human subjects.
•Including this information in the supplemental material is fine, but if the main contribu-
tion of the paper involves human subjects, then as much detail as possible should be
included in the main paper.
•According to the NeurIPS Code of Ethics, workers involved in data collection, curation,
or other labor should be paid at least the minimum wage in the country of the data
collector.
15.Institutional review board (IRB) approvals or equivalent for research with human
subjects
Question: Does the paper describe potential risks incurred by study participants, whether
such risks were disclosed to the subjects, and whether Institutional Review Board (IRB)
approvals (or an equivalent approval/review based on the requirements of your country or
institution) were obtained?
Answer: [N/A]
Justification: The paper does not involve human subjects and all the experiments are
conducted over a synthetic dataset. Thus, there are no participant risks to disclose and IRB
approval is not required.
Guidelines:
•The answer [N/A] means that the paper does not involve crowdsourcing nor research
with human subjects.
•Depending on the country in which research is conducted, IRB approval (or equivalent)
may be required for any human subjects research. If you obtained IRB approval, you
should clearly state this in the paper.
•We recognize that the procedures for this may vary significantly between institutions
and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the
guidelines for their institution.
•For initial submissions, do not include any information that would break anonymity (if
applicable), such as the institution conducting the review.
16.Declaration of LLM usage
Question: Does the paper describe the usage of LLMs if it is an important, original, or
non-standard component of the core methods in this research? Note that if the LLM is used
only for writing, editing, or formatting purposes and doesnotimpact the core methodology,
scientific rigor, or originality of the research, declaration is not required.
Answer: [Yes]
Justification: The paper clearly describes the role of LLMs as part of the evaluation pipeline,
where they are used for answer generation in the Spatial RAG setting. Additionally, the
paper acknowledges that ChatGPT was used during dataset construction, specifically to
assist in generating components of the synthetic queries and metadata. This usage does not
affect the PAS methodology but it is important for transparency.
Guidelines:
•The answer [N/A] means that the core method development in this research does not
involve LLMs as any important, original, or non-standard components.
•Please refer to our LLM policy in the NeurIPS handbook for what should or should not
be described.
22