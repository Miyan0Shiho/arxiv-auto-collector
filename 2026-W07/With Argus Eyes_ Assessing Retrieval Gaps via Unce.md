# With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots

**Authors**: Zeinab Sadat Taghavi, Ali Modarressi, Hinrich Schutze, Andreas Marfurt

**Published**: 2026-02-10 10:04:55

**PDF URL**: [https://arxiv.org/pdf/2602.09616v1](https://arxiv.org/pdf/2602.09616v1)

## Abstract
Reliable retrieval-augmented generation (RAG) systems depend fundamentally on the retriever's ability to find relevant information. We show that neural retrievers used in RAG systems have blind spots, which we define as the failure to retrieve entities that are relevant to the query, but have low similarity to the query embedding. We investigate the training-induced biases that cause such blind spot entities to be mapped to inaccessible parts of the embedding space, resulting in low retrievability. Using a large-scale dataset constructed from Wikidata relations and first paragraphs of Wikipedia, and our proposed Retrieval Probability Score (RPS), we show that blind spot risk in standard retrievers (e.g., CONTRIEVER, REASONIR) can be predicted pre-index from entity embedding geometry, avoiding expensive retrieval evaluations. To address these blind spots, we introduce ARGUS, a pipeline that enables the retrievability of high-risk (low-RPS) entities through targeted document augmentation from a knowledge base (KB), first paragraphs of Wikipedia, in our case. Extensive experiments on BRIGHT, IMPLIRET, and RAR-B show that ARGUS achieves consistent improvements across all evaluated retrievers (averaging +3.4 nDCG@5 and +4.5 nDCG@10 absolute points), with substantially larger gains in challenging subsets. These results establish that preemptively remedying blind spots is critical for building robust and trustworthy RAG systems (Code and Data).

## Full Text


<!-- PDF content starts -->

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect
and Remedy Retrieval Blind Spots
Zeinab Sadat Taghavi1Ali Modarressi2 3Hinrich Sch ¨utze2 3Andreas Marfurt1
Abstract
Reliable retrieval-augmented generation (RAG)
systems depend fundamentally on the retriever’s
ability to find relevant information. We show that
neural retrievers used in RAG systems haveblind
spots, which we define as the failure to retrieve
entities that are relevant to the query, but have low
similarity to the query embedding. We investigate
the training-induced biases that cause such blind-
spot entities to be mapped to inaccessible parts
of the embedding space, resulting in low retriev-
ability. Using a large-scale dataset constructed
from Wikidata relations and first paragraphs of
Wikipedia, and our proposed Retrieval Probability
Score (RPS), we show that blind spot risk in stan-
dard retrievers (e.g., CONTRIEVER, REASONIR)
can be predicted pre-index from entity embed-
ding geometry, avoiding expensive retrieval evalu-
ations. To address these blind spots, we introduce
ARGUS, a pipeline that enables the retrievability
of high-risk (low-RPS) entities through targeted
document augmentation from a knowledge base
(KB), first paragraphs of Wikipedia, in our case.
Extensive experiments on BRIGHT, IMPLIRET,
and RAR-Bshow that ARGUS achieves consis-
tent improvements across all evaluated retrievers
(averaging +3.4 nDCG@5 and +4.5 nDCG@10
absolute points), with substantially larger gains
in challenging subsets. These results establish
that preemptively remedying blind spots is critical
for building robust and trustworthy RAG systems
(Code and Data).
Note:Part of this work was done while Z. Taghavi was at
CIS, LMU Munich, and associated with MCML.1Lucerne Uni-
versity of Applied Sciences and Arts (HSLU)2Center for Infor-
mation and Language Processing (CIS), Ludwig Maximilian Uni-
versity of Munich (LMU)3Munich Center for Machine Learning
(MCML). Correspondence to: Zeinab Sadat Taghavi <zeinabsa-
tad.taghavi@hslu.ch>.1. Introduction
Retrieval-augmented generation (RAG) has become a core
building block of modern NLP systems, powering question
answering, assistants, and tool-using agents by grounding
generation in external evidence (Lewis et al., 2020; Guu
et al., 2020; Gao et al., 2023). In these settings, trustworthi-
ness hinges on a single component, which is the retriever’s
ability to surface the right information when it is needed
(Gao et al., 2023). Otherwise, the retriever becomes a single
point of failure for the overall pipeline, undermining robust-
ness (Gao et al., 2023). Today’s RAG pipelines increas-
ingly rely on neural retrievers, which outperform lexical
methods such as BM25 by capturing semantic similarity
beyond surface word overlap (Robertson & Zaragoza, 2009;
Karpukhin et al., 2020; Izacard et al., 2022). However, this
shift toward semantic matching can introduce a new failure
mode; retrieval depends on how queries and documents are
positioned in the embedding space, and some relevant infor-
mation can become systematically harder to retrieve due to
unfavorable embedding geometry.
This geometric failure creates a silent bottleneck, compro-
mising the retriever’s robustness against specialized or se-
mantically distant entities. Relevant evidence may exist in
the corpus, but the retriever fails to surface it, causing the
generator to fall back on ungrounded completions or hallu-
cinations (Ji et al., 2023). We therefore ask whether such
misses are merely random noise or instead reflect a deeper
systematic phenomenon in neural retrieval. In particular,
we study systematic blind spots: entity-centric gaps where
certain entities (and the documents that mention them) are
consistently missed even when they are relevant to the query,
especially when relevance is semantic rather than driven by
lexical overlap. Crucially, blind spots are relative to the
retrieval budget; with a larger top- kwindow, more evidence
becomes accessible. But, relying on very large retrieval win-
dows is often undesirable in RAG, because longer contexts
can dilute the generator’s effective focus and reduce down-
stream quality (e.g., “lost-in-the-middle” / “context rot”)
(Liu et al., 2024; Hong et al., 2025; Modarressi et al., 2025;
Taghavi et al., 2025). Consequently, ensuring that relevant
entities are geometrically accessible under practical budgets
(e.g., k∈[5,50] ) is not only an efficiency consideration but
1arXiv:2602.09616v1  [cs.IR]  10 Feb 2026

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
also fundamental for ensuring that RAG pipelines remain
robust and trustworthy across diverse downstream tasks.
Existing retrieval evaluation and optimization are largely
query-centric and post-hoc. They measure performance
conditioned on a benchmark’s queries, but do not reveal
which entities are intrinsically hard to retrieve, nor do they
support pre-deployment auditing of what a retriever will sys-
tematically miss (Bajaj et al., 2016; Thakur et al., 2021;
Muennighoff et al., 2023). This limitation is amplified
by the mismatch between domain-limited benchmarks and
the broad, web-scale training of neural retrievers, where
domain-specific test suites may overlook global failure pat-
terns (Thakur et al., 2021; SU et al., 2025; Xiao et al.,
2024). To obtain a complementary, domain-agnostic view,
we construct a large random sample of Wikidata-Wikipedia
aligned entities and use it to quantify entity-level retrievabil-
ity risk (Vrande ˇci´c & Kr ¨otzsch, 2014; Wikimedia Founda-
tion, 2025). Specifically, we introduce the Retrieval Proba-
bility Score (RPS), defined with respect to the user’s top- k
budget. Formally, RPSkis the expected value of top-k hit
probability over an entity’s related query set that is derived
from Wikidata Knowledge Graphs (KG) relations. This
metric evaluates retrievability by measuring the frequency
with which a target entity surfaces in the top- kwhen ranked
against a large pool of strictly disjoint neutral entities. These
neutral candidates are randomly sampled Wikidata entities
filtered to exclude all directly linked neighbors of both the
target and query entities in the Wikidata KG, ensuring a
controlled assessment of geometric robustness. Intuitively,
RPS krepresents retrieval consistency; for instance, an
RPS k= 0.2 indicates the entity is successfully retrieved
for only 20% of its associated queries. Equivalently, a low
RPS signals a high miss probability, a systematic blind spot,
motivating the question of whether such failures can be
predicted pre-index directly from embedding geometry.
Applying RPS at scale reveals substantial variation in re-
trievability. Across neural retrievers, entities range from
consistently retrievable to persistently missed. Moreover,
this variation is not arbitrary; when we assign entities
tolow/mid/highRPS terciles and visualize their embed-
dings using a two-dimensional Linear Discriminant Anal-
ysis (LDA) projection, low- and high-RPS entities occupy
geometrically distinguishable regions in embedding space,
indicating structured blind-spot zones rather than random
failures. As we increase the neutral pool size, these projec-
tions become more diagnostic. For robust retrievers, a larger
fraction of entity embedding points remains in the high-RPS
region, whereas for standard baselines, more entity points
shift into and accumulate within low-RPS regions. Crucially,
this geometric regularity implies that retrievability risk is
encoded in the embeddings; we train lightweight diagnostic
probes on entity embeddings labeled with empirical RPS k,
enabling pre-index prediction of high-risk entities without
Calculating Retrieval Probability Score (RPS)
1. Constructing a subgraph of randomly sampled Wikidata entities linked to valid Wikipedia ﬁrst paragraphs that explicitly mention the entity's label, while identifying 2-hop neighbors to enforce strict disjointness between Related tags and Neutral entities.
……………………
2. Constructing a unique candidate set for each related tag (acting as a query) containing the target entity and  randomly sampled, strictly disjoint Neutral entities—entities veriﬁed to have no direct Wikidata connection to the query.N−1
3. Retrieving the Top- candidates using embeddings conditioned on Wikipedia ﬁrst paragraphs, and calculating the RPS for Top-k () as the average of binary success indicators (1 if the target appears in the top-, 0 otherwise) across all related tags.kRPSkkSwitzerlandSt. Martin’s ChurchPhotosynthesisTheory of RelativityHarry PotterChurch buildingZillis-ReischenRomanesque architectureRPSk()=1(Hits)4(RelatedTags)=0.25St. Martin’s Church, ZillisAlbuquerqueAmazonPrinceton UniversityBankers TrustMacKenzie ScottRPSk()=4(Hits)5(RelatedTags)=0.80Jeff BezosZillis-ReischenChurch buildingSt. Martin's ChurchSwitzerlandRomanesque art
Jeﬀ BezosBankers TrustMacKenzie ScottAlbuquerquePrinceton UniversityAmazon…
Wikidata linkQueryTop-k missTop-k hitFirst paragraph of WikipediaWikidata entityNeutral entityRelated entity (query)Subgraph   of          Wikidata
Romanesque architectureFirst Romanesque
Target Wikidata entityFigure 1.Retrieval Probability Score (RPS) computation and
retriever blind-spot analysis. (Top)Evaluation pipeline: (1)
construct a Wikidata–Wikipedia aligned dataset, (2) build query-
specific retrieval sets with strictly disjoint neutral entities, and
(3) compute RPS from retrieval consistency.(Bottom)Average
RPS over a large random entity sample at k= 50 withN= 800
neutrals (suppressing chance hits). Standard retrievers succeed
only rarely (e.g., Contriever ≈0.11 ), implying that for a random
entity nearly 90% of valid top-kretrieval opportunities fail.
expensive end-to-end retrieval simulations.
Building on these findings, we propose ARGUS (Assessing
RetrievalGaps viaUncertaintyScoring ), a diagnosis-to-
remedy pipeline for retriever blind spots. ARGUS first pre-
dicts entity retrievability ( RPSk) under a target retriever and
flags high-risk entities via thresholding ( RPSk< τ). It then
remedies these blind spots through targeted knowledge aug-
mentation from a Reference KB, constructing augmented
document views via either document expansion by concate-
nation or KB-guided LLM synthesis, and indexing these
views alongside the original. Across BRIGHT, IMPLIRET,
and RAR-B, ARGUS (J ¨arvelin & Kek ¨al¨ainen, 2002; SU
et al., 2025; Xiao et al., 2024; Taghavi et al., 2025), yields
consistent improvements in retrieval scores (nDCG@5/10)
across eight popular neural retrievers.
Contributions:(i) We introduce RPS and a large-scale
Wikidata-Wikipedia aligned protocol for assessing entity-
level retrievability risk. (ii) We show that blind-spot risk
is predictable from embedding representations, enabling
pre-index detection via lightweight probes. (iii) We propose
2

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
BGE-M3ReasonIR
N=400N=600N=800N=200N=100
Figure 2.LDA projections of entity embeddings labeled by RPS
terciles (low/mid/high) at k= 50 under increasing neutral
pool sizesN, comparing a low-RPS retriever (BGE-M3) to a
high-RPS retriever (REASONIR-8B).Robust retrievers retain
denser high-RPS regions (blue) as Ngrows, indicating higher
expected top- kretrievability for a random entity, while persistent
low-RPS regions (red) across models confirm intrinsic blind spots.
ARGUS, a practical remedy pipeline that augments high-
risk entities through targeted KB context. (iv) ARGUS
demonstrates robust nDCG gains across benchmarks and
retriever architectures.
2. Related Work
Neural Retrieval.Neural information retrieval has largely
shifted from lexical matching to dense retrieval, where a
query encoder and a document encoder, that is often a dual-
encoder architecture, map text into a shared embedding
space and rank candidates by vector similarity (i.e., cosine
similarity) (Karpukhin et al., 2020). This paradigm under-
lies many modern retriever families used in RAG pipelines,
including general-purpose dense models and more special-
ized retrievers trained for stronger reasoning or supervi-
sion (e.g., BGE-M3, JINA-V3, and REASONIR-8B) (Chen
et al., 2024; Sturua et al., 2024; Shao et al., 2025). Because
retrieval decisions are mediated by the geometry of these
learned representations, dense retrievers can succeed be-
yond surface overlap but also exhibit systematic behaviors
tied to how entities and contexts are embedded (Izacard
et al., 2022; Santhanam et al., 2022). Our work targets this
setting, focusing on diagnosing and mitigating entity-level
blind spots in neural retrievers.
Reliable RAG and Pre-index Auditing.In RAG settings,
trustworthiness hinges on the retriever: when evidence is
not surfaced, generators can hallucinate even if the knowl-
edge exists in the corpus (Lewis et al., 2020; Shuster et al.,
2021; Ji et al., 2023). While prior work strengthens the
retrieval stack through query-side interventions or post-
retrieval reranking (Nogueira & Cho, 2019; Karpukhin et al.,
2020; Ma et al., 2023), these methods typically treat the
index as given. Consequently, standard retrieval evalua-
tion remains predominantly query-centric, relying on fixed
benchmarks (e.g., MS MARCO, BEIR) to estimate average
performance (Bajaj et al., 2016; Thakur et al., 2021). Al-though research has examined robustness to hard negatives,
dataset bias, and distribution shifts, these approaches are
largely post-hoc and require ground-truth queries to surface
failures (Xiong et al., 2021; Yu et al., 2022; Mallen et al.,
2023; Thakur et al., 2024). To enable pre-deployment au-
diting, we propose a shift to entity-centric risk estimation.
By defining the RPS, we quantify intrinsic retrievability risk
from embedding representations, allowing blind spots to be
predicted and mitigated at indexing time.
3. Assessing Retriever Blind Spots
To distinguish intrinsic blind spots from idiosyncratic query
effects, we audit neural retrievers at theentity level, asking
how reliably an entity can be surfaced under a fixed top-
kbudget across many query contexts. This motivates a
large-scale, controlled measurement setup that decouples
retrievability from query distributions.
3.1. Wikidata-Wikipedia Alignment for Retrievability
Profiling
Leveraging the diverse relations in Wikidata, we construct
a large-scale, domain-agnostic dataset for entity-centric re-
trievability auditing by aligning Wikidata’s structured graph
with Wikipedia’s text through the following pipeline (Fig-
ure 1):
(1) Entity Sampling.We randomly sample 7×106Wikidata
entities and retain those with an English Wikipedia page,
yielding a target set X. We apply lightweight cleaning
and keep only entities with at least one valid related entity
(details in Appendix B.1).
(2) Context Grounding.For each x∈X , we use the first
paragraph of its Wikipedia page as the canonical context wx,
and enforce that the entity’s Wikidata surface form appears
inwx, otherwise we minimally prepend it as a separate span
at the beginning of the text, so mention-based pooling is
well-defined, and alignments remain clean (Appendix B.1).
(3) Related Entities (Query Construction).From Wiki-
data 1-hop relations, we derive a set of related entities Tx
that have English Wikipedia pages. We treat each related
entity’s Wikidata surface form as the query, but compute
a context-conditioned query embedding by encoding its
Wikipedia first paragraph to reduce ambiguity from polyse-
mous labels (e.g., there are many “St. Martin’s” churches).
These related entities serve as proxy queries capturing con-
texts in which xshould be retrievable (e.g.,St. Martin’s
Church→Romanesque architecture).
(4) Neutral Baseline (Controlled Competition).For each
related entity t∈ T x, we construct a related-entity-specific
neutral pool Zneut(t)of size N, where each z∈ Z neut(t)
is a neutral entity, and enforce KG disjointness, i.e., each
3

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
neutral zis not directly linked to tin Wikidata (no 1-hop
edge under any property). As with targets, each neutral
item is represented by its Wikipedia first paragraph, and we
require its Wikidata surface form to be explicitly mentioned
in that paragraph. This yields a controlled setting in which
failures are less attributable to semantic ambiguity and more
indicative of the retriever’s embedding geometry. We next
define RPS over these related-entity-specific pools, treating
kas the user-defined retrieval budget, and later select a
conservative Nto ensure RPS is stable and not driven by
random hits.
3.2. Retrieval Probability Score (RPS)
We quantify entity-level retrievability under a fixed top- k
budget using theRetrieval Probability Score(RPS). Let
Eθ(·)denote the target retriever encoder (token-level when
available), and let g(·)denote the retriever-specific pooling
operator that extracts a single mention-aware vector from
these representations given a mention span. We represent
an entity uby an embedding eu=g(E θ(wu), su)∈Rh,
where wuis the Wikipedia first paragraph of uandsuis the
span of u’s Wikidata surface-form mention within wu. We
always encode the full paragraph wu; the span suis used
only by g(·)to extract a mention-aware pooled embedding
(details in Appendix B.4).
We apply the same encoding procedure to all entities. Each
related entity ti∈ Txyields a query embedding qi=eti
from its Wikipedia first paragraph with pooling at its men-
tion span, and the target xand neutrals zi,j∈ Z neut(ti)yield
candidate embeddings exandezi,jcomputed identically
from their own paragraphs and mention spans.
Controlled retrieval pools.For each related entity ti∈ Tx,
we form a candidate set of size Nby placing the target
entityxalongsideN−1neutrals:
Ci={x}∪{z i,1, . . . , z i,N−1}, z i,j∼Uniform(Z neut(ti)).
We rank candidates c∈ C iby cosine similarity cos(q i,ec)
and define a top-khitas:
Hitk(x, ti) =I
rank(x|t i,Ci)≤k
.
RPS definition and interpretation.We define RPS of
entity xas the expected value of a successful hit across the
distribution of all potentially relevant facts or queries P(Tx)
associated with it:
RPSk(x|w x) =E t∼P(T x)
Hitk(x, t)
.
In practice, we approximate this expectation via the empiri-
cal average hit rate across the sampled related entities:
RPSk(x|w x)≈1
|Tx|X
ti∈TxHitk(x, ti).
Figure 3.Impact of neutral pool size ( N) on fraction of entities
with RPS k>0.5 (k= 50 ).AtN= 100 , successful retrieval
rates match the chance regime ( k/N≈0.5 ). Beyond N≥400 ,
curves decouple from chance and plateau, revealing stable, model-
specific behavior. Hence, we adopt N= 800 , so that high RPS
reflects genuine geometric retrievability.
Intuitively, RPSk(x|w x)represents the probability of dis-
covery. Low RPS implies a high expected miss probability,
which is a blind spot, under the given top- kbudget, whereas
high RPS indicates consistent geometric discoverability in
the retriever’s embedding space.
Geometric Structure of Blind Spots.Applying RPS at
scale reveals substantial variation in retrievability: standard
retrievers such as CONTRIEVERexhibit consistently low av-
erage scores, whereas robust models like ReasonIR surface
significantly more entities (Figure 1, bottom). To confirm
these failures are structural rather than random, we visual-
ize entity embeddings via LDA after partitioning entities
into RPS terciles. The projections reveal a clear geometric
spectrum where low-RPS entities (red) cluster into distinct
blind spot regions separable from high-RPS areas (blue),
a pattern that can be seen in their histogram and persists
even in robust architectures(Figure 2). The complete set of
projections for all retrievers is provided in Appendix B.5.
Neutral Pool Sufficiency.Notably, as neutral competition
increases, the low-RPS mass becomes more pronounced for
standard retrievers, raising the question of which pool size
Nyields a stable audit. We analyze the fraction of entities
with RPSk>0.5 under increasing Nat our maximum
budget k= 50 (Figure 3). The setting of k= 50 is the
most challenging for controlling randomness, as the chance
baseline scales with k/N . At small N, the chance hit rate
is high ( k/N large), so observed success can be inflated by
random collisions and is not diagnostic of true retrievability;
however, beyond N≈400 , the curves decouple and plateau,
indicating stability. Accordingly, we adopt N= 800 as a
conservative setting to ensure that measured blind spots
reflect predictable geometric failures suitable for diagnosis
and remedy.
4

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
Table 1.Predicting RPS from embedding geometry.Reporting the best diagnostic probes that are selected based on the lowest RMSE,
we observe high correlation (Pearson r≈0.65 -0.80) and classification accuracy ( ≈0.65 -0.80). This confirms that blind spots are
geometrically encoded and detectable prior to indexing. (See Appendix C for full results of different model configurations.)
Retriever ArchitectureRegression Metrics Semi-Classification Metrics
RMSE(↓)MAE(↓)Pearsonr(↑)Spearmanρ(↑)Macro-F1(↑)Macro-Rec.(↑)Macro-Prec.(↑)Prec weighted (↑)F1 weighted (↑)Accuracy(↑)
BGE-M3XGBoost 0.168 0.118 0.681 0.644 0.658 0.540 0.573 0.781 0.762 0.781
CONTRIEVERXGBoost 0.157 0.109 0.658 0.622 0.727 0.501 0.506 0.795 0.777 0.795
QWEN3-EMBEDDINGXGBoost 0.153 0.111 0.781 0.721 0.699 0.619 0.646 0.764 0.760 0.764
NV-EMBEDXGBoost 0.173 0.124 0.640 0.595 0.657 0.517 0.541 0.757 0.740 0.757
REASON-EMBEDXGBoost 0.156 0.114 0.764 0.742 0.688 0.595 0.609 0.752 0.748 0.752
GRITLM-7B XGBoost 0.157 0.115 0.745 0.677 0.682 0.595 0.620 0.762 0.754 0.762
JINA-V3Ridge 0.178 0.137 0.667 0.659 0.641 0.557 0.574 0.655 0.653 0.655
REASONIR-8BRidge 0.156 0.121 0.779 0.788 0.710 0.658 0.674 0.674 0.674 0.674
4. Detecting Blind Spots from Embedding
Geometry
Since blind spots occupy distinct geometric regions (Sec-
tion 3), we hypothesize that retrievability risk is an intrinsic
property that can be estimated directly from an entity’s rep-
resentation, without expensive retrieval simulations.
4.1. Diagnostic Probes for RPS Prediction
We formulate blind-spot detection as a supervised regres-
sion task, learning a diagnostic function that maps an
entity embedding exto a predicted retrievability score
dRPSk(x|w x)∈[0,1] . Since xis represented in its canon-
ical Wikipedia context wx(Section 3.2), this task corre-
sponds to predicting the retrievability score directly from the
embedding ex. Because embedding geometries vary across
architectures, we train a separate probe for each retriever,
exploring three model families: linear Ridge Regression,
non-linear tree-based models (XGBoost), and Multi-Layer
Perceptrons (MLP).
Training and Selection.Probes are trained on entity em-
beddings labeled with empirical RPSk(x|w x)computed
withN=800 neutrals for a fixed retrieval budget k. We
employ a standard train/validation/test split and select hy-
perparameters by minimizing RMSE on the validation set.
We evaluate a comprehensive sweep of probe configurations
(detailed in Appendix C.1); for clarity, we report only the
best-performing probe for each retriever in Table 1. The
resulting predictor enables a pre-index audit; by thresh-
olding dRPSk(x|w x)< τ , we can flag high-risk entities
solely from their vector representations. While τcan be set
based on application sensitivity (higher for more sensitive
applications), throughout this paper we use a fixed global
threshold τ=0.3 (slightly below the lowest-tercile cutoff, ≈
0.33) across all experiments.
4.2. Regression and Semi-Classification Performance
Table 1 reports the best-performing probe for each neural
retriever, demonstrating strong agreement with empirical
RPSk(x|w x)(e.g., Pearson r≈0.65 -0.80) and low pre-
diction error under the regression objective. To translatethese scores into actionable risk categories, we also evaluate
asemi-classificationview by discretizing entities into three
RPSk(x|w x)bands:low( [0,0.33) ),mid( [0.33,0.66) ),
andhigh( [0.66,1] ), and measuring how well probes re-
cover these categories. Across retrievers, probes achieve
strong performance on this task (accuracy ≈0.65 -0.80 with
correspondingly high macro-F1), indicating that they reli-
ably separate low- RPSk(x|w x)entities from partially and
highly retrievable ones. We further assess calibration to en-
sure predictions are not systematically biased; Appendix C.2
(Figure 7) shows predicted-empirical RPSk(x|w x)densi-
ties and residuals that are well-centered around zero across
retrievers, with density skew largely reflecting the natural
imbalance of RPSk(x|w x)in standard models. Overall,
these results establish that blind-spot risk is detectable pre-
index, enabling threshold-based flagging as the first stage
of ARGUS.
5.ARGUS: Remedying Retriever Blind Spots
Building on Section 4, where we showed that RPS is pre-
dictable from embedding geometry via lightweight probes,
we now introduce ARGUS, an offline, a pre-index time
intervention for remedying retriever blind spots in IR/RAG
corpora, requiring neither query rewriting nor expensive re-
trieval evaluation over the target corpus.. ARGUS proceeds
in two stages:Diagnosisflags high-risk named entities in
each document using predicted dRPSk, andRemedyinjects
external defining context to construct augmented document
views for flagged entities and enhance their retrievability
(Figure 4)
5.1. Diagnosis: Pre-Index Risk Estimation
Given a corpus Dof documents, our goal is to identify high-
risknamed-entitiesthat are likely to be blind spots for a
target neural retriever under a fixed top- kbudget. We use
named entities to denote an NER-extracted surface-form
span in a document d∈ D . Concretely, we first run an
off-the-shelf NER tagger to extract entity spans sand their
corresponding strings m(details in Appendix D.1). For
each extracted entity mention mwith span sin document
d, we compute a context-dependent embedding using the
5

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
B. Augmentation StageA. Diagnosis: Identifying Blind Spots 
Named Entity Recognition Model   ……. Neanderthal ….
… Upper Palaeolithic ….
………….. British ………
…. Paul Pettitt ………….
…………… Trinkaus ….
Target Retriever Encoder   … Based on 45 Neanderthal long bones from 14 men and 7 women, the average height was 164 to 168 cm (5 ft 5 in to 5 ft 6 in) for males and 152 to 156 cm (5 ft 0 in to 5 ft 1 in) for females. For comparison, the average height of 20 males and 10 females Upper Palaeolithic humans is, respectively, 176.2 cm (5 ft 9.4 in) and 162.9 … Indicated from various ailments resulting from high stress at a low age, such as stunted growth, British archaeologist Paul Pettitt hypothesized that children of both sexes were put to work directly after weaning; and Trinkaus said that, upon reaching adolescence, an individual may have been expected to join in hunting large and dangerous game. …Documents Before Indexing
1. Extract Named Entities and encode them using the target retriever (e.g., ReasonIR).
2. Predict (probability of Top- retrieval). Flag high-risk entities where .RPSkkRPSk<τ
3. Retrieve deﬁning context for ﬂagged entities from Reference KB (e.g., ﬁrst paragraphs of Wikipedia).Reference KB
B.2. LLM Synthesis… Based on 45 Neanderthal long bones from 14 men and 7 women, the average height was 164 to 168 cm (5 ft 5 …   … Based on 45 Neanderthal long bones from 14 men and 7 women, the average height was 164 to 168 cm (5 ft 5 in to 5 ft 6 in) for males and 152 to 156 cm (5 ft 0 in to 5 ft 1 in) for females. For comparison, the average height of 20 males and 10 females Upper Palaeolithic humans is, respectively, 176.2 cm (5 ft 9.4 in) and 162.9 … Indicated from various ailments resulting from high stress at a low age, such as stunted growth, British archaeologist Paul Pettitt (Palaeolithic dating, Neanderthal art, burials) hypothesized that children of both sexes were put to work directly after weaning; and Trinkaus (American paleoanthropologist; NAS member) said that, upon reaching adolescence, an individual may have been expected to join in hunting large and dangerous game. …LLMB.1. Document Expansion
Concatenation… Based on 45 Neanderthal l o n g b o n e s f r o m 1 4 …             Paul Barry Pettitt,  FSA  is a British archaeologist …… Based on 45 Neanderthal long bones from 14 men …       Erik Trinkaus ( born December 24, 1948) is an …
Target Retriever Encoder
Target Retriever Encoder  Erik Trinkaus ( born December 24, 1948) is an American paleoanthropologist. … : 0.58 
✅RPSK : 0.57 
✅RPSK : 0.66 
✅RPSK : 0.24 
❌RPSK : 0.29 
❌RPSK Threshold: 0.30RPSKRetrieval Probability Score-k ModelPaul PettittTrinkaus
  Paul Barry Pettitt,  FSA  is a British archaeologist and academic. …  Erik Trinkaus ( born December 24, 1948) is an American paleoanthropologist. …… Based on 45 Neanderthal long bones from 14 men and 7 women, the average height was 164 to 168 cm (5 ft 5 …  Paul Barry Pettitt,  FSA  is a British archaeologist and academic. …  Erik Trinkaus ( born December 24, 1948) is an American paleoanthropologist. …Top- = 2kAug…
…ISBN 9780471663577.Trinkaus, Erik Paul Barry Pettitt,  FSA  is a British archaeologist and academic. …
Figure 4.The ARGUS Pipeline: Diagnosis and Remedy of Geometric Blind Spots.(A) Diagnosis: The system first extracts named
entities and predicts their retrievability ( RPS k) using the target retriever. Entities falling below the safety threshold ( RPS < τ ) are
flagged as blind spots (high-risk) located in inaccessible regions of the embedding space. (B) Augmentation: To remedy these blind spots,
ARGUS retrieves defining context from a Reference KB. We employ two strategies, (B.1) Document Expansion (Concatenation) or (B.2)
LLM Synthesis, to generate augmented document views. By indexing these views alongside the original, we enable the retrievability of
previously unknown entities.
target retriever encoder and the same span pooling operator
as in Section 3.2: em,d=g(E θ(d), s) . We then apply the
retriever-specific diagnostic model (Section 4.1), to estimate
contextual retrievability ofm: dRPSk(m|d).
If an entity appears multiple times in d, we score each
occurrence and assign the entity the minimum predicted
score across named entities (risk-conservative); we then
augment each flagged entity. We label mas high-risk when
dRPSk(m|d)< τ (default τ= 0.3 ), and output the set of
flagged entities
Erisk(d) ={m: dRPSk(m|d)< τ}.
By this diagnosis stage is fully offline and lightweight; it
requires only NER, document encoding, and probe inference
(See Figure 4, Diagnosis), and now, we can go over the
remedy.
5.2. Remedy: Targeted Knowledge Augmentation
Note: while the Wikipedia first paragraphs in Section 3 were
used for auditing via label-grounded Wikidata alignment,
here they serve as a retrievable reference KB for augmenta-
tion.
Reference KB retrieval.Given the diagnosed set of high-
risk entities Erisk(D), ARGUS injects a concise defining
context at indexing time. For each flagged entity mention
m∈ E risk(D), we query the Reference KB (Wikipedia first
paragraphs) using the surface form of mand retrieve the
topkAugpassages with a fast lexical retriever (BM25S) (L `u,2024). We set kAug=2, which provides sufficient disam-
biguating context to anchor the entity while keeping the
augmentation lightweight.
We instantiate two augmentation strategies that trade off
index growth against computation.
1. Document Expansion by Concatenationcreates one
augmented view per retrieved KB passage pby appending it
to the original document: dexp
m,p=d∥p. LetNd=|E risk(d)|
denote the number of flagged entities in d. This yields 1 +
kAug·Ndindexed views (the original dplus one expansion
per(m, p)pair).
2. KB-guided LLM Synthesisinstead aggregates all re-
trieved KB passages for all flagged entities in dand prompts
an LLM with (d,{p}) to produce a single unified augmented
document dsynththat inserts short, entity-focused clarifica-
tions only where necessary after the entity surface min
the parentheses (prompting details in Appendix D.3). This
option indexes exactly two views per document: the origi-
naldanddsynth, reducing index growth at the cost of LLM
computation.
In both cases, we index augmented views along to the orig-
inal documents; we never replace them, hence, the corpus
semantics are preserved while adding retrievable “support
views” for previously high-risk entities (Figure 4,B).
ARGUS converts pre-index blind spot risk estimates into
targeted pre-indexing time augmentations that improve
downstream retrieval by remedying entity-level blind spots.
In the next section, we evaluate this pipeline across multi-
ple retrievers and benchmarks to quantify its impact under
6

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
diverse retrieval settings.
6. Experimental Setup
We evaluate whether ARGUS turns pre-index risk detection
into downstream retrieval gains across multiple benchmarks
and dense neural retrievers.
Evaluation Benchmarks. Benchmarks:We evaluate on
BRIGHT (SU et al., 2025), IMPLIRET(Taghavi et al.,
2025), and RAR-B(Xiao et al., 2024), covering multiple
tasks/settings per benchmark (BRIGHT: 10 domains, IM-
PLIRET: 2 settings of “Word Knowledge” category, RAR-B:
7 subsets).Metrics:We report nDCG@5 and nDCG@10
in Table 2; additional cutoffs (e.g., nDCG@20/50) are pro-
vided in Appendix F.
Retrievers and Baselines.We test eight retrievers:
BGE-M3, CONTRIEVER, QWEN3-EMBEDDING, NV-
EMBED-V2, REASON-EMBED, GRITLM-7B, JINA-V3,
REASONIR-8B (Chen et al., 2024; Izacard et al., 2022;
Zhang et al., 2025; Lee et al., 2025; Chen et al., 2025;
Muennighoff et al., 2025; Sturua et al., 2024; Shao et al.,
2025). We compare Original indexing against ARGUS with
Document ExpansionorLLM Synthesis. We enforce a strict
retriever-consistent pipeline; when evaluating a model, that
retriever is used for embedding, risk diagnosis, and final
ranking. The only exception is the internal lookup over
the Reference KB, which we perform with BM25S as a
lightweight, fast baseline (L `u, 2024).
Implementation Details. ARGUS:We extract candi-
date named entities using dslim/bert-base-NER and
set the risk threshold to τ= 0.3 ondRPSk. For each
flagged entity m, we retrieve the top kAug= 2 passages
from a Reference KB consisting of Wikipedia first para-
graphs using BM25S(L `u, 2024). We query using the en-
tity surface form and use the retrieved passages to con-
struct augmented document views. We index these views
alongside the original documents and run retrieval over the
expanded index using the same target retriever and scor-
ing/ranking procedure as Original.LLM synthesis:We use
QWEN3 ( Qwen/Qwen3-30B-Instruct-2507 (Team,
2025)) to generate one synthesized view per document (Ap-
pendix D.3).Probe:For each retriever (and retrieval budget
k), we use the best-performing probe from Section 4, se-
lected by validation RMSE.
7. Experimental Results
We evaluate ARGUS across eight neural retrievers on
three benchmarks. Overall, targeted pre-index augmen-
tation yields broad improvements in retrieval quality
(nDCG@5/10) across diverse architectures and benchmarks.7.1. End-to-End Retrieval Improvements
We evaluate ARGUS, viaDocument ExpansionandLLM
Synthesis, against the Original baseline across eight neu-
ral retrievers on BRIGHT IMPLIRET, and RAR-B. To
summarize, beyond the subset shown for readability, we
report Full Benchmark Avg. over each benchmark’s eval-
uated task suite. These gains persist under the aggregate
metric; averaged over nDCG@5 and nDCG@10, Docu-
ment Expansion improves the Full Benchmark Avg. over
all retrievers by+3.44on BRIGHT,+6.76on IMPLIRET,
and+1.68on RAR-B, while LLM Synthesis yields+2.44,
+2.21, and+1.81on the same benchmarks, respectively.
Per-task results appear in Appendix F. As an auxiliary di-
agnostic, Appendix E.1 compares the maximum entity RPS
in retrieved vs. unretrieved gold documents, suggesting an
association between entity-level retrievability and retrieval
outcomes in some subsets. Taken together, these results
show that targeted index-time augmentation guided by pre-
dicted blind-spot risk translates into tangible end-to-end
retrieval gains under practical top-ksettings.
7.2. Document Expansion vs. LLM Synthesis
The two ARGUS strategies expose a practical trade-off
between retrieval stability and index efficiency, while shar-
ing the core benefit of preserving the original corpus (aug-
mented views are indexed alongside original document D).
Stability vs. Efficiency.Document Expansion is the most
consistent intervention in our experiments by appending
retrieved KB contexts, it improves the Full Benchmark Avg.
in most configurations, with particularly strong suite-level
gains on IMPLIRET(+6.76averaged over nDCG@5/10)
and solid improvements on BRIGHT (+3.44) and RAR-B
(+1.68). In contrast, LLM Synthesis adds only one addi-
tional view per document (rather than growing with the
number of flagged entities) while achieving competitive
suite-level improvements, especially on BRIGHT (+2.44)
and RAR-B(+1.81). Notably, it can outperform expansion
on RAR-Bfor standard retrievers such as CONTRIEVER
(nDCG@10: 29.2 vs. 27.9), suggesting that a coherent syn-
thesized view may sometimes align better with query se-
mantics than raw concatenation.
Retriever Sensitivity.Synthesis also exhibits higher vari-
ance, consistent with some retrievers being more sensitive
to the structure and style of generated augmentations. For
example, on IMPLIRET, JINA-V3 degrades with synthesis
(nDCG@10: 17.9 →16.4) but improves with expansion
(20.8). Since both strategies augment the same flagged enti-
ties using the same retrieved KB evidence, this divergence
suggests that augmentationformcan interact with retriever
behavior. Overall, ARGUS supports flexible deployment:
LLM Synthesis for constrained index budgets, or Document
Expansion for maximum stability and overall performance.
7

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
Table 2.Downstream retrieval performance (nDCG@5/10) on BRIGHT, IMPLIRET, and RAR-B.We compare standard baselines
against ARGUS remedies, Document Expansion, and LLM Synthesis, across eight neural retrievers. The Full Benchmark Avg. columns
report mean performance over the complete task suite for each benchmark (e.g., 10 of the BRIGHT domains), rather than only the
representative subsets shown. ARGUS yields robust gains across retrievers, supporting the efficacy of remedying geometric blind spots
by targeted augmentations. (See Appendix F for the full per-task breakdown.)Colors:Best Result, 2nd Best, 3rd Best.
Retriever AugmentationBright Impliret Rar-b
Shown (3/10) Full Benchmark Avg. Shown (2/2) Full Benchmark Avg. Shown (3/7) Full Benchmark Avg.
Biology Econ SustAvg. (All 10 Tasks)Multi UniAvg. (All 2 Tasks)ARC PIQA HellaAvg. (All 7 Tasks)
BGE-M3Baseline 7.8/9.5 10.0/11.7 9.0/10.1 10.2/11.0 17.9/23.3 13.4/18.8 15.7/21.1 7.8/9.0 20.9/22.9 23.3/25.5 17.1/19.5
ARGUS (Doc-Exp.) 8.6/11.413.1/17.610.4/12.8 12.5/15.9 30.2/38.3 26.7/34.0 28.4/36.17.8/9.221.9/24.724.0/26.917.4/20.0
ARGUS (LLM-Synth.)13.6/14.712.6/13.912.1/13.1 14.3/15.3 22.5/27.5 16.5/21.5 19.5/24.58.5/9.821.8/23.824.8/26.818.2/20.0
CONTRIEVERBaseline 7.2/9.2 10.7/10.5 7.1/8.9 9.0/9.8 12.8/18.3 10.4/15.2 11.6/16.8 7.4/8.6 23.1/25.1 24.1/26.4 21.9/23.7
ARGUS (Doc-Exp.) 8.6/13.0 13.7/16.4 8.1/11.9 11.6/14.918.0/24.916.9/22.817.4/23.97.4/9.0 24.1/26.8 24.3/26.8 24.4/27.9
ARGUS (LLM-Synth.)8.8/11.5 10.1/11.2 7.1/8.2 10.2/11.818.4/24.317.0/22.417.7/23.37.7/9.7 27.2/30.6 28.5/32.8 25.6/29.2
QWEN3-EMBEDDINGBaseline 10.5/12.4 11.7/12.7 9.2/10.2 10.0/10.8 8.0/10.8 3.9/5.3 5.9/8.0 7.4/8.8 17.1/19.4 21.8/24.0 14.5/16.2
ARGUS (Doc-Exp.)18.5/27.1 15.2/16.29.8/13.4 13.2/16.6 10.4/15.1 5.5/7.3 8.0/11.27.9/9.3 18.7/22.126.5/31.915.8/18.5
ARGUS (LLM-Synth.) 9.1/13.7 14.2/15.210.7/11.7 12.1/13.7 8.4/12.0 5.0/5.4 6.7/8.77.9/9.5 19.5/22.325.4/28.716.5/18.4
NV-EMBED-V2Baseline 14.0/16.5 12.2/13.2 9.7/10.7 11.9/13.1 33.9/38.5 24.3/29.2 29.1/33.9 14.3/16.2 34.8/37.6 33.7/36.2 21.4/24.3
ARGUS (Doc-Exp.)17.7/19.2 15.2/16.7 11.7/12.7 15.5/16.7 50.2/53.2 40.2/43.2 45.2/48.2 16.2/17.238.2/39.2 36.2/37.224.2/25.5
ARGUS (LLM-Synth.) 15.7/17.7 13.2/14.7 10.2/11.7 13.6/15.3 38.2/42.2 28.2/32.2 33.2/37.2 15.2/17.536.0/39.0 35.0/37.522.6/25.6
REASON-EMBEDBaseline 14.5/18.6 10.2/11.2 10.5/12.4 11.1/12.6 7.8/11.0 1.8/2.4 4.8/6.7 7.9/9.2 12.4/14.1 20.8/22.9 12.9/14.3
ARGUS (Doc-Exp.) 15.9/22.4 10.4/11.5 11.2/13.413.8/16.4 8.2/12.22.0/2.8 5.1/7.58.1/9.6 13.0/15.0 23.2/26.7 14.3/16.1
ARGUS (LLM-Synth.)17.5/22.8 12.0/13.7 12.7/14.013.5/15.4 8.0/11.53.0/3.4 5.5/7.48.2/10.1 13.5/15.6 24.8/28.2 14.4/16.2
GRITLM-7BBaseline 5.9/7.0 4.1/4.4 4.1/4.8 5.7/6.4 5.6/7.3 15.2/16.7 10.4/12.0 3.1/3.9 3.3/3.9 16.5/18.3 10.3/11.5
ARGUS (Doc-Exp.) 6.7/9.3 4.8/5.0 5.2/6.9 7.1/8.46.1/8.6 22.2/24.2 14.2/16.43.2/4.0 3.4/4.019.0/22.1 11.4/12.8
ARGUS (LLM-Synth.)9.5/10.2 7.2/7.8 6.2/7.0 8.2/9.05.9/8.0 16.5/18.0 11.2/13.03.6/4.2 3.9/4.518.5/20.5 11.4/13.0
JINA-V3Baseline 12.0/15.2 18.6/19.2 13.1/15.8 14.2/16.1 14.8/20.6 10.9/15.2 12.9/17.9 11.1/13.2 26.5/29.1 24.6/27.0 15.9/17.8
ARGUS (Doc-Exp.) 12.8/16.519.9/21.614.3/18.2 15.7/18.5 17.5/24.2 12.2/17.3 14.8/20.811.3/13.6 27.5/30.9 27.4/31.416.7/19.2
ARGUS (LLM-Synth.)13.4/16.220.2/21.214.7/17.0 15.7/17.4 16.2/19.7 10.7/13.2 13.4/16.412.0/14.2 27.5/30.0 26.0/28.516.9/18.9
REASONIR-8BBaseline 16.6/19.1 13.9/16.5 10.2/11.4 13.6/15.0 22.7/27.7 8.1/10.5 15.4/19.1 12.2/13.5 23.3/25.9 30.9/33.2 18.6/20.1
ARGUS (Doc-Exp.) 18.0/24.416.6/22.2 12.2/13.8 17.3/21.4 30.8/42.0 12.5/19.2 21.7/30.612.2/13.526.2/27.3 34.3/35.8 20.8/21.9
ARGUS (LLM-Synth.)20.0/25.212.7/16.8 11.3/12.8 15.8/18.3 25.8/32.0 8.7/11.3 17.3/21.712.9/14.525.7/26.8 33.8/35.3 20.4/21.6
7.3. Key Takeaways
Across our evaluation, Table 2 shows that ARGUS im-
proves end-to-end retrieval quality (nDCG@5/10) relative
toOriginal. Comparing remedies,Document Expansion
is the most reliable and best-performing option in aggre-
gate (Full Benchmark Avg. gains of+3.44on BRIGHT,
+6.76on IMPLIRET, and+1.68on RAR-B, averaged over
nDCG@5/10), whileLLM Synthesisoffers a more space-
efficient alternative (+2.44,+2.21, and+1.81). Overall,
these results establish ARGUS as an effective index-time
remedy for retriever blind spots without retriever retraining
or query rewriting.
8. Discussion and Limitations
Our experiments use a single, uniform ARGUS configu-
ration to demonstrate cross-benchmark and cross-retriever
generality, including a fixed reference KB and a shared
LLM-synthesis prompt across tasks. While this already
yields consistent gains, it may be suboptimal for specific
domains, where a more specialized reference KB, retrieval
strategy, or synthesis prompt could better match task and ev-
idence needs. More broadly, while increasing retrieval qual-
ity, the optimal augmentation form (expansion vs. synthesis)
and thresholding choices may depend on the retriever and
corpus characteristics, suggesting opportunities for adaptive
retriever- and task-aware configurations. We leave such cus-
tomization as an open direction and hope it motivates further
work on task-aware remedies for retriever blind spots.9. Conclusion
We studied a key trustworthiness bottleneck in RAG neu-
ral retrieval, asking whether retrieval failures are mostly
sporadic errors or instead reflectintrinsic blind spots: sys-
tematic failures to retrieve entities under practical top- k
budgets. To quantify entity-level retrievability, we intro-
duced RPS and a large-scale, domain-agnostic auditing pro-
tocol based on Wikidata-Wikipedia alignment. Our analysis
showed that blind spots are structural rather than random:
low- and high-RPS entities occupy distinguishable regions
in embedding space, even for strong reasoning-oriented re-
trievers. Exploiting this geometric signal, we further showed
that lightweight, retriever-specific probes can predict RPS
directly from entity embeddings, enabling pre-index risk
estimation without expensive retrieval evaluations. Building
on these insights, we proposed ARGUS, an indexing-time
pipeline that diagnoses high-risk named entities in docu-
ments and remedies them by injecting supporting context
from a reference KB via either Document Expansion or KB-
guided LLM Synthesis. Across BRIGHT, IMPLIRET, and
RAR-Band a broad set of neural retrievers, ARGUS consis-
tently improves downstream retrieval quality (measured by
nDCG@5/10) without retriever fine-tuning or query rewrit-
ing. More broadly, our results suggest that auditing, de-
tecting, and remedying blind spots at indexing time is a
practical path toward more reliable retrieval for RAG, and
we hope this motivates future work on adaptive task- and
retriever-aware remedies.
8

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
Impact Statement
This paper presents work whose goal is to advance the field
of Information Retrieval and enhance the robustness and
trustworthiness of neural retrieval systems. By analyzing the
retrievability of entities and identifying systematic ”blind
spots” in embedding geometry, our work provides tools
for pre-index auditing to prevent failures in RAG pipelines.
There are many potential societal consequences of our work,
none of which we feel must be specifically highlighted here.
Acknowledgment
We gratefully acknowledge support from the German
Research Foundation (Deutsche Forschungsgemeinschaft,
DFG), grant SCHU 2246/14-1.
References
Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu,
X., Majumder, R., McNamara, A., Mitra, B., Nguyen,
T., Rosenberg, M., Song, X., Stoica, A., Tiwary, S.,
and Wang, T. MS MARCO: A human generated ma-
chine reading comprehension dataset. InProceedings
of the Workshop on Cognitive Computation: Integrating
Neural and Symbolic Approaches, 2016. URL https:
//arxiv.org/abs/1611.09268.
Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D.,
and Liu, Z. M3-embedding: Multi-linguality, multi-
functionality, multi-granularity text embeddings through
self-knowledge distillation. In Ku, L.-W., Martins, A.,
and Srikumar, V . (eds.),Findings of the Association
for Computational Linguistics: ACL 2024, pp. 2318–
2335, Bangkok, Thailand, August 2024. Association
for Computational Linguistics. doi: 10.18653/v1/2024.
findings-acl.137. URL https://aclanthology.
org/2024.findings-acl.137/.
Chen, J., Lan, J., Li, C., Lian, D., and Liu, Z. Reasonem-
bed: Enhanced text embeddings for reasoning-intensive
document retrieval.arXiv preprint arXiv:2510.08252,
2025.
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y ., Dai, Y .,
Sun, J., Wang, H., and Wang, H. Retrieval-augmented
generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2(1), 2023.
Guu, K., Lee, K., Tung, Z., Pasupat, P., and Chang, M.
Retrieval augmented language model pre-training. In
International conference on machine learning, pp. 3929–
3938. PMLR, 2020.
Hong, K., Troynikov, A., and Huber, J. Context rot:
How increasing input tokens impacts llm performance.Technical report, Chroma, July 2025. URL https:
//research.trychroma.com/context-rot.
Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski,
P., Joulin, A., and Grave, E. Unsupervised dense in-
formation retrieval with contrastive learning.Transac-
tions on Machine Learning Research, 2022. ISSN 2835-
8856. URL https://openreview.net/forum?
id=jKN1pXi7b0.
J¨arvelin, K. and Kek ¨al¨ainen, J. Cumulated gain-based evalu-
ation of ir techniques.ACM Transactions on Information
Systems (TOIS), 20(4):422–446, 2002.
Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y ., Ishii, E.,
Bang, Y . J., Madotto, A., and Fung, P. Survey of halluci-
nation in natural language generation.ACM computing
surveys, 55(12):1–38, 2023.
Karpukhin, V ., Oguz, B., Min, S., Lewis, P., Wu, L.,
Edunov, S., Chen, D., and Yih, W.-t. Dense passage
retrieval for open-domain question answering. In Web-
ber, B., Cohn, T., He, Y ., and Liu, Y . (eds.),Proceed-
ings of the 2020 Conference on Empirical Methods
in Natural Language Processing (EMNLP), pp. 6769–
6781, Online, November 2020. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2020.emnlp-main.
550. URL https://aclanthology.org/2020.
emnlp-main.550/.
Lee, C., Roy, R., Xu, M., Raiman, J., Shoeybi, M., Catan-
zaro, B., and Ping, W. NV-embed: Improved techniques
for training LLMs as generalist embedding models. In
The Thirteenth International Conference on Learning
Representations, 2025. URL https://openreview.
net/forum?id=lgsyLSsDRe.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., Riedel, S., and Kiela, D. Retrieval-augmented gener-
ation for knowledge-intensive nlp tasks. In Larochelle,
H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H.
(eds.),Advances in Neural Information Processing Sys-
tems, volume 33, pp. 9459–9474. Curran Associates, Inc.,
2020.
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua,
M., Petroni, F., and Liang, P. Lost in the middle: How
language models use long contexts.Transactions of
the Association for Computational Linguistics, 12:157–
173, 2024. doi: 10.1162/tacl a00638. URL https:
//aclanthology.org/2024.tacl-1.9/.
L`u, X. H. Bm25s: Orders of magnitude faster lexical
search via eager sparse scoring, 2024.arXiv preprint
arXiv:2407.03618, 2024.
9

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
Ma, X., Gong, Y ., He, P., Zhao, H., and Duan, N. Query
rewriting in retrieval-augmented large language mod-
els. In Bouamor, H., Pino, J., and Bali, K. (eds.),Pro-
ceedings of the 2023 Conference on Empirical Meth-
ods in Natural Language Processing, pp. 5303–5315,
Singapore, December 2023. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2023.emnlp-main.
322. URL https://aclanthology.org/2023.
emnlp-main.322/.
Mallen, A., Asai, A., Zhong, V ., Das, R., Khashabi, D.,
and Hajishirzi, H. When not to trust language mod-
els: Investigating effectiveness of parametric and non-
parametric memories. In Rogers, A., Boyd-Graber, J., and
Okazaki, N. (eds.),Proceedings of the 61st Annual Meet-
ing of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pp. 9802–9822, Toronto, Canada,
July 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.acl-long.546. URL https:
//aclanthology.org/2023.acl-long.546/.
Modarressi, A., Deilamsalehy, H., Dernoncourt, F., Bui, T.,
Rossi, R. A., Yoon, S., and Schuetze, H. Nolima: Long-
context evaluation beyond literal matching. InForty-
second International Conference on Machine Learning,
2025. URL https://openreview.net/forum?
id=0OshX1hiSa.
Muennighoff, N., Tazi, N., Magne, L., and Reimers, N.
MTEB: Massive text embedding benchmark. In Vla-
chos, A. and Augenstein, I. (eds.),Proceedings of the
17th Conference of the European Chapter of the Asso-
ciation for Computational Linguistics, pp. 2014–2037,
Dubrovnik, Croatia, May 2023. Association for Compu-
tational Linguistics. doi: 10.18653/v1/2023.eacl-main.
148. URL https://aclanthology.org/2023.
eacl-main.148/.
Muennighoff, N., SU, H., Wang, L., Yang, N., Wei, F.,
Yu, T., Singh, A., and Kiela, D. Generative repre-
sentational instruction tuning. InThe Thirteenth In-
ternational Conference on Learning Representations,
2025. URL https://openreview.net/forum?
id=BC4lIvfSzv.
Nogueira, R. and Cho, K. Passage re-ranking with BERT.
arXiv preprint arXiv:1901.04085, 2019.
Robertson, S. and Zaragoza, H. The probabilistic relevance
framework: Bm25 and beyond.Found. Trends Inf. Retr., 3
(4):333–389, April 2009. ISSN 1554-0669. doi: 10.1561/
1500000019. URL https://doi.org/10.1561/
1500000019.
Santhanam, K., Khattab, O., Saad-Falcon, J., Potts, C.,
and Zaharia, M. ColBERTv2: Effective and efficient
retrieval via lightweight late interaction. In Carpuat, M.,de Marneffe, M.-C., and Meza Ruiz, I. V . (eds.),Pro-
ceedings of the 2022 Conference of the North American
Chapter of the Association for Computational Linguis-
tics: Human Language Technologies, pp. 3715–3734,
Seattle, United States, July 2022. Association for Compu-
tational Linguistics. doi: 10.18653/v1/2022.naacl-main.
272. URL https://aclanthology.org/2022.
naacl-main.272/.
Shao, R., Qiao, R., Kishore, V ., Muennighoff, N., Lin, X. V .,
Rus, D., Low, B. K. H., Min, S., tau Yih, W., Koh, P. W.,
and Zettlemoyer, L. Reasonir: Training retrievers for
reasoning tasks.arXiv preprint arXiv:2504.20595, 2025.
Shuster, K., Poff, S., Chen, M., Kiela, D., and We-
ston, J. Retrieval augmentation reduces hallucina-
tion in conversation. In Moens, M.-F., Huang, X.,
Specia, L., and Yih, S. W.-t. (eds.),Findings of the
Association for Computational Linguistics: EMNLP
2021, pp. 3784–3803, Punta Cana, Dominican Repub-
lic, November 2021. Association for Computational
Linguistics. doi: 10.18653/v1/2021.findings-emnlp.
320. URL https://aclanthology.org/2021.
findings-emnlp.320/.
Sturua, S., Mohr, I., Akram, M. K., G ¨unther, M., Wang, B.,
Krimmel, M., Wang, F., Mastrapas, G., Koukounas, A.,
Koukounas, A., Wang, N., and Xiao, H. jina-embeddings-
v3: Multilingual embeddings with task lora.arXiv
preprint arXiv:2409.10173, 2024.
SU, H., Yen, H., Xia, M., Shi, W., Muennighoff, N.,
yu Wang, H., Haisu, L., Shi, Q., Siegel, Z. S., Tang,
M., Sun, R., Yoon, J., Arik, S. O., Chen, D., and Yu,
T. BRIGHT: A realistic and challenging benchmark
for reasoning-intensive retrieval. InThe Thirteenth In-
ternational Conference on Learning Representations,
2025. URL https://openreview.net/forum?
id=ykuc5q381b.
Taghavi, Z. S., Modarressi, A., Ma, Y ., and Schuetze, H.
ImpliRet: Benchmarking the implicit fact retrieval chal-
lenge. In Christodoulopoulos, C., Chakraborty, T., Rose,
C., and Peng, V . (eds.),Proceedings of the 2025 Confer-
ence on Empirical Methods in Natural Language Pro-
cessing, pp. 33168–33190, Suzhou, China, November
2025. Association for Computational Linguistics. ISBN
979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.
1685. URL https://aclanthology.org/2025.
emnlp-main.1685/.
Team, Q. Qwen3 technical report, 2025. URL https:
//arxiv.org/abs/2505.09388.
Thakur, N., Reimers, N., R ¨uckl´e, A., Srivastava, A., and
Gurevych, I. BEIR: A heterogeneous benchmark for
zero-shot evaluation of information retrieval models. In
10

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
Thirty-fifth Conference on Neural Information Process-
ing Systems Datasets and Benchmarks Track (Round 2),
2021. URL https://openreview.net/forum?
id=wCu6T5xFjeJ.
Thakur, N., Bonifacio, L., Fr ¨obe, M., Bondarenko, A., Ka-
malloo, E., Potthast, M., Hagen, M., and Lin, J. System-
atic evaluation of neural retrieval models on the touch ´e
2020 argument retrieval subset of beir. InProceedings
of the 47th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval, pp.
1420–1430, 2024.
Vrande ˇci´c, D. and Kr ¨otzsch, M. Wikidata: a free collabo-
rative knowledgebase.Communications of the ACM, 57
(10):78–85, 2014.
Wikimedia Foundation. MediaWiki Action API.
https://www.mediawiki.org/wiki/API:
Main_page, 2025. Accessed: December 2025.
Xiao, C., Hudson, G. T., and Moubayed, N. A. Rar-
b: Reasoning as retrieval benchmark.arXiv preprint
arXiv:2404.06347, 2024.
Xiong, L., Xiong, C., Li, Y ., Tang, K.-F., Liu, J., Ben-
nett, P. N., Ahmed, J., and Overwijk, A. Approximate
nearest neighbor negative contrastive learning for dense
text retrieval. InInternational Conference on Learning
Representations, 2021. URL https://openreview.
net/forum?id=zeFrfgyZln.
Yu, Y ., Xiong, C., Sun, S., Zhang, C., and Overwijk, A.
COCO-DR: Combating the distribution shift in zero-shot
dense retrieval with contrastive and distributionally ro-
bust learning. In Goldberg, Y ., Kozareva, Z., and Zhang,
Y . (eds.),Proceedings of the 2022 Conference on Em-
pirical Methods in Natural Language Processing, pp.
1462–1479, Abu Dhabi, United Arab Emirates, Decem-
ber 2022. Association for Computational Linguistics. doi:
10.18653/v1/2022.emnlp-main.95. URL https://
aclanthology.org/2022.emnlp-main.95/.
Zhang, Y ., Li, M., Long, D., Zhang, X., Lin, H., Yang, B.,
Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., and Zhou,
J. Qwen3 embedding: Advancing text embedding and
reranking through foundation models.arXiv preprint
arXiv:2506.05176, 2025.A. Appendix Overview
This appendix provides additional details and analyses that
support the main text. We follow the paper’s assess-detect-
remedy pipeline: Section 3 (Assess), Section 4 (Detect),
Section 5 (ARGUS), and Section 6-7 (Experiments/Results).
B. Additional Details for Assessing Retriever
Blind Spots
This appendix section provides implementation details for
the entity-centric audit protocol used to compute RPS and
characterize retriever blind spots (Section 3).
B.1. Entity Sampling and Filtering
Wikidata sampling.We begin from a large random sam-
ple of Wikidata items and retain those that are linkable to
an English Wikipedia page. We denote the filtered set of
retained entities byX.
Wikipedia linkage and basic validity checks.For each
candidate entity, we require: (i) a resolvable English
Wikipedia page, (ii) a non-empty first paragraph that can
be parsed as text, and (iii) a valid surface form (Wikidata
label) that can be matched in the paragraph after light nor-
malization. If the valid surface does not appear in the
Wikipedia first paragraph, we attach it to the beginning
of the Wikipedia first paragraph. We discard malformed
entries (e.g., missing pages, empty/very short paragraphs,
pages that fail parsing).
Surface-form grounding.To reduce noisy alignments,
we enforce that the entity’s Wikidata surface form appears
explicitly in the Wikipedia first paragraph wx. We apply
lightweight normalization for matching, including case-
folding and whitespace normalization (and, when applicable,
punctuation-stripping). If the label occurs multiple times,
we keep the earliest occurrence as the mention spans x.
Neighbor requirement.Finally, we require each retained
entity x∈X to have at least one valid related entity (Sec-
tion B.2) so that|T x|>0and RPS k(x)is well-defined.
B.2. Wikidata-Wikipedia Alignment and Query
Construction Details
1-hop related entities.For each target entity x∈X , we
construct the set of related entities Txfrom 1-hop Wikidata
neighbors (entities connected to xby any property). We
restrict Txto entities that (i) have an English Wikipedia page
and (ii) satisfy the same surface-form grounding constraint
as targets (their own label appears in their Wikipedia first
paragraph, if not, we will add it to the beginning of the text).
Why we embed queries using paragraphs.A related
entity’s label alone may be ambiguous (e.g., polysemous
11

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
names). To reduce ambiguity, we form the query repre-
sentation for each t∈ T xby encoding t’s Wikipedia first
paragraph wtand pooling at the mention span st(rather
than encoding the short label string in isolation). Concretely,
the query embedding for tisqt=g(E θ(wt), st), matching
the same representation format used for candidates (Sec-
tion 3.2).
Controlling query sets.We treat each t∈ T xas a proxy
query context in which xshould be retrievable. In practice,
Txcan vary in size across entities; RPS averages hits across
all available related entities for eachx(Section 3.2).
B.3. Neutral Pool Construction and Disjointness Checks
Motivation.RPS is designed to measure whether an entity
is retrieved due to genuine geometric alignment rather than
random collisions. We therefore evaluate each target entity
undercontrolled competitionagainst a neutral pool that is
(by construction) unrelated to the query entity.
Neutral candidate eligibility.Each neutral candidate z
must: (i) have an English Wikipedia page with a valid first
paragraphw z, (ii) satisfy surface-form grounding (its label
appears in wz; if not, we concatenate it to the beginning
of the first paragraph), and (iii) pass the KG-disjointness
constraint described below.
KG-disjointness constraint.For a related entity (query)
t, we define Nbr(t) as its set of 1-hop Wikidata neighbors
(entities directly connected to tby any property). We require
each neutral z∈ Z neut(t)to benot directly connectedto t
in the Wikidata graph, i.e.,
z /∈Nbr(t).
(equivalently, no 1-hop KG edge exists between zandt)
This constraint reduces the chance that a “neutral” candidate
is trivially related to tvia an explicit KG link, making the
neutral pool a stronger control.
Per-query neutral pools and sampling.We maintain a
related-entity-specific neutral pool Dneut(t)for each query
entity t. For each retrieval trial, we sample N−1 neutrals
uniformly from Dneut(t)and evaluate whether the target x
appears in the top- kamong the Ncandidates (Section 3.2).
Pool size.Unless otherwise stated, we use N= 800 neu-
trals for the audit, motivated by the stability analysis in Fig-
ure 3. We further analyze the sensitivity to the user-chosen
retrieval windowkat fixedN= 800in Figure 5.
B.4. RPS Implementation Details
Retriever-specific representations.Different dense retriev-
ers expose different embedding interfaces (e.g., sentence-
level vectors vs. token-level hidden states). We unify them
through a retriever-specific pooling function g(·)that returns
Figure 5.Sensitivity of retrieval consistency to the retrieval
window size (k) at fixedN= 800. Increasing the user-defined
parameterkexpands the retrieval scope. Standard retrievers (e.g.,
Contriever, BGE-M3) exhibit approximately linear growth consis-
tent with statistical scaling of the random-hit window. In contrast,
ReasonIR displays a non-linear trajectory with a mild elbow, in-
dicating that its gains are driven by learned geometric structure
rather than simple expansion of candidate slots.
a singleh-dimensional vector per entity instance.
Mention span extraction.Given a paragraph wand an
entity label, we identify a token span scorresponding to
the first grounded occurrence of the label in w(after the
normalization described in Appendix B.1). If the label
tokenizes into multiple subwords, scovers the full subword
span.
Pooling operator g(·).Let H=E θ(w)denote the rep-
resentation produced by the retriever encoder for w. We
use:
•Span pooling (token-level models).If Hprovides
token-level embeddings, we compute the entity em-
bedding by averaging token representations over the
mention span:
g(H, s) =1
|s|X
i∈sHi.
•Sentence-level models.If the retriever only returns
a single vector for the whole input, we set g(H, s) to
that vector (the span is ignored but the same paragraph-
level input is used for all entities).
This definition ensures that thesameprocedure is applied
to targets, queries (related entities), and neutrals: each is
represented by its Wikipedia first paragraph, with mention-
aware pooling when available.
Similarity and ranking.We rank candidates by cosine
similarity cos(q t,ec)and define Hit(x, t) using a top- k
cutoff (Section 3.2). For completeness, when retrievers
12

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
support optional embedding normalization, we follow the
retriever’s recommended inference-time practice and then
apply cosine similarity consistently across models.
B.5. Additional Geometry Visualizations
This subsection provides the complete set of 2D LDA pro-
jections for all evaluated retrievers, extending the repre-
sentative main-text visualization (Figure 2) in 6. For each
retriever, we compute LDA on entity embeddings after la-
beling entities into RPS terciles (low/mid/high), and we
visualize how separable regions evolve under increasing
neutral pool sizes N(with k= 50 ). These plots support the
central observation that blind spots correspond to structured
regions in representation space.
B.6. Sensitivity tokand Other Audit Hyperparameters
RPS is defined with respect to the user-selected retrieval
budget k, so it is important to understand how audit con-
clusions change as kvaries. At fixed neutral pool size
N= 800 , increasing kmechanically increases the probabil-
ity of a hit under random ranking (chance baseline ≈k/N ),
and correspondingly increases the fraction of entities with
RPSk>0.5 . In our experiments, standard retrievers tend to
exhibit near-linear scaling with k, consistent with expand-
ing the candidate window, whereas more robust retrievers
exhibit departures from purely linear behavior, indicating
that improvements are driven by learned geometric structure
rather than chance alone.
C.Additional Details for Detecting Blind Spots
C.1. Probe Families and Hyperparameter Sweep
This section provides additional implementation details for
the embedding-based diagnostic probes introduced in Sec-
tion 4. Our goal is to learn a retriever-specific predictor
hϕ:Rd→[0,1] that maps an entity embedding exto
dRPSk(x), enablingpre-indexrisk estimation directly from
representation geometry.
Input representation.For each retriever, we use the same
entity embedding construction as in Section 3.2: the input
vector ex=g(E θ(wx), sx)is a mention-pooled embedding
extracted from the target retriever encoder Eθover the entity
context wxwith span sx. Unless stated otherwise, probes
operate on the raw embedding vector; we do not require
query-conditioned features or retrieval simulations.
Probe families.We evaluate three probe families that
span linear, non-linear, and tree-based function classes: (i)
Linear probes(Ridge regression) as a strong, low-variance
baseline, (ii)MLP probesto capture non-linear structure in
embedding geometry with layers of dense neural networks,and (iii)Gradient-boosted trees(XGBOOST) as a flexible
non-linear model well-suited to tabular features. All probes
are trainedseparately per retriever, since embedding spaces
and pooling operators differ across architectures.
Training splits and objective.For each retriever, we cre-
ate a standard train/validation/test split over entities, and
train probes to regress to empirical RPSk(x)computed un-
der our stable audit setting ( N=800 neutrals, fixed k). Hy-
perparameters are selected by minimizing validation RMSE;
final metrics are reported on the held-out test split (Table 1).
We also report a semi-classification view by discretizing
entities into three RPS bands (low/mid/high) and evaluating
accuracy and macro-F1 using the same predicted scores
(Section 4.2). For completeness, the full sweep results
across all probe families and hyperparameter settings are
provided in Table 3.
Hyperparameter sweep.For completeness, we summa-
rize the principal hyperparameters explored for each family.
Unless stated otherwise, all sweeps are performed indepen-
dently per retriever and perk.
•Ridge regression.Regularization strength α∈
{10−6,10−5, . . . ,103}(log-spaced); intercept en-
abled; features unnormalized or standardized (both
evaluated).
•MLP.Hidden widths ∈ {256,512,1024} ; depth
∈ {1,2,3} ; dropout ∈ {0.0,0.1,0.2} ; learn-
ing rate ∈ {10−4,3×10−4,10−3}; batch size ∈
{256,512,1024} ; early stopping on validation RMSE
with patience 10.
•XGBOOST.Number of trees ∈ {300,600,1000} ;
max depth ∈ {4,6,8} ; learning rate η∈
{0.03,0.05,0.1} ; subsample ∈ {0.7,0.9,1.0} ; col-
umn subsample ∈ {0.7,0.9,1.0} ; minimum child
weight ∈ {1,5,10} ; L2 regularization λ∈ {0,1,10} .
We use early stopping on validation RMSE.
Model selection.For each retriever, we select the probe
that attains the lowest validation RMSE and report its test
performance. In our experiments, XGBOOSTis frequently
the best-performing family, although linear and MLP probes
can be competitive depending on the retriever and pool-
ing scheme. We emphasize that probe performance is not
the primary contribution; rather, strong performance across
families supports the conclusion that retrievability risk is
encoded in embedding geometry and can be detected pre-
index.
13

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
N=400N=600N=800ContrieverJinaBGE-m3GritLMQwen3NV-embedReasonIRReason-Emb
N=200N=100
Figure 6.Extended geometric visualization of two-dimensional LDA projections for all evaluated retrievers under increasing
neutral pool sizes N. Consistent with the main analysis, entities are labeled by RPS terciles (low/mid/high) at k= 50 . The full
benchmark reveals that standard dense retrievers (e.g., BGE-M3, Qwen3, GritLM) exhibit a collapsing geometric structure similar to
Contriever, where low-RPS regions dominate as competition increases. In contrast, specialized models like Jina and ReasonIR maintain
more distinct high-RPS clusters, though intrinsic blind spots persist across all architectures.
14

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
Table 3.Comprehensive performance benchmark of embedding-based diagnostic probes across diverse architectures. We compare
learned probes (Ridge, XGBoost, MLP) against trivial baselines (All-One, All-Zero) for predicting RPS ( k= 50, N= 800 ). Across all
retrievers, learned probes consistently achieve significantly lower error (RMSE) and higher correlation than baselines, validating that blind
spots are predictable geometric properties rather than random noise. The optimal configuration for each retriever (selected via lowest
RMSE) is summarized in the main text (1).
Retriever ArchitectureRegression Metrics Semi-Classification Metrics
RMSE(↓)MAE(↓)Pearsonr(↑)Spearmanρ(↑)Macro-F1(↑)Macro-Rec.(↑)Macro-Prec.(↑)Prec weighted (↑)F1 weighted (↑)Accuracy(↑)
BGE-M3All One 0.783 0.749 0.000 0.000 0.027 0.333 0.049 0.080 0.012 0.080
All Zero 0.340 0.251 0.000 0.000 0.242 0.333 0.280 0.726 0.610 0.726
Ridge 0.169 0.121 0.674 0.638 0.655 0.528 0.548 0.767 0.754 0.767
XGBoost0.168 0.118 0.681 0.644 0.658 0.540 0.573 0.781 0.762 0.781
MLP 0.170 0.122 0.671 0.632 0.654 0.522 0.542 0.766 0.752 0.766
CONTRIEVERAll One 0.798 0.770 0.000 0.000 0.019 0.333 0.036 0.058 0.006 0.058
All Zero 0.310 0.230 0.000 0.000 0.251 0.333 0.286 0.753 0.647 0.753
Ridge 0.168 0.121 0.591 0.5820.7740.476 0.460 0.778 0.759 0.778
XGBoost0.157 0.109 0.658 0.6220.7270.501 0.506 0.795 0.777 0.795
MLP 0.168 0.121 0.588 0.5790.7740.475 0.460 0.779 0.759 0.779
QWEN3-EMBEDDINGAll One 0.760 0.719 0.000 0.000 0.036 0.333 0.065 0.108 0.021 0.108
All Zero 0.373 0.281 0.000 0.000 0.226 0.333 0.269 0.677 0.547 0.677
Ridge 0.177 0.135 0.699 0.647 0.677 0.582 0.593 0.721 0.727 0.721
XGBoost0.153 0.111 0.781 0.721 0.699 0.619 0.646 0.764 0.760 0.764
MLP 0.180 0.137 0.694 0.637 0.670 0.585 0.597 0.720 0.726 0.720
GRITLM-7BAll One 0.753 0.716 0.000 0.000 0.036 0.333 0.065 0.108 0.021 0.108
All Zero 0.369 0.284 0.000 0.000 0.224 0.333 0.268 0.673 0.542 0.673
Ridge 0.171 0.128 0.696 0.638 0.658 0.576 0.589 0.734 0.733 0.734
XGBoost0.157 0.115 0.745 0.677 0.682 0.595 0.620 0.762 0.754 0.762
MLP 0.162 0.116 0.731 0.669 0.671 0.586 0.611 0.760 0.750 0.760
REASON-EMBEDAll One 0.750 0.710 0.000 0.000 0.036 0.333 0.066 0.109 0.022 0.109
All Zero 0.377 0.290 0.000 0.000 0.220 0.333 0.265 0.659 0.523 0.659
Ridge 0.169 0.128 0.716 0.693 0.680 0.577 0.576 0.732 0.729 0.732
XGBoost0.156 0.114 0.764 0.742 0.688 0.595 0.609 0.752 0.748 0.752
MLP 0.169 0.129 0.716 0.694 0.680 0.580 0.578 0.733 0.731 0.733
NV-EMBED-V2All One 0.772 0.738 0.000 0.000 0.028 0.333 0.052 0.085 0.013 0.085
All Zero 0.345 0.262 0.000 0.000 0.239 0.333 0.279 0.718 0.601 0.718
Ridge 0.179 0.132 0.624 0.576 0.6260.535 0.5500.736 0.734 0.736
XGBoost0.173 0.124 0.640 0.595 0.6570.517 0.5410.757 0.740 0.757
MLP 0.180 0.133 0.623 0.576 0.6270.5350.549 0.736 0.734 0.736
JINA-V3All One 0.703 0.661 0.000 0.000 0.041 0.333 0.073 0.124 0.027 0.124
All Zero 0.414 0.339 0.000 0.000 0.183 0.333 0.236 0.548 0.388 0.548
Ridge0.178 0.137 0.667 0.659 0.6410.557 0.574 0.655 0.653 0.655
XGBoost 0.179 0.138 0.661 0.650 0.633 0.537 0.551 0.650 0.643 0.650
MLP0.178 0.137 0.6670.6580.641 0.558 0.575 0.656 0.654 0.656
REASONIR-8BAll One 0.558 0.500 0.000 0.000 0.099 0.333 0.153 0.297 0.136 0.297
All Zero 0.558 0.500 0.000 0.000 0.091 0.333 0.143 0.274 0.118 0.274
Ridge0.156 0.121 0.779 0.788 0.710 0.658 0.674 0.674 0.674 0.674
XGBoost0.156 0.1210.776 0.781 0.707 0.641 0.659 0.662 0.661 0.662
MLP0.156 0.1210.778 0.787 0.708 0.657 0.673 0.672 0.673 0.672
C.2. Calibration and Residual Diagnostics
To complement aggregate regression/classification metrics,
we analyze whether probes arewell-calibratedand whether
their errors show systematic bias. Figure 7 reports two
diagnostics for the best probe per retriever.
Predicted-empirical density.The top row visualizes the
joint density of predicted dRPSkversus empirical RPSk. Con-
centration along the diagonal indicates good calibration,
while off-diagonal mass reveals over- or under-estimation
regimes. For standard retrievers, the density is heavily con-
centrated at low empirical RPS, reflecting that a large frac-
tion of entities fall into low-retrievability regions under our
stringent neutral competition setting ( N=800 ). Reasoning-
oriented retrievers show a broader spread with higher em-
pirical RPS mass, consistent with their stronger average
retrievability.
Residual distribution.The bottom row plots residuals
(RPS k−dRPSk). Across retrievers, residuals are centerednear zero with limited skew, indicating that the probes do
not exhibit large systematic optimism or pessimism. The
remaining dispersion is primarily attributable to (i) retriever-
specific noise in empirical RPS estimation due to finite query
sets|Tx|, and (ii) class imbalance in the underlying RPS
distribution (many low-RPS entities for standard retrievers).
Overall, these diagnostics support the use of probe predic-
tions as practical,pre-indexrisk scores for threshold-based
flagging in ARGUS.
D. Additional ARGUS Implementation
Details
This appendix summarizes practical details for implement-
ing ARGUS (Section 5), including named-entity extraction,
handling repeated entities, and the two augmentation modes
used in our experiments.
15

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
ContrieverJinaBGE-m3GritLMQwen3NV-embedReasonIRReason-EmbResidualsDensity of Predictions
Figure 7.Calibration analysis of embedding-based diagnostic probes (Predicted vs. Empirical RPS). (Top) Prediction density:
Heatmaps of predicted versus true RPS illustrate that probes recover the overall retrievability structure. For standard retrievers (e.g.,
Contriever), the concentration near low RPS reflects the skew toward geometrically hard-to-retrieve entities (blind spots). In contrast,
ReasonIR shows a more dispersed mass consistent with higher entity retrievability, which the probe tracks.(Bottom) Residual analysis:
Distributions of residuals (True −Predicted) are centered near zero, indicating limited systematic over/under-estimation and supporting
the use of these probes for preindex quantification of retrievability.
D.1. Named Entity Extraction
NER model and outputs.We extract named entities from
each corpus document Dusing an off-the-shelf NER tag-
ger,dslim/bert-base-NER . We usenamed entityto
denote an NER-extracted span in D, represented as a tuple
(m, s)wheremis the extracted surface form (string) ands
is its character span (start/end offsets) inD.
Span handling and mapping to retriever tokens.Be-
cause retriever encoders operate on tokenized inputs, we
map each character span sto the corresponding token in-
dices under the target retriever’s tokenizer. If a span aligns
to multiple wordpieces, we treat the entire aligned token
range as the entity span for pooling.
D.2. Repeated Mentions and Document-Level Risk
Aggregation
A document may contain multiple mentions of the same
named entity surface form (or closely related surface forms).
ARGUS uses a conservative aggregation scheme to avoid
missing high-risk cases while preventing redundant augmen-
tation.
Per-mention scoring.For each extracted mention (m, s)
in document D, we compute a context-dependent embed-
ding using the target retriever and the same span pooling
operator as in Section 3.2:
em,D,s =g(E θ(D), s).
We then interpret the mention’s retrievability under a top- k
budget as the expected top- khit probability over its related-
query setT m:
dRPS k(m|D, s) =E t∼Uniform(T m)[Hit k(m, t)],where Hitk(m, t) =I[rank(m|t)≤k] is computed by
ranking magainst the related-entity-specific neutral pool
fort(as in Section 3.2). In practice, we estimate this expec-
tation by the empirical mean of hit indicators overt∈ T m.
Risk-conserving aggregation for repeated entities.If
the same surface form mappears multiple times in Dwith
spans{s1, . . . , s k}, we assign the entity a single document-
level risk score using the minimum predicted score:
dRPSk(m|D) = min
jdRPSk(m|D, s j).
Thisrisk-conservingrule ensures that if any occurrence of
eis embedded into a low-retrievability region (e.g., due to
local context), the entity is treated as high-risk.
De-duplication and augmentation once per entity.We
then flag eas high-risk if dRPSk(e|D)< τ and include it in
Erisk(D) (Section 5.1). Importantly, we apply augmentation
once per flagged entity per document, even if the entity
appears repeatedly:
Erisk(D) ={e: dRPSk(e|D)< τ}.
This avoids multiplying near-duplicate augmented views
solely due to repeated mentions while preserving the con-
servative detection behavior via the minimum rule above.
D.3. ARGUS Remedy Procedure (Non-Code Summary)
This subsection summarizes the remedy stage at indexing
time (Section 5.2) in three steps.
Step 1: Reference KB retrieval.For each flagged entity
m∈ E risk(D), we retrieve a concise defining context from
a Reference KB (Wikipedia first paragraphs in our experi-
ments). We query the KB using the entity surface form m
16

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
(not the full document) and retrieve the top kintpassages
with BM25S(k int=2):
{pm,1, . . . , p m,k Aug} ←BM25s(KB,query=m).
Step 2: Construct augmented document views.We in-
stantiate two alternatives:
(i) Document Expansion (Concatenation).For each re-
trieved passage pm,i, we create an expanded view by ap-
pending the passage to the document:
Dexp
m,i=D∥p m,i.
IfND=|E risk(D)| , this produces kint·NDexpanded views,
in addition to indexing the original document.
(ii) KB-guided LLM Synthesis.We aggregate all retrieved
passages for all flagged entities in Dand generate a single
unified augmented view:
Dsynth=LLM(D,{p e,i}e∈E risk(D), i≤k int),
where the LLM inserts short, entity-focused clarifications
only where needed (prompt details below). This option
produces exactly one synthesized view per document.
Step 3: Indexing strategy.In both modes, we index aug-
mented viewsalongsidethe original document (never re-
placing it). Thus, the index contains: (i) the original D, (ii)
allDexp
e,iviews (expansion), or (iii) the single Dsynthview
(synthesis). Retrieval is then performed over the expanded
index using the same target retriever and scoring procedure
as in theOriginalbaseline.
D.4. Prompt Template for KB-guided LLM Synthesis
Figure 8 shows the prompt template used forKB-guided
LLM Synthesis. Given document Dand the retrieved KB
passages for its flagged entities, the prompt instructs the
model to produce a single augmented document that pre-
serves the original meaning and adds only minimal clarifi-
cations needed to improve retrievability.
E. Additional Analyses
E.1. Association Between Entity-Level RPS and
Retrieval Success
This analysis probes whether entity-level retrievability (as
measured by RPS) isassociatedwith downstream document
retrieval success in benchmark settings. We emphasize
that retrieval outcomes depend on many query–document
factors (e.g., query phrasing, document length, topicality,
and semantic match), so this analysis is not intended to
establish causality.Setup.For each benchmark instance, we consider the gold
(relevant) document(s) and extract named entities using
the same NER procedure as in Appendix D.1. For each
extracted entity, we estimate its retrievability score under
the target retriever using our diagnostic probe, yielding
dRPSk(e|D) . We then separate gold documents into two
groups: those that are successfully retrieved within the top- k
window and those that are not.
Statistic.For each group, we compute the maximum pre-
dicted RPS among entities contained in the gold document:
max
e∈E(D)dRPSk(e|D),
and report the difference in this maximum between the re-
trieved and unretrieved groups. A positive delta indicates
that retrieved gold documents tend to contain at least one
entity with higher geometric visibility, suggesting an associ-
ation between entity-level retrievability and retrieval success
in entity-centric settings.
Results and interpretation.Table 4 summarizes this dif-
ference across BRIGHT and RAR-b subsets. Positive deltas
(shown in green) indicate subsets where retrieved documents
tend to contain entities with higher RPS, while negative
deltas indicate subsets where this association is weaker or
absent—consistent with retrieval being influenced by ad-
ditional query–document factors beyond entity geometry.
Overall, these results support the view that RPS captures
a meaningful component of retrieval difficulty in entity-
centric scenarios, while also highlighting that it is not the
sole determinant of retrieval success.
F. Additional Experimental Results
This appendix provides additional experimental results be-
yond the subset displayed in the main paper for readabil-
ity. We include (i) full per-task tables for each benchmark
and (ii) extended cutoff metrics to complement the main
nDCG@5/10 reporting.
F.1. Full Per-Task Tables and Extended Cutoffs
Full per-task results.We report complete per-task break-
downs for all benchmarks evaluated in Table 2, including:
BRIGHT(all 10 domains),ImpliRet(both settings), and
RAR-b(all 7 subsets). These tables mirror the main-table
format, comparingOriginalindexing against ARGUS with
Document ExpansionandLLM Synthesisfor each retriever.
Extended cutoff metrics.In addition to nDCG@5/10
(main paper), we report nDCG at larger cutoffs to character-
ize broader recall-oriented behavior:
nDCG@20,nDCG@50.
17

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
You are an expert editor and fact-checker. Your goal is to improve a text for a search engine by adding speciﬁc descriptions, but ONLY where the Wikipedia context clearly matches the meaning in the text. ### Task Instructions: 1. Read the ‘Input Text’. 2. Review the list of ‘Candidate Entities’ and their provided Wikipedia context. 3. For EACH entity, perform a Context Check: -Does the entity in the ‘Input Text’ refer to the exact same concept described in the Wikipedia context? -Example of MISMATCH: Input says ‘it was half past nine’ (time), but Wikipedia describes ‘Half Past Nine’ (the album). -> ACTION: Do NOT augment. -Example of MATCH: Input says ‘St. Peter’, Wikipedia describes ‘St. Peter Church in Zurich’. -> ACTION: AUGMENT. 4. Generation Step: Rewrite the Input Text. If (and ONLY if) an entity passes the Context Check, insert a short description (max 5 words, enclosed in commas) immediately after the entity. 5. If an entity fails the check or you are uncertain, leave it exactly as it is. 6. The short description must be a concise type/role/category derived from the Wikipedia context (e.g., ‘industrial EBM album’, ‘Swiss Catholic church in Zurich’), not a long story or extra sentence. 7. Do not change the wording, order, or punctuation of the Input Text, except for inserting these short descriptions. Do not introduce new entities or facts that are not supported by the Wikipedia context. 8. If the same entity appears multiple times in the Input Text, augment only the ﬁrst occurrence and leave the others unchanged. 9. Your entire output MUST be only the fully augmented text, with no headings, labels, explanations, or surrounding formatting. ### Candidate Entities & Context: [ENTITY_1] : [WIKI_FIRST_PARAGRAPH_1] [ENTITY_2] : [WIKI_FIRST_PARAGRAPH_2] … ### Input Text: [ORIGINAL_DOCUMENT_TEXT] ### Final Annotated Text:
Figure 8.Prompt template used for KB-guided LLM synthesis in ARGUS. Candidate entities are paired with retrieved Wikipedia
first-paragraph contexts; the model inserts short comma-delimited descriptors only when the context check passes.
Where space permits, we additionally include Recall@k at
matching cutoffs to highlight changes in coverage at larger
retrieval windows. These extended-cutoff results are con-
sistent with the main findings: ARGUS improves retrieval
quality across a wide range of retrievers and tasks, with
Document Expansiontypically providing the most stable
gains andLLM Synthesisoffering a more index-efficient
alternative with greater retriever-dependent variance.
F.2. Additional Notes on Metrics and Aggregation
Metric definitions.We use nDCG@k as the primary met-
ric to capture ranked relevance quality under practical top-
kbudgets. For each benchmark, we compute nDCG@k
following the dataset’s standard evaluation protocol and
relevance labeling.
Full Benchmark Avg.To summarize suite-level perfor-
mance without relying on a small set of displayed tasks,
we reportFull Benchmark Avg.for each benchmark as the
arithmetic mean of the metric across the full evaluated task
suite:
FullAvg=1
|S|X
s∈SMetric(s),
where Sis the set of tasks/subsets for that benchmark
(BRIGHT: 10 domains; ImpliRet: 2 settings; RAR-b: 7
subsets). This aggregation is computed separately for eachretriever and each system configuration (Original, ARGUS-
Expansion, ARGUS-Synthesis).
Interpretation.Because Full Benchmark Avg. averages
across all tasks, it provides a robustness-oriented summary
of performance and reduces sensitivity to which per-task
columns are displayed in the main paper. Extended per-
task results and additional cutoffs (e.g., nDCG@20/50) are
reported in Table 5.
18

With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots
Table 4.Difference in maximum RPS between retrieved and unretrieved gold documents (BRIGHT, RAR-B). Positive deltas
(green) indicate that retrieved documents tend to contain entities with higher geometric visibility (RPS), suggesting an association between
entity-level retrievability and retrieval success in entity-centric settings. Negative deltas indicate subsets where this association is weaker
or absent, consistent with retrieval being influenced by additional query–document factors beyond entity geometry.
Retriever kBRIGHT RAR-b
Biology Psychology Theorems Questions HellaSwag PIQA
BGE-M310 35.00/31.56(+3.45) 35.32/33.31(+2.01) 24.72/22.89(+1.83) 22.96/25.91(-2.95) 25.97/24.50(+1.47) 20.33/28.56(-8.23)
20 36.10/30.73(+5.37) 35.34/32.76(+2.58) 23.38/23.07(+0.31) 25.42/25.45(-0.04) 25.36/25.67(-0.30) 20.24/30.36(-10.12)
50 34.97/30.55(+4.43) 34.67/33.74(+0.93) 21.71/23.53(-1.82) 24.90/25.64(-0.74) 25.55/24.99(+0.56) 20.93/31.08(-10.15)
CONTRIEVER10 33.55/30.04(+3.52) 31.20/28.48(+2.72) 44.60/20.68(+23.93) 32.10/24.50(+7.60) 23.51/18.31(+5.19) 19.52/18.30(+1.22)
20 32.16/30.33(+1.83) 31.00/28.52(+2.49) 36.60/20.52(+16.09) 32.57/24.13(+8.44) 23.00/17.97(+5.02) 19.27/18.88(+0.39)
50 31.43/30.77(+0.66) 30.79/27.63(+3.16) 37.93/20.01(+17.92) 32.21/23.81(+8.41) 22.59/18.37(+4.22) 19.27/18.88(+0.39)
textscEeason-Embed10 46.40/47.81(-1.41) 52.62/47.03(+5.60) 34.95/36.67(-1.72) 37.52/33.09(+4.43) 37.57/40.53(-2.96) 48.74/31.93(+16.81)
20 46.55/47.87(-1.33) 52.34/46.96(+5.38) 33.26/37.98(-4.72) 37.52/33.09(+4.43) 38.12/39.99(-1.88) 47.81/31.61(+16.20)
50 46.91/47.36(-0.45) 49.83/49.20(+0.63) 34.15/37.93(-3.78) 37.28/32.97(+4.31) 39.35/35.59(+3.76) 43.08/33.20(+9.89)
GRITLM-7B10 46.80/42.68(+4.13) 35.69/39.20(-3.51) 30.21/26.91(+3.30) 24.48/25.55(-1.07) 28.98/26.15(+2.84) 24.80/26.14(-1.34)
20 47.53/42.33(+5.20) 35.32/39.53(-4.21) 30.21/26.91(+3.30) 26.15/25.35(+0.80) 28.15/26.82(+1.33) 24.80/26.14(-1.34)
50 46.72/42.26(+4.46) 37.14/39.73(-2.58) 30.21/26.91(+3.30) 27.19/25.13(+2.06) 27.24/28.27(-1.03) 24.80/26.14(-1.34)
JINA10 68.52/66.33(+2.19) 67.76/68.34(-0.58) 65.97/54.71(+11.26) 55.59/49.69(+5.90) 67.22/64.35(+2.87) 53.43/62.42(-8.99)
20 67.87/66.75(+1.12) 67.86/68.26(-0.39) 65.97/54.71(+11.26) 51.97/51.55(+0.42) 67.17/64.11(+3.06) 54.84/59.26(-4.42)
50 66.51/69.48(-2.98) 68.05/68.10(-0.04) 59.39/56.21(+3.19) 54.02/49.63(+4.39) 67.28/61.75(+5.53) 54.84/59.26(-4.42)
Table 5.Complete downstream retrieval performance (nDCG@5/10/20/50) across all individual tasks in BRIGHT, IMPLIRET,
and RAR-B.This table complements Table 2 (main text) by providing the granular performance breakdown for every specific sub-domain
(e.g., all 10 subject categories in BRIGHT, all 7 tasks in RAR-B). We observe that ARGUS (via Document Expansion or LLM Synthesis)
yields consistent improvements across the vast majority of individual tasks, confirming that the holistic gains reported in the main paper
are driven by robust, widespread enhancements rather than outlier performance in a few categories.Colors:Best Result, 2nd Best, 3rd
Best.
Retriever AugmentationBright Impliret Rar-b
Shown (10/10) Full Benchmark Avg. Shown (2/2) Full Benchmark Avg. Shown (7/7) Full Benchmark Avg.
Biology Earth Econ Pony Psych Robot Stack Sust TheoQ TheoTAvg. (All 10 Tasks)Multi UniAvg. (All 2 Tasks)ARC Alpha PIQA SiQA Spart Hella WinoAvg. (All 7 Tasks)
BGE-M3Baseline 7.8/9.5/10.8/13.0 13.6/15.5/17.2/20.3 10.0/11.7/12.8/15.9 17.5/14.8/13.2/17.5 11.2/13.2/14.7/18.0 11.0/12.2/13.5/15.8 9.6/10.8/12.2/16.6 9.0/10.1/12.7/16.9 7.7/8.4/9.3/10.4 4.2/4.2/4.9/5.6 10.2/11.0/12.1/15.0 17.9/23.3/30.4/35.7 13.4/18.8/25.3/32.3 15.7/21.1/27.9/34.0 7.8/9.0/10.0/12.0 22.8/24.8/26.2/27.6 20.9/22.9/24.8/26.8 4.0/4.9/5.6/6.5 5.3/7.5/9.7/11.5 23.3/25.5/27.3/29.1 35.8/41.7/44.9/46.9 17.1/19.5/21.2/22.9
ARAGUS (Doc-Exp.) 8.6/11.4/14.2/18.8 22.9/34.6/52.2/85.7 13.1/17.6/22.4/31.919.8/17.1/16.1/23.4 13.2/17.7/24.3/37.512.3/14.6/17.3/21.6 12.8/18.5/23.6/32.210.4/12.8/16.8/27.27.9/9.6/11.4/14.4 4.5/4.9/5.5/6.312.5/15.9/20.4/29.9 30.2/38.3/48.6/66.8 26.7/34.0/43.3/59.5 28.4/36.1/46.0/63.17.8/9.2/10.2/12.3 23.0/24.9/26.5/28.021.9/24.7/27.3/30.24.0/4.9/5.6/6.4 5.3/7.5/9.7/11.5 24.0/26.9/29.4/32.435.8/41.7/44.9/47.1 17.4/20.0/21.9/24.0
ARAGUS (LLM-Synth.)13.6/14.7/15.2/15.6 18.2/19.6/20.3/20.9 12.6/13.9/14.5/15.034.6/36.6/37.6/38.1 14.6/15.6/16.1/16.516.1/16.9/17.6/18.0 9.9/10.6/11.1/11.612.1/13.1/13.9/14.38.1/8.3/8.6/8.9 3.6/3.7/3.9/4.114.3/15.3/15.9/16.3 22.5/27.5/34.5/40.5 16.5/21.5/28.5/35.5 19.5/24.5/31.5/38.08.5/9.8/10.8/12.8 23.5/25.5/27.0/28.521.8/23.8/25.8/28.04.1/4.9/5.9/6.8 6.0/8.2/10.5/12.3 24.8/26.8/28.8/30.838.5/41.0/44.6/47.5 18.2/20.0/21.9/23.8
CONTRIEVERBaseline 7.2/9.2/10.9/13.6 11.6/13.6/16.3/18.5 10.7/10.5/11.7/14.2 18.1/14.7/13.2/17.7 9.6/12.1/13.6/15.4 9.1/9.5/11.5/13.7 8.3/9.5/12.1/14.7 7.1/8.9/11.6/14.7 5.4/6.9/7.7/8.62.8/3.2/3.6/4.3 9.0/9.8/11.2/13.5 12.8/18.3/25.5/31.1 10.4/15.2/21.9/29.7 11.6/16.8/23.7/30.4 7.4/8.6/9.8/11.4 32.0/33.7/35.3/36.5 23.1/25.1/26.9/28.6 1.9/2.2/2.4/2.7 8.3/10.2/11.8/13.5 24.1/26.4/28.2/29.9 56.5/59.9/61.3/62.0 21.9/23.7/25.1/26.4
ARAGUS (Doc-Exp.) 8.6/13.0/19.7/29.6 20.3/31.6/49.8/95.6 13.7/16.4/22.7/38.620.5/17.1/15.5/20.711.6/17.8/25.1/38.010.6/11.8/14.5/19.6 14.3/19.2/26.4/36.4 8.1/11.9/17.9/25.75.5/7.0/8.2/9.52.8/3.2/3.9/4.811.6/14.9/20.4/31.818.0/24.9/34.5/50.216.9/22.8/31.0/47.417.4/23.9/32.7/48.87.4/9.0/10.4/12.338.7/42.8/46.0/49.024.1/26.8/29.1/31.5 2.0/2.4/2.7/3.1 8.3/10.2/11.8/13.5 24.3/26.8/28.7/30.565.7/77.1/84.3/89.224.4/27.9/30.4/32.7
ARAGUS (LLM-Synth.)8.8/11.5/13.9/18.6 13.9/18.2/23.0/28.1 10.1/11.2/13.0/16.122.6/18.5/16.8/21.310.3/14.3/17.8/21.812.0/12.7/14.2/17.2 8.0/10.3/12.1/16.9 7.1/8.2/11.8/16.86.9/8.8/10.2/11.72.7/3.9/4.6/5.410.2/11.8/13.7/17.418.4/24.3/32.1/44.417.0/22.4/29.4/43.517.7/23.3/30.7/43.97.7/9.7/11.3/13.637.0/40.2/42.4/44.527.2/30.6/33.3/36.1 8.3/10.1/11.2/13.2 8.3/10.1/11.8/13.8 28.5/32.8/35.8/38.762.4/71.1/76.9/81.325.6/29.2/31.8/34.5
QWEN3-EMBEDDINGBaseline 10.5/12.4/13.6/16.2 15.7/16.7/17.7/18.2 11.7/12.7/13.2/13.7 10.5/10.6/11.2/17.46.9/8.4/9.0/11.0 10.7/11.2/11.7/12.2 10.7/11.7/12.2/12.7 9.2/10.2/11.2/11.7 8.2/8.7/9.2/9.7 5.4/5.7/6.0/6.2 10.0/10.8/11.5/12.9 8.0/10.8/13.7/17.1 3.9/5.3/6.9/9.0 5.9/8.0/10.3/13.1 7.4/8.8/9.9/11.4 20.9/22.4/23.9/25.5 17.1/19.4/20.7/22.6 1.0/1.3/1.5/1.8 1.8/2.7/3.9/6.9 21.8/24.0/25.8/27.5 31.7/34.9/37.7/40.5 14.5/16.2/17.6/19.5
ARAGUS (Doc-Exp.)18.5/27.1/39.0/62.5 23.2/25.2/26.2/27.2 15.2/16.2/16.7/17.213.2/14.2/14.7/15.2 4.5/13.7/24.2/46.0 12.2/12.7/13.2/13.7 14.2/15.2/15.7/16.29.8/13.4/18.6/27.5 15.6/21.8/26.9/35.6 6.0/6.4/6.7/7.0 13.2/16.6/20.2/26.8 10.4/15.1/21.4/32.2 5.5/7.3/10.0/15.3 8.0/11.2/15.7/23.87.9/9.3/10.6/12.3 22.2/24.1/25.9/28.0 18.7/22.1/24.8/27.9 1.2/1.5/1.9/2.2 1.8/2.7/3.9/6.9 26.5/31.9/37.0/42.632.2/37.7/41.8/46.715.8/18.5/20.9/23.8
ARAGUS (LLM-Synth.) 9.1/13.7/15.7/19.5 18.2/19.7/20.7/21.2 14.2/15.2/15.7/16.213.7/14.7/15.2/15.710.0/12.6/14.7/16.9 11.7/12.2/12.7/13.2 13.2/14.2/14.7/15.210.7/11.7/12.2/12.7 14.6/17.1/19.0/21.1 5.7/6.0/6.2/6.4 12.1/13.7/14.7/15.8 8.4/12.0/15.3/19.4 5.0/5.4/5.7/6.0 6.7/8.7/10.5/12.77.9/9.5/11.0/12.8 24.0/26.1/27.7/29.9 19.5/22.3/24.2/26.8 1.1/1.3/1.6/2.01.8/2.7/3.9/6.925.4/28.7/31.3/34.136.2/38.2/39.2/39.716.5/18.4/19.9/21.7
NV-EMBED-V2Baseline 14.0/16.5/19.8/21.3 13.7/14.7/15.2/15.7 12.2/13.2/13.7/14.0 16.2/17.4/18.2/18.7 20.0/21.5/24.1/26.1 9.2/10.0/10.4/10.7 10.9/13.1/15.9/17.4 9.7/10.7/11.2/11.4 8.7/9.2/9.4/9.7 4.7/5.0/5.2/5.4 11.9/13.1/14.3/15.0 33.9/38.5/44.2/47.6 24.3/29.2/35.1/40.1 29.1/33.9/39.7/43.9 14.3/16.2/17.9/20.2 26.2/28.3/30.1/31.9 34.8/37.6/39.4/41.1 3.3/3.9/4.3/5.1 5.0/8.0/10.4/11.5 33.7/36.2/37.9/39.6 32.7/40.2/43.8/45.8 21.4/24.3/26.3/27.9
ARAGUS (Doc-Exp.)17.7/19.2/19.7/20.222.7/24.2/25.2/25.715.2/16.7/17.2/17.719.2/20.7/21.2/21.725.2/26.7/27.2/27.710.7/11.7/12.2/12.717.7/19.2/19.7/20.211.7/12.7/13.2/13.79.7/10.2/10.7/11.05.2/5.7/6.0/6.215.5/16.7/17.2/17.750.2/53.2/55.2/56.2 40.2/43.2/45.2/46.2 45.2/48.2/50.2/51.2 16.2/17.2/18.7/19.230.2/31.7/32.2/32.738.2/39.2/39.7/41.24.2/4.7/5.0/5.26.2/8.2/8.7/12.236.2/37.2/38.7/40.238.2/40.3/42.2/45.724.2/25.5/26.5/28.1
ARAGUS (LLM-Synth.) 15.7/17.7/20.7/23.218.7/21.7/26.7/35.213.2/14.7/16.2/18.717.7/19.2/21.2/23.222.2/24.2/27.2/29.79.2/10.7/12.2/14.715.2/17.7/20.2/23.710.2/11.7/13.7/16.78.7/9.7/11.0/13.04.7/5.5/6.5/8.013.6/15.3/17.6/20.638.2/42.2/48.2/53.2 28.2/32.2/38.2/44.2 33.2/37.2/43.2/48.7 15.2/17.5/19.0/21.527.5/29.5/31.5/33.536.0/39.0/41.0/43.03.8/4.5/5.0/5.86.0/9.0/11.5/13.035.0/37.5/39.5/41.534.5/42.5/46.5/49.522.6/25.6/27.7/29.7
REASON-EMBEDBaseline 14.5/18.6/21.0/25.1 14.7/16.1/18.4/21.8 10.2/11.2/13.1/15.5 13.0/12.2/12.1/18.2 13.4/15.0/16.4/18.7 4.7/5.6/7.3/9.9 11.7/14.5/16.6/19.3 10.5/12.4/15.7/18.6 12.4/13.5/15.0/16.7 6.0/7.2/9.3/11.211.1/12.6/14.5/17.5 7.8/11.0/15.2/20.0 1.8/2.4/3.1/4.7 4.8/6.7/9.2/12.4 7.9/9.2/10.2/11.66.7/7.4/8.1/9.4 12.4/14.1/15.7/17.7 4.0/4.4/4.7/4.8 7.0/7.7/8.2/8.4 20.8/22.9/24.6/26.3 31.3/34.3/35.3/35.8 12.9/14.3/15.3/16.3
ARAGUS (Doc-Exp.) 15.9/22.4/30.4/42.8 22.3/30.5/43.1/69.610.4/11.5/15.7/24.6 23.8/20.7/20.4/27.8 16.8/19.2/22.8/26.94.7/5.7/7.9/11.014.0/17.8/20.5/24.211.2/13.4/16.9/21.214.4/16.2/18.3/20.65.0/6.3/7.9/9.513.8/16.4/20.4/27.8 8.2/12.2/17.3/26.02.0/2.8/3.8/5.85.1/7.5/10.5/15.98.1/9.6/10.7/12.46.7/7.4/8.1/9.413.0/15.0/16.9/19.44.7/5.2/5.4/5.6 8.2/9.2/9.7/10.023.2/26.7/29.8/33.036.3/39.3/40.3/40.814.3/16.1/17.3/18.7
ARAGUS (LLM-Synth.)17.5/22.8/28.4/36.3 18.8/22.7/26.8/32.412.0/13.7/16.5/20.8 15.7/16.7/17.2/17.7 16.2/17.7/18.2/18.76.2/7.0/7.4/7.714.2/15.7/16.2/16.712.7/14.0/14.4/14.714.7/15.4/15.7/16.07.4/8.0/8.4/8.7 13.5/15.4/16.9/19.0 8.0/11.5/15.9/21.83.0/3.4/3.7/4.05.5/7.4/9.8/12.98.2/10.1/11.5/13.36.3/7.3/8.3/9.313.5/15.6/17.6/19.94.6/5.0/5.2/5.4 8.0/8.7/9.2/9.424.8/28.2/30.7/33.535.3/38.3/39.3/39.814.4/16.2/17.4/18.7
GRITLM-7BBaseline 5.9/7.0/8.2/10.8 4.7/5.5/7.1/9.1 4.1/4.4/5.9/6.8 11.2/11.4/11.8/16.5 16.2/17.7/18.2/18.7 2.4/2.8/4.4/5.6 5.7/7.0/8.4/11.0 4.1/4.8/5.9/7.2 2.2/2.5/2.7/3.3 0.4/0.4/0.4/0.4 5.7/6.4/7.3/8.9 5.6/7.3/9.1/11.8 15.2/16.7/17.7/18.2 10.4/12.0/13.4/15.0 3.1/3.9/4.2/5.1 5.1/5.9/6.5/7.3 3.3/3.9/4.6/5.3 3.2/3.7/4.0/4.2 5.0/5.7/6.2/6.4 16.5/18.3/19.7/21.4 36.2/39.2/40.2/40.7 10.3/11.5/12.2/12.9
ARAGUS (Doc-Exp.) 6.7/9.3/12.1/20.06.8/9.1/13.1/24.94.8/5.0/6.8/11.013.0/14.3/16.0/25.0 19.2/20.7/21.2/21.7 2.8/3.2/5.2/7.3 9.3/11.7/15.0/22.15.2/6.9/8.2/10.72.4/2.9/3.6/4.60.3/0.5/0.5/0.8 7.1/8.4/10.2/14.8 6.1/8.6/11.6/16.6 22.2/24.2/25.2/25.7 14.2/16.4/18.4/21.23.2/4.0/4.4/5.3 5.2/6.0/6.7/7.5 3.4/4.0/4.7/5.54.0/4.4/4.7/4.86.2/7.2/7.7/8.019.0/22.1/24.8/27.7 39.2/42.2/43.2/43.711.4/12.8/13.7/14.7
ARAGUS (LLM-Synth.)9.5/10.2/10.8/11.18.1/8.9/9.4/9.87.2/7.8/8.2/8.518.2/20.2/21.2/21.8 17.5/19.0/20.5/22.0 4.2/4.8/5.2/5.5 7.6/8.3/9.0/9.36.2/7.0/7.5/7.83.2/3.5/3.8/4.00.5/0.6/0.7/0.8 8.2/9.0/9.6/10.1 5.9/8.0/11.0/14.2 16.5/18.0/19.5/20.5 11.2/13.0/15.2/17.43.6/4.2/4.8/5.5 5.8/6.5/7.2/7.8 3.9/4.5/5.2/5.83.8/4.5/5.2/5.85.8/7.0/8.5/9.518.5/20.5/22.0/23.5 38.5/43.5/48.0/52.011.4/13.0/14.4/15.7
JINA-V3Baseline 12.0/15.2/17.9/21.5 19.6/22.1/25.0/28.1 18.6/19.2/21.2/24.2 16.2/17.7/18.2/18.7 20.7/21.3/23.1/25.9 11.0/11.7/13.0/15.2 11.4/14.3/16.4/21.3 13.1/15.8/18.6/21.9 11.4/13.4/14.9/16.67.6/9.8/10.5/11.8 14.2/16.1/17.9/20.5 14.8/20.6/27.8/33.1 10.9/15.2/21.2/26.7 12.9/17.9/24.5/29.9 11.1/13.2/15.0/17.0 22.0/23.8/25.2/26.7 26.5/29.1/30.9/32.6 2.5/2.9/3.4/4.1 1.8/3.7/5.9/8.1 24.6/27.0/28.8/30.6 22.6/25.0/26.2/27.2 15.9/17.8/19.3/20.9
ARAGUS (Doc-Exp.) 12.8/16.5/20.2/24.6 25.8/29.3/34.9/40.419.9/21.6/24.4/28.4 19.2/20.7/21.2/21.723.6/26.7/29.2/33.511.4/12.5/13.3/16.210.9/15.0/17.4/23.314.3/18.2/22.2/26.4 12.5/15.0/16.9/19.26.5/9.2/11.2/12.9 15.7/18.5/21.1/24.6 17.5/24.2/32.6/47.9 12.2/17.3/24.2/36.6 14.8/20.8/28.4/42.211.3/13.6/15.6/18.1 23.5/25.7/27.2/29.027.5/30.9/33.6/36.12.6/3.0/3.5/4.3 1.7/3.6/5.7/7.927.4/31.4/34.5/37.622.6/26.2/28.2/30.016.7/19.2/21.2/23.3
ARAGUS (LLM-Synth.)13.4/16.2/18.7/22.2 24.7/26.7/27.7/28.220.2/21.2/22.7/23.2 17.7/19.2/20.7/22.222.2/23.7/25.2/27.212.2/13.0/13.7/14.212.7/15.0/17.0/19.714.7/17.0/19.2/22.7 12.2/14.0/15.4/17.0 6.7/7.7/8.2/8.7 15.7/17.4/18.9/20.5 16.2/19.7/23.7/28.2 10.7/13.2/15.7/18.2 13.4/16.4/19.7/23.212.0/14.2/16.0/18.0 23.0/25.0/26.5/28.027.5/30.0/32.0/33.53.0/3.5/4.0/4.8 2.8/4.8/7.0/9.026.0/28.5/30.5/32.524.0/26.5/28.0/29.016.9/18.9/20.6/22.1
REASONIR-8BBaseline 16.6/19.1/22.3/24.6 16.2/17.7/18.4/18.7 13.9/16.5/17.8/21.9 14.8/14.7/14.3/20.4 21.8/24.1/27.1/30.8 14.1/14.9/15.7/16.0 15.1/17.8/21.8/26.2 10.2/11.4/12.0/12.4 8.2/8.7/9.0/9.2 4.9/5.1/5.3/5.4 13.6/15.0/16.4/18.6 22.7/27.7/33.6/38.2 8.1/10.5/13.3/17.2 15.4/19.1/23.5/27.7 12.2/13.5/14.6/16.1 20.1/21.9/23.3/24.9 23.3/25.9/27.6/29.64.3/4.6/4.8/4.9 5.9/6.7/7.2/7.4 30.9/33.2/35.0/36.833.3/35.2/36.3/36.8 18.6/20.1/21.3/22.4
ARAGUS (Doc-Exp.) 18.0/24.4/34.0/46.4 30.5/41.6/57.0/93.4 16.6/22.2/29.7/47.2 18.7/20.3/21.3/21.8 28.9/39.7/56.7/82.4 15.1/16.2/19.5/25.2 18.2/19.8/20.8/21.312.2/13.8/14.5/15.1 9.2/10.1/10.5/10.8 5.7/6.0/6.2/6.4 17.3/21.4/27.0/37.0 30.8/42.0/56.0/84.3 12.5/19.2/27.7/41.6 21.7/30.6/41.9/63.012.2/13.5/14.7/16.323.2/24.3/24.8/25.1 26.2/27.3/27.8/28.15.0/5.4/5.6/5.7 7.2/8.0/8.4/8.7 34.3/35.8/36.3/36.537.3/39.3/40.3/40.8 20.8/21.9/22.6/23.0
ARAGUS (LLM-Synth.)20.0/25.2/29.3/36.8 27.2/30.5/35.5/41.1 12.7/16.8/19.1/25.5 17.3/18.8/19.8/20.3 26.5/29.6/34.5/42.2 13.3/14.1/17.1/20.8 15.3/20.6/25.4/32.511.3/12.8/13.3/13.8 9.0/9.4/9.7/10.0 5.4/5.7/5.9/6.0 15.8/18.3/21.0/24.9 25.8/32.0/38.7/48.0 8.7/11.3/14.6/19.3 17.3/21.7/26.6/33.612.9/14.5/15.7/17.622.7/23.8/24.3/24.5 25.7/26.8/27.8/27.5 4.8/5.2/5.4/5.5 6.7/7.4/7.8/8.0 33.8/35.3/35.8/36.1 36.3/38.3/39.3/39.8 20.4/21.6/22.3/22.7
19