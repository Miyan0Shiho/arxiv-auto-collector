# Query-Conditioned Knowledge Alignment for Reliable Cross-System Medical Reasoning

**Authors**: Yan Jiao, Jingran Xu, Pin-Han Ho, Limei Peng

**Published**: 2026-05-18 15:49:46

**PDF URL**: [https://arxiv.org/pdf/2605.18570v1](https://arxiv.org/pdf/2605.18570v1)

## Abstract
Cross-domain knowledge alignment is essential for integrating heterogeneous medical systems, yet existing approaches typically treat entity alignment as a static matching problem, ignoring query context and cross-system asymmetry. This limitation is particularly critical in integrative medical settings, where correspondence between concepts is inherently context-dependent, non-bijective, and direction-sensitive.
  In this paper, we propose Query-Conditioned Entity Alignment (QCEA), which reformulates entity alignment as a query-conditioned correspondence problem. Instead of learning a fixed mapping between entity representations, QCEA treats the textual description of a source entity as a query and ranks candidate entities in the target graph, enabling context-dependent alignment. The framework integrates semantic encoding, graph-based representation learning, and a direction-aware transformation module to capture asymmetric and many-to-many correspondence across heterogeneous knowledge systems.
  We evaluate QCEA on TCM--WM knowledge graphs derived from SymMap, covering both symptom alignment and herb--molecule alignment tasks. Experimental results show consistent improvements over representative baselines, particularly on rank-sensitive metrics such as Hit@K and MRR. Furthermore, downstream retrieval-augmented generation (RAG) experiments demonstrate that improved alignment leads to better evidence retrieval, stronger grounding, and higher answer accuracy. These findings highlight that alignment is not merely a data integration step, but a key factor that shapes knowledge accessibility and reliability in cross-system medical reasoning.

## Full Text


<!-- PDF content starts -->

Highlights
Query-Conditioned Knowledge Alignment for Reliable Cross-System Medical Reasoning
Yan Jiao, Jingran Xu, Pin-Han Ho, Limei Peng
•Reformulates entity alignment as query-conditioned ranking
•Handles asymmetric and many-to-many cross-system correspondence
•Captures context-dependent alignment in medical knowledge graphs
•Improves top-rank alignment under semantic ambiguity
•Supports grounded retrieval and downstream medical reasoningarXiv:2605.18570v1  [cs.AI]  18 May 2026

Query-Conditioned Knowledge Alignment for Reliable Cross-System
Medical Reasoning
Yan Jiaoa, Jingran Xua, Pin-Han Hoa,b,∗and Limei Pengc,a
aShenzhen Institute for Advanced Study, University of Electronic Science and Technology of China, Shenzhen, China
bDepartment of Electrical and Computer Engineering, University of Waterloo, Waterloo, Canada
cSchool of Computer Science and Engineering, Kyungpook National University, Daegu, South Korea
ARTICLE INFO
Keywords:
Entity alignment
Knowledge graphs
Query-conditioned ranking
Cross-system medical knowledge inte-
gration
Retrieval-augmented generationABSTRACT
Cross-domain knowledge alignment is essential for integrating heterogeneous medical systems, yet
existing approaches typically treat entity alignment as a static matching problem, ignoring query
context and cross-system asymmetry. This limitation is particularly critical in integrative medical
settings,wherecorrespondencebetweenconceptsisinherentlycontext-dependent,non-bijective,and
direction-sensitive.
In this paper, we propose Query-Conditioned Entity Alignment (QCEA), which reformulates
entityalignmentasaquery-conditionedcorrespondenceproblem.Insteadoflearningafixedmapping
between entity representations, QCEA treats the textual description of a source entity as a query and
rankscandidateentitiesinthetargetgraph,enablingcontext-dependentalignment.Theframeworkin-
tegratessemanticencoding,graph-basedrepresentationlearning,andadirection-awaretransformation
module to capture asymmetric and many-to-many correspondence across heterogeneous knowledge
systems.
We evaluate QCEA on TCM–WM knowledge graphs derived from SymMap, covering both
symptom alignment and herb–molecule alignment tasks. Experimental results show consistent im-
provements over representative baselines, particularly on rank-sensitive metrics such as Hit@K and
MRR. Furthermore, downstream retrieval-augmented generation (RAG) experiments demonstrate
that improved alignment leads to better evidence retrieval, stronger grounding, and higher answer
accuracy. These findings highlight that alignment is not merely a data integration step, but a key
factor that shapes knowledge accessibility and reliability in cross-system medical reasoning.
1. Introduction
Large language models (LLMs) have become increas-
inglyimportantformedicalreasoningandknowledge-intensive
applications, where retrieval-augmented generation (RAG)
improvesfactualitybygroundingoutputsinexternalknowl-
edge sources [17]. However, their effectiveness critically
depends on the quality of retrieved evidence, which in
practice requires consistent and well-aligned knowledge
across heterogeneous medical systems. Such heterogeneity,
arising from distinct conceptual frameworks across clinical
traditions, poses fundamental challenges for cross-system
reasoning and retrieval [22, 36].
An illustrative case is the integration of traditional Chi-
nese medicine (TCM) and Western medicine (WM). These
paradigms adopt fundamentally different conceptual frame-
works [20], leading to semantic discrepancies that hinder
cross-system interoperability and reduce the reliability of
retrievalandclinicaldecisionsupport[1,28].Withthegrow-
ingadoptionofintegrativemedicine[24],aligningheteroge-
neous medical knowledge systems has become increasingly
important.
∗Corresponding author
yanyanjiao2018@gmail.com(Y. Jiao);17348800332@163.com(J. Xu);
pinhanho71@gmail.com(P. Ho);auroraplm@knu.ac.kr(L. Peng)
ORCID(s):0009-0000-9028-4614(Y. Jiao);0000-0002-0717-1481(P.
Ho);0000-0001-9984-9861(L. Peng)Beyond interoperability, such alignment is crucial for
knowledge-driven reasoning systems. In modern LLM-
based pipelines, RAG relies on aligned knowledge to pro-
vide grounded evidence for downstream inference. How-
ever, when cross-system correspondence is ambiguous or
misaligned, errors in alignment propagate into incorrect
evidence selection, resulting in ungrounded or hallucinated
outputs. Therefore, accurate cross-system alignment is not
merely a data integration problem, but a prerequisite for re-
liableknowledgegroundingandreasoninginheterogeneous
medical artificial intelligence (AI) systems.
Acentralchallengeistheambiguousandcontext-dependent
nature of entity correspondence. In TCM, a single clinical
concept can correspond to different WM entities depend-
ing on its descriptive context. For example,Qi Deficiency
mayalignwithfatigue-relatedorshortness-of-breath-related
manifestations under different descriptions. As illustrated
in Fig. 1(a), such correspondence is inherently context-
dependent.However,asshowninFig.1(b),existingmethods
based on fixed entity representations and static similarity
functions produce identical candidate rankings for different
descriptions, failing to capture such variability.
Furthermore,cross-systemcorrespondenceisinherently
non-bijective and direction-asymmetric. For example, a
TCM syndrome such asphlegm-heat obstruction in the
lungmay correspond to multiple WM symptoms, while a
WM symptom such as cough may relate to multiple TCM
syndromes. Such many-to-many and asymmetric relations
are not well modeled by approaches that assume symmetric
Y. Jiao et al.:Preprint submitted to ElsevierPage 1 of 15

Query-Conditioned Knowledge Alignment
•Terminologicalinconsistency•One-to-manysemanticcorrespondence•DependenceondescriptivecontextTCM KGWM KGSyndromeTCM SymptomHerbDiseaseWM SymptomGeneMoleculeCross-system MisalignmentExampleCross-system alignment is inherently ambiguous and context-dependent(a)Syndrome-oriented representationMechanism-based representation{Fatigue,ShortnessofBreath,…}Qi Deficiency
(b)
(c)Static Entity Alignment𝑢"=	argmax!∈#!"𝑓(𝐡$,𝐡!)Description-agnostic scoring Description-agnosticOutputRank1. Fatigue2.ShortnessofBreath3. Functional DeclineIdenticalrankingacrossqueriesFatigue, ShortnessofBreath, ...Description-agnostic Query
Query11. Fatigue2.ShortnessofBreath1.ShortnessofBreath2. FatigueQuery-Conditioned Entity AlignmentFatigue, ShortnessofBreath, ...WMEntityCandidatesQuery-conditionedOutputRank
RankingadaptstoqueryEntity𝑣: Qi DeficiencyDescription 𝑑%: fatigue-related conditionDescription 𝑑&:shortnessofbreath-related conditionEntity	𝑣: Qi DeficiencyQuery 1Query 2
Entity𝑣: Qi DeficiencyDescription 𝑑%: fatigue-related conditionDescription 𝑑&: shortnessofbreath-related conditionEntity𝑣: Qi DeficiencyQuery 1Query 2
Description-conditioned QueryQueryRepresentation𝒒𝒗(𝒊)𝑢"	=argmax!∈#!"𝑓(𝐪$(+),𝐡!)Query-conditionedscoringWMEntityCandidates
Query2(conditionedondescription)
✅
✅GTGT
Figure 1:Cross-system semantic misalignment and query-
conditioned entity alignment. (a) Context-dependent ambigu-
ity in cross-system correspondence. (b) Description-agnostic
entity-basedalignmentproducesidenticalrankingsfordifferent
descriptions. (c) QCEA enables query-conditioned alignment
with context-dependent outputs.
or one-to-one matching, highlighting the need for more
flexible alignment mechanisms.
TheselimitationsconflictwiththerequirementsofLLM-
basedretrievalandreasoning,whererelevanceisinherently
query-dependent. However, most existing entity alignment
approaches optimize fixed pairwise matching objectives,
implicitly assuming that correspondence remains invariant
across contexts. This assumption creates a mismatch with
downstream retrieval, in which the relevance of a candidate
entity depends on how the query is expressed. Existing
methods, including structural approaches based on graph
topology[9],embedding-basedmodelswithstaticsimilarity
functions [42], and text-enhanced methods [43], perform
alignment over fixed entity pairs, which limits their ability
to capture context-dependent correspondence.
We proposeQuery-Conditioned Entity Alignment
(QCEA), which reformulates entity alignment as a query-
conditioned ranking problem. QCEA treats the textual de-
scription of a source entity as a query and ranks candidates
in the target graph, aligning the alignment objective with
query-dependent relevance in retrieval. Unlike standard
denseretrieval,wherequeriesandcandidatesarepredefined
and the correspondence relation is typically many-to-one or
relevance-based,oursettingrequiresalignmentunderasym-
metric,non-bijective,anddirection-dependentcross-system
correspondence. The framework integrates domain-specific
semantic encoding, graph-aware representation learning, a
direction-awareTuckerprojection,andamany-to-manycon-
trastive objective to model context-dependent, asymmetric,and non-bijective correspondence. As shown in Fig. 1(c),
differentdescriptionsofthesameentitycanleadtodifferent
rankingoutcomes,enablingcontext-sensitivealignmentand
improving cross-system retrieval quality for LLM-based
reasoning.
We evaluate QCEA on heterogeneous medical knowl-
edgegraphsfromtherespiratorysubgraphofSymMap[33].
This evaluation focuses on TCM symptom–WM symptom
alignment and herb–molecule alignment tasks.
QCEA improves performance over representative base-
lines, particularly on rank-sensitive metrics such as Hit@K
and mean reciprocal rank (MRR). We also examine the
downstream impact of alignment quality in LLM-based
RAG settings, where improved alignment leads to bet-
ter cross-system retrieval, stronger answer grounding, and
higher question-answering accuracy.
Overall, QCEA serves as a unified framework that in-
tegrates semantic and structural signals for context-aware
alignment,enablingmorereliablecross-systemretrievaland
reasoning in heterogeneous medical knowledge graphs.
Key novelty.The contribution of QCEA does not lie in
introducingnewmodelcomponents,butinredefiningentity
alignment as a query-conditioned correspondence problem.
This shift changes the objective from learning a fixed map-
pingtomodelingaconditionalrelation,whichisessentialin
heterogeneousmedicalsystemswherethesamesourcecon-
cept may correspond to different target concepts depending
on descriptive context.
This work makes the following contributions:
•Query-conditioned formulation of entity align-
ment:Wereformulatecross-domainentityalignment
as a retrieval-oriented problem that conditions corre-
spondence on query semantics. Rather than claiming
novelty in query-conditioned ranking itself, we focus
onadaptingthisparadigmtosettingswithasymmetric
and many-to-many cross-system mappings.
•Direction-awarealignmentmodeling:Weintroduce
adirection-awaretransformationframeworkthatcap-
tures the asymmetry between heterogeneous medical
systems,enablingdistinctalignmentbehaviorsacross
domains.
•Comprehensive evaluation with downstream vali-
dation:WeevaluatetheproposedapproachonTCM–
WM datasets and demonstrate consistent improve-
ments in alignment accuracy. Additionally, down-
streamRAGexperimentsshowthatalignmentquality
consistentlyaffectsretrievalquality,answeraccuracy,
and evidence grounding.
Theremainderofthispaperisorganizedasfollows.Sec-
tion 2 reviews related work on knowledge graph alignment,
semanticmatching,andmedicalknowledgeintegration.Sec-
tion 3 presents the proposed QCEA framework, including
query representation, graph encoding, direction-aware pro-
jection,andthemany-to-manycontrastiveobjective.Section
Y. Jiao et al.:Preprint submitted to ElsevierPage 2 of 15

Query-Conditioned Knowledge Alignment
4 describes the experimental setup, datasets, baselines, and
evaluationprotocols.Section5reportsexperimentalresults,
ablation studies, and downstream RAG evaluation. Section
6discussesimplicationsandlimitations,andSection7con-
cludes the paper.
2. Literature Review
Cross-system medical knowledge integration involves
aligning heterogeneous conceptual systems to support reli-
able retrieval and reasoning. In settings such as TCM and
WM, discrepancies in terminology and abstraction create
challenges for consistent knowledge grounding [22, 36].
Existing research spans entity alignment, semantic match-
ing,andmedicalknowledgeintegration,butthesedirections
remain loosely connected, particularly in handling context-
dependent and asymmetric correspondence.
2.1. Knowledge Graph Entity Alignment
Knowledge graph entity alignment focuses on identi-
fying corresponding entities across heterogeneous graphs
by learning representations that preserve cross-graph sim-
ilarity [44]. Existing approaches differ mainly in how such
representationsareconstructed.Structuralmethodsleverage
graphtopologytocapturerelationalconsistency[30,32,38],
embedding-based models learn unified latent spaces for
alignment[26,5,42,27,34],andmorerecenteffortsexplore
improving representation quality through contrastive learn-
ing,multi-viewmodeling,andtheuseoftextualfeatures[6,
39].
Despite these differences, most existing methods for-
mulate alignment as a fixed pairwise similarity function
between entity representations, i.e., a context-independent
mapping between entities once embeddings are learned.
This assumption is often violated in cross-system medical
settings, where correspondence depends on how entity se-
manticsaredescribedandmayvaryacrosscontexts,leading
toasymmetricandmany-to-manyrelationships.Asaresult,
fixed pairwise matching is insufficient to capture the vari-
ability of cross-system correspondence. In addition, these
approachesarenotnaturallyalignedwithretrievalpipelines,
where candidate relevance is conditioned on a query and
directly affects downstream reasoning.
2.2. Query-Conditioned Semantic Matching and
Retrieval
Semanticmatchingininformationretrievaliscommonly
framed as a query-conditioned ranking problem, with rep-
resentative approaches including DRMM, DPR, and Col-
BERT [8, 10, 11]. These methods are widely adopted in
RAG systems, where they are used to retrieve evidence
conditioned on the input query for downstream generation.
However, these models operate within a single se-
mantic space and do not explicitly model correspondence
across heterogeneous conceptual systems. While they cap-
ture query-dependent relevance, alignment additionally re-
quires resolving cross-system ambiguity, where mappings
may be asymmetric or one-to-many. Consequently, evenwhen conditioned on queries, retrieval models alone cannot
ensure consistent access to aligned knowledge across sys-
tems.
WhileRAG[17]improvesdownstreamreasoning,itsef-
fectivenessdependsonreliableaccesstorelevantknowledge
across sources. In cross-system settings, this requirement
places alignment as a prerequisite for effective retrieval.
This motivates approaches that extend query-conditioned
ranking with mechanisms for modeling cross-system corre-
spondence.
2.3. Medical Knowledge Integration and
LLM-based Applications
Heterogeneous medical knowledge integration focuses
on harmonizing multi-source biomedical data to support
applicationssuchasclinicaldecisionsupportandknowledge
retrieval [35]. In TCM–WM settings, discrepancies extend
beyond terminology to fundamental differences in concep-
tual systems and diagnostic abstractions [25, 41].
Resources such as SymMap [33] provide cross-system
links between TCM and biomedical entities, yet these map-
pingsarefrequentlyincompleteandcontext-dependent,and
are typically constructed offline without adapting to query-
specific semantics. Terminology-based frameworks such as
UMLS [4] rely on synonymy within a shared ontology, an
assumption that often breaks down in TCM–WM scenarios
wherecorrespondencedependsoninterpretationratherthan
lexical similarity [16].
InLLM-basedmedicalsystems,inconsistenciesincross-
system mappings can propagate through retrieval pipelines
and result in unreliable evidence for downstream reason-
ing [28]. This observation motivates alignment methods
that jointly incorporate structural and semantic information
while remaining compatible with query-driven retrieval.
Overall, existing approaches exhibit three main limi-
tations: (1) alignment is typically formulated as context-
independent pairwise matching, (2) retrieval models cap-
ture query relevance but do not model cross-system corre-
spondence, and (3) medical knowledge integration relies on
static mappings that are not adaptive to query semantics.
To address these limitations, we introduce QCEA, a query-
conditioned,graph-aware,anddirection-sensitivealignment
framework.
3. Proposed QCEA Framework
This section introduces QCEA, a query-conditioned
alignment framework for cross-system medical knowledge
integration. Rather than learning a fixed correspondence
function between entity pairs, QCEA constructs a query
representationfromthesource-sideentitydescription,learns
graph-aware embeddings for TCM and WM entities, and
projects target-side representations into a shared alignment
space for ranking candidate entities. An overview of the
framework is shown in Fig. 2.
Y. Jiao et al.:Preprint submitted to ElsevierPage 3 of 15

Query-Conditioned Knowledge Alignment
QueryInstance𝜉!QueryEncoderMultilingual-e5-baseQueryEmbedding𝒛!Direction-SpecificQueryProjector𝑷!QueryRepresentationq!LinearProjection𝑾"QueryRepresentationConstructionTCMEntities𝓥#$%TCMEncoderZY-BERTGNNGraph-AwareEmbeddingsG#&'InputProjection𝑾#$%TCMEntityRepresentationLearningTCMTextEmbeddings𝑿#$%WMEntities𝓥(%WMEncoderSapBERTGNNGraph-AwareEmbeddingsG(%InputProjection𝑾(%WMEntityRepresentationLearningWMTextEmbeddings𝑿(%Target-sideGraph-AwareEmbedding𝐠"(fromG#$%orG&')TuckerProjection𝐡"(")*+,ResidualProjection𝐡",+!GatedFusionTargetRepresentationh)ScoringandRankingSharedAlignmentSpaceℝ*Dot-ProductScoring𝑓𝐪!,𝐡)=𝐪!⊺𝐡)Top-kRankingDirection-AwareTuckerProjection
TrainingObjectivePositivePool𝒫,	(𝑣)BidirectionalWeighting𝜆*-.ContrastiveRankingLossSharedGNN(𝜃)
Figure 2:Overview of the proposed QCEA framework. Query descriptions are encoded into query representations. TCM and WM
entities are mapped into graph-aware embeddings. A direction-aware Tucker projection module, conditioned on the alignment
direction𝑠, produces target representations for scoring and top-𝑘ranking under a direction-weighted, many-to-many-aware
contrastive objective.
What is alignment?Alignment is not merely ranking
relevance,butidentifyingcross-systemsemanticcorrespon-
dence under representational mismatch. Unlike retrieval,
where relevance is defined within a single semantic space,
alignment requires establishing semantic correspondence
between two heterogeneous spaces, where correspondence
may be asymmetric, context-dependent, and non-bijective.
Thisdistinctionisfundamental:retrievaloperatesinashared
semantic space, while alignment must bridge two distinct
representational systems.
3.1. Problem Formulation
Weconsidertwoknowledgegraphs,TCM= (TCM,TCM)
andWM = (WM,WM), representing TCM and WM
systems, respectively. Each entity𝑣is associated with a
textual description𝑑𝑣composed of its name and definition.
A set of anchor correspondences⊆TCM×WMis
available for supervision, where entities may have multiple
counterparts in the opposite graph.
We define an alignment direction indicator𝑠∈ {0,1},
where𝑠= 0denotes TCM→WM alignment and𝑠=
1denotes WM→TCM alignment. Given direction𝑠, the
source and target entity sets are defined as:
((𝑠)
src,(𝑠)
tgt) ={
(TCM,WM), 𝑠= 0,
(WM,TCM), 𝑠= 1.(1)
Givenasourceentity𝑣∈(𝑠)
srcanditstextualdescription
𝑑𝑣, QCEA defines a query instance𝜉𝑣= (𝑣,𝑑𝑣)and derives
aqueryrepresentation𝐪𝑣torankcandidateentities𝑢∈(𝑠)
tgt:
̂ 𝑢= arg max
𝑢∈(𝑠)
tgt𝑓(𝐪𝑣,𝐡𝑢),(2)
where𝐡𝑢denotes the target entity representation in the
sharedalignmentspace,definedbythetarget-sideprojection
module below, and𝑓(⋅,⋅)denotes the alignment scoring
function, instantiated as dot product in this work.
In practice, ranking may be performed either over the
full target graph or under type constraints, depending on
theevaluationsetting.Thisformulationrecastscross-system
entity alignment as a context-dependent ranking problem,rather than static pairwise matching. Unlike standard re-
trieval, the objective is not generic relevance estimation but
alignment under asymmetric and potentially many-to-many
correspondence, where a source entity may correspond to
multiple target entities depending on direction and descrip-
tive context.
3.2. Framework Overview
As illustrated in Fig. 2, QCEA comprises five com-
ponents: query representation construction, TCM entity
representation learning, WM entity representation learn-
ing,direction-awareTuckerprojection,andscoring/ranking.
Thesecomponentsarejointlyoptimizedunderacontrastive
ranking objective that accounts for many-to-many corre-
spondence through multi-positive sampling and bidirec-
tional weighting.
3.3. Query Representation Construction
QCEA derives a query representation for each source
entity from its textual description. The query instance𝜉𝑣=
(𝑣,𝑑𝑣)is encoded using multilingual-e5-base [29] to obtain
a dense embedding𝐳𝑣∈ℝ𝑑𝑞. This query encoder cap-
tures fine-grained, context-dependent semantics from tex-
tual descriptions, which differ from entity representations
optimized for structural consistency within each graph. The
encoder is kept fixed, and a subsequent projection layer
adapts the representation to the alignment space:
̂𝐳𝑣=𝐖𝑞𝐳𝑣,(3)
where𝐖𝑞∈ℝ𝑑×𝑑𝑞is a trainable linear transformation.
This design relies on lightweight projection layers on
top of a pretrained text encoder, rather than fine-tuning the
encoderitself,therebypreservinggeneralsemanticstructure
capturedduringlarge-scalepretrainingwhileadaptingtothe
alignmenttaskwithlimitedsupervision.Incontrast,directly
fine-tuningtheencoderunderlimitedalignmentsupervision
may lead to overfitting and reduced generalization.
The query embedding is mapped into the shared align-
ment space through a direction-specific projector:
𝐪𝑣= Norm(𝐏𝑠̂𝐳𝑣),(4)
Y. Jiao et al.:Preprint submitted to ElsevierPage 4 of 15

Query-Conditioned Knowledge Alignment
where𝐏𝑠∈ℝ𝑑×𝑑is a trainable projection matrix indexed
by𝑠,andNorm(𝐱) =𝐱∕‖𝐱‖2denotes𝓁2normalization.The
projector maps ̂𝐳𝑣into a shared𝑑-dimensional space (with
𝑑= 256), consistent with the target-side representations.
3.4. TCM and WM Entity Representation
Learning
QCEA initializes entity semantics on the two graph
sides using domain-specific text encoders specialized for
capturing entity-level semantics in their respective medi-
cal systems. Specifically, TCM entities are encoded with
ZY-BERT [21], while WM entities are encoded with Sap-
BERT [18], both based on their names and definitions. The
resulting embeddings are precomputed and used as input
features during training. The text encoders are kept fixed
to preserve general semantic structure, while subsequent
modules are optimized on top of these representations to
learn task-specific alignment patterns.
Let𝐱TCM
𝑣∈ℝ𝑑𝑇and𝐱WM
𝑢∈ℝ𝑑𝑊denotetheinitialtext
embeddingsofTCMandWMentities.Asthetwoencoders
operate in different semantic spaces, QCEA projects them
into a shared node feature space:
̃𝐱TCM
𝑣= Norm(𝐖TCM𝐱TCM
𝑣),
̃𝐱WM
𝑢= Norm(𝐖WM𝐱WM
𝑢),(5)
where𝐖TCMand𝐖WMaretrainableprojectionmatricesfor
the two graph sides.
Toincorporatestructuralcontext,QCEAappliesashared
graph encoder after input projection. Let ̃𝐗TCM∈ℝ𝑁TCM×𝑑
and̃𝐗WM∈ℝ𝑁WM×𝑑denote the projected node feature
matrices of the two graphs.
A graph encoder with shared parameters is applied to
bothgraphs,withmessagepassingperformedindependently
on each graph topology:
𝐆TCM= GNN𝜃(̃𝐗TCM,TCM),(6)
𝐆WM= GNN𝜃(̃𝐗WM,WM),(7)
whereGNN𝜃denotes a graph encoder with shared param-
eters𝜃. The resulting matrices𝐆TCMand𝐆WMrepresent
graph-aware node embeddings.
This design preserves side-specific semantics through
separate input projections while introducing a shared struc-
tural inductive bias via the shared graph encoder. The en-
coder is instantiated as a GCN [14] without cross-graph
message passing. The resulting node representations are
denoted as𝐠TCM
𝑣and𝐠WM
𝑢.
3.5. Direction-Aware Tucker Projection
Why Tucker?While simpler alternatives such as indepen-
dent linear projections per direction are possible, we adopt
Tucker decomposition for three reasons:
•Parametersharingacrossdirections:thefactorized
form couples transformations via shared factors, re-
ducing overfitting under limited alignment supervi-
sion.•Structuredlow-rankinteractions:unlikelinearpro-
jections, the decomposition captures multiplicative
dependenciesbetweenlatentfactors,yieldingaricher
transformation family.
•Asymmetry with efficiency: conditioning on align-
ment direction enables direction-specific mappings,
improving generalization under asymmetric corre-
spondence patterns while maintaining parameter ef-
ficiency through low-rank structure.
To map target-side graph embeddings into the shared
alignment space, QCEA adopts a direction-aware Tucker
projectionmodule[15],whosetransformationisconditioned
onthealignmentdirection𝑠.Theprojectionisimplemented
via a factorized multi-linear transformation with low-rank
decomposition, capturing dependencies between latent fac-
tors in the embedding space. By conditioning the factoriza-
tion on𝑠, the model enables direction-sensitive transforma-
tions while maintaining parameter efficiency.
For a candidate target entity𝑢∈(𝑠)
tgt, its graph-aware
embedding is defined as:
𝐠𝑢={
𝐠WM
𝑢, 𝑠= 0,
𝐠TCM
𝑢, 𝑠= 1.(8)
Specifically,thetransformationisdefinedbyadirection
factor matrix𝐔𝑠∈ℝ2×𝑅𝑠, an output factor matrix𝐔𝑜∈
ℝ𝑑×𝑅𝑜, an input factor matrix𝐔𝑖∈ℝ𝑑in×𝑅𝑖, and a set
of core slices{𝐆𝑟}𝑅𝑠
𝑟=1with𝐆𝑟∈ℝ𝑅𝑜×𝑅𝑖, where𝑑inand
𝑑denote the input and alignment embedding dimensions,
respectively. In our formulation,𝑑in=𝑑, since the graph-
aware embeddings are already represented in the shared𝑑-
dimensional space.
Conditioned on direction𝑠, the transformation is given
by:
𝐡tucker
𝑢=𝑅𝑠∑
𝑟=1𝐔𝑠[𝑠,𝑟]⋅𝐔𝑜𝐆𝑟(𝐔⊤
𝑖𝐠𝑢),(9)
where𝐔𝑠[𝑠,𝑟]denotes the(𝑠,𝑟)-th entry of𝐔𝑠.
Equivalently,theabovetransformationcanbeexpressed
in matrix form as:
𝐖(𝑠)=𝐔𝑜(𝑅𝑠∑
𝑟=1𝐔𝑠[𝑠,𝑟]𝐆𝑟)
𝐔⊤
𝑖.(10)
The transformed representation is:
𝐡tucker
𝑢=𝐖(𝑠)𝐠𝑢.(11)
To improve stability, we introduce a residual projection
branch:
𝐡res
𝑢=𝐑𝐠𝑢,(12)
andcombineitwiththeTuckerbranchviaalearnablescalar
gate:
𝐡𝑢= (1 −𝜎(𝛼))𝐡tucker
𝑢+𝜎(𝛼)𝐡res
𝑢,(13)
followedby𝓁2normalization.Theresidualbranchprovides
adirectlinearmapping,whilethegatecontrolsthetrade-off
between the Tucker-based transformation and the residual
pathway.
Y. Jiao et al.:Preprint submitted to ElsevierPage 5 of 15

Query-Conditioned Knowledge Alignment
3.6. Scoring and Ranking
After projection, the query representation𝐪𝑣and target
representation𝐡𝑢are compared in the shared alignment
space via dot product:
𝑓(𝐪𝑣,𝐡𝑢) =𝐪⊤
𝑣𝐡𝑢.(14)
Since both representations are𝓁2-normalized, the dot prod-
uctisequivalenttocosinesimilarity.Scoresareusedtorank
candidate entities and obtain top-𝑘predictions.
Atinferencetime,rankingisperformedoveracandidate
setoftargetentitiesunderthespecifiedalignmentdirection.
Depending on the evaluation setting, this candidate set may
be the entire target graph or a subset restricted to the same
entity type. For each query entity𝑣, candidate entities are
scored and sorted according to𝑓(𝐪𝑣,𝐡𝑢), and the top-𝑘
ranked entities are returned as alignment predictions.
3.7. Many-to-Many Contrastive Ranking
Objective
Cross-system medical alignment is inherently many-to-
many, where a source entity may correspond to multiple
valid targets. Accordingly, QCEA is trained with a con-
trastive ranking objective that supports multiple positives
per query, rather than a single-positive classification loss.
For each source entity𝑣∈(𝑠)
src, we define the positive
pool under direction𝑠as:
𝑠(𝑣) = {𝑢∈(𝑠)
tgt∣ (𝑣,𝑢) ∈(𝑠)},(15)
where(0)=and(1)= {(𝑢,𝑣) ∣ (𝑣,𝑢) ∈}
denotes the reversed correspondence set under the opposite
alignment direction.
Duringtraining,weadoptastochasticsamplingstrategy
to construct training instances. Specifically, we first obtain
a training-time positive pooltrain
𝑠(𝑣), which contains only
those correspondences in𝑠(𝑣)that are observed in the
trainingsplit.Thus,train
𝑠(𝑣)⊆𝑠(𝑣).Upto𝑃positivesare
thensampleduniformlywithoutreplacementfromthispool.
To ensure training stability, the current ground-truth target
associated with the query is always included in the sampled
positives.
Negative samples are drawn uniformly from the entire
targetgraph,excludingallknownpositivesintheglobalpool
𝑠(𝑣).Thisavoidsfalsenegativesfromvalidbutunobserved
correspondences.
Let{𝑢+
1,…,𝑢+
𝑃}⊆train
𝑠(𝑣)denote sampled positives
and{𝑢−
1,…,𝑢−
𝐾}denote sampled negatives. The corre-
sponding logits are:
𝓁+
𝑖=𝑓(𝐪𝑣,𝐡𝑢+
𝑖),𝓁−
𝑗=𝑓(𝐪𝑣,𝐡𝑢−
𝑗).(16)
QCEA optimizes a multi-positive contrastive objective
related to InfoNCE [23] and supervised contrastive learn-
ing [12]:
mp= − log∑𝑃
𝑖=1exp(𝓁+
𝑖∕𝜏)
∑𝑃
𝑖=1exp(𝓁+
𝑖∕𝜏) +∑𝐾
𝑗=1exp(𝓁−
𝑗∕𝜏),(17)where𝜏is a temperature parameter.
AlignmentislearnedinbothTCM→WMandWM→TCM
directions. The overall objective is defined as:
=𝜆dirWM→TCM +(1−𝜆dir)TCM→WM +𝜆regreg,(18)
where𝜆dirbalances the two alignment directions and𝜆reg
controls the𝓁2regularization termreg.
3.8. Discussion
QCEA departs from conventional entity alignment in
three aspects. It replaces static pairwise matching with
query-conditioned ranking, decouples query construction
from graph-aware entity representation learning, and in-
troduces a direction-aware Tucker projection with residual
fusion for target-side transformation.
Under this formulation, bidirectional training induces a
many-to-many alignment structure, which is naturally opti-
mized via a multi-positive contrastive objective. In contrast
to retrieval models and LLM-enhanced alignment methods
that focus on relevance estimation or pairwise similarity,
QCEAformulatesalignmentasaquery-conditionedranking
problem,aligningthetrainingobjectivewithretrieval-based
inference.
Theoretical insight.When cross-system correspondence
is context-dependent, a fixed deterministic correspondence
between entities is inherently insufficient, as it cannot rep-
resent one-to-many or description-dependent alignments
within a single function. In such cases, the alignment target
becomesconditionalonthequerycontextratherthansolely
determined by the source entity, since the same source
concept may correspond to different targets under different
descriptions. QCEA addresses this limitation by modeling
alignmentasaquery-conditionedrankingfunction,enabling
context-aware and non-bijective correspondence.
4. Experimental Setup
We evaluate QCEA at both the alignment level and the
system level, assessing its effectiveness for cross-system
retrievalandreasoninginLLM-basedmedicalapplications.
4.1. Datasets
We construct a cross-system medical entity alignment
benchmarkbetweenTCMandWM,consistingoftwotasks:
(1)symptomalignment(TCMsymptom↔WMsymptom)
and(2) herb–molecule alignment(TCM herb↔WM
molecule).
ThebenchmarkisderivedfromSymMap[33],apublicly
available resource linking TCM and biomedical entities. To
ensureclinicalrelevance,weadoptanICD-guided,disease-
centricconstructionstrategybyselectingrespiratory-related
diseases (ICD-10-CM J00–J99) and extracting subgraphs
of associated symptoms, TCM herbs, and WM molecules.
ICDcodesareusedfordomainscopingratherthandefining
alignment supervision. This construction yields clinically
grounded associations and naturally introduces many-to-
many and asymmetric correspondences.
Y. Jiao et al.:Preprint submitted to ElsevierPage 6 of 15

Query-Conditioned Knowledge Alignment
The resulting TCM and WM graphs contain 1,048 and
3,568entities,with6,012and13,176edges,respectively.Af-
terdeduplication,thebenchmarkincludes790symptomcor-
respondences and 19,440 herb–molecule correspondences,
with multiple associations per entity.
The two tasks exhibit distinct alignment characteristics.
Symptom alignment is near one-to-one, with most entities
associatedwithasmallnumberofcounterparts.Incontrast,
herb–molecule alignment exhibits substantially higher car-
dinality, with each TCM herb linked to more than 20 WM
molecules on average, and some exceeding 100.
Eachentityisassociatedwithtextualdescriptions(name
and definition), which are used to construct query represen-
tations. The benchmark is clinically grounded yet domain-
scoped, providing a controlled setting for studying cross-
system alignment under semantic heterogeneity, rather than
targeting broad generalization across all disease domains.
This disease-centric construction retains approximately
18.6% of the original SymMap connectivity and may intro-
duce bias toward well-documented respiratory conditions.
4.2. Evaluation Protocol
Entity alignment is evaluated as a ranking task, where
a query derived from a source entity ranks candidates in
the target graph. We consider two retrieval settings:(1)
type-constrained retrieval, which restricts candidates to
semantically compatible categories, and(2) full retrieval,
which considers all entities in the target graph. We adopt
a multi-positive, group-level evaluation protocol and report
Hit@K, Recall@K, and MRR. Hit@K measures whether
at least one correct entity appears in the top-𝑘predictions,
Recall@K measures the proportion of ground-truth entities
retrieved within the top-𝑘, and MRR evaluates the average
reciprocal rank of the highest-ranked correct entity. For the
Herb task, Recall@100 is additionally reported due to the
larger candidate space. Among these metrics, we consider
Hit@10andMRRastheprimaryevaluationcriteria,asthey
reflect top-rank performance most relevant to downstream
retrieval tasks. Hit@1 is reported as a secondary indicator
of precision, while Recall@K provides supplementary cov-
erage analysis.
Thedatasetissplitintotraining,validation,andtestsets
witha60%/20%/20%ratioatthealignment-pairlevel.Dueto
many-to-many correspondence, the same source entity may
appear in different splits with different target entities.
During evaluation, we adopt grouped multi-positive
ranking, where all ground-truth targets for each query are
evaluated jointly as a set. Model selection is based on
validation Hit@10 with early stopping, and performance is
reported on the test set. We further report stratified results
by alignment direction (TCM→WM vs. WM→TCM) and
ground-truth cardinality (GT=1 vs. GT>1).
Remark on data splitting.Although the same entity may
appear across training, validation, and test splits, alignment
supervision (i.e., entity pairs) is strictly separated. This
setting evaluates generalization to unseen correspondencesrather than unseen entities, which is consistent with trans-
ductive graph learning scenarios where entity vocabularies
are fixed but alignment patterns vary.
4.3. Implementation Details
Entities from the TCM and WM graphs are encoded
using domain-specific pretrained models and projected into
a shared space with dimension𝑑= 256. A shared GCN is
used to incorporate structural information.
Thedirection-awareTuckerprojectionadoptsrank(𝑅𝑠,𝑅𝑜,𝑅𝑖) =
(16,128,128). Models are trained with a many-to-many
contrastive ranking objective using multi-positive sampling
and bidirectional weighting. The direction weight𝜆diris set
to 0.5 for Symptom and 0.3 for Herb, reflecting differences
in alignment asymmetry across tasks.
Wereportresultsaveragedoverfiverandomseeds,with
standard deviation below 0.01 across main metrics, indi-
cating stable training behavior. Training is conducted for
300 epochs using Adam [13] with a learning rate of1 ×
10−3, batch size 64, and 1,024 negative samples per query,
chosen to balance performance and computational cost. We
additionally employ early stopping, gradient clipping, and
learning-rate decay to improve optimization stability.
4.4. Compared Methods
We compare QCEA with representative baselines cov-
ering different alignment paradigms:(1) Text-only Bi-
Encoder, a text retrieval baseline that matches query and
entity embeddings via cosine similarity using separate en-
codersandthesametextualinputsasQCEA,withoutgraph
or direction-aware modeling, thereby isolating the effect
of alignment modeling;(2) Cross-Attention, implemented
as a lightweight cross-encoder with multi-head attention
over projected token-level representations and a learned
scoring function, modeling bidirectional interactions be-
tween query and candidate embeddings;(3) MLP, im-
plemented as a parametric matching function over con-
catenated query–entity embeddings;(4) Procrustes[7],
learning a linear mapping with nearest-neighbor alignment;
(5) GCN-Align[30], capturing structural consistency via
graphconvolution;(6)RDGCN[32],extendingGCN-based
alignment with relation-aware interactions.
All neural baselines operate on precomputed text em-
beddings without additional encoder fine-tuning and are
trained with the same multi-positive and negative sampling
strategy. These baselines cover semantic, interaction-based,
parametric, geometric, and structural alignment paradigms.
Allmethodsareevaluatedunderthesamecandidatesets
following a unified protocol, including multi-positive rank-
ing and group-based metrics, under bothtype-constrained
andfull-retrievalsettings.
WefurtherincludeablationvariantsofQCEAbyremov-
ing key components, including query conditioning, shared
graph propagation, direction-aware Tucker projection, and
residual fusion.
Comparison with LLM-Based Alignment Methods.
LLM-based alignment methods (e.g., prompting GPT-4 [2]
Y. Jiao et al.:Preprint submitted to ElsevierPage 7 of 15

Query-Conditioned Knowledge Alignment
for free-form entity matching) operate in an open-ended
generation setting without a predefined candidate space,
whereas our evaluation focuses on ranking within a fixed
candidate set. This reflects a fundamental difference in
problem formulation between generation-based matching
andcandidateranking,makingdirectcomparisoninfeasible.
Exploring hybrid approaches that combine dense retrieval
withLLM-basedre-rankingremainsaninterestingdirection
for future work.
4.5. Downstream RAG Evaluation
To assess whether improvements in cross-system align-
ment translate to downstream performance, we construct a
controlled evaluation under a unified RAG framework. All
RAG-based settings share the same QA benchmark, gener-
ation model, decoding configuration, prompt template, and
answernormalizationrules,isolatingtheeffectofalignment
quality. The QA benchmark is constructed solely for down-
stream evaluation and is not used during model training.
Answer annotations are separated from alignment supervi-
sion,andthegenerationmodeldoesnotaccessgoldanswers
during training or inference.
To facilitate reproducibility, we will release the bench-
mark construction protocol, evaluation scripts, and prompt
templates upon acceptance. The benchmark contains 400
questions across eight categories, including four single-hop
and four two-hop tasks. Single-hop tasks evaluate direct
cross-system transfer, while two-hop tasks require addi-
tional intra-graph reasoning following cross-system transi-
tions. Each question is associated with reference answers,
goldevidence,andvalidationrules,enablingconsistenteval-
uation of retrieval, correctness, and grounding.
We compare six settings:Oracle(using ground-truth
alignments),QCEA(using predicted first-hop alignment
candidates),QCEA-TopX(retainingonlythetop-𝑋ranked
candidatestocontrolalignmentprecision,where𝑋isvaried
from 1 to 10),QCEA-DropX(randomly removing a subset
ofcandidatestotestrobustnesstocandidatelossatdifferent
removal ratios),NoAlign(removing cross-system links),
andOnly-LLM(no retrieval).
All RAG-based settings use the same generation model
(Qwen2.5-1.5B-Instruct [37]), ensuring a controlled com-
parison across alignment settings.
For the Only-LLM setting, we additionally evaluate
multiple language models with different capability profiles,
including a larger open general-purpose model (gpt-oss-
20B[3]),aTCM-orientedmodel(CMLM-ZhongJing[40]),
andaWM-orientedreasoningmodel(MedReason-8B[31]),
to examine whether increased model capacity or domain
specialization can compensate for the absence of cross-
system alignment.
This design isolates the effect of first-hop alignment
quality and evidence availability on downstream medical
question answering.
Choice of Language Model.We intentionally adopt a
lightweight language model to better expose the impact
of alignment quality on downstream reasoning, as smallermodels are less capable of compensating for missing or
incorrect evidence. This design choice isolates the effect
of alignment from the model’s parametric knowledge, pro-
viding a clearer view of how alignment quality influences
retrieval and grounding. While larger models may exhibit
differenttrade-offs,ourfocusisonunderstandingtherelative
impactofalignmentundercontrolledconditionsratherthan
achieving maximum absolute performance.
Weevaluateperformanceatthreelevels.Retrieval-level
metrics includeevidence recall@Kandcross-system hit
rate, measuring evidence coverage and cross-system align-
mentsuccess.Generation-levelmetricsincludeanswerac-
curacy,answeraccuracy(strict),andtwo-hopslotaccuracy.
End-to-endmetrics includegroundedness,end-to-end ac-
curacy,end-to-endaccuracy(strict),andhallucinationrate.
For Only-LLM, retrieval metrics are not applicable. In
this case, without retrieval, end-to-end accuracy reduces to
answer accuracy, and strict end-to-end accuracy reduces to
answer accuracy (strict).
5. Experiment Results
5.1. Comparison with Baseline Methods
Table 1 summarizes the main alignment results. The
baselinesrepresentstructuralmethods(GCN-Align,RDGCN),
geometricalignment(Procrustes),parametricmatching(MLP),
andsemantic/interaction-basedmodels(Text-onlyBi-Encoder,
Cross-Attention), which exhibit different performance char-
acteristics across tasks and settings.
Overall, QCEA achieves the strongest and most consis-
tent performance on the Symptom task. Under both full-
retrieval and type-constrained settings, it attains the top
resultsacrossallSymptommetrics,indicatingtheeffective-
nessofquery-conditionedrankingwhencross-systemcorre-
spondence is semantically nuanced but recoverable through
contextualizedmatching.Underfullretrieval,QCEAreaches
Hit@1/Hit@10/Recall@10/MRRof0.5130/0.9130/0.8572/0.6293,
and maintains similar advantages under type-constrained
retrieval (0.5130/0.9304/0.8746/0.6302).
On Herb, performance is more differentiated. QCEA
achieves the best Hit@10 and MRR under both settings,
while Procrustes attains the best Hit@1 and RDGCN the
best Recall@100, reflecting a trade-off between top-rank
precisionandbroadercandidatecoverage.Thisfindingsug-
geststhatQCEAconcentratesrelevanttargetsnearthetopof
the ranking, whereas structurally oriented methods such as
RDGCNremainadvantageouswhenevaluationemphasizes
widerrecall;inpractice,downstreamretrievalandreasoning
aretypicallymoresensitivetotop-rankedcandidatesthanto
deeper candidate coverage.
Baselinemethodsexhibitcomplementarystrengths.Graph-
based methods remain competitive when mappings are
relativelystable,withRDGCNperformingparticularlywell
in high-recall scenarios. Procrustes performs well when
correspondences are more regular or globally compressible
(e.g., WM→TCM Symptom). In contrast, Text-only Bi-
Encoder and Cross-Attention are less robust, especially
Y. Jiao et al.:Preprint submitted to ElsevierPage 8 of 15

Query-Conditioned Knowledge Alignment
Table 1
Comparison with baseline methods undertype-constrainedandfull-retrievalsettings. Recall@100 is used for Herb. The best and
second-best results in each column are highlighted inboldand underlined, respectively.
Symptom Herb
Full Type Full Type
MethodHit@1 Hit@10 Recall@10 MRR Hit@1 Hit@10 Recall@10 MRR Hit@1 Hit@10 Recall@100 MRR Hit@1 Hit@10 Recall@100 MRR
GCN-Align 0.4609 0.8696 0.7986 0.6000 0.4609 0.8783 0.8072 0.6015 0.0573 0.3512 0.4448 0.1449 0.0573 0.3512 0.4448 0.1449
RDGCN 0.4000 0.8174 0.8174 0.5305 0.4000 0.8174 0.8174 0.5313 0.0845 0.3962 0.76530.1835 0.0845 0.3962 0.76620.1835
MLP 0.0261 0.2696 0.2283 0.1135 0.0261 0.3043 0.2630 0.1271 0.0808 0.3700 0.5319 0.1720 0.0808 0.3700 0.5319 0.1720
Text-only Bi-Encoder 0.1652 0.5913 0.5605 0.2971 0.1739 0.6957 0.6612 0.3240 0.0019 0.0113 0.0169 0.0069 0.0075 0.0507 0.1587 0.0288
Cross-Attention 0.0000 0.0087 0.0029 0.0066 0.0870 0.3652 0.3536 0.1679 0.0032 0.0374 0.1295 0.0198 0.0032 0.0438 0.1466 0.0241
Procrustes 0.4261 0.8870 0.8304 0.5854 0.4261 0.8870 0.8315 0.58610.10700.3446 0.4882 0.1868 0.10700.3446 0.4882 0.1868
QCEA0.5130 0.9130 0.8572 0.6293 0.5130 0.9304 0.8746 0.63020.0958 0.47510.5967 0.20870.0958 0.47510.6069 0.2094
TCM→WM
GCN-Align 0.6250 0.9750 0.97500.7558 0.6250 0.9750 0.97500.7568 0.1026 0.56840.5393 0.2382 0.1026 0.56840.5393 0.2382
RDGCN 0.5750 0.8875 0.8875 0.6884 0.5750 0.8875 0.8875 0.68930.11320.53160.83420.2420 0.11320.53160.83420.2420
MLP 0.0000 0.2500 0.2500 0.0935 0.0000 0.3000 0.3000 0.11280.11320.5368 0.52830.2470 0.11320.5368 0.52830.2470
Text-only Bi-Encoder 0.2125 0.6875 0.6875 0.3552 0.2125 0.7875 0.7875 0.3705 0.0053 0.0316 0.0215 0.0137 0.0053 0.0342 0.0303 0.0182
Cross-Attention 0.0000 0.0000 0.0000 0.0011 0.1125 0.5000 0.5000 0.2199 0.0000 0.0108 0.0318 0.0084 0.0000 0.0189 0.0443 0.0137
Procrustes 0.5125 0.8625 0.8625 0.6319 0.5125 0.8625 0.8625 0.6321 0.0211 0.1579 0.2038 0.0702 0.0211 0.1579 0.2038 0.0702
QCEA0.67500.9625 0.9625 0.7663 0.67500.9625 0.9625 0.76640.1000 0.5579 0.5647 0.2274 0.1000 0.5579 0.5647 0.2274
WM→TCM
GCN-Align 0.0857 0.6286 0.3952 0.2439 0.0857 0.6571 0.4238 0.2465 0.0321 0.2307 0.3924 0.0932 0.0321 0.2307 0.3924 0.0932
RDGCN 0.0000 0.6571 0.6571 0.1697 0.0000 0.6571 0.6571 0.1701 0.0686 0.32120.72700.1510 0.0686 0.32120.72850.1510
MLP 0.0857 0.3143 0.1786 0.1591 0.0857 0.3143 0.1786 0.1598 0.0628 0.2774 0.5339 0.1304 0.0628 0.2774 0.5339 0.1304
Text-only Bi-Encoder 0.0571 0.3714 0.2702 0.1641 0.0857 0.4857 0.3726 0.2178 0.0000 0.0000 0.0143 0.0031 0.0088 0.0599 0.2300 0.0347
Cross-Attention 0.0000 0.0286 0.0095 0.0193 0.0286 0.0571 0.0190 0.0490 0.0053 0.0548 0.1933 0.0272 0.0053 0.0601 0.2135 0.0309
Procrustes0.2286 0.9429 0.7571 0.4791 0.2286 0.9429 0.7607 0.4808 0.1547 0.44820.6460 0.2515 0.1547 0.44820.6460 0.2515
QCEA 0.1429 0.8000 0.6167 0.3161 0.1429 0.8571 0.6738 0.3189 0.0934 0.4292 0.6145 0.1983 0.0934 0.4292 0.6302 0.1994
GT=1
GCN-Align 0.5258 0.8866 0.8866 0.6528 0.5258 0.8969 0.8969 0.6545 0.0150 0.1175 0.4188 0.0538 0.0150 0.1175 0.4188 0.0538
RDGCN 0.4742 0.8351 0.8351 0.5973 0.4742 0.8351 0.8351 0.5981 0.0641 0.2137 0.6175 0.1174 0.0641 0.2137 0.6197 0.1175
MLP 0.0000 0.2165 0.2165 0.0802 0.0000 0.2577 0.2577 0.0963 0.0406 0.1688 0.5470 0.0846 0.0406 0.1688 0.5470 0.0846
Text-only Bi-Encoder 0.1753 0.6082 0.6082 0.3092 0.1753 0.7113 0.7113 0.3269 0.0000 0.0021 0.0192 0.0027 0.0043 0.0321 0.2179 0.0195
Cross-Attention 0.0000 0.0000 0.0000 0.0017 0.0928 0.4124 0.4124 0.1835 0.0028 0.0168 0.1592 0.0124 0.0028 0.0168 0.1676 0.0136
Procrustes 0.4536 0.8660 0.8660 0.5983 0.4536 0.8660 0.8660 0.59910.1068 0.3162 0.6496 0.1748 0.1068 0.3162 0.6496 0.1748
QCEA0.5876 0.9072 0.9072 0.6872 0.5876 0.9278 0.9278 0.68830.0705 0.2457 0.5641 0.1324 0.0705 0.2457 0.5812 0.1334
GT>1
GCN-Align 0.1111 0.7778 0.3241 0.3155 0.1111 0.7778 0.3241 0.3159 0.0905 0.5343 0.4652 0.2164 0.0905 0.5343 0.4652 0.2164
RDGCN 0.0000 0.72220.72220.1708 0.0000 0.72220.72220.1713 0.1005 0.5394 0.88110.2352 0.1005 0.5394 0.88110.2352
MLP 0.1667 0.5556 0.2917 0.2927 0.1667 0.5556 0.2917 0.2929 0.1122 0.5276 0.5200 0.2405 0.1122 0.5276 0.5200 0.2405
Text-only Bi-Encoder 0.1111 0.5000 0.3032 0.2315 0.1667 0.6111 0.3912 0.3086 0.0034 0.0184 0.0150 0.0102 0.0101 0.0653 0.1123 0.0361
Cross-Attention 0.0000 0.0556 0.0185 0.0335 0.0556 0.1111 0.0370 0.0838 0.0035 0.0502 0.1111 0.0243 0.0035 0.0606 0.1336 0.0307
Procrustes0.2778 1.00000.6389 0.5159 0.2778 1.00000.6458 0.51590.1072 0.3668 0.3618 0.1962 0.1072 0.3668 0.3618 0.1962
QCEA 0.1111 0.9444 0.5880 0.3172 0.1111 0.9444 0.5880 0.3172 0.1156 0.65490.6223 0.2685 0.1156 0.65490.6270 0.2689
on Herb, highlighting the limitation of relying solely on
semantic interaction without structural modeling.
Directionalandcardinalityanalysesfurtherclarifywhere
QCEA is most beneficial. For Symptom, QCEA achieves
thebestHit@1andMRRintheTCM→WMdirectionwhile
remaining strongest overall. For Herb with GT>1, QCEA
achieves the best Hit@1, Hit@10, and MRR, whereas
RDGCN remains strongest on Recall@100. These results
suggest that QCEA is particularly effective in ambiguous
and multi-target scenarios, where improving the top of the
ranking is more critical than expanding deeper candidate
coverage.
At the same time, QCEA does not dominate all sce-
narios. A plausible explanation is that WM→TCM of-
ten involves mapping relatively standardized biomedical
concepts to broader and more heterogeneous TCM con-
cepts, making the reverse direction intrinsically more am-
biguous and more sensitive to coarse global regularities.
GCN-Align remains stronger on TCM→WM Symptom
Hit@10/Recall@10, RDGCN on Herb Recall@100, and
ProcrustesonWM→TCMSymptom,indicatingthatsimpler
structural or linear methods can remain effective when
correspondence patterns are more standardized. Overall,the results support modeling cross-system entity alignment
as a query-conditioned ranking problem, especially when
evaluation emphasizes top-rank quality directly relevant to
downstream retrieval and RAG.
5.2. Ablation Study and Component Analysis
Table 2 summarizes the ablation results under the type-
constrainedsetting.VariantBdenotesthefullQCEAmodel,
while Variant A removes query conditioning, reducing the
model to static matching based on source-side representa-
tions.
Overall, the full model achieves the most consistent
performance acrossboth datasets.While somevariantsout-
perform on isolated metrics, such gains are not consistent,
whereas the full model remains robust.
Removing query conditioning (A) leads to the largest
andmostconsistentdegradation,confirmingthatdescription-
conditioned ranking is the primary source of improvement.
TheperformancedropisparticularlyevidentonHerb,where
ambiguity and candidate space are larger.
Removing the direction-aware Tucker projection (D)
also causes a notable decline, especially on Herb (e.g.,
Recall@100 drops from 0.6069 to 0.4847), indicating the
Y. Jiao et al.:Preprint submitted to ElsevierPage 9 of 15

Query-Conditioned Knowledge Alignment
Table 2
Ablation results undertype-constrainedevaluation. Re-
call@100isreportedforHerb.Bestresultsareinbold.Variants:
A (w/o query conditioning), B (full model), C (w/o GNN), D
(linear projection), and E (w/o residual fusion).
Symptom Herb
Method Hit@1 Hit@10 Recall@10 MRR Hit@1 Hit@10 Recall@100 MRR
Overall
A 0.3304 0.8000 0.7370 0.4911 0.0423 0.3146 0.4578 0.1249
B0.5130 0.9304 0.8746 0.6302 0.0958 0.4751 0.6069 0.2094
C 0.3217 0.7913 0.7453 0.4463 0.0732 0.3531 0.5103 0.1617
D 0.3652 0.8261 0.7685 0.5302 0.0648 0.3521 0.4847 0.1549
E 0.1043 0.5217 0.4696 0.2283 0.0704 0.3596 0.5029 0.1621
TCM→WM
A 0.4625 0.8625 0.8625 0.5971 0.0579 0.4500 0.5196 0.1721
B0.6750 0.9625 0.9625 0.76640.10000.5579 0.56470.2274
C 0.4625 0.8875 0.8875 0.5699 0.1079 0.5053 0.5222 0.2343
D 0.5000 0.9125 0.9125 0.6526 0.0974 0.5079 0.5522 0.2174
E 0.1250 0.5875 0.5875 0.27230.11580.5368 0.55350.2478
WM→TCM
A 0.0286 0.6571 0.4500 0.2488 0.0336 0.2394 0.4235 0.0987
B0.1429 0.8571 0.6738 0.3189 0.0934 0.4292 0.6302 0.1994
C 0.0000 0.5714 0.4202 0.1640 0.0540 0.2686 0.5037 0.1214
D 0.0571 0.6286 0.4393 0.2505 0.0467 0.2657 0.4473 0.1201
E 0.0571 0.3714 0.2000 0.1277 0.0453 0.2613 0.4748 0.1146
GT=1
A 0.3918 0.8144 0.8144 0.5382 0.0235 0.1197 0.3974 0.0587
B0.5876 0.9278 0.9278 0.6883 0.0705 0.2457 0.5812 0.1334
C 0.3814 0.8041 0.8041 0.4907 0.0192 0.1368 0.5192 0.0611
D 0.4124 0.8454 0.8454 0.5729 0.0256 0.1517 0.4466 0.0699
E 0.1134 0.5258 0.5258 0.2431 0.0192 0.1474 0.4829 0.0635
GT>1
A 0.0000 0.7222 0.3194 0.2374 0.0570 0.4673 0.5051 0.1768
B0.1111 0.9444 0.5880 0.3172 0.1156 0.6549 0.6270 0.2689
C 0.0000 0.7222 0.4282 0.20720.11560.5226 0.5033 0.2405
D 0.1111 0.7222 0.3542 0.3003 0.0955 0.5092 0.5147 0.2214
E 0.0556 0.5000 0.1667 0.1484 0.1106 0.5260 0.5186 0.2395
importance of direction-aware transformation for modeling
non-bijective correspondence.
Eliminating graph propagation (C) results in consistent
degradation,indicatingthatstructuralcontextprovidescom-
plementary signals beyond textual semantics.
Removing residual fusion (E) further reduces perfor-
manceinmostsettings.Theresidualbranchstabilizesrepre-
sentationlearningandmitigatesdistortionfromhigher-order
transformations.
These effects are particularly visible in challenging set-
tingssuchasWM→TCMandGT>1cases.Forexample,on
HerbWM→TCM,Recall@100dropsfrom0.6302to0.4748
without residual fusion.
In summary, query conditioning provides the largest
gain, followed by direction-aware Tucker transformation,
whilegraphpropagationandresidualfusioncontributecom-
plementary improvements in robustness and stability.
5.3. TrainingDynamicsandPerformanceAnalysis
Figure 3 shows stable optimization and consistent re-
trievaltrendsonbothdatasets.TheSymptomtaskconverges
earlierthantheHerbtask(bestepoch83vs.245),reflecting
lowerambiguityandasmallercandidatespace,whereasthe
Herb task requires more iterations due to its many-to-many
and dispersed correspondence structure.
Figure 3(b) further shows that retrieval performance
generally improves with increasing𝐾, but exhibits distinct
saturation behaviors. Symptom saturates early, indicating
that correct targets are concentrated at top ranks, whereasHerb continues to benefit from larger𝐾, reflecting more
dispersed target distributions.
As shown in Figure 3(c)–(d), directional asymmetry is
consistentlyobserved.TCM→WMoutperformsWM→TCM,
especiallyatsmall𝐾,suggestingthatmappingfromabstract
to standardized representations is relatively easier. The gap
narrows as𝐾increases, indicating that correct matches
remainpresentbutarerankedlowerinthemorechallenging
direction.
Figure 3(e)–(f) further shows that cardinality differen-
tiates performance. GT=1 cases saturate quickly, whereas
GT>1 cases benefit more from larger𝐾, particularly on
Herb, highlighting the increased difficulty of multi-target
alignment.
5.4. Impact of Seed Alignment Ratio
Figure 4 shows the effect of varying the seed alignment
ratio. Performance generally improves with increasing su-
pervision, with the largest gains in the low-resource regime
(0.1–0.4) and diminishing returns thereafter, indicating that
QCEA can establish meaningful alignment structure from
limited seeds.
Theimpactofsupervisiondiffersacrosstasks.Symptom
achieves strong performance across all ratios, reflecting its
relatively simple and near one-to-one correspondence. In
contrast, Herb benefits more from increased supervision,
showing larger gains as the seed ratio increases, while re-
maining more challenging due to its larger candidate space
and many-to-many structure. Although Recall@100 is rel-
atively high, lower Hit@1 and MRR indicate that precise
ranking remains difficult.
Overall,QCEAscaleseffectivelywithincreasingsuper-
vision and remains robust in low-resource settings, while
the main remaining challenges arise from non-bijective and
asymmetric correspondence.
5.5. Impact on Downstream RAG
Figure 5 examines the impact of alignment quality on
downstreamQAunderthecontrolledRAGsettingdescribed
inSection4.Weanalyzeoveralleffectiveness,category-wise
behavior, precision–coverage trade-offs, and the extent to
whichLLMscompensateformissingcross-systemevidence.
5.5.1. Overall comparison of downstream variants
Fig. 5(a) compares downstream QA performance under
differentalignmentsettings.Oracleservesasanupperbound
and achieves near-perfect retrieval and the strongest overall
performance, confirming that correct first-hop alignment
enables reliable and well-grounded reasoning.
Among practical variants, QCEA performs best over-
all, substantially improving retrieval, answer accuracy, and
groundedness compared with all baselines. Confidence-
basedtruncation(thebest-performingQCEA-TopXsetting)
slightly reduces retrieval quality but does not improve end-
to-end performance, indicating limited benefit from aggres-
sive candidate pruning. In contrast, removing alignment
(NoAlign) leads to near-zero performance, while LLM-
only methods achieve very low accuracy, suggesting that
Y. Jiao et al.:Preprint submitted to ElsevierPage 10 of 15

Query-Conditioned Knowledge Alignment
0 50 100 150 200 250 300
Epoch0123456Loss(a) Training and Validation Loss
symptom
herb
1 3 5 10 20 50 100 150200 300
K (log scale)0.20.40.60.81.0Value
(b) Overall Performance
Curves
Hit@K
Recall@K
1 3 5 10 20 50 100 150200 300
K (log scale)0.20.40.60.81.0Hit@K
(c) Directional Hit@K
Curves
TCM WM
WM TCM
1 3 5 10 20 50 100 150200 300
K (log scale)0.20.40.60.81.0Recall@K
(d) Directional Recall@K
Curves
TCM WM
WM TCM
1 3 5 10 20 50 100 150200 300
K (log scale)0.00.20.40.60.81.0Hit@K
(e) GT-size Hit@K
Curves
GT=1
GT>1
1 3 5 10 20 50 100 150200 300
K (log scale)0.00.20.40.60.81.0Recall@K
(f) GT-size Recall@K
Curves
GT=1
GT>1
Figure 3:Training dynamics and retrieval performance on the Symptom and Herb datasets. (a) Training and validation loss with
selected best epochs. (b) Overall Hit@K and Recall@K. (c)-(d) Directional performance (TCM→WM and WM→TCM). (e)-(f)
Performance under different ground-truth cardinalities (GT=1 vs. GT>1). Solid lines denote Symptom, dashed lines denote Herb,
and𝐾is shown on a logarithmic scale.
parametric knowledge alone is insufficient to support cross-
system reasoning.
These results suggest that accurate cross-system align-
ment is an important prerequisite for evidence-grounded
QA, particularly when reasoning requires reliable cross-
system evidence transfer.
5.5.2. Category-wise analysis
Fig. 5(b) reports end-to-end accuracy for Symptom
and Herb questions. Under Oracle, Herb questions achieve
higher accuracy, but this pattern reverses under predicted
alignment.WithQCEA,Symptomachieveshigheraccuracy,
whileHerbperformancedropsmoresubstantiallyrelativeto
Oracle.
This suggests that Herb questions are more sensitive
to alignment quality, likely due to their larger candidate
space and many-to-many correspondence, whereas Symp-
tomquestionsappearrelativelymorestableunderimperfect
alignment.
5.5.3. Effect of truncation and candidate removal
Fig. 5(c)–(d) analyzes the impact of first-hop candidate
control.Aggressivetop-𝑘truncationdegradesperformance,
while moderate truncation yields a better balance between
noise reduction and coverage. In contrast, random removal
of candidates leads to continuous degradation and near-
collapse.
These results reveal a trade-off between precision and
coverage: top-𝑘truncation can suppress noisy alignments,
butexcessivetruncationbreakscross-systemevidencechainsrequired for downstream reasoning. Strong degradation
under random removal shows that candidate coverage is
essential for preserving retrieval paths. Together, these
findings suggest that downstream performance depends not
only on graph connectivity (coverage), but also on learned
cross-system alignment quality, which determines whether
relevant evidence can be surfaced during retrieval.
5.5.4. LLM-only baselines and grounded evidence
Fig. 5(e) compares retrieval-based methods with LLM-
onlybaselines.AllLLM-onlyvariantsperformsubstantially
worsethanretrieval-augmentedmethods,withloweranswer
accuracy and significantly higher hallucination rates. Even
with increased model capacity or domain specialization,
LLM-only approaches remain unable to match the perfor-
mance of alignment-enhanced RAG.
Theseresultsindicatethatreliableperformancedepends
on grounded cross-system evidence rather than parametric
knowledge alone [19]. Without explicit access to aligned
cross-system evidence, LLMs fail to consistently retrieve
and integrate relevant information for multi-hop reasoning,
leading to unstable and ungrounded outputs. Consequently,
alignment-enhancedRAGimprovesreliabilitybygrounding
generation in aligned cross-system evidence.
5.5.5. Case Study
To further illustrate the mechanisms behind the quanti-
tative results, we present a representative example.
Query.Whichmodernmolecularcomponentscorrespondto
theTCMherbMimenghua(BuddlejaeFlos)?(Listupto3.)
Y. Jiao et al.:Preprint submitted to ElsevierPage 11 of 15

Query-Conditioned Knowledge Alignment
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Seed ratio0.00.20.40.60.81.0Score
(a) Symptom
Recall@10
Hit@1
Hit@10
MRR
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Seed ratio0.00.10.20.30.40.50.6Score
(b) Herb
Recall@100
Hit@1
Hit@10
MRR
Figure 4:Impact of seed alignment ratio on (a) Symptom
and (b) Herb tasks. Performance improves with increasing
supervision. Recall@100 is used for Herb.
NoAlign.The retriever remains within the TCM subgraph
and fails to access cross-system alignment edges, resulting
in missing evidence and ungrounded answers.
QCEA.QCEA retrieves multiple cross-system candidates
(e.g.,Luteolin,Apigenin,Protocatechuic Acid), enabling
groundedreasoningandproducingcorrectoutputs,although
the candidate set remains relatively broad.
QCEA-TopX.Truncationremovesusefullower-rankedcan-
didates, reducing retrieval coverage without improving an-
swer quality, consistent with the precision–coverage trade-
off observed in Fig. 5(c).
5.5.6. Summary of downstream RAG findings
Thedownstreamresultssupportfourmainobservations.
First, accurate cross-system alignment appears essential
for evidence-grounded QA. Second, QCEA provides the
strongest overall performance among practical settings,
while truncation does not yield additional gains. Third,
the impact of alignment is category-dependent, with herb
questionsbeingmoresensitivetoalignmentquality.Fourth,
preserving candidate coverage is critical, as both truncation
andcandidateremovaldegradeperformance,whereasLLM-
only methods remain far behind.Overall, alignment quality consistently affects retrieval
effectiveness, grounding, and end-to-end reliability in our
experiments.
6. Discussion and Implications
Theexperimentalresultsconsistentlyindicatethatcross-
domainalignmentqualityplaysacentralroleindownstream
performance across multiple dimensions. QCEA improves
top-ranked metrics such as Hit@1, Hit@10, and MRR,
particularly in the Symptom task, suggesting that query-
conditionedalignmenteffectivelyconcentratesrelevantcan-
didates near the top of the ranking.
The ablation study further highlights that query condi-
tioning is the dominant factor driving performance, while
direction-aware transformation and graph propagation pro-
videcomplementaryimprovements,especiallyinmorechal-
lenging settings with asymmetric mappings. Training dy-
namics reveal structural differences between tasks, with the
Herbtaskexhibitingslowerconvergenceduetoitsmany-to-
many correspondence nature.
Beyond alignment metrics, downstream RAG evalua-
tiondemonstratesthatalignmentqualityconsistentlyaffects
answer accuracy, grounding, and overall reliability. When
alignment is degraded, the system exhibits reduced ground-
ing and increased instability, whereas improved alignment
leads to more consistent and evidence-based outputs.
Theseobservationssuggestthatalignmentcanbeviewed
as a structural layer that influences how knowledge is ac-
cessed and organized for reasoning. Rather than acting as a
standalonereasoningmechanism,alignmentshapesthecan-
didate space from which reasoning systems draw evidence.
In this sense, improvements in alignment quality tend to
propagate to downstream reasoning performance.
Alignment as an upstream bottleneck.Our RAG exper-
imentsrevealthatalignmentactsasanupstreambottleneck:
errors in alignment propagate directly into retrieval, and
are difficult to correct by the generation model. Under a
controlled generation setting, the model (Qwen2.5-1.5B-
Instruct) is unable to compensate for missing or misaligned
cross-system evidence. Consistently, improving alignment
qualityleadstobetterend-to-endperformance,whiledegra-
dation in alignment results in corresponding performance
drops.
Failure case analysis.Despite overall improvements,
QCEA exhibits limitations in specific scenarios. For herb–
molecule alignment with GT>1, recall remains moderate
(0.6270) compared to RDGCN (0.8811). This occurs when
a TCM herb corresponds to a large set of WM molecules
with no clear semantic distinction in textual descriptions,
leading to ranked but not highly concentrated predictions.
Additionally, WM→TCM alignment remains challenging
duetotheinherentasymmetryinabstractionlevelsbetween
the two medical systems.
Y. Jiao et al.:Preprint submitted to ElsevierPage 12 of 15

Query-Conditioned Knowledge Alignment
Oracle QCEA QCEA-top5 QCEA-drop10 NoAlign OnlyLLM
(gpt-oss-20b)0.00.10.20.30.40.50.60.70.80.91.01.1End-to-End Accuracy (E2E)
0.4375
0.3975
0.3983
0.0000
N/A0.4325
N/A0.7575
0.5275
0.5275
0.4875
0.0175
0.04250.7150
0.4675
0.4675
0.4325
0.0000
0.04250.6475
0.6475
0.6125
0.0000
N/A0.2825
0.5325
0.5325
0.5675
0.71251.0000 1.0000 0.9975 1.0000
0.92500.9525
Symptom Herb0.00.20.40.60.81.0End-to-End Accuracy (E2E)0.62000.8100
0.5200
0.41500.5200
0.4150
0.0750
0.0100Oracle
qcea_top5
qcea
only-gpt-oss-20b
qcea top1 top2 top3 top4 top5 top6 top70.00.10.20.30.40.5End-to-End Accuracy (E2E)
0.4675
0.31250.35000.42750.45750.4675 0.4675 0.4675
0.3975
0.27830.33540.36210.37460.3975 0.3975 0.3975
top5 matches QCEA
E2E
R@5
drop10 drop20 drop30 drop40 drop50 drop60 drop70 drop80 drop90drop1000.00.10.20.30.40.5
0.4325
0.4100
0.3700
0.3400
0.31250.3300
0.2275
0.2075
0.1300
0.00000.3621
0.3333
0.2946
0.2608
0.22420.2367
0.1437
0.1217
0.0617
0.0000steady degradation,
severe failure > drop70
E2E
R@5
only-gpt-
oss-20bonly-qwen only-
medreason-8bonly-cmlm-
zhongjingqcea_top50.00.20.40.60.81.0Metric Value
0.0425 0.0325 0.0250 0.01250.46750.71250.9675 0.9750
0.8800
0.5325E2E Hallucination(a) Main Metrics Comparison Across Alignment Settings
(b) Alignment Gains on Symptom and Herb
(c) T op-k Truncation (d) Edge Drop Ratio (e) Scaling vs Grounded RetrievalEvidence Recall@10
Cross-System Hit RateAnswer Accuracy
End-to-End AccuracyGroundedness
Hallucination Rate
Figure 5:Downstream RAG evaluation under different alignment settings. (a) Overall comparison in terms of retrieval-level
(evidence recall@K, cross-system hit rate), generation-level (answer accuracy), and end-to-end metrics (groundedness, end-to-
end accuracy). (b) Category-wise end-to-end accuracy for Symptom and Herb questions. (c) Effect of confidence-based top-𝑘
truncation of first-hop alignment candidates. (d) Effect of random removal of first-hop alignment candidates. (e) Comparison
between retrieval-augmented settings and LLM-only baseline.
Limitations.Our conclusions are based on empirical
evaluation and do not by themselves establish causal re-
lationships between alignment and downstream reasoning.
Additionally, experiments are conducted in a transductive
setting and focus on TCM–WM integration, which may
limit generalization to other domains. Finally, the use of a
lightweight language model in RAG evaluation emphasizes
the impact of alignment quality, but results may vary with
larger models.
Implications.These findings suggest that improving
cross-domainalignmentisapracticalandeffectivedirection
for enhancing knowledge integration systems. Future work
may explore hybrid approaches that combine structured
alignment with generative models, as well as extensions to
broader multi-domain knowledge settings.
7. Conclusions
We presented QCEA, a query-conditioned entity align-
ment framework for bridging heterogeneous TCM and WM
knowledge graphs. By reformulating alignment as a query-
conditionedrankingproblem,QCEAcapturescontext-dependent,
asymmetric, and non-bijective correspondence in cross-
systemmedicalknowledge.ExperimentsonSymMap-derived
datasets demonstrate consistent improvements over rep-
resentative baselines, with notable gains in semantically
ambiguous settings and rank-sensitive retrieval scenarios.
Further downstream RAG evaluation shows that improved
alignment enhances the availability and ranking of cross-
systemevidence,leadingtoimprovedretrievaleffectiveness,grounding,andend-to-endreliability,whilereducingunsup-
ported or inconsistent outputs in cross-system medical QA.
Overall, these results suggest that query-conditioned
alignment provides an effective and practical approach
for integrating heterogeneous medical knowledge and sup-
porting knowledge-grounded reasoning. Future work will
explore incorporating richer clinical context, improving ro-
bustnessundersparsesupervision,andextendingalignment
to more complex multi-hop reasoning settings.
Data Availability
The data used in this study are derived from the pub-
liclyavailableSymMapdatabase.Theprocessedbenchmark
datasets and evaluation splits are available from the corre-
sponding author upon reasonable request.
Declaration of Competing Interest
Theauthorsdeclarethattheyhavenoknowncompeting
financial interests or personal relationships that could have
appeared to influence the work reported in this paper.
References
[1] Abu-Salih,B.,Al-Qurishi,M.,Alweshah,M.,Al-Smadi,M.,Alfayez,
R., Saadeh, H., 2023. Healthcare knowledge graph construction: A
systematic review of the state-of-the-art, open issues, and opportuni-
ties. Journal of Big Data 10, 81.
[2] Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman,
F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.,
2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 .
Y. Jiao et al.:Preprint submitted to ElsevierPage 13 of 15

Query-Conditioned Knowledge Alignment
[3] Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus,
E.,Arora,R.K.,Bai,Y.,Baker,B.,Bao,H.,etal.,2025. gpt-oss-120b
& gpt-oss-20b model card. arXiv preprint arXiv:2508.10925 .
[4] Bodenreider, O., 2004. The unified medical language system
(UMLS): integrating biomedical terminology. Nucleic Acids Re-
search 32, D267–D270.
[5] Chen,M.,Tian,Y.,Yang,M.,Zaniolo,C.,2017. Multilingualknowl-
edge graph embeddings for cross-lingual knowledge alignment, in:
Proceedings of the 26th International Joint Conference on Artificial
Intelligence (IJCAI), pp. 1511–1517.
[6] Cheng,J.,Lu,C.,Yang,L.,Chen,G.,Zhang,F.,2025.EasyEA:Large
languagemodelisallyouneedinentityalignmentbetweenknowledge
graphs,in:FindingsoftheAssociationforComputationalLinguistics:
ACL 2025, pp. 20981–20995.
[7] Conneau,A.,Lample,G.,Ranzato,M.,Denoyer,L.,Jégou,H.,2018.
Word translation without parallel data, in: International Conference
on Learning Representations (ICLR).
[8] Guo,J.,Fan,Y.,Ai,Q.,Croft,W.B.,2016.Adeeprelevancematching
modelforad-hocretrieval,in:Proceedingsofthe25thACMInterna-
tional Conference on Information and Knowledge Management, pp.
55–64.
[9] Hao,J.,Lei,C.,Efthymiou,V.,Quamar,A.,Özcan,F.,Sun,Y.,Wang,
W., 2021. Medto: Medical data to ontology matching using hybrid
graph neural networks, in: Proceedings of the 27th ACM SIGKDD
ConferenceonKnowledgeDiscovery&DataMining,pp.2946–2954.
[10] Karpukhin,V.,Oguz,B.,Min,S.,Lewis,P.,Wu,L.,Edunov,S.,Chen,
D.,Yih,W.t.,2020.Densepassageretrievalforopen-domainquestion
answering, in: Proceedings of the 2020 Conference on Empirical
MethodsinNaturalLanguageProcessing(EMNLP),pp.6769–6781.
[11] Khattab, O., Zaharia, M., 2020. ColBERT: Efficient and effective
passage search via contextualized late interaction over BERT, in:
Proceedings of the 43rd International ACM SIGIR Conference on
Research and Development in Information Retrieval, pp. 39–48.
[12] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P.,
Maschinot, A., Liu, C., Krishnan, D., 2020. Supervised contrastive
learning, in: Advances in Neural Information Processing Systems
(NeurIPS).
[13] Kingma,D.P.,Ba,J.,2014. Adam:Amethodforstochasticoptimiza-
tion. arXiv preprint arXiv:1412.6980 .
[14] Kipf, T.N., Welling, M., 2017. Semi-supervised classification with
graphconvolutionalnetworks,in:InternationalConferenceonLearn-
ing Representations (ICLR).
[15] Kolda,T.G.,Bader,B.W.,2009. Tensordecompositionsandapplica-
tions. SIAM Review 51, 455–500.
[16] Konopasky,A.,Artino,A.R.,Battista,A.,Ohmer,M.,Hemmer,P.A.,
Torre,D.,Ramani,D.,VanMerrienboer,J.,Teunissen,P.W.,McBee,
E., et al., 2020. Understanding context specificity: the effect of
contextual factors on clinical reasoning. Diagnosis 7, 257–264.
[17] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal,
N., Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., et al., 2020.
Retrieval-augmented generation for knowledge-intensive NLP tasks.
AdvancesinNeuralInformationProcessingSystems33,9459–9474.
[18] Liu, F., Shareghi, E., Meng, Z., Basaldella, M., Collier, N., 2021.
Self-alignment pretraining for biomedical entity representations, in:
Proceedings of the 2021 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language
Technologies, pp. 4228–4238.
[19] Mallen, A., Asai, A., Zhong, V., Das, R., Khashabi, D., Hajishirzi,
H., 2022. When not to trust language models: Investigating effec-
tiveness of parametric and non-parametric memories. arXiv preprint
arXiv:2212.10511 .
[20] Matos, L.C., Machado, J.P., Monteiro, F.J., Greten, H.J., 2021. Un-
derstandingtraditionalchinesemedicinetherapeutics:anoverviewof
the basics and clinical applications. Healthcare 9, 257.
[21] Mucheng, R., Heyan, H., Yuxiang, Z., Qianwen, C., Yuan, B., Yang,
G., 2022. TCM-SD: A benchmark for probing syndrome differen-
tiation via natural language processing, in: Proceedings of the 21stChineseNationalConferenceonComputationalLinguistics,pp.908–
920.
[22] Nicholson,D.N.,Greene,C.S.,2020. Constructingknowledgegraphs
and their biomedical applications. Computational and Structural
Biotechnology Journal 18, 1414–1428.
[23] Oord, A.v.d., Li, Y., Vinyals, O., 2018. Representation learning with
contrastive predictive coding. arXiv preprint arXiv:1807.03748 .
[24] Organization, W.H., 2019. WHO global report on traditional and
complementary medicine 2019. World Health Organization.
[25] Sun, D.z., Li, S.d., Liu, Y., Zhang, Y., Mei, R., Yang, M.h., 2013.
Differencesintheoriginofphilosophybetweenchinesemedicineand
western medicine: exploration of the holistic advantages of chinese
medicine. Chinese Journal of Integrative Medicine 19, 706–711.
[26] Sun,Z.,Hu,W.,Li,C.,2017. Cross-lingualentityalignmentviajoint
attribute-preserving embedding, in: Proceedings of the International
Semantic Web Conference, pp. 628–644.
[27] Sun, Z., Hu, W., Zhang, Q., Qu, Y., 2018. Bootstrapping entity
alignment with knowledge graph embedding, in: Proceedings of the
27thInternationalJointConferenceonArtificialIntelligence(IJCAI),
pp. 4396–4402.
[28] Sutton,R.T.,Pincock,D.,Baumgart,D.C.,Sadowski,D.C.,Fedorak,
R.N., Kroeker, K.I., 2020. An overview of clinical decision support
systems: benefits, risks, and strategies for success. npj Digital
Medicine 3, 17.
[29] Wang,L.,Yang,N.,Huang,X.,Yang,L.,Majumder,R.,Wei,F.,2024.
Multilingual E5 text embeddings: A technical report. arXiv preprint
arXiv:2402.05672 .
[30] Wang,Z.,Lv,Q.,Lan,X.,Zhang,Y.,2018. Cross-lingualknowledge
graph alignment via graph convolutional networks, in: Proceedings
of the 2018 Conference on Empirical Methods in Natural Language
Processing, pp. 349–357.
[31] Wu, J., Deng, W., Li, X., Liu, S., Mi, T., Peng, Y., Xu, Z., Liu,
Y., Cho, H., Choi, C.I., et al., 2025. Medreason: Eliciting factual
medicalreasoningstepsinllmsviaknowledgegraphs. arXivpreprint
arXiv:2504.00993 .
[32] Wu, Y., Liu, X., Feng, Y., Wang, Z., Zhao, D., 2019a. Relation-
aware entity alignment for heterogeneous knowledge graphs, in:
Proceedings of the 28th International Joint Conference on Artificial
Intelligence (IJCAI), pp. 5278–5284.
[33] Wu,Y.,Zhang,F.,Yang,K.,Fang,S.,Bu,D.,Li,H.,Sun,L.,Hu,H.,
Gao,K.,Wang,W.,etal.,2019b. SymMap:anintegrativedatabaseof
traditionalchinesemedicineenhancedbysymptommapping. Nucleic
Acids Research 47, D1110–D1117.
[34] Xiang,Y.,Zhang,Z.,Chen,J.,Chen,X.,Lin,Z.,Zheng,Y.,2021.On-
toEA: Ontology-guided entity alignment via joint knowledge graph
embedding, in: Findings of the Association for Computational Lin-
guistics: ACL-IJCNLP 2021, pp. 1117–1128.
[35] Xu, R., Jiang, P., Luo, L., Xiao, C., Cross, A., Pan, S., Sun, J.,
Yang, C., 2025. A survey on unifying large language models and
knowledgegraphsforbiomedicineandhealthcare,in:Proceedingsof
the 31st ACM SIGKDD Conference on Knowledge Discovery and
Data Mining, pp. 6195–6205.
[36] Xue,X.,Wu,M.E.,Khan,F.,2024. Biomedicalinformationintegra-
tionviaadaptivelargelanguagemodelconstruction. IEEEJournalof
Biomedical and Health Informatics 29, 6381–6394.
[37] Yang,A.,Yang,B.,Zhang,B.,Hui,B.,Zheng,B.,Yu,B.,Li,C.,Liu,
D.,Huang,F.,Wei,H.,etal.,2024a. Qwen2.5technicalreport. arXiv
preprint arXiv:2412.15115 doi:10.48550/arXiv.2412.15115.
[38] Yang,H.W.,Zou,Y.,Shi,P.,Lu,W.,Lin,J.,Sun,X.,2019. Aligning
cross-lingual entities with multi-aspect information, in: Proceedings
of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural
Language Processing (EMNLP-IJCNLP), pp. 4431–4441.
[39] Yang, L., Zhou, S., Cheng, J., Zhang, F., Wan, J., Wang, S., Lee, M.,
2025. DAEA: Enhancing entity alignment in real-world knowledge
graphs through multi-source domain adaptation, in: Proceedings of
the 31st International Conference on Computational Linguistics, pp.
5890–5901.
Y. Jiao et al.:Preprint submitted to ElsevierPage 14 of 15

Query-Conditioned Knowledge Alignment
[40] Yang, S., Zhao, H., Zhu, S., Zhou, G., Xu, H., Jia, Y., Zan, H.,
2024b. Zhongjing: Enhancing the chinese medical capabilities of
large language model through expert feedback and real-world multi-
turn dialogue, in: Proceedings of the AAAI conference on artificial
intelligence, pp. 19368–19376.
[41] Yu, X., Wu, S., Zhang, J., Hu, Y., Luo, M., Zhao, H., Song, X.,
Chen, Y., Wang, X., 2023. Developing TCM clinical practice guide-
lines:acomparisonbetweentraditionalchinesemedicineandwestern
medicine. Integrative Medicine Research 12, 100952.
[42] Zhang, Q., Sun, Z., Hu, W., Chen, M., Li, C., 2019. Multi-view
knowledge graph embedding for entity alignment, in: Proceedings
of the 28th International Joint Conference on Artificial Intelligence
(IJCAI), pp. 5145–5151.
[43] Zhang,R.,Su,Y.,Trisedya,B.D.,Zhao,X.,Yang,M.,Cheng,H.,Qi,
J., 2023. AutoAlign: Fully automatic and effective knowledge graph
alignment enabled by large language models. IEEE Transactions on
Knowledge and Data Engineering 36, 2357–2371.
[44] Zhu, B., Wang, R., Wang, J., Shao, F., Wang, K., 2024. A survey:
knowledge graph entity alignment research based on graph embed-
ding. Artificial Intelligence Review 57, 229.
Y. Jiao et al.:Preprint submitted to ElsevierPage 15 of 15