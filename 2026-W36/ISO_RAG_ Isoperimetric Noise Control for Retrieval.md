# ISO-RAG: Isoperimetric Noise Control for Retrieval-Augmented Generation

**Authors**: Siyuan Zhang, Hanchen Wang, Dong Wen, Ying Zhang, Wenjie Zhang

**Published**: 2026-09-01 00:27:33

**PDF URL**: [https://arxiv.org/pdf/2609.00513v1](https://arxiv.org/pdf/2609.00513v1)

## Abstract
Retrieval-Augmented Generation (RAG) mitigates large language models (LLMs) hallucinations, yet conventional dense retrieval struggles with the complex reasoning paths of multi-hop question answering (QA). Graph-based RAG captures multi-step relationships but suffers from severe semantic drift and high online latency due to noisy global graph traversals. Thus, we propose ISO-RAG (ISOperimetric Retrieval-Augmented Generation), a geometry-aware RAG framework. By projecting the underlying knowledge graph into a hyperbolic Poincare ball to precompute node-wise isoperimetric profiles, ISO-RAG prunes spurious edges during retrieval, restricting the search space to a strictly localized subgraph. This topological purification regulates Personalized PageRank (PPR) diffusion driving the retrieval process, ensuring exact and low-latency convergence. Experiments on multi-hop QA benchmarks demonstrate that ISO-RAG outperforms state-of-the-art baselines by average absolute gains of 10.0% in retrieval recall and 4.3% in downstream exact match, achieving a superior accuracy-efficiency trade-off by fundamentally eliminating the latency bottleneck of global traversals. Our source code is available at https://github.com/ZaiizaiZHANG/ISO-RAG.

## Full Text


<!-- PDF content starts -->

ISO-RAG: Isoperimetric Noise Control for Retrieval-Augmented Generation
Siyuan Zhang1, Hanchen Wang1, Dong Wen2, Ying Zhang1, Wenjie Zhang2
1University of Technology Sydney2University of New South Wales
Abstract
Retrieval-AugmentedGeneration(RAG)mitigateslargelan-
guagemodels(LLMs)hallucinations,yetconventionaldensere-
trievalstruggleswiththecomplexreasoningpathsofmulti-hop
questionanswering(QA).Graph-basedRAGcapturesmulti-
step relationships but suffers from severe semantic drift and
high online latency due to noisy global graph traversals. Thus,
we proposeISO-RAG(ISOperimetricRetrieval-Augmented
Generation), a training-free, purely topology-driven RAG
framework. Breaking away from computationally expensive
continuous geometric embeddings, ISO-RAG leverages dis-
cretegraphtheory,specificallythelocalCheegerratio(topo-
logicalexpansionrate),tocomputenode-wiseisoperimetric
profiles.Byidentifyingandpruningspuriousshortcutedges
that lead to combinatorial explosion, ISO-RAG restricts the
search space to a strictly localized, contextually safe subgraph.
This topological purification regulates deterministic Person-
alized PageRank (PPR) diffusion during retrieval, ensuring
exact and low-latency convergence without probability leak-
age.Experimentsonmulti-hopQAbenchmarksdemonstrate
that ISO-RAG outperforms state-of-the-art baselines by av-
erage absolute gains of 10.0% in retrieval recall and 4.3%
in downstream exact match, achieving a superior accuracy-
efficiencytrade-offbyfundamentallyeliminatingthelatency
bottleneck of global traversals. Our source code is available at
https://github.com/ZaiizaiZHANG/ISO-RAG.git.
1 Introduction
Retrieval-AugmentedGeneration(RAG)(Lewisetal.2020;
Guuetal.2020)isastandardparadigmforimprovingthefactu-
alityandcontrollabilityoflargelanguagemodels(LLMs)(Ka-
plan et al. 2020; Vaswani et al. 2017). Grounding generation
inexternalevidencesubstantiallyreduceshallucinations(Ji
etal.2023;Gaoetal.2023;Jiangetal.2023)andimprovesan-
swerreliability.Whileeffectiveforsingle-hopfactuallookup,
RAG remains less reliable for multi-hop question answering,
which requires connecting multiple evidence pieces scattered
across different documents. In such settings, retrieval quality
becomesthedominantbottleneck,asLLMscannotreliably
inferanswers(Weietal.2022;Yaoetal.2022,2023)without
the complete reasoning chain.
Existingsparseanddenseretrievers(e.g.,BM25(Robert-
son, Zaragoza et al. 2009), MDR (Xiong et al. 2021),
(Karpukhin et al. 2020)) rely on flat similarity matching,
often failing to reconstruct the full reasoning chains requiredbycompositionalbenchmarks(Trivedietal.2022;Chenetal.
2023). To address this, recent graph-based (Edge et al. 2024;
Jimenez Gutierrez et al. 2024; Cao et al. 2025) methods
organize corpora into interconnected structures for multi-hop
evidence aggregation. Notably, HyperbolicRAG (Cao et al.
2025)embedsdocumentnetworksintocontinuoushyperbolic
spaces to model their inherent scale-free hierarchies. This
non-Euclideanmappingenablescapturingcomplexmulti-hop
dependencies with low structural distortion.
Despitetheseadvantages,conventionalgraphparadigms
lack explicit topological intervention, manifesting in three
limitations (Figure 1): (1) Unconstrained diffusion leads to
semantic drift. Graph-based frameworks (Edge et al. 2024;
Jimenez Gutierrez et al. 2024; Cao et al. 2025) utilizing
PersonalizedPageRank(PPR)(Page etal.1999) propagate
probabilitiesoverdensegraphs.AsillustratedbytheMuSiQue
query,answeringrequirestraversingaspecifictwo-hoppathto
thetargetentity(e.g.,PoptropicatoPearson Education).How-
ever, unconstrained PPR retrieves misleading cues through
spuriousedges(e.g.,releaseyear2007).Becausethesedis-
tractors exhibit high textual overlap with the query context,
they bypass downstream re-rankers, causing the LLM to
hallucinate the incorrect year. Crucially, if unconstrained
diffusionbreaks down ona meretwo-hoppath,this seman-
tic drift compounds for more complex three- or four-hop
queries. (2) Continuous models fail to prune noise. Methods
like HyperbolicRAG (Cao et al. 2025) rely on continuous
node embeddings without explicitly pruning noisy edges
(depicted as the scissors in Figure 1), thereby retaining erro-
neouspathwaystodistractors.Thisfailstoisolatethespecific
reasoningbranchesnecessaryforaccuratemulti-hopdeduc-
tion.(3)Densegraphsdegradeefficiency.Computingrandom
walksovertheentiregraphexploresunrelatedentities(e.g.,
distantbrandslikePepsi-Cola),whichincurssignificantcom-
putational overhead and dilutes the probability mass of the
actualtargetentity.Consequently,theseparadigmsstruggle
to balance signal fidelity and retrieval latency.
In response to these limitations, we proposeISO-
RAG(ISOperimetricRetrieval-AugmentedGeneration),a
geometry-awarelocalgraphretrievalframeworkformulti-hop
QA. The core intuition of ISO-RAG is transitioning from
unconstrainedprobabilitydiffusiontogeometricallyregulated
local diffusion. After routing a query to semantically aligned
seednodestoformacandidatesubgraph,theframeworkmaps
arXiv:2609.00513v1  [cs.AI]  1 Sep 2026

Figure 1: Conceptual comparison of RAG paradigms. Top: Conventional graph-based RAG suffers from semantic drift:
unconstraineddiffusionallowsretrievalsignalstoleakintoirrelevantnoise(bluenodes)andmisleadingcues(rednodes).Bottom:
ISO-RAG leverageshyperbolic hierarchy and anisoperimetric control mechanism(scissors) to explicitly severthese erroneous
branches. This topological intervention isolates a pure reasoning path (green nodes).
these nodes into a hyperbolic space using the Poincaré ball
model (Nickel and Kiela 2017; Chami et al. 2019; Balazevic,
Allen,andHospedales2019;Gulcehreetal.2019;Nickeland
Kiela 2018;Peng et al.2022). Documentnetworksin multi-
hopQAnaturallyformhierarchicalstructures,whereafew
generic entitiesact asdense hubsconnecting numerous spe-
cific facts.Because hyperbolic spaceexpandsexponentially,
it embeds these scale-free topologies with low geometric
distortion. Crucially, this non-Euclidean embedding struc-
turallyhighlightsintrinsicbottlenecks(AlonandYahav2021;
Topping et al. 2022), such as the dense generic hubs (Girvan
and Newman 2002) that misguide retrieval. In spectral graph
theory,theclassicalCheegerconstant(Chung1997)identifies
global graph bottlenecks. Building on this geometric founda-
tion,ISO-RAGintroducesalocalizedisoperimetriccontrol
mechanism (Andersen, Chung, and Lang 2006; Krioukov
etal.2010).Becausecomputingastrictset-levelisoperimetric
constantiscomputationallyprohibitivefordynamiconlinere-
trieval,wedesignanumericallystable,node-wiseproxy.This
proxyactsasastructuralfilter,explicitlyidentifyingandsever-
ingincompatibleedgespriortopropagation.Byanchoringthe
PersonalizedPageRankdiffusiontoinitialquery-alignedseed
passages and executing it strictly within this geometrically
bounded subgraph, the framework mitigates semantic drift
and facilitates noise-controlled evidence aggregation.
Our contributions are summarized as follows.(1)We
proposeISO-RAG,anovelgeometry-awareRAGframework
that uses explicit topological control to mitigate spurious
diffusion.(2)At the core of this framework, we introduce
anisoperimetriccontrolmechanismthatexplicitlyprunes
misleadingconnectionstodensehubspriortodiffusion.(3)
Extensive evaluations demonstrate that ISO-RAG achieves
ahighlyfavorablebalancebetweenretrievalefficiencyand
downstream QA performance, delivering robust average
absolute gains of nearly 10% in Recall@5 and 4.3% in Exact
Match over competitive baselines.2 Related Works
Sparse and Dense Retrieval for Multi-Hop QA.Con-
ventionalretrievalencompassessparsemethodslikeBM25
(Robertson, Zaragoza et al. 2009) and dense models ranging
from flat bi-encoders (Karpukhin et al. 2020) to multi-hop
extensionslikeMDR(Xiongetal.2021).Whetherutilizing
exactkeywordmatchingorcosinesimilarity,theseapproaches
operate in fundamentally flat search spaces. Compressing
documents into isolated points optimized for direct semantic
overlap,densevectorscannotexplicitlymodelrelationships
between intermediate entities. This structural limitation frag-
ments reasoning by allowing lexically similar yet logically
disconnecteddistractorstoovershadowcriticalevidence.Con-
sequently, flat search spaces struggle with the compositional
reasoningpathsrequiredbyincreasinglydifficultmulti-hop
QAdatasetslikeHotpotQA(Yangetal.2018),2WikiMulti-
hopQA (Ho et al. 2020), and MuSiQue (Trivedi et al. 2022).
Conventional Graph-based RAG Systems.To overcome
flat semantic spaces, graph-based retrieval structures corpora
into networks capturing multi-hop dependencies. Notable
architectures include GraphRAG (Edge et al. 2024) utiliz-
ing hierarchical summaries with optional heuristic routing,
LightRAG (Guo et al. 2024) employing dual-level structures,
andHippoRAG2 (JimenezGutierrezetal. 2025)leveraging
neurobiologically-inspired memory networks with contin-
uous activation spreading. Despite improving recall, these
baselines rely on heuristic edge weighting and unconstrained
probabilitydiffusion;forinstance,HippoRAG2executesunre-
strictedglobalPPR.Consequently,thisuncontrolleddiffusion
introducesseveretopologicalnoiseduringretrieval(Alonand
Yahav2021;Toppingetal.2022).Withoutrigorousmathe-
maticalboundspruningthesearchspace,theseframeworks
inevitablyretrievespurioussubgraphsandmisleadingentities
before generation.
Hyperbolic Geometry and Continuous Aggregation.
The inherent hierarchical structure of knowledge graphs

makes them poorly suited for Euclidean embeddings (Nickel
and Kiela 2017; Sala et al. 2018). Foundational graph neu-
ral networks therefore extend representation learning into
hyperbolicspacethrougharchitectureslikeHGCN(Chami
etal.2019)andrelatedhyperbolicnetworks(Gulcehreetal.
2019;Pengetal.2022).Mainstreammethodologiesusethe
Poincar’e ball model (Nickel and Kiela 2017) for its intuitive
conformal geometry, or the Lorentz model (Nickel and Kiela
2018)foritsnumericaladvantagesindistanceoptimization.
Both models provide exponential capacity to embed complex
networks with minimal structural distortion. Recent works
likeHyperbolicRAG(Caoetal.2025)attempttoleveragethis
by introducing hyperbolic representations into RAG. How-
ever, despite operating in hyperbolic space, its probability
diffusion remains fundamentally unconstrained. By perform-
ing continuous neighborhood aggregation without explicit
discrete pruning, HyperbolicRAG fails to resolve the topo-
logical bottleneck, inevitably accumulating topological noise
and causing severe semantic drift during retrieval.
3 Methodology
We presentISO-RAG(ISOperimetricRetrieval-Augmented
Generation),a geometry-aware localgraphretrievalframe-
work for multi-hop question answering. The core idea is to
avoid broad graph-wide diffusion by combining four compo-
nents:(i)query-awareseedrouting,(ii)localcandidategraph
construction, (iii) isoperimetric structural filtering derived
fromhyperbolicstructure,and(iv)deterministicpersonalized
PageRank on the filtered local graph.
3.1 Problem Formulation
LetC={p 1, p2, . . . , p N}denoteacorpusofpassages.Given
a multi-hop query q, the retrieval objective is to extract a
top-Ksubset RK(q)⊂ C, such that the retrieved passages
jointly cover the evidence required to answerq.
We model the corpus as an undirected retrieval graph
G= (V,E) , where each node u∈ Vrepresents a passage
equipped with a dense semantic embedding hu∈Rd.G
serves as a question-induced passage co-occurrence graph:
an edge (u, v)∈ E is established whenever passages uandv
co-occur within a training instance or retrieval context.
Such co-occurrence graphs effectively expose latent multi-
hop dependencies, but they also inherently introduce noisy
shortcuts and hub-like regions. As a result, unconstrained
diffusionprocessescanprematurelyleakprobabilitymassinto
generic yet weakly useful passages, reducing both retrieval
precision and downstream QA performance.
3.2 Seeded Local Graph Construction
Givenaquery q,ISO-RAGfirstretrievesitsdenseembedding
hqfromaprecomputedembeddingcacheandmeasuresits
cosine similarity to every node embedding:
su(q) = cos(h q,hu), u∈ V.(1)
LetS(q)denote the set of top- mseed nodes selected
according tos u(q):
S(q) = top-m
u∈Vsu(q).(2)Rather than propagating over the full graph, ISO-RAG
restricts the search space to a compact local candidate set
Vloc(q)⊆ V. To ensure both high recall and structural con-
nectivity,weconstructthissetbyintegratingtwocomponents:
a dense semantic pool and a topology-aware neighborhood.
Specifically, let Vdense(q)denote the top- Lpassages re-
trievedvia su(q)(where L≥m,thuscontainingtheseedset
S(q)).LetVexpand (q)denotethe k-hopstructuralneighbor-
hood expanded from S(q). The local candidate set is defined
as their union:
Vloc(q) =V dense(q)∪ V expand (q).(3)
The induced local subgraph isG loc(q) =G[V loc(q)].
Thisquery-conditionedlocalgraphsubstantiallyreduces
thesearchspaceand transformsretrievalfromglobaldiffu-
sion vulnerable to topological noise into local propagation
anchored at semantically aligned entry points.
3.3 Hyperbolic Isoperimetric Edge Filtering
Hyperbolic structural signal.To characterize local graph
structure, we map node representations into the Poincaré
ball (Nickel and Kiela 2017; Chami et al. 2019; Balazevic,
Allen, and Hospedales 2019):
zu=fθ(hu),z u∈Bd,(4)
where
Bd=
z∈Rd:∥z∥ 2<1	
,(5)
and∥ · ∥ 2denotes the standard Euclidean norm. The map-
ping fθ, which includes a projection onto the open unit
ball, directly maps precomputed Euclidean text embeddings
into the hyperbolic manifold. To align textual semantics
with discrete multi-hop topology, fθis trained offline via a
graph-supervisedmargin-basedtripletlossandaradialdepth
regularizer (detailed in Appendix). This helps ensure the rep-
resentations encapsulate both raw semantics and hierarchical
structures.
For a nodeu, let
λ(zu) =2
1− ∥z u∥2
2(6)
denote the conformal factor of the Poincaré metric at zu.
In Riemannian geometry, this factor dictates the volume
expansionofthelocalspace.Therefore,thegeometricvolume
occupiedbyanodeisintrinsicallydrivenbythisconformal
scaling. We thus define the local volume proxy as:
v(u) =λ(z u)p.(7)
Geometrically, v(u)quantifiesthiscontinuousspatialoccu-
pancy, which acts as an inverse indicator of semantic breadth.
DuetotheexponentialoutwardexpansionofthePoincaréball,
genericsemantichubsattheoriginaretightlycompressedinto
minimal conformalvolumes, whereas highlyspecific factual
entities at the periphery occupy vast spatial regions. In strict
Riemanniangeometry,thetruevolumescaleswiththedimen-
sionality d.However,computing λ(zu)dforhigh-dimensional
embeddings( d≥128)inevitablyleadstonumericaloverflow.
Toensuresystemreliability,weintroduceatunablescaling
exponent p≪d. This engineering formulation provides a

numerically stable proxy for the node volume that flexibly
controls the dynamic range of the structural signal. By doing
so,wepreservethecoremonotonicconformalscalingprop-
erty of hyperbolic space while guaranteeing robust online
computation.
To construct a principled structural ratio, we define a
localized geometric measure based on the classical Cheeger
constant. For any node u∈ V, letSu={u} ∪ N(u) denote
its closed 1-hop neighborhood set. We define the internal
geometric volume of this local region as the sum of the
conformal volume proxies of its constituent nodes:
V(S u) =X
n∈Suv(n).(8)
Next,weidentifythetopologicalboundaryofthisregion.
Let∂Sudenote the 2-hop boundary shell of u, consisting
ofallnodesadjacentto N(u)thatarenotcontainedwithin
Su. The geometric volume of this boundary is analogously
defined as:
V(∂S u) =X
n∈∂S uv(n).(9)
Havingformalizedboththeinternalandboundaryvolumes
usingtheexactsamehyperbolicmeasure,wedefinethenode-
wisestructuralpruningscore ϕuasthelocalizedgeometric
Cheeger ratio:
ϕu=V(∂S u)
V(S u) +ε,(10)
where ε >0is a small stability constant. Unlike heuristic
formulations that mix mismatched topological and geometric
scales, this definition strictly preserves the isoperimetric
natureofthescore.Theproxymeasurestherelativegeometric
expansion of a node’s local neighborhood. Generic hubs
exhibit explosive boundary volumes ( V(∂S u)) compared
to their highly compressed internal neighborhood volumes
(V(S u)),yieldingextremelylarge ϕuvalues.Byevaluating
this rigorous volume-to-volume ratio, the framework can
explicitly identify structural bottlenecks without requiring
computationally prohibitive global graph partitioning.
DiscriminativePoweroftheProxy.Inscale-freehyper-
bolic embeddings, a node’s radial distance inversely tracks
its topological degree, while its conformal volume grows ex-
ponentiallywithradius.Thisdualscalingcreatesageometric
mismatchforgenerichubs:theyresideneartheoriginwith
heavily compressed individual volumes, yet their massive
topologicalconnectivitybridgestonumerousspecificnodes
at the periphery. Consequently, for a central hub, its 2-hop
boundary shell ∂Sureaches into the expansive periphery,
accumulating an explosive boundary volume V(∂S u), while
itsinternal1-hopneighborhoodvolume V(S u)remainsheav-
ilyconstrained.This extremevolume-to-volume divergence
triggers massive ϕuanomalies for spurious edges connecting
factual nodes to unrelated hubs, enabling ISO-RAG to struc-
turally isolate probability leakage. Detailed mathematical
formulations are provided in Appendix.
EdgeCompatibilityFiltering.Toprunespuriousedges
bridgingstructurally dissimilarnodes,we enforcestructural
consistencybetweenendpoints.Anedge (u, v)∈ G loc(q)isretained only if:
min(ϕ u, ϕv)
max(ϕ u, ϕv)≥β,(11)
where β∈(0,1] is a validation-tuned tolerance threshold.
Equivalently, in log-scale:
|logϕ u−logϕ v| ≤ −logβ.(12)
Here,−logβboundsthemaximumallowablestructuraldi-
vergence.Validreasoningstepsbetweennodesofcomparable
specificity exhibit small ϕ-divergences. Conversely, edges
bridging opposite structural extremes (e.g., direct transitions
betweenspecificfactualleavesandgenerichubs)inevitably
violate this threshold and are systematically pruned. This
dual-sidedfilteringmitigatessemanticdriftfromtwodirec-
tions: it blocks forward probability leakage into hubs during
propagation,andisolateserroneouslyretrievedhubanchors
before diffusion begins. Because all node-wise structural
scores ϕucan be fully precomputed and cached, this geo-
metric filtering introduces negligible online computational
latency. We provide details in Appendix.
Let
eGloc(q) =
Vloc(q),eEloc(q)
(13)
denote the resulting geometrically filtered local graph, which
provides a structurally coherent and bounded manifold for
the subsequent localized PageRank.
3.4 Localized Deterministic Personalized
PageRank
After edge filtering, retrieval operates strictly on the compact
grapheGloc(q).Were-normalizeitsadjacencymatrixtoderive
a valid column-stochastic transition matrix fW.
The personalization vector r(q)distributes the initial prob-
abilitymass exclusivelyacross theselectedseednodes S(q)
via a temperature-scaled softmax:
ru(q) =

exp(τ·sdense
u)P
v∈S(q)exp(τ·sdensev), u∈ S(q),
0,otherwise,(14)
where τ >0controls the mass concentration sharpness, and
sdense
udenotes the initial dense semantic similarity score.
The localized PPR vector π(q)is defined as the unique
fixed point of the diffusion process:
π(q) = (1−α)r(q) +α fWπ(q),(15)
where α∈(0,1) is the damping factor (with 1−αacting
astheteleportationprobability).Duringdiffusion,theprob-
ability massof dangling nodes isintrinsically redistributed
viar(q), which falls back to a uniform distribution if seed
weights degrade to zero. Rather than relying on stochastic
random walks that introduce approximation variance, we
compute π(q)deterministically via power iteration. Because
ourstructuralfilteringstrictlyboundsthediffusionspaceto
thecompactlocalsubgraph eGloc(q),thisexactcomputation
remains highly efficient and yields stable structural scores.

Model RetrieverHotpotQA2Wiki-
MultihopQAMuSiQue
F1 EM F1 EM F1 EM
Qwen2.5GraphRAG 79.1 72.3 54.1 52.8 28.8 23.2
GraphRAG+PPR 75.0 68.4 54.5 52.9 30.6 24.2
LightRAG 79.8 72.3 72.2 68.9 34.0 28.4
HippoRAG2 80.6 73.1 62.3 59.7 36.9 30.8
HyperbolicRAG 79.6 72.3 60.0 58.2 30.6 25.6
ISO-RAG 81.1 74.1 76.9 73.2 40.1 33.9
Qwen-
PlusGraphRAG 80.4 72.8 56.7 55.2 28.7 23.3
GraphRAG+PPR 78.1 70.8 57.5 55.2 30.1 24.0
LightRAG 82.375.072.8 68.8 30.9 24.6
HippoRAG2 82.2 74.6 63.7 60.9 34.2 27.7
HyperbolicRAG 81.4 73.7 61.6 59.4 32.1 25.4
ISO-RAG 82.5 75.0 80.4 75.8 42.8 31.4
Qwen3-
MaxGraphRAG 83.1 76.0 60.9 59.2 28.7 23.3
GraphRAG+PPR 81.0 74.1 59.5 57.7 30.1 24.0
LightRAG 84.6 77.4 77.0 73.8 33.6 28.3
HippoRAG2 84.8 77.7 67.4 64.6 36.9 30.9
HyperbolicRAG 83.9 76.5 64.5 62.5 34.6 28.8
ISO-RAG 85.2 78.2 84.7 80.7 40.2 33.7
Table 1: Main Results on Multi-Hop QA Datasets. F1 and
EM scores are in percentage (%). Best results arebolded.
Finally, the converged structural scores are fused with
the initial dense semantic similarities. Denoting the scalar
structural score for node uasπu(q), which is extracted from
thevector π(q),thefinalretrievalscore sfinal
uforeachpassage
uis computed via a linear combination:
sfinal
u=λ gπu(q) +λ dsdense
u,(16)
where πu(q)andsdense
uaremin-maxnormalizedoverthecan-
didatesetpriortofusion,and λgandλdaretunablebalancing
weights.Thecandidatepassagesaresubsequentlyrankedby
sfinal
uto extract the optimal top- Kevidence set RK(q)for
downstreamQA.Thisdual-signalfusionelegantlycouplesthe
semanticrecallofdensemodelswiththestructuralmulti-hop
precisionofourisoperimetricframework,ensuringthatthe
finalrankingisbothcontextuallyrelevantandtopologically
coherent.
4 Experiments
In thissection, we comprehensively evaluateISO-RAG toan-
swerthefollowingResearchQuestions(RQs):RQ1(Overall
Performance):Does ISO-RAG outperform existing dense
and graph-based retrieval methods in both retrieval accu-
racy and downstream multi-hop QA?RQ2 (Efficiency):
Can the localized deterministic routing paradigm achieve
better retrieval-time efficiency compared to unconstrained
graph traversals?RQ3 (Geometric Filtering):How does the
isoperimetricsignal( ϕ)explicitlycontributetonoise-aware
structural filtering?
4.1 Experimental Setup
Datasets&GraphConstruction.WeevaluateISO-RAGon
threestandardmulti-hopQAbenchmarkswithvaryingrea-
soningcomplexities:HotpotQA(Yangetal.2018)(primarily2-hop),2WikiMultihopQA(Hoetal.2020)(2–4hop),and
MuSiQue (Trivedi et al. 2022) (up to 4-hop compositional
reasoning). To establish a unified evaluation setting for both
structural retrieval and end-to-end QA, we randomly sample
1,000instancesfromthevalidationsetofeachbenchmark.For
graphconstruction,ratherthanbuildingatraditionalentity-
relationknowledgegraph,weconstructaquestion-induced
passage co-occurrence graph (details in Appendix. Passages
are treated as nodes, with edges connecting passages that co-
occurinthesametraininginstance.Nodetextsareembedded
offline using the text-embedding-v3 encoder (Zhang
et al. 2025). To strictly prevent data leakage, the passage
co-occurrencegraphsareconstructedexclusivelyusingthe
trainingsplitsoftherespectivedatasets.Validationandtest
sets are completely excluded from the graph construction
phase, ensuring that the retrieval framework does not benefit
from any benchmark-specific structural shortcuts.
Table3comparesretrievallatencyandtokenusageunder
Qwen3-Max to assess the computational advantage of our
localized pipeline.
Baselines.To comprehensively evaluate retrieval qual-
ity, we categorize baselines into non-graph meth-
ods (BM25 (Robertson, Zaragoza et al. 2009), Flat
Dense (Karpukhin et al. 2020), MDR (Xiong et al. 2021))
and graph-based paradigms. The latter ranges from foun-
dational algorithms (Vanilla PPR (Page et al. 1999)) to
state-of-the-artframeworks(GraphRAG (Edgeetal.2024),
GraphRAG+PPR (Page et al. 1999), LightRAG (Guo et al.
2024), HippoRAG2 (Jimenez Gutierrez et al. 2024), and
HyperbolicRAG (Cao et al. 2025)). As Vanilla PPR is a
foundational retrieval algorithm rather than an end-to-end
RAG pipeline, we focus our QA assessment exclusively on
the aforementioned state-of-the-art frameworks designed for
generative tasks. To execute this generation, the retrieval
outputsarepairedwithmultipleLLMbackbones(Qwen2.5,
Qwen-Plus (Team 2024), and Qwen3-Max (Team 2025)).
Implementation Details.We evaluate downstream QA
performanceusingstandardExactMatch(EM)andF1scores.
Forfaircomparison,allgraphframeworkssupplytherawtext
oftheirretrievedtop- kpassagestotheLLMgeneratorusingan
identicalprompttemplate.Comprehensiveimplementation
details, including hyperparameter tuning (e.g., PageRank
α),exactgrid-searchranges,andfullprompttemplates,are
detailed in Appendix.
4.2 Main Results: Retrieval and QA Performance
(RQ1)
We first evaluate the fundamental retrieval capability (Ta-
bles2)andtheend-to-endQAgenerationquality(Table1).
ISO-RAG consistently yields the best results across all set-
tings.
Consistent Gains in Shallow Reasoning.While perfor-
manceonHotpotQAisnearingsaturationforhigh-capacity
models, ISO-RAG still guarantees stable improvements,
achieving88.05R@5(+0.75absolutepointsoverthestrongest
baseline) and peaking at 85.2 F1 score with Qwen3-Max.
Notably, the improvements are robust across model scales,
suggesting that our retrieval framework itself fundamentally
drives the observed gains.

MethodHotpotQA 2WikiMultihopQA MuSiQue
R@5 R@10 P@5 P@10 R@5 R@10 P@5 P@10 R@5 R@10 P@5 P@10
BM25 62.70 75.25 25.08 15.05 39.80 48.20 18.42 11.08 27.00 32.15 10.64 6.34
Flat Dense 81.90 87.95 32.76 17.59 69.43 71.58 31.82 16.39 52.40 58.95 20.54 11.57
MDR 78.80 88.15 31.52 17.63 61.65 68.55 27.70 15.53 51.70 59.45 20.34 11.70
Vanilla PPR 81.75 94.40 32.70 18.88 69.05 78.95 31.80 18.96 70.10 77.30 27.12 14.98
GraphRAG 85.25 96.20 34.10 19.24 71.15 88.45 32.84 21.34 59.00 72.30 22.84 13.99
GraphRAG+PPR 71.45 94.20 28.58 18.84 69.78 81.95 32.38 19.91 63.20 76.15 24.40 14.74
LightRAG 87.30 97.20 34.92 19.44 77.93 88.80 35.92 20.95 55.05 67.05 21.60 13.16
HippoRAG2 86.65 97.85 34.66 19.57 70.18 82.80 32.30 19.88 63.85 75.40 26.14 14.64
HyperbolicRAG 85.30 96.85 34.12 19.37 70.45 78.38 32.34 18.55 55.35 76.05 21.70 14.70
ISO-RAG 88.05 98.05 35.22 19.61 88.10 97.15 40.10 22.90 74.75 84.00 28.98 16.31
Table 2: Retrieval performance comparison of Recall (R@) and Precision (P@) at top-5 and top-10 across three multi-hop
datasets. Best results arebolded.
Superiority in Complex Topologies.Table 1 demon-
strates that ISO-RAG outperforms all baselines on 2Wiki-
MultihopQA. Notably, it surpasses the strongest baseline,
LightRAG, achieving an R@5 of 88.10 with a 10.17-point
absolute improvement. Compared to recent topology-driven
frameworks (i.e., HippoRAG2 and HyperbolicRAG), this
gap widens, yielding a relative R@5 gain exceeding 25%.
Furthermore, QA evaluation using Qwen3-Max yields an F1
score of 84.7 and an EM score of 80.7. These results confirm
theefficacyofISO-RAGinstructuredmulti-hopscenarios,
where preserving valid intermediate hops and suppressing
spurious diffusion are critical.
Robustness against High Noise.The MuSiQue dataset
presents a severe challenge of deep compositional reasoning
amidst dense distractors. In this regime, while ISO-RAG
achieves a state-of-the-art R@10 score of 84.00 (+11.4%
relative gain over HippoRAG2) alongside highly competitive
precision of 16.31, its full potential materializes in the QA
phase. Evaluated with Qwen-Plus, the framework peaks at
an F1 score of 42.8, securing an approximate 25% relative
gain over HippoRAG2. Thisdisproportionate amplification,
transitioning from a steady retrieval improvement to a drastic
QA leap, demonstrates that ISO-RAG does not merely ac-
cumulatedisjointrelevantdocuments.Instead,drivenbyits
high retrieval precision, it successfully isolates the precise,
noise-free multi-hop evidence pathways required to cross the
reasoning threshold of the LLM.
4.3 Efficiency Analysis (RQ2)
Latency Reduction via Local Subgraphs.ISO-RAG main-
tainsstablemillisecond-levelspeedsacrossallbenchmarks,
ranking as the second fastest retriever on both HotpotQA and
MuSiQue.WhilemarginallytrailingthevanillaGraphRAG
baselineinrawspeed,itprovidesasubstantiallymoreprecise
reasoningcontext.Comparedtorecenttopology-drivenframe-
works,thelatencygapisparticularlystriking:onMuSiQue,
ISO-RAG delivers an approximate 8 ×speedup over Hip-
poRAG2 and operates over 25 ×faster than HyperbolicRAG.
ThisconfirmsthatboundingPageRankwithinageometrically
filteredsubgraphsuccessfullybypassestheheavyoverhead
of global traversals.Dataset MethodEfficiency
Retr/Q (ms)↓Avg Prompt
HotpotQAGraphRAG10.81619.91
GraphRAG+PPR 15.1 1652.10
LightRAG 22.9 1644.83
HippoRAG2 52.0 895.16
HyperbolicRAG 92.7 889.61
ISO-RAG12.3 886.44
2Wiki-
MultihopQAGraphRAG6.21389.09
GraphRAG+PPR 8.7 1225.13
LightRAG 14.4 1271.18
HippoRAG2 49.1720.89
HyperbolicRAG 71.1 758.91
ISO-RAG11.5 793.49
MuSiQueGraphRAG9.91579.64
GraphRAG+PPR 16.2 1588.76
LightRAG 65.6 1561.58
HippoRAG2 117.2 912.46
HyperbolicRAG 375.1900.95
ISO-RAG14.2 927.07
Table 3: Efficiency Comparison across Datasets. Latency
is measured in milliseconds per query (Retr/Q). Best re-
sultsineachdatasetarebolded,andsecond-bestresultsare
underlined .
Favorable Accuracy-Efficiency Trade-off.ISO-RAG ex-
hibitshighlycompetitivetokenefficiency,incurringthelowest
prompt token overhead among all baselines on HotpotQA.
While structurally complex datasets require marginally more
tokens than HippoRAG2 and HyperbolicRAG, this slight
increment is fully offset by substantial gains in retrieval ac-
curacy. Ultimately, this confirms that ISO-RAG delivers a
strictly higher density of actionable reasoning chains per
token, ensuring a highly cost-effective retrieval process.
4.4 Ablation: Isoperimetric Geometric Filtering
(RQ3)
Weisolatetheisoperimetricfilteringmoduleon2WikiMul-
tihopQA, where the impact of geometric guidance is most
pronounced, to compare true geometric guidance against un-
constrained,random,ortopologicallydecouplededgepruning
under highly noisy conditions (Figure 3).

Figure 2: Case Study
Figure3:RetrievalPerformanceandRemovedEdgesGrouped
by Floor (β) and CandidateXConfigurations (in %)
We define three core variables to control the ablation
space: Candidate X, the over-retrieval factor applied to the
initialdensepoolpriorto ϕ-filtering; β,thethresholdcontrol-
linggeometricedgepruningstrictness;andfour ϕmapping
strategies: (1) real_phi applies actual learned scores to
capturetruelocalcontinuity;(2) uniform_phi assignsa
constant value, disabling the filter to revert to vanilla PPR;
(3)shuffled_phi randomlypermutes real_phi values,
preservingglobaldistributionbutdestroyingcorrelationwith
graph topology; and (4)random_phiapplies uniform ran-
domvaluestotestarbitraryedgepruning.Theperformance
is evaluated using recall, precision, and the percentage of
removed edges.
NecessityoftheRealGeometricSignal. real_phi dom-
inates all variants, achieving 87.28% peak recall and 40.44%
peakprecisionat β= 0.40 .Conversely, uniform_phi (no
filtering)and random_phi (randomscalarfield)stagnateat
∼71-72%.Thisdemonstratesthatretrievalgainsstemfrom
thelearnedgeometricsignal,notmerelybaselinegraphtopol-
ogy(uniform_phi )orrandomedgedropoutregularization
(random_phi).
Necessity of Topological Alignment. shuffled_phi
preserves the true numerical distribution of isoperimetric
scoresbutdecouplesthemfromgraphtopology.Consequently,it aggressively removes up to 75% of edges and suffers
a massive performance drop, whereas ISO-RAG achieves
peak results by pruning only 32%. This proves multi-hop
retrieval requires topology-aware filtering, not indiscriminate
pruning. By strictly aligning the isoperimetric signal with
graphstructure,ISO-RAGselectivelyprunesedgesleaking
probability mass into irrelevant neighborhoods.
4.5 Qualitative Case Study
To illustrate how geometric filtering prevents probability
leakageinPPR,weexamineamulti-hopqueryfrom2Wiki-
MultihopQA (Figure 2). This query requires a parallel 2-hop
reasoningchain:identifyingdirectorsfortwofilms,retrieving
their biographies, and comparing their birth dates.
Preventing Probability Leakage.In the top-5 contexts,
baseline methods fail to retrieve the crucial passages; their
unconstrained PageRank diffusion is hijacked by dense, se-
mantically adjacent hub nodes (e.g., unrelated films or direc-
tors). Consequently, HippoRAG2 outputs "Unknown", while
HyperbolicRAG hallucinates the wrong film. Conversely,
ISO-RAG’s isoperimetric control severs spurious edges to
these hubs. By bounding probability mass within the local
manifold, it retrieves both required passages, enabling the
LLM to deduce the correct answer.
DecouplingRetrievalandGenerationErrors.Tofurther
illustratethevulnerabilityofdownstreamLLMstocontextual
noise, we present a "Perfect Retrieval, Failed Generation"
caseinFigure2.AlthoughISO-RAGsuccessfullyretrieved
100% of the ground-truth entities, the LLM still failed. This
demonstrates that degraded F1/EM scores are not exclusively
indicative of retrieval failure, but can also stem from LLM
reasoning bottlenecks.
5 Conclusion
In this work, we presented ISO-RAG, a novel retrieval-
augmented generation framework that imposes strict topolog-
icalcontroltomitigatespuriousdiffusionduringgraph-based
retrieval. By mapping the graph into hyperbolic space and
training a geometry-aware encoder, ISO-RAG enables con-
trolledretrievalthatisbothefficientandeffectivefordown-

stream question answering. Across benchmarks, it yields
consistent absolute improvements over competitive baselines,
underscoring the importance of topology-aware control for
multi-hop reasoning.
References
Alon, U.; and Yahav, E. 2021. On the Bottleneck of Graph
NeuralNetworksandItsPracticalImplications. InInterna-
tional Conference on Learning Representations.
Andersen, R.; Chung, F.; and Lang, K. 2006. Local Graph
Partitioning UsingPageRank Vectors. In47th Annual IEEE
Symposium on Foundations of Computer Science (FOCS’06),
475–486. IEEE.
Balazevic, I.; Allen, C.; and Hospedales, T. 2019. Multi-
Relational Poincaré Graph Embeddings. InAdvances in
Neural Information Processing Systems, volume 32.
Cao, L.; Wang, R.; Li, J.; Zhou, Z.; and Yang, M. 2025. Hy-
perbolicRAG:EnhancingRetrieval-AugmentedGeneration
with Hyperbolic Representations. arXiv:2511.18808.
Chami,I.;Ying,Z.;Ré,C.;andLeskovec,J.2019.Hyperbolic
GraphConvolutionalNeuralNetworks.InAdvances in Neural
Information Processing Systems, volume 32.
Chen, J.; Lin, H.; Han, X.; and Sun, L. 2023. Benchmarking
Large Language Models in Retrieval-Augmented Generation.
arXiv:2309.01431.
Chung, F. R. 1997.Spectral Graph Theory, volume 92.
American Mathematical Society.
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody,
A.; Truitt, S.; and Larson, J. 2024. From Local to Global:
AGraphRAGApproachtoQuery-FocusedSummarization.
arXiv:2404.16130.
Gao,Y.;Xiong,Y.;Gao,X.;Jia,K.;Pan,J.;Bi,Y.;Dai,Y.;
Sun,J.;andWang,H.2023. Retrieval-AugmentedGeneration
for Large Language Models: A Survey. arXiv:2312.10997.
Girvan, M.; and Newman, M. E. 2002. Community Struc-
tureinSocialandBiologicalNetworks.Proceedings of the
National Academy of Sciences, 99(12): 7821–7826.
Gulcehre,C.;Denil,M.;Cabukuetli,M.;Pfau,D.;Pascanu,
R.; Hoffman, M. W.; and Nando, d. F. 2019. Hyperbolic
AttentionNetworks. InInternational Conference on Learning
Representations.
Guo,Z.;Xia,L.;Yu,Y.;Ao,T.;andHuang,C.2024. Ligh-
tRAG: Simple and Fast Retrieval-Augmented Generation.
arXiv:2410.05779.
Guu, K.; Gurkurun, T.; Knecht, D.; Chang, M.-W.; and
Salakhutdinov, R. 2020. REALM: Retrieval-Augmented
LanguageModelPre-training. InInternational Conference
on Machine Learning, 3929–3938. PMLR.
Ho,X.;Nguyen,A.-K.D.;Sugawara,S.;andAizawa,A.2020.
Constructing a Multi-hop QA Dataset for Comprehensive
Evaluation of Reasoning Steps. InProceedings of the 28th
International Conference on Computational Linguistics,6609–
6625.
Ji, Z.; Lee, N.; Frieske, R.; Yu, T.; Su, D.; Xu, Y.; Ishii,
E.; Bang, Y. J.; Madotto, A.; and Fung, P. 2023. Surveyof Hallucination in Natural Language Generation.ACM
Computing Surveys, 55(12): 1–38.
Jiang,Z.;Xu,F.F.;Gao,L.;Sun,Z.;Liu,Q.;Dwivedi-Yu,J.;
Yang,Y.;Jamie,C.;andNeubig,G.2023. ActiveRetrieval
Augmented Generation. arXiv:2305.06983.
JimenezGutierrez,B.;Shu,Y.;Gu,Y.;Yasunaga,M.;andSu,
Y. 2024. HippoRAG: Neurobiologically Inspired Long-Term
Memory for Large Language Models. InAdvances in Neural
Information Processing Systems, volume 37, 59532–59569.
JimenezGutierrez,B.;Shu,Y.;Gu,Y.;Yasunaga,M.;andSu,
Y.2025. FromRAGtoMemory:Non-ParametricContinual
Learning for Large Language Models. arXiv:2502.14802.
Kaplan, J.; McCandlish, S.; Henighan, T.; Brown, T. B.;
Chess, B.; Child, R.; Gray, S.; Radford, A.; Wu, J.; and
Amodei,D.2020. ScalingLawsforNeuralLanguageModels.
arXiv:2001.08361.
Karpukhin, V.; Oguz, B.; Min, S.; Lewis, P.; Wu, L.; Edunov,
S.;Chen,D.;andYih,W.-t.2020. DensePassageRetrieval
forOpen-DomainQuestionAnswering. InProceedings of the
2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP), 6769–6781.
Krioukov, D.; Papadopoulos, F.; Kitsak, M.; Vahdat, A.;
and Boguná, M. 2010. Hyperbolic Geometry of Complex
Networks.Physical Review E, 82(3): 036106.
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;
Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel,
T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented
GenerationforKnowledge-IntensiveNLPTasks. InAdvances
in Neural Information Processing Systems, volume 33, 9459–
9474.
Nickel, M.; and Kiela, D. 2017. Poincaré Embeddings for
LearningHierarchicalRepresentations.InAdvances in Neural
Information Processing Systems, volume 30.
Nickel,M.;andKiela,D.2018. LearningContinuousHier-
archies in the Lorentz Model of Hyperbolic Geometry. In
International Conference on Machine Learning, 3779–3788.
Page, L.; Brin, S.; Motwani, R.; and Winograd, T. 1999.
The PageRank Citation Ranking: Bringing Order to the Web.
Technical report, Stanford InfoLab.
Peng, W.; Varanka, T.; Mostafa, A.; Shi, H.; and Zhao, G.
2022. HyperbolicDeepNeuralNetworks:ASurvey.IEEE
Transactions on Pattern Analysis and Machine Intelligence,
44(12): 10023–10044.
Robertson, S.; Zaragoza, H.; et al. 2009. The Probabilistic
Relevance Framework: BM25 and Beyond.Foundations and
Trends®in Information Retrieval, 3(4): 333–389.
Sala, F.; De Sa, C.; Gu, A.; and Ré, C. 2018. Representa-
tionTradeoffsforHyperbolicEmbeddings. InInternational
Conference on Machine Learning, 4460–4469. PMLR.
Team, Q. 2024. Qwen2.5 Technical Report.
arXiv:2412.15115.
Team, Q. 2025. Qwen3 Technical Report. arXiv:2505.09388.
Topping,J.;DiGiovanni,F.;Chamberlain,B.P.;Dong,X.;and
Bronstein, M. M. 2022. Understanding Over-Squashing and
Bottlenecks on Graphs via Ricci Curvature. InInternational
Conference on Learning Representations.

Trivedi,H.;Balasubramanian,N.;Khot,T.;andSabharwal,
A. 2022. MuSiQue: Multihop Questions via Single-hop
QuestionComposition.Transactions of the Association for
Computational Linguistics, 10: 539–554.
Vaswani,A.;Shazeer,N.;Parmar,N.;Uszkoreit,J.;Jones,L.;
Gomez,A.N.;Kaiser,Ł.;andPolosukhin,I.2017.AttentionIs
AllYouNeed. InAdvances in Neural Information Processing
Systems, 5998–6008.
Wei, J.; Wang, X.; Schuurmans, D.; Maeda, M.; Xia, F.; Chi,
E.;Le,Q.V.;andZhou,D.2022.Chain-of-ThoughtPrompting
ElicitsReasoninginLargeLanguageModels. InAdvances in
Neural Information Processing Systems,volume35,24824–
24837.
Xiong, W.; Li, X. L.; Iyer, S.; Du, J.; Lewis, P.; Wang,
W. Y.; Yashar, M.; Yih, W.-t.; Riedel, S.; Douze, M.; et al.
2021. Answering Complex Open-Domain Questions with
Multi-Hop Dense Retrieval. InInternational Conference on
Learning Representations (ICLR).
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.;
Salakhutdinov,R.;andManning,C.D.2018. HotpotQA:A
DatasetforDiverse,ExplainableMulti-hopQuestionAnswer-
ing. InProceedings of the 2018 Conference on Empirical
Methods in Natural Language Processing, 2369–2380.
Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; McManus, T. G.; Isbell,
R.; and Narasimhan, K. 2023. Tree of Thoughts: Deliberate
Problem Solving with Large Language Models. InAdvances
in Neural Information Processing Systems.
Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan,
K.; and Cao, Y. 2022. ReAct: Synergizing Reasoning and
Acting in Language Models. InInternational Conference on
Learning Representations.
Zhang, Y.; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.;
Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou, J.
2025. Qwen3Embedding:AdvancingTextEmbeddingand
Reranking Through Foundation Models. arXiv:2506.05176.