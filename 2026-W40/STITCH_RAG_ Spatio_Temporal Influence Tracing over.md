# STITCH-RAG: Spatio-Temporal Influence Tracing over Topic Hypergraphs for Multi-Hop Retrieval-Augmented Generation

**Authors**: Haodong Yang, Mengzhu Chen, Jia Cai

**Published**: 2026-09-28 02:05:31

**PDF URL**: [https://arxiv.org/pdf/2609.34127v1](https://arxiv.org/pdf/2609.34127v1)

## Abstract
Multi-hop retrieval-augmented generation requires a retriever to connect evidence distributed across documents while preserving a concise, faithful generation context. Existing indexes leave two complementary gaps: chunk-based RAG can break cross-passage evidence chains, whereas an unlabeled pairwise projection without generating-topic provenance cannot jointly preserve topic-level co-participation and per-occurrence entity descriptions. We propose STITCH-RAG, a hypergraph-based framework with three coupled components. First, a semi-merged topic hypergraph encodes multi-entity co-participation as topic-summary hyperedges while retaining per-chunk entity states linked by canonical-name equivalence. Second, spatio-temporal influence bridging propagation (STIBP) combines topic-space propagation with deterministic chunk-index linkage across name-equivalent states under frequency-adaptive decay. Third, continuous STIBP scores replace binary entity-match seeds in localized Personalized PageRank (PPR). We characterize the condition under which this prior assigns more PPR mass to ground-truth evidence than a binary prior. Under the reported protocol, STITCH-RAG attains the highest reported Contain-Acc and LLM-Acc point estimates among the compared methods on HotpotQA and 2WikiMultiHopQA, and higher Recall@8 than the methods included in the standardized retrieval comparison. Results on the mixed-domain benchmark remain auxiliary preference-based evidence because only LLM-judged accuracy is available.

## Full Text


<!-- PDF content starts -->

STITCH-RAG: Spatio-Temporal Influence Tracing over Topic Hypergraphs for
Multi-Hop Retrieval-Augmented Generation
Haodong Yang, Mengzhu Chen, Jia Cai∗
School of Statistics And Data Science, Guangdong University of Finance&Economics
Guangzhou, China
haodongyang@student.gdufe.edu.cn, mengzhuchen@student.gdufe.edu.cn, jiacai1999@gdufe.edu.cn
Abstract
Multi-hop retrieval-augmented generation requires a retriever
toconnectevidencedistributedacrossdocumentswhilepre-
servingaconcise,faithfulgenerationcontext.Existingindexes
leavetwocomplementarygaps:chunk-basedRAGcanbreak
cross-passage evidence chains, whereas an unlabeled pairwise
projection without generating-topic provenance cannot jointly
preserve topic-level co-participation and per-occurrence entity
descriptions. We propose STITCH-RAG, a hypergraph-based
framework with three coupled components. First, a semi-
mergedtopichypergraphencodesmulti-entityco-participation
astopic-summaryhyperedgeswhileretainingper-chunkentity
states linked by canonical-name equivalence. Second, spatio-
temporal influence bridging propagation (STIBP) combines
topic-spacepropagationwithdeterministicchunk-indexlink-
age across name-equivalent states under frequency-adaptive
decay.Third,continuousSTIBPscoresreplacebinaryentity-
match seeds in localized Personalized PageRank (PPR). We
characterizetheconditionunderwhichthispriorassignsmore
PPR mass to ground-truth evidence than a binary prior. Un-
der the reported protocol, STITCH-RAG attains the highest
reported Contain-Acc and LLM-Acc point estimates among
the compared methods on HotpotQA and 2WikiMultiHopQA,
and higher Recall@8 than the methods included in the stan-
dardizedretrievalcomparison.Resultsonthemixed-domain
benchmark remain auxiliary preference-based evidence be-
cause only LLM-judged accuracy is available.
1 Introduction
Standarddense-retrievalRAGpartitionsdocumentsintofixed-
sizechunksandranksthembyquerysimilarity.Thisrepresen-
tationcanmissmulti-hopchainsthatcrosschunkordocument
boundaries. Structured RAG adds entity and relation indexes,
but the published representations considered here do not
jointly retain two forms of context. First, an unlabeled pair-
wise projection without a generating-topic identifier discards
the topic that produced a multi-entity co-occurrence. This
limitation does not apply to pairwise graphs augmented with
relation labels, provenance, event identifiers, or equivalent
higher-orderattributes.Second,fullymergingrepeatedentity
mentions can erase chunk-specific roles, whereas treating
∗Corresponding author
Copyright©2027,AssociationfortheAdvancementofArtificial
Intelligence (www.aaai.org). All rights reserved.
(a) Chunk-based RAG (b) Pairwise graph RAG (c) STITCH-RAG (ours)User Query:
Which country was the director of Film X born in?
Chunk 1
Film X was 
directed
by Alice Smith.
Chunk 2
Alice Smith was
born in Canada.Retrieved
✓
Missed
×
⚠Misses the evidence chainStatic pairwise graph
Film XAlice
Smith
Canada
⚠Only pairwise relations
⚠One static Alice Smith nodeTopic 1
Film XAlice Smith
(Chunk 1
state)
Topic 2Alice Smith
(Chunk 2
state)Canada
Chunk 1
Film X was directed 
by Alice Smith.Chunk 2
Alice Smith was 
born in Canada.
Answer: Canada✓◌Topic 
hyperedges
Semi-merged 
entity states 
Spatio-
temporal
bridging &
localized PPRAwardsFigure 1: Representation trade-offs for an example query
about the birthplace of Film X’s director. (a) Chunk-based
RAGcanreturnlocallysimilarbutdisconnectedpassages.(b)
The illustrated unlabeled pairwise/full-merge representation
retains the path through Alice Smith, but its single static
entity node mixes the evidence-bearing birthplace context
with unrelated contexts such as awards. The panel depicts
one unlabeled full-merge representation, not all pairwise
graphs. (c) STITCH-RAG preserves the two local Alice
Smithstatesandlinksthembynameequivalence,allowing
query-conditioned propagation through topic hyperedges and
across chunk-specific entity states.
everymentionindependentlyremovesanexplicitcross-chunk
identity link. Both choices introduce retrieval noise when
evidence depends on topic context and local entity state.
Figure1illustratestheserepresentationtrade-offs.Inthe
controlled unlabeled pairwise/full-merge example, the graph
retains a route from Film X to Canada, but a single Alice
Smith node cannot distinguish the birthplace-bearing context
from the award context. STITCH-RAG instead represents
topic-levelco-participationwithhyperedgesandretainschunk-
specificentitystatesasseparatenodes.AnLLMextractstopic
summariesashyperedgesandcontext-awareentitydescrip-
tions as nodes. The semi-merged hypergraph preserves those
descriptions and links mentions with identical normalized
canonicalnames.Thislinkisaname-equivalenceheuristic,
notaground-truthentitylinker.Throughoutthepaper,spatial
denotes topical co-participation within a hyperedge, whereas
temporaldenotes proximity in deterministic preprocessing
indices.Acrossdocuments,thisquantityisanindex-proximity
heuristic rather than a reading order, chronology, event time,
arXiv:2609.34127v1  [cs.IR]  28 Sep 2026

or causal relation.
At retrieval time, dense similarity activates entity nodes.
Spatialbridgingtransfersinfluencetoco-participantsinthe
same topic hyperedge, whereas index-proximity bridging
transfers influence across name-equivalent states through
boundedfrequency-adaptivedecayandquery-relevancegat-
ing.Theresultingcontinuousscoresareprojectedontochunks
and used as priors for localized approximate Personalized
PageRank (PPR), replacing binary entity-match initialization
with graded, context-aware diffusion. On HotpotQA, 2Wiki-
MultiHopQA, and Mix, STITCH-RAG attains the highest
reported Contain-Acc and LLM-Acc point estimates under
thestatedprotocol.Recall@8iscomparedonlywherestan-
dardized retrieval outputs are available. LLM-judged metrics
remain auxiliary (Section 4.1).
The novelty claim concerns the coupling of topic-
preservinghyperedges,context-specificentitystates,opera-
tionalname-equivalencelinks,andcontinuousPPRinitializa-
tioninoneretrievalpipeline.Nosinglecomponentisclaimed
to be new in isolation. The main contributions are as follows:
•A semi-merged hypergraph with a controlled recov-
erability characterization.The construction preserves
per-chunkentitydescriptionsasdistinctnodeswhileexpos-
ing canonical-name equivalence through an operational
name-equivalence map. Proposition 2 shows that, among
thethreecontrolledrepresentationsconsideredhere,semi-
mergingaloneretainsbothpropertiesrequiredbythestated
state-specificpropagationrule.Thecontrolledmergedi-
agnostic reports a higher LLM-Acc point estimate for the
implementedsemi-mergevariantthanforthetwoimple-
mentedalternatives.Becauseretrievalmetrics,structural
compression,andrun-levelvariancesareunavailablefor
thesevariants,thisissupporting,notconclusive,evidence.
•Topic-space and index-proximity influence bridging
withfrequency-adaptivedecay.STIBPoperatesbefore
PPR through topic-hyperedge propagation and determin-
istic chunk-index linkage across name-equivalent entity
states. The bounded decay f(∆t) = exp(−tanh(α∆t))
usesα=n u/¯nto attenuate index-distant links more
strongly for frequent names. Theorem 1 provides a one-
sidedboundforcontributionspropagatedfromoneacti-
vatedsource.Itneithermodelschronologynor character-
izesthefinalscoresaftermaxaggregationovermultiple
sources.
•Continuous influence priors with a conditional
evidence-mass characterization.Existing PPR-based
methods (Zhuang et al. 2025; Gutiérrez et al. 2024) com-
monly use binary entity-match initialization. STITCH-
RAG instead derives a non-negative, normalized chunk
priorfromgradedSTIBPscores.Proposition3givesanex-
actPPRevidence-massidentityandasufficientalignment
condition under which this prior assigns more evidence
massthanabinaryprior.Thestrictbinary-onlydiagnostic
inTable3isnotusedtoverifythisconditionbecausebinary
seedcoverage,empty-matchbehavior,andprior-support
size are not measured separately.
The propositions and theorem characterize the chosen rep-
resentation, decay, and PPR prior. They clarify the designassumptions but do not constitute end-to-end superiority
guarantees.
2 Related Work
StructuredRAGmethodsdifferprimarilyinindexrepresen-
tation and query-signal propagation. Under their published
representations, the methods compared here do not jointly re-
tain topic-level co-participation, per-occurrence descriptions,
and operational name-equivalence links. STITCH-RAG cou-
plesthesepropertiestoconstructgradedlocalized-PPRpriors.
It does not claim novelty for each component in isolation.
Graph-basedRAG.GraphRAG,LightRAG,NodeRAG,and
LinearRAG instantiate entity–relation, dual-layer, heteroge-
neous,andname-matchedPPRindexes,respectively(Edge
et al. 2024; Guo et al. 2024; Xu et al. 2025; Zhuang et al.
2025).Foranunlabeledpairwiseprojection,thegenerating-
topic identity is unavailable at retrieval time (Proposition 1).
This scope excludes provenance-enriched pairwise graphs.
SubGraphRAG and GFM-RAG (Li, Miao, and Li 2025; Luo
etal.2025b)prioritizelocalprecision,whereasHippoRAG
variants(Gutiérrezetal.2024;Gutiérrezetal.2025)apply
globalPPRwithbinarymatching.STITCH-RAGpreserves
topicidentityandlinkedlocalstates,thenusestheirgraded
influence as the PPR prior (Proposition 3).
Hypergraph-based RAG.HyperGraphRAG, Cog-RAG,
and CogniRAG use relational, thematic, and causal hyper-
edges(Luoetal.2025a;Huetal.2026a;He2026).Inthecon-
trolledmergeconstructions,fullmergingremoveslocalstates
and no merging removes ϕ, whereas semi-merging retains
bothϕandψ. Hyper-RAG,Cog-RAG, andIGMiRAGtrade
traversal,filtering,orroutingcostagainstcoverage(Fengetal.
2026;Houetal.2026).STITCH-RAGinsteadpropagatesonly
within the activated subgraph. Unlike the trained precedence
model in OKH-RAG (Wu et al. 2026), its index-proximity
decay is training-free and targets complex multi-hop queries,
where structured retrieval is most useful (Xiang et al. 2025).
Complementary directions.Adaptive routing, long-context
retrieval, and granularity selection address complementary
design choices (Yan et al. 2024; Jeong et al. 2024; Asai et al.
2024; Li et al. 2024a,b; Wu et al. 2024; Chen et al. 2024;
Sarthi et al. 2024; Kim et al. 2024). STITCH-RAG targets
boundaryfragmentationinlargeordynamicknowledgebases,
where one-time indexing can be amortized across repeated
queries(Ovadiaetal.2024;Balagueretal.2024).AppendixE
provides detailed comparisons.
3 Method
3.1 Overview and Design Rationale
STITCH-RAG implements acompress-then-amplifyretrieval
pipeline(Figure2).Duringcompression,documentsaretrans-
formedintoasemi-mergedtopichypergraph H= (V,E, ϕ, ψ)
whose hyperedges are topic summaries and whose nodes are
context-aware entity descriptions for individual chunks. Dur-
ingamplification,spatio-temporalinfluencebridgingpropa-
gation (STIBP) identifies relevant entity-state pathways, and
localizedapproximatePersonalizedPageRank(PPR)diffuses
the resulting continuous scores over a chunk-level graph.

1 Input
Query
Documents → Chunks
Chunk 1
Chunk 2
Chunk N2Semi-Merged
Topic Hypergraph
Topic A
Topic B
Topic Ce¹e²e³…
e¹e4e5…
e¹e4e5 …
Topics as hyperedges,
entity states as nodes
✓Preserves entity states
across chunks3Spatio-Temporal
Influence Propagation
e¹ e² e³…
e¹ e4 e5…
e¹e4 e5 …
Spatial (within topic)
Temporal (across chunks)
Query-aware influence scoring4Influence-guided
PPR Retrieval
Chunk Graph
c₁c₂
c₃c₄ c₅
c₇c₉Influence
Score
High
Low
Continuous priors
replace binary seeds
Top-k chunks (by score)
c₅c₂c₇c₃ …5 Generation
Retrieved Top-k 
Chunks
LLM
Answer
Higher-order topic structure
Topics as hyperedges capture multi-
entity co-occurrence and interactions.Entity-state dynamics across chunks
Context-aware entity states propagate 
influence across space and time.Continuous PPR initialization
Influence scores become continuous 
priors for more flexible retrieval.
…e6e6…
……
…Node Graph
c₅c₂c₇c₃…
…
Figure 2: Framework of STITCH-RAG: (1) document chunking; (2) semi-merged topic hypergraph construction, with topic-
summary hyperedges and entity-description nodeslinked by ϕ; (3) STIBP through spatialand temporal channels; (4) influence-
guided chunk scoring as a continuous PPR teleportation prior; and (5) top-kchunk retrieval for answer generation.
Semi-merging is required because temporal bridging uses
ϕ, whereas query-conditioned activation requires local de-
scriptions ψ. Proposition 2 formalizes this requirement for
thecontrolledconstructions.STIBPscoresentitystatesonthe
hypergraph,andPPRdiffusesthosescoresovertheinduced
chunk graph. Table 3 tests their roles through controlled
replacements.
3.2 Hypergraph Construction Phase
Given document collection D, we produce chunks C=
{ck}NC
k=1withatmost cmaxtokenseach.AnLLMwithprompt
pextprocesseseach cktoextracttuples gk,r= (e k,r, Vek,r),
where ek,ris a topic-summary hyperedge and Vek,ris its
entity set:
G={g k,r|gk,r∈LLM(c k|pext), ck∈C}.
Each node v= (vname, vdes)stores a canonical name and a
chunk-specificdescription.Theheuristic ϕ:V →Σ groups
identicalnormalizednames,whileψ:V → Drecordslocal
context; members of each group are ordered by deterministic
preprocessing index. Chunks remain the retrieval targets.
A topic hyperedge with mentities uses mincidence links
ratherthan m
2
pairwiselinksandretainsitsgenerating-topic
identity (Proposition 1).
Structural Advantage over Pairwise ProjectionFor
H= (V,E) ,theprojection Π(H)connects u̸=vwhenever
u, v∈efor some e. It discards the identity of the generatinghyperedge. Proposition 1 therefore concerns retrieval from
this unlabeled, provenance-free input, not pairwise graphs
whose attributes recover the grouping (Appendix D).
Proposition 1(Non-identifiability of Pairwise Topic Projec-
tion).Thereexisttwotopichypergraphs H1andH2suchthat
Π(H 1) = Π(H 2),butquery-conditionedspatialpropagation
overH1andH2assignsdifferentinfluencescorestothesame
target entities. Therefore, any retrieval rule whose input is
limited to Π(H), without labels, provenance, or attributes
that recover the generating hyperedges, cannot in general
reproduce topic-selective propagation over those hyperedges.
RecoverabilityBenefitofSemi-MergingThecontrolled
constructionsdifferonlyintheirexposureof ϕandretentionof
ψ(Definitions2–3).Forthepropagationrulebelow,onlysemi-
merging retains both; the result is a controlled recoverability
comparison rather than a universal claim.
Proposition 2(Recoverability Separation of Semi-Merged
Entity States).Let qbe a query and let G={v 1, . . . , v m}
be a name-equivalence group with ϕ(vi) =ϕ(v j)for all
i, j. Suppose vs∈Gsatisfies Aq(vs)> δ, and vr∈G
lies in a ground-truth evidence chunk with Aq(vr)≤δ.
Assume vris not reachable from vsthrough any activated
topic hyperedge, so that its only structural path from vsis
through name-equivalencetemporal linkage.If Sq(evr)>0
andsim(q, c vr)>0,then vrreceivesnotemporalinfluence
undertheno-mergeconstruction HN,whichtreatseachmen-
tionasanindependentnodewithnocross-chunk ϕ;cannot

be assigned state-specific influence under the full-merge con-
struction HF, which collapses each name-equivalence group
into a single node; and receives strictly positive temporal
influence under the semi-mergedH.
3.3 Retrieval Phase
Given query q, retrievalhas fourstages: (1)dense-similarity
activation of entity nodes and topic hyperedges; (2) STIBP
over spatial and temporal channels to obtain graded entity-
level influence; (3) projection of entity influence onto chunks
withname-equivalencegroup-sizeweighting;and(4)local-
ized PPR on the ϕ-induced chunk graph, using the projected
influence as a continuous teleportation prior.
Initial Node and Hyperedge ActivationWe embed the
queryandentitydescriptionswiththesamemodelandassign
Aq(v) =sim(q, vdes),sim(q, vdes)> δ,
0,otherwise,
withδ= 0.5. Hyperedges use the unthresholded Sq(e) =
sim(q, e) because spatial gating already attenuates low-
relevance topics.
Spatio-TemporalInfluenceBridgingPropagationSTIBP
maintains Aq(v),aspace(v), and atime(v)per node; within
each propagated channel, it retains the maximum source
value.
Spatialbridging.Foractivated u∈eandeach v∈V e\{u}:
aspace(v) =A q(u)·S q(e).
This update requires both source activation and topic rele-
vance.
Temporal (index-proximity) bridging.For the name-
equivalencegroup V∗
u={v∈ V |ϕ(v) =ϕ(u)} ofactivated
u, setα=n u/¯nwhere nu=|V∗
u|and¯nis the corpus-wide
average group size. Here tvdenotes the deterministic pre-
processing index; across documents, the resulting distance
is order-dependent index proximity rather than chronology.
For each v∈V∗
u\ {u}with ∆t=|t u−tv|, define the
contribution proposed by sourceuas
atime
u(v) =A q(u)·S q(ev)·exp(−tanh(α∆t))·sim(q, c v).
Here evdenotes a hyperedge containing v. When vbe-
longstoseveralhyperedges, Sq(ev)standsfor max e∋vSq(e),
matching the max-aggregation rule above. With Aq={u:
Aq(u)>0}, the stored temporal channel is
atime(v) = max
0,max
u∈A q:u̸=v,
ϕ(u)=ϕ(v)atime
u(v)
.
The decay function f(∆t) = exp(−tanh(α∆t)) lies in
(e−1,1]: it retains nonzero mass for distant evidence while
attenuating it. Because α∝n u, distant states of frequent
names receive stronger attenuation (Table 9, Appendix C).
The final entity influence aggregates all three channels:
a(v) = mean 
Aq(v), aspace(v), atime(v)
.
Mean aggregation weights the three channels equally and
induces the same ranking as summation because every node
has all three channels.Per-Source Analysis of Frequency-adaptive STIBPThe
followingper-sourceresultcharacterizesboundeddecayunder
explicit signal–noise assumptions. It is not an end-to-end
retrieval guarantee.
Theorem 1(Per-Source Lower-Bound Separation under Fre-
quency-Adaptive Temporal Decay).Let ube an activated
entity state with Aq(u) =a >0 and name-equivalence
group V∗
u. For each v∈V∗
u\ {u}, let∆v=|t u−tv|
andgv=Sq(ev) sim(q, c v).LetRandNbenonemptydis-
joint subsets of V∗
u\ {u}satisfying gr≥βfor all r∈ R,
0< g n≤ϵand∆n≥τfor all n∈ N, with β > ϵ >0 and
τ >0. Withα=n u/¯n, wheren u=|V∗
u|:
max r∈Ratime
u(r)P
n∈Natimeu(n)≥β
|N|ϵexp 
tanh(ατ)−1
.
For fixed R,N,β,ϵ,τ, and ¯n, the right-hand side is non-
decreasing inα.
The bound excludes final max aggregation and end-to-end
retrieval. It alsopermits over-suppression of distant relevant
states.
ChunkScoreProjectionandLocalizedApproximatePPR
STIBP scores are projected onto chunks and diffused on
aϕ-derived chunk graph without further LLM calls. Let
Cqcontain direct ANN candidates and chunks incident to
nonzero-influence states; projection and PPR are restricted
tothisquery-inducedset.Entityinfluenceisprojectedonto
chunks by
Score(c) =X
v∈V(c)a(v)·ln(1 +N v),
where Nv=|{w∈ V |ϕ(w) =ϕ(v)}| . We form the
non-negative prior score
eI(c) =λmax{0,sim(q, c)}+ ln(1 + Score(c)),
and normalize it overC qas
pST(c) =(eI(c)/P
z∈CqeI(z),ifP
zeI(z)>0,
1/|C q|,otherwise.
The uniform fallback handles an all-zero query; λ= 0.3
is selected as described in Section 4.4. The chunk graph
connects chunks sharing a name-equivalent entity:
B(c) ={c′|c′̸=c,∃u∈ V(c), v∈ V(c′) :ϕ(u) =ϕ(v)}.
LetPbe the column-normalized transition matrix, with
Pc′c′= 1for dangling chunks. With d= 0.85 , the PPR
iteration is
π(t)= (1−d)p ST+dPπ(t−1), π(0)=pST,
We iterate until max c|π(t)(c)−π(t−1)(c)|<10−6and
pass the top- kchunks ranked by π(T)to the generator. The
followingidentitysupportsthedesign.Itssufficientalignment
condition is not independently verified.
Proposition 3(Discounted Evidence-Mass Identity under
Prior Alignment).Let pSTbe the normalized chunk prior

inducedbySTIBPandlet pBbeanormalizedbinaryentity-
match prior. For a ground-truth evidence set R, define its
discounted evidence-reachability footprint as
hR= (1−d)∞X
ℓ=0dℓ(Pℓ)⊤1R,
where Pis the column-stochastic transition matrix of the
query-induced chunk graph. The evidence-mass difference is
exactly
MR(pST)−M R(pB) =h⊤
R(pST−pB).
Consequently, ifh⊤
R(pST−pB)≥γ >0, then
MR(pST)≥M R(pB) +γ,
where MR(p) =1⊤
Rπ(p)denotes the PPR evidence mass
assigned toRunder priorp.
Here hR(c)is the discounted forward-walk reachability
from cto the evidence set. The binary diagnostic does not
verify the alignment condition because reachability, seed
coverage,emptymatches,andsupportsizeareunmeasured
(Appendix D).
Symbolic indexing requires O(N C+N V+N E+M+P
gnglogn g)time and linear storage. After candidate for-
mation, STIBP and PPR scale with activated structures;
Appendix B.1 gives the full online-complexity expression.
4 Experiments
WeevaluateSTITCH-RAGalongfouraxes:end-to-endan-
swer accuracy (RQ1), answer quality beyond correctness
(RQ2), module-level causality (RQ3), and computational
efficiency (RQ4).
4.1 Experimental Setup
DatasetsWeusethreedatasetscoveringstructuredmulti-
hop QA and open-ended domain-specific questions. For Hot-
potQA(Yangetal.2018)and2WikiMultiHopQA(Hoetal.
2020),wefollowtheHippoRAG(Gutiérrezetal.2024)proto-
colanddrawastratifiedsampleof1,000questionsperdataset
(compositioninAppendixA).Thesampledquestionsarefixed
across methods. We report three repeated end-to-end runs,
not three resampled datasets; at temperature zero, residual
variation can arise from API and extraction-service nondeter-
minism.Mixcontains512open-endedquestionsandservesas
anexploratorycross-domainevaluation.BecauseMixlacks
supporting-fact annotationsand reports onlyLLM-Acc, itis
not an independent objective test of generalization.
BaselinesThebaselinesspanthemainRAGdesignchoices:
zero-shot generation and dense-retrieval RAG as lower
bounds; LightRAG (Guo et al. 2024) for relation extrac-
tion;LinearRAG(Zhuangetal.2025)asthecloseststructural
predecessor,usingPPRwithbinaryentity-matchinitialization
overaflatentitygraph;HippoRAG(Gutiérrezetal.2024)for
PPR-basedgraphmemory;andCog-RAG(Huetal.2026a)
andHyper-RAG(Fengetal.2026)asrecenthypergraph-based
methods.MetricsWe report three end-to-end metrics: Contain-Acc,
which tests whether the generated answer contains the ref-
erence substring; LLM-Acc, in which Qwen-Max judges
answers generated by Qwen3.5-flash; and EM, in which a
distilled answer must exactly match the reference. Because
Qwen-Max and Qwen3.5-flash belong to the same model
family,LLM-Acccanreflectsame-familyjudgepreference.
Contain-Acc and EM do not use an LLM judge and therefore
serveas controls.Wetreat Contain-Acc,EM,andRecall@8
as primary objective evidence on HotpotQA and 2Wiki, and
treat LLM-Acc and pairwise quality dimensions as auxiliary
preference-basedevaluations.Forretrieval,wereportpassage-
levelRecall@8(q) =|G q∩R8
q|/|G q|usingsupporting-fact
annotations.Thismetrictestswhetheranswer-qualitydiffer-
ences are consistent with retrieval coverage, but does not
isolate retrieval as the causal source. Mix lacks such annota-
tions, so no retrieval metric is reported for it.
Implementation DetailsAll methods use
text-embedding-v4 with 1024 dimensions and
Qwen3.5-flash at temperature zero. On a 200-question
HotpotQA development subset, replacing the original
embeddingsofLightRAGandLinearRAGchangesLLM-Acc
by at most 0.4 and 0.2 points, respectively. This check
limitsembeddingsensitivityforthesetwobaselinesonthat
subset, but does not establish the same property for every
baseline.STITCH-RAGuses k= 8,δ= 0.5,and d= 0.85,
selected on the same held-out set and fixed across datasets.
All experiments run through the Alibaba Cloud API; every
repeated run uses the same evaluation questions.
4.2 Generation Accuracy (RQ1)
STITCH-RAG ranks first in Contain-Acc and LLM-Acc
on every dataset in Table 1. Relative to the strongest per-
dataset baseline, the Contain-Acc/LLM-Acc margins are
3.9/0.8 points on HotpotQA and 5.6/1.1 points on 2Wiki.
The Mix LLM-Acc margin is 0.7 points. LinearRAG is the
closest structural comparison, and STITCH-RAG exceeds it
by 15.5and6.7Contain-Accpointson HotpotQAand2Wiki,
respectively. These aggregate differences do not isolate the
responsible component. LinearRAG has higher 2Wiki EM
(0.688versus0.670);withoutanswer-lengthstatisticsorerror
analysis, we report this as a limitation rather than attribute it
toresponseverbosity.Contain-AccandRecall@8providethe
strongest objective evidence. EM is mixed, and the smaller
LLM-Acc margins remain auxiliary.
Among methods with standardized retrieval outputs,
STITCH-RAGexceedsLinearRAGinRecall@8by7.5points
on HotpotQA and 5.7 points on 2Wiki. This pattern is con-
sistent with broader retrieval coverage, but the comparison
omits methods without standardized outputs and does not
show that retrieval alone causes the answer-accuracy gains.
The margins also match the mechanism in Proposition 3,
thoughtheexperimentdoesnotverifyitsalignmentcondition
becauseγis not measured per query.
4.3 Answer Quality Comparison (RQ2)
We apply the LightRAG pairwise protocol as an auxiliary
preference-based evaluation of comprehensiveness, diversity,

Table 1: Answer accuracy on HotpotQA, 2Wiki, and Mix. Best result per column bolded. Only STITCH-RAG was repeated end
to end; its entries average three runs on the same fixed question samples, with all run-level standard deviations below 0.004.
Baseline run-level variances are unavailable. The table therefore ranks reported point estimates but does not establish statistical
superiority.
Method HotpotQA 2Wiki Mix
Contain-Acc LLM-Acc EM Contain-Acc LLM-Acc EM LLM-Acc
Zero-shot 0.422 0.451 0.300 0.505 0.398 0.345 0.269
Standard-RAG 0.700 0.702 0.460 0.689 0.617 0.467 0.669
HippoRAG 0.690 0.835 0.609 0.555 0.575 0.476 0.823
Cog-RAG 0.822 0.843 0.562 0.768 0.700 0.554 0.831
Hyper-RAG 0.735 0.808 0.511 0.785 0.745 0.541 0.808
LightRAG 0.861 0.877 0.600 0.821 0.674 0.518 0.877
LinearRAG 0.745 0.887 0.649 0.810 0.8500.6880.762
STITCH-RAG 0.900 0.895 0.654 0.877 0.8610.6700.884
Table 2: Passage-level Recall@8 on HotpotQA and 2Wiki,
defined as |Gq∩R8
q|/|G q|where Gqis the ground-truth
supportingchunksetand R8
qisthetop-8retrievedset.Mixis
excludedbecausesupporting-factannotationsareunavailable.
Only methods with publicly available retrieval outputs under
the standardized embedding protocol are included.
Method HotpotQA 2Wiki
Standard-RAG 0.612 0.587
LightRAG 0.741 0.693
LinearRAG 0.723 0.712
STITCH-RAG 0.798 0.769
and empowerment. STITCH-RAG exceeds the 50% win-rate
lineonmostdataset–dimensionpairs(Figure3,AppendixC),
withthelargestmarginsincomprehensivenessanddiversity
onHotpotQAand2Wiki,consistentwiththeRecall@8results
inTable2.OnMix,LightRAGremainscompetitiveinselected
dimensions, plausibly because explicit relation extraction
canbetterencodestructureddomainknowledgewhentopic-
hyperedge boundaries are less distinct. Both rely on an LLM
judge and remain exploratory.
4.4 Ablation Study (RQ3)
Table 3 provides controlled evidence that STIBP and PPR
contribute non-redundant gains: removing STIBP/PPR re-
duces HotpotQA LLM-Acc by 7.2/7.0 points and 2Wiki
LLM-Acc by 15.4/6.8 points. The larger STIBP loss on
2Wiki is consistent with stronger cross-chunk entity-tracking
demands. Under the fixed pipeline, semi-merging achieves
higherpointestimatesthanbothcontrolledmergealternatives
(Appendix C), although this diagnostic does not establish
general optimality. The lower block shows that both bridging
channels and frequency-adaptive decay are beneficial; adap-
tivedecayreaches0.884onMix,comparedwith0.853and
0.846forfixedandlogarithmicscaling.Eachvariantretainsthe remaining pipeline components and changes only the
stated operation. Appendix C details the binary-initialization
diagnosticand itscontrols,whileAppendixAprovidesthe
extraction prompt and phase-level visualization.
Hyperparameter Sensitivity AnalysisWe select δ= 0.5,
λ= 0.3, and k= 8on the HotpotQA development set and
transfer them unchanged to 2Wiki and Mix. Appendix B.3
reportsthecompletesweeps;amongthetesteddecayrules,
adaptiveαattains the highest reported Mix LLM-Acc.
4.5 Efficiency Analysis (RQ4)
Table 4 pairs STITCH-RAG’s top Mix LLM-Acc with the
second-loweststructural-retrievallatency.Itsaddedcostoc-
cursduringone-timeindexingofcontext-awareentitydescrip-
tions,whereasstructuralretrievalissuesnoLLMAPIcalls.
Appendix B.1 gives the component-level cost decomposition.
5 Conclusion
STITCH-RAG couples topic-preserving hyperedges, semi-
merged local entity states, frequency-adaptive STIBP,
and continuous-prior PPR. Proposition 1 applies only to
provenance-discarding pairwise inputs, and Proposition 2
compares three controlled merge constructions. The decay
and PPR results characterize the proposed mechanisms; they
are not end-to-end optimality guarantees.
Under the reported protocol, STITCH-RAG attains the
highestContain-AccandLLM-AccpointestimatesinTable1
andthehighestRecall@8inTable2.Themergeandbinary
diagnostics remain inconclusive because baseline variance
andper-queryalignmentareunmeasured,2WikiEMisbelow
LinearRAG,andthemergecomparisondoesnotestablisha
compression–relevance optimum.
Current limitations include extraction and exact-name-
linking errors, preprocessing-order-sensitive index proximity,
unmeasured baselinevariance, same-family judgingon Mix,
and the approximately 15M-token Mix indexing cost. Future
work will study robust entity linking, order-insensitive prop-
agation, adaptive routing (Jeong et al. 2024), incremental

Table 3: Ablation of STITCH-RAG (three-run LLM-Acc average). The upper block removes STIBP/PPR or uses dense retrieval;
the lower block removes one STIBP channel or replaces continuous priors with binary initialization.
Variant HotpotQA LLM-Acc 2Wiki LLM-Acc Mix LLM-Acc
Full STITCH-RAG 0.895 0.861 0.884
w/o STIBP 0.823 0.707 0.800
w/o PPR 0.825 0.793 0.854
w/o all (dense retrieval) 0.761 0.684 0.672
w/o spatial bridging 0.838 0.742 0.814
w/o index-proximity bridging 0.840 0.748 0.815
Strict binary-only initialization diagnostic 0.459 0.365 0.469
Table 4: Efficiency and LLM-Acc on Mix. Indexing costs are one-time; query-stage tokens include answer generation and
method-specific query-time LLM calls under identical hardware and API conditions.
Method Time (s) Token Consumption LLM-Acc
Indexing Retrieval Idx. Prompt Idx. Completion Query Prompt Query Completion
HippoRAG 1706.91 39.56 1382281 1321272 55845.91 3774.76 0.823
Cog-RAG 19109.94 126.88 3082137 7988713 36840.47 13246.30 0.831
Hyper-RAG 70414.64 52.20 4652473 7264711 18557.08 5521.35 0.808
LightRAG 89882.40 75.87 6588007 8685853 29795.00 8148.52 0.877
LinearRAG 633.37 23.00 0 0 6395.72 261.50 0.762
STITCH-RAG 10384.41 31.12 1229660 14196127 4054.52 2558.72 0.884
indexing, and multimodal topic hyperedges.References
Asai, A.;Wu,Z.; Wang, Y.; Sil, A.;and Hajishirzi, H.2024.
Self-RAG: Learning to Retrieve, Generate, and Critique
through Self-Reflection. InThe Twelfth International Confer-
ence on Learning Representations.
Balaguer,A.;Benara,V.;deFreitasCunha,R.L.;Perrone,V.;
et al. 2024. RAG vs Fine-Tuning: Pipelines, Tradeoffs, and a
CaseStudyonAgriculture.arXivpreprintarXiv:2401.08406.
Chen,T.;Wang,H.;Chen,S.;Yu,W.;etal.2024. DenseX
Retrieval: What Retrieval Granularity Should We Use? In
Proceedingsofthe62ndAnnualMeetingoftheAssociation
for Computational Linguistics (ACL).
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody,
A.;Truitt,S.;Metropolitansky,D.;Ness,R.O.;andLarson,
J.2024. Fromlocaltoglobal:Agraphragapproachtoquery-
focused summarization.arXiv preprint arXiv:2404.16130.
Feng, Y.; Hu, H.; Ying, S.; Hou, X.; Liu, S.; Yang, M.; Li,
J.; Du, S.; Zheng, N.; Hu, H.; and Gao, Y. 2026. Hyper-
RAG: combating LLM hallucinations using hypergraph-
driven retrieval-augmented generation.Nature Communi-
cations.
Guo,Z.;Xia,L.;Yu,Y.;Ao,T.;andHuang,C.2024. Ligh-
trag:Simpleandfastretrieval-augmentedgeneration.arXiv
preprint arXiv:2410.05779.
Gutiérrez,B.J.;Shu,Y.;Qi,W.;Zhou,S.;andSu,Y.2025.
From RAG to Memory: Non-Parametric Continual Learning
for Large Language Models. InInternational Conference on
Machine Learning, 21497–21515. PMLR.
Gutiérrez, B. J.; Shu, Y.; Gu, Y.; Yasunaga, M.; and Su, Y.
2024. HippoRAG: Neurobiologically Inspired Long-Term

MemoryforLargeLanguageModels. InTheThirty-eighth
Annual Conference on Neural Information Processing Sys-
tems.
He,L.2026. CogniRAG:IntegratingCausalHyperedgesand
Counterfactual Reasoning for Knowledge-Intensive Tasks.
Information Technology and Control, 1(55): 298.
Ho, X.; Duong Nguyen, A.-K.; Sugawara, S.; and Aizawa, A.
2020. ConstructingAMulti-hopQADatasetforComprehen-
sive Evaluation of Reasoning Steps. InProceedings of the
28th International Conference on Computational Linguistics
(COLING), 6609–6625.
Hou, X.; Liu, Y.; Sun, Q.; Hu, H.; Du, S.; Tian, Z.; et al.
2026. IGMiRAG: Intuition-Guided Retrieval-Augmented
GenerationwithAdaptiveMiningofIn-DepthMemory.arXiv
preprint arXiv:2602.07525.
Hu, H.; Feng, Y.; Li, R.; Xue, R.; Hou, X.; Tian, Z.; Gao,
Y.; and Du, S. 2026a. Cog-rag: Cognitive-inspired dual-
hypergraph with theme alignment retrieval-augmented gener-
ation. InProceedings of the AAAI Conference on Artificial
Intelligence, volume 40, 31032–31040.
Hu,Y.;Zhu,J.;Tang,L.;andHuang,C.2026b. ReMindRAG:
Low-CostLLM-GuidedKnowledgeGraphTraversalforEf-
ficient RAG.Advances in Neural Information Processing
Systems, 38: 53757–53798.
Jeong,S.;Baek,J.;Cho,S.;Hwang,S.J.;andPark,J.C.2024.
Adaptive-RAG: Learning to Adapt Retrieval-Augmented
Large Language Models through Question Complexity. In
Proceedingsofthe2024ConferenceoftheNorthAmerican
Chapter of the Association for Computational Linguistics
(NAACL).
Kim,D.;Park,B.;Seo,D.;andKim,S.2024. AutoRAG:Au-
tomatedFrameworkforOptimizationofRetrievalAugmented
Generation Pipeline.arXiv preprint arXiv:2410.20878.
Li, M.; Miao, S.; and Li, P. 2025. Simple is effective: The
roles of graphs and large language models in knowledge-
graph-basedretrieval-augmentedgeneration. InInternational
ConferenceonLearningRepresentations,volume2025,6061–
6089.
Li, Z.; Ming, X.; Shang, M.; Cao, Y.; Qi, G.; Xu, S.; Li,
W.;andWang,Y.2024a. StructRAG:BoostingKnowledge
Intensive Reasoning of LLMs via Inference-time Hybrid
Information Structuring.arXiv preprint arXiv:2410.08815.
Li, Z.; Shi, C.; Xie, X.; Regan, S.; Hu, Y.; Wang, J.; Sun, H.;
Rossi,R.A.;Kim,B.;Dernoncourt,F.;Yu,T.;etal.2024b.
Retrieval Augmented Generation or Long-Context LLMs? A
ComprehensiveStudyandHybridApproach. InProceedings
of the 2024 Conference on Empirical Methods in Natural
Language Processing.
Luo, H.; E, H.; Chen, G.; Zheng, Y.; Wu, X.; Guo, Y.; Lin,
Q.; Feng, Y.; Kuang, Z.; Song, M.; Zhu, Y.; and Luu, A. T.
2025a. HyperGraphRAG: Retrieval-Augmented Generation
via Hypergraph-Structured Knowledge Representation. In
Belgrave, D.; Zhang, C.; Lin, H.; Pascanu, R.; Koniusz,
P.; Ghassemi, M.; and Chen, N., eds.,Advances in Neural
InformationProcessingSystems,volume38,152206–152234.
Curran Associates, Inc.Luo, L.; Zhao, Z.; Haffari, G.; Phung, D.; Gong, C.; and Pan,
S.2025b. GFM-RAG:GraphFoundationModelforRetrieval
Augmented Generation.NeurIPS 2025.
Ovadia, O.; Brief, M.; Mishaeli, M.; and Elisha, O. 2024.
Fine-TuningorRetrieval?ComparingKnowledgeInjectionin
LLMs. InProceedingsofthe2024ConferenceoftheNorth
American Chapter of the Association for Computational
Linguistics (NAACL).
Sarthi, P.; Abdullah, S.; Tuli, A.; Khanna, S.; Goldie, A.; and
Manning, C. 2024. Raptor: Recursive abstractive processing
fortree-organizedretrieval. InInternationalConferenceon
Learning Representations, volume 2024, 32628–32649.
Wu, K.; Kuai, C.; Li, Z.; Jiang, J.; Shen, S.; Wang, S.; Hu,
C.-W.; Tu, Z.; and Zhou, Y. 2026. Knowledge is not static:
Order-aware hypergraph rag for language models.arXiv
preprint arXiv:2604.12185.
Wu, W.; Wang, Y.; Xiao, G.; Peng, H.; and Fu, Y. 2024.
Retrieval Head Mechanistically Explains Long-Context Fac-
tuality.arXiv preprint arXiv:2404.15574.
Xiang, Z.; Wu, C.; Zhang, Q.; Chen, S.; Hong, Z.; Huang,
X.; and Su, J. 2025. When to Use Graphs in RAG: A
Comprehensive Analysis for Graph Retrieval-Augmented
Generation.arXiv preprint arXiv:2506.05690.
Xu, T.; Zheng, H.; Li, C.; Chen, H.; Liu, Y.; Chen, R.; and
Sun,L.2025. NodeRAG:StructuringGraph-basedRAGwith
Heterogeneous Nodes.arXiv preprint arXiv:2504.11544.
Yan, S.-Q.; Gu, J.-C.; Zhu, Y.; and Ling, Z.-H. 2024. Cor-
rective Retrieval Augmented Generation.arXiv preprint
arXiv:2401.15884.
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W. W.;
Salakhutdinov,R.;andManning,C.D.2018. HotpotQA:A
DatasetforDiverse,ExplainableMulti-hopQuestionAnswer-
ing. InProceedings of the 2018 Conference on Empirical
MethodsinNaturalLanguageProcessing(EMNLP),2369–
2380.
Zhuang,L.;Chen,S.;Xiao,Y.;Zhou,H.;Zhang,Y.;Chen,
H.; Zhang, Q.; and Huang, X. 2025. Linearrag: Linear graph
retrieval augmented generation on large-scale corpora.arXiv
preprint arXiv:2510.10114.

Reproducibility Checklist
This paper:
•Includes a conceptual outline and/or pseudocode descrip-
tion of AI methods introduced (yes/partial/no/NA)yes
•Clearlydelineatesstatementsthatareopinions,hypothesis,
andspeculationfromobjectivefactsandresults(yes/no)
yes
•Provides well-marked pedagogical references for less-
familiar readers to gain background necessary to replicate
the paper (yes/no)yes
Does this paper make theoretical contributions? (yes/no)
yes
If yes, please complete the list below.
•All assumptions and restrictions are stated clearly and
formally. (yes/partial/no)yes
•All novel claims are stated formally (e.g., in theorem
statements). (yes/partial/no)yes
•Proofs of all novel claims are included. (yes/partial/no)
yes
•Proofsketches orintuitionsaregiven forcomplexand/or
novel results. (yes/partial/no)yes
•Appropriatecitationstotheoreticaltoolsusedaregiven.
(yes/partial/no)yes
•All theoretical claims are demonstrated empirically to
hold. (yes/partial/no/NA)partial
•Allexperimentalcodeusedtoeliminateordisproveclaims
is included. (yes/no/NA)yes
Does this paper rely on one or more datasets? (yes/no)
yes
If yes, please complete the list below.
•A motivation is given for why the experiments are con-
ducted on the selected datasets (yes/partial/no/NA)yes
•All novel datasets introduced in this paper are included in
a data appendix. (yes/partial/no/NA)yes
•All novel datasetsintroduced inthis paper willbe made
publicly available upon publication of the paper with
a license that allows free usage for research purposes.
(yes/partial/no/NA)yes
•All datasets drawn from the existing literature (potentially
including authors’ own previously published work) are
accompanied by appropriate citations. (yes/no/NA)yes
•All datasets drawn from the existing literature (potentially
including authors’ own previously published work) are
publicly available. (yes/partial/no/NA)yes
•Alldatasetsthatarenotpubliclyavailablearedescribedin
detail,withexplanationwhypubliclyavailablealternatives
are not scientifically satisficing. (yes/partial/no/NA)NA
Does this paper include computational experiments?
(yes/no) yes
If yes, please complete the list below.
•This paper states the number and range of values tried per
(hyper-)parameterduringdevelopmentofthepaper,along
with the criterion used for selecting the final parameter
setting. (yes/partial/no/NA)yes•Anycoderequiredforpre-processingdataisincludedin
the appendix. (yes/partial/no)no
•All source code required for conducting and analyzing
theexperimentsisincludedinacodeappendix.(yes/par-
tial/no)no
•All source code required for conducting and analyzing
the experiments will be made publicly available upon
publication of the paper with a license that allows free
usage for research purposes. (yes/partial/no)yes
•All source code implementing new methods have com-
mentsdetailingtheimplementation,withreferencestothe
paper where each step comes from. (yes/partial/no)yes
•If an algorithm depends on randomness, then the method
used for setting seeds is described in a way sufficient to
allow replication of results. (yes/partial/no/NA)NA
•This paper specifies the computing infrastructure used for
running experiments (hardware and software), including
GPU/CPU models; amount of memory; operating system;
names and versions of relevant software libraries and
frameworks. (yes/partial/no)partial
•This paper formally describes evaluation metrics used
and explains the motivation for choosing these metrics.
(yes/partial/no)yes
•This paper states the number of algorithm runs used to
compute each reported result. (yes/no)no
•Analysisof experimentsgoes beyond single-dimensional
summaries of performance (e.g., average; median) to
include measures of variation, confidence, or other distri-
butional information. (yes/no)yes
•Thesignificanceofanyimprovementordecreaseinper-
formance is judged using appropriate statistical tests (e.g.,
Wilcoxon signed-rank). (yes/partial/no)no
•This paper lists all final (hyper-)parameters used for each
model/algorithm in the paper’s experiments. (yes/par-
tial/no/NA)yes

Appendices
A Reproducibility and Implementation
Details
Dataset sample composition.The stratified 1,000-question
samplescontainapproximately520bridgeand480compar-
ison questions in HotpotQA, and approximately 280 com-
parison, 250 inference, 240 compositional, and 230 bridge
questions in 2Wiki. Mix is reported only in aggregate, so
itsresultsareexploratory.Acompletereleaseshoulddocu-
ment domain counts, corpus and chunk sizes, question and
reference-answerprovenance,constructionanddeduplication
procedures, train/test isolation checks, and data licenses.
Reproducibilityofbaselines.ForLightRAG1,Cog-RAG2,
Hyper-RAG3, and LinearRAG4, we use the authors’ released
implementationsandreplaceonlytheembeddingmodelwith
text-embedding-v4 to standardize vector representa-
tions.
Baseline verification protocol.For Cog-RAG and Hyper-
RAG,wefirstreproducethereportedresultsundertheoriginal
evaluation settings, including the default embedding models
andgenerationbackbones.Allreproducedmetricsfallwithin
1.5percentagepointsofthereportedvalues;theremainingde-
viations may reflect API-version or random-seed differences.
Wethenreruneverybaselineunderthestandardizedprotocol
in Section 4.1.
Revieweraccessandverification.Thesupplementaryma-
terial provides frozen code snapshots with written sharing
permission,checksummedDockerimageswithpinneddepen-
dencies,retrievedchunksandgeneratedanswersforall1,000
questionsoneachdataset,andacomparisonlogfortherepro-
ductioncheck.Theanonymousrepositoryincludeswrapper
scripts, evaluation scripts, and VERIFICATION.md with
checksums and expected metric outputs.
Complete extraction prompt pext.The prompt below ex-
tracts topic summaries and chunk-conditioned entity descrip-
tions. It is applied independently to each chunk and receives
no cross-chunk context.
Listing 1: Extraction promptp ext
1---Goal---
2Extract the knowledge expressed in the
task text. Divide the text into self-
contained knowledge segments. For
each segment, return:
3- reference: the supporting text span.
4- knowledge: one self-contained sentence
describing the segment.
5- entities: an object mapping each
entity’s complete name to its context
-specific description.
6Notes:
7- Do not omit information stated in the
original text.
8- Each knowledge sentence must be
understandable without additional
1https://github.com/HKUDS/LightRAG
2https://github.com/haoohu/Cog-RAG
3https://github.com/iMoonLab/Hyper-RAG
4https://github.com/DEEP-PolyU/LinearRAGcontext.
9- Return only valid JSON that can be
parsed by JSON.parse().
10- If an entity description is absent or
only repeats the name, use an empty
string ("").
11- Ground every description in the task
text.
12- Use the complete entity name as the
key and include that complete name in
its non-empty description.
13
14---Output JSON format---
15Return one or more objects in a JSON
array following this structure:
16[
17{
18"reference": "Original supporting
text",
19"knowledge": "A self-contained
knowledge statement.",
20"entities": {
21"Entity 1": "Entity 1 is described
in the task text.",
22"Entity 2": "Entity 2 is described
in the task text."
23}
24}
25]
26
27######################
28Task text:
29{content}
30Output:
Eachextractedknowledgesegmentcreatesonetopic-summary
hyperedge ek,r,whosetextisthesegment’sknowledgesen-
tence. Each entry in its entities field creates an incident
entity-state nodev= (vname, vdes).
Extraction convention summary.The extraction procedure
uses three conventions. First, entity boundaries follow the
maximal-noun-phraserule:NewYorkCityisoneentityrather
thanthreetokens.Second,eachtopicrepresentsonecoherent
claimornarrativethread;achunkcoveringaperson’searly
careerandlaterachievementsshouldyieldtwotopicsegments.
Third, entitydescriptions aregrounded in thecurrent chunk,
so repeated entities receive role-specific local descriptions.
B Algorithmic Details
B.1 Complexity Analysis
LetNC,NV=|V|,NE=|E|, and M=P
e∈E|e|denote
thenumbersofchunks,entity-statenodes,topichyperedges,
and entity–hyperedge incidence links. The symbolic com-
plexitiesexcludeLLMinferenceandembeddingexecution,
which are model-dependent and not assumed to be shared
across systems. Table 4 reports measured time and token
consumption separately.
Offline indexing.Topic incidence storage requires O(M)
links; ϕ-based grouping requires O(N V)hash opera-
tions; within-group sorting by chunk index requires
O(P
gnglogn g)where ngisthesizeofgroup g.Thetotal

Algorithm 1: STITCH-RAG: Retrieval Pipeline
Require: Query q; semi-merged hypergraph H=
(V,E, ϕ, ψ) ; chunk set C; parameters δ,λ,d,k; ANN
candidate budgetsK V, KE, KC
Ensure:Top-kretrieved chunksCk
q
1:(V0
q, E0
q, C0
q)←ANN(q;V,E, C, K V, KE, KC)
2:foreach entity nodev∈V0
qdo
3:A q(v)←sim(q, vdes)·1[sim(q, vdes)> δ]
4:end for
5:foreach hyperedgee∈E0
q∪ {e:e∋v, v∈V0
q}do
6:S q(e)←sim(q, e)
7:end for
8:Initializeaspace(v)←0,atime(v)←0for allv
9:foreach activated nodeuwithA q(u)>0do
10:foreach hyperedgee∋udo
11:foreachv∈V e\ {u}do
12:aspace(v)←max 
aspace(v), A q(u)·S q(e)
13:end for
14:end for
15:V∗
u← {v∈ V |ϕ(v) =ϕ(u)} {name-equivalence
group}
16:α← |V∗
u|/¯n{frequency-adaptive decay rate}
17:foreachv∈V∗
u\ {u}do
18:∆t← |t u−tv|
19:atime(v)←max 
atime(v), A q(u)·S q(ev)·
exp(−tanh(α∆t))·sim(q, c v)
20:end for
21:end for
22:a(v)←mean 
Aq(v), aspace(v), atime(v)
for allv
23:C q←C0
q∪ {c v:a(v)>0}
24:foreach chunkc∈C qdo
25:Score(c)←P
v∈V(c)a(v)·ln(1 +N v)
26:eI(c)←λmax{0,sim(q, c)}+ ln(1 + Score(c))
27:end for
28:p ST(c)←eI(c)/P
z∈CqeI(z)if the sum is positive; oth-
erwisep ST(c)←1/|C q|
29:Build chunk graph: B(c)← {c′| ∃u∈ V(c), v∈
V(c′)s.t.ϕ(u) =ϕ(v)}
30:Column-normalizetheadjacencyas P;setPcc= 1when
B(c) =∅
31:π(0)←p ST
32:repeat
33:foreach chunkc∈C qdo
34:π(t)(c)←(1−d)p ST(c) +
dP
c′∈CqPcc′π(t−1)(c′)
35:end for
36:untilmax c|π(t)(c)−π(t−1)(c)|<10−6
37:Ck
q←top-kchunks byπ(T)(c)
38:returnCk
q
symbolic indexing cost is
O
NC+NV+NE+M+P
gnglogn g
,
and storage is O(N C+NV+NE+M). All indexing costsAlgorithm2:STITCH-RAG:OfflineHypergraphConstruc-
tion
Require: Document collection D; max chunk size cmax;
extraction promptp ext
Ensure:Semi-merged hypergraphH= (V,E, ϕ, ψ)
1:SplitDinto chunksC={c k}NC
k=1with|c k| ≤c max
2:foreach chunkc k∈Cdo
3:{g k,r}r←LLM(c k|pext){extract (topic, entities)
tuples}
4:foreach tupleg k,r= (e k,r, Vek,r)do
5:Add hyperedgee k,rtoEwith topic summary text
6:foreach entity(vname, vdes)∈V ek,rdo
7: Createnode vwithϕ(v)←vname,ψ(v)←vdes
8: AddvtoV; record incidence (v, e k,r)and chunk
membership(v, c k)
9:end for
10:end for
11:end for
12:Group nodes by canonical name: {Gs={v|ϕ(v) =
s}}s∈Σ
13:Sort nodes within each group by the deterministic pre-
processing indext v
14:Embed all entity descriptions {vdes}, topic summaries
{e}, and chunks{c k}
15:returnH= (V,E, ϕ, ψ)
are incurred once.
Online retrieval.Let Aq,Eq, and Bqdenote the activated
entity-state set, incident topic hyperedges, and edge set of
the query-induced chunk graph. Let Tcand(NV, NE, NC)
denotecandidate-generationcostovernode,hyperedge,and
chunk vector indexes. Under exact search, it is O((N V+
NE+NC)demb);ANNcostdependsontheselectedindex
and approximation parameters. Spatial propagation costs
O(P
e∈Eq|e|),temporal propagationcosts O(P
u∈Aq|V∗
u|)
withoutmaterializingpairwiselinks,andlocalizedPPRcosts
O(T|B q|)forTiterations. The total online cost is
O
Tcand(NV, NE, NC) +X
e∈Eq|e|
+X
u∈Aq|V∗
u|+T|B q|
.
Candidate generation remains corpus-dependent, whereas
propagation depends on the activated subgraph. Table 4
reportsmeasuredlatencyratherthanextrapolatingend-to-end
scaling from propagation terms alone.
B.2 Qualitative Compression–Context
Interpretation of Semi-Merging
The three controlled merge strategies trade compact identity
representationagainstlocal-contextretention.Fullmerging
collapsesname-equivalentmentionsandremovesper-chunk
descriptions.Nomergingretainseachlocaldescriptionbut
exposes no cross-chunk name-equivalence relation. Semi-

merging retains distinct local states while exposing their
operational name equivalence to retrieval.
This interpretation motivates the representation design but
is not an information-bottleneck result: the paper neither
defines nor estimates mutual information for the three con-
structions.Table8isananswer-level controlledcomparison
anddoesnotdirectlymeasurecompressionorretainedtask
information.
B.3Complete Hyperparameter Sensitivity Results
Tables 5 and 6 report post-hoc transfer sensitivity of the
activation threshold δand balance parameter λon Mix. Bold
values were chosen on the HotpotQA development set, not
onMix.Table7reportsthetop- ksweeponthat200-question
development set. Each sweep fixes all other hyperparameters
andtheretrievalpipelineattheirdefaults( k= 8,d= 0.85,
adaptiveα=n u/¯n).
Table 5: Post-hoc Mix LLM-Acc sensitivity to δ. The bold
row denotes the value preselected on HotpotQA, not a value
selected on Mix. All other hyperparameters are fixed.
δMix
0.3 0.861
0.4 0.815
0.50.884
0.6 0.838
0.7 0.861
Table 6: Post-hoc Mix LLM-Acc sensitivity to λ. The bold
row denotes the value preselected on HotpotQA, not a value
selected on Mix. All other hyperparameters are fixed.
λMix
0.1 0.823
0.30.884
0.5 0.846
0.7 0.846
0.9 0.861
Table7:LLM-Accundervaryingtop- kretrievalcountonthe
200-questionHotpotQAdevelopmentset.Theboldcolumn
denotes the selected value. All other hyperparameters are
fixed at their defaults.
k5 6 789 10
LLM-Acc 0.823 0.853 0.8610.8840.869 0.876
C Additional Experimental Results
This appendix contains the phase-level ablation visualization,
pairwise answer-quality comparison, merge-strategy diag-
nostic, and temporal-decay comparison cited in the main
text.Table 8: LLM-Acc point estimates for one implementation
ofeachmergestrategyunderafixedretrievalpipeline.The
diagnosticdoesnotevaluaterun-levelvariance,retrievalrecall,
graph compression, statistical significance, or all possible
implementations of the three strategies.
Strategy HotpotQA 2Wiki Mix
Full-Merge 0.873 0.839 0.807
Semi-Merge0.895 0.861 0.884
No-Merge 0.879 0.844 0.831
Table 9: LLM-Acc onMix under three temporal decay func-
tions, with all other pipeline components held fixed (same
hypergraph, same PPR initialization, sameα=n u/¯n).
Decay Function LLM-Acc
exp(−α∆t)(pure exponential) 0.853
σ(−α∆t+b)(sigmoid gate) 0.815
exp(−tanh(α∆t))(ours)0.884
D Proofs of Main Results
This appendix states the definitions used by Propositions 1
and 2, then provides all proofs.
Definition 1(Pairwise Topic Projection).Given a topic
hypergraph H= (V,E) , its pairwise projection is the graph
Π(H) = (V,E 2),
E2=
{u, v} |u̸=v,∃e∈ Es.t.u, v∈e	
.
Theprojectionretainspairwiseco-occurrencebutdiscards
the identity of the topic hyperedge that generated each pair.
Definition 2(Semi-Merged Hypergraph).Let H=
(V,E, ϕ, ψ) denoteasemi-mergedhypergraph,where Visthe
setofentitynodes, Eisthesetofhyperedges, ϕ:V →Σ maps
eachnodetoanentitynameinnamespace Σ,andψ:V → D
mapseachnodetoacontext-specificdescriptionindescrip-
tion space D. Two nodes vi, vj∈ Vsatisfy ϕ(vi) =ϕ(v j)
when theirnormalized canonical-namestrings areidentical.
This operational relation approximates, but does not guaran-
tee, real-world co-reference: aliases can produce false splits,
whereas homonyms can produce false links. The equivalence
classes underϕare called name-equivalence groups.
Definition 3(Full-Merge and No-Merge Hypergraphs).A
full-merge hypergraph HFcollapses all nodes within each
name-equivalence group into a single node and discards
per-chunk descriptions. A no-merge hypergraph HNtreats
everyextractedentitymentionasanindependentnodewith
nocross-chunkidentitylinkage;equivalently, HNdoesnot
define a mapϕacross chunks.
Proof of Proposition 1. Weconstructtwotopichypergraphs
with identical pairwise projections and exhibit a query under
which topic-selective spatial propagation produces different
influence assignments, establishing the non-identifiability
claim.

/uni00000032/uni00000059/uni00000048/uni00000055/uni00000044/uni0000004f/uni0000004f/uni00000003/uni0000003a/uni0000004c/uni00000051/uni00000051/uni00000048/uni00000055 /uni00000027/uni0000004c/uni00000059/uni00000048/uni00000055/uni00000056/uni0000004c/uni00000057/uni0000005c
/uni00000028/uni00000050/uni00000053/uni00000052/uni0000005a/uni00000048/uni00000055/uni00000050/uni00000048/uni00000051/uni00000057 /uni00000026/uni00000052/uni00000050/uni00000053/uni00000055/uni00000048/uni0000004b/uni00000048/uni00000051/uni00000056/uni0000004c/uni00000059/uni00000048/uni00000051/uni00000048/uni00000056/uni00000056/uni00000015/uni00000013/uni00000008/uni00000017/uni00000013/uni00000008/uni00000019/uni00000013/uni00000008/uni0000001b/uni00000013/uni00000008/uni00000014/uni00000013/uni00000013/uni00000008
/uni00000030/uni0000004c/uni0000005b
/uni00000032/uni00000059/uni00000048/uni00000055/uni00000044/uni0000004f/uni0000004f/uni00000003/uni0000003a/uni0000004c/uni00000051/uni00000051/uni00000048/uni00000055 /uni00000027/uni0000004c/uni00000059/uni00000048/uni00000055/uni00000056/uni0000004c/uni00000057/uni0000005c
/uni00000028/uni00000050/uni00000053/uni00000052/uni0000005a/uni00000048/uni00000055/uni00000050/uni00000048/uni00000051/uni00000057 /uni00000026/uni00000052/uni00000050/uni00000053/uni00000055/uni00000048/uni0000004b/uni00000048/uni00000051/uni00000056/uni0000004c/uni00000059/uni00000048/uni00000051/uni00000048/uni00000056/uni00000056/uni00000015/uni00000013/uni00000008/uni00000017/uni00000013/uni00000008/uni00000019/uni00000013/uni00000008/uni0000001b/uni00000013/uni00000008/uni00000014/uni00000013/uni00000013/uni00000008
/uni00000015/uni0000003a/uni0000004c/uni0000004e/uni0000004c
/uni00000032/uni00000059/uni00000048/uni00000055/uni00000044/uni0000004f/uni0000004f/uni00000003/uni0000003a/uni0000004c/uni00000051/uni00000051/uni00000048/uni00000055 /uni00000027/uni0000004c/uni00000059/uni00000048/uni00000055/uni00000056/uni0000004c/uni00000057/uni0000005c
/uni00000028/uni00000050/uni00000053/uni00000052/uni0000005a/uni00000048/uni00000055/uni00000050/uni00000048/uni00000051/uni00000057 /uni00000026/uni00000052/uni00000050/uni00000053/uni00000055/uni00000048/uni0000004b/uni00000048/uni00000051/uni00000056/uni0000004c/uni00000059/uni00000048/uni00000051/uni00000048/uni00000056/uni00000056/uni00000015/uni00000013/uni00000008/uni00000017/uni00000013/uni00000008/uni00000019/uni00000013/uni00000008/uni0000001b/uni00000013/uni00000008/uni00000014/uni00000013/uni00000013/uni00000008
/uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024/uni00000018/uni00000013/uni00000008/uni00000003/uni0000003a/uni0000004c/uni00000051/uni00000003/uni00000035/uni00000044/uni00000057/uni00000048/uni00000003/uni0000002f/uni0000004c/uni00000051/uni00000048 /uni00000059/uni00000056/uni00000003/uni0000002f/uni0000004c/uni0000004a/uni0000004b/uni00000057/uni00000035/uni00000024/uni0000002a /uni00000059/uni00000056/uni00000003/uni00000036/uni00000057/uni00000044/uni00000051/uni00000047/uni00000044/uni00000055/uni00000047/uni00000035/uni00000024/uni0000002a /uni00000059/uni00000056/uni00000003/uni0000002b/uni0000005c/uni00000053/uni00000048/uni00000055/uni00000010/uni00000035/uni00000024/uni0000002a /uni00000059/uni00000056/uni00000003/uni0000002f/uni0000004c/uni00000051/uni00000048/uni00000044/uni00000055/uni00000035/uni00000024/uni0000002a /uni00000059/uni00000056/uni00000003/uni00000026/uni00000052/uni0000004a/uni00000010/uni00000035/uni00000024/uni0000002a /uni00000059/uni00000056/uni00000003/uni0000002b/uni0000004c/uni00000053/uni00000053/uni00000052/uni00000035/uni00000024/uni0000002aFigure3:Pairwiseanswer-qualitywinratesofSTITCH-RAGagainstfivebaselinesonHotpotQA,2Wiki,andMix,judgedby
Qwen-Maxoncomprehensiveness(coverageofrelevantdetails),diversity(varietyofusefulperspectives),andempowerment
(utility for informed judgment). The dashed circle marks the 50% win-rate line. Values outside it favor STITCH-RAG.
Lettheentitysetbe V={a, b, c, d} .Considertwohyper-
graphs:
H1= 
V,{{a, b, c},{a, b, d},{a, c, d},{b, c, d}}
and
H2= 
V,{{a, b, c, d}}
.
Everypairfrom {a, b, c, d} co-occursinatleastonehyperedge
ofH1(each triple contains all 3
2
= 3pairs, and the four
triples together cover all 4
2
= 6pairs), and every pair co-
occursinthesinglehyperedgeof H2.Hencebothhypergraphs
project to the complete graph on four vertices:
Π(H 1) = Π(H 2) =K 4.
Fixa query qthat issemantically relevantonlyto the topic
{a, b, c},andlet Aq(a) = 1betheonlyactivatedsourcenode.
InH 1, assign topic relevance scores
Sq({a, b, c}) = 1, S q(e) = 0for all othere∈ E 1.
Thespatialbridgingformula aspace(v) =A q(a)·S q(e)for
eachv∈V e\ {a}gives
aspace(b) = 1, aspace(c) = 1, aspace(d) = 0,
since ddoes not appear in any hyperedge ewithSq(e)>0
underH 1.
InH2, the only available hyperedge is {a, b, c, d} . Two
exhaustivecasesarise.If Sq({a, b, c, d})>0 ,spatialbridging
fromaassignsthesamepositivescore Aq(a)·S q({a, b, c, d})
to each of b,c, and d, sodreceives positive influence. If
Sq({a, b, c, d}) = 0 ,noinfluencepropagatesatall,so band
creceivezeroinfluence.Inneithercasecan H2reproducethe
selectiveoutcome aspace(b)>0,aspace(c)>0,aspace(d) =
0.
Since Π(H 1) = Π(H 2), any retrieval rule whose input
isrestrictedto Π(H)mustproduceidenticaloutputson H1
andH2, yet the propagation outcomes differ for the query
qconstructed above. Therefore, no retrieval rule defined
solelyonthepairwiseprojectioncan,ingeneral,reproduce
topic-selective propagation over hyperedges.Proof of Proposition 2. We verify the influence received by
vrundereachofthethreeconstructions,holding q,G,andall
activation scores fixed throughout. The assumptions in force
are:Aq(vs)> δ;Aq(vr)≤δ,soafterthresholding Aq(vr) =
0;vris not reachable from vsthrough any activated topic
hyperedge; Sq(evr)>0forsomehyperedge evrcontaining
vr; andsim(q, c vr)>0.
No-mergehypergraph HN.InHN,themap ϕisnotdefined
acrosschunks,soeveryentitymentionisanindependentnode.
The corresponding name-equivalence group of vsunder the
no-mergeconstructionreducestothesingleton V∗
vs={v s},
sothetemporalbridgingloopdoesnotvisit vr.Byhypothesis,
vris not reachable from vsthrough any activated topic
hyperedge, so aspace
HN(vr) = 0. Combined with Aq(vr) = 0,
the mean aggregation gives aHN(vr) = 0. The evidence
chunk cvrtherefore receives zero projected influence from
vrand is structurally unreachable fromv sunderH N.
Full-mergehypergraph HF.Allnodesin Garecollapsed
into a single representative ¯v= merge(v 1, . . . , v m)with
aggregated description ¯vdesand membership in the union of
all original hyperedges and chunks. Two cases arise.
(2a)IfAq(¯v) = sim(q,¯vdes)> δ, then ¯vis activated. Every
chunk containing any original member of Greceives
the same influence through ¯v, because the per-chunk
descriptions ψ(vi)have been discarded and replaced by
the single aggregated description. The projected score of
cvrunderHFtherefore equals theprojected score of any
other chunk cvjcontaining a member of G, regardless of
whether vjis evidence-bearing. The retrieval algorithm
cannot preferentially assign state-specific influence to cvr
overc vjusing entity-state information.
(2b)IfAq(¯v)≤δ, then ¯vis not activated and no chunk
inG’s scope receives entity-level influence. This arises
when theaggregated description ¯vdesaverages over both
query-relevantandquery-irrelevantmentionsoftheentity,

dilutingthecosinesimilaritybelowtheactivationthreshold
δeven though vswould have been activated had its per-
chunk description been retained.
In either case, HFcannot assign state-specific positive
influence to vr: in (2a) influence is distributed uniformly
across allG-members, and in (2b) it is zero.
Semi-merged hypergraph H.InH, nodes vsandvrre-
taindistinctdescriptions ψ(vs)andψ(vr),while ϕexposes
their name-equivalence relationship ϕ(vs) =ϕ(v r). The
name-equivalence group satisfies V∗
vs={v∈ V |ϕ(v) =
ϕ(vs)} ∋v r.Since Aq(vs)> δ >0 ,node vsisactivatedand
temporal bridging iterates over V∗
vs. For target vrat chunk-
index distance ∆t=|t vs−tvr|and frequency-adaptive rate
α=|V∗
vs|/¯n, the STIBP formula gives
atime
vs(vr) =A q(vs)·Sq(evr)·exp 
−tanh(α∆t)
·sim(q, c vr).
Hereevrdenotesahyperedgecontaining vrwithSq(evr)>0;
such a hyperedge exists by assumption. If vrbelongs to
multiple hyperedges, the max operation in STIBP ensures
the stored atime(vr)is at least the source-specific value
computed with this evr, so it suffices to verify positivity
for one such hyperedge. Each factor is strictly positive:
Aq(vs)> δ >0 by hypothesis; Sq(evr)>0by assump-
tion; exp(−tanh(α∆t))>0 because tanh(x)<1 for
every finite xand the exponential is always positive; and
sim(q, c vr)>0by assumption. Hence atime
vs(vr)>0and
thereforeatime(vr)>0.
Since Aq(vr) = 0after thresholding and aspace(vr) = 0
by hypothesis, the mean aggregation gives
a(vr) = mean 
Aq(vr)|{z}
= 0, aspace(vr)|{z}
= 0, atime(vr)|{z}
>0
=1
3atime(vr)>0.
Because a(vr)>0, chunk cvrreceives strictly positive
projected influence via vrand enters the PPR initialization
with non-zero prior weight.
Recoverability separation.Combining the three cases,
aHN(vr) = 0,
aHF(vr)is undifferentiated or zero,
aH(vr)>0.
Among these three constructions, the semi-merged hyper-
graph is the only one satisfying both requirements simultane-
ously: ϕexposes the name-equivalence relation needed for
temporalbridging,and ψretainstheper-chunk descriptions
neededforstate-specificactivation,completingtheproof.
Proof of Theorem 1. We establish a lower bound for the
contributionspropagatedfromonefixedactivatedsource uby
bounding max r∈Ratime
u(r)frombelowandP
n∈Natime
u(n)
fromabove.Throughout, Aq(u) =a >0 ,α=|V∗
u|/¯n >0,
and the nonempty sets RandNsatisfy the conditions in the
theorem. Because gn>0for every n∈ N, the denominator
is positive.Lower bound for relevant targets.For any r∈ R,
the definition of atime
u(r)and the assumption gr=
Sq(er) sim(q, c r)≥βgive
atime
u(r) =a g rexp(−tanh(α∆ r))
≥aβexp(−tanh(α∆ r)).
Since tanh(x)∈[0,1) for all finite x≥0, we have
−tanh(α∆ r)>−1, soexp(−tanh(α∆ r))> e−1for ev-
eryfinite ∆r.Here ∆risafinitenon-negativeintegerbecause
chunk indices are bounded, and α >0by assumption. There-
fore
atime
u(r)≥aβe−1for allr∈ R,
and in particular
max
r∈Ratime
u(r)≥aβe−1.
Upperboundfornoisytargets.Forany n∈ N,theassump-
tions0< g n≤ϵand∆ n≥τ >0give
atime
u(n) =a g nexp(−tanh(α∆ n))
≤aϵexp(−tanh(α∆ n)).
The function x7→exp(−tanh(αx)) is non-increasing for
x≥0when α >0, because tanh(αx) is non-decreasing
inx. Since ∆n≥τ, we obtain exp(−tanh(α∆ n))≤
exp(−tanh(ατ)), and hence
atime
u(n)≤aϵexp(−tanh(ατ))for alln∈ N.
Summing overN:
X
n∈Natime
u(n)≤a|N|ϵexp(−tanh(ατ)).
Signal-to-noise ratio.Combining the two bounds:
max r∈Ratime
u(r)P
n∈Natimeu(n)≥aβe−1
a|N|ϵexp(−tanh(ατ))
=β
|N|ϵexp(tanh(ατ)−1).
Monotonicity in α.For fixed R,N,β,ϵ,τ, and ¯n, the
function α7→tanh(ατ) is strictly increasing for α≥0.
Therefore exp(tanh(ατ)−1) is non-decreasing in α. This
statement holds with the sets and constants fixed and does
not compare realized ratios across name-equivalence groups
of different sizes.
Proof of Proposition 3. Theproofproceedsbyshowingthat
PPR evidence mass is linear in the initialization prior, then
applying linearity to comparep STandp Bdirectly.
Linearity of PPR evidence mass.The PPR stationary dis-
tribution π(p)under prior pis the unique fixed point of
π= (1−d)p+dPπ .SolvingbyNeumannseries,whichcon-
verges absolutely because d <1andPis column-stochastic
with∥P∥ 1= 1:
π(p) = (1−d)∞X
ℓ=0dℓPℓp.

The series converges in ℓ1norm because (1−
d)P∞
ℓ=0dℓ∥Pℓp∥1≤(1−d)P∞
ℓ=0dℓ∥p∥1=∥p∥ 1<∞
for any normalized prior p, justifying the interchange of
summation and inner product below. Therefore,
MR(p) =1⊤
Rπ(p) =1⊤
R(1−d)∞X
ℓ=0dℓPℓp
= (1−d)∞X
ℓ=0dℓ1⊤
RPℓp.
For each ℓ, the scalar 1⊤
RPℓpequals ((Pℓ)⊤1R)⊤pbecause
u⊤Av= (A⊤u)⊤vfor any compatible vectors and matrix.
Hence,
MR(p) = 
(1−d)∞X
ℓ=0dℓ(Pℓ)⊤1R!⊤
p=h⊤
Rp,
wheretheinterchangeofthefinitelinearfunctional (·)⊤pand
the absolutely convergent series is valid by the dominated
convergence theorem for sums.
Evidence-mass comparison.Since MR(p) =h⊤
Rpis linear
inp,
MR(pST)−M R(pB) =h⊤
R(pST−pB).
The conditionh⊤
R(pST−pB)≥γ >0yields immediately
MR(pST)≥M R(pB) +γ.
Interpretation of hR.The c-th entry of hRequals (1−
d)P∞
ℓ=0dℓ[(Pℓ)⊤1R]c,theprobabilitythataforwardwalk
initializedatchunk coccupies Rwhenstoppedafterageo-
metrically distributed number oftransitions. The condition
h⊤
R(pST−pB)>0thereforeholdswheneverSTIBPplaces
more prior mass on chunks with high discounted reachability
toRthanbinaryinitializationdoes.Theaggregateablation
inTable3isconsistentwiththismechanism,butitdoesnot
establish the alignment condition for every query.
Prior and dangling-node convention.The retrieval pro-
cedure directly uses the non-negative normalized prior
pST(c) =eI(c)/P
zeI(z)when the denominator is positive,
with the same uniform fallback used in the main text oth-
erwise. For every dangling chunk c, the self-loop Pcc= 1
makes Pcolumn stochastic. Thus the assumptions used in
the proposition, the main-text iteration, and Algorithm 1 are
identical; no post-hoc normalization argument is required.
Geometricallystoppedforward-walkreadingof hR.Let
a walk start at chunk cand stop at its current state with
probability 1−dbefore each transition. It stops after ℓ
transitionswithprobability (1−d)dℓ.Consequently, hR(c)
is exactly the probability that the walk occupies Rwhen it
stops.Thisinterpretationfollowsdirectlyfromthediscounted
occupancyseriesanddoesnotrequireareverse-timetransition
process.E Extended Related Work
Thisappendixexpandsthetwocomparisonsthatdetermine
STITCH-RAG’s design: what the index retains about a
multi-entity topic and how the retriever initializes propaga-
tion.Wecomparepublishedrepresentationsandalgorithms;
provenance-enriched variants outside those specifications are
not ruled out.
E.1 Representing Topic Provenance and Local
Entity State
GraphRAG extracts entity–relation triples and summarizes
graphcommunities(Edgeetal.2024).LightRAGreducesthis
indexingburdenwithadual-layergraph(Guoetal.2024),and
NodeRAG uses heterogeneous nodetypesto retain multiple
evidence granularities (Xu et al. 2025). These systems can
encoderichpairwiserelations.Thedistinctionrelevanthere
is narrower: if retrieval receives an unlabeled projection with
no shared event, topic identifier, or provenance, it cannot
reconstructwhich n-arygroupgeneratedapair.Proposition1
formalizes this input-specific loss.
LinearRAG avoids LLM-based relation extraction and
links canonical entity names for semantic bridging and
PPR (Zhuang et al. 2025). Its published index represents
matching names as one graph entity, so it does not expose
the chunk-specific states used by STIBP. In STITCH-RAG,
ψretains the local descriptions and ϕlinks their normalized
names. Proposition 2 tests this difference with controlled
full-merge, no-merge, and semi-merge constructions.
HyperGraphRAGencodes n-aryrelationalfacts,Cog-RAG
uses thematic hyperedges, and CogniRAG represents causal
structures (Luo et al. 2025a; Hu et al. 2026a; He 2026).
STITCH-RAGadoptsthematichyperedgesbutdoesnotcol-
lapsealloccurrencesofanentityname.OKH-RAGalsomod-
els cross-occurrence structure through learned precedence
andsequenceinference(Wuetal.2026);STITCH-RAGin-
steadusesatraining-freeindex-proximityrulegatedbyquery
relevance and frequency underϕ.
Chunk granularity changes the number and scope of these
localstates.Proposition-levelretrievalfavorsnarrowevidence
units(Chenetal.2024),whileRAPTORretrievesrecursively
summarizedclusters(Sarthietal.2024).AutoRAGreports
thatthepreferredgranularitydependsonquestioncomplex-
ity(Kimetal.2024).STITCH-RAGfixesthechunkingpolicy
andbridges theresultingboundaries;itdoesnotdetermine
the optimal chunk size.
E.2 Initializing and Controlling Graph
Propagation
SubGraphRAG retrieves a query-matched local subgraph (Li,
Miao, and Li 2025), and GFM-RAG learns graph retrieval
from training data (Luo et al. 2025b). HippoRAG and Hip-
poRAG2usePPRforbroaderdiffusion(Gutiérrezetal.2024;
Gutiérrez et al. 2025). Their entity-match prior is binary:
every matched entity is seeded and every unmatched entity
receives zero. STIBP instead derives a continuous prior from
the query similarity of local entity states, their topic hyper-
edges, and their name-equivalent neighbors. Proposition 3

statestheresultingevidence-massidentityanditssufficient
alignment condition.
Hyper-RAG enumerates higher-order paths with beam
search (Feng et al. 2026); Cog-RAG filters themes before
entity retrieval (Hu et al. 2026a); and IGMiRAG adjusts
mining depth from estimated question complexity (Hou et al.
2026).ReMindRAGstoresprevioustraversalexperiencein
graph-edgeembeddings(Huetal.2026b).STITCH-RAGuses
no path enumeration or query-time LLM call: STIBP scores
the activated hypergraph structures, and PPR is restricted
to the induced chunk graph. Table 4 reports the combined
retrieval latency, not the isolated cost of localization.
Corrective RAG, Adaptive-RAG, Self-RAG, and Struc-
tRAG decide whether retrieval or a particular structure is
needed(Yanetal.2024;Jeongetal.2024;Asaietal.2024;
Li et al. 2024a). That routing decision is separate from the
retrievaloperatorevaluatedhere.Long-contextstudieslike-
wiseshowthatevidenceplacementandconcentrationaffect
multi-hopreasoningevenwhenthesourcefitswithinthecon-
textwindow(Lietal.2024b;Wuetal.2024).Theseresults
motivate selective retrieval but do not identify a preferred
graph representation.
LinearRAG remains the closest end-to-end structural com-
parison. STITCH-RAG changes its two relevant design
choices:localentitystatesreplacethesinglename-matched
entityrepresentation,andcontinuousSTIBPscoresreplace
binary PPR seeds. STITCH-RAG exceeds LinearRAG in
Contain-Acc by 15.5 points on HotpotQA and 6.7 on 2Wiki,
and in Recall@8 by 7.5 and 5.7 points. These aggregate
margins do not separate the effects of representation, ini-
tialization, and localization; the controlled replacements in
Table 3 provide the component-level evidence available in
this study.
The deployment regime also matters. RAG is preferable
to knowledge internalization when the corpus is large or
updatedindependentlyofmodeltraining,whereasfine-tuning
can remain competitive on narrow static domains (Ovadia
et al. 2024; Balaguer et al. 2024). Structured retrieval is
most useful for questions that require cross-passage evidence
rather than direct lookup (Xiang et al. 2025). STITCH-RAG
furtherassumesthatrepeatedqueriesamortizeitsone-time
hypergraph construction cost.