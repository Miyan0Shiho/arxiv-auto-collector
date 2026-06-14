# An Ontology-Guided Multi-Anchor Graph Retrieval Framework for Traffic Legal Liability Determination

**Authors**: Xu Li, Shuqi Tian, Xun Han, Kuncheng Zhao, Xinyi Li

**Published**: 2026-06-10 10:40:20

**PDF URL**: [https://arxiv.org/pdf/2606.11910v1](https://arxiv.org/pdf/2606.11910v1)

## Abstract
Traffic law liability determination is critical for assigning legal penalties, requiring the simultaneous identification of interdependent statutory provisions across multiple legal dimensions. However, existing retrieval-augmented generation methods suffer from a multi-dimensional retrieval bottleneck: single axis architectures compress complex legal queries into a single pathway, causing interdependent statutory dimensions to be overlooked. To address this, we propose OMAGR, an ontology-guided framework that decomposes queries into ontology-aligned anchors and executes parallel graph retrieval across each dimension, ensuring independent retrieval across dimensions before fusion. To evaluate the proposed method, we created the TrafficLaw-QA dataset, an expert-validated benchmark dataset containing 200 questions and 527 legal provisions. Results show that TrafficOmni-RAG outperforms baselines on Context Precision and Faithfulness metrics. The findings demonstrate that parallel multi-anchor retrieval effectively resolves the multi-dimensional retrieval bottleneck, offering a promising direction for traffic law liability determination research.

## Full Text


<!-- PDF content starts -->

An Ontology-Guided Multi-Anchor Graph
Retrieval Framework for Traffic Legal Liability
Determination
Xu Li1(B)Shuqi Tian1Xun Han2(B)Kuncheng Zhao1Xinyi Li1
1Southwest Petroleum University, Chengdu, Sichuan, China
xul@swpu.edu.cn,tianshuqi2026@163.com,zkcmster@163.com,
xinyi0733@outlook.com
2Sichuan Police College, Luzhou, Sichuan, China
hldwxhx@163.com
Abstract.Traffic law liability determination is critical for assigning le-
gal penalties, requiring the simultaneous identification of interdependent
statutory provisions across multiple legal dimensions. However, existing
retrieval-augmented generation methods suffer from a multi-dimensional
retrieval bottleneck: single axis architectures compress complex legal
queries into a single pathway, causing interdependent statutory dimen-
sionstobeoverlooked.Toaddressthis,weproposeOMAGR,anontology-
guided framework that decomposes queries into ontology-aligned an-
chors and executes parallel graph retrieval across each dimension, en-
suring independent retrieval across dimensions before fusion. To eval-
uate the proposed method, we created the TrafficLaw-QA dataset, an
expert-validated benchmark dataset containing 200 questions and 527
legal provisions. Results show that TrafficOmni-RAG outperforms base-
linesonContext PrecisionandFaithfulnessmetrics.Thefindingsdemon-
strate that parallel multi-anchor retrieval effectively resolves the multi-
dimensional retrieval bottleneck, offering a promising direction for traffic
law liability determination research.
Keywords:RetrievalAugmentedGeneration·KnowledgeGraph·Multi-
Anchor Retrieval·Traffic Law Liability·LLM
1 Introduction
Traffic law liability determination is the legal process of assigning responsibility
and compensation obligations following a road traffic accident [21]. A legally
complete determination requires identifying applicable statutes across several
interdependent dimensions: the violation committed, the conditions of the acci-
dent scene, the roles and statuses of the involved parties, and the rules governing
how liability is allocated among them. Existing deep learning based approaches
predict liability labels from accident features without retrieving the underly-
ing statutory provisions[18], and ontology-based retrieval systems that search
statutes through graph-structured legal ontologies but traverse from a singlearXiv:2606.11910v1  [cs.CL]  10 Jun 2026

2 Xu Li et al.
starting entity[6]. However, these approaches ignore that a single traffic accident
could simultaneously activate provisions governing multiple statutory categories,
so an answer confined to one category is legally incomplete.
Retrieval-Augmented Generation (RAG) addresses this structural difficulty
by grounding generation in externally retrieved knowledge, enabling dynamic
provision retrieval across statutory sources[14]. Existing RAG methods, how-
ever, remain constrained by single-axis retrieval architectures. Standard dense
retrieval encodes queries and documents into a single vector space, scoring can-
didates by semantic similarity[13]. Knowledge graph-augmented RAG systems
traverse entity-relation structures from an extracted starting entity[16]. Dense
retrieval compresses a multi-faceted information need into one query vector,
letting the topically dominant dimension crowd out the remaining dimensions.
Knowledge graph retrieval begins from a single entity and follows connected
edges, limiting traversal to the neighborhood of that entity. For traffic law liabil-
ity, a single-axis retrieval architecture cannot ensure that every legally relevant
dimension contributes evidence. We characterize this as the multi-dimensional
retrieval bottleneck.
To address this bottleneck, we propose Ontology-Guided Multi-Anchor Re-
trieval (OMAGR). OMAGR decomposes an accident description into four an-
chorsalignedwithanontologydrawnfromasix-dimensionaltrafficlawontology:
accident scene, violation type, responsible party, and liability category. It per-
forms parallel graph retrieval along each anchor simultaneously. Two-hop rela-
tional expansion, dense retrieval, and reciprocal rank fusion serve as supporting
mechanisms to supplement recall and integrate evidence. Each legally relevant
dimension thus receives an independent retrieval pathway before fusion. We in-
stantiate OMAGR as the retrieval core of TrafficOmni-RAG, an end-to-end legal
QA system built on a Neo4j knowledge graph encoding 527 provisions from six
Chinese traffic law sources. We evaluate the system on TrafficLaw-QA, a 200-
item expert-verified benchmark annotated under a double-blind protocol. Our
main contributions include:
1. we propose OMAGR to resolve the multi-dimensional retrieval challenge
through parallel multi-anchor graph retrieval.
2. we introduce TrafficLaw-QA for multi-dimensional legal retrieval evaluation.
3. we demonstrate through experiments and ablation that TrafficOmni-RAG
substantially improvesContext PrecisionandFaithfulnessover baselines.
2 Related Work
Standard RAG encodes queries and documents into individual dense vectors
scored by their dot product[13,14], compressing multi-faceted information needs
into one representation. Query rewriting[17] and iterative retrieval[1] improve re-
call diversity, but operate within a single semantic space without restructuring
retrieval around the dimensional organization of the domain. In legal QA, this
compression is particularly damaging because a single accident simultaneously

TrafficOmni-RAG 3
activates provisions governing different statutory categories (violation, scene,
party, liability), yet a single query vector cannot represent all of them simulta-
neously. Graph RAG[7] introduces entity-centric summarization over community
structures yet still follows a single retrieval pathway rather than simultaneous
multi-dimensional traversal.
Knowledge graphs have been incorporated into RAG pipelines to inject en-
tity and relation structure into retrieval. In the legal domain, general legal
LLMs such as ChatLaw[5] and DISC-LawLLM[22] incorporate knowledge bases
for statutory recommendation rather than multi-dimensional liability reasoning;
Lawyer LLaMA[11] similarly treats legal knowledge as a flat article collection.
KG-based retrieval systems, exemplified by LlamaIndex KG[15], traverse from
a single starting entity: if the initial extraction misses a legally relevant dimen-
sion, the corresponding provisions remain unreachable. TrafficRAG[21] applies
hybrid BM25-dense retrieval to Chinese traffic law consultation but does not dif-
ferentiate among legal dimensions in its ranking function. None of these systems
restructures retrieval around the simultaneous activation of multiple interdepen-
dent provisions that characterizes traffic law liability determination.
Existing legal benchmarks evaluate judgment prediction[20], contract re-
view and statutory interpretation[10], and language model performance on legal
text[3], but none isolates the retrieval challenge that arises when a single legal
question requires recovering provisions from multiple interdependent statutory
dimensions. TrafficLaw-QA addresses this gap with 200 items spanning 11 the-
matic categories, each designed to require multi-statute grounding, annotated
under a double-blind protocol (Cohen’sκ= 0.82). OMAGR restructures re-
trieval around a six-dimensional legal ontology, querying four anchors in parallel
to ensure each legally relevant dimension contributes evidence before fusion.
3 Methodology
TrafficOmni-RAG consists of two stages. In the offline stage (§3.1), we construct
a Neo4j knowledge graph from 527 provisions drawn from six Chinese traffic law
sources, organized under a six-dimensional legal ontology. In the online stage,
the OMAGR framework (§3.2) decomposes each user query into structured legal
anchors, performs parallel graph retrieval along each anchor, expands results
throughtwo-hoprelationaltraversal,fusesallevidenceviareciprocalrankfusion,
and refines the final set; the generator (§3.3) then produces a citation-grounded
answer from the refined evidence. Figure 1 illustrates the full architecture.
3.1 Legal Ontology and Knowledge Graph Construction
A complete traffic law liability determination requires reasoning across multiple
legally interdependent dimensions. The applicable rules depend on the accident
scene; the penalty provisions depend on the violation; the rights and obligations
depend on the parties; and the liability allocation depends on the interaction
among all three. We define a six-dimensional legal ontologyO={D 1, ..., D 6}

4 Xu Li et al.
Fig.1.TrafficOmni-RAG framework.
that captures this interdependency by organizing provisions according to their
legal function. The six dimensions are accident scene (D 1), violation type (D 2),
responsibleparty(D 3),liabilitycategory(D 4),legalconsequence(D 5),anddam-
age type (D 6).
DimensionsD 1–D4serve as the four retrieval anchors in the OMAGR frame-
work (§3.2): they characterize the factual and legal premises from which rele-
vant provisions must be retrieved. The scene dimension (D 1) identifies location-
dependent traffic rules; the violation dimension (D 2) captures the legal classifi-
cation of the infraction; the party dimension (D 3) specifies the roles of involved
actors; and the liability dimension (D 4) governs how legal responsibility is al-
located. DimensionsD 5–D6characterize legal outcomes rather than retrieval
inputs. Consequence (D 5) specifies penalties and sanctions; damage (D 6) covers
injury categories and property loss types. These two dimensions are not directly
queried; they are recovered through the relational expansion step (Algorithm 1,
line 7), where cross-references connect violation anchor results to correspond-
ing penalties. Figure 2 illustrates the ontology and its mapping to the OMAGR
retrieval framework.
WemapeachprovisiontoitsapplicableontologydimensionsviaLLMextrac-
tion. A single provision may span multiple dimensions: for instance, Article 91
of the Road Traffic Safety Law (drunk driving penalties) maps toD 2(violation),
D4(liability), andD 5(consequence). The resulting knowledge graphG= (V, E)
stores provisions as nodes with typed edges encoding (i) ontology-dimension
associations, (ii) inter-provision cross-references, and (iii) statutory hierarchies
(law, regulation, judicial interpretation). The graph is implemented in Neo4j,
and all provision embeddings are indexed in a FAISS vector store for semantic
retrieval.

TrafficOmni-RAG 5
Fig.2.Six-dimensional traffic law ontology. The four retrieval anchorsD 1–D4(blue)
feed parallel graph queries; the two outcome dimensionsD 5–D6(green) are recovered
through two-hop relational expansion. All six dimensions converge via RRF fusion into
the refined evidence set for citation-grounded generation.
3.2 The OMAGR Framework
OMAGR is the core retrieval framework. It addresses the multi-dimensional re-
trieval bottleneck by replacing single-axis retrieval with parallel retrieval across
four ontology-aligned anchors. As formalized in Algorithm 1, the pipeline dis-
tinguishes between primary and supporting mechanisms: anchor extraction and
parallel graph queries constitute the primary retrieval pathway, ensuring that
every legally relevant dimension contributes independent evidence prior to fu-
sion. Meanwhile, query expansion, dense retrieval, two-hop relational expansion,
reciprocal rank fusion, and evidence refinement operate as supporting mecha-
nisms to enhance recall, mitigate sparsity, and improve the precision of fused
results.
Multi-anchor retrieval differs fundamentally from query expansion, and the
two mechanisms operate at different levels. Query expansion generates diverse
reformulations within a single semantic space to improve lexical and topical re-
call. Multi-anchor retrieval, in contrast, decomposes the query along predefined
ontology dimensions and executes independent graph traversals in distinct sub-
graph regions, each aligned with a specific legal dimension. The two mechanisms
are complementary: query expansion broadens coverage within each anchor’s
retrieval pathway, while multi-anchor retrieval ensures that all legally relevant
dimensions are represented before fusion.

6 Xu Li et al.
Algorithm 1: OMAGR Pipeline
Input:queryq, KGG(built on ontologyO), vector indexV
Output:diversified provision setR,|R| ≤8
1:d←ParseAccident(q)
2:Q←QueryExpansion(q, d)
3:fork∈ {scene,violation,party,liability}do
4:R k←GraphSearch(G, d k)
5:end for
6:R vec←DenseSearch(V, Q)
7:R hop←TwoHopExpand(G,top3(R violation ))
8:R←RRF(S
kRk∪R vec∪R hop, k=60)
9:R←BGERerank(q, R,top=8)
10:R←MMR(R, λ=0.7)
11:returnR
Anchor extraction uses constrained LLM parsing with a fixed output schema.
If a query provides insufficient information for a particular anchor, that branch
is skipped without blocking the pipeline. In parallel, query expansion generates
from 3 to 5 paraphrased formulations for dense retrieval diversity[17]. Each an-
chor then issues a typed Cypher query traversing the ontology edge for that
dimension. This design is the central architectural choice: because each anchor
queriesadistinctregionofthegraph,thedominantsemanticsignal(typicallythe
violation) cannot crowd out other dimensions; every legally relevant dimension
receives an independent retrieval pathway.
Dense retrieval over the expanded queries provides semantic fallback for pro-
visions lacking strong graph connectivity[13]. Two-hop expansion traverses inter-
provisioncross-referenceedgesfromthetop3violationanchorresults,recovering
co-applicable statutes not directly reachable through ontology edges (for exam-
ple, a penalty provision referencing a separate procedural article). All provision
sets are merged via Reciprocal Rank Fusion[4] (k= 60) with uniform weights
across sources. A cross-encoder reranker[19] re-scores and retains the top 8 provi-
sions. Maximal Marginal Relevance[2] removes near-duplicates; and hierarchical
context pruning allocates token budget by reranker score, assigning full text to
core provisions and progressively shorter excerpts to supplementary ones.
3.3 Generation
The refined provision setR(|R| ≤8) forms the retrieval-augmented context for
the generator LLM (GLM-4-Flash). We use citation-grounded generation[9]: the
modelisinstructedtoembedverbatimstatutoryphrasesasevidencemarkersand
isprohibitedfromcitingprovisionsoutsideR.Thisconstrainseveryanswerclaim
to a verifiable statutory source, reducing hallucination risk in legal reasoning.

TrafficOmni-RAG 7
4 TrafficLaw-QA Benchmark
Standard legal QA benchmarks evaluate whether models can apply legal knowl-
edge to specific tasks: criminal judgment prediction[20], contract review[10],
statutory interpretation[3]. None, however, is designed to expose the retrieval
problem at the center of this paper: cases where legally complete answers re-
quire recovering provisions from multiple interdependent statutory dimensions
simultaneously. TrafficLaw-QA is constructed to fill this gap. It contains 200
expert-verified items spanning 11 thematic categories, each designed to require
multi-provision grounding for a legally sufficient answer.
4.1 Corpus and Scope
TrafficLaw-QA draws from six Chinese traffic law sources spanning three tiers
of legal authority: the Road Traffic Safety Law (legislation); four administrative
regulations, namely the Implementation Regulations of the Road Traffic Safety
Law, the Provisions on Handling Road Traffic Violations, the Traffic Accident
Handling Procedure Regulations, and the Regulations on Compulsory Insurance
for Motor Vehicle Traffic Accident Liability (MTPLI); and the Supreme Court
Interpretations on Road Traffic Accident Damage Compensation (judicial inter-
pretation). Together, these sources define the legal framework for traffic accident
liability determination in China.
4.2 Question Design
The 200 questions span 11 thematic categories and three difficulty levels (Table 1
and Table 2). Categories range from high-frequency liability scenarios such as
liability determination and legal penalties to specialized subdomains such as
highway rules and traffic signals, reflecting the diversity of real-world traffic law
consultation.
Difficulty is defined by the number of statutory provisions required for a
legally complete answer. Easy questions require direct application of a single
provision. Medium questions require integrating two related provisions. Hard
questions require multi-provision reasoning across several legal dimensions. The
distribution skews toward Medium and Hard, which together account for 72.5%
benchmark, reflecting the prevalence of multi-provision scenarios in real traffic
law consultation. Each reference answer is constructed from official statutory
text as a concise legal explanation grounded in explicit provisions.
4.3 Expert Verification
All 200 items were independently reviewed by two legally trained annotators
under a double-blind protocol. The resulting inter-annotator agreement was sub-
stantial, with a Cohen’sκof 0.82. During adjudication, 183 of the 200 items re-
ceived minor wording or citation format revisions; no item required substantive

8 Xu Li et al.
Table 1.TrafficLaw-QA distribution across thematic categories and difficulty levels.
Categoryn% Key Legal Dimensions
Liability Determination 59 29.5% Scene, violation, party, liability
Legal Penalties 23 11.5% Violation type, penalty category
Insurance Compensation 18 9.0% MTPLI coverage, claim conditions
Accident Procedures 18 9.0% Reporting, mediation, investigation
Traffic Regulations 17 8.5% Speed, lane, right-of-way rules
Compensation Scope 13 6.5% Damages, injury, property loss
Vehicle Management 13 6.5% Registration, inspection, licensing
Pedestrian / NMV Rules 13 6.5% Pedestrian, e-bike, non-motorized
Comprehensive Cases 10 5.0% Multi-violation, multi-party
Highway Rules 9 4.5% Speed, distance, emergency stops
Traffic Signals 7 3.5% Signal lights, signs, markings
Total 200 100% 11 categories·44 subcategories
Table 2.Difficulty distribution of TrafficLaw-QA items.
Difficultyn% Reasoning Requirement
Easy (single-statute) 55 27.5% Direct statutory application
Medium (two-statute) 82 41.0% Integration of related provisions
Hard (multi-statute/factor) 63 31.5% Multi-dimensional legal reasoning
legal correction. In addition to verifying the reference answers, the annotators
independently identified the set of statutory provisions relevant to each question
under the same double-blind protocol; these provision sets serve as the ground
truth for Context Recall evaluation.
5 Experiments
5.1 Experimental Settings
All experiments use GLM-4-Flash as the generation model. Dense retrieval uses
FAISS[12]; graph storage and traversal use Neo4j. All methods share the same
document corpus, embedding model, chunking strategy, and generation model,
with an identical number of retrieved candidates passed to the generator.
EvaluationusesRAGAS[8]withGLM-4-Flashasthejudgemodelunderiden-
tical prompting. We report four metrics: Faithfulness (factual grounding of gen-
erated answers in the retrieved context), Answer Relevancy (whether the answer
addresses the query), Context Precision (fraction of retrieved context that is
relevant to the query), and Context Recall (fraction of relevant ground-truth
context that is retrieved).

TrafficOmni-RAG 9
5.2 Baselines
We compare TrafficOmni-RAG against three baselines, each chosen to isolate a
specific factor in retrieval design.
NaiveRAGperformsdensevectorretrievalwithdirectLLMgeneration,with-
out structured legal knowledge or graph augmentation. It represents the conven-
tional single-vector retrieval paradigm and establishes a reference point for how
much multi-anchor retrieval improves over unstructured semantic search.
TrafficRAG[21] proposes hybrid BM25-dense retrieval specialized for Chinese
traffic law consultation. Its pipeline combines sparse lexical matching with dense
semantic search, testing whether improved keyword coverage, without model-
ing legal dimensional structure, can close the dimensional coverage gap that
OMAGR targets.
LlamaIndex KG[15] performs entity-relation graph traversal over the same
Neo4j knowledge graph used by TrafficOmni-RAG. Retrieval starts from a single
extracted entity and walks along connected edges. This is the most controlled
comparison: same graph infrastructure, embedding model, and generator, differ-
ing only in a single-entity rather than a multi-anchor retrieval strategy.
5.3 Results
Table 3 reports the main results on TrafficLaw-QA. The most informative com-
parison is between TrafficOmni-RAG and LlamaIndex KG. Both use the same
Neo4j knowledge graph, embedding model, and generator. The principal struc-
tural difference is retrieval strategy: single-entity traversal rather than multi-
anchor parallel retrieval; supporting mechanisms contribute marginally, as the
ablation study confirms. TrafficOmni-RAG raises Context Precision from 0.722
to 0.915 and Faithfulness from 0.751 to 0.787. The Context Precision gap is the
largest between any two methods in Table 3: single-anchor traversal retrieves
provisions clustered around one entity type while missing co-applicable statutes
anchored in other ontological dimensions.
Table 3.Performance comparison on TrafficLaw-QA.
Method Faithfulness↑Answer Relevancy↑Context Precision↑Context Recall
Naive RAG 0.625 0.736 0.834 0.891
TrafficRAG 0.716 0.845 0.8840.896
LlamaIndex KG 0.751 0.905 0.722 0.796
TrafficOmni-RAG 0.787 0.912 0.9150.885
TrafficRAG achieves the highest Context Recall at 0.896, ahead of 0.885 for
TrafficOmni-RAG. Its BM25 component catches keyword-matching provisions
that dense retrieval alone may miss, producing broader but less precise evi-
dence sets. TrafficOmni-RAG trades this recall margin for precision: retrieval

10 Xu Li et al.
constrained by ontology edges favors legally relevant provisions over topically
similar but legally inapplicable ones. In legal QA, where irrelevant context in-
creases hallucination risk, this precision preference is functionally justified.Naive
RAG scores the lowest Faithfulness at 0.625, confirming that unstructured se-
mantic retrieval alone is insufficient for multi-provision legal questions.
5.4 Ablation Study
Table 4 reports the ablation results. Each variant removes or modifies one com-
ponent of the full pipeline to isolate its contribution.
Table 4.Ablation analysis on TrafficLaw-QA.
Variant Faith. AR CP CR
A0: Naive RAG 0.625 0.736 0.834 0.891
A1: Vector Only (no KG)†0.802 0.899 0.890 0.875
A2: w/o Multi-Anchor∗0.775 0.904 0.862 0.854
A3: w/o Two-Hop‡0.785 0.910 0.913 0.870
A4: w/o Query Expansion 0.792 0.910 0.913 0.881
A5: w/o MMR 0.781 0.916 0.912 0.884
A6: TrafficOmni-RAG 0.787 0.912 0.915 0.885
†Dense retrieval only; no knowledge graph or graph queries.
∗Single-anchor graph retrieval (violation anchor only).
‡No inter-provision edge traversal.
Multi-anchor retrieval. Replacing multi-anchor retrieval with a single graph
anchor reduces Context Precision from 0.915 to 0.862, the largest component-
level drop in Table 4. Context Recall declines from 0.885 to 0.854, and Faith-
fulness falls from 0.787 to 0.775. All four metrics decline together. This pattern
shows that single-anchor retrieval does not merely retrieve fewer provisions; it
retrieves less relevant ones. Restricting graph traversal to one ontological di-
mension concentrates the candidate set around that dimension while starving
the others. This result directly supports the central claim of the paper: traffic
law liability questions require simultaneous retrieval across multiple ontological
dimensions.
Vector-only retrieval. Variant A1 removes the knowledge graph and relies
solely on dense retrieval. Its Faithfulness of 0.802 exceeds that of the full system
by 0.015. Its Context Precision of 0.890 and Answer Relevancy of 0.899 trail
the corresponding A6 values by 0.025 and 0.013. This pattern reflects a known
property of dense retrieval. Semantic similarity concentrates evidence around
the dominant topic of the query, typically the violation. The resulting candidate
set is precise but dimensionally narrow. In multi-party and multi-violation sce-
narios, this concentration systematically omits co-applicable statutes from other
dimensions. A1 outperforms the single-anchor graph variant, A2, on Context

TrafficOmni-RAG 11
Precision by 0.028. Only the full multi-anchor system combines the precision of
semantic retrieval with the dimensional coverage of parallel graph traversal.
Relational expansion. Variant A3 removes two-hop relational expansion. Its
metrics are nearly identical to those of A6. Faithfulness, Answer Relevancy, and
Context Precision each differ by no more than 0.002. Context Recall drops by
0.015. The negligible gap on the first three metrics indicates that two-hop traver-
sal adds little beyond what the four anchor pathways and dense retrieval already
supply. The modest recall gain aligns with the design purpose of relational ex-
pansion.Ittraversesinter-provisioncross-referenceedgestosurfaceco-applicable
statutesthatareabsentfromdirectontologyqueries.Theseadditionalprovisions
widen the candidate pool without degrading precision, but their impact on final
answer quality remains marginal.
Supporting mechanisms. Query expansion, corresponding to A4, shifts each
metric by at most 0.005. Two-hop relational expansion, A3, shifts Context Pre-
cision and Faithfulness by no more than 0.002, with a Context Recall decline of
0.015. MMR diversification, A5, shifts each metric by at most 0.006. Across all
supporting mechanisms, Context Precision varies by no more than 0.003 from
the full system. The principal performance gains originate from the multi-anchor
retrieval structure.
6 Analysis
6.1 Retrieval Source Diversity and Dimensional Coverage
OMAGR retrieves from four distinct sources (scene, violation, party, and lia-
bility) while each baseline draws from a single source. Figure 3 reports per-
dimension retrieval coverage on two representative cases; a dimension value is
covered if at least one retrieved provision addresses it.
On these two representative cases, TrafficOmni-RAG averages 83% coverage,
with the advantage concentrated in Case 2 where it covers 93% of expected
dimension values compared with 9 of 15 for LlamaIndex KG. Case 1 reveals
a limitation: TrafficRAG achieves higher coverage at 82% compared with 73%
for TrafficOmni-RAG through BM25 matching, pointing to anchor extraction
omissions as irrecoverable retrieval blind spots.
A blind expert evaluation on 50 randomly sampled items yields an anchor
extraction F1 of 64.3%, with precision at 57.3% and recall at 73.3%. The recall-
dominant profile aligns with the architecture: false negatives create irrecoverable
retrieval blind spots, while false positives are suppressed by downstream RRF
fusion.
6.2 Performance Across Legal Reasoning Types
We partition TrafficLaw-QA into three types corresponding to distinct retrieval
failure modes. Category A comprises 68 Single-Anchor queries testing single-
dimensioncoverage.CategoryBcontains102Multi-Anchorqueriestestingcross-
dimensional coordination, the core challenge that OMAGR addresses. Category

12 Xu Li et al.
Fig.3.Per-dimension retrieval coverage on two representative cases. Case 1 (single-
violation, multi-consequence): a driver runs a red light while intoxicated, strikes a
pedestrian at a crosswalk, and flees; 11 dimension values are expected. Case 2 (multi-
violation, multi-party): an unlicensed driver runs a red light, causing a chain collision
involving three vehicles with expired insurance; 15 dimension values are expected.
Retrieval source count is shown below each method. TrafficOmni-RAG is the only
method drawing from four retrieval sources; the red marker highlights its 93% coverage
on Case 2.
C includes 30 Relational/Multi-Hop queries testing multi-hop traversal depth
over inter-provision dependency chains.
Table 5 reports Context Precision and Context Recall per type.
Table 5.Context Precision (CP) and Context Recall (CR) by method across different
legal reasoning types.
Method A (Single-Anchor) B (Multi-Anchor) C (Rel./Multi-Hop)
CP / CR CP / CR CP / CR
TrafficOmni-RAG0.941/ 0.9190.900/ 0.8630.903/ 0.882
Naive RAG 0.890 / 0.914 0.795 /0.8760.841 / 0.888
TrafficRAG 0.917 /0.9240.860 / 0.872 0.888 /0.904
LlamaIndex KG 0.713 / 0.796 0.736 / 0.813 0.700 / 0.737
TrafficOmni-RAG leads in Context Precision on all three types. The advan-
tage over TrafficRAG follows a gradient: largest on Category B at 0.040, moder-
ate on Category A at 0.024, and smallest on Category C at 0.015. This ranking
validates the multi-anchor design logic. Category B, the intended scenario for
parallel anchor traversal, simultaneously activates all four retrieval dimensions,
yielding the peak gain. On Category A, adequate dimensional coverage from a
single dense retrieval pass renders additional anchors redundant. On Category C,

TrafficOmni-RAG 13
cross-reference chains extending beyond the fixed two-hop radius shift the bot-
tleneck from dimensional breadth to traversal depth, capping the multi-anchor
gain.
Context Recall follows a different pattern. TrafficOmni-RAG trails the best
baseline on all three categories, with the gap widening from 0.005 on A to 0.013
on B and 0.022 on C. TrafficRAG achieves the highest Context Recall on Cat-
egories A and C through BM25 lexical matching; on Category B, Naive RAG
leads at 0.876.
Table 6 reports Faithfulness and Answer Relevancy.
Table 6.Faithfulness and Answer Relevancy by legal reasoning type.
Method A (Single-Anchor) B (Multi-Anchor) C (Rel./Multi-Hop)
Faith. / AR Faith. / AR Faith. / AR
TrafficOmni-RAG0.908/0.9330.694 / 0.903 0.828 / 0.896
Naive RAG 0.701 / 0.752 0.574 / 0.719 0.623 / 0.757
TrafficRAG 0.821 / 0.872 0.616 / 0.811 0.697 / 0.857
LlamaIndex KG 0.767 / 0.9010.730/0.906 0.835/0.913
TrafficOmni-RAG leads on Category A, with Faithfulness at 0.908 and An-
swer Relevancy at 0.933. On Categories B and C, LlamaIndex KG records higher
Faithfulness: 0.730 against 0.694 on B, and 0.835 against 0.828 on C. This in-
version reflects a precision-coverage tradeoff: LlamaIndex KG retrieves from a
narrow evidence base, its Context Precision trailing that of TrafficOmni-RAG
by 0.164 on B and 0.203 on C, and RAGAS Faithfulness measures support by
the retrieved context rather than legal completeness of that context. A system
retrieving few but internally consistent provisions can score high on Faithfulness
while omitting relevant statutes, a failure mode that Context Precision captures
but Faithfulness does not. TrafficOmni-RAG trades a modest Faithfulness cost
forasubstantialContextPrecisiongain.Thecostisasymmetric:theFaithfulness
gap narrows to 0.007 on Category C and reaches 0.036 on Category B, where
the steeper tradeoff is justified by the Context Precision gain of 0.040.
7 Conclusion and Limitations
Traffic law liability determination requires simultaneous retrieval across multiple
interdependent legal dimensions, a bottleneck that single-axis architectures can-
not address. We propose Ontology-Guided Multi-Anchor Retrieval (OMAGR),
instantiated as the core of TrafficOmni-RAG, which addresses this bottleneck
through parallel multi-anchor retrieval over the traffic law ontology; experiments
on TrafficLaw-QA confirm that multi-anchor parallelism drives the principal
gain. Limitations include anchor extraction quality, a fixed two-hop expansion
radius, and the exclusive focus on Chinese traffic law. Future work includes

14 Xu Li et al.
hardening anchor extraction, extending traversal depth, and evaluating cross-
jurisdictional generalizability.
Acknowledgments.This work is supported by the Opening Project of Intelligent
Policing Key Laboratory of Sichuan Province (Grant No. ZNJW2025KFZD004).
References
1. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H.: Self-rag: Learning to retrieve,
generate, and critique through self-reflection. In: International Conference on
Learning Representations (2024), https://proceedings.iclr.cc/paper_files/paper/
2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf
2. Carbonell, J., Goldstein, J.: The use of mmr, diversity-based reranking for reorder-
ing documents and producing summaries. In: ACM SIGIR Conference on Research
andDevelopmentinInformationRetrieval(1998).https://doi.org/10.1145/290941.
291025
3. Chalkidis, et al.: LEGAL-BERT: The muppets straight out of law school. In:
Findings of the Association for Computational Linguistics: EMNLP. Associ-
ation for Computational Linguistics (2020). https://doi.org/10.18653/v1/2020.
findings-emnlp.261, https://aclanthology.org/2020.findings-emnlp.261/
4. Cormack,G.V.,etal.:Reciprocalrankfusionoutperformscondorcetandindividual
rank learning methods. In: ACM SIGIR Conference on Research and Development
in Information Retrieval. Association for Computing Machinery (2009). https://
doi.org/10.1145/1571941.1572114
5. Cui, J., Ning, M., Li, Z., Chen, B., Yan, Y., Li, H., Ling, B., Tian, Y., Yuan,
L.: Chatlaw: A multi-agent collaborative legal assistant with knowledge graph
enhanced mixture-of-experts large language model (2024), https://arxiv.org/abs/
2306.16092
6. Dang, et al.: Information retrieval from legal documents with ontology and graph
embeddings approach. In: Advances and Trends in Artificial Intelligence. The-
ory and Applications: 36th International Conference on Industrial, Engineering
and Other Applications of Applied Intelligent Systems, IEA/AIE 2023, Shanghai,
China, July 19–22, 2023, Proceedings, Part I. Springer Nature Switzerland, Cham
(2023)
7. Edge, D., et al.: From local to global: A graph rag approach to query-focused
summarization (2025), https://arxiv.org/abs/2404.16130
8. Es, et al.: RAGAs: Automated evaluation of retrieval augmented generation. In:
Proceedings of the 18th Conference of the European Chapter of the Association
for Computational Linguistics: System Demonstrations. Association for Computa-
tional Linguistics (2024). https://doi.org/10.18653/v1/2024.eacl-demo.16, https:
//aclanthology.org/2024.eacl-demo.16/
9. Gao, et al.: Enabling large language models to generate text with citations. In:
Proceedings of the 2023 Conference on Empirical Methods in Natural Language
Processing. Association for Computational Linguistics (2023). https://doi.org/10.
18653/v1/2023.emnlp-main.398, https://aclanthology.org/2023.emnlp-main.398/
10. Guha, N., Nyarko, J., et al.: Legalbench: A collaboratively built bench-
mark for measuring legal reasoning in large language models. In: Ad-
vances in Neural Information Processing Systems. Curran Associates,
Inc. (2023), https://proceedings.neurips.cc/paper_files/paper/2023/file/
89e44582fd28ddfea1ea4dcb0ebbf4b0-Paper-Datasets_and_Benchmarks.pdf

TrafficOmni-RAG 15
11. Huang, Q., Tao, M., Zhang, C., An, Z., Jiang, C., Chen, Z., Wu, Z., Feng, Y.:
Lawyer llama technical report (2023), https://arxiv.org/abs/2305.15062
12. Johnson, J., Douze, M., Jégou, H.: Billion-scale similarity search with gpus. IEEE
Transactions on Big Data (2021). https://doi.org/10.1109/TBDATA.2019.2921572
13. Karpukhin, et al.: Dense passage retrieval for open-domain question answer-
ing. In: Proceedings of the 2020 Conference on Empirical Methods in Natural
Language Processing. Association for Computational Linguistics, Online (Nov
2020). https://doi.org/10.18653/v1/2020.emnlp-main.550, https://aclanthology.
org/2020.emnlp-main.550/
14. Lewis, P., Perez, E., et al.: Retrieval-augmented generation for knowledge-intensive
nlp tasks. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H.
(eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 9459–9474.
CurranAssociates,Inc.(2020),https://proceedings.neurips.cc/paper_files/paper/
2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf
15. Ongris, et al.: Benchmarking kg-based rag systems: A case study of legal docu-
ments. CEUR Workshop Proceedings (2025)
16. Pan, et al.: Unifying large language models and knowledge graphs: A roadmap.
IEEE Transactions on Knowledge and Data Engineering (2024). https://doi.org/
10.1109/TKDE.2024.3352100
17. Wang,etal.:Query2doc:Queryexpansionwithlargelanguagemodels.In:Proceed-
ingsofthe2023ConferenceonEmpiricalMethodsinNaturalLanguageProcessing.
Association for Computational Linguistics, Singapore (2023). https://doi.org/10.
18653/v1/2023.emnlp-main.585, https://aclanthology.org/2023.emnlp-main.585/
18. Wang, et al.: Amformer-based framework for accident responsibility attribution:
Interpretable analysis with traffic accident features. PLOS ONE (2025). https:
//doi.org/10.1371/journal.pone.0329107
19. Xiao, et al.: C-pack: Packed resources for general chinese embeddings. In: ACM
SIGIR Conference on Research and Development in Information Retrieval. Associ-
ation for Computing Machinery, New York, NY, USA (2024). https://doi.org/10.
1145/3626772.3657878, https://doi.org/10.1145/3626772.3657878
20. Xiao,C.,etal.:Cail2018:Alarge-scalelegaldatasetforjudgmentprediction(2018),
https://arxiv.org/abs/1807.02478
21. Xu, L., Zedong, F., Xinyi, L., Xun, H.: Trafficrag: A multimodal rag framework for
trafffc accident liability determination. In: Artificial Neural Networks and Machine
Learning – ICANN 2026. Springer Nature Switzerland (2026)
22. Yue, S., Chen, W., Wang, S., Li, B., Shen, C., Liu, S., Zhou, Y., Xiao, Y., Yun, S.,
Huang, X., Wei, Z.: Disc-lawllm: Fine-tuning large language models for intelligent
legal services (2023), https://arxiv.org/abs/2309.11325