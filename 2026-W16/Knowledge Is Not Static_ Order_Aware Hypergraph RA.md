# Knowledge Is Not Static: Order-Aware Hypergraph RAG for Language Models

**Authors**: Keshu Wu, Chenchen Kuai, Zihao Li, Jiwan Jiang, Shiyu Shen, Shian Wang, Chan-Wei Hu, Zhengzhong Tu, Yang Zhou

**Published**: 2026-04-14 01:31:35

**PDF URL**: [https://arxiv.org/pdf/2604.12185v1](https://arxiv.org/pdf/2604.12185v1)

## Abstract
Retrieval-augmented generation (RAG) enhances large language models by grounding outputs in retrieved knowledge. However, existing RAG methods including graph- and hypergraph-based approaches treat retrieved evidence as an unordered set, implicitly assuming permutation invariance. This assumption is misaligned with many real-world reasoning tasks, where outcomes depend not only on which interactions occur, but also on the order in which they unfold. We propose Order-Aware Knowledge Hypergraph RAG (OKH-RAG), which treats order as a first-class structural property. OKH-RAG represents knowledge as higher-order interactions within a hypergraph augmented with precedence structure, and reformulates retrieval as sequence inference over hyperedges. Instead of selecting independent facts, it recovers coherent interaction trajectories that reflect underlying reasoning processes. A learned transition model infers precedence directly from data without requiring explicit temporal supervision. We evaluate OKH-RAG on order-sensitive question answering and explanation tasks, including tropical cyclone and port operation scenarios. OKH-RAG consistently outperforms permutation-invariant baselines, and ablations show that these gains arise specifically from modeling interaction order. These results highlight a key limitation of set-based retrieval: effective reasoning requires not only retrieving relevant evidence, but organizing it into structured sequences.

## Full Text


<!-- PDF content starts -->

KNOWLEDGEISNOTSTATIC: ORDER-AWAREHYPERGRAPH
RAGFORLANGUAGEMODELS
Keshu Wu1, Chenchen Kuai1, Zihao Li1, Jiwan Jiang1, Shiyu Shen2,
Shian Wang3, Chan-Wei Hu1, Zhengzhong Tu1, Yang Zhou1,∗
1Texas A&M University
2University of Illinois Urbana-Champaign
3The University of Kansas
ABSTRACT
Retrieval-augmented generation (RAG) enhances large language models by grounding outputs in
retrieved knowledge. However, existing RAG methods including graph- and hypergraph-based
approaches treat retrieved evidence as an unordered set, implicitly assuming permutation invariance.
This assumption is misaligned with many real-world reasoning tasks, where outcomes depend not only
on which interactions occur, but also on the order in which they unfold. We proposeOrder-Aware
Knowledge Hypergraph RAG (OKH-RAG), which treats order as a first-class structural property.
OKH-RAG represents knowledge as higher-order interactions within a hypergraph augmented with
precedence structure, and reformulates retrieval as sequence inference over hyperedges. Instead
of selecting independent facts, it recovers coherent interaction trajectories that reflect underlying
reasoning processes. A learned transition model infers precedence directly from data without requiring
explicit temporal supervision. We evaluate OKH-RAG on order-sensitive question answering and
explanation tasks, including tropical cyclone and port operation scenarios. OKH-RAG consistently
outperforms permutation-invariant baselines, and ablations show that these gains arise specifically
from modeling interaction order. These results highlight a key limitation of set-based retrieval:
effective reasoning requires not only retrieving relevant evidence, but organizing it into structured
sequences.
KeywordsKnowledge Graph·Retrieval-Augmented Generation·Hypergraph Representation·Graph-based RAG
1 Introduction
Large language models (LLMs) have demonstrated impressive capabilities in reasoning and generation, yet their
performance on knowledge-intensive tasks remains constrained by the lack of explicit mechanisms for structured
inference [ 1,2,3,4]. Retrieval-augmented generation (RAG) mitigates this limitation by conditioning generation on
externally retrieved knowledge, thereby separating knowledge access from language modeling [ 5,6,7,8]. While early
RAG approaches rely on unstructured text retrieval, recent work has shown that incorporatingstructured knowledge
representationscan significantly improve reasoning fidelity and factual consistency [9, 10].
Graph-based RAG methods represent an important step in this direction by organizing knowledge as a graph of entities
and relations [ 11,12,13]. However, conventional graphs encode onlybinaryrelationships, restricting their ability to
model interactions that inherently involve more than two entities. Many phenomena in real-world systems exhibit such
higher-order dependencies, where outcomes emerge from the joint configuration of multiple factors rather than pairwise
interactions alone. Hypergraphs naturally generalize graphs by allowing edges to connect arbitrary-sized sets of nodes,
and therefore provide a principled representation for capturinghigher-order graph dynamics[14, 15, 16, 17, 18].
Recent work on hypergraph-based RAG demonstrates that representing knowledge with n-ary relations can reduce
information loss and improve retrieval effectiveness compared to binary graph formulations [ 19,20,21,22]. Despite this
∗Corresponding author.arXiv:2604.12185v1  [cs.CL]  14 Apr 2026

Hurricane HarveyPort of Houston
Storm SurgeWind GustCargo DelayedPort ClosedVessel Operations SuspendedPortReopenedWarningSSHS Level 4Hurricane HarveyPort of Houston
Storm SurgeWind GustCargo DelayedPort ClosedVessel Operations SuspendedPortReopenedWarningSSHS Level 4𝒆𝟏𝒆𝟐𝒆𝟑𝒆𝟒Cyclone State
Hazard ForecastPort OperationsImpact & RecoveryHurricane HarveyPort of Houston
Storm SurgeWind GustCargo DelayedPort ClosedVessel Operations SuspendedPortReopenedWarningSSHS Level 4𝒆𝟏𝒆𝟐𝒆𝟑𝒆𝟒Cyclone State
Hazard ForecastPort OperationsImpact & Recovery𝒆𝟏𝒆𝟐𝒆𝟑𝒆𝟒Retrieved as ordered trajectoryKnowledge Hypergraph Knowledge Graph Order-Aware Knowledge Hypergraph 
Pairwise edges onlyHyperedges as hyper-relationsDomain Knowledge“Hurricane Harvey intensifies to SSHS Level 4, triggering wind gustand storm surge hazards at Port of  Houston. A gale warning triggered suspension of vessel operationsand port closure, causing cargo delays until the port reopened during recovery.”Figure 1: Knowledge graphs fragment multi-entity interactions into pairwise edges; knowledge hypergraphs preserve
higher-order relations but retrieve them as unordered sets; our order-aware knowledge hypergraph augments hyperedges
with learned precedence, enabling retrieval as ordered trajectories that reflect how interactions unfold.
advance, existing hypergraph-based RAG methods remain fundamentallystatic: hyperedges encode timeless relational
facts, and retrieval is performed without accounting for how interactions evolve over time. This static abstraction limits
the ability of retrieval mechanisms to support reasoning about processes, causality, and delayed effects.
Though many interaction-aware RAG methods have been proposed, sequential structure is fundamental to knowledge-
intensive reasoning, where system-level outcomes typically emerge from ordered interactions rather than simple
co-occurrences. Whether representing a chain of subsystem failures or the propagation of causal dependencies,
modeling these phenomena as static relations obscures the inherent ordering, persistence, and dynamics of the system,
preventing retrieval from capturing underlying dynamics. For example, in a tropical cyclone scenario, port disruption
may depend jointly on forecast storm intensity, local infrastructure exposure, evolving port conditions, and operational
response. This requires a hypergraph representation, since the relevant dependency is higher-order rather than pairwise.
It also requires order-awareness, since the reasoning depends on how these factors evolve over time, from forecast
updates to warning escalation, port restrictions, and ultimately operational disruption [23, 24].
In this work, we argue thatorder is a fundamental dimension of knowledge structurein retrieval-augmented generation.
We introduceOrder-Aware Knowledge Hypergraph RAG (OKH-RAG), as illustrated in Figure 1, a framework
that models knowledge as higher-order interactions with explicit precedence and reformulates retrieval astrajectory
inference. Instead of retrieving a set of independent facts, OKH-RAG retrievesordered hyperedge trajectoriesthat
capture how interactions evolve and connect to form coherent reasoning chains. This shifts retrieval from selecting
relevant evidence to recovering structured sequences that reflect underlying processes. Our formulation unifies two
complementary advances: hypergraph representations for modeling higher-order dependencies, and order-aware
inference for capturing interaction dynamics. By integrating these into a single framework, OKH-RAG enables LLMs
to reason over processes rather than static snapshots. Our contributions are summarized as follows:
•We introduce anorder-aware knowledge hypergraphrepresentation that captures higher-order interactions together
with their precedence structure.
•We reformulate retrieval astrajectory inferenceover hyperedges, moving from permutation-invariant set retrieval to
order-aware sequence retrieval.
•We integrate order-aware hypergraph retrieval into the RAG framework, enabling generation grounded in structured,
temporally coherent evidence.
By integrating higher-order structure with interaction order, OKH-RAG establishes a retrieval framework where the
sequenceof evidence, not merely its content, is explicitly modeled, optimized, and empirically validated as a key driver
of generation quality.
2 Related Work
Retrieval-Augmented Generation.Retrieval-augmented generation (RAG) enhances parametric language models by
conditioning generation on externally retrieved knowledge, mitigating issues such as hallucination, outdated information,
2

and lack of traceable reasoning [ 25,26]. By introducing a retriever that accesses a non-parametric memory, RAG
enables models to incorporate domain-specific and up-to-date information at inference time. This framework has
been widely adopted in knowledge-intensive applications, particularly in domain-specific settings. Representative
applications include biomedical question answering [ 27], legal document analysis [ 28], and enterprise knowledge
retrieval [ 29]. Early RAG systems primarily rely on retrieving unstructured text passages using dense retrieval methods
such as Dense Passage Retrieval (DPR) [ 30,31]. Subsequent surveys and benchmarks [ 32,33] identify bottlenecks
in both retrieval quality and scalability, particularly with the emergence of large language models (LLMs). Recent
work has explored several directions to address these challenges, including knowledge-grounded dialogue systems [ 34],
adaptive retrieval strategies [ 35,36,37], and multi-hop reasoning frameworks [ 38,39]. These developments collectively
motivate the incorporation of structured representations into the retrieval process.
Graph and Hypergraph-Based RAG.Graph-based RAG methods [ 40,41,42], which organize knowledge as
entities and binary relations, represent one of the most effective extensions beyond unstructured retrieval in RAG. These
approaches enable graph traversal [ 43,44], path-based reasoning [ 45,46], and community-level summarization [ 11],
thereby improving evidence aggregation and interpretability compared to purely similarity-based retrieval. However,
they remain fundamentally limited by binary representations that decompose higher-order dependencies into pairwise
edges, potentially leading to information loss and fragmented reasoning. To address this limitation, hypergraph-based
approaches have been introduced to connect arbitrary-sized sets of nodes, naturally capturing n-ary relational facts [ 47]
and higher-order interactions [ 20,48]. Such representations provide a more expressive framework for modeling
complex relationships and have shown improvements in retrieval efficiency and generation quality over binary graph
structures [ 49]. Nevertheless, existing hypergraph-based RAG methods typically treat hyperedges asstaticfacts, failing
to capture the dynamics, temporal evolution, and ordering of higher-order interactions, which limits their ability to
support more complex reasoning processes.
Temporal Dynamics and Event-Centric Modeling.Temporal and dynamic graph models have been extensively
studied to capture time-varying relations and node attributes [ 50,51,52,53]. These approaches typically focus on
predictive tasks such as link forecasting or anomaly detection using message-passing or memory modules [ 54,55].
While powerful, these models are largely designed for binary relational structures and do not directly capture higher-
order interactions. Furthermore, while recent extensions [ 56] have introduced temporal hypergraphs to model evolving
n-ary relations, they focus on representation learning rather than retrieval-augmented generation. Related work in
event-centric modeling emphasizes temporal ordering and causal structure [ 57,58], yet these systems often rely on task-
specific schemas rather than general-purpose retrieval. OKH-RAG bridges these gaps by unifying hypergraph-based
representation with order-awareness. Unlike graph-based RAG, OKH-RAG models higher-order dependencies directly
via hyperedges. Unlike static hypergraph methods, it treats order as a first-class dimension, moving from set-based
selection totrajectory-based retrieval. By treating retrieval as inference over interaction sequences, OKH-RAG enables
LLMs to reason about processes and propagation—capabilities previously absent in structured RAG frameworks.
3 Methodology
OKH-RAG is built on the premise that the order of knowledge interactions is a structural property essential for faithful
reasoning. While existing RAG systems treat retrieved evidence as an unordered set, we model knowledge as an
evolving process and reformulate retrieval as trajectory inference. The framework comprises three stages (Figure 2): (1)
constructing an order-aware hypergraph with learned precedence (§ 3.1); (2) retrieving ordered hyperedge trajectories
(§ 3.2); and (3) generating responses conditioned on structured evidence chains (§ 3.3).
3.1 Knowledge Hypergraph Construction and Order Learning
The first stage transforms unstructured text into a structured, order-aware representation. Our goal is not only to capture
whatinteractions exist, but alsohowthey unfold. We first construct a knowledge hypergraph to model higher-order
interactions, and then augment it with learned precedence to encode ordering.
Knowledge hypergraph.Many real-world phenomena involve interactions that are inherently higher-order: outcomes
depend on the joint configuration of multiple factors rather than pairwise associations. Standard knowledge graphs,
restricted to binary edges, cannot represent such interactions without fragmentation. We therefore represent domain
knowledge as aknowledge hypergraph H= (V,E) , where Vis a set of entities and each hyperedge e∈ E connects an
arbitrary subset of entities ( |Ve| ≥2 ). Following(author?) [20], we employ n-ary relational extractionwith natural
language descriptions, preserving richer semantics than structured triples. While hypergraphs improve expressiveness
in representing interactions, existing hypergraph-based RAG methods treat Hasstatic and unordered: retrieved
3

Learned Precedence𝑃𝑒𝑗𝑒𝑖𝑒1𝑒4𝑒2𝑒3
𝑒1𝑒4𝑒2𝑒3𝑊1𝑊2𝑊3Knowledge Hypergraph Construction & Order LearningOrder-Aware Hypergraph RetrievalOrder Learning ModuleEntity & Relation ExtractionExtraction & Hyperedge FormationKnowledge Hypergraph
Order-AwareKnowledge HypergraphTransition ModelPrompt ConstructionQuery Encoder
Trajectory Search ModuleRetrieved Trajectory
User Query
Retrieval-Augmented Generation
Question
Scoreℒ𝜃Final Response	𝑒2	𝑒4	𝑒1	𝑒3
	𝑒1	𝑒2	𝑒4	𝑒3	𝑒1	𝑒2	𝑒4	𝑒3Retrieved Trajectory
Sequential Data ContextHypergraph StructureKnowledge Hypergraph Construction & Order LearningOrder-Aware Hypergraph RetrievalRetrieval-Augmented GenerationUnstructured DataOrder-AwareKnowledge HypergraphOrdered TrajectoryFinal Grounded ResponseUser QueryUser Query
Unstructured Text
EntityHyperedge
LLMFigure 2:Overview of OKH-RAG.The framework constructs an order-aware knowledge hypergraph from documents,
retrieves query-specific interaction trajectories via sequence inference, and generates responses conditioned on struc-
tured, temporally coherent evidence.
hyperedges are presented as a permutation-invariant set. As we show below, this abstraction is insufficient for reasoning
tasks in which order affects inference.
Entities.Each entity v∈ V represents a distinct object, concept, or state, defined as v= (n v, τv, dv, cv), where
nvis the entity name, τv∈ T is a type from a domain-specific type system, dvis a natural-language explanation, and
cv∈(0,1] is a confidence score. The type system distinguishespersistent objects(e.g., a port or cyclone) that anchor
the hypergraph,transient states(e.g., a cyclone state at a specific horizon) that capture evolution, andtemporal anchors
(e.g.,horizon:T-48) that index interactions to positions in the ordering.
Hyperedges.Entities interact through hyperedges—the primary knowledge-carrying units. Each hyperedge is a tuple
e= (V e, re, se,ae, ce), where Ve⊆ V (|Ve| ≥2 ) is the participating entity set, re∈ R is a typed relation from a
controlled vocabulary, seis anatural language descriptioncapturing the interaction’s semantic content, aeis a set
of key–value attributes recording quantitative properties, and ce∈(0,1] is a confidence score. The natural language
description seis critical: unlike structured triples, it can express multi-entity dependencies whose meaning arises from
joint configuration rather than pairwise association. The vocabulary is defined in Appendix A.1.
N-ary relational extraction.We construct Hfrom a document corpus Kusing a language model πguided by an
extraction prompt pext:Fd={f 1, . . . , f k} ∼π(F |p ext, d), where each fact fi= (e i, Vei)pairs a hyperedge with
its entity set. For multi-horizon documents, per-horizon extraction followed by merging yields higher coverage than
monolithic processing. Post-extraction normalization canonicalizes identifiers, injects horizon entities, and synthesizes
cross-horizon edges (Appendix A.2). The complete hypergraph aggregates across documents:
H=[
d∈K[
fi∈FdVei,[
d∈K{ei|fi∈ Fd}
.(1)
Order-aware hypergraph.The knowledge hypergraph captureswhatinteractions exist but not their order. To encode
precedence while remaining agnostic to clock time, we introduce a discrete sequence index ℓ∈ {1, . . . , L} capturing
4

relative order and represent knowledge as a sequence of states H(ℓ)= (V,E(ℓ)). This induces a precedence relation
ei≺ej⇔ ∃ℓ 1< ℓ2s.t.e i∈ E(ℓ1), ej∈ E(ℓ2), yielding theorder-aware knowledge hypergraph:
H≺= (V,E,≺).(2)
The index ℓencodes precedence, not duration; ≺is a partial order accommodating concurrent interactions; and
when≺=∅ , the representation reduces to a standard unordered hypergraph. A non-empty ≺breaks permutation
invariance and enables order-aware retrieval. The precedence graph is constructed via domain-informed structural rules
(Appendix A.3).
Learning precedence.Since explicit precedence annotations are rarely available, we learn a parametric transition
model:
Pθ(ej|ei) =exp(h⊤
eiWh ej)P
e′exp(h⊤eiWh e′),(3)
where he∈Rdis a dense hyperedge embedding and Wis a learnable weight matrix (low-rank factorized as U⊤V,
r≪d ). The bilinear form is inherently asymmetric— Pθ(ej|ei)̸=P θ(ei|ej)—encoding directionality without
explicit temporal features.
The model is trained via a contrastive objective with three self-supervised signals:document order(positional adjacency
as a proxy for precedence),entity-overlap consistency(shared entities as evidence for co-participation in reasoning
chains), andretrieval-induced preference(reinforcing transitions along empirically successful trajectories in a self-
training loop):
L(θ) =E (ei,ej)∼P[−logP θ(ej|ei)] +αE (ei,em)∼N[logP θ(em|ei)],(4)
wherePandNare positive and negative ordered pairs (details in Appendix A.4).
Order sensitivity of knowledge.We have described how to construct and learn H≺. Is this machinery necessary?
We establish formally that the answer is yes.
Proposition 1(Order sensitivity).There exist ka, kb, qsuch that P(y|k a, kb, q)̸=P(y|k b, ka, q). Any permutation-
invariant retrieval method is therefore insufficient for modeling P(y|q) when reasoning depends on interaction
order.
The proposition requires no assumptions about timestamps or causality—only that evidence order can alter inferred
outcomes. This complements(author?)[20]: where they establish that hypergraphs are more expressive inwhatthey
represent, we establish that order-aware hypergraphs are more expressive inhowthey can be retrieved (discussion in
Appendix A.5).
3.2 Order-aware Hypergraph Retrieval
Standard retrieval scores each element independently and returns the highest-scoring set, treating evidence selection as
a ranking problem. This discards arrangement. For order-sensitive reasoning, the quality of retrieved evidence depends
not only on its members but on their sequence. We reformulate retrieval assequence inference: given q, recover the
highest-scoring ordered trajectory throughH ≺.
Retrieval objective.We seek an ordered trajectoryγ= (e(1), . . . , e(L))maximizing:
γ∗= arg max
γLX
k=1Rel(e(k), q)
|{z }
relevance+λL−1X
k=1logP θ(e(k+1)|e(k))
| {z }
order coherence
+µ·Prec(γ)|{z}
precedence
consistency+ν·Ovlp(γ)|{z}
entity
continuity+ρ·Cov(γ)|{z}
phase
coverage(5)
Relevanceensures topical pertinence via cosine similarity.Order coherenceensures consecutive interactions form
plausible transitions under Pθ.Precedence consistencyenforces alignment with the structural relation ≺.Entity
continuityrewards entity sharing between consecutive steps, favoring coherent chains over disjointed fact assemblages.
Phase coveragerewards spanning distinct reasoning stages (advisory →hazard →operation →impact →recovery),
preventing narrow evidence concentration. The hyperparameters λ, µ, ν, ρ≥0 are modular: setting all to zero recovers
standard top-kretrieval (formal definitions in Appendix A.6).
5

Candidate scoping and inference.We scope the candidate set Cvia top- Kcosine retrieval followed by group-aware
expansion (entity overlap and domain group membership), yielding |C|.Beam searchserves as the primary inference
algorithm;Viterbi DPprovides exact optimization for smaller candidate sets (Appendix A.7).
Multi-trajectory retrieval.Many queries admit multiple valid reasoning paths. We therefore retrieve a set of diverse
trajectories Γq={γ 1, . . . , γ N}, enabling the model to consider alternative explanations and complementary evidence
chains.
3.3 Retrieval-augmented Generation
Concatenating hyperedge texts into a flat context window would collapse ordered trajectories back into unordered sets.
OKH-RAG instead presents each trajectory as a numbered evidence chain with explicit structural annotations.
Structured evidence and generation.Each hyperedge e(k)in a trajectory carries astep index, ahorizon label, aphase
label, andentity provenance—making the reasoning structure legible to the generator. The language model produces
y∼P(y|q,Γ q), and can thereby recognize preconditions, track escalation across horizons, and trace downstream
consequences—capabilities inaccessible to generators receiving unordered evidence. When multiple trajectories are
available, the primary mode presents all paths in a single prompt for cross-referencing; a fallback mode aggregates
independent per-trajectory answers via confidence-weighted voting or averaging (Appendix A.8).
4 Experiments
We evaluate the central hypothesis of this paper:when answers depend on how evidence unfolds, retrieval should
preserve order rather than treat evidence as an unordered set.Our experiments address four questions: (1) Does
the constructed order-aware knowledge hypergraph capture the structural regimes in which order matters? (2) Does
order-aware retrieval recover coherent, query-adaptive evidence trajectories? (3) Does this improve QA performance
over permutation-invariant baselines? (4) Do the gains arise specifically from modeling order?
4.1 Experimental Setup
We evaluate onCyPortQA[ 24], a domain-specific QA benchmark for tropical cyclone–port impact assessment.
CyPortQA contains 2,917 real-world disruption scenarios from 2015 to 2023, spanning 145 U.S. principal ports and 90
named storms. Each scenario includes multi-horizon descriptions from T-120 to T-12 covering storm evolution, hazard
forecasting, operational response, and impact prediction. The benchmark contains 117,178 questions across four types:
True/False (TF), Multiple Choice (MC), Short Answer / Numeric (SA), and Text Description (TD). Because many
questions require combining evidence across forecast horizons, CyPortQA provides a natural testbed for order-aware
retrieval.
Baselines.We compare OKH-RAG against baselines that span the progression from unstructured to structured
retrieval.Text-RAGperforms dense retrieval over unstructured text chunks and concatenates the top- kresults as
context.GraphRAG[ 11] retrieves over a binary knowledge graph using graph traversal and community summarization.
HyperGraphRAG[ 20] retrieves unordered hyperedges from the same knowledge hypergraph Hwithout modeling
order.OKH-RAGretrieves order-aware hyperedge trajectories from H≺using the full objective in Equation 5. All
methods use the same generator (GPT-4o) and the same embedding model ( text-embedding-3-small ), isolating
the effect of retrieval structure. For OKH-RAG, we use beam search with B= 8 ,L= 8 ,N= 3 , and default
hyperparameters(λ, µ, ν, ρ) = (1.2,0.3,0.2,0.5).
Evaluation metrics.We evaluate answer accuracy across four question types. For True/False (TF) and Multiple
Choice (MC), we use exact match; for Short Answer / Numeric (SA), tolerance-based accuracy; and for Text Description
(TD), an LLM-judged semantic score in[0,1]using GPT-4o as the evaluator.
4.2 Structure of the constructed knowledge hypergraph
We first ask whether the extracted knowledge exhibits the kinds of structure that make order-aware retrieval necessary.
Figure 3 contrasts two scenarios at opposite ends of a structural spectrum. The hypergraph for Hurricane ALEX (left
panel) shows strongwithin-horizon regularity: across horizons, the extracted hyperedges follow nearly identical local
phase patterns, yielding clean horizon-stratified clusters. Structural entities such as the cyclone, port, and temporal
anchors connect broadly, while transient states remain largely confined to individual horizons. In this regime, unordered
6

ALEX (2022) — Canaveral Port District, FL
Entities (Outer Ring)
cyclone
horizon_time
port
advisory_statuscone_status
cyclone_state
landfall_forecast
track_forecastHyperedges (Inner Ring)
HE: T-48
HE: T-36HE: T-24
HE: T-12[port] Canaveral Port District, FLCyclone State at T-48Track Forecast at T-48[cyclone] ALEXLandfall Forecast at T-48Cone Status at T-48[cyclone] ALEXAdvisory status at T-12Advisory Status at T-48[horizon_time] T-48 hours before expecteTropical Storm at T-36Track forecast at T-36[horizon_time] T-36 hours before expecte Landfall forecast at T-36
Cone status at T-36
[horizon_time] T-24 hours before expecte
Tropical Storm at T-24
Track forecast NE at T-24
[horizon_time] T-12 hours before expecte
Landfall near Naples/Ft. Mye
Within uncertainty cone at T
Cyclone state at T-12
Track forecast at T-12
Landfall forecast at T-12
Cone status at T-121. [T-48] has_cyclone_state2. [T-48] forecasts_track3. [T-48] forecasts_landfall4. [T-48] has_hours_to_landfall5. [T-48] has_uncertainty_cone_stat6. [T-48] has_warning_status7. [T-48] applies_warning_to_region8. [T-36] has_cyclone_state9. [T-36] forecasts_track10. [T-36] forecasts_landfall11. [T-36] has_hours_to_landfall12. [T-36] has_uncertainty_cone_stat13. [T-36] has_warning_status 14. [T-24] has_cyclone_state
15. [T-24] forecasts_track
16. [T-24] forecasts_landfall
17. [T-24] has_hours_to_landfall
18. [T-24] has_uncertainty_cone_stat
19. [T-24] has_warning_status
20. [T-12] has_cyclone_state
21. [T-12] forecasts_track
22. [T-12] forecasts_landfall
23. [T-12] has_hours_to_landfall
24. [T-12] has_uncertainty_cone_stat
25. [T-12] has_warning_status(a) ALEX (2022), Canaveral.
ARLENE (2023) — PortMiami, FL
Entities (Outer Ring)
cyclone
horizon_time
port
cone_statuscyclone_state
hazard_forecast
impact_timeline
track_forecastHyperedges (Inner Ring)
HE: T-36
HE: T-24HE: T-12[port] PortMiami, FLCyclone state at T-36Track forecast at T-36[cyclone] ARLENETropical DepressionTrack Forecast ARLENE T-24[horizon_time] T-48 hours before expecteCone status at T-36Cone Status ARLENE PortMiami[horizon_time] T-36 hours before expecteWind forecast at T-36Wind Forecast T-24
[horizon_time] T-24 hours before expecte
Rainfall forecast at T-36
Rainfall Forecast T-24
[horizon_time] T-12 hours before expecte
Tropical Storm
Track Forecast at T-12
Cone Status at T-12
Wind Hazard at T-12
Rainfall Hazard at T-12
Impact Timeline for ARLENE a1. [T-36] has_cyclone_state2. [T-36] forecasts_track3. [T-36] forecast_updates_to4. [T-36] forecast_updates_to5. [T-36] changes_status_to6. [T-36] changes_status_to7. [T-36] forecast_updates_to8. [T-36] forecast_updates_to9. [T-24] has_cyclone_state10. [T-24] forecasts_track 11. [T-24] forecast_updates_to
12. [T-24] forecast_updates_to
13. [T-24] changes_status_to
14. [T-24] changes_status_to
15. [T-24] forecast_updates_to
16. [T-24] forecast_updates_to
17. [T-12] has_cyclone_state
18. [T-12] forecasts_track
19. forecast_updates_to (b) ARLENE (2023), PortMiami.
Figure 3:Two structural regimes in the knowledge hypergraph.Outer ring shows typed entities; inner ring shows
ordered hyperedges.
retrieval can often succeed by selecting the correct local snapshot. Tropical Storm ARLENE (right panel) shows a
different regime. Nearly half of its hyperedges arecross-horizon transitions, such as forecast_updates_to and
changes_status_to , linking states across horizons and introducing dependencies that unfold over time. Where
ALEX forms largely separable clusters, ARLENE exhibits visible inter-horizon structure. This contrast highlights when
order-awareness matters most. In phase-regular scenarios, retrieval is largely local. In evolution-rich scenarios, the
key challenge is not only retrieving the right hyperedges, but retrieving them in the right sequence. The order-aware
hypergraphH ≺is designed to support both regimes within a unified representation.
4.3 QA retrieval results
We next examine whether OKH-RAG adapts its retrieved trajectories to the reasoning demands of the query. Figure 4
shows two examples for Hurricane ARTHUR (2020). The first question, “What is the expected landfall location?”,
requirescross-horizon reasoning. The retrieved trajectory begins at T-36 with cyclone state, track, and landfall evidence,
then transitions to T-24 for a second round of timing, probability, and landfall information. Rather than collecting
isolated facts, the trajectory assembles an evolving forecast chain, allowing the generator to synthesize the answer
“Outer Banks, North Carolina” with high confidence. The second question, “Which is Baltimore’s closest weather
forecast location?”, exhibitswithin-horizon factual retrieval. Here the trajectory remains at T-12 and follows a compact
local chain centered on the answer entity, which appears early and is reinforced later through hazard-related context.
This yields an exact match with high confidence. The contrast is informative: for the landfall question, retrieval
prioritizes breadth across horizons; for the station question, it prioritizes depth within a single horizon. In both cases,
retrieval recovers an evidence chain aligned with the query’s reasoning demands rather than a flat list of relevant
hyperedges.
4.4 Comparison with baselines
Table 1 reports answer accuracy across all four question types. The results show a consistent progression from
unstructured retrieval to structured, order-aware retrieval. Moving from Text-RAG to GraphRAG yields a large
gain, indicating the value of explicit relational structure. Replacing binary graphs with hypergraphs yields a further
improvement, confirming the importance of higher-order interactions. OKH-RAG performs best on all four question
types and achieves the highest overall accuracy. The gain over HyperGraphRAG is particularly important because both
methods use the same underlying hypergraph representation; the difference is whether retrieved evidence is treated as
an unordered set or an ordered trajectory. This isolates the contribution of order-awareness beyond the representational
7

Step 1
[T-36] has_cyclone_state
cyclone_stateStep 2
[T-36] forecasts_track
track_forecastStep 3
[T-36] forecasts_landfall
landfall_forecastStep 4
[T-36] forecasts_landfall
landfall_forecastStep 5
[T-24] has_hours_to_landfall
timing_markerStep 6
[T-24] has_cumulative_probability
probability_forecastStep 7
[T-24] forecasts_track
track_forecastStep 8
[T-24] forecasts_landfall
landfall_forecast
T-36 hours before expected l
[horizon_time]
Cyclone ARTHUR 2020
[cyclone]
Tropical Storm at T-36
[cyclone_state]Landfall forecast at T-36
[landfall_forecast]Landfall forecast near Outer
[landfall_forecast]T-24 hours before expected l
[horizon_time]
Baltimore, MD
[port]Track forecast to Outer Bank
[track_forecast]Track forecast at T-24
[track_forecast]Landfall forecast at T-24
[landfall_forecast]QAGroup: ARTHUR_2020_Canaveral_Port_District,_FL
Question: What is the expected landfall location if the CT continues at the current speed and direction? Answer with ONLY name of 
the location.
GroundTruth: Georgia and South Carolina offshore waters, North Carolina coast (especially Outer Banks), and portions of the mid-
Atlantic offshore waters
Predicted: Outer Banks, North Carolina
Q-Anchor
A-AnchorBridge
ContextRel: forecasts_landfall
Rel: forecasts_trackRel: has_cumulative_probability Rel: has_cyclone_state Rel: has_hours_to_landfall(a) Cross-horizon reasoning.
Step 1
[T-12] has_closest_data_source
closest_stationStep 2
[T-12] has_leadtime_probability
probability_forecastStep 3
[T-12] has_cumulative_probability
probability_forecastStep 4
[T-12] has_peak_probability_timing
probability_timingStep 5
[T-12] first_expected_hazard
first_hazardStep 6
[T-12] forecasts_hazard_at_horizon
hazard_forecastStep 7
[T-12] forecasts_hazard_at_horizon
hazard_forecastStep 8
[T-48] forecasts_hazard_at_horizon
hazard_forecast
T-12 hours before expected l
[horizon_time]
Cyclone ARTHUR 2020
[cyclone]OCEAN CITY MD
[closest_station]Baltimore, MD
[port]
T-48 hours before expected l
[horizon_time]
Wind forecast at T-48
[hazard_forecast]Gale probability 72h
[gale_probability_set]Rainfall hazard at T-12
[hazard_forecast]QAGroup: ARTHUR_2020_Baltimore,_MD
Question: Which is the Baltimore, MD's closest weather forecast location in the table? Answer with ONLY name of the locaton.
GroundTruth: OCEAN CITY MD
Predicted: OCEAN CITY MD
Q-Anchor
A-AnchorBridge
ContextRel: first_expected_hazard
Rel: forecasts_hazard_at_horizonRel: has_closest_data_source
Rel: has_cumulative_probabilityRel: has_leadtime_probability Rel: has_peak_probability_timing
(b) Within-horizon factual retrieval
Figure 4:Query-adaptive trajectories retrieved by OKH-RAG.Upper layer: ordered hyperedges; lower layer:
entities.
advantage of hypergraphs alone. The largest gains appear on MC and SA, suggesting that order-aware retrieval is
especially beneficial when answers require multi-step disambiguation or synthesis across related evidence.
Table 1: Answer accuracy by question type. Best inbold; second-best underlined .
Method TF MC SA TD Overall
Text-RAG 0.694 0.378 0.198 0.224 0.287
GraphRAG 0.806 0.506 0.321 0.362 0.414
HyperGraphRAG 0.819 0.620 0.435 0.432 0.511
OKH-RAG (ours)0.833 0.652 0.452 0.441 0.534
4.5 Ablation Studies
Table 2 summarizes three ablations isolating the contribution of order-aware retrieval. The permutation stress test
provides the most direct evidence:OKH-RAG (shuffled)reduces overall accuracy from 0.534 to 0.487, the largest drop
observed, despite identical retrieved content. This shows that performance depends not only on which hyperedges are
retrieved, but also on their order. The component analysis further indicates that all structural terms contribute positively.
In particular, removing precedence consistency ( µ) or phase coverage ( ρ) yields the largest degradation, suggesting that
effective reasoning requires both correct ordering and sufficient coverage. Removing entity continuity ( ν) also reduces
performance, while removing order coherence ( λ) has a smaller effect, indicating its role is primarily to guide search
toward better trajectories. The transition-model ablation reinforces this interpretation. Comparingno order,heuristic
order, andlearned orderreveals a consistent hierarchy: performance improves as stronger forms of order modeling
are introduced. Heuristic ordering provides modest gains over order-free retrieval, while the learned transition model
performs best, capturing ordering patterns beyond fixed rules. Overall, these results show that (1) evidence order is
8

a primary driver of reasoning quality, (2) effective trajectories must be both ordered and complete, and (3) learned
precedence modeling provides additional benefits beyond heuristic constraints. Together, these findings confirm that
OKH-RAG’s gains arise specifically from order-aware trajectory retrieval rather than structured retrieval alone.
Table 2:Ablation study of order-aware retrieval.
Group Variant TF MC SA TD Overall∆
Full ModelOKH-RAG (full) 0.833 0.652 0.452 0.441 0.534—
ShuffleOKH-RAG (shuffled) 0.750 0.618 0.397 0.399 0.487−0.047
Components−λ(order coherence) 0.833 0.650 0.449 0.445 0.532−0.002
−µ(precedence) 0.819 0.633 0.434 0.395 0.510−0.024
−ν(entity continuity) 0.819 0.646 0.440 0.407 0.519−0.015
−ρ(phase coverage) 0.819 0.633 0.434 0.395 0.510−0.024
TransitionNo order 0.818 0.621 0.412 0.426 0.501−0.033
Heuristic order 0.821 0.625 0.426 0.430 0.509−0.025
5 Conclusion
This work challenges a core assumption in retrieval-augmented generation: that retrieved evidence can be treated as
an unordered set. We show that this assumption fails for a broad class of reasoning tasks where outcomes depend
on how interactions unfold. To address this, we introduced OKH-RAG, which represents knowledge as order-aware
hypergraphs and formulates retrieval as trajectory inference over higher-order interactions. Across experiments, OKH-
RAG consistently outperforms permutation-invariant baselines, and ablations confirm that these gains arise specifically
from modeling interaction order rather than from structured representation alone. These results highlight that retrieval
quality depends not only on relevance, but also on organization: preserving the structure of evidence sequences is
critical for faithful reasoning. More broadly, our findings suggest a shift from set-based to trajectory-based retrieval,
enabling LLMs to reason over processes, dependencies, and evolving systems. This perspective is particularly important
for domains such as scientific discovery, decision-making under uncertainty, and complex system analysis, where order
is intrinsic to meaning.
References
[1]Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang,
Junjie Zhang, Zican Dong, et al. A survey of large language models.arXiv preprint arXiv:2303.18223, 1(2):1–124,
2023.
[2]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang
Wang, Yidong Wang, et al. A survey on evaluation of large language models.ACM transactions on intelligent
systems and technology, 15(3):1–45, 2024.
[3]Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models
and knowledge graphs: A roadmap.IEEE Transactions on Knowledge and Data Engineering, 36(7):3580–3599,
2024.
[4]Keshu Wu, Pei Li, Yang Zhou, Rui Gan, Junwei You, Yang Cheng, Jingwen Zhu, Steven T Parker, Bin Ran,
David A Noyce, et al. V2x-llm: Enhancing v2x integration and understanding in connected vehicle corridors.
arXiv preprint arXiv:2503.02239, 2025.
[5]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[6]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang,
Haofen Wang, et al. Retrieval-augmented generation for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2(1):32, 2023.
[7]Rui Yang, Boming Yang, Xinjie Zhao, Fan Gao, Aosong Feng, Sixun Ouyang, Moritz Blum, Tianwei She, Yuang
Jiang, Freddy Lecue, et al. Graphusion: A rag framework for scientific knowledge graph construction with a
global perspective. InCompanion Proceedings of the ACM on Web Conference 2025, pages 2579–2588, 2025.
9

[8]Chenchen Kuai, Zihao Li, Braden Rosen, Stephanie Paal, Navid Jafari, Jean-Louis Briaud, Yunlong Zhang, Youssef
Hashash, and Yang Zhou. Knowledge-grounded agentic large language models for multi-hazard understanding
from reconnaissance reports.arXiv preprint arXiv:2511.14010, 2025.
[9]Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. Knowledge graph-guided retrieval augmented
generation. InProceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for
Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8912–8924, 2025.
[10] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A
survey on rag meeting llms: Towards retrieval-augmented large language models. InProceedings of the 30th ACM
SIGKDD conference on knowledge discovery and data mining, pages 6491–6501, 2024.
[11] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha
Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From Local to Global: A Graph RAG Approach to
Query-Focused Summarization, April 2024.
[12] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan
Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering.
Advances in Neural Information Processing Systems, 37:132876–132907, 2024.
[13] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, and Vicente Grau. Medical
graph rag: Towards safe medical large language model via graph retrieval-augmented generation.arXiv preprint
arXiv:2408.04187, 2024.
[14] Bahare Fatemi, Perouz Taslakian, David Vazquez, and David Poole. Knowledge hypergraphs: Prediction beyond
binary relations.arXiv preprint arXiv:1906.00137, 2019.
[15] Alain Bretto. Hypergraph theory.An introduction. Mathematical Engineering. Cham: Springer, 1:209–216, 2013.
[16] Keshu Wu, Yang Zhou, Haotian Shi, Dominique Lord, Bin Ran, and Xinyue Ye. Hypergraph-based motion gener-
ation with multi-modal interaction relational reasoning.Transportation Research Part C: Emerging Technologies,
180:105349, 2025.
[17] Keshu Wu, Zihao Li, Sixu Li, Xinyue Ye, Dominique Lord, and Yang Zhou. Ai2-active safety: Ai-enabled
interaction-aware active safety analysis with vehicle dynamics.arXiv preprint arXiv:2505.00322, 2025.
[18] Chenchen Kuai, Jiwan Jiang, Zihao Zhu, Hao Wang, Keshu Wu, Zihao Li, Yunlong Zhang, Chenxi Liu,
Zhengzhong Tu, Zhiwen Fan, and Yang Zhou. How independent are large language models? a statistical frame-
work for auditing behavioral entanglement and reweighting verifier ensembles.arXiv preprint arXiv:2604.07650,
2026.
[19] Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shihui Ying, Shaoyi Du, Han Hu, and Yue Gao. Hyper-rag:
Combating llm hallucinations using hypergraph-driven retrieval-augmented generation, 2025.
[20] Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin
Kuang, Meina Song, Yifan Zhu, and Luu Anh Tuan. HyperGraphRAG: Retrieval-Augmented Generation via
Hypergraph-Structured Knowledge Representation, October 2025. arXiv:2503.21322 [cs].
[21] Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, and Ning Jiang. Cross-granularity hypergraph retrieval-
augmented generation for multi-hop question answering. InProceedings of the AAAI Conference on Artificial
Intelligence, volume 40, pages 33368–33376, 2026.
[22] Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue, Xingliang Hou, Zhiqiang Tian, Yue Gao, and Shaoyi Du. Cog-rag:
Cognitive-inspired dual-hypergraph with theme alignment retrieval-augmented generation. InProceedings of the
AAAI Conference on Artificial Intelligence, volume 40, pages 31032–31040, 2026.
[23] Chenchen Kuai, Zihao Li, Yunlong Zhang, Xiubin Bruce Wang, Dominique Lord, and Yang Zhou. Us port
disruptions under tropical cyclones: Resilience analysis by harnessing multiple-source dataset.arXiv preprint
arXiv:2509.22656, 2025.
[24] Chenchen Kuai, Chenhao Wu, Yang Zhou, Bruce Wang, Tianbao Yang, Zhengzhong Tu, Zihao Li, and Yunlong
Zhang. Cyportqa: Benchmarking multimodal large language models for cyclone preparedness in port operation.
InProceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 38781–38789, 2026.
[25] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks, April 2021. arXiv:2005.11401 [cs].
[26] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. REALM: Retrieval-Augmented
Language Model Pre-Training, February 2020.
10

[27] Chengrui Wang, Qingqing Long, Meng Xiao, Xunxin Cai, Chengjun Wu, Zhen Meng, Xuezhi Wang, and Yuanchun
Zhou. BioRAG: A RAG-LLM Framework for Biological Question Reasoning, August 2024. arXiv:2408.01107
[cs].
[28] Nicholas Pipitone and Ghita Houir Alami. LegalBench-RAG: A Benchmark for Retrieval-Augmented Generation
in the Legal Domain, August 2024.
[29] Pranav Pushkar Mishra, Kranti Prakash Yeole, Ramyashree Keshavamurthy, Mokshit Bharat Surana, and Fatemeh
Sarayloo. A Systematic Framework for Enterprise Knowledge Retrieval: Leveraging LLM-Generated Metadata to
Enhance RAG Systems, December 2025.
[30] Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense Passage Retrieval for Open-Domain Question Answering, September 2020. arXiv:2004.04906
[cs].
[31] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu,
Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot Learning with Retrieval Augmented
Language Models, August 2022.
[32] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang,
and Haofen Wang. Retrieval-Augmented Generation for Large Language Models: A Survey, March 2024.
arXiv:2312.10997 [cs].
[33] Agada Joseph Oche, Ademola Glory Folashade, Tirthankar Ghosal, and Arpan Biswas. A Systematic Review of
Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions, 2025. Version
Number: 1.
[34] Mingqiu Wang, Izhak Shafran, Hagen Soltau, Wei Han, Yuan Cao, Dian Yu, and Laurent El Shafey. Retrieval
Augmented End-to-End Spoken Dialog Models, February 2024.
[35] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to Retrieve,
Generate, and Critique through Self-Reflection, October 2023. arXiv:2310.11511 [cs].
[36] Wenjia Zhai. Self-adaptive Multimodal Retrieval-Augmented Generation, October 2024.
[37] Diji Yang, Linda Zeng, Jinmeng Rao, and Yi Zhang. Knowing You Don’t Know: Learning When to Continue
Search in Multi-round RAG through Self-Practicing, May 2025.
[38] Shahul Es, Jithin James, Luis Espinosa-Anke, and Steven Schockaert. Ragas: Automated Evaluation of Retrieval
Augmented Generation, September 2023.
[39] Yixuan Tang and Yi Yang. MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop
Queries, January 2024.
[40] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang.
Graph Retrieval-Augmented Generation: A Survey, September 2024. arXiv:2408.08921 [cs].
[41] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. GRAG: Graph Retrieval-Augmented
Generation, May 2024.
[42] Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. Knowledge Graph-Guided Retrieval Augmented
Generation, February 2025.
[43] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. Think-on-
Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented
Generation, July 2024.
[44] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A.
Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari,
Yinglong Xia, and Jiliang Tang. Retrieval-Augmented Generation with Graphs (GraphRAG), December 2024.
[45] Yuqi Wang, Boran Jiang, Yi Luo, Dawei He, Peng Cheng, and Liangcai Gao. Reasoning on Efficient Knowledge
Paths:Knowledge Graph Guides Large Language Model for Domain Question Answering, April 2024.
[46] Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang.
PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths, February 2025.
[47] Haoran Luo, Haihong E, Yuhao Yang, Tianyu Yao, Yikai Guo, Zichen Tang, Wentai Zhang, Kaiyang Wan, Shiyao
Peng, Meina Song, Wei Lin, Yifan Zhu, and Luu Anh Tuan. Text2NKG: Fine-Grained N-ary Relation Extraction
for N-ary relational Knowledge Graph Construction, October 2024. arXiv:2310.05185 [cs].
11

[48] Chulun Zhou, Chunkang Zhang, Guoxin Yu, Fandong Meng, Jie Zhou, Wai Lam, and Mo Yu. Improving
Multi-step RAG with Hypergraph-based Memory for Long-Context Complex Relational Modeling, December
2025.
[49] Costas Mavromatis and George Karypis. GNN-RAG: Graph Neural Retrieval for Large Language Model
Reasoning, May 2024.
[50] Emanuele Rossi, Ben Chamberlain, Fabrizio Frasca, Davide Eynard, Federico Monti, and Michael Bronstein.
Temporal Graph Networks for Deep Learning on Dynamic Graphs, October 2020. arXiv:2006.10637 [cs].
[51] Ke Cheng, Junchen Ye, Xiaodong Lu, Leilei Sun, and Bowen Du. Temporal Graph Network for continuous-time
dynamic event sequence.Knowledge-Based Systems, 304:112452, November 2024.
[52] Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, and Flora D Salim. Beyond single pass, looping through
time: Kg-irag with iterative knowledge retrieval.arXiv preprint arXiv:2503.14234, 2025.
[53] Qingyun Sun, Jiaqi Yuan, Shan He, Xiao Guan, Haonan Yuan, Xingcheng Fu, Jianxin Li, and Philip S Yu. Dyg-rag:
Dynamic graph retrieval-augmented generation with event-centric reasoning.arXiv preprint arXiv:2507.13396,
2025.
[54] Mikhail Galkin, Priyansh Trivedi, Gaurav Maheshwari, Ricardo Usbeck, and Jens Lehmann. Message Passing for
Hyper-Relational Knowledge Graphs, September 2020.
[55] Leonie Neuhäuser, Michael Scholkemper, Francesco Tudisco, and Michael T. Schaub. Learning the effective
order of a hypergraph dynamical system.Science Advances, 10(19):eadh4053, May 2024.
[56] Alec Kirkley. Inference of dynamic hypergraph representations in temporal interaction data.Phys. Rev. E,
109:054306, May 2024.
[57] Saiping Guan, Xueqi Cheng, Long Bai, Fujun Zhang, Zixuan Li, Yutao Zeng, Xiaolong Jin, and Jiafeng Guo. What
is Event Knowledge Graph: A Survey.IEEE Transactions on Knowledge and Data Engineering, 35(7):7569–7589,
July 2023.
[58] David Stawarczyk, Matthew A. Bezdek, and Jeffrey M. Zacks. Event Representations and Predictive Processing:
The Role of the Midline Default Network Core.Topics in Cognitive Science, 13(1):164–186, January 2021.
A Detailed Formulations of Methodology
This appendix provides the formal definitions, algorithmic details, and design rationale supporting the methodology in
§ 3. We follow the same three-stage organization: knowledge hypergraph construction (§ A.1–A.2), order learning
(§ A.3–A.4), retrieval (§ A.5–A.7), and generation (§ A.8).
A.1 Formal definitions
The main text introduces entities and hyperedges as tuples. Here we expand on the design rationale behind each
component and provide the complete relation vocabulary.
Entity.An entityv∈ Vis a tuple
v= (n v, τv, dv, cv),(6)
where nvis the entity name, grounded in the source hyperedge description ( nv⊆se);τv∈ T is a type drawn from a
domain-specific type system; dvis a natural-language explanation of the entity’s role; and cv∈(0,1] is a confidence
score reflecting extraction certainty. The grounding constraint nv⊆seensures traceability: every entity can be linked
to a specific span in the source text, supporting provenance tracking and faithfulness evaluation.
The type systemTdistinguishes three functional categories, each serving a distinct role during retrieval:
•Persistent objects(PORT,CYCLONE): domain-level entities that persist across the scenario. These appear in
hyperedges at many sequence positions and serve asanchors—their shared presence across steps is what entity
continuity scoring (Ovlp) exploits to favor coherent trajectories.
•Transient states(CYCLONE_STATE,OPERATION_STATUS,HAZARD_FORECAST): horizon-specific instances
capturing the system’s configuration at a particular point. Each is typically associated with a single horizon,
encoding how persistent objects evolve—e.g., a cyclone’s category at T-96 versus T-48.
•Temporal anchors(HORIZON_TIME, e.g., horizon:T-48 ): structural entities that index interactions to positions
in the precedence relation. Every horizon-grounded hyperedge includes exactly one temporal anchor, enabling
horizon-scoped candidate selection and order enforcement during retrieval.
12

This typology directly governs how reasoning chains are traced: persistent objects provide cross-horizon continuity,
transient states provide within-horizon specificity, and temporal anchors provide the positional scaffolding that makes
order-aware retrieval possible.
Hyperedge.A hyperedgee∈ Eis a tuple
e= (V e, re, se,ae, ce),(7)
with five components:
•Ve={v 1, . . . , v m} ⊆ V : the participating entity set ( m≥2 ). The higher-order cardinality of Veis what
distinguishes hyperedges from binary edges and enables atomic representation of multi-factor interactions.
•re∈ R: a typed relation from the controlled vocabulary (Table 3), categorizing thekindof interaction and enabling
phase-aware retrieval.
•se: a natural language description preserving the full semantic content of the interaction. Unlike structured triples, se
can express complex conditional dependencies involving multiple entities and quantitative thresholds simultaneously,
and is directly used to compute the hyperedge embeddingh e.
•ae={(k j, uj)}A
j=1: key–value attributes recording quantitative and categorical properties, enabling precise factual
grounding during generation and numeric question answering.
•ce∈(0,1]: extraction confidence, available for downstream retrieval weighting.
Relation vocabulary.The vocabulary Rspans the complete life cycle of domain processes, from initial state
characterization through impact and recovery (Table 3). The ordering of families 1–12 corresponds to the canonical
phase progression used for within-horizon precedence (§ A.3); family 13 operates between horizons. All relation
strings are normalized to this vocabulary via an alias mapping that absorbs variation in LLM-generated labels (e.g.,
closes_port→has_operation_status).
Table 3: Relation vocabularyR: 13 semantic families in canonical phase order.
# Family Representative relations
1 Cyclone statehas_cyclone_state,has_category_state,has_motion
2 Track & landfallforecasts_track,forecasts_landfall
3 Timinghas_hours_to_landfall,has_forecast_window
4 Advisoryhas_watch_status,has_warning_status
5 Probabilityhas_leadtime_probability,has_cumulative_probability
6 Hazard forecastforecasts_hazard_at_horizon
7 Hazard observationobserves_hazard_at_horizon
8 Thresholdhas_threshold_status
9 Additional hazardshas_additional_hazard
10 Operationshas_operation_status,affects_vessel_handling
11 Impacthas_impact_prediction,causes_operational_disruption
12 Recoveryhas_recovery_status,starts_recovery
13 Cross-horizonforecast_updates_to,intensifies_to,changes_status_to
A.2N-ary relational extraction
The extraction pipeline converts unstructured documents into the knowledge hypergraph H. Three aspects of the pipeline
merit detailed description: the core extraction step, the per-horizon decomposition strategy, and the post-processing
normalization.
Core extraction.For each document d∈ K , a language model π(GPT-4o) receives an extraction prompt pextand
producesn-ary relational facts:
Fd={f 1, . . . , f k} ∼π(F |p ext, d),(8)
where each fi= (e i, Vei)pairs a hyperedge with its entity set. The prompt instructs the model to segment the input
into knowledge fragments (yielding se), recognize and type all entities (yielding Vewithτvanddv), and assign relation
labels from Rwith quantitative attributes. Output is constrained to a strict JSON schema matching the definitions
above.
13

Per-horizon extraction.Multi-horizon scenario documents can span thousands of tokens. Single-call extraction
risks truncation, entity conflation across horizons, and reduced attribute coverage. We therefore split each document at
horizon headers (detected via regex patterns such as “T-48 hours before expected landfall:”) and extract each block
independently. This per-horizon strategy yields substantially higher entity and attribute coverage, at the cost of a
subsequent merge step.
Post-processing.Five normalization steps ensure consistency across per-horizon extractions:
1.Relation normalization: relation strings are mapped to Rvia the alias table and fuzzy matching, absorbing
LLM-generated variation.
2.Entity ID canonicalization: identifiers are rewritten to a hierarchical convention encoding family, storm, port, and
horizon (e.g.,wind_fcst:IRMA:port_arthur:T-48), ensuring cross-block consistency.
3.Horizon entity injection: a canonical temporal anchor ( horizon:T-48 ) is created for each detected horizon and
added to all hyperedges at that horizon, making horizon membership structurally explicit.
4.Cross-horizon edge synthesis: for entity families at multiple horizons, synthetic hyperedges
(forecast_updates_to ,changes_probability_to ) are created to represent evolution, providing the
cross-horizon links that the precedence construction requires.
5.Deduplication: unique hyperedge IDs are computed as hashes of relation, entity set, and evidence text; exact
duplicates are collapsed.
Aggregation yields the complete hypergraph:
H=[
d∈K[
fi∈FdVei,[
d∈K{ei|fi∈ Fd}
.(9)
A.3 Precedence construction
The precedence relation ≺overEis constructed from domain-informed structural rules that produce a DAG within
each knowledge group.
Sequence-indexed representation.Hyperedges are assigned to sequence positions ℓ∈ {1, . . . , L} , yielding states
H(ℓ)= (V,E(ℓ))and the precedence relation:
ei≺ej⇐⇒ ∃ℓ 1< ℓ2s.t.e i∈ E(ℓ1), ej∈ E(ℓ2).(10)
Structural rules.Four rule families determine≺:
Within-horizon phase ordering.At the same horizon, hyperedges are ordered by semantic phase following Table 3:
cyclone state →track→landfall →timing →advisory →probability →hazard →threshold →operation →impact →
recovery. This reflects logical dependency: an operational decision presupposes a hazard assessment, which presupposes
an advisory.
Cross-horizon family evolution.Hyperedges in the same semantic family at different horizons follow decreasing horizon
order: T-96≺T-72≺T-48≺T-24≺T-12. This ensures that evolving information is presented chronologically.
Causal chain rules.Four within-horizon inter-phase constraints encode dominant causal pathways: advisory ≺hazard
forecasts, hazard assessments ≺operational decisions, hazard assessments ≺impact predictions, and impact predictions
≺recovery status.
Family-to-change ordering.Within-horizon hyperedges in a semantic family precede any cross-horizon change
hyperedges (family 13) for the same family, ensuring the “before” state is presented before the transition connecting it
to the next horizon.
Canonical trajectory.Topological sorting the resulting DAG with deterministic tie-breaking (phase rank, family rank,
lead time, text position) yields a canonical linear ordering per knowledge group. This ordering serves two purposes: it
provides the ground-truth trajectory for document-order supervision in training (§ A.4), and it defines the structural
precedence thatPrec(γ)enforces during retrieval.
14

A.4 Transition model training
The learned transition modelP θcomplements the structural precedence≺by capturing soft ordering preferences that
rules alone cannot express—e.g., that a wind forecast is more likely to be followed by a surge forecast than by a recovery
update, even when both are valid under≺.
Embedding.Hyperedges are embedded as he∈R1536via OpenAI text-embedding-3-small . The input text
concatenates the relation type, evidence string se, entity names and types, and key attribute values, providing both
structural and semantic information.
Low-rank parameterization.The weight matrix W=U⊤VwithU, V∈R64×1536yields 196,608 parameters
versus ∼2.4M for the full matrix, while preserving the asymmetry essential for directional modeling. The partition
function is approximated via sampled softmax withK= 64global negatives per example.
Self-supervised signals.Three signal families, each exploiting a different form of implicit order, compose the training
set:
Document order ( Pdoc).For co-occurring hyperedges (ei, ej)withπ(ei)< π(e j), the pair joins P; the reversal (ej, ei)
and random cross-group pairs join N. This is the most abundant but noisiest signal: document order is a proxy for, not
a guarantee of, logical precedence.
Entity-overlap consistency ( Pent).Pairs sharing entities ( |Vei∩Vej| ≥1 ) within the same knowledge group join Pif
they appear in canonical trajectory order (§ A.3). This signal is sparser but higher-quality: shared entities imply topical
relatedness, and the canonical order provides a principled direction.
Retrieval-induced preference ( Pret).After initial training, retrieval traces from correct answers yield additional positive
consecutive pairs, creating a self-training loop: better retrievals improve the model, which improves subsequent
retrievals.
A.5 Order sensitivity: proof and discussion
Proposition 1 asserts that permutation-invariant retrieval is insufficient for order-sensitive reasoning. We provide a
constructive proof and discuss scope.
Constructive proof.Letk 1, k2, k3be three interactions:
k1:At T-96, the storm is a tropical depression (Category 0).
k2:At T-48, the storm has intensified to Category 2; uncertainty cone covers the port.
k3:At T-12, gale-force wind probability exceeds 80%; port restricts vessel movements.
For the query “Was the port restriction justified?”, the sequence (k1, k2, k3)presents progressive escalation supporting
“yes,” while (k3, k2, k1)presents the restriction before its justification, supporting “premature.” Since these yield
different answer distributions,P(y|k 1, k2, k3, q)̸=P(y|k 3, k2, k1, q).
Scope.The proposition requires only thatsomequery’s answer depends on order—not that all queries do. This
minimal requirement is satisfied in any domain with precedence-dependent outcomes: clinical diagnosis, legal reasoning,
cascading engineering failures, or operational decision-making under evolving hazards.
A.6 Retrieval scoring terms
All terms in Equation 5 are designed for efficient incremental computation during beam search.
Relevance. Rel(e, q) = cos(h e,q): cosine similarity in R1536between hyperedge and query embeddings. Precom-
puted once per query for all candidates.
Order coherence. Trans(γ) =PL−1
k=1logP θ(e(k+1)|e(k)): cumulative log-transition probability, computed
incrementally from the precomputed|C| × |C|matrix.
15

Precedence consistency.
Prec(γ) ={k:e(k)≺e(k+1)}
{k: (e(k), e(k+1))∈(≺ ∪ ≻)}.(11)
Fraction of ordered consecutive pairs consistent with≺; unrelated pairs are excluded from the denominator.
Entity continuity.
Ovlp(γ) =1
L−1L−1X
k=1|Ve(k)∩Ve(k+1)|
|Ve(k)∪Ve(k+1)|.(12)
Mean Jaccard similarity of entity sets between consecutive steps.
Phase coverage.
Cov(γ) ={ϕ(e(k)) :k= 1, . . . , L} ∩ S
|S|,(13)
where ϕ(e) maps a hyperedge to its reasoning phase and S={advisory,hazard_forecast,hazard_observation,operation_status,impact_prediction,recovery_status} .
A.7 Inference algorithms
Beam search.Algorithm 1 formalises order-aware beam search. The key distinction from standard beam search is
sequence-dependent scoring: the score of appending e′to beam βdepends on the last element (via Pθ), the accumulated
entity set (via Ovlp ), and the full prefix (via the diversity penalty). This dependency is what transforms retrieval from
independent ranking into trajectory inference.
Algorithm 1Order-Aware Beam Search
Require:Queryq, candidatesC, modelP θ, widthB, lengthL, weightsλ, µ, ν, ρ
1:PrecomputeRel(e, q)andlogP θ(ej|ei)for alle, e i, ej∈ C
2:B ←top-2Bsingletons byRel(e, q)
3:forℓ= 2, . . . , Ldo
4:B′← ∅
5:foreachβ= (e(1), . . . , e(ℓ−1))∈ B, eache′∈ C \βdo
6:s←Rel(e′, q) +λlogP θ(e′|e(ℓ−1)) +µ·⊮[e(ℓ−1)≺e′]
7:s←s+ν·J(Ve(ℓ−1), Ve′) +ρ·∆Cov(e′, β)
8:B′← B′∪ {(β⊕e′,score(β) +s)}
9:end for
10:Apply diversity penalty; retain top-Bby score
11:end for
12:returnTop-Ntrajectories with score breakdowns
Viterbi dynamic programming.For exact optimisation under the two-term objective (Rel +λ·Trans):
dp[0][j] = Rel(e j, q),(14)
dp[ℓ][j] = max
i
dp[ℓ−1][i] + Rel(e j, q) +λlogP θ(ej|ei)	
,(15)
with backpointers for trajectory recovery. Complexity: O(L· |C|2). Viterbi optimizes the two-term objective exactly
but omits auxiliary terms; it serves primarily as a quality ceiling for beam search.
Candidate scoping.Two stages reduce the search space: (1) top- K(K=80 ) cosine retrieval from the global index;
(2) group-aware expansion via group membership and one-hop entity overlap, with 40% of slots reserved for the query’s
own group when known. The pool is capped at|C|= 150.
A.8 Generation details
Evidence format.Each hyperedgee(k)in a trajectory is presented as:
16

[Step k] [T-XX] [phase=...] [family=...]
Relation: r_e
Evidence: s_e
Reasoning: within_horizon, hazard_to_operation, ...
Entities: entity1 [type]; entity2 [type]; ...
Step indices provide ordering cues; horizon and phase labels provide temporal and logical context; reasoning tags
indicate causal pathway participation; entity lists enable cross-step tracking. A trajectory-level quality summary is
prepended when available.
Multi-trajectory synthesis.Thesingle-callmode presents all Ntrajectories as numbered reasoning paths in one
prompt. The model is instructed to read all paths before answering, treat convergence as a reliability signal, and note
which trajectory supports the final answer. Thefallbackmode generates per-trajectory answers and aggregates via
confidence-weighted voting (categorical) or averaging (numeric). The single-call mode is preferred: it enables direct
cross-referencing rather than post-hoc reconciliation.
Output format.Structured JSON with fields answer ,confidence∈[0,1] , and rationale . Type-specific con-
straints ensure evaluation-friendly output: “Yes”/“No” for true/false, letter labels for multiple choice, numeric values
for quantitative questions, free text for descriptions.
17