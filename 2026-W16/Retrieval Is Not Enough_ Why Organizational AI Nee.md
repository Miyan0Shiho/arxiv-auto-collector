# Retrieval Is Not Enough: Why Organizational AI Needs Epistemic Infrastructure

**Authors**: Federico Bottino, Carlo Ferrero, Nicholas Dosio, Pierfrancesco Beneventano

**Published**: 2026-04-13 17:31:14

**PDF URL**: [https://arxiv.org/pdf/2604.11759v1](https://arxiv.org/pdf/2604.11759v1)

## Abstract
Organizational knowledge used by AI agents typically lacks epistemic structure: retrieval systems surface semantically relevant content without distinguishing binding decisions from abandoned hypotheses, contested claims from settled ones, or known facts from unresolved questions. We argue that the ceiling on organizational AI is not retrieval fidelity but \emph{epistemic} fidelity--the system's ability to represent commitment strength, contradiction status, and organizational ignorance as computable properties.
  We present OIDA, a framework that structures organizational knowledge as typed Knowledge Objects carrying epistemic class, importance scores with class-specific decay, and signed contradiction edges. The Knowledge Gravity Engine maintains scores deterministically with proved convergence guarantees (sufficient condition: max degree $< 7$; empirically robust to degree 43). OIDA introduces QUESTION-as-modeled-ignorance: a primitive with inverse decay that surfaces what an organization does \emph{not} know with increasing urgency--a mechanism absent from all surveyed systems. We describe the Epistemic Quality Score (EQS), a five-component evaluation methodology with explicit circularity analysis. In a controlled comparison ($n{=}10$ response pairs), OIDA's RAG condition (3,868 tokens) achieves EQS 0.530 vs.\ 0.848 for a full-context baseline (108,687 tokens); the $28.1\times$ token budget difference is the primary confound. The QUESTION mechanism is statistically validated (Fisher $p{=}0.0325$, OR$=21.0$). The formal properties are established; the decisive ablation at equal token budget (E4) is pre-registered and not yet run.

## Full Text


<!-- PDF content starts -->

Retrieval Is Not Enough:
Why Organizational AI Needs Epistemic
Infrastructure
Federico Bottino1, Carlo Ferrero1, Nicholas Dosio1, Pierfrancesco Beneventano2
1Kakashi Ventures Accelerator (KVA)
2Massachusetts Institute of Technology
Abstract
Organizational knowledge used by AI agents typically lacks epistemic structure: retrieval
systems surface semantically relevant content without distinguishing binding decisions from
abandoned hypotheses, contested claims from settled ones, or known facts from unresolved
questions. We argue that the ceiling on organizational AI is not retrieval fidelity butepis-
temicfidelity—the system’s ability to represent commitment strength, contradiction status,
and organizational ignorance as computable properties.
WepresentOIDA,aframeworkthatstructuresorganizationalknowledgeastypedKnowl-
edgeObjectscarryingepistemicclass, importancescoreswithclass-specificdecay, andsigned
contradiction edges. The Knowledge Gravity Engine maintains scores deterministically with
proved convergence guarantees (sufficient condition: max degree<7; empirically robust to
degree 43). OIDA introduces QUESTION-as-modeled-ignorance: a primitive with inverse
decay that surfaces what an organization doesnotknow with increasing urgency—a mech-
anism absent from all surveyed systems. We describe the Epistemic Quality Score (EQS), a
reusable five-component evaluation methodology with explicit circularity analysis. In a con-
trolled comparison (n=10response pairs), OIDA’s RAG condition (Minerva, 3,868 tokens)
achieves an EQS of 0.530 vs. 0.848 for a full-context baseline (Cowork, 108,687 tokens); the
28.1×token budget difference is the primary confound for the composite gap. The QUES-
TION mechanism is statistically validated: Minerva produces explicit ignorance declarations
in 10/10 responses vs. 5/10 for Cowork (Fisherp=0.0325, OR= 21.0, Haldane–Anscombe),
an architectural invariant rather than a sampling artifact. The formal properties are estab-
lished; the decisive empirical test—a tag-and-boost ablation at equal token budget (E4)—is
pre-registered and not yet run. We present a concrete evaluation agenda alongside an honest
accounting of what the system guarantees and what it does not.
1 Introduction
An AI agent is asked to summarize an organization’s position on a contested strategic decision.
It retrieves five relevant documents. All five are semantically relevant. Three support the
original hypothesis; one contradicts it with recent market data; one is an unresolved question
about regulatory risk that no one has answered in six weeks. The agent treats all five as
equivalent evidence and produces a confident summary. The summary is fluent, well-organized,
and epistemically incoherent.
This failure is not a retrieval failure. It is an epistemic one. The system found the right
documents; it had no way to interpret their epistemic status. Nothing in the retrieval substrate
distinguished a binding decision from an abandoned hypothesis, a contested claim from a settled
one, or an open question from a resolved finding.
The antagonist.The dominant response to epistemic failures in organizational AI is to im-
prove retrieval: better embeddings, denser indexes, reranking, hybrid search, longer context
1arXiv:2604.11759v1  [cs.AI]  13 Apr 2026

windows, structured entity graphs. These are legitimate improvements within their scope. But
they do not resolve the underlying problem on an epistemically flat substrate. No amount of
retrieval improvement will surface contradiction status that is not encoded, distinguish commit-
ment strength that is not represented, or model organizational ignorance that is not tracked.
The field is pushing on the wrong wall.
Three capabilities remain absent from production systems. First,commitment differ-
entiation: no surveyed system distinguishes a verified decision from a tentative hypothesis
at retrieval time with different importance dynamics. Second,contradiction propagation:
existing approaches either detect contradictions at query time from text (unreliably) or repair
inconsistencies post-hoc—none encode contradictions as signed edges that dynamically suppress
importance scores. Third,ignorance modeling: no system represents what an organization
doesnotknow as a first-class object whose operational cost increases over time.
Our approach.OIDA1addresses this gap by structuring knowledge epistemically at in-
gestion, maintaining that structure deterministically, and exposing it to agents as a first-
class retrieval property. Classification is LLM-assisted and therefore fallible; all subsequent
maintenance—decay, scoring, contradiction propagation, memory zone allocation—is determin-
istic and auditable.
Contributions.
1. AKnowledge Object modelwith nine epistemic classes, signed typed relationships,
and a deterministic importance engine with proved convergence guarantees.
2.QUESTION-as-modeled-ignorance: inverse decay for unresolved questions, opera-
tionalizing accumulated decision risk under uncertainty—a mechanism absent from all
surveyed systems.
3.Signed contradiction propagationapplied to epistemic infrastructure: an established
signed-GNN mechanism applied to organizational epistemic contradiction suppression,
with partial suppression (CONTRADICTS=−0.6) implementing epistemological toler-
ance.
4. TheEpistemic Quality Score (EQS): a reusable evaluation methodology with five
grounded sub-scores, explicit circularity analysis, and a statistical protocol for small-N
comparisons.
5.A controlled pilot comparison(n=10response pairs) with full confound treatment,
plus three formal design properties distinguishing KGE from static epistemic labeling.
2 Related Work
Knowledge graphs and ontological systems.Knowledge graphs represent information
as networks of entities and typed relationships [4]. Enterprise deployments—notably Palantir’s
Ontology [30]—map operational reality into semantically coherent graphs. Temporal knowledge
graphs (TKGs) [5, 17] extend this with validity intervals and time-aware reasoning. However,
TKGs answer “when was this true?”—they do not answer “how much should you trust this
now, given its type?” No TKG applies different decay rates based on thekindof knowledge a
fact represents.
Structured retrieval-augmented generation.RAG [20] is the dominant paradigm for
grounding LLM outputs [12]. GraphRAG [8] uses graph-based community summaries to im-
prove retrieval over flat chunks. LightRAG [14] incorporates entity-relationship graphs into
1From Greekoida(“I know because I have seen”)—a knowing that arises from experience rather than decla-
ration.
2

dual-level retrieval. Both structure entities and relations but not epistemic commitment: a
decision and a hypothesis with the same entity mentions receive the same retrieval treatment.
Li et al. [23] and Li et al. [22] characterize the efficiency frontier between RAG and long-context
LLMs—a tension OIDA’s retrieval design directly engages. Leng et al. [19] document a relia-
bility drop above 64K input tokens; Shi et al. [32] show that irrelevant context degrades LLM
reasoning—both motivations for K-score filtering.
Agent memory systems.MemGPT [29] introduced hierarchical memory paging for agents.
Zep/Graphiti[31]providestemporaltrackingandbi-temporalmodelingviagraph-basedstorage—
the closest system to OIDA in capability, but it computes importance implicitly through cen-
trality and recency rather than through typed epistemic dynamics. Mem0 [6] extracts and
consolidates facts dynamically but does not type them epistemically. MemOS [25] provides
a memory operating system for skill and tool memory. A-MEM [36] implements agentic self-
organizing memory. A recent survey [24] confirms that no production agent memory system sep-
arates LLM-assisted ingestion from deterministic post-ingestion maintenance, and none model
organizational ignorance.
Commercial organizational AI.Glean [13], Notion AI [28], and Microsoft 365 Copilot [7]
represent the current generation of production organizational AI systems—each treats retrieved
knowledge uniformly without epistemic typing, class-specific decay, or ignorance modeling.
Active retrieval and uncertainty.FLARE [18] retrieves new documents when the model
detects low-confidence generation tokens—the closest mechanistic relative to OIDA’s QUES-
TION primitive. The difference is fundamental: FLARE signals parametric model uncertainty;
QUESTION models organizational ignorance encoded at knowledge ingestion. The two are
orthogonal and potentially composable.
Organizational knowledge management.Walsh and Ungson [34] identified organizational
memoryasdistributedacrossindividuals,culture,structures,andprocesses. SteinandZwass[33]
formalized organizational memory information systems. Nonaka [27] theorized knowledge cre-
ation through socialization, externalization, combination, and internalization. Du et al. [11]
survey the implications of generative AI for knowledge management. These frameworks iden-
tify theneedfor epistemic structure; OIDA provides a computational implementation.
Trust, provenance, and contradiction in KBs.PROV-O [26] standardizes provenance
metadata but does not compute epistemic dynamics. Uncertainty management surveys [16]
and confidence propagation methods [21] address quality estimation but not class-specific decay
or signed contradiction propagation. Temporal confidence decay [15] models validity intervals
but applies uniform aging across knowledge types. Knowledge conflict surveys [35, 10] address
detection and repair but do not encode contradictions as persistent negative-gravity edges in a
scoring engine.
Signed graph neural networks and epistemic contradiction.Signed GNNs extend
graph neural network propagation to graphs with both positive and negative edges, learning
node representations that respect the sign of relational ties [9]. Approaches including SGCN
and SNEA establish the propagation mechanism—signed edge weights, balance-theory-aware
aggregation, and sign-flipping through multi-hop paths—as a mature technique in the signed
network literature. OIDA does not claim this mechanism as a novel contribution. What is
novel is theapplicationof signed propagation to epistemic contradiction handling in organiza-
tional knowledge bases: the CONTRADICTS edge (coefficient−0.6) dynamically suppresses
3

Table 1: Positioning: OIDA vs. existing systems across epistemic capability dimensions.✓=
present,∼= partial, — = absent.
System Epist. Typing Class Decay Contr. Prop. Det. Maint. Ignor. Model
GraphRAG — — — — —
LightRAG — — — — —
Zep/Graphiti — — —∼—
MemGPT/Letta — — — — —
Mem0 — — — — —
A-MEM — — — — —
MemOS — — — — —
TKGs — — —✓—
OIDA✓ ✓ ✓ ✓ ✓
the importance score of contradicted Knowledge Objects within the Knowledge Gravity En-
gine, surfacing unresolved organizational contradictions as first-class signals for downstream
AI agents. The partial suppression coefficient (−0.6, not−1.0) implements an epistemological
tolerance principle: contradicted knowledge may remain retrievable, reflecting the organiza-
tional reality that contradictions are often unresolved coexistences rather than logical defeats.
To our knowledge, no existing knowledge graph inconsistency management approach applies
signed edge weights fordynamic importance suppressionwithin a retrieval engine [10]; standard
approaches use static flagging, version pinning, or inconsistency-aware querying without score
propagation.
Cognitive architectures.OIDA’s usage force inherits structure from ACT-R’s base-level
activation [2, 3], replacing the global decay parameter with class-specific rates and extending
spreading activation to signed edges. The connection is structural, not a cognitive science
contribution.
Table 1 summarizes the landscape. No production system makes epistemic quality com-
putable across all five dimensions. The gap is not an oversight—it reflects a field assumption
that better retrieval is sufficient. OIDA tests the alternative hypothesis: that epistemic struc-
ture at the retrieval substrate level is the missing intervention.
3 The OIDA Framework
OIDA consists of three components: the Knowledge Object model, the Knowledge Gravity En-
gine, and the hybrid retrieval architecture. Figure 1 shows the lifecycle. Four design principles
guide the architecture:
Design Principles.
1.Classify at ingestion; maintain deterministically thereafter.LLM-assistedclassification
is fallible; all post-ingestion computation is reproducible and auditable.
2.Model what you do not know.QUESTIONwithinversedecayrepresentsorganizational
ignorance as a first-class object whose cost increases over time.
3.Contradictions are computable signals, not text to be discovered.Signed edges encode
contradiction as persistent negative gravity.
4.Class-specific decay>global decay.A decision and an observation should not age at
the same rate.
4

Ingestion
(LLM-assisted)KGE Cycle
(deterministic)Hybrid Retrieval
(struct+sem+topo)Agent Context
(ranked KOs)typed KOs K-scores ranked set
usage signals
Figure 1: OIDA system lifecycle. Ingestion is LLM-assisted; all subsequent maintenance and
retrieval is deterministic.
Table 2: Epistemic class taxonomy. Seed values and half-lives are working heuristics, not
validated optima.
Class SeedKDecay Half-life Epistemic Role
DECISION 1.00 None∞Binding choice—valid until
superseded
CONSTRAINT 0.90 None∞Non-negotiable structural
boundary
EVIDENCE 0.80 Exp.∼365d Verifiable supporting/refuting
data
NARRATIVE 0.70 None∞Persistent contextual anchor
PLAN 0.65 Exp.∼69d Structured intention with time
horizon
EVALUATION 0.55 Exp.∼198d Informed qualitative
assessment
OBSERVATION 0.40 Exp.∼90d Weak signal not yet
interpreted
HYPOTHESIS 0.30 Exp.∼50d Unverified testable claim
QUESTION 0.30 Inverse Urgency grows Open question requiring
resolution
3.1 Knowledge Objects and the Epistemic Taxonomy
AKnowledge Object(KO) is a tuple:
KOi= (id i,koc i,class i,content i,scores i,edgesi,meta i)
whereclass i∈Cisdrawnfromaclosedtaxonomyofnineepistemicclassesandedgesiisthesetof
typeddirectedrelationships. Scorescompriseafive-dimensionalvector(K,conf,fresh,urg,contr).
The Knowledge Object Coordinate (KOC) is a 7-axis immutable identifier providingO(1)struc-
tural similarity (specification in Appendix A).
The nine classes arise from crossing two orthogonal axes that together determine computa-
tional behavior.
Axis 1: Epistemic commitment strength.Propositions range from explicit ignorance
(a question no one has answered) through uninterpreted signals (observations), provisionally
held claims (hypotheses, plans), evidentially supported assessments (evidence, evaluations),
persistent contextual anchors (narratives), up to verified and binding commitments (decisions,
constraints). This ordering determines seed importance.
Axis 2: Temporal behavior under absence of reinforcement.A DECISION re-
quires an explicit SUPERSEDES event for deactivation. An OBSERVATION loses weight if
unreinforced. A QUESTION gains urgency—unresolved uncertainty becomesmorecostly over
time. This axis determines the decay profile: non-decaying, exponentially decaying, or inversely
decaying.
All 36 class pairs are distinguishable by at least two features (decay type, seed value, half-
life, or semantic role), establishing operational adequacy: no pair can be merged without losing
observable behavioral differences in the KGE (exhaustive enumeration in Appendix E).
5

0 7 14 21 280.420.460.50.540.58
K∗
Q= 0.556
K∗
O= 0.435Divergence from
day 1 (Theorem 1)
DaysK-scoreQUESTION (λ=−0.010)
OBSERVATION (λ=0.015)
EVIDENCE (λ=0.005)
HYPOTHESIS (λ=0.008)
DECISION (λ=0.002)
Figure 2: K-score dynamics over 28 days under stationary inputs (K 0= 0.5,η= 0.1,∆t= 1).
QUESTION KOs (—) diverge upward from day 1 while OBSERVATION KOs (- -) decay —
a direct consequence ofλ question<0(Theorem 1). This simulationexhibitsthe theorem’s
prediction; deployment validation requires production telemetry.
QUESTION as modeled ignorance.QUESTION is the only class with inverse decay:
unresolved questions becomemoreurgent, not less. We frame urgency as operationalizing
accumulated organizational decision risk under unresolved uncertainty, following a Value of
Information (VoI) interpretation: each day a QUESTION remains unresolved, the organization
makes decisions in its shadow, accumulating risk. Urgency provides a monotonically non-
decreasing lower bound on retrieval priority—a rising floor guarantee that modeled ignorance
eventually outranks stale observations. Under stationary inputs from identical initial conditions
(K0= 0.5),K QUESTION andKOBSERVATION diverge from the very first update (t= 1): the
inverse decay ensures QUESTION KOs accumulate urgency immediately. By day 28, the gap
is 0.115 score units (K∗
QUESTION = 0.556vs.K∗
OBSERVATION = 0.435), with an asymptotic gap
of 0.121 (Theorem 1 below). This growing separation—where what is unknown outranks what
is stale—is the visual thesis of the framework. When a QUESTION is resolved (typically by a
DECISION linked via IMPLEMENTS), its urgency drops to zero. No surveyed system has an
equivalent primitive.
Theorem 1(T2: QUESTION Divergence from Day 1).Under stationary inputs (u=e=g=
c= 0),K 0= 0.5,η= 0.1,∆t= 1:
K∗
question =η·seed
η+λ question·∆t=0.1×0.5
0.1 + (−0.010)×1= 0.556>K 0
K∗
observation =η·seed
η+λ observation·∆t=0.1×0.5
0.1 + 0.015×1= 0.435<K 0
SinceK∗
question> K 0> K∗
observation , the two trajectories diverge att= 1(the first discrete
update). The asymptotic gap is0.121; at day 28 the simulated gap is0.115score units. Proof
and full derivation: Appendix D.
Typed relationships.Relationships between KOs are drawn from a closed vocabulary of ten
directed edge types, each carrying a signed coefficient. Key types include SUPPORTS (+1.0),
BASED_ON (+0.8), IMPLEMENTS (+0.7), SUPERSEDES (+0.6), BLOCKS (−0.4), and
CONTRADICTS(−0.6). Positiveedgespropagateimportance; negativeedgesactivelysuppress
it through the gravity computation. Negative-coefficient edges implement signed propagation—
a technique established in the signed GNN literature [9]—applied here to organizational epis-
temic contradiction suppression. The full vocabulary is specified in Appendix B.
6

Signed contradiction and epistemological tolerance.We chose CONTRADICTS=
−0.6(not−1.0) deliberately: contradicted knowledge is suppressed, not erased. This reflects
the design requirement that contradicted knowledge should remain partially retrievable in case
the contradiction itself is wrong. Concretely, mild contradiction (one CONTRADICTS edge)
suppressesKby approximately 22%; strong contradiction (two edges) by approximately 67%.
Approximately two SUPPORTS edges are needed to counter one CONTRADICTS edge. We
term thisepistemological tolerance.
3.2 Knowledge Gravity Engine
The KGE computes an updated importance scoreKfor every active KO at each cycle (default:
every 6 hours):
K(t+1) =clamp/parenleftig
(1−η)·K(t) +η·[seed+u+e+g]−λ class·∆t·K(t)−c,0,1/parenrightig
(1)
The equation decomposes into three forces.Momentum(1−η)K(t)carries forward current
importance.Injectionη[seed+u+e+g]introduces new signals: seed is the class baseline,
usage forceuis retrieval-driven activation via an exponential recency kernel adapted from ACT-
R [2, 3], evidence forceecounts new inbound SUPPORTS edges, and gravity forcegpropagates
importance through signed edges from connected KOs.Negative forces−λ class∆t·K(t)−c
apply class-specific decay and contradiction penalty. Force formulas are detailed in Appendix C.
Under stationary inputs (before clamping), the per-node update converges to a unique fixed
point:
K∗=η·[seed+u+e+g]−c
η+λclass·∆t(2)
Coupled convergence.For the full coupled system, we prove a sufficient condition via the
Gershgorincircletheorem: convergenceisguaranteedwhenmax_degree<g scale/max|COEFF|
≈7.14(Appendix D). The per-node contraction factor is uniformly in[0.845,0.850]across all
classes, leaving a coupling budget of approximately 0.15 per node. Empirically, convergence is
observed for all tested configurations—including graphs with max degree 43 (six times beyond
the sufficient condition)—due to tanh saturation bounding gravity output to(−1,1), clamping
providing non-expansive projection, and mixed-sign edges partially canceling. The gap between
the analytical bound and empirical robustness is itself an informative finding.
Memoryzones.KOsareallocatedtofourzonesbyK-score: CoreMemory(K≥0.40,always
in agent context), Working Memory (0.10≤K <0.40, retrieved when relevant), Peripheral
(0.05≤K <0.10, targeted queries only), and Dormant (K <0.05, excluded from gravity
computation). No KO is ever deleted—only excluded from active computation.
Design properties vs. tag-and-boost.Table 3 formalizes three design guarantees that
distinguish KGE from a static epistemic labeling baseline (tag-and-boost: same nine classes,
BM25+cosine hybrid retrieval, recency weighting only, no dynamic scoring). These properties
are proved at the design level; empirical validation requires the E4 ablation (pre-registered, not
yet run).
3.3 Hybrid Retrieval
The hybrid score combines three independent similarity layers2:
H(q,i) =α·S struct(q,i) +β·S sem(q,i) +γ·S topo(q,i)(3)
2The production deployment extends this three-component formulation to a nine-componentVesta Scorethat
additionally incorporates epistemic confidence, freshness (decay state), graph mass (centrality), goal alignment,
domain match, functional fit, and historical stability, each with configurable weights summing to 1.0 and query-
7

Table 3: Three formal design properties distinguishing OIDA KGE from a tag-and-boost base-
line(staticepistemiclabels+recencyweighting). Whetherthesepropertiesproducemeasurable
EQS improvements at equal token budget is the subject of planned ablation E4.
Property OIDA KGE Tag-and-Boost (static labels)
P1:
Fixed-point
convergenceConverges to uniqueK∗
(contraction factor∈[0.845,0.850])
encoding class, usage history,
evidence, and graph neighborhood.No dynamics; importance is static
after labeling. No fixed point
encoding class-specific equilibrium.
P2:
QUESTION
rising floorλQUESTION =−0.010guarantees
K∗
QUESTION = 0.556>K 0.
Unresolved ignorance cannot be
permanently deprioritized.No mechanism differentiates
QUESTION from OBSERVATION
temporally. An aging QUESTION
is treated identically to an aging
OBSERVATION.
P3:
Contradiction
suppressionCONTRADICTS (−0.6)
propagates negative gravity each
cycle; contradicted KO converges
to lowerK∗. Contradiction is a
structured, evolving signal.A contradiction tag does not affect
retrieval priority of the
contradicted KO. No dynamic
propagation.
Structural similarityS struct: computed from KOC axis alignment inO(1), no database
access.Semantic similarityS sem: cosine similarity over embedding vectors, rescaled to[0,1].
Topological similarityS topo: inverse hop distance in the epistemic graph. Configured de-
faults:α= 0.30,β= 0.50,γ= 0.20.
The final ranking multiplies hybrid similarity by contextual importance:
R(q,i) =H(q,i)·K eff(i,q)(4)
whereK eff(i,q) =K global(i)·max(0.10,ϕ ctx(i,q)). The floor of 0.10 prevents complete col-
lapse of globally important KOs. Thecontextual attention functionϕ ctxmodulates global
importance by query-local relevance:
ϕctx(i,q) =w e·ϕentity(i,q) +w d·ϕdomain (i,q) +w a·ϕanchor (i,q)(5)
whereϕ entity(i,q) =1[KOC entity(i) =q.primaryEntity]tests entity alignment,ϕ domain (i,q) =
1[KOC domain (i) =q.domain]testsdomainalignment,andϕ anchor (i,q) =|anchors(i)∩q.activeAnchors|/max(|q.activeAnchors|,1)
measures overlap with the query’s active contextual anchors. Configured defaults:w e= 0.40,
wd= 0.35,w a= 0.25.
We chose hand-crafted weights over learned parameters because the corpus is small (500
KOs), determinism and auditability are design requirements, and understanding system behav-
ior currently matters more than marginal optimization.
4 Evaluation
4.1 The Epistemic Quality Score
We propose the Epistemic Quality Score (EQS) as a reusable evaluation framework for any
system claiming to provide epistemic structure for organizational AI. The composite metric
comprises five sub-scores:
EQS= 0.20·ECA+ 0.25·CP+ 0.20·CR+ 0.20·EC+ 0.15·DE
typepresets. Wepresentthereducedformulationhereforclarity; theVestaScorewillbespecifiedinaforthcoming
companion paper on the full OIDA architecture.
8

ECA(Epistemic Classification Accuracy, 0.20): Does the response correctly distinguish
epistemic types?CP(Contextual Precision, 0.25): Is the response grounded in evidence, not
hallucinated? The highest weight reflects the primacy of faithfulness.CR(Contextual Recall,
0.20): Does the response cover relevant context comprehensively?EC(Epistemic Coherence,
0.20): Does the response handle contradictions and epistemic tensions appropriately?DE
(Decision Enablement, 0.15): Does the response enable informed organizational decisions?
Each sub-score uses 4-point calibrated anchors (0.1, 0.4, 0.7, 1.0) with behavioral descrip-
tions. The statistical protocol uses paired Wilcoxon signed-rank tests for small-Ncomparisons,
McNemar tests for binary classification, and Cohen’sdas the primary effect size measure.
Circularity analysis.We acknowledge that 60% of composite weight (CP + CR + DE)
is system-independent. However, 20% (ECA) has high circularity—it tests whether responses
respect OIDA’s own taxonomy. The remaining 20% (EC) has moderate circularity. An inde-
pendent evaluation by domain experts assessing decision quality without reference to OIDA’s
framework would be needed to validate the design choices themselves. As an LLM-as-judge
method, EQS is subject to known biases [37]. We recommend cross-validation with human
expert evaluation for deployment decisions.
4.2 Comparative Results: Minerva vs. Cowork
Experimental setup.We compare two conditions on the ClearPath corpus (≈500KOs):
Minerva(OIDARAG,structuredretrieval, top-kKOs)andCowork(full-contextbaseline, entire
corpus in context). We rann= 10parallel LLM response pairs on the query: “What are the
main bottlenecks identified in ClearPath’s current operational processes?” (model: claude-
sonnet-4-6).
Token budget and primary confound.Condition A (Minerva) used 3,868 input tokens;
Condition B (Cowork) used 108,687—a 28.1×difference. This token budget differential is the
primary confound for all EQS sub-score comparisons and must be held in view throughout.
Minerva retrieves≈5–7 KOs; Cowork accesses the full corpus.
QUESTION declaration rate (unconfounded finding).Minerva declares its ignorance;
Coworkhappenstoknow. Inall10runs,MinervaincludesanexplicitKnowledgeGapparagraph—
adesigninvariant, notasamplingartifact(Fisherp= 0.0325). TheparallelCoworkrunssurface
the same limitation in only 5 of 10 responses—incidentally, because the right text happened to
be in context.
Table 4: QUESTION KOs withλ<0are structurally privileged in top-kretrieval (Theorem 1),
producing an explicit “Knowledge Gap” section in every Minerva response as a design invariant.
Cowork surfaces the same underlying knowledge gap in 5/10 responses incidentally.
Metric Minerva (OIDA RAG) Cowork (Full-Context)
Explicit ignorance declarations 10/10 (100%) 5/10 (50%)
Fisher’s exactp(two-tailed) 0.0325 —
Odds ratio (Haldane–Anscombe) 21.0 —
The zero non-declaration cell in Minerva is a design invariant:λ QUESTION<0guaran-
teesK∗
QUESTION> K 0from day 1, making QUESTION KOs structurally privileged in top-k
retrieval. This is the paper’s cleanest architectural result—it has a proved mechanism, a statis-
tical test, and no token-budget confound.
9

Citation Precision: hallucination-free in both conditions.Both Minerva (CP= 0.660)
andCowork(CP= 0.895)arehallucination-freeacrossall10runs. Structuralguaranteesmatter
even when both conditions produce accurate text—the ECA and EC gaps reflect epistemic
architecture, not error prevention. CP is the most confound-resistant sub-score;∆ = 0.235
reflects grounding depth, not error rate.
EQS sub-score comparison.Table 5 presents the full comparison. The dominant driver
of the composite gap is Contextual Recall (∆ = 0.540), reflecting retrieval breadth: Minerva
retrieves≈5–7 KOs; Cowork ingests the full corpus. No architectural conclusion may be drawn
from CR without token-equalized comparison (E2, planned). ECA (∆ = 0.245) and EC (∆ =
0.275) test epistemic structure; their attribution to architecture vs. breadth requires the E4
ablation.
Table 5: EQS sub-score comparison (n= 10response pairs, mean±SD). Cowork uses 28.1×
more input tokens than Minerva. CR (Contextual Recall) is predominantly breadth-mediated.
CP is hallucination-free in both conditions. Cohen’sdvalues are inflated by near-zero within-
condition variance due to architectural determinism. Wilcoxon signed-rank (one-sided):W=
55.0,p= 0.000977.
Sub-score Wt Minerva Cowork∆Note
ECA 0.200.575±0.026 0.820±0.0420.245 —
CP 0.250.660±0.021 0.895±0.0160.235 Hallucination-free (both)
CR 0.200.340±0.039 0.880±0.0260.540 Breadth-mediated
EC 0.200.540±0.032 0.815±0.0340.275 —
DE 0.150.520±0.042 0.810±0.0390.290 —
EQS—0.530±0.025 0.848±0.0170.318Wilcoxonp=0.000977
Token input — 3,868 108,68728.1×Primary confound
Both conditions are hallucination-free (CP≥0.66).The cleanest architectural signal is
Citation Precision, which is not breadth-mediated.
H1 status.Whether KGE’s dynamic machinery adds value beyond static epistemic labeling
at equal token budget remains an open empirical question (E4, pre-registered falsification con-
dition). If a tag-and-boost baseline achieves ECA and EC scores within one standard error of
full OIDA at equal token budget (≈3,868input tokens), the KGE is not justified at the current
deployment scale. The three design properties in Table 3 establish architectural distinction at
the formal level; E4 tests whether this distinction is empirically observable.
4.3 Deployment Observations
OIDA is deployed as the operational knowledge infrastructure of a venture studio. Approxi-
mately 500 KOs span five ventures, three client engagements, and internal strategy, maintained
over four weeks of KGE cycles with sources from Notion, Google Calendar, and Slack. These
observations constitute design validation, not controlled evaluation.
K-score distribution.10–15% of KOs settle in Core Memory (K≥0.40). The remaining
distribution concentrates in Working Memory with a long tail of Peripheral KOs—consistent
with the expectation that a minority of organizational knowledge is operationally central at any
time.
K-score trajectories.Simulation of class-specific dynamics over 28 days exhibits dif-
ferentiated behavior: DECISION remains stable near 1.0, EVIDENCE decays slowly, OB-
10

Table 6: What the system guarantees and what it does not.
The system guarantees The system does not guarantee
Deterministic maintenance: same
class, same parameters, same
trajectory every timeUniversal parameter validity:
configured heuristics may
underperform in organizations with
different knowledge patterns
Typed epistemic structure: every KO
carries class, scores, and signed edgesPerfect ingestion classification: typing
quality depends on LLM-assisted
classification
Explicit contradiction surfacing via
negative-gravity edges that are
computationally visibleComplete contradiction detection: the
system models contradictions explicitly
created, not those implicit in text
Coupled convergence for max degree
<7; empirically robust to degree 43Tight convergence bound: the
sufficient condition is 6×conservative;
the tightest bound is an open problem
Stable retrieval contract independent
of foundation modelOptimal retrieval quality: hybrid
weights are design priors, not
empirically optimized
Immutable audit trail: no KO is
deleted, all state changes are loggedCalibrated absolute scores:K-values
are relative rankings, not probability
estimates
SERVATION decays toward 0.435, and QUESTION with medium urgency rises toward 0.556.
Crucially,K QUESTION andKOBSERVATION diverge from the very first update (t= 1), confirm-
ing Theorem 1. By day 28, the simulated gap is 0.115 score units (K∗
QUESTION = 0.556vs.
K∗
OBSERVATION = 0.435, Figure 2). These results are simulation-based under stationary inputs;
production K-score evolution requires E6 (planned).
Calibration finding.The 90-day half-life for OBSERVATION proved too long for AI-
adjacent domains where market signals evolve rapidly. For fast-moving domains, halving the
default (to 45 days) is recommended. This was the most informative deployment finding.
5 Limitations and Evaluation Agenda
Table 6 is the paper’s most important table. We state what is established and what is not.
What is not established.E4 (tag-and-boost ablation at equal token budget) is the decisive
test for H1 and has not been run. E2 (token-equalized Minerva at≈20K tokens) is planned
but not executed. The efficiency frontier is a two-point extrapolation. E6 (production K-score
trajectory logging, 28 days) is undesigned. The QUESTION mechanism has no dedicated EQS
sub-score. Deployment is at a single site with hand-tuned parameters.
Confounds.The Minerva vs. Cowork comparison confounds epistemic typing with retrieval
selection effects and context size differences (28.1×). The ECA evaluator must not privilege
OIDA-specific vocabulary in the E4 blind evaluation. Single evaluator model (claude-sonnet-
4-6); Cohen’sdinflated by architectural determinism (n eff≈1for retrieval). Single corpus
(ClearPath), single query.
Decisive falsification commitment.If E4 finds that a tag-and-boost baseline achieves
ECA and EC scores within one standard error of full OIDA (SE ECA≈0.0083, SE EC≈0.010)
11

at equal token budget (≈3,868input tokens), then KGE’s dynamic machinery is not justified at
the current deployment scale. The contribution would reduce to “epistemic labeling is useful”
and the dynamic scoring machinery would be a research artifact.
Evaluation agenda.
•E2: Token-equalized Minerva (≈20K tokens) to isolate epistemic structure from retrieval
breadth
•E4: Tag-and-boost ablation at 3,868 tokens (pre-registered; primary H1 test)
•E5: Contradiction-detection query to isolate P3 empirically
•E6: Production K-score trajectory logging over 28 days
•Per-class classification accuracy audit (target: F1>0.7)
•QUESTION urgency validation by domain experts (5–10 QUESTION KOs)
6 Conclusion
This paper names a specific antagonist—the assumption that better retrieval solves organiza-
tional AI—and tests an alternative: that epistemic structure at the retrieval substrate level is
the missing intervention. OIDA provides a formal framework with convergence guarantees, an
ignorance-modeling primitive, and signed contradiction propagation applied to organizational
epistemic infrastructure. A pilot comparison (n= 10response pairs) documents an EQS gap
(Minerva 0.530 vs. Cowork 0.848) dominated by retrieval breadth at a 28.1×token budget dif-
ference, and statistically validates the QUESTION declaration mechanism (Fisherp= 0.0325).
The EQS provides a reusable methodology for measuring whether epistemic structure matters.
The guarantees table states what is established; the evaluation agenda specifies how to test
what is not.
Four design principles transfer beyond OIDA: classify at ingestion and maintain determin-
istically; model what you do not know; encode contradictions as computable signals; apply
class-specific decay rather than global decay. A practitioner building organizational AI in-
frastructure can adopt these principles regardless of whether OIDA’s specific implementation
survives empirical testing. The first two principles (P1: classify at ingestion; P2: QUESTION
with rising urgency) require only LLM-assisted metadata tagging on any existing knowledge
base—no graph infrastructure needed. The latter two (signed contradiction propagation and
class-specific decay) require the full KGE; their empirical value over static labeling is precisely
the question E4 will answer.
The field has spent five years improving how AI finds organizational knowledge. It may be
time to improve what organizational knowledgeis—before the agent reads it.
Acknowledgements.This work was developed within the research infrastructure of Pog-
gioAI. We partially used PoggioAI/MSc for this manuscript [1]. We thank Alberto Trivero and
Tommaso Portaluri for discussion on AI, statistical, and informatics matters.
References
[1] MahmoudAbdelmoneum, PierfrancescoBeneventano, andTomasoPoggio. PoggioAI/MSc:
ML theory research with humans on the loop. Technical Report Technical Report v0, MIT,
2026.
[2] John R. Anderson, Daniel Bothell, Michael D. Byrne, et al. An integrated theory of the
mind.Psychological Review, 111(4):1036–1060, 2004.
12

[3] John R. Anderson and Lael J. Schooler. Reflections of the environment in memory.Psy-
chological Science, 2(6):396–408, 1991.
[4] Tim Berners-Lee, James Hendler, and Ora Lassila. The semantic web.Scientific American,
284(5):34–43, 2001.
[5] Bingnan Cai, Yongqiang Xiang, et al. A survey on temporal knowledge graph: Represen-
tation learning and applications.arXiv preprint arXiv:2403.04782, 2024.
[6] Prateek Chhikara, Deshraj Khant, et al. Mem0: Building production-ready AI agents with
scalable long-term memory.arXiv preprint arXiv:2504.19413, 2025.
[7] Eleanor Wiske Dillon et al. Early impacts of M365 Copilot.arXiv preprint
arXiv:2504.11443, 2025.
[8] Darren Edge, Ha Trinh, Newman Cheng, et al. From local to global: A graph RAG
approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
[9] others. Signed graph representation learning: A survey.arXiv preprint arXiv:2402.15980,
2024.
[10] others. Dealing with inconsistency for reasoning over knowledge graphs: A survey.arXiv
preprint arXiv:2502.19023, 2025.
[11] others. Knowledge management in a world of generative AI: Impact and implications.ACM
Transactions on Management Information Systems, 2025. Verify author names against
published ACM version before submission.
[12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei
Sun, Qianyu Guo, Meng Wang, and Haofen Wang. Retrieval-augmented generation for
large language models: A survey.arXiv preprint arXiv:2312.10997, 2024.
[13] Glean Technologies. Glean: AI-powered enterprise search and
knowledge discovery.https://www.glean.com/resources/guides/
glean-ai-enterprise-search-knowledge-discovery, 2024. Product documenta-
tion.
[14] ZiruiGuo, LianghaoShi, ZhenWang, etal. LightRAG:Simpleandfastretrieval-augmented
generation.arXiv preprint arXiv:2410.05779, 2024. Accepted at EMNLP 2025.
[15] Rikui Huang, Wei Wei, Xiaoye Qu, Shengzhe Zhang, Dangyang Chen, and Yu Cheng.
Confidence is not timeless: Modeling temporal validity for rule-based temporal knowl-
edge graph forecasting. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), pages 10783–10794, 2024.
[16] Lucas Jarnac, Yoan Chabot, and Miguel Couceiro. Uncertainty management in the con-
struction of knowledge graphs: a survey.Transactions on Graph Data and Knowledge
(TGDK), 3(1), 2024.
[17] Yishi Jiang et al. A survey on temporal knowledge graph embedding: Models and appli-
cations.Knowledge-Based Systems, 304, 2024.
[18] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming
Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In
Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing,
pages 7969–7992, 2023. arXiv:2305.06983.
13

[19] Quinn Leng, Jacob Portes, Sam Havens, Matei Zaharia, and Michael Carbin. Long context
RAG performance of large language models. InNeurIPS 2024 Workshop on Adaptive
Foundation Models, 2024. arXiv:2411.03538.
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InAdvances in Neural Information Processing Systems,
volume 33, 2020.
[21] Junheng Li et al. Continuous knowledge graph refinement with confidence propagation.
IEEE Transactions on Knowledge and Data Engineering, 2023.
[22] Xinze Li, Yixin Cao, Yubo Ma, and Aixin Sun. Long context vs. RAG for LLMs: An
evaluation and revisits.arXiv preprint arXiv:2501.01880, 2025.
[23] Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. Retrieval
augmented generation or long-context LLMs? a comprehensive study and hybrid approach.
InProceedings of the 2024 Conference on Empirical Methods in Natural Language Process-
ing (Industry Track), 2024. arXiv:2407.16833.
[24] Shichun Liu et al. Memory in the age of AI agents: A survey.arXiv preprint
arXiv:2512.13564, 2025.
[25] MemTensor. MemOS: An operating system for memory-augmented generation.arXiv
preprint arXiv:2505.22101, 2025.
[26] Luc Moreau, Paolo Missier, et al. PROV-DM: The PROV data model.https://www.w3.
org/TR/prov-dm/, 2013. W3C Recommendation.
[27] Ikujiro Nonaka. A dynamic theory of organizational knowledge creation.Organization
Science, 5(1):14–37, 1994.
[28] Notion Labs. The ultimate guide to AI-powered knowl-
edge hubs in notion.https://www.notion.com/help/guides/
ultimate-guide-to-ai-powered-knowledge-hubs-in-notion, 2024. Product doc-
umentation.
[29] Charles Packer, Sarah Wooders, Kevin Lin, et al. MemGPT: Towards LLMs as operating
systems.arXiv preprint arXiv:2310.08560, 2023.
[30] PalantirTechnologies. Palantirontology: Connectingdatatotherealworld, 2023. Platform
Documentation.
[31] Preston Rasmussen et al. Zep: A temporal knowledge graph architecture for agent memory.
arXiv preprint arXiv:2501.13956, 2025.
[32] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H. Chi,
Nathanael Schärli, and Denny Zhou. Large language models can be easily distracted by
irrelevant context. InProceedings of the 40th International Conference on Machine Learn-
ing, volume 202 ofProceedings of Machine Learning Research, pages 31210–31227, 2023.
arXiv:2302.00093.
[33] Eric W. Stein and Vladimir Zwass. Actualizing organizational memory with information
systems.Information Systems Research, 6(2):85–117, 1995.
[34] James P. Walsh and Gerardo Rivera Ungson. Organizational memory.Academy of Man-
agement Review, 16(1):57–91, 1991.
14

[35] Rongwu Xu et al. Knowledge conflicts for LLMs: A survey. InProceedings of the 2024
Conference on Empirical Methods in Natural Language Processing, 2024.
[36] Wujiang Xu, Zujie Liang, et al. A-MEM: Agentic memory for LLM agents.arXiv preprint
arXiv:2502.12110, 2025. NeurIPS 2025.
[37] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. Judging LLM-as-a-judge with MT-
Bench and chatbot arena. InAdvances in Neural Information Processing Systems, vol-
ume 36, 2023. arXiv:2306.05685.
A KOC Axis Specification
The Knowledge Object Coordinate is a 7-axis structured identifier:
[Entity]-[Domain]-[Class]-[Epoch]-[Depth]-
[Author]-[Variant]
Each axis is assigned at ingestion and is immutable thereafter. Structural similarity between
two KOs is computed as the weighted sum of axis-wise matches, normalized to[0,1], atO(1)
cost.
B Edge Type Vocabulary
Table 7: Complete KOEdge vocabulary with signed semantic coefficients.
Type Coeff. Semantics (A→B: A acts on B)
SUPPORTS+1.0A provides evidence strengthening B
BASED_ON+0.8A is the logical grounding of B
IMPLEMENTS+0.7A operationally realizes B
SUPERSEDES+0.6A replaces B—B is demoted, not deleted
REFINES+0.5A narrows B without contradiction
DERIVES_FROM+0.5A follows logically from B
ENABLES+0.4A is a necessary condition for B
PRECEDES+0.3A temporally precedes B
BLOCKS−0.4A actively prevents B
CONTRADICTS−0.6A contradicts B (strongest negative
gravity)
C KGE Force Formulas
Usage force:u i=au·/summationtext
j∈recent exp(−τ j/σ), whereτ jis the time since thej-th retrieval and
σis the recency scale.
Evidence force:e i=ae·|{j: (j,i)∈E,type(j,i) =SUPPORTS}|, counting new inbound
support edges.
Gravity force:g i=ag·/summationtext
j∈N(i)COEFF(j,i)·tanh(g scale·Knorm
j/d(i,j)), whereKnorm
jis
the z-score-normalizedK-score of neighborj:
zj=Kj−µK
max(σ K, σfloor), Knorm
j = max(0, z j)
withµ Kandσ Kcomputed across neighbors ofi, andσ floor= 0.5preventing division instability
for small neighborhoods. The sign is a property of the edge coefficient; the magnitude is a
statistical property of the neighbor’s score.d(i,j)is the hop distance.
15

Contradictionpenalty:c i=ac·|{j: (j,i)∈E,type(j,i)∈{CONTRADICTS,BLOCKS}}|.
QUESTION urgency:urg(t) =clamp(age_days/30·0.3 +B·0.2 +S·0.5,0,1), where
Bis the blocking edge count andSis the stakes multiplier.
D Convergence Proof Sketch
The KGE updateK(t+ 1) =F(K(t))defines a mapF: [0,1]n→[0,1]n. The Jacobian
∂Fi/∂K jforj̸=iarises from the gravity term:|∂F i/∂K j|≤η·a g·|COEFF(j,i)|·g scale/d(i,j)2.
By the Gershgorin circle theorem,Fis a contraction if for every nodei: the diagonal entry
(per-node contraction∈[0.845,0.850]) plus the sum of off-diagonal magnitudes is<1. This
yields the sufficient conditionmax_degree< g scale/max|COEFF|= 5.0/0.7≈7.14. The
bound is conservative: tanh saturates gravity contributions for high-Kneighbors, clamping
provides non-expansive projection, and mixed-sign edges cancel. Empirically, convergence holds
formax_degree= 43.
T2 Crossover Theorem (Theorem 1 — full derivation).Under stationary inputs
withu=e=g=c= 0andK 0= 0.5:
K∗
QUESTION =η·seed
η+λQUESTION·∆t=0.1×0.5
0.1 + (−0.010)×1= 0.556>K 0= 0.5
K∗
OBSERVATION =η·seed
η+λOBSERVATION·∆t=0.1×0.5
0.1 + 0.015×1= 0.435<K 0
SinceK∗
QUESTION> K 0> K∗
OBSERVATION , the two trajectories diverge att= 1(the first
discrete update). The asymptotic gap is0.556−0.435 = 0.121; at day 28 the gap is 0.115 score
units.
E Taxonomy Adequacy
Exhaustive pairwise analysis of all/parenleftbig9
2/parenrightbig= 36class pairs confirms that every pair differs on at least
two of: decay type (none/exponential/inverse), seedKvalue, half-life, and semantic role. The
closest pairs are HYPOTHESIS–OBSERVATION (seed∆ = 0.10, half-life ratio1.8×, semantic
distinction: testable claim vs. passive signal) and DECISION–CONSTRAINT (seed∆ = 0.10,
semantic distinction: revocable choice vs. structural boundary). No merge is possible without
collapsing at least one distinction required by the KGE, retrieval, or contradiction logic.
F Notation
16

Symbol Meaning
K,K∗Importance score (emergent, per KO); fixed-point value
Keff Contextual importance (query-modulated)
ϕctx Contextual attention function: weighted combination of
entity, domain, and anchor alignment (Eq. 5)
ηLearning rate / momentum parameter (default 0.15)
λclass Class-specific decay rate
∆tTime since last KGE cycle (in 6-hour units; default 0.25)
au,ae,ac,ag Scaling constants for usage, evidence, contradiction, and
gravity forces
gscale Gravity scale parameter (default 5.0)
σfloor Z-score floor for gravity normalization (default 0.5)
α,β,γRetrieval weights for structural, semantic, and topological
similarity
we,wd,wa Contextual attention weights for entity, domain, and
anchor alignment
H(q,i)Hybrid similarity score between queryqand KOi
Sstruct,Ssem,StopoStructural, semantic, and topological similarity
components
17