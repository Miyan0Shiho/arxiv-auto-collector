# TTFT-Aware Graph Chain-of-Thought:Distance-Indexed Neural A* for Low-Hallucination Multi-Hop Medical Reasoning

**Authors**: Bechir Dardouri, Kaïs Zhioua, Yassine Msaddak

**Published**: 2026-06-22 09:51:28

**PDF URL**: [https://arxiv.org/pdf/2606.23108v1](https://arxiv.org/pdf/2606.23108v1)

## Abstract
Hallucinations and opaque reasoning remain unacceptable failure modes for clinical LLMs. We present a production-grade GraphRAG stack that constrains answers to verifiable graph chain-of-thought paths in a heterogeneous, ~700K-node medical knowledge graph powering a fertility assistant. The core idea is targeted navigation: a directed Pruned Landmark Labeling (PLL) oracle provides exact distances for sub-millisecond feasibility checks and simple-path enumeration, while a lightweight AStarNet heuristic operates strictly within the PLL corridor to prioritize clinically plausible expansions. We score and pack a small, diverse set of paths (CUI/semantic-type overlap, length prior, provenance priors) to condition generation, yielding compact prompts and improved Time to First Token (TTFT). On fertility-focused queries, the hybrid (PLL+AStarNet) establishes a better latency/recall Pareto frontier than text-only RAG and single-component baselines, lowers TTFT, and reduces clinician-audited hallucinations while preserving explanation clarity. The result is a practical recipe for explainable, low-hallucination multi-hop medical reasoning ready for real-world deployment.

## Full Text


<!-- PDF content starts -->

TTFT-Aware Graph Chain-of-Thought:
Distance-Indexed Neural A* for
Low-Hallucination Multi-Hop Medical Reasoning
Bechir Dardouri, Kaïs Zhioua, and Yassine Msaddak
Tanit Healthcare Technologies, Tunisia
kais.zhioua@tanit.ai
Abstract.Hallucinations and opaque reasoning remain unacceptable
failure modes for clinical LLMs. We present a production-grade
GraphRAG stack that constrains answers to verifiablegraph chain-of-
thoughtpaths in a heterogeneous,∼700K-node medical knowledge graph
powering a fertility assistant. The core idea istargeted navigation: a
directed Pruned Landmark Labeling(PLL) oracle provides exact
distances for sub-millisecond feasibility checks and simple-path enumer-
ation, while a lightweightAStarNetheuristic operates strictlywithin
the PLL corridor to prioritize clinically plausible expansions. We score
and pack a small, diverse set of paths (CUI/semantic-type overlap,
length prior, provenance priors) to condition generation, yielding com-
pact prompts and improvedTime to First Token(TTFT). On fertility-
focused queries, the hybrid (PLL+AStarNet) establishes a better la-
tency/recall Pareto frontier than text-only RAG and single-component
baselines, lowers TTFT, and reduces clinician-audited hallucinations
while preserving explanation clarity. The result is a practical recipe for
explainable,low-hallucinationmulti-hopmedicalreasoningreadyforreal-
world deployment.
Keywords:GraphRAG·Knowledge Graph·Heterogeneous Graphs·
Medical QA·Explainable AI·Pruned Landmark Labeling·A* Search
·AStarNet·TTFT
1 Introduction
Large language models (LLMs) can draft clinically fluent guidance, yet un-
groundedgenerationexposespatientsandclinicianstohallucinationsandopaque
reasoning. In safety-critical settings, answers must be (i)fact-boundedby verifi-
able sources, (ii)auditableend-to-end, and (iii)fastto first token for interactive
use. We take a structure-first stance: constrain every non-trivial claim to ex-
plicit, typed relations in a heterogeneous directed medical knowledge graph
(KG), yielding a verifiablegraph chain-of-thoughtrather than an uninspectable
latent trace.arXiv:2606.23108v1  [cs.AI]  22 Jun 2026

2 B. Dardouri et al.
Why this is hard.The core systems obstacle ismulti-hop retrieval at scale.
Beyond four hops, naïve (bi-)BFS over large typed graphs explodes in candi-
date paths, inflating memory and tail latency; even classicalk-path algorithms
struggle when looplessness, type constraints, and near-tie neighborhoods collide.
Meanwhile, user-perceived responsiveness is dominated byTime to First Token
(TTFT), making both compute discipline and prompt compactness first-class
constraints [19,20,22]. Text-only RAG underperforms on corpus-level and multi-
hopquestions;graph-structuredretrieval(GraphRAG)improvesauditabilitybut
still requires efficient, targeted navigation of heterogeneous graphs [8].
Our idea.We convert blind enumeration intotargeted navigationby pairing
anexact distance oraclewith alearned heuristic. First, we adaptdirected
Pruned Landmark Labeling(PLL) to answer shortest-path distances via label
intersections, enabling sub-millisecond feasibility checks and meet-in-the-middle
enumeration ofsimplepaths [2,3]. Second, we serve a compact, GPU-batched
AStarNetpriority strictlywithinthe PLL corridor to focus expansions on clini-
callyplausiblecontinuations[23].EntitiesaregroundedviaclinicalNER+linking
to UMLS [5] usingscispaCy[17,10] and, where applicable, MedCAT [15]. A
lightweight scorer (CUI/semantic-type overlap, length prior, provenance priors)
thenselectsandpacksadiversetop-kofevidencepathstokeeppromptscompact
and TTFT low while preserving auditability.
Contributions.
1.Distance-indexed retrieval.Directed PLL as an exact feasibility oracle
for distance-bounded simple-path search on large medical KGs [2,3].
2.Neural guidance inside exact bounds.AStarNet priorities operate
within the PLL corridor, cutting expansions without sacrificing correctness
[23].
3.Production-gradegrounding.TTFT-awareevidencepackingandmanda-
tory path citation reduce hallucinations and severe errors in clinician audits
[4].
This recipe is deployed in a fertility assistant backed by a∼700K-node,
bimonthly-updated heterogeneous KG; on fertility queries it achieves a supe-
rior latency/recall Pareto frontier, lower TTFT, and fewer hallucinations than
text-only RAG and single-component baselines.
2 Background & Problem
2.1 Problem definition
Setup.We operate on a large, heterogeneous medical knowledge graph (KG)
G= (V, E,R)with directed, typed edgese= (ur− →v), wherer∈ Rand
|V| ≈7×105. User queries are mapped to seed entitiesS⊆Vvia clinical
NER+linkingtoUMLS(e.g.,withscispaCy/MedCAT)[5,17,10,15].Forordered

Distance-Indexed Neural A* for GraphRAG 3
pairs(s, t)∈S×S, we must retrievesimple(loopless) evidence pathsp:s⇝t
that respect relation/type constraints and serve them as auditable conditioning
for generation.
Assumptions and constraints.We assume a vetted, multi-relational KG with
explicit provenance (e.g.,PrimeKG) [6], periodic snapshots, and a streaming
LLM that must cite graph evidence. Retrieval is bounded by strict production
budgets: hop capH, at mostkreturned paths, per-pair wall time≤T max, mem-
ory≤M max, and prompt token limits that driveTime to First Token(TTFT)
[19,20,22].
Research questions.RQ1:How can we retrieve asmall,diverse, andauditable
set of multi-hop simple paths with high recall under tight latency/memory bud-
gets?RQ2:Can we combine exact feasibility guarantees with learned guidance
so we explore far fewer candidates without sacrificing correctness?RQ3:How
should evidence bepackedto minimize prefill cost (hence TTFT) while preserv-
ing clinical auditability?
Why this is challenging.(i)Combinatorics: simple-path enumeration beyond
3–4 hops explodes on typed directed graphs; naive BFS creates minute-scale
tails [21,9,13]. (ii)Heterogeneity: type constraints and high-degree hubs in-
duce brittle branching factors and near-tie neighborhoods. (iii)Latency: TTFT
is dominated upstream by retrieval fan-out and prompt bulk [19,22]. (iv)Au-
ditability: evidence must be explicit, provenance-carrying, and cited per claim
[4].
2.2 Existing approaches and their limitations
We group prior solutions by coreintuition.
A. Text-centric RAG.Index unstructured corpora and condition the LLM on
retrieved passages. Strengths: broad coverage and simplicity. Limitations: weak
structural grounding for multi-hop reasoning, longer prompts, and higher hallu-
cination risk; audit trails are indirect. Graph-structured retrieval (GraphRAG)
improves inspectability by retrieving entities/communities/paths but still needs
efficient multi-hop navigation at scale [8].
B.Brute-force orboundedgraph search.BFS/bi-BFSwithtypefilters;k-shortest(-
simple) methods such as Yen/Eppstein [21,9,13]. Principled but prone to mem-
ory and latency blow-ups on heterogeneous KGs; poor TTFT under realistic
budgets.
B′. Point-to-point heuristic search.A* [12], ALT landmarks [11], and Hub La-
beling [1] acceleratedistancequeries on road/web graphs. Strength: strong P2P
latency. Limitations: usually optimizeoneshortest path, not adiverse top-k
simpleset; do not address prompt or TTFT constraints.

4 B. Dardouri et al.
C. Label oracles and neural reasoning.PLL answers exact shortest-path dis-
tances via compact labels and fast intersections, with sub-ms queries at million
scale [2,3]; its limitation is that it returns distances, not paths. Learned policies
(e.g., A*Net) prioritize local moves to reduce expansions [23], but without hard
feasibility bounds they can drift; most focus on link prediction rather than a
small, diverse, TTFT-aware auditable path set.
Gap.No prior category simultaneously provides: (i)exactfeasibility checks to
tightly bound search; (ii)neuralguidanceinsidethat bound; (iii) TTFT-aware
evidence packing with mandatory citation.
2.3 Our perspective
Main insight.Bound what ispossiblewith anexact, directedPLL distance
oracle, then focus on what ispromisingusing alightweight learned priority
(A*Net)restricted to the PLL corridor. Finally,packa small, diverse set of
CUI–relationtuplesandrequirepathcitationatdecodetime.Thisconvertsblind
enumeration intotargeted navigationthat respects TTFT and safety constraints
while preserving recall and auditability.
Operational objective.Formally, for each(s, t)letD=d(s, t)from directed PLL.
Enumerate only prefixess→···→xsatisfying
g(x) +d(x, t)≤Dwith node uniqueness and type masks,(1)
rank feasible frontiers via a GPU-batched learned priority, select a diverse top-k
under fixed time/memory/prompt budgets, and generate with mandatory path
citation. §3 details this design.
3 Method: Distance-Indexed, Neural-Guided Retrieval
3.1 Scope, assumptions, and non-goals
Scope.Given seed entitiesSfrom a heterogeneous, directed medical KGG=
(V, E,R), we aim to retrieve, under strict latency/memory/prompt budgets, a
small, diverse, auditableset ofsimple(loopless) multi-hop paths connecting or-
dered pairs(s, t)∈S×S. The returned paths must (i) obey type/relation con-
straints (clinical plausibility), (ii) carry provenance, and (iii) be serialized for
prompt-efficient conditioning of a streaming LLM thatmustcite paths per claim.
Assumptions.We assume a vetted, multi-relational KG with periodic snapshots
and explicit provenance; clinical NER with UMLS linking for seed generation;
andstrictproductionbudgets(H, k, T max, Mmax)togetherwithTTFTsensitivity
[19,20,22].

Distance-Indexed Neural A* for GraphRAG 5
Clinical NER
& Linking
(UMLS)Directed PLL
Distance Oracle
d(s, t)Feasible Corridor
g(x)+d(x, t)≤D
Simple-path &
Type masksA*Net Priority
GPU-batched
Top-ρkeepScore & Diversify
Eq. (3)
MMRID-centric Packing
Name MapLLM Decode
& ValidatorSeedsS D=d(s, t) Feasible expansions Top-kpaths Compact evidenceExact feasibilityNeural guidance TTFT-aware evidence
Fig. 1.End-to-end pipeline: PLL bounds search to the shortest-distance corridor;
A*Net focuses expansions within that corridor; scoring/diversity/packing yield com-
pact, auditable evidence for citation-enforced decoding.
Non-goals.We do not solve open-domain text retrieval; we do not learn the
KG; and we do not modify the generator architecture beyondcitation-enforcing
decoding.
3.2 Design overview and rationale
We convert blind enumeration intotargeted navigation, combining anexact dis-
tance oracle(directed PLL) with alearned heuristic(A*Net), and finishing
withTTFT-awarepackingpluscitation-enforcingdecoding.Figure1sum-
marizes the end-to-end flow.
High-level loop.(1) Ground the prompt to seed CUIs. (2) For each(s, t), query
D=d(s, t)from adirectedPLL service. (3) Enumerate only PLL-feasible pre-
fixess→···→xsatisfying Eq. (2). (4) Inside that corridor, prioritize feasible
expansions with GPU-batchedA*Net. (5) Score,diversify, andpacka top-kof
paths. (6) Generate withmandatorypath citation; a validator rejects out-of-
evidence claims [4].
Why these components (insight). Exactnessfrom PLL tightly bounds what is
possibleat sub-ms cost [2,3];learned prioritiesfocus compute on what ispromis-
ing[23]. This pairing collapses fan-out without sacrificing correctness, while com-
pact packing reduces prefill tokens and TTFT.
3.3 Component I: Entity grounding (seeds)
Procedure.Run clinical NER+linking (e.g.,scispaCy, MedCAT) to map men-
tions to UMLS CUIs; de-duplicate, apply type allow-lists and confidence floors.
Justification.Accurate seeds bound the search space and reduce spurious fan-
out; CUI normalization yields clean joins across heterogeneous edges [5,17,15].
3.4 Component II: Exact feasibility via directed PLL
Query.For each nodev, store out-labelsLout(v)and in-labelsLin(v)(onG⊤).
The exact directed distance
D= min
w∈Lout(s)∩Lin(t) 
dout(s, w) +din(w, t)
is computed inO(|Lout(s)|+|Lin(t)|)and is sub-ms in practice [2].

6 B. Dardouri et al.
Build & serve.Construct labels by pruned BFS/SSSP in a pivot order; when a
visit is covered by existing labels, prune; build onGandG⊤; serve labels from
RAM behind a low-latency RPC; snapshot rebuilds [2,3,14].
Why PLL.It certifies feasibilityexactlyat negligible latency, defining a tight
corridor for path search—something heuristic-only methods cannot guarantee.
3.5 Component III: Distance-disciplined enumeration
Constraint.Keep only prefixess→···→xsatisfying
g(x) +d(x, t)≤D,(2)
with node uniqueness (simple paths) and type/relation masks. Optionally run
meet-in-the-middle: symmetric feasibility fromtand join frontiers.
Why corridor search.Eq. (2) collapses fan-out to a predictable set while preserv-
ing all shortest-distance completions; type masks reduce low-value expansions.
3.6 Component IV: Neural guidance inside the PLL corridor
Priority.Learnh θ(x|t,C)(context: prefix length, last relation, semantic types).
Rank feasible candidates by
A*:f(x) =g(x) +λh θ(x|t,C)or Beam:s(x) =λh θ(x|t,C)−µg(x),
and expand under budgets (beamB, hop capH, time/memory caps). This
follows A* [12] but remainsstrictlyinside the exact PLL corridor.
Training.Shallow MLP over entity/relation embeddings, semantic-type indica-
tors, prefix features, ande(t); train on KG triples with uniform+degree-aware
negatives (logistic pairwise loss);upweightedges on short PLL-feasible prefixes
to correlate withpathutility; temperature-calibrate. Serve with dynamic micro-
batching and keep-ratioρ(retain top-ρfeasible per step) [23,7].
Why learned guidance.Within the exact corridor, admissibility is unnecessary;
the heuristic simply prioritizes promising moves, dramatically reducing expan-
sions with no loss of correctness.
3.7 Component V: Path scoring, diversity, and packing
Utility.Forp=v 0r0− →···rℓ−1− − − →v ℓ:
Score(p) =αCUIOverlap(p) +βSemTypeOverlap(p) +γ1
1+ℓ(p)
+δEdgeReliability(p)−ηHubPenalty(p).(3)
Tune(α, β, γ, δ, η)on validation for path nDCG and clinician faithfulness.

Distance-Indexed Neural A* for GraphRAG 7
Diversity.SelectP kvia MMR:max pλScore(p)−(1−λ) max q∈PSim(p, q), with
Jaccard similarity over CUIs/relations.
Packing (TTFT-aware).Serialize as compact CUI–relation tuples with a single
Name Mapper prompt to minimize tokens; decodingrequiresciting path IDs;
a validator rejects out-of-evidence claims or missing citations and can trigger
abstention [4].
Why this head.Clinically shaped priors (+ hub penalty) prefer concise, plausi-
ble chains; MMR avoids near-duplicates; ID-centric packing reduces prefill and
TTFT.
3.8 Complexity and systems considerations
Asymptotics.PLL queryO(|Lout(s)|+|Lin(t)|); enumeration is bounded by the
feasible frontier induced by Eq. (2); A*Net adds negligible GPU latency due to
microbatching and top-ρfiltering.
Caching and fallbacks.LRU cache hot(s, t)distance queries; degrade to PLL-
only when GPU is cold/saturated; capBandρunder bursty load; enforce per-
pairT maxto protect p95.
Alternatives considered (why not). ALT/Hub Labeling only[11,1]: great fordis-
tances, but do not curate an auditable top-kof simple paths or address TTFT.
k-shortest(-simple) paths only[21,9,13]: hard to keep loopless/type-constrained
at scale; poor tails.Heuristic-only search: no exact feasibility, unstable tails, and
recall loss.
3.9 Operational algorithm (concise pseudo-code)
Input:seed setS, hop capH, budgets(k, T max, Mmax)
For eachordered(s, t)∈S×S:
•QueryD←d(s, t)from directed PLL.
•Initialize frontier withs; enforce node-uniqueness and type masks.
•While time/memory budget remains:
– Keep only feasible nodes:g(x) +d(x, t)≤D.
– Rank feasible nodes viaf(x)ors(x); expand top-ρunder beamB.
– Whentis reached withℓ(p)≤H, addpto candidate set.
•Score candidates by Eq. (3); selectP kvia MMR; pack as tuples with a single
Name Map.
Output:compact, diverseP kwith path IDs for citation-enforcing decoding.
4 Experimental Setup
We evaluate under production-like conditions for a fertility assistant.

8 B. Dardouri et al.
Graph & queries.KG:Directed, typed medical graph (∼700K nodes) span-
ning drug–disease, phenotype, procedure, biomarker, and guideline edges;bi-
monthlysnapshots.Seeds:Prompts mapped to UMLS CUIs via NER+linking
withscispaCy/MedCAT [5,17,15].Splits:De-identified live prompts and clini-
cian scenarios (etiology→diagnostic→intervention→outcome), stratified by
PLL distance; 20% val, 20% test.
Systems & hardware.Storage:Neo4j cluster [16].Distance oracle:Directed
PLL service (labels in RAM), blue/green rebuild per snapshot, LRU hot-pair
cache [2,14].Heuristic:AStarNet GPU microservice with dynamic microbatch-
ing and warm pools [23].Generation:Streaming LLM conditioned strictly on
selected path IDs.Compute:DGX-class system with multiple A100 GPUs [18].
Methods.We compare six configurations under identicalH, time/memory,
and evidence-token budgets:Text RAG(dense retrieval, no graph);BFS
(depth-H, type-filtered);Bi-BFS(meet-in-the-middle);PLL-only(distance-
bounded enumeration);AStarNet-only(A*/beam, no PLL corridor);Ours
(PLL+AStarNet). Figs. 2–3 and Table 2 cover the three strongest baselines;
Fig. 4 adds BFS/Bi-BFS to isolate unguided traversal cost.
Metrics & protocol.Retrieval:recall@kvs. a high-budget oracle; path nDCG;
cumulativeexpansions;%PLL-feasible.Latency:end-to-endp50/p95andTTFT
[19,20,22].Memory:peak RSS (enumerator) and peak GPU (heuristic).Faith-
fulness:clinician audit—hallucination %, severe error %, explanation clarity
(1–5); automatic citation-fidelity check [4].Stats:95% CIs via bootstrap; paired
Wilcoxon for ours vs. baselines.
5 Results
We report end-to-end performance under identical budgets forBFS,Bi-BFS,
PLL-only,AStarNet-only, andOurs(PLL+AStarNet). Metrics follow §4:
recall@k,p50/p95latency,TTFT,cumulativeexpansions,memory,andclinician-
audited faithfulness. All results are on the held-out test split; CIs (95%) are from
bootstrap over prompts.
5.1 Main Comparison: Pareto & TTFT
Latency/recall frontier.Figure 2 shows recall@kvs. p95 latency fork∈ {3,5}.
The hybrid establishes a new frontier: atk=3it attains0.74recall with1.18s
p95latency,outperformingTextRAG(0.58,2.51s),AStarNet-only(0.63,1.74s),
and PLL-only (0.68, 1.49s). At iso-recall 0.68, the hybrid reduces p95 latency
by21.0%vs. PLL-only and32.2%vs. AStarNet-only.

Distance-Indexed Neural A* for GraphRAG 9
TTFT distributions.Figure 3 plots TTFT violins. Median TTFT drops from
980ms(Text RAG) and610ms(PLL-only) to420ms(Ours), with tail (p95)
shrinking from 1.92s to 0.93s. Two factors drive the shift: (i) sub-msdistance
checks gate enumeration early; (ii) ID-centric path packing cuts prefill tokens.
Variance also narrows, yielding more predictable interactivity.
Cumulative expansions & memory.Figure 4 shows an order-of-magnitude re-
duction in explored nodes/edges vs. BFS/Bi-BFS. Peak RSS of the enumerator
drops by2.4×vs. Bi-BFS and1.5×vs. AStarNet-only; GPU memory for the
heuristic service remains under 6GB at QPS≤5 due to microbatching.
5.2 Ablations
Rationale.To attribute gains to specific design choices under identical budgets,
we vary one knob at a time while freezing the others at their validated defaults
(community-awarePLL,B=32,ρ=0.2,H=6) with a p95 latency target of 1.25s.
Thethreeablationsprobeorthogonalcontributors:(i)index tightness(PLLpivot
order) for feasibility pruning; (ii)guidance aggressiveness(beamB, keep-ratio
ρ) for expansion economy vs. recall; and (iii)evidence reach(hop capH) for
near-long-range reasoning under bounded tails.
PLL pivot order.Table 1 summarizes label statistics at 700K nodes. Degree-first
orderingshrinkslabelsizeandbuildtimevs.random;community-awareordering
performs best:480MBlabels,1.6hbuild,0.51msp50 query,95.2%pruned.
Tighter construction pruning correlates with fewer futile runtime expansions,
improving TTFT stability without recall loss.
Beam widthBand keep-ratioρ.We sweepB∈ {16,32,64,128}andρ∈
{0.1,0.2,0.3}inside the PLL corridor. The best operating point under the
1.25s p95 target isB=32,ρ=0.2, balancing recall and latency. Very smallρ
under-explores rare relations—recall falls at fixed TTFT; very largeByields
diminishing recall gains but steeper tails. Annealingρwith depth (0.3→0.15)
gains +0.7pp recall@3 at unchanged latency.
Hop capH.RaisingHfrom 5 to 7 increases recall@5by+3.4pp with a mild
p95 cost (+90ms) thanks to the PLL feasibility envelope, which suppresses infea-
sible branches even as the search radius grows. In contrast, BFS/Bi-BFS exhibit
superlinear tail growth beyond 4 hops, underscoring the necessity of distance-
indexing for interactive use.
5.3 Faithfulness and Explanation Quality
Table 2 reports clinician-audited outcomes. The hybrid reduces hallucinations
to6.3%(vs. 22.7% Text RAG; 10.4% PLL-only; 13.1% AStarNet-only), severe
errors to1.1%, and improves explanation clarity to4.4/5. Auditors favored
concise answers citing1–3distinct paths per claim; our diversity-aware selection
avoided near-duplicate chains without hurting TTFT.

10 B. Dardouri et al.
5.4 Qualitative Case Studies
In representative fertility scenarios, the system surfaces compact, clinically plau-
sible chains, e.g.,etiology→biomarker→intervention→outcome, while down-
weighting hub-dominated detours. Evidence is cited inline (path IDs), enabling
rapid provenance checks and targeted corrections when KG gaps are encoun-
tered.
Takeaways.(i)Distance feasibility is indispensable: directed PLL converts
unbounded fan-out into a tight corridor with predictable tails. (ii)Neural guid-
ance pays off when bounded: AStarNet inside the corridor cuts expansions
and improves TTFT without sacrificing correctness. (iii)Faithfulness follows
structure: mandatory citation of explicit paths lowers hallucinations and clar-
ifies explanations. (iv)TTFT tracks prompt discipline: ID-centric packing
reduces prefill, compressing both median and tail TTFT.
Fig. 2.Latency–recall frontier (lower-right is better). Hybrid (PLL+AStarNet) im-
proves the Pareto frontier relative to Text RAG, AStarNet-only, and PLL-only at
k∈{3,5}.
Table 1.PLL index statistics by pivot order (700K-node directed KG).
Order Label size (MB) Build time (h) Query p50 (ms) %Pruned
Random 780 2.7 0.78 88.1
Degree-first 520 1.9 0.54 93.7
Community-aware480 1.6 0.51 95.2

Distance-Indexed Neural A* for GraphRAG 11
Fig. 3.TTFT distributions across methods (lower is better). The hybrid lowers median
TTFT and tightens the tail (p95) vs. all baselines.
Fig. 4.Cumulative expansions vs. time (lower is better). The hybrid explores far
fewer nodes/edges than BFS/Bi-BFS and AStarNet-only, aligning with observed la-
tency gains.
Table 2.Clinician-audited faithfulness on fertility prompts. Lower is better for error
rates; higher is better for explanation clarity.
Method Hallucination (%) Severe error (%) Expl. clarity (1–5)
Text RAG 22.7 6.8 2.8
PLL-only 10.4 2.4 3.9
AStarNet-only 13.1 3.5 3.6
Ours (PLL+AStarNet)6.3 1.1 4.4

12 B. Dardouri et al.
6 Discussion, Limitations and Ethics
Discussion
Why the pairing works.Directed PLL provides an exact, sub-ms feasibility
test that collapses fan-out into a tight corridor; AStarNet prioritizes expansions
withinthat corridor, cutting visits without sacrificing correctness. A compact
scorer selects a small, diverse path set that anchors generation to facts and
improves TTFT.
Engineering lessons.(i)TTFT is upstream: ID-centric packing and early
distance gating help more than decoder tweaks. (ii)Predictable tails: enforce fea-
sibility at every step to control p95. (iii)Prefer simple paths: node-uniqueness
avoids hub-cycling and eases audit. (iv)Graceful degradation: on GPU pressure,
PLL-only keeps plausibility and stable TTFT.Portability:the recipe general-
izes to any snapshot-able, typed KG with modest GPU capacity.
Limitations
– Shortest-distance bias:Focusing onDcan miss slightly longer but salient
chains; consider tightε-relaxation or boundedk-shortest-simple paths.
– Label memory/rebuilds:Hundreds of MB at 700K nodes; snapshot re-
builds are simple but static—dynamic/historical PLL adds operational com-
plexity [3].
– Heuristicbias:AStarNetmayover-preferfrequentrelationsorhubs;degree-
aware negatives, hub penalties, and a PLL-only fallback reduce but do not
eliminate this risk [23].
– Entity-linking brittleness:NER/linking errors propagate to seed re-
trieval; type filters and confidence floors help, but ambiguity persists.
– Loadsensitivity:Burstytrafficcanqueuemicrobatches;wecapbeam/keep-
ratio and shed gracefully to PLL-only.
– KG coverage/provenance:Gaps or conflicts remain; source-weighted pri-
ors and abstention guard against insufficient evidence.
Ethics
Safety posture.Clinicalinformationtool, not diagnostic; every non-trivial
claim must cite explicit paths (graph chain-of-thought); a validator blocks out-
of-evidence content; abstain and escalate when paths are missing or contradic-
tory [4].Privacy/governance.Minimize PHI; de-identify where applicable;
encrypt in transit and at rest; enforce access controls; audit path-level prove-
nance with bounded retention.Fairness.Monitor by subpopulation; mitigate
via KG source audits, relation priors, diversity-aware selection, and clinician
review.Transparency.Node–edge citations enable provenance checks and er-
ror analysis; model and index updates use change control and A/B safeguards
with clear scope and abstention messaging.Future safeguards.Calibrated un-
certainty for expansion,ε-relaxed feasibility under strict caps, and incremental
labeling to reduce snapshot staleness [2,3].

Distance-Indexed Neural A* for GraphRAG 13
7 Conclusion
Adistance-indexed, neural-guidedretriever—directed PLL plus AStarNet—
converts combinatorial multi-hop search into bounded, targeted navigation and
emits a compact, auditablegraph chain-of-thought. On a∼700K-node fertility
KG, the hybrid improves the latency/recall Pareto frontier, reduces hallucina-
tions, and tightens TTFT tails. The recipe is modular—PLL can be swapped
for ALT [11] or Hub-Labeling [1]—and deploys on commodity graph stores [16].
Future work.Near-shortest evidence (tightε-relaxation / boundedk-shortest-
simple[21,9,13]); incremental/dynamic labeling [3]; graph-aware decoding and
calibrated validators.
Acknowledgments
We gratefully acknowledge theAI Garageprogram for access to an NVIDIA
DGX system with multiple A100 GPUs, which enabled the training and large-
scale experiments reported here [18]. We also thank our clinical collaborators for
carefulauditsoffaithfulnessandsafety,andtheengineeringteamformaintaining
the production GraphRAG stack.
References
1. Abraham, I., Delling, D., Goldberg, A.V., Werneck, R.F.: Hierarchical hub
labelings for shortest paths. In: Proceedings of the 20th Annual European
Symposium on Algorithms (ESA). pp. 24–35 (2012). https://doi.org/10.1007/
978-3-642-33090-2_3
2. Akiba, T., Iwata, Y., Yoshida, Y.: Fast exact shortest-path distance queries on
large networks by pruned landmark labeling. In: Proceedings of the 2013 ACM
SIGMOD International Conference on Management of Data. pp. 349–360 (2013).
https://doi.org/10.1145/2463676.2465312, https://arxiv.org/abs/1304.4661
3. Akiba, T., Iwata, Y., Yoshida, Y.: Dynamic and historical shortest-path distance
queries on large evolving networks by pruned landmark labeling. In: Companion
Proceedings of the 23rd International Conference on World Wide Web (WWW
Companion). pp. 237–238 (2014). https://doi.org/10.1145/2567948.2579222
4. Asgari, E., Montaña-Brown, N., Dubois, M., Khalil, S., Balloch, J., Au Ye-
ung, J., Pimenta, D.: A framework to assess clinical safety and hallucination
rates of LLMs for medical text summarisation. npj Digital Medicine8(274), 1–
15 (2025). https://doi.org/10.1038/s41746-025-01670-7, https://www.nature.com/
articles/s41746-025-01670-7
5. Bodenreider, O.: The unified medical language system (UMLS): Integrating
biomedical terminology. Nucleic Acids Research32(suppl_1), D267–D270 (2004).
https://doi.org/10.1093/nar/gkh061
6. Chandak, P., Huang, K., Zitnik, M.: Building a knowledge graph to enable pre-
cision medicine. Scientific Data10(67), 1–16 (2023). https://doi.org/10.1038/
s41597-023-01960-3, https://www.nature.com/articles/s41597-023-01960-3

14 B. Dardouri et al.
7. DeepGraphLearning: AStarNet: Official implementation of a* networks. GitHub
repository (2024), https://github.com/DeepGraphLearning/AStarNet
8. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Larson,
J.: From local to global: A graph RAG approach to query-focused summarization.
arXiv preprint (2024), https://arxiv.org/abs/2404.16130
9. Eppstein, D.: Finding the k shortest paths. SIAM Journal on Computing28(2),
652–673 (1999). https://doi.org/10.1137/S0097539795290477
10. ExplosionAI:ScispaCyEntityLinkerdocumentation.Documentation(2024),https:
//scispacy.readthedocs.io/en/latest/linking.html
11. Goldberg, A.V., Harrelson, C.: Computing the shortest path: A* search meets
graph theory. In: Proceedings of the 16th Annual ACM–SIAM Symposium on
DiscreteAlgorithms(SODA).pp.156–165(2005),https://dl.acm.org/doi/10.5555/
1070432.1070455
12. Hart, P.E., Nilsson, N.J., Raphael, B.: A formal basis for the heuristic determina-
tion of minimum cost paths. IEEE Transactions on Systems Science and Cybernet-
ics4(2), 100–107 (1968). https://doi.org/10.1109/TSSC.1968.300136
13. Hershberger, J., Suri, S.: Vickrey prices and shortest paths: What is an edge worth?
In: Proceedings of the 44th Annual IEEE Symposium on Foundations of Computer
Science (FOCS). pp. 252–259 (2003). https://doi.org/10.1109/SFCS.2003.1238196
14. Iwata, Y.: Pruned landmark labeling (PLL): Reference implementation. GitHub
repository (2013), https://github.com/iwiwi/pruned-landmark-labeling
15. Kraljevic, Z., Searle, T., Shek, A., Roguski, L., Noor, K., Bean, D., Mascio, A.,
Zhu, L., Folarin, A.A., Roberts, A., Bendayan, R., Richardson, M.P., Stewart,
R., Shah, A.D., Wong, W.K., Ibrahim, Z., Teo, J.T., Dobson, R.J.B.: Multi-
domain clinical natural language processing with MedCAT: The medical con-
cept annotation toolkit. Artificial Intelligence in Medicine117, 102083 (2021).
https://doi.org/10.1016/j.artmed.2021.102083
16. Neo4j, Inc.: Neo4j graph database documentation. Product documentation (2025),
https://neo4j.com/docs/
17. Neumann, M., King, D., Beltagy, I., Ammar, W.: ScispaCy: Fast and robust mod-
els for biomedical natural language processing. In: Proceedings of the 18th BioNLP
Workshop and Shared Task. pp. 319–327 (2019). https://doi.org/10.18653/v1/
W19-5034, https://aclanthology.org/W19-5034/
18. NVIDIA: NVIDIA DGX A100 system: User guide. Product documentation (2020),
https://docs.nvidia.com/dgx/pdf/dgx-a100-user-guide.pdf
19. NVIDIA: LLM inference benchmarking: Fundamental concepts (time to first token
and related metrics). NVIDIA NIM Benchmarking Documentation (2024), https:
//docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html
20. NVIDIA: LLM inference benchmarking: Parameters and best practices.
NVIDIA NIM Benchmarking Documentation (2024), https://docs.nvidia.com/
nim/benchmarking/llm/latest/parameters.html
21. Yen, J.Y.: Finding the k shortest loopless paths in a network. Management Science
17(11), 712–716 (1971). https://doi.org/10.1287/mnsc.17.11.712
22. Zaharia,M.,Lee,J.,Wendell,P.,Chien,C.,Graves,P.:Timetofirsttoken(TTFT):
What it is and how to improve it. Databricks Blog (2023), https://www.databricks.
com/blog/time-first-token-ttft-what-it-and-how-improve-it
23. Zhu, Z., Yuan, X., Galkin, M., Xhonneux, S., Zhang, M., Gazeau, M., Tang,
J.: A*net: A scalable path-based reasoning approach for knowledge graphs. In:
Advances in Neural Information Processing Systems (NeurIPS). vol. 36 (2023),
https://arxiv.org/abs/2206.04798