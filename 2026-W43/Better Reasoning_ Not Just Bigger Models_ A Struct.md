# Better Reasoning, Not Just Bigger Models: A Structure-Grounded Framework for Efficient and Reliable Inference

**Authors**: Amirhossein Aghaei, Liangyan Li, Shiyi Yang, Shaohua Wang, Yan Wang

**Published**: 2026-10-22 02:20:21

**PDF URL**: [https://arxiv.org/pdf/2610.30611v1](https://arxiv.org/pdf/2610.30611v1)

## Abstract
Scaling large language models has substantially improved linguistic fluency, yet reliable autonomous agents require more than fluent generation. In multi-turn inference, current agentic systems often produce semantically plausible but structurally ungrounded reasoning, especially under incomplete observations. We propose Structure-Grounded Uncertainty Reduction (SGUR), a training-free framework that complements token-level generation with active uncertainty reduction over structured hypothesis spaces. SGUR represents incomplete observations as a query graph and aligns them with candidate hypothesis graphs through a prior-weighted partial graph matching mechanism, where database-derived empirical priors assign greater importance to decision-relevant relations rather than treating all graph edges uniformly. The resulting inference state is iteratively refined by an active information-gain strategy that selects the next most informative node to query, systematically pruning the candidate space. We instantiate SGUR on a challenging PrimeKG-derived sequential medical diagnosis benchmark, demonstrating that structural grounding can partially substitute for model scale: starting from only two observed symptoms, SGUR achieves 92.5% Hit@3, outperforming the strongest Graph-RAG baseline by 9.4 percentage points while using approximately 5x fewer tokens. Notably, SGUR enables smaller backbone models to match or exceed the performance of models up to 10x larger, suggesting that structural grounding provides a more robust and efficient paradigm for scaling agent performance than relying exclusively on increasing model size.

## Full Text


<!-- PDF content starts -->

Better Reasoning, Not Just Bigger Models: A
Structure-Grounded Framework for Efficient and
Reliable Inference
Amirhossein Aghaei∗
Huawei Research Canada
Markham, Canada
amirhossein.aghaei@h-partners.comLiangyan Li∗
Huawei Research Canada
Waterloo, Canada
lillian.li2@h-partners.com
Shiyi Yang
Huawei Research Canada
Waterloo, Canada
shiyi.yang1@huawei.com
Shaohua Wang
Huawei Technologies Co., Ltd.
Beijing, China
wangshaohua25@huawei.comYan Wang
Central University of Finance and Economics
Beijing, China
dayanking@gamil.com
Abstract
Scaling large language models has substantially improved linguistic fluency, yet
reliable autonomous agents require more than fluent generation. In multi-turn infer-
ence, current agentic systems often produce semantically plausible but structurally
ungrounded reasoning, especially under incomplete observations. We propose
Structure-Grounded Uncertainty Reduction (SGUR), a training-free framework
that complements token-level generation with active uncertainty reduction over
structured hypothesis spaces. SGUR represents incomplete observations as a query
graph and aligns them with candidate hypothesis graphs through a prior-weighted
partial graph matching mechanism, where database-derived empirical priors assign
greater importance to decision-relevant relations rather than treating all graph
edges uniformly. The resulting inference state is iteratively refined by an active
information-gain strategy that selects the next most informative node to query,
systematically pruning the candidate space. We instantiate SGUR on a challeng-
ing PrimeKG-derived sequential medical diagnosis benchmark, demonstrating
that structural grounding can partially substitute for model scale: starting from
only two observed symptoms, SGUR achieves92.5% Hit@3, outperforming the
strongest Graph-RAG baseline by9.4 percentage pointswhile using approxi-
mately 5×fewer tokens. Notably, SGUR enables smaller backbone models to
match or exceed the performance of models up to 10× larger, suggesting that
structural grounding provides a more robust and efficient paradigm for scaling
agent performance than relying exclusively on increasing model size.
∗Equal contribution.
Preprint.
arXiv:2610.30611v1  [cs.AI]  22 Oct 2026

fever
fever
User
Figure 1: Overview of the SGUR framework. The offline phase constructs a hierarchical expert graph
database Gdbfrom expert data under a shared schema. The online phase performs closed-loop multi-
turn inference through four stages: (a) query graph construction, which grounds observations Otinto
G(t)
qwith latent routing; (b) candidate pruning, which applies path filtering and specificity-aware
structural priors to obtain G(t)
cand; (c) prior-weighted partial FGW alignment, which ranks candidate
graphs by structural consistency with the query; and (d) active acquisition, which selects informative
unobserved nodes to obtain Ot+1. Across rounds, SGUR progressively contracts the candidate space
under partial observation.
1 Introduction
The design of autonomous agents capable of reasoning underpartial, noisy, and sequentially revealed
informationremains a central challenge for building reliable and efficient agentic systems. Although
scaling large language models (LLMs) has substantially improved linguistic fluency[OpenAI, 2024b,
2025b,c], reliable multi-turn inference requires more than fluent generation. In complex decision-
making environments, an agent must maintain a coherent hypothesis space, strategically acquire
missing evidence, and update its belief state as new information arrives. However, many current
agentic systems still rely primarily on token-level generation and linguistic similarity, which can
lead to weakly grounded multi-turn inference when semantic evidence is not jointly constrained by
relational structure[Fastowski and Kasneci, 2025, Zheng et al., 2026]. Under iterative interaction,
such systems may exhibitheuristic substitution[Kahneman, 2011], relying on associative pattern
matching rather than refining hypotheses against jointly grounded semantic evidence and relational
structure.
Existing paradigms only partially address this challenge. Chain-of-Thought and Tree-of-Thought
improve reasoning decomposition, but continue to operate largely within the model’s internal proba-
bilistic space[Wei et al., 2022, Yao et al., 2023a]. Recent memory-augmented systems also compress
conversational history, yet topology-insensitive compression can discard decision-relevant cues[You
et al., 2026]. GraphRAG frameworks introduce additional structure by retrieving graph-organized evi-
dence[Edge et al., 2024a, Guo et al., 2024, Peng et al., 2024, Zhang et al., 2025a]. Recent hierarchical
GraphRAG methods further strengthen this line by organizing knowledge into attributed communities
and hierarchical indexes, improving retrieval relevance and reducing online token cost[Wang et al.,
2026]. However, their main contribution remains on the retrieval side. The graph is primarily used to
access supporting evidence, which is then flattened into textual context for downstream generation.
Even with hierarchical indices such as ArchRAG[Wang et al., 2026], the reasoning process can remain
2

structurally underconstrained. Retrieved communities are typically converted into textual context
for generation, while graph topology and relational distances do not directly define the inference
objective. This limits the agent’s ability to use the knowledge graph as an active hypothesis space
rather than only as a retrieval substrate.
This limitation is especially consequential in high-stakes sequential settings, where deployment-
relevant agent performance depends on uncovering missing evidence through iterative, goal-directed
interaction[Bobrova, 2026]. Sequential, multi-hop clinical diagnosis provides a representative exam-
ple, where each intermediate inference can shape subsequent evidence acquisition and hypothesis
revision, a process recently formalized as theSequential Diagnosisparadigm[Nori et al., 2025, Trivedi
et al., 2023]. As a result, early reasoning errors may propagate across turns and compromise the final
decision[Zhao et al., 2025]. Recent empirical analyses further support this concern. LLMs often
become path-dependent in extended interactions, committing early to brittle assumptions and failing
to recover once the dialogue drifts[Laban et al., 2025]. Moreover, from an information-theoretic
perspective, multi-turn interaction can exhibit diminishing information gain and increasing redun-
dancy across turns[Gorle et al., 2025]. Although recent methods introduce iterative exploration, their
step-wise search often remains heuristic, making them vulnerable to shallow retrieval, error accumu-
lation, and task-shift failures[Luo et al., 2023, Sun et al., 2023, Lv et al., 2026]. These challenges
motivate a shift from likelihood-driven generation and retrieval-augmented context construction
toward semantic-structural uncertainty reduction over organized hypothesis spaces[Scanlin et al.,
2026].
In this paper, we introduceStructure-GroundedUncertaintyReduction (SGUR), a training-free
framework for efficient and reliable multi-turn inference. SGUR reformulatesmulti-step reasoning
as uncertainty reduction over structured hypothesis spaces, rather than unconstrained free-form
generation. At its core, SGUR couples geometric structural alignment with probabilistic belief updat-
ing. It maps the alignment cost between fragmented observations and candidate knowledge graphs
into a likelihood score, enabling a principled update of the agent’s belief state. By jointly grounding
each inference step in semantic evidence, relational structure, and domain-specific priors, SGUR
constrains the evolving reasoning trajectory to remain both semantically aligned and structurally
coherent under incomplete observations.
Operationally, SGUR uses two coupled mechanisms. (i) aprior-weighted partial alignmentthat
enables robust semantic-structural matching under sparse, incomplete evidence without requiring full
graph correspondence; and (ii) anExpected Information Gain (EIG)-driven acquisition strategy
that selects the next most informative query node. This cycle ensures that both hypothesis refinement
and evidence acquisition are governed by shared semantic-structural constraints, leading to a rapid
and efficient collapse of the candidate hypothesis space.
We evaluate SGUR in sequential clinical diagnosis over a large-scale hierarchical biomedical knowl-
edge graph, a demanding setting in which evidence is incomplete, observations are revealed iteratively,
and reliability is essential. Unlike small single-turn question-answering benchmarks, this setting
requires the agent to reason over a structured hypothesis space, acquire missing evidence across
turns, and maintain consistency over long-horizon inference. Empirically, SGUR delivers significant
system-level gains in diagnostic accuracy, inference efficiency, and token cost across strong baselines.
Crucially, when paired with smaller backbone models, SGUR achieves performance comparable to or
better than baselines powered by substantially larger state-of-the-art models. These results show that
reliable agent performance depends not only on model scale, but also onthe quality of the reasoning
framework that organizes inference.
In summary, our contributions are threefold. First, we introduce a training-freeinference paradigm
that reformulates multi-turn reasoning as Bayesian uncertainty contraction over structured hypothesis
spaces, providing a principled alternative to unconstrained free-form generation. Second, we develop
a unifiedsemantic-structural reasoning mechanismthat couples prior-weighted partial graph
alignment with information-theoretic evidence acquisition, enabling robust belief refinement and
goal-directed probing under sparse and sequentially revealed observations. Third, we provide a
large-scale evaluation on a hierarchy-structured sequential diagnosis benchmark, demonstrating an
empiricalstructure-scale substitution effect: SGUR improves Hit@3 by 9.4 percentage points over
the strongest GraphRAG baseline while reducing token overhead by approximately 5×, and enables
smaller backbone models to match or exceed substantially larger LLMs.
3

2 Related Work
Reasoning and Retrieval under Weak Structural Control.Recent advances in LLM reason-
ing have improved performance through structured prompting, retrieval augmentation, and graph-
organized evidence. Structured prompting methods, including CoT, ToT, GoT, least-to-most prompt-
ing, and self-consistency, improve decomposition and deliberative search[Wei et al., 2022, Yao et al.,
2023a, Besta et al., 2024, Zhou et al., 2022, Wang et al., 2022]. In parallel, RAG-style systems
improve factual grounding by incorporating external evidence into generation[Lewis et al., 2020,
Gao et al., 2023, Borgeaud et al., 2022]. Hybrid approaches such as RAT and BayesRAG variants
further interleave retrieval with reasoning or verification[Wang et al., 2024a, Li et al., 2026, Yang
et al., 2026, Wan and Yu, 2025, Jiang et al., 2023]. GraphRAG methods extend retrieval-augmented
reasoning by organizing evidence into explicit graph structures for local and global querying[Edge
et al., 2024a, Guo et al., 2024, Hu et al., 2025, Zhu et al., 2025], including medical variants such
as MedGraphRAG and KARE[Wu et al., 2025, Jiang et al., 2024]. Recent hierarchical GraphRAG
variants further improve retrieval relevance and efficiency through community-level organization and
hierarchical indexing[Wang et al., 2026].
However, these methods still organize reasoning primarily in prompt, text, or retrieval space. Even
when graphs are introduced, they are typically used as retrieval scaffolds whose outputs are flattened
into textual context for downstream generation[Edge et al., 2024a, Wu et al., 2025, Li et al., 2024,
Jiménez Gutiérrez et al., 2024, Wang et al., 2026]. As a result, they may improve evidence access
without explicitly constraining the evolving reasoning trajectory itself[Guo et al., 2024, Li et al., 2024,
Chen et al., 2024, Liang et al., 2025, Han et al., 2025, Xiang et al., 2025]. The central gap is therefore
not only how to retrieve better evidence, but how to make graph structure part of the inference state
itself.
Geometric Alignment and Active Uncertainty Reduction.Optimal Transport (OT) provides a
principled foundation for graph alignment under soft correspondence. Gromov–Wasserstein (GW)
and Fused Gromov–Wasserstein (FGW) compare structured objects by jointly accounting for node
semantics and relational structure[Mémoli, 2011, Vayer et al., 2019, 2020, 2018], while scalable
approximations improve practicality for larger graphs[Scetbon et al., 2022, Nguyen and Tsuda, 2023].
Partial transport methods relax the full-correspondence requirement and instead focus alignment on
overlapping substructures, making them especially relevant under incomplete observations[Chapel
et al., 2020, Bai et al., 2025].
A complementary line of work treats reasoning as an interactive process of sequential uncertainty
reduction rather than one-shot generation[Yao et al., 2023b, Trivedi et al., 2023]. In medical settings,
this appears in active question selection and information-seeking diagnosis[Hu et al., 2024, Wong
et al., 2026, Wang et al., 2024b, Nori et al., 2025], while related systems explore adaptive inference
control, solver arbitration, and memory management[Bergamaschi Ganapini et al., 2025, Zhang et al.,
2025b, You et al., 2026, Chung et al., 2025, Hu et al., 2026]. Yet these approaches generally manage
alignment, questioning, and belief refinement as separate procedures in text, memory, or retrieval
space. SGUR differs by coupling semantic-structural partial alignment with EIG-driven evidence
acquisition within a shared belief state, so that both hypothesis refinement and question selection are
governed by the same structure-grounded inference objective.
3 Methodology: The SGUR Framework
SGUR follows the two-phase framework in Figure 1. Section 3.1 describes the offline phase on the
left, where expert data are organized into a fixed hierarchical graph database Gdbunder a shared
diagnostic schema. Section 3.2 describes the online phase on the right, where each interaction round
follows four figure-aligned stages: (a) latent routing first aligns the current observations Otwith
database-level observation anchors and uses the matched anchors to construct the query graphG(t)
q;
(b) candidate pruning retrieves the associated candidate hypothesis graphs G(t)
candusing path filtering
and specificity-aware structural priors; (c) prior-weighted partial Fused Gromov–Wasserstein (PFGW)
alignment compares G(t)
qwith each candidate graph and induces a belief ranking; and (d) active
acquisition selects informative unobserved nodes to obtain Ot+1. Together, the fixed expert database
and the online loop define SGUR as a structure-grounded inference procedure that progressively
contracts the candidate hypothesis space under partial observation.
4

3.1 Offline Expert Graph Construction
The offline phase constructs the fixed expert graph database used by the online inference loop. As
shown on the left side of Figure 1, SGUR extracts hierarchical entities from expert data, maps
them into a shared diagnostic schema, and assembles directed inter-layer relations between adjacent
semantic layers. In our PrimeKG-derived clinical instantiation, the schema contains observations ( L1),
localizations ( L2), candidate diagnoses ( L3), and treatments ( L4). SGUR also builds an auxiliary L0
routing layer by clustering observation embeddings; L0serves as a routing index for fast query-graph
construction, not as a diagnostic semantic layer.
Formally, the expert graph database is
Gdb={G dj}n
j=1,(1)
where each diagnosis-centered graph is
Gdj= (V dj,Edj, ωdj,Xdj, ℓ).(2)
Here,Vdjis the node set, Edj⊆ Vdj× Vdjis the directed edge set, Xdjstores node features, and
ℓ:V dj→ {L 1, L2, L3, L4}maps nodes to semantic layers. In our benchmark, each djcorresponds
to a disease, but the construction only requires a structured diagnostic schema.
The edge weights ωdjinstantiate the specificity-aware structural priors. For each outgoing relation
(u, v)∈ E dj, SGUR uses
ωuv=1
|Nout(u)|,(3)
where Nout(u)denotes the outgoing neighbors of u. This down-weights generic nodes with many
downstream continuations and emphasizes more discriminative relations during candidate pruning
and prior-weighted partial FGW alignment. During online inference, Gdbis kept fixed: SGUR
grounds current observations, routes them to a candidate subset, aligns the query graph with candidate
graphs, and updates the belief ranking without reconstructing or updating the expert database.
3.2 Online Interactive Inference
The online phase performs closed-loop inference over the fixed database Gdb. At round t, the
accumulated evidenceO tis represented as a partial query graph
G(t)
q= Γ(O t) = (V(t)
q,E(t)
q, ω(t)
q,X(t)
q, ℓ(t)
q),(4)
where Γ(·) maps extracted entities into the shared schema. The online loop refines a belief ranking
over candidate graphs while acquiring the next evidence that is most useful under the current ranking.
(a) Query graph construction.SGUR first grounds the current observations into G(t)
q. To avoid
aligning every query against the full database, the auxiliary L0routing layer localizes the query to a
coarse observation neighborhood before fine-grained matching. After passing through L0, entities are
extracted from the query (see Appendix A.3 for the entity recognition prompt) and used to construct
the query graphG(t)
q. This corresponds to the query graph construction block in Figure 1.
(b) Candidate pruning and structural priors.After latent routing, SGUR applies path filtering to
retain only candidate graphs that are structurally compatible with the current query:
G(t)
cand=n
Gdj∈G db:R0(G(t)
q, Gdj) = 1, F path(G(t)
q, Gdj) = 1o
,(5)
where R0denotes the L0-based routing condition and Fpath denotes inter-layer path compatibility.
SGUR reuses the specificity-aware relation weights in Eq. (3). For node-level saliency, for any graph
Gwe define
SG(v) = log 
1 +|NG
in(v)|+|NG
out(v)|
, πG
v=SG(v)P
u∈VGSG(u).(6)
The normalized saliency πG
vprovides node masses for alignment and utility scores for active acquisi-
tion.
5

(c) Prior-weighted partial FGW alignment and belief ranking.For each Gdj∈G(t)
cand, SGUR
compares G(t)
qwithGdjusing prior-weighted PFGW alignment. Let M(j,t)be the node feature dis-
crepancy matrix, instantiated as cosine distances between query-node and candidate-node embeddings.
LetC(t)
qandCdjdenote weighted shortest-path distance matrices encoding internal graph structure.
We set the candidate and query node masses as p(j)= (πGdja)a∈Vdjandq(t)= (πG(t)
q
b)b∈V(t)
q. SGUR
computes
DPFGW (G(t)
q, Gdj) = min
γ∈Π m(p(j),q(t))n
(1−α)⟨γ,M(j,t)⟩
+αX
a,a′,b,b′Cdj(a, a′)−C(t)
q(b, b′)2
γabγa′b′o
.(7)
where Πm(p(j),q(t))is the set of partial couplings with transported mass m. The partial coupling
allows unmatched query evidence and unmatched expert structure to remain uncoupled, which is
essential when the query covers only a sparse substructure of the candidate graph.
The resulting discrepancies are converted into a normalized belief ranking:
P(t)(dj) =exp(−D PFGW (G(t)
q, Gdj)/β)
P
Gdk∈G(t)
candexp(−D PFGW (G(t)
q, Gdk)/β),(8)
whereβ >0controls ranking sharpness. The induced rank is
rt(dj) = rank 
P(t)(dj)
, G dj∈G(t)
cand.(9)
We measure remaining uncertainty by the entropy of this candidate belief distribution:
H(t)=−X
Gdj∈G(t)
candP(t)(dj) logP(t)(dj).(10)
Thus, candidate-space contraction is realized as progressive concentration of P(t)across interaction
rounds.
(d) Active acquisition.SGUR selects the next evidence from unobserved L1/L2nodes in the top- κ
candidate graphs. The ideal objective is to choose the query that most reduces uncertainty in the
belief distribution:
EIG(s) =Er∼p(r|s,P(t))
H(t)−H(P(t+1)|r, s)
.(11)
Exact EIG requires a response model over all possible user answers. SGUR therefore uses the
structure-guided proxy shown in Figure 1. Let Dκ
tbe the top- κcandidates under P(t), and define the
acquisition pool
At=[
dj∈Dκ
tVL1∪L2
dj\ V(t)
q.(12)
For eachv∈ A t, SGUR scores
ut(v) =X
dj∈Dκ
tπGdjv,(13)
and selects the next query batch as
A⋆
t= top-b 
{ut(v) :v∈ A t}
.(14)
The selected query is presented to the user, the returned observation is incorporated into Ot+1, and
the pruning–alignment–ranking–acquisition loop repeats.
Termination and output.The interaction terminates when the turn budget Tis exhausted or when
no informative unobserved node remains. SGUR outputs the top- kcandidate diagnoses under the
terminal belief ranking:
bC(O τ) = top-k 
{P(τ)(dj) :G dj∈G(τ)
cand}
.(15)
For evaluation, we record the first round in which the ground-truth diagnosis enters the top- kset,
yielding the MTTC metric used in Section 4.
6

Table 1: Main accuracy, convergence, and efficiency results. Final retrieval metrics are measured
after round 3 in the complex two-symptom regime; convergence reports complex-regime Hit@3
across rounds; efficiency reports three-round average total tokens and latency. All Hit@ Kvalues are
percentages. SGUR achieves the best final ranking quality, lowest MTTC, and lowest inference cost.
MethodComplex regime after R3 Complex Hit@3 Efficiency Avg.
MTTC↓MRR↑Hit@1↑Hit@2↑Hit@3↑R1 R2 R3 Tokens↓Latency↓
GraphRAG 1.74 0.71 61.20 77.60 83.10 59.60 76.90 83.10 10454.00 34.90
NodeRAG 1.90 0.71 60.20 75.20 80.10 54.20 75.00 80.10 8461.10 15.20
HippoRAG2 1.83 0.68 59.50 74.50 81.70 60.40 77.40 81.70 2352.80 11.40
ArchRAG 2.01 0.65 56.70 70.40 76.10 56.70 70.40 76.10 2683.40 10.40
SGUR (Ours) 1.62 0.77 64.90 83.40 92.50 61.40 84.70 92.50 1845.40 8.90
4 Experiments
We evaluate SGUR on a multi-round diagnostic retrieval task that stresses inference under incomplete
and progressively revealed evidence. The experiments are organized into two parts. Section 4.1
compares SGUR with graph-based RAG baselines in terms of accuracy, convergence, efficiency, and
robustness to model scale. Section 4.2 isolates the contribution of SGUR’s core mechanisms.
Experimental setup.We construct a controlled multi-round diagnostic retrieval benchmark from
PrimeKG[Chandak et al., 2023]. Each case is anchored by a ground-truth diagnosis d⋆, and evidence
is progressively revealed over three interaction rounds. PrimeKG provides typed symptom–anatomy–
disease–treatment relations, enabling hierarchical graph construction and database-derived edge-prior
initialization. By contrast, standard agent, tool-use, web-interaction, and GraphRAG benchmarks
mainly evaluate final-answer accuracy, tool execution, short-horizon task completion, or retrieval-
generation quality [Liu et al., 2024, Mialon et al., 2023, Qin et al., 2023, Zhou et al., 2023, Xiao et al.,
2025, Xiang et al., 2025], but typically do not expose the typed hierarchy and relational statistics
required to evaluate structure-grounded uncertainty reduction.
For each disease, we randomly permute its associated symptoms once and reveal the same sequence
to all methods to remove ordering randomness. We evaluate two regimes:Simple Diagnosis, with
b= 4 newly revealed symptoms per round, andComplex Diagnosis, with b= 2 newly revealed
symptoms per round. The simple regime contains 854 test cases and the complex regime contains
615 test cases, yielding 1,469 reasoning cases in total. We emphasize the complex regime in the main
results because it best captures sparse and ambiguous evidence under partial observation. Additional
construction details are provided in Appendix A.1.
4.1 Comparison with Graph-RAG Baselines
Baselines and Multi-round Adaptation.To verify the superiority of structure-grounded reasoning,
we compareSGURagainst four representative graph-RAG paradigms: the PageRank-basedHip-
poRAG2[Jiménez Gutiérrez et al., 2025], node-centricNodeRAG[Xu et al., 2025], community-level
GraphRAG[Edge et al., 2024b], and the hierarchicalArchRAG[Wang et al., 2026]. Since these
baselines are primarily designed for single-shot inference, we adapt them to the three-round setting
using atextual context re-injectionscheme. In this setup, each baseline receives a compact textual
summary of prior predictions and supporting evidence as conversational context in subsequent rounds.
This ensures all methods access the same revealed evidence while preserving their native retrieval
mechanisms. In contrast, SGUR facilitates a transition from context stacking to structured state
evolution. By maintaining an incrementally updated query graph G(t)
q, SGUR accumulates evidence
directly as graph state, circumventing the semantic drift and computational redundancy inherent in
repeatedly re-injecting dialogue history as text.
Metrics, Results, and Implementation Details.We assess retrieval effectiveness throughHit@ K
(K∈ {1,2,3} ),MRR@R3, andMean Turns to Confidence (MTTC), defined as the first round
in which the ground-truth diagnosis enters the top-3 candidate set, with a penalty of T+ 1 = 4 for
unretrieved cases. System efficiency is quantified via total token consumption and end-to-end latency.
All models utilize text-embedding-3-large OpenAI [2024a] for indexing and retrieval. Unless
7

Table 2: Model-scaling stress test in the complex regime. Each method is evaluated with large and
small backbones; retention reports the small/large ratio. SGUR preserves over 99% Hit@3 after
backbone scaling and remains above all large-backbone graph-RAG baselines.
Method Backbone Hit@3↑MRR↑MTTC↓
GraphRAGGPT-5.2 (L) 83.10 0.71 1.74
GPT-5.4-nano (S) 74.70 0.62 2.02
Retention (%) 90.30 87.10 86.10
NodeRAGGPT-5.2 (L) 80.10 0.70 1.90
GPT-5.4-nano (S) 73.60 0.60 2.12
Retention (%) 91.90 84.50 89.60
HippoRAG2GPT-5.2 (L) 81.70 0.68 1.83
GPT-5.4-nano (S) 79.80 0.60 2.11
Retention (%) 97.70 87.30 86.70
ArchRAGGPT-5.2 (L) 76.10 0.65 2.01
GPT-5.4-nano (S) 68.70 0.57 2.30
Retention (%) 90.20 87.40 87.30
SGURGPT-5.2 (L)92.50 0.77 1.62
GPT-5.4-nano (S)92.40 0.73 1.63
Retention (%)99.80 94.60 99.40
otherwise specified, online inference is evaluated with GPT-5.2 OpenAI [2025a], and robustness to
backbone scale is tested withGPT-5.4-nanoOpenAI [2026].
As detailed in Table 1, SGUR achieves the strongest accuracy–efficiency profile in the complex
diagnostic regime. It reaches92.50% Hit@3, outperforming the strongest graph-RAG baseline by
9.40 percentage points, while also obtaining the lowest MTTC. Its Hit@3 improves from 61.40% to
92.50% across rounds, indicating that the structured query state progressively narrows the candidate
hypothesis space as new evidence is acquired. Furthermore, SGUR reduces token consumption by
5.70×and latency by3.90 ×compared with GraphRAG. Together, these results show that SGUR
improves diagnostic reliability and lowers the online cost of multi-turn inference by organizing
reasoning through structure-grounded state updates, rather than relying solely on larger backbone
models.
Robustness to backbone scale.We further test whether SGUR’s advantage persists when the
backbone model is scaled down. Table 2 shows that SGUR with the smallerGPT-5.4-nanobackbone
remains above all graph-RAG baselines using the largerGPT-5.2backbone. Moreover, SGUR retains
over99% of its large-backbone Hit@3 after scaling down, while the baselines exhibit larger drops
in ranking quality and convergence. This result supports the central claim that reliable multi-turn
inference depends not only on backbone scale, but also on how the reasoning process is structurally
organized. Full scaling results, including Hit@1 and Hit@2, are provided in Table 7 in Appendix A.6.
Accuracy–efficiency trade-off.Figure 2 visualizes the system-level trade-off between final Hit@3
and average token cost. SGUR occupies the upper-left Pareto-optimal region, achieving the highest
retrieval accuracy with the lowest inference budget among all methods. This demonstrates that SGUR
shifts the accuracy–efficiency frontier, rather than improving accuracy by using larger prompts, longer
interaction histories, or higher online computation.
4.2 Ablation Study
Component contribution under sparse evidence.Table 3 jointly evaluates component necessity and
evidence sufficiency. Removing the L0latent routing layer causes the largest degradation, reducing
final Hit@3 from 92.50% to68.90% and increasing MTTC from 1.62 to2.31. This shows that latent
routing is not merely a computational shortcut: by aligning raw observations to database-level anchors
before query-graph construction, it provides the grounding needed to build focused query graphs
and retrieve relevant candidate hypotheses. Replacing partial transport with full FGW also causes
a substantial drop, reducing Hit@3 to 73.30% and increasing MTTC to 2.45. This confirms that
8

Figure 2: Accuracy–efficiency Pareto frontier in the complex regime. The x-axis reports average total
tokens over three interaction rounds, and the y-axis reports final Hit@3 after round 3. Higher is better
on the y-axis and lower is better on the x-axis. SGUR achieves the highest retrieval accuracy with the
lowest inference cost.
Table 3: Combined ablation study in the complex regime. The left block isolates each SGUR
component at the final round, while the right block reports Hit@3 as observed evidence increases.
All Hit@3 and∆Hit@3 values are percentages.
VariantFinal complex-regime performance Evidence sufficiency: Hit@3
Hit@3↑∆Hit@3 MRR↑MTTC↓2 Obs. 4 Obs. 6 Obs.
Full SGUR92.50–0.77 1.62 61.40 85.70 92.50
Semantic-only Alignment 91.90−0.600.75 1.64 59.00 85.30 91.90
Structure-only Alignment 91.60−0.900.76 1.67 56.20 84.40 91.60
Uniform Importance 83.40−9.100.64 2.01 38.20 71.80 83.40
Full FGW 73.30−19.200.53 2.45 15.30 59.50 73.40
w/o Latent Routing 68.90−23.600.57 2.31 35.70 59.80 68.90
incomplete observations should be matched only to supported substructures rather than forced into
full graph correspondence. Uniform importance reduces Hit@3 to 83.40% , indicating that database-
derived specificity priors help distinguish generic relations from diagnostically informative ones.
Finally, semantic-only and structure-only variants remain competitive but consistently underperform
the full model, showing that SGUR benefits from coupling node-level evidence with relational
consistency.
Effect of evidence sufficiency.The right block of Table 3 shows that SGUR’s advantage is most
pronounced when evidence is sparse. With only two observed symptoms, Full SGUR reaches
61.40% Hit@3, whereas removing latent routing drops performance to 35.70% , and replacing partial
transport with full FGW drops it further to 15.30% . As additional observations are revealed, all
variants improve, but the full model remains consistently strongest. This pattern supports the central
mechanism of SGUR: structure-grounded routing, partial alignment, and specificity-aware weighting
are most valuable when the agent must reduce uncertainty from incomplete evidence rather than
simply rank hypotheses from abundant observations.
5 Discussion and Conclusion
Our results support a central thesis: reliable multi-turn inference depends not only on the scale of the
backbone model, but also on how the reasoning process is organized. SGUR improves sequential
inference by turning fragmented observations into a structured query state, aligning this state with
9

candidate hypothesis graphs through prior-weighted partial matching, and actively selecting the
next evidence node to reduce uncertainty. Across accuracy, convergence, latency, token cost, and
model-scaling experiments, this structure-grounded design consistently improves performance under
sparse and progressively revealed evidence. These findings suggest that reasoning frameworks can
partially substitute for model scale by providing the structural discipline that free-form generation
often lacks.
SGUR should nevertheless be interpreted as a research framework, not as a deployed clinical decision
system. We instantiate the framework in sequential clinical diagnosis because it provides a demanding
testbed with incomplete evidence, multi-round interaction, and structured hypothesis spaces. The
system is not intended for real-world diagnosis, treatment recommendation, or medical decision-
making without expert validation, clinician oversight, and appropriate regulatory review.
The framework also relies on several assumptions. It assumes that domain knowledge can be
organized into a meaningful hierarchy, that observations can be mapped to standardized evidence
nodes, and that the expert graph is sufficiently reliable during inference. Although PrimeKG provides
a controlled large-scale benchmark, real-world settings may contain noisy observations, missing
relations, evolving knowledge, and heterogeneous evidence sources. Future work should evaluate
SGUR on broader long-horizon agent benchmarks and extend it to domains such as technical support,
device maintenance, and software debugging, where agents must iteratively gather evidence, refine
hypotheses, and make decisions under structural constraints.
References
Yikun Bai, Shuang Wang, Huy Tran, Hengrong Du, Juexin Wang, and Soheil Kolouri. Fused
partial gromov–wasserstein for structured objects.arXiv preprint arXiv:2502.09934, 2025. URL
https://arxiv.org/abs/2502.09934.
M Bergamaschi Ganapini, M Campbell, F Fabiano, L Horesh, J Lenchner, A Loreggia, N Mattei,
F Rossi, B Srivastava, and K B Venable. Fast, slow, and metacognitive thinking in AI.npj Artificial
Intelligence, 1(1):27, 2025. doi: 10.1038/s44387-025-00027-5. URL https://www.nature.
com/articles/s44387-025-00027-5.
Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi,
Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler.
Graph of thoughts: Solving elaborate problems with large language models. InProceedings of
the AAAI Conference on Artificial Intelligence, volume 38, pages 17682–17690, 2024. URL
https://ojs.aaai.org/index.php/AAAI/article/view/29720.
Natalia Bobrova. Clinical agentic systems: Methodologies for development, orchestration, and gover-
nance of autonomous AI agents in healthcare workflows—alignment with 2026 FDA/EMA guiding
principles. SSRN, 2026. URL https://papers.ssrn.com/sol3/papers.cfm?abstract_
id=6350259. Preprint, SSRN abstract 6350259.
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican,
George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego
De Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren
Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol
Vinyals, Simon Osindero, Karen Simonyan, Jack Rae, Erich Elsen, and Laurent Sifre. Improving
language models by retrieving from trillions of tokens. InInternational Conference on Machine
Learning, pages 2206–2240. PMLR, 2022. URL https://proceedings.mlr.press/v162/
borgeaud22a.html.
Payal Chandak, Kexin Huang, and Marinka Zitnik. Building a knowledge graph to enable precision
medicine.Scientific Data, 10(1):67, 2023. doi: 10.1038/s41597-023-01960-3. URL https:
//doi.org/10.1038/s41597-023-01960-3.
Laetitia Chapel, Mokhtar Z Alaya, and Gilles Gasso. Partial optimal transport with applications
on positive-unlabeled learning.Advances in Neural Information Processing Systems, 33:2903–
2913, 2020. URL https://proceedings.neurips.cc/paper_files/paper/2020/hash/
1e6e25d952a0d639b676ee20d0519ee2-Abstract.html.
10

Rubing Chen, Xulu Zhang, Jiaxin Wu, Wenqi Fan, Xiao-Yong Wei, and Qing Li. Knowledge pyramid
construction for multi-level retrieval-augmented generation.arXiv preprint arXiv:2407.21276,
2024. URL https://arxiv.org/abs/2407.21276 . v1 (2024) was titled “Multi-Level Query-
ing using A Knowledge Pyramid”; retitled in v3 (2025).
Stephen Chung, Wenyu Du, and Jie Fu. Thinker: Learning to think fast and slow.arXiv preprint
arXiv:2505.21097, 2025. URLhttps://arxiv.org/abs/2505.21097.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A
graph RAG approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024a.
URLhttps://arxiv.org/abs/2404.16130.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A
graph RAG approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024b.
URLhttps://arxiv.org/abs/2404.16130.
Alina Fastowski and Gjergji Kasneci. Understanding knowledge drift in LLMs through misin-
formation. InInternational Workshop on Discovering Drift Phenomena in Evolving Land-
scapes, pages 74–85. Springer, 2025. doi: 10.1007/978-3-031-82346-6_5. URL https:
//link.springer.com/chapter/10.1007/978-3-031-82346-6_5.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng
Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey.
arXiv preprint arXiv:2312.10997, 2023. URLhttps://arxiv.org/abs/2312.10997.
Abhiram Rao Gorle, Amit Kumar Singh Yadav, and Tsachy Weissman. Quantifying information
gain and redundancy in multi-turn LLM conversations. InNeurIPS 2025 Workshop on Multi-Turn
Interactions in Large Language Models, 2025. URL https://openreview.net/forum?id=
5gpABTkcUJ.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. LightRAG: Simple and fast retrieval-
augmented generation.arXiv preprint arXiv:2410.05779, 2024. URL https://arxiv.org/
abs/2410.05779v1.
Haoyu Han, Li Ma, Harry Shomer, Yu Wang, Yongjia Lei, Kai Guo, Zhigang Hua, Bo Long, Hui
Liu, Charu C. Aggarwal, and Jiliang Tang. RAG vs. GraphRAG: A systematic evaluation and
key insights.arXiv preprint arXiv:2502.11371, 2025. URL https://arxiv.org/abs/2502.
11371v2.
Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. GRAG: Graph retrieval-
augmented generation. InFindings of the Association for Computational Linguistics: NAACL 2025,
pages 4145–4157, 2025. URLhttps://aclanthology.org/2025.findings-naacl.232/.
Zhanghao Hu, Qinglin Zhu, Runcong Zhao, Di Liang, Hanqi Yan, Yulan He, and Lin Gui.
Beyond RAG for agent memory: Retrieval by decoupling and aggregation.arXiv preprint
arXiv:2602.02007, 2026. URLhttps://arxiv.org/abs/2602.02007.
Zhiyuan Hu, Chumin Liu, Xidong Feng, Yilun Zhao, See-Kiong Ng, Anh Tuan Luu, Junxian He,
Pang Wei Koh, and Bryan Hooi. Uncertainty of thoughts: Uncertainty-aware planning enhances
information seeking in large language models. InAdvances in Neural Information Processing
Systems, volume 37, pages 24181–24215, 2024. URLhttps://arxiv.org/abs/2402.03271.
Pengcheng Jiang, Cao Xiao, Minhao Jiang, Parminder Bhatia, Taha Kass-Hout, Jimeng Sun, and
Jiawei Han. Reasoning-enhanced healthcare predictions with knowledge graph community retrieval.
arXiv preprint arXiv:2410.04585, 2024. URLhttps://arxiv.org/abs/2410.04585.
Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992, 2023.
URLhttps://aclanthology.org/2023.emnlp-main.495/.
11

Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. HippoRAG:
Neurobiologically inspired long-term memory for large language models.Advances in Neural
Information Processing Systems, 37:59532–59569, 2024. URL https://arxiv.org/abs/2405.
14831.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From RAG to memory:
Non-parametric continual learning for large language models.arXiv preprint arXiv:2502.14802,
2025. doi: 10.48550/arXiv.2502.14802. URLhttps://arxiv.org/abs/2502.14802.
Daniel Kahneman.Thinking, Fast and Slow. Macmillan, 2011. URL https://us.macmillan.
com/books/9780374275631/thinkingfastandslow/.
Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. LLMs get lost in multi-turn
conversation.arXiv preprint arXiv:2505.06120, 2025. URL https://arxiv.org/abs/2505.
06120.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-augmented generation for knowledge-intensive NLP tasks. InAdvances in Neural Infor-
mation Processing Systems, volume 33, pages 9459–9474, 2020. URL https://proceedings.
neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html.
Xuan Li, Yining Wang, Haocai Luo, Shengping Liu, Jerry Liang, Ying Fu, Weihuang, Jun Yu, and
Junnan Zhu. BayesRAG: Probabilistic mutual evidence corroboration for multimodal retrieval-
augmented generation.arXiv preprint arXiv:2601.07329, 2026. URL https://arxiv.org/
abs/2601.07329.
Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yaojie Lu, Qiaoyu Tang, Fei Huang, Xianpei
Han, Le Sun, and Yongbin Li. StructRAG: Boosting knowledge intensive reasoning of LLMs via
inference-time hybrid information structurization.arXiv preprint arXiv:2410.08815, 2024. URL
https://arxiv.org/abs/2410.08815.
Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao, Mengshu Sun,
Zhiqiang Zhang, Jun Zhou, Wenguang Chen, et al. KAG: Boosting LLMs in professional domains
via knowledge augmented generation. InCompanion Proceedings of the ACM on Web Conference
2025, pages 334–343, 2025.
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding,
Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui
Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang.
Agentbench: Evaluating llms as agents. InInternational Conference on Learning Representations,
2024. URLhttps://arxiv.org/abs/2308.03688.
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful
and interpretable large language model reasoning.arXiv preprint arXiv:2310.01061, 2023. URL
https://arxiv.org/abs/2310.01061.
Can Lv, Heng Chang, Shengyu Tao, Mingju Chen, Zhaoxin Fan, Ziwei Zhang, Yuchen Guo, and
Shiji Zhou. All-mem: Agentic lifelong memory via dynamic topology evolution.arXiv preprint
arXiv:2603.19595, 2026. URLhttps://arxiv.org/abs/2603.19595.
Facundo Mémoli. Gromov–wasserstein distances and the metric approach to
object matching.Foundations of Computational Mathematics, 11:417–487,
2011. URL https://www.researchwithrutgers.org/en/publications/
gromov-wasserstein-distances-and-the-metric-approach-to-object-ma/.
Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom.
GAIA: A benchmark for general AI assistants.arXiv preprint arXiv:2311.12983, 2023. URL
https://arxiv.org/abs/2311.12983.
Dai Hai Nguyen and Koji Tsuda. On a linear fused gromov–wasserstein distance for graph struc-
tured data.Pattern Recognition, 138:109351, 2023. URL https://www.sciencedirect.com/
science/article/abs/pii/S0031320323000523.
12

Harsha Nori, Mayank Daswani, Christopher Kelly, Scott Lundberg, Marco Tulio Ribeiro, Marc
Wilson, Xiaoxuan Liu, Viknesh Sounderajah, Jonathan Carlson, Matthew P. Lungren, Bay Gross,
Peter Hames, Mustafa Suleyman, Dominic King, and Eric Horvitz. Sequential diagnosis with
language models.arXiv preprint arXiv:2506.22405, 2025. URL https://arxiv.org/abs/
2506.22405.
OpenAI. New embedding models and API updates, 2024a. URL https://openai.com/index/
new-embedding-models-and-api-updates/. Introducestext-embedding-3-large.
OpenAI. New embedding models and API updates, 2024b. URL https://openai.com/index/
new-embedding-models-and-api-updates/. Introducestext-embedding-3-large.
OpenAI. Introducing GPT-5.2, 2025a. URL https://openai.com/index/
introducing-gpt-5-2/.
OpenAI. Introducing GPT-5.2, 2025b. URL https://openai.com/index/
introducing-gpt-5-2/.
OpenAI. Update to GPT-5 system card: GPT-5.2, 2025c. URL https://openai.com/index/
gpt-5-system-card-update-gpt-5-2/.
OpenAI. Introducing GPT-5.4 mini and nano, March 2026. URL https://openai.com/index/
introducing-gpt-5-4-mini-and-nano/.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. Graph retrieval-augmented generation: A survey.arXiv preprint arXiv:2408.08921, 2024.
URLhttps://arxiv.org/abs/2408.08921.
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong,
Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark
Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language
models to master 16000+ real-world APIs.arXiv preprint arXiv:2307.16789, 2023. URL https:
//arxiv.org/abs/2307.16789.
Joe Scanlin, Armando Cuesta, and Martín Varsavsky. Representation before retrieval: Structured
patient artifacts reduce hallucination in clinical AI systems.medRxiv, 2026. doi: 10.64898/
2026.02.13.26346256. URL https://www.medrxiv.org/content/10.64898/2026.02.13.
26346256v1. Preprint, not peer reviewed.
Meyer Scetbon, Gabriel Peyré, and Marco Cuturi. Linear-time Gromov–Wasserstein distances using
low rank couplings and costs. InInternational Conference on Machine Learning, pages 19347–
19365. PMLR, 2022. URLhttps://proceedings.mlr.press/v162/scetbon22b.html.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M
Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of
large language model on knowledge graph.arXiv preprint arXiv:2307.07697, 2023. URL
https://arxiv.org/abs/2307.07697.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval
with chain-of-thought reasoning for knowledge-intensive multi-step questions. InProceedings of
the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pages 10014–10037, 2023. URLhttps://aclanthology.org/2023.acl-long.557/.
Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard, and Nicolas Courty. Optimal
transport for structured data with application on graphs.arXiv preprint arXiv:1805.09114, 2018.
URLhttps://arxiv.org/abs/1805.09114.
Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard, and Nicolas Courty. Optimal
transport for structured data with application on graphs. InInternational Conference on Machine
Learning, pages 6275–6284. PMLR, 2019. URL https://proceedings.mlr.press/v97/
titouan19a.html.
13

Titouan Vayer, Laetitia Chapel, Rémi Flamary, Romain Tavenard, and Nicolas Courty. Fused Gromov–
Wasserstein distance for structured objects.Algorithms, 13(9):212, 2020. doi: 10.3390/a13090212.
URLhttps://www.mdpi.com/1999-4893/13/9/212.
Xueyao Wan and Hang Yu. MMGraphRAG: Bridging vision and language with interpretable
multimodal knowledge graphs.arXiv preprint arXiv:2507.20804, 2025. URL https://arxiv.
org/abs/2507.20804.
Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, and Yuchi Ma. ArchRAG: Attributed community-
based hierarchical retrieval-augmented generation. InProceedings of the AAAI Conference on
Artificial Intelligence, volume 40, pages 15868–15876, 2026. doi: 10.1609/aaai.v40i19.38619.
URLhttps://ojs.aaai.org/index.php/AAAI/article/view/38619.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdh-
ery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models.
arXiv preprint arXiv:2203.11171, 2022. URLhttps://arxiv.org/abs/2203.11171.
Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xiaojian Ma, and Yitao Liang. RAT: Retrieval
augmented thoughts elicit context-aware reasoning in long-horizon generation.arXiv preprint
arXiv:2403.05313, 2024a. URLhttps://arxiv.org/abs/2403.05313.
Ziyu Wang, Hao Li, Di Huang, and Amir M. Rahmani. HealthQ: Unveiling questioning capabilities
of LLM chains in healthcare conversations.arXiv preprint arXiv:2409.19487, 2024b. URL
https://arxiv.org/abs/2409.19487v1.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H Chi,
Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language
models. InAdvances in Neural Information Processing Systems, volume 35, pages 24824–
24837, 2022. URL https://proceedings.neurips.cc/paper_files/paper/2022/hash/
9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html.
Hui Min Wong, Philip Heesen, Pascal Janetzky, Martin Bendszus, and Stefan Feuerriegel. MedClarify:
An information-seeking AI agent for medical diagnosis with case-specific follow-up questions.
arXiv preprint arXiv:2602.17308, 2026. URLhttps://arxiv.org/abs/2602.17308.
Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina, Yueming Jin, and
Vicente Grau. Medical graph RAG: Evidence-based medical large language model via graph
retrieval-augmented generation. InProceedings of the 63rd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pages 28443–28467, 2025. URL https:
//aclanthology.org/2025.acl-long.1381/.
Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and
Jinsong Su. When to use graphs in RAG: A comprehensive analysis for graph retrieval-augmented
generation.arXiv preprint arXiv:2506.05690, 2025. URL https://arxiv.org/abs/2506.
05690.
Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qian-wen Zhang, Di Yin, Xing Sun, and Xiao
Huang. GraphRAG-Bench: Challenging domain-specific reasoning for evaluating graph retrieval-
augmented generation.arXiv preprint arXiv:2506.02404, 2025. URL https://arxiv.org/
abs/2506.02404.
Tianyang Xu, Haojie Zheng, Chengze Li, Haoxiang Chen, Yixin Liu, Ruoxi Chen, and Lichao
Sun. NodeRAG: Structuring graph-based RAG with heterogeneous nodes.arXiv preprint
arXiv:2504.11544, 2025. doi: 10.48550/arXiv.2504.11544. URL https://arxiv.org/abs/
2504.11544.
Jiashu Yang, Chi Zhang, Abudukelimu Wuerkaixi, Xuxin Cheng, Cao Liu, Ke Zeng, Xu Jia, and
Xunliang Cai. AutoThinkRAG: Complexity-aware control of retrieval-augmented reasoning for
image-text interaction.arXiv preprint arXiv:2603.05551, 2026. URL https://arxiv.org/abs/
2603.05551.
14

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem solving with large language mod-
els. InAdvances in Neural Information Processing Systems, volume 36, pages 11809–
11822, 2023a. URL https://proceedings.neurips.cc/paper_files/paper/2023/hash/
271db9922b8d1f4dd7aaef84ed5ac703-Abstract.html.
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.
ReAct: Synergizing reasoning and acting in language models. InThe Eleventh International
Conference on Learning Representations, 2023b. URL https://arxiv.org/abs/2210.03629 .
Zhixing You, Jiachen Yuan, and Jason Cai. D-mem: A dual-process memory system for LLM agents.
arXiv preprint arXiv:2603.18631, 2026. URLhttps://arxiv.org/abs/2603.18631.
Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao Chen,
Yilin Xiao, Chuang Zhou, Junnan Dong, Yi Chang, and Xiao Huang. A survey of graph retrieval-
augmented generation for customized large language models.arXiv preprint arXiv:2501.13958,
2025a. URLhttps://arxiv.org/abs/2501.13958.
Shao Zhang, Xihuai Wang, Wenhao Zhang, Chaoran Li, Junru Song, Tingyu Li, Lin Qiu, Xuezhi
Cao, Xunliang Cai, Wen Yao, Weinan Zhang, Xinbing Wang, and Ying Wen. Leveraging dual
process theory in language agent framework for real-time simultaneous human-AI collaboration.
InProceedings of the 63rd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 4081–4108, 2025b. URLhttps://aclanthology.org/2025.
acl-long.206/.
Xinjie Zhao, Fan Gao, Xingyu Song, Yingjian Chen, Rui Yang, Yanran Fu, Yuyang Wang, Yusuke
Iwasawa, Yutaka Matsuo, and Irene Li. ReAgent: Reversible multi-agent reasoning for knowledge-
enhanced multi-hop QA. InProceedings of the 2025 Conference on Empirical Methods in
Natural Language Processing, pages 4067–4089, 2025. URL https://aclanthology.org/
2025.emnlp-main.202.pdf.
Lifan Zheng, Jiawei Chen, Qinghong Yin, Jingyuan Zhang, Xinyi Zeng, and Yu Tian. Rethinking
the reliability of multi-agent systems: A perspective from byzantine fault tolerance. InProceed-
ings of the AAAI Conference on Artificial Intelligence, volume 40, pages 35012–35020, 2026.
URL https://ojs.aaai.org/index.php/AAAI/article/view/40806 . Extended version:
arXiv:2511.10400.
Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans,
Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. Least-to-most prompting enables complex
reasoning in large language models.arXiv preprint arXiv:2205.10625, 2022. URL https:
//arxiv.org/abs/2205.10625.
Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng,
Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A realistic
web environment for building autonomous agents.arXiv preprint arXiv:2307.13854, 2023. URL
https://arxiv.org/abs/2307.13854.
Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. Knowledge graph-guided retrieval
augmented generation. InProceedings of the 2025 Conference of the Nations of the Ameri-
cas Chapter of the Association for Computational Linguistics: Human Language Technologies
(Volume 1: Long Papers), pages 8912–8924, 2025. URL https://aclanthology.org/2025.
naacl-long.449/.
15

A Additional Experimental Details and Results
This section provides the additional implementation details needed to reproduce our main experi-
mental results, including benchmark construction, test case sampling, hyperparameters, prompts, the
compute environment, code and data release, and comprehensive results.
A.1 Source Data, Schema Construction, and Test Case Sampling
Source data.We usePrimeKGChandak et al. [2023] as the source of clinical knowledge. From
the full graph we retain four node types corresponding to our hierarchical schema: disease ( L3),
phenotype ( L1), anatomy ( L2), and drug ( L4). All other node types and their incident edges are
removed.
Phenotype–anatomy linkage.PrimeKG does not directly link phenotypes to the body parts where
they appear. The closest available connection goes through genes: a phenotype is linked to genes,
and those genes are linked to the organs where they are expressed. However, this only tells us where
the relevant genes happen to be active at the molecular level, not where the phenotype is actually
felt or observed in the clinic. For example, a phenotype like headache may be associated with genes
expressed throughout the body, which does not mean the phenotype manifests in all of those locations.
For this reason, we do not use this indirect path. Instead, we build the phenotype–anatomy edges
directly using a prompted LLM as a medical knowledge source. For each phenotype s∈L 1, we ask
the LLM to list the body parts where stypically appears in clinical practice, restricting its answers
to the anatomy vocabulary already present in PrimeKG. These (phenotype, anatomy) pairs are then
added toG dbasL 1→L 2edges, completing the four-layer schema.
Test case sampling.We sample disjoint test sets for the simple ( b=4) and complex ( b=2) regimes.
For each regime, we draw diseases uniformly at random from the disease pool. For each sampled
disease, symptoms are revealed over three rounds according to the disease’s associated symptom set.
If a disease has fewer than 3bassociated symptoms, symptoms are re-sampled from its symptom set
in subsequent rounds to complete the required observation budget, while preserving diversity within
each round whenever possible.
A.2 Hyperparameters
Table 4 summarizes the main SGUR hyperparameters used across all experiments. Below we provide
the additional implementation-level settings required for full reproduction.
Table 4: Hyperparameter settings used in all SGUR experiments unless stated otherwise.
Symbol Description Value
TTurn budget 3
bPer-turn question batch size (complex regime) 2
bPer-turn question batch size (simple regime) 4
kTop-kreturn list size 3
κTop-κcandidate pool for evidence acquisition 5
τsim Entity-matching cosine-similarity threshold 0.85
αPFGW feature–structure trade-off (Eq. 7) 0.5
cmis PFGW feature-mismatch penalty 1
βBelief-distribution temperature (Eq. 8) 1.0
NL0 Number ofL 0routing clusters 256
PFGW solver.The PFGW in Eq. (7)is solved with the Frank–Wolfe algorithm from the POT
library, using the squared-loss coupling L(x, y) =|x−y|2and a maximum of 2000 outer iterations.
The structural cost matrices CdjandC(t)
qare computed as directed shortest-path distances under the
specificity-aware edge weights of Eq. (3), then jointly rescaled to [0,1] . The partial transport mass
mis set to the inherited overlap mass between query and candidate marginals, allowing unmatched
evidence and unmatched expert structure to remain uncoupled. The trade-off αand feature-mismatch
penaltyc misfollow Table 4.
16

L0pre-filter.The auxiliary routing layer L0is constructed offline over the L1observation em-
beddings produced bytext-embedding-3-large, in two stages. We first reduce dimensionality with
UMAP ( n_neighbors= 30 , controlling the local-vs-global trade-off; n_components= 10 target di-
mensions; cosine metric), then cluster the projected vectors with HDBSCAN ( min_cluster_size=
15, the minimum cluster size; min_samples= 5 , controlling cluster density). At query time, the top
10clusters ranked by cosine similarity between the query embedding and each cluster’s centroid are
forwarded to the schema-constrained extractor as the candidate observation pool, which then drives
the path-compatibility filter that yieldsG(t)
cand.
Entity matching.Extracted entities are aligned to graph nodes by cosine similarity at threshold
τsim, retaining the top 2observation ( L1) matches per extracted symptom and the top 1localization
(L2) match per observation.
Baseline hyperparameters.We use the official open-source implementations and default hyperpa-
rameters of each baseline, with two exceptions made for fair comparison: (i) the same embedding
model (text-embedding-3-large) is used across all methods, and (ii) the same indexing LLM (GPT-
5.2) is used during corpus construction.
A.3 Prompts
We document the prompts used for each LLM-invoked component below. All prompts are released in
full with our code.
SGUR: schema-constrained extractor.A single prompt asks the LLM to read the patient’s
message and pull out two things: the observations being described ( L1) and the localizations where
they are felt ( L2). To keep the model from guessing, we also pass in a shortlist of likely observations
produced by the L0pre-filter and instruct the LLM to choose only from this list. The output must be
returned as JSON with three fields: observations and localizations.
Entity recognition prompt
System message:
You are a medical Named Entity Recognizer (NER).
Task.Extract symptoms mentioned or strongly implied by the user query, and list
exactly 3 organs most specifically associated with each symptom.
Matching.
• The user may describe symptoms directly, indirectly, or in lay language.
•You MUST semantically match to the closest symptom names from the Symptom
List.
• Use meaning-based matching, not exact keywords.
Hard rules.
1. Choose ONLY from the Symptom List (never invent symptoms).
2. Return minimum 1 and AT MOST 3 symptoms total.
3.For each symptom, list 1 of the most directly affected organs that are implicated
by that symptom.
Symptom List (flat list):{routed_observation_shortlist}
Organ List:{organ_vocabulary}
Output format (JSON only):
{"symptom name from list": ["organ1", "organ2", "organ3"], ...}
User message:{query}
Baselines: multi-round adaptation prompt.As described in Section 4, single-shot baselines are
adapted to the multi-round protocol by prompting them at each round to summarize their top-ranked
diseases together with the supporting observations and localizations. The exact prompt template used
for this adaptation is shown in the box below.
17

Multi-Round Adaptation Prompt for Single-Shot Baselines
ROLE.You are a clinical reasoning assistant performing differential disease identification basedonlyon
retrieved medical evidence.
TASK.The user’s query describes symptoms experienced by a patient. Using only the retrieved information:
1. Identify which diseases are supported by the patient’s symptoms.
2.For each candidate disease, extract the supportingobservations(matched symptoms) andlocalizations
(affected organs or anatomical sites) from the retrieved context.
3. Select the ten most likely diseases.
RULES.
• Return exactly the top 10 diseases.
• Base reasoning solely on the retrieved context.
• Do not introduce diseases, observations, or localizations absent from the retrieved context.
• Do not fabricate symptoms, organs, or clinical facts.
•A disease is more likely if more of its phenotypes match the patient’s symptoms and its associated organs
are consistent with the described localizations.
•If observation or localization support is weak or ambiguous, still return the best 10 candidates with
whatever evidence is available (use an empty list if none).
• Do not explain general medical knowledge or include disclaimers.
OUTPUT.Return the following raw JSON only:
{
" top_10_diseases ": [
{
" disease_name ": "<name >",
" supporting_observations ": ["< symptom >", "< symptom >"] ,
" supporting_localizations ": ["< organ >", "<organ >"]
},
...
]
}
A.4 Compute Environment
All experiments are run on a single machine with an 11th Gen Intel Core i7-11370H processor (4
cores, 8 threads) and 16 GB of RAM. LLM inference is performed through the TensorBlock API
gateway, which provides unified access to the evaluated backbones (GPT-5.2 and GPT-5.4-nano from
OpenAI). Graph operations and PFGW solving are implemented in Python using networkx andPOT.
The end-to-end latency reported in Table 6 includes API round-trip time and is measured under the
same network conditions for all methods to avoid bias.
A.5 Code and Data Release
To support reproducibility, we will release the source code, experimental scripts, processed data
splits, and evaluation protocols upon acceptance of the paper. The released materials will include
the implementation of SGUR, baseline configuration files, and instructions for reproducing the main
results reported in the paper.
A.6 Full Accuracy, Convergence, and Efficiency Results
Table 5 expands the main results by reporting the full retrieval metrics and round-wise Hit@3
trajectories in both simple and complex regimes. This table complements the compact summary in
the main text and shows that SGUR’s advantage is not limited to the final round: in the complex
two-symptom regime, SGUR obtains the best Hit@3 at every interaction round and reaches the
correct diagnosis in fewer turns, as reflected by the lowest MTTC. In the simple regime, SGUR
is slightly below HippoRAG2 at the first round but improves more rapidly as additional evidence
is acquired, achieving the best R2 and R3 Hit@3. These trends support the claim that structure-
18

Table 5: Full retrieval performance and multi-round convergence. Final metrics are measured after
round 3 in the complex two-symptom regime, while the simple and complex columns report Hit@3
across interaction rounds. All Hit@ Kvalues are percentages. SGUR achieves the best final ranking
quality and the lowest MTTC, and its advantage becomes more pronounced as evidence accumulates.
MethodFinal complex-regime performance (R3) Simple Hit@3 Complex Hit@3
MTTC↓MRR@R3↑Hit@1↑Hit@2↑Hit@3↑R1 R2 R3 R1 R2 R3
GraphRAG 1.74 0.71 61.20 77.60 83.10 64.20 81.30 84.50 59.60 76.90 83.10
NodeRAG 1.90 0.71 60.20 75.20 80.10 63.00 79.00 86.00 54.20 75.00 80.10
HippoRAG2 1.83 0.68 59.50 74.50 81.7068.3082.90 87.70 60.40 77.40 81.70
ArchRAG 2.01 0.65 56.70 70.40 76.10 58.20 74.10 80.90 56.70 70.40 76.10
SGUR (Ours) 1.62 0.77 64.90 83.40 92.5067.1086.70 93.40 61.40 84.70 92.50
Table 6: Full per-round inference efficiency breakdown. We report input tokens, output tokens,
total tokens, and end-to-end latency across three interaction rounds. SGUR keeps total token usage
nearly flat from R1 to R3 and achieves the lowest latency in every round, confirming that structured
query-state updates avoid the growing cost of re-injecting accumulated dialogue history.
MethodInput Tokens↓Output Tokens↓Total Tokens↓Latency (s)↓
R1 R2 R3 R1 R2 R3 R1 R2 R3 R1 R2 R3
GraphRAG 9886.00 10076.00 10080.00 376.00 452.00 491.00 10263.00 10529.00 10570.00 34.10 35.00 35.50
NodeRAG 7609.00 8118.40 8394.70 337.40 427.90 496.00 7946.40 8546.30 8890.70 14.10 15.10 16.40
HippoRAG2 2014.20 2058.90 2116.10 249.40 293.50 326.20 2263.70 2352.40 2442.40 10.10 11.00 13.00
ArchRAG 2223.40 2283.10 2384.40 326.10 402.30 431.00 2549.50 2685.40 2815.40 9.50 10.40 11.20
SGUR (Ours) 1755.50 1759.70 1765.50 78.60 79.40 85.40 1838.00 1843.20 1854.90 8.10 9.10 9.70
grounded alignment and active acquisition convert newly revealed evidence into faster candidate-space
contraction.
Table 6 provides the full per-round efficiency breakdown used to compute the average token and
latency results reported in the main text. SGUR keeps input tokens nearly constant from R1 to R3 and
produces substantially fewer output tokens than all baselines, indicating that the online loop relies on
structured query-state updates rather than repeated free-form generation. In contrast, graph-RAG
baselines accumulate larger input contexts as previous dialogue is re-injected. This explains why
SGUR is both more accurate and more efficient in the main results.
19

Table 7: Full model-scaling stress test in the complex regime. Each method is evaluated under a
Large (L) and Small (S) backbone, with retention reported in the third row of each method. Hit@ k
and retention values are percentages. Retention is computed as Small/Large for Hit@ kand MRR;
for MTTC, where lower is better, retention is computed as min(L,S)/max(L,S) . All metrics are
computed after round 3.
Method Backbone Hit@1 Hit@2 Hit@3 MRR MTTC
GraphRAGGPT-5.2 (L) 61.20 77.60 83.10 0.71 1.74
GPT-5.4-nano (S) 52.10 64.40 74.70 0.62 2.02
Retention (%) 85.10 83.00 90.30 87.10 86.10
NodeRAGGPT-5.2 (L) 60.20 75.20 80.10 0.70 1.90
GPT-5.4-nano (S) 48.50 65.40 73.60 0.60 2.12
Retention (%) 80.60 87.00 91.90 84.50 89.60
HippoRAG2GPT-5.2 (L) 59.50 74.50 81.70 0.68 1.83
GPT-5.4-nano (S) 54.30 71.60 79.80 0.60 2.11
Retention (%) 91.40 96.10 97.70 87.30 86.70
ArchRAGGPT-5.2 (L) 56.70 70.40 76.10 0.65 2.01
GPT-5.4-nano (S) 49.80 63.80 68.70 0.57 2.30
Retention (%) 87.80 90.60 90.20 87.40 87.30
SGURGPT-5.2 (L)64.90 83.40 92.50 0.77 1.62
GPT-5.4-nano (S)60.40 80.50 92.40 0.73 1.63
Retention (%)93.10 96.50 99.80 94.60 99.40
20