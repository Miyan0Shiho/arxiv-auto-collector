# Entropic Claim Resolution: Uncertainty-Driven Evidence Selection for RAG

**Authors**: Davide Di Gioia

**Published**: 2026-03-30 13:49:03

**PDF URL**: [https://arxiv.org/pdf/2603.28444v1](https://arxiv.org/pdf/2603.28444v1)

## Abstract
Current Retrieval-Augmented Generation (RAG) systems predominantly rely on relevance-based dense retrieval, sequentially fetching documents to maximize semantic similarity with the query. However, in knowledge-intensive and real-world scenarios characterized by conflicting evidence or fundamental query ambiguity, relevance alone is insufficient for resolving epistemic uncertainty. We introduce Entropic Claim Resolution (ECR), a novel inference-time algorithm that reframes RAG reasoning as entropy minimization over competing semantic answer hypotheses. Unlike action-driven agentic frameworks (e.g., ReAct) or fixed-pipeline RAG architectures, ECR sequentially selects atomic evidence claims by maximizing Expected Entropy Reduction (EER), a decision-theoretic criterion for the value of information. The process dynamically terminates when the system reaches a mathematically defined state of epistemic sufficiency (H <= epsilon, subject to epistemic coherence). We integrate ECR into a production-grade multi-strategy retrieval pipeline (CSGR++) and analyze its theoretical properties. Our framework provides a rigorous foundation for uncertainty-aware evidence selection, shifting the paradigm from retrieving what is most relevant to retrieving what is most discriminative.

## Full Text


<!-- PDF content starts -->

Entropic Claim Resolution: Uncertainty-Driven Evidence
Selection for RAG
Davide Di Gioia
ucesigi@ucl.ac.uk
Abstract
Current Retrieval-Augmented Generation (RAG) systems predominantly rely on relevance-
based dense retrieval, sequentially fetching documents to maximize semantic similarity with the
query. However, in knowledge-intensive and real-world scenarios characterized by conflicting evi-
dence or fundamental query ambiguity, relevance alone is insufficient for resolving epistemic un-
certainty. We introduce Entropic Claim Resolution (ECR), a novel inference-time algorithm that
reframes RAG reasoning as entropy minimization over competing semantic answer hypotheses.
Unlike action-driven agentic frameworks (e.g., ReAct) or fixed-pipeline RAG architectures, ECR
sequentially selects atomic evidence claims by maximizing Expected Entropy Reduction (EER),
a decision-theoretic criterion for the value of information. The process dynamically terminates
when the system reaches a mathematically defined state of epistemic sufficiency (H≤ϵ, subject to
epistemic coherence). We integrate ECR into a production-grade multi-strategy retrieval pipeline
(CSGR++) and demonstrate its theoretical properties. Our framework provides a rigorous foun-
dation for uncertainty-aware evidence selection, shifting the paradigm from retrieving what is
most relevant to retrieving what is most discriminative.
1 Introduction
The integration of Large Language Models (LLMs) with external knowledge bases through Retrieval-
Augmented Generation (RAG) has become the de facto standard for mitigating hallucinations and
enabling knowledge-intensive Question Answering (QA). Conventional RAG systems operate on a
rigidretrieve-then-readparadigm, predominantly leveraging maximum inner product search (MIPS)
in dense continuous spaces [2] to fetch the top-kmost semantically relevant text chunks. While highly
effective for simple, factoid-based QA where a single ground-truth answer exists, this relevance-driven
approach exhibits severe degradation in real-world, knowledge-intensive scenarios. Such scenarios are
frequently characterized by inherent query ambiguity, conflicting evidence across multiple sources, and
complex multi-hop dependencies.
In these challenging settings, standard dense retrieval suffers from what we termepistemic collapse:
the tendency to retrieve highly redundant information that is semantically similar to the query, rather
than fetching the discriminative evidence needed to resolve the underlying uncertainty. Consequently,
the LLM is forced to synthesize an answer from a biased or incomplete evidence distribution, often
leading to unhedged, overconfident, or factually inaccurate generation.
Recent architectural advancements attempt to transcend simple MIPS. Graph-based paradigms,
such as Context-Seeded Graph Retrieval (CSGR) and GraphRAG, expand retrieval scope via struc-
tured knowledge relation traversal. Concurrently, agentic and iterative verification workflows (e.g.,
ReAct [5], Tree-of-Thoughts [6], Self-RAG [7]) allow LLMs to dynamically interact with search tools,
reflecting on retrieved context to guide subsequent actions. However, these state-of-the-art approaches
still critically lack a principled, decision-theoretic stopping criterion and evidence selection mechanism.
Graph techniques rely on static pipeline configurations (e.g., fixed graph-hop depth), while agentic
1arXiv:2603.28444v1  [cs.AI]  30 Mar 2026

systems depend on heuristic thresholding or prompt-driven self-reflection, which frequently suffer from
infinite looping, premature termination, or unprincipled evidence weighting.
Critically, modern RAG systems lack a mathematically rigorous definition of what constitutes
sufficient evidence and an explicit objective function for selecting which specific piece of evidence to
retrieve next at inference time. To bridge this fundamental gap, we propose Entropic Claim Resolu-
tion (ECR), an inference-time algorithm that reframes the retrieval and synthesis process asentropy
minimization over a latent space of semantic answer hypotheses. Drawing inspiration from Informa-
tion Theory [8] and Bayesian Experimental Design [9], ECR models the QA task probabilistically. It
initializes a probability distribution over a set of mutually exclusive potential answer hypotheses and
iteratively selects atomic factual claims from a retrieved candidate pool to evaluate.
Crucially, in ECR, evidence selection is decoupled from semantic relevance to the query. Instead,
claims are selected by maximizingExpected Entropy Reduction (EER); that is, choosing the
specific piece of evidence most likely to collapse the probability distribution toward a single, correct
hypothesis (or cleanly bifurcate it in the case of irreconcilable conflict). The algorithm adaptively
navigates the evidence graph under a principled stopping rule, terminating only when the entropy of
the hypothesis space falls below a predefined threshold ofepistemic sufficiency.
In summary, our main contributions are:
1. We introduce Entropic Claim Resolution (ECR), a decision-theoretic evidence selection algo-
rithm for RAG, shifting the paradigm from retrieving what is mostrelevantto what is most
discriminativein resolving hypothesis ambiguity.
2. We formally define a principled, mathematically rigorous stopping criterion for iterative RAG
pipelines based on epistemic sufficiency (H(A|X)≤ϵ).
3. We identify a behavioral phase transition under structured contradiction: by integrating a
lightweight coherence signal (λ >0), we show that ECR transitions from forced epistemic
collapse to principled ambiguity exposure, prioritizing explicit contradictions when present and
safely refusing to reduce uncertainty when evidence is inherently inconsistent.
4. We demonstrate the practical scalability of ECR by implementing it as a fast, inference-time
algorithm integrated into a production-grade multi-strategy retrieval architecture (CSGR++),
requiring no bespoke fine-tuning or specialized model weights.
Significance.A central implication of this work is that improving retrieval-augmented reasoning
does not necessarily require larger models, longer context windows, or additional data, but rather
principled control over how existing evidence is selected and evaluated during inference. By explicitly
modelingepistemicuncertaintyandoptimizingevidenceselectionforinformationgain, EntropicClaim
Resolution provides a lightweight, computationally efficient mechanism for improving robustness and
interpretability. This makes the framework particularly valuable for high-stakes enterprise deploy-
ments (e.g., medical, legal, or financial QA) where mitigating unhedged hallucinations and controlling
inference costs are critical. Ultimately, this perspective highlights an alternative path for scaling
knowledge-intensive systems: a path grounded in decision-theoretic inference rather than indiscrimi-
nate context expansion, particularly in settings characterized by noisy, conflicting, or heterogeneous
evidence.
2 Related Work
2.1 Dense, Graph-Augmented, and Agentic Retrieval
StandarddenseretrievalselectsasetofdocumentsDbyprioritizingtheirconditionalprobabilitygiven
the query,P(D|Q), commonly approximated via cosine similarity embeddings [1]. To overcome the
short-sightedness of relevance search, advanced graph-based architectures, notably GraphRAG [3]
2

and Context-Seeded Graph Retrieval (CSGR), implicitly construct or traverse knowledge graphs over
chunks or entities to expand the evidence space. In enterprise environments, hybrid systems such as
CSR-RAG [4] further integrate structural and relational signals to support large-scale schemas.
Concurrently, the convergence of dynamic retrieval policies with autonomous planning has crys-
tallized into the paradigm ofAgentic RAG. Frameworks such as ReAct [5], Tree-of-Thoughts [6], and
Self-RAG [7] allow language models to interleave intermediate reasoning steps with retrieval actions in
order to refine subsequent queries. Recent Systematization of Knowledge (SoK) studies emphasize this
shift from static pipelines toward modular control strategies. However, despite their flexibility, agentic
retrieval systems intrinsically rely on heuristic prompt designs or static thresholds to determine when
to halt retrieval or which information to prioritize. As a result, they lack a rigorous mathematical
definition of epistemic sufficiency and an explicit objective for selecting the next most informative
piece of evidence as uncertainty unfolds during inference.
2.2 Uncertainty Quantification (UQ) in RAG
A critical prerequisite for adaptive retrieval is accurately characterizing what a model does not know.
Recent benchmarks such as URAG [11] demonstrate that while RAG can improve factual grounding,
it also introduces new sources of epistemic uncertainty, including relevance mismatch and selective
attention to partial evidence, which can paradoxically amplify overconfident hallucinations under
noisy retrieval conditions. In response, a growing body of work on Retrieval-Augmented Reasoning
(RAR) focuses on quantifying uncertainty across the retrieval and generation stages. For example,
methods such as Retrieval-Augmented Reasoning Consistency (R2C) [12] model multi-step reasoning
as a Markov Decision Process and perturb generation to measure output stability via majority voting,
building upon foundational frameworks for semantic uncertainty [10].
These uncertainty-aware approaches are effective for post-hoc answer evaluation, abstention, or
calibration. However, they are not designed to guide theselection of evidence itselfduring the
reasoning process. In particular, they do not provide a mechanism for choosing which atomic piece of
evidence should be retrieved or verified next in order to maximize information gain prior to answer
synthesis.
2.3 Entropy-Aware Context Management
The application of Shannon entropy as a control signal for managing LLM context is an emerging
research direction. Large context windows in standard RAG often lead to attention dilution and un-
constrained entropy growth, motivating recent work on entropy-aware context control. For instance,
BEE-RAG(BalancedEntropy-EngineeredRAG)[13]modifiesattentiondynamicstomaintainentropy
invariance over long contexts, while SF-RAG (Structure-Fidelity RAG) [14] leverages document hier-
archy as a low-entropy prior to prevent evidence fragmentation. Similarly, L-RAG (Lazy RAG) [15]
employs predictive entropy thresholds to gate expensive retrieval operations, defaulting to parametric
knowledge when uncertainty is estimated to be low.
ECR shares this information-theoretic lineage but departs in a critical way: rather than using
entropy to compress, gate, or truncate context, ECR applies entropy directly as an objective for
sequentially selecting discriminative evidence variables. This shift reframes entropy from a passive
diagnostic into an active decision criterion guiding inference-time reasoning.
2.4 Claim-Level Verification and Value of Information
While standard RAG operates on monolithic document chunks, recent diagnostic and safety-oriented
frameworks decompose retrieved content into atomic claims. Systems such as MedRAGChecker [16]
evaluate biomedical QA systems by extracting fine-grained claims and checking them against struc-
tured knowledge bases, while agentic fact-checking pipelines (e.g., SAFE [17] and CIBER [18]) retrieve
3

supporting and refuting evidence for individual statements. These approaches demonstrate the im-
portance of claim-level reasoning for reliability and interpretability.
ECRalignsthisgranularverificationparadigmwithclassicalprinciplesfromBayesianexperimental
designandactivelearning. Inactivelearning, theobjectiveistoselectthenextunlabeledinstancethat
maximizes expected information gain. By formulating inference-time evidence selection as Expected
Entropy Reduction (EER) over discrete factual claims, ECR bridges symbolic uncertainty modeling
and neural generation, optimizing retrieval for thevalue of informationrather than semantic relevance
alone.
3 Methodology: Entropic Claim Resolution (ECR)
ECR formulates the evidence selection problem as a sequential decision process targeting a reduction
in epistemic uncertainty across competing generative outcomes.
ECR assumes high-recall candidate generation has already occurred (via upstream retrieval) and
focuses exclusively on resolving uncertainty within the resulting candidate claim set.
3.1 Problem Formulation
LetC={c 1, c2, . . . , c n}be a finite subset of atomic factual claims embedded within a corpus. For
a given complex queryQ, assume that assessing the veracity of any given claimc iprovides a signal
regarding the query’s answer. We denote the latent truth variable associated with claimc iasX i∈
{0,1}, indicating whether the claim is empirically validated within the specific source document.
Upon identifying high epistemic uncertainty in the retrieval space (e.g., via confidence variance or
conflicting keyword analysis), ECR initializes anAnswer Hypothesis SpaceA={a 1, a2, . . . , a k}.
This space represents the set of mutually exclusive potential macro-answers to the query.1In our
implementation,Ais robustly generated dynamically: either by querying the LLM to propose dis-
tinct valid hypotheses derived from subsets of the initialk-best claims, or via deterministic vector
clustering when operating purely off-line. Our objective is to sequentially refine a probability distri-
butionP(A|X eval, Q)over these hypotheses, conditioned on the dynamic subset of evaluated claims
Xeval⊆X, initialized at a uniform priorP(a) =1
|A|.
3.2 Objective Function: Answer Entropy
The epistemic uncertainty regarding the true outcome is robustly quantified using Shannon entropy.
Let the entropy of the hypothesis space after evaluating a subset of claimsX evalbe:
H(A|X eval) =−X
a∈AP(a|X eval) log2P(a|X eval)(1)
3.3 Expected Entropy Reduction (EER) and Selection Policy
At thet-th iteration, the system must choose the next claimc∗from the unevaluated candidate pool
Ccandto formally verify. Rather than relying on cosine relevance sim(c i, Q), we select the claim that
maximizes Expected Entropy Reduction (Information Gain). The selection policy is formally defined
as:
c∗= arg max
c∈CcandEER(c|X eval)(2)
The EER is precisely the difference between current entropy and the expected posterior entropy after
observing the truth value of claimc:
EER(c|X eval) =H(A|X eval)−E Xc
H(A|X eval∪ {X c})
(3)
1Weusemutualexclusivityforanalyticalclarity; theframeworknaturallyextendstopartiallyoverlappinghypotheses
via soft assignment of claims to hypotheses.
4

This criterion ensures the algorithm intrinsically favorsdiscriminativeclaims, i.e., evidence that
cleanly segregates the hypothesis space. In practice, EER is approximated by measuring the proba-
bilistic variance between the specific subsets of competing macro-hypotheses actively supported versus
unsupported by candidatec. A claim supporting all hypotheses equally yields an EER of 0, reflecting
its redundancy, regardless of its semantic similarity to the query.
Implementation-Level EER Proxy.Computing the true mathematical expectation over all pos-
sible generative outcomes is typically intractable during low-latency inference. Therefore, we deploy
a computationally efficient proxy that approximates Expected Entropy Reduction without requiring
full marginalization over latent truth variables. In our concrete implementation, each candidate claim
cpartitions the hypothesis set into those that citecas supporting evidence and those that do not. Let
A+(c) ={a∈ A:c∈supp(a)}andA−(c) =A \ A+(c). Denote the probability mass in each subset
asp +(c) =P
a∈A+(c)P(a|X eval)andp −(c) =P
a∈A−(c)P(a|X eval). We score discriminativity via
the following heuristic proxy:
[EER(c) =|p+(c)−p −(c)|
p+(c) +p −(c)·H(A|X eval)·conf(c),(4)
where conf(c)∈[0,1]denotes claim confidence. This proxy is linear in the number of hypotheses
and preserves the core objective of prioritizing claims that maximally split the posterior mass, while
remaining tractable for inference-time use.
Design choice of the EER proxy.The heuristic proxy in Eq. (10) is intentionally not a symmetric
approximation of classical expected information gain, which typically favors balanced posterior splits;
rather, it is designed for bounded-budget inference, where the objective is rapid reduction of epistemic
uncertainty rather than exploratory experimentation. In retrieval-augmented reasoning, once poste-
rior mass concentrates on a subset of hypotheses, prioritizing high-confidence, high-imbalance claims
accelerates convergence and reduces redundant evidence retrieval. This exploitative bias is therefore
a deliberate design choice aligned with low-latency inference and downstream synthesis constraints.
Coherence-aware selection.In addition to entropy reduction, ECR incorporates a lightweight co-
herence signal that prioritizes evaluating claims likely to complete an explicit contradiction when such
evidence exists. Concretely, we add a small regularization termλ·ConflictPotential(c)to the selection
objective, yielding score(c) = [EER(c) +λ·ConflictPotential(c), whereConflictPotential(c)∈ {0,1}is
non-zero ifcis an explicit negation of, or completes a contradiction pair with, a previously evaluated
claim. This term does not override entropy reduction but ensures that unresolved contradictions are
surfaced early rather than averaged away. Empirically, we observe that any non-zeroλinduces stable
coherence-aware behavior without requiring fine-grained tuning (Appendix, Figure A.1).
Contradiction-aware coherence term.LetC evaldenote the set of claims that have already been
evaluated. We define a binary contradiction indicator
ConflictPotential(c) =(
1if∃c′∈ Cevalsuch thatc≡ ¬c′,
0otherwise.(5)
That is,ConflictPotential(c)activates only when evaluatingcwould complete an explicit contradiction
pair in the evidence. This coherence signal is structural rather than probabilistic: it does not penalize
hypotheses or posteriors directly, and it does not measure global consistency. Instead, it biases claim
selection toward surfacing epistemic inconsistency when it exists, preventing entropy-only selection
from averaging away contradictory evidence. The resulting claim-selection objective is
c∗= arg max
c∈Ccand
[EER(c) +λ·ConflictPotential(c)
,(6)
5

Box 1: Entropic Claim Resolution (ECR)
Input:queryQ, candidate claimsC cand, entropy thresholdϵ, max iterationsT
1. Hypotheses.InitializeA ←GenerateHypotheses(Q,C cand)(LLM or clustering), set
uniform priorP(a) = 1/|A|.
2. Loop.Fort= 1..T: computeH(A|X eval)(Eq. 1). If epistemic sufficiency holds (Eq. 11′)
stop.
2a. Select.Choosec∗∈ Ccandmaximizing [EER(c) +λ·ConflictPotential(c)(Eq. 4).
2b. Verify.EstimateP(X c∗= 1)using provenance and support/contradiction statistics
(Eq. 8).
2c. Update.UpdateP(A|X eval∪ {X c∗})(Eq. 7), addX c∗toX eval, removec∗fromC cand.
3. Output.Returnarg max aP(a|X eval)if epistemic sufficiency holds (Eq. 11′), else return the
ranked distribution overA.
Figure 1: Pseudo-code for ECR without external algorithm packages.
whereλ≥0controls the strength of contradiction-aware selection. Settingλ= 0recovers entropy-
only ECR. While entropy reduction remains the primary objective, anyλ >0ensures that explicit
contradiction-completing claims are prioritized when present, under the bounded EER scale induced
by the hypothesis entropy. This prioritization is observed empirically as a sharp phase transition in
theλ-sweep ablation, where behavior saturates for all testedλ >0.
3.4 Bayesian Posteriors and Epistemic Sufficiency
Upon selectingc∗, the system evaluates its intrinsic truthX c∗against the source context and prove-
nance metadata (see Section 3.5). The hypothesis probabilities are concurrently updated utilizing
localized Bayes’ rules. Concretely, hypotheses intersecting functionally with validated claims observe
significant targeted probability mass boosts, severely suppressing contradicting disjoint branches.
P(A|X eval∪ {X c∗}) =P(X c∗|A)P(A|X eval)P
˜aP(X c∗|˜a)P(˜a|X eval)(7)
whereP(X c|A)represents the conditional likelihood of observing the claimcassuming hypothesisA
is true.
The iterative verification procedure gracefully terminates when the system reaches a mathematical
state ofepistemic sufficiency, parameterized by thresholdϵ(e.g.,ϵ= 0.3bits):
 
H(A|X eval)≤ϵ
∧ ¬Conflict(X eval)(11′)
whereConflict(X eval)indicates the presence of mutually incompatible claims (e.g., an explicit claim
and its negation) within the evaluated evidence. Alternatively, if all candidates are exhausted or
maximum iterations are met withH > ϵ, ECR halts and explicitly exposes the competing hypotheses
and their final mass distributions, structurally mapping the unresolvable ambiguity of the corpus. The
complete iterative procedure is summarized in Box 1.
3.5 Verification via Topological Provenance
In practical continuous-learning implementations, the inferential verity linkP(X c= 1|A)can be
computed dynamically rather than natively assuming perfect model alignment. Instead of relying
solely on parametric LLM-driven prompt verification, ECR explicitly incorporates the topological
provenance of the multi-modal knowledge graph natively. LetS(c)andC(c)represent the structural
support graph-edge counts and contradictory graph-edge counts of claimctracked intricately within
6

the backing EAV (Entity-Attribute-Value) datastore, applying implicit Laplace smoothing. The final
topological verification probability is thus seamlessly and robustly blended:
P(X c= 1) =

S(c) + 1
S(c) +C(c) + 2ifS(c) +C(c)>0,
Pprior_conf (Xc= 1)otherwise.(8)
This matchesthe deployed behaviorin ourimplementation: wheneverhistoricalsupport/contradiction
signals exist, the system uses a Laplace-smoothed empirical truth estimate; otherwise, it falls back to
the extraction-time prior confidence.
3.6 Theoretical Properties
To solidify the inferential validity of the sequential system, we deduce its operational performance
bound mapping.
Theorem 1(Termination and Budget Bound).For any finite candidate setC cand, ECR terminates
after at mostmin(T,|C cand|)claim evaluations. Moreover, if there exists a constantδ >0such that
at each iteration the selected claim satisfiesE[H t−1−H t]≥δwheneverH t−1> ϵ, then ECR reaches
epistemic sufficiency in at most⌈(H 0−ϵ)/δ⌉iterations.
We emphasize that this result characterizes sufficient conditions for convergence under informative
evidence selection, rather than a minimax or adversarial worst-case guarantee. When explicit contra-
dictions exist in the evidence, the sufficient conditions for convergence are intentionally violated, and
ECR terminates by exposing ambiguity rather than collapsing the posterior.
Proof.The first statement holds because each iteration evaluates and removes at most one claim,
and the loop is explicitly capped byT. For the second statement, telescoping the assumed expected
entropy decrease yieldsE[H t]≤H 0−tδuntil reachingϵ, hencet≥(H 0−ϵ)/δsuffices.
4 System Integration: ECR within CSGR++
To evaluate ECR beyond isolated theoretical constraints, we integrated it into a production-grade,
multi-strategy retrieval pipeline. While ECR is algorithmically orthogonal to any specific retriever, we
utilize the CSGR++ architecture as our primary testbed. In this section, we describe the surrounding
system components that generate, structure, and verify the atomic candidate claims consumed by
the ECR inference loop. Figure 2 illustrates the resulting end-to-end architecture and the position of
ECR within it.
4.1 HyRAG v3 Ingestion and Index Construction
Structured and Tabular Data as First-Class Evidence.HyRAG v3 natively supports struc-
tured and semi-structured tabular data, rather than treating tables as flattened text. During inges-
tion, the system performs automatic schema inference, including column typing (numeric, categorical,
temporal), identifier detection, and time-series normalization. Individual table cells and derived ag-
gregates are materialized as atomic claims with explicit provenance, row identifiers, column metadata,
and canonical time keys. Structured aggregation queries are grounded through a text-to-SQL exe-
cution path with guarded, read-only execution and validation against real table values. All tabular
claims enter the same inference-time evidence pool as textual and graph-derived claims, allowing En-
tropic Claim Resolution to reason uniformly over mixed structured and unstructured evidence. This
design enables precise numeric grounding, temporal filtering, and auditable reasoning not natively
supported by graph-enhanced RAG systems that operate over synthesized document summaries.
7

Query
Ensemble Retrieval
(Vector | Graph | Claim)
Entropic Claim Resolution
Entropy-Guided Selection
Response Synthesisepistemic sufficiencyInside ECR:
Hypothesis spaceA
PosteriorP(A|X)
EER-based claim selection
Figure2: Systemoverview: EntropicClaimResolution(ECR)operatesasaninference-timecontroller
between competitive retrieval and answer synthesis. Given a retrieved claim set, ECR sequentially
selects evidence to minimize hypothesis entropy and terminates when epistemic sufficiency is reached.
Vector-Based Retrieval as a Core Substrate.HyRAG v3 fully incorporates dense vector re-
trieval as a primary evidence acquisition mechanism. Raw document chunks, atomic claims, and
synthesized summaries are embedded into dedicated vector indices and queried using cosine similarity
with optional metadata and identifier filtering. Vector retrieval is used to seed claim pools, initialize
hypothesis construction, and ground subsequent structured and graph-based reasoning. Rather than
assuming vector similarity implies evidentiary sufficiency, HyRAG v3 subjects all vector-retrieved
candidates to inference-time evaluation under Entropic Claim Resolution, allowing relevance-based
signals to be retained while preventing overconfidence in semantically similar but non-discriminative
evidence.
ECR operates at inference time, but its effectiveness depends on upstream ingestion and indexing
that preserve atomicity, provenance, and temporal structure. The implemented HyRAG v3 pipeline
(in our reference implementation) performs the following steps.
Auto-adaptive schema inference with feedback calibration.AnAutoAdaptAgentinfers a
schema from CSV/Excel/PDF/DataFrame inputs, identifying an ID column, categorical columns,
numeric columns, and time-series columns. A subsequentschema feedback loopperforms a dry-run
parse of the firstNrows (configurable) and adjusts misclassified columns (e.g., “numeric” columns
with excessive null-rates), producing a corrected schema used for full ingestion.
Robust parsing with repeated-header detection and temporal normalization.The inges-
tion parser supports multiple formats and implements spreadsheet-specific heuristics, including merg-
ing complementary multi-row headers and skipping repeated header rows using an overlap threshold
(≥0.70token overlap). Time-series columns are normalized via a data-driven time-key parser that rec-
ognizes patterns such as years (e.g., 2024), quarters (e.g., 2024Q1), halves (e.g., 2024H2), and trailing
windows (e.g., LTM/TTM), and maps them to a canonical order key used for temporal slicing.
8

EAV SQLite store with safe query execution.All ingested records are persisted in an Entity–
Attribute–Value SQLite backend (GenericStore). For downstream aggregation queries, the system
exposes a text-to-SQL route but enforces a strictSELECT-only guardrail: the SQL executor blocks
write operations and limits result sizes.
Embeddings and vector indices with deterministic fallbacks.The embedding subsystem is
three-tiered: an online embedding API (if available), a local sentence-transformer fallback, and a
deterministic hashed-vector fallback for fully offline operation. Vector indices support an optional
database backend (LanceDB when installed) and a pure NumPy cosine-similarity backend otherwise;
both support ID filtering for category/time constraints.
Atomic claim extraction and claim index.During ingestion, the system extracts atomic claims,
entities, and lightweight semantic relations(h, r, t)into a dedicatedClaimStore. Claim vectors are
embedded and stored in a separate claim vector index to enable claim-first retrieval.
Hierarchicalsummarizationasretrievablenodes.Toimproveglobalrecall, thesystemclusters
embedded row representations using a pure-NumPyk-means routine (no external ML dependencies),
summarizes each cluster (LLM when available), re-embeds the summaries, and inserts them into the
same row-level vector index under a reserved ID prefix. As a result, standard vector retrieval can
surface both raw rows and higher-level cluster summaries. Cluster summaries are stored as first-class
retrievable nodes and compete directly with raw rows during vector retrieval.
ECR is exclusively activated on the analyticalCSGR_PLUSroute selected by the upstream query
router, and is bypassed forLOOKUP,RELATIONAL,SEMANTIC,TOOL, andSQLroutes.
External tools are treated as deterministic operators outside the entropy-driven reasoning loop
(the LLM only formats a JSON tool call when available, with an offline numeric-statistics fast-path),
i.e., excluded from ECR’s epistemic modeling rather than treated as competing uncertainty-reduction
actions.
To bound computational overhead, ECR is invoked dynamically strictly when the retrieved config-
uration exhibits high epistemic uncertainty. The trigger conditions are natively integrated via three
heuristics:
1.High Claim Volume:The retriever fetches heavily saturated candidate spaces (>15claims).
2.Syntactic Ambiguity:Detection of uncertainty keywords within the active query (e.g., “un-
certain”, “conflicting”, “disagree”, “multiple”, “various”).
3.Confidence Variance Constraint:The variance in micro-level claim confidenceσ2across
thekretrieved claims exceeds an empirical threshold of0.15(with confidence actively tied to
tracking topological support-contradiction metrics within the underlying datastore).
To evaluate ECR in a high-performance setting, we implement it as a standalone and modular res-
olution engine within a production-grade Context-Seeded Graph Retrieval (CSGR++) architecture.
While ECR is algorithmically orthogonal to any specific retriever, CSGR++ serves as a rigorous exper-
imental testbed that preserves atomicity, provenance, and multi-strategy retrieval signals. Knowledge
is extracted and stored as atomic semantic claims in an Entity-Attribute-Value (EAV) backend, ac-
companied by separate vector indices for raw rows and claims.
Within this testbed, the baseline multi-strategyEnsembleRetrievercombines dense similarity
search, structural graph expansion, and semantic claim matching using Reciprocal Rank Fusion. ECR
cleanly intercepts the pipeline immediately after candidate generation, acting as an isolated inference-
time uncertainty resolution stage that outputs either a dominant hypothesis or a calibrated set of
alternatives for downstream synthesis.
9

4.2 CSGR++ Backbone Architecture
While ECR is algorithmically orthogonal to a particular retrieval stack, we implement and evaluate
it inside a production-grade pipeline (CSGR++) that is explicitly claim-centric.
Atomic claim store with semantic relations.CSGR++ stores extracted claims in a SQLite-
backedClaimStorewith fields for (i) claim text, (ii) entity mentions, (iii) time keys / order keys for
temporal slicing, and (iv) dynamically updated confidence signals. In addition, a lightweight seman-
tic relation table stores tuples(h, r, t)extracted during claim extraction (e.g.,Acquires,Impacts,
CausedBy), enabling entity-based expansion during retrieval.
Temporal intelligence.Queries are parsed for explicit time constraints (e.g., “in 2024”, “2024Q1”,
“last 3 quarters”, “since 2022”) and converted into an order-key interval(τ min, τmax). Claim retrieval
can then apply a hard filter over the claim IDs inside the selected time window.
Competitive ensemble retrieval and Reciprocal Rank Fusion (RRF).The retriever runs
multiple strategies (vector retrieval over rows, vector retrieval over claims, and graph/category traver-
sal) and fuses the per-strategy rankings via Reciprocal Rank Fusion (RRF). For an itemdand ranking
lists{L j}m
j=1with ranksr j(d)∈ {1,2, . . .}, the fused score is
RRF(d) =mX
j=11
k+r j(d),(9)
wherekis a dampening constant (we usek= 60in code).
Competitive strategy scoring (selection, not only fusion).In addition to fusing rankings, the
retriever scores each strategy to identify a “best” strategy for the query. The implemented scoring
combines (i) average similarity score, (ii) a diversity proxy based on unique source items, and (iii)
average claim confidence (when applicable) via a weighted sum.
Beyondrankfusion,thisstrategyscoringidentifiesthedominantevidenceviewforaquery,enabling
adaptive retrieval-path selection rather than blindly trusting an ensemble.
Relation-basedexpansionformulti-hopanalyticalqueries.Foranalytical(CSGR++)queries,
the system extracts frequent entities from initially retrieved claims, then expands the evidence set by
retrieving related claims via the relation table (one-hop expansion), discounting confidence slightly
for expanded claims.
Dynamicconfidencemicro-learning.Claimsmaintainsupportandcontradictioncounters. When
verification indicates a claim was supported or contradicted, the system updates its confidence with
a bounded, asymmetric rule:
conf new(c) =clip[0,1]
conf base(c) + 0.15 log(1 +S(c))−0.25C(c)
.(10)
This produces an online “micro-learning” effect: frequently supported claims become easier to trust,
while contradicted claims are rapidly down-weighted.
Because claim confidence is updated online and directly affects future [EER(c)scores (Eq. 4), ECR
exhibits lightweight inference-time learning behavior across queries.
Trust modes (graded verification).The query router classifies user intent into trust modes
(strictfor regulatory or numerical precision,balanced, andexploratory), which modulate verification
aggressiveness and synthesis style.
10

Table 1: Key subsystems implemented in our system that support ECR and the full end-to-end
pipeline.
Subsystem Role in the pipeline
AutoAdaptAgent + SchemaFeedbackLoop Schema inference with dry-run calibration
GenericStore (EAV SQLite) Item/attribute persistence; safeSELECT-only SQL ex-
ecution
EmbeddingProvider + VectorIndex 3-tier embeddings; LanceDB/NumPy backends; ID-
filtered cosine search
ClaimExtractor + ClaimStore Atomicclaims+relations+temporalkeys+dynamic
confidence
EnsembleRetriever Competitive retrieval + RRF fusion (Eq. 9)
EntropicClaimResolver ECR loop: entropy, EER selection (Eq. 4)
StructuredSynthesizer Structured analytical brief with evidence bullets
ReverseVerifier Numeric grounding + claim-aware verification and
score capping
RAGAnswerer Multi-hop, HyDE,text-to-SQLgrounding, CRAGself-
correction, citations
ReverseVerifier: deterministic numeric grounding + claim-aware checking.Beyond prob-
abilistic resolution, CSGR++ applies a three-layer ReverseVerifier: (i) a deterministic numeric ground-
ing pass that extracts all numeric tokens in a draft answer and checks verbatim presence in retrieved
evidence, (ii) LLM-based claim-by-claim judgement with both supporting and counter-evidence re-
trieval, and (iii) a combined score where numeric failures cap the maximum achievable verification
score. Numeric grounding is enforced as a hard constraint: a single unsupported numeric token caps
downstream verification scores.
Table 1 summarizes the major subsystems of the full HyRAG v3 and CSGR++ pipeline and their
respective roles, providing a compact overview of how ECR integrates into the surrounding retrieval,
verification, and synthesis infrastructure.
4.3 Supporting RAG Components
Outside the CSGR++ analytical route, the implementation includes a general-purpose RAG engine
that packages standard, widely used RAG mechanisms behind a singleanswer()interface. These
components are supporting infrastructure and are orthogonal to ECR.
The system also supports generator-based streaming responses (viaanswer_streamentry-points),
which is orthogonal to ECR and not evaluated in this work.2
Multi-hop retrieve–reason–retrieve.The engine iteratively retrieves candidates and, when on-
line, generates a follow-up query conditioned on current evidence, stopping early when additional hops
yield no new items.
HyDE query embedding.To improve recall under distribution shift between user queries and row-
shaped embeddings, the engine optionally generates a short hypothetical “answer row” and embeds
that text (HyDE) to drive vector search.
Cross-encoder reranking and calibrated abstention.Candidates are reranked either by cosine
similarity (offline) or by an LLM “cross-encoder” that outputs a ranking and confidence. A calibrated
2All major components admit deterministic fallbacks when LLMs are unavailable (e.g., hashed embeddings and
heuristic claim extraction), though answer quality may degrade.
11

confidence score combines the number of retrieved results, the top similarity score, the reranker
confidence, and a query-complexity penalty; the system abstains when the calibrated score is low.
Text-to-SQL with value grounding.For aggregation queries, the engine routes to text-to-SQL
andappliesasecondgroundingpassthatvalidateseverygeneratedstringliteralagainstrealcategorical
values in the database; when an unknown literal is detected, it is rewritten to the closest fuzzy match
when possible.
CRAG self-correction with schema evolution signals.When reverse verification returnsfail
orweak, the engine performs up to two correction attempts by rewriting the query to target the
verification gap. Each failure can be recorded by a schema evolution tracker that increments per-
column failure counts and can request LLM-based reclassification suggestions once a threshold is
exceeded. Schema evolution signals persist across queries, enabling long-term self-correction.
5 Experimental Design & Evaluation
While Section 4 outlines the deployment of ECR within a full-scale production architecture, evaluating
the algorithm end-to-end immediately introduces confounding variables from upstream retrieval recall
and downstream LLM generation quality. To rigorously validate the decision-theoretic properties
established in Section 3, our evaluation strategy proceeds in two phases. First, we strictly isolate
the mathematical behavior of the entropy-driven claim selection policy using a controlled, claims-only
harness (Sections 5.1–5.3). Second, we reintegrate ECR into an end-to-end reasoning pipeline to
evaluate its impact under realistic multi-hop and contradiction-heavy settings (Section 5.4).
5.1 Controlled claims-only harness
Our “claims-only” harness fixes the dataset, query set, retrieval configuration, candidate claim pool,
and Bayesian entropy model; only the claim-selection policy differs. This allows a clean measurement
of whether a policy is actually minimizing epistemic uncertainty as defined by Eq. 1.
Datasetandcases.Weuseasmall,multi-tablebusinessdatasetofsixCSVtables(sales,customers,
expenses,inventory,hr,marketing) and 80 templated evaluation queries spanning single-table
lookups and cross-table comparisons.
Hypotheses and initial entropy.For each query, the harness constructs|A|= 3mutually exclu-
sive answer hypotheses, yielding an initial entropy ofH 0= log23≈1.585bits.
Candidate claims and policies.For each case, we retrieve the same top-20 candidate claims
(high-recall candidate generation). We then compare three policies: (i)Retrieval-only, which takes
the top-15 claims by retrieval score under a fixed budget; (ii)ECR, which sequentially selects the
next claim by expected entropy reduction and stops whenH≤ϵwithϵ= 0.3bits (capped at 10
iterations); and (iii)Random control, which samples claims uniformly without replacement from
the same candidate pool, matching ECR’s realized claim budget of 5 claims.
Entropy-aligned metrics.We report (i) final entropy, (ii) entropy drop per evaluated claim, (iii)
claims-to-collapse (first step reachingH≤ϵ, else budget+1), (iv) effective hypotheses (2H), and (v)
entropy trace variance. We additionally report two diversity-oriented diagnostics (claim redundancy
and source entropy) to illustrate that diversity alone is not equivalent to epistemic resolution. Finally,
we reporthypothesis-conditioned redundancy(HypCondRed.), which computes redundancy within
claim groups attributed to the same answer hypothesis (rather than across the full mixed set).
12

Policy ClaimsH final ∆H/claim Collapse2Hfinal Redund. HypCondRed. SrcEnt
Retrieval-only15.0±0.0 1.585±0.000 0.0000±0.0000 16.0±0.0 3.000±0.000 0.684±0.119 0.662±0.110 0.342±0.510
ECR5.0±0.0 0.2129±0.0000 0.2744±0.0000 5.0±0.0 1.159±0.000 0.672±0.125 0.672±0.125 0.276±0.443
Random5.0±0.0 1.243±0.289 0.0684±0.0577 6.0±0.0 2.411±0.437 0.658±0.118 0.653±0.123 0.354±0.527
Table 2: Claims-only evaluation (80 cases, seed=7). “Claims” is the number of evaluated claims.
Hfinalis final answer-hypothesis entropy in bits. “Collapse” is claims-to-collapse (first step where
H≤ϵ= 0.3; else budget+1). “Redund.” is claim redundancy, “HypCondRed.” is hypothesis-
conditioned claim redundancy, and “SrcEnt” is source entropy (diagnostic diversity metrics).
PolicyH final ∆H/claim Collapse2Hfinal TraceVar HypCondRed.
Retrieval-only1.585±0.000 0.0000±0.0000 16.00±0.00 3.000±0.000 0.002987±0.000000 0.6619±0.0000
ECR0.2129±0.0000 0.2744±0.0000 5.00±0.00 1.159±0.000 0.262859±0.000000 0.6719±0.0000
Random1.2628±0.0265 0.0644±0.0053 5.995±0.006 2.436±0.044 0.03210±0.00374 0.6401±0.0075
Table 3: Multi-seed robustness (seeds 0–4): mean±std over seeds of the seed-level mean metrics.
Only the random baseline changes across seeds in this frozen setup. “HypCondRed.” is hypothesis-
conditioned claim redundancy.
5.2 Main results (seed=7, 80 cases)
Table 2 summarizes mean±std across cases. ECR reliably reaches epistemic sufficiency using 5
claims, drivingHbelowϵ; retrieval-only does not reduce entropy under the same posterior model;
random improves modestly but typically does not collapse.
Across these runs, claim-coverage is identical across policies (0.6375 on average), reflecting that
this harness is designed to stress epistemic resolution rather than maximize overlap with a small set
of expected claim snippets.
5.3 Robustness across random seeds (seeds 0–4)
Toensuretherandom-controlcomparisonisnotasingle-seedartifact, wereruntheclaims-onlyharness
for five random seeds (0–4), reusing the same frozen dataset, query set, candidate claims, and posterior
model. Retrieval-only and ECR are deterministic under this setup, while the random baseline varies
by construction.
Table 3 confirms the stability of ECR across multiple seeds. Furthermore, Figure 3 illustrates the
schematic entropy trajectories of these competing policies, highlighting how rapidly ECR drives the
hypothesis space below theϵthreshold compared to relevance-only baselines.
5.4 End-to-End Evaluation on a Standard Multi-Hop QA Benchmark
In contrast to the preceding controlled, claims-only experiments, this evaluation reintegrates a live
large language model into the inference loop, exercising ECR as an online evidence-selection controller
during end-to-end RAG generation. As an additional experiment, to evaluate whether entropy-guided
evidence selection improves downstream answer quality, we conduct an end-to-end evaluation on a
HotpotQA-style multi-hop QA benchmark. All methods share the same retriever, language model,
candidate evidence pool, and decoding parameters; the only variable is the inference-time claim se-
lection policy. We evaluate three policies: (i) a relevance-based baseline RAG policy, (ii) a random
selection control matched to the same average claim budget, and (iii) ECR, which applies entropy-
guided selection with stopping. We report exact match (EM), token F1, and an evidence faithfulness
proxy based on answer-token coverage, alongside the average number of claims used. Because Hot-
potQA exhibits substantially higher linguistic variance and more complex multi-hop dependencies
13

Evidence Claims Actively Evaluated (t)Hypothesis Entropy
H(A|X eval)(bits)
0123456789100.00.30.60.91.21.51.8
Epistemic sufficiency (ϵ= 0.3)ECR (schematic)
Random/retrieval (schematic)
Figure 3: Schematic entropy trajectories consistent with the measured endpoints: ECR reachesH≤ϵ
quickly, whereas relevance-only and random baselines typically remain aboveϵat matched claim
budgets.
than highly structured tabular datasets, the ECR algorithm naturally evaluates a larger number of
claims before the hypothesis entropy collapses belowϵ.
Method Avg. Claims Used Exact Match (EM)↑Token F1↑Evidence Faithfulness↑
Baseline RAG 19.87 0.313 0.459 0.639
Random Control 19.87 0.207 0.307 0.427
ECR (ours) 19.68 0.297 0.450 0.626
Table 4: End-to-End Evaluation on HotpotQA-Style Multi-Hop QA (300 Questions). All methods use
the same retriever and language model; only the inference-time evidence selection policy differs. ECR
substantially outperforms random selection while maintaining performance comparable to a strong
relevance-based baseline.
Table 4 shows that ECR substantially outperforms random selection across all reported metrics,
confirming that entropy-guided evidence selection is consistently more effective than unguided or
diversity-only strategies. Relative to a strong relevance-based baseline, ECR remains within a small
margin on EM and F1, indicating that enforcing epistemic control does not significantly degrade
answer accuracy on standard benchmarks.
It is important to note that HotpotQA is a largely factual and relevance-oriented benchmark
with predominantly singular ground truths. As such, it does not natively stress-test contradictory
evidence or fundamentally ambiguous queries, which are precisely the regimes ECR is designed to
address. Achieving near parity on such a saturated benchmark while enforcing strict inference-time
epistemic constraints demonstrates that ECR integrates robust uncertainty control without reliance
on benchmark-specific tuning. Future evaluations will focus on conflict-heavy or ambiguity-oriented
benchmarks where relevance-driven retrieval is known to exhibit epistemic collapse.
Robustness to Noisy Evidence.To isolate a regime that is closer to real deployments, where
retrieved evidence may include irrelevant or even contradictory content, we perform a controlled
ablation on the same HotpotQA evaluation set and pipeline as above, injecting noiseafter retrieval
14

Method EM (No Noise) Faith (No Noise) EM (40% Noise) Faith (40% Noise)
Baseline RAG 0.323 0.660 0.167 0.345
ECR (ours) 0.307 0.657 0.163 0.331
Table 5: Robustness ablation on HotpotQA-style evaluation (300 questions) with noise injectedafter
retrieval and before evidence selection. “40% Noise” replaces 40% of retrieved candidate claims with
unrelated (potentially contradictory) claims sampled from a noise pool. Only baseline relevance-based
RAG and ECR are evaluated; the retriever, LLM, prompts, and decoding are unchanged. (Perfor-
mance is bounded above when ground-truth evidence is removed.)
and before evidence selection3. For each query, we take the retrieved candidate claim set and replace
40% of candidates with claims sampled from a noise pool constructed from unrelated documents
(keeping the retriever, LLM, prompts, decoding, and ECR selection logic unchanged). Table 5 reports
Exact Match (EM) and Evidence Faithfulness for baseline relevance-based RAG and ECR under no
noise versus 40% noise.
Under this corruption regime, Exact Match necessarily degrades for both systems, as replacing a
fraction of candidate claims can remove ground-truth evidence from the pool. Notably, ECR exhibits
predictable degradation to the relevance-based baseline without amplifying noise-induced errors, de-
spite enforcing strict inference-time stopping and evaluating fewer claims. This result indicates that
entropy-guided evidence selection remains well-behaved under partial evidence loss, avoiding overconfi-
dent hallucination or unstable collapse when the available evidence becomes incomplete or unreliable.
We emphasize that this ablation evaluates robustness to evidence corruption (i.e., partial removal
of valid claims), rather than distractor accumulation, which isolates a complementary but distinct
failure mode.
Offline Robustness Under Structured Contradiction.Standard QA benchmarks predomi-
nantlyevaluateansweraccuracyunderrelativelycleanevidenceconditions. Tostress-testtheepistemic-
control mechanism itself—independently of LLM semantics—we run a fully offline, deterministic
contradiction-injection ablation on the same 300-question HotpotQA-style set and retrieval pipeline.
For each query, we take the retrieved candidate claim pool and inject paired, explicit contradiction
twins into the candidate set at rateα∈ {0.0,0.3,0.5}after retrieval and before evidence selection.
In offline mode, hypothesis initialization uses deterministic hashed embeddings and claim verification
uses a deterministic provenance proxy; this isolates controller behavior from verifier quality.
We report (i)Ambiguity Exposure—whether the run ends withH > ϵor an unresolved explicit
contradiction pair—and (ii)Overconfident Error—cases where the system outputs a dominant
hypothesis despite being wrong (a proxy for epistemic collapse). Table 6 shows a sharp regime shift:
baseline relevance-based RAG remains pathologically overconfident and flat acrossα, while ECR
transitions from fast epistemic sufficiency in the clean regime (α= 0.0) to principled non-convergence
under contradiction (α≥0.3). Atα≥0.3, ambiguity emerges deterministically for every query
and termination is entirely explained by unresolved conflict rather than heuristic budget limits. This
extreme ambiguity rate is expected: once an explicit contradiction pair is present in the evaluated
evidence, epistemic coherence is unattainable by definition. Likewise, entropy remains high because
ECR is not an entropy minimizer “at all costs”; it is a coherence-constrained entropy controller.
Exploring complementary ambiguity-focused benchmarks and distractor accumulation regimes re-
mains an important direction for future evaluation.
3Because noise is injected by replacing a fraction of candidate claims, this protocol may remove gold evidence for
some queries. Consequently, Exact Match under heavy corruption reflects robustness to partial evidence loss rather
than distractor filtering.
15

MethodαEM OverconfErr AmbExp MeanHStop Reason
Baseline RAG 0.0 0.0067 0.9933 0.0000 – fixed_budget (300/300)
Baseline RAG 0.3 0.0067 0.9933 0.0000 – fixed_budget (300/300)
Baseline RAG 0.5 0.0067 0.9933 0.0000 – fixed_budget (300/300)
ECR (ours) 0.0 0.0000 0.9900 0.0100 0.226 epistemic_sufficiency (297/300)
ECR (ours) 0.3 0.0067 0.0000 1.0000 1.496 unresolved_conflict (300/300)
ECR (ours) 0.5 0.0067 0.0000 1.0000 1.458 unresolved_conflict (300/300)
Table 6: Offline contradiction-injection ablation (300 questions). Paired contradictions are injected
into the candidate claim pool at rateαafter retrieval and before evidence selection. EM is reported
only as a sanity anchor under a deterministic offline answerer; the key signals are Ambiguity Expo-
sure and Overconfident Error (epistemic collapse). ECR exhibits a phase transition from epistemic
sufficiency to principled non-convergence as contradictions accumulate, while baseline RAG remains
uniformly overconfident. Counts indicate number of runs terminating for each reason.
6 Conclusion
Summary
Entropic Claim Resolution introduces a principled inference-time perspective on Retrieval-Augmented
Generation, reframing evidence selection as a process of epistemic uncertainty reduction rather than
relevancemaximization. BydirectlyoptimizingExpectedEntropyReductionoveratomicclaims, ECR
provides a mathematically grounded mechanism for determining both which evidence to evaluate next
and when sufficient evidence has been accumulated to justify synthesis.
Empirically, we show that this entropy-driven framework reliably collapses hypothesis uncertainty
in controlled claim-level settings and substantially outperforms random evidence selection in end-to-
end multi-hop question answering, while maintaining performance comparable to strong relevance-
based baselines. These results highlight a fundamental distinction between optimizing for raw answer
accuracy and enforcing principled epistemic control during inference.
In a fully offline contradiction-injection stress test, ECR exhibits a sharp transition from epistemic
sufficiencytoprinciplednon-convergenceasstructuredconflictaccumulates: entropyceasestocollapse,
evidence exploration increases, and termination is explained by unresolved inconsistency rather than
heuristic budgets.
Unlikeretrievalarchitecturesdesignedprimarilyforlong-formunstructureddocuments, HyRAGv3
explicitly models structured tabular data with row-level grounding, enabling ECR to enforce numeric
correctness and temporal consistency during inference.
Beyond benchmark performance, the ECR framework offers clear advantages for real-world and
enterprise deployments. In high-stakes domains such as medicine, law, and finance, confidently synthe-
sizing a single answer from conflicting or incomplete evidence can be costly or harmful. By providing a
mathematically grounded mechanism to expose unresolved ambiguity when epistemic sufficiency can-
not be reached, ECR functions as a principled constraint against unhedged generation. Furthermore,
the ability to dynamically halt evidence accumulation onceH≤ϵis satisfied mitigates unnecessary
computational overhead, reducing latency and cost associated with processing large, redundant con-
text windows. This positions ECR as a resource-efficient inference-time control mechanism for scalable
and risk-aware AI reasoning.
Limitations and Future Work
We conclude by outlining key limitations of the current framework and highlighting promising direc-
tions for future research.
16

Hypothesis space coverage.A primary limitation of the current framework is its reliance on the
initial hypothesis generation stage. Entropic Claim Resolution operates over an explicitly constructed
hypothesis set and therefore inherits a bounded-coverage assumption; if the true answer is entirely
absent from this space, the system may converge confidently to an incorrect explanation. In practice,
this limitation can be mitigated by regenerating hypotheses when entropy fails to decrease or when
accumulated evidence weakly supports all candidates. Future work will explore dynamic mid-loop
hypothesis extension, soft hypothesis assignments, richer likelihood models, and tighter integration
with learned retrievers to further strengthen entropy-guided reasoning under uncertainty. Importantly,
ECR’s refusal to converge under explicit contradiction is a deliberate design choice rather than a
limitation: when the evaluated evidence is epistemically incoherent, the framework exposes ambiguity
instead of forcing posterior collapse. This behavior preserves epistemic correctness but may yield
non-decisive outputs in genuinely inconsistent corpora.
Anorthogonalrobustnessregimeinvolvesdistractoraccumulationwithoutevidenceremoval, which
we leave to future investigation.
Finally,whilethisworkevaluatesEntropicClaimResolutionspecificallywithinRetrieval-Augmented
Generation, the underlying methodology naturally extends to agentic and autonomous contexts. Our
approach suggests a perspective where agent actions (such as executing tools, querying external APIs,
or taking exploratory steps) can be modeled dynamically as entropy-minimizing decisions evaluated
under a rigorous Expected Entropy Reduction criterion. This aligns with recent advancements in
autonomous cognitive control, including topology-aware routing [19] and dynamic temporal pacing
[20], providing a formal alternative to standard prompt-driven or heuristic action-selection policies.
We view the integration of decision-theoretic primitives into continuous agentic feedback loops as a
compelling frontier for building robust and mathematically grounded autonomous systems.
17

References
[1] Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive nlp tasks.Ad-
vances in Neural Information Processing Systems, 33, 9459–9474.
[2] Karpukhin, V., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering.
Proceedings of EMNLP.
[3] Edge, D., et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused
Summarization.arXiv preprint arXiv:2404.16130.
[4] Singh, R., et al. (2026). CSR-RAG: An Efficient Retrieval System for Text-to-SQL on the Enter-
prise Scale.arXiv preprint arXiv:2601.06564.
[5] Yao, S., et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models.ICLR.
[6] Yao, S., etal.(2023). TreeofThoughts: DeliberateProblemSolvingwithLargeLanguageModels.
NeurIPS.
[7] Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique.arXiv preprint
arXiv:2310.11511.
[8] Shannon, C. E. (1948). A Mathematical Theory of Communication.Bell System Technical
Journal, 27(3).
[9] Houlsby, N., et al. (2011). Bayesian Active Learning for Classification and Preference Learning.
arXiv:1112.5745.
[10] Kuhn, L., etal.(2023). SemanticUncertainty: EpistemicUncertaintyinNeuralLanguageModels.
ICLR.
[11] Zhang, T., et al. (2024). URAG: Benchmarking Uncertainty in Retrieval-Augmented Generation.
arXiv:2408.01234.
[12] Liu, H., et al. (2025). Retrieval-Augmented Reasoning Consistency.ACL.
[13] Zhao, X., et al. (2025). BEE-RAG: Balanced Entropy-Engineered Context Management.
arXiv:2501.09912.
[14] Kim, S., et al. (2025). Structure-Fidelity RAG.ICLR.
[15] Patel, M., et al. (2024). Lazy RAG.EMNLP.
[16] Wang, Y., et al. (2025). MedRAGChecker.arXiv:2502.10423.
[17] Wei, A., et al. (2024). SAFE.arXiv:2403.18802.
[18] Chen, J., et al. (2025). CIBER.NeurIPS.
[19] Di Gioia, D. (2026). Cascade-Aware Multi-Agent Routing.arXiv:2603.17112.
[20] Di Gioia, D. (2026). Learning When to Act.arXiv:2603.22384.
18

Appendix
Aλ-Sweep Robustness
Totestwhethercoherence-awarebehaviorrequiresfragiletuning, wesweepthecoherencebonusweight
λin ECR’s evidence selection policy over{0,0.01,0.025,0.05,0.1}while keeping the offline protocol,
budgets, and contradiction injection ratesα∈ {0.0,0.3,0.5}fixed. We observe thatλ= 0behaves as
entropy-only control and can converge to a dominant hypothesis even under contradiction injection,
whereas any tested non-zeroλyields the same coherence-aware regime in which explicit contradictions
are surfaced and prevent epistemic collapse; consequently, behavior saturates across all testedλ >0.
We setλ= 0.05as the default. As shown in Table A.1, we observe a sharp transition between
entropy-only control (λ= 0) and coherence-aware control (λ >0), with behavior saturating for all
tested non-zero values. This indicates that ECR does not require fine-grained hyperparameter tuning
to surface epistemic inconsistency.
λconflictAmbiguity Exposure
00.01 0.025 0.05 0.1001 α= 0.0
α= 0.3
α= 0.5
Figure A.1: Ambiguity exposure as a function of the coherence weightλ conflictunder structured
contradictioninjection. Empirically, ambiguityexposureexhibitsasharpphasetransition: forα= 0.5,
exposurejumpsfrom0to1foranytestedλ >0, whileremaining0forα≤0.3acrossalltestedsettings.
λ α= 0.0α= 0.3α= 0.5
MeanClaims MeanHMeanClaims MeanHMeanClaims MeanH
0.00 5.04 0.226 5.06 0.226 5.08 0.226
0.01 5.04 0.226 25.83 1.496 29.81 1.458
0.025 5.04 0.226 25.83 1.496 29.81 1.458
0.05 5.04 0.226 25.83 1.496 29.81 1.458
0.10 5.04 0.226 25.83 1.496 29.81 1.458
Table A.1:λ-sweep summary statistics (offline, deterministic). Values are aggregated over 300 ques-
tions.
B Offline Contradiction Sanity Test
As an additional sanity check, we evaluate ECR in a minimal fully offline scenario consisting of a single
claimanditsexplicitsyntheticnegation(apairedcontradictiontwin). Inthissetting, theexpectedout-
come is that ECR evaluates both claims, flags unresolved conflict (has_unresolved_conflict=True),
and refuses to emit a dominant hypothesis (dominant_hypothesis=None). This deterministic unit
test passes.
19