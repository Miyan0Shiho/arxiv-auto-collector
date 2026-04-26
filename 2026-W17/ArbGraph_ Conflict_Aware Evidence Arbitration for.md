# ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation

**Authors**: Qingying Niu, Yuhao Wang, Ruiyang Ren, Bohui Fang, Wayne Xin Zhao

**Published**: 2026-04-20 14:51:43

**PDF URL**: [https://arxiv.org/pdf/2604.18362v1](https://arxiv.org/pdf/2604.18362v1)

## Abstract
Retrieval-augmented generation (RAG) remains unreliable in long-form settings, where retrieved evidence is noisy or contradictory, making it difficult for RAG pipelines to maintain factual consistency. Existing approaches focus on retrieval expansion or verification during generation, leaving conflict resolution entangled with generation. To address this limitation, we propose ArbGraph, a framework for pre-generation evidence arbitration in long-form RAG that explicitly resolves factual conflicts. ArbGraph decomposes retrieved documents into atomic claims and organizes them into a conflict-aware evidence graph with explicit support and contradiction relations. On top of this graph, we introduce an intensity-driven iterative arbitration mechanism that propagates credibility signals through evidence interactions, enabling the system to suppress unreliable and inconsistent claims before final generation. In this way, ArbGraph separates evidence validation from text generation and provides a coherent evidence foundation for downstream long-form generation. We evaluate ArbGraph on two widely used long-form RAG benchmarks, LongFact and RAGChecker, using multiple large language model backbones. Experimental results show that ArbGraph consistently improves factual recall and information density while reducing hallucinations and sensitivity to retrieval noise. Additional analyses show that these gains are evident under conflicting or ambiguous evidence, highlighting the effectiveness of evidence-level conflict resolution for improving the reliability of long-form RAG. The implementation is publicly available at https://github.com/1212Judy/ArbGraph.

## Full Text


<!-- PDF content starts -->

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable
Long-Form Retrieval-Augmented Generation
QINGYING NIU∗,Renmin University of China, China
YUHAO WANG∗,Renmin University of China, China
RUIYANG REN†,Renmin University of China, China
BOHUI FANG,Carnegie Mellon University, USA
WAYNE XIN ZHAO†,Renmin University of China, China
Retrieval-augmented generation (RAG) remains unreliable in long-form settings, where retrieved evidence
is noisy or contradictory, making it difficult for RAG pipelines to maintain factual consistency. Existing
approaches focus on retrieval expansion or verification during generation, leaving conflict resolution entangled
with generation. To address this limitation, we propose ArbGraph, a framework for pre-generation evidence
arbitration in long-form RAG that explicitly resolves factual conflicts. ArbGraph decomposes retrieved
documents into atomic claims and organizes them into a conflict-aware evidence graph with explicit support
and contradiction relations. On top of this graph, we introduce an intensity-driven iterative arbitration
mechanism that propagates credibility signals through evidence interactions, enabling the system to suppress
unreliable and inconsistent claims before final generation. In this way, ArbGraph separates evidence validation
from text generation and provides a coherent evidence foundation for downstream long-form generation. We
evaluate ArbGraph on two widely used long-form RAG benchmarks, LongFact and RAGChecker, using multiple
large language model backbones. Experimental results show that ArbGraph consistently improves factual recall
and information density while reducing hallucinations and sensitivity to retrieval noise. Additional analyses
show that these gains are evident under conflicting or ambiguous evidence, highlighting the effectiveness
of evidence-level conflict resolution for improving the reliability of long-form RAG. The implementation is
publicly available at https://github.com/1212Judy/ArbGraph.
Additional Key Words and Phrases: Evidence arbitration; Retrieval-augmented generation; Long-form genera-
tion; Evidence graphs; Large language models
ACM Reference Format:
Qingying Niu, Yuhao Wang, Ruiyang Ren, Bohui Fang, and Wayne Xin Zhao. 2026. ArbGraph: Conflict-Aware
Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation.ACM Trans. Inf. Syst.1, 1
(April 2026), 23 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn
∗Equal Contribution.
†Corresponding Authors.
Authors’ addresses: Qingying Niu, qingyingniu36@gmail.com, Gaoling School of Artificial Intelligence, Renmin University
of China, Beijing, China, 100872; Yuhao Wang, yh.wang@ruc.edu.cn, Gaoling School of Artificial Intelligence, Renmin
University of China, Beijing, China, 100872; Ruiyang Ren, reyon_ren@outlook.com, Gaoling School of Artificial Intelligence,
Renmin University of China, Beijing, China, 100872; Bohui Fang, bohuif@alumni.cmu.edu, Carnegie Mellon University,
Pittsburgh, Pennsylvania, USA, 15213; Wayne Xin Zhao, batmanfly@gmail.com, Gaoling School of Artificial Intelligence,
Renmin University of China, Beijing, China, 100872.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee
provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and
the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored.
Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires
prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2026 Association for Computing Machinery.
1046-8188/2026/4-ART $15.00
https://doi.org/10.1145/nnnnnnn.nnnnnnn
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.arXiv:2604.18362v1  [cs.CL]  20 Apr 2026

2 Niu et al.
1 INTRODUCTION
Retrieval-Augmented Generation (RAG) has emerged as a widely used paradigm for grounding
large language models in external knowledge [ 7,17,26,29]. However, its reliability depends not
only on whether relevant evidence can be retrieved, but also on whether that evidence can be
consolidated into a coherent factual basis for generation [ 42]. These challenges are especially
pronounced in long-form settings, where models must synthesize multiple interdependent facts
into extended responses rather than produce isolated short answers [ 19]. In such contexts, factual
errors rarely remain local. Instead, noisy, redundant or mutually inconsistent evidence can distort
the evolving discourse structure, allowing early mistakes to propagate across subsequent claims
and ultimately undermine global factual coherence [ 14,37]. Simply increasing retrieval depth does
not necessarily improve reliability, since additional retrieved evidence often introduces further
inconsistencies rather than greater certainty that complicate downstream reasoning.
Typically, attempts to improve long-form factuality have largely proceeded along two lines.
One seeks to make generation itself more reliable, through self-evaluation, iterative refinement or
agentic reasoning that attempts to detect and correct errors as the answer unfolds [ 1,13,31,38,39].
The other makes evidence more structured, organizing retrieved content into semantic or entity-
level representations that improve retrieval coverage and reasoning connectivity [ 2,4,11,46]. Both
directions have yielded important gains, yet both stop short of resolving the same core problem: they
do not explicitly determine the credibility of competing evidence before synthesis. Consequently,
factual conflict is handled either implicitly during decoding or only indirectly through structural
organization, rather than through a direct decision over which claims should be trusted. Under
noisy or contradictory retrieval, this limitation becomes critical. Generation-time correction acts
only after potentially unreliable evidence has already shaped the response, whereas structural
connectivity alone cannot determine which of several plausible but incompatible claims is correct.
As a result, although both strategies improve robustness, neither fully addresses the challenge
of maintaining factual consistency in long-form generation when retrieved evidence is noisy or
contradictory.
We argue that this limitation reflects a deeper mismatch in how long-form RAG currently handles
evidence conflict. During generation, decoding is driven primarily by token-level continuation,
whereas resolving factual conflict requires explicit comparison among competing pieces of evidence
under a broader view of their consistency and support. These two processes are not naturally
aligned. When retrieved evidence is noisy or contradictory, claims that are linguistically salient,
locally plausible or prominently positioned in context may influence generation even if they are
only weakly supported. Moreover, once such claims are incorporated early, subsequent decoding
can become conditioned on them, allowing local errors to propagate across the emerging discourse.
In long-form settings, where multiple interdependent claims must be integrated into a coherent
response, this mismatch becomes particularly consequential. These observations motivate a different
strategy: rather than relying primarily on implicit conflict handling during decoding, evidence
credibility should be assessed and competing claims should be resolved before generation begins.
To address these challenges, we proposeArbGraph, a framework for pre-generation evidence
arbitration in long-form RAG. Rather than leaving conflict handling to the generation process,
ArbGraph introduces an explicit arbitration stage between retrieval and synthesis, where competing
pieces of evidence are evaluated before they shape the final response. Specifically, ArbGraph
decomposes retrieved content into fine-grainedatomic claims, so that reasoning can be performed
over explicit factual units rather than coarse passages. It then organizes these claims into a conflict-
aware evidence graph with explicitsupportandcontradictionrelations. On top of this structure,
ArbGraph performs iterative arbitration to reinforce mutually supported claims while suppressing
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 3
claims that are weakly grounded or mutually inconsistent, yielding a more reliable evidence basis
for downstream generation. This process can be viewed as a form of structured evidence filtering,
where the credibility of each claim is not determined in isolation but emerges from its interactions
with other claims in the graph. The graph serves as an intermediate representation to encode
relationships, propagate constraints, and support conflict-aware reasoning. By shifting conflict
handling from the generation stage to the evidence level, ArbGraph transforms evidence selection
from a local, token-driven process into a global, structure-aware decision problem. This allows
conflict resolution to operate over a complete and consistent view of evidence, reducing sensitivity
to prompt structure, attention dynamics, and early decoding biases. As a result, ArbGraph provides
a more interpretable and stable foundation for long-form generation under noisy and contradictory
retrieval conditions.
Experimental results demonstrate that ArbGraph consistently improves factual recall, faithfulness,
and information density, while reducing hallucination and sensitivity to retrieval noise. Further
analysis shows that the proposed arbitration mechanism is particularly effective in scenarios
involving semantically similar but conflicting evidence or entity ambiguity, where local similarity
signals are often insufficient to disambiguate correctness. Overall, these findings indicate that
explicit evidence-level arbitration is a promising direction for improving the reliability of long-form
RAG under conflicting retrieval conditions.
Our main contributions are summarized as follows:
•Evidence-level arbitration paradigm:We formulate evidence arbitration before generation
as a distinct reliability mechanism for long-form RAG that moves conflict handling from
implicit decoding-time integration to explicit evidence-level decision-making, providing a
more stable foundation for downstream generation.
•Conflict-aware graph modeling:We construct a structured evidence graph over atomic
claims with explicit support and contradiction relations, enabling conflict-aware reasoning
under structural constraints. The graph serves as an intermediate representation to support
arbitration.
•Improved robustness in long-form generation:We conduct extensive experiments on
LongFact and RAGChecker across multiple backbones, showing that ArbGraph improves
factual robustness and grounding quality, particularly under noisy and conflicting retrieval
settings.
2 RELATED WORK
In this section, we review related work from three perspectives: long-form question answering
(LFQA), trustworthy RAG under conflicting evidence, and graph-based evidence modeling and
reasoning.
2.1 Long-Form Question Answering (LFQA)
LFQA research has evolved from early abstractive methods [ 5] to retrieval-augmented frame-
works that simulate human information seeking and integration, such as WebGPT [ 23] and We-
bCPM [ 25]. To improve verifiability, recent approaches have introduced citation-based supervi-
sion [ 22], post-hoc attribution mechanisms [ 6], and intent-aware attribution strategies for long-form
answering [ 45]. However, recent benchmarks show that LFQA systems remain vulnerable to noisy
retrieval [ 12,24], often producing globally inconsistent narratives even when local citations are
correct. These findings suggest that relying solely on source attribution is insufficient for ensuring
consistency in long-form generation, as attribution signals alone do not capture interactions or
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

4 Niu et al.
conflicts among multiple pieces of evidence [ 12,24,45]. Consequently, improving long-form fac-
tuality requires explicitly modeling relationships among evidence and resolving inconsistencies
across them. Our work addresses this limitation by performing explicit evidence-level reasoning
and resolving factual conflictsprior togeneration, rather than relying solely on attribution signals
during or after generation.
2.2 Trustworthy RAG with Conflicting Evidence
A major challenge for trustworthy RAG lies in handling knowledge conflicts, where retrieved
documents may contradict each other or the model’s internal knowledge. This issue is particularly
harmful in long-form generation, as errors in key claims can propagate and disrupt the entire
output [ 37]. Furthermore, irrelevant yet plausible context may distract the model, leading to
hallucinations or misinterpretation of evidence [8, 42].
Existing approaches attempt to mitigate these issues through generation-time or post-hoc mech-
anisms, including evidence-aware generation, source-level reliability estimation [ 15,28,34], and
evidence-driven methods [ 44], as well as verification-based methods such as Self-RAG [ 1], Chain-of-
Verification [ 3], Chain-of-Note [ 43], and TAPERA [ 46]. In addition, recent works explore claim-level
verification or falsification strategies, such as RAC [ 18] and FVA-RAG [ 27], as well as fine-tuning
approaches for improving reliability [ 16,35]. While these methods improve robustness in certain
scenarios, they share a common limitation: conflict handling is primarily performed during or after
generation, often relying on heuristic strategies or the model’s implicit reasoning ability.
As a result, conflict resolution remains entangled with the generation process and is sensitive
to prompt structure, attention allocation, and early decoding decisions, leading to instability
in long-horizon reasoning scenarios [ 33]. Moreover, these approaches typically lack an explicit
representation of evidence credibility and do not perform structured evaluation over competing
evidence. In contrast, our approach addresses this limitation by explicitly modeling and resolving
conflicts at the evidence level prior to generation, thereby decoupling conflict handling from text
synthesis and reducing reliance on implicit and often unstable generation-time reasoning.
2.3 Graph-based Evidence Modeling and Arbitration
Graph structures have been widely adopted to improve information organization and reasoning
in RAG systems, primarily for enhancing retrieval coverage and connectivity. Representative
methods such as GraphRAG [ 4] and HippoRAG [ 11] construct entity or semantic graphs to improve
information propagation and multi-hop retrieval. Meanwhile, reasoning-oriented frameworks such
as Graph of Thoughts (GoT) [ 2] and Think-on-Graph [ 32] leverage graph topologies to structure
inference, improving interpretability and faithfulness [21].
Recent work has also explored resolving factual conflicts via structured representations prior to
generation, including argumentation-based approaches [ 47] and knowledge-graph-based conflict
resolution methods [ 20]. These studies highlight the potential of graph-structured representations
for modeling complex evidence interactions and supporting more structured reasoning.
However, most existing methods use graphs primarily for retrieval expansion or structural orga-
nization, rather than as a mechanism for explicit conflict modeling and resolution. In particular, they
often lack explicit representations of contradiction edges and do not perform systematic evaluation
or arbitration among conflicting evidence nodes [ 4,11,20,47]. As a result, these approaches do
not explicitly decide between competing claims and are limited in their ability to directly resolve
inconsistencies in retrieved evidence.
In contrast, our work treats the graph as a computational framework for conflict-aware arbitration,
explicitly modeling support and contradiction relations and enabling structured decision-making
over competing evidence prior to generation. Rather than using graphs solely for connectivity or
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 5
propagation, ArbGraph leverages them to represent and resolve conflicts, producing a globally
consistent and reliable evidence foundation for long-form generation.
3 METHODOLOGY
In this section, we present the overall methodology of ArbGraph. We first introduce the long-form
RAG setting and provide an overview of the proposed framework. We then describe its three core
components: atomic claim extraction and semantic alignment, evidence graph construction, and
iterative credibility arbitration.
3.1 Framework Overview
We consider the long-form retrieval-augmented generation (RAG) setting, where a retriever
selects a set of evidence documents D𝑞for a query𝑞, and a generator produces a multi-paragraph
response grounded in these documents:
𝑦=G(𝑞,D 𝑞).
For each document𝑑∈D 𝑞, we decompose it into a set of atomic claims
C𝑑={𝑐 1,𝑐2,...,𝑐𝑛𝑑},
where each 𝑐𝑖represents a semantically irreducible and independently verifiable atomic claim. The
union of extracted atomic claims forms the initial claim pool for subsequent structured evidence
modeling.
To handle noisy, redundant, and potentially conflicting evidence in long-form RAG, we propose
ArbGraph, a framework that transforms the conventional linear RAG pipeline into anarbitrate-first,
generate-laterparadigm. Instead of implicitly integrating potentially conflicting evidence during
generation, ArbGraph explicitly performs evidence-level arbitration prior to decoding. This design
separatescredibility reasoningfromtext generation, enabling more robust handling of noisy and
contradictory evidence.
As illustrated in Figure 1, ArbGraph consists of three stages: (1) query-aware claim filtering, (2)
evidence graph construction, and (3) iterative credibility arbitration. These stages progressively
refine the evidence pool and produce a validated claim set Vval, which serves as the trusted context
for long-form generation. We briefly outline each stage below and defer implementation details to
subsequent sections.
Stage I: Query-Aware Claim Filtering.ArbGraph first converts retrieved documents into atomic
claims and filters them with respect to the query. This step removes background or weakly related
information, ensuring that subsequent reasoning operates on a focused and task-relevant evidence
set. By controlling the scale and relevance of claims early on, it also improves both efficiency and
reliability of later stages.
Stage II: Evidence Graph Construction.The filtered claims are organized into anEvidence Graph
that captures logical relationships between claims. In particular, the graph encodes both supportive
and contradictory interactions, providing an explicit structural view of evidence dependencies.
This representation enables reasoning beyond independent claim scoring by exposing how claims
reinforce or conflict with each other.
Stage III: Iterative Credibility Arbitration.Given the constructed graph, ArbGraph performs
iterative arbitration over conflicting claims to refine their credibility. The arbitration process
resolves high-impact conflicts using contextual evidence and progressively suppresses less reliable
claims. The remaining high-credibility claims form the validated set Vval, which is then used as
grounded input for final generation.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

6 Niu et al.
Query-A ware Claim Filtering
Query-A ware Evidence
Retrieval
Retrieval
Relevance 
Threshold
Vector  Space Embedding
c1 c2 c3...
0.2 0.9 c2c4c6
c5c3LLM-based verifier
classifies: Support/ Contradictc2 c3 c1 c3Candidate Pairs
Support Path ：
Top-M Support
Selector...
Contradiction Path:
All Conflicts
RetainedRound 1: Topology-driven
Scheduling
Prioritize claims with higher conflict
degree to resolve structural
bottlenecks first.
c3 c6
c5 c6
Intensity-driven Scheduling
Focus on high-credibility yet evenly
matched contradictions.
c3 c6c3
Win
Credibility Update & DecisionClaim
Extraction 
Alignmentc1 c2 c3...
0.8Evidence Graph Construction Iterative Cr edibility Arbitration
Query–claim Similarityc1 c2 c3📄
Cosine 
Similarityc1
G = ( V , E )
Contradiction SupportEvidence Graph 
e ij ∈ { Support  , Contradiction  }LLM-
based
Arbitrator deg(c5)=2deg(c3)=3
Fig. 1. Overview of the ArbGraph framework. The pipeline comprises three stages: query-aware claim filtering,
evidence graph construction, and iterative credibility arbitration, which together realize an arbitrate-first,
generate-later paradigm.
3.2 Atomic Claim Extraction and Semantic Alignment
The efficacy of evidence-level arbitration hinges on the precision of the underlying reasoning
units. Reasoning directly over coarse-grained passages often introduces linguistic noise and ob-
scures fine-grained factual dependencies, making it difficult to isolate specific points of contention.
ArbGraphaddresses this by decomposing evidence documents into atomic claims—the fundamen-
tal, irreducible units of information that facilitate explicit comparison and conflict resolution. This
decomposition is particularly vital because factual contradictions are typically localized within
specific attributes, such as temporal markers, geospatial coordinates, or quantitative values, which
are frequently embedded within dense, non-atomic textual contexts. This step establishes the basis
for query-aware claim filtering, while the final relevance-based selection is performed later during
node selection.
Atomic Decomposition.Given a set of retrieved documents D𝑞for a query 𝑞, we employ an
instruction-tuned Large Language Model (LLM) to transform each document𝑑∈D 𝑞into a set of
atomic claimsC𝑑={𝑐 1,...,𝑐𝑛𝑑}. Each claim is formulated as a self-contained assertion expressing
a single, independently verifiable fact. For instance, a sentence such as“Tesla was founded in 2003
and is headquartered in Austin”is bifurcated into two discrete units: one specifying the founding
year and another identifying the corporate headquarters.
The granularity of this decomposition is a critical determinant of the resulting evidence graph’s
topology. We recognize a fundamental trade-off: over-segmentation may fragment semantically
coherent statements into excessively sparse units, leading to a disconnected graph where relation
verification becomes numerically unstable; conversely, under-segmentation tends to entangle
multiple factual dimensions within a single node, thereby masking potential contradictions and
diluting the precision of the arbitration process. Consequently, ArbGraph is designed to produce
minimal yet self-containedfactual units that preserve semantic integrity while maximizing the
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 7
resolution of conflict detection. While LLM-based extraction may introduce occasional heuristic
errors, the systemic impact of such noise is attenuated in subsequent stages through multi-source
aggregation, which prioritizes consistent evidence patterns over idiosyncratic segmentation noise.
Semantic Alignment and Normalization.Information gathered from diverse sources often exhibits
significant lexical variance despite expressing identical propositions. To mitigate this redundancy,
we perform semantic normalization by mapping claims into a latent vector space using a sentence
encoder𝑓(·). Claims that exceed a conservative similarity threshold and maintain consistency
across key entities and attributes are merged into a unified canonical representation.
This normalization step is not merely a data compression technique but a prerequisite for valid
evidence weighting. Without it, the frequency of a claim’s appearance—often a byproduct of
document popularity or paraphrasing—could be misinterpreted as independent corroboration. By
consolidating these duplicates, we ensure that the perceived strength of a claim reflects a genuine
consensus across distinct information providers rather than the mere repetition of a single source’s
phrasing.
In the resulting evidence graph, each canonical cluster is instantiated as a singleclaim node
that aggregates all source attributions, while claims representing genuine factual disagreements
are intentionally preserved as distinct, competing nodes. This architecture explicitly decouples
redundancy from disagreement: support edges represent independent cross-source verification,
whereas contradiction edges isolate fundamental inconsistencies. This structural refinement en-
hances the density of meaningful interactions within the graph, providing a stable foundation for
the iterative credibility arbitration that follows.
3.3 Evidence Graph Construction
Given the normalized and query-filtered claim nodes, we construct anEvidence Graph
G𝐸=(V,E),
whereVdenotes canonical claim nodes and Erepresents verified logical relations between them.
This graph-based formulation enables explicit modeling of inter-claim dependencies, allowing
downstream arbitration to operate over structured evidence rather than independent claims.
Compared to flat aggregation or independent scoring approaches, the graph representation
explicitly captures how claims interact, making both agreement and conflict observable at the
structural level. This is particularly important in long-form RAG settings, where evidence may be
redundant, noisy, or mutually inconsistent.
We consider two types of relations:Support, indicating mutual reinforcement, andContradiction,
capturing logical conflicts. The edge set is thus partitioned asE=E sup∪E con.
The construction process consists of candidate pair mining and relation verification.
3.3.1 Query-Aware Node Selection.The node set Vis inherited from Stage I, where query-aware
filtering ensures that only task-relevant claims are retained. Specifically, we map the query 𝑞and
each claim𝑐𝑖into a shared embedding space via a sentence encoder 𝑓(·), and retain a claim as a
node if its semantic similarity to the query exceeds a predefined threshold:
cos 𝑓(𝑞),𝑓(𝑐𝑖)≥𝜏𝑞,(1)
where𝜏𝑞denotes the query relevance threshold.
This step effectively bounds the graph size and reduces the likelihood of introducing spurious
relations during edge construction. In addition to efficiency considerations, it also improves the
signal-to-noise ratioof the graph. Including irrelevant or weakly related claims would introduce
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

8 Niu et al.
noisy edges that may distort the structural patterns of support and contradiction, thereby degrading
the reliability of subsequent arbitration.
3.3.2 Threshold-Based Candidate Pair Mining.Naïve pairwise verification incurs quadratic complex-
ity (𝑂(|V|2)), which is impractical for large claim sets. To address this, we adopt a similarity-based
pruning strategy:
Pcand={(𝑐𝑖,𝑐𝑗)|cos(𝑓(𝑐 𝑖),𝑓(𝑐𝑗))≥𝜏 sim},(2)
where𝜏 simcontrols the semantic proximity required for further verification.
This design is motivated by the observation that meaningful logical interactions—whether
supportive or contradictory— typically arise among semantically related claims that refer to the
same entities or attributes. In contrast, semantically distant claims are unlikely to exhibit direct
logical relationships.
By restricting verification to local semantic neighborhoods, we significantly reduce computational
cost while preserving informative interactions. This also implicitly enforces a form ofstructural
sparsity, which is beneficial for downstream reasoning by preventing the graph from becoming
overly dense and noisy.
At the same time, 𝜏simgoverns a trade-off between coverage and reliability in graph construc-
tion. An overly dense graph (i.e., low 𝜏sim) may introduce spurious interactions and amplify noise
propagation during arbitration, while an overly sparse graph (i.e., high 𝜏sim) may miss critical rela-
tionships between claims, reducing the effectiveness of conflict detection. By tuning 𝜏sim, ArbGraph
ensures that arbitration operates over informative yet well-controlled semantic neighborhoods.
3.3.3 Relation Verification and Edge Instantiation.For each candidate pair (𝑐𝑖,𝑐𝑗)∈P cand, a veri-
fier model predicts their relation asSupport,Contradiction, orNeutral, along with an associated
confidence score.
To ensure reliability, only relations with confidence exceeding a threshold 𝜏confare instantiated
as edges inE. This conservative filtering mitigates noise from imperfect classification, especially in
cases involving numerical or temporal reasoning.
Beyond filtering, the graph structure itself provides robustness throughmulti-source aggregation.
Consistent relations supported by multiple independent claim pairs are reinforced through redun-
dant connections, while isolated or spurious edges are less likely to influence global reasoning.
To further control graph density, we retain at most 𝑀support edges for each node, preventing
excessive propagation over weak or noisy interactions.
Furthermore, explicitly separating support and contradiction edges enables downstream arbitra-
tion to reason over agreement and conflict in a structured manner, rather than relying on implicit
signal aggregation within text. This separation is critical for accurately resolving inconsistencies in
complex evidence settings.
3.4 Iterative Credibility Arbitration
Given the evidence graph G𝐸, we perform iterative credibility arbitration to obtain a high-
confidence subset of claims. Each claim 𝑐𝑖∈V is associated with a credibility score 𝑝𝑖∈[0,1]. For
numerical stability, we parameterize credibility in the logit space:
𝑠𝑖=logit(𝑝𝑖)=log𝑝𝑖
1−𝑝𝑖.(3)
At round𝑡, logits are converted to probabilities via 𝑝(𝑡)
𝑖=𝜎(𝑠(𝑡)
𝑖), which are used for both
arbitration scheduling and validation. The iterative design reflects the interdependent nature of
contradictions, where resolving one conflict may affect the credibility of others.
The full procedure is summarized in Algorithm 1.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 9
Algorithm 1ArbGraph: Iterative Credibility Arbitration
Require:Evidence graphG 𝐸=(V,E sup∪E con), step size𝜂, threshold𝜏 accept , budget𝑘
Ensure:Validated claim setV val
1:Initialize:𝑠 𝑖←logit(𝑝(0)
𝑖)for all𝑐𝑖∈V
2:for𝑡=1to𝑇do
3:𝑝𝑖←𝜎(𝑠𝑖)for all𝑐𝑖∈V⊲Current credibility scores
4:P active←∅
5:for(𝑐 𝑖,𝑐𝑗)∈E conwith𝑝𝑖,𝑝𝑗>𝜏accept do
6:𝐼 𝑖𝑗←𝑝𝑖+𝑝𝑗
1+|𝑝𝑖−𝑝𝑗|
7:Add(𝑐 𝑖,𝑐𝑗,𝐼𝑖𝑗)toP active
8:end for
9:Select top-𝑘conflicting pairsP top⊂P active based on𝐼 𝑖𝑗
10:for(𝑐 𝑖,𝑐𝑗)∈P topdo
11:E ctx←N sup(𝑐𝑖)∪N sup(𝑐𝑗)
12:(𝑤,𝑐 win,𝑐lose)←Arbitrate(𝑐 𝑖,𝑐𝑗,Ectx)
13:if𝑤=1then
14:𝑠 win←𝑠 win+𝜂;𝑠 lose←𝑠 lose−𝜂
15:end if
16:end for
17:end for
18:returnV val←{𝑐𝑖|𝜎(𝑠𝑖)≥𝜏 accept}
Active Contradiction Mining.At each round, we prioritize conflicts among currently plausible
claims. For each contradiction edge(𝑐 𝑖,𝑐𝑗)∈E con, we compute an intensity score
𝐼𝑖𝑗=𝑝𝑖+𝑝𝑗
1+|𝑝𝑖−𝑝𝑗|,(4)
which favors pairs with both high credibility and small confidence gaps.
This prioritization focuses computation on high-impact ambiguities, while avoiding unnecessary
updates on low-confidence or easily separable cases.
Gated Arbitration with Supporting Context.For each selected pair, we perform context-aware
arbitration using local graph structure. Specifically, we construct a supporting context
Ectx=N sup(𝑐𝑖)∪N sup(𝑐𝑗),
whereN sup(𝑐)denotes support neighbors of𝑐.
An LLM-based arbitrator evaluates (𝑐𝑖,𝑐𝑗)conditioned onEctxand outputs a winner 𝑐win, a loser
𝑐lose, and a binary gate 𝑤∈{ 0,1}. The gate indicates whether sufficient evidence is available to
support a confident decision. This mechanism prevents premature or unreliable updates when the
supporting context is insufficient, thereby reducing the risk of propagating incorrect decisions
across iterations. In practice, the gating decision is implemented using a heuristic confidence
threshold, which is empirically stable within a reasonable range.
When𝑤=1, we apply symmetric logit updates:
𝑠win←𝑠 win+𝜂, 𝑠 lose←𝑠 lose−𝜂.
This update rule ensures stable and bounded adjustments of credibility scores, while preserving
relative ordering without introducing bias drift. Together with the gating mechanism, it enables
ArbGraph to perform reliable credibility updates driven by sufficiently supported evidence rather
than local or noisy signals.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

10 Niu et al.
Stability Mechanisms.To mitigate noise in individual arbitration decisions, ArbGraph incorpo-
rates several stabilizing components: (1) gated updates that suppress low-confidence decisions; (2)
top-𝑘scheduling that limits the influence of spurious conflicts; (3) iterative refinement that allows
decisions to be revised as new evidence is incorporated; and (4) contextual arbitration grounded in
multi-source supporting evidence.
Validated Claim Set.After𝑇rounds, we obtain the validated claim set
Vval={𝑐𝑖|𝜎(𝑠(𝑇)
𝑖)≥𝜏 accept},(5)
which serves as the conflict-aware evidence context for final generation.
3.5 Computational Efficiency Discussion
Pre-generation evidence arbitration inevitably introduces additional computation beyond a
standard retrieve-and-generate pipeline. However, ArbGraph is designed so that this overhead
remains explicitly bounded by structural constraints at each stage, rather than growing through
open-ended reasoning or repeated retrieval–generation loops.
First, query-aware claim filtering limits the size of the initial claim pool by removing irrelevant
or weakly related information before graph construction. Second, relation verification is restricted
to local semantic neighborhoods through threshold-based candidate mining parameterized by
𝜏sim, avoiding exhaustive pairwise comparison over all claims. In addition, retaining at most 𝑀
support edges bounds graph density and prevents excessive propagation over noisy or weak
interactions. Third, during arbitration, LLM-based decisions are allocated selectively rather than
exhaustively: only the top- 𝑘high-impact conflicts are considered in each round, and confidence-
based gating suppresses unnecessary updates on marginal or weakly supported cases. Together,
these mechanisms constrain the effective search space for arbitration and reduce the number of
expensive verification and decision steps.
Under this design, the computational cost of ArbGraph is governed primarily by the size of the
filtered claim graph and by the arbitration budget, rather than by unbounded search trajectories.
Compared with agentic baselines such as ReAct or MCTS-style methods, which may involve
multiple rounds of retrieval or long-horizon reasoning paths, ArbGraph operates under a fixed
retrieval budget and a bounded number of verification and arbitration steps. We therefore view its
cost profile as more controlled and predictable, especially in settings where the main challenge lies
in resolving conflicting evidence rather than repeatedly expanding the search process.
Overall, the inference cost of ArbGraph scales with the amount of retained evidence and the
complexity of evidence conflicts, while remaining explicitly constrained by filtering, sparsification,
and bounded arbitration. This makes ArbGraph a practically manageable reliability mechanism for
long-form RAG, rather than an open-ended reasoning procedure.
4 EXPERIMENTS
This section presents a systematic empirical evaluation of ArbGraph for long-form retrieval-
augmented generation. We evaluate its performance from multiple perspectives, including overall
generation quality, robustness under noisy retrieval, the contribution of key components, and sensi-
tivity to graph construction and arbitration parameters. We begin by introducing the experimental
settings, followed by the main results, ablation studies, and detailed analyses on LongFact and
RAGChecker.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 11
4.1 Experimental Settings
4.1.1 Datasets.We evaluate ArbGraph on two complementary benchmarks,LongFact[ 37] and
RAGChecker[ 30], enabling a comprehensive assessment of factuality, grounding, and robustness
in long-form retrieval-augmented generation.
LongFactis a manually curated benchmark for long-form factual generation, containing complex
information-seeking queries spanning 38 knowledge domains. Following prior practice, we group
these domains into eight broader categories for analysis. A key advantage of LongFact is that each
reference answer is annotated withatomic information units, which enables fine-grained evaluation
of factual coverage and evidence completeness at the level of atomic claims. This benchmark is
therefore particularly suitable for assessing whether ArbGraph improves factual recall and preserves
informative content in long-form responses.
RAGCheckerprovides a complementary diagnostic benchmark for retrieval-augmented gener-
ation. It covers queries from 10 public domains and evaluates systems from multiple perspectives,
including grounding quality, retrieval usage, and robustness to noisy evidence. Since the original
benchmark is primarily designed for short-form responses, we follow its diagnostic protocol while
extending it to long-form generation to better match our task formulation. This benchmark is
especially useful for analyzing whether ArbGraph improves faithfulness and reduces the impact of
both relevant and irrelevant retrieval noise.
Taken together, LongFact emphasizesfine-grained factual coveragein long-form answers, whereas
RAGChecker provides adiagnostic view of grounding robustness. Their combination enables a
comprehensive evaluation of ArbGraph from both generation quality and evidence reliability
perspectives.
4.1.2 Evaluation Metrics.We evaluate model performance using a suite of metrics covering three
aspects: generative informativeness, factual grounding, and robustness to retrieval noise.
For generative informativeness, we report Fact Recall (FR) and Information Density (ID). Fol-
lowing long-form factual generation evaluation [ 5], FR measures the proportion of ground-truth
atomic facts covered by the generated answer, reflecting factual completeness. ID measures factual
efficiency, defined as the number of atomic facts per token.
For factual grounding, we adopt the diagnostic metrics from RAGChecker. Faithfulness measures
the proportion of generated claims that are supported by the retrieved evidence, indicating how
reliably the model grounds its output in external context. Context Utilization measures how well
the retrieved evidence captures the ground-truth facts that should appear in the response, reflecting
the effectiveness of retrieval usage.
For robustness to noisy retrieval, we report two noise-sensitivity metrics from RAGChecker:
Relevant Noise Sensitivity (Noise-S) and Irrelevant Noise Sensitivity (Noise-I). Noise-S evaluates
how susceptible the model is to misleading but topically related evidence, while Noise-I measures
the impact of distracting evidence that is unrelated to the query. Lower sensitivity indicates stronger
robustness under noisy retrieval conditions.
Furthermore, to analyze knowledge attribution, we report Hallucination and Self-knowledge.
Hallucination quantifies the proportion of generated claims that are unsupported or incorrect
with respect to the retrieved evidence. Self-knowledge measures correct claims produced from the
model’s internal parametric knowledge rather than from the retrieved context.
4.1.3 Baseline Models.To evaluate the impact of explicit evidence-level arbitration under long-form
generation settings, we compare ArbGraph with representative baselines that incorporate reasoning
or verification to handle potentially conflicting evidence. Recent graph-based RAG approaches
(e.g., GraphRAG [ 4], HippoRAG [ 11]) have shown strong performance in enhancing retrieval
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

12 Niu et al.
coverage and multi-hop connectivity. However, they primarily focus on improving information
organization and connectivity, without explicitly addressing evidence-level conflict resolution.
In addition, methods such as Self-RAG [ 1] introduce token-level verification during generation,
but are mainly designed for short-form QA settings. Therefore, we compare ArbGraph against
representative methods for long-form RAG under similar evaluation settings. The selected baselines
span three categories: Naive RAG, Reasoning RAG, and Agentic RAG, reflecting a progression from
static pipelines to more autonomous reasoning frameworks.
(1)Naive RAG. Standard RAG [ 17] serves as a basic baseline, following a retrieve-and-generate
pipeline without intermediate reasoning or verification. It directly concatenates retrieved documents
into the context window, making the generation process susceptible to input noise and implicit
factual conflicts.
(2)Reasoning RAG. This category introduces explicit reasoning steps to improve generation
quality. CoT (Chain-of-Thought) [ 36] prompts the model to produce intermediate reasoning trajec-
tories prior to the final answer, encouraging deeper parametric reasoning over the provided context.
ReAct [ 41] interleaves reasoning with active retrieval, enabling iterative refinement of context.
However, both methods rely on the language model’s implicit ability to resolve inconsistencies
among retrieved evidence during generation.
(3)Agentic RAG. These methods incorporate agent-like behaviors, such as self-verification
and iterative refinement, to handle complex queries. CRITIC [ 9] adopts a generate–verify–correct
loop with external tools, attempting to rectify factual errors after the initial generation phase.
FIRE [ 38] improves factuality through fine-grained, sequential verification during generation, pri-
marily emphasizing compact and controlled outputs rather than explicit, structure-aware evidence
consolidation. In our implementation, we follow its original formulation by prompting the model
to perform step-by-step verification during generation, rather than applying it as a post-hoc filter-
ing mechanism on top of naive RAG outputs. MCTS-RAG [ 13] formulates long-form generation
as a principled search process, using Monte Carlo Tree Search to explore alternative reasoning
trajectories and evaluate state values.
4.1.4 Implementation Details.We adopt two instruction-tuned large language models as generators:
Qwen3-4B-Instruct [ 40] and LLaMA-3.1-8B-Instruct [ 10]. Unless otherwise specified, all components
of ArbGraph, including claim extraction, relation verification, and arbitration, share the same
configuration across both backbones.
To ensure a fair comparison, we use a unified prompting strategy and decoding setup for all
models. Specifically, we employ deterministic decoding with temperature set to 0, and fix the
maximum generation length for long-form responses. For retrieval, we use a fixed Wikipedia-based
corpus with a top- 𝑘=5setting for all methods and retrieve relevant documents by the official
Wikipedia API1. Given that both backbones exhibit consistent relative performance trends in our
main experiments, we conduct ablation studies and detailed analyses on Qwen3-4B-Instruct for
efficiency, while reporting the main results on both models.
Atomic claims are embedded using bge-large-en-v1.5 , and cosine similarity is employed for
both query-aware filtering and claim matching. We set the query relevance threshold to 𝜏𝑞=0.3
and adopt a threshold-based candidate retrieval strategy for relation mining. All contradiction
edges verified by the LLM are preserved, while support edges are globally pruned by retaining
only the top- 𝑀edges ranked by semantic similarity (with 𝑀= 60in all experiments). The iterative
credibility arbitration module runs with an arbitration budget of 𝑘=3per round, step size 𝜂=0.8,
and acceptance threshold𝜏 accept=0.3.
1https://www.mediawiki.org/wiki/API:Main_page
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 13
Table 1. Main results on LongFact across eight domains using two backbones. We report (a) Fact Recall (FR)
and (b) Information Density (ID). Sci.: Science, Tech.: Technology, Med.: Medicine, Cult.: Culture, Comm.:
Communication, Life.: Lifestyle. The best results in each backbone group are highlighted in bold.
Method Sci. Tech. Med. Law Cult. Event Comm. Life. Avg.
(a) Fact Recall (FR)↑
Backbone: Qwen3-4B
Standard RAG 64.9 73.7 64.7 61.6 58.5 56.7 56.6 68.6 63.5
ReAct 65.6 59.2 71.3 62.4 64.4 66.0 50.6 70.8 63.3
CoT 65.7 70.0 65.6 62.5 55.2 58.5 55.8 69.4 63.1
CRITIC 78.4 79.7 82.4 73.3 76.6 75.2 69.3 75.4 76.3
FIRE 79.7 76.6 82.8 77.1 80.2 79.1 74.0 81.2 78.8
MCTS-RAG 80.4 77.4 85.2 75.8 82.4 79.2 76.3 77.9 79.2
Ours 83.5 84.4 86.2 83.3 82.5 79.8 84.7 81.6 83.3
Backbone: LLaMA-3.1-8B
Standard RAG 59.6 68.8 69.8 66.0 62.2 59.5 57.7 72.1 64.6
CoT 69.1 56.1 68.3 61.9 66.6 69.0 73.4 60.8 64.7
ReAct 69.5 69.2 71.9 68.7 67.3 68.3 66.8 70.4 69.0
CRITIC 78.2 69.2 78.1 71.3 70.1 74.3 69.2 75.2 73.3
FIRE 79.2 78.4 80.5 77.1 76.2 75.1 78.2 73.6 78.3
MCTS-RAG 83.8 81.6 84.0 82.5 80.5 81.9 77.1 81.9 81.7
Ours 84.2 85.1 86.1 85.3 84.0 83.0 85.9 84.7 84.9
(b) Information Density (ID)↑
Backbone: Qwen3-4B
Standard RAG 79.1 99.2 83.6 77.3 77.6 72.5 73.2 86.2 81.4
ReAct 72.1 70.3 81.1 62.0 70.5 56.9 60.2 86.4 70.6
CoT 100.1 115.9 111.7 91.3 93.9 95.5 96.6 124.0 104.3
CRITIC 71.1 78.5 81.0 70.6 71.4 76.1 71.9 83.8 75.5
FIRE 229.9 232.5 107.6 240.9 155.1 107.8 93.7219.7177.2
MCTS-RAG 97.8 120.0 95.6 76.6 92.8 98.0 93.8 85.8 95.2
Ours 247.3 235.5 140.6 241.2 156.4 132.5 124.5190.2187.6
Backbone: LLaMA-3.1-8B
Standard RAG 42.7 51.3 54.3 47.6 43.4 38.0 36.3 58.5 46.9
CoT 56.9 61.0 77.7 57.8 54.8 51.7 48.7 53.5 57.8
ReAct 70.4 66.6 76.2 61.8 72.0 48.1 63.0 73.5 67.1
CRITIC 79.0 68.7 85.7 63.9 77.6 52.5 64.2 79.1 72.0
FIRE 162.4 170.1 158.9 175.2 160.2 161.6 155.1 158.1 165.7
MCTS-RAG 61.3 61.9 62.0 59.5 59.5 58.3 54.6 62.9 60.0
Ours 178.6 191.0 176.2 195.7 174.1 177.5 171.2 176.0 180.0
4.2 Main Results
ArbGraph consistently outperforms all baselines across both benchmarks and backbones, achiev-
ing improvements in factual coverage, information density, and robustness under noisy retrieval.
Detailed results are reported in Table 1 and Table 2.
Strong performance in long-form generation.Across both benchmarks, ArbGraph achieves
the best overall performance under both backbones. On LongFact, prior methods exhibit a trade-off
between factual coverage and generation efficiency, where higher recall often comes with less
compact outputs, while density-oriented methods tend to miss supporting facts. Unlike these
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

14 Niu et al.
Table 2. Overall performance comparison across paradigms and backbones. FR: Fact Recall; ID: Information
Density; Util.: Evidence Utilization; Noise-S/I: Support/Irrelevant Noise; Hallu.: Hallucination Rate; Self-K:
Self-Knowledge; Faith.: Faithfulness.
Paradigm Model FR↑ID↑Util.↑Noise-S↓Noise-I↓Hallu.↓Self-K↓Faith.↑
Backbone: Qwen3-4B
NaiveStandard RAG 54.3 75.8 31.5 7.9 7.7 51.4 6.6 32.2
ReasoningReAct 54.4 45.0 28.9 7.1 9.4 55.2 6.5 35.1
CoT 57.6 110.6 28.4 7.0 7.9 49.5 6.8 30.1
AgenticCRITIC 54.8 41.8 25.0 4.8 5.2 45.0 6.9 35.0
FIRE 70.7 59.6 34.6 11.6 8.1 48.6 7.1 37.8
MCTS-RAG 72.0 98.7 32.6 7.0 10.4 36.9 4.6 50.5
Ours ArbGraph 74.5 135.4 36.6 4.2 4.9 33.3 4.4 53.8
Backbone: LLaMA-3.1-8B
NaiveStandard RAG 54.7 51.7 29.7 12.5 6.0 59.9 6.9 29.2
ReasoningCoT 59.8 57.8 30.4 9.4 6.2 52.5 6.8 29.8
ReAct 62.1 62.6 31.5 12.2 7.7 59.4 7.1 25.8
AgenticCRITIC 62.5 78.8 32.9 13.6 5.2 56.1 7.0 31.0
MCTS-RAG 71.3 96.6 33.5 9.4 9.8 43.1 5.9 39.7
FIRE 68.7 88.2 33.1 8.7 4.7 41.8 5.3 33.6
Ours ArbGraph 75.6 99.3 41.2 4.2 3.6 36.3 4.6 51.5
approaches, ArbGraph improves both Fact Recall (FR) and Information Density (ID) simultaneously.
Under Qwen3-4B, it increases FR from 79.2 to 83.3 (vs. MCTS-RAG) and ID from 177.2 to 187.6
(vs. FIRE), with similar gains observed under LLaMA-3.1-8B. Notably, these improvements over
the strongest baselines are statistically significant ( 𝑝< 0.01). This joint improvement can be
attributed to ArbGraph’s ability to filter out redundant or unreliable claims while preserving
complementary evidence through structured arbitration. As a result, the model can generate more
compact responses without sacrificing factual completeness.
Consistency across backbones.ArbGraph shows consistent improvements on both Qwen3-4B
and LLaMA-3.1-8B, indicating that the gains are largely model-agnostic and stem from improved
evidence consolidation rather than model scale.
Improved trustworthiness.On RAGChecker, ArbGraph achieves the highest Faithfulness and
lowest Hallucination across both backbones, while maintaining strong factual recall. It also attains
the best evidence utilization, suggesting more effective consolidation of useful evidence. This result
indicates that explicit evidence-level arbitration helps filter unreliable or conflicting evidence before
generation.
Robustness to noise.ArbGraph demonstrates strong robustness under noisy retrieval, achieving
the lowest noise sensitivity across both backbones. For example, under Qwen3-4B, MCTS-RAG
obtains FR 72.0 but suffers from high irrelevant-noise sensitivity (10.4), whereas ArbGraph achieves
higher FR (74.5) with substantially lower sensitivity (4.9). This suggests that resolving conflicts
prior to generation reduces the impact of both relevant and irrelevant noise.
Overall, these results demonstrate that resolving conflicts at the evidence level provides a more
reliable foundation for long-form generation than generation-time heuristics.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 15
Table 3. Ablation results on the LongFact dataset. We show the impact of different components on (a) Fact
Recall and (b) Information Density across eight domains. Bold numbers indicate the best performance.
Method Sci. Tech. Med. Law Cult. Event Comm. Life. Avg.
(a) Fact Recall (FR)↑
ArbGraph (Ours) 83.5 84.4 86.2 83.3 82.5 79.8 84.7 81.6 83.3
w/o Evidence Graph 65.6 68.2 64.8 63.4 66.1 67.5 58.9 72.4 66.1
w/o Soft Arbitration 77.1 74.4 78.0 65.4 74.2 64.1 66.9 73.2 72.2
w/o Initial Screening 73.0 74.8 78.7 72.5 71.6 69.0 77.1 74.9 74.0
w/o Rejection Control 73.7 75.0 78.1 72.5 75.4 74.8 77.3 74.7 75.1
w/o Gating 75.1 75.8 78.3 76.4 75.8 75.1 75.8 74.1 75.8
(b) Information Density (ID)↑
ArbGraph (Ours) 247.3 235.5 140.6 241.2 156.4 132.5 124.5 190.2 187.6
w/o Evidence Graph 82.1 85.4 78.6 75.2 80.3 77.8 72.4 92.6 81.4
w/o Soft Arbitration 120.1 123.8 109.9 111.3 115.3 82.3 101.4 107.9 110.0
w/o Initial Screening 153.4 82.5 88.6 79.4 77.7 99.7 85.1 70.6 93.3
w/o Rejection Control 97.0 185.5 112.5 238.5 79.7 130.2 73.9 160.7 132.5
w/o Gating 109.2 149.3 104.4 180.0 106.8 105.5 103.5 106.9 120.7
4.3 Ablation Study
We conduct ablation studies on the LongFact dataset to assess the contribution of key components
in ArbGraph. Table 3 reports Fact Recall (FR) and Information Density (ID) across eight domains.
Each variant removes or modifies a single component while keeping other settings fixed. Overall,
removing any major component leads to consistent performance degradation, highlighting the
importance of the interaction between structured evidence modeling and iterative arbitration.
Evidence graph.Thew/o Evidence Graphvariant reduces ArbGraph to a flat set of independent
claims. The substantial performance drop indicates that modeling inter-claim dependencies is
critical for capturing both supportive and contradictory relations. Without an explicit structure,
evidence aggregation becomes local and fragmented, preventing the system from forming globally
consistent judgments, especially under multi-source or conflicting evidence.
Soft conflict arbitration.Replacing multi-round arbitration with a single-step strategy (w/o
Soft arbitration) consistently degrades FR and ID. This suggests that conflict resolution requires
progressive refinement rather than one-shot decisions. Hard elimination tends to discard partially
correct evidence, whereas iterative updates enable gradual calibration of credibility under conflicting
signals, leading to more stable evidence integration.
Initial relevance screening.Removing initial screening (w/o Initial Screening) leads to unstable
performance. This reflects the role of early-stage filtering in controlling the search space: without
it, irrelevant claims introduce noisy interactions that propagate through the graph and distort
subsequent arbitration, especially under dense or weakly related retrieval results.
Rejection control.Omitting rejection control (w/o Rejection Control) reduces Fact Recall, indi-
cating that suppressing low-credibility or contradicted claims is necessary to maintain effective
evidence composition. Without this mechanism, unreliable nodes remain active and interfere with
the aggregation of consistent evidence.
Binary gating.Disabling gating (w/o Gating) harms both FR and ID, suggesting that arbitration
decisions must be conditioned on sufficient supporting context. Without confidence-aware updates,
unreliable decisions are more likely to propagate, reducing overall stability.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

16 Niu et al.
0.1 0.3 0.5 0.7
Query Relevance Threshold (¿q)72.575.077.580.082.5Fact Recall (FR) %
0 30 60 90
Graph Sparsity (M)707580Fact Recall (FR) %
1 3 5 7 9
Conflict Nodes (k)72.575.077.580.082.5Fact Recall (FR) %
(a) FR vs.𝜏 𝑞 (c) FR vs.𝑀(e) FR vs.𝑘
0.1 0.3 0.5 0.7
Query Relevance Threshold (¿q)140150160170180190Info. Density (ID)
0 30 60 90
Graph Sparsity (M)120140160180Info. Density (ID)
1 3 5 7 9
Conflict Nodes (k)120140160180Info. Density (ID)
(b) ID vs.𝜏 𝑞 (d) ID vs.𝑀(f) ID vs.𝑘
Fig. 2.Sensitivity analysis of key parameters.Moderate values of 𝜏𝑞,𝑀, and𝑘achieve a balance between
evidence coverage, structural connectivity, and arbitration stability.
4.4 Analysis of Graph Construction Factors
We conduct the following sensitivity analyses on the LongFact benchmark, where claim-level
annotations enable fine-grained evaluation of how graph construction affects factual coverage and
evidence aggregation.
4.4.1 Sensitivity to Query Relevance Threshold ( 𝜏𝑞).We further examine the sensitivity of ArbGraph
to the query relevance threshold 𝜏𝑞, which controls whether a claim node is retained based on its
semantic similarity to the query via the query-aware filtering criterion in Eq. 1. Figure 2(a) and (b)
report the effects of varying𝜏 𝑞on Fact Recall (FR) and Information Density (ID).
When𝜏𝑞is set too low, a large number of weakly related background claims are introduced into
the evidence graph. Although coverage increases, excessive noise dilutes informative content and
interferes with evidence arbitration, leading to degraded ID and even reduced FR. This behavior
reflects the dependence of arbitration on the quality of initial node selection, as noisy nodes may
dominate local interactions and distort credibility propagation.
In contrast, overly strict thresholds aggressively prune claim nodes, discarding supporting but
implicitly relevant evidence and causing a substantial drop in recall. Such pruning reduces the
diversity of evidence sources, limiting the ability of the graph to capture complementary support
signals.
A moderate threshold ( 𝜏𝑞=0.3) achieves the best balance between relevance and coverage,
yielding the highest FR and ID. Accordingly, we adopt 𝜏𝑞=0.3in all subsequent experiments.
This result highlights that effective graph construction requires a careful balance between noise
suppression and evidence completeness.
4.4.2 Sensitivity to Graph Sparsity ( 𝑀).We analyze the sensitivity ofArbGraphto the graph
sparsity parameter 𝑀, which controls the maximum number of support edges retained after relation
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 17
verification. Specifically, all contradiction edges are preserved, while only the top- 𝑀support edges
ranked by semantic similarity are kept.
As shown in Figure 2(c) and (d), the results exhibit a clear bell-shaped trend. When 𝑀is small
(𝑀= 0or30), performance is limited by insufficient connectivity, leading to a fragmented graph
where support and contradiction signals cannot effectively propagate.
Performance peaks at 𝑀= 60, achieving the highest Fact Recall (83.3%) and Information Density
(187.6), indicating that moderate sparsity balances noise filtering and relational coverage. In contrast,
larger values ( 𝑀= 90) introduce noisy or weakly related edges, which dilute meaningful constraints
and reduce stability.
Overall, retaining a moderate number of support edges (Top-60) provides a robust trade-off
between structural efficiency and reasoning effectiveness.
4.4.3 Discussion on Semantic Similarity Threshold ( 𝜏sim).The similarity threshold 𝜏simcontrols the
semantic neighborhood for candidate pair mining. A lower threshold increases graph density but
may introduce spurious relations and amplify noise, while a higher threshold enforces sparsity at
the risk of missing valid interactions.
Empirically, ArbGraph remains stable within a moderate range of 𝜏sim, with only limited per-
formance variation observed. This robustness arises because arbitration operates over aggregated
multi-source evidence, rather than relying on individual pairwise relations. As a result, occasional
spurious or missing edges have limited impact on the overall arbitration outcome.
These observations suggest that effective graph construction requires balancing relational cover-
age with structural reliability, ensuring that informative interactions are preserved while avoiding
excessive noise.
4.4.4 Sensitivity to Conflict Intensity ( 𝑘).We analyze the sensitivity ofArbGraphto the conflict
intensity𝑘, which controls the number of contradiction pairs prioritized for arbitration in each
round.
Figure 2(e) and (f) show that increasing 𝑘from small to moderate values improves both Fact
Recall (FR) and Information Density (ID), as a larger arbitration budget enables better reconciliation
of locally inconsistent evidence.
However, performance degrades when 𝑘becomes too large, suggesting that overly aggressive
arbitration introduces noise by forcing decisions on marginal or weakly supported contradictions.
Overall, a moderate value of 𝑘provides the best trade-off, supporting stable performance and
indicating that resolving high-impact conflicts is more effective than exhaustive arbitration.
4.4.5 Cost-Performance Trade-off Analysis.A recurring pattern across our sensitivity analyses
for𝜏𝑞,𝑀, and𝑘is the emergence of a clear bell-shaped trend (see Figure 2). This pattern provides
insight into the relationship between computational investment and factual robustness.
Specifically, increasing the graph sparsity parameter 𝑀or the arbitration budget 𝑘introduces
additional computation (e.g., more relation verification or arbitration steps). However, beyond mod-
erate values, performance degrades, suggesting that excessive computation introduces structural or
reasoning noise (e.g., spurious edges or marginal contradictions) that dilutes informative evidence.
In contrast, ArbGraph performs best under moderate settings (e.g., 𝜏𝑞=0.3,𝑀= 60,𝑘=3),
effectively resolving high-impact conflicts while maintaining high information density. Overall,
these results indicate that ArbGraph does not exhibit monotonic scaling with respect to compu-
tational cost, and that optimal performance depends on balanced, structure-aware allocation of
computation.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

18 Niu et al.
Random Hard-First Easy-First ArbGraph01020304050607080Score (FR) \%73.079.276.283.3
(a) Fact Recall (FR)
Random Hard-First Easy-First ArbGraph0255075100125150175Density (ID)140.6163.9160.8187.6 (b) Information Density (ID)
Fig. 3.Comparison of Scheduling Policies.ArbGraph’s conflict-aware scheduling improves factuality and
evidence density.
4.4.6 Effectiveness of Conflict-Aware Scheduling.We analyze the impact of different scheduling
strategies on the arbitration process. While heuristic-based scheduling policies such asHard-
FirstandEasy-Firstrely on static ordering rules, ArbGraph employs a dynamic, conflict-aware
scheduling policy guided by theIntensity Score(Eq. 4) within the evidence graph. This dynamic
mechanism further enables adaptive allocation of reasoning effort toward high-impact conflicts,
while continuously adjusting the arbitration trajectory based on evolving graph states, rather than
following a fixed processing order.
As shown in Figure 3, ArbGraph consistently outperforms all heuristic-based alternatives. Com-
pared with static scheduling strategies, conflict-aware scheduling prioritizes claim pairs with higher
contradiction intensity, allowing the model to resolve the most critical inconsistencies earlier in
the arbitration process. This leads to more stable intermediate states and reduces the propagation
of unresolved conflicts.
In particular, ArbGraph achieves higher Fact Recall (FR) and the best Information Density
(ID) among all compared strategies. This indicates that effective scheduling is not merely about
processing order, but about selectively focusing on structurally important conflicts. Overall, these
results highlight that leveraging graph-based conflict signals is essential for efficient and reliable
evidence arbitration.
4.5 Qualitative and Reliability Analysis
4.5.1 Component-wise Reliability Analysis and Error Propagation.We conduct a component-wise
reliability analysis of ArbGraph’s key modules, including relation verification and arbitration.
For relation verification, we construct a manually annotated set of 200 claim pairs from LongFact,
labeled as Support, Contradict, or Neutral. As shown in Table 4, the verifier achieves high accuracy,
indicating a generally reliable evidence graph. For arbitration, evaluation on 50 representative cases
yields 96.0% accuracy, indicating stable decisions under noisy retrieval.
We make three observations on the impact of local errors. First, structural redundancy across
multiple evidence sources mitigates isolated verification errors (approximately ∼4%). Second,
mechanism-level safeguards, including query-aware filtering and confidence-based gating, prevent
low-confidence updates from propagating. Third, manual tracing shows that incorrect intermediate
updates are often filtered by the acceptance threshold (𝜏 𝑎𝑐𝑐𝑒𝑝𝑡 ) or not selected during generation.
Overall, these results suggest that ArbGraph is robust to local errors, as inaccuracies are effec-
tively contained by the graph structure and arbitration mechanisms, though more comprehensive
evaluation remains future work.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 19
Table 4. Component-wise reliability analysis of ArbGraph.
Component Category Samples Accuracy
Relation Verifier Support 72 95.8%
Contradict 51 96.1%
Neutral 77 96.1%
Avg.200 96.5%
Conflict Arbitrator Arbitration 50 96.0%
(A) Input: Query & Retrieved Documents
Query:“How is the United States related to the East Asia Summit (EAS)?”
Retrieved documents:(D1)EAS Wiki: U.S. joined in 2011. (D2)ASEAN 2025: U.S. attended meetings. (D3)Trump
Trips: Skipped 2017 meeting. (D4)Homonym: Emergency Alert System (EAS).
(B) Standard RAG: Flat & Vulnerable (C) Ours: Evidence-Level Arbitration
Step 0: Input processing.The model ingests all retrieved
texts equally.
Step 1: Attention dilution.
•Homonym noise:“Emergency Alert System”
(D4).
•Fact decay:“joined 2011” weakened.
Step 2: Implicit conflict.
•“Skipping” misread as non-membership.
Outcome (failure):
“The U.S. is not a member of the EAS. ”Step 0: Claim extraction.
•𝑐 1: joined EAS𝑐 3: skipped meeting
•𝑐 2: attended meetings
•𝑐 4: alert system
Step 1: Filtering.
•𝑐 4discarded,𝑐 2deprioritized
•Retained:𝑐 1,𝑐3
Step 2: Arbitration.
•membership and attendance are distinct relations
Outcome (success):
“The U.S. joined the EAS in 2011. ”
Fig. 4.Comparison of standard RAG vs. ArbGraph.Standard RAG fails due to noise and unresolved
conflicts, while ArbGraph resolves conflicts via evidence-level arbitration.
4.5.2 Case Study.We analyze a challenging query:“How is the United States related to the East
Asia Summit (EAS)?”(Figure 4), which requires distinguishing formal membership from episodic
attendance under ambiguous evidence.
The retrieved documents contain the target fact ( 𝐷1: U.S. joined in 2011) mixed with hetero-
geneous noise, including contextual ambiguity from attendance-related descriptions ( 𝐷2,𝐷3) and
homonym noise ( 𝐷4) referring to the unrelated “Emergency Alert System.” Notably, these signals
correspond to different semantic roles, making naive aggregation prone to confusion.
Standard RAG (Panel B) processes documents as a flat sequence and fails to resolve these
ambiguities. The presence of homonym noise dilutes attention, and the implicit conflict between
“joining” (𝐷1) and “skipping” ( 𝐷3) is misinterpreted, leading to the incorrect conclusion that the U.S.
is not a member of the EAS. Without explicit modeling of such conflicts, contradictory evidence is
implicitly treated as complementary, amplifying reasoning errors.
In contrast, ArbGraph (Panel C) decomposes documents into atomic claims and filters structurally
irrelevant nodes. It then identifies and resolves the conflict between status and attendance claims
through evidence-level arbitration. This explicit separation between status and event-level evidence
avoids spurious interactions during reasoning. Conditioning generation on this consistent structure
enables ArbGraph to correctly infer formal membership.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

20 Niu et al.
4.6 Implications for Long-Form RAG Design
Based on the empirical results on LongFact and RAGChecker, we derive several key insights into
the design of reliable long-form RAG systems.
Evidence consistency is more critical than retrieval breadth.While aggressive retrieval
expansion may improve recall, it also introduces irrelevant or contradictory evidence. Our results
show that increased coverage does not necessarily translate into better generation quality, as noisy
evidence can degrade both factual density and reliability. In contrast, ArbGraph achieves strong
performance by operating on a smaller but internally consistent set of verified claims, highlighting
the importance of evidence coherence over sheer quantity.
Pre-generation arbitration is more robust than post-hoc correction.Post-hoc methods
are limited by theerror cascade: once incorrect information is introduced, it propagates through
the generation process and becomes exceedingly difficult to correct. By resolving conflicts before
decoding, ArbGraph ensures that the model is conditioned on a coherent evidence set, thereby
reducing error propagation and improving output stability.
Structured evidence modeling enables explicit conflict resolution.Unlike prior graph-
based RAG approaches that primarily focus on retrieval connectivity, ArbGraph uses graphs to
explicitly modelsupportandcontradictionrelations. This transforms evidence integration from
implicit, token-level aggregation into structured reasoning over competing claims, enabling explicit
credibility estimation prior to generation.
Improved reliability comes with additional pre-generation cost.ArbGraph introduces
extra steps such as claim extraction, relation verification, and iterative arbitration, which invari-
ably increase computational cost and latency. However, this overhead reflects a necessary shift
from generation-time heuristics to structured pre-generation reasoning. By resolving high-impact
conflicts early, ArbGraph secures a highly dependable foundation for downstream generation,
highlighting a fundamental trade-off between factual reliability and computational efficiency.
Overall, these findings suggest that reliable long-form RAG requires not only stronger retrieval
mechanisms and generation models, but also the effective organization and arbitration of evidence
prior to synthesis.
5 CONCLUSION AND FUTURE WORK
In this paper, we presentedArbGraph, an evidence-level arbitration framework designed to
address the persistent challenge of factual inconsistencies in long-form retrieval-augmented gener-
ation. By shifting from the conventional implicit integration of retrieved content to a structured
“arbitrate-first, generate-later” paradigm, ArbGraph establishes a more reliable factual foundation
for the generation process. The framework’s core strength lies in its ability to decompose evidence
into atomic claims and model their logical interdependencies via a contradiction-aware evidence
graph. Experimental results on LongFact and RAGChecker demonstrate that ArbGraph consistently
outperforms strong baselines in terms of factual coverage, grounding quality, and robustness under
noisy or conflicting retrieval conditions. These findings highlight the importance of explicitly mod-
eling and resolving evidence-level conflicts as a principled approach toward trustworthy long-form
RAG.
Looking ahead, several directions are worth further exploration to improve the practicality
and generality of the proposed framework. First, the current evidence graph focuses on support
and contradiction relations; extending it to capture richer logical structures, such as temporal
dependencies or causal relations, may further enhance reasoning capabilities. Second, while the
iterative arbitration process improves evidence reliability, it introduces additional pre-generation
reasoning overhead. Understanding the trade-off between factual consistency and computational
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 21
efficiency, as well as developing lightweight arbitrators or adaptive scheduling strategies, remains
an important direction for future work. Finally, we plan to extend ArbGraph to multimodal settings,
enabling consistent reasoning across heterogeneous evidence sources (e.g., text and images), and to
explore its integration with post-hoc verification methods to form a unified multi-stage reliability
framework.
REFERENCES
[1]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023. Self-rag: Learning to retrieve,
generate, and critique through self-reflection. InThe Twelfth International Conference on Learning Representations.
[2]Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski, Lukas Gianinazzi, Joanna Gajda,
Tomasz Lehmann, Hubert Niewiadomski, Piotr Nyczyk, et al .2024. Graph of thoughts: Solving elaborate problems
with large language models. InProceedings of the AAAI conference on artificial intelligence, Vol. 38. 17682–17690.
[3]Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2024.
Chain-of-verification reduces hallucination in large language models. InFindings of the association for computational
linguistics: ACL 2024. 3563–3578.
[4]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitan-
sky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[5]Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long form
question answering. InProceedings of the 57th annual meeting of the association for computational linguistics. 3558–3567.
[6]Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao,
Hongrae Lee, Da-Cheng Juan, et al .2023. Rarr: Researching and revising what language models say, using language
models. InProceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers). 16477–16508.
[7]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, Haofen
Wang, et al .2023. Retrieval-augmented generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023), 32.
[8]Ziyu Ge, Yuhao Wu, Daniel Wai Kit Chin, Roy Ka-Wei Lee, and Rui Cao. 2025. Resolving conflicting evidence in
automated fact-checking: A study on retrieval-augmented llms.arXiv preprint arXiv:2505.17762(2025).
[9]Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2023. Critic: Large
language models can self-correct with tool-interactive critiquing.arXiv preprint arXiv:2305.11738(2023).
[10] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al .2024. The llama 3 herd of models.arXiv preprint
arXiv:2407.21783(2024).
[11] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. HippoRAG: Neurobiologically
Inspired Long-Term Memory for Large Language Models. InAdvances in Neural Information Processing Systems,
A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (Eds.), Vol. 37. Curran Associates,
Inc., 59532–59569.
[12] Rujun Han, Yuhao Zhang, Peng Qi, Yumo Xu, Jenyuan Wang, Lan Liu, William Yang Wang, Bonan Min, and Vittorio
Castelli. 2024. Rag-qa arena: Evaluating domain robustness for long-form retrieval augmented question answering. In
Proceedings of the 2024 conference on empirical methods in natural language processing. 4354–4374.
[13] Yunhai Hu, Yilun Zhao, Chen Zhao, and Arman Cohan. 2025. Mcts-rag: Enhancing retrieval-augmented generation
with monte carlo tree search.arXiv preprint arXiv:2503.20757(2025).
[14] Yukun Huang, Yixin Liu, Raghuveer Thirukovalluru, Arman Cohan, and Bhuwan Dhingra. 2024. Calibrating long-form
generations from large language models. InFindings of the association for computational linguistics: EMNLP 2024.
13441–13460.
[15] Jeongyeon Hwang, Junyoung Park, Hyejin Park, Dongwoo Kim, Sangdon Park, and Jungseul Ok. 2025. Retrieval-
augmented generation with estimation of source reliability. InProceedings of the 2025 Conference on Empirical Methods
in Natural Language Processing. 34267–34291.
[16] Zhan Peng Lee, Andre Lin, and Calvin Tan. 2025. Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination
in Retrieval-Augmented Generation.arXiv preprint arXiv:2505.10792(2025).
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing Systems, H. Larochelle, M. Ranzato,
R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 9459–9474.
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

22 Niu et al.
[18] Changmao Li and Jeffrey Flanigan. 2024. Rac: Efficient llm factuality correction with retrieval augmentation.arXiv
preprint arXiv:2410.15667(2024).
[19] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024.
Lost in the middle: How language models use long contexts.Transactions of the association for computational linguistics
12 (2024), 157–173.
[20] Shuyi Liu, Yu-Ming Shang, and Xi Zhang. 2026. Truthfulrag: Resolving factual-level conflicts in retrieval-augmented
generation with knowledge graphs. InProceedings of the AAAI Conference on Artificial Intelligence, Vol. 40. 32168–32176.
[21] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2023. Reasoning on graphs: Faithful and interpretable
large language model reasoning.arXiv preprint arXiv:2310.01061(2023).
[22] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah
Young, Lucy Campbell-Gillingham, Geoffrey Irving, et al .2022. Teaching language models to support answers with
verified quotes.arXiv preprint arXiv:2203.11147(2022).
[23] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu
Jain, Vineet Kosaraju, William Saunders, et al .2021. WebGPT: Browser-Assisted Question-Answering with Human
Feedback.arXiv preprint arXiv:2112.09332(2021).
[24] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, Ian Soboroff, Hoa Trang Dang,
and Jimmy Lin. 2025. The great nugget recall: Automating fact extraction and rag evaluation with large language
models. InProceedings of the 48th International ACM SIGIR Conference on Research and Development in Information
Retrieval. 180–190.
[25] Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao Liang, Kunlun Zhu, Yankai Lin, Xu Han, Ning Ding, Huadong Wang,
et al.2023. Webcpm: Interactive web search for chinese long-form question answering. InProceedings of the 61st
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 8968–8988.
[26] Yingqi Qu, Yuchen Ding, Jing Liu, Kai Liu, Ruiyang Ren, Wayne Xin Zhao, Daxiang Dong, Hua Wu, and Haifeng Wang.
2021. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.
InProceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies. 5835–5847.
[27] Mayank Ravishankara. 2025. FVA-RAG: Falsification-Verification Alignment for Mitigating Sycophantic Hallucinations.
arXiv preprint arXiv:2512.07015(2025).
[28] Ruiyang Ren, Yuhao Wang, Junyi Li, Jinhao Jiang, Wayne Xin Zhao, Wenjie Wang, and Tat-Seng Chua. 2025. LLM-
based Search Assistant with Holistically Guided MCTS for Intricate Information Seeking. InProceedings of the 48th
International ACM SIGIR Conference on Research and Development in Information Retrieval. 1098–1108.
[29] Ruiyang Ren, Yuhao Wang, Yingqi Qu, Wayne Xin Zhao, Jing Liu, Hua Wu, Ji-Rong Wen, and Haifeng Wang. 2025.
Investigating the factual knowledge boundary of large language models with retrieval augmentation. InProceedings of
the 31st International Conference on Computational Linguistics. 3697–3715.
[30] Dongyu Ru, Lin Qiu, Xiangkun Hu, Tianhang Zhang, Peng Shi, Shuaichen Chang, Cheng Jiayang, Cunxiang Wang,
Shichao Sun, Huanyu Li, Zizhao Zhang, Binjie Wang, Jiarong Jiang, Tong He, Zhiguo Wang, Pengfei Liu, Yue Zhang,
and Zheng Zhang. 2024. RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation. In
Advances in Neural Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak,
and C. Zhang (Eds.), Vol. 37. Curran Associates, Inc., 21999–22027. https://doi.org/10.52202/079017-0692
[31] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language
agents with verbal reinforcement learning. InAdvances in Neural Information Processing Systems, A. Oh, T. Naumann,
A. Globerson, K. Saenko, M. Hardt, and S. Levine (Eds.), Vol. 36. Curran Associates, Inc., 8634–8652.
[32] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M Ni, Heung-Yeung Shum,
and Jian Guo. 2023. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph.
arXiv preprint arXiv:2307.07697(2023).
[33] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. 2023. Plan-
Bench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change.
InAdvances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and
S. Levine (Eds.), Vol. 36. Curran Associates, Inc., 38975–38987.
[34] Yuhao Wang, Ruiyang Ren, Yucheng Wang, Jing Liu, Xin Zhao, Hua Wu, and Haifeng Wang. 2026. Bee-rag: Balanced
entropy engineering for retrieval-augmented generation. InProceedings of the AAAI Conference on Artificial Intelligence,
Vol. 40. 33737–33745.
[35] Yuhao Wang, Ruiyang Ren, Yucheng Wang, Wayne Xin Zhao, Jing Liu, Hua Wu, and Haifeng Wang. 2025. Reinforced
Informativeness Optimization for Long-Form Retrieval-Augmented Generation.arXiv preprint arXiv:2505.20825(2025).
[36] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou.
2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. InAdvances in Neural Information
Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (Eds.), Vol. 35. Curran Associates,
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.

ArbGraph: Conflict-Aware Evidence Arbitration for Reliable Long-Form Retrieval-Augmented Generation 23
Inc., 24824–24837.
[37] Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng, Ruibo Liu,
Da Huang, Cosmo Du, and Quoc V. Le. 2024. Long-form factuality in large language models. InAdvances in Neural
Information Processing Systems, A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang
(Eds.), Vol. 37. Curran Associates, Inc., 80756–80827. https://doi.org/10.52202/079017-2567
[38] Zhuohan Xie, Rui Xing, Yuxia Wang, Jiahui Geng, Hasan Iqbal, Dhruv Sahnan, Iryna Gurevych, and Preslav Nakov.
2025. FIRE: fact-checking with iterative retrieval and verification. InFindings of the Association for Computational
Linguistics: NAACL 2025. 2901–2914.
[39] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective Retrieval Augmented Generation.
arXiv:2401.15884
[40] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang,
Chenxu Lv, et al. 2025. Qwen3 technical report.arXiv preprint arXiv:2505.09388(2025).
[41] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2023. React: Syner-
gizing reasoning and acting in language models. InThe eleventh international conference on learning representations.
[42] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2023. Making retrieval-augmented language models robust
to irrelevant context.arXiv preprint arXiv:2310.01558(2023).
[43] Wenhao Yu, Hongming Zhang, Xiaoman Pan, Peixin Cao, Kaixin Ma, Jian Li, Hongwei Wang, and Dong Yu. 2024.
Chain-of-note: Enhancing robustness in retrieval-augmented language models. InProceedings of the 2024 conference on
empirical methods in natural language processing. 14672–14685.
[44] Zhenrui Yue, Huimin Zeng, Yimeng Lu, Lanyu Shang, Yang Zhang, and Dong Wang. 2024. Evidence-driven retrieval
augmented response generation for online misinformation. InProceedings of the 2024 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers).
5628–5643.
[45] Xinran Zhao, Aakanksha Naik, Jay DeYoung, Joseph Chee Chang, Jena D Hwang, Tongshuang Wu, and Varsha Kishore.
2026. Improving Attributed Long-form Question Answering with Intent Awareness.arXiv preprint arXiv:2603.27435
(2026).
[46] Yilun Zhao, Lyuhao Chen, Arman Cohan, and Chen Zhao. 2024. TaPERA: Enhancing faithfulness and interpretability
in long-form table QA by content planning and execution-based reasoning. InProceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers). 12824–12840.
[47] Yuqicheng Zhu, Nico Potyka, Daniel Hernández, Yuan He, Zifeng Ding, Bo Xiong, Dongzhuoran Zhou, Evgeny
Kharlamov, and Steffen Staab. 2025. Argrag: Explainable retrieval augmented generation using quantitative bipolar
argumentation.arXiv preprint arXiv:2508.20131(2025).
ACM Trans. Inf. Syst., Vol. 1, No. 1, Article . Publication date: April 2026.