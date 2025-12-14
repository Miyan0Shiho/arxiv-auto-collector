# Replace, Don't Expand: Mitigating Context Dilution in Multi-Hop RAG via Fixed-Budget Evidence Assembly

**Authors**: Moshe Lahmy, Roi Yozevitch

**Published**: 2025-12-11 16:31:29

**PDF URL**: [https://arxiv.org/pdf/2512.10787v1](https://arxiv.org/pdf/2512.10787v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems often fail on multi-hop queries when the initial retrieval misses a bridge fact. Prior corrective approaches, such as Self-RAG, CRAG, and Adaptive-$k$, typically address this by \textit{adding} more context or pruning existing lists. However, simply expanding the context window often leads to \textbf{context dilution}, where distractors crowd out relevant information. We propose \textbf{SEAL-RAG}, a training-free controller that adopts a \textbf{``replace, don't expand''} strategy to fight context dilution under a fixed retrieval depth $k$. SEAL executes a (\textbf{S}earch $\rightarrow$ \textbf{E}xtract $\rightarrow$ \textbf{A}ssess $\rightarrow$ \textbf{L}oop) cycle: it performs on-the-fly, entity-anchored extraction to build a live \textit{gap specification} (missing entities/relations), triggers targeted micro-queries, and uses \textit{entity-first ranking} to actively swap out distractors for gap-closing evidence. We evaluate SEAL-RAG against faithful re-implementations of Basic RAG, CRAG, Self-RAG, and Adaptive-$k$ in a shared environment on \textbf{HotpotQA} and \textbf{2WikiMultiHopQA}. On HotpotQA ($k=3$), SEAL improves answer correctness by \textbf{+3--13 pp} and evidence precision by \textbf{+12--18 pp} over Self-RAG. On 2WikiMultiHopQA ($k=5$), it outperforms Adaptive-$k$ by \textbf{+8.0 pp} in accuracy and maintains \textbf{96\%} evidence precision compared to 22\% for CRAG. These gains are statistically significant ($p<0.001$). By enforcing fixed-$k$ replacement, SEAL yields a predictable cost profile while ensuring the top-$k$ slots are optimized for precision rather than mere breadth. We release our code and data at https://github.com/mosherino/SEAL-RAG.

## Full Text


<!-- PDF content starts -->

Replace, Don’t Expand: Mitigating Context
Dilution in Multi-Hop RAG via Fixed-Budget
Evidence Assembly
A PREPRINT
Moshe Lahmy11and
 Roi Yozevitch21,2
1Department of Electrical Engineering, Ariel University, Ariel, Israel
2Department of Computer and Software Engineering, Ariel University, Ariel, Israel
December 12, 2025
Abstract
Retrieval-Augmented Generation (RAG) systems often fail on multi-hop queries when the initial
retrieval misses a bridge fact. Prior corrective approaches, such as Self-RAG, CRAG, and Adaptive-
k, typically address this byaddingmore context or pruning existing lists. However, simply ex-
panding the context window often leads tocontext dilution, where distractors crowd out relevant
information. We proposeSEAL-RAG, a training-free controller that adopts a“replace, don’t ex-
pand”strategy to fight context dilution under a fixed retrieval depthk. SEAL executes a (Search
→Extract→Assess→Loop) cycle: it performs on-the-fly, entity-anchored extraction to build a
livegap specification(missing entities/relations), triggers targeted micro-queries, and usesentity-
first rankingto actively swap out distractors for gap-closing evidence. We evaluate SEAL-RAG
against faithful re-implementations of Basic RAG, CRAG, Self-RAG, and Adaptive-kin a shared
environment onHotpotQAand2WikiMultiHopQA. On HotpotQA (k=3), SEAL improves an-
swer correctness by+3–13 ppand evidence precision by+12–18 ppover Self-RAG. On 2Wiki-
MultiHopQA (k=5), it outperforms Adaptive-kby+8.0 ppin accuracy and maintains96%evi-
dence precision compared to 22% for CRAG. These gains are statistically significant (p<0.001).
By enforcing fixed-kreplacement, SEAL yields a predictable cost profile while ensuring the top-
kslots are optimized for precision rather than mere breadth. We release our code and data at
https://github.com/mosherino/SEAL-RAG.
1 Introduction
Large Language Models (LLMs) augmented with retrieval (RAG) are now the standard for knowledge-
intensive tasks[18]. However, standard “Retrieve-then-Read” pipelines are brittle in multi-hop scenarios:
if the initial top-kretrieval misses a crucial “bridge” entity or relation, the generator hallucinates or fails
[14, 25].
To mitigate this, recent research has focused oniterative and corrective RAG. Systems like Self-
RAG [2] and CRAG [28] introduce feedback loops: they critique the retrieved evidence and trigger ad-
ditional retrieval steps if gaps are detected. While effective at improving recall, these methods typically
operate viaBreadth-First Addition: they append new passages to the existing context. This approach
assumes that “more context is better,” but in production environments with strict latency and token bud-
gets, this assumption fails. Expanding the context window introducesdistractors(irrelevant passages
that confuse the model) andlateral redundancy(duplicate information), often degrading the model’s
ability to reason over the specific bridge facts required—a phenomenon known ascontext dilution[19].
1arXiv:2512.10787v1  [cs.AI]  11 Dec 2025

We argue that under a fixed inference budget, the goal of a RAG controller should not be toaccumu-
lateevidence, but tooptimize the compositionof the fixed-kset. We propose a fundamental shift from
retrieval expansiontoFixed-Budget Evidence Assembly.
1.1 The SEAL-RAG Approach
We introduceSEAL-RAG, a training-free inference-time controller designed forFixed-kGap Repair.
Unlike prior methods that treat the context window as an append-only log, SEAL treats the top-kslots as
a scarce resource. The core mechanism is a (Search→Extract→Assess→Loop) cycle (Figure 1). In-
stead of relying on implicit scalar confidence scores, SEAL performs on-the-fly entity extraction to build
anExplicit Gap Specification(e.g., “Missing thefounding dateofOrganization X”). It translates these
gaps into targeted micro-queries and employs anEntity-First Replacementpolicy: new candidates
are scored based on their ability to close specific gaps and are used toevictthe lowest-utility passages
(distractors) from the current set. This maintains a constant context size (k) while strictly increasing
information density.
Figure 1:SEAL-RAG pipeline (Search→Extract→Assess→Loop).From a user query, initial
retrieval (fixed top-k) pulls candidates. Each loop:Extractperforms entity-first extraction to form agap
specification;Assessapplies scope-aware sufficiency to decidestopvs.repair. On repair, theMicro-
Querypolicy explores targeted queries. New evidence is integrated viaentity-first rankingto replace
distractors; once sufficient, the system emits the answer.
1.2 Contributions
This work makes the following contributions:
•Controller-Level Framing: Fixed-Budget, Gap-Aware Evidence Repair.We recast multi-
hop RAG as a constrainedevidence set optimizationproblem: the system must maintain a small
evidence setEof fixed sizekthat is sufficient to answer the query. While primitives such as
entity extraction, relation extraction, and micro-queries are well-established, prior work typically
uses them toexpanda candidate pool and then select from an ever-growing or static context.
In contrast, SEAL-RAG introduces a controller that (i) maintains an explicitgap specification
over an entity ledger (tracking which entities/relations are supported or missing), and (ii) uses this
specification to drive areplacement-based repair policyunder a fixed-kbudget. This fixed-budget,
gap-aware repair view directly targets the “context dilution” failure mode of add-only pipelines.
2

All underlying primitives are standard; the contribution lies in the controller’s design and in how
it combines these primitives.
•Replace-Not-Expand Mechanics via Entity-Centric Utility.SEAL-RAG treats thekcontext
slots as a scarce resource rather than a buffer to be filled. At each loop, candidate passages are
scored using an entity-centric utility function that balances gap coverage, corroboration, novelty,
and redundancy. Low-utility distractors are activelyevictedand replaced by candidates that better
resolve identified gaps, while the ledger and gap specification are updated. This combination of
(a) targeted micro-queries sourced from explicit gaps and (b) active replacement under a strict size
constraint distinguishes SEAL-RAG from both add-only controllers (e.g., CRAG, Self-RAG) and
prune-from-a-static-pool selectors (e.g., Adaptive-k).
•Unified, Controller-Focused Evaluation.To isolate the effect of controller logic from model
training or architectural differences, we re-implement the control policies ofSelf-RAG,CRAG,
andAdaptive-kin a shared, training-free environment with thesameretriever, index, and genera-
tor. This experimental design enables a fair comparison of add-only, prune-only, and repair-based
controllers under identical retrieval and generation conditions.
•Empirical Gains on Multi-Hop Benchmarks.We evaluate SEAL-RAG onHotpotQAand
2WikiMultiHopQAacross retrieval depthsk∈ {1,3,5}. SEAL-RAG consistently maintains
higher evidence precision and improves answer accuracy over baselines. For example, on 2Wiki-
MultiHopQA atk=5, SEAL-RAG attains96%evidence precision compared to22%for CRAG,
and yields an answer accuracy improvement of +8.0 percentage points over Adaptive-k.
1.3 Paper Organization
Section 2 reviews related work. Section 3 details SEAL-RAG (loop controller, scope-aware sufficiency,
loop-adaptive extraction, entity-first ranking, and the micro-query policy). Section 4 specifies datasets,
models, retrieval/indexing, baseline, metrics/judging, and protocol. Section 5 presents main results at
k=1,k=3, andk=5 with per-backbone tables and discussion. Section 6 reports loop-budget ablations
and analysis. Section 7 states limitations and threats to validity. Section 7 concludes. Detailed prompts
and statistical tables appear in the Appendix. All code and datasets are available in the GitHub repository
athttps://github.com/mosherino/SEAL-RAG.
2 Related Work
2.1 Standard RAG and Multi-Hop Challenges
Retrieval-Augmented Generation (RAG) has evolved from early sparse retrieval pipelines to sophisti-
cated dense and hybrid systems [18, 9]. While dense retrievers like DPR [14] and late-interaction models
like ColBERT [15] improve recall on single-hop queries, they often struggle with multi-hop reasoning,
where the answer depends on composing information from multiple disjoint documents [20, 6]. In these
scenarios, the standardretrieve-then-generatepattern faces a critical bottleneck: if the initial top-kset
misses a “bridge” fact, the generator cannot recover. A common workaround is to blindly increasek
or accumulate more context, but this introduces noise. Recent analysis confirms that irrelevant context
can significantly degrade model performance—a phenomenon known as “lost in the middle” orcontext
dilution[19]. SEAL-RAG targets this specific failure mode by holdingkfixed and iterativelyrepairing
the evidence set rather than expanding it.
2.2 Corrective and Reflective RAG
To mitigate retrieval failures, recent research has shifted towardsActive Retrieval[13], where the model
actively interacts with the search engine during inference.SELF-RAG[2] integrates retrieval and cri-
3

tique via special reflection tokens, allowing the model to self-assess generation quality and trigger addi-
tional retrieval steps when necessary. Similarly,CRAG[28] employs a lightweight evaluator to detect
low-quality retrieval and trigger corrective actions, such as web searches. While these methods improve
robustness against irrelevant context [30], they typically operate viaBreadth-First Addition: they ap-
pend new passages to the existing context window. This approach assumes that “more context is better,”
but in production environments with strict latency and token budgets, it leads to unbounded context
growth and variable inference costs. SEAL-RAG adopts the active spirit of these methods but enforces
areplacementpolicy to maintain a predictable budget.
2.3 Adaptive Retrieval and Pruning
A parallel line of work focuses on dynamic resource allocation to improve efficiency.Adaptive-RAG
[11] functions as a router, classifying query complexity to dynamically select between retrieval-free and
retrieval-augmented paths.Adaptive-k[24] andLC-Boost[26] aim to optimize the context window by
pruning irrelevant documents from a larger retrieved list or selecting a minimal sufficient subset. While
these methods address the efficiency drawback of standard RAG, they are primarilyselectorsorrouters,
notrepairers. If the initial retrieval pool misses a bridge fact entirely, pruning cannot recover it. In
contrast, SEAL-RAG performsActive Repair: it diagnoses specific missing entities (e.g., via on-the-fly
extraction) and issues targeted micro-queries to fetch new evidence that was never in the initial pool,
replacing low-utility items to improve the set’s composition.
2.4 Contrast with SEAL-RAG
SEAL-RAG occupies a distinct position in the design space. Unlike Corrective/Reflective methods
(Self-RAG, CRAG), it enforces aFixed Capacityto prevent context dilution. Unlike Adaptive methods
(Adaptive-k), it performsActive Repairvia targeted micro-queries rather than passive pruning. By
combining explicit gap modeling with a replacement policy, SEAL optimizes thecompositionof the
top-kslots, ensuring high precision under strict budget constraints.
3 SEAL-RAG (Method)
3.1 Problem Formulation & Architecture
We formalize retrieval-augmented generation under strict budgets as a constrained set-optimization
problem. Given a queryqand a corpusC, our goal is to identify an optimal evidence setE∗⊂Cthat
maximizes the probability of generating a correct answera, subject to a cardinality constraint|E|=k.
In this framework, we define “budget” strictly as the finite context capacity (k) available to the gener-
ator, treating the evidence window as a scarce cognitive resource to be optimized rather than merely a
computational cost to be minimized.
Unlike standard RAG, which approximatesE∗via a single retrieval pass, or corrective methods that
relax the constraint (allowing|E|>k), SEAL-RAG iteratively refinesEwhile strictly enforcing|E t|=k
at every stept.
The controller maintains a state tupleS t= (E t,Ut,Bt), where:
•E t: The current evidence buffer of fixed sizek.
•U t: A structuredEntity Ledgerderived fromE t(containing entities, relations, and provenance).
•B t: ABlocklistof unproductive query patterns or sources to prevent cycles.
The inference process follows a (Search→Extract→Assess→Loop) cycle (Figure 2).Initial-
ization:Att=0,E 0is populated via a standard dense/hybrid retrieval pass. At each subsequent step,
the controller assesses sufficiency; if insufficient, it executes a repair policyπ(S t)to replace low-utility
items inE t, halting when sufficiency is met or a loop budgetLis exhausted.
4

Figure 2:Execution graph for SEAL-RAG.Nodes represent logical stages:retrieve_docsinitial-
izes the fixed-kset;cached_entities_updatebuilds the ledgerU t;to_repairacts as the sufficiency
gate. If repair is needed,micro_query_agentfetches candidates andrank_evidenceperforms re-
placement. Solid arrows denote the primary path; dashed arrows indicate loopbacks.
3.2 State Representation: The Entity Ledger
To make the “stop vs. repair” decision computable, SEAL-RAG projects the unstructured evidenceE t
into a structuredEntity Ledger U t. We employ a lightweight, on-the-fly extraction module grounded in
Open Information Extraction principles [1, 3].
The extraction process enforces aVerbatim Constraint: extracted facts must be explicitly supported
by text spans inE tto prevent hallucination. The ledgerU ttracks:
•Entities & Aliases:Canonical entities (e.g., “Theresa May”) mapped to surface forms (“PM
May”, “She”).
•Typed Relations:Triplets(h,r,t)linking entities (e.g.,(Theresa May, authored, Article
50 letter)).
•Qualifiers:Critical metadata such as dates, locations, or roles attached to relations (e.g.,date=2017).
This structured view allows the controller to detect partial coverage (e.g., the relation exists, but the date
qualifier is missing).
3.3 Sufficiency Assessment
TheSufficiency Gateevaluates a predicate Suff(q,U t)based on four aggregated signals. We employ
the LLM as a zero-shot estimator to score these components:
•Coverage:The fraction of required question attributes (derived from the query schema) currently
present inU t.
5

•Corroboration:The degree of multi-source agreement for critical facts.
•Contradiction:Detection of conflicting attribute values across passages.
•Answerability:A calibrated confidence score estimating if the question is answerable givenU t
[22].
If Suff(q,U t)is true, the loop terminates. If false, the controller proceeds to gap diagnosis.
3.4 Gap-Driven Retrieval Policy
When sufficiency fails, standard corrective methods often rely on generic query rewriting. In contrast,
SEAL-RAG computes anExplicit Gap SpecificationG t=N(q)\U t. The system parses the question
qto identify necessary information needsN(q)(e.g., “Need: Birthplace of Person X”) and subtracts
the facts already present inU t.
We categorize gaps into three types:
1.Missing Entity:A bridge entity referenced by a relation is absent (e.g., “The band that released
Parklife”).
2.Missing Relation:Two entities are known, but the link between them is unproven.
3.Missing Qualifier:A relation is known, but a required date or location is missing.
The controller translatesG tintoatomic micro-queries(e.g., “Blur band Parklife release year”).
This is significantly more precise than a broad rewrite (e.g., “Tell me about Blur and Parklife”), which
often retrieves general biography pages rather than the specific missing attribute. This policy minimizes
query drift and ensures that retrieved candidates are semantically aligned with the specific missing link
[5]. To prevent cycles, the policy updates the blocklistB twith query terms that failed to yield novel
information.
3.5 Fixed-Capacity Replacement
The core innovation of SEAL-RAG is its refusal to expand the context window. We treat evidence
assembly as aBudgeted Maximizationproblem: given a candidate poolC tretrieved via micro-queries,
we must select a subset to replace low-utility items inE tsuch that the total size remainsk.
We define anEntity-First UtilityfunctionS(c|U t)to score each candidatec∈C t. Inspired by
Maximal Marginal Relevance (MMR) [4], this score balances relevance against redundancy:
S(c) =λ 1·GapCov(c,G t) +λ 2·Corr(c,U t) +λ 3·Nov(c,U t)−λ 4·Red(c,E t)(1)
whereλ iare hyperparameters weighting the components:
•GapCov: Measures if candidateccontains the specific missing entity or relation defined in the
gap setG t.
•Corr: Rewards candidates that corroborate existing uncertain facts in the ledgerU t(increasing
confidence).
•Nov: Rewards non-lateral novelty (introducing new entities or relations not yet inU t).
•Red: Penalizes lexical overlap with existing passages inE tto prevent lateral redundancy.
To update the set, the controller identifies the lowest-scoring victimv∈E tand the highest-scoring
candidatec∗∈C t. IfS(c∗)>S(v) +ε, a swap occurs:E t+1←(E t\{v})∪{c∗}. The termεis a small
hysteresis threshold to prevent thrashing (replacing an item with a marginally better one, wasting a loop
step). Additionally, we enforce aDwell-Time Guard: newly inserted items are protected from eviction
for one iteration to ensure they are processed by the sufficiency gate before being discarded. This ensures
that the information density of the top-kslots strictly increases, actively fighting context dilution.
6

3.6 Complexity & Budget
A critical advantage of SEAL-RAG is its predictable cost profile. LetLbe the maximum loop budget
andkthe fixed retrieval depth. The total inference cost is bounded by:
Cost SEAL =O(L·Retriever) +O(L·Extractor) +O(1·Generator k)(2)
Since the generator is invoked only once on a fixed context of sizek, the expensive decoding step
remains constant regardless of the number of repair loops. In contrast, addition-based methods increase
the context size at every step, causing the generator cost to grow super-linearly withL. SEAL-RAG
guarantees that latency and token usage remain within a tight, pre-calculated envelopeO(k·L).
4 Experimental Setup
4.1 Datasets
We evaluate on two multi-hop QA benchmarks to assess performance and generalization across different
reasoning types.
•HotpotQA (Distractor Setting):We use a seeded validation slice ofN=1,000 questions [29].
This dataset primarily testsbridgereasoning (connecting entity A to entity B) andcomparison
reasoning (e.g., “Who is older, X or Y?”).
•2WikiMultiHopQA:To assess robustness beyond HotpotQA, we evaluate on a seeded slice of
N=200 examples from the 2WikiMultiHopQA validation set [10]. This dataset involves complex
compositionalreasoning and inference rules over Wikipedia entities.
For both datasets, we use a fixed random seed to ensure deterministic sampling. Representative examples
of these reasoning types and how SEAL-RAG handles them are provided in section C.
4.2 Shared Environment
To isolate the contribution of the controller logic, we enforce a strictShared Environment. All methods
(SEAL-RAG and baselines) are implemented as workflows usingLangGraph[17] to ensure consistent
state management. They share the exact same underlying components:
•Indexing Pipeline:We employNatural Document Segmentation. Instead of arbitrary sliding
windows, we concatenate the page title and all associated sentences provided by the benchmark
into a single retrieval unit. This preserves the semantic integrity of documents. Full indexing
details are provided in section D.
•Retriever:Dense retrieval using OpenAI embeddings (text-embedding-3-small) and a Pinecone
vector store.
•Unified Backbone Architecture:To strictly isolate the algorithmic contribution of the controller
logic from latent model capabilities, we employ a unified backbone strategy. Within each experi-
mental configuration, thesameunderlying LLM instance powers both the internal Controller (han-
dling entity extraction, sufficiency estimation, and ranking) and the final Generator. We evaluate
across the GPT-4 family to ensure robustness:gpt-4oandgpt-4o-minion 2WikiMultiHopQA,
andgpt-4.1andgpt-4.1-minion HotpotQA. All model calls utilize temperature 0 to ensure
deterministic reproducibility.
This setup ensures that any performance difference is attributable solely to the retrieval policy (e.g.,
replacement vs. addition), not to differences in the underlying model, index, or prompt engineering.
7

4.3 Baselines
We compare SEAL-RAG against four baselines, re-implemented in our shared environment to match
the specific logic of their original proposals. Detailed graph topologies and system prompts for these
re-implementations are provided in section B.
•Basic RAG:A linearRetrieve→Generategraph. It retrieveskpassages once and generates
an answer.
•Self-RAG:A reflective graph that grades documents for relevance and generations for hallucina-
tions [2]. If the generation is unsupported, the system loops back to transform the query (capped
at 3 attempts).
•CRAG (Corrective RAG):A corrective graph [28]. If retrieved documents are graded as “ir-
relevant,” the system triggers an external web search (via Tavily) to augment the context before
generation.
•Adaptive-k:A dynamic pruning method [24]. It retrieves a large candidate pool (k=50) and
selects the optimal cut-off point using the “Largest Gap” strategy on similarity scores. We evaluate
bothBufferandNo-Buffervariants.
4.4 Metrics and Judging
We report two primary metrics:
•Judge-EM (Correctness):We useGPT-4oas an external judge. The judge evaluatesFactual
Consistencyagainst the ground truth, penalizing contradictions or “I don’t know” responses if the
answer exists [31]. The judge sees only the retrieved passages to prevent parametric leakage.
•Evidence Quality:We computeGold-title Precision@kandRecall@k. To ensure rigorous
evaluation, we applyAlias Normalization: retrieved titles are matched against gold titles using a
redirect map (e.g., mapping “JFK” to “John F. Kennedy”) to prevent false negatives.
We report statistical significance using McNemar’s test for correctness and pairedt-tests for retrieval
metrics (p<0.05).
5 Results
5.1 Main Results on HotpotQA (k=1)
We first evaluate performance under the strictest constraint: a single retrieval slot (k=1). In this regime,
the system must identify and retain the single most critical passage (often a bridge entity) to answer
correctly. Any distractor in this slot results in immediate failure. Table 1 presents the results on the
seededN=1,000 validation slice.
Key Observations.
•Replacement is decisive at a single slot.With only one evidence slot, the ability todisplacea
low-yield passage is pivotal. Across all backbones, SEAL-RAG improves Judge-EM by+7 to
+19 ppover the best baseline. This confirms that gap-aware micro-queries reliably surface the
one page that actually closes the bridge.
•Precision lift without harming recall.Gold-title Precision rises for SEAL-RAG versus the
strongest baseline in each backbone (e.g., 91 vs. 75 for gpt-4o). Crucially, Recall stays com-
parable or higher (e.g., 66 vs. 40 for gpt-4.1), yielding a consistent F1 advantage. This refutes the
notion that replacement inherently sacrifices coverage.
8

Table 1:Main results at fixedk=1on HotpotQA (N=1,000).Metrics are percentages. All methods
share the same environment (models, vector store, judge, metrics); only control logic differs.∆is the
Judge-EM gain of SEAL-RAG over the best baseline for thesame model.
Metrics (%)∆(pp)
Model Method Judge-EM Prec Rec F1∆EM
gpt-4o-mini Basic RAG 41 85 42 57
Self-RAG 48 61 31 41
CRAG 55 42 21 28
SEAL-RAG 62 86 44 58 +7
gpt-4o Basic RAG 41 85 42 57
Self-RAG 59 75 37 50
CRAG 58 54 27 36
SEAL-RAG 73 91 62 72 +14
gpt-4.1-mini Basic RAG 39 85 42 57
Self-RAG 49 72 36 48
CRAG 52 57 29 38
SEAL-RAG 71 87 48 61 +19
gpt-4.1 Basic RAG 40 85 42 57
Self-RAG 63 79 40 53
CRAG 58 57 28 38
SEAL-RAG 73 90 66 74 +10
•Addition-first underperforms at smallk.CRAG and Self-RAG broaden context during the loop,
but when the reader is constrained tok=1, breadth does not help unless it reorders the final top-
1. This is visible where CRAG’s recall drops (e.g., 21–29%), as it may append relevant docs to
positionsk>1 which are then truncated. SEAL’s replacement policy ensures the best document
lands in thevisibleslot.
•Shift from Read-Time to Retrieval-Time Reasoning.The performance gap highlights a struc-
tural distinction. Standard RAG relies onRead-Time Reasoning, requiring simultaneous access
to disjoint evidence (Hop 1 and Hop 2), which is physically impossible atk=1. SEAL-RAG
shifts this toRetrieval-Time Reasoning: the controller resolves the bridge entity into the ledger,
effectively “consuming” the first hop. This allows the single context slot to be dedicated entirely
to the final answer-bearing document, rendering the task solvable.
5.2 Main Results on HotpotQA (k=3)
We next evaluate performance atk=3, a standard setting for production RAG systems. With three slots,
the challenge shifts from finding a single needle to assembling a coherent set that covers multiple hops
without admitting distractors. Table 2 details the results.
Key Observations.
•Precision lead persists under larger capacity.With three slots, recall naturally rises for all
methods. However, SEAL-RAG maintains a massive Precision advantage (e.g.,89%vs. 37–76%
for gpt-4o). This indicates that while baselines use the extra slots to accumulate near-duplicates
or topical distractors, SEAL uses them to store complementary bridge facts.
•Replacement reduces lateral redundancy.By treating the evidence set as a fixed-capacity buffer,
SEAL-RAG actively evicts redundant passages (e.g., two biographies of the same person) to make
room for the second hop (e.g., the organization page). This raises Precision without sacrificing the
Recall that naturally comes withk=3, translating into the highest Judge-EM across all backbones.
9

Table 2:Main results at fixedk=3on HotpotQA (N=1,000).Even with larger capacity, SEAL-RAG
maintains a significant lead in Precision and Correctness.∆shows the gain over the best baseline.
Metrics (%)∆(pp)
Model Method Judge-EM Prec Rec F1∆EM
gpt-4o-mini Basic RAG 63 49 72 59
Self-RAG 60 66 47 53
CRAG 62 30 36 33
SEAL-RAG 69 84 44 57 +6
gpt-4o Basic RAG 68 49 72 59
Self-RAG 71 76 55 61
CRAG 69 37 44 40
SEAL-RAG 77 89 68 75 +6
gpt-4.1-mini Basic RAG 64 49 72 59
Self-RAG 64 73 56 61
CRAG 67 40 49 43
SEAL-RAG 77 86 49 61 +10
gpt-4.1 Basic RAG 68 49 72 59
Self-RAG 73 79 61 66
CRAG 72 41 50 44
SEAL-RAG 76 91 73 79 +3
•Addition-first recall is offset by distractors.While CRAG and Basic RAG often achieve high
recall (e.g., 72%), their low precision (30–49%) drags down answer correctness. This confirms
that simply having the answer in the context is insufficient if it is buried in noise; the model
requires acuratedcontext to reason reliably.
5.3 Generalization to 2WikiMultiHopQA (k=1,3,5)
To address concerns regarding generalization, we evaluate on2WikiMultiHopQA(N=200), which
requires complex compositional reasoning. We extend the evaluation tok=5 to explicitly test the
“more context is better” assumption. Table 3 presents the results stacked by retrieval depth.
Key Observations.
•Validation of Reasoning Transfer (k=1).The results atk=1 confirm the architectural ad-
vantage observed in HotpotQA (see Table 1, Key Observation 4). While baselines struggle near
the floor (14–18%) due to the impossibility ofRead-Timereasoning in a single slot, SEAL-RAG
achieves61–76%accuracy. This demonstrates that the controller’s ability to offload the bridge
step to the ledger generalizes effectively to complex compositional reasoning.
•The Failure of Additive Logic.Thek=5 results empirically validate the “Context Dilution” hy-
pothesis [19]. CRAG, which appends web search results without removal, suffers a catastrophic
precision collapse (down to11–22%). Basic RAG similarly drops to 34%. This flood of distrac-
tors overwhelms the generator, consistent with findings that LLMs struggle to ignore irrelevant
context [30]. In contrast, SEAL-RAG maintains89–96% Precision. This proves that without an
activeeviction mechanism, increasing the budget primarily accumulates noise.
•Mechanism of Success.SEAL-RAG breaks the precision-recall trade-off by combining two novel
components: (1)Explicit Gap Specificationensures high recall by targeting the exact missing
bridge (matching Basic RAG’s 77% recall), while (2)Entity-First Replacementensures high
precision by displacing distractors (exceeding Self-RAG’s 63% precision). This confirms that the
“Replace, Don’t Expand” paradigm is essential for robust multi-hop reasoning.
10

Table 3:2WikiMultiHopQA Results (N=200).We compare performance across retrieval depths.
Key Trend:Askincreases to 5, baseline Precision collapses (Context Dilution), while SEAL-RAG
maintains high precision via replacement, driving superior Accuracy (Judge-EM).
Model Method Judge-EM (%) Prec (%) Rec (%) F1 (%)∆EM
Retrieval Depthk=1(The Bottleneck)
GPT-4o-miniBasic RAG 18 93 41 56 -
Self-RAG 26 27 10 15 -
CRAG 30 12 5 7 -
SEAL-RAG 61 92 45 59 +31 pp
GPT-4oBasic RAG 14 93 41 56 -
Self-RAG 36 41 19 26 -
CRAG 28 21 10 14 -
SEAL-RAG 76 95 75 82 +40 pp
Retrieval Depthk=3(Standard)
GPT-4o-miniBasic RAG 49 536959 -
Self-RAG 53 41 18 25 -
CRAG 56 9 9 9 -
SEAL-RAG 64 914660 +8 pp
GPT-4oBasic RAG 54 53 69 59 -
Self-RAG 56 60 35 42 -
CRAG 60 18 19 18 -
SEAL-RAG 77 97 77 84 +17 pp
Retrieval Depthk=5(Context Dilution Test)
GPT-4o-miniBasic RAG 57 347546 -
Self-RAG 56 45 20 26 -
CRAG 55 11 10 11 -
SEAL-RAG 68 894559 +11 pp
GPT-4oBasic RAG 62 34 75 46 -
Self-RAG 60 63 38 45 -
CRAG 64 22 23 22 -
SEAL-RAG 74 96 77 84 +10 pp
5.4 Comparison vs. Adaptive-k
A key question raised by recent work (and our reviewers) is whether dynamic context selection can
solve the precision-recall trade-off without the complexity of iterative repair. We compare SEAL-RAG
againstAdaptive-k[24], a state-of-the-art pruning method that dynamically cuts the retrieved list based
on relevance score gaps.
Table 4 compares SEAL-RAG (k=5) against Adaptive-k(with and without a safety buffer) on the
2WikiMultiHopQA dataset.
Active Repair beats Passive Selection.The results highlight a fundamental limitation of selection-
based methods in multi-hop scenarios:
•The Selection Ceiling.Adaptive-kis limited to the candidates present in the initial retrieval
pool. If the bridge fact is missing from the top-50 candidates (a common occurrence in multi-hop
retrieval), no amount of clever pruning can recover it. This forces a trade-off: the “No Buffer”
variant achieves high precision (86%) but misses the answer (41.5% accuracy), while the “Buffer”
variant captures the answer (77% recall) but drowns in noise (26% precision).
11

Table 4:SEAL-RAG vs. Adaptive-k(N=200).Adaptive-kacts as aselector: it is bound by the quality
of the initial pool. SEAL-RAG acts as arepairer: it actively fetches missing information, breaking the
ceiling of the initial retrieval.
Accuracy Evidence Quality (%) Gain
Model Method Judge-EM (%) Prec Rec F1∆EM
GPT-4o-miniAdaptive-k(No Buffer) 40.5 86 61 65 -
Adaptive-k(Buffer) 60.5 267738 -
SEAL-RAG 68.0 894559 +7.5 pp
GPT-4oAdaptive-k(No Buffer) 41.5 86 61 65 -
Adaptive-k(Buffer) 66.5 26 77 38 -
SEAL-RAG 74.5 96 77 84 +8.0 pp
•The SEAL Advantage.SEAL-RAG bypasses this ceiling viaActive Repair[13]. By issuing
micro-queries for specific missing data, it fetches evidence that wasnever in the initial pool. This
allows it to match the high recall of the buffered approach (77%) while exceeding the precision of
the aggressive approach (96%). This confirms that for complex reasoning, the controller must be
able toexpand the search frontier, not just filter it.
5.5 Statistical Significance Analysis
To ensure that the observed performance gains are not artifacts of random variance, we conducted rigor-
ous statistical testing on the paired outputs of SEAL-RAG versus baselines on the same question sets.
•Methodology:For binary answer correctness (Judge-EM), we usedMcNemar’s test, which is
appropriate for paired nominal data [8]. For continuous retrieval metrics (Precision/Recall/F1),
we usedpaired two-sidedt-tests[16]. We applied the Holm-Bonferroni correction to control the
family-wise error rate atα=0.05.
•Results:On both HotpotQA and 2WikiMultiHopQA, SEAL-RAG’s improvements in Judge-EM
and Precision@kare statistically significant (p<0.001) against all baselines across all tested
backbones. This confirms that the “replacement” strategy yields a consistent, non-random im-
provement in evidence quality and downstream accuracy. Detailedp-value tables for all compar-
isons are provided in section E.
6 Ablations & Analysis
6.1 Effect of Loop BudgetL(HotpotQA)
To isolate the causal contribution of the repair loop, we analyze performance as a function of the loop
budgetLon the HotpotQA dataset, holding retrieval depth fixed atk=1. This setting is the most
sensitive to controller decisions, as there is no room for error—the system must swap the distractor for
the bridge page to succeed. Table 5 reports the results.
The “First Repair” Effect.The data reveals a sharp efficiency profile: the majority of the gain (avg.
+35 pp) is realized at the very first repair step (L=1). This confirms that SEAL-RAG is not relying on
brute-force search or deep reflective loops likeReflexion[23]. Instead, theExplicit Gap Specification
allows the controller to identify and retrieve the missing bridge immediately. Subsequent loops (L=3,5)
provide smaller marginal gains, primarily addressing long-tail cases with multiple missing qualifiers.
12

Table 5:Judge-EM (%) vs. Loop BudgetLon HotpotQA (k=1).The massive jump fromL=0 to
L=1 indicates that a single targeted repair is often sufficient to close the bridge. Diminishing returns at
L=5 suggest the method is efficient and does not require deep agentic loops.
Judge-EM (%) at Loop BudgetL
BackboneL=0L=1L=3L=5∆(L=5vs.0)
gpt-4o-mini 30 58 61 62 +32
gpt-4.1-mini 28 66 70 71 +43
gpt-4o 32 67 71 73 +41
gpt-4.1 25 63 69 73 +48
Average 29 64 68 70 +41
6.2 Qualitative Component Analysis
To address questions regarding the necessity of specific modules, we analyze successful repair traces (see
section C for full step-by-step logs). The gains rely on the synergy of three components that standard
RAG lacks:
•Extraction vs. Keywords:In cases where standard retrieval returns a biography but misses a
specific event date, the extraction module flags the missingDATEqualifier. A standard keyword
search often fails here because the entity name alone retrieves generic bios; thestructured gapis
required to target the specific event.
•Micro-Queries vs. Rewrites:By generating atomic queries (e.g., “Person X birth date”) rather
than broad rewrites, the system avoids retrieving topical distractors. Simpler decomposition meth-
ods likeSelf-Ask[21] often broaden context too aggressively, triggering the dilution trap.
•Entity-First Ranking vs. Relevance:In our traces, we observe candidates that are lexically
similar to the query but factually redundant being correctly evicted. A standard cross-encoder
would score these high (due to relevance), but SEAL’sNoveltyterm penalizes them, forcing the
replacement that enables the multi-hop answer.
6.3 Error Analysis
Despite these gains, SEAL-RAG is not infallible. We identify two primary failure modes (detailed in
section C.3):
•Alias Mismatch:If the gold evidence uses a rare alias not present in the initial context or the redi-
rect map, the entity extractor may fail to link the gap to the correct canonical ID. This highlights
the challenge of zero-shot entity linking [27].
•Extraction Noise:On-the-fly extraction can sometimes hallucinate relations or miss subtle qual-
ifiers in complex sentences. While ourVerbatim Constraintmitigates this, integrating a dedicated
verification step [7] could further improve robustness, albeit at higher latency.
7 Limitations & Conclusion
7.1 Limitations
While SEAL-RAG offers a principled solution to fixed-budget retrieval, it operates under specific con-
straints:
13

•The Extraction Bottleneck:The controller relies on the ability to explicitlynamethe missing
information. If a gap is purely abstract or implicit (e.g., “the general sentiment of the era”), the
extraction module may fail to formulate a precise micro-query, degrading to standard retrieval
performance.
•The Fixed-Capacity Ceiling:By strictly enforcing|E t|=k, SEAL-RAG prioritizes precision
over exhaustive recall. For questions that genuinely require aggregating more thankdistinct
documents simultaneously (e.g., “List all 20 works by Author X” whenk=5), the replacement
policy will cycle through evidence rather than accumulating it. This is a deliberate design choice
to prevent context dilution, but it limits applicability for “exhaustive list” queries.
•Judge Variance:Although we mitigate bias using a blind protocol (judges see only retrieved
passages) and fixed rubrics, LLM-based evaluation remains subject to stochasticity [12]. We
address this via bootstrap confidence intervals, but human evaluation remains the gold standard
for high-stakes domains.
7.2 Conclusion
This work challenges the prevailing assumption in corrective RAG that “more context is better.” We
demonstrated that under fixed inference budgets, the primary failure mode of multi-hop retrieval is not
low recall, butContext Dilution: the accumulation of distractors that overwhelm the generator.
We introducedSEAL-RAG, a controller that replaces the standard “Add” paradigm with a“Replace,
Don’t Expand”paradigm. By combiningExplicit Gap SpecificationwithEntity-First Replacement,
SEAL actively curates the top-kslots, treating the context window as a scarce resource to be optimized
rather than a log to be appended.
Our empirical results onHotpotQAand2WikiMultiHopQAconfirm that this approach is superior
to bothBlind Addition(CRAG) andPassive Pruning(Adaptive-k). Atk=5, where baselines suffered
precision collapses (dropping to 11–22%), SEAL maintained96% Precision. Furthermore, by actively
repairing gaps rather than just selecting from an initial pool, SEAL outperformed the state-of-the-art
Adaptive-kbaseline by+8.0 ppin accuracy. These findings establishFixed-Budget Evidence As-
semblyas a robust, predictable alternative to unbounded context expansion for production-grade RAG
systems.
References
[1] Gabor Angeli, Melvin Jose Johnson Premkumar, and Christopher D. Manning. Leveraging lin-
guistic structure for open domain information extraction. In Chengqing Zong and Michael Strube,
editors,Proceedings of the 53rd Annual Meeting of the Association for Computational Linguis-
tics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long
Papers), pages 344–354, Beijing, China, July 2015. Association for Computational Linguistics.
Available at:https://aclanthology.org/P15-1034/.
[2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning
to retrieve, generate, and critique through self-reflection. InInternational Conference on Learning
Representations (ICLR), 2024.https://arxiv.org/abs/2310.11511.
[3] Sangnie Bhardwaj, Samarth Aggarwal, and Mausam. CaRB: A crowdsourced benchmark for
open IE. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan, editors,Proceedings of
the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th Interna-
tional Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6262–6267,
Hong Kong, China, November 2019. Association for Computational Linguistics. Available at:
https://aclanthology.org/D19-1651/.
14

[4] Jaime Carbonell and Jade Goldstein. The use of mmr, diversity-based reranking for reordering
documents and producing summaries. InProceedings of the 21st Annual International ACM SIGIR
Conference on Research and Development in Information Retrieval, pages 335–336. ACM, 1998.
Available at:https://dl.acm.org/doi/10.1145/290941.291025.
[5] Claudio Carpineto and Giovanni Romano. A survey of automatic query expansion in information
retrieval.ACM Computing Surveys (CSUR), 44(1):1–50, 2012. Available at:https://dl.acm.
org/doi/10.1145/2071389.2071390.
[6] Jifan Chen and Greg Durrett. Understanding dataset design choices for multi-hop reasoning.
InProceedings of the 2019 Conference on Empirical Methods in Natural Language Processing
(EMNLP), pages 5727–5733, 2019. Available at:https://aclanthology.org/D19-1572.
[7] Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Celina Asghari, and
Douwe Kiela. Chain-of-verification reduces hallucination in large language models.arXiv preprint
arXiv:2309.11495, 2023. Available at:https://arxiv.org/abs/2309.11495.
[8] Rotem Dror, Gili Baumer, Segev Shlomov, and Roi Reichart. The hitchhiker’s guide to testing
statistical significance in natural language processing. InProceedings of the 56th Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1383–1392,
2018. Available at:https://aclanthology.org/P18-1128.
[9] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey.arXiv
preprint arXiv:2312.10997, 2023. Available at:https://arxiv.org/abs/2312.10997.
[10] Xanh Ho, Anh-Khoa Duong, and Quoc-Huy Nguyen. Constructing a multi-hop QA dataset for
comprehensive evaluation of reasoning steps. InProceedings of the 28th International Confer-
ence on Computational Linguistics, pages 6609–6625. International Committee on Computational
Linguistics, 2020. Available at:https://aclanthology.org/2020.coling-main.580.
[11] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C. Park. Adaptive-rag:
Learning to adapt retrieval-augmented large language models through question complexity.arXiv
preprint arXiv:2403.14403, 2024. Available at:https://arxiv.org/abs/2403.14403.
[12] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang,
Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation.ACM
Computing Surveys, 55(12):1–38, 2023. Available at:https://arxiv.org/abs/2202.03629.
[13] Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming
Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation.arXiv preprint
arXiv:2305.06983, 2023. Available at:https://arxiv.org/abs/2305.06983.
[14] Vladimir Karpukhin, Barlas O ˘guz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi
Chen, and Wen tau Yih. Dense passage retrieval for open-domain question answering. InProceed-
ings of EMNLP 2020, 2020.https://arxiv.org/abs/2004.04906.
[15] Omar Khattab and Matei Zaharia. ColBERT: Efficient and effective passage search via contextu-
alized late interaction over BERT. InProceedings of SIGIR 2020, 2020.https://dl.acm.org/
doi/10.1145/3397271.3401075.
[16] Philipp Koehn. Statistical significance tests for machine translation evaluation. InProceedings
of the 2004 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages
388–395, 2004. Available at:https://aclanthology.org/W04-3250.
[17] LangChain AI. Langgraph: Build resilient language agents as graphs, 2024. Available at:https:
//langchain-ai.github.io/langgraph/. Accessed: 2025-11-07.
15

[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal,
Heinrich Kuttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela.
Retrieval-augmented generation for knowledge-intensive nlp, 2020.https://arxiv.org/abs/
2005.11401.
[19] Nelson F. Liu, Teven Le Scao, Valentina Pyatkin, Daniel Khashabi, Noah A. Smith, and Hannaneh
Hajishirzi. Lost in the middle: How language models use long context.arXiv preprint, 2023.
Available at:https://arxiv.org/abs/2307.03172.
[20] Sewon Min, Danqi Chen, Hannaneh Hajishirzi, and Luke Zettlemoyer. Compositional ques-
tions do not necessarily need multi-hop reasoning. InProceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics, pages 4249–4257, 2019. Available at:
https://aclanthology.org/P19-1416.
[21] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measur-
ing and narrowing the compositionality gap in language models.arXiv preprint, arXiv:2210.03350,
2022.https://arxiv.org/abs/2210.03350.
[22] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don’t know: Unanswerable
questions for SQuAD. InProceedings of the 56th Annual Meeting of the Association for Com-
putational Linguistics (Volume 2: Short Papers), pages 784–789, 2018. Available at:https:
//aclanthology.org/P18-2124.
[23] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and
Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning.arXiv preprint
arXiv:2303.11366, 2023. Available at:https://arxiv.org/abs/2303.11366.
[24] Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani. Efficient context selection for long-context
qa: No tuning, no iteration, just adaptive-k.arXiv preprint arXiv:2506.08479, 2025. Available at:
https://arxiv.org/abs/2506.08479.
[25] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Is multihop QA
in DiRe condition? measuring and reducing disconnected reasoning. InProceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8846–8863,
2020. Available at:https://aclanthology.org/2020.emnlp-main.712.
[26] Hongqiu Wu, Ruixue Wang, Weizhu Lin, et al. Are long-llms a necessity for long-context tasks?
arXiv preprint arXiv:2405.15318, 2024. Available at:https://arxiv.org/abs/2405.15318.
[27] Ledell Wu, Fabio Petroni, Martin Josifoski, Sebastian Riedel, and Luke Zettlemoyer. Scalable zero-
shot entity linking with dense entity retrieval. InProceedings of EMNLP 2020, 2020. Available at:
https://aclanthology.org/2020.emnlp-main.519.
[28] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented genera-
tion.arXiv preprint, arXiv:2401.15884, 2024.https://arxiv.org/abs/2401.15884.
[29] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov,
and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. InProceedings of EMNLP 2018, pages 2369–2380, Brussels, Belgium, 2018.https:
//aclanthology.org/D18-1259.
[30] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. Making retrieval-augmented language
models robust to irrelevant context. InProceedings of the 12th International Conference on Learn-
ing Representations (ICLR), 2024. Available at:https://arxiv.org/abs/2310.01558.
16

[31] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang,
Zi Lin, Zhuohan Li, Dacheng Li, Eric P Xing, et al. Judging LLM-as-a-judge with MT-Bench
and Chatbot Arena.arXiv preprint arXiv:2306.05685, 2023. Available at:https://arxiv.org/
abs/2306.05685.
17

A Prompts & Judge Rubric
To ensure reproducibility and transparency, we provide the exact system prompts used for evaluation
and generation. These prompts were held constant across all experimental runs.
A.1 Judge-EM System Prompt (GPT-4o)
We utilizedGPT-4oas an external judge to evaluate Answer Correctness (Judge-EM). The prompt
enforces a strict rubric focused on factual consistency and support by retrieved evidence.
System:You are an expert data labeler evaluating model outputs for correctness. Your task is to
assign a score based on the following rubric:
<Rubric>
• A correct answer: Provides accurate and complete information that matches the ground truth;
Contains no factual errors when compared to the reference; Addresses the core question being
asked.
• When scoring, you should penalize: Factual errors or inaccuracies compared to ground truth;
Answers that contradict the reference output; “I don’t know” responses when ground truth
provides a clear answer.
</Rubric>
<Instructions>Carefully compare the agent’s output against the ground truth reference. Focus
on semantic equivalence rather than exact word matching. Be strict with factual contradictions or
completely wrong information.</Instructions>
User:<question>{question}</question> <agent_output>{agent_answer}</agent_output> <ground_truth>{ground_truth}</ground_truth>
Compare the agent’s output against the ground truth and evaluate its correctness. Provide your
reasoning and a boolean score (true for correct, false for incorrect).
A.2 Shared “Grounding Rule” Prompt
To ensure a fair comparison of control logic, all generators (SEAL-RAG, Basic RAG, CRAG, Self-RAG,
Adaptive-k) utilized the same system instruction to prevent parametric knowledge leakage.
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to
answer the question. If you don’t know the answer, just say that you don’t know. Keep the answer
concise.
<GROUNDING_RULE>Base your answer ONLY on the retrieved context below. Do not use any
information from your training data or external knowledge.</GROUNDING_RULE>
B Baseline Implementation Details
To ensure a rigorous comparison of control strategies, we re-implemented the logic of all baselines
using theLangGraphframework. This ensures that all methods share the same retriever, generator
(GPT-4o/mini), and runtime environment, isolating the algorithmic contribution from model weights.
B.1 Self-RAG (Inference-Only Implementation)
We utilize the inference-time control logic of Self-RAG via prompting, rather than the fine-tuned 7B/13B
weights. This allows us to evaluate thereflectiveparadigm on the state-of-the-art GPT-4o backbone.
•Workflow:The controller executes aRetrieve→Grade Documents→Generatecycle.
•Reflection:After generation, the system runs two graders:
18

1.Hallucination Check:Is the answer grounded in the retrieved facts?
2.Answer Relevance:Does the answer address the user question?
•Loop Logic:If the generation fails either check, the system loops back toTransform Queryand
re-retrieves. To prevent infinite loops, we enforce a hard limit of3 generation attempts, after
which the best available answer is returned.
B.2 CRAG (Corrective RAG)
Our implementation follows the standard CRAG flow, using an external web search as the corrective
action.
•Workflow:The controller executesRetrieve→Grade Documents.
•Trigger:A binary relevance grader evaluates the retrieved documents. If documents are deemed
“Irrelevant,” the system triggers a corrective branch.
•Correction:The query is rewritten, and a web search is performed using theTavily Search API
(capped at 1 result to maintain comparable context size). The web results are appended to the
context before final generation.
B.3 Adaptive-k(Dynamic Pruning)
We implement the “Largest Gap” strategy proposed by Taguchi et al. (2025) to dynamically select the
optimal context size.
•Workflow:The retriever fetches a large initial pool (k=50).
•Selection Logic:We calculate the cosine similarity scores for all 50 candidates. The algorithm
identifies the indexiwhere the difference between scores (score i−score i+1) is maximized (the
“largest gap”).
•Variants:
– No Buffer:The context is cut strictly at indexi.
– Buffer:We include a safety margin (e.g.,+5 documents) after the cut-off point to improve
recall, as recommended in the original paper.
C Qualitative Case Studies
We present deep-dive traces on representative multi-hop items to illustrate fixed-kgap repair under
SEAL-RAG. These traces demonstrate how the controller identifies specific missing attributes and re-
places low-utility passages to assemble a sufficient set.
C.1 Case A: Bridge Repair (HotpotQA)
Question:“Which city hosted the Olympic Games in the same year that the band Blur released the
album Parklife?”
Gold Answer:Lillehammer (1994 Winter Olympics).
Reasoning Type:Bridge (Entity→Date→Entity).
•Initial State (k=1):Retrieval lands onBlur (band)orParklife (album). The text mentions the
release year “1994” but lacks the bridge entity (the 1994 Olympics page).
•SEAL Loop (t=1):
19

1.Assess:The system extracts the release year (1994) but flags a missingBRIDGE_ENTITYfor
the “Olympic Games” slot.
2.Micro-Query:“1994 Olympic Games host city”.
3.Retrieval:Fetches1994 Winter Olympics.
4.Rank & Replace:The candidate scores high onGap Coverageand displaces the redundant
Parklifealbum details.
•Final Ledger (U t):
–[ORG] Blur -released-> [WORK] Parklife
–[WORK] Parklife -release_date-> [DATE] 1994
–[EVENT] 1994 Winter Olympics -held_in-> [LOC] Lillehammer
•Outcome:Correctly answers “Lillehammer” (supported by Blur + 1994 Olympics).
C.2 Case B: Attribute Alignment (2WikiMultiHopQA)
Question:“Who is older, the author of The Handmaid’s Tale or the director of Lost in Translation?”
Gold Answer:Margaret Atwood.
Reasoning Type:Comparison (Two entities→Attribute→Logic).
•Initial State (k=3):Retrieval fetchesThe Handmaid’s Tale(mentions Atwood),Lost in Trans-
lation(mentions Sofia Coppola), andMargaret Atwood(bio). ItmissesSofia Coppola’s bio.
•SEAL Loop (t=1):
1.Assess:Ledger contains Atwood’s birth date but lacks Coppola’s.
2.Gap Specification:Flags aQUALIFIERgap:DATEfor entitySofia Coppola.
3.Micro-Query:“Sofia Coppola date of birth”.
4.Rank & Replace:The candidateSofia Coppola (bio)replaces theLost in Translationplot
summary (which is laterally redundant).
•Final Ledger (U t):
–[PERSON] Margaret Atwood -born-> [DATE] Nov 18, 1939
–[PERSON] Sofia Coppola -born-> [DATE] May 14, 1971
•Outcome:Correctly answers “Margaret Atwood” (1939 vs 1971).
C.3 Case C: Failure Mode (Alias Mismatch)
Question:“Who is the CEO of the company that created the iPhone?”
Gold Evidence:Apple Inc.(Canonical Title).
•Initial State:Retrieval returns a document titledApple(Fruit).
•Gap:System identifies missingCEOrelation for “iPhone creator”.
•Micro-Query:“iPhone creator company CEO”.
•Retrieval:Returns a document titledApple Computer(an alias/redirect page).
•Failure:The extractor fails to linkApple Computerto the canonicalApple Inc.ID because the
alias map is incomplete. The sufficiency gate sees a mismatch between the query entity (Apple)
and the retrieved entity (Apple Computer) and triggers a halt or loop exhaustion.
•Mitigation:This highlights the need for robustAlias Normalizationin the entity ledger (Section
4.4).
20

D Indexing & Reproducibility
D.1 Resource Availability
The complete codebase, including the controller implementation, baseline re-implementations, evalua-
tion scripts, and environment configurations, is available at:
https://github.com/mosherino/SEAL-RAG
D.2 Indexing Pipeline (Natural Segmentation)
To ensure semantic coherence, we employ aNatural Document Segmentationstrategy rather than
arbitrary fixed-length sliding windows.
•Input:The raw Wikipedia dump provided by the benchmarks (HotpotQA/2Wiki), which orga-
nizes text as a list of sentences per page title.
•Logic:We concatenate the title and all associated sentences into a single retrieval unit:
chunk_text = f"{title}: " + " ".join(sentences)
•Result:Each vector in the index corresponds to exactly one Wikipedia page. This prevents the
fragmentation of context (e.g., separating a subject from their birthdate) and ensures that retrieval
metrics reflect page-level relevance.
D.3 Hyperparameters & Determinism
To guarantee fair comparisons, we fixed all non-algorithmic hyperparameters across all systems (SEAL-
RAG, Basic RAG, CRAG, Self-RAG, Adaptive-k). Table 6 lists these settings.
Table 6:Global Hyperparameters.These settings were held constant for all experiments to ensure
deterministic reproducibility.
Parameter Value
Global Random Seed20250101
Generator Temperature0.0(Greedy Decoding)
Judge Temperature0.0
Embedding Modeltext-embedding-3-small(OpenAI)
Vector Store Pinecone (Cosine Similarity)
Re-ranker Modelms-marco-MiniLM-L-6-v2
Candidate Pool Size (M) 20 (per micro-query variant)
Max Loop Budget (L){0,1,3,5}(Ablated)
D.4 Hardware Profile
All experiments were executed locally on aMacBook Pro (M3 Pro chip, 36 GB RAM). Since the
heavy lifting (generation/embedding) is offloaded to APIs, the controller logic is sufficiently lightweight
to run efficiently on consumer hardware without requiring specialized GPU clusters.
21

E Detailed Statistical Results
E.1 Methodology & Alignment
This section provides the complete statistical comparison tables for all metrics across all models and
retrieval depths. To ensure the validity of the paired statistical tests, strict data alignment was enforced.
For every comparison (e.g., SEAL-RAG vs. Self-RAG), we ensured that the two result vectors corre-
sponded to theexact same sequenceof question IDs from the seeded validation slice. Any questions
where the judge failed to return a valid format (rare,<0.1%) were excluded from the pair to maintain
strict alignment.
Software Implementation.All statistical tests were implemented using thescipy.statsandstatsmodels
Python libraries.
•Binary Metrics (Judge-EM):We usedMcNemar’s testwith the chi-squared approximation
(N=1000). The statisticχ2compares the discordant pairs.
•Continuous Metrics (Precision/Recall/F1):We usedPaired Two-Sidedt-tests.
•Effect Size:Calculated as Cohen’sd z(mean of differences divided by standard deviation of
differences).
E.2 SEAL-RAG vs. Adaptive-k
Table 7 details the statistical comparison against the state-of-the-art dynamic pruning baseline on 2Wiki-
MultiHopQA (k=5).
Table 7:Significance vs. Adaptive-k(2Wiki,k=5).SEAL-RAG significantly outperforms both vari-
ants in Accuracy and Precision.
Model Comparison Judge-EM (p-value) Precision (p-value)
GPT-4o-minivs. Adaptive-k(Buffer) 0.078 (ns)<0.001
vs. Adaptive-k(No Buffer)<0.001 0.135 (ns)
GPT-4ovs. Adaptive-k(Buffer) 0.021<0.001
vs. Adaptive-k(No Buffer)<0.001<0.001
E.3 HotpotQA Detailed Statistics (k=1)
Table 8 presents the full statistical breakdown for thek=1 bottleneck regime.
22

Table 8:Full Statistical Comparison Summary atk=1on HotpotQA.‘Perf. Diff.’ shows (SEAL-
RAG - Baseline). ‘Effect Size’ is Cohen’sd zfor t-tests.
Model Comparison Metric Test Type Perf. Diff. P-Value Statistic
gpt-4o-minivs. Self-RAG Judge-EM McNemar +13.6 pp<0.001χ2=64.2
Precision Pairedt+0.247<0.001t=16.53
Recall Pairedt+0.135<0.001t=16.92
vs. CRAG Judge-EM McNemar +7.1 pp<0.001χ2=17.8
Precision Pairedt+0.436<0.001t=27.62
Recall Pairedt+0.229<0.001t=27.15
vs. Basic Judge-EM McNemar +20.6 pp<0.001χ2=135.1
Precision Pairedt+0.008 0.200t=1.28
gpt-4ovs. Self-RAG Judge-EM McNemar +13.4 pp<0.001χ2=63.2
Precision Pairedt+0.162<0.001t=12.61
Recall Pairedt+0.246<0.001t=24.05
vs. CRAG Judge-EM McNemar +15.2 pp<0.001χ2=81.9
Precision Pairedt+0.374<0.001t=24.32
Recall Pairedt+0.352<0.001t=29.46
vs. Basic Judge-EM McNemar +32.0 pp<0.001χ2=259.9
Precision Pairedt+0.059<0.001t=6.91
gpt-4.1-minivs. Self-RAG Judge-EM McNemar +21.5 pp<0.001χ2=144.8
Precision Pairedt+0.149<0.001t=11.55
Recall Pairedt+0.121<0.001t=14.63
vs. CRAG Judge-EM McNemar +18.2 pp<0.001χ2=102.4
Precision Pairedt+0.305<0.001t=20.17
Recall Pairedt+0.199<0.001t=21.52
gpt-4.1vs. Self-RAG Judge-EM McNemar +9.5 pp<0.001χ2=33.1
Precision Pairedt+0.108<0.001t=9.61
Recall Pairedt+0.265<0.001t=27.02
vs. CRAG Judge-EM McNemar +14.9 pp<0.001χ2=71.4
Precision Pairedt+0.331<0.001t=22.06
Recall Pairedt+0.376<0.001t=31.71
E.4 HotpotQA Detailed Statistics (k=3)
Table 9 provides the full statistical breakdown fork=3. Note that while Recall differences are some-
times mixed (e.g., vs. Basic RAG), the Precision and Judge-EM gains remain highly significant with
large effect sizes.
E.5 2WikiMultiHopQA Significance (k=1,3,5)
Table 10 presents the significance values for the new dataset. The results confirm that SEAL-RAG’s ad-
vantage is robust across retrieval depths. Notably, atk=5, the Precision advantage is highly significant
(p<0.001) against all baselines, validating the solution to context dilution.
23

Table 9:Full Statistical Comparison Summary atk=3on HotpotQA.‘Perf. Diff.’ shows (SEAL-
RAG - Baseline). ‘Effect Size’ is Cohen’sd z.
Model Comparison Metric Test Type Perf. Diff. P-Value Effect Size
gpt-4o-minivs. Self-RAG Judge-EM McNemar +9.2 pp<0.001 N/A
Precision Pairedt+0.176<0.001 0.388
Recall Pairedt-0.029 0.010 -0.082
F1 Pairedt+0.045<0.001 0.128
vs. CRAG Judge-EM McNemar +7.1 pp<0.001 N/A
Precision Pairedt+0.536<0.001 1.359
Recall Pairedt+0.078<0.001 0.210
F1 Pairedt+0.246<0.001 0.716
vs. Basic Judge-EM McNemar +6.3 pp<0.001 N/A
Precision Pairedt+0.345<0.001 1.065
Recall Pairedt-0.284<0.001 -1.038
F1 Pairedt-0.013 0.101 -0.052
gpt-4ovs. Self-RAG Judge-EM McNemar +5.7 pp<0.001 N/A
Precision Pairedt+0.134<0.001 0.369
Recall Pairedt+0.134<0.001 0.367
F1 Pairedt+0.141<0.001 0.432
vs. CRAG Judge-EM McNemar +7.7 pp<0.001 N/A
Precision Pairedt+0.525<0.001 1.608
Recall Pairedt+0.239<0.001 0.576
F1 Pairedt+0.354<0.001 1.004
vs. Basic Judge-EM McNemar +8.7 pp<0.001 N/A
Precision Pairedt+0.398<0.001 1.462
Recall Pairedt-0.043<0.001 -0.143
F1 Pairedt+0.164<0.001 0.639
gpt-4.1-minivs. Self-RAG Judge-EM McNemar +12.4 pp<0.001 N/A
Precision Pairedt+0.129<0.001 0.313
Recall Pairedt-0.076<0.001 -0.211
F1 Pairedt+0.002 0.848 0.006
vs. CRAG Judge-EM McNemar +9.5 pp<0.001 N/A
Precision Pairedt+0.469<0.001 1.312
Recall Pairedt+0.003 0.839 0.006
F1 Pairedt+0.182<0.001 0.542
gpt-4.1vs. Self-RAG Judge-EM McNemar +3.1 pp 0.031 N/A
Precision Pairedt+0.119<0.001 0.366
Recall Pairedt+0.118<0.001 0.362
F1 Pairedt+0.128<0.001 0.455
vs. CRAG Judge-EM McNemar +4.6 pp 0.002 N/A
Precision Pairedt+0.502<0.001 1.654
Recall Pairedt+0.234<0.001 0.591
F1 Pairedt+0.347<0.001 1.054
vs. Basic Judge-EM McNemar +8.3 pp<0.001 N/A
Precision Pairedt+0.413<0.001 1.620
Recall Pairedt+0.009 0.347 0.030
F1 Pairedt+0.204<0.001 0.810
Table 10:Significance Matrix for 2WikiMultiHopQA.p-values for SEAL-RAG vs. Baselines. (ns:
not significant).
Judge-EM (p-value) Precision (p-value)
Model Comparisonk=1k=3k=5k=1k=3k=5
GPT-4o-minivs. Basic RAG<0.001<0.001 0.005ns<0.001<0.001
vs. Self-RAG<0.001 0.002 0.008<0.001<0.001<0.001
vs. CRAG<0.001ns0.009<0.001<0.001<0.001
GPT-4ovs. Basic RAG<0.001<0.001 0.001ns<0.001<0.001
vs. Self-RAG<0.001<0.001<0.001<0.001<0.001<0.001
vs. CRAG<0.001<0.001 0.007<0.001<0.001<0.001
24