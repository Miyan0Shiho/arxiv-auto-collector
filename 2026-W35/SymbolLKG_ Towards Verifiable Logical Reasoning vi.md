# SymbolLKG: Towards Verifiable Logical Reasoning via Logical Knowledge Graph and Symbolic Solvers

**Authors**: Haizhao Fan, Yuchi Xiong, Jize Wang, Xinping Guan, Xinyi Le

**Published**: 2026-08-27 09:06:52

**PDF URL**: [https://arxiv.org/pdf/2608.26836v1](https://arxiv.org/pdf/2608.26836v1)

## Abstract
Large Language Models (LLMs) have demonstrated remarkable proficiency in natural language understanding, yet they struggle with strict multi-step reasoning, frequently suffering from hallucinations and inconsistency. Existing solutions like Chain-of-Thought (CoT) lack rigorous verification mechanisms, while standard Retrieval-Augmented Generation (RAG) often misses the complex, structural dependencies inherent in logical tasks. To bridge this gap, we propose a Neuro-Symbolic architecture that integrates a Logical Knowledge Graph (LKG) with dynamic solver routing. Specifically, we introduce an ontology-based LKG that treats logical rules and constraints as first-class topological nodes, enabling explicit modeling of dependencies extracted from text. We further design a Logic Router to dynamically dispatch tasks to the optimal symbolic engine, which is supported by a topology-aware hybrid retrieval mechanism. Experimental results on logical reasoning benchmarks demonstrate that our framework significantly outperforms state-of-the-art prompting and RAG baselines, delivering higher accuracy and verifiable reasoning paths.

## Full Text


<!-- PDF content starts -->

SymbolLKG: Towards Verifiable Logical Reasoning
via Logical Knowledge Graph and Symbolic Solvers
Haizhao Fan
Shanghai JiaoTong UniversityYuchi Xiong
Shanghai JiaoTong UniversityJize Wang
Shanghai JiaoTong University
Xinping Guan
Shanghai JiaoTong UniversityXinyi Le
Shanghai JiaoTong University
Abstract
Large Language Models (LLMs) have demonstrated remarkable proficiency in
natural language understanding, yet they struggle with strict multi-step reasoning,
frequently suffering from hallucinations and inconsistency. Existing solutions like
Chain-of-Thought (CoT) lack rigorous verification mechanisms, while standard
Retrieval-Augmented Generation (RAG) often misses the complex, structural
dependencies inherent in logical tasks. To bridge this gap, we propose a Neuro-
Symbolic architecture that integrates a Logical Knowledge Graph (LKG) with
dynamic solver routing. Specifically, we introduce an ontology-based LKG that
treats logical rules and constraints as first-class topological nodes, enabling explicit
modeling of dependencies extracted from text. We further design a Logic Router
to dynamically dispatch tasks to the optimal symbolic engine, which is supported
by a topology-aware hybrid retrieval mechanism. Experimental results on logical
reasoning benchmarks demonstrate that our framework significantly outperforms
state-of-the-art prompting and RAG baselines, delivering higher accuracy and
verifiable reasoning paths.
1 Introduction
Logical reasoning is the cornerstone of reliable artificial intelligence, particularly in high-stakes
domains where precision is mandatory—healthcare [ 1], law [ 2], finance [ 3], and scientific discovery
[4]. Driven by their natural language understanding capabilities, Large Language Models (LLMs) are
increasingly deployed for complex tasks in these fields. However, the probabilistic nature of their
“next-token prediction” architecture fundamentally limits the rigorous, multi-step logic required for
strict reasoning scenarios.
Reasoning encompasses deductive, inductive, and abductive forms [ 5]; yet achieving trustworthy
logical reasoning, characterized by reliability, interpretability, and adherence to formal rules, remains
a fundamental challenge [ 5]. LLMs tend to generate plausible but incorrect steps and fail to maintain
consistency across long contexts [6].
To address these issues, prompting methods such as Chain-of-Thought (CoT) [ 7] and Tree-of-
Thoughts (ToT) [ 8] decompose complex problems, and Retrieval-Augmented Generation (RAG)
[9] grounds responses in external data. However, these methods lack rigorous verification: CoT
can propagate errors without symbolic grounding [ 10], and standard RAG misses the directional
dependencies inherent in logical rules.
In contrast, Knowledge Graphs (KGs) provide structured models of entities and relationships [ 11,12],
supporting rigorous formal reasoning and interpretability [ 5]; but symbolic systems traditionally
struggle with scalability on open-ended unstructured data. Integrating KGs with LLMs has thus
emerged as a promising direction [ 13]: Think-on-Graph [ 14] and RoG [ 15] guide LLMs along graph
arXiv:2608.26836v1  [cs.AI]  27 Aug 2026

paths; GraphRAG [ 16] structures retrieval from local contexts to global summarization; LightRAG
[17] introduces a dual-level retrieval mechanism capturing both specific entities and conceptual
themes; and KAG [ 18] pushes neuro-symbolic alignment in professional domains by coupling
LLMs with a logical-form-guided engine. However, these approaches typically treat logical rules as
unstructured text or implicit edge attributes rather than explicit, computable topological nodes. This
representational gap limits the deterministic mapping required for rigorous symbolic verification,
often forcing reliance on probabilistic text-to-code generation. Without distinct constraint nodes,
analyzing the logical topology to dynamically route problems to the appropriate solver (e.g., Z3
vs. Prover9) also becomes significantly more challenging, restricting adaptability across diverse
reasoning tasks.
Natural Language Input
 4. Formal Solver Output [GreenHouse Club] [Top Innovator Prize]LKG Representation
Logical Knowledge Graph
Direct LLM PipelineLLM Processing
Output Box[Jamie] [Constraint]
SymbolLKG PipelineSymbolic Representation
Output Box
Context: In the 2024 Science Fair, exactly one student from the “GreenHouse Club” won 
the “Top Innovator” prize. The fair had 500 attendees. The RedRobotics Club also … 
Jamie is a registered student in the “GreenHouse Club” and presented a project.
Query: Did Jamie win the “Top Innovator” prize?
Assuming: 
1.Member(Jamie, GreenHouseClub) 
2.IsPrice(TopInnovator)
Constraint:
∃!x,Member(x,GreenHouseClub)∧
Won(x,TopInnovator)
Query:    
Won(Jamie,TopInnovator)
Answer: 
Unknown/Insufficient 
Information</think> 
1. The text states the winner is from 
the GreenHouse Club. 
2. Jamie is identified as a member of 
this club. 
3. Since Jamie is the only specific 
member mentioned in relation to the 
winning group, the text likely implies he 
is the winner. 
(CWA Assumption Error)
4. It is highly probable that Jamie won. 
</think>
Answer: Yes, Jamie won the 
“Top Innovator” prize. 
(Hallucination)
Figure 1: Comparison of reasoning
paradigms.The input highlights constraints
in green, noise in grey, and premises in or-
ange. The Direct LLM Pipeline (left) ig-
nores this structure. It makes a “Closed
World Assumption” (CWA) error and incor-
rectly guesses Jamie won. The SymbolLKG
Pipeline (right) converts text into a Logical
Knowledge Graph. It uses symbolic verifica-
tion to strictly check the premises. The sys-
tem correctly concludes the answer is “Un-
known”.Driven by these insights, we propose a neuro-symbolic
framework that integrates Logical Knowledge Graphs
(LKG) with external symbolic solvers, treating logical
rules and constraints as first-class nodes. This combines
the natural language understanding of LLMs for parsing
problems into a structured graph with the deterministic
power of external solvers for the actual reasoning.
Our main contributions are summarized as follows:
•SymbolLKG Framework:We propose SymbolLKG,
a novel neuro-symbolic framework that grounds LLM
reasoning via Logical Knowledge Graphs (LKG) and
router-guided solvers. The pipeline parses natural lan-
guage into an LKG, retrieves a topologically connected
subgraph, and dynamically routes the problem to the
optimal symbolic solver, bridging neural semantic flex-
ibility and symbolic deterministic precision.
•Topology-Aware Logical Knowledge Graph:We
introduce a dynamic LKG architecture that treats
Rules and Constraints as independent nodes, using
a “Schema-on-Read” strategy to build Concept-Entity
hierarchies from context. This topology enables hybrid
retrieval combining vector search with graph traver-
sal, refined via LLM pruning into aLogical Hull—a
complete yet minimal subgraph of essential premises.
•Adaptive Logic Routing and Self-Refining Trans-
lation:We implement a Logic Router that analyzes
problem structure to select the optimal symbolic tool
(e.g., Z3 for math, Prover9 for logic), with a feedback
loop where the LLM uses solver error traces to refine its generated code, improving reliability on
complex logical tasks.
We evaluate our framework on standard logical reasoning and multi-hop QA datasets. Experimental
results demonstrate that our approach significantly outperforms standard LLM prompting and baseline
RAG methods, validating the efficacy of combining structured knowledge graphs with deterministic
solvers for trustworthy reasoning.
2 Related Works
Large Language Models for Logical Reasoning.Large Language Models show strong performance
in reasoning tasks. Techniques like CoT [ 7] and Zero-Shot CoT [ 19] are now standard for generating
multi-step reasoning. To improve reliability, methods such as Self-Consistency [ 20] and ToT [ 8]
explore multiple reasoning paths. Despite these advances, pure neural models still suffer from
hallucinations and “unfaithful” explanations, meaning their generated reasons often do not match their
actual decision process [ 21]. Furthermore, Transformers struggle with compositional generalization,
failing on multi-hop deduction problems outside their training data [ 22,23]. Self-Correction is also
difficult because models rarely catch their own logical errors without external feedback [ 24]. This
highlights the need for robust external verification mechanisms.
2

Neuro-Symbolic AI and Auto-formalization.Neuro-symbolic AI combines neural flexibility with
symbolic rigor. A key approach is auto-formalization, where LLMs translate text into formal code
[25,10] or logic rules [ 26]. Systems like Logic-LM [ 27] and LINC [ 28] use external solvers to
execute these rules. However, most existing systems use a static strategy, assigning one solver (e.g.,
Z3 or Prover9) to an entire dataset. This rigid method fails on complex queries mixing arithmetic and
relational logic. As shown by [ 29], tool effectiveness depends heavily on the specific structure of each
problem. Our SymbolLKG solves this with Dynamic Solver Routing. Instead of using dataset-level
rules, our system acts as a controller, analyzing each query at runtime to select the best solver.
Retrieval-Augmented Generation for Structured Knowledge.Retrieval-augmented generation
(RAG) reduces hallucinations by using external documents [ 9,30]. Standard RAG relies on dense
vector similarity to fetch relevant context. However, for logical reasoning tasks, this often leads to
“semantic drift”—where the system retrieves semantically similar but logically irrelevant documents.
While good for simple facts, standard retrieval fails at multi-hop reasoning, which requires following
logical chains rather than just matching similar words. Recent works use Knowledge Graphs to solve
this. Methods like Think-on-Graph [ 14], RoG [ 15], StructGPT [ 31] and GraphRAG [ 16] guide LLMs
through graph structures. Others, like Selection-Inference [ 32], break reasoning into smaller steps.
However, these methods usually treat KGs as static lists of entities and ignore complex logical rules.
We propose a Logical Knowledge Graph (LKG) that treats rules and constraints as primary nodes.
By combining vector search with Logical Hull Expansion, we ensure implicit axioms are retrieved
for the symbolic solver, bridging the gap between semantic search and logical deduction.
3 Methodology
We propose theSymbolic-CognitiveLogicalKnowledgeGraph (SymbolLKG) framework, a neuro-
symbolic architecture (Figure 2) that maps unstructured text into verifiable reasoning paths. The
workflow begins at the top with the Logical Knowledge Graph Construction phase, where an LLM
parses the natural language corpus to extract entities, concepts, rules, and constraints into a structured
LKG. The process flows into the Inference Pipeline, which first employs Hybrid Retrieval and Pruning
to isolate a context-aware subgraph containing only relevant premises. Next, an Adaptive Logic
Router analyzes the subgraph’s topology to pick the best solver, passing the problem to the Code
Generation module for symbolic execution and self-refining verification.
Phase1: Logical Knowledge Graph Construction
NL CorpusConcepts & Entities Extraction
Entity
 Logical 
Knowledge
Graph
(LKG)
Phase2: Inference Pipeline
User
QueryQuery Entity 
Extraction
Logical Subgraph
Solver Selection
Rules & Constraints ExtractionC1:F-L Scholarship C2: STEM major
E1: Alice E2:portfoilo
Rule1: 
applicants must 
be enrolled in a 
STEM major.Constraint1: 
applicants 
must hold a 
GPA ≥ 3.8. 
Relation ExtractionF-L ScholarshipA-N AwardSTEM major GPA Arts
BobAlice
paper portfolioRule1 Rule2Constraint1
Concept Constraints Rules[:APPLIES_TO]
[:IS_A] / [:HAS]
Is Alice eligible for the Future 
Leaders Scholarship?
Alice
Future Leaders
Scholarship
Entity
ExtractionHybrid Retrieval  & Pruning
F-L Scholarship STEM major
Constraint1GPA
Rule1Alice
paperPremises:
1. ∀x, g ((Major(x, STEM) ∧ HasGPA(x, g)∧(g>
3.8) → Eligible(x, FLScholarship)
2. Major(Alice,STEM)
3. HasGPA(Alice,3.9)
Query:
1. Eligible(Alice,FLScholarship)Yes, Alice is 
Eligible.Logical 
Entity 
ExtractionC3: GPA
E1C1
Code Generation
Syntax Error
Informalization
Vector Search
&
Graph Traversal
Router
Solver SetTopology
Error Message
Selected Router Execution
RefinementR1C1C2[:MENTIONS][Constraints]
To qualify for the “Future 
Leaders Scholarship”, 
applicants must be enrolled 
in a STEM major and hold 
a GPA ≥ 3.8. 
[Data]
Alice is a STEM major 
with a ​3.9 GPA​. She 
recently ​published a paper​.
[Distractor]
Bob studies Arts (GPA 3.5). 
The "Arts Nova Award" 
requires a portfolio.
Pruning
Z3 (SMT)
Prover9 (FOL)
Pyke (Inf)
Translation
Figure 2: The architecture of the SymbolLKG framework.The pipeline consists of two phases:Phase 1
(LKG Construction)transforms text into an LKG, establishing a dynamic Concept-Entity hierarchy alongside
reified Rule and Constraint nodes.Phase 2 (Inference)employs a hybrid retrieval strategy (combining vector
search with graph traversal) to extract a context-aware subgraph. Then a topology-aware router dispatch it to the
optimal symbolic solver for verifiable execution.
3

3.1 Logical Knowledge Graph Construction
Phase 1 begins by processing the natural language corpus through an LLM-based extraction pipeline.
As illustrated in the “LKG Construction” panel of Figure 2, we adopt an Open Information Extraction
(OpenIE) paradigm to handle the diversity of natural language. We formally define the LKG as a
directed, heterogeneous multigraph G= (V, E,A,T) , where Vis the set of nodes, Eis the set of
edges, Ais a set of attributes, and Trepresents the node types. The vertex set Vis partitioned into
four disjoint subsets, as shown in Table1.
First, we extract Concepts and Entities using a dynamic “Schema-on-Read” strategy. Unlike tradi-
tional graphs with rigid categories, our system generates Concept nodes (e.g., “Student”) and Entity
nodes (e.g., “Alice”) based on their specific roles in the text. This approach ensures that the graph
captures the precise domain of discourse needed for logical reasoning.
Next, we identify logical dependencies and instantiate them as Rule and Constraint nodes. We
treat logic as “first-class citizens” rather than simple text attributes. For example, a restriction like
GPA>3.8 is extracted as a specific Constraint node, while an implication like “If A then B” becomes
a Rule node. These nodes are then connected to their respective entities via relation edges.
Finally, these components are assembled into the Logical Knowledge Graph (LKG). To ensure
consistency, we assign a unique ID to each node using a hash function. This step merges variations
like “The blue book” and “Blue Book” into a single node, preventing duplicates.
This design offers three key advantages. First, it makes logic semantically addressable. Rules get
their own vector embeddings hv∈Rd, so the system can find “seating restrictions” directly via
semantic search. Second, it separates the T-Box (logical rules) from the A-Box (assertions) [ 33]. This
keeps the reasoning structure stable even when specific entity data changes. Third, the Concept nodes
in our framework are extracted directly from the corpus based on their logical role in the specific
problem context, rather than mapping to a universal ontology. They define theDomains of Discourse
for logical quantifiers (e.g., ∀x∈Students ), enabling the direct translation of natural language into
verifiable First-Order Logic (FOL). This “Schema-on-Read” approach ensures that symbolic solvers
operate on precise, problem-relevant subsets rather than generic classes.
3.2 Hybrid Retrieval and Pruning
After the construction of the LKG, the inference pipeline begins by setting a precise entry point for
the user query. As shown in the “Hybrid Retrieval” module (Figure 2), we first perform Query Entity
Extraction to ground the search. The system parses the natural language query to find specific target
entities (e.g., “Alice” or “Future Leaders Scholarship”) before any retrieval occurs.
These extracted entities, combined with the original query, drive the identification of Anchor Nodes.
We employ a dense embedding model (BGE-M3 [ 34]) to encode the user query and calculate the
cosine similarity between query embedding vqand all node embeddings. Simultaneously, we perform
exact matching against the names of the entities extracted in the previous step. This dual approach
selects a set of anchorsS anchor defined as:
Sanchor ={v∈V|cos(v q,hv)> τ sim}
∪ {v∈V E|exactmatch(v.name, q)}(1)
This strategy captures both the explicit targets mentioned by the user and the abstract logical concepts
relevant to the context.
However, semantic search alone is not enough for logic. A rule constraining an entity often lacks
shared keywords with that entity. To fix this, we define theLogical Hullof the anchor set. We expand
the context by traversing the graph’s [:MENTIONS], [:IS_A] and [:APPLIES_TO] edges. For each
anchor v∈S anchor , we include its k-hop (determined by datasets) neighbors from the rule set VRor
constraint setV S.
Stmp={u∈V R∪VS| ∃v∈S anchor,dist(u, v)≤k}
Shull=S anchor∪S tmp(2)
This step ensures Logical Completeness. If an entity is found, its related rules are automatically
added. E.g., finding “Alice” also fetches the “AllDifferent” constraint for her group.
Finally, the expanded set Shullcan be large. To fit the solver’s context window, we use an LLM-
based pruning module Ψ. This module checks if each node is needed to answer qand removes
4

Node Type Symbol Definition & Categories (Codebase) Key Attributes (Class Fields) Role in Reasoning
EntityV E Concrete instances implemented as
Entityclass.
IDs are generated via SHA256 hash of
lemmatized name and type.name,entity_type,
id(Canonical Hash),
propertiesAnchors for retrieval. Hash-based
IDs ensure uniqueness across the
graph.
ConceptV C Abstract categories implemented as
Conceptclass.name,labelsSimple taxonomy nodes supporting
type grouping.
RuleV R Logical implications implemented as Rule
class.expression,description,
embeddingStores logic formulas;
descriptionallows vector-based
retrieval of rules.
ConstraintV S Restrictions on states, implemented as 4
subclasses:
1.Arithmetic(Math ops)
2.AllDifferent(Combinatorial)
3.Ordering(Spatial/Temporal relations
likeleft_of,newer_than)
4.Generic(Constraints outside the
above-listed categories)raw_expression,
constraint_type,
Specific:relation,
operation,valueDefines solution boundaries.
Ordering handles both topological
and chronological constraints via
relation types.
Table 1:Node Taxonomy in the Symbolic-Cognitive Logical Knowledge Graph (SymbolLKG). The schema
distinguishes between static entities, hierarchical concepts, deductive rules, and arithmetic constraints to facilitate
dynamic solver routing.
“distractor” constraints. The result is Gfinal= Ψ(S hull, q), a compact, mathematically bounded context
for reasoning.
3.3 Solver Selection via Logic Router
Given the pruned subgraph Gfinal, the framework next determines the best strategy to solve the logical
problem. As depicted in the “Solver Selection” block of Figure 2, we introduce an Adaptive Logic
Router to analyze the topology of the subgraph.
Real-world logical problems vary widely, so using a single solver is often inefficient. Research has
shown that while Z3 [ 35] excels at arithmetic and combinatorial tasks, it is often less intuitive for
pure first-order logic compared to Prover9 [ 29]. To optimize performance, we implement an Adaptive
Logic Router. This meta-cognitive module froute: (G final, q)→Ω analyzes the subgraph’s structure
to select the best solver backend.
SMT Solver (Z3) Pathway:Used when Gfinalhas many VSnodes (Constraints), especially for
arithmetic or global uniqueness (e.g., CSPs). We leverage Z3’s DPLL(T) algorithm to efficiently
prune large search spaces.
Automated Theorem Prover (Prover9) Pathway:Selected for strict FOL tasks or proof generation.
This runs when the subgraph is mostlyV Rnodes (Rules) with logical implications but no numbers.
Pyke Pathway:Used for direct relational queries (e.g., “Who is X’s father?”). Here, lightweight
engines like Pyke are faster and more efficient than heavy solvers.
This adaptive approach ensures robustness. By categorizing the problem, we stop the LLM from
trying unreliable arithmetic through token prediction—a known Transformer weakness. Instead, we
force the system to delegate computation to symbolic engines designed for mathematical correctness.
3.4 Code Generation
Once the optimal solver is determined, the final phase translates the logical subgraph into executable
symbolic code. The LLM acts as a compiler, mapping the nodes in Gfinaldirectly to solver-specific
syntax. For instance, in the Z3 pathway, entities e∈V Ebecome variables ( x=Int(x) ); concept
nodesV Cset domains (solver.add(0≤x≤100)); and constraint nodesv c∈VSbecome assertions
(solver.add(Distinct(x, y, z))).
To ensure reliability, we use a self-refining execution loop. As shown in the bottom-right of Figure
2, the system captures any error messages during execution. It then feeds this feedback back to the
model to refine the code. This cycle repeats until success or timeout.
5

This proposed methodology effectively transforms the stochastic nature of probabilistic text generation
into the precision of deterministic computation. Consequently, the resulting output constitutes a
rigorous formal proof or a logically valid assignment, delivering a level of interpretability and
systemic trust that remains fundamentally unattainable through standard neural Chain-of-Thought
prompting alone.
4 Experiments
4.1 Experimental Settings
DatasetsTo comprehensively evaluate the SymbolLKG framework, we utilize eight diverse bench-
marks that span the spectrum from formal symbolic logic to open-domain multi-hop reasoning. We
categorize them into two groups based on the evaluation focus:
Logical Reasoning BenchmarksThese datasets validate translation precision and deductive
validity.FOLIO[ 36] provides premises with FOL ground truths and complex logical structures.AR-
LSAT[ 37] focuses on analytical reasoning and constraint satisfaction from law exams.ProofWriter
[38] assesses rule-based reasoning over multi-hop chains and proof generation.LogicalDeduction
[39] uses constraint satisfaction problems for abstract reasoning, whileProntoQA[ 40] evaluates
transitive reasoning via synthetic multi-hop deductive chains.
Multi-hop Retrieval & QA BenchmarksThese benchmarks assess the robustness of LKG Con-
struction and Hybrid Retrieval.2WikiMultiHopQA[ 41] requires information aggregation across
multiple articles with reasoning chains up to 4 hops.HotpotQA[ 42] tests noise filtration and bridge-
entity identification by requiring the system to find supporting facts among irrelevant distractors.
Musique[ 43] challenges models with highly compositional questions designed to systematically
minimize disconnected reasoning shortcuts.
Implementation DetailsOur SymbolLKG framework utilizes Llama-3.3-70B-Instruct as the
backbone LLM for the extraction and translation modules due to its favorable balance between
economic efficiency and inference speed. To ensure a fair comparison, we consistently applied
Llama-3.3-70B-Instruct as the backbone model for our primary baseline methods, including IRCoT +
HippoRAG. To minimize generation randomness and ensure reproducibility, we set the temperature
to 0 for all LLM inference calls. For the symbolic solvers, we employ a hybrid strategy: Z3 is used for
constraint satisfaction and arithmetic tasks, Prover9 handles strict first-order logic theorem proving,
and Pyke is utilized as a lightweight inference engine for direct relational queries. The hybrid retrieval
combines dense embeddings (using BGE-M3 [34]) with our topology-aware graph traversal.
4.2 Logical Reasoning Performance
BaselinesWe compare our framework against two representative settings:Standard LLM with
CoT:Uses LLM with Chain-of-Thought prompting to establish a strong proprietary model baseline.
Logic-LM:A neuro-symbolic framework that integrates LLMs with symbolic solvers, serving as a
direct competitor in the symbolic reasoning domain.
DATASET COT LOGIC-LM OURS
FOLIO 70.5878.9271.39
AR-LSAT 35.06 43.0457.85
LOGICALDEDUCTION 75.25 87.6390.81
PRONTOQA 98.79 83.20100.00
PROOFWRITER 68.1179.6673.60
AVG. 69.56 74.4978.73
Table 2:Logical Reasoning Accuracy (%) across five benchmarks. Methods compared include Standard CoT
prompting and the neuro-symbolic baseline Logic-LM.
ResultsTable 2 summarizes the accuracy across five benchmarks. SymbolLKG achieves an average
of78.73% , outperforming Logic-LM ( 74.49% ) and Standard CoT ( 69.56% ). Two design choices
6

drive its strongest cells: the typed constraint subclasses (Ordering / AllDifferent / Arithmetic) feed
a structural signal to the Logic Router, deterministically selecting Z3 on AR-LSAT and yielding
57.85% ; and the three-attempt self-refining loop, which echoes solver compiler errors back to the
LLM, completes ProntoQA’s transitive Horn chains at 100.00% . SymbolLKG lags Logic-LM on
FOLIO and ProofWriter because Logic-LM relies on dataset-specific prompts, whereas our generalist
ontology often demotes free-form quantified premises to Generic rules, weakening the router signal
and limiting the recovery scope of compiler-trace feedback.
4.3 Multi-hop Retrieval Performance
We evaluate the retrieval effectiveness of our system on two challenging multi-hop datasets: 2Wiki-
MultiHopQA, HotpotQA and Musique. This experiment tests the system’s ability to identify and
retrieve disjointed pieces of evidence required for multi-step reasoning.
BaselinesWe categorize our baselines into Single-step and Multi-step retrieval approaches to
provide a comprehensive comparison:Single-step Retrieval:We include sparse retrieval (BM25
[44]), dense retrieval (Contriever [ 45], GTR [ 46], NativeRAG [ 9]), and advanced structure-aware
methods (RAPTOR [ 47], Proposition [ 48], HippoRAG [ 49]).Multi-step Retrieval:We evaluate
IRCoT [ 50] (Interleaving Retrieval with Chain-of-Thought) combined with four different retrievers:
BM25, Contriever, NativeRAG, and HippoRAG. This setup allows us to compare our topology-aware
retrieval against iterative feedback loops powered by different backbone retrievers.
MetricsWe reportRecall@k, defined as the proportion of ground-truth supporting documents
found within the top- kretrieved candidates. Given the ground-truth supporting fact set Gqand the
top-kretrieved setR q,kfor a queryq, the recall is calculated as:
Recall@k=1
|Q|X
q∈Q|Rq,k∩Gq|
|Gq|(3)
We use Recall@2 and Recall@5, which measure the percentage of queries where the top-2 and top-5
retrieved documents contain the complete set of ground-truth supporting facts.
TYPEMETHOD2WIKIMULTIHOPQA HOTPOTQA
RECALL@2 RECALL@5 RECALL@2 RECALL@5
SINGLE-STEPBM25 51.8 61.9 55.4 72.2
CONTRIEVER 46.6 57.5 57.2 75.5
GTR 60.2 67.9 59.4 73.3
RAPTOR 46.3 53.8 58.1 71.2
PROPOSITION 56.4 63.1 58.7 71.1
NATIVERAG 59.2 68.2 64.7 79.3
HIPPORAG 70.7 89.1 60.5 77.7
MULTI-STEPIRCOT + BM25 55.4 66.8 62.1 75.3
IRCOT + CONTRIEVER 51.6 63.8 65.9 81.6
IRCOT + NATIVERAG 64.1 74.4 67.9 82.0
IRCOT + HIPPORAG 75.3 93.4 67.0 83.0
OURSSYMBOLLKG 79.488.2 68.5 84.1
Table 3:Retrieval Performance (Recall@2 and Recall@5) on 2WikiMultiHopQA and HotpotQA. Baselines
are categorized into Single-step and Multi-step approaches. SymbolLKG leverages topology-aware traversal to
optimize evidence retrieval.
ResultsTable 3 presents the retrieval results. SymbolLKG attains 79.4 /88.2 on 2WikiMultiHopQA
and68.5 /84.1 on HotpotQA. The Recall@2 lead is driven by hybrid anchoring: three independently
thresholded vector queries (over Rule, Constraint and Entity indices) are unioned with a BM25
entity-name pass, so the second slot is reliably occupied either by an entity-string hit or by a rule
whose raw_text contains the bridge. The HotpotQA gain further reflects the Logical Hull: two-hop
traversal along [:MENTIONS], [:APPLIES_TO] and typed predicate edges surfaces distractor-buried
supporting facts that share no surface tokens with the query. The single trail, R@5 on 2Wiki, is
consistent with our single-pass anchor-and-prune strategy, which maximises top-rank precision but
cannot re-introduce evidence rejected by the initial threshold, unlike IRCoT’s iterative re-querying.
7

4.4 Question Answering Performance
Finally, we assess the end-to-end question answering capabilities on 2WikiMultiHopQA and Hot-
potQA. This evaluation determines whether the retrieved context effectively supports the generation
of accurate answers.
BaselinesWe benchmark against three primary RAG frameworks:NativeRAG: A standard
retrieval-augmented generation approach using dense retrieval.GraphRAG[ 16]: A typical graph-
enhanced RAG framework with structured knowledge modeling.HippoRAG 2[ 51]: A state-of-the-
art graph-based RAG method inspired by the hippocampal indexing theory.
Method2Wiki HotpotQA Musique Average
EM F1 EM F1 EM F1 EM F1
NativeRAG 33.4 43.3 43.4 57.7 15.5 26.4 30.8 42.5
GraphRAG 51.4 58.6 55.2 68.6 27.3 38.5 44.6 55.2
IRCoT + HippoRAG 47.7 62.7 45.7 59.2 21.9 33.3 38.4 51.7
HippoRAG 2 65.0 71.0 62.7 75.5 37.2 48.6 55.0 65.0
SymbolLKG 70.2 74.9 73.8 81.5 48.6 59.4 64.2 71.9
Table 4:End-to-end QA Performance (EM and F1). We compare our SymbolLKG against baselines on
2WikiMultiHopQA, HotpotQA and Musique.
MetricsWe utilizeExact Match (EM)andF1 Score. EM measures whether the predicted answer
ˆastrictly matches the ground truth a∗after normalization. F1 Score evaluates the token-level overlap.
LetT(·)denote the bag of tokens function; precision (P) and recall (R) are computed as:
P=|T(ˆa)∩T(a∗)|
|T(ˆa)|, R=|T(ˆa)∩T(a∗)|
|T(a∗)|(4)
The F1 score is the harmonic mean:
F1= 2·(P·R)/(P+R)(5)
ResultsTable 4 summarizes EM and F1 on the three multi-hop QA datasets. SymbolLKG attains
70.2 /74.9 on 2Wiki, 73.8 /81.5 on HotpotQA and 48.6 /59.4 on MuSiQue, leading every prior
baseline on every cell. The largest gain on MuSiQue reflects two complementary mechanisms:
hash-based entity canonicalization merges paraphrased mentions of the same bridge entity (e.g., “The
Beatles” and “the band Beatles”) into a single graph node, so a bridge cannot be silently fragmented
across single-hop sub-facts; meanwhile, the Logical Hull expansion admits rules and constraints
two hops away from a question anchor even when they share no surface tokens with the query.
The+6.0 F1 on HotpotQA reflects the same mechanism on noise rather than on composition: the
topologically bounded subgraph isolates anchor-connected facts among 8–10 distractor paragraphs
and is appended alongside the raw context, acting as a high-precision prior over which titles the
answer LLM should attend to. The narrower gap on 2Wiki indicates that on cleaner Wikipedia text,
dense retrieval baselines already cover most evidence, and the LKG’s marginal contribution is mainly
in entity disambiguation across same-surname or same-topic candidates.
4.5 Router Performance
SetupTo isolate the contribution of the Logic Router, we sample 100 instances from each of the
five logical benchmarks ( N= 500 in total) and shuffle them into a single mixed evaluation stream.
Each instance is processed end-to-end (LKG construction →hybrid retrieval →router decision)
using Llama-3.3-70B-Instruct as the routing LLM. The ground-truth solver per dataset follows
the design intent of the three engines: Z3 for AR-LSAT and LogicalDeduction (CSP / ordering
/ arithmetic); Prover9 for FOLIO (FOL with true/false/unknown) and ProofWriter, and Pyke for
ProntoQA (Horn-clause forward chaining).
ResultsTable 5 reports the per-dataset routing accuracy. SymbolLKG’s router reaches an overall
accuracy of 86.0% , more than 2.5× the random baseline. Routing is essentially perfect on the two
8

Z3-targeted datasets ( 100.0% on both AR-LSAT and LogicalDeduction), because the typed constraint
subclasses appear directly in the retrieved subgraph and trigger the router’s first decision rule. Routing
is also strong on the two Prover9-targeted datasets ( 85.0% on FOLIO, 92.0% on ProofWriter), where
the three-valued cue “true, false, or uncertain/unknown” in the question text reliably steers the router
to Prover9. The remaining error mass concentrates on ProntoQA ( 53.0% ), where the dominant
confusion is Pyke-vs-Prover9: ProntoQA’s “true or false” phrasing is closed-world but lexically
close enough to a Prover9 trigger to occasionally mislead the router. This residual confusion is
largely benign in practice, however, because both engines support Horn-clause forward chaining—an
instance routed from Pyke to Prover9 in this regime is still solved correctly more often than not.
Table 5:Router classification accuracy (%) on a mixed evaluation set of 500 instances (100 per dataset). The
ground-truth solver per dataset is determined by the design intent of each symbolic engine.
Dataset Ground-Truth Solver Random SymbolLKG
AR-LSAT Z3 33.3100.0
LogicalDeduction Z3 33.3100.0
FOLIO PROVER9 33.385.0
ProofWriter PROVER9 33.392.0
ProntoQA PYKE 33.353.0
Average – 33.386.0
4.6 Computational Cost Analysis
OnN= 100 samples per benchmark with Llama-3.3-70B-Instruct, the full SymbolLKG pipeline
averages 122.2 s per logical reasoning problem and 166–407s per multi-hop QA query, dominated
by LLM API calls ( >99% of wall-clock time). For deployments with fixed corpora the LKG is built
once offline and reused across queries, reducing inference-time latency to retrieval ( <0.1 s) plus
a single answer-generation call ( ∼10 s). Full per-stage breakdowns, distributions, and theoretical
complexity are provided in Appendix C.
5 Limitations & Conclusion
In this paper, we presented SymbolLKG, a neuro-symbolic framework that bridges the gap between
the semantic flexibility of LLMs and the rigorous precision of symbolic solvers. By treating logical
rules as topological nodes within a Logical Knowledge Graph and employing an adaptive router
to dispatch tasks to the most appropriate engine (Z3, Prover9, or Pyke), our system significantly
outperforms standard prompting and retrieval baselines on complex reasoning benchmarks. The
core advantage of SymbolLKG lies in itsverifiability: unlike the opaque generation process of
Chain-of-Thought, our framework provides deterministic proof traces, ensuring that the reasoning
process is both transparent and logically sound.
However, several limitations warrant further investigation. First is thetranslation bottleneck. Our
framework operates on the premise that the LLM can accurately parse natural language into formal
graph representations. While the self-refining loop mitigates syntax errors, semantic misinterpreta-
tions during the extraction phase can still propagate to the solver. Essentially, the symbolic engine
guarantees validity (the structure of the argument) but not soundness (the truth of the premises) if the
initial grounding is flawed. Second, thehandling of ambiguityremains a challenge. Formal logic
requires precise definitions, whereas natural language is inherently ambiguous. When the input text
contains vague quantifiers or metaphorical expressions, the rigid schema of the LKG may struggle to
capture the full nuance, potentially leading to oversimplification. Finally, the introduction of graph
construction and external solvers incurs additionalcomputational overhead. While more accurate,
our pipeline is computationally heavier than direct single-pass generation. Future work will focus on
optimizing the extraction latency and exploring end-to-end differentiable neuro-symbolic alignment
to reduce the dependency on intermediate discrete translation.
9

References
[1]Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura Gutierrez,
Ting Fang Tan, and Daniel Shu Wei Ting. Large language models in medicine.Nature medicine,
29(8):1930–1940, 2023.
[2]Daniel Martin Katz, Michael James Bommarito, Shang Gao, and Pablo Arredondo. Gpt-4
passes the bar exam.Philosophical Transactions of the Royal Society A, 382(2270):20230254,
2024.
[3]Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann,
Prabhanjan Kambadur, David Rosenberg, and Gideon Mann. Bloomberggpt: A large language
model for finance.arXiv preprint arXiv:2303.17564, 2023.
[4]Hanchen Wang, Tianfan Fu, Yuanqi Du, Wenhao Gao, Kexin Huang, Ziming Liu, Payal
Chandak, Shengchao Liu, Peter Van Katwyk, Andreea Deac, et al. Scientific discovery in the
age of artificial intelligence.Nature, 620(7972):47–60, 2023.
[5]Avinash Patil and Aryan Jadon. Advancing reasoning in large language models: Promising
methods and approaches.arXiv preprint arXiv:2502.03671, 2025.
[6]Karthik Valmeekam, Matthew Marquez, Sarath Sreedharan, and Subbarao Kambhampati. On
the planning abilities of large language models-a critical investigation.Advances in Neural
Information Processing Systems, 36:75993–76005, 2023.
[7]Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le,
Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models.
Advances in neural information processing systems, 35:24824–24837, 2022.
[8]Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem solving with large language models.Ad-
vances in neural information processing systems, 36:11809–11822, 2023.
[9]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks.Advances in neural information processing
systems, 33:9459–9474, 2020.
[10] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan,
and Graham Neubig. Pal: Program-aided language models. InInternational Conference on
Machine Learning, pages 10764–10799. PMLR, 2023.
[11] Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard De Melo, Claudio
Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier,
et al. Knowledge graphs.ACM Computing Surveys (Csur), 54(4):1–37, 2021.
[12] Shaoxiong Ji, Shirui Pan, Erik Cambria, Pekka Marttinen, and Philip S Yu. A survey on
knowledge graphs: Representation, acquisition, and applications.IEEE transactions on neural
networks and learning systems, 33(2):494–514, 2021.
[13] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large
language models and knowledge graphs: A roadmap.IEEE Transactions on Knowledge and
Data Engineering, 36(7):3580–3599, 2024.
[14] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M
Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of
large language model on knowledge graph.arXiv preprint arXiv:2307.07697, 2023.
[15] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful
and interpretable large language model reasoning.arXiv preprint arXiv:2310.01061, 2023.
[16] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global:
A graph rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
10

[17] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint arXiv:2410.05779, 2024.
[18] Lei Liang, Zhongpu Bo, Zhengke Gui, Zhongshu Zhu, Ling Zhong, Peilong Zhao, Mengshu
Sun, Zhiqiang Zhang, Jun Zhou, Wenguang Chen, et al. Kag: Boosting llms in professional
domains via knowledge augmented generation. InCompanion Proceedings of the ACM on Web
Conference 2025, pages 334–343, 2025.
[19] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large
language models are zero-shot reasoners.Advances in neural information processing systems,
35:22199–22213, 2022.
[20] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language
models.arXiv preprint arXiv:2203.11171, 2022.
[21] Miles Turpin, Julian Michael, Ethan Perez, and Samuel Bowman. Language models don’t
always say what they think: Unfaithful explanations in chain-of-thought prompting.Advances
in Neural Information Processing Systems, 36:74952–74965, 2023.
[22] Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin,
Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, et al. Faith and fate: Limits
of transformers on compositionality.Advances in Neural Information Processing Systems,
36:70293–70332, 2023.
[23] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin,
David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show
your work: Scratchpads for intermediate computation with language models.arXiv preprint
arXiv:2105.05246, 2021.
[24] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying
Song, and Denny Zhou. Large language models cannot self-correct reasoning yet.arXiv preprint
arXiv:2310.01798, 2023.
[25] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts
prompting: Disentangling computation from reasoning for numerical reasoning tasks.arXiv
preprint arXiv:2211.12588, 2022.
[26] Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki,
and Chris Callison-Burch. Faithful chain-of-thought reasoning. InThe 13th International Joint
Conference on Natural Language Processing and the 3rd Conference of the Asia-Pacific Chapter
of the Association for Computational Linguistics (IJCNLP-AACL 2023), 2023.
[27] Liangming Pan, Alon Albalak, Xinyi Wang, and William Wang. Logic-lm: Empowering
large language models with symbolic solvers for faithful logical reasoning. InFindings of the
Association for Computational Linguistics: EMNLP 2023, pages 3806–3824, 2023.
[28] Theo Olausson, Alex Gu, Ben Lipkin, Cedegao Zhang, Armando Solar-Lezama, Joshua Tenen-
baum, and Roger Levy. Linc: A neurosymbolic approach for logical reasoning by combining
language models with first-order logic provers. InProceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pages 5153–5176, 2023.
[29] Long Hei Matthew Lam, Ramya Keerthy Thatikonda, and Ehsan Shareghi. A closer look at
logical reasoning with llms: The choice of tool matters.arXiv preprint arXiv:2406.00284, 2024.
[30] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval
augmented language model pre-training. InInternational conference on machine learning,
pages 3929–3938. PMLR, 2020.
[31] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. Structgpt:
A general framework for large language model to reason over structured data.arXiv preprint
arXiv:2305.09645, 2023.
11

[32] Antonia Creswell, Murray Shanahan, and Irina Higgins. Selection-inference: Exploiting large
language models for interpretable logical reasoning.arXiv preprint arXiv:2205.09712, 2022.
[33] Franz Baader.The description logic handbook: Theory, implementation and applications.
Cambridge university press, 2003.
[34] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu. Bge m3-embedding: Multi-lingual,
multi-functionality, multi-granularity text embeddings through self-knowledge distillation.arXiv
preprint arXiv:2402.03216, 4(5), 2024.
[35] Leonardo De Moura and Nikolaj Bjørner. Z3: An efficient smt solver. InInternational
conference on Tools and Algorithms for the Construction and Analysis of Systems, pages
337–340. Springer, 2008.
[36] Simeng Han, Hailey Schoelkopf, Yilun Zhao, Zhenting Qi, Martin Riddell, Wenfei Zhou, James
Coady, David Peng, Yujie Qiao, Luke Benson, et al. Folio: Natural language reasoning with
first-order logic. InProceedings of the 2024 Conference on Empirical Methods in Natural
Language Processing, pages 22017–22031, 2024.
[37] Wanjun Zhong, Siyuan Wang, Duyu Tang, Zenan Xu, Daya Guo, Yining Chen, Jiahai Wang,
Jian Yin, Ming Zhou, and Nan Duan. Analytical reasoning of text. InFindings of the Association
for Computational Linguistics: NAACL 2022, pages 2306–2319, 2022.
[38] Oyvind Tafjord, Bhavana Dalvi, and Peter Clark. Proofwriter: Generating implications, proofs,
and abductive statements over natural language. InFindings of the Association for Computa-
tional Linguistics: ACL-IJCNLP 2021, pages 3621–3634, 2021.
[39] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid,
Adam Fisch, Adam R Brown, Adam Santoro, Aditya Gupta, Adrià Garriga-Alonso, et al.
Beyond the imitation game: Quantifying and extrapolating the capabilities of language models.
Transactions on machine learning research, 2023.
[40] Abulhair Saparov and He He. Language models are greedy reasoners: A systematic formal
analysis of chain-of-thought.arXiv preprint arXiv:2210.01240, 2022.
[41] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing
a multi-hop qa dataset for comprehensive evaluation of reasoning steps.arXiv preprint
arXiv:2011.01060, 2020.
[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov,
and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 conference on empirical methods in natural language
processing, pages 2369–2380, 2018.
[43] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique:
Multihop questions via single-hop question composition.Transactions of the Association for
Computational Linguistics, 10:539–554, 2022.
[44] Stephen E Robertson and Steve Walker. Some simple effective approximations to the 2-poisson
model for probabilistic weighted retrieval. InSIGIR’94: Proceedings of the Seventeenth Annual
International ACM-SIGIR Conference on Research and Development in Information Retrieval,
organised by Dublin City University, pages 232–241. Springer, 1994.
[45] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand
Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning.
arXiv preprint arXiv:2112.09118, 2021.
[46] Jianmo Ni, Chen Qu, Jing Lu, Zhuyun Dai, Gustavo Hernandez Abrego, Ji Ma, Vincent Zhao,
Yi Luan, Keith Hall, Ming-Wei Chang, et al. Large dual encoders are generalizable retrievers.
InProceedings of the 2022 Conference on Empirical Methods in Natural Language Processing,
pages 9844–9855, 2022.
12

[47] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D
Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. InThe Twelfth
International Conference on Learning Representations, 2024.
[48] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming
Zhang, and Dong Yu. Dense x retrieval: What retrieval granularity should we use? In
Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
pages 15159–15177, 2024.
[49] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag:
Neurobiologically inspired long-term memory for large language models.Advances in Neural
Information Processing Systems, 37:59532–59569, 2024.
[50] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving
retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In
Proceedings of the 61st annual meeting of the association for computational linguistics (volume
1: long papers), pages 10014–10037, 2023.
[51] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory:
Non-parametric continual learning for large language models.arXiv preprint arXiv:2502.14802,
2025.
13

A Prompt Templates
Appendix A provides the specific prompt templates utilized within the SymbolLKG framework to
facilitate reproducibility and illustrate the detailed interaction mechanisms between the system and
the Large Language Model.
A.1 Prompt for LKG Construction
Task Description: Extract a structured Logic Knowledge Graph from the context and the question based on the strict Ontology Rules below.
Ontology Rules:
  1. Nodes Classification:
      - Entity: Specific instances/individuals. 
      Note: For the 'type' field of an Entity, infer the specific semantic class (e.g., 'Person', 'Book', 'Course', 'Location'), DO NOT just put 'Entity’.
      - Concept: Abstract classes/types (e.g., 'Student', 'Course', 'Exam', 'Grade', 'Chair’).
      Note: The 'type' field for concepts should be 'Concept'.
      - Structure for Entity/Concept:
           "Name": The name of the entity or concept mentioned in the text.
           Note: "name" can't be start with a number, and can't contain special characters except underscore (_) and space.
           "Type": The type of the entity or concept mentioned in the text.
  2. Relationship Restrictions:        
      - Entity -> Concept: Use relation "IS_A" (e.g., Alice IS_A Student).
      - Concept -> Concept: Abstract relationships (e.g., Course RELATED_TO Exam).
      - Entity -> Entity: Extract Specific Relationships from text.
          Examples: 1. If text says "Alice sits left of Bob", use relation "LEFT_OF"; 2. If text says "Peter is father of John", use relation "FATHER_OF".
          Note: Avoid generic "RELATED_TO" if a more specific verb exists in the text.
      - Structure for Relationships:
           "source": The name of the entity or concept on the left side of the relationship.
           "relation": The specific relationship verb or phrase (e.g., "LEFT_OF", "FATHER_OF").
           "target": The name of the entity or concept on the right side of the relationship.
           "raw_text": The exact natural language text segment from the input that defines this relationship.
               Note: "raw_text" must be the same sentence in the input text with no information loss or added. 
  3. Logical Rules & Constraints:
      - Extract "Rules" (implications/logic formulas).
      - Extract "Constraints" (restrictions).
      - Structure for Rules/Constraints:
           "expression": The formal logic or mathematical formula.
           "raw_text": The exact natural language text segment from the input that defines this constraint/rule.
           Note: "raw_text" must be the same sentence in the input text with no information loss or added.
           "description": A short summary.
           "related_entities": Entities/Concepts mentioned in the rule/constraint context.
           "Constriant_type": The type of extracted constraints (For Constraints Only)
           Note: Constraint Type MUST be one of:
             "Ordering": Sequencing, time, positioning, comparison (e.g., "X is before Y", "higher than", "left of").
             "AllDifferent": Unique assignment (e.g., "Each X has a different Y", "Exactly one X per Y").
             "Arithmetic": Mathematical operations, counts, differences, or specific value assignments.
             "Generic": Only if it strictly does not fit the above categories.
  4. Linking Strategy:
      - For every Rule and Constraint, you MUST explicitly list ALL related Entity and Concept names in the "related_entities" field.
      - Look back at your extracted "concepts" list. If a constraint mentions "Each chair...", and you have a Concept "Chair", you MUST include "Chair" in the 
constraint's entity list. Make sure each element in the "related_entities" DO EXISTS in the Entity/Concept set you have extracted before.
  5. Note: 
      - One sentence in the text might contain multiple relationships.
      - Some relationships are hidden in the context, you can extract by analyzing the context and the meaning of pronouns like "he", "she", "it", "they".
      - Before output, check if you have extracted all the nodes and relationships carefully, also check if the each raw_text is the same with a sentence in text.
      - Do output in strict JSON format.
Figure 3:Prompt template for Logical Knowledge Graph Construction. This prompt instructs the LLM to
extract structured logic from text based on a dynamic ontology. It defines specific schemas for Entities, Concepts,
and logical relationships (Rules/Constraints), enforcing a strict JSON output format to ensure graph integrity.
14

A.2 Prompt for Solvers
Task Description: Your task is to analyze natural language problems and generate executable Z3 solver code that correctly answers the question. You 
will receive:
  1. A QUESTION that needs to be answered
  2. A LOGFILE containing multiple pieces of information related to the question (Maybe contains noises)
 Your job is to:
  1. Process ALL information from the logfile, regardless of whether it appears relevant
  2. Generate complete Z3 Python code that encodes ALL given information
  3. Ensure the code correctly answers the question
Requirement: You must include ALL information from the context in your code. DO NOT attempt to filter or judge what is relevant vs irrelevant.
Note:
  - For every entity mentioned, create a variable
  - For every stated fact, add a constraint
  - For every relationship described, encode it in Z3
  - Even if something seems completely unrelated, include it
Output Structure: 
  Your output should be a complete, runnable Python script that:
    - Imports Z3: from z3 import *
    - Encodes ALL information from the context
    - Correctly answers the question
    - Prints the answer in a clear format
Z3 Code Generation Guide: 
from z3 import *
 
def solve_problem():
    # ===== PART 1: VARIABLE DECLARATION =====
    # Declare variables for EVERY entity mentioned in context
    # Use appropriate types: String, Int, Bool, etc.
    # Example: entity1_property = String('entity1_property')
    
    # ===== PART 2: SOLVER INITIALIZATION =====
    s = Solver()
    
    # ===== PART 3: ENCODE ALL INFORMATION =====
    # Add constraints for EVERY fact in the context
    # Do NOT filter or skip any information
    # Example: s.add(entity1_property == "value_from_context")
    
    # ===== PART 4: ENCODE RELATIONSHIPS =====
    # Encode all relationships between entities
    # Use Implies() for conditional relationships
    
    # ===== PART 5: ENCODE QUESTION LOGIC =====
    # Create variables/constraints needed to answer the question
    # Example: are_equal = Bool('are_equal')
    #          s.add(are_equal == (var1 == var2))
    
    # ===== PART 6: SOLVE =====
    result = s.check()
    
    # ===== PART 7: EXTRACT AND FORMAT ANSWER =====
    if result == sat:
        m = s.model()
        # Extract answer
        # Format output clearly
        return formatted_answer
    elif result == unsat:
        return "No solution exists"
    else:
        return "Unknown"Z3 Code Generation Example:
from z3 import *
 
def solve_problem():    
    # ===== PART 1: CREATE SOLVER =====
    s = Solver()
    
    # === PART 2: DEFINE CONSTANTS (using integer encoding) ===
    COUNTRY_INDIA = 1
    COUNTRY_USA = 2
    (…more country encoding…) 
    PERSON_SAMIR_GANGULY = 1
    PERSON_DAVID_RUSSELL = 2
    (…more person encoding…)    
    MOVIE_SHARMEELEE = 1
    MOVIE_AMERICAN_HUSTLE = 2
    
    # ===== PART 3: CREATE VARIABLES =====
    # Director variables
    sharmeelee_director = Int('sharmeelee_director')
    american_hustle_director = Int('american_hustle_director')
    (… more variables created…)
    
    # ===== PART 4: ADD CONSTRAINTS =====
    # 1. Sharmeelee is directed by Samir Ganguly
    s.add(sharmeelee_director == PERSON_SAMIR_GANGULY)
    s.add(directs(PERSON_SAMIR_GANGULY, 
MOVIE_SHARMEELEE) == True)
    (…more constraints…)    
    # ===== PART 5: QUESTION LOGIC =====
    # Get nationalities of both directors
    director1_nat = person_nationality(sharmeelee_director)
    director2_nat = person_nationality(american_hustle_director)
    
    # ===== PART 6: SOLVE =====
    result = s.check()
    
    # ===== PART 7: OUTPUT RESULTS =====
    (Python Judge Code)
   
Figure 4:Prompt template for the Symbolic Solver (Z3) Code Generation. This prompt guides the model to
translate the retrieved logical subgraph into an executable Python script utilizing the Z3 SMT solver. It enforces
a structured coding process: declaring variables for entities, encoding facts as constraints, and formulating the
specific query logic to verify the answer.
15

A.3 Prompt for Pruner
Task Description: Your task is to identify which of the provided Graph Nodes are relevant to answering the specific Query.
Criteria for Relevance:
1. Direct Mention: The node contains entities or attributes explicitly mentioned in the query.
2. Logical Dependency: The node represents a rule, constraint, or relationship that logically affects the entities in the query (e.g., "All men are mortal" is 
relevant to "Is Socrates mortal?").
3. Contextual Isolation: STRICTLY EXCLUDE nodes from unrelated scenarios (e.g., rules about "fruit pricing" are irrelevant to a "family tree" query).
Example:
Query: "Determine the seating arrangement of Alice."
Candidates: [{"ID": "node_1", "Type": "Entity", "Content": "Alice"}, {"ID": "node_2", "Type": "Rule", "Content": " If Person A sits next to Person B, 
they cannot be enemies ."},…]
Relevant IDs: ["node_1", "node_2", "node_4"]
(Reasoning: 'Alice' and 'Bob' are people involved in seating. The 'Rule' affects seating logic. The 'Constraint' about a car is completely irrelevant.)
Your Task:
Query: "{query}"
Candidates:
{candidates_text}
Output Format:
Return strictly a JSON list of the IDs of the relevant nodes. 
Do not explain. Return ONLY the JSON.
Figure 5:Prompt template for the Logic Pruning Module. This prompt instructs the LLM to filter the expanded
candidate set.
A.4 Prompt for Router
Task Description: Analyze the following User Query and the Structured Knowledge retrieved from a database to determine the best Solver 
Engine.
Solver Definitions:
1. Z3: Choose this if the problem involves:
     - Arithmetic operations (math, calculation, greater than, less than).
     - Ordering/Sequencing (left of, before, rank, next to).
     - Global Constraints (AllDifferent, Sudoku-style puzzles).
     - "How many", "What is the value", "Sort" questions.
     - question with multiple choices.
           
2. PROVER9: Choose this if the problem involves:
     - Complex First-Order Logic proofs using contradiction.
     - Deep logical entailment checks (Does A strictly imply B?).
     - Abstract logical theorems without specific numbers.
     - Proving universal truths based on axioms.
           
3. PYKE: Choose this if the problem involves:
     - ONLY simple logical questions without XOR, XNOR logic.
     - Knowledge-based inference (Expert Systems).
     - "Horn Clause" logic (If A then B, If B then C...).
     - Forward-chaining or Backward-chaining tasks.
     - Finding relationships in a specific knowledge base (e.g., Family trees, simple causal chains).
Context: User Query: "{query}"
Retrieved Knowledge:
     - Constraints Found: {json.dumps(constraint_types)}
     - Rules Found (Count): {rule_count} (Example: {rules[0]['description'] if rules else 'None'})
     - Entities Found (Count): {entity_count}
Note:
     Return ONLY one word: Z3 OR PROVER9 OR PYKE.
Figure 6:Prompt template for the Adaptive Logic Router. This prompt guides the model to classify the reasoning
task by evaluating the retrieved constraints and rules. It explicitly defines the decision boundaries for solver
selection: Z3 is chosen for arithmetic and constraint satisfaction problems (CSPs), Prover9 for first-order logic
theorem proving, and Pyke for knowledge-based relational inference.
16

B Case Study
Appendix B presents a comprehensive case study derived from the AR-LSAT benchmark to demon-
strate the end-to-end workflow of the SymbolLKG framework. It demonstrates the complete lifecycle
of a logical reasoning task, detailing the intermediate outputs at each stage: from the initial extraction
of the Logical Knowledge Graph (LKG) components (Entities, Concepts, Rules, and Constraints) to
the dynamic inference process involving hybrid retrieval, logic routing, and the final generation of
executable Z3 symbolic code.
NL Corpus:
{
    "id": "ar_lsat_199106_2-G_2_8",
    "context": "A small software firm has four offices, numbered 1, 2, 3, and 4. Each of its offices has exactly one computer and exactly one printer. Each of these 
eight machines was bought in either 1987, 1988, or 1989. The eight machines were bought in a manner consistent with the following conditions: The computer in 
each office was bought either in an earlier year than or in the same year as the printer in that office.
The computer in office 2 and the printer in office 1 were bought in the same year. The computer in office 3 and the printer in office 4 were bought in the same year. 
The computer in office 2 and the computer in office 3 were bought in different years. The computer in office 1 and the printer in office 3 were bought in 1988.",
    "question": "If the computer in office 3 was bought in an earlier year than the printer in office 3 was, then which one of the following statements could be true?",
    "options": [
      "A) The computer in office 2 was bought in 1987.",
      "B) The computer in office 2 was bought in 1988.",
      "C) The computer in office 4 was bought in 1988.",
      "D) The printer in office 4 was bought in 1988.",
      "E) The printer in office 4 was bought in 1989."
    ]
  }
LLM Extraction Result:
{"entities":[{"name":"SmallSoftwareFirm","type":"Organization"},{"name":"Year1987","type":"Year"},{"name":"Year1988","type":"Year"},…], 
"concepts":[{"name":"Computer","type":"Concept"},{"name":"Printer","type":"Concept"},{"name":"Office","type":"Concept"},…],
"relationships":[{"source":"Small Software Firm","target":"Office 1","relation":"HAS","raw_text":"A small software firm has four offices, numbered 1, 2, 3, 4},…], 
"rules":[{"expression":"Year(Computer_1) <= Year(Printer_1) AND Year(Computer_2) <= Year(Printer_2) AND Year(Computer_3) <= Year(Printer_3) AND 
Year(Computer_4) <= Year(Printer_4)","raw_text":"The computer in each office was bought either in an earlier year than or in the same year as the printer in that 
office.","description":"The computer in each office was bought either in an earlier year than or in the same year as the printer in that office.", 
"related_entities":["Computer1","Computer 2","Computer 3","Computer 4","Printer 1","Printer 2","Printer 3","Printer 4"]},…],
"constraints":[{"type":"AllDifferent","expression":"AllDifferent(Computer, Year1987, Year1988, Year1989) AND AllDifferent(Printer, Year1987, Year1988, 
Year1989)","raw_text":"Each of these eight machines was bought in either 1987, 1988 or 1989."},…]}
Inference Pipeline: See Inference Output log on the next page
Generated Z3 Code:
 from z3 import *
 def solve_problem():
    # ===== PART 1: VARIABLE DECLARATION =====
    # Declare variables for every entity mentioned in the context
    year_computer_1 = Int('year_computer_1')
    year_computer_2 = Int('year_computer_2')
    …    
    # ===== PART 2: SOLVER INITIALIZATION =====
    s = Solver()
    
    # ===== PART 3: ENCODE ALL INFORMATION =====
    # Add constraints for every fact in the context
    s.add(year_computer_1 <= year_printer_1)
    s.add(year_computer_2 <= year_printer_2)
    s.add(year_computer_3 <= year_printer_3)
    (… more information encoded…)    
    # ===== PART 4: ENCODE RELATIONSHIPS =====
    # Add constraints for the question logic
    s.add(year_computer_3 < year_printer_3)
    
    # ===== PART 5: ENCODE QUESTION LOGIC =====
    # Create variables for the options
    option_A = Bool('option_A')
    s.add(option_A == (year_computer_2 == 1987))
    (the same encoding method to option B,C,D,E)
        
    # ===== PART 6: SOLVE =====
    result = s.check()
    
    # ===== PART 7: EXTRACT AND FORMAT ANSWER =====
    * Python Judge Code(omitted)
Figure 7:Case Study: Logical Knowledge Graph Extraction Result. The figure displays the structured JSON
output extracted from an AR-LSAT logical reasoning problem. It demonstrates how the system identifies entities
(e.g., Office, Computer), maps concepts, and formalizes complex constraints (e.g., “AllDifferent”, “Ordering”)
and logical rules from the raw text context.
17

 Inference Log Output:
>>> [Inference Pipeline] Step 1: Hybrid Search for '['Computer', 'Office 3', 'printer', 'Office 3', 'same', 'year']' (Pruning: True)
  
--- Vector Search Rules ---
  Found Rule: Year(Computer_3) == Year(Printer_4) (Desc: The computer in office 3 and the printer in office)... (Score: 0.8921) [KEPT for Pruning]
  Found Rule: Year(Computer_2) == Year(Printer_1) (Desc: The computer in office 2 and the printer in office)... (Score: 0.8641) [KEPT for Pruning]
  … 
--- Vector Search Constraints ---
  Found Constraint: AllDifferent(Computer, Year1987, Year1988, Year198... (Score: 0.8294) [KEPT for Pruning]
  …
--- Vector Search Entities ---
  Found Entity: Office (Office) (Score: 0.8726) [KEPT for Pruning]
  Found Entity: Printer (Printer) (Score: 0.8436) [KEPT for Pruning]
  … 
--- LLM Pruning (24 candidates) ---
  LLM decided to keep 10 nodes out of 24.
  [KEPT] Rule: Expression: Year(Computer_3) == Year(Printer_4), Description: The computer in office 3 and the printer in office 4 were bought in the same year.
  [DROPPED] Rule: Expression: Year(Computer_2) == Year(Printer_1), Description: The computer in office 2 and the printer in office 1 were bought in the same year.
  … 
--- Graph Traversal from 10 nodes ---
  -> Constraint via [Computer]: AllDifferent(Computer, Year198...
  -> Rule via [Computer 3]: Year(Computer_2) != Year(Compu...
  Found: 11 Entities, 4 Rules, 1 Constraints
 
>>> [Inference Pipeline] Step 2: Recall (Top-2/5 Raw Text Extraction)
    Recall@2 Results: 2 unique items
    1. The computer in office 3 and the printer in office 4 were bought in the same year.
    2. The computer in each office was bought either in an earlier year than or in the same year as the printer in that office.
    Recall@5 Results: 5 unique items
    1. The computer in office 3 and the printer in office 4 were bought in the same year.
    2. The computer in each office was bought either in an earlier year than or in the same year as the printer in that office.
    3. Each of its offices has exactly one computer and exactly one printer.
    … 
>>> [Inference Pipeline] Step 3: Solver Routing
    [Router Decision]: Z3
 
>>> [Inference Pipeline] Step 4: Execution Plan
    -> Action: Translate Arithmetic/Ordering constraints to Z3 Python API.
    -> Action: Check Satisfiability or Optimize.Figure 8:Case Study: Inference Pipeline Log. This log trace illustrates the step-by-step execution flow: (1)
Hybrid Search retrieves relevant rules regarding “Year” and “Office”; (2) Pruning filters candidates down to the
essential logic; (3) The Logic Router correctly identifies the task as an arithmetic/ordering problem and selects
the Z3 Solver; (4) The system generates and executes valid Z3 code to derive the final answer.
C Time Complexity Analysis
C.1 Empirical Latency
We profile per-sample wall-clock time onN= 100instances per benchmark.
Multi-hop QATable 6 reports the per-sample latency of the full SymbolLKG pipeline on the three
multi-hop QA datasets. End-to-end latency ranges from 166.0 s on 2WikiMultiHopQA to 407.1 s on
MuSiQue, with the bulk of the cost ( 159.2 –395.8 s) spent on per-question LKG construction. The
LKG itself triggers entity, rule, and constraint extraction over the question’s 8–10paragraph context;
the answer-generation call afterwards takes only6.0–11.4s.
Table 6:Per-sample wall-clock latency (seconds) on the multi-hop QA pipeline, averaged over N= 100
instances per benchmark.
Dataset LKG LLM Total
2Wiki 159.2 6.79 165.96
HotpotQA 208.5 6.00 214.57
MuSiQue 395.8 11.35 407.14
Logical ReasoningTable 7 reports the per-sample latency of the logical reasoning pipeline (single-
path: LKG construction plus the routed-solver path including self-refining retries). End-to-end
latency averages 122.2 s per problem (LKG construction 29.0 s, routed solver 93.2 s). ProofWriter
and AR-LSAT exhibit the longest mean times ( 166.5 s and 135.6 s respectively), reflecting their
longer Prover9 / Z3 programs and higher retry rates; ProntoQA is shortest ( 71.4 s) thanks to the
lightweight Pyke forward-chaining path.
18

Table 7:Per-sample wall-clock latency (seconds) on the logical reasoning pipeline, averaged over 20 instances
per dataset ( N= 100 in total). “Total” reflects single-path production deployment (LKG construction plus the
routed-solver path with up to three self-refining retries).
Dataset LKG Solver Total
AR-LSAT 33.1 102.5 135.6
LogicalDeduction 18.9 103.3 122.2
FOLIO 23.4 91.8 115.2
ProntoQA 34.3 37.1 71.4
ProofWriter 35.1 131.4 166.5
Average 29.0 93.2 122.2
C.2 Discussion
The dominant cost is LLM API latency rather than graph operations: across both pipelines, the
two recurring LLM calls (LKG extraction plus solver code generation or answer generation) jointly
account for over 99% of wall-clock time, while retrieval and BFS combined remain below 0.1s
per query. Three properties of the framework limit the practical impact of this overhead. (i) For
fixed corpora served at deployment time, the LKG can be built once offline and reused across all
queries, eliminating the 155–494s per-question construction cost that dominates the multi-hop QA
numbers; under this amortisation, the inference-time cost reduces to retrieval ( <0.1 s) plus a single
answer-generation call ( ∼10 s). (ii) The self-refining loop is bounded at three retries, so worst-case
solver latency is at most 3×the single-attempt cost. (iii) The routing and pruning calls operate
on bounded-length prompts and can be served by a smaller, cheaper backbone without materially
affecting accuracy. The remaining overhead reflects the cost of trading probabilistic single-pass
inference for verifiable symbolic reasoning.
19