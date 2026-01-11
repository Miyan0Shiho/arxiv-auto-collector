# A Navigational Approach for Comprehensive RAG via Traversal over Proposition Graphs

**Authors**: Maxime Delmas, Lei Xu, André Freitas

**Published**: 2026-01-08 11:50:40

**PDF URL**: [https://arxiv.org/pdf/2601.04859v1](https://arxiv.org/pdf/2601.04859v1)

## Abstract
Standard RAG pipelines based on chunking excel at simple factual retrieval but fail on complex multi-hop queries due to a lack of structural connectivity. Conversely, initial strategies that interleave retrieval with reasoning often lack global corpus awareness, while Knowledge Graph (KG)-based RAG performs strongly on complex multi-hop tasks but suffers on fact-oriented single-hop queries. To bridge this gap, we propose a novel RAG framework: ToPG (Traversal over Proposition Graphs). ToPG models its knowledge base as a heterogeneous graph of propositions, entities, and passages, effectively combining the granular fact density of propositions with graph connectivity. We leverage this structure using iterative Suggestion-Selection cycles, where the Suggestion phase enables a query-aware traversal of the graph, and the Selection phase provides LLM feedback to prune irrelevant propositions and seed the next iteration. Evaluated on three distinct QA tasks (Simple, Complex, and Abstract QA), ToPG demonstrates strong performance across both accuracy- and quality-based metrics. Overall, ToPG shows that query-aware graph traversal combined with factual granularity is a critical component for efficient structured RAG systems. ToPG is available at https://github.com/idiap/ToPG.

## Full Text


<!-- PDF content starts -->

A NAVIGATIONALAPPROACH FORCOMPREHENSIVERAGVIA
TRAVERSAL OVERPROPOSITIONGRAPHS
A PREPRINT
Maxime Delmas∗, 1, Lei Xu1,2, and André Freitas1,3,4
1Idiap Research Institute, Switzerland
2École Polytechnique Fédérale de Lausanne (EPFL), Switzerland
3Department of Computer Science, University of Manchester, United Kingdom
4Cancer Biomarker Centre, CRUK Manchester Institute, United Kingdom
ABSTRACT
Standard RAG pipelines based on chunking excel at simple factual retrieval but fail on complex
multi-hop queries due to a lack of structural connectivity. Conversely, initial strategies that interleave
retrieval with reasoning often lack global corpus awareness, while Knowledge Graph (KG)-based
RAG performs strongly on complex multi-hop tasks but suffers on fact-oriented single-hop queries.
To bridge this gap, we propose a novel RAG framework: ToPG (Traversal over Proposition Graphs).
ToPG models its knowledge base as a heterogeneous graph of propositions, entities, and passages,
effectively combining the granular fact density of propositions with graph connectivity. We leverage
this structure using iterative Suggestion-Selection cycles, where the Suggestion phase enables a query-
aware traversal of the graph, and the Selection phase provides LLM feedback to prune irrelevant
propositions and seed the next iteration. Evaluated on three distinct QA tasks (Simple, Complex,
and Abstract QA), ToPG demonstrates strong performance across both accuracy- and quality-based
metrics. Overall, ToPG shows that query-aware graph traversal combined with factual granularity is
a critical component for efficient structured RAG systems. ToPG is available at https://github.
com/idiap/ToPG.
1 Introduction
Retrieval-Augmented Generation (RAG) has become the dominant paradigm for grounding Large Language Models
(LLMs). RAG directly addresses the limitations of static parametric memory, mitigating hallucinations [ 1] and
improving recall, particularly for long-tail knowledge [ 2]. The standard RAG pipeline relies on Dense Passage Retrieval
(DPR) over chunked documents [ 3]. While large embedding models (e.g., NV-Embed-v2 [ 4]) have achieved state-
of-the-art performance on the MTEB benchmark [ 5], retrieval granularity represents a critical, often overlooked line
for improvements. Coarse-grained passages often contain irrelevant or distracting information that degrades LLM
generation [ 6]. Conversely, proposition-level retrieval (decomposing text into decontextualized atomic facts) has proven
superior for direct, single-hop QA and fact checking [7, 8].
Real-world complex queries often require multi-hop reasoning that necessitates connecting disparate pieces of evidence
across documents. While iterative retrieval and Chain-of-Thought (CoT) approaches [ 9] commonly operationalize this
via successive local searches, they inherently lack a global, structured view of the corpus. To bridge this structural gap,
structure-augmented RAG strategies have integrated Knowledge Graphs (KGs) [ 10]. These methods explicitly model
entities and relationships to support both multi-hop inference and broader, abstract queries [11].
Despite their structural advantages, current approaches face fundamental challenges. First, they lead to information
loss as standard KGs enforce triples (s,p,o) representations, compressing complex text into binary relations. Second, a
practical challenge exists in navigating the graph. Current strategies are broadly polarized between methods relying
on purely topological heuristics (e.g., neighbours, random walks, etc.) and thus inherently ignoring edge semantics,
∗Corresponding author: maxime.delmas@idiap.charXiv:2601.04859v1  [cs.CL]  8 Jan 2026

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Figure 1: The ToPG framework. The system operates on a heterogeneous graph where propositions connect entities and
passages. ToPG navigates this graph using iterative Suggestion-Selection cycles, allowing for three operational modes:
Naive(factoid retrieval),Local(multi-hop inference), andGlobal(community-based search).
supervised GNNs [ 12], or LLM-driven exploration [ 13]. To this end, we introduce ToPG (Traversal Over Proposition
Graphs), a novel RAG framework that combines the granularity of propositions withquery-aware graph traversal
(Figure 1).
Unlike traditional KGs, we model the knowledge base as a heterogeneous graph of entities, propositions, and passages.
This structure retains the semantic richness of atomic facts while enabling the topological connectivity of a graph. To
leverage this structure, we propose a graph exploration method based onSuggestion-Selection cycles. The Suggestion
phase leverages both query similarity and graph topology to efficiently suggest new relevant propositions. The
subsequent Selection phase acts as a feedback mechanism, using in-context LLM-based interpretation to prune
irrelevant suggestions and seed the next iteration with high-quality evidence. To address diverse QA requirements, ToPG
supports three complexity levels: Naive proposition retrieval, Local multi-hop inference, and Global community-based
abstract QA.
2 Methods
2.1 Graph Construction
We represent the knowledge base as a heterogeneous graph G= (V, E) where the node set V=V p∪Ve∪VPcomprises
three disjoint types of nodes: atomic factual statements (propositions Vp), named entities that appear within propositions
(entities Ve), and document segments that provide the source context for propositions (passages VP). The edge set
E=E p↔e∪Ep↔P contains two types of undirected edges, connecting each proposition to its associated entities and
to the passage from which it originates.
Given a document chunk, we apply an LLM-based in-context learning function (in a few-shot setting) to sequentially
extract named entities Veand propositions Vp. Each extracted proposition and entity node is encoded using an encoder
h(·). Entity reconciliation is performed using cosine similarity thresholding. Details in Appendix A.
In the resulting graph, each proposition effectively acts as a hyperedge linking multiple entities while being grounded in
textual evidence2. Passage nodes ( VP) serve a structural role by connecting propositions originating from the same
passage, thereby enforcing local neighborhood coherence. Unlike classicalentity-centricKGs, our representation is
2Reciprocally, entities also create hyperedges between propositions.
2

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
explicitlyproposition-centric: propositions are modeled as first-class nodes, enabling richer reasoning over factual,
compositional, and multi-hop relations.
2.2 Graph Navigation: Suggestion–Selection Cycles
Suggestion-Selection Retrieval.We propose to navigate the graph Gthrough Suggestion-Selection cycles. Concep-
tually, the suggestion step defines a function:
Snew= Suggestk(q, G, s old)(1)
Given a query q, a graph Gand a set of already collected proposition nodes sold, proposes knew potentially relevant
nodesS new.
An effective suggestion mechanism should account for both the semantic relevance of nodes to the query q, and the
connectivity of nodes to the seed set soldinG. Therefore, the suggestion process should ideally be both query and
graph aware.
The Selection phase defines a function that prune irrelevant propositions from the poolS new:
snew= Select(q, S new)(2)
It acts as feedback from the LLM and seeds the next iteration of Suggest by performing LLM-based relevance pruning:
PROMPT Select(q, S new). By modulating the query qand collected propositions soldduring iterative Suggestion-Selection
cycles, we can adapt the exploration behavior overGto different question types (see section 2.3).
Query and Graph Aware SuggestionsWe introduce a retrieval strategy based on a query-aware Personalized
PageRank (PPR) [ 14]. An intuitive example is available in Appendix B.1. We propose to determine new candidate
propositions as
Snew= topk 
PPR(M, s old)
,(3)
where the transition matrixMcombines structural and semantic information:
M= QueryAwareTransiton(q, G, λ)
=λT s+ (1−λ)T n.(4)
The parameter λcontrols the balance between structural and semantic guidance in M. The structural component Ts
encodes the topology of G: the higher the connectivity between two propositions through shared entities or passages, the
greater the probability of transition between them. Thus, Tscaptures connectivity to the seed nodes, but, is independent
of the current query q. In contrast, the semantic component Tnmaintains the same adjacency pattern as Ts, but weights
each potential transition (i, j) according to the similarity between node jand the query q, making nodes similar to
the query more attractive. Therefore, random walks are biased toward proposition nodes that are not only structurally
connected to the current context sold, but, also semantically relevant to the question. Intuitively, the resulting transition
matrix Mencourages exploration along paths that remain consistent with the graph structure, while biased toward
semantically relevant regions. Then, setting λ= 1 , gives a purely graph-based and query independent Suggest function.
More formally, Ts∈Rn×nis the degree-normalized transition matrix derived from the proposition–entity–passage
connectivity:
Ts=˜Ap→eP˜AeP→p ,(5)
where ˜Ap→eP and˜AeP→p denote the normalized transition matrices between propositions and entities/passages from
the graph.
Then, to build Tn, we compute the query-based similarities c= cosine(h(q), h(V p))and apply temperature scaling and
thresholding:
˜ci=exp(c i/τ),ifc i≥θ,
0,otherwise,(6)
where τis the temperature (default 0.1) and θis the cosine threshold (default 0.4). The semantic transition matrix is
then defined as
Tn(i, j) =˜cj1Ts(i,j)>0P
k˜ck1Ts(i,k)>0.(7)
In bothT sandT n, self-connections are also canceled (eg.T s(i, i) = 0).
3

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Subgraph ExtractionAs traversing the full graph Gis computationally expensive, we first extract a local subgraph
G∗using a Random Walk with Restart (hereafter named GExtract ) around the set of seeds with a target size l. This
process mitigates hub bias and constrains the exploration space.
2.3Naive,LocalandGlobalmodes
We propose three search modes: Naive for simple factual queries, Local for complex (eg., multi-hop) queries, and
Globalfor abstract questions.
Figure 2: On the left, panel Local shows a step-by-step example for the Local mode. On the right panel, step-by-step
description of theGlobalmode.
2.3.1Naive
Using a retrieval encoder h, a simple top- kretrieval based on cosine similarity Snew= topk 
cosine(h(q), h(V p))
can
be seen as a naive suggestion process: a retrieval over propositions that ignores both the graph Gand previously collected
nodes sold. We define this as SuggestNaivek(q, G,∅) .Naive mode uses propositions retrieved from SuggestNaive as
context to answer the question, bypassing the Selection step.
2.3.2Local
In the Local mode, the graph Gis explored through Suggestion-Selection cycles guided by LLM feedback up to a
maximum number of iterations ( max-iter ). A step-by-step example is provided in Figure 2 -Local . Starting from an
initial query qstart, it completes a local set of propositions sloc={u 1, u2,..., u m}that represent the evolving context
4

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
for answering the query. The initial propositions are collected with SuggestNaive and the irrelevant ones are pruned by
Select(step 1). This represents the initialseedingstep inG.
Then, at each iteration, new candidate propositions ( Snew) are proposed using SuggestLocal , conditioned on the current
query and the current context ( spool). Candidates are then pruned by Select . The retained propositions are added to sloc
and will seed the next iteration. In step 2, the first iteration yields nodesu 1andu 2.
The suggestionS new= SuggestLocalk(q, G, s pool)proceeds in three stages:
G∗= GExtract(G, s pool, l),
M= QueryAwareTransiton(q, G∗, λ),
π= PPR(M, s pool),
Snew= topk(π)(8)
If the accumulated propositions remain insufficient to answer the query (determined by PROMPT Eval(qstart, sloc)),
additional targeted sub-questions are generated to guide the next iteration via PROMPT NextQ(qstart, sloc). In the running
example (step 3), this yields two new queries, which in turn trigger a second Suggestion–Selection cycle. This cycle is
seeded on spool, containing u1andu2(step 4). At this iteration, Snewcontained the suggested propositions: u3,u4,u5,
andu6and the Select call pruned u5. With slocnow completed by u3,u4andu6, the query is answered in (step 5).
Details and an illustrative example in Appendix B.2.
2.3.3Global
While Local retrieval effectively handles fact-oriented or multi-hop reasoning tasks, abstract or conceptual queries
"How does soil health influence overall farm productivity?"require a broader and more diverse exploration of the graph
G. In such cases, identifying a missing fact or reasoning chain is insufficient, where a comprehensive answer spans
multiple, complementary and non-local perspectives that need to be retrieved from G. Rather than using PROMPT NextQ
to predict new directions/questions as in Local ,Global refines queries after each Suggestion-Selection cycle. Gathered
anchor propositions sglb, are then used to identify communities in G. Communities emerge naturally from the graph’s
topology and can represent potential facets (i.e., individual aspects or perspectives) relevant to the query. Intermediate
answers are generated from these communities, scored by relevance, and aggregated into the final response. A detailed
diagram of the step-by-step process is presented in Figure 2-Global.
Steps 1-2: SeedingThe parameter mcontrols the breadth of the exploration. We begin by decomposing the initial
query qstartintomsub-queries using PROMPT decompose . Each sub-query is sent to SuggestNaive and populates the pool
of anchors spoolafter irrelevant propositions are pruned with Select . This corresponds to theseedingstep and the first
anchors added tos glb.
Step 3: Compute QueriesAt each subsequent iteration, every proposition node in the current pool ( ui∈s pool)
becomes an independent exploration center, performing its own local walk through the graph. To guide these walks, we
refine the queryq ifor each proposition using relevance feedback [15]:
qi=αqo
i+βq+
i−γq−
i.(9)
Intuitively, qo
icaptures directions that previously led to ui;q+
iencodes the proposition uias a relevance signal from
Select , encouraging further exploration in this direction; and q−
idiscourages directions that were previously pruned.
Therefore, query vectors qiare refined by the LLM feedback provided by Select , guiding the exploration toward
promising directions while avoiding previously pruned paths (further details in Appendix B.3).
Step 4: Iteratively explore and collectTo prevent the search space from growing as more facts accumulate, spool
is partitioned into msubsets. The SuggestGlobal strategy operates independently on each partition spartwith its
associated queries qpart. Each node uiinsparthas its own query qiinqpart. Within a partition, a subgraph G∗is
extracted around spart, then, each proposition uiacts as a singleton seed {ui}with its query qi, performing an individual
query-aware random walk. The resulting probability distributions are aggregated within the partition, and the top- k
propositions are selected as new suggestions.
5

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
In summary,SuggestGlobalk(qpart, G, s part)is defined as:
G∗= GExtract(G, s part, l),
Mi= QueryAwareTransiton(q i, G∗, λ),
πi= PPR(M i,{ui}),
Snew= topk X
iπi)(10)
The process is iterated until min_facts are collected (or max-iter iterations are completed). See details in Appendix
B.4.
Step 5: Identifying communitiesThe final set of collected anchor propositions sglbis used to extract associated
communities from G. Communities are identified using a hierarchical Leiden algorithm [ 16]. We then follow a greedy
budgeted strategy: each community cis assigned a score|nodes(c)\S′|
size(c), representing how many yet-uncovered anchor
propositions ( S′) it includes relative to its size. Given a budget limit of Btotal nodes, communities with the highest
score are iteratively added to maximize coverage over anchor propositions. See details of the procedure in B.5.
Steps 6-7: Generating answersEach community contains a heterogeneous mix of nodes: propositions, entities, and
passages. Entities highlight central topics, propositions ground the key facts, and passages provide broader context
and connect propositions together. Community content is divided into chunks of pre-specified token size and used to
generate intermediary answers. Intermediary answers are ranked and combined into the final prompt, inserting the most
relevant information at the beginning and the end (“lost-in-the-middle” effect [ 17]), before generating the final answer.
3 Experimental Setup
We evaluate Naive ,Local , and Global modes across complementary QA settings. Our evaluation spans: (i)Simple
QA, testing the ability to retrieve isolated factual evidence; (ii)Complex QA, requiring the retrieval and composition
of multiple evidence (eg., multi-hop queries); and (iii)Abstract QA, involving conceptual or multi-faceted queries that
require broad, long-form synthesis beyond explicit facts.
3.1 Datasets
Simple QAFollowing prior work [ 18], we evaluated on a subset of 1,000 queries from PopQA [ 19] and included
GraphRAG-Benchmark [ 20] Task 1 (Fact Retrieval), covering two distinct corpora: Medical, containing NCCN clinical
guidelines, and Novel, a collection of pre-20th-century literary texts from Project Gutenberg.
Complex QA.We used the 1,000-query subsets of the multi-hop QA datasets HotPotQA [ 21] and MusiQue [ 22]
from Gutiérrez et al. [23]. We also included two GraphRAG-Benchmark tasks:Complex Reasoning, which requires
chaining multiple evidence, andContextual Summarization, which requires synthesis of fragmented information. For
them, we follow the Answer Accuracy metric [20].
Abstract QA.To evaluate abstract queries, we follow the LightRAG setup [ 24] and generate abstract questions on
three corpora from the UltraDomain benchmark (college-level textbooks): Agriculture, Computer Science, and Legal
[25]. We compare responses on 4 dimensions with LLM-as-a-judge [ 26]: Comprehensiveness, Diversity, Empowerment
and finally Overall. See details and examples of queries in Appendix C.
3.2 Settings and Baselines
For Simple and Complex QA, we evaluated three structure-augmented RAG baselines: GraphRAG [ 11], LightRAG
[24], and HippoRAG 2 [ 23]. For both Simple and Complex QA, we used k= 20 propositions for Naive andLocal
modes, assessing the latter with max-iter∈ {1,3} . Hyperparameters analysis can be found in Appendix D. To isolate
the benefits of proposition-level retrieval, we also include a vanilla passage-level RAG baseline that uses the same
prompting configuration as theNaivemode.
For Abstract QA, we evaluate the Global mode with varying numbers of collected anchor propositions (200–1000).
We compare two variants of query refinement: Rocchio-style feedback using (α=1, β=0.7, γ=0.15) [15],
incorporating both selected and pruned propositions; and Simple-feedback , where qiignores signals from Select
6

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
(α=1, β=γ=0) . For fairness, we evaluate against GraphRAG and LightRAG, as both explicitly support abstract-level
QA with dedicated global/hybrid modes.
To ensure fair comparison, all baselines use the same embedding model ( bge-large-en-v1.5 ) and the same open
LLM Gemma-3-27B [27] for indexing and inference using vLLM [ 28] on one H100. For experiments on GraphRAG-
Benchmark, we align with the associated protocol and used GPT-4o-mini [ 29] for both indexing and inference. For
additional details, please see Appendix E.
4 Results
Method MusiQue HotPotQA PopQA MEDICAL†NOVEL†
FR CR CS FR CR CS
Vanilla-RAG 19.7 / 30.6 52.7 / 65.5 49.2 / 62.2 63.7 57.6 63.7 58.8 41.4 50.1
GraphRAG (local) 17.8 / 26.7 47.3 / 60.2 38.1 / 52.6 38.6 47.0 41.9 49.3 50.964.4
LightRAG (local) 16.7 / 25.6 48.0 / 59.9 39.7 / 53.4 62.6 63.3 61.3 58.6 49.1 48.9
HippoRAG 2 24.7 / 36.2 55.1 / 66.9 38.4 / 48.6 66.3 62.0 63.1 60.1 53.4 64.1
ToPG-Naive19.5 / 30.3 49.2 / 61.051.6 / 63.9 72.968.5 67.7 67.355.663.7
ToPG-Local(1) 28.0 / 41.1 55.3 / 67.8 48.4 / 59.5 72.5 68.5 68.867.0 55.0 61.2
ToPG-Local(3)34.0 / 47.0 59.3 / 72.748.9 / 60.2 72.669.268.3 67.653.9 61.0
∆Local(3) -Naive ↑14.5 / 16.7 ↑10.1 / 11.7 ↓2.7 / 3.7 ↓0.3 ↑0.5 ↑0.6 ↑0.3 ↓1.7 ↓2.7
Table 1: Results on Simple and Complex QA tasks, highlighting thebestand second-best results. Performance (Exact
Match / F1) on Simple and Multi-Hop QA (Left: MusiQue, HotPotQA, PopQA) and GraphRAG-Benchmark tasks
(Right: Fact Retrieval, Complex Reasoning, Contextual Summarization) measured using the Answer Accuracy metric.
ToPG- Local (1 and 3) report results for ( max-iter= 1 and3) respectively. ∆Local (3) - Naive shows the difference
betweenLocal(max-iter= 3) andNaive.
Figure 3: Win rates (%) of ToPG against GraphRAG and LightRAG across 3 corpora and 4 criteria, with increasing
number of collected propositions (200-1000) and w/ or w/o Rocchio-style feedback.
Table 1 reports QA performance for Simple and Complex QA tasks. On Simple QA, Naive mode and Vanilla-RAG on
passages outperform graph-based approaches by a significant margin, particularly on PopQA. However, the structural
advantages of graph-based methods become apparent in Complex QA settings, notably in multi-hop scenarios where
ToPG- Local demonstrates superior performance. Interestingly, even when configured with max-iter= 1 , ToPG- Local
already exhibits significant improvements in multi-hop settings compared to its Naive mode. Increasing iterations to
max-iter= 3 yields substantial gains in multi-hop tasks but offers only marginal improvements in Complex Reasoning
and Contextual Summarization (Medical and Novel corpora). For summarization tasks, both GraphRAG (local) and
HippoRAG 2 also achieve competitive performance.
7

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Figure 4: Log10(Token-counts)between baselines evaluated on Agriculture, CS, Legal, MusiQue datasets.
Figure 3 illustrates the win rates of ToPG- Global against baselines on Abstract QA across four criteria (see an example
in Appendix F). ToPG- Global significantly outperforms LightRAG across all configurations, reaching comparable
performance with GraphRAG ( ≈50% win rate) on the Agriculture and CS datasets, though it underperforms on the
Legal dataset. While GraphRAG consistently outperforms on the Comprehensiveness axis, ToPG achieves greater
diversity and is perceived as more empowering in its answers. For all criteria except Comprehensiveness, increasing the
number of collected facts shows a positive impact that plateaus around 600propositions, beyond which performance
stagnates or degrades. In contrast, feedback settings for query refinement show only a negligible impact on overall
performance, providing a minor improvement only in Comprehensiveness.
Figure 4 compares the average token cost per abstract query, indicating that LightRAG has the lowest token cost for
both input and output tokens. GraphRAG is identified with the highest token cost, particularly regarding input tokens.
ToPG is cheaper than GraphRAG in completion tokens when configured with less than 600collected anchors, but is
more costly on the MusiQue dataset.
5 Discussion
Graph-based approaches demonstrate competitive performance, particularly in Complex QA (multi-hop), where the
graph layer effectively connects disparate named entities central to the query. Similarly to [ 30], we also note that this
structural advantage, however, is often detrimental or minimal for standard factual QA, where proposition-level retrieval
with ToPG-Naiveachieves higher information density due to their self-contained and factoid content.
While baselines typically construct a standard KG with subject-predicate-object triples, their traversal often relies purely
on topological heuristics (e.g., neighbours, random walks), thus neglecting the semantics encoded in the predicate.
ToPG proposes a query and graph aware Suggestion mechanism to explicitly leverage the semantics of propositions,
coupled with an LLM-driven Selection step that provides explicit feedback for the next iteration, but, entails the overall
token cost. This Suggestion-Selection mechanism, even with only one iteration ( max-iter= 1 ), significantly improves
performance over ToPG-Naiveand alternative baselines in multi-hop settings.
In abstract QA, both GraphRAG and ToPG- Global rely on iterative graph exploration and exploit the inner graph
modularity to extract and generate intermediary answers from node communities. While this process significantly
increases token costs, it significantly improves the depth (comprehensiveness, diversity, empowerment) of generated
answers over simpler keyword expansion strategies (e.g., LightRAG ). Moreover, ToPG is designed for easier scalability
and updates as it avoids pre-computing community summaries and instead uses Suggestion-Selection cycles for
community exploration. Our observations suggest that the utility of collecting additional anchors is saturated by the
current LLM’s reasoning capacity, implying that further benefits would only arise when using a stronger base model.
Overall, our results suggest that query-aware exploration over a graph of granular information units is the critical
component, rather than the formal structure of the KG with strict predicates. Traversal of the proposed heterogeneous
graph through an effective Suggestion-Selection mechanism shows robust performance and versatility across different
QA tasks.
8

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
6 Related Work
Early strategies for complex question answering combine retrieval with reasoning via interleaving DPR with CoT or
question decomposition techniques [ 9,31,32,33], an approach likewise employed in the Local mode. Furthermore,
propositions have been explored as an efficient granularity level, particularly for fact-oriented QA [ 34] and claim or fact
checking [ 8,35]. To address the need for global structural awareness, recent approaches construct KGs directly from
the corpus [ 10]. These systems seed retrieval using DPR over entities or triples and then navigate the resulting graph
using topological heuristics such as community detection (GraphRAG) [ 11], ego-network (LightRAG) [ 24], path search
(PathRAG) [36], or Personalized PageRank (HippoRAG, HippoRAG 2) [18, 23].
Combining propositions with graph structure, Wang and Han [37] proposes to apply a similar approach to HippoRAG
on a graph where nodes represent entities and passages, and edges link entities that co-occur within the same proposition.
Luo et al. [38] instead constructs a graph of propositions and perform neighborhood expansion after an initial seeding
step. Unlike these approaches, ToPG leverages its Suggestion-Selection cycles and query-aware traversal to support
three distinct modes tailored to different QA requirements: factoid, multi-hop, and abstract.
The Selection phase, which provides LLM-based feedback, also aligns with a broader line of work on LLM-guided KG
exploration [13, 39, 40]. These approaches typically alternate phases of search and pruning over entities and relations
in the KG. Finnaly, in contrast to GraphRAG or RAPTOR [ 41], which rely on pre-processed summaries for abstract QA
[42,43], ToPG instead derives intermediary answers directly from the communities extracted around anchor nodes
obtained through multiple Suggestion-Selection cycles.
7 Conclusion
ToPG reconciles fact-level granularity with graph connectivity through a heterogeneous graph composed of passages,
propositions, and entities. The proposed graph navigation strategy based on iterative Suggestion-Selection cycles, while
simple by design, proves highly versatile and adaptable to diverse QA requirements. The strategic modulation of the
query and the collected evidence enables distinct operational modes: Naive (for factoid retrieval), Local (for complex,
multi-hop reasoning), and Global (for abstract questions). Overall, our experiments demonstrate the efficacy of this
framework and suggest that structure-augmented RAG architectures should prioritize query-aware graph traversal and
factual granularity over the restrictive formal structure of traditional KGs.
8 Limitations
A primary limitation of our framework is the computational overhead in token cost, both during indexing and inference.
Similar to other structure-augmented methods, the process of extracting propositions and building the graph significantly
increases indexing costs compared to standard RAG. During inference, token consumption is inflated by the LLM-driven
Selection phase (in Local mode) and the generation of intermediate community answers (in Global mode). While
these mechanisms are essential for answer depth, they make ToPG less suitable for cost-critical scenarios compared to
lighter alternatives like LightRAG. Future work could mitigate this by replacing the LLM selector with a specialized,
lightweight classifier or by fine-tuning prompts for token efficiency.
Second, performance is also bound by the quality of the underlying graph. Relying on embedding similarity for
entity disambiguation can occasionally introduce noisy or misleading edges. Furthermore, while proposition extraction
enhances information density, it may result in minor information loss compared to full paragraphs. Therefore, integrating
external knowledge bases (e.g., Wikipedia or DBpedia) for more robust entity linking and maintaining hybrid access to
original passages could be beneficial. However, we deliberately restricted our evaluation to the proposition level for this
work.
Finally, while ToPG offers three distinct operational modes ( Naive ,Local ,Global ), the current framework lacks an
automated routing mechanism. A learned classifier capable of dynamically selecting the optimal mode based on the
query complexity would make the framework more end-to-end.
References
[1]Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua
Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles,
taxonomy, challenges, and open questions.ACM Trans. Inf. Syst., 43(2), January 2025. ISSN 1046-8188. doi:
10.1145/3703155. URLhttps://doi.org/10.1145/3703155.
9

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
[2]Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language models struggle to
learn long-tail knowledge. InProceedings of the 40th International Conference on Machine Learning, ICML’23.
JMLR.org, 2023.
[3]Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webber, Trevor Cohn,
Yulan He, and Yang Liu, editors,Proceedings of the 2020 Conference on Empirical Methods in Natural Language
Processing (EMNLP), pages 6769–6781, Online, November 2020. Association for Computational Linguistics.
doi: 10.18653/v1/2020.emnlp-main.550. URLhttps://aclanthology.org/2020.emnlp-main.550/.
[4]Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei
Ping. NV-embed: Improved techniques for training LLMs as generalist embedding models. InThe Thirteenth
International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=
lgsyLSsDRe.
[5]Niklas Muennighoff, Nouamane Tazi, Loic Magne, and Nils Reimers. MTEB: Massive text embedding benchmark.
In Andreas Vlachos and Isabelle Augenstein, editors,Proceedings of the 17th Conference of the European Chapter
of the Association for Computational Linguistics, pages 2014–2037, Dubrovnik, Croatia, May 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.eacl-main.148. URL https://aclanthology.org/2023.
eacl-main.148/.
[6]Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed Chi, Nathanael Schärli, and Denny
Zhou. Large language models can be easily distracted by irrelevant context. InProceedings of the 40th
International Conference on Machine Learning, ICML’23. JMLR.org, 2023.
[7]Sihao Chen, Hongming Zhang, Tong Chen, Ben Zhou, Wenhao Yu, Dian Yu, Baolin Peng, Hongwei Wang,
Dan Roth, and Dong Yu. Sub-sentence encoder: Contrastive learning of propositional semantic representations.
In Kevin Duh, Helena Gomez, and Steven Bethard, editors,Proceedings of the 2024 Conference of the North
American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1:
Long Papers), pages 1596–1609, Mexico City, Mexico, June 2024. Association for Computational Linguistics.
doi: 10.18653/v1/2024.naacl-long.89. URLhttps://aclanthology.org/2024.naacl-long.89/.
[8]Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer,
and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form text
generation. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore, December 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.741. URL https://aclanthology.org/
2023.emnlp-main.741/.
[9]Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-
of-thought reasoning for knowledge-intensive multi-step questions. In Anna Rogers, Jordan Boyd-Graber, and
Naoaki Okazaki, editors,Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pages 10014–10037, Toronto, Canada, July 2023. Association for Computational
Linguistics. doi: 10.18653/v1/2023.acl-long.557. URLhttps://aclanthology.org/2023.acl-long.557/.
[10] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao
Chen, Yi Chang, and Xiao Huang. A survey of graph retrieval-augmented generation for customized large
language models.arXiv preprint arXiv:2501.13958, 2025.
[11] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha
Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to
query-focused summarization, 2025. URLhttps://arxiv.org/abs/2404.16130.
[12] Costas Mavromatis and George Karypis. GNN-RAG: Graph neural retrieval for efficient large language model
reasoning on knowledge graphs. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher
Pilehvar, editors,Findings of the Association for Computational Linguistics: ACL 2025, pages 16682–16699,
Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/
v1/2025.findings-acl.856. URLhttps://aclanthology.org/2025.findings-acl.856/.
[13] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung
Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large language model on knowledge
graph. In B. Kim, Y . Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y . Sun, editors,International Conference
on Representation Learning, volume 2024, pages 3868–3898, 2024. URL https://proceedings.iclr.cc/
paper_files/paper/2024/file/10a6bdcabbd5a3d36b760daa295f63c1-Paper-Conference.pdf.
[14] Taher H. Haveliwala. Topic-sensitive pagerank. InProceedings of the 11th International Conference on World
Wide Web, WWW ’02, page 517–526, New York, NY , USA, 2002. Association for Computing Machinery. ISBN
1581134495. doi: 10.1145/511446.511513. URLhttps://doi.org/10.1145/511446.511513.
10

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
[15] Joseph John Rocchio Jr. Relevance feedback in information retrieval.The SMART retrieval system: experiments
in automatic document processing, 1971.
[16] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. From louvain to leiden: guaranteeing well-connected
communities.Scientific reports, 9(1):1–12, 2019.
[17] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang.
Lost in the middle: How language models use long contexts.Transactions of the Association for Computational
Linguistics, 12:157–173, 2024. doi: 10.1162/tacl_a_00638. URL https://aclanthology.org/2024.tacl-1.
9/.
[18] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically
inspired long-term memory for large language models. InThe Thirty-eighth Annual Conference on Neural
Information Processing Systems, 2024. URLhttps://openreview.net/forum?id=hkujvAPVsg.
[19] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When
not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In
Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors,Proceedings of the 61st Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, Toronto, Canada,
July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.546. URL https:
//aclanthology.org/2023.acl-long.546/.
[20] Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and Jinsong Su.
When to use graphs in rag: A comprehensive analysis for graph retrieval-augmented generation, 2025. URL
https://arxiv.org/abs/2506.05690.
[21] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christo-
pher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. InConference on
Empirical Methods in Natural Language Processing (EMNLP), 2018.
[22] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions
via single-hop question composition.Transactions of the Association for Computational Linguistics, 10:539–554,
2022. doi: 10.1162/tacl_a_00475. URLhttps://aclanthology.org/2022.tacl-1.31/.
[23] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: Non-parametric
continual learning for large language models, 2025. URLhttps://arxiv.org/abs/2502.14802.
[24] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. LightRAG: Simple and fast retrieval-augmented
generation. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors,
Findings of the Association for Computational Linguistics: EMNLP 2025, pages 10746–10761, Suzhou, China,
November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.
findings-emnlp.568. URLhttps://aclanthology.org/2025.findings-emnlp.568/.
[25] Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. Memorag:
Boosting long context processing with global memory-enhanced retrieval augmentation. InProceedings of the
ACM Web Conference 2025 (TheWebConf 2025), Sydney, Australia, 2025. ACM. URL https://arxiv.org/
abs/2409.05591.
[26] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie
Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. A survey on
llm-as-a-judge, 2025. URLhttps://arxiv.org/abs/2411.15594.
[27] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin,
Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron,
Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, Gaël Liu,
Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng,
Noveen Sachdeva, Benjamin Coleman, Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal,
Colin Cherry, Jan-Thorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi,
Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe Friesen, Abhanshu
Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa Saade, Alex Feng, Alexander
Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András György, André Susano Pinto, Anil Das, Ankur
Bapna, Antoine Miech, Antoine Yang, Antonia Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu,
Bobak Shahriari, Bryce Petrini, Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac
Brick, Daniel Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar
Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene Kharitonov, Frederick
Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna Klimczak-Pluci ´nska, Harman Singh, Harsh
Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne, Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie,
11

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Jetha Chan, Joe Stanton, John Wieting, Jonathan Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji,
Jyotinder Singh, Kat Black, Kathy Yu, Kevin Hui, Kiran V odrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine,
Marina Coelho, Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min
Ma, Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen Sachdeva,
Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton, Philipp Schmid, Pier Giuseppe
Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna, Renjie Wu, Renke Pan, Reza Rokni, Rob
Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome, Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy,
Shruti Sheth, Siim Põder, Sijal Bhatnagar, Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu,
Trevor Yacovone, Tyler Liechty, Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad
Kolesnikov, Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed,
Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo, Erica Moreira,
Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris Warkentin, Vahab Mirrokni, Evan
Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, Yossi Matias, D. Sculley, Slav Petrov,
Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet,
Elena Buchatskaya, Jean-Baptiste Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem,
Armand Joulin, Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard Hussenot. Gemma 3 technical
report, 2025. URLhttps://arxiv.org/abs/2503.19786.
[28] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez,
Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention.
InProceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.
[29] OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow,
Akila Welihinda, Alan Hayes, Alec Radford, Aleksander M ˛ adry, Alex Baker-Whitcomb, Alex Beutel, Alex
Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, Alex Paino, Alex Renzin, Alex Tachard Passos,
Alexander Kirillov, Alexi Christakis, Alexis Conneau, Ali Kamali, Allan Jabri, Allison Moyer, Allison Tam,
Amadou Crookes, Amin Tootoochian, Amin Tootoonchian, Ananya Kumar, Andrea Vallone, Andrej Karpathy,
Andrew Braunstein, Andrew Cann, Andrew Codispoti, Andrew Galu, Andrew Kondrich, Andrew Tulloch, Andrey
Mishchenko, Angela Baek, Angela Jiang, Antoine Pelisse, Antonia Woodford, Anuj Gosalia, Arka Dhar, Ashley
Pantuliano, Avi Nayak, Avital Oliver, Barret Zoph, Behrooz Ghorbani, Ben Leimberger, Ben Rossen, Ben
Sokolowsky, Ben Wang, Benjamin Zweig, Beth Hoover, Blake Samic, Bob McGrew, Bobby Spero, Bogo Giertler,
Bowen Cheng, Brad Lightcap, Brandon Walkin, Brendan Quinn, Brian Guarraci, Brian Hsu, Bright Kellogg,
Brydon Eastman, Camillo Lugaresi, Carroll Wainwright, Cary Bassin, Cary Hudson, Casey Chu, Chad Nelson,
Chak Li, Chan Jun Shern, Channing Conger, Charlotte Barette, Chelsea V oss, Chen Ding, Cheng Lu, Chong
Zhang, Chris Beaumont, Chris Hallacy, Chris Koch, Christian Gibson, Christina Kim, Christine Choi, Christine
McLeavey, Christopher Hesse, Claudia Fischer, Clemens Winter, Coley Czarnecki, Colin Jarvis, Colin Wei,
Constantin Koumouzelis, Dane Sherburn, Daniel Kappler, Daniel Levin, Daniel Levy, David Carr, David Farhi,
David Mely, David Robinson, David Sasaki, Denny Jin, Dev Valladares, Dimitris Tsipras, Doug Li, Duc Phong
Nguyen, Duncan Findlay, Edede Oiwoh, Edmund Wong, Ehsan Asdar, Elizabeth Proehl, Elizabeth Yang, Eric
Antonow, Eric Kramer, Eric Peterson, Eric Sigler, Eric Wallace, Eugene Brevdo, Evan Mays, Farzad Khorasani,
Felipe Petroski Such, Filippo Raso, Francis Zhang, Fred von Lohmann, Freddie Sulit, Gabriel Goh, Gene Oden,
Geoff Salmon, Giulio Starace, Greg Brockman, Hadi Salman, Haiming Bao, Haitang Hu, Hannah Wong, Haoyu
Wang, Heather Schmidt, Heather Whitney, Heewoo Jun, Hendrik Kirchner, Henrique Ponde de Oliveira Pinto,
Hongyu Ren, Huiwen Chang, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian O’Connell, Ian Osband,
Ian Silber, Ian Sohl, Ibrahim Okuyucu, Ikai Lan, Ilya Kostrikov, Ilya Sutskever, Ingmar Kanitscheider, Ishaan
Gulrajani, Jacob Coxon, Jacob Menick, Jakub Pachocki, James Aung, James Betker, James Crooks, James
Lennon, Jamie Kiros, Jan Leike, Jane Park, Jason Kwon, Jason Phang, Jason Teplitz, Jason Wei, Jason Wolfe,
Jay Chen, Jeff Harris, Jenia Varavva, Jessica Gan Lee, Jessica Shieh, Ji Lin, Jiahui Yu, Jiayi Weng, Jie Tang,
Jieqi Yu, Joanne Jang, Joaquin Quinonero Candela, Joe Beutler, Joe Landers, Joel Parish, Johannes Heidecke,
John Schulman, Jonathan Lachman, Jonathan McKay, Jonathan Uesato, Jonathan Ward, Jong Wook Kim, Joost
Huizinga, Jordan Sitkin, Jos Kraaijeveld, Josh Gross, Josh Kaplan, Josh Snyder, Joshua Achiam, Joy Jiao, Joyce
Lee, Juntang Zhuang, Justyn Harriman, Kai Fricke, Kai Hayashi, Karan Singhal, Katy Shi, Kavin Karthik, Kayla
Wood, Kendra Rimbach, Kenny Hsu, Kenny Nguyen, Keren Gu-Lemberg, Kevin Button, Kevin Liu, Kiel Howe,
Krithika Muthukumar, Kyle Luther, Lama Ahmad, Larry Kai, Lauren Itow, Lauren Workman, Leher Pathak, Leo
Chen, Li Jing, Lia Guy, Liam Fedus, Liang Zhou, Lien Mamitsuka, Lilian Weng, Lindsay McCallum, Lindsey
Held, Long Ouyang, Louis Feuvrier, Lu Zhang, Lukas Kondraciuk, Lukasz Kaiser, Luke Hewitt, Luke Metz, Lyric
Doshi, Mada Aflak, Maddie Simens, Madelaine Boyd, Madeleine Thompson, Marat Dukhan, Mark Chen, Mark
Gray, Mark Hudnall, Marvin Zhang, Marwan Aljubeh, Mateusz Litwin, Matthew Zeng, Max Johnson, Maya
Shetty, Mayank Gupta, Meghan Shah, Mehmet Yatbaz, Meng Jia Yang, Mengchao Zhong, Mia Glaese, Mianna
Chen, Michael Janner, Michael Lampe, Michael Petrov, Michael Wu, Michele Wang, Michelle Fradin, Michelle
12

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Pokrass, Miguel Castro, Miguel Oom Temudo de Castro, Mikhail Pavlov, Miles Brundage, Miles Wang, Minal
Khan, Mira Murati, Mo Bavarian, Molly Lin, Murat Yesildal, Nacho Soto, Natalia Gimelshein, Natalie Cone,
Natalie Staudacher, Natalie Summers, Natan LaFontaine, Neil Chowdhury, Nick Ryder, Nick Stathas, Nick Turley,
Nik Tezak, Niko Felix, Nithanth Kudige, Nitish Keskar, Noah Deutsch, Noel Bundick, Nora Puckett, Ofir Nachum,
Ola Okelola, Oleg Boiko, Oleg Murk, Oliver Jaffe, Olivia Watkins, Olivier Godement, Owen Campbell-Moore,
Patrick Chao, Paul McMillan, Pavel Belov, Peng Su, Peter Bak, Peter Bakkum, Peter Deng, Peter Dolan, Peter
Hoeschele, Peter Welinder, Phil Tillet, Philip Pronin, Philippe Tillet, Prafulla Dhariwal, Qiming Yuan, Rachel
Dias, Rachel Lim, Rahul Arora, Rajan Troll, Randall Lin, Rapha Gontijo Lopes, Raul Puri, Reah Miyara, Reimar
Leike, Renaud Gaubert, Reza Zamani, Ricky Wang, Rob Donnelly, Rob Honsby, Rocky Smith, Rohan Sahai,
Rohit Ramchandani, Romain Huet, Rory Carmichael, Rowan Zellers, Roy Chen, Ruby Chen, Ruslan Nigmatullin,
Ryan Cheu, Saachi Jain, Sam Altman, Sam Schoenholz, Sam Toizer, Samuel Miserendino, Sandhini Agarwal, Sara
Culver, Scott Ethersmith, Scott Gray, Sean Grove, Sean Metzger, Shamez Hermani, Shantanu Jain, Shengjia Zhao,
Sherwin Wu, Shino Jomoto, Shirong Wu, Shuaiqi, Xia, Sonia Phene, Spencer Papay, Srinivas Narayanan, Steve
Coffey, Steve Lee, Stewart Hall, Suchir Balaji, Tal Broda, Tal Stramer, Tao Xu, Tarun Gogineni, Taya Christianson,
Ted Sanders, Tejal Patwardhan, Thomas Cunninghman, Thomas Degry, Thomas Dimson, Thomas Raoux, Thomas
Shadwell, Tianhao Zheng, Todd Underwood, Todor Markov, Toki Sherbakov, Tom Rubin, Tom Stasi, Tomer
Kaftan, Tristan Heywood, Troy Peterson, Tyce Walters, Tyna Eloundou, Valerie Qi, Veit Moeller, Vinnie Monaco,
Vishal Kuo, Vlad Fomenko, Wayne Chang, Weiyi Zheng, Wenda Zhou, Wesam Manassra, Will Sheu, Wojciech
Zaremba, Yash Patil, Yilei Qian, Yongjik Kim, Youlong Cheng, Yu Zhang, Yuchen He, Yuchen Zhang, Yujia Jin,
Yunxing Dai, and Yury Malkov. Gpt-4o system card, 2024. URLhttps://arxiv.org/abs/2410.21276.
[30] Haoyu Han, Li Ma, Harry Shomer, Yu Wang, Yongjia Lei, Kai Guo, Zhigang Hua, Bo Long, Hui Liu, Charu C.
Aggarwal, and Jiliang Tang. Rag vs. graphrag: A systematic evaluation and key insights, 2025. URL https:
//arxiv.org/abs/2502.11371.
[31] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah Smith, and Mike Lewis. Measuring and narrowing
the compositionality gap in language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,Findings
of the Association for Computational Linguistics: EMNLP 2023, pages 5687–5711, Singapore, December
2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.378. URL https:
//aclanthology.org/2023.findings-emnlp.378/.
[32] Pruthvi Patel, Swaroop Mishra, Mihir Parmar, and Chitta Baral. Is a question decomposition unit all we
need? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors,Proceedings of the 2022 Conference
on Empirical Methods in Natural Language Processing, pages 4553–4569, Abu Dhabi, United Arab Emirates,
December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.302. URL
https://aclanthology.org/2022.emnlp-main.302/.
[33] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Enhancing retrieval-
augmented large language models with iterative retrieval-generation synergy. In Houda Bouamor, Juan Pino, and
Kalika Bali, editors,Findings of the Association for Computational Linguistics: EMNLP 2023, pages 9248–9274,
Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.
620. URLhttps://aclanthology.org/2023.findings-emnlp.620/.
[34] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu.
Dense X retrieval: What retrieval granularity should we use? In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung
Chen, editors,Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing,
pages 15159–15177, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi:
10.18653/v1/2024.emnlp-main.845. URLhttps://aclanthology.org/2024.emnlp-main.845/.
[35] Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and Greg Durrett. WiCE: Real-world entailment for claims
in Wikipedia. In Houda Bouamor, Juan Pino, and Kalika Bali, editors,Proceedings of the 2023 Conference on
Empirical Methods in Natural Language Processing, pages 7561–7583, Singapore, December 2023. Association
for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.470. URL https://aclanthology.org/
2023.emnlp-main.470/.
[36] Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, and Cheng Yang.
Pathrag: Pruning graph-based retrieval augmented generation with relational paths, 2025. URL https://arxiv.
org/abs/2502.14902.
[37] Jingjin Wang and Jiawei Han. PropRAG: Guiding retrieval with beam search over proposition paths. In Christos
Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors,Proceedings of the 2025
Conference on Empirical Methods in Natural Language Processing, pages 6223–6238, Suzhou, China, November
2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.
317. URLhttps://aclanthology.org/2025.emnlp-main.317/.
13

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
[38] Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin
Kuang, Meina Song, Yifan Zhu, and Anh Tuan Luu. HypergraphRAG: Retrieval-augmented generation via
hypergraph-structured knowledge representation. InThe Thirty-ninth Annual Conference on Neural Information
Processing Systems, 2025. URLhttps://openreview.net/forum?id=ravS5h8MNg.
[39] Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, and Hui Xiong. Plan-on-graph: Self-correcting
adaptive planning of large language model on knowledge graphs. InThe Thirty-eighth Annual Conference on
Neural Information Processing Systems, 2024. URLhttps://openreview.net/forum?id=CwCUEr6wO5.
[40] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. Think-
on-graph 2.0: Deep and faithful large language model reasoning with knowledge-guided retrieval augmented
generation. InThe Thirteenth International Conference on Learning Representations, 2025. URL https:
//openreview.net/forum?id=oFBu7qaZpS.
[41] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. RAPTOR:
Recursive abstractive processing for tree-organized retrieval. InThe Twelfth International Conference on Learning
Representations, 2024. URLhttps://openreview.net/forum?id=GN921JHCRw.
[42] Fangyuan Xu, Junyi Jessy Li, and Eunsol Choi. How do we answer complex questions: Discourse structure of
long-form answers. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors,Proceedings of the
60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3556–3572,
Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.249.
URLhttps://aclanthology.org/2022.acl-long.249/.
[43] Konstantinos Papakostas and Irene Papadopoulou. Model analysis & evaluation for ambiguous question answering.
In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors,Findings of the Association for Computational
Linguistics: ACL 2023, pages 4570–4580, Toronto, Canada, July 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.findings-acl.279. URLhttps://aclanthology.org/2023.findings-acl.279/.
A Knowledge Base Extraction
Figure 5: Knowledge base extraction process. Propositions and entities are extracted from input passages and populate
the graph. Entity embeddings (used for synonym resolution) are omitted for clarity reasons.
An illustration of the knowledge base extraction is presented in Figure 5. The prompt strategy used for entities and
propositions extraction is presented in Figure 6. For synonym reconciliation, given an encoder h, two entities eande′
are considered synonymous ifcosine 
h(e), h(e′)
≥θ(default0.9).
14

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
We also report statistics for the knowledge base construction (indexing) stage of our approach. Table 2 summarizes the
resulting graph sizes, including counts of passages, propositions, and entities, as well as the total number of edges. The
overall indexing cost, using the MusiQue corpus as a reference, is also provided and compared against baseline systems
in Table 3.
Figure 6: Prompts for Named Entity Recognition and Propositions Extraction. First, named entities are extracted from
the passage ( Prompt-NER ). Then, using the previously extracted entities and the original passage, propositions are
extracted withPrompt-Propositions. Propositions are returned with their associated entities.
MusiQue HotPotQA PopQA Agriculture CS Legal GB-Medical GB-Novel
# passages 11,704 9,959 9,101 9,055 7,337 16,169 883 2,400
# propositions 83,247 77,409 73023 9,2840 58,322 84,134 8,442 37,868
# entities 82,721 82,909 79,783 62,341 31,108 35,732 3,955 27,071
# edges 350,436 333,799 309,812 613,688 320,440 786,049 49,863 14,9731
Table 2: Number of nodes (passages, propositions and entities) and edges in the graph associated with each corpora
used in our experiments.
15

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
ToPG LightRAG†GraphRAG†HippoRAG 2†
Prompts 70.5M 68.5M 115.5M 9.2M
Completion 11.9M 12.3M 36.1M 3.0M
Table 3: Token usage comparison (prompt and completion) at indexing time for the baselines on the MuSiQue corpus
(11,656 passages for 1.3M tokens).†Values reported from Gutiérrez et al. [23].
Figure 7: Illustrative example of the construction and realization of the Query Aware Transitions matrix Mon an
hypothetical subgraph G∗.P1shows the subgraph G∗with 4 landmark nodes A,B,CandD.P2andP3describe the
two components of M:TsandTn. The width of the arrows is proportional to the transition probability between nodes.
P4illustrates the ranking obtained from the stationary distribution πof probabilities with the PPR using M. The larger
the node the greater the final probability and rank.
B Supplementary Methods
B.1 Query Aware Transition: an illustrative example
Figure 7 provides an illustrative and intuitive representation behind the construction of the query-aware transition matrix
M. The panel P1shows a hypothetical input subgraph G∗, with four annotated nodes ( A,B,CandD) that serve as
reference points for the next panels. P2describes the proposition-projected graph associated with Ts, as apropositions
to propositionsgraph. For instance, nodes around Dare all connected to the same entity in P1, creating a clique in
the resulting projected graph in P2. The width of the arrows is proportional to the transition probability between two
nodes, according to their connectivity (through entities and passages) in the original graph. P3describes the second
component of M:Tn. In this graph, the attraction of a node relative to its neighbors, indicated by the width of the arrow,
is proportional to its similarity to the query (default: cosine similarity). Nodes A,B,CandDbecome attractive as their
embeddings are similar to the query compared to other nodes (e.g., in the neighborhood of D). InP4, we exemplify
the results of running a PPR using Aas the starting node and following the built Mtransition matrix, balancing the
transitions between TsandTn. In this example, proposition nodes like DorBwould be among the top-ranked nodes.
B.2Localmode
Algorithm 1 describes the query process inLocalmode. An illustrative example is also provided in Figure 8.
B.3Globalmode: Compute queries
The approach for query vector refinement is inspired by the Rocchio algorithm [ 15], which applies relevance feedback
to refine a query by weighting vectors of relevant and non-relevant documents.
16

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Algorithm 1Local mode with Suggestion-Selection cycles
Require:GraphG= (V, E), initial queryq start, parametermax-iter
1:InitializeQ← {q start}
2:s loc← ∅
3:S 0←SuggestNaive(q start,∅)▷Initial seeding in the graph
4:s pool←Select(q start, S0)
5:s pool-new ←s pool
6:whileiteration<max-iterdo
7:for allq∈Qdo▷Gather propositions for all questions ins pool-new
8:S new←SuggestLocal(q, s pool)▷Suggestions seeded ons pooland biased towardq
9:s new←Select(q, S new)
10:s pool-new ←s pool-new ∪s new
11:end for
12:s loc←s loc∪s pool-new ▷Completes loc
13:s pool←s pool-new
14:s pool-new ← ∅
15:ifPROMPT Eval(qstart, sloc)returns an answerthen
16:returnanswer
17:else
18:Q←PROMPT NextQ(qstart, sloc)▷Evaluate withs loc
19:end if
20:iteration+ +
21:end while
22:returnfailure to determine answer
We adapt this principle to compute a refined query vector qifor each newly collected proposition node ui∈s pool. The
refinement process combines: the directions that lead to uiin the previous walks, the semantic representation of ui
itself, and, the directions that were pruned.
LetCidenote the set of partitions where uiwas identified. For a given partition c∈ C i, letq(c)
k∗be the query vector of
the walker that most likely reached ui, where k∗= arg max kπ(c)
k(ui). Furthermore, let ¯S(c)
new=S(c)
new\s(c)
newbe the set
of candidate nodes that were pruned by theSelectprocedure in that partition and ¯h ¯S(c)
new
their averaged embedding.
The query vector foru iis then given by:
qi=α qo
i+βq+
i−γq−
iwhere,
qo
i=1
|Ci|X
c∈Ciq(c)
k∗,
q+
i=h(u i),
q−
i=1
|Ci|X
c∈Ci¯h ¯S(c)
new(11)
The coefficients α, β,andγ are positive weights that modulate the influence of the initial, positive, and negative
feedback components, respectively.
B.4Globalmode: Collecting anchor nodes
Algorithm 2 describes the iterative exploration and collection process that builds the set of anchor propositions, before
community extraction inGlobalmode.
B.5Global mode: Greedy community selection with budget
A complete description of the greedy procedure is presented in Algorithm 3. Candidate communities c∈C are
pre-fitlered by size (number of nodes):10≤ |c| ≤150and the budgetBis fixed to8000in the experiments.
17

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Figure 8: Illustrative example of theLocalmode.
C Abstracts Questions: Protocol and Examples
We follow the procedure described by LightRAG authors3. To emulate a large variety of potential queries, the LLM
is first instructed to generate 5 potential users with 5 related tasks for each, given a summary of the corpus. For each
task, 5 questions are generated that require a high-level understanding of the corpus. Below in Figure 9 is a subset of
questions generated from the Agriculture corpus containing textbooks on beekeeping.
3https://github.com/HKUDS/LightRAG
18

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Algorithm 2Anchors selection via Iterative Suggestion-Selection (Globalmode)
Require:GraphG, queryq start, breadthm,max-iter, minimum factsmin_facts
1:Q←PROMPT decompose (qstart, m)▷Decompose initial query intomsub-queries
2:s glb← ∅
3:s pool← ∅
4:iteration←0
5:foreachq∈Qdo▷Initializes poolwith themquestions
6:S 0←SuggestNaive(q,∅)
7:s pool←s pool∪Select(q, S 0)
8:end for
9:s glb←s pool
10:while|s glb|<min_factsanditeration<max-iterdo
11:s pool-new ← ∅
12:q pool←ComputeQueries(s pool)▷Compute queries for the new selected nodes
13:foreach partitions partinPartition(s pool, m)do▷Each partition is explored independently
14:S new←Suggest Global (qpart, G, s part)
15:s new←Select(q start, Snew)
16:s pool-new ←s pool-new ∪s new
17:end for
18:s glb←s glb∪s pool-new ▷Complete the globals glband prepare the next seeds (s pool)
19:s pool←s pool-new
20:iteration+ +
21:end while
22:returns glb ▷Final pool of collected propositions (anchors)
Algorithm 3Greedy Budgeted Communities Extraction
Require:S: anchor nodes,B: budget limit,C: candidate communities
1:b←0,S′← ∅,C′← ∅▷budget used, nodes covered, communities selected
2:whileS′̸=Sandb < Bdo
3:c∗←arg max c∈C|nodes(c)\S′|
size(c)▷best coverage / size ratio
4:C′←C′∪ {c∗}▷Update candidates, nodes covered and budget
5:S′←S′∪nodes(c∗)
6:b←b+size(c∗)
7:end while
8:returnC′
Figure 9: Example of abstract questions generated for the Agriculture Corpora
19

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
D Hyperparameters Evaluation
Figure 10 reports the performance of different combinations of PPR damping factors and λvalues on the MusiQue
dataset, using the Local mode with max-iter =3. Across both damping settings ( 0.5and0.85), we observe only minor
variation in EM and F1, indicating that the restart probability has limited influence on retrieval quality in these settings.
In contrast, λ, which controls the relative contribution of semantic transitions ( Tn) versus structural transitions ( Ts), has
a pronounced impact. Performance degrades as λgoes to 1and the semantic component is canceled. This highlights
the importance ofT nin suppressing semantically irrelevant paths when neighbors are unrelated to the query.
These observations motivated our choice of default hyperparameters: a damping factor of 0.85 and a balanced λ= 0.5 .
Figure 10: Impact of the damping factor and λparameters on model performance measured by F1-score and Exact
Match (EM) on MusiQue.
E Dataset and Baseline Details
E.1 Dataset details
Simple and Complex QA datasetsSubsets of MusiQue ( CC-By-4.0 License), HotPotQA ( CC-By-4.0 License)
and PopQA ( MITLicense) have been extracted from the repository provided by Gutiérrez et al. [23]4. Each corpus is
composed of 1,000 questions that require retrieval over one or several passages originating from Wikipedia. Additional
details can also be found in Table 2.
GraphRAG-Benchmark CorporaWe evaluate on the two corpora of GraphRAG-Benchmark5. The Medical
corpus (NCCN Guidelines) integrates data from the National Comprehensive Cancer Network (NCCN) clinical
guidelines, covering diagnosis criteria, treatment protocols, and drug interactions. Additionally, the Novel corpus
(Project Gutenberg) is a curated collection of pre-20th-century novels from the Project Gutenberg library. These texts
exhibit complex narrative and temporal relationships. Finally, we use the same Answer Accuracy metric as used in the
benchmark (see Xiang et al. [20]), which combines semantic similarity with statement-level fact checking.
UltraDomain CorporaAbstract QA uses three specialized corpora from the UltraDomain corpus6, with sizes
specified in Table 4.
Abstract QA (LLM-as-a-Judge)Following Guo et al. [24], abstract queries are evaluated using LLM-as-a-Judge
across four criteria:
•Comprehensiveness:How much detail does the answer provide to cover all aspects and details of the question?
•Diversity:How varied and rich is the answer in providing different perspectives and insights on the question?
4https://huggingface.co/datasets/osunlp/HippoRAG_2
5MITLicense:https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
6Apache 2.0 License:https://huggingface.co/datasets/TommyChien/UltraDomain
20

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Corpus Content Focus Size (Tokens)
Agriculture Beekeeping, agricultural policy, farmers, diseases and pests 1.9M
Computer Science (CS) Machine learning, data processing 2.0M
Legal Corporate finance, regulatory compliance, finance 4.7M
Table 4: Details and size metrics for the UltraDomain corpus used in Abstract QA evaluation.
•Empowerment:How well does the answer help the reader understand and make informed judgments about
the topic?
•Overall:The final aggregate score combining the three criteria.
We used Gemma-3-27B [27] as the LLM during the evaluation. Gemma-3-27B is licensed under theGemma Terms of
Use7. Details on the prompts can be found on the GitHub repository athttps://github.com/idiap/ToPG.
E.2 Baseline details
The configurations for all baseline models (HippoRAG 2, LightRAG, GraphRAG, and ToPG) are detailed in Table 5.
On the granularity level, HippoRAG 2, LightRAG, GraphRAG operate with passage-level context. LightRAG and
GraphRAG additionally augment context with auxiliary KG elements (entities/relations).
For a fair comparison on the Abstract QA task, a domain-specific set of topic-related entities was defined during the
indexing stage. These entities, used by GraphRAG and LightRAG, are grouped by domain:
•Agriculture:organization, geo, event, agriculture, economic, environment.
•Computer Science (CS):organization, technology, software, metric, mathematics, hardware, com-
puter_science, networking.
•Legal:organization, geo, legal, regulation, financial, asset, risk, law, financial_instrument.
We empirically found that for the Simple/Complex QA tasks, both LightRAG and GraphRAG performed optimally
using their local search mode with a smaller top_k= 5 compared to standard default settings ( 60for LightRAG and 10
for GraphRAG). We hypothesized that the resulting large context ( ≥8k tokens) is detrimental for accurate factual QA
with the used LLM.
To establish a strong baseline for comparison against our proposed strategy, the global mode of GraphRAG was
configured with community_level= 2 . While this choice significantly increased the granularity of community search,
it also comes at a substantial computational cost (consuming on average 79kand2.1M tokens for Completion and
Prompts, per query.
F Example Abstract QA and evaluation
An example of evaluation with LLM-as-a-judge on the Agriculture corpora considering the 4 criteria (Comprehensive-
ness, Diversity, Empowerment and Overall) is provided in Table 6.
7https://ai.google.dev/gemma/terms
21

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
Model Task Parameter Value
HippoRAG 2 Simple/Complex QAtop_k5
GraphRAG Simple/Complex QAmodelocal
max_context_tokens8000
text_unit_prop0.5
community_prop0.25
top_k_mapped_entities5
top_k_relationships5
Abstract QAmodelocal
community_level2
use_community_summaryTrue
min_community_rank0
max_tokens12000
LightRAGSimple/Complex QAmodelocal
top_k5
chunk_top_k5
max_total_tokens6000
Abstract QAmodehybrid
TOP_K40
CHUNK_TOP_K10
MAX_TOTAL_TOKENS32000
ToPG Simple/Complex QAλ0.5
dampingd0.85
cosine_thresholdθ0.4
subgraphG∗max sizel500
temperatureτ0.1
top_k20
Abstract QAtop_k20
breadth m10
node budget B8000
min_community_size10
max_community_size150
max_tokens_report8000
passage_token_limit500
max_tokens_community_chunks8000
Table 5: Parameter settings for the RAG baselines (HippoRAG 2, LightRAG, GraphRAG, and ToPG), across the
evaluation tasks (Simple QA, Complex QA, and Abstract QA). Additional configuration parameters defined for ToPG in-
clude: max_tokens_report (the maximum token limit available for the final response context), passage_token_limit
sets the limit for tokens used from community passages, within the max_tokens_community_chunks available for
intermediary answers. For a comprehensive description of other baseline-specific parameters, please refer to their
respective documentation.
22

A Navigational Approach for Comprehensive RAG via Traversal over Proposition GraphsA PREPRINT
QueryHow does soil health influence overall farm productivity?
GraphRAG The Central Role of Soil Health in Farm Productivity
Soil health is fundamentally ...
Key Components of Healthy SoilSeveral key components contribute to overall soil
health.Organic matteris paramount, improving soil structure....
Agricultural Practices & Soil HealthNumerous agricultural practices directly impact
soil health.No-till farmingsignificantly improves soil health ...
The Impact of Soil Health on Yield & SustainabilityMaintaining soil health is
crucial for maximizing agricultural output...
Soil Health & Economic ViabilitySoil health is intrinsically linked to agricultural
economics...
In conclusion, soil health is not merely a component ...
ToPG Soil Health and Farm Productivity: A Comprehensive Report
Improving soil health is increasingly recognized as central ...
The Central Role of Organic MatterOrganic matter is paramount to soil health and
fertility...
Practices for Enhancing Soil HealthSeveral key practices contribute to improved
soil health...
Soil Biology and Nutrient AvailabilityHealthy soil directly impacts nutrient avail-
ability for plants...
Water Management and Soil StructureSoil health significantly impacts water
retention and availability...
Long-Term Sustainability and ResilienceImproving soil health is not a complex
process and can lead to cost-effective ways to feed the world and restore land...
In conclusion, prioritizing soil health is essential for maximizing farm productivity,
ensuring long-term sustainability, and building resilience.
LLM Decision: Compre-
hensivenessWinner: Answer 2
Explanation:Both answers cover the core components of soil health and its impact
on farm productivity. [...] Answer 2 also acknowledges challenges and considerations,
like the adjustment period for transitioning to new practices, which adds to its com-
pleteness.
LLM Decision: Diver-
sityWinner: Answer 2
Explanation:While both answers touch on various aspects of soil health, Answer 2
demonstrates greater diversity in its perspectives. [...] Answer 2 also acknowledges
the potential downsides of conventional practices, offering a more balanced view.
LLM Decision: Empow-
ermentWinner: Answer 2
Explanation:Answer 2 is more empowering because it provides concrete examples
and quantifiable data that allow the reader to understand the *magnitude* of the
benefits associated with improved soil health. [...] It states that soil health is important,
but doesn’t provide the same level of evidence to support that claim in a way that
empowers the reader to act.
LLM Decision: Overall
WinnerWinner: Answer 2
Explanation:Answer 2 is the overall winner because it excels in all three criteria.
[...] While Answer 1 provides a solid overview of the topic, Answer 2 goes further
in explaining the *why* and *how* of soil health, making it a more valuable and
insightful resource for anyone seeking to understand this critical aspect of agricultural
productivity.
Table 6: Example of an evaluation of Abstract QA using LLM-as-a-judge, comparing answers from GraphRAG and
ToPG. The model used isGemma-3-27B.
23