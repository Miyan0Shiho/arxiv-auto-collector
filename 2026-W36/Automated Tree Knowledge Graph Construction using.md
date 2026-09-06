# Automated Tree Knowledge Graph Construction using Ontology Expansion and Retrieval from Vietnamese History Textbooks

**Authors**: Ket Doan Nguyen, Minh N. H. Nguyen

**Published**: 2026-09-01 05:52:50

**PDF URL**: [https://arxiv.org/pdf/2609.00763v1](https://arxiv.org/pdf/2609.00763v1)

## Abstract
Hierarchical Knowledge graph (KG)-based retrieval augmented generation (RAG) has emerged as a powerful approach for supporting large language models with structured knowledge. However, there are primary challenges: (i) the lack of methods for automatic KG construction using ontology expansion for low-resource languages such as Vietnamese, (ii) the absence of systematic evaluation for knowledge retrieval strategies leveraging the hierarchical structures. In this paper, we propose an end-to-end pipeline for KG construction and retrieval strategies evaluation. In the KG construction, we employ a three-phase hybrid relation extraction pipeline: intra-batch deduplication via Union-Find, approximate cross-batch search, and LLM extraction with a centroid filter that reduces prompts combined with a five-step dual-LLM validator to prevent bloated ontology. A two-tier architecture consists of unmergeable structural nodes to preserve the document structure and mergeable content nodes. The retrieval evaluation consists of three graph traversal strategies: Top-Down, Horizontal, and Bottom-Up, which are evaluated on a synthetically generated benchmark of 1,210 Vietnamese queries from 109 subgraphs, categorized by five query directions. In this paper, we construct the tree knowledge graph from Vietnamese high school History textbooks (nearly 400 pages) to produce 750 nodes and 4,341 semantic edges with controlled ontology growth from 40 to 41 types. Among experimental graph traversal strategies, the Top-Down strategy with structure surpasses the vector baseline by 4.7 percentage points in NDCG@10. As a result, tree-structural information provides valuable information beyond flat cosine similarity but degrades performance when the query does not require structural context.

## Full Text


<!-- PDF content starts -->

Automated Tree Knowledge Graph Construction
using Ontology Expansion and Retrieval from
Vietnamese History Textbooks
Ket Doan Nguyen1[0009−0005−6698−9629]and Minh N. H. Nguyen1
The University of Danang, Vietnam-Korea University of Information and
Communication Technology, Danang, Vietnam
{nkdoan,nhnminh}@vku.udn.vn
Abstract.Hierarchical Knowledge graph (KG)-based retrieval aug-
mented generation (RAG) has emerged as a powerful approach for
supporting large language models with structured knowledge. However,
there are primary challenges: (i) the lack of methods for automatic KG
construction using ontology expansion for low-resource languages such as
Vietnamese, (ii) the absence of systematic evaluation for knowledge re-
trieval strategies leveraging the hierarchical structures. In this paper, we
propose an end-to-end pipeline forKG constructionand retrieval strate-
gies evaluation. In theKG construction, we employ a three-phase hybrid
relation extraction pipeline: intra-batch deduplication via Union-Find,
approximate cross-batch search, and LLM extraction with a centroid fil-
ter that reduces prompts combined with a five-step dual-LLM validator
to prevent bloated ontology. A two-tier architecture consists of unmerge-
able structural nodes to preserve the document structure and mergeable
content nodes. Theretrieval evaluationconsists of three graph traver-
sal strategies: Top-Down, Horizontal, and Bottom-Up, which are evalu-
atedonasyntheticallygeneratedbenchmarkof1,210Vietnamesequeries
from 109 subgraphs, categorized by five query directions. In this paper,
we construct the tree knowledge graph from Vietnamese high school
History textbooks (nearly 400 pages) to produce 750 nodes and 4,341
semantic edges with controlled ontology growth from 40 to 41 types.
Among experimental graph traversal strategies, the Top-Down strategy
with structure surpasses the vector baseline by 4.7 percentage points in
NDCG@10. As a result, tree-structural information provides valuable in-
formation beyond flat cosine similarity but degrades performance when
the query does not require structural context.
Keywords:Hierarchical knowledge graph·Ontology expansion·RAG
·Relation extraction·Knowledge retrieval·History
1 Introduction
Vietnam’s National Digital Transformation Program lists Education as one of
eight priority sectors and calls for intelligent systems in teaching and learning.
Among school subjects, History presents a distinct challenge. History lessons
arXiv:2609.00763v1  [cs.AI]  1 Sep 2026

2 N. K. Doan and N. H. N. Minh
require students to trace connections across events, people, dates, and causal
chains. Hence, reasoning over structured knowledge rather than memorizing iso-
lated facts is mandatory for students to learn history more efficiently. Connected
reasoning demands more than keyword matching against flat text and requires a
knowledge representation for relations between entities. In Vietnamese language
processing, there are existing emerging pretrained models such as PhoBERT [17]
that can perform well on multiple language tasks. To the best of our knowledge,
no existing framework has been developed for constructing knowledge graphs
from Vietnamese educational content, introducing an important gap for future
research.
Knowledge graphs (KGs) store structured knowledge for supporting multi-
hop reasoning and cross-graph information combination [13]. Incorporating KGs
with retrieval-augmented generation (RAG) [14], question-answering systems
can follow chains of relations rather than depending on text similarity in
GraphRAG [19,8]. Recent methods [10,16] propose single traversal strategies;
none compare multiple directions given hierarchical structures of knowledge. On
the KG construction stage, LLMs perform zero-shot relation extraction [24] and
speed up ontology creation [23], but the ontology could grow too large with
overlapping entities and relations [12]. Our main contributions are as follows:
–A hierarchical KG construction pipeline with controlled ontology expansion
is constructed using a two-layer ontology with 26 general, 14 history-specific
types, an embedding centroid filter that reduces the number of prompts by
50%, and a five-step dual-LLM validator that prevents 87.5% of duplicate
relations.
–An automatic benchmark generation pipeline from KGs, including a three-
phase subgraph selection process such as SEED, ENRICH, BOUND gener-
ates1,210Vietnamesequeriesfrom109subgraphswithhierarchicalrelevance
labels (i.e., 0/1/2), categorized by five query directions.
–Threegraphtraversalstrategies,suchasTop-Down,Horizontal,andBottom-
Up, are evaluated using a hierarchical structure along different directions.
–A multi-dimensional evaluation framework including 12 methods at 4 cut-
off thresholdsK, analyzes performance across 5 query directions, and the
influences of the structural relations.
2 Related Work
RAG and GraphRAG:The RAG systems were designed with a generative
retriever, and the LLM can produce its answers in retrieved documents [14].
Adding a knowledge graph to the pipeline, GraphRAG [19,8] enables the system
to follow relational paths rather than rely on keyword overlap alone. Exist-
ing methods traverse the graph in different ways: ReMindRAG [10] allows the
LLM to decide which edges to follow, GFM-RAG [16] trains a graph foundation
model for the task on hierarchical structures. RAPTOR [22] builds a retrieval
tree through recursive abstractive summarization. Unlike existing works, our ap-
proach keeps the original document hierarchy and evaluates multiple traversal
directions rather than a single bottom-up clustering strategy.

Automated Tree KG Construction with Ontology Expansion 3
Knowledge Graph Construction from Documents:Recent works in
[21,6,15] use LLMs to extract knowledge graphs from unstructured documents.
Doc-KG [21] and AutoMathKG [6] both combine embedding-based relation fil-
tering with LLM extraction. LLMs can perform zero-shot relation extraction [24]
and speed up ontology design [23], the expansion quickly produces duplicate
types [12]. The work in [15] uses multi-agent LLMs to enrich existing KGs,
though enrichment differs from construction. Our proposed system will combine
controlled expansion with dual-LLM validation to prevent the ontology from
growing too large. In the education domain, Ain et al. [2,3] build a KG pipeline
for educational content and compare top-down with bottom-up construction.
Abu-Rasheed et al. [1] uses LLMs to fill in missing relations in curriculum KGs.
To the best of our knowledge, there is a lack of KGs built from Vietnamese
textbooks.
Benchmarks for KG-based Retrieval:CRAG is the most popular RAG
benchmark [25], which has 4,409 question-answer pairs and evaluates flat text
and entity-level question answering. Graded relevance labels, where a node can
be partially relevant rather than simply relevant or not, are absent from existing
benchmarks.
Therefore, three open challenges remain and will be addressed in our paper:
(i) building a KG with controlled ontology expansion and dual-LLM validation
for a low-resource language, (ii) designing a retrieval benchmark with graded
relevance labels and query direction categories, and (iii) comparing traversal
strategies on a hierarchical structure for Vietnamese history textbooks.
3 Tree Knowledge Graph Construction using Ontology
Expansion
Fig. 1.Four-stage pipeline for hierarchical KG construction.
Figure 1 illustrates an overview of the proposed system, which has two parts:
Part (A) builds the KG through four stages, and Part (B) evaluates retrieval
using three graph traversal strategies. The four construction stages are:

4 N. K. Doan and N. H. N. Minh
–Stage 1 - Document parsing: We convert PDF files into ann-ary hierarchical
tree using a deep learning-based layout analysis tool [26] and produce two
kinds of nodes:structural nodes(e.g., books, chapters, lessons, sections) and
content nodes(e.g., paragraphs, tables). The content nodes are initially split
intosmallpieces;LLM-basedmergingofsegmentswithinthesamestructural
node is therefore necessary to preserve context. In our design, the structural
nodes will not be merged to keep the original structure of documents, but
the content nodes could be merged to reduce duplication.
–Stage 2 - Knowledge tree enrichment: We perform post-order traversal to
enhance each structural node with extra context based on the summary
from its child content nodes.
–Stage 3 - Embedding generation:Vectorrepresentationsarecreatedfornodes.
–Stage 4 - Relation extraction:Wedevelopedapipelineforrelationextraction,
including embedding filtering, LLM classification, and ontology.
The resulting KG is stored in ArangoDB [5] (multi-model: document +
graph), which serves both benchmark generation (Section 5) and retrieval (Sec-
tion 4).
Two-layer ontology:Theontologyincludes40initialrelationtypesorganizedinto
two layers. The general layer contains 26 relation types that are reusable across
all domains. The history-specific layer contains 14 relation types covering causal,
temporal, historical figure, and other relations. The two-layer design allows the
domain-specific layer to be replaced when changing to a different domain, while
keeping both the general layer and the extraction pipeline. Each relation type
is designed with a short description, keywords, node type constraints, and an
embedding vector. The tier partitioning separates Tier 1 for structural nodes
from Tier 2 for content nodes.
Stage 4 - Relation Extraction with three phases:
Phase 1 - Pre-filtering:We first calculate the cosine similarity scoresfor the
node pair. The pairs having scores above the thresholdτ mergewill be merged
automatically or linked with an equivalence edge. Those falling betweenτ reland
τmergego through the LLM for relation classification in Phase 2, and pairs below
τrelstay separate. In this way, LLM focused on the middle range cosine similarity
only.
Phase 2 - Ontology filtering:Given the node pair(A, B), the embedding centroid
iscAB=1
2(eA+eB). The relevance score for relation typeRis:
score(R) =c AB·eR (1)
whereeare the embedding vectors of nodes or relations. To save the tokens
and processing time, we employ the max-pooling strategy to identify the most
relevant relation types of pairspin batch such asmax p∈batch (cp·eR). In this
way, our process can select the top 15 most relevant relation types from 40 types
and reduce the prompt size by nearly 50%.

Automated Tree KG Construction with Ontology Expansion 5
Phase 3 - Dual LLM relation extraction:A weak LLM model receives batches
of node pairs and 15 filtered relation candidates to extract relations. For each
pair, the model either selects a type from the candidates, declaresno relation,
or proposes anovel relation. In our experiments, the weak LLM proposed many
novel relation types per batch; however, nearly 95% relations were almost simi-
lar to the existing 40 relations (e.g.,LED_TOvs.CAUSED). Without filtering, the
ontology would introduce many new relations having almost similar meaning
and cause thebloated ontologyproblem. Hence, the novel relations need to be
double-checked with the existing 40 relation types and filtered based on embed-
ding similarity. Thereafter, a strong LLM model is used to validate the proposed
novel relations. First, the proposed relation of node pairs from the weak LLM
will be either accepted or introduced a better relation, including the name, de-
scription, relation family, and constraints using the strong LLM model. Then
the novel relations will be double-checked with the existing relations using the
cosine similarity score with the thresholdτ onto. For example,CONTRIBUTED_TO
andPLAYED_ROLE_INthat share the same meaning under different names.
4 Graph Retrieval-Augmented Generation Strategies
Fig. 2.Three graph traversal strategies: Top-Down traverses from structural nodes
down to content, Horizontal traverses via semantic relations, Bottom-Up aggregates
from content up to parent nodes.
In this paper, we experimented with three Graph-RAG strategies for the
constructed tree knowledge, as illustrated in Figure 2. We also propose ranking
scores based on the hierarchical position, parent-child relationships, and the
semantic score between nodes.

6 N. K. Doan and N. H. N. Minh
Reciprocal Rank Fusion (RRF) [9] was applied to build the scores for strate-
gies as follows:
RRF(d) =X
s∈S1
k+rank s(d)(2)
wherekis a smoothing constant, used to merge multiple ranking criteria.
Top-Down Strategy:The strategy first identifies the best-matching sections
usingasemanticsummaryofeachstructuralnodebasedonembeddingsimilarity.
Thenittraversesdownwardtocollectallcontentnodesinthesubtreeofsections.
Three ranking scores are evaluated:E1auses a cosine similarity score,E1b
uses the RRF score givenseed_score, andE1cextends E1b with MMR [7] to
enhance the diversity.
Horizontal Strategy:Many queries require connecting scattered information
across different sections. For example, a cause might appear in one section while
its consequence events appear in another. The Horizontal strategy addresses the
problem by traversing semantic relation edges between branches of the tree. The
process starts with a vector search to select seed nodes and the corresponding
seed_scorebased on the cosine similarity score. Then, it filters to select the
relevant relation types and traverses one hop to reach cross-branch nodes (i.e.,
extended nodes). Among all of the selected nodes, the strategy performs the
following three rankings to select the top-K relevant nodes. First,E2auses only
cosine similarity between the content nodes, including the seed and extended
nodes. Second,E2buses the RRF score givenseed_score, the cosine similarity
of the extended content nodes. Third,E2cis a restricted variant when keeping
only the top-3 relation types.
Bottom-Up Strategy:Given the query, when the content nodes have a high
similarity score, the shared parent structural node is likely relevant. The ap-
proach is especially useful when the parent summary is too general to match the
query directly. The main factor for ranking is the threshold-based child aggre-
gation. We denoteC(p)as the set of child nodes belonging to the parent node
p, ands c=sim(q,e c)is the cosine similarity between the query and child node
c. The thresholdτ pis computed as the median of all child scores belonging to
the parent nodep. Only children whose score is greater than or equal toτ pare
selected. The process starts with a vector search to retrieve candidate content
nodes, then follows inbound edges to identify parent nodes using the following
three variants. First,E3auses cosine similarity based on the semantic summary
to select top-K parent nodes. Second,E3bapplies the aggregated score of the
parent nodepbased child scores:
agg_score(p) =P
c∈C(p)sc
|C(p)|×ln
1 +{c∈C(p) :s c≥τp}
(3)
The first term captures the average cosine quality of all children under parentp.
The second term introduces the logarithmic scaling factor based on the number

Automated Tree KG Construction with Ontology Expansion 7
of relevant children, thereby avoiding the explosion of dense parents. Third,E3c
utilizes RRF using the parent cosine and agg_score(p).
5 Benchmark Generation Pipeline
To validate three Graph-RAG strategies, we construct the benchmark using the
proposed pipeline in Figure 3. In this way, we can convert the constructed KG
into a benchmark dataset with corresponding graded relevance labels for five
query directions.
Fig. 3.Benchmark generation pipeline: SEED–ENRICH–BOUND subgraph selection,
LLM-based query generation along 5 directions, 7-step quality control→1,210 Viet-
namese queries.
Three-Phase Subgraph Selection:From the KG stored in ArangoDB, the
system selects all of the L5-level structural nodes that have at least one child
content node. Each L5 candidate is assembled through three phases:
–SEED: Traversal from the selected L5 node to collect core content nodes.
–ENRICH: From each core content node, one-hop traversal via Tier-2 seman-
tic relations discovers content nodes belonging to other sections and selects
at most two sections ranked by the number of relations between content
nodes. Then we retrieve the parent nodes and child nodes that belong to
them.
–BOUND: We filter to select the relevant relations among collected nodes
(i.e., core and enriched nodes), forming a closed subgraph.
Five Query Directions:The benchmark categorizes queries along five direc-
tionsreflectingdistinctinformationneeds.Neutral(NT)queriesperformasingle-
fact lookup, require only one node serving as a baseline. Top-down (TD) queries
synthesize multiple nodes within the same section. Horizontal (HZ) queries re-
late to the information from different sections. Bottom-up (BU) queries target
summaries or themes at the section level. Finally, Cross-section (CS) queries
compare content from two distinct sections.

8 N. K. Doan and N. H. N. Minh
Query Generation and Quality Control:The LLM generates JSON includ-
ing the query, a list of relevant nodes and their relevant level 0/1/2, direction,
and a confidence score. A seven-step QA verification for each output will help to
checkthenodesexistence,contentrelevanceexceedingacosinethreshold,remove
the duplicates, remove non-relevant nodes, and validate directions given their
quotas. High-confidence outputs receive a 10% spot-check; medium-confidence
outputs are reviewed at a 50% rate; every low-confidence item is reviewed in full.
6 Experiments and Results
6.1 Benchmark dataset
The KG is constructed from three Vietnamese high school History textbooks
of the Canh Dieu edition for Grades 10, 11, and 12, totaling approximately 400
pagesand covering worldand Vietnamesehistory from ancient timestothe mod-
ern era, following the 2018 General Education Curriculum. In this work, 3 LLM
models were employed across the pipelines. The weak LLM model Qwen3.5 397B
A17B [27] handles bulk relation extraction, accounting for over 95% of the work-
load,andalsosynthesizessummariesforstructuralnodes;thestrongLLMmodel
Claude 4.6 Sonnet [4] validates novel relation types. The embedding model text-
embedding-3-large [18] withd=3072dimensions generates vector representations
for all nodes and ontology entries.
The automated generation pipeline described in Section 5 produces 1,210
Vietnamese queries from 109 subgraphs. After quality verification, 1,182 queries
are clean and 28 are flagged. The relevance label distribution is as follows: grade-
0, meaning not relevant, accounts for 2,400 labels at 36.9%; grade-1, meaning
partially relevant, accounts for 1,143 labels at 17.6%; and grade-2, meaning di-
rectly relevant, accounts for 2,968 labels at 45.6%. Query direction distribution:
TD with 321 queries at 26.5%, HZ with 363 at 30.0%, BU with 290 at 24.0%,
NT with 136 at 11.2%, and CS with 100 at 8.3%.
Metrics:We evaluate all strategies at cutoffsK={1,3,5,10}. We use
NDCG@K [11] as the main metric because it works well with our three-level
graded relevance labels (i.e., grade 0/1/2). Recall@K and Precision@K measure
coverage and exactness. MRR measures how quickly the first correct result ap-
pears.
Baselines:We evaluated three baselines in this paper:B1performs vector
search with cosine similarity on ArangoSearch,B2uses BM25 [20] for lexical
search,B3combines both methods through RRF.
Graph Construction Results:Table 1 gives an overview of the graph. It
contains 750 nodes, including 3 root, 263 structural, and 484 content nodes,
connected by 5,372 edges of which 1,031 are tree-structural and 4,341 are se-
mantic. The four structural levels span 15 chapters, 28 lessons, 109 sections, and
111 subsections.

Automated Tree KG Construction with Ontology Expansion 9
Table 1.Knowledge graph statistics and distribution of the 10 most frequent semantic
relation types
Graph component Counts Top 10 Relation #Edges
Root nodes (books) 3Contextualizes830
Structural nodes 263Refers_To620
Content nodes 484Contrasts444
Total nodes 750Exemplifies345
Structural edges 1,031Elaborates324
Semantic edges 4,341Preceded_By319
Total edges 5,372Related_To253
Supports230
Part_Of128
Followed_By115
6.2 Overall Retrieval Ranking
In the experiments, we evaluate and rank 12 methods, including 9 Graph-RAG
variant strategies and 3 baselines, using NDCG@10 across all 1,210 queries as
shown in Table 2.
Table 2.Overall ranking of 12 methods at K=10 on 1,210 queries.
# ID Name NDCG Recall Precision MRR
1E1bTopDown RRF0.8397 0.9053 0.30400.8967
2E3cBottomUp RRF 0.8122 0.8307 0.27110.9331
3E3bBottomUp AVG_LOG 0.8110 0.8300 0.2704 0.9310
4E1aTopDown Cosine 0.8075 0.8284 0.2721 0.9288
5E1cTopDown MMR 0.8065 0.8268 0.2704 0.9299
6E3aBottomUp Parent 0.8036 0.8236 0.2710 0.9253
7E2aHoriz. Cosine 0.8009 0.8090 0.2634 0.9298
8 B1 Vector-Only 0.7923 0.8179 0.2637 0.9301
9 B3 Hybrid RRF 0.7806 0.7903 0.2523 0.9262
10E2cHoriz. Filtered 0.7539 0.7369 0.2812 0.9256
11E2bHoriz. RRF 0.7343 0.7422 0.2382 0.9190
12 B2 BM25 0.6770 0.6791 0.2112 0.8508
E1b — The TopDown RRF strategy outperforms the other retrieval methods
for all metrics due to the prevalence of top-down query styles of the benchmark
datasets. In practice, many queries are inherently inclusive from structured text-
book sections and align naturally with the hierarchical approach. Consequently,
TopDown RRF achieves an NDCG@10 of 0.8397 and a Recall@10 of 0.9053,
representing improvements of 4.7 and 8.7 percentage points, respectively, over
the B1 baseline. Most of Graph-RAG variants outperform the three baselines,
demonstrating the usefulness of structural information in the RAG process. Be-
sides, Bottom-Up E3c achieves the highest MRR of any method at 0.9331, mean-
ingitreliablyplacesthebestresultfirst,butitrankssecondonNDCG.Thescore
aggregation mechanism from child nodes described in Equation 3 is effective to
identify the single most relevant section. E2a - Horizontal strategy slightly out-

10 N. K. Doan and N. H. N. Minh
performs B1 by 0.86 percentage points in NDCG through reaching cross-branch
nodes, but E2b drops 6.7 percentage points relative to E2a. There are some over-
lapping information between Seed node and extended nodes introducing noise
in RRF.
Analysis for different K:Table 3 compares NDCG at four cutoff values for
the best method in each strategy alongside two baselines. Bottom-Up strategy
Table 3.NDCG, Recall, and Precision atK=1,3,5,10for representative methods.
K=1K=3K=5K=10
Method NDCG Rec. Pre. NDCG Rec. Pre. NDCG Rec. Pre. NDCG Rec. Pre.
E1b(TD RRF) 0.820 0.308 0.8360.758 0.627 0.640 0.786 0.774 0.497 0.840 0.905 0.304
E3c(BU RRF)0.8810.3450.8930.754 0.596 0.596 0.759 0.699 0.439 0.812 0.831 0.271
E2a(HZ Cos) 0.871 0.344 0.883 0.745 0.590 0.584 0.756 0.698 0.434 0.801 0.809 0.263
B1 (Vector) 0.871 0.344 0.883 0.724 0.590 0.584 0.746 0.698 0.434 0.792 0.818 0.264
B3 (Hybrid) 0.8670.3470.881 0.718 0.563 0.548 0.726 0.660 0.405 0.781 0.790 0.252
withK= 1results in the highest score because the child-node can produce the
single best section. From K≥3, Top-Down obtains the highest score because the
seed_score propagated downward helps all children in the right section and ob-
tains sufficient coverage in many queries from the benchmark. The gap between
graph strategies and baselines grows as K increases. AtK= 1, B1 matches E2a
at 0.871; atK= 10, E1b has a higher score than B1 by 4.7 point. Note that the
higherKstrategies produce a lower NDCG@10 and Precision score compared to
K= 1; however, the coverage has been extended as the increment of the recall
scores. Structural information becomes more influenced whenKincreases.
6.3 Strategy×Query Direction Matrix
Table 4 presents NDCG@10 broken down by five query directions. The direction-
level breakdown shows large differences between strategies:
Table 4.NDCG@10 by query direction,K= 10.
Method TD HZ BU NT CS
B1 (Vector) 0.762 0.832 0.716 0.925 0.755
B3 (Hybrid) 0.755 0.826 0.6800.9620.742
E1b(TD RRF)0.8760.8260.8780.7410.799
E2a(HZ Cos) 0.7710.8610.709 0.923 0.777
E3c(BU RRF) 0.791 0.858 0.744 0.897 0.797
–TD:E1b reaches 0.876, which is 11.4 point above B1.
–HZ:E2a leads at 0.861 by finding causally or temporally related nodes in
other sections.
–BU:E1b scores 0.878 and surpasses even E3c at 0.744.
–NT:B3 reaches 0.962, the highest score in the entire table, while E1b drops
to 0.741 because structural traversal spreads results across irrelevant nodes.
–CS:The most challenged direction for all methods, as no strategy exceeds
0.80.

Automated Tree KG Construction with Ontology Expansion 11
7 Discussions
In this paper, we proposed a hierarchical KG construction pipeline with con-
trolledontologyexpansionandcomparedthreeGraph-RAGstrategiesfromViet-
namese history textbooks. The preliminary results exhibit a promising direction
for future Graph-RAG assisted educational agents to produce confident and
high-quality answers. While Graph-RAG strategies utilizing structural informa-
tion could produce better and inclusive answers for complex queries that span
multiple sections, certain limitations remain. Due to limited capacity, our study
was validated on high school history textbooks using an automated benchmark
construction process, resulted in a relatively small scale of nodes and relations.
Expert-annotated benchmarks for both extraction and retrieval evaluation are
also important to enhance the consistency of experimental studies. The use of
hand-tuned thresholds may affect the variant performance of graph retrieval sys-
tems. In future work, we will extend the pipeline to other domains and experi-
ment more efficient retrieval strategies capable of handling increasingly complex
queries.
AcknowledgmentThis research is funded by Vietnam - Korea University
of Information and Communication Technology under project number ÐHVH-
2026-23.
References
1. Abu-Rasheed, H., Jumbo, C., et al.: LLM-assisted knowledge graph completion
for curriculum and domain modelling in personalized higher education recommen-
dations. In: Proceedings of the IEEE Global Engineering Education Conference
(EDUCON). pp. 1–5 (2025)
2. Ain, Q.U., Chatti, M.A., et al.: An optimized pipeline for automatic educational
knowledge graph construction. In: Proceedings of the International Joint Confer-
ence on Knowledge Graphs (IJCKG) (2025)
3. Ain, Q.U., Chatti, M.A., et al.: Top-down vs. bottom-up approaches for automatic
educational knowledge graph construction in CourseMapper. In: Proceedings of
the European MOOC Stakeholder Summit (EMOOCs). pp. 119–129 (2025)
4. Anthropic: System card: Claude Sonnet 4.6. Tech. rep., Anthropic PBC (2 2026),
https://www.anthropic.com/system-cards
5. ArangoDB GmbH: ArangoDB: Multi-model database for graph, document, key-
value, and search (2024)
6. Bian, R., Geng, Y., et al.: AutoMathKG: The automated mathematical knowledge
graphbasedonLLMandvectordatabase.ComputationalIntelligence41(4)(2025)
7. Carbonell, J., Goldstein, J.: The use of MMR, diversity-based reranking for re-
ordering documents and producing summaries. In: Proceedings of the 21st Annual
International ACM SIGIR Conference. pp. 335–336 (1998)
8. Edge, D., Trinh, H., et al.: From local to global: A graph RAG approach to query-
focused summarization. arXiv preprint arXiv:2404.16130 (2024)
9. Gordon V. Cormack, Charles L A Clarke, S.B.: Reciprocal rank fusion outperforms
condorcet and individual rank learning methods. In: Proceedings of the 32nd In-
ternational ACM SIGIR Conference on Research and Development in Information
Retrieval. pp. 758–759 (2009)

12 N. K. Doan and N. H. N. Minh
10. Hu, Y., Zhu, J., et al.: ReMindRAG: Low-cost LLM-guided knowledge graph
traversal for efficient RAG. In: Advances in Neural Information Processing Sys-
tems (NeurIPS) (2025)
11. Järvelin, K., Kekäläinen, J.: Cumulated gain-based evaluation of IR techniques.
ACM Transactions on Information Systems20(4), 422–446 (2002)
12. Jeff Z. Pan ORCID-Logo, S.R., et al.: Large language models and knowledge
graphs: Opportunities and challenges. Transactions on Graph Data and Knowl-
edge (TGDK)1(1), 2:1–2:38 (2023)
13. Ji, S., Pan, S., et al.: A survey on knowledge graphs: Representation, acquisition,
and applications. IEEE Transactions on Neural Networks and Learning Systems
33(2), 494–514 (2022)
14. Lewis, P., Perez, E., et al.: Retrieval-augmented generation for knowledge-intensive
NLP tasks. In: Advances in Neural Information Processing Systems (NeurIPS).
vol. 33, pp. 9459–9474 (2020)
15. Lu, Y., Wu, W., Zhao, X., Peng, R., Wang, J.: KARMA: Leveraging multi-agent
LLMs for automated knowledge graph enrichment. In: Advances in Neural Infor-
mation Processing Systems (NeurIPS) (2025)
16. Luo, L., Zhao, Z., et al.: GFM-RAG: Graph foundation model for retrieval
augmented generation. In: Advances in Neural Information Processing Systems
(NeurIPS) (2025)
17. Nguyen, D.Q., Nguyen, A.T.: PhoBERT: Pre-trained language models for Viet-
namese. In: Findings of the Association for Computational Linguistics: EMNLP
2020. pp. 1037–1042 (2020)
18. OpenAI: text-embedding-3-large.https://platform.openai.com/docs/models/
text-embedding-3-large(2024)
19. Peng, B., Zhu, Y., et al.: Graph retrieval-augmented generation: A survey. ACM
Transactions on Information Systems44(2), 1–52 (2025)
20. Robertson, S., Zaragoza, H.: The probabilistic relevance framework: BM25 and
beyond. Foundations and Trends in Information Retrieval3(4), 333–389 (2009)
21. Salman, M., Haller, A., et al.: Doc-KG: Unstructured documents to knowledge
graph construction, identification and validation with wikidata. Expert Systems
41(9) (2024)
22. Sarthi,P.,Abdullah,S.,etal.:RAPTOR:Recursiveabstractiveprocessingfortree-
organized retrieval. In: The Twelfth International Conference on Learning Repre-
sentations (ICLR) (2024)
23. Shimizu, C., Hitzler, P.: Accelerating knowledge graph and ontology engineering
with large language models. Journal of Web Semantics85, 100862 (2025)
24. Somin Wadhwa, Silvio Amir, B.C.W.: Revisiting relation extraction in the era of
large language models. In: Proceedings of the 61st Annual Meeting of the ACL.
pp. 15566–15589 (2023)
25. Yang, X., Sun, K., et al.: CRAG – comprehensive RAG benchmark. In: Advances
in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks
Track (2024)
26. Zhang,Q.,Wang,B.,Huang,V.S.J.,etal.:Documentparsingunveiled:Techniques,
challenges, and prospects for structured information extraction. arXiv preprint
arXiv:2410.21169 (2024)
27. Zheng, C., Jiang, Y., Xu, Z., et al.: Qwen3 technical report (2025)