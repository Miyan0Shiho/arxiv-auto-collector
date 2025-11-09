# PROPEX-RAG: Enhanced GraphRAG using Prompt-Driven Prompt Execution

**Authors**: Tejas Sarnaik, Manan Shah, Ravi Hegde

**Published**: 2025-11-03 18:00:56

**PDF URL**: [http://arxiv.org/pdf/2511.01802v1](http://arxiv.org/pdf/2511.01802v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become a robust framework for
enhancing Large Language Models (LLMs) with external knowledge. Recent advances
in RAG have investigated graph based retrieval for intricate reasoning;
however, the influence of prompt design on enhancing the retrieval and
reasoning process is still considerably under-examined. In this paper, we
present a prompt-driven GraphRAG framework that underscores the significance of
prompt formulation in facilitating entity extraction, fact selection, and
passage reranking for multi-hop question answering. Our approach creates a
symbolic knowledge graph from text data by encoding entities and factual
relationships as structured facts triples. We use LLMs selectively during
online retrieval to perform semantic filtering and answer generation. We also
use entity-guided graph traversal through Personalized PageRank (PPR) to
support efficient, scalable retrieval based on the knowledge graph we built.
Our system gets state-of-the-art performance on HotpotQA and 2WikiMultiHopQA,
with F1 scores of 80.7% and 78.9%, and Recall@5 scores of 97.1% and 98.1%,
respectively. These results show that prompt design is an important part of
improving retrieval accuracy and response quality. This research lays the
groundwork for more efficient and comprehensible multi-hop question-answering
systems, highlighting the importance of prompt-aware graph reasoning.

## Full Text


<!-- PDF content starts -->

PROPEX-RAG: Enhanced GraphRAG using
Prompt-Driven Prompt Execution
Tejas Sarnaik[0009−0009−6834−3847], Manan Shah[0009−0003−7996−4768], and Ravi
Hegde[0000−0002−0418−5861]
Indian Institute of Technology, Gandhinagar, Gujarat, India
{tejas.sarnaik, mananshah, hegder}@iitgn.ac.in
Abstract.Retrieval-Augmented Generation (RAG) has become a ro-
bust framework for enhancing Large Language Models (LLMs) with
external knowledge. Recent advances in RAG have investigated graph-
based retrieval for intricate reasoning; however, the influence of prompt
design on enhancing the retrieval and reasoning process is still con-
siderably under-examined. In this paper, we present a prompt-driven
GraphRAG framework that underscores the significance of prompt for-
mulation in facilitating entity extraction, fact selection, and passage re-
ranking for multi-hop question answering. Our approach creates a sym-
bolic knowledge graph from text data by encoding entities and factual
relationships as structured facts triples. We use LLMs selectively dur-
ing online retrieval to perform semantic filtering and answer generation.
We also use entity-guided graph traversal through Personalized PageR-
ank (PPR) to support efficient, scalable retrieval based on the knowledge
graphwebuilt.Oursystemgetsstate-of-the-art performanceonHot-
potQA and 2WikiMultiHopQA, with F1 scores of80.7%and78.9%,
and Recall@5 scores of97.1%and98.1%, respectively. These results
show that prompt design is an important part of improving retrieval
accuracy and response quality. This research lays the groundwork for
moreefficientandcomprehensiblemulti-hopquestion-answeringsystems,
highlighting the importance of prompt-aware graph reasoning. Code and
data are available at https://github.com/tejas-sarnaik/ProPEX-RAG.
Keywords:Retrieval-AugmentedGeneration,Prompt-drivenRetrieval,
Symbolic Knowledge Graph, Graph-based Reasoning, Multi-hop QA,
Large Language Models
1 Introduction
Large language models (LLMs) have shown significant prowess in understanding
and generating natural language, matching human-level performance in varied
tasks. However, they face core shortcomings in accessing current information,
reasoning with external knowledge, and providing factually verifiable answers,
especially in multi-hop tasks that require the synthesis of information from var-
ious sources through intricate reasoning chains. The RAG paradigm improves
LLM by enabling dynamic access to external knowledge bases. Traditional RAGarXiv:2511.01802v1  [cs.CV]  3 Nov 2025

2 Tejas Sarnaik et al.
uses dense retrieval to identify relevant passages, conditioned on the contexts
retrieved [1]. This method is effective for single-hop queries, but it often fails
in multihop reasoning where the relevant data is distributed across multiple
distinct passages [2]. New methods explore graph-based retrieval for advanced
reasoning. Graph structures illustrate explicit entity relationships, supporting
inferential pathways similar to human reasoning. HippoRAG2 used neurobiolog-
ical architectures to improve performance by up to 20%, with cost reductions
of 10-30x [3]. These innovations underscore graph-based strategies as powerful
for complex tasks requiring integrated knowledge sourcing. Despite progress,
the role of prompt design in RAG system efficiency is under explored. Current
methods treat it as a secondary optimization rather than a central feature of
the architecture. This neglect is critical given the evidence that prompt design
affectsentityextractionaccuracy,factselection,andpassageclassification.Step-
back prompting alone increases the precision of complex reasoning when used
with retrieval. This research presents a prompt-based GraphRAG framework
prioritizing prompt design to enhance retrieval-augmented generation, featuring
prompt-driven execution integration, embedding prompt engineering through-
out the pipeline, and prompt-guided semantic filtering, using prompts for pre-
cise sub-graph selections, maintaining semantic integrity, and reducing compu-
tational demands.
Related Work:Recent advancements in retrieval-augmented generation pro-
pose that incorporating symbolic structures, notably knowledge graphs, into re-
trieval can substantially enhance outcomes. Traditionally, RAG employs dense
vector retrieval and semantic similarity to support assertions, which often fails
to uncover link structures vital for advanced reasoning. Graph-Based RAG
solves this by integrating knowledge graph connections, simplifying discovery.
NodeRAG [4], an RAG variant, uses various graph nodes that show interconnec-
tions in long-form queries, thus optimizing multi-hop question answering (QA).
Document GraphRAG helps to understand documents using knowledge graphs
[5], improving search efficiency across multiple texts. KG-Based RAG employs
graph vector embeddings to enhance retrieval semantics [6], vital for multi-hop
QA requiring interconnected evidence synthesis and data flow tracking. Cur-
rent research transcends linear search, inspired by datasets like HotpotQA and
2WikiMultiHopQA. Fact-Centric Knowledge Web [7] highlight fact-centric webs
for domain-agnostic retrieval, focusing on prompt-sensitive symbolic reasoning.
Query-Aware GNN [8] enhance retrieval using query-sensitive graph neural net-
works, demonstrating the role of question structure in graph-based reasoning for
complex multi-hop inference. Beyond graph-augmented retrieval, recent work
elevates prompt policies to first-class, auditable components that steer seed-
ing, local fact filtering, typed graph traversal, and evidence presentation at test
time [21]. Rather than re-training large retrievers or relying on heavy learned
re-rankers,acompactpolicylayerinstantiatedviaprecisepromptsandlightcon-
trollers enables controllable navigation in the spirit of topic-sensitive PageRank,
transparent error surfaces, and low-friction domain adaptation [22,23].

PROPEX-RAG: Enhanced GraphRAG 3
2 Proposed Methodology
We propose a prompt-driven multi-hop RAG framework built around a novel
Prompt-Driven Prompt Execution (ProPEX) mechanism that actively guides
the retrieval process. Our system constructs a symbolic entity-centric knowledge
graph from LLM-extracted factual triples and employs prompt-conditioned LLM
filtering to select relevant facts. Inference, query prompts drive both seed entity
selectionandaPersonalizedPageRank(PPR)traversaloverthegraphtoretrieve
contextually linked passages. This integration of prompt design with structured
graph reasoning enables interpretable, scalable, and accurate multi-hop QA. The
overall methodology is illustrated in Figure 1.
2.1 Symbolic Graph Construction (Offline Indexing)
To enable efficient and interpretable multi-hop reasoning, we implement a sym-
bolic knowledge graph as a persistent memory layer. A modular, prompt-guided
pipeline extracts high-precision entities and factual assertions using a two-stage,
LLM-assisted process. The prompts identify salient entities such as persons, or-
ganizations and locations, forming factual triplesT={(s, p, o)}where the sub-
jectsand the objectoare structurally coherent entities. In a knowledge graph,
the entity and the passage nodes are connected via directed mentioned-in links,
forming typed relational edges within a heterogeneous graph. This structure
includes bidirectional synonymy edges (from dense embedding similarity) and
contextual relatedness edges derived fromTco-occurrence data. Node scores are
computed using the inverse passage frequency, while a sparse entity-to-passage
incidence matrix enables retroactive reasoning and localized exploration. The
resulting symbolic graph provides a query-time index for controlled, multi-hop
retrieval and inference.
2.2 Entity-Guided Graph Traversal (Online Retrieval)
At inference time, we perform entity-centric multi-hop retrieval using a symbolic
propagation framework based on PPR over a heterogeneous knowledge graph
G= (V,E), where nodes represent entities or passages, and edges encode typed
semantic relations. Given a queryq, we compute its embeddingqand retrieve
top-kfact triplesT q={(s i, pi, oi)}via cosine similarity with pre-embedded
triples. These are refined by an LLM to obtainT∗
q, from which we extract seed
entities:
Eq={s i, oi|(si, pi, oi)∈ T∗
q}(1)
We initialize a restart distributionv(0)over seed entities:
v(0)
e=(
α
|Eq|,ife∈ E q
0,otherwise(2)
whereα= 1.0controls the probability of restart. This is propagated acrossG
through weighted edges representingsynonymy(w sim),contextual related-
ness(w rel), and references to the passage, using the iterative update of PPR:
v(t+1)= (1−α)·A⊤v(t)+α·v(0)(3)

4 Tejas Sarnaik et al.
Phase I: Symbolic Graph Construction  (Offline Indexing)
Passages Entity and Facts Similarity Matrix 
Symbolic Knowledge Graph
11Prompt driven information
extractor 2
2Synonym detection by
embedding
33
Similarity augmented graph
encodingEntity NodesPassage Nodes
Mentioned in Edges
Context Edges
Synonymy EdgesLegend
ProPEX  (Prompt-Driven
Prompt EXecution)
Phase II: Entity-Guided Graph T raversal ( Online Retrieval)
Embed Seed
Entities (ε 𝑞 )  Retrieve T op-K
Similar Facts Ranked
Passages (V ₚ  )
Answer 
5 6
7Query
1
2 3 4
Entity-Guided Graph
Traversal (PPR)Filtered facts ( T* 𝑞) 
+
Seed Entities (ε 𝑞 )
Embed question via embedding model 2Top-K facts via cosine similarity on fact
embeddings.
43 Filter relevant facts from top-K
5 6 7 Top Passages Retrieved (V ₚ )Rerank T op Passages by entity & fact
overlapEntity seed scoring (εₓ), graph propagation
(PPR)LLM Final answer generationTop-k Passages
+
Facts ( T* 𝑞)
1
Fig. 1.Architecture of our retrieval-augmented QA framework. Phase I constructs
a symbolic knowledge graph from LLM-extracted entities and facts triples. Phase II
performs PPR-based traversal using query-aligned seeds and filtered facts to retrieve
and re-rank passages for grounded answer generation.
whereAis the normalized adjacency matrix. After convergence, the passage
nodes accumulate relevance scoresv pthat reflect their multi-hop connectivity
to the query.
The final retrieval is performed by re-ranking:
Retrieve(q) = TopK (v p+λrerank·overlap(p, q))(4)
where overlap(p, q)rewards symbolic alignment with query entities and filtered
facts. This PPR-guided traversal enables interpretable, prompt-conditioned rea-
soning for structured multi-hop QA tasks.
Answer Generation:Following PPR-based symbolic propagation, the topk
passages reflecting multi-hop relational alignment within the knowledge graph
are selected as the evidence context. To synthesize the final answer, we construct
a structured prompt containing: (i) the ranked passages, (ii) seed entitiesE q,
(iii) LLM-filtered factual triplesT∗
q, and (iv) a directive to produce an evidence-
based response. A deterministic decoding strategy is employed using GPT-4.1
mini in zero- or few-shot mode without fine-tuning. This prompt-executed gener-
ationenhancesfactualconsistency,reduceshallucinations,andensurestraceable,
retrieval-aligned answers.

PROPEX-RAG: Enhanced GraphRAG 5
3 Results & discussion
3.1 Baselines
We benchmark our framework against representative dense, multi-hop, and sym-
bolic retrieval baselines. Dense retrievers include DPR [1] and RAG [9], which
retrieve passages through embedding similarity and fuse them with generative
models, we consider Contriever [10] and GTR [11], both widely used embedding-
based retrievers. From the category of large embedding models, we include
GritLM/GritLM-7B[12]andNV-Embed-v2[13].Forstructure-augmentedmeth-
ods, we benchmark against RAPTOR [14], which organizes the corpus hierarchi-
cally by semantic similarity, GraphRAG [15] and LightRAG [16], both of which
employ knowledge graph structures for reasoning, and HippoRAG2 [3], a recent
extension that integrates entity-guided Personalized PageRank for multi-hop re-
trieval.
3.2 Implementation Details
Our framework is training-free and lightweight. Symbolic memory is constructed
once per corpus and is accessed read-only at inference. At query time, the con-
troller applies a seed policy, a compact keep–drop prompt for fact gating, and
a typed PPR traversal with fixed damping. Entity mentions are extracted with
GPT-4.1-mini, and similarity is scored using text-embedding-3-large. For each
query, the top-5 triples are retained, and the QA module conditions on these
passages with an evidence-first prompt under deterministic decoding.
Table 1.Statistics of the multi-hop QA datasets used in our evaluation.
Dataset Number of Queries Number of Passages
HotpotQA 1,000 9,811
2WikiMultihopQA 1,000 6,119
3.3 Dataset Description
We evaluated our system using two prominent multi-hop question answering
datasets:HotpotQA[17] and2WikiMultihopQA[18]. These datasets are
designed to assess complex reasoning across texts, making them suitable for
measuring retrieval and inference efficiency in symbolic graph-enhanced RAG
systems. HotpotQA focuses on linking facts with confirmed supporting informa-
tion, while 2WikiMultihopQA presents a wider range of evidential diversity and
challenges of associative retrieval. Table 1 provides an overview of the statistics
of the data set.
Evaluation Metrics:We evaluate on HotpotQA [17] and 2WikiMulti-
HopQA [18] using Exact Match (EM) and token-level F1 for answer accuracy,
and Recall@k(k∈ {1,2,5,8,10}) for retrieval quality. Recall@5 highlights early
evidence grounding. All metrics are computed on held-out sets with per-query
logging for detailed analysis.

6 Tejas Sarnaik et al.
Table 2.QA performance Exact Match(EM) and F1 scores on 2WikiMultiHopQA and
HotpotQA datasets. Bold values indicate the best scores
Method2WikiMultiHopQA HotpotQA Average
EM F1 EM F1 EM F1
Simple Baselines
Contriever [10] 38.1 41.9 51.3 62.3 44.7 52.1
GTR (T5-base) [11] 49.2 52.8 50.6 62.8 49.9 57.8
Large Embedding Models
GritLM-7B [12] 55.8 60.6 60.7 73.3 58.2 66.9
NV-Embed-v2 (7B) [13] 57.5 61.5 62.8 75.3 60.1 68.4
Structure-Augmented RAG
RAPTOR [14] 47.3 52.1 56.8 69.5 52.0 60.8
GraphRAG [15] 51.4 58.6 55.2 68.6 53.3 63.6
LightRAG [16] 9.4 11.6 2.0 2.4 5.7 7.0
HiRAG [19] 69.0 74.4 62.0 72.9 65.5 73.6
HippoRAG [20] 65.0 71.8 52.6 63.5 58.8 67.6
HippoRAG 2 [3] 65.0 71.0 62.7 75.5 63.8 73.2
ProPEX-RAG 76.4 78.9 79.9 80.7 78.1 79.8
Table 3.Retrieval performance (passage recall@5) on 2WikiMultiHopQA and Hot-
potQA datasets. Results show the percentage of queries with at least one gold-
supporting passage in the top-5 retrieved candidates. GraphRAG, LightRAG and Hi-
RAG are not presented because they do not directly produce passage retrieval results.
Method 2WikiMultiHopQA HotpotQA Average
Simple Baselines
Contriever [10] 57.5 75.3 66.4
GTR (T5-base) [11] 67.9 73.9 70.9
Large Embedding Models
GritLM-7B [12] 76.0 92.4 84.2
NV-Embed-v2 (7B) [13] 76.5 94.5 85.5
Structure-Augmented RAG
RAPTOR [14] 66.2 86.9 76.5
HippoRAG [20] 89.1 77.7 83.4
HippoRAG 2 [3] 90.4 96.3 93.3
ProPEX-RAG 98.1 97.1 97.6
Results:In this section we present the experimental results on two standard
multi-hop QA benchmarks, HotpotQA [17] and 2WikiMultihopQA [18] which
are designed to test multi-step reasoning over diverse and compositional queries.
To assess both retrieval and answer generation quality, we use the GPT-4.1
mini model as the decoding engine, conditioned on top-ranked passages retrieved
through symbolic graph traversal. Our system demonstrates consistently strong
performance across both datasets. The symbolic propagation mechanism, guided

PROPEX-RAG: Enhanced GraphRAG 7
by entity-centric cues and fact-level context, enables a high recall of relevant
passages within a restricted top-kbudget. When combined with our fact-aligned
reranking strategy, the approach achieves robust end-to-end answer accuracy,
outperforming many recent structure-aware RAG systems. Detailed results, in-
cluding F1 and exact match metrics, together with retrieval recall comparisons,
are summarised in Table 2 and Table 3
4 Conclusion
We introduced ProPEX-RAG, an interpretable and prompt-driven QA frame-
work that combines LLM-guided fact selection with Personalized PageRank-
based traversal over symbolic knowledge graphs. The model beats the previously
achieved SOTA. Further testing is being conducted on multiple other multi-hop
datasets, and a second consideration is to assess how ProPEX-RAG handles
scaling of corpus sizes.
Acknowledgements
We gratefully acknowledge the Walmart Center for Technical Excellence at IIT
Madras for their generous support and resources that enabled this research.
References
1. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W.:
Dense Passage Retrieval for Open-Domain Question Answering.Proceedings of the
2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)
(2020)
2. Xiong, W., Xiong, Y., Li, Y., Tang, J., Liu, Z., Bennett, P., Ahmed, F., Bosma, M.,
Craswell, N.: Answering Complex Open-Domain Questions with Multi-Hop Dense
Retrieval.International Conference on Learning Representations (ICLR)(2021)
3. Gutiérrez, B.J., Shu, Y., Qi, W., Zhou, S., Su, Y.: From RAG to memory:
Non-parametric continual learning for large language models.arXiv preprint
arXiv:2502.14802(2025)
4. Xu, T., Zheng, H., Li, C., Liu, Y., Chen, R., & Zhang, M. (2025). NodeRAG: Struc-
turing graph-based rag with heterogeneous nodes.arXiv preprint arXiv:2504.11544.
5. Knollmeyer, S., Caymazer, O., & Grossmann, D. (2025). Document GraphRAG:
Knowledge Graph Enhanced Retrieval Augmented Generation for Document Ques-
tion Answering Within the Manufacturing Domain.Electronics
6. Shavaki, M. A., Omrani, P., & Toosi, R. (2024). Knowledge Graph Based Retrieval-
Augmented Generation for Multi-Hop Question Answering Enhancement.Proc. of
the 15th Int. Conf. on Data Science
7. Sinha, R., & Shiramatsu, S. (2024). Fact-Centric Knowledge Web for Information
Retrieval.IEEE/WIC Int. Conference on Web Intelligence
8. Agrawal, V., Wang, F., & Puri, R. (2025). Query-Aware Graph Neural Networks
for Enhanced Retrieval-Augmented Generation.ICLR Submission

8 Tejas Sarnaik et al.
9. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kul-
mizev, A., Lewis, M., Yih, W., Riedel, S.: Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks.Advances in Neural Information Processing Sys-
tems (NeurIPS)(2020)
10. Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., Grave,
E.: Unsupervised Dense Information Retrieval with Contrastive Learning.arXiv
preprint arXiv:2112.09118(2021)
11. Muennighoff, N., Su, H., Wang, L., Yang, N., Wei, F., Yu, T., Singh, A., Kiela,
D.: Generative Representational Instruction Tuning.The Thirteenth International
Conference on Learning Representations (ICLR)(2024)
12. Ni, J., Qu, C., Lu, J., Dai, Z., Hernandez Abrego, G., Ma, J., Zhao, V.Y., Luan, Y.,
Hall, K.B., Chang, M.-W., et al.: Large Dual Encoders are Generalizable Retrievers.
arXiv preprint arXiv:2112.07899(2021)
13. Lee, C., Roy, R., Xu, M., Raiman, J., Shoeybi, M., Catanzaro, B., Ping, W.: NV-
Embed: Improved Techniques for Training LLMs as Generalist Embedding Models.
arXiv preprint arXiv:2405.17428(2024)
14. Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., Manning, C.D.: RAP-
TOR: Recursive Abstractive Processing for Tree-Organized Retrieval.The Twelfth
International Conference on Learning Representations (ICLR)(2024)
15. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S.,
Metropolitansky, D., Ness, R.O., Larson, J.: From Local to Global: A Graph
RAG Approach to Query-Focused Summarization.arXiv preprint arXiv:2404.16130
(2024)
16. Guo, Z., Xia, L., Yu, Y., Ao, T., Huang, C.: LightRAG: Simple and Fast Retrieval-
Augmented Generation.arXiv preprint arXiv:2410.05779(2024)
17. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R., Manning,
C.D.: HotpotQA: A dataset for diverse, explainable multi-hop question answering.
arXiv preprint arXiv:1809.09600(2018)
18. Ho, X., Nguyen, A.-K.D., Sugawara, S., Aizawa, A.: Constructing a Multi-hop
QA Dataset for Comprehensive Evaluation of Reasoning Steps.arXiv preprint
arXiv:2011.01060(2020)
19. Huang, H., Huang, Y., Yang, J., Pan, Z., Chen, Y., Ma, K., Chen, H., Cheng,
J.: Retrieval-Augmented Generation with Hierarchical Knowledge.arXiv preprint
arXiv:2503.10150(2025)
20. Jimenez Gutierrez, B., Shu, Y., Gu, Y., Yasunaga, M., Su, Y.: HippoRAG: Neuro-
biologically Inspired Long-Term Memory for Large Language Models.Advances in
Neural Information Processing Systems37, 59532–59569 (2024)
21. Wei,J.,Wang,X.,Schuurmans,D.,Bosma,M.,Xia,F.,Chi,E.,Le,Q.V.,Zhou,D.,
others: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
Advances in Neural Information Processing Systems, 35, 24824–24837 (2022)
22. Haveliwala, T.H.: Topic-sensitive pagerank.Proceedings of the 11th International
Conference on World Wide Web, 517–526 (2002)
23. Peng, B., Zhu, Y., Liu, Y., Bo, X., Shi, H., Hong, C., Zhang, Y., Tang, S.: Graph
Retrieval-Augmented Generation: A Survey.arXiv preprint arXiv:2408.08921
(2024)