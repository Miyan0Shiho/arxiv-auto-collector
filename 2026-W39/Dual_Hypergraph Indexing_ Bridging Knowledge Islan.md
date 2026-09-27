# Dual-Hypergraph Indexing: Bridging Knowledge Islands for Multi-Hop Reasoning in Retrieval-Augmented Generation

**Authors**: Qi Sun, Xingliang Hou, Caibo Li, Yijia Zhang, Qiang Li, Yu Guo

**Published**: 2026-09-23 13:39:19

**PDF URL**: [https://arxiv.org/pdf/2609.28108v1](https://arxiv.org/pdf/2609.28108v1)

## Abstract
While hypergraph-based Retrieval-Augmented Generation (RAG) effectively captures higher-order multi-entity correlations, existing paradigms treat extracted hyperedges as isolated factual assertions. This structural fragmentation engenders rigid "knowledge islands" that bottleneck multi-hop causal inference, temporal tracking, and narrative synthesis. To systematically address these challenges, we introduce Dual-Hypergraph Indexing (DHI), a hierarchical representation framework that elevates discrete facts into structured analytical insights. DHI couples a foundational entity-relation factual hypergraph ($H_K$) with an elevated deep-insight hypergraph ($H_D$) via a dual-pathway aggregation algorithm. Specifically, DHI employs: (1) importance-driven hub aggregation via 5-metric topological profiling and adaptive thresholding to capture spatial semantic clusters; and (2) temporal chunk-chain progressive aggregation via sliding-window greedy exploration to track chronological evolutions. Across five benchmarks, DHI achieves state-of-the-art performance, boosting logical coherence by +1.53 on the multidisciplinary Mix benchmark and scoring 85.78\% on complex medical pathology reasoning tasks. DHI provides a robust architecture for next-generation multi-hop RAG.

## Full Text


<!-- PDF content starts -->

DUAL-HYPERGRAPH INDEXING: BRIDGING KNOWLEDGE ISLANDS FOR MULTI-HOP
REASONING IN RETRIEV AL-AUGMENTED GENERATION
Qi Sun1*, Xingliang Hou1*, Caibo Li2, Yijia Zhang2, Qiang Li3, Yu Guo1†
1School of Software Engineering, Xi’an Jiaotong University, Xi’an, Shaanxi, China
2State Key Laboratory of Human-Machine Hybrid Augmented Intelligence,
and Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University, Xi’an, Shaanxi, China
3EHV Power Transmission Company of China Southern Power Grid Co., Ltd
ABSTRACT
While hypergraph-based Retrieval-Augmented Generation (RAG)
effectively captures higher-order multi-entity correlations, existing
paradigms treat extracted hyperedges as isolated factual assertions.
This structural fragmentation engenders rigid ”knowledge islands”
that bottleneck multi-hop causal inference, temporal tracking, and
narrative synthesis. To systematically address these challenges, we
introduce Dual-Hypergraph Indexing (DHI), a hierarchical rep-
resentation framework that elevates discrete facts into structured
analytical insights. DHI couples a foundational entity-relation fac-
tual hypergraph (H K) with an elevated deep-insight hypergraph
(HD) via a dual-pathway aggregation algorithm. Specifically, DHI
employs: (1) importance-driven hub aggregation via 5-metric topo-
logical profiling and adaptive thresholding to capture spatial se-
mantic clusters; and (2) temporal chunk-chain progressive aggrega-
tion via sliding-window greedy exploration to track chronological
evolutions. Across five benchmarks, DHI achieves state-of-the-art
performance, boosting logical coherence by +1.53 on the multidis-
ciplinary Mix benchmark and scoring 85.78% on complex medical
pathology reasoning tasks. DHI provides a robust architecture for
next-generation multi-hop RAG.
Index Terms—Retrieval-augmented generation, hypergraph rep-
resentation, knowledge aggregation.
1. INTRODUCTION
The advent of Large Language Models (LLMs) has fundamen-
tally transformed natural language processing, enabling unprece-
dented capabilities in complex reasoning. However, their paramet-
ric memory remains inherently static and susceptible to factual
hallucinations, particularly when navigating specialized, propri-
etary, or rapidly evolving domains [1]. Non-parametric retrieval
augmentation—commonly known as Retrieval-Augmented Gen-
eration (RAG)—curtails these hallucinations by grounding the
generative space in dynamically retrieved, verifiable external ev-
idence [2, 3].
As downstream user queries transition from localized, single-fact
retrieval toward multifaceted deductive reasoning and multi-hop
synthesis, knowledge representation architectures must evolve. Ini-
tial paradigms relying on unstructured text chunk embeddings fre-
quently fail to capture cross-document relational semantics [4]. To
address this, structured graph topographies, such as GraphRAG [5]
and LightRAG [6], explicitly construct entity-relation knowledge
* Equal contribution. † Corresponding author: yu.guo@xjtu.edu.cngraphs to enable macro-level community summaries. More recently,
hypergraph-driven architectures have demonstrated distinct topo-
logical advantages. By natively modeling complex, non-pairwise
interactions among multiple entities (n≥3) as unified hyperedges,
hypergraph RAG prevents the severe semantic distortion caused by
binary edge decomposition [7].
Despite these topological advancements, contemporary hyper-
graph RAG architectures predominantly exhibit a critical struc-
tural limitation we formally term the “Knowledge Island Problem.”
Specifically, existing pipelines typically extract multi-entity hyper-
edges independently from localized text chunks [8,9], inheriting the
structural fragmentation common in chunk-based retrieval [5, 10]
and treating each hyperedge as a static, isolated semantic silo. Con-
sequently, these architectures lack explicit mathematical pathways
to synthesize broader thematic patterns, global causal dependen-
cies, and longitudinal chronological progressions spanning multiple
independent hyperedges. During multi-hop retrieval, standard re-
trievers supply a disconnected subset of assertions [11]. Deprived
of pre-computed relational bridges, the generator is forced to infer
cross-passage dependencies on the fly—a process highly vulnerable
to attention dispersion, cognitive overload, and cascading hallucina-
tions [12, 13].
To transcend these limitations, we propose Dual-Hypergraph In-
dexing (DHI), a novel hierarchical representation framework driven
by a robust dual-pathway knowledge aggregation mechanism. DHI
fundamentally decouples the indexing structure into a meticulously
coordinated two-tiered hierarchy: a foundational factual hypergraph
HKcapturing granular atomic assertions, and an elevated deep in-
sight hypergraphH Dmodeling synthesized conceptual schemas.
DHI’s core computational engine executes two orthogonal aggre-
gation pathways:
•Importance-Driven Hub Aggregation (Path A):Cap-
tures spatial topological significance. We evaluate a robust
5-dimensional structural profiling vector. Applying P90-
normalization and an adaptive elbow cutoff, DHI identifies
and synthesizes systemic cross-context observations centered
around core domain hubs.
•Temporal Chunk-Chain Progressive Aggregation (Path
B):Captures longitudinal narrative evolutions and sequential
causality. By tracking source-chunk chronological footprints
within a constrained sliding window, DHI perfectly bridges
sequential causalities spanning vast document distances.
By seamlessly bridging disconnected knowledge islands, DHI
achieves state-of-the-art logical coherence (85.47 on multidisci-
plinary benchmarks) while significantly reducing context tokens,
arXiv:2609.28108v1  [cs.IR]  23 Sep 2026

establishing a robust blueprint for reasoning-intensive Graph RAG.
2. RELATION TO PRIOR WORK
2.1. Chunk-based and Graph-based RAG Architectures
Standard RAG systems (e.g., RETRO [14], NaiveRAG) perform
dense vector similarity matching over fixed-length text chunks.
While efficient, chunking inherently severs relational continuity
across paragraph boundaries [10]. To resolve this, GraphRAG [5]
pioneered automated entity-relation graphs paired with Leiden com-
munity clustering to generate hierarchical summaries. LightRAG [6]
further optimized this via a dual-level entity-relation scheme, while
HippoRAG [15] introduced neurobiologically inspired memory in-
tegration. Moreover, active RAG frameworks [16] have explored
multi-turn retrieval paths. Nevertheless, traditional pairwise graphs
fundamentally fail to represent high-order interactions involving
three or more entities simultaneously, leading to unavoidable struc-
tural loss [17].
2.2. Hypergraph Representation in Information Retrieval
Hypergraph Neural Networks [7] mathematically validate that hy-
peredges preserve non-decomposable group interactions. Subse-
quent works further introduced hypergraph attention mechanisms
to dynamically capture high-order correlation structures. Hyper-
RAG [9] proposed localized hypergraph modeling, while Cog-
RAG [8] introduced cognitive-inspired theme alignment.
Despite these advances, existing hypergraph RAG approaches
almost universally treat extracted hyperedges as static structural
entities. They fail to distill cross-hyperedge causal trajectories or
longitudinal temporal developments [18]. DHI departs from these
paradigms by formalizing a dual-hypergraph architecture that ex-
plicitly synthesizes higher-order insights via synergistic spatial and
chronological exploration, directly neutralizing the “Knowledge
Island Problem.”
3. PROPOSED METHODOLOGY
3.1. System Architecture and Formal Definition
Given a raw textual corpus partitioned into sequentially ordered,
overlapping chunksC={c 1, . . . , c M}, DHI constructs a hierarchi-
cally coupled dual-hypergraph framework, defined as(H K, HD):
•Factual HypergraphH K= (V K, EK):The foundational
layer. Verticesv∈V Kdenote named entities. Hyperedges
e∈E Krepresent simple pairwise (|e|= 2) or complex
higher-order (|e| ≥3) relations. Every higher-order hy-
peredgeeencapsulates a specific relation summary and an
atomic direct insightι(e), generated during the initial extrac-
tion phase.
•Deep Insight HypergraphH D= (V D, ED):The synthesis
layer. Verticesu∈V Destablish a strict bijective mapping
with higher-order hyperedges ofH K, formulated asV D=
{π(e)|e∈E K,|e| ≥3}. Each deep insight hyperedgeε∈
EDbinds a targeted subset of these insight vertices, enriched
with a synthesized narrative insightI(ε).
3.2. The Knowledge Island Formulation
The necessity ofH Darises from spatial isolation withinH K. For
a multi-hop queryqrequiring logical traversal fromv atovcviaAlgorithm 1Dual-Path Dual-Hypergraph Indexing
Require:Sequential CorpusC, Sliding window limitW= 10.
Ensure:Factual HypergraphH K, Deep Insight HypergraphH D.
1:ParseCvia LLM to extractV K,EK, and direct atomic insights
ι(e).
2:InitializeH K←(V K, EK).
3:InitializeH D←(V D={π(e)|e∈E K,|e| ≥3}, E D=∅).
4:/* Path A: Importance-Driven Hub Aggregation */
5:foreach unique entityv∈V Kdo
6:Computem(v)via Eq. (1) and robustS(v)via Eqs. (2)–(3).
7:end for
8:V hub←ElbowCutoff(SortDescending(V K, S)).
9:foreach identified hub entityv∗∈V hubdo
10:I hub←LLM Synthesize Hub(v∗, E(v∗)).
11:E D←E D∪ {({π(e)|e∈E(v∗)},I hub)}.
12:end for
13:/* Path B: Temporal Chunk-Chain Aggregation */
14:Chains←GreedyBranchingSearch(E K, W)using Eq. (5).
15:Chains pruned←FilterAndDeduplicate(Chains, L min= 3).
16:foreach valid temporal chainX ∈Chains pruned do
17:I chain←LLM Synthesize Temporal(X).
18:E D←E D∪ {({π(e)|e∈ X},I chain)}.
19:end for
20:Construct dense vector database indices forH KandH D.
21:returnH K, HD
intermediaryv b, standard retrieval yields disjoint hyperedgese 1=
{va, vb, . . .}ande 2={v b, vc, . . .}. The semantic void between
ι(e1)andι(e 2)constitutes the knowledge island boundary. DHI ac-
tively computes topological bridgesε={π(e 1), π(e 2)}insideH D
prior to retrieval.
3.3. Dual-Path Insight Aggregation Algorithm
DHI executes two orthogonal aggregation pathways (Algorithm 1)
targeting distinct axes of information distribution.
3.3.1. Path A: Importance-Driven Hub Entity Aggregation
Path A identifies authoritative semantic hubs. Because vertex degree
alone is structurally insufficient for evaluating prominence within
hypergraphs [19], we formulate a 5-dimensional structural profiling
vector for candidate entitiesv∈V K:
m(v) =
d(v),|E(v)|,¯s(v), w(v),|N(v)|T(1)
Here,d(v)is vertex degree,|E(v)|signifies incident hyperedge
count,¯s(v) =|E(v)|−1P
e∈E(v)|e|represents average hyper-
edge scale,w(v)is cumulative semantic weight, and|N(v)|tracks
distinct neighbor coverage.
To prevent extreme high-degree generic outliers from statistically
obfuscating the true semantic backbone, we implement a P90-robust
normalization strategy:
˜mk(v) =min(m k(v), P 90(mk))−mmin
k
P90(mk)−mmin
k(2)
wherek∈ {1, . . . ,5},mmin
k= min umk(u), andP 90is the 90th
percentile. The composite importance score is the uniform mean:
S(v) =1
55X
k=1˜mk(v)(3)

Documents
Books
ReportsChunk 1
Chunk 2
Chunk 3
Chunk n
...
LLM
Chain
CenterAssociations Insights Insight 
HypergraphPairwise
Beyond -PairwiseRelations Entities Fact HypergraphEntity -Relation Fact Hypergraph Index
Center -Chain Insight Hypergraph IndexHypergraph
DB
Vector DB
Fig. 1. The end-to-end operational workflow of the Dual-Hypergraph Indexing (DHI) framework. Source documents are iteratively segmented
into sequential sliding chunks. An LLM extracts multi-entity hyperedges equipped with atomic direct insights to construct the foundational
factual hypergraphH K(bottom). Dual pathways—topological hub aggregation (Path A) and temporal chunk-chain progressive aggregation
(Path B)—synthesize these assertions into high-order semantic correlations, populating the deep insight hypergraphH D(top). Bidirectional
mapping ensures rigorous cross-layer provenance, mitigating hallucinations during context assembly.
Entities are sorted in descending order ofS(v). We autonomously
determine the optimal hub inclusion boundary indexk∗via maxi-
mum first-order difference, bounded by a 20% system safety cap to
prevent insight dilution:
k∗= arg max
k 
S(vk)−S(v k+1)
,s.t.k≤0.20· |V K|(4)
For each identified hubv∗, an LLM synthesizes its entire incident
hyperedge setE(v∗)into a systemic hub insightI hub(v∗). The in-
sight hyperedgeε hub={π(e)|e∈E(v∗)}is instantiated within
HD.
3.3.2. Path B: Temporal Chunk-Chain Progressive Aggregation
While Path A extracts static radial clusters, Path B discovers longi-
tudinal narrative chains and chronologies. Letτ(c)∈N+denote
the absolute sequential position of chunkc. For any hyperedgee, its
temporal footprint isτ(e) ={τ(c)|c∈source(e)}.
We initialize potential analytical chains and iteratively expand
them via a greedy branching algorithm within a sliding windowW.
A candidate hyperedgee′is appended to an active evolving chain
X= (e 1, . . . , e L)if three strict conditions are met:


minτ(e′)−maxτ(e L)≤W,(Temporal Proximity)
|eL∩e′| ≥1,(Entity Overlap)
max
v∈V(X)Freq(v)<2
3,(Anti-Hub Penalty)(5)
whereV(X)denotes the collective entity union of the chain, and
Freq(v)is the fraction of hyperedges containingv.
Crucially, the2/3dominance threshold (Anti-Hub Penalty) ex-
plicitly guarantees that Path B remains mathematically orthogonalto Path A. It actively penalizes sequences that merely orbit a sin-
gle central entity, forcing the algorithm to trace true linear narrative
evolutions rather than topologically degenerating into localized hub
clusters. Chains satisfying minimum lengthL≥3are retained, and
redundant sub-chains are aggressively pruned. An LLM synthesizes
each valid chain intoI chain, forming a temporal hyperedgeε chainin
HD.
3.4. Cross-Layer Joint Retrieval and Context Assembly
Upon receiving a complex user queryq, DHI executes a highly co-
ordinated, multi-scale semantic retrieval cascade:
1.Factual Subgraph Retrieval:Compute dense query-entity
similaritys(v, q) =ev·q
∥ev∥∥q∥. Top-kscoring entities induce
an active factual hyperedge subsetE sub.
2.Topological Projection and Diffusion:Utilizing the deter-
ministic mappingπ, active hyperedges project to seed insight
verticesU sub={π(e)|e∈E sub,|e| ≥3}. A rapid 1-hop
traversal onH Dharvests deep insightsE∗.
3.Structured Context Formatting:Granular facts fromE sub
and overarching narrative summaries fromE∗are concate-
nated into a unified prompt contextP(q)for final generation.
4. EXPERIMENTAL EV ALUATION
4.1. Experimental Setup and Implementation Details
Datasets:We rigorously validate DHI across five multi-domain pub-
lic benchmarks testing basic retrieval to multi-step causal etiology:
Mix (multidisciplinary, 61 documents, 560 chunks), CS (Computer
Science, 10 docs, 1992 chunks), Agriculture (12 docs, 1813 chunks),

Table 1. Overall composite performance comparison across five
multi-domain benchmarks (0–100 scale, higher is better). Agri.,
Neuro., and Patho. denote Agriculture, Neurology, and Pathology,
respectively.
Method Mix CS Agri. Neuro. Patho.
LLM (Zero-shot) 79.30 81.08 79.64 81.15 82.81
NaiveRAG [2] 78.09 79.43 76.20 79.20 82.04
GraphRAG [5] 81.06 84.03 79.98 83.10 82.72
LightRAG [6] 81.01 81.25 79.05 81.82 84.43
HiRAG [21] 82.85 83.33 81.19 82.71 84.13
Hyper-RAG [9] 80.39 83.88 81.98 83.74 84.41
DHI (Ours) 83.18 84.66 82.97 84.23 85.78
Table 2. Granular capability breakdown evaluating five specific NLP
dimensions on the Mix benchmark.
Method Comp. Diver. Empo. Logi. Read.
LLM 85.40 73.80 73.76 81.54 82.00
NaiveRAG [2] 85.00 72.36 71.64 81.50 79.96
GraphRAG [5] 87.70 77.10 75.10 83.58 81.82
LightRAG [6] 88.00 78.10 74.82 82.38 81.76
HiRAG [21]90.0080.00 77.6683.94 82.64
Hyper-RAG [9] 84.10 78.48 74.68 83.16 81.54
DHI (Ours)88.98 80.2077.35 85.47 83.92
Neurology (medical textbook, 1790 chunks), and Pathology (clinical
causality, 824 chunks) [20].
Baselines:We benchmark DHI against six strong baselines: LLM
(zero-shot), NaiveRAG [2], GraphRAG [5], LightRAG [6], HiRAG
[21], and Hyper-RAG [9]. Systems universally deploy GPT-4o-mini
for generation andtext-embedding-3-smallfor embeddings
(T= 0). Chunking size is 500 tokens with 50-token overlap. DHI
setsW= 10. Evaluations follow LLM-as-a-Judge protocols across
five dimensions (0–100 scale).
4.2. Main Results and Domain Adaptability
As detailed in Table 1, DHI achieves state-of-the-art composite met-
rics across all domains. On the heavily scrutinized Mix benchmark,
DHI records 83.18, outperforming Hyper-RAG (80.39) by +2.79
absolute points and surpassing the advanced HiRAG architecture
(82.85).
Performance gains are most acutely notable in domains requiring
intensive deductive reasoning. On the Pathology dataset—where
medical causality spans multiple chapters—DHI achieves a leading
score of 85.78, outstripping LightRAG (84.43) and GraphRAG
(82.72) by 1.35 and 3.06 points. This firmly verifies that pre-
synthesizing higher-order topological insights provides highly reli-
able inductive support [22].
4.3. Dimensional Analysis and Logical Coherence
Table 2 provides a granular capability breakdown on the Mix
benchmark. DHI achieves undeniably superior results in Logicality
(85.47), Readability (83.92), and Diversity (80.20).
The substantial +1.53 margin in Logicality over HiRAG demon-
strates that mathematically presenting structured causal progressions
relieves the generator LLM from making unassisted inferential leapsover disjointed contexts [23]. While HiRAG achieves a higher Com-
prehensiveness score (90.00 vs. 88.98), it accomplishes this by
exhaustively concatenating macro-level community summaries, in-
advertently introducing massive non-essential context. Conversely,
DHI prioritizes strict structural relevance, deliberately avoiding
token bloat.
4.4. Ablation Study: The Vital Synergy of Dual Pathways
We evaluate ablative variants of the DHI architecture directly on
Mix:
•w/o Path A (No Hubs):Removing topological hub aggre-
gation causes the composite score to precipitously decline to
81.74, and Logicality drops to 82.80. Without spatial distilla-
tion, central themes remain fragmented.
•w/o Path B (No Temporal Chaining):Omitting sequence
chaining yields a suppressed composite of 82.25. This
proves that explicit sequential chaining is absolutely essential
for tracing longitudinal narrative evolutions across distant
chunks.
•Full DHI Architecture:Unifying both pathways effortlessly
attains the maximal 83.18, verifying that hub clustering and
sequential progression are entirely orthogonal yet mutually
reinforcing inductive biases.
4.5. Qualitative Case Study
Consider a multi-hop query in Pathology:“How does prolonged
exposure to Agent X subsequently trigger the cascade leading to
Syndrome Z?”Hyper-RAG retrieves isolated hyperedges (Agent X
causes degradation; degraded pathways cause Syndrome Z) from
distinct chapters. Under DHI, Path B proactively instantiates an over-
arching hyperedgeε chain∈H Dduring indexing that synthesizes
this exact etiology. During retrieval, the model instantly pulls this
verified chronological chain, providing a flawless, hallucination-free
logical bridge.
5. DISCUSSION AND LIMITATIONS
While DHI achieves robust reasoning coherence, it inherently re-
lies on the absolute precision of the LLM during the initial entity
extraction phase. Factual misinterpretations risk being algorithmi-
cally amplified intoH D. Future iterations must investigate dynamic
confidence-weighting algorithms and self-corrective feedback mech-
anisms [24] to mathematically guarantee the strict fidelity of synthe-
sized schemas.
6. CONCLUSION
We comprehensively expose and resolve the “Knowledge Island
Problem” bottlenecking existing hypergraph-based RAG architec-
tures. By innovatively decoupling knowledge representations into a
foundational factual hypergraphH Kand an elevated deep insight
hypergraphH D, Dual-Hypergraph Indexing (DHI) natively captures
both static topological prominence and highly dynamic chronologi-
cal evolutions. Through synergistic dual-pathway aggregation, DHI
synthesizes isolated facts into cohesive analytical insights, elevating
logical coherence by +1.53 points over current systems. Moving
forward, future research will actively investigate recursive insight
hierarchies and lightweight edge deployments to democratize robust
multi-hop reasoning.

7. REFERENCES
[1] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan
Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale
Fung, “Survey of hallucination in natural language genera-
tion,”ACM computing surveys, vol. 55, no. 12, pp. 1–38, 2023.
[2] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike
Lewis, Wen-tau Yih, Tim Rockt ¨aschel, et al., “Retrieval-
augmented generation for knowledge-intensive nlp tasks,”Ad-
vances in neural information processing systems, vol. 33, pp.
9459–9474, 2020.
[3] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini,
Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin,
Sebastian Riedel, and Edouard Grave, “Atlas: Few-shot learn-
ing with retrieval augmented language models,”Journal of Ma-
chine Learning Research, vol. 24, no. 251, pp. 1–43, 2023.
[4] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun, “Bench-
marking large language models in retrieval-augmented gener-
ation,” inProceedings of the AAAI conference on artificial in-
telligence, 2024, vol. 38, pp. 17754–17762.
[5] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex
Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky,
Robert Osazuwa Ness, and Jonathan Larson, “From local to
global: A graph rag approach to query-focused summariza-
tion,”arXiv preprint arXiv:2404.16130, 2024.
[6] Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and Chao
Huang, “Lightrag: Simple and fast retrieval-augmented gen-
eration.,” inEMNLP (Findings), 2025, pp. 10746–10761.
[7] Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and
Yue Gao, “Hypergraph neural networks,” inProceedings of
the AAAI conference on artificial intelligence, 2019, vol. 33,
pp. 3558–3565.
[8] Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue, Xingliang
Hou, Zhiqiang Tian, Yue Gao, and Shaoyi Du, “Cog-
rag: cognitive-inspired dual-hypergraph with theme alignment
retrieval-augmented generation,” inProceedings of the AAAI
Conference on Artificial Intelligence, 2026, vol. 40, pp. 31032–
31040.
[9] Yifan Feng, Hao Hu, Shihui Ying, Xingliang Hou, Shiquan
Liu, Mingyuan Yang, Junchang Li, Shaoyi Du, Nanning
Zheng, Han Hu, et al., “Hyper-rag: Combating llm halluci-
nations using hypergraph-driven retrieval-augmented genera-
tion,”Nature Communications, 2026.
[10] Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Kumar Roy,
Yu Zhang, Zheng Li, Ruirui Li, Xianfeng Tang, Suhang Wang,
Yu Meng, et al., “Graph chain-of-thought: Augmenting large
language models by reasoning on graphs,” inFindings of the
Association for Computational Linguistics: ACL 2024, 2024,
pp. 163–184.
[11] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis,
Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih,
“Dense passage retrieval for open-domain question answer-
ing,” inProceedings of the 2020 conference on empirical meth-
ods in natural language processing (EMNLP), 2020, pp. 6769–
6781.
[12] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and
Ashish Sabharwal, “Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step ques-
tions,” inProceedings of the 61st annual meeting of the asso-
ciation for computational linguistics (volume 1: long papers),
2023, pp. 10014–10037.
[13] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo,
Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau
Yih, “Replug: Retrieval-augmented black-box language mod-
els,” inProceedings of the 2024 conference of the north amer-
ican chapter of the association for computational linguistics:
Human language technologies (volume 1: Long papers), 2024,
pp. 8371–8384.
[14] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor
Cai, Eliza Rutherford, Katie Millican, George Bm Van
Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan
Clark, et al., “Improving language models by retrieving from
trillions of tokens,” inInternational conference on machine
learning. PMLR, 2022, pp. 2206–2240.
[15] Bernal J Guti ´errez, Yiheng Shu, Yu Gu, Michihiro Yasunaga,
and Yu Su, “Hipporag: Neurobiologically inspired long-term
memory for large language models,”Advances in neural infor-
mation processing systems, vol. 37, pp. 59532–59569, 2024.
[16] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian
Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Gra-
ham Neubig, “Active retrieval augmented generation,” inPro-
ceedings of the 2023 conference on empirical methods in nat-
ural language processing, 2023, pp. 7969–7992.
[17] Bahare Fatemi, Perouz Taslakian, David Vazquez, and David
Poole, “Knowledge hypergraphs: Prediction beyond binary re-
lations,”arXiv preprint arXiv:1906.00137, 2019.
[18] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang,
Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung Shum, and
Jian Guo, “Think-on-graph: Deep and responsible reasoning
of large language model on knowledge graph,” inInternational
Conference on Learning Representations, 2024, vol. 2024, pp.
3868–3898.
[19] Austin R Benson, “Three hypergraph eigenvector centralities,”
SIAM Journal on Mathematics of Data Science, vol. 1, no. 2,
pp. 293–312, 2019.
[20] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Ja-
son Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani,
Heather Cole-Lewis, Stephen Pfohl, et al., “Large language
models encode clinical knowledge,”Nature, vol. 620, no. 7972,
pp. 172–180, 2023.
[21] Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan,
Yongqiang Chen, Kaili Ma, Hongzhi Chen, and James Cheng,
“Retrieval-augmented generation with hierarchical knowl-
edge.,” inEMNLP (Findings), 2025, pp. 6044–6060.
[22] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran,
Karthik Narasimhan, and Yuan Cao, “React: Synergizing
reasoning and acting in language models,”arXiv preprint
arXiv:2210.03629, 2022.
[23] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Grif-
fiths, Yuan Cao, and Karthik Narasimhan, “Tree of thoughts:
Deliberate problem solving with large language models,”Ad-
vances in neural information processing systems, vol. 36, pp.
11809–11822, 2023.
[24] Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh
Hajishirzi, “Self-rag: Learning to retrieve, generate, and cri-
tique through self-reflection,” inInternational conference on
learning representations, 2024, vol. 2024, pp. 9112–9141.