# Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs

**Authors**: Su Dong, Qinggang Zhang, Yilin Xiao, Shengyuan Chen, Chuang Zhou, Xiao Huang

**Published**: 2026-02-03 14:26:28

**PDF URL**: [https://arxiv.org/pdf/2602.03578v1](https://arxiv.org/pdf/2602.03578v1)

## Abstract
Large language models (LLMs) often struggle with knowledge-intensive tasks due to hallucinations and outdated parametric knowledge. While Retrieval-Augmented Generation (RAG) addresses this by integrating external corpora, its effectiveness is limited by fragmented information in unstructured domain documents. Graph-augmented RAG (GraphRAG) emerged to enhance contextual reasoning through structured knowledge graphs, yet paradoxically underperforms vanilla RAG in real-world scenarios, exhibiting significant accuracy drops and prohibitive latency despite gains on complex queries. We identify the rigid application of GraphRAG to all queries, regardless of complexity, as the root cause. To resolve this, we propose an efficient and adaptive GraphRAG framework called EA-GraphRAG that dynamically integrates RAG and GraphRAG paradigms through syntax-aware complexity analysis. Our approach introduces: (i) a syntactic feature constructor that parses each query and extracts a set of structural features; (ii) a lightweight complexity scorer that maps these features to a continuous complexity score; and (iii) a score-driven routing policy that selects dense RAG for low-score queries, invokes graph-based retrieval for high-score queries, and applies complexity-aware reciprocal rank fusion to handle borderline cases. Extensive experiments on a comprehensive benchmark, consisting of two single-hop and two multi-hop QA benchmarks, demonstrate that our EA-GraphRAG significantly improves accuracy, reduces latency, and achieves state-of-the-art performance in handling mixed scenarios involving both simple and complex queries.

## Full Text


<!-- PDF content starts -->

Use Graph When It Needs: Efficiently and Adaptively Integrating
Retrieval-Augmented Generation with Graphs
Su Dong
su.dong@connect.polyu.hk
The Hong Kong Polytechnic
University
Hung Hom, Kowloon, Hong KongQinggang Zhang
qinggang.zhang@polyu.edu.hk
The Hong Kong Polytechnic
University
Hung Hom, Kowloon, Hong KongYilin Xiao
The Hong Kong Polytechnic
University
Hung Hom, Kowloon, Hong Kong
Shengyuan Chen
The Hong Kong Polytechnic
University
Hung Hom, Kowloon, Hong KongChuang Zhou
The Hong Kong Polytechnic
University
Hung Hom, Kowloon, Hong KongXiao Huang
xiao.huang@polyu.edu.hk
The Hong Kong Polytechnic
University
Hung Hom, Kowloon, Hong Kong
Abstract
Large language models (LLMs) often struggle with knowledge-
intensive tasks due to hallucinations and outdated parametric knowl-
edge. While Retrieval-Augmented Generation (RAG) addresses this
by integrating external corpora, its effectiveness is limited by frag-
mented information in unstructured domain documents. Graph-
augmented RAG (GraphRAG) emerged to enhance contextual rea-
soning through structured knowledge graphs, yet paradoxically
underperforms vanilla RAG in real-world scenarios, exhibiting sig-
nificant accuracy drops and prohibitive latency despite gains on
complex queries. We identify the rigid application of GraphRAG
to all queries, regardless of complexity, as the root cause. To re-
solve this, we propose an efficient and adaptive GraphRAG frame-
work called EA-GraphRAG that dynamically integrates RAG and
GraphRAG paradigms through syntax-aware complexity analysis.
Our approach introduces: (i) a syntactic feature constructor that
parses each query and extracts a set of structural features; (ii) a light-
weight complexity scorer that maps these features to a continuous
complexity score; and (iii) a score-driven routing policy that selects
dense RAG for low-score queries, invokes graph-based retrieval
for high-score queries, and applies complexity-aware reciprocal
rank fusion to handle borderline cases. Extensive experiments on a
comprehensive benchmark, consisting of two single-hop and two
multi-hop QA benchmarks, demonstrate that our EA-GraphRAG
significantly improves accuracy, reduces latency, and achieves state-
of-the-art performance in handling mixed scenarios involving both
simple and complex queries.
Keywords
Retrieval-Augmented Generation, Graph-based Retrieval-Augmented
Generation, Query Complexity, Model Efficiency
1 Introduction
Large Language Models (LLMs) have significantly advanced the
field of natural language processing, yet their tendency to generate
inaccurate or outdated information remains a critical challenge.
To mitigate this, Retrieval-Augmented Generation (RAG) [ 4,20]
has emerged as a prominent technique, enriching LLMs with real-
time access to external knowledge sources. By retrieving relevant
Simple caseQ: when is season 8 for game of thrones?QueriesGold answers
A: 2019GraphRAGRAG
A: Not specified in the text.
A:2019
Multi-hop caseQ:Whichfilmhasthedirectorwhoisolderthantheother,Women‘SWeaponsorSheWantsMe?
A:Women'SWeaponsGraphRAGRAG
A:Women'sWeapons 
A: Cannot determineResponses
Figure 1: An illustrative example showing that GraphRAG of-
ten underperforms vanilla RAG on simple single-hop queries,
but excels on multi-hop queries requiring relational rea-
soning; conversely, vanilla RAG tends to perform better on
single-hop queries while struggling with multi-hop reason-
ing.
text passages from the external corpus using semantic matching,
RAG grounds model responses in verifiable evidence, significantly
improving accuracy for knowledge-intensive tasks.
Despite the success, traditional RAG systems often fall short
when faced with complex tasks that require reasoning across mul-
tiple pieces of interconnected information, such as those involving
causal relationships or multi-step logical inferences. To this end,
graph retrieval-augmented generation (GraphRAG) [ 25,26,36] has
recently emerged as a powerful paradigm that leverages external
structured graphs to enable deep retrieval and contextual compre-
hension. GraphRAG can effectively answer questions about indi-
rect influences or causal chains by exploring connections between
nodes. Early GraphRAG approaches established the core method-
ology. Microsoft GraphRAG [ 3] pioneered the use of a globally
constructed knowledge graph, employing community detection
algorithms to identify thematic clusters and enabling comprehen-
sive querying through a combination of local (entity-specific) andarXiv:2602.03578v1  [cs.CL]  3 Feb 2026

Conference, Under Review, Dong et al.
global (community-level) searches. Building on this foundation,
RAPTOR [ 30] introduced a recursive, tree-based indexing strategy
as an alternative to graphs, creating a hierarchy of summarized
text chunks to facilitate multi-scale reasoning without relying on
explicit entity-relationship extraction. Subsequent research focused
on enhancing the efficiency and biological plausibility of retrieval.
HippoRAG [ 7] and its successor HippoRAG2 [ 8] marked a sig-
nificant departure by modeling human memory processes, using
a neurobiologically-inspired framework that integrates an LLM
with a knowledge graph and the Personalized PageRank algorithm
for efficient, single-step associative recall. Parallel efforts aimed
at optimization led to LightRAG [ 5], which maintains a graph’s
relational benefits while drastically improving scalability through a
dual-level retrieval system and a more lightweight, graph-enhanced
indexing process. Collectively, these strategies in the GraphRAG
lineage have significantly advanced retrieval precision and contex-
tual depth, empowering LLMs to tackle intricate, multi-hop queries
more effectively than ever before.
However, GraphRAG models still suffer from several key issues
in real-world scenarios. (i) Low efficiency: Most existing GraphRAG
models rely heavily on LLM-based graph construction, which in-
curs significant token costs and update latency, making them im-
practical for large-scale or dynamically evolving knowledge bases.
(ii) Poor generalizability: GraphRAG exhibits poor generalizability
on simpler tasks. While GraphRAG outperforms RAG models on
complex reasoning tasks, they frequently underperform traditional
RAG approaches on many simple NLP tasks [ 9,37]. Empirical evi-
dence [ 33,37] shows that GraphRAG underperforms vanilla RAG
on single-hop question-answering, achieving 13.4% lower accuracy
on Natural Questions and a 16.6% drop for time-sensitive queries.
It is because while graph-based retrieval improves reasoning depth,
it also introduces noise and ambiguities into the retrieved contexts.
These observations highlight a fundamental trade-off: while
GraphRAG excels in complex reasoning, its inefficiencies and limi-
tations in handling simple queries make it suboptimal as a universal
solution. An ideal system should dynamically route queries to the
appropriate paradigm based on reasoning complexity, i.e., employ-
ing RAG for simplicity and efficiency, and GraphRAG for depth
and complexity. However, it is hard to (i) accurately and efficiently
quantify query complexity, and (ii) build a lightweight mechanism
to perform this routing.
To bridge this gap, we introduce EA-GraphRAG, a novel frame-
work that adaptively integrates RAG with graphs by effectively
evaluating query syntax complexity. Our approach consists of three
key components: (i) a syntactic feature constructor that parses each
query and extracts structural features, (ii) a lightweight complexity
scorer that maps these features to a continuous complexity score,
and (iii) a score-driven routing policy that selects dense RAG for
low-complexity queries, invokes GraphRAG for high-complexity
queries, and applies complexity-aware fusion for borderline cases.
This syntax-aware strategy maximizes efficiency while preserving
reasoning depth. Extensive experiments on two single-hop and two
multi-hop QA benchmarks verify EA-GraphRAG’s effectiveness in
balancing accuracy, latency, and contextual fidelity. To summarize,
our contribution is listed as follows:•We identify effectiveness and efficiency limitations in ex-
isting GraphRAG methods for simple single-hop QA tasks.
Based on these findings, we propose EA-GraphRAG, a novel
framework that dynamically integrates RAG with graphs by
evaluating query syntactic complexity.
•EA-GraphRAG introduces a lightweight query complexity
analyzer to evaluate the complexity of user queries.
•Based on the syntax complexity, EA-GraphRAG designs an
adapter that routes simple queries to vanilla RAG while
reserving GraphRAG for complex reasoning.
•Extensive experiments on comprehensive benchmarks con-
sisting of both single-hop QA datasets and multi-hop QA
datasets demonstrate that EA-GraphRAG outperforms the
state-of-the-art baselines.
2 Related Work
2.1 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG) has emerged as a key par-
adigm to address the limitations of large language models (LLMs)
in knowledge-intensive tasks, enabling access to external corpora
to enhance response accuracy and reduce hallucinations [ 4,20].
Early advancements in RAG focused on optimizing retrieval mech-
anisms: BM25 [ 28] leverages term frequency statistics for efficient
text matching, while dense retrievers like Contriever [ 15] and Col-
BERTv2 [ 29] use contextual embeddings to improve semantic align-
ment. More recent methods, such as IRCoT [ 31], integrate retrieval
with chain-of-thought reasoning to handle multi-step queries, and
Atlas [ 16] enhances few-shot learning by aligning retrieval with
in-context examples. However, traditional RAG systems struggle
with large-scale, unstructured domain corpora, where information
fragmentation and loss of contextual relationships during chunking
compromise performance on complex reasoning tasks [10, 36].
2.2 Graph Retrieval-Augmented Generation
To mitigate RAG’s limitations in contextual comprehension, Graph-
based RAG integrates structured knowledge graphs to model con-
ceptual relationships [ 25,26], enhancing reasoning depth. Key
innovations include hierarchical community-based search in Mi-
crosoft GraphRAG [ 3] and its variant LazyGraphRAG [ 1], which
combine local and global querying for comprehensive responses.
LightRAG [ 5] improves scalability through dual-level retrieval and
graph-enhanced indexing, while GRAG [ 13] introduces soft pruning
to reduce irrelevant entities and uses graph-aware prompt tuning
to aid LLM interpretation of topological structures. StructRAG [ 21]
dynamically selects optimal graph schemas for specific tasks, and
KAG [ 22] constructs domain expert knowledge using conceptual
semantic reasoning and human-annotated schemas, reducing noise
from OpenIE systems. Despite these advances, GraphRAG para-
doxically underperforms vanilla RAG on real-world tasks, with
significant accuracy drops on simple queries and prohibitive la-
tency due to its rigid application across all query types [9, 37].
2.3 Query Complexity Measurement
Quantifying query complexity is crucial for adaptive retrieval strat-
egy selection. Existing approaches [ 9,17] typically define query

Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs Conference, Under Review,
(a)Syntactic Parsing
(b)FeatureConstruction with TregexROOTSBARQSVPNPNPPPNPWHNPWhoVBZisDTtheINforDTtheROOTSBARQWHADVPWRBWhereSQNPVBDdidVPPPVBgraduateINfromNPNPNNPCharlesNNPStewartNPCD3NNPRdNNPDateINOfPPNPNPNNfatherNNPRichmondPOS’sROOTSBARQSQNPWHADVPVBD…………NNcoachNNPottawaNNSsenators(ottawa senators, coach, ?)(Charles Stewart, father, ?, graduate from, ?)Q: who is the coachfor the ottawasenators?Simple caseQ: Where did Charles Stewart, 3Rd Duke Of Richmond's fathergraduate from?Multi-hop case
WSCDCTCTCPCNVPFeaturesType 1: Length of production unit•Mean length of clause•…Type 2:Sentence complexity•Sentence complexity ratio•…1st hop1st hop2nd hop
Syntactic FeaturesSyntactic treePattern match by tregexproduction units and syntactic structures
Query
CorpusLLM
Vectordatabase
Finalresponse
RAGSearch
Similardocuments
GraphRAG
Corpus
Entitiesandrelationships
Querygraphdatabase
Searchand traversalLLM
Finalresponse
LLM GraphGenerator
ContextNodes&re-ships
Adapter
(c)Decision Module
Complexityprobability
Fusion⓵𝒔𝒒≥𝝉𝑯
LLM
Finalresponse
⓵⓶⓷
⓶𝒔𝒒<𝝉𝑳⓷𝝉𝑳≤𝒔𝒒<𝝉𝑯
Figure 2: The overview of our EA-GraphRAG framework. In the (c) Decision Module stage, an MLP adapter produces ascalar
complexity score 𝑠(𝑞)∈( 0,1). Two thresholds route the query: 𝑠(𝑞)<𝜏𝐿⇒Dense retrieval; 𝑠(𝑞)≥𝜏𝐻⇒Graph-based retrieval;
𝜏𝐿≤𝑠(𝑞)<𝜏 𝐻⇒Fusion, where the documents retrieved by dense retrieval and graph-based retrieval are merged by weighted
Reciprocal Rank Fusion (wRRF).
complexity from a semantic perspective, either by training ded-
icated language models or by leveraging large language models
to classify queries into different complexity levels. Although ef-
fective, these methods require extensive annotated data and incur
high computational costs, making them inefficient for large-scale
systems.
In contrast, the field of linguistics provides a complementary
perspective: syntactic complexity has long been studied as a mea-
surable property of sentences, reflecting the structural difficulty of
language. For instance, Lu [23] proposed 14 syntactic complexity
measures based on clauses, T-units, and complex nominals, which
have been validated for distinguishing language proficiency. In-
spired by this, we posit that syntactic features offer an efficient
and low-cost alternative for modeling query complexity. Therefore,
we train a lightweight classifier using syntactic features to pre-
dict query complexity, providing a scalable foundation for adaptive
routing between retrieval strategies.
Limitations of Existing Approaches.Current methods suffer from
two critical limitations: (1) static application of RAG or GraphRAG
regardless of query complexity, leading to inefficiencies on simple
tasks and underperformance on complex ones; (2) lack of integra-
tion between syntactic complexity metrics and adaptive strategies
selection mechanisms. This study addresses these gaps with EA-
GraphRAG, which dynamically routes queries based on syntax-
aware complexity analysis, bridging RAG and GraphRAG to opti-
mize both accuracy and latency.3 EA-GraphRAG
We consider the retrieval-augmented question answering with a
complexity-aware tri-routing architecture. Given a question 𝑞in
natural language, the system retrieves relevant passages from a
corpusC={𝑐 1,𝑐2,...,𝑐𝑁}, where𝑁is the total number of passages
and each𝑐𝑖is a text passage that can be independently indexed
and retrieved. Retrieved passages serve as evidence for an LLM
generatorGto produce answers.
Our framework provides three retrieval paths: (i) adenseretriever
Rragthat returns a ranked list 𝜋rag(𝑞)⊂C of passages ordered by
semantic similarity, (ii) agraph-basedretriever Rgroperating on
a pre-built graph 𝐺=(𝑉,𝐸) overC, which returns 𝜋gr(𝑞)⊂C by
combining structure-aware signals with semantic similarity, and (iii)
acomplexity-aware fusionoperator Ffusthat merges two ranked lists
via weighted Reciprocal Rank Fusion (wRRF), producing 𝜋fus(𝑞)=
Ffus 𝜋rag(𝑞),𝜋 gr(𝑞),𝑠(𝑞), where fusion weights are determined by
the complexity score𝑠(𝑞).
Routing among these paths is controlled by a decision module.
A featurizer 𝜙(𝑞)∈R𝑑extracts syntactic-complexity features from
the query, where 𝑑is the feature dimensionality. An MLP adapter 𝑔𝜃
with feature attention and residual connections maps these features
to a complexity score 𝑠(𝑞) ∈ ( 0,1), where𝜃denotes the model
parameters. If 𝑠(𝑞)≥𝜏𝐻, the graph-based retrieval path is acti-
vated. If𝑠(𝑞)≤𝜏𝐿, the dense retrieval path is chosen. Otherwise,
the complexity-aware fusion path is selected, where contributions
from dense and graph retrieval are weighted by 𝑠(𝑞) and1−𝑠(𝑞) ,
respectively.
Finally, the answer is generated by a fixed LLM Ggiven𝑞and the
packed context. This formulation separates evidence units, retrieval

Conference, Under Review, Dong et al.
operators, and a learnable complexity-aware router that selects the
most appropriate path per query, enabling multi-hop reasoning
when needed with smooth interpolation between methods in the
fusion regime.
3.1 Offline Indexing
Following [ 8], we index the corpus Cas a heterogeneous graph 𝐺=
(𝑉,𝐸) . The node set 𝑉=𝑁∪C , where𝑁contains noun phrases and
entities extracted from Cvia a two-stage LLM prompting strategy.
The edge set 𝐸includes three types: relation edges from OpenIE
triples(ℎ,𝑟,𝑡) within𝑁, occurrence edges linking each entity 𝑛𝑖∈
𝑁to its source passage 𝑐𝑗∈ C, and synonymy edges between
nodes𝑛,𝑣∈𝑁 when cos(𝑀(𝑛),𝑀(𝑣))≥𝜏. This structure enables
efficient multi-hop retrieval while preserving passage-level context.
Hyperparameters follow default settings in [8].
3.2 Query Complexity Measurement
To quantify query complexity for routing decisions, we design a
featurizer that captures syntactic, structural, semantic, and lexical
properties. Our approach is based on the principle that query com-
plexity manifests through multiple dimensions: structural depth
through nested clauses, dependency relationships via long-distance
connections, semantic richness measured by entity density, and
lexical diversity.
3.2.1 Syntactic Structure Features.We extract features from con-
stituency parse trees using Stanza [ 27] that capture hierarchical
structure. We identify nine fundamental units using Tregex pattern
matching [ 19]: words W, sentences S, clauses C, dependent clauses
DC, T-units T, complex T-units CT, coordinate phrases CP, complex
nominals CN, and verb phrases VP. From these counts, we compute
ratio-based features including mean length measures MLS=W/S
and MLT=W/T, sentence complexity C/S, subordination ratios C/T,
CT/T, DC/C, and DC/T, coordination ratios CP/C, CP/T, and T/S,
and phrasal sophistication CN/C, CN/T, and VP/T.
3.2.2 Dependency-based Structural Features.From dependency
parse trees using SpaCy, we extract features capturing syntactic re-
lationships: maximum and average dependency distances, counts of
long-range dependencies with distance greater than 5, relation type
counts including subject-verb, object-verb, modifier, coordination,
and subordination relations, and tree imbalance metrics.
3.2.3 Semantic and Lexical Features.We extract named entity counts
and types including person, organization, location, and date, entity
density, semantic role indicators such as agents, patients, tempo-
ral, and locative roles, lexical diversity measured by unique token
ratio, content-to-function word ratios, information density met-
rics, question-type indicators, and complexity markers including
coordination, subordination, negation, and passive voice.
3.2.4 Tree Structure and Interaction Features.From the dependency
tree, we extract global properties: maximum depth, maximum width,
leaf-to-non-leaf ratios, branching factors, and depth-to-width ratios.
We also compute interaction terms combining multiple dimensions,
such as tokens per clause, entities per token, depth per token, and
connectors per clause.3.2.5 Feature Construction and Selection.The featurizer 𝜙(𝑞) as-
sembles all feature categories into a 𝑑′-dimensional vector, where
𝑑′is the raw feature dimensionality before selection and typically
exceeds 80. We apply mutual information-based feature selection
to identify the top- 𝑘most discriminative features for predicting
whether GraphRAG outperforms RAG, where 𝑘is the number of
selected features and typically equals 85. The final feature vector
x∈R𝑑has dimensionality 𝑑=𝑘 and is standardized using z-score
normalization: ˜x=( x−𝝁)⊘𝝈 , where 𝝁∈R𝑑and𝝈∈R𝑑are
the mean and standard deviation vectors estimated on the training
data, and⊘denotes element-wise division.
3.3 MLP Adapter
Given the𝑑-dimensional feature vectorx =𝜙(𝑞) , the adapter pro-
duces a complexity score 𝑠(𝑞) ∈ ( 0,1)that controls tri-routing.
We implement 𝑠(𝑞) with a lightweight MLP trained as a binary
classifier predicting whether GraphRAG will outperform RAG on a
query.
Architecture.Our MLP incorporates feature attention and resid-
ual connections. The model first applies feature-level attention to
the input, then passes through hidden layers with dimensions 256,
128, and 64, with residual connections where dimensions match.
The final hidden representation is processed through an output
attention layer before computing the logit ℓ𝜃(x), whereℓ𝜃denotes
the MLP function parameterized by𝜃. The complexity score is:
𝑠(𝑞)=𝜎(ℓ 𝜃(𝜙(𝑞)))∈(0,1),(1)
where𝜎(𝑡)= 1/(1+𝑒−𝑡)is the sigmoid function that maps the
logit to a probability in(0,1).
Training.We form a disagreement training set T={𝑖 :𝑧rag
𝑖⊕
𝑧gr
𝑖=1}, where𝑧rag
𝑖,𝑧gr
𝑖∈{0,1}are binary indicators of whether
RAG and GraphRAG answer sample 𝑖correctly as measured by
automatic QA metrics, and ⊕denotes exclusive OR. Each sample
𝑖∈T is labeled𝑦𝑖=1if GraphRAG is correct and RAG is incorrect,
and𝑦𝑖=0if RAG is correct and GraphRAG is incorrect. Samples
where both methods agree are excluded from training. We minimize
BCE-with-logits loss with label smoothing using AdamW optimizer
with cosine annealing learning rate schedule.
Routing.At inference, 𝑠(𝑞) is mapped to a path via two thresh-
olds𝜏𝐿<𝜏𝐻tuned on a validation set:
route(𝑞)= 
GraphRAG, 𝑠(𝑞)≥𝜏 𝐻,
RAG, 𝑠(𝑞)≤𝜏 𝐿,
Fusion, 𝜏 𝐿<𝑠(𝑞)<𝜏 𝐻,(2)
where𝜏𝐿is the low threshold and 𝜏𝐻is the high threshold. In the
fusion region, we use complexity-aware RRF where weights are
𝑤GraphRAG =𝑠(𝑞) and𝑤RAG=1−𝑠(𝑞) , with𝑤GraphRAG and𝑤RAG
denoting the weights for graph and dense retrieval contributions,
respectively. Thresholds are selected to maximize a composite ob-
jective of validation accuracy and expected latency.

Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs Conference, Under Review,
Algorithm 1The inference pipeline of our EA-GraphRAG framework
Require: query𝑞; thresholds(𝜏𝐿,𝜏𝐻); graph𝐺=(𝑉,𝐸) where
𝑉=𝑁∪C ; fact setF; encoder𝑀; damping𝛼; retrieved chunks
𝐾; RRF constant𝑘; generatorG
Ensure:answer ˆ𝑦
1:Extract features:x←𝜙(𝑞)
2:Compute complexity score:𝑠(𝑞)←𝜎(ℓ 𝜃(𝜙(𝑞)))∈(0,1)
3:if𝑠(𝑞)≥𝜏 𝐻then⊲Graph-based retrieval
4:Compute fact similarities:𝑠 𝑖←⟨𝑀(𝑞),𝑀(𝑓 𝑖)⟩for𝑓𝑖∈F
5: Select and rerank top- 𝑘facts:Ftop←Rerank(TopK( ˜s,𝑘),𝑞)
6: Compute entity weights from facts: 𝑤(𝑛)←˜𝑠𝑓
|{𝑐:(𝑛,𝑐)∈𝐸}|for
𝑛∈𝑁
7:Select seed entities:𝑄←TopK({𝑤(𝑛):𝑛∈𝑁},𝑘′)
8: Initialize PPR:r 0[𝑣] ←𝑤(𝑣) if𝑣∈𝑄 , else0;r←
PPR(𝐺,r 0,𝛼)
9:Extract passage scores and rank:𝐿←TopK(r[C],𝐾)10:else if𝑠(𝑞)≤𝜏 𝐿then⊲Dense retrieval
11:Encode query:q←𝑀(𝑞)
12: Compute similarities and rank: 𝐿←TopK({⟨ q,𝑀(𝑐)⟩ :𝑐∈
C},𝐾)
13:else⊲Complexity-aware fusion
14: Dense retrieval:q ←𝑀(𝑞) ;𝐿𝑟←TopK({⟨ q,𝑀(𝑐)⟩ :𝑐∈
C},𝐾)
15: Graph retrieval: compute Ftop,𝑄,ras in graph path; 𝐿𝑔←
TopK(r[C],𝐾)
16:Compute fusion weights:𝑤 gr←𝑠(𝑞);𝑤 rag←1−𝑠(𝑞)
17: Weighted RRF: for 𝑐∈𝐿𝑟∪𝐿𝑔,RRF(𝑐)←𝑤 rag·1
𝑘+𝑟𝑟(𝑐)+
𝑤gr·1
𝑘+𝑟𝑔(𝑐)
18:Fused ranking:𝐿←TopK({RRF(𝑐):𝑐∈𝐿 𝑟∪𝐿𝑔},𝐾)
19:end if
20:Pack context and generate: ˆ𝑦←G(𝑞,Pack(𝐿))
21:return ˆ𝑦
3.4 Online Retrieval
3.4.1 Dense Retrieval (RAG path).We encode the query and docu-
ments using encoder 𝑀, which maps text to dense vector embed-
dings. The similarity between query 𝑞and passage 𝑐is computed
as the inner product of their embeddings: ⟨𝑀(𝑞),𝑀(𝑐)⟩ , where
⟨·,·⟩ denotes the dot product. We retrieve the top- 𝐾passages with
highest similarity scores:
𝜋rag(𝑞)=TopK𝑐∈C⟨𝑀(𝑞),𝑀(𝑐)⟩,(3)
where𝐾is the number of passages to retrieve, and TopK returns
the𝐾passages with highest scores.
3.4.2 Graph-Based Retrieval (GraphRAG path).We perform re-
trieval over the heterogeneous graph 𝐺=(𝑉,𝐸) constructed during
offline indexing, where 𝑉=𝑁∪C with𝑁denoting the set of noun
phrases and entities, and Cthe corpus passages. The edge set 𝐸com-
prises relation edges from OpenIE triples, occurrence edges linking
entities to their source passages, and synonymy edges connecting
semantically similar entities.
Given a query 𝑞, we first compute similarity scores between the
query and all facts (triples) extracted from the corpus:
𝑠𝑖=⟨𝑀(𝑞),𝑀(𝑓 𝑖)⟩,∀𝑓𝑖∈F,(4)
whereFdenotes the set of all facts, 𝑀is the encoder for fact-query
matching,𝑓𝑖represents the 𝑖-th fact. The similarity scores are nor-
malized to obtain ˜s=[ ˜𝑠1,˜𝑠2,..., ˜𝑠|F|]⊤, where ˜𝑠𝑖is the normalized
similarity score for fact𝑓 𝑖.
We then select the top- 𝑘facts based on their similarity scores
and apply an LLM-based reranker to refine the selection:
Ftop=Rerank TopK( ˜s,𝑘),𝑞,(5)
where𝑘is a hyperparameter controlling the number of candidate
facts, TopK( ˜s,𝑘) returns the top- 𝑘facts with highest normalized
similarity scores, and Rerank uses an LLM to rerank these candi-
dates based on their relevance to query𝑞.
We then assign weights to entity nodes based on the selected
facts. For each fact 𝑓=(ℎ,𝑟,𝑡)∈F top, we extract its head entity
ℎand tail entity 𝑡, and locate the corresponding entity nodes inthe graph. For each entity node 𝑛∈𝑁 that appears inFtop, we
assign a weight computed as the normalized similarity score of the
fact containing the entity, divided by the number of passages that
contain this entity. This normalization prevents hub entities that
appear in many passages from dominating the ranking. If an entity
appears in multiple facts during iteration, we update its weight
using the similarity score from the most recently encountered fact.
After processing all facts, we select the top- 𝑘′entity nodes with
the highest weights to form the seed set 𝑄⊂𝑁 , where𝑘′is a
hyperparameter controlling the number of seed entities.
We initialize the reset probability vectorr 0for Personalized
PageRank by assigning the computed weights to entity nodes in
𝑄, while setting all passage nodes in Cto zero. This initialization
strategy ensures that relevance signals originate exclusively from
the selected entity nodes and propagate through the graph struc-
ture via relation edges connecting related entities and synonymy
edges connecting semantically similar entities, eventually reaching
passage nodes through occurrence edges that link entities to their
source passages.
We then perform Personalized PageRank with reset probability
r0:
r=PPR(𝐺,r 0,𝛼),(6)
wherer∈R|𝑉|is the resulting diffusion vector, PPR denotes the
Personalized PageRank algorithm, and 𝛼∈( 0,1)is the damping
factor controlling the teleportation probability, which is the prob-
ability of resetting to seed nodes. The PPR algorithm propagates
weights from seed entity nodes through relation edges and syn-
onymy edges within 𝑁, and then diffuses to passage nodes in Cvia
occurrence edges.
Finally, we extract relevance scores for passage nodes directly
from the diffusion vectorrand select the top-𝐾passages:
𝜋gr(𝑞)=TopK r[C],𝐾,(7)
wherer[C]denotes the subvector ofrcorresponding to passage
nodes, which are the entries for nodes in C, and𝐾is the number of

Conference, Under Review, Dong et al.
passages to retrieve. The selected passages are then passed to the
generator for answer synthesis.
3.4.3 Fusion for Uncertain Cases (Fusion path).When route(𝑞)=
Fusion , we fuse𝜋ragand𝜋grusing complexity-aware RRF. The
weighted RRF score for document𝑐is:
RRF(𝑐)=(1−𝑠(𝑞))·1
𝑘+𝑟 rag(𝑐)+𝑠(𝑞)·1
𝑘+𝑟 gr(𝑐),(8)
where𝑟rag(𝑐)and𝑟gr(𝑐)are the 1-based ranks of document 𝑐in
the dense retrieval list 𝜋rag(𝑞)and graph retrieval list 𝜋gr(𝑞), re-
spectively. If 𝑐does not appear in a list, its rank contribution is
zero. Here𝑘> 0is the RRF smoothing constant, and 𝑠(𝑞) is the
complexity score. Higher complexity queries with 𝑠(𝑞) closer to 1
give more weight to graph retrieval, while lower complexity queries
with𝑠(𝑞)closer to 0 favor dense retrieval. The fused list is:
𝜋fus(𝑞)=TopK𝑐∈𝜋 rag(𝑞)∪𝜋 gr(𝑞)RRF(𝑐),(9)
where the union 𝜋rag(𝑞)∪𝜋 gr(𝑞)contains all documents appearing
in either list, and TopK returns the𝐾documents with highest RRF
scores.
Final evidence set and answer generation.The evidence set 𝐷(𝑞)
is determined by the routing decision:
𝐷(𝑞)= 
𝜋rag(𝑞),ifroute(𝑞)=RAG,
𝜋gr(𝑞),ifroute(𝑞)=GraphRAG,
𝜋fus(𝑞),ifroute(𝑞)=Fusion,(10)
where𝐷(𝑞)⊂C is the set of retrieved passages. The final answer
is generated by:
ˆ𝑦=G(𝑞,Pack(𝐷(𝑞))),(11)
whereGis a fixed LLM generator, Pack(𝐷(𝑞)) packs the passages
in𝐷(𝑞) into a context string with truncation to fit the model’s input
length budget, and ˆ𝑦is the generated answer.
4 Experiments
In this section, we conduct comprehensive experiments on four
benchmark datasets to verify the effectiveness and efficiency of our
framework.
4.1 Baselines
We selected three categories of models as our baselines: (1) Popular
large language models, including Llama-3-8B [ 2], Qwen3-8B [ 34],
GPT-4o-mini [ 14], and GPT-3.5-turbo. (2) Strong and widely used re-
trieval methods to form RAG baselines: BM25 [ 28], Contriever [ 15],
and ColBERTv2 [ 29]. (3) Several popular and strong GraphRAG
baselines, including RAPTOR [ 30], G-retriever [ 11], LightRAG [ 6],
KGP [32], HippoRAG [7] and HippoRAG2 [8].
4.2 Datasets
To simulate realistic scenarios with queries of varying complexity
levels, we employ both single-hop and multi-hop QA datasets in
our experiments. Single-hop QA datasets effectively evaluate a RAG
system’s information retrieval and question answering capabilities
for factual queries, while multi-hop QA datasets assess its reasoning
ability when processing complex queries requiring multi-document
evidence. To emulate real-world scenarios that contain a mixtureof both simple and complex questions, we constructed a mixed
benchmark comprising two single-hop and two multi-hop datasets
used in our paper. This experimental setup comprehensively reveals
the comparative strengths and limitations of RAG and GraphRAG,
reflecting their performance in realistic scenarios. Table 2 presents
the statistical characteristics of our sampled dataset.
4.2.1 Single-hop QA.For single-hop queries, we employ two widely
adopted benchmarks: Natural Questions (NQ) [ 18] and PopQA [ 24].
Following the data processing methodology of [ 8], we randomly
sample 1,000 queries and corresponding passages from each bench-
mark, ensuring a balanced and representative subset. The NQ
dataset provides real user queries with a wide range of topics, while
PopQA offers open-domain queries, covering diverse factual re-
trieval scenarios.
4.2.2 Multi-hop QA.For multi-hop queries, we select 2WikiMulti-
hopQA [ 12] and HotpotQA [ 35], both established as gold standards
for evaluating complex reasoning capabilities of QA systems. From
each dataset, we extract 1,000 queries and corresponding passages
that explicitly require multi-passage reasoning, where answers
must be synthesized across at least two distinct documents.
For the mixed benchmark, we pool the above datasets, uniformly
mix equal numbers of single-hop (NQ/PopQA) and multi-hop (Hot-
potQA/2Wiki) queries, and merge their corpus to emulate a real-
world query distribution.
4.3 Metrics
Evaluation is conducted from both answer accuracy and retrieval
performance perspectives. For answer evaluation, exact string match-
ing metrics, while widely adopted in QA evaluation, can be overly
rigid since variations in casing, tense, grammar, or paraphrasing
may incorrectly penalize semantically correct answers. We employ
two complementary metrics that jointly assess surface fidelity and
semantic correctness:
Contain-Match Accuracy, which measures whether the gold an-
swer appears as a substring within the model’s prediction, allowing
minor surface-form variations while maintaining semantic preci-
sion.
GPT-Evaluation Accuracy, an LLM-based criterion in which the
evaluator receives the question, gold answer, and prediction, and
determines whether the predicted answer is semantically equivalent
to the reference.
Together, these metrics offer a balanced and interpretable assess-
ment by combining the reproducibility of string matching with the
nuanced semantic understanding of LLM-based judgment.
For retrieval evaluation, we report two complementary metrics.
Retrieval Time per Query (s) quantifies efficiency by measuring the
average time the retriever needs to return candidate documents,
reflecting end-to-end retrieval latency. Recall@k assesses quality as
the proportion of queries for which at least one relevant document
appears among the top-k results. Together, these metrics character-
ize the accuracy–efficiency trade-off and offer a more holistic view
of system performance.

Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs Conference, Under Review,
Table 1: QA performance (Acc. and GPT-Acc.). Best per column in bold; second-best underlined .
Simple QA Multi-Hop QAMix
Method NQ PopQA HotpotQA 2Wiki
Acc. GPT-Acc. Acc. GPT-Acc. Acc. GPT-Acc. Acc. GPT-Acc. Acc. GPT-Acc.
Large Language Models
Llama-3-8B [2] 28.2 30.8 9.3 8.1 16.4 19.9 24.5 11.6 19.6 17.6
Qwen3-8B [34] 37.6 38.7 24.4 22.8 24.3 28.8 35.4 25.3 30.5 28.9
GPT-3.5-turbo 56.5 62.5 36.1 35.3 37.2 45.9 37.2 33.2 41.8 44.3
GPT-4o-mini [14] 54.2 61.1 34.3 33.8 34.8 44.1 34.3 34.8 39.5 43.4
RAG Baselines
BM25 [28] 62.3 67.5 50.4 51.3 56.1 68.2 48.0 52.9 54.2 59.9
Contriever [15] 64.9 68.1 68.8 68.2 53.6 65.5 37.3 46.3 56.1 62.2
ColBERTv2 [29] 68.7 71.7 72.8 71.1 63.4 75.9 56.4 62.7 65.3 70.2
GraphRAG Baselines
RAPTOR [5] 62.4 66.7 55.7 53.5 60.7 72.0 34.7 44.3 53.4 59.1
G-retriever [11] 66.4 67.1 28.8 28.3 45.3 51.6 50.6 32.1 47.7 44.7
LightRAG [6] 64.1 69.4 68.5 68.1 59.3 72.2 46.9 57.4 59.6 66.7
KGP [32] 61.3 67.6 52.6 51.6 54.8 63.7 43.6 53.8 53.1 59.2
HippoRAG [7] 57.2 61.7 70.6 70.4 57.6 70.6 68.6 73.8 63.5 69.1
HippoRAG2 [8] 67.7 70.9 73.2 71.7 65.5 79.5 67.7 72.5 68.5 73.6
Ours 69.1 72.6 75.1 73.5 65.9 80.2 76.3 81.5 71.6 76.9
4.4 Implementation Details
Our experiments were conducted on a server equipped with six
RTX-3090 GPUs. For a fair comparison, all RAG and GraphRAG
baseline models were evaluated under the same experimental setup,
employing the GPT-4o-mini model for both question answering
and graph construction, with the number of retrieved chunks (top-
k) set to 5. We split the 4000-query dataset into a training set of
1000 queries and a test set of 3000 queries. From the 1000 training
queries, we selected 200 samples on which RAG and GraphRAG
produced different predictions, and used these disagreement cases
to train the adapter to assign complexity scores for routing queries
to the appropriate retrieval path.
4.5 Main Results
We now present the main experimental results for question answer-
ing and retrieval, in which the QA model utilizes retrieved passages
as context.
4.5.1 QA Performance.As shown in Table 1, the proposed EA-
GraphRAG framework achieves consistent improvements on both
single-hop and multi-hop QA tasks, and obtains the best overall
performance on the mixed benchmark withAcc.71.6 andGPT-Acc.
Table 2: Dataset statistics
NQ PopQA 2Wiki HotpotQA Mix
Num of queries1,000 1,000 1,000 1,000 4,000
Num of docs9,633 8,676 6,119 9,811 34,23976.9. These results demonstrate that the framework can effectively
adapt to questions with different levels of complexity.
Compared with strong language model baselines such as GPT-4o-
mini, EA-GraphRAG achieves substantial gains across all datasets.
On the mixed benchmark, it improves Acc. from 39.5 to 71.6 and
GPT-Acc. from 43.4 to 76.9, indicating that external retrieval remains
indispensable even for powerful generators.
When compared with dense retrieval baselines, EA-GraphRAG
shows clear advantages, especially on 2Wiki, where relational rea-
soning across multiple passages is critical. Relative to the strongest
dense baseline ColBERTv2, EA-GraphRAG improves 2Wiki Acc.
from 56.4 to 76.3 and GPT-Acc. from 62.7 to 81.5. Compared with
GraphRAG baselines, EA-GraphRAG attains the highest perfor-
mance across all datasets. It slightly improves over HippoRAG2
on HotpotQA, and delivers large gains on 2Wiki in both Acc. and
GPT-Acc., validating the effectiveness of complexity-aware routing.
Overall, EA-GraphRAG achieves state-of-the-art results in both
Acc. and GPT-Acc. across single-hop, multi-hop, and mixed settings.
The results confirm that dynamically selecting between dense re-
trieval and graph-based retrieval, with a fusion fallback for border-
line cases, successfully combines their complementary strengths
for more robust question answering.
4.5.2 Retrieval Performance.We evaluate whether our routing and
fusion strategy improves retrieval recall in Table 3. On the mixed
benchmark, EA-GraphRAG achieves the best overall recall with
R@3 66.0 and R@5 74.9. This result exceeds the strongest dense
baseline ColBERTv2 by 8.6 points on R@3 and 10.0 points on R@5,
and improves over the strongest graph baseline HippoRAG2 by 0.4
points on R@3 and 0.8 points on R@5.

Conference, Under Review, Dong et al.
Table 3: Retrieval recall (R@3 / R@5) shown as percentages with one decimal. Best per column in bold; second-best underlined .
Simple QA Multi-Hop QAMix
Method NQ PopQA HotpotQA 2Wiki
R@3 R@5 R@3 R@5 R@3 R@5 R@3 R@5 R@3 R@5
RAG Baselines
BM25 [28] 37.2 51.9 24.8 29.8 62.6 71.8 57.3 61.4 45.5 53.7
Contriever [15] 36.2 50.9 34.3 42.4 60.6 68.5 45.1 50.9 44.0 50.9
ColBERTv2 [29] 47.4 65.4 47.0 48.9 71.5 77.8 63.9 67.6 57.4 64.9
GraphRAG Baselines
RAPTOR [5] 44.3 59.7 28.9 29.3 45.9 50.1 33.4 34.9 38.1 43.5
HippoRAG [7] 29.6 43.4 44.6 52.8 68.9 76.2 77.6 86.0 55.2 64.7
HippoRAG2 [8] 52.3 70.8 48.0 50.982.7 89.679.4 85.2 65.6 74.1
Ours 53.9 71.3 49.2 54.180.5 87.2 80.9 88.1 66.0 74.9
The gains are consistent across both simple and multi-hop regimes.
On NQ, EA-GraphRAG achieves R@3 53.9 and R@5 71.3, outper-
forming HippoRAG2 by 1.6 and 0.5 points. On PopQA, it reaches
R@3 49.2 and R@5 54.1, improving over ColBERTv2 by 2.2 and
5.2 points. On multi-hop QA, EA-GraphRAG obtains R@3 80.5 and
R@5 87.2 on HotpotQA, and R@3 80.9 and R@5 88.1 on 2Wiki.
These results indicate that complexity-aware routing and fusion
surface high-recall evidence for both simple factual queries and
multi-hop reasoning queries, validating our retrieval design.
We further measure per-query retrieval latency in Table 4. The
dense path is highly efficient, yielding a 96.4% reduction versus
graph retrieval. Leveraging this, our router without fusion runs at
1.14 s/query and the full system at 2.19 s/query. The overhead of the
full system is mainly introduced by invoking fusion on borderline
queries, reflecting a tunable accuracy–efficiency trade-off controlled
by the two thresholds𝜏 𝐿and𝜏𝐻.
Table 4: Average retrieval time per query (seconds).
Retrieval strategy Time/query (s)
dense retrieval 0.08
graph-based retrieval 3.23
Ours w/o fusion 1.14
Ours 2.19
4.6 Analysis of the Query Adapter
We analyze the proposed query adapter from the perspective of end-
to-end efficiency. Figure 3 presents the accuracy–latency Pareto
plot comparing our adapter-based framework with representative
semantic model baselines. The adapter yields a markedly better
trade-off, achieving higher accuracy while operating at substan-
tially lower average inference time. This advantage stems from
the adapter’s lightweight design: it relies on inexpensive syntactic
feature extraction and a small predictor to estimate query difficulty
and trigger routing or fusion, avoiding the need to run an additional
heavyweight semantic model for every query.
0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6
Average Time (s)0.400.450.500.550.600.650.700.75AccuracyOurs
Gpt-4o-miniGpt-3.5-turbo
Llama3-8BQwen3-8BFigure 3: Accuracy and efficiency comparison of adapters.
Importantly, the adapter improves not only runtime but also
overall effectiveness. Semantic scorers can be accurate but often
introduce non-trivial per-query overhead, which becomes prohibi-
tive under mixed workloads and interactive settings. In contrast,
our adapter enables fast complexity-aware decisions with minimal
latency, allowing the system to allocate expensive graph-based re-
trieval only when needed and to fall back to dense retrieval for
simpler cases. As a result, our framework reaches a superior oper-
ating point on the Pareto frontier, demonstrating that the proposed
adapter is both practical and effective for real-world deployment.
4.7 Ablation Study
To assess the contribution of each component in EA-GraphRAG,
we conduct an ablation study on the mixed benchmark; results are
reported in Table 6. The generator-only baseline achieves 39.5 Acc.
and 43.4 GPT-Acc., indicating that parametric knowledge alone
is insufficient for this setting. Adding dense retrieval leads to a
large gain, reaching 65.1 Acc. and 70.1 GPT-Acc., which highlights
the importance of grounding generation with retrieved evidence.

Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs Conference, Under Review,
Table 5: Cases comparing Vanilla RAG, GraphRAG and EA-GraphRAG.
Single-hop QuestionWhen is season 8 for game of thrones?
Ground Truth2019
GraphRAGRetrieved context:
(1)✗“Game of Thrones”: Game of Thrones Game of Thrones is an American fanta. . .
(2)✗“The Walking Dead (season 8)”: marked the first crossover between the two series. . .
. . .
Prediction:✗The text does not provide a specific date for Season 8 of Game of Thrones.
EA-GraphRAGRetrieved context:
(1)✓“Game of Thrones (season 8)”: . . . to premiere inApril 2019. Ramin Djawadi. . .
(2)✓“Game of Thrones (season 8)”: . . . The season is scheduled to premiere inApril 2019. . .
. . .
Prediction:✓April 2019.
Multi-hop QuestionWhat is the date of death of the director of film The Organization (Film)?
Ground TruthDecember 12, 2012
Vanilla RAGRetrieved context:
(1)✗“Lino Brocka”: Catalino Ortiz Brocka (April 3, 1939 – May 22, 1991) was a Filipino film director. . .
(2)✗“Wallace Fox”: Wallace Fox (March 9, 1895 – June 30, 1958) was an American film director. . .
. . .
Prediction:✗Not provided in the text.
EA-GraphRAGRetrieved context:
(1)✓ “The Organization (film)”: ...The Organizationis a 1971 DeLuxe Color American crime thriller film starring
Sidney Poitier as Virgil Tibbs and directed byDon Medford. . .
(2)✓“Don Medford”: . . .Donald Muller(November 26, 1917 –December 12, 2012). . .
. . .
Prediction:✓December 12, 2012.
Table 6: Ablation on the mixed benchmark. Starting from a
generator-only baseline, progressively adding dense retrieval,
graph-based retrieval, and finally the fusion module (RRF)
yields steady gains; the full EA-GraphRAG achieves the best
accuracy and GPT-based accuracy.
Ablation setting Acc. GPT-Acc.
GPT-4o-mini (no retrieval) 39.5 43.4
+ Dense retrieval 65.1 70.1
+ Dense & Graph-based retrieval 70.5 75.9
+ Dense & Graph-based retrieval + Fusion71.6 76.9
Incorporating graph-based retrieval further improves performance
to 70.5 Acc. and 75.9 GPT-Acc., showing that structural signals
provide complementary benefits beyond dense retrieval. Finally,
adding the fusion module yields the best results, achieving 71.6
Acc. and 76.9 GPT-Acc. This demonstrates that combining dense
and graph retrieval outputs with Reciprocal Rank Fusion effectively
improves robustness on queries where a single retrieval path is
insufficient.
4.8 Case Study
Table 5 presents two representative examples that illustrate why
EA-GraphRAG outperforms using either retrieval paradigm alone.
For the single-hop factoid “When is season 8 for game of thrones?,”
the graph-based retriever HippoRAG places excessive emphasis on
entity matches and link structure. As a result, it retrieves passagesthat contain relevant entities but do not include the local sentence-
level evidence specifying the release date, which introduces off-
target context and leads to an incorrect answer. Dense retrieval, by
prioritizing sentence- and passage-level semantic similarity, directly
surfaces passages stating that the season “premiere[s] in April 2019, ”
which is sufficient to answer the question.
For the multi-hop query “What is the date of death of the director
of The Organization (Film)?, ” dense retrieval tends to return generic
director biographies and fails to recover the required evidence
chain from the film to its director Don Medford and then to his
death date. In contrast, the graph-based path uses entity linking
and diffusion over the graph to traverse these relations and retrieve
evidence containing “December 12, 2012.” EA-GraphRAG addresses
these complementary failure modes by routing simple factoids to
dense retrieval and compositional multi-hop questions to graph-
based retrieval, while reserving the fusion path for ambiguous
cases. Together, these examples show that dynamic routing enables
a single framework to consistently retrieve appropriate evidence
across both simple and complex queries.
5 Conclusion
Graph retrieval-augmented generation (GraphRAG) has recently
emerged as a powerful paradigm that leverages external structured
graphs to enable deep retrieval and contextual comprehension. Ex-
isting GraphRAG frameworks excel at multi-hop reasoning but
struggle with simple factoid queries and are far less efficient than
vanilla RAG, due to their uniform reliance on graph construction

Conference, Under Review, Dong et al.
and multi-step traversal that introduces additional retrieval over-
head and can inject irrelevant graph-induced noise for queries that
only require localized evidence. To remedy this, we proposedEA-
GraphRAG, which performs syntactic complexity analysis of the
query androutessimple queries to dense (RAG) retrieval while
sending complex ones to graph-based retrieval, with an optional
fusion fallback for borderline cases. This design preserves seman-
tic precision on easy queries and leverages graph structure for
genuinely compositional ones, enabling a single system to han-
dle mixed difficulty levels effectively. Comprehensive experiments
demonstrate that EA-GraphRAG delivers consistent gains in re-
trieval and end-to-end QA accuracy while improving efficiency,
validating the effectiveness of complexity-aware routing in real-
world, heterogeneous query distributions.
References
[1]Jonathan Larson Darren Edge, Ha Trinh. 2024. LazyGraphRAG: Setting a new
standard for quality and cost.Microsoft Blog(2024).
[2]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models.arXiv e-prints(2024), arXiv–2407.
[3]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130
(2024).
[4]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large
language models: A survey.arXiv preprint arXiv:2312.10997(2023).
[5]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. LightRAG:
Simple and Fast Retrieval-Augmented Generation.arXiv preprint arXiv:2410.05779
(2024).
[6]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. arXiv:2410.05779 [cs.IR] https:
//arxiv.org/abs/2410.05779
[7]Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models.
[8]Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From rag to memory: Non-parametric continual learning for large language
models.arXiv preprint arXiv:2502.14802(2025).
[9]Haoyu Han, Harry Shomer, Yu Wang, Yongjia Lei, Kai Guo, Zhigang Hua, Bo
Long, Hui Liu, and Jiliang Tang. 2025. Rag vs. graphrag: A systematic evaluation
and key insights.arXiv preprint arXiv:2502.11371(2025).
[10] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al .
2024. Retrieval-augmented generation with graphs (graphrag).arXiv preprint
arXiv:2501.00309(2024).
[11] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann
LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and question answering.arXiv
preprint arXiv:2402.07630(2024).
[12] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning
steps.arXiv preprint arXiv:2011.01060(2020).
[13] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2024.
GRAG: Graph Retrieval-Augmented Generation.arXiv preprint arXiv:2405.16506
(2024).
[14] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh,
Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al .2024.
Gpt-4o system card.arXiv preprint arXiv:2410.21276(2024).
[15] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense in-
formation retrieval with contrastive learning.arXiv preprint arXiv:2112.09118
(2021).
[16] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni,
Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard
Grave. 2023. Atlas: Few-shot learning with retrieval augmented language models.
(2023).
[17] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park.
2024. Adaptive-rag: Learning to adapt retrieval-augmented large language models
through question complexity.arXiv preprint arXiv:2403.14403(2024).[18] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton
Lee, et al .2019. Natural questions: a benchmark for question answering research.
Transactions of the Association for Computational Linguistics7 (2019), 453–466.
[19] Roger Levy and Galen Andrew. 2006. Tregex and Tsurgeon: Tools for querying
and manipulating tree data structures. InInternational Conference on Language
Resources and Evaluation. 2231–2234.
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .
2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
[21] Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu Lin, Yaojie Lu, Qiaoyu Tang,
Fei Huang, Xianpei Han, Le Sun, and Yongbin Li. 2024. StructRAG: Boosting
Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information
Structurization.
[22] Lei Liang, Mengshu Sun, Zhengke Gui, Zhongshu Zhu, Zhouyu Jiang, Ling Zhong,
Yuan Qu, Peilong Zhao, Zhongpu Bo, Jin Yang, et al .2024. Kag: Boosting llms
in professional domains via knowledge augmented generation.arXiv preprint
arXiv:2409.13731(2024).
[23] Xiaofei Lu. 2010. Automatic analysis of syntactic complexity in second language
writing.International journal of corpus linguistics15, 4 (2010), 474–496.
[24] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2022. When not to trust language models: Investigat-
ing effectiveness of parametric and non-parametric memories.arXiv preprint
arXiv:2212.10511(2022).
[25] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2024. Graph retrieval-augmented generation: A survey.
arXiv preprint arXiv:2408.08921(2024).
[26] Tyler Thomas Procko and Omar Ochoa. 2024. Graph retrieval-augmented genera-
tion for large language models: A survey. InConference on AI, Science, Engineering,
and Technology (AIxSET).
[27] Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning.
2020. Stanza: A Python natural language processing toolkit for many human
languages. InAnnual Meeting of the Association for Computational Linguistics.
101–108.
[28] Stephen E Robertson and Steve Walker. 1994. Some simple effective approxi-
mations to the 2-poisson model for probabilistic weighted retrieval. InSIGIR’94:
Proceedings of the Seventeenth Annual International ACM-SIGIR Conference on
Research and Development in Information Retrieval, organised by Dublin City
University. Springer, 232–241.
[29] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei
Zaharia. 2022. ColBERTv2: Effective and Efficient Retrieval via Lightweight Late
Interaction. InProceedings of the 2022 Conference of the North American Chapter
of the Association for Computational Linguistics: Human Language Technologies.
3715–3734.
[30] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InInternational Conference on Learning Representations
(ICLR).
[31] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 10014–10037.
[32] Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr.
2024. Knowledge graph prompting for multi-document question answering.
[33] Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong,
Xiao Huang, and Jinsong Su. 2025. When to use graphs in rag: A compre-
hensive analysis for graph retrieval-augmented generation.arXiv preprint
arXiv:2506.05690(2025).
[34] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al .2025. Qwen3 technical
report.arXiv preprint arXiv:2505.09388(2025).
[35] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering.
[36] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou,
Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, and Xiao Huang. 2025. A Sur-
vey of Graph Retrieval-Augmented Generation for Customized Large Language
Models.arXiv preprint arXiv:2501.13958(2025).
[37] Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He,
Yongwei Zhang, Sicong Liang, Xilin Liu, Yuchi Ma, et al .2025. In-depth Analysis
of Graph-based RAG in a Unified Framework.arXiv preprint arXiv:2503.04338
(2025).