# HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG

**Authors**: Yuqi Huang, Ning Liao, Kai Yang, Anning Hu, Shengchao Hu, Xiaoxing Wang, Junchi Yan

**Published**: 2026-02-24 14:05:29

**PDF URL**: [https://arxiv.org/pdf/2602.20926v1](https://arxiv.org/pdf/2602.20926v1)

## Abstract
Large Language Models (LLMs) often struggle with inherent knowledge boundaries and hallucinations, limiting their reliability in knowledge-intensive tasks. While Retrieval-Augmented Generation (RAG) mitigates these issues, it frequently overlooks structural interdependencies essential for multi-hop reasoning. Graph-based RAG approaches attempt to bridge this gap, yet they typically face trade-offs between accuracy and efficiency due to challenges such as costly graph traversals and semantic noise in LLM-generated summaries. In this paper, we propose HyperNode Expansion and Logical Path-Guided Evidence Localization strategies for GraphRAG (HELP), a novel framework designed to balance accuracy with practical efficiency through two core strategies: 1) HyperNode Expansion, which iteratively chains knowledge triplets into coherent reasoning paths abstracted as HyperNodes to capture complex structural dependencies and ensure retrieval accuracy; and 2) Logical Path-Guided Evidence Localization, which leverages precomputed graph-text correlations to map these paths directly to the corpus for superior efficiency. HELP avoids expensive random walks and semantic distortion, preserving knowledge integrity while drastically reducing retrieval latency. Extensive experiments demonstrate that HELP achieves competitive performance across multiple simple and multi-hop QA benchmarks and up to a 28.8$\times$ speedup over leading Graph-based RAG baselines.

## Full Text


<!-- PDF content starts -->

HELP: HyperNode Expansion and Logical Path-Guided Evidence
Localization for Accurate and Efficient GraphRAG
Yuqi Huang1Ning Liao1Kai Yang1Anning Hu1Shengchao Hu1Xiaoxing Wang1†Junchi Yan1
Abstract
Large Language Models (LLMs) often struggle
with inherent knowledge boundaries and hallu-
cinations, limiting their reliability in knowledge-
intensive tasks. While Retrieval-Augmented Gen-
eration (RAG) mitigates these issues, it frequently
overlooks structural interdependencies essential
for multi-hop reasoning. Graph-based RAG ap-
proaches attempt to bridge this gap, yet they typ-
ically face trade-offs between accuracy and ef-
ficiency due to challenges such as costly graph
traversals and semantic noise in LLM-generated
summaries. In this paper, we propose HyperNode
Expansion and Logical Path-Guided Evidence
Localization strategies for GraphRAG (HELP),
a novel framework designed to balance accuracy
with practical efficiency through two core strate-
gies: 1) HyperNode Expansion, which iteratively
chains knowledge triplets into coherent reasoning
paths abstracted as HyperNodes to capture com-
plex structural dependencies and ensure retrieval
accuracy; and 2) Logical Path-Guided Evidence
Localization, which leverages precomputed graph-
text correlations to map these paths directly to the
corpus for superior efficiency. HELP avoids ex-
pensive random walks and semantic distortion,
preserving knowledge integrity while drastically
reducing retrieval latency. Extensive experiments
demonstrate that HELP achieves competitive per-
formance across multiple simple and multi-hop
QA benchmarks and up to a 28.8 ×speedup over
leading Graph-based RAG baselines.
1. Introduction
Large Language Models (LLMs) have demonstrated strong
capabilities in natural language understanding and gener-
ation, but they are prone to hallucination and often lack
up-to-date domain knowledge, which limits their reliability
†Corresponding author.1Shanghai Jiao Tong University, China.
Correspondence to: Xiaoxing Wang <figure1_wxx@sjtu.edu.cn>.
Preprint. February 25, 2026.in knowledge-intensive applications (Huang et al., 2025).
A common solution is Retrieval-Augmented Generation
(RAG), which splits documents into passages, retrieves the
top-k relevant passages via dense vector search, and appends
them to the prompt to ground the LLM’s response (Gao
et al., 2023; Lewis et al., 2020). This pipeline is simple,
robust, and scalable, making it practical for real-world de-
ployment. However, dense retrieval often treats seemingly
unstructured text as flat content and overlooks the underly-
ing structure that can be distilled from it (Han et al., 2024).
This structured information is crucial for organizing and re-
trieving knowledge in massive, multi-source corpora, which
limits the effectiveness of dense retrieval in realistic set-
tings (Peng et al., 2024). Many queries require multi-step
reasoning (Plaat et al., 2025) and a deep understanding of
the interconnections between entities and events across dis-
parate sources. In these cases, RAG’s similarity-based top -k
retrieval emphasizes local paragraph relevance rather than
knowledge structure, which can introduce redundant or un-
focused context and reduce performance on multi-hop and
relational questions (Zhang et al., 2025).
To bridge this structural gap, Graph-based RAG approaches
have been proposed to incorporate Knowledge Graphs into
the retrieval process. The standard workflow of Graph-
based RAG (Gao et al., 2023; Zhang et al., 2025) typ-
ically involves two stages: Knowledge Graph Construc-
tion and Graph-Augmented Retrieval. In the construction
phase, entities and their corresponding relations are ex-
tracted from raw corpus to build a structured graph (Edge
et al., 2024; Jimenez Gutierrez et al., 2024; Gutiérrez et al.,
2025), though some lightweight variants (Zhuang et al.,
2025) focus exclusively on entity indexing. During retrieval,
these methods navigate the relational topology using graph
algorithms such as Personalized PageRank (PPR) (Yang
et al., 2024) to identify relevant evidence, which are then
synthesized into textual context for the LLM.
However, existing solutions face a dilemma between accu-
racy and efficiency (Han et al., 2025; Xiang et al., 2025).(1)
Graph-based RAG methods can suffer from inefficient
retrieval of information stored in knowledge graphs.
Methods such as HippoRAG (Jimenez Gutierrez et al.,
2024), HippoRAG2 (Gutiérrez et al., 2025) and ToG (Sun
1arXiv:2602.20926v1  [cs.AI]  24 Feb 2026

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
et al., 2023; Ma et al., 2024) preserve rich relational infor-
mation for complex reasoning, but their reliance on graph
traversal introduces substantial computational overhead and
latency. Moreover, GraphRAG’s LLM-generated commu-
nity summaries (Edge et al., 2024) can introduce semantic
noise and secondary hallucinations, degrading retrieval pre-
cision.(2) Some methods ignores edge information to
simplify retrieval and improve efficiency, but at the cost
of accuracy.LinearRAG (Zhuang et al., 2025) accelerates
retrieval by constructing a relation-free hierarchical graph,
termed Tri-Graph, using only lightweight entity extraction
and semantic linking, effectively bypassing complex re-
lational modeling. Similarly, E2GraphRAG (Zhao et al.,
2025) streamlines the indexing process by utilizing SpaCy
to construct bidirectional indexes between entities and pas-
sages to capture their many-to-many relations. While these
approaches achieve significant speedups in indexing and
retrieval, their reliance on simplified topologies weakens the
explicit semantic dependencies required for precise multi-
hop reasoning. Thus, a solution is still needed that maintains
high accuracy while improving efficiency.
We propose HyperNode Expansion and Logical Path-
Guided Evidence Localization strategies for GraphRAG,
named asHELP. It is a relation-aware retrieval framework
that balances multi-hop accuracy with practical efficiency.
On the one hand, we introduce HyperNode as a core unit to
represent and chain knowledge triplets for multi-hop reason-
ing. In this framework, each retrieved knowledge triplet is
instantiated as a HyperNode, which is then used to update
the working graph, enabling iterative retrieval that chains
HyperNodes together. This process naturally expands rea-
soning paths, transforming isolated facts into multi-hop
relational chains. On the other hand, we introduce Logical
Path-Guided Evidence Localization, an efficient strategy
that utilizes the expanded final HyperNodes as precise evi-
dence to fetch the most relevant supporting passages from
the corpus. By first constructing the relational path in the
structured space, this approach converts the complex multi-
hop question into a targeted retrieval task, ensuring that
the returned passages are both semantically relevant and
logically connected. Extensive experiments show that our
method can preserve the structural integrity of knowledge
graphs while maintaining the high retrieval efficiency. Our
contributions are summarized as follows:
1) An Iterative HyperNode Retrieval Strategy for Rea-
soning Path Expansion.We introduce the HyperNode, a
higher-order retrieval unit that bundles triples together with
their relational paths into a unified entity. By explicitly
modeling knowledge paths rather than isolated fragments,
HyperNode better captures inter-fact dependencies and sub-
stantially improves multi-hop reasoning, while maintaining
strong performance on single-hop QA.2) An Efficient Reasoning Path-Guided Evidence Local-
ization Strategy.We propose a lightweight Path-Guided
Evidence Localization strategy that leverages precomputed
graph-text correlations. By mapping the HyperNodes con-
structed in the structured space directly to the source corpus,
this strategy converts complex multi-hop questions into tar-
geted retrieval tasks.
3) Superior Performance and Consistent Effectiveness.
Our framework achieves an excellent balance between multi-
hop reasoning accuracy and computational efficiency. Cru-
cially, our method demonstrates strong and consistent ef-
fectiveness across a variety of benchmarks under the same
configuration. This consistent effectiveness renders our
method a compelling, low-overhead enhancement across a
wide array of Large Language Models.
2. Related Work
RAG for Complex Reasoning.Retrieval-Augmented Gen-
eration (RAG) serves as a robust paradigm to mitigate hal-
lucinations by grounding Large Language Models (LLMs)
in external knowledge corpora (Lewis et al., 2020). While
standard dense retrieval frameworks excel at addressing sim-
ple factoid queries, they often falter in complex, multi-hop
reasoning scenarios where evidence is fragmented across
multiple documents (Yang et al., 2018; Thakur et al., 2021).
To overcome these limitations, iterative retrieval methods
such as IR-CoT (Trivedi et al., 2023) have been proposed
to interleave reasoning traces with retrieval steps. However,
these approaches primarily operate on a flat semantic space,
which lack explicit modeling of the structural dependencies
between entities. This limitation underscores the necessity
of structured retrieval mechanisms, such as Graph-based
RAG (Edge et al., 2024), which leverage relational priors
to maintain logical coherence across disparate information
nodes.
Graph-based RAG.Recognizing the limitations of stan-
dard RAG, recent works incorporate Knowledge Graphs
(KGs) to introduce structural priors (Hu et al., 2025;
Mavromatis & Karypis, 2024). Approaches like Hip-
poRAG (Jimenez Gutierrez et al., 2024) and its succes-
sor HippoRAG2 (Gutiérrez et al., 2025) draw inspiration
from hippocampal memory indexing, utilizing Personal-
ized PageRank algorithm to navigate relational paths. By
mimicking associative memory, they effectively bridge dis-
connected entities across the corpus. At the same time,
methods such as GraphRAG (Edge et al., 2024) focus on hi-
erarchical abstraction; they employ LLMs to pre-summarize
clustered graph communities, providing a "map-reduce"
style global context that handles high-level thematic queries
more effectively than point-wise retrieval. In parallel, GNN-
RAG (Mavromatis & Karypis, 2024) introduces a neuro-
symbolic framework that utilizes GNNs to reason over dense
2

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
subgraphs for answer retrieval, verbalizing the extracted
shortest paths into natural language to enable LLMs to per-
form complex multi-hop reasoning. Despite their effec-
tiveness in improving reasoning accuracy, these methods
typically incur high computational costs and latency due to
complex graph algorithms or extensive LLM usage, limiting
their scalability in real-time applications.
Efficient Retrieval with Structural Awareness.Balanc-
ing structural utilization with retrieval efficiency remains a
formidable challenge. Current research primarily diverges
into two extremes. On one hand, lightweight frameworks
like LinearRAG (Zhuang et al., 2025) achieve rapid retrieval
by simplifying graph topology; however, this reductionist
approach inevitably discards critical relational semantics
and high-order dependencies necessary for precise multi-
hop reasoning. On the other hand, structure-centric ap-
proaches prioritize reasoning depth but frequently encounter
scalability bottlenecks. For instance, path-based methods
such as ToG (Sun et al., 2023) rely on iterative LLM-guided
beam searches, which capture complex dependencies at the
cost of prohibitive inference latency. Similarly, subgraph-
based methods like QA-GNN (Yasunaga et al., 2021) and
GRAG (Hu et al., 2025) try to preserve relational contexts
viak-hop ego networks, yet incur substantial overhead dur-
ing subgraph extraction and encoding. To bridge this gap,
hybrid approaches (Jin et al., 2024; Jiang et al., 2025) use
adaptive agents to dynamically select retrieval granularities
(nodes, paths, or subgraphs). While these methods mitigate
noise, the decision-making process of agents often intro-
duces extensive overhead that limits real-time applicability.
3. Methodology
We elaborate on the mechanism of HELP, as illustrated in
Fig. 1. Our approach comprises of three phases: Knowl-
edge Graph Construction (Sec. 3.1), Iterative HyperNode
Expansion (Sec. 3.2), and Logical Path-Guided Evidence
Localization (Sec. 3.3).
3.1. Knowledge Graph Construction
To incorporate structural knowledge, Graph-Based RAG
leverages a Knowledge Graph denoted as G= (V,E) , where
Vrepresents entities and Erepresents relations. The graph is
typically constructed by extracting triplets from the corpus.
Existing methods perform graph traversal, such as Personal
PageRank or BFS, starting from entities mentioned in q.
However, traversing the entire graph topology is computa-
tionally expensive, and ignoring edge semantics leads to
accuracy loss. Our method aims to resolve this trade-off.
To bridge the gap between unstructured text and struc-
tured reasoning, we first construct a Knowledge Graph G.
Given a corpus of documents, we partition each documentinto passages P={p 1, p2, . . . , p N}. Then we employ
an Open Information Extraction (OpenIE) module to ex-
tract a set of relational triplets from each passage pi. Let
Ti={(h k, rk, tk)}|Ti|
k=1denote the sequence of relational
triplets extracted from passagep i.
Crucially, to support our retrieval mechanism, we construct
a Triple-to-Passage Index Φ, which captures graph-text cor-
relations. For every unique triplet τ= (h, r, t) , we maintain
a mapping to its provenance passages:
Φ(τ) =
(pi, wi)|τ∈ T i, w i=1
|Ti|
(1)
wiserves as a density-normalized weight, ensuring passages
containing fewer yet more specific facts outweigh dense,
noisy passages in retrieval scores.
3.2. Iterative HyperNode Expansion
To enable multi-hop reasoning, we introduce HyperNode
Expansion, a strategy built upon the iterative merging and
expanding of relational contexts. In our framework, a Hy-
perNode H={τ 1, . . . , τ m}is defined as a cumulative unit
instantiated by merging mcoherent knowledge triplets into
a unified semantic representation. By recursively expanding
these nodes with adjacent relations, we transform discrete,
isolated facts into integrated multi-hop reasoning paths.
3.2.1. HYPERNODESERIALIZATION
To map a HyperNode into a dense vector space, we must han-
dle the permutation invariance of the set structure. Unlike
traditional sequential data, a HyperNode His an unordered
set of triplets. Without a canonical ordering, the encoder
E(·) might produce inconsistent embeddings for the same
set of facts in different sequences, hindering the effective
representation of evidence within the structured data.
To mitigate this, we employ a deterministic linearization
function S(·) designed to bridge the gap between structured
sets and sequential modeling. Specifically, we first sort the
triplets lexicographically based on their constituent elements
to ensure a unique, reproducible order regardless of the orig-
inal data extraction sequence. Once sorted, these structured
components are flattened into a unified text sequence:
S(H) =join
[
τ∈sorted(H)concat(h τ, rτ, tτ)
 (2)
The dense vector representation of the HyperNode is then ob-
tained via a pre-trained Transformer-based encoder: vH=
E(S(H)) . By mapping the relational structure into a contin-
uous latent space, this process allows the model to capture
the aggregate semantic context.
3

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
Phase I: Knowledge Graph Construction
Phase III:  Logical Path-
Guided  Evidence LocalizationHybrid Retrieval Strategy
Dense Passage
Retrieval
Logical Passages Dense Passages
Phase II:  Iterative
HyperNode ExpansionQuery
N hopsExpansion
PruningUpdate
HyperNodes
HyperNodeEntityNode
Relation Edge
Triple-to-Passage IndexKnowledge Graph
Initialize
HyperNode
Path-Guided
Scoring
AnswerGeneration
Raw Corpus
 PassagesIndexing Chunking…Mapping
Triple-to-Passage 
Index
Query
 Final Context
(Top-L Passages)
Figure 1.Overview of the HELP framework. The workflow consists of three stages: (I) Knowledge Graph Construction, utilizing OpenIE
to build the Triple-to-Passage Index; (II) Iterative HyperNode Expansion, which iteratively chains triples as HyperNodes while pruning
irrelevant ones to maintain reasoning coherence; and (III) Logical Path-Guided Evidence Localization, where HyperNodes are grounded
back to the original passages for precise evidence retrieval via a hybrid mechanism.
3.2.2. ITERATIVEEXPANSIONPROCESS
The core of our approach is the iterative expansion mecha-
nism. Given a user query q, we first map it to a normalized
vector representation vq=E(q)
∥E(q)∥ 2, where E(·) denotes
the encoder. The expansion process then updates the set of
active HyperNodes over Nhops to generate the final set
Hfinal . The overall procedure is summarized in Algorithm
1 and details are provided below.
HyperNode Initialization.It begins with the identification
of seed HyperNodes, which serve as the semantic anchors
for subsequent reasoning. Instead of starting from raw en-
tities, we leverage the fine-grained semantics of individual
triplets {τ} extracted from the corpus. We compute the
cosine similarity between the query vector vqand the nor-
malized embedding of each candidate triplet. By selecting
the top- ntriplets with the highest alignment scores, we
form the initial set H1. This step ensures that the expansion
starts from the most promising factual kernels, effectively
reducing the noise introduced by irrelevant graph regions.
Iterative Expansion.At each step i(where 2≤i≤N , and
Ndenotes the number of expansion hops), the model ex-
tends reasoning paths to incorporate broader context. Specif-
ically, in the i-th expansion round, we process each Hyper-
Node H∈H i−1individually. We identify a set of adjacent
triplets Tadjfrom the Knowledge Graph Gthat are directly
connected to the entities contained within the current H.
For each adjacent triplet τnext∈ Tadj, we instantiate a new
expanded HyperNode H′=H∪ {τ next}. The global can-
didate set Ccand is then formed by accumulating all such
generated H′instances derived from each preceding Hy-perNode in Hi−1. This incremental growth allows each
HyperNode to transition from a single fact to a complex
path structure representing a multi-hop chain of evidence.
Pruning via Semantic Distance.To counteract the expo-
nential growth of the search space and to maintain focus on
the query’s core intent, we implement a semantic-guided
beam search strategy. This strategy ensures that only the
most semantically pertinent reasoning paths are retained
at each hop. We serialize the content of each candidate
H′into a sequence S(H′)and compute its distance to the
query. Since the embeddings are normalized, we utilize the
Euclidean distance as our metric:
dist(H′, q) =E(S(H′))
∥E(S(H′))∥2−vq
2(3)
Based on Eq. (3), we select the top- kHyperNodes with the
minimal distance to form Hi. By maintaining a fixed beam
width k, it constrains the search space to a manageable scale,
preventing the computational overhead typically associated
with deep graph traversals. Consequently, it ensures that
the final converged set, Hfinal , is not merely a collection
of isolated facts, but a structured assembly of complete,
logically coherent, and semantically aligned reasoning paths
tailored to the specific information need.
3.3. Logical Path-Guided Evidence Localization
The final phase of our framework is Logical Path-Guided
Evidence Localization, a process designed to transform ab-
stract structural information into concrete textual evidence.
By utilizing the expanded HyperNodes as precise seman-
tic anchors, we can pinpoint the most relevant supporting
4

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
Algorithm 1Iterative HyperNode Expansion Algorithm
Require: User query q, Knowledge Graph G, Expansion
hopsN, Initial triple sizen, Beam sizek
Ensure:Final set of HyperNodesH final
1:Initialize:v q←Normalize(E(q))
2:// Step 1: Initialization with Seed HyperNodes
3:Retrieve all triples{τ}fromG
4:Score triples by similarity tov q
5:H 1←Top-nranked triples
6:// Step 2: Iterative Expansion Process
7:fori= 2toNdo
8:C cand← ∅
9:foreach HyperNodeHinH i−1do
10:T adj← {τ∈ G |τis adjacent to entities inH}
11:foreachτ next inTadjdo
12:Expand:H′←H∪ {τ next}
13:AddH′toCcand
14:end for
15:end for
16:// Step 3: Pruning (Beam Search)
17:H i← ∅
18:foreach candidateH′inCcand do
19:v H′←Normalize(E(S(H′)))
20:Score:Calculate dist(H′, q)using Eq. (3)
21:end for
22:H i←Select top- kHyperNodes from Ccand with
minimal distance
23:end for
24:returnH final =H N
passages within the massive corpus, ensuring that the down-
stream generation is grounded in verified facts.
To bridge the semantic gap between the retrieved structural
knowledge and the original source corpus, we leverage the
Triple-to-Passage Index Φ. This specialized inverted index,
as constructed in Sec. 3.1, establishes a one-to-many map-
ping between each unique triplet τand the set of source
passages from which it was extracted. By recognizing that a
single factual claim can be instantiated across multiple dis-
parate passages, this index ensures comprehensive evidence
coverage. Such a mechanism enables robust provenance
tracking, allowing the model to backtrack from a specific
hop in the reasoning path to all supporting textual fragments
that validate the underlying logic.
We compute a relevance score for each passage pby ag-
gregating the evidentiary signals from the complete set of
reasoning paths. Instead of considering only unique triplets,
we sum the contributions of all triplets τacross the final
set of HyperNodes Hfinal to account for the consensus of
evidence. The relevance score for passagepis derived as:
Score(p) =X
H∈H finalX
τ∈HI(p∈Φ(τ))·e−dist(H,q)·wp,τ(4)where wp,τrepresents the provenance weight associated
with the extraction density, as mentioned in Sec. 3.1.
In this formulation, the term κ(H, q) =e−dist(H,q)functions
as a non-linear soft-matching mechanism. By aggregating
scores across all HyperNodes, this approach naturally yields
a consensus-based ranking: passages are scored higher not
only for semantic alignment with the query, but also for
being repeatedly reinforced through multiple distinct rea-
soning paths. This ensures that the evidence localization is
robust to individual triplet noise and favors passages that
reside at the intersection of diverse logical chains.
Drawing inspiration from hybrid search paradigms
(Bhagdev et al., 2008; Wang et al., 2025), we adopt a com-
posite strategy to construct a robust final context of size
K. We recognize that Logical Path-Guided Evidence Lo-
calization and Dense Passage Retrieval (DPR) operate on
distinct and complementary relevance signals. Our logical
path-guided approach ensures high precision by leveraging
the consensus of multi-hop reasoning chains: passages that
are reinforced by multiple intersecting paths receive priori-
tized ranking through our additive scoring mechanism. In
contrast, DPR excels at capturing broad semantic coverage
that structural methods might overlook.
To leverage this complementarity, we assign a primary quota
ofMpassages to the evidence localized via the Logical
Path-Guided Evidence Localization, ensuring the final con-
text is anchored in structured reasoning. This foundation is
then supplemented with top-ranked candidates from DPR
to fill the remaining slots, with a deduplication step to max-
imize incremental information gain. As demonstrated in
our ablation study (see Sec. 4.4), this hybrid retrieval strat-
egy—prioritizing structure-based consensus while backfill-
ing with semantic matches—yields superior performance
compared to single-stream methods by balancing the depth
of logical rigor with the breadth of semantic recall.
4. Experiments
We conduct comprehensive experiments to verify the ac-
curacy and efficiency of HELP across a diverse range of
single-hop and multi-hop QA benchmarks. By comparing
HELP against traditional dense retrievers, embedding mod-
els, and current Graph-based RAG frameworks, we aim to
demonstrate its superior ability to localize high-quality evi-
dence. Furthermore, we provide in-depth ablation studies to
validate our Hypernode Expansion and Logical Path-Guided
Evidence Localization Strategies.
4.1. Experimental Settings
Datasets.Our evaluation spans both Simple QA and
Multi-Hop QA tasks to ensure a comprehensive assessment.
For Simple QA, we utilize NaturalQuestions (NQ) dataset
5

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
Table 1.Performance (F1 scores) comparison of various retrieval methods across different tasks, with all models evaluated using
Llama-3.3-70B-Instructas the backbone. Results are directly reported from HippoRAG2 (Gutiérrez et al., 2025) by default, while methods
marked with (∗) denote our local reproduction. The best results are inbold, and the second best are underlined . The†symbol indicates a
failure in graph construction due to large corpus size.
Simple QA Multi-Hop QA
Retrieval NQ PopQA MuSiQue 2Wiki HotpotQA LV-Eval Avg
Simple Baselines
None 54.9 32.5 26.1 42.8 47.3 6.0 34.9
Contriever (Izacard et al., 2021) 58.9 53.1 31.3 41.9 62.3 8.1 42.6
BM25 (Robertson & Walker, 1994) 59.0 49.9 28.8 51.2 63.4 5.9 43.0
GTR (Ni et al., 2022) 59.9 56.2 34.6 52.8 62.8 7.1 45.6
Large Embedding Models
GTE-Qwen2-7B (Ni et al., 2022) 62.0 56.3 40.9 60.0 71.0 7.1 49.6
GritLM-7B (Muennighoff et al., 2024) 61.3 55.8 44.8 60.6 73.3 9.8 50.9
NV-Embed-v2 (Lee et al., 2024) 61.9 55.7 45.7 61.5 75.3 9.8 51.7
Graph-based RAG Methods
RAPTOR (Sarthi et al., 2024) 50.7 56.2 28.9 52.1 69.5 5.0 43.7
GraphRAG (Edge et al., 2024) 46.9 48.1 38.5 58.6 68.6 11.2 45.3
LightRAG (Guo et al., 2024) 16.6 2.4 1.6 11.6 2.4 1.0 5.9
HippoRAG (Jimenez Gutierrez et al., 2024) 55.3 55.9 35.1 71.8 63.5 8.4 48.3
HippoRAG2 (Gutiérrez et al., 2025) 63.3 56.2 48.671.0 75.5 12.954.6
LinearRAG∗(Zhuang et al., 2025) 54.5 26.0 31.0 54.9 59.0 10.3 39.3
HyperGraphRAG∗(Luo et al., 2025) 53.6 26.4 41.6 61.2 73.1 N/A†-
HELP (Ours) 63.5 57.6 48.4 73.9 75.6 12.5 55.3
(Wang et al., 2024) and PopQA (Mallen et al., 2023); for
the more challenging Multi-Hop QA, we include MuSiQue
(Trivedi et al., 2022), 2WikiMultiHopQA(2Wiki) (Ho et al.,
2020), HotpotQA (Yang et al., 2018), and LV-Eval (Yuan
et al., 2024). By adopting identical data splits, corpora,
and evaluation protocols as HippoRAG2 (Gutiérrez et al.,
2025), we ensure a rigorous and fair comparison across both
retrieval quality and generation accuracy.
Baselines.We compare HELP against a comprehensive set
of baselines categorized into three groups, following the ex-
perimental protocol established by HippoRAG2 (Gutiérrez
et al., 2025). First, we include classic and dense retrieval
methods, namely BM25 (Robertson & Walker, 1994), Con-
triever (Izacard et al., 2021), and GTR (Ni et al., 2022).
Second, we evaluate against state-of-the-art 7B-parameter
embedding models that lead the BEIR benchmark(Thakur
et al., 2021), including GTE-Qwen2-7B-Instruct (Li et al.,
2023), GritLM-7B (Muennighoff et al., 2024), and NV-
Embed-v2 (Lee et al., 2024). Finally, we compare HELP
with Graph-based RAG methods, encompassing hierarchical
approaches like RAPTOR (Sarthi et al., 2024), global sum-
marization methods such as GraphRAG (Edge et al., 2024)
and LightRAG (Guo et al., 2024), and knowledge-graph-
based frameworks like HippoRAG (Jimenez Gutierrez et al.,
2024). For these aforementioned baselines, we directly re-
port the results published in HippoRAG2 as they share an
identical evaluation setting with our work. Furthermore,
to ensure a timely and rigorous comparison, we faithfullyreproduce the most recent advancements, including Lin-
earRAG (Zhuang et al., 2025), and HyperGraphRAG (Luo
et al., 2025), under the same configurations to maintain
complete experimental parity.
Evaluation Metrics.Consistent with the evaluation settings
in HippoRAG (Jimenez Gutierrez et al., 2024) and Hip-
poRAG2 (Gutiérrez et al., 2025), we employ the token-level
F1 metric defined in the MuSiQue (Trivedi et al., 2022).
Implementations.For the implementation of HELP, we
maintain strict architectural parity with the settings de-
scribed in HippoRAG2 (Gutiérrez et al., 2025) to ensure the
validity of our comparative analysis. Specifically,Llama-
3.3-70B-Instruct(AI@Meta, 2024) powers both the knowl-
edge extraction (NER/OpenIE) and response generation
phases, whileNV-Embed-v2(Lee et al., 2024) serves as the
primary retriever to bridge structural knowledge with seman-
tic embeddings. Our QA module uses the top-5 retrieved
passages as context for an LLM to generate the final answer.
This same combination of extractor and retriever is applied
across all reproduced Graph-based RAG baselines to ensure
a fair evaluation.
4.2. QA performance
Table 1 shows the performance of HELP compared to com-
petitive baselines. The hyperparameter configurations for
HELP are consistently applied across all datasets as detailed
in Appendix A.1. Overall, HELP sets a new state-of-the-art
6

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
PopQA(Simple QA) 2Wiki(Multi-Hop QA)051015202530Speedup Ratio Relative to HippoRAG2
1.0x
(1403s)1.0x
(1455s)2.0x
(695s)1.7x
(832s)7.8x
(180s)8.7x
(167s)16.5x
(85s)28.8x
(51s)HippoRAG2
HyperGraphRAG
LinearRAG
HELP
Figure 2.Retrieval efficiency on PopQA (Simple QA) and 2Wiki
(Multi-Hop QA). Absolute retrieval time (in seconds) for process-
ing 1,000 queries are annotated above the bars, highlighting that
HELP reduces the total retrieval latency to under 90 seconds.
across the evaluated benchmarks, achieving a consistent
performance gain over Graph-based RAG methods.
Overall Superiority.Compared to the previous leading
Graph-based RAG method HippoRAG2, HELP yields a rela-
tive improvement of 1.3% in average F1 score. Furthermore,
HELP significantly outperforms the strongest large-scale
embedding model NV-Embed-v2, with a 7.0% relative gain
in average F1, demonstrating that even state-of-the-art 7B
embedding models can be significantly enhanced through
our structural knowledge integration. When compared to tra-
ditional dense retrievers such as GTR, our method exhibits a
substantial performance leap of 21.3%, underscoring the vi-
tal necessity of bridging structural knowledge with semantic
retrieval for complex question answering.
Performance on Simple QA.In single-hop scenarios rep-
resented by NQ and PopQA, HELP demonstrates superior
retrieval precision. While the Hypernode Expansion mech-
anism is specifically engineered to navigate high-order re-
lational dependencies in multi-hop reasoning, it notably
maintains a competitive advantage in simpler, factoid-based
tasks. Specifically, HELP achieves a 2.5% relative improve-
ment over the second-best performing method HippoRAG2
on PopQA dataset. Furthermore, it maintains a steady gain
of 2.6% to 3.4% over the strongest 7B embedding models,
such as NV-Embed-v2. This indicates that our knowledge-
enhanced approach effectively anchors semantic search with
precise evidence localization, ensuring that simple factoid
queries benefit from the structural clarity of the knowledge
graph while maintaining a decisive competitive edge over
purely latent-space representations.
Performance on Multi-Hop QA.The advantages of HELP
are most pronounced in complex reasoning tasks where re-
lational dependencies are critical. On the 2Wiki dataset,
HELP outperforms HippoRAG2 by 4.1%. Compared to
more recent methods like LinearRAG, HELP achieves a
staggering 34.6% relative improvement on the same bench-
mark, suggesting that our method more effectively navigatesTable 2.Ablation study of the Hybrid Retrieval Strategy on the
2Wiki dataset. Mdenotes the primary quota of passages assigned
to the logical path-guided retrieval, while the total context size is
fixed at 5. M= 0 corresponds to a purely dense retrieval baseline.
Quota(M) EM(%) F1(%) Recall@5(%)
0 57.5 61.55 76.25
1 57.7 62.19 76.52
2 63.0 69.52 84.95
3 64.3 71.01 89.00
4 66.6 73.90 92.15
5 65.9 73.09 91.65
multi-hop paths than linear structures. Notably, while the in-
dexing complexity of HyperGraphRAG rendered it unusable
for the large-scale corpora of LV-Eval, HELP successfully
processed the entire dataset, demonstrating both great rea-
soning capabilities and superior scalability.
4.3. Retrieval Efficiency
As illustrated in Fig. 2, we evaluated the retrieval time of
HELP against three strong baselines: HippoRAG2, Hyper-
GraphRAG, and LinearRAG.
Our method demonstrates overwhelming efficiency across
both simple and multi-hop QA benchmarks. Specifically,
on the PopQA dataset, HELP achieves a 16.5 ×speedup
relative to HippoRAG2, drastically compressing the total
retrieval time for 1,000 queries from 1,403s to a mere 85s.
This performance gap widens even further on the complex
2Wiki dataset, where HELP attains a remarkable 28.8 ×
speedup. Traditional graph-based baselines still suffer from
the combinatorial explosion of graph traversals, requiring
over 20 minutes to resolve dependencies, whereas HELP
consistently completes the same tasks in under 90 seconds.
The significant efficiency gain of HELP stems from its
HyperNode-based pruning strategy and a purely embedding-
driven retrieval process. Unlike iterative graph methods
that rely on LLMs for intermediate node filtering or path
expansion, HELP bypasses these costly generative steps and
the redundant neighbor explorations typical of traditional
graph walks. Even when compared to LinearRAG, which
was specifically designed to mitigate graph overhead, HELP
further accelerates the process by approximately 2-3 ×. This
confirms that HELP successfully retains the structural rea-
soning benefits of knowledge graphs without succumbing
to their traditional computational bottlenecks, proving its
scalability for large-scale scenarios.
4.4. Ablation Study
1) Hybrid Retrieval Strategy.Table 2 underscores the
synergy between structural precision and semantic coverage
within our Hybrid Retrieval Strategy. To provide a multi-
dimensional assessment, we report both Exact Match (EM),
which measures the percentage of predicted answers that ex-
7

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
N=1 N=2 N=3 N=4
Expansion Hops (N)025050075010001250150017502000Retrieval Time (s)
39.0s 66.5s577.2s1770.5sRetrieval Time (s)
F1 Score
73.073.574.074.575.075.576.076.577.077.5
F1 Score (%)Balanced Performance
 and EfficiencyPeak: 76.18
Figure 3.Experimental analysis of expansion hops Non Hot-
potQA dataset. The bars indicate retrieval time (left axis), while
the line tracks the QA F1 score (right axis).
actly match the ground truth, and Recall@5, which indicates
whether at least one of the top-5 retrieved passages contains
the gold evidence required to answer the query. The evalua-
tion shows that performance peaks at M= 4 , a 20%+ im-
provement over pure dense retrieval ( M= 0 ), proving that
logical paths effectively resolve multi-hop dependencies.
However, when semantic backfill is constrained ( M= 5 ),
performance slightly declines. This non-monotonic trend
suggests that a purely logical path-guided approach is vul-
nerable to graph incompleteness and structural noise; thus,
maintaining a dense retrieval component is essential as a ro-
bust fallback to ensure comprehensive evidentiary support.
2) Impact of Expansion Hops.To investigate the trade-off
between retrieval latency and QA performance, we con-
ducted an ablation study on the number of expansion hops,
denoted as N. In Fig. 3, varying Nreveals a significant di-
vergence between computational cost and QA performance.
On one hand, the retrieval time exhibits an exponential
growth pattern. While the latency remains manageable at
N= 1 andN= 2 , it surges dramatically to 577.2 s at
N= 3 and becomes prohibitive at N= 4 . On the other
hand, the F1 score follows a non-monotonic trajectory, yet
remarkably remains above 74.5% across all values of N,
demonstrating HELP’s robustness and strong performance.
Specifically, performance peaks at 76.18% when N= 3 , as
deeper exploration retrieves more relevant context. However,
increasing Nto4leads to a performance degradation. This
decline suggests that excessive expansion introduces sig-
nificant noise and irrelevant information, which outweighs
the benefits of broader context coverage. Given that the F1
score remains competitive even at lower hops, N= 2 offers
a compelling balance, achieving high accuracy with only a
fraction of the computational overhead required for N= 3 .
3) Hyperparameter Sensitivity Analysis.To investigate
the impact of the search space configuration, we performed
a grid search over the Initial Triple Size n∈ {1, . . . ,5} and
the Hypernode Beam Size k∈ {30,50,70,100} . In Fig. 4,
the model exhibits significant robustness to hyperparameter
30 50 70 100
Beam Size (k)1 2 3 4 5Initial Triple Size (n)
46.90 47.25 47.11 47.1746.77 47.29 46.72 46.8247.96 48.37 47.99 48.0248.14 47.58 47.96 48.3448.02 47.78 47.92 47.98
46.847.047.247.447.647.848.048.2
F1 Score (%)Figure 4.Hyperparameter sensitivity analysis of the Initial Triple
Size (n) and Hypernode Beam Size ( k) on MuSiQue dataset. The
heatmap reports the F1 scores (%), demonstrating the impact of
varying the seed set size and pruning threshold.
variations, with F1 scores fluctuating within a narrow range
of 46.7% to 48.4%. This stability suggests that HELP is
not overly sensitive to the specific choice of seed set size
or pruning thresholds. Such minimal numerical variance,
coupled with the consistent heatmap pattern, confirms that
our method operates reliably without the need for extensive
hyperparameter tuning, proving its practical scalability.
4) Robustness Across Different Backbones.
To evaluate cross-model generalization, we tested HELP
usingQwen3-30B-A3B-Instruct-2507(Team, 2025) as the
backbone. The results, summarized in Table 4, demon-
strate that HELP consistently outperforms recent graph-
based RAG methods across both simple and complex rea-
soning tasks, achieving an average EM of 42.4% and F1
of 52.6%. Notably, HELP significantly outperforms Hip-
poRAG2 in the long-context LV-Eval task, demonstrating
superior evidentiary localization. These results confirm that
HELP’s gains are model-agnostic and stem from its robust
methodology rather than specific LLM capabilities, proving
its versatility for diverse real-world deployments.
5. Conclusion
In this paper, we introduced HELP, a robust GraphRAG
framework that synergizes structural reasoning with retrieval
efficiency. Through HyperNode Expansion and Logical
Path-Guided Evidence Localization, HELP successfully
transforms isolated knowledge triplets into integrated rea-
soning chains without incurring the high latency of exhaus-
tive graph searches. Empirical results confirm that HELP
significantly outperforms existing graph-based and dense
retrieval methods in both simple and multi-hop reasoning
accuracy and retrieval speed (up to 28.8 ×faster). Our work
bridges the gap between structured knowledge representa-
tion and practical scalability, establishing a new standard
for accurate and efficient GraphRAG methods.
8

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
Impact Statement
This paper presents work on Retrieval-Augmented Gener-
ation (RAG) to advance the field of precise information
retrieval for large language models. While our work may
have various societal implications, we do not identify any
concerns that warrant specific emphasis beyond those gener-
ally associated with large language models and information
retrieval.
References
AI@Meta. Llama 3 model card. 2024. URL
https://github.com/meta-llama/llama3/
blob/main/MODEL_CARD.md.
Bhagdev, R., Chapman, S., Ciravegna, F., Lanfranchi, V .,
and Petrelli, D. Hybrid search: Effectively combining
keywords and semantic searches. In European semantic
web conference, pp. 554–568. Springer, 2008.
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A.,
Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O.,
and Larson, J. From local to global: A graph rag ap-
proach to query-focused summarization. arXiv preprint
arXiv:2404.16130, 2024.
Gao, Y ., Xiong, Y ., Gao, X., Jia, K., Pan, J., Bi, Y ., Dai, Y .,
Sun, J., Wang, H., and Wang, H. Retrieval-augmented
generation for large language models: A survey. arXiv
preprint arXiv:2312.10997, 2(1), 2023.
Guo, Z., Xia, L., Yu, Y ., Ao, T., and Huang, C. Lightrag:
Simple and fast retrieval-augmented generation. arXiv
preprint arXiv:2410.05779, 2024.
Gutiérrez, B. J., Shu, Y ., Qi, W., Zhou, S., and Su, Y . From
rag to memory: Non-parametric continual learning for
large language models. arXiv preprint arXiv:2502.14802 ,
2025.
Han, H., Wang, Y ., Shomer, H., Guo, K., Ding, J., Lei, Y .,
Halappanavar, M., Rossi, R. A., Mukherjee, S., Tang,
X., et al. Retrieval-augmented generation with graphs
(graphrag). arXiv preprint arXiv:2501.00309, 2024.
Han, H., Ma, L., Shomer, H., Wang, Y ., Lei, Y ., Guo, K.,
Hua, Z., Long, B., Liu, H., Aggarwal, C. C., et al. Rag
vs. graphrag: A systematic evaluation and key insights.
arXiv preprint arXiv:2502.11371, 2025.
Ho, X., Nguyen, A.-K. D., Sugawara, S., and Aizawa,
A. Constructing a multi-hop qa dataset for compre-
hensive evaluation of reasoning steps. arXiv preprint
arXiv:2011.01060, 2020.
Hu, Y ., Lei, Z., Zhang, Z., Pan, B., Ling, C., and Zhao, L.
Grag: Graph retrieval-augmented generation, 2025. URL
https://arxiv.org/abs/2405.16506.Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H.,
Chen, Q., Peng, W., Feng, X., Qin, B., et al. A survey on
hallucination in large language models: Principles, taxon-
omy, challenges, and open questions. ACM Transactions
onInformation Systems, 43(2):1–55, 2025.
Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski,
P., Joulin, A., and Grave, E. Unsupervised dense infor-
mation retrieval with contrastive learning. arXiv preprint
arXiv:2112.09118, 2021.
Jiang, J., Zhou, K., Zhao, W. X., Song, Y ., Zhu, C., Zhu,
H., and Wen, J.-R. Kg-agent: An efficient autonomous
agent framework for complex reasoning over knowledge
graph. In Proceedings ofthe63rd Annual Meeting of
theAssociation forComputational Linguistics (V olume
1:Long Papers), pp. 9505–9523, 2025.
Jimenez Gutierrez, B., Shu, Y ., Gu, Y ., Yasunaga, M.,
and Su, Y . Hipporag: Neurobiologically inspired long-
term memory for large language models. Advances
inNeural Information Processing Systems , 37:59532–
59569, 2024.
Jin, B., Xie, C., Zhang, J., Roy, K. K., Zhang, Y ., Li, Z., Li,
R., Tang, X., Wang, S., Meng, Y ., et al. Graph chain-of-
thought: Augmenting large language models by reason-
ing on graphs. arXiv preprint arXiv:2404.07103, 2024.
Lee, C., Roy, R., Xu, M., Raiman, J., Shoeybi, M., Catan-
zaro, B., and Ping, W. Nv-embed: Improved techniques
for training llms as generalist embedding models. arXiv
preprint arXiv:2405.17428, 2024.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel,
T., et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks. NeurIPS, 33:9459–9474, 2020.
Li, Z., Zhang, X., Zhang, Y ., Long, D., Xie, P., and Zhang,
M. Towards general text embeddings with multi-stage
contrastive learning. arXiv preprint arXiv:2308.03281 ,
2023.
Luo, H., Chen, G., Zheng, Y ., Wu, X., Guo, Y ., Lin,
Q., Feng, Y ., Kuang, Z., Song, M., Zhu, Y ., et al.
Hypergraphrag: Retrieval-augmented generation via
hypergraph-structured knowledge representation. arXiv
preprint arXiv:2503.21322, 2025.
Ma, S., Xu, C., Jiang, X., Li, M., Qu, H., and Guo, J.
Think-on-graph 2.0: Deep and interpretable large lan-
guage model reasoning with knowledge graph-guided
retrieval. arXiv e-prints, pp. arXiv–2407, 2024.
Mallen, A., Asai, A., Zhong, V ., Das, R., Khashabi, D., and
Hajishirzi, H. When not to trust language models: Inves-
tigating effectiveness of parametric and non-parametric
9

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
memories. In Rogers, A., Boyd-Graber, J., and Okazaki,
N. (eds.), Proceedings ofthe61st Annual Meeting of
theAssociation forComputational Linguistics (V olume
1:Long Papers) , pp. 9802–9822, Toronto, Canada, July
2023. Association for Computational Linguistics. doi:
10.18653/v1/2023.acl-long.546.
Mavromatis, C. and Karypis, G. Gnn-rag: Graph neural
retrieval for large language model reasoning, 2024. URL
https://arxiv.org/abs/2405.20139.
Muennighoff, N., Hongjin, S., Wang, L., Yang, N., Wei, F.,
Yu, T., Singh, A., and Kiela, D. Generative representa-
tional instruction tuning. In ICLR, 2024.
Ni, J., Qu, C., Lu, J., Dai, Z., Abrego, G. H., Ma, J., Zhao,
V ., Luan, Y ., Hall, K., Chang, M.-W., et al. Large dual
encoders are generalizable retrievers. In EMNLP , pp.
9844–9855, 2022.
Peng, B., Zhu, Y ., Liu, Y ., Bo, X., Shi, H., Hong, C., Zhang,
Y ., and Tang, S. Graph retrieval-augmented generation:
A survey. ACM Transactions onInformation Systems ,
2024.
Plaat, A., Wong, A., Verberne, S., Broekens, J., and
Van Stein, N. Multi-step reasoning with large language
models, a survey. ACM Computing Surveys, 2025.
Robertson, S. E. and Walker, S. Some simple
effective approximations to the 2-poisson model
for probabilistic weighted retrieval. In SIGIR’94:
Proceedings oftheSeventeenth Annual International
ACM-SIGIR Conference onResearch andDevelopment
inInformation Retrieval, organised byDublin City
University, pp. 232–241. Springer, 1994.
Sarthi, P., Abdullah, S., Tuli, A., Khanna, S., Goldie, A., and
Manning, C. D. Raptor: Recursive abstractive processing
for tree-organized retrieval. In ICLR, 2024.
Sun, J., Xu, C., Tang, L., Wang, S., Lin, C., Gong, Y .,
Ni, L. M., Shum, H.-Y ., and Guo, J. Think-on-graph:
Deep and responsible reasoning of large language model
on knowledge graph. arXiv preprint arXiv:2307.07697 ,
2023.
Team, Q. Qwen3 technical report, 2025. URL https:
//arxiv.org/abs/2505.09388.
Thakur, N., Reimers, N., Rücklé, A., Srivastava, A., and
Gurevych, I. Beir: A heterogenous benchmark for zero-
shot evaluation of information retrieval models. arXiv
preprint arXiv:2104.08663, 2021.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabhar-
wal, A. MuSiQue: Multihop questions via single-hop
question composition. Transactions oftheAssociationforComputational Linguistics , 10:539–554, 2022. doi:
10.1162/tacl_a_00475.
Trivedi, H., Balasubramanian, N., Khot, T., and Sabhar-
wal, A. Interleaving retrieval with chain-of-thought
reasoning for knowledge-intensive multi-step questions.
InProceedings ofthe61st annual meeting ofthe
association forcomputational linguistics (volume 1:
long papers), pp. 10014–10037, 2023.
Wang, M., Tan, B., Gao, Y ., Jin, H., Zhang, Y ., Ke, X., Xu,
X., and Zhu, Y . Balancing the blend: An experimental
analysis of trade-offs in hybrid search. arXiv preprint
arXiv:2508.01405, 2025.
Wang, Y ., Ren, R., Li, J., Zhao, W. X., Liu, J., and Wen,
J.-R. Rear: A relevance-aware retrieval-augmented frame-
work for open-domain question answering. arXiv preprint
arXiv:2402.17497, 2024.
Xiang, Z., Wu, C., Zhang, Q., Chen, S., Hong, Z., Huang, X.,
and Su, J. When to use graphs in rag: A comprehensive
analysis for graph retrieval-augmented generation. arXiv
preprint arXiv:2506.05690, 2025.
Yang, M., Wang, H., Wei, Z., Wang, S., and Wen, J.-R. Effi-
cient algorithms for personalized pagerank computation:
A survey. IEEE Transactions onKnowledge andData
Engineering, 36(9):4582–4602, 2024.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W., Salakhut-
dinov, R., and Manning, C. D. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering. In
EMNLP, pp. 2369–2380, 2018.
Yasunaga, M., Ren, H., Bosselut, A., Liang, P., and
Leskovec, J. Qa-gnn: Reasoning with language mod-
els and knowledge graphs for question answering. arXiv
preprint arXiv:2104.06378, 2021.
Yuan, T., Ning, X., Zhou, D., Yang, Z., Li, S., Zhuang, M.,
Tan, Z., Yao, Z., Lin, D., Li, B., et al. Lv-eval: A balanced
long-context benchmark with 5 length levels up to 256k.
arXiv preprint arXiv:2402.05136, 2024.
Zhang, Q., Chen, S., Bei, Y ., Yuan, Z., Zhou, H., Hong, Z.,
Chen, H., Xiao, Y ., Zhou, C., Dong, J., et al. A survey
of graph retrieval-augmented generation for customized
large language models. arXiv preprint arXiv:2501.13958 ,
2025.
Zhao, Y ., Zhu, J., Guo, Y ., He, K., and Li, X. Eˆ 2graphrag:
Streamlining graph-based rag for high efficiency and ef-
fectiveness. arXiv preprint arXiv:2505.24226, 2025.
Zhuang, L., Chen, S., Xiao, Y ., Zhou, H., Zhang, Y ., Chen,
H., Zhang, Q., and Huang, X. Linearrag: Linear graph
retrieval augmented generation on large-scale corpora.
arXiv preprint arXiv:2510.10114, 2025.
10

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
A. Appendix
This appendix provides supplementary information to the main paper. The contents are organized as follows:
• Section A.1 details the core hyperparameter setting used in the HELP framework.
•Section A.2 presents additional experimental results across various retrieval baselines using theQwen3-30B-A3B-
Instruct-2507backbone.
• Section A.3 describes the implementation details and reproduction alignment of the baseline methods to ensure a fair
comparison.
• Section A.4 provides a qualitative analysis via a detailed case study on multi-hop reasoning performance.
A.1. HyperParameter Setting
The core hyperparameters for HELP were configured as follows.
Table 3.Hyperparameter settings for the HELP framework.
Symbol Description Value
NNumber of expansion hops 2
nInitial seed triple size 3
kHypernode beam search size 50
MQuota for logical path-guided passages 4
LTotal context size (M+ Dense passages) 5
A.2. Different LLM Backbones
To further validate that the performance gains of HELP are architectural rather than dependent on a specific language
model, we conducted extensive evaluations usingQwen3-30B-A3B-Instruct-2507as the generation backbone. The results,
summarized in Table 4, demonstrate that HELP consistently outperforms recent graph-based retrieval methods across both
simple and complex reasoning tasks.
Table 4.Performance comparison (EM / F1 scores) of various retrieval methods across different tasks, with all models evaluated using
Qwen3-30B-A3B-Instruct-2507as the backbone. In each cell, the first value represents the Exact Match (EM) score, and the second
represents the F1 score. The best results are inbold. The†symbol indicates a failure in graph construction due to large corpus size.
Simple QA Multi-Hop QA
Retrieval NQ PopQA MuSiQue 2Wiki HotpotQA LV-Eval Avg
Graph-based RAG Methods
HippoRAG2 (Gutiérrez et al., 2025) 40.4 / 54.4 42.2 / 56.134.3/45.860.2 / 69.4 58.4 / 72.9 7.26 / 9.67 40.5 / 51.4
LinearRAG (Zhuang et al., 2025) 31.6 / 43.8 6.6 / 21.9 18.7 / 28.1 42.4 / 53.2 39.6 / 52.4 6.5/ 10.7 24.2 / 35.0
HyperGraphRAG (Luo et al., 2025) 30.3 / 42.1 7.9 / 22.5 22.7 / 36.9 45.5 / 58.6 54.3 / 70.2 N/A†-
HELP (Ours) 43.0/56.9 44.5/57.4 33.7 / 44.7 62.0/69.5 60.4/73.9 10.5/12.9 42.4/52.6
A.3. Baseline Implementation Details
To ensure a fair and comprehensive comparison, all baseline methods we reproduced were evaluated under a strictly
controlled environment. Specifically, we aligned these baselines with HELP by using an identical corpus, evaluation datasets,
and the same LLM backbones for answer generation. Beyond these necessary alignments, all other internal configurations
were kept strictly consistent with their original official implementations as reported in their respective literature. This ensures
that the observed performance gains of HELP are attributable to its structural and logical innovations rather than disparate
parameter tuning or modified baseline settings.
11

HELP: HyperNode Expansion and Logical Path-Guided Evidence Localization for Accurate and Efficient GraphRAG
A.4. Case Study
For this case study, HELP is configured with the hyperparameters specified in Table 3 and utilizesLlama-3.3-70B-Instructas
the underlying LLM backbone. Table 5 presents a qualitative comparison of multi-hop reasoning between HELP and several
baselines. The query requires a 2-hop link: Princess Elene of Georgia →Solomon II of Imereti →Prince Archil of Imereti.
LinearRAG and HyperGraphRAG fail due to semantic distractions, retrieving irrelevant information. While HippoRAG2
retrieves the correct passages, it fails to produce the final answer. In contrast, our method, HELP, anchors the reasoning
process by expanding from the Initial Triple Seed HyperNodes. This strategy effectively filters out distractor entities and
successfully reconstructs the reasoning path to identify the final answer.
Table 5.A case study illustrating how different methods handle a 2-hop query requiring cross-passage inference. HELP successfully
leverages seed HyperNodes to anchor the correct reasoning path, whereas other baselines suffer from semantic drift or hallucination.
QuestionWho is the husband of Princess Elene Of Georgia?
Ground TruthPrince Archil of Imereti
Reasoning ChainPrincess Elene of Georgiamother of− − − − →Solomon II of Imeretihad father− − − − − →Prince Archil of Imereti
LinearRAG Retrieved Passages:
1)✗Grand Duchess Elena Vladimirovna of Russia: ...husband was Prince Nicholas...
2)✗Prince Adarnase of Kartli: ...natural son of Levan of Kartli by a concubine...
3)✗Princess Charlotte of Württemberg: ...the wife of Grand Duke Michael Pavlovich of Russia...
4)✗Clous van Mechelen: ...is a Dutch musician, arranger, and actor...
5)✗Bernhard III, Prince of Anhalt-Bernburg: ...was the eldest son of bernhard ii...
Prediction: ✗There is no direct information provided about the husband of Princess Elene of Georgia
in the given passages.
HyperGraphRAG Retrieved Passages:
1)✗Gundobad: ...He was the husband of Caretene.
2)✗Eunoë (wife of Bogudes): Eunoë was the wife of Bogudes, King of Mauretania.
3)✗Engelbert III of the Mark: ...Adolph was the eldest son of Count Adolph II...
4)✗Megingoz of Guelders: ...He married Gerberga of Lorraine...
5)✗Princess Rodam of Kartli: ...Princess Rodam married King George VII...
Prediction:✗King George VII of Imereti
HippoRAG2 Retrieved Passages:
1)✓Princess Elene of Georgia: ...She was the mother of Solomon II of Imereti...
2)✗Grand Duchess Elena Vladimirovna of Russia: ...husband was Prince Nicholas...
3)✗Princess Rodam of Kartli: ...Princess Rodam married King George VII...
4)✓Solomon II of Imereti: ...He was born as David to Prince Archil of Imereti...
5)✗Grand Duchess Elena Pavlovna of Russia: ...his second wife Sophie Dorothea of Württemberg...
Prediction:✗None mentioned.
HELP Expansion from Initial Triples as Seed HyperNodes
("princess elene of georgia", "daughter of", "heraclius ii")
("princess elene of georgia", "mother of", "solomon ii of imereti")
("princess elene of georgia", "born in", "georgia")
Retrieved Passages:
1)✓Princess Elene of Georgia: ...She was the mother of Solomon II of Imereti...
2)✓Solomon II of Imereti: ...He was born as David to Prince Archil of Imereti...
3)✗Grand Duchess Elena Vladimirovna of Russia: ...husband was Prince Nicholas...
4)✗Princess Rodam of Kartli: ...Princess Rodam married King George VII...
5)✗Grand Duchess Elena Pavlovna of Russia: ...his second wife Sophie Dorothea of Württemberg...
Prediction:✓Prince Archil of Imereti
12