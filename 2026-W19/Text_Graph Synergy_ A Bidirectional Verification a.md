# Text-Graph Synergy: A Bidirectional Verification and Completion Framework for RAG

**Authors**: Jiarui Zhong, Hong Cai Chen

**Published**: 2026-05-07 03:49:37

**PDF URL**: [https://arxiv.org/pdf/2605.05643v1](https://arxiv.org/pdf/2605.05643v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become a core paradigm for enhancing factual grounding and multi-hop reasoning in Large Language Models (LLMs). Traditional text-based RAG often retrieves logically irrelevant pseudo-evidence, while graph-based RAG is frequently hindered by search-time pruning, which may discard potentially valid reasoning paths. Existing hybrid approaches primarily adopt simple evidence concatenation or unidirectional enhancement, which fails to address the fundamental "Information Island" problem caused by asymmetric reasoning flows between unstructured text and structured graphs. We propose \textbf{TGS-RAG}, a unified framework for \textbf{T}ext-\textbf{G}raph \textbf{S}ynergistic enhancement. TGS-RAG introduces a bidirectional mechanism: (i) a \textbf{Graph-to-Text} channel that employs a Global Voting strategy from visited graph nodes to re-rank and refine textual evidence, filtering out semantic noise; and (ii) a \textbf{Text-to-Graph} channel that utilizes the \textbf{Memory-based Orphan Entity Bridging} algorithm. This algorithm utilizes textual cues to proactively resurrect valid but previously pruned reasoning paths from the search history without additional database overhead. Experimental results on multiple multi-hop reasoning benchmarks demonstrate that TGS-RAG significantly outperforms state-of-the-art baselines, achieving a superior balance between retrieval precision and computational efficiency.

## Full Text


<!-- PDF content starts -->

Text-Graph Synergy:
A Bidirectional Verification and Completion Framework for RAG
Jiarui Zhong,Hong Cai Chen∗
School of Automation, Southeast University, Nanjing 210096, China
220245143@seu.edu.cn, chenhc@seu.edu.cn
Abstract
Retrieval-Augmented Generation (RAG) has be-
come a core paradigm for enhancing factual
grounding and multi-hop reasoning in Large Lan-
guage Models (LLMs). Traditional text-based
RAG often retrieves logically irrelevant pseudo-
evidence, while graph-based RAG is frequently
hindered by search-time pruning, which may dis-
card potentially valid reasoning paths. Exist-
ing hybrid approaches primarily adopt simple ev-
idence concatenation or unidirectional enhance-
ment, which fails to address the fundamental ”In-
formation Island” problem caused by asymmet-
ric reasoning flows between unstructured text and
structured graphs. We proposeTGS-RAG, a uni-
fied framework forText-GraphSynergistic en-
hancement. TGS-RAG introduces a bidirectional
mechanism: (i) aGraph-to-Textchannel that em-
ploys a Global V oting strategy from visited graph
nodes to re-rank and refine textual evidence, filter-
ing out semantic noise; and (ii) aText-to-Graph
channel that utilizes theMemory-based Orphan
Entity Bridgingalgorithm. This algorithm uti-
lizes textual cues to proactively resurrect valid
but previously pruned reasoning paths from the
search history without additional database over-
head. Experimental results on multiple multi-hop
reasoning benchmarks demonstrate that TGS-RAG
significantly outperforms state-of-the-art baselines,
achieving a superior balance between retrieval pre-
cision and computational efficiency.
1 Introduction
Large Language Models (LLMs) have achieved remarkable
performance in natural language understanding and gener-
ation across a wide range of tasks. However, despite their
impressive parametric capacity, LLMs remain fundamentally
limited by their reliance on static internal knowledge and
their tendency to produce hallucinated or logically inconsis-
tent responses [Jiet al., 2023], particularly in knowledge-
intensive and multi-hop reasoning scenarios. Retrieval-
∗Corresponding author.
Figure 1: Comparison between isolated retrieval paradigms and
the TGS-RAG framework. (a) Existing paradigms suffer from the
”Information Island” gap: text-based methods often retrieve se-
mantically similar but logically irrelevantpseudo-evidence, while
graph-based methods are hindered bybroken reasoning pathsdue
to search-time pruning. (b) TGS-RAG bridges this gap through a
bidirectional synergy mechanism, where graph structure guides the
re-ranking of text to identify true evidence, and textual context facil-
itates the recovery of pruned but potentially valid reasoning paths.
Augmented Generation (RAG) has therefore emerged as a
central paradigm for grounding LLM outputs in external, ver-
ifiable evidence, enabling more factual, controllable, and in-
terpretable generation [Lewiset al., 2020].
Existing RAG systems predominantly follow two distinct
paradigms.Text-based RAGretrieves relevant information
from large unstructured corpora using dense vector similar-
ity. While this paradigm offers high coverage, it is prone to
retrieving pseudo-evidence [Cossio, 2025]—textual chunks
that are semantically similar to the query but logically irrele-
vant, and struggle with complex multi-hop reasoning. In con-
trast,Graph-based RAGleverages structured Knowledge
Graphs (KGs) to provide high logical interpretability. How-
ever, these systems are often constrained by both structural
sparsity in KGs and search-time pruning that may discard po-
tentially useful reasoning paths.
Recent research attempts to combine Text-based RAG and
Graph-based RAG to exploit their complementary strengths.
However, these approaches often rely on simple evidence
concatenation or pipeline-style augmentation, which treat
textual and graphical evidence as independent sources, fail-arXiv:2605.05643v1  [cs.AI]  7 May 2026

ing to achieve deep integration. Furthermore, unidirectional
enhancement frameworks (e.g., KG-infused RAG) typically
utilize one modality merely as an auxiliary signal for the
other. Such designs fail to establish a closed-loop interac-
tion, leaving the deeper potential of mutual verification and
co-discovery largely unexplored.
At the core of this limitation lies theInformation Island
problem: textual evidence and graphical evidence are re-
trieved and processed in isolation, without a mechanism for
mutual validation or collaborative reasoning. Specifically,
current systems lack a principled framework in which (i) the
structured logical constraints encoded in graphs can guide
and refine text retrieval, while simultaneously (ii) the rich
contextual cues present in unstructured text can validate, en-
rich, and recover pruned but potentially useful graph reason-
ing paths. In practice, this often means that the text and graph
channels retrieve evidence that appears relevant on their own,
but fails to mutually support a coherent reasoning chain.
To address these challenges, we proposeText-Graph Syn-
ergy RAG (TGS-RAG)1, a novel framework that enables
deep, bidirectional integration between unstructured text and
structured knowledge graphs during retrieval. TGS-RAG in-
troduces a bidirectional enhancement mechanism: aGraph-
to-Textchannel that utilizes structured logic to re-rank textual
chunks, and aText-to-Graphchannel that leverages contex-
tual clues to validate and complete KG paths. Our main con-
tributions are as follows:
•Bidirectional Synergy Framework:We propose TGS-
RAG, a unified framework that breaks the ”informa-
tion island” barrier by establishing a closed-loop feed-
back mechanism: graph structures guide the re-ranking
of textual evidence (Graph-to-Text), while textual con-
text validates and repairs graph reasoning paths (Text-
to-Graph). This closed-loop design allows the two chan-
nels to move beyond simple coexistence and mutually
verify the evidence retrieved by each other.
•Memory-based Orphan Entity Bridging:We intro-
duce a novel algorithm that addresses the loss of poten-
tially useful reasoning paths during beam-search prun-
ing by treating pruned nodes as adeferred reasoning
memory. By leveraging textual cues to ”resurrect” these
orphan entities, we recover potentially valid reasoning
paths discarded during initial search.
•Cost-Effective Dual-Channel Reasoning:We design a
synergistic scoring mechanism that combines semantic
similarity with structural voting. This approach effec-
tively filters out pseudo-evidence and achieves an opti-
mal Pareto frontier, delivering state-of-the-art accuracy
with significantly lower token consumption than global
graph indexing methods.
•Empirical Superiority:Extensive experiments on
MuSiQue and HotpotQA demonstrate that TGS-RAG
significantly outperforms existing text-based, graph-
based, and hybrid baselines, particularly in scenarios re-
quiring complex multi-hop reasoning across disjointed
evidence.
1https://github.com/EvannZhongg/TGS RAG.git2 Related Work
The evolution of RAG has transitioned from simple text
matching to complex multi-modal and structural fusion. Our
work builds upon three main research areas.
2.1 Unstructured Retrieval for RAG
Traditional RAG systems primarily rely on unstructured re-
trieval, where dense vector representations facilitate seman-
tic matching between queries and documents. Dense Pas-
sage Retrieval (DPR) Karpukhinet al.[2020] pioneered
the dual-encoder architecture for open-domain question an-
swering, significantly improving recall over lexical meth-
ods. Subsequent models like ColBERT Khattab and Za-
haria [2020] introduced late interaction mechanisms to cap-
ture fine-grained token-level similarities. While these models
excel at broad knowledge coverage, they often struggle with
multi-hop queries where semantic similarity does not neces-
sarily imply logical relevance, leading to the retrieval of mis-
leading ”pseudo-evidence.”
2.2 Structured Retrieval for RAG
To address the logical limitations of text-based retrieval, re-
cent studies have explored Knowledge Graphs (KGs) as struc-
tured knowledge sources. Methods in this category typi-
cally extract subgraphs or relational paths to provide deter-
ministic evidence for LLMs. For instance,G-RetrieverHe
et al.[2024] employs a GNN-based adapter to filter irrele-
vant nodes and edges, targeting efficient subgraph retrieval
for textual graph understanding. Similarly, KG-based rea-
soning frameworks like ToG Sunet al.[2024] utilize LLMs
as agents to execute path-searching algorithms (e.g., beam
search) over structured facts. Beyond path reasoning, recent
graph-based indexing approaches leverage LLM-generated
structures to capture global information.GraphRAGEdge
et al.[2024] constructs hierarchical community summaries
to answer global queries but suffers from high indexing costs.
In contrast,LightRAGGuoet al.[2025] introduces a dual-
level retrieval paradigm incorporating both graph structures
and vector representations. However, these indexing-focused
approaches primarily address retrieval efficiency and cover-
age, but are less equipped to handle scenarios where poten-
tially useful reasoning paths are discarded during search.
2.3 Text-Graph Integration Paradigms
The integration of text and graphs has become a promising
direction for robust RAG systems. Early approaches typi-
cally employed unidirectional enhancement strategies. For
instance, KG-Infused RAG utilizes graph entities to expand
queries for text retrieval, while frameworks likeKG2RAG
Zhuet al.[2025] adopt a ”Semantic-to-Graph” paradigm, us-
ing text seeds from initial semantic retrieval to expand graph
components. However, these methods often face a ”unidi-
rectional bottleneck,” where the strengths of one modality do
not actively feed back to correct or complete the other in real-
time.
To overcome these limitations, recent frameworks have
moved toward cross-source validation.Think-on-Graph
2.0 (ToG-2)Maet al.[2025] introduces a tight-coupling

Figure 2: The overall architecture of TGS-RAG. The framework operates in three phases: (1)Knowledge Base Construction, where text
chunks and entities are mapped in a unified vector DB; (2)Bidirectional Synergistic Retrieval, the core stage utilizingSemantic Beam
Searchto accumulate aVisited Memorycontaining both selected and pruned nodes. This memory facilitatesGraph-Guided Re-rankingvia
Global Votingand enablesText-Guided Path Completionbyresurrectingpruned orphan entities from memory to bridge logical gaps; and
(3)Generation, synthesizing the refined evidence.
paradigm that iteratively alternates between KG-based graph
retrieval and document-based context retrieval, using text ev-
idence to prune graph paths and graph structures to guide text
retrieval. While TGS-RAG shares the high-level goal of mu-
tual verification, the underlying mechanisms differ substan-
tially. ToG-2 follows a subtractive strategy, repeatedly prun-
ing and re-expanding the search space through costly itera-
tive retrieval. In contrast, TGS-RAG adopts an additive strat-
egy via Memory-based Orphan Entity Bridging, which treats
pruning as postponement rather than elimination and directly
resurrects pruned paths from the semantic beam search his-
tory. This retrospective repair mechanism enables efficient
recovery of logical chains that are difficult to preserve un-
der sparse graph connectivity and search-time pruning with-
out additional database queries.
3 The TGS-RAG Framework
As illustrated in Figure 2, TGS-RAG is designed as a two-
stage framework that tightly integrates unstructured text and
structured knowledge graphs for retrieval-augmented reason-
ing. It consists of an offlineKnowledge Base Construction
stage, responsible for extracting and fusing knowledge from
unstructured documents; and an onlineBidirectional Syn-
ergistic Retrievalstage, which performs mutual verification
and enhancement between text and graph evidence during in-
ference.
3.1 Knowledge Base Construction
Given an unstructured document corpusC, TGS-RAG con-
structs a unified knowledge base through an offline process-
ing pipeline. We leverage Large Language Models (LLMs) toextract salient entitiesEand their semantic relationsR(see
extraction prompt in Appendix A.1), forming a global knowl-
edge graphG= (E, R).
Crucially, we establish abidirectional mappingMbe-
tween the text corpusCand the knowledge graphG. Each
relational triplet inGis explicitly linked to its originating
text chunks, while each text chunk is annotated with the en-
tities and relations it contains. This mapping transforms the
knowledge graph from a standalone structure into a reasoning
interface over text. Formally, the knowledge base is defined
as a triplet:
KB= (C, G, M)(1)
whereMrepresents the bidirectional links betweenCandG.
This unified and interconnected knowledge base serves as the
foundation for the proposed bidirectional enhanced retrieval
framework.
3.2 Dual-Channel Initial Retrieval
Given a user queryq, TGS-RAG first extracts a set of query-
relevant entity mentions using an LLM. These entities are em-
bedded to obtain vectorsv E, which are then used to retrieve
a seed entity setE seedfrom the knowledge graphG.
Starting fromE seed, TGS-RAG performs two complemen-
tary retrieval processes in parallel: a Text Channel and a
Graph Channel. This dual-channel design intentionally cap-
tures heterogeneous evidence signals that exhibit different
failure modes.
•Text Channel: This channel performs standard seman-
tic retrieval to quickly recall textual evidence directly
related to the user query. We use the query vectorv q

Figure 3: Illustration of the Memory-based Orphan Entity Bridging algorithm. The process operates in three steps: (1)The Visited Memory:
Semantic Beam Search generates a set ofActive Paths(P initial , blue nodes) while implicitly storing pruned nodes in theVisited Memory
(Evisited , grey area); (2)Contextual Recall: Contextual clues from retrieved text chunks (C initial ) perform aContextual Scanto identify
relevant but pruned ”orphan entities”; (3)Path Resurrection: The pruning decision is reversed, and theResurrected Path(p bridge , orange)
connecting the resurrected entity to the main subgraph is recovered without additional database queries.
of user queryqto perform vector indexing on the text
chunk collectionC, retrieving the initial text setC initial
with the highest similarity tov q.
•Graph Channel: Starting from the seed entity set
Eseed, the Semantic Beam Search algorithm (see Algo-
rithm 1 in Appendix B) computes, at each hop, the co-
sine similarity between the embeddings of neighboring
nodes and the query vectorv q. Only the top-K(Beam
Width) most semantically relevant paths are retained for
the next expansion step. This heuristic pruning ensures
that the retrieval focuses on the semantic neighborhood
of the query, yielding a high-quality path setP initial
within a preset depthd.
The outcome of this stage is two heterogeneous evidence
sets:C initial, which emphasizes semantic recall, andP initial,
which captures structured reasoning paths.
3.3 Synergistic Fusion and Enhancement
After obtaining the initial text evidenceC initial and graph
path setP initial , the TGS-RAG introduces a bidirectional
scoring and discovery mechanism that enables mutual ver-
ification and refinement of information. This process in-
cludes two enhancement directions:Graph-to-TextandText-
to-Graph.
Graph-to-Text Synergy
This process utilizes structured knowledge from the graph
channel to guide the rediscovery and re-ranking of text evi-
dence, rather than relying only on surface-level similarity to
the query.
1.Chunk Recommendation: We aggregate all entities
visited during the Semantic Beam Search (including
those in pruned paths) to form a global visited set
Evisited . These entities act as ”recommenders” to ”vote”
for their source text chunks. The graph-recommended
chunks are merged withC initial to create a broader can-
didate poolC candidate , allowing text chunks that have
low semantic similarity but are structurally relevant to
be recalled.2.Chunk Re-ranking: We calculate a recommendation
scoreRec(c)for each text chunkc∈C candidate , pro-
portional to the number of entities recommending it. The
final scoreScore final(c)is a weighted fusion of its orig-
inal semantic similarity and the graph recommendation
score:
Score final(c) =α·Norm 
sim(v q, vc)
+ (1−α)·Norm 
Rec(c)
(2)
whereαis a hyperparameter balancing the two scores.
Text-to-Graph Synergy
This process utilizes the rich contextual information in the
text channel to verify and enhance the structured paths dis-
covered in the graph channel. As illustrated in Figure 3, this
involves confirming existing paths and resurrecting pruned
connections.
For each pathp∈P initial , we first calculate a base qual-
ity scoreScore base(p)based on the semantic similarity of its
components to the queryqand their structural importance.
1.Path Confirmation: The path is validated by check-
ing the intersection of entities on the path and entities
contained inC initial . If a path receives support from
text evidence (i.e., its entities appear in the retrieved text
chunks), its score is boosted:
Score conf(p) =Score base(p)
+ϵ· |Entities(p)∩Entities(C initial )|(3)
whereϵis the weight for text confirmation reward.
2.Memory-based Orphan Entity Bridging: To discover
potential knowledge ignored by the initial path prun-
ing, we defineOrphan EntitiesE orphan as key entities
that are activated by the retrieved text evidenceC initial
but are absent from the initially selected graph paths
Pinitial . Instead of performing a costly new search or
re-expansion, we leverage theVisited Memoryas ade-
ferred reasoning buffer. If an orphan entity exists in the
visited but pruned nodes, the original pruning decision
is reversed by directly replaying its stored path, thereby

Dataset MethodRetrieval Metrics Generation
Strict Hit Rate (%) Recall (%) Precision (%) Support F1 (%) LLM Judge Acc (%)
MuSiQueNaive RAG 14.23 43.96 21.17 28.11 21.51
Hybrid RAG 14.23 43.98 21.18 28.13 21.56
GraphRAG (Local) 30.70 59.62 7.19 12.59 40.67
LightRAG (Max) 33.54 60.44 20.23 18.58 40.81
KG2RAG 9.10 31.02 11.50 15.61 15.06
TGS-RAG (Ours) 34.84 62.01 24.85 20.60 41.37
HotpotQANaive RAG 45.01 62.48 26.99 38.56 59.41
Hybrid RAG 46.05 65.45 27.62 39.42 61.92
GraphRAG (Local) 55.78 70.15 10.11 20.85 71.79
LightRAG (Max) 60.55 72.04 10.23 15.58 72.09
KG2RAG 38.83 58.92 16.49 25.40 60.46
TGS-RAG 62.00 77.55 27.41 26.06 79.99
Table 1: Main results on MuSiQue and HotpotQA datasets. Retrieval performance is evaluated using Provenance Mapping. Generation
quality is evaluated by DeepSeek-V3.2 as a Judge.
resurrecting a valid reasoning chain without additional
database queries. We check if any orphan entity exists
in the set of visited nodes that were pruned during the
beam selection (detailed procedure in Algorithm 2, Ap-
pendix B). If found, the path leading to this orphan entity
is ”resurrected” as a bridge pathp bridge . This mecha-
nism efficiently recovers logical connections that are se-
mantically distant but contextually bridged by the text,
without incurring additional database query overhead.
3.4 Context Consolidation and Answer Generation
After precise scoring and ranking, the Top-K pathsP Top−K
and text chunksC Top−K are selected. EachP Top−K is for-
matted into clear natural language reasoning chains. If sup-
ported byp bridge , these bridging paths are also appended as
supporting metadata. Finally, the consolidated context of
PTop−K andC Top−K , along with the original queryq, is
submitted to the LLM to generate a logically coherent and
evidence-grounded answer.
4 Experiments
In this section, we present a comprehensive evaluation of
TGS-RAG against state-of-the-art retrieval paradigms. We
aim to validate the effectiveness of our bidirectional synergy
mechanism in resolving the ”information island” problem in-
herent in multi-hop reasoning tasks.
4.1 Experimental Setup
Datasets
We utilize two widely recognized benchmarks for multi-hop
question answering:
•MuSiQue-AnsTrivediet al.[2022]: A dataset char-
acterized by connected reasoning chains (typically 2-4
hops) and low lexical overlap between connected docu-
ments. It serves as the primary testbed for our ”Orphan
Entity Bridging” mechanism.
•HotpotQA (Distractor)Yanget al.[2018]: A dataset
requiring reasoning over two supporting documentsamidst logically irrelevant but semantically similar dis-
tractor documents.
Implementation Details
To ensure a robust and cost-effective experimental environ-
ment, we employGPT-4o-minias the backbone LLM for
all tasks, including Knowledge Graph construction, entity
extraction, and final answer generation. For semantic rep-
resentation, we utilizeQwen3-Embedding-0.6B[Zhanget
al., 2025] as the unified embedding model for all queries,
text chunks, and graph entities. For evaluation, we utilize
DeepSeek-V3.2 [Liuet al., 2024] as an impartial LLM-as-a-
Judge [Liet al., 2025] (evaluation prompt provided in Ap-
pendix A.4). We instruct DeepSeek-V3.2 to evaluate seman-
tic equivalence between the generated answer and the ground
truth, accounting for valid aliases while penalizing hallucina-
tions.
Baselines
We compare TGS-RAG with three categories of retrieval
methods. All baselines are implemented following their orig-
inal configurations, utilizing identical LLM backbones (GPT-
4o-mini) and consistent text chunking strategies to ensure a
fair comparison. Detailed implementation parameters and
sensitivity analyses are provided in Appendix C and E.
•Naive RAG:Standard dense retrieval using vector sim-
ilarity.
•Hybrid RAG (Dense+Sparse Text Retrieval):A com-
bination of dense vector retrieval and sparse keyword re-
trieval (BM25/TF-IDF) fused via Reciprocal Rank Fu-
sion (RRF) Cormacket al.[2009].
•Graph-centric Methods:We evaluate state-of-the-art
graph retrieval systems, includingGraphRAG[Edge
et al., 2024] andLightRAG[Guoet al., 2025]. To
rigorously benchmark the performance ceiling of each
method, we employ their most comprehensive retrieval
modes and include their associated textual evidence
or community summaries where applicable: ”Local

Dataset MethodToken Usage
Embedding LLM Prompt LLM Completion Total Cost
MuSiQueNaive RAG 54,735 1,756,984 13,461 1,825,180
Hybrid RAG 54,735 1,758,189 13,437 1,826,361
GraphRAG (Local) 51,845 13,908,681 554,681 14,515,224
LightRAG (Max) 174,024 16,541,984 374,635 17,090,643
KG2RAG 1,795,495 909,001 10,213 2,714,709
TGS-RAG 78,030 5,113,149 252,224 5,443,403
HotpotQANaive RAG 2,202,470 59,485,904 345,673 62,034,047
Hybrid RAG 2,280,240 63,283,087 346,239 65,909,566
GraphRAG (Local) 1,475,556 634,239,236 10,674,261 646,389,053
LightRAG (Max) 4,574,224 746,163,808 7,163,934 757,901,966
KG2RAG 2,119,623 41,072,699 398,399 43,590,721
TGS-RAG 3,406,165 187,692,537 26,438,352 217,537,053
Table 2: Efficiency and token usage analysis. Detailed breakdown of token consumption across MuSiQue and HotpotQA datasets. Grouping
by dataset allows for a direct comparison of computational costs among different retrieval paradigms.
Search” for GraphRAG (focusing on entity-centric rea-
soning) and ”Max” mode for LightRAG (integrating
both global and local signals). Additionally, we compare
againstKG2RAG[Zhuet al., 2025] as a representative
of semantic-to-graph expansion paradigms.
For methods with explicit text chunk retrieval, we fix the
text chunk top-ktok= 5.
Evaluation Metrics
To rigorously assess both retrieval completeness and genera-
tion accuracy, we employ a multi-dimensional metric system.
Retrieval Scope with Provenance Mapping.Unlike stan-
dard RAG which only retrieves text chunks, TGS-RAG re-
trieves both structured paths and unstructured chunks. To en-
sure a fair comparison with document-level ground truth, we
define the set of retrieved documentsD retas the union of
explicit document retrieval and implicit graph provenance:
Dret=D chunks ∪ {doc| ∃e∈P graph, doc∈Source(e)}
(4)
whereD chunks denotes the source documents of the top-k
retrieved text chunks, andSource(e)represents the source
documents of entities contained in the retrieved graph paths
Pgraph . Based onD ret, we calculateStrict Hit Rate (SHR),
the percentage of queries whereD retcontainsallground
truth supporting documents, andSupport F1, measuring
the quality of evidence. Because graph-based retrieval ex-
pands evidence through entity-to-document provenance, its
retrieved document set is typically broader than chunk-only
baselines, making Support F1 a conservative metric that may
be lower despite stronger structural coverage.
Generation Quality (LLM-as-a-Judge).Traditional
string-matching metrics (e.g., Exact Match) often penal-
ize correct but verbose answers. Therefore, we adopt
LLM Judge Accuracy, employing DeepSeek-V3.2 to
judge whether the generated response contains the correct
information specified in the gold answer.4.2 Main Results
Performance Comparison
Table 1 presents a comprehensive evaluation of retrieval
and generation performance across MuSiQue and HotpotQA
benchmarks.
Limitations of Shallow Hybridization and Unidirectional
Expansion.The results reveal critical bottlenecks in exist-
ing retrieval paradigms. As observed in the MuSiQue dataset,
Hybrid RAGyields a Strict Hit Rate (14.23%) and Judge Ac-
curacy (21.56%) virtually identical to Naive RAG. This stag-
nation confirms that the primary challenge in multi-hop rea-
soning is not lexical mismatch—which keyword search ad-
dresses—butstructural disconnection. Hybrid methods fail
to bridge the gap when documents share no lexical overlap
but are logically linked. Similarly,KG2RAGunderperforms
significantly (9.10% Hit Rate on MuSiQue), demonstrating
the fragility of unidirectional ”semantic-to-graph” expansion;
if the initial semantic retrieval misses key anchor points, the
subsequent graph reasoning collapses due to the lack of a
feedback loop.
The Precision-Recall Trade-off in Graph Baselines.
While heavy graph-based approaches likeGraphRAGand
LightRAGachieve competitive recall, they suffer from se-
vere precision degradation. On HotpotQA, both methods ex-
hibit extremely low retrieval precision (≈10%), compared to
27.41% for TGS-RAG. This indicates aninformation over-
loadphenomenon: these systems retrieve excessive, logically
loosely related graph components (e.g., entire communities),
which introduces significant noise into the context window.
Consequently, despite high recall, their generation accuracy
(LLM Judge) is compromised by the distraction of irrelevant
evidence.
Superiority of TGS-RAG. TGS-RAGestablishes a new
state-of-the-art by effectively balancing retrieval scope and
precision. On MuSiQue, it achieves a Strict Hit Rate of
34.84%, more than doubling the performance of Hybrid
RAG. On HotpotQA, it attains the highest Strict Hit Rate,

Variant SHR (%) Recall (%) Prec (%) F1 (%)
TGS-RAG (Full) 62.00 77.55 27.41 26.06
w/o Bridging 47.82 68.16 22.74 23.46
w/o Re-ranking 52.65 70.31 23.88 24.08
Table 3: Ablation study on HotpotQA. Detailed impact of removing
synergy modules.
Recall, and Judge Accuracy, reaching a Judge Accuracy of
79.99%despite a conservative Support F1 of26.06%under
broader graph provenance. Unlike baselines that trade preci-
sion for recall, TGS-RAG utilizes the bidirectional synergy
to filter out pseudo-evidence while resurrecting subtle logical
links, providing the LLM with a concise and rigorous reason-
ing chain.
4.3 Efficiency Analysis
Table 2 details the computational costs associated with each
paradigm. A comparative analysis highlights the superior
cost-effectiveness of our framework.
Prohibitive Costs of Global Indexing.Graph-based global
indexing methods impose a prohibitive computational bur-
den.LightRAGandGraphRAGincur astronomical token
usage (e.g., over 757M tokens for LightRAG on HotpotQA
vs. 62M for Naive RAG), primarily driven by the exhaustive
construction of community summaries and dual-level indices.
This massive overhead renders them impractical for dynamic
or large-scale knowledge bases where frequent updates are
required.
Optimal Pareto Frontier of TGS-RAG.TGS-RAG
achieves an optimal trade-off between performance and ef-
ficiency. While it incurs a moderate cost increase over Naive
RAG (≈3x) to support graph reasoning, it is drastically more
efficient than global graph methods—consuming only37%
of the tokens used by GraphRAG on MuSiQue and less
than30%of LightRAG on HotpotQA. This efficiency stems
from our algorithmic design: instead of pre-computing ex-
pensive global summaries, TGS-RAG employs anon-demand
retrieval strategy. TheSemantic Beam SearchandMemory-
based Orphan Entity Bridgingselectively explore and resur-
rect only contextually relevant paths during inference, avoid-
ing the redundant processing of the entire graph structure.
4.4 Ablation Study
To verify the contribution of each module in our bidirectional
synergy mechanism, we conducted an ablation study on the
HotpotQAdataset. We developed two variants:
•w/o Graph-to-Text (Re-ranking):Disables theGlobal
Votingmechanism. Text chunks are ranked solely by
vector similarity.
•w/o Text-to-Graph (Bridging):Disables theMemory-
based Orphan Entity Bridging. The system relies solely
on the initial graph paths found by beam search.
As shown in Table 3, removing the Graph-Guided Re-
ranking module leads to a decline in Strict Hit Rate to
52.65%and Support F1 to24.08%. This indicates that with-
out structured filtering, the retriever struggles to distinguishbetween semantically similar distractors and true supporting
facts, thereby reducing the precision and overall quality of
the evidence context. A concrete example of this ”Semantic
Drift” and how our synergy mechanism resolves it is analyzed
in Appendix D.
Similarly, removing the Bridging module results in the
most significant performance degradation, with Strict Hit
Rate dropping sharply from62.00%to47.82%. This con-
firms that standard graph traversal may fail when potentially
valid evidence nodes are pruned early, whereas our memory-
based resurrection mechanism effectively recovers these ”or-
phan” connections.
5 Conclusion
We have proposedTGS-RAG, a bidirectional text–graph syn-
ergistic framework designed to resolve the isolation between
unstructured textual evidence and structured graph-based rea-
soning. By modeling retrieval as a collaborative process,
TGS-RAG effectively bridges the ”Information Island” gap
inherent in multi-hop reasoning tasks.
The framework introduces two complementary enhance-
ment channels:Graph-Guided Re-rankingleverages struc-
tured logic to filter semantic noise from text retrieval, while
Memory-based Orphan Entity Bridgingutilizes textual
context to validate and complete graph reasoning paths. A
critical innovation is the use of ”visited memory” toresur-
rectpruned paths, which recovers potentially useful logical
links that were previously discarded during search, without
the computational overhead of iterative expansion.
Extensive experiments on MuSiQue and HotpotQA
demonstrate that TGS-RAG consistently outperforms strong
text-based, graph-based, and hybrid baselines. Our results
indicate that shallow hybridization is insufficient; instead, ef-
fective multi-hop reasoning requires structured and unstruc-
tured evidence to mutually verify and correct each other. No-
tably, TGS-RAG achieves this state-of-the-art performance
with significantly lower token consumption compared to
global graph retrieval methods. Overall, this work under-
scores the importance of bidirectional synergy and provides
a scalable foundation for integrating symbolic structure with
neural representations in LLMs.

A Prompt Templates
To ensure reproducibility, we provide the core prompt tem-
plates used inTGS-RAG. All prompts are engineered to be
concise and task-specific for the underlying LLM (GPT-4o-
mini).
A.1 Entity & Relation Extraction Prompt
This prompt guides the LLM to extract structured knowledge
triplets from unstructured text chunks during the Knowledge
Base Construction phase.
System Prompt: Knowledge Graph Specialist
—Role—
You are a Knowledge Graph Specialist responsible for extracting entities and rela-
tionships from the input text.
—Instructions—
1.Entity Extraction:Identify clearly defined and meaningful entities.
•Fields:
–name: The name of the entity (Title Case).
–type: Categorize the entity using the providedEntity types.
If none apply, classify asother.
–description: A concise description basedsolelyon the text.
•Target Format (JSON Lines):
{"type": "entity", "name": "...", "category":
"...", "desc": "..."}
2.Relationship Extraction:Identify direct binary relationships between previ-
ously extracted entities.
•Rule:Decompose complex N-ary relationships into binary pairs.
•Fields:
–source/target: The exact names of the source and target enti-
ties.
–keywords: Comma-separated high-level keywords summarizing
the relation.
–description: A concise explanation of the connection.
•Target Format (JSON Lines):
{"type": "relation", "source": "...",
"target": "...", "keywords": ["..."], "desc":
"..."}
3.General Rules:
• Output all entities first, followed by all relationships.
• Use third-person perspective. Avoid pronouns like ’I’, ’you’, ’this arti-
cle’.
• The entire output must be in{language}. Proper nouns should be
retained in their original language.
•End Signal:Output the literal string<|COMPLETE|>on the final line.
—Real Data to be Processed—
¡Input¿
Entity types:[{entity types}]
Text:
{input text}
A.2 Query Entity Extraction Prompt
This prompt is used in the online retrieval stage. To handle
complex user intents and avoid keyword noise, we employ
detailed exclusion rules and few-shot demonstrations to guide
the LLM in identifying the core search targets.
User Prompt: Query Analysis
—Role—
You are a highly intelligent query analysis engine for a Retrieval-Augmented Gen-
eration (RAG) system.
—Goal—
Accurately identify and extract key concepts or entities that serve as thecore sub-
jectsof the user’s query.
—Definition of Core Entity—
A core entity is a noun or proper noun with clear referential meaning. It usually
belongs to:
•Specific Objects:Person names, locations, organizations, models (e.g.,
”iPhone 15”).
•Abstract Concepts:Technical terms, theories, strategies (e.g., ”RoHS di-rective”).
•Broad Themes:The central topic of vague queries (e.g., extract ”market-
ing” from ”tell me about marketing”).
—Exclusion Criteria (Do NOT Extract)—
1.User Intent Verbs:Words indicating what the user wants to do (e.g., ”com-
pare”, ”find”, ”list”, ”describe”).
2.Functional Words:Stop words or general nouns acting as sentence structures
(e.g., ”information”, ”detail”, ”introduction”).
—Few-Shot Demonstrations—
1.User:”Help me introduce a fighter.”
Output:["fighter"](Ignore ”introduce”)
2.User:”What is the relationship between RoHS and peak forward current?”
Output:["RoHS", "peak forward current"]
3.User:”Hello, how are you?”
Output:[](No core entities)
—Task—
User Query:"[User Query String]"
Extracted Entities (JSON Format):
A.3 Final Answer Generation Prompt
This prompt synthesizes the retrieved graph paths and text
chunks. It includes strict instructions to prioritize evidence
over prior knowledge and explicitly handle missing informa-
tion to prevent hallucinations.
User Prompt: Context Consolidation
—Role—
You are a knowledgeable and logically rigorous AI knowledge assistant.
—Core Instructions—
1.Strict Adherence to Context:Your answer must becompletelyandsolely
based on the provided evidence. You arestrictly prohibitedfrom using inter-
nal prior knowledge.
2.Synthesis & Reasoning:
• UseKnowledge Graph Pathsto establish the logical backbone (rela-
tionships between A and B).
• UseTextual Evidenceto fill in specific details (dates, descriptions, at-
tributes).
• If a logical link is provided in the graph but missing in the text (or vice
versa), synthesize them to form a complete chain.
3.Anti-Hallucination:If the provided context does not contain sufficient infor-
mation to answer the question, explicitly state:”Based on the provided mate-
rials, I cannot answer this question. ”
4.Structure:Organize the response clearly using headings, bullet points, or
Markdown tables for comparisons.
—Context Provided—
## Knowledge Graph Paths
[Path 1: Entity A --(relation)--> Entity B
--(relation)--> Entity C]
[Path 2: ...]
## Textual Evidence
[Evidence 1 (Source: Doc A): "...specific content
text..."]
[Evidence 2: ...]
—Task—
User’s Original Question:"[User Query String]"
Your Answer:
A.4 LLM-as-a-Judge Evaluation Prompt
To evaluate generation quality, we adopt an LLM-as-a-Judge
paradigm to assess whether a generated answer contains the
correct information specified by the gold answer.
System Prompt: Answer Correctness Judge
You are an impartial and strict judge evaluating the correctness of a generated
answer compared to a gold standard answer. Your task is to determine if the ”Gen-
erated Answer” contains the correct information specified in the ”Gold Answer”.
Rules:
1.Answer-Centric Evaluation:Judge solely based on whether the core answer
required by the Gold Answer is correctly provided. Additional correct or irrel-
evant information should not affect the judgement unless it introduces contra-
dictions.
2.Semantic Equivalence:If the generated answer is verbose but clearly contains

the correct core entity, fact, or conclusion, mark it asCORRECT.
3.Aliases:Treat any provided aliases of the Gold Answer as equally valid correct
answers.
4.Hallucination or Contradiction:If the generated answer contains incorrect
facts or conflicts with the Gold Answer, mark it asINCORRECT.
5.Non-Answering:If the generated answer states uncertainty or fails to provide
the required answer, mark it asINCORRECT.
Output Format (JSON):
{
"is_correct": boolean,
"reason": "Short explanation."
}
B Algorithm Pseudocode
We present the pseudocode for the core algorithms ofTGS-
RAG: Semantic Beam Search (which constructs the Visited
Memory) and Memory-based Orphan Entity Bridging.
Algorithm 1Semantic Beam Search with Visited Memory
Require:Query embeddingq, Seed EntitiesE seed, Knowl-
edge GraphG, Beam WidthK, Max DepthD
Ensure:Top PathsP initial , Visited MemoryM visited
1:InitializeP curr← {[e]|e∈E seed}
2:InitializeM visited ← ∅
3:foreache∈E seeddo
4:M visited [e]← {path: [e], score: 1.0} {Initialize
memory with seeds}
5:end for
6:ford= 1toDdo
7:Candidates← ∅
8:foreach pathp∈P curr do
9:e curr←plast{Last node in path}
10:N ← G.GetNeighbors(e curr){Limit by edge
weight if necessary}
11:foreach neighborn∈ Ndo
12:ifn /∈pthen{Cycle prevention}
13:n←GetEmbedding(n)
14:sim←CosineSim(n,q)
15:p new←p+[n]{Key Step:Track all explored
nodes and their paths in Memory}
16:ifn /∈ M visited orsim >M visited [n].score
then
17:M visited [n]← {path:p new, score:sim}
18:end if
19:Candidates←Candidates∪{(p new, sim)}
20:end if
21:end for
22:end for
23:P curr ←SelectTopK(Candidates, K){Pruning
step: keep only top-K paths}
24:end for
25:returnP curr,MvisitedAlgorithm 2Memory-based Orphan Entity Bridging
Require:Retrieved Text ChunksC text, Visited Memory
Mvisited , Initial Graph PathsP initial
Ensure:Final Path SetP final
1:E text←ExtractEntityIds(C text)
2:E graph←S
p∈P initial{e|e∈p}
3:E orphan ←E text\Egraph{Identify entities present in
text but missing from graph paths}
4:P bridge← ∅
5:foreach orphan entitye∈E orphan do
6:ife∈ M visited then
7:p recovered ← M visited [e].path{Zero-overhead
bridging: Resurrect path from memory}
8:P bridge←P bridge∪ {p recovered }
9:end if
10:end for
11:P final←P initial∪Pbridge
12:returnP final
C Implementation Details
This section provides implementation details of TGS-RAG
to ensure reproducibility, including hyperparameter settings,
model choices, and system configuration. All parameters re-
ported here correspond to the best-performing configuration
identified through the sensitivity analysis described in Ap-
pendix E.
C.1 Hyperparameters
Table 4 summarizes the hyperparameters used in all experi-
ments unless otherwise specified. These parameters govern
the bidirectional synergy between text and graph retrieval, in-
cluding semantic–structural balancing, graph search behav-
ior, and memory-based orphan entity bridging.
Table 4: Hyperparameter Settings for TGS-RAG
Parameter Symbol Value
Chunk Synergy Weightα0.5
Text Confirmation Bonusϵ0.4
Beam Search WidthK20
Search Depthd3
Max Neighbors per NodeN max 30
Top Orphan Entities Bridgedk o 3
Seed Density Bonusγ0.4
Entity Degree Weightλ e 0.01
Relation Degree Weightλ r 0.01
Top Recommended Chunksk r 4
Embedding DimensionDim1024
The chunk synergy weightαcontrols the balance between
semantic similarity and graph-based recommendation during
text re-ranking, while the text confirmation bonusϵdeter-
mines the strength of textual evidence in validating graph
paths. The remaining parameters primarily affect search effi-
ciency and structural bias and are shown to have limited sen-
sitivity within reasonable ranges.

C.2 Environment and Models
•Large Language Model (LLM):GPT-4o-mini is used
for all LLM-invoked components, including entity ex-
traction, knowledge graph construction, and final answer
generation.
•Embedding Model:We employ theQwen3-
Embedding-0.6Bmodel with an output dimension
of 1024 for encoding queries, text chunks, and graph
entities.
•Database:All text chunks, graph entities, relations,
and their embeddings are stored in PostgreSQL with the
pgvectorextension, enabling unified vector-based re-
trieval over both structured and unstructured data.
•Execution Environment:Experiments are conducted
on a standard Linux server with 32GB RAM. No GPU
acceleration is required, as all model inference is per-
formed via API-based services.
C.3 Reproducibility Notes
To ensure reproducibility and stable evaluation:
• All entity extraction results from user queries are cached
to eliminate stochastic variation from repeated LLM
calls.
• Hyperparameter tuning is conducted on a dynamically
balanced validation subset of MuSiQue, with deter-
ministic sampling to guarantee consistent comparisons
across configurations.
• All reported results use the same fixed hyperparame-
ter configuration identified as optimal in the sensitivity
analysis.
D Case Study: Overcoming Semantic Drift
To intuitively demonstrate the effectiveness of our bidirec-
tional synergy, we analyze a representative multi-hop query
from the MuSiQue dataset where baseline methods failed due
to semantic ambiguity.
As shown in Figure 4, the baseline method falls into the
”keyword trap,” retrieving semantically similar but logically
irrelevant entities (various people named ”Jones”). In con-
trast, TGS-RAG successfully bridges the logical gap. The
Semantic Beam Searchdiscovered that ”Abigail Breslin” is
the structural intersection of the two movies. Crucially, this
structural signal allowed the system to ”resurrect” the cor-
rect textual evidence that was otherwise buried by the vec-
tor retriever, validating the necessity of our graph-guided re-
ranking mechanism.
E Hyperparameter Sensitivity Analysis
We analyze the sensitivity of TGS-RAG to key hyperparam-
eters that govern the bidirectional synergy between text and
graph retrieval. All experiments are conducted on a dynam-
ically balanced validation subset of MuSiQue, and perfor-
mance is measured using Strict Hit Rate.Case Study: Semantic Drift vs. Structural Bridging
Query:”Which actress had roles in the movies ’Janie Jones’
and ’Signs’?”
Baseline (w/o Synergy):
•Retrieved Text:Chunks related to”Caro Jones”,”Shirley
Jones”, and”Madison Jones”.
•Failure Cause:The vector retriever was misled by the high
semantic density of the token ”Jones,” resulting inSemantic
Drift. It retrieved entities sharing the surname but lacking
logical connection to the movie ”Signs”.
•Outcome:”Cannot Answer” (Precision: 0.00).
TGS-RAG (Ours):
•Graph Path Discovery:Janie Jonescast− − − →Abigail
Breslincast← − − −Signs
•Mechanism:Although the text chunk for ”Abigail Bres-
lin” initially had a lower semantic score than ”Caro Jones”,
theGraph Channelidentified her as acommon neighbor
(bridge node) connecting both query entities.
•Synergy:TheGraph-Guided Re-rankingmodule boosted the
priority of the ”Abigail Breslin” chunk, suppressing the irrel-
evant ”Jones” family chunks.
•Outcome:Correct Answer: ”Abigail Breslin” (Recall: 1.00).
Figure 4:Qualitative comparison on a multi-hop query.While
the baseline suffers from semantic drift (focusing on the surname
”Jones”), TGS-RAG leverages the graph structure to identify the
true bridge entity (”Abigail Breslin”) that logically connects the two
distinct movies, effectively correcting the retrieval focus.
Effect of Chunk Synergy Weightα.Varying the chunk
synergy weightαexhibits a clear unimodal trend. Perfor-
mance improves asαincreases from 0.4 to 0.5 and degrades
for larger values. This behavior reflects the trade-off be-
tween semantic similarity and graph-based structural guid-
ance: smallerαunderweights structural signals, while larger
αoveremphasizes graph recommendations at the expense of
semantic relevance. The variance across runs is consistently
small, indicating stable behavior around the optimal region.
Effect of Text Confirmation Bonusϵ.The text confirma-
tion bonusϵshows a threshold-and-saturation pattern. Perfor-
mance remains largely unchanged for small values ofϵand
peaks aroundϵ= 0.4, after which further increases yield no
additional benefit. This confirms that textual evidence is most
effective when used for validating graph paths rather than ag-
gressively promoting them.
Beam Width and Search Depth.TGS-RAG demonstrates
robustness to moderate variations in beam widthKand
search depthd. Increasing the search depth fromd= 3to
d= 4yields no observable improvement in retrieval per-
formance, suggesting diminishing returns from deeper explo-
ration. This robustness indicates that the proposed memory-
based orphan entity bridging mechanism effectively recovers

potentially valid reasoning chains discarded during search-
time pruning.
Pairwise Sensitivity ofαandϵ.We further analyze the
joint effect ofαandϵto assess parameter interaction stabil-
ity. Results show that the optimal configuration (α= 0.5,
ϵ= 0.4) is not an isolated point: neighboring combinations
yield comparable performance with smooth degradation away
from the optimum. This indicates that TGS-RAG does not
rely on fragile parameter coupling and that its performance
is stable across a contiguous region of the hyperparameter
space.
Overall, the selected configuration (α= 0.5,ϵ= 0.4,
K= 20,d= 3) achieves a favorable balance between ef-
fectiveness and efficiency and is used throughout all experi-
ments.
References
Gordon V . Cormack, Charles L A Clarke, and Stefan
Buettcher. Reciprocal rank fusion outperforms condorcet
and individual rank learning methods. InProceedings
of the 32nd International ACM SIGIR Conference on Re-
search and Development in Information Retrieval, SIGIR
’09, page 758–759, New York, NY , USA, 2009. Associa-
tion for Computing Machinery.
Manuel Cossio. A comprehensive taxonomy of hallucinations
in large language models.http://arxiv.org/abs/2508.01781,
2025.
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley,
Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropoli-
tansky, Robert Osazuwa Ness, and Jonathan Larson. From
local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130, 2024.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao
Huang. LightRAG: Simple and fast retrieval-augmented
generation. In Christos Christodoulopoulos, Tanmoy
Chakraborty, Carolyn Rose, and Violet Peng, editors,
Findings of the Association for Computational Linguis-
tics: EMNLP 2025, pages 10746–10761, Suzhou, China,
November 2025. Association for Computational Linguis-
tics.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V . Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan
Hooi. G-retriever: Retrieval-augmented generation for
textual graph understanding and question answering. In
A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet,
J. Tomczak, and C. Zhang, editors,Advances in Neural In-
formation Processing Systems, volume 37, pages 132876–
132907. Curran Associates, Inc., 2024.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su,
Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and
Pascale Fung. Survey of hallucination in natural language
generation.ACM Comput. Surv., 55(12), March 2023.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-
tau Yih. Dense passage retrieval for open-domain question
answering. In Bonnie Webber, Trevor Cohn, Yulan He,and Yang Liu, editors,Proceedings of the 2020 Confer-
ence on Empirical Methods in Natural Language Process-
ing (EMNLP), pages 6769–6781, Online, November 2020.
Association for Computational Linguistics.
Omar Khattab and Matei Zaharia. Colbert: Efficient and ef-
fective passage search via contextualized late interaction
over bert. InProceedings of the 43rd International ACM
SIGIR conference on research and development in Infor-
mation Retrieval, pages 39–48, 2020.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni,
Vladimir Karpukhin, Naman Goyal, Heinrich K ¨uttler,
Mike Lewis, Wen-tau Yih, Tim Rockt ¨aschel, Sebastian
Riedel, and Douwe Kiela. Retrieval-augmented genera-
tion for knowledge-intensive nlp tasks. In H. Larochelle,
M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors,
Advances in Neural Information Processing Systems, vol-
ume 33, pages 9459–9474. Curran Associates, Inc., 2020.
Dawei Li, Bohan Jiang, Liangjie Huang, Alimohammad
Beigi, Chengshuai Zhao, Zhen Tan, Amrita Bhattachar-
jee, Yuxuan Jiang, Canyu Chen, Tianhao Wu, et al. From
generation to judgment: Opportunities and challenges of
llm-as-a-judge. InProceedings of the 2025 Conference on
Empirical Methods in Natural Language Processing, pages
2757–2791, 2025.
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu,
Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu
Zhang, Chong Ruan, et al. Deepseek-v3 technical report.
arXiv preprint arXiv:2412.19437, 2024.
Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren
Qu, Cehao Yang, Jiaxin Mao, and Jian Guo. Think-on-
graph 2.0: Deep and faithful large language model rea-
soning with knowledge-guided retrieval augmented gener-
ation. In Y . Yue, A. Garg, N. Peng, F. Sha, and R. Yu, ed-
itors,International Conference on Representation Learn-
ing, volume 2025, pages 52782–52806, 2025.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo
Wang, Chen Lin, Yeyun Gong, Lionel Ni, Heung-Yeung
Shum, and Jian Guo. Think-on-graph: Deep and respon-
sible reasoning of large language model on knowledge
graph. InThe Twelfth International Conference on Learn-
ing Representations, 2024.
Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and
Ashish Sabharwal. MuSiQue: Multihop questions via
single-hop question composition.Transactions of the As-
sociation for Computational Linguistics, 2022.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christopher D.
Manning. HotpotQA: A dataset for diverse, explainable
multi-hop question answering. In Ellen Riloff, David Chi-
ang, Julia Hockenmaier, and Jun’ichi Tsujii, editors,Pro-
ceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing, pages 2369–2380, Brussels,
Belgium, October-November 2018. Association for Com-
putational Linguistics.
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang,
Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Day-

iheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.
Qwen3 embedding: Advancing text embedding and
reranking through foundation models, 2025.
Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei
Hu. Knowledge graph-guided retrieval augmented genera-
tion. InNAACL, 2025.