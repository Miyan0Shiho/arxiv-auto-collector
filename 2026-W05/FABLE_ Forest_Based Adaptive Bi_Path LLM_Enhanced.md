# FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval for Multi-Document Reasoning

**Authors**: Lin Sun, Linglin Zhang, Jingang Huang, Change Jia, Zhengwei Cheng, Xiangzheng Zhang

**Published**: 2026-01-26 04:00:56

**PDF URL**: [https://arxiv.org/pdf/2601.18116v1](https://arxiv.org/pdf/2601.18116v1)

## Abstract
The rapid expansion of long-context Large Language Models (LLMs) has reignited debate on whether Retrieval-Augmented Generation (RAG) remains necessary. However, empirical evidence reveals persistent limitations of long-context inference, including the lost-in-the-middle phenomenon, high computational cost, and poor scalability for multi-document reasoning. Conversely, traditional RAG systems, while efficient, are constrained by flat chunk-level retrieval that introduces semantic noise and fails to support structured cross-document synthesis.
  We present \textbf{FABLE}, a \textbf{F}orest-based \textbf{A}daptive \textbf{B}i-path \textbf{L}LM-\textbf{E}nhanced retrieval framework that integrates LLMs into both knowledge organization and retrieval. FABLE constructs LLM-enhanced hierarchical forest indexes with multi-granularity semantic structures, then employs a bi-path strategy combining LLM-guided hierarchical traversal with structure-aware propagation for fine-grained evidence acquisition, with explicit budget control for adaptive efficiency trade-offs.
  Extensive experiments demonstrate that FABLE consistently outperforms SOTA RAG methods and achieves comparable accuracy to full-context LLM inference with up to 94\% token reduction, showing that long-context LLMs amplify rather than fully replace the need for structured retrieval.

## Full Text


<!-- PDF content starts -->

FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval
for Multi-Document Reasoning
Lin Sun∗
Qiyuan Tech
Beijing, ChinaLinglin Zhang
Qiyuan Tech
Beijing, ChinaJingang Huang
Qiyuan Tech
Beijing, China
Change Jia
Qiyuan Tech
Beijing, ChinaZhengwei Cheng
Qiyuan Tech
Beijing, ChinaXiangzheng Zhang
Qiyuan Tech
Beijing, China
Abstract
The rapid expansion of long-context Large Language Models (LLMs)
has reignited debate on whether Retrieval-Augmented Generation
(RAG) remains necessary. However, empirical evidence reveals per-
sistent limitations of long-context inference, including the lost-
in-the-middle phenomenon, high computational cost, and poor
scalability for multi-document reasoning. Conversely, traditional
RAG systems, while efficient, are constrained by flat chunk-level
retrieval that introduces semantic noise and fails to support struc-
tured cross-document synthesis.
We presentFABLE, aForest-basedAdaptiveBi-pathLLM-Enhanced
retrieval framework that integrates LLMs into both knowledge orga-
nization and retrieval. FABLE constructs LLM-enhanced hierarchi-
cal forest indexes with multi-granularity semantic structures, then
employs a bi-path strategy combining LLM-guided hierarchical tra-
versal with structure-aware propagation for fine-grained evidence
acquisition, with explicit budget control for adaptive efficiency
trade-offs.
Extensive experiments demonstrate that FABLE consistently out-
performs SOTA RAG methods and achieves comparable accuracy to
full-context LLM inference with up to 94% token reduction, show-
ing that long-context LLMs amplify rather than fully replace the
need for structured retrieval.
1 Introduction
The rapid advancement of Large Language Models (LLMs) with ex-
tended context windows has fundamentally reshaped the landscape
of knowledge-intensive question answering. Models like GPT-4
Turbo with 128K tokens [ 24], Claude 3 with 200K tokens [ 1], and
Gemini 1.5 Pro with up to 1M tokens [ 28] have led to claims in
industry discussions and technical commentaries that Retrieval-
Augmented Generation (RAG) [ 19] may become obsolete: a senti-
ment echoed across industry blogs and technical media. The under-
lying assumption is straightforward: if an LLM can process entire
document collections within its context window, why bother with
the complexity of retrieval systems?
However, this narrative oversimplifies a multifaceted challenge.
Recent research has exposed critical limitations of long-context
LLMs: the "lost in-the-middle" phenomenon [ 21] where relevant
information buried in long contexts is often overlooked, with perfor-
mance varying significantly based on information placement [ 17],
quadratic computational complexity of attention mechanisms [ 36]
∗Corresponding author. Email:lincharliesun@gmail.commaking large-scale deployment prohibitively expensive, and insuffi-
cient capacity for repositories requiring synthesis across hundreds
of documents [32, 35].
Traditional RAG systems [ 9,19], while more scalable, suffer
from semantic noise, which retrieved passages may exhibit high
surface-level similarity to queries without containing actual an-
swers [ 27,31]. This disconnect between surface-level semantic
matching performed by embedding models [ 18,22] and deep an-
swer relevance becomes particularly evident in multi-document
benchmarks. Additionally, most RAG systems operate at individ-
ual document level, struggling with cross-document reasoning
tasks such as comparative analysis, trend synthesis, or contradic-
tion detection [ 13,37]. Recent structured approaches have sought
middle ground: graph-based methods like GraphRAG [ 7], Ligh-
tRAG [ 10], and HippoRAG [ 11] organize information around entity
relations and leverage techniques like personalized PageRank for
multi-hop reasoning, while hierarchical methods like RAPTOR [ 29],
TreeRAG [ 33], and HiRAG [ 15] construct tree structures over docu-
ment collections.
Yet these advances remain fundamentally constrained by a paradig-
matic divide.Structural RAG approacheswhether organizing in-
formation around entity-centric relations and community sum-
maries [ 7,10] or employing predefined hierarchical schemas [ 15,
29,33], constructstaticknowledge representations with passive
similarity-matching retrieval decoupled from LLM reasoning. Con-
versely,long-context optimization approachesremain fundamentally
reactive, providing LLMs with pre-assembled, largely flat informa-
tion pools rather than enablingactive, structured navigationthrough
hierarchical knowledge spaces. Neither paradigm endows LLMs
with the cognitive control humans naturally exercise: dynamically
deciding whether to "zoom in" for granular details or "zoom out"
for holistic synthesis based on query demands.
Our Approach. We argue that overcoming the limitations of
prior RAG systems requires fundamentally rethinking the relation-
ship between retrieval and reasoning. Rather than treating them
as separate stages(first retrieve, then reason), we proposeFABLE
(Forest-BasedAdaptiveBi-PathLLM-Enhanced Retrieval), a uni-
fied framework where LLMs actively participate in both organizing
knowledge structures and dynamically navigating them according
to query-specific cognitive demands. Unlike prior structured RAG
methods, FABLE treats retrieval as a query-conditioned navigation
problem over LLM-constructed semantic hierarchies, rather than
static similarity matching over pre-defined structures. Our key in-
sight is twofold: (1) LLMs should not merely consume retrieved
1arXiv:2601.18116v1  [cs.CL]  26 Jan 2026

Figure 1: Overview of FABLE, a forest-based adaptive bi-path LLM-enhanced retrieval framework
information but actively construct semantically meaningful, multi-
granularity knowledge hierarchies during indexing; (2) retrieval
should not follow one-size-fits-all logic but adaptively employ dif-
ferent traversal strategies based on query characteristics. FABLE
implements this vision through three core innovations:
•Hierarchical Knowledge Forests: LLMs dynamically or-
ganize information into multi-level semantic hierarchies,
enabling structured reasoning across both fine-grained de-
tails and high-level summaries. Importantly, indexing is a
one-time offline process, amortized across all queries, and
can be performed incrementally. In contrast, long-context
inference incurs per-query quadratic costs.
•Query-Conditioned Traversal: Retrieval is treated as an
active navigation process, allowing the model to decide when
to "zoom in" for granular content or "zoom out" for holistic
synthesis, directly operationalizing the cognitive control
humans exercise during reasoning.
•Bi-Path Retrieval: Unlike prior graph- or tree-based RAG
methods that rely on a single static traversal, FABLE simulta-
neously employs a semantic navigation path and a structural
aggregation path. This bi-path mechanism allows per-query
adaptive exploration, ensuring that retrieval both captures
deep semantic relevance and preserves document-level struc-
tural integrity a capability absent in existing structured or
long-context RAG approaches.
These innovations unify retrieval and reasoning in FABLE, over-
coming the static and decoupled limitations of prior methods, and
enabling query-adaptive, multi-granularity access for complex multi-
document tasks. We evaluate FABLE on synthetic and real-world
multi-hop QA and large-scale agent tasks, demonstrating significant
gains in effectiveness and efficiency.
•Superior reasoning with fewer hallucinations. FABLE
achieves93.65%completeness with low hallucination (5.37%)
and irrelevance (2.52%), gaining+7.0 EMon HotpotQA and
+8.0 EMon 2Wiki over structured RAG baselines.
•Dramatic token efficiency. FABLE matches full-context
LLM performance (517K tokens) using only31K tokens(
21.5K for LLM to select docs, 1.5K for LLM to select nodes,
and 8K as the input for generator) up to94%reduction, whileachieving 92.07% completeness vs. Gemini-2.5-Pro’s 91.05%
with full context.
•Complementary bi-path strengths. Bi-path fusion im-
proves completeness by+1.6 pointsat 4K tokens; node-level
navigation achieves81.6%at 1K tokens (+30 pointsover
flat retrieval).
•Scalability to large-scale collections. On BrowseComp-
plus (100K+docs), FABLE boosts agent accuracy to66.60%
(+22.14) and recall to76.60%(+14.28) while reducing search
calls to21.74, without changing the agent LLM.
Contributions. This work makes the following contributions:
•Unified hierarchical retrieval-reasoning framework:
FABLE integrates LLM-driven semantic structuring with
budget-adaptive bi-path navigation over multi-granularity
semantic forests, enabling dynamic balance between cover-
age breadth and detail depth.
•Bi-path mechanisms at multiple granularities: We in-
troduce complementary bi-path retrieval combining depth-
adaptive LLM selection with vector retrieval at document
level, and hierarchical navigation with TreeExpansion’s struc-
tural propagation at node level, capturing both symbolic
understanding and embedding similarity while exploiting
structural relationships.
•Budget-adaptive routing: FABLE dynamically adjusts re-
trieval granularity based on budget constraints with structure-
aware fusion and position-preserving ordering, avoiding un-
necessary fine-grained retrieval when coarse results suffice.
•Comprehensive evaluation: Results across diverse bench-
marks show FABLE consistently achieves a superior balance
between completeness and faithfulness, outperforming struc-
tured RAG and full-context LLMs across multiple real-world
QA and agent tasks.
2 Related Work
2.1 Long-Context Large Language Models
Recent years have seen rapid expansion of LLM context windows,
from 4K tokens in GPT-3 to 128K in GPT-4 Turbo and beyond. These
advances are enabled by architectural techniques such as efficient
2

attention variants and improved positional encodings (e.g., RoPE).
However, longer context does not necessarily imply better perfor-
mance on knowledge-intensive tasks. Liu et al. [ 21] systematically
demonstrate thelost-in-the-middlephenomenon, showing substan-
tial accuracy degradation when relevant information appears in
the middle of long contexts rather than near the boundaries. Subse-
quent analyses further confirm strong positional biases in practical
attention utilization.
Moreover, long-context inference incurs significant computa-
tional and economic costs due to the quadratic complexity of atten-
tion, making it expensive for large-scale or production use. As a
result, several studies argue that retrieval-based methods remain
more efficient and reliable for tasks requiring precise information
access or multi-document reasoning, motivating hybrid designs
that combine retrieval with selective long-context modeling.
2.2 Retrieval-Augmented Generation
Retrieval-Augmented Generation (RAG), introduced by Lewis et
al. [19], has become a standard paradigm for knowledge-intensive
NLP. A typical RAG pipeline consists of chunking and indexing
documents, retrieving relevant passages using dense or hybrid
retrievers, and conditioning an LLM on the retrieved context. Ad-
vances in dense retrieval, including bi-encoder models trained with
contrastive objectives, have substantially improved recall, while
cross-encoder rerankers further enhance precision at higher cost.
Despite these improvements, traditional chunk-based RAG ex-
hibits fundamental limitations. Retrieved passages may be semanti-
cally similar to the query yet fail to contain the required evidence,
especially in multi-document or research-oriented settings. Recent
benchmarks show that vanilla RAG performs poorly on complex
multi-hop queries [ 5]. These issues stem from flat chunking, limited
global context modeling, and the lack of structured mechanisms
for cross-document synthesis.
Various extensions have been proposed, including query rewrit-
ing and decomposition (e.g., HyDE [ 8]) and iterative retrieval with
self-reflection (Self-RAG [ 2]). While effective in some cases, these
methods largely remain within the passage-level retrieval paradigm.
2.3 Structured Representations for RAG
To overcome the limitations of flat retrieval, recent work explores
structured knowledge representations. Graph-based approaches or-
ganize information around entities and their relations. For example,
GraphRAG constructs entity-centric graphs with community-level
summarization to support global queries [ 7]. HippoRAG further
integrates knowledge graphs with Personalized PageRank for multi-
hop retrieval [ 11]. While effective for relational reasoning, such
methods remain inherently entity-centric and rely on graph diffu-
sion over predefined relations, often missing higher-level semantic
abstractions spanning full documents.
Hierarchical approaches instead organize information into multi-
level abstractions. RAPTOR recursively clusters and summarizes
text to form tree structures, enabling multi-granularity retrieval [ 29].
However, its bottom-up construction produces static hierarchies
that may not align with discourse-level semantics and enforces
fixed-level retrieval without query-conditioned traversal; cross-
document semantic connections also remain limited. We compareagainst HippoRAG2 as the SOTA structured RAG baseline, which
has been shown to outperform RAPTOR, GraphRAG, and LightRAG
in recent evaluations.
Overall, existing structured RAG methods primarily emphasize
either entity-centric graph diffusion or document-internal hierar-
chies, but rarely construct document-spanning semantic hierarchies
or support query-adaptive retrieval that jointly preserves structural
integrity and semantic precision across documents.
2.4 LLM-Augmented Retrieval and Reasoning
Beyond static pipelines, several works integrate LLM reasoning into
retrieval. Iterative methods such as IRCoT [ 35] and Self-Ask [ 26]
perform multi-step retrieval guided by chain-of-thought reasoning.
Agent-based systems following the ReAct paradigm [ 38] interleave
reasoning and tool use, with extensions like Toolformer [ 30] and
WebGPT [ 23] enabling autonomous retrieval invocation, treating
retrieval as an external action rather than an internalized structure.
In contrast to these approaches, our work integrates LLMs di-
rectly into knowledge organization during indexing, enabling re-
trieval to operate over LLM-constructed multi-granularity semantic
hierarchies rather than flat passages or purely entity-centric graphs.
3 Methodology
This section presents the core design and technical implementa-
tion ofFABLE(Forest-basedAdaptiveBi-pathLLM-Enhanced
Retrieval). FABLE achieves efficient multi-document reasoning
through a bi-path collaborative mechanism: (1)Context-driven
Hierarchical Traversalenables adaptive navigation in the global
semantic space; (2)Structure-Aware Propagation Retrievalpro-
vides fine-grained evidence supplementation. These two paths form
a complementary retrieval paradigm, achieving dynamic balance be-
tween coverage breadth and detail depth. All hierarchical construc-
tion in FABLE is performed offline and amortized across queries;
query-time computation only involves lightweight traversal.
3.1 LLM-Enhanced Hierarchical Forest
Given a document collection D={𝑑 1,𝑑2,...,𝑑𝑁}, FABLE con-
structs a semantic forest F={𝑇 1,𝑇2,...,𝑇𝑁}, where each tree 𝑇𝑖
represents a hierarchical, multi-granularity semantic abstraction
of document 𝑑𝑖. Each𝑇𝑖is independently built and serves as the
fundamental indexing unit for downstream retrieval and reasoning.
Semantic-Aware Document Chunking.Instead of fixed-length
or heuristic-based segmentation, we adopt LLM-guided semantic
chunking to preserve discourse coherence. For each document 𝑑𝑖,
we obtain a sequence of semantically self-contained chunks:
C𝑖=LLM segment(𝑑𝑖)={𝑐 1,𝑐2,...,𝑐𝑚𝑖}(1)
where each chunk 𝑐𝑗is defined as 𝑐𝑗=(chunk_id 𝑗,content𝑗). Chunk
boundaries are aligned with semantic units (e.g., paragraphs or top-
ical shifts), avoiding sentence fragmentation and semantic overlap.
Tree Structure Definition.For a document 𝑑𝑖, its semantic
structure is formalized as a rooted tree 𝑇𝑖=(𝑉𝑖,𝐸𝑖), subject to the
following structural constraints:
•Bounded Depth: depth(𝑇𝑖)≤𝐷 , where𝐷is a configurable
hyperparameter.
3

Figure 2: Semantic tree construction and multi-granularity vector indexing: (1) semantic-aware chunking, (2) LLM-based
hierarchical tree generation, (3) vector indexing of internal (toc+summary) and leaf (chunk) nodes for hierarchical retrieval.
•Typed Nodes: Each node 𝑣∈𝑉𝑖is associated with a semantic
level, such that𝑉 𝑖=𝑉root∪𝑉 section∪𝑉 subsection∪𝑉 leaf.
Node Types and Attributes.Each node is represented as a
structured record with type-specific attributes:
•Leaf Nodes( 𝑣∈𝑉 leaf): Correspond to original semantic
chunks. Each leaf node is defined as 𝑣=(node_id,chunk_id) ,
wherechunk_idindexes an element inC 𝑖.
•Internal Nodes( 𝑣∈𝑉𝑖\𝑉leaf): Represent higher-level se-
mantic abstractions. Each internal node is defined as 𝑣=
(node_id,title,summary,children) , where title is a concise
topic descriptor (e.g., a ToC-style heading), summary is a
summary of all descendant nodes, and children denotes an
ordered set of child nodes.
Tree Construction.Given a set of semantic chunks C𝑖, the se-
mantic tree𝑇𝑖is constructed by an LLM-based structuring module:
𝑇𝑖=LLM structure(C𝑖|𝑑𝑖)(2)
which jointly generates a hierarchical table of contents and node-
level summaries. The construction process combines bottom-up
semantic aggregation (from chunks to section-level summaries)
with top-down structural constraints, including a maximum tree
depth and predefined node types.
Progressive Construction for Long Documents.For docu-
ments exceeding the LLM context limit, we employ a batch-wise tree
construction strategy by partitioning the document into sequential
parts. Each part is independently processed to build a partial tree.
The final document tree is obtained by merging all partial trees.
This approach enables scalable long-document processing while
maintaining cross-part semantic coherence.
T(𝑘)
𝑖=TreeBuild(part𝑘), 𝑘=1,...,𝐾(3)T𝑖=TreeMerge T(1)
𝑖,...,T(𝐾)
𝑖(4)
Multi-Granularity Vector Indexing.For a non-leaf node 𝑣,
let toc_path(𝑣)be the path from the root to𝑣. Its embedding is:
e𝑣=Embed(toc_path(𝑣)⊕summary(𝑣) ) (5)
For a leaf node𝑐(an original semantic chunk),
e𝑐=Embed(content(𝑐)) (6)
3.2 Budget-Adaptive Bi-Path Retrieval
FABLE adopts a hierarchical retrieval architecture that dynamically
adapts to context budget constraints while ensuring comprehensive
recall through bi-path mechanisms at each granularity level (Fig-
ure 1). The complete retrieval process is formalized in Algorithm 1.
Document-Level Bi-Path Recall.The retrieval begins with
parallel document selection through two complementary paths.
The first path employsdepth-adaptive LLM-guided selection, a novel
approach that constrains LLM reasoning to non-leaf nodes within
depth threshold 𝐿. Unlike prior work that feeds entire documents or
flat chunk lists to LLMs, this depth-adaptive strategy exploits the hi-
erarchical structure of semantic trees: by analyzing only high-level
table-of-contents and summaries, the LLM performs structured
reasoning over document semantics with bounded context length,
avoiding token budget explosion while maintaining global docu-
ment understanding. Concurrently, the second path performsmulti-
granularity vector-based retrievalusing FAISS indexing over all node
embeddings, capturing semantically similar documents through
dense retrieval that complements LLM’s reasoning-based selec-
tion. The fusion of these paths produces a deduplicated candidate
setDfusion that balances symbolic understanding with embedding-
space similarity.
4

Algorithm 1Budget-Adaptive Bi-Path Retrieval
Input:Query𝑞, Semantic ForestF, Hierarchy Threshold𝐿, Budget𝐵 max
Output:Retrieved ContentC
1:// Doc-Level Retrieval
2://Path 1: Depth-Adaptive LLM-Guided Docs Selection:
3:Initialize candidate set:𝑉 𝐿={all non-leaf nodes with depth≤𝐿}
4:Context={(toc(𝑣),summary(𝑣))|𝑣∈𝑉 𝐿}
5:D llm=LLM select(𝑞,Context)
6://Path 2: Vector-Based Docs Selection:
7:D vector←TopKFAISS(𝑞,𝐾 doc,index=nodes_vector)
8:
9:// Doc-Level Fusion
10:D fusion←Deduplicate(D llm∪D vector)
11:
12:// Budget-Adaptive Routing
13:ifÍ
𝑑∈Dfusion|content(𝑑)|≤𝐵 maxthen
14:return{content(𝑑)|𝑑∈D fusion}
15:end if
16:
17:// Node-Level Retrieval
18:// Path 1: LLM-Guided Navigation
19:Context←Ð
𝑑∈Dfusion{(toc(𝑣),summary(𝑣))|𝑣∈NonLeaf(𝑑)}
20:N llm←LLM navigate(𝑞,Context)
21:// Path 2: Tree-Based Expansion with Budget-Adaptive
22:N treexp←TreeExpansion(𝑞,D fusion,𝐵max)
23:
24:// Node-Level Fusion
25:C ordered←NodeFusion(N llm,Ntreexp)
26:
27:// Budget Control
28:C final←BudgetControl(C ordered)
29:returnC final
Budget-Adaptive Routing.A key innovation of FABLE is its
dynamic granularity control: if the total content size of Dfusion
falls within budget constraint 𝐵max, the framework terminates at
document-level and directly returns full document contents. This
adaptive mechanism avoids unnecessary fine-grained retrieval over-
head when coarse-grained results suffice, significantly improving
efficiency for queries with limited relevant scope.
Node-Level Bi-Path Recall.When budget constraints neces-
sitate fine-grained retrieval, FABLE activates node-level bi-path
selection within the fused document set. The LLM-guided path
performs hierarchical navigation through hierarchical non-leaf
nodes, progressively narrowing down to relevant semantic nodes.
In parallel, we employ Structure-Aware PropagationTreeExpansion
over the semantic tree to supplement vector retrieval. Traditional
dense retrieval operates in flat embedding space, missing valuable
parent-child relationships encoded in our hierarchical structure. We
propagate relevance through tree edges by combining three signals:
(1) direct query-node similarity, (2) ancestor-inherited relevance
(capturing topic continuity), and (3) child-aggregated relevance
(capturing subtopic importance). For node𝑣, composite score is:
𝑆(𝑣)=1
3
𝑆sim(𝑣)+𝑆 inh(𝑣)+𝑆 child(𝑣)
(7)
where𝑆sim(𝑣)=cos( e𝑣,e𝑞)/depth(𝑣) applies depth decay to fa-
vor high-level abstractions, 𝑆inh(𝑣)=max 𝑢∈Anc(𝑣)𝑆sim(𝑢)inherits
the maximum ancestor score, and 𝑆child(𝑣)=avg𝑐∈Children(𝑣)𝑆(𝑐)Algorithm 2NodeFusion
Input:N llm,Ntreexp
Output:Ordered chunksC ordered
1:// Ancestor-descendant deduplication
2:N dedup←RemoveDescendants(N llm∪N treexp)
3:
4:// Document-level sorting
5:D llm←Docs(N dedup∩N llm)
6:D treexp←Docs(N dedup∩N treexp)\D llm
7:
8:// Intra-document sorting
9:N ordered←[]
10:for𝑑∈[D llm,Dtreexp]do
11:N 𝑑←sort_by_position_order_in_d(N dedup∩Tree(𝑑))
12:AppendN 𝑑toN ordered
13:end for
14:
15:// Extract chunks
16:C ordered←GetChunks(N ordered)
17:
18:returnC ordered
aggregates child scores. We assign uniform weights (1/3 each) to
the three relevance signals. No special tuning or optimization was
performed for these weights; despite their simplicity, this scheme
works robustly across diverse document structures, preserves in-
terpretability, and enables efficient local graph diffusion over the
semantic trees.
Node-Level Fusion Strategy.The final fusion operation (Algo-
rithm 2) integrates results through three steps: (1)Structure-aware
deduplicationremoves redundant ancestor-descendant pairs by re-
taining ancestor subtrees that encompass descendant content, (2)
Priority-based partitioningseparates LLM-selected nodes N∗
llmfrom
tree-expanded nodes N∗
treexp , and (3)Position-preserving ordering
sorts nodes within each document by their original positions, con-
catenating LLM results before TreeExpansion results. This strategy
ensures that explicitly identified relevant content precedes struc-
turally inferred context while maintaining document-native read-
ing order, providing generation models with optimally arranged
retrieved evidence.
TreeExpansion Details.TreeExpansion traverses each tree in
Dfusion , computing scores via Eq. 4 recursively (bottom-up for 𝑆child,
top-down for 𝑆inh). Nodes are ranked by 𝑆(𝑣) and greedily selected
until budget 𝐵maxis exhausted, with ancestor nodes taking priority
to avoid redundancy. Complexity is 𝑂(|𝑉𝑖|·𝐷) per tree where 𝐷is
depth.
Budget Control. Given the maximum input length 𝐵maxof the
generation model, dynamically adjust the retrieval results through
greedy algorithms or knapsack problem solvers:
Cfinal={𝑐 1,...,𝑐𝑘∗}𝑘∗=max(
𝑘𝑘∑︁
𝑖=1ℓ(𝑐𝑖)≤𝐵 max)
(8)
The bi-path design at both granularities ensures robust recall:
LLM-guided paths capture semantically precise matches through
structured reasoning over hierarchical abstractions, while multi-
granularity vector or tree-based paths provide coverage over diverse
5

Table 1: Results on synthetic and real-world multi-hop QA benchmarks. FABLE variants: "llm-docs/nodes" use LLM-guided
selection only, "docs/nodes" use full bi-path fusion. HippoRAG2 represents the current SOTA for structured RAG. Best results
in bold. TreeRAG results unavailable in original paper.
Synthetic Knowledge Real-World Knowledge
Method Dragonball/DragBalance Dragonball/DragSingleZh HotpotQA 2WikiMultiHopQA
Recall(%)↑EIR(%)↑Comp.(%)↑Hall.(%)↓Irr.(%)↓Recall(%)↑EIR(%)↑EM(%)↑F1(%)↑EM(%)↑F1(%)↑
Traditional RAG
BM25 66.14 5.58 67.85 16.97 15.18not reported25.00 36.40 15.00 15.98
BGE-M3 64.02 5.43 67.16 17.83 15.01not reported37.50 51.50 1.00 1.85
Structure-Enhanced RAG
TreeRAGnot reported57.31 18.38not reported
LongRefiner 24.72 1.53 41.14 33.77 25.09 15.79 1.30 31.50 43.18 8.00 11.27
HippoRAG2 39.16 5.77 62.19 26.72 11.00 28.08 5.48 41.00 58.87 44.50 59.67
LLM-in-context (Full-context Inference)
Qwen3-32B 100 0.02 65.73 25.13 9.02 100 0.01 0.00 5.56 0.00 7.80
Gemini-2.5-Flash 100 0.02 88.37 5.93 5.58 100 0.01 31.00 47.44 38.50 54.94
Gemini-2.5-Pro 100 0.02 91.05 5.5 3.45 100 0.01 42.21 55.97 52.02 63.68
Our Proposed Method
FABLE(llm-docs) 85.55 3.46 92.014.883.00 74.61 5.19 31.00 42.22 30.5 38.46
FABLE(llm-nodes) 77.3018.6084.78 10.98 4.22 66.0721.9029.00 38.34 30.50 38.43
FABLE(docs)85.800.7692.075.372.52 74.990.4 46.50 62.2252.50 63.87
FABLE(nodes) 84.55 1.76 89.39 5.98 4.61 72.97 3.3348.00 63.4350.00 60.26
similarity patterns through dense retrieval and structural propaga-
tion, collectively addressing the complementary strengths of neural
retrieval and structured navigation.
4 Experimental Setup
4.1 Evaluation Datasets
We evaluate FABLE on diverse datasets covering synthetic reason-
ing, multi-hop QA, and agent-based retrieval.
Synthetic Knowledge QA: DragonBall [ 40] contains LLM-
generated documents and queries following predefined schemas,
enabling controlled evaluation without real-world confounds.
Real-World Knowledge Multi-hop QA: We use two subsets
from LongBench [ 3]: HotpotQA [ 37] requiring reasoning across
Wikipedia documents and 2Wiki [ 14] emphasizing multi-step entity
reasoning focusing on logical reasoning over structured and un-
structured sources. We adapt these two datasets from a long-context
inference benchmark to a RAG setting by treating the candidate
documents as a retrieval corpus.
Agent-based Downstream Application: BrowseComp-plus [ 5]
evaluates the integration of retrieval modules within autonomous
research agents. In this setting, FABLE replaces the original retrieval
component used in DeepResearch-style agents, and performance
is measured by the agent’s ability to conduct multi-turn naviga-
tion and selective information acquisition over large document
collections.
4.2 Experimental Configuration
To ensure fair and reproducible comparisons, we strictly align our
experimental settings with those of the selected baselines and onlymodify components that are directly related to the proposed method.
We intentionally align backbone LLMs with dataset language and
official evaluation protocols, and relative comparisons are always
performed under identical LLM settings. Therefore, observed per-
formance differences can be attributed to retrieval architecture
rather than generator capacity.
Agent Baselines and Retrieval Strategy.For agent baselines,
we use Qwen3-Embed-8B [ 39] as in the original implementations,
with results from official leaderboards. We replace the original re-
trieval with our proposed mechanism using GPT-OSS-120B [ 25]
and Qwen3-Embed-8B, while keeping the agent policy model un-
changed.
Embedding and Reranking Models.For non-agent methods
with dense retrieval, we use BGE-M3 [ 4] for embedding and BGE-
Reranker-v2-M3 [20] for reranking when required.
Backbone Language Models.For DragonBall (which includes
Chinese subsets), we use DeepSeek-V3.2 [ 6] throughout. For Hot-
potQA and 2Wiki (English benchmarks), we use GPT-OSS-120B.
Document Chunking and Hyperparameters.Document chunk-
ing uses LLM-based segmentation: DeepSeek-V3.2 for DragonBall,
GPT-OSS-120B for HotpotQA, 2Wiki, and BrowseComp-plus. Other
hyperparameters follow baseline defaults unless specified. We set
hierarchy depth 𝐷=4, and budget 𝐵maxranging from 1K to 128K
tokens depending on experimental conditions. We will release code,
prompts, and constructed semantic forests upon acceptance.
Overall, this setup ensures that performance differences primar-
ily arise from the proposed retrieval strategy rather than changes
in embedding models, agent policies, or hyperparameter tuning.
6

Figure 3: Performance of retrieval and context construction strategies across varying input lengths. Long-context models use
fixed 517K windows, while retrieval methods are tested from 1K to 128K tokens. Chunk-based retrieval shows limited gains and
degrades at large contexts, whereas FABLE consistently improves completeness while reducing hallucination and irrelevance.
FABLE (docs/llm-docs) achieves the best performance at moderate budgets.
Table 2: Performance on BrowseComp-plus. Top five rows:
official leaderboard results with rankings. Bottom two
rows: our methods with FABLE retriever. TDR-30B-A3B and
Q3-E8B denote Tongyi-DeepResearch-30B-A3B and Qwen3-
Embed-8B. FABLE(ln): LLM-based node selection; FABLE:
bi-path selection. Best in bold; second-best underlined.
Method (LLM/Retriever) RankBrowseComp-plus
Acc(%)↑Recall(%)↑SearchCalls↓
GPT5/MixedbreadSearch 1st78.4148.85 44.67
GPT5/Q3-E8B 3rd71.69 78.98 21.74
o3/Q3-E8B 4rd65.90 73.24 23.97
GPT5/BM25 5rd57.59 61.70 23.23
TDR-30B-A3B/Q3-E8B 11th44.46 62.32 30.37
TDR-30B-A3B/FABLE(ln)– 64.92 68.07 46.99
TDR-30B-A3B/FABLE– 66.60 76.60 21.74
4.3 Evaluation Metrics
We use dataset-specific metrics for faithful assessment across tasks.
DragonBall.Following [ 40], we report retrieval metrics (Recall
andEIR) and end-to-end metrics (Completeness,Hallucination,
Irrelevance). Recall measures whether gold supporting sentences
are retrieved; EIR evaluates the proportion of relevant retrieved
information. End-to-end metrics assess answer completeness, un-
supported content, and irrelevant information, evaluated using
DeepSeek-V3.2.
Multi-hop QA Datasets.Following [ 3], we reportExact Match
(EM)andF1for answer correctness. EM measures exact answer
matching; F1 evaluates token-level overlap. We use official Hip-
poRAG2 [12] evaluation scripts.
BrowseComp-plus.Following [ 5], we reportAccuracy,Recall,
andSearch-Calls. Accuracy measures answer correctness, Recall
evaluates retrieval of supporting information, and Search-Calls
quantify search actions for efficiency assessment.5 Evaluation
5.1 Effectiveness on Multi-Document Reasoning
To comprehensively evaluate the effectiveness of our method, we
compare against a diverse set of baselines, including the tradi-
tional sparse retrieval approach BM25, the modern dense retriever
BGE-M3 [ 4], as well as several structured or hierarchical retrieval-
augmented generation methods, namely TreeRAG [ 33], LongRe-
finer [ 16], and HippoRAG2 [ 12] which is the current SOTA that has
been shown to outperform RAPTOR, GraphRAG, and LightRAG.
We additionally evaluate several strong large language models,
including Qwen3-32B [ 34], Gemini-2.5-Flash, and Gemini-2.5-Pro.
For these evaluations, all documents from the benchmark are con-
catenated and fed to the LLMs to leverage their long-context capabil-
ities. When the total content exceeds the model’s context window,
we process the documents in parallel batches and aggregate the
outputs to produce the final answer.
For TreeRAG, we report results on DragSingleZh as provided in
the original paper [ 33]. For all other methods, we conduct exper-
iments under consistent experimental settings as described. The
results are reported in Table 1.
Across synthetic knowledge benchmarks, traditional RAG base-
lines (BM25 and BGE-M3) exhibit reasonable recall but suffer from
low EIR and relatively high hallucination and irrelevance, indi-
cating limited ability to aggregate and reason over multiple re-
trieved documents. Structure-enhanced baselines partially allevi-
ate this issue: while TreeRAG improves recall and EIR on Drag-
onBall/DragSingleZh, its applicability is limited to specific settings,
and methods such as LongRefiner and HippoRAG2 still show a clear
trade-off between completeness and faithfulness, with increased
hallucination under complex multi-hop reasoning.
In contrast, LLM-in-context inference with full documents, de-
spite achieving perfect recall by construction, does not consistently
translate retrieval coverage into correct reasoning. Models such
7

as Qwen3-32B exhibit substantial hallucination and low complete-
ness, highlighting that long-context capacity alone is insufficient
for structured multi-document reasoning. Even strong proprietary
models (Gemini-2.5-Flash and Gemini-2.5-Pro), while significantly
reducing hallucination, still lag behind our method in terms of
overall reasoning faithfulness and controllability, particularly on
synthetic benchmarks with explicitly structured dependencies.
Our proposed FABLE framework consistently outperforms all
baselines across both synthetic and real-world datasets. On Drag-
onBall/DragBalance, FABLE (docs) achieves the highest complete-
ness (92.07%) while simultaneously minimizing hallucination (5.37%)
and irrelevance (2.52%), demonstrating a strong ability to preserve
global document semantics while suppressing spurious genera-
tion. Notably, FABLE variants with node-level representations yield
substantially higher EIR, indicating more effective evidence aggre-
gation and utilization during reasoning, even when recall is slightly
lower than document-level variants.
On real-world multi-hop QA benchmarks (HotpotQA and 2Wiki),
FABLE further demonstrates robust generalization. Both FABLE
(docs) and FABLE (nodes) achieve the strongest EM and F1 scores
among retrieval-augmented methods, substantially outperforming
traditional and structure-enhanced RAG baselines. These results
suggest that explicitly modeling hierarchical knowledge structures
enables more reliable cross-document reasoning in open-domain
settings, where evidence is noisy and distributed.
Overall, the results confirm that FABLE effectively bridges the
gap between retrieval coverage and reasoning faithfulness. By en-
forcing structured knowledge organization and controlled evidence
aggregation, our method delivers consistently higher completeness
with lower hallucination across diverse knowledge regimes, out-
performing both conventional RAG pipelines and full-context LLM
inference.
5.2 Impact of Retriever Choice on Agent
Table 2 presents the performance of different LLM and Retriever
combinations on the BrowseComp-plus benchmark. The top five
rows correspond to official leaderboard results, with their respec-
tive rankings indicated in the “Rank” column. These results show
that both the choice of LLM and the retriever significantly af-
fect agent performance: higher-ranked methods generally leverage
stronger LLM backbones or more effective retrievers. For exam-
ple, GPT5 with Mixedbread Search achieves the 1strank, while
TongyiDeepResearch-30B-A3B with Qwen3-Embed-8B ranks 11th,
demonstrating that even a strong LLM can be limited by the re-
triever quality.
The bottom two rows report our results using FABLE as the
retriever while keeping TongyiDeepResearch-30B-A3B as the LLM
backbone. Compared to the official leaderboard entries using Qwen3-
Embed-8B, replacing the retriever with FABLE substantially im-
proves accuracy and recall, moving the method closer to higher-
ranked leaderboard entries despite using the same LLM. This indi-
cates that FABLE effectively enhances retrieval quality, which in
turn improves downstream agent reasoning.
Overall, these findings highlight that agent performance is jointly
determined by the LLM and the retriever, and that improving the
retriever alone without changing the LLM, can lead to meaningfulperformance gains. FABLE demonstrates a strong alternative to
existing retrievers, particularly in scenarios where upgrading the
LLM is costly or impractical.
5.3 Ablation Study
We conduct ablation studies to validate each component’s contri-
bution in FABLE. All experiments use the DragonBall benchmark
with context budgets from 1k to 128k tokens and Qwen3-32B for
LLM generation.
5.3.1Impact of LLM-Enhanced Indexing.We compare two in-
dexing strategies: (1) fixlength-chunks: Fixed-length chunking (128
tokens); (2) llm-chunks: LLM-guided semantic chunking without
hierarchy.
Results. Figure 3 (left panel) shows llm-chunks substantially out-
performs fixlength-chunks across all context lengths. At 64k tokens,
llm-chunks achieves 66.6% completeness vs. 61.2% for fixlength-
chunks, approaching the 65.7% upper bound of qwen3-32B with full
document access. More critically, llm-chunks reaches near-optimal
performance at 4k tokens, while fixlength-chunks requires 128k
tokens for comparable results. At 8k tokens, llm-chunks achieves
98.2% of the full-document upper bound (64.5% vs. 65.7%), demon-
strating 64×efficiency gain. This validates that semantic-aware
chunking preserves discourse coherence, enabling smaller context
windows without sacrificing quality.
5.3.2Document-Level Bi-Path Ablation.We evaluate FABLE(docs)
against single-path document selection variants: (1) FABLE(llm-
docs): Depth-adaptive LLM-guided selection only; (2) FABLE(docs):
Bi-path fusion at document level (includes vector retrieval)
Results. Figure 3 shows FABLE(docs) with bi-path mechanism
consistently achieves higher completeness than single-path vari-
ants. Interestingly, FABLE(llm-docs) shows slightly higher com-
pleteness at 1k-2k tokens (56.0% vs. 52.1% at 1k), suggesting that
pure LLM reasoning is more effective for extremely limited budgets
where vector retrieval may introduce noise. However, at 4k+ tokens,
FABLE(docs) consistently outperforms the single-path variant (+1.6
points at 4k, +1.3 points at 8k).
Notably, FABLE(docs) at 4k tokens (92.5%) already surpasses
gemini-2.5-pro with full document access (91.1%), demonstrating
that bi-path hierarchical retrieval enables smaller context budgets
to exceed the performance of significantly larger models with unre-
stricted access. This validates that intelligent retrieval architecture
is more impactful than simply scaling context or model size.
The hallucination and irrelevance metrics show similar patterns:
FABLE(docs) achieves superior precision at longer contexts (5.37%
hallucination and 2.52% irrelevance at 8k vs. 5.5% and 3.45% for
gemini-2.5-pro), validating that vector-based retrieval supplements
LLM reasoning with embedding-space similarity signals that reduce
false positives.
5.3.3Node-Level Bi-Path Ablation.We compare three fine-
grained retrieval strategies: (1) FABLE(llm-nodes): LLM-guided
hierarchical navigation only; (2) FABLE(treexp-nodes): TreeExpan-
sion with structural propagation only; (3) FABLE(nodes): Bi-path
fusion at node level.
Results. Figure 3 reveals distinct strengths across context bud-
gets.
8

Short context (1k-2k): At 1k tokens, FABLE (llm-nodes) reaches
81.6% completeness, significantly surpassing llm-chunks (57.2%)
and fixlength-chunks (51.4%), with improvements of 24.4 and 30.2
points, respectively. This highlights the importance of LLM-guided
hierarchical navigation for resource-constrained scenarios, directly
reasoning over ToC to identify relevant sections. In contrast, vector-
based TreeExpansion may retrieve semantically similar but irrel-
evant passages under tight budgets. The bi-path FABLE(nodes)
(61.9%) lies between FABLE(llm-nodes) and FABLE(treexp) (52.5%),
indicating TreeExpansion introduces noise at extreme budget limits.
Convergence at longer contexts (4k-8k): The gap rapidly closes as
context increases. At 4k tokens, FABLE(nodes) achieves 89.4% com-
pleteness, significantly outperforming flat retrieval baselines (llm-
chunks: 62.3%, fixlength-chunks: 58.2%) by 27+ points. By 8k tokens,
FABLE(treexp) (88.9%) and FABLE(nodes) (89.1%) approach the per-
formance of gemini-2.5-pro with full document access (91.1%), while
using only 8k tokens compared to 517k.
Hallucination and irrelevance control: FABLE(nodes) consistently
shows the lowest hallucination rates at 4k+ tokens (6.0% at 4k,
7.0% at 8k), substantially better than fixlength-chunks and llm-
chunks. For irrelevance, FABLE(nodes) achieves 4.6% at 4k and 3.9%
at 8k, compared to 10.6% and 8.1% for the flat retrieval baselines,
validating that fine-grained hierarchical retrieval enables more
precise evidence localization.
5.3.4Bi-path Component Analysis.Our results validate the
core hypothesis that combining global and local information pro-
cessing yields complementary advantages. As shown in the task-
specific results (Figure 5), FABLE(llm-nodes) excels at tasks re-
quiring broad information synthesis (e.g., Summarization: 91.3%),
while TreeExpansion performs better on tasks demanding local
detail extraction (e.g., Temporal Sequence and Factual Question).
FABLE(Nodes) effectively integrates both strengths, achieving su-
perior performance across all task types (average 97.7%).
The cross-domain experiments as shown in Figure 4 further
confirm this bi-path advantage. In the Medical domain, where FA-
BLE and TreeExpansion show opposing performance patterns (En:
91.7% vs. 86.2%; Zh: 81.2% vs. 72.3%), FABLE(Nodes) successfully
reconciles both approaches (97.6% and 94.3%). This pattern holds
consistently across Finance and Law domains in both English and
Chinese, demonstrating that the architectural benefits are both
domain- and language-agnostic.
The most substantial improvements appear in tasks requiring
complex reasoning: Multi-hop Reasoning (+22.8% over baseline)
and Summarization Questions (+35.5%), where neither global con-
text alone nor local details alone suffice. This confirms that the
bi-path design addresses a fundamental limitation of single-path
approaches in handling the full spectrum of question-answering
challenges.
6 Conclusion
Limitations.FABLE requires upfront indexing and benefits most
from semantically structured documents. Its advantages diminish
on highly unstructured corpora or queries relying solely on key-
word matching. Code and constructed forests will be released upon
acceptance.
Figure 4: bi-path performance across domains and languages.
Figure 5: bi-path performance across different query types.
This work revisits whether long-context LLMs can replace retrieval-
augmented systems for knowledge-intensive reasoning. We show
that increasing context length alone does not resolve core chal-
lenges such as semantic distraction, hallucination, and inefficient
multi-document evidence utilization.
We introducedFABLE, a forest-based adaptive bi-path retrieval
framework that positions LLMs as knowledge organizers rather
than passive consumers. By constructing LLM-enhanced hierarchi-
cal representations and enabling query-adaptive bi-path retrieval
over global abstractions and local evidence, FABLE balances re-
trieval coverage with reasoning faithfulness. With fixed LLM back-
bones, FABLE consistently outperforms strong RAG and struc-
tured retrieval baselines, achieving performance comparable to
full-context inference with lower token budgets.
Our results indicate that gains stem from retrieval-side architec-
ture rather than simply longer context, emphasizing the importance
of structured knowledge organization and controlled traversal for
efficient, faithful LLM reasoning.
References
[1]Anthropic. 2024. Introducing the Claude 3 Model Family. https://www.anthropic.
com/news/claude-3-family. Accessed: 2024-01-16.
[2]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
arXiv:2310.11511 [cs.CL] https://arxiv.org/abs/2310.11511
9

[3]Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang,
Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al .2024. Longbench: A bilingual,
multitask benchmark for long context understanding. InProceedings of the 62nd
annual meeting of the association for computational linguistics (volume 1: Long
papers). 3119–3137.
[4]Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2024. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity
Text Embeddings Through Self-Knowledge Distillation. InFindings of the As-
sociation for Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins,
and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok,
Thailand, 2318–2335. doi:10.18653/v1/2024.findings-acl.137
[5]Zijian Chen, Xueguang Ma, Shengyao Zhuang, Ping Nie, Kai Zou, Andrew Liu,
Joshua Green, Kshama Patel, Ruoxi Meng, Mingyi Su, Sahel Sharifymoghad-
dam, Yanxi Li, Haoran Hong, Xinyu Shi, Xuye Liu, Nandan Thakur, Crystina
Zhang, Luyu Gao, Wenhu Chen, and Jimmy Lin. 2025. BrowseComp-Plus: A
More Fair and Transparent Evaluation Benchmark of Deep-Research Agent.
arXiv:2508.06600 [cs.CL] https://arxiv.org/abs/2508.06600
[6]DeepSeek-AI. 2025. DeepSeek-V3.2: Pushing the Frontier of Open Large Language
Models.
[7]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From Local to Global: A Graph
RAG Approach to Query-Focused Summarization.arXiv preprint arXiv:2404.16130
(2024).
[8]Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022. Precise Zero-
Shot Dense Retrieval without Relevance Labels. arXiv:2212.10496 [cs.IR] https:
//arxiv.org/abs/2212.10496
[9]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi
Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented
Generation for Large Language Models: A Survey. arXiv:2312.10997 [cs.CL]
https://arxiv.org/abs/2312.10997
[10] Zirui Guo, Xiaohua Lian, Yanhua Yang, Hanzhi Huang, Shuwen Liu, Yixuan Feng,
Yiding Liu, and Jinhao Li. 2024. LightRAG: Simple and Fast Retrieval-Augmented
Generation.arXiv preprint arXiv:2410.05779(2024).
[11] Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. HippoRAG: Neurobiologically Inspired Long-Term Memory for Large
Language Models. InAdvances in Neural Information Processing Systems, Vol. 37.
[12] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From RAG to Memory: Non-Parametric Continual Learning for Large Language
Models. arXiv:2502.14802 [cs.CL] https://arxiv.org/abs/2502.14802
[13] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics. 6609–6625.
[14] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics, Donia Scott, Nuria Bel, and Chengqing Zong (Eds.). International
Committee on Computational Linguistics, Barcelona, Spain (Online), 6609–6625.
doi:10.18653/v1/2020.coling-main.580
[15] Haoyu Huang, Yongfeng Huang, Junjie Yang, Zhenyu Pan, Yongqiang Chen, Kaili
Ma, Hongzhi Chen, and James Cheng. 2025. Retrieval-Augmented Generation
with Hierarchical Knowledge.arXiv preprint arXiv:2503.10150(2025).
[16] Jiajie Jin, Xiaoxi Li, Guanting Dong, Yuyao Zhang, Yutao Zhu, Yongkang Wu,
Zhonghua Li, Ye Qi, and Zhicheng Dou. 2025. Hierarchical Document Refinement
for Long-context Retrieval-augmented Generation. InProceedings of the 63rd
Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher
Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 3502–
3520. doi:10.18653/v1/2025.acl-long.176
[17] Greg Kamradt. 2023. Needle In A Haystack - Pressure Testing LLMs. https:
//github.com/gkamradt/LLMTest_NeedleInAHaystack. Accessed: 2024-01-16.
[18] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval for Open-
Domain Question Answering. InProceedings of the 2020 Conference on Empirical
Methods in Natural Language Processing (EMNLP). 6769–6781.
[19] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks. InAdvances in Neural Information Processing
Systems, Vol. 33. 9459–9474.
[20] Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao. 2023. Making Large Lan-
guage Models A Better Foundation For Dense Retrieval. arXiv:2312.15503 [cs.CL]
[21] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2023. Lost in the Middle: How Language Models
Use Long Contexts. InAdvances in Neural Information Processing Systems, Vol. 36.
[22] Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. MTEB:
Massive Text Embedding Benchmark. InProceedings of the 17th Conference of the
European Chapter of the Association for Computational Linguistics. 2014–2037.[23] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina
Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu
Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew
Knight, Benjamin Chess, and John Schulman. 2022. WebGPT: Browser-assisted
question-answering with human feedback. arXiv:2112.09332 [cs.CL] https:
//arxiv.org/abs/2112.09332
[24] OpenAI. 2023. GPT-4 Turbo: Announcing New Models and Developer Prod-
ucts. https://openai.com/blog/new-models-and-developer-products-announced-
at-devday. Accessed: 2024-01-16.
[25] OpenAI. 2025. gpt-oss-120b & gpt-oss-20b Model Card. arXiv:2508.10925 [cs.CL]
https://arxiv.org/abs/2508.10925
[26] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike
Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language
Models. arXiv:2210.03350 [cs.CL] https://arxiv.org/abs/2210.03350
[27] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-Context Retrieval-Augmented Lan-
guage Models. InTransactions of the Association for Computational Linguistics,
Vol. 11. 1316–1331.
[28] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy
Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat,
Julian Schrittwieser, et al .2024. Gemini 1.5: Unlocking Multimodal Understanding
Across Millions of Tokens of Context.arXiv preprint arXiv:2403.05530(2024).
[29] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and
Christopher D. Manning. 2024. RAPTOR: Recursive Abstractive Processing for
Tree-Organized Retrieval. InInternational Conference on Learning Representations.
[30] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli,
Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer:
Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761 [cs.CL]
https://arxiv.org/abs/2302.04761
[31] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike
Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. REPLUG: Retrieval-Augmented
Black-Box Language Models.arXiv preprint arXiv:2301.12652(2023).
[32] Yixuan Tang and Yi Yang. 2024. MultiHop-RAG: Benchmarking Retrieval-
Augmented Generation for Multi-Hop Queries. arXiv:2401.15391 [cs.CL] https:
//arxiv.org/abs/2401.15391
[33] Wenyu Tao, Xiaofen Xing, Yirong Chen, Linyi Huang, and Xiangmin Xu. 2025.
TreeRAG: Unleashing the Power of Hierarchical Storage for Enhanced Knowledge
Retrieval in Long Documents. InFindings of the Association for Computational
Linguistics: ACL 2025. Association for Computational Linguistics, Vienna, Austria,
356–371. doi:10.18653/v1/2025.findings-acl.20
[34] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https:
//arxiv.org/abs/2505.09388
[35] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-
Intensive Multi-Step Questions. InProceedings of the 61st Annual Meeting of the
Association for Computational Linguistics. 10014–10037.
[36] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is All
You Need. InAdvances in Neural Information Processing Systems. 5998–6008.
[37] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing. 2369–2380.
[38] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. arXiv:2210.03629 [cs.CL] https://arxiv.org/abs/2210.03629
[39] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.
2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through
Foundation Models.arXiv preprint arXiv:2506.05176(2025).
[40] Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan, Zhenghao Liu, Shi Yu, Ruobing
Wang, Shuo Wang, Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun.
2025. RAGEval: Scenario Specific RAG Evaluation Dataset Generation Framework.
InProceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina
Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational
Linguistics, Vienna, Austria, 8520–8544. doi:10.18653/v1/2025.acl-long.418
10