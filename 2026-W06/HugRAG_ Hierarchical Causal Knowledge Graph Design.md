# HugRAG: Hierarchical Causal Knowledge Graph Design for RAG

**Authors**: Nengbo Wang, Tuo Liang, Vikash Singh, Chaoda Song, Van Yang, Yu Yin, Jing Ma, Jagdip Singh, Vipin Chaudhary

**Published**: 2026-02-04 23:59:02

**PDF URL**: [https://arxiv.org/pdf/2602.05143v1](https://arxiv.org/pdf/2602.05143v1)

## Abstract
Retrieval augmented generation (RAG) has enhanced large language models by enabling access to external knowledge, with graph-based RAG emerging as a powerful paradigm for structured retrieval and reasoning. However, existing graph-based methods often over-rely on surface-level node matching and lack explicit causal modeling, leading to unfaithful or spurious answers. Prior attempts to incorporate causality are typically limited to local or single-document contexts and also suffer from information isolation that arises from modular graph structures, which hinders scalability and cross-module causal reasoning. To address these challenges, we propose HugRAG, a framework that rethinks knowledge organization for graph-based RAG through causal gating across hierarchical modules. HugRAG explicitly models causal relationships to suppress spurious correlations while enabling scalable reasoning over large-scale knowledge graphs. Extensive experiments demonstrate that HugRAG consistently outperforms competitive graph-based RAG baselines across multiple datasets and evaluation metrics. Our work establishes a principled foundation for structured, scalable, and causally grounded RAG systems.

## Full Text


<!-- PDF content starts -->

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
Nengbo Wang1 2Tuo Liang1Vikash Singh1Chaoda Song1Van Yang1
Yu Yin1Jing Ma1Jagdip Singh2Vipin Chaudhary1
Abstract
Retrieval augmented generation (RAG) has en-
hanced large language models by enabling access
to external knowledge, with graph-based RAG
emerging as a powerful paradigm for structured
retrieval and reasoning. However, existing graph-
based methods often over-rely on surface-level
node matching and lack explicit causal modeling,
leading to unfaithful or spurious answers. Prior
attempts to incorporate causality are typically lim-
ited to local or single-document contexts and also
suffer from information isolation that arises from
modular graph structures, which hinders scalabil-
ity and cross-module causal reasoning. To address
these challenges, we propose HugRAG, a frame-
work that rethinks knowledge organization for
graph-based RAG through causal gating across
hierarchical modules. HugRAG explicitly models
causal relationships to suppress spurious correla-
tions while enabling scalable reasoning over large-
scale knowledge graphs. Extensive experiments
demonstrate that HugRAG consistently outper-
forms competitive graph-based RAG baselines
across multiple datasets and evaluation metrics.
Our work establishes a principled foundation for
structured, scalable, and causally grounded RAG
systems.
1. Introduction
While Retrieval-Augmented Generation (RAG) effectively
extends Large Language Models (LLMs) with external
knowledge (Lewis et al., 2021), traditional pipelines pre-
dominantly rely on text chunking and semantic embedding
search. This paradigm implicitly frames knowledge ac-
cess as a flat similarity matching problem, overlooking the
structured and interdependent nature of real-world concepts.
1Department of Computer and Data Sciences, Case Western Re-
serve University, Cleveland, OH, USA2Design and Innovation De-
partment, Case Western Reserve University, Cleveland, OH, USA.
Correspondence to: Nengbo Wang<nengbo.wang@case.edu>.
Preprint. February 6, 2026.Consequently, as knowledge bases scale in complexity, these
methods struggle to maintain retrieval efficiency and reason-
ing fidelity.
Graph-based RAG has emerged as a promising solution
to address these gaps, led by frameworks like GraphRAG
(Edge et al., 2024) and extended through agentic search
(Ravuru et al., 2024), GNN-guided refinement (Liu et al.,
2025a), and hypergraph representations (Luo et al.). How-
ever, three unintended limitations still persist. First, cur-
rent research prioritizes retrieval policies while overlook-
ing knowledge graph organization. As graphs scale, in-
trinsic modularity (Fortunato & Barth ´elemy, 2007) often
restricts exploration within dense modules, triggeringin-
formation isolation.Common grouping strategies rang-
ing from communities (Edge et al., 2024), passage nodes
(Guti ´errez et al., 2025), node-edge sets (Guo et al., 2024) to
semantic grouping (Zhang et al., 2025) often inadvertently
reinforce these boundaries, severely limiting global recall.
Second, most formulations rely on semantic proximity and
superficial traversal on graphs withoutcausal awareness,
leading to alocality issuewhere spurious nodes and irrel-
evant noise degrade precision (see Figure 1). Despite the
inherent causal discovery potential of LLMs, this capability
remains largely untapped for filtering noise within RAG
pipelines. Finally, these systemic flaws are often masked by
popular QA datasets evaluation, which reward entity-level
“hits” over holistic comprehension. Consequently, there is
a pressing need for a retrieval framework thatreconciles
global knowledge accessibility with local reasoning pre-
cisionto support robust, causally-grounded generation.
To address these challenges, we propose HugRAG, a frame-
work that rethinks knowledge graph organization throughhi-
erarchical causal gate structures. HugRAG formulates the
knowledge graph as a multi-layered representation where
fine-grained facts are organized into higher-level schemas,
enabling multi-granular reasoning. This hierarchical ar-
chitecture, integrated with causal gates, establishes logical
bridges across modules, thereby naturally breaking informa-
tion isolation and enhancing global recall. During retrieval,
HugRAG transcends pointwise semantic matching to ex-
plicit reasoning over causal graphs. By actively distinguish-
ing genuine causal dependencies from spurious associations,
HugRAG mitigates the locality issue and filters retrieval
1arXiv:2602.05143v1  [cs.AI]  4 Feb 2026

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
H u g R A G
T r a f f i c  D e l a y sM 3 :  R o a d  O u t c o m e sU n m a n a g e d  
j u n c t i o n sG r i d l o c kM 1M 2M 3
K n o w l e d g e  G r a p hS e e d  N o d eN - h o p  N o d e s  /  S p u r i o u s  N o d e sG r a p h  M o d u l e sC a u s a l  G a t eC a u s a l  P a t hC o n t r o l l e r s  
d o w nF l a s h i n g  
m o d eM 2 :  S i g n a l  C o n t r o lS u b s t a t i o n  
f a u l tB l a c k o u tP o w e r  
r e s t o r e dM 1 :  P o w e r  O u t a g eQ u e r y :  W h y  d i d  c i t y w i d e  c o m m u t e  d e l a y s  s u r g e  r i g h t  a f t e r  t h e  b l a c k o u t ?
A n s w e r :  B l a c k o u t  k n o c k e d  o u t  s i g n a l  c o n t r o l l e r s ,  i n t e r s e c t i o n s  w e n t  fl a s h i n g ,  g r i d l o c k  s p r e a d .. . . . . . . . . . . . . . .  
 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .S u b s t a t i o n  f a u l t  c a u s e d  a  
c i t y w i d e  b l a c k o u t
. . . . . . . . . . . . . S i g n a l  c o n t r o l l e r  n e t w o r k  l o s t  
p o w e r .  M a n y  j u n c t i o n s  w e n t  f l a s h i n g  .. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
S u b s t a t i o n  
f a u l tB l a c k o u tP o w e r  
r e s t o r e dM 1 :  P o w e r  O u t a g e
C o n t r o l l e r s  
d o w nF l a s h i n g  
m o d eM 2 :  S i g n a l  C o n t r o lT r a f f i c  D e l a y sM 3 :  R o a d  O u t c o m e sU n m a n a g e d  
j u n c t i o n sG r i d l o c kS t a n d a r d  R A GG r a p h - b a s e d  R A G
H a r d  t o  b r e a k  c o m m u n i t i e s  /  i n t r i n s i c  m o d u l a r i t yB r e a k  i n f o r m a t i o n  i s o l a t i o n  &  I d e n t i f y  c a u s a l  p a t hS e m a n t i c  s e a r c h  m i s s e s  k e y  c o n t e x tM i s s e d  (  N o  k e y w o r d  m a t c h ). . . . . . .  S t o p  a n d  g o  b a c k u p s  a n d  g r i d l o c k  
a c r o s s  m a j o r  c o r r i d o r s  . . . . . . . . . . . . . . . . . . . . . . . . . . .. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
Figure 1.Comparison of three retrieval paradigms, Standard RAG, Graph-based RAG, and HugRAG, on a citywide blackout query.
Standard RAG misses key evidence under semantic retrieval. Graph-based RAG can be trapped by intrinsic modularity or grouping
structure. HugRAG leverages hierarchical causal gates to bridge modular boundaries, effectively breaking information isolation and
explicitly identifying the underlying causal path.
noise to ensure precise, grounded, and interpretable genera-
tion.
To validate the effectiveness of HugRAG, we conduct ex-
tensive evaluations across datasets in multiple domains,
comparing it against a diverse suite of competitive RAG
baselines. To address the previously identified limitations
of existing QA datasets, we introduce a large-scale cross-
domain dataset HolisQA focused onholistic comprehen-
sion, designed to evaluate reasoning capabilities in complex,
real-world scenarios. Our results consistently demonstrate
that causal gating and causal reasoning effectively recon-
cile the trade-off between recall and precision, significantly
enhancing retrieval quality and answer reliability.
2. Related Work
2.1. RAG
Retrieval augmented generation grounds LLMs in external
knowledge, but chunk level semantic search can be brittle
and inefficient for large, heterogeneous, or structured cor-
pora (Lewis et al., 2021). Graph-based RAG has therefore
emerged to introduce structure for more informed retrieval.
Graph-based RAG.GraphRAG constructs a graph struc-
tured index of external knowledge and performs query time
retrieval over the graph, improving question focused ac-
cess to large scale corpora (Edge et al., 2024). Building
on this paradigm, later work studies richer selection mecha-
nisms over structured graph. Agent driven retrieval exploresthe search space iteratively (Ravuru et al., 2024). Critic
guided or winnowing style methods prune weak contexts
after retrieval (Dong et al.; Wang et al., 2025b). Others
learn relevance scores for nodes, subgraphs, or reasoning
paths, often with graph neural networks (Liu et al., 2025a).
Representation extensions include hypergraphs for higher
order relations (Luo et al.) and graph foundation models for
retrieval and reranking (Wang et al.).
Knowledge Graph Organization.Despite these ad-
vances, limitations related to graph organization remain
underexamined. Most work emphasizes retrieval policies,
while the organization of the underlying knowledge graph is
largely overlooked, which strongly influences downstream
retrieval behavior. As graphs scale, intrinsic modularity can
emerge (Fortunato & Barth ´elemy, 2007; Newman, 2018),
making retrieval prone to staying within dense modules
rather than crossing them, largely limiting the retrieved in-
formation. Moreover, many work assume grouping knowl-
edge for efficiency at scale, such as communities (Edge et al.,
2024), phrases and passages(Guti ´errez et al., 2025), node
edge sets (Guo et al., 2024), or semantic aggregation (Zhang
et al., 2025) (see Table 1), which can amplify modular con-
finement and yieldinformation isolation. Thisglobal issue
primarily manifests asreduced recall. Some hierarchical
approaches like LeanRAG attempt to bridge these gaps via
semantic aggregation, but they remain constrained by seman-
tic clustering and rely on tree-structured traversals (Zhang
et al., 2025), often failing to capture logical dependencies
that span across semantically distinct clusters.
2

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
Method Knowledge Graph Organization Retrieval and Generation Process
Standard RAG (Lewis et al., 2021) Flat text chunks, unstructured.
Gidx={d i}N
i=1Semantic vector search over chunks.
S= TopK(sim(q, d i));y=G(q, S)
Graph RAG (Edge et al., 2024) Partitioned communities with summaries.
Gidx={Sum(c)|c∈ C}Map-Reduceover community summaries.
Apart={G(q,Sum(m))};y=G(A part)
Light RAG (Guo et al., 2024) Dual-level indexing (Entities + Relations).
Gidx= (V ent∪V rel, E)Keyword-basedvector retrieval + neighbor.
Kq=Key(q);S= Vec(K q,Gidx)∪ N 1
HippoRAG 2 (Guti ´errez et al., 2025) Dense-sparse integration (Phrase + Passage).
Gidx= (V phrase∪V doc, E)PPRdiffusion from LLM-filtered seeds.
Useed=Filter(q, V);S=PPR(U seed,Gidx)
LeanRAG (Zhang et al., 2025) Hierarchical semantic clusters (GMM).
Gidx=Tree(Semantic Aggregation)Bottom-uptraversal to LCA (Ancestor).
U= TopK(q, V);S=LCA(U,G idx)
CausalRAG (Wang et al., 2025a) Flat graph structure.
Gidx= (V, E)Top-K retrieval + Implicit causal reasoning.
S=Expand(TopK(q, V));y=G(q, S)
HugRAG (Ours) Hierarchical Causal Gatesacross modules.
Gidx=H={H 0, . . . , H L}Causal Gating+Causal Path Filtering.
S=Traverse(q,H)| {z }
Break Isolation∩Filter causal(S)| {z }
Reduce Noise
Table 1.Comparison of RAG frameworks based on knowledge organization and retrieval mechanisms.Notation: Mmodules, Sum(·)
summary,PPRPersonalized PageRank,Hhierarchy,N 11-hop neighborhood.
Retrieval Issue.A second limitation concerns how re-
trieval is formulated. Much work operates as a multi-hop
search over nodes or subgraphs (Guti ´errez et al., 2025; Liu
et al., 2025b), prioritizing semantic proximity to the query
without explicit awareness of the reasoning in this search-
ing process. This design can pull in topically similar yet
causally irrelevant evidence, producing conflated retrieval
results. Even when the correct fact node is present, the gen-
erator may respond with generic or superficial content, and
the extra noise can increase the risk of hallucination. We
view this as alocality issue that lowers precision.
QA Evaluation Issue.These tendencies can be rein-
forced by common QA evaluation practice. First, many
QA datasets emphasize short answers such as names, na-
tionalities, or years (Kwiatkowski et al., 2019; Rajpurkar
et al., 2016), sohitting the correct entityin the graph
may be sufficient even without reasoning. Second, QA
datasets often comprise thousands of independent question-
answer-context triples. However, many approaches still rely
on linear context concatenation to construct a graph, and
then evaluate performance on isolated questions. This setup
largely reduces the incentive for holistic comprehension
of the underlying material, even though such end-to-end
understanding is closer to real-world use cases. Third, some
datasets are stale enough that answers may be partially mem-
orized by pretrained LLM models,confounding retrieval
quality with parametric knowledge. Therefore, these QA
dataset issues are critical for evaluating RAG, yet relatively
few works explicitly address them by adopting open-ended
questions and fresher materials in controlled experiments.2.2. Causality
LLM for Identifying Causality.LLMs have demon-
strated exceptional potential in causal discovery. By lever-
aging vast domain knowledge, LLMs significantly improve
inference accuracy compared to traditional methods (Ma,
2024). Frameworks like CARE further prove that fine-tuned
LLMs can outperform state-of-the-art algorithms (Dong
et al., 2025). Crucially, even in complex texts, LLMs main-
tain a direction reversal rate under 1.1% (Saklad et al., 2026),
ensuring highly reliable results.
Causality and RAG.While LLMs increasingly demon-
strate reliable causal reasoning capabilities, explicitly in-
tegrating causal structures into RAG remains largely un-
derexplored. Current research predominantly focuses on
internal attribution graphs for model interpretability (Walker
& Ewetz, 2025; Dai et al., 2025), rather than external knowl-
edge retrieval. Recent advances like CGMT (Luo et al.,
2025) and LACR (Zhang et al., 2024) have begun to bridge
this gap, utilizing causal graphs for medical reasoning path
alignment or constraint-based structure induction. How-
ever, these works inherently differ in scope from our objec-
tive, as they prioritize rigorous causal discovery or recovery
tasks in specific domain, which limits their scalability to the
noisy, open-domain environments that we address. Exist-
ing causal-enhanced RAG frameworks either utilize causal
feedback implicitly in embedding (Khatibi et al., 2025) or,
like CausalRAG (Wang et al., 2025a), are restricted to small-
scale settings with implicit causal reasoning. Consequently,
a significant gap persists in leveraging causal graphs to
guide knowledge graph organization and retrieval across
large-scale, heterogeneous knowledge bases. Note that in
this work, we use the termcausalto denote explicit logi-
cal dependencies and event sequences described in the text,
3

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
rather than statistical causal discovery from observational
data.
3. Problem Formulation
We aim to retrieve an optimal subgraph S∗⊆ G for a query q
to generate an answer y. Graph-based RAG ( S=R(q,G) )
usually faces two structural bottlenecks.
1. Global Information Isolation (Recall Gap).Intrin-
sic modularity often traps retrieval in local seeds, missing
relevant evidence v∗located in topologically distant mod-
ules (i.e., S∩ {v∗}=∅ as no path exists within hhops).
HugRAGintroducescausal gatesacross H, to bypass mod-
ular boundaries and bridge this gap. The efficacy of causal
gates is empirically verified in Appendix E and further ana-
lyzed in the ablation study (see Section 5.3).
2. Local Spurious Noise (Precision Gap).Semantic
similarity sim(q, v) often retrieves topically related but
causally irrelevant nodes Vsp, diluting precision (where
|S∩ V sp| ≫ |S∩ V causal|). We address this by lever-
aging LLMs to identify explicitcausal paths, filtering Vsp
to ensure groundedness. While as discussed LLMs have
demonstrated causal identification capabilities surpassing
human experts (Ma, 2024; Dong et al., 2025) and proven
effectiveness in RAG (Wang et al., 2025a), we further cor-
roborate the validity of identified causal paths through ex-
pert knowledge across different domains (see Section 5.1).
Consequently, HugRAG redefines retrieval as finding a map-
pingΦ :G → H and a causal filter Fcto simultaneously
minimize isolation and spurious noise.
4. Method
Overview.As illustrated in Figure 2, HugRAG operates in
two distinct phases to address the aforementioned structural
bottlenecks. In theoffline phase, we construct a hierarchi-
cal knowledge structure Hpartitioned into modules, which
are then interconnected viacausal gates Gcto enable logical
traversals. In theonline phase, HugRAG performs agated
expansionto break modular isolation, followed by acausal
filteringstep to eliminate spurious noise. The overall pro-
cedure is formalized in Algorithm 1, and we detail each
component in the subsequent sections.
4.1. Hierarchical Graph with Causal Gating
To address theglobal information isolationchallenge (Sec-
tion 3), we construct a multi-scale knowledge structure that
balances global retrieval recall with local precision.
Hierarchical Module Construction.We first extract a
base entity graph G0= (V 0, E0)from the corpus DusingAlgorithm 1HugRAG Algorithm Pipeline
Require: Corpus D, query q, hierarchy levels L, seed bud-
get{K ℓ}L
ℓ=0, hoph, gate thresholdτ
Ensure:Answery, Support SubgraphS∗
1:// Phase 1: Offline Hierarchical Organization
2:G 0= (V 0, E0)←BUILDBASEGRAPH(D)
3:H={H 0, . . . , H L} ←LEIDENPARTITION(G 0, L)
{Organize into modulesM}
4:G c← ∅
5:for allpair(m i, mj)∈MODULEPAIRS(M)do
6:score←LLM-ESTCAUSAL(m i, mj)
7:ifscore≥τthen
8:G c← G c∪{(m i→m j, score)} { Establish causal
gates}
9:end if
10:end for
11:// Phase 2: Online Retrieval & Reasoning
12:U←SL
ℓ=0TopK(sim(q, u), K ℓ, Hℓ){Multi-level se-
mantic seeding}
13:S raw←GATEDTRAVERSAL(U,H,G c, h){ Break iso-
lation via gates}
14:S∗←CAUSALFILTER(q, S raw){Remove spurious
nodesV sp}
15:y←LLM-GENERATE(q, S∗)
an information extraction pipeline (see details in Appendix
B.1), followed by entity canonicalization to resolve aliasing.
To establish the hierarchical backbone H={H 0, . . . , H L},
we iteratively partition the graph intomodulesusing the
Leiden algorithm (Traag et al., 2019), which optimizes mod-
ularity to identify tightly-coupled semantic regions. For-
mally, at each level ℓ, nodes are partitioned into modules
Mℓ={m(ℓ)
1, . . . , m(ℓ)
k}. For each module, we generate
a natural language summary to serve as a coarse-grained
semantic anchor.
Offline Causal Gating.While hierarchical modularity
improves efficiency, it risks trapping retrieval within local
boundaries. We introduceCausal Gatesto explicitly model
cross-module affordances. Instead of fully connecting the
graph, we construct a sparse gate set Gc. Specifically, we
identify candidate module pairs (mi, mj)that are topologi-
cally distant but potentially logically related. An LLM then
evaluates the plausibility of a causal connection between
their summaries. We formally define the gate set via an
indicator functionI(·):
Gc={(m i→m j)|I causal(mi, mj) = 1},(1)
where Icausal denotes the LLM’s assessment (see Appendix
B.1 for construction prompts and the Top-Down Hierarchi-
cal Pruning strategy we employed to mitigate the O(N2)
evaluation complexity). These gates act asshortcutsin the
4

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
L L ML L M
. . .    . . .    . . .    . . .    . . .    . . .    . . .    . . .I E
. . .    . . .    . . .    . . .    . . .    . . .    . . .    . . .HnH0Hn - 1. . .. . .    . . .    . . .    . . .    . . .    . . .    . . .    . . .. . .    . . .    . . .    . . .    . . .    . . .    . . .    . . .D i s t i n g u i s h  C a u s a l  v s  S p u r i o u s  I d e n t i f y  C a u s a l i t yE m b e dG r a p h  C o n s t r u c t i o n  ( O f fl i n e )R a w  T e x t sK n o w l e d g e  G r a p hV e c t o r  S t o r eP a r t i t i o nH i e r a r c h i c a l  G r a p hG r a p h  w i t h  C a u s a l  G a t e s  T o p  K  e n t i t i e sQ u e r yN  h o p  v i a  g a t e s ,  c r o s s  m o d u l e sw i t h  Q u e r yE m b e d  a n d  S c o r eC o n t e x t  S u b g r a p hC o n t e x tA n s w e rR e t r i e v e  a n d  A n s w e r  ( O n l i n e )
Figure 2.Overview of the HugRAG pipeline.In the offline stage, raw texts are embedded to build a knowledge graph and a vector store,
then partitioning forms a hierarchical graph and an LLM identifies causal relations to construct a graph with causal gates. In the online
stage, the query is embedded and scored to retrieve top K entities, then N hop traversal uses causal gates to cross modules and assemble a
context subgraph; an LLM further distinguishes causal versus spurious relations to produce the final context and answer.
retrieval space, permitting the traversal to jump across dis-
joint modules only when logically warranted, thereby break-
ing information isolation without causing semantic drift (see
Appendix C for visualizations of hierarchical modules and
causal gates).
4.2. Retrieve Subgraph via Causally Gated Expansion
Given the hierarchical structure Hand causal gates Gc,
HugRAG retrieves a support subgraph Sby coupling multi-
granular anchoring with a topology-aware expansion. This
process is designed to maximize recall (breaking isolation)
while suppressing drift (controlled locality).
Multi-Granular Hybrid Seeding.Graph-based RAG of-
ten struggles to effectively differentiate between local details
and global contexts within multi-level structures (Zhang
et al., 2025; Edge et al., 2024). We overcome this by iden-
tifying a seed set Uacross multiple levels of the hierarchy.
We employ a hybrid scoring function s(q, v) that interpo-
lates between semantic embedding similarity and lexical
overlap (details in Appendix B.2). This function is applied
simultaneously to fine-grained entities in H0and coarse-
grained module summaries in Hℓ>0. Crucially, to prevent
thesemantic redundancyproblem where seeds cluster in a
single redundant neighborhood, we apply a diversity-aware
selection strategy (MMR) to ensure the initial seeds Ucover
distinct semantic facets of the query. This yields a set of
anchors that serve as the starting nodes for expansion.
Gated Priority Expansion.Starting from the seed set U,
we model retrieval as a priority-based traversal over a unified
edge space Euni. This space integrates three distinct types of
connectivity: (1)Structural Edges( Estruc) for local context,
(2)Hierarchical Edges( Ehier) for vertical drill-down, and
(3)Causal Gates(G c) for cross-module reasoning.
Euni=E struc∪E hier∪ Gc.(2)The expansion follows a Best-First Search guided by a
query-conditioned gain function. For a frontier node v
reached from a predecessor uat hop t, the gain is defined
as:
Gain(v) =s(q, v)·γt·w(type(u, v)),(3)
where γ∈(0,1) is a standard decay factor to penalize
long-distance traversal. The weight function w(·) adjusts
traversal priorities: we simply assign higher importance
to causal gates and hierarchical links to encourage logic-
driven jumps over random structural walks. By traversing
Euni, HugRAG prioritizes paths that drill down (via Ehier),
explore locally (via Estruc), or leap to a causally related
domain (via Gc), effectivelybreaking modular isolation.
The expansion terminates when the gain drops below a
threshold or the token budget is exhausted.
4.3. Causal Path Identification and Grounding
The raw subgraph Sraw retrieved via gated expansion
optimizes for recall but inevitably includes spurious
associations(e.g., high-degree hubs or coincidental co-
occurrences). To address thelocal spurious noisechallenge
(Section 3), HugRAG employs a causal path refinement
stage to directly distill Srawinto a causally grounded graph
S⋆. See Appendix D for a full example of the HugRAG
pipeline.
Causal Path Refinement.We formulate the path refine-
ment task as a structural pruning process. We first linearize
the subgraph Srawinto a token-efficient table where each
node and edge is mapped to a unique short identifier (see
Appendix B.3). The LLM is then prompted to analyze the
topology and output the subset of identifiers that consti-
tute valid causal paths connecting the query to the potential
answer. Leveraging the robust causal identification capabili-
ties of LLMs (Saklad et al., 2026), this operation effectively
functions as a reranker, distilling the noisy subgraph into an
5

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
Datasets Nodes Edges Modules Size (Char) Domain
MS MARCO (Bajaj et al., 2018) 3,403 3,107 446 1,557,990 Web
NQ (Kwiatkowski et al., 2019) 5,579 4,349 505 767,509 Wikipedia
2WikiMultiHopQA (Ho et al., 2020) 10,995 8,489 1,088 1,756,619 Wikipedia
QASC (Khot et al., 2020) 77 39 4 58,455 Science
HotpotQA (Yang et al., 2018) 20,354 15,789 2,359 2,855,481 Wikipedia
HolisQA-Biology 1,714 1,722 165 1,707,489 Biology
HolisQA-Business 2,169 2,392 292 1,671,718 Business
HolisQA-CompSci 1,670 1,667 158 1,657,390 Computer Science
HolisQA-Medicine 1,930 2,124 226 1,706,211 Medicine
HolisQA-Psychology 2,019 1,990 211 1,751,389 Psychology
Table 2.Statistics of the datasets used in evaluation.
explicit causal structure:
S⋆=LLM-CAUSALEXPERT(S raw, q).(4)
The returned subgraph S⋆contains only model-validated
nodes and edges, effectively filtering irrelevant context.
Spurious-Aware Grounding.To further improve the pre-
cision of this selection, we employ aspurious-aware
prompting strategy(see prompts in Appendix A.1). In
this configuration, the LLM is instructed to explicitly dis-
tinguish between causal supports and spurious correlations
during its reasoning process. While the prompt may ask
the model to identify spurious items as an auxiliary rea-
soning step, the primary objective remains the extraction
of the valid causal subset. This explicit contrast helps the
model resisthallucinated connectionsinduced by semantic
similarity, yielding a cleaner S⋆compared to standard se-
lection prompts and consequently improving downstream
generation quality. This mechanism specifically targets the
precision challenges outlined in Section 4.2. Finally, the
answer yis generated by conditioning the LLM solely on
the text content corresponding to the pruned subgraph S⋆
(see prompts in Appendix A.2), ensuring that the generation
is strictly grounded in verified evidence.
5. Experiments
Overview.We conducted extensive experiments on di-
verse datasets across various domains to comprehensively
evaluate and compare the performance of HugRAG against
competitive baselines. Our analysis is guided by the follow-
ing five research questions:
RQ1 (Overall Performance).How does HugRAG compare
against state-of-the-art graph-based baselines across diverse,
real-world knowledge domains?
RQ2 (QA vs. Holistic Comprehension).Do popular
QA datasets implicitly favor the entity-centric retrieval
paradigm, thereby inflating graph-based RAG that finds
the right node without assembling a support chain?
RQ3 (Trade-off Reconciliation).Can HugRAG simulta-neously improveContext Recall(Globality) andAnswer
Relevancy(Precision), mitigating the classic trade-off via
hierarchical causal gating?
RQ4 (Ablation Study).What are the individual contribu-
tions of different components in HugRAG?
RQ5 (Scalability Robustness).How does HugRAG’s per-
formance scale and remain robust under varying context
lengths?
5.1. Experimental Setup
Datasets.We evaluate HugRAG on a diverse suite of
datasets covering complementary difficulty profiles. For
standard evaluation, we use five established datasets:MS
MARCO(Bajaj et al., 2018) andNatural Questions
(Kwiatkowski et al., 2019) emphasize large-scale open-
domain retrieval;HotpotQA(Yang et al., 2018) and2Wiki-
MultiHop(Ho et al., 2020) require evidence aggregation;
andQASC(Khot et al., 2020) targets compositional scien-
tific reasoning. However, these datasets often suffer from
entity-centric biases and potential data leakage (memoriza-
tion by LLMs). To rigorously test the holistic understanding
capability of RAG, we introduceHolisQA, a dataset derived
from high-quality academic papers sourced (Priem et al.,
2022). Spanning over diverse domains (including Biology,
Computer Science, Medicine, etc.), HolisQA features dense
logical structures that naturally demand holistic comprehen-
sion (see more details in Appendix F.2). All dataset statistics
are summarized in Table 2. While LLMs have demonstrated
strong capabilities in identifying causality (Ma, 2024; Dong
et al., 2025) and effectiveness in RAG (Wang et al., 2025a),
to ensure rigorous evaluation, we incorporated cross-domain
expert review to validate the quality of baseline answers and
confirm the legitimacy of the induced causal relations.
Baselines.We compare HugRAG against eight baselines
spanning three retrieval paradigms. First, to cover Naive and
Flat approaches, we includeNaive Generation(no retrieval)
as a lower bound, alongsideBM25(sparse) andStandard
RAG(Lewis et al., 2021) (dense embedding-based), rep-
resenting mainstream unstructured retrieval. Second, we
6

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
Table 3.Main results on HolisQA across five domains.We reportF1(answer overlap),CR(Context Recall: how much gold context is
covered by retrieved evidence), andAR(Answer Relevancy: evaluator-judged relevance of the answer to the question), all scaled to %for
readability.Boldindicates best per column. NaiveGeneration has CR= 0by definition (no retrieval).
Baselines Medicine Computer Science Business Biology Psychology
F1 CR AR F1 CR AR F1 CR AR F1 CR AR F1 CR AR
Naive Baselines
NaiveGeneration 12.63 0.00 44.70 18.93 0.00 48.79 18.58 0.00 46.14 11.71 0.00 45.76 22.91 0.00 50.00
BM25 17.72 52.04 50.64 24.00 39.12 52.40 28.11 37.06 55.52 19.61 43.02 52.32 30.46 33.44 56.63
StandardRAG 26.87 61.08 56.24 28.87 49.44 57.10 47.57 46.79 67.42 28.31 42.69 57.58 37.19 52.21 59.85
Graph-based RAG
GraphRAG Global 17.13 54.56 48.19 23.75 37.65 53.17 23.62 25.01 48.12 20.67 40.90 52.41 31.09 34.26 54.62
GraphRAG Local 19.03 56.07 49.52 25.10 39.90 53.30 25.01 27.36 49.05 22.21 41.88 52.73 32.31 35.22 55.02
LightRAG 12.16 52.38 44.15 22.59 41.86 51.62 29.98 34.22 54.50 17.70 41.24 50.32 33.63 45.54 56.42
Structural / Causal Augmented
HippoRAG2 21.12 57.50 51.08 16.94 21.05 47.29 21.10 18.34 45.83 12.60 16.85 44.56 20.10 34.13 46.77
LeanRAG 34.25 60.43 56.60 30.51 57.61 55.45 48.30 59.29 60.35 33.82 58.43 56.10 42.85 57.46 58.65
CausalRAG 31.12 58.90 58.77 30.98 54.10 57.54 45.20 44.55 66.10 33.50 51.20 58.90 42.80 55.60 61.90
HugRAG (ours)36.45 69.91 60.65 31.60 60.94 58.34 51.51 67.34 68.76 34.80 61.97 59.99 44.42 60.87 63.53
Table 4.Main results on five QA datasets.Metrics follow Table 3:F1,CR(Context Recall), andAR(Answer Relevancy), reported in
%.Boldand underline denote best and second-best per column.
Baselines MSMARCO NQ TwoWiki QASC HotpotQA
F1 CR AR F1 CR AR F1 CR AR F1 CR AR F1 CR AR
Naive Baselines
NaiveGeneration 5.28 0.00 15.06 7.17 0.00 10.94 9.15 0.00 11.77 2.69 0.00 13.74 14.38 0.00 15.74
BM25 6.97 45.78 20.33 4.68 49.98 9.13 9.43 37.12 13.73 2.49 6.12 13.17 15.81 41.08 16.08
StandardRAG 14.93 48.55 31.11 7.57 45.82 11.14 10.33 32.28 13.57 2.01 5.50 13.16 6.68 43.17 14.66
Graph-based RAG
GraphRAG Global 9.41 3.65 13.08 3.91 4.48 8.00 1.41 9.42 9.55 0.68 3.38 3.56 6.28 14.59 16.26
GraphRAG Local 30.87 25.71 57.76 23.56 44.56 44.68 18.85 32.03 37.29 8.30 9.54 46.59 33.14 44.07 40.82
LightRAG 37.70 54.22 63.54 24.97 60.65 50.53 14.44 40.98 36.56 8.20 20.40 44.35 28.3948.1743.78
Structural / Causal Augmented
HippoRAG2 23.35 45.45 55.18 29.64 57.21 37.50 18.4755.5317.34 14.734.3849.94 38.80 42.06 24.66
LeanRAG 38.02 54.01 58.49 35.46 65.91 49.87 20.27 40.53 38.37 13.19 22.80 45.51 48.68 46.29 43.50
CausalRAG 27.66 39.38 46.03 29.45 68.04 17.35 15.93 28.38 19.76 7.65 46.86 35.56 40.00 27.83 21.32
HugRAG (ours)38.40 60.48 66.02 49.50 70.36 55.09 31.9741.95 42.67 13.35 70.8049.40 64.8340.3045.72
evaluate established graph-based frameworks:GraphRAG
(Local and Global) (Edge et al., 2024), utilizing commu-
nity summaries; andLightRAG(Guo et al., 2024), relying
on dual-level keyword-based search. Third, we benchmark
against RAGs with structured or causal augmentation:Hip-
poRAG 2(Guti ´errez et al., 2025), utilizing passage nodes
and Personalized PageRank diffusion;LeanRAG(Zhang
et al., 2025), employing semantic aggregation hierarchies
and tree-based LCA retrieval; andCausalRAG(Wang et al.,
2025a), which incorporates causality without explicit causal
reasoning. This selection comprehensively covers the spec-
trum from unstructured search to advanced structure-aware
and causally augmented graph methods.
Metrics.For metrics, we first report the token-level an-
swer quality metric F1 for surface robustness. To measure
whether retrieval actually supports generation, we addition-
ally compute grounding metrics, context recall and answerrelevancy (Es et al., 2024), which jointly capture coverage
and answer quality (see Appendix F.4).
Implementation Details.For all experiments, we uti-
lize gpt-5-nano as the backbone LLM for both the open
IE extraction and generation stages, and Sentence-BERT
(Reimers & Gurevych, 2019) for semantic vectorization. For
HugRAG, we set the hierarchical seed budget to KL= 3
for modules and K0= 3for entities, causal gate is enabled
by default except ablation study. Experiments run on a clus-
ter using 10-way job arrays; each task uses 2 CPU cores
and 16 GB RAM (20 cores, 160GB in total). See more
implementation details in Appendix F.3.
5.2. Main Experiments
Overall Performance (RQ1).HugRAG consistently
achieves superior performance across all HolisQA domains
7

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
and standard QA metrics (Table 3, Table 4). While tradi-
tional methods (e.g., BM25, Standard RAG) struggle with
structural dependencies, graph-based baselines exhibit dis-
tinct limitations. GraphRAG-Global relies heavily on high-
level community summaries and largely suffers from de-
tailed QA tasks, necessitating its GraphRAG Local variant
to balance the granularity trade-off. LightRAG struggles to
achieve competitive results, limited by its coarse-grained
key-value lookup mechanism. Regarding structurally aug-
mented methods, while LeanRAG (utilizing semantic aggre-
gation) and HippoRAG2 (leveraging phrase/passage nodes)
yield slight improvements in context recall, they fail to fully
break information isolation compared to our causal gating
mechanism. Finally, although CausalRAG occasionally at-
tains high Answer Relevancy due to its causal reasoning
capability, it struggles to scale to large datasets due to the
lack of efficient knowledge graph organization.
Holistic Comprehension vs. QA (RQ2).The contrast
between the results on HolisQA (Table 3) and standard
QA datasets (Table 4) is revealing. On popular QA bench-
marks, entity-centric methods like LightRAG, GraphRAG-
Local, LeanRAG could occasionally achieve good scores.
However, their performance degrades collectively and
significantly on HolisQA. A striking counterexample is
GraphRAG-Global: while its reliance on community sum-
maries hindered performance on granular standard QA tasks,
now it rebounds significantly in HolisQA. This discrepancy
strongly suggests that standard QA datasets, which often
favor short answers,implicitly reward the entity-centric
paradigm. In contrast, HolisQA, with its open-ended ques-
tions and dense logical structures,necessitates a compre-
hensive understandingof the underlying document—a sce-
nario closer to real-world applications. Notably, HugRAG is
the only framework that remains robust across this paradigm
shift, demonstrating competitive performance on both entity-
centric QA and holistic comprehension tasks.
Reconciling the Accuracy-Grounding Trade-off (RQ3).
HugRAG effectively reconciles the fundamental tension be-
tween Recall and Precision. While hierarchical causal gat-
ing expands traversal boundaries to secure superiorContext
Recall (Globality), the explicitcausal path identification
rigorously prunes spurious noise to maintain highF1 Score
and Answer Relevancy (Locality). This dual mechanism
allows HugRAG to simultaneously optimize for global cov-
erage and local groundedness, achieving a balance often
missed by prior methods.
5.3. Ablation Study
To addressRQ4, we ablate hierarchy, causal gates, and
causal path refinement components (see Figure 3), finding
that their combination yields optimal results. Specifically,
F1 CR AR
Metric010203040506070Score
26.854.7 55.7
24.058.0
53.6
23.360.2
52.6
30.155.460.0
36.860.064.1
38.660.467.4w/o H · w/o CG · w/o Causal
w/ H · w/o CG · w/o Causal
w/ H · w/ CG · w/o Causalw/o H · w/o CG · w/ Causal
w/ H · w/ CG · w/ Causal
w/ H · w/ CG · w/ SP-CausalFigure 3.Ablation Study.H: Hierarchical Structure; CG: Causal
Gates; Causal/SP-Causal: Standard vs. Spurious-Aware Causal
Identification. w/o and w/ denote exclusion or inclusion.
5K 10K 25K 100K 300K 750K 1M 1.5M
Source Text Length (chars)0102030405060Score
Naive
BM25
Standard RAGGraphRAG Global
GraphRAG Local
LightRAGHippoRAG2
LeanRAG
CausalRAG
HugRAG
Figure 4.Scalability analysis of HugRAG and other RAG baselines
across varying source text lengths (5K to 1.5M characters).
we observe a mutually reinforcing dynamic: while hier-
archical gates break information isolation to boost recall,
the spurious-aware causal identification is indispensable for
filtering the resulting noise and achieving a significant im-
provement. This mutual reinforcement allows HugRAG to
reconcile global coverage with local groundedness, signifi-
cantly outperforming any isolated component.
5.4. Scalability Analysis
Robustness to Information Scale (RQ5).To assess ro-
bustness against information overload, we evaluated per-
formance across varying source text lengths ( 5kto1.5M
characters) sampled from HolisQA, reporting the mean of
F1, Context Recall, and Answer Relevancy (see Figure 4).
As illustrated, HugRAG (red line) exhibits remarkable sta-
bility across all scales, maintaining high scores even at 1.5M
characters. This confirms that our hierarchical causal gating
structure effectively encapsulates complexity, enabling the
retrieval process to scale via causal gates without degrading
reasoning fidelity.
8

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
6. Conclusion
We introduced HugRAG to resolve information isolation and
spurious noise in graph-based RAG. By leveraging hierarchi-
cal causal gating and explicit identification, HugRAG recon-
ciles global context coverage with local evidence grounding.
Experiments confirm its superior performance not only in
standard QA but also in holistic comprehension, alongside
robust scalability to large knowledge bases. Additionally,
we introduced HolisQA to evaluate complex reasoning ca-
pabilities for RAG. We hope our findings contribute to the
ongoing development of RAG research.
Impact Statement
This paper presents work whose goal is to advance the field
of machine learning, specifically by improving the reliabil-
ity and interpretability of retrieval-augmented generation.
There are many potential societal consequences of our work,
none of which we feel must be specifically highlighted here.
References
Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu,
X., Majumder, R., McNamara, A., Mitra, B., Nguyen,
T., Rosenberg, M., Song, X., Stoica, A., Tiwary, S., and
Wang, T. MS MARCO: A Human Generated MAchine
Reading COmprehension Dataset, October 2018.
Dai, X., Guo, K., Lo, C.-H., Zeng, S., Ding, J., Luo, D.,
Mukherjee, S., and Tang, J. GraphGhost: Tracing Struc-
tures Behind Large Language Models, October 2025.
Dong, G., Jin, J., Li, X., Zhu, Y ., Dou, Z., and Wen, J.-
R. RAG-Critic: Leveraging Automated Critic-Guided
Agentic Workflow for Retrieval Augmented Generation.
Dong, J., Liu, Y ., Aloui, A., Tarokh, V ., and Carlson, D.
CARE: Turning LLMs Into Causal Reasoning Expert,
November 2025.
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody,
A., Truitt, S., and Larson, J. From Local to Global: A
Graph RAG Approach to Query-Focused Summarization,
April 2024.
Es, S., James, J., Espinosa Anke, L., and Schockaert, S.
RAGAs: Automated Evaluation of Retrieval Augmented
Generation. In Aletras, N. and De Clercq, O. (eds.),Pro-
ceedings of the 18th Conference of the European Chapter
of the Association for Computational Linguistics: System
Demonstrations, pp. 150–158, St. Julians, Malta, March
2024. Association for Computational Linguistics. doi:
10.18653/v1/2024.eacl-demo.16.
Fortunato, S. and Barth ´elemy, M. Resolution limit in
community detection.Proceedings of the NationalAcademy of Sciences, 104(1):36–41, January 2007. doi:
10.1073/pnas.0605965104.
Guo, Z., Xia, L., Yu, Y ., Ao, T., and Huang, C. LightRAG:
Simple and Fast Retrieval-Augmented Generation, Octo-
ber 2024.
Guti´errez, B. J., Shu, Y ., Qi, W., Zhou, S., and Su, Y . From
RAG to Memory: Non-Parametric Continual Learning
for Large Language Models, February 2025.
Ho, X., Nguyen, A.-K. D., Sugawara, S., and Aizawa, A.
Constructing A Multi-hop QA Dataset for Comprehen-
sive Evaluation of Reasoning Steps, November 2020.
Khatibi, E., Wang, Z., and Rahmani, A. M. CDF-
RAG: Causal Dynamic Feedback for Adaptive Retrieval-
Augmented Generation, April 2025.
Khot, T., Clark, P., Guerquin, M., Jansen, P., and Sabhar-
wal, A. QASC: A Dataset for Question Answering via
Sentence Composition, February 2020.
Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M.,
Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., De-
vlin, J., Lee, K., Toutanova, K., Jones, L., Kelcey, M.,
Chang, M.-W., Dai, A. M., Uszkoreit, J., Le, Q., and
Petrov, S. Natural questions: A benchmark for ques-
tion answering research.Transactions of the Association
for Computational Linguistics, 7:452–466, 2019. doi:
10.1162/tacl a00276.
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V .,
Goyal, N., K ¨uttler, H., Lewis, M., Yih, W.-t., Rockt ¨aschel,
T., Riedel, S., and Kiela, D. Retrieval-Augmented Gener-
ation for Knowledge-Intensive NLP Tasks, April 2021.
Liu, H., Wang, S., and Li, J. Knowledge Graph Retrieval-
Augmented Generation via GNN-Guided Prompting.
2025a.
Liu, H., Wang, Z., Chen, X., Li, Z., Xiong, F., Yu, Q., and
Zhang, W. HopRAG: Multi-Hop Reasoning for Logic-
Aware Retrieval-Augmented Generation, May 2025b.
Luo, H., Lin, Q., Feng, Y ., Kuang, Z., Song, M., Zhu, Y .,
and Tuan, L. A. HyperGraphRAG: Retrieval-Augmented
Generation via Hypergraph-Structured Knowledge Rep-
resentation.
Luo, H., Zhang, J., and Li, C. Causal Graphs Meet Thoughts:
Enhancing Complex Reasoning in Graph-Augmented
LLMs, March 2025.
Ma, J. Causal Inference with Large Language Model: A
Survey, September 2024.
9

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
Newman, M.Networks, volume 1. Oxford University
Press, October 2018. ISBN 978-0-19-880509-0. doi:
10.1093/oso/9780198805090.001.0001.
Priem, J., Piwowar, H., and Orr, R. OpenAlex: A fully-open
index of scholarly works, authors, venues, institutions,
and concepts. 2022.
Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD:
100,000+ Questions for Machine Comprehension of Text,
October 2016.
Ravuru, C., Sakhinana, S. S., and Runkana, V . Agentic
Retrieval-Augmented Generation for Time Series Analy-
sis, August 2024.
Reimers, N. and Gurevych, I. Sentence-BERT: Sentence
Embeddings using Siamese BERT-Networks. InPro-
ceedings of the 2019 Conference on Empirical Methods
in Natural Language Processing and the 9th Interna-
tional Joint Conference on Natural Language Processing
(EMNLP-IJCNLP), pp. 3980–3990, Hong Kong, China,
2019. Association for Computational Linguistics. doi:
10.18653/v1/D19-1410.
Saklad, R., Chadha, A., Pavlov, O., and Moraffah, R. Can
Large Language Models Infer Causal Relationships from
Real-World Text?, January 2026.
Traag, V ., Waltman, L., and van Eck, N. J. From Louvain
to Leiden: Guaranteeing well-connected communities.
Scientific Reports, 9(1):5233, March 2019. ISSN 2045-
2322. doi: 10.1038/s41598-019-41695-z.
Walker, C. and Ewetz, R. Explaining the Reasoning of Large
Language Models Using Attribution Graphs, December
2025.
Wang, N., Han, X., Singh, J., Ma, J., and Chaudhary, V .
CausalRAG: Integrating Causal Graphs into Retrieval-
Augmented Generation. In Che, W., Nabende, J., Shutova,
E., and Pilehvar, M. T. (eds.),Findings of the Association
for Computational Linguistics: ACL 2025, pp. 22680–
22693, Vienna, Austria, July 2025a. Association for Com-
putational Linguistics. ISBN 979-8-89176-256-5. doi:
10.18653/v1/2025.findings-acl.1165.
Wang, S., Chen, Z., Wang, P., Wei, Z., Tan, Z., Meng,
Y ., Shen, C., and Li, J. Separate the Wheat from the
Chaff: Winnowing Down Divergent Views in Retrieval
Augmented Generation, November 2025b.
Wang, X., Liu, Z., Han, J., and Deng, S. RAG4GFM:
Bridging Knowledge Gaps in Graph Foundation Models
through Graph Retrieval Augmented Generation.
Yang, Z., Qi, P., Zhang, S., Bengio, Y ., Cohen, W. W.,
Salakhutdinov, R., and Manning, C. D. HotpotQA: ADataset for Diverse, Explainable Multi-hop Question An-
swering, September 2018.
Zhang, Y ., Zhang, Y ., Gan, Y ., Yao, L., and Wang, C. Causal
Graph Discovery with Retrieval-Augmented Generation
based Large Language Models, June 2024.
Zhang, Y ., Wu, R., Cai, P., Wang, X., Yan, G., Mao, S.,
Wang, D., and Shi, B. LeanRAG: Knowledge-Graph-
Based Generation with Semantic Aggregation and Hier-
archical Retrieval, November 2025.
10

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
A. Prompts used in Online Retrieval and Reasoning
This section details the prompt engineering employed during the online retrieval phase of HugRAG. We rely on Large
Language Models to perform two critical reasoning tasks: identifying causal paths within the retrieved subgraph and
generating the final grounded answer.
A.1. Causal Path Identification
To address thelocal spurious noiseissue, we design a prompt that instructs the LLM to act as a “causality analyst.” The
model receives a linearized list of potential evidence (nodes and edges) and must select the subset that forms a coherent
causal chain.
Spurious-Aware Selection (Main Setting).Our primary prompt, illustrated in Figure 5, explicitly instructs the model to
differentiate between valid causal supports (output in precise ) and spurious associations (output in ctprecise ). By
forcing the model to articulate what isnotcausal (e.g., mere correlations or topical coincidence), we improve the precision
of the selected evidence.
Standard Selection (Ablation).To verify the effectiveness of spurious differentiation, we also use a simplified prompt
variant shown in Figure 6. This version only asks the model to identify valid causal items without explicitly labeling spurious
ones.
A.2. Final Answer Generation
Once the spurious-filtered support subgraph S⋆is obtained, it is passed to the generation module. The prompt shown in
Figure 7 is used to synthesize the final answer. Crucially, this prompt enforces strict grounding by instructing the model to
relyonlyon the provided evidence context, minimizing hallucination.
- - - R o l e - - -
Y o u  a r e  a  c a r e f u l  c a u s a l i t y  a n a l y s t  a c t i n g  a s  a  r e r a n k e r  f o r  r e t r i e v a l .

- - - G o a l - - -
G i v e n  a  q u e r y  a n d  a  l i s t  o f  c o n t e x t  i t e m s  ( s h o r t  I D  +  c o n t e n t ) ,  s e l e c t  t h e  m o s t  i m p o r t a n t  i t e m s  c o n s i s t i n g   t h e  c a u s a l  g r a p h  a n d  o u t p u t  t h e m  i n  " p r e c i s e " .
A l s o  o u t p u t  t h e  l e a s t  i m p o r t a n t  i t e m s  a s  t h e  s p u r i o u s  i n f o r m a t i o n  i n  " c t _ p r e c i s e " .
Y o u  M U S T :
-  U s e  o n l y  t h e  p r o v i d e d  i t e m s .
-  R a n k  ` p r e c i s e `  f r o m  m o s t  i m p o r t a n t  t o  l e a s t  i m p o r t a n t .
-  R a n k  ` c t _ p r e c i s e `  f r o m  l e a s t  i m p o r t a n t  t o  m o r e  i m p o r t a n t .
-  O u t p u t  J S O N  o n l y .  D o  n o t  a d d  m a r k d o w n .
-  U s e  t h e  s h o r t  I D s  e x a c t l y  a s  s h o w n .
-  D o  N O T  i n c l u d e  a n y  I D s  i n  ` p _ a n s w e r ` .

- - - I n p u t s - - -
Q u e r y :
{ q u e r y }
C o n t e x t  I t e m s  ( s h o r t  I D  |  c o n t e n t ) :
{ c o n t e x t _ t a b l e }

- - - O u t p u t  F o r m a t  ( J S O N ) - - -
{ {
   " p r e c i s e " :  [ " C 1 " ,  " N 2 " ,  " E 3 " ] ,
   " c t _ p r e c i s e " :  [ " T 7 " ,  " N 9 " ] ,
   " p _ a n s w e r " :  " c o n c i s e  d r a f t  a n s w e r "
} }

- - - C o n s t r a i n t s - - -
-  ` p r e c i s e `  l e n g t h :  a t  m o s t  { m a x _ p r e c i s e _ i t e m s }  i t e m s .
-  ` c t _ p r e c i s e `  l e n g t h :  a t  m o s t  { m a x _ c t _ p r e c i s e _ i t e m s }  i t e m s .
-  ` p _ a n s w e r `  l e n g t h :  a t  m o s t  { m a x _ a n s w e r _ w o r d s }  w o r d s .
Figure 5.Prompt for Causal Path Identification withSpurious Distinction(HugRAG Main Setting). The model is explicitly instructed to
segregate non-causal associations into a separate list to enhance reasoning precision.
11

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
- - - R o l e - - -
Y o u  a r e  a  c a r e f u l  c a u s a l i t y  a n a l y s t  a c t i n g  a s  a  r e r a n k e r  f o r  r e t r i e v a l .

- - - G o a l - - -
G i v e n  a  q u e r y  a n d  a  l i s t  o f  c o n t e x t  i t e m s  ( s h o r t  I D  +  c o n t e n t ) ,  s e l e c t  t h e  m o s t  i m p o r t a n t  i t e m s  t h a t  b e s t  s u p p o r t  a n s w e r i n g  t h e  q u e r y  a s  a  c a u s a l  g r a p h .
Y o u  M U S T :
-  U s e  o n l y  t h e  p r o v i d e d  i t e m s .
-  R a n k  t h e  ` p r e c i s e `  l i s t  f r o m  m o s t  i m p o r t a n t  t o  l e a s t  i m p o r t a n t .
-  O u t p u t  J S O N  o n l y .  D o  n o t  a d d  m a r k d o w n .
-  U s e  t h e  s h o r t  I D s  e x a c t l y  a s  s h o w n .
-  D o  N O T  i n c l u d e  a n y  I D s  i n  ` p _ a n s w e r ` .
-  I f  e v i d e n c e  i s  i n s u f fi c i e n t ,  s a y  s o  i n  ` p _ a n s w e r `  ( e . g . ,  " U n k n o w n " ) .

- - - I n p u t s - - -
Q u e r y :
{ q u e r y }

C o n t e x t  I t e m s  ( s h o r t  I D  |  c o n t e n t ) :
{ c o n t e x t _ t a b l e }

- - - O u t p u t  F o r m a t  ( J S O N ) - - -
{ {
   " p r e c i s e " :  [ " C 1 " ,  " N 2 " ,  " E 3 " ] ,
   " p _ a n s w e r " :  " c o n c i s e  d r a f t  a n s w e r "
} }

- - - C o n s t r a i n t s - - -
-  ` p r e c i s e `  l e n g t h :  a t  m o s t  { m a x _ p r e c i s e _ i t e m s }  i t e m s .
-  ` p _ a n s w e r `  l e n g t h :  a t  m o s t  { m a x _ a n s w e r _ w o r d s }  w o r d s .
Figure 6.Ablation Prompt: Causal Path Identificationwithoutdifferentiating spurious relationships. This baseline is used to assess the
contribution of the spurious filtering mechanism.
B. Algorithm Details of HugRAG
This section provides granular details on the offline graph construction process and the specific algorithms used during the
online retrieval phase, complementing the high-level description in Section 4.
B.1. Graph Construction
Entity Extraction and Deduplication.The base graph H0is constructed by processing text chunks using LLM. We
utilize the prompt shown in Appendix 8, adapted from (Edge et al., 2024), to extract entities and relations (see prompts in
Figure 8). Since raw extractions from different chunks inevitably contain duplicates (e.g., “J. Biden” vs. “Joe Biden”), we
employ a two-stage deduplication strategy. First, we perform surface-level canonicalization using fuzzy string matching.
Second, we use embedding similarity to identify semantically identical nodes, merging their textual descriptions and pooling
their supporting evidence edges.
Hierarchical Partitioning.We employ the Leiden algorithm (Traag et al., 2019) to maximize the modularity Qof the
partition. We recursively apply this partitioning to build bottom-up levels H1, . . . , H L, stopping when the summary of a
module fits within a single context window.
Causal Gates.The prompt we used to build causal gates is shown in Figure 9. Constructing causal gates via exhaustive
pairwise verification across all modules results in a quadratic time complexity O(N2), where Nis the total number of
modules. Consequently, as the hierarchy depth scales, this becomes computationally prohibitive for LLM-based verification.
To address this, we implement aTop-Down Hierarchical Pruningstrategy that constructs gates layer-by-layer, from the
coarsest semantic level ( HL) down to H1. The core intuition leveragesthe transitivity of causality: if a causal link is
established between two parent modules, it implicitly covers the causal flow between their respective sub-trees (see full
algorithm in Algorithm 2).
The pruning process follows three key rules:
12

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
- - - R o l e - - -
Y o u  a r e  a  h e l p f u l  a s s i s t a n t  a n s w e r i n g  t h e  u s e r ' s  q u e s t i o n .

- - - G o a l - - -
A n s w e r  t h e  q u e s t i o n  u s i n g  t h e  p r o v i d e d  e v i d e n c e  c o n t e x t .  A  d r a f t  a n s w e r  m a y  b e  p r o v i d e d ;  u s e  i t  o n l y  i f  i t  i s  s u p p o r t e d  b y  t h e  e v i d e n c e .

- - - E v i d e n c e  C o n t e x t - - -
{ r e p o r t _ c o n t e x t }

- - - D r a f t  A n s w e r  ( o p t i o n a l ) - - -
{ d r a f t _ a n s w e r }

- - - Q u e s t i o n - - -
{ q u e r y }

- - - A n s w e r  F o r m a t - - -
C o n c i s e ,  d i r e c t ,  a n d  n e u t r a l .
Figure 7.Prompt for Final Answer Generation. The model is conditioned solely on the filtered causal subgraph S⋆to ensure groundedness.
1.Layer-wise Traversal:We iterate from top (L) (usually sparse) to bottom (1) (usually dense).
2.Intra-layer Verification:We first identify causal connections between modules within the current layer.
3.Inter-layer Look-Ahead Pruning:When searching for connections between a module u(current layer) and modules
in the next lower layer (l−1), we prune the search space by:
• Excludingu’s own children (handled by hierarchical inclusion).
•Excluding children of modules already causally connected to u.Ifu→v is established, we assume the
high-level connection covers the relationship, skipping individual checks forChildren(v).
This strategy ensures that we only expend computational resources on discovering subtle, granular causal links that were not
captured at higher levels, effectivelyreducing the complexity from quadratic to near-linearin practice.
B.2. Online Retrieval
Hybrid Scoring and Diversity.To robustly anchor the query, our scoring function combines semantic and lexical signals:
sα(q, x) =α·cos(Enc(q),Enc(x)) + (1−α)·Lex(q, x),(5)
where Lex(q, x) computes the normalized token overlap between the query and the node’s textual attributes (title and
summary). We empirically set α= 0.7 to favor semantic matching while retaining keyword sensitivity for rare entities. To
ensure seed diversity, we apply Maximal Marginal Relevance (MMR) selection. Instead of simply taking the Top- K, we
iteratively select seeds that maximize sαwhile minimizing similarity to already selected seeds, ensuring the retrieval starts
from complementary viewpoints.
Edge Type Weights.In Equation (3), the weight function w(type(e)) controls the traversal behavior. We assign higher
weights to Causal Gates ( w= 1.2 ) and Hierarchical Links ( w= 1.0 ) to encourage the model to leverage the organized
structure, while assigning a lower weight to generic Structural Edges (w= 0.8) to suppress aimless local wandering.
B.3. Causal Path Reasoning
Graph Linearization Strategy.To reason over the subgraph Srawwithin the LLM’s context window, we employ a
linearization strategy that compresses heterogeneous graph evidence into a token-efficient format. Each evidence item
x∈S rawis mapped to a unique short identifier ID(x) . The LLM is provided with a compact list mapping these IDs to their
textual content (e.g., “N1: [Entity Description]”). This allows the model to perform selection by outputting a sequence of
valid identifiers (e.g., “[”N1”, ”R3”, ”N5”]”), minimizing token overhead.
13

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
Algorithm 2Top-Down Hierarchical Pruning for Causal Gates
Require:HierarchyH={H 0, H1, . . . , H L}
Ensure:Set of Causal GatesG c
1:G c← ∅
2:forl=Ldown to1do
3:for eachmoduleu∈H ldo
4:// 1. Intra-layer Verification
5:ConnectedPeers← ∅
6:forv∈H l\ {u}do
7:ifLLM Verify(u, v)then
8:G c.add((u, v))
9:ConnectedPeers.add(v)
10:end if
11:end for
12:// 2. Inter-layer Pruning (Look-Ahead)
13:ifl >1then
14:Candidates←H l−1
15:// Prune own children
16:Candidates←Candidates\Children(u)
17:// Prune children of connected parents
18:forv∈ConnectedPeersdo
19:Candidates←Candidates\Children(v)
20:end for
21:// Only verify remaining candidates
22:forw∈Candidatesdo
23:ifLLM Verify(u, w)then
24:G c.add((u, w))
25:end if
26:end for
27:end if
28:end for
29:end forreturnG c
Spurious-Aware Prompting.To mitigate noise, we design two variants of the selection prompt (in Appendix A.1):
•Standard Selection:The model is asked to output only the IDs of valid causal paths.
•Spurious-Aware Selection (Ours):The model is explicitly instructed to differentiate valid causal links from spurious
associations (e.g., coincidental co-occurrence) . By forcing the model to articulate (or internally tag) what isnotcausal,
this strategy improves the precision of the final output listS⋆.
In both cases, the output is directly parsed as the final set of evidence IDs to be retained for generation.
C. Visualization of HugRAG’s Hierarchical Knowledge Graph
To provide an intuitive demonstration of HugRAG’s structural advantages, we present 3D visualizations of the constructed
knowledge graphs for two datasets:HotpotQA(see Figure 11) andHolisQA-Biology(see Figure 10). In these visualizations,
nodes and modules are arranged in vertical hierarchical layers. The base layer ( H0), consisting of fine-grained entity nodes,
is depicted ingrey. The higher-level semantic modules ( H1toH4) are colored by their respective hierarchy levels. Crucially,
theCausal Gates—which bridge topologically distant modules—are rendered asred links. To ensure visual clarity and
prevent edge occlusion in this dense representation, we downsampled the causal gates, displaying only a representative
subset (r= 0.2).
14

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
- G o a l - 
G i v e n  a  t e x t  d o c u m e n t  t h a t  i s  p o t e n t i a l l y  r e l e v a n t  t o  t h i s  a c t i v i t y  a n d  a  l i s t  o f  e n t i t y  t y p e s ,  i d e n t i f y  a l l  e n t i t i e s  o f  t h o s e  t y p e s  f r o m  t h e  t e x t  a n d  a l l  r e l a t i o n s h i p s  a m o n g  
t h e  i d e n t i fi e d  e n t i t i e s . 
 
- S t e p s - 
1 .  I d e n t i f y  a l l  e n t i t i e s .  F o r  e a c h  i d e n t i fi e d  e n t i t y ,  e x t r a c t  t h e  f o l l o w i n g  i n f o r m a t i o n : 
-  e n t i t y _ n a m e :  N a m e  o f  t h e  e n t i t y ,  c a p i t a l i z e d 
-  e n t i t y _ t y p e :  O n e  o f  t h e  f o l l o w i n g  t y p e s :  [ { e n t i t y _ t y p e s } ] 
-  e n t i t y _ d e s c r i p t i o n :  C o m p r e h e n s i v e  d e s c r i p t i o n  o f  t h e  e n t i t y ' s  a t t r i b u t e s  a n d  a c t i v i t i e s 
F o r m a t  e a c h  e n t i t y  a s  ( " e n t i t y " { t u p l e _ d e l i m i t e r } < e n t i t y _ n a m e > { t u p l e _ d e l i m i t e r } < e n t i t y _ t y p e > { t u p l e _ d e l i m i t e r } < e n t i t y _ d e s c r i p t i o n > ) 
 
2 .  F r o m  t h e  e n t i t i e s  i d e n t i fi e d  i n  s t e p  1 ,  i d e n t i f y  a l l  p a i r s  o f  ( s o u r c e _ e n t i t y ,  t a r g e t _ e n t i t y )  t h a t  a r e  * c l e a r l y  r e l a t e d *  t o  e a c h  o t h e r . 
F o r  e a c h  p a i r  o f  r e l a t e d  e n t i t i e s ,  e x t r a c t  t h e  f o l l o w i n g  i n f o r m a t i o n : 
-  s o u r c e _ e n t i t y :  n a m e  o f  t h e  s o u r c e  e n t i t y ,  a s  i d e n t i fi e d  i n  s t e p  1 
-  t a r g e t _ e n t i t y :  n a m e  o f  t h e  t a r g e t  e n t i t y ,  a s  i d e n t i fi e d  i n  s t e p  1 
-  r e l a t i o n s h i p _ d e s c r i p t i o n :  e x p l a n a t i o n  a s  t o  w h y  y o u  t h i n k  t h e  s o u r c e  e n t i t y  a n d  t h e  t a r g e t  e n t i t y  a r e  r e l a t e d  t o  e a c h  o t h e r 
-  r e l a t i o n s h i p _ s t r e n g t h :  a  n u m e r i c  s c o r e  i n d i c a t i n g  s t r e n g t h  o f  t h e  r e l a t i o n s h i p  b e t w e e n  t h e  s o u r c e  e n t i t y  a n d  t a r g e t  e n t i t y 
 F o r m a t  e a c h  r e l a t i o n s h i p  a s  ( " r e l a t i o n s h i p " { t u p l e _ d e l i m i t e r } < s o u r c e _ e n t i t y > { t u p l e _ d e l i m i t e r } < t a r g e t _ e n t i t y > { t u p l e _ d e l i m i t e r }
< r e l a t i o n s h i p _ d e s c r i p t i o n > { t u p l e _ d e l i m i t e r } < r e l a t i o n s h i p _ s t r e n g t h > ) 
 
3 .  R e t u r n  o u t p u t  i n  E n g l i s h  a s  a  s i n g l e  l i s t  o f  a l l  t h e  e n t i t i e s  a n d  r e l a t i o n s h i p s  i d e n t i fi e d  i n  s t e p s  1  a n d  2 .  U s e  * * { r e c o r d _ d e l i m i t e r } * *  a s  t h e  l i s t  d e l i m i t e r . 
 
4 .  W h e n  fi n i s h e d ,  o u t p u t  { c o m p l e t i o n _ d e l i m i t e r } 
 
# # # # # # # # # # # # # # # # # # # # # # 
- E x a m p l e s - 
E x a m p l e  1 : 
E n t i t y _ t y p e s :  O R G A N I Z A T I O N , P E R S O N 
T e x t : 
T h e  V e r d a n t i s ' s  C . . . . . . . . . . . . . . . . .
O u t p u t : 
( " e n t i t y " { t u p l e _ d e l i m i t e r } C E N T R A L  I N S T I T U T I O N { t u p l e _ d e l i m i t e r } O R G A N I Z A T I O N { t u p l e _ d e l i m i t e r } T h e  C e n t r a l  I n s t i t u t i o n  i s  t h e  F e d e r a l  R e s e r v e  o f  V e r d a n t i s ,  
w h i c h . . . . . . . . . . . . . . . . . .
E x a m p l e  2 :  . . . . .
E x a m p l e  3 :  . . . . .

# # # # # # # # # # # # # # # # # # # # # # 
- R e a l  D a t a - 
E n t i t y _ t y p e s :  { e n t i t y _ t y p e s } 
T e x t :  { i n p u t _ t e x t } 
# # # # # # # # # # # # # # # # # # # # # # 
O u t p u t :
Figure 8.Prompt for LLM-based Information Extraction (modified from GraphRAG (Edge et al., 2024)). Used in Step 1 of Offline
Construction.
D. Case Study: A Real Example of the HugRAG Full Pipeline
To concretely illustrate the HugRAG full pipeline, we present a step-by-step execution trace on a query from theHolisQA-
Biologydataset in Figure 12. The query asks for a comparison of specific enzyme activities (Apase vs. Pti-interacting kinase)
in oil palm genotypes under phosphorus limitation—a task requiring the holistic comprehension of biology knowledge in
HolisQA dataset.
E. Experiments on the Effectiveness of Causal Gates
To isolate the real effectiveness of the causal gate in HugRAG, we conduct a controlled A/B test comparing gold context
access with the gate disabled (off) versus enabled (on). The evaluation is performed on two datasets:NQ(Standard QA)
andHolisQA. We define “Gold Nodes” as the graph nodes mapping to the gold context. Metrics are computed only on
examples where gold nodes are mappable to the graph. While this section focuses on structural retrieval metrics, we evaluate
the downstream impact of causal gates on final answer quality in our ablation study in Section 5.3.
15

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
- G o a l - 
G i v e n  t w o  t e x t  s n i p p e t s  A  a n d  B ,  d e c i d e  w h e t h e r  t h e r e  i s  a n y  p l a u s i b l e  c a u s a l  r e l a t i o n s h i p  b e t w e e n  t h e m  ( e i t h e r  d i r e c t i o n )  u n d e r  s o m e  r e a s o n a b l e  c o n t e x t .

- S t e p s -
R e a d  A  a n d  B ,  a n d  c o n s i d e r  w h e t h e r  o n e  c o u l d  p l a u s i b l y  i n fl u e n c e  t h e  o t h e r  ( d i r e c t l y  o r  i n d i r e c t l y ) .
R e q u i r e  a  p l a u s i b l e  m e c h a n i s m ;  i g n o r e  m e r e  c o r r e l a t i o n  o r  c o - o c c u r r e n c e .
I f  u n c e r t a i n  o r  o n l y  a s s o c i a t i v e ,  c h o o s e  " n o " .

- O u t p u t - 
R e t u r n  e x a c t l y  o n e  t o k e n :  " y e s "  o r  " n o " .  N o  e x t r a  t e x t .

# # # # # # # # # # # # # # # # # # # # # # 
- R e a l  D a t a - 
A :  { a _ t e x t } 
B :  { b _ t e x t } 
# # # # # # # # # # # # # # # # # # # # # # 
O u t p u t :
Figure 9.Prompt for Binary Causal Gate Verification. Used to determine the existence of causal links between module summaries.
H4
H3
H2
H1
H0
Figure 10.A3D viewof theHierarchical Graph with Causal Gatesconstructed from HolisQA-biology dataset.
Metrics.We report four structural metrics to evaluate retrieval quality and efficiency. Shaded regions in Figure 13 denote
95% bootstrap confidence intervals.Reachability: The fraction of examples where at least one gold node is retrieved in the
subgraph.Weighted Reachability (Depth-Weighted): A distance-sensitive metric defined as DWR =1
1+min hops(0 if
unreachable), rewarding retrieval at smaller graph distances.Coverage: The average proportion of total gold nodes retrieved
per example.Min Hops: The mean shortest path length to gold nodes, computed on examples reachable in bothoffandon
settings.
As shown in Figure 13, enabling the causal gate yields distinct behaviors across datasets. On the more complex HolisQA
dataset, the gate provides a statistically significant improvement in reachability and coverage. This confirms that causal
edges effectively bridge structural gaps in the graph that are otherwise traversed inefficiently. The increase in Weighted
Reachability and decrease in min hops indicate that the gate not only findsmoreevidence but creates structuralshortcuts,
allowing the retrieval process to access evidence at shallower depths.
16

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
H4
H3
H2
H1
H0
Figure 11.A3D viewof theHierarchical Graph with Causal Gatesconstructed from HotpotQA dataset.
F. Evaluation Details
F.1. Detailed Graph Statistics
We provide the complete statistics for all knowledge graphs constructed in our experiments. Table 5 details the graph
structures for the five standard QA datasets, while Table 6 covers the five scientific domains within the HolisQA dataset.
Table 5.Graph Statistics for Standard QA Datasets.Detailed breakdown of nodes, edges, and hierarchical module distribution.
Dataset Nodes Edges L3 L2 L1 L0 Modules Domain Chars
HotpotQA 20,354 15,789 27 1,344 891 97 2,359 Wikipedia 2,855,481
MS MARCO 3,403 3,107 2 159 230 55 446 Web 1,557,990
NQ 5,579 4,349 2 209 244 50 505 Wikipedia 767,509
QASC 77 39 - - - 4 4Science 58,455
2WikiMultiHop 10,995 8,489 8 461 541 78 1,088 Wikipedia 1,756,619
Table 6.Graph Statistics for HolisQA Datasets.Graph structures constructed from dense academic papers across five scientific domains.
Dataset Nodes Edges L3 L2 L1 L0 Modules Domain Chars
Holis-Biology 1,714 1,722 - 30 104 31 165 Biology 1,707,489
Holis-Business 2,169 2,392 8 77 166 41 292 Business 1,671,718
Holis-CompSci 1,670 1,667 7 28 91 30 158 CompSci 1,657,390
Holis-Medicine 1,930 2,124 7 56 129 34 226 Medicine 1,706,211
Holis-Psychology 2,019 1,990 5 45 126 35 211 Psychology 1,751,389
F.2. HolisQA Dataset
We introduceHolisQA, a comprehensive dataset designed to evaluate the holistic comprehension capabilities of RAG
systems, explicitly addressing the ”node finding” bias prevalent in existing QA datasets—where retrieving a single entity
(e.g., a year or name) is often sufficient. Our goal is to enforceholistic comprehension, compelling models to synthesize
coherent evidence from multi-sentence contexts.
We collected high-quality scientific papers across multiple domains as our primary source (Priem et al., 2022), focusing
exclusively on recent publications (2025) to minimize parametric memorization by the LLM. The dataset spans five distinct
domains—Biology, Business, Computer Science, Medicine, and Psychology—to ensure domain robustness (see full statistics
in Table 6). To necessitate cross-sentence reasoning, we avoid random sentence sampling; instead, we extract contiguous
textslicesfrom papers within each domain. Each slice is sufficiently long to encapsulate multiple interacting claims (e.g.,
Problem →Method →Result) yet short enough to remain self-contained, thereby preserving the logical coherence and
contextual foundation required for complex reasoning. Subsequently, we employ a rigorous LLM-based generation pipeline
to create Question-Answer-Context triples, imposing two strict constraints (as detailed in Figure 14):
17

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
Q u e r y :
 H o w  d o e s  t h e  a c t i v i t y  o f  a c i d  p h o s p h a t a s e  ( A p a s e )  a n d  P t i - i n t e r a c t i n g  s e r i n e / t h r e o n i n e  k i n a s e  d i f f e r  i n  o i l  p a l m  g e n o t y p e s  u n d e r  p h o s p h o r u s  l i m i t a t i o n ,  a n d  w h a t  a r e  
t h e  i m p l i c a t i o n s  f o r  t h e i r  a d a p t a b i l i t y ? 
S e e d  S t a g e :
s e e d  ( m a t c h e d  v i a  s h o r t _ i d _ m a p ) :   [ T 2 ,  T 4 ,  T 6 ,  S P ,  C A T ,  E S ,  A D A ,  I N D O N E S I A . . . . ]
e . g . :
-  T 2 :  [ t e x t _ u n i t ,  s c o r e = 0 . 4 6 1 5 ]  o l s  o f  P E  d i r e c t i o n  a n d  i n t e n s i t y ,  c o n t e x t ‐ d e p e n d e n t  m i c r o b i a l  s t r a t e g i e s ,  a n d  t h e  s c a r c i t y  o f  l o n g ‐ t e r m  C  b a l a n c e  a s s e s s m e n t s . . . . . . . . . .
-  T 4 :  [ t e x t _ u n i t ,  s c o r e = 0 . 4 6 1 5 ]  a c t i v i t y  i n  P ‐ o p t i m u m  w a s  h i g h e r  t h a n  s t a r v a t i o n  a n d  d e fi c i e n c y  i n  l e a f  a n d  r o o t  t i s s u e s  i n  b o t h  g e n o t y p e s ,  w h e r e a s  P t i  s e r i n e / t . . . . . . .
P o s t  n - h o p  S u b g r a p h :
t o p _ s u b g r a p h _ n o d e s  ( b y  c o m b i n e d  s c o r e ) .  e . g . :
-  E : d c e 6 6 3 0 3 - 2 b 2 c - 4 7 2 f - a 9 6 4 - d a 0 b 5 5 2 9 8 1 7 d  |  S P  ( c o m b i n e d = 0 . 4 1 2 7 ) 
-  E : 3 4 5 e b 0 d 6 - 5 6 f b - 4 8 7 8 - a 0 5 c - 9 9 f 0 1 d 5 2 8 c d 8  |  C A T  ( c o m b i n e d = 0 . 3 8 3 2 ) 
s a m p l e _ s u b g r a p h _ e d g e s .  e . g . :  ( ‘ u p ’  m e a n s  l o w e r  l e v e l  m o d u l e  t o  h i g h e r  l e v e l  m o d u l e . )
-  s t r u c t : 7 - > 6 9 : u p  |  X i x i a  C o u n t y  C o r n u s  O f fi c i n a l i s  P a t h o g e n  S t u d y  i n  H e n a n  - >  H e b e i - L i a o n i n g - Y u n n a n  R e s e a r c h  C o r r i d o r  ( s t r u c t u r a l ) 
-  s t r u c t : 7 - > 6 9  |  H e b e i - L i a o n i n g - Y u n n a n  R e s e a r c h  C o r r i d o r  - >  X i x i a  C o u n t y  C o r n u s  O f fi c i n a l i s  P a t h o g e n  S t u d y  i n  H e n a n  ( s t r u c t u r a l ) 
-  s t r u c t : 7 - > 6 8 : u p  |  J e h o l  B i o t a  &  Y i x i a n  F o r m a t i o n  i n  N E  C h i n a  - >  H e b e i - L i a o n i n g - Y u n n a n  R e s e a r c h  C o r r i d o r  ( s t r u c t u r a l ) 

C a u s a l  L L M  o u t p u t :
I d e n t i f y i n g  c a u s a l  g r a p h  a n d  s p u r i o u s  i n f o r m a t i o n  ( F u l l  p r o m p t  a v a i l a b l e  i n  a p p e n d i x ) :
{ 
  " p r e c i s e " :  [ " T 1 " ,  " T 4 " ,  . . . . . . . . . ] , 
  " c t _ p r e c i s e " :  [ " T 7 " ,  " N 9 " ,  . . . . . . . . ] , 
  " p _ a n s w e r " :  " U n d e r  p h o s p h o r u s  l i m i t a t i o n  i n  o i l  p a l m ,  A p a s e  a c t i v i t y  i s  e x p e c t e d  t o  b e  i n d u c e d  ( P  d e p r i v a t i o n ) ,  . . . . . . . . . . . . . . . . . . . . " 
} 
A n s w e r  L L M  o u t p u t :
G e n e r a t e  fi n a l  a n s w e r  ( F u l l  p r o m p t  a v a i l a b l e  i n  a p p e n d i x ) :
-  A c i d  p h o s p h a t a s e  ( A p a s e ) :  I n  o i l  p a l m ,  A p a s e  a c t i v i t y  i n  l e a v e s  a n d  r o o t s  i s  h i g h e r  u n d e r  P  o p t i m u m  t h a n  u n d e r  s t a r v a t i o n / d e fi c i e n c y  i n  b o t h  g e n o t y p e s .  T h i s  i n d i c a t e s  
A p a s e  i s  n o t  u p r e g u l a t e d  b y  p h o s p h o r u s  l i m i t a t i o n  i n  t h e  s t u d i e d  c o n d i t i o n s ,  s u g g e s t i n g  i t  m a y  n o t  b e  t h e  m a i n  a d a p t i v e  r e s p o n s e  t o  P  d e p r i v a t i o n  h e r e . 
-  P t i - i n t e r a c t i n g  s e r i n e / t h r e o n i n e  k i n a s e :  A c t i v i t y  i s  h i g h e r  i n  p r o l i fi c  v e r s u s  n o n - p r o l i fi c  g e n o t y p e s  s p e c i fi c a l l y  u n d e r  P - d e fi c i e n t  c o n d i t i o n s ,  i n d i c a t i n g . . . . . . . . . . . . . . . . .
G o l d  A n s w e r : 
A p a s e  a c t i v i t y  w a s  h i g h e r  i n  P - o p t i m a l  c o n d i t i o n s  t h a n  i n  s t a r v a t i o n  o r  d e fi c i e n c y  f o r  b o t h  l e a f  a n d  r o o t  t i s s u e s  a c r o s s  b o t h  o i l  p a l m  g e n o t y p e s .  C o n v e r s e l y ,  P t i  s e r i n e /
t h r e o n i n e  k i n a s e  a c t i v i t y  w a s  h i g h e r  i n  p r o l i fi c  g e n o t y p e s  c o m p a r e d  t o  n o n - p r o l i fi c  o n e s  u n d e r  P - d e fi c i e n t  d o s a g e .  A d d i t i o n a l l y ,  a b s c i s i c  a c i d  c o n t e n t  w a s  h i g h e r  i n  
p r o l i fi c  g e n o t y p e s  d u r i n g  s t a r v a t i o n  a n d  d e fi c i e n c y .  T h e s e  fi n d i n g s  s u g g e s t  t h a t  t h e  p r o l i fi c  g e n o t y p e  i s  m o r e  a d a p t a b l e  t o  p h o s p h o r u s  d e fi c i e n c y ,  p o t e n t i a l l y . . . . . . . . . . . . . . . . . .
Figure 12.A real example of HugRAG on a biology-related query.The diagram visualizes the data flow from initial seed matching
and hierarchical graph expansion to the causal reasoning stage, where the model explicitly filters spurious nodes to produce a grounded,
high-fidelity answer.
1.Integration Constraint:The question must require integrating information from at least three distinct sentences. We
explicitly reject trivia-style questions that can be answered by a single named entity (e.g., ”Who founded X?”).
2.Evidence Verification:The generation process must output the IDs of all supporting sentences. We validate the dataset
via a necessity check, verifying that the correct answer cannot be derived if any of the cited sentences are removed.
Through this strict construction pipeline, HolisQA effectively evaluates the model’s ability to perform holistic comprehension
and isolate it from parametric knowledge, providing a cleaner signal for evaluating the effectiveness of structured retrieval
mechanisms.
F.3. Implementation
Backbone Models.We consistently use OpenAI’s gpt-5-nano with a temperature of 0.0 to ensure determinis-
tic generation. For vector embeddings, we employ the Sentence-BERT (Reimers & Gurevych, 2019) version of
all-MiniLM-L6-v2 with a dimensionality of 384. All evaluation metrics involving LLM-as-a-judge are implemented
using the Ragas framework (Es et al., 2024), with Gemini-2.5-Flash-Lite serving as the underlying evaluation
18

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
off on
Reachability0.70.80.9
off on
W . Reachability0.60.8
off on
Coverage0.20.4
off on
Min Hops0.51.01.5
HolisQA Dataset Standard QA Dataset
Figure 13.Experiments on Causal Gate effectiveness.We compare graph traversal performance with the causal gate disabled (off)
versus enabled (on). Shaded areas represent 95% bootstrap confidence intervals. The causal gate significantly improves evidence
accessibility (Reachability, Coverage) and traversal efficiency (lower Min Hops, higher Weighted Reachability).
Y o u  a r e  b u i l d i n g  a  r e a d i n g - c o m p r e h e n s i o n  d a t a s e t .
 
Y o u  w i l l  r e c e i v e  a  s l i c e  o f  s e n t e n c e s  f r o m  a  l o n g  d o c u m e n t .  E a c h  l i n e  s t a r t s  w i t h  a  s e n t e n c e  I D ,  a  t a b ,  t h e n  t h e  s e n t e n c e  t e x t .

G e n e r a t e  { q a s _ p e r _ r u n }  q u e s t i o n - a n s w e r  p a i r s  i n  J S O N  a r r a y  f o r m a t .  Q u e s t i o n s  m u s t  r e q u i r e  m u l t i - s e n t e n c e  r e a s o n i n g  a n d  a n  u n d e r s t a n d i n g  o f  t h e  o v e r a l l  s l i c e .  A v o i d  
s h o r t  f a c t u a l  q u e s t i o n s ,  n a m e d - e n t i t y  t r i v i a ,  o r  s i n g l e - s e n t e n c e  l o o k u p s .
 
E a c h  J S O N  i t e m  m u s t  i n c l u d e :
" q u e s t i o n " :  s t r i n g
" a n s w e r " :  s t r i n g  ( 2 - 4  s e n t e n c e s )
" c o n t e x t _ s e n t e n c e _ i d s " :  a r r a y  o f  { m i n _ c o n t e x t } - { m a x _ c o n t e x t }  I D s  d r a w n  o n l y  f r o m  t h e  p r o v i d e d  s l i c e 
R e t u r n  J S O N  o n l y ,  n o  e x t r a  t e x t .

S e n t e n c e s : 
{ s l i c e _ t e x t }
Figure 14.Prompt for generating the Holistic Comprehension Dataset (Question-Answer-Context Triplets) from academic papers.
engine.
Baseline Parameters.To ensure a fair comparison among all graph-based RAG methods, we utilize a unified root
knowledge graph (see Appendix B.1 for construction details). For the retrieval stage, we set a consistent initial k= 3 across
all baselines. Other parameters are kept at their default values to maintain a neutral comparison, with the exception of
method-specific configurations (e.g., global vs. local modes in GraphRAG) that are essential for the algorithm’s execution.
All experiments were conducted on a high-performance computing cluster managed by Slurm. Each evaluation task was
allocated uniform resources consisting of 2 CPU cores and 16 GB of RAM, utilizing 10-way job arrays for concurrent query
processing.
F.4. Grounding Metrics and Evaluation Prompts
We assess performance using two categories of metrics: (i) Lexical Overlap (F1 score), which measures surface-level
similarity between model outputs and gold answers; and (ii) LLM-as-judge metrics, specifically Context Recall and Answer
Relevancy, computed using a fixed evaluator model to ensure consistency (Es et al., 2024). To guarantee stable and fair
comparisons across baselines with varying retrieval outputs, we impose a uniform cap on the retrieved context length and the
number of items passed to the evaluator. The specific prompt template used for assessing Answer Relevancy is illustrated in
Figure 15.
19

HugRAG: Hierarchical Causal Knowledge Graph Design for RAG
# # #  C o r e  T e m p l a t e
{ i n s t r u c t i o n } 
P l e a s e  r e t u r n  t h e  o u t p u t  i n  a  J S O N  f o r m a t  t h a t  c o m p l i e s  w i t h  t h e  f o l l o w i n g  s c h e m a  a s  s p e c i fi e d  i n  J S O N  S c h e m a : 
{ o u t p u t _ s c h e m a } D o  n o t  u s e  s i n g l e  q u o t e s  i n  y o u r  r e s p o n s e  b u t  d o u b l e  q u o t e s , p r o p e r l y  e s c a p e d  w i t h  a  b a c k s l a s h . 

{ e x a m p l e s } 
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

N o w  p e r f o r m  t h e  s a m e  w i t h  t h e  f o l l o w i n g  i n p u t 
i n p u t :  { i n p u t _ j s o n } 
O u t p u t : 
 
# # #  A n s w e r  R e l e v a n c y  p r o m p t 
G e n e r a t e  a  q u e s t i o n  f o r  t h e  g i v e n  a n s w e r  a n d  i d e n t i f y  i f  t h e  a n s w e r  i s  n o n c o m m i t t a l . 
G i v e  n o n c o m m i t t a l  a s  1  i f  t h e  a n s w e r  i s  n o n c o m m i t t a l  ( e v a s i v e ,  v a g u e ,  o r  a m b i g u o u s )  a n d  0  i f  t h e  a n s w e r  i s  s u b s t a n t i v e . 
E x a m p l e s  o f  n o n c o m m i t t a l  a n s w e r s :  " I  d o n ' t  k n o w " ,  " I ' m  n o t  s u r e " ,  " I t  d e p e n d s " . 

# # #  E x a m p l e s
I n p u t :  { " r e s p o n s e " :  " A l b e r t  E i n s t e i n  w a s  b o r n  i n  G e r m a n y . " } 
O u t p u t :  { " q u e s t i o n " :  " W h e r e  w a s  A l b e r t  E i n s t e i n  b o r n ? " ,  " n o n c o m m i t t a l " :  0 } 

I n p u t :  { " r e s p o n s e " :  " T h e  c a p i t a l  o f  F r a n c e  i s  P a r i s ,  a  c i t y  k n o w n  f o r  i t s  a r c h i t e c t u r e  a n d  c u l t u r e . " } 
O u t p u t :  { " q u e s t i o n " :  " W h a t  i s  t h e  c a p i t a l  o f  F r a n c e ? " ,  " n o n c o m m i t t a l " :  0 } 

I n p u t :  { " r e s p o n s e " :  " I  d o n ' t  k n o w  a b o u t  t h e  g r o u n d b r e a k i n g  f e a t u r e  o f  t h e  s m a r t p h o n e  i n v e n t e d  i n  2 0 2 3  a s  I  a m  u n a w a r e  o f  i n f o r m a t i o n  b e y o n d  2 0 2 2 . " } 
O u t p u t :  { " q u e s t i o n " :  " W h a t  w a s  t h e  g r o u n d b r e a k i n g  f e a t u r e  o f  t h e  s m a r t p h o n e  i n v e n t e d  i n  2 0 2 3 ? " ,  " n o n c o m m i t t a l " :  1 } 
Figure 15.Example prompt used in RAGAS: Core Template and Answer Relevancy (Es et al., 2024).
20