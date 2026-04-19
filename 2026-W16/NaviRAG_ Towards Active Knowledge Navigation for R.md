# NaviRAG: Towards Active Knowledge Navigation for Retrieval-Augmented Generation

**Authors**: Jihao Dai, Dingjun Wu, Yuxuan Chen, Zheni Zeng, Yukun Yan, Zhenghao Liu, Maosong Sun

**Published**: 2026-04-14 14:07:01

**PDF URL**: [https://arxiv.org/pdf/2604.12766v1](https://arxiv.org/pdf/2604.12766v1)

## Abstract
Retrieval-augmented generation (RAG) typically relies on a flat retrieval paradigm that maps queries directly to static, isolated text segments. This approach struggles with more complex tasks that require the conditional retrieval and dynamic synthesis of information across different levels of granularity (e.g., from broad concepts to specific evidence). To bridge this gap, we introduce NaviRAG, a novel framework that shifts from passive segment retrieval to active knowledge navigation. NaviRAG first structures the knowledge documents into a hierarchical form, preserving semantic relationships from coarse-grained topics to fine-grained details. Leveraging this reorganized knowledge records, a large language model (LLM) agent actively navigates the records, iteratively identifying information gaps and retrieving relevant content from the most appropriate granularity level. Extensive experiments on long-document QA benchmarks show that NaviRAG consistently improves both retrieval recall and end-to-end answer performance over conventional RAG baselines. Ablation studies confirm performance gains stem from our method's capacity for multi-granular evidence localization and dynamic retrieval planning. We further discuss efficiency, applicable scenario, and future directions of our method, hoping to make RAG systems more intelligent and autonomous.

## Full Text


<!-- PDF content starts -->

NaviRAG: Towards Active Knowledge Navigation for Retrieval-Augmented
Generation
Jihao Dai1,2, Dingjun Wu1, Yuxuan Chen1, Zheni Zeng2*, Yukun Yan1*, Zhenghao Liu3, Maosong Sun1
1Tsinghua University,2Nanjing University,3Northeastern University
Abstract
Retrieval-augmented generation (RAG) typi-
cally relies on a flat retrieval paradigm that
maps queries directly to static, isolated text seg-
ments. This approach struggles with more com-
plex tasks that require the conditional retrieval
and dynamic synthesis of information across
different levels of granularity (e.g., from broad
concepts to specific evidence). To bridge this
gap, we introduce NaviRAG, a novel frame-
work that shifts from passive segment retrieval
to active knowledge navigation. NaviRAG first
structures the knowledge documents into a hi-
erarchical form, preserving semantic relation-
ships from coarse-grained topics to fine-grained
details. Leveraging this reorganized knowledge
records, a large language model (LLM) agent
actively navigates the records, iteratively iden-
tifying information gaps and retrieving relevant
content from the most appropriate granularity
level. Extensive experiments on long-document
QA benchmarks show that NaviRAG consis-
tently improves both retrieval recall and end-
to-end answer performance over conventional
RAG baselines. Ablation studies confirm per-
formance gains stem from our method’s capac-
ity for multi-granular evidence localization and
dynamic retrieval planning. We further discuss
efficiency, applicable scenario, and future di-
rections of our method, hoping to make RAG
systems more intelligent and autonomous.
1 Introduction
In recent years, Retrieval-Augmented Generation
(RAG) has been widely regarded as an effective
paradigm for improving the reliability and scal-
ability of large language models in knowledge-
intensive tasks (Lewis et al., 2020). Existing stud-
ies focus on cross-document retrieval and multi-
hop information integration over large-scale knowl-
edge corpora (e.g., HotpotQA (Yang et al., 2018),
*Corresponding author: yanyk.thu@gmail.com
Code available at: https://github.com/ZzzDJH/
NaviRAGMultiHopRAG (Xiong et al., 2020)), approximat-
ing reasoning as the localization and aggregation of
dispersed facts. However, such approaches primar-
ily emphasize retrieval difficulty while often over-
looking the complex semantic structures within
texts and their contextual constraints.
In complex long-chain reasoning scenarios (Fig-
ure 1), queries involve explicit contextual condi-
tions, with evidence distributed across different po-
sitions or semantic levels of a text, and exhibiting
significant variation in granularity. The relation-
ship between evidence and queries is not a sim-
ple one-to-one mapping; rather, it is jointly con-
strained by contextual semantics, thereby imposing
higher demands on retrieval mechanisms. Tradi-
tional vector-based retrieval methods relying on
fixed-granularity text chunking (Karpukhin et al.,
2020; Izacard and Grave, 2021; Liu et al., 2024)
face an inherent trade-off: fine-grained segments
facilitate local matching but lack sufficient context,
whereas coarse-grained segments provide more
complete information at the cost of introducing
noise and reducing matching precision. Although
some studies attempt to enhance information or-
ganization through structured representations such
as knowledge graphs, global modeling incurs high
computational costs, while local subgraph retrieval
struggles to ensure completeness of evidence re-
quired for complex reasoning.
Information Foraging Theory (Pirolli and Card,
1999) in cognitive science conceptualizes informa-
tion acquisition as a foraging process, suggesting
that “information foragers” make sequential deci-
sions to stay in or transition between information
sources based on information scent, thereby pro-
gressively approaching high-value information re-
gions. This theory highlights that efficient informa-
tion acquisition is not a one-shot retrieval process,
but a sequential exploration guided by local signals.
Inspired by this perspective, an ideal RAG system
should not treat evidence retrieval as a single match-arXiv:2604.12766v1  [cs.CL]  14 Apr 2026

Figure 1: Two types of complex long-chain reasoning scenarios, each illustrated with example queries, associated
retrieval challenges, limitations of traditional RAG, and how NaviRAG addresses these challenges.
ing step, but model it as a multi-stage, navigable,
and dynamic exploration process, which further
necessitates a structured and navigable semantic
organization of the knowledge base.
Based on these observations, we propose Navi-
RAG, a navigation-based retrieval-augmented gen-
eration framework for complex reasoning ques-
tion answering. NaviRAG integrates hierarchi-
cal knowledge organization with a staged retrieval
mechanism of “locate first, then forage.” At the
knowledge organization level, NaviRAG does not
rely on predefined document structures; instead, it
constructs a hierarchical semantic representation
grounded in traceable raw text chunks, progres-
sively expanding from global summaries to local
details. This design enables both navigability and
multi-granularity expressiveness in the information
space. During reasoning, NaviRAG models evi-
dence acquisition as a coarse-to-fine two-stage ex-
ploration process: it first locates relevant semantic
subspaces within the hierarchical knowledge base
via vector retrieval to define the “foraging scope”;
it then performs top-down, multi-step navigational
retrieval within this scope, dynamically deciding
whether to remain or further refine the search based
on information scent and query requirements. This
approach achieves efficient yet fine-grained adapta-
tion to queries of varying granularity, while better
satisfying context-sensitive retrieval needs.
We conduct systematic evaluations of NaviRAG
on multiple representative benchmarks for complex
reasoning question answering, covering diverse
text types and evidence distribution patterns. The
results demonstrate that, compared to mainstream
RAG methods, NaviRAG achieves significant over-
all performance improvements. Meanwhile, itsreasoning cost is comparable to structured meth-
ods based on local graph retrieval, and lower than
that of graph-based reasoning frameworks relying
on global structural aggregation. Further analysis
shows that the performance gains primarily stem
from the synergy between hierarchical semantic
organization and the staged navigation mechanism,
thereby validating the effectiveness of modeling
evidence acquisition as a multi-stage, navigable
process in complex reasoning scenarios.
2 Related Work
Retrieval-Augmented Generation (RAG) is a frame-
work that enhances the performance of large lan-
guage models on knowledge-intensive tasks by in-
corporating external knowledge bases (Lewis et al.,
2020; Guu et al., 2020; Izacard et al., 2022). The
core idea is to introduce retrievable non-parametric
knowledge into the generation process, enabling
models to access information beyond their train-
ing corpus during inference, thereby mitigating
issues such as outdated knowledge and factual er-
rors (Lewis et al., 2020; Izacard et al., 2022). In
a typical RAG system, source documents are seg-
mented into text chunks and indexed in a vector
space (Karpukhin et al., 2020; Izacard and Grave,
2021). During inference, relevant chunks are re-
trieved based on semantic similarity to the query
and provided as context to the generation model
(Karpukhin et al., 2020). In this way, RAG inte-
grates knowledge retrieval with text generation, al-
lowing language models to perform reasoning and
question answering with the support of external
knowledge (Lewis et al., 2020; Gao et al., 2023).
To address the limitation of traditional text
chunk-based retrieval methods in modeling cross-

Figure 2: Framework of NaviRAG under a two-stage paradigm of knowledge organization and navigational retrieval.
The offline stage organizes documents into a hierarchical structure through top-down semantic routing and bottom-up
refusion and summarization. The online stage retrieves evidence via semantic localization and subsequent navigation
within the identified semantic subtree.
chunk semantic relationships, recent studies have
explored the use of structured knowledge represen-
tations to improve information organization (Sarthi
et al., 2024; Edge et al., 2024; Li et al., 2024b).
These approaches typically organize textual con-
tent using hierarchical or graph-based structures,
enabling retrieval processes to aggregate and rea-
son across different semantic levels. For exam-
ple, some methods construct hierarchical semantic
structures via recursive clustering and summariza-
tion to support multi-granularity retrieval (Sarthi
et al., 2024; Yao et al., 2023). Others extract entity
relations from corpora to build knowledge graphs
(Edge et al., 2024; Guo et al., 2024; He et al., 2024;
Peng et al., 2025), supporting complex question
answering through graph traversal and community-
level information aggregation (GraphRAG (Edge
et al., 2024), LightRAG (Guo et al., 2024)). Ad-
ditionally, some work draws inspiration from cog-
nitive science (Gutiérrez et al., 2024, 2025; Wu
et al., 2025)„ modeling knowledge retrieval as an
associative activation process over semantic graphs
to enhance cross-evidence semantic connections
(HippoRAG (Gutiérrez et al., 2025)). While these
methods provide richer representational capacity
for information organization, constructing global
structures typically incurs substantial preprocess-
ing costs and may introduce additional computa-
tional overhead during inference.
Meanwhile, a series of new benchmark tasks
have emerged to evaluate language models’ capa-
bilities in complex reasoning scenarios (Pang et al.,2022; Bai et al., 2024; Li et al., 2024a). Unlike tra-
ditional multi-hop question answering (Yang et al.,
2018; Xiong et al., 2020), these tasks not only re-
quire identifying multiple relevant facts but also
demand understanding and integrating evidence
distributed across different parts of the text under
specific contextual constraints. For instance, Narra-
tiveQA (Ko ˇcisk`y et al., 2018) and QuALITY (Pang
et al., 2022) emphasize comprehension of long-
form narratives and document structures, while
benchmarks such as LongBench (Bai et al., 2024,
2025) and LooGLE (Li et al., 2024a) further in-
crease text scale and introduce diverse task formats,
where evidence is often distributed across different
positions or even different semantic levels within
documents. These studies suggest that, in com-
plex reasoning tasks, evidence acquisition relies
more on understanding contextual conditions rather
than merely retrieving semantically similar text
chunks. Therefore, single-step similarity-based re-
trieval mechanisms struggle to flexibly adapt to
varying evidence granularity requirements and to
progressively locate critical evidence in complex
contexts, which limits the performance of tradi-
tional RAG methods in such scenarios.
3 Method
3.1 Paradigm Definition
To characterize NaviRAG’s core mechanism, we
formalize the evidence acquisition process in com-
plex reasoning question answering as a two-stage
paradigm consisting of knowledge organization

and navigational retrieval(Figure 2). Given a doc-
ument collection Dand a query q, the process is
defined as:
H=F org(D),C=F nav(q,H)
Here, Forgconstructs a hierarchical knowl-
edge base Hwith continuous semantic granular-
ity, while Fnavperforms query-driven, staged ev-
idence acquisition over this space, producing a
multi-granularity context set C. Specifically, H
consists of semantic nodes, each corresponding to
a semantic unit, organized into a tree structure from
global to local granularity. This structure enables
partitioning of the knowledge space into multiple
semantic subtrees, providing navigable local se-
mantic regions for subsequent retrieval.
During the retrieval stage, Fnavmodels evi-
dence acquisition as a coarse-to-fine exploration
process. It first identifies relevant semantic regions
via vector retrieval to narrow down the search space.
Then, within the candidate subtrees, it performs top-
down navigational retrieval, dynamically deciding
whether to remain at the current level or further
refine the search across different semantic granular-
ities to satisfy contextual constraints of the query.
The final context set is represented as:
C=C vec∪Csum∪Craw
where Cvec,Csum, and Crawdenote the initially
retrieved segments, intermediate-level summaries,
and on-demand expanded raw text, respectively.
Based on this context, the generation model Gpro-
duces the answer:
y=G(q,C)
3.2 Hierarchical Knowledge Organization
In this section, we introduce the implementation of
the knowledge organization function Forg, whose
goal is to transform a document collection Dinto
a hierarchical knowledge base Hwith continuous
semantic granularity in an offline stage. In Navi-
RAG, each document is constructed as aKnowl-
edge Tree. Specifically, given a document D0, it
is first segmented into a set of text chunks S=
{s1, s2, ..., s n}, which serve as the basic semantic
units. A hierarchical structure is then progressively
built through an LLM-guided organization process.
All nodes follow a unified representation:title
serves as the semantic identifier,valuerepresents
the associated content (with intermediate nodescorresponding to sets of child nodes and leaf nodes
corresponding to text segments), and asummary
field is maintained to support cross-level access.
Knowledge Tree construction is achieved
through top-down structural generation and bottom-
up semantic abstraction. First, a high-level seman-
tic outline is generated from the document:
H0=Outline(D 0)
Subsequently, the structure is iteratively updated
by inserting text chunks:
Hi=U(H i−1, si), i= 1,2, . . . , n
The detailed procedure is illustrated in Algorithm 1
To ensure structural compactness, the insertion
process incorporates several constraints, including
restricting multi-path assignments, performing se-
mantic splitting when nodes become overly long,
and grouping nodes when the number of children
exceeds a predefined threshold. After the structure
is constructed, the system performs content refine-
ment and bottom-up summarization, yielding the
final knowledge base:
H=Refusion(Sum(H n))
This hierarchical structure clusters semantically
related content into localized regions within the
tree, thereby providing an operational semantic
space for subsequent navigational retrieval.
3.3 Navigational Retrieval over the
Knowledge Tree
NaviRAG models evidence acquisition as a staged
navigation process. First, vector retrieval over text
chunk setSto obtain a candidate set:
R=TopK(q, S)
These candidates are mapped to corresponding
nodes in the Knowledge Tree, yielding candidate
semantic subtrees Hsubthat constrain the subse-
quent search space. The system then performs
top-down navigational retrieval over Hsub, progres-
sively transitioning from coarse-grained semantic
regions to fine-grained evidence acquisition.
During the navigation stage, at each level the
system considers the current set of candidate nodes
Ntand selects a subset based on the queryq:
At=Select(q,N t)

Algorithm 1Segment Insertion for Knowledge
Tree
Require:Knowledge TreeH, segments i, current pathp
Ensure:Updated treeH
1:N←Children(H, p){candidate nodes at current level}
2:A←Select(s i, N) {select relevant titles (limited multi-
assignment)}
3:ifA=∅then
4:H ←CreateNode(H, p, s i)
5:returnH
6:end if
7:foreach noden∈Ado
8:ifnis an intermediate nodethen
9:H ←U(H, s i, n){recursive descent}
10:else
11:H ←MergeContent(H, n, s i)
12:ifLen tokens (n)> τ textthen
13:H ←SplitNode(H, n){node expansion}
14:end if
15:end if
16:end for
17:ifNumberOfNodes(level(p))> τ level then
18:H ←SoftGrouping(H,level(p)) {semantic regroup-
ing}
19:end if
20:returnH
For each node n∈ A t, the system determines
whether to absorb summary information at the cur-
rent level or to expand into its child nodes:
π(n)∈ {absorb,expand}
When a leaf node is reached, evidence is extracted
from the corresponding raw text. If no valid can-
didates remain at a given level (i.e., At=∅), the
corresponding branch is terminated.
The final context set is represented as:
C=C vec∪Csum∪Craw
where Cvecdenotes the initial retrieval results,
Csumthe summaries selected during navigation,
andCrawthe underlying raw text content. This de-
sign enables efficient acquisition of evidence across
multiple semantic granularities.
3.4 Memory-guided Navigational Retrieval
(Exploratory)
In the default NaviRAG framework, the navigation
process is solely driven by the query qand com-
pleted through a single top-down retrieval pass. In-
spired by recent note-centric adaptive retrieval that
incrementally accumulates and refines knowledge
(Wang et al., 2025), to enhance evidence integra-
tion across semantic regions in complex reasoning
scenarios, we introduce an optional memory-based
mechanism that maintains a dynamic state mtdur-
ing navigation to accumulate acquired information.Specifically, the context obtained at each step
ctis written into memory and consolidated via a
lightweight update to reduce redundancy:
mt+1=Update(m t, ct)
During node selection, memory is incorporated
with the query as a conditioning signal:
At=Select(q, m t,Nt)
This enables the system to dynamically adjust ex-
ploration paths and avoid redundant visits.
The memory is maintained only within a single
query session and is not used for long-term storage.
It does not alter the original retrieval pipeline, but
instead serves as an auxiliary signal to enhance
state awareness during the navigation process.
4 Experiments
4.1 Experimental Setup
4.1.1 Baselines
Method Setup.We consider two representative
categories of methods as baselines:
(1) flat vector-based retrieval RAG (vanilla
RAG), which constructs a vector index over fixed-
granularity text chunks and retrieves top- kcontexts
based on similarity.
(2) structure-enhanced RAG, which leverages
explicit structures (e.g., knowledge graphs) to im-
prove information organization and retrieval.
Specifically, we includeGraphRAG(Edge
et al., 2024),LightRAG(Guo et al., 2024), and
HippoRAG2(Gutiérrez et al., 2025) (introduced
in Related Work 2). All methods are reproduced
following their public implementations and default
configurations, and are evaluated under a unified
retrieval and generation setting for fair compari-
son. Detailed hyperparameters and implementation
specifics are provided in the appendix A.2.
Model Setup.To ensure fairness and repro-
ducibility, all methods are compared under a uni-
fied configuration unless otherwise specified.
In the knowledge organization stage, NaviRAG
employsQwen2.5-72B(Qwen et al., 2025) for
summarization and structure construction. For
structure-enhanced baselines involving summariza-
tion or structural building, the same model is used
to eliminate discrepancies in generation capability.
In downstream question answering evaluation,
we assess NaviRAG and vanilla RAG across mod-
els, includingQwen3-14B,Qwen3-32B,Qwen3-
30B-A3B(Yang et al., 2025), andLLaMA3.3-70B

(Grattafiori et al., 2024). Structure-enhanced base-
lines are evaluated onLLaMA3.3-70B, following
default configurations.
During retrieval, we adoptbge-m3(Chen et al.,
2024) as the embedding model. For vanilla RAG
and NaviRAG, we set top- k= 5 , while other meth-
ods retain default retrieval parameters.
4.1.2 Datasets
We evaluate NaviRAG on single-document ques-
tion answering benchmarks to assess its perfor-
mance in complex long-chain reasoning scenarios.
These tasks require locating and integrating dis-
persed evidence within long texts and performing
multi-step reasoning under contextual constraints.
The selected datasets cover narrative texts
(scripts), encyclopedic content (Wikipedia), and
domain-specific documents in finance, law,
academia, and government reports. We analyze
performance along two dimensions: text type and
evidence dependency span. Dataset statistics are
provided in Table 7. Specifically:
NarrativeQA(Ko ˇcisk`y et al., 2018) focuses
on narrative understanding under contextual con-
straints. We adopt the same subset as used in Hip-
poRAG2 for evaluation. We use token-levelF1to
measure generation quality and reportRecall@1
as an auxiliary indicator of retrieval coverage.
LooGLE(Li et al., 2024a) categorizes tasks
into short- and long-dependency subsets based on
evidence dependency span, corresponding to lo-
cal retrieval and cross-segment integration scenar-
ios. Considering computational cost, we sample
the first 501 QA pairs from the short-dependency
subset and use all QA pairs from the long-
dependency subset. Within the long-dependency
subset, texts are further categorized into script-style
and Wikipedia-style formats to analyze structural
differences. We follow the officialLLM-as-Judge
protocol to evaluate semantic equivalence accuracy.
LongBench-v2(Bai et al., 2025) provides com-
plex tasks over ultra-long documents across mul-
tiple domains that require global understanding.
In this work, we adopt its single-document subset
and include all QA pairs across diverse domains,
including academic, legal, financial, and govern-
ment reports, to evaluate model generalization in
vertical scenarios. As the task is formulated as
single-choice question answering, we report stan-
dardaccuracy.4.2 Main Results
We evaluate NaviRAG across benchmarks, with
main results presented in Table 1 and Table 2.
Overall, NaviRAG consistently outperforms
vanilla RAG across all datasets and model con-
figurations, and achieves the best performance on
complex long-chain reasoning tasks. Compared
to structure-enhanced methods, NaviRAG demon-
strates more stable advantages in scenarios requir-
ing cross-region evidence integration and reasoning
under complex contextual constraints.
From the perspective oftask types, the per-
formance gains are closely related to the span of
evidence dependency. In retrieval-oriented tasks
(LooGLE-short), where answers primarily rely on
a single local piece of evidence, the improvements
brought by NaviRAG are limited. In contrast, for
complex long-chain reasoning tasks that require
contextual understanding and multi-evidence inte-
gration, the advantages become more pronounced.
Specifically, onNarrativeQA, NaviRAG consis-
tently improves retrieval recall by approximately
5%across different model scales and achieves
nearly a5%gain in F1 on LLaMA3.3-70B. On
LooGLE-long, it delivers consistent improvements
across both the script and Wikipedia subsets. On
LongBench-v2, due to the increased task difficulty,
noticeable gains are primarily observed on larger
models (LLaMA3.3-70B and Qwen3-30B-A3B).
From the perspective ofmodel scaling(Table 2),
NaviRAG demonstrates consistent gains across dif-
ferent model sizes, with more pronounced improve-
ments observed on stronger models.
4.3 Ablation Study
To analyze the sources of performance gains in
NaviRAG, we conduct two groups of ablation stud-
ies by removing the navigation mechanism and the
hierarchical knowledge structure. The results are
summarized in Table 3. When navigation is re-
moved, the system performs a single-step retrieval
over all nodes in the knowledge base, losing the
ability to progressively localize relevant semantic
regions, which leads to a significant performance
drop. When the hierarchical structure is removed,
the navigation process degenerates into local filter-
ing over retrieved results, which also fails to yield
effective improvements.
These results indicate that neither component
alone is sufficient to support complex reasoning
tasks. The performance gains stem from the syn-

MethodNarrativeQA LoogleLongBench v2
F1 Recall Short Long-Script Long-Wikipedia
Vanilla 27.80 73.23 76.64 41.58 44.66 40.78
GraphRAG 30.17 85.34 66.46 44.85 42.48 38.83
LightRAG 28.9886.6967.46 37.38 36.81 34.95
HippoRAG2 27.88 74.5786.0241.74 44.66 40.78
NaviRAG32.6078.69 79.04 45.01 44.88 42.72
Table 1: Performance comparison between NaviRAG and baselines. Best results are in bold and second-best results
are underlined.
LLMNarrativeQA LoogleLongBench v2
F1 Recall Short Long-Script Long-Wikipedia
Vanilla
Qwen3-14B 25.76 73.23 75.44 34.42 38.34 33.98
Qwen3-32B 27.05 73.23 77.64 41.58 42.48 41.75
Qwen3-30B-A3B 26.31 73.23 76.84 43.30 46.84 42.72
Llama3.3-70B 27.80 73.23 76.64 41.58 44.66 40.78
NaviRAG
Qwen3-14B 28.10(+2.34)78.81(+5.58)76.24(+0.8)40.03(+5.61)40.95(+2.61)32.04(-1.94)
Qwen3-32B 28.92(+1.87)78.82(+5.59)77.84(+0.2)44.85(+3.27)43.79(+1.31)40.78(-0.97)
Qwen3-30B-A3B 27.43(+1.12)77.82(+4.59)79.44(+2.6)42.67(-0.63)49.23(+2.39)45.63(+2.91)
Llama3.3-70B 32.60(+4.80)78.69(+5.46)79.04(+2.4)45.01(+3.43)44.88(+0.22)42.72(+1.94)
Table 2: Performance comparison between Vanilla RAG and NaviRAG across different model sizes. Numbers in
parentheses indicate the absolute improvement over the corresponding Vanilla baseline.
ergy between the hierarchical structure and the nav-
igation mechanism: the former provides semantic
scope constraints, while the latter progressively lo-
cates fine-grained evidence within this scope.
5 Discussion
5.1 Memory Module Analysis
To analyze the role of cross-step information ac-
cumulation in navigational retrieval, we intro-
duce a lightweight memory module into NaviRAG.
The experimental results are summarized in Ta-
ble 4. Under the LooGLE-long benchmark with
LLaMA3.3-70B, the memory module brings con-
sistent performance improvements over the base
model, indicating its effectiveness in enhancing
cross-region evidence integration. Compared to
purely query-driven local navigation, the memory
module accumulates intermediate information and
provides a more global semantic reference for sub-
sequent decisions, thereby improving the direction-
ality of multi-step retrieval (see Tables 19 and 20
for case studies).
However, this mechanism also has certain lim-
itations. The memory update process relies on
summarizing and integrating node content, which
may introduce semantic uncertainty without origi-nal contextual structure, thereby affecting the qual-
ity of memory writing. This issue is more pro-
nounced in smaller models, potentially causing in-
formation drift or conflicts. In addition, the current
memory module only serves as an auxiliary signal
in decision-making and does not yet constitute an
explicit goal-oriented retrieval strategy. Overall,
while the memory module demonstrates the poten-
tial value of cross-step information accumulation,
more systematic memory mechanisms and globally
guided navigation strategies remain to be explored.
5.2 Efficiency Analysis
We evaluate the inference overhead of NaviRAG
from two perspectives: computational and context
efficiency (Table 5, Table 6, and Table 24).
Computational Efficiency.Taking NarrativeQA
as an example, NaviRAG incurs higher inference
time than vanilla RAG and HippoRAG2, but re-
mains more efficient than LightRAG and signif-
icantly outperforms GraphRAG (Table 5). This
suggests that navigation based on local semantic
scopes maintains high efficiency while avoiding
global aggregation and reasoning overhead.
In terms of knowledge base construction, Navi-
RAG achieves efficiency between LightRAG and

SettingNarrativeQA LoogleLongBench v2
F1 Recall Short Script Wikipedia
NaviRAG32.60 78.69 79.04 45.01 44.88 42.72
w/o reading 30.52 73.00 72.85 39.71 41.17 42.72
w/o knowledge base 27.69 59.09 77.64 40.03 43.13 40.78
Table 3: Ablation study of NaviRAG components.
Method Script Wikipedia
Vanilla 41.58 44.66
NaviRAG 45.01 44.88
w/ Memory45.48 47.27
Table 4: Impact of the memory mechanism on NaviRAG
performance.
Method F1 Tokens Time (s/query)
Vanilla 27.80 2560 1.01
HippoRAG2 27.882635 1.36
GraphRAG 30.17 7249 14.30
LightRAG 28.98 18896 4.27
NaviRAG32.603305 3.23
Table 5: Performance–efficiency comparison of RAG
methods. Time denotes average inference latency
per query. Bold and underline indicate the best and
second-best results among structured methods (exclud-
ing vanilla RAG).
GraphRAG. With batch processing optimization,
NaviRAG’s generation efficiency is second only to
HippoRAG2 (Table 6). Detailed analysis of batch
generation will be provided in the appendix C.2.
Context Efficiency.Among all methods that
achieve substantial performance gains, NaviRAG
demonstrates significantly better context efficiency.
We further compare NaviRAG and vanilla RAG
under different retrieval scales (top- k= 3, 5, 7, 9,
15)(Table 24).
The results show NaviRAG achieves stable per-
formance at top- k= 3, whereas vanilla RAG re-
quires top- k= 15 to reach or surpass this level
(with performance at top- k= 9 being comparable
to NaviRAG at top- k= 3). Under the same top- k,
NaviRAG increases context length by about 500–
800 tokens (chunk size 512), yet utilizes retrieved
content more effectively, demonstrating higher in-
formation efficiency.
5.3 Text-type Preference Analysis
In the LooGLE-long dataset, we divide samples
into two categories based on text source: script
and Wikipedia (Tables 21 and 22). The resultsMethod KB Construction Time (min)
HippoRAG240.58
LightRAG 61.93
GraphRAG 225.93
NaviRAG 99.95
NaviRAG (batch) 51.28
Table 6: Knowledge base construction time of different
methods.
show that NaviRAG achieves significantly larger
performance gains on script documents than on
Wikipedia. This difference mainly stems from the
distinct semantic structures of the two text types.
Script texts exhibit strong semantic continuity, with
tightly coupled contextual dependencies across seg-
ments, allowing navigational retrieval to progres-
sively follow the narrative context to locate relevant
information. In contrast, Wikipedia documents
are typically composed of relatively independent
sections with a more modular semantic structure,
where weaker cross-segment dependencies reduce
the benefits of multi-step navigation.
These findings suggest that navigational retrieval
is more suitable for texts with strong semantic con-
tinuity. For highly structured documents, how to
effectively incorporate native section structures re-
mains an open question for future research.
6 Conclusion
We propose NaviRAG, a navigational retrieval-
augmented generation framework for complex
long-chain reasoning. By structuring knowledge
hierarchically and formulating retrieval as a multi-
stage navigation process, NaviRAG enables effi-
cient and context-aware evidence localization. Ex-
periments across multiple benchmarks show con-
sistent and substantial improvements over existing
RAG methods, particularly on multi-hop reasoning
tasks. Overall, our results suggest that viewing re-
trieval as a navigable exploration process offers a
promising and generalizable paradigm for enhanc-
ing complex reasoning.

Limitations
Experiments show that NaviRAG significantly im-
proves RAG performance on complex long-chain
reasoning tasks through its hierarchical structure
and navigational retrieval mechanism. However,
a certain limitation still warrants attention. This
work focuses on complex reasoning tasks under
constrained semantic contexts; future work should
explore query understanding and information inte-
gration in multi-source settings.
References
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu,
Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao
Liu, Aohan Zeng, Lei Hou, and 1 others. 2024. Long-
bench: A bilingual, multitask benchmark for long
context understanding. InProceedings of the 62nd
annual meeting of the association for computational
linguistics (volume 1: Long papers), pages 3119–
3137.
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xi-
aozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei
Hou, Yuxiao Dong, and 1 others. 2025. Longbench
v2: Towards deeper understanding and reasoning
on realistic long-context multitasks. InProceedings
of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 3639–3664.
Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu
Lian, and Zheng Liu. 2024. Bge m3-embedding:
Multi-lingual, multi-functionality, multi-granularity
text embeddings through self-knowledge distillation.
arXiv preprint arXiv:2402.03216, 4(5).
Darren Edge, Ha Trinh, Newman Cheng, Joshua
Bradley, Alex Chao, Apurva Mody, Steven Truitt,
Dasha Metropolitansky, Robert Osazuwa Ness, and
Jonathan Larson. 2024. From local to global: A
graph rag approach to query-focused summarization.
arXiv preprint arXiv:2404.16130.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia,
Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen
Wang, Haofen Wang, and 1 others. 2023. Retrieval-
augmented generation for large language models: A
survey.arXiv preprint arXiv:2312.10997, 2(1):32.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,
Abhinav Pandey, Abhishek Kadian, Ahmad Al-
Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
Alex Vaughan, and 1 others. 2024. The llama 3 herd
of models.arXiv preprint arXiv:2407.21783.
Zirui Guo, Lianghao Xia, Yanhua Yu, Tian Ao, and
Chao Huang. 2024. Lightrag: Simple and fast
retrieval-augmented generation.arXiv preprint
arXiv:2410.05779, 2(3).Bernal J Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Ya-
sunaga, and Yu Su. 2024. Hipporag: Neurobiologi-
cally inspired long-term memory for large language
models.Advances in neural information processing
systems, 37:59532–59569.
Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi,
Sizhe Zhou, and Yu Su. 2025. From rag to memory:
Non-parametric continual learning for large language
models.arXiv preprint arXiv:2502.14802.
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasu-
pat, and Mingwei Chang. 2020. Retrieval augmented
language model pre-training. InInternational confer-
ence on machine learning, pages 3929–3938. PMLR.
Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla,
Thomas Laurent, Yann LeCun, Xavier Bresson, and
Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and ques-
tion answering.Advances in Neural Information
Processing Systems, 37:132876–132907.
Gautier Izacard and Edouard Grave. 2021. Leveraging
passage retrieval with generative models for open do-
main question answering. InProceedings of the 16th
conference of the european chapter of the association
for computational linguistics: main volume, pages
874–880.
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lu-
cas Hosseini, Fabio Petroni, Timo Schick, Jane
Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and
Edouard Grave. 2022. Few-shot learning with re-
trieval augmented language models.arXiv preprint
arXiv:2208.03299, 1(2):4.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 conference on empirical methods in natural
language processing (EMNLP), pages 6769–6781.
Tomáš Ko ˇcisk`y, Jonathan Schwarz, Phil Blunsom, Chris
Dyer, Karl Moritz Hermann, Gábor Melis, and Ed-
ward Grefenstette. 2018. The narrativeqa reading
comprehension challenge.Transactions of the Asso-
ciation for Computational Linguistics, 6:317–328.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan
Zhang. 2024a. Loogle: Can long-context language
models understand long contexts? InProceedings
of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers),
pages 16304–16333.

Zhuoqun Li, Xuanang Chen, Haiyang Yu, Hongyu
Lin, Yaojie Lu, Qiaoyu Tang, Fei Huang, Xian-
pei Han, Le Sun, and Yongbin Li. 2024b. Struc-
trag: Boosting knowledge intensive reasoning of llms
via inference-time hybrid information structurization.
arXiv preprint arXiv:2410.08815.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paran-
jape, Michele Bevilacqua, Fabio Petroni, and Percy
Liang. 2024. Lost in the middle: How language mod-
els use long contexts.Transactions of the association
for computational linguistics, 12:157–173.
Richard Yuanzhe Pang, Alicia Parrish, Nitish Joshi,
Nikita Nangia, Jason Phang, Angelica Chen, Vishakh
Padmakumar, Johnny Ma, Jana Thompson, He He,
and 1 others. 2022. Quality: Question answering
with long input texts, yes! InProceedings of the
2022 Conference of the North American Chapter of
the Association for Computational Linguistics: Hu-
man Language Technologies, pages 5336–5358.
Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo,
Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang
Tang. 2025. Graph retrieval-augmented generation:
A survey.ACM Transactions on Information Systems,
44(2):1–52.
Peter Pirolli and Stuart Card. 1999. Information forag-
ing.Psychological review, 106(4):643.
Qwen, :, An Yang, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin
Yang, Jiaxi Yang, Jingren Zhou, and 25 oth-
ers. 2025. Qwen2.5 technical report.Preprint,
arXiv:2412.15115.
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh
Khanna, Anna Goldie, and Christopher D Manning.
2024. Raptor: Recursive abstractive processing for
tree-organized retrieval. InThe Twelfth International
Conference on Learning Representations.
Ruobing Wang, Qingfei Zhao, Yukun Yan, Daren Zha,
Yuxuan Chen, Shi Yu, Zhenghao Liu, Yixuan Wang,
Shuo Wang, Xu Han, and 1 others. 2025. Deep-
note: Note-centric deep retrievalaugmented genera-
tion.Preprint.
Dingjun Wu, Yukun Yan, Zhenghao Liu, Zhiyuan Liu,
and Maosong Sun. 2025. Kg-infused rag: Augment-
ing corpus-based rag with external knowledge graphs.
arXiv preprint arXiv:2506.09542.
Wenhan Xiong, Xiang Lorraine Li, Srini Iyer, Jingfei
Du, Patrick Lewis, William Yang Wang, Yashar
Mehdad, Wen-tau Yih, Sebastian Riedel, Douwe
Kiela, and 1 others. 2020. Answering complex open-
domain questions with multi-hop dense retrieval.
arXiv preprint arXiv:2009.12756.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
Binyuan Hui, Bo Zheng, Bowen Yu, Chang
Gao, Chengen Huang, Chenxu Lv, and 1 others.2025. Qwen3 technical report.arXiv preprint
arXiv:2505.09388.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio,
William Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. 2018. Hotpotqa: A dataset for
diverse, explainable multi-hop question answering.
InProceedings of the 2018 conference on empiri-
cal methods in natural language processing, pages
2369–2380.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
2023. Tree of thoughts: Deliberate problem solving
with large language models.Advances in neural
information processing systems, 36:11809–11822.

A Implementation Details
A.1 Dataset Statistics
We provide detailed statistics of the datasets used in
our experiments, including the number of question-
answer pairs and corresponding documents for
each dataset, as well as the average document
length measured in tokens (see Table 7).
NarrativeQA Loogle LongBench v2
Short Script Wikipedia
# Queries 293 501 642 459 103
# Documents 10 24 80 60 103
Avg. Doc Length (k tokens) 59.56 23.84 41.26 22.12 77.92
Table 7: Statistics of evaluation datasets.
A.2 Baseline Configurations
We report the implementation details and param-
eter settings for all baselines. For each method,
we follow its official implementation and retain
default configurations whenever possible, without
modifying core algorithms unless explicitly stated.
To ensure fair comparison, we adopt a unified in-
ference procedure across all methods to accommo-
date dataset-specific answer formats (e.g., ranking
tasks in Loogle-long and multiple-choice settings
in LongBench-v2).
For structure-enhanced baselines, the key hyper-
parameters of GraphRAG and LightRAG are listed
in Table 8, while the configuration of HippoRAG2
is provided in Table 9. For NaviRAG, the main hy-
perparameters related to hierarchical construction
and navigation are summarized in Table 10. Specif-
ically,Num Select Titlescontrols the maximum
number of titles selected at each layer;Max Con-
tent Lengthlimits the length of each node;Max
Titles Numrestricts the number of child nodes;
andMax Context Tokensis used to prevent ex-
ceeding the model context limit without affecting
the retrieval process.
Hyperparameter GraphRAG LightRAG
Mode Local Local
Chunk Size (tokens) 1200 1200
Chunk Overlap (tokens) 100 100
Max Context (tokens) 8000 30000
Community Report Length 2000 –
Max Cluster Size 10 –
Entity Summary Tokens – 500
Table 8: Hyperparameter settings for GraphRAG and
LightRAG.Hyperparameter HippoRAG2
Num to Retrieve 5
Table 9: Hyperparameter setting for HippoRAG2.
Hyperparameters NaviRAG
Organization
Chunk Token Size 512
Overlap Rate 0.2
Num Select Titles 2
Max Content Length 1536
Max Titles Num 12
Retrieval
Topk 5
Max Context tokens 8192
Table 10: Hyperparameter settings for NaviRAG.
A.3 Prompt Templates
We provide the main prompt templates used in our
framework. Prompts for knowledge organization
are shown in black, while those for retrieval are
highlighted in cyan. The detailed formats are pro-
vided below.

Prompt Template: Selecting Relevant Titles from Text
You will receive a list of titles and a passage of text. Please follow these steps:
1. Read and understand the text carefully.
Note: a single text passage may contain multiple independent facts,
or partial information about a fact. These may correspond to different titles in the title
list.
2. Identify and select the titles from the list that match the text.
You must select no more than {select_num} titles.
3. Summarize the content from the text that corresponds to each selected title.
4. Each output line must follow the format:
"Index//Title//Summary".
5. If no title matches the content of the text, output "None".
Example output:
2//Production Growth//The city achieved an agricultural product processing industry output
value of 4.41 billion yuan, an increase of 11.7%. The total industrial output value
reached 38.66 billion yuan, up 17.4%.
5//Green Agriculture//The city added 3 agricultural products with the right to use the green
food label, and 12 products obtained pollution-free food certification.
Notes:
1. Output must strictly follow the specified format, with a line break only after the summary.
2. Only select titles that have substantial correspondence or clear relevance to the content;
otherwise, output "None".
3. Do not select more than {select_num} titles, and do not use variables or pronouns to
replace specific data or facts.
Title List: {outlines}
Text: {text}
Please respond directly, one line per match, with no extra commentary.
Table 11: Prompt template for selecting relevant titles from current layer (knowledge organization).

Prompt Template: Creating a New Node
You will receive a new piece of text along with the title list of its sibling nodes.
Based on the instructions below, generate a new structured title node and its corresponding
content:
1. Read and understand the text within the context of the given parent title,
identifying its key topic, object, domain, or factual focus under that thematic scope.
2. Based on the parent title’s scope, generate a clear and specific new title
for this text, which will serve as a first-level structural node.
3. The new title must be both semantically and literally distinct from all existing titles
in the provided list, and should reflect the main informational dimension of the text
as it relates to the parent topic.
4. Then, summarize the factual content that best corresponds to the new title.
Be concise and concrete. Do not use pronouns or vague references;
at the same time, retain key factual details whenever possible,
such as time, location, people, events, and numerical data
(e.g., amounts, ratios, statistics).
5. Output must follow this strict format:
-1//New Title//Summarized Content
Example Output:
-1//Production Growth//The city achieved an agricultural product processing industry output
value of 4.41 billion yuan, an increase of 11.7%. The total industrial output value
reached 38.66 billion yuan, up 17.4%.
Important Notes:
- Only generate one new title and its content.
- The new title must be structural, not generic terms like "situation", "issues", or "status".
- The new title must not duplicate or partially copy any existing title in the list.
- The summary must be fact-based and precise. Do not use terms like "its", "that item", or
other abstract references.
- Only break the line after the summarized content. Do not add any explanations or extra text.
Title List: {outlines}
Text: {text}
Parent title: {parent_title}
Please respond strictly in the format: -1//New Title//Summarized Content.
Do not add any explanations or other content.
Table 12: Prompt template for generating a new node from input segment (knowledge organization).

Prompt Template: Merging segment into an Existing Node
You will receive a topic keyword, an existing paragraph about that topic,
and a new supplementary text. Please complete the following task:
1. Understand the topic by combining the topic keyword and the existing content
to determine the specific focus of this topic.
2. Extract only the information from the supplementary content that is closely related to the
topic.
3. Without repeating the existing content, preserve as much factual detail
from the supplementary content as possible, especially information related to
time, location, people, events, and numerical data
(such as amounts, ratios, and statistics).
4. Rewrite the extracted information into a natural, coherent,
and well-structured paragraph.
Instructions:
- Do not include any explanations, headings, or commentary.
- Do not repeat the existing content.
- Output format: wrap the final supplementary paragraph within << and >>,
for example: <<This is the new paragraph>>.
- Do not output anything else beyond the wrapped content.
Topic keyword: {topic}
Existing content: {exist_content}
Supplementary content: {supply_content}
Please output the result directly.
Table 13: Prompt template for merging supplementary segment into an existing node (knowledge organization).

Prompt Template: Refusing Content into Coherent Wiki-Style Text
You will receive a topic title and a set of information snippets related to that topic.
Each snippet is in the format [index] text, indicating that the sentence was extracted
from the original document at the given index.
These snippets have already been selected for their relevance to the topic,
but they may be redundant, fragmented, or lacking coherence.
Your task is to process and reorganize them into a clean, coherent,
and well-structured wiki-style text.
Please follow these instructions when generating your output:
1. Ensure that all content revolves around the given title.
2. Eliminate redundancy and merge similar or repetitive information.
3. Organize the content into natural paragraphs,
each ideally consisting of 1 to 3 sentences.
4. At the end of each paragraph, cite all index numbers that contributed
to the content using this format: <1><3><5>.
If the paragraph is based on multiple snippets, include all relevant index numbers.
5. Ensure that all key information from the original snippets is preserved and covered.
Additionally, make sure that every snippet’s index is referenced in the final output,
and no original snippet is left out or ignored.
All snippet indexes should be included in one or more of the paragraphs,
even if the content needs to be merged to fit the context.
6. Output only the wiki content - do not include any other explanation,
notes, or formatting.
Example:
Title: Machine Learning
Input snippets:
[1] Machine learning is a method of artificial intelligence.
[2] It enables systems to learn from data.
[3] Machine learning relies on statistical techniques.
[4] It can be used in image recognition and speech processing.
Output:
Machine learning is a method of artificial intelligence that relies on statistical techniques
to enable systems to learn from data. <1><2><3>
It is commonly applied in areas such as image recognition and speech processing. <4>
---
Title: {title}
Input snippets:
{text}
Table 14: Prompt template for reorganizing and fusing multiple snippets into coherent wiki-style content (knowledge
organization).

Prompt Template: Generating a Summary for a Node
You will be given the content of a wiki node.
Please summarize it concisely and objectively,
clearly stating the main ideas and key information.
Requirements:
- The summary should be informative and between 100 and 200 words in length.
- Ensure that the summary captures the essential content without introducing new information.
Instructions:
- Output only the summary result.
- Do not include any explanation, reasoning, or additional text.
Input:
{text}
Table 15: Prompt template for generating a concise summary of a node (knowledge organization).

Prompt Template: Navigation-Based Retrieval in a Hierarchical Knowledge-Tree
Your task is to assist in answering a natural language question
by selecting relevant entries from a hierarchical Wiki knowledge base.
The Wiki is organized as a tree structure:
- Each node contains a list of entries (subtopics).
- Each entry may contain further sub-entries.
You are currently located at a specific node in this tree,
indicated by the given path. At each step, you are given a list
of entries at the current level. Each entry consists of a title and a summary.
Evaluate each entry based on its relevance to the question:
EXPLORE:
- The title suggests that its sub-entries may contain useful information.
- The summary is relevant but incomplete, requiring deeper exploration.
INFO:
- The summary already provides complete and useful information
for a specific aspect of the question.
- No further exploration is needed.
The Wiki is hierarchical:
- Higher-level nodes provide general summaries.
- Deeper nodes provide more detailed and specific information.
Use this structure to decide whether an entry should be used directly (INFO)
or explored further (EXPLORE).
---
Inputs:
- Question: A natural language question based on a specific article.
The Wiki is a structured reorganization of that article’s full content.
- Entries: A list of entries in the current node, each in the format:
id. title: summary
- id: index of the entry
- title: topic name
- summary: concise overview of the topic
- Path: the current position in the Wiki tree
---
Output Format:
For each selected entry, output one line:
Index//Title//Category
Category must be either INFO or EXPLORE.
Example Output:
1//Climate Policy//EXPLORE
3//Carbon Emissions Statistics//INFO
If no entries are relevant, output:
None
---
Important Constraints:
- Output only the final result (no explanations or reasoning).
- Assign exactly one category per selected entry.
- Do not include irrelevant or redundant entries.
---
Input:
Path: {path}
Entries: {entries}
Question: {question}
Please output only the result.
Table 16: Prompt template for navigation-based retrieval over a hierarchical knowledge base (retrieval module).

Prompt Template: Passage Selection at the Leaf Level
You are an information retrieval assistant.
Your task is to select the most relevant document passages
that can help answer the user’s question.
Each passage is labeled with a line number, such as "[0]", "[1]", etc.
Use these numbers to refer to passages in your output.
---
User Question:
{query}
Texts to be filtered:
{texts}
---
Instructions:
1. Return only the line numbers of passages that are relevant to the question.
2. If only one passage is relevant, output:
0//
3. If multiple passages are relevant, output:
0//2//5//
4. If none of the passages are relevant, output:
None
5. Do not include any explanation or commentary.
---
Example:
User Question:
What is the historical development of Apple?
Texts to be filtered:
[0] Apple Inc. was founded in 1976 by Steve Jobs and others.
[1] Oranges are rich in vitamin C and are a common fruit.
[2] The release of the iPhone significantly influenced the smartphone industry.
Output:
0//2//
---
Now, based on the rules and example above,
please output your filtering result.
Table 17: Prompt template for selecting relevant passages at the leaf level of the knowledge structure (retrieval
module).

Vanilla NaviRAG NaviRAG w/ Memory
Document Anti-Peruvian sentiment
Question How many times has Chile adopted anti_peru policies?
Gold Answer 3 times.
Response At least 5 times At least 7 times3 times
Table 18: Case study comparing responses from different methods. NaviRAG with memory produces the correct
answer.
Navigation Trajectories of NaviRAG without Memory
Subtree 1: Anti-Peruvian Sentiment Across South American Nations
• Role of Media and Politics in Fostering Anti-Peruvian Sentiment
–Historical Conflicts and Anti-Peruvian Sentiment
*Early Conflicts and Anti-Peruvian Sentiment
*Diplomatic Maneuvers and Media Bias
*Modern Diplomatic Tensions and Immigration Issues
–Strategic Concerns and Regional Power Dynamics
*Modern Incidents and Geopolitical Doctrines
• Bolivian Nationalist Movements and Anti-Peruvian Sentiment
Subtree 2: Geopolitical Implications of Anti-Peruvian Policies and Disputes
• Impact of Anti-Peruvian Policies on Regional Relations
–Chilean Anti-Peruvian Policies and Nationalism
• Geopolitical Rivalries and Territorial Disputes
–Economic and Military Policies Against Peru
–Diplomatic and Military Interactions in the Late 20th Century
–Historical Anti-Peruvian Sentiments in Bolivia
Subtree 3: Diplomatic and Strategic Dynamics in 20th Century South America
• Continuation of Tensions in the 20th Century
Table 19: Navigation Trajectories of NaviRAG w/o Memory
Navigation Trajectories of NaviRAG with Memory
Subtree 1: Anti-Peruvian Sentiment Across South American Nations
• Role of Media and Politics in Fostering Anti-Peruvian Sentiment
–Historical Conflicts and Anti-Peruvian Sentiment
–Strategic Concerns and Regional Power Dynamics
• Bolivian Nationalist Movements and Anti-Peruvian Sentiment
–Simón Bolívar’s Anti-Peruvian Strategies and Impact
Subtree 2: Geopolitical Implications of Anti-Peruvian Policies and Disputes
• Geopolitical Rivalries and Territorial Disputes
–Economic and Military Policies Against Peru
–Diplomatic and Military Interactions in the Late 20th Century
–Historical Anti-Peruvian Sentiments in Bolivia
–Anti-Peruvian Sentiment in Ecuador
• Bolívar’s Derogatory Views on Peruvians and Their Military Capabilities
• Peru as a Catalyst for Regional Discord
–Impact of Anti-Peruvian Policies on Regional Relations
Subtree 3: Diplomatic and Strategic Dynamics in 20th Century South America
• Continuation of Tensions in the 20th Century
Table 20: Navigation Trajectories of NaviRAG w/Memory

B Case Studies
B.1 Effect of Memory Mechanism
We analyze the impact of the memory mechanism
on navigational retrieval through a representative
case study. We compare three settings: vanilla
RAG, NaviRAG without memory, and NaviRAG
with memory, including both the final outputs and
their corresponding navigation trajectories.
As shown in the case(Tables 18, 19 and 20), with-
out memory-based global information integration,
both vanilla RAG and NaviRAG (w/o memory) ex-
hibit repeated counting of the same event, leading
to incorrect answers. This suggests that retrieval
and navigation relying solely on local semantic
signals struggle to maintain a consistent global se-
mantic state in multi-step reasoning.
In contrast, with the memory mechanism, the
model incrementally accumulates previously ac-
quired information during navigation, forming a
more coherent global representation. This enables
more targeted navigation decisions, avoids redun-
dant exploration, and significantly shortens the
search path. As a result, the system can more ef-
ficiently locate key evidence and produce correct
answers.
B.2 Analysis of Document Types
We further analyze how document structure affects
retrieval behavior by comparing two representative
text types from the Loogle-long dataset: script and
Wikipedia documents(Tables 21 and 22).
Script documents exhibit strong semantic con-
tinuity, where segments (e.g., scenes) form a co-
herent narrative flow with high interdependence.
In contrast, Wikipedia documents are organized
into explicit sections corresponding to relatively
independent topics, with weaker cross-section de-
pendencies. These structural differences lead to
distinct retrieval dynamics. For script-like texts,
the continuous semantic flow facilitates both hi-
erarchical organization and navigational explo-
ration, allowing NaviRAG to progressively refine
evidence localization. For Wikipedia documents,
however, the inherent modular structure reduces
cross-segment semantic cohesion, and the current
knowledge construction process does not explicitly
leverage such structural cues. Consequently, Navi-
RAG shows weaker performance on Wikipedia
compared to script-based documents.C Additional Experimental Results
C.1 Efficiency under Different Top-kSettings
We report the performance of NaviRAG and vanilla
RAG under different top- ksettings (Table 24). The
detailed results complement the analysis presented
in the main text and are included here for complete-
ness.
C.2 Batch-wise Generation Exploration
During the knowledge base generation for
LongBench-v2, we encountered an issue: when the
number of document slices becomes too large, the
semantic differences between consecutive slices
become too significant. As a result, new slices tend
to accumulate at the outermost layer of the tree,
repeatedly triggering clustering operations at the
same level. This causes the original nodes to grad-
ually shift deeper into the tree, leading to the loss
of the hierarchical granularity structure.
To address this, we introduced a batch-wise gen-
eration strategy, where subtrees are generated in-
dependently in each batch and then merged. The
default batch size is set to 250, based on the num-
ber of slices in the longest document in theNarra-
tiveQAdataset.
This strategy improves generation efficiency, es-
pecially in long documents, effectively preventing
"tree structure deformation."
We further conducted experiments across all
datasets with a batch size of 100, usingLLaMA3.3-
70Bfor inference (Tables 23). Results show
that batch-wise generation improves both gener-
ation efficiency and inference performance forNar-
rativeQAandLongBench-v2. However, on the
LooGLEdataset, where the text structure was ig-
nored, the results were less favorable.
Future work will focus on refining this strategy
by incorporating a paragraph-based semantic batch-
ing approach.
D Future Work
We outline several directions for extending Navi-
RAG.
Hybrid Retrieval Mechanism.One promising
direction is to unify vertical navigation with hori-
zontal structure-based retrieval. While NaviRAG
focuses on hierarchical exploration, introducing ex-
plicit cross-node connections may improve local
evidence localization without sacrificing multi-step
reasoning capability.

Structure-aware Knowledge Organization.
Another direction is to incorporate explicit doc-
ument structures (e.g., sections and headings) into
the knowledge organization process. Combining
semantic continuity with structural cues may lead
to more accurate and stable hierarchical representa-
tions, and enable more flexible retrieval strategies.
Furthermore, combining batch generation with se-
mantic structures could significantly enhance gener-
ation efficiency, enabling faster processing without
compromising accuracy.
Memory-driven Navigation.Finally, extending
the navigation process with memory- or state-aware
mechanisms may enable more effective global in-
formation integration. Such approaches could sup-
port more proactive retrieval behaviors and improve
evidence utilization in complex reasoning scenar-
ios.

Internal Structure of a Script Document
Category:Script
Document Title:three-thousand-years-of-longing-2022
Internal Structure
1. Scene: Plane
2. Scene: Istanbul airport
3. Scene: Hotel
4. Scene: Conference hall
5. Scene: Grand Bazaar
6. Scene: Hotel bathroom (Djinn appears)
7. Scene: Djinn story – Sheba
8. Scene: Djinn story – Gülten
Table 21: Internal Structure of a Script Document
Internal Structure of a Wikipedia Document
Category:Wikipedia
Document Title:2023 Turkey–Syria earthquake
Internal Structure
• 1. Tectonic setting
–1.1 Geology
–1.2 Seismicity
• 2. Earthquake sequence
–2.1 Mainshock (the first major earthquake)
–2.2 The Second Great Earthquake
–. . .
• . . .
• 7. International humanitarian efforts
–7.1 Countries
–7.2 Arab League
–. . .
• 8. Reactions
–8.1 Criticism of the Turkish government
–8.2 Disaster management
Table 22: Internal Structure of a Wikipedia Document

SettingNarrativeQA LoogleLongBench v2
F1 Recall Short Long-Script Long-Wikipedia
Default 32.60 78.69 79.04 45.01 44.88 42.72
BatchSize=100 33.54(+0.94)78.26(-0.43)76.84(-2.2)42.83(-2.18)42.70(-2.18)44.66(+1.94)
Table 23: Comparison of NaviRAG under the default setting and batch generation with batch size 100 across
different datasets.
k=3 k=5 k=7 k=9 k=15
Method F1 Tokens F1 Tokens F1 Tokens F1 Tokens F1 Tokens
Vanilla 25.06 1536 27.80 2560 29.75 3584 30.24 4608 31.28 7908
NaviRAG 30.57 2089 32.60 3305 33.75 4392 33.22 5422 34.27 8411
Table 24: Efficiency analysis under different retrieval depths ( k). We report F1 and the final context length (tokens).