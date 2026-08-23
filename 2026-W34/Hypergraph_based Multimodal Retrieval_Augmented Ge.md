# Hypergraph-based Multimodal Retrieval-Augmented Generation with Incremental Refinement

**Authors**: Shenao Chen, Yidan Xu, Xiangmin Han, Rundong Xue, Duanpo Wu, Yuhan Gao, Chenggang Yan, Yue Gao

**Published**: 2026-08-17 14:30:09

**PDF URL**: [https://arxiv.org/pdf/2608.16628v1](https://arxiv.org/pdf/2608.16628v1)

## Abstract
Modern Multimodal Retrieval-Augmented Generation (M-RAG) systems are fundamentally limited by the binary connectivity paradigm of traditional simple graphs, which fails to capture the intricate, high-order correlations among heterogeneous entities, such as the N-ary relationships between a visual chart, its scattered textual descriptions, and underlying numerical data. Furthermore, existing refinement strategies often rely on exhaustive, full-page reconstruction to align cross-modal information, leading to prohibitive computational redundancy and the introduction of contextual noise in long-form document processing. In this paper, we propose Hyper-M2RAG, a novel framework that redefines multimodal document retrieval through High-order Hypergraph Representation Learning. We first formalize the document structure as a Multimodal Hypergraph, utilizing hyperedges as unified semantic containers to encapsulate multi-way associations across text, images, and tables, thereby transcending point-to-point modeling. To mitigate semantic fragmentation caused by physical pagination, we introduce an Anchor-driven Incremental Refinement mechanism. Rather than performing a global sweep, our approach identifies boundary-crossing anchor nodes and reconstructs their local hyper-topology using one-hop neighborhood contexts. This targeted refinement effectively bridges cross-page knowledge gaps with minimal computational footprints. Extensive evaluations on multimodal benchmarking datasets demonstrate that Hyper-M2RAG significantly outperforms state-of-the-art methods in both retrieval precision and generation coherence. Our code is available at https://github.com/ShenAoChen2001/MMHRAG.

## Full Text


<!-- PDF content starts -->

Hypergraph-based Multimodal Retrieval-Augmented Generation
with Incremental Refinement
Shenao Chen∗
241080005@hdu.edu.cn
Hangzhou Dianzi University
Hangzhou, ChinaYidan Xu∗
yidanxu2024@163.com
Hangzhou Dianzi University
Hangzhou, ChinaXiangmin Han✉
hanxiangmin@bjut.edu.cn
Beijing University of Technology
Beijing, China
Rundong Xue
xuerundong2002@gmail.com
Xi’an Jiaotong University
Xi’an, ChinaDuanpo Wu
wuduanpo@hdu.edu.cn
Hangzhou Dianzi University
Hangzhou, ChinaYuhan Gao✉
yuhangao@hdu.edu.cn
Hangzhou Dianzi University
Hangzhou, China
Chenggang Yan
cgyan@hdu.edu.cn
Hangzhou Dianzi University
Hangzhou, ChinaYue Gao
gaoyue@tsinghua.edu.cn
Tsinghua University
Beijing, China
Abstract
Modern Multimodal Retrieval-Augmented Generation (M-RAG)
systems are fundamentally limited by the binary connectivity para-
digm of traditional simple graphs, which fails to capture the intri-
cate, high-order correlations among heterogeneous entities—such
as the N-ary relationships between a visual chart, its scattered
textual descriptions, and underlying numerical data. Furthermore,
existing refinement strategies often rely on exhaustive, full-page
reconstruction to align cross-modal information, leading to pro-
hibitive computational redundancy and the introduction of con-
textual noise in long-form document processing. In this paper, we
propose Hyper-M2RAG, a novel framework that redefines multi-
modal document retrieval through High-order Hypergraph Repre-
sentation Learning. We first formalize the document structure as
a Multimodal Hypergraph, utilizing hyperedges as unified seman-
tic containers to encapsulate multi-way associations across text,
images, and tables, thereby transcending point-to-point modeling.
To mitigate semantic fragmentation caused by physical pagination,
we introduce an Anchor-driven Incremental Refinement mecha-
nism. Rather than performing a global sweep, our approach surgi-
cally identifies boundary-crossing anchors—nodes and reconstructs
their local hyper-topology using one-hop neighborhood contexts.
This targeted refinement effectively bridges cross-page knowledge
gaps with minimal computational footprints. Extensive evaluations
on multimodal benchmarking datasets demonstrate that Hyper-
M2RAG significantly outperforms state-of-the-art methods in both
retrieval precision and generation coherence. Our code is accessible
at https://github.com/ShenAoChen2001/MMHRAG.
∗Both authors contributed equally to this research.
This work is licensed under a Creative Commons Attribution 4.0 International License.
MM ’26, Rio de Janeiro, Brazil.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2213-4/2026/11
https://doi.org/10.1145/3767308.3835434CCS Concepts
•Information systems →Document representation;Multime-
dia information systems;•Computing methodologies →Knowl-
edge representation and reasoning;Natural language generation.
Keywords
Hypergraph, Multimodal RAG, Document Understanding, Cross-
page Association, Incremental Refinement
ACM Reference Format:
Shenao Chen, Yidan Xu, Xiangmin Han, Rundong Xue, Duanpo Wu, Yuhan
Gao, Chenggang Yan, and Yue Gao. 2026. Hypergraph-based Multimodal
Retrieval-Augmented Generation with Incremental Refinement. InProceed-
ings of the 34th ACM International Conference on Multimedia (MM ’26),
November 10–14, 2026, Rio de Janeiro, Brazil.ACM, New York, NY, USA,
9 pages. https://doi.org/10.1145/3767308.3835434
1 Introduction
Document Question Answering (Doc-QA) [ 19,22] aims to generate
accurate answers grounded in document content. With the rapid
progress of Large Vision-Language Models (LVLMs), Doc-QA has
evolved from text-only understanding to multimodal reasoning
over long and structurally complex documents. In real-world sce-
narios such as academic papers, technical reports, and business
documents, critical evidence is often distributed across text, figures,
tables, formulas, and page layouts, and may span many pages. This
setting requires not only long-context modeling, but also faithful
preservation of cross-modal structure and document-level semantic
dependencies.
Retrieval-augmented generation (RAG) has become an effective
paradigm for document understanding by retrieving relevant evi-
dence before answer generation. Existing RAG frameworks, how-
ever, are still largely rooted in text-centric assumptions [ 9,23].
Even recent multimodal variants often remain limited in how they
represent document structure. A common strategy is to linearize
visual content into captions or OCR-derived text, and then per-
form retrieval over textual surrogates [ 4,26]. Although practical,
this design inevitably weakens the original spatial organization of
arXiv:2608.16628v1  [cs.AI]  17 Aug 2026

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil. Shenao Chen et al.
the document and discards part of the visual semantics embedded
in figures, tables, and layouts. As a result, evidence that depends
on visual context, layout proximity, or joint interpretation across
modalities can be overlooked during retrieval.
Beyond this general limitation, a more fundamental issue lies
in the relational assumption inherited by many graph-based or
knowledge-enhanced RAG systems. In text-centric settings, knowl-
edge is commonly modeled through binary relations, typically rep-
resented as triples of the form(head, relation, tail). This abstraction
is often adequate when semantic dependencies can be decomposed
into pairwise links between entities or text spans. However, multi-
modal documents exhibit a different structural nature. Key evidence
is frequently formed not by a single pairwise connection, but by the
joint interaction among multiple heterogeneous elements, such as a
figure region, its caption, a nearby paragraph, a table cell, and their
shared layout context. In such cases, the meaning emerges from a
group-wise association rather than from isolated binary edges.
This mismatch makes conventional graph modeling inherently
limited for multimodal RAG. When high-order multimodal depen-
dencies are decomposed into a set of pairwise edges, the original
evidence unit becomes fragmented. The system may preserve lo-
cal connections, yet lose the higher-level semantic coherence that
binds multiple elements into a single interpretable structure. For
example, a textual claim may only be fully supported when a chart,
its legend, and the surrounding explanatory paragraph are con-
sidered together; reducing this evidence to independent pairwise
links weakens structural fidelity and makes downstream reasoning
less reliable. Therefore, the challenge in multimodal RAG is not
merely to add more modalities into existing retrieval pipelines, but
to redesign the underlying relational structure from graph-based
pairwise modeling to hypergraph-based high-order organization.
Recent advances in document parsing pipelines [ 13], such as
MinerU2.5 [ 17], DeepSeekOCR [ 21], and PaddleOCR [ 2], make this
shift increasingly practical. These systems can extract high-fidelity
multimodal elements from complex PDFs, including text blocks,
figure regions, table structures, and layout coordinates, thereby
providing a stronger foundation for structured multimodal indexing.
This progress opens the door to a retrieval framework that operates
directly on rich document elements and their structural relations,
instead of relying on lossy textual abstractions alone.
Motivated by this gap, we ask the following question:how can we
extend RAG from graph-based pairwise retrieval to hypergraph-based
high-order multimodal reasoning, while keeping refinement efficient
for long documents?
To answer this question, we identify two core challenges.Chal-
lenge 1: Pairwise graph modeling is insufficient for high-
order multimodal evidence.Existing graph-based retrieval struc-
tures are designed for binary relations and therefore struggle to
preserve group-wise dependencies among text, figures, tables, and
layout context. This limitation leads to fragmented evidence repre-
sentation and weakens the retriever’s ability to discover visually
grounded or structurally coupled information.Challenge 2: Ef-
ficient refinement in long multimodal documents remains
difficult.In long documents, relevant evidence and entity relations
often span many pages. Existing refinement strategies frequently
revisit entire pages or large contexts to recover missing cross-pageconnections [ 10], which introduces substantial computational re-
dundancy, especially when multimodal tokens dominate the cost.
We proposeHyper-M2RAG, a hypergraph-based framework
for multimodal retrieval-augmented generation. The central idea
is to move the structural foundation of RAG from graphs to hy-
pergraphs. Hyper-M2RAG constructs a multimodal hypergraph in
which vertices represent textual, visual, and tabular entities, while
hyperedges connect multiple heterogeneous elements that jointly
express a semantic unit. This design allows the retriever to pre-
serve and exploit high-order multimodal dependencies in a unified
topological space, rather than approximating them through decom-
posed pairwise links. As a result, retrieval becomes natively aware
of textual semantics, visual evidence, and layout structure.
To further support long-document reasoning, Hyper-M2RAG
introduces ananchor-driven surgical refinementmechanism. Our
key observation is that entities recurring across pages often signal
incomplete or ambiguous structural associations. Instead of repro-
cessing entire pages, the system identifies these boundary-crossing
entities as anchors and selectively reconstructs their local one-hop
neighborhoods. By refining only the hyperedges around these an-
chors and merging the updates back into the global hypergraph,
our method concentrates computation on the structurally uncertain
regions that matter most. This makes refinement substantially more
efficient while preserving document-level reasoning capacity.
Our contributions are summarized as follows:
•From graph to hypergraph for multimodal RAG.We show
that the pairwise relational assumption underlying conventional
graph-based RAG is insufficient for multimodal documents, and
introduce a hypergraph formulation that naturally models high-
order dependencies among text, figures, tables, and layout ele-
ments.
•Native multimodal hypergraph indexing.We leverage visual
and layout information during document parsing to extract struc-
tured entities and relationships. These extracted textual represen-
tations are then organized into a unified hypergraph, supporting
retrieval over semantically enriched structural evidence.
•Anchor-driven surgical refinement.We propose a localized
refinement strategy that updates only the structurally ambiguous
regions around cross-page anchors, avoiding exhaustive page-
level reprocessing and improving efficiency in long-document
settings.
•Effective and efficient multimodal reasoning.Through ex-
tensive experiments, we show that Hyper-M2RAG improves mul-
timodal retrieval and downstream answer quality while achieving
a better efficiency profile on long and complex documents.
The remainder of this paper details the proposed method. Sec-
tion 4 formalizes Hyper-M2RAG and presents its multimodal hy-
pergraph construction and anchor-driven refinement mechanism.
2 Related Work
Traditional RAG typically relies on flattened chunk retrieval, which
struggles to capture long-range dependencies. To bridge this gap,
graph-enhanced approaches like GraphRAG [ 3] and LightRAG [ 8]
extract entity-centric relations to improve multi-hop reasoning.
However, these binary graph models are limited to pairwise associ-
ations. To model higher-order group semantics, Hypergraph-based

Hypergraph-based Multimodal Retrieval-Augmented Generation with Incremental Refinement MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil.
Figure 1: Overview of the Hyper-M2RAG framework. The system progresses through three core stages: (1) Multi-modal Data
Ingestion via PMUs; (2) Two-stage Hypergraph Construction featuring anchor-driven structural refinement; (3) Cognitive-
inspired Hybrid Retrieval. The comparative case (bottom right) highlights our advantage in cross-page factual grounding over
vanilla LLMs.
RAG (HyperRAG) [ 5] introduces hyperedges to connect multiple
vertices simultaneously. Recent variants like CogRAG[ 11] further
explore cognitive-inspired hypergraph structures to enhance adap-
tive reasoning. While these methods excel in text-based high-order
modeling, they remain largely "vision-blind" when faced with com-
plex multi-modal layouts.
Recent studies have begun extending RAG to multi-modal do-
mains, encompassing cross-modal alignment, specialized document
parsing, and multi-source retrieval strategies [ 27]. Systems such as
RAG-Anything [ 7], MegaRAG [ 10], and MHier-RAG [ 6] attempt to
retrieve interleaved visual and textual components from complex
layouts, while complementary approaches address query refor-
mulation [ 14], heterogeneous source aggregation [ 16], multi-turn
reasoning [ 12], and evidence fusion [ 25]. However, existing meth-
ods suffer from retrieval-modality decoupling: they either treat
images as secondary "post-retrieval supplements" or rely on lossy
text-only indices (e.g., captions) that collapse the original structural
layout. This leads to critical information decay and visual blind
spots, especially when evidence spans across multiple pages. Unlike
these approaches, our work pursues a native multi-modal indexing
paradigm that preserves high-order structural and visual integrity
within a unified hypergraph.
3 Task Description and Preliminaries
The objective of multi-modal long-context Document-based Ques-
tion Answering (Doc-QA) is to generate a comprehensive answerY𝑎𝑛𝑠based on a queryQand a documentDcontaining heteroge-
neous modalities. We formalize the generation process as:
𝑌𝑎𝑛𝑠=LLM𝑔𝑒𝑛(𝑦𝑖|𝑦<𝑖,𝑄,𝑆),(1)
where𝑆represents the Structured Evidence Package retrieved from
the document-indexed corpus.
In ourHyper-M2RAGframework, the document 𝐷is organized
into a multi-modal hypergraph 𝐺=(𝑉,𝐸) , where𝑉denotes a set
of normalized semantic entities vertices and 𝐸={𝐸𝑙𝑜𝑤∪𝐸ℎ𝑖𝑔ℎ∪
𝐸𝑟𝑒𝑓𝑖𝑛𝑒}represents the set of hyperedges capturing multi-order re-
lationships. Specifically, 𝐸𝑙𝑜𝑤and𝐸ℎ𝑖𝑔ℎcapture intra-page pairwise
and group associations, respectively, while 𝐸𝑟𝑒𝑓𝑖𝑛𝑒 encapsulates la-
tent cross-page dependencies reconstructed through anchor-driven
refinement. The retrieval component is defined as a mapping func-
tion:
𝑆=HyperRetriever(𝐺,𝑄),(2)
where HyperRetriever(·) executes a dual-path alignment across the
symbolic hypergraph and the vector space. This strategy ensures
that𝑆integrates both vertex-anchored entity evidence 𝐶𝑒𝑛𝑡and
hyperedge-directed relational context 𝐶𝑟𝑒𝑙, facilitating complex
reasoning over the high-order multi-modal substrate.
4 Methodology
The core workflow of Hyper-M2RAG is illustrated in Fig. 1, high-
lighting the transition from fragmented multi-modal pages to a
unified, high-order hypergraph index.

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil. Shenao Chen et al.
4.1 Multi-modal Data Ingestion
To better preserve heterogeneous document structures, we redefine
the primary unit of data ingestion for structured knowledge extrac-
tion. Unlike conventional RAG systems that treat documents as
linear text streams, we adopt the Page-level Multi-modal Unit (PMU)
as the fundamental atomic unit. Specifically, we utilize MinerU as a
high-fidelity engine to parse raw PDFs into a structured sequence
D={𝑝 1,𝑝2,...,𝑝𝑛}. The𝑖-th unit𝑝 𝑖is defined as:
𝑝𝑖={𝑇𝑜𝑐𝑟
𝑖,𝐼𝑝𝑎𝑔𝑒
𝑖,F𝑖,𝑙𝑜𝑐𝑎𝑙},(3)
where𝑇𝑜𝑐𝑟
𝑖is the aggregated textual content, 𝐼𝑝𝑎𝑔𝑒
𝑖is the global page
image, andF𝑖,𝑙𝑜𝑐𝑎𝑙 ={𝐼𝑓𝑖𝑔
𝑖,𝑗}𝑛𝑖
𝑗=1represents localized image patches
(e.g., figures, tables). This representation assigns distinct semantic
roles to visual inputs: 𝐼𝑝𝑎𝑔𝑒
𝑖provides macro-visual context for layout
awareness, whileF𝑖,𝑙𝑜𝑐𝑎𝑙 offers fine-grained evidence from detected
regions. To ensure robust ingestion, vision-dominant pages with
sparse text are preserved via semantic placeholders (e.g.,“Please see
the Figures”). Furthermore, this unified schema treats visual fields
as optional (∅), allowing the system to handle both text-only and
multi-modal data through a consistent interface.
4.2 Multi-modal Hypergraph Index
Construction
The construction of the HyperIndex follows a hierarchical paradigm:
Initial Page-level Extractionfollowed byAnchor-driven Struc-
tural Refinement. This process transforms fragmented page-level
units into a cohesive, high-fidelity semantic network.
4.2.1 Initial Page-level Indexing and Aggregation.For each PMU
𝑝𝑖∈D, we first execute a multi-modal extraction operator LLM𝑒𝑥𝑡
to capture the localized entity set 𝑉𝑖and their corresponding high-
order associations 𝐸𝑖. Unlike traditional text-based extraction, this
process simultaneously digests the textual substrate and hierarchi-
cal visual evidence. The extraction is formalized as follows:
 
𝑉𝑖=LLM𝑒𝑥𝑡(𝑃𝑒𝑛𝑡(𝑝𝑖))
𝐸𝑖,𝑙𝑜𝑤=LLM𝑒𝑥𝑡(𝑃𝑙𝑜𝑤(𝑝𝑖,𝑉𝑖))
𝐸𝑖,ℎ𝑖𝑔ℎ=LLM𝑒𝑥𝑡(𝑃ℎ𝑖𝑔ℎ(𝑝𝑖,𝑉𝑖))for𝑝𝑖∈D,(4)
where𝑃𝑒𝑛𝑡,𝑃𝑙𝑜𝑤, and𝑃ℎ𝑖𝑔ℎare specialized prompts (detailed in
Appendix) for identifying vertices, pairwise relations, and high-
order correlations, respectively. Each extracted element is tagged
with a provenance attribute src={𝑖} to anchor semantic entities
to their original page index𝑖.
To synthesize a document-level structure, a Canonical Merging
operatorMis applied to aggregate page-level sets. Vertices are
normalized via case-insensitive alignment, while hyperedges are
collapsed based on their unique, sorted constituent vertex sets:
G𝑖𝑛𝑖𝑡=M 𝑛Ø
𝑖=1{𝑉𝑖,𝐸𝑖,𝑙𝑜𝑤,𝐸𝑖,ℎ𝑖𝑔ℎ}!
,(5)
whereG𝑖𝑛𝑖𝑡denotes the initial global hypergraph. In this stage,
cross-page alignment is achieved through entity unification, where
each global node or edge maintains an aggregated provenance index
src𝑔𝑙𝑜𝑏𝑎𝑙 =Ðsrc𝑖.4.2.2 Anchor-driven Structural Refinement.While G𝑖𝑛𝑖𝑡aggregates
local evidence, it remains constrained by the page-level receptive
field, often failing to capture long-range semantic dependencies that
transcend individual page boundaries. To bridge these disparate
fragments, we introduce a refinement stage centered on Cross-
page Anchors 𝑉𝑎𝑛𝑐. These anchors are identified as entities with
high provenance frequency, representing consistent semantic pivots
across the document:
𝑉𝑎𝑛𝑐={𝑣∈𝑉||src(𝑣)|≥𝜏},(6)
where𝜏is the frequency threshold.
Instead of re-processing entire page sequences with high com-
putational redundancy as in MegaRAG, our approach performs
targeted refinement within the topological neighborhoods of 𝑉𝑎𝑛𝑐.
For each anchor 𝑣∈𝑉𝑎𝑛𝑐, the system extracts its star-expansion
subgraph𝐺𝑠𝑢𝑏(𝑣)=(𝑉𝑠𝑢𝑏,𝐸𝑠𝑢𝑏)to encapsulate the multi-source
contexts linked to this pivot through shared hyperedges:
(
𝐸𝑠𝑢𝑏(𝑣)={𝑒∈(𝐸 𝑙𝑜𝑤∪𝐸ℎ𝑖𝑔ℎ)|𝑣∈𝑒}
𝑉𝑠𝑢𝑏(𝑣)={𝑢∈𝑉|∃𝑒∈𝐸 𝑠𝑢𝑏(𝑣),𝑢∈𝑒}(7)
where𝐸𝑠𝑢𝑏represents the set of all hyperedges (both pairwise and
high-order) incident to 𝑣, and𝑉𝑠𝑢𝑏denotes the union of all vertices
contained within these incident hyperedges.
This localized sub-hypergraph 𝐺𝑠𝑢𝑏acts as a structural nexus,
providing a condensed reasoning substrate for the refinement op-
eratorRto bridge disparate page-level fragments and synthesize
latent cross-page associations:
𝐸𝑟𝑒𝑓𝑖𝑛𝑒 =LLM𝑟𝑒𝑓(R(𝐺𝑠𝑢𝑏(𝑣),𝑃𝑟𝑒𝑓)),(8)
where𝑃𝑟𝑒𝑓is a specialized prompt directing the model to synthesize
novel associations or consolidate existing relations by reconciling
multi-source evidence. Finally, the refined relations are integrated
into the global multi-modal hypergraphG 𝑓𝑖𝑛𝑎𝑙:
G𝑓𝑖𝑛𝑎𝑙={𝑉,𝐸𝑙𝑜𝑤,𝐸ℎ𝑖𝑔ℎ,𝐸𝑟𝑒𝑓𝑖𝑛𝑒}.(9)
By shifting the refinement focus from raw page sequences to
topological anchor neighborhoods, the system effectively eliminates
the redundant processing of stable local information. This ensures
a more precise discovery of high-order relational structures while
maintaining significant computational efficiency.
4.3 Cognitive-inspired Hybrid Retrieval
Guided by the hierarchical structure of theHyperIndex, we design
a cognitive-inspired hybrid retrieval strategy to bridge user queries
with high-fidelity evidence. As illustrated in our framework, the
process initiates with a Query Decomposition stage, followed by a
Dual-channel Retrieval mechanism that traverses both the symbolic
hypergraph and the dense vector space.
4.3.1 Multi-modal Query Decomposition.For a given user query 𝑞,
the system first employs an MLLM-based decomposition operator
to extract two levels of semantic anchors:
{𝐾𝑙𝑜𝑤,𝐾ℎ𝑖𝑔ℎ}=LLM𝑑𝑒𝑐(𝑃𝑑𝑒𝑐(𝑞)),(10)
where𝐾𝑙𝑜𝑤represents fine-grained entity keywords (e.g., specific
objects or technical terms), and 𝐾ℎ𝑖𝑔ℎdenotes high-level thematic
keywords (e.g., abstract concepts or cross-page relations). 𝑃𝑑𝑒𝑐is
the decomposition prompt detailed in the Appendix.

Hypergraph-based Multimodal Retrieval-Augmented Generation with Incremental Refinement MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil.
4.3.2 Dual-channel Evidence Retrieval.The retrieval process simul-
taneously executes two complementary paths to collect structured
evidence from the HyperIndex:
Path I: Vertex-anchored Entity Retrieval.This path targets fine-
grained evidence by anchoring the query to specific semantic ver-
tices within the hypergraph. For each keyword 𝑘∈𝐾𝑙𝑜𝑤, the sys-
tem identifies the most relevant vertices 𝑉𝑞via semantic similarity
search in the entity vector space:
𝑉𝑞={𝑣∈𝑉|sim(𝜙(𝑘),𝜙(𝑣))>𝛾},(11)
where𝜙(·) denotes the embedding function and 𝛾is the similarity
threshold. Using 𝑉𝑞as entry points, the entity-centric context 𝐶𝑒𝑛𝑡
is expanded via their incident hyperedges in the HypergraphDB:
𝐶𝑒𝑛𝑡={𝑒∈(𝐸 𝑙𝑜𝑤∪𝐸ℎ𝑖𝑔ℎ∪𝐸𝑟𝑒𝑓𝑖𝑛𝑒)|∃𝑣∈𝑉 𝑞,𝑣∈𝑒}.(12)
This operation captures the topological neighbors of the query
entities as structural evidence, preserving the immediate relational
context and leveraging refined cross-page associations.
Path II: Hyperedge-directed Relational Retrieval.To capture over-
arching themes and cross-page narratives, this path interfaces di-
rectly with the relational vector index using 𝐾ℎ𝑖𝑔ℎ. For each thematic
keyword𝑘∈𝐾ℎ𝑖𝑔ℎ, the system retrieves a candidate set of high-
order hyperedges 𝐸𝑞whose semantic embeddings align with the
query intent:
𝐸𝑞={𝑒∈(𝐸 𝑙𝑜𝑤∪𝐸ℎ𝑖𝑔ℎ∪𝐸𝑟𝑒𝑓𝑖𝑛𝑒)|sim(𝜙(𝑘),𝜙(𝑒))>𝛾},(13)
where𝐸𝑞specifically targets the refined associations that transcend
individual page boundaries. The relation-centric context 𝐶𝑟𝑒𝑙is
then synthesized by back-tracing the constituent vertices and their
multi-source provenance:
𝐶𝑟𝑒𝑙={src(𝑒),nodes(𝑒)|𝑒∈𝐸 𝑞},(14)
where nodes(𝑒) extracts the semantic entities within the hyperedge
andsrc(𝑒) retrieves the original multi-page text chunks. This path
ensures the model reconciles broad conceptual queries with the
high-order structural evidence consolidated during the refinement
stage.
4.3.3 Evidence Fusion and Generation.Finally, the system inte-
grates the multi-source evidence from both retrieval channels into
a unified, structured context 𝑆=Fuse(𝐶 𝑒𝑛𝑡,𝐶𝑟𝑒𝑙). Unlike conven-
tional RAG systems that return fragmented raw text, 𝑆is a Struc-
tured Evidence Package comprising: (i) normalized entities, (ii) high-
order hyperedges, and (iii) their synchronized multi-page source
chunks. The final response𝑅is synthesized as:
𝑅=LLM𝑔𝑒𝑛(𝑃𝑔𝑒𝑛(𝑞,𝑆)),(15)
where𝑃𝑔𝑒𝑛is a generation prompt that instructs the model to reason
over the provided topological structures and cross-page evidence.
The efficacy of this dual-channel paradigm is fundamentally
rooted in the structural depth of the 𝐻𝑦𝑝𝑒𝑟𝐼𝑛𝑑𝑒𝑥 . Specifically, the
cross-page hyperedges 𝐸𝑟𝑒𝑓𝑖𝑛𝑒 generated during the anchor-driven
refinement stage enable Path II to retrieve integrated evidence that
transcends individual page boundaries. By reconciling these high-
order associations with local entity-centric details, the framework
provides a holistic and factually grounded understanding of com-
plex, long-form documents.5 Experiments
In this section, we provide a comprehensive evaluation of Hyper-
M2RAG. We first detail our experimental configurations and bench-
mark selections, followed by a quantitative analysis comparing our
approach against state-of-the-art RAG baselines.
5.1 Datasets
To evaluate Hyper-M2RAG’s proficiency in managing multi-modal
document structures encompassing both holistic understanding
and granular retrieval, we conduct experiments across two task
granularities: Global QA and Local QA.
5.1.1 Global QA.To evaluate the system’s ability to synthesize
information across entire document collections (book-level), we
employ both textual and multi-modal benchmarks:
•Textual Corpus:We utilize theMixed-Domain( ≈0.62M tokens)
from theUltraDomain[ 18] dataset andNeurologyCorp( ≈1.97M
tokens) from MedRAG [ 24]. These corpora specifically test the
model’s capacity for maintaining long-range semantic consis-
tency within dense, specialized knowledge domains.
•multi-modal Benchmark:To address the scarcity of standard-
ized multi-modal global QA datasets, we utilize a specialized
evaluation set characterized by dense, visually-rich content. This
benchmark incorporates a World History textbook (788 pages)
and a Sustainable Development Report (SDR) slide deck (491
pages), both of which feature a high concentration of interleaved
graphical elements, including intricate charts, tables, and the-
matic illustrations that are deeply integrated with the primary
text.
5.1.2 Global Question Generation Protocol.To address the lack of
human-labeled queries, we use an automated protocol with two
modality-specific pipelines:
•Textual Generation:Using the document outline as a scaffold,
we prompt an LLM to simulate 5 distinct professional users. For
each user, we define 5 strategic tasks, each generating 5 complex
questions that require a holistic understanding of the document.
This results in 125 global questions per dataset.
•multi-modal Generation:For vision-heavy documents, we
establish a generation pipeline. We sample 25 representative
pages per dataset (organized into 5 batches of 5 pages). Each
batch involves a specific user persona and 5 tasks. Crucially, we
enforce a multi-modal-dependency constraint: questions must
be unanswerable by text alone, requiring reasoning over visual
data such as map coordinates, chart trends, or diagram structures.
This ensures 125 high-quality multi-modal questions per dataset.
5.1.3 Local QA.To evaluate retrieval precision at the granular level
(page- or slide-level), we utilize the RealMMBench [ 20] benchmark.
RealMMBench is specifically designed to assess multi-modal RAG
systems under stressed conditions, including visual-rich layouts,
table-heavy content, and sophisticated query rephrasing. While
RealMMBench spans multiple domains, we perform a strategic eval-
uation on the TechReport (1,674 pages) and TechSlides (1,963 pages)
sub-datasets. These technical domains are characterized by dense
high-order structural dependencies, making them ideal for vali-
dating our high-order hypergraph construction and anchor-driven

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil. Shenao Chen et al.
refinement mechanisms. This selection ensures a rigorous stress
test of document intelligence while maintaining computational
feasibility during the evaluation process.
5.2 Baselines and Evaluation Metrics
We compareHyper-M2RAGagainst several state-of-the-art base-
lines: GraphRAG [ 3], a widely adopted knowledge graph frame-
work; HyperRAG [ 5], which utilizes text-only hypergraphs; and
MegaRAG [ 10], a multi-modal graph-based RAG system. To ensure
a fair assessment, we evaluate all methods across both textual and
multi-modal benchmarks to test their core retrieval and reasoning
capabilities.
5.2.1 Global QA Metrics.Given the absence of ground-truth an-
swers for book-level queries, prior works often rely on LLM-based
evaluation. However, such evaluations are frequently prone topo-
sition bias, where the evaluator model favors responses presented
earlier in the prompt. To mitigate this and ensure reliability, we
implement a Dual-Response Swap & Average strategy:
•Dual-Prompt Evaluation:For every pair of responses (𝐴,𝐵) ,
we generate two distinct evaluation prompts: a forward prompt
(A followed by B) and a reversed prompt (B followed by A).
•Result Normalization:To maintain a consistent reference frame,
results from the reversed evaluation are flipped (e.g., if B was
preferred in the reversed prompt, it is mapped to a preference
for the second position in the forward frame).
•Consistency Arbitration:We compare the forward and normal-
ized reversed results. If the evaluator remains consistent across
both prompts, the result is recorded. If the two evaluations con-
flict, the pair is marked as aTieto filter out stochastic noise and
ensure robust win-rate calculation.
Responses are assessed across four qualitative dimensions fol-
lowing [ 8]: (1) Comprehensiveness: Coverage of all query facets;
(2) Diversity: Richness of perspectives; (3) Empowerment: Support
for user understanding; and (4) Overall: An aggregate measure of
the preceding criteria.
5.2.2 Local QA Metrics.For granular (page- or slide-level) QA,
performance is measured by semantic alignment with ground-truth
reference answers. We utilize a powerful LLM as an automated
judge to determine semantic consistency between the generated
and reference answers, reporting Accuracy as the primary metric.
5.3 Implementation Details
To ensure consistency across all evaluated RAG frameworks, we
standardize the backbone models and processing pipeline. For re-
sponse generation, we utilize Qwen3-VL-8B [ 1], while DeepSeek-v3
[15] is employed for synthetic question generation and evaluation
due to its superior reasoning robustness.
All methods share a unified embedding space facilitated by gme-
Qwen2-VL-2B-Instruct [ 28], which supports single-, cross-, and
fused-modality retrieval tasks. Textual corpora are partitioned into
1,200-token chunks with a 100-token overlap. For multi-modal
documents, we leverage the MinerU2.5 toolkit [ 17] to extract text,
figures, and tables. MinerU’s ability to preserve complex layouts
and symbols in machine-readable formats is particularly effective
for the technical documents used in our study.5.4 Global QA Performance
The primary results in Table 1 compare Hyper-M2RAG against three
state-of-the-art baselines across four domains: two purely textual
corpora (Mix and Neurology) and two multimodal datasets (World
History and Sustainable Report). While our method maintains a
consistent lead in textual tasks, it achieves a decisive, overwhelming
victory in multimodal scenarios. Our analysis yields the following
key insights:
Superiority across Diverse Domains.Hyper-M2RAGconsistently
outperforms GraphRAG, HyperRAG, and MegaRAG across all qual-
itative dimensions. Notably, in the Sustainable Report domain char-
acterized by high-density multimodal content and complex layouts,
our method achieves a dominant Overall win rate of 88.8% against
MegaRAG and 77.6% against GraphRAG. These results demonstrate
that our high-order hypergraph structure captures multi-scale se-
mantic dependencies in technical documents more effectively than
traditional binary graphs.
Gains in Diversity and Empowerment.Hyper-M2RAG demon-
strates a substantial lead in Diversity and Empowerment metrics.
For instance, in the World History domain, our method achieves a
75.2% win rate in Diversity against MegaRAG. This performance
gain is driven by our hyper-relational modeling mechanism, which
transcends the limitations of binary-edge retrieval. Unlike tradi-
tional graphs that only link pairwise entities, our high-order hy-
peredges group multi-modal entities into unified semantic clusters.
This allows the retriever to capture non-local, multi-hop connec-
tions across disparate document sections in a single traversal.
Robustness against Hypergraph Baselines.Compared to the text-
only HyperRAG, Hyper-M2RAG maintains a consistent margin of
improvement, particularly in Overall win rates ranging from 22.4%
to 44.8%. While HyperRAG introduces basic hyper-structures, our
anchor-driven refinement and multimodal integration ensure that
the constructed hypergraphs are semantically precise rather than
merely dense. The high tie rates in Comprehensiveness (e.g., 91.2%
in Mix) indicate that while most hypergraph methods achieve basic
content coverage, Hyper-M2RAG excels in the qualitative depth
and organizational clarity of the generated responses.
Multimodal Advantage.In multimodal datasets like World His-
tory and Sustainable Report, the performance gap between Hyper-
M2RAG and baselines widens significantly. This advantage stems
from our layout-aware hyperedges, which directly link visual ele-
ments to their corresponding textual descriptions within a unified
topological space. For instance, the 90.4% Empowerment score in
the Sustainable domain demonstrates the model’s superior ability
to interpret and synthesize information from complex tables and
figures.
5.5 Local QA Performance
Table 2 reports the accuracy of different methods on the TechRe-
port and TechSlides sub-datasets of RealMMBench. Hyper-M2RAG
achieves superior performance, reaching 70.0% and 78.0% accuracy,
respectively. Compared with the state-of-the-art baseline MegaRAG,
our method achieves a notable 9 percentage point improvement on

Hypergraph-based Multimodal Retrieval-Augmented Generation with Incremental Refinement MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil.
Table 1: Win-rate analysis comparing Hyper-M2RAG against GraphRAG, HyperRAG, and MegaRAG across four distinct
domains. Results are presented as percentages (%). Bold values indicate the winning method (excluding ties).
MetricsMix Neurology World History Sustainable Report
GraphRAGOursTie GraphRAGOursTie GraphRAGOursTie GraphRAGOursTie
Comprehensiveness 0.046.453.6 0.837.661.6 0.847.252.0 0.062.437.6
Diversity 11.260.828.0 9.656.034.4 4.869.625.6 3.291.25.6
Empowerment 4.056.040.0 3.261.635.2 5.666.428.0 1.677.620.8
Overall 2.4 51.2 46.4 2.4 52.0 45.6 4.8 61.6 33.6 0.8 77.6 21.6
HyperRAGOursTie HyperRAGOursTie HyperRAGOursTie HyperRAGOursTie
Comprehensiveness 0.08.891.2 6.417.676.0 0.044.056.0 1.629.668.8
Diversity 23.235.241.6 3.218.478.4 15.254.430.4 17.643.239.2
Empowerment 15.228.856.0 14.428.856.8 16.846.436.8 20.826.452.8
Overall 13.6 27.2 59.2 11.2 22.4 66.4 14.4 44.8 40.8 11.2 25.6 63.2
MegaRAGOursTie MegaRAGOursTie MegaRAGOursTie MegaRAGOursTie
Comprehensiveness 4.853.641.6 0.020.080.0 0.044.056.0 0.064.036.0
Diversity 8.844.846.4 6.452.840.8 2.475.222.4 1.690.48.0
Empowerment 3.263.233.6 0.844.854.4 3.262.434.4 1.690.48.0
Overall 5.6 58.4 36.0 0.0 36.8 63.2 1.6 59.2 39.2 1.6 88.8 9.6
Table 2: Local QA performance (Accuracy %) on RealMM-
Bench.
Method TechReport TechSlides
GraphRAG 30.0 57.0
HyperRAG 56.0 67.0
MegaRAG 68.0 69.0
Hyper-M2RAG (Ours) 70.0 78.0
TechSlides. This specific gain highlights the efficacy of our frame-
work in visual-heavy contexts. Technical slides often contain frag-
mented information across different visual blocks; while MegaRAG
relies on dense graph connectivity, our high-order hypergraph
structure more effectively clusters spatially disparate but seman-
tically related elements (e.g., a figure and its corresponding bullet
points), leading to more precise local retrieval.
5.6 Ablation Study
To investigate the contribution of different components in Hyper-
M2RAG, we progressively introduce multi-modal extraction and
anchor-driven refinement based on the HyperRAG baseline, as
shown in Table 3. Our analysis yields two primary insights:
Effect of Multi-modal Hypergraph Construction.Adding multi-
modal extraction (+ MM Extraction) consistently improves perfor-
mance over the text-only HyperRAG baseline on both TechReport
and TechSlides datasets. This demonstrates that incorporating vi-
sual elements and layout information helps preserve cross-modal
semantic dependencies, enabling more complete evidence retrieval
from complex documents.
Effect of Anchor-driven Refinement.Introducing anchor-driven
refinement further improves the performance over the multi-modal
hypergraph variant. This verifies that selectively reconstructinglocal neighborhoods around cross-page anchors effectively captures
long-range dependencies while avoiding unnecessary processing
of irrelevant regions.
Overall, the ablation results confirm that both multi-modal repre-
sentation and anchor-driven refinement are essential components
of Hyper-M2RAG, with their combination enabling more effective
multimodal document understanding.
Table 3: Ablation study of Hyper-M2RAG components.
Variant TechReport TechSlides
HyperRAG (Baseline) 56.0 67.0
+ MM Extraction 64.0 74.0
+ Anchor Refinement 60.0 69.0
Hyper-M2RAG (Full) 70.0 78.0
5.7 Structural Complexity Analysis
To investigate the structural characteristics ofHyper-M2RAG,
Table 4 compares the hypergraph statistics of different methods,
with particular focus on high-order relations (degree>2).
High-Order Relational Modeling.Hyper-M2RAGconstructs sub-
stantially more high-order hyperedges than existing approaches.
OnTechReport, Hyper-M2RAG builds 8,257 high-order hyperedges,
compared with 2,579 from HyperRAG, representing a220% in-
crease. Similarly, onTechSlides, the number of high-order hyper-
edges increases from 1,716 to 6,571. Since high-order hyperedges
explicitly represent dependencies among multiple entities, this in-
crease indicates that Hyper-M2RAG provides a richer structural
representation beyond conventional pairwise relations.
Effect of Multi-modal Hypergraph Construction.Interestingly,
simply incorporating multi-modal information does not necessarily

MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil. Shenao Chen et al.
Table 4: Comparison of structural hyperedge counts across datasets.
VariantTechReport(1,674 pages)TechSlides(1,963 pages)
Low High Total Low High Total
HyperRAG (Baseline) 12,957 2,579 15,536 8,543 1,716 10,259
HyperRAG + MM 10,016 2,078 12,094 9,309 2,004 11,313
Hyper-M2RAG (Ours) 14,221 8,257 22,478 12,522 6,571 19,093
lead to more high-order structures. The HyperRAG+MM variant
produces fewer high-order hyperedges than HyperRAG onTechRe-
port(2,078 vs. 2,579). This suggests that modality augmentation
alone is insufficient to establish effective high-order relations. In
contrast, Hyper-M2RAG jointly models textual, visual, and lay-
out elements through hypergraph construction and anchor-driven
refinement, resulting in substantially richer high-order relational
structures.
Overall Structural Connectivity.Beyond high-order relations,
Hyper-M2RAG also achieves the largest number of total hyper-
edges across both datasets, with 22,478 edges onTechReportand
19,093 edges onTechSlides. This demonstrates that our framework
constructs a more densely connected document representation,
providing a stronger structural foundation for retrieving evidence
distributed across long and complex documents.
Table 5: Multi-Judge Audit (%). H-RAG = HyperRAG.
Sustainable Report World History
EvaluatorH-RAGOursTie H-RAGOursTie
Kimi-K2.6 25.655.219.2 24.044.032.0
GLM-5.1 25.651.223.2 24.838.436.8
DeepSeek-V4 Pro 15.241.643.2 20.036.044.0
5.8 Cross-Model Judge Audit
Table 5 presents a multi-judge evaluation on the Sustainable Report
and World History corpora, conducted with three recently released,
architecturally diverse LLMs: Kimi-K2.6, GLM-5.1, and DeepSeek-
V4 Pro. We select HyperRAG as the primary baseline for this audit
because it is the most structurally aligned counterpart to Hyper-
M2RAG: both methods organize knowledge using hypergraphs,
while only Hyper-M2RAG introduces multimodal anchors and star-
expansion refinement. This design isolates the contribution of our
multimodal mechanism from general hypergraph-induced gains.
Across both domains and all three evaluators, Hyper-M2RAG
consistently outperforms HyperRAG, with win rates ranging from
36.0% to 55.2%. Crucially, the preference for Hyper-M2RAG is not
judge-specific: the same ranking trend holds for all model families,
indicating that the observed improvements are not a single-judge
artifact but a robust, cross-model phenomenon.
5.9 Reduction in Token Cost and Latency
Table 6 provides a head-to-head efficiency comparison on the World
History corpus (per 100 pages). Hyper-M2RAG reduces build andTable 6:Efficiency on World History (per 100 pages).
Metric MegaRAG HyperRAG + Page Ours
Build & Refine Time (min) 16 18 12
Token Consumption (103) 2,742.2 3,376.1 765.9
refinement time to 12 min and lowers token consumption by 72%
relative to MegaRAG and 77% relative to HyperRAG+Page. This
gain stems from a fundamental shift in refinement granularity.
Rather than performing document- or page-level rescans dur-
ing incremental updates, Hyper-M2RAG organizes multimodal evi-
dence into a hypergraph, where anchors serve as structural pivots
linking text, tables, figures, and layout elements. Refinement is then
confined to anchor-centered star neighborhoods: only hyperedges
incident to updated anchors are recomputed.
Consequently, the update process is transformed from a global
sweep into a topology-guided local operation. This design elimi-
nates redundant computation across unrelated content while pre-
serving rich cross-modal linkages. Because the number of anchors
requiring updates grows more slowly than document length, the re-
finement cost scales sub-linearly, allowing Hyper-M2RAG to main-
tain high efficiency on long, multimodal documents without altering
the input scope or representation.
6 Conclusion
In this paper, we presentedHyper-M2RAG, a novel multi-modal
retrieval-augmented generation framework designed for complex,
high-density technical documents. By transitioning from traditional
binary graphs to a high-order hypergraph structure, our approach
effectively captures the multi-scale semantic dependencies that are
often fragmented across disparate document sections and modali-
ties. Our core contribution lies in the integration of layout-aware hy-
peredge construction and an anchor-driven refinement mechanism.
This synergy ensures that visual elements, such as tables and fig-
ures, are not merely treated as isolated inputs but are topologically
aligned with their textual context within a unified semantic space.
Furthermore, our dual-path retrieval strategy—comprising Vertex-
anchored Entity Retrieval and Hyperedge-directed Relational Re-
trieval—enables the model to reconcile fine-grained evidence with
overarching thematic narratives. Experimental results across four
diverse domains demonstrate that Hyper-M2RAG consistently out-
performs state-of-the-art baselines, particularly in multi-modal
scenarios where it achieves significant gains in Diversity and Em-
powerment.

Hypergraph-based Multimodal Retrieval-Augmented Generation with Incremental Refinement MM ’26, November 10–14, 2026, Rio de Janeiro, Brazil.
7 Acknowledgments
This work was supported by Brain Science and Brain-like Intel-
ligence Technology——National Science and Technology Major
Project (2025ZD0217300), the National Key Research and Develop-
ment Program of China under Grant (2023YFB4502803), the Na-
tional Natural Science Foundation of China (No. U25A20532) and
the Beijing Natural Science Foundation under Grant (No. L242167).
References
[1]Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng,
Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al .2025. Qwen3-vl
technical report.arXiv preprint arXiv:2511.21631(2025).
[2]Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu,
Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al .2025. Paddleocr
3.0 technical report.arXiv preprint arXiv:2507.05595(2025).
[3]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[4]Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline
Hudelot, and Pierre Colombo. 2025. Colpali: Efficient document retrieval with
vision language models. InInternational Conference on Learning Representations,
Vol. 2025. 61424–61449.
[5]Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shihui Ying, Shaoyi Du, Han Hu,
and Yue Gao. 2026. Hyper-rag: Combating LLM hallucinations using hypergraph-
driven retrieval-augmented generation.Nature Communications17 (2026), 5778.
doi:10.1038/s41467-026-71411-1
[6]Ziyu Gong, Chengcheng Mai, and Yihua Huang. 2025. MHier-RAG: Multi-Modal
RAG for Visual-Rich Document Question-Answering via Hierarchical and Multi-
Granularity Reasoning.arXiv preprint arXiv:2508.00579(2025).
[7]Zirui Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. 2025. Rag-
anything: All-in-one rag framework.arXiv preprint arXiv:2510.12323(2025).
[8]Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. LightRAG:
Simple and Fast Retrieval-Augmented Generation. InFindings of the Association
for Computational Linguistics: EMNLP 2025. Association for Computational Lin-
guistics, Suzhou, China, 10746–10761. doi:10.18653/v1/2025.findings-emnlp.568
[9]Mahd Hindi, Linda Mohammed, Ommama Maaz, and Abdulmalik Alwarafy. 2025.
Enhancing the precision and interpretability of retrieval-augmented generation
(rag) in legal technology: A survey.IEEE Access(2025).
[10] Chi-Hsiang Hsiao, Yi-Cheng Wang, Tzung-Sheng Lin, Yi-Ren Yeh, and Chu-Song
Chen. 2026. MegaRAG: Multimodal Knowledge Graph-Based Retrieval Aug-
mented Generation. InProceedings of the 64th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers). 48031–48059.
[11] Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue, Xingliang Hou, Zhiqiang Tian, Yue
Gao, and Shaoyi Du. 2026. Cog-rag: Cognitive-inspired dual-hypergraph with
theme alignment retrieval-augmented generation. InProceedings of the AAAI
Conference on Artificial Intelligence, Vol. 40. 31032–31040.
[12] Pankaj Joshi, Aditya Gupta, Pankaj Kumar, and Manas Sisodia. 2024. Robust multi
model rag pipeline for documents containing text, table & images. In2024 3rd
International Conference on Applied Artificial Intelligence and Computing (ICAAIC).
IEEE, 993–999.
[13] Wenjun Ke, Yifan Zheng, Yining Li, Hengyuan Xu, Dong Nie, Peng Wang, and
Yao He. 2025. Large language models in document intelligence: A comprehensive
survey, recent advances, challenges, and future trends.ACM Transactions on
Information Systems44, 1 (2025), 1–64.[14] Zhicong Li, Jiahao Wang, Zhishu Jiang, Hangyu Mao, Zhongxia Chen, Jiazhen
Du, Yuanxing Zhang, Fuzheng Zhang, Di Zhang, and Yong Liu. 2024. Dmqr-rag:
Diverse multi-query rewriting for rag.arXiv preprint arXiv:2411.13154(2024).
[15] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng-
gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al .2024. Deepseek-v3
technical report.arXiv preprint arXiv:2412.19437(2024).
[16] Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, and Jun
Ma. 2025. Hm-rag: Hierarchical multi-agent multimodal retrieval augmented
generation. InProceedings of the 33rd ACM international conference on multimedia.
2781–2790.
[17] Junbo Niu, Zheng Liu, Zhuangcheng Gu, Bin Wang, Linke Ouyang, Zhiyuan
Zhao, Tao Chu, Tianyao He, Fan Wu, Qintong Zhang, et al .2026. Mineru2.5: A
decoupled vision-language model for efficient high-resolution document parsing.
InProceedings of the 64th Annual Meeting of the Association for Computational
Linguistics (Volume 6: Industry Track). Association for Computational Linguistics,
San Diego, California, USA, 13–42. doi:10.18653/v1/2026.acl-industry.3
[18] Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng
Dou, and Tiejun Huang. 2025. MemoRAG: Boosting Long Context Processing
with Global Memory-Enhanced Retrieval Augmentation. InProceedings of the
ACM Web Conference 2025 (TheWebConf 2025). ACM, Sydney, Australia. https:
//arxiv.org/abs/2409.05591
[19] Minzheng Wang, Longze Chen, Fu Cheng, Shengyi Liao, Xinghua Zhang, Bingli
Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, et al .2024. Leave no document
behind: Benchmarking long-context llms with extended multi-doc qa. InProceed-
ings of the 2024 Conference on Empirical Methods in Natural Language Processing.
5627–5646.
[20] Navve Wasserman, Roi Pony, Oshri Naparstek, Adi Raz Goldfarb, Eli Schwartz,
Udi Barzelay, and Leonid Karlinsky. 2025. Real-mm-rag: A real-world multi-
modal retrieval benchmark. InProceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers). 31660–31683.
[21] Haoran Wei, Yaofeng Sun, and Yukun Li. 2026. DeepSeek-OCR 2: Visual Causal
Flow.arXiv preprint arXiv:2601.20552(2026).
[22] Junda Wu, Yu Xia, Tong Yu, Xiang Chen, Sai Sree Harsha, Akash V Maharaj,
Ruiyi Zhang, Victor Bursztyn, Sungchul Kim, Ryan A Rossi, et al .2025. Doc-react:
Multi-page heterogeneous document question-answering. InProceedings of the
63rd Annual Meeting of the Association for Computational Linguistics (Volume 2:
Short Papers). 67–78.
[23] Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming
Huang, Xue Liu, Tei-Wei Kuo, Nan Guan, et al .2026. Retrieval-augmented
generation for natural language processing: a survey.Artificial Intelligence Review
(jun 2026). doi:10.1007/s10462-026-11605-7
[24] Guangzhi Xiong, Qiao Jin, Zhiyong Lu, and Aidong Zhang. 2024. Benchmarking
retrieval-augmented generation for medicine. InFindings of the Association for
Computational Linguistics: ACL 2024. 6233–6251.
[25] Diji Yang, Jinmeng Rao, Kezhen Chen, Xiaoyuan Guo, Yawen Zhang, Jie Yang, and
Yi Zhang. 2024. Im-rag: Multi-round retrieval-augmented generation through
learning inner monologues. InProceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval. 730–740.
[26] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao
Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al .2025. Visrag: Vision-based retrieval-
augmented generation on multi-modality documents. InInternational Conference
on Learning Representations, Vol. 2025. 21074–21098.
[27] Rui Zhang, Chen Liu, Yixin Su, Ruixuan Li, Xuanjing Huang, Xuelong Li, and
Philip S Yu. 2025. A comprehensive survey on multimodal RAG: all combinations
of modalities as input and output.Authorea Preprints(2025).
[28] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long,
Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. GME: im-
proving universal multimodal retrieval by multimodal LLMs.arXiv preprint
arXiv:2412.16855(2024).