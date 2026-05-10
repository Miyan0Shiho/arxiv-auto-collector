# GATHER: Convergence-Centric Hyper-Entity Retrieval for Zero-Shot Cell-Type Annotation

**Authors**: Zhonghui Zhang, Feng Jiang, Shaowei Qin, Jiahao Zhao, Min Yang

**Published**: 2026-05-07 15:11:54

**PDF URL**: [https://arxiv.org/pdf/2605.06403v1](https://arxiv.org/pdf/2605.06403v1)

## Abstract
Zero-shot single-cell cell-type annotation aims to determine a cell's type from a given set of expressed genes without any training. Existing knowledge-graph-based RAG approaches retrieve evidence by expanding from source entities and relying on iterative LLM reasoning. However, in this setting each query contains tens to hundreds of genes, where no single gene is decisive and the label emerges only from their collective co-occurrence. Such hyper-entity queries fundamentally challenge local, entity-wise exploration strategies, which reason from individual genes, leading to poor scalability and substantial LLM cost. We propose GATHER (Graph-Aware Traversal with Hyper-Entity Retrieval), a convergence-centric retriever tailored to hyper-entity queries. It performs global multi-source graph traversal and identifies topological convergence points -- nodes jointly reachable from many input genes. These convergence nodes act as high-information hyper-entities that capture entity synergy. By incorporating node- and path-importance scoring, GATHER selects informative evidence entirely without LLM involvement during retrieval. Instantiated on a self-constructed cell-centric biological knowledge graph (VCKG), GATHER outperforms strong KG-RAG baselines (ToG, ToG-2, RoG, PoG) on two datasets (Immune and Lung), achieving the highest exact-match accuracy (27.45% and 59.64%) with only a single LLM call per sample, compared to 2--61 calls for KG-RAG baselines. Our results demonstrate that convergence nodes compress multi-entity signals into compact, high-information evidence that conveys more per item than multi-hop paths, providing an efficient global alternative to local entity-wise reasoning.

## Full Text


<!-- PDF content starts -->

GATHER: Convergence-Centric Hyper-Entity Retrieval for
Zero-Shot Cell-Type Annotation
Zhonghui Zhang
Artificial Intelligence Research
Institute, Shenzhen University of
Advanced Technology
Shenzhen, China
zhangzhonghui@suat-sz.edu.cnFeng Jiang∗
Artificial Intelligence Research
Institute, Shenzhen University of
Advanced Technology
Shenzhen, China
jiangfeng@suat-sz.edu.cnShaowei Qin
Artificial Intelligence Research
Institute, Shenzhen University of
Advanced Technology
Shenzhen, China
qinshaowei@suat-sz.edu.cn
Jiahao Zhao
Software College, Northeastern
University
Shenyang, China
zhaojh3417@gmail.comMin Yang∗
Shenzhen Institutes of Advanced
Technology, Chinese Academy of
Sciences;
Artificial Intelligence Research
Institute, Shenzhen University of
Advanced Technology
Shenzhen, China
min.yang@siat.ac.cn
Abstract
Zero-shot single-cell cell-type annotation aims to determine a cell’s
type from a given set of expressed genes without any training. Exist-
ing knowledge-graph-based RAG approaches retrieve evidence by
expanding from source entities and relying on iterative LLM reason-
ing. However, in this setting each query contains tens to hundreds
of genes, where no single gene is decisive and the label emerges
only from their collective co-occurrence. Such hyper-entity queries
fundamentally challenge local, entity-wise exploration strategies,
which reason from individual genes, leading to poor scalability and
substantial LLM cost. We proposeGATHER(Graph-Aware Traver-
sal with Hyper-Entity Retrieval), a convergence-centric retriever
tailored to hyper-entity queries. It performs global multi-source
graph traversal and identifies topological convergence points—
nodes jointly reachable from many input genes. These conver-
gence nodes act as high-information hyper-entities that capture
entity synergy. By incorporating node- and path-importance scor-
ing, GATHER selects informative evidence entirely without LLM
involvement during retrieval. Instantiated on a self-constructed
cell-centric biological knowledge graph (VCKG), GATHER outper-
forms strong KG-RAG baselines (ToG, ToG-2, RoG, PoG) on two
datasets (Immune and Lung), achieving the highest exact-match ac-
curacy (27.45% and 59.64%) with only a single LLM call per sample,
compared to 2–61 calls for KG-RAG baselines. Our results demon-
strate that convergence nodes compress multi-entity signals into
compact, high-information evidence that conveys more per item
∗Corresponding authors.
This work is licensed under a Creative Commons Attribution 4.0 International License.
SIGIR ’26, Melbourne, VIC, Australia
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2599-9/2026/07
https://doi.org/10.1145/3805712.3809935than multi-hop paths, providing an efficient global alternative to
local entity-wise reasoning.
CCS Concepts
•Information systems →Retrieval models and ranking;•
Computing methodologies →Knowledge representation and
reasoning;•Applied computing→Bioinformatics.
Keywords
Hyper-Entity Retrieval; Knowledge Graph RAG; Convergence-Centric
Retrieval; Cell Type Annotation
ACM Reference Format:
Zhonghui Zhang, Feng Jiang, Shaowei Qin, Jiahao Zhao, and Min Yang.
2026. GATHER: Convergence-Centric Hyper-Entity Retrieval for Zero-Shot
Cell-Type Annotation. InProceedings of the 49th International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR
’26), July 20–24, 2026, Melbourne, VIC, Australia.ACM, New York, NY, USA,
6 pages. https://doi.org/10.1145/3805712.3809935
1 Introduction
Single-cell cell-type annotation [ 2,16,22] aims to assign a cell type
based on a cell’s gene expression profile and is fundamental to
computational biology, enabling downstream analyses such as cell-
type discovery and disease mechanism study. Given a cell’s gene
expression profile, the task relies on thejoint expression patternof
many genes rather than any single marker. Thus, the prediction
signal emerges from theglobal interactionamong tens to hundreds
of genes.
Supervised foundation models such as scGPT [ 7], scBERT [ 21],
and Geneformer [ 20] achieve strong accuracy but operate as black
boxes, limiting interpretability. In training-free settings, large lan-
guage models (LLMs) [ 1,4] offer explainable reasoning but suffer
from imprecise domain knowledge [ 10,23]. Retrieval-AugmentedarXiv:2605.06403v1  [cs.CL]  7 May 2026

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Zhang et al.
Generation (RAG) [ 9,12] has therefore emerged as a promising
paradigm to ground LLMs with structured knowledge.
However, applying RAG to cell-type annotation introduces a
key challenge: each query consists of tens to hundreds of genes,
and effectively leveraging external knowledge graphs over such a
large set of entities becomes non-trivial. The central question is
how to integrate structured knowledge while jointly considering
all source entities. Existing knowledge graph-based RAG meth-
ods [ 15,24], such as ToG [ 18] and PoG [ 19], primarily adopt a
local expansionparadigm. They start from each source entity in-
dependently, explore neighboring nodes, and aggregate retrieved
paths as separate evidence. While effective for few-entity queries,
this strategy becomes problematic in hyper-entity settings. First,
independent expansion fragments the collective signal and fails
to explicitly model interactions among many entities. Second, the
search and LLM-interaction costs scale rapidly with the number of
source entities, expansion breadth, and depth.
To address these limitations, we proposeGATHER(Graph-
AwareTraversal withHyper-EntityRetrieval), which shifts from
local expansion to a global convergence paradigm. Instead of reason-
ing from each entity separately, GATHER identifies importantcon-
vergence points—nodes jointly reachable from many source entities—
which serve as high-informationhyper-entities. These nodes nat-
urally capture the structural interactions among genes and act as
consolidated evidence. Figure 1 contrasts this convergent retrieval
pattern with divergent per-entity expansion.
We obtain such hyper-entities through a three-stage process.
First, we perform multi-source graph traversal to propagate signals
from all input entities simultaneously. Second, we rank candidate
convergence nodes using rank- and topology-aware scores, select-
ing the most informative hyper-entities. The selected nodes are
then passed to the LLM for final reasoning.
We instantiate GATHER on a self-constructed cell-centric bi-
ological knowledge graph and apply it to zero-shot cell-type an-
notation. Experiments on two datasets (Immune and Lung) show
that GATHER achieves the highest exact-match accuracy (27.45%
and59.64%) with only one LLM call per sample, outperforming all
KG-RAG baselines while using2–61 ×fewer LLM calls than KG-
RAG baselines. These results demonstrate that convergence nodes
compress multi-entity signals into compact, high-information evi-
dence that conveys more per item than multi-hop paths, making
convergence-centric retrieval an effective and efficient principle
for hyper-entity reasoning. Code is available at https://github.com/
SUAT-AIRI/GATHER.
2 Method
2.1 Task Formulation
We consider zero-shot cell-type annotation under a RAG framework.
We use zero-shot in the training-free sense: no model is trained or
fine-tuned on labeled cells from the evaluation datasets. The method
still relies on curated prior knowledge in VCKG and ranked gene
lists, and is therefore best understood as knowledge-driven, training-
free inference. Given a cell’s gene expression profile, we construct a
cell sentenceS={𝑔 1,...,𝑔𝑛}following Cell2Sentence [ 17], where
genes are ranked by discriminative power. We normalize gene
symbols against VCKG Gene node symbols and synonyms and map
Figure 1: Divergent vs. convergent retrieval for hyper-entity
queries. (a) Divergent: per-entity LLM-guided expansion.
(b) Convergent (GATHER): multi-source traversal identify-
ing topological convergence points.
them to canonical Gene nodes. Uninformative housekeeping genes
(e.g., RPL* ,MT-* ) and symbols without matching nodes are filtered,
yielding the grounded gene set ˜S. The objective is to predict a cell
type𝑐∗∈C, whereCis defined by the Cell Ontology.
In the RAG paradigm, prediction consists of two stages: (i) retriev-
ing relevant knowledge from a knowledge graph, and (ii) LLM-based
reasoning over the retrieved evidence.
As discussed in the Introduction, hyper-entity queries differ fun-
damentally from few-entity settings: the correct cell type emerges
from thejoint supportof many genes. Accordingly, we reformulate
retrieval as amulti-source convergence problem: instead of expand-
ing from each entity independently, the goal is to identify graph
nodes jointly supported by multiple genes in ˜S.
2.2 GATHER: Convergence-Centric Retrieval
Building upon this reformulation, we proposeGATHER, a retrieval
algorithm designed for hyper-entity queries, as shown in Figure 2.
Rather than performing local expansion from each gene, GATHER
identifiestopological convergence points— nodes that receive strong
structural support from many source genes. These nodes act as
hyper-entities, serving as consolidated evidence that captures global
interactions among genes. Here, a hyper-entity is not a new biolog-
ical entity type; it denotes a retrieved graph node whose relevance
is defined by joint support from a set of source entities rather than
by an individual source alone.
GATHER obtains such hyper-entities in three compact stages:
(1) multi-source traversal, which propagates from grounded genes
through the graph and records shared reachability patterns; (2) gene
weighting, which combines gene rank with graph specificity; and
(3) convergence scoring, which aggregates hop-binned support to
select the final evidence nodes.

GATHER: Convergence-Centric Hyper-Entity Retrieval for Zero-Shot Cell-Type Annotation SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
2.2.1 Stage 1: Multi-Source Graph Traversal.For each 𝑔∈ ˜S, we
traverse the knowledge graph up to 𝑘hops in a relation-agnostic
manner (all edge types, both directions). A semantic-type constraint
prevents consecutive nodes of identical type, avoiding degenerate
chains.
Crucially, traversals from all genes proceed simultaneously. For
each discovered target node 𝑡(candidate cell-type node), we record
its hop-binned support:
𝑆ℎ(𝑡)={𝑔∈ ˜S|𝑔reaches𝑡in exactlyℎhops}.(1)
Nodes jointly reachable from many genes through short paths
naturally emerge as candidate convergence points.
2.2.2 Stage 2: Context-Aware Gene Weighting.Stage 1 identifies
which genes can support each candidate target, but raw support
counts treat all genes equally. This is undesirable because top-
ranked genes in the cell sentence are more discriminative, whereas
broadly connected genes may reach many targets and provide less
specific evidence. We therefore assign each gene 𝑔a combined
weight with two components.
Rank-based importance:
𝑤rank
𝑔=1
log2(rank(𝑔)+2).(2)
Graph specificity (IDF-style):
𝑤IDF
𝑔=log|T|
df(𝑔)+1
,(3)
where df(𝑔) is the number of candidate targets reachable from
𝑔, andTis the union of all discovered targets.
The rank term favors salient genes in the cell sentence, whereas
the IDF term downweights genes that reach many candidate targets.
Together, they make the subsequent convergence score depend on
selective, high-rank support rather than raw reachability.
2.2.3 Stage 3: Topology-Aware Convergence Scoring.We rank each
candidate node𝑡by aggregating weighted support:
Score(𝑡)=𝑘∑︁
ℎ=1𝛼ℎ∑︁
𝑔∈𝑆ℎ(𝑡)𝑤rank
𝑔·𝑤IDF
𝑔,(4)
where𝛼ℎare hop-decay weights favoring shorter paths.
This scoring function directly operationalizes the convergence
principle: nodes jointly supported by many informative and specific
genes through short paths receive higher scores.
The top-𝐾convergence nodes, together with their supporting
gene coalitions, form a compressed evidence context that is passed
to a single LLM call for final reasoning.
Compared to entity-wise local expansion, GATHER provides:
(1)Global coordination:explicitly modeling joint structural sup-
port across genes; (2)Context compression:distilling 𝑁source
genes into𝐾≪𝑁 convergence nodes; (3)LLM cost reduction:
requiring zero LLM calls during retrieval. Thus, retrieval shifts from
fragmented local exploration to structured global aggregation.
The retrieval cost is governed by the number of grounded genes,
the traversal horizon, and the local graph fan-out. In the worst
case, a naive traversal grows as 𝑂(|˜S|𝑑𝑘)for average fan-out 𝑑, but
GATHER uses a shallow fixed horizon and ranks only candidate cell-
type targets. The IDF term is therefore traversal-specific: df(𝑔) is
Figure 2: The three stages of GATHER: (1) multi-source graph
traversal from the grounded gene set ˜S, where𝑔denotes
a source gene and 𝑡a candidate target; (2) context-aware
gene weighting (node size encodes 𝑤rank
𝑔; color depth encodes
𝑤IDF
𝑔); and (3) topology-aware convergence scoring, where the
top-𝐾candidates ranked by Score(𝑡) are selected as hyper-
entities.
computed under the traversal horizon 𝑘used for a given run, rather
than as a global gene frequency or a per-sample tuned quantity.
2.3 VCKG as a Cell-Centric Knowledge Graph
for Hyper-Entity Retrieval
GATHER requires a knowledge graph where genes serve as entry
points, functional and cellular semantics enable multi-hop paths to
cell types, and cell-type nodes are grounded in a formal ontology. No
existing public KG satisfies all three requirements: general-purpose
biomedical KGs (e.g., PrimeKG [ 5]) lack a cell-centric schema, while
domain-specific resources (e.g., CellMarker [ 22]) are flat databases
without the graph topology needed for multi-hop convergence.
We therefore constructVCKG, a cell-centric biological knowl-
edge graph that integrates 20+ databases and 7 domain ontolo-
gies through a four-step pipeline: (1)data collectionfrom sources
spanning genes (NCBI Gene, HGNC, UniProt), functions (GO [ 3],
Reactome), cells (Cell Ontology, CellMarker 2.0 [ 11]), anatomy
(UBERON), and diseases (DO, MONDO, HPO); (2)ontology normal-
ization, mapping every entity to its canonical identifier to resolve
synonyms and cross-database conflicts; (3)relation standardization
via the Relation Ontology; and (4)graph assemblyinto a Neo4j prop-
erty graph. To reduce cross-source ambiguity, gene symbols and
aliases are normalized against HGNC and NCBI Gene identifiers,
protein-level references are linked through UniProt accessions, and
cell types are grounded to Cell Ontology IDs. When multiple sources
describe the same entity, we merge by canonical identifier and pre-
serve source-specific fields, such as field_sources ,source , or

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Zhang et al.
PMID metadata, when available. Because direct marker annota-
tions are incomplete, VCKG supports both direct IS_MARKER_FOR
evidence and indirect functional or ontological paths, allowing
GATHER to exploit convergence beyond one-hop marker lookup.
Table 1: Key statistics of VCKG.
Metric Value Metric Value
Total Nodes 120K+ Total Edges 2,500K+
Node Types 9 Edge Types 14
Ontologies 7 Data Sources 20+
Avg. Node Degree∼21 Max Path Length 5
The key design principle isgene-centric structural connectivity:
43,000+ gene nodes act as hubs linking to functional concepts (GO
terms, pathways) and 2,500+ Cell Ontology cell-type nodes. A gene
connects to a cell type either directly via IS_MARKER_FOR (1-hop)
or indirectly through shared annotations, e.g., GenePARTICIPATES_IN−−−−−−−−−−−−−→
BPCAPABLE_OF←−−−−−−−−−CellType (2-hop), enabling convergence signals
from both direct marker evidence and indirect functional overlap.
Table 1 summarizes the key statistics.
3 Experiments
3.1 Experimental Setup
Datasets.We evaluate on two single-cell datasets whose labels
adopt standardized Cell Ontology nomenclature, enabling direct
alignment with VCKG’s ontology-grounded cell type nodes: (1)Im-
mune Human[ 8]: a subset from the cross-tissue immune cell
atlas (donors A29 and A31, following the Cell2Sentence [ 17] split),
comprising 2,962 test cells across 34 fine-grained types; (2)Tabula
Sapiens Lung[ 6]: the lung tissue partition from the Tabula Sapiens
multi-organ atlas, comprising 2,416 test cells across 30 types. Each
cell is converted to a cell sentence of 𝑛=50ranked genes (source
entities), and the number of retained convergence nodes is set to
𝐾=10. In all main experiments, we set the traversal horizon to 𝑘=2
to balance coverage and specificity, while the GATHER formulation
supports other traversal horizons.
Baselines.We compare GATHER against representative base-
lines spanning direct LLM prompting and diverse KG-based RAG
paradigms.LLMperforms direct prompting on gene lists with-
out external knowledge. Among KG-based methods, existing ap-
proaches predominantly adopt alocal, entity-wise expansionpar-
adigm, where each gene is explored independently and evidence
is aggregated afterward.RoG[ 13] follows a template-based strat-
egy by predicting relation paths and performing per-entity traver-
sal.ToG[ 18] and its enhanced variantToG-2[ 14] perform itera-
tive, LLM-guided entity-level exploration with dynamic pruning.
PoG[ 19] conducts multi-stage path-centric search with LLM-based
refinement. All methods use the same LLM backbone (GPT-4o-mini)
and VCKG as the knowledge source for fair comparison.
Metrics.We report exact-match accuracy, average LLM calls per
sample, and evidence quantity (average number of evidence items
retrieved per sample). To account for the hierarchical nature ofcell types, we additionally evaluate against the Cell Ontology DAG:
Ancestor Match (Anc.) credits predictions that lie on the same root-
to-leaf path as the true label (i.e., one is an ancestor or descendant of
the other). We focus on exact and ontology-aware matching metrics,
as they directly reflect whether retrieval succeeds in identifying the
correct semantic region in the Cell Ontology, rather than smoothing
errors through label averaging.
Table 2: Main results on cell-type annotation. Exact: exact-
match accuracy (%); Anc.: ancestor match (%), where a pre-
diction is credited if it is an ancestor or descendant of the
true label in the Cell Ontology; Calls: average LLM calls per
sample; Evid.: average evidence items per sample (conver-
gence nodes for GATHER; relation paths for RoG; reasoning
triples for ToG/ToG-2; retrieved paths for PoG). All methods
use GPT-4o-mini and VCKG.
Immune(34 types)Lung(30 types)
Method Exact Anc. Calls Evid. Exact Anc. Calls Evid.
LLM 14.01 26.40 1.0 — 54.88 54.97 1.0 —
RoG 17.39 31.53 2.0 12.8 56.37 56.79 2.0 24.8
ToG-2 18.13 29.03 13.3 18.8 53.35 53.39 12.6 10.9
PoG 20.80 33.15 15.9 6.8 48.59 48.80 8.2 7.8
ToG 20.50 36.61 56.2 20.0 56.04 56.21 60.5 20.0
GATHER 27.4533.091.09.359.64 60.18 1.010.0
+Path 26.54 31.94 1.0 9.3 58.65 59.19 1.0 10.0
3.2 Main Results
Overall Performance and Efficiency.From Table 2, GATHER achie-
ves the highest exact-match accuracy on both datasets:27.45%on
Immune (vs. 20.80% for PoG) and59.64%on Lung (vs. 56.37% for
RoG). Notably, these gains are obtained with only 1 LLM call per
sample, whereas KG-RAG baselines require between 2.0 and 60.5
LLM calls. Although increased LLM interaction generally correlates
with improved accuracy (e.g., ToG-2: 13.3 calls →18.13% on Im-
mune; ToG: 56.2 calls→20.50%), GATHER surpasses all baselines
without additional LLM reasoning steps. Beyond LLM efficiency,
GATHER also achievesevidence compression. Although GATHER
retrieves 9.3–10.0 evidence items, comparable in count to baselines
such as PoG (6.8–7.8 paths), the information granularity differs
fundamentally: each baseline evidence item is amulti-hop path
consisting of several nodes and edges (e.g., a 2-hop path contains
3 nodes and 2 relations), whereas each GATHER evidence item is
a singleconvergence nodethat aggregates structural support from
multiple source genes. Thus, in terms of total information volume
fed to the LLM, GATHER uses substantially less knowledge than
path-based methods while achieving the highest exact-match ac-
curacy. This demonstrates that convergence modeling produces
inherently more informative evidence per item, and that retrieval
quality—not quantity—drives performance.
Cross-Dataset Analysis.Performance patterns differ between the
two datasets. Lung yields higher absolute accuracy for all methods
(e.g., LLM baseline: 54.88% on Lung vs. 14.01% on Immune), reflect-
ing its relatively coarser and more transcriptionally distinct cell
types. Immune, containing fine-grained T-cell subtypes, is more sen-
sitive to multi-gene interactions. Despite this increased difficulty,

GATHER: Convergence-Centric Hyper-Entity Retrieval for Zero-Shot Cell-Type Annotation SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia
GATHER maintains the best exact-match accuracy in both regimes.
On Lung, it also achieves the highest ancestor match (60.18%), in-
dicating correct placement within the ontology hierarchy. On Im-
mune, while the ancestor match (33.09%) is below ToG (36.61%), the
higher exact accuracy suggests that GATHER favors precise sub-
type predictions over conservative coarse-grained guesses. These
observations support that convergence modeling is particularly
beneficial in fine-grained hyper-entity settings, where the label
cannot be resolved from any single gene and must instead emerge
from multi-gene structural agreement.
Ablation and Scaling Analysis.The path ablation further clarifies
the effective signal. When explicit traversal paths are added, accu-
racy decreases from 27.45% to 26.54% on Immune and from 59.64%
to 58.65% on Lung. This suggests that the LLM primarily benefits
fromwhich genes converge on a candidate and at what distance,
rather than verbose path descriptions, reinforcing that topological
convergence is the core signal.
Gene-scaling results (Table 3) show that exact-match accuracy
generally improves as the number of input genes grows. On Lung,
accuracy rises from 55.46% (10 genes) to 59.64% (50 genes); on Im-
mune, it improves from 24.85% (10 genes) to 27.72% (40 genes) and
remains comparable at 27.45% with 50 genes. This trend suggests
that additional genes strengthen the convergence signal until it sat-
urates rather than dilute it. In contrast, traditional entity-wise meth-
ods do not exhibit comparable scaling gains, as their retrieval treats
each gene independently. Finally, varying the number of retained
convergence nodes ( 𝐾∈{ 5,10,15}) yields similar performance
(Immune: 26.87%/27.45%/27.04%; Lung: 59.48%/59.64%/59.81%), with
𝐾=10achieving a good balance across both datasets, indicating that
convergence signals are stably concentrated among a small set of
top candidates.
Table 3: Effect of input gene count on GATHER (GPT-4o-
mini). Genes: number of input genes from the cell sentence;
Grounded: genes successfully mapped to VCKG nodes; Exact:
exact-match accuracy (%).
Immune Lung
Genes Grounded Exact Grounded Exact
10 10.0 24.85 10.0 55.46
20 20.0 26.03 20.0 57.12
30 30.0 26.84 30.0 58.11
40 40.0 27.72 40.0 59.44
50 48.8 27.45 47.7 59.64
3.3 Discussion and Limitations
Scope of Comparison.This work focuses on training-free KG-
RAG retrieval rather than supervised cell-type classification. Non-
KG methods such as linear classifiers, PCA-based workflows, set-
based encoders, and graph neural networks are important comple-
mentary baselines, but they typically require labeled cells, feature
learning, or task-specific training. Comparing these paradigms un-
der a unified data and supervision protocol is an important direction
for future work.Dependence on VCKG..GATHER assumes that the underlying
graph contains meaningful convergence targets and that cell-type
nodes are sufficiently grounded in ontology and marker evidence.
Its performance may degrade when the true cell type is absent from
the graph, when marker coverage is sparse, or when input genes
cannot be reliably grounded. Applying the method to a general
biomedical KG such as PrimeKG would require additional schema
alignment, target typing, and cell-type grounding, because such
KGs are not organized around cell-type retrieval.
Evidence and Failure Modes.Convergence nodes should be in-
terpreted as retrieval evidence, not as direct proof of a biological
mechanism. On fine-grained immune subtypes, GATHER improves
exact match, but can still make overly specific errors, as reflected
by its lower ancestor match than ToG. Future work should report
top-𝐾retrieval ceilings, per-class confusion patterns, expert vali-
dation of retrieved convergence nodes, and robustness to missing
genes or noisy graph edges.
Parameter Sensitivity.Our main experiments use 𝑛=50input
genes and a two-hop traversal horizon as a compact default set-
ting. The gene-count analysis suggests that convergence signals
strengthen once sufficient source genes are available, but the best
traversal depth and relation constraints may vary across knowledge
graphs and annotation granularity. Future work should study when
additional graph context improves coverage or introduces overly
broad or noisy convergence targets.
4 Conclusion
We introducedGATHER, a convergence-centric retrieval frame-
work for hyper-entity queries in KG-RAG, where answers emerge
from the collective support of many entities. Instead of independent
entity-wise expansion, GATHER detects topological convergence
points to model multi-source structural synergy during retrieval.
Across two benchmarks, GATHER achieves the best exact-match
accuracy (27.45% on Immune and 59.64% on Lung) while requiring
only a single LLM call per sample, reducing LLM usage by 2–61 ×
compared to KG-RAG baselines. These results demonstrate that
convergence nodes compress collective entity signals into compact,
high-information evidence, and that improving retrieval quality,
rather than increasing evidence volume or iterative LLM reasoning,
is key to effective hyper-entity modeling. More broadly, our find-
ings highlight evidence informativeness as a key factor in zero-shot
biomedical reasoning with collective entity signals.
Acknowledgments
This work was supported by the project of Shenzhen Applica-
tion Research and Development Special Fund Support (Grant No.
XLQSQ20250427092505008), the National Key Research and Devel-
opment Program of China (Grant No. 2024YFF0908200), the Natural
Science Foundation of Guangdong Province of China (Grant Nos.
2024A1515030166 and 2025B1515020032), the Shenzhen Science and
Technology Innovation Program (Grant No. KQTD20190929172835-
662), and the Innovation Team Project of Guangdong Province
(Grant No. 2024KCXTD017).

SIGIR ’26, July 20–24, 2026, Melbourne, VIC, Australia Zhang et al.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774.
doi:10.48550/arXiv.2303.08774
[2]Dvir Aran, Agnieszka P Looney, Leqian Liu, Esther Wu, Valerie Fong, Austin
Hsu, Suzanna Chak, Ram P Naikawadi, Paul J Wolters, Adam R Abate, et al .2019.
Reference-based analysis of lung single-cell sequencing reveals a transitional
profibrotic macrophage.Nature immunology20, 2 (2019), 163–172.
[3]Michael Ashburner, Catherine A Ball, Judith A Blake, David Botstein, Heather
Butler, J Michael Cherry, Allan P Davis, Kara Dolinski, Selina S Dwight, Janan T
Eppig, et al .2000. Gene ontology: tool for the unification of biology.Nature
genetics25, 1 (2000), 25–29.
[4]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al .2020. Language models are few-shot learners.Advances in neural
information processing systems33 (2020), 1877–1901.
[5] Payal Chandak, Kexin Huang, and Marinka Zitnik. 2023. Building a knowledge
graph to enable precision medicine.Scientific Data10, 1 (2023), 67.
[6]The Tabula Sapiens Consortium*, Robert C Jones, Jim Karkanias, Mark A Krasnow,
Angela Oliveira Pisco, Stephen R Quake, Julia Salzman, Nir Yosef, Bryan Bulthaup,
Phillip Brown, et al .2022. The Tabula Sapiens: A multiple-organ, single-cell
transcriptomic atlas of humans.Science376, 6594 (2022), eabl4896.
[7]Haotian Cui, Chloe Wang, Hassaan Maan, Kuan Pang, Fengning Luo, Nan Duan,
and Bo Wang. 2024. scGPT: toward building a foundation model for single-cell
multi-omics using generative AI.Nature methods21, 8 (2024), 1470–1480.
[8]C Domínguez Conde, Chao Xu, Louie B Jarvis, Daniel B Rainbow, Sara B Wells,
Tamir Gomes, SK Howlett, O Suchanek, K Polanski, HW King, et al .2022. Cross-
tissue immune cell analysis reveals tissue-specific features in humans.Science
376, 6594 (2022), eabl5197.
[9]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-
Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards retrieval-
augmented large language models. InProceedings of the 30th ACM SIGKDD
conference on knowledge discovery and data mining. Association for Computing
Machinery, New York, NY, USA, 6491–6501. doi:10.1145/3637528.3671470
[10] Wenpin Hou and Zhicheng Ji. 2024. Assessing GPT-4 for cell type annotation in
single-cell RNA-seq analysis.Nature methods21, 8 (2024), 1462–1465.
[11] Congxue Hu, Tengyue Li, Yingqi Xu, Xinxin Zhang, Feng Li, Jing Bai, Jing Chen,
Wenqi Jiang, Kaiyue Yang, Qi Ou, et al .2023. CellMarker 2.0: an updated database
of manually curated cell markers in human/mouse and web tools based on scRNA-
seq data.Nucleic acids research51, D1 (2023), D870–D876.
[12] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[13] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning
on Graphs: Faithful and Interpretable Large Language Model Reasoning. InInternational Conference on Learning Representations. OpenReview.net, Vienna,
Austria, 14400–14423.
[14] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang,
Jiaxin Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large
Language Model Reasoning with Knowledge-guided Retrieval Augmented Gen-
eration. InThe Thirteenth International Conference on Learning Representations.
OpenReview.net, Singapore, 52782–52806.
[15] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu.
2024. Unifying large language models and knowledge graphs: A roadmap.IEEE
Transactions on Knowledge and Data Engineering36, 7 (2024), 3580–3599.
[16] Giovanni Pasquini, Jesus Eduardo Rojo Arias, Patrick Schäfer, and Volker
Busskamp. 2021. Automated methods for cell type annotation on scRNA-seq
data.Computational and Structural Biotechnology Journal19 (2021), 961–969.
[17] Syed Asad Rizvi, Daniel Levine, Aakash Patel, Shiyang Zhang, Eric Wang, Cur-
tis Jamison Perry, Nicole Mayerli Constante, Sizhuang He, David Zhang, Cerise
Tang, et al .2025. Scaling large language models for next-generation single-cell
analysis. 2025–04 pages.
[18] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph: Deep
and Responsible Reasoning of Large Language Model on Knowledge Graph. In
The Twelfth International Conference on Learning Representations. OpenReview.net,
Vienna, Austria, 3868–3898.
[19] Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, and Wenjie Zhang.
2025. Paths-over-graph: Knowledge graph empowered large language model
reasoning. InProceedings of the ACM on Web Conference 2025(Sydney NSW,
Australia)(WWW ’25). Association for Computing Machinery, New York, NY,
USA, 3505–3522. doi:10.1145/3696410.3714892
[20] Christina V Theodoris, Ling Xiao, Anant Chopra, Mark D Chaffin, Zeina R
Al Sayed, Matthew C Hill, Helene Mantineo, Elizabeth M Brydon, Zexian Zeng,
X Shirley Liu, et al .2023. Transfer learning enables predictions in network
biology.Nature618, 7965 (2023), 616–624.
[21] Fan Yang, Wenchuan Wang, Fang Wang, Yuan Fang, Duyu Tang, Junzhou Huang,
Hui Lu, and Jianhua Yao. 2022. scBERT as a large-scale pretrained deep language
model for cell type annotation of single-cell RNA-seq data.Nature Machine
Intelligence4, 10 (2022), 852–866.
[22] Xinxin Zhang, Yujia Lan, Jinyuan Xu, Fei Quan, Erjie Zhao, Chunyu Deng, Tao
Luo, Liwen Xu, Gaoming Liao, Min Yan, et al .2019. CellMarker: a manually
curated resource of cell markers in human and mouse.Nucleic acids research47,
D1 (2019), D721–D728.
[23] Suyuan Zhao, Jiahuan Zhang, Yushuai Wu, Yizhen Luo, and Zaiqing Nie. 2024.
LangCell: Language-Cell Pre-training for Cell Identity Understanding. InInterna-
tional Conference on Machine Learning (Proceedings of Machine Learning Research,
Vol. 235). PMLR, Vienna, Austria, 61159–61185.
[24] Yuqi Zhu, Xiaohan Wang, Jing Chen, Shuofei Qiao, Yixin Ou, Yunzhi Yao, Shumin
Deng, Huajun Chen, and Ningyu Zhang. 2024. Llms for knowledge graph con-
struction and reasoning: Recent capabilities and future opportunities.World
Wide Web27, 5 (2024), 58.