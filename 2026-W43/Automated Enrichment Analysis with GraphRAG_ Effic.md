# Automated Enrichment Analysis with GraphRAG: Efficient Multi-Database ORA Extraction and Biological Interpretation

**Authors**: Guangcan Li, Shuyuan Zheng, Jiayao Liu, Shigeto Seno, Makoto Onizuka

**Published**: 2026-10-22 04:29:25

**PDF URL**: [https://arxiv.org/pdf/2610.30710v1](https://arxiv.org/pdf/2610.30710v1)

## Abstract
Over-representation analysis (ORA) is widely used to interpret gene-expression studies, but its outputs are often fragmented across gene-set databases and presented as flat term lists that obscure gene-level links among related enrichment signals. We propose Graph-Based Over-Representation Retrieval-Augmented Generation (GBOR-RAG), an integrated and traceable framework for ORA interpretation. GBOR-RAG represents heterogeneous gene-set databases as a unified term--gene knowledge graph, where shared genes connect related terms across sources. After graph-based ORA scoring, the framework retrieves a compact connected evidence subgraph that preserves both cross-database relationships and gene-level support. The subgraph conditions an LLM to generate biological interpretations, and we introduce a claim-level faithfulness metric to assess whether generated claims are supported by explicit evidence. Experiments show that graph-based ORA preserves conventional ORA results with high consistency (mean Pearson $r=0.997$) while reducing query time by a median of $37\times$ (up to $48\times$). The retrieved subgraphs average 36.8 nodes and achieve the highest overall integration score (4.12), and GBOR-RAG attains the highest claim-level faithfulness (0.90) among the compared systems. Adding a PubMed literature layer further improves the overall interpretation quality from 4.17 to 4.75.

## Full Text


<!-- PDF content starts -->

Automated Enrichment Analysis with GraphRAG:
Efficient Multi-Database ORA Extraction and
Biological Interpretation
Guangcan Li∗, Shuyuan Zheng∗, Jiayao Liu†, Shigeto Seno∗, Makoto Onizuka∗
∗Graduate School of Information Science and Technology, The University of Osaka, Japan
{guangcan.li, zheng, senoo, onizuka}@ist.osaka-u.ac.jp
†The First Affiliated Hospital of China Medical University, China
liujiayao@cmu.edu.cn
Abstract—Over-representation analysis (ORA) is widely used
to interpret gene-expression studies, but its outputs are often
fragmented across gene-set databases and presented as flat
term lists that obscure gene-level links among related enrich-
ment signals. We propose Graph-Based Over-Representation
Retrieval-Augmented Generation (GBOR-RAG), an integrated
and traceable framework for ORA interpretation. GBOR-RAG
represents heterogeneous gene-set databases as a unified term–
gene knowledge graph, where shared genes connect related terms
across sources. After graph-based ORA scoring, the framework
retrieves a compact connected evidence subgraph that preserves
both cross-database relationships and gene-level support. The
subgraph conditions an LLM to generate biological interpreta-
tions, and we introduce a claim-level faithfulness metric to assess
whether generated claims are supported by explicit evidence.
Experiments show that graph-based ORA preserves conventional
ORA results with high consistency (mean Pearsonr= 0.997)
while reducing query time by a median of37×(up to48×).
The retrieved subgraphs average 36.8 nodes and achieve the
highest overall integration score (4.12), and GBOR-RAG attains
the highest claim-level faithfulness (0.90) among the compared
systems. Adding a PubMed literature layer further improves the
overall interpretation quality from 4.17 to 4.75.
Index Terms—over-representation analysis, knowledge graph,
graph-based retrieval-augmented generation, large language
models
I. INTRODUCTION
Over-representation analysis (ORA) is one of the most
widely used methods for interpreting gene-expression profiles:
it tests whether predefined gene sets are over-represented
among differentially expressed genes relative to a background
universe [1]. ORA relies on a rich ecosystem of curated gene-
set resources, including Gene Ontology (GO) [2], pathway
databases such as KEGG [3], Reactome [4], and WikiPathways
[5], integrative collections such as MSigDB [6], and disease-
association resources such as DisGeNET [7].
Despite its maturity, ORA still has two practical limitations.
The first is limited cross-database evidence integration. Con-
ventional ORA analyzes each database separately, leaving the
comparison of their terms (database-defined gene sets such
as pathways or biological processes) to the reader. Although
recent tools connect or cluster terms using shared genes
or semantic similarity [8]–[10], their outputs often remainfragmented or coarse-grained, making it difficult to trace how
related signals are connected through specific genes. As more
databases are included, the cost of cross-database analysis also
increases, raising scalability concerns. Second, current ORA
interpretation pipelines rarely verify whether each generated
biological claim is supported by the underlying enrichment
evidence, a property we refer to as claim-level faithfulness.
Conventional ORA returns ranked enrichment tables rather
than coherent biological explanations, while visualization tools
[11], [12] organize enriched terms into networks but still
leave the biological reading to the user. Recent LLM- and
RAG-based systems generate textual interpretations, but they
are usually evaluated by semantic similarity [13], manual
inspection [14] or reproducibility [9]. This can be problematic
in biomedical interpretation because unsupported gene–term
associations can lead to misleading conclusions.
To address these two limitations, we proposeGraph-
Based Over-Representation RAG(GBOR-RAG)1, a graph-
based RAG framework for ORA (Fig. 1). For cross-database
evidence integration, we formulate the task as prize-collecting
subgraph extraction over a multi-database knowledge graph
[15], [16]. Instead of simply concatenating the top-ranked
terms from individual databases, our method retrieves a
compact query-relevant subgraph that links significant terms
through their shared significant genes. This representation
preserves the gene-level evidence underlying each term and
can surface bridge signals that connect related biological
processes across databases, even when those signals are not
among the top-ranked results in any single database. For
quality of textual interpretation, we define a claim-level faith-
fulness metric to quantify whether each generated biological
claim is supported by the retrieved evidence. Furthermore,
we introduce an advanced variant of GBOR-RAG that adds
a PubMed citation layer to the knowledge graph, providing
gene–literature evidence that enriches the biological context
used for textual interpretation.
We make these contributions:
•We implement ORA on a term–gene knowledge graph
1Our source code is available at https://github.com/ggnb367/GBOR-RAG.
arXiv:2610.30710v1  [cs.IR]  22 Oct 2026

Fig. 1. Overview of GBOR-RAG. Significant genes are split by log-fold-change sign; each directional set is scored by graph-based ORA on the multi-database
knowledge graph, enriched terms and PubMed literature nodes become node prizes, and prize-collecting retrieval extracts a compact evidence subgraph per
direction that conditions the LLM interpretation.
spanning 7 databases, reproducing conventional ORA
with high consistency (mean Pearsonr= 0.997) while
reducing query time by a median of37×(up to48×).
•We formulate cross-database ORA integration as compact
connected subgraph retrieval, producing smaller evidence
subgraphs (36.8 nodes on average) with full connectivity
and the highest overall integration score (4.12).
•We introduce a claim-level faithfulness metric for evaluat-
ing whether generated ORA interpretations are supported
by their retrieved evidence, with GBOR-RAG achieving
the highest score (0.90) among the compared systems.
•We extend GBOR-RAG with a PubMed literature layer,
improving textual interpretation quality from 4.17 to 4.75.
II. PRELIMINARIES
A. Over-Representation Analysis
Over-representation analysis (ORA) tests whether a prede-
fined gene set is enriched among a list of genes of interest.
LetUbe the background gene universe withN=|U|,S⊆U
the significant (differentially expressed) genes withn=|S|,
andG⊆Uthe genes annotated to a term, withK=|G|and
k=|G∩S|. ORA evaluates the one-tailed hypergeometric
probability of observing at leastkhits by chance,
p=P(X≥k) =min(K,n)X
i=kn
iN−n
K−i
N
K ,(1)
and a smallp-value marks the term as enriched; this test is
the statistical core of widely used enrichment tools [1], [17].
In expression studies, significant genes are further split by the
sign of their log fold change∆ g,
S±={g∈U|adj.p(g)< α DE,±∆ g> δ},(2)whereδ≥0is a log fold change threshold (differential
expression by limma [18], withα DE= 0.05andδ= 0in
our experiments, following [19]), and ORA is run separately
onS+andS−against the sameU.
B. Graph-Based Retrieval-Augmented Generation
Retrieval-augmented generation (RAG) grounds a language
model in retrieved evidence rather than parametric mem-
ory alone, reducing hallucination [20]. Graph-based RAG
retrieves a subgraph whose nodes and edges the model can
cite directly [21]. G-Retriever [15] formalizes this as Prize-
Collecting Steiner Tree (PCST) extraction [16], selecting a
compact connected subgraph that collects high-value nodes
while paying for the edges that connect them. We adopt this
prize-collecting view but retrieve a general connected evidence
subgraph, rather than a spanning tree, from the term–gene
graph, and condition the LLM on it.
III. RELATEDWORK
Cross-database integration.Enrichr-KG [8] links terms
across libraries and EnrichGT [10] groups pathways with
overlapping gene content, providing useful term-centered net-
works or clusters for exploring multi-database outputs. How-
ever, when the selected terms are determined by fixed top-
krankings, the resulting evidence may remain fragmented or
may not include intermediate bridge terms that connect related
signals across databases. In contrast, we retrieve a compact,
connected, gene-anchored subgraph that explicitly represents
these cross-database links.
Natural-language interpretation.Beyond visualization
tools [11], [12], recent LLM-based systems move toward
automated textual interpretation: GeneAgent [13] self-verifies
claims against curated databases, MAPA [9] annotates clus-
tered pathway modules, and Huet al.[14] benchmark LLMs
for gene-set function interpretation, but claim support is rarely
evaluated against the exact evidence provided to the model.

We therefore condition the LLM on the retrieved gene–term
subgraph and assess claim-level faithfulness with respect to
that evidence.
IV. GRAPH-BASEDOVER-REPRESENTATIONRAG
Those limitations of cross-database integration and quality
of interpretation partly stem from the database-wise design of
conventional ORA: biologically related terms across databases
may share genes, but such links are not directly represented in
isolated enrichment outputs. To address this, we build a multi-
database term–gene knowledge graph and retrieve a compact,
query-relevant subgraph that connects enriched terms through
shared genes. This graph-structured evidence provides explicit
support for LLM-based interpretation, improving traceability
and enabling claim-level faithfulness to be assessed.
This section follows the pipeline: we build the term–
gene graph (Section IV-A), formulate integration as subgraph
retrieval on multi-database knowledge graph (Section IV-B),
score terms by graph-based ORA (Section IV-C), convert
the scores into node prizes and retrieve a compact con-
nected evidence subgraph by prize-collecting optimization
(Section IV-D), and condition an LLM on it to generate a
grounded interpretation (Section IV-E). Significant genes are
split by the sign of their log fold change and one subgraph
is retrieved per pole, so the interpretation contrasts terms
enriched among up- and down-regulated genes. We call this
base pipelinenaive GBOR-RAG, and extend its retrieval with
a PubMed literature layer (Section IV-F) to obtainadvanced
GBOR-RAG.
A. Multi-Database Knowledge Graph
We construct a heterogeneous multi-database knowledge
graph that integrates the gene-set resources listed in Table I,
covering pathways, biological processes, disease associations,
and drug-target associations. The curated libraries are taken
from Enrichr [22], and drug-target associations from Open
Targets Platform release 25.03 [23].
For each sources∈ D, letT sbe its terms andA s(t)the
genes annotated to termt. We represent all annotations as a
bipartite graphG= (V T∪VG, E), whereV T={(t, s)|t∈
Ts}denotes the set of source-specific term nodes,V Gdenotes
the set of gene nodes, andE={{(t, s), g} |g∈A s(t)}
encodes term–gene membership.
Genes are identified by Entrez ID, with HGNC symbols
retained as metadata. The source attributesdistinguishes terms
from different databases, while shared gene nodes provide the
connections through which evidence from different sources
can meet. The resulting graph contains33,199terms,20,604
genes, and1,100,511term–gene memberships.
B. Problem Formulation
a) Problem.:After ORA assigns each termta sig-
nificance valuep twith respect to the significant gene set
S, the resulting signals remain scattered across heteroge-
neous databases. Because each database yields an independent
ranked list, it is unclear whether terms from different sourcesTABLE I
COMPOSITION OF THE KNOWLEDGE GRAPH. GENES/TERM:MEAN
NUMBER OF GENE NEIGHBORS PER TERM ORPMID;BOTTOM ROWS:
PUBMED LAYER OF ADVANCEDGBOR-RAG (SECTIONIV-F).
Database Content Terms/PMIDs Edges Genes/term
Open Targets [23] Drug-target assoc. 17,457 443,777 25.4
DisGeNET [7] Gene-disease assoc. 6,621 266,752 40.3
GO BP [2] Biological processes 6,036 197,436 32.7
Reactome [4] Reaction pathways 2,093 126,227 60.3
WikiPathways [5] Community pathways 622 26,663 42.9
KEGG Pathway [3] Molecular pathways 320 32,347 101.1
MSigDB Hallmark [6] Hallmark gene sets 50 7,309 146.2
Total (curated)33,199 1,100,511 33.2
PubMed layer [24] Gene citations (PMIDs) 799,126 1,991,461 2.5
Combined832,325 3,091,972 3.7
support the same underlying mechanism. Our goal is therefore
to integrate these separate enrichment signals across databases.
b) Why direct merging fails.:Directly merging terms by
rawp-value is unsuitable because database size, term granular-
ity, and annotation density can bias significance distributions,
causing a global ranking to favor particular resources rather
than balanced cross-database evidence.
c) Formulation.:We formulate cross-database integra-
tion as retrieving an evidence subgraphG ev⊆ Gfrom the
knowledge graphG, containing term nodes relevant toSand
connecting evidence from different databases through shared
genes. We requireG evto satisfy three objectives:
•Relevance.The subgraph should prioritize terms strongly
associated withSwhile keeping heterogeneous databases
balanced, rather than ranking all terms by rawp-values.
•Connectivity.The subgraph should connect terms
through shared genes, so that cross-database integration
reflects overlapping gene-level evidence rather than inde-
pendent ranked lists.
•Compactness.The subgraph should avoid unnecessary
expansion, keeping the retrieved evidence focused and
easier to interpret and verify.
C. Graph-Based ORA
To drive the retrieval, every term needs a relevance score
for the significant genes, which we obtain by computing ORA
directly on the knowledge graph. Each term node already
carries its gene set as its neighborhood in the graph, so we
reuse the hypergeometric test of Eq. (1) and only replace the
predefined gene set of a term with this neighborhood
Gt,s={g∈U| {(t, s), g} ∈E},(3)
where the universeUis the intersection of the genes in the
KG and the measured genes.
The significant genes are split by the sign of their log fold
change into a positive setS+and a negative setS−, and we
score each term against both sets with the same upper-tail
test: the significancep±
tof termtat each pole is Eq. (1) with
G=G t,sandS=S±.
As established in the problem formulation, thep±
tare not
comparable across databases, so we do not pool all terms into
a single global ranking over the knowledge graph. Instead we

rank the terms of each database separately, giving every term
a rank within its own source. This within-database ranking is
the comparable signal that Section IV-D uses to select candi-
date terms for retrieval. We rank by nominalp; Benjamini–
Hochberg adjustment within a database is monotone and leaves
this ranking unchanged.
D. Evidence Subgraph Retrieval
To assemble ranked terms into a single gene-anchored
structure, we formulate retrieval as a prize-collecting subgraph
problem. Term nodes receive prizes according to enrichment
relevance, significant genes receive small connector prizes, and
term–gene edges carry a uniform cost. The objective favors
subgraphs that collect high-prize terms while retaining only
the genes and edges needed to connect them, thereby removing
weakly relevant or isolated nodes and producing a compact
cross-database evidence core.
We instantiate this as follows. From each database we
take its top-ranked terms (Section IV-C) as candidate term
nodes, so that every database offers its strongest evidence and
none floods the graph by sheer size. Cross-database balance
therefore comes from this per-database selection, and the
significance-based prize below only grades candidates within
the balanced pool. A candidate termtreceives a prize graded
by its significance,
π(t) =min 
cmax,−log10pt
Z,(4)
where the capc max limits how much a single term can
contribute andZis the normaliser defined below. A significant
gene receives a small fixed prizeπ 0so it can act as a connector,
any other gene receives no prize, and every term-gene edge
carries a uniform costc 0.
We cap−log10ptatcmax to prevent a single extremely
significant term from dominating the retrieval, and normalize
byZ= max tmin(c max,−log10pt)so that term prizes lie in
[0,1]and are comparable with the gene connector prizeπ 0.
The retrieved evidence subgraph is the connected subgraph
that maximises the collected prize minus the connection cost,
Gev= arg max
G′⊆GX
v∈G′π(v)−X
e∈G′c(e)
.(5)
We solve this objective as an unrooted, single-tree PCST with
the Goemans–Williamson-basedpcst_fastsolver with
strong pruning [16], [25], and induce the evidence subgraph
from the selected nodes.
We run this retrieval once for each pole with the same
prizes and costs. The positive pole uses the positive rankings
with the positive genesS+as connectors and returnsG+
ev, and
the negative pole uses the negative rankings with the negative
genesS−and returnsG−
ev. Both subgraphs are passed to the
interpretation stage of Section IV-E.
E. Graph Retrieval Augmented Interpretation
We condition a large language model on both subgraphs
G+
evandG−
evand constrain it to ground every statement in this
retrieved evidence, to promote claim-level faithfulness.We linearize each subgraph into a textual evidence block.
Within each database the terms are listed in order of their
prize, and every term is followed by the significant genes
that connect it, which form the overlap driving its enrichment.
Because a flat listing hides the topology of the subgraph, we
add two structural cues read directly off the graph, the hub
genes that connect the most terms and the bridging genes that
link terms from different databases.
The positive and negative blocks are presented in parallel,
so the model reports the processes enriched among the positive
genes and, separately, those enriched among the negative
genes, and contrasts the two directions. Both blocks are gen-
uine enrichment signals rather than a presence-versus-absence
comparison.
F . Advanced GBOR-RAG
Naive GBOR-RAG uses only curated gene-set libraries.
To add literature context, we augment the knowledge graph
with a PubMed layer from NCBI, restricted to human genes,
where each article is represented as a PMID node connected
to the genes it cites. Articles citing multiple significant genes
can link these genes with disease, pathway, or drug-related
context beyond curated terms. Table I (bottom rows) reports
the scale of this layer, which covers 26,309 genes (26,506 in
the combined graph).
Unlike curated terms, PMID nodes do not define annotated
gene sets and therefore cannot be scored by ORA. To bring
it into the prize-collecting retrieval of Section IV-D, we need
a prize for each PMID that scales with the term prizes and
reflects how specifically the article relates to the significant
genes.
We obtain this prize from the normalized pointwise mutual
information (NPMI) between being cited by an article and
being significant [26]. WithNgenes in the universe,nof
them significant, and an article citingmuniverse genes of
whichkare significant,
NPMI =ln 
kN/(nm)
−ln 
k/N∈[−1,1].(6)
The denominator normalizes the association to the fixed range
[−1,1], putting a PMID prize on the same bounded scale as
the normalized term prizes of Section IV-D. NPMI rewards a
small, focused article whose cited genes are mostly significant
and penalizes a promiscuous article, whose large gene set
makes the overlap expected by chance, so large screens and
reviews are down-weighted automatically.
The scored PMIDs are added to the retrieval as additional
prize nodes, each connected to the significant genes it co-
cites, with its NPMI as the prize. The retrieval then returns
a richer evidence subgraph in which the curated terms and
their genes are accompanied by the primary articles that link
the driver genes, providing citable mechanistic support for the
interpretation beyond the curated databases.
V. EXPERIMENTS
We evaluate GBOR-RAG on ORA accuracy and efficiency,
cross-database integration, claim-level faithfulness, and lit-

TABLE II
AGREEMENT BETWEEN GRAPH-BASED AND CONVENTIONALORA,
AVERAGED OVER EIGHT QUERIES AND BOTH DIRECTIONS.
Database Pearson Spearman Jaccard@10 F1
KEGG Pathway 0.998 0.996 0.932 0.926
GO BP 0.999 0.979 0.934 0.944
WikiPathways 0.996 0.989 0.913 0.879
Reactome 0.999 0.988 0.943 0.936
MSigDB Hallmark 0.994 0.988 0.898 0.800
DisGeNET 0.998 0.980 0.343 0.727
Open Targets 0.998 0.981 0.873 0.709
Overall 0.997 0.986 0.834 0.846
1 2 3 4 5 6 7
Databases queried (small→large)0200400600Total time, 42 queries (s)
(a) Time vs. collection size
Conventional ORA
Graph-based ORA
time saved
42 GEO queries (sorted)02040Speedup (×)median 37×(b) Speedup, all 7 DBs
Fig. 2. Query-time efficiency on the 42 GEO queries. (a) Total query time
as databases are added from smallest to largest, with shaded savings from
Graph-based ORA. (b) Per-query speedup on all seven databases (sorted),
with the dashed line marking the median.
erature augmentation, under the following common setup.
All 42 real-world queries are from GEO2KEGG [19], with
the Alzheimer’s disease query (GSE1297) used as a running
example. Evidence-subgraph retrieval uses a single fixed con-
figuration across all experiments: the topk= 20candidate
terms per database, prize capc max= 20, gene connector
prizeπ 0= 0.2, and edge costc 0= 0.45. All interpretations
were generated by the open-weight DeepSeek-V4-Pro (API
modeldeepseek-v4-pro, temperature 0.2), while GPT-5-
mini (gpt-5-mini-2025-08-07, temperature 0) served
as the independent LLM judge. The LLM-as-a-judge [27]
rubrics were reviewed by bioinformatics experts and calibrated
on small blinded human-rated sets; thus, the scores are used
as comparative evidence under the same rubric rather than
as absolute measures of biological correctness. Differences
between methods are tested with two-sided paired Wilcoxon
signed-rank tests over queries. Full experimental details are
provided in the GitHub repository.
A. Accuracy and Efficiency of Graph-based ORA
Purpose.This experiment checks that graph-based ORA
is a sound foundation: scoring ORA on the graph should
not change the result (accuracy) and scale to many databases
(efficiency).
Settings.Efficiency uses all 42 queries of GEO2KEGG and
the seven databases of Table I; accuracy compares against
conventional ORA from GSEABenchmarkeR [19] on the eight
queries with a reference, both sharing the same significantgenes and size filter. Accuracy is measured over query-
database-direction pairs by Pearson and Spearman correlation
of−log10(p), by Jaccard@10 on the top-10 terms, and by F1
between the sets of terms with nominalp <0.05; efficiency
is per-query time after the one-time graph build, adding
databases cumulatively.
Accuracy and efficiency.Graph-based ORA closely repro-
duces conventional ORA (Table II), with an overall Pearson
correlation of0.997and database-level correlations of0.994–
0.999; lower top-term overlap in DisGeNET and Open Targets
mainly reflects differences in library version and background
universe between the two implementations rather than in the
test itself. Because it uses the same hypergeometric test on
a pre-built graph index, it preserves conventional ORA while
improving efficiency: on all seven databases, query time drops
from about15.6s to0.4s, a median37×speedup (up to48×)
that grows as databases are pooled (Fig. 2).
B. Quality of Cross-Database Integration
Purpose.We formulate cross-database integration result as
retrieving an evidence subgraph that links terms from multi-
databases through shared genes, and we ask whether the
retrieved subgraph isrelevant, cross-databaseconnected, and
compact(Section IV-B).
Settings.We evaluate GBOR-RAG retrieval subgraph on
42 queries against two baselines derived from the same ORA
scores and per-database candidate terms. Top-kpooling ag-
gregates the highest-ranked terms from each database without
modeling cross-database connections, whereas Enrichr-KG [8]
links enriched terms through genes shared across libraries.
We evaluate each retrieved evidence subgraph using structural
metrics for compactness and connectivity, together with a
rubric covering relevance, term-level integration, gene-level
bridging, and overall usefulness as interpretation evidence.
Judge reliability.To calibrate the LLM-as-a-judge evalu-
ation, we randomly selected 3 queries and evaluated all 18
retrieved positive and negative subgraphs. Each subgraph was
independently scored by two trained human annotators blinded
to method identity, as well as by the LLM judge. The human
annotators showed strong rank agreement (pooled Spearman
ρ= 0.74), suggesting that the rubric supports consistent hu-
man assessment. The LLM judge showed moderate agreement
with the human average (ρ= 0.53), with stronger agreement
on integration and overall usefulness (Table III). We therefore
use the LLM judge for scalable aggregate comparisons, not to
replace human annotation of individual subgraphs.
Result.GBOR-RAG best balances compactness and inte-
gration (Table IV); paired tests confirm its advantage over
top-kpooling in integration, bridging, and overall usefulness
(p <10−3), while its overall usefulness does not differ
significantly from Enrichr-KG (p= 0.08).
•Relevance.Under strong compression, GBOR-RAG
matches top-kpooling in relevance but trails Enrichr-KG.
•Connectivity.Top-kpooling yields fragmented sub-
graphs, whereas GBOR-RAG forms a single connected

TABLE III
JUDGE CALIBRATION(SPEARMANρ)AMONG HUMAN ANNOTATORS AND
BETWEEN THEIR AVERAGE AND THELLMJUDGE,ON18SUBGRAPHS
(TOP; SECTIONV-B)AND12INTERPRETATIONS(BOTTOM;
SECTIONV-D).
Subgraph Relev. Integr. Bridg. Overall All
Inter-annotator 0.59 0.68 0.85 0.680.74
Human–LLM 0.51 0.61 0.59 0.640.53
Interpretation Rel. Mec. Int. Anc. Cla. Ovr. All
Inter-annotator 0.62 0.75 0.62 0.53 0.56 0.450.67
Human–LLM 0.65 0.59 0.74 0.62 0.77 0.740.69
TABLE IV
CROSS-DATABASE INTEGRATION(MEAN PER POLE). STRUCTURE:NODES,
CONNECTED COMPONENTS(COMP.),AND FRACTION OF
SINGLE-COMPONENT POLES(SINGLE CC). CONTENT: LLM-JUDGE
RATING(1–5).
Structure Content
Method Nodes Comp. single cc Rel. Int. Bri. Ovr.
Top-kpooling 1466.7 15.64 0.00 3.89 3.68 2.52 3.20
Enrichr-KG 470.9 1.05 0.954.094.294.623.92
GBOR-RAG 36.8 1.00 1.003.894.314.434.12
component for every pole and obtains the highest mean
integration score.
•Compactness.GBOR-RAG reduces the mean subgraph
size from 1466.7 (top-kpooling) and 470.9 (Enrichr-KG)
to 36.8 nodes.
C. Claim-Level Faithfulness of Textual Interpretation
Metric.We introduceclaim-level faithfulnessto quantify
whether a generated interpretation is supported by its evidence.
A reportRis decomposed into a set of atomic claimsC(R)
[28], each asserting a gene, a term, a gene–term relation, or a
biological interpretation. Given the evidenceEthat the method
supplied to its generator, an independent LLM judge labels
each claimcas supported (s(c, E) = 1) when it is entailed
byEand unsupported (0) otherwise. Claim-level faithfulness
is the fraction of supported claims,
Faithfulness(R, E) =1
|C(R)|X
c∈C(R)s(c, E),(7)
averaged over all queries. The metric measures consistency
with the evidence a system was given, not biological correct-
ness.
Setting.On 42 queries we compare GBOR-RAG against
two published gene-set interpretation systems: GeneAgent
[13], which verifies claims through database tool calls, and
MAPA [9], which clusters enriched pathways into functional
modules before generation. Each system produces one inter-
pretation from its own evidence, decomposed and judged by
an independent LLM judge from a different model family than
the generators.
Result.As shown in Table V, GBOR-RAG attains the
highest claim-level faithfulness (0.90) with compact evidence,TABLE V
CLAIM-LEVEL FAITHFULNESS. CLAIMS ANDSUPPORTED:MEAN COUNTS
PER REPORT; FAITHFULNESS:MEAN PER-REPORT SUPPORTED-CLAIM
RATIO; EVIDENCE:TOKENS.
Method Evidence Claims Supported Faithfulness
GeneAgent 32.33K 19.90 13.88 0.71
MAPA 0.80K 20.52 5.80 0.29
GBOR-RAG 1.19K9.39 8.150.90
Fig. 3. Advanced evidence subgraph for the negative ORA of the Alzheimer’s
disease query. Curated terms and genes define a mitochondrial oxidative-
phosphorylation core; gold squares denote PubMed articles, and crimson rings
mark literature-layer nodes absent from the curated-only subgraph.
using about27×less evidence than GeneAgent and only
slightly more than MAPA. GeneAgent reaches0.71despite its
much larger evidence context, whereas MAPA reaches0.29on
the subset of queries where its module-clustering step applies.
GBOR-RAG also leaves the fewest unsupported claims, and
its faithfulness advantage is significant over shared queries (vs.
GeneAgentp <10−5; vs. MAPAp <10−6). These results
suggest that a compact, traceable evidence subgraph improves
faithfulness while avoiding the large evidence context required
by agentic verification.
D. Advanced GBOR-RAG: Literature-Augmented Retrieval
Purpose.We evaluate whether adding a PubMed literature
layer improves the generated interpretation.
Setting.Naive and advanced GBOR-RAG use the same
generator and prose prompt, differing only in the retrieved
evidence. On 42 queries, a blind independent LLM judge rates
each interpretation on six dimensions: relevance and speci-
ficity, mechanistic insight, integration and synthesis, evidence
anchoring, clarity, and overall quality.
Judge reliability.Three trained annotators rated a blind
calibration set of 6 queries, comprising 12 interpretations,
using the same rubric. Agreement was substantial among
annotators (pooled Spearmanρ= 0.67) and between the
human average and the LLM judge (pooledρ= 0.69, overall-
qualityρ= 0.74; Table III).

TABLE VI
INTERPRETATION QUALITY OF NAIVE VS.ADVANCEDGBOR-RAG (MEAN
LLM-JUDGE RATING,1–5,THREE JUDGMENTS PER TEXT; 42QUERIES).
Dimension Naive Advanced
Relevance & specificity 3.814.16
Mechanistic insight 3.724.11
Integration & synthesis 4.484.80
Evidence anchoring 4.234.98
Clarity 4.985.00
Overall4.174.75
Result.As shown in Table VI, advanced GBOR-RAG
improves all content-related dimensions while keeping clarity
essentially unchanged. The overall score increases from4.17
to4.75(p <10−6), with the largest gains in evidence anchor-
ing (+0.75) and mechanistic insight (+0.39). This suggests
that the PubMed layer provides stronger grounding and richer
mechanistic context without reducing readability.
Qualitative example.For GSE1297, as shown in Fig. 3
for the negative subgraph, advanced GBOR-RAG incorporates
PMIDs into prize-collecting retrieval, allowing it to collect
richer biological evidence than naive GBOR-RAG.
VI. CONCLUSION
We presented GBOR-RAG, which integrates multi-database
ORA through prize-collecting retrieval on a term–gene knowl-
edge graph and conditions an LLM on the retrieved gene-
anchored evidence. Graph-based ORA reproduces conven-
tional ORA with a median37×speedup, the retrieved sub-
graphs are compact and connected, the resulting interpretations
are more faithful to their evidence than those of existing
systems, and a PubMed literature layer further improves
interpretation quality. Our evaluation relies on an LLM judge
calibrated on small human-rated sets, and faithfulness reflects
consistency with the evidence rather than biological correct-
ness; a larger expert study is left for future work.
ACKNOWLEDGMENT
This work was supported by JST BOOST, Japan Grant
Number JPMJBS2402, and JSPS KAKENHI Grant Number
JP26K23914.
REFERENCES
[1] E. I. Boyleet al., “GO::TermFinder—open source software for accessing
Gene Ontology information and finding significantly enriched Gene
Ontology terms associated with a list of genes,”Bioinformatics, vol.
20, no. 18, pp. 3710–3715, 2004.
[2] M. Ashburneret al., “Gene Ontology: tool for the unification of biology,”
Nat. Genet., vol. 25, no. 1, pp. 25–29, 2000.
[3] M. Kanehisa and S. Goto, “KEGG: Kyoto Encyclopedia of Genes and
Genomes,”Nucleic Acids Res., vol. 28, no. 1, pp. 27–30, 2000.
[4] B. Jassalet al., “The reactome pathway knowledgebase,”Nucleic Acids
Res., vol. 48, no. D1, pp. D498–D503, 2020.
[5] M. Martenset al., “WikiPathways: connecting communities,”Nucleic
Acids Res., vol. 49, no. D1, pp. D613–D621, 2021.
[6] A. Liberzonet al., “The Molecular Signatures Database hallmark gene
set collection,”Cell Syst., vol. 1, no. 6, pp. 417–425, 2015.
[7] J. Pi ˜neroet al., “The DisGeNET knowledge platform for disease
genomics: 2019 update,”Nucleic Acids Res., vol. 48, no. D1, pp. D845–
D855, 2020.[8] J. E. Evangelistaet al., “Enrichr-KG: bridging enrichment analysis
across multiple libraries,”Nucleic Acids Res., vol. 51, no. W1, pp.
W168–W179, 2023.
[9] Y . Geet al., “Leveraging large language models for redundancy-aware
pathway analysis and deep biological interpretation,”bioRxivpreprint,
doi: 10.1101/2025.08.23.671949, 2025.
[10] R. Wanget al., “EnrichGT: a comprehensive R-based tool for func-
tional genomics enrichment analysis based on large language models,”
Artificial Intelligence Surgery, vol. 6, no. 1, pp. 18–35, 2026.
[11] E. Glaab, A. Baudot, N. Krasnogor, R. Schneider, and A. Valencia, “En-
richNet: network-based gene set enrichment analysis,”Bioinformatics,
vol. 28, no. 18, pp. i451–i457, 2012.
[12] D. Merico, R. Isserlin, O. Stueker, A. Emili, and G. D. Bader,
“Enrichment Map: a network-based method for gene-set enrichment
visualization and interpretation,”PLoS ONE, vol. 5, no. 11, p. e13984,
2010.
[13] Z. Wang, Q. Jin, C.-H. Wei, S. Tian, P.-T. Lai, Q. Zhu, C.-P. Day, C.
Ross, R. Leaman, and Z. Lu, “GeneAgent: self-verification language
agent for gene-set analysis using domain databases,”Nat. Methods,
vol. 22, no. 8, pp. 1677–1685, 2025.
[14] M. Hu, S. Alkhairy, I. Lee, R. T. Pillich, D. Fong, K. Smith, R.
Bachelder, T. Ideker, and D. Pratt, “Evaluation of large language models
for discovery of gene set function,”Nat. Methods, vol. 22, no. 1, pp. 82–
91, 2025.
[15] X. Heet al., “G-Retriever: retrieval-augmented generation for textual
graph understanding and question answering,” inProc. Adv. Neural Inf.
Process. Syst. (NeurIPS), 2024.
[16] M. X. Goemans and D. P. Williamson, “A general approximation
technique for constrained forest problems,”SIAM J. Comput., vol. 24,
no. 2, pp. 296–317, 1995.
[17] P. Khatri, M. Sirota, and A. J. Butte, “Ten years of pathway analysis:
current approaches and outstanding challenges,”PLoS Comput. Biol.,
vol. 8, no. 2, p. e1002375, 2012.
[18] M. E. Ritchieet al., “limma powers differential expression analyses for
RNA-sequencing and microarray studies,”Nucleic Acids Res., vol. 43,
no. 7, p. e47, 2015.
[19] L. Geistlingeret al., “Toward a gold standard for benchmarking gene
set enrichment analysis,”Brief. Bioinform., vol. 22, no. 1, pp. 545–556,
2021.
[20] P. Lewiset al., “Retrieval-augmented generation for knowledge-intensive
NLP tasks,” inProc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2020.
[21] D. Edgeet al., “From local to global: a graph RAG approach to query-
focused summarization,” arXiv:2404.16130, 2024.
[22] M. V . Kuleshovet al., “Enrichr: a comprehensive gene set enrichment
analysis web server 2016 update,”Nucleic Acids Res., vol. 44, no. W1,
pp. W90–W97, 2016.
[23] D. Ochoaet al., “The next-generation Open Targets Platform: reimag-
ined, redesigned, rebuilt,”Nucleic Acids Res., vol. 51, no. D1,
pp. D1353–D1359, 2023.
[24] G. R. Brownet al., “Gene: a gene-centered information resource at
NCBI,”Nucleic Acids Res., vol. 43, no. D1, pp. D36–D42, 2015.
[25] C. Hegde, P. Indyk, and L. Schmidt, “A nearly-linear time framework
for graph-structured sparsity,” inProc. Int. Conf. Mach. Learn. (ICML),
2015, pp. 928–937.
[26] G. Bouma, “Normalized (pointwise) mutual information in collocation
extraction,” inFrom Form to Meaning: Processing Texts Automatically,
Proc. Biennial GSCL Conf. 2009, Potsdam, Germany, 2009, pp. 31–40.
[27] L. Zhenget al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot
Arena,” inProc. Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and
Benchmarks Track, 2023.
[28] S. Minet al., “FActScore: Fine-grained atomic evaluation of factual
precision in long form text generation,” inProc. Conf. Empirical
Methods Natural Lang. Process. (EMNLP), 2023, pp. 12076–12100.