# Towards Platonic Representation for Table Reasoning: A Foundation for Permutation-Invariant Retrieval

**Authors**: Willy Carlos Tchuitcheu, Tan Lu, Ann Dooms

**Published**: 2026-04-13 23:33:43

**PDF URL**: [https://arxiv.org/pdf/2604.12133v1](https://arxiv.org/pdf/2604.12133v1)

## Abstract
Historical approaches to Table Representation Learning (TRL) have largely adopted the sequential paradigms of Natural Language Processing (NLP). We argue that this linearization of tables discards their essential geometric and relational structure, creating representations that are brittle to layout permutations. This paper introduces the Platonic Representation Hypothesis (PRH) for tables, positing that a semantically robust latent space for table reasoning must be intrinsically Permutation Invariant (PI). To ground this hypothesis, we first conduct a retrospective analysis of table-reasoning tasks, highlighting the pervasive serialization bias that compromises structural integrity. We then propose a formal framework to diagnose this bias, introducing two principled metrics based on Centered Kernel Alignment (CKA): (i) PI, which measures embedding drift under complete structural derangement, and (ii) rho, a Spearman-based metric that tracks the convergence of latent structures toward a canonical form as structural information is incrementally restored. Our empirical analysis quantifies an expected flaw in modern Large Language Models (LLMs): even minor layout permutations induce significant, disproportionate semantic shifts in their table embeddings. This exposes a fundamental vulnerability in RAG systems, in which table retrieval becomes fragile to layout-dependent noise rather than to semantic content. In response, we present a novel, structure-aware TRL encoder architecture that explicitly enforces the cognitive principle of cell header alignment. This model demonstrates superior geometric stability and moves towards the PI ideal. Our work provides both a foundational critique of linearized table encoders and the theoretical scaffolding for semantically stable, permutation invariant retrieval, charting a new direction for table reasoning in information systems.

## Full Text


<!-- PDF content starts -->

Towards Platonic Representation for Table Reasoning: A
Foundation for Permutation-Invariant Retrieval
Willy Carlos Tchuitcheu∗
Department of Mathematics and Data
Science, Vrije Universiteit Brussel
(VUB)
Brussels, BelgiumTan Lu
Department of Mathematics and Data
Science, Vrije Universiteit Brussel
(VUB)
Brussels, Belgium
Data Science Lab, Royal Library of
Belgium (KBR)
Brussels, BelgiumAnn Dooms
Department of Mathematics and Data
Science, Vrije Universiteit Brussel
(VUB)
Brussels, Belgium
Abstract
Historical approaches to Table Representation Learning (TRL) have
largely adopted the sequential paradigms of Natural Language Pro-
cessing (NLP). We argue that this linearization of tables discards
their essential geometric and relational structure, creating represen-
tations that are brittle to layout permutations. This paper introduces
the Platonic Representation Hypothesis (PRH) for tables, positing
that a semantically robust latent space for table reasoning must
be intrinsically Permutation-Invariant (PI). To ground this hypoth-
esis, we first conduct a retrospective analysis of table-reasoning
tasks, highlighting the pervasive "serialization bias" that compro-
mises structural integrity. We then propose a formal framework
to diagnose this bias, introducing two principled metrics based on
Centered Kernel Alignment (CKA): (i) PIderange , which measures
embedding drift under complete structural derangement, and (ii)
𝜌mono, a Spearman-based metric that tracks the convergence of
latent structures toward a canonical form as structural information
is incrementally restored. Our empirical analysis quantifies an ex-
pected flaw in modern Large Language Models (LLMs): even minor
layout permutations induce significant, disproportionate seman-
tic shifts in their table embeddings. This exposes a fundamental
vulnerability in Retrieval-Augmented Generation (RAG) systems,
in which table retrieval becomes fragile to layout-dependent noise
rather than to semantic content. In response, we present a novel,
structure-aware TRL encoder architecture that explicitly enforces
the cognitive principle of cell–header alignment. This model demon-
strates superior geometric stability and moves towards the PI ideal.
Our work provides both a foundational critique of linearized table
encoders and the theoretical scaffolding for semantically stable,
permutation-invariant retrieval, charting a new direction for table
reasoning in information systems.
Keywords
Table Representation Learning, Permutation Invariance, Platonic
Representation Hypothesis, Evaluation Metrics, Semantic Drift,
Table Reasoning, Information Retrieval
1 Introduction
Information Retrieval (IR) with Retrieval-Augmented Generation
(RAG) [ 14] is undergoing a major shift: rather than generating
∗Corresponding author (willy.Carlos.tchuitcheu@vub.be).answers only from retrieved unstructured text chunks, modern sys-
tems increasingly aim to produce concise answers by synthesizing
evidence from heterogeneous and (semi-)structured sources such
as tables [ 4]. However, the strong performance of Large Language
Models (LLMs) on plain text has not translated into comparable re-
liability on (semi-)structured inputs. Recent table reasoning bench-
marks and systems show that LLMs still struggle to consistently
interpret and reason over tables, which remains a key bottleneck
for interactive systems built on large-scale structured knowledge
[20, 24].
Tables are not just “text in cells.” They encode relationships,
hierarchies, and strict structure. A key property is row–column
permutation invariance (PI): permuting rows or columns should
not change the table’s meaning. Conventional dense retrieval, how-
ever, typically fails to preserve this property. Because retrieval is
the foundation of any RAG system, the quality and structure of
the vector representations directly affect what context is retrieved
and, in turn, how well the system can reason. Recent evaluations,
including BRIGHT [ 20] and TableRAG [ 4], reveal a major limitation:
modern dense retrievers struggle with reasoning-heavy queries,
where surface-level semantic similarity does not match true logical
or relational relevance. A major cause is the common practice of
serializing tables into linear text strings.
In this work, we first empirically quantify how such transforma-
tions, especially linearization, erase a table’s structural properties,
with a focus on PI. We show that this issue persists even when
using LLMs fine-tuned for tabular tasks, suggesting a deeper limi-
tation of current approaches. This aligns with recent evidence that
LLM performance on table understanding has started to plateau, as
observed in TableBench [ 24]. Motivated by these findings, we ar-
gue for Platonic representations: structure-aware table embeddings
that explicitly preserve key invariants. We present these represen-
tations as a necessary step toward a dual-pathway IR vision (Figure
1). First, we introduce an Online Table Question Answering (OTQA)
setting, where retrieval-style lookup questions can be answered
directly in embedding space, enabling fast, zero-shot responses
without expensive prompting. Second, we argue that invariant-
aware representations are crucial for practical IR workloads in
industry, including multi-step aggregation, relational filtering, and
cross-table reasoning.
The idea of “Platonic” representations has recently been intro-
duced in representation learning, not as a fixed architecture, but
as a hypothesis. The Platonic Representation Hypothesis (PRH)arXiv:2604.12133v1  [cs.AI]  13 Apr 2026

Figure 1: Platonic view of permutation invariance in table embeddings and prospective long-term impact paradigm shift in IR.
The left part illustrates the theoretical concept of "Platonic table embedding," investigating whether an original input table
and its structurally transposed counterpart (with rows and columns swapped) can be projected by embedding models into a
consistent, semantically invariant latent space. The t-SNE visualizations and the question mark between them highlight the
core research question regarding the alignment of these embedding spaces across different table representations. The right
panel depicts prospective downstream applications that could be built upon such robust embeddings. The top section shows an
"Online Table Question Answer" system that uses cell embeddings and nearest-neighbor search to directly answer user queries,
while the bottom section outlines a "table-centric RAG application" that leverages table vectors for RAG tasks.
[11] is assumed to reflect the underlying representation that stays
invariant to specific sources of variation. A core idea in PRH is
that different views of the same entity are only different obser-
vations—or “shadows”—of a single underlying essence. We adopt
this perspective for Table Representation Learning (TRL) and ask
a fundamental question: Do current table embeddings capture a
“Platonic” representation that is invariant to row and column per-
mutations? Under this view, any row- or column-permuted version
of a table is simply another observation of the same table (see the
example in the left panel of Figure 1).
The remainder of this paper is structured as follows: Section 2
provides an overview of our empirical investigation intolineariza-
tion biasand a vision for a paradigm shift in IR. Section 3 provides
relevant background. In Section 4, we detail our proposed method-
ology for diagnosing the serialization bias, followed by our experi-
mental framework and results in Section 5. A dedicated prospective
analysis is presented in Section 6, where we explore the implica-
tions of our findings for the future of IR, before concluding the
paper in Section 7.
2 Overview
This prospective paper argues thatpermutation invariance (PI)is a
pinpoint yet overlooked property for table-centric RAG: if a repre-
sentation cannot remain stable under row/column permutations,
it is structurally misaligned with table reasoning. We therefore
assess the PI of cell-level contextual embeddings and use the result-
ing empirical observations to motivate a forward-looking research
direction. Our contributions are threefold:
•A diagnostic framework via the Platonic lens:We in-
stantiate thePlatonic Representation Hypothesis(PRH) fortabular data and formalize the notion that row/column per-
mutations are “shadows” of the same table. Concretely, we
evaluate the stability of cell-level embeddings across differ-
ent serialization manifolds under controlled permutations.
•Structural invariance metrics ( 𝑃𝐼derange &𝜌mono):We
introduce a CKA-based evaluation protocol to quantify rep-
resentational drift and recovery under permutations. From
this protocol, we derive two scalar metrics:
(1)PI derange : residual similarity under complete permutation
(full derangement).
(2)𝜌 mono: monotonicity of representational recovery as struc-
tural constraints are incrementally restored.
•A prospective vision for permutation-invariant re-
trieval:Using the above diagnostics and metrics as empirical
grounding, we outline a research agenda towardPlatonic
representationsfor tables, meaning structure-aware
embeddings designed to preserve PI. We discuss how such
invariant-preserving encoders can serve as a foundation
for table-centric retrieval and downstream reasoning, and
identify open directions to make permutation-invariant
retrieval a standard building block in IR pipelines.
3 Background
Why permutation invariance matters now.A central open
question in table retrieval is whether table embeddings are robust
to the structural symmetries of tables. A natural desideratum is
PI: semantically equivalent row/column permutations should yield
similar embeddings. Yet this property has not been systematically
characterizedat the embedding levelfor table representations, de-
spite its value as a lightweight diagnostic for table-centric retrieval.
2

This gap is becoming increasingly consequential as the commu-
nity moves toward LLM-powered interactive systems over struc-
tured data. Much of the work in Table Question Answering (TQA)
[15], TableGPT [ 26], and text-to-SQL [ 7] assumes that the relevant
table (or database) is already provided. In prospective table-centric
RAG settings, however, identifying the relevant table is itself the
first bottleneck, and it depends directly on what the embedding
model preserves about table semantics and structure. A related line
of work evaluates table retrieval in end-to-end analytic query sys-
tems using embeddings from (non-)commercial LLMs [ 12], but does
not explicitly test whether these embedders satisfy intrinsic table
properties such as PI. One reason is the lack of clear, model-agnostic
metrics for quantifying PI in table embeddings—a methodological
gap that this paper aims to foreground.
Limited diagnostics in table embedding space.Many
structure-aware TRL approaches follow a largelytask-oriented
paradigm: the impact of permutations is assessed indirectly
through downstream performance (e.g., column type annotation,
entity linking, relation extraction) rather than through direct
analysis of the embedding space [ 5]. Pre-trained table models
such as TaBERT [ 25], TAPAS [ 10], and TURL [ 6] incorporate table
structure during pretraining, yet they are not explicitly designed to
enforce PI. Similarly, strong tabular predictors such as TabNet [ 2]
and FT-Transformer [ 8] optimize task accuracy but do not target
robustness properties of the learned representations.
More recently, HYTREL [ 3] introduced a fine-grained, cell-
level assessment and showed that embeddings can be sensitive
to row and column order. However, its evaluation primarily
considered training-domain settings, limiting conclusions about
out-of-distribution (zero-shot) robustness. Overall, a systematic
and metric-agnostic assessment of PIin the embedding space
remains underexplored, despite its potential to guide the next
generation of table retrievers.
TRL bottleneck and the signal from benchmarks.Recent
work on table-centric RAG increasingly explores hybrid architec-
tures that combine dense retrieval with structured reasoning, e.g.,
BRIGHT [ 20], while TARGET [ 12] emphasizes the need to evaluate
the table retrieval component itself. In parallel, the evaluation land-
scape has evolved from simple cell-lookup tasks (e.g., WikiTableQA)
toward reasoning-intensive benchmarks such asBRIGHT[ 20] and
TableBench[ 24], which demand logically grounded retrieval and
industrial-scale multi-step reasoning. Despite this progress, these
benchmarks expose a persistent representation bottleneck: most
pipelines still rely ontable linearizationto fit tables into standard
language-model inputs, and even instruction-tuned approaches
such asTableLlama[ 27] do not reliably preserve table structure
in their embeddings. As we will demonstrate later, this manifests
asstructural perturbation—a failure to achieve true row/column
PI—and it amplifieslinearization bias, where retrieval defaults to
localized text matching rather than global relational and quan-
titative reasoning. Collectively, these observations motivate our
diagnostic focus and prospective agenda: table representation learn-
ing must explicitly target structural invariants as a foundation for
permutation-invariant retrieval in future table-RAG systems.4Methodology: Diagnosing Serialization Bias in
TRL
We diagnose serialization bias in LLM-based table representations
by comparing embeddings from linearized tables with those pro-
duced by a table-structure-aware encoder. Building on this setup,
we instantiate the PRH for tables and formalize how PI can be as-
sessed in the embedding space as an intrinsic diagnostic of this
bias. To make this comparison easier to follow, we first present the
tables and introduce our notation.
Let a table𝑇have𝑛rows and𝑚columns with cell set
𝐶={𝑐 𝑖 𝑗}𝑖∈[𝑛], 𝑗∈[𝑚],
row headers 𝑅={𝑟 𝑖}𝑖∈[𝑛] , and column headers 𝐻={ℎ 𝑗}𝑗∈[𝑚] . We
write𝑇=𝐶∪𝑅∪𝐻with|𝐶|=𝑛𝑚.
Let𝑇∗denote the space of unstructured text (strings). We convert
any (semi-)structured table into an LLM-readable sequence via a
linearization function 𝛿:𝑇−→𝑇∗,e.g., row-wise serialization
with special delimiters for [ROW] ,[CELL] , and [SEP] to preserve
extractability of each cell span within the sequence as implemented
in RAG frameworks.
4.1 Serialization: LLM-based table cell
embeddings
Given a serialized table 𝛿(𝑇) , we obtain a contextual embedding
for any cell𝑥∈𝑇using a base embedding function
𝜙:(𝑥,𝛿(𝑇))↦−→𝜙(𝑥|𝛿(𝑇))∈R𝑑,(1)
where𝑑is the embedding dimension and 𝜙depends on the LLM
family (encoder-only models use bidirectional context; decoder-
only models use causal context). For simplicity, we consider that
the LLM-only table embedding space is produced by the function
Φ:
Φ(𝑇)=
𝜙(𝑐 11|𝛿(𝑇)),...,𝜙(𝑐 𝑛𝑚|𝛿(𝑇))⊤∈R𝑛𝑚×𝑑.
Because𝜙(·|𝛿(𝑇)) conditions on the serialized table, identical
surface values can receive different embeddings when their headers
differ. For example, the numeric value 12under the column header
Agemay be embedded differently from 12underWeight, reflecting
context capturing when LLMs are applied to produce table cell
embeddings.
4.2 Structure-aware cell embeddings via TRL
We compareLLM linearizationto astructure–awareTRL embedder
based onSemantic Meta-Paths(SMP) [ 21], which help to preserve
header–cell semantic context:
𝑇𝜙−−→𝐷 𝑦SMP−−−−→𝑆TRL−−−−→
𝜓𝜃𝐷𝑧.(2)
where𝐷𝑦={𝜙(𝑥) :𝑥∈𝑇} are the initial context-free LLM-based
cell embeddings and and 𝐷𝑧={𝜓 𝜃(𝑥|𝑆) :𝑥∈𝑇} arecontext-aware
embeddings produced by the TRL encoder 𝜓𝜃(𝜃denotes trainable
parameters of the encoder) given the set 𝑆of SMP sequences ex-
tracted from 𝑇. SMPs extractsemantically meaningful paths(e.g.,
header↔cell↔header), mimicking human table reading by ac-
tively tying each cell to its row/column headers. TRL optimizes cell
embeddings based on two complementary signals: (i)local align-
ment (intra-SMP):pull together a cell and its corresponding headers,
enforcing header–cell correspondence within each SMP; and (ii)
3

global separation (inter-SMP):push apart embeddings of semanti-
cally distinct cells (even if their surface text is similar), increasing
the distance across SMPs’ embeddings. Combining these two sig-
nals, TRL aims to learn cell embeddings by linking them to the
corresponding headers. We refer to the TRL-based table embedding
space as
Ψ𝜃(𝑇)=
𝜓𝜃(𝑐11|𝑆),...,𝜓 𝜃(𝑐𝑛𝑚|𝑆)⊤∈R𝑛𝑚×𝑑.
4.3 Platonic Representation Hypothesis (PRH)
in TRL
Let𝐺=𝑆 𝑛×𝑆𝑚be the product of symmetric groups acting on
row and column indices of a table 𝑇. We define a group action
ℎ:𝐺×𝑇→𝑇 such that for any(𝜎,𝜏)∈𝐺 and table element 𝑥∈𝑇 :
ℎ(𝜎,𝜏)(𝑥)= 
𝑐𝜎(𝑖)𝜏(𝑗) if𝑥=𝑐 𝑖 𝑗,
𝑟𝜎(𝑖) if𝑥=𝑟 𝑖,
ℎ𝜏(𝑗) if𝑥=ℎ 𝑗.(3)
Since𝜎∈𝑆 𝑛,𝜏∈𝑆 𝑚are bijections, ℎ(𝜎,𝜏) is a well-defined group
action. We investigate the PRH within tables by quantifying the
invariance of cell embeddings under the group action ℎ. For any
𝑔∈𝐺 , we compare embeddings computed on the permuted table
𝑔·𝑇:=ℎ 𝑔(𝑇)to those on𝑇using a similarity score𝑠(·,·):
(Intra-model)𝑠 Ψ𝜃(𝑔·𝑇),Ψ 𝜃(𝑇),(4)
(Cross-model)𝑠 Ψ𝜃(𝑔·𝑇),Ψ 𝛽(𝑇),(5)
(LLM baseline)𝑠 Φ(𝑔·𝑇),Φ(𝑇).(6)
Here, Ψ𝜃andΨ𝛽denote TRL encoders with different parameters,
andΦrepresents the pretrained LLM-derived cell embeddings func-
tion, as explained in Section 4.1. Higher scores across permutations
𝑔∈𝐺 indicate stronger PI, thereby providing more substantial
evidence for the PRH for tables.
To compare embedding spaces (i.e., Equations. 4, 6), we employ
Centered Kernel Alignment (CKA)[ 13], a normalized HSIC-based
similarity widely used for cross-network representation analysis
(related tools include CCA/SVCCA [17, 19]):
CKA(𝑋,𝑌)=HSIC(𝑋,𝑌)√︁
HSIC(𝑋,𝑋)HSIC(𝑌,𝑌),(7)
where𝑋and𝑌are the cell-embeddings of the original table 𝑇
and its permuted counterpart 𝑔·𝑇, respectively. This formulation
includes several desirable properties, such as invariance to:
•Isotropic Scaling, i.e., CKA(𝑋,𝑌)=CKA(𝛼 1𝑋,𝛼 2𝑌)for any
𝛼1,𝛼2∈R+.
•Orthogonal Transformation, i.e., CKA(𝑋,𝑌)=CKA
(𝑋𝑈,𝑌𝑉) for any full-rank orthogonal matrices 𝑈and𝑉
such that𝑈𝑇𝑈=𝐼and𝑉𝑇𝑉=𝐼.
These properties make CKA a well-suited choice for studying the
PI ofΦ(·),Ψ 𝜃(·), andΨ 𝛽(·).
5 Experiments and Discussion
To probePlatonicinvariance from a table 𝑇with𝑛rows and𝑚
columns, we use the row–column permutation action ℎ(see Equa-
tion 3) to build a grid of controlled permutations parameterized by
the number of fixed rows and columns 𝑎and𝑏. Meaning choosing
𝜎𝑎∈𝑆 𝑛and𝜏𝑏∈𝑆𝑚such that𝑎=fix(𝜎 𝑎)=|{𝑖 :𝜎𝑎(𝑖)=𝑖}| ,and𝑏=fix(𝜏 𝑏)=|{𝑗 :𝜏𝑏(𝑗)=𝑗}| . We refer to a parameterized
(𝑎,𝑏) -permuted table as 𝑇𝑏
𝑎:=ℎ (𝜎𝑎,𝜏𝑏),𝑇:=ℎ 𝑔(𝑇), yielding a
test bench
{𝑇𝑏
𝑎:𝑎=0,···,𝑛;𝑏=0,···,𝑚},
which contains(𝑛+ 1)(𝑚+ 1)permuted tables, spanning identity
(i.e.,𝑇𝑚
𝑛) to full derangement 𝑇0
0(maximal shuffling on both rows
and columns).
5.1 Proposed metrics and models baseline
Proposed metrics.For each (𝑎,𝑏) , we compare the cell embedding
matrix of the original table 𝑋=Ψ 𝜃(𝑇)(orΦ(𝑇) for LLM baselines)
with that of the permuted table 𝑌𝑏
𝑎=Ψ 𝜃(𝑇𝑏
𝑎)(orΦ(𝑇𝑏
𝑎)), producing
aPlatonic heatmap𝐻∈[0,1](𝑛+1)×(𝑚+1)such that:
𝐻[𝑎,𝑏]=CKA 𝑋, 𝑌𝑏
𝑎,∀(𝑎,𝑏)∈{0,...,𝑛}×{0,...,𝑚}(8)
This heatmap is further summarised with two scalars:
PIderange =𝐻[0,0], 𝜌 mono=Spearman 𝐻[𝑎,𝑏],𝑎+𝑏.(9)
Here, PIderange quantifies Permutation Invariance ( PI) underfull
derangement 𝑇0
0(higher is better), and 𝜌mono∈[− 1,1]measures
𝐻[𝑎,𝑏] changes as structure is restored 𝑇𝑚
𝑛(i.e., as𝑎and𝑏increase
toward𝑛and𝑚): values near+1indicate strong increasing mono-
tonicity from PIderange , near0no monotonic trend, and near −1
strong decreasing monotonicity.
As an additional summary, we also report a normalized Area
Under Curve (AUC) computed along a single axis of restoration,
either rows ( 𝑎) or columns ( 𝑏), while keeping the other axis fixed.
Formally, for a fixed𝑏,
AUC row=1
𝑛𝑛−1∑︁
𝑎=0𝐻[𝑎,𝑏]+𝐻[𝑎+1,𝑏]
2,(10)
and symmetrically, for a fixed𝑎,
AUC col=1
𝑚𝑚−1∑︁
𝑏=0𝐻[𝑎,𝑏]+𝐻[𝑎,𝑏+1]
2.(11)
Both AUC rowandAUC colare normalized to lie in [0,1], providing
a smooth global measure of invariance along each axis separately.
Baseline models.To evaluate the embedding LLM base model 𝜙
(see Equation 1), we select a diverse suite of models, which collec-
tively illustrate a larger spectrum of Transformers [ 23] architecture
implementations available. This selection allows us to investigate
whether the Platonic convergence of structural invariants is a uni-
versal property of Transformers or dependent on specific training
paradigms:
•Commercial closed-source decoder-only models:We
includeGPT-3.5[ 1], representing the industry standard for
proprietary black-box embeddings in RAG pipelines.
•Open-source decoder-only families:We evaluate a spec-
trum of high-performance models including Gemma 2 [22],
Llama 3 [9], and the reasoning-optimized DeepSeek-R1 [16].
These represent the state-of-the-art in autoregressive repre-
sentation learning.
•Specialized encoder-Only architectures:We utilize the
lightweight nomic-embed-text [18], a model specifically
optimized for long-context retrieval tasks.
4

Club Played Won Drawn Lost Points for Points against Tries for Tries against Try bonus Losing bonus Points
Bridgend Athletic RFC 22 16 0 6 523 303 68 31 10 4 78
Builth Wells RFC 22 17 0 5 473 305 57 29 7 2 77
Kidwelly RFC 22 14 1 7 532 386 63 45 5 3 66
Loughor RFC 22 13 1 8 532 388 69 43 9 1 64
Ammanford RFC 22 13 0 9 447 394 58 51 6 4 62
Waunarlwydd RFC 22 12 2 8 504 439 57 55 6 3 61
Pencoed RFC 22 13 0 9 425 328 53 36 4 4 60
BP RFC 22 9 1 12 367 358 39 43 2 7 47
Mumbles RFC 22 8 2 12 373 450 50 56 4 4 44
Cwmavon RFC 22 6 2 14 332 515 39 66 3 5 36
Penclawdd RFC 22 4 1 17 263 520 28 68 1 3 22
Gorseinon RFC 22 2 0 20 340 725 48 106 3 4 15
Figure 2: Single-table test example of Rugby Club Performance Statistics ( from WikiSQL dataset, Table ID 560). Identical
content values (e.g., “22”) can denote different semantics depending on headers.
Figure 3: LLM-only table embeddings across LLMs ( OpenAI ,nomic-embed-text ,Gemma2 ,DeepSeek-R1 , and llama3 ). Rows corre-
spond to models. Columns show: Left t -SNE projections of LLM-basedcell embeddingsfor the baseline table (Figure 2); a point
indexed by(row_id,col_id) denotes the projected embedding of a cell. Center pairwise cosine similarity within the resulting
cell-embedding set. Right heatmaps𝐻[𝑎,𝑏]of similarity between the original table and its169permutations𝑇𝑏
𝑎.
•TRL-refined baselines ( Ψ𝜃):We compare these general-
purpose LLMs against a specialized structure-aware TRLencoder [ 21]. This model uses a refined Transformer-based
5

Figure 4: Distribution of heatmap entries 𝐻[𝑎,𝑏] and the mean ¯𝑥comparing LLM and TRL-derived cell embeddings under two
settings: (i)intra-model, which examines embeddings of permuted tables generated by the same model that was trained on the
original table. (ii)Cross-model, where embeddings of permuted tables are extracted from different models trained on each
other’s samples.
architecture [ 23] with three layers and 12 attention heads
(𝑑=768), serving as a structure-aware control group.
5.2 Evaluation on single-table
Our first benchmark uses169permuted tables 𝑇𝑏
𝑎derived from
the baseline table shown in Figure 2 where 𝑛=12rows and 𝑚=12
columns. For each model, we extract per–cell embeddings ( Φfor
LLM baselines via Equation 1; Ψ𝜃for TRL), and: (i) visualize cell
embeddings of the original table (left column of Figure 3); (ii) the
pairwise cosine similarity shown in the center of Figure 3; (iii) build
thePlatonic heatmap𝐻shown in the right column of Figure 3.
Analysing cell embeddings visualization. openai_GPT3.5 and
nomic-embed-text exhibit clearcell-typeclustering (headers vs.
data cells), whereas other models do not, highlighting that different
LLMs induce qualitatively different embedding geometries and
cluster structure.
Analysing embedding space geometric.The within-table pair-
wise cosine-similarity patterns vary substantially across models,
indicating that the relative neighborhood structure among cells
is model-dependent. This suggests that the resulting embedding
spaces arenotreadily comparable (and not isometric in practice),
which complicates cross-model alignment. To contrast LLM-only
embeddings with structure-aware TRL embeddings, Figure 5 reportsnormalized (to[0,1]) pairwise cosine-similarity matrices for both
approaches. Compared with LLM-only baselines, TRL-refined em-
beddings exhibit a more coherent similarity structure (e.g., clearer
block patterns and more consistent density), consistent with the
hypothesis that injecting an equivariant table structure during re-
finement yields more stable representations.
Analysing the CKA heatmap.The right panel of Figure 5 reports
the CKA heatmap 𝐻between the original table and its permuted
variants. As expected, 𝐻[12,12]=1, corresponding to the identity
permutation 𝑇12
12. Along the rightmost column ( 𝑎<12, i.e., row-only
permutations), 𝐻[𝑎, 12]remains high—particularly for openai_-
GPT3.5 ,nomic-embed-text , and gemma2 —indicating relatively low
sensitivity to row shuffles. In contrast, any column permutation
(𝑏<12) substantially reduces 𝐻[12,𝑏], suggesting that LLM-only
embeddings are markedly more sensitive to column order than to
row order. To compare LLM-only embeddings with TRL-refined em-
beddings, we flatten each model’s heatmap values and summarize
their distributions using boxplots (Figure 4). From this perspective,
TRL-derived embeddings consistently achieve higher CKA scores,
aligning more closely with the PRH. This improvement is consistent
with the structural awareness introduced during refinement (e.g.,
explicit header–cell alignment).
6

Figure 5: Normalized histograms of pairwise cosine similarities for cell-cell embeddings: LLM-only (blue) vs. TRL-refined
(orange) across five base models. LLM-only spaces exhibit narrow peaks at moderate similarity ( ∼0.30−0.50), whereas TRL
refinement produces broader, flatter distributions with more mass up to ∼0.75. The TRL-refined model yields more consistent
histogram shapes across models, indicating better cross-model compatibility.
(b) LLM embeddings
 (c) TRL-refined embeddings (intra model)
 (c) TRL-refined embeddings (cross model)
Figure 6: Analysis of the Platonic Representation Hypothesis (PRH) under progressive row and column shuffling. Each curve
illustrates the similarity score 𝐻[𝑎,𝑏] as the number of fixed rows ( 𝑎) or columns ( 𝑏) increases, ranging from full derangement ( 𝑇0
0)
to complete identity ( 𝑇𝑚
𝑛). Solid lines represent column permutations ( 𝐻[12,𝑏]), while dashed lines represent row permutations
(𝐻[𝑎, 12]). The three panels compare: (left) baseline LLM embeddings, (center) TRL-refined embeddings (intra-model), and
(right) TRL-refined embeddings (cross-model). The legend indicates the model size and embedding dimension. The TRL curves
are flatter and more stable, demonstrating stronger PI, whereas the LLM-only curves exhibit a steeper decline, particularly
under column permutations.
Ablation: row vs. column permutations.We isolate row- and
column-permutation effects by analyzing the two heatmap slices
in Figure 6b–c): 𝐻[𝑎, 12](column order fixed; rows permuted)
and𝐻[12,𝑏](row order fixed; columns permuted). We summa-
rize each slice with two scalars: (i) anaxis-derangement score,
PIderange =𝐻[ 0,12]for rows (or 𝐻[12,0]for columns), capturing
invariance under full derangement along one axis; and (ii) a
monotonicity coefficient, 𝜌mono=Spearman({𝐻[𝑎, 12]},{𝑎}) (or
Spearman({𝐻[ 12,𝑏]},{𝑏}) ), which quantifies how consistently
similarity recovers as fixed rows/columns are restored toward
𝐻[12,12]=1.
Across LLM-only embeddings we observe PIderange∈[0.08,0.76]
and𝜌mono∈ [− 0.06,0.66], whereas TRL-refined embeddings at-
tain higher values, [0.34,0.94]and[−0.02,0.79], respectively. For
example, the strongest LLM baseline ( openai_GPT3.5 ) achieves
0.76/0.03under row permutations, while TRL reaches0 .94/0.02, re-
flecting a higher overall similarity profile and greater permutation
robustness, consistent with closer adherence to the PRH.Interestingly, among LLM baselines, the 270M-parameter nomic-
embed-text consistently rivals a 175B-parameter model, suggest-
ing thatembedding dimensionalityandtraining objectivesmay influ-
ence PI more than sheer parameter count. Finally, in the embedding
space (Figure 6a), TRL embeddings exhibit more similar global ge-
ometry across models, whereas the best-performing LLM baseline
forms a noticeably denser configuration.
5.3 Evaluation on multiple tables
We further conduct experiments on a benchmark of 3,791 tables
generated from 20 base tables with sizes 𝑛∈[ 10,30](rows) and
𝑚∈[ 10,15](columns). We report the mean PIderange and𝜌mono
along with the standard deviation in Table 1. We provide a detailed
analysis of Table 1 for each metric.
Monotonicity coefficient 𝜌mono:Measures whether similarity
𝐻[𝑎,𝑏] increases monotonically as more rows ( 𝑎) or columns ( 𝑏)
are restored from derangement back to identity. For LLM baselines,
𝜌mono is slightly lower (e.g.,0 .17overall for OpenAI GPT-3.5 ,0.12
7

Table 1: Average performance (mean ±std.) over 3,791 permuted tables generated from 20 larger base tables with 𝑛∈[ 10,30]
rows and𝑚∈[ 10,15]columns. We evaluate each model under row- and column-permutation settings derived from the heatmap
𝐻[𝑎,𝑏] , and report results for both base LLM embeddings and their TRL-refined counterparts, highlighting gains in permutation
robustness.
Models𝜌 mono PIderange AUC
rows
H[a,0]cols
H[0,b]all
H[a,b]rows
H[a,0]cols
H[0,b]all
H[a,b]rows
H[a,0]Cols
H[0,b]
LLM base Embedding
openai_GPT3.5 0.23±0.19 0.20±0.23 0.17±0.06 0.77±0.130.42±0.17 0.38±0.17 0.78±0.12 0.40±0.21
nomic-embed-text 0.29±0.21 0.22±0.26 0.20±0.07 0.61±0.16 0.39±0.16 0.35±0.16 0.63±0.15 0.37±0.19
gemma2 0.20±0.21 0.18±0.14 0.18±0.1 0.56±0.16 0.22±0.11 0.18±0.09 0.60±0.12 0.22±0.14
deepseek-r1 0.23±0.29 0.32±0.18 0.17±0.07 0.34±0.14 0.10±0.07 0.11±0.1 0.38±0.13 0.15±0.14
llama3 0.23±0.25 0.16±0.31 0.12±0.14 0.32±0.18 0.19±0.13 0.19±0.1 0.38±0.17 0.22±0.16
TRL-refined LLM embedding via TRL (intra-model)
openai_GPT3.5 0.26±0.21 0.31±0.2 0.2±0.08 0.73±0.06 0.52±0.05 0.49±0.08 0.75±0.05 0.49±0.19
nomic-embed-text 0.29±0.210.33±0.190.18±0.06 0.74±0.08 0.53±0.070.56±0.14 0.77±0.07 0.49±0.19
gemma2 0.27±0.22 0.25±0.19 0.23±0.09 0.77±0.12 0.61 + 0.140.55±0.13 0.79±0.1 0.56±0.26
deepseek-r10.31±0.270.29±0.190.25±0.09 0.60±0.12 0.49±0.12 0.44±0.11 0.63±0.1 0.46±0.23
llama3 0.30±0.23 0.31±0.23 0.23±0.07 0.69±0.08 0.55±0.1 0.51±0.09 0.71±0.08 0.51±0.21
forLlama3 ), indicating that embeddings fail to recover consistently
with the structure. In contrast, TRL-refined embeddings system-
atically improve (up to0 .31for DeepSeek-r1 ), showing stronger
alignment with the PRH.
Permutation Invariance PIderange :quantifies how much embed-
dings degrade under full derangement relative to identity. Larger
values indicate more invariance to shuffling. LLMs show sharp
drops, especially under column permutations (e.g., GPT-3.5 :0.42
for columns vs.0 .77for rows), confirming that column order is more
affected by the permutation than row order. TRL models achieve
consistently higher PIderange scores (e.g., Gemma2 improves from0 .22
to0.61on columns), reflecting greater robustness to permutations.
Normalized AUC:captures the smoothness of recovery across the
full permutation trajectory from 𝑇0
0(full shuffle) to 𝑇𝑚
𝑛(identity).
For LLMs, AUC remains low under column permutations (e.g.,
Llama3 :0.22), reflecting poor global performance. TRL refinement
substantially increases AUC (up to0 .56for Gemma2 and0.49for
GPT-3.5 ), indicating that structural information encoded via TRL
supports smoother, more reliable recovery.
Key Findings.Across all metrics, LLM-derived embeddings are
more sensitive to column than row permutations, revealing a fun-
damental weakness in PRH. By contrast, TRL consistently improves
𝜌mono,PIderange , and AUC, demonstrating more stable invariance
curves and stronger empirical support for PRH. Notably, even
smaller models (e.g., nomic-embed-text ,270M parameters) bene-
fit from TRL and can rival or exceed the permutation robustness
of very large LLMs (e.g., GPT-3.5 ,175B parameters). This high-
lights that robustness is driven less by raw model size and more by
embedding structure and training objectives.5.4 Limitations of the methodology
Our analysis focuses on controlled row and column permutations
within fully structured tables, which we define using a symmetric
group action on tables ℎ(see Equation 3). However, in real docu-
ments, tables often feature merged cells and hierarchical or nested
headers, which contradict the assumptions required for a clean
row/column-group action. This discrepancy renders the definition
of the group action ℎill-posed, complicating our evaluation of the
proposed PRH. Therefore, extending PRH to accommodate such
irregularities is an important direction for future research.
Finally, if PI is truly necessary for claiming semantic faithful-
ness—or for improving table retrieval—this must be established
end-to-end rather than inferred from intrinsic diagnostics alone.
Concretely, it requires a benchmark thatpairs each table with con-
trolled row/column-permuted variantsand evaluates whether re-
trieval and downstream reasoning remain consistent across these
semantically equivalent inputs. Since such an end-to-end dataset
and protocol are not yet available, constructing them is left for
future work. We position this as part of the prospective vision this
paper aims to push: making invariance-based evaluation a standard
axis for next-generation table retrieval and table-RAG systems.
5.5 Conclusion from the serialization-bias
diagnostic
We instantiate thePlatonic Representation Hypothesis(PRH) for
tables and propose two permutation-based invariance metrics,
PIderange and𝜌mono, derived from CKA similarity. Applying these
diagnostics shows that permutation sensitivity is not uniform
across embedders. Several open-source models (e.g., DeepSeek-r1 ,
gemma2 ,llama3 ) exhibit different sensitivity, while openai_GPT3.5
is comparatively more stable underrowpermutations. Across all
8

evaluated models, however, we observe clear semantic drift under
columnpermutations: different column orders of the same table can
map far apart in embedding space, causing semantically equivalent
tables to be treated as distinct retrieval candidates and weakening
empirical support for PRH in current LLM-based embeddings.
In contrast, structure-aware TRL representations, anchored in
header–cell correspondences, consistently yield higher invariance
and reduced drift. Beyond reporting these findings, our metrics
provide practical diagnostics for table-centric RAG: they can be used
to compare embedders and to estimate how robust table retrieval
will be under layout changes.
Recommendation.For table-centric RAG, commercial embedders
such as openai_GPT3.5 can be strong choices when cost permits,
while lightweight open-source embedders such as nomic-embed-
text provide a competitive alternative. When robustness to lay-
out changes is a priority, TRL-refined embeddings are preferred;
however, table-specific TRL methods remain less mature than the
broader LLM embedding ecosystem.
6 Prospective Vision: Toward
Permutation-Invariant Retrieval
The diagnostic results above suggest a concrete bottleneck for table-
centric IR: under common serialization choices, embeddings can
drift substantially under benign permutations, especially column
re-ordering. This motivates a prospective shift in emphasis from
improving table-RAG via prompting and generation alone to im-
proving it at therepresentation layer, by explicitly targeting struc-
tural invariants. In this view, “Platonic representations” are not a
claim that a universal embedding already exists, but an agenda:
define invariance requirements that table representations should
satisfy and evaluate retrieval systems against them. We outline two
research directions that follow from this agenda.
First, we proposeOnline Table Question Answering (OTQA)
as a potential downstream use-case of invariant-aware embeddings.
If a representation is stable under table permutations, then a sub-
set of queries—such as point lookups and lightweight relational
checks—may be answerable through fast retrieval and simple geo-
metric operations in embedding space (e.g., nearest-neighbor selec-
tion under constraints), reducing reliance on repeated prompting
and generation. This direction is conditional on stronger invariance
than current generic embeddings provide, and therefore serves as a
clear target for future table representation learning.
Second, as industrial IR increasingly requirescomplex rela-
tional operations—including multi-step filtering, aggregation,
and cross-table linking—the need for a standardized, structure-
compatible representation becomes more pressing. Our findings
indicate that column-order sensitivity can undermine retrieval reli-
ability even before reasoning begins, suggesting that future systems
will benefit from embeddings that are stable under benign table
transformations. A practical implication is the need for end-to-end
benchmarks that include controlled permuted variants of the same
table and evaluate whether retrieval and downstream reasoning re-
main consistent. Establishing such protocols would help make PI a
standard evaluation axis for table retrieval and table-RAG pipelines.
Finally, invariant-aware representations connect directly to
three operations that are currently fragile in linearized table-RAGpipelines.Multi-step aggregationshould not depend on row order
and therefore benefits from row-stable representations.Rela-
tional filteringrequires capturing relationships (e.g., header–cell
semantics) beyond local lexical overlap.Cross-table referencing
requires identifying compatible columns or entities across hetero-
geneous tables despite schema differences. These directions are
prospective, but they are grounded in the failure modes revealed
by our diagnostics and provide concrete targets for future work on
permutation-invariant retrieval.
7 Conclusion
In this prospective paper, we have explored the critical intersec-
tion of Table Representation Learning (TRL) and the Retrieval-
Augmented Generation (RAG) landscape through the lens of the
Platonic Representation. We empirically demonstrated that the ubiq-
uitous reliance on table linearization in current Information Re-
trieval (IR) pipelines can introduce a destructive bias, dissolving the
essential structural invariants—specifically row-column PI—that de-
fine tabular data. This structural dissolution persists even in models
specifically fine-tuned for tabular tasks, highlighting a fundamental
need to rethink how tables are learned.
By positioning Platonic representations as a necessary paradigm
shift, we outlined a dual-pathway vision for the future of IR. We
introduced the concept ofOnline Table Question Answering
(OTQA), demonstrating how invariant embeddings can facilitate
direct, "prompt-less" reasoning within the latent space, thereby dras-
tically reducing the computational overhead of RAG. Furthermore,
we detailed how these robust representations provide the "struc-
tural fuel" for complex industrial operations, including multi-step
aggregation and semantic cross-table referencing.
Finally, we posit that the quest for the "Platonic representation"
of structured data is not merely a theoretical exercise but a practical
prerequisite for the next generation of intelligent systems. As the IR
field moves toward a RAG-centric future, adopting a standardized,
invariant representation for tables will be the cornerstone that
transforms generative models from surface-level retrievers into
deep analytical engines.
8 Acknowledgments
This research was supported by funding from the Flemish Govern-
ment under the “Onderzoeksprogramma Artificiële Intelligentie
(AI) Vlaanderen” program.
9

References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report.arXiv preprint arXiv:2303.08774
(2023).
[2]Sercan Ö Arik and Tomas Pfister. 2021. Tabnet: Attentive interpretable tabular
learning. InProceedings of the AAAI conference on artificial intelligence, Vol. 35.
6679–6687.
[3]Pei Chen, Soumajyoti Sarkar, Leonard Lausen, Balasubramaniam Srinivasan,
Sheng Zha, Ruihong Huang, and George Karypis. 2023. Hytrel: Hypergraph-
enhanced tabular data representation learning.Advances in Neural Information
Processing Systems36 (2023), 32173–32193.
[4]Si-An Chen, Lesly Miculicich, Julian Eisenschlos, Zifeng Wang, Zilong Wang,
Yanfei Chen, Yasuhisa Fujii, Hsuan-Tien Lin, Chen-Yu Lee, and Tomas Pfister. 2024.
Tablerag: Million-token table understanding with language models.Advances in
Neural Information Processing Systems37 (2024), 74899–74921.
[5]Sarthak Dash, Sugato Bagchi, Nandana Mihindukulasooriya, and Alfio Gliozzo.
2022. Permutation invariant strategy using transformer encoders for table un-
derstanding. InFindings of the Association for Computational Linguistics: NAACL
2022. 788–800.
[6]Xiang Deng, Yoshihiko Suhara, Jinfeng Zhang, Yuliang Li, and Wang-Chiew
Tan. 2020. TURL: Table Understanding through Representation Learning. In
Proceedings of the VLDB Endowment (PVLDB), Vol. 14. 307–319.
[7]Dawei Gao, Haibin Wang, Yaliang Li, Xiuyu Sun, Yichen Qian, Bolin Ding, and
Jingren Zhou. 2024. Text-to-SQL Empowered by Large Language Models: A
Benchmark Evaluation.Proc. VLDB Endow.17, 5 (Jan. 2024), 1132–1145. doi:10.
14778/3641204.3641221
[8]Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. 2021.
Revisiting deep learning models for tabular data.Advances in neural information
processing systems34 (2021), 18932–18943.
[9]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek
Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex
Vaughan, et al .2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783
(2024).
[10] Jonathan Herzig, Paul Nowak, Thomas Müller, Francesco Piccinno, and Julian
Eisenschlos. 2020. TAPAS: Weakly Supervised Table Parsing via Pre-training.
InProceedings of the 58th Annual Meeting of the Association for Computational
Linguistics (ACL). 4320–4333.
[11] Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. 2024. Posi-
tion: The Platonic Representation Hypothesis. InProceedings of the 41st Interna-
tional Conference on Machine Learning (Proceedings of Machine Learning Research,
Vol. 235), Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller,
Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (Eds.). PMLR, 20617–20642.
https://proceedings.mlr.press/v235/huh24a.html
[12] Xingyu Ji, Parker Glenn, Aditya G Parameswaran, and Madelon Hulsebos.
2025. Target: Benchmarking table retrieval for generative tasks.arXiv preprint
arXiv:2505.11545(2025).
[13] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. 2019.
Similarity of neural network representations revisited. InInternational conference
on machine learning. PMLR, 3519–3529.[14] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[15] Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle
Rifinski Fainman, Dongmei Zhang, and Surajit Chaudhuri. 2024. Table-gpt: Table
fine-tuned gpt for diverse table tasks.Proceedings of the ACM on Management of
Data2, 3 (2024), 1–28.
[16] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Cheng-
gang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al .2024. Deepseek-v3
technical report.arXiv preprint arXiv:2412.19437(2024).
[17] Ari Morcos, Maithra Raghu, and Samy Bengio. 2018. Insights on representational
similarity in neural networks with canonical correlation.Advances in neural
information processing systems31 (2018).
[18] Zach Nussbaum, John X Morris, Brandon Duderstadt, and Andriy Mulyar. 2024.
Nomic embed: Training a reproducible long context text embedder.arXiv preprint
arXiv:2402.01613(2024).
[19] Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. 2017.
Svcca: Singular vector canonical correlation analysis for deep learning dynamics
and interpretability.Advances in neural information processing systems30 (2017).
[20] Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han-yu
Wang, Liu Haisu, Quan Shi, Zachary S Siegel, Michael Tang, et al .2025. BRIGHT:
A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval. Inter-
national Conference on Learning Representations (ICLR).
[21] Willy Carlos Tchuitcheu, Tan Lu, and Ann Dooms. 2024. Table representation
learning using heterogeneous graph embedding.Pattern Recognition156 (2024),
110734. doi:10.1016/j.patcog.2024.110734
[22] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy
Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari,
Alexandre Ramé, et al .2024. Gemma 2: Improving open language models at a
practical size.arXiv preprint arXiv:2408.00118(2024).
[23] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in neural information processing systems30 (2017).
[24] Xianjie Wu, Jian Yang, Linzheng Chai, Ge Zhang, Jiaheng Liu, Xeron Du, Di
Liang, Daixin Shu, Xianfu Cheng, Tianzhen Sun, et al .2025. Tablebench: A com-
prehensive and complex benchmark for table question answering. InProceedings
of the AAAI Conference on Artificial Intelligence, Vol. 39. 25497–25506.
[25] Pengcheng Yin, Graham Neubig, Wen-tau Yih, and Sebastian Riedel. 2020.
TaBERT: Pretraining for Joint Understanding of Textual and Tabular Data. In
Proceedings of the 58th Annual Meeting of the Association for Computational Lin-
guistics (ACL). 8413–8426.
[26] Liangyu Zha, Junlin Zhou, Liyao Li, Rui Wang, Qingyi Huang, Saisai Yang,
Jing Yuan, Changbao Su, Xiang Li, Aofeng Su, et al .2023. Tablegpt: Towards
unifying tables, nature language and commands into one gpt.arXiv preprint
arXiv:2307.08674(2023).
[27] Tianshu Zhang, Xiang Yue, Yifei Li, and Huan Sun. 2024. Tablellama: Towards
open large generalist models for tables. InProceedings of the 2024 Conference
of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 1: Long Papers). 6024–6044.
10