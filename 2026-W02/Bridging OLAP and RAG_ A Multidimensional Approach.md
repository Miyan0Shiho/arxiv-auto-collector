# Bridging OLAP and RAG: A Multidimensional Approach to the Design of Corpus Partitioning

**Authors**: Dario Maio, Stefano Rizzi

**Published**: 2026-01-07 09:37:36

**PDF URL**: [https://arxiv.org/pdf/2601.03748v1](https://arxiv.org/pdf/2601.03748v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems are increasingly deployed on large-scale document collections, often comprising millions of documents and tens of millions of text chunks. In industrial-scale retrieval platforms, scalability is typically addressed through horizontal sharding and a combination of Approximate Nearest-Neighbor search, hybrid indexing, and optimized metadata filtering. Although effective from an efficiency perspective, these mechanisms rely on bottom-up, similarity-driven organization and lack a conceptual rationale for corpus partitioning. In this paper, we claim that the design of large-scale RAG systems may benefit from the combination of two orthogonal strategies: semantic clustering, which optimizes locality in embedding space, and multidimensional partitioning, which governs where retrieval should occur based on conceptual dimensions such as time and organizational context. Although such dimensions are already implicitly present in current systems, they are used in an ad hoc and poorly structured manner. We propose the Dimensional Fact Model (DFM) as a conceptual framework to guide the design of multidimensional partitions for RAG corpora. The DFM provides a principled way to reason about facts, dimensions, hierarchies, and granularity in retrieval-oriented settings. This framework naturally supports hierarchical routing and controlled fallback strategies, ensuring that retrieval remains robust even in the presence of incomplete metadata, while transforming the search process from a 'black-box' similarity matching into a governable and deterministic workflow. This work is intended as a position paper; its goal is to bridge the gap between OLAP-style multidimensional modeling and modern RAG architectures, and to stimulate further research on principled, explainable, and governable retrieval strategies at scale.

## Full Text


<!-- PDF content starts -->

Bridging OLAP and RAG: A Multidimensional Approach to the
Design of Corpus Partitioning
Dario Maio1,*, Stefano Rizzi1
1DISI - University of Bologna, Viale Risorgimento, 2, Bologna, 40136 Italy
Abstract
Retrieval-Augmented Generation (RAG) systems are increasingly deployed on large-scale document collections, often comprising
millions of documents and tens of millions of text chunks. In industrial-scale retrieval platforms, scalability is typically addressed
through horizontal sharding and a combination of Approximate Nearest-Neighbor search, hybrid indexing, and optimized metadata
filtering. Although effective from an efficiency perspective, these mechanisms rely on bottom-up, similarity-driven organization and
lack a conceptual rationale for corpus partitioning. In this paper, we claim that the design of large-scale RAG systems may benefit from
the combination of two orthogonal strategies: semantic clustering, which optimizes locality in embedding space, and multidimensional
partitioning, which governs where retrieval should occur based on conceptual dimensions such as time and organizational context.
Although such dimensions are already implicitly present in current systems, they are used in an ad hoc and poorly structured manner.
We propose the Dimensional Fact Model (DFM) as a conceptual framework to guide the design of multidimensional partitions for
RAG corpora. The DFM provides a principled way to reason about facts, dimensions, hierarchies, and granularity in retrieval-oriented
settings. This framework naturally supports hierarchical routing and controlled fallback strategies, ensuring that retrieval remains
robust even in the presence of incomplete metadata, while transforming the search process from a ’black-box’ similarity matching into
a governable and deterministic workflow. This work is intended as a position paper; its goal is to bridge the gap between OLAP-style
multidimensional modeling and modern RAG architectures, and to stimulate further research on principled, explainable, and governable
retrieval strategies at scale.
Keywords
Retrieval-Augmented Generation, Partitioning, Multidimensional Model, Conceptual Modeling, Metadata-Driven Retrieval, Retrieval
Governance
1. Introduction
In recent years, advances in transformer-based architectures
have enabled Large Language Models (LLMs) to achieve re-
markable capabilities in natural language understanding
and generation [ 1,2], supporting a wide range of applica-
tions that require flexible access to large and heterogeneous
knowledge sources. These models are increasingly regarded
as foundation models, providing general-purpose linguistic
capabilities while also raising challenges related to knowl-
edge freshness, control, and integration with external data
sources [ 3]. However, LLMs are inherently parametric and
static, and their knowledge is limited to what has been en-
coded during training. To address this limitation, Retrieval-
Augmented Generation (RAG) has emerged as a dominant
architectural paradigm, combining generative models with
external retrieval mechanisms over large document collec-
tions [4].
In a RAG system, user queries are used to retrieve relevant
textual fragments from a corpus, which are then provided
as contextual input to the LLM during generation. This
approach allows systems to ground responses in up-to-date
and domain-specific information, and has become the de
facto standard for deploying LLM-based applications over
enterprise-scale data. As RAG systems are increasingly
applied to collections comprising millions of documents
and tens or hundreds of millions of text chunks, issues of
scalability, robustness, and governability of retrieval become
central.
Current large-scale RAG architectures primarily rely on
similarity-based retrieval over vector representations, often
combined with hybrid indexing and approximate nearest-
neighbor search techniques [ 5,6]. Modern industrial RAG
systems already combine vector-based similarity search
with clustering and metadata filtering strategies; however,
*Corresponding author.
/envel⌢pe-⌢pendario.maio@unibo.it (D. Maio); stefano.rizzi@unibo.it (S. Rizzi)
/orcidID 0000-0002-0094-0022 (D. Maio); 0000-0002-4617-217X (S. Rizzi)these mechanisms are primarily driven by bottom-up se-
mantic similarity and do not provide an explicit conceptual
rationale for corpus partitioning.
Scalability in industrial RAG systems is typically achieved
through horizontal sharding, hybrid indexing strategies, and
metadata-based filtering used to restrict the retrieval space
when possible. While effective from an efficiency stand-
point, these solutions are largely implementation-driven:
corpus organization is dominated by embedding-space local-
ity, and logical constraints on where retrieval should occur
are handled in an ad hoc and weakly structured manner.
In parallel, the problem of organizing and querying large
volumes of data has been long addressed in Online Ana-
lytical Processing (OLAP) systems and data warehouses
[7,8]. Multidimensional modeling provides a principled
framework for representing facts, dimensions, hierarchies,
and levels of granularity, supporting systematic reasoning
about data organization. In particular, the Dimensional Fact
Model (DFM) operates at a conceptual level, guiding design
decisions while remaining agnostic with respect to imple-
mentation details such as physical storage and execution
strategies [9].
In this paper, we argue that large-scale RAG systems
would benefit from a similar conceptual design layer. We
propose to reinterpret the DFM as a framework for the mul-
tidimensional partitioning of unstructured document cor-
pora used in RAG. In our view, effective RAG architectures
should explicitly separate two orthogonal concerns: (i) the
multidimensional partitioningof the corpus along domain-
relevant dimensions, used to guide routing decisions and
determine where retrieval should take place; and (ii) se-
mantic clustering and sub-clustering applied within each
multidimensional partition, used to support similarity-based
retrieval over vector representations. Note that clustering
and sub-clustering are cited solely as representative exam-
ples of semantic organization strategies that may be adopted
within multidimensional partitions. The proposed frame-
work does not impose any specific retrieval logic, nor does

it interfere with the internal filtering, indexing, or execution
strategies adopted by existing industrial retrieval systems.
Rather, it provides a principled way to govern retrieval by
making conceptual assumptions explicit and controllable.
The proposed approach is intentionally agnostic with re-
spect to underlying system architectures. Multidimensional
partitions are defined independently of physical sharding,
although in practical deployments a mapping between mul-
tidimensional partitions and physical shards naturally exists
and follows standard scalability and load-balancing prac-
tices. Similarly, the presence or absence of a data ware-
house is orthogonal to the applicability of the framework.
When a data warehouse or other structured repositories
are available, they facilitate the design of dimensions as
their multidimensional schemata may be available. When
such infrastructures are absent, the DFM remains a purely
conceptual tool for reasoning about corpus organization
and retrieval strategies. In both cases, the problem of meta-
data extraction and management is inherent to large-scale
retrieval systems and is not introduced by the proposed
framework.
This work is a position paper and addresses methodolog-
ical issues. Rather than proposing new retrieval algorithms
or experimental evaluations, it introduces a conceptual per-
spective that complements existing RAG architectures, with
the goal of stimulating discussion on principled, explainable,
and governable design strategies for retrieval over massive
unstructured corpora. The contributions of this work can
be summarized as follows:
•We identify a conceptual gap in current RAG archi-
tectures, where corpus partitioning and metadata-
based routing are critical for scalability and robust-
ness, yet are addressed in an ad hoc, implementation-
driven manner without an explicit conceptual design
layer.
•We argue that effective large-scale RAG systems re-
quire the explicit combination of two orthogonal
organizational principles: semantic clustering for
embedding-space locality, and multidimensional par-
titioning for governing where retrieval should occur
based on domain-relevant dimensions.
•We propose the DFM as a conceptual framework
to guide multidimensional partitioning of large un-
structured corpora for retrieval purposes.
•We show that a DFM-based design naturally enables
hierarchical routing and controlled fallback strate-
gies during retrieval, supporting deterministic, ex-
plainable, and governable retrieval workflows even
in the presence of incomplete or partially missing
metadata.
•We present an illustrative example from a com-
plex legal domain, involving collections of judicial
decisions or contractual documents at industrial
scale, to demonstrate how multidimensional mod-
eling can structure the retrieval space along mean-
ingful dimensions without constraining underlying
similarity-based retrieval mechanisms.
The remainder of the paper is organized as follows. Sec-
tion 2 reviews related work on RAG architectures, metadata-
driven retrieval, and multidimensional modeling. Section 3
introduces the DFM and discusses its reinterpretation in
a retrieval-oriented setting. Section 4 presents illustrativeexamples drawn from complex application domains to mo-
tivate the proposed approach. Finally, Section 5 discusses
the implications of the framework, its limitations, and some
open directions for future research.
2. Related Work
2.1. Conceptual Modeling and LLMs in
OLAP
Conceptual modeling has long played a central role in the
design of data warehouses and OLAP systems. The mul-
tidimensional model provides a principled framework for
representing facts, dimensions, hierarchies, and granularity
at a conceptual level, supporting systematic reasoning about
aggregation paths, drill-down and roll-up operations, and
alignment between analytical requirements and physical
implementations [ 7,8]. The DFM formalizes these concepts
at the conceptual level and has been widely adopted as a ref-
erence model for data warehouse design [ 9]. Recent studies
have investigated the role of LLMs in the conceptual design
of OLAP systems. Rizzi [ 10] explores the use of ChatGPT to
refine draft conceptual schemata in supply-driven design of
multidimensional cubes, while Rizzi et al. [ 11] provide an
empirical investigation of the potential and limitations of
LLMs in assisting multidimensional modeling tasks. These
works demonstrate that LLMs can effectively operate at the
conceptual level of OLAP design.
Research on natural language interfaces to databases has
a long tradition in the data management community, with
the goal of translating user questions into formal query
languages such as SQL. Recent advances in neural and
transformer-based language models have significantly revi-
talized this area, leading to modern Text-to-SQL and schema-
aware semantic parsing approaches that achieve strong per-
formance on complex and cross-domain benchmarks such
asSpider[ 12]. This line of work primarily aims at improv-
ing accessibility to structured data by mapping user intents
to queries over an existing database schema. More recently,
LLMs have been explored as flexible interfaces for analytical
systems, supporting conversational interaction and explana-
tion over OLAP and Business Intelligence platforms [ 13]. In
these approaches, LLMs act as front-ends for querying and
exploring multidimensional data, while assuming that the
underlying OLAP schema and data organization are given.
Despite its maturity and widespread use in analytical sys-
tems, the multidimensional model has not been explicitly
applied to the design of retrieval structures for large-scale
unstructured document collections. In current RAG archi-
tectures, conceptual modeling is largely absent: retrieval
units, metadata, and partitions are defined in an ad hoc man-
ner, and semantic clustering is relied upon as the primary
organizational principle.
2.2. Partitioning and Metadata in RAG
Several works in the RAG literature have recognized the
importance of structuring the retrieval space to improve gen-
eration quality. Wang et al. [ 14] propose a multi-partition
RAG architecture (M-RAG), where multiple partitions act
as basic units for retrieval and generation, showing that
partitioning can significantly improve performance across
different language generation tasks. In this approach, how-
ever, partitions are treated as operational units optimized

during execution, without an explicit conceptual schema
guiding their design.
Other studies explore the use of metadata to filter or re-
strict the retrieval space in RAG pipelines. Di Oliveira et
al. [15] propose a two-step RAG approach in which meta-
data filtering precedes vector-based retrieval, demonstrating
that structured metadata can effectively reduce noise and
improve statistical evaluation of LLM outputs.
Finally, early attempts to integrate OLAP-style workflows
with RAG architectures have been proposed. Ouafiq and
Saadane [ 16] introduce a Retrieval-Augmented OLAP archi-
tecture in which multidimensional analytical structures are
used to support the reasoning process of generative models.
While these approaches confirm the relevance of multidi-
mensional organization and metadata-driven partitioning in
retrieval systems, they do not provide a general conceptual
framework for the design of partitions over unstructured
document corpora.
2.3. RAG with Structured Data Sources
While the previous works focus on partitioning and
metadata-driven control of the retrieval space, another
line of research investigates the integration of structured
data sources into RAG pipelines. Several works combine
RAG with relational databases, knowledge bases, or graph
databases, enabling LLMs to retrieve both unstructured text
and structured facts during answer generation [ 4,17,18]. In
these systems, structured queries are typically executed via
external tools or dedicated query engines, while unstruc-
tured retrieval relies on vector similarity search over text
passages.
Hybrid architectures that interleave semantic retrieval
with database querying have been proposed to improve fac-
tual accuracy, reduce hallucinations, and support multi-step
reasoning [ 17,18]. However, in most of these approaches
the unstructured corpus is treated as a flat or weakly struc-
tured collection, possibly augmented with simple metadata
filters. The design of partitions for the text corpus is gener-
ally left to heuristic decisions or similarity-based clustering,
without an explicit conceptual model guiding partitioning
choices.
2.4. Positioning of This Work
In contrast to the aforementioned lines of research, this pa-
per focuses on the conceptual design of multidimensional
partitions for retrieval over unstructured corpora. Rather
than introducing new retrieval algorithms or hybrid query
mechanisms, we argue that principles from multidimen-
sional modeling can provide a formal design layer for large-
scale RAG systems that is currently missing. To the best of
our knowledge, this is the first work that explicitly proposes
the DFM as a conceptual framework to reason about multi-
dimensional partitioning, hierarchical routing, and fallback
strategies in RAG.
3. Applying the DFM to
Retrieval-Oriented Partitioning
In OLAP systems, the DFM provides a conceptual represen-
tation of analytical data in terms of facts, dimensions, hier-
archies, and levels of granularity, independently of physical
storage and query execution strategies [ 8]. In this context,a fact represents a business event of interest, such as an
invoice or a sales transaction, described through multiple
dimensions such as time, customer, product, and location.
Building on this well-established abstraction, we reinter-
pret the basic retrieval unit in a RAG system (e.g., a text
chunk) as the analogue of a fact, while dimensions corre-
spond to domain-relevant attributes that characterize docu-
ments and guide retrieval decisions, such as time, document
type, jurisdiction, or organizational context. Hierarchies
within dimensions capture meaningful levels of abstraction
and support progressive refinement of the retrieval space.
Importantly, the proposed model does not prescribe how
such dimensions are physically indexed or stored, but only
how they are conceptually organized and combined. In
other words, in this work, we reinterpret these concepts in
the context of RAG, where the primary goal is not aggrega-
tion but the governable selection of relevant portions of a
large unstructured corpus.
Within this framework, multidimensional partitioning
and semantic retrieval play complementary roles. Multidi-
mensional partitions, defined along one or more dimensions,
are used to route queries and determine where retrieval
should occur. Semantic clustering and sub-clustering are
then applied within each multidimensional partition to sup-
port similarity-based retrieval over vector representations.
This separation allows embedding-based methods to oper-
ate locally, while global retrieval behavior remains governed
by explicit conceptual choices.
The model explicitly accommodates incomplete or miss-
ing metadata by allowing dedicated fallback partitions that
collect documents whose dimensional values cannot be re-
liably determined. From a multidimensional perspective,
such partitions correspond to facts with optional dimen-
sional coordinates and are treated as first-class elements of
conceptual design, ensuring robustness without relying on
implicit or ad hoc behaviors.
Overall, the proposed reinterpretation positions the DFM
as a conceptual design layer for large-scale RAG systems,
enabling principled reasoning about corpus organization,
routing strategies, and retrieval scope, while remaining fully
agnostic with respect to underlying architectures and exe-
cution mechanisms. By making partitioning assumptions
explicit, the DFM provides a stable conceptual reference
for governing the evolution of routing, clustering strate-
gies, and fallback policies as the corpus and usage patterns
change over time.
4. Illustrative Example
To illustrate the proposed approach, we consider a large-
scale legal domain scenario involving collections of judicial
decisions or contractual documents. Such corpora are char-
acterized by high structural complexity, strong domain se-
mantics, and industrial-scale volumes, often comprising mil-
lions of documents and hundreds of millions of text chunks
when processed for RAG. In the absence of a multidimen-
sional perspective, retrieval in such settings is typically
performed over a flat or weakly-structured corpus, possi-
bly augmented with ad hoc metadata filters. As a result,
semantically similar fragments originating from different
jurisdictions, procedural levels, or temporal contexts may be
retrieved together, despite being conceptually incompatible
for a given information need.
Using a multidimensional perspective, the corpus can

Figure 1:DFM schema for judicial opinion chunks in the le-
gal domain. The box represents the basic retrieval unit with its
quantitative measures; dashed areas represent dimensions (e.g.,
Structure), while circles represent their hierarchical levels from
the finest ones (those attached to the box, e.g., paragraph) to the
coarsest ones (the leaves, e.g., section). Dashes on arcs model
optional dimensions.
instead be logically partitioned along legally-meaningful
dimensions, such as time, jurisdiction, procedural posture,
and document type. These dimensions define partitions that
guide query routing and determine where retrieval should
occur. For instance, a query concerning recent appellate
decisions in a specific jurisdiction can be routed to the cor-
responding multidimensional partitions, while documents
lacking reliable temporal or jurisdictional metadata are ex-
plicitly assigned to a dedicated fallback partition. Within
each multidimensional partition, semantic clustering and
sub-clustering strategies can be applied to organize vector
representations and support similarity-based retrieval at dif-
ferent granularities. In this way, embedding-based methods
operate within conceptually coherent subsets of the corpus,
while global control over the retrieval space is preserved by
the multidimensional partitioning layer.
Figure 1 illustrates a possible DFM schema for judicial
opinion chunks in the context of the United States legal
system. The basic retrieval unit corresponds to anopinion
chunk, i.e., a portion of a judicial decision obtained by seg-
menting court opinions into manageable text fragments for
RAG. Opinion chunks are derived from documents such
as judicial opinions, orders, or memoranda, typically avail-
able in formats such as PDF or HTML and processed into
text chunks during ingestion. They correspond to atomic
units of legal discourse that may be retrieved and provided
as contextual evidence to a language model. Each opinion
chunk is characterized by multiple orthogonal dimensions
commonly used in legal research and analysis, including:
•Jurisdiction, capturing the institutional context of
the decision in terms of (from the finest to the coars-est level)division,specific court,court level(e.g.,
’supreme’ and ’appeals district’), andsystem(either
’federal’ or ’state’);
•Legal subject, which gives a taxonomic description
of the substantive legal area addressed by the opin-
ion with levels such asarea(’civil’, ’criminal’, ’con-
stitutional’) andbroad topic(e.g. ’contracts’ and
’torts’);
•Document type, distinguishing different kinds of
legal documents and opinions mainly in terms of
category(e.g., ’opinion’ and ’order’) andtype(e.g.,
’majority’ and ’dissent’);
•Procedural posture, describing the procedural con-
text in which the decision was issued (e.g., ’trial’ and
’appeal’);
•Structure, identifying the rhetorical or functional
role of the chunk within the opinion (e.g., ’facts’,
’legal analysis’, ’holding’ for levelsection).
Opinion chunks whose dimensional values cannot be re-
liably determined are explicitly associated with dedicated
fallback partitions; in the DFM, these are represented as
optional dimensions (in our example,Legal subjectandJu-
risdiction).
Note that the proposed schema is intended solely as a con-
ceptual reference and does not prescribe any specific physi-
cal storage, indexing strategy, or system architecture. The
mapping between multidimensional partitions and phys-
ical design choices—such as shards, indexes, and cluster
organizations—is deployment-dependent and remains out-
side the scope of this work.
Within this framework, multidimensional information
is used to guide query routing and determine the scope of
retrieval before any similarity-based search is performed.
Queries are analyzed to extract dimensional constraints —–
explicit or implicit–— which are then used to select the
relevant multidimensional partitions of the corpus. For ex-
ample, a query seeking recent Supreme Court decisions on a
specific legal doctrine may activate constraints on theTime,
Jurisdiction, andLegal subjectdimensions, routing retrieval
to partitions corresponding to recent Supreme Court opin-
ions addressing that doctrine. Similarly, queries aimed at
comparing appellate decisions across multiple circuits may
intentionally span several jurisdictional partitions, enabling
structured comparison rather than conflating semantically
similar but institutionally distinct decisions.
Dimensions and their hierarchical levels further allow
retrieval to be focused on specific parts of opinions, such
as legal analysis sections, avoiding the inclusion of factual
background or procedural history when these are not rele-
vant to the information need. When dimensional constraints
are weak, ambiguous, or incomplete, fallback partitions (rep-
resented via optional dimensions) ensure that retrieval re-
mains robust and explicitly governed rather than relying on
implicit behavior.
Within each selected multidimensional partition, seman-
tic retrieval techniques—such as vector similarity search
supported by clustering or sub-clustering—can then be ap-
plied to identify the most relevant opinion chunks. In this
way, dimensional routing governswhereretrieval should
occur, while semantic similarity determineswhatcontent is
most relevant within that scope.
It is important to note that the proposed multidimensional
partitioning does not impose restrictive access patterns on
retrieval. In scenarios where users explicitly seek broad

coverage, such as surveying legal precedents across multi-
ple years, courts, or jurisdictions, queries may legitimately
span multiple or all multidimensional partitions. In such
cases, the proposed framework does not reduce the inherent
computational cost of retrieval and is not intended to do so.
Rather, it provides an explicit and controlled way to express
and govern these retrieval scopes, making broad searches a
deliberate design choice rather than an implicit side effect
of a flat corpus organization.
A similar rationale applies to large-scale biomedical and
healthcare-related document collections, such as scientific
articles, clinical guidelines, and technical reports, which are
typically available in heterogeneous formats and processed
into text chunks for RAG. These corpora may comprise
millions of documents and hundreds of millions of chunks
and are characterized by strong domain semantics and rich,
yet often incomplete, metadata. Meaningful dimensions
may include publication time, medical condition, type of
study or document, target population, and source. Parti-
tioning along such dimensions can guide retrieval toward
conceptually appropriate subsets of the corpus (e.g., recent
guidelines versus primary studies, or adult versus pediatric
populations), while semantic retrieval techniques operate
within each partition. As in the legal scenario, documents
whose metadata cannot be reliably determined can be con-
sistently assigned to dedicated fallback areas of the retrieval
space, ensuring robustness without implicit or uncontrolled
retrieval behavior.
Similarly, while certain dimensions such as time may
appear universal and already widely used in practice, the
contribution of the DFM lies not in the introduction of in-
dividual dimensions, but in their systematic organization,
combination, and governance. By making dimensions, hi-
erarchies, and fallback strategies explicit at the conceptual
level, the proposed approach supports flexible routing strate-
gies that adapt to different query intents, without constrain-
ing physical execution or performance optimization mecha-
nisms.
5. Discussion and Implications
This paper does not introduce a new retrieval technique or
system architecture, but rather it advocates the introduc-
tion of a conceptual perspective on the design of large-scale
RAG systems, inspired by well-established principles from
multidimensional modeling. As such, its primary impli-
cations concern governability, explainability, and design
clarity, rather than performance optimization. By making
partitioning decisions explicit and grounding them in a mul-
tidimensional schema, the proposed framework enables sys-
tematic reasoning about where retrieval should occur, how
retrieval scopes are defined, and how fallback strategies are
handled.
From a query perspective, RAG systems are expected to
support heterogeneous information needs, ranging from
document-centric queries (e.g., lookup and document-level
summarization), to retrieval-oriented queries (e.g., thematic
or multi-document retrieval), per-document views, corpus-
level aggregation, and multi-step reasoning queries. These
different query types naturally interact with the retrieval
space at different granularities and scopes. Accordingly, dif-
ferent classes of queries may interact with multidimensional
partitioning in different ways: while some queries require
minimal or no partition-based restriction, others benefitfrom explicit control over the retrieval scope to ensure com-
pleteness, correctness, and reproducibility of results.
An important implication of this approach is its flexibil-
ity with respect to different query intents. While multi-
dimensional partitioning can be used to restrict retrieval
to focused subsets of the corpus, it does not prevent broad
queries that legitimately span multiple or all partitions, such
as exploratory searches over large temporal or organiza-
tional ranges. In these cases, the computational cost of
retrieval remains inherent to the task and is not reduced by
the proposed framework. Instead, the framework provides
an explicit and controlled way to express such retrieval
scopes, avoiding implicit or unintended behavior. While
performance improvements are not the primary goal of the
proposed framework, queries whose information needs can
be satisfied within a limited set of multidimensional parti-
tions may naturally benefit from a reduced retrieval scope.
The proposed use of the DFM does not rely on the nov-
elty of individual dimensions, some of which may be widely
adopted in practice (e.g., time). Its contribution lies in the
systematic organization of dimensions, hierarchies, and lev-
els of granularity, as well as in the explicit treatment of
incomplete or missing metadata. These aspects are often
handled implicitly in existing systems, whereas the pro-
posed approach elevates them to first-class design decisions.
Some limitations should be acknowledged. Our frame-
work does not address the problem of metadata extraction,
which remains a challenge in large-scale document process-
ing regardless of the retrieval architecture. Moreover, it
does not prescribe how multidimensional partitions should
be mapped to physical shards, indexed, or executed, leaving
these choices to existing industrial solutions and optimiza-
tion strategies. Consequently, the framework should be
viewed as complementary to, rather than a replacement for,
current retrieval technologies.
Overall, we believe that the multidimensional perspec-
tive introduced in this work will be able to support more
principled, transparent, and adaptable retrieval architec-
tures, and we hope it will stimulate further research on
conceptual modeling approaches for retrieval over large
unstructured corpora. We encourage the development of
open-source tools to support conceptual design in this direc-
tion, particularly for assisting the definition and governance
of multidimensional partitions in RAG systems; when data
warehouses or other structured repositories are already in
place, such tools should naturally integrate with existing
infrastructures rather than replace them.
Acknowledgment
We acknowledge financial support under the PR Puglia
FESR FSE+ 2021-2027 - European Regional Development
Fund, Net Service project QUICK SHIELD of Puglia Region,
CUP:B85H24000920007.
References
[1]A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit,
L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, At-
tention is all you need, in: Advances in neural infor-
mation processing systems, 2017, pp. 5998–6008.
[2]T. Brown, et al., Language models are few-shot learn-
ers, Advances in neural information processing sys-
tems 33 (2020) 1877–1901.

[3]R. Bommasani, et al., On the opportunities and risks
of foundation models, CoRR abs/2108.07258 (2021).
[4]P. Lewis, et al., Retrieval-augmented generation for
knowledge-intensive NLP tasks, in: Advances in Neu-
ral Information Processing Systems, 2020.
[5]Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai,
J. Sun, H. Wang, Retrieval-augmented generation for
large language models: A survey, Frontiers of Com-
puter Science 18 (2024) 1–20.
[6]J. Chen, et al., Benchmarking large language models in
retrieval-augmented generation, CoRR abs/2309.01431
(2024).
[7]R. Kimball, M. Ross, The Data Warehouse Toolkit:
The Definitive Guide to Dimensional Modeling, Wiley,
2013.
[8]M. Golfarelli, S. Rizzi, Data Warehouse Design: Mod-
ern Principles and Methodologies, McGraw-Hill, 2009.
[9]M. Golfarelli, D. Maio, S. Rizzi, The dimensional fact
model: A conceptual model for data warehouses, Inter-
national Journal of Cooperative Information Systems
7 (1998) 215–247.
[10] S. Rizzi, Using ChatGPT to refine draft conceptual
schemata in supply-driven design of multidimensional
cubes, CoRR abs/2502.02238 (2025).
[11] S. Rizzi, M. Francia, E. Gallinucci, M. Golfarelli, Con-
ceptual design of multidimensional cubes with LLMs:
An investigation, Data & Knowledge Engineering
(2025).
[12] T. Yu, R. Zhang, K. Yang, M. Yasunaga, D. Wang, Z. Li,
J. Ma, I. Li, Q. Yao, S. Roman, D. Radev, Spider: A large-
scale human-labeled dataset for complex and cross-
domain semantic parsing and Text-to-SQL task, in:
Proceedings of the Conference on Empirical Methods
in Natural Language Processing, 2018.
[13] S. Bimonte, S. Rizzi, Text-to-MDX: LLM-assisted gener-
ation of MDX queries from user questions, in: Proceed-
ings of the International Conference on Conceptual
Modeling, Poitiers, France, 2025, pp. 165–181.
[14] Z. Wang, S. Teo, J. Ouyang, Y. Xu, W. Shi, M-RAG: Re-
inforcing large language model performance through
retrieval-augmented generation with multiple parti-
tions, in: Proceedings of the Annual Meeting of the
Association for Computational Linguistics, 2024, pp.
1966–1978.
[15] V. Di Oliveira, P. C. Brom, L. Weigang, Two-step RAG
for metadata filtering and statistical LLM evaluation,
IEEE Latin America Transactions 23 (2025).
[16] E. M. Ouafiq, R. Saadane, Retrieval-augmented OLAP:
Generative AI architecture for smart systems & equip-
ment, in: Proceedings of the AAAI Symposium, 2025,
pp. 304–312.
[17] S. Yao, et al., ReAct: Synergizing reasoning and acting
in language models, in: Proceedings of the Interna-
tional Conference on Learning Representations, 2023.
[18] T. Schick, et al., Toolformer: Language models can
teach themselves to use tools, in: Advances in Neural
Information Processing Systems, 2023.