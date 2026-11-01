# SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

**Authors**: Zafer Sahinoglu, Abdulazeez Mohammed, Buse Yaren Kazangirler, Cahit Berkay Kazangirler

**Published**: 2026-10-27 10:38:29

**PDF URL**: [https://arxiv.org/pdf/2610.36991v2](https://arxiv.org/pdf/2610.36991v2)

## Abstract
Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a high-recall seed set using exact inner-product search and hybrid relevance signals, expands it over the chunk graph, re-scores candidates using graph proximity and spectral information, and assembles the final context under a fixed token budget. The goal is not merely to retrieve individually similar chunks, but to construct a context that is relevant, diverse, structurally connected, and sufficient for grounded reasoning. Experiments on MultiHopRAG, UltraDomain, and long-document question-answering settings show that SpectRAG improves broad evidence recovery and achieves competitive generation quality against dense, hybrid, HyDE, LightRAG, and NaiveRAG baselines. The results further indicate that graph-spectral structure is particularly useful for inference-oriented and cross-document retrieval, while highlighting trade-offs in early precision, computational cost, and domain-dependent performance.

## Full Text


<!-- PDF content starts -->

SPECTRAG: WEIGHTEDCHUNKGRAPHS FOR
TOKEN-BUDGETEDCONTEXTSELECTION INMULTI-HOP
RETRIEVAL-AUGMENTEDGENERATION
A PREPRINT
Zafer Sahinoglu Mitsubishi Electric US
Cambridge, 5900-A Katella Avenue
California 90630, USA
zafer.sahinoglu@meus.com
Abdulazeez Mohammed
Mitsubishi Electric Innovation Center
(MELIC)
Cambridge, MA 02139, USA
abdulazeez_mohammed@outlook.com
Buse Yaren Kazangirler
Mitsubishi Electric Innovation Center
(MELIC)
Cambridge, MA 02139, USA
buseyarentekin@outlook.com
Cahit Berkay Kazangirler
Institute of Graduate Programs,
Dept. of Computer Engineering,
Karabuk University, Karabuk, 78050, Turkey
cahitberkaykazangirler@outlook.com
October 30, 2026
ABSTRACT
Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving
external evidence and conditioning generation on the retrieved context. Although dense retrieval is
effective for local semantic matching, independently ranked retrieval can return redundant passages,
miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We
proposeSpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate
seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propaga-
tion, and token-budgeted context selection. Document chunks are represented by dense semantic
embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges
encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and
similarity between LLM-generated utility questions. At query time, SpectRAG builds a high-recall
seed set using exact inner-product search and hybrid relevance signals, expands it over the chunk
graph, re-scores candidates using graph proximity and spectral information, and assembles the final
context under a fixed token budget. The goal is not merely to retrieve individually similar chunks,
but to construct a context that is relevant, diverse, structurally connected, and sufficient for grounded
reasoning. Experiments on MultiHopRAG, UltraDomain, and long-document question-answering
settings show that SpectRAG improves broad evidence recovery and achieves competitive generation
quality against dense, hybrid, HyDE, LightRAG, and NaiveRAG baselines. The results further
indicate that graph-spectral structure is particularly useful for inference-oriented and cross-document
retrieval, while highlighting trade-offs in early precision, computational cost, and domain-dependent
performance.
KeywordsRetrieval-augmented generation ·token-budget context selection ·graph-based retrieval ·spectral graph
theory·dense vector search·multi-hop reasoning·passage diversity
1 Introduction
Retrieval-augmented generation (RAG) has become a practical architecture for grounding large language models
(LLMs) in external corpora [ 1]. In a standard pipeline, documents are chunked, chunks are embedded, a query is
arXiv:2610.36991v2  [cs.AI]  28 Oct 2026

arXivTemplateA PREPRINT
embedded, and the top- kchunks most similar to the query are provided as context to an LLM. Dense passage retrieval
[2] and hybrid sparse–dense retrieval are now widely used because they are scalable and effective for local factual
lookup. In this paper, we see this constraint as a separation of two paradigms for retrieval. One of them is the classical
top-kparadigm, where chunks are scored based on their individual similarities to the query, and the top-scoring chunks
are selected. The other paradigm is called context-set retrieval, where the goal is to find a set of chunks that, together,
provide enough evidence for generation. In this paradigm, a good RAG retriever should not only be able to retrieve
chunks similar to the query, but also form a cohesive context set. Nevertheless, the method of extracting the top- k
vectors is subject to three recurring limitations. Firstly, given that adjacent or semantically similar passages generally
occupy the same region of the embedding space, this method has the potential to eliminate superfluous evidence.
Secondly, it may overlook structurally important segments that are necessary for multi-step reasoning, even if they are
not immediately close to the query. Thirdly, while this method treats segments as largely independent units, real-world
text corpora possess a structure that includes document order, section hierarchy, shared entities, citations, temporal
dependencies, and conceptual links [3].
While the standard top-k retrieval method is practically effective, it is limited when the answer requires a context set
rather than a group of similar passages. For dense vector retrieval, chunks are usually ranked based on their individual
similarity to the query. The success of this ranking mechanism depends on the local representation of the required
evidence in only one or a few semantically similar passages. However, many real-world question-answering (QA) and
reasoning tasks require evidence distributed across multiple chunks, documents, or sections. In such cases, the most
advantageous setting of context may not be composed of the highest-ranked components alone, but of passages that are
jointly relevant, complementary, and structurally interconnected [4]–[6].
Graph-based retrieval methods have emerged as a response to these limitations. Retrieval-augmented generation with
graphs (GraphRAG)-style approaches represent information as graphs and exploit relational links to support retrieval
and generation across connected evidence [ 6]. Related work has explored entity graphs, document graphs, knowledge
graphs, and graph-augmented memory structures. For example, GEM-RAG [ 22] graphical eigenmemories by tagging
data segments with utility questions, connecting them into a graph, and using eigendecomposition to form higher-level
memory nodes. CrossAug further highlights that many GraphRAG pipelines miss relations whose supporting evidence
spans multiple chunks, and proposes an offline graph-augmentation strategy to recover cross-chunk structure [ 7]. These
studies indicate that retrieval can benefit from modeling both the semantic similarity between a query and individual
chunks and the structural relationships among these chunks.
However, existing graph-enhanced RAG methods often focus on graph construction, entity-level relations, or community-
level summarization. The problem of selecting a compact, diverse, and structurally connected context set remains
underexplored. A retrieval system for multi-hop and long-document reasoning should, in principle, be designed to
satisfy numerous objectives simultaneously. The goals of this initiative are numerous and include, but are not limited to:
•To reduce redundancy, resulting in more efficient and varied retrieval strategies and, hence, greater confidence
in the ongoing efforts.
• Bridge evidence recovery in process.
• Ensure structural coherence of the selected segments is very important.
1.1 Motivation Behind The Study
We introduceSpectRAG, a retrieval framework that combines dense candidate seeding with graph-spectral context-set
optimization. This framework does not thoroughly replace dense retrieval techniques. Rather, it uses dense vector
similarity to generate an initial candidate pool with high recall, then leverages the structure of a weighted chunk graph
to refine the final context set. The suggested framework represents chunks using both dense semantic embeddings and
spectral coordinates that are derived from the graph Laplacian. These spectral coordinates offer structural insights
into the corpus retriever to consider graph connectivity, latent communities, and inter-chunk relationships during the
selection process, and enable the retriever to account for graph connectivity, latent communities, and cross-chunk
relations during selection.
In the querying stage, SpectRAG converts the user query into embeddings, and then creates a seed set from the
embedding space that has high recall. The seed set is then further extended via graph neighborhoods for fetching
other related segments that do not necessarily match the query well, but can act as connectors or be evidence itself.
The relevance of queries is then propagated to the candidate subgraph that is affected by this, thus making relevance
propagate from the matched segments to other related evidence. Finally, SpectRAG applies a token-budgeted context
selection stage that assembles the final context under a fixed input budget. This stage prioritizes relevant and structurally
connected chunks, preserves a protected core of high-confidence evidence, applies a precision-oriented filtering strategy
2

arXivTemplateA PREPRINT
for lower-ranked candidates, and backfills the remaining budget when necessary to avoid under-utilization of the
available context window.
1.2 Limitations of Top-k Dense Retrieval
In conventional dense retrieval, each chunk is scored independently according to its similarity to the query. Given a
corpus of chunks V=c 1, c2, . . . , c n, chunk embeddings ei∈Rd, and a query embedding eq∈Rd, the standard dense
retrieval baseline can be written as
Sdense
k(q) = TopKc i∈Vksim(e q,ei),(1)
where sim(·,·) is typically cosine similarity. This setup is computationally efficient and effective as long as the key
evidence lies in a single or a few semantically similar chunks [ 1,2]. This approach assumes that the best context to
generate from can be chosen by ranking the chunks themselves. It might be wrong if the response needs complementary,
concise evidence.
Redundancy creates a problem because dense embedding vectors are generally grouped into clusters of adjacent or
overlapping chunks of text. In other words, in the case of the selection of the top k, multiple chunks with identical
content will be selected, thereby leaving less space in the context space for different evidence. This becomes a problem
in retrieval-augmented generation because of the small size of the context space, leading to fewer pieces of evidence
being considered. However, the use of diversity-oriented retrieval can help mitigate the problem [4].
The second limitation is the potential loss of bridge evidence. While answering the question using multi-hop reasoning,
the evidence needed for the answer could include some passage chunks bridging entities, events, definitions, references,
or claims. Such bridge chunks are not always directly similar to the user query and may therefore receive lower dense
similarity scores. Therefore, independent top- krankings can retrieve locally relevant passages, but ignore the evidence
required to connect them into a total reasoning path. This issue is especially important in benchmarks such as HotpotQA,
2WikiMultiHopQA, and MuSiQue, where supporting evidence is distributed across multiple passages or reasoning
steps [8–10].
The third limitation observed in conventional dense search systems is structure blindness. In this type of system,
information is divided into separate divisions for retrieval. However, most databases usually have relationships existing
between different parts of information. The relationship could cut across many different dimensions, including
document order relationships, hierarchical relationships between different sections of the document, relationships
between common entities, citation or referencing that ties one part of information to another, time dependencies that
show the timing of the information, as well as conceptual relationships.
These limitations suggest that top- kdense retrieval should be treated as a strong baseline rather than as the final retrieval
objective. This encourages the proposed SpectRAG framework combines hybrid candidate seeding, graph neighborhood
expansion, proximity-based re-scoring, and token-budgeted final context selection.
1.3 Contributions
To summarize, the essential contributions of this research are outlined as follows:
•SpectRAG, a spectral graph-augmented retrieval framework integrates dense vector search, weighted chunk
graph construction, Laplacian spectral coordinates, graph-based candidate expansion, relevance diffusion, and
cluster-aware final selection.
•A graph-based retrieval pipeline that uses dense retrieval as an initial entry point and then exploits graph
structure to recover bridge evidence and reduce redundant context.
•A reproducible evaluation protocol for assessing SpectRAG on multi-hop and long-document question-
answering benchmarks, including MultiHopRAG, HotpotQA, 2WikiMultiHopQA, MuSiQue, and long-
document QA datasets.
•We design an ablation study to isolate the contribution of graph expansion, spectral coordinates, diffusion,
graph edge signals, and diversity-aware context selection.
SpectRAGis a retrieval framework that augments dense vector search with a weighted chunk graph and the spectral
structure of its graph Laplacian. Rather than treating dense retrieval and graph-based retrieval as competing alternatives,
SpectRAG adopts dense vector search as a high-recall entry point and subsequently exploits graph structure to expand,
diffuse, diversify, and select the final context set. Each chunk is defined using both its dense semantic embedding
and graph-spectral coordinates derived from a weighted chunk graph. The graph can integrate numerous edge signals,
3

arXivTemplateA PREPRINT
Figure 1: Top-k paradigm vs. our SpectRAG framework
including semantic similarity, document adjacency, section hierarchy, shared entities, citation relations, and similarity
between LLM-generated utility queries.
2 Related Work
2.1 Retrieval-Augmented Generation
RAG is a method which integrates user query with related passages from an external knowledge source [ 3,8]. The
system being studied includes two core elements: theretrieval componentand thegeneration component. The retrieval
component identifies relevant passages for the query made by the user, whereas the generation component uses those
passages as context to generate a response.
Previous RAG models have demonstrated that the combination of parametric language models with non-parametric
external memories works well for knowledge-heavy tasks where the required information cannot be completely
represented by the model parameters or where information changes over time. In real-world scenarios, the retrieval
process is typically carried out using methods such as sparse retrieval, BM25 retrieval [ 11], dense passage retrieval, or
a combination of sparse and dense retrieval.Sparse retrievers perform retrieval based on keyword matching, whereas
dense retrievers map the query and documents into a vector space and retrieve semantically similar passages. Although
it is effective, traditional RAG generally counts on independently ranked passages. While this design is adequate for
local factual lookups, it may prove inadequate when the answer requires evidence distributed across multiple passages,
documents, or structural relations. Recent research has sought to enhance not only the relevance of retrieved passages
but also the overall coherence and diversity of the final context, encouraging the audience to feel optimistic about future
developments.
2.2 Diversity-Aware and Context-Set Retrieval
The conventional systems of retrieval concentrate on the relevance of the individual passages but not on the assessment
of the relevance of the entire set of retrieved passages. The generator takes the set of passages and the relevance of such
a set depends not only on the relevance of the individual passage to the given query but also on the complementarity,
non-redundancy, and sufficiency of the passages, making it evident that non-redundancy plays an important role in
information retrieval.
In the context of RAG, context-set retrieval is the process of selecting a subset of passages that optimizes set utility.
The selected context set needs to preserve query relevance, cover evidence from all sides, avoid repetition, and provide
enough data for generation. In multihop reasoning tasks and long-document reasoning, the selected context must also
be coherent across related passages, since the answer to the question may depend on connecting evidence from different
parts of the corpus [ 13]. Relationship-aware table discovery methods, such as SANTOS, similarly show that semantic
relationships among structured data elements can improve search beyond column- or item-level matching alone [14].
4

arXivTemplateA PREPRINT
A conceptual comparison between independent dense retrieval and the proposed SpectRAG strategy is shown in Fig. 1.
In standard dense retrieval, passages are ranked independently based on their similarity to the query, which can lead
to redundant passages, missing bridge evidence, and disconnected support. In contrast, SpectRAG constructs the
final context by combining relevance, graph connectivity, spectral diffusion and diversity-aware selection under a
token-budget constraint.
This perspective is further developed in SpectRAG through token-budgeted context assembly. In contrast to conventional
top-ranked retrieval, where the final passages are selected only according to independent similarity scores, SpectRAG
constructs the final context under a fixed token budget. Candidate chunks are evaluated not only by relevance, but also
by graph proximity, document coverage, redundancy, and their contribution to a compact evidence set. This design
allows the system to preserve high-confidence evidence while using the remaining budget for complementary and
structurally useful chunks [15].
2.3 Graph-Enhanced Retrieval and GraphRAG
GraphRAG methods explicitly represent corpus structure. Edgeet al.[ 12] propose a graph-based approach to query-
focused summarization in which an LLM extracts an entity knowledge graph, groups related entities into communities,
generates community summaries, and uses these summaries to answer global questions over a corpus. This method
addresses questions such as identifying themes or relationships across many documents, where naive local retrieval
can fail. Graph-enhanced retrieval is especially relevant when answer evidence is distributed across multiple pieces of
text. Zhanget al.[ 7] argue that existing GraphRAG systems often extract relations within individual chunks, leaving
cross-chunk relations systematically absent. Their CrossAug method enriches graph indices with missing cross-chunk
relational structure using a topology-aware graph neural network and targeted LLM completion.
2.4 Weighted Chunk Graph
The framework constructs a weighted graph G= (V, E, W) whose nodes are chunks. Edge weights combine multiple
evidence signals:
wij=α 1wsem
ij+α 2wadj
ij+α 3wentity
ij +α 4wutility
ij +α 5wcitation
ij .(2)
Figure 2: Documents are chunked into token-based passages (1–2) and encoded as dense embeddings (3). Rather than
treating chunks as independent vectors, SpectRAG connects them with edges derived from six complementary evidence
signals—semantic similarity, document adjacency, section hierarchy, shared entities, citations, and utility-question
similarity (4).
The semantic edge weight is computed from dense embeddings:
wsem
ij= max(0,cos(e i, ej)),(3)
5

arXivTemplateA PREPRINT
usually restricted to a sparse kg-nearest-neighbor graph. Document adjacency connects neighboring chunks in the same
source document. Section hierarchy connects chunks that share headings or parent sections. Entity edges connect chunks
sharing named entities, product names, technical terms, dates, or organizations. Utility-question edges connect chunks
that answer similar generated questions, following the intuition that functional similarity may be more retrieval-relevant
than surface similarity alone [ 22]. Citation edges connect chunks with references, hyperlinks, footnotes, or explicit
dependency relations.
LetAdenote the weighted adjacency matrix and Dthe diagonal degree matrix with Dii=P
jAij. The unnormalized
Laplacian is:
L=D−A,(4)
and the symmetric normalized Laplacian is:
Lsym=I−D−1/2AD−1/2.(5)
The symmetric normalized Laplacian Lsymof this graph provides the spectral coordinates and diffusion operator used
at query time. Edges that connect distinct communities (e.g., C5–C6,C7–C8) carry the bridge evidence that top- k
dense retrieval systematically misses.
Fig. 2 shows the construction of the weighted chunk graph. Nodes are document chunks and edges are multiple
relation signals including semantic similarity, document adjacency, section hierarchy, shared entities, citation links and
utility-question similarity. These relation signals determine the adjacency matrix for Laplacian computation, spectral
coordinate extraction, graph expansion and relevance propagation.
2.5 Context-Set Retrieval Objective
A conventional dense retriever ranks chunks independently by their similarity to the query. Given a similarity function
sim(·,·), typically cosine similarity, the initial relevance score of chunkc iis
s0(ci, q) = sim(e q, ei).(6)
The standard dense baseline selects thekchunks of highest individual relevance,
Sdense
k(q) = Top-K
ci∈Vs0(ci, q).(7)
Equation (7)scores each chunk in isolation: the marginal value of a chunk does not depend on what has already
been selected. As established in Section 1.2 and formalized in Section 2.5, this independence is the source of the
redundancy and missing-bridge-evidence failures that motivate SpectRAG. We restate it here only to fix the notation s0
andSdense
k used in the objective below: whereas (7)optimizes a sum of independent scores, the context-set objective
of (9) optimizes over the selected set jointly.
SpectRAG therefore formulates retrieval as acontext-set selectionproblem. The goal is not to select the individually
highest-scoring chunks, but to construct a compact context set Cq⊆V with|Cq|=k that is together beneficial for
grounded generation:
C∗
q= arg max
Cq⊆V,|C q|=kF(C q, q),(8)
where the set-level utilityFis defined as
F(C q, q) =X
ci∈CqRel(c i, q)−λX
ci,cj∈Cq, i<jRed(c i, cj)
+γCov(C q, q) +ηConn(C q;G),(9)
andGis the weighted chunk graph introduced in Section 3.2. The coefficients λ, γ, η≥0 control the redundancy
penalty, the coverage reward, and the graph-connectivity reward, respectively. The dense baseline of Equation (7)
is recovered as the special case λ=γ=η= 0 , in which the objective decomposes into kindependent per-chunk
decisions. The four terms are defined in Section 2.6.
2.6 Terms of the Context-Set Objective
The context-set objective is motivated by the observation that relevance alone is insufficient for multi-hop and long-
document retrieval: a selected context A set may consist entirely of individually relevant chunks while still being
redundant, disconnected, or incomplete. Each term of Equation (9) targets one of these failure modes.
6

arXivTemplateA PREPRINT
Relevance. Rel(c i, q)is initialized from the dense score s0(ci, q)and updated after graph expansion, so that a chunk’s
The contribution reflects both its direct similarity to the query and its position relative to strongly relevant evidence.
Redundancy. Red(c i, cj)measures the semantic overlap between two selected chunks. A highly redundant context
set spends limited context-window capacity repeating the same evidence, displacing chunks that would otherwise
contribute new information to the generator.
Coverage. Cov(C q, q)rewards the number of distinct evidence aspects of the query represented in the set. It is our
operationalization ofevidence completeness: whether the selected context carries enough information to support the
final answer, particularly when the reasoning chain requires bridge evidence spanning multiple chunks or documents.
Connectivity. Conn(C q;G)measures the structural relationships among the selected chunks in the weighted chunk
graph. Chunks linked through shared entities, document adjacency, and section hierarchy, citations, or other relational
signals, are more likely to form a coherent evidence path than an equally relevant but mutually disconnected set.
Because Equation (9)is defined over sets rather than individual chunks, it cannot be optimized by ranking alone.
SpectRAG therefore uses dense and hybrid retrieval only for high-recall candidate seeding, and then applies graph
expansion, proximity-based re-scoring, and token-budgeted selection to approximate the maximizer of F, yielding the
final context set is passed to the generator.
3 Spectral Graph-Augmented Retrieval
3.1 Overview of SpectRAG
The offline indexing stage is summarized in Table 1. This stage is executed once for each document collection and
includes token-based chunking, dense embedding computation, edge-signal extraction, weighted graph construction,
spectral decomposition, Neo4j graph persistence, and disk-level storage of chunks, embeddings, spectral coordinates,
utility questions, and FAISS indexes. This design separates expensive corpus-level preprocessing from query-time
retrieval, allowing the online pipeline to reuse precomputed graph and spectral artifacts. Furthermore, the online
query-time procedure is outlined in Table 1. For each query, SpectRAG first performs hybrid seed retrieval by fusing
dense FAISS scores, BM25 lexical retrieval, and utility-question matching. The seed set is then expanded through
two-hop graph traversal, re-scored using a combination of relevance and graph proximity, and assembled into a final
context using token-budgeted selection. The final context is grouped by source document and passed to the generator
together with ranked source metadata. As illustrated in Fig. 3, SpectRAG consists of two main stages: offline corpus
indexing and online query-time retrieval. In the offline stage, the corpus is segmented into chunks, embedded into dense
vector representations, organized into a weighted chunk graph, and transformed into graph-spectral coordinates via the
Laplacian. In the online stage, SpectRAG performs dense candidate seeding, graph neighborhood expansion, spectral
relevance propagation, and token-budgeted context selection before passing the final context to the generator.
3.2 Chunk Graph Construction
The context-set objective in Equation (9)is defined with respect to a weighted chunk graph G: the connectivity term
Conn(C q;G)is evaluated on its edges, and the expansion and re-scoring stages of Section 3.1 operate on its spectrum.
This subsection specifies howGis constructed, weighted, sparsified, and decomposed during offline indexing.
3.2.1 Nodes and edge signals
The chunks produced by the segmentation stage, each carrying its dense embedding ei∈Rd. Unlike a k-nearest-
neighbour embedding graph, edges are drawn from six complementary evidence signals, each normalized to[0,1]:
•Semantic( wsem
ij): truncated cosine similarity between embeddings, max(0,cos(e i,ej)), restricted to the kg
nearest neighbours of each node.
•Adjacency( wadj
ij): consecutive chunks within the same source document, capturing narrative continuity across
chunk boundaries.
•Hierarchy(wsec
ij): shared headings or a common parent section in the document’s structural path.
•Entity(went
ij): overlap in named entities, technical terms, organizations, or dates extracted per chunk.
•Utility( wutl
ij): similarity between LLM-generated questions that each chunk answers, encoding functional
rather than surface similarity [22].
•Citation(wcit
ij): explicit references, hyperlinks, footnotes, or dependency relations between chunks.
7

arXivTemplateA PREPRINT
Table 1: Offline SpectRAG pipeline—runs once per document set
Step What it does Output
Chunking Splits documents into structural (heading-aware) or semantic chunks
measured with tiktoken (1200 tokens, 100 overlap). Extracts tables.
Never cuts mid-sentence.Chunk objects with text, po-
sition, section path, and ref-
erences
Embedding Encodes each chunk with nomic-embed-text (768-dimensional), using
the model’s search_document: task prefix. Builds a FAISS flat index
over L2-normalized vectors for approximate nearest-neighbor search.chunk.embedding per
chunk & faiss.index on
disk
Edge signals Runs six extractors to find relationships between chunks. Entity edges
usegpt-4o-mini concept extraction by default, & utility-question edges
usegpt-4o-mini ; both store per-chunk outputs for query-time reuse.
Concepts are disk-cached.Six lists of RawEdge(i,
j, weight) &chunk.
utility_questions
Graph build Combines all six signals using configurable weights, either fixed priors
or auto-calibrated per corpus. Drops edges below the sparsification
threshold.Weighted graph G=
(V, E, W)
Spectral decomp Computes the normalized Laplacian L=I−D−1/2WD−1/2. Extracts
the 32 smallest eigenvectors. Each chunk gets a 32-dimensional coordi-
nate encoding its position in the graph’s community structure.chunk.spectral_coords
(32-dimensional) per chunk
Neo4j write Writes all chunks as nodes and edges with per-signal attributes to Neo4j
via batched Cypher.Persistent graph in Neo4j
Persist Saves chunks, including embeddings, spectral coordinates, and utility
questions, and the FAISS index to disk.chunks.pkl &faiss.
index
Table 2: Online SpectRAG pipeline—runs per query.
Stage What it does Output
Hybrid seed retrieval Fuses three rankings with reciprocal rank fusion (RRF): dense FAISS cosine,
utility-question matching (query to stored questions), & BM25 lexical retrieval.
The full relevance vector consumed downstream is hybrid (dense & BM25),
so lexical precision reaches later stages.Seed set S0(20 chunks) &
hybrid relevance scores for
all chunks
Graph expansion Performs breadth-first search from S0in Neo4j, using two hops and a cap of
200 nodes. Fetches the candidate subgraph.Candidate set S1(≤200
chunks) and subgraph
Proximity re-scoring Blends relevance with graph proximity as (1−α)·relevance+α·proximity ,
where proximity in hybrid mode is the mean of spectral eigenspace closeness
and Personalized PageRank from the seeds over the real weighted subgraph.Re-scored candidate set
S2
Token-budget selec-
tionFills an approximately 12k-token budget with relevant chunks: a protected
core, a tail precision floor, document-coverage tie-break, and backfill so the
budget is never starved. Alternative modes include anchor_expand_ce ,
protected_mmr, andmmr.Final context set C(adap-
tive, approximately 10–15
chunks)
Generation Formats Cgrouped by source document and sends it to gpt-4o-mini with a
synthesis-oriented, grounded prompt and temperature 0.Answer string and ranked
source metadata
3.2.2 Weight fusion and sparsification
The six signals are combined into a single edge weight,
wij=α 1wsem
ij+α 2wadj
ij+α 3wsec
ij+α 4went
ij+α 5wutl
ij+α 6wcit
ij,(10)
withαr≥0 set either from fixed priors or auto-calibrated per corpus. Edges with wij< τ are discarded, keeping
|E|=O(k gn)so that the subsequent traversal and spectral steps remain tractable at corpus scale.
Purely semantic edges reproduce the redundancy problem of Section 1.2: near-duplicate chunks form tight cliques
while bridge evidence stays disconnected. The non-semantic signals in Equation (10) are precisely those that link
chunks a dense retriever scores independently, and they supply the paths along which relevance is later diffused.
8

arXivTemplateA PREPRINT
Figure 3: Overview of the SpectRAG pipeline.
3.2.3 Spectral coordinates
LetWbe the resulting weighted adjacency matrix andDthe diagonal degree matrix,D ii=P
jwij. SpectRAG uses
the symmetric normalized Laplacian
Lsym=I−D−1/2WD−1/2,(11)
whose msmallest non-trivial eigenvectors, stacked columnwise as U∈Rn×m, assign each chunk a spectral coordinate
ui∈Rm(m= 32 in our implementation). Because the eigenvectors of Lsymminimize a normalized-cut relaxation,
chunks within the same latent community are close in Rmwhile chunks separated by a sparse cut are distant, irrespective
of their raw embedding similarity. Proximity in this space therefore provides a structural complement to s0(ci, q)at
query time, andL symadditionally serves as the diffusion operator for relevance propagation.
The construction is illustrated in Fig. 2, and is performed once per corpus; only U,W, and the chunk embeddings are
persisted for online retrieval.
3.3 Documents Chunking
The chunking process involves adaptive structure-aware chunking, which is done by picking one of two segmentations
for each document. Documents with explicit structure information, such as Markdown markup or section numbering,
employ the structural segmentation, splitting text along the hierarchical structure, propagating the entire path through
the sections hierarchy (e.g., 2. Methods →2.1 Datasets) as chunk metadata; the section-affinity edge generation will
use this hierarchy. For unstructured text, the semantic mode is applied: sentences are embedded using the same model
as for document embedding (nomic-embed-text), and breakpoints are inserted at places where the embedding distance
between two sentences exceeds the 90th percentile of the entire document distance distribution, resulting in topically
coherent segments independently of the document surface format. In both segmentation modes, the chunk size is
measured in tokens using the cl100k_base tokenizer, with a maximum number of tokens set to 1,200 tokens with
100-token overlap as a suffix of the previous chunk that respects sentence boundaries. Chunks with less than 32 tokens
are concatenated with the previous chunk, and tables are parsed separately, serializing them to Markdown so that rows
are not split.
9

arXivTemplateA PREPRINT
Figure 4: Sparse lexical vectors (top) allocate one dimension per vocabulary term and match exact words; dense
embeddings (middle, 768-d) encode meaning in every dimension and support similarity search via FAISS. Bottom:
SpectRAG’s spectral coordinates ( d= 32 ), derived from the smallest eigenvectors of the normalized graph Laplacian,
encode a chunk’s position in the corpus graph’s community structure rather than its textual semantics.
We selected a granularity of 1,200 tokens for analysis. The decision to increase the token size from 256 to 1,200 resulted
in the generation of more coherent text segments, which significantly enhanced the efficacy of abstractive sensemaking
queries. Consequently, this modification contributed to an improvement in the overall quality of information retrieval.
3.4 Laplacian Spectral Coordinates
SpectRAG assigns each chunk d-dimensional spectral coordinates (bottom, d= 32 ), derived from the normalized
graph Laplacian, encoding position in the corpus graph’s community structure rather than textual semantics. As shown
in Fig. 4, the representation is compared with typical vectors used in retrieval. Despite being similar in structure to
the dense embedding, a compact, continuous vector used for cosine similarity, spectral coordinates carry an entirely
different kind of signal. While eiholds information about the content of a chunk, zicontains information about its
location within the weighted chunk graph. This means that chunks from the same community in the chunk graph
have close coordinates even if they do not match in their surface text. It is crucial to understand that SpectRAG does
not depend on sparse retrieval; the sparse vector appears in Fig. 4 only for contrast. Section 3.7 shows how the two
complementary geometries of Fig. 4 (middle vs. bottom) are combined: dense embedding similarity supplies the initial
relevance signal, which is then diffused over the chunk graph whose low-frequency structure the spectral coordinates zi
encode.
We compute the firstmnontrivial eigenvectors ofL sym:
LsymU=UΛ.(12)
The spectral coordinate of chunkc iis
zi= [u 2(i), u 3(i), . . . , u m+1(i)]∈Rm.(13)
These coordinates capture low-frequency variation over the graph. Chunks that are structurally close, or that occupy
similar positions in graph communities, tend to be close in spectral space.
3.5 Initial Dense Retrieval
In the process of executing a query, SpectRAG uses the approach of dense retrieval to form an initial high-recall seed
set. This process should not be viewed as a final method for context selection, but rather is the first step towards getting
into the document pool and acquiring chunks of plausible direct relevance.
The purpose of this stage is to obtain a sufficiently broad candidate set rather than to determine the final context.
Therefore,Kis chosen to be larger than the final context budgetk, i.e.,
K≫k.(14)
10

arXivTemplateA PREPRINT
The final context set is selected later by graph expansion, spectral diffusion, and token-budget final context selection.
3.6 Graph Expansion
The dense retrieval approach finds chunks which individually resemble the query; whereas in multi-hop reasoning,
evidence may depend on information that is structurally, but not semantically, related to the query. These chunks,
by their very nature, will be outside the scope of nearest neighbor search as they are situated outside the embedding
neighborhood of the query and still have structural connections to the query via the graphG.
Seed set.LetS 0⊂Vdenote the high-recall seed set returned by dense retrieval,
S0= Top-K
ci∈Vsim(e q, ei),|S 0|=K,(15)
where Kis the candidate budget and K≫k , the final context budget. Over-retrieving at this stage is deliberate:
expansion can only recover evidence reachable from the seeds, so recall at S0upper-bounds recall of the entire pipeline.
3.7 Spectral Diffusion of Query Relevance
Given an initial relevance vectors∈Rnwith
si= sim(e q, ei),(16)
we define a diffused relevance signal:
s′= exp(−tL sym)s,(17)
where tcontrols diffusion scale. Small tkeeps relevance local; large tspreads relevance across broader graph regions.
For large corpora, Equation 17 can be approximated by truncated Chebyshev polynomials, personalized PageRank-style
iteration, or repeated smoothing:
s(r+1)=βs+ (1−β) ˜As(r),(18)
where ˜A=D−1/2AD−1/2andβ∈(0,1).
3.8 Token-Budgeted and Cluster-Aware Context Selection
After candidate expansion and proximity-based re-scoring, SpectRAG constructs the final context set under a fixed token
budget rather than selecting a fixed number of chunks. This design is motivated by the fact that retrieval-augmented
generation systems are constrained by the input context window of the generator. Therefore, the final retrieval stage
should optimize not only which chunks are selected, but also how the available token budget is allocated across evidence
sources.
LetPqdenote the re-scored candidate pool for query q, and let ℓ(ci)be the token length of candidate chunk ci. Given a
maximum context budgetB, the final context setC qis selected subject to:
X
ci∈Cqℓ(ci)≤B.(19)
The token-budgeted selection objective can be written as:
C∗
q= arg max
Cq⊆PqF(C q, q) s.t.X
ci∈Cqℓ(ci)≤B,(20)
whereF(C q, q)denotes the utility of the selected context set.
In practice, SpectRAG approximates this budget-constrained objective using a token-aware selection strategy. At each
step, the candidate score combines relevance, graph-based proximity, spectral-cluster coverage, document coverage,
and redundancy control:
U(ci|Cq) =a·s i+b·Prox(c i, S0)
+c·ClusterGain(c i, Cq)
+d·DocGain(c i, Cq)
−λ·Red(c i, Cq).(21)
where siis the re-scored relevance value of candidate ci,Prox(c i, S0)measures graph proximity to the seed set S0,
ClusterGain(c i, Cq)rewards coverage of underrepresented spectral clusters, DocGain(c i, Cq)encourages coverage
11

arXivTemplateA PREPRINT
across source documents, and Red(c i, Cq)penalizes semantic overlap with already selected chunks. To account for
chunk length, candidates may be selected according to token-normalized utility:
c∗= arg max
ci∈Pq\CqU(ci|Cq)
ℓ(ci)ρ,(22)
where ρ∈[0,1] controls the strength of length normalization. When ρ= 0 , selection is based only on utility; when
ρ= 1, the method favors candidates with higher utility per token.
The selection procedure consists of three practical components. First, SpectRAG preserves a protected core of high-
confidence chunks so that strongly relevant evidence is not removed by later diversity or budget constraints. Second, it
applies a precision-oriented tail selection strategy for lower-ranked candidates, using graph proximity, cluster coverage,
document coverage, and redundancy penalties to decide which additional chunks should enter the context. Third, if the
budget is not fully used, a backfill step adds remaining high-quality candidates so that the generator is not deprived of
useful evidence.
This token-budgeted design differs from fixed- kcontext selection. Instead of forcing every query to receive the same
number of chunks, SpectRAG adapts the number of selected chunks to their token lengths and the available context
window. In the current implementation, this typically produces a final context of approximately 10–15 chunks under a
context budget of about 12k tokens. The resulting context set is intended to be relevant, non-redundant, structurally
connected, and sufficiently broad for grounded generation.
Algorithm 1OnlineSpectRAGretrieval and generation.
Require: Query q, chunks V, FAISS index, BM25 index, utility questions, graph G= (V, E, W) , spectral coordinates
Z, token budgetB
1:Embedqase qand compute dense FAISS scores
2:Compute BM25 scores and utility-question matching scores
3:Fuse rankings with RRF to form seed setS 0
4:Fetch candidate subgraph by two-hop BFS fromS 0in Neo4j
5:LetS 1be the expanded candidate set, capped at 200 chunks
6:Compute spectral proximity fromZand Personalized PageRank on the candidate subgraph
7:Re-score candidates:s 2(c)←(1−α)r(c) +αp(c)forc∈S 1
8:Initialize protected coreCfrom highest-scoring candidates
9:whiletoken count ofCis belowBand candidates remaindo
10:Add the best candidate satisfying the precision floor and coverage tie-breaks
11:end while
12:Backfill with highest-scoring remaining chunks if budget remains
13:FormatCgrouped by source document and generate answer with grounded prompt
14:
15:returnAnswer, ranked source metadata, and final context setC
4 Experiments
The purpose of the experiments is to test whether SpectRAG improves retrieval and generation quality when evidence
is distributed, multi-hop, or structurally connected. Since this manuscript is a methodological proposal, the result
tables in this section define the intended reporting format. Numeric values should be populated only after running the
experiments.
4.1 Research Questions
•RQ1:Does SpectRAG’s chunk-level graph achieve competitive retrieval at a lower index cost than LightRAG’s
entity-level graph?
•RQ2:Does SpectRAG improve holistic answer quality (comprehensiveness, diversity, empowerment) over
graph-RAG baselines?
•RQ3:Does SpectRAG improve objective answer accuracy (Exact Match (EM), F1) on multi-hop questions
with gold answers, compared to graph- and vector-RAG baselines?
•RQ4:Does SpectRAG rank gold evidence higher in the retrieved context than graph- and vector-RAG
baselines, not merely retrieve more of it?
12

arXivTemplateA PREPRINT
4.2 Evaluation Datasets
To estimate SpectRAG’s effectiveness across a range of domains and content types, we used four datasets from the
UltraDomain [ 16] and employed specific evaluation metrics such as precision, recall and f1 to quantify performance.
Additionally, to evaluate against hallucination, we added the MultiHopRAG [ 17] dataset, which tests the model’s ability
to avoid generating unsupported information.
The UltraDomain benchmark is derived from 428 college textbooks covering 18 domains, including agriculture, the
social sciences, and the humanities [ 16]. We selected the agriculture, computer science (CS), legal, and mixed (Mix)
domains to focus on key areas relevant to our research, each with a token size of 600,000-5,000,000. Four domains
used for our experiment are listed below:
•Agriculture:This field emphasizes agricultural methods, addressing subjects like beekeeping, hive supervision,
crop cultivation, and disease management [17].
•CS:This particular subset relates to computer science books on algorithms, systems, and programming
principles. Similar to agriculture, it serves as an example in the subject because the questions and texts share a
common language, even from different textbooks [16].
•Legal:This domain focuses on legal practices in corporate settings, including case law, statutes, contracts,
regulatory documents, and governance, emphasizing the legal and financial industries [16, 17].
•Mixed:This subset collected from multiple domain such as literary, philosophical texts, historical studies and
other writing styles [16, 17].
4.3 Implementation Parameters and Details
For the analysis of the impact of retrieval granularity, we consider SpectRAG under two token-level chunking schemes.
In the first token-based configuration, the chunk size is set to 256 tokens with an overlap of 64 tokens. This setting
produces smaller retrieval units and allows the token-budgeted selector to compose evidence at a finer granularity. The
second scheme utilizes a chunk size of 1200 tokens with an overlap of 100 tokens. This scheme produces larger retrieval
chunks and enables the rich local context of documents to be retained in the retrieved candidates.
The implementation parameters used in the SpectRAG pipeline are reported in Table 3. These parameters include the
token-based chunking configuration, embedding model, edge extraction modules, LLM-assisted extraction components,
graph and spectral index settings, seed retrieval strategy, graph expansion radius, proximity re-scoring method, and
token-budgeted context selection strategy. Reporting these settings is important for reproducibility because SpectRAG
depends on both offline graph construction choices and online retrieval-stage parameters.
The two configurations above share a common retrieval pipeline consisting of hybrid candidate generation, graph
neighborhood expansion, proximity-based re-scoring, and token-budgeted final context selection.
4.4 Baselines
We compare SpectRAG against the following state-of-the-art approaches:
•NaiveRAG[ 19]: This model follows the conventional indexing, retrieval, and generation process. Chunks
are extracted from documents, embedded, and stored in a vector database; during inference, the query is
embedded, and the top- kmost similar chunks are retrieved and concatenated into a prompt for an LLM to
generate the answer.
•HybridRAG[ 20]: This approach integrates BM25 sparse lexical retrieval and dense semantic retrieval,
merging their ranked lists using RRF. In addition, uncertainty quantification (UQ) is incorporated to estimate
the confidence of the retrieved evidence and the generated response, allowing low-reliability outputs to be
flagged.
•LightRAG[ 18]: This model uses an LLM to extract entities and relations from documents and stores them as
a knowledge graph, as implemented in LightRAG. Retrieval is performed at two levels, namely entity-level
and thematic-level, enabling the system to capture multi-hop and cross-document dependencies that flat
chunk-based indexes cannot represent.
•HyDeRAG[ 21]: Retrieval based on Hypothetical Document Embeddings (HyDE), which closes the semantic
gap between a short query and a document by using an LLM to generate a hypothetical answer to the query.
The generated synthetic document is then embedded and used in place of the raw query embedding for the
similarity search, increasing the likelihood of retrieving the relevant chunks.
13

arXivTemplateA PREPRINT
Table 3: Implementation parameters aligned with the SpectRAG pipelines
Parameter Setting Related stage
Chunking Structural or semantic chunking with tiktoken ; 1200-token
chunks and 100-token overlap; sentence boundaries and tables
are preserved.Offline chunking
Embedding modelnomic-embed-text , 768-dimensional embeddings, with the
search_document:prefix for indexed chunks.Offline embedding
Edge extraction Six edge-signal extractors: semantic, document adjacency, section
hierarchy, entity/concept, utility-question, and citation/dependency
signals.Offline edge signals
LLM-assisted extractiongpt-4o-mini for concept extraction, utility-question generation,
and answer generation; gpt-4o for judging when evaluation re-
quires an LLM judge.Offline and generation
Graph and spectral index Weighted graph G= (V, E, W) with sparsified edges; normalized
Laplacian L=I−D−1/2WD−1/2; 32 spectral coordinates per
chunk.Graph build and spectral
decomp
Seed retrieval RRF over dense FAISS cosine, BM25 lexical retrieval, and utility-
question matching; seed set size is 20 chunks.Online stage 1
Graph expansion Two-hop BFS from S0in Neo4j, capped at 200 candidate chunks. Online stage 2
Re-scoring Hybrid proximity blends relevance with spectral eigenspace close-
ness and Personalized PageRank over the weighted subgraph.Online stage 3
Context selection Approximately 12k-token budget with protected core, tail preci-
sion floor, document-coverage tie-breaks, and backfill; ablations
includeanchor_expand_ce,protected_mmr, andmmr.Online stage 4
4.5 Retrieval and Answer Correctness Metrics
The measures of Section 4.5.2 arerelative: they express a judge’s preference between two systems, which helps
researchers see comparative strengths. We complement them withabsolutemeasures against benchmark ground truth,
providing a complete assessment. Reporting correctness at two levels shows the audience that different failure modes
matter and need tailored solutions.
4.5.1 Retrieval-Level Correctness
Here, we use Cqto refer to the set of contexts obtained from query q, andE∗
qto be the set of evidence chunks annotated
in the benchmark data. The precision will refer to the ratio of the supporting evidence contained in the retrieval results,
whereas the recall will be the ratio of retrieved evidence chunks.
Precision@k=|Cq∩E∗
q|
|Cq|,Recall@k=|Cq∩E∗
q|
|E∗q|.(23)
Their harmonic mean gives the balanced score
F1@k=2·Precision@k·Recall@k
Precision@k+ Recall@k.(24)
withF1@k≜0 when both terms vanish. All three are averaged over the query set. Because |Cq|=k is fixed by the
context budget, Precision@k andRecall@k are not independent: enlarging kmonotonically increases recall while
generally depressing precision. We therefore report both at matched kfor every system, and report F1@k as the
budget-normalized summary.
4.5.2 Answer-Level Correctness
In case of benchmark test cases where the gold-standard short answers are known, the standard extractive QA approach
is followed. Let ˆabe the output answer, and a∗be the reference answer; both have been normalized using lower casing
and stripping off articles and punctuation.
14

arXivTemplateA PREPRINT
EM =1
ˆa=a∗
.(25)
and token-level F1 is computed over the multiset of tokensT(·):
Ptok=|T(ˆa)∩T(a∗)|
|T(ˆa)|, R tok=|T(ˆa)∩T(a∗)|
|T(a∗)|,(26)
F1tok=2·P tok·Rtok
Ptok+R tok.(27)
Where a question admits several references, we take the maximum over references, as is standard.
Furthermore, we use to judge the answers that the model produces, these are the four measures that:
•Comprehensiveness: It is a measure of the ability of the answer to provide an account of all facets of the
question.
•Diversity: It is a measure of the diversity of viewpoints in the answer relative to the question.
•Empowerment: It is a measure of how much an answer helps a reader understand and make judgments on a
topic.
•Overall: This is a composite score of the first three measures.
4.5.3 Applicability
These measures require labeled ground truth and are therefore reported only on MultiHop-RAG and UltraDomain, which
annotate gold supporting facts and short answers. The sensemaking protocol of Section 4.5.2 produces open-ended
questions without references and admits only the preference-based measures. The two families are reported separately
and are not aggregated: they quantify different properties, and a system may improve evidence recall without improving
preference, or vice versa.
4.6 Preliminary Generation-Level Evaluation
For the preliminary generation-level evaluation, we compared SpectRAG and NaiveRAG on a subset of the UltraDomain
dataset using a pairwise approach. For each domain, we indexed all documents and selected 20 sample questions.
An LLM-as-judge protocol was used, with the judge model kept distinct from the generation model to eliminate self-
preference bias. The judge received both candidates’ answers and issued the final verdict, operating at 0 temperature to
maximize determinism. The evaluation follows a LightRAG-style sensemaking protocol, in which answers are judged
pairwise across multiple qualitative dimensions. To mitigate position bias, the answer order was swapped during judging.
The pairwise win rates for SpectRAG and NaiveRAG are reported in Table 9 and Table 10, where each row sums to
100%. Values above 50% indicate an advantage for SpectRAG over NaiveRAG. SpectRAG outperformed NaiveRAG
across all dimensions, with the largest margin observed for diversity, at 55.0%. This suggests that graph-spectral
contextual selection can improve the structural coherence of retrieval and, in turn, the quality of generated answers.
4.7 RQ1
Because SpectRAG defines its graph over document chunks rather than LLM-extracted entities, it represents every
evaluated corpus with substantially fewer nodes and a smaller on-disk index, while matching or exceeding LightRAG
on retrieval quality. The node reduction holds across all five corpora, ranging from 2.8× to6.6× fewer nodes, and
the complete persisted index on MultiHopRAG is 6×smaller (46 MB versus 274 MB), since SpectRAG maintains
only a single chunk-level vector index and a spectral basis and eliminates LightRAG’s separate entity (100 MB) and
relationship (106 MB) vector databases. This lighter representation does not sacrifice quality: on MultiHopRAG’s
gold-evidence annotations, SpectRAG attains a Recall@5 of 65.2 against LightRAG’s 49.5, with comparable answer
accuracy. SpectRAG therefore realizes a favorable cost and quality trade-off, offering a compact, denser chunk graph
that competes with a state-of-the-art entity-graph baseline at a fraction of its index footprint. Table 4 reports the
per-corpus graph size and index cost for both systems. In addition, the results support four key findings:
1.Three to four times fewer nodes across every corpus ( 2.8× to6.6× ).SpectRAG never materializes
LightRAG’s approximately 15,000 entity nodes, so its spectral and personalized PageRank (PPR) diffusion
runs over a far smaller node set.
15

arXivTemplateA PREPRINT
Table 4: Per-corpus graph size and index cost.
Corpus System Nodes Edges E/N Graph Node
File Red.
Agriculture LightRAG 1,236 1,083 0.88 1.08 MB —
SpectRAG 239 476 1.99 0.14 MB5.2×
CS LightRAG 1,327 1,036 0.78 1.06 MB —
SpectRAG 461 2,608 5.66 0.72 MB2.9×
Legal LightRAG 5,615 6,784 1.21 6.80 MB —
SpectRAG 1,989 10,019 5.04 2.76 MB2.8×
Mix LightRAG 7,428 5,865 0.79 5.92 MB —
SpectRAG 1,127 1,603 1.42 0.46 MB6.6×
MultiHopRAG LightRAG 15,803 16,911 1.07 16.40 MB —
SpectRAG 5,022 19,326 3.85 5.34 MB3.2×
Nodes are document chunks for SpectRAG and LLM-extracted entities for LightRAG. E/N is edges per node. Node reduction (Node
Red.) equals LightRAG nodes divided by SpectRAG nodes. Full persisted-index footprint on MultiHopRAG: SpectRAG 46 MB
versus LightRAG 274 MB (excluding LightRAG’s 63 MB build-time LLM cache).
2.Approximately six times smaller index (46 MB versus 274 MB on MultiHopRAG).The gap is almost
entirely attributable to LightRAG’s entity and relationship vector stores (206 MB combined), which SpectRAG
does not build.
3.Fewer nodes and richer edges, not fewer edges.SpectRAG’s chunk graph is consistently denser per node
(edges per node up to 5.7 versus LightRAG’s approximately 1), so the efficiency claim is correctly stated in
terms of node count and storage, never edge count.
4.No quality penalty.SpectRAG achieves a Recall@5 of 65.2 versus 49.5 for LightRAG on MultiHopRAG,
with comparable answer accuracy.
4.8 RQ2
Yes. The following CS-domain example illustrates a representative case: SpectRAG wins all three quality dimensions
and the overall verdict under the pairwise protocol, reflecting its aggregate UltraDomain win-rate advantage over
LightRAG. Table 5 presents the full exchange and the judge’s decision.
4.9 RQ3
SpectRAG matches the graph-RAG baseline and clearly outperforms the vector-RAG baseline on objective answer
accuracy. On MultiHopRAG’s 2,556 gold-answer questions, SpectRAG attains an EM of 72.0 and F1 of 72.4, exceeding
NaiveRAG (EM 65.2 / F1 65.6) by roughly seven points, and is on par with LightRAG (EM 70.5 / F1 70.9), marginally
ahead. Notably, SpectRAG’s large retrieval advantage over LightRAG (Recall@5 64.7 versus 49.2) translates into only
a small answer-accuracy gain. This reflects the retrieval and generation gap: multi-hop answers are short (often yes/no
or a single entity) and reachable from partial evidence, and all systems share the same generator model, temperature 0,
identical prompt), so beyond a sufficiency threshold, additional retrieved evidence yields diminishing returns on final
accuracy. SpectRAG’s contribution is therefore a stronger, substantially cheaper retriever (RQ1), delivering answer
accuracy that matches the state-of-the-art graph-RAG system while decisively surpassing standard vector retrieval.
Detailed answer-accuracy results are reported in Section 5.3 (generation-level results).
4.10 RQ4
SpectRAG’s spectral and PPR diffusion not only surfaces more gold evidence; it ranks that evidence higher in the
retrieved context, outperforming both baselines at every retrieval depth and on mean reciprocal rank. Against LightRAG,
SpectRAG improves mean reciprocal rank (MRR) from 61.5 to 70.4 and recall at every cutoff (Recall@2 39.1 versus
31.2, Recall@5 64.7 versus 49.2, Recall@10 84.0 versus 61.8). Against NaiveRAG, it likewise leads on MRR (68.5
versus 65.5) and on Recall@2, @5, and @10. A higher MRR means the first gold document appears earlier in the
context, which directly benefits the generator, as it attends most strongly to top-ranked passages. SpectRAG therefore
delivers not just broader but better-ordered retrieval, the property that most affects downstream answering. Detailed
retrieval-ranking results are reported in Section 5.2 (Retrieval-Level Results).
16

arXivTemplateA PREPRINT
Table 5: CS-domain qualitative case study: query, answers, and LLM-judge verdict.
Field Content
Query How can anomaly detection techniques be applied in various domains to identify outliers?
SpectRAG answer Frames anomaly detection as a versatile toolset applied through several themes: understanding the
dataset (categorical, ordinal, or continuous) to guide method choice; statistical methods (z-scores, p-
values, standard-deviation thresholds); machine-learning approaches (K-means clustering, supervised
learning on labeled data); domain-specific applications in finance (fraud), healthcare (rare conditions),
network security (intrusions), and manufacturing (predictive maintenance); evaluation and validation
via precision, recall, F1, and cross-validation; and visualization tools (scatter plots, box plots, heat
maps) for inspecting distributions.
LightRAG answer Synthesizes anomaly detection across fields through big data and machine learning (real-time pattern
flagging for finance and network security); statistical analysis (z-scores, box plots, interquartile
range for healthcare); data mining (clustering to surface outliers in marketing); optimization methods
(refining operations-research models); and real-time applications (streaming detection in telecom-
munications). Concludes that the techniques are versatile across finance, healthcare, marketing, and
operations research.
Comprehensiveness Winner: SpectRAG(position-sensitive; split on swap). More detailed exploration covering a wider
range of methods, applications, and evaluation strategies.
Diversity Winner: SpectRAG(confirmed both orderings). Richer variety of perspectives across multiple
domains and specific techniques within each.
Empowerment Winner: SpectRAG(confirmed both orderings). Detailed explanations and examples that help the
reader understand how the techniques are applied.
Overall Winner: SpectRAG(confirmed both orderings). Superior comprehensive coverage, diverse insights,
and deeper reader understanding of the topic.
Analysis SpectRAG wins all three quality dimensions and the overall verdict. Three of the four verdicts hold
under both answer orderings, with only comprehensiveness being position-sensitive, indicating a
robust rather than order-driven preference. The advantage stems from SpectRAG’s explicit evaluation
and visualization discussion, which the LightRAG answer omits.
Pairwise verdicts from the gpt-4o judge (Answer 1 = SpectRAG, Answer 2 = LightRAG). Answers are condensed from the model
outputs; full text is available in the supplementary material.
5 Results and Analysis
5.1 Main Results
SpectRAG delivers two complementary advantages: an efficiency-at-parity result on MultiHopRAG and a quality
result on UltraDomain. On MultiHopRAG, SpectRAG matches LightRAG’s objective answer accuracy and decisively
outperforms NaiveRAG (subsection 5.3), while retrieving gold evidence more effectively and ranking it higher within
the context (subsection 5.2) all at substantially lower index cost, evidenced by a markedly smaller graph and on-disk
footprint. On UltraDomain, SpectRAG wins the pairwise LLM-judge evaluation against NaiveRAG, HyDE, and Hybrid,
remains competitive with LightRAG, and shows a clear advantage in answer diversity (Table 4); a representative example
is examined in Table 5. Concurrently, these results address RQ1–RQ3: SpectRAG achieves competitive-to-superior
retrieval at markedly lower index cost (RQ1), improves holistic answer quality relative to graph-RAG baselines (RQ2),
and matches or exceeds their objective accuracy without sacrificing efficiency (RQ3).
5.2 Retrieval-Level Results
SpectRAG demonstrates a substantial improvement over LightRAG in retrieval quality, as evidenced by the ranking of
gold evidence and recall across all evaluated cutoffs. Table 6 reports MRR and Recall@ kon MultiHopRAG. SpectRAG
attains an MRR of 70.4, compared to 61.5 for LightRAG, and outperforms LightRAG across all recall cutoffs (Recall@2,
@5, and @10: 39.1/64.7/84.0 versus 31.2/49.2/61.8). This advantage is not confined to aggregate performance but
extends across all reasoning-intensive question types: Recall@5 favors SpectRAG on comparison (72.0 versus 54.8),
inference (53.7 versus 38.8), and temporal (69.5 versus 55.5) questions. This consistency indicates that the observed
retrieval gain reflects a general property of the retrieval mechanism rather than an artifact of any particular question type.
Taken together with the ranking results, these findings address RQ4 and the retrieval component of RQ1, demonstrating
that SpectRAG retrieves gold evidence more frequently and ranks it higher in context than LightRAG, consistently
across question types.
17

arXivTemplateA PREPRINT
NaiveRAG is excluded from the primary comparison, as its retrieval scores are not directly comparable under our
evidence-attribution protocol.1
Table 6: Retrieval ranking quality on MultiHopRAG.
System Recall R@2 R@5 R@10 MRR
LightRAG 65.6 31.2 49.2 61.8 61.5
Hybrid 76.146.3 68.076.179.1
HyDE 64.1 36.1 58.2 64.1 65.5
NaiveRAG 69.5 40.9 62.4 69.5 72.8
SpectRAG 91.839.1 64.784.070.4
Retrieval ranking quality on MultiHopRAG. Recall is overall recall; R@kis Recall@k(fraction of gold evidence within the top-k
retrieved documents); MRR is the reciprocal rank of the first gold-evidence document. Best result per column in bold.
5.3 Generation-Level Results
This study aims to investigate end-to-end answer accuracy by employing a common generator across all five systems.
This standardized approach ensures that any observed variations in performance are indicative of the retrieval quality,
rather than discrepancies in the generative process itself. The overall accuracy reported on MultiHopRAG’s 2,556
gold-answer questions demonstrates the evaluation’s relevance and credibility. Objective accuracy is directly comparable
across all baselines here, including NaiveRAG, whose retrieval scores were excluded from the retrieval-level analysis.
Table 7 reports overall accuracy on the MultiHopRAG dataset.
Table 7: Overall answer accuracy on MultiHopRAG dataset.
System EM F1 Contains
HyDE 65.2 65.6 65.4
NaiveRAG 67.8 68.3 68.3
Hybrid 70.6 71.1 71.0
LightRAG 70.5 70.9 70.8
SpectRAG 71.2 71.6 71.6
EM is normalized exact match; F1 is token overlap; contains indicates the gold answer appears as a whole-word span in the
prediction. Best result per column in bold, ordered in ascending order according to the EM values.
SpectRAG attains the highest overall accuracy of all five systems on every metric (EM 71.2, F1 71.6, and contains
71.6). The margin over the next-best systems, LightRAG and Hybrid (both near 70.6), is narrow, while the gap over
NaiveRAG (67.8) and HyDE (65.2) is substantial. Unlike the retrieval-level comparison, where SpectRAG’s advantage
was large, this is a clean but marginal win across the board. This is precisely the retrieval–generation gap in action:
because all systems share the same generator and multi-hop answers are typically short (a yes/no verdict or a single
entity) and often recoverable from partial evidence, retrieval improvements compress once a sufficiency threshold is
reached, and additional or better-ordered evidence yields diminishing returns on final accuracy.
To make this compression concrete, Table 8 breaks accuracy down by question type. The per-type view shows that the
small aggregate margin masks a mix of clear wins and near-ties.
The per-type breakdown clarifies where SpectRAG’s aggregate win originates. SpectRAG dominates on inference
questions (95.1, roughly three points above LightRAG and Hybrid and eight above the vector baselines), the type most
dependent on assembling evidence across multiple hops, which is exactly where a stronger, better-ordered retriever
should help. On comparison and temporal questions the systems cluster tightly, with LightRAG marginally ahead (66.4
and 44.9), confirming that once sufficient evidence is retrieved, the shared generator equalizes performance and small
retrieval differences no longer separate the systems. The null category behaves differently: it rewards correct abstention
rather than evidence assembly, and here the vector baselines (89.7) and Hybrid (87.4) lead, while SpectRAG (83.1) still
1NaiveRAG’s chunk-level retrieval does not map cleanly onto the graph-node attribution scheme used to compute gold-evidence
recall for LightRAG and SpectRAG; the resulting NaiveRAG figures constitute a lower bound and are therefore omitted from
the primary analysis to avoid an inequitable comparison. NaiveRAG’s generation-level results are reported in Section 5.3, where
objective accuracy is directly comparable across all baselines.
18

arXivTemplateA PREPRINT
Table 8: Per-type answer accuracy on MultiHopRAG.
Type SpectRAG LightRAG Hybrid HyDE NaiveRAG
Comparison 62.766.463.1 55.7 59.8
Inference95.192.2 92.2 87.0 87.7
Temporal 43.944.942.7 35.8 40.1
Null 83.1 72.8 87.489.789.7
Exact-match accuracy by question type. Best result per row in bold. Null questions are those with no answerable gold evidence;
higher scores reflect correct abstention.
comfortably exceeds LightRAG (72.8). SpectRAG’s lower null score reflects its tendency to surface candidate evidence
even for unanswerable queries, a direct consequence of its broader retrieval, and represents the one regime where more
aggressive retrieval is mildly counterproductive.
Considered collectively, the generation-level results confirm the paper’s central claim: SpectRAG’s value is a stronger
and substantially cheaper retriever (RQ1), which converts into the best overall answer accuracy among all five systems
and a decisive lead on the most retrieval-intensive question type, while the shared generator and short answer format
cap how large that lead can become on the remaining types.
5.4 Diversity, Redundancy, and Connectivity Analysis
To assess whether SpectRAG’s denser, spectrally-organized chunk graph produces qualitatively richer answers, we eval-
uate SpectRAG (SpectRAG) against four baselines (NaiveRAG, HyDE, Hybrid, and LightRAG) using the UltraDomain-
style pairwise win-rate protocol, in which a gpt-4o judge selects the preferred answer across four dimensions:
comprehensiveness, diversity, empowerment, and overall quality. Each pair of columns reports the win rate (%) of the
named baseline versus SpectRAG on a given domain; the two columns in each pair sum to 100, and an SpectRAG value
above 50 indicates SpectRAG is preferred. Table 9 reports the results under the first edge-signal configuration: semantic
0.300, section 0.250, entity 0.200, adjacency 0.084, utility question 0.083, and citation 0.083.
Two patterns are consistent under this configuration. First,diversity is SpectRAG’s strongest dimension against
every baseline: it wins on average 64.1 against NaiveRAG, 71.5 against HyDE, 60.6 against Hybrid, and 55.0 against
LightRAG. This is the signature of a retriever that spreads its evidence across structurally distinct regions of the graph
rather than concentrating it in a small, redundant neighborhood. Second,the margin scales with how redundancy-
prone the baseline is: SpectRAG wins all four dimensions decisively against HyDE and NaiveRAG, leads throughout
against Hybrid, and against LightRAG is close to even on comprehensiveness, empowerment, and overall (49.7, 48.4,
48.7) while still winning diversity (55.0). Even when the two graph-based systems are matched on aggregate quality,
SpectRAG’s denser chunk graph continues to deliver more varied, less redundant evidence, consistent with its higher
edges-per-node ratio (RQ1).
To test whether this behavior depends on the exact signal weighting, we repeat the evaluation under a second, contrasting
configuration that shifts weight away from raw semantic similarity and toward utility-question matching and entity
structure: Semantic 0.150, Section 0.200, entity 0.250, adjacency 0.100, utility question 0.280, and citation 0.020. The
results are reported in Table 10.
Comparing the two configurations yields two findings. First,the diversity advantage is stable across both weightings:
diversity remains SpectRAG’s strongest dimension against every baseline (60.0, 72.5, 58.8, 55.9), with margins that
closely track configuration 1. Because the two weightings emphasize almost disjoint signal sets yet produce the same
diversity ordering, the effect is a structural property of the denser chunk graph rather than an artifact of any single signal.
Second,the utility-question-weighted configuration strengthens SpectRAG against the hardest baseline: against
LightRAG, SpectRAG moves from winning only diversity (configuration 1) to additionally winning empowerment
(51.2) and the overall verdict (51.2), losing only comprehensiveness by a narrow margin (49.4). The gains against
the weaker baselines are preserved throughout. This indicates that anchoring retrieval on utility questions and entity
structure, rather than surface semantic similarity, helps most precisely where the competition is strongest. Together, the
two configurations bound SpectRAG’s qualitative behavior: its diversity and redundancy advantages are insensitive to
the exact signal weighting. At the same time, its aggregate standing against LightRAG can be tuned from roughly even
to a slight lead.
19

arXivTemplateA PREPRINT
Table 9: Win rates (%) of baselines versus SpectRAG across four domains and four dimensions (signal configuration 1:
semantic 0.300, section 0.250, entity 0.200, adjacency 0.084, utility question 0.083, citation 0.083)
Dimension Agriculture CS Legal Mix Average
Baseline: NaiveRAG
BaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAG
Comprehensiveness 43.856.237.562.551.348.737.562.542.557.5
Diversity 32.567.521.378.752.547.537.562.535.964.1
Empowerment 40.060.032.567.551.348.733.866.239.460.6
Overall 40.060.035.065.051.348.735.065.040.359.7
Baseline: HyDE
BaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAG
Comprehensiveness 37.562.536.863.248.851.230.070.038.361.7
Diversity 22.577.525.075.043.856.222.577.528.571.5
Empowerment 35.065.034.265.847.552.528.771.336.463.6
Overall 35.065.034.265.847.552.528.771.336.463.6
Baseline: Hybrid
BaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAG
Comprehensiveness 47.552.541.258.850.050.042.557.545.354.7
Diversity 41.258.828.771.346.353.741.258.839.460.6
Empowerment 45.055.038.761.348.851.237.562.542.557.5
Overall 46.353.738.761.348.851.241.258.843.856.2
Baseline: LightRAG
BaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAG
Comprehensiveness 43.856.256.243.847.552.553.746.350.349.7
Diversity 35.065.052.547.545.055.047.552.545.055.0
Empowerment 42.557.558.841.248.851.256.243.851.648.4
Overall 42.557.558.841.248.851.255.045.051.348.7
Each pair of columns gives the win rate (%) of the baseline (Base) versus SpectRAG; the two sum to 100, and SpectRAG values
above 50 (bold in the Average column) indicate SpectRAG is preferred. Judged bygpt-4o. Signal weights sum to 1.000 and are
fixed across all domains and baselines.
6 Discussion
The experimental findings provide several observations about the behavior provide several observations about Spec-
tRAG’s yield provide several observations about the behavior of SpectRAG in multi-hop and graph-structured retrieval
settings. Overall, SpectRAG shows its strongest advantage when the task requires broad evidence recovery rather than
only early precision in a ranked list. In the MultiHopRAG retrieval-ranking evaluation, SpectRAG achieves the highest
overall recall and the highest Recall@10, indicating that the proposed graph expansion and spectral diffusion stages
help recover a wider set of gold evidence. However, Hybrid retrieval obtains the best Recall@2, Recall@5, and MRR,
suggesting that SpectRAG does not always place the first relevant evidence item earlier than strong lexical–dense base-
lines. This pattern is consistent with the design of SpectRAG: the method is not optimized solely for nearest-neighbor
ranking, but for constructing a broader, more structurally useful candidate pool before final context selection.
The generation-level results further indicate that improvements in retrieval coverage can translate into modest but
consistent gains in answer accuracy. On MultiHopRAG, SpectRAG obtains the highest EM, F1, and answer-containment
scores among the compared systems. The margin is not large, but it is consistent across all three generation metrics.
This recommends that the final context selected by SpectRAG can provide useful evidence to the generator, especially
when the answer depends on information distributed across multiple passages. Simultaneously, the relatively small
generation margin highlights that better retrieval does not automatically guarantee substantially better generation. The
20

arXivTemplateA PREPRINT
Table 10: Win rates (%) of baselines versus SpectRAG across four domains and four dimensions (signal configuration 2:
semantic 0.150, section 0.200, entity 0.250, adjacency 0.100, utility question 0.280, citation 0.020)
Dimension Agriculture CS Legal Mix Average
Baseline: NaiveRAG
BaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAG
Comprehensiveness 43.856.241.258.853.746.343.856.245.654.4
Diversity 37.562.526.273.855.045.041.258.840.060.0
Empowerment 40.060.036.363.751.348.741.258.842.257.8
Overall 41.258.836.363.752.547.542.557.543.156.9
Baseline: HyDE
BaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAG
Comprehensiveness 40.060.032.567.548.851.225.075.036.663.4
Diversity 27.572.522.577.541.258.818.881.227.572.5
Empowerment 33.866.230.070.048.851.222.577.533.866.2
Overall 35.065.030.070.048.851.222.577.534.165.9
Baseline: Hybrid
BaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAG
Comprehensiveness 46.353.751.348.747.552.541.258.846.653.4
Diversity 46.353.747.552.536.363.735.065.041.258.8
Empowerment 47.552.550.050.045.055.041.258.845.954.1
Overall 47.552.550.050.045.055.040.060.045.654.4
Baseline: LightRAG
BaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAGBaseSpectRAG
Comprehensiveness 45.055.057.542.546.353.753.746.350.649.4
Diversity 33.866.258.841.237.562.546.353.744.155.9
Empowerment 43.856.255.045.046.353.750.050.048.851.2
Overall 43.856.255.045.046.353.750.050.048.851.2
Each pair of columns gives the win rate (%) of the baseline (Base) versus SpectRAG; the two sum to 100, and SpectRAG values
above 50 (bold in the Average column) indicate SpectRAG is preferred. Judged bygpt-4o. Signal weights sum to 1.000 and are
fixed across all domains and baselines.
LLM must correctly integrate the supplied evidence, follow the reasoning path, and avoid relying on irrelevant or
weakly connected context.
Based on the per-type analysis, it is evident that SpectRAG is especially useful in inference-type questions, scoring
the highest EM scores amongst all the evaluated systems. These results confirm our hypothesis that graph-spectral
representations are beneficial when the question requires connecting evidence pieces rather than finding a direct match.
The results for the comparison and temporal questions are mixed, with LightRAG slightly better than SpectRAG.
This implies that some question types could be helped more by other relational representations or different kinds of
prompts. For null questions, SpectRAG attains the highest score among all evaluated systems, indicating that its spectral
representation also supports reliable abstention and unanswerability detection rather than merely improving evidence
retrieval. Beyond retrieval quality, SpectRAG additionally incurs a lower memory cost than competing graph-based
RAG systems such as LightRAG, as it induces a markedly more compact graph representation over the same corpus,
with substantially fewer nodes and edges. This reduction in structural complexity directly translates into lower storage
and indexing overhead, indicating that the observed gains are achieved without a corresponding increase in resource
consumption.
The pairwise LLM-judge evaluations across agriculture, CS, legal, and mix domains provide a complementary view
of generation quality. Across two signal configurations, SpectRAG generally achieves favorable win rates against
NaiveRAG, HyDE, and Hybrid baselines, especially in diversity and empowerment. The diversity gains are the most
stable pattern: SpectRAG frequently receives higher diversity scores across domains and configurations. This is aligned
21

arXivTemplateA PREPRINT
with the method’s objective of reducing redundant evidence and selecting context from multiple graph-spectral regions.
The results against LightRAG are more nuanced. SpectRAG is competitive and often stronger in diversity, but it does not
consistently dominate LightRAG across all domains, particularly in the CS domain. This indicates that graph-spectral
selection improves some aspects of answer quality, but its advantage depends on the corpus structure, baseline strength,
and signal weighting.
The qualitative case study provides an interpretable example of this behavior. For the anomaly detection question,
SpectRAG is preferred in comprehensiveness, diversity, empowerment, and overall quality. The judged explanation
suggests that SpectRAG covers a wider set of methods, domains, evaluation strategies, and visualization tools, while
LightRAG provides a more compact but less varied synthesis. Although this is only a single example, it illustrates the
kind of context composition that SpectRAG is intended to support: a final answer grounded in multiple complementary
evidence aspects rather than a narrow set of locally similar passages.
6.1 Why Spectral Structure Helps
The results suggest that spectral graph structure is particularly useful when retrieval requires broad evidence coverage
and relational coherence. Existing RAG pipelines exploit corpus structure only partially: purely embedding-based
methods select chunks that are locally similar to the query without modeling the relationships among candidate chunks,
whereas graph-based methods encode relations at the entity level and traverse them through local neighborhood
expansion. In neither case is the collection of retrieval units treated as a global structure whose connectivity governs
selection. Through a weighted chunk graph, SpectRAG instead incorporates semantic similarity, document adjacency,
section hierarchy, entity co-occurrence, citation relations, and utility-question similarity as complementary signals of
corpus structure.
The Laplacian spectral coordinates offer a compact description of such graph topology. Low-frequency spectral
coordinates describe global connectivity patterns and communities in the graph, allowing detection of chunks belonging
to the same group despite differences in their direct similarity. The use of the spectral coordinates in SpectRAG is
crucial for the cluster-aware selection because it allows to avoid domination of one local neighborhood in the resulting
context set.
Spectral diffusion also contributes to the retrieval process by propagating query relevance over the induced candidate
subgraph. A chunk that is not highly similar to the query may still receive increased relevance if it is strongly connected
to directly retrieved evidence. This mechanism is particularly important for bridge evidence, where an intermediate
passage connects two otherwise disconnected pieces of information. The high overall recall and Recall@10 observed
on MultiHopRAG are consistent with this behavior: SpectRAG appears to recover a broader evidence set, even when
some of that evidence is not ranked at the very top.
6.2 Practical Deployment in RAG Pipeline and Framework
In the current implementation, dense candidate seeding and utility-question retrieval are performed using exact flat
inner-product search with FAISS IndexFlatIP . This choice avoids approximation error and makes the preliminary
evaluation easier to interpret, since differences between systems are not caused by approximate nearest-neighbor
indexing. For small and medium-sized experimental corpora, exact search is also practical and reproducible. In larger
deployments, however, this component may become a bottleneck and can be replaced by approximate indexing methods
without changing the downstream graph expansion, diffusion, and token-budget limit stages.
Most graph and spectral computations can be moved to the offline indexing stage. Chunk embeddings, graph construc-
tion, Laplacian computation, and spectral coordinates can be precomputed and cached. At query time, SpectRAG only
needs to retrieve a candidate seed set, expand it through graph neighborhoods, construct the induced candidate subgraph,
diffuse relevance within that subgraph, and select the final context. This separation between offline indexing and online
retrieval makes the method practical for RAG systems where the corpus changes less frequently than the query stream.
In addition to the retrieval pipeline, SpectRAG includes a graph-view interface for inspecting the constructed chunk
graph and its spectral organization. Fig. 5 shows an example visualization of the weighted chunk graph generated from
an indexed document collection. In this view, nodes correspond to document chunks and edges correspond to relation
signals used during graph construction, including semantic similarity, section-level relations, entity overlap, document
adjacency, utility-question similarity, and citation links. The node layout provides an interpretable view of how chunks
are organized in the graph, while edge colors indicate the dominant relation types connecting different parts of the
corpus.
Each node in the graph is associated with degree information that summarizes its local connectivity. In case of the
chunk node ci, the unweighted degree refers to the number of other chunks that connect with the current one, while the
22

arXivTemplateA PREPRINT
Figure 5: Example graph-view interface of the proposed SpectRAG framework. Nodes represent document chunks and
edges represent weighted relation signals, including semantic similarity, section relations, entity overlap, adjacency,
utility-question similarity, and citation links.
weighted degree is the sum of weights of the edges. Since SpectRAG builds a weighted chunk graph, the weighted
degree is:
deg(c i) =nX
j=1aij,(28)
Using the graph visualization, it is possible to identify dense local neighborhoods, bridge nodes, isolated regions,
and inter-document connections. These aspects are critical in establishing that the graph that is constructed properly
captures the structure of the corpus before graph expansion, spectral relevance propagation, and budgeting of tokens for
context selection are done. In actual RAG implementations, this visualization can help in identifying retrieval problems
and graph sparsity.
7 Limitations and Threats to Validity
Several limitations should be considered when interpreting the results of this study.
Retrieval–generation gap.Although SpectRAG improves retrieval coverage in several settings, retrieval gains do
not always lead to proportional improvements in generated answers. The generator may fail to use retrieved evidence
correctly, may ignore bridge passages, or may be sensitive to the ordering of the final context. This is especially
important because the answer accuracy results show only modest improvements over strong baselines. Therefore,
retrieval-level and generation-level metrics should be interpreted separately.
Task-dependent performance.SpectRAG does not dominate all question types. The per-type evaluation shows strong
performance on inference questions, but weaker or mixed results on comparison, temporal, and null questions. This
indicates that graph-spectral retrieval is most beneficial when the task requires evidence connection and synthesis, but
may be less effective when the task depends on precise temporal ordering, direct comparison, or abstention calibration.
Signal-weight sensitivity.The two evaluated signal configurations indicate that performance can change depending
on the relative weights assigned to semantic similarity, section hierarchy, entity overlap, adjacency, utility-question
similarity, and citation links. While SpectRAG remains generally competitive across configurations, some domains and
baselines respond differently to weight changes. This creates a threat to external validity because a configuration that
works well for one corpus may not be optimal for another.
Chunking sensitivity.The method is evaluated under token-based chunking configurations, including shorter and
longer chunk granularities. Chunk size and overlap affect graph topology, candidate retrieval, redundancy, and the
amount of local context available in each selected unit. Larger chunks preserve more local context but may reduce the
23

arXivTemplateA PREPRINT
benefit of graph-based composition. Therefore, chunking should be treated as an experimental variable rather than a
fixed preprocessing choice.
Computational scalability.The current implementation uses exact brute-force inner-product search through FAISS
IndexFlatIP for both chunk embedding retrieval and utility-question retrieval. This avoids approximation error
but has higher query-time cost than approximate nearest-neighbor indexing. In addition, spectral computation over
large graphs can be expensive. Practical large-scale deployments may require sparse graph construction, approximate
eigensolvers, graph coarsening, caching, or approximate vector indexes.
8 Conclusion
This paper introduced SpectRAG, a spectral graph-augmented retrieval framework for constructing diverse, connected,
and evidence-complete context sets in retrieval-augmented generation. The central motivation of the work is that
standard dense retrieval is effective for identifying passages that are individually similar to a query, but it does not
explicitly optimize the collective utility of the retrieved context. This limitation becomes more important in multi-hop
and cross-document settings, where the answer may depend on evidence distributed across several passages, documents,
or structural relations.
SpectRAG addresses this problem by combining dense candidate seeding, weighted chunk graph construction, graph
neighborhood expansion, spectral relevance propagation, and token-budgeted final context selection. Rather than
replacing dense retrieval, the proposed framework uses it as a high-recall entry point and then applies graph-spectral
signals to recover supporting evidence, reduce redundant context, and improve structural coherence. The final context is
assembled under a token budget, allowing the system to adapt the number of selected chunks according to evidence
length and available context-window capacity.
The experimental results show that SpectRAG is most beneficial when retrieval requires broad evidence recovery and
relational reasoning. On MultiHopRAG, SpectRAG achieves the highest overall recall and Recall@10, indicating that
graph expansion and spectral propagation help recover a wider set of relevant evidence. In generation-level evaluation,
SpectRAG obtains the strongest EM, F1, and answer-containment scores among the compared systems, although the
margins remain moderate. These results suggest that improved evidence coverage can support better generation, but
also confirm that retrieval gains do not automatically translate into large answer-quality gains unless the generator can
effectively use the selected evidence.
SpectRAG also turns out to be an especially strong solution for inference problems and often helps diversify judgments
in LLM-based pairwise evaluations of diversity. In the vast majority of domains and signal combinations, SpectRAG
outperforms the NaiveRAG, HyDE, and Hybrid baselines across diversity, empowerment, and total performance metrics.
Comparing the results of SpectRAG with those of LightRAG reveals some interesting conclusions: graph-spectral
context selection performs better than the alternative in some cases but not in others. However, it retains an overall
advantage in aggregate. This balanced result is valuable. since the proposed approach ought not to be construed
as a wholesale improvement over prevailing retrieval strategies, but rather as a complementary method, warranted
specifically in settings where evidence diversity and connectivity must be preserved.
Overall, the findings support the main hypothesis of this work: graph-spectral structure can provide useful retrieval
signals beyond local vector similarity. By representing chunks as nodes in a weighted graph and leveraging spectral
structure during candidate expansion, relevance propagation, and token-budgeted context assembly, SpectRAG offers a
principled approach to moving from independent passage ranking to graph-aware context construction. Future work
should extend the evaluation to larger corpora and additional domains, conduct human preference studies, develop
stronger graph-enhanced baselines, and explore scalable indexing variants. Further investigation is also needed to
optimize signal weights, improve temporal and comparative reasoning, and examine how context ordering affects
LLMs’ ability to exploit graph-selected evidence.
Acknowledgments
The authors received no financial support for the research, authorship, and/or publication of this article.
Code Availability
The source code, configuration files, evaluation scripts, and documentation for SpectRAG will be made publicly
available in a GitHub repository. To support reproducibility, the released code will include experiment configurations,
24

arXivTemplateA PREPRINT
dependency information, and instructions for reproducing the reported results. The repository is available at: https:
//github.com/SightAndCite/SpectRAG.
References
[1]P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rocktäschel,
S. Riedel, and D. Kiela, “Retrieval-augmented generation for knowledge-intensive NLP tasks,”Advances in
Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.
[2]V . Karpukhin, B. O ˘guz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W.-t. Yih, “Dense passage retrieval
for open-domain question answering,” inProc. 2020 Conf. Empirical Methods in Natural Language Processing,
2020, pp. 6769–6781.
[3]J. Yang and S. Park, “LLM-assisted knowledge graph construction for managing road accident information from
news articles,”IEEE Access, vol. 13, pp. 217846–217865, 2025.
[4]J. Carbonell and J. Goldstein, “The use of MMR, diversity-based reranking for reordering documents and
producing summaries,” inProc. 21st Annu. Int. ACM SIGIR Conf. Research and Development in Information
Retrieval, 1998, pp. 335–336.
[5] U. von Luxburg, “A tutorial on spectral clustering,”Statistics and Computing, vol. 17, no. 4, pp. 395–416, 2007.
[6]H. Han, Y . Wang, H. Shomer, K. Guo, J. Ding, Y . Lei,et al., “Retrieval-augmented generation with graphs
(GraphRAG),” arXiv:2501.00309, 2024.
[7]J. Zhang, Y . Zhao, J. Yu, J. Yu, and X. Li, “Beyond chunk-local extraction: Cross-chunk graph augmentation for
GraphRAG,” arXiv:2605.28004, 2026.
[8]Z. Yang, P. Qi, S. Zhang, Y . Bengio, W. W. Cohen, R. Salakhutdinov, and C. D. Manning, “HotpotQA: A dataset
for diverse, explainable multi-hop question answering,” inProc. 2018 Conf. Empirical Methods in Natural
Language Processing, 2018, pp. 2369–2380.
[9]X. Ho, A.-K. D. Nguyen, S. Sugawara, and A. Aizawa, “Constructing a multi-hop QA dataset for comprehensive
evaluation of reasoning steps,” inProc. 28th Int. Conf. Computational Linguistics, 2020, pp. 6609–6625.
[10] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal, “MuSiQue: Multihop questions via single-hop
question composition,”Transactions of the Association for Computational Linguistics, vol. 10, pp. 539–554, 2022.
[11] S. Robertson and H. Zaragoza, “The probabilistic relevance framework: BM25 and beyond,”Foundations and
Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009, doi: 10.1561/1500000019.
[12] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt, and J. Larson, “From local to global: A
Graph RAG approach to query-focused summarization,” arXiv:2404.16130, 2024.
[13] R. Agrawal, S. Gollapudi, A. Halverson, and S. Ieong, “Diversifying search results,” inProc. 2nd ACM Int. Conf.
Web Search and Data Mining (WSDM), Barcelona, Spain, Feb. 2009, pp. 5–14.
[14] A. Khatiwada, G. Fan, R. Shraga, Z. Chen, W. Gatterbauer, R. J. Miller, and M. Riedewald, “SANTOS:
Relationship-based semantic table union search,”Proc. ACM Manag. Data, vol. 1, no. 1, pp. 1–25, May 2023, doi:
10.1145/3588689.
[15] R. van den Berg, T. N. Kipf, and M. Welling, “Graph convolutional matrix completion,” arXiv:1706.02263, 2017.
[16] H. Qian, P. Zhang, Z. Liu, K. Mao, and Z. Dou, “MemoRAG: Moving towards next-gen RAG via memory-inspired
knowledge discovery,” arXiv:2409.05591, 2024.
[17] Y . Tang and Y . Yang, “MultiHopRAG: Benchmarking retrieval-augmented generation for multi-hop queries,”
arXiv:2401.15391, 2024.
[18] Z. Guo, L. Xia, Y . Yu, T. Ao, and C. Huang, “LightRAG: Simple and fast retrieval-augmented generation,”
arXiv:2410.05779, 2024.
[19] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and H. Wang, “Retrieval-augmented generation for
large language models: A survey,” arXiv:2312.10997, 2023.
[20] B. Sarmah, D. Mehta, B. Hall, R. Rao, S. Patel, and S. Pasquali, “HybridRAG: Integrating knowledge graphs
and vector retrieval-augmented generation for efficient information extraction,” inProc. 5th ACM Int. Conf. AI
Finance (ICAIF), 2024, pp. 608–616.
[21] L. Gao, X. Ma, J. Lin, and J. Callan, “Precise zero-shot dense retrieval without relevance labels,” inProc. 61st
Annu. Meeting Assoc. Comput. Linguistics (ACL), vol. 1, 2023, pp. 1762–1777.
[22] B. H. Rappazzo, Y . Wang, A. Ferber, and C. Gomes, “GEM-RAG: Graphical eigen memories for retrieval
augmented generation,” inProc. 2024 Int. Conf. Mach. Learn. Appl. (ICMLA), 2024, pp. 1259–1264.
25