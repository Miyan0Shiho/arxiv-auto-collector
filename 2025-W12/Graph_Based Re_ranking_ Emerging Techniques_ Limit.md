# Graph-Based Re-ranking: Emerging Techniques, Limitations, and Opportunities

**Authors**: Md Shahir Zaoad, Niamat Zawad, Priyanka Ranade, Richard Krogman, Latifur Khan, James Holt

**Published**: 2025-03-19 00:28:54

**PDF URL**: [http://arxiv.org/pdf/2503.14802v1](http://arxiv.org/pdf/2503.14802v1)

## Abstract
Knowledge graphs have emerged to be promising datastore candidates for
context augmentation during Retrieval Augmented Generation (RAG). As a result,
techniques in graph representation learning have been simultaneously explored
alongside principal neural information retrieval approaches, such as two-phased
retrieval, also known as re-ranking. While Graph Neural Networks (GNNs) have
been proposed to demonstrate proficiency in graph learning for re-ranking,
there are ongoing limitations in modeling and evaluating input graph structures
for training and evaluation for passage and document ranking tasks. In this
survey, we review emerging GNN-based ranking model architectures along with
their corresponding graph representation construction methodologies. We
conclude by providing recommendations on future research based on
community-wide challenges and opportunities.

## Full Text


<!-- PDF content starts -->

Graph-Based Re-ranking:
Emerging Techniques, Limitations, and Opportunities
Md Shahir Zaoad*1,Niamat Zawad*1,Priyanka Ranade2,Richard Krogman2,Latifur
Khan1,James Holt2
1Department of Computer Science, University of Texas, Dallas, USA.
2The Laboratory for Physical Sciences, Baltimore, MD, USA.
∗
{mxz230002, nxz190009, lkhan }@utdallas.edu, {psranad, holt,rokrogm }@lps.umd.edu
Abstract
Knowledge graphs have emerged to be promising
datastore candidates for context augmentation dur-
ing Retrieval Augmented Generation (RAG). As a
result, techniques in graph representation learning
have been simultaneously explored alongside prin-
cipal neural information retrieval approaches, such
as two-phased retrieval, also known as re-ranking .
While Graph Neural Networks (GNNs) have been
proposed to demonstrate proficiency in graph learn-
ing for re-ranking, there are ongoing limitations in
modeling and evaluating input graph structures for
training and evaluation for passage and document
ranking tasks. In this survey, we review emerg-
ing GNN-based ranking model architectures along
with their corresponding graph representation con-
struction methodologies. We conclude by provid-
ing recommendations on future research based on
community-wide challenges and opportunities.
1 Introduction
Retrieval Augmented Generation (RAG) is an established re-
search area that combines pretrained parametric and non-
parametric memory for downstream language generation
[Lewis et al. , 2020 ]. Recently, there has been an emergence
of using Knowledge Graphs as the external non-parametric
datastore, in which structural information is queried to cap-
ture relational knowledge [Dong et al. , 2024b ]. Graph-RAG
approaches typically follow a two-phased retrieval proce-
dure, also known as re-ranking [Peng et al. , 2024 ]. Two-
phased retrieval approaches consist of a primary retrieval
method in which given a search query, an initial set of prob-
able responses is fetched using techniques such as approxi-
mate nearest neighbor (ANN) indexing, traditional keyword-
based search, or embedding-based retrieval. The initial re-
trieval process generally often prioritizes compute efficiency
over perfect accuracy, leading to prompting with irrelevant,
noisy context and increased hallucination to the final output
[Glass et al. , 2022 ]. In a two-phase setup, re-ranking methods
distill the initial set of retrieved documents by re-scoring the
initial list according to a refined relevance score.
∗Equal Contribution.Querying the complex structure of the knowledge graph
presents challenges for popular Large Language Model
(LLM)-based re-ranker models. This challenge has in-
spired recent research in exploring the potential of Graph
Neural Networks (GNNs) for exploiting structural informa-
tion across entities and capturing relational knowledge for
prompting a language model for generation.
For effective graph-based re-ranking, researchers have de-
veloped expansive and specialized methods focused primarily
on developing (1) unique GNN model architectures and (2)
constructing unique graph representation structures specifi-
cally optimized for retrieval tasks. However, many of these
methods have only recently emerged, leading to gaps in
community-wide evaluation of model architectures and best
practices. Additionally, these methods have not yet been col-
lated into a systematic review. To fill this gap, we provide
a comprehensive overview of emerging GNN-based ranking
model architectures and their corresponding graph represen-
tation construction methodologies. We conclude by provid-
ing recommendations on future research based on our find-
ings and analysis of limitations.
2 Graph-Based Retrieval and Re-ranking
Re-ranking problems abstractly consider a situation where
there is a query qand a finite set D={di}i∈[n]of documents,
assumed to have been selected a priori from an arbitrary re-
trieval process. The goal is to obtain a re-ranking of the el-
ements of D, via a scoring function rq:D→(0,1)where
rq(di)∈(0,1)represents the degree of the relevance of the
document dito the given query q. The re-ranked document
is the new sequence D′consisting of the elements disorted
in order of increasing relevance score rq(di). The re-ranking
process is popularly executed through neural language mod-
eling methods. However, an emerging research area involves
using graph models such as Graph Neural Networks (GNNs)
for re-ranking. In this scenario, for each pairing of the query
qand a document d∈D, we may attribute a graph represen-
tation G= (V, E)that may be specified in multiple ways.
After a graph representation Gof the data is established,
one may then initialize a randomized feature representation
on the nodes and adjacency matrix . A GNN can then be
applied to this data to obtain a higher level representation
of the nodes, condensed into a single feature vector (oftenarXiv:2503.14802v1  [cs.IR]  19 Mar 2025

Figure 1: The overall re-ranking pipeline. (1) The input query is
passed to the retriever, (2) which then retrieves the top ndocuments.
(3) The query and the documents are processed by the encoder. (4)
The embeddings are passed to the relevance matcher for similarity
calculations (5) which are then utilized for creating edges between
nodes. (6) The graph information is then passed to the re-ranker to
(7) get the final re-ranked list of documents.
mean-pooling) to obtain a uniform representation for the to-
tal graph. This representation is used in the computation of
the relevance score rq(d). An overview of the components of
a graph-based re-ranking pipeline is displayed in Figure 1.
3 Re-ranking Datasets
There are a number of generic Information Retrieval (IR)
datasets that have been well-studied and long term standards
for a wide range of downstream tasks including passage &
document ranking and question-answering [Thakur et al. ,].
This section provides a broad overview of a select subset of
popular examples that are modified for training and evaluat-
ing graph-based re-ranking techniques discussed in the rest of
this paper.
Passage and Document Retrieval
Re-ranking is typically presented as a subtask within generic
passage and document-level retrieval benchmarks [Bajaj et
al., 2016 ].Document retrieval tasks aim to rank a set of doc-
uments based on similarity to a query, while passage retrieval
tasks aim to rank relevant document-parts or pieces of text
(passages) rather than an entire ranked set of documents.
The Microsoft Machine Reading Comprehension (MS
MARCO) dataset is an example of a benchmark that provides
test and evaluation datasets for both document and passage-
level retrieval tasks [Bajaj et al. , 2016 ]. MSMARCO is de-
rived from sampled Bing search queries and corresponding
web pages and is a widely used benchmark for passage and
document ranking. MSMARCO has established itself as a
standard in the re-ranking community. It has been reused
within several community-wide shared tasks such as those
presented Text REtrieval Conference (TREC) [V oorhees and
Harman, 2005 ]. Similar document and passage-level datasets
have been developed for domain-specific retrieval tasks. For
example, CLEF-IP [Piroi et al. , 2011 ]is a domain-specificretrieval dataset that focuses on information retrieval within
the Intellectual Property (IP) domain.
Ranking Algorithms
Ranking algorithms can generally be categorized into Point-
wise, Pairwise, and Listwise approaches [Cao et al. , 2007 ].
The difference among the three categories depend on the
number of documents considered at a time in the loss func-
tion during training. Pointwise approaches consider single
document similarity, Pairwise focus on similarity of a pair of
documents to a query, and lastly Listwise approaches con-
sider a similarity of a list of documents [Caoet al. , 2007 ]. In
the next section, we document current graph-based re-ranking
models, organized in the above categories.
4 Graph Re-ranking Models
The widespread impact of Pre-trained Language Models
(PLM) has also extended to the information retrieval field.
In recent years, IR has greatly benefited from various state-
of-the-art (SOTA) retrieval frameworks. However, graph-
based retrieval remains relatively unexplored in the PLM era.
Therefore, we aim to present a comprehensive overview of
contemporary graph-based retrieval methods, emphasizing
but not limited to PLMs. Table 1 presents a summary of these
approaches, focusing on the key aspects of both the graph-
based and IR approaches. In the table, the Task represents the
retrieval goal (Document/Passage retrieval). Point, Pair, and
List stands for the respective re-ranking strategies, are defined
in Section 3. Doc, Doc-part, and Entity are named based on
the node’s content. Finally, the Relation column indicates
whether the approach directly incorporates inter-document or
inter-passage relationships in the re-ranking process.
4.1 Pointwise Re-ranking
PassageRank [Reed and Madabushi, 2020 ]a graph-based
passage ranking pipeline represents each passage as a node
while the edge is represented by similarity scores between
the nodes in a directed graph. Scores of each node Viis cal-
culated as,
W(Vi) = (1 −d) +dX
Vj∈In(Vi)wji×W(Vj)P
Vk∈Out(Vj)wjk(1)
where In(Vi)is set of vertices pointing toward Vi,Out(Vi)
is set of vertices where Vipoints to, wjiis the weight between
node i and j, and dis the damping factor. Finally, a Bidirec-
tional Encoder Representations from Transformers (BERT)
encoder model re-ranks the documents [Devlin, 2018 ].
Similar to the previous pointwise approach, traditional
two-phased retrievers suffer from recall limitations where the
re-ranker’s performance is subjected to the initially retrieved
documents. To address this, [MacAvaney et al. , 2022b ]pro-
pose Graph Adaptive Re-ranking (GAR), based on the clus-
tering hypothesis [Jardine and van Rijsbergen, 1971 ], which
suggest closely related documents being relevant to a given
query. The authors implement a feedback process to incre-
mentally update the pool of candidate documents with the
neighbors from a corpus graph , a directed graph encoding the

Model Task Point Pair List Doc Doc-part Entity Relation
G-RAG [Dong et al. , 2024a ] D ✓ ✓ Y
GNRR [Di Francesco et al. , 2024 ] D ✓ ✓ Y
GAR [MacAvaney et al. , 2022b ] D ✓ ✓ N
MiM-LiM [Albarede et al. , 2022 ] P ✓ ✓ Y
GCN-reRanker [V ollmers et al. ,] D ✓ ✓ Y
IDRQA [Zhang et al. , 2021 ] D ✓ ✓ Y
KERM [Dong et al. , 2022 ] P ✓ ✓ Y
TRM [Veningston and Shanmugalakshmi, 2014 ] D ✓ ✓ Y
PRP-Graph [Luoet al. , 2024 ] D ✓ ✓ Y
QDG [Frayling et al. , 2024 ] D ✓ ✓ N
KGPR [Fang et al. , 2023 ] P ✓ ✓ Y
SPR-PageRank [Gienapp et al. , 2022 ] D ✓ ✓ Y
GraphMonoT5 [Gupta and Demner-Fushman, 2024 ]D ✓ ✓ N
PassageRank [Reed and Madabushi, 2020 ] D ✓ ✓ N
KG-FiD [Yuet al. , 2021 ] P ✓ ✓ Y
GAR Agent [MacAvaney et al. , 2022a ] D ✓ ✓ N
Fairness-Aware [Jaenich et al. , 2024 ] D ✓ ✓ N
Doc-Cohesion [Sarwar and O’Riordan, 2021 ] D ✓ ✓ Y
SlideGAR [Rathee et al. , 2025 ] D ✓ ✓ Y
GBRM [Deng et al. , 2009 ] D ✓ ✓ Y
Table 1: Comparative Overview of Graph-Based Re-Ranking Techniques. In the header, D = Document, P = Passage, Y = Yes, N = No.
similarity between documents. This allows for the re-ranking
of documents otherwise skipped due to the re-ranking budget.
GAR takes an initial pool of candidate documents R0, a batch
sizeb, a re-ranking budget RB, and a corpus graph as input.
The output of the process is a re-ranked pool R1. The re-
ranking further leverages a dynamically updated re-ranking
poolPand graph frontier F.Pis initialized with the docu-
ment pool and the Fis kept empty. At each iteration, top- b
documents are re-ranked, represented by B. Consequently,
the neighbors of Bare extracted from the corpus graph and
are inserted into F. The process repeats, however, instead
of only scoring top- bdocuments from the initial candidate
pool the process scores documents alternatively from the ini-
tial candidate pool and the frontier. This process continues
until the re-ranking budget permits.
Another work [Jaenich et al. , 2024 ]extends GAR while
focusing on fair exposure of individual document groups in
the candidate list. This experiment leverages groups from an
already labeled dataset. The proposed approach modifies the
original GAR based on two categories of policies: the pre-
retrieval policies concerned with modifying the corpus graph
and the in-process policies that modify the GAR pipeline it-
self. Policy-1 dictates that only documents from different
groups can share an edge in the corpus graph. Policy-2 re-
laxes policy-1 by allowing a limited number of documents
from each group in the corpus graph. Policy-3 modifies the
set of neighbors corresponding to each re-ranked document,
diin a batch before adding them to the frontier. A document
is discarded from the neighbor set if it belongs to the same
group as di. Similar to Policy 2, policy-4 defines a quota
for each group of documents in the frontier. Rather than
only selecting the top-b documents in each batch, policy-5
enforces selecting a specific number of highest-scoring doc-uments from each group. It ensures that the neighbors in the
corpus graph have sufficient diversity. In addition to select-
ing a batch based on the highest-scoring document from each
group, policy-6 further considers the order in which the doc-
uments were inserted into the frontier leading to prioritizing
the documents scored in the first iteration.
4.2 Inter-Document Relationship
While many approaches inspired by the clustering hypothesis
leverage inter-document similarity, they are limited by larger
proportions of irrelevant passages. [Sarwar and O’Riordan,
2021 ]propose a novel graph-based method to model the inter-
passage similarities for re-ranking documents. The model
represents the cohesion of each document by capturing the
topic-shift of passages, where passages related to the same
topic represent high document cohesion. The process starts
with decomposing each document into passages. It then
generates a graph with the passages as nodes and their rel-
evance as edges. The cohesion score is the average simi-
larity of all possible pairs of passages in a document. The
document re-ranking is implemented in one of three set-
tings: i) sim(di, q)×C(di), ii)sim(di, q) +C(di), or iii)
sim(di, q) +C(di)×X, where X∈[0.0−1.0]allows se-
lective inclusion of cohesion, sim(di, q)is query-similarity,
andC(i)is the cohesion score.
The previously discussed methods did not leverage the
structural information in the corpus graphs to update doc-
ument feature representations. To overcome this, Yu et al.
[Yuet al. , 2021 ]propose KG-FiD, a Knowledge Graph (KG)
enhanced FiD that incorporates GNN to leverage the inter-
passage relationship. It introduces a two-stage re-ranking
pipeline. The first stage is concerned with re-ranking the N0
passages returned by the initial retriever, the top N1of which

is leveraged in the second stage. The second stage re-ranks
topN2passages from N1and generates an answer.
Another method of modeling inter-document relationships
for GNN learning is via Abstract Meaning Representation
(AMR) graphs. [Dong et al. , 2024a ]develop G-RAG,
which models the inter-document relationship via an Abstract
Meaning Representation (AMR) graph. The feature represen-
tations of the nodes that appear in the shortest path between
the query and target document are aggregated for contextu-
ally augmenting the document node representation, facilitat-
ing the final re-ranking process.
Graph Neural Re-ranking (GNRR) [Di Francesco et al. ,
2024 ]is another method that addresses document relations
by allowing each query to incorporate document distribution
during the inference process. This approach models the docu-
ment relationship using corpus sub-graphs and encodes these
representations using GNN. GNRR comprises 3 main phases:
i) Data Retrieval, ii) Subgraph Construction, and iii) Fea-
tures and Score Computation. Given a query q and the doc-
ument corpus C, a sparse retrieval fetches top-1000q docu-
ments R0and builds a semantic corpus graph, GCleveraging
TCT-ColBERT [Linet al. , 2020 ]. TCT-ColBERT also en-
codes the query. GCthen facilitate the generation of a query-
induced subgraph, GCq= (Vq, Eq)where, Vq∈R0. There-
fore,GCqretains the lexical information from the sparse re-
triever and the structural information of the corpus graph. The
node representations of GCqare computed using an element-
wise product between the query and each document. While
the document-interactions are modeled by the GNN, a Multi-
layer Perceptron (MLP) independently calculates the rele-
vance of each query-document pair. Leveraging the merged
form of these two representations a score module estimates
the final ranks.
A common practice in passage retrieval and re-ranking
tasks is to leverage contextual information to enhance per-
formance. To this end, [Albarede et al. , 2022 ]proposed a
merge and late interaction model based on Graph Attention
Network (GAT) for passage contextualization. They intro-
duced a novel document graph representation based on inter-
and intra-document similarities.
[Zhang et al. , 2021 ]proposed Iterative Document Re-
ranking (IDR), an integral component of their ODQA frame-
work, that addresses the lexical overlap problem leveraging
document relationship. IDR models the inter-document re-
lationship by constructing an entity-centric document graph
where two documents share an edge based on the common
entities. The graph-based re-ranking module in IDR com-
prises 4 components: i) Contextual Encoding, ii) Graph At-
tention, iii) Multi-document Fusion, and iv) Document Fil-
ter. The process starts with encoding each question q, and
document dkfollowed by the concatenation of the document
representations v. A document graph is then constructed
where the node representation eifor each shared entity Ei
is generated by pooling token embeddings from vasei=
pooling (t(i)
1, t(i)
2, . . . , t(i)
|Ei|); here, t(i)
jis the j-th token in Ei.
A GAT processes the document graph by updating each en-
tity representation. The next stage, multi-document fusion,
updates the non-entity tokens by projecting them over the en-tity tokens to generate v′which is fed into a transformer layer
to obtain the fused representation vector ˜v. Finally, lever-
aging the fused representation a binary classifier scores the
documents.
4.3 External Knowledge Integration
The structural information of knowledge graphs provides an
effective means to capture the relationship between docu-
ments. Motivated by its importance in re-ranking, [V ollmers
et al. ,]applied a Graph Attention Network (GAT) as a cross-
encoder over the proposed query-document knowledge sub-
graph. Within the graph, the nodes represent entities ex-
tracted from a query-document pair, while the edges represent
their semantic relationship. The proposed approach computes
the query-document relevance in 3 steps: entity extraction
and linking, subgraph retrieval, and GAT-based re-ranking.
In the first step, FLAIR [Akbik et al. , 2018 ]is utilized to ex-
tract entities from the documents, while Llama3 is used to
disambiguate and extract entities from the queries as it has
higher accuracy with shorter text. Following this, GENRE
[De Cao et al. , 2021 ], an autoregressive model is used to
link the extracted entities with their corresponding URIs in
a knowledge graph. The next step is subgraph extraction. To
this end, the Subgraph Retrieval Toolkit (SRTK) [Shen, 2023;
Zhang et al. , 2022 ]is used to provide a list of RDF triples
from the target KG against a list of entities. In the final step,
the GAT cross-encoder is used to encode the subgraph where
the output node embedding of the query and the documents
are used to calculate the relevance score.
Cross-encoder is the driving force of many SOTA re-
ranking pipelines. However, by itself, the cross-encoder lacks
background knowledge, a critical element for effective pas-
sage retrieval, especially in domain-specific tasks. Therefore,
[Fang et al. , 2023 ]propose KGPR, a KG-enhanced cross-
encoder for passage re-ranking. At its core KGPR uses LUKE
[Yamada et al. , 2020 ], an entity-aware pre-trained language
model, to incorporate KG to facilitate background informa-
tion. LUKE-based cross-encoder calculates query-passage
relevance utilizing query, passage, and their corresponding
entity embedding. However, by default, the LUKE model dis-
regards the entity relations. Therefore, the proposed method
leverages a knowledge subgraph extracted from the Freebase
[Bollacker et al. , 2008 ]. The subgraph extraction is carried
out in two phases: entity linking and subgraph retrieval. The
former identifies entities from both the query and the passage
to link them with Freebase nodes. To this end, KGPR utilizes
ELQ [Liet al. , 2020 ], an entity-linking model for questions.
The extracted passage entities are then traced up to 1-hop
from the query entities in the Firebase to extract the subgraph
Gq,d, while filtering out edges not related to passage entities.
The authors introduce an additional embedding for relations
in a KG triple. During the re-ranking process, the triple em-
bedding servers as an additional input for LUKE, effectively
infusing background knowledge.
The inherent noise associated with existing KGs makes
them suboptimal for re-ranking tasks. To address this, [Dong
et al. , 2022 ]introduced a re-ranking-centric knowledge meta
graph distillation module with their Knowledge Enhanced
Re-ranking Model (KERM). Additionally, the authors in-

troduced a knowledge injector, an aggregation module to
bridge the semantic gap that emerges during the aggrega-
tion of implicit and explicit knowledge. At first, KERM
employs TransE [Bordes et al. , 2013 ], a knowledge graph
embedding model, to prune noises from a global knowledge
graph, Gto obtain G′. In the pruning process, only the top- π
neighbors of each entity are considered based on the distance
metric, dist(eh, et) = 1 /{E(eh).E(r) +E(eh).E(et) +
E(r).E(et)}where, eh,r, and etare a head entity, relation,
and tail entity of a triplet, respectively. Following this, the bi-
partite meta-graph, Gq,pis constructed in three phases. It be-
gins by selecting the most relevant sentence to the query, fol-
lowed by extracting its entities, and finally linking them with
the top-k hops from G′using Breadth-First Search (BFS). At
this stage, a knowledge injector is utilized to further enhance
the explicit knowledge. A Graph Meta Network (GMN) mod-
ule, a multi-layer GNN integrated into the knowledge injec-
tor, dynamically enriches the meta-graph by utilizing the en-
coded textual information (i.e., implicit knowledge). It leads
to the mutual enhancement of both the text and the knowledge
embedding. This enhanced representation facilitates cross-
encoder’s re-ranking process.
[Gupta and Demner-Fushman, 2024 ]also addresses the
shortcomings of PLM-based re-rankers on domain-specific
tasks. The proposed approach, GraphMonoT5, fuses external
knowledge from KG into PLMs to facilitate biomedical doc-
ument re-ranking. On top of the default encoder-decoder T5,
the GraphMonoT5 complements the encoder with a GNN to
account for the external knowledge modeled in KG. Entities
extracted from each query-document pair are linked to a KG
followed by the subgraph extraction, connecting all the nodes
within 2-hop paths of corresponding entities. The Graph-
MonoT5 introduces a novel interaction module aimed at fus-
ing the text and the graph embeddings leveraging interaction
tokens, tint, interaction nodes, nint, and their correspond-
ing embeddings. The interaction embeddings are merged
into a single representation, which together with the text and
node embedding from a query-document pair, is fed into the
GraphMonoT5 decoder to yield a ranking soccer. The docu-
ments are then re-ranked based on these scores.
[Veningston and Shanmugalakshmi, 2014 ]proposed a doc-
ument re-ranking strategy leveraging a term graph that mod-
els word relations among documents. The term graph, GT
= (VT,ET), is built using frequent item-sets, FSwhich
represents the document with its most frequent terms. In
the graph, VT∈fSwhere fs∈FSis an unique item,
andET⊆VTi×VTjonly if VTiandVTjshare the
same item-set. The edge weight corresponds to the largest
support value, support dof the item-set, FSdcontaining
both the corresponding nodes. Support dis calculated asPn
i=1fSd(ti)/PN
j=1Pn
i=1fdj(ti)where fdj(ti)is the fre-
quency of term tiin document d,nis the number of terms in
item-set, and Nis total number of item-sets. The term graph
is leveraged in document re-ranking following either of the
two approaches: i) Term Rank-based document re-ranking,
or ii) Term Distance matrix-based document re-ranking. The
former is based on the PageRank algorithm [Brin and Page,
1998 ]that computes the term’s rankings. Term ranks, cal-culated via PageRank, are utilized to re-rank the documents
based on the shared terms between the query and the docu-
ments. The latter re-ranking approach leverages the term dis-
tance matrix, an adjacency matrix representing the distance
between any two terms as the least number of hops. A doc-
ument is ranked higher if its terms are at a closer distance to
the terms of a given query.
4.4 Pairwise Re-ranking
So far, we’ve only focused on pointwise re-ranking, which
evaluates each document in isolation. In contrast, pairwise
re-ranking scores ndocuments by evaluating all n×npairs,
capturing subtle nuances during re-ranking. PLM-based pair-
wise re-ranking has demonstrated remarkable performance.
However, the high inference overhead diminishes its feasibil-
ity. To address this, [Gienapp et al. , 2022 ]employs a subset
of pairs sampled from the full set of document pairs. Conse-
quently, the aggregation process that computes final rankings
from the preference probabilities of document pairs, becomes
even more crucial in the proposed approach. One of the
proposed aggregation approaches draws inspiration from the
PageRank algorithm. To this end, the sampled subset of doc-
ument Cis converted into a directed graph where nodes rep-
resent documents while edges represent the preference prob-
abilities. For a given pair of documents (di, dj), the prefer-
ence probability, pijrepresents the likelihood of dioutrank-
ingdj. The re-ranking is facilitated by the PageRank algo-
rithm adapted for weighted edges. In the main re-ranking
pipeline, the proposed approach utilizes a mono-duo archi-
tecture. Initially, monoT5, a pointwise re-ranker, ranks the
top-1000 documents. This is followed by the duoT5, pairwise
re-ranker, which re-ranks the top-50 results from the previous
step.
Pairwise Ranking Prompting (PRP) is a zero-shot re-
ranking method based on Large Language Models (LLM).
However, contemporary PRP methods do not account for the
uncertainty associated with the labels. To overcome this limi-
tation, [Luoet al. , 2024 ]proposed a PRP-graph coupled with
a novel scoring PRP unit. The PRP-graph functions in two
distinct phases. First, it generates a ranking graph discussed
in section 5. Inspired by the weighted PageRank algorithm,
the vertices of the document graph are iteratively updated to
reflect the final re-ranking score.
[Deng et al. , 2009 ]present a unique approach to re-rank
documents. The documents in the corpus are initially ranked
using p(q|θd) =Q
t∈qp(t|θd)n(t,q), where p(q|θd)is
the maximum likelihood estimation of the term tin a docu-
ment d, and n(t, q)is the number of times that term toccurs
in query q. A document is ranked higher if it has a higher
likelihood of generating the query. After a latent space graph
is created(explained in Section 5), the following cost function
is optimized to re-rank the documents.
R(F, q, G ) =1
2nX
i,j=1wijf(di, q)√Dii−f(dj, q)p
Djj2
+µnX
i=1f(di, q)−f0(di, q)2,(2)

The first sum1
2Pn
i,j=1wijf(di,q)√Dii−f(dj,q)√
Djj2
ensures
that documents with high similarity should have simi-
lar ranking scores. This is the global consistency term,
where the cost increases if similar documents end up
with very different ranking scores. The second sum
µPn
i=1f(di, q)−f0(di, q)2ensures that the refined rank-
ing scores stay close to the initial ranking scores. This term
prevents the re-ranking from deviating too much from the
original scores.
4.5 Listwise Re-ranking
[Rathee et al. , 2025 ]extends GAR [MacAvaney et al. , 2022b ]
to the LLM-based listwise re-ranker. The existing GAR, re-
lying on the Probability Ranking Principle, does not con-
sider the document relationship while calculating the rele-
vance score. Therefore, the proposed approach SlideGAR
adapts the GAR from pointwise re-ranking to listwise set-
ting to account for document relationship. After retrieving
an initial ranked list of documents using BM25, SlideGar ap-
plies a sliding window approach with window size wand step
sizeb. The top wdocuments from the initial ranking are
selected, and, similar to GAR, the graph frontier is updated
with their neighboring nodes. However, unlike GAR, which
ranks documents individually, SlideGar ranks batches of doc-
uments simultaneously using a listwise LLM re-ranker such
as RankZephyr [Pradeep et al. , 2023 ]. The ranking process
alternates between the initial ranked list and the graph fron-
tier, ensuring a more adaptive and context-aware document
retrieval strategy.
5 Graph Construction for Graph-based
Reranking
Graph-based re-ranking methods enhance retrieval perfor-
mance by incorporating document relationships, surpass-
ing traditional ranking models. The re-ranking models dis-
cussed thus far rely on graph data structures, which can
be broadly categorized into document-level and entity-level
graphs. In document-level graphs, edges represent relation-
ships between documents/document parts, whereas entity-
level graphs establish connections between individual tokens
or concepts. This section explores some of the notable strate-
gies used to construct both types of graphs, detailing their
formation, edge weighting, and relevance to re-ranking tasks.
Definition
Let G = (V , E, A, R, Adj) be a directed/undirected graph,
where V is the set of nodes and E⊆V×Vis the set of edges.
The number of nodes in G is denoted by |V|=N, and the
number of edges by |E|. A is the set of node types while R
denotes the set of edge types. The set of edges E can also be
expressed as the adjacency matrix Adj∈ {0,1}N×N, where
Adjuv= 1 if nodes (u, v)∈Eare connected, and Adjuv= 0
otherwise.Document-level graphs
In[Di Francesco et al. , 2024; MacAvaney et al. , 2022b ]each
pair of documents ( di,dj) in the graph corpus shares a con-
nection based on the cosine similarity between their docu-
ment encodings. After the graph structure for each query is
established, the features of the nodes are defined by perform-
ing an element-wise product between the query representa-
tion and each document representation. Therefore the node
feature for document iis given by xi=zq⊙zdi,zqand
zdibeing the query and document encodings respectively.
[Zhang et al. , 2021 ]utilizes the Named Entity Recognition
system to extract entities and connect documents if they have
shared entities.
Instead of merely linking nodes based on document simi-
larity, [Albarede et al. , 2022 ]introduces a more structured ap-
proach by dividing document nodes into section nodes (non-
textual units with titles) and passage nodes (textual units
without titles). Additionally, the framework defines eight dis-
tinct edge types, consisting of four primary relations and their
respective inverses: (1) Order relation – captures the sequen-
tial arrangement between passage nodes. (2) Structural rela-
tion – represents connections between a passage and its parent
section node or between two section nodes. (3) Internal rela-
tion – links nodes within the same document. (4) External
relation – connects nodes across different documents. This
hierarchical and relational structure enhances the granularity
and contextual understanding of document graphs.
G-RAG [Dong et al. , 2024a ]considers each document to
be a text block of 100 words. Each question is concatenated
with documents in the corpus. AMRBART [Baiet al. , 2022 ]
is utilized to create AMR graphs [Banarescu et al. , 2013 ].
From the graphs, the connection information between differ-
ent documents is incorporated into the edge features in the
subsequent document-level graph. Thus an undirected docu-
ment graph Gq={V , E}based on AMRs {Gqp1, · · · , Gqpn}
is established. Each node vi∈Vcorresponds to the doc-
ument pi. For vi, vj∈Vwhere i̸=jif AMRs Gqpiand
Gqpjhave common nodes, there will be an undirected edge
between viandvjin the document-level graph.
[Deng et al. , 2009 ]incorporates link information in a
document-level latent space graph. The content matrix term
C∈Rn×mis a sparse matrix whose rows represent doc-
uments and columns represent terms, where mis the num-
ber of terms. The document-author bipartite graph can sim-
ilarly be described by a matrix A∈Rn×l, which is also
a sparse matrix whose rows correspond to documents and
whose columns correspond to authors, where lis the num-
ber of authors. These matrices are mapped to a shared latent
space Xthrough joint factorization [Zhu et al. , 2007 ], thus
combining the authorship and content information. The edge
weights are determined using a heat kernel as follows
wij= exp
−∥xi−xj∥2
2σ2
(3)
where xiandxjare the latent representations of documents
iandjandσis the heat kernel that controls the spread or
reach of the edges between nodes in the graph. Specifically,
it affects how sensitive the weight of the edge wijis to the

Euclidean distance between two nodes. Smaller σwill corre-
spond to far apart nodes having edge weights closer to zero
and vice versa.
[Luo et al. , 2024 ]investigates the application of Pair-
wise Ranking Prompts [Qinet al. , 2023 ]for graph construc-
tion. A document diand its closest subsequent document
djare selected and a bidirectional relationship is established
through directed edges di→janddj→i. The edge weights
are assigned based on PRP-derived preference scores si→j
andsj→i, quantifying the relative ranking between the docu-
ments. scores are iteratively updated over r rounds using the
following update rules
Sr
i=Sr−1
i+sj→i×Sr−1
j
r
Sr
j=Sr−1
j+si→j×Sr−1
i
r(4)
After r rounds, the final ranking graph G is obtained, where
node scores reflect the cumulative ranking adjustments over
multiple iterations.
Entity-level graphs
[Deng et al. , 2009 ]propose a two-stage process for graph
construction. The process begins with node embedding ini-
tialization using TransE [Bordes et al. , 2013 ]. The pairwise
distances between embeddings are computed, and only the
top n closest nodes are retained while the rest are pruned.
Next, relevant sentence retrieval is performed by measuring
the similarity between Word2Vec [Mikolov et al. , 2013 ]em-
beddings of the query and sentences from the corpus. Com-
mon entities between the query and retrieved sentences are
identified, and a Breadth-First Search (BFS) is conducted to
locate all nodes within K-hops of these entities. The result-
ing nodes are then used to construct a meta graph, capturing
the refined structure of the knowledge graph for downstream
tasks. [Gupta and Demner-Fushman, 2024; Yu et al. , 2021 ]
follow a similar approach for knowledge graph entity align-
ment and subgraph creation. [Fang et al. , 2023 ]also retrieves
entities from a knowledge graph, selectively retaining only
the edges that connect entities appearing in either the query
or the passage.
6 Conclusion and Summary of Findings
6.1 Limitations and Discussion
Graph-based retrieval is an established research area that has
produced a wide variety of datasets used primarily for Knowl-
edge Graph Question Answering (KGQA) downstream tasks.
KGQA datasets are currently designed for one/multi-hop en-
tity and relationship classification tasks. Current datasets are
not well suited for more complex tasks such as passage and
document ranking/re-ranking . As a result, there has been an
emergence of curated graph construction methodologies and
datasets used for graph-based passage and document ranking
tasks (Section 5).
The diversity of these methods leads to an incongruity
across evaluation techniques that aim to measure the perfor-
mance of current methods. This scenario differs from estab-
lished benchmark standards developed for language modelretrievers. Examples of established benchmark datasets used
to evaluate language model-based ranking tasks are provided
in Section 3. These datasets were derived from large collec-
tions of unstructured data and preprocessed into query and
target pairs. Unlike language models, Graph-based tech-
niques such as GNNs additionally require a transformation of
unstructured text samples into adjacency matrices of nodes
and edges that can be used for training (See Section 2). Sec-
tion 4 outlines the range of methods demonstrated for first
curating the adjacency matrix and additionally, incorporating
ranking-based features such as node/edge similarity scores
into the graph.
The design and quality of the input representation can
drastically change architectural considerations for proposed
graph-based ranking models, ultimately impacting down-
stream ranking performance. Despite ongoing progress in
this area, a standard benchmark to measure performance has
not yet been developed to evaluate graph-based passage and
document ranking tasks. For each of the methods presented
in Section 4, downstream performance is evaluated on estab-
lished benchmark datasets like MSMARCO, originally de-
veloped specifically for language model based ranking tasks.
In traditional settings, datasets such as MSMARCO are typ-
ically used for both train and test workloads, with separate
tracks for further data augmentation and/or zero-shot perfor-
mance criteria.
6.2 Future Work
In graph-based settings, we have observed that authors derive
their own methods for (1) generating an adjacency matrix for
the train set, (2) potentially augmenting the train set, with ad-
ditional external data, and lastly, (3) performing evaluation
on the downstream ranking task. We also have documented
settings in which the distribution of the training input graph
is either only minimally overlapping, or differs completely
than that of a particular downstream task. In this case, eval-
uation assumes a transfer learning process, and is based on
zero-shot performance. While relying on downstream perfor-
mance provides a reference for comparing graph-based rank-
ing models, the community should move towards creating a
static benchmark that addresses limitations such as evaluating
model architectures and the data curation process, as well as
reproducibility.
References
[Akbik et al. , 2018 ]Alan Akbik, Duncan Blythe, and
Roland V ollgraf. Contextual string embeddings for se-
quence labeling. In COLING 2018, 27th International
Conference on Computational Linguistics , pages 1638–
1649, 2018.
[Albarede et al. , 2022 ]Lucas Albarede, Philippe Mulhem,
Lorraine Goeuriot, Claude Le Pape-Gardeux, Sylvain
Marie, and Trinidad Chardin-Segui. Passage retrieval on
structured documents using graph attention networks. In
European Conference on Information Retrieval , pages 13–
21. Springer, 2022.

[Baiet al. , 2022 ]Xuefeng Bai, Yulong Chen, and Yue
Zhang. Graph pre-training for amr parsing and generation.
arXiv preprint arXiv:2203.07836 , 2022.
[Bajaj et al. , 2016 ]Payal Bajaj, Daniel Campos, Nick
Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Ran-
gan Majumder, Andrew McNamara, Bhaskar Mitra, Tri
Nguyen, et al. Ms marco: A human generated machine
reading comprehension dataset. arXiv e-prints , pages
arXiv–1611, 2016.
[Banarescu et al. , 2013 ]Laura Banarescu, Claire Bonial,
Shu Cai, Madalina Georgescu, Kira Griffitt, Ulf Herm-
jakob, Kevin Knight, Philipp Koehn, Martha Palmer, and
Nathan Schneider. Abstract meaning representation for
sembanking. In Proceedings of the 7th linguistic annota-
tion workshop and interoperability with discourse , pages
178–186, 2013.
[Bollacker et al. , 2008 ]Kurt Bollacker, Colin Evans,
Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase:
a collaboratively created graph database for structuring
human knowledge. In Proceedings of the 2008 ACM
SIGMOD international conference on Management of
data, pages 1247–1250, 2008.
[Bordes et al. , 2013 ]Antoine Bordes, Nicolas Usunier,
Alberto Garcia-Duran, Jason Weston, and Oksana
Yakhnenko. Translating embeddings for modeling
multi-relational data. Advances in neural information
processing systems , 26, 2013.
[Brin and Page, 1998 ]Sergey Brin and Lawrence Page. The
anatomy of a large-scale hypertextual web search engine.
Computer networks and ISDN systems , 30(1-7):107–117,
1998.
[Caoet al. , 2007 ]Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-
Feng Tsai, and Hang Li. Learning to rank: from pairwise
approach to listwise approach. In Proceedings of the 24th
international conference on Machine learning , pages 129–
136, 2007.
[De Cao et al. , 2021 ]Nicola De Cao, Gautier Izacard, Se-
bastian Riedel, and Fabio Petroni. Autoregressive entity
retrieval. In 9th International Conference on Learning
Representations, ICLR 2021, Virtual Event, Austria, May
3-7, 2021 . OpenReview.net, 2021.
[Deng et al. , 2009 ]Hongbo Deng, Michael R Lyu, and Irwin
King. Effective latent space graph-based re-ranking model
with global consistency. In Proceedings of the second acm
international conference on web search and data mining ,
pages 212–221, 2009.
[Devlin, 2018 ]Jacob Devlin. Bert: Pre-training of deep bidi-
rectional transformers for language understanding. arXiv
preprint arXiv:1810.04805 , 2018.
[Di Francesco et al. , 2024 ]Andrea Giuseppe Di Francesco,
Christian Giannetti, Nicola Tonellotto, and Fabrizio Sil-
vestri. Graph neural re-ranking via corpus graph. arXiv
preprint arXiv:2406.11720 , 2024.
[Dong et al. , 2022 ]Qian Dong, Yiding Liu, Suqi Cheng,
Shuaiqiang Wang, Zhicong Cheng, Shuzi Niu, and DaweiYin. Incorporating explicit knowledge in pre-trained lan-
guage models for passage re-ranking. In Proceedings
of the 45th International ACM SIGIR Conference on Re-
search and Development in Information Retrieval , pages
1490–1501, 2022.
[Dong et al. , 2024a ]Jialin Dong, Bahare Fatemi, Bryan Per-
ozzi, Lin F Yang, and Anton Tsitsulin. Don’t forget to
connect! improving rag with graph-based reranking. arXiv
preprint arXiv:2405.18414 , 2024.
[Dong et al. , 2024b ]Yuxin Dong, Shuo Wang, Hongye
Zheng, Jiajing Chen, Zhenhong Zhang, and Chihang
Wang. Advanced rag models with graph structures: Opti-
mizing complex knowledge reasoning and text generation.
In2024 5th International Symposium on Computer Engi-
neering and Intelligent Communications (ISCEIC) , pages
626–630. IEEE, 2024.
[Fang et al. , 2023 ]Jinyuan Fang, Zaiqiao Meng, and Craig
Macdonald. Kgpr: Knowledge graph enhanced passage
ranking. In Proceedings of the 32nd ACM International
Conference on Information and Knowledge Management ,
pages 3880–3885, 2023.
[Frayling et al. , 2024 ]Erlend Frayling, Sean MacAvaney,
Craig Macdonald, and Iadh Ounis. Effective adhoc re-
trieval through traversal of a query-document graph. In
European Conference on Information Retrieval , pages 89–
104. Springer, 2024.
[Gienapp et al. , 2022 ]Lukas Gienapp, Maik Fr ¨obe, Matthias
Hagen, and Martin Potthast. Sparse pairwise re-ranking
with pre-trained transformers. In Proceedings of the 2022
ACM SIGIR International Conference on Theory of Infor-
mation Retrieval , pages 72–80, 2022.
[Glass et al. , 2022 ]Michael Glass, Gaetano Rossiello,
Md Faisal Mahbub Chowdhury, Ankita Rajaram Naik,
Pengshan Cai, and Alfio Gliozzo. Re2g: Retrieve, rerank,
generate. In Annual Conference of the North American
Chapter of the Association for Computational Linguistics ,
2022.
[Gupta and Demner-Fushman, 2024 ]Deepak Gupta and
Dina Demner-Fushman. Empowering language model
with guided knowledge fusion for biomedical document
re-ranking. In International Conference on Artificial
Intelligence in Medicine , pages 251–260. Springer, 2024.
[Jaenich et al. , 2024 ]Thomas Jaenich, Graham McDonald,
and Iadh Ounis. Fairness-aware exposure allocation via
adaptive reranking. In Proceedings of the 47th Interna-
tional ACM SIGIR Conference on Research and Develop-
ment in Information Retrieval , pages 1504–1513, 2024.
[Jardine and van Rijsbergen, 1971 ]Nick Jardine and Cor-
nelis Joost van Rijsbergen. The use of hierarchic clustering
in information retrieval. Information storage and retrieval ,
7(5):217–240, 1971.
[Lewis et al. , 2020 ]Patrick Lewis, Ethan Perez, Aleksan-
dra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih, Tim
Rockt ¨aschel, et al. Retrieval-augmented generation for

knowledge-intensive nlp tasks. Advances in Neural Infor-
mation Processing Systems , 33:9459–9474, 2020.
[Liet al. , 2020 ]Belinda Z Li, Sewon Min, Srinivasan Iyer,
Yashar Mehdad, and Wen-tau Yih. Efficient one-pass
end-to-end entity linking for questions. arXiv preprint
arXiv:2010.02413 , 2020.
[Linet al. , 2020 ]Sheng-Chieh Lin, Jheng-Hong Yang, and
Jimmy Lin. Distilling dense representations for rank-
ing using tightly-coupled teachers. arXiv preprint
arXiv:2010.11386 , 2020.
[Luoet al. , 2024 ]Jian Luo, Xuanang Chen, Ben He, and
Le Sun. Prp-graph: Pairwise ranking prompting to llms
with graph aggregation for effective text re-ranking. In
Proceedings of the 62nd Annual Meeting of the Associa-
tion for Computational Linguistics (Volume 1: Long Pa-
pers) , pages 5766–5776, 2024.
[MacAvaney et al. , 2022a ]Sean MacAvaney, Nicola Tonel-
lotto, and Craig Macdonald. Adaptive re-ranking as an
information-seeking agent. 2022.
[MacAvaney et al. , 2022b ]Sean MacAvaney, Nicola Tonel-
lotto, and Craig Macdonald. Adaptive re-ranking with a
corpus graph. In Proceedings of the 31st ACM Interna-
tional Conference on Information & Knowledge Manage-
ment , pages 1491–1500, 2022.
[Mikolov et al. , 2013 ]Tom´aˇs Mikolov, Wen-tau Yih, and
Geoffrey Zweig. Linguistic regularities in continuous
space word representations. In Proceedings of the 2013
conference of the north american chapter of the associa-
tion for computational linguistics: Human language tech-
nologies , pages 746–751, 2013.
[Peng et al. , 2024 ]Boci Peng, Yun Zhu, Yongchao Liu, Xi-
aohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and
Siliang Tang. Graph retrieval-augmented generation: A
survey. arXiv e-prints , pages arXiv–2408, 2024.
[Piroi et al. , 2011 ]Florina Piroi, Mihai Lupu, Allan Han-
bury, and Veronika Zenz. Clef-ip 2011: Retrieval in the
intellectual property domain. In CLEF (notebook pa-
pers/labs/workshop) , 2011.
[Pradeep et al. , 2023 ]Ronak Pradeep, Sahel Sharify-
moghaddam, and Jimmy Lin. Rankzephyr: Effective and
robust zero-shot listwise reranking is a breeze! arXiv
preprint arXiv:2312.02724 , 2023.
[Qinet al. , 2023 ]Zhen Qin, Rolf Jagerman, Kai Hui, Hon-
glei Zhuang, Junru Wu, Le Yan, Jiaming Shen, Tianqi Liu,
Jialu Liu, Donald Metzler, et al. Large language models
are effective text rankers with pairwise ranking prompting.
arXiv preprint arXiv:2306.17563 , 2023.
[Rathee et al. , 2025 ]Mandeep Rathee, Sean MacAvaney,
and Avishek Anand. Guiding retrieval using llm-based
listwise rankers. arXiv preprint arXiv:2501.09186 , 2025.
[Reed and Madabushi, 2020 ]Kyle Reed and Harish Tayyar
Madabushi. Faster bert-based re-ranking through candi-
date passage extraction. In TREC , 2020.[Sarwar and O’Riordan, 2021 ]Ghulam Sarwar and Colm
O’Riordan. A graph-based approach at passage level to in-
vestigate the cohesiveness of documents. In DATA , pages
115–123, 2021.
[Shen, 2023 ]Yuanchun Shen. Srtk: A toolkit for
semantic-relevant subgraph retrieval. arXiv preprint
arXiv:2305.04101 , 2023.
[Thakur et al. ,]Nandan Thakur, Nils Reimers, Andreas
R¨uckl´e, Abhishek Srivastava, and Iryna Gurevych. Beir:
A heterogeneous benchmark for zero-shot evaluation of
information retrieval models.
[Veningston and Shanmugalakshmi, 2014 ]K Veningston
and R Shanmugalakshmi. Information retrieval by
document re-ranking using term association graph. In
Proceedings of the 2014 international conference on
interdisciplinary advances in applied computing , pages
1–8, 2014.
[V ollmers et al. ,]Daniel V ollmers, Manzoor Ali, Hamada M
Zahera, and Axel-Cyrille Ngonga Ngomo. Document
reranking using gat-cross encoder.
[V oorhees and Harman, 2005 ]Ellen M V oorhees and
Donna K Harman. The text retrieval conference. TREC:
Experiment and evaluation in information retrieval , pages
3–19, 2005.
[Yamada et al. , 2020 ]Ikuya Yamada, Akari Asai, Hiroyuki
Shindo, Hideaki Takeda, and Yuji Matsumoto. Luke:
Deep contextualized entity representations with entity-
aware self-attention. arXiv preprint arXiv:2010.01057 ,
2020.
[Yuet al. , 2021 ]Donghan Yu, Chenguang Zhu, Yuwei Fang,
Wenhao Yu, Shuohang Wang, Yichong Xu, Xiang Ren,
Yiming Yang, and Michael Zeng. Kg-fid: Infusing knowl-
edge graph in fusion-in-decoder for open-domain question
answering. arXiv preprint arXiv:2110.04330 , 2021.
[Zhang et al. , 2021 ]Yuyu Zhang, Ping Nie, Arun Rama-
murthy, and Le Song. Answering any-hop open-domain
questions with iterative document reranking. In Proceed-
ings of the 44th International ACM SIGIR Conference
on Research and Development in Information Retrieval ,
pages 481–490, 2021.
[Zhang et al. , 2022 ]Jing Zhang, Xiaokang Zhang, Jifan Yu,
Jian Tang, Jie Tang, Cuiping Li, and Hong Chen. Subgraph
retrieval enhanced model for multi-hop knowledge base
question answering. In Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers) , pages 5773–5784, 2022.
[Zhuet al. , 2007 ]Shenghuo Zhu, Kai Yu, Yun Chi, and Yi-
hong Gong. Combining content and link for classifica-
tion using matrix factorization. In Proceedings of the 30th
annual international ACM SIGIR conference on Research
and development in information retrieval , pages 487–494,
2007.