# ZeroRAG: Adaptive Semantic Peer Routing for Decentralized Retrieval-Augmented Generation

**Authors**: Petru Neague, Quinten Stokkink, Martijn de Vos, Rafael Pires, Mathis Randl, Anne-Marie Kermarrec, Johan Pouwelse

**Published**: 2026-11-05 12:33:45

**PDF URL**: [https://arxiv.org/pdf/2611.07928v1](https://arxiv.org/pdf/2611.07928v1)

## Abstract
Retrieval-augmented generation (RAG) is a standard approach to enhance the reliability of large language model (LLM) outputs by retrieving documents relevant to a query and supplying them in the LLM prompt. Today's RAG systems collect, index, and serve these documents from centralized stores. In practice, however, documents are owned by organizations, devices, and individuals who cannot or will not surrender custody of them to a central party, e.g., because of privacy, copyright or regulatory compliance. Existing modern decentralized retrieval techniques either reintroduce a coordinating authority, or rely on a global data structure that requires continuous network-wide coordination (e.g., DHTs). We present ZeroRAG, a decentralized routing layer for document retrieval in RAG that performs semantic query routing across an overlay of peers with- out global data structures or centralized orchestration. Documents remain with the peers that own them, and each peer keeps only lightweight local state: a bounded set of neighbor advertisements and a cache of recently routed queries. Using these, ZeroRAG steers each query through a best-first walk that progressively rewires the overlay toward semantically relevant neighbors, so recall improves over time as the network self-organizes. We evaluate ZeroRAG using three real-world workloads and two decentralized baselines. We find that ZeroRAG outperforms baselines in terms of recall and required network messages, approaches the recall and LLM answer accuracy of a centralized document store, self-heals after catastrophic peer failures, and scales to 100 000 peers.

## Full Text


<!-- PDF content starts -->

ZeroRAG: Adaptive Semantic Peer Routing for
Decentralized Retrieval-Augmented Generation
Petru Neague
Delft University of Technology
Delft, The Netherlands
p.m.neague@tudelft.nlQuinten Stokkink
Delft University of Technology
Delft, The Netherlands
q.a.stokkink@tudelft.nlMartijn de Vos
EPFL
Lausanne, Switzerland
martijn.devos@epfl.ch
Rafael Pires
EPFL
Lausanne, Switzerland
rafael.pires@epfl.chMathis Randl
EPFL
Lausanne, Switzerland
mathis.randl@epfl.chAnne-Marie Kermarrec
EPFL
Lausanne, Switzerland
anne-marie.kermarrec@epfl.ch
Johan Pouwelse
Delft University of Technology
Delft, The Netherlands
j.a.pouwelse@tudelft.nl
Abstract
Retrieval-augmented generation ( RAG ) is a standard approach to
enhance the reliability of large language model ( LLM ) outputs by
retrieving documents relevant to a query and supplying them in the
LLM prompt. Today’s RAG systems collect, index, and serve these
documents from centralized stores. In practice, however, documents
are owned by organizations, devices, and individuals who cannot or
will not surrender custody of them to a central party,e.g., because
of privacy, copyright or regulatory compliance. Existing modern
decentralized retrieval techniques either reintroduce a coordinating
authority, or rely on a global data structure that requires continu-
ous network-wide coordination (e.g., DHTs). We presentZeroRAG,
a decentralized routing layer for document retrieval in RAG that
performs semantic query routing across an overlay of peers with-
out global data structures or centralized orchestration. Documents
remain with the peers that own them, and each peer keeps only
lightweight local state: a bounded set of neighbor advertisements
and a cache of recently routed queries. Using these,ZeroRAGsteers
each query through a best-first walk that progressively rewires the
overlay toward semantically relevant neighbors, so recall improves
over time as the network self-organizes. We evaluateZeroRAG
using three real-world workloads and two decentralized baselines.
We find thatZeroRAGoutperforms baselines in terms of recall
and required network messages, approaches the recall and LLM
answer accuracy of a centralized document store, self-heals after
catastrophic peer failures, and scales to100 000peers.
CCS Concepts
•Information systems →Peer-to-peer retrieval;Top-k re-
trieval in databases;•Networks→Peer-to-peer networks.
This work is licensed under a Creative Commons Attribution 4.0 International License.
Middleware ’26, Tarragona, Spain
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2621-7/2026/12
https://doi.org/10.1145/3801927.3858412Keywords
Decentralized retrieval, Retrieval-augmented generation (RAG),
Approximate nearest neighbor search
ACM Reference Format:
Petru Neague, Quinten Stokkink, Martijn de Vos, Rafael Pires, Mathis Randl,
Anne-Marie Kermarrec, and Johan Pouwelse. 2026. ZeroRAG: Adaptive
Semantic Peer Routing for Decentralized Retrieval-Augmented Generation.
In27th International Middleware Conference (Middleware ’26), December
14–18, 2026, Tarragona, Spain.ACM, New York, NY, USA, 14 pages. https:
//doi.org/10.1145/3801927.3858412
1 Introduction
Large language models ( LLM s) have transformed natural language
processing because of their strong capabilities across tasks such
as text generation, translation, and summarization [ 10,56]. One of
the most popular applications of LLM s is question answering [ 64].
In this domain, a fundamental challenge is to ensure the reliability
ofLLM -generated answers. However, LLM s are prone to hallucina-
tions where they confidently generate false or misleading informa-
tion [ 4]. This makes it difficult to trust the outputs of LLM s without
double-checking factuality against other sources of truth [ 27]. These
reliability concerns are particularly problematic in high-stakes set-
tings, such as healthcare, finance, or law, where inconsistent or
fabricated answers can have serious real-world consequences [ 24].
RAG is a popular approach to improving LLM reliability without
retraining or fine-tuning the model [ 14,28]. Rather than relying
solely on parametric knowledge embedded in the model itself, RAG
first retrieves relevant documents from an external source based
on the user’s query, then includes them in the prompt before gen-
erating a response. Both queries and documents are typically repre-
sented as high-dimensional embedding vectors capturing semantic
meaning, and retrieval involves finding the embeddings nearest to
the query, a process known as nearest neighbor search ( NNS ) [42].
This allows the LLM to ground its answers in up-to-date or domain-
specific information, improving factual accuracy and consistency.
arXiv:2611.07928v1  [cs.DC]  5 Nov 2026

Middleware ’26, December 14–18, 2026, Tarragona, Spain Neague, et al.
Existing RAG systems typically assume that documents are col-
lected, indexed, and served from a centralized store under unified
administrative control [ 14,28,62]. While this simplifies the RAG
pipeline and results in low latency for document retrieval, it is not
always a viable or desirable architecture. Data may be inherently
distributed across organizations, devices, or individuals who can-
not or will not surrender custody of their documents to a central
party, whether for reasons of privacy, regulatory compliance, data
ownership, or operational boundaries [ 19,26]. A hospital network,
for instance, cannot pool patient records into a shared index with-
out additional access control [ 46,55]. Similarly, a federation of
research institutions may each hold proprietary datasets they are
unwilling to freely share with other parties. Beyond ownership con-
cerns, centralizing large, heterogeneous, and continuously evolving
document collections introduces infrastructure costs and creates
a single point of failure [ 17,60,61]. Despite these limitations, no
general-purpose decentralized retrieval infrastructure exists for
RAG : existing systems either assume centralized control, or dis-
tribute retrieval only within tightly managed federated or cloud
settings that still require some coordinating authority.
We presentZeroRAG, a decentralized routing layer that en-
ables document retrieval by routing queries across a decentralized
overlay, without requiring global data structures or centralized or-
chestration. Documents remain with the peers that own them and
peers themselves are in control whether they want to share doc-
uments with the party requesting them. Retrieval withZeroRAG
only uses local coordination. Each peer in ZeroRAG maintains only
lightweight local state: compact summaries of the content exposed
by neighboring peers and a bounded cache of previously-routed
queries, and uses these to guide incoming queries to peers likely to
hold documents relevant to that query. This is a key departure from
prior decentralized nearest-neighbor systems, which route based
only on what peershold:ZeroRAGadditionally exploits what peers
have recentlyasked, referring query issuers toward peers whose
recent query history is semantically aligned, not only their local
inventory. For example, REMINDIN’ [ 54] only learns from past re-
sponses and INGA [ 33] routes toward previous query issuers using
explicit query topics. This design leaves owners in control of which
documents they share, while peers route queries using document
and query embeddings and peer referrals.
The key technical challengeZeroRAGsolves is effective routing
and overlay management without global knowledge or coordina-
tion. Rather than consulting a centralized directory or broadcasting
queries to all peers, a query inZeroRAGinvolves a best-first walk
over the overlay that adapts as queries flow through the network
while also progressively rewiring peer connections toward seman-
tically relevant neighbors. The result is a retrieval layer for RAG
whose recall improves over time as the overlay organizes, scales
gracefully with the number of peers and documents in terms of
communication and computation overhead, and remains resilient
to peer churn, without requiring network-wide maintenance.
We implementZeroRAGand evaluate its efficiency and effec-
tiveness for decentralized document retrieval in RAG pipelines,
using three real-world workloads (AOL-IA[ 36,43],Amazon Elec-
tronics[ 3], andPubMedQA[ 25]) and two decentralized base-
lines (Semantica[ 41] andDHT-LSH). Our experimental evaluationdemonstrates thatZeroRAGconsistently outperforms other decen-
tralized baselines while requiring no globally maintained structure.
On theAOL-IAworkload,ZeroRAGachieves this at one tenth
of the message overhead of the strongest baseline (DHT-LSH).Ze-
roRAGapproaches the recall and end-to-end RAG answer accuracy
of a centralized document store, without any global index, and 99%
of queries locate the best supporting document in at most 13 peer
contacts. Beyond static performance, our ablation studies show
thatZeroRAGself-heals after catastrophic failure in which 50% of
all peers crash simultaneously, quickly adapts as new topics are
introduced, and scales to networks of up to one hundred thousand
peers and documents. We make the following contributions:
•We introduceZeroRAG, a decentralized routing layer for
RAG that enables document retrieval by routing queries
across a decentralized overlay without labeled data, shared
data structures, or centralized orchestration (Section 3).
•We evaluate three real-world datasets, comparing against
centralized and decentralized baselines on recall, RAG an-
swer accuracy, message efficiency, resilience to peer churn,
and topic introduction (Section 4). We show thatZeroRAG
matches or exceeds the performance of decentralized alter-
natives while scaling gracefully to large peer populations.
2 Background and Problem Description
Our aim is to construct a decentralized document retrieval layer
forRAG pipelines. We first describe the standard RAG workflow
and then discuss the technical challenges that this work addresses.
2.1 Centralized RAG Workflows
Retrieval-augmented generation ( RAG ) is a standard and popular
approach to improve the factual reliability of LLM outputs by sup-
plying external context relevant to the user query directly in the
prompt [ 28]. Whereas conventional language models depend en-
tirely on parametric knowledge encoded during pre-training, RAG
consults an external data source at inference time, yielding more
accurate, up-to-date responses and reducing hallucinations without
requiring costly retraining or finetuning. We illustrate the RAG
workflow in Figure 1, using an example where a user asks a medical
question, and explain the individual operations next.
The pipeline operates in two phases. In the offline phase, doc-
uments, videos, or other source material are preprocessed: each
item is divided into chunks, and every chunk is encoded into a
high-dimensional vector by an embedding model [ 45]➊, with the
resulting embeddings stored in a vector database ➋. For presen-
tation clarity, we will refer to chunks as documents throughout
this work. In the online phase, a user query ➌is embedded and
forwarded to the retriever ➍, which locates the most semantically
similar stored vectors ➎and fetches the corresponding documents
from the vector database ➏. The original query and the retrieved
context are then jointly passed to the LLM , which produces a re-
sponse grounded in that evidence ➐, yielding a more accurate and
context-aware answer than generation without retrieval.
The vector database is the central component enabling retrieval.
Rather than matching exact keywords, it uses similarity search over
high-dimensional embedding spaces. Prior to insertion, each doc-
ument is encoded by a pre-trained machine learning ( ML) model,

ZeroRAG Middleware ’26, December 14–18, 2026, Tarragona, Spain
Documents
(e.g.,
articles,
notes, ...)ACE inhibitors reduce
mortality after [...]
In a randomized trial
of 2006 post-MI [...]
[...]...Chunks 1
Embedding
model...
EmbeddingsVector DBOfﬂine Online
Yes, ACE inhibitors
reduce post-MI [...]AnswerWhat are the
mortality outcomes of
ACE inhibitors after
MI?
QueryRetriever
Embedding
modelQuery
embedding
Large Language
ModelChunks +
user query
Vector searchGenerate2
3
4
56
7
Figure 1: The standard RAG pipeline: offline indexing (left) encodes documents into a vector database; online retrieval (right)
matches a query to relevant chunks before LLM answer generation.
typically a sentence transformer [ 45]. Encodings allow the data-
base to capture semantic relationships between queries and docu-
ments, supporting retrieval beyond what keyword-based indices
permit [ 35]. Incoming queries are embedded in the same space, and
the database returns the nearest neighbors using vector distance.
For tractability on large corpora, Approximate Nearest Neighbor
(ANN) indexing techniques are used to accelerate this search [29].
2.2 Data Ownership and Availability
Our objective is to build a decentralized mechanism for the retrieval
of relevant documents. A vast body of works exists surrounding
peer-to-peer technologies and information retrieval [ 47]. Our work
challenges two commonplace assumptions of these technologies
andRAG : (1) that data can be freely ingested regardless of copyright
or access control, and (2) that data must be globally available.
Copyright and access control.Both peer-to-peer retrieval
systems and RAG architectures typically assume that information is
generally available; RAG systems in particular oftentimes expect to
freely ingest any corpus. We argue that this assumption will collapse
under scrutiny when it comes to the intended access control or
copyright law [ 48].LLM s have been shown to be trained on scraped
websites, whereas their terms of service prohibit this, and some even
on pirated material, completely circumventing access control [ 37].
There have also been various court cases, that were lost by the
companies behind the technology, where MLalgorithms generated
copyrighted material verbatim [ 5]. Strict access control is therefore
a hard requirement in RAG systems, particularly in domains where
data sovereignty is key, like healthcare or finance. By adhering
to purely decentralized interactions,ZeroRAGis compatible with
existing access control mechanisms and peers can refuse or allow
other peers to access their information at their discretion.
Global data availability.Major decentralized systems,e.g.,Bit-
coin[ 40],BitTorrent[ 44], andIPFS[ 11], are built around the
assumption that data should be available everywhere, and they
solve for this by replicating data across nodes. As previously dis-
cussed, replication is not always legal or permissible: a hospital, for
instance, cannot freely copy patient records to arbitrary peers. Data
has to stay with the peer that owns it and peers make no guaran-
tees about availability or access. Rather than solving the replication
problem,ZeroRAGroutes queries to find data where it already
resides, using only local queries and data set descriptions, without
replicas of the underlying documents in the network. This comes
at a cost: becauseZeroRAGmaintains no global index, it cannot
offer the routing-time guarantees that a system likeKademlia[ 39]
provides, and retrieval is therefore best-effort by design.2.3 Towards Decentralized RAG
Our routing protocol should, given a user query, locate non-local
owners of target or relevant documents to that query under bounded
coordination and without centralized control. Forwarding queries to
retrieve documents is a promising approach to decentralizing RAG
workflows. This allows peers to perform access control on three lev-
els: identity-based (or attribute-based) access control, query-based
access control, and view-based (or purpose-based, or context-aware)
access control. Our innovation therefore is in theRetrievercompo-
nent in the RAG workflow (step ➍in Figure 1), which still receives
results from the local vector database, but also combines these with
results from (remote) peers in an overlay network. This introduces
two new challenges: (1)ZeroRAGmust quickly find and query
peers in the network that hold relevant documents to a given query,
and (2)ZeroRAGmust adapt to changes in the available documents
over time,e.g., due to peers joining and leaving or updating their
local stores, causing semantic drift when the topology becomes
misaligned with the current distribution of documents and queries.
Challenge 1: locality-aware peer sampling.Simply querying
all peers in a decentralized overlay ensures maximum recall in terms
of document retrieval, but cannot work on Internet-scale networks
or data volumes. Instead, each peer should maintain connections
to a small set of semantically proximate peers, such that relevant
documents can be retrieved from currently connected neighbors
alone, without sacrificing recall. This is the goal oflocalityoptimiza-
tion: steering peer connections toward neighbors whose document
collections are likely to satisfy incoming queries [ 57]. For example,
in a network of hospitals with different specializations, two pedi-
atric hospitals are more likely to find similar cases in each other’s
documents than in those of hospitals with unrelated specializations.
However, in semantically homogeneous settings, such as a network
ofonlypediatric hospitals, all peers hold similar documents, so
any neighbor is as good as any other and a locality-aware overlay
brings no benefit over random connections. Therefore, we assume
a heterogeneous setting. However, locality optimization cannot be
treated as a one-time configuration: peers may leave the network,
and the documents they hold may change over time, shifting what
constitutes a relevant neighborhood. The first challenge addressed
byZeroRAGismaintaining temporally adaptive peer connections
that remain semantically relevantas the network evolves.
Challenge 2: connection reconfiguration.Existing decentral-
ized retrieval systems that achieve efficient lookup typically do so
by maintaining a globally shared index structure, such as a semantic
trie [ 41] or a distributed hash table ( DHT ) [39]. These structures
require continuous network-wide reorganization as peers join and

Middleware ’26, December 14–18, 2026, Tarragona, Spain Neague, et al.
leave and as document collections evolve. In our setting, where
coordination must remain strictly local, maintaining such a global
structure is not an option. Thus, the second challenge thatZe-
roRAGtakes on is toreconfigure neighborhood connections directly
as queries and documents evolve. Our approach is comparable to
how border gateway protocol ( BGP) routes packets without a global
directory: each router maintains only local routing tables that are
updated incrementally as traffic flows through the network, and
efficient paths emerge from these local updates alone.
3 ZeroRAG Architecture and Design
ZeroRAGis a decentralized retrieval layer that routes queries to
peers that hold relevant documents. In this section, we first discuss
the actors and assumptions, and then peers’ local algorithms for
neighborhood maintenance and query answering.
3.1 System Model
Our system model focuses on query routing; we empirically evalu-
ate the accuracy of the downstream RAG answer, but do not model
the quality of the citation or the reduction of hallucinations.
Network and communication.ZeroRAGcoordinates decen-
tralized document retrieval by maintaining a permissioned decen-
tralized overlay network and reconfiguring connections between
peers when queries are issued. The incoming and outgoing con-
nections by each peer form a directed graph. Peers joining the
ZeroRAGnetwork can bootstrap into the network by connecting
(outgoing) to several random peers, which can be obtained by using
existing peer sampling services [ 18,23]. Threats targeted at the
network level, like Sybil [ 13] and Eclipse [ 21] attacks, are beyond
the scope of this paper, but we note that peer sampling mechanisms
can overcome these after a cold-boot period of several minutes [ 52].
ZeroRAGonly uses primitive request-response messages (pre-
sented shortly), does not maintain shared data structures, or require
synchronization steps between peers. Therefore, messages may be
asynchronous or drop. Likewise, peer connections can drop at any
time,e.g., when a peer goes offline or crashes. We assume the peer
sampling mechanism can detect dropped connections and that this
can be implemented with minimal overhead using a ping-pong
scheme, message-reception acknowledgments and timeouts. Be-
yond this, we do not investigate Byzantine behavior of peers, like
attempts to bias others’ search results or black hole attacks [1].
Documents and embeddings.InZeroRAG, peers exchange
embeddings of documents and queries. The embeddings are used
for indexing and network reconfiguration. The transfer of the doc-
uments corresponding to those embeddings is outsideZeroRAG.
Exchanged embeddings need to be stored by peers and each peer
operates a local vector database for this. We assume that peers use
the same embedding model 𝜙𝑒, so embeddings produced by differ-
ent peers remain comparable. As embeddings are local, routing does
not depend on centrally assigned labels or a global topic taxonomy,
which was the common approach in the late 2000’s [32, 50].
Peers are expected to accrue documents with attribution over
time,i.e., a verifiable record of which peer a document originates
from, as they exchange them afterZeroRAGmatches their embed-
dings to queries. The attribution itself is out of the scope ofZe-
roRAGand can be implemented with a digital signature. Within thescope of the paper we are interested in routing, not in the manner
in which this history is built. Therefore, we shortcut the emergent
effect of collecting documents by using a HISTORY() remote pro-
cedure call, which gathers the history of accrued documents from
neighborsin memory. When considering deployment ofZeroRAG,
this can (and should) be replaced with a performant storage layer.
3.2 High-Level Description
ZeroRAGis built around the insight that if a peer’s neighbors hold
semantically similar content, then a query is best served by greed-
ily walking toward the most similar peer at each step. To this end,
whenever a peer 𝑖connects to another peer 𝑗,𝑗sends a set ofadver-
tisementsto 𝑖. These advertisements are compact summaries, in the
form of document embedding vectors, of the content 𝑗locally holds
and wants to advertise for routing. This allows owners to retain
their documents and decide whether and what to disclose, which is
not the same as privacy-preserving retrieval: advertisements, query
embeddings, and issuer referrals may reveal information even when
documents are withheld. Now, some peer 𝑝aims to obtain the top- 𝑘
documents that are relevant to a query 𝑞. To facilitate this search,
peer𝑝maintains afrontier, which is a ranked list of candidate peers
to visit, ordered by their estimated relevance to the query. The best-
first walk proceeds by repeatedly popping the highest-scoring peer
from the frontier, sending it the query, and collecting its response:
a set of relevant local document embeddings together with new
referralsto that peer’s own neighbors. These referrals are scored
against the query and put back into the frontier, extending the
search into previously unvisited parts of the network. Meanwhile,
peer𝑝keeps track of the document embeddings that it has seen.
After a fixed number of steps, 𝑝will inspect the these embeddings
and compare their relevance against query 𝑞. Peer𝑝then retrieves
the top-𝑘relevant documents from the peers that offer them.
ZeroRAGcontinuously reconnects the overlay itself as queries
are issued. After each query completes, the querier adds a directed
edge toward whichever peer returned the most relevant result,
evicting the oldest existing connection if the neighbor list is full.
This implicit feedback loop causes the overlay to self-organize
around semantic affinity: peers that frequently serve useful results
for a given topic accumulate more incoming connections, and future
queries on that topic are naturally steered toward them earlier in
the walk. As the overlay network reconnects, peers are increasingly
likely to quickly find the documents they need, without any global
coordination or shared data structure. In addition, each peer keeps
a bounded routed-query memory: a cache of embedding vectors
from queries it has recently responded to, which were issued by
other peers. When scoring an incoming query, a peer consults this
memory alongside its neighbor advertisements, and can refer the
querier not only toward peers with relevant documents but also
toward peers that have recently asked similar questions, and are
thus likely to be connected to, or themselves hold, relevant content.
Illustrative example.Figure 2 (left) shows the progression of a
document search, issued by peer 𝑎➊. Peer𝑎iteratively sends search
queries to peers in its frontier and receives replies with relevant
documents and referrals. We show a document search where four
peers are queried by 𝑎(➋-➎). After the search completes, peer 𝑎
adds peer𝑖as neighbor since it has the most relevant documents ➏.

ZeroRAG Middleware ’26, December 14–18, 2026, Tarragona, Spain
$$a$$$$b$$
$$c$$$$e$$
$$g$$$$f$$$$h$$
$$i$$2
Node
b is neighbor
of a
Query (by a)
and result
msg (by b)Legend
34
5
Create new neighbor edge6Issue
doc
search 
1$$a$$Issue
search
with
query
1
$$b$$2$$h$$
4
Add peer a to
query cache356Referred by b
Issue
search
with
query$$q_1$$$$q_2$$
$$q_1 \approx q_2$$Searching for relevant documents Caching queries
$$a$$ $$b$$
$$a$$ $$b$$
Figure 2:(left)Example of 4 query steps of a document searched by peer 𝑎with query 𝑞. After the search concludes, peer 𝑎adds
peer𝑖, which has the most relevant document to𝑎’s query, as neighbor.(right)The dynamics of query caching inZeroRAG.
Figure 2 (right) shows query caching. Peer 𝑎issues a search
with query𝑞1and queries peer 𝑏(➊,➋). Peer𝑏adds peer𝑎and the
incoming search query to its local query cache ➌. When another
peerℎlater issues a query 𝑞2, which is similar to 𝑞1(i.e., their
embeddings are close) and interacts with peer 𝑏,𝑏refers peerℎto
peer𝑎as𝑎is likely to have found relevant documents for𝑞 2.
3.3 Technical Description
We describe the concrete steps inZeroRAGthrough the roles of
a peer𝑢that aims to retrieve 𝑘documents relevant to a query
𝑞(with embedding 𝑒𝑞) by repeatedly sendingqerymessages to
candidate peers in the frontier, and a peer 𝑝that receives such a
qerymessage. These roles are purely explanatory: any peer can
fulfill either role, even at the same time. Algorithm 1 explains the
search procedure from the perspective of peer 𝑢and Algorithm 2
shows how peer𝑝handles an incoming query.
Local state.To facilitate routing, every peer 𝑖stores a local state
𝜎𝑖, given by the 4-tuple (𝐷𝑖,𝑁𝑖,𝐴𝑖,𝑀𝑖). The documents stored in
the local database of peer 𝑖are represented by 𝐷𝑖. The neighbors
of𝑖are given by 𝑁𝑖, which is a FIFO list with limited capacity 𝛿𝑛.
Insertion and eviction from this list is based on the most recent
highest-scoring query. The collection of advertisements collected by
𝑖, of its neighbors, is given by 𝐴𝑖. Lastly, the collection of the most
recent𝛿𝑚received pairs ( 𝑢,𝑒𝑞), as a result of acting as the role of the
queried peer 𝑝, form the chronological FIFO list 𝑀𝑖. The local state
is designed for the simplest effective implementation. FIFO avoids
maintaining estimates of each neighbor’s usefulness: eviction needs
only insertion order [ 57]. Maximum scoring allows one close match
to guide routing, despite unrelated vectors [ 53]. One neighbor per
query limits connection changes and advertisement exchanges [ 51].
Both algorithms 1 and 2 usescoresof relevance. These scores are
simply the cosine similarity of two embeddings, or a score of minus
infinity if the two have not been compared.ZeroRAGattempts to
localize documents with the highest score in the network.
Retrieving documents.A peer 𝑢invokes the retrieveDoc-
uments procedure, given in Algorithm 1, to retrieve up to 𝑘of
the most-relevant documents to the query 𝑞. The remainder of
the algorithm will use the embedding 𝑒𝑞of𝑞(line 2) to query
neighbors of 𝑢. If the required documents are not found locally,
the peer constructs a frontier 𝐹of uncontacted neighbors, and
successive neighbors of neighbors. It initially consists of pairs of
currently-connected neighbors 𝑁𝑢of𝑢and scores of minus infinityto signify that they have not been queried (line 5). During the
following best-first-search loop (line 7), 𝐹may already receive
scores for identities from previously queried peers. The search loop
successively visits any (there may be multiple with the same score)
most high-scoring peer 𝑥in the frontier, up to 𝐶peers, inserting 𝑥
into the visited peer set 𝑄(line 7-13). Peer 𝑢then sends aqery
message to peer 𝑥, containing the query embedding 𝑒𝑞and the
value of𝑘(line 14). Peer 𝑢awaits a responseqeryResultmessage
from𝑥, containing the top remote results and top referrals of 𝑥(line
15). The document results are merged into the set 𝑅, which is then
trimmed down to the top- 𝑘(line 16). Returned identities and their
scores are inserted into 𝐹, or the identity’s score is updated if it was
already present, while the best-known score and its (potentially
unvisited) associated identity is saved in 𝐵(lines 17-22). After the
search loop is finished, the network connections are reconfigured
(line 24) and the documents are retrieved, by their embedding, from
the peer that stores them and returned (line 25 and 26).
Neighborhood reconfiguration.As a result of a query and its
retrieved document embeddings 𝑅and best identity-score pair 𝐵,
the current neighbors of 𝑢are updated using the updateNeighbors
procedure, given in Algorithm 1 (line 28). The identity 𝑎of the peer
that has the highest-scoring document embedding in 𝑅is retrieved
(line 31). Otherwise, if 𝑅was empty, the identity of the peer 𝑎that
was advertised as the best score is retrieved (line 33), if there was
one. If an identity 𝑎is available, it is connected to and if that causes
the current number of neighbors to exceed 𝛿𝑛the oldest neighbor
is removed from the list (lines 35-40).
Query handling.We next explain how peer 𝑝handles an in-
comingqerymessage, also see Algorithm 2. First, peer 𝑝queries
its local vector database of to retrieve document embeddings 𝐻
based on𝑒𝑞(line 2). Then, 𝑝forms a pool of routable, known em-
beddings𝐸by all known pairs of document embeddings and query
embeddings of neighbors, coupled to the identity of the neighbor
identifier𝑣that sent them (line 3). The scores 𝑆of identities are
initialized with minus infinity (line 4) and then set to the highest
score over the elements in 𝐸that are associated with it (lines 5-9).
The top𝑟peers are then taken from 𝑆(line 10) to be returned as
the referrals 𝑅𝑒𝑓𝑠 later. The pair of the querying peer 𝑢and its
query embedding 𝑒𝑞is stored in the routed-query memory 𝑀𝑝(line
11), to support future routing of queries similar to 𝑢. The oldest
entry in𝑀𝑝is evicted if the number of these tuples exceeds 𝛿𝑚size
(lines 12-14). Finally, 𝑝sends aqeryResult, with the local top- 𝑘
embeddings in𝐻and the top-𝑟referrals in𝑅𝑒𝑓𝑠(line 15).

Middleware ’26, December 14–18, 2026, Tarragona, Spain Neague, et al.
Algorithm 1Retrieving top-𝑘documents for query𝑞of peer𝑢.
Require:𝐶 : contact budget, 𝑟: referral limit, 𝛿𝑛: max amount of neighbors
each peer maintains
Require:Persistent state𝜎 𝑢=(𝐷𝑢,𝑁𝑢,𝐴𝑢,𝑀𝑢)⊲local
documents𝐷𝑢, outgoing neighbors 𝑁𝑢, neighbor advertisements 𝐴𝑢,
and routed-query memory𝑀 𝑢.
1:procedureretrieveDocuments(𝑞: query,𝑘: result limit)
2:𝑒𝑞←𝜙𝑒(𝑞)⊲generate embedding of𝑞
3:𝑄←∅⊲peers already contacted
4:𝑅←∅⊲document results found so far
5:𝐹←{(𝑢,−∞)}⊲frontier of peers to contact
6:𝐵←∅⊲best referral score seen for each peer
7:while𝐹≠∅and|𝑄|<𝐶do
8:(𝑥,_)←arg max (𝑣,𝑠)∈𝐹𝑠⊲peer with highest score
9:𝐹.removePeer(𝑥)
10:if𝑥∈𝑄then
11:continue
12:end if
13:𝑄←𝑄∪{𝑥}
14:sendqery(𝑒 𝑞,𝑘)message to peer𝑥
15:receiveqeryResult(𝐻 𝑥,Refs𝑥)message from peer𝑥
16:𝑅←topK(𝑅∪𝐻 𝑥,𝑘)⊲keep the best document results
17:for all(𝑣,𝑠)∈Refs𝑥do
18:if𝑣∉𝑄and(𝐵[𝑣]is undefinedor𝑠>𝐵[𝑣])then
19:𝐵[𝑣]←𝑠
20:𝐹.update((𝑣,𝑠))
21:end if
22:end for
23:end while
24:updateNeighbors(𝑅,𝐵)⊲Overlay maintenance
25:𝐷←obtainDocuments(𝑅)⊲retrieve docs from docrefs
26:return𝐷
27:end procedure
28:procedureupdateNeighbors(𝑅: found docrefs,𝐵: ref scores)
29:𝑎←⊥
30:if𝑅≠∅then⊲We found at least one document
31:𝑎←peer returning the highest-scoring document in𝑅
32:else if𝐵≠∅then
33:𝑎←arg max 𝑣𝐵[𝑣]
34:end if
35:if𝑎≠⊥and𝑎≠𝑢then
36:𝑁 𝑢.add(𝑎)
37:if|𝑁 𝑢|>𝛿𝑛then
38:𝑁 𝑢.evict()⊲Evict oldest neighbor
39:end if
40:end if
41:end procedure
Retrieval parallelization.To reduce end-to-end retrieval la-
tency, it is possible to parallelize Algorithm 1. Instead of performing
a best-first search, a breadth-first search can send and await up to
𝑓qeryandqeryResultmessages in parallel. As requests and
responses can be asynchronous, the number of visited peers that
have responded might then overshoot the contact budget 𝐶by the
node fanout of the connection graph. In the remainder of this paper,
we use the sequential version of Algorithm 1 to avoid additional
variance in our metrics, which do not include wall-clock time.
Cleanup.Our algorithms do not show the response ofZeroRAG
to churn, as signaled by the peer sampling. When peer 𝑝leaves,Algorithm 2Query handling at peer𝑝.
Require:𝑟: referral limit,𝛿 𝑚: memory size limit
Require:Persistent state𝜎 𝑝=(𝐷𝑝,𝑁𝑝,𝐴𝑝,𝑀𝑝)
1:On receiveQuery(𝑒 𝑞,𝑘)messagefrompeer𝑢
2:𝐻←𝐷 𝑝.search(𝑒𝑞,𝑘)⊲local document search
3:𝐸←𝐴 𝑝∪𝑀𝑝 ⊲advertisements and past routed queries
4:𝑆[𝑣]←−∞forevery(𝑣,_)∈𝐸
5:for all(𝑣,𝑧)∈𝐸do⊲compute best scores for peers
6:if𝑣≠𝑢then
7:𝑆[𝑣]←max{𝑆[𝑣],⟨𝑒 𝑞,𝑧⟩}
8:end if
9:end for
10:Refs←topK({(𝑣,𝑆[𝑣]):𝑆[𝑣]>−∞},𝑟)
11:𝑀𝑝.add((𝑢,𝑒𝑞))
12:if|𝑀 𝑝|>𝛿𝑚then
13:𝑀 𝑝.evict()⊲Evict oldest routed-query memory item
14:end if
15:sendQueryResult(𝐻,Refs)message to peer𝑢
peer𝑢removes𝑝from𝑁𝑢and deletes entries associated with 𝑝
from𝐴𝑢and𝑀𝑢. Documents in 𝐷𝑢do not need cleanup and other
peers in the network will now start being routed to𝑢instead of𝑝.
ZeroRAGonly has aqeryand aqeryResultmessage that
can be dropped. The dropping of either of these can be remedied by
waiting until some timeout when receiving the response message in
Algorithm 1 (line 15). If the timeout is triggered, the query results
are an empty list of documents and an empty list of referrals.
Scalability and topology convergence.With fixed-size encod-
ings, each query sends at most 𝐶requests and 𝐶replies, carrying
𝑂(𝐶((𝑘+ 1)𝑑+𝑟)) bytes for𝑑-dimensional embeddings, 𝑘results and
𝑟referrals. If each neighbor advertises at most𝑎vectors, removed
when that neighbor is evicted, persistent routing state occupies
𝑂(𝛿𝑛+𝑑(𝛿𝑛𝑎+𝛿𝑚))bytes, where 𝛿𝑛and𝛿𝑚cap neighbors and
cached queries. Each search adds at most one connection, transfer-
ring𝑂(𝑎𝑑) advertisement bytes. The total routing state and traffic
for one query per peer are 𝑂(𝑛) for𝑛peers, excluding document
transfer, retries, and background maintenance. Regarding topol-
ogy convergence, stability can only be achieved if the number of
connected peers matches the number of interests. At capacity 𝛿𝑛,
adding a new neighbor for the current query evicts the oldest con-
nection, potentially displacing a peer useful to other interests; in
this configuration therefore the topology cannot feasibly converge.
4 Experimental Evaluation
We implementZeroRAGas routing middleware between a network
overlay and local document stores. Our experimental evaluation
answers the following five questions:
(1)What is the performance ofZeroRAGon real-world data
sets in terms of recall and accuracy (Section 4.2)?
(2)How does the end-to-end accuracy of RAG change when
usingZeroRAG(Section 4.3)?
(3)How resilient isZeroRAGto network-wide failure (Section
4.4)?
(4)What is the efficacy ofZeroRAG’s inputs and what is the
effect of their ablation (Section 4.5)?
(5)How quickly doesZeroRAGmake queries routeable when
documents with new topics are introduced (Section 4.6)?

ZeroRAG Middleware ’26, December 14–18, 2026, Tarragona, Spain
4.1 Experimental Setup
We next describe our experimental setup, including our workloads,
baselines and implementation details.
4.1.1 Workloads.We evaluateZeroRAGand baselines using three
real-world workloads:AOL-IA[ 36,43],Amazon Electronics[ 3],
andPubMedQA[ 25]. The three workloads cover complementary
retrieval settings: single-target and multi-relevant retrieval, real
and synthetic user behavior, and general-domain, e-commerce, and
biomedical query distributions. We additionally usePubMedQA
for an end-to-end RAG evaluation withZeroRAG, extending our
evaluation beyond routing performance alone.
AOL-IAprovides a query-and-click stream with user ids, query
text, clicked documents, and timestamps. The full workload con-
tains657 426users,428 157documents and2 594 705total queries.
We adapt the workload to our setting by selecting the10 000users
with the most queries, together performing2 171 460queries. Prior
to running this workload, we populate the local document stores
with the documents that the user has clicked. During our evaluation,
we then replay these events in a randomized order by having peers
search for the document they are interested in (e.g., 𝑘=1), with-
out them looking into their local database first. The single-target
retrieval objective (find the exact document) enables us to test the
effectiveness of query routing inZeroRAGwith a binary objective
score (whether the document is found or not).
Amazon Electronicsis an adaptation of the personalized
product-search benchmark introduced by Ai et al. [ 3]. This bench-
mark contains63 001product descriptions, and each product con-
tains, besides some basic metadata, a category path (e.g., “Camera,
Photo→Digital Camera Lenses”). The workload authors syntheti-
cally constructed search queries by taking each product’s category
path and flattening it into a query string, stripping stopwords and
duplicates. For example, “Camera, Photo →Digital Camera Lenses”
becomes the query “photo digital camera lenses.” Thus, queries
are categorical and fairly repetitive across users. The queries were
then mapped to users using the reviews of purchased products: if a
reviewer purchased a product associated with a category-derived
query, a corresponding reviewer–query pair is made.
In our evaluation, each reviewer in theAmazon Electronics
benchmark becomes a peer in the network. Each reviewer–query
pair becomes one simulation event,i.e., one search query issued
by one peer. The relevant products for that reviewer–query pair
become the documents that peer is searching for. The retrieval
objective forZeroRAGis to recover as many relevant products as
possible from other peers, rather than to recover a single clicked
item. Those documents are distributed across other peers in the
network, based on which peers reviewed those products. Because
queries come from product categories rather than individual user
intent, the same query string will naturally recur across many
different reviewers. This means the experiment ends up simulating
a setting of repeated, overlapping product-search demand, where
many peers are asking semantically similar things at different times.
PubMedQAis a biomedical question-answering dataset built
from abstracts in thePubMeddatabase [ 25]. We adopt the discrete-
answer subset ofPubMedQAwith1000items, in which each item
contains a biomedical question, an abstract, and a categorical an-
swer ( yes,no, ormaybe ). To increase query-level semantic variety,we supplement each original question with five rephrased queries
generated using theGPT-5-nanomodel, resulting in six query
variants (the original question and five rephrasings).
To assignPubMeddocuments to peers, we use Medical Subject
Headings ( MeSH ) terms, assigned toPubMedarticles, as biomedical
topic labels. We sample a MeSH -level preference vector 𝜽𝑢from
aDirichlet(𝜶) distribution, where 𝜽𝑢defines user 𝑢’s preference
distribution over all23 904 MeSH topics. This produces user profiles
that are biased toward a small number of topics. We fix 𝜶= 7×10−5,
where a peer gets on average 15.9 documents on the first topic, 6.6
on the second topic, 3.4 on the third topic and so on. Given a tar-
get number of documents 𝑡, we sample the number of documents
assigned to each MeSH topic according to 𝜽𝑢. We fix𝑡=30un-
less otherwise stated. We then uniformly sample documents from
each selected MeSH topic, and for each document uniformly choose
one of its six query variants. Under this user-model, most peers
are interested in 1-3 main topics with the interest dropping off for
other topics. Each document of a peer 𝑖also carries one randomly
selected query variant. This (peer, document, query) triple defines
a simulation event: peer 𝑖issues the query, and the retrieval objec-
tive is to locate the corresponding document on another peer.Each
peer issues exactly as many queries as it holds documents. Unless
otherwise stated, experiments with thePubMedQAworkload con-
tain50 000simulated peers and30queries per peer, leading to1.5
million query-document events, issued in random sequence.
4.1.2 Implementation.The experiments were simulated by follow-
ing a timeline of ‘events’, one event representing a peer sending a
query which aims to retrieve one or more target documents. We
fix the contact budget 𝐶= 50, meaning that a peer contacts at
most 50 peers during document retrieval. We used theall-mpnet-
base-v2embedding model for embedding generation of documents
and queries [ 49]. For embedding generation of documents in the
AOL-IAworkload, we concatenated the title ,body ,text , and url
values from the data set using the string “\n\n”. For documents in
theAmazon Electronicsworkload, we concatenated their title ,
description ,categories ,brand , and price in a similar fashion.
The overlay topology is implemented as a global directed con-
nection graph, where each graph node represents a peer. Message
passing between nodes is implemented using remote procedure
calls. Document embeddings are kept in a single unified memory
and the local storage of peers are references in that memory.
Figures 3, 4, 6, and 7(b) and all ablation tables use five runs
per configuration: a randomized starting topology for each run
ofZeroRAG, and order of the peers in the tree for Semantica.
For matching windows, bins, or table summaries, we compute the
five-run mean ¯𝑥and sample SD in Figures 4 and 7(b), and the
tables. Figures 3 and 6 instead show the mean with 95% Student- 𝑡
confidence intervals; DHT-LSH was not repeated because its hash
functions are fixed, retrieval does not depend on randomized peer
connections, and unrepeated baselines have no uncertainty bars.
4.1.3 Hardware.All simulations were performed using 12 comput-
ing nodes which are equipped with two AMD EPYC 7282 processors
(16 cores per socket, 2.8 GHz), for 32 physical cores in total.
4.1.4 Baselines.We evaluateZeroRAGagainst two related base-
lines that together occupy different parts in the design space.

Middleware ’26, December 14–18, 2026, Tarragona, Spain Neague, et al.
Semantica[ 41] is the closest prior decentralized semantic re-
trieval system and organizes peers with a globally maintained se-
mantic trie. We use the source code ofSemanticaas published in
the author’s Github repository. For the hierarchical navigable small
world (HNSW) implementation we utilizehnswlib[38].
DHT-LSHis a baseline that pairs a DHT (e.g.,Kademlia[ 39])
with locality-sensitive hashing ( LSH) schemes to provide ANN
search in fully decentralized settings.DHT-LSHrelies on 𝑝-stable
hash functions that map embedding vectors into discrete buck-
ets such that semantically similar vectors collide into the same
bucket with high probability [ 12]. To improve recall,DHT-LSH
maintains𝐿independent hash tables, each using a composition of 𝑘
such functions with bin width 𝑤. At indexing time, each document
embedding is hashed across all 𝐿tables, and the resulting bucket
identifiers are used as keys in the DHT , which distributes storage of
bucket contents across peers. At query time, the query embedding
is hashed through the same 𝐿tables, yielding 𝐿bucket keys, each
of which requires a separate DHT lookup to retrieve candidate doc-
uments. We adopt the suggested hyperparameter values from the
original LSH paper [ 12], fixing𝑘=10. We adopted 𝑤= 3instead
the of the𝑤= 4used by the authors because 𝑤= 4placed 44.7% of
the AOL-IA documents in the single largest bucket.
4.1.5 Metrics.Our primary success metric isrecall: the percentage
of relevant documents retrieved from the entire set available in the
network, excluding documents already held locally by the querying
peer. ForAOL-IA, where each query has a single target document,
this reduces to a binary success rate. ForAmazon Electronics,
where a query may have hundreds of relevant documents, we in-
stead report thefraction of available relevant documents retrieved
within the routing budget. ForPubMedQAwe additionally measure
end-to-end RAG accuracy: the fraction of questions answered cor-
rectly when the retrieved top- 𝑘documents are supplied as context.
This assesses whether routing quality translates into downstream
generation quality. For brevity, in the remainder of this section,
we will use the notationrecall@kfor the recall when the retrieved
top-𝑘documents are supplied as context. Unless stated otherwise,
we report recall averaged over batches of 1000 queries to smooth
per-query variance and surface trends in routing performance as
the overlay evolves over time.
4.2 Effectiveness of Document Retrieval
We first evaluate the effectiveness of document retrieval with differ-
ent configurations ofZeroRAG, and theSemanticaandDHT-LSH
baselines, using theAOL-IAandAmazon Electronicsworkloads.
ForDHT-LSH, we vary the number of buckets𝐿∈{10, 20, 50}.
4.2.1AOL-IAworkload.We issue the search queries in theAOL-
IAworkload, in randomized order, and measure recall@1, which
indicates whether the document has been retrieved or not. Figure 3
shows average recall@1 for each query batch, for all evaluated
systems.ZeroRAGobtains an average recall@1 of 0.91 for the
last query batch. In contrast, this same number forSemanticais
0.65 and significantly lower. We attribute this to the fact that peers
inSemanticahave a single (averaged) embedding that is used
to advertise their documents, whereas inZeroRAGpeers share
more than one non-averaged embedding.DHT-LSHwith 𝐿=50
0 500 1,000 1,500 2,00000.20.40.60.81
Query batches processedAverage recall@1ZeroRAG Semantica DHT-LSH(𝐿=10)
DHT-LSH(𝐿=20) DHT-LSH(𝐿=50)Figure 3: Average recall@1 forAOL-IAquery batches, for
ZeroRAGand different baseline configurations.
1-10 11-50 51-100020406080100
Available relevant documents per queryRetrieved relevant docs (%)ZeroRAG DHT-LSH(𝐿=50) DHT-LSH(𝐿=20)
DHT-LSH(𝐿=10) Semantica
Figure 4: Percentage of retrieved results, binned by the num-
ber available results for a query, for the Amazon workload.
approachesZeroRAGwith an average recall@1 of 0.80, whileDHT-
LSHwith𝐿=20and𝐿=10reach only 0.56 and 0.41, respectively.
However, the drawback of LSH methods is high communication
and computation cost. Every bucket check requires traversing the
DHT : the overhead in our10 000-peer network, for instance, is that
each search query requires approximately50 ·𝑙𝑜𝑔 2(10000)≈664
return messages (to another peer and back). In contrast,ZeroRAG
only requires50return messages per search query for similar per-
formance. Furthermore, the largest DHT bucket for𝐿=50holds
25% of all documents, therefore every query routed to that bucket
has to be compared against all of them.
Takeaway (Figure 3).Using theAOL-IAworkload,ZeroRAG
shows superior retrieval effectiveness compared toDHT-LSHand
Semantica, at a lower message complexity.
4.2.2Amazon Electronicsworkload.We use the same setup as in
the experiment in Section 4.2.1 but run theAmazon Electronics
workload. However, queries in this workload may have many rele-
vant documents rather than just a single result. Therefore, recall@1
is not well defined and we address this by considering different
values of𝑘when computing recall@ 𝑘for three different bin sizes,
based on how many relevant documents there are for each query: if

ZeroRAG Middleware ’26, December 14–18, 2026, Tarragona, Spain
0 5 10020406080100
Queries issued (×105)Performance (%)ZeroRAGrecall HNSW recall
ZeroRAGRAG acc. HNSW RAG acc.
0 5 1002040
Queries issued (×105)Requestsp10–p90 p25–p75
median p99
(a) Recall and RAG accuracy (b) Requests until
the first relevant
document is found
Figure 5: (a) ThePubMedQAend-to-end RAG performance,
usingZeroRAGand HNSW . (b) the distribution of number
ofZeroRAGrequests before the first relevant document.
there are 1-10 available relevant documents, we use recall@10, re-
call@50 for queries with 11-50 available documents, and recall@100
for queries with 51-100 available documents.
Figure 4 shows the fraction of retrieved relevant documents
forZeroRAGand baselines, grouped by the number of available
documents for each query.ZeroRAGretrieves 94.8-98.9% of avail-
able relevant products across all three bins.DHT-LSHbenefits from
broader hash-table coverage but falls behindZeroRAG: with 𝑤=3.0,
𝑘=10,𝐿=50it reaches 90.3-91.7%, with 𝐿=20being 62.1-70.0%, while
a value of𝐿=10drops it to 42.5-50.3%.Semanticadoes match the
performance ofDHT-LSH 𝐿=20in the largest bin (66.7%) but falls
behind again in the smaller and middle bins ( 41.7-52.9%).
Takeaway (Figure 4).In terms of recall,ZeroRAGconsistently
outperformsDHT-LSHandSemanticaon theAmazon Electron-
icsworkload, which has varying numbers of results per query
instead of one best result.
4.3 End-to-End RAG Performance
We now compare the performance of document retrieval withZe-
roRAGin an end-to-end RAG pipeline, using thePubMedQAwork-
load. In this experiment, peers retrieve documents usingZeroRAG
and then use an LLM (Phi3.5-mini-instruct [2]) to generate an
answer. We consider the generated answer to be successful if it
matches the (yes, no, or maybe) ground truth answer provided by
PubMedQA. We use𝑘=5for document retrieval.
Figure 5(a) shows the document retrieval recall and RAG ac-
curacy forZeroRAGand when using a centralized HNSW index,
which resembles an upper-bound on performance. The solid and
dashed lines represent recall and accuracy, respectively. Both the
recall and accuracy ofZeroRAGincrease as peers issue searches,
showing thatZeroRAGsuccessfully adapts its routing to improve
RAG performance. The recall and accuracy of the central HNSW
database remains constant across the experiment since it is a static
data store. At the end of the experiment, the HNSW baseline reaches
97.3%recall with an accuracy of79%, whileZeroRAGreaches93.0%
0 2 4 6 8 10 12 1400.20.40.60.81
catastrophic failure
(50% of peers leaving)
Queries issued (×105)Recall@1Figure 6: Average recall@1 per processed query batch in
ZeroRAGwith thePubMedQAworkload, when there is a
catastrophic failure event (dashed red line).
with77%accuracy. The reason for the maximum recall of97.3%,
even with access to global data and the centralized index, is the im-
perfect embedding of queries and their (GPT5-nano) rephrasings:
it could be that an embedding of a document relevant to a query
is not in the top-5 closest embeddings. This shows thatZeroRAG
gets performance relatively close to that of a centralized baseline.
ZeroRAGcan act as a progressive retrieval mechanism: a peer
may contact only a few other peers for a fast, approximate result,
or contact more peers to improve retrieval quality at the cost of
additional messages. We next measure how many request messages
(see Section 3) are needed untilZeroRAGfinds the closest document
that exists in the network as the overlay organizes. Figure 5(b)
shows this information, grouped by percentile range and including
the median. We observe that as the overlay self-organizes over time,
the number of query messages required to find the best document
decreases. In the final state, 99% of queries require no more than13
query messages to find the best supporting document.
On average, the centralized HNSW baseline processes its longest-
duration queries in24ms/query, whereasZeroRAGrequires379ms/
query (excluding network latency). This difference is the expected
cost of decentralization: a centralized index can exploit global
knowledge and optimized single-machine data structures, whereas
ZeroRAGachieves comparable recall through purely local inter-
actions across an overlay of peers, without any central authority,
global index, or centralized data stores.
Takeaway (Figure 5).End-to-end RAG performance withZe-
roRAGapproaches the recall and accuracy of the centralized HNSW
baseline. For99%of queries,ZeroRAGcan find the best document
in a network in just13requests.
4.4 Resilience to Catastrophic Peer Failure
We now analyzeZeroRAG’s resilience under a catastrophic failure
event, in which 50% of all peers go offline simultaneously at the
midpoint of the experiment and are gradually replaced by new
peers over the remainder of the run. For this experiment, we use
thePubMedQAworkload and we compute the average recall@1
for each batch of1000queries performed. Figure 6 shows these re-
sults and we annotate the catastrophic failure event with a vertical
dashed line. Recall drops sharply at the moment of failure because

Middleware ’26, December 14–18, 2026, Tarragona, Spain Neague, et al.
0 1,000 2,0000.70.80.91
Query batches processedAverage recall@1ZeroRAG
ZeroRAG(no query memory)
ZeroRAG(static topology)
1-10 11-50 51-100020406080100
Available relevant
docs per queryRetrieved relevant docs (%)ZeroRAG
ZeroRAG(no query memory)
ZeroRAG(static topology)
ZeroRAG(no mem, static)
(a)AOL-IA (b)Amazon Electronics
Figure 7: Ablation of query memory and adaptive topology
on the (a)AOL-IAand (b)Amazon Electronicsworkloads.
the learned paths and routed-query memories of the departed peers
are lost. Nevertheless,ZeroRAGrecovers to 95.5% recall in the
final 10% of the run. This recovery happens entirely through local
interactions: replacement peers accumulate routed-query memory
and adaptive edges as they receive search queries, rebuilding use-
ful routing state. After the catastrophic failure, peers require 15.1
requests per query to find the relevant document, which is higher
compared to the same number before the catastrophic event (8.9):
the neighborhoods of newly-joined peers are not yet optimal.
Takeaway (Figure 6).ZeroRAGis robust against catastrophic
failure where routing dynamics get disrupted significantly, and
self-heals entirely through local interactions as replacement peers
rebuild routing state from incoming queries.
4.5 Ablation Study
We analyze the effect of key parameters and components on rout-
ing performance. Unless stated otherwise, experiments use the
PubMedQAworkload and results are reported over the final 10%
of queries, by which point the overlay has converged. We report
average recall@1, measuring the fraction of target documents suc-
cessfully retrieved, andaverage requests required, measuring the
number ofqerymessages sent until the target document is found.
Together, these capture both routing effectiveness and efficiency.
4.5.1 The effect of query cache and topology adaptation.Figure 7
isolates the effect ofZeroRAG’s two routing inputs by disabling
routed-query memory, adaptive topology formation, or both, and
comparing against the full system on both theAOL-IAandAmazon
Electronicsworkloads. Routed-query memory is the dominant
contributor on both workloads: onAOL-IAits removal lowers av-
erage recall@1 from0.89to0.85, while on the more demanding
multi-relevantAmazon Electronicscoverage task it has a much
larger effect. Disabling topology adaptation alone leaves retrieval
close to the full system, and removing both inputs degrades it the
most, confirming that the two mechanisms are complementary.Table 1: The effect of query cache size.
Max. cache size (𝛿 𝑚) Avg. Recall@1 Avg. requests required
0(77.860±0.177)%16.796±0.067
10(85.834±0.120)%12.757±0.041
100(97.612±0.037)%5.900±0.017
1,000(99.652±0.007)%3.448±0.006
10,000(99.652±0.007)%3.448±0.006
Table 2: The effect of network size.
Network size (peers) Avg. Recall@1 Avg. requests required
1,000(99.365±0.272)%3.209±0.111
5,000(98.276±0.066)%4.330±0.059
10,000(98.378±0.051)%4.656±0.015
50,000(99.652±0.007)%3.448±0.006
100,000(99.913±0.005)%5.724±0.011
Table 3: The effect of document count per peer.
Docs. per peer (𝑡) Avg. Recall@1 Avg. requests required
10(98.665±0.018)%5.940±0.023
20(99.292±0.016)%5.545±0.017
30(99.667±0.007)%5.361±0.012
40(99.836±0.006)%5.276±0.009
50(99.919±0.004)%5.222±0.007
4.5.2 The effect of query routing cache size ( 𝛿𝑚).We vary the size
of the query routing cache ( 𝛿𝑚) and report the average recall@1 and
number of requests required to obtain a document for the last 10%
of the queries (of thePubMedQAworkload). Table 1 shows that 𝛿𝑚
is a key contributor to the routing effectiveness ofZeroRAG. With
no routed-query memory ( 𝛿𝑚=0),ZeroRAGobtains 77.9% recall
and requires 16.8 requests to find the target document. Storing 10
routed queries ( 𝛿𝑚=10) improves recall to 85.8% and reduces the
average required requests to 12.75. Increasing 𝛿𝑚to 100 improves
recall to 97.6% and the request count drops to 5.9. Even higher
values lead to diminishing returns for recall and required requests.
4.5.3 Scalability.We evaluate the routing effectiveness and effi-
ciency as the number of peers in the network increases. We vary
the network size from1000to100 000peers and analyze the average
recall@1 and number of requests required to obtain a document
for the last 10% of the queries. Because of our setup, there could
be a situation where a peer attempts to retrieve a document that is
not advertised by any other peer in the network. We exclude these
queries when computing recall@1, so recall is measured conditional
on the target document being advertised by at least one remote peer.
Table 2 shows that average recall@1 remains high (above 98%), even
as the network size grows to100 000peers. This indicates that, once
a relevant document exists in the network,ZeroRAGcan locate it
with high probability even as the peer population grows by two
orders of magnitude. The average request count is non-monotonic
as the network size increases. Furthermore, the average number of
requests required to obtain the target document remains low.

ZeroRAG Middleware ’26, December 14–18, 2026, Tarragona, Spain
Table 4: The effect of neighborhood size. Five seeds.
Max. neighbors (𝛿 𝑛) Avg. Recall@1 Avg. requests required
3(98.614±0.022)%6.198±0.020
5(99.080±0.012)%5.360±0.009
10(99.417±0.012)%4.431±0.010
25(99.606±0.004)%3.702±0.010
50(99.644±0.010)%3.505±0.005
75(99.666±0.008)%3.388±0.004
100(99.675±0.012)%3.305±0.006
Table 5: The effect of parallel searches.
Fanout Rounds Avg. Recall@1 Avg. rounds required
3 10(99.374±0.010)%1.718±0.002
6 5(99.203±0.012)%1.508±0.001
10 3(97.456±0.040)%1.408±0.001
4.5.4 Documents per peer.We evaluate the routing effectiveness
and efficiency as we assign 𝑡morePubMedQAdocuments to in-
dividual peers. We show average recall@1 and requests required
in Table 3. Even for 𝑡=10, recall reaches 98.66% and increases to
99.91% for𝑡=50. As documents are more available, the number
of requests to obtain them decreases from 5.94 for 𝑡=10to 5.22
for𝑡=50: a decrease of 11.5%. We conclude thatZeroRAGis ef-
fective even with small local document stores, while larger stores
substantially improve coverage and reduce message overhead.
4.5.5 Neighborhood size ( 𝛿𝑛).We evaluate routing effectiveness
and efficiency as we vary the number of neighbors ( 𝛿𝑛) each peer is
connected to, see Table 4. Varying 𝛿𝑛from 3 to100has a moderate
effect on recall: 98.61% for 𝛿𝑛=3compared to 99.7% for 𝛿𝑛=100.
Increasing𝛿𝑛shows diminishing returns: a small neighborhood
size is sufficient to obtain high recall as the network converges. In
contrast, the average number of requests shows a sharper decrease:
6.2 for𝛿𝑛=3to 3.30 for 𝛿𝑛=100, a decrease of 46.5%. Larger values
of𝛿𝑛primarily reduce the number of requests by giving each peer
more immediate semantic routing options. This result suggests
that moderate neighborhood sizes (e.g., 50) are reasonable default
choices, striking a good balance between recall and efficiency.
4.5.6 Parallel requests.We evaluate how parallelizing search af-
fects recall and the number of rounds to find a target document.
A peer sends 𝑓qeryandqeryResultin parallel for a fixed
number of rounds. We fix the contact budget 𝐶= 30and redis-
tribute it across fanout and number of rounds: 3 requests per round
for up to 10 rounds, 6 requests per round for 5 rounds, and 10
requests per round for 3 rounds. Table 5 reports the recall and av-
erage rounds over the final 10% of queries. The 𝑓=3configuration
reaches 99.37% recall and requires 1.72 rounds to find the target
document, while increasing 𝑓to 6 we get almost the same recall,
99.20%. With 𝑓=10, recall drops very slightly to 97.46%. A query
typically requires several referral steps to reach the right semantic
neighborhood. Consequently, parallelization is most beneficial in
low-traffic networks where the added message volume is acceptable,
as it reduces wall-clock latency without sacrificing routing depth.
0 0.2 0.4 0.6 0.8 100.51
concept appears
Queries issued (×106)Recall@101 owner 5 owners 10 owners
25 owners 50 owners non-topic baselineFigure 8: Introduction of a new topic at the 60% mark (vertical
line). The lines of owners≥10overlap.
4.6 Spreading New Concepts
Our last experiment shows how fastZeroRAGspreads a new topic
in a network, analyzing how recall evolves for topics which appear
anew, and are very sparsely available, but suddenly become popular
with all peers. To do so, we omit a single topic from thePubMedQA
workload and re-introduce it at the 60% mark into the experiment.
We chose a random topic (“Peptostreptococcus”) that has 10 relevant
documents, and we vary how many peers own it initially. All peers
then continuously send queries to find documents on our chosen
(initially omitted) topic. A document starts appearing in other peers’
local data stores only once a query successfully reaches it.
Figure 8 shows the rolling recall@10 for both the topic’s queries
per initial owner count and the average for other topics as a gray
line. Seeding the topic at more owners lets recall recover faster
once the concept reappears: with 10 or more owners the documents
are immediately reachable across the network and the recall be-
comes1.0almost instantly, causing the lines to overlap. With less
initial owners the documents also spread less fast, with the single-
owner case being the slowest, climbing back only as the documents
propagate through the overlay once queries begin to reach them.
Takeaway (Figure 8).ZeroRAGadapts to newly introduced, ini-
tially sparse topics through local interactions alone, with its speed
governed by how widely the new content is initially held.
5 Related Work
We discuss works on RAG and nearest-neighbor search, federated
and distributed RAG that scale retrieval without removing a co-
ordinating authority, and decentralized peer-to-peer retrieval and
routing, which is closest to our setting. Table 6 gives an overview.
Retrieval-augmented generation. RAG grounds LLM gener-
ation in external documents retrieved at query time, improving
factual accuracy without retraining [ 14,28,62]. Retrieval is typi-
cally cast as nearest-neighbor search over high-dimensional embed-
dings: queries and documents are embedded into a shared space,
and the documents closest to the query are returned [ 42]. Because
exact search is expensive at scale, practical systems rely on ANN
indexes such as HNSW graphs [ 38] or LSH [12,29]. These tech-
niques assume documents are collected and indexed under a single

Middleware ’26, December 14–18, 2026, Tarragona, Spain Neague, et al.
administrative domain, so the index can be built globally.ZeroRAG
targets the opposite: when documents remain with their owners
and no global index can be constructed.
Federated and distributed RAG.Federated approaches route
or merge results across independently managed collections: RA-
GRoute selects which remote sources to query for a given re-
quest [ 19], FeB4RAG benchmarks federated search for RAG [58],
and pFedRAG personalizes retrieval models across federated
clients [ 20]. Others target the privacy or efficiency of the re-
trieval: FedVSE provides privacy-preserving vector search over fed-
erated databases [ 15], PRAG offers end-to-end privacy-preserving
RAG [30], and Proximity caches semantically similar queries to
avoid repeated vector-database lookups [ 7]. A parallel line scales the
retrieval substrate itself: DRAGON distributes RAG retrieval across
local and cloud resources [ 31], DGRAG organizes graph-based re-
trieval over edge-cloud systems [ 63], and HARMONY and SPIRE
build scalable distributed vector databases for high-throughput
ANN search [ 60,61]. These systems improve scalability, privacy,
or cost, but assume a federated, cloud, or managed deployment in
which a coordinating component still orchestrates retrieval.Ze-
roRAGinstead routes each query through peer-local state alone.
Decentralized retrieval and routing.The closest systems to
ZeroRAGuse a decentralized overlay and a local or global state. One
family imposes a shared logical structure that must be maintained
as peers and documents change. Semantica organizes peers into a
global LLM -guided semantic tree overlay [ 41], while DHT designs
such as Kademlia and LSH-based distributed indexes place keys in
a global address space for approximate lookup [ 12,39]. Blockchain-
based designs decentralize control through shared ledger state:
D-RAG coordinates decentralized RAG through a blockchain [ 6],
and a related system records a reliability score for each document
source on a blockchain so that the trustworthiness of retrieved
content cannot be tampered with [ 34]. These approaches remain
decentralized but trade a central authority for global coordination.
A second family keeps routing state local but depends on ex-
plicit symbolic descriptions. REMINDIN’ routes semantic queries
using social metaphors and observations of previously success-
ful peers [ 54]; DRAG/TARW performs topic-aware random walks
given global knowledge of the available topics [ 59]; and SHIMI or-
ganizes decentralized agent memory through an explicit semantic
hierarchy [ 22]. These systems shareZeroRAG’s local-state philos-
ophy, but rely on topic labels, expertise categories, or maintained
hierarchies rather than routing directly from unlabeled embeddings.
A third, longer line of decentralized recommendation and nearest-
neighbor work anticipates our overlay mechanics. WHATSUP dis-
seminates news recommendations through an epidemic protocol
that clusters users by interest without any central server [ 8], and
a follow-up obfuscation-based protocol makes such profile-based
collaborative filtering privacy-preserving [ 9]. Most directly, Cluster-
and-Conquer constructs a 𝑘-nearest-neighbor graph by starting
from a random overlay and greedily improving it through neighbor-
of-neighbor exploration [ 16], the same locality-exploiting princi-
pleZeroRAGuses to rewire its overlay. These build on classical
decentralized overlay techniques such as gossip-based peer sam-
pling [ 23], semantic-proximity content search [ 57], and the broader
literature on robust peer-to-peer search [ 47].ZeroRAGroutes fromTable 6: Whether various retrieval and routing systems have
(“Decentr.”) decentralized governance, use (“Unlab.”) unla-
beled data, or use a global, local, or no (“N/A”) data structure.
System Decentr. Unlab. Data Structure
ZeroRAG[this work]✓ ✓Local
Semantica[41]✓ ✓Global
LSH/Kademlia[12, 39]✓ ✓Global
REMINDIN’[54]✓Local
SHIMI[22]✓Local
Proximity[7]✓N/A
DRAG(TARW) [59]✓Local
pFedRAG[20]✓N/A
DRAGON[31]✓N/A
D-RAG(Blockchain) [6]✓Global
HARMONY/SPIRE[60, 61]✓N/A
unlabeled query and document embeddings and, crucially, addition-
ally remembers recently routed queries, letting peers steer toward
neighbors with semantically similar request histories.
Overall, prior work occupies only part of the design space. Shared-
structure methods are decentralized and often unlabeled, but re-
quire global maintenance; some local routing methods reduce coor-
dination, but rely on labeled or structured semantics; and federated
vector-search systems improve scalability, but are not fully decen-
tralized. To our knowledge,ZeroRAGis unique in treating retrieval
as a querier-based local routing problem: it avoids the global main-
tenance thatSemanticaand Kademlia require, while remaining
fully decentralized, unlike centralized or federated alternatives.
6 Summary
We presentedZeroRAG, a decentralized routing layer for RAG that
locates relevant documents across an overlay of peers without any
central authority, labeled data, or global index. Each peer keeps
only a bounded set of neighbor advertisements and a small cache
of recently routed queries, routing each query through a best-first
walk that continuously reconfigures the overlay toward semanti-
cally relevant neighbors. Our evaluation on theAOL-IA,Amazon
Electronics, andPubMedQAworkloads shows thatZeroRAGcon-
sistently outperformsSemanticaandDHT-LSH, approaches the
recall of a centralized HNSW index in an end-to-end RAG pipeline
(93.0%vs.97.3%at comparable accuracy).ZeroRAGrecovers to
95.5%recall after losing half the network, scales to100 000peers,
and absorbs new topics through local interactions alone. These
results indicate that decentralization is a practical foundation for
RAG , even when documents remain with the peers that own them.
Availability
The implementation of this work is available at https://github.com/
pneague/ZeroRAG-System-Simulation.
Acknowledgments
This work has been funded by the Swiss National Science Founda-
tion, under the project "FRIDAY: Frugal, Privacy-Aware and Practi-
cal Decentralized Learning", SNSF grant number 10001796, and by
NWO/TKI grant BLOCK.2019.004

ZeroRAG Middleware ’26, December 14–18, 2026, Tarragona, Spain
References
[1]Ashraf Abdelhamid, Mahmoud Said Elsayed, Anca D Jurcut, and Marianne A Azer.
2023. A lightweight anomaly detection system for black hole attack.Electronics
12, 6 (2023), 1294.
[2]Marah Abdin, Jyoti Aneja, Hany Awadalla, Ahmed Awadallah, Ammar Ahmad
Awan, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Jianmin Bao, Harkirat Behl,
Alon Benhaim, Misha Bilenko, Johan Bjorck, Sébastien Bubeck, Martin Cai, Qin
Cai, Vishrav Chaudhary, Dong Chen, Dongdong Chen, Weizhu Chen, Yen-Chun
Chen, Yi-Ling Chen, Hao Cheng, Parul Chopra, Xiyang Dai, Matthew Dixon, Ro-
nen Eldan, Victor Fragoso, Jianfeng Gao, Mei Gao, Min Gao, Amit Garg, Allie Del
Giorno, Abhishek Goswami, Suriya Gunasekar, Emman Haider, Junheng Hao,
Russell J. Hewett, Wenxiang Hu, Jamie Huynh, Dan Iter, Sam Ade Jacobs, Mojan
Javaheripi, Xin Jin, Nikos Karampatziakis, Piero Kauffmann, Mahoud Khademi,
Dongwoo Kim, Young Jin Kim, Lev Kurilenko, James R. Lee, Yin Tat Lee, Yuanzhi
Li, Yunsheng Li, Chen Liang, Lars Liden, Xihui Lin, Zeqi Lin, Ce Liu, Liyuan
Liu, Mengchen Liu, Weishung Liu, Xiaodong Liu, Chong Luo, Piyush Madan, Ali
Mahmoudzadeh, David Majercak, Matt Mazzola, Caio César Teodoro Mendes,
Arindam Mitra, Hardik Modi, Anh Nguyen, Brandon Norick, Barun Patra, Daniel
Perez-Becker, Thomas Portet, Reid Pryzant, Heyang Qin, Marko Radmilac, Lil-
iang Ren, Gustavo de Rosa, Corby Rosset, Sambudha Roy, Olatunji Ruwase, Olli
Saarikivi, Amin Saied, Adil Salim, Michael Santacroce, Shital Shah, Ning Shang,
Hiteshi Sharma, Yelong Shen, Swadheen Shukla, Xia Song, Masahiro Tanaka,
Andrea Tupini, Praneetha Vaddamanu, Chunyu Wang, Guanhua Wang, Lijuan
Wang, Shuohang Wang, Xin Wang, Yu Wang, Rachel Ward, Wen Wen, Philipp
Witte, Haiping Wu, Xiaoxia Wu, Michael Wyatt, Bin Xiao, Can Xu, Jiahang Xu,
Weijian Xu, Jilong Xue, Sonali Yadav, Fan Yang, Jianwei Yang, Yifan Yang, Ziyi
Yang, Donghan Yu, Lu Yuan, Chenruidong Zhang, Cyril Zhang, Jianwen Zhang,
Li Lyna Zhang, Yi Zhang, Yue Zhang, Yunan Zhang, and Xiren Zhou. 2024. Phi-3
Technical Report: A Highly Capable Language Model Locally on Your Phone.
(2024). arXiv:2404.14219 [cs.CL]
[3]Qingyao Ai, Yongfeng Zhang, Keping Bi, Xu Chen, and W. Bruce Croft. 2017.
Learning a Hierarchical Embedding Model for Personalized Product Search. In
Proceedings of the 40th International ACM SIGIR Conference on Research and Devel-
opment in Information Retrieval(Shinjuku, Tokyo, Japan)(SIGIR ’17). Association
for Computing Machinery, New York, NY, USA, 645–654. doi:10.1145/3077136.
3080813
[4]Aisha Alansari and Hamzah Luqman. 2025. Large language models hallucination:
A comprehensive survey. arXiv:2510.06265 [cs.CL]
[5]Isha Amjad. 2026. AI Copyright Infringement: Navigating the Legal Risks of
AI-Generated Content.Trends in Intellectual Property Research4 (02 2026), 24–28.
doi:10.69971/tipr.4.1.2026.104
[6]Tessa Andersen, Ayanna Avalos, Gaby Dagher, and Min Long. 2025. D-RAG: A
Privacy-Preserving Framework for Decentralized RAG using Blockchain. (02
2025), 183–198. doi:10.5121/csit.2025.150417
[7]Shai Bergman, Anne-Marie Kermarrec, Diana Petrescu, Rafael Pires, Mathis
Randl, Martijn de Vos, and Ji Zhang. 2025. Leveraging Approximate Caching for
Faster Retrieval-Augmented Generation. InProceedings of the 26th International
Middleware Conference(Vanderbilt University, Nashville, TN, USA)(Middleware
’25). Association for Computing Machinery, New York, NY, USA, 340–353. doi:10.
1145/3721462.3770776
[8]Antoine Boutet, Davide Frey, Rachid Guerraoui, Arnaud Jégou, and Anne-Marie
Kermarrec. 2013. WHATSUP: A Decentralized Instant News Recommender. In
2013 IEEE 27th International Symposium on Parallel and Distributed Processing.
741–752. doi:10.1109/IPDPS.2013.47
[9]Antoine Boutet, Davide Frey, Rachid Guerraoui, Arnaud Jégou, and Anne-Marie
Kermarrec. 2015. Privacy-Preserving Distributed Collaborative Filtering.Com-
puting8593 (03 2015). doi:10.1007/s00607-015-0451-z
[10] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan,
Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter,
Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin
Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya
Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In
Proceedings of the 34th International Conference on Neural Information Processing
Systems(Vancouver, BC, Canada)(NeurIPS ’20). Curran Associates Inc., Red Hook,
NY, USA, Article 159, 25 pages.
[11] Erik Daniel and Florian Tschorsch. 2022. IPFS and friends: A qualitative com-
parison of next generation peer-to-peer data networks.IEEE Communications
Surveys & Tutorials24, 1 (2022), 31–52.
[12] Mayur Datar, Nicole Immorlica, Piotr Indyk, and Vahab S. Mirrokni. 2004. Locality-
sensitive hashing scheme based on p-stable distributions. InProceedings of the
Twentieth Annual Symposium on Computational Geometry(Brooklyn, New York,
USA)(SCG ’04). Association for Computing Machinery, New York, NY, USA,
253–262. doi:10.1145/997817.997857
[13] John R Douceur. 2002. The sybil attack. InInternational workshop on peer-to-peer
systems. Springer, 251–260.[14] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InProceedings of the 30th ACM
SIGKDD conference on knowledge discovery and data mining. 6491–6501.
[15] Zeheng Fan, Yuxiang Zeng, Zhuanglin Zheng, and Yongxin Tong. 2025. FedVSE:
A Privacy-Preserving and Efficient Vector Search Engine for Federated Databases.
Proc. VLDB Endow.18, 12 (Aug. 2025), 5371–5374. doi:10.14778/3750601.3750674
[16] George Giakkoupis, Anne-Marie Kermarrec, Olivier Ruas, and François Taïani.
2021. Cluster-and-Conquer: When Randomness Meets Graph Locality. In2021
IEEE 37th International Conference on Data Engineering (ICDE). 2027–2032. doi:10.
1109/ICDE51399.2021.00195
[17] Seth Gilbert and Nancy Lynch. 2002. Brewer’s conjecture and the feasibility
of consistent, available, partition-tolerant web services.Acm Sigact News33, 2
(2002), 51–59.
[18] Rachid Guerraoui, Anne-Marie Kermarrec, Anastasiia Kucherenko, Rafael Pinot,
and Martijn de Vos. 2024. Peerswap: A peer-sampler with randomness guarantees.
In2024 43rd International Symposium on Reliable Distributed Systems (SRDS). IEEE,
271–281.
[19] Rachid Guerraoui, Anne-Marie Kermarrec, Diana Petrescu, Rafael Pires, Mathis
Randl, and Martijn de Vos. 2025. Efficient Federated Search for Retrieval-
Augmented Generation. InProceedings of the 5th Workshop on Machine Learning
and Systems(World Trade Center, Rotterdam, Netherlands)(EuroMLSys ’25). As-
sociation for Computing Machinery, New York, NY, USA, 74–81. doi:10.1145/
3721146.3721942
[20] Hangyu He, Xin Yuan, Kai Wu, Ren Ping Liu, and Wei Ni. 2025. pFe-
dRAG: A Personalized Federated Retrieval-Augmented Generation System with
Depth-Adaptive Tiered Embedding Tuning. InFindings of the Association for
Computational Linguistics: EMNLP 2025, Christos Christodoulopoulos, Tanmoy
Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computa-
tional Linguistics, Suzhou, China, 14255–14268. doi:10.18653/v1/2025.findings-
emnlp.769
[21] Ethan Heilman, Alison Kendler, Aviv Zohar, and Sharon Goldberg. 2015. Eclipse
attacks on{Bitcoin’s}{peer-to-peer}network. In24th USENIX security sympo-
sium (USENIX security 15). 129–144.
[22] Tooraj Helmi. 2025. Decentralizing AI Memory: SHIMI, a Semantic Hierarchical
Memory Index for Scalable Agent Reasoning. arXiv:2504.06135 [cs.AI]
[23] Márk Jelasity, Spyros Voulgaris, Rachid Guerraoui, Anne-Marie Kermarrec, and
Maarten Van Steen. 2007. Gossip-based peer sampling.ACM Transactions on
Computer Systems (TOCS)25, 3 (2007), 8–es.
[24] Ziwei Ji, Tiezheng Yu, Yan Xu, Nayeon Lee, Etsuko Ishii, and Pascale Fung.
2023. Towards mitigating LLM hallucination via self reflection. InFindings of the
Association for Computational Linguistics: EMNLP 2023. 1827–1843.
[25] Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu. 2019.
Pubmedqa: A dataset for biomedical research question answering. InProceedings
of the 2019 conference on empirical methods in natural language processing and the
9th international joint conference on natural language processing (EMNLP-IJCNLP).
2567–2577.
[26] Kevin Kuo, Chhavi Yadav, and Virginia Smith. 2025. Research in Collabo-
rative Learning Does Not Serve Cross-Silo Federated Learning in Practice.
arXiv:2510.12595 [cs.LG]
[27] Yoonjoo Lee, Kihoon Son, Tae Soo Kim, Jisu Kim, John Joon Young Chung, Eytan
Adar, and Juho Kim. 2024. One vs. many: Comprehending accurate information
from multiple erroneous and inconsistent ai generations. InProceedings of the
2024 ACM Conference on Fairness, Accountability, and Transparency. 2518–2531.
[28] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for
knowledge-intensive NLP tasks. InProceedings of the 34th International Conference
on Neural Information Processing Systems(Vancouver, BC, Canada)(NeurIPS ’20).
Curran Associates Inc., Red Hook, NY, USA, Article 793, 16 pages.
[29] Wen Li, Ying Zhang, Yifang Sun, Wei Wang, Mingjie Li, Wenjie Zhang, and
Xuemin Lin. 2020. Approximate Nearest Neighbor Search on High Dimensional
Data — Experiments, Analyses, and Improvement .IEEE Transactions on Knowl-
edge & Data Engineering32, 08 (Aug. 2020), 1475–1488. doi:10.1109/TKDE.2019.
2909204
[30] Zhijun Li, Minghui Xu, Huayi Qi, Wenxuan Yu, Tingchuang Zhang, Qiao Zhang,
GuangYong Shang, Zhen Ma, and Xiuzhen Cheng. 2026. PRAG: End-to-End
Privacy-Preserving Retrieval-Augmented Generation. arXiv:2604.26525 [cs.CR]
[31] Shangyu Liu, Zhenzhe Zheng, Xiaoyao Huang, Fan Wu, Guihai Chen, and Jie
Wu. 2025. Efficient Distributed Retrieval-Augmented Generation for Enhancing
Language Model Performance. arXiv:2504.11197 [cs.LG]
[32] Alexander Loser, Kai Schubert, and Frederik Zimmer. 2004. Taxonomy-based rout-
ing overlays in p2p networks. InProceedings. International Database Engineering
and Applications Symposium, 2004. IDEAS’04.IEEE, 407–412.
[33] Alexander Löser, Steffen Staab, and Christoph Tempich. 2005. Semantic Methods
for P2P Query Routing. InMultiagent System Technologies, Torsten Eymann,
Franziska Klügl, Winfried Lamersdorf, Matthias Klusch, and Michael N. Huhns
(Eds.). Springer Berlin Heidelberg, Berlin, Heidelberg, 15–26.

Middleware ’26, December 14–18, 2026, Tarragona, Spain Neague, et al.
[34] Yining Lu, Wenyi Tang, Max Johnson, Taeho Jung, and Meng Jiang. 2025. A
Decentralized Retrieval Augmented Generation System with Source Reliabilities
Secured on Blockchain. arXiv:2511.07577 [cs.CR]
[35] Le Ma, Ran Zhang, Yikun Han, Shirui Yu, Zaitian Wang, Zhiyuan Ning, Jing-
han Zhang, Ping Xu, Pengjiang Li, Ziyue Qiao, Wei Ju, Chong Chen, Dongjie
Wang, Kunpeng Liu, Pengyang Wang, Pengfei Wang, Yanjie Fu, Chunjiang
Liu, Yuanchun Zhou, and Chang-Tien Lu. 2026. A Comprehensive Survey
on Vector Database: Storage and Retrieval Technique, Challenge. (2026).
arXiv:2310.11703 [cs.DB]
[36] Sean MacAvaney, Craig Macdonald, and Iadh Ounis. 2022. Reproducing Per-
sonalised Session Search Over the AOL Query Log. InAdvances in Information
Retrieval: 44th European Conference on IR Research, ECIR 2022, Stavanger, Norway,
April 10–14, 2022, Proceedings, Part I(Stavanger, Norway). Springer-Verlag, Berlin,
Heidelberg, 627–640. doi:10.1007/978-3-030-99736-6_42
[37] Pratyush Maini, Hengrui Jia, Nicolas Papernot, and Adam Dziedzic. 2024. LLM
Dataset Inference: Did you train on my dataset? (2024). arXiv:2406.06443 [cs.LG]
[38] Yu A. Malkov and D. A. Yashunin. 2020. Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.
IEEE Trans. Pattern Anal. Mach. Intell.42, 4 (April 2020), 824–836. doi:10.1109/
TPAMI.2018.2889473
[39] Petar Maymounkov and David Mazières. 2002. Kademlia: A Peer-to-Peer Informa-
tion System Based on the XOR Metric. InRevised Papers from the First International
Workshop on Peer-to-Peer Systems (IPTPS ’01). Springer-Verlag, Berlin, Heidelberg,
53–65.
[40] Satoshi Nakamoto. 2008. Bitcoin: A peer-to-peer electronic cash system. (2008).
[41] Petru Neague, Quinten Stokkink, Naman Goel, and Johan Pouwelse. 2025. Se-
mantica: Decentralized Search using a LLM-Guided Semantic Tree Overlay.
arXiv:2502.10151 [cs.IR]
[42] Sameer A Nene and Shree K Nayar. 1997. A simple algorithm for nearest neighbor
search in high dimensions.IEEE Transactions on pattern analysis and machine
intelligence19, 9 (1997), 989–1003.
[43] Greg Pass, Abdur Chowdhury, and Cayley Torgeson. 2006. A picture of search.
InProceedings of the 1st International Conference on Scalable Information Systems
(Hong Kong)(InfoScale ’06). Association for Computing Machinery, New York,
NY, USA, 1–es. doi:10.1145/1146847.1146848
[44] J.A. Pouwelse, Pawel Garbacki, D. Epema, and Henk Sips. 2005. The Bittorrent
P2P File-Sharing System: Measurements and Analysis.Peer-to-Peer Systems IV
3640, 205–216. doi:10.1007/11558989_19
[45] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. 3973–3983. doi:10.18653/v1/D19-1410
[46] Nicola Rieke, Jonny Hancox, Wenqi Li, Fausto Milletari, Holger R Roth, Shadi
Albarqouni, Spyridon Bakas, Mathieu N Galtier, Bennett A Landman, Klaus Maier-
Hein, et al .2020. The future of digital health with federated learning.NPJ digital
medicine3, 1 (2020), 119.
[47] John Risson and Tim Moors. 2007.Survey of research towards robust peer-to-peer
networks: Search methods. Technical Report.
[48] Eleonora Rosati. 2025. Infringing AI: Liability for AI-Generated Outputs under
International, EU, and UK Copyright Law.European Journal of Risk Regulation
16, 2 (2025), 603–627. doi:10.1017/err.2024.72
[49] Sentence Transformers. [n. d.]. all-mpnet-base-v2. https://huggingface.co/
sentence-transformers/all-mpnet-base-v2.
[50] Dou Shen, Jian-Tao Sun, Qiang Yang, and Zheng Chen. 2006. Building bridges
for web query classification. InProceedings of the 29th annual international ACM
SIGIR conference on Research and development in information retrieval. 131–138.
[51] K. Sripanidkulchai, B. Maggs, and H. Zhang. 2003. Efficient content location
using interest-based locality in peer-to-peer systems. InIEEE INFOCOM 2003.
Twenty-second Annual Joint Conference of the IEEE Computer and Communications
Societies (IEEE Cat. No.03CH37428), Vol. 3. 2166–2176 vol.3. doi:10.1109/INFCOM.
2003.1209237
[52] Quinten Stokkink, Can Umut Ileri, Dick Epema, and Johan Pouwelse. 2023. Web3
Sybil avoidance using network latency.Computer Networks227 (2023), 109701.
[53] Chunqiang Tang, Zhichen Xu, and Sandhya Dwarkadas. 2003. Peer-to-peer infor-
mation retrieval using self-organizing semantic overlay networks. InProceedings
of the 2003 Conference on Applications, Technologies, Architectures, and Protocols for
Computer Communications(Karlsruhe, Germany)(SIGCOMM ’03). Association for
Computing Machinery, New York, NY, USA, 175–186. doi:10.1145/863955.863976
[54] Christoph Tempich, Steffen Staab, and Adrian Wranik. 2004. Remindin’: semantic
query routing in peer-to-peer networks based on social metaphors. InProceedings
of the 13th International Conference on World Wide Web(New York, NY, USA)
(WWW ’04). Association for Computing Machinery, New York, NY, USA, 640–649.
doi:10.1145/988672.988759
[55] U.S. Department of Health & Human Services. 1996. Health Insurance Portability
and Accountability Act of 1996. https://www.hhs.gov/hipaa Public Law 104-191,
110 Stat. 1936.
[56] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all
you need.Advances in neural information processing systems30 (2017).[57] Spyros Voulgaris, A-M Kermarrec, and Laurent Massoulié. 2004. Exploiting
semantic proximity in peer-to-peer content searching. InProceedings. 10th IEEE
International Workshop on Future Trends of Distributed Computing Systems, 2004.
FTDCS 2004.IEEE, 238–243.
[58] Shuai Wang, Ekaterina Khramtsova, Shengyao Zhuang, and Guido Zuccon. 2024.
FeB4RAG: Evaluating Federated Search in the Context of Retrieval Augmented
Generation. InProceedings of the 47th International ACM SIGIR Conference on
Research and Development in Information Retrieval(Washington DC, USA)(SIGIR
’24). Association for Computing Machinery, New York, NY, USA, 763–773. doi:10.
1145/3626772.3657853
[59] Chenhao Xu, Longxiang Gao, Yuan Miao, and Xi Zheng. 2025. Distributed
Retrieval-Augmented Generation. arXiv:2505.00443 [cs.DC]
[60] Qian Xu, Feng Zhang, Chengxi Li, Lei Cao, Zheng Chen, Jidong Zhai, and Xi-
aoyong Du. 2025. HARMONY: A Scalable Distributed Vector Database for High-
Throughput Approximate Nearest Neighbor Search.Proc. ACM Manag. Data3, 4,
Article 249 (Sept. 2025), 28 pages. doi:10.1145/3749167
[61] Yuming Xu, Qianxi Zhang, Qi Chen, Baotong Lu, Menghao Li, Philip Adams,
Mingqin Li, Zengzhong Li, Jing Liu, Cheng Li, and Fan Yang. 2025. Scal-
able Distributed Vector Search via Accuracy Preserving Index Construction.
arXiv:2512.17264 [cs.DC]
[62] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng,
Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2026. Retrieval-
augmented generation for ai-generated content: A survey.Data Science and
Engineering(2026), 1–29.
[63] Wenqing Zhou, Yuxuan Yan, and Qianqian Yang. 2026. DGRAG: Dis-
tributed Graph-based Retrieval-Augmented Generation in Edge-Cloud Systems.
arXiv:2505.19847 [cs.AI]
[64] Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. 2023.
ToolQA: a dataset for LLM question answering with external tools. InProceedings
of the 37th International Conference on Neural Information Processing Systems
(New Orleans, LA, USA)(NeurIPS ’23). Curran Associates Inc., Red Hook, NY,
USA, Article 2180, 27 pages.