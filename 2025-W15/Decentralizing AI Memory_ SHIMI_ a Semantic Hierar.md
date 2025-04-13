# Decentralizing AI Memory: SHIMI, a Semantic Hierarchical Memory Index for Scalable Agent Reasoning

**Authors**: Tooraj Helmi

**Published**: 2025-04-08 15:31:00

**PDF URL**: [http://arxiv.org/pdf/2504.06135v1](http://arxiv.org/pdf/2504.06135v1)

## Abstract
Retrieval-Augmented Generation (RAG) and vector-based search have become
foundational tools for memory in AI systems, yet they struggle with
abstraction, scalability, and semantic precision - especially in decentralized
environments. We present SHIMI (Semantic Hierarchical Memory Index), a unified
architecture that models knowledge as a dynamically structured hierarchy of
concepts, enabling agents to retrieve information based on meaning rather than
surface similarity. SHIMI organizes memory into layered semantic nodes and
supports top-down traversal from abstract intent to specific entities, offering
more precise and explainable retrieval. Critically, SHIMI is natively designed
for decentralized ecosystems, where agents maintain local memory trees and
synchronize them asynchronously across networks. We introduce a lightweight
sync protocol that leverages Merkle-DAG summaries, Bloom filters, and
CRDT-style conflict resolution to enable partial synchronization with minimal
overhead. Through benchmark experiments and use cases involving decentralized
agent collaboration, we demonstrate SHIMI's advantages in retrieval accuracy,
semantic fidelity, and scalability - positioning it as a core infrastructure
layer for decentralized cognitive systems.

## Full Text


<!-- PDF content starts -->

arXiv:2504.06135v1  [cs.AI]  8 Apr 2025Decentralizing AI Memory: SHIMI, a Semantic
Hierarchical Memory Index for Scalable Agent
Reasoning⋆
Tooraj Helmi[0009−0000−4959−1735]
University of Southern California, Los Angeles, CA, USA
thelmi@usc.edu
Abstract. Retrieval-Augmented Generation (RAG) and vector-based
search have become foundational tools for memory in AI syste ms, yet
theystrugglewithabstraction, scalability,andsemantic precision—especially
in decentralized environments. We present SHIMI (Semantic Hierarchi-
cal Memory Index), a uniﬁed architecture that models knowle dge as a
dynamically structured hierarchy of concepts, enabling ag ents to retrieve
information based on meaning rather than surface similarit y. SHIMI or-
ganizes memory into layered semantic nodes and supports top -down
traversal from abstract intent to speciﬁc entities, oﬀerin g more pre-
cise and explainable retrieval. Critically, SHIMI is nativ ely designed
for decentralized ecosystems, where agents maintain local memory trees
and synchronize them asynchronously across networks. We in troduce a
lightweight sync protocol that leverages Merkle-DAG summa ries, Bloom
ﬁlters, and CRDT-style conﬂict resolution to enable partia l synchro-
nization with minimal overhead. Through benchmark experim ents and
use cases involving decentralized agent collaboration, we demonstrate
SHIMI’s advantages in retrieval accuracy, semantic ﬁdelit y, and scalabil-
ity—positioning it as a core infrastructure layer for decen tralized cogni-
tive systems.
Keywords: Decentralized AI Agents ·Semantic Memory Retrieval ·Hi-
erarchical Knowledge Representation ·Blockchain-Based Infrastructure
·Retrieval-Augmented Generation (RAG) ·Knowledge Indexing
1 Introduction
As decentralized computing moves beyond ﬁnancial transactions in to cognitive
infrastructure, there is growing demand for intelligent agents cap able of reason-
ing, adaptation, and autonomous coordination. From decentralize d autonomous
organizations (DAOs) [12] to blockchain-native AI agents [17], the se systems
require not only execution logic, but also access to structured, int erpretable
memory. Agents must retrieve knowledge, match tasks, and shar e context across
evolving environments—yet most memory systems in AI today are ne ither ex-
plainable [16] nor decentralized [27].
⋆Supported by Concia

2 Tooraj Helmi
Existingretrievalmethods,suchasdensevectorsearchandRet rieval-Augmented
Generation (RAG) [15], are limited in three ways. First, they rely on ﬂa t, un-
structured embeddings that cannot express conceptual abstr action or composi-
tional meaning. Second, their reliance on centralized indices conﬂict s with de-
centralized trust and infrastructure models. Third, their retriev al results are
often diﬃcult to interpret or justify—especially when the system op erates au-
tonomously or interacts with multiple peers.
To address these limitations, we introduce SHIMI (Semantic Hierarchical
Memory Index), a new memory architecture designed for decentr alized AI sys-
tems. Inspired by cognitive theories of abstraction [23], SHIMI mod els memory
asadynamic treeofsemanticconcepts. Agentsretrieveknowledg ebydescending
from abstract goals toward speciﬁc entities, enabling meaning-bas ed matching
and interpretable reasoning paths.
Unlike RAG and similar approaches, SHIMI embeds semantics into both its
retrieval and synchronization layers. Each agent maintains its own local seman-
tic tree and updates it independently. Synchronization across age nts is handled
through a hybrid protocol using Merkle-DAG summaries [3], Bloom ﬁlte rs [5],
and CRDT-style merging [19]. This allows partial, low-bandwidth update s while
preserving convergence and consistency across asynchronous networks.
We evaluate SHIMI in a simulated task assignment scenario involving mu l-
tiple agents. Our results show that SHIMI improves retrieval accu racy and in-
terpretability while signiﬁcantly reducing synchronizationcost. The se properties
positionSHIMI asafoundationalcomponentfordecentralizedmem oryinemerg-
ing AI-native infrastructures. As contributions, in this paper, we :
1. Propose SHIMI, a semantic hierarchical memory structure for decentralized
AI reasoning;
2. Designalow-bandwidthsynchronizationprotocolusingMerkle-D AGs,Bloom
ﬁlters, and CRDTs;
3. Evaluate SHIMI against vector-based baselines across accura cy, traversal
eﬃciency, and sync overhead;
4. Demonstrate SHIMI’s generalizability across agent matching, de centralized
knowledge indexing, and memory synchronization use cases.
2 Background and Related Work
In this section, we review existing approaches to knowledge retriev al in AI sys-
tems, discuss their limitations in decentralized environments, and ou tline the
foundational technologies that inﬂuence SHIMI’s design.
2.1 Retrieval-Augmented Generation and Vector Search
Retrieval-Augmented Generation (RAG) [15] augments language mo dels with
external memory by retrieving passages relevant to a given query and feeding
them as additional context into the model. Most RAG implementations use

Title Suppressed Due to Excessive Length 3
dense vector representations and rely on embedding similarity to ide ntify rele-
vant content, with pretrained models such as BERT [7] and Senten ce-BERT [18]
commonly used to generate these embeddings. Libraries like FAISS [ 13] or An-
noy [4] support scalable approximate nearest-neighbor search ov er millions of
items.
While vector-based retrieval systems have demonstrated eﬀect iveness in cen-
tralized applications, they exhibit several limitations that hinder the ir broader
applicability.Firstly,thesesystemsoftenemployaﬂatmemorystru cturewithout
inherent hierarchy or abstraction, complicating the modeling of con ceptual rela-
tionships and impeding nuanced reasoning over data [22]. Secondly, e mbedding-
based approachesaresusceptible to semantic drift, where retrie ved items may be
lexically similar but semantically irrelevant, leading to decreased retrie val preci-
sion and potential inaccuracies in outputs [9]. Additionally, the decis ion-making
processes in vector-based retrieval often operate as ”black bo xes,” oﬀering lim-
ited transparency and making it challenging to provide human-aligned justiﬁca-
tions for retrieval outcomes [14]. Furthermore, many implementat ions rely on a
single index within a centralized infrastructure, posing challenges fo r scalability
and limiting applicability in decentralized environments where data distr ibu-
tion is essential [20]. Addressing these limitations is crucial for enha ncing the
performance and applicability of vector-based retrieval systems , particularly in
contextsrequiringexplainability,semanticprecision,anddecentra lizedoperation
2.2 Semantic, Hierarchical, and Graph-Based Memory Struct ures
Memorysystemsinspiredbyhumancognitiontypicallyorganizeknowle dgeacross
layers of abstraction [23]. These structures allow for ﬂexible gener alization, par-
tial recall, and top-down conceptual reﬁnement. Ontology-base dapproaches and
semantic web frameworks [11] aim to formalize such organization usin g struc-
tured schemas, but often require extensive manual curation.
Graph-based retrieval systems such as DBpedia [1], ConceptNet [2 1], and
Wikidata [25] represent knowledge as nodes and labeled edges, allowin g re-
lational reasoning via graph traversal. More recently, graph neur al networks
(GNNs) [26] have been used to propagate contextual information across nodes,
enabling deep semantic learning. However, these systems face cha llenges with
scalability, schema rigidity, and centralization. Furthermore, they lack native
support for decentralized synchronization or agent-local updat es.
Relatedworkhasalsoexploredhierarchicalembeddingspaces[24] andmemory-
augmented neural networks [10], but these models typically operat e in closed
environments with static or preloaded data.
2.3 Memory in Decentralized Systems
Decentralized storage frameworks such as IPFS [3], OrbitDB [6], an d peer-to-
peer knowledge networks [8] provide alternatives to centralized d ata storage.
However,these systems generallyoﬀercontent-addressedper sistence ratherthan

4 Tooraj Helmi
meaning-aware memory. They are not designed to support retriev al by concep-
tual relevance or abstraction level.
To address the challenge of decentralized memory synchronization , several
primitives have proven useful. Merkle-DAGs [3] enable eﬃcient struc tural hash-
ing and diﬃng. Bloom ﬁlters [5] provide compact probabilistic summaries of
large sets. CRDTs (Conﬂict-Free Replicated Data Types) [19] allow distributed
systems to converge on a shared state without global coordinatio n. These com-
ponents form the building blocks of SHIMI’s decentralized synchron ization pro-
tocol.
2.4 Positioning SHIMI
To our knowledge, SHIMI is the ﬁrst system to unify semantic abstr action, hi-
erarchical memory organization, and decentralized synchronizat ion into a sin-
gle retrieval architecture. It addresses the shortcomings of ﬂa t vector search,
static graphs, and centralized indexes by enabling agents to retrie ve informa-
tion meaningfully, eﬃciently, and independently—while ensuring event ual con-
vergence across a distributed network.
3 Approach
SHIMI ( Semantic Hierarchical Memory Index ) is a memory architecture de-
signed to enable meaning-driven retrieval in decentralized agent sy stems. It
combines a semantically structured memory model with a lightweight, scalable
synchronization protocol. In this section, we describe both aspec ts in detail.
3.1 Semantic Memory Architecture
SHIMI organizesmemory asa dynamic tree structure in which seman tic abstrac-
tions are layered hierarchically. Formally, the memory is modeled as a r ooted
directed tree T= (V,E), where each node v∈Vrepresents a semantic concept,
and each directed edge ( vi,vj)∈Eencodes a parent-child relationship. Nodes
may store entities, act as abstractions, or dynamically adapt their structure
through merging.
We deﬁne the following parameters, aligned with our implementation: T, the
maximum number of child nodes under a parent (tree branching fact or);L, the
maximum number of abstraction levels used for generalization; δ∈[0,1], the
similarity threshold for placement or matching; and γ∈[0,1], the compression
ratio for semantic abstraction (word reduction).
Each node v∈Vcontains: a semantic summary s(v)∈ L; a list of children
C(v)⊆V; a set of entities Ev⊆ E; and a parent pointer p(v)∈V∪{⊥}.
The core operation in SHIMI is entity insertion. It involves matching a n in-
coming entity to a subtree, descending semantically, and either att aching it to
a leaf or generating a new abstraction. The algorithm is summarized in Algo-
rithm 1.

Title Suppressed Due to Excessive Length 5
Algorithm 1 AddEntity
Require: Entitye= (c,x): concept cand explanation x
1: Identify candidate root buckets R←MatchToBucket (e)
2:A←DescendTree (R,x)
3:for allparentp∈Ado
4:v←AddNode (new Node( e),p)
5:ife /∈v.Evthen
6:v.Ev←v.Ev∪{e}
7:end if
8:end for
Semantic Descent. The function DescendTree traverses the memory from
matched roots downward. At each level, a generalization check is ap plied using
an LLM-based function GetRelation (a,b)∈ {−1,0,1},where 1 indicates that
ais an ancestor of b; 0 implies semantic equivalence; and −1 means the concepts
are unrelated. This allows semantic narrowing of the insertion point.
Sibling Matching and Merging. If the parent node palready has semantically
equivalent children, detected via FindSimilarSibling using LLM-based simi-
larity,the entityis attached directly. Otherwise,anew node is crea tedand added
top.
If|C(p)|> T, SHIMI triggers merging. The two most similar children ( vi,vj)
are merged under a new parent vmvia:
vm=MergeConcepts (s(vi),s(vj),s(p))
If merging fails (semantic incompatibility), it is aborted. Otherwise, t he new
nodevmreplaces vi,vjand they are reattached as its children.
Abstraction and Compression. When generalizing, SHIMI constructs a semantic
chain of length ≤L, where the number of words in each level wisatisﬁes:
wi+1≤γ·wi
This enforces meaningful compressionand abstractionup the hier archy,avoiding
vague or overly general categories.
Querying. The query operation follows the same traversal pattern, evaluat ing
sim(q,s(v))≥δat each node and continuing only through relevant branches.
The resulting leaf nodes are used to retrieve relevant entities.
Semantic Retrieval SHIMI’s retrieval process mirrors its insertion logic but
operates in reverse: it traverses from root nodes toward leaves , selecting only
branches semantically aligned with the query statement. The algorit hm ensures
eﬃcient pruning by evaluating semantic relevance at each level and c ontinuing
traversal only through matching nodes.

6 Tooraj Helmi
Algorithm 2 RetrieveEntities
Require: Query statement q∈L, similarity threshold δ
1: Initialize result set R←∅
2: Initialize frontier F←{r1,r2,...,r R} { All root nodes}
3:whileF/ne}ationslash=∅do
4:N←NextLevel (F)
5:for allnodev∈Ndo
6:ifsim(q,s(v))≥δthen
7: ifvis leafthen
8:R←R∪E v
9: else
10: F←F∪C(v)
11: end if
12: end if
13:end for
14: Remove processed nodes from F
15:end while
16:return Top-k ranked entities from R
The function simmeasures semantic similarity between the query qand each
node’s summary s(v). Only nodes above the threshold δare expanded, enabling
SHIMI to prune large irrelevant subtrees. Leaf nodes return ass ociated entities
Ev, which are optionally ranked based on frequency, recency, or pro ximity to the
query.
This algorithm maintains explainability via explicit traversal paths and s up-
ports fallback to embeddings if no semantic path exists. In practice , retrieval
depth is bounded by tree height, and branching is limited by the thres holdT
and pruning rate.
3.2 Decentralized Synchronization
SHIMI supports decentralized environments where each node main tains an inde-
pendent semantic tree Ti. Synchronization between trees is performed incremen-
tally using a hybrid protocol that ensures eventual consistency w hile minimizing
bandwidth. The synchronization protocol is summarized in Algorithm 3.
Divergence Detection. Each tree is assigned a Merkleroot hash [2] H(T). When
two peers communicate, they exchange hashes and detect diverg ence ifH(Ti)/ne}ationslash=
H(Tj).Theprotocolrecursivelyidentiﬁesthesmallestdiﬀeringsubtree Tdthrough
a structural hash comparison.
Probabilistic Filtering. To minimize data transfer, each node generates a Bloom
ﬁlter [5]Birepresenting its subtree contents. When a peer receives Bi, it uses it
to infer which nodes it is missing, avoiding full scans or redundant exc hange.

Title Suppressed Due to Excessive Length 7
Algorithm 3 SHIMIPartialSync
Require: Local tree Ti, remote peer j
1: Compute Merkle root Hi←MerkleHash (Ti)
2: Exchange root hashes with peer j
3:ifHi/ne}ationslash=Hjthen
4: Identify divergent subtree Td←FindDiff (Ti,Tj)
5: Generate Bloom ﬁlter Bi←BloomFilter (Td)
6: Send Bito peerj
7: Peer returns missing or changed nodes ∆j
8:for allconﬂict ( vi,vj)∈∆jdo
9:vk←µ(vi,vj) {CRDT-style merge }
10: Replace viwithvkinTi
11:end for
12:end if
13:return Updated tree Ti
Conﬂict Resolution. For each diﬀering node vifrom peer j’s version vj, SHIMI
uses a conﬂict-resolution function µ(vi,vj) that follows CRDT properties [19]:
µ(vi,vj) =µ(vj,vi) (commutativity)
µ(v,v) =v(idempotence)
µ(µ(vi,vj),vk) =µ(vi,µ(vj,vk)) (associativity)
If semantic summaries diﬀer, the protocol favors the summary wit h greater ab-
straction depth or observed usage. Note that only the minimal dive rgent subtree
and related edits are exchanged. This design minimizes overhead and supports
eventual consistency in decentralized settings where updates ar e local and asyn-
chronous.
3.3 Complexity Analysis
We analyze the computational cost of SHIMI’s three core procedu res: seman-
tic entity insertion (Algorithm 1), semantic retrieval (Algorithm 2), and decen-
tralized synchronization (Algorithm 3). These insights inform SHIMI ’s runtime
behavior and LLM call eﬃciency in large-scale agent systems.
Semantic Insertion (Algorithm 1) Inserting an entity involves matching it
to a semantic branch, generating abstraction nodes if necessary , and merging
overloaded children. Let Rdenote the number of root domains, Tthe maximum
number of children allowed per node (i.e., the branching factor), and nthe total
number of entities stored in the memory. The depth of the resulting semantic
tree is denoted by d, which in balanced conﬁgurationsgrowslogarithmicallywith
n. Finally, let Arepresent the average number of nodes visited per level due to
semantic overlap—this value captures the extent of ambiguity or re dundancy
present in the memory hierarchy.

8 Tooraj Helmi
Assuming a balanced semantic tree, the total number of entities nis dis-
tributed approximately evenly across the branches rooted at Rroot nodes,
with each node having at most Tchildren. In such a case, the total number
of leaf nodes at depth dcan be estimated by the exponential growth of the tree:
n≈R·Td. Taking the logarithm of both sides yields an approximate expression
for the depth:
d≈logn
log(RT)
This depth represents the longest semantic path from a root to a le af in a bal-
anced hierarchy, assuming that child nodes are created dynamically as needed
duringinsertionandthatmergesorcompressionsareappliedonlywh enrequired.
Each level includes semantic similarity checks against A·Tnodes, resulting in
total LLM API calls:
APIcallsinsert=R+A·T+2A·T+...+dA·T=R+0.5·A·T·d(d+1)
Example. LetR= 5,T= 4,n= 106,A= 0.5. Then:
d≈log(106)
log(20)≈4.6⇒d≈5
APIcallsinsert≈5+0.5·0.5·4·5·6 = 5+30 = 35
Each semantic comparison in SHIMI involves evaluating the similarity be -
tween two short textual summaries: one from the query or new en tity, and one
from a node’s semantic representation s(v). In our implementation, these sum-
maries typically contain around 20 words, which—given an average to keniza-
tion rate of 1.3 tokens per word—results in approximately 26 tokens per input.
These inputs are passed to a lightweight language model or similarity e ngine
to determine whether the concepts match or relate semantically (e .g., ancestor-
descendant,equivalence,orunrelated).Thetokencountaﬀect sboththeresponse
time and cost of each comparison, and 26 tokens falls well within the e ﬃciency
rangeofcurrentLLMAPIssuchasGPT-4Turbo,whichcanproces ssuchqueries
in under 3 milliseconds.
Semantic Retrieval (Algorithm 2) SHIMI retrieves entities by descending
semantically from root to leaves. At each level, it expands only nodes where
similarity with the query exceeds threshold δ.
Assuming the same parameters as above, and that retrieval follow s a similar
path length as insertion, the number of API calls is:
APIcallsretrieval=R+0.5·A·T·d(d+1)
However, retrieval is often cheaper than insertion because:
–Merging and abstraction steps are skipped.
–Many retrievals can terminate early once relevant leaves are found .
In balanced trees, retrieval latency remains sublinear and bounde d, and in-
terpretability improves with shallower trees and well-separated con cepts.

Title Suppressed Due to Excessive Length 9
Decentralized Synchronization (Algorithm3) SynchronizationusesMerkle-
DAGs and Bloom ﬁlters to identify minimal divergent subtrees betwee n agents.
Let:
–Td: Size of divergent subtree.
–OpsTd: Local edit operations since last sync.
The cost of sync is:
Csync=O(|Td|+|OpsTd|)
Merkle root comparison is constant-time. Bloom ﬁlters reduce unne cessary
transmission, and CRDT-style merges run in bounded time assuming d etermin-
istic node representations. Overall, SHIMI avoids full replication an d supports
bandwidth-eﬃcient convergence even in asynchronous networks .
4 Evaluation
We evaluate SHIMI in a simulated decentralizedsetting to assessits e ﬀectiveness
along four axes: semantic retrieval accuracy, traversal eﬃcien cy, synchronization
cost, and scalability. Each experiment is motivated by known limitation s of ﬂat
vector retrieval and decentralized memory systems [15,19,3]. We pr ovide justiﬁ-
cation for each setup, report quantitative results, and brieﬂy int erpret what each
reveals about SHIMI’s behavior.
4.1 Retrieval Accuracy and Precision
To assess SHIMI’s semantic matching quality, we simulate a use case w here
agents are described using varied lexical forms for overlapping fun ctions. We
compare SHIMI against a RAG-style embedding-based retrieval ba seline on 20
semantically non-trivial queries.
Table 1: Retrieval accuracy comparison between SHIMI and RAG ba seline
Method Top-1 Accuracy Mean Precision@3 Interpretability ( 1–5)
SHIMI (ours) 90% 92.5% 4.7
RAG (baseline) 65% 68.0% 2.1
These results conﬁrm that SHIMI not only improves raw accuracy b ut also
produces more interpretable results due to its use of layered sema ntic general-
izations, in contrast to the ﬂatter vector-based approach.

10 Tooraj Helmi
2 3 4 501020
Tree DepthAvg. Nodes VisitedSHIMI
RAG
Fig.1: Traversal cost vs. tree depth (nodes visited per query)
4.2 Traversal Eﬃciency
Eﬃcient retrieval depends on limiting the number of semantic compar isons per
query. We measure average node visits per query at increasing tre e depths. This
reﬂects how well SHIMI prunes irrelevant branches.
ThegraphillustratesSHIMI’sability tolimit explorationthroughearlyp run-
ing. This sublinear growth contrasts with RAG’s nearly linear cost, sh owing
SHIMI’s eﬃciency in semantically structured contexts.
4.3 Synchronization Setup and Cost
Eﬀective decentralized systems must synchronize knowledge witho ut excessive
bandwidth use. We simulate two strategies:
– Full-state replication: nodes share entire trees.
– SHIMI partial sync: nodes share only divergent subtrees identiﬁed using
Merkle hashes and Bloom ﬁlters.
Table 2: Bandwidth cost of sync with varying number of nodes
Nodes SHIMI Sync (KB) Full Sync (KB) Savings (%)
3 118 1320 91%
4 162 1740 90.7%
5 204 2210 90.8%
6 248 2650 90.6%
Thebandwidthreductionacrossnodecountsconsistentlyexceed s90%,demon-
strating the eﬃciency and scalability of SHIMI’s delta-based sync mo del.

Title Suppressed Due to Excessive Length 11
5 10 15 20 25 3050100
Conﬂict Rate (%)Resolution Time (ms)SHIMI Merge
Fig.2: Conﬂict resolution time vs. conﬂict rate during sync
As conﬂict rate increases,SHIMI’s resolutiontime scalesmoderate ly, remain-
ing bounded and eﬃcient even in higher contention scenarios.
4.4 Scalability
To evaluate scale, we growSHIMI trees up to 2,000 entities and meas ure latency.
This models growing agent ecosystems.
0 500 1 ,000 1 ,500 2 ,000050100150
Number of EntitiesQuery Latency (ms)SHIMI
RAG (ﬂat scan)
Fig.3: Query latency vs. number of entities
Despite entity growth, SHIMI’s latency curve remains relatively ﬂat , validat-
ing its use of pruning and hierarchical structure to avoid linear degr adation.

12 Tooraj Helmi
4.5 Discussion
The evaluation demonstrates that SHIMI consistently outperfor ms ﬂat vector
retrieval methods across multiple performance dimensions relevan t to decen-
tralized AI ecosystems. Starting with retrieval accuracy, the us e of hierarchi-
cal abstraction enables SHIMI to semantically disambiguate agent c apabilities
even when the lexical expression of those capabilities varies widely. T his is
critical in heterogeneous environments where naming conventions are inconsis-
tent or where agents evolve independently. The observed improve ment in inter-
pretability—measuredthroughhumanevaluatorsassigningsemant ictraceability
scores—further emphasizes the advantage of maintaining semant ically meaning-
ful paths through memory.
In terms of traversal eﬃciency, the semantic tree’s ability to prun e large
portions of memory before performing any deep matching results in a signif-
icantly lower number of comparisons per query. This translates dire ctly into
faster response times and lower computational overhead. Compa red to RAG,
which evaluates similarity globally across all candidates, SHIMI’s selec tive at-
tention mechanism oﬀers a more scalable alternative that becomes in creasingly
important as the number of agents or memory entries grows.
Synchronizationeﬃciency is another critical metric for decentraliz ed applica-
tions. SHIMI’s partial sync mechanism, which leveragesMerkle-DAG summaries
and Bloom ﬁlters, ensures that only relevant substructures are c ommunicated
between peers. This results in over 90% reduction in data transfer red during
sync events, a dramatic improvement over naive replication approa ches. As the
number of nodes increases, this eﬃciency becomes even more pron ounced, con-
ﬁrming SHIMI’s suitability for federated deployments. Furthermor e, conﬂict res-
olution scales gracefully with the rate of conﬂict, maintaining low laten cy even
in high-concurrency environments. This is due to the design of SHIM I’s conﬂict-
resolution strategy, which leverages CRDT-style monotonicity and deterministic
summarization to merge competing updates without central arbitr ation.
Scalability tests show that SHIMI’s performance remains robust ev en when
the memory size reaches thousands of entities. While traditional ve ctor-based
methods degrade linearly with increased entity count, SHIMI’s retr ieval remains
bounded due to its logarithmic traversal structure. This propert y ensures that
SHIMI can support real-world use cases involving large, constantly evolving
memory graphs such as task repositories, capability registries, or distributed
knowledge graphs.
Taken together, these results suggest that SHIMI provides not only a more
accurate retrieval model but also a protocol-level framework fo r operating under
decentralized constraints. The architecture is well-suited for age ntic infrastruc-
tures where memory is distributed, ownership is federated, and kn owledge is
constantly updated. By embedding semantic structure into both t he storage
and synchronization layers, SHIMI addresses foundational limitat ions in current
memory models for distributed AI systems.

Title Suppressed Due to Excessive Length 13
5 Applications
SHIMI is designed as a memory protocol for decentralized AI ecosy stems where
semantic retrieval, interpretability, and scalable sync are essentia l. Its architec-
ture enables several classes of applications that are otherwise limit ed by current
ﬂat or centralized memory systems.
5.1 Decentralized Agent Markets
In decentralized marketplaces where agents advertise their capa bilities (e.g.,
computetasks,legalsummarization,sensoranalysis),usersoro rchestratorsmust
semantically match high-level tasks to available agents. SHIMI enab les this via
layered concept matching and provides traceable reasoning paths —crucial for
auditing and agent reputation systems. Unlike RAG, which retrieves based on
surface similarity, SHIMI surfaces conceptually suitable agents ev en when de-
scriptions diﬀer lexically.
5.2 Federated Knowledge Graphs
Collaborative networks, such as research consortia or medical ins titutions, often
maintain independent ontologies. SHIMI allows these networks to sy nchronize
partial views while preserving structure. Its CRDT-style conﬂict r esolution and
semantic merge strategy supports asynchronous updates witho ut requiring a
globalontologyagreement.As eachnode locally evolves,SHIMI ens ureseventual
semantic consistency.
5.3 Autonomous Multi-Agent Systems
In scenarios where agents learn and evolve over time—such as swar m robotics,
ﬁnancial bots, or AI-based assistants—retaining and querying th eir internal
knowledge becomes a bottleneck. SHIMI supports local memory gr owth and
inter-agent sharing without centralized indexing. When an agent jo ins or leaves
a subnet, SHIMI enables fast onboarding or memory reallocation th rough its
partial sync strategy, allowing agent memory to scale with autonom y.
5.4 Blockchain-Based Task Orchestration
Blockchain-native task platforms that assign and verify jobs using smart con-
tracts can integrate SHIMI as a semantic index layer for task desc riptions, agent
bids, and milestone evaluations. Since these systems are inherently distributed
and need transparent memory logic, SHIMI oﬀers a structured alt ernative to
embedding-heavy retrieval methods that cannot guarantee exp lainability or de-
terministic matching.

14 Tooraj Helmi
5.5 Limitations and Threats to Validity
While SHIMI oﬀers notable improvements in decentralized memory ret rieval,
several limitations remain. First, the current implementation assum es a strictly
tree-based semantic structure. While eﬀective for many forms of knowledge or-
ganization, this design may struggle with concepts that require poly hierarchical
or graph-based representations, where entities belong to multiple overlapping
categories. Future work may explore generalizing SHIMI to accomm odate more
ﬂexible topologies.
Second, SHIMI’s semantic reasoning relies on language model-driven opera-
tions for generalization, similarity, and merging. While these operatio ns provide
high-qualityabstraction,theyintroduceadependencyonopaque modelbehavior
and lack formal guarantees. This makes it diﬃcult to audit or reprod uce certain
memory transformations. Incorporating symbolic constraints or logic-based rea-
soning may improve transparency.
Third, our evaluation is conducted in a controlled simulation environme nt.
Although it emulates decentralized conditions, it abstracts away re al-world con-
cerns such as network latency variability, partial node failure, or a dversarial
input. Additionally, entity descriptions and queries were synthetica lly generated
orcuratedtoreﬂectconceptualdiversity,but theymaynotfully capturedomain-
speciﬁc noise or inconsistency in real deployments.
Threats to validity include potential overﬁtting to speciﬁc query types
or entity descriptions. Because queries were designed to test sem antic depth,
performance may diﬀer on surface-level tasks. Moreover, the u se of simulated
nodes may underrepresent emergent behaviors in large-scale, he terogeneously
administered networks.
Despite these limitations, the results demonstrate strong alignmen t between
SHIMI’s theoretical design and its practical performance. Addre ssing the above
constraints will be crucial for robust deployment in open, real-wor ld systems.
6 Conclusion and Future Work
We introduced SHIMI, a semantic hierarchical memory index purpos e-built for
decentralized AI environments. Unlike traditional retrieval-augme nted genera-
tion (RAG) models that rely on surface-level similarity, SHIMI organ izes knowl-
edge semantically, enabling meaningful and interpretable retrieval even across
lexically diverse inputs. Through a series of simulation-based experim ents, we
demonstrated SHIMI’s superiority over ﬂat vector approaches a cross four criti-
cal dimensions: retrieval accuracy, traversal eﬃciency, synch ronization cost, and
scalability.
SHIMI’s architecture is well-aligned with the demands of distributed a gent
ecosystems. Its partial synchronization mechanism enables band width-eﬃcient
updates without sacriﬁcing consistency, and its tree-based stru cture supports
fast and explainable memory access. These properties make SHIMI particularly
suited for settings where agents are autonomous, memory is dyna mic, and in-
frastructure is decentralized.

Title Suppressed Due to Excessive Length 15
Lookingforward,severalpromisingdirectionsremain.First,wepla ntoevalu-
ate SHIMI under real-world deployment scenarios, including live agen t networks
withadversarialorasynchronousbehavior.Second,weaimtogen eralizeSHIMI’s
semantic model beyond hierarchical structures to support non- tree-basedgraphs
and cyclic ontologies. Third, integrating lightweight on-device LLMs f or local
semantic resolution may further reduce network latency and expa nd SHIMI’s
applicability in edge environments. Lastly, we envision formalizing SHIM I’s se-
mantic generalization and merge procedures as part of an interope rable protocol
layer for decentralized memory systems.
Together, these directions extend SHIMI from a proof-of-conc eptinto a foun-
dational infrastructure for next-generation distributed AI.
References
1. Auer, S., Bizer, C., Kobilarov, G., Lehmann, J., Cyganiak , R., Ives, Z.: Dbpedia:
A nucleus for a web of open data. In: International Semantic W eb Conference. pp.
722–735. Springer (2007)
2. Becker, G.: Merkle signature schemes, merkle trees and th eir cryptanalysis. Ruhr-
University Bochum, Tech. Rep 12, 19 (2008)
3. Benet, J.: Ipfs - content addressed, versioned, p2p ﬁle sy stem.
https://ipfs.tech/ipfs-paper/ (2014)
4. Bernhardsson, E.: Annoy (approximate nearest neighbors oh yeah) (2015),
https://github.com/spotify/annoy
5. Bloom, B.H.: Space/time trade-oﬀs in hash coding with all owable errors. Commu-
nications of the ACM 13(7), 422–426 (1970)
6. Contributors, O.: Orbitdb: Serverless, peer-to-peer da tabase (2021),
https://github.com/orbitdb/orbit-db
7. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre -training of deep bidi-
rectional transformers for language understanding. In: Pr oceedings of the 2019
Conference of the North American Chapter of the Association for Computational
Linguistics (2019)
8. Fensel, D., S ¸im¸ sek, U., Angele, K., Huaman, E., K¨ arle, E., Panasiuk, O., Toma, I.,
Umbrich, J., Wahler, A.: Knowledge Graphs. Springer (2021)
9. Gao, L., Dai, Z., Chen, T., Fan, Z., Van Durme, B., Callan, J .: Complementing
lexical retrieval with semantic residual embedding (2021)
10. Graves, A., Wayne, G., Danihelka, I.: Hybrid computing u sing a neural network
with dynamic external memory. Nature 538(7626), 471–476 (2016)
11. Gruber, T.R.: Toward principles for the design of ontolo gies used for knowl-
edge sharing? International Journal of Human-computer Stu dies43(5-6), 907–928
(1995)
12. Hassan, S., Reijers, W., Wuyts, K., de Filippi, P.: Decen tralized autonomous or-
ganizations: Concept, model, and applications. In: IEEE In ternational Conference
on Blockchain (Blockchain). pp. 1–10. IEEE (2021)
13. Johnson, J., Douze, M., J´ egou, H.: Billion-scale simil arity search with gpus. IEEE
Transactions on Big Data (2019)
14. Lakkaraju, H., Kamar, E., Caruana, R., Leskovec, J.: Int erpretable & explorable
approximations of black box models. arXiv preprint arXiv:1 707.01154 (2017)

16 Tooraj Helmi
15. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin , V., Goyal, N., Fan, A.,
Chaudhary, A., Joshi, M., Schick, T., et al.: Retrieval-aug mented generation for
knowledge-intensive nlp tasks. Advancesin Neural Informa tion Processing Systems
33, 9459–9474 (2020)
16. Liebherr, M., G¨ oßwein, E., Kannen, C., Babiker, A., Al- Shakhsi, S., Staab, V., Li,
B., Ali, R., Montag, C.: Working memory and the need for expla inable ai–scenarios
from healthcare, social media and insurance. Heliyon 11(2) (2025)
17. Liu, J., Liu, Y., Chen, C., Zhang, Y., Li, K.: Blockchain e mpowered decentral-
ized multi-agent systems. IEEE Transactions on Systems, Ma n, and Cybernetics:
Systems (2021)
18. Reimers, N., Gurevych, I.: Sentence-bert: Sentence emb eddings using siamese bert-
networks. In: Proceedings of the 2019 Conference on Empiric al Methods in Natural
Language Processing (2019)
19. Shapiro, M., Pregui¸ ca, N., Baquero, C., Zawirski, M.: A comprehensive study of
convergent and commutative replicated data types. Researc h Report RR-7506,
INRIA (2011)
20. Sharma, S., Henderson, J., Ghosh, J.: Certifai: Counter factual explanations for ro-
bustness, transparency, interpretability, and fairness o f artiﬁcial intelligence mod-
els. arXiv preprint arXiv:1905.07857 (2019)
21. Speer, R., Chin, J., Havasi, C.: Conceptnet 5.5: An open m ultilingual graph of
general knowledge. Proceedings of the Thirty-First AAAI Co nference on Artiﬁcial
Intelligence (AAAI-17) (2017)
22. Taipalus, T.: Vector database management systems: Fund amental concepts, use-
cases, and current challenges. arXiv preprint arXiv:2309. 11322 (2023)
23. Tulving, E.: Episodic and semantic memory. Organizatio n of memory 1, 381–403
(1972)
24. Vendrov, I., Kiros, R., Fidler, S., Urtasun, R.: Order-e mbeddings of images and
language. International Conference on Learning Represent ations (ICLR) (2016)
25. Vrandeˇ ci´ c, D., Kr¨ otzsch, M.: Wikidata: A free collab orative knowledgebase. Com-
munications of the ACM 57(10), 78–85 (2014)
26. Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., Yu, P.S.: A c omprehensive survey
on graph neural networks. IEEE Transactions on Neural Netwo rks and Learning
Systems 32(1), 4–24 (2020)
27. Zheng, C., Tao, X., Dong, L., Zukaib, U., Tang, J., Zhou, H ., Cheng, J.C., Cui,
X., Shen, Z.: Decentralized artiﬁcial intelligence in cons truction using blockchain.
Automation in Construction 166, 105669 (2024)

ϬϱϭϬ ϭϱ ϮϬ Ϯϱ ϯϬ ϯϱ ϰϬ ϰϱ ϱϬ 
Ϭ ϱ ϭϬ ϭϱ ϮϬ Ϯϱ ϯϬ ĂƚĂ 
ĂƚĂ 