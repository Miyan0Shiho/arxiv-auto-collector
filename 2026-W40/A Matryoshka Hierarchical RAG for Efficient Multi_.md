# A Matryoshka Hierarchical RAG for Efficient Multi-Hop Question Answering

**Authors**: Gianluca Bonifazi, Christopher Buratti, Michele Marchetti, Federica Parlapiano, Giulia Quaglieri, Davide Traini, Domenico Ursino, Luca Virgili

**Published**: 2026-10-01 14:25:23

**PDF URL**: [https://arxiv.org/pdf/2610.01767v1](https://arxiv.org/pdf/2610.01767v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems for multi-hop Question Answering (QA) must balance retrieval quality with computational cost. This cost is incurred during indexing time, through the use of expensive Knowledge Graphs (KGs) or Large Language Models (LLMs) to generate summaries, or during querying, through iterative LLM-driven retrieval. To reduce it while maintaining retrieval quality, we present MatRAG, a hierarchical framework that combines RAG systems with Matryoshka Representation Learning (MRL). MatRAG addresses both kinds of cost by aligning the semantic hierarchy of a clustering structure with the nested structure of MRL. Specifically, it organizes the corpus of documents into a Directed Acyclic Graph (DAG) of clusters with progressively coarser granularity. Each level is indexed by a lower Matryoshka dimension. MatRAG pairs an iterative, top-down traversal of the DAG with an entity-driven mechanism that controls the hop budget and re-ranks candidates. We evaluated MatRAG on three standard multi-hop QA benchmarks against seven representative baselines. MatRAG outperforms its strongest competitors in terms of retrieval quality; furthermore, it reduces indexing costs by avoiding KG construction and LLM-based summarization, and lowers query-time costs through dimension-aware similarity.

## Full Text


<!-- PDF content starts -->

A Matryoshka Hierarchical RAG for Efficient Multi-Hop
Question Answering
Gianluca Bonifazia, Christopher Burattia, Michele Marchettia, Federica
Parlapianoa, Giulia Quaglieria, Davide Trainib, Domenico Ursinoa, Luca
Virgilia,∗
aPolytechnic University of Marche, Ancona, Italy
bUniversity of Modena and Reggio Emilia, Modena, Italy
Abstract
Retrieval-AugmentedGeneration(RAG)systemsformulti-hopQuestionAn-
swering (QA) must balance retrieval quality with computational cost. This
cost is incurred during indexing time, through the use of expensive Knowl-
edge Graphs (KGs) or Large Language Models (LLMs) to generate sum-
maries, or during querying, through iterative LLM-driven retrieval. To re-
duce it while maintaining retrieval quality, we present MatRAG, a hierarchi-
cal framework that combines RAG systems with Matryoshka Representation
Learning (MRL). MatRAG addresses both kinds of cost by aligning the se-
mantic hierarchy of a clustering structure with the nested structure of MRL.
Specifically, it organizes the corpus of documents into a Directed Acyclic
Graph (DAG) of clusters with progressively coarser granularity. Each level
is indexed by a lower Matryoshka dimension. MatRAG pairs an iterative,
top-down traversal of the DAG with an entity-driven mechanism that con-
trols the hop budget and re-ranks candidates. We evaluated MatRAG on
three standard multi-hop QA benchmarks against seven representative base-
lines. MatRAG outperforms its strongest competitors in terms of retrieval
quality; furthermore, it reduces indexing costs by avoiding KG construc-
tion and LLM-based summarization, and lowers query-time costs through
dimension-aware similarity.
∗Corresponding author
Email addresses:g.bonifazi@univpm.it(Gianluca Bonifazi),
c.buratti@pm.univpm.it(Christopher Buratti),michele.marchetti@univpm.it
(Michele Marchetti),f.parlapiano@pm.univpm.it(Federica Parlapiano),
g.quaglieri@pm.univpm.it(Giulia Quaglieri),davide.traini@unimore.it(Davide
Traini),d.ursino@univpm.it(Domenico Ursino),luca.virgili@univpm.it(Luca
Virgili)
arXiv:2610.01767v1  [cs.CL]  1 Oct 2026

Keywords:Retrieval-Augmented Generation, Multi-hop Question
Answering, Matryoshka Representation Learning, Large Language Models,
Efficient Retrieval
1. Introduction
Retrieval-Augmented Generation (RAG) systems have become the dom-
inant paradigm for grounding LLMs responses in external knowledge bases
and reducing hallucinations [24, 37].
A significant challenge in RAG is multi-hop Question Answering (QA),
where the correct answer requires aggregating information dispersed across
several documents linked by entities or relationships [33]. A dense similarity
between a query and a single document cannot capture chains of reasoning
that span multiple documents, and flat indices do not exploit the hierarchi-
cal semantic structure of the corpus [5]. Recent literature has proposed two
families of solutions to this problem. Graph-based approaches [21] construct
an explicit Knowledge Graph (KG) of entities and relationships offline and
explore it at query time via traversal [39], Personalized PageRank [8], or
LLM-based agents [32, 14]. Hierarchical approaches organize the corpus into
a multi-level index, enabling retrieval at multiple granularities [34, 30]. How-
ever, both families incur significant computational costs that limit their scal-
ability. Specifically, graph-based approaches incur offline costs through en-
tity linking and relationship extraction, while hierarchical approaches incur
offline costs through recursive LLM calls to summarize internal nodes [30].
Additionally, both families incur online costs through iterative LLM-driven
planning [13] and expensive KG traversals [32].
We argue that these limitations could be overcome by using the nested
representationsprovidedbyMatryoshkaRepresentationLearning(MRL)[15].
MRL produces embeddings whose low-dimensional prefixes retain useful in-
formation. This allows us to obtain representations of different dimension-
alities from a single embedding. We hypothesize that this property can be
exploited in a hierarchical clustering structure, where progressively coarser
levels require less fine-grained representations. Thus, the upper levels of the
hierarchy can be indexed using shorter Matryoshka prefixes while retaining
the full embedding dimension for individual documents.
Starting from this intuition, in this paper we present MatRAG, a Ma-
tryoshka-indexed, hierarchical RAG framework designed for multi-hop QA.
MatRAG encodes each document offline by using a Matryoshka embedding
and builds a hierarchical index through density-based clustering. This index
2

is structured as a Directed Acyclic Graph (DAG). The leaves of the DAG
store full-dimensional document embeddings, while the internal nodes store
centroids at progressively lower Matryoshka dimensions towards the coarser
levels. When a query is received, retrieval proceeds via a top-down traversal
of the DAG. During this process, the query embedding is truncated to the
dimension of the level being explored. This reduces the cost of similarity
computations at the coarser levels. An entity-aware iterative mechanism
controls how many documents are collected at each hop and how candidate
documents are re-ranked. This replaces the iterative LLM calls of plan-then-
retrieve approaches with a lightweight signal based on the entities found in
the query and in the already retrieved context.
We evaluated MatRAG against seven representative RAG baselines on
three multi-hop QA benchmarks, namely HotpotQA, 2WikiMultiHopQA,
and MuSiQue. Our experimental campaign assessed retrieval effectiveness,
answer quality, and computational efficiency. MatRAG achieved the highest
Exact Match (EM) and F1 scores across all three benchmarks, while consis-
tently improving Recall@5 over the document-retrieval baselines. Moreover,
it reduced the average response time compared to flat FAISS-based retrieval
and substantially lowered the indexing costs compared to graph-based and
hierarchical approaches.
The main contributions of this paper are:
•A resolution-aligned indexing strategy that aligns MRL dimensions
with index depth. This reduces the cost of query-time similarity at
coarser levels while preserving retrieval accuracy.
•An iterative retrieval mechanism that does not rely on graph-based
multi-hoptraversal, butratherprogressivelyexpandstheretrievedcon-
text through hierarchical search.
•An entity-aware re-ranking and drift-control mechanism, which uses
entities to prioritize candidate nodes and prevents retrieval drift across
iterations.
The rest of this paper is organized as follows: Section 2 reviews related
literature. Section 3 introduces MatRAG. Section 4 reports our experimen-
tal results. Section 5 discusses findings and limitations. Finally, Section 6
presents our conclusions and outlines directions for future work.
3

2. Related Work
This section describes related work and is divided into four subsections.
In particular, Subsection 2.1 discusses Retrieval-Augmented Generation and
multi-hop QA. Subsection 2.2 presents graph-based RAG approaches. Sub-
section 2.3 covers hierarchical RAG approaches. Finally, Subsection 2.4 in-
troduces MRL for retrieval.
2.1. Retrieval-Augmented Generation and Multi-Hop QA
A RAG [5] incorporates external knowledge bases into LLM generation
via an explicit retrieval step. The application of RAGs to multi-hop QA [33]
rangesfromsingle-passretrievaltoiterativepipelinesthatinterleaveretrieval
and reasoning. Recent approaches have also explored LLM-based re-ranking
strategiestoimprovethequalityofretrievedevidence[31]. Onesuchstrategy
involves decomposing the query into sub-steps and planning the subsequent
retrieval [13]. Instead, the strategy in [44] uses explicit logic trees to gen-
erate more interpretable answers. While these strategies improve quality,
they require repeated LLM calls at query time, and the cost increases with
each additional hop. Other approaches improve retrieval by enriching the
context used to identify relevant evidence [2]. A known failure mode of the
pipelines that append retrieved evidence to the query is query drift, which
was originally studied in Pseudo-Relevance Feedback (PRF) [46, 27]. In this
mode, the expansion of the query with the top-ranked documents can shift
the search away from the original information need. Recently, query drift
has reemerged in dense retrieval [17, 26] and iterative, multi-hop and graph-
based RAG [12, 16]. The classical mitigation strategy [27], which combines
the original and expanded queries in the similarity computation, is the basis
of our anchoring mechanism.
MatRAG fits into this framework while avoiding iterative LLM calls.
Instead of relying on LLM-based retrieval planning, it uses entity counts to
determine the retrieval budget at each iteration and entity overlap to re-
rank candidate documents. Additionally, it mitigates query drift through an
anchoring schedule that strengthens the original query as retrieval proceeds.
2.2. Graph-Based RAG
Graph-based approaches [21] construct a KG offline starting from the
entities and relationships extracted from documents. These approaches use
the KG’s structure to guide retrieval when a query is submitted. The main
distinction among these approaches is the granularity level with which the
graph is built and queried. GraphRAG [4] operates at the community level,
4

with communities detected by the Leiden algorithm [35], and employs multi-
level summaries for query-focused summarization. LightRAG [6] combines
this global view with a local, entity-based view through dual-level retrieval.
HippoRAG2 [8] operates at the entity level. It reframes retrieval as an asso-
ciative memory problem and uses Personalized PageRank to identify relevant
passages in one step. KGP [39] moves further away from entities by building
a graph of passages with edges based on semantic and structural relation-
ships traversed by an LLM-based agent. Another way of proceeding exploits
the LLM as a reasoning engine to navigate the graph step by step, as in
Think-on-Graph (ToG) [32] and Graph-CoT [14], or to extract task-relevant
substructures, as in SG-RAG [29] and KG2RAG [45]. These approaches per-
form well on multi-hop tasks but incur a significant indexing cost and depend
on entity linking and relationship extraction, which are noisy processes by
nature.
MatRAG only uses entities as a lightweight signal for budget control and
re-ranking, and it does not construct an explicit KG.
2.3. Hierarchical RAG
Hierarchical RAG approaches organize the corpus into tree-like struc-
tures, in which retrieval operates at various granularity levels [38, 22]. RAP-
TOR [30] builds the tree recursively, representing each internal node with
an LLM-generated summary of its child nodes. Several approaches build on
this paradigm by enriching the index with additional signals. For instance,
SiReRAG [42] indexes similar and related information together to support
multi-hop reasoning, and [11] integrates hierarchical knowledge to bridge the
gap between local and global contexts. Other works focus on how the hier-
archy itself is constructed. For instance, ArchRAG [38] groups documents
into communities with attributes, and TreeRAG [34] exploits the internal
structure of lengthy documents to generate a hierarchy. Despite their dif-
ferences, these approaches share two characteristics. First, they use a single
embedding dimension across all tree levels. Second, they rely on LLM calls
to derive representations of internal nodes.
MatRAG differs from these approaches in both respects. In fact, in-
ternal nodes are computed as centroids of their child nodes without LLM
intervention. Additionally, MatRAG uses lower embedding dimensions for
coarserclustersandprogressivelylargerdimensionsforfiner-grainedclusters,
retaining full-dimensional embeddings only for individual documents.
5

2.4. Matryoshka Representation Learning for Retrieval
MRL [15] introduces a multi-scale loss that renders the prefixes of an
embedding independently informative. Thus, a single model produces nested
representations that are truncated at deployment time to balance cost and
accuracy. This property has been combined with knowledge distillation for
denseretrieval[43],appliedtoRAGinlow-resourcelanguages[19],integrated
into a hybrid retriever for general RAG [18], used for similarity analysis
on trajectory embeddings [23], and leveraged for interpretable, hierarchical
clustering of multilingual news articles [9]. All these approaches use MRL to
select one deployment dimension that balances cost and accuracy. However,
they apply the chosen representation uniformly across the downstream task.
Unlike them, MatRAG proposes a structured use of MRL as an indexing
strategy for hierarchical clustering, where progressively shorter prefixes are
used at coarser levels of the hierarchy.
3. Proposed Approach
In this section, we present MatRAG, our hierarchical RAG framework
designed for multi-hop QA. Its behavior consists of two phases. During the
offline phase, MatRAG encodes the corpus using a Matryoshka embedding
model and builds a hierarchical index, in which internal nodes are stored
at progressively lower prefix dimensions. During the online phase, given a
query,MatRAGiterativelyretrievesdocumentsthroughatop-downtraversal
of the index and scores candidates with two complementary signals applied
at different stages of the traversal. Figures 1 and 2 illustrate the MatRAG
workflow.
3.1. Hierarchical Indexing
LetD={d 1, . . . , d N}be a corpus of documents. We encode each docu-
mentd i∈ Dusing an MRL embedding model [15], which produces a dense
vectore i∈RM. The MRL model organizes the information ine ialongL
nested levels of granularity, such that, for anyl∈ {1, . . . , L}, the firstm l
coordinates ofe iform a self-contained representation ofd iat the dimension
ml, withm 1< m 2<···< m L=M. Thus, the full vectore icontainsL
nested representations ofd i, obtained by considering prefixes of increasing
dimensionality. We compute each embeddinge ionce and store it at the full
dimensionM, since any lower-dimensional representation can be obtained
as a prefix of it on the fly. We also extract a set of named entitiesE ifrom
each documentd i∈ Dusing GLiNER [41], an open-schema NER model. We
compute the entity sets once and store them so that we can efficiently access
6

Coarse Clusters
Fine-Grained Clusters
All Documents
Entity ExtractorFigure 1: Hierarchical indexing (offline). The corpus is encoded using a Matryoshka em-
bedding model and organized from the bottom up into a hierarchical DAG via HDBSCAN
with overlapping cluster assignments. The internal nodes are progressively coarser clus-
ter centroids stored at shorter Matryoshka prefixes. The leaf layer retains full-dimensional
document embeddings. Named entities are extracted from each document and stored with
the embeddings. This allows for retrieval using both semantic and entity-level informa-
tion.
Eifor any documentd iat retrieval time. Entity counts and overlaps allow
us to control the iterative retrieval loop and re-rank candidate documents
(see Section 3.2).
With the document embeddings and entity sets in place, we organize
the corpus into a hierarchical index. This index exposes the semantic struc-
ture ofDat multiple levels of granularity through hierarchical clustering.
Specifically, we build a DAG whose nodes are partitioned intoLlevels.
We index the document level of the DAG byl=L, and the coarsest
level byl= 1. The nodes at the levelLrepresent the documents inD, while
each levell < Lgroups the nodes of the levell+ 1into clusters by applying
HDBSCAN [1].
To capture the fact that a node can be semantically related to multiple
clusters, we associate each node at the levell+ 1with itspnearest clusters
at the levell, as measured by the cosine similarity between the node and the
clustercentroids. Therefore,eachnodehasatmostpincomingedges, andthe
7

0.2 0.7 0.5 0.3 0.9 0.2 ....
....
....
....
....0.4 0.2 0.3 0.5 0.7 0.9Figure 2: Entity-budgeted retrieval (online). Retrieval involves an iterative, top-down
traversal of the hierarchy. Starting with the coarsest level, the query is scored against
cluster centroids. At each level, the search space narrows to the most promising cluster
until reaching the document level. The traversal score considers both similarity to the
original query and similarity to the cumulative query. The latter incorporates previously
retrieved documents to guide multi-hop reasoning while limiting query drift. At the leaves,
candidates are re-ranked using a score that blends semantic similarity and entity overlap
to prioritize documents that are both relevant and entity-consistent.
graph is acyclic because edges always link subsequent layers. Figure 1 shows
the structure of the DAG. Thep-nearest assignment provides redundant
coverageduringtop-downtraversal. Infact, anodethatsemanticallybelongs
to multiple clusters can be accessed from any of them, which makes the
descent robust to suboptimal centroid choices at a given level.
We couple the clustering procedure with the storage of the internal nodes
using the Matryoshka organization. Nodes at the levelLstore document
embeddings at the full dimensionM, while every internal node at the level
l < Lrepresents a centroid truncated at the reduced dimensionm l. The
centroid of a clusterCat the levellis computed as follows:
8

µ(l)
C= 
1
|C|X
c∈Cc!
ml(3.1)
The elements ofC, which are already stored at the dimensionm l+1, are
averaged directly. The symbol| mlindicates that the resulting centroid is
truncated to the dimensionm lfor storage.
This alignment is based on the idea that the coarser levels of the DAG
aggregate semantically heterogeneous content into broad clusters. The dis-
criminative information needed to rank candidates within these clusters is
coarse and can be captured by shorter Matryoshka prefixes. In contrast, we
reserve the full embedding dimension for the levelL, where the final ranking
of individual documents requires the most precise representation.
3.2. Entity-Budgeted Retrieval
We organize the retrieval procedure in the DAG as an iterative loop; each
iteration of this loop collects a number of documents equal to the entities
observed in the previous iteration. The idea is that the number of distinct
entities in the query and the retrieved context serves as an indicator of the
residual complexity of the multi-hop reasoning chain and defines the retrieval
budget. The latter indicates the maximum number of documents to retrieve.
Figure 2 illustrates this process.
LetGbe the hierarchical DAG built offline (Section 3.1), letKbe the
total retrieval budget defined by the user, and letqbe the input query. The
retrieval process is iterative and progressively expands the query context
throughentitiesextractedfrompreviouslyretrieveddocuments. Algorithm1
formalizes it.
During the first retrieval step (t= 1), the process relies solely on the en-
tities extracted from the input query. The functionRetrieve(Section 3.3)
performs a top-down traversal ofGand returns an initial set of retrieved
documents, denoted asD 1. These documents form the initial retrieved set
R1=D 1. The entities extracted from the retrieved documents are accu-
mulated into the entity set ˜E1, along with the query entities. The subset
of entities not present in the query defines the frontier entity setB 1, which
guides the next retrieval step.
For each subsequent iterationt >1, the retrieval process is guided by
the entities discovered in the previous iteration. Letq cdenote the cu-
mulative query, obtained by concatenating the original queryqwith the
text of all documents retrieved thus far. At iterationt,Retrievetra-
versesGusingq cand re-ranks the candidate documents based on the en-
9

tities inB t−1according to the multi-signal document score (Section 3.3).
Then, the process selects the topk tdocuments not contained in the cu-
mulative retrieved setR t−1and returns them as the new document set
Dt.kt= min (K− |R t−1|,max (1,|B t−1|))assures that the number of se-
lected documents is no greater thanK. The retrieved set is updated as
Rt=R t−1∪ Dt. The entities extracted fromD tare accumulated into the
cumulative entity set ˜Et; the entities present in ˜Etand not in ˜Et−1define the
new frontier setB t, which drives the next iteration.
Finally, the retrieved setR tis passed to an LLM together withqto
obtain the final answer.
Algorithm 1Entity-Budgeted Retrieval Loop
Require:K: a positive integer denoting the maximum number of documents to retrieve;q: a
query; ˆEq: the entities ofq;G: a DAG
1:R 0← ∅, ˜E0←ˆEq,B0←ˆEq,t←1
2:while|R t−1|< KandB t−1̸=∅do
3:q c←q∥L
d∈R t−1d ▷concatenateqwith the documents retrieved up to the iteration
t−1
4:k t←min (K− |R t−1|,max (1,|B t−1|))▷number of documents to retrieve according to
the remaining budget
5:D t←Retrieve(G, q, q c,Rt−1, kt)
6: ˜Et←˜Et−1∪ S
d∈D tEd
▷accumulate entities from the retrieved documents
7:R t← R t−1∪ Dt
8:B t←˜Et\˜Et−1
9:t←t+ 1
10:end while
11:t←t−1
12:returnR t
3.3. Multi-Signal Document Scoring
Multi-signal document scoring is performed by theRetrievefunction.
As discussed in Section 3.2, this function traversesGusingq cand re-ranks
the candidate documents according to the entities ofB t−1. To perform this
task, it ranks candidates using two complementary scores, i.e.,S l(x)and
R(d). The traversal scoreS l(x)combines similarity to the original query
(which acts as an anchor against query drift) with a contribution from the
cumulative query (which provides context from documents retrieved thus
far). Once the traversal reaches the levelL, the re-ranking scoreR(d)refines
theorderingofthecandidates. Forthispurpose,R(d)incorporatesanentity-
overlap signal that rewards documents whose entities are coherent with those
introduced at the previous iterations. Algorithm 2 details theRetrieve
function.
10

Algorithm 2Retrievefunction
Require:G: a DAG;q: a query;q c: a cumulative query;R t−1: a set of documents;k ta
non-negative integer denoting the number of documents to retrieve at iterationt
1:N ←FirstLayer(G)▷centroid vectors at level1
2:forl= 1toL−1do
3:foreach centroid vectore i∈ Ndo
4:ComputeSim(q, e i, l)andSim(q c, ei, l)by applying Equation 3.2
5:ComputeS l(ei)by applying Equation 3.3
6:end for
7:e x←arg max ei∈NSl(ei)
8:N ←Children(e x,G)▷vectors at levell+1reachable frome x
9:end for▷At this point,Ncontains the embeddings of the candidate documents at levell
10:S ← ∅▷set of (documents, score) pairs
11:foreache i∈ Nsuch thatd i/∈ Rt−1do
12:ComputeSim(q, e i, L)andSim(q c, ei, L)by applying Equation 3.2
13:ComputeS L(ei)by applying Equation 3.3
14:ComputeJ(d i)by applying Equation 3.5
15:ComputeR(d i)by applying Equation 3.6
16:S ← S ∪ {(d i, R(d i))}
17:end for
18:returnTop(S, k t)▷return the firstk tdocuments inSranked byR(d i)
3.3.1. Computation of the traversal score
For each levellof the DAG, the candidate nodes are ranked by a function
Sl(·), which combines contributions from the original queryqand the cumu-
lative queryq c. In order to defineS l(·), we first introduce the level-aware
similarity function as follows:
Sim(s, e i, l) = cos(e s|ml, ei)(3.2)
Here,sis a text,e sis its embedding, ande s|mlis them l-length prefix
ofes. The second argument of the function is the embeddinge istored at
the levellof the DAG. Iflis an internal level, thene iis the embedding of a
centroid atl; instead, iflis a leaf, thene iis the embedding of a document
inD. In both cases,e iis already stored as anm l-length vector, so it does
not require truncation. The similarity functionS l(·)returns a score that
combines two contributions. The first contribution is the similarity ofe ito
the original queryq, which anchors the search to the original information
need [27]. The second contribution is the similarity ofe ito the cumulative
queryq c, which expandsqwith the content of the documents retrieved thus
far via PRF [46]:
Sl(ei) =β t·Sim(q, e i, l) + (1−β t)·Sim(q c, ei, l)(3.3)
where the anchoring coefficientβ tis defined as follows:
11

βt=|Rt−1|
K(3.4)
Inthefirstiteration,q c=qandβ t= 0,whichreducesS l(ei)toSim(q, e i, l).
As documents are collected,q cincorporates their content progressively and
may drift away from the original information need. Concurrently,β tin-
creases, shifting the weight towardqand preventing semantic drift.
The traversal begins at the coarsest level (i.e.,l= 1) and continues
downward. At each levell, we apply the functionS l(·)to each centroid
atl. Then, we select the centroide xoflwith the highest value ofS l(·).
Afterwards, we restrict the candidate set to the nodes at the levell+ 1
that are connected to the node corresponding toe x. Due to thep-nearest
assignment introduced in Section 3.1, each node at the levell+ 1can be
reached from up topnodes at the levell. Thus, the traversal can tolerate
suboptimal centroid choices without losing relevant candidates.
We repeat this process until the levelLis reached. At this level, the
candidates are documents inD, and their embeddings are stored at the full
dimensionM. Since each levell < Loperates at a lower dimensionm l< M
rather than the full dimensionM, and since the candidate set is progres-
sively restricted as the traversal proceeds, the similarity computations at
the coarser levels are cheaper and performed over a smaller set of candi-
dates. This reduces the overall cost compared to a flat search over the full
corpus at the dimensionM.
3.3.2. Computation of the re-ranking score
After traversing to the levelL, we re-rank the candidates using the set of
newentitiesB t−1(seeSection3.2). Tothisend, weusetheJaccardcoefficient
to measure the overlap between the set of entitiesE iof a candidate document
diand the set of new entitiesB t−1as follows:
J(di) =|Bt−1∩ Ei|
|Bt−1∪ Ei|(3.5)
The final scoreR(d i)of a documentd iat the levelLcombines semantic
similarity and entity overlap as follows:
R(di) =α·S L(ei) + (1−α)·J(d i)(3.6)
whereαcontrolstherelativeimportanceoftheentitysignal. TheJaccard
coefficient ranges in the real interval[0,1]and is therefore stable with respect
to the scale of cosine similarity. TheRetrievefunction returns the top
12

documents, ranked byR(d i), within the budgetKdefined in Section 3.2.
These documents form the setD tand are then accumulated intoR tby the
retrieval loop.
4. Experimental Campaign
This section presents an experimental evaluation of MatRAG. Specifi-
cally, Subsection 4.1 describes the experimental setup. Subsection 4.2 ex-
amines the impact of the main hyperparameters. Subsection 4.3 reports
the main results. Finally, Subsection 4.4 presents an ablation study that
evaluates the contribution of each component of MatRAG.
4.1. Experimental Setup
In this section, we describe the setup used during our experimental cam-
paign. Specifically, we focus on the datasets used, the baselines selected for
comparison, some implementation choices for MatRAG, and the evaluation
metrics.
4.1.1. Datasets
We evaluated MatRAG on three standard multi-hop QA benchmarks,
namely HotpotQA [40], 2WikiMultiHopQA (2Wiki) [10], and MuSiQue [36].
Each dataset requires aggregating evidence from multiple documents to an-
swer a question, though datasets differed in the type of reasoning involved.
In fact, HotpotQA covers bridge and comparison questions, 2Wiki focuses
on compositional reasoning over Wikipedia, and MuSiQue is designed to re-
sist shortcut-based retrieval. All three benchmarks provide gold answers,
i.e., reference strings against which generated answers are evaluated. The
benchmarks also provide gold supporting documents, which are subsets of
thecorpusdocumentscontainingtheevidencenecessarytoanswereachques-
tion. We used these documents to compute retrieval metrics.
The document corpora contain9,811documents for HotpotQA,6,119
documents for 2Wiki, and11,656documents for MuSiQue. While previous
studies typically relied on a single sample of1,000questions [7, 25], we
drew five independent samples of1,000questions each to reduce variability
in question selection. We evaluated all methods on the same five samples,
and we report the mean and standard deviation of each metric across them.
We selected hyperparameter values using a separate validation set of 500
questions per benchmark that was disjoint from the test split.
13

4.1.2. Baselines
WecomparedMatRAGwithsevenrepresentativeRAGsystemsthatspan
the main families of approaches discussed in Section 2. As a reference for
flat retrieval, we included NaiveRAG, which performs dense retrieval over
a FAISS1flat index. As a representative of the hierarchical family, we in-
cluded RAPTOR [30]. The remaining baselines covered graph-based and
reasoning-augmented approaches; they are: GraphRAG [4], LightRAG [6],
HippoRAG2 [8], KGP [39], and ToG [32]. For all baselines, we followed the
original implementations and adopted the hyperparameter settings reported
in the papers introducing them. To ensure a fair comparison, all baselines
used the same generative model (gemma3:27b-it-qat2via Ollama) for an-
swer generation.
4.1.3. Implementation Details
All systems that require dense retrieval, including MatRAG, use the
nomic-embed-text-v1.5 [20] as the embedding model. This model natively
supports MRL at the following dimensions:{64,128,256,512,768}. We
adopteditastheMatryoshkadimensionscheduleforMatRAG,settingL= 5
andM= 768. We stored the five levels of the DAG at the corresponding di-
mensions. Documents at the leaves maintained the full dimensionM= 768,
while internal clusters used progressively shorter prefixes toward the coars-
est level, which was stored atm 1= 64. We selected the values of the total
retrieval budgetK, the similarity weightα, and the DAG overlapping factor
pvia the hyperparameter analysis reported in Section 4.2. In particular,
based on this analysis, we setK= 10,p= 2, andα= 0.5consistently
across all datasets. We conducted all experiments on a server equipped with
an NVIDIA A100 GPU (40 GB VRAM) and 128 GB of system RAM. The
interested reader can find the implementation code at the following link:
https://anonymous.4open.science/r/MatRAG.
4.1.4. Metrics
We evaluated the quality of the answers using Exact Match (EM) and
token-level F1, computed against the gold answers [40, 10, 36]. To mea-
sure retrieval quality independently of the generative step, as in previous
studies [8, 7], we employed Recall@2 (R@2) and Recall@5 (R@5). For each
question, R@2 (resp., R@5) measures the number of gold supporting docu-
ments found among the top two (resp., five) retrieved documents divided by
1https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatIP.html
2https://ollama.com/library/gemma3:27b-it-qat
14

the total number of gold supporting documents for that question. We aver-
aged these values across all questions in a dataset. We only computed these
retrieval metrics for MatRAG, HippoRAG2, LightRAG, and NaiveRAG, as
these approaches retrieve documents directly from the corpus. The remain-
ing baselines retrieve units not directly comparable to corpus documents.
These include the node summaries of RAPTOR, the community summaries
of GraphRAG, the passages of KGP (which have a different level of detail
than the gold supporting documents), and the KG entities and relationships
of ToG. For these systems, the gold document overlap required by R@2 and
R@5 was undefined.
In addition to assessing retrieval and answer quality, we evaluated the
efficiency of each system based on indexing time (Idx), response time (Res),
andLLMcontextsize(Tok). Indexingtimeistheamountoftime, inseconds,
required to build the index from the corpus. Response time is the average
time, in seconds, for each retrieval and answer generation. LLM context size
is the average number of tokens passed to the generative model per question.
4.2. Hyperparameter Analysis
In this section, we describe the analyses we performed to tune the hyper-
parameter values. To avoid tuning on the test data, we ran our analyses on
a separate validation set of 500 questions for each dataset. We sampled this
set disjointedly from the questions used for the test data (see Section 4.1).
All reported values are means over this validation set.
First, we evaluated the effect of the retrieval budgetKby considering
two values of this parameter:K= 5andK= 10. The results are reported in
Table 1. In each column of this and the following tables, the optimal value is
in bold, and the suboptimal value is underlined. An upward-pointing (resp.,
downward-pointing) arrow next to a metric indicates that its optimal value is
the highest (resp., lowest). The analysis of this table shows that increasing
Kfrom 5 to 10 substantially improves the EM and F1 scores across all
datasets. As for EM, there are gains of 16.34% on HotpotQA, 19.25% on
2Wiki and 21.36% on MuSiQue. Regarding F1, we observe improvements of
8.56%onHotpotQA,19.34%on2Wikiand21.45%onMuSiQue. Conversely,
we observe an increase in Tok, which increased by 108.00% on HotpotQA,
104.03% on 2Wiki, and 102.26% on MuSiQue. These results confirm that
multi-hop QA accuracy improves with larger contexts because additional
documents help cover longer reasoning chains. However, this comes at the
expense of efficiency. We chose to prioritize accuracy over efficiency and set
Kto 10. At the same time, we did not setKto a greater value so as to not
dramatically worsen efficiency.
15

Table 1: Values of Exact Match (EM), token-level F1 (F1) and LLM context size (Tok)
obtained by MatRAG on the three benchmarks forK= 5andK= 10. The values
represent the results computed on the validation set consisting of 500 different questions.
For each column, the optimal value is shown in bold.
HotpotQA 2Wiki MuSiQue
KEM↑F1↑Tok↓EM↑F1↑Tok↓EM↑F1↑Tok↓
551.40 65.1732542.60 49.8529820.60 29.97354
1059.80 70.7567650.80 59.4960825.00 36.40716
After analyzingK, we proceeded to analyze the impact ofp, which con-
trols the number of cluster assignments per node in the DAG. Table 2 shows
the obtained results. From the analysis of this table, we can see that the best
performance is consistently achieved withp= 2, improving F1 (resp., EM)
by 3.04% (resp., 5.65%) on HotpotQA, 4.85% (resp., 7.63%) on 2Wiki, and
8.20% (resp., 9.65%) on MuSiQue compared top= 1. Increasingpbeyond
2 does not provide further gains and, in some cases, it slightly degrades per-
formance (e.g., we can observe a 0.11% decrease in F1 on HotpotQA when
increasingpfrom 2 to 3). Analogous conclusions can be drawn for R@2 and
R@5. These results suggest that a limited amount of redundancy is sufficient
to ensure robust traversal without introducing excessive noise.
Table 2: Values of EM, F1, Recall@2 (R@2) and Recall@5 (R@5) obtained by MatRAG
on the three benchmarks forpranging from 1 to 4. The values represent the results
computed on the validation set consisting of 500 different questions. For each column, the
optimal value is shown in bold.
HotpotQA 2Wiki MuSiQue
pEM↑F1↑R@2↑
R@5↑EM↑F1↑R@2↑
R@5↑EM↑F1↑R@2↑
R@5↑
156.60 68.6673.80
88.7047.20 56.7463.45
78.3522.80 33.6442.60
55.87
259.80 70.7575.80
91.1050.80 59.4966.05
81.8525.00 36.4043.78
57.40
359.60 70.6775.60
90.3050.00 59.1965.35
80.2524.00 35.6642.92
57.22
459.00 70.3674.20
89.5049.00 58.1965.05
79.7523.20 33.5242.37
56.98
Finally, we examined the effect ofα, which regulates the contribution of
the entity-aware re-ranking. Table 3 shows the obtained results. It reveals
that the best overall results are obtained atα= 0.50, which provides a
16

balanced combination of semantic similarity and entity overlap. Relying
only on similarity-based ranking (α= 1.00) returns poorer performance. In
fact, settingα= 0.50improves F1 (resp., EM) by 4.63% (resp., 3.10%) on
HotpotQA, 11.59% (resp., 10.92%) on 2Wiki, and 22.02% (resp., 34.41%)
on MuSiQue, compared to settingαto 1.00. Relying exclusively on entity
overlap (α= 0.00) also degrades performance. In fact, settingαto 0.50
improvesF1(resp., EM)by4.78%(resp., 7.17%)onHotpotQA,2.13%(resp.,
0.79%)on2Wiki, and3.79%(resp., 4.16%)onMuSiQue, comparedtosetting
αto 0.00. Whenαis set to 0.25 (resp., 0.75) rather than 0.00 (resp., 1.00),
results are sometimes better. However, they continue to be worse than when
αis set to 0.50. R@2 and R@5 follow the same trends. Overall, these results
suggest that neither semantic similarity nor entity overlap is sufficient on its
own and that a balanced combination of the two, withαset to0.50, provides
the most robust performance across the three benchmarks.
Table 3: Values of EM, F1, R@2 and R@5 obtained by MatRAG on the three benchmarks
forαset to 0.00, 0.25, 0.50, 0.75, and 1.00. The values represent the results computed
on the validation set consisting of 500 different questions. For each column, the optimal
value is shown in bold.
HotpotQA 2Wiki MuSiQue
αEM↑F1↑R@2↑
R@5↑EM↑F1↑R@2↑
R@5↑EM↑F1↑R@2↑
R@5↑
0.0055.80 67.5274.70
86.8050.40 58.2564.70
81.1524.00 35.0743.80
56.70
0.2557.20 68.4874.30
90.3050.20 58.7965.80
80.7023.40 34.7943.03
56.22
0.50 59.80 70.7575.80
91.1050.80 59.4966.05
81.8525.00 36.4043.78
57.40
0.7558.80 69.8176.40
90.2048.40 56.7464.70
76.6019.80 30.2043.70
55.75
1.0058.00 67.6274.30
88.9045.80 53.3163.45
73.8018.60 29.8342.61
53.37
4.3. Comparison Results
We first computed the retrieval performance of MatRAG and the base-
lines. The results are reported in Table 4. In each column of this and
the following tables, the optimal value is in bold, and the suboptimal value
is underlined. The analysis of Table 4 shows that MatRAG consistently
achieves optimal results across all benchmarks in most settings. On Hot-
potQA, it obtains the highest R@2 and R@5 scores, surpassing the strongest
17

baselines. Specifically, itoutperformsNaiveRAGby2.98%onR@2andLigh-
tRAG by 5.99% on R@5. On 2Wiki, MatRAG achieves the best R@5 score,
outperforming HippoRAG2 by 1.89%. However, its R@2 is slightly below
that of NaiveRAG, with a difference of 0.51%. On the more challenging
MuSiQue benchmark, MatRAG delivers the strongest improvements, sur-
passing NaiveRAG, the best baseline, by 14.48% on R@2 and 13.37% on
R@5. These results suggest that MatRAG is particularly effective in com-
plex, multi-hop retrieval scenarios.
Table 4: Retrieval performance on the three multi-hop benchmarks measured by Recall@2
(R@2) and Recall@5 (R@5). The results are reported only for systems that retrieve corpus
documents directly. The values represent the means and standard deviations computed
over 5 samples, each of which contains 1,000 different questions. For each column, the
optimal value is shown in bold and the suboptimal value is underlined.
HotpotQA 2Wiki MuSiQue
R@2↑R@5↑R@2↑R@5↑R@2↑R@5↑
HippoRAG2 56.40±.4282.95±.5762.50±.8677.80±.31 34.60±.4450.30±.39
LightRAG 68.05±.3784.25±.29 58.90±.8268.70±.9734.80±.4648.75±.40
NaiveRAG 70.55±.54 84.10±.2863.40±.5869.55±.8537.50±.43 50.35±.37
MatRAG72.65±.8989.30±.4663.08±.65 79.27±.8142.93±.4057.08±.33
Next, we analyzed whether these improvements in retrieval translate into
better performance in downstream QA. To this end, we computed the EM
and F1 scores. The results are reported in Table 5. As can be seen from
this table, MatRAG achieves the best performance across all datasets and
metrics. On HotpotQA, it improves upon NaiveRAG by 14.04% in EM and
LightRAG by 10.68% in F1. On 2Wiki, the gains over the strongest base-
line, HippoRAG2, are 5.71% in EM and 14.89% in F1. Finally, on MuSiQue,
MatRAG achieves the highest scores again, improving upon HippoRAG2 by
4.58% in EM and 12.94% in F1. Overall, these results demonstrate that im-
provements in retrieval quality lead to better QA performance, particularly
in more challenging multi-hop settings.
Afterward, we evaluated the indexing time for each benchmark. The re-
sults are shown in Table 6. As reported in this table, MatRAG is faster than
most graph-based and structured approaches. Compared to HippoRAG2,
it reduces indexing time by over 99.60% across all datasets. It also outper-
formsothermethods, suchasGraphRAGandLightRAG,byseveralordersof
magnitude. However, it remains slower than NaiveRAG, though it achieves
better retrieval quality on most metrics. This efficiency gain primarily stems
from the fact that many competing approaches (e.g., GraphRAG and RAP-
18

Table 5: QA performance on the three multi-hop benchmarks, measured by EM and
F1 computed against the gold answers. The values represent the means and standard
deviations computed over 5 samples, each of which contains 1,000 different questions. For
each column, the optimal value is shown in bold and the suboptimal value is underlined.
HotpotQA 2Wiki MuSiQue
EM↑F1↑EM↑F1↑EM↑F1↑
HippoRAG2 52.40±.5864.10±.6249.85±.54 55.20±.73 26.20±.80 33.55±.39
RAPTOR 39.50±.8155.35±.9332.10±.7437.90±.4114.15±.4823.85±.93
KGP 23.20±.6633.10±.4412.05±.8913.80±.7011.75±.8218.90±.61
ToG 21.30±.7227.45±.8515.20±.9117.75±.766.40±.699.25±.59
GraphRAG 30.65±.6340.60±.788.25±.569.10±.511.10±.322.30±.38
LightRAG 52.15±.8166.55±.69 45.35±.4250.80±.3721.85±.9331.80±.41
NaiveRAG 53.05±.38 64.50±.9544.10±.3248.15±.4920.65±.3929.55±.30
MatRAG60.50±.3773.66±.2952.70±.3563.42±.3127.40±.3837.89±.34
TOR) require expensive document summarization steps during graph con-
struction, and that others (e.g., HippoRAG2 and LightRAG) necessitate
a complex KG construction phase. In contrast, MatRAG relies on sim-
ple cluster-level node representations obtained through node averaging, thus
avoiding costly preprocessing.
Table 6: Indexing time (Idx) in seconds across the three benchmarks. The values represent
the means and standard deviations computed over 5 independent runs of the indexing
procedure on the full corpus of each benchmark. For each column, the optimal value is
shown in bold, and the suboptimal value is underlined.
HotpotQA 2Wiki MuSiQue
HippoRAG2 312,441±1,842123,187±1,253321,905±1,976
RAPTOR 28,763±41216,724±31836,204±487
KGP 1,941±631,478±514,213±74
ToG 141,820±93469,340±721148,573±1,102
GraphRAG 661,392±2,34175,614±843197,840±1,587
LightRAG 728,471±2,813347,605±1,934598,317±2,645
NaiveRAG98±471±3109±5
MatRAG 316±8 411±11 528±14
In the last experiment, we computed the response time in seconds and
the average context length in tokens for each benchmark. The results are
reported in Table 7. The analysis of this table shows that the MatRAG’s
most notable advantage lies in its response time. In fact, MatRAG achieves
the fastest response time of all methods, outperforming the fastest base-
line, NaiveRAG, by 47.44% on HotpotQA, 35.82% on 2Wiki, and 35.14% on
MuSiQue. This is particularly significant because NaiveRAG relies on FAISS
19

for similarity searches, and FAISS is already a highly optimized index. Out-
performing this baseline confirms that dimension-aware traversal of the DAG
provides a genuine computational advantage over flat index searches at full
dimensionality. Compared to more complex methods, such as HippoRAG2,
MatRAG achieves an even greater reduction, exceeding 96.40% across all
benchmarks.
Regarding context length, MatRAG maintains moderate token usage.
Although it uses slightly more tokens than NaiveRAG (7.80% on HotpotQA
and 15.78% on 2Wiki), it is substantially more efficient than graph-based
methods, such as LightRAG and GraphRAG. The disproportionately large
context of these approaches derives from the KG-based retrieval pipelines,
which may include extracted triples into the prompt alongside the retrieved
documents. MatRAG avoids this overhead entirely because it does not use a
KG, reducing context size by over 79.33% compared to GraphRAG and over
86.51% compared to LightRAG on HotpotQA. ToG and HippoRAG2 obtain
the smallest contexts by refining the evidence before generation. ToG itera-
tively prunes candidate entities and relations through LLM calls, and Hip-
poRAG2 filters and re-scores the retrieved evidence through its graph-based
pipeline. However, this compression does not remove the cost of selecting the
evidence; rather, it shifts the cost upstream. Since Tok only counts the to-
kens passed to the final generative model, it does not capture the additional
LLM calls and graph operations performed during retrieval, which result in
response times that are more than an order of magnitude higher than those
of MatRAG.
Table 7: Efficiency comparison across the three benchmarks in terms of response time
(Res), measured in seconds, and Tok. The values represent the means and standard
deviations computed over 5 samples, each composed of 1,000 different questions. For each
column, the optimal value is shown in bold and the suboptimal value is underlined.
HotpotQA 2Wiki MuSiQue
Res↓Tok↓Res↓Tok↓Res↓Tok↓
HippoRAG2 76.84±2.83591±12 80.73±3.91624±14 83.12±2.97671±15
RAPTOR 15.23±.41854±1814.89±.38843±1715.91±.44856±19
KGP 31.40±1.572,478±3126.55±1.49359±938.04±1.632,298±28
ToG 56.81±1.74124±640.67±1.62112±549.93±1.68121±6
GraphRAG 15.34±.393,812±4216.08±.43871±1916.20±.451,398±24
LightRAG 20.87±.985,841±5320.44±.965,512±4921.03±.917,268±61
NaiveRAG 3.52±.18 731±133.88±.21 704±124.61±.24 814±15
MatRAG1.85±.12788±102.49±.15815±112.99±.17813±12
These results, combined with those in Tables 4 and 5, suggest that Ma-
20

tRAG successfully strikes a balance between computational efficiency and
contextual richness. In fact, it guarantees a considerably faster retrieval
while identifying the most informative documents.
4.4. Ablation Analysis
In this section, we assess the two core components of MatRAG (i.e., the
anchoringstrategyandtheMatryoshkaindexingscheme)bycomparingthem
with simpler alternatives. Unlike the hyperparameter analysis, which was
performed on the validation set to avoid tuning on test data, this ablation
study does not involve hyperparameter selection. Therefore, we employed
the same test samples used in Section 4.3 to ensure comparability with the
main results.
We started by analyzing the effect of the query anchoring strategy, con-
trolled byβ t. Specifically, we considered three alternatives, namely:(i)us-
ingonlytheoriginalquery(q);(ii)usingonlythecumulativeexpandedquery
(qc); and(iii)using an adaptive combination of the two. The results are re-
ported in Table 8. This table shows that relying exclusively on either signal
is suboptimal. Using only the original query (β t= 1.00) limits the ability to
incorporate newly discovered contexts, thereby undermining multi-hop per-
formance. Conversely, using only the expanded query (β t= 0.00) introduces
semantic drift because irrelevant information may accumulate across itera-
tions. The formulation proposed in MatRAG (see Equation 3.4) balances
these two contributions dynamically by gradually increasing the weight of
the original query to maintain alignment with the initial information need.
It achieves the optimal trade-off across all datasets and metrics. For in-
stance, it increases F1 by1.95%compared to the base variant (β t= 1.00)
on HotpotQA and by11.34%on 2Wiki. It also improves F1 by4.75%over
the expanded-query variant (β t= 0.00) on MuSiQue. These results high-
light the importance of managing query drift in iterative retrieval and show
that the proposedβ tscheduling provides a consistent and effective scoring
mechanism for different levels of reasoning complexity.
We then examined whether using shorter Matryoshka prefixes at coarser
levels instead of the full embedding dimension would result in a hierarchy
with lower-quality clusters. To do so, we ran HDBSCAN on all three bench-
marks at each levellof the hierarchy, from the coarsest clusters (l= 1)
down to the documents (l=L) using the truncated Matryoshka prefixes em-
ployed by MatRAG. We compared the resulting clusters to those obtained by
running the same procedure on full-dimensional embeddings. We used the
Davies-Bouldin (DB) index [3] and the Silhouette (Sl) score [28] as internal
clustering quality measures. The DB index quantifies the average similarity
21

Table 8: Values of EM, F1, R@2 and R@5 obtained by MatRAG on the three benchmarks
forβ tset to 1.00,|Rt−1|
Kand 0.00. The values represent the means computed over 5
samples, each composed of1,000different questions. For each column, the optimal value
is shown in bold.
HotpotQA 2Wiki MuSiQue
βtEM↑F1↑R@2↑
R@5↑EM↑F1↑R@2↑
R@5↑EM↑F1↑R@2↑
R@5↑
1.00 59.22 72.2572.10
88.7050.90 56.9662.05
76.7525.70 35.9241.30
55.52
|Rt−1|
K60.50 73.6672.65
89.3052.70 63.4263.08
79.2727.40 37.8942.93
57.08
0.00 58.70 71.2472.05
87.7251.24 62.3162.58
78.8126.42 36.1741.17
56.31
between each cluster and its most similar clusters. Lower DB values indicate
betterseparatedandmorecompactclusters. TheSlscoremeasureshowclose
each point is to its own cluster compared to the neighboring ones. Higher
Sl values denote more coherent clusters. Table 9 reports the comparison.
Across all benchmarks, the two strategies produce similar clustering quality
at every level. The Matryoshka prefixes even achieve slightly better DB and
Sl scores in most combinations of levels and datasets. This demonstrates
that truncating the representations toward coarser levels does not degrade
the quality of the clusters that form the hierarchy. Therefore, the efficiency
gains in response time obtained through the usage of Matryoshka prefixes
and discussed in Section 4.3 do not come at the expense of clustering quality.
Finally, Tables 10 and 11 report the comparison of Matryoshka’s trun-
cated and full-dimensional embeddings during retrieval. Across all available
benchmarks, the Matryoshka configuration achieves better or comparable
performanceintermsofQAmetricsandRecallwhilereducingresponsetime.
These results suggest that truncating embeddings to a lower-dimensional
subspace, as enabled by MRL, does not negatively impact retrieval qual-
ity. This behavior can be attributed to the way Matryoshka embeddings are
trained. Indeed, themostsemanticallyrelevantinformationisencodedinthe
leading dimensions, efficiently concentrating the signal and reducing noise in
the higher-dimensional components. Consequently, using only the truncated
prefix of the embeddings acts as a form of implicit regularization, discarding
dimensionsthatprovidelittlediscriminativeinformationandcouldintroduce
spurious similarity signals. We expect this effect to be particularly relevant
for the clustering step. Density-based algorithms such as HDBSCAN rely on
distances between points. In high-dimensional spaces these distances tend
22

Table9: ValuesoftheDavies-Bouldinindex(DB)andtheSilhouettescore(Sl)forthelevel
land embedding dimension across benchmarks. Them lcolumn indicates the embedding
dimension used for clustering with full-dimensional and Matryoshka embeddings. For each
column, the optimal value is shown in bold.
HotpotQA 2Wiki MuSiQue
l m lDB↓Sl↑DB↓Sl↑DB↓Sl↑
1768 0.8667 0.4484 0.7313 0.5512 0.8362 0.4492
640.8630 0.4628 0.7198 0.5536 0.7557 0.5406
2768 0.8342 0.4663 0.75030.5085 0.75730.4746
1280.7616 0.4749 0.73890.5059 0.76610.4860
3768 0.7428 0.52060.66080.5390 0.7330 0.4876
2560.7209 0.52980.68900.5544 0.6642 0.5420
4768 0.6294 0.56010.58180.5755 0.6408 0.5561
5120.6131 0.56710.58380.5784 0.6315 0.5567
5768 0.5695 0.5871 0.5460 0.5849 0.5994 0.5780
768 0.5695 0.5871 0.5460 0.5849 0.5994 0.5780
to concentrate, making density estimates and cluster separation less reli-
able. In MatRAG, the dimensionality decreases progressively towards the
coarser levels, so clustering is performed in lower-dimensional spaces where
distances remain more informative. This is consistent with Table 9, which
shows that truncated prefixes yield equal or better clustering quality in most
level-dataset combinations. In particular, we obtain the largest gain at the
coarsest level on MuSiQue and nearly identical scores atl= 4. The observed
efficiency gains in response time further reinforce the practical appeal of this
design choice. Operating on lower-dimensional vectors makes the retrieval
step computationally cheaper without sacrificing the quality of the retrieved
documents or the downstream answers.
Table 10: Ablation study on QA performance, as measured by EM and F1, on the three
benchmarks. TheEmbeddingcolumn indicates whether the system uses Matryoshka trun-
cated (Matryoshka) or full-dimensional (Full) embeddings during retrieval. Values repre-
sent the results computed over 5 samples, composed of 1,000 different questions. For each
column, the optimal value is shown in bold.
HotpotQA 2Wiki MuSiQue
Embedding EM↑F1↑EM↑F1↑EM↑F1↑
Full 59.28 72.47 49.60 60.29 26.00 36.23
Matryoshka60.50 73.66 52.70 63.42 27.40 37.89
23

Table 11: Ablation study on retrieval and efficiency performance, as measured by R@2,
R@5, and Res (computed as the sum of retrieval and answer generation time) on the
three benchmarks. TheEmbeddingcolumn indicates whether the system uses Matryoshka
truncated (Matryoshka) or full-dimensional (Full) embeddings during retrieval. Values
represent the results computed over 5 samples composed of 1,000 different questions. For
each column, the optimal value is shown in bold.
HotpotQA 2Wiki MuSiQue
Embedding R@2↑R@5↑Res↓R@2↑R@5↑Res↓R@2↑R@5↑Res↓
Full 70.86 86.72 3.10 62.12 76.88 4.04 41.86 55.80 4.82
Matryoshka72.65 89.30 1.85 63.08 79.27 2.49 42.93 57.08 2.99
5. Discussion
One of the core hypotheses of MatRAG is that the semantic hierarchy of
a clustering structure can be exploited through the nested MRL structure.
This means that coarser levels can be indexed at shorter prefix dimensions
without a significant loss of discriminative power. The retrieval results sup-
port this hypothesis; in fact, MatRAG outperforms the baselines on most
metrics while operating with a lower dimensionality of the embeddings at the
upper levels of the DAG. The largest improvements are seen in MuSiQue,
which may suggest that hierarchical traversal is most beneficial when single-
document similarity is insufficient to identify the relevant evidence chain.
Another core component of MatRAG is the entity-Jaccard re-ranking
signal. Hyperparameter analysis (Table 3) shows that the entity score com-
plements semantic similarity. Removing it entirely (α= 1.00) reduces per-
formance, particularly on 2Wiki, where compositional reasoning chains tend
to introduce numerous named entities across hops. Conversely, relying ex-
clusively on the entity signal (α= 0.00) also results in poor performance,
as entity overlap alone cannot capture semantic relevance. The optimal bal-
ance is achieved atα= 0.50across all datasets, indicating that the two
signals complement each other. Furthermore, the adaptive anchoring sched-
ule shows that neither anchoring the search exclusively to the original query
nor expanding it without control is sufficient. This is because the interpo-
lation guided byβ tis necessary to exploit the retrieved context and, at the
same time, prevent semantic drift. This tension cannot be solved by static
scoring strategies.
The clustering analysis provided further evidence in support of the cen-
tral hypothesis of MatRAG. As shown in Table 9, the clusters obtained from
truncated Matryoshka prefixes are competitive with those obtained from
full-dimensional embeddings and tend to produce slightly better clustering
24

quality. This is refelected in lower DB indices and higher Silhouette scores
in several configurations. This suggests that, at the coarser levels of the hi-
erarchy, the additional dimensions of the full embedding contain redundant
information for forming well-separated clusters, which is removed by trun-
cation. Consequently, the Matryoshka indexing scheme reduces the cost of
similarity computations without compromising the quality of the hierarchy.
After verifying that MRL maintains the quality of the hierarchy, a com-
plementary question is whether this also applies to end-to-end retrieval, in
which truncated representations are used to traverse the DAG and rank doc-
uments. Tables10and11addressthisissuebycomparingMatryoshka’strun-
cated and full-dimensional embeddings during retrieval. Across all bench-
marks, the Matryoshka configuration achieves better performance consis-
tently in terms of both QA metrics and retrieval recall while reducing re-
sponse time. These results suggest that truncating embeddings to a lower-
dimensional subspace, as enabled by MRL, does not degrade retrieval qual-
ity. This behavior can be attributed to the way Matryoshka embeddings are
trained, as the most semantically relevant information is encoded in the lead-
ing dimensions. This effectively concentrates the signal and reduces noise in
the higher-dimensional components. Consequently, using only the truncated
embedding prefixes acts as a form of implicit regularization, as it discards di-
mensions that provide little discriminative information and could introduce
spurious similarity signals. The observed efficiency gains in response time
further reinforce the practical appeal of this design choice, as operating on
lower-dimensional vectors makes the retrieval step computationally cheaper
without sacrificing the quality of the retrieved documents or the downstream
answers.
The results obtained in our tests have several important implications
for the design of retrieval systems. First, coupling the embedding dimen-
sion with the index depth is a productive inductive bias because it exploits
the multi-scale structure of existing MRL models without requiring addi-
tional trainable parameters or training. Second, lightweight, entity-based
signals can effectively proxy the reasoning and planning typically performed
by LLMs, substantially reducing query time without a corresponding loss
in answer quality. Third, a performance comparable to that of graph-based
pipelines can be achieved with structured retrieval, without the need for
explicit KGs, entity linking and LLM-based summarization. This makes
MatRAG’s approach applicable to a broad range of corpora without requir-
ing explicit KG construction, eliminating the need for the preprocessing that
theseoperationsrequire. Together, thesefindingssuggestthatthedichotomy
between cheap, flat retrieval and expensive, structured retrieval can be par-
25

tiallybridgedbycarefullyco-designingtheembeddingstrategyandtheindex
structure.
As for the MatRAG’s limitations, we observe that the entity extraction
step of MatRAG relies on GLiNER [41], which can fail to identify domain-
specific entities or produce inaccurate extractions on corpora with unusual
terminology. Errors in this step propagate into the budget control mecha-
nism and the re-ranking signal. Additionally, the quality of the hierarchical
index depends on the density of the embedding space. For instance, clusters
may be poorly defined on corpora with very uniform semantic distributions,
reducing the discriminative value of the DAG traversal. Finally, the greedy,
top-down traversal of the DAG selects the most similar cluster centroid at
each level, committing irrevocably to that branch without a backtracking
mechanism. An error at a coarse level, where shorter Matryoshka prefixes
provide less discriminative representations, propagates downward and may
exclude relevant documents from the candidate set, regardless of their simi-
larity at finer levels. The nearest overlapping assignment mitigates this risk
by ensuring that each node is reachable frompparent clusters, thus provid-
ing redundant paths through the hierarchy. However, if the correct branch
is not among the selected parents of a relevant node, the traversal offers no
recovery mechanism.
6. Conclusion
Inthispaper,wehavepresentedMatRAG,ahierarchicalRAGframework
for multi-hop Question Answering that leverages the structural alignment
between density-based clustering and MRL. MatRAG organizes a document
corpus into a DAG of clusters, associating each level with a correspond-
ingly shorter MRL prefix dimension. This enables top-down retrieval that is
dimensionality-aware and semantically structured. Unlike the multi-hop ap-
proach used by many KG-based RAGs, MatRAG employs an entity-driven
iterative mechanism that controls the retrieval budget through entity counts
and re-ranks candidates based on entity overlap. Furthermore, an adap-
tive anchoring schedule mitigates query drift by progressively increasing the
weight of the original query as the budget is consumed.
Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue, designed
to compare MatRAG with seven representative baselines, demonstrate that
MatRAG consistently delivers superior retrieval and answer quality across
a wide range of settings. Additionally, it reduces indexing time by orders
of magnitude compared to graph-based methods and reduces query-time
latency by over35.14%compared to NaiveRAG.
26

Future work will explore three main directions. The first is predicting
a query-dependent retrieval budget, which adjusts the number of retrieved
documents based on the query complexity. The second direction involves
learning adaptive cluster connectivity, which replaces a fixed number of par-
entassignmentswithahierarchythatbetterreflectsthesemanticstructureof
the corpus. The third direction involves a query-dependent anchoring sched-
ule that balances the original and expanded queries dynamically, enabling
more effective exploration while preserving alignment with the question.
Declaration of competing interest
The authors declare that they have no known competing financial inter-
ests or personal relationships that could have appeared to influence the work
reported in this paper.
Declaration on Generative AI
The authors declare that they used generative AI tools solely to polish
the language of the manuscript, such as improving phrasing and correcting
grammar.
Data availability
The code used for our study are publicly available at the linkhttps:
//anonymous.4open.science/r/MatRAG.
References
[1] R.J.G.B. Campello, D. Moulavi, A. Zimek, and J. Sander. Hierar-
chical Density Estimates for Data Clustering, Visualization, and Out-
lier Detection.ACM Transactions on Knowledge Discovery from Data,
10(1):5:1–5:51, 2015.
[2] C. Chu, Y. Jeong, H. Cho, J. Kim, J. Lee, B. Bang, J. Lee, U. Song,
and S.B. Kim. Cats-RAG: Contextual Augmented Triplet Synthesis for
RAG in Technical QA.Expert Systems with Applications, page 131491,
2026. Elsevier.
[3] D.L. Davies and D.W. Bouldin. A cluster separation measure.IEEE
Transactions on Pattern Analysis and Machine Intelligence, 1(2):224–
227, 1979.
27

[4] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Truitt,
D. Metropolitansky, R.O. Ness, and J. Larson. From local to global: A
graph RAG approach to query-focused summarization.arXiv preprint
arXiv:2404.16130, 2024.
[5] W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T. Chua, and
Q. Li. A survey on RAG meeting LLMS: Towards retrieval-augmented
large language models. InProc. of the ACM SIGKDD Conference on
Knowledge Discovery and Data Mining (KDD’24), pages 6491–6501,
Barcelona, Catalunya, Spain, 2024. ACM.
[6] Z.Guo,L.Xia,Y.Yu,T.Ao,andC.Huang. LightRAG:SimpleandFast
Retrieval-Augmented Generation. InFindings of the Association for
Computational Linguistics: EMNLP 2025, pages 10746–10761, Suzhou,
China, 2025. ACL.
[7] B.J.Gutiérrez, Y.Shu, Y.Gu, M.Yasunaga, andY.Su. Hipporag: Neu-
robiologically inspired long-term memory for large language models. In
Proc. of the Annual Conference on Neural Information Processing Sys-
tems (NeurIPS’24), volume 37, pages 59532–59569, Vancouver, British
Columbia, Canada, 2024.
[8] B.J. Gutiérrez, Y. Shu, W. Qi, S. Zhou, and Y. Su. From RAG to Mem-
ory: Non-ParametricContinualLearningforLargeLanguageModels. In
Proc. of the International Conference on Machine Learning (ICML’25),
pages 21497–21515, Vancouver, British Columbia, Canada, 2025. Open-
Review.net.
[9] H.W.A. Hanley and Z. Durumeric. Hierarchical level-wise news article
clustering via multilingual Matryoshka embeddings. InProc. of the An-
nual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers), pages 2476–2492, Vienna, Austria, 2025. ACL.
[10] X. Ho, A.D. Nguyen, S. Sugawara, and A. Aizawa. Constructing a
multi-hop QA dataset for comprehensive evaluation of reasoning steps.
InProc. of the International Conference on Computational Linguistics
(COLING’20), pages 6609–6625, Barcelona, Catalunya, Spain, 2020.
International Committee on Computational Linguistics.
[11] H. Huang, Y. Huang, J. Yang, Z. Pan, Y. Chen, K. Ma, H. Chen, and
J. Cheng. Retrieval-Augmented Generation with Hierarchical Knowl-
edge. InFindings of the Association for Computational Linguistics
(EMNLP’25), pages 6044–6060, Suzhou, China, 2025. ACL.
28

[12] Y. Huang, L. Yang, X.H. Yang, and X. Xu. Retrieval-Augmented Gen-
eration for Multi-Hop Question Answering Based on Structured Plan-
ning.ACM Transactions on Knowledge Discovery from Data, 20(3):1–
20, 2026.
[13] Z. Jiang, M. Sun, L. Liang, and Z. Zhang. Retrieve, summarize, plan:
Advancing multi-hop question answering with an iterative approach. In
Proc. of the ACM on Web Conference (WWW’25), pages 1677–1686,
Sydney, Australia, 2025. ACM.
[14] B. Jin, C. Xie, J. Zhang, K.K. Roy, Y. Zhang, Z. Li, R. Li, X. Tang,
S. Wang, Y. Meng, and J. Han. Graph Chain-of-thought: Augmenting
Large Language Models by Reasoning on Graphs. InFindings of the
Association for Computational Linguistics (ACL’24), pages 163–184,
Bangkok, Thailand, 2024. ACL.
[15] A. Kusupati, G. Bhatt, A. Rege, M. Wallingford, A. Sinha, V. Ramanu-
jan, W.Howard-Snyder, K.Chen, S.M.Kakade, P.Jain, andA.Farhadi.
Matryoshka Representation Learning. InAdvances in Neural Infor-
mation Processing Systems: Annual Conference on Neural Information
Processing Systems (NeurIPS’22), New Orleans, LA, USA, 2022.
[16] K.H.Lau, F.Zhang, B.Ruan, Y.Zhou, Q.Guo, R.Zhang, andX.Zhou.
Breaking the Static Graph: Context-Aware Traversal for Robust
Retrieval-Augmented Generation.arXiv preprint arXiv:2602.01965,
2026.
[17] H. Li, A. Mourad, S. Zhuang, B. Koopman, and G. Zuccon. Pseudo rel-
evance feedback with deep language models and dense retrievers: Suc-
cessesandpitfalls.ACM Transactions on Information Systems, 41(3):1–
40, 2023.
[18] X. Li and S. Wang. MRL-RAG: Enhancing the Accuracy of Retrieval-
Augmented Generation via an Optimized Hybrid Query Retriever. In
Proc. of the International Conference on Frontier Technologies of In-
formation and Computer (ICFTIC’25), pages 93–98, Qingdao, China,
2025. IEEE.
[19] O. Nacar, S. Sibaee, and A. Koubaa. Enhanced Arabic Retrieval Aug-
mented Generation Using Nested Embedding Models. InProc. of the
International Conference on Smart Systems and Emerging Technolo-
gies (SMARTTECH’24), pages 120–131, Marrakesh, Morocco, 2024.
Springer.
29

[20] Z. Nussbaum, J. Xavier Morris, A. Mulyar, and B. Duderstadt. Nomic
Embed: Training a Reproducible Long Context Text Embedder.Trans-
actions on Machine Learning Research, 1, 2025.
[21] B.Peng, Y.Zhu, Y.Liu, X.Bo, H.Shi, C.Hong, Y.Zhang, andS.Tang.
Graph retrieval-augmented generation: A survey.ACM Transactions
on Information Systems, 44(2):1–52, 2025. ACM.
[22] Q. Peng and X. Luo. Dual-granularity chunking and dynamic context
augmentation: An optimization method for retrieval-augmented gener-
ation.Expert Systems with Applications, 332:133546, 2027. Elsevier.
[23] F. Pennino, A. Gurioli, and M. Gabbrielli. Trajectory-Embedded Ma-
tryoshka Representation Learning for Enhanced Similarity Analysis.
InProc. of the European Symposium on Artificial Neural Networks
(ESANN’25), pages 1–6, Bruges, Belgium, 2025.
[24] G. Perkovi’c, A. Drobnjak, and I. Botički. Hallucinations in LLMS:
Understanding and addressing challenges. InProc. of the MIPRO ICT
and Electronics Convention (MIPRO’24), pages 2084–2088, Opatija,
Croatia, 2024. IEEE.
[25] O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, and M. Lewis.
Measuring and narrowing the compositionality gap in language models.
InFindings of the Association for Computational Linguistics: EMNLP
2023, pages 5687–5711, Singapore, 2023.
[26] M. Rathee, V. Venktesh, S. MacAvaney, and A. Anand. Test-time Cor-
pus Feedback: From Retrieval to RAG. InFindings of the Association
for Computational Linguistics: European Chapter of the Association for
Computational Linguistics (EACL’26), pages 5637–5656, Rabat, Mo-
rocco, 2026. ACL.
[27] J.J. Rocchio Jr. Relevance feedback in information retrieval.The
SMART retrieval system: experiments in automatic document process-
ing, 1971. Englewood Cliffs.
[28] P.J. Rousseeuw. Silhouettes: a graphical aid to the interpretation and
validation of cluster analysis.Journal of Computational and Applied
Mathematics, 20:53–65, 1987.
[29] A.O. Saleh, G. Tur, and Y. Saygin. SG-RAG: Multi-hop question an-
swering with large language models through knowledge graphs. InProc.
30

of the International Conference on Natural Language and Speech Pro-
cessing (ICNLSP’24), pages 439–448, Trento, Italy, 2024. ACL.
[30] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C.D. Man-
ning. Raptor: Recursive abstractive processing for tree-organized re-
trieval. InProc. of the International Conference on Learning Represen-
tations (ICLR’24), Vienna, Austria, 2024. OpenReview.net.
[31] Z. Song, X. Kong, X. Bao, Y. Zhou, J. Jiao, S. Liu, Y. Zhou, and
H. Qi. LLM-confidence reranker: a training-free approach for enhancing
retrieval-augmented generation systems.Expert Systems with Applica-
tions, page 131627, 2026. Elsevier.
[32] J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y. Gong, L. Ni, H. Shum, and
J. Guo. Think-on-Graph: Deep and Responsible Reasoning of Large
Language Model on Knowledge Graph. InProc. of the International
Conference on Learning Representations (ICLR’24), Vienna, Austria,
2024. OpenReview.net.
[33] Y. Tang and Y. Yang. MultiHop-RAG: Benchmarking Retrieval-
Augmented Generation for Multi-Hop Queries. InProc. of the Inter-
national Conference on Language Modeling (COLM’24), Philadelphia,
PA, USA, 2024.
[34] W. Tao, X. Xing, Y. Chen, L. Huang, and X. Xu. Treerag: Unleash-
ing the power of hierarchical storage for enhanced knowledge retrieval
in long documents. InFindings of the Association for Computational
Linguistics: ACL 2025, pages 356–371, Vienna, Austria, 2025.
[35] V.A. Traag, L. Waltman, and N.J. Van Eck. From Louvain to Leiden:
guaranteeing well-connected communities.Scientific reports, 9(1):5233,
2019. Nature Publishing Group UK London.
[36] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal. MuSiQue:
MultihopQuestionsviaSingle-hopQuestionComposition.Transactions
of the Association for Computational Linguistics, 10:539–554, 2022.
MIT Press.
[37] A. Vidyarthi, M.K. Singh, and D.S. Moirangthem. SageRAG: Query
Rewriting for Retrieval Enhancement and Retrieval-Augmented Gener-
ation for Grounded Responses in AI Research Assistance.Expert Sys-
tems with Applications, page 131160, 2026. Elsevier.
31

[38] S. Wang, Y. Fang, Y. Zhou, X. Liu, and Y. Ma. Archrag: At-
tributed community-based hierarchical retrieval-augmented generation.
InProc. of the AAAI Conference on Artificial Intelligence (AAAI’26),
volume 40, pages 15868–15876, Singapore, 2026. AAAI Press.
[39] Y. Wang, N. Lipka, R. A. Rossi, A. Siu, R. Zhang, and T. Derr.
Knowledge graph prompting for multi-document question answering.
InProc. of the AAAI Conference on Artificial Intelligence (AAAI’24),
volume 38, pages 19206–19214, Vancouver, British Columbia, Canada,
2024.
[40] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and
C.D. Manning. HotpotQA: A dataset for diverse, explainable multi-
hop question answering. InProc. of the International Conference on
Empirical Methods in Natural Language Processing (EMNLP’18), pages
2369–2380, Brussels, Belgium, 2018. ACL.
[41] U. Zaratiana, N. Tomeh, P. Holat, and T. Charnois. GLiNER: Gen-
eralist Model for Named Entity Recognition using Bidirectional Trans-
former. InProc. of the Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technolo-
gies (NAACL’24), pages 5364–5376, Mexico City, Mexico, 2024. ACL.
[42] N. Zhang, P. K. Choubey, A. Fabbri, G. Bernadett-Shapiro, R. Zhang,
P. Mitra, C. Xiong, and C.S. Wu. SiReRAG: Indexing Similar and Re-
lated Information for Multihop Reasoning. InProc. of the International
Conference on Learning Representations (ICLR’25), Singapore, 2025.
OpenReview.net.
[43] X. Zhang, R. Zhang, X. Xing, S. Zhou, and J. Chen. A Dense Re-
trieval Model Training Method Combining Matryoshka Representation
Learning and Knowledge Distillation. InProc. of the Asian Conference
on Artificial Intelligence Technology (ACAIT’24), pages 46–53, Fuzhou,
China, 2024. IEEE.
[44] X. Zhang, F. Zhao, Y. Liu, P. Chen, Y. Wang, X. Wang, D. Ma, H. Xu,
M. Chen, and H. Li. TreeQA: Enhanced LLM-RAG with logic tree
reasoning for reliable and interpretable multi-hop question answering.
Knowledge-Based Systems, 330:114526, 2025.
[45] X. Zhu, Y. Xie, Y. Liu, Y. Li, and W. Hu. Knowledge Graph-Guided
Retrieval Augmented Generation. InProc. of the Conference of the
32

Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (NAACL’25) - Volume 1:
Long Papers, pages 8912–8924, Albuquerque, NM, USA, 2025. ACL.
[46] L. Zighelnic and O. Kurland. Query-drift prevention for robust query
expansion. InProc. of the International ACM SIGIR Conference on
Research and Development in Information Retrieval (SIGIR’08), pages
825–826, Singapore, 2008.
33