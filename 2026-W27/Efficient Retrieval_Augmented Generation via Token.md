# Efficient Retrieval-Augmented Generation via Token Co-occurrence Graphs

**Authors**: Gianluca Bonifazi, Christopher Buratti, Michele Marchetti, Federica Parlapiano, Giulia Quaglieri, Davide Traini, Domenico Ursino, Luca Virgili

**Published**: 2026-06-29 10:29:51

**PDF URL**: [https://arxiv.org/pdf/2606.30093v1](https://arxiv.org/pdf/2606.30093v1)

## Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Language Models (LLMs) by grounding the generation process on external knowledge. However, standard RAG approaches struggle with multi-hop reasoning. While recent graph-based RAG methods improve the retrieval of interconnected chunks, they often rely on computationally expensive and error-prone LLM-based extraction pipelines. To address these issues, we propose TIGRAG (Token-Induced GraphRAG), an efficient graph-augmented RAG framework based on a token co-occurrence Knowledge Graph. TIGRAG directly models topological relationships between tokens using sliding-window co-occurrence statistics, thus enabling scalable graph construction. During inference, it combines graph-based semantic expansion and neural reranking to retrieve interconnected evidence for multi-hop reasoning. Specifically, it introduces an iterative entity-driven retrieval strategy that progressively expands the query using bridging entities extracted from previously retrieved contexts. We evaluated TIGRAG on three widely adopted multi-hop Question Answering (QA) benchmarks. Experimental results demonstrated that our framework consistently outperforms dense retrieval and graph-based RAG methods in both retrieval and downstream QA tasks, while substantially reducing indexing time, inference latency, and prompt footprint.

## Full Text


<!-- PDF content starts -->

Efficient Retrieval-Augmented Generation via Token
Co-occurrence Graphs
Gianluca Bonifazia, Christopher Burattia, Michele Marchettia, Federica
Parlapianoa, Giulia Quaglieria, Davide Trainib, Domenico Ursinoa, Luca
Virgilia,∗
aUniversità Politecnica delle Marche, Ancona, Italy
bUniversità di Modena e Reggio Emilia, Modena, Italy
Abstract
Retrieval-Augmented Generation (RAG) mitigates hallucinations in Large Lan-
guage Models (LLMs) by grounding the generation process on external knowl-
edge. However, standard RAG approaches struggle with multi-hop reason-
ing. While recent graph-based RAG methods improve the retrieval of intercon-
nected chunks, they often rely on computationally expensive and error-prone
LLM-based extraction pipelines. To address these issues, we propose TIGRAG
(Token-Induced GraphRAG), an efficient graph-augmented RAG framework
based on a token co-occurrence Knowledge Graph. TIGRAG directly mod-
els topological relationships between tokens using sliding-window co-occurrence
statistics, thus enabling scalable graph construction. During inference, it com-
bines graph-based semantic expansion and neural reranking to retrieve inter-
connected evidence for multi-hop reasoning. Specifically, it introduces an itera-
tive entity-driven retrieval strategy that progressively expands the query using
bridging entities extracted from previously retrieved contexts. We evaluated
TIGRAG on three widely adopted multi-hop Question Answering (QA) bench-
marks. Experimental results demonstrated that our framework consistently
outperforms dense retrieval and graph-based RAG methods in both retrieval
and downstream QA tasks, while substantially reducing indexing time, infer-
ence latency, and prompt footprint.
Keywords:Retrieval-Augmented Generation, Knowledge Graphs, Multi-hop
∗Corresponding author
Email addresses:g.bonifazi@univpm.it(Gianluca Bonifazi),c.buratti@pm.univpm.it
(Christopher Buratti),michele.marchetti@univpm.it(Michele Marchetti),
f.parlapiano@pm.univpm.it(Federica Parlapiano),g.quaglieri@pm.univpm.it(Giulia
Quaglieri),davide.traini@unimore.it(Davide Traini),d.ursino@univpm.it(Domenico
Ursino),luca.virgili@univpm.it(Luca Virgili)
1This is the first author footnote.
2Another author footnote, this is a very long footnote and it should be a really long
footnote. But this footnote is not yet sufficiently long enough to make two lines of footnote
text.
3Yet another author footnote.arXiv:2606.30093v1  [cs.CL]  29 Jun 2026

Question Answering, Large Language Models, Information Retrieval
1. Introduction
As Large Language Models (LLMs) become more widespread, the research
community is paying increasing attention to their inherent vulnerabilities, par-
ticularly the hallucination phenomenon [30, 47]. Retrieval-Augmented Genera-
tion (RAG) has emerged as a well-established approach to mitigate hallucina-
tions by anchoring LLM outputs in external, factual knowledge [8, 29, 3]. In
a standard RAG pipeline, a user query is encoded as a vector, which is then
employed to retrieve relevant text chunks from a large document corpus through
a similarity search [27, 17]. This retrieved information is then used to provide
the LLM with contextual evidence during generation [21, 23, 48]. However,
traditional vector-based RAG systems are ineffective because they treat docu-
ments as largely disconnected units, which limits their ability to reason across
multiple pieces of information. Consequently, they often struggle with multi-
hop queries, which require synthesizing dispersed evidence across independent
chunks [2, 38, 26, 49].
To address this issue, Graph-augmented RAG (GraphRAG) approaches rely
on structured representations, often formalized as Knowledge Graphs (KGs).
In KGs, nodes and edges capture textual elements and their complex relation-
ships, which provides a topological space to map and navigate knowledge [5, 11].
Theseapproachesrangefrominjectinggraph-derivedknowledge(suchastriplets
or community summaries) directly into the context [5], to using the topologi-
cal structure as a routing mechanism to retrieve the most relevant raw text
chunks [13]. Due to their interconnected structure, GraphRAG approaches can
inherently address the complex reasoning paths required by multi-hop Ques-
tion Answering (QA) [33, 19]. Nevertheless, two critical challenges hinder their
broader adoption. First, offline KG construction introduces significant compu-
tational overhead and latency [28, 4]. Second, traditional GraphRAG pipelines
[5, 11] rely heavily on LLM-based entity and relationship extraction, making
them susceptible to hallucinated, noisy, or missing connections. These graph
construction errors can propagate through the retrieval pipeline and ultimately
cause cascading failures in downstream reasoning and generation [9].
This paper aims to address these computational overheads and extraction-
based vulnerabilities by proposing TIGRAG (Token-Induced GraphRAG), a
highly efficient graph-augmented RAG. Unlike traditional GraphRAG systems,
TIGRAG statistically constructs its topological foundation by extracting tokens
from the corpus and linking them based on co-occurrences within a sliding
window. TIGRAG’s architecture consists of four core components:(i)an offline
graph extraction stage that efficiently builds the topological KG,(ii)a semantic
expansion and candidate retrieval module powered by Personalized PageRank
(PPR) [14],(iii)a dynamic graph-weighted scoring mechanism for contextual
chunk evaluation, and(iv)a final neural reranking module that distills the top
chunks based on semantic alignment with the query.
2

We evaluated TIGRAG using the standard protocols outlined in the lit-
erature [13, 12] on three multi-hop QA benchmarks: HotpotQA [46], 2Wiki-
MultiHopQA [15], and MuSiQue [40]. Our empirical results demonstrate that
TIGRAG not only outperforms existing RAG baselines across all datasets but
also yields substantial efficiency improvements. Indeed, it significantly reduces
indexing and retrieval latency, as well as overall token consumption.
Thispaperisorganizedasfollows: Section2discussesrelatedwork. Section3
details the TIGRAG’s architecture and its core algorithms. Section 4 describes
the experimental campaign performed to evaluate TIGRAG. Section 5 presents
a discussion on our framework. Finally, Section 6 draws our conclusions and
outlines some promising directions for future research.
2. Related Work
This section illustrates approaches related to TIGRAG. Specifically, Subsec-
tion 2.1 presents standard RAG approaches. Subsection 2.2 describes Structure-
AugmentedRetrievalApproaches. Finally,Subsection2.3discussestheNetwork-
Based Retrieval and Multi-Hop Reasoning paradigms.
2.1. Standard Retrieval-Augmented Generation
RAG has emerged as a key paradigm for limiting LLM hallucinations. It
anchors LLM outputs in reliable and up-to-date external knowledge sources,
thereby improving factual accuracy and response quality [6].
Within this established paradigm, traditional RAG frameworks typically use
lexicalsearches[32]orsemanticsimilarities[20]toretrievethenecessarycontext.
These approaches encode text chunks as continuous vectors and apply similarity
metrics, such as cosine similarity, to effectively retrieve documents that share
semantic overlap with the input query. However, standard RAG approaches
treat documents as isolated units. Consequently, they have significant difficulty
with complex, multi-hop queries that require synthesizing information across
disparate texts. In fact, in this case they are limited by the lexical and semantic
gap between the initial question and the distant supporting facts [2, 38].
Furthermore, standard dense retrieval methods often retrieve overly long
contexts, resulting in severe information redundancy and substantial compu-
tational overhead. To address this issue, recent studies have explored context
pruning techniques. For instance, AttentionRAG [7] introduces an attention-
guided compression mechanism to reduce token consumption and filter out irrel-
evant information from the retrieved passages. While context compression offers
a partial remedy, it often struggles to balance compression rates with the risk of
information loss. Therefore, a more structural solution is needed to accurately
identify interconnected evidence instead of just compressing flat text.
2.2. Structure-Augmented Retrieval Approaches
To overcome the limitations of flat vector spaces, recent approaches have
integrated structural dependencies into the RAG pipeline. In this context,
3

GraphRAG has emerged as a promising paradigm that leverages graph-based
indexing and retrieval to exploit relationships among entities. This allows for
more precise and context-aware reasoning over complex knowledge bases [29].
Several specific architectures have been proposed based on this paradigm.
For instance, RAPTOR [35] organizes textual knowledge into recursive tree
structures through iterative abstractive processing, which facilitates retrieval
at various levels of granularity. Similarly, TreeRAG [39] addresses the frag-
mentation typical of long documents by employing tree-based chunking and
bidirectional traversal strategies. This approach strictly preserves hierarchical
dependencies during retrieval.
Concurrently, graph-basedsystemshavegainedsignificantattentionfortheir
ability to improve information aggregation through hierarchical indexing and
summarization [34]. GraphRAG [5] leverages KGs to generate hierarchical
community-level summaries, while LightRAG [11] integrates graph topologies
with dual-level vector representations to efficiently retrieve entities and relation-
ships. Approaches likeKG2RAG[50] and HyperGraphRAG [24] further exploit
structured inter-chunk relationships by using KGs and hypergraph representa-
tions, respectively, to enrich the retrieved information. Furthermore, GFM-
RAG [25] introduces Graph Foundation Models to directly augment the gen-
erative process with deep structural awareness. ToG [36] seamlessly integrates
information retrieval from both structured KGs and unstructured text, employ-
ing a tightly coupled reasoning mechanism to ground the generative process.
KGP [42] explicitly targets Multi-Document QA (MD-QA) by constructing a
passage-level KG and deploying an LLM-based traversal agent. This dual mech-
anism regulates the transition space among interconnected documents while
dynamically gathering relevant context, thereby balancing retrieval latency and
generation quality.
Many of these approaches are effective, but they rely on computationally ex-
pensive LLM-driven extraction pipelines or extensive structural pre-training to
construct their underlying graphs [45]. Furthermore, recent studies have shown
that implementations like GraphRAG encounter significant scalability issues in
real-world, large-scale deployments. In these scenarios, the computational com-
plexityofgraphtraversalalgorithmsandretrievallatencyincreaseexponentially
with the size of the KG [29].
TIGRAG builds a highly traversable KG directly from normalized text to-
kens and their local co-occurrences. This streamlined approach bypasses the
heavy computational cost of generative graph construction, ensuring highly ef-
ficient, low-latency retrieval operations while preserving the intrinsic structural
integrity of the corpus.
2.3. Network-Based Retrieval and Multi-Hop Reasoning
While many approaches focus on structuring the knowledge index for bet-
ter summarization, another line of research emphasizes the direct use of graph
topologies to retrieve interconnected contexts [18, 43]. For example, GRAG [16]
combines textual and topological information to pinpoint relevant subgraphs,
whereas KG-RAG [22] employs explicit question decomposition and multi-hop
4

KG retrieval to enable deep reasoning. However, graph-based retrieval ap-
proaches often face a granularity tradeoff [44]. In fact, fine-grained, entity-level
graphscancapturedetailedrelationshipsbutarecomputationallyexpensiveand
may lose contextual information. Conversely, coarse document-level graphs pre-
serve context but struggle to model complex, multi-hop relationships. Recent
methods, such as QCG-RAG [44], address this issue by building query-centric
graphs to retrieve relevant chunks at different levels of granularity. Other ap-
proaches rely on Graph Neural Networks (GNNs) with query-aware attention
mechanisms to dynamically identify and aggregate relevant information from
KGs [1].
Rather than relying on explicit query decomposition or training computa-
tionally intensive GNN architectures, a parallel and highly effective strategy
leverages network theory algorithms to organically navigate these knowledge
structures. Approaches like HippoRAG [13] and its successor HippoRAG2 [12]
have demonstrated the effectiveness of using PPR to emulate human cognitive
indexing. This allows the retrieval signal to naturally spread across semantic
nodes.
While TIGRAG is inspired by these network-based principles, it introduces
a token-level extraction phase, a dynamic candidate pooling strategy, and an
iterative, entity driven multi-hop retrieval process. In particular, it extracts
Named Entities from the first retrieval hop to expand the query and applies a
neural reranking step. This allows it to retrieve interconnected evidence that
traditional dense or single-hop graph-based methods often overlook.
3. Proposed Approach
This section outlines the technical features of TIGRAG. Specifically, Sub-
section 3.1 explains the rationale behind our framework, whereas the following
subsections describe the various steps that comprise its underlying approach.
3.1. Motivations
Standard RAG systems often struggle with multi-hop reasoning, whereas
graph-based retrieval approaches frequently incur the cost of complex LLM-
driven graph construction. To address these issues, we propose TIGRAG, a
lightweight graph-augmented retrieval framework based on token co-occurrence
graphs. Instead of relying on LLM-driven graph construction, TIGRAG builds
a token-level topological representation directly from textual co-occurrences, en-
abling efficient and scalable graph creation. During retrieval, it combines graph-
based semantic expansion with neural reranking to identify interconnected evi-
dence for multi-hop reasoning. Then, an iterative, entity-driven retrieval strat-
egy progressively augments the query using bridging entities extracted from
previously retrieved contexts. TIGRAG substantially reduces prompt size, in-
ference latency, and overall computational overhead by dynamically selecting
compact and highly relevant contexts, outperforming existing GraphRAG ap-
proaches.
5

Figure 1 illustrates the TIGRAG’s workflow. As shown in this figure, the
TIGRAGarchitectureoperatesacrosstwofundamentalstages:(i)OfflineGraph
Construction, and(ii)Online Inference. During the offline phase, TIGRAG seg-
ments the raw corpus into overlapping chunks, from which token co-occurrence
frequenciesareusedtoconstructthetopologicalKG.Atthesametime,TIGRAG
precomputes and indexes chunk-level dense embeddings and token-to-chunk
provenance maps. At inference time, it tokenizes the incoming query to iden-
tify topological seed nodes. These seeds then drive a semantic expansion process
thatfetchesaninitialpoolofcandidatechunks. Next, TIGRAGappliesagraph-
weighted scoring mechanism to dynamically filter the candidates before passing
them to a neural reranker for precise semantic alignment. Ultimately, it injects
the top distilled chunks into the LLM’s prompt to ground the final generation.
This single-hop retrieval process is then repeated to enable multi-hop reasoning.
Offline Graph Construction
...
Co-occurrence Frequency Analysis Text Chunks DocumentsMulti-document
corpus
...Topological Knowledge Graph
Online Inference
User
question Seed Nodes
Semantic Expansion
and Candidate
Chunks RetrievalToken-Chunks
Association
Graph-Weighted
Scoring and Rankingc1:0.35
c2:0.23
c3:0.12c5:0.03
c6:0.02
c7:0.01c1
c3
c4
c2
Dynamic
cutoff
Neural 
Reranking Top chunksc1
c3
c4
c2
c5
c4:0.03 c8:0.01c5
c6User
question+
LLMFinal
AnswerToken-Chunks
Association
...
Figure 1: Workflow of TIGRAG, showing graph construction, semantic expansion, and itera-
tive multi-hop retrieval
3.2. Pre-processing and Text Chunking
LetD={d 1, . . . , d f}be the initial corpus offdocuments. Each docu-
mentd i∈Dis defined as a tuple⟨t i, bi⟩, representing its title and its body,
respectively. First, we apply a sentence tokenizer to segment the bodyb iinto
an ordered sequence of sentencesS i= (ϱ i1, . . . , ϱ im). Afterwards, we perform a
preprocessing pipeline to prepare the raw corpus for subsequent graph construc-
tion. Since TIGRAG relies on token co-occurrences, preserving local semantic
coherence during chunking is essential to avoid noisy graph connections.
Although recent attention-based segmentation strategies [10] improve se-
mantic preservation, they introduce additional computational overhead. There-
6

fore, TIGRAG does not adopt these strategies but uses a lightweight, sentence-
level, sliding-window strategy.
Letsbe the window size (i.e., number of sentences per chunk) and letobe
the overlap. We define the stride length asl=s−o. Finally, letλ ibe the
number of chunks ofd i. Thek-th chunk ofd i,1≤k≤λ i, denoted asc ik, is
formally defined as the subsequence:
cik= (ϱ i(k−1)l+1 , . . . , ϱ i(k−1)l+s )(3.1)
To facilitate the computation of semantic similarity during the retrieval
phase, we map each chunkc ikinto a dense vector representation. Specifically,
we use an embedding modelMto compute the chunk-level embeddingϵ ik∈Rδ:
ϵik=M(c ik)(3.2)
whereδis the embedding dimensionality. We store these dense representa-
tionsandusethemforcosinesimilarity-basedretrieval(seebelow). Thisenables
us to efficiently select the most relevant chunks for generation.
After chunking, we tokenize each chunkc ikthrough a functionτ(·)that re-
ceives a chunk and returns the set of its tokens. To reduce noise, we remove
stopwords that could negatively affect the graph structure. LetSbe the set
of stopwords and letϕ(·)be a boolean validation function that takes a token
as input and enforces minimum length and numerical constraints (i.e., it pre-
serves years and large numerical values while discarding punctuation and tokens
shorter than three characters). The final setV ikof valid tokens extracted from
cikis defined as:
Vik={v∈τ(c ik)|v /∈ S ∧ϕ(v)}(3.3)
ThetokensofV ikformthebasisforconstructingtheKGsupportingTIGRAG,
as explained in the next subsections.
3.3. Knowledge Graph Construction
After extracting the valid token setsV ikfor each chunkc ik, we construct a
weighted, undirected KG:
G=⟨N, E⟩(3.4)
Nis the set of nodes inG. There is a noden p∈Nfor each tokenv p∈V.
Vis defined as the union of all valid tokens extracted across the entire corpus.
Formally speaking:
V=[
i=1..f,k=1..λ iVik (3.5)
Since each noden p∈Ncorresponds to exactly one tokenv p∈V, we use the
terms “node” and “token”, as well as the symbolsn pandv p, interchangeably.
Eis the set of edges inG. It is based on token co-occurrences in such a way
as to establish semantic relationships between extracted concepts. Specifically,
7

for each processed chunkc ik, letV ikrepresent the ordered sequence of its valid
tokens. We traverse this sequence using a sliding window of sizewand add
an edge(n p, nq, ωpq)toEif the tokensv pandv qco-occur at least once within
this window. The weightω pqis equal to the cumulative frequency of the co-
occurrences ofv pandv qacross all sliding windows in the corpus. It reflects the
strengths of the semantic association betweenv pandv q.
3.4. Query Pre-processing and Semantic Expansion
During inference, an incoming natural language queryµundergoes the same
preprocessing and tokenization pipeline used for graph construction. LetV µ
denote the set of valid tokens extracted fromµ, and letV seed=Vµ∩Vdenote
the subset of the tokens ofµappearing inG. We use the nodes corresponding to
Vseedto perform a PPR process overG. This allows us to explore the topological
neighborhood ofµand identify semantically related concepts.
Specifically, we define a personalization vectorv∈R|V|, where the restart
probability is uniformly distributed across the nodes ofV seedand set to zero
otherwise. In particular, the componentv pofv, corresponding to the node
vp∈V, is defined as follows:
vp=(
1
|Vseed|ifvp∈Vseed
0otherwise(3.6)
We then execute PPR onGusingvas the personalization vector. Upon
convergence, we sort the nodes ofGaccording to their PageRank scores, and
select the top-γnodes to form the expanded token setΦ µcorresponding to
µ. Finally, weL 1-normalize the corresponding PPR scores to obtain a discrete
probability scoreˆp(v)for each tokenv∈Φ µ.
3.5. Candidate Retrieval and Graph-Weighted Scoring
After the expansion step, we retrieve a set of candidate text chunks. A chunk
cis considered a candidate if it contains at least one token from the union of
the original query tokens and the expanded set of tokens, i.e., fromV µ∪Φ µ.
We introduce a graph-weighted scoring to rank these candidate chunks. In
particular, we combine the probabilistic retrieval capabilities of BM25 [37] with
the semantic information provided by the PPR expansion. The relevance score
ρ(c, µ)for a candidate chunkc, given the queryµ, is defined as the sum of the
BM25 contributions of the tokens ofΦ µpresent inc, weighted by their PPR:
ρ(c, µ) =X
v∈Φµ∩Vcˆp(v)·BM25(v, c)(3.7)
Here,V cis the set of tokens withinc,BM25(v, c)represents the standard
BM25 score contribution of the tokenvofc, andˆp(v)acts as a topological
modulating factor derived from the PPR distribution. This formulation ensures
that candidate chunks are ranked not merely by superficial lexical overlap but
also by the structural relevance of the underlying concepts.
8

3.6. Dynamic Candidate Pooling and Reranking
Although graph-weighted scoring effectively retrieves a broad set of contex-
tually relevant chunks, relying strictly on a fixed cutoff can introduce noise. To
dynamically filter the candidate set, we apply a cumulative score thresholding
mechanism.
LetC µbe the list of retrieved candidate chunks sorted in descending order
by their hybrid scoreρ(·,·). First, we calculate the total retrieval score mass
ρµtotalas the sum of all the scores of all chunks inC µ:
ρµtotal=X
ck∈Cµρ(ck, µ)(3.8)
Then, we construct a refined candidate poolP µof chunks by selecting the
topαchunks such that their cumulative score reaches a predefined fractional
thresholdth ρ:
α= min
m=1..|C µ|Pm
k=1ρ(ck, µ)
ρµtotal≥th ρ
(3.9)
This dynamic cutoff ensures that TIGRAG retains only the most signifi-
cant chunks, automatically adjusting the size of the pool based on the model’s
confidence level.
Finally, to ensure the maximum semantic fidelity to the queryµ, we perform
a neural reranking on the dynamic poolP µ. To this end, we use the embedding
modelMto project the natural language queryµinto the same continuous
vector space as the precomputed chunk embeddings, yieldingϵ µ=M(µ)∈Rδ.
For each candidate chunkc∈ P µwith an embeddingϵ c, we determine the
final ranking by computing the cosine similarityσ cµbetween the vectorsϵ cand
ϵµ. After constructingP µ, we reorder its chunksc∈ P µin descending order
according to the cosine similarityσ cµ. This yields the definitive ranking for the
initial retrieval phase that returns the results of the one-hop query. We call ˆPµ
the ordered list of chunks thus obtained.
3.7. Multi-Hop Reasoning and Entity-Driven Query Expansion
To address complex, multi-hop queries, i.e., queries that may lack the ex-
plicit vocabulary required to retrieve all supporting evidence, we introduce an
iterative, entity-driven retrieval mechanism. At each hopj, letHj
µdenote the
list of the topιretrieved chunks sorted by the similarity degrees between their
embeddingsandtheembeddingofµ. Clearly,H1
µcoincideswiththetopιchunks
ofˆPµ. Finally, we denote byσj
cµthe cosine similarity between the embeddings
ϵcandϵ µat the hopj. Clearly,σ1
cµcoincides withσ cµ. We process the chunks
ofHj
µthrough a Named Entity Recognition (NER) pipeline to extract the set
of salient entities:
Nj
µ=NER(Hj
µ)(3.10)
To avoid semantic looping and encourage the discovery of new evidence, we
only retain entities that have not yet been seen:
9

Nj
µ=Nj
µ\j−1[
r=1Nr
µ (3.11)
IfNj
µ=∅, the retrieval process terminates. Otherwise, we update the
cumulative entity set:
Nj
µtotal=j[
r=1Nr
µ (3.12)
We assume thatNj
µtotalconsists ofzentities:
Nj
µtotal={e 1, e2,···, e z}(3.13)
Lettj
µtotal=L
x=1..zexdenote the textual concatenation of all entities in
Nj
µtotal. We then construct the expanded query:
µj
exp=µ⊕tj
µtotal(3.14)
together with an expanded set of embeddings:
Ej
µexp={M(µ⊕e x)|ex∈ Nj
µtotal}(3.15)
We use the expanded queryµj
expto reapply the same graph-based retrieval
and candidate selection procedure described previously. Instead, the neural
reranking stage works on the expanded embedding setEj
µexp. This design main-
tains the semantic connection to the original query throughout the multi-hop
exploration process and prevents the expanded entity context from overriding
the retrieval objective. The resulting ranked chunks define the next retrieval
hopHj+1
µ.
Finally, we merge the retrieved candidate sets from all executed hops. In
particular, lethbe the number of executed hops. For any chunkcdetected
during the retrieval hops, we retain its maximum relevance score across all
hops:
σfinal
cµ = max
u=1..hσu
cµ (3.16)
After computing the final relevance scores for all chunks, we construct a list
Lµof chunks sorted by their corresponding final scores in a descending order.
Finally, we construct the listHfinal
µby selecting the topβchunks fromL µ.
Hfinal
µrepresents the final context window.
4. Experimental Campaign
In this section, we describe the experimental campaign that we conducted to
validate TIGRAG. Specifically, Subsection 4.1 presents the experimental setup,
and Subsection 4.2 illustrates the obtained results.
10

4.1. Experimental Setup
In this section, we describe the experimental setup for our tests. Specifically,
we outline the datasets used, the baselines considered, the model and prompting
protocol, the evaluated metrics, and finally the hyperparameter configuration.
We conducted all experiments on a server with 256 AMD EPYC 7742 CPUs, 1
TB of RAM, and two NVIDIA A100 GPUs, each with 40 GB of memory. To en-
sure a fair comparison among the evaluated methods, we executed all indexing,
retrieval, and generation experiments on the same hardware infrastructure.
4.1.1. Datasets
We evaluated TIGRAG’s retrieval and reasoning capabilities across three
widely adopted multi-hop Question Answering datasets: HotpotQA [46], 2Wiki-
MultiHopQA [15], and MuSiQue [40]. To mitigate the computational overhead
while maintaining statistical significance, we subsampled1,000queries from
each dataset, following standard practices from previous studies [13, 41, 31].
Each dataset provided questions to be answered and a corpus of associated doc-
uments. This corpus served as the evidence pool from which relevant context
had to be retrieved. Furthermore, each query was paired with a ground-truth
answer, enablingarigorousassessmentduringthedownstreamevaluationphase.
Table 1 provides a comprehensive overview of the statistics of the graph
generated from each dataset.
Method HotpotQA 2WikiMultiHopQA MuSiQue
Number of nodes 55,733 35,251 56,467
Number of edges 1,315,330 658,326 1,370,777
Average degree 47.20 37.35 48.55
Average number of documents per node 8.15 6.99 8.18
Number of chunks 11,851 7,870 13,067
Density 0.0008 0.0011 0.0009
Number of connected components 1 1 1
Table 1: Graph statistics for each dataset
4.1.2. Baselines
To rigorously evaluate the effectiveness of TIGRAG, we benchmarked it
against a diverse set of state-of-the-art RAG methods, covering both conven-
tional dense retrieval techniques and more advanced structure-aware architec-
tures. Specifically, we usedNaiveRAGas a standard dense retrieval base-
line, and employedRAPTOR[35],GraphRAG[5],LightRAG[11],ToG[36],
KGP[42], andHippoRAG2[12] as recent approaches incorporating hierarchical,
graph-based, or knowledge-guided retrieval mechanisms. We also considered the
case in which no RAG is used. To ensure fairness, we used the same generation
prompt, designed to strictly enforce evidence-grounded reasoning. It is shown
in Figure 2.
11

4.1.3. Models and prompting protocol
The LLM we used for our experiments wasgemma3:27b-it-qat4. During
the generation phase, we supplied it with the user question and the retrieved
context via a prompt template shown in Figure 2. Finally, to ensure a fair
comparison, we standardized the generation stage across all three evaluated
baselinesbyapplyinganequivalentpromptingstrategy. WeemployedtheNomic
Text 1.5 model5for dense retrieval and semantic similarity computations. We
performed tokenization using NLTK, which is based on the Treebank tokenizer
and Punkt sentence segmentation. We extracted named entities using the spacy
NER pipeline.
You are a strict information extraction assistant.
Examine the context to find the exact answer.
You MUST follow this format:
Thought: <your step-by-step reasoning>
Answer: <the concise answer only>
If the answer is in the text, you must find it. Do not chat.
Example:
Context:
Wikipedia Title: The Last Horse
The Last Horse (Spanish: El último caballo) is a 1950 Spanish comedy film directed by
Edgar Neville starring Fernando Fernán Gómez.
Question: Who directed The Last Horse?
Thought: The text states that the film ‘The Last Horse’ was directed by Edgar Neville.
Answer: Edgar Neville.
–-
Context: <CONTEXT_STR>
Question: <QUESTION>
Thought:
Answer:
Figure 2: Prompt template used during the generation phase for Question Answering.
<CONTEXT_STR>represents the retrieved context, while<QUESTION>is the question to be an-
swered.
4.1.4. Metrics
Following the procedure of [13], we evaluated TIGRAG across three dimen-
sions: retrieval efficacy, downstream generation quality, and computational effi-
ciency.
We quantified retrieval performance using Recall@2 (R@2) and Recall@5
(R@5), which measure the proportion of queries for which gold-standard sup-
porting evidence is successfully retrieved within the top two (R@2) or top five
(R@5) candidate chunks. For the generative Question Answering phase, we as-
sessed performance using Exact Match (EM) and token-level F1. EM evaluates
strict string equivalence, while token-level F1 evaluates the overlap between the
LLM’s output and the ground-truth answers at the token level. All of these
4https://ollama.com/library/gemma3:27b-it-qat
5https://ollama.com/library/nomic-embed-text:v1.5
12

metrics fall within the real range[0,1], or[0,100]if we express them in percent-
ages; the higher their value, the better the performance. We omitted Recall@k
for KGP, ToG, and RAPTOR because their structural retrieval mechanisms
are incompatible with chunk-level evaluation. In fact, KGP and ToG retrieve
KG entities and relationships, while RAPTOR retrieves hierarchical summary
nodes. Thus, these three systems do not permit reliable evaluation of retrieval
quality through Recall. Finally, we omitted Recall@k for NoRAG because this
system has no retrieval mechanism.
In addition to accuracy, we evaluated the computational load of RAG ap-
proaches. Specifically, we measured the time required for offline graph construc-
tion in order to evaluate scalability with large corpora. For the online phase, we
reported end-to-end inference latency, capturing both retrieval and LLM gen-
eration. We also measured the prompt footprint as the average token length of
the retrieved context. This computational metric is critical because reducing
the context footprint not only lowers computational overhead and latency but
also helps mitigate the loss of focus during generation.
4.1.5. Hyperparameter configuration
We evaluated TIGRAG using a fixed hyperparameter configuration. In par-
ticular, we set the thresholdth ρto 0.8 (see Equation 3.9), the hyperparameterι
(denoting the cardinality ofHj
µ- see Section 3.7) to1and the numberhof hops
to2(see Section 3.7). Furthermore, we set the hyperparameterβ(denoting the
number of chunks composing the final context window - see Section 3.7) to 5.
We also set the lengthwof the token co-occurrence window to 4. Finally, we
set the window sizesto 6 and the overlapoto 2; therefore, the stride length
lwas set to 4 (see Section 3.2). In Section 4.2.3, we report a hyperparameter
and ablation study supporting this configuration.
4.2. Results
In this section, we present the results obtained from our experimental cam-
paign. Specifically, Subsection 4.2.1 compares the performance of TIGRAG
with that of other related systems proposed in the literature. Subsection 4.2.2
presents a comparison of the computational resources required by TIGRAG and
other related approaches. Finally, Subsection 4.2.3 presents a hyperparameter
and ablation study of TIGRAG.
4.2.1. Performance Comparison
Table 2 reports the retrieval performance of the compared methods in terms
of R@2 and R@5 across the considered multi-hop Question Answering bench-
marks. For each column, the number in bold indicates the maximum value,
while the underlined number denotes the sub-maximum value.
This table shows that TIGRAG outperforms all baselines consistently across
all evaluated scenarios. On HotpotQA, it improves R@2 by 9.70% and R@5 by
8.23% compared to NaiveRAG. This trend continues on 2WikiMultiHopQA,
where TIGRAG achieves a 13.41% gain on R@2 over NaiveRAG, and a 10.76%
13

Method HotpotQA 2WikiMultiHopQA MuSiQue Average
R@2 R@5 R@2 R@5 R@2 R@5 R@2 R@5
HippoRAG2 57.45% 84.05% 63.42% 78.73% 35.50% 51.29% 52.12% 71.36%
LightRAG 69.05% 85.25% 59.98% 69.58% 35.62% 49.68% 54.88% 68.17%
NaiveRAG 71.65% 85.05% 64.35% 70.40% 38.39% 51.21% 58.13% 68.89%
TIGRAG78.60% 92.05% 72.98% 87.20% 42.57% 56.28% 64.72% 78.51%
Table2: Retrievalperformance(R@2andR@5)onmulti-hopQuestionAnsweringbenchmarks
gain on R@5 over HippoRAG2. On the more challenging MuSiQue benchmark,
which requires multi-step semantic reasoning, TIGRAG’s margin of improve-
ment increases further, with a 19.92% gain on R@2 and a 9.73% gain on R@5
over HippoRAG2. These results demonstrate the effectiveness of token-level
topological expansion in enhancing multi-hop retrieval.
Afterwards, we shifted our focus from retrieval metrics to downstream gen-
erative performance. Table 3 shows the Question Answering performance of the
compared methods on the multi-hop benchmarks using Exact Match (EM) and
token-level F1 (F1).
Method HotpotQA 2WikiMultiHopQA MuSiQue Average
EM F1 EM F1 EM F1 EM F1
NoRAG 28.30% 37.93% 30.90% 35.20% 7.40% 17.59% 22.20% 30.24%
HippoRAG2 52.80% 64.40% 50.20% 55.57% 26.50%33.89% 43.17% 51.29%
Raptor 39.80% 55.69% 32.40% 38.24% 14.40% 24.21% 28.87% 39.38%
KGP 22.50% 32.26% 11.40% 13.08% 11.10% 18.15% 15.00% 21.16%
ToG 20.60% 26.63% 14.50% 17.02% 5.80% 8.60% 13.63% 17.42%
GraphRAG 29.99% 39.91% 7.60% 8.41% 0.50% 1.67% 12.70% 16.66%
LightRAG 52.50% 66.91% 45.70% 51.17% 22.20% 32.14% 40.13% 50.07%
NaiveRAG 53.40% 64.84% 44.40% 48.49% 21.00% 29.88% 39.60% 47.74%
TIGRAG60.70% 72.60% 50.90% 58.60%26.20% 35.80% 45.93% 55.67%
Table 3: Question answering performance (EM and F1) on multi-hop Question Answering
benchmarks
As shown in Table 3, TIGRAG improves answer quality on HotpotQA con-
sistently, achieving gains of 13.67% in EM and 11.97% in F1 over NaiveRAG.
A similar trend is observed on 2WikiMultiHopQA, where TIGRAG surpasses
HippoRAG2 by 1.39% in EM and 5.45% in F1. On the MuSiQue benchmark,
HippoRAG2 obtains a marginally higher EM value, but TIGRAG preserves a
higher F1 value. Since EM heavily penalizes partially correct but more detailed
answers, TIGRAG’s stronger F1 results suggest that our framework provides
more comprehensive and semantically reliable evidence for multi-hop reasoning.
4.2.2. Resource Comparison
The practical viability of a RAG architecture depends not only on retrieval
accuracy but also on computational efficiency. After analyzing retrieval perfor-
mance in Section 4.2.1, we now focus on resource consumption during the offline
and online phases. To this end, we analyze three critical dimensions: offline
graph construction time (also referred to as indexing time), online end-to-end
inference latency, and prompt footprint (quantified by the retrieved context’s
length expressed in number of tokens). These metrics evaluate TIGRAG’s scala-
14

bility and practical usability. Offline indexing time measures the cost of process-
inglargecorpora, whileend-to-endinferencelatencycapturestheresponsiveness
of the entire retrieval-generation pipeline. Excessively large contexts can slow
down generation and distract the LLM; thus, reducing the context footprint is
essential for improving both efficiency and reasoning quality. Table 4 reports
the offline indexing time and the average online inference latency per query ob-
tained by TIGRAG and related approaches when performing the tests described
in Section 4.2.1.
Method HotpotQA 2WikiMultiHopQA MuSiQue
Indexing (s) Retrieval (s) Indexing (s) Retrieval (s) Indexing (s) Retrieval (s)
HippoRAG2 308,664 75.60 121,945 79.48 318,282 81.67
Raptor 29,452 14.80 17,001 14.56 35,851 15.46
KGP 1,893 30.57 1,514 25.90 4,096 37.16
ToG 139,372 55.27 70,660 39.91 146,146 48.58
GraphRAG 657,055 14.91 74,802 15.75 195,325 15.75
LightRAG 721,833 20.31 343,812 19.96 593,942 20.46
NaiveRAG102 3.35 663.71 1144.45
TIGRAG 191 3.72 155 3.26202 4.14
Table 4: Indexing and retrieval time; the latter includes both the effective retrieval time and
the LLM response time
The analysis of this table shows that NaiveRAG has the shortest indexing
time in the offline phase due to its simple, dense-retrieval architecture, which
bypasses graph construction. Although TIGRAG is slower than NaiveRAG,
it is still substantially faster than other graph-based methods while achieving
stronger retrieval and generation performance (see Table 2).
A compelling trend emerges during the online inference phase. On Hot-
potQA, TIGRAG is slightly slower than NaiveRAG. However, on more com-
plex datasets, such as 2WikiMultiHopQA and MuSiQue, TIGRAG surpasses
NaiveRAG in end-to-end speed. This efficiency gain validates our hypothe-
sis that, by dynamically extracting highly concentrated and relevant context,
TIGRAG can drastically reduce the LLM’s prompt processing and generation
time. This reduction more than offsets the minimal latency introduced by the
graph traversal necessary for PPR. Compared to RAPTOR, the fastest graph-
based baseline, TIGRAG is roughly four times faster, setting a highly scalable
standard for graph-augmented RAG systems.
Table 5 details the average context footprint, measured in number of tokens,
generated per query.
Asthistableshows, approachesthatrelystrictlyondynamicallyfilteredtext
chunks produce a significantly more compact context. In contrast, frameworks
that augment their prompts by injecting extensive sets of KG triplets or commu-
nity summaries, such as GraphRAG and LightRAG, suffer from highly inflated
context windows. TIGRAG reduces prompt size by relying on a compact and
highly relevant chunk pool, thereby improving inference speed and limiting the
loss of focus.
15

Method HotpotQA 2WikiMultiHopQA MuSiQue
HippoRAG2 579.45 639.43 656.61
Raptor 867.88 860.08 869.98
GraphRAG 3,843.83 888.46 1421.63
KGP 2,501.16 367.28 2,323.68
ToG117.96 117.10 115.30
LightRAG 5,893.51 5,579.22 7,304.93
NaiveRAG 742.80 719.44 800.26
TIGRAG 578.02 570.66635.41
Table 5: Average context length (in number of tokens) across datasets
4.2.3. Hyperparameter and Ablation Study
First, we examine how the retrieval budget affects the way an LLM con-
structs context, with a specific focus on the numberβof chunks supplied to
the LLM for context construction (see Section 3.7). By varyingβ, we evaluate
the tradeoff between providing sufficient supporting evidence and introducing
semantic noise. Thus, we assess how context size ultimately influences EM and
F1 scores. In Table 6, we present the results obtained from this experiment.
In each column, we show in bold the best value obtained for the correspond-
ing metric. The results of this table highlight that performance peaks at an
intermediate value ofβ(β= 5). Increasing the context window beyond this
threshold decreases both EM and F1 scores. This behavior underscores LLMs’
susceptibility to distraction when exposed to overly long and noisy contexts.
Conversely, settingβtoo low starves the generative model of necessary sup-
porting evidence. Thus,β= 5is the optimal configuration, which maximizes
informational density without overwhelming the reasoning model’s capacity.
HotpotQA 2WikiMultiHopQA MuSiQue
βEM F1 EM F1 EM F1
3 57.40% 70.02% 45.20% 52.41% 23.30% 32.68%
560.70% 72.60% 50.90% 58.60% 26.20% 35.80%
10 55.80% 69.21% 44.90% 52.66% 22.80% 33.09%
Table 6: Sensitivity analysis for the hyperparameterβ
Table 7 reports the impact of the hyperparameterι, which denotes the cardi-
nality ofHj
µ(see Section 3.7) on the effectiveness of retrieval, measured in terms
of R@2 and R@5. In this table, we note thatιdirectly influences the precision
of the retrieval space. Empirical results indicate that the optimal configuration
is achieved withι= 1across most scenarios. This suggests that the essential
bridging context is concentrated in the highest-ranked chunk, and incorporating
subsequent chunks introduces distracting entities. In fact, in the latter case, the
entity pool used for query augmentation excessively expands, inevitably intro-
ducing semantic drift and degrading overall performance. The only exception
is observed on the 2WikiMultiHopQA dataset, where R@5 peaks atι= 2.
Table8showshowthehyperparameterh, whichdenotesthenumberofquery
hops (see Section 3.7), affects the TIGRAG’s retrieval performance. We carried
16

ιHotpotQA 2WikiMultiHopQA MuSiQue
R@2 R@5 R@2 R@5 R@2 R@5
178.60% 92.05% 72.98%87.20%42.57% 56.28%
2 74.05% 91.45% 63.52%90.58%37.78% 55.17%
3 68.00% 91.10% 55.13% 88.18% 33.92% 55.16%
Table 7: Sensitivity analysis for the hyperparameterι
out this evaluation considering one, two, and three hops, aligning this hyperpa-
rameter with the multi-hop complexity of the underlying benchmarks. We did
not explore beyond three hops because that exceeds the maximum ground-truth
reasoning chain required by the selected datasets. Thus, additional hops would
primarily introduce computational overhead and semantic noise rather than
useful evidence. As shown in this table, using two hops yields the best tradeoff
between semantic expansion and topological noise across all benchmarks.
hHotpotQA 2WikiMultiHopQA MuSiQue
R@2 R@5 R@2 R@5 R@2 R@5
1 71.50% 85.55% 61.92% 70.27% 38.01% 50.72%
278.60% 92.05% 72.98% 87.20% 42.57% 56.28%
3 78.40% 91.90% 72.75% 87.18% 42.47% 56.19%
Table 8: Sensitivity analysis for the hyperparameterh
Table 9 evaluates the effect of the cumulative score thresholdth ρused in
the candidate pooling stage (see Equation 3.9). As shown in this table, higher
values ofth ρimprove retrieval effectiveness by enabling broader semantic explo-
ration before the final reranking stage, filtering out the least relevant evidence.
Performance consistently peaks whenth ρ= 0.8, suggesting that a large and di-
verse candidate pool benefits multi-hop reasoning. A lower value ofth ρ, instead,
would introduce overly aggressive pruning and would limit access to intercon-
nected evidence.
thρ HotpotQA 2WikiMultiHopQA MuSiQue
R@2 R@5 R@2 R@5 R@2 R@5
0.1 78.30% 91.70% 72.78% 87.05% 41.83% 54.49%
0.2 78.55% 92.05% 72.72% 86.92% 42.05% 55.09%
0.578.60% 92.10%72.52% 86.75% 42.42% 55.91%
0.878.60%92.05%72.98% 87.20% 42.57% 56.28%
0.978.60%92.05% 72.92% 87.15% 42.57% 56.28%
Table 9: Sensitivity analysis for the hyperparameterth ρ
Table 10 presents an ablation study designed to isolate the contributions
of TIGRAG’s main components. Specifically, we evaluated:(i)a TIGRAG
variant without BM25, where retrieval relies solely on graph topology-aware
matching;(ii)a TIGRAG variant based on degree centrality instead of PPR-
drivenexpansion; and(iii)aTIGRAGvariantwithoutthefinalneuralreranking
stage.
17

Approach HotpotQA 2WikiMultiHopQA MuSiQue
R@2 R@5 R@2 R@5 R@2 R@5
TIGRAG without BM25 78.55% 92.00% 72.92% 87.15% 42.33% 55.91%
TIGRAG based on degree centrality 22.00% 24.15% 44.97% 51.98% 12.86% 16.54%
TIGRAG without reranking 51.10% 69.00% 48.18% 64.32% 29.53% 41.40%
TIGRAG78.60% 92.05% 72.98% 87.20% 42.57% 56.28%
Table 10: Ablation study evaluating the contribution of the main components of TIGRAG on
retrieval performance across the considered benchmarks
The ablation results highlight the importance of each component of the pro-
posed framework. Removing BM25 yields only marginal performance drops,
confirming that the graph-based expansion mechanism alone captures highly
relevant evidence. Replacing PPR with degree centrality dramatically decreases
performance across all datasets, demonstrating the importance of topology-
aware semantic propagation over simple node degree. Finally, eliminating the
neural reranking stage significantly reduces retrieval effectiveness, indicating
that the reranking is essential for filtering out noisy candidates introduced dur-
ing graph expansion.
5. Discussion
From a performance point of view, the topological approach underlying
TIGRAG enhances retrieval and consistently outperforms baselines. This high-
fidelity retrieval directly translates into superior downstream Question Answer-
ing generation, setting new state-of-the-art Exact Match and token-level F1
score benchmarks. Beyond strict accuracy, the most significant implication of
TIGRAG lies in its resource efficiency. By constructing the structural graph
solely through statistical co-occurrences, TIGRAG eliminates the need for the
computationally expensive LLM-driven entity extraction pipelines required by
traditional GraphRAG approaches. Consequently, it consumes a fraction of the
time and token budget during offline graph construction and online inference,
offering a highly scalable solution for real-world deployment.
Despite these promising results, TIGRAG has some limitations that suggest
interesting directions for future research. First, TIGRAG’s iterative, entity-
drivenretrievalstrategyreliesheavilyonthequalityoftheNamedEntityRecog-
nition (NER) pipeline. Since query expansion is guided by entities extracted
from previously retrieved chunks, inaccurate, ambiguous, or irrelevant entities
can spread across retrieval stages, resulting in semantic drift and reduced re-
trieval precision. Therefore, exploring more robust entity disambiguation and
filtering mechanisms is an important direction for future work. Second, as with
other co-occurrence-based approaches, TIGRAG primarily captures relation-
ships emerging from token proximity. Future work will investigate lightweight
semantic enrichment strategies that can better model implicit connections while
preserving TIGRAG’s computational efficiency.
18

6. Conclusion
In this work, we presented TIGRAG, an effective and highly efficient RAG
framework driven by a token co-occurrence KG. Unlike traditional GraphRAG
approaches, which rely on expensive LLM-based entity and relation extraction
pipelines, TIGRAG constructs a lightweight topological representation directly
from token co-occurrences. It also uses PPR-based semantic expansion and dy-
namic neural reranking to retrieve interconnected evidence across multiple hops.
By bypassing computationally expensive LLM-based graph construction index-
ing, the token-level approach underlying TIGRAG enables faster offline graph
construction while preserving the structural dependencies necessary for multi-
hop reasoning. Empirical evaluations across HotpotQA, 2WikiMultiHopQA,
and MuSiQue demonstrate that TIGRAG significantly improves retrieval per-
formance, establishing new state-of-the-art performance compared to baseline
models. These findings prove that graph-augmented RAG systems can solve
complex, multi-hop reasoning tasks without sacrificing computational efficiency.
Our current evaluation focuses on textual, multi-hop datasets; however, this
study suggests several promising directions for future research. First, we plan
to expand the graph topology beyond statistical co-occurrences by introducing
semantic edges. This semantic smoothing will connect lexically distinct but
semantically similar tokens, capturing implicit relationships. Second, we plan
to address the inherent limitations of treating tokens as only lexical units, by
introducing semantic-aware nodes to address token polysemy. This will allow
the graph to dynamically disambiguate meanings for more granular and precise
retrieval. Finally, since modern information ecosystems are heterogeneous by
nature, a future development is to extend TIGRAG to multimodal settings.
This will involve integrating images and videos into the topological space to
support more complex QA tasks.
Declaration of competing interest
Theauthorsdeclarethattheyhavenoknowncompetingfinancialinterestsor
personal relationships that could have appeared to influence the work reported
in this paper.
Data availability
The code used for our study will be available after publication.
References
[1] V. Agrawal, F. Wang, and R. Puri. Query-Aware Graph Neural Net-
works for Enhanced Retrieval-Augmented Generation.arXiv preprint
arXiv:2508.05647, 2025.
19

[2] P.J. Ammann, J. Golde, and A. Akbik. Question decomposition for
retrieval-augmented generation. InProc. of the Annual Meeting of the As-
sociation for Computational Linguistics (ACL’25), pages 497–507, Vienna,
Austria, 2025.
[3] H. Cheng, K. Wang, Q. Wang, T. Liu, and K. Sheng. A Chinese financial
eventknowledgegraph-basedretrieval-augmentedgenerationframeworkfor
financial question answering.Engineering Applications of Artificial Intelli-
gence, 175:114670, 2026. Elsevier.
[4] P.K. Choubey, X. Su, M. Luo, X. Peng, C. Xiong, T. Le, S. Rosenman,
V. Lal, P.L. Mui, R. Ho, P. Howard, and C.S. Wu. Scaling Knowl-
edge Graph Construction through Synthetic Data Generation and Distilla-
tion. InProc. of the International Conference on Learning Representations
(ICLR’26), Rio de Janeiro, Brasil, 2025.
[5] D. Edge, H. Trinh, N. Cheng, J. Bradley, A. Chao, A. Mody, S. Tru-
itt, D. Metropolitansky, R.O. Ness, and J. Larson. From local to global:
A graph RAG approach to query-focused summarization.arXiv preprint
arXiv:2404.16130, 2024.
[6] W. Fan, Y. Ding, L. Ning, S. Wang, H. Li, D. Yin, T. Chua, and Q. Li. A
survey on RAG meeting LLMS: Towards retrieval-augmented large lan-
guage models. InProc. of the ACM SIGKDD Conference on Knowl-
edge Discovery and Data Mining (KDD’24), pages 6491–6501, Barcelona,
Catalunya, Spain, 2024.
[7] Y. Fang, T. Sun, Y. Shi, and X. Gu. Attentionrag: Attention-
guided context pruning in retrieval-augmented generation.arXiv preprint
arXiv:2503.10720, 2025.
[8] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, M. Wang,
and H. Wang. Retrieval-augmented generation for large language models:
A survey.arXiv preprint arXiv:2312.10997, 2(1):32, 2023.
[9] H. Ghanem and C. Cruz. Fine-tuning or prompting on llms: evaluating
knowledge graph construction task.Frontiers in Big Data, 8:1505877, 2025.
Frontiers Media SA.
[10] T. Guo, C. Wang, Q. Yang, J. Tang, Y. Liu, and Y. Wen. WADSeg:
Exploiting Weak Attention Associations for Enhanced Knowledge Segmen-
tation in RAG.Expert Systems with Applications, page 129297, 2025. El-
sevier.
[11] Z. Guo, L. Xia, Y. Yu, T. Ao, and C. Huang. LightRAG: Simple and Fast
Retrieval-Augmented Generation. InFindings of the Association for Com-
putational Linguistics: EMNLP 2025, pages 10746–10761, Suzhou, China,
2025.
20

[12] B.J.Guti’errez, Y.Shu, W.Qi, S.Zhou, andY.Su. FromRAGtoMemory:
Non-Parametric Continual Learning for Large Language Models. InProc.
of the International Conference on Machine Learning (ICLR’25), pages
21497–21515, Singapore, 2025.
[13] B.J. Gutiérrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su. Hipporag: Neu-
robiologically inspired long-term memory for large language models. In
Proc. of the Annual Conference on Neural Information Processing Sys-
tems (NeurIPS’24), volume 37, pages 59532–59569, Vancouver, British
Columbia, Canada, 2024.
[14] T.H. Haveliwala. Topic-sensitive pagerank. InProc. of the International
Conference on World Wide Web (WWW’02), pages 517–526, Honolulu, HI,
USA, 2002.
[15] X. Ho, A.D. Nguyen, S. Sugawara, and A. Aizawa. Constructing a multi-
hop qa dataset for comprehensive evaluation of reasoning steps. InProc.
of the International Conference on Computational Linguistics (ICLR’20),
pages 6609–6625, Barcelona, Catalunya, Spain, 2020.
[16] Y. Hu, Z. Lei, Z. Zhang, B. Pan, C. Ling, and L. Zhao. GRAG: Graph
Retrieval-Augmented Generation. InFindings of the Association for Com-
putational Linguistics: NAACL 2025, pages 4145–4157, Albuquerque, NM,
USA, 2025. ACL.
[17] Y.B. Jeong, H.I. Seo, Y.H. Kim, and W.Y. Kim. Retrieval-augmented
visual parcel invoice understanding transformer for address correction.En-
gineering Applications of Artificial Intelligence, 158:111542, 2025. Elsevier.
[18] Y. Ji, K. Wu, J. Li, W. Chen, M. Zhong, J. Xu, and M. Zhang. Retrieval
and Reasoning on KGs: Integrate Knowledge Graphs into Large Language
Models for Complex Question Answering. InFindings of the Association
for Computational Linguistics: EMNLP 2024, pages 7598–7610, Miami,
FL, USA, 2024. ACL.
[19] B. Jin, C. Xie, J. Zhang, K.K. Roy, Y. Zhang, Z. Li, R. Li, X. Tang,
S. Wang, Y. Meng, and J. Han. Graph chain-of-thought: Augmenting large
language models by reasoning on graphs. InFindings of the Association for
Computational Linguistics: ACL 2024, pages 163–184, St. Julian’s, Malta,
2024. ACL.
[20] V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen,
and W. Yih. Dense passage retrieval for open-domain question answering.
InProc. of the International Conference on Empirical Methods in Natural
Language Processing (EMNLP’20), pages 6769–6781, online, 2020.
[21] Y. Kuratov, A. Bulatov, P. Anokhin, D. Sorokin, A. Sorokin, and M. Burt-
sev. In search of needles in a 11m haystack: Recurrent memory finds what
llms miss.arXiv preprint arXiv:2402.10790, 2024.
21

[22] J. Linders and J.M. Tomczak. Knowledge graph-extended retrieval
augmented generation for question answering.Applied Intelligence,
55(17):1102, 2025.
[23] N.F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni,
and P. Liang. Lost in the middle: How language models use long contexts.
Transactions of the Association for Computational Linguistics, 12:157–173,
2024. MIT Press.
[24] H. Luo, G. Chen, Y. Zheng, X. Wu, Y. Guo, Q. Lin, Y. Feng, Z. Kuang,
M. Song, Y. Zhu, and A.T. Luu. HyperGraphRAG: Retrieval-Augmented
Generation via Hypergraph-Structured Knowledge Representation. In
Proc. of the Annual Conference on Neural Information Processing Systems
(NeurIPS’25), San Diego, CA, USA, 2025.
[25] L. Luo, Z. Zhao, G. Haffari, D. Phung, C. Gong, and S. Pan. GFM-
RAG: Graph Foundation Model for Retrieval Augmented Generation. In
Proc. of the Annual Conference on Neural Information Processing Systems
(NeurIPS’25), San Diego, CA, USA, 2025.
[26] C. Ma, Y. Chen, T. Wu, A. Khan, and H. Wang. Large language models
meet knowledge graphs for question answering: Synthesis and opportuni-
ties. InProc. of the Conference on Empirical Methods in Natural Language
Processing (EMNLP’25), pages 24589–24608, Suzhou, China, 2025.
[27] M. S. Malik, V. The Le, and Y. Y. Ou. Enhancing the classification of
metal-bindingresidueinproteinswithretrieval-augmentedgeneration, pro-
tein language models, and deep learning.Engineering Applications of Ar-
tificial Intelligence, 171:114330, 2026. Elsevier.
[28] C. Min, S. Bansal, J. Pan, A. Keshavarzi, R. Mathew, and A.V. Kannan.
Towards Practical GraphRAG: Efficient Knowledge Graph Construction
and Hybrid Retrieval at Scale.arXiv preprint arXiv:2507.03226, 2025.
[29] B. Peng, Y. Zhu, Y. Liu, X. Bo, H. Shi, C. Hong, Y. Zhang, and S. Tang.
Graph retrieval-augmented generation: A survey.ACM Transactions on
Information Systems, 44(2):1–52, 2025. ACM.
[30] G. Perkovi’c, A. Drobnjak, and I. Botički. Hallucinations in llms: Un-
derstanding and addressing challenges. InProc. of the MIPRO ICT and
Electronics Convention (MIPRO’24), pages 2084–2088, Opatija, Croatia,
2024. IEEE.
[31] O. Press, M. Zhang, S. Min, L. Schmidt, N. A. Smith, and M. Lewis.
Measuring and narrowing the compositionality gap in language models. In
Findings of the Association for Computational Linguistics: EMNLP 2023,
pages 5687–5711, Singapore, 2023.
22

[32] O. Ram, Y. Levine, I. Dalmedigos, D. Muhlgay, A. Shashua, K. Leyton-
Brown, and Y. Shoham. In-context retrieval-augmented language models.
Transactions of the Association for Computational Linguistics, 11:1316–
1331, 2023. MIT Press.
[33] A.O. Saleh, G. Tur, and Y. Saygin. SG-RAG: Multi-hop question answer-
ing with large language models through knowledge graphs. InProc. of
the International Conference on Natural Language and Speech Processing
(ICNLSP’24), pages 439–448, Trento, Italy, 2024.
[34] B. Sarmah, D. Mehta, B. Hall, R. Rao, S. Patel, and S. Pasquali. Hy-
bridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented
GenerationforEfficientInformationExtraction. InProc. of the ACM Inter-
national Conference on AI in Finance, (ICAIF’24), pages 608–616, Brook-
lyn, NY, USA, 2024. ACM.
[35] P. Sarthi, S. Abdullah, A. Tuli, S. Khanna, A. Goldie, and C.D. Man-
ning. RAPTOR: Recursive abstractive processing for tree-organized re-
trieval. InProc. of the International Conference on Learning Representa-
tions (ICLR’24), Vienna, Austria, 2024.
[36] J. Sun, C. Xu, L. Tang, S. Wang, C. Lin, Y. Gong, L. Ni, H. Shum, and
J. Guo. Think-on-Graph: Deep and Responsible Reasoning of Large Lan-
guageModelonKnowledgeGraph. InProc. of the International Conference
on Learning Representations (ICLR’24), Vienna, Austria, 2024.
[37] K.M. Svore and C.J. Burges. A machine learning approach for improved
BM25 retrieval. InProc. of the Proceedings of the ACM Conference on
Information and Knowledge Management (CIKM’09), pages 1811–1814,
Hong Kong, China, 2009.
[38] Y. Tang and Y. Yang. MultiHop-RAG: Benchmarking Retrieval-
Augmented Generation for Multi-Hop Queries. InProc. of the First Con-
ference on Language Modeling (COLM’24), Philadelphia, PA, USA, 2024.
[39] W. Tao, X. Xing, Y. Chen, L. Huang, and X. Xu. Treerag: Unleashing
the power of hierarchical storage for enhanced knowledge retrieval in long
documents. InFindings of the Association for Computational Linguistics:
ACL 2025, pages 356–371, Vienna, Austria, 2025.
[40] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal.MuSiQue:
Multihop Questions via Single-hop Question Composition.Transactions
of the Association for Computational Linguistics, 10:539–554, 2022. MIT
Press.
[41] H. Trivedi, N. Balasubramanian, T. Khot, and A. Sabharwal. Interleaving
retrievalwithchain-of-thoughtreasoningforknowledge-intensivemulti-step
questions. InProc. of the Annual Meeting of the Association for Com-
putational Linguistics (ACL’23), pages 10014–10037, Toronto, Ontario,
Canada, 2023.
23

[42] Y. Wang, N. Lipka, R. A. Rossi, A. Siu, R. Zhang, and T. Derr. Knowledge
graph prompting for multi-document question answering. InProc. of the
AAAI Conference on Artificial Intelligence (AAAI’24), volume 38, pages
19206–19214, Vancouver, British Columbia, Canada, 2024.
[43] Y. Wang, N. Lipka, R. Zhang, A.F. Siu, Y. Zhao, B. Ni, X. Wang, R.A.
Rossi, and T. Derr. Topology-aware Retrieval Augmentation for Text Gen-
eration. InProc. of the ACM International Conference on Information and
Knowledge Management, (CIKM’24), pages 2442–2452, Boise, ID, USA,
2024. ACM.
[44] Y. Wu, J. Bo, Y. Zhang, S. Liang, and Y. Liu. Query-Centric Graph
Retrieval Augmented Generation.arXiv preprint arXiv:2509.21237, 2025.
[45] Y. Xiao, J. Dong, C. Zhou, S. Dong, Q. Zhang, D. Yin, X. Sun,
and X. Huang. Graphrag-bench: Challenging domain-specific reason-
ing for evaluating graph retrieval-augmented generation.arXiv preprint
arXiv:2506.02404, 2025.
[46] Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. Cohen, R. Salakhutdinov, and
C.D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop
question answering. InProc. of the conference on empirical methods in
natural language processing (EMNLP’18), pages 2369–2380, Brussels, Bel-
gium, 2018.
[47] J. Yao, K. Ning, Z. Liu, M. Ning, Y. Liu, and L. Yuan. LLM lies: Halluci-
nations are not bugs, but features as adversarial examples.arXiv preprint
arXiv:2310.01469, 2023.
[48] T.Yu, A.Xu, andR.Akkiraju. IndefenseofRAGintheeraoflong-context
language models.arXiv preprint arXiv:2409.01666, 2024.
[49] S.H. Zhang and J.W. Lin. Direction-aware graph traversal for efficient
multi-hop retrieval-augmented generation.Engineering Applications of Ar-
tificial Intelligence, 181:115383, 2026. Elsevier.
[50] X. Zhu, Y. Xie, Y. Liu, Y. Li, and W. Hu. Knowledge Graph-Guided
Retrieval Augmented Generation. InProc. of the Conference of the Nations
of the Americas Chapter of the Association for Computational Linguistics:
Human Language Technologies (NAACL’25) - Volume 1: Long Papers,
pages 8912–8924, Albuquerque, NM, USA, 2025. ACL.
24