# GraphRAFT: Retrieval Augmented Fine-Tuning for Knowledge Graphs on Graph Databases

**Authors**: Alfred Clemedtson, Borun Shi

**Published**: 2025-04-07 20:16:22

**PDF URL**: [http://arxiv.org/pdf/2504.05478v1](http://arxiv.org/pdf/2504.05478v1)

## Abstract
Large language models have shown remarkable language processing and reasoning
ability but are prone to hallucinate when asked about private data.
Retrieval-augmented generation (RAG) retrieves relevant data that fit into an
LLM's context window and prompts the LLM for an answer. GraphRAG extends this
approach to structured Knowledge Graphs (KGs) and questions regarding entities
multiple hops away. The majority of recent GraphRAG methods either overlook the
retrieval step or have ad hoc retrieval processes that are abstract or
inefficient. This prevents them from being adopted when the KGs are stored in
graph databases supporting graph query languages. In this work, we present
GraphRAFT, a retrieve-and-reason framework that finetunes LLMs to generate
provably correct Cypher queries to retrieve high-quality subgraph contexts and
produce accurate answers. Our method is the first such solution that can be
taken off-the-shelf and used on KGs stored in native graph DBs. Benchmarks
suggest that our method is sample-efficient and scales with the availability of
training data. Our method achieves significantly better results than all
state-of-the-art models across all four standard metrics on two challenging
Q\&As on large text-attributed KGs.

## Full Text


<!-- PDF content starts -->

GraphRAFT: Retrieval Augmented Fine-Tuning for
Knowledge Graphs on Graph Databases
Alfred Clemedtson
Neo4j
alfred.clemedtson@neo4j.comBorun Shi
Neo4j
brian.shi@neo4j.com
Abstract
Large language models have shown remarkable language processing and reasoning
ability but are prone to hallucinate when asked about private data. Retrieval-
augmented generation (RAG) retrieves relevant data that fit into an LLM’s context
window and prompts the LLM for an answer. GraphRAG extends this approach to
structured Knowledge Graphs (KGs) and questions regarding entities multiple hops
away. The majority of recent GraphRAG methods either overlook the retrieval step
or have ad hoc retrieval processes that are abstract or inefficient. This prevents
them from being adopted when the KGs are stored in graph databases supporting
graph query languages. In this work, we present GraphRAFT, a retrieve-and-reason
framework that finetunes LLMs to generate provably correct Cypher queries to
retrieve high-quality subgraph contexts and produce accurate answers. Our method
is the first such solution that can be taken off-the-shelf and used on KGs stored in
native graph DBs. Benchmarks suggest that our method is sample-efficient and
scales with the availability of training data. Our method achieves significantly
better results than all state-of-the-art models across all four standard metrics on
two challenging Q&As on large text-attributed KGs.
1 Introduction
Large language models (LLMs) have shown remarkable ability to reason over natural languages.
However, when prompted with questions regarding new or private knowledge, LLMs are prone to
hallucinations[ 1]. A popular approach to remediate this issue is Retrieval-Augmented Generation
(RAG) [ 2]. Given a natural language question and a set of text documents, RAG retrieves a small set
of relevant documents and feeds them into an LLM.
In addition to unstructured data such as text documents, many of the real-world structured data
come in the form of Knowledge Graphs (KGs). KGs can be used to model a variety of domains
such as the Web, social networks, and financial systems. Knowledge Base Question Answering
(KBQA)[ 3] studies the methods used to answer questions about KGs. Most of the work studies
multi-hop questions on the Web (open-domain)[4, 5, 6].
Historically, there are two categories of approaches for KBQA. One approach is based on semantic
parsing[ 7,8,9,10,11,12,13], which involves converting natural language questions into logical
forms that are often SPARQL[ 14]. Another approach is based on information retrieval[ 15,16,17,18],
which involves embedding and retrieving data (mostly triplets) from KGs and reranking the extracted
data.
There is a growing demand to leverage the language understanding and reasoning power of LLMs
to improve open-domain KBQA. At the same time, GraphRAG extends the RAG setup to private
KGs. The methodologies began to merge, despite the different application areas. The vast majority of
Preprint. Under review.arXiv:2504.05478v1  [cs.LG]  7 Apr 2025

recent literature follows the general framework of 1) identifying some entities in the given question,
2) retrieving a subgraph and 3) prompting an LLM for reasoning.
[19,20] retrieves and reranks the top nodes (or k-hop ego-subgraphs) and prompts the LLM. [ 21]
prompts an LLM to generate relation path templates and retrieves concrete paths using in-memory
breadth-first search and fintunes the LLM. [ 22,23,24,25,26] identify starting entities and iteratively
perform retrieval step-by-step by prompting the LLM what is the next step to take. [ 27] prunes
a retrieved subgraph with parameterised graph algorithms and prompt an LLM with a textualised
version of the local subgraph. [ 28] extends LLM chain-of-thought (CoT) by iteratively retrieving
from the KG. [29] enhances the quality of the QA training data.
Although significant progress has been made, there are several limitations of existing methods. Many
of the real-world KGs are stored in native graph DBs. Graph DBs come with query engines that
optimise user queries, such as Cypher. However, none of the existing methods leverage such graph
DBs at all. Any method that explicitly requires step-by-step retrieval (due to iteratively prompting
the LLMs) prevents direct multi-hop querying on DBs. This implies that each step of the retrieval
process needs to be fully materialised and the fixed sequential order of traversal implementation is
frequently suboptimal. Benchmarking are often performed with in-memory graphs represented as
tensors with adhoc implementations of subgraph retrieval.
On the other hand, many methods that involve multi-hop retrieval as a single step remain abstract.
They usually stop at the point of specifying some path patterns. Converting the abstract path
definitions into queries remains a separate challenging task, reardless of specific query languages[ 30,
31,32,33]. Another stream of work explicitly requires the questions to be of some logic fragment
and converts them into SPARQL. There is also no general way of guaranteeing the generated queries
are syntactically and semantically correct with respect to specific KGs.
In this paper, we propose a simple, modular and reliable approach for question answering on private
KGs (GraphRAG). Specifically, the contributions of our papers are:
•We propose a new method of finetuning an LLM to generate optimal Cyphers that only
requires textual QAs as training data but not ground-truth queries.
•At inference time, we deploy a novel grounded constrained decoding strategy to generate
provably syntactically and semantically correct Cypher queries.
•Our method follows a modular and extensible retrieve-and-reason framework and can be
taken off-the-shelf to perform GraphRAG on native Graph DBs.
In addition, benchmarks on text-rich KGs suitable for GraphRAG show our method achieves sig-
nificantly beyond SOTA results across all 4 metrics on STaRK-prime and STaRK-mag, two large
text-attributed KGs with Q&As for GraphRAG. For example, our method achieves 63.71% Hit@1
and 68.99% Mean Reciprocal Rank (MRR), which are 22.81% and 17.79% better than the best previ-
ous baseline respectively on STaRK-prime. Our method is sample-efficient, for example achieving
beyond SOTA results on all metrics on STaRK-prime when trained with only 10% of the data, and
scales with more training data.
2 Related Work
RAG. Recent work extends RAG[ 2] to broader settings. RAFT[ 34] studies domain-specific retrieval-
augmented fine-tuning on by sampling relevant and irrelavant documents. GraphRAG[ 35] generalises
RAG to global text summarisation tasks in multiple documents. The term GraphRAG has also been
widely adopted to mean RAG on (knowledge) graphs, which is the problem setup we study in this
paper. [ 36] applies RAFT for GraphRAG on document retrieval and summarisation. There are many
methodological and implementation improvements[ 37] and industrial applications of GraphRAG[ 38].
Message Passing Graph Neural Networks and Graph Transformers. Message-passing Graph
Neural Networks (GNNs) iteratively aggregates and updates embeddings on nodes and edges using
the underlying graph as the computation graph, hence incorporating strong inductive bias. GNNs have
been used within the framework of GraphRAG to improve retrieval[ 39] or to enhance the graph pro-
cessing power of LLMs[ 40,41]. Alternative model architectures such as Graph Transformers, which
employ graph-specific tokenization or positional embedding techniques, have been developed[ 42,43].
2

UNWIND $source_names AS source_name
MATCH (s {name: source_name})-[r1]-(var)-[r2]-t WHERE tgt <> s
RETURN labels(s)[1] AS label1, s.name AS name1, type(r1) AS type1,
labels(var)[1] AS label2, type(r2) AS type2, labels(t)t AS label3,
size([t IN collect(DISTINCT tgt) WHERE t.nodeId in $target_ids | t]) AS
correctCnt, count(DISTINCT tgt) AS totalCnt
Figure 1: An example Cypher query. It takes as user input list variables source_names. It iterates
through them and find all two-hop neighbours of each source_name node, requiring the second-hop
node to be distinct to the source. The query returns aggregate information of the subgraph such as
labels and types of nodes and edges, and arithmetic over how many second-hop nodes have ids that
are in the user-defined node id list.
There are also work[ 44,45,46] that leverage LLMs to improve classical graph problems such as
node classification and link prediction on text-attributed graphs.
3 Preliminary
3.1 Graph database and graph query language
Graph DB treats both nodes and edges as primary citizens of the data model. A common graph data
model is Labeled Property Graph (LPG). LPGs support types and properties on nodes and edges in a
flexible way. For example, nodes of the same type are allowed to have different properties, and nodes
and edges can share the same types. LPGs can be used to model virtually all real-world KGs.
A Graph DB stores LPGs efficiently on disks. It also comes with a query engine that allows one to
query the graph. A widely used query language is Cypher[ 47][48] (and a variant openCypher[ 49]),
which are implementations of the recent GQL[ 50][51] standard. We will refer to the two interchange-
ably throughout this paper. The key ingredient of Cypher is graph pattern matching. A user can query
the graph by matching on patterns (e.g paths of a certain length) and predicates (e.g filtering on node
and edge properties). An example Cypher query is provided in Figure 1. The execution of a query is
carried out by the query engine which heavily optimises the order of executing low-level operators.
An alternative graph data model is Resource Description Framework (RDF)[ 52]. It was originally
created to model the Web. A graph is modeled as a collection of triples, commonly referred to
as subject-predicate-object, where each element can be identified with a URI (mimicing the Web).
Query languages on top of RDFs include SPARQL[ 14]. RDFs also support ontology languages such
as Web Ontology Language (OWL)[ 53] to model and derive facts through formal logic. Much of
the KBQA literature has taken inspiration from RDFs by defining a graph as a collection of triples
and reasoning over it in a formal style. Widely used benchmarks such as WebQSP[ 4] are exactly
questions about the Web that are answerable by SPARQL.
Many open-source and commercial graph DBs support Cypher over LPGs, such as Neo4j[ 54],
ArangoDB[ 55], TigerGraph[ 56]. There are also popular RDF stores such as Neptune[ 57], which also
supports Cypher over RDFs.
3.2 LLM
Modern LLMs are usually auto-regressive decoder-only Transformers as backbones that are trained
on the Web. An LLM fθhas a fixed vocabulary set V. Given a sequence of characters c0,···, cn,
a tokenizer converts it to a sequence of tokens t0,···, tkwhere ti∈Rdanddis the embedding
dimension. The tokenizer often compresses multiple characters into one token and splits a word into
multiple tokens. Given t0,···, tk, suppose fθhas generated tokens tk+1,···, tk+n, it computes the
logits for the k+n+ 1thtoken as lk+n+1=fθ(t0;tk+n)where lk+n+1∈R|V|. The probability
of generating any token xis obtained by applying softmax to the logits, p(tk+1=x|t0;tk+n+1) =
exp(lx
k+n+1)/P
y∈Vexp(ly
k+n+1). Greedy decoding picks the token with the highest probability
at each step. Beam search with width mkeeps msequence of tokens with the highest product
of probabilities so far. The generative process normally terminates when some end-of-sequence
3

< eos > token is decoded. Pretrained LLMs can be finetuned efficiently using techniques such as
LoRA[58] optimising the product of conditional probabilities of next-token prediction.
4 Approach
LetG= (V, E, L, l v, le, K, W, p v, pe)be a KG stored in a graph DB. V, E are nodes and edges. L
is a set of labels. lv:V→ P (L)the label assignment function for V.K, W the set of property
keys and values. pv:V→K×Wthe property key-value assignment function on the nodes. le, pe
are defined analogously for edges. Note that we support the most flexible definition of KGs, where
nodes and edges can share labels, and those of the same label can have different properties keys. We
assume that each node has at least one property that is text, which is common in the GraphRAG setup.
All nodes vare equipped with an additional property which we abbreviate as zv, which is the text
embedding produced by some text embedder LM on the text attribute.
Given a set of training QAs {Qi, Ai}where Ai⊂V, we want our model to produce good answers
Ajfor unseen questions Qjaccording to a variety of metrics. Our approach consists of several steps.
First, we create a set of training question-Cypher pairs {Qi, Ci}to finetune a LLM (Section 4.1). At
inference time, we deploy a simple method of constrained decoding that is grounded to Gto guarantee
syntactically and semantically optimal Cypher (Section 4.2). The optimal Cypher is executed to
retrieve a text-attributed subgraph for each question and a second LLM is finetuned to jointly reason
over text and subgraph topology to select the final answers (Section 4.3).
4.1 Synthesize ground-truth cypher
For a given Qi, we few-shot prompt an LLM L0to identify the relevant entities (strings) n1,···nk
mentioned in the question. In order to address the inherent linguistic ambiguity and noise present
in the questions and the graph, if a generated node name does not correspond to an existing node,
we perform an efficient vector similarity search directly using vector index at the database level to
retrieve the most similar nodes.
vi=cossim (LM(ni),{zv}v∈V) (1)
This enables more accurate identification of entities (that correspond to nodes in the graph) than per-
forming native vector similarity between all nodes in the graph and an embedded vector representing
the question.
We then construct several Cypher templates that pattern matches multi-hop graphs around the
identified entities v1,···, vkwith filters on node and relationship types. For each Cypher query Cj
we execute the query and compute the hits in answers Ai, as well as the number of total nodes. The
calculation (as an aggregation step) is performed as part of the query itself which means the nodes
and edges and their properties do not need to be materialised and retrieved and hence saves memory
and I/O workload. An example such query is shown in Figure 1.
We then rank the queries according to their hits and number of results and retains the top query Ci.
This gives us a set of {Qi, Ci}between questions and best Cypher query that contains the answers
{Ai}. The best Cypher can be decided by reranking and for example filtering by recall and precision.
We finetune an LLM L1with the given set of training and validation data. The workflow is illustrated
in Figure 2.
4.2 Grounded constrained decoding
OurL1has learned to generate graph pattern matching Cypher queries. However, there is still no
guarantee that the generated Cypher is executable at inference time. The major bottlenecks are 1)
syntactical correctness and 2) semantic correctness (i.e any type filters must correspond to existing
node labels or edge types of G). A common approach of generate-and-validate is both costly and slow
and requires a sophisticated correction scheme post-validation to guarantee eventual correctness.
We use a new method of next-token constrained decoding at the level of logits processor at inference
time. The constraints are applied at the token-level instead of word (or query keyword) level since
the tokenizers used by the majorioty of modern LLMs do not have a simple mapping between words
and query keywords to the tokens.
4

Figure 2: An example of creating ground-truth Cypher for a QA. In Step 1, few-shot LLM produces
candidate entities which we ground with Gin the DB with vector index. Step 2 shows part of the
subgraph around the entity and answer nodes. With the DB, we execute the all one-hop, two-hop
around each entity, and all length-two paths connecting the two entities in Step 3. We aggregate the
hits and number of nodes for each query and rank them.
Given a question at inference time, we first create a set of possible m-hop queries involving iden-
tified entities. This step is efficient since the schema of the graph is available. Let there be
Q={Q1,···,QM}valid queries, each has tokenization Qk= (n0,···, nQk)∈ VQkof variable
length. When L1has generated itokens [t0,···ti]and is generating a vector of logits (l0,···, lV),
our logits processor masks all invalid token with the value −∞ by comparing with the i+ 1thtoken
of all possible tokenized queries that match the initial itokens. The masking is defined as Equation 2
where Mi+1is the set of valid tokens at i+ 1thposition grounded in GviaQ.
˜ℓ(x)
i+1=(
ℓ(x)
i+1,ifx∈Mi+1
−∞,ifx /∈Mi+1where Mi+1={t∈ V|∃Qk∈ Q ,Qk
0:i+1= (t0,···, tk, t)}(2)
The tokenizer for L1then applies decoding after softmax. By construction, our decoded query is
executable. This constraint decoding is non-invasive for both greedy decoding and beam search with
sufficiently large beams. Unlikely existing constrained generation methods[ 59,60], our approach
does not require any formal grammar to be defined and is context-aware with respect to G. An
example of grounded constrained decoding is illustrated in Figure 3. Since our method is applied
entirely on the logits and before softmax, various ways of decoding such as greedy and beam search
can be used independently.
4.3 Finetuning LLM as local subgraph reasoner
Given any question for KG G, our trained LLM L1produces guaranteed executable and grounded
Cypher queries. If we perform beam search with width m, we are able to obtain mvalid queries
with highest total probabilties. For the simplest questions and sparse G, the subgraph produced by
the best query may be exactly the answer nodes. However, for more difficult multi-hop questions
on dense G, the subgraph still contains other relevant nodes (and edges) that are not the answers. In
order to obtain only the answer nodes by reasoning over the small subgraph, which often requires
reasoning on the textual properties beyond graph patterns, we train an LLM L2to perform the task.
5

21.6
17 .0
14. 7
12.6
. . .-∞
-∞
-∞
12.6
. . .0.0
0.0
0.0
1.0
. . .. . .. . .“}
 (
 d ise a se
 s ynd r ome
. . .MA T CH (x1: Dise a se  {n ame:  "X -l i nk ed ic h th y o sisL o git sM a sk edSo f tm a x
P a th w a y
E f f ec t
Dise a se
Dru g
. . .19.1
18.3
22.3
17 .9
. . .19.1
-∞
22.3
-∞
. . .0.0 3
0.0
0.9 7
0.0
. . .. . .. . .MA T CH (x1:L o git sM a sk edSo f tm a x
I
a s sis t an t
MA T CH
. . .L o git sM a sk edSo f tm a xL o git sM a sk edSo f tm a x<bo s>12. 2
12.1
22.3
. . .-∞
-∞
22.3
. . .0.0
0.0
1.0
. . .. . .
   T op  p a th s
M A T C H  ( x 1 : D i s e a s e  { n a m e :  " X - l i n k e d  i c h t h y o s i s  s y n d r o m e " } ) - [ r 1 : A S S O C I A T E D _ W I T H ] - ( x 2 : G e n e O r P r o t e i n ) - [ r 2 : I N T E R A C T S _ W I T H ] -
( x 3 : P a t h w a y  { n a m e :  " G l y c o s p h i n g o l i p i d  m e t a b o l i s m " } )  R E T U R N  x 
. . .P o s s i bl e  p a th s
M A T C H  ( x 1 : P a t h w a y  { n a m e :  " G l y c o s p h i n g o l i p i d  m e t a b o l i s m " } ) - [ r 1 : P A R E N T _ C H I L D ] - ( x 2 : P a t h w a y )  R E T U R N  x 
M A T C H  ( x 1 : D i s e a s e  { n a m e :  " X - l i n k e d  i c h t h y o s i s  s y n d r o m e " } ) - [ r 1 : A S S O C I A T E D _ W I T H ] - ( x 2 : G e n e O r P r o t e i n )  R E T U R N  x 
M A T C H  ( x 1 : P a t h w a y  { n a m e :  " G l y c o s p h i n g o l i p i d  m e t a b o l i s m " } ) - [ r 1 : I N T E R A C T S _ W I T H ] - ( x 2 : G e n e O r P r o t e i n ) - [ r 2 : P P I ] -
( x 3 : G e n e O r P r o t e i n )  R E T U R N  x 
3 0  m o r e . . .Ma t c hed en titie s 
“ Gly cosphing olipid me t a bolism 
“X -link ed ichth y osis s yndrome ”Que s tion:
“ Whic h g ene  or pr o t ei n is  i n v ol v ed i n Gl y c o sphi ng ol i pid me t abol ism 
and al so  a s soc ia t ed with the  de v el opmen t o f X -l i nk ed ic h th y o sis ?”Figure 3: An example of grounded constrained decoding. For the given question, we tokenize all
possible queries around it’s identified entities. At each step during generation, our logits processor
masks out invalid tokens. For example, after "ichthyosis" , the LLM would have generated the symbols
")which has the highest logit. Our processor masks it out since this predicate name: "X-linked
ichthyosis" is invalid.
We construct the subgraph by executing mqueries until it contains some threshold on the size of
graph or tokenization required to encode the graph. The outputs of L2can be viewed as selecting or
reranking the nodes in the input textualised graph. An example prompt we used provided in Figure 4.
5 Experiments
Setup We use Neo4j as the graph database. The default database configuration is used. For the
main result, we using OpenAI text-embedding-ada-002[ 61] as the text embedder LM in Equation 1,
OpenAI gpt-4o-mini as LLM L0for few-shot entity resolution., gemma2-9b-text2cypher[ 62,63]
as our LLM L1, Llama-3.1-8B-Instruct[ 64] as our LLM L2. All experiments are run on a single
40G A100 GPU. Additional detailed experiment setup is provided in Appendix A. All of our code is
available on Github1.
Datasets We benchmark our method on the stark-prime and stark-mag datasets[ 65]. stark-prime is
a set of Q&As over a large biomedical knowledge graph PrimeKG[ 66]. The questions mimic roles
such as doctors, medical scientists and patients. It contains 10 node types and 18 edge types with rich
1https://github.com/AlfredClemedtson/graphraft
6

<|start_header_id|>user<|end_header_id|>
Given the information below, return the correct nodes for the following question:
What drugs target the CYP3A4 enzyme and are used to treat strongyloidiasis?
Retrieved information:
pattern: [ '(x1:GeneOrProtein {name: "CYP3A4"})-[r1:ENZYME]-
(x2:Drug {name: "Ivermectin"})-[r2:INDICATION]-(x3:Disease {name: "strongyloidiasis"}) ',
'(x1:Disease {name: "strongyloidiasis"})-[r1:INDICATION]-(x2:Drug {name: "Ivermectin"}) ',
'(x1:GeneOrProtein {name: "CYP3A4"})-[r1:ENZYME]-(x2:Drug {name: "Ivermectin"}) ']
name: Ivermectin
details: { 'description ':'Ivermectin is a broad-spectrum anti-parasite medication.
It was first marketed under the name Stromectol ®and used against worms (except tapeworms),
but, in 2012, it was approved for the topical treatment of head lice
infestations in patients 6 months of age and older,
and marketed under the name Sklice ™as well. ...
pattern: [ '(x1:Disease {name: "strongyloidiasis"})-[r1:INDICATION]-
(x2:Drug {name: "Thiabendazole"}) ']
name: Thiabendazole
details: { 'description ':'2-Substituted benzimidazole first introduced in 1962.
It is active against a variety of nematodes and is the drug of choice for
strongyloidiasis. It has CNS side effects and hepatototoxic potential. ...
...
<|start_header_id|>model<|end_header_id|>
Figure 4: An example prompt that describes a local subgraph retrieved by Cypher queries around
identified entities. This prompt contains both textual information and patterns used to retrieved them,
which encodes topology information.
textual properties on the nodes. With 129k nodes and 8 million edges and at the same time a high
density (average node degree 125), it serves as a challenging and highly suitable dataset to benchmark
retrieval and reasoning on large real-world KGs. stark-mag is a set of Q&As over ogbn-mag[ 67] that
models a large academic citation network with relations between papers, authors, subject areas and
institutions.
There is a large collection of Q&A datasets on graphs, we benchmark on datasets most suit-
able to the GraphRAG domain[ 68] and elaborate on why we don’t benchmark on the others.
WebQ[ 69], WebQSP[ 4], CWQ[ 6] ad GrailQA[ 70] are popular KBQA benchmarks containing
SPARQL-answerable few-hop questions over Freebase[ 5], a database of general knowledge with
URIs on nodes. Our problem setting has no requirement on the form of the questions and targets
private KGs instead of such largely wikipedia-based KG which LLMs are explicitly trained on.
Freebase has been deprecated since 2015. HotpotQA[ 71] and BeerQA[ 72] are few-hop questions
over Wikidata[ 73]. STaRK-amazon[ 65] models properties of products (such as color) as nodes (with
has-color relation) and the product co-purchasing graph itself is homogeneous.
5.1 Main results
We show our result in Table 1. The baselines range from pure vector-based retrievers to GraphRAG
solutions to agentic methods. We use the four metrics originally proposed in [ 65]. Hit@1 measure
the ability of exact answering a right answer while the other three metrics provides more holistic
view on the answer quality.
As is shown in Table 1, GraphRAFT gives best results on all metrics on both STaRK-prime and
STaRK-mag. Even without using L2to reason over the local subgraph, our L1when used to retrieve
nodes using generated Cypher queries (up to 20 nodes, for measuring recall), already gives better
metrics than all SOTA methods on STaRK-prime and is close to the best baselines for STaRK-mag.
7

Table 1: Main table of our results against previous baselines. Bold and underline represent the best
method, underline represents the second best.
STARK-PRIME STARK-MAG
Hit@1 Hit@5 R@20 MRR Hit@1 Hit@5 R@20 MRR
BM25[74] 12.75 27.92 31.25 19.84 25.85 45.25 45.69 34.91
voyage-l2-instruct[75] 10.85 30.23 37.83 19.99 30.06 50.58 50.49 39.66
GritLM-7b[76] 15.57 33.42 39.09 24.11 37.90 56.74 46.40 47.25
multi-ada-002[61] 15.10 33.56 38.05 23.49 25.92 50.43 50.80 36.94
ColBERTv2[77] 11.75 23.85 25.04 17.39 31.18 46.42 43.94 38.39
Claude3 Reranker (10%) 17.79 36.90 35.57 26.27 36.54 53.17 48.36 44.15
GPT4 Reranker (10 %) 18.28 37.28 34.05 26.55 40.90 58.18 48.60 49.00
HybGRAG[78] 28.56 41.38 43.58 34.49 65.40 75.31 65.70 69.80
AvaTaR[79] 18.44 36.73 39.31 26.73 44.36 59.66 50.63 51.15
MoR[80] 36.41 60.01 63.48 46.92 58.19 78.34 75.01 67.14
KAR[81] 30.35 49.30 50.81 39.22 50.47 65.37 60.28 57.51
MFAR[82] 40.9 62.8 68.3 51.2 49.00 69.60 71.79 58.20
GraphRAFT w/o L2 52.12 71.55 75.52 60.72 62.63 71.86 72.97 66.28
GraphRAFT 63.71 75.39 76.39 68.99 71.05 85.34 76.96 76.92
Table 2: Metrics on STaRK-prime using 10% of validation data. Percentage of train data used is
specified next to the method. Numbers in brackets representing using the method without applying
constrained decoding. No schema is provided in prompt and response executed as is in all queries.
Method (% Training data used) Hit@1 Hit@5 Recall@20 MRR
Finetuned, 100% 44.20(43.75) 69.20(63.39) 75.87(69.83) 55.28(52.55)
Finetuned, 10% 41.07(35.71) 66.07(54.91) 76.11(60.53) 51.97(44.00)
LLM, 0% 14.73(0.0) 23.21(0.0) 27.53(0.0) 18.65(0.0)
5.2 Impact of constrained decoding and scaling with training data
We measure the benefit of applying grounded constrained decoding to using our model without it.
We also examine how well the method scales with the availability of training data with and without
constrained decoding. We measure the metrics directly on the output Cypher queries, executing the
ones with highest probabilities first, if executable, and falling back to lower-probability queries, until
there are up to 20 nodes. We do not apply LLM L2to reason over the local subgraph to improve
Hits@1 to accurately evaluate LLM L1and constrained decoding.
As can be seen in Table 2, when constrained decoding is not used and the model is trained on 100% of
available data, it gives slightly lower metrics. When we only use 10% of the training data, our method
only shows a slight decrease in all metrics. However, when used without constrained decoding, the
drop becomes larger (e.g 16% for Recall@20). This suggests that our method is both extremely
sample efficient and scales well with more training Q&As. The advantage of constrained decoding is
the most significant when training data is scarce, which is common in any real-world setting. The
final row uses the gemma2-9b-text2cypher model without any finetuning on STaRK-prime. It is not
able to out-of-the-box answer any question and applying constrained decoding without finetuning the
LLM at all already gives us results close to several baselines.
5.3 The use of query engine
The use of query engines to optimise query plans on DBs has always been one of the main advantages
of DB systems. Table 3 shows an example executed query plan that is optimal according to the query
planner. It first fetches nodes of the type Drug from the node label indexes and then traverses along
the ENZYME relationship type. It then filter the joined records with predicates on x1. Afterwards it
performs the similar traversal from x2 to x3 along INDICATION and filter on x3. Intuitively, the
optimal plan finds the all x2:Drug nodes and filter down twice by joining the two ends.
8

Table 3: The optimal execution plan for an example retrieval query: MATCH (x1:GeneOrProtein
name: "CYP3A4")-[r: ENZYME]-(x2: Drug)-[r2: INDICATION]-(x3: Disease name: "
strongyloidiasis") RETURN x2.name .Operator are executed from the bottom up. Details
represent the exact execution parameters. Estimated Rows represent expected rows produced.
Rows represent actual rows produced. DB Hits measure the amount of work by the storage engine.
Total database access is 103989, total allocated memory is 328 bytes.
Operator Id DetailsEstimated
RowsRows DB Hits
+ProduceResults 0 n‘x2.name‘ 63 4 0
+Projection 1 x2.name AS ‘x2.name‘ 63 4 4
|+Filter 2 x3.name = $autostring_1 AND x3:Disease 63 4 34872
|+Expand(All) 3 (x2)-[r2:INDICATION]-(x3) 1255 17432 17432
|+Filter 4 x1.name = $autostring_0 AND x1:GeneOrProtein 532 1932 25132
|+Expand(All) 5 (x2)-[r:ENZYME]-(x1) 10634 10634 18591
+NodeByLabelScan 6 x2:Drug 7957 7957 7958
Table 4: A valid but suboptimal query plan for an example retrieval query: MATCH (x1:
GeneOrProtein name: "CYP3A4")-[r: ENZYME]-(x2: Drug)-[r2: INDICATION]-(x3: Disease
name: "strongyloidiasis") RETURN x2.name . Compared with the optimal plan in Table 3, this
plan has both more operators and more costly ones.
Operator Id Details Estimated Rows
+ProduceResults 0 n‘x2.name‘ 63
+Projection 1 x2.name AS ‘x2.name‘ 63
|+NodeHashJoin 2 x2 109441
||+Filter 3 x1.name = $autostring_0 AND x1:GeneOrProtein 51795
||+Expand(All) 4 (x2)-[r2:ENZYME]-(x1) 19893
||+NodeByLabelScan 5 x2:Drug 7957
|+Filter 6 x2:Drug 7957
|+Expand(All) 7 (x3)-[r:INDICATION]-(x2) 52521
|+Filter 8 x3.name = $autostring_1 51240
+NodeByLabelScan 9 x3:Disease 17080
An alternative valid but suboptimal query plan is provided in Table 4. It starts with scanning all
x3:Disease nodes and traverses towards x2 and then x1. It happens to be suboptimal due to the
exploding neighbourhood of Disease nodes along the INDICATION relationship type. The query
engine with it’s cardinality estimator therefore rules out executing this sequence of operators.
This analysis serves as an example of confirming the advantage of using a query engine. As we have
pointed out, any of the existing GraphRAG work that iteratively traverses the graph step-by-step
is not able to leverage the query engine since the order of execution is fixed. Therefore, extremely
inefficient retrieval (such as the ordering shown in Table 4) is possible and unavoidable.
5.4 Impact of named entity recognition and ground-truth Cyphers.
Given {Qi, Ai}as input data, we prepare a set of {Qi, Ci}as a set of training data to finetune the
LLM to generate optimal Cypher queries. We first few-shot prompt an LLM to identify entities and
then reconcile the identified names against the DB. Next, by using the graph schema, we obtain the
set of all possible Cypher queries with type predicates k-hop around or connecting the entity nodes.
The quality of the identified entities is therefore implicitly measured by the best Cypher that it allows.
We first verify that using an LLM for entity resolution is needed better than simpler methods such as
k-nearest-neighbour (kNN) using text embeddings. We measure that by looking at the quality of the
best Cypher created from entities identified by an LLM, 2NN and 5NN, as shown in Figure 5.
9

(a) The number of questions that
map to good Cyphers when using
different entity resolution methods.
(b) The method that gives the high-
est recall with the best Cypher for
each question.
(c) The method where the best
Cypher returns a small subgraph
when Recall = 1.
Figure 5: The impact of entity resolution on the quality of Cypher queries.
6 Conclusion
In this work, we introduce GraphRAFT, a simple and modular GraphRAG that leverages graph DBs
by retrieving from it using provably correct and optimal Cypher queries. Our experiments consistently
show that GraphRAG achieves results beyond the state of the art. Our framework can be applied
off-the-shelf to any KG in any domain stored in graph DBs. The finetuning process is sample-efficient
and scales with more training data.
GraphRAFT is illustrated on graph DBs supporting Cypher. Exactly the same approach can be used
for any other graph query language. Our finetuning process requires existing Q&A set. Future work
that addresses these limitations will improve the general applicability of the method.
References
[1]Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qiang-
long Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large
language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on
Information Systems , 43(2):1–55, 2025.
[2]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in neural information processing
systems , 33:9459–9474, 2020.
[3]Yunshi Lan, Gaole He, Jinhao Jiang, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen. A survey
on complex knowledge base question answering: Methods, challenges and solutions. arXiv
preprint arXiv:2105.11644 , 2021.
[4]Wen tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The
value of semantic parse labeling for knowledge base question answering. In Annual Meeting of
the Association for Computational Linguistics , 2016. URL https://api.semanticscholar.
org/CorpusID:13905064 .
[5]Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a
collaboratively created graph database for structuring human knowledge. In Proceedings
of the 2008 ACM SIGMOD International Conference on Management of Data , SIGMOD
’08, page 1247–1250, New York, NY , USA, 2008. Association for Computing Machinery.
ISBN 9781605581026. doi: 10.1145/1376616.1376746. URL https://doi.org/10.1145/
1376616.1376746 .
[6]Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex
questions. In Marilyn Walker, Heng Ji, and Amanda Stent, editors, Proceedings of the 2018
Conference of the North American Chapter of the Association for Computational Linguis-
tics: Human Language Technologies, Volume 1 (Long Papers) , pages 641–651, New Orleans,
10

Louisiana, June 2018. Association for Computational Linguistics. doi: 10.18653/v1/N18-1059.
URL https://aclanthology.org/N18-1059 .
[7]Wen-tau Yih, Ming-Wei Chang, Xiaodong He, and Jianfeng Gao. Semantic parsing via
staged query graph generation: Question answering with knowledge base. In Chengqing Zong
and Michael Strube, editors, Proceedings of the 53rd Annual Meeting of the Association for
Computational Linguistics and the 7th International Joint Conference on Natural Language
Processing (Volume 1: Long Papers) , pages 1321–1331, Beijing, China, July 2015. Association
for Computational Linguistics. doi: 10.3115/v1/P15-1128. URL https://aclanthology.
org/P15-1128 .
[8]Kangqi Luo, Fengli Lin, Xusheng Luo, and Kenny Zhu. Knowledge base question answering
via encoding of complex query graphs. In Ellen Riloff, David Chiang, Julia Hockenmaier,
and Jun’ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing , pages 2185–2194, Brussels, Belgium, October-November 2018.
Association for Computational Linguistics. doi: 10.18653/v1/D18-1242. URL https://
aclanthology.org/D18-1242/ .
[9]Hamid Zafar, Giulio Napolitano, and Jens Lehmann. Formal query generation for question
answering over knowledge bases. In The Semantic Web: 15th International Conference, ESWC
2018, Heraklion, Crete, Greece, June 3–7, 2018, Proceedings , page 714–728, Berlin, Heidelberg,
2018. Springer-Verlag. ISBN 978-3-319-93416-7. doi: 10.1007/978-3-319-93417-4_46. URL
https://doi.org/10.1007/978-3-319-93417-4_46 .
[10] Hannah Bast and Elmar Haussmann. More accurate question answering on freebase. In
Proceedings of the 24th ACM International on Conference on Information and Knowledge
Management , CIKM ’15, page 1431–1440, New York, NY , USA, 2015. Association for
Computing Machinery. ISBN 9781450337946. doi: 10.1145/2806416.2806472. URL
https://doi.org/10.1145/2806416.2806472 .
[11] Zi-Yuan Chen, Chih-Hung Chang, Yi-Pei Chen, Jijnasa Nayak, and Lun-Wei Ku. UHop: An
unrestricted-hop relation extraction framework for knowledge-based question answering. In
Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference
of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies, Volume 1 (Long and Short Papers) , pages 345–356, Minneapolis,
Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1031.
URL https://aclanthology.org/N19-1031/ .
[12] Nikita Bhutani, Xinyi Zheng, and H V Jagadish. Learning to answer complex questions over
knowledge bases with query composition. In Proceedings of the 28th ACM International
Conference on Information and Knowledge Management , CIKM ’19, page 739–748, New
York, NY , USA, 2019. Association for Computing Machinery. ISBN 9781450369763. doi:
10.1145/3357384.3358033. URL https://doi.org/10.1145/3357384.3358033 .
[13] Yunshi Lan and Jing Jiang. Query graph generation for answering multi-hop complex questions
from knowledge bases. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault, editors,
Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics , pages
969–974, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/
2020.acl-main.91. URL https://aclanthology.org/2020.acl-main.91/ .
[14] SPARQL. https://www.w3.org/TR/sparql11-query/ , 2013.
[15] Haitian Sun, Tania Bedrax-Weiss, and William W. Cohen. Pullnet: Open domain question
answering with iterative retrieval on knowledge bases and text, 2019. URL https://arxiv.
org/abs/1904.09537 .
[16] Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhutdinov, and
William Cohen. Open domain question answering using early fusion of knowledge bases and
text. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Proceedings
of the 2018 Conference on Empirical Methods in Natural Language Processing , pages 4231–
4242, Brussels, Belgium, October-November 2018. Association for Computational Linguistics.
doi: 10.18653/v1/D18-1455. URL https://aclanthology.org/D18-1455/ .
11

[17] Wenhan Xiong, Mo Yu, Shiyu Chang, Xiaoxiao Guo, and William Yang Wang. Improv-
ing question answering over incomplete KBs with knowledge-aware reader. In Anna Ko-
rhonen, David Traum, and Lluís Màrquez, editors, Proceedings of the 57th Annual Meet-
ing of the Association for Computational Linguistics , pages 4258–4264, Florence, Italy,
July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1417. URL
https://aclanthology.org/P19-1417/ .
[18] Shizhu He, Cao Liu, Kang Liu, and Jun Zhao. Generating natural answers by incorporating
copying and retrieving mechanisms in sequence-to-sequence learning. In Regina Barzilay
and Min-Yen Kan, editors, Proceedings of the 55th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers) , pages 199–208, Vancouver, Canada,
July 2017. Association for Computational Linguistics. doi: 10.18653/v1/P17-1019. URL
https://aclanthology.org/P17-1019/ .
[19] Jinheon Baek, Alham Fikri Aji, and Amir Saffari. Knowledge-augmented language model
prompting for zero-shot knowledge graph question answering, 2023. URL https://arxiv.
org/abs/2306.04136 .
[20] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. Grag: Graph
retrieval-augmented generation, 2024. URL https://arxiv.org/abs/2405.16506 .
[21] Linhao Luo, Yuan-Fang Li, Reza Haf, and Shirui Pan. Reasoning on graphs: Faithful and
interpretable large language model reasoning. In The Twelfth International Conference on
Learning Representations , 2024. URL https://openreview.net/forum?id=ZGNWW7xZ6Q .
[22] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel
Ni, Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning
of large language model on knowledge graph. In The Twelfth International Conference on
Learning Representations , 2024. URL https://openreview.net/forum?id=nnVO1PvbTv .
[23] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin Mao,
and Jian Guo. Think-on-graph 2.0: Deep and faithful large language model reasoning with
knowledge-guided retrieval augmented generation. arXiv preprint arXiv:2407.10805 , 2024.
[24] Yuan Sui, Yufei He, Nian Liu, Xiaoxin He, Kun Wang, and Bryan Hooi. Fidelis: Faithful
reasoning in large language model for knowledge graph question answering, 2024. URL
https://arxiv.org/abs/2405.13873 .
[25] Song Wang, Junhong Lin, Xiaojie Guo, Julian Shun, Jundong Li, and Yada Zhu. Reasoning
of large language models over knowledge graphs with super-relations. In The Thirteenth
International Conference on Learning Representations , 2025. URL https://openreview.
net/forum?id=rTCJ29pkuA .
[26] Yu Xia, Junda Wu, Sungchul Kim, Tong Yu, Ryan A. Rossi, Haoliang Wang, and Julian
McAuley. Knowledge-aware query expansion with large language models for textual and
relational retrieval, 2024. URL https://arxiv.org/abs/2410.13765 .
[27] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann LeCun, Xavier
Bresson, and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph under-
standing and question answering. In The Thirty-eighth Annual Conference on Neural Informa-
tion Processing Systems , 2024. URL https://openreview.net/forum?id=MPJ3oXtTZl .
[28] Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong,
and Zhang Xiong. Knowledge-driven cot: Exploring faithful reasoning in llms for knowledge-
intensive question answering, 2023. URL https://arxiv.org/abs/2308.13259 .
[29] Hanzhu Chen, Xu Shen, Jie Wang, Zehao Wang, Qitan Lv, Junjie He, Rong Wu, Feng Wu, and
Jieping Ye. Knowledge graph finetuning enhances knowledge manipulation in large language
models. In The Thirteenth International Conference on Learning Representations , 2025. URL
https://openreview.net/forum?id=oMFOKjwaRS .
12

[30] Fangyu Lei, Jixuan Chen, Yuxiao Ye, Ruisheng Cao, Dongchan Shin, Hongjin Su, Zhaoqing
Suo, Hongcheng Gao, Wenjing Hu, Pengcheng Yin, Victor Zhong, Caiming Xiong, Ruoxi Sun,
Qian Liu, Sida Wang, and Tao Yu. Spider 2.0: Evaluating language models on real-world
enterprise text-to-sql workflows, 2025. URL https://arxiv.org/abs/2411.07763 .
[31] Caio Viktor S. Avila, Vânia M.P. Vidal, Wellington Franco, and Marco A. Casanova. Experi-
ments with text-to-sparql based on chatgpt. In 2024 IEEE 18th International Conference on
Semantic Computing (ICSC) , pages 277–284, 2024. doi: 10.1109/ICSC59802.2024.00050.
[32] Felix Brei, Johannes Frey, and Lars-Peter Meyer. Leveraging small language models for
text2sparql tasks to improve the resilience of ai assistance, 2024. URL https://arxiv.org/
abs/2405.17076 .
[33] Makbule Gulcin Ozsoy, Leila Messallem, Jon Besga, and Gianandrea Minneci. Text2cypher:
Bridging natural language and graph databases, 2024. URL https://arxiv.org/abs/2412.
10064 .
[34] Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and
Joseph E. Gonzalez. Raft: Adapting language model to domain specific rag. 2024.
[35] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven
Truitt, and Jonathan Larson. From local to global: A graph rag approach to query-focused
summarization. arXiv preprint arXiv:2404.16130 , 2024.
[36] Sonya Jin, Sunny Yu, and Natalia Kokoromyti. Graft: Graph retrieval augmented fine tuning for
multi-hop query summarization, 2025. URL https://web.stanford.edu/class/cs224n/
final-reports/256724569.pdf .
[37] Mufei Li, Siqi Miao, and Pan Li. Simple is effective: The roles of graphs and large language mod-
els in knowledge-graph-based retrieval-augmented generation. arXiv preprint arXiv:2410.20724 ,
2024.
[38] Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Deshpande, Xiaofeng
Wang, and Zheng Li. Retrieval-augmented generation with knowledge graphs for customer
service question answering. In Proceedings of the 47th International ACM SIGIR Conference
on Research and Development in Information Retrieval , SIGIR 2024, page 2905–2909. ACM,
July 2024. doi: 10.1145/3626772.3661370. URL http://dx.doi.org/10.1145/3626772.
3661370 .
[39] Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language
model reasoning. arXiv preprint arXiv:2405.20139 , 2024.
[40] Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D
Manning, and Jure Leskovec. Greaselm: Graph reasoning enhanced language models for
question answering. arXiv preprint arXiv:2201.08860 , 2022.
[41] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn:
Reasoning with language models and knowledge graphs for question answering. arXiv preprint
arXiv:2104.06378 , 2021.
[42] Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen,
and Tie-Yan Liu. Do transformers really perform bad for graph representation?, 2021. URL
https://arxiv.org/abs/2106.05234 .
[43] Jinwoo Kim, Tien Dat Nguyen, Seonwoo Min, Sungjun Cho, Moontae Lee, Honglak Lee,
and Seunghoon Hong. Pure transformers are powerful graph learners, 2022. URL https:
//arxiv.org/abs/2207.02505 .
[44] Jianan Zhao, Meng Qu, Chaozhuo Li, Hao Yan, Qian Liu, Rui Li, Xing Xie, and Jian Tang.
Learning on large-scale text-attributed graphs via variational inference. In The Eleventh Inter-
national Conference on Learning Representations , 2023. URL https://openreview.net/
forum?id=q0nmYciuuZN .
13

[45] Xiaoxin He, Xavier Bresson, Thomas Laurent, Adam Perold, Yann LeCun, and Bryan Hooi.
Harnessing explanations: LLM-to-LM interpreter for enhanced text-attributed graph represen-
tation learning. In The Twelfth International Conference on Learning Representations , 2024.
URL https://openreview.net/forum?id=RXFVcynVe1 .
[46] Xinke Jiang, Rihong Qiu, Yongxin Xu, WentaoZhang, Yichen Zhu, Ruizhe zhang, Yuchen Fang,
Xu Chu, Junfeng Zhao, and Yasha Wang. RAGraph: A general retrieval-augmented graph
learning framework. In The Thirty-eighth Annual Conference on Neural Information Processing
Systems , 2024. URL https://openreview.net/forum?id=Dzk2cRUFMt .
[47] Nadime Francis, Alastair Green, Paolo Guagliardo, Leonid Libkin, Tobias Lindaaker, Victor
Marsault, Stefan Plantikow, Mats Rydberg, Petra Selmer, and Andrés Taylor. Cypher: An
evolving query language for property graphs. In Gautam Das, Christopher M. Jermaine, and
Philip A. Bernstein, editors, Proceedings of the 2018 International Conference on Management
of Data, SIGMOD Conference 2018, Houston, TX, USA, June 10-15, 2018 , pages 1433–1445.
ACM, 2018. doi: 10.1145/3183713.3190657. URL https://doi.org/10.1145/3183713.
3190657 .
[48] Nadime Francis, Alastair Green, Paolo Guagliardo, Leonid Libkin, Tobias Lindaaker, Victor
Marsault, Stefan Plantikow, Mats Rydberg, Martin Schuster, Petra Selmer, and Andrés Taylor.
Formal semantics of the language cypher. CoRR , abs/1802.09984, 2018. URL http://arxiv.
org/abs/1802.09984 .
[49] Alastair Green, Martin Junghanns, Max Kießling, Tobias Lindaaker, Stefan Plantikow, and
Petra Selmer. opencypher: New directions in property graph querying. In Michael H. Böhlen,
Reinhard Pichler, Norman May, Erhard Rahm, Shan-Hung Wu, and Katja Hose, editors, Pro-
ceedings of the 21st International Conference on Extending Database Technology, EDBT
2018, Vienna, Austria, March 26-29, 2018 , pages 520–523. OpenProceedings.org, 2018. doi:
10.5441/002/EDBT.2018.62. URL https://doi.org/10.5441/002/edbt.2018.62 .
[50] GQL. https://www.iso.org/standard/76120.html , 2024.
[51] Alin Deutsch, Nadime Francis, Alastair Green, Keith Hare, Bei Li, Leonid Libkin, Tobias
Lindaaker, Victor Marsault, Wim Martens, Jan Michels, Filip Murlak, Stefan Plantikow, Petra
Selmer, Hannes V oigt, Oskar van Rest, Domagoj Vrgoc, Mingxi Wu, and Fred Zemke. Graph
pattern matching in GQL and SQL/PGQ. CoRR , abs/2112.06217, 2021. URL https://arxiv.
org/abs/2112.06217 .
[52] RDF. https://www.w3.org/RDF/ , 2014.
[53] OWL. https://www.w3.org/TR/owl2-overview/ , 2012.
[54] Neo4j. https://neo4j.com/ .
[55] ArangoDB. https://arangodb.com/ .
[56] TigerGraph. https://www.tigergraph.com/ .
[57] AWS Neptune. https://aws.amazon.com/neptune/ .
[58] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang,
Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR , 1
(2):3, 2022.
[59] Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Guiding llms the right way: Fast,
non-invasive constrained generation, 2024. URL https://arxiv.org/abs/2403.06988 .
[60] Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. Grammar-constrained
decoding for structured nlp tasks without finetuning, 2024. URL https://arxiv.org/abs/
2305.13971 .
[61] OpenAI embeddings. https://platform.openai.com/docs/guides/embeddings , 2025.
[62] Makbule Gulcin Ozsoy, Leila Messallem, Jon Besga, and Gianandrea Minneci. Text2cypher:
Bridging natural language and graph databases. arXiv preprint arXiv:2412.10064 , 2024.
14

[63] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin,
Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé,
et al. Gemma 2: Improving open language models at a practical size. arXiv preprint
arXiv:2408.00118 , 2024.
[64] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian,
Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama
3 herd of models. arXiv preprint arXiv:2407.21783 , 2024.
[65] Shirley Wu, Shiyu Zhao, Michihiro Yasunaga, Kexin Huang, Kaidi Cao, Qian Huang, Vassilis
Ioannidis, Karthik Subbian, James Y Zou, and Jure Leskovec. Stark: Benchmarking llm retrieval
on textual and relational knowledge bases. Advances in Neural Information Processing Systems ,
37:127129–127153, 2025.
[66] Payal Chandak, Kexin Huang, and Marinka Zitnik. Building a knowledge graph to enable
precision medicine. Scientific Data , 10(1):67, 2023. URL https://doi.org/10.1038/
s41597-023-01960-3 .
[67] Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele
Catasta, and Jure Leskovec. Open graph benchmark: Datasets for machine learning on graphs.
Advances in neural information processing systems , 33:22118–22133, 2020.
[68] Maya Bechler-Speicher, Ben Finkelshtein, Fabrizio Frasca, Luis Müller, Jan Tönshoff, Antoine
Siraudin, Viktor Zaverkin, Michael M Bronstein, Mathias Niepert, Bryan Perozzi, et al. Position:
Graph learning will lose relevance due to poor benchmarks. arXiv preprint arXiv:2502.14546 ,
2025.
[69] Jonathan Berant, Andrew K. Chou, Roy Frostig, and Percy Liang. Semantic parsing on
freebase from question-answer pairs. In Conference on Empirical Methods in Natural Language
Processing , 2013. URL https://api.semanticscholar.org/CorpusID:6401679 .
[70] Yu Gu, Sue Kase, Michelle Vanni, Brian Sadler, Percy Liang, Xifeng Yan, and Yu Su. Beyond
iid: three levels of generalization for question answering on knowledge bases. In Proceedings
of the Web Conference 2021 , pages 3477–3488, 2021.
[71] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhut-
dinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop
question answering. In Conference on Empirical Methods in Natural Language Processing
(EMNLP) , 2018.
[72] Peng Qi, Haejun Lee, Oghenetegiri "TG" Sido, and Christopher D. Manning. Answering
open-domain questions of varying reasoning steps from text. In Empirical Methods for Natural
Language Processing (EMNLP) , 2021.
[73] Denny Vrande ˇci´c and Markus Krötzsch. Wikidata: a free collaborative knowledgebase. Com-
mun. ACM , 57(10):78–85, September 2014. ISSN 0001-0782. doi: 10.1145/2629489. URL
https://doi.org/10.1145/2629489 .
[74] Stephen E. Robertson and Hugo Zaragoza. The probabilistic relevance framework: Bm25 and
beyond. Found. Trends Inf. Retr. , 3:333–389, 2009. URL https://api.semanticscholar.
org/CorpusID:207178704 .
[75] V oyage AI. V oyage ai text embedding models. https://docs.voyageai.com/reference/
embeddings-api , 2009.
[76] Niklas Muennighoff, SU Hongjin, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh,
and Douwe Kiela. Generative representational instruction tuning. In ICLR 2024 Workshop:
How Far Are We From AGI , 2024.
[77] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia.
ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In Marine Carpuat,
Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, editors, Proceedings of the 2022
Conference of the North American Chapter of the Association for Computational Linguistics:
15

Human Language Technologies , pages 3715–3734, Seattle, United States, July 2022. Association
for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.272. URL https://
aclanthology.org/2022.naacl-main.272/ .
[78] Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N. Ioannidis,
Huzefa Rangwala, and Christos Faloutsos. Hybgrag: Hybrid retrieval-augmented generation on
textual and relational knowledge bases, 2024. URL https://arxiv.org/abs/2412.16311 .
[79] Shirley Wu, Shiyu Zhao, Qian Huang, Kexin Huang, Michihiro Yasunaga, Kaidi Cao, Vassilis
Ioannidis, Karthik Subbian, Jure Leskovec, and James Y Zou. Avatar: Optimizing llm agents
for tool usage via contrastive reasoning. Advances in Neural Information Processing Systems ,
37:25981–26010, 2025.
[80] Yongjia Lei, Haoyu Han, Ryan A. Rossi, Franck Dernoncourt, Nedim Lipka, Mahantesh M
Halappanavar, Jiliang Tang, and Yu Wang. Mixture of structural-and-textual retrieval over
text-rich graph knowledge bases, 2025. URL https://arxiv.org/abs/2502.20317 .
[81] Yu Xia, Junda Wu, Sungchul Kim, Tong Yu, Ryan A. Rossi, Haoliang Wang, and Julian
McAuley. Knowledge-aware query expansion with large language models for textual and
relational retrieval, 2025. URL https://arxiv.org/abs/2410.13765 .
[82] Millicent Li, Tongfei Chen, Benjamin Van Durme, and Patrick Xia. Multi-field adaptive retrieval,
2024. URL https://arxiv.org/abs/2410.20056 .
A Experimental details
A.1 Few-shot prompt for entity resolution
A.1.1 STaRK-prime
Question : "Which anatomical structures lack the expression of genes or proteins involved in the
interaction with the fucose metabolism pathway?"
Answer : "fucose metabolism"
Question : "What liquid drugs target the A2M gene/protein and bind to the PDGFR-beta receptor?"
Answer : "A2M gene/protein|PDGFR-beta receptor"
Question : "Which genes or proteins are linked to melanoma and also interact with TNFSF8?"
Answer : "melanoma|TNFSF8"
LLM insturction: "You are a knowledgeable assistant which identifies medical entities in the given
sentences. Separate entities using ’|’."
A.1.2 STaRK-mag
Question : "Could you find research articles on the measurement of radioactive gallium isotopes
disintegration rates?"
Answer : "FieldOfStudy:measurement of radioactive gallium isotopes disintegration rates"
Question : "What research on water absorption in different frequency ranges have been referenced or
deemed significant in the paper entitled ’High-resolution terahertz atmospheric water vapor continuum
measurements’"
Answer: "FieldOfStudy:water absorption in different frequency ranges Paper:High-resolution tera-
hertz atmospheric water vapor continuum measurements"
Question : "Publications by Point Park University authors on stellar populations in tidal tails"
Answer : "Institution:Point Park University
nFieldOfStudy:stellar populations in tidal tails"
Question : "Show me publications by A.J. Turvey on the topic of supersymmetry particle searches."
16

Answer: "Author:A.J. Turvey
nField of study: supersymmetry particle searches"
LLM instruction: "You are a smart assistant which identifies entities in a given questions. There are
institutions, authors, fields of study and papers."
A.2 K-hop query path templates
For both STaRK-prime and STaRK-mag, we use three path templates which are 1-hop, 2-hop and
length two paths that connect two entities to curate training Cypher queries. Figure 1 shows the query
for 2-hop. For 1-hop we substitue the pattern matching with:
MATCH (src {name: srcName})-[r]-(tgt)
and for length-two paths connecting entities, our query is:
UNWIND $src_names AS srcName1
UNWIND $src_names AS srcName2
MATCH (src1 {name: srcName1})-[r1]-(tgt)-[r2]-(src2 {name: srcName2}) WHERE src1 <>
src2
RETURN labels(src1)[0] AS label1, src1.name AS name1, type(r1) AS type1, labels(tgt)
[0] AS label2, type(r2) AS type2, labels(src2)[0] AS label3, src2.name AS name3,
count(DISTINCT tgt) AS totalCnt
17