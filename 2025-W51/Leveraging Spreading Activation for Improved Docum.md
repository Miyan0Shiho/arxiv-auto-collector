# Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems

**Authors**: Jovan Pavlović, Miklós Krész, László Hajdu

**Published**: 2025-12-17 19:38:35

**PDF URL**: [https://arxiv.org/pdf/2512.15922v1](https://arxiv.org/pdf/2512.15922v1)

## Abstract
Despite initial successes and a variety of architectures, retrieval-augmented generation (RAG) systems still struggle to reliably retrieve and connect the multi-step evidence required for complicated reasoning tasks. Most of the standard RAG frameworks regard all retrieved information as equally reliable, overlooking the varying credibility and interconnected nature of large textual corpora. GraphRAG approaches offer potential improvement to RAG systems by integrating knowledge graphs, which structure information into nodes and edges, capture entity relationships, and enable multi-step logical traversal. However, GraphRAG is not always an ideal solution as it depends on high-quality graph representations of the corpus, which requires either pre-existing knowledge graphs that are expensive to build and update, or automated graph construction pipelines that are often unreliable. Moreover, systems following this paradigm typically use large language models to guide graph traversal and evidence retrieval, leading to challenges similar to those encountered with standard RAG. In this paper, we propose a novel RAG framework that employs the spreading activation algorithm to retrieve information from a corpus of documents interconnected by automatically constructed knowledge graphs, thereby enhancing the performance of large language models on complex tasks such as multi-hop question answering. Experiments show that our method achieves better or comparable performance to iterative RAG methodologies, while also being easily integrable as a plug-and-play module with a wide range of RAG-based approaches. Combining our method with chain-of-thought iterative retrieval yields up to a 39\% absolute gain in answer correctness compared to naive RAG, achieving these results with small open-weight language models and highlighting its effectiveness in resource-constrained settings.

## Full Text


<!-- PDF content starts -->

LEVERAGINGSPREADINGACTIVATION FORIMPROVED
DOCUMENTRETRIEVAL INKNOWLEDGE-GRAPH-BASEDRAG
SYSTEMS
Jovan Pavlovi ´c2, Miklós Krész1,2, and László Hajdu1,2,3
1Innorenew CoE
2University of Primorska, FAMNIT
3Cognee Inc.
December 19, 2025
ABSTRACT
Despite initial successes and a variety of architectures, retrieval-augmented generation (RAG) systems
still struggle to reliably retrieve and connect the multi-step evidence required for complicated
reasoning tasks. Most of the standard RAG frameworks regard all retrieved information as equally
reliable, overlooking the varying credibility and interconnected nature of large textual corpora.
GraphRAG approaches offer potential improvement to RAG systems by integrating knowledge
graphs, which structure information into nodes and edges, capture entity relationships, and enable
multi-step logical traversal. However, GraphRAG is not always an ideal solution as it depends on high-
quality graph representations of the corpus, which requires either pre-existing knowledge graphs that
are expensive to build and update, or automated graph construction pipelines that are often unreliable.
Moreover, systems following this paradigm typically use large language models to guide graph
traversal and evidence retrieval, leading to challenges similar to those encountered with standard
RAG. In this paper, we propose a novel RAG framework that employs the spreading activation
algorithm to retrieve information from a corpus of documents interconnected by automatically
constructed knowledge graphs, thereby enhancing the performance of large language models on
complex tasks such as multi-hop question answering. Experiments show that our method achieves
better or comparable performance to iterative RAG methodologies, while also being easily integrable
as a plug-and-play module with a wide range of RAG-based approaches. Combining our method with
chain-of-thought iterative retrieval yields up to a 39% absolute gain in answer correctness compared
to naive RAG, achieving these results with small open-weight language models and highlighting its
effectiveness in resource-constrained settings.
1 Introduction
Initially proposed in [ 1,2], Retrieval-Augmented Generation (RAG) has become a popular technique for improving
the capabilities of Large Language Models (LLMs) across various tasks, including question answering and code
generation. The basic idea of RAG is to provide LLM access to an external, easily updatable knowledge source by
integrating it with the Information Retrieval (IR) component. This technique aims to improve the quality of LLM
responses by reducing hallucinations and enhancing text creation accuracy and coherence. Numerous enhancements to
the foundational RAG paradigm have been further proposed, ranging from innovative pipeline designs to improvements
in individual components. Additionally, several surveys have been conducted to classify, decompose, and identify key
elements within the myriad of proposed methodologies, helping track their evolution and highlighting promising future
directions [3, 4, 5, 6].
A particularly active research area explores the combination of RAG with graph-based systems that represent diversified
and relational information. The goal of these integrations is to improve performance on complex tasks involving large,arXiv:2512.15922v1  [cs.AI]  17 Dec 2025

APREPRINT- DECEMBER19, 2025
structured, and interconnected knowledge corpora, resulting in methodologies sometimes referred to as GraphRAG.
Comprehensive surveys covering the background, components, downstream tasks, and industrial use cases of GraphRAG,
as well as the technologies and evaluation methods used, are available in [7, 8].
As mentioned in [ 7], knowledge graph question answering (KGQA) is a key natural language processing task that has
motivated the development of KG-based systems aimed at responding to user queries through structured reasoning
over predefined knowledge graphs [ 9,10,11]. The multi-hop question answering (MHQA) task is a related but broader
concept, commonly applied in areas such as academic research, customer support, and financial or legal inquiries,
where a comprehensive analysis of multiple textual documents is required. In this scenario, given a document corpus,
the QA system generates responses to user queries by reasoning across multiple sources, involving sequential reasoning
steps where responses at one step may rely on answers from prior steps. Several standard and graph-based RAG
methodologies have been proposed to enhance the performance of LLMs on this specific task [ 12,13,14,15]. Despite
significant advancements, most of these approaches still rely primarily on standard or iterative retrieve-read-answer
workflows, which are enhanced by optimized pipeline components, LLM decision-making, or LLM-guided graph
traversal [ 16,4]. However, hallucination and weak faithfulness remain unresolved problems in the context of MHQA.
LLM-guided iterative retrieval may fail due to myopic knowledge exploration, retrieving context based on partial
reasoning from previous steps, which can be incorrect or contain fabricated information. Similarly, one-step RAG
systems with optimized retrieval components may fetch more useful evidence in the retrieved context, but they often
introduce a lot of noise or fail to capture information from "bridge" documents whose entities or phrases are not
mentioned in the input query. Thus, although adding improvements to vanilla RAG, the paradigm behind these advanced
methods stays fundamentally the same, thereby overlooking powerful and well-established IR algorithms that can
automatically identify relationships between documents and adapt retrieval for multi-hop query scenarios.
Motivated by this gap, this paper proposes a novel GraphRAG framework that integrates Spreading Activation (SA),
a method for searching associative networks originating in cognitive psychology. Beyond information retrieval [ 17],
SA has been successfully applied in several domains, including natural language processing [ 18,19], robotics, and
reinforcement learning [ 20]. Specifically, we propose a RAG system that operates on a hybrid structure combining a
knowledge graph with textual information from knowledge documents. It retrieves relevant documents from textual
corpora by executing an SA algorithm through the knowledge graph, identifying and fetching chunks of text with the
most pertinent information required to answer the multi-hop query, thereby capturing the reasoning structure and key
relationships between entities mentioned in the query.
To the best of our knowledge, only one prior paper has integrated RAG and SA-based methodologies [ 21]. However,
our approach differs in several key respects. First, the mentioned system relies on human-crafted knowledge graphs,
whereas our pipeline includes an automated knowledge-graph construction phase. Second, it uses a prompted LLM to
perform the SA procedure and expand the initially retrieved subgraph; by contrast, we perform SA automatically by
exploring the knowledge graph in a breadth-first manner and spreading the activation based on edge weights assigned
by an embedding model. Finally, the prior method fine-tunes the LLM to improve its instruction-following abilities,
while our method requires no retraining. Overall, we believe our methodology provides an innovative direction for
enhancing retrieval-augmented generation by leveraging a cognitively inspired mechanism that emphasizes associative
relevance rather than surface-level similarity.
We summarize the key contributions of our work as follows:
1.We introduce a knowledge-base indexing strategy that uses a prompt-tuned LLM to construct text-attributed
knowledge graphs, linking documents and enabling the use of graph-traversal techniques during document
retrieval.
2.We introduce a novel integration of the Spreading Activation algorithm with LLMs, aiming to enhance
grounded reasoning over large corpora of interconnected textual documents.
3.We demonstrate notable performance improvements of our pipeline over standard RAG frameworks through
experiments on two well-known multi-hop QA benchmarks. The results show that our SA-based document
retrieval method enables more efficient exploration of the knowledge corpus than dense-vector retrieval alone,
going beyond mere semantic similarity and enabling LLMs to produce more accurate answers to complex
multi-hop questions.
4.When combined with iterative retrieval, our method yields absolute improvements in answer accuracy ranging
from 25% to 39% relative to naive RAG.
5.Given the fact that our experiments are performed with small-open weight models, which require less
computational power, and yet we achieved significant results, we highlight the potential for deploying high-
performance reasoning systems in resource-constrained environments.
2

APREPRINT- DECEMBER19, 2025
2 Background and Related Work
2.1 RAG and Multi-Hop Question Answering
Although traditional RAG systems enhance the capabilities of LLMs for both knowledge-intensive and standard
NLP tasks, they encounter several critical issues that limit their performance when handling complex tasks. This is
particularly evident in MHQA [ 22], where the system is required to reason across multiple pieces of information or
documents to answer complicated questions. Traditional systems rely on simple, single-shot retrieval, which often fails
to capture all necessary information or may even overload the LLM’s context window with unnecessary or incorrect
data. Numerous additions are proposed through the evolution of RAG paradigms with the aim of overcoming naive
"retrieve-then-generate" methodology, which, according to the survey [ 4], can be classified into two categories:training-
freeandtraining-basedapproaches. Training-free approaches aim to optimize RAG pipelines through techniques
such as prompt engineering, in-context learning, improved organization of data in the knowledge corpus, or agentic
design [ 16]. In contrast, training-based methods rely on fine-tuning the retriever and generator components to boost the
system’s performance on downstream tasks.
Since our methodology does not require pre-training of any RAG components, we highlight a few training-free
approaches that have been shown to enhance the performance of RAG systems in multi-hop question answering.
Multiple advanced RAG techniques usequery expansionas a means to improve retrieval steps by generating richer,
semantically varied reformulations of the original question to match more relevant evidence from the knowledge corpus.
Articles [ 23,24] propose methods that first generate plausible answer-like passages for the input query and then embed
them using sparse or dense encoders to match semantically similar real documents from the knowledge base.
Papers [ 25,26] suggest that complex, implicit, or multi-hop queries should be rewritten, disambiguated, or decomposed
before retrieval to optimize the fetched context.
Several papers have discussed the implementation of iterative retrieval procedures to enhance system performance
on multi-hop questions, suggesting that retrieval should be viewed as an operation interleaved with the LLM’s own
intermediate reasoning.
In [27] authors propose a RAG system integrated with Chain -of-thought (CoT), a prompting technique, where each
CoT step becomes the query for the next retrieval. Similarly, [ 28] proposes system based on Tree -of-Thoughts
(ToT) structured prompting framework; starting from an underspecified root question, the model generates candidate
clarification subquestions, prunes uninformative branches, and for each surviving node runs a retrieval round targeted
to that clarification, so that by the time the leaves are answered the system has accumulated disjoint, high-precision
evidence for answering the original query.
Few papers propose integrating advanced RAG mechanisms with reflection modules that allow LLMs to determine the
most suitable retrieval strategy based on the input query [ 13,29,30,31]. By reflecting on the question, the LLM can
choose whether to perform simple retrieval, engage in iterative retrieval steps, or rely on its own internal knowledge.
Some systems can dynamically guide the direction and number of retrieval steps, evaluate the relevance of retrieved
content [30], or make use of external tools such as online search [31].
Many of these training-free methods are plug-and-play, agnostic to the specific retriever or model, and can be combined
both with each other and with different RAG paradigms or fine-tuned models.
2.2 GraphRAG
The GraphRAG paradigm [ 7,8] aims to extend RAG systems by leveraging the structured relationships and hierarchical
organization of information in graph data to improve multi-hop reasoning and contextual understanding.
One of the first successful GraphRAG systems, introduced in [ 32], was designed for Query-Focused Summarization
(QFS), where, provided a query and a document corpus, the system seeks to compress the contents of multiple
documents, gathering the identified themes and providing a unified summary of the entire document corpus. The
proposed approach focuses on the connection between the summarization task and the inherent modularity of graphs,
which allows partitioning of graphs into communities. LLM is first instructed to create a knowledge graph from source
texts by extracting entities and relationships between them. The system then uses the Leiden community detection
algorithm to partition this graph into hierarchical communities. Finally, each community is summarized separately, and
the summaries are combined to produce a final, comprehensive response to the user’s question.
With respect to MHQA, few works have received significant attention for effectively integrating graph-based retrieval
and reasoning within the RAG.
3

APREPRINT- DECEMBER19, 2025
LightRAG proposed in [ 33] firstly builds a lightweight knowledge graph over a document corpus by using an LLM to
extract entities and relations from chunks. For each node and edge, it then constructs a key–value entry, where the key
is a short textual label and the value is a concise summary derived from the source documents. At query time, another
language model analyses the input and produces two sets of query terms, one focusing on concrete objects and relations,
and another capturing broader topics. Terms from the first set are embedded and used for vector search over keys tied to
entity nodes, while the broader theme terms are used to match keys associated with relation edges. The retrieved graph
elements are expanded by one hop to include their neighbors, then the corresponding value texts are combined into a
compact context that is passed to a generative model to produce the final answer.
In [34], the authors present G-Retriever, a framework that leverages the integration of graph neural networks and LLMs.
The system efficiently responds to user queries by retrieving relevant subgraphs using a Prize-Collecting Steiner Tree
algorithm and translating graph data to a textual representation. The graph encoder and projection layer, which are in
charge of aligning subgraph information with the language model’s hidden space, are fine-tuned via backpropagation
based on the difference between the generated answer and the ground truth. This technique is evaluated on different
tasks, ranging from commonsense reasoning and scene comprehension to knowledge-based question answering.
Another design for an MHQA pipeline is described in [ 15], which includes creating a schemaless knowledge graph
with instruction-tuned LLM. When a query is received, the language model extracts the key named entities from the
query, which are then mapped to their associated nodes in the knowledge graph using similarity scores calculated by the
retrieval encoder. These nodes serve as seeds for a graph search using the Personalized PageRank algorithm, which
propagates activation through the network to identify nodes that may be associated with the provided question.
These systems differ, however, from our approach in that they search a constructed knowledge graph to gather relational
knowledge, then convert this context into textual form before passing it to the LLM. In contrast, our methodology uses
knowledge graph search to match textual documents directly without requiring an intermediate step of transforming
graph data into text.
2.3 Spreading Activation
Paper [ 35] proposed that human semantic memory is organized as a network and was one of the first to implement a
semantic network model of knowledge in a computer. In this framework, concepts and their properties were modeled as
nodes in a network connected by links encoding semantic relationships, arranged hierarchically so that more specific
concepts point to more general ones. Activation spreading was introduced as a mechanism to search this network
in order to verify factual sentences (e.g., canary is a bird). This was achieved by activating a concept and letting
that activation move through the network until it intersected with activation from another concept. Later in [ 36] the
computational model was tested against human performance in sentence–verification tasks, where people’s response
times to statements at different hierarchical levels (e.g. “a canary is a bird” vs. “a canary is an animal”) were compared
to the model’s predictions, showing that longer network traversals correlate with slower verification times. This model
was then modified in [ 37,38] to account for varying association strengths and decay of activation over distance, dropping
the requirement that the network is strictly hierarchical. The spreading mechanism was generalized so that activation
spreads from a currently active concept to all linked concepts, in parallel, with the strength that depends on link weight
and fades with distance.
After its development in the field of cognitive psychology, SA was adapted in later works as a computational tool
for information retrieval and recommender systems. In this context, original principles were repurposed so that input
queries or user profiles trigger activation in a graph of terms, documents, or items, and the resulting activation pattern is
used to rank candidates by their inferred relevance or interest.
In [39], the authors argued that SA can be implemented in IR systems to overcome the limitations of standard TF-
IDF-based approaches, particularly when the user’s wording does not match the wording of the documents and when
relevant documents are only indirectly connected to the query. They proposed a system using network representation
of the document corpus, where both documents and index terms are modeled as nodes, interconnected by various
types of links: term–document links modeling the presence of a term in a document, term–term links connecting
terms that frequently co-occur or are semantically related according to a thesaurus, and document–document links
connecting documents that share co-occurring terms. In this setting, the SA procedure starts by taking the query terms
or documents retrieved by the normal vector-space search and assigning them an initial activation value. The activation
then propagates from every active node to its neighbors, with the amount passed along each link scaled by the link’s
strength, and nodes whose accumulated activation exceeds a specified threshold are finally retrieved. However, the
authors implemented and evaluated only a simplified version of the described procedure due to the computational costs
of a full implementation at that time. They concluded that the spreading activation procedure is a useful complementary
module to standard IR systems, but it needs to be constrained by limiting the number of steps to just a few hops from
4

APREPRINT- DECEMBER19, 2025
the original query nodes and by using a decay factor smaller than one. Later works expanded criticism of unconstrained
SA along the same lines; In [40], the authors showed that the pure SA algorithm, where activation propagates over all
outgoing links with no limits on hops, fan-out, or thresholds, converges to a fixed state and retrieval results become
nearly independent of the input query. This underscores that heuristic constraints in applied SA-based systems are not
just pragmatic features but essential for the retrieval of query-dependent results.
More recent applications in IR and recommendation systems, therefore, employ adapted, task-driven variants of SA,
where the spread is guided by the user’s information need or context and then combined with conventional ranking
steps. In [ 41], the authors propose a semantic text search method based on SA, which, given an input query, performs
named entity recognition and relationship extraction, then maps the extracted elements to an ontology to represent the
query as a concise structured pattern. The method then spreads activation only to concepts that match the relationships
actually expressed in the query, as well as to closely related information such as alternative names or parent categories.
Activated concepts are retrieved and appended to the original query to form an expanded query, which is then executed
by a classical sparse-vector search engine, allowing ordinary document ranking to exploit both the user’s words and the
newly discovered latent concepts.
Article [ 42] describes a semantic, knowledge-based recommender built around an ontology that stores domain concepts,
user interests, and item descriptors, and then runs a controlled spreading of activation from the user profile toward
semantically adjacent items so that recommendations reflect both explicit preferences and latent relations present in
the knowledge base. The system routes the activation only through specific, typed links in the ontology, assigning
weight to each link based on its type and finally ranking associated items according to their activation values and
standard recommendation filters. Similarly, [ 43] extends ontology-based SA with contextual cues, such as current
task or situational attributes, so that activation is fired not only from the long-term user profile but also from context
nodes. Activation propagates only along meaningful ontology links, with decay and link weights, thereby avoiding
uncontrolled diffusion through the network.
3 Experiment Design and Evaluations
3.1 Benchmarks
We evaluate our system on two well-known MHQA benchmarks: MuSiQue [44] and 2WikiMultiHopQA [45].
One of the earliest and most widely used multi-hop QA datasets was HotpotQA [ 46], created through crowdsourced
multi-hop questions over Wikipedia articles. However, it has faced considerable criticism for not consistently requiring
genuine multi-hop reasoning, as many questions could be answered through dataset artifacts or inference shortcuts [ 47,
48, 49].
In response, 2WikiMultiHopQA [ 45] refined the evaluation paradigm by combining unstructured Wikipedia text with
structured Wikidata knowledge, thereby mitigating some of these issues. MuSiQue [ 44] advanced this approach further
by composing multi-hop questions in a bottom-up manner from a large pool of single-hop questions while enforcing
strict connected reasoning. Therefore, in our experiments, we used 2WikiMultiHopQA and MuSiQue to evaluate the
performance gains introduced by our methodology.
Given the high computational expense associated with LLM-based knowledge graph construction, we restrict our
evaluation to a random subset of 100 questions per benchmark.
3.2 Baselines
In the experiments, we include three different types of baseline RAG methods against which we compare our proposed
pipeline. First, we employ the Naive RAG baseline, which retrieves the top- kdocuments with the highest cosine
similarity between their embeddings and the query embedding, and feeds them directly into the LLM’s prompt together
with the input query. We evaluate two variants of this baseline, withk= 5andk= 10.
Second, we consider CoT-based RAG systems, which perform iterative retrieval. Here, the LLM is prompted to reason
over the context gathered in each retrieval step and attempt to answer the original question. If the answer cannot
be determined from the current context, the LLM formulates a follow-up question, based on which a new context is
fetched and added to a short-term memory containing summarised knowledge from previous steps. This process runs
sequentially for a pre-specified maximum of three steps (as increasing the number of steps beyond this did not improve
performance). We also have two variants for this method for top-kvalues of 5 and 10.
5

APREPRINT- DECEMBER19, 2025
Finally, we have a system based on query-decomposition methodology, where the LLM is prompted to decompose a
complex question into a sequence of simpler, preferably single-hop queries, which are then used to sequentially gather
context and derive intermediate answers.
Details and exact prompts used for all baselines are provided in Appendix A.
4 Results
Table 1 presents the experimental results, comparing the performance of different RAG approaches across the bench-
marks. The experiments were conducted using two LLMs for reasoning and answer generation,phi4andgemma3, and
BAAI/bge-large-en-v1.5was used to generate the text embeddings. Beyond standard EM and F1 metrics, we included
a manualCorrectnesscheck to detect cases where the model produced semantically correct responses (e.g., valid
aliases) that EM failed to reward, as well as for answers that obtained high F1 scores due to partial lexical overlap while
remaining factually incorrect.
Two main observations can be made from the table. First, in experiments conducted with thephi4model, SA-RAG alone
outperformed all other baseline approaches on both benchmarks. In contrast, this result did not replicate withgemma3.
Our inspection of the retrieved context ingemma3experiments suggests that the absence of similar improvements
was not due to missing or insufficient contextual information, but rather due to the model’s own reasoning limitations.
Althoughphi4is smaller, it was specifically designed for complex reasoning tasks, unlikegemma3.
The second observation is that SA-RAG provides consistent performance gains when introduced as a plug-and-play
module in any of the examined training-free pipelines, in all of the experiment setups. As shown in the table, combining
SA-RAG with CoT iterative retrieval yields the strongest performance, significantly outperforming all baseline RAG
approaches, and providing 25% to 39% absolute gains in answer correctness compared to Naive RAG.
Table 1: Results of RAG evaluation on two multi-hop benchmarks.
(a) MuSiQue
phi4 gemma3
Methodology EM F1 Correctness EM F1 Correctness
Naive RAG 0.25 0.36 45 0.26 0.36 44
Naive RAG (k=10) 0.23 0.38 48 0.32 0.41 53
CoT RAG 0.33 0.44 55 0.39 0.50 58
CoT RAG (k=10) 0.33 0.46 55 0.34 0.45 56
Query-decomposition 0.34 0.48 55 0.40 0.51 60
SA-RAG 0.40 0.54 67 0.35 0.48 56
SA-RAG + CoT0.44 0.61 74 0.43 0.57 69
SA-RAG + decomposition 0.39 0.51 66 0.34 0.54 62
(b) 2WikiMultihopQA
phi4 gemma3
Methodology EM F1 Correctness EM F1 Correctness
Naive RAG 0.36 0.42 48 0.36 0.44 48
Naive RAG (k=10) 0.46 0.52 58 0.49 0.57 62
CoT RAG 0.53 0.62 68 0.52 0.62 72
CoT RAG (k=10) 0.45 0.51 55 0.56 0.65 72
Query-decomposition 0.60 0.68 75 0.47 0.57 64
SA-RAG 0.57 0.66 76 0.50 0.59 66
SA-RAG + CoT0.72 0.78 87 0.58 0.69 77
SA-RAG + decomposition 0.60 0.72 83 0.46 0.60 69
6

APREPRINT- DECEMBER19, 2025
5 Methodology
5.1 Problem definition
Given a question qand a knowledge base KB={D 1, . . . , D n}containing a corpus of textual documents organized in
some manner, the system generates an answer aby retrieving a subset of relevant documents D ⊆KB . Formally, this
can be expressed as:
D=R(q, KB)
a=LLM(q,D)
where Rdenotes the retrieval component of the system, andLLMrefers to the large language model responsible for
generating the answer. The objective is to maximize the probability of producing the correct answera∗, that is,
p(a∗=a|q, KB) =X
D⊆KBp(a∗=a|q,D)p(D |q, KB).
Since the generation process is not directly optimized, the task reduces to identifying the subset of documents D∗that
maximizes the probability of providing the correct answer:
D∗= arg max
D⊆KBp(a∗|q,D)p(D |q, KB).
5.2 System description
We summarize the architecture of our RAG system enhanced with a spreading activation–based retrieval mechanism.
It consists of four main stages: indexing, subgraph fetching, spreading activation and document retrieval, and finally,
answer generation. These stages are summarized below:
1.Indexing: In this phase, we break knowledge documents into chunks, create vector embeddings for these
text chunks, and use them to build a knowledge graph of the entities, entity descriptions, and relationships
mentioned. We also record references between each chunk and every entity it contains by storing the chunks
as nodes in the graph, with outgoing links connecting them to the corresponding entity nodes.
2.Subgraph fetching: Given a query, we fetch a subgraph consisting of the most relevant nodes and relations
based on the cosine similarity calculated between the vector embedding of the query and the embeddings of
entity descriptions. Here, we also determine the set of initially activated entities according to their relevance as
indicated by cosine similarity.
3.Spreading activation and document retrieval: We run spreading activation from the initially activated entities
to obtain a subset of relevant nodes and fetch the textual paragraphs that mention these nodes. We also record
the most relevant relations and combine their textual descriptions with the fetched documents.
4.Answer generation: The LLM then generates a grounded answer using the information from the fetched
context.
Figure 1 provides a high-level overview of the proposed methodology. The prompts used at each step of our pipeline
are detailed in Appendix B.
7

APREPRINT- DECEMBER19, 2025
Figure 1: High-level overview of methodology
5.3 Indexing
We begin the indexing phase by performing word-based chunking, by splitting input documents into chunks of 500
words with an overlap of 200 words between consecutive chunks. Each chunk is passed to the embedding model and
then to a prompt-tuned LLM, which is instructed to extract entities, relations, and entity descriptions from the text.
We construct a knowledge graph containing three types of nodes: entities, entity descriptions, and documents (which
represent the textual chunks). The graph also includes three types of links:describesandrelated_to. Thedescribes
links connect document and entity description nodes to their corresponding entities, while therelated_tolinks connect
two entities that are described as related within the text chunk. Additionally, eachrelated_tolink stores a string attribute
holding the textual description of the relationship between the endpoint entities. Each entity node stores itsnameas well
as a list of alternative names within analiasesattribute. When adding new elements to the knowledge graph, we ensure
that duplicate entities are not created by first checking whether thenameor any of thealiasesof the newly extracted
entities overlap with those of previously extracted entities. If the entity already exists, we create new entity description
nodes and adescribeslink connecting them to the corresponding entity. Additionally, we create vector embeddings for
both entity description nodes andrelated_tolinks using the same embedding model applied to the textual chunks. The
indexing process is illustrated in Figure 2.
8

APREPRINT- DECEMBER19, 2025
Figure 2: Knowledge graph creation during the indexing phase. The figure shows an example graph constructed from
a single document describing the capital cities of European states. The document node is marked in green, entity
description nodes are colored orange, and entity nodes are shown in blue.
5.4 Subgraph fetching
At this stage, we retrieve the subgraph consisting of entities and relations that are potentially relevant to answering
the input query. Specifically, we first identify the top- kentity descriptions whose embeddings have the highest cosine
similarity to the query embedding. We then include the entities referenced by these descriptions through incoming
describeslinks and expand the graph by retrieving all entities andrelated_tolinks within their n-hop neighborhood.
The “seed” entities matched in the first step are marked as initially activated for the spreading activation process, and
each link is assigned a weight equal to the cosine similarity between the query and the link embedding. We set the
values of kandnto 3 and 4, respectively, for the MuSiQue experiments, and to 10 and 3 for the 2WikiMultiHopQA
experiments, as these parameter choices yielded the best results. The process is illustrated in Figure 3.
9

APREPRINT- DECEMBER19, 2025
Figure 3: Subgraph fetching step: Orange nodes represent thetop-kentity description nodes, while entity nodes
are shown in blue. Links of thedescribestype are colored orange, whereas blue links representrelated_torelations
connecting entity nodes. Initially activated “seed” entities are indicated by a red border.
5.5 Spreading activation and document retrieval
This stage simulates spreading activation on the retrieved subgraph, starting from the initially activated nodes identified
in the previous step. Before the propagation begins, we rescale the edge weights by a linear factor c, using the formula
w′=w−c
1−c. This adjustment helps prevent overactivation, which could otherwise result in nearly every node becoming
strongly activated and ultimately leading to context explosion. In our experiments, we found that setting c= 0.4
provided the best results in terms of the recall/precision tradeoff. A more detailed illustration of how the value of c
impacts the outcome of the spreading activation can be found in Figure 5.
The SA process operates as follows: initially, activated nodes are assigned an activation value of ai= 1, while all
other nodes get a value of aj= 0. Activation then propagates outward from these initially activated nodes to their
neighboring nodes in a BFS manner, through the weighted links wij. In each iteration, the activation value of each
target node is updated as aj=min(a j+P
i∈N(j)ai·wij)and after the process is complete, nodes whose activation
exceeds the threshold τaare marked as activated. The entire procedure is summarized in Algorithm 1, and Figure 4
provides a visualization of the SA run on the subgraph retrieved for a multi-hop query from the MuSiQue benchmark.
10

APREPRINT- DECEMBER19, 2025
Algorithm 1:Spreading Activation algorithm on entity subgraph
Input:Adjacency dictionaryadj_dictmapping entities to outgoing weighted arcs;
Set of initially activated entitiesI a;
Activation thresholdτ a;
Output:Set of activated entitiesR a.
foreache∈keys(adj_dict)do
activation_value[e]←0
foreache∈I ado
activation_value[e]←1
visited← ∅
Q←queue initialized withe
whileQis not emptydo
node←Q.pop_left()
ifnode∈visitedthen
continue
visited←visited∪ {node}
foreach(target, weight)∈adj_dict[node]do
activation_value[target]←min 
1, activation_value[target]+weight·activation_value[node]
iftarget /∈visitedthen
Q.append(target)
Ra← {e|activation_value[e]> τ a}
returnR a
Figure 4: Spreading activation on the subgraph fetched for the query3hop2__655849_223623_162182from MuSiQue
dataset. Golden entities are marked in yellow, activated entities in red, unactivated entities in light blue, and activated
golden entities in pink.
After obtaining the set of activated entities, we fetch the set of documents D={D 1, ...D k1}from the knowledge
base, consisting of document nodes that have outgoingdescribeslinks connecting them to the activated entity nodes.
Here, we also perform filtering by removing documents whose cosine similarity to the query is lower than the pruning
threshold τd. Additionally, we augment the set of retrieved documents with the textual representations of relevant
relationships r1, ...rk2between activated entities, as encoded in the links between them. A relationship is considered
relevant if its weight, determined based on cosine similarity at the beginning, exceeds the threshold τr. In this way, we
obtain the final context C={D 1, ...D k1, r1, ...rk2}, which is then passed to the LLM in the next stage. The values of
τa,τd, andτ rthat produced the best performance in our experiments are 0.5, 0.45, and 0.5, respectively.
11

APREPRINT- DECEMBER19, 2025
Figure 5: Effect of applying a linear normalization factor to the edge weights of the fetched subgraph on spreading
activation dynamics. The first column corresponds toc= 0.5, the second toc= 0.4, and the third toc= 0.3.
5.6 Answer generation
Finally, provided with the fetched context and the multi-hop query, the LLM is instructed to reason through the
knowledge text in a step-by-step manner, verifying all arithmetic calculations if the question requires quantitative
reasoning. It then outputs its reasoning and the final answer if it is able to deduce one based solely on the provided
knowledge, or otherwise notes that the question is not answerable based on the retrieved information.
6 Conclusion & Limitation
In this paper, we addressed the challenges of document retrieval in RAG systems for complex tasks that require
multi-step reasoning and evidence aggregation from multiple documents, such as MHQA. We proposed integrating
the SA information retrieval algorithm with RAG approaches based on knowledge graphs to overcome the limitations
of current systems without the need for expensive language model fine-tuning. Our experiments demonstrate the
potential of this technique by achieving, with a single retrieval step, performance equivalent to or better than several
training-free RAG baselines that use iterative retrieval, and further show that it can be combined with methods like CoT
to significantly boost system performance, even when using small open-weight language models.
However, several limitations of the current work should be acknowledged and addressed in future research. First, due to
budget constraints and high experimental costs, we evaluated our method on a small sample of 100 randomly selected
questions from each benchmark. More extensive validation is needed to empirically demonstrate the scalability of
our system. Second, when assigning association weights to relationship links in the knowledge graph, we used cosine
similarity values from an "off-the-shelf" embedding model and adapted them only through simple linear scaling for our
task. Future work should explore improvements to the retrieval components, such as fine-tuning the embedding model
for better task alignment.
References
[1]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language
model pre-training. InInternational conference on machine learning, pages 3929–3938. PMLR, 2020.
[2]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich
Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems, 33:9459–9474, 2020.
[3]Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling Yang, Wentao
Zhang, Jie Jiang, and Bin Cui. Retrieval-augmented generation for ai-generated content: A survey.arXiv preprint
arXiv:2402.19473, 2024.
[4]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A
survey on rag meeting llms: Towards retrieval-augmented large language models. InProceedings of the 30th ACM
SIGKDD conference on knowledge discovery and data mining, pages 6491–6501, 2024.
[5]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2(1), 2023.
12

APREPRINT- DECEMBER19, 2025
[6]Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. A comprehensive survey of retrieval-augmented
generation (rag): Evolution, current landscape and future directions.arXiv preprint arXiv:2410.12837, 2024.
[7]Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang.
Graph retrieval-augmented generation: A survey.arXiv preprint arXiv:2408.08921, 2024.
[8]Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A
Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al. Retrieval-augmented generation with graphs (graphrag).
arXiv preprint arXiv:2501.00309, 2024.
[9]Jing Zhang, Bo Chen, Lingxi Zhang, Xirui Ke, and Haipeng Ding. Neural, symbolic and neural-symbolic
reasoning on knowledge graphs.AI Open, 2:14–35, 2021.
[10] Ernests Lavrinovics, Russa Biswas, Johannes Bjerva, and Katja Hose. Knowledge graphs, large language models,
and hallucinations: An nlp perspective.Journal of Web Semantics, 85:100844, 2025.
[11] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models
and knowledge graphs: A roadmap.IEEE Transactions on Knowledge and Data Engineering, 36(7):3580–3599,
2024.
[12] Jiahao Zhang, Haiyang Zhang, Dongmei Zhang, Yong Liu, and Shen Huang. End-to-end beam retrieval for
multi-hop question answering.arXiv preprint arXiv:2308.08973, 2023.
[13] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. Adaptive-rag: Learning to adapt
retrieval-augmented large language models through question complexity.arXiv preprint arXiv:2403.14403, 2024.
[14] Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. Knowledge graph prompting for
multi-document question answering. InProceedings of the AAAI conference on artificial intelligence, volume 38
(17), pages 19206–19214, 2024.
[15] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically
inspired long-term memory for large language models.Advances in Neural Information Processing Systems,
37:59532–59569, 2024.
[16] Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Talaei Khoei. Agentic retrieval-augmented generation: A
survey on agentic rag.arXiv preprint arXiv:2501.09136, 2025.
[17] Fabio Crestani. Application of spreading activation techniques in information retrieval.Artificial Intelligence
Review, 11(6):453–482, 1997.
[18] Jordan Pollack and David Waltz. Natural language processlng using spreading activation and lateral inhibition. In
Proceedings of the Annual Meeting of the Cognitive Science Society, volume 4, 1982.
[19] George Tsatsaronis, Michalis Vazirgiannis, and Ion Androutsopoulos. Word sense disambiguation with spreading
activation networks generated from thesauri. InIJCAI, volume 27, pages 223–252. Hyderabad, 2007.
[20] Hitoshi Kono, Ren Katayama, Yusaku Takakuwa, Wen Wen, and Tsuyoshi Suzuki. Activation and spreading
sequence for spreading activation policy selection method in transfer reinforcement learning.International Journal
of Advanced Computer Science and Applications, 10(12), 2019.
[21] Dingjun Wu, Yukun Yan, Zhenghao Liu, Zhiyuan Liu, and Maosong Sun. Kg-infused rag: Augmenting corpus-
based rag with external knowledge graphs.arXiv preprint arXiv:2506.09542, 2025.
[22] Vaibhav Mavi, Anubhav Jangra, Adam Jatowt, et al. Multi-hop question answering.Foundations and Trends® in
Information Retrieval, 17(5):457–586, 2024.
[23] Liang Wang, Nan Yang, and Furu Wei. Query2doc: Query expansion with large language models.arXiv preprint
arXiv:2303.07678, 2023.
[24] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval without relevance
labels. InProceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers), pages 1762–1777, 2023.
[25] Xinbei Ma, Yeyun Gong, Pengcheng He, Nan Duan, et al. Query rewriting in retrieval-augmented large language
models. InThe 2023 Conference on Empirical Methods in Natural Language Processing, 2023.
[26] Ruiliu Fu, Han Wang, Xuejun Zhang, Jun Zhou, and Yonghong Yan. Decomposing complex questions makes
multi-hop qa easier and more interpretable.arXiv preprint arXiv:2110.13472, 2021.
[27] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with
chain-of-thought reasoning for knowledge-intensive multi-step questions.arXiv preprint arXiv:2212.10509, 2022.
13

APREPRINT- DECEMBER19, 2025
[28] Gangwoo Kim, Sungdong Kim, Byeongguk Jeon, Joonsuk Park, and Jaewoo Kang. Tree of clarifications:
Answering ambiguous questions with retrieval-augmented large language models. InProceedings of the 2023
Conference on Empirical Methods in Natural Language Processing, pages 996–1009, 2023.
[29] Yile Wang, Peng Li, Maosong Sun, and Yang Liu. Self-knowledge guided retrieval augmentation for large
language models.arXiv preprint arXiv:2310.05002, 2023.
[30] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve,
generate, and critique through self-reflection.arXiv preprint arXiv:2310.11511, 2023.
[31] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective retrieval augmented generation.arXiv e-prints,
pages arXiv–2401, 2024.
[32] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha
Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to
query-focused summarization.arXiv preprint arXiv:2404.16130, 2024.
[33] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented
generation.arXiv preprint arXiv:2410.05779, 2024.
[34] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan
Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering.
Advances in Neural Information Processing Systems, 37:132876–132907, 2024.
[35] M Ross Quillian. Word concepts: A theory and simulation of some basic semantic capabilities.Behavioral
science, 12(5):410–430, 1967.
[36] Allan M Collins and M Ross Quillian. Retrieval time from semantic memory.Journal of verbal learning and
verbal behavior, 8(2):240–247, 1969.
[37] Allan M Collins and Elizabeth F Loftus. A spreading-activation theory of semantic processing.Psychological
review, 82(6):407, 1975.
[38] John R Anderson. A spreading activation theory of memory.Journal of verbal learning and verbal behavior,
22(3):261–295, 1983.
[39] Gerard Salton and Chris Buckley. On the use of spreading activation methods in automatic information. In
Proceedings of the 11th annual international ACM SIGIR conference on Research and development in information
retrieval, pages 147–160, 1988.
[40] Michael R Berthold, Ulrik Brandes, Tobias Kötter, Martin Mader, Uwe Nagel, and Kilian Thiel. Pure spreading
activation is pointless. InProceedings of the 18th ACM conference on Information and knowledge management,
pages 1915–1918, 2009.
[41] Vuong M Ngo. Discovering latent information by spreading activation algorithm for document retrieval.Ngo,
Vuong M.“Discovering Latent Information By Spreading Activation Algorithm for Document Retrieval. ” Academy
and Industry Research Collaboration Center, January 31, 2014. https://doi. org/10.5121/ijaia. 2014.5102., 2014.
[42] Yolanda Blanco-Fernández, Martín López-Nores, and José J Pazos-Arias. Adapting spreading activation techniques
towards a new approach to content-based recommender systems. InIntelligent interactive multimedia systems and
services, pages 1–11. Springer, 2010.
[43] Sachin Papneja, Kapil Sharma, and Nitesh Khilwani. Context aware personalized content recommendation using
ontology based spreading activation.International Journal of Information Technology, 10(2):133–138, 2018.
[44] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via
single-hop question composition.Transactions of the Association for Computational Linguistics, 10:539–554,
2022.
[45] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for
comprehensive evaluation of reasoning steps.arXiv preprint arXiv:2011.01060, 2020.
[46] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christo-
pher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering.arXiv preprint
arXiv:1809.09600, 2018.
[47] Jifan Chen and Greg Durrett. Understanding dataset design choices for multi-hop reasoning.arXiv preprint
arXiv:1904.12106, 2019.
[48] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Is multihop qa in dire condition?
measuring and reducing disconnected reasoning.arXiv preprint arXiv:2005.00789, 2020.
[49] Sewon Min, Eric Wallace, Sameer Singh, Matt Gardner, Hannaneh Hajishirzi, and Luke Zettlemoyer. Composi-
tional questions do not necessitate multi-hop reasoning.arXiv preprint arXiv:1906.02900, 2019.
14

APREPRINT- DECEMBER19, 2025
A Implementation Details of Baseline RAG Methodologies
In this appendix, we provide a detailed description of the implementation details for the baseline RAG methods
evaluated in our experiments, including the retrieval systems and the prompts used. As specified, we consider three
systems:Naive RAG,CoT RAGbased on iterative retrieval, and a system based onquery-decomposition prompting.
It is important to note that all three systems use the same prompt template, shown in Listing 1, to generate the final
answer after the retrieval step. Once the relevant context is gathered from the document corpus, it is embedded into
this template to instruct the LLM to reason over the retrieved information and produce a final answer in the form of a
JSON object. This object uses the structured output capabilities of the LLM and contains two textual fields: reasoning
andfinal_answer . Including the reasoning field improves LLM performance by allowing the model to carry out
intermediate deduction steps and calculations before producing the final answer, while also reducing the likelihood of
unnecessary information appearing in thefinal_answerfield.
Below is a question followed by some context from different sources. Please answer the question based on
,→the context. The answer to the question could be either single word, yes/no or consist of
,→multiple words describing single entity. If the provided information is insufficient to answer
,→the question, respond ’Insufficient Information’. Answer directly without explanation. Provide
,→answer in JSON object with two string attributes, ’reasoning’, which" provides your detailed
,→reasoning about the answer, and ’final_answer’ where you provide your short final answer without
,→explaining your reasoning.
Listing 1: Answering prompt for baseline systems
The design of theNaive RAGsystem is relatively simple. First, the knowledge documents are split into overlapping
chunks of 500 words, with an overlap size of 100 words, and stored in a database together with their vector embeddings.
Given an input query, the embedding model generates a dense vector representation of the query text, which is then
compared against the embeddings of the document chunks in the knowledge corpus to retrieve thetop- kmost similar
chunks based on cosine similarity.
CoT RAG, on the other hand, performs a sequence of iterative retrieval steps. At each step, it determines whether the
answer to the question can be inferred from the currently available information and generates a follow-up question if the
answer cannot be deduced from the provided context. During each reasoning step, the LLM is also asked to summarize
the information from the current context, and this summary is passed to subsequent steps to be combined with the text
from newly retrieved chunks. The prompt used for this purpose is shown in Listing 2.
You are given a **multi-hop question** that requires combining information across multiple source
,→documents. Along with the question, you are provided a list of **short knowledge paragraphs**,
,→each with different pieces of information potentially required for answering the query.
## Task description
Your goal is to **understand and reason through the question** using only the information provided.
To achieve this you should:
- **Extract the relevant** facts from the knowledge paragraphs that are needed to answer the question.
- **Paraphrase and organize** these facts into a clear and coherent summary called ’provided_context’.
,→This summary should explain how the relevant entities and relationships come together to support
,→answering the question.
- Based on this reasoning:
- If the information is **sufficient to directly answer the question**, provide the answer.
- If important information is **missing**, return a **specific follow-up question** that would help
,→fill the gap.
### Output Format (JSON)
‘‘‘json
{
"provided_context": "Summarize only the facts relevant to answering the question, combining them into a
,→clear explanation.",
"answer_possible": true | false,
"final_answer": "Give your answer here if ’answer_possible’ is true, otherwise leave this blank.",
"additional_question": "If ’answer_possible’ is false, write a clear and specific follow-up question
,→that would help get the missing information."
}
‘‘‘
# Input
You are given a **multi-hop question** that requires combining information across multiple source
,→documents. Along with the question, you are provided: a list of **short knowledge paragraphs**,
,→each describing specific entities and facts
‘‘‘
# Input
15

APREPRINT- DECEMBER19, 2025
Listing 2: Reasoning prompt for baseline systems
Finally, Listing 3 displays the prompt that serves as the backbone of thequery-decompositionmethod. Here, the LLM
is asked to decompose a multi-hop query into a list of simpler queries whose answers provide intermediate steps needed
to answer the original query. After decomposition, the system answers these simpler questions sequentially, one at
a time, in the specified order. At each step, the system first combines the current sub-query with the questions and
answers from previous steps to generate a vector embedding for retrieval of the most relevant knowledge passages, and
the LLM then uses this retrieved context to answer the current sub-query. Once all sub-queries have been answered,
the system performs a final retrieval step based on an embedding of the original query to gather additional relevant
documents. This retrieved context is then combined with the full list of sub-queries and their answers and provided to
the LLM as context for deriving the final answer.
You are a decomposition module for a multi-hop question answering system.
Your task: **given one user query, rewrite it as the smallest possible sequence of simpler, preferably
,→single-hop questions** that, when answered in order, are sufficient to answer the original query.
Follow these rules strictly:
1. **Use only information explicitly present in the original query.**
* Do **not** introduce new entities, definitions, or clarifying questions that the user did not ask.
* If the query says “John Phan,” do **not** ask “Who is John Phan?” because that is not needed to
,→resolve the chain.
2. **Decompose by inference steps, not by wording.**
* Each subquestion should retrieve **one missing fact** or **resolve one reference** needed by a later
,→subquestion.
* Prefer single-hop, factual questions (entity→attribute, entity→location, entity→relation).
3. **Keep only necessary subquestions.**
* If a fact is already given in the original query, do **not** restate it as a question.
* Stop decomposing when the main query becomes answerable.
4. **Preserve dependency order.**
* Later subquestions may refer to entities/answers from earlier ones.
* Number the subquestions in the order they should be executed.
5. **Output format (JSON):**
‘‘‘json
{
"original_question": "<the user question>",
"subquestions": [
{"id": 1, "question": "..."},
{"id": 2, "question": "..."},
{"id": 3, "question": "..."}
]
}
‘‘‘
6. **If the query is already single-hop, return it as one subquestion** in the same format.
**Worked example**
Input question:
> “The Argentine PGA Championship record holder has won how many tournaments worldwide?”
Decomposition:
‘‘‘json
{
"original_question": "The Argentine PGA Championship record holder has won how many tournaments
,→worldwide?",
"subquestions": [
{
"id": 1,
"question": "Who is the record holder for the Argentine PGA Championship?"
},
{
"id": 2,
"question": "How many tournaments worldwide has that person won?"
16

APREPRINT- DECEMBER19, 2025
}
]
}
‘‘‘
Notes:
* You **must not** create questions like “What is Argentine PGA Championship?” because they are not
,→required to answer the original query.
* You **must not** guess missing constraints.
* Your objective is **faithful, minimal, query-conditioned decomposition**, not general elaboration.
# Input
Listing 3: Question decomposition prompt
B Prompts Used in the SA-RAG Framework
Here, we list the exact language model prompts used at various stages of the proposed SA-RAG methodology. Listings 4
and 5 display the prompts employed in the indexing step for performing named entity recognition (NER) and relationship
extraction (RE), respectively. Both prompts instruct the LLM to produce structured responses in the form of JSON
objects and include one-shot demonstrations for the corresponding tasks.
When performing NER, the LLM is instructed to extract, in addition to entity names, the entity type, all name aliases,
and entity descriptions (stored in theentity_informationfield). For each specific entity type, the LLM is instructed to
identify certain types of descriptive information in order to avoid capturing duplicated information for multiple related
entities. For example, for Geopolitical Entities (GPE) such as cities or countries, the LLM should avoid extracting
context-dependent information and should focus only on general information (e.g., "South Africa is a country in
southern Africa" and not "South Africa is the country where Elon Musk was born," since birthplace information is
captured in the description of a person and via relationships). The extracted names and entity types are then used to
create entity nodes in the knowledge graph. Extracted descriptions form entity description nodes, aliases support entity
resolution, and identified relationships are used to create links connecting the nodes.
Listings 7 and 6 present the reasoning and answering prompts, respectively. These prompts are similar to those used in
CoT RAG, but have been adapted to instruct the LLM to take into account retrieved information about the identified key
relationships.
Given a textual paragraph, your task is to identify and extract entities from the provided text following
,→the detailed step-by-step procedure described below:
- **Identify all entities explicitly mentioned in the text.**
- Entities fall into one of four categories:
- **PERSON**: Individual people.
- **ORGANIZATION**: Companies, institutions, and organized groups.
- **GPE** (Geopolitical Entities): Countries, cities, states, regions, or territories.
- **MISC**: All other notable entities that don’t fit the above categories, such as historical
,→events, artworks, patents, buildings, etc.
- **Handle Names and Aliases:**
- If an entity has multiple mentions (aliases or abbreviations), use the **full official name** as the
,→primary reference.
- Include all the mentioned aliases in output information
- **Extract factual attributes:**
- For **PERSON** entities, extract:
- Birthplace, birth/death dates, nationality, occupation, titles, and achievements explicitly stated
,→in the text.
- For **ORGANIZATION** entities, extract:
- Foundation dates, location of headquarters, industry or sectors, associated institutions, co-
,→founders, and explicit criticisms if any.
- For **GPE** entities, extract:
- Extract explicitly stated geographical or political details, historical characteristics, capital
,→city, demographics, and cultural or administrative aspects
- **By any means do not include** information like "birthplace of...", "city of origin of...", or "
,→location where someone died" or any information that describe GPE in context dependent manner
- For **MISC** entities, extract:
- Explicitly stated temporal attributes, scale, historical significance, or explicit descriptive
,→facts provided in the text.
- **Strict adherence to text:**
- **Only** extract details explicitly mentioned in the text.
- **Do not** infer, assume, or fabricate information not present in the provided text.
17

APREPRINT- DECEMBER19, 2025
## Output Format
- Return your response as a JSON array.
- Each element in the array should be a JSON object with exactly four keys: ‘"name"‘, ‘"type"‘, ‘"aliases
,→"‘, and ‘"entity_information"‘.
# Demonstration
## Input
Nikola Tesla (born July 9/10, 1856, Smiljan, Austrian Empire [now in Croatia]–died January 7, 1943, New
,→York, New York, U.S.) was a Serbian American inventor and engineer who discovered and patented
,→the rotating magnetic field, the basis of most alternating-current machinery. He also developed
,→the three-phase system of electric power transmission. He immigrated to the United States in 1884
,→and sold the patent rights to his system of alternating-current dynamos, transformers, and
,→motors to George Westinghouse. In 1891 he invented the Tesla coil, an induction coil widely used
,→in radio technology.
## Desired output
[
{
"name": "Nikola Tesla",
"type": "PERSON",
"aliases": [],
"entity_information": "Born July 9/10, 1856, in Smiljan, Austrian Empire (now in Croatia), died
,→January 7, 1943, in New York, U.S. Serbian American inventor and engineer who discovered and
,→patented the rotating magnetic field, developed the three-phase system of electric power
,→transmission, immigrated to the United States in 1884, sold patent rights of alternating-current
,→systems to George Westinghouse, and invented the Tesla coil in 1891."
},
{
"name": "Smiljan",
"type": "GPE",
"aliases": [],
"entity_information": "Place previously in Austrian Empire, currently part of Croatia."
},
{
"name": "Austrian Empire",
"type": "GPE",
"aliases": [],
"entity_information": "Empire existing in 1856."
},
{
"name": "Croatia",
"type": "GPE",
"aliases": [],
"entity_information": "Country in Europe."
},
{
"name": "New York",
"type": "GPE",
"aliases": [],
"entity_information": "City in the United States."
},
{
"name": "United States",
"type": "GPE",
"aliases": ["U.S."],
"entity_information": "Country."
},
{
"name": "George Westinghouse",
"type": "PERSON",
"aliases": [],
"entity_information": "Person who purchased patent rights of Nikola Tesla’s system of alternating-
,→current dynamos, transformers, and motors."
},
{
"name": "Tesla coil",
"type": "MISC",
"aliases": [],
"entity_information": "Induction coil invented by Nikola Tesla in 1891, widely used in radio
,→technology."
}
]
## Forbidden outputs
[
18

APREPRINT- DECEMBER19, 2025
{
"name": "United States",
"type": "GPE",
"aliases": [],
"entity_information": "Country where Nikola Tesla immigrated to in 1884."
},
{
"name": "New York",
"type": "GPE",
"aliases": [],
"entity_information": "City in the United States, where Nikola Tesla died."
},
{
"name": "Smiljan",
"type": "GPE",
"aliases": [],
"entity_information": "Town in the Austrian Empire (now in Croatia), birthplace of Nikola Tesla."
}
]
# Real input
Listing 4: NER prompt
You are given a textual paragraph and a list of named entities. Your task is to perform relationship
,→extraction by identifying all explicit relationships mentioned in the text that connect two
,→entities from the provided list.
# Instructions:
- **Entity Pairing:** Only consider relationships between two entities from the provided named entities
,→list.
- **Triple Format:** For each relationship found, output a triple formatted as a list of three strings:
,→‘["entity1", "relationship", "entity2"]‘.
- **Pronoun Resolution:** Replace any pronouns with their corresponding specific named entities so that
,→all references are clear.
- **Output Format:** Your final answer must be a JSON dictionary containing a single key ‘"triples"‘,
,→whose value is a list of all the extracted triples.
- **Explicit Relationships:** Only include relationships that are explicitly mentioned in the text.
# Demonstration
### Input
Jack Parsons (born October 2, 1914, Los Angeles, California, U.S.–died June 17, 1952, Pasadena,
,→California) was an American rocket scientist and chemist who made significant contributions to
,→the development of rocket technology and missile systems and was a cofounder of the Jet
,→Propulsion Laboratory (JPL) at the California Institute of Technology (Caltech) and of the
,→Aerojet Engineering Corporation.
Entity list: Jack Parsons, Los Angeles, Pasadena, Jet Propulsion Laboratory, California Institute of
,→Technology, Aerojet Engineering Corporation
### Reasoning
1. **Identifing Provided Information:**
- **Input Text:** Reading the paragraph about Jack Parsons; Noting key phrases like "born in," "died
,→in," "cofounder," and the association of JPL with Caltech.
- **Entity List:** Recognize the entities: Jack Parsons, Los Angeles, Pasadena, Jet Propulsion
,→Laboratory, California Institute of Technology, Aerojet Engineering Corporation.
2. **Extracting Explicit Relationships:**
- Locating explicit statements in the text that connect two entities. For example, "Jack Parsons was
,→born in Los Angeles" and "died in Pasadena."
- Identifing that Jack Parsons "co-founded" both the Jet Propulsion Laboratory and the Aerojet
,→Engineering Corporation.
- Noting the phrase linking JPL to Caltech: "Jet Propulsion Laboratory at the California Institute of
,→Technology."
3. **Performing Pronoun Resolution:**
- Ensuring that any pronouns referring to named entities are replaced by their proper names; In this
,→case, no pronoun resolution was required since the entities are explicitly named.
### Response
{
"triples": [
["Jack Parsons", "born in", "Los Angeles"],
["Jack Parsons", "died in", "Pasadena"],
["Jack Parsons", "co-founded", "Jet Propulsion Laboratory"],
19

APREPRINT- DECEMBER19, 2025
["Jack Parsons", "co-founded", "Aerojet Engineering Corporation"],
["Jet Propulsion Laboratory", "is located at", "California Institute of Technology"]
]
}
# Real input
Listing 5: RE prompt
You are given a multi-hop question and supporting context that describes relevant facts, entities and
,→their relationships. Your task is to use the context to reason through the problem with clear,
,→detailed, and correct step-by-step logic. Pay special attention to quantitative reasoning tasks;
,→Verify every arithmetic calculation and check that the contextual details are interpreted
,→accurately.
Proceed by first outlining all necessary steps to determine the answer, including any calculations or
,→comparisons required. Once you have verified all arithmetic and double-checked key details,
,→provide only a short final answer (which may be a single word, a yes/no, or a short phrase)
,→without additional explanation.
Respond with a JSON object containing two string fields: ’reasoning’, which provides your detailed
,→reasoning about the answer, and ’final_answer’ where you provide your short final answer without
,→explaining your reasoning.
Listing 6: Answering prompt for SA-RAG
You are given a **multi-hop question** that requires combining information across multiple textual
,→paragraphs. Along with the question, you are provided:
1. **Short knowledge paragraphs**, each describing specific entities and facts.
2. A **list of key relationships** between these entities.
## Task description
Your goal is to **understand and reason through the question** using only the information provided.
To achieve this you should:
1. **Identify** only the facts and relationships that are directly relevant to answering the question.
2. **Paraphrase and weave** those facts into one concise, coherent paragraph called ‘provided_context‘.
- Write in full sentences.
- Avoid listing facts one per line.
3. Decide whether you can answer the question with the information in your summary:
- If **yes**, set ‘"answer_possible": true‘ and put your answer in ‘final_answer‘.
- If **no**, set ‘"answer_possible": false‘ and craft a specific follow-up question in ‘
,→additional_question‘ that would help fill the gap in provided information
### Output Format (JSON)
‘‘‘json
{
"provided_context": "A single narrative paragraph summarizing all relevant facts.",
"answer_possible": true | false,
"final_answer": "Give your answer here if ’answer_possible’ is true, otherwise leave this blank.",
"additional_question": "If ’answer_possible’ is false, write a clear and specific follow-up question
,→that would help get the missing information."
}
‘‘‘
# Input
Listing 7: Reasoning prompt for SA-RAG
20