# Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks

**Authors**: Martin Böckling, Heiko Paulheim, Andreea Iana

**Published**: 2025-05-22 16:11:35

**PDF URL**: [http://arxiv.org/pdf/2505.16849v1](http://arxiv.org/pdf/2505.16849v1)

## Abstract
Large Language Models (LLMs) have showcased impressive reasoning abilities,
but often suffer from hallucinations or outdated knowledge. Knowledge Graph
(KG)-based Retrieval-Augmented Generation (RAG) remedies these shortcomings by
grounding LLM responses in structured external information from a knowledge
base. However, many KG-based RAG approaches struggle with (i) aligning KG and
textual representations, (ii) balancing retrieval accuracy and efficiency, and
(iii) adapting to dynamically updated KGs. In this work, we introduce
Walk&Retrieve, a simple yet effective KG-based framework that leverages
walk-based graph traversal and knowledge verbalization for corpus generation
for zero-shot RAG. Built around efficient KG walks, our method does not require
fine-tuning on domain-specific data, enabling seamless adaptation to KG
updates, reducing computational overhead, and allowing integration with any
off-the-shelf backbone LLM. Despite its simplicity, Walk&Retrieve performs
competitively, often outperforming existing RAG systems in response accuracy
and hallucination reduction. Moreover, it demonstrates lower query latency and
robust scalability to large KGs, highlighting the potential of lightweight
retrieval strategies as strong baselines for future RAG research.

## Full Text


<!-- PDF content starts -->

arXiv:2505.16849v1  [cs.IR]  22 May 2025Walk&Retrieve: Simple Yet Effective Zero-shot
Retrieval-Augmented Generation via Knowledge Graph Walks
Martin Böckling
martin.boeckling@students.uni-
mannheim.de
University of Mannheim
Mannheim, GermanyHeiko Paulheim
heiko.paulheim@uni-mannheim.de
University of Mannheim
Mannheim, GermanyAndreea Iana
andreea.iana@uni-mannheim.de
University of Mannheim
Mannheim, Germany
Abstract
Large Language Models (LLMs) have showcased impressive rea-
soning abilities, but often suffer from hallucinations or outdated
knowledge. Knowledge Graph (KG)-based Retrieval-Augmented
Generation (RAG) remedies these shortcomings by grounding LLM
responses in structured external information from a knowledge
base. However, many KG-based RAG approaches struggle with (i)
aligning KG and textual representations, (ii) balancing retrieval
accuracy and efficiency, and (iii) adapting to dynamically updated
KGs. In this work, we introduce Walk&Retrieve , a simple yet effec-
tive KG-based framework that leverages walk-based graph traversal
and knowledge verbalization for corpus generation for zero-shot
RAG. Built around efficient KG walks, our method does not require
fine-tuning on domain-specific data, enabling seamless adaptation
to KG updates, reducing computational overhead, and allowing inte-
gration with any off-the-shelf backbone LLM. Despite its simplicity,
Walk&Retrieve performs competitively, often outperforming exist-
ing RAG systems in response accuracy and hallucination reduction.
Moreover, it demonstrates lower query latency and robust scalabil-
ity to large KGs, highlighting the potential of lightweight retrieval
strategies as strong baselines for future RAG research.
CCS Concepts
•Information systems →Information retrieval ;Language
models ;Question answering .
Keywords
Knowledge Graph Retrieval-Augmented Generation, Graph Walks,
Zero-Shot Retrieval, Question Answering
ACM Reference Format:
Martin Böckling, Heiko Paulheim, and Andreea Iana. 2025. Walk&Retrieve:
Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowl-
edge Graph Walks. In Proceedings of Information Retrieval’s Role in RAG
Systems (IR-RAG 2025). ACM, New York, NY, USA, 6 pages. https://doi.org/
XXXXXXX.XXXXXXX
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
IR-RAG 2025, Padua, Italy
©2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXX1 Introduction
Large Language Models (LLMs) are pivotal to question answering
(QA) due to their strong language understanding and text gener-
ation capabilities [ 3,27,28,35,50,59]. However, LLMs often (i)
struggle with outdated knowledge, (ii) lack interpretability due to
their black-box nature [ 7], and (iii) can hallucinate convincingly yet
factually inaccurate answers [ 18,19,40]. These issues are particu-
larly pronounced in knowledge-intensive tasks [ 31], when dealing
with domain-specific [ 47,49] or rapidly changing knowledge [ 51].
Retrieval-augmented generation (RAG) mitigates these limitations
by grounding responses in relevant external information [ 9,11,24].
Yet, text-based RAG primarily relies on semantic similarity search
of textual content [ 9], which fails to capture the relational knowl-
edge necessary to integrate passages with large semantic distance
from the query in multi-step reasoning [6, 21, 23, 30, 37].
Consequently, several works leverage knowledge graphs (KGs) –
structured knowledge bases representing real-world information
as networks of entities and relations [ 15] – as external information
sources to overcome standard RAG limitations [ 37]. Given a query,
KG-based RAG systems retrieve relevant facts as nodes, triplets,
paths, or subgraphs using graph search algorithms, or parametric
retrievers based on graph neural networks or language models
[37]. The retrieved graph data is then reformatted for the language
model – via linearized triples [ 22], natural language descriptions
[8,10,26,52,53], code-like forms [ 12], or node sequences [ 29,32,46]
– and finally used by an LLM to generates the final response [37].
The existing body of work exhibits several drawbacks. First, aug-
menting a query with relevant KG triples [ 25,42,45] can lead to
suboptimal retrieval performance due to the misalignment of struc-
tured graphs and the sequential token-based nature of the language
model. Although converting KG data to a LLM-suitable tokenized
format can help, naive triple linearization [ 2,14], which directly
converts KG triples into plain text without considering context,
coherence, or structural nuances, often produces semantically inco-
herent descriptions [ 54].1Second, RAG systems that directly reason
over KGs with LLMs perform a step-by-step graph traversal for fact
retrieval [ 21,30,46]. This requires multiple LLM calls per query,
significantly increasing complexity and latency. Third, KG-based
RAG models often fine-tune retrievers [ 13,29,53] or generators
[14,17,29,32,56] on task-specific data to better adapt to diverse
KG structures and vocabularies. However, collecting high-quality
instruction data is costly [ 4], and fine-tuning large models – even
1Given the triples: A Fistful of Dollars →writtenBy→Sergio Leone , and
The Godfather Part II →sequelOf→The Godfather , a prompt based on naive
linearization would be: These facts might be relevant to answer the question: (A Fistful
of Dollars, writtenBy, Sergio Leone), (The Godfather Part II, sequelOf, The Godfather) [...].

IR-RAG 2025, July 17, 2025, Padua, Italy Böckling et al.
Figure 1: Overview of the Walk&Retrieve framework: (1) We combine walk-based graph traversal with knowledge verbalization
for corpus generation; (2) The answer is generated with a prompt augmenting the query with the most similar verbalized walks.
with parameter-efficient methods [ 5,16,39] – is expensive and
limits generalization to dynamic KGs or unseen domains [25, 53].
Contributions. We propose Walk&Retrieve , a lightweight zero-
shot KG-based RAG framework, designed as a simple yet competi-
tive baseline to address these challenges. It combines efficient graph
traversal, via random or breadth-first search walks, with verbal-
ization of KG-derived information to build a contextual corpus
of relevant facts for each KG entity. At inference, we retrieve the
most similar nodes to the query, and their corresponding walks,
respectively. We generate the final answer by prompting an LLM
with the query, augmented with this relevant context. Unlike many
existing KG-based RAG systems, Walk&Retrieve :(1)isadaptable
to dynamic KGs – updates (e.g., node insertion or deletion) require
no retraining, as new knowledge can be added by incrementally
generating additional walks; (2)is more efficient , requiring no fine-
tuning of the backbone LLM, and only a single LLM call per query;
(3)enables zero-shot RAG with any off-the-shelf LLM . We show that
Walk&Retrieve consistently generates accurate responses, while
minimizing hallucinations. Our findings render walk-based corpus
generation as a promising approach for scalable KG-based RAG, and
establish Walk&Retrieve as a strong baseline for future research.
2 Methodology
Fig. 1 illustrates our proposed framework, comprising two stages:
corpus generation and knowledge-enhanced answer generation.
2.1 Corpus Generation
In the first stage, we leverage the knowledge stored in KGs to
construct a corpus of relevant facts. A Knowledge Graph is defined
as𝐺=(𝑉,𝐸,𝑅), where𝑉denotes a set of nodes 𝑣∈𝑉, and𝐸⊆
𝑉×𝑅×𝑉a set of directed edges labeled with relation types from
the set𝑅. For each node 𝑣∈𝑉, we define its neighbor set as 𝑁(𝑣):=
{𝑣′:∃𝑟∈𝑅|(𝑣,𝑟,𝑣′)∈𝐸}. Corpus generation consists of walk-
based graph traversal, knowledge verbalization, and indexing.
Walk-based Graph Traversal. We extract relevant facts for all
entities in the KG using two walk-based graph traversal approaches.
Random Walks (RW). In this method, we retrieve facts for a given
vertex𝑣∈𝑉by generating 𝑛𝑤graph walksW𝑙of length𝑙rooted in
𝑣. A random walk is a stochastic process with variables 𝑋0,𝑋1,𝑋2,..,where each𝑋𝑡∈𝑉denotes the vertex visited at time 𝑡[38]. At each
step, when the random walker is at vertex 𝑣𝑖, it chooses the next
node uniformly at random from one of its neighbors 𝑣𝑗∈𝑁(𝑣𝑖)
according to the following transition probability:
𝑃(𝑋𝑡+1=𝑗|𝑋𝑡=𝑖)=(1
|𝑁(𝑣𝑖)|if(𝑣𝑖,𝑟,𝑣𝑗)∈𝐸
0otherwise,(1)
where|𝑁(𝑣𝑖)|denotes the neighborhood size of 𝑣𝑖. Finally, the
graph corpus is obtained by aggregating 𝑛𝑤random walksW𝑙=
(𝑋0,𝑟𝑖,𝑋1,...,𝑟𝑘,𝑋𝑙),𝑟∈𝑅per vertex, asC𝑅𝑊=Ð|𝑉|
𝑖=1Ð𝑛𝑤𝑎𝑙𝑘𝑠
𝑗=1W𝑙.
Breadth-First Search (BFS) Walks. In this approach, we construct a
spanning tree for each entity in 𝐺using the BFS algorithm. For a
given root𝑣𝑟∈𝑉, we build walks by partitioning the reachable
nodes𝑣𝑗into layers𝐿𝑖based on their shortest-path distance to the
root [ 41]. Starting with 𝐿0={𝑣𝑟}, layers are recursively defined as
𝐿𝑖+1={𝑣𝑗∈𝑉\∪𝑖
𝑘=0𝐿𝑘:∃𝑣𝑖∈𝐿𝑖|(𝑣𝑖,𝑟,𝑣𝑗)∈𝐸} (2)
for𝑖∈[0,𝑑], where𝑑is the maximum depth (i.e., the maximum
allowed shortest-path distance). This guarantees that each vertex is
explored only once per search. Hence, the resulting corpus C𝐵𝐹𝑆=Ð|𝑉|
𝑖=1𝐿𝑖contains only non-duplicate walks for each vertex. We
note that the maximum allowed shortest-distance path 𝑑of the BFS
walks is equivalent to the length 𝑙of the randomly generated walks.
Knowledge Verbalization. As LLMs require textual inputs, we
convert the extracted walks for each entity in 𝐺into free-form
textual descriptions, to enable knowledge-enhanced reasoning for
answer generation. In contrast to recent works that fine-tune an
LLM on question-answer pairs to learn a graph-to-text transforma-
tion [ 53], we directly prompt the LLM – using the prompt template
shown in Fig. 2a – to provide a natural language representation of
the walks, obtaining the verbalized corpus. This approach aligns the
KG-derived information with the LLM’s representation space, while
preserving the order of the nodes and edges in the walks. Moreover,
by not fine-tuning the LLM, we (i) eliminate the need for labeled
graph-text pairs, (ii) improve generalization to unseen KGs, and (iii)
enable the usage of any LLM in the knowledge verbalization step.
Indexing. Lastly, we index the graph for efficient retrieval. After
knowledge verbalization, each walk 𝑤𝑣
𝑖of vertex𝑣is converted into
a vector w𝑣
𝑖. Moreover, we compute each node’s global representa-
tion from the concatenation of its respective walks. We store the

Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks IR-RAG 2025, July 17, 2025, Padua, Italy
System : Please provide me from an extracted triple set of a Knowledge Graph a sentence. The triple set consists of one extracted random walk. Therefore, a logical order of the
shown triples is present. Please consider this fact when constructing the sentence. Prevent introduction words.
Human : Please return only the constructed sentence from the following set of node and edge labels extracted from the Knowledge Graph: {triples}.
(a) Knowledge verbalization.
System : You are provided with context information from a RAG retrieval, which gives you the top k context information. Please use the provided context information to answer
the question. If you are not able to answer the question based on the context information, please return the following sentence: “I do not know the answer".
Human : Please answer the following question: {question}. Use the following context information to answer the question: {context}.
(b) Knowledge-enhanced answer generation.
Figure 2: Prompt templates used for knowledge verbalization and answer generation.
embeddings of all nodes and corresponding walks to facilitate effi-
cient retrieval during inference. Crucially, our walk-based corpus
generation renders Walk&Retrieve highly adaptable to dynamic
KGs: updates (deletions, modifications, or additions of nodes and
edges) require recomputing only the walks involving the changed
graph elements – a much smaller subset than the entire corpus.
2.2 Knowledge-enhanced Answer Generation
Given a query 𝑞, we encode it with the same LLM used for knowl-
edge verbalization, so that the query and the retrieved facts share
the same vector space. We then perform a 𝑘-nearest neighbor search
to retrieve the 𝑘most similar nodes in 𝐺to𝑞and, for each node,
the𝑘most relevant verbalized walks. Concretely, we define the
sets of relevant nodes 𝑉𝑘and corresponding walks 𝑊𝑘based on the
cosine similarity between the embeddings of the query qand each
node vor walk w𝑣, respectively. To this end, we compute:
𝑉𝑘=𝑎𝑟𝑔𝑡𝑜𝑝𝑘𝑣∈𝑉𝑐𝑜𝑠(q,v)
𝑊𝑘=Ø
𝑣𝑘∈𝑉𝑘𝑎𝑟𝑔𝑡𝑜𝑝𝑘𝑤𝑣𝑘∈C𝑐𝑜𝑠(q,w𝑣𝑘),(3)
whereC∈{C𝑅𝑊,C𝐵𝐹𝑆}, and the argtopk operation retrieves the 𝑘
nodes with the highest cosine similarity to the query. For zero-shot
inference, we design a prompt that integrates the query 𝑞with
the relevant context 𝑊𝑘, cf. template from Fig. 2b. Importantly, we
instruct the LLM to refrain from responding if the context is insuf-
ficient, thereby grounding responses in the extracted structured
knowledge, and reducing hallucinations. Finally, the prompt is fed
into the previously used LLM to generate a response. By avoiding
LLM fine-tuning, we reduce computational costs and eliminate the
need for task-specific training data. Moreover, we reduce inference
latency as Walk&Retrieve uses a single call to the LLM per query.2
3 Experimental Setup
Baselines. We compare Walk&Retrieve against three kinds of
baselines: standard LLM, text-based RAG, and KG-based RAG. With
LLM only , we test whether the LLM can answer questions without
external data. For Vanilla RAG , following [ 45], we uniformly sample
5 triples from all 1-hop facts of the question entities. We consider
two KG-RAG models. SubgraphRAG [25] retrieves subgraphs us-
ing a MLP and parallel-triple scoring; the LLM then reasons over
the linearized triples of the subgraph to generate a response. Re-
trieveRewriteAnswer [53] uses constrained path search and relation
2Note that the preprocessing step’s computational overhead is a one-time cost, as
subsequent graph changes require only incremental, inexpensive updates to the corpus.Table 1: Statistics of MetaQA [58] and CRAG [55] test sets.
MetaQACRAG
1-hop 2-hop 3-hop
# Question types 13 21 15 8
# Questions 9,947 14,872 14,274 1,335
path prediction for subgraph retrieval, which it then converts into
free-form text to augment the prompt for response generation.
Data. We conduct experiments on MetaQA [ 58] and CRAG [ 55].
MetaQA [58] is a knowledge base QA benchmark, with over 400K
questions (single- and multi-hop), and a KG containing 43K entities
and 9 relation types. We use all its 1-hop, 2-hop, and 3-hop subsets
with the "vanilla" question version. CRAG [ 55] is a factual QA
benchmark for RAG, featuring over 4.4K question-answer pairs
across five domains and eight question categories. It provides mock
KGs with 2.6 million entries.3Table 1 summarizes their statistics.
Evaluation Metrics. We follow prior work [ 25,43,44] and use
Hits@1 to measure if a response includes at least one correct entity.
Additionally, we adopt the model-based evaluation setup of Yang
et al. [55] to assess the quality of the generated answers using a
three-way scoring system: accurate (1),incorrect (-1), or missing (0).
Exact matches are labeled accurate ; all others are evaluated with two
LLMs, gpt-4-0125-preview [33] and Llama-3.1-70B-instruct
[1], to mitigate self-preference [ 36]. We report averages of accurate ,
hallucinated , and missing responses, and the overall truthfulness
(i.e., accuracy minus hallucination) from the LLM evaluators.
Implementation Details. We retrieve 𝑘=3similar nodes and
walks, respectively, for answer generation.4Our main experiments
useLlama-3.1-70B-instruct [1] with temperature 𝑡=0and spec-
ulative decoding for all models. We perform 60 walks for random-
walk corpus generation. For both Walk&Retrieve model variants,
we use walks of depth 4 on MetaQA and 3 on CRAG. We train
and evaluate the baselines using their official implementations, and
conduct all experiments on two NVIDIA A6000 48 GB GPUs.5
4 Results and Discussion
Table 1 summarizes the QA performance of Walk&Retrieve and
the baselines with Llama-3.1 . On MetaQA, Walk&Retrieve-BFS
consistently outperforms all other models in answer accuracy and
3In our experiments, we use the public test set of CRAG.
4In preliminary experiments with 𝑘∈[1,5], we found𝑘=3to be the optimal value
that balances accuracy and hallucination.
5Code available at https://github.com/MartinBoeckling/KGRag

IR-RAG 2025, July 17, 2025, Padua, Italy Böckling et al.
Table 2: Question-answering performance. We report numbers in percentage, and the query runtime in seconds. For MetaQA,
we average results over its k-hop subsets. The best results per column are highlighted in bold, the second best underlined.
MetaQA CRAG
Baseline Type Model Hits@1↑Accuracy↑Hallucination↓Missing↓Time (s)↓Hits@1↑Accuracy↑Hallucination↓Missing↓Time (s)↓
LLM only Direct 30.37 31.79 18.86 61.89 13.03 11.05 9.31 23.95 67.49 14.14
Text-based RAG Vanilla RAG 25.08 14.73 13.52 65.70 22.11 15.21 16.94 19.53 51.39 26.01
KG-based RAGSubgraphRAG 43.88 41.17 18.08 32.53 23.12 – – – – –
RetrieveRewriteAnswer 47.49 34.01 22.92 32.12 22.37 – – – – –
Walk&Retrieve-RW 55.60 41.11 15.31 37.13 22.12 19.31 19.40 19.64 51.94 22.15
Walk&Retrieve-BFS 67.99 57.08 12.74 28.27 21.31 21.31 21.53 23.01 53.40 23.34
20 30 40 50 60 70 80
Missing (%)10
0102030405060Truthfulness (%)Walk&Retrieve-RW 1-hop
Walk&Retrieve-BFS 1-hop
RewriteRetrieveAnswer 1-hop2-hop
2-hop
2-hop3-hop
3-hop
3-hopSubgraphRAG 1-hop
VanillaRAG 1-hop
LLM only 1-hop2-hop
2-hop
2-hop3-hop
3-hop
3-hop
Figure 3: Missing vs. truthfulness rates over MetaQA subsets.
Hits@1, achieving a relative improvement of 38.64% over the best
baseline (SubgraphRAG). While other KG-based RAG systems yield
high accuracy, they tend to hallucinate more than the simpler LLM-
only and Vanilla RAG systems, which often produce no answer
rather than an incorrect one. In contrast, Walk&Retrieve-BFS min-
imizes both hallucinations and missing responses. Although LLM-
only has the lowest query latency due to the absence of a retrieval
step, Walk&Retrieve achieves the fastest inference time per query
among all RAG approaches, underscoring its efficiency. Fig. 3 breaks
down MetaQA performance by number of hops. LLM-only and
Vanilla RAG fail to answer over 60% of 2- and 3-hop questions.
Both SubgraphRAG and RetrieveRewriteAnswer lower the missing
rate below 35% across hops, although truthfulness remains under
25%. Conversely, Walk&Retrieve-BFS better trades off accuracy
and hallucination (55%+ truthfulness for 1-hop and 37%+ for 2- and
3-hop questions), while greatly reducing non-responses.
On CRAG, both Walk&Retrieve variants outperform LLM-only
and Vanilla RAG in answer accuracy, while matching them in hal-
lucination and missing rates. Note that, SubgraphRAG and Retriev-
eRewriteAnswer could not be evaluated on CRAG due to scalability
and computational constraints.6These results highlight the scala-
bility of our walk-based corpus generation approach, which limits
traversal to small-hop neighborhoods rather than the full graph.
While performance drops on CRAG, likely due to its greater com-
plexity (i.e., MetaQA expects only entity answers) and focus on
holistic RAG performance, Walk&Retrieve remains robust. Even
though the findings are promising, we plan to further evaluate
Walk&Retrieve on larger KGs and other challenging benchmarks
(e.g., WebQSP [57], CWQ [48]) to fully showcase its capabilities.
Ablation of Walk Approach. The graph traversal strategy and
its hyperparameters define a node’s relevant context, directly im-
pacting corpus quality and, consequently, retrieval accuracy in
6SubgraphRAG fails to scale to CRAG’s KG (over 1 million edges), and Retriev-
eRewriteAnswer requires fine-tuning the backbone LLM beyond our available
resources.
1 2 3 4 5 6
Walk Depth25
15
5
51525354555Truthfulness (%)Walk&Retrieve-BFS 1-hop
Walk&Retrieve-RW 1-hop2-hop
2-hop3-hop
3-hop
Mixtral 8x70B GPT-4o LlaMa 3.1 70B
LLM0102030405060Truthfulness (%)Walk&Retrieve-BFS 1-hop
Walk&Retrieve-RW 1-hop2-hop
2-hop3-hop
3-hopWalk Approaches & Settings Backbone LLMsFigure 4: Truthfulness rates for different (i) walk approaches
and (ii) backbone LLMs, over the MetaQA subsets.
RAG systems. The left graph in Fig. 4 shows MetaQA results for
Walk&Retrieve with walk depths ranging from 1 to 6.7We find
that a walk depth of 4 offers the best trade-off between answer
accuracy and hallucination. Notably, regardless of walk length,
Walk&Retrieve-BFS consistently yields higher truthfulness than
Walk&Retrieve-RW , likely due to is systematic graph exploration,
which avoids duplicate walks (cf. §2). In contrast, random walks
tend to produce noisier context and fewer unique paths, thus cap-
turing less relevant information from the KG.8While they may
be more efficient on large-scale KGs, as they do not compute full
neighborhoods, this efficiency comes at the cost of increased noise.9
Robustness to Backbone LLMs. Lastly, we evaluate model ro-
bustness using different LLMs (see right graph of Fig. 4), including
Mixtral-8x7B-Instruct [20] and GPT-4o [34].Mixtral improves
answer truthfulness over Llama-3.1 on 2- and 3-hop questions,
while GPT-4o yields the highest truthfulness across all types of
questions. The RW approach exhibits considerably higher variance
across LLMs compared to the BFS-based model, which we attribute
to the noisier and less relevant information in its generated corpus.
5 Conclusion
Current KG-based RAG faces challenges in aligning structured and
textual representations, balancing accuracy with efficiency, and
adapting to dynamic KGs. We proposed Walk&Retrieve , a simple
yet effective KG-based framework for zero-shot RAG. It leverages
walk-based graph traversal and LLM-driven knowledge verbaliza-
tion for corpus generation. At inference time, the LLM is prompted
7ForWalk&Retrieve-RW , we also ablate 𝑛𝑤∈[10,100](step of 10); for brevity, we
report results for 𝑛𝑤=60, as other values perform comparably.
8On average, each node yields 60 duplicated and 8.74 unique random walks, whereas
BFS generates 9.41 unique walks. Although RW could be modified to avoid duplicates,
our current setup spans the full spectrum from randomness (RW) to structure (BFS).
9The time complexity of BFS is O(|𝑉|+|𝐸|), whereas that of RW varies between
O(|𝑉|log|𝑉|)andO(|𝑉|3).

Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks IR-RAG 2025, July 17, 2025, Padua, Italy
with the query augmented by relevant verbalized walks for en-
hanced reasoning. Its efficient retrieval mechanism supports seam-
less adaptation to evolving KGs through incremental generation
of new walks. Walk&Retrieve is compatible with any off-the-shelf
LLM, and reduces computational overhead by avoiding fine-tuning
of the backbone LLM. Despite its simplicity, Walk&Retrieve out-
performs existing RAG approaches in answer accuracy and in the
reduction of hallucinated or missing responses, while maintaining
low query latency. Our results highlight walk-based corpus gener-
ation as a promising strategy for scaling to large-size KGs. These
findings establish Walk&Retrieve as a simple, yet strong baseline
for KG-based RAG, and we hope they inspire further research into
adaptable and scalable RAG systems.
References
[1]AI@Meta. 2024. Llama 3 Model Card. (2024). https://github.com/meta-llama/
llama3/blob/main/MODEL_CARD.md
[2]Jinheon Baek, Alham Fikri Aji, and Amir Saffari. 2023. Knowledge-Augmented
Language Model Prompting for Zero-Shot Knowledge Graph Question Answer-
ing. In Proceedings of the 1st Workshop on Natural Language Reasoning and Struc-
tured Explanations (NLRSE) . 78–106. doi:10.18653/v1/2023.nlrse-1.7
[3]Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al .2020. Language models are few-shot learners. Advances in neural
information processing systems 33 (2020), 1877–1901.
[4]Yihan Cao, Yanbin Kang, Chi Wang, and Lichao Sun. 2023. Instruction Mining:
Instruction Data Selection for Tuning Large Language Models. arXiv preprint
arXiv:2307.06290 (2023). doi:10.48550/arXiv.2307.06290
[5]Ziwei Chai, Tianjie Zhang, Liang Wu, Kaiqiao Han, Xiaohai Hu, Xuanwen Huang,
and Yang Yang. 2023. Graphllm: Boosting graph reasoning ability of large lan-
guage model. arXiv preprint arXiv:2310.05845 (2023). doi:10.48550/arXiv.2310.
05845
[6]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large
language models in retrieval-augmented generation. In Proceedings of the AAAI
Conference on Artificial Intelligence , Vol. 38. 17754–17762. doi:10.1609/aaai.v38i16.
29728
[7]Marina Danilevsky, Kun Qian, Ranit Aharonov, Yannis Katsis, Ban Kawas, and
Prithviraj Sen. 2020. A Survey of the State of Explainable AI for Natural Language
Processing. In Proceedings of the 1st Conference of the Asia-Pacific Chapter of
the Association for Computational Linguistics and the 10th International Joint
Conference on Natural Language Processing . 447–459. doi:10.18653/v1/2020.aacl-
main.46
[8]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, and Jonathan Larson. 2024. From local to global: A graph
rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130
(2024). doi:10.48550/arXiv.2404.16130
[9]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. In Proceedings of the 30th ACM
SIGKDD Conference on Knowledge Discovery and Data Mining . 6491–6501. doi:10.
1145/3637528.3671470
[10] Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. 2024. Talk like a Graph:
Encoding Graphs for Large Language Models. In The Twelfth International Confer-
ence on Learning Representations . https://openreview.net/forum?id=IuXR1CCrSi
[11] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai,
Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large
language models: A survey. arXiv preprint arXiv:2312.10997 (2023). doi:10.48550/
arXiv.2312.10997
[12] Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi He, and Shi Han. 2023.
Gpt4graph: Can large language models understand graph structured data? an
empirical evaluation and benchmarking. arXiv preprint arXiv:2305.15066 (2023).
doi:10.48550/arXiv.2305.15066
[13] Tiezheng Guo, Qingwen Yang, Chen Wang, Yanyi Liu, Pan Li, Jiawei Tang, Dapeng
Li, and Yingyou Wen. 2024. Knowledgenavigator: Leveraging large language
models for enhanced reasoning over knowledge graph. Complex & Intelligent
Systems 10, 5 (2024), 7063–7076.
[14] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V Chawla, Thomas Laurent, Yann
LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented
generation for textual graph understanding and question answering. arXiv
preprint arXiv:2402.07630 (2024). doi:10.48550/arXiv.2402.07630
[15] Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia d’Amato, Gerard De Melo,
Claudio Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli,Sebastian Neumaier, et al .2021. Knowledge graphs. ACM Computing Surveys
(Csur) 54, 4 (2021), 1–37.
[16] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu
Wang, Weizhu Chen, et al .2022. LoRA: Low-Rank Adaptation of Large Language
Models. In International Conference on Learning Representations .
[17] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. 2024.
GRAG: Graph Retrieval-Augmented Generation. arXiv preprint arXiv:2405.16506
(2024). doi:10.48550/arXiv.2405.16506
[18] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian
Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al .2024.
A Survey on Hallucination in Large Language Models: Principles, Taxonomy,
Challenges, and Open Questions. ACM Transactions on Information Systems
(2024). doi:10.1145/3703155
[19] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii,
Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination in
Natural Language Generation. Comput. Surveys 55, 12 (2023), 1–38. doi:10.1145/
3571730
[20] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, De-
vendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel,
Guillaume Lample, Lucile Saulnier, et al .2023. Mistral 7B. arXiv preprint
arXiv:2310.06825 (2023). doi:10.48550/arXiv.2310.06825
[21] Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Kumar Roy, Yu Zhang, Zheng Li,
Ruirui Li, Xianfeng Tang, Suhang Wang, Yu Meng, et al .2024. Graph Chain-of-
Thought: Augmenting Large Language Models by Reasoning on Graphs. arXiv
preprint arXiv:2404.07103 (2024). doi:10.48550/arXiv.2404.07103
[22] Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi. 2023. KG-GPT: A General
Framework for Reasoning on Knowledge Graphs Using Large Language Models.
InFindings of the Association for Computational Linguistics: EMNLP 2023 . 9410–
9421. doi:10.18653/v1/2023.findings-emnlp.631
[23] Jonathan Larson and Steven Truitt. 2024. GraphRAG: Unlocking
LLM discovery on narrative private data . Retrieved 2025-01-27 from
https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-
discovery-on-narrative-private-data/
[24] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in Neural Information Processing Systems 33 (2020), 9459–9474.
[25] Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is effective: The roles of graphs
and large language models in knowledge-graph-based retrieval-augmented
generation. In International Conference on Learning Representations . https:
//openreview.net/pdf?id=JvkuZZ04O7
[26] Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin, Zheng Li, Xifeng Yan, Chao
Zhang, and Bing Yin. 2023. Graph Reasoning for Question Answering with
Triplet Retrieval. In Findings of the Association for Computational Linguistics: ACL
2023. 3366–3375. doi:10.18653/v1/2023.findings-acl.208
[27] Valentin Liévin, Christoffer Egeberg Hother, Andreas Geert Motzfeldt, and Ole
Winther. 2024. Can large language models reason about medical questions?
Patterns 5, 3 (2024).
[28] Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and
Graham Neubig. 2023. Pre-train, prompt, and predict: A systematic survey of
prompting methods in natural language processing. Comput. Surveys 55, 9 (2023),
1–35. doi:10.1145/3560815
[29] Linhao Luo, Yuan-Fang Li, Reza Haf, and Shirui Pan. 2024. Reasoning on Graphs:
Faithful and Interpretable Large Language Model Reasoning. In The Twelfth
International Conference on Learning Representations .
[30] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2024. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation.
arXiv preprint arXiv:2407.10805 (2024). doi:10.48550/arXiv.2407.10805
[31] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and
Hannaneh Hajishirzi. 2023. When Not to Trust Language Models: Investigating
Effectiveness of Parametric and Non-Parametric Memories. In Proceedings of the
61st Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers) . 9802–9822. doi:10.18653/v1/2023.acl-long.546
[32] Costas Mavromatis and George Karypis. 2024. GNN-RAG: Graph Neural Retrieval
for Large Language Model Reasoning. arXiv preprint arXiv:2405.20139 (2024).
doi:10.48550/arXiv.2405.20139
[33] OpenAI. 2023. ChatGPT . Retrieved 2025-02-14 from https://openai.com/index/
chatgpt/
[34] OpenAI. 2023. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774 (2023).
doi:10.48550/arXiv.2303.08774
[35] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al .2022.
Training language models to follow instructions with human feedback. Advances
in neural information processing systems 35 (2022), 27730–27744.
[36] Arjun Panickssery, Samuel R Bowman, and Shi Feng. 2024. Llm evaluators
recognize and favor their own generations. arXiv preprint arXiv:2404.13076
(2024). doi:10.48550/arXiv.2404.13076

IR-RAG 2025, July 17, 2025, Padua, Italy Böckling et al.
[37] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2024. Graph retrieval-augmented generation: A survey.
arXiv preprint arXiv:2408.08921 (2024).
[38] Bryan Perozzi, Rami Al-Rfou, and Steven Skiena. 2014. Deepwalk: Online learning
of social representations. In Proceedings of the 20th ACM SIGKDD international
conference on Knowledge discovery and data mining . 701–710. doi:10.1145/2623330.
2623732
[39] Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi,
Rami Al-Rfou, and Jonathan Halcrow. 2024. Let your graph do the talking:
Encoding structured data for llms. arXiv preprint arXiv:2402.05862 (2024).
[40] Vipula Rawte, Swagata Chakraborty, Agnibh Pathak, Anubhav Sarkar,
SM Towhidul Islam Tonmoy, Aman Chadha, Amit Sheth, and Amitava Das.
2023. The Troubling Emergence of Hallucination in Large Language Models-An
Extensive Definition, Quantification, and Prescriptive Remediations. In Proceed-
ings of the 2023 Conference on Empirical Methods in Natural Language Processing .
2541–2573. doi:10.18653/v1/2023.emnlp-main.155
[41] Petar Ristoski and Heiko Paulheim. 2016. Rdf2vec: Rdf graph embeddings for
data mining. In International semantic web conference . Springer, 498–514.
[42] Ahmmad O. M. Saleh, Gokhan Tur, and Yucel Saygin. 2024. SG-RAG: Multi-
Hop Question Answering With Large Language Models Through Knowledge
Graphs. In Proceedings of the 7th International Conference on Natural Language
and Speech Processing (ICNLSP 2024) , Mourad Abbas and Abed Alhakim Freihat
(Eds.). Association for Computational Linguistics, Trento, 439–448. https://
aclanthology.org/2024.icnlsp-1.45/
[43] Apoorv Saxena, Adrian Kochsiek, and Rainer Gemulla. 2022. Sequence-to-
Sequence Knowledge Graph Completion and Question Answering. In Proceedings
of the 60th Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers) . 2814–2828. doi:10.18653/v1/2022.acl-long.201
[44] Priyanka Sen, Alham Fikri Aji, and Amir Saffari. 2022. Mintaka: A Complex, Natu-
ral, and Multilingual Dataset for End-to-End Question Answering. In Proceedings
of the 29th International Conference on Computational Linguistics . 1604–1619.
[45] Priyanka Sen, Sandeep Mavadia, and Amir Saffari. 2023. Knowledge graph-
augmented language models for complex question answering. In Proceedings of
the 1st Workshop on Natural Language Reasoning and Structured Explanations
(NLRSE) . 1–8. doi:10.18653/v1/2023.nlrse-1.1
[46] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph: Deep
and Responsible Reasoning of Large Language Model on Knowledge Graph. In
The Twelfth International Conference on Learning Representations .
[47] Kai Sun, Yifan Xu, Hanwen Zha, Yue Liu, and Xin Luna Dong. 2024. Head-
to-Tail: How Knowledgeable are Large Language Models (LLMs)? AKA Will
LLMs Replace Knowledge Graphs?. In Proceedings of the 2024 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers) . 311–325. doi:10.18653/v1/2024.
naacl-long.18
[48] Alon Talmor and Jonathan Berant. 2018. The Web as a Knowledge-Base for
Answering Complex Questions. In Proceedings of the 2018 Conference of the
North American Chapter of the Association for Computational Linguistics: HumanLanguage Technologies, Volume 1 (Long Papers) . 641–651. doi:10.18653/v1/N18-
1059
[49] SM Tonmoy, SM Zaman, Vinija Jain, Anku Rani, Vipula Rawte, Aman Chadha,
and Amitava Das. 2024. A comprehensive survey of hallucination mitigation
techniques in large language models. arXiv preprint arXiv:2401.01313 (2024).
doi:10.48550/arXiv.2401.01313
[50] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yas-
mine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhos-
ale, et al .2023. Llama 2: Open foundation and fine-tuned chat models. arXiv
preprint arXiv:2307.09288 (2023). doi:10.48550/arXiv.2307.09288
[51] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar,
Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong. 2024. FreshLLMs: Re-
freshing Large Language Models with Search Engine Augmentation. In Findings
of the Association for Computational Linguistics: ACL 2024 , Lun-Wei Ku, Andre
Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics,
Bangkok, Thailand, 13697–13720. doi:10.18653/v1/2024.findings-acl.813
[52] Yilin Wen, Zifeng Wang, and Jimeng Sun. 2023. Mindmap: Knowledge graph
prompting sparks graph of thoughts in large language models. arXiv preprint
arXiv:2308.09729 (2023). doi:10.48550/arXiv.2308.09729
[53] Yike Wu, Nan Hu, Sheng Bi, Guilin Qi, Jie Ren, Anhuan Xie, and Wei Song. 2023.
Retrieve-rewrite-answer: A kg-to-text enhanced llms framework for knowledge
graph question answering. arXiv preprint arXiv:2309.11206 (2023). doi:10.48550/
arXiv.2309.11206
[54] Yike Wu, Yi Huang, Nan Hu, Yuncheng Hua, Guilin Qi, Jiaoyan Chen, and
Jeff Pan. 2024. CoTKR: Chain-of-Thought Enhanced Knowledge Rewriting for
Complex Knowledge Graph Question Answering. In Proceedings of the 2024
Conference on Empirical Methods in Natural Language Processing . 3501–3520.
doi:10.18653/v1/2024.emnlp-main.205
[55] Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sa-
jal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, et al .2024.
CRAG–Comprehensive RAG Benchmark. 38th Conference on Neural Information
Processing Systems (NeurIPS 2024), Track on Datasets and Benchmarks (2024).
[56] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure
Leskovec. 2021. QA-GNN: Reasoning with Language Models and Knowledge
Graphs for Question Answering. In Proceedings of the 2021 Conference of the
North American Chapter of the Association for Computational Linguistics: Human
Language Technologies . 535–546. doi:10.18653/v1/2021.naacl-main.45
[57] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and
Jina Suh. 2016. The value of semantic parse labeling for knowledge base question
answering. In Proceedings of the 54th Annual Meeting of the Association for Com-
putational Linguistics (Volume 2: Short Papers) . 201–206. doi:10.18653/v1/P16-2033
[58] Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander Smola, and Le Song.
2018. Variational reasoning for question answering with knowledge graph. In
Proceedings of the AAAI conference on artificial intelligence , Vol. 32.
[59] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou,
Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al .2023. A survey
of large language models. arXiv preprint arXiv:2303.18223 (2023). doi:10.48550/
arXiv.2303.18223