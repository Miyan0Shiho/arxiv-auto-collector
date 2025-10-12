# Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs

**Authors**: Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, Jiliang Tang

**Published**: 2025-10-08 19:29:19

**PDF URL**: [http://arxiv.org/pdf/2510.07484v1](http://arxiv.org/pdf/2510.07484v1)

## Abstract
Reasoning over structured graphs remains a fundamental challenge for Large
Language Models (LLMs), particularly when scaling to large graphs. Existing
approaches typically follow the retrieval-augmented generation (RAG) paradigm:
first retrieving subgraphs relevant to the query and then generating answers
conditioned on the retrieved subgraphs. However, such two-phase pipelines often
struggle to faithfully incorporate graph structure, since the generation
process is ultimately constrained by the quality and completeness of the
retrieved subgraph. Although many advanced retrievers have been proposed
recently to mitigate this issue, they are usually tailored to the training
graphs and generalize poorly to unseen graphs, which limits their practical
applicability. In this work, we propose Reasoning by Exploration (RoE), a novel
approach that unifies retrieval and generation by framing reasoning over graphs
as a process of graph exploration. At each step, the LLM selects candidate
nodes and edges to explore, gradually constructing reasoning paths and
generating answers along the way. To enable effective exploration, RoE is
trained in two stages: supervised fine-tuning (SFT) on gold reasoning paths,
followed by reinforcement learning (RL) to enhance exploration effectiveness
and generalization. Experiments on benchmark datasets demonstrate that RoE
achieves substantial overall improvements over baselines, while also
generalizing effectively to unseen graphs.

## Full Text


<!-- PDF content starts -->

Reasoning by Exploration: A Unified Approach to Retrieval and
Generation over Graphs
Haoyu Han
Michigan State University
USA
hanhaoy1@msu.eduKai Guo
Michigan State University
USA
guokai1@msu.eduHarry Shomer
University of Texas at Arlington
USA
harry.shomer@uta.edu
Yu Wang
University of Oregon
USA
yuwang@uoregon.eduYucheng Chu
Michigan State University
USA
chuyuch2@msu.eduHang Li
Michigan State University
USA
lihang4@msu.edu
Li Ma
Michigan State University
USA
mali13@msu.eduJiliang Tang
Michigan State University
USA
tangjili@msu.edu
Abstract
Reasoning over structured graphs remains a fundamental challenge
for Large Language Models (LLMs), particularly when scaling to
large graphs. Existing approaches typically follow the retrieval-
augmented generation (RAG) paradigm: first retrieving subgraphs
relevant to the query and then generating answers conditioned
on the retrieved subgraphs. However, such two-phase pipelines
often struggle to faithfully incorporate graph structure, since the
generation process is ultimately constrained by the quality and
completeness of the retrieved subgraph. Although many advanced
retrievers have been proposed recently to mitigate this issue, they
are usually tailored to the training graphs and generalize poorly
to unseen graphs, which limits their practical applicability. In this
work, we propose Reasoning by Exploration (RoE), a novel ap-
proach that unifies retrieval and generation by framing reasoning
over graphs as a process of graph exploration. At each step, the
LLM selects candidate nodes and edges to explore, gradually con-
structing reasoning paths and generating answers along the way.
To enable effective exploration, RoE is trained in two stages: su-
pervised fine-tuning (SFT) on gold reasoning paths, followed by
reinforcement learning (RL) to enhance exploration effectiveness
and generalization. Experiments on benchmark datasets demon-
strate that RoE achieves substantial overall improvements over
baselines, while also generalizing effectively to unseen graphs.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM
https://doi.org/XXXXXXX.XXXXXXXKeywords
Large Language Models, Graph Reasoning, Retrieval-Augmented
Generation, Multi-hop Question Answering
ACM Reference Format:
Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li,
Li Ma, and Jiliang Tang. 2018. Reasoning by Exploration: A Unified Ap-
proach to Retrieval and Generation over Graphs. InProceedings of Make
sure to enter the correct conference title from your rights confirmation email
(Conference acronym ’XX).ACM, New York, NY, USA, 12 pages. https:
//doi.org/XXXXXXX.XXXXXXX
1 Introduction
Large language models (LLMs) [ 5,24,49] have demonstrated im-
pressive abilities across a wide range of natural language processing
tasks. To further enhance their reliability and factuality, Retrieval-
Augmented Generation (RAG) [ 14,23,26] has become a widely
adopted paradigm. By retrieving external knowledge before gener-
ating responses, RAG has achieved strong performance in diverse
domains, such as healthcare [ 60], law [ 55], finance [ 65], and ed-
ucation [ 7]. However, most existing RAG systems operate over
unstructured text, retrieving passages through lexical or seman-
tic similarity, and therefore do not fully leverage the relational
structure inherent in many forms of knowledge.
Graphs provide a natural representation for structured knowl-
edge, capturing rich relational dependencies among entities. They
are widely used in domains such as knowledge bases, social net-
works, and scientific applications including biology and chem-
istry [ 32,56,58]. Recently, Graph Retrieval-Augmented Generation
(GraphRAG) has emerged as a promising way to integrate such
graph-structured knowledge into LLMs [17, 37]. There are mainly
two steps in GraphRAG: first, a retriever selects query-relevant con-
tent from the entire graph, where the retrieved units may be nodes,
paths, or subgraphs; then, a generator, usually an LLM, produces
answers conditioned on the retrieved subgraphs. This framework
has been successfully applied to tasks such as knowledge grapharXiv:2510.07484v1  [cs.IR]  8 Oct 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, and Jiliang Tang
question answering (KGQA) [ 61,62,68], where answering a query
often requires multi-step reasoning over the graphs.
While efficient, this two-phase pipeline may constrain reasoning
quality due to the disjoint optimization of retriever and generator.
First, the retriever often ignores the specific needs of the generator,
which can lead to missing critical evidence or introducing irrelevant
noise. For example, consider the multi-hop question “Who are the
nephews of Winston Churchill?”. This query may have multiple
answers and can be reached through multiple reasoning paths, such
as “Churchill→sibling→child" or “Churchill →parent→child
→child". Heuristic-based retrievers [ 13,61] (e.g., retrieving fixed
𝑘-hop subgraphs) or GNN-based retrievers [ 28,33] may include
a large amount of irrelevant information about Churchill, over-
whelming the generator with noise [ 16]. In contrast, LLM-based
retrievers [ 31,57] may generate only a limited set of reasoning
paths, leading to missing valid entities and incomplete answers,
as shown in Section 3.Second, retrievers or generators trained
on specific graphs or datasets usually generalize poorly to unseen
graphs as shown in Section 3. For instance, a path-based retriever
tuned on a movie knowledge base may learn to generate reasoning
paths primarily around relations such as “actor →film→director",
but it struggles to generalize when applied to graphs with very
different relational structures, such as biomedical graphs involving
“gene→protein→disease" paths. This dependence leads to sharp
performance drops when deployed on unseen graphs, limiting the
robustness and real-world applicability of current GraphRAGs.
To address these limitations, we propose Reasoning by Explo-
ration (RoE), a unified framework that leverages the reasoning
ability of LLMs to perform retrieval and generation simultane-
ously within the same model. Instead of separating the two stages,
RoEframes reasoning as a step-by-step graph exploration process,
where the model incrementally expands nodes and edges while
constructing reasoning paths and generating answers in a unified
manner. In this way,RoEhas the access to the whole graph. This
design directly mitigates the problem of missing or noisy retrieval,
since the model actively decides which parts of the whole graph to
explore based on its evolving reasoning process, instead of relying
on a static retriever. However, it faces two main challenges.First,
how can we equip LLMs, which pretrained primarily on unstruc-
tured text, with the ability to perform structured graph exploration?
Second, how can we ensure that the model learns generalizable ex-
ploration strategies rather than memorizing specific graph patterns,
so that it can adapt to unseen graphs? To tackle these challenges, we
adopt a two-stage training strategy. In the first stage, we construct
gold exploration trajectories and apply supervised fine-tuning (SFT)
to teach the LLM to expand nodes and edges step by step. In the
second stage, we employ reinforcement learning (RL) with different
reward signals to further refine the exploration policy, encouraging
the model to efficiently discover valid reasoning paths while improv-
ing its ability to generalize across diverse and unseen graphs. This
combination enablesRoEto leverage both explicit supervision and
adaptive learning signals, resulting in more faithful reasoning and
stronger generalization to unseen graphs. Our key contributions
are summarized as follows:
•We introduce Reasoning by Exploration (RoE), a unified frame-
work that integrates retrieval and generation into a single processby framing reasoning as graph exploration, allowing LLMs to
construct reasoning paths and answers simultaneously.
•We develop a two-stage training strategy that equips LLMs with
exploration ability: supervised fine-tuning (SFT) on gold rea-
soning trajectories to learn step-wise expansion, followed by
reinforcement learning (RL) to refine exploration policies and
enhance generalization to unseen graphs.
•We conduct extensive experiments on multi-hop reasoning bench-
marks, demonstrating thatRoEsignificantly outperforms state-
of-the-art baselines while exhibiting strong generalization across
diverse unseen graphs.
2 Related Works
2.1 GraphRAG
GraphRAG [ 17,37] aims to retrieve relevant information from exter-
nal graphs to improve the performance and reduce the hallucination
of LLMs. Compared to traditional RAG systems [ 14,26] that operate
on unstructured text, GraphRAG leverages the relational structure
of graphs, where nodes represent entities and edges encode rela-
tions between nodes. Similar to RAG, GraphRAG generally consists
of two main components: a retriever, which selects query-relevant
graph content, and a generator, typically an LLM, which produces
answers based on the retrieved subgraphs. To accommodate the
graph structure, there are mainly three types of retrievers:
Heuristic-based Retriever.Heuristic-based retrievers typically
begin by using similarity-based methods [ 39,54] or entity and rela-
tion extraction techniques [ 2,13] to identify seed nodes and edges
that are relevant to the query. These seeds are then expanded by
retrieving their 𝑘-hop neighbors [ 6,62], forming a candidate sub-
graph for reasoning. A representative example is G-Retriever [ 19],
which first leverages embedding models to retrieve the most similar
nodes and edges to the query, and then applies the Prize-Collecting
Steiner Tree (PCST) algorithm to extract a compact subgraph cen-
tered around the selected seeds.
GNN-based Retriever.GNN-based retrievers [ 12,28] typically
train Graph Neural Networks (GNNs) [ 32,58] to model retrieval
as a node classification task, where the correct answer entities
are labeled as positive and others as negative. For example, GNN-
RAG [ 33] first trains a GNN model and then retrieves the nodes
with the highest probability scores as candidate answers, while also
identifying the shortest paths connecting the question entities with
these candidates to construct supporting evidence. Compared to
heuristic-based methods, GNN-based retrievers can capture richer
relational dependencies and are more adaptive to complex queries.
LLM-based Retriever.LLM-based retrievers leverage the rea-
soning ability of LLMs to generate retrieval plans instead of relying
solely on graph embeddings or heuristics. For example, they may
prompt an LLM to produce SQL queries over structured knowledge
bases [ 21] or to explicitly plan multi-hop paths from the query en-
tities to answers [ 72]. A representative method is RoG [ 31], which
fine-tunes an LLM to generate reasoning relation paths and then
retrieves the corresponding paths from the graph.
While these retrievers have demonstrated success across differ-
ent tasks, they remain unaware of the generation process, which
often leads to incomplete retrieval and the introduction of substan-
tial noise as shown in Section 3. Moreover, training-based methods

Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
are usually tailored to a specific graph, further limiting their ability
to generalize to unseen graphs. Instead, the proposedRoEunifies re-
trieval and generation within the same process, enabling the model
to actively explore the graph while constructing reasoning paths
and generating answers in an integrated manner.
2.2 Reasoning of Large Language Models
Large Language Models (LLMs) have demonstrated strong perfor-
mance on natural language tasks [ 5,24,49], but they may struggle
with more complex reasoning problems [ 35,70]. To further improve
their reasoning ability, early works leverage prompting-based meth-
ods, such as Chain-of-Thought [ 53], and Graph-of-Thought [ 3],
which encourage models to generate and organize intermediate rea-
soning steps during inference. Think-on-graph (ToG) [ 45] also fol-
low this paradigm to prompt LLM reason on the graph. While effec-
tive, these methods are often sensitive to prompt design and compu-
tationally inefficient. Building on this, more recent approaches [ 42,
59] focus on supervised fine-tuning (SFT) with curated reasoning
traces [ 10,29] and reinforcement learning (RL) with process or out-
come rewards [ 38,66]. These methods allow models to internalize
reasoning strategies rather than relying only on inference-time
prompting. However, most of these advances focus on text-only
tasks, whereas reasoning over graph-structured knowledge intro-
duces additional challenges, requiring models to explore nodes,
edges, and multi-hop paths. Although several works [ 30,51,64]
have attempted to adapt LLMs for graph reasoning tasks, most
of them explicitly decompose complex queries into simpler sub-
queries and rely on external retrievers. It remains underexplored
how to enable LLMs to automatically explore graphs and generate
answers in a unified manner.
In this paper, we follow previous studies [ 19,31,45] and leverage
the Knowledge Graph Question Answering (KGQA) task to evalu-
ate the effectiveness of graph reasoning, as it inherently involves
multi-hop relational reasoning and explicit interaction with graph
structures to obtain correct answers.
3 Preliminary
In this section, we first provide definitions of the key concepts and
tasks used throughout the paper. We then analyze the retrievers
adopted in current popular GraphRAG methods.
3.1 Definitions
Knowledge Graphs (KGs).Let G={V,R,E} denote a knowl-
edge graph, where Vis the set of entities (nodes), Ris the set of
relations, andEis the set of edges. Each edge 𝑒∈E represents a
directed link between the head and tail entities ℎ,𝑡∈V under a
relation𝑟∈𝑅 , and can be denoted as a triple (ℎ,𝑟,𝑡) . For a given
node𝑣∈V , we useN𝑣to denote its neighboring entities, R𝑣to
denote the associated relations and E𝑣to denote the connected
edges of node𝑣.
In this paper, we also augment the knowledge graph with inverse
edges. Specifically, for each triple (ℎ,𝑟,𝑡)∈E , we add its inverse
edge(𝑡,𝑟 inverse,ℎ), following prior works [ 11,25]. This augmenta-
tion allows the model to explore the graph in both directions and
has been shown to improve reasoning performance.Knowledge Graph Question Answering (KGQA).KGQA is the
task of answering natural language questions by reasoning over
a knowledge graph. Given a question and a KG, G={V,R,E} ,
the goal is to identify the correct entity or set of entities in V
that answer the query. This typically requires multi-hop reasoning,
where the model must traverse nodes and relations along valid
paths (e.g.,(ℎ,𝑟 1,𝑣1),(𝑣 1,𝑟2,𝑡)) to reach the answer.
Due to the large size of knowledge graphs, which cannot fit into
the limited context window of LLMs at once, most KGQA methods
(i.e., GraphRAG) first retrieve a subgraph relevant to the query and
then restrict the LLM’s reasoning to this subgraph. However, in
this two-stage pipeline, the retriever and generator are usually op-
timized independently and executed sequentially. In the following
subsections, we present experiments that illustrate the limitations
of this design, focusing on the quality of the retrieved subgraphs,
which can greatly affect the overall generation performance.
3.2 Experimental Settings
To conduct a comprehensive evaluation, we select one representa-
tive method from each retriever category as introduced in Section 2:
G-Retriever [ 19] for heuristic-based methods, GNN-RAG [ 33] for
GNN-based methods, and RoG [31] for LLM-based methods.
We select two widely used datasets: WebQSP [ 63], which primar-
ily contains questions requiring one- or two-hop reasoning, and
CWQ [ 46], which consists of more complex questions that require
multi-hop reasoning. For each question, there can be multiple cor-
rect answers. On average, the number of answers per test question
is 10.20 for WebQSP and 1.89 for CWQ. For the heuristic-based
method, i.e., G-Retriever, no training of the retriever is required. In
contrast, both GNN-RAG and RoG require training. Specifically, for
GNN-RAG, we train a GNN model as the retriever on each dataset,
while for RoG we fine-tune the Llama-3.1-8B-Instruct model [ 50]
to serve as the retriever on each dataset. To evaluate retriever per-
formance, we adopt three metrics:Hit, which measures whether
at least one correct answer is included in the retrieved content;
Recall, which measures the proportion of ground-truth answers
retrieved for each question; andPrecision, which measures the
ratio of correctly retrieved entities among all retrieved entities.
3.3 In-Distribution Retriever Performance
In this subsection, we aim to measure the performance of different
retrievers when they are both trained and tested on the same dataset
(i.e., WebQSP or CWQ). The performance of the three selected
methods is shown in Figure 1a, where we also include the ground-
truth Hit and Recall as upper bounds, since the underlying graph
does not contain all the answers for every question.
From the results in Figure 1a, we observe that none of the retriev-
ers achieve perfect performance, even in terms of Hit. For example,
the Recall and Precision of G-Retriever on the CWQ dataset are only
38.65 and 5.14, respectively, indicating that many correct answers
are missing while substantial noise is included in the retrieved
subgraphs. This is because G-Retriever relies on embedding sim-
ilarity to retrieve nodes, which is often insufficient for handling
complex queries. This aligns with the well-known limitation of
RAG methods in retrieving evidence for multi-hop queries [ 41,47],
where iterative retrieval strategies are usually adopted by com-
bining intermediate results from the generator. Among the three

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, and Jiliang Tang
G-RetrieverRoGGNN-RAG020406080100Performance (%)
WebQSPG-RetrieverRoGGNN-RAG
CWQ
GT Hit GT Recall Hit Recall Precision
(a) In-distribution performance comparison of three retrievers on
WebQSP and CWQ datasets. Bars show Hit, Recall, and Precision, while
dashed lines indicate ground-truth (GT) upper bounds.
G-RetrieverRoGGNN-RAG020406080100Performance (%)
CWQ  WebQSP
G-RetrieverRoGGNN-RAG
WebQSP  CWQ
Hit Recall Precision(b) Cross-distribution generalization performance between CWQ and
WebQSP. Solid bars denote transfer results, while transparent bars
show in-distribution results.
Figure 1: Retrieval and generalization performance of different methods on WebQSP and CWQ datasets.
methods, GNN-RAG performs best, but it still shows around a 10%
gap compared to the ground-truth Hit upper bound on the CWQ
dataset. This demonstrates that thelimitations of current retrievers
will substantially constrain downstream generation performance.
3.4 Cross-Distribution Generalization
In Section 3.3, we mainly studied the retriever performance when
training and testing on the same dataset. However, in real-world
applications it is often impractical to train a separate retriever
for every new graph. In this subsection, we therefore explore the
generalization ability of pretrained retrievers. Instead of fine-tuning
the retrievers for each dataset as in GNN-RAG [ 33] and RoG [ 31],
we directly apply a retriever pretrained on one dataset to test on
the other, in order to evaluate its cross-distribution generalization.
We first conduct experiments on the WebQSP and CWQ datasets,
which are both built on Freebase KG [ 4] but differ substantially in
question distribution. WebQSP consists of short, naturally collected
questions that require 1-2-hop reasoning, whereas CWQ contains
synthetically expanded questions with up to 4 hops and more com-
plex linguistic structures such as conjunctions and comparatives.
As shown in Figure 1b, both RoG and GNN-RAG suffer a significant
performance drop under the cross-distribution setting. For instance,
the Hit of GNN-RAG decreases from 92.07 to 55.09, and its Recall
drops from 85.43 to 31.68 on WebQSP when using the retriever
trained on CWQ, performing even worse than the heuristic-based
G-Retriever, which requires no retriever pretraining.
We further evaluate cross-graph generalization by applying re-
trievers trained on WebQSP to datasets with different knowledge
graphs, including MetaQA [ 69], which uses the WikiMovies KG,
and PathQuestion [ 71], which uses a Freebase subset with altered
relation names. As shown in Table 1, RoG fails to retrieve any valid
content, while GNN-RAG retrieves overly large subgraphs with
very low precision.
Table 1: Cross-graph generalization performance.
Webqsp->MetaQA Webqsp->PathQuestion
Hit Recall Precision Hit Recall Precision
RoG 0 0 0 0 0 0
GNN-RAG 38.3 13.36 0.02 63.8 51.7 0.05Overall, these results show that representative GraphRAG re-
trievers generalize poorly. We attribute this to two major factors:
(1)Distribution gap in questions.Retrievers are typically trained
to recognize the lexical and structural patterns of questions
within a single dataset (e.g., WebQSP or CWQ). When the ques-
tion style or reasoning composition changes, such as shifting
from short, 1-2-hop natural questions to complex, synthetic
multi-hop questions, the retriever fails to capture the correct
reasoning paths and relevant entities. This suggests that retriev-
ers overfit to dataset-specific surface forms rather than learning
transferable reasoning patterns.
(2)Distribution gap in graphs.Differences in the underlying
knowledge graphs, such as relation vocabulary, and entity type
distributions, further limit generalization. GNN-based retrievers
depend on node and edge embeddings learned from a specific
graph topology, while LLM-based retrievers (e.g., RoG) rely on
relation paths that may not align across graphs. As a result, both
classes of retrievers struggle to adapt when transferred to new
graphs with different relation semantics or structural patterns.
3.5 Discussions
From Section 3.3, we observe that a substantial portion of an-
swers cannot be retrieved by existing retrievers. Consequently,
even strong generators fail to produce these correct answers since
the necessary evidence is missing from the retrieved subgraphs.
This highlights a key limitation of current two-stage GraphRAG
paradigms—the retriever and generator operate independently and
cannot interact with each other on the graph during reasoning.
The results in Section 3.4 further suggest that existing retriev-
ers are over-specialized to their training datasets and knowledge
graphs. They lack the capability to adjust their retrieval behavior
based on the evolving environment or intermediate reasoning out-
comes.This static retrieval behavior prevents effective adaptation to
unseen question types and graph structures.
These insights motivates a new perspective on GraphRAG: rather
than decoupling retrieval and generation or retrieving all poten-
tially relevant content at once, the model should be able tointeract
with the whole graph step by step, selecting and exploring new
nodes and relations as reasoning progresses for the question. Such
a reasoning-driven retrieval process enables the model to dynami-
cally incorporate feedback from intermediate steps, allowing it to

Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
generalize across diverse questions and graph structures by learning
toreason and exploreinstead of memorizing fixed retrieval patterns.
4 Method
Motivated by the limitations of existing GraphRAG methods iden-
tified in Section 3, we propose Reasoning by Exploration (RoE), a
unified framework that integrates retrieval and generation within a
single process, as illustrated in Figure 2. Unlike traditional two-stage
GraphRAG pipelines, where a retriever extracts a static subgraph be-
fore reasoning begins,RoEenables a single LLM to actively explore
the entire knowledge graph step by step.
4.1 Exploration Formulation
We formulate reasoning inRoEas anexploration processover
a knowledge graph, where the LLM incrementally traverses the
graph to construct reasoning paths that lead to correct answers.
Formally, we formulate the exploration process as follows:
Given a question 𝑞and a knowledge graph G={V,R,E} , the
model begins from one or moreseed entities V0that are linked to
entities mentioned in the question. We expect the model to explore
one-hop neighbors from the currently explored nodes at each step,
progressively expanding the reasoning space. Specifically, at each
step𝑑, the model maintains a set ofcurrent reasoning paths:
P𝑑={𝑝𝑑
1,𝑝𝑑
2,...,𝑝𝑑
𝑛},
where each path𝑝𝑑
𝑖is defined as
𝑝𝑑
𝑖=[𝑣 0,𝑟1,𝑣1,...,𝑟𝑘,𝑣𝑘],
starting from a seed entity 𝑣0∈V 0and expanding through a se-
quence of relations and entities. We definefrontier nodesas
F𝑑={𝑣𝑘|𝑝𝑑
𝑖∈P𝑑},
representing the boundary entities that can be expanded at the next
step. At step𝑑, the model observes astate
𝑠𝑑=(𝑞,P𝑑,NF𝑑),
whereNF𝑑denotes neighbor nodes and edges of all frontier nodes.
The model then predicts an action𝑎that consists of two lists:
•Answers:A𝑑=[𝑎𝑛𝑠𝑑
1,𝑎𝑛𝑠𝑑
2,...,𝑎𝑛𝑠𝑑
𝑚], representing entities
predicted as current answers at step𝑑.
•New Exploration Paths: P𝑑+1={𝑝𝑑
𝑖⊕(𝑣𝑘,𝑟𝑘+1,𝑣𝑘+1)|𝑝𝑑
𝑖=
[𝑣0,𝑟1,𝑣1,...,𝑟𝑘,𝑣𝑘]∈P𝑑,(𝑣𝑘,𝑟𝑘+1,𝑣𝑘+1)∈E} , where⊕denotes
the extension of an existing path with a newly explored triplet.
The exploration continues until the model predicts that no new
paths should be expanded (i.e., P𝑑+1=∅) or a predefined maximum
number of steps𝐷 maxis reached.
The final outputs include the aggregated predicted answers
A=Ø
𝑑A𝑑,
which enables the model to find multiple answers through reason-
ing paths of different lengths.
Through this design, retrieval and generation become mutu-
ally informed and jointly optimized, enabling LLMs to construct
reasoning paths that are both complete and contextually relevant.
However, empowering LLMs to perform such structured graph ex-
ploration introduces several challenges. First, LLMs are pretrainedprimarily on unstructured text and therefore lack the inherent abil-
ity to navigate and reason over graphs [ 9]. Second, naively training
an LLM on exploration trajectories may cause it to memorize spe-
cific graph patterns rather than learning transferable exploration
strategies. To address these challenges, we introduce a two-stage
training strategy that equips the model with both graph exploration
ability and strong generalization across different graphs.
4.2 Stage I: Supervised Fine-Tuning for
Step-wise Exploration
In the first stage, we aim to equip the LLM with the basic ability
to perform step-wise graph exploration. Specifically, we require
thegold actionsfor each step, which include both the step-wise
answers and the new exploration paths as defined in Section 4.1.
Previous works [ 43,67] typically use theshortest pathfrom the
query entities to the answers to supervise their models. However,
we argue that relying solely on the shortest path is overly restrictive.
In real knowledge graphs, there may exist multiple valid reason-
ing paths connecting the same query entity to the correct answer;
constraining training to only the shortest one discourages explo-
ration and limits generalization. Moreover, the model does not
have a global view of the entire graph at each step. When several
neighboring entities share the same relation with the current fron-
tier entity, selecting only a single target node can be ambiguous,
especially when entity names are represented as IDs.
To address these issues, we construct the gold exploration paths
by including all reasoning paths whose lengths are below a thresh-
old, ensuring that the model learns to explore multiple plausible
reasoning routes. Additionally, when one node is selected for ex-
pansion, other nodes connected to the frontier entity through the
same relation are also treated as valid expansion candidates. For
the answers at each step, we use the newly explored entities that
correspond to the ground-truth answers as the gold answers. The
detailed procedure is summarized in Algorithm 1 in Appendix A.
In the supervised fine-tuning (SFT) stage, each training instance
is represented as a pair (𝑠𝑑,𝑎∗
𝑑)inDSFT, where𝑠𝑑denotes the state
at step𝑑(including the question, current reasoning paths, and local
neighborhood), and 𝑎∗
𝑑is the corresponding gold action constructed
in Algorithm 1. At each step, the model takes 𝑠𝑑as input and predicts
action𝑎𝑑=(A𝑑,P𝑑), which includes the step-wise answers and
the new exploration paths. The training objective is to maximize
the likelihood of the gold action 𝑎∗
𝑑given its state, leading to the
loss:
LSFT=−∑︁
𝑑log𝑃(𝑎∗
𝑑|𝑠𝑑;𝜃),
where𝜃denotes the model parameters. This loss encourages the
model to imitate the expert exploration behavior, deciding which
nodes to expand and which entities to output as answers at each
step, thereby acquiring the fundamental ability for structured graph
exploration before reinforcement-learning refinement in Stage II.
4.3 Stage II: Reinforcement Learning for
Exploration Optimization
Although the supervised fine-tuning (SFT) stage enables the model
to imitate gold trajectories, it may cause the LLM to memorize fixed
exploration patterns within a single dataset, limiting its ability to

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, and Jiliang Tang
               
Winston
ChurchillLord R andolph
Churchill
Jennie Jerome
Jack Churchillsiblingfather
mother
John George
Spencer  Churchill
Henry Winston
Spencer ChurchillchildchildJack Churchill
spouse
spouse
president
John George
Spencer  ChurchillnephewSeed Step = 1 Step = 2 Step = T
child
office Question: Who are the nephews of Winston Churchill?
Henry Winston
Spencer Churchill...
...
Gold Paths
Supervised Fine-T uning
LLM LLM
Generated Paths
LLMReward
Reinforcement LearningSTAGE 1
STAGE 2...
Reward  System...
Figure 2: The framework ofRoE. The model begins from the seed entity (green), incrementally expands to explored entities
(yellow), while discovering answer entities (red). Stage 1 uses SFT to learn gold reasoning paths, and the trained model is then
used as the initial model for Stage 2, where RL refines the exploration policy based on reward feedback.
generalize to unseen graphs or question types. To address this, we
introduce a reinforcement learning (RL) stage that allows the model
to interact with the knowledge graph and refine its exploration
strategy through feedback rather than imitation.
During this stage, we not only encourage the LLM to generate
correct step-wise answers and exploration paths, but also aim to
promote broader and more diverse exploration behaviors, enabling
the model to discover additional valid reasoning trajectories and
answers beyond those seen in the gold data. To enable such flexible
and interpretable optimization, we adopt a rule-based reinforce-
ment learning framework, which provides explicit feedback signals
without relying on a learned value model. This choice is inspired
by recent advances in reasoning LLMs, such as DeepSeek-R1 [ 15],
which have shown that rule-based rewards can effectively guide
reasoning and exploration, while simplifying training compared to
the original RLHF framework [36].
InRoE, we expect the model to generate outputs in the correct
format, align its predictions with the gold answers and exploration
paths, and simultaneously discover new valid answers and paths
while penalizing hallucinated ones. Therefore, we define the fol-
lowing five rewards:
Format Reward.Different from previous methods that rely on a
“thinking” process, we directly prompt the LLM to generate both
the step-wise answers and exploration paths in a structured JSON
format to improve efficiency. Specifically, at each step the model is
required to output:
{"answers": [ ], "exploration_paths": [ ]}.
A non-zero reward Rformat is given only when the model success-
fully follows the required format. This encourages the LLM to
produce well-structured outputs, facilitating reliable parsing and
consistent downstream evaluation during training.Answer Reward.We measure how many of the gold answers
A∗
𝑑in the current state 𝑠𝑑are successfully predicted by the model.
Specifically, we compute the recall between the predicted and gold
answer sets at each step𝑑:
Rans=|A𝑝
𝑑∩A∗
𝑑|
|A∗
𝑑|,
whereA𝑝
𝑑denotes the predicted answers. A higher reward is as-
signed when more ground-truth answers are correctly identified.
Answer Discovery Reward.To promote broader and more ef-
fective exploration, we further encourage the model to discover
new valid answers that were not present in the previous steps. Let
Adenote the set of all gold answers for the question. We reward
the model for predicting new correct answers A𝑝
𝑑∩(A\A∗
𝑑)and
penalize it for generating invalid onesA𝑝
𝑑\A:
Rans-dis=|A𝑝
𝑑∩(A\A∗
𝑑)|−𝛽·|A𝑝
𝑑\A|,
where𝛽is a penalty weight controlling the impact of hallucinated
predictions. This reward encourages the model to find novel correct
answers, by leveraging its internal knowledge and the reasoning
paths explored so far, while maintaining answer precision.
Exploration Reward.Similar to the answer reward, this reward
evaluates how well the model’s predicted reasoning paths match the
gold paths in the current state. Let P𝑝
𝑑andP∗
𝑑denote the predicted
and gold reasoning paths at step 𝑑, respectively. We compute the
reward as the recall between the predicted and gold path sets:
Rexplore =|P𝑝
𝑑∩P∗
𝑑|
|P∗
𝑑|.
A higher reward indicates that the model successfully explores the
correct reasoning paths in the current step, guiding it to perform
faithful and consistent graph traversal.

Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Exploration Discovery Reward.To further promote effective
and diverse reasoning, we encourage the model to explore new
valid paths while penalizing invalid ones. Let P𝑝
𝑑denote the set of
predicted paths at step 𝑑, andP∗
𝑑denote the set of gold paths. We
definenew pathsas those that do not appear in P∗
𝑑but exist on the
knowledge graphG, andinvalid pathsas those whose triplets are
not present inG. The reward is computed as:
Rexp-dis=|P𝑝
𝑑\P∗
𝑑|valid−𝛽·|P𝑝
𝑑|invalid,
where|P𝑝
𝑑\P∗
𝑑|validcounts the number of new but valid paths on the
graph and|P𝑝
𝑑|invalid counts the number of paths containing nonex-
istent triplets. This reward encourages the model to explore novel
and meaningful reasoning directions while discouraging invalid
graph traversals.
Final Reward.The overall reward for each step is defined as the
sum of all the individual rewards introduced above:
Rtotal=R format+R ans+R ans-dis+R explore+R exp-dis.
This combined reward jointly encourages the model to produce
well-structured outputs, generate correct answers, and discover
diverse and valid reasoning paths during reinforcement learning.
We optimizeRoEusingGroup Relative Policy Optimization
(GRPO)[ 15], a variant of Proximal Policy Optimization (PPO) [ 40].
GRPO eliminates the need for an additional value function approxi-
mation by using the average reward of multiple sampled outputs
as a relative baseline. For each input question 𝑥, GRPO samples
a group of outputs {𝑦1,𝑦2,...,𝑦𝐺}from the old policy 𝜋𝜃oldand
optimizes the new policy 𝜋𝜃by maximizing the following objective:
JGRPO(𝜃)=E𝑥∼D,{𝑦𝑖}𝐺
𝑖=1∼𝜋𝜃old(·|𝑥)"
1
𝐺𝐺∑︁
𝑖=11
|𝑦𝑖||𝑦𝑖|∑︁
𝑡=1min
𝑟𝑖,𝑡(𝜃)ˆ𝐴𝑖,𝑡,
clip 𝑟𝑖,𝑡(𝜃),1−𝜖,1+𝜖ˆ𝐴𝑖,𝑡
−𝛾𝐷 KL
𝜋𝜃(·|𝑥)∥𝜋 ref(·|𝑥)#
where
𝑟𝑖,𝑡(𝜃)=𝜋𝜃(𝑦𝑖,𝑡|𝑥,𝑦𝑖,<𝑡)
𝜋𝜃old(𝑦𝑖,𝑡|𝑥,𝑦𝑖,<𝑡), ˆ𝐴𝑖,𝑡=𝑅𝑖−mean({𝑅 𝑗}𝐺
𝑗=1)
std({𝑅𝑗}𝐺
𝑗=1),
and𝛾is a hyperparameter balancing task-specific rewards and KL-
divergence regularization. This formulation encourages the model
to prefer outputs with higher relative rewards within each group,
stabilizing training while improving sample efficiency. In our case,
the rewards𝑅 𝑖correspond to the total rule-based rewardsR total.
In summary,RoEis first supervised through step-wise fine-
tuning (Stage I) to learn the fundamental exploration behavior
from gold trajectories, and is then refined via reinforcement learn-
ing (Stage II) to improve its reasoning and generalization ability.
Through this two-stage training framework,RoElearns to jointly
reason and explore over graphs, constructing accurate and diverse
reasoning paths while producing faithful multi-hop answers.
5 Experiments
In this section, we conduct comprehensive experiments to validate
the effectiveness of the proposedRoE. Specifically, we aim to an-
swer the following research questions:(1) RQ1:How doesRoE
perform on benchmark datasets compared with baseline methods?
(2) RQ2:CanRoEgeneralize to unseen datasets better than existingGraphRAG approaches?(3) RQ3:How do different design choices
and factors influence the performance ofRoE?
5.1 Experimental Settings
Datasets.To evaluate the effectiveness ofRoE, we use two widely
adopted datasets:WebQSP[ 63] andCWQ[ 46], both constructed
upon the Freebase knowledge graph [ 4]. To further assess the gener-
alization ability of GraphRAG methods across different knowledge
graphs, we additionally useMetaQA[ 69], which is built on the
WikiMovies KG [ 34], andPathQuestion[ 71], which uses a Free-
base subset with modified relation names. For cross-graph general-
ization evaluation, we randomly sample 1,000 multi-hop questions
from each of these two datasets.
Evaluation Metrics.We follow previous works [ 16,18,27] and
adopt two evaluation metrics:HitandF1. Specifically, the Hit
measures the proportion of questions for which at least one correct
predicted answer, while F1 provides a balanced evaluation of both
precision and recall.
Baselines.We compareRoEwith a diverse set of baselines, which
can be grouped into four categories: (1)LLM-Only:methods that
rely solely on the internal knowledge of LLMs to answer ques-
tions without using external knowledge graphs. We include Flan-
T5-XL [ 8], Alpaca-7B [ 48], Llama-2-7B-Chat [ 50], Llama-3.1-8B-
Instruct [ 50], and GPT-4.1 [ 1]. (2)GNN-Only:models that use
GNNs for both retrieval and reasoning. We include GraftNet [ 44],
SR+NSM [ 67], and UniKGQA [ 22]. (3)LLMs + GNNs:hybrid ap-
proaches that combine LLMs and GNNs for retrieval and generation.
For instance, G-Retriever [ 19] leverages GNNs to enhance LLM rea-
soning, while GNN-RAG [ 33] employs GNNs for retrieval and LLMs
for generation. (4)LLMs + KG:methods that employ LLMs for both
retrieval and generation over knowledge graphs. We compare with
ToG [45], RoG [31], KD-CoT [52], and SubGraphRAG [27].
RoEsettings.ForRoE, we use the Llama-3.1-8B-Instruct model [ 50]
as the backbone. We divide the training data of both WebQSP and
CWQ into two parts: 60% is used for the SFT (stage I) and the re-
maining 40% for the RL (stage II). To reduce memory consumption
during training, we apply the LoRA [20] in both stages.
To ensure a fair comparison, we train each retriever and genera-
tor on a single dataset, rather than jointly training across multiple
datasets as done in RoG and GNN-RAG. Moreover, to eliminate
the influence of backbone model, we use the sameLlama-3.1-8B-
Instructmodel [ 50] for most LLM-based baselines, including RoG,
KD-CoT, SubGraphRAG, G-Retriever, and GNN-RAG.
5.2 Overall Performance Comparison
The overall performance on the WebQSP and CWQ datasets is
shown in Table 2. We make the following observations:
•The proposedRoEachieves the best performance across most
metrics on both datasets. Specifically,RoEattains relative im-
provements of 2.9% and 3.8% on Hit and F1, respectively, com-
pared to the second-best models on the WebQSP dataset. On the
CWQ dataset,RoEalso achieves a 3.1% relative improvement
in Hit, demonstrating its strong ability to reason on graphs.
•The LLM-only models, which lack access to external graph infor-
mation, generally perform worse than methods that incorporate

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, and Jiliang Tang
Table 2: Performance comparison on two KGQA benchmarks. The best and second-best are in bold and underline .
∗indicates the model uses Llama3.1-8B-Instruct as backbone.
Category MethodWebQSP CWQAvg. RankHits@1 F1 Hits@1 F1
LLMs-onlyFlan-T5-xl 31.00 - 14.70 - 12.00
Alpaca-7B 51.80 - 27.40 - 11.00
Llama-2-7B-Chat 64.40 - 34.60 - 8.50
Llama-3.1-8B-Instruct∗58.96 31.66 31.49 20.66 10.25
GPT-4.1 64.99 44.04 44.09 36.56 8.25
GNNs-onlyGraftNet - 62.40 - 32.70 8.00
SR+NSM - 64.10 - 47.10 5.50
UniKGQA - 70.20 - 48.00 3.50
LLMs+GNNsG-Retriver∗74.07 54.51 51.51 45.18 6.75
GNNRAG∗85.25 72.01 63.61 54.922.00
LLMs+KGsToG + Llama2-70B-Chat 63.70 - 53.60 - 7.50
RoG∗78.62 64.39 53.83 45.97 5.00
KD-CoT∗68.60 52.50 55.70 - 6.33
SubGraphRAG∗86.61 70.57 56.98 47.16 3.00
RoE∗89.13 74.76 66.4953.21 1.25
graphs. However, stronger models, such as GPT-4.1, achieve bet-
ter results, suggesting that large LLMs already possess internal
knowledge relevant to the benchmarks. This observation justi-
fies our choice that using the same LLM backbone for baselines
to ensure a fair comparison.
•The GNN-only models generally perform worse than methods
that incorporate LLMs and graph information, indicating that
the internal knowledge and reasoning capability of LLMs are
essential for solving multi-hop KGQA tasks.
•Compared with LLMs+GNNs methods,RoEachieves better
performance in most cases, indicating additional GNN modules
may not be necessary, suggesting LLMs themselves can acquire
effective graph reasoning capabilities through fine-tuning.
•For the LLMs+KGs methods, such as ToG, KD-CoT, and Sub-
GraphRAG, which prompt pretrained LLMs to iteratively re-
trieve and generate sub-questions or directly answer based on
a retrieved subgraph, the performance is still lower than that of
the proposedRoE. This suggests that pretrained LLMs are not
well-suited for graph reasoning without additional fine-tuning.
5.3 Generalization Performance Comparison
In this section, we evaluate the generalization capability ofRoE. Un-
like previous works [ 31,33], which fine-tune their models on the tar-
get graphs, we directly apply the pretrained model from one dataset
to unseen datasets without any additional training. Specifically, we
select G-Retriever, RoG, and GNN-RAG as baselines, and evaluate
the transfer performance under three settings:CWQ →WebQSP,
WebQSP→PathQuestion, andWebQSP→MetaQA.
The generalization results are presented in Figure 3. As shown,
the proposedRoEsignificantly outperforms all baselines across dif-
ferent transfer settings. For example, under the WebQSP →MetaQA
transfer,RoEachieves a Hit of93.90, while the second-best method
only reaches52.30, demonstrating the strong generalization ca-
pability ofRoEacross distributional shifts in both questions and
knowledge graphs. Specifically, although G-Retriever employs a
heuristic-based retriever without explicit retriever training, it still
requires training a GNN model to encode the graph structure. As aresult, G-Retriever performs well on the CWQ →WebQSP transfer
setting, where both datasets share the same underlying knowledge
graph, but performs poorly in other settings with different graph
structures. Similarly, the RoG model relies on predicting reasoning
relation paths, which are difficult to transfer across graphs with
different relation vocabularies. GNN-RAG relies on trained GNNs
for retrieval, which also makes it difficult to generalize to different
question distributions and knowledge graphs. As a result, it tends to
retrieve overly large subgraphs under transfer settings, as discussed
in Section 3, thereby introducing substantial noise that negatively
affects the generator’s reasoning performance.
5.4 Ablation Studies
In this subsection, we conduct ablation studies to verify the effec-
tiveness of the key components inRoE. Specifically, we analyze
the impact of the reinforcement learning (RL) stage and the pro-
posed discovery rewards for both answers and exploration paths.
We conduct experiments on two variants ofRoE: (1) removing
the reinforcement learning (RL) stage and using only supervised
fine-tuning (SFT), denoted asRoEw/o RL; and (2) removing the dis-
covery rewards for both Answer Discovery Reward and Exploration
Discovery Reward during the RL stage, denoted asRoEw/o Dis-
covery. We train these variants on WebQSP dataset and test their
performance on WebQSP, MetaQA and PathQuestion dataset.
The results are presented in Figure 4, from which we make the
following observations:(1)The RL stage is essential forRoE, es-
pecially under transfer settings.RoEw/o RL shows significantly
lower performance on the MetaQA and PathQuestion datasets,
demonstrating that models trained only with SFT tend to memo-
rize dataset-specific patterns and struggle to generalize to unseen
graphs without reinforcement learning.(2)The discovery rewards,
which guide the model to find more valid answers and paths while
penalizing hallucinations, are also crucial for performance. With-
out these rewards, the model tends to generate more hallucinated
answers, resulting in a notable drop in F1 under transfer settings.

Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
G-RetrieverRoGGNN-RAGOurs020406080100Performance (%)Hit F1
(a) CWQ→WebQSP
G-RetrieverRoGGNN-RAGOurs020406080100Performance (%)Hit F1 (b) WebQSP→MetaQA
G-RetrieverRoGGNN-RAGOurs020406080100Performance (%)Hit F1 (c) WebQSP→PathQuestion
Figure 3: Generalization performance (Hit/F1) of different methods across dataset transfers.
WebQSP MetaQA PathQuestion020406080100Performance (%)
RoE w/o RL RoE w/o Discovery RoE
Hit (filled) F1 (hollow)
Figure 4: Performance ofRoEand its variants pretrained on
the WebQSP dataset. Marker shapes denote different model
variants, while filled and hollow markers represent Hit and
metrics, respectively.
6 Conclusion
In this paper, we revisited the limitations of existing two-stage
GraphRAG frameworks, where the retriever and generator are op-
timized independently and lack interaction during reasoning. We
proposedReasoning by Exploration (RoE), a unified framework
that integrates retrieval and generation into a step-wise exploration
process. By allowing the model to interact with the graph dynami-
cally and incorporate feedback from intermediate steps,RoElearns
to reason and explore rather than rely on static retrieval patterns.
Extensive experiments on multiple KGQA benchmarks demonstrate
thatRoEconsistently outperforms strong baselines and generalizes
effectively across different question distributions and graphs.
References
[1]Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Floren-
cia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal
Anadkat, et al .2023. Gpt-4 technical report.arXiv preprint arXiv:2303.08774
(2023).
[2]Tareq Al-Moslmi, Marc Gallofré Ocaña, Andreas L Opdahl, and Csaba Veres.
2020. Named entity extraction for knowledge graphs: A literature overview.IEEE
access8 (2020), 32862–32881.
[3]Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Michal Podstawski,
Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Hubert Niewiadomski, Piotr
Nyczyk, et al .2024. Graph of thoughts: Solving elaborate problems with large
language models. InProceedings of the AAAI conference on artificial intelligence,
Vol. 38. 17682–17690.
[4]Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor.
2008. Freebase: a collaboratively created graph database for structuring human
knowledge. InProceedings of the 2008 ACM SIGMOD international conference onManagement of data. 1247–1250.
[5]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao
Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al .2024. A survey on
evaluation of large language models.ACM transactions on intelligent systems and
technology15, 3 (2024), 1–45.
[6]Nurendra Choudhary and Chandan K Reddy. 2023. Complex logical rea-
soning over knowledge graphs using large language models.arXiv preprint
arXiv:2305.01157(2023).
[7]Yucheng Chu, Peng He, Hang Li, Haoyu Han, Kaiqi Yang, Yu Xue, Tingting Li,
Joseph Krajcik, and Jiliang Tang. 2025. Enhancing LLM-Based Short Answer
Grading with Retrieval-Augmented Generation.arXiv preprint arXiv:2504.05276
(2025).
[8]Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus,
Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al .2024.
Scaling instruction-finetuned language models.Journal of Machine Learning
Research25, 70 (2024), 1–53.
[9]Xinnan Dai, Haohao Qu, Yifen Shen, Bohang Zhang, Qihao Wen, Wenqi Fan,
Dongsheng Li, Jiliang Tang, and Caihua Shan. 2024. How do large language mod-
els understand graph patterns? a benchmark for graph pattern comprehension.
arXiv preprint arXiv:2410.05298(2024).
[10] Xiang Deng, Yu Su, Alyssa Lees, You Wu, Cong Yu, and Huan Sun. 2021.
ReasonBERT: Pre-trained to reason with distant supervision.arXiv preprint
arXiv:2109.04912(2021).
[11] Ritam Dutt, Kasturi Bhattacharjee, Rashmi Gangadharaiah, Dan Roth, and Car-
olyn Rose. 2022. PerKGQA: Question answering over personalized knowledge
graphs. InFindings of the Association for Computational Linguistics: NAACL 2022.
253–268.
[12] Jinyuan Fang, Zaiqiao Meng, and Craig Macdonald. 2024. Reano: Optimising
retrieval-augmented reader models through knowledge graph generation. In
Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 2094–2112.
[13] Yanlin Feng, Xinyue Chen, Bill Yuchen Lin, Peifeng Wang, Jun Yan, and Xiang
Ren. 2020. Scalable multi-hop relational reasoning for knowledge-aware question
answering.arXiv preprint arXiv:2005.00646(2020).
[14] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023).
[15] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning.arXiv
preprint arXiv:2501.12948(2025).
[16] Kai Guo, Harry Shomer, Shenglai Zeng, Haoyu Han, Yu Wang, and Jiliang Tang.
2025. Empowering graphrag with knowledge filtering and integration.arXiv
preprint arXiv:2503.13804(2025).
[17] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al .
2024. Retrieval-augmented generation with graphs (graphrag).arXiv preprint
arXiv:2501.00309(2024).
[18] Gaole He, Yunshi Lan, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen. 2021. Im-
proving multi-hop knowledge base question answering by learning intermediate
supervision signals. InProceedings of the 14th ACM international conference on
web search and data mining. 553–561.
[19] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun,
Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented gen-
eration for textual graph understanding and question answering.Advances in
Neural Information Processing Systems37 (2024), 132876–132907.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, and Jiliang Tang
[20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, Weizhu Chen, et al .2022. Lora: Low-rank adaptation of large
language models.ICLR1, 2 (2022), 3.
[21] Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu,
and Ji-Rong Wen. 2024. Kg-agent: An efficient autonomous agent framework
for complex reasoning over knowledge graph.arXiv preprint arXiv:2402.11163
(2024).
[22] Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. 2022. Unikgqa:
Unified retrieval and reasoning for solving multi-hop question answering over
knowledge graph.arXiv preprint arXiv:2212.00959(2022).
[23] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented
generation. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing. 7969–7992.
[24] Enkelejda Kasneci, Kathrin Seßler, Stefan Küchemann, Maria Bannert, Daryna
Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke
Hüllermeier, et al .2023. ChatGPT for good? On opportunities and challenges
of large language models for education.Learning and individual differences103
(2023), 102274.
[25] Seyed Mehran Kazemi and David Poole. 2018. Simple embedding for link predic-
tion in knowledge graphs.Advances in neural information processing systems31
(2018).
[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[27] Mufei Li, Siqi Miao, and Pan Li. 2024. Simple is effective: The roles of graphs and
large language models in knowledge-graph-based retrieval-augmented genera-
tion.arXiv preprint arXiv:2410.20724(2024).
[28] Guangyi Liu, Yongqi Zhang, Yong Li, and Quanming Yao. 2024. Dual reasoning:
A gnn-llm collaborative framework for knowledge graph question answering.
arXiv preprint arXiv:2406.01145(2024).
[29] Elita Lobo, Chirag Agarwal, and Himabindu Lakkaraju. 2024. On the impact
of fine-tuning on chain-of-thought reasoning.arXiv preprint arXiv:2411.15382
(2024).
[30] Haoran Luo, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang,
Meina Song, Xiaobao Wu, Yifan Zhu, Luu Anh Tuan, et al .2025. Graph-r1:
Towards agentic graphrag framework via end-to-end reinforcement learning.
arXiv preprint arXiv:2507.21892(2025).
[31] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2023. Reasoning
on graphs: Faithful and interpretable large language model reasoning.arXiv
preprint arXiv:2310.01061(2023).
[32] Yao Ma and Jiliang Tang. 2021.Deep learning on graphs. Cambridge University
Press.
[33] Costas Mavromatis and George Karypis. 2024. Gnn-rag: Graph neural retrieval
for large language model reasoning.arXiv preprint arXiv:2405.20139(2024).
[34] Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bor-
des, and Jason Weston. 2016. Key-value memory networks for directly reading
documents.arXiv preprint arXiv:1606.03126(2016).
[35] Marianna Nezhurina, Lucia Cipolina-Kun, Mehdi Cherti, and Jenia Jitsev. 2024.
Alice in wonderland: Simple tasks showing complete reasoning breakdown in
state-of-the-art large language models.arXiv preprint arXiv:2406.02061(2024).
[36] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al .2022.
Training language models to follow instructions with human feedback.Advances
in neural information processing systems35 (2022), 27730–27744.
[37] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2024. Graph retrieval-augmented generation: A survey.
arXiv preprint arXiv:2408.08921(2024).
[38] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano
Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language
model is secretly a reward model.Advances in neural information processing
systems36 (2023), 53728–53741.
[39] Diego Sanmartin. 2024. Kg-rag: Bridging the gap between knowledge and cre-
ativity.arXiv preprint arXiv:2405.12035(2024).
[40] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
2017. Proximal policy optimization algorithms.arXiv preprint arXiv:1707.06347
(2017).
[41] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu
Chen. 2023. Enhancing retrieval-augmented large language models with iterative
retrieval-generation synergy.arXiv preprint arXiv:2305.15294(2023).
[42] Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi
Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, et al .2025. Stop
overthinking: A survey on efficient reasoning for large language models.arXiv
preprint arXiv:2503.16419(2025).
[43] Haitian Sun, Tania Bedrax-Weiss, and William W Cohen. 2019. Pullnet: Open
domain question answering with iterative retrieval on knowledge bases and text.
arXiv preprint arXiv:1904.09537(2019).[44] Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhut-
dinov, and William W Cohen. 2018. Open domain question answering using
early fusion of knowledge bases and text.arXiv preprint arXiv:1809.00782(2018).
[45] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel M Ni, Heung-Yeung Shum, and Jian Guo. 2023. Think-on-graph:
Deep and responsible reasoning of large language model on knowledge graph.
arXiv preprint arXiv:2307.07697(2023).
[46] Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for
answering complex questions.arXiv preprint arXiv:1803.06643(2018).
[47] Yixuan Tang and Yi Yang. 2024. Multihop-rag: Benchmarking retrieval-
augmented generation for multi-hop queries.arXiv preprint arXiv:2401.15391
(2024).
[48] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos
Guestrin, Percy Liang, and Tatsunori B Hashimoto. 2023. Stanford alpaca: An
instruction-following llama model.
[49] Arun James Thirunavukarasu, Darren Shu Jeng Ting, Kabilan Elangovan, Laura
Gutierrez, Ting Fang Tan, and Daniel Shu Wei Ting. 2023. Large language models
in medicine.Nature medicine29, 8 (2023), 1930–1940.
[50] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne
Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal
Azhar, et al .2023. Llama: Open and efficient foundation language models.arXiv
preprint arXiv:2302.13971(2023).
[51] Chaojie Wang, Yishi Xu, Zhong Peng, Chenxi Zhang, Bo Chen, Xinrun Wang,
Lei Feng, and Bo An. 2023. keqing: knowledge-based question answering is a
nature chain-of-thought mentor of LLM.arXiv preprint arXiv:2401.00426(2023).
[52] Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin,
Wenge Rong, and Zhang Xiong. 2023. Knowledge-driven cot: Exploring faithful
reasoning in llms for knowledge-intensive question answering.arXiv preprint
arXiv:2308.13259(2023).
[53] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi,
Quoc V Le, Denny Zhou, et al .2022. Chain-of-thought prompting elicits reasoning
in large language models.Advances in neural information processing systems35
(2022), 24824–24837.
[54] Yilin Wen, Zifeng Wang, and Jimeng Sun. 2023. Mindmap: Knowledge graph
prompting sparks graph of thoughts in large language models.arXiv preprint
arXiv:2308.09729(2023).
[55] Nirmalie Wiratunga, Ramitha Abeyratne, Lasal Jayawardena, Kyle Martin, Stew-
art Massie, Ikechukwu Nkisi-Orji, Ruvan Weerasinghe, Anne Liret, and Bruno
Fleisch. 2024. CBR-RAG: case-based reasoning for retrieval augmented gen-
eration in LLMs for legal question answering. InInternational Conference on
Case-Based Reasoning. Springer, 445–460.
[56] Yaozu Wu, Yankai Chen, Zhishuai Yin, Weiping Ding, and Irwin King. 2023.
A survey on graph embedding techniques for biomedical data: Methods and
applications.Information Fusion100 (2023), 101909.
[57] Yike Wu, Nan Hu, Sheng Bi, Guilin Qi, Jie Ren, Anhuan Xie, and Wei Song. 2023.
Retrieve-rewrite-answer: A kg-to-text enhanced llms framework for knowledge
graph question answering.arXiv preprint arXiv:2309.11206(2023).
[58] Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and
S Yu Philip. 2020. A comprehensive survey on graph neural networks.IEEE
transactions on neural networks and learning systems32, 1 (2020), 4–24.
[59] Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang,
Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, et al .2025. Towards
large reasoning models: A survey of reinforced reasoning with large language
models.arXiv preprint arXiv:2501.09686(2025).
[60] Ran Xu, Wenqi Shi, Yue Yu, Yuchen Zhuang, Bowen Jin, May D Wang, Joyce C Ho,
and Carl Yang. 2024. Ram-ehr: Retrieval augmentation meets clinical predictions
on electronic health records.arXiv preprint arXiv:2403.00815(2024).
[61] Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D
Manning, Percy S Liang, and Jure Leskovec. 2022. Deep bidirectional language-
knowledge graph pretraining.Advances in Neural Information Processing Systems
35 (2022), 37309–37323.
[62] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure
Leskovec. 2021. QA-GNN: Reasoning with language models and knowledge
graphs for question answering.arXiv preprint arXiv:2104.06378(2021).
[63] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and
Jina Suh. 2016. The value of semantic parse labeling for knowledge base ques-
tion answering. InProceedings of the 54th Annual Meeting of the Association for
Computational Linguistics (Volume 2: Short Papers). 201–206.
[64] Chuanyue Yu, Kuo Zhao, Yuhan Li, Heng Chang, Mingjian Feng, Xiangzhe
Jiang, Yufei Sun, Jia Li, Yuzhi Zhang, Jianxin Li, et al .2025. GraphRAG-R1:
Graph Retrieval-Augmented Generation with Process-Constrained Reinforce-
ment Learning.arXiv preprint arXiv:2507.23581(2025).
[65] Boyu Zhang, Hongyang Yang, Tianyu Zhou, Muhammad Ali Babar, and Xiao-
Yang Liu. 2023. Enhancing financial sentiment analysis via retrieval augmented
large language models. InProceedings of the fourth ACM international conference
on AI in finance. 349–356.

Reasoning by Exploration: A Unified Approach to Retrieval and Generation over Graphs Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
[66] Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang.
2024. Rest-mcts*: Llm self-training via process reward guided tree search.Ad-
vances in Neural Information Processing Systems37 (2024), 64735–64772.
[67] Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, and Hong
Chen. 2022. Subgraph retrieval enhanced model for multi-hop knowledge base
question answering.arXiv preprint arXiv:2202.13296(2022).
[68] Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang,
Christopher D Manning, and Jure Leskovec. 2022. Greaselm: Graph reasoning en-
hanced language models for question answering.arXiv preprint arXiv:2201.08860
(2022).
[69] Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J Smola, and Le Song.
2018. Variational Reasoning for Question Answering with Knowledge Graph. In
AAAI.
[70] Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, and Minlie Huang. 2023.
Large language models are not robust multiple choice selectors.arXiv preprint
arXiv:2309.03882(2023).
[71] Mantong Zhou, Minlie Huang, and Xiaoyan Zhu. 2018. An interpretable reasoning
network for multi-relation question answering.arXiv preprint arXiv:1801.04726
(2018).
[72] Yuqi Zhu, Shuofei Qiao, Yixin Ou, Shumin Deng, Shiwei Lyu, Yue Shen, Lei
Liang, Jinjie Gu, Huajun Chen, and Ningyu Zhang. 2024. Knowagent: Knowledge-
augmented planning for llm-based agents.arXiv preprint arXiv:2403.03101(2024).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, and Jiliang Tang
A Algorithm of SFT Dataset Construction
In this section, we detail the algorithm to construct the SFT datasets
to trainRoE. The algorithm is summarized in Algorithm 1, where
lines 1–14 construct all gold-consistent reasoning paths up to length
𝐿max; line 20 derives the step-wise gold answers; and lines 21–22
generate the step-wise new exploration paths.
B Datasets
In the experiments, we adopt four different datasets, i.e.WebQSP[ 63]
andCWQ[ 46],MetaQA[ 69] andPathQuestion[ 71]. For the We-
bQSP and CWQ dataset, we leverage the processed dataset in Luo
et al. [31] . For the PathQuestion and MetaQA dataset, we randomly
sample 1,000 samples of two hop questions. The statistics of these
datasets are shown in Table 3.
Table 3: Statistics of datasets.
Datasets #Train #Test Max #hop
WebQSP 2,826 1,628 2
CWQ 27,639 3,531 4
PathQuestion 0 1,000 2
MetaQA 0 1,000 2
C Experimental Settings
In this section, we present the detailed experimental settings for
RoE. ForRoEand most of baseline models involving LLMs, we
use theLlama-3.1-8B-Instructmodel [ 50] as the backbone. We
first leverage Algorithm 1 to construct the gold trajectories for
supervised fine-tuning. The training data of both WebQSP and
CWQ are divided into two parts: 60% for theSupervised Fine-
Tuning (SFT)stage and the remaining 40% for theReinforcement
Learning (RL)stage. To reduce memory consumption, we adopt
LoRA[20] in both stages with a rank of 32.
During SFT, the loss is calculated only on the action prediction
component, and the model is optimized using the AdamW optimizer
with a learning rate of1 ×10−4. For the RL stage, we apply a
smaller learning rate of5 ×10−6and use the Group Relative Policy
Optimization (GRPO) algorithm to optimize the policy with rule-
based rewards. For each sample, we generate 4 responses. Each
training step uses a batch size of 4 with gradient accumulation to
fit within GPU memory. Training is performed on 2×H200 GPUs.
During inference, we follow a depth-first exploration strategy.
Because each node in the knowledge graph may have numerous
neighbors that cannot fit into the LLM context window at once,
we split the neighbors into multiple batches and iteratively feed
them to the model. We use a depth-first search algorithm to explore
the graph, and we set the maximum exploration depth to 5 hops
to prevent overly long reasoning chains and context overflow. For
reproducibility, all random seeds are fixed, and the same prompt
templates for SFT and RL.Algorithm 1 SFT Dataset Construction
Input: Question𝑞; KGG={V,R,E} ; seed entitiesV0; gold an-
swersA∗; max path length𝐿 max
Output:DatasetD SFT={(𝑠𝑑, 𝑎∗
𝑑)}𝐿max−1
𝑑=0with𝑎∗
𝑑=(A∗
𝑑,P∗
𝑑)
1:Phase I: Mine all gold-consistent paths (length≤𝐿 max)
2:Π←∅;Q←{[𝑣 0]|𝑣 0∈V 0}
3:whileQnot emptydo
4:𝑝←pop fromQ;𝑣←frontier(𝑝)
5:if|𝑝|−1>𝐿 maxthen continue
6:end if
7:if𝑣∈A∗thenΠ←Π∪{𝑝}
8:end if
9:for all(𝑣,𝑟,𝑢)∈Edo
10:if𝑢∉𝑝then⊲simple-path constraint (avoid cycles)
11:Q←Q∪{𝑝⊕(𝑣,𝑟,𝑢)}
12:end if
13:end for
14:end while
Phase II: Build step-wise actions from mined paths
15:Define depth-indexed prefix pools for𝑑=0,...,𝐿 max:
Pref(𝑑)={all prefixes of any𝑝∈Πwith|prefix|−1=𝑑}
16:D SFT←∅
17:for𝑑=0to𝐿 max−1do
18:P𝑑←Pref(𝑑);F 𝑑←{frontier(𝑝)|𝑝∈P 𝑑}
19:State:𝑠 𝑑←(𝑞,P𝑑,NF𝑑)
20:Gold step-answers:
A∗
𝑑={𝑢∈A∗| ∃(𝑣,𝑟,𝑢)∈N F𝑑}
21:Gold one-hop extensions:
EΠ
𝑑={(𝑝,(𝑣,𝑟,𝑢))∈Pref(𝑑+1) |𝑝∈P 𝑑,frontier(𝑝)=𝑣}
22:Same-relation sibling expansion:
P∗
𝑑=
𝑝⊕(𝑣,𝑟,𝑢′)(𝑝,(𝑣,𝑟,𝑢))∈EΠ
𝑑,(𝑣,𝑟,𝑢′)∈E	
23: Record step: 𝑎∗
𝑑←(A∗
𝑑,P∗
𝑑);DSFT←D SFT∪{(𝑠𝑑, 𝑎∗
𝑑)}
24:end for
25:returnD SFT