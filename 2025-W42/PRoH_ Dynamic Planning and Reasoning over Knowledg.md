# PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation

**Authors**: Xiangjun Zai, Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Wenjie Zhang

**Published**: 2025-10-14 12:13:23

**PDF URL**: [http://arxiv.org/pdf/2510.12434v1](http://arxiv.org/pdf/2510.12434v1)

## Abstract
Knowledge Hypergraphs (KHs) have recently emerged as a knowledge
representation for retrieval-augmented generation (RAG), offering a paradigm to
model multi-entity relations into a structured form. However, existing KH-based
RAG methods suffer from three major limitations: static retrieval planning,
non-adaptive retrieval execution, and superficial use of KH structure and
semantics, which constrain their ability to perform effective multi-hop
question answering. To overcome these limitations, we propose PRoH, a dynamic
Planning and Reasoning over Knowledge Hypergraphs framework. PRoH incorporates
three core innovations: (i) a context-aware planning module that sketches the
local KH neighborhood to guide structurally grounded reasoning plan generation;
(ii) a structured question decomposition process that organizes subquestions as
a dynamically evolving Directed Acyclic Graph (DAG) to enable adaptive,
multi-trajectory exploration; and (iii) an Entity-Weighted Overlap (EWO)-guided
reasoning path retrieval algorithm that prioritizes semantically coherent
hyperedge traversals. Experiments across multiple domains demonstrate that PRoH
achieves state-of-the-art performance, surpassing the prior SOTA model
HyperGraphRAG by an average of 19.73% in F1 and 8.41% in Generation Evaluation
(G-E) score, while maintaining strong robustness in long-range multi-hop
reasoning tasks.

## Full Text


<!-- PDF content starts -->

PRoH: Dynamic Planning and Reasoning over Knowledge
Hypergraphs for Retrieval-Augmented Generation
Xiangjun Zai
University of New South Wales
Sydney, Australia
xiangjun.zai@unsw.edu.auXingyu Tan
University of New South Wales
Sydney, Australia
xingyu.tan@unsw.edu.auXiaoyang Wang
University of New South Wales
Sydney, Australia
xiaoyang.wang1@unsw.edu.au
Qing Liu
Data61, CSIRO
Sydney, Australia
q.liu@data61.csiro.auXiwei Xu
Data61, CSIRO
Sydney, Australia
xiwei.xu@data61.csiro.auWenjie Zhang
University of New South Wales
Sydney, Australia
wenjie.zhang@unsw.edu.au
Abstract
Knowledge Hypergraphs (KHs) have recently emerged as a knowl-
edge representation for retrieval-augmented generation (RAG),
offering a paradigm to model multi-entity relations into a struc-
tured form. However, existing KH-based RAG methods suffer from
three major limitations: static retrieval planning, non-adaptive re-
trieval execution, and superficial use of KH structure and semantics,
which constrain their ability to perform effective multi-hop ques-
tion answering. To overcome these limitations, we proposePRoH,
a dynamic Planning and Reasoning over Knowledge Hypergraphs
framework. PRoH incorporates three core innovations: (i) a context-
aware planning module that sketches the local KH neighborhood to
guide structurally grounded reasoning plan generation; (ii) a struc-
tured question decomposition process that organizes subquestions
as a dynamically evolving Directed Acyclic Graph (DAG) to enable
adaptive, multi-trajectory exploration; and (iii) an Entity-Weighted
Overlap (EWO)–guided reasoning path retrieval algorithm that pri-
oritizes semantically coherent hyperedge traversals. Experiments
across multiple domains demonstrate that PRoH achieves state-
of-the-art performance, surpassing the prior SOTA model Hyper-
GraphRAG by an average of 19.73% in F1 and 8.41% in Generation
Evaluation (G-E) score, while maintaining strong robustness in
long-range multi-hop reasoning tasks.
ACM Reference Format:
Xiangjun Zai, Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, and Wenjie
Zhang. 2018. PRoH: Dynamic Planning and Reasoning over Knowledge
Hypergraphs for Retrieval-Augmented Generation. InProceedings of Make
sure to enter the correct conference title from your rights confirmation emai
(Conference acronym ’XX).ACM, New York, NY, USA, 13 pages. https://doi.
org/XXXXXXX.XXXXXXX
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/18/06
https://doi.org/XXXXXXX.XXXXXXX1 Introduction
To improve the factual accuracy and specificity of large language
model (LLM) responses, Retrieval-Augmented Generation (RAG)
has emerged as a promising approach that integrates external
knowledge through the in-context learning capabilities of LLMs.
However, traditional RAG systems rely primarily on semantic sim-
ilarity, fail to capture the structured relational knowledge inher-
ent in many information domains, and often retrieve redundant or
noisy content [ 25]. To address this limitation, Graph-based RAG has
been introduced to integrate explicitly structured representations of
knowledge into the retrieval process, enabling more accurate and in-
terpretable reasoning [ 5,10–12]. By representing entities and their
relationships as Knowledge Graphs (KGs), these approaches can
capture indirect relations and support multi-hop reasoning across
interconnected facts. Most existing Graph-based RAG frameworks,
however, model only relations that involve exactly two entities.
This design overlooks a fundamental property of real-world knowl-
edge, which is, many relations are inherently n-ary, involving more
than two entities simultaneously. As shown in Figure 1, the relation
"Mario + Rabbids Kingdom Battle is the first major collaboration
between Nintendo and Ubisoft." connects three entities: "Mario +
Rabbids Kingdom Battle", "Nintendo" and "Ubisoft". In such cases,
the semantics of a relation is established only when all participating
entities are considered together. Decomposing these n-ary relations
into multiple binary edges inevitably causes a loss of critical struc-
tural and semantic information [6, 9, 26, 32].
To address the representational gap, Knowledge Hypergraphs
(KHs) have been proposed as a more compact and semantically
expressive knowledge structure for Graph-based RAG [ 4,7,19,31].
KHs generalize standard KGs by allowing hyperedges to connect
multiple nodes simultaneously, naturally capturing complex multi-
entity interactions and preserving n-ary relational semantics more
faithfully. As a result, KHs enhance both the storage of knowledge
and its retrieval for downstream comprehension tasks. Currently,
most KH-based RAG systems follow a three-stage pipeline: 𝑖)KH
construction: extract entities and n-ary relations from the text
sources to build the KH; 𝑖𝑖)KH-guided retrieval: link query topics
with entities and map queries to hyperedges through predefined
similarity metrics, followed by a shallow, heuristic-based graph
fusion that retrieves graph elements along with their correspond-
ing source text chunks;𝑖𝑖𝑖)Answer generation: pass the retrievedarXiv:2510.12434v1  [cs.CL]  14 Oct 2025

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Xiangjun Zai et al.
Knowledge HypergraphWhich tactical role-playing game released in 2017 was the first collaboration
between Nintendo and a French game publisher that  also exhibited at E3 2021?
Knowledge Graph
NintendoE3 2021Ubisoft
CapcomSquare EnixBandai Namco
Mario + Rabbids
Kingdom BattleFirst
Collaboration?
Mario + Rabbids Kingdom              
Battle is the first major collaboration
between Nintendo and Ubisoft.Nintendo, Capcom, Ubisoft, Square
Enix and  Bandai Namco were part
of the E3 2021 line up.
CapcomBandai
NamcoSquare
Enix
E3 2021 Nintendo UbisoftMario +
Rabbids
Kingdom
Battle
Figure 1: Illustration of Knowledge Hypergraph.
knowledge directly to the generation module. Although this ap-
proach has shown promising results in question-answering (QA)
tasks, it does not fully exploit the expressive potential of KHs and
suffers from the following limitations:
Limitation 1: Static Retrieval Planning. Existing frameworks rely
on predefined, hard-coded retrieval pipelines that apply the same
sequence of operations regardless of query content or graph context.
For example, HGRAG [ 31] performs a KH diffusion with a fixed
number of iterations, without evaluating whether the retrieved
passages are sufficient. This rigid design prevents the model from
adapting its retrieval plan to the question semantics or the KH
topology, leading to inefficient and misaligned knowledge access.
Limitation 2: Non-Adaptive Retrieval Execution. Current systems
perform retrieval in a one-shot, non-iterative manner, relying solely
on the original query. For instance, HyperGraphRAG [ 19] identifies
and retrieves relevant entities and hyperedges based on a predefined
similarity threshold in one graph call. Such static execution fails to
refine retrieval using intermediate reasoning results, limiting the
system’s capability for multi-hop reasoning.
Limitation 3: Superficial Use of Graph Structure and Semantics.
Existing methods primarily treat hyperedges as simple links or
routing mechanisms for accessing associated text chunks [ 7,19].
This superficial treatment ignores the rich relational semantics
encoded in hyperedges and misses the opportunity to guide more
precise retrieval and reasoning within KH-based RAG frameworks.
Contribution. To better realize the potential of KHs for RAG, we
introducePRoH, a dynamic KH-based RAG framework that fully
leverages the expressive power of KHs. PRoH is designed to perform
structured planning and reasoning directly over KHs, enabling the
retriever to adaptively explore, refine, and integrate knowledgefor multi-hop question answering. The key ideas of PRoH can be
summarized as follows:
Context aware planning. PRoH employs a graph scope–aware plan-
ning strategy. Before performing question decomposition, PRoH
first sketches the local neighborhood of the topic entities within the
KH. This pre-planning step provides the LLM with a brief yet infor-
mative view of the topological and semantic scope of the question-
relevant subgraph, mitigating the mismatch between linguistic-only
decomposition and the actual graph knowledge available. Conse-
quently, the LLM produces reasoning plans that are more feasible
and better aligned with the structure of the underlying KH.
Structured question decomposition with iterative refinement.
PRoH adopts a structured question decomposition approach to
explicitly capture the dependencies among subquestions. Instead
of treating subquestions as a linear sequence, the reasoning plan
is represented as a Directed Acyclic Graph (DAG) that captures
logical precedence among them. As reasoning progresses following
the topological order of the subquestions, the DAG is iteratively
refined. To be more specific, later subquestions are updated, and
new subquestions and dependencies may emerge. This mechanism
allows the reasoning plan to evolve dynamically and remain
consistent with the current reasoning state. PRoH also introduces
a state-space search mechanism that performs reasoning as a
branching exploration from the current reasoning state, effectively
modeling the process as a tree. Unlike prior methods that assume
each subquestion has one single correct answer, our approach
allows multiple candidate answers per subquestion. That is, several
reasoning trajectories can coexist. This design corresponds to the
multi-entity nature of n-ary relations, allowing PRoH to manage
ambiguity and recover from local errors. The state exploration
continues until one or more trajectories reach a goal state, where
all subquestions are resolved and a final answer can be derived.
EWO–guided reasoning path retrieval. PRoH employs a fine-
grained reasoning path retrieval strategy guided by the Entity-
Weighted Overlap (EWO) score, specifically designed for KHs.
When visiting a hyperedge, for each hyperedge that overlaps with
the current one, the retriever evaluates how strongly the overlap-
ping entities contribute to answering the current subquestion and
aggregates these relevance scores to determine the next traversal
direction. This process allows the retriever to prioritize semanti-
cally meaningful connections rather than relying on purely struc-
tural overlaps. As a result, the retrieved reasoning paths are better
aligned with the underlying semantics of the question, enabling
more accurate and interpretable multi-hop reasoning.
In summary, the main contributions of this paper are as follows:
•We propose PRoH, a dynamic Knowledge Hypergraph-based
RAG framework that fully leverages the expressive power of
hypergraphs for multi-hop question answering.
•We introduce a context-aware planning mechanism that sketches
the underlying Knowledge Hypergraph and generates feasible
reasoning plans.
•We develop an Entity-Weighted Overlap (EWO)–guided reason-
ing path retrieval strategy for fine-grained, semantically aware
exploration of Knowledge Hypergraphs.
•PRoH consistently achieves better performance and interpretabil-
ity than the state-of-the-art HyperGraphRAG framework across

PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
multiple knowledge domains, surpassing it by an average of
19.73% in F1 and 8.41% in Generation Evaluation (G-E).
2 Related Work
Graph-based RAG. Unlike traditional RAG systems, which rely on
flat textual corpora as their primary knowledge source, Graph-based
RAG leverages structured representations of relational knowledge.
These systems either construct knowledge graphs from documents
or utilize existing large-scale graphs such as Wikidata [ 30] and
Freebase [ 1], enabling more precise retrieval and relation-aware
reasoning [ 13,25,35,40]. EmbedKGQA [ 27] first introduces KG
embedding methods into multi-hop KGQA. Recent representative
frameworks [ 5,10,11,33,39] extract entities and relations from
unstructured text using LLMs and index them for retrieval via
graph traversal or embedding-based matching. More advanced ap-
proaches [ 3,8,16–18,22,23,28,29,37,38] incorporate iterative
reasoning and adaptive planning mechanisms to enhance graph-
guided inference. Learning-based methods [ 14,15,20,21,24,34,36]
have also shown promising results. Despite these advances, exist-
ing graph-based RAG frameworks remain limited by their binary
relational representations, which restrict their capacity to model
multi-entity facts and capture higher-order relational semantics.
Knowledge Hypergraph-based RAG. Early systems such as Hy-
perGraphRAG [ 19] and Hyper-RAG [ 7] extract hyperedges from tex-
tual sources to capture higher-order relations and employ similarity-
based retrieval to identify relevant entities, hyperedges, and text
chunks for question answering. IdepRAG [ 4] leverages the hy-
pergraph structure by performing Personalized PageRank over
topic entities to locate contextually relevant evidence. Meanwhile,
HGRAG [ 31] introduces a cross-granularity retrieval mechanism
that integrates fine-grained entity similarity with coarse-grained
passage similarity through hypergraph diffusion, effectively balanc-
ing structural precision and semantic coherence. However, current
hypergraph-based approaches still rely on heuristic, one-shot re-
trieval pipelines and lack context-aware and iterative reasoning
capabilities, motivating the framework proposed in this study.
3 Preliminary
LetH=(V,E) denote a knowledge hypergraph, where Vand
Erepresent the set of entities and hyperedges, respectively. Each
hyperedge𝑒∈E links a sequence of entities in V, i.e.,𝑉(𝑒)=
(𝑣1,𝑣2···,𝑣𝑛)where𝑛≥1. An n-ary relationship fact is modeled
as𝑓=(𝑒,𝑉(𝑒)) . We denote the set of hyperedges in Ethat contains
the entity𝑣as𝐸(𝑣), i.e.,𝐸(𝑢)={𝑒|𝑣∈𝑉(𝑒)∧𝑒∈E} . Given a
knowledge hypergraph H=(V,E) , a subgraphHS=(VS,ES)
is an induced subgraph of H, ifES⊆E,VS={𝑣|𝑣∈𝑉(𝑒)∧
𝑒∈ES}. Two hyperedges 𝑒𝑖and𝑒𝑗are considered connected iff
𝑉(𝑒𝑖)∩𝑉(𝑒𝑗)≠∅ , i.e., an overlap of entities exists between 𝑒𝑖and
𝑒𝑗. Given a hyperedge 𝑒, the set of hyperedges connected to 𝑒are
defined as the neighbor edge set Nbr(𝑒)={𝑒′|𝑉(𝑒′)∩𝑉(𝑒)≠
∅∧𝑒′≠𝑒∧𝑒′∈E}.
Definition 1 (Hypergraph-based RAG).Given a question 𝑞
and a knowledge hypergraph H=(V,E) , hypergraph-based RAG
retrieves question-related knowledge, i.e., a set of facts FfromHand
then generates an answer𝑎(𝑞)based on𝑞andF.Definition 2 (Reasoning Path).Given a knowledge hyper-
graphH=(V,E) , a reasoning path within His defined as a con-
nected sequence of hyperedges, represented as 𝑝𝑎𝑡ℎ(𝑒𝑠,𝑒𝑡)=[𝑒 1=
𝑒𝑠,𝑒2,···,𝑒𝑙−1,𝑒𝑙=𝑒𝑡], where𝑙denotes the length of the path, i.e.,
𝑙=|𝑝𝑎𝑡ℎ(𝑒 𝑠,𝑒𝑡)|.
4 Method
In this section, we introduce the framework of PRoH. Compared
to previous approaches, our model focuses on solving multi-hop
questions based on KHs by generating and dynamically refining
reasoning plans that consist of subquestions of the original question,
and retrieving knowledge guided by reasoning paths in the KHs.
The framework of PRoH consists of four main components. We
outline its workflow in Figure 2.
4.1 Graph Construction and Indexing
KH Construction. We adopt the graph construction method in-
troduced in HyperGraphRAG [ 19]. Given the documents 𝐾, the
method first extracts hyperedges from text chunks, then identifies
entities within these hyperedges, and finally constructs the KH
H=(V,E) . Each entity 𝑣∈V and hyperedge 𝑒∈E is identified
by its name 𝑣𝑛𝑚or𝑒𝑛𝑚, respectively. Each entity is associated with
a textual description 𝑣𝑑𝑒𝑠𝑐, while each hyperedge 𝑒maintains a
reference𝑒𝑟𝑒𝑓to the text chunk that it originates from. For efficient
retrieval, vector databases are maintained for entity names, entity
descriptions, and hyperedge names.
Synonym Hyperedge Augmentation. In the original
method [ 19], entity de-duplication relies on exact name matching,
which results in isolated hyperedges and weakens the connectivity
of the constructed KH. To better utilize graph structure in the
later retrieval stage, inspired by HippoRAG2 [ 12], we introduce
synonym hyperedges to the constructed KH.To be more specific,
the synonym hyperedges are augmented in three steps. First, for
each pair of entities (𝑣𝑖,𝑣𝑗) ∈ V×V , we compute the cosine
similarity
sim(𝑣𝑖,𝑣𝑗)=cos
z(𝑣nm
𝑖),z(𝑣nm
𝑗)
,(1)
wherez(·)denotes the embedding function. We add a similarity
edge𝑒𝑠𝑖𝑚=(𝑣𝑖,𝑣𝑗)if𝑠(𝑣𝑖,𝑣𝑗)≥𝜏 . Second, we form the similarity
subgraphH𝑠𝑖𝑚=(V𝑠𝑖𝑚,E𝑠𝑖𝑚)and compute its connected com-
ponents{𝐶1,𝐶2,...,𝐶𝑚}, where each 𝐶𝑖⊆V𝑠𝑖𝑚. Third, for each
connected component 𝐶𝑖, we query an LLM with the entity de-
scriptions{𝑣𝑑𝑒𝑠𝑐|𝑣∈𝐶𝑖}to determine whether they represent
synonymous entities. If the set of synonymous entities 𝑉𝑠𝑦𝑛⊆𝐶𝑖
is confirmed, we add a synonym hyperedge 𝑒synwhich links all
entities in𝑉 𝑠𝑦𝑛.
4.2 Graph Anchoring
Topic Entity Identification. Given a question 𝑞, we first utilize an
LLM to extract a set of keywords 𝑇𝑤. Each keyword is then linked
to candidate entities in Vby computing cosine similarity scores
as defined in (1). The highest-scoring candidates above a similarity
threshold𝜃 𝑣are collected to form the topic entity setT.
Target Hyperedge Matching. To exploit the semantic information
contained in the question 𝑞, we also retrieve related hyperedges

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Xiangjun Zai et al.
+
+
+ + + +
+DAG 0
DAG 00
DAG 000 DAG 001 DAG 002 DAG 003
DAG 0030 DAG 0020 DAG 0010
DAG 00300State Search
Tree depth  
 dT=1
Subquestion
AnsweringReasoning Path 
SelectionGraph 
Re-anchoring
EWO-based 
KH exploration
YesNoAnswer & Path
Retrieval
 dT=2
 dT=3 dT=0
 dT=4DAG 1
(In Frontier)DAG 2
(In Frontier)
Graph AnchoringQuestion Knowledge Hypergraph
Subgraph
Sketching
Initial Reasoning Plan Generation
and DAG constructionPlan
initialization
ReasoningFinal Answer GenerationSubq 0.aSubquestion
    0.aFound 1 Answer-Path Pair  for Subq 0.a
Found 2 Answer-Path Pairs  for Subq 00.b
       2 Answer-Path Pairs  for Subq 00.cSubq 00.b
Subq 00.cRefinementRefine DAG 0 based on Answer 0.a
Refine DAG 00 based on combinations
of Answer 00.b and Answer 00.c
Refinement Refinement(Initial State)
(Completed)(Completed) (Completed)Refinement Refinement
Refinement Refinement Refinement Refinement
RefinementDAG 0000
(Completed)Answer-Path Pair
Figure 2: An overview architecture of PRoH. Given the input question and KH, the workflow begins with Graph Anchoring
(green) and constructs a question subgraph. The following Plan Initialization Module (purple) sketches the question subgraph
and obtains context to generate initial reasoning plans and constructs reasoning DAGs for the question. Those DAGs serve as
roots of the state search trees in the Reasoning stage (gray). At each state within the State Search Tree, the model retrieves
answer-path pairs from the KH and then iteratively completes and refines the DAGs to transit into the next state until one or
more DAGs are completed. These completed DAGs are then passed to the final answer generation stage (yellow) as the retrieved
knowledge for producing the final answer.
fromE. For each hyperedge, we compute a similarity score between
𝑞and the hyperedge name, following a formulation analogous to
the cosine similarity in (1). The top-ranked hyperedges that satisfy
the threshold𝜃 𝑒form the target hyperedge setR.
Question Subgraph Construction. Once the topic entities T
and target hyperedges Rare identified, we construct a question
subgraph to constrain subsequent retrieval during planning and
reasoning. Specifically, for each 𝑣∈T and𝑒∈R , we extract its
𝐷max-hop neighborhood from H=(V,E) . The question subgraph
H𝑞is defined as the union of these neighborhoods. We also merge
synonymous entities during this subgraph construction phase to
obtain a compact representation of the original KH, which benefits
the subsequent planning and reasoning.
4.3 Reasoning Plan Initialization
For multi-hop questions, directly retrieving graph elements from
the immediate neighborhood of topic entities or target hyperedges
is often insufficient. However, naively expanding to deeper neigh-
borhoods quickly leads to an information explosion, as the number
of reachable entities and hyperedges grows exponentially with
depth. This effect is particularly critical in hypergraphs, where n-
ary relations link multiple entities within one single hyperedge,
allowing one hyperedge to connect to many hyperedges and thereby
rapidly increasing the branching width of the search. Therefore,
to control the knowledge retrieval process and selectively retrieve
only the most relevant and necessary knowledge from the KH, we
introduce the concept of reasoning plans.
Definition 3 (Reasoning Plan).Given a question 𝑞, a reasoning
plan is a structured decomposition of𝑞represented as a pair(Q,L).
Here,Q={𝑞 1,𝑞2,...,𝑞𝑚}denotes the set of subquestions, where each𝑞𝑖addresses a partial aspect of 𝑞, andL⊆Q×Q encodes dependency
constraints among them. The relationLdefines a partial order, that
is, if(𝑞𝑖,𝑞𝑗)∈L, then𝑞 𝑖must be answered before𝑞 𝑗.
Question Subgraph Sketching. While it is possible to generate
reasoning plans directly from the internal knowledge of an LLM,
such plans are often misaligned with the underlying KH. In partic-
ular, the LLM may introduce subquestions that cannot be resolved
because it lacks awareness of domain-specific relations. Moreover,
it is not sufficient to rely solely on the anchored graph elements, i.e.,
topic entitiesTand target hyperedges R, since these elements alone
do not reflect the broader relational structure which is critical for
multi-hop reasoning. To mitigate this issue, we construct a plan con-
text graph that efficiently sketches the structure of the hypergraph.
This is achieved by treating topic entities and target hyperedges as
seeds for controlled exploration. The resulting subgraph provides
explicit grounding for plan generation and improves the alignment
between the reasoning plan and the available knowledge.
Definition 4 (Plan Context Graph).Given a question 𝑞, a KH
H=(V,E) , the question subgraph H𝑞=(V𝑞,E𝑞), topic entity set
T⊆V , target hyperedge set R⊆E , and a plan depth 𝑑𝑝, the plan
context graphH𝑝=(V𝑝,E𝑝)is defined as a subgraph of H, where
V𝑝andE𝑝include entities and hyperedges that are highly relevant
to𝑞and are within the𝑑 𝑝-hop neighborhood of eitherTorR.
We initialize the plan graph H𝑝=(V𝑝,E𝑝)using target hyper-
edgesRand the hyperedges incident to the topic entities T. These
hyperedges also serve as the initial frontier for exploration. To guide
the search direction, we employ a hyperedge scoring strategy. For
each frontier hyperedge 𝑒, we first compute entity-level relevance
scores for all entities 𝑣∈𝑒 with respect to the question 𝑞, using a

PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
function on its description and the question:
SE(𝑣|𝑞)=cos
z(𝑣desc),z(𝑞)
,(2)
wherez(·)denotes the embedding function. For each neighboring
hyperedge𝑒′∈Nbr(𝑒) , we then aggregate the scores of the overlap-
ping entities 𝑉(𝑒)∩𝑉(𝑒′)to obtain an hyperedge-level relevance
score with respect to𝑞and𝑒:
SH(𝑒′|𝑞, 𝑒)=Aggregate
{SE(𝑣,𝑞)|𝑣∈𝑉(𝑒)∩𝑉(𝑒′)}
.(3)
Based on the hyperedge-level relevance scores, low-scoring direc-
tions (hyperedges) are pruned, and the exploration will focus on
directions that are supported by highly relevant entities.
Initial Reasoning Plan Generation. After constructing the plan
context graphH𝑝, we leverage it as a structured input for the LLM
to propose reasoning plans. We transfer the graph structure into a
natural language plan context by augmenting the hyperedges layer
by layer, from the nearest neighborhood of the anchored graph
elements to the more distant neighborhoods. This ensures that the
context reflects local relevance and progressively broader structural
information. Formally, given the plan context graph H𝑝, the plan
context𝑐𝑝is defined as 𝑐𝑝=FormPlanContext(H 𝑝,T,R) , where
FormPlanContext denotes the procedure that extracts and formats
the relevant subgraph structure into plan context. The LLM is then
prompted with the question 𝑞, the topic entities T, and the plan
context𝑐𝑝to generate initial reasoning plans.
Initial Reasoning DAG construction. Given a reasoning plan
(Q,L) , the dependency relation Lmay contain transitive edges. To
obtain a minimal representation, we apply a Hasse Reduction that
removes all dependency relations that can be transitively covered.
That is, if(𝑞𝑖,𝑞𝑗)∈L and there exists 𝑞𝑘∈Qsuch that(𝑞𝑖,𝑞𝑘)∈L
and(𝑞𝑘,𝑞𝑗)∈L , then(𝑞𝑖,𝑞𝑗)is considered redundant and excluded
fromL𝐻. The reduced relation L𝐻captures only the immediate
dependencies between subquestions.
Definition 5 (Reasoning DAG).A reasoning DAG is the graph
abstraction of a reasoning plan (Q,L) . It is defined as a directed
acyclic graph (DAG) 𝐷=(Q,L 𝐻), where each node represents a
subquestion 𝑞𝑖∈Q and each directed edge (𝑞𝑖,𝑞𝑗)∈L𝐻encodes the
immediate dependency between𝑞 𝑖and𝑞𝑗.
For each reasoning plan, we construct a corresponding reasoning
DAG. We then apply a topological sort on the reduced dependency
relationL𝐻to obtain an execution order over the subquestions.
This order will guide the level-by-level completion of the reasoning
DAG. The processed reasoning DAGs are collected to form the initial
reasoning DAG set D0. Due to space limitations, the pseudo-code
of the plan initialization algorithm is provided in Appendix A.1.
4.4 Reasoning
Once the initial reasoning DAGs are constructed, the next step
is to query the KH under their guidance. More specifically, for a
reasoning DAG 𝐷, questions at the first level without dependencies
are resolved first. The retrieved answers are used to refine the 𝐷,
and the questions of the next level will be unlocked for reasoning.
This iterative process repeats until all subquestions are answered.
The step answers to subquestions and supporting knowledge arestored in the completed reasoning DAG. We refer to this process
of progressively resolving and refining DAGs as reasoning. Due to
space limitations, the pseudo-code of the reasoning algorithm is
provided in Appendix A.2.
Definition 6 (Completed Reasoning DAG).A completed rea-
soning DAG is defined as a DAG 𝐷comp=(Q,L𝐻,AP) , where each
node𝑞𝑗∈Q is associated with a non-empty set of answer-path pairs
AP[𝑞𝑗]={(𝑎𝑗,𝑝𝑗)}. Here,𝑎𝑗is a candidate answer to 𝑞𝑗and𝑝𝑗is
a supporting reasoning path inH 𝑞.
Reasoning as State-Space Search. The aforementioned reasoning
process can be viewed as a structured search problem over DAG
states. A reasoning state represents a partially completed reasoning
DAG. It captures both the current reasoning DAG and the current
progress in resolving its subquestions. The initial state corresponds
to an initial reasoning DAG 𝐷∈D 0with no subquestions resolved.
The goal state is a completed reasoning DAG. Formally, we define
a reasoning state as follows.
Definition 7 (Reasoning State).A reasoning state is a pair
(𝐷,𝑖) , where𝐷=(Q,L 𝐻,AP) is a reasoning DAG and 𝑖is the index
of the current reasoning level. A transition from (𝐷,𝑖) to(𝐷′,𝑖+1)
occurs once all subquestions in the 𝑖-th level of𝐷, denoted asQ𝑖, are
resolved with non-empty sets of answer–path pairsAP 𝑖.
Now, we formulate the reasoning process as a search over states
and perform it using Depth-First Search (DFS). The frontier Fis ini-
tialized with the set of initial reasoning DAGs D0. At each iteration,
a state(𝐷,𝑖) is popped fromF. If𝐷is incomplete, the subquestions
Q𝑖at the current level 𝑖are attempted. Each subquestion is resolved
by retrieving reasoning paths from the anchored graph elements
in the query subgraph H𝑞, within a KH exploration depth limit
𝑑𝑚𝑎𝑥. Details of the retrieval procedure are given in the following
subsection. The retrieved answer–path pairs are then used to gen-
erate successor states. These successor states are pushed back into
the frontier for later iterations. If 𝐷is complete, it is added to the
solution setDcomp . The search terminates once |Dcomp|≥𝐾 or
the frontier is empty. Formally, the search procedure is summarized
asD comp=Reasoning(D 0,H𝑞,𝑑max,𝐾).
State Transitions. Given a state (𝐷,𝑖) , the sets of answer–path
pairsAP𝑖at level𝑖, we illustrate the state transition process as
follows. Since each subquestion 𝑞𝑗∈Q𝑖is associated with a set
of candidate assignments in the form of answer-path pairs 𝐴𝑃𝑗,
a valid joint assignment for level 𝑖can be obtained by selecting
one answer-path pair (𝑎𝑗,𝑝𝑗)∈𝐴𝑃𝑗for every𝑞𝑗∈Q𝑖. If multi-
ple joint assignments exist, the current state branches accordingly.
For each joint assignment AP, a successor reasoning DAG is con-
structed as𝐷new=LLMGenerateNewDAG (𝐷,AP). Here, the LLM
is prompted with the current set of subquestions Qtogether with
one candidate answer for each completed subquestion. A refined
reasoning plan is proposed by the LLM, which is then validated
against the existing reasoning DAG 𝐷, and then merged with 𝐷to
produce the successor reasoning DAG𝐷 new.
4.5 Answer and Path Retrieval
The core subroutine for resolving subquestions is the retrieval of
answer–path pairs from the question subgraph H𝑞=(V𝑞,E𝑞).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Xiangjun Zai et al.
We employ an iterative deepening beam search that progressively
exploresH𝑞. At each depth level, the subquestion is attempted using
the knowledge retrieved based on a reasoning path discovered so far.
The process terminates once a reasoning path provides sufficient
knowledge to answer the subquestion. Due to space limitations, we
detail the algorithm of answer and path retrieval in Appendix A.3.
Graph Re-anchoring. For a subquestion 𝑞𝑗, the relevant knowl-
edge should be within a more concise subregion of the question
subgraphH𝑞, where the topic entities Tand target hyperedges R
of the original question provide weaker guidance for the search. To
account for this, we re-anchor the graph by identifying the topic en-
titiesT𝑗and target hyperedges R𝑗specific to𝑞𝑗, following the same
procedure described in Section 4.2. These graph elements serve as
seeds for the subsequent beam search, constraining the exploration
ofH𝑞to the regions most relevant to 𝑞𝑗. Based onT𝑗andR𝑗, we
initialize the search frontier F𝑒with the target hyperedges R𝑗and
the hyperedges incident to the topic entities T𝑗. The key challenge
is then to guide the search toward discovering knowledge most
relevant to the question 𝑞𝑗. To address this challenge, we design a
strategy tailored to KHs.
EWO-based Search Direction Selection. In standard graphs, two
adjacent edges can share at most a single node, in contrast, hyper-
edges in a hypergraph may overlap on multiple entities. Moreover,
the contribution of these overlapping entities to answering a ques-
tion is not uniform: some are irrelevant, while others provide critical
evidence. Thus, neighboring hyperedges should neither be treated
as equally relevant only because an overlap exists, nor should their
relevance be determined solely by the number of shared entities.
To address this, we propose theEntity-Weighted Overlap (EWO)
score, a fine-grained strategy in which the relevance of a hyperedge
is computed by aggregating the question-specific relevance of its
overlapping entities. We now detail the 2-step computation of the
EWO score.
Entity scoring. Each overlapping entity 𝑣∈𝑉(𝑒)∩𝑉(𝑒′)is first
assigned a provisional relevance score with respect to 𝑞𝑗using the
embedding-based similarity defined in (2). Entities with similarity
scores above the threshold 𝜃embare retained and further evaluated
by the LLM to obtain a finer-grained relevance score. Entities with
similarity scores below the threshold 𝜃embare assigned a relevance
score of0. This score reflects how semantically relevant 𝑣is for𝑞𝑗.
EW(𝑣|𝑞 𝑗)= 
LLMScore(𝑣,𝑞 𝑗),ifSE(𝑣|𝑞 𝑗)≥𝜃 emb,
0,otherwise.(4)
Hyperedge scoring. Entity scores are then aggregated to produce
a hyperedge-level score for each neighbor 𝑒′. This score reflects
how semantically relevant 𝑒′is as a potential bridge for a partial
reasoning path to answering𝑞 𝑗.
EWO(𝑒′|𝑞, 𝑒)=Aggregate
{𝑆𝐸(𝑣,𝑞)|𝑣∈𝑉(𝑒)∩𝑉(𝑒′)}
.(5)
With the EWO score, we now determine where to expand the search
within the candidate search directions (partial reasoning paths)
𝐹cand. In the first stage, 𝐹candis ordered according to the hyperedge-
level EWO scores. From the resulting top-ranked directions, we
then apply an LLM-based selection function to select the top- 𝑏di-
rections:𝐹sel=LLMSelectDirections(𝐹 cand,𝑞𝑗,𝑏). This evaluatesthe partial reasoning paths in context, rather than relying solely
on the EWO score of the terminal hyperedge.
Reasoning Path Selection. At each depth, we construct a set of
candidate reasoning paths 𝑃candfrom the partial paths Pexplored
so far. Each candidate reasoning path is then ranked using a path-
level relevance score that aggregates the EW scores of entities along
the path. Formally, the path-level score is defined as:
SP(𝑝)=Aggregate
{EW(𝑣|𝑞 𝑗) |𝑣∈𝑒∧𝑒∈𝑝}
.(6)
The top-ranked candidate reasoning paths are then passed to
the LLM, to determine whether one or more of them provide
sufficient knowledge to yield a step answer for 𝑞𝑗:𝑃sel =
LLMSelectPaths(𝑃 cand,𝑞𝑗).
Step Answer for Subquestion. If 𝑃selcontains valid reasoning
paths, we attempt to answer subquestion 𝑞𝑗. For each selected path
𝑝𝑗, besides the path itself, we also retrieve the descriptions of the
entities covered by the path, and, following existing work, the text
chunks from which the hyperedges originate. These three compo-
nents together form the context for 𝑞𝑗, which is then provided to the
LLM to produce a step answer 𝑎𝑗:𝑎𝑗=LLMAnswerStep(𝑝 𝑗,𝑞𝑗).
The answer and path retrieval procedure terminates once 𝑃selhas
been fully processed, and no further exploration beyond the current
depth will be performed.
4.6 Final Answer Generation
As discussed in Section 4.4, the reasoning module produces a so-
lution setDcomp, where multiple reasoning plans (DAGs) are exe-
cuted to completion and all sub-questions within those plans are
answered. For each completed DAG 𝐷comp=(Q,L𝐻,AP) , our
system aggregates the retrieved knowledge along its reasoning pro-
cess followingL𝐻and uses this aggregated knowledge to generate
a candidate answer 𝑎(𝑞) to the original question 𝑞. Since different
reasoning plans may yield redundant or overlapping answers, we
introduce an LLM-based evaluation agent that acts as a judge to
aggregate and assess these candidate answers A(𝑞) . This judge
evaluates each candidate answer 𝑎(𝑞) according to its consistency
with the corresponding reasoning path, ultimately selecting the
top-ranked answer as the final answer𝑎∗(𝑞).
5 Experiments
Experimental Settings. We evaluate PRoH on the KHQA datasets
introduced in HyperGraphRAG [ 19], which span five knowledge
domains: Medicine, Agriculture, CS, Legal, and Mix. Since the ques-
tions in the KHQA datasets are constructed from sampled knowl-
edge fragments located only 1–3 hops away. We further extend
these datasets with long-range questions to better assess the multi-
hop reasoning capability of PRoH. Specifically, we generate 200
additional questions per domain using knowledge fragments 3–6
hops away. We also develop and evaluate PRoH-L, a variant of PRoH
that employs a fully embedding-based EWO score and uses only
the hyperedges along the reasoning paths as context for answer
generation. Following [ 19], we adopt three evaluation metrics: F1,
Retrieval Similarity (R-S), and Generation Evaluation (G-E). Due
to the space limitation, experiment details, including baselines and
implementation details, are provided in Appendix B.

PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 1: Results of PRoH across different domains, compared with the state-of-the-art (SOTA). Bold denotes the best
performance and underline denotes the runner-up.
Method Medicine Agriculture CS Legal Mix
F1 R-S G-E F1 R-S G-E F1 R-S G-E F1 R-S G-E F1 R-S G-E
LLM-only 12.89 0 43.27 12.74 0 46.85 18.65 0 48.87 21.64 0 49.05 16.93 0 45.65
StandardRAG 27.90 62.57 55.66 27.43 45.81 57.10 28.93 48.40 56.89 37.34 51.68 59.97 43.20 47.26 64.62
PathRAG 14.94 53.19 44.06 21.30 42.37 52.48 26.73 41.89 54.13 31.29 44.03 55.36 37.07 33.73 59.11
HippoRAG2 21.34 59.52 49.57 12.63 18.58 44.85 17.34 23.99 47.87 18.53 34.42 45.93 21.53 18.42 46.35
HyperGraphRAG 35.35 70.19 59.35 33.89 62.27 59.79 31.30 60.09 57.94 43.81 60.47 63.61 48.7168.2166.90
PRoH-L 45.63 70.84 59.90 50.47 64.2863.13 46.61 60.7960.17 51.40 64.58 63.71 53.81 59.32 61.32
PRoH 52.94 74.02 67.35 56.67 58.88 69.46 54.15 57.72 66.79 58.81 65.22 69.88 69.16 59.86 76.17
Table 2: Ablation study results (F1).
Method Agriculture CS Mix
PRoH 58.49 59.47 69.39
w/o EWO Guide 53.22 56.27 65.63
w/o Synonym Merge 53.26 54.96 64.74
w/o Plan Context 53.70 53.67 64.59
w/o Src Chunks 53.55 54.75 63.76
w/o Target Hyperedge 53.27 55.60 60.81
w/o ALL 51.93 51.96 55.84
5.1 Main Results
(RQ1) Does PRoH outperform other methods?As shown in
Table 1, PRoH achieves state-of-the-art performance across all do-
mains in terms of F1 and G-E scores, outperforming the previous
SOTA baseline HyperGraphRAG by an average of 19.73% and up
to 22.85% in F1 on the CS domain, as well as by an average of
8.41% and up to 9.67% in G-E. For the R-S score, PRoH generally
achieves comparable results to HyperGraphRAG, with up to a 4.75%
improvement in the Legal domain. The main weakness appears in
the Mix domain, which is reasonable since it integrates knowledge
from multiple domains. Unlike HyperGraphRAG, which prioritizes
retrieving text with high semantic similarity, PRoH retrieves knowl-
edge that contributes directly to reasoning toward the answer—even
when such knowledge is not semantically similar to the surface
context. Notably, the Mix domain is also where PathRAG, another
reasoning-path-based retrieval method, attains its lowest R-S score,
indicating a similar behavior pattern. For the variant PRoH-L, it sur-
passes HyperGraphRAG by an average of 10.97% and up to 16.58%
in F1 on the Agriculture domain. It also remains competitive with
HyperGraphRAG in terms of both G-E and R-S scores, showing
up to a 3.34% improvement in the Agriculture domain and a 4.11%
improvement in the Legal domain, respectively.
5.2 Ablation Study
(RQ2) Does the main component of PRoH work effectively?
As shown in Table 2, we conduct an ablation study across three
domains to quantify the contribution of each module. From each
domain, Agriculture, CS and Mix, we randomly sample 200 ques-
tions and report the F1 score for comparison. The results show that
removing the EWO Guide Direction Selection decreases F1 by up toTable 3: #Token per Question across Domains.
Domain HyperGraphRAG PRoH-L #Token%↓F1↑
Medicine 21,112 19,7326.54% 10.28
Agriculture 17,914 12,52830.07% 16.58
CS 18,666 12,16634.82% 15.31
Legal 22,086 28,831−30.54%7.59
Mix 13,856 9,68730.09%5.10
2 3 4 5
dmax505560657075Score
F1 G-E
(a) F1 & G-E vs.𝑑 max
2 3 4 5
dmax1.01.52.02.53.0davg
Average Actual Depth davg (b)𝑑 𝑎𝑣𝑔vs.𝑑 max
Figure 3: Impact of depth limit𝑑 maxon performance.
5.3%, showing its effectiveness in guiding the reasoning path search
based on semantic flow from one hyperedge to another. Removing
Synonym Merge reduces F1 by up to 5.2%, indicating that this graph
reduction technique benefits the planning and reasoning. Excluding
Plan Context leads to a reduction of up to 5.8%, confirming that
context for planning improves the feasibility and alignment of the
initial reasoning plan. Without Source Chunks, F1 declines by up
to 5.6%, suggesting that the source text provides additional con-
text that contributes to more accurate answers. Eliminating Target
Hyperedge Matching in graph anchoring yields the largest drop,
up to 8.6%, demonstrating the importance of leveraging semantics
encoded in hyperedges. When all modules mentioned above are
removed (w/o ALL), performance drops sharply by up to 13.6%,
indicating their complementary contributions.
(RQ3) How does the KH exploration depth affect the perfor-
mance of PRoH?PRoH performs dynamic exploration on KH
within the depth limit 𝑑max. To assess the impact of 𝑑maxon per-
formance, we conducted experiments on 200 randomly sampled
questions from the CS domain. We vary 𝑑maxfrom 2 to 5 and re-
port the corresponding F1 and G-E scores in Figure 3(a). We also
collect the actual exploration depth when the step answer for each

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Xiangjun Zai et al.
Table 4: Performance of PRoH and baselines on long-range Multi-hop QA tasks across domains. Bold denotes the best
performance and underline denotes the runner-up.
Method Medicine Agriculture CS Legal Mix
F1 R-S G-E F1 R-S G-E F1 R-S G-E F1 R-S G-E F1 R-S G-E
IO (LLM-only) 23.36 0 0 30.95 0 0 39.27 0 0 43.20 0 0 48.98 0 0
COT (LLM-only) 22.70 0 43.42 34.32 0 52.22 42.41 0 58.16 43.36 0 57.71 50.48 0 62.24
SC (LLM-only) 26.06 0 44.80 37.49 0 53.84 46.42 0 60.34 40.59 0 55.36 54.04 0 63.16
StandardRAG 21.90 66.02 50.13 48.00 50.92 67.81 27.34 59.42 57.11 35.86 55.25 60.37 52.20 59.79 69.91
HyperGraphRAG 39.94 71.72 60.81 52.40 64.36 70.76 29.9568.7858.72 38.45 58.30 61.87 64.55 70.2376.63
PRoH-L 47.30 71.24 60.98 70.02 65.7574.44 62.11 65.09 69.08 60.25 59.21 67.78 61.19 62.48 66.16
PRoH 51.26 75.11 65.23 81.01 62.22 84.18 74.83 68.44 80.00 71.93 64.89 77.68 79.69 68.45 82.25
1 2 3 4 6
Context Length0255075100F1 ScoreMedicine
AgricultureCS
LegalMix
Figure 4: F1 score vs. length of ground-truth context.
2 3 4 5 6 7 8 9 10 overall
Number of Entities0255075100F1 ScoreAgriculture Mix
Figure 5: F1 score vs. #entities in ground-truth context.
subquestion is found and the exploration terminates. As shown
in Figure 3(a), increasing 𝑑maxinitially improves both F1 and G-E
scores, with performance peaking at 𝑑max=3. Beyond this depth, a
deeper search even introduces a slight degradation in both metrics.
This suggests that the additional search depth does not uncover
additional useful information and instead introduces redundant
context that fails to improve reasoning quality.
The trend in the actual exploration depth supports this interpre-
tation. As𝑑maxincreases from 2 to 5, the average actual depth 𝑑𝑎𝑣𝑔
grows modestly from 1.41 (at 𝑑max=2) to 2.30 (at 𝑑max=5). This
indicates that PRoH rarely needs to utilize the full depth budget;
most subquestions are resolved within a relatively short reasoning
path. This behavior may have two main reasons. 𝑖)PRoH decom-
poses questions and dynamically refines its reasoning plan, thus
simplifying each subquestion and shortening the reasoning path.
𝑖𝑖)The EWO-guided exploration helps the system identify relevant
paths early, minimizing unnecessary exploration.
5.3 Effectiveness and Efficiency Evaluation
(RQ4) Does PRoH stay effective on long-range multi-hop
questions?We further evaluate PRoH on 200 additional long-rangequestions per domain (3–6 hops). As shown in Table 4, PRoH sus-
tains strong performance under these long-range settings, outper-
forming HyperGraphRAG by an average of 26.68% and up to 44.87%
in F1 in the CS domain, as well as by an average of 12.11% and
up to 21.28% in G-E. For the R-S score, PRoH achieves an average
of 1.14% and up to 6.59% improvement in the Legal domain. The
variant PRoH-L, also demonstrates strong performance under these
long-range settings, outperforming HyperGraphRAG by an average
of 15.11% and up to 32.15% in F1 score. The robustness of PRoH is
also supported by the results in Figure 4, which analyzes the effect
of ground truth context length (the number of knowledge frag-
ments used when the question and golden answer are generated)
on PRoH ’s performance. The F1 scores remain stable as the con-
text length grows, suggesting that PRoH can maintain reasoning
coherence even when the relevant knowledge spans multiple hops,
demonstrating its robustness in long-range, multi-hop reasoning.
(RQ5) How does PRoH perform with multiple entities in the
ground-truth context?Figure 5 reports the average F1 scores of
PRoH across different levels of relational complexity in the ques-
tions. The complexity is measured by the number of entities partic-
ipating in the ground-truth context for the question. Overall, PRoH
maintains stable performance as the number of entities increases,
which suggests that PRoH effectively handles moderately complex
relational structures.
(RQ6) Is PRoH-L cost efficient in token usage?As shown in
Table 3, PRoH-L demonstrates notable efficiency in token usage
while maintaining competitive performance across all domains.
Compared with HyperGraphRAG, PRoH-L significantly reduces
the number of tokens used (input + output) per question, with the
largest savings observed in the Computer Science domain at 34.82%.
Despite the reduced token budget, PRoH-L achieves consistent F1
improvements with up to 16.58% in Agriculture. The only exception
appears in the Legal domain, where token consumption increases
for PRoH-L; however, this increase still yields a positive F1 gain
of 7.59%. Overall, these results confirm that PRoH-L achieves a
superior balance between efficiency and accuracy, offering a cost-
effective alternative to full PRoH.
To further evaluate the effectiveness and efficiency of PRoH, we
conduct additional experiments on its state search strategy and plan
depth, and analyze the token usage across modules. We also con-
ducted a case study on PRoH’s structured question decomposition.
Detailed results are provided in Appendix C.

PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
6 Conclusion
This paper presents PRoH, a dynamic Knowledge Hyper-
graph–based RAG framework for multi-hop question answering.
By introducing context-aware planning, structured iterative ques-
tion decomposition and an Entity-Weighted Overlap(EWO)–guided
reasoning path retrieval strategy, PRoH enables adaptive planning
and reasoning on Knowledge Hypergraphs with beyond binary
relational structures. Experimental results demonstrate that PRoH
achieves state-of-the-art performance across multiple knowledge
domains, surpassing the prior SOTA HyperGraphRAG by an av-
erage of 19.73% in F1 and 8.41% in Generation Evaluation (G-E)
score, while maintaining high robustness in long-range multi-hop
reasoning tasks.
References
[1]Kurt D. Bollacker, Colin Evans, Praveen K. Paritosh, Tim Sturge, and Jamie Taylor.
2008. Freebase: a collaboratively created graph database for structuring human
knowledge. InSIGMOD Conference. ACM, 1247–1250.
[2]Boyu Chen, Zirui Guo, Zidan Yang, Yuluo Chen, Junze Chen, Zhenghao Liu,
Chuan Shi, and Cheng Yang. 2025. PathRAG: Pruning Graph-based Retrieval
Augmented Generation with Relational Paths.CoRRabs/2502.14902 (2025).
[3]Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, and Hui Xiong.
2024. Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language
Model on Knowledge Graphs. InAdvances in Neural Information Processing Sys-
tems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS
2024, Vancouver, BC, Canada, December 10 - 15, 2024, Amir Globersons, Lester
Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and
Cheng Zhang (Eds.).
[4]Chengfeng Dou, Ying Zhang, Zhi Jin, Wenpin Jiao, Haiyan Zhao, Yongqiang
Zhao, and Zhengwei Tao. 2025. Enhancing LLM Generation with Knowledge
Hypergraph for Evidence-Based Medicine.CoRRabs/2503.16530 (2025).
[5]Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[6]Bahare Fatemi, Perouz Taslakian, David Vázquez, and David Poole. 2020. Knowl-
edge Hypergraphs: Prediction Beyond Binary Relations. InIJCAI. ijcai.org, 2191–
2197.
[7]Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shihui Ying, Shaoyi Du,
Han Hu, and Yue Gao. 2025. Hyper-RAG: Combating LLM Hallucinations
using Hypergraph-Driven Retrieval-Augmented Generation.arXiv preprint
arXiv:2504.08758(2025).
[8]Junqi Gao, Xiang Zou, Ying Ai, Dong Li, Yichen Niu, Biqing Qi, and Jianxing Liu.
2025. Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to
Enhance LLM Reasoning. InACL (1). Association for Computational Linguistics,
24650–24668.
[9]Saiping Guan, Xiaolong Jin, Yuanzhuo Wang, and Xueqi Cheng. 2019. Link
Prediction on N-ary Relational Data. InWWW. ACM, 583–593.
[10] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. Lightrag:
Simple and fast retrieval-augmented generation.arXiv preprint arXiv:2410.05779
(2024).
[11] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su.
2024. Hipporag: Neurobiologically inspired long-term memory for large language
models.Advances in Neural Information Processing Systems37 (2024), 59532–
59569.
[12] Bernal Jimenez Gutierrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025.
From rag to memory: Non-parametric continual learning for large language
models.arXiv preprint arXiv:2502.14802(2025).
[13] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi
He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, and
Jiliang Tang. 2025. Retrieval-Augmented Generation with Graphs (GraphRAG).
CoRRabs/2501.00309 (2025).
[14] Jinhao Jiang, Kun Zhou, Xin Zhao, and Ji-Rong Wen. 2023. UniKGQA: Uni-
fied Retrieval and Reasoning for Solving Multi-hop Question Answering Over
Knowledge Graph. InICLR. OpenReview.net.
[15] Xinke Jiang, Ruizhe Zhang, Yongxin Xu, Rihong Qiu, Yue Fang, Zhiyuan Wang,
Jinyi Tang, Hongxin Ding, Xu Chu, Junfeng Zhao, and Yasha Wang. 2025. HyKGE:
A Hypothesis Knowledge Graph Enhanced RAG Framework for Accurate and
Reliable Medical LLMs Responses. InACL (1). Association for Computational
Linguistics, 11836–11856.[16] Jiho Kim, Yeonsu Kwon, Yohan Jo, and Edward Choi. 2023. KG-GPT: A General
Framework for Reasoning on Knowledge Graphs Using Large Language Models.
InFindings of the Association for Computational Linguistics: EMNLP 2023.
[17] Xujian Liang and Zhaoquan Gu. 2025. Fast Think-on-Graph: Wider, Deeper and
Faster Reasoning of Large Language Model on Knowledge Graph. InAAAI. AAAI
Press, 24558–24566.
[18] Runxuan Liu, Luobei Luobei, Jiaqi Li, Baoxin Wang, Ming Liu, Dayong Wu, Shijin
Wang, and Bing Qin. 2025. Ontology-Guided Reverse Thinking Makes Large
Language Models Stronger on Knowledge Graph Question Answering. InACL
(1). Association for Computational Linguistics, 15269–15284.
[19] Haoran Luo, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin,
Yu Feng, Zemin Kuang, Meina Song, Yifan Zhu, et al .2025. HyperGraphRAG:
Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Repre-
sentation.arXiv preprint arXiv:2503.21322(2025).
[20] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning
on Graphs: Faithful and Interpretable Large Language Model Reasoning. InICLR.
OpenReview.net.
[21] Linhao Luo, Zicheng Zhao, Chen Gong, Gholamreza Haffari, and Shirui Pan. 2024.
Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with
Large Language Models.CoRRabs/2410.13080 (2024).
[22] Jie Ma, Zhitao Gao, Qi Chai, Wangchun Sun, Pinghui Wang, Hongbin Pei, Jing
Tao, Lingyun Song, Jun Liu, Chen Zhang, and Lizhen Cui. 2025. Debate on Graph:
A Flexible and Reliable Reasoning Framework for Large Language Models. In
AAAI. AAAI Press, 24768–24776.
[23] Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaren Qu, Cehao Yang, Jiaxin
Mao, and Jian Guo. 2025. Think-on-Graph 2.0: Deep and Faithful Large Language
Model Reasoning with Knowledge-guided Retrieval Augmented Generation. In
ICLR. OpenReview.net.
[24] Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph Neural Retrieval
for Efficient Large Language Model Reasoning on Knowledge Graphs. InACL
(Findings). Association for Computational Linguistics, 16682–16699.
[25] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong,
Yan Zhang, and Siliang Tang. 2024. Graph Retrieval-Augmented Generation: A
Survey.CoRRabs/2408.08921 (2024).
[26] Paolo Rosso, Dingqi Yang, and Philippe Cudré-Mauroux. 2020. Beyond Triplets:
Hyper-Relational Knowledge Graph Embedding for Link Prediction. InWWW.
ACM / IW3C2, 1885–1896.
[27] Apoorv Saxena, Aditay Tripathi, and Partha P. Talukdar. 2020. Improving Multi-
hop Question Answering over Knowledge Graphs using Knowledge Base Em-
beddings. InACL. Association for Computational Linguistics, 4498–4507.
[28] Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun
Gong, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-Graph:
Deep and Responsible Reasoning of Large Language Model on Knowledge Graph.
InThe Twelfth International Conference on Learning Representations, ICLR 2024,
Vienna, Austria, May 7-11, 2024. OpenReview.net.
[29] Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, and Wenjie Zhang.
2025. Paths-over-Graph: Knowledge Graph Empowered Large Language Model
Reasoning. InWWW. ACM, 3505–3522.
[30] Denny Vrandecic and Markus Krötzsch. 2014. Wikidata: a free collaborative
knowledgebase.Commun. ACM57, 10 (2014), 78–85. doi:10.1145/2629489
[31] Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, and Ning Jiang. 2025.
Cross-Granularity Hypergraph Retrieval-Augmented Generation for Multi-hop
Question Answering.CoRRabs/2508.11247 (2025).
[32] Jianfeng Wen, Jianxin Li, Yongyi Mao, Shini Chen, and Richong Zhang. 2016. On
the Representation and Embedding of Knowledge Bases beyond Binary Relations.
InIJCAI. IJCAI/AAAI Press, 1300–1307.
[33] Junde Wu, Jiayuan Zhu, Yunli Qi, Jingkun Chen, Min Xu, Filippo Menolascina,
Yueming Jin, and Vicente Grau. 2025. Medical Graph RAG: Evidence-based
Medical Large Language Model via Graph Retrieval-Augmented Generation. In
ACL (1). Association for Computational Linguistics, 28443–28467.
[34] Derong Xu, Xinhang Li, Ziheng Zhang, Zhenxi Lin, Zhihong Zhu, Zhi Zheng,
Xian Wu, Xiangyu Zhao, Tong Xu, and Enhong Chen. 2025. Harnessing large
language models for knowledge graph question answering via adaptive multi-
aspect retrieval-augmentation. InProceedings of the Thirty-Ninth AAAI Conference
on Artificial Intelligence and Thirty-Seventh Conference on Innovative Applications
of Artificial Intelligence and Fifteenth Symposium on Educational Advances in
Artificial Intelligence.
[35] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou,
Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, and Xiao Huang. 2025. A Sur-
vey of Graph Retrieval-Augmented Generation for Customized Large Language
Models.CoRRabs/2501.13958 (2025).
[36] Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander J. Smola, and Le Song.
2018. Variational Reasoning for Question Answering With Knowledge Graph. In
AAAI. AAAI Press, 6069–6076.
[37] Qi Zhao, Hongyu Yang, Qi Song, Xin-Wei Yao, and Xiangyang Li. 2025. Know-
Path: Knowledge-enhanced Reasoning via LLM-generated Inference Paths over
Knowledge Graphs.CoRRabs/2502.12029 (2025).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Xiangjun Zai et al.
[38] Ruilin Zhao, Feng Zhao, Long Wang, Xianzhi Wang, and Guandong Xu. 2024. KG-
CoT: Chain-of-Thought Prompting of Large Language Models over Knowledge
Graphs for Knowledge-Aware Question Answering. InProceedings of the Thirty-
Third International Joint Conference on Artificial Intelligence, IJCAI-24.
[39] Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge
Graph-Guided Retrieval Augmented Generation. InProceedings of the 2025 Con-
ference of the Nations of the Americas Chapter of the Association for Computational
Linguistics: Human Language Technologies (Volume 1: Long Papers).
[40] Zulun Zhu, Tiancheng Huang, Kai Wang, Junda Ye, Xinghe Chen, and Siqiang
Luo. 2025. Graph-based Approaches and Functionalities in Retrieval-Augmented
Generation: A Comprehensive Survey.CoRRabs/2504.10499 (2025).

PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Algorithm
A.1 Plan Initialization
We summarize the comprehensive algorithmic procedure of rea-
soning plan initialization as shown in Algorithm 1.
Algorithm 1: PlanInit
Input: Question Subgraph H𝑞(V𝑞,E𝑞), Question𝑞, Topic Entities
T, Target HyperedgesR, Plan Depth𝑑 𝑝, #Initial Plans𝑛 0
,Output: Set of initial DAGsD 0 1
H𝑝(V𝑝,E𝑝)←(∅,∅);2
F𝑒←{𝑒|𝑣∈𝑒∧𝑣∈T}∪R;3
for𝑑←1to𝑑 𝑝do4
F′
𝑒←∅;5
for each𝑒∈F 𝑒do6
𝐹cand←∅;7
for each𝑒′∈Nbr(𝑒)do8
𝑆𝑣={ScoreEntity(𝑣,𝑞 𝑗)|𝑣∈𝑉(𝑒)∩𝑉(𝑒′)}; 9
𝑠𝑒′←Aggregate(𝑆 𝑣);10
𝐹cand←𝐹 cand∪{(𝑒′,𝑠𝑒′)};11
𝐹sel←RankSelectDirections(𝐹 cand);12
F′
𝑒←F′
𝑒∪𝐹sel;13
V𝑝←V 𝑝∪{𝑣|𝑣∈𝑒∧𝑒∈𝐹 sel};14
E𝑝←E 𝑝∪𝐹sel;15
F𝑒←F′
𝑒;16
𝑐𝑝←FormPlanContext(H 𝑝);D 0←∅;17
for𝑖←1to𝑛 0do18
Q,L← LLM(𝑞,T,𝑐 𝑝);𝐷← TopSort(Q,L) ;D0←D 0∪{𝐷} ; 19
A.2 Reasoning
We summarize the comprehensive algorithmic procedure of rea-
soning as shown in Algorithm 2.
Algorithm 2: Reasoning
Input: Initial DAGsD 0, Question SubgraphH 𝑞, KH Exploration
Depth Limit𝑑 max, Max #Solutions𝐾
Output: Completed DAGsD comp
Dcomp←∅;1
F←D 0;2
whileF≠∅and|D comp|<𝐾do3
𝐷←F.𝑝𝑜𝑝();4
𝑖←𝐷.completed_level+1;5
if𝑖≥|𝐷.levels|then6
Dcomp←D comp∪{𝐷};7
continue8
Q𝑖←subquestions at level𝑖of𝐷;9
AP 𝑖←∅;10
for each𝑞 𝑗∈Q 𝑖do11
𝐴𝑃𝑗←RetrieveAnswersWithPaths(H 𝑞,𝑞𝑗,𝑑max); 12
AP 𝑖[𝑗]←𝐴𝑃 𝑗;13
for eachcombination of answersAPinAP 𝑖do14
𝐷new←LLMGenerateNewDAG(𝐷,AP);15
F.𝑝𝑢𝑠ℎ(𝐷 new);16
returnD comp;17A.3 Answer and Path Retrieval
We summarize the comprehensive algorithmic procedure of answer
and path retrieval as shown in Algorithm 3.
Algorithm 3: RetrieveAnswersWithPaths
Input: Question SubgraphH 𝑞(V𝑞,E𝑞), Subquestion𝑞 𝑗, KH
Exploration Depth Limit𝑑 max, Beam Width𝑏
Output: Set of Step Answer - Reasoning Path Pair𝐴𝑃 𝑗
// Graph re-anchoring
T𝑗←TopicEntityInit(𝑞 𝑗,H𝑞);1
R𝑗←TargetHyperedgeMatch(𝑞 𝑗,H𝑞);2
// Initialize frontier
F𝑒←∅,P←∅;3
for each𝑒∈{𝑒|𝑣∈𝑒∧𝑣∈T 𝑗}∪R 𝑗do4
F𝑒←F 𝑒∪{(𝑒,[𝑒])};5
P←P∪{[𝑒]};6
for𝑑←1to𝑑 maxdo7
// Beam search from current frontier
𝐹cand←∅;8
for each(𝑒,𝑝 𝑒)∈F 𝑒do9
for each𝑒′∈Nbr(𝑒)do10
𝑆𝑣={ScoreEntityWithLLM(𝑣,𝑞 𝑗)|𝑣∈𝑉(𝑒)∩𝑉(𝑒′)}; 11
𝑠𝑒′←Aggregate(𝑆 𝑣);12
𝑝𝑒′←𝑝 𝑒⊕[𝑒′];13
𝐹cand←𝐹 cand∪{(𝑝 𝑒′,𝑠𝑒′)};14
𝐹cand←RankSelectDirections(𝐹 cand);15
𝐹sel←LLMSelectDirections(𝐹 cand,𝑞𝑗,𝑏);16
updateF 𝑒andPwith𝐹 sel;17
// Form and select reasoning paths
𝑃cand←FormPaths(P);18
𝑃cand←RankSelectPaths(𝑃 cand);19
𝑃sel←LLMSelectPaths(𝑃 cand,𝑞𝑗);20
if𝑃 sel≠∅then21
// Attempt subquestion
𝐴𝑃𝑗←∅;22
for each𝑝 𝑗∈𝑃 seldo23
𝑐𝑗←KnowledgeFusion(𝑝 𝑗);24
𝑎𝑗←LLMAnswerStep(𝑐 𝑗,𝑞𝑗);25
𝐴𝑃𝑗←𝐴𝑃 𝑗∪{(𝑎 𝑗,𝑝𝑗)};26
return𝐴𝑃 𝑗 27
return∅;28

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Xiangjun Zai et al.
Table 5: Performance vs. State Search Strategy.
# Init
𝑛0# Soln
𝐾BFS DFS
F1 ˆ𝑤𝑇𝑉𝑇 F1 ˆ𝑤𝑇𝑉𝑇
1 1 57.86 28.00 8.31 55.52 11.83 3.29
1 2 56.93 41.37 9.70 53.22 12.88 4.50
1 3 57.70 18.58 9.24 57.16 12.65 5.67
2 2 54.80 44.62 14.02 54.98 10.21 4.77
2 3 58.58 39.13 15.15 57.15 10.04 6.24
3 3 61.94 126.85 20.56 58.48 12.25 6.58
B Experiment Details
Baselines. We compare PRoH against five baselines:LLM-only,
which directly generates answers using the intrinsic knowledge of
the LLM;StandardRAG, a traditional chunk-based RAG approach;
PathRAG[ 2];HippoRAG2[ 12]; and the state-of-the-artHyper-
GraphRAG[ 19]. Since HyperGraphRAG is the current SOTA we
directly refer to their results and those of other baselines reported
in their paper for comparisons.
Experimental Settings. Experiments are conducted using GPT-4o-
mini as the LLM, and text-embedding-3-small for vector embedding.
For PRoH, we set plan depth 𝑑𝑝=3, KH exploration depth limit
𝑑𝑚𝑎𝑥=3, number of initial plans𝑛 0=2, max number of solutions
𝐾= 2. For PRoH-L, we set plan depth 𝑑𝑝=2, KH exploration depth
limit𝑑𝑚𝑎𝑥=3, number of initial plans 𝑛0=1, max number of
solutions𝐾=1.
C Additional Experiment
C.1 Ablation Study
(RQ7) How does the state search strategy affect the perfor-
mance of PRoH?For this experiment, we randomly sampled 200
questions from the Medicine domain and compare breadth-first
(BFS) and depth-first (DFS) state search strategies under different
settings of𝑛0, the number of initial plans, and 𝐾, the maximum num-
ber of solutions. As reported in Table 5, BFS consistently achieves
higher F1 scores than DFS, however this performance advantage
comes with significant extra computational cost. when 𝑛0=3 and
𝐾=3, BFS exhibits explosive growth in search width, it also visits
20.56 states in average, which is more than 3x of the DFS strategy.
DFS though, as expected has a much stable width. Also, when we
fix one of𝑛0or𝐾, increasing the other will always improve F1 score
for both strategies. Overall, DFS offers a better performance-to-cost
ratio as shows a more stable scaling behavior.
(RQ8) How does the plan depth affect the state search tree?As
shown in Table 6, increasing the plan depth consistently improves
both F1 and G-E scores. When 𝑑𝑝increases from1to3, F1 and G-E
rise from 55.65% to 59.47% and from 67.71% to 70.45%, respectively.
This indicates that deeper plan depth provides more comprehen-
sive context for planning. The average peak search tree depth ˆ𝑑𝑇
increases when 𝑑𝑝increases from1to2and then decreases to 1.360
when𝑑𝑝=3, suggesting that, with a richer planning context, a
more efficient plan can be generated. Overall, deeper plan depth
𝑑𝑝enhances performance without introducing excessive reasoning
complexity through question decomposition.Table 6: Performance vs. plan depth𝑑 𝑝.
Metrics F1 G-E ˆ𝒅𝑻
𝑑𝑝=1 55.65 67.71 1.375
2 57.13 68.57 1.415
3 59.47 70.45 1.360
MedicineAgricultureCS
LegalMix
Domain0%20%40%60%80%100%T oken %Module
Graph Anchoring
Plan Initialization
State Search
Answer and Path Retrieval
Final Answer Generation
Figure 6: Token Usage among Modules.
C.2 Efficiency Evaluation
(RQ9) How does token usage vary across reasoning modules?
Figure 6 reports the average token usage of PRoH across five do-
mains, segmented by reasoning module. Overall, Answer and Path
Retrieval dominates token consumption, indicating that most com-
putational effort is spent retrieving and integrating intermediate
reasoning states. Plan Initialization and State Search show moder-
ate usage, reflecting the cost of constructing and exploring internal
representations prior to retrieval. Final Answer Generation consis-
tently requires fewer tokens, suggesting that linguistic synthesis
is relatively lightweight compared to reasoning. Graph Anchoring
contributes minimally, serving as a brief setup phase. These trends
imply that PRoH ’s efficiency is primarily determined by retrieval
and reasoning dynamics rather than generation overhead.
C.3 Case study: Structured Question
Decomposition
In this section, we present Table 7, which illustrates PRoH’s struc-
tured question decomposition mechanism. It also demonstrates
PRoH’s effectiveness in handling multi-entity and multi-hop ques-
tion answering tasks.

PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Table 7: Example of Planning and Reasoning of PRoH.
Field Content
QuestionWhat must be prepared in accordance with GAAP for financial and tax reporting purposes?
Golden AnswerFINANCIAL STATEMENTS
Context (1) The books of the Partnership shall be maintained, for financial and tax reporting purposes, on an accrual basis
in accordance with GAAP.
(2) 1ACCOUNTING AND OTHER TERMS
(3) all accounting terms used herein shall be interpreted, all accounting determinations hereunder shall be made,
and all financial statements required to be delivered hereunder shall be prepared in accordance with GAAP as in
effect from time to time.
nary2
nhop3
DAG Edges0→1,0→2
Subquestion 0 Subquestion:What does GAAP stand for?
Topics:"GAAP"
Level:0
Answer:Generally Accepted Accounting Principles
Reasoning Path:<hyperedge>"“GAAP” means U.S. generally accepted accounting principles."
Subquestion 1 Subquestion:What standards do GAAP require for financial reporting?
Topics:"GAAP", "FINANCIAL REPORTING"
Level:1
Answer:GAAP requires financial statements to be prepared in accordance with its principles, ensuring accurate
representation of a company’s financial health.
Reasoning Path: <hyperedge>"All financial statements required to be delivered hereunder shall
be prepared in accordance with GAAP."
Subquestion 2 Subquestion:What standards do GAAP require for tax reporting?
Topics:"GAAP", "TAX REPORTING"
Level:1
Answer:Generally Accepted Accounting Principles (GAAP) require accurate and consistent reporting of financial
information in compliance with applicable tax laws and regulations.
Reasoning Path: <hyperedge>"The Company maintains internal controls over financial reporting...
sufficient to provide reasonable assurance that transactions are recorded as necessary to
permit preparation of financial statements for external purposes in accordance with GAAP."
-> <hyperedge>"(xxvii) Since December 31, 2019, the Association has not: (A) changed its tax
reporting or accounting policies or practices; (B) settled or compromised any liability for
taxes or made or rescinded any tax election; (C) incurred any debt except in the ordinary
course of business; (D) suffered any theft."
PRoH answer Answer:Financial statements
Reasoning: The question asks about what is prepared in accordance with GAAP for financial and tax reporting
purposes. The context specifies that financial statements must be prepared in accordance with GAAP, which
dictates how financial transactions are recorded and presented. This indicates that financial statements are the
expected answer.