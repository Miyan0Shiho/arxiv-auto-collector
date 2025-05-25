# Divide by Question, Conquer by Agent: SPLIT-RAG with Question-Driven Graph Partitioning

**Authors**: Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, Flora D. Salim

**Published**: 2025-05-20 06:44:34

**PDF URL**: [http://arxiv.org/pdf/2505.13994v1](http://arxiv.org/pdf/2505.13994v1)

## Abstract
Retrieval-Augmented Generation (RAG) systems empower large language models
(LLMs) with external knowledge, yet struggle with efficiency-accuracy
trade-offs when scaling to large knowledge graphs. Existing approaches often
rely on monolithic graph retrieval, incurring unnecessary latency for simple
queries and fragmented reasoning for complex multi-hop questions. To address
these challenges, this paper propose SPLIT-RAG, a multi-agent RAG framework
that addresses these limitations with question-driven semantic graph
partitioning and collaborative subgraph retrieval. The innovative framework
first create Semantic Partitioning of Linked Information, then use the
Type-Specialized knowledge base to achieve Multi-Agent RAG. The attribute-aware
graph segmentation manages to divide knowledge graphs into semantically
coherent subgraphs, ensuring subgraphs align with different query types, while
lightweight LLM agents are assigned to partitioned subgraphs, and only relevant
partitions are activated during retrieval, thus reduce search space while
enhancing efficiency. Finally, a hierarchical merging module resolves
inconsistencies across subgraph-derived answers through logical verifications.
Extensive experimental validation demonstrates considerable improvements
compared to existing approaches.

## Full Text


<!-- PDF content starts -->

arXiv:2505.13994v1  [cs.AI]  20 May 2025Divide by Question, Conquer by Agent: SPLIT-RAG
with Question-Driven Graph Partitioning
Ruiyi Yang1Hao Xue1Imran Razzak2,1Hakim Hacid3Flora D. Salim1
1University of New South Wales, Sydney, NSW, Austalia
2Mohamed Bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE
3Technology Innovation Institute, Abu Dhabi, UAE
ruiyi.yang@student.unsw.edu.au, hao.xue1@unsw.edu.au
imran.razzak@mbzuai.ac.ae, hakim.hacid@tii.ae
flora.salim@unsw.edu.au
Abstract
Retrieval-Augmented Generation (RAG) systems empower large language models
(LLMs) with external knowledge, yet struggle with efficiency-accuracy trade-
offs when scaling to large knowledge graphs. Existing approaches often rely
on monolithic graph retrieval, incurring unnecessary latency for simple queries
and fragmented reasoning for complex multi-hop questions. To address these
challenges, this paper propose SPLIT-RAG , a multi-agent RAG framework that
addresses these limitations with question-driven semantic graph partitioning and
collaborative subgraph retrieval. The innovative framework first create Semantic
Partitioning of Linked Information, then use the Type-Specialized knowledge base
to achieve Multi-Agent RAG . The attribute-aware graph segmentation manages to
divide knowledge graphs into semantically coherent subgraphs, ensuring subgraphs
align with different query types, while lightweight LLM agents are assigned to par-
titioned subgraphs, and only relevant partitions are activated during retrieval, thus
reduce search space while enhancing efficiency. Finally, a hierarchical merging
module resolves inconsistencies across subgraph-derived answers through logi-
cal verifications. Extensive experimental validation demonstrates considerable
improvements compared to existing approaches.
1 Introduction
Even State-of-the-art LLMs may fabricate fact answering a question beyond it’s pretrained knowl-
edge [ 49], or giving a wrong answer for those require latest data [ 22]. Retrieval-Augmented Genera-
tion (RAG) methods solve this problem by connecting external knowledge base to LLM without the
need to retrain the model. By offering reliable supplementary knowledge to LLM, factual errors are
reduced especially in those domain-specific questions, leading to higher accuracy and less hallucina-
tions [ 53,14,51]. Moreover, by updating database, up-to-date information can correct LLMs outdated
memory, generating correct response in those fast evolving areas, like politic [ 4,25], traffic [ 19],
and business [ 3,7]. RAG frameworks build up a bridge between black-box LLMs and manageable
database to provide external knowledge. However, effectively dealing with the knowledge base
remains a problem. Current RAG framework still faces several challenged C1,C2andC3:
C1:Efficiency : LLMs face both time and token limits in querying multi-document vectors. The
millions of documents would slow down search speed of LLM Retriever, while tiny-size databases are
not enough to cover many domain-related questions that need identification for specific knowledge.
While large-scale knowledge base contains more reliable and stable information, leading to a higher
accuracy, the redundant information costs extra latency during the query process. On the other hand,
Preprint. Under review.

utilization of small-scaled database would help solving domain specific questions, like medical [ 44]
or geography [ 8], but it would be hard to apply the framework on other area due to the specialty of
data structure and limit knowledge.
C2:Hallucination Knowledge bases are not following the same structure for data storage, same
information can be expressed by different format, including text, tables, graph, pictures, etc. The
diverse structure of data may pose extra hallucinations [ 36]. Even with correct extra data input, LLM
may still not following the facts, thus an answer in consistent with retrieved information would not
be exactly guaranteed [13].
C3:Knowledge Conflict It’s hard for LLMs to decide whether external databases contain errors.
Even if multiple knowledge sources are used, the mixture of inconsistent correct and wrong data
generates conflicts. On the other hand, knowledge is timeliness. Different frequency of updating the
knowledge base would also cause errors [10].
Figure 1: Utilizing SPLIT-RAG Framework on Example Dataset
To solve above limitations for existing RAG framework, this work introduces a new framework called
Semantic Partitioning of Linked Information for Type-Specialized Multi-Agent RAG (SPLIT-RAG ).
Figure 1 shows an example of how SPLIT-RAG framework works under a toy database, which
contains essential knowledge and historical questions. The SPLIT-RAG framework, while making
use of historical (training) questions to divide large-scale general knowledge base, uses multiple
agents to answer different types of questions. Agents choosing process further enhancing retrieval
speed by selective using only useful knowledge. Finally, conflicts are detected after merging triplets
from multiple retrievers to generate final answer. Specifically, our key contributions are:
•QA-Driven Graph Partitioning and Subgraph-Guided Problem Decomposition : A
novel mechanism is proposed that dynamically partitioning graphs into semantically coherent
subgraphs through training question analysis (entity/relation patterns, intent clustering),
ensuring each subgraph aligns with specific query types. Also, a hierarchical query splitting
is developed to decomposing complex queries into sub-problems constrained by subgraph
boundaries, enabling stepwise agent-based reasoning.
•Efficient Multi-Agent RAG Framework : A distributed RAG architecture where
lightweight LLM agents selectively activate relevant subgraphs during retrieval, achiev-
ing faster inference higher accuracy compared to monolithic retrieval baselines, while
maintaining scalability.
•Conflict-Resistant Answer Generation : While results are aggregate from different agents,
potential conflicts will be solved by importing a confidence score for each set, and misleading
2

triplets with low score will be cleared out, then head agent uses facts and supporting evidence
to answer the original questions.
2 Background
2.1 Retrieval-Augmented Generation with Large Language Models
RAG systems apply external knowledge bases to LLMs, retrieving extra knowledge according to
queries and thereby improving the accuracy of LLM response. External databases ensure knowledge
offered is domain-specific and timely, adding reliability and interpretability [ 27,21]. RAG systems are
designed and explored from different perspectives, like database modalities [ 51], model architectures,
strategies for training [ 10], or the diversity of domains that fit the system [ 14].Accuracy of knowledge
retrieval and quality of responses are two key factors for RAG systems evaluation [47].
Recent researches have combined graph-structured data into RAG systems(GraphRAG) to improve
the efficiency of knowledge interpretability by capturing relationships between entities and utilizing
triplets as the primary data source [ 33,17]. However, seldom researches consider the efficiency of
query speed. Large size knowledge base contains too much redundancy and unnecessary information,
which would cause latency during retrieval. Our work aims to extend existing effective approach using
structured graph data, while reducing latency and redundancy by segmenting the graph knowledge
base into smaller subgraphs.
2.2 Knowledge Graph Partition
Graph partition, as an NP-complete problem, has a wide range of applications, including parallel
processing, road networks, social networks, bioinformatics, etc.[ 6,26]. Partitioning a graph can
be done from two perspectives: edge-cut and vertex-cut. Many classic graph partition algorithms
have been proved to be effective mathematically. Kernighan-Lin algorithm [ 24] partition the graph
by iteratively swapping nodes between the two partitions to reduce the number of edges crossing
between them. Similarly, Fiduccia-Mattheyses algorithm [ 11] uses an iterative mincut heuristic to
reduce net-cut costs, and can be applied on hypergraph. METIS [ 23] framework use three phases
multilevel approach coming with several algorithms for each phase to generate refined subgraphs.
More recently, JA-BE-JA algorithm [ 35] uses local search and simulated annealing techniques to
fulfill both edge-cut and vertex-cut partitioning, outperforming METIS on large scale social networks.
However, many algorithms are not applicable on knowledge graph, since they generally apply global
operations over the entire graph, while sometimes the knowledge base would be too large, under
which dealing with them may need compression or alignment first [ 34,42]. LargeGNN proposes
a centrality-based subgraph generation algorithm to recall some landmark entities serving as the
bridges between different subgraphs. Morph-KGC [ 2] applies groups of mapping rules that generate
disjoint subsets of the knowledge graph, decreases both materialization time and the maximum peak
of memory usage. While existing works focus on graph content, our work manages to consider the
KS-partition problem from the natures of RAG system, by making full use of historical (training)
questions as the guide.
2.3 LLM-based multi-agent systems
Although single LLMs already meet demands of many tasks, they may encounter limitations when
dealing with complex problems that require collaboration and multi-source knowledge. LLM
based multi-agent systems (LMA) managed to solve problem through diverse agent configurations,
agent interactions, and collective decision-making processes [ 28]. LMA systems can be applied
to multiple areas, including problem solving scenes like software developing [ 9,43], industrial
engineering [ 31,45] and embodied agents [ 5,18]; LMA systems can also be applied on world
simulation, in application of societal simulation [ 12], recommendation systems [ 48], and financial
trading [ 15,52]. An optimal LMA system should enhance individual agent capabilities, while
optimizing agent synergy [16].
Subgraphs used in our QA scenario perfectly match the necessity of applying multi-agent RAG
framework. While each agent in charge of a subset of knowledge base, our framework applies only
active agents collaborate together, specializing individual agent’s task.
3

3 Divide by Question, Conquer by Agent: SPLIT-RAG framework
The SPLIT-RAG framework, marked as F, is defined in equation 1, as a combination of Multi-Agent
Generator ˆGand Retrieval module R. As part of R, the data indexer τgenerates a set of knowledge
base ˆKusing training RAG questions Qtrain and database D, while the other part of R, the data
retriever ψ, is used to choose useful knowledge bases and send to generator ˆGto answer new questions
qincoming. The main notations of SPLIT-RAG are summarized in Table 6, Appendix A.
F=
ˆG, R= (τ, ψ)
, F(q, D) =ˆG
q, ψ(ˆK)
,ˆK=τ(Qtrain, D) (1)
Specifically, the whole process for SPLIT-RAG framework contain five components: 1) Question-
Type Clustering and Knowledge Graph Partitioning; 2) Matching of Agent-Subgraph and Question-
Agent; 3) Decision making for retrieval plans for new questions; 4) Multi-agent RAG; 5) Knowledge
combination and conflict solving. Figure 2 concludes the general process for the entire framework.
Figure 2: The SPLIT-RAG framework: After preprocessing, knowledge base are divided into multiple
sub-bases according to attributes of training questions. New query is matched in question base by its
entity & relation, as well as the similarity of the QA route, to get the retrieval plan, including the
decomposed questions, parts of needed knowledge bases, as well as used agents. After multi-agent
retrieval through chosen subgraphs and agents, the triplet sets and route information are merged
together after eliminating the conflict, to generate the final answer and explanation.
3.1 Question-Type Clustering and Knowledge Graph Partitioning
The initial stage include two steps: At first, questions Qtrain are clustered according to their attributes,
such as pre-labeled types Lqtype , or entity type Letype contained inside questions. Each raw question
is transferred into three format, while an example of how questions are preprocessed is shown in
Appendix E.2.1.
•Semantic Context Qs: Stop words are removed, and all [E]are marked inside questions.
•Entity Type Context Qe: Detailed entities are changed into entity types labeled in the
knowledge base, which is used to decide the semantic question type of the questions.
•Path Context p: Path used to retrieve correct answers is also stored, to facilitate searching
process during multi-agent RAG.
QTrain based KG Partition Based on path context Qp, the raw large knowledge graph Dare parti-
tioned into smaller, more manageable subgraphs. One path piinP={p1, ...pm}can be expressed
as{ep1;rp12;ep2...;rp(n−1)(n);epn}, all path contexts are split into consecutive 2-hop maximum
element, forming set ϕ(pi) =Sk−1
j=1 
epj;rp(j)(j+1);ej+1
, while all splited path set are combined
as˜P=Sm
i=1ϕ(pi). The optimal graph partitioning maximizes information gain while controlling
subgraph size. Information gain IGis calculated as IG(S) =P
si∈S[H(P|si)−λ·H(si)], a
combination of conditional entropy H(P|si) =−P
˜pj∈siP(˜pj|si) logP(˜pj|si)and size penalty
4

H(si) =|Vsi|
|V|log|V|
|Vsi|. The best subgraph partition S∗is calculated in Equation 2, where ηmax
restricts maximum graph size.
S∗= arg max
SIG(S)s.t.∀si∈ S,|Vsi| ≤ηmax (2)
The process of subgraph partition is somewhat similar to node coloring, while each subgraph
ˆDi→Cidenote one color, except for some node Ei∈ {Cj, ...C k}. Algorithm 1 demonstrate the
detailed process for knowledge base partitioning. This process manages to make a balance between
P(ˆDi|Letype)andP(ˆDi|Lqtype).
3.2 Subgraph-Agent-Question Matching
Multi-Agent is used in our framework to increase query efficiency. To generate the best match
between D ⇔ A ⇔ Q train , theQtrain={q1, . . . , q m}initially have and partitioned subgraphs D=
{ˆD1, . . . , ˆDn}are used to form Association Matrix A∈[0,1]m×n:Aij=|{p∈Paths(qi)|p⊆ˆDj}|
|Paths(qi)|
where Paths (qi)denotes all reasoning paths for question qi. To reflect subgraphs’ mutual overlap, Cov-
erage Set Ci={ˆDj∈ D | Aij>0}is established for question qi. The minimum number of agents
are generated while each agent Gis allocated to one ’area’ of knowledge. The matching optimization
aims to minimize cross-agent coordination costs: min{G1,...Gk}Pm
i=1l
|{j|ˆDj∈Sk
l=1Gl∩Ci}|
Nmaxm
, which
subject to
•Coverage CompletenessSk
l=1Gl=D;
•Capacity Constraint |Gl| ≤Nmax,∀l;
•Semantic Coherence1
|Gl|P
ˆDj∈GlSim(ˆDj, µl)≥θcohwhere µlis the centroid of group
Gland Sim (ˆDj, µl)represents a similarity comparison.
The detailed subgraph-agent matching algorithm, guiding by the question coverage, is shown in
Algorithm 2. Through this process, agent allocation is also linked with Qtrain , which is stored for
next step.
3.3 Retrieval Plans Decision for Incoming Questions
For an incoming question qnew, a retrieval plan is generated before enter into retriever ψ. Similarly
as training question, three format of the question, {Qsnew,Qenew,pqnew}are generated first. The
potential agents needed for this question are generated through 1)Training Question Similarity
Search; 2)Path-Driven Subgraph Matching, represented as finding optimal agent set A∗⊆ A through:
A∗={a|a∈ A(qsim)} if Sim (qnew, qsim)≥θdirect
{a| ∃p∈ϕ(qnew∧a∋ˆDp}otherwise
Mentioned in detail in Algorithm 3, finding the optimal multi-agent set is also seen a complex
question decomposition process. Each subgraph, each agent, would only be useful to answer part of
the complex question (while obviously for simple-structured 1-hop question it would be easy to find
similar question and subgraph from training question set). The question decomposition operation is
defined as: Ψ(qnew) =Ψsim(qnew) ={(s1,A1), ...,(sk,Ak)} if∃qsim
Ψpath(qnew) ={(p1,A′
1), ...,(pm,A′
m)}otherwise
By generating retrieval plan for the incoming new test question, decisions are made on the usage of
exact agents for next step, while the question decomposition process further facilitate RAG process,
since each agent only needs to consider simpler-structured subquestions.
3.4 Multi-Agent RAG
Distributed Retrieval Process Given the decomposed subquestions {ti}and associated subgraphs
{ˆDi}from last routing phase, each agent Ai∈ A∗processes assigned tiparallel, while each process
•Graph Traversal with path matching score Match (p, t) =|Entities (p)∩Entities (t)|
|Entities (t)|;
5

•Triplet Retrieval generate T RI from Subgraph;
•Textualize triplet to generate evidence text ET.
Together returning both of the retrieved triplet T RI iand evidence text ETi, the detailed process is
presented in Algorithm 4.
3.5 Final Answer Generation
Result Aggregation After results are collected from all agents, aggregation is made through M=S
Ai∈A∗{(T RI i,ETi,conf(Ai))}, to generate the final triplet set T RI all=S|A∗|
i=1Ti,and final
evidence text set ET all=S|A∗|
i=1ETi.
Conflict solving Defined as Conflict (τ1, τ2) =1ifτ1⊢ ¬τ2orτ2⊢ ¬τ1
0otherwise, the conflict detection
function is used to filter out logically incorrect triplets. With the confidence score for each Aiand
conflict graph built from triplets retrieved, scores of triplet are compute and a final maximal clique
is retained with misleading triplets from low score Ai. The conflict resolution algorithm ensures:
∀τ1, τ2∈ T clean,Conflict (τ1, τ2) = 0 .
Head Agent Synthesis Final answer is generated through: Answer =LLM head(Eall,Tclean, qnew)
where the head agent Ahead is prompted with given verified facts, supporting evidence, and qorigin .
3.6 Information-Preserving Partitioning and Computational Efficiency
Proven in Appendix D, Theorem 1, the subgraphs ensure there is no lose in information comparing
with using whole graph based on IGcalculation. Also, Theorem 2 proves the matching between
Ai− Gi− Q satisfies mutual information match. The time effectiveness of applying SPLIT-RAG
on KG is proved in Theorem 3, based on the prominent search space reduction. Let Nbe the total
entities in KG and kthe average subgraph size. Comparing with using single KG, SPLIT-RAG
achieves: Tretrieve =O N
klogk
vsTbase=O(N), Given m=⌈N/k⌉subgraphs, each requires
O(logk)search via B-tree indexing. Considering the process of graph partition, the more general the
dataset, the higher search improvement SPLIT-RAG can achieve.
4 Experiment
Several experiments are set to verify the effectiveness of our SPLIT-RAG framework. Metrics are
designed for evaluation from overall correctness, efficiency, and factuality.
4.1 Experiment Setup: Benchmarks and Baselines
Table 1: Dataset statistics
Dataset #Train #Test Max Hop
WebQSP 2,826 1,628 2
CWQ 27,639 3,531 4
MetaQA-2hop 119,986 114,196 2
MetaQA-3hop 17,482 14,274 3Four widely used KGQA benchmarks
are used for experiments: 1) WebQues-
tionsSP(WebQSP) [ 46], which contains
full semantic parses in SPARQL queries
answerable using Freebase.; 2) Com-
plex WebQuestions(CWQ) [ 41], which
takes SPARQL queries from WebQSP
and automatically create more complex
queries; 3-4) MetaQA-2-Hop and MetaQA-
3-Hop [ 50], consisting of a movie ontology
derived from the WikiMovies Dataset and three sets of question-answer pairs written in natural
language, ranging from 1-3 hops. 1-hop questions in MetaQA dataset are not used in experiments
since the queries are too basic. The detailed dataset statistics are presented in table 1.
Three types of baselines are included in the experiments. Listed as below, details of those baselines
are described in Appendix B.
•1) Embedding method , including KV-Mem [ 32], GraftNet [ 40], PullNet [ 39], Embed-
KGQA [37], TransferNet [38];
6

•2) LLM output , used models includes Llama3-8b, Davinci-003, ChatGPT, Gemini 2.0
Flash; Gemini 2.0 Flash-Lite and Gemini 2.5 Flash Preview 04-17 is also used in Section 4.4
for judging agents’ importance in our framework.
•3) KG+LLM method , including StructGPT [ 20], Mindful-RAG [ 1], standard graph-based
RAG, RoG [29]. Our SPLIT-RAG also falls into this category.
Hit, Hits@1 (H@1), and F1 metrics are used for evaluation. Hit measures whether there is at least
one gold entity returned, especially useful in LLM-style recall. Hits@1 (H@1) measures exact-match
accuracy of the top prediction. Finally F1 achieves a span-level harmonic mean over predicted
answers vs. true answers.
4.2 Experiment Result: How competitive is SPLIT-RAG with other baselines?
Table 2: Performance comparison of different methods on KGQA benchmarks. The best and
second-best methods are denoted. Some results of embedding and KG+LLM models come from
existing experiments in paper [29, 30, 20]
.
Overall Results
Type MethodWebQSP CWQ MetaQA-2Hop MetaQA-3Hop
Hit H@1 F1 Hit H@1 F1 Hit H@1 Hit H@1
EmbeddingKV-Mem [32] - 46.7 - - 21.1 - - 82.7 - 48.9
GraftNet [40] - 66.4 - - 32.8 - - 94.8 - 77.1
PullNet [39] - 68.1 - - 47.2 - - 99.9 - 91.4
EmbedKGQA [37] - 66.6 - - 45.9 - - 98.8 - 94.8
TransferNet [38] - 71.4 - - 48.6 - - 100 - 100
LLMLlama3-8b 59.2 - - 33.1 - - - - 31.7 -
Davinci-003 48.3 - - - - 25.3 - 42.5 -
ChatGPT 66.8 - - 39.9 - - 31.0 - 43.2 -
Gemini 2.0 Flash 65.3 - - 41.1 - - - - 46.9 -
KG+LLMStructGPT [20] - 72.6 - - - - - 97.3 - 87.0
Mindful-RAG [1] - 84.0 - - - - - - - 82.0
Graph-RAG 77.2 73.1 67.7 58.8 54.6 53.9 - 85.4 - 78.2
RoG [29] 85.7 80.0 70.8 62.6 57.8 56.2 - - - 84.8
SPLIT-RAG 87.7 84.9 72.6 64.2 61.1 59.3 97.9 95.2 91.8 88.5
SPLIT-RAG beats all baselines from in WebQSP and CWQ datasets, resulting in hit rates of87.7
in WebQSP and 64.2 in CWQ. In MetaQA datasets, SPLIT-RAG also outferforms comparing LLM
and KG+LLM baselines, resulting in Hit rate of 97.6 in MetaQA-3Hop and 91.8 in MetaQA-3Hop.
Standard Embedding Methods achieved higher accuracy on MetaQA dataset due to the limit size of
the KG, but it’s not comparative with existing KG+LLM methods since they rely too much on size
and structure of graphs, when applied on large knowledge base like free base, embedding baselines
hasmuch lower accuracy .
Table 3: KG and Subgraph Size
Dataset KG #Entity #Relations #Triplets #Avg GEntity GCoverage
WebQSP Freebase 2,566,291 7,058 8,309,195 65802.3 91.3
CWQ Freebase 2,566,291 7,058 8,309,195 38302.9 72.8
MetaQA WikiMovie 43,234 9 133,582 8646.8 99.9
The size of knowledge base influence the retrieval process. In WebQSP and CWQ dataset, from
Table 3 there is prominent differences in GCoverage(whether or not the combination of used
subgraphs are enough to cover all knowledge to answer questions) between subgraphs Gbuilt
on freebase and that on MetaQA. With less relation numbers and enough training questions, the
subgraphs of MetaQA can cover mostly incoming rest questions. While in WebQSP and CWQ
experiments, combinations of larger Gare still not enough to cover many of test questions.
4.3 Ablation Study: The Value of Key Component
7

Table 4: Ablation Study on WebQSP dataset. Details of A(retrieval
plan generation), B(multi-agent usage), and C(conflict detection) are
explained in 4.3.
Method Hit H@1 F1 Avg# {G} Avg{G}size
SPLIT-RAG 87.7 84.9 72.6 4.6 67,302.9
SPLIT-RAG - A 80.6 79.0 64.1 6.3 66,841.2
SPLIT-RAG - B 70.1 66.1 52.2 1 303,071.1
SPLIT-RAG - C 84.2 82.6 72.2 4.6 67,302.9Ablations study are done
to test the necessity of
1)Generating retrieval plan
using training questions;
2)Using multi-agent with
subgraphs; 3)Applying
conflict detection on final
triplet sets. Experiments
are set for 3 stages compar-
ing with initial results on
WebQSP dataset: A)Use
subgraphs labels other than finding question similarities to calculate the final useful {G}; B) Other
than retrieve knowledge separately, merging all useful Gand use only one agent Aito generate
retrieve; C) Omit conflict detection, throwing all retrieved triplets to Ahead to generate answers.
From Table 4, without using qtrain similarities, the accuracy of retrieval plan for incoming questions
will drop, adding some labels is not enough to precisely describe partitioned G(like the combination
of entity & relation types), even though it would need more subgraphs(and waste agent calls ). Hit
rate and F1 drop more seriously when not retrieve triplets separately. {G} may be unmergable
to one connected graph , therefore many complex query are not executive at all when applying
to questions – thus the accuracy is more close to using LLM only to answer the questions. Also
the usage of subgraph is time saving(which is also theoretical provable in Appendix D), while B)
causes prominent increase in search space . Finally, without conflict detection, the accuracy and
F1 didn’t drop too much since only differences exist in few conflict triplets, which happens only
in small # questions. The ablation study demonstrates the necessity of applying qtrain supported
type-specialized retrieval plan generation and Multi-Agent RAG, while adding conflict detection can
also avoid redundancy and hallucinations: SPLIT-RAG decreases the search space while using
limit agent calls to get high accuracy .
4.4 Accuracy-Latency Tradeoff & Model Flexibility : Is it necessary to unify LLM agents?
Two kinds of agents are used in Generator ˆGof SPLIT-RAG framework: 1) {Ai}in multi-agent
framework controlling subgraphs {Gi}, and 2) Head agent Ahead receiving all triplets and corpus
then generate final answer. Advanced models with more recent knowledge will do better in reasoning,
thus resulting in more accurate responses. However, {Ai}andAhead are doing different jobs, while
subgraph control only taking in graph structure and generating routing plan, Ahead receives more
complex triplets and corpus and need to sort out the final answer. Experiments are done on MetaQA-
3Hop using different models to answer the following questions: Is it necessary to apply same models
on all agents to ensure accuracy? Or there could be a trade-off between accuracy and cost to apply
less powerful model for ˆGwhile using a more advanced AHead t for the final reasoning? Three
models are used for testing: 1) Gemini 2.5 Flash Preview 04-17; 2) Gemini 2.0 Flash; 3) Gemini 2.0
Flash-Lite. Table 5 presents the Hit,H@1 results, as well as time spent on using agent per QA .
All models are applied on {Ai}andAhead, without assuming which agent is more important. Six
experiments are done under different agent settings. Table 5 presents the detailed results of hit rates,
Considering the model performance, Gamini 2.5 Flash Preview 04-17 is the best, while costing the
most especially in reasoning mode. Gemini 2.0 Flash is in the middle, while the 2.0 Flash-Lite one is
least costly and weaker in reasoning.
Table 5: Results on MetaQA-3Hop using different model for agents
Group Ai Ahead Hit H@1 Avg Time(s)
G1 Gemini 2.0 Flash Gemini 2.0 Flash-Lite 81.6 79.9 29.1
G2 Gemini 2.5 Flash Preview 04-17 Gemini 2.0 Flash-Lite 83.1 81.5 38.8
G3 Gemini 2.0 Flash-Lite Gemini 2.0 Flash 90.4 87.3 20.2
G4 Gemini 2.0 Flash Gemini 2.0 Flash 91.8 88.5 33.7
G5 Gemini 2.0 Flash-Lite Gemini 2.5 Flash Preview 04-17 92.1 89.2 28.6
G6 Gemini 2.0 Flash Gemini 2.5 Flash Preview 04-17 93.9 90.7 37.1
8

More advanced models achieve better results: the combination of Gemini 2.0 Flash + 2.5 Flash
Preview in Group 6 leads to the best results, better than that listed in Table 2 where only Gemini
2.0 Flash is used on SPLIT-RAG. Even though a weaker Gemini 2.0 Flash-Lite is used for subgraph
agents Ai, a prominent result is still generated in Group 5 , with a better AHead resulting in a hit rate
of 92.1%. On the other hand, however, the model drops an accuracy of 9% in Hit@1 inGroup 2 if
the combination is switched to a reversed matching of Gemini 2.5 – Aiand Gemini 2.0 Flash-Lite
–AHead . Similarly, both Gemini 2.0 Flash and Flash-Lite are used in Group 1 andGroup 3 , but
there is 8% difference in performance – A better AHead would be more important than a better Aiin
deciding overall accuracy.
(a) Performance–degradation analysis.
20.0 22.5 25.0 27.5 30.0 32.5 35.0 37.5
Average Query Time (s)808284868890Hit@1 (%)
G1G2G3G4G5G6Accuracy Latency Trade-off (MetaQA-3Hop)
Pareto frontier (b) Time–Hit comparison with Pareto frontier.
Figure 3: Comparison of model performance and latency, while (a) presents the performance
degradation, with darker color showing higher degradation. (b) presents Tim-Hit comparison, with a
Pareto frontier showing acceptable tradeoff in between G3-G5-G6.
To further visualize the difference of importance between two kinds of agents, figure 3a presents a
performance degradation analysis. While setting the best Hits@1 result as baseline, rate differences of
other groups are calculate and the proportion of degradation is drawn. Y-axis lists settings of AHead
and X-axis lists that of Ai. Noticing nodes in bottom-right corner have darker colors–more serious
performance degradation–than that on top-left corner, it means using advanced models in Aiis less
efficient than updating AHead . More reasoning is required for AHead to draw the final answer, while
lighter models can be applied on Aito reduce cost. Figure 3b further presents the trade-off between
accuracy and latency using Pareto frontier . G1, G4 and G2 under the frontier shows a less efficient
setting, G3-G5-G6 on the Pareto frontier means there can be a trade-off: While ensuring the reasoning
ability of AHead , it is acceptable or even better, considering the cost of applying multi-agent to Gi,
to degrade Aito a lighter model to accelerate and save money on the whole process. For example,
moving from G6 to G5 (changing Gemini 2.0 Flash to Flash-Lite in Ai) caused 2% reduce in the hit
rate, while saving 8 seconds on agents: SPLIT-RAG avoids the necessity of using advanced costly
agent all the time .
5 Conclusion and Future Work
In this paper, a new RAG framework is proposed, i.e., Semantic Partitioning of Linked Information
forType-Specialized Multi-Agent RAG(SPLIT-RAG) that contains 1) A novel mechanism that does
graph partitioning based on training QA, while decomposing incoming questions using subgraphs;
2) A efficient multi-agent RAG framework that reduce searching space and accelerate retrieval; 3)
Conflict detection during knowledge aggregation to reduce hallucination facing redundant or false
knowledge. Based on extensive experiments, SPLIT-RAG achieves state-of-the-art performance
on three benchmark KGQA datasets(WebQSP, CWQ and MetaQA), and ablation study verifies
the necessity of all key module. Efficiency tests further show that lightweight subgraph agents
plus a strong head agent yield near-peak accuracy at lower cost. While SPLIT-RAG still has space
for improvement like effectively dealing with dynamic KG, future work includes (1) Extend the
partitioner to streaming graphs via online rebalancing and (2) Develop stronger verification schemes
for rare-entity conflicts challengeing current logical checks.
9

Acknowledgments and Disclosure of Funding
This research is partially supported by the Technology Innovation Institute, Abu Dhabi, UAE.
Additionally, computations were performed using the Wolfpack computational cluster, supported by
the School of Computer Science and Engineering at UNSW Sydney.
References
[1]Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, and Huan Liu. Mindful-rag: A study
of points of failure in retrieval augmented generation. In 2024 2nd International Conference on
Foundation and Large Language Models (FLLM) , pages 607–611. IEEE, 2024.
[2]Julián Arenas-Guerrero, David Chaves-Fraga, Jhon Toledo, María S Pérez, and Oscar Corcho.
Morph-kgc: Scalable knowledge graph materialization with mapping partitions. Semantic Web ,
15(1):1–20, 2024.
[3]Muhammad Arslan and Christophe Cruz. Business-rag: Information extraction for business
insights. In 21st International Conference on Smart Business Technologies , pages 88–94.
SCITEPRESS-Science and Technology Publications, 2024.
[4]Muhammad Arslan, Saba Munawar, and Christophe Cruz. Political-rag: using generative ai to
extract political information from media content. Journal of Information Technology & Politics ,
pages 1–16, 2024.
[5]Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel
Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, et al. Do as i can, not as i say: Grounding
language in robotic affordances. In Conference on robot learning , pages 287–318. PMLR, 2023.
[6]Aydın Buluç, Henning Meyerhenke, Ilya Safro, Peter Sanders, and Christian Schulz. Recent
advances in graph partitioning . Springer, 2016.
[7]Bruno Amaral Teixeira de Freitas and Roberto de Alencar Lotufo. Retail-gpt: leveraging
retrieval augmented generation (rag) for building e-commerce chat assistants. arXiv preprint
arXiv:2408.08925 , 2024.
[8]T Dong, C Subia-Waud, and S Hou. Geo-rag: Gaining insights from unstructured geological
documents with large language models. In Fourth EAGE Digitalization Conference & Exhibition ,
volume 2024, pages 1–4. European Association of Geoscientists & Engineers, 2024.
[9]Yihong Dong, Xue Jiang, Zhi Jin, and Ge Li. Self-collaboration code generation via chatgpt.
ACM Transactions on Software Engineering and Methodology , 33(7):1–38, 2024.
[10] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua,
and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language
models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and
Data Mining , pages 6491–6501, 2024.
[11] Charles M Fiduccia and Robert M Mattheyses. A linear-time heuristic for improving network
partitions. In Papers on Twenty-five years of electronic design automation , pages 241–247.
1988.
[12] Chen Gao, Xiaochong Lan, Zhihong Lu, Jinzhu Mao, Jinghua Piao, Huandong Wang, Depeng
Jin, and Yong Li. S3: Social-network simulation system with large language model-empowered
agents. arXiv preprint arXiv:2307.14984 , 2023.
[13] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to
generate text with citations. arXiv preprint arXiv:2305.14627 , 2023.
[14] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv
preprint arXiv:2312.10997 , 2023.
[15] Fulin Guo. Gpt in game theory experiments. arXiv preprint arXiv:2305.05516 , 2023.
10

[16] Junda He, Christoph Treude, and David Lo. Llm-based multi-agent systems for software
engineering: Literature review, vision and the road ahead. ACM Transactions on Software
Engineering and Methodology , 2024.
[17] Yuntong Hu, Zhihan Lei, Zheng Zhang, Bo Pan, Chen Ling, and Liang Zhao. Grag: Graph
retrieval-augmented generation. arXiv preprint arXiv:2405.16506 , 2024.
[18] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng,
Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied
reasoning through planning with language models. arXiv preprint arXiv:2207.05608 , 2022.
[19] Mohamed Manzour Hussien, Angie Nataly Melo, Augusto Luis Ballardini, Carlota Salinas
Maldonado, Rubén Izquierdo, and Miguel Angel Sotelo. Rag-based explainable prediction of
road users behaviors for automated driving using knowledge graphs and large language models.
Expert Systems with Applications , 265:125914, 2025.
[20] Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. Structgpt:
A general framework for large language model to reason over structured data. arXiv preprint
arXiv:2305.09645 , 2023.
[21] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings of
the 2023 Conference on Empirical Methods in Natural Language Processing , pages 7969–7992,
2023.
[22] Nikhil Kandpal, Haikang Deng, Adam Roberts, Eric Wallace, and Colin Raffel. Large language
models struggle to learn long-tail knowledge. In International Conference on Machine Learning ,
pages 15696–15707. PMLR, 2023.
[23] George Karypis and Vipin Kumar. Metis: A software package for partitioning unstructured
graphs, partitioning meshes, and computing fill-reducing orderings of sparse matrices. 1997.
[24] Brian W Kernighan and Shen Lin. An efficient heuristic procedure for partitioning graphs. The
Bell system technical journal , 49(2):291–307, 1970.
[25] M Abdul Khaliq, Paul Chang, Mingyang Ma, Bernhard Pflugfelder, and Filip Mileti ´c. Ragar,
your falsehood radar: Rag-augmented reasoning for political fact-checking using multimodal
large language models. arXiv preprint arXiv:2404.12065 , 2024.
[26] Jin Kim, Inwook Hwang, Yong-Hyuk Kim, and Byung-Ro Moon. Genetic approaches for
graph partitioning: a survey. In Proceedings of the 13th annual conference on Genetic and
evolutionary computation , pages 473–480, 2011.
[27] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented
generation for knowledge-intensive nlp tasks. Advances in neural information processing
systems , 33:9459–9474, 2020.
[28] Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu, and Yi Yang. A survey on llm-based multi-agent
systems: workflow, infrastructure, and challenges. Vicinagearth , 1(1):9, 2024.
[29] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful
and interpretable large language model reasoning. arXiv preprint arXiv:2310.01061 , 2023.
[30] Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language
model reasoning. arXiv preprint arXiv:2405.20139 , 2024.
[31] Nikhil Mehta, Milagro Teruel, Patricio Figueroa Sanz, Xin Deng, Ahmed Hassan Awadallah,
and Julia Kiseleva. Improving grounded language understanding in a collaborative environment
by interacting with agents through help feedback. arXiv preprint arXiv:2304.10750 , 2023.
[32] Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, and
Jason Weston. Key-value memory networks for directly reading documents. arXiv preprint
arXiv:1606.03126 , 2016.
11

[33] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang,
and Siliang Tang. Graph retrieval-augmented generation: A survey. arXiv preprint
arXiv:2408.08921 , 2024.
[34] Kyle K Qin, Flora D Salim, Yongli Ren, Wei Shao, Mark Heimann, and Danai Koutra. G-crewe:
Graph compression with embedding for network alignment. In Proceedings of the 29th ACM
International Conference on Information & Knowledge Management , pages 1255–1264, 2020.
[35] Fatemeh Rahimian, Amir H Payberah, Sarunas Girdzijauskas, Mark Jelasity, and Seif Haridi. A
distributed algorithm for large-scale graph partitioning. ACM Transactions on Autonomous and
Adaptive Systems (TAAS) , 10(2):1–24, 2015.
[36] Tolga ¸ Sakar and Hakan Emekci. Maximizing rag efficiency: A comparative analysis of rag
methods. Natural Language Processing , 31(1):1–25, 2025.
[37] Apoorv Saxena, Aditay Tripathi, and Partha Talukdar. Improving multi-hop question answering
over knowledge graphs using knowledge base embeddings. In Proceedings of the 58th annual
meeting of the association for computational linguistics , pages 4498–4507, 2020.
[38] Jiaxin Shi, Shulin Cao, Lei Hou, Juanzi Li, and Hanwang Zhang. Transfernet: An effective and
transparent framework for multi-hop question answering over relation graph. arXiv preprint
arXiv:2104.07302 , 2021.
[39] Haitian Sun, Tania Bedrax-Weiss, and William W Cohen. Pullnet: Open domain question
answering with iterative retrieval on knowledge bases and text. arXiv preprint arXiv:1904.09537 ,
2019.
[40] Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhutdinov, and
William W Cohen. Open domain question answering using early fusion of knowledge bases
and text. arXiv preprint arXiv:1809.00782 , 2018.
[41] Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex
questions. arXiv preprint arXiv:1803.06643 , 2018.
[42] Bayu Distiawan Trisedya, Flora D Salim, Jeffrey Chan, Damiano Spina, Falk Scholer, and
Mark Sanderson. i-align: an interpretable knowledge graph alignment model. Data Mining and
Knowledge Discovery , 37(6):2494–2516, 2023.
[43] Haoyuan Wu, Zhuolun He, Xinyun Zhang, Xufeng Yao, Su Zheng, Haisheng Zheng, and Bei
Yu. Chateda: A large language model powered autonomous agent for eda. IEEE Transactions
on Computer-Aided Design of Integrated Circuits and Systems , 2024.
[44] Junde Wu, Jiayuan Zhu, and Yunli Qi. Medical graph rag: Towards safe medical large language
model via graph retrieval-augmented generation. arXiv preprint arXiv:2408.04187 , 2024.
[45] Yuchen Xia, Manthan Shenoy, Nasser Jazdi, and Michael Weyrich. Towards autonomous
system: flexible modular production system enhanced with large language model agents. In
2023 IEEE 28th International Conference on Emerging Technologies and Factory Automation
(ETFA) , pages 1–8. IEEE, 2023.
[46] Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The
value of semantic parse labeling for knowledge base question answering. In Proceedings of
the 54th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short
Papers) , pages 201–206, 2016.
[47] Hao Yu, Aoran Gan, Kai Zhang, Shiwei Tong, Qi Liu, and Zhaofeng Liu. Evaluation of
retrieval-augmented generation: A survey. arXiv preprint arXiv:2405.07437 , 2024.
[48] Junjie Zhang, Yupeng Hou, Ruobing Xie, Wenqi Sun, Julian McAuley, Wayne Xin Zhao, Leyu
Lin, and Ji-Rong Wen. Agentcf: Collaborative learning with autonomous language agents for
recommender systems. In Proceedings of the ACM Web Conference 2024 , pages 3679–3689,
2024.
12

[49] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo
Zhao, Yu Zhang, Yulong Chen, et al. Siren’s song in the ai ocean: a survey on hallucination in
large language models. arXiv preprint arXiv:2309.01219 , 2023.
[50] Yuyu Zhang, Hanjun Dai, Zornitsa Kozareva, Alexander Smola, and Le Song. Variational
reasoning for question answering with knowledge graph. In Proceedings of the AAAI conference
on artificial intelligence , volume 32, 2018.
[51] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng, Fangcheng Fu, Ling
Yang, Wentao Zhang, and Bin Cui. Retrieval-augmented generation for ai-generated content: A
survey. arXiv preprint arXiv:2402.19473 , 2024.
[52] Qinlin Zhao, Jindong Wang, Yixuan Zhang, Yiqiao Jin, Kaijie Zhu, Hao Chen, and Xing Xie.
Competeai: Understanding the competition dynamics in large language model-based agents.
arXiv preprint arXiv:2310.17512 , 2023.
[53] Fengbin Zhu, Wenqiang Lei, Chao Wang, Jianming Zheng, Soujanya Poria, and Tat-Seng Chua.
Retrieving and reading: A comprehensive survey on open-domain question answering. arXiv
preprint arXiv:2101.00774 , 2021.
13

A Notations
Important notations are listed in Table 6.
Table 6: Notations Tables in SPLIT-RAG
Notation Definition
F Abbreviation of the SPLIT-RAG framework.
ˆG Multi-Agent Generator.
R Retrieval Module, return data from initial database. D
τ Data Indexer, receiving training Questions Qtrain and essential data.
ψ Data Retriever, searching in knowledge base and return matching results.
ˆK Preprocessed knowledge base based on D, containing question type information from Qtrain .
L Labels matching Qtrain , containing semantic information behind questions.
Qs Semantic context of one question, with entities marked.
Qe Entities in Qsare replaces by entity types
p Path information for answering one question.
P Paths in KG matching how answers retrieved for training questions.
IG Information gain used to control sugbraph size, details in Equation 2.
ˆDi Initial generated smaller subgraphs set based on Qtrain andIGcontrol.
ˆCi Coverage set for questions to reflect subgraphs’ mutual overlap.
ˆGi Final merged larger subgraphs from ˆDisatisfying constraint explained in 3.2 .
Ai Agents controlling each subgraph ˆGi.
T RI Triplets retrieved by ψfor decomposed subquestions.
ET Evidence text generated from T RI for better reasoning.
Ahead Head agent draw conclusion based on merged knowledge.
B Baseline Description
Detailed description of used baselines are listed in Table 7.
Table 7: Baselines
.
Used Baselines
Type Method Description
EmbeddingKV-Mem [32] Utilizes a Key-Value memory network to store triples, achieved multi-hop reasoning
GraftNet [40] Use KG subgraphs to achieve reasoning
PullNet [39] Using a graph neural network to retrieve a question-specific subgraph
EmbedKGQA [37] Seeing reasoning on KG as a link prediction problem, using embeddings for calculation
TransferNet [38] Uses graph neural network for calculating relevance in between entities
KG+LLMStructGPT [20] Utilizing an invoking-linearization-generation procedure to support reasoning
Mindful-RAG [1] Designed for intent-based and contextually aligned knowledge retrieval
Graph-RAG Standard KG-based RAG framework.
RoG [29] Using a planning-retrieval-reasoning framework for reasoning
SPLIT-RAG Our framework, used subgraph partition combined with multi-agents
14

C Algorithms
C.1 Knowledge Graph Partitioning and Merging
The subgraph partition is based on 1) preprocessed routing in Qpgenerated from training questions
Qtrain and 2) Information gain IGincluding both conditional entropy and size penalty. While the
best subgraph partition aiming to maximize the information gain, the key idea behind the algorithm is
toConcentrating path that is used by semantically or structurally similar types of questions ,
while limiting the size of one subgraph. Detailed explaination is included in Algorithm 1, matching
Section 3.1.
Algorithm 1 Knowledge Graph Partitioning and Merging Algorithm
Require: Original graph G= (V,E), decomposed paths ˜P, merge threshold θ
Ensure: Final subgraphs ˆD={ˆD1, . . . , ˆDk}
1:Initialize candidate subgraphs {s(0)
j}from ˜P
2:fort←1toTmaxdo
3: for all pairs (sa, sb)∈ S(t−1)× S(t−1)do
4: Compute merge gain: ∆ab=IG(sa∪sb)−[IG(sa) +IG(sb)]
5: end for
6: (a∗, b∗)←arg max (a,b)∆ab
7: if∆a∗b∗> θthen
8: S(t)← S(t−1)\ {sa∗, sb∗} ∪ {sa∗∪sb∗}
9: else
10: break
11: end if
12:end for
13:Merge small subgraphs: ˆD ← { sj∈ S(T)| |Vsj| ≥τmin}
C.2 Question-Centric Agent-Subgraph Allocation
While Algorithm 1 aiming to split one large KG into several small connected subgraphs {Di}, the
question-centric agent-subgraph allocation aims to further aggregate {Di}to a larger subgraph
groups . Following similar but different principle, Dare aggregated base on 1) limited number of
agents 2) limited subgraph group Gsize. The detailed algorithm matches context in 3.2
15

Algorithm 2 Question-Centric Agent-Subgraph Allocation Algorithm
Require: Association matrix A, max capacity Nmax, coherence threshold θcoh
Ensure: Agent groups {G1, . . . ,Gk}
1:Initialize coverage map M ← {C i| ∀qi∈ Q train}
2:Initialize priority queue PQwithCisorted by coverage density:
ρ(Ci) =Pm
j=1|Ci∩ Cj|p
|Ci|
3:whilePQ̸=∅do
4: Extract C∗←arg max C∈PQ ρ(C)
5: Create candidate group Gcand← {ˆDj∈ C∗}
6: if|Gcand|> N maxthen
7: Sort subgraphs by coverage frequency:
f(ˆDj) =mX
i=1I(ˆDj∈ Ci)
8: TrimGcand←Top-Nmaxbyf(ˆDj)
9: end if
10: ifCoherence (Gcand)≥θcohthen
11: Commit Gk← G cand
12: Update M ← M \ { qi| Ci∩ Gk̸=∅}
13: Update PQwith remaining coverage sets
14: end if
15:end while
16:forRemaining ˆDj/∈SGldo
17: Assign to nearest group:
l∗= arg max
l|{ˆDj} ∩ G l|
|Gl|
18:end for
C.3 Test Question Decomposition and Agent Routing
Matching Section 3.3, Algorithm 3 generate a retrieve plan for the incoming questions following 2
stages:
Stage 1 : Embeddings are used to find the most similar questions in Qtrain , which are stored and
served as part of the knowledge base. New questions are decomposed based on the structure and
route information in the similar Qtrain
Stage 2 : Decomposed questions are used for agent matching using the subpaths(1/2-hop). Each agent
finally assigned aims for control the subgraph Gthat suit at least one or more subpath(s) routing.
16

Algorithm 3 Test Question Decomposition and Agent Routing Algorithm
Require: New question qnew, agent registry R={A,D,M}
Ensure: Agent-task mapping MAP ={(ai, ti)}
1:Phase 1: Similar Question Guided Decomposition
2:Tokenize qnew’sEntity Type Context Qeintotnew= [w1, . . . , w k]
3:Compute embedding for question vnew=Emb(tnew)
4:foreachqi∈ Q traindo
5: Sim_ETC new,i = cos( vnew,vi)·exp(−β·rank(qi))
6:end for
7:Retrieve top-k similar questions {q1, ..., q k}using:
Sim(qnew, qi) =Sim_ETC (qnew, qi) +α·PathOverlap (qnew, qi)
8:if∃qiwith Sim (qnew, qi)≥θsimthen
9: Extract decomposition pattern:
Pi={(sj,Aj,ˆDj)}m
j=1← M [qi]
10: Adapt pattern to qnew:
11: foreach(sj,Aj,ˆDj)∈ Pido
12: t′
j←AlignSubproblem (sj, qnew)
13: Verify coverage: Cover (t′
j,ˆDj)≥θalign
14: MAP ← MAP ∪ { (Aj, t′
j)}
15: end for
16: return MAP
17:end if
18:Phase 2: Path-Driven Adaptive Decomposition
19:Decompose into atomic subpaths:
Patomic =[
p∈Paths(qnew)ϕatomic(p)
where ϕatomic splits paths into 1/2-hop segments
20:foreachpj∈ P atomic do
21: Find maximal matching subgraphs:
Dj={ˆD∈ D | PathMatch (pj,ˆD)≥θmatch}
22: Select optimal agent:
a∗
j= arg max
a∈Ah
Conf(a|ˆD)·Load(a)−1i
23: MAP ← MAP ∪ { (a∗
j, pj)}
24:end for
C.4 Parallel Subgraph Retrieval
Subgraph retrieval mentioned in Algorithm 4 retrieves both triplets, as well as generating some
evidence based on triplets.
17

Algorithm 4 Parallel Subgraph Retrieval
Require: Sub-question ti, subgraph ˆDi, retriever ψ
Ensure: Retrieved triplets T RI i, evidence text ETi
1:Graph Traversal :
Pi={p∈Paths (ˆDi)|Match (p, ti)≥θmatch}
where path matching score:
Match (p, t) =|Entities (p)∩Entities (t)|
|Entities (t)|
2:Triplet Retrieval :
T RI i=[
p∈Piψ(p)where ψ(p) ={(es, r, eo)∈p}
3:Evidence Generation :
Ei=LLM sum
[
(es,r,eo)∈TiTextualize (es, r, eo)

4:return (T RI i, ET i)
C.5 Multi-Agent RAG Results Conflict Solving
The conflict graph mentioned in Algorithm 5 contains the score linking current question and assigned
subgraphs. So if two triplets is semantically conflict (for example, having same head entity and tail
entity, but have different relations), the triplets come from less relative(credit) Gwill be cleaned.
Algorithm 5 Conflict Solving on Multi Agent RAG Results
Require: T RI all, confidence scores {conf(Ai)}
Ensure: Cleaned triplets T RI clean
1:Build conflict graph Gc= (V, E):
V=T RI all, E ={(τa, τb)|Conflict (τa, τb) = 1}
2:foreach connected component C∈Gcdo
3: Compute triplet scores:
s(τ) =X
ai∈A∗conf(ai)·I(τ∈ T RI i)
4: Retain maximal clique:
Ckeep= arg max
C′⊆CX
τ∈C′s(τ)
5: T RI clean← T RI clean∪Ckeep
6:end for
D Proof of Effectiveness
Theorem 1 (Information-Preserving Partitioning) .Given the balancing factor λin Algorithm 1, the
graph partitioning achieves:
I(Q;G)≥1
λ(E[IG(S)−H(G)) (3)
where I(·;·)denotes mutual information, H(·)is entropy, Qis the question distribution, and Gis the
subgraph collection.
18

Proof. Starting from the information gain definition:
E[IG] =kX
i=1[H(P|Gi)−λH(Gi)] (4)
=H(P)−kX
i=1|Gi|
|G|H(P|Gi)
| {z }
Mutual information I(P;G)−λH(G) (5)
Applying the data processing inequality for the Markov chain Q → P → G :
I(Q;G)≥I(P;G)≥1
λ(E[IG]−H(G)) (6)
Theorem 2 (Semantic Interpretability) .For subgraph ˆDiand its question cluster Qi, the mutual
information satisfies:
I(ˆDi;Qi)≥log|C| − H(ˆDi|Qi) (7)
whereCis the entity type set.
Proof. From clustering objective in Eq. 3:
I=H(Qi)−H(Qi|ˆDi)≥H(Qi)−ϵ≥log|C| − ϵ (8)
withϵcontrolled by the H(ˆDi|Qi)bound.
Theorem 3 (Search Space Reduction) .The expected retrieval time satisfies:
E[Tretrieve ]≤Tfull
exp(I(Q;G))(9)
where Tfullis the full-graph search time.
Proof. LetNi=|Gi|be the size of subgraph i. The per-subgraph search complexity is:
Ti=O(Ni·exp(−I(Q;Gi))) (10)
Applying Jensen’s inequality to the convex function f(x) = exp( −x):
E[T] =kX
i=1Ni
NTi (11)
≤Tfull·exp 
−kX
i=1Ni
NI(Q;Gi)!
(12)
≤Tfull
exp(I(Q;G))(13)
E Experiment Data and Prompt Template
E.1 Experiment compute resources
Graph-partitioning and local inference with Llama-3-8B were executed on four NVIDIA RTX A5000
GPUs (24 GB VRAM each). Calls to Gemini and GPT were made through remote APIs, so no local
GPU was required; a CPU-only setup was sufficient. All experiments ran on a server equipped with
an Intel Xeon Silver 4310 (12 cores, 2.10 GHz) processor.
19

Table 8: Illustrative Example of Three Formats Training Questions Stored in Question Base
Question Format Example
Raw Question the films that share directors with the film [Black Snake Moan] were in which genres
Semantic Context films share directors film [Black Snake Moan] genres
Entity-Type Context movie share director movie genre
Path Context movie-[directed_by]-director-[directed_by]-{ Black Snake Moan }-[has_genre]-genre
E.2 Example Data
E.2.1 Example of how training questions are preprocessed and stored
Examples of different formats of questions are presented in Table 8. Beside raw question, semantic
context, entity-type context and path context are used in our experiments.
E.3 Prompt Template
E.3.1 Subgraph Control Agent AiPrompt
Given Subgraph Context:
- Entity Types: {<T_entity>}
- Relation Types: {<T_relations>}
- Coverage: This subgraph focuses on <label> relationships
Given Current Subquestion: <Q_sub>
Task:
1. Analyze the subquestion’s core information need
2. Generate {SPARQL/Cypher} query matching the subgraph schema
Critical Constraints:
- Use ONLY entities/relations from the subgraph context
- Query returns triples that directly answer the subquestion
Output Requirements:
{
"query": "<generated_query>",
"reasoning": "<brief_explanation_of_strategy>",
}
E.3.2 Head Agent AHPrompt
Given verified facts: <T_clean>
Supporting evidence: <E_all>
Original question: <q_new>
Generate final answer with explanations,
resolving any remaining ambiguities.
F Broader Impacts
Our work combines Graph-RAG technique with multi-agent framework. The conflict-aware fusion
module explicitly cross-checks answers drawn from multiple subgraphs, reducing single-source
hallucinations and exposing contradictions that might signal biased or outdated knowledge. While not
a panacea, this architectural safeguard raises the auditability bar relative to monolithic LLM reasoning.
At web-scale usage, such improvements compound into measurable energy savings, aligning with
global efforts to make AI more sustainable.
20