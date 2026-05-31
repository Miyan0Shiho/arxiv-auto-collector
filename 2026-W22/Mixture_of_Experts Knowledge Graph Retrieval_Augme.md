# Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation

**Authors**: Shijie Wang, Chengyi Liu, Yujuan Ding, Shanru Lin, See-Kiong Ng, Xu Xin, Wenqi Fan

**Published**: 2026-05-27 08:56:57

**PDF URL**: [https://arxiv.org/pdf/2605.28175v1](https://arxiv.org/pdf/2605.28175v1)

## Abstract
Large language models (LLMs) have recently been adopted for recommendations due to their ability to understand user intent and item semantics. However, LLM-based recommender systems often rely on parametric knowledge and suffer from outdated knowledge, motivating knowledge graph retrieval-augmented generation (KG-RAG) to ground recommendations on structured, up-to-date KGs. Despite this promise, effective KG-RAG in recommendations faces great challenges. First, users' queries vary in complexity and require KG knowledge at different granularities, whereas existing methods adopt a one-size-fits-all retrieval strategy, leading to over-retrieval for simple queries and under-retrieval for complex ones. In addition, augmenting LLMs with KG knowledge requires translating graph-structured data into linear text, which may introduce noise and cause structural information loss. Moreover, the selection of retrieval granularity lacks direct supervision and must be inferred from the final recommendation after alignment and downstream utilization, making query-aware retrieval hard to learn end-to-end. To address these issues, we propose MixRAGRec, a cooperative multi-agent framework for KG-RAG recommendations. MixRAGRec integrates a Mixture-of-Experts Retrieval Agent that routes each query to a KG retrieval expert with different granularities, a Knowledge Preference Alignment Agent that converts structured knowledge into LLM-friendly natural language, and a Contrastive Learning-reinforced Recommendation Agent trained with contrastive preference feedback. Notably, we introduce Mixture-of-Experts Multi-Agent Policy Optimization (MMAPO) to train three agents under a unified objective. Extensive experiments on real-world datasets demonstrate the effectiveness of our framework.

## Full Text


<!-- PDF content starts -->

Mixture-of-Experts Knowledge Graph Retrieval-Augmented
Generation for Multi-Agent LLM-based Recommendation
Shijie Wang
shijie.wang@connect.polyu.hk
The Hong Kong Polytechnic
University
Hong Kong SAR, ChinaChengyi Liu
chengyi.liu@connect.polyu.hk
The Hong Kong Polytechnic
University
Hong Kong SAR, ChinaYujuan Ding∗
dingyujuan385@gmail.com
The Hong Kong Polytechnic
University
Hong Kong SAR, China
Shanru Lin
lllam32316@gmail.com
City University of Hong Kong
Hong Kong SAR, ChinaSee-Kiong Ng
seekiong@nus.edu.sg
National University of Singapore
Singapore, SingaporeXu Xin
xin.xu@polyu.edu.hk
The Hong Kong Polytechnic
University
Hong Kong SAR, China
Wenqi Fan
wenqifan03@gmail.com
The Hong Kong Polytechnic
University
Hong Kong SAR, China
Abstract
Large language models (LLMs) have recently been adopted for rec-
ommendations due to their ability to understand user intent and
item semantics. However, LLM-based recommender systems often
rely on parametric knowledge and suffer from outdated knowledge,
motivating knowledge graph retrieval-augmented generation (KG-
RAG) to ground recommendations on structured, up-to-date KGs.
Despite this promise, effective KG-RAG in recommendations faces
great challenges. First, users’ queries vary in complexity and require
KG knowledge at different granularities, whereas existing methods
adopt a one-size-fits-all retrieval strategy, leading to over-retrieval
for simple queries and under-retrieval for complex ones. In addi-
tion, augmenting LLMs with KG knowledge requires translating
graph-structured data into linear text, which may introduce noise
and cause structural information loss. Moreover, the selection of
retrieval granularity lacks direct supervision and must be inferred
from the final recommendation after alignment and downstream uti-
lization, making query-aware retrieval hard to learn end-to-end. To
address these issues, we proposeMixRAGRec, a cooperative multi-
agent framework for KG-RAG recommendations. MixRAGRec inte-
grates a Mixture-of-Experts Retrieval Agent that routes each query
to a KG retrieval expert with different granularities, a Knowledge
Preference Alignment Agent that converts structured knowledge
∗Corresponding author
This work is licensed under a Creative Commons Attribution 4.0 International License.
KDD 2026, Jeju Island, Republic of Korea.
©2026 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-2259-2/2026/08
https://doi.org/10.1145/3770855.3817630into LLM-friendly natural language, and a Contrastive Learning-
reinforced Recommendation Agent trained with contrastive pref-
erence feedback. Notably, we introduce Mixture-of-Experts Multi-
Agent Policy Optimization (MMAPO) to train three agents under
a unified objective. Extensive experiments on real-world datasets
demonstrate the effectiveness of our framework. Our codes are
available at https://github.com/Sjay-Wang/MixRAGRec.
CCS Concepts
•Information systems→Recommender systems.
Keywords
Large Language Models, Recommendations, Retrieval-augmented
Generation, Multi-agent Reinforcement Learning
ACM Reference Format:
Shijie Wang, Chengyi Liu, Yujuan Ding, Shanru Lin, See-Kiong Ng, Xu
Xin, and Wenqi Fan. 2026. Mixture-of-Experts Knowledge Graph Retrieval-
Augmented Generation for Multi-Agent LLM-based Recommendation. In
Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and
Data Mining V.2 (KDD 2026), August 9–13, 2026, Jeju Island, Republic of Korea.
ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/3770855.3817630
Resource Availability:
The source code of this paper has been made publicly available at https:
//doi.org/10.5281/zenodo.20372864.
1 Introduction
Recommender systems, as a key technique for alleviating infor-
mation overload and enabling personalized decision-making, have
attracted increasing attention across domains such as e-commerce
and social media [ 5,8,34,36]. Recently, Large Language Models
(LLMs) have achieved significant breakthroughs and show strong
potential as a new paradigm for recommender systems [ 2,29,45].arXiv:2605.28175v1  [cs.IR]  27 May 2026

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
actor
...direct_byact
direct
Not Retrieval
direct_by
Nolan
Nolan
War
I recently watched
Spider-man,Doctor
Strange,  Recommend    
 a movie for me.
I recently watched
Inception, Interstellar .
Recommend a movie
for me.
I recently watched
Oppenheimer, Dunkirk .
Recommend a movie
for me.
Benedict
Nolan
Moraltypetype
Moral
According to your
watching history, I
recommend you to
watch the movie >>>
LLM
Augmented
Noise Injection
(Over Retrieval)
Missing Knowledge
(Under Retrieval)
Right Granularity
Cost-Aware
 (a) Existing RAGRec Methods 
 (b) Our MixRAGRec 
Triple
Triple
Triple
Empty
Triple
Subgraph
directdirect_by
Benedict
Nolan
War
NolanEasy 
Query
Medium 
Query
Hard 
Query
Figure 1: Illustration of existing RAGRec methods and
our proposed MixRAGRec methods. (a) Existing RAGRec
methods apply the same KG retrieval granularity to all
queries, causing either noise from over-retrieval or missing
knowledge from under-retrieval; (b) Our MixRAGRec routes
queries of different complexity to appropriate retrieval gran-
ularity, improving the utility–cost trade-off.
By training on vast and diverse data, LLMs exhibit strong gener-
alization and nuanced language understanding capabilities [ 4,28].
These capabilities enable LLM-based recommender systems to dis-
till user preferences from profiles, item metadata, and interaction
histories for more tailored suggestions [33, 41].
Despite their remarkable advantages in language understanding
and generalization, existing LLM-based recommender systems still
suffer from critical limitations, including hallucinations and insuffi-
cient up-to-date, domain-specific knowledge [ 23,35]. For instance,
an LLM may recommend a non-existent sequel like“The Matrix 5:
Reawakened”to a fan of the “Matrix”. Furthermore, the static nature
of LLMs’ training data often leaves them unable to recommend the
latest items absent from the pre-training corpus. To alleviate these
issues, recent Retrieval-Augmented Generation (RAG) leveraging
external Knowledge Graph (KG) has emerged as a promising solu-
tion to provide the latest and domain-specific structured knowledge
without costly fine-tuning [ 32,35]. By grounding LLMs inexter-
nal, structured, updated, andeditableknowledge sources, KG-RAG
provides a robust mechanism for supplying factual, up-to-date, and
domain-specific information, thereby mitigating hallucinations and
knowledge latency.
However, achieving effective KG-RAG recommender systems
faces significant challenges.First, user queries require KG knowl-
edge at different granularities, while existing KG-RAG recommender
systems typically adopt a one-size-fits-all retrieval granularity [ 16,
35]. This mismatch leads to over-retrieval for simple queries (intro-
ducing redundant context and extra cost) and under-retrieval for
complex queries (missing multi-hop relations needed for reasoning),
as illustrated in Figure 1(a). Hence, it remains challenging to design
a query-aware retrieval mechanism that selects an appropriate re-
trieval granularity to balance knowledge utility and retrieval cost.
Additionally, the retrieved knowledge is naturally graph-structured,
while the input to LLMs is plain text. Directly describing structured
knowledge into text can introduce redundant noise and lose struc-
tural information, reducing the utility of retrieved knowledge [ 31].Thus, it is challenging to convert KG knowledge into an LLM-
friendly representation that preserves recommendation-relevant
knowledge while avoiding unnecessary noise.Furthermore, the se-
lection of retrieval granularity lacks direct supervision and can only
be inferred from the final recommendation outcome after alignment
and downstream utilization. This intricate interlinking makes the
influence of retrieval granularity hard to disentangle, while apply-
ing vanilla end-to-end optimization with only the recommendation
criteria may converge to a fixed granularity preference. This poses
a special challenge to effectively coordinate the retrieval, alignment,
and recommendation under a unified training objective.
To address these challenges, we proposeMixRAGRec(Mixture-
of-Experts KG-RAGfor LLM-basedRecommendation), a coopera-
tive multi-agent KG-RAG framework that integrates query-aware
retrieval, knowledge alignment, and knowledge-augmented recom-
mendation. Specifically, MixRAGRec consists of three cooperating
agents: (1) aMixture-of-Experts Retrieval Agentthat routes
each query to one of multiple KG retrieval experts with different
granularities, ranging fromno retrieval(direct generation) to triple
retrieval, neighborhood subgraph retrieval, and connected-graph
retrieval; (2) aKnowledge Preference Alignment Agentthat
transforms retrieved structured knowledge into an LLM-friendly
natural-language knowledge snippet, bridging the gap between
graph-structured knowledge and textual inputs; and (3) aCon-
trastive Learning-reinforced Recommendation Agenttrained
withcontrastive preference feedback, which encourages the
model to rank the target item above competitive negatives under
the same context. To jointly optimize these agents, we introduce
MMAPO(Mixture-of-ExpertsMulti-AgentPolicyOptimization),
which coordinates discrete expert selection, knowledge alignment,
and downstream recommendation under a shared objective. In
particular, MMAPO incorporates amarginal information gain
term that quantifies retrieval utility relative to its computational
cost, encouraging the system to retrieve richer knowledge only
when the expected benefit justifies the added overhead. The main
contributions of this paper can be summarized as follows:
•We propose MixRAGRec, a multi-agent KG-RAG framework for
LLM-based recommendation that performs query-aware retrieval
with different granularities via a mixture of KG experts.
•We introduce a Knowledge Preference Alignment agent that
converts structured KG knowledge into concise, LLM-friendly
natural language, mitigating the structure-semantics gap.
•We propose MMAPO, a unified optimization framework that
couples agents with a shared objective and a retrieval-utility
reward for cost-aware retrieval.
•We conduct comprehensive experiments on various real-world
datasets to demonstrate the effectiveness of the proposed MixRA-
GRec framework.
2 Related Work
Retrieval-Augmented Generation (RAG) has emerged as a robust
strategy to mitigate LLM limitations, such as hallucinations and
lack of knowledge, by grounding outputs in external knowledge
sources [ 7,9,11,20]. Early studies such as RETRO [ 3] and FLARE [ 18]
typically retrieve fragments from unstructured corpora (e.g., pas-
sages or documents) and condition generation on the retrieved

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
Knowledge Align
Select Retrieve 
Expert
Knowledge Triple
Subgraphs
Connected GraphsWithout RAG
r movie
 actor
Retrieved Knowledge Aligned Knowledge
The Avengers is a
science fiction film
directed by Joss
Whedon. Similar films
include The Matrix...Mixture-of-Experts Retrieval #Query: 
The user has watched the movie Avengers ...
#Retrieved Knowledge:
The Avengers is a science fiction film directed by
Joss Whedon. Similar films include The Matrix...
#Response : 
Based on the retrieved background knowledge,  I
have a few recommendations... Recommendation
Generation
Mixture -of-Experts  Retrieval Agent Knowledge Preference Alignment Agent Contrastive Learning-reinforced Rec Agent
Policy  Model
Policy  Model
Policy  ModelValue Model
Reward Model
GAE
qMixture -of-Experts Multi-Agent Policy Optimization (MMAPO)
Reference  Model
Loss Function
q
Query
Negative ItemsGround TruthExpert 1
Expert 2
Expert 3
Expert 4
Figure 2: The overview of the MixRAGRec, which consists of three agents: (1)Mixture-of-Experts Retrieval Agentselects a retrieval
expert to obtain structured knowledge, (2)Knowledge Preference Alignment Agentconverts the retrieved KG knowledge into an
LLM-friendly representation, and (3)Contrastive Learning-reinforced Recommendation Agentproduces the recommendation.
Mixture-of-Experts Multi-Agent Policy Optimization (MMAPO)jointly optimizes these agents under a shared objective.
context. Beyond text corpora, recent studies explore augmenting
LLMs with graph-structured knowledge (e.g., KGs) to provide struc-
tured knowledge for LLM inference [ 13,14,27,37]. Representative
studies such as G-retriever [ 14] and GNN-RAG [ 27] retrieve sub-
graphs from KGs and incorporate them into LLM inference via
either structured encoders or text-based serialization.
Building on these advances, RAG has been introduced into rec-
ommender systems to ground LLM-based recommendations on ex-
ternal knowledge, mitigating issues such as hallucinations and out-
dated knowledge. Depending on the source of retrieved knowledge,
existing RAG-based recommender systems can be broadly grouped
into two types. The first type of methods typically retrieve knowl-
edge from datasets (e.g., historical user–item interactions and user
behaviors) to provide domain-specific knowledge for LLMs to im-
prove user modeling [ 21,23,26,39,40,47]. For instance, ReLLa [ 23]
and RALLRec [ 39] retrieve the most relevant behaviors from long
interaction histories instead of relying solely on the most recent
actions, while G-Refer [ 21] retrieves user profiles and explanation
paths from interaction graphs to enhance explainability. The other
methods retrieve knowledge from external knowledge bases, most
notably knowledge graphs, to provide structured and up-to-date
knowledge for LLM-based recommendations [ 32,35,44]. A rep-
resentative method, K-RagRec [ 35], retrieves relevant knowledge
subgraphs for cold-start items to improve knowledge-augmented
recommendation. Despite their success, existing methods are typi-
cally built on fixed retrieval and ignore the difficulties of queries,
which can lead to suboptimal accuracy and efficiency. To fill this
gap, we propose MixRAGRec to enable query-aware retrieval and
effective utilization of retrieved KG knowledge within a unified
framework. We provide more related work in Appendix F.3 Methodology
In this section, we first introduce some key notations and concepts
in this paper. Then, we provide the details for each component of
our proposed framework MixRAGRec.
3.1 Problem Formulation
In this paper, we consider a knowledge-augmented recommenda-
tion scenario. For each instance, the system receives a user query 𝑞
describing the user intent and profile context (e.g., historical inter-
actions), and outputs a recommended item 𝑜from a candidate set
O. To support retrieval-augmented recommendation, the system
can access a KG 𝐺=(V,R,T) , whereVis the node set,Ris the
relation set, andT⊆V×R×V is the triple set. Given a query
𝑞, the system may retrieve a knowledge set from 𝐺. We define our
retrieval expert set with four experts, which is extendable to more.
Specifically,Eexp={𝑒 1,𝑒2,𝑒3,𝑒4}and each𝑒returns structured
knowledgeK 𝑒at a specific granularity.
It is worth noting that MixRAGRec is formulated as a cooper-
ative multi-agent decision process. Generally, given 𝑞, a Mixture-
of-Experts retrieval agent chooses an expert 𝑒∈E expto execute
retrieval; a knowledge preference alignment agent transforms the
retrieved knowledge K𝑒into a natural-language representation ˜K;
finally, a recommendation agent outputs 𝑜conditioned on(𝑞,˜K,O) .
The framework aims to optimize recommendation performance
while accounting for retrieval utility. Formally, the objective func-
tion is defined as:
(𝜃∗,𝜓∗,𝜔∗)=arg max
𝜃,𝜓,𝜔E(𝑞𝑖,O𝑖,𝑜★
𝑖)∼Dh
E𝑒𝑖∼𝜋𝜃(·|𝑠𝑖)
E˜K𝑖∼𝜋𝜓(·|𝑞𝑖,K𝑖,𝑒𝑖)E𝑜𝑖∼𝜋𝜔(·|𝑞𝑖,˜K𝑖,O𝑖)
𝑅(𝑞𝑖,O𝑖,𝑜★
𝑖;𝑒𝑖,˜K𝑖,𝑜𝑖)i
,(1)

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
where𝜋𝜃,𝜋𝜓, and𝜋𝜔are the policies of the retrieval, alignment,
and recommendation agents, respectively. Ddenotes the training
distribution and 𝑜★∈O is the ground-truth item. 𝑠𝑖is the retrieval
state constructed from the query context, and 𝑅(·) is the reward
function detailed in Section 3.6.2.
3.2 The Overview of MixRAGRec
As illustrated in Figure 2, MixRAGRec is a multi-agent framework
for knowledge-augmented recommendation that consists of three
crucial agents: theMixture-of-Experts Retrieval Agent, theKnowledge
Preference Alignment Agent, and theContrastive Learning-reinforced
Recommendation Agent. Specifically, given a user query, the Mixture-
of-Experts retrieval agent selects the appropriate retrieval expert
to obtain structured knowledge from the KG. Next, the knowledge
preference alignment agent converts the retrieved knowledge into
an LLM-friendly representation, bridging the representational gap
between graph-structured knowledge and text-based modeling.
Finally, the contrastive learning-reinforced recommendation agent
produces the recommendation based on the query and aligned
knowledge. To train these agents under a shared objective and
enable query-aware expert selection, we proposeMixture-of-Experts
Multi-Agent Policy Optimization (MMAPO), which coordinates them
through a shared reward that accounts for both recommendation
performance and retrieval utility. This design enables MixRAGRec
to generate accurate recommendations by adapting retrieval for
diverse queries through a unified multi-agent framework.
3.3 Multi-Granularity KG Retrieval with
Mixture-of-Experts
MixRAGRec adopts a Mixture-of-Experts retrieval module that
retrieves structured knowledge from an item-centric KG at multiple
granularities. As illustrated in Figure 3, the left part shows KG
construction and indexing, while the right part depicts a Mixture-
of-Experts retrieval design, ranging from direct generation without
retrieval to triple-, subgraph-, and connected-graph retrieval. This
design aims to provide query-relevant knowledge while avoiding
unnecessary retrieval cost and noise.
3.3.1 KG Construction and Indexing.We represent the knowledge
source as a KG 𝐺=(V,R,T) , whereVandRdenote the sets
of entities and relations, and T⊆V×R×V denotes the set of
triples(ℎ,𝑟,𝑡) . To facilitate flexible retrieval, we encode both enti-
ties and triples into a shared embedding space using a pre-trained
language encoder. For an entity 𝑛∈V with textual description
𝑥𝑛, we capture the semantic information asz 𝑛=Encode(𝑥 𝑛)∈R𝑑,
where𝑑denotes the dimension of the output representation. Then
we build an entity vector database ZV={z𝑛}𝑛∈V. Similarly, for a
triple(ℎ,𝑟,𝑡)∈Twith text descriptions(𝑥 ℎ,𝑥𝑟,𝑥𝑡), we compute
z(ℎ,𝑟,𝑡)=Encode Concat(𝑥 ℎ,𝑥𝑟,𝑥𝑡)∈R𝑑,(2)
and construct a triple vector database ZT={z(ℎ,𝑟,𝑡)}(ℎ,𝑟,𝑡)∈T . By
utilizing this dual indexing, retrieval experts can access KG knowl-
edge at different levels of granularity.
3.3.2 Mixture-of-Experts KG Retrieval.While existing KG-RAG
methods provide a general mechanism to incorporate KG knowl-
edge, they typically rely on a fixed retrieval strategy. In practice,
-Entity type
-Relation
-Attribute type
Filtering
KG Vector Databases
I have watched {Godfather,Matrix,
The Lion King,Lightyear} . I
especially like the movies directed
by Nolan.  What movie do you
recommend  I watch next? User
KG Construction and Indexing
Without RAG
entity entityr
...
1. Personal PageRank 2. Kruskal's MSTExpert 1 Expert 2 Expert 3 Expert 4
Mixture-of-Experts Retrieval
Indexing
Query
Retrieve
Connected GraphSubgraphsTriples
0.1
0.90.1
0.80.9
0.2KG from Web ...Figure 3: Demonstration of KG Construction and Indexing
and the Retrieval Experts.
user queries vary widely, ranging from simple cases requiring little
or no external knowledge to complex ones involving multi-hop
knowledge subgraphs. As a result, a fixed KG retrieval strategy may
under-retrieve for complex queries, leading to insufficient knowl-
edge for accurate recommendation, or over-retrieve for simple
queries, introducing unnecessary noise and computational over-
head.
To this end, we propose a Mixture-of-Experts retrieval agent that
selects the appropriate retrieval granularity for each query. Specifi-
cally, four retrieval experts are designed to address diverse query
scenarios, including direct generation without retrieval, triple-level
retrieval for precise facts, 𝑘-hop neighborhood subgraphs captur-
ing higher-order relations, and connected subgraphs representing
global knowledge, as shown in Figure 3.
Given a user query 𝑞, we compute its embedding using the same
encoder as in indexing to ensure a consistent representation space:
z𝑞=Encode(𝑞)∈R𝑑.
The Mixture-of-Experts retrieval agent then selects one expert
𝑒∈ { 1,2,3,4}and returns the retrieved KG knowledge K𝑒, as
illustrated in Figure 3 (right).
Expert 1: DirectGenerator (no retrieval).This expert performs
no KG retrieval and returns an empty set, i.e.,K 1=∅.
Expert 2: TripleRetriever (triple-level retrieval).This expert re-
trieves the most relevant KG triples by semantic similarity. Formally,
the retrieved triple set is:
K2=TopM(ℎ,𝑟,𝑡)∈T sim z𝑞,z(ℎ,𝑟,𝑡),(3)
where𝑀is the retrieval budget, and is instantiated differently for
different granularities.sim(·,·)measures the cosine similarity.
Expert 3: SubgraphRetriever ( 𝑘-hop neighborhood subgraph).
To capture multi-hop relational context, this expert first selects a
seed set of relevant entities:
S=TopM𝑛∈V sim z𝑞,z𝑛,(4)
and then expands each seed into its 𝑘-hop neighborhood subgraph.
Denote the𝑘-hop neighborhood subgraph around entity 𝑛as𝐺(𝑘)
𝑛,
the retrieved knowledge is the unionK 3=Ð
𝑛∈S𝐺(𝑘)
𝑛.
Expert 4: ConnectedGraphRetriever (connected subgraph).
For complex queries that benefit from globally coherent knowledge,
the expert constructs a connected subgraph that captures global
knowledge. It begins with the same seed set Sas Expert 3, and
then executes a Personalized PageRank (PPR) [ 30] fromSto obtain
importance scores over entities. Then the top- 𝑀entities by PPR
score, denotedVPPR, will be selected and consider the induced

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
subgraph𝐺[V PPR]. To obtain a connected subgraph containing the
most relevant entities while minimizing edge costs, we formulate
the retrieval objective as finding a minimum-cost spanning tree
(MST) over 𝐺[V PPR]. Specifically, we assign each edge 𝑒a cost
based on the semantic match between the query and the relation
type of the edge as cost(𝑒)= 1−cos z𝑞,z𝑟(𝑒),wherez𝑞is the
query embedding, andz 𝑟(𝑒)=Encode(𝑥 𝑟(𝑒))is the embedding of
the relation type associated with edge 𝑒computed by the same
language encoder used in KG indexing, with 𝑥𝑟(𝑒)being the textual
description of relation 𝑟(𝑒). Then we compute the MST by applying
Kruskal’s algorithm [19]:
K4=MST 𝐺[V PPR].(5)
Kruskal’s algorithm greedily adds edges in increasing order of
cost(·) while avoiding cycles, thereby connecting the subgraph
with important entities via relations most relevant to the query.
3.4 Knowledge Preference Alignment
While the Mixture-of-Experts retrieval agent provides structured
KG knowledge, the retrieved triples or subgraphs are misaligned
with the natural-language inputs expected by LLM-based recom-
menders. Naive textual serialization can introduce noise and lose
structural information, degrading recommendation performance.
To address this issue, MixRAGRec introduces a knowledge prefer-
ence alignment agent that converts retrieved KG knowledge into
LLM-friendly representations while preserving recommendation-
relevant information.
LetKdenote the retrieved knowledge returned by the selected
retrieval expert. Expert 2 returns a set of triples, Expert 3 returns a
𝑘-hop neighborhood subgraph, Expert 4 returns a connected sub-
graph, and Expert 1 corresponds to no retrieval. Since the Kis
structured, we first transform it into a stable textual draft using a
carefully crafted template module Temp(·) . To be more specific, for
triple-level retrieval (i.e., Expert 2), Temp(·) converts each triple
(ℎ,𝑟,𝑡) into a short natural-language statement with relation-aware
patterns, e.g.,“{head} has {relation} {tail}”, and concatenates them
into a compact list of knowledge statements. For graph-based re-
trieval (i.e., Expert 3 and Expert 4), Temp(·) linearizes the retrieved
graph into a set of salient relational statements (e.g., edges along
short paths among the retrieved entities), providing a consistent
input format across retrieval granularities.
Given the query 𝑞and the templated text Temp(K) , the knowl-
edge preference alignment agent further refines the draft into
an aligned knowledge snippet ˜K=Align𝜓(𝑞,Temp(K)),where
Align𝜓(·)is the knowledge preference alignment agent and trained
to produce knowledge that is maximally useful to the downstream
recommendation agent. The aligned knowledge ˜Kserves as a tex-
tual interface between structured KG knowledge and LLM-based
recommendations and is appended to the downstream recommen-
dation prompt along with the user’s query.
3.5 Preference-Optimized Recommendation via
Contrastive Feedback
In knowledge-augmented recommendation, users’ contexts often
contain numerous reasonable items that are highly similar to thetarget item, such as items sharing the same genre, entities, or over-
lapping attributes. Such similar items can easily confuse the model,
especially when external knowledge from the KG is integrated.
Directly optimizing the recommender may fail to sufficiently dif-
ferentiate the target items from these strong alternatives, reducing
the recommendation performance. To this end, we introduce a con-
trastive learning-reinforced recommendation agent to optimize the
recommendation agent with contrastive preference feedback.
Specifically, given a user’s query 𝑞and aligned knowledge ˜K,
we define the recommendation context as 𝑥=(𝑞, ˜K). LetOdenote
the candidate item set under context 𝑥,𝑜★∈O is defined as the
target item and other items 𝑜−∈O\{𝑜★}are treated as negative
items. The objective is to train the model to assign a higher proba-
bility to the positive item than to negative items. However, directly
optimizing this objective is costly and less efficient due to the large
scale of the action space. To solve this problem, we introduce hard
negative sampling to select the top- 𝑁competitive negative items
according to the current recommendation policy, encouraging the
model to learn sharper distinctions among semantically similar
items. Formally, we define the hard negative set as:
Nhard(𝑥)=TopN𝑜∈O\{𝑜★}𝜋𝜔(𝑜|𝑥),(6)
and construct preference pairs (𝑥,𝑜★,𝑜−)with𝑜−∈N hard(𝑥). The
training objective is then defined as:
Lpref=−log𝜎
𝛽
log𝜋𝜔(𝑜★|𝑥)
𝜋ref(𝑜★|𝑥)−log𝜋𝜔(𝑜−|𝑥)
𝜋ref(𝑜−|𝑥)
,(7)
where𝜎(·) is the sigmoid function and 𝛽is a temperature hyper-
parameter. We introduce 𝜋refas an anchoring baseline to stabilize
learning, and optimize 𝜋𝜔to assign higher probability to the tar-
get item. When multiple hard negatives are available, we compute
Eq.(7)for each(𝑜★,𝑜−)pair and average the loss across negatives.
3.6 Mixture-of-Experts Multi-Agent Policy
Optimization (MMAPO)
Due to the intricate interconnections among the three agents, it
is not trivial to train MixRAGRec. Specifically, the recommenda-
tion performance depends on the selected retrieval expert, knowl-
edge alignment quality, and the recommendation agent’s capabili-
ties. These tight interconnections make it difficult to evaluate each
agent’s influence during training. Therefore, vanilla optimization
methods make the model tend to select the expert based on fixed
preferences rather than selecting experts based on the query.
To this end, we propose Mixture-of-Experts Multi-Agent Policy
Optimization (MMAPO), which coordinates the three agents under
a shared objective and enables the model to identify the most suit-
able retrieval expert based on the user’s query, as shown in Figure 2.
Specifically, MMAPO couples agents via a shared reward that com-
bines recommendation performance with a marginal information
gain. This design encourages flexible expert selection that balances
knowledge richness and efficiency, while jointly improving knowl-
edge alignment and recommendation performance.
3.6.1 Advantage estimation.In MixRAGRec, the retrieval expert
selected by the Mixture-of-Experts retrieval agent affects the final
reward only through the downstream aligned knowledge and the

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
subsequent recommendation. This delayed dependency makes the
retrieval policy difficult to optimize from raw returns. Therefore,
MMAPO adopts advantage-based credit assignment with learned
value baselines.
We unfold the pipeline into a trajectory of decision steps, where
the retrieval action and the alignment token generations are treated
as actions at different steps. Let 𝑟𝑡denote the reward at step 𝑡and
define the discounted return 𝐺𝑡=Í𝑇−𝑡
𝑙=0𝛾𝑙𝑟𝑡+𝑙,where𝛾∈( 0,1]
is the discount factor. We use a sparse terminal reward 𝑟𝑡=0
for𝑡<𝑇 and𝑟𝑇=𝑅 total. MMAPO maintains value baselines for
the retrieval and alignment agents. Specifically, the retrieval agent
uses a value network 𝑉𝜙(𝑠𝑡)over its state 𝑠𝑡, and the alignment
agent is equipped with a value head 𝑉𝜓(𝑐𝑡)defined on its genera-
tion context 𝑐𝑡(i.e., the prompt plus the generated prefix). Let 𝑉(·)
denote the corresponding value baseline (i.e., 𝑉(𝑢𝑡)=𝑉𝜙(𝑠𝑡)for
retrieval steps, and 𝑉(𝑢𝑡)=𝑉𝜓(𝑐𝑡)for alignment steps). We com-
pute the temporal-difference residual: 𝛿𝑡=𝑟𝑡+𝛾𝑉(𝑢𝑡+1)−𝑉(𝑢𝑡),
and estimate advantages using Generalized Advantage Estimation
(GAE):
𝐴GAE
𝑡=𝑇−𝑡∑︁
𝑙=0(𝛾𝜆)𝑙𝛿𝑡+𝑙,(8)
where𝜆∈[ 0,1]controls the bias–variance trade-off. This estimator
provides low-variance learning signals and propagates downstream
reward back to the retrieval decision and the alignment generation
process through their respective value baselines.
3.6.2 Reward Design.To encourage flexible expert selection that
balances recommendation performance and retrieval utility, we
introduce a two-dimensional reward function, as follows:
𝑅total=𝑅 rec+𝛼·𝑅 MIG,(9)
where𝛼is the weighting coefficient for coordinating the two re-
ward dimensions. The recommendation reward 𝑅recmeasures the
performance of the final recommendation:
𝑅rec=(
conf(𝑜),if𝑜=𝑜★,
−0.1,otherwise,(10)
where conf(𝑜)=𝜋 𝜔(𝑜|𝑞, ˜K,O) denotes the model confidence
assigned to the recommended item.
To make expert selection cost-aware, we introduce a marginal
information gain reward:
𝑅MIG=Δ𝐼(𝑒)−𝜂·𝐶(𝑒),whereΔ𝐼(𝑒)=𝐷 KL 𝑃expert∥𝑃base,(11)
where𝑃expert denotes the recommendation distribution when expert
𝑒is invoked,𝑃baseis a baseline distribution (e.g., Expert 1 without
retrieval), and 𝐶(𝑒) is a computational cost assigned to expert 𝑒,
increasing with retrieval complexity. By rewarding information
gain while penalizing retrieval cost, 𝑅MIGencourages expensive
retrieval only when it yields sufficient utility to justify the added
cost. MMAPO optimizes the retrieval and alignment policies under
this shared reward while training the recommendation agent with
contrastive preference feedback. The core procedure is summarized
in Algorithm 1, and the full training objective and update rules are
provided in Appendix B.Table 1: Basic statistics of three datasets and the KGs. “Items
in KG” indicates the number of items that appeared in both
the KG and the dataset.
Datasets MovieLens-1M MovieLens-20M LFM-1K
User6,040 138,287 992
Item3,883 27,278 107,528
Interaction1,000,209 20,000,263 19,150,868
Items in KG3,253 24,700 33,736
Entities24,840 72,673 180,457
Relations168 223 312
KG triples64,688 272,173 516,604
4 Experiments
In this section, we evaluate the effectiveness of MixRAGRec through
extensive experiments. We will first introduce the experimental
settings, including the datasets, compared baseline methods, eval-
uation metrics, and implementation details. Then, we report the
main experimental results, highlighting the performance of MixRA-
GRec compared with various strong baselines. Finally, we conduct
comprehensive additional analyses, including efficiency evaluation,
ablation studies, and hyper-parameter analysis.
4.1 Datasets
We evaluate the performance of MixRAGRec on three widely used
real-world recommendation datasets covering both movie and mu-
sic domains.MovieLens-1M1contains approximately one million
user-movie ratings, along with basic textual metadata such as movie
titles.MovieLens-20M2is a large-scale movie dataset containing
over 20 million ratings from more than 138,000 users for approxi-
mately 27,000 movies.LastFM-1K3is a music dataset that records
more than 19 million user listening histories and associated artist in-
formation. Following our framework design, we adopt DBpedia4as
the data source and construct corresponding knowledge graphs for
each dataset. The statistics of the datasets and their corresponding
KGs are summarized in Table 1.
4.2 Baselines
We compare MixRAGRec with a carefully constructed set of base-
lines to provide a comprehensive evaluation. These baselines cover
(i) Zero-shot Inference, (ii)LLM-based Recommendation Models,
and (iii) KG-RAG for LLM-based Recommendation Models.
Zero-shot inference.These baselines directly prompt LLMs such
asGemini-2.0-Flash,GPT-4o[ 17],Deepseek-r1[ 12],LLaMA3-8B[ 6],
andMistral-7Bto perform recommendations without any task-
specific fine-tuning or external knowledge augmentation.
LLM-based Recommendation Models.These methods adapt
LLMs to recommendation tasks by fine-tuning on recommenda-
tion datasets. We includeTallRec[ 2], which fine-tunes LLMs using
instruction-tuning for recommendation, andRec-r1[ 24], which fur-
ther adapts LLMs for reasoning-augmented recommendation by
reinforcement learning.
1https://grouplens.org/datasets/movielens/1m/
2https://grouplens.org/datasets/movielens/20m/
3http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
4https://www.dbpedia.org/

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
Table 2: Performance comparison across LLM-based recommendation baselines and KG-RAG methods. Best results are in bold
and the second-best results are underlined .
ModelMovieLens-1M MovieLens-20M LFM-1K
Accuracy Recall@3 Recall@5 Accuracy Recall@3 Recall@5 Accuracy Recall@3 Recall@5
Zero-shot Inference
Gemini-2.0-Flash 0.389 N/A N/A 0.467 N/A N/A 0.728 N/A N/A
GPT-4o 0.420 N/A N/A 0.524 N/A N/A 0.685 N/A N/A
Deepseek-r1 0.393 N/A N/A 0.445 N/A N/A 0.634 N/A N/A
LLaMA3-8B zero-shot 0.130 N/A N/A 0.129 N/A N/A 0.234 N/A N/A
LLM-based Recommendation Models
TallRec LLaMA3-8B [2] 0.391 0.663 0.779 0.521 0.781 0.868 0.699 0.848 0.891
Rec-r1 LLaMA3-8B [24] 0.402 0.558 0.634 0.594 0.761 0.815 0.872 0.926 0.938
KG-RAG for LLM-based Recommendation Models
KG-Text LLaMA3-8B [38] 0.234 N/A N/A 0.301 N/A N/A 0.545 N/A N/A
KAPING LLaMA3-8B [1] 0.232 N/A N/A 0.274 N/A N/A 0.530 N/A N/A
G-retriever LLaMA3-8B [14] 0.395 0.648 0.743 0.491 0.722 0.801 0.690 0.825 0.860
K-RagRec LLaMA3-8B [35] 0.454 0.659 0.707 0.578 0.720 0.755 0.845 0.886 0.906
MixRAGRec LLaMA3-8B (Ours) 0.504 0.798 0.882 0.676 0.900 0.940 0.934 0.990 0.992
Improvement 11.0% 20.4% 13.2% 13.8% 15.2% 8.3% 7.1% 6.9% 5.8%
Other Backbone LLM
Mistral-7B zero-shot 0.105 N/A N/A 0.097 N/A N/A 0.318 N/A N/A
TallRec Mistral-7B [2] 0.393 0.665 0.761 0.545 0.809 0.883 0.674 0.812 0.867
Rec-r1 Mistral-7B [24] 0.338 0.433 0.490 0.505 0.599 0.648 0.362 0.477 0.552
KG-Text Mistral-7B [38] 0.228 N/A N/A 0.277 N/A N/A 0.492 N/A N/A
KAPING Mistral-7B [1] 0.249 N/A N/A 0.272 N/A N/A 0.536 N/A N/A
G-retriever Mistral-7B [14] 0.372 0.641 0.775 0.542 0.758 0.846 0.763 0.856 0.890
K-RagRec Mistral-7B [35] 0.438 0.727 0.831 0.573 0.801 0.888 0.818 0.874 0.894
MixRAGRec Mistral-7B (Ours) 0.518 0.785 0.879 0.652 0.892 0.935 0.912 0.953 0.962
Improvement 18.3% 8.0% 5.8% 13.8% 10.3% 5.3% 11.5% 9.0% 7.6%
KG-RAG for LLM-based Recommendation Models.These base-
line models augment LLMs by retrieving structured knowledge
from the KG and are adapted in our experiments to support rec-
ommendation tasks. We include triple-level retrieval methods KG-
Text [ 38] and subgraph-level retrieval methods KAPING [ 1] and G-
retriever [ 14]. We also compare MixRAGRec with K-RagRec [ 35], de-
signed for KG-RAG recommendation tasks, which retrieves knowl-
edge subgraphs of different hops from KG for knowledge-augmented
recommendation.
4.3 Evaluation Metrics
We evaluate recommendation performance using Accuracy and
Recall@𝐾(𝐾=3,5). Accuracy measures whether the model’s rec-
ommendation matches the target item, while Recall@ 𝐾measures
whether the target item appears in the top- 𝐾recommended list.
Following recent studies [ 35,42], we select the item most recently
interacted with by the user as the target item and use the 10 interac-
tions preceding it as the user context. The model is then required to
predict the user’s preferred item from a candidate pool consisting
of the target item and 19 distractor items. For inference-based mod-
els that only output a single option without providing comparable
scores over all candidates, we report Accuracy only.
4.4 Parameter Settings
Our framework is implemented in PyTorch, and all experiments
are conducted on one NVIDIA H20-96G GPU. For the Mixture-of-
Experts retrieval agent and knowledge preference alignment agent,
we optimize the policy using a learning rate3 ×10−4, discount factor𝛾= 0.99, GAE parameter 𝜆= 0.95, clipping coefficient 𝜖= 0.2,
and4update epochs per iteration. For preference optimization in
the recommendation agent, we set the temperature parameter to
𝛽=0.2and employ hard negative sampling with 𝑁= 10negatives.
For the shared reward design, we set the MIG weight 𝛼=0.2and
the cost penalty 𝜂= 0.005. Notably, the knowledge preference
alignment agent and the recommendation agent adopt the same
LLM backbone architecture (e.g., LLaMA3-8B and Mistral-7B) with
independent parameters. Additional implementation details appear
in Appendix A.
4.5 Overall Performance Comparison
We compare the recommendation performance of MixRAGRec with
various carefully constructed baselines under the LLM backbones
LLaMA3-8B and Mistral-7B, on three datasets across three metrics.
The experimental results are reported in the Table 2. From the
comparison, we have the following key observations:
•Zero-shot prompting is insufficient for recommendation. Across
three datasets, zero-shot inference consistently underperforms
methods adapted to recommendation (e.g., TallRec/Rec-r1) and
knowledge-augmented approaches. Although closed-source LLMs
(e.g., GPT-4o and Deepseek-r1) achieved stronger zero-shot Accu-
racy than open-source backbones (e.g., LLaMA3-8B and Mistral-
7B), they still lag considerably behind the best baselines and
MixRAGRec, highlighting the limitation of direct prompting for
recommendations.
•Retrieving external knowledge from the KG can substantially im-
prove recommendation performance, while the benefit depends

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
Table 3: Efficiency comparison on MovieLens-1M with the
LLaMA3-8B backbone. We report average retrieval time and
end-to-end total time in seconds (s). ACC denotes Accuracy.
Methods ACC Retrieval Time (s) Total Time (s)
KG-Text 0.234 0.311 2.974
KAPING 0.232 0.247 2.708
G-retriever 0.395 2.411 2.907
K-RagRec 0.454 2.867 3.776
Expert 1 (DirectGenerator) - 0.000 0.042
Expert 2 (TripleRetriever) 0.488 0.066 1.827
Expert 3 (SubgraphRetriever) 0.491 0.212 2.103
Expert 4 (ConnectedGraphRetriever) 0.499 0.374 2.271
MixRAGRec 0.504 0.063 1.563
on how knowledge is retrieved and integrated. Directly injecting
retrieved KG knowledge as text (e.g., KG-Text and KAPING) does
not get consistent improvements over strong LLM-based recom-
mender systems. In contrast, methods that leverage graph-level
retrieval and representation (e.g., G-Retriever and K-RagRec)
generally achieve stronger performance, suggesting that preserv-
ing structural information is important for effective knowledge
augmentation.
•MixRAGRec outperforms the strongest baselines on all three
datasets under both backbones, with relative improvements rang-
ing from 5.8% to 20.4% under LLaMA3-8B and around 5.3% to
18.3% under Mistral-7B, highlighting the effectiveness of MixRA-
GRec. Notably, MixRAGRec also demonstrates significant im-
provement in the Recall@ 𝐾metric, indicating that the proposed
framework improves the quality of the top- 𝐾ranked recommen-
dations rather than only the Accuracy.
4.6 Efficiency Study
To evaluate the efficiency of MixRAGRec, we measure the aver-
age retrieval time and end-to-end total latency on MovieLens-1M
with the LLaMA3-8B backbone, using one NVIDIA H20-96G GPU.
The results are summarized in Table 3. While lightweight retrieval
methods like KG-Text and KAPING present fast retrieval efficiency,
they have limited accuracy. Although graph-level retrieval methods
(e.g., G-retriever and K-RagRec) achieve higher accuracy, they incur
significantly higher retrieval time and total latency. In particular,
K-RagRec exhibits the largest latency because it requires individual
retrieval for each item. In contrast, MixRAGRec achieves the best
accuracy while maintaining significantly lower retrieval time and
total latency than baselines, suggesting that flexible expert selec-
tion can preserve the benefits of structured knowledge without
unnecessary retrieval costs.
We further analyze different retrieval experts in MixRAGRec.
As the retrieval granularity increases from Expert 2 to Expert 4,
both retrieval time and accuracy increase, reflecting the expected
cost–utility trade-off of richer KG knowledge. It is worth noting
that MixRAGRec achieves the highest accuracy while keeping re-
trieval times close to the Expert 2, indicating that the learned policy
invokes expensive retrieval only when it provides sufficient utility.
These results highlight that MixRAGRec improves recommenda-
tion performance with lower cost, validating the effectiveness and
efficiency of the proposed framework. We also analyze the retrieval
complexity in Appendix C.
w/o MoESelw/o Align
w/o HardNegw/o PrefOpt w/o MMAPO MixRAGRec0.4800.4850.4900.4950.5000.5050.510ACCURACY
ML-1M
w/o MoESelw/o Align
w/o HardNegw/o PrefOpt w/o MMAPO MixRAGRec0.880.890.900.910.920.930.94ACCURACY
LFM-1K
w/o MoESelw/o Align
w/o HardNegw/o PrefOpt w/o MMAPO MixRAGRec0.740.750.760.770.780.790.800.81RECALL@3
ML-1M
w/o MoESelw/o Align
w/o HardNegw/o PrefOpt w/o MMAPO MixRAGRec0.950.960.970.980.99RECALL@3
LFM-1K
w/o MoESelw/o Align
w/o HardNegw/o PrefOpt w/o MMAPO MixRAGRec0.800.820.840.860.88RECALL@5
ML-1M
w/o MoESelw/o Align
w/o HardNegw/o PrefOpt w/o MMAPO MixRAGRec0.960.970.980.99RECALL@5
LFM-1KFigure 4: Ablation results of MixRAGRec and five variants on
MovieLens-1M and LFM-1K with the LLaMA3-8B backbone.
We report Accuracy, Recall@3, and Recall@5.
4.7 Ablation Study
We conduct ablation studies with LLaMA3-8B on MovieLens-1M
and LFM-1K to assess the contribution of each key component in
MixRAGRec. Specifically, we evaluate five ablated variants: (i)w/o
MoESel, which replaces the Mixture-of-Experts selector policy
with random expert selection; (ii)w/o Align, which removes the
Knowledge Preference Alignment agent and uses only the tem-
plated verbalization Temp(𝐾) as input to the recommender; (iii)
w/o HardNeg, which replaces hard negative sampling with ran-
dom negatives in contrastive preference training; (iv)w/o PrefOpt,
which removes contrastive preference feedback for training the
recommender; and (v)w/o MMAPO, which removes the shared
objective and coordinated updates by dropping the MIG term and
optimizing each agent separately with an accuracy-only reward us-
ing generic policy-gradient updates. The results are demonstrated
in Figure 4.
Overall, the full MixRAGRec achieves the best performance
across all metrics on both datasets, demonstrating the effectiveness
of each component. Notably, the w/o Align variant resulted in the
most significant performance degradation, especially on the LFM-
1K dataset, suggesting that the Knowledge Preference Alignment
agent is crucial for bridging the gap between retrieval structured
knowledge and LLM inputs. In addition, removing hard negative
sampling or contrastive feedback consistently degrades Accuracy
and Recall@ 𝐾on both datasets, highlighting the importance of

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
0 0.1 0.2 0.5
0.4800.4850.4900.4950.500ACCURACY
(a) ML1M ACC
0 0.1 0.2 0.5
0.750.760.77RECALL@3
 (b) ML1M R@3
0 0.1 0.2 0.5
0.10.20.3Retrieve Time (s)
 (c) Average Retrieval Time
Figure 5: Effect of the MIG weight 𝛼on MovieLens-1M with
the LLaMA3-8B backbone.
1 3 5 10 15
N0.4000.4250.4500.4750.500ACCURACY
(a) ML1M ACC
1 3 5 10 15
N0.60.70.8RECALL@3
 (b) ML1M R@3
1 3 5 10 15
N0.60.70.80.9RECALL@5
 (c) ML1M R@5
Figure 6: Effect of the number of hard negatives 𝑁on
MovieLens-1M with the LLaMA3-8B backbone.
1 3 5 10
M0.480.490.500.51ACCURACY
(a) ML1M ACC
1 3 5 10
M0.750.760.77RECALL@3
 (b) ML1M R@3
1 3 5 10
M0.820.830.840.85RECALL@5
 (c) ML1M R@5
Figure 7: Effect of the retrieval budget 𝑀on MovieLens-1M
with the LLaMA3-8B backbone.
explicitly separating the target item from highly competitive neg-
atives in knowledge-augmented recommendation. Moreover, re-
placing the Mixture-of-Experts retrieval agent with random expert
selection and removing MMAPO both lead to sub-optimal results,
demonstrating the effectiveness of our proposed Mixture-of-Experts
retrieval and optimization approach.
4.8 Hyper-parameter Analysis
In this sub-section, we study the impact of three key hyper-parameters
of MixRAGRec on MovieLens-1M with LLaMA3-8B. Specifically,
we evaluate the marginal information gain weight 𝛼, the number
of hard negatives𝑁, and the retrieval budget𝑀.
4.8.1 Effect of the MIG weight 𝛼.We vary the MIG weight 𝛼to
control the trade-off between retrieval utility and recommendation
effectiveness in the shared reward. As shown in Figure 5, setting
𝛼= 0bases the shared reward solely on recommendation accu-
racy, resulting in the longest average retrieval time and relatively
poorer ranking quality (Recall@3/5). Increasing 𝛼to a moderate
value improves ranking performance while reducing retrieval time.
In particular, 𝛼= 0.2achieves the best Recall@3/5 performance
while significantly reducing the average retrieval time, suggesting
that an appropriate MIG 𝛼encourages more cost-aware retrieval
while guaranteeing the recommendation performance. When 𝛼is
further increased to 0.5, retrieval efficiency improves slightly but
performance is significantly reduced. Overall, these results high-
light the importance of selecting an appropriate MIG weight to
balance retrieval utility and recommendation effectiveness.4.8.2 Effect of the number of hard negatives 𝑁.We analyze the
impact of the number of hard negatives on recommendation perfor-
mance on MovieLens-1M with the LLaMA3-8B backbone. Results
are shown in Figure 6. Using too few hard negatives severely de-
grades performance. For example, when 𝑁= 1, both Accuracy
and Recall@3/5 drop substantially, indicating that a single negative
provides an insufficient contrastive signal. Increasing 𝑁to 5 and 10
improves both Accuracy and Recall@ 𝐾, suggesting that additional
hard negatives help the model better distinguish target items from
competitive candidates. Further increasing 𝑁to 15 yields limited
gains with trade-offs across metrics. Accordingly, we set 𝑁= 10in
our experiments.
4.8.3 Effect of the retrieval budget 𝑀.We study the effect of the
retrieval budget 𝑀on MovieLens-1M with the LLaMA3-8B back-
bone. The budget 𝑀controls the retrieval scale across experts,
and is instantiated differently for different granularities. For triple-
level retrieval, we return the top-𝑐𝑀triples (with𝑐=5), while for
subgraph-based retrieval, we use 𝑀to determine the number of
retained nodes (or seed entities) before graph expansion. Results
are shown in Figure 7. We observe that a small budget can miss
useful knowledge, while a large budget may introduce noise and
increase the burden on downstream alignment and utilization. On
MovieLens-1M, 𝑀= 3and𝑀= 5yield optimal Accuracy and
Recall, respectively. Overall, it is important to select an appropriate
retrieval budget based on the scale of the dataset and the KG.
5 Conclusion
In this work, we present MixRAGRec, a cooperative multi-agent
framework for KG-RAG recommendation that improves both ef-
fectiveness and efficiency by enabling query-aware retrieval and
knowledge utilization. Specifically, MixRAGRec includes a Mixture-
of-Experts Retrieval Agent that selects among KG retrieval experts
with different granularities, a Knowledge Preference Alignment
Agent that converts structured KG knowledge into LLM-friendly
text, and a Contrastive Learning-reinforced Recommendation Agent
trained with contrastive preference feedback to better distinguish
the target item from competitive candidates. To jointly train the
three agents, we propose Mixture-of-Experts Multi-Agent Policy
Optimization (MMAPO), which coordinates learning under a shared
objective with recommendation feedback and marginal information
gain. Experiments on three real-world datasets demonstrate the
effectiveness and efficiency of MixRAGRec.
Acknowledgments
The research described in this paper has been partially supported by
the Key Special Project of National Natural Science Foundation of
China (Project No. 72442017), the General Research Funds from the
Hong Kong Research Grants Council (project No. PolyU 15207322,
15200023, 15206024, and 15224524), Internal research funds from
Hong Kong Polytechnic University (project no. P0059586, P0042693,
P0048625, and P0051361), and Sheertek International (HK) Limited.
This work was supported by computational resources provided
by The Centre for Large AI Models (CLAIM) of The Hong Kong
Polytechnic University.

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
References
[1]Jinheon Baek, Alham Fikri Aji, and Amir Saffari. 2023. Knowledge-Augmented
Language Model Prompting for Zero-Shot Knowledge Graph Question Answer-
ing. InProceedings of the 1st Workshop on Natural Language Reasoning and Struc-
tured Explanations (NLRSE). 78–106.
[2]Keqin Bao, Jizhi Zhang, Yang Zhang, Wenjie Wang, Fuli Feng, and Xiangnan He.
2023. Tallrec: An effective and efficient tuning framework to align large language
model with recommendation. InProc. 17th ACM Conf. Recomm. Syst.1007–1014.
[3]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bog-
dan Damoc, Aidan Clark, et al .2022. Improving language models by retrieving
from trillions of tokens. InProc. Int. Conf. Mach. Learn.PMLR, 2206–2240.
[4]Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao
Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al .2024. A survey on
evaluation of large language models.ACM transactions on intelligent systems and
technology15, 3 (2024), 1–45.
[5]Yujuan Ding, PY Mok, Yi Bin, Xun Yang, and Zhiyong Cheng. 2023. Modeling
multi-relational connectivity for personalized fashion matching. InProceedings
of the 31st ACM international conference on multimedia. 7047–7055.
[6]Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models.arXiv e-prints(2024), arXiv–2407.
[7]Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InProc. ACM SIGKDD Int. Conf.
Knowl. Discov. Data Min.6491–6501.
[8]Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin.
2019. Graph neural networks for social recommendation. InProc. World Wide
Web Conf.417–426.
[9]Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023).
[10] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022.
Recommendation as language processing (rlp): A unified pretrain, personalized
prompt & predict paradigm (p5). InProc. 16th ACM Conf. Recomm. Syst.299–315.
[11] Esmail Gumaan. 2025. ExpertRAG: Efficient RAG with Mixture of Experts–
Optimizing Context Retrieval for Adaptive LLM Responses.arXiv preprint
arXiv:2504.08744(2025).
[12] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin
Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al .2025. Deepseek-r1:
Incentivizing reasoning capability in llms via reinforcement learning.arXiv
preprint arXiv:2501.12948(2025).
[13] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al .
2024. Retrieval-augmented generation with graphs (graphrag).arXiv preprint
arXiv:2501.00309(2024).
[14] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun,
Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented genera-
tion for textual graph understanding and question answering.Proc. Adv. Neural
Inf. Process. Syst.37 (2024), 132876–132907.
[15] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley,
and Wayne Xin Zhao. 2024. Large language models are zero-shot rankers for
recommender systems. InEuropean Conference on Information Retrieval. Springer,
364–381.
[16] Jiani Huang, Shijie Wang, Liang-bo Ning, Wenqi Fan, Shuaiqiang Wang, Dawei
Yin, and Qing Li. 2025. Towards Next-Generation Recommender Systems: A
Benchmark for Personalized Recommendation Assistant with LLMs.arXiv
preprint arXiv:2503.09382(2025).
[17] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh,
Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al .2024.
Gpt-4o system card.arXiv preprint arXiv:2410.21276(2024).
[18] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu,
Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented
generation. InProceedings of the 2023 Conference on Empirical Methods in Natural
Language Processing. 7969–7992.
[19] Joseph B Kruskal. 1956. On the shortest spanning subtree of a graph and the
traveling salesman problem.Proceedings of the American Mathematical society7,
1 (1956), 48–50.
[20] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al .
2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.Proc.
Adv. Neural Inf. Process. Syst.33 (2020), 9459–9474.
[21] Yuhan Li, Xinni Zhang, Linhao Luo, Heng Chang, Yuxiang Ren, Irwin King,
and Jia Li. 2025. G-refer: Graph retrieval-augmented large language model for
explainable recommendation. InProc. World Wide Web Conf.240–251.[22] Jiayi Liao, Sihang Li, Zhengyi Yang, Jiancan Wu, Yancheng Yuan, Xiang Wang,
and Xiangnan He. 2024. Llara: Large language-recommendation assistant. In
Proceedings of the 47th International ACM SIGIR Conference on Research and
Development in Information Retrieval. 1785–1795.
[23] Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan,
Ruiming Tang, Yong Yu, and Weinan Zhang. 2024. Rella: Retrieval-enhanced
large language models for lifelong sequential behavior comprehension in recom-
mendation. InProc. World Wide Web Conf.3497–3508.
[24] Jiacheng Lin, Tian Wang, and Kun Qian. 2025. Rec-r1: Bridging generative large
language models and user-centric recommendation systems via reinforcement
learning.arXiv preprint arXiv:2503.24289(2025).
[25] Xinyu Lin, Wenjie Wang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua.
2024. Bridging items and language: A transition paradigm for large language
model-based recommendation. InProc. ACM SIGKDD Int. Conf. Knowl. Discov.
Data Min.1816–1826.
[26] Reza Yousefi Maragheh, Pratheek Vadla, Priyank Gupta, Kai Zhao, Aysenur Inan,
Kehui Yao, Jianpeng Xu, Praveen Kanumala, Jason Cho, and Sushant Kumar. 2025.
ARAG: Agentic Retrieval Augmented Generation for Personalized Recommenda-
tion.arXiv preprint arXiv:2506.21931(2025).
[27] Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph neural retrieval
for efficient large language model reasoning on knowledge graphs. InFindings of
the Association for Computational Linguistics: ACL 2025. 16682–16699.
[28] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar,
Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2025. A com-
prehensive overview of large language models.ACM Transactions on Intelligent
Systems and Technology16, 5 (2025), 1–72.
[29] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel
Cer, and Yinfei Yang. 2022. Sentence-t5: Scalable sentence encoders from pre-
trained text-to-text models. InFindings of the association for computational lin-
guistics: ACL 2022. 1864–1874.
[30] Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd. 1999.The
PageRank citation ranking: Bringing order to the web.Technical Report. Stanford
infolab.
[31] Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi,
Rami Al-Rfou, and Jonathan Halcrow. 2024. Let your graph do the talking:
Encoding structured data for llms.arXiv preprint arXiv:2402.05862(2024).
[32] Zhangchi Qiu, Linhao Luo, Zicheng Zhao, Shirui Pan, and Alan Wee-Chung Liew.
2025. Graph Retrieval-Augmented LLM for Conversational Recommendation
Systems. InPacific-Asia Conference on Knowledge Discovery and Data Mining.
Springer, 344–355.
[33] Haohao Qu, Wenqi Fan, Zihuai Zhao, and Qing Li. 2025. Tokenrec: Learning to
tokenize id for llm-based generative recommendations.IEEE Trans. Knowl. Data
Eng.(2025).
[34] J Ben Schafer, Joseph A Konstan, and John Riedl. 2001. E-commerce recommen-
dation applications.Data mining and knowledge discovery5, 1 (2001), 115–153.
[35] Shijie Wang, Wenqi Fan, Yue Feng, Lin Shanru, Xinyu Ma, Shuaiqiang Wang, and
Dawei Yin. 2025. Knowledge graph retrieval-augmented generation for llm-based
recommendation. InProceedings of the 63rd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers). 27152–27168.
[36] Shijie Wang, Wenqi Fan, Xiao-Yong Wei, Xiaowei Mei, Shanru Lin, and Qing
Li. 2024. Multi-agent attacks for black-box social recommendations.ACM
Transactions on Information Systems43, 1 (2024), 1–26.
[37] Shijie Wang, Jiani Huang, Zhikai Chen, Yu Song, Wenzhuo Tang, Haitao Mao,
Wenqi Fan, Hui Liu, Xiaorui Liu, Dawei Yin, et al .2025. Graph machine learning
in the era of large language models (llms).ACM Transactions on Intelligent Systems
and Technology16, 5 (2025), 1–40.
[38] Yike Wu, Nan Hu, Sheng Bi, Guilin Qi, Jie Ren, Anhuan Xie, and Wei Song. 2023.
Retrieve-rewrite-answer: A kg-to-text enhanced llms framework for knowledge
graph question answering.arXiv preprint arXiv:2309.11206(2023).
[39] Jian Xu, Sichun Luo, Xiangyu Chen, Haoming Huang, Hanxu Hou, and Linqi
Song. 2025. RALLRec: Improving Retrieval Augmented Large Language Model
Recommendation with Representation Learning. InCompanion Proceedings of
the ACM on Web Conference 2025. 1436–1440.
[40] Wooseong Yang, Weizhi Zhang, Yuqing Liu, Yuwei Han, Yu Wang, Junhyun Lee,
and Philip S Yu. 2025. Cold-Start Recommendation with Knowledge-Guided
Retrieval-Augmented Generation.arXiv preprint arXiv:2505.20773(2025).
[41] Chao Zhang, Shiwei Wu, Haoxin Zhang, Tong Xu, Yan Gao, Yao Hu, and Enhong
Chen. 2024. Notellm: A retrievable large language model for note recommenda-
tion. InCompanion Proceedings of the ACM Web Conference 2024. 170–179.
[42] Junjie Zhang, Yupeng Hou, Ruobing Xie, Wenqi Sun, Julian McAuley, Wayne Xin
Zhao, Leyu Lin, and Ji-Rong Wen. 2024. Agentcf: Collaborative learning with
autonomous language agents for recommender systems. InProc. World Wide Web
Conf.3679–3689.
[43] Yang Zhang, Fuli Feng, Jizhi Zhang, Keqin Bao, Qifan Wang, and Xiangnan He.
2025. Collm: Integrating collaborative embeddings into large language models
for recommendation.IEEE Trans. Knowl. Data Eng.(2025).
[44] Zihuai Zhao, Yujuan Ding, Wenqi Fan, and Qing Li. 2025. WebRec: Enhancing
LLM-based Recommendations with Attention-guided RAG from Web.arXiv

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
preprint arXiv:2511.14182(2025).
[45] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen
Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, et al .2024. Recommender systems
in the era of large language models (llms).IEEE Trans. Knowl. Data Eng.36, 11
(2024), 6889–6907.
[46] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming
Chen, and Ji-Rong Wen. 2024. Adapting large language models by integrating
collaborative semantics for recommendation. In2024 IEEE 40th International
Conference on Data Engineering (ICDE). IEEE, 1435–1448.
[47] Yaochen Zhu, Chao Wan, Harald Steck, Dawen Liang, Yesu Feng, Nathan Kallus,
and Jundong Li. 2025. Collaborative Retrieval for Large Language Model-based
Conversational Recommender Systems. InProc. World Wide Web Conf.3323–
3334.
Appendix
A Implementation Details
The implementation details of MixRAGRec are summarized in
Table 4. The first block reports the optimization-related hyper-
parameters for the Mixture-of-Experts retrieval agent and the knowl-
edge alignment agent, including the learning rate (3 ×10−4), dis-
count factor ( 𝛾=0.99), GAE parameter ( 𝜆=0.95), retrieval budget
(Top-𝑀with𝑀= 3), clipping coefficient ( 𝜖=0.2), and training
epochs. The optimization is set with batch size 64, buffer size 2048,
and gradient clipping with max norm 0.5. The second block lists
the preference optimization settings for the recommendation agent,
including the temperature 𝛽=0.2and the number of hard nega-
tives𝑁= 10. The third block presents the shared reward hyper-
parameters for marginal information gain, namely the MIG weight
𝛼= 0.2and cost penalty 𝜂= 0.005. The remaining blocks pro-
vide the parameter-efficient fine-tuning configuration (LoRA rank
𝑟=16, scaling𝛼LoRA=32, dropout 0.1, and int8/fp16 settings) and
the decoding/evaluation protocol (decoding temperature 0.8, candi-
date pool size 20, and Recall@ 𝐾with𝐾∈{ 3,5}). In addition, for
KG indexing and similarity-based retrieval, we encode queries, en-
tities, and triples into the same semantic space using a pre-trained
sentence encoder, sentence-transformers/all-MiniLM-L6-v2 .
This encoder is used to build the entity and triple vector databases,
and retrieval is conducted via Top- 𝑀similarity computation over
the embedding space.
B MMAPO: Detailed Update Rules
With advantages from Eq. (8)and the shared reward in Eq. (9),
MMAPO updates the three agents in a coordinated manner. As the
retrieval and alignment agents affect the recommendation through
sequential decisions, we adopt a clipped update rule to keep each
policy update stable and prevent overly large changes between
iterations. Specifically, for the Mixture-of-Experts retrieval agent
and the knowledge preference alignment agent, we maximize the
following objective:
Jprox(𝜋)=Eh
min
𝜌𝑡𝐴GAE
𝑡,clip(𝜌𝑡,1−𝜖,1+𝜖)𝐴GAE
𝑡i
,
𝜌𝑡=exp(log𝜋(𝑎𝑡|𝑢𝑡)−log𝜋 old(𝑎𝑡|𝑢𝑡)),(12)
where𝑎𝑡is the action at step 𝑡(i.e., selecting an expert or generating
a token),𝑢𝑡denotes the corresponding input to the policy, and 𝜖
controls the update range.
For the contrastive learning-reinforced recommendation agent,
we optimize the contrastive preference objective in Section 3.5,
which encourages the model to assign a higher probability to thetarget item than to hard negatives. By updating the three agents,
MMAPO enables MixRAGRec to learn when to retrieve, how to
present KG knowledge, and how to leverage the aligned knowledge
for recommendation. The core procedure of the MMAPO algorithm
is presented in Algorithm 1.
Table 4: Hyper-parameter settings of MixRAGRec.
Item Value
Policy Optimization (Retrieval & Alignment)
learning rate 3×10−4
discount factor𝛾 0.99
GAE parameter𝜆 0.95
Retrieval budget𝑀 3
clipping coefficient𝜖 0.2
epochs 3
Preference Optimization (Recommendation)
temperature𝛽 0.2
number of hard negatives𝑁 10
Shared Reward (MIG)
MIG weight𝛼 0.2
cost penalty𝜂 0.005
Parameter-efficient Fine-tuning (LoRA)
LoRA rank𝑟 16
LoRA scaling𝛼 LoRA 32
int8 True
fp16 True
LoRA dropout 0.1
Decoding & Evaluation Protocol
decoding temperature 0.8
candidate pool size 20
evaluated Recall@𝐾 𝐾∈{3,5}
C Retrieval Complexity
We analyze the retrieval complexity of MixRAGRec, compared
with two representative baselines. Table 5 summarizes the worst-
case retrieval time complexity. Let 𝑀denote the retrieval budget
(Top-𝑀),𝐿the user history length, and 𝑑the embedding dimen-
sion. We use|𝐾|to denote the size of returned structured knowl-
edge (e.g., the number of triples/edges). G-retriever performs query-
level retrieval by retrieving relevant nodes and triples/edges from
their embedding spaces and then constructing a connected sub-
graph (e.g., via a Steiner-tree-style procedure), resulting in a cost of
𝑂((|V|+|T|)·(𝑑+log𝑀)+|𝐾| ). In contrast, K-RagRec indexes
multi-hop item-centered subgraphs in a single vector database and
conducts similarity search for each item in the user history. Thus,
its retrieval complexity scales linearly with the user history length
𝐿, i.e.,𝑂 𝐿· |S|·(𝑑+log𝑀)+|𝐾|.
Our Mixed KG experts span different retrieval granularities with
distinct costs. Expert 1 incurs no retrieval cost. Expert 2 performs
Top-𝑀triple retrieval over T, costing𝑂(|T|·(𝑑+log𝑀)). Ex-
pert 3 and Expert 4 first retrieve Top- 𝑀seed entities fromVand
then expand/construct subgraphs. Therefore, the complexity is
𝑂(|V|·(𝑑+log𝑀)+|𝐾 𝑒|)for𝑒∈{ 3,4}, where|𝐾𝑒|is linear in
the size of the returned subgraph.
Importantly, MixRAGRec does not commit to a single retrieval
strategy. Instead, it flexibly routes each query to an appropriate
expert, enabling a query-dependent trade-off between retrieval cost

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
Algorithm 1MixRAGRec (MMAPO Training)
Require: Training setDwith instances(𝑞,O,𝑜★); KG𝐺; retrieval experts
Eexp; retrieval policy 𝜋𝜃with value𝑉𝜙; alignment policy 𝜋𝜓with
value head𝑉𝜓; recommender policy 𝜋𝜔with reference 𝜋ref; hyper-
parameters𝛾,𝜆,𝜖 (advantage estimation and proximal update), 𝛼,𝜂
(MIG),𝛽(preference update),𝑁(hard negatives), retrieval budget𝑀.
1:Initialize parameters(𝜃,𝜙,𝜓,𝜔).
2:foreach training iterationdo
3:Sample a mini-batchB⊂D.
4:Initialize buffersT(trajectories) andP(preference pairs).
5:foreach(𝑞,O,𝑜★)∈Bdo
6:Construct retrieval state𝑠from𝑞.
7: Select retrieval expert 𝑒∼𝜋 𝜃(· |𝑠) and retrieve structured
knowledgeK 𝑒from𝐺with budget𝑀.
8:ConvertK 𝑒into a stable textual draft ˆK←Temp(K 𝑒).
9:Generate aligned evidence ˜K∼𝜋 𝜓(·|𝑞, ˆK).
10: Predict item distribution 𝑃expert(·)←𝜋 𝜔(·|𝑞, ˜K,O) and obtain
output𝑜.
11: Compute shared reward 𝑅total←𝑅 rec+𝛼·𝑅 MIG, where𝑅recmea-
sures recommendation correctness/confidence and 𝑅MIGmeasures
marginal information gain with a retrieval cost penalty (using
Expert 1 as the baseline to form𝑃 base).
12: Store the coupled trajectory (states/contexts and actions for expert
selection and alignment token generation) into Twith terminal
reward𝑅 total.
13: Mine hard negatives Nhard←TopN𝑜′∈O\{𝑜★}𝜋𝜔(𝑜′|𝑞, ˜K,O) .
14: Add preference pairs (𝑥,𝑜★,𝑜−)intoPfor each𝑜−∈ N hard,
where𝑥=(𝑞, ˜K).
15:end for
16:Advantage estimation.
17: Estimate advantages for retrieval using 𝑉𝜙(𝑠)and for alignment
using𝑉 𝜓(𝑐)(Eq. 8).
18:Coordinated policy update (MMAPO).
19: Update the retrieval selector 𝜋𝜃and alignment policy 𝜋𝜓via a prox-
imal clipped policy update driven by the estimated advantages.
20: Update the recommender 𝜋𝜔via preference-based contrastive opti-
mization onPwith reference𝜋 ref.
21:end for
22:returnTrained parameters(𝜃∗,𝜓∗,𝜔∗).
and knowledge utility. As a result, MixRAGRec can avoid unneces-
sary heavy retrieval for simple queries while still invoking richer
subgraph retrieval when needed, yielding a favorable accuracy–
efficiency balance compared with baselines that rely on a fixed
retrieval pipeline or perform retrieval repeatedly along the user
history.
Table 5: Comparison of retrieval time complexity. 𝑀is the
retrieval budget (Top-𝑀) and𝐿is the user history length.
Method Online retrieval time complexity
G-retriever 𝑂((|V|+|T|)·(𝑑+log𝑀)+|𝐾| )
K-RagRec 𝑂 𝐿· |S|·(𝑑+log𝑀)+|𝐾|
Expert 1 (DirectGenerator) 𝑂(1)
Expert 2 (TripleRetriever) 𝑂(|T|·(𝑑+log𝑀))
Expert 3 (SubgraphRetriever) 𝑂(|V|·(𝑑+log𝑀)+|𝐾 𝑒|),𝑒=3
Expert 4 (ConnectedGraphRetriever) 𝑂(|V|·(𝑑+log𝑀)+|𝐾 𝑒|),𝑒=4
MixRAGRec depends on the selected expert aboveTable 6: Cold-start recommendation performance compari-
son with the strongest KG-RAG-enhanced LLM recommenda-
tion methods on LLaMA3-8B across three metrics. The best
performances are labeled in bold.
Methods Accuracy Recall@3 Recall@5
LLaMA3-8B zero-shot 0.083 N/A N/A
G-retriever 0.356 0.641 0.738
K-RagRec 0.392 0.706 0.748
MixRAGRec 0.501 0.778 0.846
D Expert Selection Distribution
We further analyze the routing behavior of the Mixture-of-Experts
retrieval agent under both standard and cold-start settings. Figure 9
reports the expert selection distribution on LFM-1K with the Mistral-
7B backbone. The agent allocates most instances to Expert 1–3,
with proportions of 30.2%, 28.6%, and 28.2%, respectively, while
Expert 4 is selected less frequently (13.0%). This pattern indicates
that, for LFM-1K, the selector predominantly relies on lightweight
or moderate-granularity retrieval and invokes the most expensive
expert only for a smaller subset of instances.
We also examine the selection distribution in the cold-start sce-
nario. As shown in Figure 10, MixRAGRec chooses Expert 3 for
37.9% of instances and Expert 4 for 58.7% of instances, suggesting
that cold-start queries more often benefit from higher-granularity
relational knowledge. These results highlight that MixRAGRec
adapts retrieval granularity to different data contexts rather than
using a fixed strategy.
E Cold-start Recommendation Study
Cold-start recommendation remains an important challenge in prac-
tice. To evaluate MixRAGRec under this setting, we compare it with
strong KG-RAG-enhanced baselines on MovieLens-1M using the
LLaMA3-8B backbone, as reported in Table 6. Zero-shot prompting
performs poorly in the cold-start scenario, which further motivates
grounding recommendations with RAG. Compared with other KG-
RAG baselines, MixRAGRec achieves the best results and remains
robust under the cold-start setting, demonstrating the effectiveness
of our framework.
We further examine the expert-selection behavior in cold-start
scenarios. Figure 10 shows that MixRAGRec selects subgraph-level
retrieval (Expert 3) for 37.9% of queries and connected-graph re-
trieval (Expert 4) for 58.7% of queries, suggesting that cold-start
instances frequently require richer relational context. This trend
aligns with the gains in Table 6 and highlights the effectiveness
of MixRAGRec in adapting retrieval granularity to the cold-start
recommendations.
F More Related Work
Large language models (LLMs) have recently emerged as a promis-
ing paradigm for recommendation due to their strong semantic
understanding and generalization capabilities. Early research refor-
mulates recommendations as a language modeling task to leverage
LLMs’ language understanding and in-context learning capabilities
for user preference modeling [ 2,10,25]. For example, LLMRank [ 15]

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
0 5 10 15 20 25 30
Percentage (%)starring
others (213 types)
birth place
writer
director
producer
distributor
music composer
cinematography
editing
death place28.6%
20.4%
8.5%
8.0%
7.2%
7.0%
6.3%
5.5%
3.6%
2.6%
2.4%
(a) KG for MovieLens-20M
0 5 10 15 20
Percentage (%)genre
associated band
associated musical artist
others (302 types)
record label
hometown
instrument
birth place
occupation
former band member
band member19.0%
14.6%
14.6%
12.3%
10.3%
8.6%
5.6%
5.2%
4.2%
2.9%
2.6% (b) KG for LFM-1K
Figure 8: Relation-type distributions of the constructed KGs for (a) MovieLens-20M and (b) LFM-1K.
Figure 9: Expert selection distribution of the Mixture-of-
Experts retrieval agent on LFM-1K with Mistral-7B.
Figure 10: Expert selection distribution of the Mixture-of-
Experts retrieval agent on cold-start recommendation sce-
nario.
employs instruction-based prompts constructed from historical in-
teractions to perform item re-ranking, whereas TALLRec [ 2] intro-
duces an instruction-tuning framework that directly recommends
target items by modeling user interest from text-enriched historicalinteractions (e.g., item titles). To further enhance performance, sub-
sequent efforts incorporate collaborative signals into LLM-based
recommendations. Specifically, LLaRa [ 22] and CoLLM [ 43] treat
collaborative information as an additional modality and align it
with the LLM token space through learned mappings. Moreover,
tokenization-based approaches encode external signals into item
identifiers, representing items as discrete semantic or collaborative
indices to facilitate LLM-based recommendation [ 33,46]. Despite
these advancements, LLM-based recommender systems remain con-
strained by static parametric knowledge and have limitations in
obtaining up-to-date information. Therefore, to avoid costly fine-
tuning, it is important to introduce RAG to incorporate up-to-date
external knowledge.
G Knowledge Graph Statistics
To support the KG-RAG recommendations, we construct an item-
centric knowledge graph (KG) for each dataset. Table 1 reports
basic statistics of the three real-world datasets and their associated
KGs, including the numbers of users, items, interactions, and the
aligned items that appear in both the dataset and the KG (Items in
KG). Figure 8 further visualizes the relation-type distributions for
the MovieLens-20M and LFM-1K KGs. Overall, the KG coverage is
substantial across datasets, providing rich, structured knowledge
for grounding LLM-based recommendations. In addition, the KGs
exhibit diverse schema complexity, with hundreds of relation types
and up to hundreds of thousands of triples, which motivates re-
trieval designs that can flexibly control the granularity and amount
of retrieved knowledge.
H Prompts used in MixRAGRec
We present the prompts for our MixRAGRec in Table 7 and 8. Specif-
ically, the prompt used for Knowledge Preference Alignment Agent
is shown in Tables 7, and the prompt used for Contrastive Learning-
reinforced Recommendation Agent is shown in Tables 8.
I Case Study
To provide an intuitive understanding of how MixRAGRec oper-
ates in practice, we present three representative instances from
MovieLens-1M and LFM-1K in Tables 9, 10, and 11. All cases follow

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
the same pipeline: MixRAGRec first routes the input instance to
a retrieval expert, optionally retrieves structured KG knowledge,
aligns the knowledge into concise natural language, and finally
produces a recommendation over the candidate pool.
Table 9 shows a MovieLens-1M example routed toExpert 2
(TripleRetriever). The retrieved triples highlight significant re-
lations related to the user’s recent interests, and the alignment
agent converts them into a short, fluent knowledge snippet directly
consumable by the recommendation agent. Conditioned on this
aligned knowledge, the model assigns a highly peaked probability
mass to the correct option, illustrating how lightweight triple-level
retrieval can be sufficient when a small number of key facts already
distinguish the target item.
Table 10 presents a MovieLens-1M instance where the selector
choosesExpert 3 (SubgraphRetriever). Compared with isolated
triples, subgraph-level retrieval provides richer relational context.
In this example, the retrieved subgraph captures a compact family-
oriented cluster, and the alignment agent converts it into an LLM-
friendly description for downstream recommendation. Conditionedon the aligned knowledge, the recommendation agent ranks the
ground-truth option highest, illustrating the benefit of subgraph-
level evidence when a multi-relation context is needed.
Finally, Table 11 shows an LFM-1K instance routed toExpert 1
(DirectGenerator), where MixRAGRec skips retrieval and align-
ment entirely. The listening history already provides strong textual
cues for the backbone model, and bypassing retrieval avoids unnec-
essary overhead while still yielding a confident correct prediction.
This case highlights that MixRAGRec can preserve efficiency by
not invoking retrieval when external evidence is unlikely to offer
additional benefit.
Overall, these cases reflect the key advantage of MixRAGRec in
controlling the effectiveness–utility trade-off. It can rely on light-
weight evidence when it is sufficient (Table 9), avoid over-retrieval
that may introduce noise (Table 10), and skip retrieval altogether
for easy instances (Table 11).

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
Table 7: General prompt template for the Knowledge Preference Alignment Agent.
Knowledge Preference Alignment Agent Prompt
Shared Input
User Query:Watching history:“Movie A”, “Movie B”, . . . , “Movie J”.
User Preferences:general user
Domain:Movie/Music
History Label:"watching history"/ "listening history"
Item Type:"movie"/"artist"
Expert 2 (Triple-level Retrieval)
Task Description:Convert the following factual triples into natural language knowledge suitable for movie recommenda-
tions.
Retrieved Triples:{structured_triples_from_KG}
Instructions:
(1) Extract relevant facts from the triples
(2) Convert to natural fluent text
(3) Focus on genres, themes, directors, and user preferences
Expert 3 (Neighborhood Subgraph)
Task Description:Transform the following knowledge graph subgraph into natural language suitable for movie recom-
mendations.
Retrieved Subgraph:{structured_subgraph_from_KG}
Instructions:
(1) Identify key entities and relationships
(2) Describe connection patterns
(3) Highlight genres, themes, and stylistic similarities
(4) Keep description concise and relevant
Expert 4 (Connected Subgraph)
Task Description:Transform the following connected knowledge graph into natural language for movie recommendations.
Retrieved Connected Graph:{pagerank_mst_graph_from_KG}
Instructions:
(1) Extract core themes from connected entities
(2) Describe shared characteristics
(3) Focus on genres, directors, and cinematography
(4) Provide a coherent narrative summary
Output
Knowledge Summary:{aligned_knowledge_summary}

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
Table 8: General prompt template for the Contrastive Learning-reinforced Recommendation Agent.
Contrastive Learning-reinforced Recommendation Agent Prompt
Shared Input
Domain:Movie/Music
History Label:"watching history"/ "listening history"
Item Type:"movie"/"artist"
Default (with KG knowledge)
Instruction:Based on the user’s watching history, select the best movie from options A-T.
User Query:Watching history:“Movie A”, “Movie B”, . . . , “Movie J”.
Knowledge:{aligned_knowledge_summary}
Options:“A: Movie”, “B: Movie”, . . . , “T: Movie”.
Select a movie from options A to T that the user is most likely to be interested in.
For Expert 0 (no KG knowledge)
Instruction:Based on the user’s watching history, select the best movie from options A-T.
User Query:Watching history:“Movie A”, “Movie B”, . . . , “Movie J”.
Options:“A: Movie”, “B: Movie”, . . . , “T: Movie”.
Select a movie from options A to T that the user is most likely to be interested in.

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
Table 9: Case study on Movie with Expert 2 (TripleRetriever).
Case Study: Movie (TripleRetriever)
(1) Input instance.
User Query (watching history):“Parenthood”, “Dark City”, “Reservoir Dogs”, “Aliens”, “The Sixth Sense”, “The Terminator”, “A Simple Plan”,
“Die Hard”, “Seven”, “The Long Kiss Goodnight”.
Candidate options (A–T):A: Hellbound: Hellraiser II, B: Bring It On, C: Escape from L.A., D: One Tough Cop, E: Boxing Helena, F: Separation,
The, G: From the Journals of Jean Seberg, H: Terminator 2: Judgment Day, I: Following, J: The Big Country, K: One Flew Over the Cuckoo’s Nest,
L: Grandview, U.S.A., M: Cotton Mary, N: Smiling Fish and Goat on Fire, O: School Daze, P: Soldier’s Daughter Never Cries, A, Q: Weekend at
Bernie’s, R: Love and Death on Long Island, S: Roseanna’s Grave, T: Maurice.
Ground-truth: H.
(2) Mixture-of-Experts routing.
Selected expert: Expert 2 (TripleRetriever).
(3) KG retrieval output (structured).
Retrieved triples:
•The Terminatordirector−−−−−→James Cameron
•Terminator 2: Judgment Dayis− →highest-grossing film list
•Indiana Jones and the Temple of Doomstar−−→Kate Capshaw
•. . .
(4) Knowledge Preference Alignment (Agent).
Alignment prompt (abridged):Convert the following factual triples into natural language knowledge suitable for movie recommendations.
Aligned knowledge (output):
Terminator 2: Judgment Day is a movie highly relevant to user queries, as its lead actor is associated with the query and it is related to other
related films such as The Terminator. The Terminator, a movie starring Arnold Schwarzenegger, is a highly relevant film related to the user’s
query . . .
(5) Recommendation (Contrastive Learning-reinforced Rec Agent).
Output: H. Terminator 2: Judgment Day.
Model confidence / option distribution (top):
•H (Terminator 2): 0.92⇐(ground truth)
•J (The Big Country): 0.01
•M (Cotton Mary): 0.01
•N (Smiling Fish and Goat on Fire): 0.01

KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea. Wang et al.
Table 10: Case study on Movie with Expert 3 (SubgraphRetriever).
Case Study: Movie (SubgraphRetriever)
(1) Input instance.
User Query (watching history):“The Lion King”, “The Road to El Dorado”, “Oliver & Company”, “The Incredible Journey”, “Home Alone”,
“Jumanji”, “Flubber”, “Anastasia”, “Labyrinth”, “Tom and Huck”.
Candidate options (A–T):A: Brassed Off, B: Mighty Ducks, The, C: Porky’s II: The Next Day, D: Ninotchka, E: The City, F: The Grass Harp,
G: Like Water for Chocolate, H: Night of the Comet, I: Shaft, J: The Kid, K: Leave It to Beaver, L: Inspector Gadget, M: Fright Night Part II,
N: Chopping Mall, O: Henry: Portrait of a Serial Killer, Part 2, P: Austin Powers: International Man of Mystery, Q: Romancing the Stone, R:
NeverEnding Story II: The Next Chapter, S: Men in Black, T: Re-Animator.
Ground-truth: L.
(2) Mixture-of-Experts routing.
Selected expert: Expert 3 (SubgraphRetriever).
(3) KG retrieval output (structured).
Retrieved subgraph:
Entities:{Flubber (Movie), Jumanji (Movie), Oliver & Company (Movie)}.
Relations:
•Flubbergenre−−−→family comedy
•Flubbertheme−−−−→playful invention / slapstick
•Jumanjigenre−−−→family adventure
•Jumanjitheme−−−−→fantasy adventure / kid-friendly
•Oliver & Companygenre−−−→animated family film
•Oliver & Companytheme−−−−→lighthearted, for all ages
•. . .
(4) Knowledge Preference Alignment (Agent).
Alignment prompt (abridged):Transform the following knowledge graph subgraph into natural language suitable for movie recommen-
dations.
Aligned knowledge (output):
The subgraph contains 3 core entities and a compact set of relations, where Flubber receives the highest relevance score and serves as the central
node. The subgraph connects these entities through recurring family-oriented signals and lightweight entertainment attributes: For Flubber, the
evidence emphasizes a comedy-focused profile with playful, kid-friendly cues; for Jumanji, the relations highlight family adventure elements and
a fantasy-leaning storyline . . .
(5) Recommendation (Contrastive Learning-reinforced Rec Agent).
Output: L Inspector Gadget.
Model confidence / option distribution (top):
•L (Inspector Gadget): 0.55⇐(ground truth)
•B (Mighty Ducks): 0.22
•S (Men in Black): 0.09
•J (The Kid): 0.05

Mixture-of-Experts Knowledge Graph RAG for Multi-Agent LLM-based Recommendation KDD 2026, August 9–13, 2026, Jeju Island, Republic of Korea.
Table 11: Case study on Music with Expert 1 (DirectGenerator).
Case Study: Music (DirectGenerator)
(1) Input instance.
User Query (listening history):“Boards Of Canada”, “Zazen Boys”, “Recloose”, “Flying Lotus”, “The Ananda Project”, “Afterlife”, “Onra &
Quetzal”, “Ian O’Brien”, “Onra & Quetzal”, “Recloose”.
Candidate options (A–T):A: Dumptruck, B: Dj Lol, C: The Livids, D: Hei, E: Tactics, F: Onra & Quetzal, G: Eric Smax & Terri B., H: Dan
Curtin, I: Pebbles, J: Jochen Trappe, K: Versbox, L: Hans Zimmer, M: Melanie Doane, N: Ana D, O: Dave The Hustler, P: Jo O’Meara, Q: Codes In
The Clouds, R: Fatal Bazooka, S: Only Fools And Horses, T: Pell Mell.
Ground-truth: F.
(2) Mixture-of-Experts routing.
Selected expert: Expert 1 (DirectGenerator).
(3) KG retrieval output (structured).
Retrieved knowledge:Skipped(DirectGenerator uses no KG retrieval).
(4) Knowledge Preference Alignment (Agent).
Aligned knowledge (output):Skipped(no retrieved knowledge; ˜K=∅).
(5) Recommendation (Contrastive Learning-reinforced Rec Agent).
Output: F Onra & Quetzal.
Model confidence / option distribution (top):
•F (Onra & Quetzal): 0.76⇐(ground truth)
•L (Hans Zimmer): 0.12
•P (Jo O’Meara): 0.06
•K (Versbox): 0.03