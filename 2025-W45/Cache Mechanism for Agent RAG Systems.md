# Cache Mechanism for Agent RAG Systems

**Authors**: Shuhang Lin, Zhencan Peng, Lingyao Li, Xiao Lin, Xi Zhu, Yongfeng Zhang

**Published**: 2025-11-04 19:02:29

**PDF URL**: [http://arxiv.org/pdf/2511.02919v1](http://arxiv.org/pdf/2511.02919v1)

## Abstract
Recent advances in Large Language Model (LLM)-based agents have been
propelled by Retrieval-Augmented Generation (RAG), which grants the models
access to vast external knowledge bases. Despite RAG's success in improving
agent performance, agent-level cache management, particularly constructing,
maintaining, and updating a compact, relevant corpus dynamically tailored to
each agent's need, remains underexplored. Therefore, we introduce ARC (Agent
RAG Cache Mechanism), a novel, annotation-free caching framework that
dynamically manages small, high-value corpora for each agent. By synthesizing
historical query distribution patterns with the intrinsic geometry of cached
items in the embedding space, ARC automatically maintains a high-relevance
cache. With comprehensive experiments on three retrieval datasets, our
experimental results demonstrate that ARC reduces storage requirements to
0.015% of the original corpus while offering up to 79.8% has-answer rate and
reducing average retrieval latency by 80%. Our results demonstrate that ARC can
drastically enhance efficiency and effectiveness in RAG-powered LLM agents.

## Full Text


<!-- PDF content starts -->

Cache Mechanism for Agent RAG Systems
Shuhang Lin1*Zhencan Peng1*Lingyao Li2Xiao Lin3Xi Zhu1Yongfeng Zhang1
1Rutgers University2University of South Florida3University of Illinois Urbana–Champaign
{shuhang.lin, zhencan.peng, xi.zhu, yongfeng.zhang}@rutgers.edu
lingyaol@usf.edu, xiaol13@illinois.edu
Abstract
Recent advances in Large Language Model
(LLM)-based agents have been propelled
by Retrieval-Augmented Generation (RAG),
which grants the models access to vast exter-
nal knowledge bases. Despite RAG’s success
in improving agent performance, agent-level
cache management, particularly constructing,
maintaining, and updating a compact, relevant
corpus dynamically tailored to each agent’s
need, remains underexplored. Therefore, we
introduceARC (Agent RAG Cache Mecha-
nism), a novel, annotation-free caching frame-
work that dynamically manages small, high-
value corpora for each agent. By synthesizing
historical query distribution patterns with the
intrinsic geometry of cached items in the em-
bedding space, ARC automatically maintains a
high-relevance cache. With comprehensive ex-
periments on three retrieval datasets, our exper-
imental results demonstrate that ARC reduces
storage requirements to 0.015% of the original
corpus while offering up to 79.8% has-answer
rate and reducing average retrieval latency by
80%. Our results demonstrate that ARC can
drastically enhance efficiency and effectiveness
in RAG-powered LLM agents.
1 Introduction
LLM agents have demonstrated significant poten-
tial in various domains, which exhibit remarkable
capabilities in performing complex reasoning tasks
(Wang et al., 2024b; Jin et al., 2024; Hua et al.,
2024), executing knowledge-intensive operations
(Jiang et al., 2024), and implementing customized
automated workflows (Li et al., 2024; Xu et al.,
2025; Mei et al., 2025b). Researchers have em-
ployed multiple techniques to enhance these agents’
ability, including RAG, finetuning, and prompt en-
gineering. Among these approaches, RAG empow-
ers agents with the capacity to access extensive
*Equal contribution.external knowledge repositories, facilitating more
informed response generation and consequently
becoming widely implemented across agent archi-
tectures (Chujie et al., 2024; Huang et al., 2025a;
Guo et al., 2025a).
The enhancement of LLM agent capabilities has
led to significant growth in knowledge reposito-
ries, with extensive-scale knowledge bases now
prevalent in advanced systems. With the accumu-
lation of diverse and multifarious documents, sev-
eral hundred GB or TB-level corpora are common
among enterprise and government entities. These
huge knowledge repositories not only enable LLM
agents to address a more comprehensive range of
inquiries, but also bring the following challenges:
(1) high storage costs, and (2) deployment difficul-
ties on edge and mobile devices.
Firstly, massive knowledge repositories drive
up hardware and maintenance expenses—storing
terabytes of embeddings and documents demands
high-performance disks or distributed storage clus-
ters—and complicate system engineering due to
the need for sophisticated indexing, sharding, and
cache invalidation logic (Mei et al., 2025a). Fur-
thermore, deploying RAG at the edge or on mobile
devices is particularly challenging: such platforms
have limited local storage and CPU/GPU resources,
and often operate under unreliable or high-latency
network conditions. Every off-device retrieval in-
curs tens to hundreds of milliseconds in round-trip
delay, degrading user experience and potentially
causing service unavailability during network out-
ages. These constraints motivate the introduction
of an efficient, high-speed caching layer that stores
only the most semantically central passages. By
dramatically reducing both remote calls and on-
device compute, a well-managed cache cuts latency,
lowers bandwidth consumption, and simplifies the
overall architecture—eliminating the need to bun-
dle or host the full knowledge base on every device.
While the constraints outlined above motivate
1arXiv:2511.02919v1  [cs.CL]  4 Nov 2025

Query Retrieval
Result
Query
CacheRetrieval
Result
Agent
 Agent
Previous Design Proposed Design
Agent N-1
 Agent 1
 Agent 2 Agent N
Cache Cache Cache Cache
Knowledge Repository……
……A.Retrieval -Augmented Agent ：
Traditional vs. CachedB.ARCM SchemaFigure 1: The motivations of our research: (a) The comparison between traditional and cached retrieval-augmented
agent. (b) Our proposed ARCM schema.
efficient caching solutions, recent approaches have
fallen short. Numerous studies have attempted to
introduce caching mechanisms to accelerate LLM
systems, yet the majority of these approaches pri-
marily focus on key-value caches, which incur sig-
nificant storage costs and consequently limit their
applicability in the very resource-constrained en-
vironments they aim to serve. These conventional
caching strategies operate independently of seman-
tic relationships between queries and documents,
thus failing to fully leverage the semantic-aware re-
trieval techniques that make modern RAG systems
effective in the first place.
Our insight brings guidance to cache algorithm
design. We propose theAgent RAG Cache Mech-
anism (ARC), which introduces two innovative
components: (1) a rank-distance based frequency
that weights query relevance by both occurrence
frequency and semantic similarity to top-ranked
results, and (2) a centrality quantification through
hubness, where items with higher neighbor con-
nectivity intuitively contain more general knowl-
edge relevant to the agent’s domain. We com-
bine these two factors to create a priority score for
cached items that governs cache maintenance. On
a large-scale corpus comprising millions of embed-
ding–passage pairs, ARC achieves a79.8% cache
has-answer ratewhile using just0.015%of the
original storage.
In summary, our main contributions are as fol-
lows:
•Problem.To the best of our knowledge,
this work presents the first dedicated caching
mechanism designed for retrieval-augmented
LLM agents, and establishes a formulation of
this problem for analyzing cache efficiencyand performance.
•Methodology.We proposeARC, an agent
RAG caching algorithm that leverages query-
based dynamics and the structural properties
of the item representation space, which can
drastically reduce storage requirements while
preserving retrieval effectiveness.
•Evaluation.We evaluate ARC on three
datasets, with the 6.4-million-document
Wikipedia as a realistic, noisy external cor-
pus. Compared to prior methods, ARC attains
a cache size of0.015%of the full index, a
79.8%cache has-answer rate, and an80%
average reduction in retrieval latency.
2 Related Work
RAG for agent.RAG enables agents to efficiently
access external knowledge. Early studies embed-
ded one-off retrieval calls directly into prompts
for single-step decision-making or multi-hop ques-
tion answering, allowing LLMs to reference ex-
ternal documents during generation and thereby
improving the accuracy of individual responses
or action instructions (Shi et al., 2024; Lee et al.,
2024; Huang et al., 2025b; Guo et al., 2025a).
Subsequently, frameworks such as PlanRAG (Lee
et al., 2024) proposed a phased pipeline, which de-
composes complex tasks into subgoals (Guo et al.,
2025b), retrieves pertinent information for each
subtask, and synthesizes the results in a unified
generation step; this reduces redundant queries and
enhances the traceability and efficiency of multi-
step decision processes (Lee et al., 2024; Huang
et al., 2024). In multi-hop reasoning contexts, the
Generate-then-Ground (Shi et al., 2024) paradigm
was introduced: the model first generates inter-
2

mediate hypotheses, which are then grounded via
document-level verification by the retrieval module,
improving the robustness of chain-of-thought rea-
soning (Shi et al., 2024). More recently, methods
such as RAP (Kagaya et al., 2024) and RAT (Wang
et al., 2024c) have incorporated mechanisms for
contextual memory and dynamic retrieval trigger-
ing, enabling agents in multimodal or long-horizon
interactions to invoke external knowledge adap-
tively based on internal confidence metrics or new
sensory inputs, thus providing flexible, on-demand
information supplementation. Cutting-edge frame-
works like RAG-Gym employ fine-grained process
supervision to jointly optimize retrieval and reason-
ing workflows (Xiong et al., 2025).
Efficient RAG.Efficient Retrieval-Augmented
Generation aims to reduce resource consumption
while improving end-to-end performance. Re-
search in this field can be organized into three ma-
jor areas: (1) Retrieval algorithm optimizations, in-
cluding sparse indexing such as IVF/PQ (Johnson
et al., 2017) and low-dimensional vector represen-
tations with approximate nearest neighbor search
(Quinn et al., 2025); (2) Pipeline optimizations,
such as multi-stage cascaded retrieval, which pro-
gressively refines candidate sets to focus expensive
ranking on high-recall subsets (Bai et al., 2024);
and (3) Hardware-aware compression techniques,
such as 4-bit embedding quantization for RAG
(Jeong, 2025), alongside comprehensive evalua-
tions of compression and dimensionality-reduction
trade-offs (Huerga-Pérez et al., 2025).
Semantic Encoding.The field of text embed-
ding has progressively advanced from static rep-
resentations such as Word2Vec (Mikolov et al.,
2013) to contextualized models like ELMo (Pe-
ters et al., 2018) and BERT (Devlin et al., 2019),
which generate dynamic word vectors conditioned
on surrounding context. Building upon these
frameworks, Siamese-network and contrastive-
learning approaches such as SBERT (Reimers and
Gurevych, 2019) and SimCSE (Gao et al., 2021)
produce high-quality sentence and paragraph em-
beddings, while retrieval-augmented methods like
DPR (Karpukhin et al., 2020) and ColBERT (Khat-
tab and Zaharia, 2020) integrate explicit docu-
ment retrieval with dense vector representations.
More recent work including LM-Steer (Han et al.,
2023) has explored mechanisms for task- or style-
guided embedding control. Concurrently, intrinsic
challenges of high-dimensional spaces have been
widely recognized, including the curse of dimen-sionality (Bellman, 1957), hubness (Radovanovic
et al., 2010), and anisotropy (Ethayarajh, 2019).
These challenges have motivated a range of miti-
gation strategies, including local scaling, isotropy
regularization, and manifold-alignment techniques.
Despite advances in retrieval systems, two crit-
ical gaps remain: the unexploited hierarchical na-
ture of embeddings within topic clusters, and the
neglect of embedding-space geometry in existing
neural-retriever caches. We address these limita-
tions by (1) formalizing embedding stratification
to identify semantically central passages, and (2)
designing a cache mechanism that leverages both
spatial geometry and agent-specific semantics. Our
approach combines hubness-based centrality with
dynamic rank-distance metrics to enable efficient
RAG retrieval in LLM agents.
3 Problem Formulation
In this section, we cast the agent’s RAG pipeline
as a constrained optimization problem, and further
develop a detailed formulation that factorizes the
pipeline into three sequential stages: (1) query gen-
eration, (2) retrieval search, and (3) cache storage.
Query generation.Unlike general-purpose con-
versational LLMs, agents are typically associated
with a predefined domain specialization that speci-
fies their intended application context (Russell and
Norvig, 2016). This specialization induces domain-
specific response preferences. For instance, a med-
ical agent tends to provide precise answers to clini-
cal queries (Ely et al., 2005). Once users become
aware of such preferences, their interactions often
exhibit implicit biases: users are more likely to
ask medical agents health-related questions rather
than queries from unrelated domains (Ebell, 1999),
such as astrophysics. This latent query bias implies
that an agent’s query distribution is shaped by its
operational context. Formally, this relationship can
be expressed as q∼ P(q|Θ) , where qis the user
query, Θrefers to the agent’s domain specializa-
tion, and Pdenotes the probability distribution of
queries conditioned on the agent’s domain special-
ization.
Retrieval Search.In the retrieval phase, a RAG-
enhanced agent leverages the user-provided query
to identify highly relevant documents from an exter-
nal corpus. Given a query qand an external corpus
U, the set of documents retrieved by the agent can
be formally defined as:
Ret(q, U) ={x i:xi∈arg TopKxi∈U(sim(q, x i))},
3

where xidenotes a candidate retrievable item
(e.g., document passage) from the corpus U, and
sim(q, x) is the agent’s internal similarity function
that scores the relevance between the query qand
each itemx.
Cache storage.As the agent engages in frequent
interactions with users, the cache utilization of a
RAG-enhanced system tends to grow continuously
due to the need to retrieve new documents over
time. However, retaining all retrieved content is
impractical, as it would require prohibitively large
cache capacity. In real-world deployments, it is
therefore critical to design RAG agents that oper-
ate efficiently under limited cache budgets. Let
w(x i)denote the cache space occupied by the i-th
retrieved item. The constraint of operating within a
fixed cache budget naturally introduces a capacity
constraint into the optimization problem:
X
xi∈Ctw(x i)≤W max,∀t≥0
where Wmaxis the max capacity of the cache. Let
Ctdenote the set of items stored in the cache at
timet.
Cache Metrics.To quantify query-conditioned
cache efficiency, we define the cumulative miss
count for query qtand the empirical has-answer
rate overTqueries as
Mt=mX
j=1I[xt,j/∈Ct,j−1]
Has-AnswerRate T= 1−1
mTTX
t=1Mt
Here, I(·)serve as the indicator function. xt,jde-
notes the j-th item in query qt, and Ct,j−1 is the
cache state before accessing it. The has-answer rate
reflects the average fraction of items served from
the cache, offering a proxy objective for optimizing
learned caching policies based on retrieval success.
3.1 Agent RAG Cache: From Query to Cache
Letqtdenote the t-th query sampled from a distri-
bution P(q|Θ) . Prior to processing qt, the cache is
in state Ct−1⊆U, constrained by a total capacityP
x∈Ct−1w(x)≤W max, where w(x) represents
the memory footprint of itemx.
The retrieval module returns a ranked list of can-
didates:
Rt= Ret(q t, U) = (x t,1, xt,2, . . . , x t,m),with each xt,jselected from the top- Kcorpus items
under a similarity function:
xt,j∈TopKx∈U(sim(q t, x)).
Upon accessing candidate x=x t,j, the aug-
mented candidate pool becomes D=C t,j−1∪
{x}. The cache state is then updated via
selection: Ct,j= arg max S⊆DP
y∈Sp(y) subject
to,P
y∈Sw(y)≤W max,where p(y) denotes the
utility score guiding cache prioritization. If the can-
didate is redundant or infeasible—i.e., x∈C t,j−1
orw(x)> W max—no update is applied: Ct,j=
Ct,j−1 . After all mcandidates are considered, the
final cache state is set as Ct=C t,m.Sranges over
all subsets of the augmented candidate setD.
3.2 Optimization Objective
At inference step n, our objective is to optimize
the cache policy—specifically, the priority function
p(·) and update rule—so as to maximize the ex-
pected has-answer rate across a future horizon of
Hqueries:
max
pEqn+1:n+H ∼P(q|Θ)
Has-AnswerRate n+1:n+H
= max
pE"
1−1
mHn+HX
t=n+1Mt#
.
This formulation defines our core goal: to design a
cache strategy that maximizes future query perfor-
mance by learning from historical retrieval patterns,
subject to fixed capacity constraints.
4 Agent RAG Cache Mechanism
In this section, we introduce ARC, a cache algo-
rithm designed for RAG-powered agents. ARC
maintains the cache by deriving two complemen-
tary factors into a unifiedpriorityvalue for each
cache item. Our ARC algorithm leverages both
query-based dynamics and the structural proper-
ties of the item representation space: while the
distance–rank frequency score (DRF) quantifies
the dynamic demand for passages demonstrated
through historical queries, the hubness score iden-
tifies passages within the cache that are inherently
more likely to be retrieved due to their inherent
spatial structure. By synthesizing these comple-
mentary signals, we establish a balanced item pri-
oritization mechanism that adapts to agent query
distribution patterns while ensuring cache priorities
remain grounded in the geometric properties of the
passage embedding space.
4

4.1 Distance–Rank Frequency Score
While a significant number of caching algorithms
are designed with a heavy dependence on item ac-
cess frequency, these approaches usually lack con-
sideration of retrieval system characteristics. In
these frequency-based methods, all retrievals of a
passage typically contribute equally to its cache im-
portance score. However, when a passage appears
as a top-ranked result, it generally provides more
value than another passage appearing in the tenth
position for the identical query. In other words,
higher-ranked appearances indicate stronger seman-
tic alignment with queries and should be weighted
more heavily in caching decisions. By implement-
ing differential weighting based on retrieval rank,
our caching mechanism incorporates rank position
as a key signal in our priority formulation, creating
a more retrieval-aware caching strategy.
Rank-based weighting improves cache priori-
tization by emphasizing higher-ranked retrievals,
which captures the relative order of results. How-
ever, we still need to reflect actual semantic close-
ness between queries and passages. For example,
a tenth-rank hit with high cosine distance should
score lower than another tenth-rank result that is
semantically closer. To address this, we also in-
corporate the absolute distance between query and
passage embeddings into our scoring mechanism.
By combining both rank for ordinal importance
and distance for semantic proximity, our priority
formula delivers a more nuanced evaluation of each
passage’s cache value.
Distance–Rank Frequency (DRF) Score.Let Q
denote the set of cached queries and for each query
q∈ Q letRet(q) be the ranked retrieval set (top-
k) returned from the cache or external database.
Define the DRF score of itempas
DRF(p) =X
q∈Q:p∈Ret(q)1
rank(q, p)·dist(q, p)α
where rank(q, p) denotes the 1-based position of p
in the retrieval results for query q,dist(q, p) repre-
sents the embedding distance between qandp, and
α >0 is a tunable parameter controlling distance
sensitivity.
In summary, DRF factor accumulates a passage’s
retrieval history, weighted by both its rank position
and embedding distance for each query. It therefore
directly captures how frequently and meaningfully
the passage responds to the current query distribu-tion, reflecting real-time demand and relevance in
the agent system.
4.2 Hubness Score
In this subsection, we introduce the hubness score,
computed on the nearest neighbor search index of
the cache candidate set. This metric highlights pas-
sages in the dense cores of the embedding space,
specifically those that frequently appear in other
embeddings’ neighbor lists, without relying on any
query history. As a result, it provides a truly query-
agnostic approach for identifying which cache en-
tries are most semantically central and broadly use-
ful.
Space-Aware Approach.Hubness refers to the
statistical phenomenon in high-dimensional spaces
where certain points called "hubs" appear dispro-
portionately often in others’ k-nearest-neighbor
lists (Tomasev et al., 2013). Prior studies have
examined hubness from algorithmic and statisti-
cal perspectives but have paid little attention to its
semantic implications.
Existing works (Bogolin et al., 2022; Wang
et al., 2023) rrevealthat passages with high hub-
ness scores consistently exhibit higher retrieval
rates across diverse query sets, reflecting a space-
invariant measure of semantic centrality. This in-
sight motivates the first component of our cache
mechanism. However, while hubness effectively
captures the global importance of items, our for-
mulation must also consider local retrieval dynam-
ics. We therefore introduce both rank and abso-
lute distance components into our prioritization for-
mula. The rank component addresses the varying
importance of different positions in retrieval results,
while absolute distance proves crucial in practice;
for instance, when query-document similarity is
exceptionally low, such results should receive cor-
respondingly reduced weighting regardless of rank.
Furthermore, due to sharing the same mathe-
matical formulation in top-K selection, hubness
serves as a natural bridge between retrieval behav-
ior and embedding space characteristics, enabling
us to simulate retrieval dynamics without making
assumptions about the query distribution.
Hubness Score.Hubness score of xcounts
its occurrences in other points’ k-nearest-neighbor
lists. Which in math can be written as
hk(xi) =nX
j=1
j̸=iI(xi∈ N k(xj))
5

Nk(xj)denotes the k-nearest neighbors of xj,
I(·)serve as the indicator function. In the ARC
algorithm, we compute hubness over the cache’s
own candidate set.
Combined Priority.Our priority calculation in-
tegrates two key metrics: DRF and the hubness
score introduced in Section 3. With an additional
penalty term for memory utilization. Using w(p) to
denote the memory footprint of itemp, we define:
Priority(p) =1
log 
w(p) + 1h
βlog 
hk(p) + 1
+ (1−β) DRF(p)i
.
where β∈[0,1] balances centrality versus query
frequency. The log(w(p) + 1) term in the denom-
inator penalizes items with large memory foot-
prints and encourages the cache to store more items.
Items with lower Priority(p) are deemed less valu-
able and evicted first to make room for new inser-
tions.
Our comprehensive formula accounts for global
importance via hubness, relative positional relation-
ships via rank, and absolute semantic proximity via
distance. By integrating these multiple dimensions
of retrieval relevance, our approach offers superior
performance compared to frequency-based formu-
lations alone. This provides a more nuanced mecha-
nism for determining cache priority that aligns with
the semantic characteristics of embedding-based
retrieval systems and remains robust across varying
query distributions.
4.3 Cache Maintenance
Algorithm 1 shows the maintenance procedure of
ARC managing a cache Cwith maximum capac-
ityWmaxand escalating the query to D. When a
new query qarrives, the algorithm first retrieves
the top- kcandidates Ret(q)←topK(q, C) from
the cache. The relevance of these results is evalu-
ated by computing the mean embedding distance
avgp∈Ret(q) (dist(q, p)) across all retrieved items.
Only if the average distance exceeds τ, indicating
low relevance, ARC escalates retrieval to the full
external corpus Dand updates the retrieved items
Ret(q)←topK(q,D).
Subsequently, ARC updates the distance-rank
metrics of each item p∈Ret(q) . To sim-
plify the notation, we defined DR(p, q) =
1/rank(q, p)·dist(q, p)α.Existing cache entries
accumulate their DRF score via DRF(p) +=Algorithm 1Agent RAG Cache Maintenance and
Escalation
Input: Query vector q, External corpus D, Cache
C(with capacity Wmax), Top- kretrieval size k,
Drift toleranceδ
Output: Top-K ItemsRet(q)
1:Ret(q)←Ret(q, C)▷ Return the topK result
from the cache
2:ifavgp∈Ret(q) (dist(q, p))> τthen
3:Ret(q)←Ret(q, U)▷Return the topK
result from the external corpus U
4:end if
5:foreachp∈Ret(q)do▷Update the DSF
Score of inserted/existed items in cache
6:ifp∈Cthen
7:DRF(p) +=DR(p, q)
8:else
9:DRF(p) =DR(p, q)
10:C←C∪ {p}
11:end if
12:end for
13:whileP
x∈Cw(x)> W maxdo▷Evict if
cache exceeds capacity
14:C←C\
arg min x∈CPriority(x)	
15:end while
16:returnRet(q)
DR(p, q), while new items initialize DRF(p) =
DR(p, q) before being inserted into C. Fi-
nally, cache Citeratively evicts the lowest-priority
itemarg min x∈CPriority(x) until the total cache
weight satisfiesP
x∈Cw(x)≤W max.
5 Empirical Evaluation
Datasets: Large-scare Retrieval Corpus.To sim-
ulate a real-world, large-scale RAG deployment,
we build our primary index from the English subset
of the 2023 Wikipedia dump1which has over 6.4
million documents. We segment each document in
passage level, where each chunk have one or more
passages with no more than 2048 characters. The
total chunk amount is over 14 milion. This full-
scale document repository introduces substantial
distractor noise—forcing the retriever to discrimi-
nate finely between relevant and irrelevant content.
We also inject into this index all passages contain-
ing the ground-truth answers for our evaluation sets
to evaluate performance.
1https://huggingface.co/datasets/wikimedia/
wikipedia
6

(a) Has-answer Rate at Varying Cache Capacity
 (b) Cache Has-answer Rate Across Query Stream
Figure 2: Cache performance analysis: (a) Effects of varying cache capacity; (b) Continuous improvement of
has-answer rate with streaming queries.
Datasets: Evaluation Datasets.We assess
ARC on three diverse question-answering datasets
including SQuAD, MMLU and Adversarial QA.
We evaluate our method on three
datasets:SQuAD(Rajpurkar et al., 2016),
MMLU(Hendrycks et al., 2020), andAdversar-
ialQA(Jia and Liang, 2017). SQuAD contains
536 short Wikipedia articles yielding 107,785
QA pairs, where answers are contiguous spans
within each context. MMLU spans 57 academic
and professional subjects, with 15,908 test and
1,540 calibration questions. AdversarialQA is built
on the SQuAD v1.1 dev set (10,570 examples)
and introduces perturbations via AddSent and
AddOneSent distractors.
Baselines.We compare our cache method with
several standard policies, including LFU, FIFO,
Proximity, and GPTCache. LFU and FIFO are the
most classic baselines in cache-related methods:
LFU evicts the least frequently used items, while
FIFO removes the oldest entries in the cache.Prox-
imity(Bergman et al., 2025) maintains historical
query–document pairs and returns previously re-
trieved passages from the most semantically similar
past query when the similarity exceeds its thresh-
oldτ, evicting the oldest query–document pair.
GPTCache(Bang et al., 2023) reuses results from
semantically similar queries as an eviction mecha-
nism; however, its eviction strategy relies solely on
embedding similarity, neglecting valuable retrieval-
specific signals such as rank-weighted frequency or
embedding-space centrality that could better cap-
ture the demand intensity and semantic centrality of
cached items. Optimal parameter configurations forboth Proximity and GPTCache were determined as
τ= 0.2 through systematic grid search. In ARC,
we set α= 0.4 , and β= 0.7 ,0.15, and 0.2for
the SQuAD, MMLU, and AdversarialQA datasets,
respectively. Unless stated otherwise, all experi-
ments use a cache capacity of 3.0 MB and a top-K
retrieval size ofK= 50.
Models.For the embedding model, given that
we need to index all wiki content, utilizing closed-
source models would impose significant compu-
tational costs. Therefore, following the compre-
hensive RAG evaluation(Wang et al., 2024a), we
adopted the open-source embedding models bge-
small-en(Liu et al., 2023a) and llm-embedder(Liu
et al., 2023b) for our indexing architecture, result-
ing in an embedding memory footprint of approx-
imately 20 GB and 40 GB, respectively. We em-
ployed FAISS IndexFlatIP (Douze et al., 2024) for
our vector similarity search.
Metrics.To evaluate the effectiveness of our
cache system, we measure performance from two
critical aspects: efficiency and latency reduction.
We employ Has-Answer Rate, which measures the
proportion of queries that can be successfully an-
swered using cached content without accessing the
full index, where a higher has-answer rate indicates
better cache utilization and retrieval efficiency. Ad-
ditionally, we use Average Memory Access Time
(AMAT) (Hennessy and Patterson, 2011), which
quantifies the average time required to retrieve
information, accounting for both cache hits and
misses. This metric effectively captures the overall
latency reduction achieved by our caching mecha-
nism compared to the baseline retrieval system.
7

Table 1: Has-Answer Rate↑(%)
Methodbge-small-en llm-embedder
MMLU AdversarialQA SQuAD MMLU AdversarialQA SQuAD
LFU 53.26 66.11 69.46 42.09 57.57 59.37
FIFO 41.84 66.64 77.04 34.51 58.67 71.49
GPTCache 43.65 46.49 41.23 40.08 41.54 40.13
Proximity 46.41 54.25 59.79 38.41 50.73 58.84
ARC (w/o hubness) 62.37 68.14 76.09 51.92 58.48 69.55
ARC62.63 71.18 79.80 52.46 62.78 71.94
Table 2: AMAT↓(s)
Methodbge-small-en llm-embedder
MMLU AdversarialQA SQuAD MMLU AdversarialQA SQuAD
LFU 0.668 0.427 0.388 1.754 1.219 1.175
FIFO 0.814 0.429 0.298 1.969 1.182 0.821
GPTCache 0.788 0.704 0.773 1.810 1.680 1.720
Proximity 0.758 0.604 0.539 1.840 1.406 1.198
ARC (w/o hubness) 0.559 0.411 0.307 1.489 1.196 0.878
ARC0.556 0.377 0.269 1.468 1.046 0.818
Baselines Comparison.To evaluate the perfor-
mance of ARC, we conduct experiments comparing
it against baselines including LFU, FIFO, Prox-
imity, GPTCache, and ARC without the hubness
component, across multiple retrieval benchmarks
using two embedding models: bge-small-en and
llm-embedder. We report the has-answer rate and
AMAT in Table 1 and Table 2.Best results are
bolded; second-best are underlined .
As shown in Table 2, it is observed that GPT-
Cache and Proximity perform worst among all
methods. This is because their effectiveness relies
on the assumption that incoming queries are highly
similar to previously seen ones, which only holds
under extensive historical data and large cache. In
contrast, ARC highest has-answer rates and the
lowest AMAT across all benchmarks. For example,
under the bge-small-en embedding, even without
the hubness score, ARC achieves a has-answer rate
of over 62% and an AMAT of 0.559s on MMLU,
while LFU only achieves 53% and 0.668s. With the
hubness score, ARC further improves by over 3%
has-answer rate with hubness on Squad. Similarly,
ARC outperforms all baselines in the llm-embedder
setup, attaining ranging from 52.46% to 71.94%.
The advantage of ARC comes from two combined
score: DRF, which helps the cache identify valu-
able items based on their ranking position and se-
mantic proximity to queries, and hubness score
which naturally favors passages that hold central or
influential positions within the embedding space.It is remarked that without caching, retrieving the
entries with the highest semantic similarity on the
Wikipedia index takes 1.313s per query on SQuAD,
while ARC only needs 0.269s, reducing the query
time by almost 80%.
Cache Capacity Ablation.Figure 2a illustrates
cache has-answer rates varying cache capacity from
1MB to 5MB in Dataset MMLU with embedder
bge-small-en. ARC consistently maintains the
highest has-answer rates across various cache sizes,
which significantly outperform baseline methods,
particularly at smaller cache capacities. For exam-
ple, ARC achieves 47.64% has-answer rate at 1
MB of cache capacity on MMLU, surpassing LFU
by 12.8%.
Streaming Query.Figure 2b shows the stream-
ing cache has-answer rates over 80 sequential
queries on MMLU. ARC quickly adapts to the in-
coming stream and consistently maintains the high-
est cumulative has-answer rate among all methods.
While all baselines experience an initial warm-up
phase, the has-answer rate of ARC continues to
improve and stabilizes above 50% after around 60
queries, significantly outperforming all alternatives
such as LFU and Proximity.
6 Conclusion
We introduce ARC, an annotation-free cache con-
struction mechanism using stratified hub selection
and adaptive escalation. On a 30 million-pair cor-
pus, ARC achieves a 79.8% has-answer rate while
8

caching only 0.015% of the data, outperforming
baseline methods. By exploiting the hierarchical
structure of embeddings, ARC provides a princi-
pled framework for data-efficient retrieval in band-
width and latency-constrained RAG systems, open-
ing avenues for research on geometric properties
of representation spaces.
7 Limitations
We focus our evaluation on the standard single-turn
QA setting, which allows controlled measurement
of caching behavior. While multi-turn dialogue is
outside our current scope, ARC is compatible with
session-level extensions; exploring dialogue-aware
scoring and evaluation on conversational bench-
marks is a natural next step. We have not yet per-
formed experiments in transfer or cross-domain
settings. Extending ARC to evaluate transfer ro-
bustness will be a direction for future work.
References
Yu Bai, Yukai Miao, Li Chen, Dan Li, Yanyu Ren,
Hongtao Xie, Ce Yang, and Xuhui Cai. 2024.
Pistis-rag: A scalable cascading framework towards
trustworthy retrieval-augmented generation.CoRR,
abs/2407.00072.
Fu Bang, Liling Tan, Dmitrijs Milajevs, Geeticka
Chauhan, Jeremy Gwinnup, and Elijah Rippeth. 2023.
Gptcache: An open-source semantic cache for llm
applications enabling faster answers and cost savings.
InProceedings of the 3rd Workshop for Natural Lan-
guage Processing Open Source Software (NLP-OSS),
pages 212–218.
Richard Bellman. 1957.Dynamic Programming.
Princeton University Press.
Shai Aviram Bergman, Zhang Ji, Anne-Marie Kermar-
rec, Diana Petrescu, Rafael Pires, Mathis Randl,
and Martijn de V os. 2025. Leveraging approximate
caching for faster retrieval-augmented generation. In
Proceedings of the 5th Workshop on Machine Learn-
ing and Systems, pages 66–73.
Simion-Vlad Bogolin, Ioana Croitoru, Hailin Jin, Yang
Liu, and Samuel Albanie. 2022. Cross modal re-
trieval with querybank normalisation. InProceedings
of the IEEE/CVF Conference on Computer Vision
and Pattern Recognition, pages 5194–5205.
Gao Chujie, Siyuan Wu, Yue Huang, Dongping Chen,
Qihui Zhang, Zhengyan Fu, Yao Wan, Lichao Sun,
and Xiangliang Zhang. 2024. Honestllm: Toward an
honest and helpful large language model.Advances
in Neural Information Processing Systems, 37:7213–
7255.Jacob Devlin, Ming-Wei Chang, Kenton Lee, and
Kristina Toutanova. 2019. Bert: Pre-training of deep
bidirectional transformers for language understand-
ing. InProceedings of the 2019 Conference of the
North American Chapter of the Association for Com-
putational Linguistics: Human Language Technolo-
gies, volume 1, pages 4171–4186.
Matthijs Douze, Alexandr Guzhva, Chengqi Deng,
Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel
Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé
Jégou. 2024. The faiss library.arXiv preprint
arXiv:2401.08281.
Mark Ebell. 1999. Information at the point of care:
answering clinical questions.The Journal of the
American Board of Family Practice, 12(3):225–235.
John W Ely, Jerome A Osheroff, M Lee Chambliss,
Mark H Ebell, and Marcy E Rosenbaum. 2005. An-
swering physicians’ clinical questions: obstacles and
potential solutions.Journal of the American Medical
Informatics Association, 12(2):217–224.
Karthik Ethayarajh. 2019. Towards understanding
the anisotropic geometry of bert.arXiv preprint
arXiv:1908.08576.
Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021.
Simcse: Simple contrastive learning of sentence em-
beddings.arXiv preprint arXiv:2104.08821.
Minghao Guo, Qingcheng Zeng, Xujiang Zhao, Yanchi
Liu, Wenchao Yu, Mengnan Du, Haifeng Chen,
and Wei Cheng. 2025a. Deepsieve: Information
sieving via llm-as-a-knowledge-router.Preprint,
arXiv:2507.22050.
Minghao Guo, Xi Zhu, Jingyuan Huang, Kai Mei,
and Yongfeng Zhang. 2025b. Reagan: Node-as-
agent-reasoning graph agentic network.Preprint,
arXiv:2508.00429.
Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai
Sun, Nan Jiang, Tarek Abdelzaher, and Heng Ji. 2023.
Word embeddings are steers for language models.
arXiv preprint arXiv:2305.12798.
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou,
Mantas Mazeika, Dawn Song, and Jacob Steinhardt.
2020. Measuring massive multitask language under-
standing.arXiv preprint arXiv:2009.03300.
John L Hennessy and David A Patterson. 2011.Com-
puter architecture: a quantitative approach. Elsevier.
Wenyue Hua, Kaijie Zhu, Lingyao Li, Lizhou Fan,
Shuhang Lin, Mingyu Jin, Haochen Xue, Zelong
Li, JinDong Wang, and Yongfeng Zhang. 2024. Dis-
entangling logic: The role of context in large lan-
guage model reasoning capabilities.arXiv preprint
arXiv:2406.02787.
Yue Huang, Chujie Gao, Siyuan Wu, Haoran Wang,
Xiangqi Wang, Yujun Zhou, Yanbo Wang, Jiayi Ye,
Jiawen Shi, Qihui Zhang, and 1 others. 2025a. On
9

the trustworthiness of generative foundation mod-
els: Guideline, assessment, and perspective.arXiv
preprint arXiv:2502.14296.
Yue Huang, Yanbo Wang, Zixiang Xu, Chujie Gao,
Siyuan Wu, Jiayi Ye, Xiuying Chen, Pin-Yu Chen,
and Xiangliang Zhang. 2025b. Breaking focus: Con-
textual distraction curse in large language models.
arXiv preprint arXiv:2502.01609.
Yue Huang, Siyuan Wu, Chujie Gao, Dongping Chen,
Qihui Zhang, Yao Wan, Tianyi Zhou, Chaowei Xiao,
Jianfeng Gao, Lichao Sun, and 1 others. 2024. Data-
gen: Unified synthetic dataset generation via large
language models. InThe Thirteenth International
Conference on Learning Representations.
Naamán Huerga-Pérez, Rubén Álvarez, Rubén Ferrero-
Guillén, Alberto Martínez-Gutiérrez, and Javier
Díez-González. 2025. Optimization of embed-
dings storage for rag systems using quantization
and dimensionality reduction techniques.CoRR,
abs/2505.00105.
Taehee Jeong. 2025. 4bit-quantization in vector-
embedding for rag.CoRR, abs/2501.10534.
Robin Jia and Percy Liang. 2017. Adversarial exam-
ples for evaluating reading comprehension systems.
InProceedings of the 2017 Conference on Empiri-
cal Methods in Natural Language Processing, pages
2021–2031.
Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song,
Chen Zhu, Hengshu Zhu, and Ji-Rong Wen. 2024.
Kg-agent: An efficient autonomous agent framework
for complex reasoning over knowledge graph.arXiv
preprint arXiv:2402.11163.
Mingyu Jin, Qinkai Yu, Jingyuan Huang, Qingcheng
Zeng, Zhenting Wang, Wenyue Hua, Haiyan Zhao,
Kai Mei, Yanda Meng, Kaize Ding, and 1 others.
2024. Exploring concept depth: How large language
models acquire knowledge and concept at different
layers?arXiv preprint arXiv:2404.07066.
Jeff Johnson, Matthijs Douze, and Hérvé Jégou. 2017.
Billion-scale similarity search with gpus.CoRR,
abs/1702.08734.
Tomoyuki Kagaya, Thong Jing Yuan, Yuxuan Lou,
Jayashree Karlekar, Sugiri Pranata, Akira Kinose,
Koki Oguri, Felix Wick, and Yang You. 2024.
Rap: Retrieval-augmented planning with contex-
tual memory for multimodal llm agents.Preprint,
arXiv:2402.03610.
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick
Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and
Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the
2020 Conference on Empirical Methods in Natural
Language Processing, pages 6769–6781.Omar Khattab and Matei Zaharia. 2020. Colbert: Effi-
cient and effective passage search via contextualized
late interaction over bert. InProceedings of the 43rd
International ACM SIGIR Conference on Research
and Development in Information Retrieval, pages
39–48.
Myeonghwa Lee, Seonho An, and Min-Soo Kim. 2024.
Planrag: A plan-then-retrieval augmented genera-
tion for generative large language models as decision
makers.Preprint, arXiv:2406.12430.
Zelong Li, Shuyuan Xu, Kai Mei, Wenyue Hua, Bal-
aji Rama, Om Raheja, Hao Wang, He Zhu, and
Yongfeng Zhang. 2024. Autoflow: Automated work-
flow generation for large language model agents.
arXiv preprint arXiv:2407.12821.
Zheng Liu, Shitao Xiao, and 1 others. 2023a. Bge: A
general embedding model for information retrieval.
Preprint, arXiv:2309.07597.
Zheng Liu, Shitao Xiao, and 1 others. 2023b. Flagem-
bedding: Unified embedding for retrieval-augmented
generation.Preprint, arXiv:2310.07554.
Kai Mei, Wujiang Xu, Shuhang Lin, and Yongfeng
Zhang. Omnirouter: Budget and performance con-
trollable multi-llm routing, 2025.URL https://arxiv.
org/abs/2502.20576.
Kai Mei, Wujiang Xu, Shuhang Lin, and Yongfeng
Zhang. 2025a. Smart routing: Cost-effective multi-
llm serving for multi-core aios.arXiv preprint
arXiv:2502.20576.
Kai Mei, Xi Zhu, Hang Gao, Shuhang Lin, and
Yongfeng Zhang. 2025b. Litecua: Computer as mcp
server for computer-use agent on aios.arXiv preprint
arXiv:2505.18829.
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Cor-
rado, and Jeffrey Dean. 2013. Distributed representa-
tions of words and phrases and their compositionality.
InAdvances in Neural Information Processing Sys-
tems, volume 26, pages 3111–3119.
Matthew E Peters, Mark Neumann, Mohit Iyyer, Matt
Gardner, Christopher Clark, Kenton Lee, and Luke
Zettlemoyer. 2018. Deep contextualized word repre-
sentations. InProceedings of the 2018 Conference
of the North American Chapter of the Association
for Computational Linguistics: Human Language
Technologies, volume 1, pages 2227–2237.
Derrick Quinn, Mohammad Nouri, Neel Patel, John
Salihu, Alireza Salemi, Sukhan Lee, Hamed Zamani,
and Mohammad Alian. 2025. Accelerating retrieval-
augmented generation. InProceedings of the 30th
ACM International Conference on Architectural Sup-
port for Programming Languages and Operating Sys-
tems (ASPLOS ’25), pages 15–32, Rotterdam, The
Netherlands.
10

Milos Radovanovic, Alexandros Nanopoulos, and Mir-
jana Ivanovic. 2010. Hubs in space: Popular nearest
neighbors in high-dimensional data.Journal of Ma-
chine Learning Research, 11(sept):2487–2531.
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and
Percy Liang. 2016. Squad: 100,000+ questions for
machine comprehension of text. InProceedings of
the 2016 Conference on Empirical Methods in Natu-
ral Language Processing, pages 2383–2392.
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert:
Sentence embeddings using siamese bert-networks.
InProceedings of the 2019 Conference on Empiri-
cal Methods in Natural Language Processing, pages
3973–3983.
Stuart J Russell and Peter Norvig. 2016.Artificial intel-
ligence: a modern approach. pearson.
Zhengliang Shi, Shuo Zhang, Weiwei Sun, Shen Gao,
Pengjie Ren, Zhumin Chen, and Zhaochun Ren. 2024.
Generate-then-ground in retrieval-augmented genera-
tion for multi-hop question answering. InProceed-
ings of the 62nd Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Pa-
pers), pages 7339–7353, Bangkok, Thailand. Associ-
ation for Computational Linguistics.
Nenad Tomasev, Milos Radovanovic, Dunja Mladenic,
and Mirjana Ivanovic. 2013. The role of hubness in
clustering high-dimensional data.IEEE transactions
on knowledge and data engineering, 26(3):739–751.
Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran
Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi,
Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng
Yin, Changze Lv, Xiaoqing Zheng, and Xuanjing
Huang. 2024a. Searching for best practices in
retrieval-augmented generation.
Yimu Wang, Xiangru Jian, and Bo Xue. 2023. Bal-
ance act: Mitigating hubness in cross-modal re-
trieval with query and gallery banks.arXiv preprint
arXiv:2310.11612.
Yulong Wang, Tianhao Shen, Lifeng Liu, and Jian Xie.
2024b. Sibyl: Simple yet effective agent framework
for complex real-world reasoning.arXiv preprint
arXiv:2407.10718.
Zihao Wang, Anji Liu, Haowei Lin, Jiaqi Li, Xi-
aojian Ma, and Yitao Liang. 2024c. Rat: Re-
trieval augmented thoughts elicit context-aware
reasoning in long-horizon generation.Preprint,
arXiv:2403.05313.
Guangzhi Xiong, Qiao Jin, Xiao Wang, Yin Fang,
Haolin Liu, Yifan Yang, Fangyuan Chen, Zhixing
Song, Dengyu Wang, Minjia Zhang, Zhiyong Lu,
and Aidong Zhang. 2025. Rag-gym: Optimizing rea-
soning and search agents with process supervision.
Preprint, arXiv:2502.13957.Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zu-
jie Liang, and Yongfeng Zhang. 2025. A-mem:
Agentic memory for llm agents.arXiv preprint
arXiv:2502.12110.
11