# When should I search more: Adaptive Complex Query Optimization with Reinforcement Learning

**Authors**: Wei Wen, Sihang Deng, Tianjun Wei, Keyu Chen, Ruizhi Qiao, Xing Sun

**Published**: 2026-01-29 03:16:53

**PDF URL**: [https://arxiv.org/pdf/2601.21208v1](https://arxiv.org/pdf/2601.21208v1)

## Abstract
Query optimization is a crucial component for the efficacy of Retrieval-Augmented Generation (RAG) systems. While reinforcement learning (RL)-based agentic and reasoning methods have recently emerged as a promising direction on query optimization, most existing approaches focus on the expansion and abstraction of a single query. However, complex user queries are prevalent in real-world scenarios, often requiring multiple parallel and sequential search strategies to handle disambiguation and decomposition. Directly applying RL to these complex cases introduces significant hurdles. Determining the optimal number of sub-queries and effectively re-ranking and merging retrieved documents vastly expands the search space and complicates reward design, frequently leading to training instability. To address these challenges, we propose a novel RL framework called Adaptive Complex Query Optimization (ACQO). Our framework is designed to adaptively determine when and how to expand the search process. It features two core components: an Adaptive Query Reformulation (AQR) module that dynamically decides when to decompose a query into multiple sub-queries, and a Rank-Score Fusion (RSF) module that ensures robust result aggregation and provides stable reward signals for the learning agent. To mitigate training instabilities, we adopt a Curriculum Reinforcement Learning (CRL) approach, which stabilizes the training process by progressively introducing more challenging queries through a two-stage strategy. Our comprehensive experiments demonstrate that ACQO achieves state-of-the-art performance on three complex query benchmarks, significantly outperforming established baselines. The framework also showcases improved computational efficiency and broad compatibility with different retrieval architectures, establishing it as a powerful and generalizable solution for next-generation RAG systems.

## Full Text


<!-- PDF content starts -->

When should I search more: Adaptive Complex Query Optimization
with Reinforcement Learning
Wei Wen*1, Sihang Deng*2Tianjun Wei*†3, Keyu Chen1, Ruizhi Qiao†1, Xing Sun1
1Tencent Youtu Lab,2The University of Hong Kong,3Nanyang Technological University
{jawnrwen,yolochen,ruizhiqiao,winfredsun}@tencent.com
tjwei2-c@my.cityu.edu.hk
Abstract
Query optimization is a crucial component for
the efficacy of Retrieval-Augmented Gener-
ation (RAG) systems. While reinforcement
learning (RL)-based agentic and reasoning
methods have recently emerged as a promising
direction on query optimization, most existing
approaches focus on the expansion and abstrac-
tion of a single query. However, complex user
queries are prevalent in real-world scenarios,
often requiring multiple parallel and sequen-
tial search strategies to handle disambiguation
and decomposition. Directly applying RL to
these complex cases introduces significant hur-
dles. Determining the optimal number of sub-
queries and effectively re-ranking and merg-
ing retrieved documents vastly expands the
search space and complicates reward design,
frequently leading to training instability. To
address these challenges, we propose a novel
RL framework called Adaptive Complex Query
Optimization (ACQO). Our framework is de-
signed to adaptively determine when and how
to expand the search process. It features two
core components: an Adaptive Query Refor-
mulation (AQR) module that dynamically de-
cides when to decompose a query into multiple
sub-queries, and a Rank-Score Fusion (RSF)
module that ensures robust result aggregation
and provides stable reward signals for the learn-
ing agent. To mitigate training instabilities,
we adopt a Curriculum Reinforcement Learn-
ing (CRL) approach, which stabilizes the train-
ing process by progressively introducing more
challenging queries through a two-stage strat-
egy. Our comprehensive experiments demon-
strate that ACQO achieves state-of-the-art per-
formance on three complex query benchmarks,
significantly outperforming established base-
lines. The framework also showcases improved
computational efficiency and broad compati-
bility with different retrieval architectures, es-
tablishing it as a powerful and generalizable
solution for next-generation RAG systems.
*Equal Contribution.1 Introduction
Retrieval-Augmented Generation (RAG) has be-
come a core paradigm in the LLM era be-
cause it grounds generation in external evidence,
thereby improving factuality, recency, and attribu-
tion (Huang and Huang, 2024; Lewis et al., 2020).
Achieving these benefits in RAG hinges on obtain-
ing high-quality retrieved evidence, which in turn
depends on transforming a user’s natural-language
question into a self-contained, retrieval-friendly
query. This step is known asQuery Optimiza-
tion (QO)(Yu et al., 2020; Vakulenko et al., 2021;
Zhang et al., 2024).
Existing QO techniques primarily optimize a
single query through expansion or abstraction (Yu
et al., 2020; Vakulenko et al., 2021; Zhang et al.,
2024) in different approaches. Prompt-based ap-
proaches (Azad and Deepak, 2019) leverage metic-
ulously crafted instructions to guide the LLM in
generating more effective search queries. For in-
stance, a simple prompt might instruct the LLM to
“rephrase the user’s question to be more suitable
for a search engine.”. Interactive-learning based
methods (Xu et al., 2024; Zhu et al., 2025; Feng
et al., 2023) go a step further by engaging in a feed-
back loop with the user or a simulated environment,
allowing the model to refine its queries iteratively
based on the quality of retrieved results. Pseudo-
document generation techniques (Wang et al., 2023;
Gao et al., 2023) transform the original query into a
hypothetical, longer document that contains richer
context, which can then be used to retrieve more rel-
evant information from the knowledge base. More
recently, agentic and reasoning-augmented rein-
forcement learning (RL) methods—valued for their
reduced dependence on labeled supervision—have
shown strong empirical gains (Singh et al., 2025;
Zhu et al., 2025). However, most of these solutions
implicitly assume a one-to-one correspondence be-
tween a user query and an optimized query, whicharXiv:2601.21208v1  [cs.AI]  29 Jan 2026

limits their coverage of complex information needs.
In real-world RAG applications, complex
queries are common and often require multiple
parallel or sequential sub-queries, notably for dis-
ambiguation and decomposition (Song and Zheng,
2024).
•Disambiguation queries, such as a user ask-
ing, “When did Arsenal last win the FA Cup?
[SEP] 2005 [SEP] What about them compared
to Chelsea in league titles?", require the system
to interpret multi-turn contexts and clarify entity
references (e.g., linking "them" back to Arsenal
while introducing Chelsea for comparison). This
may necessitate generating multiple parallel or
sequential sub-queries to retrieve and contrast
evidence.
•Decomposition queries, such as a user asking,
“What were the global shipments of iPhones in
2022 and 2023, respectively?", require breaking
down a multi-objective problem into independent
sub-queries (e.g., "global iPhone shipments in
2022" and "global iPhone shipments in 2023"),
retrieving results for each, and then synthesizing
a final answer.
While some prior work has explored these prob-
lems (Ammann et al., 2025; Perez et al., 2020;
Liu et al., 2024), applying reinforcement learn-
ing to such complex scenarios still presents a se-
ries of challenges: (1) deciding query number and
depth (when to stop, whether to branch, how to
merge); (2) performing multi-path retrieval and
document aggregation across heterogeneous re-
trievers (sparse, dense, hybrid) with consistent, ro-
bust signals; and (3) coping with expanded search
spaces and sparse/delayed rewards, which destabi-
lize training. We argue that an effective QO system
for complex queries should satisfy two goals:
•Adaptive query handling: it should adaptively
decide the number and depth of sub-queries and
switch among disambiguation, decomposition
and single-query expansion and abstraction.
•Stability and integrability: it should support
an end-to-end pipeline (query reformulation
→multi-retrieval →document re-ranking →
answer generation), seamlessly integrate with
sparse and dense retrieval backends, and incor-
porate stabilizing training mechanisms tailored
to RL.
To meet these goals, in this paper we propose
Adaptive Complex Query Optimization (ACQO),an RL framework that learns when and how to
expand the search process and how to accumu-
late evidence robustly. First, we let LLM decide
whether to trigger decomposition or disambigua-
tion, producing a set of parallel or staged sub-
queries based on query complexity and intent diver-
sity. Then, we perform model-agnostic re-ranking
and fusion by jointly exploiting rank positions and
retrieval scores, enabling smooth integration with
heterogeneous retrievers and providing stable in-
termediate signals for the RL agent. Finally, we
introduce a Curriculum Reinforcement Learning
(CRL) strategy with two stages: an initial phase
for broad exploration over all samples to estab-
lish general policies, followed by a focused phase
that emphasizes challenging cases. This curricu-
lum mitigates reward sparsity and improves con-
vergence stability across the spectrum of query
complexities. In experiments, ACQO achieves
state-of-the-art performance on widely used RAG
benchmarks, including conversational query re-
formulation (TopiOCQA) (Adlakha et al., 2022)
and multi-hop reasoning (HotpotQA) (Yang et al.,
2018), with additional out-of-domain evaluation
on MultiHop-RAG (Tang and Yang, 2024) demon-
strating strong generalization capabilities. Notably,
our lightweight components achieve performance
comparable to approaches requiring specialized
retrieval modifications or complex re-ranking ar-
chitectures, while maintaining significantly lower
computational overhead. Experimental results
demonstrate substantial improvements over base-
line methods in both quantitative metrics and quali-
tative analysis. The contributions of this work are
as follows:
•We propose ACQO, which unifies adaptive
multi-query decision-making with robust evi-
dence fusion in an end-to-end RL framework
for complex queries.
•We introduce a universal re-ranking mechanism
to combine rank positions and retrieval scores
in a model-agnostic manner, improving stability
and transferability across heterogeneous retriev-
ers.
•Through extensive experiments on benchmark
datasets, we demonstrate that ACQO signifi-
cantly outperforms existing methods while main-
taining computational efficiency, establishing
its superiority for complex query processing in
RAG systems.

2 What makes queries complex in
real-world RAG scenarios?
In this section, we conduct a systematic analysis
of query complexity patterns in real-world RAG
benchmark. By examining the inherent characteris-
tics of queries across different datasets, we identify
the key challenges that motivate our ACQO frame-
work design.
/uni0000002b/uni00000044/uni00000056/uni00000003/uni00000024/uni00000050/uni00000045/uni0000004c/uni0000004a/uni00000058/uni0000004c/uni00000057/uni0000005c /uni0000002b/uni00000044/uni00000056/uni00000003/uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni00000010/uni0000002c/uni00000051/uni00000057/uni00000048/uni00000051/uni00000057 /uni00000031/uni00000048/uni00000048/uni00000047/uni00000056/uni00000003/uni00000027/uni00000048/uni00000046/uni00000052/uni00000050/uni00000053/uni00000052/uni00000056/uni0000004c/uni00000057/uni0000004c/uni00000052/uni00000051 /uni00000024/uni00000059/uni0000004a/uni00000011/uni00000003/uni00000032/uni00000053/uni00000057/uni0000004c/uni00000050/uni00000044/uni0000004f
/uni00000034/uni00000058/uni00000048/uni00000055/uni0000005c/uni00000003/uni00000026/uni00000052/uni00000058/uni00000051/uni00000057/uni00000013/uni00000015/uni00000018/uni00000018/uni00000013/uni0000001a/uni00000018/uni00000014/uni00000013/uni00000013/uni00000033/uni00000048/uni00000055/uni00000046/uni00000048/uni00000051/uni00000057/uni00000003/uni0000000b/uni00000008/uni0000000c/uni0000001c/uni00000019/uni00000011/uni00000013
/uni00000013/uni00000011/uni00000018/uni00000014/uni00000018/uni00000011/uni00000018/uni00000019/uni0000001c/uni00000011/uni00000013
/uni00000016/uni00000015/uni00000011/uni00000018/uni00000017/uni00000017/uni00000011/uni00000018/uni0000001a/uni00000019/uni00000011/uni0000001c/uni0000001a/uni0000001c/uni00000011/uni0000001c/uni0000001b/uni00000017/uni00000011/uni0000001c/uni00000037/uni00000052/uni00000053/uni0000004c/uni00000032/uni00000026/uni00000034/uni00000024 /uni0000002b/uni00000052/uni00000057/uni00000053/uni00000052/uni00000057/uni00000034/uni00000024 /uni00000030/uni00000058/uni0000004f/uni00000057/uni0000004c/uni0000004b/uni00000052/uni00000053/uni00000010/uni00000035/uni00000024/uni0000002a
/uni00000013/uni00000014/uni00000015/uni00000016
/uni00000026/uni00000052/uni00000058/uni00000051/uni00000057/uni00000014/uni00000011/uni00000015/uni00000014/uni00000011/uni0000001c/uni00000015/uni00000011/uni00000018
Figure 1: Distribution of query complexity.
1 2 3 4 5
Query Count35404550556065Performance (%)
TopiOCQA-ANCE
TopiOCQA-BM25
HotpotQA-ANCE
HotpotQA-BM25
ACQO (ours)
Figure 2: Performance with different query counts.
2.1 Query Complexity Analysis Framework
We analyze three representative RAG benchmarks:
TOPIOCQA for multi-turn conversational QA,
HOTPOTQA for multi-hop factual reasoning, and
MULTIHOP-RAG for real-world multi-hop re-
trieval. For each query, we conduct a structured
analysis using the following criteria:
•Ambiguity Detection: Flag ambiguous entities
or references that need disambiguation.
•Multi-Intent Analysis: Identify distinct intents
embedded in the query.
•Decomposition Assessment: Judge whether de-
composition improves answerability.
•Optimal Granularity: Identify the minimum
number of sub-queries from the generated set
that yields optimal retrieval metrics.
We analyze 200 representative queries from each
dataset, focusing on understanding the distribution
and characteristics of complex queries in real-world
scenarios.MethodTopiOCQA (Recall@10) HotpotQA (MAP@10)
ANCE BM25 ANCE BM25
Easy Hard Easy Hard Easy Hard Easy Hard
Prompt-based 59.4 52.6 34.3 45.1 36.8 31.2 50.1 40.5
SFT 56.2 54.8 33.1 38.7 44.7 33.5 45.2 43.7
Vanilla RL 63.9 54.8 58.5 61.2 42.3 38.2 50.4 46.0
ACQO (ours) 66.2 58.0 60.3 64.5 50.4 41.5 53.1 48.3
Table 1: Performance comparison on easy vs. hard
query subsets across datasets and retrievers (%).
2.2 Dataset Analysis: Prevalence of Complex
Queries
Our structured analysis reveals significant complex-
ity patterns across the three toy datasets, with Fig-
ure 1 illustrating the distribution of query char-
acteristics. Specifically, a substantial proportion
of queries are complex: on average, 48.3% re-
quire decomposition, and 37.6% exhibit multiple
intents. Moreover, the optimal number of sub-
queries varies across domains (1.2–2.5 on average),
indicating that decomposition strategies must be
context-sensitive rather than one-size-fits-all.
2.3 Why Current Methods Struggle with
Complex Queries
We evaluate representative query optimization ap-
proaches across different paradigms: prompt-based
optimization usingDeepSeek-V3.1(DeepSeek-AI,
2024) with decomposition prompts, supervised
fine-tuning (SFT) viaQwen2.5-3B(Qwen, 2024)
query rewriter, and vanilla reinforcement learning
(REINFORCE with sparse rewards) also based on
Qwen2.5-3B.
The performance analysis in Table 1 reveals
critical limitations of existing approaches when
handling complex queries. Current methods ex-
hibit substantial performance variations between
easy and hard queries, with SFT approaches
showing dramatic drops of up to 11.2% on Hot-
potQA (44.7% to 33.5% with ANCE). Moreover,
optimal approaches vary significantly across re-
trieval systems—vanilla RL excels with BM25
(61.2%) but degrades with ANCE (54.8%) on hard
queries.Figure 2 further demonstrates that fixed de-
composition strategies suffer from dual limitations
in both efficiency and effectiveness. These incon-
sistent patterns highlight the absence of principled
approaches for systematic query optimization, re-
vealing three critical gaps: adaptive complexity
recognition, retriever-aware optimization, and ef-
fective integration for decomposed queries.

3 A daptive Complex Query Optimization
3.1 Task Formulation
In traditional Query Optimization (QO) pipeline,
the task is defined as refining the query to retrieve
the golden document(s) relevant to the user’s cur-
rent query and conversational context (if any) from
a large collection of documents. Formally, given
the current query q(t)(t≥1 ) and its historical con-
textC(t−1)={(q i, ai)}(t−1)
i=1 (ift≥2) , where
tdenotes the current turn number, a query opti-
mization model Θgenerates a de-contextualized
query ˆq(t)c.ˆq((t)is omitted for simplicity) is
subsequently input into a retrieval system, which
returns a ranked list of the top- kdocuments from
the collection P. We denote this ranked set as
Rk(ˆq) ={p 1, p2, . . . , p k},R k(ˆq)⊆ P , where
pirepresents the document ranked at position i.
LetP∗⊆ P denote the set of golden documents
corresponding to ˆq. The objective of QO is (1) to
maximize the probability that at least one golden
document in P∗appears in Rk(ˆq); and (2) to mini-
mize the ranking positions of the golden documents
withinR k(ˆq).
In our work, we extend this formulation by con-
sidering the disambiguation and decomposition
scenarios, where an optimized query set ˆQqwill
be generated. Each sub-query ˆqq∈ˆQqretrieves
its own top- kdocuments Rk(ˆqq), and these can-
didates are subsequently combined and re-ranked
to produce the final top- kdocuments, denoted as
Rk(ˆQq). This design enhances both the coverage
and ranking quality of golden documents.
3.2 Overall Framework
As illustrated in Figure 3, ACQO proceeds in two
curriculum reinforcement learning (CRL) stages:
(1)Explore CRL, which promotes broad explo-
ration and early stabilization; and (2)Converge
CRL, which emphasizes precision and convergence
on harder cases.
The core idea is to integrate query optimization
with CRL in a fully self-directed manner. With-
out external supervision or intervention, the model
learns to adaptively converge to suitable query num-
bers and optimization strategies across heteroge-
neous retrieval systems. In the following, we first
introduce our re-ranker design, which consolidates
multiple retrieval lists produced from the query set,
and then detail the two-stage CRL procedure.
Figure 3: Overview of ACQO. ACQO employs two-
stage curriculum reinforcement learning to adaptively
optimize complex queries and integrate multi-retrieval
results via Rank-Score Fusion.
3.3 Re-ranker Design
Method.Inspired by Reciprocal Rank Fusion
(RRF), we propose a new method namedRank-
Score Fusion (RSF)to address two key limitations
of RRF: it only considers rank positions while ig-
noring absolute retrieval scores, and it cannot prop-
erly handle cases where documents obtain identical
ranks across multiple lists.
In RSF, each sub-query returns a ranked list of
candidate documents, where each document is as-
sociated with a retrieval(e.g., ANCE) score and a
rank position. For a given document p, we collect
its appearances across all Msub-queries into a set
{(sj, rj)}M
j=1,where (sj, rj)denotes the score and
rank of document pin the j-th sub-query. We then
compute two aggregated quantities forp:
P(p) =1PM
j=11
rj, S(p) = max
j=1,...,Msj.(1)
Here, P(p) reflects the combined influence of rank
positions (relative values), while S(p) captures the
strongest absolute score observed for document
p. We therefore perform lexicographical sorting
with P(p) as the primary key (ascending order:
lower rank indicates better consensus) and S(p) as
the secondary key (descending order: higher score
indicates stronger evidence). Formally, candidate
documents are re-ranked according to:
Rk=Top-k 
sort{(p, P(p), S(p))|
p∈ R k(q1)∪ ··· ∪ R k(qM)}
,(2)
where the sorting key is (P(p),−S(p)) in ascend-
ing order. This encodes a hierarchical preference:

"Trust rank consensus first; use scores only to break
ties among similarly-ranked documents."
Advantages.Our RSF method inherits RRF’s
simplicity and efficiency while extending its capa-
bility through score integration. RSF offers three
key advantages: (1)Zero latency overhead: intro-
duces no inference delay and seamlessly integrates
with neural re-rankers. (2)Universal compatibil-
ity: directly applicable to both sparse (e.g., BM25)
and dense (e.g., ANCE) retrievers across differ-
ent index structures. (3)Enhanced robustness:
leverages both rank positions and absolute scores
for more balanced re-ranking while resolving rank
ambiguities.
3.4 Curriculum Reinforcement Learning
3.4.1 Base Reward Function
We build upon the Rank-Incentive Reward Shap-
ing (RIRS) framework proposed in ConvSearch-
R1 (Zhu et al., 2025), which provides dense rank-
based reward signals and alleviates the sparsity of
traditional metrics such as NDCG and MRR. Here,
the rank ris defined as the position assigned to a
document in the re-ranked list Rkfrom our RSF
module. The base rank-to-score mapping employs
a continuous piecewise linear transformation:
Φ(r) =f [1,10]→[1,2] (r)·I [1,10](r)
+f(10,100]→[0,1) (r)·I (10,100] (r),(3)
where fA→B represents a linear mapping function
from interval Ato interval B,IA(r)is the indicator
function that equals 1 when r∈A and 0 otherwise,
andris the rank variable.
To accommodate multiple relevant documents,
we employ a weighted aggregation score empha-
sizes the most promising retrieval results. Sup-
pose the rank of nretrieved relevant documents in
ranked set Rarer1, r2, ..., r nrespectively, the ri
score is defined as:
s(ri) =ηi·Φ(r i),(4)
where ηis the decay coefficient. This generaliza-
tion retains the dense reward structure of RIRS
while providing additional flexibility to adapt the
weighting scheme for different retrieval scenarios.
Taking the format correctness into the considera-
tion, the complete reward score is defined as:
S(R) =nX
i=1s(ri)·Iformat +δ·(1−I format )(5)
where Iformat serves as the format compliance
gate, and δ <0 represents the non-compliance
penalty coefficient.3.4.2 Stage I: Explore-Oriented CRL
Data Curriculum.In the exploration stage, we
employ the full training dataset without filtering.
This ensures that the model is exposed to both easy
and hard cases, providing sufficient diversity to
stabilize early training and improve robustness. By
leveraging the entire dataset, the model can better
explore the space of optimization without being
biased toward specific difficulty levels.
Reward Design.Building upon the base reward
function, Stage I encourages exploration by rein-
forcing thecombination of the best-performed sub-
queries. Suppose ˆQis the set of optimized sub-
queries, and for any non-empty subset ˆQ′in the
power set of ˆQ, denoted as P(ˆQ), we compute its
the stage-specific reward as:
G(I)(ˆQ) = max
ˆQ′∈P(ˆQ)\∅S(R k(ˆQ′)).(6)
This design allows the model to explore diverse
decomposition strategies and ensures that promis-
ing sub-queries are strongly reinforced, even in the
early stage when the model is not yet stable.
3.4.3 Stage II: Converge-Oriented CRL
Data Curriculum.In the convergence stage, we
refine the training distribution by focusing on the
tougher cases. Rather than arbitrary filtering, we
identify the optimal learning frontier by analyzing
the performance distribution of Stage I models.
Formally, let Qtrain denote the full training
query set. We define thelearning complexity score
for each input queryqas:
τ(q) =1
KKX
k=1G(I)(ˆQ(k)
q),(7)
where Kdenotes the number of rollouts. The con-
vergence curriculum Qconv is constructed by re-
taining samples withinoptimal challenge zone:
Qconv={q∈ Q train :τ(x i)≤τ thres}(8)
where τthres is the theoretical boundary indicating
when retrieval performance is sufficient to continue
learning without destabilizing optimization.
This principled approach ensures that the model
focuses on samples that are neither trivially easy
(already mastered) nor prohibitively difficult (lead-
ing to sparse learning signals), thereby maximizing
learning efficiency in the convergence phase.

Method NS NCoTSparse(BM25) Dense(ANCE)
MRR@3 NDCG@3 R@10 R@100 MRR@3 NDCG@3 R@10 R@100
DeepSeek-V3.1 - - 15.5 17.0 36.7 65.3 28.4 30.8 56.3 77.8
vanilla RL(Qwen2.5-3B)- - 31.2 36.1 60.8 82.5 34.5 38.3 62.1 81.1
IterCQR(T5-base)×✓ 16.5 14.9 29.3 54.1 26.3 25.1 42.6 62.0
ADACQR(T5-base+LLaMA7B)×✓ 28.3 26.5 48.9 71.2 38.5 37.6 58.4 75.0
LLM4CS-RAR(ChatGPT)✓× 27.9 26.4 48.4 71.1 35.4 34.4 55.2 72.2
CHIQ-Fusion(T5-base+LLaMA2-7B)×✓ 25.6 23.5 44.7 – 38.0 37.0 61.6 –
RETPO((LLaMA2-7B)×✓ 28.3 26.5 48.3 73.1 32.2 31.1 51.6 69.5
AdaQR(T5-base)×✓ 20.3 18.0 37.1 66.2 38.1 36.6 61.3 79.9
ConvSearch-R1(Qwen2.5-3B)× × 37.836.2 59.6 80.1 50.5 50.1 72.0 86.3
ACQO(ours, Qwen2.5-3B)✓ ✓ 34.9 37.7 62.6 83.2 36.6 39.4 65.6 85.1
Table 2: Retrieval performance comparison on TopiOCQA (%).NSdenotes training without rewrite supervised
data, andNCoTdenotes training without chain-of-thought reasoning.
Reward Design.Stage II transitions from ex-
ploratory reward maximization to precision-
focused optimization via a reward architecture that
emphasizes ranking quality over quantity explo-
ration. The Stage II reward function directly evalu-
ates the complete sub-query ensemble:
G(II)(ˆQ) =S(R k(ˆQ)).(9)
To address the inherent challenge of sparse pos-
itive signals in top-ranked positions, we intro-
duce alogarithmic precision weightingmecha-
nism, inspired by NDCG’s theoretical foundation,
which reflects the information-theoretic principle
that higher-ranked results contribute exponentially
more to user satisfaction, which is defined as:
Φ′(r) = Φ(r) +λ·Ir≤k∗
log2(r+ 1),(10)
where λ >0 is a precision amplification parameter,
andk∗represents the critical ranking threshold,
Ir≤k∗is the indicator function ensuring bonuses
apply only to top-tier results.
This bonus-based design provides stronger incen-
tives for exact top placements while still leveraging
the smooth decay ofΦ(r)for other positions. The
model then gradually shifts from broad exploration
in Stage I to precise convergence in Stage II.
4 Experiments
4.1 Experiments Setup
Datasets.We train and evaluate our model
on three representative benchmarks that cover
bothmulti-turnconversational query optimization,
which primarily focus on querydisambiguation,
andmulti-hopquery optimization task focusing
on querydecomposition. For disambiguation task,
we use TopiOCQA (Adlakha et al., 2022), a chal-
lenging open-domain conversational QA dataset
with topic shifts. For decomposition task, we adoptMethod R@4 R@10 R@100 MAP@10 MRR@10 NDCG@10
Sparse (BM25)
Raw 83.3 88.9 96.7 49.5 75.4 70.5
Qwen2.5-3B-inst (wo/qd) 72.0 79.3 89.7 41.2 64.2 60.5
Qwen2.5-3B-inst (w/qd) 75.3 81.2 89.5 42.7 65.9 62.4
DeepSeek-V3.1 (w/qd) 81.1 86.6 93.3 49.1 70.6 66.2
vanilla RL(Qwen2.5-3B)82.3 89.9 95.6 48.8 77.5 73.2
ConvSearch-R1(Qwen2.5-3B)83.0 90.2 96.0 51.1 77.0 72.3
ACQO(ours, Qwen2.5-3B)86.9 91.6 97.5 51.2 77.7 74.2
Dense (ANCE)
Raw 68.3 74.8 86.1 34.8 60.4 59.5
Qwen2.5-3B-inst (wo/qd) 64.6 70.7 81.8 32.8 56.6 56.0
Qwen2.5-3B-inst (w/qd) 67.0 73.0 81.8 34.9 57.5 57.3
DeepSeek-V3.1 (w/qd) 77.4 82.5 88.9 46.1 66.8 65.7
vanilla RL(Qwen2.5-3B)79.2 83.9 89.1 41.1 75.5 74.4
ConvSearch-R1(Qwen2.5-3B)75.0 79.4 87.5 44.4 72.8 72.2
ACQO(ours, Qwen2.5-3B)82.2 85.8 91.2 49.6 73.4 73.6
Table 3: Retrieval performance comparison on Hot-
potQA (%). (qd: query decomposition)
HotpotQA (Yang et al., 2018) and evaluate general-
ization on MultiHop-RAG (Tang and Yang, 2024),
a RAG-focused multi-hop retrieval benchmark.
Baselines.We compare against three categories
of prior work. For single query optimization
reformulation and abstarction, we includeIter-
CQR(Jang et al., 2024),ADACQR(Lai et al.,
2025), andConvSearch-R1(Zhu et al., 2025).
For query optimization with expansion, we evalu-
ateddLLM4CS-RAR(Mao et al., 2023),CHIQ-
Fusion(Mo et al., 2024),RETPO(Yoon et al.,
2025), andAdaQR(Zhang et al., 2024). For the
complex query optimization setting, as there are no
dedicated methods, we construct few-shot prompt-
ing baselines by adapting the above methods. We
report post optimization retrieval performance after
applying each baseline’s optimization procedure.
The details regarding retriever, implementation
and evaluation metrics are provided in Appendix A.
4.2 Main Results
Table 2 and 3 show the retrieval performance of
our method on TopiOCQA and HotpotQA. We fur-
ther present comprehensive end-to-end RAG ex-
periment results in B.4, which also validate the
effectiveness of our method.

The results on TopiOCQA demostrate that
ACQO significantly outperforms most methods
across different retrieval settings. Notably, our
method achieves competitive performance (34.9%
MRR@3, 37.7% NDCG@3) using self-supervised
via retrieval feedback, whileConvSearch-R1
achieves a strong 37.8% MRR@3 in sparse re-
trieval, this performance stems primarily from its
extended reasoning process and aggressive rewrite
expansion mechanisms, which are also present in
other methods. As shown in Table 7 and Table 10a,
its strong performance comes at the cost of both
over 10× more tokens than our method and a 9.1×
increase in inference latency for the observed gains,
making it too slow and resource-heavy for practical
end-to-end RAG use, which gains driven by scale,
not scalable design. However, ACQO demonstrates
superior generalization capabilities, achieving the
best R@10 (62.6%) and R@100 (83.2%) perfor-
mance on sparse retrieval. In dense retrieval set-
tings, ACQO shows remarkable effectiveness, at-
taining competitive MRR@3 (36.6%), NDCG@3
(39.4%) and R@10 (65.6%), demonstrating its abil-
ity to work across different retrieval architectures.
On HotpotQA, using only a 3B parameter model,
ACQO achieves the best results across all met-
rics under both sparse and dense retrieval set-
tings. Notably, ACQO outperforms ConvSearch-
R1 on this more challenging multi-hop dataset
(49.6% vs. 44.4% MAP@10), demonstrating su-
perior decomposition capability. While query de-
composition often boosts performance over non-
decomposition methods, even strong baselines
(e.g., DeepSeek V3.1) fall short of raw queries in
sparse retrieval—indicating straightforward decom-
position may harm multi-hop retrieval. In contrast,
ACQO avoids such degradation and significantly
outperforms the raw query: in sparse retrieval, it
achieves 86.9% R@4 (+3.6%) and 91.6% R@10
(+2.7%); in dense retrieval, it reaches 82.2% R@4
(+13.9%) and 85.8% R@10 (+11.0%), outperform-
ing the best baseline by +4.8% and +3.3% respec-
tively. These results demonstrate that ACQO suc-
cessfully bridges the gap between query decompo-
sition and retrieval alignment, delivering superior
and robust performance without relying on larger
models or sacrificing efficiency.
4.3 Evaluation on Out-of-Distribution Data
A critical strength of our ACQO framework lies in
its strong generalization to entirely unseen datasets.
As shown in Table 4, when evaluated on MultiHop-Methodbge-large-en-v1.5
MRR@10 MAP@10 R@10 R@4
Raw 45.5 21.5 81.3 62.5
Qwen2.5-3B(w/qd) 44.8 21.2 80.5 61.7
ACQO(ours)47.7 23.6 84.0 65.5
Methodllm-embedder
MRR@10 MAP@10 R@10 R@4
Raw 32.9 14.4 65.7 45.7
Qwen2.5-3B(w/qd) 33.2 14.7 65.7 45.9
ACQO(ours)35.6 17.3 72.6 49.7
Table 4: Retrieval performance comparison on
MultiHop-RAG (%).
RAG, ACQO consistently outperforms raw queries
and all baselines across different retrievers. It
achieves 49.7% R@4 compared to 45.7% for
raw, with clear gains of 4% usingllm-embedder
and 3% usingbge-large-en-v1.5, confirming its
compatibility with varying retrieval architectures.
ACQO maintains strong performance on unseen do-
mains and query types, indicating it learns domain-
invariant reformulation principles. All gains are
achieved zero-shot without fine-tuning, which con-
firming it generalizes beyond dataset-specific pat-
terns, making it highly adaptable to real-world re-
trieval systems with shifting data.
4.4 Ablation Study
In this work, we have presented ACQO with three
core components: Query Decomposition (QD) for
adaptive query optimization, Rank-Score Fusion
(RSF) for robust result aggregation, and a two-stage
Curriculum Reinforcement Learning approach for
stable training. We conduct comprehensive abla-
tion studies on these components across both Topi-
OCQA and HotpotQA datasets to understand their
individual contributions. As shown in Table 5, all
three components are essential for optimal perfor-
mance, with removing any single component lead-
ing to noticeable performance drops across both
dense and sparse retrievers.
Rank-Score Fusion (RSF)emerges as the most
critical component, with its removal causing the
most significant performance degradation on Top-
iOCQA (37.7% →35.0% NDCG@3 for sparse,
39.4%→38.8% for dense), demonstrating that ef-
fective aggregation of multiple query results is fun-
damental to our approach.Curriculum Reinforce-
ment Learningshows dramatic impact on training
stability, with substantial performance drops with-
out it (37.7% →24.9% NDCG@3 for sparse on
TopiOCQA), indicating that the convergence phase
is essential for stable learning. We argue that Stage

Dataset TopiOCQA HotpotQA
Retriver Sparse Dense Sparse Dense
Method NDCG@3 R@3 R@10 NDCG@3 R@3 R@10 MAP@10 R@3 R@10 MAP@10 R@3 R@10
- wo/ RSF 35.0 42.1 58.8 38.8 46.6 63.4 51.2 83.5 91.1 49.0 80.185.6
- wo/ Stage II 24.9 30.6 49.1 36.3 44.2 64.9 52.084.691.840.6 69.4 75.1
- wo/ QD 36.5 44.1 61.1 38.7 46.1 63.2 49.9 83.1 90.5 42.3 79.6 84.7
ACQO 37.7 45.8 62.6 39.4 47.8 65.6 51.284.891.649.4 80.5 85.6
Table 5: Ablation study on retrieval performance (%).
I (exploration) discovers diverse query reformula-
tion strategies, while Stage II (convergence) refines
these strategies for optimal performance.Query
Decomposition (QD)shows moderate but consis-
tent improvements (37.7% →36.5% NDCG@3
for sparse on TopiOCQA), which aligns with ex-
pectations since TopiOCQA primarily involves dis-
ambiguation rather than complex query decompo-
sition, yet QD still provides benefits for handling
multi-faceted information needs.
The synergistic effects of all components create
a robust framework where each component com-
pensates for the limitations of others, establishing
that ACQO requires all three components working
in concert to achieve state-of-the-art performance.
4.5 Training Dynamics Analysis
Figure 4 illustrates the training progression of our
two-stage curriculum learning approach on Topi-
OCQA and HotpotQA datasets. The results demon-
strate the expected behavior of our adaptive query
optimization framework.
As shown in both datasets, the average
query count follows a characteristicexplore-then-
convergepattern: initially increasing during Stage
I (exploration) as the model learns to decompose
complex queries, then stabilizing or slightly de-
creasing during Stage II (convergence) as the model
refines its decomposition strategies. This behavior
aligns with our curriculum learning design, where
the model first explores diverse query reformula-
tion patterns before strategy convergence.
The retrieval performance (R@10 for Topi-
OCQA, MAP@10 for HotpotQA) shows consistent
improvement throughout training, with merged sub-
queries significantly outperforming baselines and
approaching the performance of best subqueries.
Notably, different retrievers result in different op-
timal query counts after training, which corrobo-
rates our finding that effective query optimization
requires retriever-specific adaptation.
The training dynamics validate that our two-
stage approach successfully balances exploration
and exploitation, achieving both improved retrievaleffectiveness and computational efficiency through
adaptive query count optimization. We demon-
strate the effectiveness of ACQO through state-of-
the-art performance on TopiOCQA and HotpotQA
datasets. In addition, the experimental results indi-
cate that ACQO learns retriever-specific optimiza-
tion strategies, with different retrievers yielding dif-
ferent optimal query patterns. Furthermore, ACQO
exhibits superior performance in challenging set-
tings such as generalization on unseen datasets and
computational efficiency with smaller models.
0 100 200 300 400 500 572
Step020406080R@10 (%)
170Stage 1 Stage 2Worst Subquery
Best SubqueryMerged Subquery
Query CountBaseline
Baseline Query Count
1.01.52.02.53.0
Average Query Count
0 100 200 300 362
Step020406080MAP@10 (%)
170Stage 1 Stage 2Worst Subquery
Best SubqueryMerged Subquery
Query CountBaseline
Baseline Query Count
1.01.52.02.53.0
Average Query Count
Figure 4: Query adaptation and performance improve-
ment on TopiOCQA(L) and HotpotQA(R).
5 Conclusion
In this work, we propose ACQO, a two-stage rein-
forcement learning framework that addresses com-
plex query optimization in RAG systems through
self-supervised retrieval feedback, which leverages
retrieval signals via adaptive query decomposition
and rank-score fusion to provide retriever-specific
guidance for query optimization. Experimental
results demonstrate state-of-the-art performance
on TopiOCQA and HotpotQA, while achieving
9.1× faster inference than strong baselines. Our
analysis further reveals that ACQO learns retriever-
specific optimization strategies, with each retriever
yielding distinct optimal query patterns. Further-
more, our framework demonstrates superior perfor-
mance in challenging scenarios, including strong
generalization to unseen datasets and efficient op-
eration with smaller models, establishing a power-
ful, efficient, and generalizable solution for next-
generation RAG systems.

References
Vaibhav Adlakha, Shehzaad Dhuliawala, Kaheer Sule-
man, Harm de Vries, and Siva Reddy. 2022. Topi-
ocqa: Open-domain conversational question answer-
ing with topic switching.Transactions of the Associ-
ation for Computational Linguistics, 10:468–483.
Paul JL Ammann, Jonas Golde, and Alan Akbik. 2025.
Question decomposition for retrieval-augmented gen-
eration.arXiv preprint arXiv:2507.00355.
Hiteshwar Kumar Azad and Akshay Deepak. 2019.
Query expansion techniques for information retrieval:
a survey.Information Processing & Management,
56(5):1698–1735.
Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng,
Jianfeng Gao, Xiaodong Liu, Rangan Majumder, An-
drew McNamara, Bhaskar Mitra, Tri Nguyen, and
1 others. 2016. Ms marco: A human generated ma-
chine reading comprehension dataset.arXiv preprint
arXiv:1611.09268.
DeepSeek-AI. 2024. Deepseek-v3 technical report.
Preprint, arXiv:2412.19437.
Jiazhan Feng, Chongyang Tao, Xiubo Geng, Tao Shen,
Can Xu, Guodong Long, Dongyan Zhao, and Daxin
Jiang. 2023. Synergistic interplay between search
and large language models for information retrieval.
arXiv preprint arXiv:2305.07402.
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan.
2023. Precise zero-shot dense retrieval without rel-
evance labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 1762–1777.
Yizheng Huang and Jimmy Huang. 2024. A survey
on retrieval-augmented text generation for large lan-
guage models.arXiv preprint arXiv:2404.10981.
Yunah Jang, Kang-il Lee, Hyunkyung Bae, Hwanhee
Lee, and Kyomin Jung. 2024. Itercqr: Iterative con-
versational query reformulation with retrieval guid-
ance. InProceedings of the 2024 Conference of the
North American Chapter of the Association for Com-
putational Linguistics: Human Language Technolo-
gies (Volume 1: Long Papers), pages 8114–8131.
Pengcheng Jiang, Jiacheng Lin, Lang Cao, Runchu
Tian, SeongKu Kang, Zifeng Wang, Jimeng Sun,
and Jiawei Han. 2025. Deepretrieval: Hacking real
search engines and retrievers with large language
models via reinforcement learning.arXiv preprint
arXiv:2503.00223.
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019.
Billion-scale similarity search with gpus.IEEE
Transactions on Big Data, 7(3):535–547.
Yilong Lai, Jialong Wu, Congzhi Zhang, Haowen Sun,
and Deyu Zhou. 2025. Adacqr: Enhancing query
reformulation for conversational search via sparse
and dense retrieval alignment. InProceedings of
the 31st International Conference on Computational
Linguistics, pages 7698–7720.Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio
Petroni, Vladimir Karpukhin, Naman Goyal, Hein-
rich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, and 1 others. 2020. Retrieval-augmented gen-
eration for knowledge-intensive nlp tasks.Advances
in neural information processing systems, 33:9459–
9474.
Jimmy Lin, Xueguang Ma, Sheng-Chieh Lin, Jheng-
Hong Yang, Ronak Pradeep, and Rodrigo Nogueira.
2021. Pyserini: An easy-to-use python toolkit to
support replicable ir research with sparse and dense
representations.arXiv preprint arXiv:2102.10073.
Jerry Liu. 2022. LlamaIndex.
Yanming Liu, Xinyue Peng, Xuhong Zhang, Weihao
Liu, Jianwei Yin, Jiannan Cao, and Tianyu Du. 2024.
Ra-isf: Learning to answer and understand from
retrieval augmentation via iterative self-feedback.
arXiv preprint arXiv:2403.06840.
Kelong Mao, Zhicheng Dou, Fengran Mo, Jiewen Hou,
Haonan Chen, and Hongjin Qian. 2023. Large lan-
guage models know your contextual search intent:
A prompting framework for conversational search.
InFindings of the Association for Computational
Linguistics: EMNLP 2023, pages 1211–1225.
Fengran Mo, Abbas Ghaddar, Kelong Mao, Mehdi Reza-
gholizadeh, Boxing Chen, Qun Liu, and Jian-Yun Nie.
2024. Chiq: Contextual history enhancement for im-
proving query rewriting in conversational search. In
EMNLP.
Ethan Perez, Patrick Lewis, Wen-tau Yih, Kyunghyun
Cho, and Douwe Kiela. 2020. Unsupervised ques-
tion decomposition for question answering.arXiv
preprint arXiv:2002.09758.
Team Qwen. 2024. Qwen2.5: A party of foundation
models.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. 2025a. Hybridflow: A flex-
ible and efficient rlhf framework. InProceedings
of the Twentieth European Conference on Computer
Systems, pages 1279–1297.
Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin
Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin
Lin, and Chuan Wu. 2025b. Hybridflow: A flex-
ible and efficient rlhf framework. InProceedings
of the Twentieth European Conference on Computer
Systems, pages 1279–1297.
Aditi Singh, Abul Ehtesham, Saket Kumar, and Tala Ta-
laei Khoei. 2025. Agentic retrieval-augmented gen-
eration: A survey on agentic rag.arXiv preprint
arXiv:2501.09136.
Mingyang Song and Mao Zheng. 2024. A survey
of query optimization in large language models.
Preprint, arXiv:2412.17558.

Yixuan Tang and Yi Yang. 2024. Multihop-rag: Bench-
marking retrieval-augmented generation for multi-
hop queries.arXiv preprint arXiv:2401.15391.
Nandan Thakur, Nils Reimers, Andreas Rücklé, Ab-
hishek Srivastava, and Iryna Gurevych. Beir: A het-
erogeneous benchmark for zero-shot evaluation of
information retrieval models.
Svitlana Vakulenko, Nikos V oskarides, Zhucheng Tu,
and Shayne Longpre. 2021. A comparison of ques-
tion rewriting methods for conversational passage
retrieval. InEuropean Conference on Information
Retrieval, pages 418–424. Springer.
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query expansion with large language
models.arXiv preprint arXiv:2303.07678.
Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muen-
nighoff, Defu Lian, and Jian-Yun Nie. 2024. C-pack:
Packed resources for general chinese embeddings. In
Proceedings of the 47th international ACM SIGIR
conference on research and development in informa-
tion retrieval, pages 641–649.
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang,
Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold
Overwijk. 2020. Approximate nearest neighbor neg-
ative contrastive learning for dense text retrieval.
arXiv preprint arXiv:2007.00808.
Shicheng Xu, Liang Pang, Huawei Shen, Xueqi Cheng,
and Tat-Seng Chua. 2024. Search-in-the-chain: Inter-
actively enhancing large language models with search
for knowledge-intensive tasks. InProceedings of the
ACM Web Conference 2024, pages 1362–1373.
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Ben-
gio, William W Cohen, Ruslan Salakhutdinov, and
Christopher D Manning. 2018. Hotpotqa: A dataset
for diverse, explainable multi-hop question answer-
ing.arXiv preprint arXiv:1809.09600.
Chanwoong Yoon, Gangwoo Kim, Byeongguk Jeon,
Sungdong Kim, Yohan Jo, and Jaewoo Kang. 2025.
Ask optimal questions: Aligning large language mod-
els with retriever’s preference in conversation. In
Findings of the Association for Computational Lin-
guistics: NAACL 2025, pages 5899–5921.
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan,
Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan,
Gaohong Liu, Lingjun Liu, and 1 others. 2025. Dapo:
An open-source llm reinforcement learning system
at scale.arXiv preprint arXiv:2503.14476.
Shi Yu, Jiahua Liu, Jingqin Yang, Chenyan Xiong, Paul
Bennett, Jianfeng Gao, and Zhiyuan Liu. 2020. Few-
shot generative conversational query rewriting. In
Proceedings of the 43rd International ACM SIGIR
conference on research and development in Informa-
tion Retrieval, pages 1933–1936.Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng
Dou, and Jian-Yun Nie. 2023. Retrieve anything
to augment large language models.arXiv preprint
arXiv:2310.07554.
Tianhua Zhang, Kun Li, Hongyin Luo, Xixin Wu,
James R Glass, and Helen Meng. 2024. Adaptive
query rewriting: Aligning rewriters through marginal
probability of conversational answers. InEMNLP.
Changtai Zhu, Siyin Wang, Ruijun Feng, Kai Song,
and Xipeng Qiu. 2025. Convsearch-r1: Enhancing
query reformulation for conversational search with
reasoning via reinforcement learning.arXiv preprint
arXiv:2505.15776.
A Experimental Details
A.1 Retrieval
In TopiOCQA and HotpotQA, we use the BM25
retriever implemented by Pyserini (Lin et al.,
2021), and the ANCE retriever implemented by
Faiss (Johnson et al., 2019). The hyperparame-
ters of BM25 are set to k1= 0.9, b= 0.4 for
TopiOCQA, and k1= 1.2, b= 0.75 for Hot-
potQA during all training and evaluation. For
ANCE, to improve training efficiency, we first gen-
erate embeddings for documents and then build an
HNSW index using Faiss’s IndexHNSWFlat , with
parameters M= 64 andef_construction= 2000 .
The index construction for HotpotQA partially
follows (Jiang et al., 2025). During evaluation,
we use IndexFlatIP to construct a flat index
to ensure accuracy. In MultiHop-RAG, we fol-
low its original setup with the LlamaIndex (Liu,
2022) framework and adopt BGE-large-en-v1.5
and LLM-Embedder as retrievers. Both retriev-
ers use a chunking strategy with chunk_size =256
andchunk_overlap =20, splitting the 609 original
documents into 7786 chunks. For BGE-large-en-
v1.5, we follow the official recommendation and
add the instruction ”Represent this sentence for
searching relevant documents:” when converting
text into embeddings.
A.2 Training and evaluation
Evaluation Metrics.For TopiOCQA, we em-
ploy Mean Reciprocal Rank@K (MRR@K),
Normalized Discounted Cumulative Gain@K
(NDCG@K), and Recall@K (R@K) as evalua-
tion metrics. For HotpotQA and MultiHop-RAG,
we additionally use Mean Average Precision@10
(MAP@10) for assessment. For MultiHop-RAG,
we follow the evaluation code and metric provided
in the benchmark.

Retrieval Systems.We evaluated the perfor-
mance of model under both sparse and dense re-
trievers. For TopiOCQA and HotpotQA, we select
BM25as the sparse retriever and ANCE as the
dense retriever, whereANCE(Xiong et al., 2020)
is trained on MS-MARCO (Bajaj et al., 2016) doc-
ument retrieval tasks. For MultiHop-RAG, we
usebge-large-en-v1.5(Xiao et al., 2024) andllm-
embedder(Zhang et al., 2023) as the retrievers.
Implementation.We deploy Qwen2.5-3B as the
backbone and train the model individually on Top-
iOCQA and HotpotQA, following the two-stage
CRL described in §3. We use verl (Sheng et al.,
2025a) as our RL training framework, and adopt
DAPO (Yu et al., 2025) as the optimization algo-
rithm, training the models under BM25 and ANCE
retrievers independently.
Training Hyperparameters.We adopt the de-
fault hyperparameters established by ConvSearch-
R1 (Zhu et al., 2025) and the verl frame-
work (Sheng et al., 2025b), rather than performing
dataset-specific tuning. The only modification we
make is setting the maximum response length to
256 tokens (vs. 1024 in ConvSearch-R1), since
ACQO generates concise sub-queries rather than
chain-of-thought reasoning, reducing the required
output length.
Specifically, for both TopiOCQA and HotpotQA,
the models are trained under BM25 and ANCE
retrievers with essentially the same hyperparam-
eter configuration across both stages. We adopt
DAPO optimization with mini-batch size 64 and
micro-batch size 8 per GPU (8 GPUs in total).
The actor learning rate is 1×10−6with gradi-
ent clipping at 1.0. Entropy regularization is dis-
abled ( entropy_coeff= 0 ). KL control is not used
in reward shaping ( use_kl_in_reward=False ). The
clipping ratios are set as [0.2,0.28] with an addi-
tional coefficient clip_ratio_c= 10.0 , following
the default configuration of DAPO. We sample
K= 8 rollouts per query. Generation uses tem-
perature 0.8, top-p= 0.8 , and top- k=−1 during
training; for validation we set temperature= 0.7 ,
top-p= 0.8 , and top-k= 20 . Dynamic batch
sizing is enabled for efficiency, with maximum
batched tokens set to 11408 and GPU memory uti-
lization capped at 0.8. We set the training batch
size to 256 and the generation batch size to 512.
For HotpotQA, the maximum prompt length is 512
tokens, while for TopiOCQA it is 1536 tokens.
In both datasets, the maximum response lengthis fixed to 256 tokens. We set the decay coefficient
η= 0.6 for HotpotQA, while η= 1.0 is used
for TopiOCQA. For stage II reward design, we set
k∗= 3for TopiOCQA and k∗= 0for HotpotQA.
For the training epochs, Stage I CRL is trained
for 2 epochs on TopiOCQA and 3 epochs on Hot-
potQA. Stage II CRL is trained for 10 epochs on
TopiOCQA with ANCE retriever and 8 epochs on
TopiOCQA with BM25 retriever. For HotpotQA,
Stage II is trained for 4 epochs with ANCE and 6
epochs with BM25.
A.3 Datasets
We use three datasets in our experiments: Topi-
OCQA, HotpotQA, and MultiHop-RAG. All ex-
periments are conducted on standard training/test
splits and document collections, as summarized
in Table 6. For HotpotQA, we follow the corpus
provided by the BEIR benchmark (Thakur et al.),
which standardizes the document collection for
retrieval-based evaluation.
TopiOCQA follows CC-BY-NC-SA 4.0 li-
cense. HotpotQA follows CC-BY-SA 4.0 license.
MultiHop-RAG follows ODC-BY license. All
reuse, modification, and related operations on these
datasets strictly adhered to the copyright statements
of the data owners.
Dataset Split #Queries #documents #Golden / query
TopiOCQAtrain 4545025700592 1test 2514
HotpotQAtrain 850005233329 2test 7405
MultiHop-RAG test 2556 7786 multiple
Table 6: Statistics of datasets used in our experiments.
“#Golden / query” denotes the number of golden docu-
ments associated with each query.
Data CollectionOur method does not require
any supervised data; instead, it employ RL with
different levels of difficulty across the two RL
stages(Section 3.4). In Stage I CRL, we use the full
official training set for TopiOCQA. For HotpotQA,
however, given the large training set size and rel-
atively high initial performance, we first filter out
the higher-performing samples and retain only 50%
of the data for Stage I training. In Stage II CRL,
we apply dynamic filtering with the Stage I model
(Section 3.4.3), again retaining roughly 50% of the
samples. Basically, we set τthres =5
3and roll-
outsn= 8 . This guides the model to focus on

moderately difficult instances, thereby improving
learning efficiency and convergence in Stage II.
B Further Experiments
B.1 Case Study
In this section, we begin with a representative case
to further discuss how our method improves re-
trieval performance.
In Figure 5, we compare the performance across
different training–retrieval combinations on differ-
ent datasets, i.e., the effectiveness of reformulated
queries generated by models trained with a specific
retriever when evaluated on other types of retriev-
ers.
From the perspective of retrieval performance,
we observe that retrieval effectiveness drops
significantly when switching retrievers; more-
over, model-generated reformulations outperform
human-written ones (i.e., reformulations that are
intuitively considered correct). This suggests that
the evaluation of query quality should be retriever-
dependent and may not necessarily align with hu-
man intuition.
From the perspective of query generation behav-
ior, queries generated with different retrievers vary
in both quantity and style, indicating that the refor-
mulation style learned by the model is closely tied
to the retriever used during training.
What behavior does the model learn when
trained with a specific retriever?Models
trained with the ANCE retriever tend to generate
multiple queries resembling natural language ques-
tions or statements, capturing complete semantic
relations and emphasizing keywords or core en-
tities with fewer stopwords. In contrast, models
trained with the BM25 retriever are inclined to gen-
erate a single query that explicitly enumerates all
relevant keywords.
Are these behaviors aligned with retriever pref-
erences?Indeed, the observed behaviors are con-
sistent with the characteristics favored by differ-
ent retrievers. For dense retrievers such as ANCE,
queries expressed in a natural language style, of-
ten decomposed into multiple sub-queries, better
capture semantic relations and leverage embedding-
based similarity. In contrast, sparse retrievers like
BM25 prefer a single query containing exhaustive
keyword coverage, where term frequency and ex-
act lexical overlap dominate ranking. This align-
ment indicates that our model effectively adapts itsreformulation strategy to the underlying retriever,
learning to generate query styles that are inherently
compatible with the retriever’s scoring mechanism.
Why does our method also yield improvements
on TopiOCQA?TopiOCQA consists of single-
intent questions, each associated with only one
golden document, which suggests that the opti-
mal query should ideally be a single reformulation.
Traditional approaches mainly rely onexpansion,
where the model generates a lengthy reformulation
that conveys complete semantic information while
leveraging its parametric knowledge to give an an-
swer of the question, thereby increasing semantic
similarity with candidate passages (see Table 7). In
practice, however, we observe that the model of-
ten employsrephrasing—for example, expressing
the same intent as either a question or a declara-
tive statement—to broaden the search space and
consequently achieve better retrieval results. Its
advantages lie in stronger readability and higher
efficiency, while also mitigating the negative im-
pact of erroneous expansions when the model en-
counters unfamiliar or ambiguous queries, thereby
improving robustness.
Method Retrieval Reasoning Response Total
ConvSearch-R1 - 106 248 354
ACQO(ours)ANCE - 28 28
BM25 - 36 36
Table 7: Comparison of generation lengths across meth-
ods.
Why is human-preferred query reformulation
worse than model-generated?Here we col-
lectively refer to strong instruction models (e.g.,
DeepSeek-V3) and human-written rewrites as
human-preferred query reformulation, since such
models can generate queries that are generally re-
garded as high-quality under instruction or few-
shot settings. In contrast, we denote reformula-
tions produced by our trained models asmodel-
generated query reformulation. However, experi-
mental results show that human-preferred reformu-
lations still underperform compared to our method
or other advanced baselines. Based on the above
analysis, we summarize two main reasons:
(1)Human-preferred methods do not know
what constitutes a retrieval-effective query.They
tend to follow human instructions by completing
the current query with context or decomposing
multi-intent queries into several sub-queries. Yet,

(a) TopiOCQA
(b) HotpotQA
Figure 5: Comparison of queries generated by models trained with different retrievers and their retrieval performance
across different retrievers.
when no clear sub-intent is present, they fail to
decide how todecompose, and typically do not
performrephrasingorexpansion.
(2)Human-preferred methods are not capa-
ble of generating retrieval-specific reformula-
tions.For example, their relative performance
gap to state-of-the-art baselines is larger on BM25
than on ANCE, since—as we have observed ear-
lier—some queries with “poorer readability” may
actually perform better under BM25.
This finding suggests that analyzing retriever-
specific data generated through ACQO can provide
insights into retriever preferences, which in turn
can be used to optimize prompts for large language
models and improve their performance on query
optimization tasks.
Taken together, our method enables the model,
without any supervised data and solely using re-
trieval performance as the reward, to autonomously
adapt to the retriever type. In doing so, the model isable to capture reformulation patterns that are more
compatible with the retriever, ultimately leading to
optimal reformulations.
B.2 Scaling Capabilities
Figure 6 presents the experimental results of
our method with Qwen2.5-3B and Qwen2.5-7B.
Across both datasets and retrievers, the larger
model consistently achieves better performance,
demonstrating that our approach exhibits strong
scaling capability.
B.3 SFT vs. RL Comparison.
Table 8 presents a systematic comparison of train-
ing strategies on the TopiOCQA dataset with the
ANCE retriever. For SFT baselines, the training
data are constructed by rolling out the Stage I
CRL model under our framework and filtering out
queries with poor rankings. These results collec-
tively demonstrate that ACQO’s two-stage curricu-
lum reinforcement learning effectively addresses

MRR@3 NDCG@3 R@3 R@10 R@100020406080Percentage (%)34.937.745.862.683.2
36.939.747.864.685.2
36.639.447.865.685.1
38.941.950.667.986.2Sparse + 3B
Sparse + 7BDense + 3B
Dense + 7B(a) Comparison on TopiOCQA.
MRR@10 NDCG@10 R@4 R@10 MAP@10020406080Percentage (%)77.7
74.286.991.6
51.280.8
76.588.892.9
55.973.4 73.682.285.8
49.675.774.784.187.3
52.9Sparse + 3B
Sparse + 7BDense + 3B
Dense + 7B (b) Comparison on HotpotQA.
Figure 6: Scaling Capabilities.
the fundamental challenges of complex query op-
timization, consistently outperforming both super-
vised baselines and vanilla RL approaches while
maintaining training stability and data efficiency.
Method MRR@3 NDCG@3 R@3 R@10 R@100
SFT 28.4 30.7 37.3 53.4 71.5
Vanilla RL 34.5 38.3 34.6 62.1 81.1
SFT + RL 33.4 37.8 45.7 61.6 82.2
Stage I only 33.6 36.6 44.2 64.985.8
Stage I + SFT 28.5 30.8 37.7 53.3 70.2
ACQO(ours)36.6 39.4 47.8 65.685.1
Table 8: Comparison between SFT and RL methods on
the TopiOCQA dataset with the ANCE retriever.
B.4 End-to-End Question Answering
Evaluation
While the retrieval metrics presented above demon-
strate ACQO’s effectiveness in query optimization,
a critical question remains: do these retrieval im-
provements translate to better final answers in real-
world RAG applications? To address this concern,
we conduct comprehensive end-to-end question an-
swering experiments that evaluate the complete
RAG pipeline from query optimization to answer
generation.
Experimental Setup.We use Qwen2.5-7B-
Instruct (Qwen, 2024) as the reader model and
DeepSeek-R1 (DeepSeek-AI, 2024) as the eval-
uation judge to assess answer quality on HotpotQA
with the ANCE retriever. For each method (Raw
Query, ConvSearch-R1, ACQO), we retrieve the
top-10 documents using the optimized queries and
provide them as context to the reader model, then
evaluate the generated answers based on accuracy.Results and Analysis.Table 9 presents the
end-to-end evaluation results, comparing retrieval
performance (MAP@10) with answer accuracy
(ACC L). The results reveal a strong correlation
between retrieval quality and final answer accu-
racy across all methods. Starting from the raw
query baseline (34.8% MAP@10, 16.4% ACC L),
ConvSearch-R1 achieves substantial improvements
(44.4% MAP@10, 27.7% ACC L), while ACQO
further advances the state-of-the-art to 49.6%
MAP@10 and 31.6%ACC L.
Method MAP@10ACC L ∆ACC L
Raw Query 34.8% 16.4% -
ConvSearch-R1 44.4% 27.7% +11.3%
ACQO(ours)49.6% 31.6% +15.2%
Table 9: End-to-end question answering evaluation on
HotpotQA-ANCE. MAP@10 measures retrieval quality,
while ACC Levaluates final answer accuracy judged by
DeepSeek-R1.
Notably, ACQO achieves a +3.9% improvement
inACC Lover ConvSearch-R1, confirming that
our curriculum reinforcement learning design ef-
fectively addresses the convergence challenges in
mixed-complexity query optimization. This vali-
dates that the adaptive query decomposition and
robust rank-score fusion mechanism not only im-
prove retrieval metrics but also enhance the qual-
ity of final generated answers. Moreover, ACQO
reaches 9.1× lower latency, representing a favor-
able efficiency-accuracy trade-off for production
deployment.

Method #Q Gen Retri Rerank Tot Speed
SFT (Qwen2.5-3B) 2514 297 27 0 324 1.09×
ACQO 2514 320 30 53551.0×
ConvSearch-R1 2514 3230 25 0 32550.11×
9.16× faster than ConvSearch-R1; +31ms for +8.2% MRR@3 vs. SFT
(a) Avg Inference Latency (ms,TopiOCQA-ANCE)
Method GPU-H Conv MAP@10
Vanilla RL 8.4 No 41.1
SFT + RL 15.4 yes 45.3
ACQO (Stage I)4.2 yes 42.3
ACQO (Full)12.1 yes 49.6
(b) Training Cost (HotpotQA-ANCE)
Table 10: Efficiency analysis: inference latency and
training cost.
B.5 Latency and Cost Analysis
Inference Latency Analysis.Table 10a presents
a detailed breakdown of inference latency across
different pipeline stages. Our measurements on
TopiOCQA-ANCE with a single H20 GPU show
that ACQO adds only 31ms overhead compared to
the SFT baseline (355ms vs. 324ms). More impor-
tantly, ACQO is 9.16 ×faster than ConvSearch-R1
(355ms vs. 3255ms) while maintaining compara-
ble accuracy (as shown in Tables 2 and 3). This
substantial speedup makes ACQO a Pareto-optimal
choice for production deployment, offering the best
balance between accuracy and efficiency.
The latency breakdown reveals that the addi-
tional overhead primarily comes from query gen-
eration (+23ms) and retrieval (+3ms), with our
lightweight Rank-Score Fusion module contribut-
ing only 5ms. This validates our design philosophy
of achieving strong performance through algorith-
mic innovations rather than computationally expen-
sive components.
Training Cost Analysis.Table 10b compares
training costs on HotpotQA-ANCE using 8 H20
GPUs. Full ACQO training requires 12.1 GPU-
hours, comparable to the SFT+RL baseline (15.4
GPU-hours) but without requiring any supervised
query rewriting data. Notably, ACQO-Stage I con-
verges in only 4.2 GPU-hours while achieving
42.3% MAP@10, demonstrating efficient initial
exploration.
While vanilla RL appears faster (8.4 GPU-
hours), it fails to converge properly, getting stuck at
a low performance ceiling (41.1% MAP@10) due
to training instability. The root cause is insufficient
valid samples—the DAPO algorithm fails to col-
lect enough qualified samples within its sampling
budget ( max_num_gen_batches =20), causing pre-mature termination with suboptimal performance.
This validates the necessity of our curriculum learn-
ing strategy for stable convergence.
These results demonstrate that ACQO achieves
superior performance with practical computational
costs: (1) Inference efficiency: 9.16× faster
than ConvSearch-R1 with minimal overhead over
SFT; (2) Training efficiency: comparable cost
to SFT+RL but without supervised data require-
ments; (3) Training stability: successful conver-
gence where vanilla RL fails. This favorable
efficiency-accuracy trade-off establishes ACQO as
a practical solution for production RAG systems.
C Prompts
Figure 7a shows the prompt used in ACQO, which
remains the same across retrievers and datasets. If
no context is available, it is set to empty. The same
prompt is also employed in other experiments (e.g.,
ablation studies and supervised fine-tuning) with
query decomposition. We also provide in Figure 7b
the prompt version without query decomposition,
which is used in experiments without query decom-
position.

(a) Prompt for standard ACQO.
(b) Prompt without query decomposition.
Figure 7: Prompts used in our experiments.