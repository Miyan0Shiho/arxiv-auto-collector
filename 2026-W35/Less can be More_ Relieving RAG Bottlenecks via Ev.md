# Less can be More: Relieving RAG Bottlenecks via Evidence Frontloading and Pressure-Adaptive Budgeting

**Authors**: Weibin Cai, Reza Zafarani

**Published**: 2026-08-25 20:10:06

**PDF URL**: [https://arxiv.org/pdf/2608.25115v1](https://arxiv.org/pdf/2608.25115v1)

## Abstract
Existing methods for improving Retrieval-Augmented Generation (RAG) efficiency mainly optimize downstream LLM generation, such as context compression or serving optimization. However, RAG is an end-to-end system, and its bottleneck can shift between upstream reranking and downstream generation under different serving loads and reranking budgets.In this paper, we first empirically characterize this shifting-bottleneck behavior and show that upstream reranking can become the dominant bottleneck under high query rates or large reranking budgets. Reducing the reranking budget can relieve this bottleneck, but it may also drop supporting evidence and degrade recall. To address this problem, we propose \textbf{\textsf{PACE}} (\textbf{P}rioritized \textbf{A}daptive \textbf{C}overage of \textbf{E}vidence), a training-free framework that combines \textit{evidence frontloading} with \textit{pressure-adaptive budgeting}. \textsf{PACE} first reorders candidates by marginal evidence coverage, prioritizing documents that are query-relevant, complementary, and useful for forming multi-hop evidence chains. We show that this objective is monotone submodular, giving greedy selection a $(1-1/e)$ approximation guarantee. \textsf{PACE} then dynamically adjusts the reranking budget according to the relative pressure of the reranker and the LLM. Experiments on three multi-hop QA datasets and online serving simulations show that \textsf{PACE} improves evidence recall, reduces p95 latency under ranking-heavy workloads. More importantly, the two components together reveal that \textit{less can be more}: an evidence-dense top-ranked candidates enable higher final recall with fewer reranked documents.

## Full Text


<!-- PDF content starts -->

Less can be More: Relieving RAG Bottlenecks
via Evidence Frontloading and Pressure-Adaptive Budgeting
Weibin Cai
weibin44@data.syr.edu
Data Lab, EECS Department
Syracuse University
Syracuse, NY, USAReza Zafarani
reza@data.syr.edu
Data Lab, EECS Department
Syracuse University
Syracuse, NY, USA
Abstract
Existing methods for improving Retrieval-Augmented Generation
(RAG) efficiency mainly optimize downstream LLM generation,
such as context compression or serving optimization. However,
RAG is an end-to-end system, and its bottleneck can shift between
upstream reranking and downstream generation under different
serving loads and reranking budgets. In this paper, we first empiri-
cally characterize this shifting-bottleneck behavior and show that
upstream reranking can become the dominant bottleneck under
high query rates or large reranking budgets. Reducing the reranking
budget can relieve this bottleneck, but it may also drop supporting
evidence and degrade recall. To address this problem, we propose
PACE(PrioritizedAdaptiveCoverage ofEvidence), a training-
free framework that combinesevidence frontloadingwithpressure-
adaptive budgeting.PACEfirst reorders candidates by marginal
evidence coverage, prioritizing documents that are query-relevant,
complementary, and useful for forming multi-hop evidence chains.
We show that this objective is monotone submodular, giving greedy
selection a(1−1/𝑒)approximation guarantee.PACEthen dynami-
cally adjusts the reranking budget according to the relative pressure
of the reranker and the LLM. Experiments on three multi-hop QA
datasets and online serving simulations show thatPACEimproves
evidence recall, reduces p95 latency under ranking-heavy work-
loads. More importantly, the two components together reveal that
less can be more: an evidence-dense top-ranked candidates enable
higher final recall with fewer reranked documents.
CCS Concepts
•Information systems →Retrieval models and ranking;Eval-
uation of retrieval results.
Keywords
Retrieval-augmented generation, RAG bottleneck mitigation, evi-
dence frontloading, pressure-adaptive budgeting, evidence recall,
multi-hop question answering
1 Introduction
RAG has become a widely used paradigm for equipping Large Lan-
guage Models (LLMs) with external knowledge across diverse down-
stream tasks [ 3,17,29]. A typical RAG pipeline first retrieves query-
relevant chunks from a corpus using a dense retriever [ 12,14,35],
then applies a reranker [ 20,21,30] to refine their order and feeds
the top-𝑘chunks to the LLM as context. The quality of the gen-
erated answer therefore largely depends on whether the context
(top-𝑘chunks) covers the evidence needed to answer the query.
Evidence Recall
Reranking Budget D1.Evidence 
Frontloading
2.Pressure -adaptive BudgetingFixed𝐷
Generation -heavy Ranking -heavyAdaptive 𝐷Recall 
Gain
Lower LatencyStandard dense retrievalOurs (PACE)Denser reranker input, Higher final recall
3.Less can be moreFigure 1: HowPACEimproves RAG effectiveness and effi-
ciency. (1) Evidence frontloading makes top-ranked candi-
dates more evidence-dense, achieving higher evidence recall
at a smaller reranking budget 𝐷. (2) Pressure-adaptive bud-
geting then reduces 𝐷under ranking-heavy workloads to
lower latency. (3) Together,PACEenables denser reranker
input and higher final recall with fewer reranked documents.
Increasing𝑘can improve evidence recall, but longer contexts also
make RAG systems harder to process effectively and efficiently:
useful information may be lost in the middle [ 18], noisy sentences
can degrade generation [ 28], and long inputs introduce substantial
inference overhead [ 8,40]. Existing work addresses these issues
from two main directions. At the system level, techniques such as
KV-cache reuse [ 41], scheduling [ 39], and optimized prefilling [ 1]
accelerate LLM inference. At the data and algorithm level, context
compressors [ 6,11,37] aim to preserve core information while re-
moving irrelevant tokens from the retrieved context. These methods
reduce input length or accelerate generation, thereby alleviating
downstream bottlenecks (i.e., generation stage bottleneck) while
largely maintaining question-answering performance.
However, RAG is an end-to-end system, and improving it re-
quires coordinating upstream retrieval/reranking with downstream
generation for both effectiveness and efficiency. For effectiveness,
if necessary evidence is not recalled upstream, removing noisy
context downstream cannot recover the missing information or im-
prove answer accuracy. For efficiency, If queries are already queued
at the reranker, optimizing only the generator may not improve
end-to-end latency. Consequently, approaches that mainly optimize
arXiv:2608.25115v1  [cs.CL]  25 Aug 2026

Cai and Zafarani
downstream generation may not generalize across different RAG
configurations and serving loads. In particular, as the reranking
budget (i.e., the number of retrieved candidates sent to the reranker)
and QPS (queries per second) vary, the system bottleneck can shift
between ranking and generation. When queries arrive frequently
and many documents must be reranked, the ranking stage can
become the bottleneck; otherwise, latency may be dominated by
generation. We refer to these two scenarios asranking-heavyand
generation-heavy. Although using a small fixed reranking budget
can relieve ranking-heavy workloads, this fixed budget is difficult
to choose under dynamic serving loads, and an overly small budget
can miss necessary evidence and degrade answer quality. There-
fore, how todynamically relieve upstream ranking bottlenecks while
preserving evidence recallremains an open problem. A promising
solution requires two key properties, as illustrated in Figure 1:
Evidence frontloading.This is especially important for multi-
hop questions, where answering a query often requires multiple
supporting documents rather than a single highly relevant chunk.
If these supporting documents can be frontloaded into the first
few candidates, the system can use a smaller reranking budget,
reducing upstream workload while preserving evidence recall. Ex-
isting retrieval-side methods partially address this goal through
multi-hop retrieval or diversity-aware ranking. Multi-hop retriev-
ers usually acquire supporting evidence through iterative retrieval
or reasoning-guided search [ 13,33,36], but they require specialized
retriever training or introduce additional LLM-retrieval iterations,
which can increase serving cost. In contrast, the goal in this scenario
isnot to acquire new evidence through extra retrieval steps, but to
prioritize evidence within an existing candidate poolso that a smaller
reranking budget can still preserve recall under serving pressure.
Diversity-aware methods are closer to this setting because they
usually reorder existing candidates by reducing redundancy. For ex-
ample, maximal marginal relevance (MMR) [ 5] penalizes similarity
to previously selected documents, while Dartboard [ 22] optimizes
relevant information gain for diverse RAG retrieval. However, these
methods remain limited for multi-hop evidence frontloading due
to three issues: (i)coarse-grained relevance modeling, as they typ-
ically rely on document-level similarity rather than fine-grained
semantic dimensions; (ii)missing inter-document dependencies, as
a supporting document may be weakly related to the query but
strongly connected to another evidence document; and (iii)limited
robustness, as their performance often depends on dataset-sensitive
hyperparameters and lacks principled performance guaranties.
Pressure-adaptive budgeting.Instead of relying on a fixed budget,
it should dynamically choose the largest budget that does not make
reranking a bottleneck relative to generation. This requires jointly
considering upstream reranking pressure and downstream genera-
tion pressure, so that the system can preserve as much evidence as
possible while relieving ranking-heavy workloads. Together, these
two properties enable the system to usea smaller reranking budget
with higher evidence recall, thereby reducing upstream bottlenecks
while maintaining or even improving evidence recall.
In this work.We first characterize RAG bottlenecks under on-
line serving simulation with varying QPS and reranking budgets.
Our analysis shows that the bottleneck is not determined solely
by the relative size of the reranker and the LLM:increasing QPS orthe reranking budget can shift the dominant bottleneck from genera-
tion to upstream reranking. This finding motivates us to focus on
the ranking-heavy scenario, where reducing downstream genera-
tion cost alone is insufficient. To address this problem, we propose
PACE, short forPrioritizedAdaptiveCoverage ofEvidence, to re-
lieve upstream reranking bottlenecks while preserving evidence
recall.PACEconsists of two components. First,evidence frontload-
ingreorders candidates so that the top-ranked candidates contain
evidence that is directly relevant to the query, complementary to
already selected documents, and useful for forming a complete ev-
idence chain. We formulate this as a marginal evidence coverage
problem: using fewer documents to cover more query semantics.
Each document’s coverage is weighted by its direct query rele-
vance and its potential to bridge evidence chains. This objective
is monotone submodular, giving greedy frontloading a (1−1/𝑒)
approximation guarantee. Second,pressure-adaptive budgeting
dynamically adjusts the reranking budget according to the relative
queueing pressure of the reranker and the LLM, reducing reranker
workload when upstream ranking becomes the bottleneck. Together,
the two components enable a key effect:less can be more when the
top-ranked candidates are evidence-dense. Although adaptive budget-
ing sends fewer candidates to the reranker, evidence frontloading
makes these candidates more useful and less noisy, increasing the
chance that the reranker promotes complete evidence into the final
top-𝐾context. Figure 1 summarizes how these two components
jointly improve RAG effectiveness and efficiency, leading to the
“less can be more” effect. Our contributions are as follows:
•We empirically identify shifting bottlenecks in RAG serving,
showing that the dominant bottleneck can move between
reranking and generation as the number of queries per
second and the reranking budget change.
•We proposePACE, a training-free framework that combines
marginal evidence frontloading with pressure-adaptive bud-
geting to reduce upstream reranking workload while pre-
serving evidence recall. We further show that the proposed
evidence coverage objective ofPACEis monotone submod-
ular, which leads to its greedy selection to provide a (1−1/𝑒)
approximation guarantee under a cardinality constraint.
•Experiments on three multi-hop QA datasets and online
serving simulations show thatPACEimproves evidence
recall, substantially reduces latency under ranking-heavy
workloads, and demonstrate thatless can be more when top-
ranked candidates are evidence-dense: achieve higher final
recall with a smaller reranking budget.
2RAG Bottlenecks Shift Across Configurations
and Loads
In a RAG system, the LLM contains most of the parameters and
computational complexity, so it is often assumed to be the main
serving bottleneck. However, the bottleneck is not always fixed
at generation. It can shift between upstream reranking and down-
stream generation under different model choices, configurations
(e.g., reranking budget 𝐷), and serving loads (e.g., queries per sec-
ond). We characterize this shift by comparing how long requests
wait in the reranker queue and the LLM queue: if requests wait
longer in the reranker queue than in the LLM queue, reranking is

Less can be More: Relieving RAG Bottlenecks
via Evidence Frontloading and Pressure-Adaptive Budgeting
0.50 0.75 1.00 1.25 1.50 1.75 2.00 2.25 2.50
Queries Per Second (QPS)4
2
02468P95 queue-time ratio (reranker / LLM)
Ranking-heavy
Generation-heavyD=200, K=5, natural EOS (max 128 tokens)
Generation model / ranking model: parameter ratio
Qwen-3B / DeBERT a: 7.1×
Qwen-7B / DeBERT a: 17.5×Qwen-3B / MiniLM-L12: 92.5×
Qwen-7B / MiniLM-L12: 228.3×
(a) Bottleneck tendency across reranker–LLM
model pairs under increasing QPS.
0.5 0.75 1 1.15 1.3 1.45 1.6 1.8
Queries per Second (QPS)100101102P95 latency (seconds)
D=100, K=5, 128 output tokens
Compression
w/o compressor
w/ compressor
Measured time
End-to-end latency
Reranker queue time
LLM queue time(b) Ranking-heavy workloads
0.5 0.75 1 1.15 1.3 1.45 1.6 1.8
Queries per Second (QPS)100101102P95 latency (seconds)
D=50, K=5, 128 output tokens
Compression
w/o compressor
w/ compressor
Measured time
End-to-end latency
Reranker queue time
LLM queue time (c) Generation-heavy workloads
Figure 2: Shifting RAG bottlenecks under different configurations and serving loads. Here, 𝐷denotes reranking budget and 𝐾de-
notes the number of documents sent to the LLM. In (a), we vary QPS across two rerankers: trecdl22-crossencoder-debertav3 , and
ms-marco-MiniLM-L12-v2 , and two LLMs: Qwen2.5-3B-Instruct , and Qwen2.5-7B-Instruct . We report log2
P95 reranker queue time
P95 LLM queue time
,
where positive values indicate ranking-heavy serving and negative values indicate generation-heavy serving. In (b) and (c) we
useQwen2.5-3B-Instructandtrecdl22-crossencoder-debertav3pair and vary𝐷with and without context compressor.
the bottleneck; otherwise, generation is the bottleneck. We report
p95 queueing latency, i.e., the latency below which 95% of requests
fall, because serving bottlenecks often appear as tail latency and
indicate system saturation under high load. This paper focuses on
the reranking and generation stages, and measures their queue-
ing delays to identify which stage limits system throughput. To
study RAG bottlenecks under realistic serving conditions, we use
an open-loop workload, where requests are issued according to a
fixed arrival process independent of previous request completion.
The detailed datasets, models, and serving setup are described in
Section 4.1; here we focus on the observed bottleneck behavior.
1. Model size affects bottleneck tendency, while load and rerank-
ing budget determine the actual bottleneck.Figure 2a shows
that, across different reranker–LLM model pairs, increasing queries
per second generally increases upstream reranking pressure. Rela-
tive model size affects the bottleneck, but high load can still make
reranking the bottleneck even with a much larger LLM. For ex-
ample, the parameter ratio between Qwen-3B and MiniLM-L12 is
92.5, and this pair behaves as generation-heavy under low QPS but
shifts to ranking-heavy when QPS reaches 2.5. Reranking budget
also changes the bottleneck. For the Qwen-3B and DeBERTa pair,
Figure 2b shows that with reranking budget 𝐷=100, end-to-end
latency becomes dominated by reranker queueing delay once QPS
reaches 1, whereas Figure 2c shows that with 𝐷= 50, latency
remains dominated by LLM queueing delay.
2. Relieving downstream bottlenecks cannot necessarily reduce
upstream reranking pressure.Figures 2b and 2c show that context
compression reduces LLM queueing delay by shortening the input
context. As a result, it improves end-to-end latency when genera-
tion is the dominant bottleneck. However, this gain becomes limited
when the system is ranking-heavy. In Figure 2b, once QPS reaches
1, most latency comes from the reranker queue. Although compres-
sion accelerates LLM processing, it does not reduce reranker queue-
ing delay; moreover, the compressor consumes additional compute,
which can further increase reranker pressure and queueing time.
As a result, its end-to-end benefit remains limited, especially under
high QPS and large𝐷.3PACE: Prioritized Adaptive Coverage of
Evidence
The observations in Section 2 show that RAG bottlenecks shift with
serving load and system configuration. To relieve shifting bottle-
necks, the system should adaptively allocate the reranking budget
𝐷rather than using a fixed value. However, simply reducing 𝐷may
miss necessary evidence and degrade answer quality. We therefore
proposePACE(Prioritized Adaptive Coverage of Evidence), to pace
the RAG system by addressing two requirements: (1) adapting the
reranking budget to system pressure, and (2) preserving evidence
recall by moving useful evidence into top-ranked candidates.
3.1 Evidence Frontloading
To preserve evidence recall under a small reranking budget, we first
frontload useful evidence to the top of the candidate ranking. Given
a query𝑞and a candidate document set C={𝑑𝑖}𝐷max
𝑖=1returned by
the dense retriever, where 𝐷maxdenotes the maximum of reranking
budget, e.g., top-100, our goal is to reorder Cso that the first few
documents cover as much necessary and non-redundant evidence
as possible. To make the top-ranked documents useful, each se-
lected document should satisfy two conditions. First, it should be
relevant to the query and cover part of the query’s information
need, such as a specific entity, event, or location. Second, it should
provide complementary evidence rather than repeat dimensions
that have already been covered by previously selected documents.
Thus, documents should be prioritized not only by their individual
query relevance, but also by their marginal contribution to the
uncovered evidence space. As more documents are selected, the
remaining uncovered dimensions become fewer, and the gain of
adding redundant documents naturally decreases. We formalize
this intuition as a marginal evidence coverage problem.
Marginal evidence covergae.Let 𝑞(𝑣)≥ 0and𝑑𝑖(𝑣)≥ 0denote
the representation values of query 𝑞and document 𝑑𝑖on semantic
dimension𝑣, respectively. The value on each dimension reflects the
amount of information carried by the query or document along
that dimension. For a selected document set 𝑆⊆C , we define the

Cai and Zafarani
coverage objective as
𝐹(𝑆)=∑︁
𝑣𝑞(𝑣)max
𝑑𝑖∈𝑆√𝜌𝑖𝑑𝑖(𝑣)
.(1)
where𝜌𝑖denotes the query-document relevance weight, which can
be instantiated by the dense retriever score. The current coverage
of dimension𝑣is
𝑚𝑆(𝑣)=max
𝑑𝑖∈𝑆√𝜌𝑖𝑑𝑖(𝑣)
.(2)
Thus, the marginal gain of adding document𝑑 𝑖to𝑆is
Δ(𝑑𝑖|𝑆)=∑︁
𝑣𝑞(𝑣)max√𝜌𝑖𝑑𝑖(𝑣)−𝑚𝑆(𝑣),0
.(3)
At each step, we greedily select the document with the largest
marginal gain:
𝑑★=arg max
𝑑𝑖∈C\𝑆Δ(𝑑𝑖|𝑆).(4)
This produces a ranking prefix that prioritizes relevant and com-
plementary evidence.
Soft-anchor relevance refinement.However, using only the dense
retriever score as 𝜌𝑖can miss supporting documents that are weakly
related to the query but strongly connected to other evidence docu-
ments. This is common in multi-hop QA, where one document may
provide an intermediate fact or entity bridge rather than directly
matching the query. To capture such dependencies, we refine the
relevance weight using soft anchors, i.e., candidate documents with
high query relevance that serve as evidence seeds. Documents sim-
ilar to these anchors receive higher weights, allowing the ranking
prefix to include not only directly query-relevant documents but
also bridge documents that connect multiple supporting evidence
into a more complete evidence chain. Together withmarginal evi-
dence coverage, this refined relevance weight enables the system to
frontload evidence that is both comprehensive and useful. We next
describe how to compute this refinement.
Let𝜌𝑖denote the initial query-document relevance score from
the dense retriever, and let sim(𝑑𝑖,𝑑𝑗)denote the similarity be-
tween two candidate documents. We first robustly standardize the
relevance scores using the median absolute deviation:
𝑧𝑖=𝜌𝑖−median(𝜌)
MAD(𝜌)+𝜖,(5)
where𝜖=10−12is used for numerical stability, and
MAD(𝜌)=median 𝑖|𝜌𝑖−median(𝜌)|.(6)
We then convert the standardized scores into soft-anchor weights:
𝑤𝑖=exp(𝑧𝑖)Í
𝑗exp(𝑧𝑗).(7)
Thus, candidates with higher query relevance receive larger weights
and act as soft evidence anchors. We estimate each document’s soft-
anchor relevance by aggregating its similarity to these anchors:
𝑎𝑗=Í
𝑖≠𝑗𝑤𝑖sim(𝑑𝑖,𝑑𝑗)
Í
𝑖≠𝑗𝑤𝑖.(8)
Finally, we combine the direct query-document relevance and the
anchor-based relevance as
˜𝜌𝑗=1−(1−𝑏 𝑗)(1− ¯𝑎𝑗),(9)where𝑏𝑗=norm(𝜌 𝑗),¯𝑎𝑗=norm(𝑎 𝑗), and norm(·) denotes min-
max normalization. The refined weight ˜𝜌𝑗is then used as the rel-
evance weight in Eq. 1, allowing the coverage objective to favor
documents that are directly relevant to the query while also recov-
ering documents connected to query-relevant evidence through
document-document dependencies. This refinement can be applied
either before or after reranking to improve recall; the only differ-
ence is the source of the query-document relevance score in Eq. 5.
Before reranking, 𝜌𝑖is obtained from the dense retriever, while
after reranking, it is obtained from the reranker. We evaluate this
combination of usage in Section 4.3, Figure 7.
Theoretical property.Beyond its empirical effectiveness, the pro-
posedmarginal evidence coverageobjective 𝐹(𝑆) is monotone sub-
modular, which gives greedy selection a standard approximation
guarantee to the optimal evidence coverage under a cardinality
constraint.
Theorem 3.1.Assume 𝑞(𝑣)≥ 0,˜𝜌𝑖≥0, and𝑑𝑖(𝑣)≥ 0for all𝑣
and𝑑𝑖. Then𝐹(𝑆)is monotone submodular.
Proof. First, we prove monotonicity. For any 𝐴⊆𝐵⊆C and
any dimension𝑣, we have
max
𝑑𝑖∈𝐴h√︁
˜𝜌𝑖𝑑𝑖(𝑣)i
≤max
𝑑𝑖∈𝐵h√︁
˜𝜌𝑖𝑑𝑖(𝑣)i
.(10)
Since𝑞(𝑣)≥0, summing over all dimensions gives
𝐹(𝐴)≤𝐹(𝐵).(11)
Therefore,𝐹(·)is monotone.
Next, we prove submodularity by showing diminishing returns.
For any𝐴⊆𝐵⊆Cand any𝑑 𝑗∉𝐵, define
𝑚𝐴(𝑣)=max
𝑑𝑖∈𝐴h√︁
˜𝜌𝑖𝑑𝑖(𝑣)i
, 𝑚𝐵(𝑣)=max
𝑑𝑖∈𝐵h√︁
˜𝜌𝑖𝑑𝑖(𝑣)i
.(12)
Because𝐴⊆𝐵, we have𝑚 𝐴(𝑣)≤𝑚𝐵(𝑣)for every𝑣. Therefore,
maxh√︁
˜𝜌𝑗𝑑𝑗(𝑣)−𝑚𝐴(𝑣),0i
≥maxh√︁
˜𝜌𝑗𝑑𝑗(𝑣)−𝑚𝐵(𝑣),0i
.(13)
Multiplying by𝑞(𝑣)≥0and summing over𝑣yields
Δ(𝑑𝑗|𝐴)≥Δ(𝑑 𝑗|𝐵).(14)
Thus,𝐹(·)is monotone submodular.
□
Corollary 3.2.Under a cardinality constraint |𝑆|≤𝐾 , let𝑆greedy
be the set selected by the greedy algorithm that iteratively adds the
document with the largest marginal gain. Then,
𝐹(𝑆 greedy)≥(1−1/𝑒)𝐹(𝑆★),(15)
where
𝑆★=arg max
|𝑆|≤𝐾𝐹(𝑆).(16)
is the optimal document set under the same constraint.
Proof. Since𝐹(𝑆) is non-negative, monotone, and submodular,
the result follows from the classical greedy approximation guar-
antee for monotone submodular maximization under a cardinality
constraint [19].□
This result shows that greedy evidence frontloading achieves at
least a(1−1/𝑒)approximation to the optimal evidence coverage,
while remaining efficient enough for reranking-time use.

Less can be More: Relieving RAG Bottlenecks
via Evidence Frontloading and Pressure-Adaptive Budgeting
Table 1: Notation for pressure-adaptive budgeting.
Symbol Meaning
𝑃𝑅 Number of unprocessed document pairs in the reranker queue
ˆ𝜇𝑅 Real-time estimated reranker throughput
𝑊𝑅 Estimated time to clear the reranker queue
|𝑄𝐿|Number of queries waiting in the LLM queue
𝐵𝐿 LLM batch size
𝐵𝑅 Reranker batch size
ˆ𝑇𝐿 Estimated execution time of one LLM batch
𝑇remain Remaining time of the current LLM batch
𝐸𝑅 Excess reranker workload compared with the LLM
𝐷min Minimum reranking budget
𝐷max Maximum reranking budget
𝑡now Current timestamp
𝑡start Start timestamp of the current LLM batch
3.2 Pressure-adaptive Budgeting
Evidence frontloading makes the ranking prefix more evidence-
dense, but the system still needs to decide how many candidates
should be processed by the reranker under changing serving pres-
sure. A fixed reranking budget 𝐷is suboptimal: a large 𝐷preserves
recall but can create a reranker bottleneck, while a small 𝐷reduces
latency but may miss necessary evidence. We therefore introduce
pressure-adaptive budgeting, which dynamically selects the largest
affordable reranking budget for each query based on the relative
pressure of the reranker and the LLM. Together with evidence
frontloading, this policy balances the pressure between the two
stages while preserving high evidence recall. Table 1 summarizes
the notation used in this section.
We estimate reranker pressure as the time needed to clear the
current reranker backlog:
𝑊𝑅=𝑃𝑅
ˆ𝜇𝑅.(17)
For the LLM, we estimate the remaining service time by combining
the current batch and the queued batches:
𝑇remain=max
ˆ𝑇𝐿−(𝑡 now−𝑡start),0
,(18)
𝑊𝐿=𝑇remain+|𝑄𝐿|
𝐵𝐿
ˆ𝑇𝐿.(19)
When𝑊𝑅≤𝑊𝐿, reranking is not more congested than generation,
so the system uses 𝐷max. When𝑊𝑅>𝑊𝐿, we estimate the excess
reranker backlog as
𝐸𝑅=ˆ𝜇𝑅(𝑊𝑅−𝑊𝐿).(20)
Given reranker batch size 𝐵𝑅, we reduce the reranking budget by
full reranker batches:
𝐷(𝑞)=max
𝐷min,𝐷max−𝐵𝑅𝐸𝑅
𝐵𝑅
.(21)
This rule keeps the largest possible reranking budget when rerank-
ing is not the bottleneck, and decreases 𝐷(𝑞) only when reranker
pressure exceeds LLM pressure. Together with evidence frontload-
ing, it reduces upstream workload under ranking-heavy conditions
while keeping useful evidence concentrated in the processed prefix.4 Experiments
We evaluatePACEfrom two perspectives: whether evidence front-
loading improves evidence recall among the top-ranked candidates,
and whether pressure-adaptive budgeting reduces end-to-end la-
tency under shifting bottlenecks while preserving recall. All ex-
periments are conducted on Quadro RTX 6000 GPUs with approxi-
mately 22 GB of available memory.
4.1 Experimental Settings
Model selection.Following prior work [ 6], we use splade-v3 [16]
as the retriever. It produces non-negative query and document
representations, satisfying the assumptions in Section 3.1. We use
trecdl22-crossencoder-debertav3 [9] as the reranker, Prov
ence [6] as the context compressor, and Qwen2.5-3B-Instru
ct[23] as the generator under a resource-limited serving setting.
All generation experiments use prefilling.
Datasets.We evaluate on the dev splits of three multi-hop QA
datasets with gold evidence annotations: HotpotQA [ 38] with 1,087
queries, MuSiQue [ 32] with 2,317 queries, and 2WikiMultiHop
QA [ 10] with 2,861 queries. For each dataset, we reserve an ad-
ditional 100 queries to tune baselines that require hyperparameter
calibration; these queries are excluded from evaluation. Our method
does not use this calibration set. For HotpotQA and 2WikiMulti-
HopQA, we retain queries whose complete supporting evidence
appears in the top-100 retrieved documents. This controls for initial
retrieval failures and lets us focus on how effectively each method
ranks supporting evidence toward earlier positions at different
document budgets. Following Provence [ 6], we use the Wikipedia
corpus preprocessing and passage segmentation provided through
BERGEN [ 24]. For MuSiQue, we use the official closed-context
candidate paragraphs; therefore, all queries are retained, the maxi-
mum reranking budget is 20, and no external Wikipedia retrieval
or additional passage segmentation is performed.
Metrics.To evaluateevidence frontloading, we measure evidence re-
call among top-ranked candidates. We reportcomplete evidence
recall@D, the percentage of queries whose top- 𝐷documents con-
tain all gold supporting evidence, andsupporting evidence re-
call@D, the average fraction of gold supporting evidence covered
by the top-𝐷documents. We evaluate document selection for 𝐷=
1,..., 100on HotpotQA and 2WikiMultiHopQA, and 𝐷= 1,..., 20
on MuSiQue, whose closed-context setting provides at most 20
candidate paragraphs per query. To evaluatepressure-adaptive bud-
geting, we reportp95 end-to-end latency,p95 reranker queue
time, andp95 LLM queue time. Here, p95 latency is the latency
below which 95% of requests complete.
Batch size calibration.To ensure that both reranker and LLM
achieve maximum processing speed on the local device, thereby
yielding reliable efficiency results. We profile reranker and LLM
on our hardware before the main experiments. For the reranker,
batch size denotes the number of query-document pairs scored
together, and we sweep 𝐵𝑟∈{2,4,8,16,32,64}. For the LLM, batch
size denotes the maximum number of concurrent generation se-
quences, and we sweep 𝐵𝑙∈{6,7,8,9,10,11,12}with a maximum
of 128 new tokens. Each candidate batch size is evaluated on the
same 100 randomly sampled TydiQA queries [ 7]. In this profiling

Cai and Zafarani
run, the reranker processes the top-50 retrieved documents and
returns the top-5 documents to the LLM. We measure throughput,
p95 latency, and GPU memory usage, repeat each run three times
with a fixed random seed. We select the smallest batch size whose
throughput is within 95% of the best observed throughput while
satisfying latency and memory constraints. The selected batch sizes,
𝐵𝑅=8for DeBERTa and 𝐵𝐿=10for Qwen2.5-3B, are fixed in all
main experiments so that bottleneck shifts are caused by workload
changes rather than per-setting batch-size tuning.
Online serving simulation.To study RAG bottlenecks under re-
alistic serving conditions, we use an open-loop workload, where
requests are issued according to a fixed arrival process independent
of previous request completion. Queries arrive following a Pois-
son process [25], with QPS varying from 0.5 to 2.5. We deploy the
reranker and the LLM on separate GPUs and connect them with
independent asynchronous queues to enable pipeline parallelism.
To examine whether downstream context reduction translates into
system-level efficiency gains, we insert Provence [ 6], a context
compressor, between the reranker and the LLM and run it on a
separate GPU. For analyzing RAG bottlenecks in Figure 2, we use
the first 60 seconds as warm-up and record per-stage latency over
the following 300 seconds. For the end-to-end evaluation ofPACE
in Figure 7 and 6, we set 𝐷min=20, and𝐷max=100, issue all eval-
uation queries according to the target QPS and continue running
until all queries are completed. This allows us to measure evidence
recall under dynamically selected 𝐷in an online setting. We fix the
query order and document order across workloads for fair compar-
ison. Generation uses natural end-of-sequence termination with a
maximum of 128 output tokens.
Use of AI tools.We used AI tools to assist with the implemen-
tation of experimental scripts for online serving simulation. All
generated code was reviewed, tested, and validated by the authors.
The AI tool was not used to generate datasets, labels or conclusions.
Baselines.To evaluate the effectiveness of evidence frontload-
ing, we compare with training-free reordering methods that do
not require additional model training. All methods operate on the
same candidate set returned by the retriever. For baselines with
tunable hyperparameters, we select the best configuration on the
100 query calibration set of each dataset and apply it unchanged to
the evaluation split.
•Standard Dense: It uses the original ranking returned by
the dense retriever without any reordering.
•Rocchio Pseudo-Relevance Feedback (PRF)[26]: It as-
sumes the top retrieved documents are pseudo-relevant,
updates the query representation by combining the original
query with these documents, and then reorders candidates
using the expanded query.
•The Maximal Marginal Relevance (MMR)[ 5]: It greed-
ily selects documents by balancing query relevance and
novelty, explicitly penalizing candidates that are similar to
previously selected documents.
•Dartboard[ 22]: It selects documents by maximizing rele-
vant information gain, encouraging the selected context to
contain information that is both useful for the query and
diverse from previously selected documents.•Adaptive-K[ 31]: It heuristically selects 𝐷by stopping
where query-document similarity has the largest drop be-
tween two consecutive ranked documents. Since 𝐷is de-
termined by the method rather than fixed externally, we
report the mean selected 𝐷and its corresponding recall as
a single point in the plots.
To further analyze the role of the relevance weight in Eq. 1, we
conduct ablation studies with three variants:
•w/o query & anchor relevance: removes the relevance
weight𝜌𝑖from the coverage objective:
𝐹(𝑆)=∑︁
𝑣𝑞(𝑣)max
𝑑𝑖∈𝑆[𝑑𝑖(𝑣)].(22)
•w/o query relevance: uses only soft-anchor relevance by
setting𝑏𝑗=0in Eq. 9.
•w/o anchor relevance: uses only direct query relevance
by setting ¯𝑎𝑗=0in Eq. 9.
4.2 Evidence Frontloading Improves Recall
We first evaluate the effectivenes ofPACEevidence frontloading
under a fixed budget of 𝐷= 100, focusing on whether it can move
supporting evidence into top-ranked candidates. Figure 3 and 4
report evidence recall at different reranking budgets. Across all
three datasets,PACEconsistently improves both complete evidence
recall@D and supporting evidence recall@D over training-free
baselines, especially at small 𝐷, which is the key regime for reducing
reranker workload. For example, on HotpotQA,PACEat 𝐷= 20
achieves comparable evidence recall to the best baseline at 𝐷= 40,
using only half of the reranking budget.
The baselines show that heuristic diversity does not necessarily
improve evidence coverage. MMR and Dartboard often underper-
form the original dense ranking because they penalize similar docu-
ments, while multi-hop supporting documents can be semantically
related and jointly necessary for answering. PRF can benefit from
expanding the query with top-ranked documents, but it is sensitive
to noisy initial results and may suffer from query drift. Adaptive-K
is shown as a single point because it selects 𝐷automatically; its
recall is bounded by the original dense ranking at the selected posi-
tion since it does not reorder documents. These results show that
PACEmoves necessary evidence into top-ranked candidates, rather
than only improving recall at large budgets.
Figure 5 further analyzes the relevance weight 𝜌in the marginal
evidence coverage objective in Eq. 1. Removing both query and
anchor relevance usually leads to the worst performance, show-
ing that pure coverage without relevance guidance cannot reliably
identify useful evidence. Using only query relevance prioritizes doc-
uments directly related to the question, but can miss evidence that
is weakly related to the query but instead connected to its one-hop
evidence. Using only anchor relevance can recover such evidence,
but may also promote irrelevant documents when the anchors are
noisy. As a result, using either source alone is generally suboptimal,
especially at small 𝐷, wherePACEconsistently achieves higher
recall. MuSiQue shows a slightly different pattern, where anchor
relevance is relatively more effective than query relevance. This
is likely because MuSiQue uses a closed-context setting with at
most 20 candidates, which reduces noise in anchor estimation, and
its questions are constructed by composing single-hop questions

Less can be More: Relieving RAG Bottlenecks
via Evidence Frontloading and Pressure-Adaptive Budgeting
20 40 60 80 100
D0.00.20.40.60.81.0Complete Evidence Recall
HotpotQA
2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0
D
MuSiQue
20 40 60 80 100
D
2WikiMultiHopQAStandard dense Rocchio PRF MMR Dartboard Ours (PACE) Adaptive-K
Figure 3: Complete evidence recall@D across three multi-hop QA datasets.PACEtends to cover all required evidence with
smaller reranking budgets than training-free baselines.
20 40 60 80 100
D0.00.20.40.60.81.0Supporting Evidence Recall
HotpotQA
2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0
D
MuSiQue
20 40 60 80 100
D
2WikiMultiHopQAStandard dense Rocchio PRF MMR Dartboard Ours (PACE) Adaptive-K
Figure 4: Supporting evidence recall@D across three multi-hop QA datasets.PACEretrieves a larger fraction of supporting
evidence with small reranking budgets.
20 40 60 80 100
D0.00.20.40.60.81.0Complete Evidence Recall
HotpotQA
2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0
D
MuSiQue
20 40 60 80 100
D
2WikiMultiHopQAw/o query & anchor relevance w/o anchor relevance w/o query relevance Ours (PACE)
Figure 5: Ablation study of the marginal evidence coverage objective in Eq. 1. Removing query relevance, anchor relevance, or
both weakens complete evidence recall, showing that direct query relevance and soft-anchor document dependencies jointly
contribute to effective evidence frontloading.
through bridge entities. This also explains why PRF performs rela-
tively better on MuSiQue than on the open-domain datasets, since
its pseudo-relevance feedback is less affected by noisy top-ranked
documents in the closed-context candidate set. Overall, combining
query and anchor relevance gives the strongest and most stable
evidence frontloading across datasets.
4.3PACEImproves Efficiency while Preserving
Recall
The static offline results in Section 4.2 show that evidence front-
loading improves recall across different cutoffs 𝐷. We now evaluatewhetherPACEcan translate this advantage into end-to-end latency
gains under online serving, while preserving evidence recall after
dynamically reducing 𝐷. We conduct online serving experiments
on HotpotQA and report latency in Figure 6 and recall in Figures 7.
Pressure-adaptive budget selection improves system latency.Figure 6
comparesPACEwith a fixed 𝐷= 100reranking budget under
ranking-heavy workloads. As QPS increases, the fixed- 𝐷= 100
baseline quickly accumulates reranker queueing delay, causing p95
end-to-end latency to grow sharply. In contrast,PACEdynamically
reduces the reranking budget according to system pressure, which

Cai and Zafarani
0.6 0.8 1.0 1.2 1.4 1.6 1.8
QPS101102Seconds
P95 end-to-end latency
0.6 0.8 1.0 1.2 1.4 1.6 1.8
QPS100101102Seconds
P95 reranker queue time
0.6 0.8 1.0 1.2 1.4 1.6 1.8
QPS234567Seconds
P95 LLM queue timeFixed D=100 Ours (PACE)
Figure 6: Latency comparison under ranking-heavy workloads.PACEsubstantially reduces p95 end-to-end latency by lowering
reranker queueing time with a smaller adaptive reranking budget. Although LLM queueing time may increase as more requests
pass through the reranker, but the dominant upstream bottleneck is greatly relieved. The fixed baseline uses 𝐷= 100throughout.
0.6 0.8 1.0 1.2 1.4 1.6 1.8
QPS0.750.800.850.900.951.00Recall
Complete Evidence Recall@D
0.6 0.8 1.0 1.2 1.4 1.6 1.8
QPS0.840.860.880.900.920.940.960.981.00Recall
Supporting Evidence Recall@D
0.6 0.8 1.0 1.2 1.4 1.6 1.8
QPS0.350.400.450.500.550.60Recall
Complete Evidence Recall@5
0.6 0.8 1.0 1.2 1.4 1.6 1.8
QPS0.6250.6500.6750.7000.7250.7500.7750.800Recall
Supporting Evidence Recall@5Fixed D=100
DenseRocchio PRF
MMRDartboard
w/o query & anchor relevancew/o anchor relevance
w/o query relevanceOurs (PACE)
Figure 7: Evidence recall under pressure-adaptive budgeting on HotpotQA. The first two subfigures report recall@D before
reranking, and the last two report recall@5 after reranking, where 𝐷is dynamically selected and 𝐾= 5documents are sent to
the LLM.Less can be more when the top-ranked candidates are evidence-dense: as QPS increases,PACEreduces the reranking
budget but maintains higher complete and supporting evidence recall than baselines and ablation variants.
0.6 0.8 1.0 1.2 1.4 1.6 1.8
QPS5060708090100Number of documents
Selected DFixed D=100
Dense
Rocchio PRFMMR
Dartboard
w/o query & anchor relevancew/o anchor relevance
w/o query relevance
Ours (PACE)
Figure 8: Average selected reranking budget under increasing
QPS.PACEreduces 𝐷as serving pressure grows, but main-
tains higher evidence recall than other methods at compara-
ble selected budgets.
substantially lowers reranker queueing time and stabilizes end-to-
end latency. Although LLM queue time can increase slightly because
more requests pass through the reranker, the dominant upstream
bottleneck is relieved, leading to much lower overall latency.
PACEpreserves recall under smaller adaptive budget.To fairly com-
pare methods in the online serving setting, we apply the same
pressure-adaptive budgeting policy to all reordering methods. As aresult, all adaptive methods operate with similar latency and compa-
rable selected reranking budgets, since reordering a small candidate
set is negligible compared with total serving latency; for example,
PACEselection accounts for less than 0.007% of p95 end-to-end
latency. This allows us to focus on how evidence recall changes
under increasing serving load. Figure 8 shows that as QPS increases,
PACEgradually reduces 𝐷, similar to other adaptive methods. At
QPS=1.8, most adaptive methods select a budget around50–60
documents to relieve reranker pressure.
Figure 7 reports evidence recall at two stages: recall@D before
reranking, and recall@K after reranking, where we use the common
setting𝐾= 5[6]. Before reranking, the fixed 𝐷= 100baseline main-
tains recall@D1 .0because no candidates are dropped. All adaptive
methods show lower recall@D as QPS increases because their se-
lected𝐷decreases. However,PACEmaintains higher complete and
supporting evidence recall than baselines and ablation variants at
comparable budgets. This shows that its latency reduction does
not come from simply dropping candidates, but from combining
adaptive budgeting to an evidence-dense candidate ranking.
After reranking,PACEshows thatless can be more when the
top-ranked candidates are evidence-dense: it achieves the highest re-
call@5 among the documents sent to the LLM, even surpassing the
fixed𝐷= 100, and remains stable as QPS increases. This suggests
that a larger reranking budget does not always improve final top- 𝐾
evidence recall, since additional candidates may introduce noise.

Less can be More: Relieving RAG Bottlenecks
via Evidence Frontloading and Pressure-Adaptive Budgeting
Table 2:Less can be more. Complete evidence recall under
fixed reranking budgets. Fixed 𝐷= 100denotes the full-
budget baseline, and 𝑅@𝐾with𝐾= 5measures final evidence
recall after reranking. Values in parentheses indicate abso-
lute percentage-point changes in 𝑅@𝐾over the full-budget
baseline on the same dataset.
Dataset Method𝐷Comp. R@𝐷Comp. R@𝐾
HotpotQAStandard Dense 100 100 41.13
Dartboard80 95.31 40.48(−0.65)
50 83.17 38.18(−2.95)
20 52.99 33.22(−7.91)
PACE80 97.2560.81(+19.68)
50 91.3662.38(+21.25)
20 77.1060.26(+19.13)
2WikiStandard Dense 100 100 48.76
Dartboard80 95.15 48.10(−0.66)
50 81.97 47.40(−1.36)
20 53.41 39.96(−8.80)
PACE80 97.2454.22(+5.46)
50 91.0654.39(+5.63)
20 75.1953.94(+5.18)
By frontloading useful evidence before reranking and reducing 𝐷
under pressure,PACEprovides the reranker with a smaller but more
evidence-dense candidate set, making useful evidence more likely
to appear in the final context. For example, at QPS =1.8,PACEuses
nearly half the reranking budget of fixed 𝐷= 100, but achieves
about 20% higher recall@5. In contrast, most baselines fall below
fixed𝐷=100after reranking, and their recall generally decreases
as QPS increases because the adaptive budget becomes smaller. We
further validate this effect under different fixed reranking budgets
on two datasets in Table 2:PACEachieves higher final recall at
𝐾= 5than the full-budget baseline even with substantially smaller
𝐷, showing that its gains come from evidence frontloading rather
than simply preserving more candidates.
5 Related Work
Relieving generation-heavy bottlenecks.Existing work on effi-
cient RAG often reduces the cost of downstream LLM generation.
Serving systems improve throughput and latency through opti-
mized batching, scheduling, KV-cache management, and prefill
execution [ 1,15,39,41]. METIS [ 25] adapts query-level configura-
tions, such as retrieved chunks and synthesis methods, to balance
response quality and latency. Context compression methods, such
as RECOMP [ 37], LLMLingua [ 11], and Provence [ 6], shorten re-
trieved contexts by removing or summarizing less useful tokens.
These methods reduce generation-side cost, but they do not directly
address shifting bottlenecks between reranking and generation or
relieve reranker-side pressure while preserving evidence recall.
Relieving ranking-heavy bottlenecks.Rerankers improve re-
trieval quality but become expensive when many query-document
pairs must be scored [ 2,8,40]. Prior work mainly reduces this
cost by accelerating the reranker itself. Early-exit methods make
decisions at intermediate Transformer layers using auxiliary clas-
sifiers [ 34] or layer-wise query-document similarity estimates [ 4].Adaptive-K [ 31] selects the budget at the largest drop in query-
document similarity. These methods reduce reranking computation
or choose a heuristic budget point, but they do not explicitly adapt
the reranking budget to real-time reranker–LLM pressure.PACE
is complementary: it keeps the reranker unchanged, but controls
which candidates enter reranking and how many are processed
under online serving pressure.
Improving evidence recall.Evidence recall is often improved
through multi-hop retrieval or diversity-aware ranking. Multi-hop
retrievers acquire supporting evidence through iterative retrieval
or reasoning-guided search: MDR [ 36] recursively conditions later
retrieval steps on previously retrieved evidence, Baleen [ 13] uses
condensed intermediate evidence to guide subsequent hops, and
IRCoT [ 33] interleaves LLM reasoning with retrieval. These meth-
ods target evidence acquisition, but require specialized retriever
training or additional LLM-retrieval iterations, which can increase
serving cost. Diversity-aware methods, which improves recall by
selecting non-redundant documents from a candidate set. MMR [ 5]
balances query relevance with novelty, xQuAD [ 27] promotes cov-
erage of different query aspects, and Dartboard [ 22] maximizes
relevant information gain for diverse RAG retrieval. These methods
are closer to our setting because they reorder or select from existing
candidates. However, they typically rely on coarse document-level
distances or manually designed trade-offs, while multi-hop QA may
require preserving documents that are similar but jointly necessary.
PACEdiffers bymeasuring complementarity over fine-grained se-
mantic dimensions,supporting both diversity and evidence chaining
without additional training or tuned parameters, andproviding mono-
tone submodular objective with a (1−1/𝑒)approximation guarantee.
6 Conclusion
In this paper, we study RAG as an end-to-end serving system and
show that its bottleneck is not fixed at LLM generation. Instead, the
dominant bottleneck can shift between reranking and generation
as query arrival rates and reranking budgets change. Motivated by
this observation, we proposePACE, a training-free framework that
relieves upstream reranking bottlenecks while preserving evidence
recall.PACEfrontloads useful evidence into the top of the candi-
date ranking through a monotone submodular marginal coverage
objective, and then adaptively selects the reranking budget based
on real-time reranker and LLM pressure. Experiments on multi-
hop QA datasets and online serving simulations demonstrate that
PACEimproves evidence recall, substantially reduces latency under
ranking-heavy workloads, and shows that less can be more when
the top-ranked candidates are evidence-dense: a smaller reranking
budget can lead to higher final evidence recall.
References
[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra,
Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming
{Throughput-Latency }tradeoff in{LLM}inference with{Sarathi-Serve}. In
18th USENIX symposium on operating systems design and implementation (OSDI
24). 117–134.
[2]Yuwei An, Yihua Cheng, Seo Jin Park, and Junchen Jiang. 2025. Hyperrag:
Enhancing quality-efficiency tradeoffs in retrieval-augmented generation with
reranker kv-cache reuse.arXiv preprint arXiv:2504.02921(2025).
[3] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bog-
dan Damoc, Aidan Clark, et al .2022. Improving language models by retrieving

Cai and Zafarani
from trillions of tokens. InInternational conference on machine learning. PMLR,
2206–2240.
[4] Francesco Busolin, Claudio Lucchese, Franco Maria Nardini, Salvatore Orlando,
Raffaele Perego, Salvatore Trani, and Alberto Veneri. 2025. Efficient re-ranking
with cross-encoders via early exit. InProceedings of the 48th International ACM
SIGIR Conference on Research and Development in Information Retrieval. 2534–
2544.
[5] Jaime G Carbonell and Jade Goldstein. 1998. The use of MMR, diversity-based
reranking for reordering documents and producing summaries.. InSIGIR, Vol. 98.
290941–291025.
[6] Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, and Stéphane Clin-
chant. 2025. Provence: efficient and robust context pruning for retrieval-
augmented generation. InThe Thirteenth International Conference on Learning
Representations. https://openreview.net/forum?id=TDy5Ih78b4
[7] Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski,
Vitaly Nikolaev, and Jennimaria Palomaki. 2020. TyDi QA: A Benchmark for
Information-Seeking Question Answering in Typologically Diverse Languages.
Transactions of the Association for Computational Linguistics(2020).
[8] Hervé Déjean and Stéphane Clinchant. 2026. Efficient Listwise Reranking with
Compressed Document Representations.arXiv preprint arXiv:2604.26483(2026).
[9]Hervé Déjean, Stéphane Clinchant, and Thibault Formal. 2024. A thorough
comparison of cross-encoders and LLMs for reranking SPLADE.arXiv preprint
arXiv:2403.10407(2024).
[10] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational Lin-
guistics. International Committee on Computational Linguistics, Barcelona, Spain
(Online), 6609–6625. https://www.aclweb.org/anthology/2020.coling-main.580
[11] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023.
Llmlingua: Compressing prompts for accelerated inference of large language
models. InProceedings of the 2023 conference on empirical methods in natural
language processing. 13358–13376.
[12] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. InProceedings of the 2020 conference on empirical
methods in natural language processing (EMNLP). 6769–6781.
[13] Omar Khattab, Christopher Potts, and Matei Zaharia. 2021. Baleen: Robust multi-
hop reasoning at scale via condensed retrieval.Advances in Neural Information
Processing Systems34 (2021), 27670–27682.
[14] Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage
search via contextualized late interaction over bert. InProceedings of the 43rd
International ACM SIGIR conference on research and development in Information
Retrieval. 39–48.
[15] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serving with pagedattention. In
Proceedings of the 29th symposium on operating systems principles. 611–626.
[16] Carlos Lassance, Hervé Déjean, Thibault Formal, and Stéphane Clinchant. 2024.
SPLADE-v3: New baselines for SPLADE.arXiv preprint arXiv:2403.06789(2024).
[17] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp
tasks.Advances in neural information processing systems33 (2020), 9459–9474.
[18] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua,
Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models
use long contexts.Transactions of the association for computational linguistics12
(2024), 157–173.
[19] George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. 1978. An analy-
sis of approximations for maximizing submodular set functions—I.Mathematical
programming14, 1 (1978), 265–294.
[20] Rodrigo Nogueira and Kyunghyun Cho. 2019. Passage Re-ranking with BERT.
arXiv preprint arXiv:1901.04085(2019).
[21] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. 2020. Docu-
ment ranking with a pretrained sequence-to-sequence model. InFindings of the
association for computational linguistics: EMNLP 2020. 708–718.
[22] Marc Pickett, Jeremy Hartman, Ayan Kumar Bhowmick, Raquib-ul Alam, and
Aditya Vempaty. 2024. Better rag using relevant information gain.arXiv preprint
arXiv:2407.12101(2024).
[23] A Yang Qwen, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen
Yu, Chengpeng Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al .2025. Qwen2. 5
technical report.arXiv preprint arXiv:2412.15115(2025), 2412–15115.
[24] David Rau, Hervé Déjean, Nadezhda Chirkova, Thibault Formal, Shuai Wang,
Stéphane Clinchant, and Vassilina Nikoulina. 2024. BERGEN: A benchmarking
library for retrieval-augmented generation. InFindings of the Association for
Computational Linguistics: EMNLP 2024. 7640–7663.
[25] Siddhant Ray, Rui Pan, Zhuohan Gu, Kuntai Du, Shaoting Feng, Ganesh Anan-
thanarayanan, Ravi Netravali, and Junchen Jiang. 2025. Metis: fast quality-aware
rag systems with configuration adaptation. InProceedings of the ACM SIGOPS31st symposium on operating systems principles. 606–622.
[26] Joseph John Rocchio Jr. 1971. Relevance feedback in information retrieval.The
SMART retrieval system: experiments in automatic document processing(1971).
[27] Rodrygo LT Santos, Jie Peng, Craig Macdonald, and Iadh Ounis. 2010. Explicit
search result diversification through sub-queries. InEuropean conference on
information retrieval. Springer, 87–99.
[28] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H
Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language models can be
easily distracted by irrelevant context. InInternational Conference on Machine
Learning. PMLR, 31210–31227.
[29] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021.
Retrieval augmentation reduces hallucination in conversation. InFindings of the
Association for Computational Linguistics: EMNLP 2021. 3784–3803.
[30] Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin
Chen, Dawei Yin, and Zhaochun Ren. 2023. Is ChatGPT good at search? inves-
tigating large language models as re-ranking agents. InProceedings of the 2023
conference on empirical methods in natural language processing. 14918–14937.
[31] Chihiro Taguchi, Seiji Maekawa, and Nikita Bhutani. 2025. Efficient Context
Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k. In
Proceedings of the 2025 Conference on Empirical Methods in Natural Language
Processing. 20116–20141.
[32] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabhar-
wal. 2022. musique: Multihop questions via single-hop question composition.
Transactions of the Association for Computational Linguistics10 (2022), 539–554.
[33] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving retrieval with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of the 61st annual meeting of the
association for computational linguistics (volume 1: long papers). 10014–10037.
[34] Ji Xin, Rodrigo Nogueira, Yaoliang Yu, and Jimmy Lin. 2020. Early exiting BERT
for efficient document ranking. InProceedings of SustaiNLP: Workshop on Simple
and Efficient Natural Language Processing. 83–88.
[35] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett,
Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor nega-
tive contrastive learning for dense text retrieval.arXiv preprint arXiv:2007.00808
(2020).
[36] Wenhan Xiong, Xiang Lorraine Li, Srini Iyer, Jingfei Du, Patrick Lewis,
William Yang Wang, Yashar Mehdad, Wen-tau Yih, Sebastian Riedel, Douwe
Kiela, et al .2020. Answering complex open-domain questions with multi-hop
dense retrieval.arXiv preprint arXiv:2009.12756(2020).
[37] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2023. Recomp: Improving retrieval-
augmented lms with compression and selective augmentation.arXiv preprint
arXiv:2310.04408(2023).
[38] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering. InProceedings of the 2018
conference on empirical methods in natural language processing. 2369–2380.
[39] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-
Gon Chun. 2022. Orca: A distributed serving system for {Transformer-Based}
generative models. In16th USENIX symposium on operating systems design and
implementation (OSDI 22). 521–538.
[40] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Moham-
mad Shoeybi, and Bryan Catanzaro. 2024. Rankrag: Unifying context ranking
with retrieval-augmented generation in llms.Advances in Neural Information
Processing Systems37 (2024), 121156–121184.
[41] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H
Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al .2024.
Sglang: Efficient execution of structured language model programs.Advances in
neural information processing systems37 (2024), 62557–62583.