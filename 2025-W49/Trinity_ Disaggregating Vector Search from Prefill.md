# Trinity: Disaggregating Vector Search from Prefill-Decode Disaggregation in LLM Serving

**Authors**: Yi Liu, Chen Qian

**Published**: 2025-12-01 23:53:42

**PDF URL**: [https://arxiv.org/pdf/2512.02281v1](https://arxiv.org/pdf/2512.02281v1)

## Abstract
Prefill and decode (PD) disaggregation separates prompt prefill and token-by-token decode stages into distinct GPU pools and has become the dominant architecture for large-scale LLM serving in industry. Also, retrieval tasks via vector search remains entangled with the model inference process, like heterogeneous RAG requests and prompt answer caches, inflating tail latency. We are motivated to investigate how vector search should be orchestrated along with PD disaggregation with a dedicated deployment architecture without violating SLOs in various retrieval workloads. We present Trinity, a practical framework that consolidates all retrieval into a single, shared vector-search GPU pool and make it work with PD disaggregated LLM serving in match. Trinity introduces (1) a novel architecture for deploying GPU-based vector search service in PD disaggregation. (2) Continuous batching for vector search that make full used of GPUs under heterogeneous queries; (3) Stage-aware scheduling that preempts vector search requests between both decode and prefill tasks.

## Full Text


<!-- PDF content starts -->

Trinity:Disaggregating Vector Search for Prefill-Decode
Disaggregation in LLM Serving
Yi Liu, Chen Qian
University of California Santa Cruz
Abstract
Prefill and decode (PD) disaggregation separates prompt
prefill and token-by-token decode stages into distinct GPU
pools and has become the dominant architecture for large-
scale LLM serving in industry. Also, retrieval tasks via vector
search remains entangled with the model inference process,
like heterogeneous RAG requests and prompt answer caches,
inflating tail latency. We are motivated to investigate how
vector search should be orchestrated along with PD disag-
gregation with a dedicated deployment architecture without
violating SLOs in various retrieval workloads. We present
Trinity, a practical framework that consolidates all retrieval
into a single, shared vector-search GPU pool and make it
work with PD disaggregated LLM serving in match.Trinity
introduces (1) a novel architecture for deploying GPU-based
vector search service in PD disaggregation. (2) Continuous
batching for vector search that make full used of GPUs under
heterogeneous queries; (3) Stage-aware scheduling that pre-
empts vector search requests between both decode and prefill
tasks. Our analysis demonstrates that an independent vector
search pool can serve diverse vector retrievals while sustain-
ing high throughput and low tail latency for LLM serving
with PD disaggregation.
1 Introduction
Retrieval-augmented generation (RAG) mitigates hallucina-
tions and enhances output quality by retrieving semantically
relevant document chunks during inference [2, 8, 9, 11, 12,
16, 20, 23]. Vector search [3, 7, 19, 22] serves as the founda-
tion of RAG services and prompt cache (e.g., GPTCache [4]),
enabling efficient nearest neighbor search from large-scale
embeddings. Since exact kNN search is computationally
prohibitive at scale, production systems adopt approximate
nearest-neighbor (ANN) indexes, such as IVF/IMI with PQ or
OPQ compression, or graph-based structures like HNSW, to
trade slight recall loss for substantial gains in throughput and
memory efficiency. To further accelerate large-scale vector
retrieval, recent GPU-based ANN schemes [6, 8, 18, 21, 24]
GPU 
Utilization
Batch SizePrefill
ANNS (CAGRA)
DecodeFigure 1: Roofline of Vector Search, Prefill and Decode in
LLM Serving.
are exploited to batch distance computations, leverage high-
bandwidth HBM, and overlap data transfer, search, and re-
ranking. These optimizations enable sub-millisecond query
latency even under high-throughput workloads. For example,
CAGRA [18], a GPU-optimized graph ANN implementation
in cuVS, adopts fixed-degree graphs and warp-efficient traver-
sal to sustain high recall while maintaining low tail latency
with batched execution.
PD disaggregation [1, 10, 13, 14, 17, 25, 26] has emerged
as a state-of-the-art paradigm for LLM serving in industry.
Traditional monolithic deployments execute both stages on
the same GPU, resulting in inefficient resource utilization
since prefill and decode exhibit fundamentally different char-
acteristics: theprefillstage is compute-intensive and benefits
from high parallelism across long input sequences, while the
decodestage is memory-bound for generating tokens sequen-
tially with billion-scale parameters of experts [1, 5, 15] and
MLP. By decoupling these stages into distinct GPU or node
pools, PD disaggregation enables independent scaling, higher
hardware utilization, and lower latency, and it has become
central to modern LLM infrastructures with low cost.
While numerous studies [9, 10, 18, 23] have explored how
to integrate vector search into existing LLM inference, the
interaction between GPU-based vector retrieval and current
PD disaggregation remains largely underexplored. In a dis-
aggregated serving architecture, the prefill stage typically
performs a single retrieval to initialize context at the begin-
ning, whereas the decode stage may require multiple rounds
of retrieval as generation progresses to maintain topical rele-
1arXiv:2512.02281v1  [cs.DB]  1 Dec 2025

LLM GPU Vector Search GPU KVCache  Transfer
Prefill Instances
Each node maintains the same vector data and 
graph index, and one GPU is used for vector search.GPUs
Data parallelism 
for vectors
Decode Instances
Same as Prefill instancesGPUs
Data parallelism 
for vectors(a) Vector search GPU is coupled
with LLM GPU in the same servers.
Vector Search 
Instances
GPUs
(e.g., CAGRA, Milvus)Decode Instances
Same as Prefill instancesGPUs
 Retrieval reqs 
and resultsPrefill Instances
Each node maintains the same vectordata .GPUs
Data parallelism 
for vectors(b) Vector search GPU is decoupled from Decode
instances, but being together with Prefill Instance.
Vector Search  
Instances
GPUs
(e.g., CAGRA, Milvus)Prefill Instances
All GPUs are used for LLM serving.GPUs
Decode Instances
Same as Prefill instancesGPUs
 Retrieval  reqs 
and result s(c) Vector search is offloaded to an independent
GPU pool.
Figure 2. Architectures of serving GPU-based vector search instances in PD disaggregation.
vance or inject new knowledge. This asymmetry introduces
unique challenges in coordinating retrieval frequency, manag-
ing cross-stage communication, and optimizing end-to-end
latency across disaggregated resources.
In this work, we explore several GPU disaggregation ar-
chitectures for incorporating vector search service into large-
scale LLM serving systems and analyze how different designs
coordinate these GPUs efficiently. Building on these insights,
we designTrinity, a disaggregated serving framework with
three key innovations. (1) To reduce document retrieval la-
tency, we explore and analysis three potential architectures
for running vector search either coupled with P/D GPUs or
decoupled into shared GPU pools under PD disaggregation.
(2) To improve GPU utilization for vector search, we intro-
duce a continuous batching mechanism that executes graph-
based vector search efficiently on GPUs with a dynamically
running batch of concurrent queries. (3) To guarantee SLOs
for both prefill and decode instances (e.g., time-to-first-token
(TTFT)),Trinityemploys an adaptive scheduling and pre-
emption scheme using multiple priority queues to balance
latency and throughput across serving stages.
2 Motivation
In this section, we contrast the computation and storage need
for vector search, prefill, and decode, and show that each has
a different optimal batch size. We observe that both decode
and vector search are primarilymemory-bound: greedy graph
traversal repeatedly loads neighbor vectors and indices at
each traversal step, resulting in low Arithmetic Intensity (AI).
Then, we analyze the GPU utilization roofline of each stage
to identify their plateaus and the batch sizes at which they
saturate, as shown in Fig. 1.
Letu(·)∈[0,1] be GPU utilization. Let Ppeakbe peak com-
pute (FLOP/s) for the chosen datatype, Bmemthe effective
memory bandwidth (B/s), and AIthe arithmetic intensity
(FLOP/Byte). Let Bbe batch size (sequences) for LLM. ForANN search let Qbe the number of concurrent queries. Define
the generic roofline as:
umax≜min
1,AI·B mem
Ppeak
.
We model the pre-plateau rise with a saturation scale Xsatand
an optional sublinearity exponentα∈(0,1]:
u(X)≈min
umax, X
Xsatα
,X∈ {B,Q}.
The pre-plateau slope at smallXisdu
dX
X≪X sat≈α/X sat.
(1) Prefill.Large GEMMs operations means high AI, which
indicates it is compute-bound.
uprefill(B)≈min
1, B
Bsat,pαp
,umax
prefill≈1.
Bsat,pis the smallest batch size at which the prefill phase
(big GEMMs) effectively saturates the GPU. Beyond this
point, increasing Byields negligible utilization gains. Thus,
its plateau can reach≈100% once tensor cores are filled.
(2) Decode.One new token per request, multi-layer KV-
Cache reads will be incurred means it is low AI(memory-
bound).
udecode(B)≈min
umax
decode , B
Bsat,dαd
,umax
decode≈AIdec·Bmem
Ppeak.
Plateau:flat, bandwidth-limited (often well below 100%).
(3) GPU ANN (e.g., CAGRA in cuVS).This phase is dom-
inated by graph traversal and distance computations, because
per-vector bytes dominate, AI is low. Thus, it is memory-
bound, even with only a small compute-leaning re-rank stage
when queries are batched.
2

ucagra(Q)≈min
umax
cagra, Q
Qsatαa
,umax
cagra≈AIgraph·Bmem
Ppeak.
Plateau:bandwidth roof, which is similar order to decode at
equal precision.
3 Methodology
3.1 Architecture
From the motivation, we know that vector search and the
prefill/decode phases have different optimal batch sizes, sug-
gesting they should run on separate GPUs. To reduce the
retrieval latency, the most intuitive way is co-locate a vector
search GPU on the P/D servers, thus, fast intra-node links
(e.g., NVLink) can reduce the data transfer latency for re-
trieval. The trade-off is that P/D server has to reserve a GPU
for vector search, which reduces available computation ca-
pacity for LLM inference and can introduce bandwidth con-
tention. In the following, we discuss co-location by trading
off SLO improvements on vector search against lost inference
throughput and contention to see when co-location is worthy.
Coupled placement of vector search and LLM GPUs.
Fig. 2a shows a design where each prefill and decode server
co-locates one vector-search GPU with the LLM GPUs. The
vector database (embeddings and graph index) is replicated
per node and sharded across the local vector GPU via Data
Parallelism (DP). At query time, both prefill and decode work-
ers send retrievals to the in-node vector GPU over NVLink,
minimizing retrieval latency and avoiding network hops. Also,
the per-node replication of the graph/index inflates memory
footprint and complicates updates, we can also make several
LLM nodes to share one vector search GPU.
However, in the decode stage we typically use expert paral-
lelism (EP): each GPU stores parameters for a subset of MLP
experts and participates in dispatch/combine kernels. If one
GPU on a node is reserved for vector search, an additional
GPU must be provisioned elsewhere to host the displaced
experts, pushing part of the EP traffic inter-node. The extra
network latency during dispatch/combine can outweigh the
latency saved by in-node retrieval.
Decoupled vector search from Decode but co-located with
Prefill.The Fig. 2b depicts a layout where vector search
GPUs are running in a separated in a independent pool only
for serving retrieval requests from decode instances, but some
of them are co-located with the prefill servers with DP. Pre-
fill sends retrieval requests to the in-node vector GPUs over
NVLink/PCIe, while decode reaches the vector pool over the
network.
Co-locating vector search with prefill removes bandwidth
contention between decode and vector search , and gives pre-
fill the shortest retrieval path with intra-node fabric, reducing
retrieval latency for prefill instances. However, even though
prefill is compute-bound, tensor parallelism (TP) introduces
Waiting queueReqs
Document store 
(e.g., docstore  pickle)GPU-based 
vector search
Vector data
Neighbor list ExtendFor each extend  iteration:
(1) SelectParents
(2) ExpandNeighbors
(3) MarkVisited
(4) Distance Computation
(5) MergeTopM
R1-E1R1-E2R1-E3R1-E4R6-E1R6-E2 …
R2-E1R2-E2R2-E3R5-E1R5-E2R7-E1 …
R3-E1R3-E2R4-E1R4-E2R4-E3R5-E1 …HBM
Fixed degree graph
Continuous Batching with batch size as 3.Ri-Ej refers to the j -th extend 
in graph for Request i.ExtendFigure 3: Continuous batching for vector search inTrinity.
collective synchronization that sits on the critical path. In par-
ticular, the Q/K/V projections and subsequent attention blocks
require all-reduces to merge partial GEMM results across
TP shards. The saved microseconds from in-node document
retrieval are typically smaller than the added or unchanged
collective latency for merging these GEMMs, especially at
higher TP degrees. Consequently, the fast retrieval path can-
not compensate for the step-time dominated by TP collectives,
so this architecture gains are limited despite co-location with
prefill.
Offloading vector search to an independent GPU pool.
Based on the preceding analysis of rooflines and saturation
scales, we derive the design shown in Fig. 2c, vector search is
offloaded to a separate GPU pool, while all GPUs on prefill
and decode servers are dedicated to LLM serving. Retrieval
requests and results traverse the network between the LLM
tiers and the vector-search tier.
The vector-search cluster shards embeddings and the graph
index across its own GPUs (e.g., CAGRA, Milvus). Prefill and
decode instances issue retrieval RPCs to this pool and receive
top-kdocument chunks. No GPU on the LLM side is reserved
for vector search, thus, also eliminating intra-node and inter-
node bandwidth contention for both prefill and decode.
3.2 Continuous batching on vector search.
To improve GPU utilization and solve latency jitter that arise
when CAGRA executes each request in batches, we design
a continuous-batch execution that treats oneextendstep on
graph as the unit and interleaves many requests precisely at
that granularity. As illustrated in Fig. 3, the GPU keeps the
database vectorsX ∈RN×dresident in HBM together with
a fixed-degree neighbor list G∈NN×D, so distance and ex-
pansion are pure device-memory operations. Each request
maintains compact device-side state: an internal topM (ids
and distances) used for storing candidates by far, a visited
table used to admit only first-seen candidates to computation.
A scheduling loop advances all active requests through oneex-
tend: for each request we select up to pnon-expanded parents
3

Vector Search Instances
GPUsPrefill Instances
GPUs
Decode Instances
GPUsAccepting reqsRound -
robin
Round -
robinQ_pre  target TTFT
Q_dec  target TPTEarliest DDL First Policy
FIFO and aging policy Running 
batchmerge
Two priority queues for prefill and decode requestsFigure 4: Priority queue-based scheduling for prefill/decode
instances inTrinity.
from its topM , read Dneighbors per parent from the neighbor
list, filter by visited , and emit the surviving ids asdistance
tasksinto a global, cross-request task array. We then evaluate
all accumulated distance tasks with a single fixed-shape kernel
that uses warp-splitting teams with high SM occupancy and
cuda graph. If the available tasks are fewer than the captured
kernel size, we round up with masked dummies to preserve a
stable operator shape. Then, the computed (id,dist) are scat-
tered back to their origin requests, locally merged with topM ,
and the consumed parents are marked expanded . Early-stop
is decided independently per request when the candidates in
topM list are not changed, so completed requests exit immedi-
ately and their device buffers are recycled, while new arrivals
are admitted and begin contributing tasks to the next distance
batch without idling the GPU.
Compared to the original per-request batching, continuous
batching yields a distinct set of advantages in a single, com-
patible design. First, cross-request aggregation converts many
tiny, shape-varying distance evaluations into a single fixed-
shape operator that maintains near-peak HBM throughput and
SM utilization even when individual requests are short, un-
even, or bursty. Second, using a small number of fixed-shape
launches (or a captured CUDA Graph) amortizes CPU/driver
overhead and reduces variance, producing smoother P50/P95
and predictable capacity at scale. Third, the kernel’s stable
shape with fixed number of degree Dand fixed captured batch
size with safe padding simplifies memory planning. All fin-
ished queries vacate resources immediately, and newcomers
join the very next distance batch with only a short, config-
urable flush timeout bounding additional wait. Since CAGRA
maintains all vectors and the fixed-degree graph in HBM, our
design keeps vector search results accuracy/recall behavior
intact while turning the global distance stage into a steady,
high-throughput engine that better matches GPU hardware
characteristics.
3.3 Latency-aware requests scheduling for pre-
fill/decode instances.
To meet TTFT for prefill and TPT for decode with best ef-
forts, we add a scheduler in the vector pool that coordinates
search requests from both prefill and decode stages. For pre-
fill instances, it will make latency-critical RAG requests toretrieve prompt-relevant context upon it receives batched re-
quests„ whereas decode emits periodic RAG probes every
∆tokens1to preserve perplexity and relevance during new
token generation. A naïve policy that always favors vector
search requests from prefill side will cause decode to wait
on retrieval, it will leave token generation idling for context
on decode side, thereby reducing average token throughput
of ranks even if TTFT appears healthy. Conversely, always
favoring decode will delay prefill stage, push TTFT beyond
the latency budget, and fails to fill the KVCache transfer link
(e.g., Mooncake) bandwidth between prefill and decode pools,
enough to keep decode workers busy, and the PD disaggre-
gation pipeline never saturates. Thus, we design an adaptive
scheduler that works with two requests queues, and feed the
running batch with vector requests from prefill and decode
sides flexibly, so that latency SLOs will be met and the GPU
utilization achieved by PD disaggregation will be preserved.
As shown in Fig. 4, we use two priority queues to schedule
vector search requests: a high priority prefill queue Qpreand
a lower priority decode queue Qdec. When we select request
candidates from two queues into the current running batch,
we build a request buffer of length N, the scheduler sets N
to match the number of available slots in the current batch,
reserves a fraction r∈[r min,rmax]forQpreso that at least
rNentries come from prefill, and fills the remainder with
Qdec. IfQprecannot supply enough entries at that moment, its
unused share is immediately given to Qdec. Ifnpre+n dec<N,
the builder pads with masked dummy tasks to preserve the
kernel’s fixed shape.
ForQprewe apply Earliest Deadline First (EDF) policy
and slack-driven scheduling approach: each request carries a
deadline as ddl=t arr+L pre,max and an estimate of remaining
extends ˜Ein vector search, and we rank by slack=ddl−
(tnow+˜E·T ext), where Textis the measured average latency
per extend. Furthermore, we assign Qprea short flush timeout
τpre, ensuring urgent prefill search request can be appended
into running batch without waiting for the global timeout.
ForQdecwe use FIFO to select request candidates. Together
with the prefill reservation rand the short prefill timeout
τpre, FIFO allows decode to steadily consume the remaining
capacity while prefill meets TTFT and keeps the KV cache
pipeline saturated.
The scheduler materializes a pair (npre,ndec)with npre≥rN
andnpre+n dec=N, launches one fixed shape distance kernel
with warp teams, and scatters results back for per request topM
merge. A lightweight control loop runs every few hundred
milliseconds and adjusts randτpreusing real-time feedback
to improve the KVCache link bandwidth Bkvtoward its target
B⋆
kv, reduce the prefill P95 wait (a proxy for TTFT), and lower
the fraction of decode time stalled by RAG. When ukv<u⋆
kv
the scheduler increases ror shortens τprein order to acceler-
ate prefill, while rising decode stalls decrease rso that Qdec
1There are many of different RAG search triggering policies, here we use
a fixed-tokens policy as an example.
4

occupies more ofN.
This design intentionally makes decode as absorption point
and gives prefill first class latency protection. Prefill receives
earliest deadline first with step aware slack, a reserved batch
share r, and a short τpre, therefore the KV cache is primed
quickly and the KV link remains saturated rather than oscil-
lating.
References
[1] Ernie 4.5 technical report.
https://ernie.baidu.com/blog/publication/ernie_technical
_report.pdf.
[2] Langchain: The platform for reliable agents.
https://github.com/langchain-ai/langchain.
[3]ADAMS, P., LI, M., ZHANG, S., TAN, L., CHEN, Q.,
LI, M., LI, Z., RISVIK, K.,ANDSIMHADRI, H. V.
Distributedann: Efficient scaling of a single diskann
graph across thousands of computers.arXiv preprint
arXiv:2509.06046(2025).
[4]BANG, F. Gptcache: An open-source semantic cache
for llm applications enabling faster answers and cost
savings. InProceedings of the 3rd Workshop for Natural
Language Processing Open Source Software (NLP-OSS
2023)(2023), pp. 212–218.
[5]DEEPSEEK-AI, D. G., YANG, D., ZHANG, H., SONG,
J., ZHANG, R., XU, R.,ET AL. Deepseek-r1: Incen-
tivizing reasoning capability in llms via reinforcement
learning.” arxiv.Preprint posted online on 22(2025),
13–14.
[6]GUI, Y., YIN, P., YAN, X., ZHANG, C., ZHANG,
W.,ANDCHENG, J. Pilotann: Memory-bounded
gpu acceleration for vector search.arXiv preprint
arXiv:2503.21206(2025).
[7]JAYARAMSUBRAMANYA, S., DEVVRIT, F.,
SIMHADRI, H. V., KRISHNAWAMY, R.,AND
KADEKODI, R. Diskann: Fast accurate billion-point
nearest neighbor search on a single node.Advances in
neural information processing Systems 32(2019).
[8]JIANG, W., LI, S., ZHU, Y.,DEFINELICHT, J., HE,
Z., SHI, R., RENGGLI, C., ZHANG, S., REKATSINAS,
T., HOEFLER, T.,ET AL. Co-design hardware and al-
gorithm for vector search. InProceedings of the Inter-
national Conference for High Performance Computing,
Networking, Storage and Analysis(2023), pp. 1–15.
[9]JIANG, W., SUBRAMANIAN, S., GRAVES, C.,
ALONSO, G., YAZDANBAKHSH, A.,ANDDADU,
V. Rago: Systematic performance optimization for
retrieval-augmented generation serving. InProceedingsof the 52nd Annual International Symposium on
Computer Architecture(2025), pp. 974–989.
[10] JIANG, W., ZELLER, M., WALEFFE, R., HOEFLER, T.,
ANDALONSO, G. Chameleon: a heterogeneous and dis-
aggregated accelerator system for retrieval-augmented
language models.arXiv preprint arXiv:2310.09949
(2023).
[11] JIANG, W., ZHANG, S., HAN, B., WANG, J., WANG,
B.,ANDKRASKA, T. Piperag: Fast retrieval-augmented
generation via adaptive pipeline parallelism. InProceed-
ings of the 31st ACM SIGKDD Conference on Knowl-
edge Discovery and Data Mining V . 1(2025), pp. 589–
600.
[12] JIN, C., ZHANG, Z., JIANG, X., LIU, F., LIU, S., LIU,
X.,ANDJIN, X. Ragcache: Efficient knowledge caching
for retrieval-augmented generation.ACM Transactions
on Computer Systems(2024).
[13] JIN, Y., WANG, T., LIN, H., SONG, M., LI, P., MA,
Y., SHAN, Y., YUAN, Z., LI, C., SUN, Y.,ET AL. P/d-
serve: Serving disaggregated large language model at
scale.arXiv preprint arXiv:2408.08147(2024).
[14] KWON, W., LI, Z., ZHUANG, S., SHENG, Y., ZHENG,
L., YU, C. H., GONZALEZ, J. E., ZHANG, H.,AND
STOICA, I. Efficient memory management for large lan-
guage model serving with pagedattention. InProceed-
ings of the ACM SIGOPS 29th Symposium on Operating
Systems Principles(2023).
[15] LIU, A., FENG, B., XUE, B., WANG, B., WU, B.,
LU, C., ZHAO, C., DENG, C., ZHANG, C., RUAN, C.,
ET AL. Deepseek-v3 technical report.arXiv preprint
arXiv:2412.19437(2024).
[16] LIU, S., ZENG, Z., CHEN, L., AINIHAER, A., RA-
MASAMI, A., CHEN, S., XU, Y., WU, M.,ANDWANG,
J. Tigervector: Supporting vector search in graph
databases for advanced rags. InCompanion of the
2025 International Conference on Management of Data
(2025), pp. 553–565.
[17] LIU, Z., CHENG, S., TAN, G., YOU, Y.,ANDTAO,
D. Elasticmm: Efficient multimodal llms serving
with elastic multimodal parallelism.arXiv preprint
arXiv:2507.10069(2025).
[18] OOTOMO, H., NARUSE, A., NOLET, C., WANG, R.,
FEHER, T.,ANDWANG, Y. Cagra: Highly parallel
graph construction and approximate nearest neighbor
search for gpus. In2024 IEEE 40th International Con-
ference on Data Engineering (ICDE)(2024), IEEE,
pp. 4236–4247.
5

[19] PIKTUS, A., PETRONI, F., KARPUKHIN, V.,
OKHONKO, D., BROSCHEIT, S., IZACARD, G.,
LEWIS, P., OGUZ, B., GRAVE, E., YIH, W.,AND
RIEDEL, S. The web is your oyster - knowledge-
intensive NLP against a very large web corpus.CoRR
abs/2112.09924(2021).
[20] SHENG, Y., ZHENG, L., YUAN, B., LI, Z., RYABININ,
M., CHEN, B., LIANG, P., RÉ, C., STOICA, I.,AND
ZHANG, C. Flexgen: High-throughput generative in-
ference of large language models with a single gpu. In
International Conference on Machine Learning(2023),
PMLR, pp. 31094–31116.
[21] TIAN, B., LIU, H., TANG, Y., XIAO, S., DUAN, Z.,
LIAO, X., JIN, H., ZHANG, X., ZHU, J.,ANDZHANG,
Y. Towards high-throughput and low-latency billion-
scale vector search via {CPU/GPU }collaborative filter-
ing and re-ranking. In23rd USENIX Conference on File
and Storage Technologies (FAST 25)(2025), pp. 171–
185.
[22] WANG, J., YI, X., GUO, R., JIN, H., XU, P., LI, S.,
WANG, X., GUO, X., LI, C., XU, X.,ET AL. Milvus:
A purpose-built vector data management system. In
Proceedings of the 2021 international conference on
management of data(2021), pp. 2614–2627.
[23] YU, W., LIAO, N., LUO, S.,ANDLIU, J. Ragdoll:
Efficient offloading-based online rag system on a single
gpu.arXiv preprint arXiv:2504.15302(2025).
[24] ZHAO, W., TAN, S.,ANDLI, P. Song: Approximate
nearest neighbor search on gpu. In2020 IEEE 36th
International Conference on Data Engineering (ICDE)
(2020), IEEE, pp. 1033–1044.
[25] ZHENG, L., YIN, L., XIE, Z., SUN, C. L., HUANG,
J., YU, C. H., CAO, S., KOZYRAKIS, C., STOICA, I.,
GONZALEZ, J. E.,ET AL. Sglang: Efficient execu-
tion of structured language model programs.Advances
in neural information processing systems 37(2024),
62557–62583.
[26] ZHONG, Y., LIU, S., CHEN, J., HU, J., ZHU, Y., LIU,
X., JIN, X.,ANDZHANG, H. {DistServe }: Disaggre-
gating prefill and decoding for goodput-optimized large
language model serving. In18th USENIX Symposium on
Operating Systems Design and Implementation (OSDI
24)(2024), pp. 193–210.
6