# HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous Mobile SoC

**Authors**: Maoliang Li, Jiayu Chen, Zihao Zheng, Ziqian Li, Xinhao Sun, Guojie Luo, Chenchen Liu, Xiang Chen

**Published**: 2026-03-02 09:51:01

**PDF URL**: [https://arxiv.org/pdf/2603.01661v1](https://arxiv.org/pdf/2603.01661v1)

## Abstract
With the increasing computational capability of mobile devices, deploying agentic retrieval-augmented generation (RAG) locally on heterogeneous System-on-Chips (SoCs) has become a promising way to enhance LLM-based applications. However, agentic RAG induces multi-stage workflows with heterogeneous models and dynamic execution flow, while mobile SoCs exhibit strong accelerator affinity, shape sensitivity, and shared-memory bandwidth contention, making naive scheduling ineffective. We present HeRo, a heterogeneous-aware framework for low-latency agentic RAG on mobile SoCs. HeRo builds profiling-based performance models for each sub-stage and model-PU configuration, capturing latency, workload shape, and contention-induced slowdown, and leverages them in a lightweight online scheduler that combines shape-aware sub-stage partitioning, criticality-based accelerator mapping, and bandwidth-aware concurrency control. Experiments on commercial mobile devices show that HeRo reduces end-to-end latency by up to $10.94\times$ over existing deployment strategies, enabling practical on-device agentic RAG.

## Full Text


<!-- PDF content starts -->

HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous
Mobile SoC
Maoliang Li
School of Computer Science
Peking University
Beijing, ChinaJiayu Chen
School of Computer Science
Peking University
Beijing, ChinaZihao Zheng
School of Computer Science
Peking University
Beijing, China
Ziqian Li
School of Computer Science
Northwestern Polytechnical
University
Xi’an, ChinaXinhao Sun
School of Computer Science
Peking University
Beijing, ChinaGuojie Luo
School of Computer Science
Peking University
Beijing, China
Chenchen Liu
School of Computer Science
Peking University
Beijing, China†Xiang Chen
School of Computer Science
Peking University
Beijing, China
Abstract
With the increasing computational capability of mobile devices,
deploying agentic retrieval-augmented generation (RAG) locally on
heterogeneous System-on-Chips (SoCs) has become a promising
way to enhance LLM-based applications. However, agentic RAG
induces multi-stage workflows with heterogeneous models and
dynamic execution flow, while mobile SoCs exhibit strong accel-
erator affinity, shape sensitivity, and shared-memory bandwidth
contention, making naive scheduling ineffective. We presentHeRo,
a heterogeneous-aware framework for low-latency agentic RAG
on mobile SoCs.HeRobuilds profiling-based performance models
for each sub-stage and model–PU configuration, capturing latency,
workload shape, and contention-induced slowdown, and leverages
them in a lightweight online scheduler that combines shape-aware
sub-stage partitioning, criticality-based accelerator mapping, and
bandwidth-aware concurrency control. Experiments on commercial
mobile devices show thatHeRoreduces end-to-end latency by up
to10.94×over existing deployment strategies, enabling practical
on-device agentic RAG.
1 Introduction
With the integration of diverse AI processing units (PUs) into mo-
bile Systems-on-Chip(SoCs) [ 20] and the advancement of optimiza-
tion techniques, large language model–based applications (LLM
Apps) are increasingly being deployed on mobile devices. Given
the limited computing resources on mobile platforms, recent state-
of-the-art systems such as HeteroInfer [ 3] have begun exploiting
concurrent execution across heterogeneous accelerators to max-
imize inference throughput. Meanwhile, because modern mobile
devices store sensitive personal data that must be processed locally
for privacy, on-device retrieval-augmented generation (RAG) has
emerged as a highly promising scenario for mobile LLMs.
RAG augments LLMs with local context by retrieving informa-
tion relevant to a user query from a vector database [ 5] and in-
corporating it into the prompt during response generation. While
Doc.IndexVectorSearchRerankVectorDatabaseQuery RewriterGenerationEmbedUserQueryDoc.ApplicationGraphModelllama.cppHeteroInferOptionalSearchPlannerContextRefiner
AcceleratorAyoHeRoRetrieval  Augmented Phone Agent
Rerank ModelChat ModelEmbed ModelWeb Search
Search Model
CPU
GPU
NPU
LPDDRUnified Memory BusMobile SoCFigure 1: RAG Application Optimization Stack. Our work is
a hardware and graph co-scheduling framework.
naive RAG adopts a simple retrieval–then–generation pipeline,
recent progress in agentic RAG [ 23] has substantially increased
workflow complexity. With specialist agents such as query refiners,
rerankers, and web searchers (we refer to the execution of each
agent as astage), agentic RAG systems involve multiple LLMs with
heterogeneous architectures, complex inter-stage dependencies,
and dynamic runtime execution flows.
Prior works on mobile LLM optimization focus primarily on
model-level optimizations in the AI compiling stack illustrated in
Fig. 1. Existing systems either statically map a model to an accel-
erator (e.g., llama.cpp [ 8], mllm.npu [ 26]) or partition operators
across heterogeneous PUs (e.g., PowerInfer2 [ 28], HeteroInfer [ 3]).
In contrast, agentic RAG is inherently multi-model and multi-stage:
each agent stage can be decomposed intosub-stages, forming a
fine-grained task graph. The scheduling problem thus extends from
intra-model operator placement to inter-stage coordination over a
sub-stage DAG, where both graph topology and execution cost are
runtime-dependent. This cross-layer design space from the RAG
graph to the accelerator level remains largely unexplored.arXiv:2603.01661v1  [cs.DC]  2 Mar 2026

Maoliang Li, Jiayu Chen, Zihao Zheng, Ziqian Li, Xinhao Sun, Guojie Luo, Chenchen Liu, and†Xiang Chen
Table 1: Comparison with Prior Arts for RAG Deployment
Ayo HedraRAG llama.cpp HeteroInfer Ours
Hetero
Affinity×CPU-GPU×GPU-NPU xPU
Contention
ControlDiscrete Discrete× × ✓
xPU
MappingStatic Static Static Static Adaptive
Intra-Stage
Opt×✓Operator Op + Graph Stage
Inter-Stage
Dependency✓ ✓× × ✓
From the hardware perspective, mobile SoCs introduce three key
issues. First,stage–accelerator affinity: heterogeneous PUs ex-
hibit very different latencies for the same RAG sub-stage. Second,
workload shape sensitivity: mobile runtimes and accelerators
offer limited shape adaptability, making inference sensitive to input
shape besides workload size. Third,bandwidth contention: all
PUs share a unified DRAM; concurrent sub-stages increase traffic
and can dominate the latency benefit of parallelism.
From the workflow and algorithmic perspective, agentic RAG fur-
ther complicates orchestration.Intra-stage partitionis required to
expose more parallelism: a logical stage can be split into sub-stages
via batching or token-group partitioning, with different latency and
downstream concurrency. At the same time,dynamic inter-stage
dependenciescause the workflow to evolve at runtime, since agent
decisions may create extra retrieval or tool calling.
These issues jointly lead to unique challenges for cross-layer
scheduling: (1) choosingintra-stage workload partitionwith proper
shape and semantics, (2)mapping sub-stages to acceleratorswith
stage-PU affinity awareness, (3) controllinginter-stage concurrency
under shared-bandwidth interference, and (4) scheduling on a par-
tial and evolving DAG withdynamic dependencies.
To address these challenges, we proposeHeRo, a heterogeneity-
aware RAG orchestration framework for mobile SoCs.HeRobuilds
profiling-based performance models that capture latency and band-
width for each model–PU combination. On top of this, it builds
a scheduler that integrates shape-aware sub-stage partitioning,
criticality-affinity joint accelerator mapping that prioritizes sub-
stages on the critical path, and bandwidth-aware concurrency con-
trol that selectively enables parallel execution with minimal slow-
down. Our contributions are summarized as follows:
•Heterogeneous performance modeling.We present
profiling-based models for RAG stages that capture PU
affinity, shape sensitivity, and bandwidth contention on
mobile SoCs, enlarging the scheduling space.
•Adaptive heterogeneous RAG scheduler.We design an
online scheduler that combines stage partition, accelera-
tor mapping, and concurrency control for agentic RAG on
mobile SoCs.
•System implementation and evaluation.We implement
HeRoas the first mobile system for efficient scheduling
of agentic RAG workflows, and demonstrate up to10 .94×
latency reduction on commercial mobile phones.2 Background
2.1 On-Device LLM Inference
Considering the imperatives of personal privacy, deploying LLMs
on local mobile devices is prevalent. Vendors are advancing their
mobile platforms [ 1,18,20,21] with various accelerators, includ-
ing GPUs and neural processing units (NPUs). To enhance the
responsiveness of LLM-based apps, substantial efforts focus on
fully exploiting hardware on resource-constrained mobile devices.
llama.cpp [ 8] provides highly optimized kernels and model sup-
port for diverse accelerators, establishing a foundation for LLM
inference on mobile SoCs. mllm.npu [ 26] integrates CPU/NPU co-
execution with chunked prefilling and fine-grained graph schedul-
ing, while PowerInfer2 [ 28] employs sparse model partitioning and
selective offloading. Recent frameworks such as HeteroInfer [ 3]
and Agent.xpu further coordinate GPU–NPU execution through
fine-grained tensor partition and inter-device communication opti-
mizations. Despite these advances, existing approaches primarily
target the latency optimization of asinglemodel and fall short
in handling concurrent execution ofmultiplemodels. The unique
challenges of agentic RAG systems, which require coordinated exe-
cution across multiple LLM instances with size-varying workloads
and data dependencies, are still unaddressed.
2.2 Agentic RAG Workflows
In general, RAG systems follow a retrieval-then-generation pipeline.
However, this structure struggles with complex inputs contain-
ing ambiguous or multifaceted queries. To enhance retrieval ac-
curacy, recent works incorporate specialist agents [ 7,25,27]. Pre-
processing agents, such as query rewriters and decomposers, trans-
form casual user inputs into search-friendly queries or structured
sub-queries. Conversely, post-processing agents, including docu-
ment rerankers and context refiners, aim to provide more rele-
vant and coherent context to the LLM, thereby improving gener-
ation quality. These optimizations are particularly beneficial for
on-device LLMs with limited generalist knowledge. Nevertheless,
the introduction of multiple agents brings significant multi-model
coordination [ 24,31] and increases runtime dynamism, posing new
challenges for efficient execution on mobile platforms.
2.3 RAG Deployment Optimization
With the prevalence of RAG, deployment optimization becomes
a pivotal problem. Most works lie instage-specific optimization,
Table 2: Specifications of Mobile SoCs Adopted in this Work
Redmi K80 (Snapdragon 8 Gen 3 SoC)
CPU Kryo, 8-core, up to 3.4 GHz
GPU Adreno 750 | FP16=2.8 TFlops
NPU Hexagon v75 | INT8=34Tops
Memory 12GB LPDDR5x | Bandwidth: 76.8 GB/s with 64 bits
OnePlus 13 (Snapdragon 8 Gen 4 SoC)
CPU Oryon, 8-core, up to 4.5 GHz
GPU Adreno 830 | FP16=3.4 TFlops
NPU Hexagon v79 | INT8=50Tops
Memory 24GB LPDDR5x | Bandwidth: 84.8 GB/s with 64 bits

HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous Mobile SoC
such as long-context LLM generation [ 26], vector search [ 19] and
reranking [ 32]. In contrast, workflow-level optimization orchestrat-
ing multiple stages for higher throughput emerges in cloud ser-
vices like [ 15,30]. On the edge side, Ayo [ 23] and HedraRAG [ 11]
make preliminary progress. Ayo attempts to improve intra-stage
parallelism through task decomposition, but suffers from redun-
dant computation and limited adaptability under dynamic loads.
HedraRAG addresses interleaved retrieval–generation scheduling
for hybrid CPU-GPU cooperation, yet its optimization remains
restricted to a narrow set of model types and lacks system-level
heterogeneity awareness. When considering on-device scenarios
with complex agentic workflows, there is still a gap in efficient
mapping from workflow stages to multiple accelerators without
exploiting hardware characteristics.
3 Agentic RAG on Heterogeneous SoCs
In this section, we provide analysis and modeling for agentic RAG
workflow inference on heterogeneous accelerators. In this work, for
development convenience, we use Qualcomm’s Snapdragon 8Gen3
and 8 Elite (8Gen4) [20, 21] as specified in Tab 2.
3.1 Agentic RAG Workflows
As illustrated in Fig. 1, agentic RAG introduces substantially more
complex workflows than classical two-stage retrieval–generation
pipelines. We identify two key characteristics.
Intra-Stage Partition.Although the user application specifies
the high-level workflow between agents, a given agent may be
instantiated multiple times during execution. Moreover, stage-level
dependencies often introduce unnecessary synchronization. For
example, a query rewriter may issue several search requests, but the
first search need not wait for all subsequent queries to be generated.
To expose additional parallelism, we model dependencies at a finer
granularity by decomposing stages intosub-stages, which represent
a partition of the workload within a stage, such as indexing part of
the documents or decoding a sub-query.
Dynamic Inter-Stage Dependencies.In agentic RAG, the exe-
cution flow is not predetermined, but rather evolves based on the
responses of the agents. Agents may generate responses of varying
lengths, perform additional retrieval steps, or skip optional stages.
This means that the graph is partially observable. At time 𝑡, the
scheduler observes only apartialDAG 𝐺obs(𝑡), where𝑉obs(𝑡)⊆𝑉
and𝐸obs(𝑡)⊆𝐸 . New nodes and edges materialize as upstream
decision stages finish. This dynamic depth requires scheduling un-
der incomplete future knowledge, leveraging statistical priors over
typical agent behaviors to anticipate downstream computation.
3.2 Modeling RAG on Heterogeneous SoCs
We consider an SoC equipped with a set of heterogeneous process-
ing units (PUs)K, which share a unified DRAM subsystem. Under
this setting, RAG inference exhibits the following characteristics.
Stage-Accelerator Affinity.For each sub-stage 𝑣∈𝑉 and PU
𝑘∈K , differences in model implementations and kernel optimiza-
tions yield distinct behaviors. We capture this via a configuration
setC𝑣, where each configuration 𝑐∈C𝑣specifies the target PU 𝑘𝑐
and a workload shape 𝑠𝑣(𝑐)(e.g., batch size or sequence length).
CPUGPUGPUXPU(16, 512)XPU(16, 1024)XPU(64, 1024)
IndexingThroughput (token/s)1001000
QueryExpansionRerankingLLM Decode10104481041021291326496522301
33161545503924326014195921181275966072184
1612101518151315169711Performance is sensitive to the workloadshapeStages differ in accelerator affinityFigure 2: Stage-Accelerator Affinity and Shape Sensitivity.
GPUNPUGPU+NPUGPUNPUGPU+NPU
1Workload 1Memory bandwidth (GB/s)Execution time(s)23456
203040506070
Workload 2Workload 3Workload 4+34.7%+21.3%-14.7%-15.6%Memory bandwidth contentionPer-PU execution time increases
Figure 3: Contention Slowdown under Various Parallelism.
Given𝑐, thebase latency 𝑝0
𝑣(𝑐)without interference is profiled of-
fline. The affinity between stages and accelerators is reflected in
the variation among base latencies. For example, stages such as
indexing or reranking often run much faster on NPUs, whereas
LLM generation stages favor GPUs, as shown in Fig. 2.
Workload Shape Sensitivity.Many stages, such as document
indexing or reranking, can naturally process multiple items in
batches. For a fixed stage and PU, the base execution time depends
on batch size in a highly non-linear manner due to effects such
as tiling, kernel fusion, and memory hierarchy behavior. Formally,
𝑝0
𝑣(𝑐)=𝑓𝑚,𝑘 𝑠𝑣(𝑐), where𝑓is obtained via offline profiling. Larger
batches do not always yield better per-item efficiency, as shown in
Fig. 2. Thus, the orchestrator must selectively exploit batch-size sen-
sitivity for batchable stages, while relying on sub-stage partitioning
for streaming stages, i.e., autoregressive decoding.
Inter-Stage Bandwidth Contention.On mobile SoCs, all ac-
celerators share a unified DRAM subsystem with peak bandwidth
𝐵0. When multiple independent stages execute on different PUs,
bandwidth contention may incur latency overhead, as depicted in
Fig. 3. Each configuration 𝑐of a sub-stage 𝑣incurs a bandwidth
demand𝑏𝑣(𝑐), estimated via profiling-based modeling. At time 𝑡,
letA(𝑡) denote the set of active sub-stages; the aggregate band-
width demand is 𝐵(𝑡)=Í
𝑣∈A(𝑡)𝑏𝑣(𝑐𝑣). Rather than treating 𝐵0
as a hard limit, we model contention-induced slowdown using a
factor𝜙𝑣(𝐵(𝑡))≥ 1that increases monotonically with 𝐵(𝑡): when
𝐵(𝑡)≪𝐵 0,𝜙𝑣(𝐵(𝑡))≈ 1; as𝐵(𝑡) approaches or exceeds 𝐵0,𝜙𝑣(𝐵(𝑡))
grows due to rising DRAM latency and reduced effective through-
put. The function 𝜙𝑣(·)is stage- and configuration-dependent, cap-
turing model-level and hardware-level sensitivity to bandwidth
pressure. The effective execution time can now be approximated
as:
𝑝𝑣(𝑐𝑣,𝐵(𝑡))≈𝑝0
𝑣(𝑐𝑣)𝜙𝑣,𝑐(𝐵(𝑡)).(1)

Maoliang Li, Jiayu Chen, Zihao Zheng, Ziqian Li, Xinhao Sun, Guojie Luo, Chenchen Liu, and†Xiang Chen
3.3 Heterogeneous Deployment Challenges
The combination of agentic RAG workflows and heterogeneous
mobile SoCs introduces several deployment challenges:
Intra-Stage Workload Partitioning.Many stages support multi-
ple workload shapes and partitioning strategies, leading to various
per-PU latencies and degrees of downstream parallelism.
Stage–Accelerator Mapping.Each stage exhibits heterogeneous
performance across PUs. The scheduler must assign stages to ac-
celerators to exploit affinity while avoiding workload imbalance.
Inter-Stage Concurrency Control.Although concurrent use of
multiple accelerators improves utilization, for a latency-critical
request, excessive parallelism can degrade performance when band-
width contention induces slowdown.
Scheduling with Dynamic Dependency.The workflow graph
evolves based on agent decisions. The scheduler must operate on a
partial and evolving DAG, using statistical priors to anticipate and
adapt to future stages.
4 Orchestration Methods
4.1 Problem Formulation
We focus on minimizing the latency of a single RAG graph, with
each node𝑣∈𝑉 being a sub-stage associated with: (1) model, (2)
configuration set including the PU and workload shape. and (3)
profiled performance model for runtime, bandwidth, and slowdown.
For each𝑣, the scheduler chooses a configuration 𝑐𝑣∈C𝑣and a
start time𝑆 𝑣≥0. The end-to-end latency of the RAG flow is:
𝑇=max
𝑣∈𝑉𝐹𝑣,where𝐹𝑣=𝑆𝑣+𝑝0
𝑣(𝑐𝑣)·¯𝜙𝑣,(2)
where𝐹𝑣denotes the completion time of 𝑣,¯𝜙𝑣is an average slow-
down factor over the execution of 𝑣, induced by the time-varying
𝐵(𝑡) and𝜙𝑣(·). Minimizing 𝑇requires determining {𝑐𝑣,𝑆𝑣}𝑣∈𝑉un-
der dependency and interference, forming a variant of the precedence-
constrained task-graph scheduling problem on unrelated machines
which is NP-hard [ 6]. We therefore design online heuristics that
exploit the specific structure of the agentic RAG workflow and the
slowdown model.
4.2 Adaptive Heterogeneous Scheduling
Our orchestration algorithm consists of three key components,
namely the sub-stage partitioner, the priority estimator, and the
concurrency controller, combined into an online heterogeneous
scheduler for effectively handling RAG execution challenges.
Shape-Aware Sub-Stage Partition.Proper sub-stage bound-
aries and batching decisions are crucial for balancing latency and
execution efficiency, as illustrated in Fig. 4(a). For streaming work-
loads such as query rewriting, scheduling is performed at the granu-
larity of token groups to amortize scheduling overhead. As decoding
progresses, the runtime monitors the generated content and triggers
downstream stages once their data dependencies are satisfied. For
batchable stages (e.g., indexing document chunks), we offline profile
a candidate set of batch sizes N𝑚,𝑘for each model-PU pair (𝑚,𝑘) ,
recording both latency and bandwidth usage. Given an input work-
load, the runtime partitions the stage by selecting the batch size
that minimizes the estimated execution time. We assume each stage
executes on a single PU; if inter-PU migration occurs, partitioningis recomputed based on the remaining workload. Specifically, for
stage𝑚with workload size 𝐿executed on PU 𝑘, we search for the
optimal workload unit corresponding to batch size𝑛:
arg min
𝑛∈N𝑚,𝑘⌈𝐿/𝑛⌉·𝑝0
𝑣(𝑐𝑣),where𝑣=𝑚,𝑐 𝑣=(𝑛,𝑘).(3)
Critical-Stage Prioritized Accelerator Mapping.On-device
RAG inference is single-batched. Thus, mapping stages on the crit-
ical path to the fastest PU is pivotal to latency reduction. Since
the full execution graph is unknown at runtime (Fig. 4(b)), we esti-
mate node priority using acriticality scorecomposed of anobserved
term and afutureterm. At time 𝑡, the runtime maintains 𝐺obs(𝑡)
and theready setR(𝑡) containing nodes whose predecessors have
completed. The observed term 𝐶𝑆L(𝑣)is computed on 𝐺obs(𝑡)as
the longest remaining path from 𝑣to any undetermined node. Dur-
ing this simulation, PU assignment follows a dependency-agnostic
heuristic similar to SJF, prioritizing shorter-latency tasks; 𝐶𝑆L(𝑣)
is updated whenever 𝐺obs(𝑡)evolves. To account for uncertain fu-
ture execution, we compute the future term 𝐶𝑆F(𝑣)on a predefined
RAG workflow graph, using historical averages to estimate the
likelihood of downstream activation. Intuitively, agents such as the
search planner tend to trigger more subsequent computation than
lightweight post-processing modules, resulting in higher expected
future criticality. The final criticality score is defined in Eq. 4, where
𝛽controls the influence of future dependencies. Nodes with larger
𝐶𝑆(𝑣) are prioritized, as they lie on long observed paths and are
more likely to unlock future work:
𝐶𝑆(𝑣)=𝐶𝑆 L(𝑣)+𝛽·𝐶𝑆 F(𝑣).(4)
Bandwidth-Aware Concurrency Control.Under single-batch
execution and shared DRAM, overly aggressive parallelism can
stall nodes on the critical path and increase end-to-end latency,
as exemplified in Fig. 4(c). To avoid this, we limit concurrency
to prevent harmful interference, even when doing so temporarily
reduces PU utilization. When PU 𝑘becomes idle, the scheduler
considers issuing a ready node 𝑣∈R(𝑡) with configuration 𝑐𝑣. The
slowdown incurred by admitting 𝑣is estimated as 𝜙𝑣(𝐵(𝑡)+𝑏𝑣(𝑐𝑣)).
We first enforce a soft bandwidth constraint 𝐵softto avoid system-
wide degradation. Because evaluating global performance impact is
costly, we approximate it by measuring interference on the current
critical-path node 𝑣∗, defined as the ready or running node with the
highest criticality score 𝐶𝑆(𝑣∗). We introduce a contention penalty
in Eq. 5 that jointly reflects contention-induced slowdown and
node criticality, where (𝑡−𝑆𝑣∗)is the time𝑣∗has been active. This
formulation admits parallelism only when it does not significantly
impede critical-path progress, favoring decisions that minimize end-
to-end latency rather than maximize instantaneous PU utilization:
𝑊𝐵=𝜙𝑣∗ 𝐵(𝑡)+𝑏𝑣(𝑐𝑣)(𝑡−𝑆𝑣∗)𝑝𝑣∗(𝑐𝑣∗).(5)
Overall Scheduler.Alg. 1 outlines the scheduler ofHeRo, which
prioritizes nodes on the critical path while considering both affinity
and contention. At each scheduling step, the scheduler identifies the
most critical ready node 𝑣candaccording to Eq. 4, and selects the PU
and configuration that minimize the predicted completion time plus
the contention penalty. For all PUs capable of executing 𝑣cand, the
runtime enumerates a small set of shape-aware configurations and
prunes those that violate the bandwidth budget. For each remaining

HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous Mobile SoC
Search(b) Priority on DynamicGraph(d) HeRo’s Framework Workflow(a) Stage Partition(c) Bandwidth Contention ControlApp FlowGraphRuntime MappingProfilerIndexerRerankVectorSearchQuery RewriterQR Decode1t	QR Decode2RerankTimeline    IndexingmIndexingnQR PrefillQR Decode0Observed    Future    SearchCritical Pathcurrent
NPUGPUQR-DRankGen-PQR-DGen-PSearchbatch=8
PDDDVVVbs=4bs=4batch=8bs=4bs=4mapsplit
RRRNPUGPUQR-DRankGen-PQR-DSearchTimeline    
Contention Slowdown Critial Node    
Latency BandwidthBatch… Candidate Workload    ModelInstantializeDependencyAffinityContentionRankSearchRankGen-PSearchContention Control
Figure 4: Orchestration Techniques.
Algorithm 1Node-centric Online Heterogeneous RAG Scheduler
1:Input:profiling model𝑝0
𝑣(𝑐),𝑏𝑣(𝑐),𝜙𝑣(·), bandwidth budget𝐵 soft
2:whilethere exists an unfinished node in𝑉do
3: Update𝐺obs(𝑡),R(𝑡)with finished nodes
4: Update criticality𝐶𝑆(𝑣)for𝑣∈R(𝑡)∪A(𝑡)⊲Eq. 4
5:R tmp←R(𝑡)
6:whileexists idle PU andR tmp≠∅do
7:𝑣∗←arg max 𝑥∈R(𝑡)∪A(𝑡)𝐶𝑆(𝑥)⊲current critical-path node
8:𝑣cand←arg max 𝑣∈R tmp𝐶𝑆(𝑣)⊲most critical ready node
9:for allPU𝑘that can execute𝑣canddo
10:for all𝑐∈ShapeAwareConfigs(𝑣cand,𝑘)do⊲Eq.3
11:if𝐵(𝑡)+𝑏𝑣cand(𝑐)>𝐵 softthen continue
12: Calculate𝐹𝑣cand(𝑐)and𝑊𝐵(𝑣∗,𝑣cand,𝑡,𝑐)with Eq 2 and Eq 5
13: score(𝑐)←𝐹 𝑣∗(𝑐)+𝛼·𝑊 𝐵(𝑣∗,𝑣cand,𝑡,𝑐)
14:Call
𝑣cand←Call
𝑣cand∪{(𝑘,𝑐)}
15:ifCall
𝑣cand=∅thenR tmp\{𝑣cand},continue⊲try next critical node
16:𝑐 cand←arg minscore(𝑐), where𝑐∈Call
𝑣cand
17: Dispatch𝑣candon PU𝑘∗with config𝑐cand;𝑆𝑣cand←𝑡
18:R(𝑡)←R(𝑡)\{𝑣cand},R tmp←R tmp\{𝑣∗}
19:A(𝑡)←A(𝑡)∪{𝑣∗},𝐵(𝑡)←𝐵(𝑡)+𝑏𝑣cand(𝑐cand)
20:waituntil any𝑣∈A(𝑡)finishes; UpdateA(𝑡),𝐵(𝑡)and current time𝑡
configuration, it estimates the execution time of 𝑣candand its inter-
ference on the current critical-path node 𝑣∗using the contention
penalty𝑊𝐵(𝑣∗,𝑣cand,𝑡,𝑐) , then chooses the configuration with the
lowest overall score. If no feasible configuration is available for the
current most critical node, the scheduler defers it and considers
the next node in priority order. This ensures that high-criticality
nodes are dispatched to PUs that complete them earliest, while still
respecting bandwidth-aware concurrency constraints.
5 Implementation
We implementedHeRoon top of llama.cpp [ 8], Powerserve [ 4] (a
mobile LLM inference framework with Hexagon NPU support),
and Faiss [ 5], with roughly 5,000 lines of C/C++ code.HeRoreuses
the CPU and OpenCL backends from llama.cpp and integrates the
NPU backend from Powerserve. We extend the runtime to sup-
port both embedding and reranking model inference, as well as
chunked prefill mechanisms. To enable efficient heterogeneous ex-
ecution, we establish shared memory between CPU and GPU via
OpenCL, and between CPU and NPU using the QNN API [ 22]. Fur-
thermore, by leveraging the CL_MEM_USE_HOST_PTR flag, we map
NPU-side buffers into the GPU address space, enabling zero-copy
data sharing among all three processors. Following prior online
scheduling work [13, 14], we estimate latency, bandwidth require-
ments, and slowdown functions for irregular workload sizes usinga multi-feature linear regression model trained on sampled mea-
surements across different models, workload sizes, and background
contention levels. The scheduler exposes two hyperparameters: the
bandwidth-contention penalty weight 𝛼in priority assignment and
the future-term weight 𝛽in criticality estimation. We tune both
parameters for each deployment via grid search.
6 Experiments
6.1 Experiment Setup
Hardware Platform.As summarized in Tab. 2, we evaluateHeRo
on two representative mobile devices, Redmi K80 and OnePlus 13,
equipped with Snapdragon 8Gen4 and 8Gen4 SoCs, respectively.
Applications.Our experiments cover three agentic RAG work-
flows with varying complexity, derived from the template workflow
in Fig. 1.Workflow 1,Fast Document Finder, segments documents
into chunks (default size: 128, overlap: 10), embeds them, and stores
them in the vector DB. Then it retrieves and reranks the most rele-
vant chunks to generate the final response.Workflow 2,Advanced
Document QA Bot, extends Workflow 1 by enabling LLM-based
query rewriting and context refinement, trading off latency for im-
proved retrieval and comprehension accuracy.Workflow 3,Deep
Researcher, further incorporates online resources, with a lightweight
LLM–based search planner for issuing web-search requests.
Datasets.Following prior work [ 11,23] on RAG workflow schedul-
ing, we adopt four widely used RAG datasets with diverse workload
characteristics: FinqaBench [ 12], TruthfulQA [ 17], HotpotQA [ 29],
and 2WikiMultihopQA [ 10]. FinqaBench and TruthfulQA generally
contain shorter inputs (query length ≤70 tokens; context ∼200
tokens), whereas HotpotQA and 2WikiMultihopQA include longer
and more complex contexts, reaching up to 1k tokens. To accom-
modate mobile devices’ limited computing capability, we exclude
datapoints with extremely long queries or contexts.
Models.To evaluate the scalability ofHeRo, we instantiate the
workflows using two families of on-device LLMs with different
model scales and architectures: (1) the Qwen3 series [ 24,31], and
(2) the bge series [ 2,16] together with Llama3 [ 9]. The detailed
workflow–model compositions are shown in Fig. 5 and Fig. 6.
Baselines.To our knowledge, few studies have specifically opti-
mized RAG, especially agentic RAG workflows, on heterogeneous
mobile SoCs. By composing RAG workflows on top of popular
mobile LLM frameworks, we construct three strong baselines: (1)
Llama.cpp-GPU: all models except FAISS run on GPU with the
OpenCL backend adapted from llama.cpp ; (2)Powerserve-NPU:
all models except FAISS run on NPU with the QNN backend adapted

Maoliang Li, Jiayu Chen, Zihao Zheng, Ziqian Li, Xinhao Sun, Guojie Luo, Chenchen Liu, and†Xiang Chen
Table 3: SpeedUp Breakdown of Proposed Techniques
Technique 𝐶1Latency(s)𝐶 1SpdUp𝐶 2Latency(s)𝐶 2SpdUp
Baseline 5.79s 1.0x 17.23s 1.0x
+ Sub-Stage Partition 5.07s 1.14x 8.79s 1.96x
+ Criticality Guidance 4.23s 1.37x 6.81s 2.53x
+ Concurrency Control 4.63s 1.25x 8.24s 2.09x
ALL 3.82s 1.52x 5.38s 3.20x
from Powerserve; (3)Ayo-like: models are manually mapped to all
xPUs based on workflow dependencies and model sizes.
Metric.Mobile users typically issue a single request at a time. Thus,
we mainly evaluateHeRousing average end-to-end latency for each
query. For fair comparison, we remove traces with abnormally short
or long latencies, which often arise from unexpected behaviors due
to the limited capability of on-device LLMs.
6.2 Evaluation Results
End-to-End Speedup.HeRodelivers consistent improvements
over all baselines across datasets and workflows. llama.cpp shows
substantially higher latency due to the limited compute capability
of mobile GPUs. Although Powerserve provides noticeably better
performance, it still suffers from under-utilized resources, mak-
ing multi-accelerator placement inherently more effective. How-
ever, static mappings cannot address the utilization loss caused by
dependency constraints and heterogeneous workload affinity. By
performing online stage-accelerator mapping with heterogeneity-
aware affinity modeling,HeRoachieves up to1 .5×speedup over the
Ayo-like baseline and up to 10.94 ×improvement over the GPU-only
execution. These results highlight the necessity of exploiting multi-
ple accelerators jointly and performing affinity-aware scheduling.
Workflow-level Analysis.Workflow 1 is relatively simple and ex-
poses limited parallelism, soHeRoyields modest gains over baseline.
Most of the execution time is dominated by the document-indexing
stage, which is already highly NPU-friendly; hence, longer docu-
ments further diminish the attainable speedup. In contrast, Work-
flow 3 contains the most complex execution graph. Single-processor
baselines perform substantially worse due to strictly sequential ex-
ecution.HeRoachieves the highest acceleration with minimal PU
under-utilization. Even compared to the Ayo-like strategy,HeRo
still provides a notable speedup because the static Ayo-like mapping
often stalls on dependency bottlenecks and delays the execution of
critical-path stages.
Dataset-level Analysis.The runtime of linear baselines scales
nearly proportionally with input size; datasets with long contexts,
such as HotpotQA, incur much larger slowdowns. ForHeRo, the
observed speedup is less tightly correlated with input length, as the
agent may trigger additional computation when processing difficult
queries—particularly when the query rewriter or search planner is
activated in the more complex Workflows 2 and 3.
Model-level Analysis.HeRoprovides larger relative speedups
when using the Qwen3 family. This is primarily because the Qwen3
configuration employs a smaller 4B chat model, whereas the al-
ternative configuration uses an 8B Llama 3.1 model. The smaller
model results in a more balanced workload distribution and re-
duces hard dependencies that cannot be mitigated by sub-stagepartitioning. This effect is especially pronounced in Workflow 1,
where sequential components dominate overall latency.
Platform-level Analysis.AlthoughHeRoconsistently accelerates
execution on both platforms with different compute capabilities,
the degree of speedup varies. This variation stems from the dis-
crepancy between NPU performance and the memory-bandwidth-
to-compute ratio across platforms. For instance, on the 8 Gen 3
platform, larger relative speedups are observed for Workflows 2
and 3 because the memory bandwidth is higher relative to the NPU
FLOPs, enabling more effective concurrent accelerator utilization.
Ablation Studies.We further explore how the components of our
method contribute to the result. Performance was measured using
two samples on the 8 Gen 4 platform: 𝐶1: Qwen3 family configura-
tion with Workflow 2 and input data sampled from FinqaBench. 𝐶2:
BGE family configuration with Workflow 3 and input data sampled
from 2WikiQA. The baseline is “Ayo-like” static mapping, and the
results are illustrated in Tab 3.
7 Conclusion
This paper presentsHeRo, an adaptive orchestration system de-
signed to address the challenges of deploying complex agentic
RAG workflows onto heterogeneous mobile SoCs, including stage-
accelerator mapping, workload partitioning, concurrency control,
and dynamic dependency scheduling. By designing a dependency-
guided scheduler with awareness of shape, bandwidth, and hard-
ware affinity,HeRoachieves efficient resource utilization and re-
duced latency, achieving up to10 .94×speedup. Experimental results
highlightHeRo’s effectiveness in optimizing RAG inference and its
potential for broader agentic applications on personal devices in
real-world mobile usage scenarios.

HeRo: Adaptive Orchestration of Agentic RAG on Heterogeneous Mobile SoC
Llamacpp-GPUPowerserve-NPUAyo-likeScheduleHeRo
2.04.06.08.010.012.02.04.06.08.010.012.047
FinqaBenchTruthfulQAHotpotQAWikiMultihopQAFinqaBenchTruthfulQAHotpotQAWikiMultihopQALatency(s)152025303540
SpeedupLatency(s)152025303540Workflow3@8Gen4
Speedup49Workflow3@8Gen3
FinqaBenchTruthfulQAHotpotQAWikiMultihopQAFinqaBenchTruthfulQAHotpotQAWikiMultihopQALatency(s)51015202530
SpeedupLatency(s)51015202530Workflow1@8Gen4
Speedup2.04.06.08.010.012.0
2.04.06.08.010.012.0Workflow1@8Gen3
FinqaBenchTruthfulQAHotpotQAWikiMultihopQAFinqaBenchTruthfulQAHotpotQAWikiMultihopQA
2.04.06.08.010.012.02.04.06.08.010.012.0Latency(s)51015202530Workflow2@8Gen3
SpeedupLatency(s)51015202530Workflow2@8Gen4
Speedup
Figure 5: End-to-End Latency on Qwen3 Model Family. Embed model: Qwen3-Embedding-0.6B, Rerank model: Qwen3-Reranker-
0.6B, Search model: Qwen3-1.7B, Chat model: Qwen3-4B. All quantized to INT8.
Llamacpp-GPUPowerserve-NPUAyo-likeScheduleHeRo
SpeedupSpeedupFinqaBenchTruthfulQAHotpotQAWikiMultihopQAFinqaBenchTruthfulQAHotpotQAWikiMultihopQALatency(s)102030405060Latency(s)102030405060Workflow3@8Gen42.04.06.08.010.012.02.04.06.08.010.012.0Workflow3@8Gen3Latency(s)Latency(s)FinqaBenchTruthfulQAHotpotQAWikiMultihopQA
FinqaBenchTruthfulQAHotpotQAWikiMultihopQA24681012
Speedup
24681012Workflow1@8Gen4
Speedup2.04.06.08.010.012.0
2.04.06.08.010.012.0Workflow1@8Gen3
FinqaBenchTruthfulQAHotpotQAWikiMultihopQAFinqaBenchTruthfulQAHotpotQAWikiMultihopQA
2.04.06.08.010.012.02.04.06.08.010.012.0Latency(s)51015202530Workflow2@8Gen3
SpeedupLatency(s)51015202530Workflow2@8Gen4
Speedup41
Figure 6: End-to-End Latency on BGE and LLaMA3 Model Families. Embed model: bge-large-en-v1.5(0.3B), Rerank model:
bge-reranker-large(0.6B), Search model: Llama-3.2-1B, Chat model: LLaMA-3.1-8B. All quantized to INT8.

Maoliang Li, Jiayu Chen, Zihao Zheng, Ziqian Li, Xinhao Sun, Guojie Luo, Chenchen Liu, and†Xiang Chen
References
[1] Apple. 2025. A19. https://nanoreview.net/en/soc/apple-a19.
[2] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024.
BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text
Embeddings Through Self-Knowledge Distillation. arXiv:2402.03216 [cs.CL]
[3]Le Chen, Dahu Feng, Erhu Feng, Yingrui Wang, Rong Zhao, Yubin Xia, Pin-
jie Xu, and Haibo Chen. 2025. Characterizing Mobile SoC for Accelerating
Heterogeneous LLM Inference. InProceedings of the ACM SIGOPS Symposium
on Operating Systems Principles(Lotte Hotel World, Seoul, Republic of Korea)
(SOSP). Association for Computing Machinery, New York, NY, USA, 359–374.
doi:10.1145/3731569.3764808
[4]PowerServe Contributors. 2025. PowerServe:a high-speed and easy-use LLM
serving framework for local deployment. https://github.com/powerserve-project/
PowerServe.
[5] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024.
The Faiss library.
[6] Maciej Drozdowski. 2009.Scheduling for parallel processing. Vol. 18. Springer.
[7] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2023. Precise Zero-Shot
Dense Retrieval without Relevance Labels. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers).
Association for Computational Linguistics, Toronto, Canada, 1762–1777.
[8]ggml org. 2025. ggml-org/llama.cpp: LLM inference in C/C++. https://github.
com/ggml-org/llama.cpp.
[9]Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey,
Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan
Schelten, Alex Vaughan, and et al. 2024. The Llama 3 Herd of Models.
arXiv:2407.21783 [cs.AI] https://arxiv.org/abs/2407.21783
[10] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics. International Committee on Computational Linguistics, Barcelona,
Spain (Online), 6609–6625.
[11] Zhengding Hu, Vibha Murthy, Zaifeng Pan, Wanlu Li, Xiaoyi Fang, Yufei Ding,
and Yuke Wang. 2025. HedraRAG: Co-Optimizing Generation and Retrieval
for Heterogeneous RAG Workflows. InProceedings of the ACM SIGOPS 31st
Symposium on Operating Systems Principles (SOSP ’25). Association for Computing
Machinery, New York, NY, USA, 623–638. doi:10.1145/3731569.3764806
[12] Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, and
Bertie Vidgen. 2023. FinanceBench: A New Benchmark for Financial Question
Answering. arXiv:2311.11944 [cs.CL] https://arxiv.org/abs/2311.11944
[13] Joo Seong Jeong, Jingyu Lee, Donghyun Kim, Changmin Jeon, Changjin Jeong,
Youngki Lee, and Byung-Gon Chun. 2022. Band: coordinated multi-DNN in-
ference on heterogeneous mobile processors. InProceedings of the 20th Annual
International Conference on Mobile Systems, Applications and Services(2022).
235–247.
[14] Fucheng Jia, Deyu Zhang, Ting Cao, Shiqi Jiang, Yunxin Liu, Ju Ren, and Yaoxue
Zhang. 2022. CoDL: efficient CPU-GPU co-execution for deep learning inference
on mobile devices. InProceedings of the 20th Annual International Conference on
Mobile Systems, Applications and Services(New York, NY, USA, 2022). 209–221.
[15] Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdan-
bakhsh, and Vidushi Dadu. 2025. Rago: Systematic performance optimization
for retrieval-augmented generation serving. InProceedings of the 52nd Annual
International Symposium on Computer Architecture. 974–989.[16] Chaofan Li, Zheng Liu, Shitao Xiao, and Yingxia Shao. 2023. Making Large Lan-
guage Models A Better Foundation For Dense Retrieval. arXiv:2312.15503 [cs.CL]
[17] Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how
models mimic human falsehoods. InProceedings of the 60th annual meeting of
the association for computational linguistics (volume 1: long papers). 3214–3252.
[18] MediaTek. 2025. MediaTek Dimensity 9500. https://nanoreview.net/en/soc/
mediatek-dimensity-9500.
[19] Taehwan Park, Geonho Lee, and Min-Soo Kim. 2025. MobileRAG: A
Fast, Memory-Efficient, and Energy-Efficient Method for On-Device RAG.
arXiv:2507.01079 [cs.DB] https://arxiv.org/abs/2507.01079
[20] Qualcomm. 2023. Snapdragon 8 Gen 3 Mobile Platform | Our Newest Mobile
Processor | Qualcomm. https://www.qualcomm.com/smartphones/products/8-
series/snapdragon-8-gen-3-mobile-platform.
[21] Qualcomm. 2024. Snapdragon 8 Elite Mobile Platform | Qualcomm.
https://www.qualcomm.com/smartphones/products/8-series/snapdragon-
8-elite-mobile-platform.
[22] Inc. Qualcomm Technologies. 2025. QNN SDK. https://docs.qualcomm.com/
bundle/publicresource/topics/80-63442-50/introduction.html.
[23] Xin Tan, Yimin Jiang, Yitao Yang, and Hong Xu. 2025. Towards End-to-End
Optimization of LLM-based Applications with Ayo. InProceedings of the ACM
International Conference on Architectural Support for Programming Languages
and Operating Systems (ASPLOS), Vol. 2. 1302–1316.
[24] Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388 [cs.CL] https:
//arxiv.org/abs/2505.09388
[25] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li
Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah,
Ryen W White, Doug Burger, and Chi Wang. 2023. AutoGen: Enabling Next-
Gen LLM Applications via Multi-Agent Conversation. arXiv:2308.08155 [cs.AI]
https://arxiv.org/abs/2308.08155
[26] Daliang Xu, Hao Zhang, Liming Yang, Ruiqi Liu, Gang Huang, Mengwei Xu, and
Xuanzhe Liu. 2025. Fast On-device LLM Inference with NPUs. InInternational
Conference on Architectural Support for Programming Languages and Operating
Systems (ASPLOS).
[27] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RECOMP: Improving retrieval-
augmented LMs with context compression and selective augmentation. InThe
Twelfth International Conference on Learning Representations.
[28] Zhenliang Xue, Yixin Song, Zeyu Mi, Xinrui Zheng, Yubin Xia, and Haibo Chen.
2024. PowerInfer-2: Fast Large Language Model Inference on a Smartphone.
arXiv:2406.06282 [cs.LG] https://arxiv.org/abs/2406.06282
[29] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InConference on Empirical
Methods in Natural Language Processing (EMNLP).
[30] Weiping Yu, Ningyi Liao, Siqiang Luo, and Junfeng Liu. 2025. RAGDoll: Efficient
Offloading-based Online RAG System on a Single GPU. arXiv:2504.15302 [cs.DC]
https://arxiv.org/abs/2504.15302
[31] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang,
Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou.
2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through
Foundation Models. arXiv:2506.05176 [cs.CL] https://arxiv.org/abs/2506.05176
[32] Jiahao Zhou, Chengliang Lin, Dingji Li, Mingkai Dong, and Haibo Chen. 2025.
GRATING: Low-Latency and Memory-Efficient Semantic Selection on Device.
arXiv:2510.15620 [cs.LG] https://arxiv.org/abs/2510.15620