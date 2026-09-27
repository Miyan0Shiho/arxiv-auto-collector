# Efficient Iterative Retrieval with Heterogeneous Batching

**Authors**: Dohyun Park, Hubertus Franke, Daniel G. Waddington, Swaminathan Sundararaman, Yongjoo Park

**Published**: 2026-09-21 20:55:10

**PDF URL**: [https://arxiv.org/pdf/2609.25405v1](https://arxiv.org/pdf/2609.25405v1)

## Abstract
Modern information retrieval increasingly employs both embedding and generative models to handle complex queries. However, current serving systems suffer from low throughput and poor GPU utilization because they execute these models in isolation. Coarse-grained partitioning, such as dedicating GPUs to specific tasks, fails to adapt to dynamic workloads and creates computational "bubbles". To address these, we present Orthrus, a serving system that performs heterogeneous batching within a unified inference loop. The primary challenge lies in unifying embedding and generation workloads with conflicting computational patterns while optimizing batch composition for high performance. Orthrus addresses these challenges through chunked embedding with incremental pooling and by adjusting batch composition in a workload-aware manner. Evaluation on four A100 GPUs shows that, relative to baseline deployments, Orthrus achieves 1.28$\times$--4.52$\times$ higher throughput on controlled workloads and up to 55.8% lower end-to-end p99 latency on an iterative-RAG benchmark. We release our code at https://github.com/illinoisdata/Orthrus .

## Full Text


<!-- PDF content starts -->

Efficient Iterative Retrieval with Heterogeneous Batching
Dohyun Park1, Hubertus Franke2, Daniel G. Waddington2,
Swaminathan Sundararaman2,Yongjoo Park1
1University of Illinois Urbana-Champaign2IBM Research
{dohyunp2,yongjoo}@illinois.edu {frankeh,daniel.waddington,swami}@ibm.com
Abstract
Modern information retrieval increasingly
employs both embedding and generative
models to handle complex queries. How-
ever, current serving systems suffer from
low throughput and poor GPU utiliza-
tion because they execute these models
in isolation. Coarse-grained partition-
ing, such as dedicating GPUs to specific
tasks, fails to adapt to dynamic work-
loads and creates computational “bub-
bles”. To address these, we presentOr-
thrus, a serving system that performshet-
erogeneous batchingwithin a unified infer-
ence loop. The primary challenge lies in
unifying embedding and generation work-
loads with conflicting computational pat-
terns while optimizing batch composition
for high performance.Orthrusaddresses
these challenges through chunked embed-
ding with incremental pooling and by ad-
justing batch composition in a workload-
aware manner. Evaluation on four A100
GPUs shows that, relative to baseline
deployments,Orthrusachieves 1.28×–
4.52×higher throughput on controlled
workloads and up to 55.8% lower end-
to-end p99 latency on an iterative-RAG
benchmark. We release our code athttps:
//github.com/illinoisdata/Orthrus.
1 Introduction
Modern retrieval techniques (Asai et al., 2023;
Gao et al., 2023; Jiang et al., 2023b; Shao
et al., 2023) integrate both embedding and
generation models to identify information gaps
and retrieve documents iteratively. Consider a
user query: “Does the Gold plan cover my gym
membership?” This query is firstembedded
into a vector (Wang et al., 2024; Izacard et al.,
2021) to search a vector database (Malkov
and Yashunin, 2020; Li et al., 2025b). If
a retrieved document states that “The Goldplan covers Category B activities,” the sys-
tem clarifies the ambiguity by letting agen-
erativemodel (Asai et al., 2023; Shao et al.,
2023) formulate a follow-up query—“What are
Category B activities?”—which is then em-
bedded to retrieve supplementary documents.
This cycle continues until sufficient context
has been gathered. To accelerate this, serving
systems must be explicitly designed to handle
mixed embedding/generation workloads.
Unfortunately, existing systems suffer from
low throughput and poor GPU utilization
when handling mixed workloads. Serving
frameworks like vLLM (Kwon et al., 2023) are
designed to execute embedding and generation
models in isolation. Simply running two serv-
ing instances (i.e., OS processes) on the same
GPU—one for each task, as in Fig. 1(b)—is
inefficient in current architectures (Yu et al.,
2022). In a multi-GPU setup, resources can
be statically partitioned for embedding and
generation (Fig. 1(a)), but this approach cre-
ates two major issues: (i) accurately pre-
dicting the optimal resource ratio is difficult,
and (ii) dynamic repartitioning requires costly
model reloading. Even with perfect workload
foresight, coarse-grained GPU partitioning re-
mains suboptimal because embedding tasks
are typically compute-bound, whereas genera-
tion is memory-bound (Agarwal et al., 2025).
We hypothesize thathomogeneous batchingis
a fundamental bottleneck in iterative retrieval.
Core IdeaWe presentOrthrus, a serv-
ing system optimized for mixed embedding
and generation workloads. By unifying these
distinct compute patterns into a shared ab-
straction,Orthrusenablesheterogeneous
batchingby executing embedding and gen-
eration requests together in a shared model
execution pass. Unlike homogeneous batching
arXiv:2609.25405v1  [cs.AI]  21 Sep 2026

Time 0
Time 1
Time 2
Time 3Queue GPU-1 GPU-2
×4×14 empty empty
×1×11
×0×8
×0×5 emptyQueue GPU-1 GPU-2
×4×14 empty empty
×1×11
×0×8
×0×2Queue GPU-1 GPU-2
×4×14 empty empty
×2×10
×0×6
×0×0
(a) Homogeneous batching
w/ GPU-level split(b) Homogeneous batching
within GPU(c) Heterogeneous batching
within GPU (ours)
Figure 1: Example iterative retrieval. The workload arrives at Time 0. Neither (a) GPU-level splitting
nor (b) homogeneous batching within each GPU completes all requests by Time 3; (c) our heterogeneous
batching completes them all.
(Fig. 1(b)), this approach co-schedules tasks
to maximize resource utilization (Fig. 1(c))
while eliminating the overhead of concurrent
server instances. This architecture enables
fine-grained load balancing and generalizes to
multi-GPU settings, where every node can
serve mixed workloads. Importantly,Or-
thrusrequires no additional model training,
making it applicable to existing deployments.
ChallengeWe address two primary chal-
lenges.(Heterogeneous Workloads)To
maximize efficiency, it is desirable to execute
embedding and generation requests together in
the same model execution pass. Forming such
heterogeneous passes is non-trivial because
the requests exhibit conflicting computational
patterns. A generation decode step requires
relatively little compute to predict the next to-
ken, whereas embedding requires substantial
compute to aggregate internal states across
a full input. Na¨ ıvely batching them with-
out chunking forces lightweight work to wait
for heavier requests, causingbubbles.(Batch
Composition)The ratio of embedding to
generation requests in a batch heavily im-
pacts user-perceived performance, yet finding
the optimal ratio is difficult. FCFS schedul-
ing causes head-of-line blocking for specific re-
quest types. Static heuristics (e.g., a 1:2 ratio)
are also brittle because the number of tokens
generated varies per request, so the optimal re-
source mix shifts dynamically and makes fixed
ratios suboptimal.
Our ApproachWe employ the following
techniques.(Chunked Embedding for
Compute Balance)To balance embedding
and generation work within a batch, we adapt
incremental computation (Sheoran et al.,
2023; Hellerstein et al., 1997) to embedding in-
ference. We call this execution strategychun-ked embedding: the input is processed in fine-
grained token chunks across scheduling itera-
tions, whileincremental poolingmaintains the
cross-chunk state needed to produce the final
embedding (§3.3). The unified runner sched-
ules embedding chunks alongside generation
decode steps to balance compute loads and
minimizebubbles.
(Intra-Batch Scheduling)Our design is
governed by a central principle:retrieval ends
only when all embedding and generation re-
quests are fully processed.To achieve this,
Orthrusdynamically adjusts batch compo-
sition using acost-aware queue length. The
scheduler measures each backlog by its total
remaining token count rather than its raw re-
quest count, allocating capacity in proportion
to the outstanding token work in each queue.
Our technique is practical and delivers high
performance. We implemented our prototype
system (calledOrthrus) on top of vLLM ver-
sion 0.92, without modifying memory manage-
ment or model specification. Evaluation on
four NVIDIA A100 GPUs with state-of-the-
art open-weight models shows thatOrthrus
delivers 1.28×–4.52×higher throughput than
GPU-level splits. Chunked and single-pass
embeddings are equivalent up to floating-point
reduction-order effects (Tab. 7).
This work focuses on compute patterns in-
side GPU kernels, orthogonal to parameter
sharing (Vasani et al., 2025; Gauthier, 2024)
and multi-adapter scheduling (Sheng et al.,
2024a; Chen et al., 2024).
2 Background
2.1 Modern RAG Workflow
Modern RAG pipelines combine embedding-
based retrieval with language model genera-
tion (Lewis et al., 2020). Recent techniques

employ iterative retrieval, where embedding
and generation alternate to refine queries dy-
namically (Asai et al., 2023; Jiang et al.,
2023b; Shao et al., 2023; Trivedi et al., 2023).
A single prompt thus decomposes into a se-
quence of embedding and generation requests,
creating a mixed workload.Orthruschanges
model-side embedding inference but not the
vector-search backend, which is orthogonal to
our serving design.
2.2 Computational Characteristics of
Embedding and Generation
Embedding and generation requests differ fun-
damentally in their execution patterns and re-
source utilization.
GenerationA generation request produces
a variable-length output autoregressively. It
executes in two phases:prefillanddecode. The
prefillphase processes the input prompt in a
single parallelforwardpass. Thedecodephase
then generates tokens one at a time: each
step reads the key-value (KV) cache (Pope
et al., 2023) containing previous tokens’ repre-
sentations, computesattention, and appends
the new token’s KV entries. This decode
phase ismemory-bound, as each step transfers
large KV cache while performing relatively few
arithmetic operations (Sheng et al., 2023).
EmbeddingAn embedding request struc-
turally resembles theprefillphase of gener-
ation, but follows with a pooling operation
rather than entering a decode loop. The model
processes input tokens through transformer
layers (Vaswani et al., 2017) and aggregates
the final hidden states via pooling (typically
mean pooling or last-token pooling) to pro-
duce a fixed-length vector (Wang et al., 2024;
Shen et al., 2024). A vector database indexes
these vectors for similarity search (Malkov and
Yashunin, 2020; Johnson et al., 2021; Wang
et al., 2021a; Chockchowwat et al., 2023; Li
et al., 2025a). Because embedding execution
is a single dense forward pass, it iscompute-
bound(Ivanov et al., 2021), contrasting with
memory-bound decoding.
Resource utilization analysisTo quan-
tify these compute differences, we profiled
GPU utilization under pure embedding and
pure generation workloads using Mistral-7BTable 1: GPU utilization under pure embedding
and pure generation workloads (Mistral-7B on
A100-40GB, 128-token inputs, 512-token outputs
for generation).
GPU Compute (%) GPU Memory (%)
Workload Mean Max Mean Max
Embedding 85.398.0 36.5 37.1
Generation 75.5 98.0 90.590.5
on an A100-40GB GPU. Tab. 1 summarizes
the results. Neither workload fully exploits
both resources, suggesting an opportunity to
co-schedule them for improved efficiency.
Parameter sharingState-of-the-art em-
bedding models are increasingly derived from
generative models via LoRA fine-tuning (Li
et al., 2023; Wang et al., 2024; Meng et al.,
2024; Ma et al., 2024; BehnamGhader et al.,
2024). Other approaches train unified models
for both tasks (Muennighoff et al., 2025; Zhang
et al., 2025; Li et al., 2024; Tang et al., 2025).
Orthrusis optimized for shared-backbone
embedding and generation models. For sep-
arate backbones, our extension retains IBS
across model-specific queues but does not com-
bine the models in one forward pass (Tab. 6).
2.3 Batching and Scheduling
Modern LLM serving systems employ
iteration-level scheduling(Yu et al., 2022;
Kwon et al., 2023), re-evaluating batch
composition at every token generation step.
Within each iteration, the system executes
ahomogeneous batchconsisting exclusively
of generation requests, potentially mixing
chunked prefill and decode phases (Agrawal
et al., 2024).Orthrusextends this chunked
execution to embeddings via incremental
pooling. Recent work targets scheduling
policies (Wu et al., 2024; Sheng et al., 2024b),
kernel efficiency and GPU co-location (Wang
et al., 2021b; Xia et al., 2023; Strati et al.,
2024), KV cache management (Qin et al.,
2025; Yu et al., 2025; Gao et al., 2025;
Agarwal et al., 2025; Xia et al., 2026),
storage efficiency (Shah et al., 2026), and
prefill–decode disaggregation (Zhong et al.,
2024; Hu et al., 2024). Multi-tenant LoRA
systems (Sheng et al., 2024a; Chen et al.,
2024) improve adapter throughput but as-
sume generation-only workloads.Orthrus
schedules embedding and generation jointly

Algorithm 1:Heterogeneous Batching
Inference Loop
Input:Embedding queueQ e, generation queueQ g
whileQ e̸=∅orQ g̸=∅orin-flight requestsdo
B ←Schedule(Q e, Qg) ;// Mix embed & gen
H←Forward(B) ;// Attention + FFN
// Parallel output processing
beginin parallel
Re←IncPooler(H[emb idx]);
Rg←Sampler(H[gen idx]);
Emit(R e,Rg) ;// Return completed
while preserving arrival order within each
request type (§4).
3 Unified Runner
This section presents the design ofOrthrus’s
unified runner, which enables heterogeneous
batching of embedding and generation re-
quests within a single inference loop.
3.1 High-Level Workflow
Our runner is designed for iteration-level
scheduling (Yu et al., 2022), aiming for seam-
less integration with vLLM’s (Kwon et al.,
2023) open-sourced model management and
server. Yet, our approach differs from existing
approaches in that both embedding and gen-
eration requests are processed using the same
runner, which we achieve by consolidating dis-
tinct embedding and generation runners into
one in a structured way.
Our runner operates in three blocking
stages: (1)schedule, (2)forward, and (3)
emit(Fig. 2).Scheduledetermines batch
composition, including generation prefills, de-
codes, or embedding.Forwardapplies atten-
tion and feed-forward layers, identical across
request types except for input length.Emit
diverges: generation samples the next token,
while embedding performsincremental pooling
to compute the final embedding vector.
3.2 Batch Unit
A batch inOrthruscontains three types of
items: (1)embedding chunks, which are seg-
ments of an embedding request’s input se-
quence, (2)generation prefill chunks, which
are segments of a generation request’s prompt,
and (3)generation decode tokens, which are
single tokens generated during autoregressive
decoding. We call execution of an embedding
request across multiple such chunkschunked
embedding. All three types share the sameforward pass through the transformer layers,
differing only in their input lengths and out-
put handling. Embedding and prefill chunks
process multiple tokens per request, while de-
code tokens process exactly one token per re-
quest. This unified representation allows the
scheduler to mix request types freely within
a single batch, filling available token capacity
while preserving the workload-proportional al-
location described in§4.2. The scheduler out-
puts batches sorted by request type, ensur-
ing that items of the same type are contigu-
ous in memory. This layout eliminates mem-
ory movement when invoking the output ker-
nels: the incremental pooler, the sampler, and
the prefill handler each operate on a contigu-
ous slice of the hidden state tensor, avoiding
gather operations or intermediate copies.
3.3 Incremental Pooler
Incremental pooling is the cross-chunk aggre-
gation used by chunked embedding.
Consider an embedding request with input
sequence ofntokens. Under chunked embed-
ding, the system splits the sequence intoK
chunks of sizesc 1, c2, . . . , c KwherePK
k=1ck=
n. At iterationk, the forward pass produces
hidden statesH(k)∈Rck×dfor thek-th chunk,
wheredis the hidden dimension.
Pooling HeadsMean and weighted-mean
pooling share the same streaming formulation.
For weightsw k,j, the pooler maintains
S(k)=S(k−1)+ckX
j=1wk,jH(k)
j, S(0)=0,(1)
W(k)=W(k−1)+ckX
j=1wk,j, W(0)= 0,(2)
and emitsS(K)/W(K)after the final chunk.
Mean pooling is the special casew k,j= 1. Po-
sitional heads retain only the selected hidden
state: CLS pooling usesH(1)
1, while last-token
pooling usesH(K)
cK. These additive and po-
sitional heads requireO(d) state per request.
An arbitrary deterministic head remains com-
patible by bufferingA(k)= [A(k−1);H(k)] and
applying the head after the final chunk, at an
O(nd) memory cost. For all of these heads,
incremental and single-pass pooling are alge-
braically equivalent, although floating-point

EmbeddingRunner
(over entire passage)
schedule forwardglobal
pool
GenerationRunner
(over one or more tokens)
schedule forwardn/a
sampleOR
(a) Exclusive iterations
for embedding/generationPrefillRunner
(over prompt chunks)
schedule forwardemit
KV
DecodeRunner (Decode loop)
schedule forward sampleKV cache transfer
(b) Disaggregated prefilling
(prefill GPU→decode GPU)UnifiedRunner
(over one or more tokens)
schedule forwardincremental
pool
n/a
sample
→indicatesblockingtransition
between sub-steps
(c) Unified iteration for
heterogeneous batching
Figure 2: Runner structure comparison. (a) Task-separated runners use global pooling for embedding or token
sampling for generation. (b) Phase-separated prefill and decode runners transfer the KV cache between GPUs. (c)
Our unified runner supports both output paths in each iteration; incremental pooling allows embedding chunks
to execute alongside generation requests.
non-associativity may introduce small numer-
ical differences.
4 Intra-Batch Scheduling
We present Intra-Batch Scheduling (IBS), a
scheduling algorithm that dynamically allo-
cates capacity proportional to the workload
mix. We first define three requirements for
efficient scheduling (§4.1), then detail the IBS
design (§4.2).
4.1 Requirements
We define three requirements: (1)Propor-
tional allocation—capacity should match the
instantaneous embedding-to-generation de-
mand ratio; (2)Per-type FCFS—requests of
the same type are served in arrival order,
though types may interleave; (3)No head-
of-line blocking—neither request type should
starve the other.
4.2 Scheduler Design
Algorithm 2 presents our intra-batch schedul-
ing algorithm, which satisfies all three require-
ments. The scheduler first reserves capacity
for in-flight generation requests that are in the
decode phase. These requests have allocated
KV cache entries and must continue execution
to avoid KV cache movement. This phase mir-
rors vLLM scheduling and uses the same swap-
in and swap-out mechanisms.
The scheduler then fills the remaining bud-
get by alternating between the embedding and
generation queues.
Why chunking improves packing.LetR
be the residual token budget after reserving
capacity for in-flight decode requests. With-
out chunking, an embedding request of lengthAlgorithm 2:Intra-Batch Scheduling
Input:Embedding queueQ e, generation queueQ g,
token budgetB, persistent creditsa e, ag
Output:Scheduled batchB
(B, u)←ScheduleRunning(B);
B←B−u;
// Proportionally select from both queues
we←P
r∈Qetokens(r);w g←P
r∈Qgtokens(r);
ifQe=∅then
ae←0
ifQg=∅then
ag←0
whileB >0and(Q e̸=∅orQ g̸=∅)do
ae←ae+we;a g←ag+wg;
ifQg=∅or(Q e̸=∅anda e≥ag)then
(I, u)←ScheduleOne(Q e, B);
B ← B ∪ I;B←B−u;
ae←ae−(w e+wg);
else
(I, u)←ScheduleOne(Q g, B);
B ← B ∪ I;B←B−u;
ag←ag−(w e+wg);
//a e, agpersist across iterations; reset above
when a queue drains
returnB
eis admitted only whene≤R; otherwise,
the remainingRtokens go unused.Sched-
uleOneinstead admits embedding chunks of
at mostCtokens. When token capacity is
the limiting constraint, a backlogged embed-
ding queue therefore leaves fewer thanCto-
kens unused once no additional chunk fits. Let
weandw gbe the sums of the remaining to-
ken counts in the embedding and generation
queues, respectively. The algorithm main-
tains accumulatorsa eanda gthat track the
scheduling credit of each queue. The credits
persist across scheduling iterations while their
queues remain nonempty. At each selection
step, the scheduler increments both accumu-
lators by their corresponding weights and se-
lects a request from the queue with the larger
accumulator, decrementing the selected accu-
mulator byw e+wg. If one queue becomes
empty, the scheduler continues selecting from
the remaining queue until the token budget is

exhausted.
The accumulator mechanism ensures that
the ratio of scheduled embeddings to gener-
ations converges tow e:wg, achieving pro-
portional allocation that matches the instan-
taneous workload mix. When only embed-
dings are queued (w g= 0), all capacity goes
to embeddings, and vice versa. Finally, by
maintaining separate queues and alternating
between them proportionally, the scheduler
avoids head-of-line blocking. Short embedding
requests are never stuck behind long genera-
tion requests, and each type makes progress
according to its share of the workload. Assume
each nonempty queue has positive weight, its
head request or next chunk can eventually
be admitted, and each iteration completes in
bounded time. Persistent credits then prevent
scheduler starvation: a backlogged queue that
is not selected continues accumulating credit
until it is chosen. For fixed weights, its allo-
cation differs from its ideal proportional share
by at most one scheduling opportunity. This
guarantee addresses scheduler-induced block-
ing. It does not bound queuing delay when
offered load exceeds capacity.
5 Evaluation
We evaluateOrthrus, an embedding-
generation serving system that combines
heterogeneous batching with Intra-Batch
Scheduling (IBS). The setup is described in
§5.1. Our results show that:
•High Throughput:1.28×–4.52×higher
throughput than GPU-level splits on syn-
thetic workloads; 2.4×on a RAG bench-
mark with 4 GPUs (§5.2).
•Low Latency:9% lower p99 generation
and 16% lower p99 embedding latency; up
to 55.8% lower p99 on RAG (§5.3).
•High GPU Utilization:41 pp higher av-
erage utilization, completing mixed work-
loads 43% faster (§5.4).
•Negligible Overhead:At saturation,
matches dedicated-engine throughput for
both workload types (§5.5).
•Generalizability:Validated on 3 models
with LoRA ranks up to 64 (§5.6).Table 2: System configurations evaluated.Or-
thrusis the only configuration that combines
same-GPU placement, heterogeneous batching,
and proportional scheduling (IBS).
Config.Model
PlacementBatching Scheduling
Dedicated Task-specific GPUs Homo FCFS
Disagg-PD Phase-specific GPUs Homo FCFS
Unified-Homo Same GPU Homo FCFS
Unified-Hetero Same GPU Hetero FCFS
Orthrus Same GPU Hetero IBS
5.1 Experimental Setup
MethodsWe consider four baselines(Bs),
varying model placement, batching, and
scheduling (Tab. 2).B1-Dedicatedruns
complete embedding and generation requests
on disjoint, task-specific GPU pools (e.g.,
Dedicated 1:3assigns 1 GPU to embedding
and 3 to generation);B2-Disagg-PDdisag-
gregates generation prefill and decode across
phase-specific GPU pools (Zhong et al., 2024);
embedding requests complete after pooling
on the prefill GPU;B3-Unified-Homoco-
locates both types on the same GPU with
homogeneous batching;B4-Unified-Hetero
adds heterogeneous batching but uses FCFS
scheduling. Finally,Orthrus (ours)com-
bines same-GPU placement, heterogeneous
batching, and IBS. Unless otherwise specified,
we report the median value across three inde-
pendent runs.
Workloads, Models, and HardwareWe
used Mistral-7B (Jiang et al., 2023a) as
the base model with e5-mistral-7b-instruct
LoRA (Wang et al., 2024) for embeddings
(more models in§5.6). Synthetic workloads
use 128-token embeddings and 128/512-token
generation (prompt/output). For realistic
evaluation, we used Iter-RetGen (Shao et al.,
2023) on 2WikiMultihopQA (Ho et al., 2020)
(avg. 500-token prompts, avg. 3000-token de-
codes, 1–10 documents per query). Experi-
ments ran on up to four A100 40GB GPUs
with round-robin request distribution.
ForOrthrus, Unified-Homo, and Unified-
Hetero, embedding and generation requests
are co-located on a single GPU using LoRA
adapters. For Dedicated, embeddings are by
e5-mistral-7b-instruct; Mistral-7B serves gen-
eration. We also study scheduling-parameter
sensitivity (§A.2) and chunked embedding and
incremental pooling (§A.3).

0 0.5 1020406080100
Ratio of Generation RequestsThroughput (req/s)Dedicated 2:2 Dedicated 1:3 Dedicated 3:1 Generation Requests
Unified-Homo Unified-Hetero Orthrus Embedding Requests
(a) Overall Throughput0200GPU0
0200
GPU1
0200# of Active requestsGPU2
0 60 1200200
Time (s)GPU3
(b) Active Requests over time
(Dedicated 1:3)0200
GPU0
0200
GPU1
0200# of Active requestsGPU2
0 60 1200200
Time (s)GPU3
(c) Active Requests over time
(Orthrus)
Figure 3: Throughput and GPU utilization comparison. (Fig. 3a) compares throughput of existing
serving techniques vs our heterogeneous batching,Orthrus(see Tab. 5 for absolute values). (Fig. 3b,
Fig. 3c) shows request timelines for Dedicated 1:3andOrthrusunder a 75% generation / 25% embedding
workload. Each timeline reports the number of active requests per GPU.Orthrusfully utilizes all GPUs,
while Dedicated 1:3leaves some underutilized.
5.2 High Throughput
We evaluate whether heterogeneous batching
improves throughput compared to static GPU
partitioning across varying workload ratios.
Controlled WorkloadsWe use fixed
embedding-to-generation ratios as controlled
stress tests that isolate system behavior under
known workload compositions. We evaluated
throughput across different embedding-to-
generation ratios with 512 clients over 180
seconds. Fig. 3a reports combined through-
put; Tab. 5 shows per-type throughput.
Dedicated deployments underutilize GPUs
when workload ratios diverge from the static
split, while Unified-Homo suffers under
balanced workloads. Across all evaluated
workload ratios,Orthrusachieved higher
combined throughput than every static GPU
partition. Fig. 3b and Fig. 3c illustrate
thatOrthrusfully utilizes all GPUs, while
Dedicated 1:3leaves the embedding GPU
underutilized.
RAG BenchmarksTo complement the
controlled workloads, we evaluateOrthrus
on a benchmark-derived Iter-RetGen pipeline.
We exclude vector-search and network latency
to isolate model-side serving. Prior RAG stud-
ies report retrieval on a millisecond scale com-
pared with multi-second generation (Jin et al.,
2024; Wang et al., 2023; Trivedi et al., 2023).
Each query alternates embedding, retrieval,
and generation across multiple iterations, pro-Table 3: E2E request throughput (req/s) on
Iter-RetGen (Shao et al., 2023) with 2WikiMul-
tihopQA (Ho et al., 2020). The workload natu-
rally results in variable-length prompts (avg 500,
max 2000 tokens), long decode lengths (avg 3000,
max 4000 tokens), and multi-document retrieval
(1–10 documents per query).Orthrusachieves
the highest throughput across all configurations.
method 1 GPU 2 GPUs 4 GPUs
Embed Gen Embed Gen Embed Gen
Dedicated N/A N/A 0.08 0.11 0.119 0.141
Unified-Homo 0.067 0.08 0.126 0.148 0.252 0.296
Disagg-PD N/A N/A 0.118 0.14 0.276 0.327
Orthrus (ours) 0.075 0.088 0.142 0.166 0.288 0.336
ducing variable-length prompts and long de-
codes while retrieving 1–10 documents per
query. Tab. 3 shows thatOrthrusachieves
the highest throughput at every GPU count,
including the single-GPU setting, where disag-
gregation is infeasible. At four GPUs, its ag-
gregate throughput is 0.624 requests/s, com-
pared with 0.548 for Unified-Homo, 0.603 for
Disagg-PD, and 0.260 for Dedicated.
5.3 Low Latency with IBS
We evaluate whether IBS reduces latency by
preventing head-of-line blocking, where long-
running generation requests stall shorter em-
bedding requests under FCFS scheduling.
Controlled WorkloadsWe submitted
1,000 generation requests followed by 1,000
embedding requests at time 0, simulating
worst-case head-of-line blocking. Measured
from submission through completion, in-
cluding queueing,Orthrusreduced p99

020 40 60 80100050100150
Latency (s)# of Requests020 40 60 80100
Latency (s)020 40 60 80100
Latency (s)020 40 60 80100
Latency (s)Generation Requests Embedding Requests
(a) Dedicated 1:3 (b) Unified-Homo (c) Unified-Hetero (d)Orthrus
Figure 4: Latency distributions with various serving methods. Curves farther to the left indicate lower
latencies. The workload consists of 1,000 generations followed by 1,000 embeddings, all submitted at time
0.Orthrusachieves 9% lower p99 generation latency than Dedicated, and up to 16% lower embedding
latency than other configurations.
Table 4: E2E p99 latency (s) on Iter-RetGen (Shao
et al., 2023) with 2WikiMultihopQA (Ho et al.,
2020). We measure p99 latency per-query end-to-
end (including all retrieval iterations).Orthrus
achieves the lowest latency across all GPU counts.
MethodE2E p99 Latency (s)
1 GPU 2 GPUs 4 GPUs
Dedicated N/A 960.1 1342.0
Disagg-PD N/A 602.6 617.0
Unified-Homo 673.4 676.0 700.6
Orthrus (ours) 573.9 600.5 593.4
generation latency by 9% over Dedicated (89 s
→81 s) and p99 embedding latency by 16%
over the other schedulers (87 s→73 s). In
contrast, Unified-Homo and Unified-Hetero
both suffered from embedding requests queu-
ing behind generations. We further evaluate
latency robustness across workload ratios and
decode lengths in§A.4.
RAG BenchmarksTab. 4 shows thatOr-
thrusachieved the lowest latency across all
GPU counts: 14.8% lower than Unified-Homo
on 1 GPU (573.9s vs. 673.4s), comparable to
Disagg-PD on 2 GPUs (600.5s vs. 602.6s),
and 55.8% lower than Dedicated on 4 GPUs
(593.4s vs. 1342.0s).Orthrusavoids both
static partitioning bottlenecks and cross-GPU
KV cache transfer.
5.4 High GPU Utilization
We evaluate whether heterogeneous batch-
ing improves GPU utilization over static
partitioning under shifting workload ratios.
Across three phases (10%/50%/90% embed-
ding),Orthrusachieved 79% average utiliza-
tion versus 38% for Dedicated 1:3, completing
each phase 43% faster by balancing load across
GPUs. Static partitioning underperformed:Table 5: Generalizability analysis comparing ab-
solute throughput (requests/sec) across different
models and request mixes. We report embedding
and generation throughput separately for three ra-
tios (embedding:generation). Dedicated runs sep-
arate models on a 1:3 GPU split.Orthrusco-
locates both tasks on every GPU.
Model LoRA Ratio Embed Gen Embed Gen
Dedicated Orthrus
Mistral 7B e5-mistral9:1 15.64 1.49 58.80 7.00
5:5 12.54 11.97 25.32 26.24
1:9 5.10 29.56 4.2039.10
Qwen2 7BSynthetic
Rank 329:1 13.23 1.72 40.90 4.40
5:5 10.72 10.08 26.80 25.93
1:9 2.81 27.76 6.20 48.23
LLaMA3.1 8BSynthetic
Rank 649:1 13.43 1.32 56.46 5.90
5:5 10.18 9.95 34.60 37.50
1:9 3.06 25.63 4.23 47.32
the dedicated embedding GPU was underuti-
lized during generation-heavy phases and over-
loaded during embedding-heavy phases.
5.5 Minimal Overhead
Fig. 6 compares embedding- and generation-
only workloads on a single GPU with 1–256
clients.Orthrusreaches dedicated vLLM
generation throughput once the workload sat-
urates (≥64 clients), yet it trails at lower
concurrency. Its embedding throughput is
comparable to or higher than the correspond-
ing dedicated configuration; the lower abso-
lute throughput of both LoRA curves reflects
adapter overhead. These results show that
heterogeneous batching preserves saturated
per-task throughput while enabling GPU shar-
ing. We study the contribution of chunked em-
bedding and incremental pooling in§A.3.
5.6 Generalization
We evaluate whether heterogeneous batch-
ing generalizes across model architectures and
LoRA adapter ranks. We tested Mistral 7B,

050100
050100
050100
0 60 120 180050100GPU Utilization (%)
Time (s)Phase 1
1 Emb:9 GenPhase 2
5 Emb:5 GenPhase 3
9 Emb:1 GenGeneration Model Utilization Embedding Model Utilization OrthrusUtilization
(a) Homogeneous batching with Dedicated 1:3050100
050100
050100
0 60 120 180050100
Time (s)Phase 1
1 Emb:9 GenPhase 2
5 Emb:5 GenPhase 3
9 Emb:1 Gen
(b) Heterogeneous batching withOrthrus
Figure 5: GPU utilization under three workload phases. Dedicated 1:3(Fig. 5a) underutilizes GPUs (avg.
38%), whileOrthrus(Fig. 5b) sustains 79% utilization and completes each phase 43% faster.
1 64 128 256010203040
# clientsThroughput (req/s)
(a) Embedding throughputDedicated (LoRA) Dedicated (combined)
Orthrus(combined) Orthrus(LoRA)
1 64 128 256051015
# clients
(b) Generation throughput
Figure 6: Overhead analysis ofOrthruson a sin-
gle GPU. At≥64 clients,Orthrusreaches sim-
ilar saturated generation throughput to dedicated
vLLM. Embedding throughput is comparable to
or higher than the corresponding dedicated config-
uration; the gap between the combined-model and
LoRA curves reflects adapter overhead.
Qwen2 7B (Yang et al., 2024), and LLaMA3.1
8B (Grattafiori et al., 2024) with LoRA ranks
up to 64 and embedding-to-generation ratios
of 9:1, 5:5, and 1:9. As shown in Tab. 5,Or-
thrusachieved higher combined embedding-
plus-generation throughput than Dedicated 1:3
across all configurations. Per-type throughput
was also generally higher; the exception was
Mistral 7B at 1:9, where embedding through-
put decreased from 5.10 to 4.20 requests/s
while generation throughput increased from
29.56 to 39.10 requests/s.
Separate Models.We also evaluateOr-
thruswith separate models (OPT-1.3B for
generation and GTR-T5-XL for embeddings)
on a single A40 GPU.Orthrusco-serves both
models with a 0.9 GPU-memory cap, while
the baseline uses two vLLM instances capped
at 0.45 each. We use the Iter-RetGen-styleTable 6: Separate-model performance on one A40
GPU. Gain denotes a throughput increase or la-
tency reduction relative to 2×vLLM; positive val-
ues favorOrthrus.
Metric Workload 2×vLLMOrthrusGain
Throughput
(req/s)Emb. 29.736.5 +22.9%
Gen. 28.233.9 +20.2%
Avg. latency
(ms)Emb. 187.0151.8 +18.8%
Gen. 4337.13561.5 +17.9%
p95 latency
(ms)Emb. 421.4318.6 +24.4%
Gen.4875.35059.7 -3.8%
2WikiMultihopQA workload from§5.2.
Orthrusimproves embedding and genera-
tion throughput by 22.9% and 20.2%, respec-
tively, and reduces both average latencies and
embedding p95 latency. The 3.8% increase in
generation p95 latency is outweighed by the
17.9% reduction in mean generation latency.
6 Conclusion
This work presentsOrthrus, a serving sys-
tem that co-serves embedding and generation
within a unified model runner through het-
erogeneous batching. Unlike existing systems,
Orthrusintegrates embedding into the same
scheduling and execution path as generation.
By combining chunked embedding with incre-
mental pooling and workload-aware schedul-
ing,Orthrusachieves higher throughput and
lower latency under mixed workloads, enabling
efficient serving for knowledge-intensive LLM
applications.

Limitations
Our evaluation scalesOrthrusby running
one model replica per GPU, with up to four
A100 GPUs. Deployments with more replicas
may exhibit different load-balancing behavior
and workload skew across GPUs, which can
affect utilization and tail latency. Thus, our
experiments do not establish how the observed
gains scale beyond four GPUs.
Orthrusdoes not currently implement
support for a single model sharded across mul-
tiple GPUs. The separate-model experiment
likewise considers only two models that fit to-
gether on one A40. Models that require tensor
or pipeline parallelism introduce communica-
tion and synchronization within each iteration,
which may interact differently with heteroge-
neous batching and scheduling.
Our experiments use open-weight models
with fewer than 10 billion parameters. Larger
models have greater memory demands and
may exhibit different compute-to-memory ra-
tios and batch-capacity constraints. Although
the results are consistent across the tested
model families, they do not establish thatOr-
thrusprovides the same throughput and la-
tency benefits for larger models.
Acknowledgments
This work was supported in part by the IBM-
Illinois Discovery Accelerator Institute and by
the National Science Foundation under grants
#2312561 and #2440498. We used Delta and
DeltaAI at the National Center for Super-
computing Applications (NCSA) through al-
location CIS240661 from the ACCESS pro-
gram, which is supported by NSF grants
#2138259, #2138286, #2138307, #2137603,
and #2138296.
References
Shubham Agarwal, Sai Sundaresan, Subrata Mi-
tra, Debabrata Mahapatra, Archit Gupta,
Rounak Sharma, Nirmal Joshua Kapu, Tong Yu,
and Shiv Saini. 2025. Cache-craft: Managing
chunk-caches for efficient retrieval-augmented
generation.Proc. ACM Manag. Data, 3(3).
Amey Agrawal, Nitin Kedia, Ashish Panwar,
Jayashree Mohan, Nipun Kwatra, Bhargav S.
Gulavani, Alexey Tumanov, and Ramachandran
Ramjee. 2024. Taming Throughput-Latencytradeoff in LLM inference with Sarathi-Serve. In
18th USENIX Symposium on Operating Systems
Design and Implementation (OSDI 24), pages
117–134, Santa Clara, CA. USENIX Associa-
tion.
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup
Sil, and Hannaneh Hajishirzi. 2023. Self-
RAG: Learning to retrieve, generate, and cri-
tique through self-reflection.arXiv preprint
arXiv:2310.11511.
Parishad BehnamGhader, Vaibhav Adlakha, Mar-
ius Mosbach, Dzmitry Bahdanau, Nicolas Cha-
pados, and Siva Reddy. 2024. LLM2Vec: Large
Language Models Are Secretly Powerful Text
Encoders.arXiv preprint. ArXiv:2404.05961
[cs].
Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo,
Luis Ceze, and Arvind Krishnamurthy. 2024.
Punica: Multi-Tenant LoRA Serving.Proceed-
ings of Machine Learning and Systems, 6:1–13.
Supawit Chockchowwat, Wenjie Liu, and Yongjoo
Park. 2023. AirIndex: Versatile index tuning
through data and storage.Proceedings of the
ACM on Management of Data, 1(3).
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie
Callan. 2023. Precise Zero-Shot Dense Retrieval
without Relevance Labels. InProceedings of
the 61st Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Pa-
pers), pages 1762–1777, Toronto, Canada. Asso-
ciation for Computational Linguistics.
Shiwei Gao, Youmin Chen, and Jiwu Shu. 2025.
Fast state restoration in LLM serving with
HCache. InProceedings of the Twentieth Eu-
ropean Conference on Computer Systems, Eu-
roSys ’25, pages 128–143, Rotterdam, Nether-
lands. ACM.
Thomas Gauthier. 2024. Lord: Low-rank de-
composition of finetuned large language mod-
els.https://github.com/thomasgauthier/
LoRD. GitHub repository; commit fe03a8c
(2024-05-02). Accessed 2025-09-19.
Aaron Grattafiori, Abhimanyu Dubey, Abhinav
Jauhri, Abhinav Pandey, Abhishek Kadian, Ah-
mad Al-Dahle, Aiesha Letman, Akhil Mathur,
Alan Schelten, Alex Vaughan, Amy Yang, An-
gela Fan, Anirudh Goyal, Anthony Hartshorn,
Aobo Yang, Archi Mitra, Archie Sravankumar,
Artem Korenev, Arthur Hinsvark, and 542 oth-
ers. 2024. The Llama 3 Herd of Models.arXiv
preprint. ArXiv:2407.21783 [cs].
Joseph M Hellerstein, Peter J Haas, and Helen J
Wang. 1997. Online aggregation. InProceed-
ings of the 1997 ACM SIGMOD international
conference on Management of data, pages 171–
182.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sug-
awara, and Akiko Aizawa. 2020. Construct-
ing a multi-hop QA dataset for comprehen-
sive evaluation of reasoning steps. InPro-
ceedings of the 28th International Conference
on Computational Linguistics, pages 6609–6625,
Barcelona, Spain (Online). International Com-
mittee on Computational Linguistics.
Cunchen Hu, Heyang Huang, Liangliang Xu,
Xusheng Chen, Jiang Xu, Shuang Chen, Hao
Feng, Chenxi Wang, Sa Wang, Yungang Bao,
Ninghui Sun, and Yizhou Shan. 2024. Inference
without Interference: Disaggregate LLM Infer-
ence for Mixed Downstream Workloads.arXiv
preprint. ArXiv:2401.11181 [cs].
Andrei Ivanov, Nikoli Dryden, Tal Ben-Nun, Shi-
gang Li, and Torsten Hoefler. 2021. Data move-
ment is all you need: A case study on optimizing
transformers. InProceedings of Machine Learn-
ing and Systems, volume 3, pages 711–732.
Gautier Izacard, Mathilde Caron, Lucas Hos-
seini, Sebastian Riedel, Piotr Bojanowski, Ar-
mand Joulin, and Edouard Grave. 2021. Unsu-
pervised dense information retrieval with con-
trastive learning.Preprint, arXiv:2112.09118.
Albert Q. Jiang, Alexandre Sablayrolles, Arthur
Mensch, Chris Bamford, Devendra Singh Chap-
lot, Diego de las Casas, Florian Bressand,
Gianna Lengyel, Guillaume Lample, Lucile
Saulnier, L´ elio Renard Lavaud, Marie-Anne
Lachaux, Pierre Stock, Teven Le Scao, Thibaut
Lavril, Thomas Wang, Timoth´ ee Lacroix, and
William El Sayed. 2023a. Mistral 7B.arXiv
preprint. ArXiv:2310.06825 [cs].
Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing
Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang,
Jamie Callan, and Graham Neubig. 2023b. Ac-
tive Retrieval Augmented Generation. InPro-
ceedings of the 2023 Conference on Empirical
Methods in Natural Language Processing, pages
7969–7992, Singapore. Association for Compu-
tational Linguistics.
Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue
Liu, Xin Liu, Xuanzhe Liu, and Xin Jin.
2024. RAGCache: Efficient knowledge caching
for retrieval-augmented generation.Preprint,
arXiv:2404.12457.
Jeff Johnson, Matthijs Douze, and Herv´ e J´ egou.
2021. Billion-scale similarity search with gpus.
IEEE Transactions on Big Data, 7(3):535–547.
Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
Sheng, Lianmin Zheng, Cody Hao Yu, Joseph
Gonzalez, Hao Zhang, and Ion Stoica. 2023.
Efficient Memory Management for Large Lan-
guage Model Serving with PagedAttention. In
Proceedings of the 29th Symposium on Operat-
ing Systems Principles, pages 611–626, Koblenz
Germany. ACM.Patrick Lewis, Ethan Perez, Aleksandra Piktus,
Fabio Petroni, Vladimir Karpukhin, Naman
Goyal, Heinrich K¨ uttler, Mike Lewis, Wen-tau
Yih, Tim Rockt¨ aschel, Sebastian Riedel, and
Douwe Kiela. 2020. Retrieval-Augmented Gen-
eration for Knowledge-Intensive NLP Tasks. In
Advances in Neural Information Processing Sys-
tems, volume 33, pages 9459–9474. Curran As-
sociates, Inc.
Xiaoxi Li, Yujia Zhou, and Zhicheng Dou. 2024.
UniGen: A Unified Generative Framework
for Retrieval and Question Answering with
Large Language Models.Proceedings of the
AAAI Conference on Artificial Intelligence,
38(8):8688–8696.
Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun
Long, Pengjun Xie, and Meishan Zhang. 2023.
Towards General Text Embeddings with Multi-
stage Contrastive Learning.arXiv preprint.
ArXiv:2308.03281 [cs].
Zhaoheng Li, Wei Ding, Silu Huang, Zikang Wang,
Yuanjin Lin, Ke Wu, Yongjoo Park, and Jian-
jun Chen. 2025a. Cloud-native vector search: A
comprehensive performance analysis.Preprint,
arXiv:2511.14748.
Zhaoheng Li, Silu Huang, Wei Ding, Yongjoo Park,
and Jianjun Chen. 2025b. Sieve: Effective fil-
tered vector search with collection of indexes.
Proc. VLDB Endow., 18(11):4723–4736.
Xueguang Ma, Liang Wang, Nan Yang, Furu Wei,
and Jimmy Lin. 2024. Fine-Tuning LLaMA
for Multi-Stage Text Retrieval. InProceedings
of the 47th International ACM SIGIR Confer-
ence on Research and Development in Informa-
tion Retrieval, SIGIR ’24, pages 2421–2425, New
York, NY, USA. Association for Computing Ma-
chinery.
Yu A. Malkov and D. A. Yashunin. 2020. Effi-
cient and robust approximate nearest neighbor
search using hierarchical navigable small world
graphs.IEEE Trans. Pattern Anal. Mach. In-
tell., 42(4):824–836.
Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caim-
ing Xiong, Yingbo Zhou, and Semih Yavuz.
2024. SFR-Embedding-Mistral: Enhance text
retrieval with transfer learning. Salesforce AI
Research Blog.
Niklas Muennighoff, Hongjin Su, Liang Wang, Nan
Yang, Furu Wei, Tao Yu, Amanpreet Singh, and
Douwe Kiela. 2025. Generative representational
instruction tuning. InThe Thirteenth Interna-
tional Conference on Learning Representations.
Reiner Pope, Sholto Douglas, Aakanksha Chowd-
hery, Jacob Devlin, James Bradbury, Jonathan
Heek, Kefan Xiao, Shivani Agrawal, and Jeff
Dean. 2023. Efficiently scaling transformer in-
ference. InProceedings of Machine Learning and
Systems, volume 5, pages 606–624.

Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui,
Heyi Tang, Feng Ren, Teng Ma, Shangming Cai,
Yineng Zhang, Mingxing Zhang, Yongwei Wu,
Weimin Zheng, and Xinran Xu. 2025. Moon-
cake: A kvcache-centric disaggregated architec-
ture for llm serving.ACM Transactions on Stor-
age.
Raunak Shah, Zhaoheng Li, and Yongjoo Park.
2026. QStore: Quantization-aware compressed
model storage.Proceedings of the VLDB En-
dowment, 19(3):388–398.
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie
Huang, Nan Duan, and Weizhu Chen. 2023. En-
hancing Retrieval-Augmented Large Language
Models with Iterative Retrieval-Generation Syn-
ergy. InFindings of the Association for Compu-
tational Linguistics: EMNLP 2023, pages 9248–
9274, Singapore. Association for Computational
Linguistics.
Tao Shen, Guodong Long, Xiubo Geng,
Chongyang Tao, Yibin Lei, Tianyi Zhou,
Michael Blumenstein, and Daxin Jiang.
2024. Retrieval-Augmented Retrieval: Large
Language Models are Strong Zero-Shot Re-
triever. InFindings of the Association for
Computational Linguistics: ACL 2024, pages
15933–15946, Bangkok, Thailand. Association
for Computational Linguistics.
Ying Sheng, Shiyi Cao, Dacheng Li, Coleman
Hooper, Nicholas Lee, Shuo Yang, Christo-
pher Chou, Banghua Zhu, Lianmin Zheng, Kurt
Keutzer, Joseph E. Gonzalez, and Ion Stoica.
2024a. SLoRA: Scalable Serving of Thousands
of LoRA Adapters.Proceedings of Machine
Learning and Systems, 6:296–311.
Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu,
Zhuohan Li, Danyang Zhuo, Joseph E. Gonza-
lez, and Ion Stoica. 2024b. Fairness in serving
large language models. In18th USENIX Sym-
posium on Operating Systems Design and Im-
plementation (OSDI 24), pages 965–988, Santa
Clara, CA. USENIX Association.
Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuo-
han Li, Max Ryabinin, Beidi Chen, Percy Liang,
Christopher Re, Ion Stoica, and Ce Zhang. 2023.
FlexGen: High-Throughput Generative Infer-
ence of Large Language Models with a Single
GPU. InProceedings of the 40th International
Conference on Machine Learning, pages 31094–
31116. PMLR. ISSN: 2640-3498.
Nikhil Sheoran, Supawit Chockchowwat, Arav
Chheda, Suwen Wang, Riya Verma, and
Yongjoo Park. 2023. A step toward deep on-
line aggregation.Proceedings of the ACM on
Management of Data, 1(2):1–28.
Foteini Strati, Xianzhe Ma, and Ana Klimovic.
2024. Orion: Interference-aware, fine-grainedgpu sharing for ml applications. InProceedings
of the Nineteenth European Conference on Com-
puter Systems, EuroSys ’24, page 1075–1092,
New York, NY, USA. Association for Comput-
ing Machinery.
Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten
de Rijke, Yixing Fan, and Xueqi Cheng.
2025. Boosting Retrieval-Augmented Genera-
tion with Generation-Augmented Retrieval: A
Co-Training Approach. InProceedings of the
48th International ACM SIGIR Conference on
Research and Development in Information Re-
trieval, SIGIR ’25, pages 2441–2451, New York,
NY, USA. Association for Computing Machin-
ery.
Harsh Trivedi, Niranjan Balasubramanian, Tushar
Khot, and Ashish Sabharwal. 2023. Interleav-
ing Retrieval with Chain-of-Thought Reason-
ing for Knowledge-Intensive Multi-Step Ques-
tions. InProceedings of the 61st Annual Meet-
ing of the Association for Computational Lin-
guistics (Volume 1: Long Papers), pages 10014–
10037, Toronto, Canada. Association for Com-
putational Linguistics.
Bhoomit Vasani, Jack FitzGerald, Anjie Fang, and
Sushmit Vaish. 2025. Phlora: data-free post-
hoc low-rank adapter extraction from full-rank
checkpoint.arXiv preprint arXiv:2509.10971.
Ashish Vaswani, Noam Shazeer, Niki Parmar,
Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
 L ukasz Kaiser, and Illia Polosukhin. 2017. At-
tention is all you need. InAdvances in Neu-
ral Information Processing Systems, volume 30.
Curran Associates, Inc.
Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai
Jin, Peng Xu, Shengjun Li, Xiangyu Wang, Xi-
angzhou Guo, Chengming Li, Xiaohai Xu, Kun
Yu, Yuxing Yuan, Yinghao Zou, Jiquan Long,
Yudong Cai, Zhenxiang Li, Zhifeng Zhang, Yi-
hua Mo, Jun Gu, and 3 others. 2021a. Milvus:
A purpose-built vector data management sys-
tem. InProceedings of the 2021 International
Conference on Management of Data, SIGMOD
’21, page 2614–2627, New York, NY, USA. As-
sociation for Computing Machinery.
Liang Wang, Nan Yang, Xiaolong Huang, Lin-
jun Yang, Rangan Majumder, and Furu Wei.
2024. Improving Text Embeddings with Large
Language Models. InProceedings of the 62nd
Annual Meeting of the Association for Compu-
tational Linguistics (Volume 1: Long Papers),
pages 11897–11916, Bangkok, Thailand. Associ-
ation for Computational Linguistics.
Liang Wang, Nan Yang, and Furu Wei. 2023.
Query2doc: Query Expansion with Large Lan-
guage Models. InProceedings of the 2023 Con-
ference on Empirical Methods in Natural Lan-
guage Processing, pages 9414–9423, Singapore.
Association for Computational Linguistics.

Xiaohui Wang, Ying Xiong, Yang Wei, Mingxuan
Wang, and Lei Li. 2021b. LightSeq: A High Per-
formance Inference Library for Transformers. In
Proceedings of the 2021 Conference of the North
American Chapter of the Association for Com-
putational Linguistics: Human Language Tech-
nologies: Industry Papers, pages 113–120, On-
line. Association for Computational Linguistics.
Bingyang Wu, Yinmin Zhong, Zili Zhang, Shengyu
Liu, Fangyue Liu, Yuanhang Sun, Gang Huang,
Xuanzhe Liu, and Xin Jin. 2024. Fast Dis-
tributed Inference Serving for Large Language
Models.arXiv preprint. ArXiv:2305.05920 [cs].
Haocheng Xia, Mihir Pamnani, Hanxi Fang, Su-
pawit Chockchowwat, and Yongjoo Park. 2026.
LazyAttention: Efficient retrieval-augmented
generation with deferred positional encoding. In
Proceedings of the 43rd International Confer-
ence on Machine Learning, Proceedings of Ma-
chine Learning Research. PMLR.
Haojun Xia, Zhen Zheng, Yuchao Li, Donglin
Zhuang, Zhongzhu Zhou, Xiafei Qiu, Yong Li,
Wei Lin, and Shuaiwen Leon Song. 2023. Flash-
llm: Enabling cost-effective and highly-efficient
large generative model inference with unstruc-
tured sparsity.Preprint, arXiv:2309.10285.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng,
Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang,
Guanting Dong, Haoran Wei, Huan Lin, Jia-
long Tang, Jialin Wang, Jian Yang, Jianhong
Tu, Jianwei Zhang, Jianxin Ma, and 43 others.
2024. Qwen2 Technical Report.arXiv preprint.
ArXiv:2407.10671 [cs].
Gyeong-In Yu, Joo Seong Jeong, Geon-Woo
Kim, Soojeong Kim, and Byung-Gon Chun.
2022. Orca: A distributed serving system
for{Transformer-Based}generative models. In
16th USENIX Symposium on Operating Systems
Design and Implementation (OSDI 22), pages
521–538.
Lingfan Yu, Jinkun Lin, and Jinyang Li. 2025.
Stateful large language model serving with pen-
sieve. InProceedings of the Twentieth European
Conference on Computer Systems, EuroSys ’25,
page 144–158, New York, NY, USA. Association
for Computing Machinery.
Caojin Zhang, Qiang Zhang, Ke Li, Sai Vid-
yaranya Nuthalapati, Benyu Zhang, Jason Liu,
Serena Li, Lizhu Zhang, and Xiangjun Fan.
2025. GEM: Empowering LLM for both Embed-
ding Generation and Language Understanding.
arXiv preprint. ArXiv:2506.04344 [cs].
Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo
Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao
Zhang. 2024. DistServe: Disaggregating prefilland decoding for goodput-optimized large lan-
guage model serving. In18th USENIX Sym-
posium on Operating Systems Design and Im-
plementation (OSDI 24), pages 193–210, Santa
Clara, CA. USENIX Association.

A Additional Experiments
A.1 Pooling Equivalence
We test whether incremental pooling repro-
duces single-pass embeddings across pooling
heads and chunk sizes. On 250 samples span-
ning a range of input lengths, we compare
paired outputs for mean, CLS, and weighted-
mean pooling at chunk sizes 256, 512, and 1024
using cosine similarity and relativeL 2error.
In Tab. 7, minimum cosine similarity exceeds
0.9999 in all nine configurations, while both
mean and p99 relativeL 2errors remain below
1%. These small discrepancies reflect floating-
point reduction-order effects rather than an al-
gorithmic approximation.
Table 7: Equivalence between chunked and single-
pass embeddings.Cis the chunk size;L 2errors
are relative.
PoolingCCosine similarity RelativeL 2
Min Mean p50 Mean p99
Mean256 .9999512 .9999844 .9999864 .005527 .008995
512 .9999555 .9999861 .9999885 .005176 .009478
1024 .9999484 .9999877 .9999881 .004820 .008156
CLS256 .9999894 .9999912 .9999912 .005072 .005113
512 .9999837 .9999887 .9999892 .004908 .005479
1024 .9999920 .9999930 .9999920 .003692 .004236
Weighted
mean256 .9999456 .9999882 .9999907 .004650 .009628
512 .9999339 .9999887 .9999918 .004493 .009927
1024 .9999452 .9999905 .9999915 .004215 .007636
A.2 Scheduling-Parameter Sensitivity
Chunk-Size SweepWe test whether per-
formance depends on the granularity of chun-
ked embedding and select a chunk size for the
subsequent budget sweep. Using the 5:5 work-
load from§5.2, we vary the embedding chunk
sizeCacross 128, 256, 512, and 1024 tokens.
As shown in Fig. 7(a), combined throughput
ranges from 2.6167 to 2.6500 requests/s, gen-
eration p99 from 4.08 to 4.16 s, and embedding
p99 from 83.58 to 98.77 ms. This shows that
performance is insensitive to chunk size over
the tested range.
Token-Budget SweepWe next test
whether performance at the selected chunk
size depends on the per-iteration token
budget. The 256- and 512-token settings
tie for the highest combined throughput,
so we selectC= 512 because it has the
lower embedding p99. Holding this chunk
size and the workload fixed, we sweep thetoken budgetBacross 2048, 4096, and 8192
tokens. As shown in Fig. 7(b), combined
throughput remains between 2.6125 and
2.6375 requests/s, generation p99 between
4.06 and 4.12 s, and embedding p99 between
82.95 and 109.35 ms. Combined throughput
varies by less than 1%, and latency shows no
consistent degradation as the budget changes.
These results show thatOrthrusdoes not
require a narrowly tuned token budget.
0123Throughput
(req/s)
128 256 512 102401234Gen. p99
(s)
2048 4096 819204080120
Embed. p99
(ms)
(a) Chunk size (b) Token budgetCombined throughput Gen. p99
Embed. p99
Figure 7: Scheduling-parameter sensitivity for (a)
chunk size and (b) token budget. Throughput and
latency for embedding and generation remain sta-
ble across different chunk sizes and token budgets.
A.3 Standalone Chunked Embedding
Ablation
Because chunked embedding and incremental
pooling are enabled together in our implemen-
tation, we evaluate their combined effect in
two complementary settings.
Scheduler-Packing SimulationWe first
test whether chunking improves token-budget
packing without reducing generation service.
We compare two otherwise identical FCFS it-
eration schedulers. In the atomic condition,
an embedding request must be processed in
one iteration. In the chunked condition, it can
be divided into chunks of at mostC= 64 to-
kens. Each iteration has a token budget of
B= 256, and the request mix is 80% genera-
tion and 20% embedding. Each generation re-
quest has a 128-token prefill followed by a 256-
token decode, which the simulator advances in
eight-token steps. Embedding requests con-
tain either 96 or 224 tokens, with 224-token
inputs comprising 45% of the embedding work-
load. As shown in Tab. 8, chunking raises av-
erage token-budget fill from 8.8% to 85.7%.

Generation throughput remains unchanged at
93.3 requests/s, while embedding throughput
increases from 25.5 to 26.9 requests/s. Thus,
chunking recovers unused token capacity with-
out sacrificing generation throughput.
Table 8: Atomic-versus-chunked embedding simu-
lation. Budget fill is the average fraction of the
per-iteration token budget occupied by scheduled
tokens; throughput is in requests/s.
EmbeddingBudget
fill (%)Total Gen Embed
Atomic 8.8 118.8 93.3 25.5
Chunked 85.7 120.2 93.3 26.9
Embedding-Only ExecutionWe next
test whether the implemented chunked path
introduces execution overhead when embed-
dings run without generation contention. We
compare chunking disabled against chunking
with incremental pooling enabled on 250 in-
puts using mean-pooled OPT-1.3B embed-
dings. As shown in Tab. 9, chunking reduces
mean latency by 9.8% (20.4 to 18.4 ms) and
increases throughput by 10.9% (49.0 to 54.4
embeddings/s). Thus, the scheduling flexibil-
ity provided by chunking does not come at the
cost of standalone embedding efficiency on this
workload.
Table 9: Embedding-only comparison using mean-
pooled OPT-1.3B embeddings on 250 inputs.
Setting Avg. latency (ms) Embed./s Relative
No chunk 20.4 49.044 1.000×
Chunked 18.4 54.384 1.109×
A.4 Latency Robustness
Workload-Ratio SensitivityWe evaluate
whether IBS maintains low generation startup
and embedding latency as the workload com-
position shifts. We sweep the generation re-
quest share across 10%, 50%, and 90%, us-
ing three repeats per setting, 512 clients, and
180-second submission windows. Generation
prompts and embedding inputs contain 128 to-
kens, and generation is capped at 1024 output
tokens. As shown in Fig. 8, generation TTFT
p99 remains below 200 ms across the sweep,
ranging from 173.57 to 181.96 ms. Embedding
p50 also remains stable at 278.70–301.87 ms.0150300450
10% 50% 90%050100150200
Generation request shareGen. TTFT p99 (ms)1,800TTFT p99 Embed. p50 Embed. p95
Embed. latency (ms)
Figure 8: Generation and embedding latency
across workload ratios. Generation TTFT p99 and
embedding p50 remain stable, while embedding
p95 increases under generation-heavy workloads.
Embedding p95 is 322.59 and 327.35 ms at
10% and 50% generation, respectively, but
rises to 1.71 s at 90% generation. Thus, gener-
ation startup and typical embedding latency
are robust to the request mix, while an ex-
tremely generation-heavy workload primarily
affects the embedding tail.
Long-Decode SensitivityWe evaluate
whether IBS preserves prompt embedding ser-
vice as longer generation decodes occupy the
system. Using the 5:5 workload, we vary only
the generation decode length, from 256 to 4096
tokens. As shown in Tab. 10, generation p99
latency increases from 7.06 to 79.68 s. In con-
trast, embedding p50 remains between 52.58
and 54.62 ms, and embedding p99 remains be-
tween 53.09 and 59.92 ms. Throughput is not
monotonic. For example, it increases from
0.1167 requests/s at 2048 tokens to 0.1333 re-
quests/s at 4096 tokens. Thus, we do not in-
terpret differences between adjacent settings
as a scaling trend. The central result is la-
tency isolation. Despite an over 11×increase
in generation p99, embedding p99 varies by
less than 7 ms. IBS continues to schedule em-
bedding requests promptly even when genera-
tion becomes decode-dominated.
Table 10: Decode-length sensitivity on the 5:5
workload. Throughput is reported in requests/s
and latency in milliseconds.
Decode
(tokens)Throughput (rps) Latency (ms)
Gen Embed Gen p99 Embed p50 Embed p99
256 0.2333 0.2333 7057.75 53.32 59.02
512 0.1611 0.1611 13959.11 52.58 53.09
1024 0.1528 0.1528 27806.13 53.37 59.92
2048 0.1167 0.1167 56780.18 54.62 54.97
4096 0.1333 0.1333 79675.66 54.42 55.42