# Efficient Distributed Retrieval-Augmented Generation for Enhancing Language Model Performance

**Authors**: Shangyu Liu, Zhenzhe Zheng, Xiaoyao Huang, Fan Wu, Guihai Chen, Jie Wu

**Published**: 2025-04-15 13:53:08

**PDF URL**: [http://arxiv.org/pdf/2504.11197v2](http://arxiv.org/pdf/2504.11197v2)

## Abstract
Small language models (SLMs) support efficient deployments on
resource-constrained edge devices, but their limited capacity compromises
inference performance. Retrieval-augmented generation (RAG) is a promising
solution to enhance model performance by integrating external databases,
without requiring intensive on-device model retraining. However, large-scale
public databases and user-specific private contextual documents are typically
located on the cloud and the device separately, while existing RAG
implementations are primarily centralized. To bridge this gap, we propose
DRAGON, a distributed RAG framework to enhance on-device SLMs through both
general and personal knowledge without the risk of leaking document privacy.
Specifically, DRAGON decomposes multi-document RAG into multiple parallel token
generation processes performed independently and locally on the cloud and the
device, and employs a newly designed Speculative Aggregation, a dual-side
speculative algorithm to avoid frequent output synchronization between the
cloud and device. A new scheduling algorithm is further introduced to identify
the optimal aggregation side based on real-time network conditions. Evaluations
on real-world hardware testbed demonstrate a significant performance
improvement of DRAGON-up to 1.9x greater gains over standalone SLM compared to
the centralized RAG, substantial reduction in per-token latency, and negligible
Time to First Token (TTFT) overhead.

## Full Text


<!-- PDF content starts -->

Efficient Distributed Retrieval-Augmented Generation for
Enhancing Language Model Performance
Shangyu Liu1, Zhenzhe Zheng1, Xiaoyao Huang2, Fan Wu1, Guihai Chen1Jie Wu2
1Shanghai Jiao Tong University
2Cloud Computing Research Institute, China Telecom
{liushangyu,zhengzhenzhe}@sjtu.edu.cn, huangxy32@chinatelecom.cn,
{fwu,gchen}@cs.sjtu.edu.cn, wujie@chinatelecom.cn
Abstract
Small language models (SLMs) support efficient deployments on
resource-constrained edge devices, but their limited capacity com-
promises inference performance. Retrieval-augmented generation
(RAG) is a promising solution to enhance model performance by in-
tegrating external databases, without requiring intensive on-device
model retraining. However, large-scale public databases and user-
specific private contextual documents are typically located on the
cloud and the device separately, while existing RAG implemen-
tations are primarily centralized. To bridge this gap, we propose
DRAGON, a distributed RAG framework to enhance on-device
SLMs through both general and personal knowledge without the
risk of leaking document privacy. Specifically, DRAGON decom-
poses multi-document RAG into multiple parallel token generation
processes performed independently and locally on the cloud and
the device, and employs a newly designed Speculative Aggregation,
a dual-side speculative algorithm to avoid frequent output synchro-
nization between the cloud and device. A new scheduling algorithm
is further introduced to identify the optimal aggregation side based
on real-time network conditions. Evaluations on real-world hard-
ware testbed demonstrate a significant performance improvement
of DRAGON—up to 1.9×greater gains over standalone SLM com-
pared to the centralized RAG, substantial reduction in per-token
latency, and negligible Time to First Token (TTFT) overhead.
1 Introduction
Although large language models (LLMs) such as GPT-4 [ 42] and
DeepSeek-V3 [ 13] have demonstrated remarkable performance in
real-world applications, their substantial deployment costs have led
to predominant cloud-based hosting. As a result, users are required
to upload private context along with their queries, raising serious
privacy concerns [ 12]. Recently, small language models (SLMs)
such as Phi-4-mini [ 1] and Qwen2.5-1.5B [ 57], have emerged as
promising alternatives, offering efficient local deployment on edge
devices. However, although SLMs are notably smaller than cloud-
hosted LLMs—leading to reduced performance on both personal and
general tasks—they still remain too large for resource-constrained
devices to support on-device fine-tuning or training [ 27] to adapt
to newly generated data and user feedback.
Retrieval-augmented generation (RAG) [ 36,46] has demonstrated
effectiveness in boosting the performance of SLMs by incorporating
contextually relevant documents from external databases, such as
Wikipedia [ 4]. The performance gain increases monotonically with
the scale of the database [ 49], showing an opportunity for SLMs to
achieve comparable or even better performance than standalone
LLMs [ 11]. More importantly, by expanding user-specific external
DistributedRetrievalCentralizedRAG
+
+
+
+
+
AggregateRusty Lake2simulation1puzzle1TheSims2simulation1StardewValley2
longwaitStardewValley2simulation1DistributedRAGCloudDevice
I like games, among which is a hot selleronSteam.①②
Languagemodels
Cloud-OnlyDevice-OnlyFigure 1: Comparison between different RAG architectures.
database (also known as the non-parametric memory [ 36]), model
customization and knowledge updates can be achieved efficiently
without model training. Typically, large-scale public databases con-
taining general knowledge are hosted in the cloud, whereas user-
specific private databases are maintained on-device. Since the query
context may involve both general and personal data, it is essential
for retrieval-augmented SLMs to support distributed databases lo-
cated in the cloud and device. Unfortunately, most existing RAG so-
lutions [ 7,36,46] adopt a centralized architecture. Figure 1 presents
an example of game recommendation. The cloud-only RAG returns
an incorrect game genre, although private documents indicate a
preference for simulation games. In contrast, the device-only RAG
fails to retrieve the best-selling game lists without accessing to the
general knowledge in the cloud.
An intuitive solution, similar to federated search [ 52], is to re-
trieve documents from the cloud-side database, merge them with
those retrieved locally on-device, and perform model inference in
a centralized manner. However, this approach may incur substan-
tial latency overhead considering key-value (KV) caching [ 64], a
fundamental mechanism in language model serving that stores
intermediate attention states to enable efficient reuse of past com-
putations. Given a context sequence length 𝑛, the KV cache trades
𝑂(𝑛)storage for a reduction in decoding time complexity from
𝑂(𝑛2)to𝑂(𝑛). Therefore, the KVs of documents are often pre-
computed, stored in the database, and retrieved. This leads to a
dilemma: when retrieving the raw text of cloud-side documents,
the device must compute their KVs from scratch, incurring signifi-
cant computation latency; Conversely, directly retrieving the KVs
from the cloud introduces substantial transmission latency, as the
size of KVs can be comparable to, or even larger than, the model
parameters, especially as the number of document grows [18].arXiv:2504.11197v2  [cs.LG]  16 Apr 2025

To address these issues, we propose DRAGON, a distributed
retrieval- augmented generati onframework designed to enhance
the performance of on-device language model inference. Follow-
ing the Law of Total Probability, DRAGON first decomposes the
multi-document RAG process into a dual-side workflow, and then
aggregates their output tokens for the final result. In this workflow,
the cloud and device sides independently execute their own LM
instances using documents retrieved from their databases. Docu-
ment KVs are stored and loaded locally without transmission or
re-computation, thereby reducing first-token latency and preserv-
ing document privacy. Nonetheless, the output aggregation requires
frequent exchange of data packets between the cloud and device at
every token generation step, due to the auto-regressive nature of
language models. This transmission pattern requires a persistent
low-latency connection between the cloud and device, which is
difficult to guarantee in real-world scenarios [39, 40].
To solve this challenge, we draw inspiration from the draft-then-
verify paradigm in Speculative Decoding [33] and propose a new
dual-side speculative algorithm, namely Speculative Aggregation . In
this algorithm, the decoding processes on both sides continuously
generates draft tokens, and an Aggregator on either side (depending
on certain scheduling criteria) asynchronously verifies and aggre-
gates them. Decoding is interrupted and the corresponding KV
states are rolled back for re-computation only when a draft is re-
jected. As our theoretical analysis proves the equivalence between
Speculative Aggregation and the vanilla synchronized version, the
end-to-end latency can be reduced by overlapping transmission
and decoding processes, especially when the output distributions
of the two sides are similar.
We implement a fully functional distributed RAG workflow and
construct a testbed using real-world hardware. Based on this, we
evaluate DRAGON against various RAG architectures using repre-
sentative SLMs on large-scale retrieval corpora and datasets. Exper-
imental results of language modeling on WikiText demonstrate that
DRAGON achieves 1.9×and1.4×greater performance gains over
the standalone SLM than the centralized method, using Qwen2.5-
1.5B and OPT-1.3B, respectively. Moreover, DRAGON shows strong
robustness under various network conditions and achieves a 42.4%-
49.5% reduction in per-token latency compared to synchronized
methods under 300 ms network latency. While retrieving raw text
and KV states incurs up to 8.9×and15.3×overhead in response
time, DRAGON introduces negligible overhead. Extensive simula-
tions further verify that the proposed scheduling algorithm achieves
increasing delay reduction as network latency grows.
We summarize the key contributions of this work as follows:
•We propose DRAGON, the first distributed RAG system that sup-
ports distributed documents retrieval and collaborative output
generation between cloud and device. It significantly enhances
the performance of on-device SLMs with the integration of both
personal and general knowledge.
•We introduce Speculative Aggregation , a dual-side speculative
algorithm that decouples synchronized aggregation from sequen-
tial decoding by asynchronously verifying the output alignment
between cloud and device, greatly reducing end-to-end latency.•We further design an adaptive scheduling algorithm to dynam-
ically identify the optimal aggregation side under varying net-
work conditions, effectively improving decoding efficiency.
•We implement DRAGON in a real-world hardware testbed and
perform comprehensive evaluations using representative SLMs
and large-scale retrieval corpora, demonstrating significant per-
formance improvements of on-device SLMs with negligible over-
head even under high-latency network conditions.
2 Preliminaries
2.1 Retrieval-Augmented Generation
Retrieval-augmented generation [ 36] integrates off-the-shelf lan-
guage models with documents retrieved from an external database
to capture long-tail knowledge and keep up-to-date with new infor-
mation. In traditional LM inference, given an input token sequence
𝑥<𝑀={𝑥0,...,𝑥𝑀−1}(indices of tokens in vocabulary 𝑉) and the
maximum context length 𝑁, the output generation process aims to
maximize the probabilityÎ𝑁−1
𝑡=𝑀𝑝(𝑥𝑡|𝑥<𝑡). In order to incorporate
external documents, we process each document concatenated with
the query separately, and then interpolate the output distributions
(termed as output aggregation [3,36,51])1. Following the Law of
Total Probability, we can derive the interpolation as
𝒑(𝑥𝑡|𝑥<𝑡)=∑︁
𝑑∼𝑝(𝑑)𝑝(𝑑|𝑥<𝑡)·𝒑(𝑥𝑡|𝑑,𝑥<𝑡), (1)
where𝑝(𝑑|𝑥<𝑡)denotes the weight of the document 𝑑on the out-
put distribution 𝒑(𝑥𝑡|𝑑,𝑥<𝑡). Since𝑝(𝑑|𝑥<𝑡)cannot be directly
obtained in practice, we retrieve 𝑑from a sufficiently large corpus
Dand only consider top- 𝑘documents with the highest relevance
scoreRD(𝑑,𝑥<𝑡). Equation (1)offers the opportunity to decom-
pose the multi-document RAG workflow into parallel generation
processes, enabling device-cloud distributed RAG. This decomposi-
tion also significantly alleviates the limitation of maximum context
length on resource-constraint devices.
2.2 Device-Cloud Distributed RAG
To enhance the performance of on-device LLM inference, we pro-
pose a device-cloud distributed RAG framework based on the above
discussed output aggregation paradigm. Given an input 𝑥<𝑡, we
retrieve personalized documents 𝐷devicefrom a device-side private
database and then compute the next-token distributions 𝑷device
𝑡=𝒑(𝑥𝑡|𝑑,𝑥<𝑡)⊤
𝑑∈𝐷device using an on-device LLM Mdevice. In paral-
lel, we employ a similar process in the cloud and obtain the cloud-
side next-token distributions 𝑷cloud
𝑡. After gathering all documents
𝐷=𝐷device∪𝐷cloudand their corresponding output distributions
𝑷𝑡=
𝑷device
𝑡,𝑷cloud
𝑡⊤, we sample the next token according to
𝑥𝑡∼𝝎⊤
𝑡𝑷𝑡=∑︁
𝑑∈𝐷𝜔𝑡(𝑑)·𝒑(𝑥𝑡|𝑑,𝑥<𝑡), (2)
where 𝝎𝑡=𝜔𝑡(𝑑)⊤
𝑑∈𝐷denotes the interpolation weights, which
are computed based on relevance scores Ras
𝜔𝑡(𝑑)=expR(𝑑,𝑥<𝑡)/∑︁
𝑑′∈𝐷expR(𝑑′,𝑥<𝑡).
We refer to this workflow as the vanilla distributed RAG.
1The output aggregation is different from context aggregation [46]), where external
documents are concatenated and prepended to the input query 𝑥<𝑡all at once.
2

Despite its effectiveness, frequent synchronization between the
device and the cloud can introduce a substantial latency. On one
hand, the tight data coupling in distributed RAG leads to idle wait-
ing, especially when decoding latencies significantly differ due
to hardware heterogeneity in cloud and device. During the auto-
regressive LLM model inference, the output 𝑥𝑡−1is expected on
both sides as the input for generating 𝑷𝑡. At each token genera-
tion step𝑡, computing Equation (2)requires waiting for both the
device-side and cloud-side output distributions, 𝑷device
𝑡and𝑷cloud
𝑡.
On the other hand, frequent transmission of data packets makes
this device-cloud distributed RAG paradigm highly sensitive to
network stability. Transmitted data packets at each step includes
a 2-byte integer representing the token 𝑥𝑡and a float matrix 𝑷𝑡
encoding the output distributions2. Due to the small data packet
size, transmission time is often dominated by data-independent fac-
tors [9, 20], such as the connection round-trip time (RTT). Finally,
idle waiting and transmission latency at each token generation step
accumulate over a long output sequence, significantly amplifying
the overall overhead.
2.3 Problem Formulation
We define the LLM inference as a distributed process where the
device-side and cloud-side token generation processes, Fdeviceand
Fcloud, executes alternatively. We assume the final output token
sequence is generated on-device by sampling 𝑥∼𝒑𝑡. Let𝐴𝑡be an
auxiliary set for transferring information between the device and
the cloud at iteration 𝑡, which is initially empty, and let 𝒑𝑡denote
the next-token distribution. The workflow can then be expressed as
𝐴device
𝑡,𝒑𝑡←Fdevice(𝐴cloud
𝑡−1,Mdevice,𝐷device,𝑥<𝑡)on the device,
and𝐴cloud
𝑡← Fcloud(𝐴device
𝑡,Mcloud,𝐷cloud,𝑥<𝑡)on the cloud,
respectively. Finally, the optimization objective is given by
min
F1
𝑁∑︁𝑁
𝑡=1 −𝑝(𝑥∗
𝑡|𝑥<𝑡)log𝒑𝑡(𝑥∗
𝑡|𝑥<𝑡)+𝜆𝐶(𝐴𝑡,F),(3)
where𝑥∗
𝑡represents the optimal token at step 𝑡and𝐶denotes the
end-to-end latency per token resulted from the transmission of 𝐴𝑡
and execution ofFbetween the cloud and device. The coefficient
𝜆controls the trade-off between performance and efficiency.
3 Overview of DRAGON
To enhance on-device LLM inference while minimizing the latency
overhead, we propose DRAGON, a device-cloud distributed RAG
framework. In this framework, we sample tokens from distribu-
tions aggregated from the device-side and cloud-side RAG outputs,
enabling an integration of personalized information and generic
knowledge. To mitigate the inherent latency caused by frequent
device-cloud synchronizations in vanilla distributed RAG, we per-
form distribution aggregation and next-token sampling in a spec-
ulative manner, where draft tokens are generated on both sides
and then verified on either side. Accordingly, as shown in Figure 2,
DRAGON consists of four modules deployed on both sides, in-
cluding Decoders, Queues, Profilers, and Schedulers, along with a
device/cloud-switchable Aggregator on either side.
2The float matrix 𝑷𝑡has a size of|𝑉|max(|𝐷device|,|𝐷cloud|), where the vocabulary
size|𝑉|is typically less than 50,000.
“science”“and”“technology”“games”“.”DraftQueues
“I”“love”“computer”
TargetQueue
“games”
“science”
“games”
decode()“.”
Aggregator+
SchedulerDecoder
+❶❷
❸❹
❹DecoderDraft&TargetQueuesTransmissionBus
DeviceCloud
QueryTargettokenAcceptancestatusDrafttokensandoutputdistributionsFigure 2: Overview of the DRAGON framework.
We organize Decoders, Queues and Aggregator by a producer-
consumer paradigm, enabling asynchronous decoding of draft to-
kens. The Decoder serves as a token producer, and on each side
𝑠∈{device,cloud}it decodes draft tokens 𝑥𝑠
𝑡independently based
on locally-aggregated output distributions 𝒑𝑠
𝑡=(˜𝝎𝑠
𝑡)⊤𝑷𝑠
𝑡where
˜𝝎𝑡=𝜔𝑡(𝑑)⊤
𝑑∈𝐷𝑠, similar to Equation (2)but using the retrieved
local documents 𝐷𝑠only ( 1). The draft tokens 𝑥𝑠
𝑡and their cor-
responding distribution vectors 𝒑𝑠
𝑡are broadcast to the other side.
On each side, we enqueue 𝑥𝑠
𝑡into Draft Queues ( 2). The Aggrega-
tor, as a consumer, continuously consumes draft tokens from the
front of local queues and performs aggregation process ( 3). Sub-
sequently, the aggregation results of the draft token are broadcast
to Draft Queues on both sides. For each queue, the first token is
dequeued if accepted, or the entire queue is cleared if rejected. The
final target token output by Aggregator is enqueued into Target
Queue on both sides ( 4). Although the dependencies between the
aggregator and decoder cannot be eliminated, the data transmission
latency can be overlapped with the decoding time, mitigating idle
waiting. To accommodate dynamic computing resources on both
sides and network bandwidth between them, we further design
Profilers and Schedulers to identify the optimal aggregation side.
4 Speculative Aggregation
Inspired by Speculative Decoding [33], we propose Speculative Ag-
gregation to reduce the device-cloud communication latency. Spec-
ulative Decoding adopts a draft-then-verify decoding paradigm to
reduce the number of calls to the resource-intensive LLM. Simi-
larly, Speculative Aggregation utilizes two independent decoding
processes, the device-side and cloud-side Decoders, to draft mul-
tiple candidate future tokens, which are then verifies through an
Aggregator. This is equivalent to directly sampling from the dis-
tributions aggregated from the device-side and cloud-side outputs.
As the aggregation involves collecting output distributions over
the network, we expect the speculative algorithm to reduce its
frequency and mitigate data transmission costs.
More specifically, the Aggregator stays in a blocked wait state
until both local Draft Queues are non-empty. Once this condition
is met, it retrieves one token 𝑥device
𝑡/𝑥cloud
𝑡from the front of each
3

queue and fetches corresponding locally-aggregated output distri-
butions 𝒑device
𝑡/𝒑cloud
𝑡from the cache. The tokens and the distribu-
tions are then provided as inputs to the aggregation. Subsection 4.1
presents the target distribution of the aggregation while Subsec-
tion 4.2 introduces an speculative strategy to sample from the target
distribution equivalently. Subsection 4.3 analyzes the acceptance
probability, providing guidance for further scheduling design. Since
the workflows of the device and cloud sides are designed to be
symmetric, we define {𝑙,𝑟}={device,cloud}to maintain general-
ity and avoid repetition. From the perspective of the Aggregator, 𝑙
refers to the local side that performs aggregation, while 𝑟denotes
the remote side, which only generates draft tokens.
4.1 Target Token Distribution
The objective of speculative aggregation is to generate tokens
that are equivalent to those sampled from the target distribution
𝒑𝑡=𝝎⊤
𝑡𝑷𝑡as defined in Equation (2). We partition 𝑷𝑡block-
wise, grouping its distribution vectors by generation side, and have
𝒑𝑡=(𝝎𝑙
𝑡)⊤𝑷𝑙
𝑡+(𝝎𝑟
𝑡)⊤𝑷𝑟
𝑡. For each𝑠∈{𝑙,𝑟}, we have 𝝎𝑙
𝑡=𝜂𝑠
𝑡˜𝝎𝑙
𝑡
where𝜂𝑠
𝑡=ℎ𝑠
𝑡/(ℎ𝑙
𝑡+ℎ𝑟
𝑡)andℎ𝑠
𝑡=Í
𝑑∈𝐷𝑠expR(𝑑,𝑥<𝑡). As a result,
given the locally-aggregated output distributions 𝒑𝑙
𝑡and𝒑𝑟
𝑡(§ 3),
the target distribution 𝒑𝑡can be obtained by an interpolation:
𝒑𝑡=𝜂𝑙
𝑡𝒑𝑙
𝑡+𝜂𝑟
𝑡𝒑𝑟
𝑡. (4)
To align with this computation process, on each side 𝑠∈{𝑙,𝑟}, a
corrected value3ofℎ𝑠
𝑡is computed and retained during decoding
𝑥𝑠
𝑡, and then broadcast and stored along with draft tokens and the
locally-aggregated distributions.
Dynamic weights. The interpolation weights 𝜂𝑙
𝑡and𝜂𝑟
𝑡in Equa-
tion (4)can be dynamic, as the relevance between documents and
the ongoing context may vary during generation. While current
studies [ 36,51] that employ output aggregation adopt static doc-
ument relevance scores, we explore dynamic weights by adopt-
ing a strategy inspired by those used in recommendation systems.
Upon receiving the input query 𝑥<𝑀, each side𝑠∈{𝑙,𝑟}performs
a one-time retrieval of a relatively large document set 𝐷𝑠(as in
most existing works [ 23,35,51]) to avoid key-value recomputation
caused by changes in the document prefix. During the decoding of
draft token 𝑥𝑡, we re-estimate the relevance scores R(𝑑,𝑥<𝑡)for
𝑑∈𝐷𝑠using a local re-ranking model (e.g., a Cross-Encoder [ 48])
and re-calculate the corrected ℎ𝑠
𝑡before transmission.
4.2 Design of Aggregation Strategy
To sample𝑥𝑡∼𝒑𝑡, we instead perform two independent speculative
sampling processes as follows:
•Keep the draft token 𝑥𝑙
𝑡as˜𝑥𝑙
𝑡if𝒑𝑙
𝑡(𝑥𝑙
𝑡)≤𝒑𝑟
𝑡(𝑥𝑙
𝑡), and in case
𝒑𝑙
𝑡(𝑥𝑙
𝑡)>𝒑𝑟
𝑡(𝑥𝑙
𝑡)we reject the sample with probability 𝜂𝑟
𝑡(1−
𝒑𝑟
𝑡(𝑥𝑙
𝑡)/𝒑𝑙
𝑡(𝑥𝑙
𝑡))and sample ˜𝑥𝑙
𝑡again from an adjusted distribu-
tion ˜𝒑𝑙
𝑡=norm(max(0,𝒑𝑟
𝑡−𝒑𝑙
𝑡)).
•Keep the draft token 𝑥𝑟
𝑡as˜𝑥𝑟
𝑡if𝒑𝑟
𝑡(𝑥𝑟
𝑡)≤𝒑𝑙
𝑡(𝑥𝑟
𝑡), and in case
𝒑𝑟
𝑡(𝑥𝑟
𝑡)>𝒑𝑙
𝑡(𝑥𝑟
𝑡)we reject the sample with probability 𝜂𝑙
𝑡(1−
𝒑𝑙
𝑡(𝑥𝑟
𝑡)/𝒑𝑟
𝑡(𝑥𝑟
𝑡))and sample ˜𝑥𝑟
𝑡again from an adjusted distribu-
tion ˜𝒑𝑟
𝑡=norm(max(0,𝒑𝑙
𝑡−𝒑𝑟
𝑡)).
3We adopt the log-sum-exp trick to maintain numerical stability (See Appendix A.1).Algorithm 1: SpeculativeAggregation
Input: Draft tokens 𝑥𝑠
𝑡, locally-aggregated distributions 𝒑𝑠
𝑡,
and aggregation weights ℎ𝑠
𝑡, for𝑠∈{𝑙,𝑟}
Output: Target token 𝑥𝑡, acceptance status S𝑙andS𝑟
Function Sample(𝑥,𝒑𝑎,𝒑𝑏,𝜂):
˜𝑥←𝑥,𝜎𝑎∼𝑈(0,1);
if𝒑𝑎(𝑥)>𝒑𝑏(𝑥),𝜎𝑎<𝜂(1−𝒑𝑏(𝑥)
𝒑𝑎(𝑥))then
˜𝑥∼norm(max(0,𝒑𝑏−𝒑𝑎));
return ˜𝑥;
𝜂𝑙
𝑡←ℎ𝑙
𝑡/(ℎ𝑙
𝑡+ℎ𝑟
𝑡),𝜂𝑟
𝑡←1−𝜂𝑙
𝑡;
˜𝑥𝑙
𝑡←Sample(𝑥𝑙
𝑡,𝒑𝑙
𝑡,𝒑𝑟
𝑡,𝜂𝑟
𝑡),˜𝑥𝑟
𝑡←Sample(𝑥𝑟
𝑡,𝒑𝑟
𝑡,𝒑𝑙
𝑡,𝜂𝑙
𝑡);
𝜎∼𝑈(0,1),𝑥𝑡←˜𝑥𝑙
𝑡·1𝜎≤0.5+˜𝑥𝑟
𝑡·1𝜎>0.5;
S𝑙←𝑥𝑙
𝑡=𝑥𝑡,S𝑟←𝑥𝑟
𝑡=𝑥𝑡;
return𝑥𝑡,S𝑙,S𝑟;
It is straightforward to show4that through these sampling pro-
cesses, both ˜𝑥𝑙
𝑡and ˜𝑥𝑟
𝑡are indeed drawn from the aggregated dis-
tribution𝜂𝑙
𝑡𝒑𝑙
𝑡+𝜂𝑟
𝑡𝒑𝑟
𝑡. We select either ˜𝑥𝑙
𝑡or˜𝑥𝑟
𝑡as𝑥𝑡with uniform
probability, ensuring 𝑥𝑡∼𝒑𝑡. Finally, each draft token 𝑥𝑙
𝑡and𝑥𝑟
𝑡
is accepted if it matches the target token 𝑥𝑡; otherwise, it is re-
jected. The aggregation strategy at each step 𝑡is summarized in
Algorithm 1. It is worth noting that we design a sampling-based
method rather than simply selecting between 𝑥𝑙
𝑡and𝑥𝑟
𝑡, in order to
ensure that𝑥𝑡∼𝒑𝑡holds. A counterexample for binary selection is
illustrated in cases where arg max 𝒑𝑡differs from both arg max 𝒑𝑙
𝑡
andarg max 𝒑𝑟
𝑡.
We now present a general procedure for sampling multiple con-
secutive tokens. At each step 𝑡, the following workflow is executed:
1)The Aggregator waits until both Draft Queues are non-empty,
then fetches 𝑥𝑠
𝑡from the front of the local Draft Queues and
retrieves the auxiliary variables 𝒑𝑠
𝑡andℎ𝑠
𝑡from the local cache,
for each𝑠∈{𝑙,𝑟}.
2)The Aggregator performs aggregation as defined in Algorithm 1.
The outputs, including the target token 𝑥𝑡and the acceptance
status of each draft token, are broadcast to notify both sides.
3)Upon receiving the message, each side checks the acceptance
status of both 𝑥𝑙
𝑡and𝑥𝑟
𝑡. If a token is accepted, it is dequeued
from the corresponding Draft Queue and step 5) is executed;
otherwise, step 4) is executed.
4)If𝑥𝑠
𝑡is rejected, its corresponding Draft Queues on both sides are
cleared and the side 𝑠rolls back its KV cache and re-computes
the next draft token 𝑥𝑠
𝑡+1using the target token 𝑥𝑡as input.
5) Update step 𝑡←𝑡+1, and go back to step 1).
We adopt a pipeline approach rather than performing aggrega-
tion in parallel. In centralized Speculative Decoding , each execution
of the target LLM requires waiting for the draft model to generate
the current token and for the LLM to verify the previously gener-
ated one. By verifying consecutive tokens in parallel, multiple LLM
inferences can be merged into a single pass, shifting the primary
latency bottleneck from target LLM inference to the sequential
decoding of draft tokens. Conversely, in Speculative Aggregation for
4The proof is included in Appendix A.2.
4

distributed RAG, the time to the next token is dominated by data
transmission over the network. Consecutive transmission of small
data can be naturally overlapped since each transmission does not
significantly occupy the I/O for an extended period. Parallelizing
the aggregation process instead introduces waiting between draft
tokens until the batch is fully populated. We employ queues to
construct a pipeline, where each draft token is transmitted and
enqueued immediately upon generation, ensuring it is verified at
the earliest opportunity.
4.3 Analysis of Acceptance Rate
We now analyze the factors that influence the acceptance rate of
draft tokens on both the device and the cloud sides.
Definition 4.1. For𝑠∈{𝑙,𝑟}, the acceptance rate 𝛽𝑠
𝑡, is the prob-
ability of accepting 𝑥𝑠
𝑡∼𝒑𝑠
𝑡=Í
𝑑∈𝐷𝑠𝜔𝑡(𝑑)𝒑(𝑥𝑡|𝑑,𝑥<𝑡)by the
aggregation strategy, given a prefix 𝑥<𝑡.
First, we consider the side 𝑙as an example. The acceptance
of the draft token 𝑥𝑙
𝑡, sampled from 𝒑𝑙
𝑡by the Decoder, can be
classified into two cases: i) it is accepted during the speculative
sampling of ˜𝑥𝑙
𝑡and ii) it is output by the speculative sampling
of˜𝑥𝑟
𝑡, where either 𝑥𝑟
𝑡=𝑥𝑙
𝑡and is accepted, or 𝑥𝑙
𝑡∼˜𝒑𝑟
𝑡. Let𝛾𝑙
and𝛾𝑟represent the weights assigned to ˜𝑥𝑙
𝑡and ˜𝑥𝑟
𝑡in the ran-
dom selection following these sampling processes ( 𝛾𝑙+𝛾𝑟=1).
We adopt the definition of divergence from [ 33], given by 𝛿=
𝐷𝐿𝐾(𝒑𝑙
𝑡,𝒑𝑟
𝑡)=1−Í
𝑥min(𝒑𝑙
𝑡(𝑥),𝒑𝑟
𝑡(𝑥)). The expected acceptance
rate𝛼𝑙
𝑡=E𝑥∼𝒑𝑙
𝑡(𝑥)(𝛽𝑙
𝑡)is computed as
𝛼𝑙
𝑡=𝛾𝑙(1−𝜂𝑟
𝑡𝛿)+𝛾𝑟∑︁
𝑥𝒑𝑙
𝑡(𝑥)𝒑𝑡(𝑥), (5)
where the two terms represent the acceptance probability of the
two cases above, respectively. These terms are mutually-exclusive
and influenced by the mixture weights 𝛾𝑙and𝛾𝑟.
Theorem 4.2. Given any distributions 𝒑𝑙
𝑡and𝒑𝑟
𝑡, when𝜂𝑟
𝑡is fixed,
maximizing 𝛼𝑙
𝑡is equivalent to maximizing 𝛾𝑙.
Proof . Substituting 𝒑𝑡=𝜂𝑙
𝑡𝒑𝑙
𝑡+𝜂𝑟
𝑡𝒑𝑟
𝑡5and subtracting the two
terms in Equation (5)yields(1−𝜂𝑟
𝑡𝛿)−Í𝒑𝑙
𝑡𝒑𝑡=𝜂𝑙
𝑡(1−Í(𝑝𝑙
𝑡)2)+
𝜂𝑟
𝑡Í(min(𝒑𝑙
𝑡,𝒑𝑟
𝑡)−𝒑𝑙
𝑡𝒑𝑟
𝑡). Since 1>Í
𝑥(𝒑𝑙
𝑡)2,min(𝒑𝑙
𝑡,𝒑𝑟
𝑡) ≥
𝒑𝑙
𝑡𝒑𝑟
𝑡and0≤𝜂𝑙
𝑡,𝜂𝑟
𝑡≤1, it follows that 1−𝜂𝑟
𝑡𝛿≥Í𝒑𝑙
𝑡𝒑𝑡al-
ways holds, with equality holding only when 𝜂𝑙
𝑡=0,𝜂𝑟
𝑡=1and
𝛿=0. This condition implies that the two distributions 𝒑𝑙
𝑡and𝒑𝑟
𝑡
are completely disjoint. Consequently, maximizing 𝛾𝑙leads to the
maximization of the expected acceptance rate 𝛼𝑙
𝑡. □
For the side 𝑟, Theorem 4.2 holds symmetrically, where maximiz-
ing the acceptance of 𝑥𝑟
𝑡corresponds to maximizing 𝛾𝑟. Clearly, the
objectives on sides 𝑙and𝑟conflict with each other. For simplicity
in framework design, we adopt 𝛾𝑙=𝛾𝑟=0.5to strike a balance (as
shown in Algorithm 1).
The expected acceptance rate is then influenced by the degree
of overlap between the draft distributions on the two sides. When
the distributions 𝒑𝑙
𝑡and𝒑𝑟
𝑡perfectly coincide, i.e., the divergence
5For brevity, the variable 𝑥is omitted in the distribution notation throughout the
following proof.𝑥𝑙
𝑡−1𝑥𝑟
𝑡−1Waiting Time for 𝑥𝑙
𝑡and𝑥𝑟
𝑡
rejected accepted max(𝑐𝑙
dec,𝜑(𝑐𝑟
dec+𝑐𝑟
trans))
accepted rejected max(𝜑(𝑐𝑙
dec),𝑐𝑙
trans+𝑐𝑟
dec+𝑐𝑟
trans)
accepted accepted max(𝜑(𝑐𝑙
dec),𝜑(𝑐𝑟
dec+𝑐𝑟
trans))
rejected rejected max(𝑐𝑙
dec,𝑐𝑙
trans+𝑐𝑟
trans+𝑐𝑟
dec)
Table 1: Waiting time for the next pair of draft tokens 𝑥𝑙
𝑡and
𝑥𝑟
𝑡under different acceptance scenarios of the previous draft
tokens𝑥𝑙
𝑡−1and𝑥𝑟
𝑡−1.
𝛿becomes zero, the first term of Equation (5),1−𝜂𝑟
𝑡𝛿, reaches its
maximum value. Simultaneously, since the second term follows
∑︁
𝑥𝒑𝑙
𝑡(𝑥)𝒑𝑡(𝑥)≤√︃∑︁
𝑥𝒑𝑙
𝑡(𝑥)2∑︁
𝑥𝒑𝑡(𝑥)2,
based on the Cauchy-Schwarz inequality and achieves its maximum
when 𝒑𝑙
𝑡(𝑥)=𝒑𝑟
𝑡(𝑥)=𝒑𝑡(𝑥), the expected acceptance rate is max-
imized. Conversely, when the support sets of the two distributions
are completely disjoint, i.e., 𝛿=1, the product 𝒑𝑙
𝑡(𝑥)𝒑𝑡(𝑥)becomes
zero for every 𝑥, resulting in a minimized expected acceptance rate.
This characteristic provides insight into the principle behind
Speculative Aggregation : we assume that the device-side and cloud-
side RAG workflows generate similar results by default, allowing
them to asynchronously decode the next tokens without aggrega-
tion. Only when they disagree with each other, the acceptance is
adjusted by their aggregation weights 𝜂𝑙
𝑡and𝜂𝑟
𝑡.
5 Greedy Scheduling
To further minimize the latency 𝐶(𝐴𝑡,F)in Equation (3), We adap-
tively schedule which side performs the next aggregation after the
current one is completed. The principle behind this is to maximize
the overlap between the device-side and cloud-side decoding and
transmission processes, jointly considering dynamic computing
resources, network bandwidth, and acceptance of draft tokens.
5.1 Scheduling Strategy
Since predicting future acceptance is challenging due to dynamic
document relevance and LLM outputs, we employ a greedy strategy,
where at each step, we minimize the expected latency per token
based on current observations.
The latency per token, denoted as 𝑍𝑡, is computed as the average
duration between two consecutive aggregations. It can be viewed
as the waiting time for the next pair of draft tokens, 𝑥device
𝑡and
𝑥cloud
𝑡, including both decoding and transmission delays, as the ag-
gregation duration is negligible. For each side 𝑠∈{device,cloud},
let𝑐𝑠
decdenote the decoding delay of a draft token 𝑥𝑠
𝑡, and𝑐𝑠
transde-
note the transmission delay of this token and its auxiliary variables
from𝑠to the other side. Since the decoding and transmission pro-
cesses are asynchronous, they may still be ongoing when the sched-
uling algorithm is executed. Therefore, we define 𝜑(𝑇total(𝑢))=
max(0,𝑇total(𝑢)+𝑇begin(𝑢)−𝑇now)as a function that estimates the
remaining time of the total duration 𝑇totalto complete the process
𝑢, where𝑇begin and𝑇noware the beginning and current timestamps,
respectively. Let 𝑙be the side that currently performs aggregation
and𝑟be the other one. The best side is then selected as
𝑠∗=arg min𝑠∈{𝑙,𝑟}𝑍𝑠
𝑡(𝜑,𝑐𝑙
𝑑𝑒𝑐,𝑐𝑙
𝑡𝑟𝑎𝑛𝑠,𝑐𝑟
𝑑𝑒𝑐,𝑐𝑟
𝑡𝑟𝑎𝑛𝑠), (6)
5

cr
dec(cl
trans+cr
trans)
cr
dec cr
dec+(cl
trans+cr
trans)
Local Decoding Latency per Token0Latency Differencel-side is betterr-side is better
slope=1l
t
slope=1r
t
l
t=0.5,r
t=0.8
l
t=0.8,r
t=0.5
l
t=0.5,r
t=0.5
Figure 3: Difference in per-token latencies when side 𝑙and𝑟
performs aggregation, versus varying 𝑙-side decoding latency.
where𝑍𝑠
𝑡denotes the latency per token when 𝑠continuously per-
forms the aggregations in the future.
Next, we present the calculation of 𝑍𝑠
𝑡. Table 1 illustrates the
waiting time for the next pair of draft tokens after a previous aggre-
gation. To estimate an averaged 𝑍𝑠
𝑡over multiple future steps, rather
than enumerating all possible combinations of acceptance scenar-
ios, we assume each acceptance scenario repeats continuously6and
occurs with an expected probability given by the acceptance rate.
Therefore, the waiting time in Table 1 can be simplified to eliminate
the function 𝜑. First, assuming that draft tokens from 𝑟are always
accepted, the decoding process for consecutive draft tokens will be
continuous on 𝑟. In other words, the decoding of 𝑥𝑟
𝑡begins exactly
when𝑥𝑟
𝑡−1is decoded and ready for transmission. Therefore, we
have𝜑(𝑐𝑟
dec+𝑐𝑟
trans)=(𝑇begin+𝑐𝑟
trans−𝑇now)+𝑐𝑟
dec=𝑐𝑟
dec. Moreover,
since the aggregation process can exhaustively consume the token
pairs in the Draft Queues, 𝜑(𝑐𝑙
dec)<𝑐𝑙
decholds only when the wait-
ing time for 𝑥𝑟
𝑡dominates. Hence, max(𝜑(𝑐𝑙
dec),·)=max(𝑐𝑙
dec,·).
Finally,𝑍𝑙
𝑡is calculated as
𝛼𝑟
𝑡max(𝑐𝑙
dec,𝑐𝑟
dec)+(1−𝛼𝑟
𝑡)max(𝑐𝑙
dec,𝑐𝑟
dec+𝑐𝑙
trans+𝑐𝑙
trans).(7)
Symmetrically, 𝑍𝑟
𝑡is computed by exchanging 𝑙and𝑟in Equa-
tion (7). Based on this, we can conclude that when the local de-
coding latency 𝑐𝑙
deccannot cover the waiting time for draft tokens
from the other side, i.e., 𝑐𝑙
dec<𝑐𝑟
dec+𝑐𝑙
trans+𝑐𝑙
trans, minimizing the
overall latency 𝑍𝑙
𝑡requires maximizing the acceptance rate 𝛼𝑟
𝑡.
To decide the optimal side in Equation (6), we calculate the differ-
ence in latencies per token when side 𝑙and𝑟performs aggregation.
The result is presented as a piecewise function,
Δ𝑍𝑡= 
(1−𝛼𝑟
𝑡)rtt, 𝑐𝑙
dec≤𝑐𝑟
dec−rtt
(1−𝛼𝑙
𝑡)𝑗+(𝛼𝑙
𝑡−𝛼𝑟
𝑡)rtt, 𝑐𝑟
dec−rtt<𝑐𝑙
dec≤𝑐𝑟
dec
(1−𝛼𝑟
𝑡)𝑗+(𝛼𝑙
𝑡−𝛼𝑟
𝑡)rtt, 𝑐𝑟
dec<𝑐𝑙
dec≤𝑐𝑟
dec+rtt
(𝑎𝑙
𝑡−1)rtt, 𝑐𝑟
dec+rtt<𝑐𝑙
dec,(8)
where rtt=𝑐𝑙
trans+𝑐𝑟
trans, and𝑗is the difference in decoding la-
tencies,𝑐𝑟
dec−𝑐𝑙
dec. Accordingly, we select side 𝑟for aggregation
when Δ𝑍𝑡>0, and side𝑙otherwise. Figure 3 shows the influence
of varying acceptance rates on Δ𝑍𝑡. As the acceptance rate of draft
tokens from one side increases, the Scheduler tends to favor the
opposite side. Moreover, the relationship between 𝑐𝑙
decand𝑐𝑟
decalso
influences the strategy. For instance, when the decoding process
on one side becomes the latency bottleneck, aggregation is always
performed on that side, which is demonstrated by (1−𝛼𝑟
𝑡)rtt≥0
and(𝛼𝑙
𝑡−1)rtt≤0. Clearly, our strategy minimizes the likelihood
6Please refer to Appendix A.3 for pipeline illustrations of different cases.of repeated bottleneck decoding due to rejection, while maximiz-
ing the overlap between the decoding and transmission processes
across the two sides.
5.2 Profiling
The Profiler helps estimate a set of parameters required to compute
Equation (6), including the decoding delay ( 𝑐dec) on both the device
and cloud sides, the transmission delay ( 𝑐device
trans+𝑐cloud
trans) between
them, and the acceptance rates ( 𝛼). The Profiler operates in two
stages: i) offline and ii) runtime.
Offline stage. For each side 𝑠∈{device,cloud}, the Profiler mea-
sures the decoding delay by simulating the output-aggregation RAG
workflow locally. We randomly generate |𝐷𝑠|dummy text chunks
as retrieved documents with the same chunk size 𝑀as in the real
corpus. We use the dummy text directly as the prefix (without a
query prompt) and prefill its KV cache in advance. Next, we perform
auto-regressive decoding with the same batch size as during run-
time, until the context length reaches its maximum 𝑁. We record
the decoding delay 𝑐𝑠
dec(𝑡)at each step 𝑡=𝑀,...,𝑁−1by averag-
ing over multiple runs and fit the records 𝒄𝑠
dec=𝑐𝑠
dec(𝑡)
𝑡∈𝒕using
a least square errors (LSE) estimator, ˆ𝑐𝑠
dec=𝑘𝑠𝑎𝑡/𝑘𝑠
𝑏+𝑘𝑠𝑐, where the
coefficients 𝑘𝑠𝑎,𝑘𝑠
𝑏, and𝑘𝑠𝑐are synchronized across both sides.
We model the transmission delay as ˆ𝑐𝑠
trans(𝑔)=𝐿𝑠+𝑔/𝐵𝑠, where
𝑔represents the data size and 𝐿𝑠and𝐵𝑠correspond to the network
latency and bandwidth for transmitting data from 𝑠to the other
side. Since one-way delay is hardly measurable, we measure the
bi-directional delay ˆ𝑐device
trans+ˆ𝑐cloud
transaltogether. We utilize sockperf
to evaluate the round trip time 𝐿device+𝐿cloudand use iperf3 to
measure the bi-directional bandwidths.
Runtime stage. To assess decoding latency at runtime, the Decoder
on each side 𝑠∈{device,cloud}measures the duration ˜𝑐𝑠
dec(𝑡)of
decoding a draft token at step 𝑡using the time.perf_counter function
in Python. This measurement is then piggybacked onto the draft
token message for convenient synchronization. Next, the value
ofˆ𝑐𝑠
decis re-estimated with the intercept 𝑘𝑠𝑐frozen and the slope
updated as
(1−𝜁)𝑘𝑠𝑎+𝜁(˜𝑐𝑠
dec(𝑡)−𝑘𝑠𝑐)𝑡
(1−𝜁)𝑘𝑠
𝑏+𝜁𝑡2,
where𝜁is the weight on new observation. The estimation of trans-
mission delay is refined by means of two moving averages: a real-
time estimate and a historical moving average. For the former, we
update the round-trip time measurement 𝐿device+𝐿cloudat each
step𝑡using the ICMP-based network diagnostic tool, ping. In con-
trast, the sending and receiving bandwidth 𝐵𝑠are updated using
iperf3 every few tokens to avoid excessive overhead. Similarly, we
estimate acceptance rates using Equation 5 and apply a moving
average to prevent abrupt changes.
6 Theoretical Wall-Time Improvement
In this section, we present a theoretical analysis to demonstrate the
improvement in wall-time efficiency achieved by DRAGON over the
vanilla distributed RAG framework described in § 2.2. Specifically,
the synchronized aggregation strategy used in the vanilla RAG can
be viewed as a special case of speculative aggregation in which
draft tokens from both sides are consistently rejected. To facilitate
6

10 30 50 70 90
Decoding Latency cl
dec (ms)1.01.21.41.61.82.0Speedupr
t=0.50
10 30 50 70 90
Decoding Latency cl
dec (ms)510152025
1r
t=0.99
cr
dec=10 ms, rtt=100 ms
cr
dec=10 ms, rtt=300 ms
cr
dec=30 ms, rtt=100 ms
cr
dec=30 ms, rtt=300 msFigure 4: Theoretical speedup of DRAGON compared to the
vanilla distributed RAG vs. varying 𝑐𝑙
dec,𝑐𝑟
dec, rtt and𝛼𝑟
𝑡.
analysis, we assume the aggregation is always performed on the
device in following discussions.
Definition 6.1. Let𝑍𝑡and ˜𝑍𝑡be the expected per-token latencies
at step𝑡when using DRAGON and the vanilla distributed RAG,
respectively. Define the speedup as 𝑆𝑡=˜𝑍𝑡/𝑍𝑡.
Theorem 6.2. Given𝑙=device and𝑟=cloud , the speedup can
be described as a piecewise function dependent on the relationship
among𝑐𝑙
dec,𝑐𝑟
decand rtt, as follows:
1
𝑆𝑡= 
1−𝛼𝑟
𝑡
1+𝑐𝑟
dec/rtt, 𝑐𝑙
dec≤𝑐𝑟
dec
1−(1−𝑐𝑙
dec
𝑐𝑟
𝑑𝑒𝑐+rtt)𝛼𝑟
𝑡, 𝑐𝑟
dec<𝑐𝑙
dec≤𝑐𝑟
dec+rtt
1, 𝑐𝑟
dec+rtt<𝑐𝑙
dec(9)
Proof .𝑍𝑡is computed according to Equation (7). By substituting
𝛼𝑙
𝑡=𝛼𝑟
𝑡=0and we obtain ˜𝑍𝑡=max(𝑐𝑙
dec,𝑐𝑟
dec+rtt). The result
then follows from a simple case-by-case analysis. □
Figure 4 illustrates the theoretical speedup characterized in The-
orem 6.2. The speedup achieves its maximum when the device-side
decoding latency is minimal and maintains saturated until it sur-
passes that of the cloud. Thereafter, the speedup decreases inversely
with𝑐𝑙
dec, gradually approaching 1 and eventually stabilizing at 1
once𝑐𝑙
decexceeds𝑐𝑟
dec+rtt. Finally, we have following corollaries:
Corollary 6.3. DRAGON is particularly effective when the decod-
ing latency gap between the device and the cloud is small and the
transmission cost becomes the primary bottleneck.
This property broadens the potential application of DRAGON to
general scenarios in which distributed computing nodes have com-
parable computational resources, but communication remains a
key bottleneck requiring further optimization.
Corollary 6.4. DRAGON’s improvement in wall time can be sub-
stantially amplified when the cloud-side acceptance rate is high.
Numerous existing works [ 18,63] have shown that a small subset
of tokens receives the majority of attention and replacing them
significantly changes the output sequence [ 37]. Accordingly, we
argue that draft tokens that differ from those on the other side pri-
marily originate from this subset and are synchronized across both
sides. In contrast, other tokens (such as stop words, punctuations
and common-knowledge terms) are often context-independent and
shared across both sides, leading to a considerable acceptance rate.
Corollary 6.5. DRAGON’s improvement in wall time is independent
of the device-side acceptance rate.When the device-side decoding latency is much lower, the Aggre-
gator must wait for the arrival of cloud-side draft tokens before
generating the next target token, regardless of whether the device-
side draft is accepted. Similarly, when the device-side latency is
substantially higher, the next target token is generated immediately
and fed as input for the next decoding step after completing the
current one. As a result, the acceptance of the local draft has no
impact on the overall latency. However, it remains important when
aggregation is shifted to the other side via DRAGON’s scheduling
algorithm.
7 Experiments
7.1 Implementation
We implemented DRAGON for distributed RAG workflow compris-
ing ~3,000 lines of Python code.7The System consists of two sym-
metric processes, the device-side and cloud-side ones, each utilizing
eight threads for core functionalities (e.g., decoding, aggregation
and transmission) along with a memory-resident service process
for document retrieval. We implemented information synchroniza-
tion between threads using multi-producer, multi-consumer queues,
and between processes using socket-based communication. We uti-
lized PyTorch [44] (version 2.6.0) for algorithm implementations,
Hugging Face Transformers [54] for LLM utilities, LangChain [31]
for document chunking, Sentence Transformers [48] for document
re-ranking, and Faiss [26] for indexing and similarity search of
document embeddings.
Efficient transmission. We implemented data transmission over
the TCP/IP protocol using the socket library. A fixed-length mes-
sage header is defined using the struct module, containing the
message type and body size. All Python objects in the message
body are finally serialized using the pickle.dumps() function and
compressed by means of an LZ4compressor, while numeric vec-
tors are first serialized with numpy.tobytes() . For transmitting the
output distributions 𝒑device
𝑡and𝒑cloud
𝑡, we employ an aggressive
top-𝑝selection strategy [ 16] with𝑝=0.8, encoding the selected
indices as unsigned 8-bit integers and the values as 16-bit floating-
point numbers. While preserving the inference performance, the
transmission data size is significantly reduced—by approximately
2,363 times when given the vocabulary size of 50,272—compared to
the unoptimized JSON-based implementation.
Preemptible generation. We implemented a hook function that
raises an exception upon the occurrence of a stop event (e.g., receiv-
ing a draft token rejection message) and registered it in the forward
pass of each model layer to enable layer-wise interruption of draft
decoding. When the generation caller catches the exception, it rolls
back the KV cache and attention masks based on the number of
generated target tokens so far and feeds the latest target token as
input to trigger re-computation.
7.2 Experiment Setups
Testbed. We evaluated our framework and baseline methods using
a high-performance computer as the cloud server and a MacBook
Pro as the edge device. The server is equipped with an Intel Xeon
Silver 4210R CPU, 64GB of memory, and a GeForce RTX 3090 GPU,
7Our code is available at GitHub: https://github.com/ThomasAtlantis/DRAGON
7

while the MacBook Pro features an Intel Core i7 CPU, 16GB of
memory, and no dedicated GPU. The cloud and the device are
connected via a 2.4 GHz Wi-Fi local-area network, with latency
and jitter measured by sockperf as 2ms and 6ms, respectively. To
simulate network jitter, we replay a predefined random latency
trace by adjusting the network interface controller (NIC) latency
using the traffic control tool, tc.
Datasets and metrics. We evaluated the long-sequence generation
performance of DRAGON on the large-scale language modeling
dataset WikiText [ 41], which comprises over 100 million tokens ex-
tracted from verified Good and Featured articles on Wikipedia. We
constructed retrieval corpora from the training sets of two different-
scale versions, WikiText2 and WikiText103. During evaluation, we
applied rolling windows of 1024 and 512 tokens, respectively, over
their test sets, using the first 1/8 of each window as the query for
retrieval and the remaining tokens for perplexity evaluation. To
further assess the efficiency of our method, we measure the time
to first token (TTFT) and per-token latency. In this measurement,
we used the retrieval corpus and index pre-built by Facebook from
a Wikipedia dump dated December 20, 2018, which contains 21
million documents.
Models and baselines. We evaluated our framework using OPT-
1.3B [62] and Qwen2.5-1.5B [57], with vocabulary sizes of 151,936
and 50,272, respectively. For language modeling and latency mea-
surement, we adopted Contriever [ 22] and DPR [ 29] as the retriev-
ers, respectively. Additionally, we employed ms-marco-MiniLM-L6-
v2 [48] for document re-ranking. We compare DRAGON with four
baseline methods:
•CRCG, centralized generation augmented with centralized re-
trieval from local corpus, using the context-aggregation strategy,
which represents most existing RAG methods [24, 38, 46].
•DRCG, on-device generation augmented with documents re-
trieved from a distributed corpus spanning both the device and
the cloud, using the context-aggregation strategy.
•DRDG/TW, distributed RAG using the output aggregation strat-
egy and token-wise synchronization, as discussed in § 2.2. The
target tokens are collected and aggregated on the device side.
•DRDG/SW, distributed RAG using the output aggregation strat-
egy and sequence-wise synchronization, i.e., one-time aggrega-
tion of the independently generated output sequences from the
device and the cloud. This baseline is implemented by extend-
ing the official REPLUG [ 51] implementation and Facebook’s
RAG-Sequence model [36] with distributed support.
To simulate insufficient but complementary corpus in the cloud and
device sides, we constrain the on-cloud and on-device retrieval by
selecting the first and second halves of the top-k documents from
the same corpus, respectively. Moreover, to study the overhead of
DRCG, we evaluate two variants: DRCG/Text retrieves raw text and
prefill KV cache from scratch and DRCG/KV retrieves and reuses
the KV cache of documents directly.
7.3 Overall Performance and Efficiency
We first present the overall performance and efficiency of DRAGON
in comparison to the baselines. In the following experiments, we
set the maximum context length to 256 tokens on both the device
and cloud sides, with each retrieved document limited to 64 tokens.
/uni00000013/uni00000015/uni00000017/uni00000019/uni0000001b/uni00000014/uni00000013/uni00000014/uni00000015/uni00000014/uni00000017/uni00000014/uni00000019
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000035/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000048/uni00000059/uni00000048/uni00000047/uni00000003/uni00000027/uni00000052/uni00000046/uni00000058/uni00000050/uni00000048/uni00000051/uni00000057/uni00000056/uni00000014/uni00000013/uni00000011/uni00000015/uni00000014/uni00000013/uni00000011/uni00000017/uni00000014/uni00000013/uni00000011/uni00000019/uni00000014/uni00000013/uni00000011/uni0000001b/uni00000033/uni00000048/uni00000055/uni00000053/uni0000004f/uni00000048/uni0000005b/uni0000004c/uni00000057/uni0000005c
/uni00000013/uni00000015/uni00000017/uni00000019/uni0000001b/uni00000014/uni00000013/uni00000014/uni00000015/uni00000014/uni00000017/uni00000014/uni00000019
/uni00000031/uni00000058/uni00000050/uni00000045/uni00000048/uni00000055/uni00000003/uni00000052/uni00000049/uni00000003/uni00000035/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000048/uni00000059/uni00000048/uni00000047/uni00000003/uni00000027/uni00000052/uni00000046/uni00000058/uni00000050/uni00000048/uni00000051/uni00000057/uni00000056/uni00000014/uni00000017/uni00000011/uni00000015/uni00000014/uni00000017/uni00000011/uni00000017/uni00000014/uni00000017/uni00000011/uni00000019/uni00000014/uni00000017/uni00000011/uni0000001b/uni00000014/uni00000018/uni00000011/uni00000013/uni00000033/uni00000048/uni00000055/uni00000053/uni0000004f/uni00000048/uni0000005b/uni0000004c/uni00000057/uni0000005c
/uni0000005a/uni00000012/uni00000052/uni00000003/uni00000035/uni00000048/uni00000057/uni00000055/uni0000004c/uni00000048/uni00000059/uni00000044/uni0000004f /uni00000026/uni00000035/uni00000026/uni0000002a/uni00000012/uni00000026/uni0000004f/uni00000052/uni00000058/uni00000047 /uni00000026/uni00000035/uni00000026/uni0000002a/uni00000012/uni00000027/uni00000048/uni00000059/uni0000004c/uni00000046/uni00000048 /uni00000027/uni00000035/uni00000026/uni0000002a /uni00000027/uni00000035/uni00000024/uni0000002a/uni00000032/uni00000031(a) Qwen2.5-1.5B/WikiText2. (b) OPT-1.3B/WikiText103.
Figure 5: Performance on WikiText.
/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013 /uni00000014/uni00000018/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000015/uni00000018/uni00000013 /uni00000016/uni00000013/uni00000013
/uni00000028/uni0000005b/uni00000057/uni00000055/uni00000044/uni00000003/uni00000031/uni00000048/uni00000057/uni0000005a/uni00000052/uni00000055/uni0000004e/uni00000003/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000050/uni00000056/uni0000000c/uni00000013/uni00000014/uni00000013/uni00000013/uni00000015/uni00000013/uni00000013/uni00000016/uni00000013/uni00000013/uni00000017/uni00000013/uni00000013/uni00000018/uni00000013/uni00000013/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni00000053/uni00000048/uni00000055/uni00000003/uni00000037/uni00000052/uni0000004e/uni00000048/uni00000051/uni00000003/uni0000000b/uni00000050/uni00000056/uni0000000c
/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013 /uni00000014/uni00000018/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000015/uni00000018/uni00000013 /uni00000016/uni00000013/uni00000013
/uni00000028/uni0000005b/uni00000057/uni00000055/uni00000044/uni00000003/uni00000031/uni00000048/uni00000057/uni0000005a/uni00000052/uni00000055/uni0000004e/uni00000003/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000050/uni00000056/uni0000000c/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013/uni00000014/uni00000018/uni00000013/uni00000015/uni00000013/uni00000013/uni00000015/uni00000018/uni00000013/uni00000016/uni00000013/uni00000013/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni00000053/uni00000048/uni00000055/uni00000003/uni00000037/uni00000052/uni0000004e/uni00000048/uni00000051/uni00000003/uni0000000b/uni00000050/uni00000056/uni0000000c
/uni00000026/uni00000035/uni00000026/uni0000002a/uni00000012/uni00000027/uni00000048/uni00000059/uni0000004c/uni00000046/uni00000048 /uni00000026/uni00000035/uni00000026/uni0000002a/uni00000012/uni00000026/uni0000004f/uni00000052/uni00000058/uni00000047 /uni00000027/uni00000035/uni00000027/uni0000002a/uni00000012/uni00000036/uni0000003a /uni00000027/uni00000035/uni00000027/uni0000002a/uni00000012/uni00000037/uni0000003a /uni00000027/uni00000035/uni00000024/uni0000002a/uni00000032/uni00000031
(a) Qwen2.5-1.5B. (b) OPT-1.3B.
Figure 6: Per-token latency in various network conditions.
/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013 /uni00000014/uni00000018/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000015/uni00000018/uni00000013 /uni00000016/uni00000013/uni00000013
/uni00000028/uni0000005b/uni00000057/uni00000055/uni00000044/uni00000003/uni00000031/uni00000048/uni00000057/uni0000005a/uni00000052/uni00000055/uni0000004e/uni00000003/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000050/uni00000056/uni0000000c/uni00000013/uni00000017/uni0000001b/uni00000014/uni00000015/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni00000057/uni00000052/uni00000003/uni00000029/uni0000004c/uni00000055/uni00000056/uni00000057/uni00000003/uni00000037/uni00000052/uni0000004e/uni00000048/uni00000051/uni00000003/uni0000000b/uni00000056/uni0000000c
/uni00000013/uni00000018/uni00000013/uni00000014/uni00000013/uni00000013 /uni00000014/uni00000018/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000015/uni00000018/uni00000013 /uni00000016/uni00000013/uni00000013
/uni00000028/uni0000005b/uni00000057/uni00000055/uni00000044/uni00000003/uni00000031/uni00000048/uni00000057/uni0000005a/uni00000052/uni00000055/uni0000004e/uni00000003/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000050/uni00000056/uni0000000c/uni00000013/uni00000018/uni00000014/uni00000013/uni00000014/uni00000018/uni00000015/uni00000013/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni00000057/uni00000052/uni00000003/uni00000029/uni0000004c/uni00000055/uni00000056/uni00000057/uni00000003/uni00000037/uni00000052/uni0000004e/uni00000048/uni00000051/uni00000003/uni0000000b/uni00000056/uni0000000c
/uni00000027/uni00000035/uni00000026/uni0000002a/uni00000012/uni00000037/uni00000048/uni0000005b/uni00000057 /uni00000027/uni00000035/uni00000026/uni0000002a/uni00000012/uni0000002e/uni00000039 /uni00000027/uni00000035/uni00000024/uni0000002a/uni00000032/uni00000031 /uni00000026/uni00000035/uni00000026/uni0000002a/uni00000012/uni00000027/uni00000048/uni00000059/uni0000004c/uni00000046/uni00000048 /uni00000026/uni00000035/uni00000026/uni0000002a/uni00000012/uni00000026/uni0000004f/uni00000052/uni00000058/uni00000047
(a) Qwen2.5-1.5B. (b) OPT-1.3B.
Figure 7: Time-to-First-Token in various network conditions.
Performance. We linearly increase the number of retrieved docu-
ments on both the device and the cloud sides from 0 to 16 and report
the corresponding language modeling perplexity on WikiText. As
shown in Figure 5, DRAGON matches or outperforms all baseline
methods across all settings. As more documents are integrated,
the performance gap between DRAGON and the baseline methods
widens. Finally, DRAGON achieves 1.9×and1.4×improvements
over the non-RAG method, compared to the second-best RAG base-
lines, for Qwen and OPT, respectively. In contrast, CRCG methods
perform poorly due to an insufficient number of retrieved docu-
ments, which indicates incomplete knowledge for the given context.
Additionally, the performance of DRCG quickly saturates once the
amount of retrieved text reaches the context budget limit. How-
ever, we observe a gap between DRCG and our method prior to
the saturation, suggesting that output aggregation may inherently
outperform context aggregation. The results of DRDG methods are
omitted, as they produce identical outputs to DRAGON under the
language modeling setting.
Efficiency. We inject additional latency to the server’s NIC, ranging
from 0 to 300 ms, along with a jitter equal to 1/5 of the corresponding
latency value. We sample prompts from 10k_prompts_ranked [21],
a collection of synthetic and human-generated prompts with asso-
ciated ranking, and report the average end-to-end decoding latency
8

over 20 output tokens8. Figure 6 presents the per-token latency
when incorporating the top-2 relevant documents for the RAG pro-
cess on each side. As shown in the figure, DRAGON demonstrates
strong robustness under different network conditions compared to
other distributed baseline methods. Specifically, DRAGON achieves
latency reduction of 49.5% and 42.4% when using OPT-1.3B com-
pared to the sequence-wise and token-wise DRDG methods, re-
spectively. In contrast, the per-token latency of DRDG methods
fluctuates significantly and tends to increase under higher network
latency conditions. Sequence-wise DRDG collects output distribu-
tions of all tokens once after generation completes, resulting in
a one-time large data transmission and increased sensitivity to
network latency. Token-wise DRDG amortizes data transmission
over the entire generation process, partially hiding latency within
decoding. However, it still under-performs compared to DRAGON
due to frequent output synchronizations. Additionally, DRCG meth-
ods yields the same per-token latency with corresponding CRCG
methods, because they do not involve cooperation between the
device and the cloud. Although DRAGON incurs an average la-
tency overhead of 15.6%–20.3% compared to device-only methods,
it effectively supports tasks that require both personal and general
knowledge, where device-only or cloud-only methods may fail.
We further compare the TTFT of DRAGON with that of the
baseline methods under identical network conditions. TTFT typi-
cally includes the time for document retrieval and the latency of
the prefill stage, during which the key-value (KV) activations for
the concatenation of retrieved documents and the input query are
either computed from scratch in parallel or loaded from cache. As
shown in Figure 7, DRAGON incurs negligible TTFT overhead com-
pared to the device-only CRCG method. In contrast, as KV cache
is hosted on the same side with the corpus, DRCG/Text performs
prefill from scratch, resulting in high computation latency and 8.6×
TTFT on average compared to DRAGON. DRCG/KV directly fetches
KV activations from the server, leading to increased transmission
time under higher network latency and yielding over 15.3×TTFT
compared to DRAGON, rendering it entirely impractical. Notably,
DRCG/Text incurs larger prefill latency when using Qwen2.5-1.5B
compared to OPT-1.3B, due to its larger number of parameters. In
contrast, DRCG/KV exhibits higher TTFT on OPT-1.3B, as Qwen2.5-
1.5B employs Grouped-Query Attention (GQA [ 2]) to reduce the
size of KV activations. The transmission data size in DRCG/KV is
114 MB for OPT-1.3B and 16 MB for Qwen2.5-1.5B when retrieving
2 documents of 64 tokens each. Latency for local document retrieval
is measured at 52.6 ms, while latency for remote raw-text retrieval
ranges from 107.2 ms to 745.2 ms as extra network latency increases
from 0 to 300 ms.
7.4 Effectiveness of Scheduling
To thoroughly evaluate the effectiveness of scheduling, we imple-
mented a simulator to run DRAGON repeatedly using different
scheduling strategies under consistent settings. We compare our
scheduling strategy with three baseline methods: (1) Cloud and (2)
Device , where aggregation is statically performed in the cloud and
8Despite averaging, the results still exhibits fluctuations due to varying CPU load and
network jitter, but do not affect the overall conclusion.
/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000016/uni00000013/uni00000013 /uni00000017/uni00000013/uni00000013
/uni00000028/uni0000005b/uni00000057/uni00000055/uni00000044/uni00000003/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000050/uni00000056/uni0000000c/uni00000013/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013/uni00000017/uni00000013/uni00000037/uni00000052/uni00000057/uni00000044/uni0000004f/uni00000003/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c
/uni00000013 /uni00000014/uni00000013/uni00000013 /uni00000015/uni00000013/uni00000013 /uni00000016/uni00000013/uni00000013 /uni00000017/uni00000013/uni00000013
/uni00000028/uni0000005b/uni00000057/uni00000055/uni00000044/uni00000003/uni0000002f/uni00000044/uni00000057/uni00000048/uni00000051/uni00000046/uni0000005c/uni00000003/uni0000000b/uni00000050/uni00000056/uni0000000c/uni00000013/uni00000014/uni00000013/uni00000015/uni00000013/uni00000016/uni00000013/uni00000017/uni00000013/uni00000037/uni00000052/uni00000057/uni00000044/uni0000004f/uni00000003/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000026/uni0000004f/uni00000052/uni00000058/uni00000047 /uni00000035/uni00000044/uni00000051/uni00000047/uni00000052/uni00000050 /uni00000027/uni00000048/uni00000059/uni0000004c/uni00000046/uni00000048 /uni00000027/uni00000035/uni00000024/uni0000002a/uni00000032/uni00000031(a) Qwen2.5-1.5B. (b) OPT-1.3B.
Figure 8: Comparison of different scheduling strategies.
/uni00000027/uni00000035/uni00000024/uni0000002a/uni00000032/uni00000031
/uni00000033/uni0000004c/uni00000053/uni00000048/uni0000004f/uni0000004c/uni00000051/uni00000048/uni00000026/uni0000004f/uni00000052/uni00000058/uni00000047/uni00000027/uni00000048/uni00000046/uni00000052/uni00000047/uni00000048 /uni00000027/uni00000055/uni00000044/uni00000049/uni00000057/uni00000037/uni00000052/uni0000004e/uni00000048/uni00000051 /uni00000037/uni00000044/uni00000055/uni0000004a/uni00000048/uni00000057/uni00000037/uni00000052/uni0000004e/uni00000048/uni00000051 /uni00000036/uni0000005a/uni0000004c/uni00000057/uni00000046/uni0000004b/uni00000024/uni0000004a/uni0000004a/uni00000055/uni00000048/uni0000004a/uni00000044/uni00000057/uni00000052/uni00000055
/uni00000027/uni00000048/uni00000059/uni0000004c/uni00000046/uni00000048
/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000019/uni00000035/uni00000037/uni00000037/uni00000003/uni0000000b/uni00000056/uni0000000c
/uni00000010/uni00000013/uni00000011/uni00000018/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000018/uni00000014/uni00000011/uni00000013Z
 /uni00000026/uni0000004f/uni00000052/uni00000058/uni00000047 /uni00000027/uni00000048/uni00000059/uni0000004c/uni00000046/uni00000048
/uni00000013/uni00000011/uni00000018 /uni00000014/uni00000011/uni00000014/uni00000014/uni00000011/uni00000016/uni00000014/uni00000011/uni00000018/uni00000014/uni00000011/uni0000001b/uni00000015/uni00000011/uni00000013 /uni00000015/uni00000011/uni00000017/uni00000015/uni00000011/uni00000019/uni00000015/uni00000011/uni0000001c/uni00000016/uni00000011/uni00000014/uni00000016/uni00000011/uni00000017 /uni00000016/uni00000011/uni0000001a/uni00000017/uni00000011/uni00000013/uni00000017/uni00000011/uni00000016/uni00000017/uni00000011/uni00000018/uni00000037/uni0000004c/uni00000050/uni00000048/uni00000003/uni0000000b/uni00000056/uni0000000c/uni00000013/uni00000011/uni00000013/uni00000013/uni00000011/uni00000018/uni00000014/uni00000011/uni00000013/uni00000024/uni00000046/uni00000046/uni00000011
/uni00000026/uni0000004f/uni00000052/uni00000058/uni00000047 /uni00000027/uni00000048/uni00000059/uni0000004c/uni00000046/uni00000048
Figure 9: A random snapshot of the generation pipeline and
scheduling decisions of DRAGON.
the device, respectively, and (3) Random , which randomly selects
the side for aggregation.
To implement the simulation, we record and replay the accep-
tance decisions of the Aggregator, and use real-world measurements
of decoding latency on each side. We simulate varying network
conditions by adding an extra latency and a sinusoidal jitter to
the measured base latency. The period of the jitter is set to 20𝜋
seconds with its amplitude set to 1/5 of the corresponding latency,
consistent with the settings in § 7.3.
Figure 8 presents the total time required to generate 100 tokens
under varying network conditions, each averaged over 50 different
acceptance decision sequences. The results show that DRAGON’s
scheduling strategy matches or outperforms all baselines across all
settings, with the efficiency gains increasing as the extra latency
grows. Due to the substantial gap in decoding latencies between the
device and the cloud (as shown in Figure 6), performing aggregation
on the device naturally hides cloud-side decoding and transmis-
sion within device-side decoding. When network latency is low,
Cloud andRandom tend to incur higher latency while DRAGON
consistently selects the device side for aggregation. As network
latency grows and transmission becomes the bottleneck, DRAGON
dynamically selects the side with higher acceptance rate to mini-
mize transmission resulted from draft rejection. Finally, we argue
that when device-side and cloud-side decoding latencies become
closer in value, the overall generation time will be more sensitive
to the network latency. In that case, our scheduling strategy will
achieve greater improvement compared to these baseline methods.
Case study. To illustrate DRAGON’s detailed scheduling process,
we present a 15-token snapshot of a random simulation with the
extra latency set to 500 ms. Figure 9 shows, from top to bottom,
9

the cloud-side and device-side generation pipelines, the instanta-
neous RTT, the estimation score Δ𝑍as defined in Equation (8), and
the accumulated acceptance rates. The pipeline graph comprises
vertically arranged bars representing decoding and different trans-
mission tasks (including transmission of draft tokens, target tokens
and instruction signals for switching aggregation place).
Initially, the Aggregator resides on the device by default. From
the perspective of the device, 𝑐𝑟
dec<𝑐𝑙
dec≤𝑐𝑟
dec+rttconsistently
holds and Δ𝑍is computed as the sum of two terms, 𝐴=(1−
𝛼𝑟
𝑡)(𝑐𝑟
dec−𝑐𝑙
dec)and𝐵=(𝛼𝑙
𝑡−𝛼𝑟
𝑡)rtt. After the first aggregation at 0.5
s, the acceptance rates are updated to 𝛼𝑙
0=1and𝛼𝑟
0=0. As a result,
the positive term 𝐵dominates and Δ𝑍>0. The Scheduler decides
to switch the Aggregator to the cloud, sending the switching signal
along with the target token. It then shifts to the cloud’s perspective
and reverses the sign of Δ𝑍. Subsequently, since the accumulated
cloud-side acceptance rate remains lower, the Scheduler continues
to estimating Δ𝑍<0, indicating that cloud-side aggregation is
more efficient. This case shows that DRAGON’s scheduling strategy
dynamically minimizes decoding and transmission costs on the side
with a lower acceptance rate, which is consistent with our analysis
in § 5.1 and the results shown in Figure 8.
8 Related Works
RAG with multiple documents. RAG approaches commonly
retrieve multiple documents to improve performance during in-
ference [ 3], but the way of aggregating them primarily diverge
into two categories: output aggregation and context aggregation
(§ 2.1). For output aggregation, pioneering works [ 15,36] prior to
LLMs have proven its effectiveness for encoder-only and seq2seq
models on both extractive [ 15] and abstractive [ 36] NLP tasks. RE-
PLUG [ 51] expands this method to off-the-shelf decoder-only LLMs
by fine-tuning a dense retriever. CAD [ 50] adopts the same idea
to strike a balance between retrieval-augmented outputs and LLM-
only outputs. RA-CM3 [ 59] enables few-shot image classification
for multimodal language model by aggregating the predictions
given different retrieved examples. Context aggregation prepend
the concatenation of all documents to the input and is adopted by a
line of in-context RAG methods [ 24,38,46] for simplicity. PCW [ 47]
eliminates cross-attentions between documents to mitigate the high
computational overhead introduced by this architecture. Our frame-
work leverages output aggregation to facilitate the decomposition
of the multi-document RAG workflow across the device and the
cloud, whereas existing works adopt a centralized architecture.
Device-cloud collaborative inference. To simultaneously achieve
privacy preservation and low latency in mobile computing while
benefiting from the robust computational power of cloud, numerous
studies [ 6,19,28,32,61] have investigated device-cloud collabo-
rative inference for conventional neural networks. Recently, this
collaborative inference paradigm has been extended to large lan-
guage models [ 43,45]. CE-CoLLM [ 25] splits LLMs along depth
and offloads deeper layers to the cloud, with a context manager to
cache and reuse transmitted intermediate hidden states. Crayon [ 5]
offloads difficult or non-customized tasks to a more capable cloud-
hosted LLM rather than the on-device SLM. However, only a fewexisting works have explored enhancing on-device RAG with cloud-
side knowledge. Hybrid-RACA [ 56] implements a real-time compo-
sition assistant, in which cloud-side documents are retrieved, com-
pressed by an LLM and subsequently downloaded to enhance an
on-device SLM. [ 14] utilizes user’s historical interactions with the
cloud-based LLM to enhance on-device kNN-LMs [ 30]. These works
prioritize service availability over privacy preservation, retrieving
information from a single database processed by LLMs instead
of employing inter-model collaborative generation. In contrast,
DRAGON adopts a symmetric architecture, leveraging databases
on both the device and cloud sides, enabling model collaboration
without compromising document privacy.
Speculative decoding. Speculative decoding, initially proposed
in [55], accelerates the sequential decoding process of LLMs through
a draft-then-verify paradigm, where at each decoding step, mul-
tiple consecutive future tokens are efficiently drafted by a small
LM, and then verified in parallel by the target LLM. Concurrent
studies by [ 34] and [ 10] introduced Speculative Sampling, extend-
ing this paradigm to support diverse sampling strategies. These
works utilize readily available smaller language models from the
same model family as the target LLM for drafting, thus avoiding
additional training. Another line of research directly utilizes the
target LLM for drafting. Medusa [8] and Blockwise Decoding [53]
integrate feed-forward network (FFN) heads into the Transformer
decoder, enabling parallel generation of draft tokens per step. Other
works [ 17,58,60] have investigated early exiting and layer skip-
ping within the target LLM to implement drafting. In contrast to
speculative decoding, where a single drafter fast predicts the output
of the target LLM, speculative aggregation in DRAGON verifies the
consistency between outputs generated by two distinct LLMs.
9 Conclusion
To address privacy risks of cloud LLMs and limited capabilities of on-
device SLMs, we propose DRAGON, a distributed RAG framework
that enhances on-device SLMs using both personal and general
knowledge without raw document transmission. DRAGON parti-
tions the RAG workflow across device and cloud, using Speculative
Aggregation to minimize output synchronization overhead. Experi-
mental results show that DRAGON notably improves generation
quality while maintaining low latency.
References
[1]Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla,
Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary,
Congcong Chen, et al .2025. Phi-4-mini technical report: Compact yet powerful
multimodal language models via mixture-of-loras. arXiv:2503.01743
[2] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico
Lebrón, and Sumit Sanghai. 2023. GQA: Training Generalized Multi-Query
Transformer Models from Multi-Head Checkpoints. In EMNLP . 4895–4901.
[3] Akari Asai, Zexuan Zhong, Danqi Chen, Pang Wei Koh, Luke Zettlemoyer, Han-
naneh Hajishirzi, and Wen-tau Yih. 2024. Reliable, adaptable, and attributable
language models with retrieval. arXiv preprint arXiv:2403.03187 (2024).
[4] AI at Meta. 2024. facebook/wiki_dpr. https://huggingface.co/datasets/facebook/
wiki_dpr. Accessed: 2025-04-01.
[5] Jihwan Bang, Juntae Lee, Kyuhong Shim, Seunghan Yang, and Simyung Chang.
2024. Crayon: Customized On-Device LLM via Instant Adapter Blending and
Edge-Server Hybrid Inference. In ACL. 3720–3731.
[6] Amin Banitalebi-Dehkordi, Naveen Vedula, Jian Pei, Fei Xia, Lanjun Wang, and
Yong Zhang. 2021. Auto-split: a general framework of collaborative edge-cloud
ai.SIGKDD (2021), 2543–2553.
[7] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Ruther-
ford, et al .2022. Improving language models by retrieving from trillions of
10

tokens. In ICML . 2206–2240.
[8] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming
Chen, and Tri Dao. 2024. Medusa: Simple LLM Inference Acceleration Framework
with Multiple Decoding Heads. In PMLR . 5209–5235.
[9] Neal Cardwell, Yuchung Cheng, C Stephen Gunn, Soheil Hassas Yeganeh, and Van
Jacobson. 2016. Bbr: Congestion-based congestion control: Measuring bottleneck
bandwidth and round-trip propagation time. Queue (2016), 20–53.
[10] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Lau-
rent Sifre, and John Jumper. 2023. Accelerating large language model decoding
with speculative sampling. arXiv:2302.01318
[11] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large
language models in retrieval-augmented generation. In AAAI . 17754–17762.
[12] Lars Daniel. 2025. DeepSeek Data Leak Exposes 1 Million Sensitive Records.
Forbes (2025). https://www.forbes.com/sites/larsdaniel/2025/02/01/deepseek-
data-leak-exposes--1000000-sensitive-records/
[13] DeepSeek-AI. 2024. DeepSeek-V3 Technical Report. arXiv:2412.19437
[14] Yucheng Ding, Chaoyue Niu, Fan Wu, Shaojie Tang, Chengfei Lyu, and Guihai
Chen. 2024. Enhancing on-device llm inference with historical cloud-based llm
interactions. In SIGKDD . 597–608.
[15] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang.
2020. Retrieval augmented language model pre-training. In ICML . 3929–3938.
[16] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2020. The
Curious Case of Neural Text Degeneration. In ICLR .
[17] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Hasan Genc, Kurt
Keutzer, Amir Gholami, and Yakun Sophia Shao. 2023. SPEED: Speculative
Pipelined Execution for Efficient Decoding. arXiv:2310.12072
[18] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney,
Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. 2024. KVQuant: Towards
10 Million Context Length LLM Inference with KV Cache Quantization. In NIPS .
1270–1303.
[19] Chuang Hu, Wei Bao, Dan Wang, and Fengming Liu. 2019. Dynamic adaptive
DNN surgery for inference acceleration on the edge. ICCC (2019), 1423–1431.
[20] Junxian Huang, Feng Qian, Alexandre Gerber, Z Morley Mao, Subhabrata Sen,
and Oliver Spatscheck. 2012. A close examination of performance and power
characteristics of 4G LTE networks. In MobiSys . 225–238.
[21] Data is Better-Together. 2024. data-is-better-together/10k_prompts_ranked.
https://huggingface.co/datasets/data-is-better-together/10k_prompts_ranked.
Accessed: 2025-03-31.
[22] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bo-
janowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised Dense Infor-
mation Retrieval with Contrastive Learning. https://arxiv.org/abs/2112.09118
[23] Gautier Izacard and Edouard Grave. 2021. Leveraging Passage Retrieval with
Generative Models for Open Domain Question Answering. In EACL . 874–880.
[24] Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-
Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active Retrieval
Augmented Generation. In EMNLP . 7969–7992.
[25] Hongpeng Jin and Yanzhao Wu. 2024. CE-CoLLM: Efficient and Adaptive Large
Language Models Through Cloud-Edge Collaboration. arXiv:2411.02829
[26] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs. IEEE Transactions on Big Data 3 (2019), 535–547.
[27] Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi
Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cor-
mode, Rachel Cummings, et al .2021. Advances and open problems in federated
learning. Foundations and Trends in Machine Learning 1–2 (2021), 1–210.
[28] Yiping Kang, Johann Hauswald, Cao Gao, Austin Rovinski, Trevor Mudge, Jason
Mars, and Lingjia Tang. 2017. Neurosurgeon: Collaborative intelligence between
the cloud and mobile edge. ACM SIGARCH Computer Architecture News 1 (2017),
615–629.
[29] Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey
Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-
domain question answering. In EMNLP . 6769–6781.
[30] Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike
Lewis. 2020. Generalization through Memorization: Nearest Neighbor Language
Models. In ICLR .
[31] LangChain. 2025. langchain-ai/langchain: Build context-aware reasoning appli-
cations. https://github.com/langchain-ai/langchain.
[32] Stefanos Laskaridis, Stylianos I Venieris, Mario Almeida, Ilias Leontiadis, and
Nicholas D Lane. 2020. SPINN: synergistic progressive inference of neural
networks over device and cloud. MobiCom (2020), 1–15.
[33] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from
transformers via speculative decoding. In ICML . 19274–19286.
[34] Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from
Transformers via Speculative Decoding. In ICML . 19274–19286.
[35] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rock-
täschel, et al .2020. Retrieval-augmented generation for knowledge-intensive
nlp tasks. In NIPS . 9459–9474.[36] Patrick S. H. Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks. In NIPS . 9459–9474.
[37] Zicheng Lin, Tian Liang, Jiahao Xu, Xing Wang, Ruilin Luo, Chufan Shi, Siheng
Li, Yujiu Yang, and Zhaopeng Tu. 2024. Critical Tokens Matter: Token-Level
Contrastive Estimation Enhence LLM’s Reasoning Capability. arXiv:2411.19943
[38] Hongyin Luo, Tianhua Zhang, Yung-Sung Chuang, Yuan Gong, Yoon Kim, Xixin
Wu, Helen Meng, and James Glass. 2023. Search Augmented Instruction Learning.
InEMNLP . 3717–3729.
[39] Yu Ma, Weifa Liang, Jing Li, Xiaohua Jia, and Song Guo. 2020. Mobility-aware
and delay-sensitive service provisioning in mobile edge-cloud networks. TMC 1
(2020), 196–210.
[40] Pavel Mach and Zdenek Becvar. 2017. Mobile Edge Computing: A Survey on
Architecture and Computation Offloading. IEEE Communications Surveys and
Tutorials 3 (2017), 1628–1656.
[41] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016.
Pointer Sentinel Mixture Models. arXiv:1609.07843 [cs.CL]
[42] OpenAI. 2023. GPT-4 Technical Report. arXiv:2303.08774
[43] Yanghe Pan, Zhou Su, Yuntao Wang, Shaolong Guo, Han Liu, Ruidong Li, and
Yuan Wu. 2024. Cloud-Edge Collaborative Large Model Services: Challenges and
Solutions. IEEE Network (2024), 1–1.
[44] Adam Paszke, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang,
Zachary DeVito, Zeming Lin, Alban Desmaison, Luca Antiga, and Adam Lerer.
2017. Automatic differentiation in PyTorch.
[45] Guanqiao Qu, Qiyuan Chen, Wei Wei, Zheng Lin, Xianhao Chen, and Kaibin
Huang. [n. d.]. Mobile Edge Intelligence for Large Language Models: A Contem-
porary Survey. arXiv:2407.18921
[46] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin
Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language
models. TACL (2023), 1316–1331.
[47] Nir Ratner, Yoav Levine, Yonatan Belinkov, Ori Ram, Inbal Magar, Omri Abend,
Ehud Karpas, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023.
Parallel Context Windows for Large Language Models. In ACL. 6383–6402.
[48] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
using Siamese BERT-Networks. In EMNLP . Association for Computational Lin-
guistics. https://arxiv.org/abs/1908.10084
[49] Rulin Shao, Jacqueline He, Akari Asai, Weijia Shi, Tim Dettmers, Sewon Min,
Luke Zettlemoyer, and Pang Wei Koh. 2024. Scaling Retrieval-Based Language
Models with a Trillion-Token Datastore. In NIPS . 91260–91299.
[50] Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and
Wen-tau Yih. 2024. Trusting Your Evidence: Hallucinate Less with Context-aware
Decoding. In NAACL . 783–791.
[51] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike
Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. REPLUG: Retrieval-Augmented
Black-Box Language Models. In NAACL . 8371–8384.
[52] Milad Shokouhi, Luo Si, et al .2011. Federated Search. Foundations and Trends ®
in Information Retrieval 1 (2011), 1–102.
[53] Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. Blockwise Parallel
Decoding for Deep Autoregressive Models. In NIPS . 10107–10116.
[54] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement De-
langue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz,
et al.2019. Huggingface’s transformers: State-of-the-art natural language pro-
cessing. arXiv preprint arXiv:1910.03771 (2019).
[55] Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui.
2023. Speculative Decoding: Exploiting Speculative Execution for Accelerating
Seq2seq Generation. In EMNLP . 3909–3925.
[56] Menglin Xia, Xuchao Zhang, Camille Couturier, Guoqing Zheng, Saravan Ra-
jmohan, and Victor Rühle. 2024. Hybrid-RACA: Hybrid Retrieval-Augmented
Composition Assistance for Real-time Text Prediction. In EMNLP . 120–131.
[57] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al .2024. Qwen2.5
technical report. arXiv:2412.15115
[58] Seongjun Yang, Gibbeum Lee, Jaewoong Cho, Dimitris Papailiopoulos, and Kang-
wook Lee. 2024. Predictive Pipelined Decoding: A Compute-Latency Trade-off
for Exact LLM Decoding. TMLR (2024).
[59] Michihiro Yasunaga, Armen Aghajanyan, Weijia Shi, Rich James, Jure Leskovec,
Percy Liang, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. Retrieval-
augmented multimodal language modeling. In ICML . Article 1659, 15 pages.
[60] Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad
Mehrotra. 2024. Draft& Verify: Lossless Large Language Model Acceleration via
Self-Speculative Decoding. In ACL. 11263–11282.
[61] Shigeng Zhang, Yinggang Li, Xuan Liu, Song Guo, Weiping Wang, Jianxin Wang,
Bo Ding, and Di Wu. 2020. Towards real-time cooperative deep inference over the
cloud and edge end devices. ACM on Interactive, Mobile, Wearable and Ubiquitous
Technologies 2 (2020), 1–24.
[62] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui
Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov,
11

Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali
Sridhar, Tianlu Wang, and Luke Zettlemoyer. 2022. OPT: Open Pre-trained
Transformer Language Models. arXiv:2205.01068 [cs.CL]
[63] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi
Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang,
and Beidi Chen. 2023. H2O: heavy-hitter oracle for efficient generative inference
of large language models. In NIPS . Article 1506, 50 pages.
[64] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou,
Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al .2023. A survey
of large language models. arXiv:2303.18223
A Appendix
A.1 Numerical Stability
We leverage the log-sum-exp trick to enhance numerical stability.
Specifically, after decoding a draft token on each side 𝑠, the corrected
value ofℎ𝑠
𝑡is computed as
˜ℎ𝑠
𝑡=log∑︁
𝑑∈𝐷𝑠exp(R(𝑑,𝑥<𝑡)).
and is synchronized across both sides along with log𝒑𝑠
𝑡. During
aggregation, we compute log𝜂𝑠
𝑡as follows:
logsoftmax([˜ℎ𝑙
𝑡,˜ℎ𝑟
𝑡])=logexp˜ℎ𝑠
𝑡
exp˜ℎ𝑙
𝑡+exp˜ℎ𝑟
𝑡
=logÍ
𝑑∈𝐷𝑠expR(𝑑,𝑥<𝑡)Í
𝑑′∈𝐷𝑙∪𝐷𝑟expR(𝑑′,𝑥<𝑡)=logℎ𝑠
𝑡
ℎ𝑙
𝑡+ℎ𝑟
𝑡=log(𝜂𝑠
𝑡).
The log of the target distribution log𝒑𝑡is then obtained by:
log∑︁
𝑠∈{𝑙,𝑟}exp(log𝒑𝑠
𝑡+log𝜂𝑠
𝑡)=log(𝑝𝑙
𝑡𝜂𝑙
𝑡+𝑝𝑟
𝑡𝜂𝑟
𝑡)=log𝒑𝑡.
On one hand, both the log-sum-exp and log-softmax operations
are inherently numerically stable. On the other hand, since our
data compression algorithm only transmits the top- 𝑝values of
the locally-aggregated output distributions, it effectively avoids
numerical underflow of log𝒑𝑠
𝑡.
A.2 Correctness of the Aggregation Strategy
We will show that for any locally-aggregated distributions 𝒑𝑙
𝑡and
𝒑𝑟
𝑡, the target token 𝑥𝑡produced by the aggregation strategy follows
a distribution identical to that sampled from 𝒑𝑡=𝜂𝑙
𝑡𝒑𝑙
𝑡+𝜂𝑟
𝑡𝒑𝑟
𝑡,
where{𝑙,𝑟}={𝑑𝑒𝑣𝑖𝑐𝑒,𝑐𝑙𝑜𝑢𝑑}.
First, we demonstrate that the intermediate outputs ˜𝑥𝑙
𝑡and ˜𝑥𝑟
𝑡
from the two independent speculative sampling processes are in-
deed drawn from 𝒑𝑡. Note that, since 𝜂𝑠
𝑡=ℎ𝑠
𝑡/(ℎ𝑙
𝑡+ℎ𝑟
𝑡)for𝑠∈{𝑙,𝑟},
we have𝜂𝑙
𝑡+𝜂𝑟
𝑡=1.
For side𝑙, the probability to reject a draft token is
𝑃(𝑟𝑒𝑗𝑒𝑐𝑡𝑒𝑑)=𝐸𝑥∼𝒑𝑙
𝑡(𝑥)(1−min(1,𝜂𝑙
𝑡+𝜂𝑟
𝑡𝒑𝑟
𝑡(𝑥)/𝒑𝑙
𝑡(𝑥)))
=1−∑︁
min(𝒑𝑙
𝑡(𝑥),𝜂𝑙
𝑡𝒑𝑙
𝑡(𝑥)+𝜂𝑟
𝑡𝒑𝑟
𝑡(𝑥))
=1−∑︁
(𝒑𝑙
𝑡(𝑥)+min(0,𝜂𝑟
𝑡(𝒑𝑟
𝑡(𝑥)−𝒑𝑙
𝑡(𝑥))))
=𝜂𝑟
𝑡∑︁
−min(0,𝒑𝑟
𝑡(𝑥)−𝒑𝑙
𝑡(𝑥))
=𝜂𝑟
𝑡∑︁
(𝒑𝑙
𝑡(𝑥)−min(𝒑𝑙
𝑡(𝑥),𝒑𝑟
𝑡(𝑥))).
Dec  1 2 3 4 5 6Remote Tokens TransDec TransDec TransDec TransDecDec
TransTrans
Dec
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
Time 1 2 3 4 5 6Local TokensDecAggr. 1 Aggr. 2 Aggr. 3 Aggr. 4 Aggr. 5 Aggr. 6
TransDec
TransTrans
Dec TransDecDec TransDecDecDec
TransDecDec
TransDec TransFigure 10: Decoding pipeline when the Aggregator continu-
ously rejects 𝑥𝑙and accepts 𝑥𝑟.
The adjusted distribution, from which we sample after the draft
token is rejected, can be expressed as
˜𝒑𝑙
𝑡(𝑥)=norm(max(0,𝒑𝑟
𝑡(𝑥)−𝑝𝑙
𝑡(𝑥))
=norm(𝒑𝑟
𝑡(𝑥)−min(𝒑𝑙
𝑡(𝑥),𝒑𝑟
𝑡(𝑥)))
=𝒑𝑟
𝑡(𝑥)−min(𝒑𝑙
𝑡(𝑥),𝒑𝑟
𝑡(𝑥))
Í
𝑥′(𝒑𝑟
𝑡(𝑥′)−min(𝒑𝑙
𝑡(𝑥′),𝒑𝑟
𝑡(𝑥′))).
SinceÍ(𝒑𝑙
𝑡(𝑥)−min(𝒑𝑙
𝑡(𝑥),𝒑𝑟
𝑡(𝑥)))is equivalent toÍ(𝒑𝑟
𝑡(𝑥)
−min(𝒑𝑙
𝑡(𝑥),𝒑𝑟
𝑡(𝑥))),𝑃(𝑟𝑒𝑗𝑒𝑐𝑡𝑒𝑑,𝑥 =˜𝑥𝑙
𝑡), the probability that ˜𝑥𝑙
𝑡
is re-sampled after rejecting 𝑥𝑙
𝑡, is
𝑃(𝑟𝑒𝑗𝑒𝑐𝑡𝑒𝑑)˜𝒑𝑙
𝑡(˜𝑥𝑙
𝑡)=𝜂𝑟
𝑡(𝒑𝑟
𝑡(˜𝑥𝑙
𝑡)−min(𝒑𝑙
𝑡(˜𝑥𝑙
𝑡),𝒑𝑟
𝑡(˜𝑥𝑙
𝑡))).
Finally, the sampled token ˜𝑥𝑙
𝑡follows the distribution
𝑃(𝑥=˜𝑥𝑙
𝑡)=𝑃(𝑎𝑐𝑐𝑒𝑝𝑡𝑒𝑑,𝑥 =˜𝑥𝑙
𝑡)+𝑃(𝑟𝑒𝑗𝑒𝑐𝑡𝑒𝑑,𝑥 =˜𝑥𝑙
𝑡)
=𝒑𝑙
𝑡(˜𝑥𝑙
𝑡)min(1,𝜂𝑙
𝑡+𝜂𝑟
𝑡𝒑𝑟
𝑡(˜𝑥𝑙
𝑡)/𝒑𝑙
𝑡(˜𝑥𝑙
𝑡))
+𝜂𝑟
𝑡(𝒑𝑟
𝑡(˜𝑥𝑙
𝑡)−min(𝒑𝑙
𝑡(˜𝑥𝑙
𝑡),𝒑𝑟
𝑡(˜𝑥𝑙
𝑡)))
=𝜂𝑙
𝑡𝒑𝑙
𝑡(˜𝑥𝑙
𝑡)+𝜂𝑟
𝑡min(𝒑𝑙
𝑡(˜𝑥𝑙
𝑡),𝒑𝑟
𝑡(˜𝑥𝑙
𝑡))
+𝜂𝑟
𝑡𝒑𝑟
𝑡(˜𝑥𝑙
𝑡)−𝜂𝑟
𝑡min(𝒑𝑙
𝑡(˜𝑥𝑙
𝑡),𝒑𝑟
𝑡(˜𝑥𝑙
𝑡)))
=𝜂𝑙
𝑡𝒑𝑙
𝑡(˜𝑥𝑙
𝑡)+𝜂𝑟
𝑡𝒑𝑟
𝑡(˜𝑥𝑙
𝑡)=𝒑𝑡(˜𝑥𝑙
𝑡).
As a result, ˜𝑥𝑙
𝑡is distributed identically to tokens sampled from 𝒑𝑡.
Since the correctness proof for the other side 𝑟is symmetric, we
can conclude straightforwardly that ˜𝑥𝑟
𝑡∼𝒑𝑡.
Finally, the aggregation strategy randomly select either ˜𝑥𝑙
𝑡or
˜𝑥𝑟
𝑡as the target token 𝑥𝑡, with a uniform probability. Obviously,
𝑥𝑡∼0.5𝒑𝑡+0.5𝒑𝑡=𝒑𝑡.
A.3 Decoding Pipelines
Apart from the theoretical analysis of latency per token in Sec-
tion 5, we use pipeline graphs to illustrate scenarios where each
acceptance case repeats continuously. This is not necessarily how
pipelines occur in practice, but it provides us with an heuristics of
the scheduling strategy. In the following discussion, we define 𝑙as
the side responsible for aggregation (i.e., the local side) and 𝑟as the
other side (i.e., the remote side). We set random delays to analyze
specific cases where all time values are expressed in the same unit.
12

Dec 1 2 3 4 5 6Remote TokensDec
TransDec
TransTransDec Trans
DecDec
TransDec
TransTrans
DecDec
TransDec
TransTrans
DecDec
Dec
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
Time 1 2 3 4 5 6Local TokensAggregation 1 Aggregation 2 Aggregation 3
TransDec DecTrans
Dec
TransTrans
DecDec
TransDecTrans
Dec
TransFigure 13: Decoding pipeline when the Aggregator continu-
ously rejects both 𝑥𝑙and𝑥𝑟.
Dec  1 2 3 4 5 6Remote Tokens TransDec Trans Dec TransDec Trans Dec TransDec TransTrans Dec
Dec
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
Time 1 2 3 4 5 6Local TokensAggr. 1 Aggr. 2 Aggr. 3
Dec
TransTrans
Dec
TransDec
TransTrans
TransDec TransDecTrans
Trans
Figure 11: Decoding pipeline when the Aggregator continu-
ously accepts 𝑥𝑙and rejects 𝑥𝑟.
Dec  1 2 3 4 5 6Remote TokensDec
TransTransDecDec
TransTransDec TransDec Trans
Dec
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
Time 1 2 3 4 5 6Local TokensDecAggr. 1 Aggr. 2 Aggr. 3 Aggr. 4 Aggr. 5 Aggr. 6
TransDec
TransDec
TransDec
TransTransDec Trans
Figure 12: Decoding pipeline when the Aggregator continu-
ously accepts both 𝑥𝑙and𝑥𝑟.i) Continuously reject 𝑥𝑙and accept 𝑥𝑟.The pipeline is shown
in Figure 10, where 𝑐𝑙
trans=1.2,𝑐𝑟
trans=1.8,𝑐𝑙
dec=1, and𝑐𝑟
dec=1.5.
The latency bottleneck is 𝑐𝑟
dec, causing𝑙to wait for𝑥𝑟
1before the
first Aggregation. As 𝑙generates draft tokens faster, subsequent
aggregations begin upon the arrival of each 𝑥𝑟
𝑡. When𝑟decodes
faster, the bottleneck becomes 𝑐𝑙
dec. As a result, the latency per
token is max(𝑐𝑙
dec,𝑐𝑟
dec).
ii) Continuously accept 𝑥𝑙and reject 𝑥𝑟.The pipeline is
shown in Figure 11, where 𝑐𝑙
trans=1.5,𝑐𝑟
trans=1,𝑐𝑙
dec=2, and
𝑐𝑟
dec=2. Although𝑐𝑙
dec=𝑐𝑟
dec, local draft tokens do not require
transmission; therefore, the latency bottleneck thus lies on the
remote side. After each aggregation at step 𝑡,𝑙must wait for a
duration of 𝑐𝑙
trans+𝑐𝑟
dec+𝑐𝑟
trans, including the transmission of tar-
get token𝑥𝑡, as well as the decoding and transmission of the re-
mote draft token 𝑥𝑟
𝑡+1. Only when 𝑐𝑙
decexceeds this duration, the
bottleneck shifts to the local side. The latency per token is thus
max(𝑐𝑙
dec,𝑐𝑙
trans+𝑐𝑟
dec+𝑐𝑟
trans).
iii) Continuously accept both 𝑥𝑙and𝑥𝑟.The pipeline is shown
in Figure 12, where 𝑐𝑙
trans=1.5,𝑐𝑟
trans=1.8,𝑐𝑙
dec=1, and𝑐𝑟
dec=1.5.
Clearly, the bottleneck lies on the side with the larger decoding
delay. Consequently, the latency per token is max(𝑐𝑙
dec,𝑐𝑟
dec).
iv) Continuously reject both 𝑥𝑙and𝑥𝑟The pipeline is shown
in Figure 13, where 𝑐𝑙
trans=1.5,𝑐𝑟
trans=1.8,𝑐𝑙
dec=2, and𝑐𝑟
dec=
1. Since rejecting the local draft token resets 𝜑(𝑐𝑙
dec)to𝑐𝑙
dec, the
scenario is exactly the same as ii). The latency per token is computed
asmax(𝑐𝑙
dec,𝑐𝑙
trans+𝑐𝑟
dec+𝑐𝑟
trans).
13