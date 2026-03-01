# CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval

**Authors**: Xinzhao Li, Alptekin Vardar, Franz Müller, Navya Goli, Umamaheswara Rao Tida, Kai Ni, X. Sharon Hu, Thomas Kämpfe, Ruiyang Qin

**Published**: 2026-02-23 17:49:10

**PDF URL**: [https://arxiv.org/pdf/2602.20083v2](https://arxiv.org/pdf/2602.20083v2)

## Abstract
Deploying Retrieval-Augmented Generation (RAG) on edge devices is in high demand, but is hindered by the latency of massive data movement and computation on traditional architectures. Compute-in-Memory (CiM) architectures address this bottleneck by performing vector search directly within their crossbar structure. However, CiM's adoption for RAG is limited by a fundamental ``representation gap,'' as high-precision, high-dimension embeddings are incompatible with CiM's low-precision, low-dimension array constraints. This gap is compounded by the diversity of CiM implementations (e.g., SRAM, ReRAM, FeFET), each with unique designs (e.g., 2-bit cells, 512x512 arrays). Consequently, RAG data must be naively reshaped to fit each target implementation. Current data shaping methods handle dimension and precision disjointly, which degrades data fidelity. This not only negates the advantages of CiM for RAG but also confuses hardware designers, making it unclear if a failure is due to the circuit design or the degraded input data. As a result, CiM adoption remains limited. In this paper, we introduce CQ-CiM, a unified, hardware-aware data shaping framework that jointly learns Compression and Quantization to produce CiM-compatible low-bit embeddings for diverse CiM designs. To the best of our knowledge, this is the first work to shape data for comprehensive CiM usage on RAG.

## Full Text


<!-- PDF content starts -->

CQ-CiM: Hardware-Aware Embedding Shaping for Robust
CiM-Based Retrieval
Xinzhao Li*1, Alptekin Vardar*2, Franz Müller2, Navya Goli3, Umamaheswara Rao Tida3, Kai Ni4,
X. Sharon Hu4, Thomas Kämpfe2, Ruiyang Qin1
1Villanova University,2Fraunhofer IPMS,3NDSU,4University of Notre Dame
ABSTRACT
Deploying Retrieval-Augmented Generation (RAG) on edge devices
is in high demand, but is hindered by the latency of massive data
movement and computation on traditional architectures. Compute-
in-Memory (CiM) architectures address this bottleneck by perform-
ing vector search directly within their crossbar structure. However,
CiM’s adoption for RAG is limited by a fundamental “representation
gap,” as high-precision, high-dimension embeddings are incompat-
ible with CiM’s low-precision, low-dimension array constraints.
This gap is compounded by the diversity of CiM implementations
(e.g., SRAM, ReRAM, FeFET), each with unique designs (e.g., 2-bit
cells, 512x512 arrays). Consequently, RAG data must be naively
reshaped to fit each target implementation. Current data shaping
methods handle dimension and precision disjointly, which degrades
data fidelity. This not only negates the advantages of CiM for RAG
but also confuses hardware designers, making it unclear if a failure
is due to the circuit design or the degraded input data. As a result,
CiM adoption remains limited. In this paper, we introduce CQ-CiM,
a unified, hardware-aware data shaping framework that jointly
learns Compression and Quantization to produce CiM -compatible
low-bit embeddings for diverse CiM designs. To the best of our
knowledge, this is the first work to shape data for comprehensive
CiM usage on RAG.
1 INTRODUCTION
Retrieval-Augmented Generation (RAG) gives Large Language Mod-
els (LLMs) access to up-to-date knowledge. RAG is vital for on-
device tasks, such as personalized AI or assistive robotics, where
new real-time context is often outside the LLM’s pretrained data.
The core engine of RAG is a massive vector similarity search, often
using Max Inner Product Search (MIPS) [ 1] to find the most rele-
vant information for a query. However, two main issues prevent
wide RAG adoption on edge devices. First, limited edge resources
cause large data movement between storage (e.g., SSD) and memory
(RAM), leading to retrieval latency that can even exceed the LLM’s
inference time. Second, as data volume grows, the MIPS compu-
tation time on traditional systems scales poorly, creating a major
performance bottleneck.
* Equally contributed.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
DAC ‘26, July 26–29, 2026, Longbeach, CA
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-x-xxxx-xxxx-x/YY/MM
https://doi.org/10.1145/nnnnnnn.nnnnnnn
Top-n 
𝑠𝑖𝑚(𝑞⋅𝑐9)
𝑠𝑖𝑚(𝑞⋅𝑐1)
𝑠𝑖𝑚(𝑞⋅𝑐6)
𝑠𝑖𝑚(𝑞⋅𝑐2)
𝑠𝑖𝑚(𝑞⋅𝑐5)
QueryMIPSWordline Driver Corpus 
embeddings
Queries
 embeddings
LLMWl1
Wl2
WlmDl2 Sl2 Dl1 Sl1 Dln Sln Dln-1 Sln-1
Analog -to-Digital ConvertorsWlm-1Figure 1: Illustration of CiM-based embedding retrieval on a
FeFET crossbar array within a RAG pipline
Compute-in-Memory (CiM) emerges as a promising solution by
directly addressing both RAG bottlenecks. The core MIPS operation
(i.e., dot-product) is naturally suited to the CiM crossbar structure.
As shown in Figure 1, the entire corpus of embedding vectors is
stored as a matrix of conductances within the FeFET-based crossbar
array. An input query vector is then applied to the array, performing
a massive parallel vector-matrix multiplication directly in mem-
ory. This architecture fundamentally eliminates the data movement
latency. It also solves the computational scaling problem, as the
search latency is largely independent of the number of stored vec-
tors. Therefore, CiM-based MIPS enables efficient on-device RAG
by drastically reducing both latency and core resource usage.
Unfortunately, a fundamental “representation gap” compromises
CiM’s potential to accelerate RAG retrieval. On the one hand, sen-
tence embedding models produce high-precision (e.g., FP32) and
high-dimension (e.g., 1x2048) vectors by default. On the other hand,
the physical CiM arrays can only support limited precision and
dimension. Specifically, ReRAM-based CiM arrays sustain only 3 to
4 programable levels in controlled conditions but can degrade to an
effective 2-level precision over time, while FeFET-based CiM arrays
realistically support around 2 stable levels. The specific precision
and dimensionality that can be supported vary across different
CiM implementations (i.e., SRAM, ReRAM, and FeFETs). This hard-
ware diversity means that embeddings must be "reshaped" on a
per-target basis. The critical challenge is therefore the lack of a
general, end-to-end method that can flexibly perform this shaping
for any given CiM array.
To bridge this gap, we introduce an unified hardware-aware,
parameter-efficient framework to joint compression and quantiza-
tion for CiM’s information retrieval and relevant downstream tasks
like RAG, referred to as CQ-CiM, that shatters this bottleneck by si-
multaneously shaping both precision and dimension to fit any given
crossbar-based CiM design. Our approach introduces an unified,
end-to-end trainable pipeline built on three synergistic components:
(1) a parameter-efficient adapter for lightweight fine-tuning of the
given embedding model; (2) a learnable compression head to reduce
dimensionality to fit CiM array size constraints; and (3) a non-linear
quantization head to map embeddings to low-bit CiM-compatiblearXiv:2602.20083v2  [cs.ET]  25 Feb 2026

DAC ‘26, July 26–29, 2026, Longbeach, CA Li and Vardar et al.
Table 1: Level-dependent device variation profiles used in the
conductance transition model. For multi-level cells, 𝐿0–𝐿3de-
note the nominal states and 𝜎𝑣their corresponding Gaussian
deviations.
Name # of LevelsDevice Variations𝜎 𝑣𝐿0𝐿1𝐿2𝐿3
𝑅𝑅𝐴𝑀 1(Device-1) 1 0.0100 0.0100 0.0100 0.0100
𝐹𝑒𝐹𝐸𝑇 2(Device-2) 4 0.0067 0.0135 0.0135 0.0067
𝐹𝑒𝐹𝐸𝑇 3(Device-3) 4 0.0049 0.0146 0.0146 0.0049
𝑅𝑅𝐴𝑀 4(Device-4) 4 0.0038 0.0151 0.0151 0.0038
𝐹𝑒𝐹𝐸𝑇 6(Device-5) 4 0.0026 0.0155 0.0155 0.0026
format. This entire framework is trained using a specifically de-
signed loss function that uniquely enables self-supervised learning
by leveraging a contrastive objective, removing the restriction of
data labeling. This contrastive loss is complemented by a recon-
struction loss (MSE) to guide and stabilize the compression and
quantization heads. Furthermore, our design uniquely allows phys-
ical hardware characteristics, such as device variance (noise), to be
incorporated into the training process (via noise injection), shaping
the embeddings to be robust to such noise.
Our major contributions are summarized as follows:
•To the best of our knowledge, this is the first work to in-
troduce a unified framework that jointly handles precision
and dimension shaping with explicit hardware-awareness,
enabling embedding representations to be directly aligned
with CiM crossbar array designs.
•We introduce a novel adaptation pipeline that incorporates
dimension compression, precision quantization, and parameter-
efficient tuning into a single process. This design enables a
single embedding model to be flexibly specialized for a wide
range of CiM devices with different hardware constraints.
•We demonstrate our framework’s superiority through com-
prehensive experiments, showing it significantly outper-
forms SOTA methods on diverse retrieval benchmarks, and
maintains high robustness in end-to-end RAG evaluations
under practical CiM device variations
2 BACKGROUND AND CHALLENGE
2.1 Analog CiM Arrays and Multi-Level FeFETs
CiM architectures have been realized using several device technolo-
gies. SRAM-based CiM is the most mature and widely adopted due
to its high reliability and negligible device variation [ 2]. Meanwhile,
non-volatile memory (NVM) technologies such as resistive RAM
(ReRAM) [ 3] and ferroelectric transistor (FeFET) devices [ 4] can
also be used to build crossbar-based CiM arrays that support analog
current-mode vector–matrix multiplication (VMM) and multi-level
weight storage. Our CQ-CiM framework is designed to support all
three CiM device classes.
Crossbar-based CiM arrays perform analog VMM by program-
ming device conductances at WL-BL intersections and accumulat-
ing bitline currents according to Ohm’s and Kirchhoff’s laws. This
structure naturally fits similarity search operations such as MIPS
and enable large parallelism in minimal data movement. Figure 2(a-
c) shows the 28nm GF HKMG mixed-signal FeFET CiM array [ 5,6]
used as the reference platform, where measured bitline currents ex-
hibit near-linear accumulation where activating multiple wordlines
[7], confirming the array’s suitability for analog inference.
(a) (b) (d)
(c)
Figure 2: Experimental characterization and functionality of
the FeFET-based compute-in-memory array in 28 nm CMOS.
(a) Layout of the mixed-signal FeFET crossbar array (b) fab-
ricated die photograph, (c) measured bitline current accu-
mulation under progressive wordline activation, and (d) pro-
grammed 2-bit multi-level FeFET operation of 50 devices
exhibiting four distinct𝑉 Tstates (𝐿 0–𝐿3).
To support larger embedding capacity under tight area and la-
tency constraints, per-cell storage density becomes essential. Multi-
level FeFETs provide an efficient solution by storing multiple quan-
tized values within a single device through stable intermediate
threshold-voltage states [ 4] . As shown in Figure 2(d), the fabri-
cated 2-bit FeFET cells demonstrate four well-separated 𝐿0to𝐿3
states[ 8–11]. Measurements from 50 fabricated devices validate
these states, showing well-separated 𝐼D–𝑉Gcharacteristics with
minimal overlap, as illustrated in Fig. 2(d).
Integrating multi-level cells into CiM arrays provides a natu-
ral match to low-bit (2-bit) neural representations but introduces
level-dependent conductance variation [ 12]. Table 1 summarizes
the variation profiles ( 𝐿0–𝐿3) used in this work, each modeled by a
nominal conductance and a Gaussian deviation 𝜎𝑣. These parame-
ters are derived from three representative measured devices—two
RRAM technologies [ 13,14] and one FeFET device [ 15]—denoted
as𝑅𝑅𝐴𝑀 1,𝑅𝑅𝐴𝑀 4, and𝐹𝑒𝐹𝐸𝑇 2. We further extrapolate two syn-
thesized variants ( 𝐹𝑒𝐹𝐸𝑇 3,𝐹𝑒𝐹𝐸𝑇 6) to span a broader multi-level
range. An𝑥-level device supports 𝑥distinct conductance states (e.g.,
𝜎𝐿2=0.01indicates a variation of 0.01 at level 2), forming the basis
of the transition model used in our hardware-aware training.
2.2 Representation Gap between RAG and CiM
In RAG, a pretrained sentence embedding model [ 16] converts both
queries and corpus documents into high-dimensional floating-point
embeddings, and retrieval is performed through Max Inner Product
Search (MIPS). Compute-in-Memory (CiM) architectures provide
an appealing substrate for this operation: embedding vectors can
be stored as crossbar conductances, and queries can be applied
as analog voltages to perform massively parallel vector–matrix
multiplication (VMM), yielding low-latency and low-movement
similarity search [17].
However, deploying RAG on CiM reveals a fundamentalrep-
resentation gap. Modern sentence embeddings are high-precision
(FP32) and high-dimensional (e.g., 768–2048) [ 18], whereas CiM
arrays—whether based on SRAM, ReRAM, or FeFET—support only
a few multi-level conductance states (typically 1–2 bits per cell) and
fixed array dimensions (e.g., 64 ×64 or 128×128) [ 4,19]. Bridging this
mismatch requires reshaping embeddings along two orthogonal
axes: compressing their dimension to fit array size and quantiz-
ing their values to match low-bit device states. Naive PCA-based

CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval DAC ‘26, July 26–29, 2026, Longbeach, CA
Positive 
SamplesNegative 
Samples
LossTraining Deployment
CQ-CiMData 
Corpus
…
ADCDACEmbed 
& StoreUser 
Query
CrossbarEmbedding
MIPS
User Query    + 
LLMRetrieved 
Data
Output
Adapter
Compress
(Noise Injection)
Quantization
Sentence
Embedding 
Model
Positive 
embeddingsNegative 
Embeddings
Figure 3: Overview of CQ-CiM. Left: The framework jointly
shapes embedding dimension and precision via a LoRA-based
adapter with noise injection for hardware robustness. Right:
Demonstrate that the trained embedding model based on our
framework (CQ-CiM) is used to bridge the CiM and RAG.
compression [ 20] or uniform quantization [ 21] degrades semantic
structure and ignores device non-idealities such as multi-level vari-
ance and read noise [ 22], making them insufficient for real CiM
deployment.
Existing CiM accelerators such as ISAAC, PRIME, and Newton
demonstrate efficient analog VMM for similarity search [ 23–25], but
all assume that embeddings are already stored in a CiM-compatible
form and do not address how high-dimensional FP embeddings
should be reshaped before storage. Retrieval quantization methods
[26] and recent low-bit embedding schemes [ 27] improve semantic
efficiency but remain hardware-agnostic, overlooking array dimen-
sion constraints [ 28], multi-level programming behavior [ 29], and
device variation [ 30]. Even CiM-focused optimizations primarily
target circuit-level VMM efficiency or non-ideality compensation
[31], without addressing how the embedding representation it-
self should be reshaped for CiM constraints. Consequently, no
prior work jointly addresses dimensionality compression, low-bit
quantization, and device-aware robustness in a unified, end-to-end
framework tailored for CiM—leaving a critical gap between RAG’s
embedding requirements and the constraints of practical CiM hard-
ware.
3 PROPOSED WORK
3.1 Overall Architecture
This section introduces the full CQ-CiM framework, including its
overall architecture, adaptation module, compression module, noise
injection, and low-bit quantization design. The overall architecture
of CQ-CiM is illustrated in Figure 3. CQ-CiM enables an end-to-end
training pipeline to convert high-precision and high-dimension
sentence embeddings into a low-precision and low-dimension for-
mat compatible with various designs and implementations of CiM
crossbar arrays.
The data flow passes through three main stages in the training
pipeline. First, the input sentence is encoded by the backbone sen-
tence embedding model equipped with a LoRA adapter [ 32], used as
our parameter-efficient adaptation mechanism. LoRA injects a light-
weight low-rank update into selected weight matrices, allowing the
encoder to adjust its representations without full fine-tuning. Thisenables the embedding model to adapt to our downstream com-
pression and quantization objectives while keeping the trainable
parameter budget small. Then, the output is fed into the compres-
sion head to reduce its dimensionality based on the crossbar array
size (e.g., 128D). Finally, the compressed embeddings pass through
our noise injection module to be injected with simulated device
noise. The processed embeddings are then quantized by our de-
signed quantization head to map them to low-bit values.
To enable the training pipeline in a self-supervised manner, posi-
tive and negative samples are constructed by modifying the dropout
rate of the embedding model [ 17]. As shown in Figure 3, after these
two types of samples pass through the pipeline, the loss is calculated
and used for backpropagation. The Deployment side of the figure
illustrates the deployment process: the CQ-CiM block (containing
the LoRA, compression, and quantization heads) shapes the corpus
and query embeddings before they are sent to the CiM crossbar for
efficient RAG retrieval.
3.2 Adaptive Embedding Compression
The sentence embedding model is the core of the framework. The
way to get the embedding model involved in the training is critical.
Freezing the model can be computationally efficient, but it poten-
tially limits the framework’s learning space, as the backbone cannot
adapt its representations to various compression and quantization
tasks. Conversely, fine-tuning all parameters of the embedding
model risks creating a parameter imbalance [ 33], where the massive
model’s gradients overwhelm the learning of lightweight compres-
sion and quantization heads. Therefore, we employ a LoRA-based
adaptation mechanism. Backed by the experiments in the following
section, this choice provides an optimal balance, allowing the core
embedding model to adapt while keeping training costs low and
the parameter space balanced.
A similar design rationale applies to the compression head. We
evaluated several alternatives, including PCA [ 34] and autoencoder
[35]. PCA is a strong baseline, but it is a non-learnable method
and therefore cannot be optimized end-to-end or adapt to the task-
specific loss functions. An Autoencoder is learnable, but it is in-
compatible with our single-stage, unified training objective, as it
typically requires a separate pre-training phase before its decoder
is discarded. In this work, we employ a simple yet effective dense
projection layer as our compression head. This design is fully learn-
able, computationally lightweight, and perfectly compatible with
our end-to-end pipeline.
3.3 Noise Injection Training
The purpose of our noise injection training is to make the final
embeddings resilient to hardware-level device variance, when non-
volatile memories are applied in CiM. The placement of this module
is a critical design choice, as shown in Figure 3. We inject the noise
directly after the compression head, but before the quantization
head. This location is chosen to balance the noise impact: if noise
were injected before compression, its effect would be diminished by
the projection layer; conversely, if noise were injected after quanti-
zation, it would become a discrete "bit-flip" problem. This would
cause significant information distortion and, crucially, prevent the
quantizer’s learnable thresholds from adapting to the noise. The

DAC ‘26, July 26–29, 2026, Longbeach, CA Li and Vardar et al.
Figure 4: Visualization of quantized values of a single embed-
ding (dimension=25) using fixed nonuniform threshold (left)
and learned N2UQ threshold (right)
device variation parameters in Table 1 are derived from measured
FeFET and ReRAM arrays, ensuring that the simulated noise closely
reflects real CiM hardware behavior.
To implement this, we take the compressed embedding, 𝑒𝑚𝑏𝐶∈
R𝑑, where the dimension 𝑑matches the compression setting. We
construct a noise mask, 𝝐, which is a new tensor with the same
dimension𝑑as the compressed embedding.
To build this mask, we use the user-provided hardware charac-
teristics from Table 1. For 𝐾conductance levels, the user defines
𝐾−1static thresholds {𝜏1,...,𝜏𝐾−1}(e.g.,{0.25,0.5,0.75}) and𝐾
variance values{𝐿0,...,𝐿𝐾−1}. Each element 𝜖𝑗in the noise mask
is associated with a level index 𝑘=find_level(𝑒𝑚𝑏 𝐶,𝑗;{𝜏𝑘}), and is
drawn from a Gaussian distribution with level-dependent deviation:
𝜖𝑗∼N(0,𝐿2
𝑘)where𝜎 𝑘=𝐿𝑘(refer to Table 1) (1)
Finally, this level-aware noise mask 𝝐is scaled by a global noise
factor𝜎𝑔(simulating the overall disturbance level) and added to
the compressed embedding to produce the final, noise-injected
embedding𝑒𝑚𝑏 𝜎:
𝑒𝑚𝑏𝜎=𝑒𝑚𝑏𝐶+𝜎𝑔·𝝐(2)
This noise-injected embedding 𝑒𝑚𝑏𝜎is then passed to the subse-
quent quantization head for training.
3.4 Quantization Head and Joint Loss
Low-bit quantization is essential for deploying embeddings on CiM
crossbar arrays, which are often composed of memory cells that
support only a few discrete conductance levels (i.e., 2-bit). A key
challenge is that the distribution of these compressed embeddings is
typically highly non-uniform. Instead of employing the nonuniform
quantization method [ 36], we formalize compressed embeddings
into a matrix and employ the Nonuniform-to-Uniform Quantization
(N2UQ) method [ 37,38], which is originally designed to quantize
model weights, to lower the precision of the matrix.
To ensure the N2UQ head learns effectively in conjunction with
the compression and noise injection modules, we employ a joint
loss function. Our framework is trained end-to-end using a dual-
objective loss that combines contrastive learning with reconstruction-
based supervision. This design reflects the practical constraint of
corpus-only training: contrastive learning refines the semantic ge-
ometry, while reconstruction stabilizes the compression and quan-
tization modules. As shown in Figure 4, fixed nonuniform quanti-
zation (left) fails to capture the true data distribution, while N2UQ
(right) learns to distribute the embedding values more effectively
across the available levels.
Following the behavior of the quantization head, we next detail
the training objectives that make the quantization and compression
heads learnable within a unified pipeline.Contrastive Loss:For self-supervised learning, we leverage a
contrastive objective. Two dropout-masked views [ 39] of the same
sentence are processed by our framework to form a positive pair
(ℎ𝑖,ℎ+
𝑖), while all other batch samples serve as negatives:
LCSE=−log𝑒sim(ℎ 𝑖,ℎ+
𝑖)/𝜏
Í𝑁
𝑗=1
𝑒sim(ℎ 𝑖,ℎ+
𝑗)/𝜏+𝑒sim(ℎ 𝑖,ℎ−
𝑗)/𝜏.(3)
We use this objective primarily to update the LoRA adapter to
enhance the quality of the learned representations.
Reconstruction Loss:To guide the compression and quantiza-
tion heads, we use a standard mean squared error (MSE) reconstruc-
tion objective, which minimizes the distance between the original
floating-point embedding and the final compressed-quantized out-
put:
LMSE=𝑓orig(𝑥)− ˆ𝑓cq(𝑥)2
2.(4)
Although the output embedding has lower dimensionality and dis-
crete precision, this reconstruction objective stabilizes training and
helps preserve the quality of the compressed and quantized embed-
dings.
Overall Objective:The final loss is a weighted sum of the two
objectives:
L=L CSE+𝜆MSELMSE,(5)
where𝜆MSEbalances the semantic and reconstruction goals. In
practice, for 384→128 compression, a moderately large weight
(𝜆MSE≈5–10) is effective.
4 EXPERIMENTAL EVALUATION
In this section, we present experiments to evaluate the performance,
design, generalizability, and hardware-aware robustness of our
proposed CQ-CiM framework. Our primary task is information
retrieval, while RAG serves as the downstream validation of robust-
ness. We first present our main results, benchmarking our complete,
optimized framework against existing baselines across five diverse
information retrieval datasets to demonstrate its advanced perfor-
mance in the low-bit range. Then, we perform a detailed ablation
study to justify our framework’s advantage, validating its design.
Later on, we validate the generalizability of our framework on
four different sentence embedding models. Finally, we provide the
end-to-end evaluation on RAG, the downstreaming task, where the
device variances from non-volatile memories are injected to assess
the robustness of CQ-CiM on different CiM implementations. By
default, experiments are run on an NVIDIA A10 GPU.
4.1 Evaluating CQ-CiM Performance
To validate our framework’s performance, we benchmark it across 5
informative retrieval datasets: ARCChallenge [ 40], NanoHotpotQA
[41], CQADupStackGisRetrieval [ 42], and ArguAna [ 40]. We use
all-MiniLM-L6-v2[ 43] as the base sentence embedding model with
the embedding dimension set to 128. Our framework is compared
with three baselines: UMAP [ 44], PCA + uniform quantization [ 34],
and native scale (Vanilla) [45].
Under the dimension of 128, we evaluate three low-bit settings: 1-
bit [46], 1.58-bit [ 47], and 2-bit [ 48]. Such low-bit precision levels are
commonly adopted in crossbar-based CiM designs [4, 19]. In addi-
tion, we include the uncompressed PQ-8bit [ 26] as a high-precision

CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval DAC ‘26, July 26–29, 2026, Longbeach, CA
Table 2: Comparison of different compression approaches at 128D (128-dimensional embeddings) across five retrieval datasets.
Corpus sizes are shown in parentheses (e.g., 9.35k = 9.35k documents). PQ-8bit is included as a high-precision vector-quantization
reference (not CiM-compatible) and does not necessarily represent the highest retrieval performance.
Precision MethodARCChallenge (9.35k) NanoHotpotQA (5.09k) CQADupStack (37.6k) FiQA (57.6k) ArguAna (8.67k)
Recall@5 nDCG@10 Recall@5 nDCG@10 Recall@5 nDCG@10 Recall@5 nDCG@10 Recall@5 nDCG@10
1 bitUMAP 0.024 0.021 0.160 0.151 0.035 0.033 0.026 0.026 0.167 0.138
PCA 0.069 0.059 0.490 0.450 0.145 0.246 0.192 0.194 0.410 0.329
Vanilla 0.071 0.055 0.310 0.357 0.255 0.227 0.182 0.179 0.378 0.307
CQ-CiM 0.082 0.062 0.380 0.393 0.302 0.256 0.212 0.210 0.449 0.360
1.58 bitUMAP 0.010 0.009 0.080 0.070 0.003 0.003 0.013 0.012 0.078 0.070
PCA 0.098 0.076 0.470 0.501 0.201 0.311 0.293 0.292 0.511 0.418
Vanilla 0.094 0.073 0.530 0.463 0.345 0.297 0.240 0.248 0.474 0.386
CQ-CiM 0.0950.077 0.540 0.543 0.382 0.322 0.288 0.303 0.5070.419
2 bitUMAP 0.033 0.027 0.140 0.157 0.068 0.056 0.036 0.042 0.230 0.190
PCA 0.083 0.068 0.480 0.482 0.192 0.309 0.244 0.249 0.410 0.338
Vanilla 0.092 0.071 0.470 0.470 0.346 0.302 0.244 0.248 0.468 0.385
CQ-CiM 0.110 0.082 0.540 0.547 0.399 0.346 0.299 0.297 0.529 0.433
PQ (8-bit) Reference 0.117 0.074 0.760 0.475 0.334 0.241 0.435 0.222 0.456 0.244
reference. Note that PQ is a codebook-based vector quantization
method rather than a task-adapted or CiM-oriented representation,
and therefore its performance does not necessarily exceed that of
optimized low-bit embeddings. All performance is measured using
Recall@5 and nDCG@10 [49].
As shown in Table 2, the results validate our approach. The
UMAP baseline’s performance collapses at ultra-low-bit settings, as
its manifold structures are not robust to aggressive discretization.
PCA performs competitively at 1.58-bit but becomes unstable at 1-
bit. The Vanilla truncation method consistently underperforms, con-
firming that this simple approach discards important semantic infor-
mation. In contrast, our CQ-CiM framework consistently achieves
the best or most competitive performance across all datasets, show-
ing its largest advantages at the most aggressive 1-bit and 2-bit
settings. UMAP performs poorly because its non-linear manifold
is highly sensitive to aggressive low-bit quantization, making it
inherently unsuitable for CiM-style discretized representations.
As a concluding remark, these results highlight the value of
jointly learning adaptation, compression, and quantization. Within
this unified framework, our method produces robust, high-quality
low-bit embeddings that outperform individual embedding-shaping
approaches under realistic CiM hardware constraints.
4.2 Validating Core Components
We perform an ablation study to examine how each learnable com-
ponent contributes to the overall effectiveness of CQ-CiM. Specifi-
cally, we examine three learnable components that shape embed-
dings: the adaptation mechanism, the compression head, and the
quantization head. All experiments are conducted on the ArguAna
dataset using theall-MiniLM-L6-v2encoder, where the original
384D FP32 embeddings are shaped to 128D 2-bit for evaluation.
Across the three components, we compare the case where LoRA
is involved with two settings ( 𝑟𝑎𝑛𝑘= 4,𝛼= 8and𝑟𝑎𝑛𝑘= 8,𝛼= 16)
and the case where LoRA is not used to enable the embedding model
adaptation along with the training process. Then, we compare our
Dense projection layer with autoencoder (AE). Finally, we compare
the fixed 2-bit Straight-Through-Estimator (STE) and the learned
N2UQ method.
As shown in Figure 5, the results validate our design choices.
First, enabling adaptation with LoRA consistently improves perfor-
mance over the No-LoRA baseline, with the 𝑟=8,𝛼= 16setting
Figure 5: Ablation study on ArguAna. Retrieval performance
across LoRA settings, compression methods, and quantiza-
tion strategies. LoRA (8,16) + Dense + N2UQ achieves the
strongest retrieval accuracy.
achieving the strongest gains. Second, the Dense projection proves
more robust than the autoencoder variants, which degrade retrieval
quality due to their mismatch with the single-stage training pipeline.
Finally, N2UQ provides the largest individual improvement, with its
learned thresholds outperforming the fixed STE by 3–7% absolute
across all settings.
Taken together, these observations indicate that the combination
of LoRA adaptation, Dense compression, and N2UQ quantization
forms the most effective embedding-shaping pipeline, which is the
setting used for the evaluations in Section 4.1.
4.3 Generalizability Across Embedding Models
To evaluate the generality of CQ-CiM, we include four embed-
ding models on NanoFeverRetrieval [ 50] benchmark:Nomic-embed-
text-v1.5[ 51],KaLM-embedding-mini-instruct-v2.5[ 52,53],Qwen3-
Embedding-0.6B[ 45], andGranite-embedding-278m[ 54]. This se-
lection covers models varying in scale and training objectives. For
each model, embeddings are reduced to multiple target dimensions
and quantized under four precision settings—1-bit, 1.58-bit, 2-bit,
and Int4. We use nDCG@5 as the evaluation metric for retrieval.
Across all models, Figure 7 shows a consistent trend: retrieval
quality decreases smoothly as dimension is reduced, while 1.58-bit
and 2-bit quantization preserve strong performance down to 128
dimensions, whereas 1-bit quantization degrades sharply at very
low dimension. Although INT4 quantization achieves the strongest
accuracy, low-bit quantization is far better suited to CiM. It avoids

DAC ‘26, July 26–29, 2026, Longbeach, CA Li and Vardar et al.
(a) NanoHotpot on Qwen -3-4B
 (b) NanoHotpot on Phi3 -mini
 (b) NanoHotpot on Mistral -7B
 (b) NanoHotpot on Llama -3-1B
(a) NanoFiQA on Qwen -3-4B (b) NanoFiQA on Phi3 -mini (b) NanoFiQA on Mistral -7B (b) NanoFiQA on Llama -3-1B
Figure 6: End-to-end RAG accuracy under device non-idealities. D-1 to D-5 correspond to the device variation settings in Table 1.
Quantized embeddings are perturbed using transition matrices derived from these device variations.
/uni00000016/uni00000015 /uni00000014/uni00000015/uni0000001b /uni00000015/uni00000018/uni00000019 /uni00000016/uni0000001b/uni00000017 /uni00000018/uni00000014/uni00000015 /uni00000019/uni00000017/uni00000013 /uni0000001a/uni00000019/uni0000001b
/uni00000028/uni00000050/uni00000045/uni00000048/uni00000047/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000027/uni0000004c/uni00000050/uni00000048/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000051/uni00000027/uni00000026/uni0000002a/uni00000023/uni00000018
/uni00000031/uni00000052/uni00000050/uni0000004c/uni00000046/uni00000010/uni00000048/uni00000050/uni00000045/uni00000048/uni00000047/uni00000010/uni00000057/uni00000048/uni0000005b/uni00000057/uni00000010/uni00000059/uni00000014/uni00000011/uni00000018
/uni00000014/uni00000045/uni0000004c/uni00000057
/uni00000014/uni00000011/uni00000018/uni0000001b/uni00000045/uni0000004c/uni00000057
/uni00000015/uni00000045/uni0000004c/uni00000057
/uni0000002c/uni00000051/uni00000057/uni00000017
/uni00000016/uni00000015 /uni00000014/uni00000015/uni0000001b /uni00000015/uni00000018/uni00000019 /uni00000016/uni0000001b/uni00000017 /uni00000018/uni00000014/uni00000015 /uni00000019/uni00000017/uni00000013 /uni0000001a/uni00000019/uni0000001b
/uni00000028/uni00000050/uni00000045/uni00000048/uni00000047/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000027/uni0000004c/uni00000050/uni00000048/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000051/uni00000027/uni00000026/uni0000002a/uni00000023/uni00000018
/uni0000002e/uni00000044/uni0000002f/uni00000030/uni00000010/uni00000048/uni00000050/uni00000045/uni00000048/uni00000047/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000010/uni00000050/uni0000004c/uni00000051/uni0000004c/uni00000010/uni0000004c/uni00000051/uni00000056/uni00000057/uni00000055/uni00000058/uni00000046/uni00000057/uni00000010/uni00000059/uni00000015/uni00000011/uni00000018
/uni00000014/uni00000045/uni0000004c/uni00000057
/uni00000014/uni00000011/uni00000018/uni0000001b/uni00000045/uni0000004c/uni00000057
/uni00000015/uni00000045/uni0000004c/uni00000057
/uni0000002c/uni00000051/uni00000057/uni00000017
/uni00000016/uni00000015 /uni00000014/uni00000015/uni0000001b /uni00000015/uni00000018/uni00000019 /uni00000016/uni0000001b/uni00000017 /uni00000018/uni00000014/uni00000015 /uni00000019/uni00000017/uni00000013 /uni0000001a/uni00000019/uni0000001b
/uni00000028/uni00000050/uni00000045/uni00000048/uni00000047/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000027/uni0000004c/uni00000050/uni00000048/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000051/uni00000027/uni00000026/uni0000002a/uni00000023/uni00000018
/uni00000034/uni0000005a/uni00000048/uni00000051/uni00000016/uni00000010/uni00000028/uni00000050/uni00000045/uni00000048/uni00000047/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000010/uni00000013/uni00000011/uni00000019/uni00000025
/uni00000014/uni00000045/uni0000004c/uni00000057
/uni00000014/uni00000011/uni00000018/uni0000001b/uni00000045/uni0000004c/uni00000057
/uni00000015/uni00000045/uni0000004c/uni00000057
/uni0000002c/uni00000051/uni00000057/uni00000017
/uni00000016/uni00000015 /uni00000014/uni00000015/uni0000001b /uni00000015/uni00000018/uni00000019 /uni00000016/uni0000001b/uni00000017 /uni00000018/uni00000014/uni00000015 /uni00000019/uni00000017/uni00000013 /uni0000001a/uni00000019/uni0000001b
/uni00000028/uni00000050/uni00000045/uni00000048/uni00000047/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000003/uni00000027/uni0000004c/uni00000050/uni00000048/uni00000051/uni00000056/uni0000004c/uni00000052/uni00000051/uni00000013/uni00000011/uni00000015/uni00000018/uni00000013/uni00000011/uni00000018/uni00000013/uni00000013/uni00000011/uni0000001a/uni00000018/uni00000051/uni00000027/uni00000026/uni0000002a/uni00000023/uni00000018
/uni0000002a/uni00000055/uni00000044/uni00000051/uni0000004c/uni00000057/uni00000048/uni00000010/uni00000048/uni00000050/uni00000045/uni00000048/uni00000047/uni00000047/uni0000004c/uni00000051/uni0000004a/uni00000010/uni00000015/uni0000001a/uni0000001b/uni00000050
/uni00000014/uni00000045/uni0000004c/uni00000057
/uni00000014/uni00000011/uni00000018/uni0000001b/uni00000045/uni0000004c/uni00000057
/uni00000015/uni00000045/uni0000004c/uni00000057
/uni0000002c/uni00000051/uni00000057/uni00000017
Figure 7: Retrieval accuracy curves for four embedding mod-
els across different embedding dimensions and four quanti-
zation precisions (1-bit, 1.58-bit, 2-bit, and INT4).
multi-step arithmetic and eliminates the shifting and stitching in
higher-precision representation. In INT4, each value must be split
across multiple CiM cells (e.g., four 1-bit SRAM cells), and the
partial results must be shifted and stitched back together during
accumulation. This process increases data movement, adds extra
compute cycles, and amplifies noise from device mismatch.
In contrast, CQ-CiM enables dot-product computation in a single
crossbar step on modern multi-level NVM arrays operating at 2-bit
quantization [ 4]. This reduces computational complexity and allows
a substantially larger corpus under the same memory budget.
We further assess the practical efficiency of CQ-CiM by measur-
ing the end-to-end training time required to adapt each embedding
model. The results in Table 3 are obtained using a fixed target
embedding dimension of 128D, matching the 2-bit CiM crossbar
setting used throughout this work. As shown in the table, CQ-CiM
completes CiM-specific adaptation within only a few minutes even
for models with hundreds of millions of parameters, demonstrating
that the framework incurs minimal computational overhead and
scales efficiently across diverse embedding architectures.
4.4 End-to-End RAG
To close the loop between embedding quantization and system-
level retrieval, we integrate device variation directly into an end-to-
end RAG pipeline. Following the Gaussian conductance–variation
model widely used in multi-level NVM characterization [ 22], and
using the device parameters in Table 1, we derive a level-dependent
4×4transition matrix that captures the probability of each 2-
bit state flipping to another during readout. During evaluation,
every quantized embedding is perturbed according to this transition
matrix before retrieval.Table 3: Parameter size and end-to-end CQ-CiM training time
for different embedding models at a fixed 128D target dimen-
sion under a 2-bit CiM setting.
Model Para. Size Time
Granite-embedding-278m 278 M 5m28s
Qwen3-Embedding-0.6B 600 M 19m40s
KaLM-embedding-mini-instruct-v2.5 500 M 14m18s
Nomic-embed-text-v1.5 137 M 4m53s
We then evaluate the full RAG stack: the embedding model (after
noise injection) retrieves the top-5 documents for each query, and
an LLM selects the single most relevant document among these
candidates. The prediction is compared to the ground-truth docu-
ment, producing an end-to-end accuracy metric that reflects both
embedding robustness and LLM selection reliability.
In these experiments, noise injection follows Eq. 2 with𝜖=0.1.
LLMs in experiments operate in FP16 with temperature =0, and
all embedding models are evaluated using batch size 16. For LoRA-
enabled encoders, we set rank to 8,𝛼to 16, and dropout to 0.05.
As shown in Figure 6, the proposed compressed embeddings
remain stable across datasets and LLM architectures even under
realistic device-induced level-flip noise. Although device variation
inevitably reduces retrieval performance, the degradation is modest,
and the relative ordering among methods remains consistent. These
results confirm that our low-bit representations are resilient not
only in embedding-space metrics but also in a full RAG pipeline,
where hardware noise affects both retrieval candidates and down-
stream LLM reasoning.
5 CONCLUSION
In this paper, we introduceCQ-CiM, a unified hardware-aware
data-shaping framework that addresses the fundamental "represen-
tation gap" of deploying RAG on CiM architectures. Our approach
flexibly transforms high-precision, high-dimension embeddings to
meet the low-precision, low-dimension constraints of diverse CiM
arrays. By jointly optimizing a LoRA adapter, a Compression head,
and a Quantization head, our self-supervised framework learns
embedding representations that are robust to device-level variation.
Experiments show that CQ-CiM outperforms traditional methods
across retrieval benchmarks while maintaining strong robustness
under realistic device noise. This is the first unified data shaping
framework enabling comprehensive and reliable CiM-based RAG.

CQ-CiM: Hardware-Aware Embedding Shaping for Robust CiM-Based Retrieval DAC ‘26, July 26–29, 2026, Longbeach, CA
REFERENCES
[1]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al.
Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in
neural information processing systems, 33:9459–9474, 2020.
[2]Kentaro Yoshioka, Shimpei Ando, Satomi Miyagi, Yung-Chin Chen, and Wenlun
Zhang. A review of sram-based compute-in-memory circuits.Japanese Journal
of Applied Physics, 63(12):120802, dec 2024.
[3]Shimeng Yu, Wonbo Shim, Xiaochen Peng, and Yandong Luo. Rram for compute-
in-memory: From inference to training.IEEE Transactions on Circuits and Systems
I: Regular Papers, 68(7):2753–2765, 2021.
[4]Soliman et al. First demonstration of in-memory computing crossbar using
multi-level cell fefet.Nature Communications, 14(1):6348, 2023.
[5]Vardar et al. 28nm ferroelectric field effect transistor based associative memory
array for few-shot learning and genome analysis. In2025 International VLSI
Symposium on Technology, Systems and Applications (VLSI TSA), pages 01–02,
2025.
[6]Yin et al. Ferroelectric compute-in-memory annealer for combinatorial optimiza-
tion problems.Nature Communications, 15(1):2419, 2024.
[7]De et al. 28 nm hkmg-based current limited fefet crossbar-array for inference
application.IEEE Transactions on Electron Devices, 69(12):7194–7198, 2022.
[8]Müller et al. Multi-level operation of ferroelectric fet memory arrays for compute-
in-memory applications. In2023 IEEE International Memory Workshop (IMW),
pages 1–4, 2023.
[9]Müller et al. Current percolation path impacting switching behavior of ferro-
electric fets. In2021 International Symposium on VLSI Technology, Systems and
Applications (VLSI-TSA), pages 1–2, 2021.
[10] Ni et al. On the channel percolation in ferroelectric fet towards proper analog
states engineering. In2021 IEEE International Electron Devices Meeting (IEDM),
pages 15.3.1–15.3.4, 2021.
[11] Müller et al. Multilevel operation of ferroelectric fet memory arrays considering
current percolation paths impacting switching behavior.IEEE Electron Device
Letters, 44(5):757–760, 2023.
[12] Alptekin Vardar, Marcel Günther, Franz Müller, Nellie Laleni, Konrad Seidel,
and Thomas Kämpfe. Reliable multi-level cell programming in fefet arrays for
in-memory computing.Japanese Journal of Applied Physics, 64(5):05SP09, may
2025.
[13] Yao et al. Fully hardware-implemented memristor convolutional neural network.
Nature, 577(7792):641–646, 2020.
[14] Liu et al. Architecture-circuit-technology co-optimization for resistive random
access memory-based computation-in-memory chips.Science China Information
Sciences, 66(10):200408, 2023.
[15] Wei et al. Switching pathway-dependent strain-effects on the ferroelectric prop-
erties and structural deformations in orthorhombic hfo2.Journal of Applied
Physics, 131(15), 2022.
[16] Ruiyang Qin, Dancheng Liu, Chenhui Xu, Zheyu Yan, Zhaoxuan Tan, Zhenge
Jia, Amir Nassereldine, Jiajie Li, Meng Jiang, Ahmed Abbasi, et al. Empirical
guidelines for deploying llms onto resource-constrained edge devices.ACM
Transactions on Design Automation of Electronic Systems, 30(5):1–58, 2025.
[17] Ruiyang Qin, Zheyu Yan, Dewen Zeng, Zhenge Jia, Dancheng Liu, Jianbo Liu,
Zhi Zheng, Ningyuan Cao, Kai Ni, Jinjun Xiong, et al. Robust implementa-
tion of retrieval-augmented generation on edge-based computing-in-memory
architectures.arXiv preprint arXiv:2405.04700, 2024.
[18] Reimers et al. Sentence-bert: Sentence embeddings using siamese bert-networks.
arXiv preprint arXiv:1908.10084, 2019.
[19] Yunseok Lee, Jongmin Park, Daewon Chung, Kisong Lee, and Sungjun Kim.
Multi-level cells and quantized conductance characteristics of al2o3-based rram
device for neuromorphic system.Nanoscale Research Letters, 17(1):84, 2022.
[20] Ian T Jolliffe and Jorge Cadima. Principal component analysis: a review and recent
developments.Philosophical transactions of the royal society A: Mathematical,
Physical and Engineering Sciences, 374(2065):20150202, 2016.
[21] Benoit et al. Quantization and training of neural networks for efficient integer-
arithmetic-only inference. InProceedings of the IEEE conference on computer
vision and pattern recognition, pages 2704–2713, 2018.
[22] Shim et al. Two-step write–verify scheme and impact of the read noise in
multilevel rram-based inference engine.Semiconductor Science and Technology,
35(11):115026, 2020.
[23] Shafiee et al. Isaac: A convolutional neural network accelerator with in-situ
analog arithmetic in crossbars.ACM SIGARCH Computer Architecture News,
44(3):14–26, 2016.
[24] Chi et al. Prime: A novel processing-in-memory architecture for neural network
computation in reram-based main memory.ACM SIGARCH Computer Architecture
News, 44(3):27–39, 2016.
[25] He et al. Newton: A dram-maker’s accelerator-in-memory (aim) architecture for
machine learning. In2020 53rd Annual IEEE/ACM International Symposium on
Microarchitecture (MICRO), pages 372–385. IEEE, 2020.[26] Herve Jegou, Matthijs Douze, and Cordelia Schmid. Product quantization for
nearest neighbor search.IEEE transactions on pattern analysis and machine
intelligence, 33(1):117–128, 2010.
[27] Taehee Jeong. 4bit-quantization in vector-embedding for rag. In2024 International
Conference on Machine Learning and Applications, pages 1037–1042. IEEE, 2024.
[28] Ruiyang Qin, Pengyu Ren, Zheyu Yan, Liu Liu, Dancheng Liu, Amir Nassereldine,
Jinjun Xiong, Kai Ni, Sharon Hu, and Yiyu Shi. Nvcim-pt: An nvcim-assisted
prompt tuning framework for edge llms. In2025 Design, Automation & Test in
Europe Conference (DATE), pages 1–7. IEEE, 2025.
[29] Ni et al. A novel ferroelectric superlattice based multi-level cell non-volatile
memory. In2019 IEEE International Electron Devices Meeting (IEDM), pages 28–8.
IEEE, 2019.
[30] Ni et al. Impact of extrinsic variation sources on the device-to-device variation in
ferroelectric fet. In2020 IEEE International Reliability Physics Symposium (IRPS),
pages 1–5. IEEE, 2020.
[31] Shubham Jain and Anand Raghunathan. Cxdnn: Hardware-software compen-
sation methods for deep neural networks on resistive crossbar systems.ACM
Transactions on Embedded Computing Systems (TECS), 18(6):1–23, 2019.
[32] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean
Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language
models, 2021.
[33] Houlsby et al. Parameter-efficient transfer learning for nlp. InInternational
conference on machine learning, pages 2790–2799. PMLR, 2019.
[34] Andrzej Maćkiewicz and Waldemar Ratajczak. Principal components analysis
(pca).Computers & Geosciences, 19(3):303–342, 1993.
[35] Dor Bank, Noam Koenigstein, and Raja Giryes. Autoencoders, 2021.
[36] Choukroun et al. Low-bit quantization of neural networks for efficient inference.
In2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW),
pages 3009–3018. IEEE, 2019.
[37] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, Michael W. Mahoney, and
Kurt Keutzer. A survey of quantization methods for efficient neural network
inference, 2021.
[38] Liu et al. Nonuniform-to-uniform quantization: Towards accurate quantization
via generalized straight-through estimation, 2022.
[39] Tianyu Gao, Xingcheng Yao, and Danqi Chen. Simcse: Simple contrastive learning
of sentence embeddings.arXiv preprint arXiv:2104.08821, 2021.
[40] Henning Wachsmuth, Shahbaz Syed, and Benno Stein. Retrieval of the best
counterargument without prior topic knowledge. InACL, 2018.
[41] Yang et al. Hotpotqa: A dataset for diverse, explainable multi-hop question
answering. InProceedings of the 2018 Conference on Empirical Methods in Natural
Language Processing (EMNLP), pages 2369–2380, 2018.
[42] Doris Hoogeveen, Karin M. Verspoor, and Timothy Baldwin. Cqadupstack: A
benchmark data set for community question-answering research. InProceedings
of the 20th Australasian Document Computing Symposium (ADCS), ADCS ’15,
pages 3:1–3:8, New York, NY, USA, 2015. ACM.
[43] Nils Reimers and Iryna Gurevych. Making monolingual sentence embeddings
multilingual using knowledge distillation. InProceedings of the 2020 Conference on
Empirical Methods in Natural Language Processing. Association for Computational
Linguistics, 11 2020.
[44] Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold
approximation and projection for dimension reduction, 2020.
[45] Zhang et al. Qwen3 embedding: Advancing text embedding and reranking
through foundation models.arXiv preprint arXiv:2506.05176, 2025.
[46] Wang et al. Bitnet: Scaling 1-bit transformers for large language models.arXiv
preprint arXiv:2310.11453, 2023.
[47] Ma et al. The era of 1-bit llms: All large language models are in 1.58 bits, 2024.
[48] Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. Quip: 2-bit
quantization of large language models with guarantees, 2024.
[49] Wang et al. A theoretical analysis of ndcg type ranking measures. InConference
on learning theory, pages 25–54. PMLR, 2013.
[50] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
FEVER: a large-scale dataset for fact extraction and VERification. In Marilyn
Walker, Heng Ji, and Amanda Stent, editors,NAACL, pages 809–819, New Orleans,
Louisiana, June 2018. Association for Computational Linguistics.
[51] Zach Nussbaum, John X. Morris, Brandon Duderstadt, and Andriy Mulyar. Nomic
embed: Training a reproducible long context text embedder, 2024.
[52] Zhao et al. Kalm-embedding-v2: Superior training techniques and data inspire a
versatile embedding model, 2025.
[53] Hu et al. Kalm-embedding: Superior training data brings a stronger embedding
model, 2025.
[54] Awasthy et al. Granite embedding models, 2025.