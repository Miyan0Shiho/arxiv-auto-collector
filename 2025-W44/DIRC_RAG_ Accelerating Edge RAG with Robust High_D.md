# DIRC-RAG: Accelerating Edge RAG with Robust High-Density and High-Loading-Bandwidth Digital In-ReRAM Computation

**Authors**: Kunming Shao, Zhipeng Liao, Jiangnan Yu, Liang Zhao, Qiwei Li, Xijie Huang, Jingyu He, Fengshi Tian, Yi Zou, Xiaomeng Wang, Tim Kwang-Ting Cheng, Chi-Ying Tsui

**Published**: 2025-10-29 08:38:02

**PDF URL**: [http://arxiv.org/pdf/2510.25278v1](http://arxiv.org/pdf/2510.25278v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by
integrating external knowledge retrieval but faces challenges on edge devices
due to high storage, energy, and latency demands. Computing-in-Memory (CIM)
offers a promising solution by storing document embeddings in CIM macros and
enabling in-situ parallel retrievals but is constrained by either low memory
density or limited computational accuracy. To address these challenges, we
present DIRCRAG, a novel edge RAG acceleration architecture leveraging Digital
In-ReRAM Computation (DIRC). DIRC integrates a high-density multi-level ReRAM
subarray with an SRAM cell, utilizing SRAM and differential sensing for robust
ReRAM readout and digital multiply-accumulate (MAC) operations. By storing all
document embeddings within the CIM macro, DIRC achieves ultra-low-power,
single-cycle data loading, substantially reducing both energy consumption and
latency compared to offchip DRAM. A query-stationary (QS) dataflow is supported
for RAG tasks, minimizing on-chip data movement and reducing SRAM buffer
requirements. We introduce error optimization for the DIRC ReRAM-SRAM cell by
extracting the bit-wise spatial error distribution of the ReRAM subarray and
applying targeted bit-wise data remapping. An error detection circuit is also
implemented to enhance readout resilience against deviceand circuit-level
variations. Simulation results demonstrate that DIRC-RAG under TSMC40nm process
achieves an on-chip non-volatile memory density of 5.18Mb/mm2 and a throughput
of 131 TOPS. It delivers a 4MB retrieval latency of 5.6{\mu}s/query and an
energy consumption of 0.956{\mu}J/query, while maintaining the retrieval
precision.

## Full Text


<!-- PDF content starts -->

DIRC-RAG: Accelerating Edge RAG with Robust
High-Density and High-Loading-Bandwidth
Digital In-ReRAM Computation
Kunming Shao1,*,†, Zhipeng Liao2,*, Jiangnan Yu1,*, Liang Zhao3, Qiwei Li4, Xijie Huang1, Jingyu He1,
Fengshi Tian1, Yi Zou3, Xiaomeng Wang1,†, Tim Kwang-Ting Cheng1, Chi-Ying Tsui1
1The Hong Kong University of Science and Technology2University of Southampton
3South China University of Technology4Wuhan University
*Equally Contributed Authors. †Correspondence: kshaoaa@connect.ust.hk and xwangee@connect.ust.hk.
Abstract—Retrieval-Augmented Generation (RAG) enhances
large language models (LLMs) by integrating external knowledge
retrieval but faces challenges on edge devices due to high storage,
energy, and latency demands. Computing-in-Memory (CIM)
offers a promising solution by storing document embeddings
in CIM macros and enabling in-situ parallel retrievals but is
constrained by either low memory density or limited computa-
tional accuracy. To address these challenges, we present DIRC-
RAG, a novel edge RAG acceleration architecture leveraging
Digital In-ReRAM Computation (DIRC). DIRC integrates a
high-density multi-level ReRAM subarray with an SRAM cell,
utilizing SRAM and differential sensing for robust ReRAM
readout and digital multiply-accumulate (MAC) operations. By
storing all document embeddings within the CIM macro, DIRC
achieves ultra-low-power, single-cycle data loading, substantially
reducing both energy consumption and latency compared to off-
chip DRAM. A query-stationary (QS) dataflow is supported for
RAG tasks, minimizing on-chip data movement and reducing
SRAM buffer requirements. We introduce error optimization
for the DIRC ReRAM-SRAM cell by extracting the bit-wise
spatial error distribution of the ReRAM subarray and applying
targeted bit-wise data remapping. An error detection circuit is
also implemented to enhance readout resilience against device-
and circuit-level variations.
Simulation results demonstrate that DIRC-RAG under TSMC
40nm process achieves an on-chip non-volatile memory density of
5.18Mb/mm2and a throughput of 131 TOPS. It delivers a 4MB
retrieval latency of 5.6µs/query and an energy consumption of
0.956µJ/query, while maintaining the retrieval precision.
I. INTRODUCTION
Large Language Models (LLMs), relying on extensive pa-
rameters and diverse training data, have demonstrated excep-
tional capabilities in language understanding and generation.
However, for certain user-defined private data, such as medical
records or personal information, privacy concerns prevent
these data from being transmitted to the cloud for processing.
At the same time, performing costly retraining or fine-tuning
locally is not feasible. A secure and efficient solution is
to integrate personalized knowledge documents into LLM
inference using edge Retrieval-Augmented Generation (RAG)
This research was supported by ACCESS - AI Chip Center for Emerging
Smart Systems, sponsored by InnoHK funding, Hong Kong SAR, partially
supported by SCUT Research Fund No. K3200890, as well as partially by
Guangzhou GJYC Fund No. 2024D01J0010.
Fig. 1:Demonstration of the workflow of RAG, where the document
embedding storage and loading for retrieval are the bottlenecks.
technique. Edge RAG ensures data privacy while significantly
improving the model’s performance across various domains
without the need for expensive fine-tuning [1]–[4].
As illustrated in Figure 1, the private database is first
converted into document embeddings using an embedding
model and stored locally. When a user submits a real-time
query, it is transformed into a query embedding, which is then
compared with the pre-stored document embeddings using co-
sine similarity or Maximum Inner Product Search (MIPS). The
most relevant document chunks are retrieved and, along with
the original query, input into the LLM for augmented inference
and generation. However, the storage and loading of large-
scale document embeddings and retrieval computation with
query embeddings pose significant challenges to hardware
systems in terms of storage capacity, energy consumption, and
latency [5]–[8].
Computing-in-Memory (CIM) architectures [9]–[15] have
been proposed as a promising solution to address the memory
wall issue by caching weights on CIM macros and adopting a
weight-stationary dataflow. CIM integrates computation logics
within memory arrays, enabling in-situ parallel computation
and significantly reducing data movement overhead. This ar-
chitecture is particularly well-suited for workloads dominated
by multiply-accumulate (MAC) operation, such as those found
in neural networks and other AI applications. By leveraging
CIM, large-scale document embeddings can be stored in CIM
macro, query embedding is taken as input, and the retrieval
MAC operation is processed directly within memory array,
minimizing the need for frequent data movement between 979-8-3315-2710-5/25/$31.00 © 2025 IEEEarXiv:2510.25278v1  [cs.AR]  29 Oct 2025

off-chip memory and on-chip processing units. This not only
reduces latency but also eliminates the energy overhead asso-
ciated with data loading, making CIM an attractive option for
accelerating RAG workloads on edge devices.
However, current CIM architectures remain constrained
by their limited storage density and update bandwidth for
memory-intensive RAG retrieval operations. To address the
above challenges, we propose DIRC-RAG, a novel edge RAG
acceleration architecture based on Digital In-ReRAM Compu-
tation (DIRC). DIRC combines the high-density storage capa-
bilities of ReRAM with the robust and efficient computation
capabilities of SRAM, providing a high-density, low-latency
solution optimized for RAG workloads [16]–[18]. Compared
to existing CIM architectures, DIRC-RAG offers several key
advantages tailored to RAG tasks:
•ReRAM and SRAM Coupled Memory:By coupling
high-density ReRAM subarrays with robust SRAM cells
using a differential sensing circuit, DIRC achieves signif-
icantly higher memory density and reliability, addressing
the storage and loading demands of large-scale document
embeddings.
•Query Stationary Dataflow:Leveraging the ultra-low-
power and one-cycle-latency data-loading capability of
DIRC macro, DIRC-RAG could support the query-
stationary dataflow for retrieval, minimizing on-chip data
movement and minimizing SRAM buffer requirements.
•Digital Multiply-Accumulation (MAC):DIRC inte-
grates 128x128 ReRAM-SRAM coupled DIRC cells with
bitwise multipliers, 128-input carry-save-adders and ac-
cumulators, enabling high-throughput and efficient digital
MAC operations for various INT precisions.
Additionally, DIRC introduces error-aware optimization
techniques to address the variability of both ReRAM devices
and CMOS circuits. Specifically:
•Error-Aware Bitwise Mapping:Through detailed post-
layout Monte-Carlo simulation, a ReRAM subarray bit-
wise spatial read-out error map was generated based on
ReRAM deviation and MOS mismatch. This map guides
the bitwise mapping of the most significant bits (MSBs)
in the most robust ReRAM positions, while the less stable
positions are for the least significant bits (LSBs).
•Error Detection after Sensing:An error detection cir-
cuit is embedded to check the read-out error of the
ReRAM-SRAM coupled cell, and if an error is detected,
the data is re-sensed to ensure accuracy.
The remainder of this paper is structured as follows: Section
II presents the Preliminaries on RAG, CIM and non-volatile
CIM; Section III demonstrates the overall architecture of
DIRC-RAG; Section IV showcases the experimental results
of DIRC-RAG. Finally, Section V concludes the paper.
II. PRELIMINARIES
A. Retrieval-Augmented Generation
RAG is a technique that integrates external knowledge into
pre-trained LLMs in a non-parametric manner, significantly
Fig. 2:Comparison between mainstream CIM memories.
enhancing their performance in domain-specific tasks. As
illustrated in Figure 1, RAG employs embedding models to
convert textual data into high-dimensional vector represen-
tations (embeddings), which are then passed to a retriever
for vector-based retrieval. Common retrieval methods include
Cosine Similarity and Maximum Inner Product Search (MIPS).
The choice of method depends on the output form of the
embedding model: if the embeddings are normalized, Cosine
Similarity is preferred; if the embeddings are unnormalized,
MIPS can be used and computed by vector inner products.
Q=q1, q2, q3, . . . , q n
, D=d1, d2, d3, . . . , d n
Inner Product(Q, D) =nX
i=1qidi
Cosine Similarity(Q, D) =Pn
i=1qidipPn
i=1q2
i·pPn
i=1d2
i
Both Cosine Similarity and MIPS rely on vector dot prod-
uct computations, often referred to as MAC operations. To
accelerate this computation pattern, CIM architectures can be
employed, offering significant performance improvements for
such tasks.
B. Mainstream Computing-in-Memory Technologies
Figure 2 introduces four mainstream memory technologies
for Computing-In-Memory (CIM):
•ROM-CIM:Utilizes fixed ROM with parameters set
during fabrication, unsuitable for dynamic updates [9].
•ReRAM-CIM:Based on non-volatile resistive RAM,
offering high density and rewritability. However, analog
computation in mainstream designs suffers from devia-
tions (e.g., resistance drift), reducing accuracy [10], [11].
•SRAM-CIM:Leverages static RAM with high compu-
tational accuracy but limited storage density due to its
complex 6T or larger cell structure [12], [13].
•eDRAM-CIM:Typically based on 3T1C embedded
DRAM, combining high accuracy and density. However,
periodic refresh increases power consumption and la-
tency, limiting efficiency in edge devices [14], [15].
C. ReRAM-SRAM Coupled CIM
To address the limitations of mainstream CIM architec-
tures in storage density and computational accuracy, a novel
ReRAM-SRAM Coupled CIM paradigm has been proposed,
balancing high storage density and computational precision
by coupling ReRAM with SRAM-CIM. [16] firstly inte-
grated ReRAM into SRAM-CIM by embedding multiple SLC
ReRAMs into SRAM cells. Weight data stored in ReRAM

Fig. 3:(a) DIRC-RAG architecture for document retrieval with query-stationary dataflow. (b) DIRC macro with 128x128 DIRC cells, digital
MAC and error detection circuit. (c) DIRC cell with 8x8 MLC ReRAM subarray, differential sensing circuit and 1bit SRAM cell.
is loaded into SRAM via current-based sensing, significantly
improving storage density. [17] introduced a two-SRAM-cell,
triple-level ReRAM-SRAM Coupled ternary CIM architecture,
improving both storage density and data loading accuracy.
[18] supported four-level MLC ReRAM with a single SRAM
cell, boosting storage density and efficiency, while proposing
a robust differential sensing scheme for tackling data loading
errors.
III. DIRC-RAG FRAMEWORK
A. Overall Architecture
Figure 3 illustrates the architectural details of the DIRC-
RAG, including the DIRC-RAG architecture, the DIRC macro
circuit, and the transistor-level design of the DIRC cell.
As shown in Figure 3(a), the DIRC Architecture consists
of sixteen DIRC-RAG Cores, where each core independently
stores and processes different document embeddings. The
architecture also includes a norm unit for real-time compu-
tation of the query embedding vector norm to support cosine
similarity calculations, an SRAM buffer to store the similarity
scores and indices of the local top-k results produced by
each core, and a Global Top-k Comparator that compares
the local results from sixteen cores and outputs the indices
corresponding to the final top-k results. Each core internally
comprises a DIRC macro that performs document embedding
storage and MAC operations, a ReRAM buffer for storing the
norms and indices of document embeddings, a calculator for
computing cosine similarity that can be bypassed when MIPS,
and a local top-k comparator that executes top-k selection
within the core.
The sixteen DIRC-RAG Cores operate in parallel and inde-
pendently. When a query embedding is input, it is broadcast
to all sixteen cores. Each core then computes the cosine
similarity or inner product between the query embedding and
all document embeddings stored within the core. The results
are compared locally within each core to select the local
top-k results, which are subsequently stored in the SRAM
buffer. Since the local top-k selection eliminates the majority
of candidates, the SRAM buffer’s storage requirements are
minimal. Finally, the Global Top-k Comparator selects the
most similar document embeddings from the SRAM buffer
and outputs their corresponding indices.As shown in Figure 3(b), a DIRC macro consists of 128
DIRC columns and peripheral input registers. Each DIRC
column is composed of the following components: 128 verti-
cally aligned DIRC cells with NVM storage; 128 NOR gates
functioning as bit multipliers; a high-speed and efficient 128-
input sign-less carry-save adder [19]–[21]; and an accumulator
circuit that performs cycle-by-cycle accumulation for adder
output. Additionally, to address potential ReRAM readout
errors, each DIRC column is equipped with an optional Error
Detection Circuit designed to detect errors in the SRAM
cached data. During the error detection cycle, the peripheral
input registers input 128 logical ‘1‘s, causing the adder to
output the sum of all D data in the column, denoted asΣD.
This result is then compared to the pre-stored value in the D
Sum Look-up Table. If the comparison results differ, an error
is detected, and the DIRC cell will re-sense the data to ensure
computational accuracy. The high density and large capacity
of the DIRC macro enable a storage capacity of 2Mb and
a storage density of up to 5.88Mb/mm2. The error detection
circuit further ensures the accuracy of computations, making
the design both reliable and efficient. Moreover, the SRAM
can be written from outside data, which means that if the
ReRAM is not large enough for stoarge, the computational
part of DIRC macro can be used as a general SRAM-CIM
macro.
Refer to [18], we build a DIRC cell as shown in Figure
3(c). The DIRC cell consists of the following components:
an 8x8 MLC ReRAM subarray in the top-left corner for
data storage; three reference ReRAMs in the top-right corner
to store intermediate reference values that distinguish four
ReRAM resistance levels; a differential sensing circuit in the
middle, which performs readout through Latch and Precharge;
and a 1-bit SRAM cell at the bottom to store the readout result.
During the differential sensing process, the Latch signal is
first disabled, disconnecting the feedback loop of the SRAM.
The Precharge circuit charges the Q and QB nodes of the
SRAM toV DD/2. Subsequently, the Latch circuit is enabled,
restoring the feedback loop of the SRAM, which discharges
the read bitline (ReadBL) and reference bitline (RefBL). The
discharging rate depends on the relative loads of the two
bitlines.

Fig. 4:Bit-level query-stationary dataflow in a DIRC column, which
stores sixteen document embeddings and enables error detection after
ReRAM sensing.
For the MLC ReRAM to be read from the 8x8 subarray,
its MSB is first sensed to determine whether its resistance
value resides in the lower two levels or the higher two levels.
Specifically, the GlobalSL is set to 0, and the corresponding
WL and BL in the subarray are selected, and the WL MSB in
the Reference is also selected, creating two relative grounded
loads on both ReadBL and RefBL. When the voltage at the
Q and QB nodes isV DD/2, the side with lower load on
bitline discharges faster to 0, while the other side with higher
load is charged toV DDby the feedback loop. If the ReRAM
resistance is smaller thanR M, the Q node of the SRAM
discharges to 0; otherwise, the Q node charges toV DD. For
LSB sensing, the LSBEn signal is enabled, and based on the
MSB sensing result of the previous cycle, the M and MB
signals select eitherR LorR Has the reference. The same
differential sensing process is repeated and the LSB is read.
Differential sensing effectively reduces the impact of ReRAM
variations on computational accuracy and makes it feasible to
embed logic circuits in high-density ReRAM arrays [22].
B. Query-Stationary Dataflow
Taking SRAM-CIM as an example, mainstream CIM tech-
nologies typically adopt weight stationary (WS) dataflow to
enhance the reuse of weight data in DNN/Transformer al-
gorithms. However, for retrieval tasks, where a single query
embedding is matched against thousands or millions of docu-
ment embeddings for MAC computations, two major dataflow
challenges arise:
•Storage limitation with WS:Storing document embed-
dings in the SRAM of CIM macros using WS dataflow
is constrained by limited SRAM density. This prevents
the CIM macro from accommodating all document em-
beddings, requiring tens to hundreds of cycles for row-
by-row SRAM updates after a few MAC computation
cycles, which significantly increases latency and energy
consumption.
Fig. 5:(a) The LSB spatial error map of the 8x8 ReRAM subarray
from post-layout Monte Carlo simulation. (b) The error detection
circuit for checking the sensing error of the DIRC cell.
•Low utilization with IS:Using an input stationary
(IS) dataflow, as proposed in [23], [24], results in low
CIM macro utilization due to the small number of query
embeddings (usually one). Each retrieval also incurs high
memory access overhead from on-chip/off-chip buffers to
the CIM macro, further increasing computational costs.
To address these challenges, DIRC adopts a fitting query-
stationary dataflow, effectively overcoming the dataflow issues
faced by CIM in retrieval tasks. First, DIRC utilizes Non-
volatile MLC ReRAM, which offers significantly higher mem-
ory density than SRAM-CIM, enabling long-term storage of
a large number of document embeddings that require infre-
quent updates. Second, it employs In-ReRAM Computation,
achieving ultra-low power consumption and enabling data
to be loaded from ReRAM into all SRAMs in the array
within a single cycle. Finally, DIRC stores the limited query
embeddings in sufficient input registers, eliminating the need
for on-chip buffer access until the retrieval process is complete.
If the document embedding dimension exceeds 128, it
can be folded and mapped within the same DIRC column.
For instance, a single DIRC column can store sixteen INT8
embeddings with a dimension of 128 or two INT8 embeddings
with a dimension of 1024. Additionally, DIRC can store
twice as many INT4 quantized embeddings as INT8 quantized
embeddings in a single column.
As shown in Figure 4, each DIRC column can hold sixteen
INT8 quantized document embeddings with a dimension of
128. The SRAM reads 1-bit of the INT8 document embedding
at a time and performs cycle-by-cycle MAC computations
with the 8-bit serial input query embedding. Partial sums
generated in each cycle are accumulated by an accumulator.
Since all DIRC macros operate in parallel, the vector dot
product between the query embedding and all 4MB document
embeddings can be computed in approximately 1300 cycles, or
about5.2µs at a 250MHz frequency. By adopting the query-
stationary dataflow, DIRC-RAG achieves efficient and low-
latency retrieval.
C. Error-Aware Optimization Techniques
Even though the differential sensing scheme adopted by
the DIRC cell effectively reduces the impact of ReRAM

Fig. 6:The effectiveness of the error-aware optimization techniques
for increasing the retrieval precision.
variation on computational accuracy [18], potential errors in
high-density ReRAM still arise due to outlier ReRAM devia-
tions and MOS process mismatches. These errors manifest as
unpredictable bit flip errors during readout operations.
To address these bit flip errors, we conducted a 1000-
point post-layout Monte Carlo simulation for the DIRC cell.
The simulation introduced ReRAM deviations (σ= 0.1) and
MOS mismatches under 0.8V and 250MHz, considering the
actual spatial distribution of ReRAM, voltage supply, and
routing constraints. The resulting 8x8 subarray LSB spatial
error map is shown in Figure 5(a). The MSB of MLC ReRAM
demonstrated 100% reliability due to its large signal margin,
so we focused on the LSB error distribution. In Figure 5(a),
two VSS power rails were placed on the left and right sides of
the subarray for NMOS transistors, while the sensing circuit
and SRAM were on the right side. ReRAM cells closer to
the VSS rail exhibited smaller read errors, while those farther
from the readout circuit exhibited larger errors, resulting in
the observed spatial error pattern.
Given the significant differences in error distribution, we
adopted a bit-wise remapping strategy. INT8 data is divided
into two groups: bits 0-3 and bits 4-7 (where bit 7 is the
sign bit). Since MSBs showed 100% reliability, bits 4-7 were
mapped to MSBs, while errors in bits 0-3, located on LSBs,
were considered. Using the LSB spatial error map, bit 3 of the
sixteen INT8 data was mapped to locations with the smallest
error rates, while bit 0 was mapped to locations with the
largest error rates. This strategy minimizes the impact of errors
on computational accuracy. Figure 6 demonstrates the 24.6%
precision improvement achieved by the bitwise remapping.
Since bit flip errors are unpredictable and cannot be com-
pensated for or retrained as proposed in [6], we designed
an error detection circuit, shown in Figure 5(b), to identify
sensing errors in the DIRC cell. Assuming correct data is
written into ReRAM, the circuit detects bit flip errors caused
by sensing issues. To support this, DIRC-RAG computes the
bitwise sum of document embeddings offline and stores the
results in a D Sum Look-Up Table (LUT) based on the
ReRAM buffer. During runtime, LUT values are compared
with the adder output. If differences occur, an error is detected,
and the DIRC cell re-senses the data to ensure accuracy. This
method could effectively mitigate errors caused by transient
interference during sensing, as shown in Figure 6.TABLE I: DIRC-RAG SPEC
Process TSMC40nm
DIRC-RAG Area 6.18mm2
Frequency 250MHz
Voltage 0.8V
Precisions INT4/8
Embedding Dimension 128∼1024
Macro Size 16Kb
Macro Area 0.34mm2
Macro Efficiency 1176TOPS/W, 24.9TOPS/mm2
Macro NVM Storage 2Mb
Total NVM Storage 4MB
Total Memory Density 5.178Mb/mm2
Retrieval Latency 5.6µs (4MB retrieval)
Energy/Query 0.956µJ (4MB retrieval)
(a) DIRC cell layout
(b) DIRC macro layout
Fig. 7: (a) The detailed layout of DIRC cell. (b) The layout
overview of DIRC macro.
IV. EXPERIMENTS ANDVALIDATION
A. Experiment Setup
For the hardware experiments, we implemented a DIRC
macro using the TSMC 40nm process for post-layout sim-
ulation. The digital MAC and error detection circuits were
synthesized with Synopsys Design Compiler, while Cadence
Innovus was used for automatic placement and routing. The
DIRC cell was designed and verified in Cadence Virtuoso,
where the top-level layout and post-layout simulations were
also conducted. The ReRAM model is adopted from [25] and
placed in the VIA1 layer. The SRAM buffer is generated using
the TSMC40nm SRAM compiler, while the remaining digital
circuits are synthesized with Synopsys Design Compiler and
simulated in Synopsys VCS.
For the software experiments, we modified the open-source
BEIRframework [26] to evaluate the retrieval precision of
INT4 and INT8 quantized embeddings [27] across datasets
including SciFact [28], NFCorpus [29], TREC-COVID [30],
ArguAna [31], and SciDocs [32]. The embedding model
usesall-MiniLM-L6-v2, integrated into the open-source
Sentence-BERT[33] framework with an embedding di-
mension of 512. Retrieval performance is measured using
Precision@k (P@k) for k = 1, 3, 5, reflecting the proportion
of relevant documents in the top-k results.

TABLE II: Retrieval Precision (P@1, 3, 5) Comparison across Different Datasets and Quantizations.
Embedding Size (MB) P@1 P@3 P@5
Dataset FP32 INT8 INT4 FP32 INT8 INT4 FP32 INT8 INT4 FP32 INT8 INT4
SciFact 7.59 1.90 0.95 0.5067 0.5033 0.4833 0.2400 0.2378 0.2244 0.1633 0.1640 0.1553
NFCorpus 5.32 1.33 0.66 0.4210 0.4149 0.3684 0.3540 0.3488 0.3034 0.3046 0.3028 0.2743
TREC-COVID(1)15.68 3.92 1.96 0.6400 0.6200 0.5400 0.5667 0.5600 0.5533 0.5640 0.5520 0.4960
ArguAna 12.71 3.18 1.59 0.2525 0.2560 0.2489 0.1669 0.1650 0.1562 0.1255 0.1255 0.1172
SciDocs(2)12.53 3.13 1.57 0.2410 0.2400 0.2160 0.1907 0.1917 0.1683 0.1570 0.1572 0.1408
(1)The TREC-COVID dataset is sampled by a factor of 16 for storing all INT8 quantized embeddings on DIRC-RAG.
(2)The SciDocs dataset is sampled by a factor of 3 for storing all INT8 quantized embeddings on DIRC-RAG.
For system efficiency simulation, we developed a Python-
based simulator to evaluate the energy efficiency and retrieval
latency of the DIRC-RAG architecture. The simulator mod-
els energy consumption, interconnect bandwidth, and cycle
latency for components such as the DIRC macro, norm unit,
SRAM buffer, and global top-k comparator.
B. Hardware Evaluation
As shown in Figure 7, in this work, we designed a DIRC-
RAG architecture capable of storing 4MB embeddings, sup-
porting INT4/8 precision, and flexible dimensions ranging
from 128 to 1024. We first conducted post-layout simulations
for the DIRC macro. The results demonstrate that the DIRC
macro, with an dimension of 128x128 and 2Mb of NVM
storage, achieves an energy efficiency of 1176 TOPS/W and
an area efficiency of 24.9 TOPS/mm2at 250MHz and 0.8V .
We then performed system-level simulations for the overall
DIRC-RAG with an area of 6.18 mm2, which could accom-
modate 4MB of data, achieving a storage density of 5.178
Mb/mm2. For a 4MB INT8-quantized document embedding
database with 512 dimensions, our simulations reveal that
querying the 4MB database requires only 5.6µs of latency and
0.956µJ of energy consumption, while utilizing less than 1KB
of SRAM buffer. Furthermore, we observed that the latency
and energy consumption of DIRC-RAG scale linearly with the
embedding database size.
However, it should be acknowledged that despite the high
storage density achieved by DIRC-RAG, it is still insufficient.
Two potential solutions could be adopted: First, when DIRC
storage is insufficient, DIRC can function as a conventional
SRAM-CIM for MAC operations by writing data from on-chip
buffers or off-chip DRAM into SRAM; Second, scaling-up by
leveraging chiplet technology to integrate multiple DIRC-RAG
chips into a larger-scale system.
C. Software Evaluation
DIRC-RAG leverages a hardware-software codesign strat-
egy by quantizing the query and document embeddings from
FP32 to INT8 and INT4. To evaluate the impact of low-
precision INT quantization on retrieval accuracy, we conducted
extensive experiments on wide-range datasets and compared
the results with the FP32 baseline.
As shown in Table II, the retrieval precision of INT8
quantization is nearly identical to FP32, while INT4 intro-
duces a slight but acceptable drop in P@k. For instance, on
the NFCorpus dataset, P@1 decreases from 0.4210 (FP32)TABLE III: Comparison with RTX3090
Hardware DIRC-RAG RTX3090
Process TSMC 40nm Samsung 8nm
Area 6.18mm2628.4mm2
Frequency 250MHz 1395MHz
Embeddings INT8 FP32
Dataset SciFact
Precision@3 0.2378 0.2400
Energy/Query 0.46µJ 86.8 mJ
Latency/Query 2.77µs 21.7 ms
to 0.3684 (INT4), and P@5 from 0.3046 to 0.2743. With
slight P@k drop, INT4 significantly boosts computational
throughput and reduces storage compared to FP32. These
results highlight that edge-based RAG systems can adopt
low-precision INT quantization for document embeddings to
simplify computation and minimize storage and bandwidth
overhead while maintaining high retrieval accuracy.
D. Comparison with Baseline
Although [6] introduces CIM for RAG, it does not include a
hardware evaluation. As there are few end-to-end edge RAG
systems like DIRC-RAG available for a fair comparison of
retrieval latency and efficiency, we conduct a performance
evaluation against the NVIDIA RTX3090 GPU. As shown in
Table III, for a single query in the SciFact dataset, DIRC-
RAG with INT8 quantized embeddings achieves a latency
of only 2.77µs and an energy consumption of 0.46µJ.
In contrast, the RTX3090 GPU, also using INT8 quantized
embeddings, requires 21.7 ms of latency and 86.8 mJ of energy
consumption. The GPU results are averaged over 30,000
queries to ensure accurate and reliable comparison.
V. CONCLUSION
In conclusion, DIRC-RAG effectively demonstrates a novel
and efficient architecture for accelerating RAG on edge de-
vices by leveraging digital in-ReRAM computation. Through
the hybrid ReRAM-SRAM memory cell design and query-
stationary dataflow, DIRC achieves high memory density,
robust computational accuracy, and significant reductions in
energy consumption and latency for retrieval. The proposed
error optimization techniques further enhance the reliability of
the system under device- and circuit-level variations. Simula-
tion results validate the effectiveness of DIRC-RAG, achieving
5.6µs/query latency, and 0.956µJ/query energy consumption,
making it a promising solution for energy-efficient and high-
performance RAG tasks on edge devices.

REFERENCES
[1] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir
Karpukhin, Naman Goyal, Heinrich K ¨uttler, Mike Lewis, Wen-tau Yih,
Tim Rockt ¨aschel, et al. Retrieval-augmented generation for knowledge-
intensive nlp tasks.Advances in neural information processing systems,
33:9459–9474, 2020.
[2] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng
Geng, Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin
Cui. Retrieval-augmented generation for ai-generated content: A survey.
arXiv preprint arXiv:2402.19473, 2024.
[3] Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. Corrective
retrieval augmented generation. 2024.
[4] Runtao Ren, Yinyu Wu, Xuhui Zhang, Jinke Ren, Yanyan Shen,
Shuqiang Wang, and Kim-Fung Tsang. Retrieval-augmented generation
for mobile edge computing via large language model.arXiv preprint
arXiv:2412.20820, 2024.
[5] Michael Shen, Muhammad Umar, Kiwan Maeng, G Edward Suh,
and Udit Gupta. Towards understanding systems trade-offs in
retrieval-augmented generation model inference.arXiv preprint
arXiv:2412.11854, 2024.
[6] Ruiyang Qin, Zheyu Yan, Dewen Zeng, Zhenge Jia, Dancheng Liu,
Jianbo Liu, Zhi Zheng, Ningyuan Cao, Kai Ni, Jinjun Xiong, et al.
Robust implementation of retrieval-augmented generation on edge-based
computing-in-memory architectures.arXiv preprint arXiv:2405.04700,
2024.
[7] Derrick Quinn, Mohammad Nouri, Neel Patel, John Salihu, Alireza
Salemi, Sukhan Lee, Hamed Zamani, and Mohammad Alian. Acceler-
ating retrieval-augmented generation.arXiv preprint arXiv:2412.15246,
2024.
[8] Wooyoung Jo, Seongyon Hong, Jiwon Choi, Beomseok Kwon, Haoyang
Sang, Dongseok Im, Sangyeob Kim, Sangjin Kim, Taekwon Lee, and
Hoi-Jun Yoo. 23.7 broca: A 52.4-to-559.2 mw mobile social agent
system-on-chip with adaptive bit-truncate unit and acoustic-cluster bit
grouping. In2025 IEEE International Solid-State Circuits Conference
(ISSCC), volume 68, pages 418–420. IEEE, 2025.
[9] Guodong Yin, Yiming Chen, Mufeng Zhou, Wenjun Tang, Mingyen
Lee, Zekun Yang, Tianyu Liao, Xirui Du, Vijaykrishnan Narayanan,
Huazhong Yang, et al. Cramming more weight data onto compute-in-
memory macros for high task-level energy efficiency using custom rom
with 3984-kb/mm 2 density in 65-nm cmos.IEEE Journal of Solid-State
Circuits, 59(6):1912–1925, 2023.
[10] Kodai Ueyoshi, Ioannis A Papistas, Pouya Houshmand, Giuseppe M
Sarda, Vikram Jain, Man Shi, Qilin Zheng, Sebastian Giraldo, Peter
Vrancx, Jonas Doevenspeck, et al. Diana: An end-to-end energy-
efficient digital and analog hybrid neural network soc. In2022 IEEE
International Solid-State Circuits Conference (ISSCC), volume 65, pages
1–3. IEEE, 2022.
[11] Win-San Khwa, Tai-Hao Wen, Hung-Hsi Hsu, Wei-Hsing Huang, Yu-
Chen Chang, Ting-Chien Chiu, Zhao-En Ke, Yu-Hsiang Chin, Hua-Jin
Wen, Wei-Ting Hsu, et al. A mixed-precision memristor and sram
compute-in-memory ai processor.Nature, pages 1–7, 2025.
[12] Yu-Der Chih, Po-Hao Lee, Hidehiro Fujiwara, Yi-Chun Shih, Chia-Fu
Lee, Rawan Naous, Yu-Lin Chen, Chieh-Pu Lo, Cheng-Han Lu, Haruki
Mori, et al. 16.4 an 89tops/w and 16.3 tops/mm 2 all-digital sram-based
full-precision compute-in memory macro in 22nm for machine-learning
edge applications. In2021 IEEE International Solid-State Circuits
Conference (ISSCC), volume 64, pages 252–254. IEEE, 2021.
[13] Hidehiro Fujiwara, Haruki Mori, Wei-Chang Zhao, Kinshuk Khare,
Cheng-En Lee, Xiaochen Peng, Vineet Joshi, Chao-Kai Chuang, Shu-
Huan Hsu, Takeshi Hashizume, et al. 34.4 a 3nm, 32.5 tops/w, 55.0
tops/mm 2 and 3.78 mb/mm 2 fully-digital compute-in-memory macro
supporting int12×int12 with a parallel-mac architecture and foundry
6t-sram bit cell. In2024 IEEE International Solid-State Circuits
Conference (ISSCC), volume 67, pages 572–574. IEEE, 2024.
[14] Sangjin Kim, Zhiyong Li, Soyeon Um, Wooyoung Jo, Sangwoo Ha,
Juhyoung Lee, Sangyeob Kim, Donghyeon Han, and Hoi-Jun Yoo. Dy-
naplasia: An edram in-memory computing-based reconfigurable spatial
accelerator with triple-mode cell.IEEE Journal of Solid-State Circuits,
59(1):102–115, 2023.
[15] Yi Zhan, Wei-Han Yu, Ka-Fai Un, Rui P Martins, and Pui-In Mak.
Gslp-cim: A 28-nm globally systolic and locally parallel cnn/transformer
accelerator with scalable and reconfigurable edram compute-in-memorymacro for flexible dataflow.IEEE Transactions on Circuits and Systems
I: Regular Papers, 2024.
[16] Liukai Xu, Songyuan Liu, Zhi Li, Dengfeng Wang, Yiming Chen, Yanan
Sun, Xueqing Li, Weifeng He, and Shi Xu. Cream: computing in reram-
assisted energy and area-efficient sram for neural network acceleration.
InProceedings of the 59th ACM/IEEE Design Automation Conference,
pages 115–120, 2022.
[17] Dengfeng Wang, Liukai Xu, Songyuan Liu, Zhi Li, Yiming Chen,
Weifeng He, Xueqing Li, and Yanan Sun. Tl-nvsram-cim: Ultra-
high-density three-level reram-assisted computing-in-nvsram with dc-
power free restore and ternary mac operations. In2023 IEEE/ACM
International Conference on Computer Aided Design (ICCAD), pages
1–9. IEEE, 2023.
[18] Xiaomeng Wang, Jingyu He, Kunming Shao, Jiakun Zheng, Fengshi
Tian, Tim Kwang-Ting Cheng, and Chi-Ying Tsui. Rescim: Variation-
resilient high weight-loading bandwidth in-memory computation based
on fine-grained hybrid integration of multi-level reram and sram cells.
InProceedings of the 41st IEEE/ACM International Conference on
Computer-Aided Design (ICCAD), pages 1–9, 2024.
[19] Heng You, Weijun Li, Delong Shang, Yumei Zhou, and Shushan Qiao.
A 1–8b reconfigurable digital sram compute-in-memory macro for
processing neural networks.IEEE Transactions on Circuits and Systems
I: Regular Papers, 71(4):1602–1614, 2024.
[20] Liang Zhao, Kunming Shao, Fengshi Tian, Tim Kwang-Ting Cheng,
Chi-Ying Tsui, and Yi Zou. A flexible precision scaling deep neural
network accelerator with efficient weight combination.arXiv preprint
arXiv:2502.00687, 2025.
[21] Kunming Shao, Fengshi Tian, Xiaomeng Wang, Jiakun Zheng, Jia
Chen, Jingyu He, Hui Wu, Jinbo Chen, Xihao Guan, Yi Deng,
et al. Syndcim: A performance-aware digital computing-in-memory
compiler with multi-spec-oriented subcircuit synthesis.arXiv preprint
arXiv:2411.16806, 2024.
[22] Miguel Angel Lastras-Monta ˜no and Kwang-Ting Cheng. Resistive
random-access memory based on ratioed memristors.Nature Electron-
ics, 1(8):466–472, 2018.
[23] Bokyung Kim, Shiyu Li, and Hai Li. Inca: Input-stationary dataflow at
outside-the-box thinking about deep learning accelerators. In2023 IEEE
International Symposium on High-Performance Computer Architecture
(HPCA), pages 29–41. IEEE, 2023.
[24] Sheng Ma, Yunping Zhao, Yuhua Tang, and Yi Dai. Hpa: A hybrid data
flow for pim architectures. In2024 IEEE 42nd International Conference
on Computer Design (ICCD), pages 470–478. IEEE, 2024.
[25] Peng Yao, Huaqiang Wu, Bin Gao, Sukru Burc Eryilmaz, Xueyao
Huang, Wenqiang Zhang, Qingtian Zhang, Ning Deng, Luping Shi, H-
S Philip Wong, et al. Face classification using electronic synapses.
Nature communications, 8(1):15199, 2017.
[26] Nandan Thakur, Nils Reimers, Andreas R ¨uckl´e, Abhishek Srivastava,
and Iryna Gurevych. Beir: A heterogenous benchmark for zero-shot eval-
uation of information retrieval models.arXiv preprint arXiv:2104.08663,
2021.
[27] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew
Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko.
Quantization and training of neural networks for efficient integer-
arithmetic-only inference. InProceedings of the IEEE conference on
computer vision and pattern recognition, pages 2704–2713, 2018.
[28] David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine
van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or fiction:
Verifying scientific claims. InEMNLP, 2020.
[29] Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. A
full-text learning to rank dataset for medical information retrieval. 2016.
[30] Kirk Roberts, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman,
Kyle Lo, Ian Soboroff, Ellen V oorhees, Lucy Lu Wang, and William R
Hersh. Searching for scientific evidence in a pandemic: An overview of
trec-covid.Journal of Biomedical Informatics, 121:103865, 2021.
[31] Henning Wachsmuth, Shahbaz Syed, and Benno Stein. Retrieval of the
best counterargument without prior topic knowledge. InProceedings
of the 56th Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers), pages 241–251, 2018.
[32] Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel S
Weld. Specter: Document-level representation learning using citation-
informed transformers.arXiv preprint arXiv:2004.07180, 2020.
[33] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings
using siamese bert-networks.arXiv preprint arXiv:1908.10084, 2019.