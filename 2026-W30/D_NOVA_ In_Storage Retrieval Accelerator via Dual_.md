# D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound 3D NAND-Optimized Similarity Search with Vector Adaptation

**Authors**: Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S. Rosing, Mingu Kang

**Published**: 2026-07-20 04:31:09

**PDF URL**: [https://arxiv.org/pdf/2607.17538v1](https://arxiv.org/pdf/2607.17538v1)

## Abstract
Retrieval-Augmented Generation (RAG) enhances the factual grounding of large language model (LLM) inference by retrieving relevant information from external knowledge bases. However, its dense vector retrieval introduces significant latency and energy overhead, becoming the primary performance bottleneck. Although recent in-storage accelerators aim to reduce data movement, they still rely on host or embedded processors outside the memory, where nearly 70% of the total retrieval time is spent. As a result, they cannot fully overcome the bandwidth limitations, leading to yet another memory bottleneck. To tackle these limitations, we present D-NOVA, a hardware-software co-designed in-storage retrieval accelerator. D-NOVA executes an inverted file (IVF)-based hierarchical retrieval pipeline by deeply embedding the search functionality directly into the NAND memory array. This is achieved by incorporating a new distance metric, Dual-Bound Tight Similarity Sensing (DTS), which is specifically tailored for searching within the NAND string. In addition, we introduce a lightweight contrastive adapter that maps embedding vectors into a DTS-friendly domain, recovering near-software recall while improving performance and energy efficiency. D-NOVA is up to 41.7x faster and 71x more energy-efficient than a CPU baseline, and achieves 12.13x higher throughput while being up to 1.26x more energy-efficient than state-of-the-art in-storage RAG accelerators, demonstrating the potential of fully in-storage vector search for scalable RAG acceleration.

## Full Text


<!-- PDF content starts -->

D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound
3D NAND-Optimized Similarity Search with Vector Adaptation
Chang Eun Song∗
University of California, San Diego
La Jolla, USA
cesong@ucsd.eduSumukh Pinge∗
University of California, San Diego
La Jolla, USA
spinge@ucsd.eduTianqi Zhang
University of California, San Diego
La Jolla, USA
tiz014@ucsd.edu
Sung Eun Kim
University of California, San Diego
La Jolla, USA
suk062@ucsd.eduTajana S Rosing
University of California, San Diego
La Jolla, USA
tajana@ucsd.eduMingu Kang
University of California, San Diego
La Jolla, USA
mingu@ucsd.edu
Abstract
Retrieval-Augmented Generation (RAG) enhances the factual ground-
ing of large language model (LLM) inference by retrieving relevant
information from external knowledge bases. However, its dense
vector retrieval introduces significant latency and energy overhead,
becoming the primary performance bottleneck. Although recent
in-storage accelerators aim to reduce data movement, they still rely
on host or embedded processors outside the memory, where nearly
70% of the total retrieval time is spent. As a result, they cannot fully
overcome the bandwidth limitations, leading to yet another mem-
ory bottleneck. To tackle these limitations, we present D-NOVA,
a hardware–software co-designed in-storage retrieval accelerator.
D-NOVA executes an inverted file (IVF)-based hierarchical retrieval
pipeline by deeply embedding the search functionality directly into
the NAND memory array. This is achieved by incorporating a new
distance metric, Dual-bound Tight Similarity Sensing (DTS), which
is specifically tailored for searching within the NAND string. In
addition, we introduce a lightweight contrastive adapter that maps
embedding vectors into a DTS–friendly domain, recovering near-
software recall while improving performance and energy efficiency.
D-NOVA is up to 41.7×faster and 71×more energy-efficient than a
CPU baseline, and achieves 12.13×higher throughput while being
up to 1.26×more energy-efficient than state-of-the-art in-storage
RAG accelerators, demonstrating the potential of fully in-storage
vector search for scalable RAG acceleration.
1 Introduction
Large language models (LLMs) have revolutionized language under-
standing and, through recent extensions, enabled reasoning across
multiple modalities. Their inference increasingly relies on GPUs
and specialized hardware for acceleration [ 22,40,62,72,73,85,87].
However, their knowledge remains largely static, constrained by
pre-training data and costly to update. To mitigate this limitation,
Retrieval-Augmented Generation (RAG) has emerged as an effective
solution. As illustrated in Fig. 1(a), RAG augments LLMs with exter-
nal knowledge bases, grounding outputs in up-to-date information
without retraining, but this benefit comes with a major retrieval
∗These authors contributed equally to this work.
MICRO 2026, Athens, Greece
2026. ACM ISBN 978-X-XXXX-XXXX-X/XX/XX
https://doi.org/XXXXXXX.XXXXXXX
3.081.190.2220.3669.4694.2676.5729.355.52HotpotQA(16GB)wiki_en/NQ(120GB)Synthetic Dataset0.0025.0050.0075.00EncodingRetrievalGenerationUserQuestion
LLMRetrieval QueryRetrieved TextsDatabase
Context Enhanced Question“What is the capital of France?”“Paris. ” “Capital city of France?” “Paris is the capitaland the largest city of France.”Response
(b)(a)3.081.190.2220.3669.4694.2676.5729.355.52HotpotQA(16GB)wiki_en/NQ(120GB)Synthetic Dataset0.0025.0050.0075.00EncodingRetrievalGeneration
(1TB)End to end latency breakdown (%)
wiki_enFigure 1: (a) RAG system overview, and (b) end-to-end latency
breakdown of IVF-Flat with [ 18], [86], and synthetic datasets.
bottleneck. Even with algorithmic optimizations and hardware ac-
celeration on CPUs and GPUs [ 27,31,32,55,63], each query must
still traverse billions of high-dimensional embeddings distributed
across the memory hierarchy from storage to DRAM and on to
the processor [ 10,47,67]. This massive data movement dominates
latency and energy, offsetting much of the benefit of computational
acceleration. As the vector database (DB) scales, empirical stud-
ies [10,30,47,67,68] show that retrieval alone can spend more
than 70% of the total end-to-end latency (Fig. 1(b)), highlighting
data movement as the main bottleneck in current RAG systems.
To address this data bandwidth limitation, approximate near-
est neighbor (ANN) algorithms such as Hierarchical Navigable
Small World (HNSW) graphs and Inverted File (IVF) indices are
widely adopted for improving retrieval efficiency [ 10,19,55,79].
However, they impose very different demands on hardware design.
HNSW achieves high recall accuracy, which comes at the cost of
irregular pointer-chasing behavior and unpredictable access pat-
terns [ 37,84], making it difficult to implement efficiently on highly
parallel hardware platforms [ 55]. In contrast, IVF provides a more
structured access pattern, making it better suited for parallel and
hardware-efficient execution. Nevertheless, both HNSW and IVF
remain fundamentally constrained by data movement overhead, as
large volumes of vector data must still be transferred across the
memory/storage hierarchy. This shared bottleneck has motivated
the development of in- and near-storage architectures.
1
arXiv:2607.17538v1  [cs.AR]  20 Jul 2026

MICRO 2026, October 31–November 04, 2026, Athens, Greece Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S Rosing, and Mingu Kang
Near-storage retrieval accelerators [ 47,79] bring processing
closer to SSDs, but still require substantial architectural modifi-
cations and depend on host CPUs or GPUs for synchronization and
final re-ranking. For instance, RAGX [ 47] and SmartSSD [ 41,79]
embed reconfigurable or FPGA-based accelerators near the storage
device. However, their limited on-device memory capacity and strict
power constraints hinder scalability beyond relatively lightweight
data pipelines. REIS [ 10] leverages in-storage processing (ISP) by
applying binary quantization for coarse ANN search. However, due
to its limited accuracy, it requires an expensive INT8 re-ranking
that retrieves high-precision data from flash to the SSD controller
and DRAM. This re-ranking step accounts for nearly 70% of the
total retrieval time, dominated by multi-level cell (MLC) read la-
tency and data movement to the embedded processor, resulting
in frequent data transfers between storage, controllers, and on-
chip memory, which create a new memory wall that fundamentally
limits bandwidth, energy efficiency, and parallelism in search.
A natural next step is to embed search operations deeper in the
NAND array. Unlike ReRAM or PCM crossbar arrays [ 21,71], which
rely on analog current accumulation for dot-product computation,
NAND flash inherently operates on discrete states, e.g., serial string
connect vs. disconnect, making such analog crossbar-style accu-
mulation impractical. In particular, the compact cell with only one
transistor makes it difficult to add any functionality, which greatly
restricts the capability of the NAND-based ISP.
To address these shortcomings, we present D-NOVA ( Dual-Bound
NAND-based Optimized search with Vector Adaptation). D-NOVA
is a highly parallel in-storage retrieval architecture that embeds
search functionality deeply into the memory array with negligible
peripheral hardware changes. It also incorporates a new distance
metric,dual-bound tight similarity sensing(DTS), tailored for in-
array execution within NAND strings. DTS reformulates vector
retrieval as query-dependent threshold search over ordered 𝑉TH
levels, using complement sensing for accurate search and compact
scoring. Unlike prior in-memory computing approaches [ 21,71]
that often rely on analog current accumulation, D-NOVA is strictly
digital and operates solely through binary connect/disconnect deci-
sions along NAND strings, without any reliance on analog accumu-
lation or current summation. It also executes the three-stage IVF
pipeline, including centroid search, coarse Top- 𝐾2retrieval, and
fine Top-𝐾1re-ranking, entirely within 3D NAND arrays, while
the SSD controller handles only lightweight scoring operations.
D-NOVA’s key contributions are:
•In-storage searching and retrieval:D-NOVA performs true in-
storage search within the NAND’s multi-level cells, where similarity
is evaluated directly inside the NAND array by comparing the stored
cell values with query-driven wordline voltages. It reuses existing
peripheral circuitry to perform local group-wise comparisons with
negligible hardware overhead. The operation is fully digital, relying
on on/off evaluations that avoid the analog non-idealities observed
in analog-domain processing.
•Accurate DTS search:We propose a DTS framework that serves
as a tailored distance metric for performing search operations in the
NAND string. DTS narrows the candidate set by applying tight dual
bounds (upper and lower) during search. This enables fine-grained
similarity evaluation using simple binary sensing instead of the
iterative read process required for multi-level cell reads.
nlist= 5nprobe= 2QueryEmbeddingsCentroidsSelect!
(a) 1stStage (Centroid search)
Select!
(b) 2ndStage (Top-K2Search)K2= 5
FinalTop-K1K1= 1
(c) 3rdStage (Top-K1Search)d	=	128nlist	=	5#indexingquantizer	=	faiss.IndexFlatL2(d)index	=	faiss.IndexIVFFlat(quantizer,	d,	nlist)Index.train(data)Index.add(data)#stage	1nprobe	=	2index.nprobe=	nprobe#stage	2K2	=	5	D_all,	I_all=	index.search(query,	K2)	#stage	3K1	=	1	D_top1	=	D_all[:,	:K1]	I_top1	=	I_all[:,	:K1]
Indexing
Accelerated by D-NOVA ISPJFigure 2: IVF three stages. (a) Centroid search, (b) coarse-
grained Top-𝐾 2search, and (c) Top-𝐾 1search (Re-ranking)
•Database domain conversion through custom adapter:We
introduce an adapter that projects embeddings into a DTS-aligned
space, bridging the gap between widely used cosine/L2 metrics and
DTS. The adapter can be trained onceofflineusing a contrastive
objective on document vectors, restoring near-full IVF recall with
near-zero runtime overhead for in-storage search.
•Controllable efficiency and accuracy configuration:We pro-
vide a controllable knob to trade off efficiency and accuracy. The
number of wordlines sensed in parallel per block, denoted as 𝑚,
is dynamically adjusted across IVF stages, given their accuracy
requirements and data volume for the stage-wise optimization.
Through this cross-layer co-design, D-NOVA demonstrates that
large-scale semantic retrieval can be executed almost entirely in-
storage, effectively eliminating data movement while maintaining
high accuracy. When evaluated on the NQ and HotpotQA datasets,
D-NOVA achieves33 .3−41.7×and4.3−12.1×higher throughput
and58.8−71×and1.1−1.26×lower energy consumption than CPU
baseline and SOTA [ 10], respectively, establishing a new direction
for scalable and energy-efficient in-storage retrieval acceleration.
Its robustness and feasibility are also validated through detailed
noise simulations (Section 6.4), while implementation overheads,
including area and power, are quantified in Section 6.7.
2 Background and Related Works
2.1 RAG Retrieval Pipeline and IVF Hierarchy
RAG systems consist of two main stages: (1)retrieval, which locates
semantically relevant documents from a large embedding DB given
the query, and (2)generation, where an LLM uses the retrieved
context to produce the final response. The retrieval stage dominates
the system cost, as it involves searching billions of high-dimensional
vectors per query [ 74,82]. This search is typically implemented
using ANN algorithms, such as the IVF hierarchy [ 19,90], due to
IVF’s scalability and predictable access patterns.
The IVF hierarchy decomposes search into three coarse-to-fine
stages (Fig. 2). First, all DB embedding vectors are partitioned into
𝑁𝑛𝑙𝑖𝑠𝑡 (e.g.,103–104) clusters via 𝑘-means [ 2], and the query 𝑞is
compared against all centroids to select the top 𝑁𝑛𝑝𝑟𝑜𝑏𝑒 (e.g.,8–
256) clusters (Fig. 2(a)). Second, only vectors in the probed clusters
are evaluated to produce a coarse Top- 𝐾2candidate list (typically
𝐾2≈5–20×𝐾 1) (Fig. 2(b)). Finally, a fine-grained re-ranking stage
ranks the coarse Top- 𝐾2candidates and selects the Top- 𝐾1(e.g.,1–
100) results (Fig. 2(c)). After the Top- 𝐾1indices are determined, the
2

D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound
3D NAND-Optimized Similarity Search with Vector Adaptation MICRO 2026, October 31–November 04, 2026, Athens, Greece
SystemTowards Fully In-Situ Execution JFinal-stage storage-sideCandidate reduction [X]Total storage-side movement reduction [X](a)(b)D-NOVA(Our Work)
LowEnergy &FastRetrievalJ: ISPD-NOVA(Our Work)
SPANN&FusionANNSREIS3rdStageRe-ranking
CPU
SSDRaw VectorsData Movement ↑LRe-ranking
EmbeddingsData Movement ↑L
Embeddings1st& 2ndStageL
CPU
SSDJCentroids & EmbeddingsBinaryXOR SearchBinaryDTS
CPUJCentroids & EmbeddingsSearch
CPU
SSDCentroids
GPU
orInv. List & PQ Code Data Movement ↑
No Data MovementNo Data Movement
All accumulation and similarity evaluation occurs in-storage
SSDCtrl.JNo Data MovementRe-rankingw/ Binary DTS
SSDCtrl.
Flash
Flash
FlashHotpotQAw/ mpnet
Figure 3: (a) Movement-centric comparison of representative
retrieval designs, and (b) data movement reduction plot.
corresponding raw documents are fetched from storage to the LLM
for generation. This structured, batched pipeline (centroid search
→coarse-grained search →compact re-ranking) yields predictable
and parallel access patterns, making IVF hardware-friendly.
Conventional software frameworks such as FAISS [ 19] typically
store the IVF index in host memory [ 90], but execute different
retrieval stages—centroid assignment, candidate scoring, and re-
ranking—on heterogeneous compute engines such as CPUs and
GPUs [ 80], where CPUs handle control-intensive centroid assign-
ment while GPUs accelerate large-scale similarity computations. As
a result, embeddings and partial scores are frequently transferred
between host DRAM and accelerator memory, and for billion-scale
databases, these PCIe or NVLink transfers dominate both latency
and energy [ 33,45]. Unlike graph-based methods such as HNSW,
whose irregular pointer-chasing and unpredictable access patterns
hinder efficient hardware mapping [ 84], IVF offers structured and
highly parallel access patterns, making it a strong candidate for
ISP or near-storage processing (NSP) acceleration, where similarity
computations can be executed directly inside or near the storage
arrays [ 10,79]. However, if any pipeline stage is not covered by
the ISP/NSP functionality, intermediate data must still leave the
storage, reintroducing data movement as the dominant bottleneck.
2.2 Limitations of Existing RAG Accelerators
Several prior works explore executing the retrieval pipeline using
memory hierarchy optimization or ISP/NSP. Fig. 3(a) shows where
each IVF stage executes and what data are transferred outside stor-
age, while Fig. 3(b) quantifies both overall data movement and
re-ranking-stage traffic, which typically dominates and is hardest
to process in storage due to its high accuracy requirement. Across
these designs, intermediate data are still exported for off-storage
processing, leaving data movement as the key bottleneck. For ex-
ample, SPANN [ 12] adopts a DRAM–SSD hybrid design that keeps
centroids in host DRAM and stores large inverted lists on SSD. Fu-
sionANNS [ 80] further extends this design to a CPU/GPU/SSD hier-
archy, incorporating PQ-based filtering and raw-vector re-ranking
from SSD. In both cases, later-stage candidate processing remains
on the host side, requiring intermediate data to move between stor-
age and host memory. RAGX [ 47] and related near-storage and
PIM-based designs [ 5,11,13,28,83] push portions of computation
closer to storage, reducing host-side traffic. However, they still
rely on off-array processing and require intermediate data to be
transferred to near-storage processors across multiple stages of
the pipeline, leaving data movement as the dominant bottleneck.
REIS [ 10] moves cluster selection and coarse binary search into
(a)# of cellsEP1P2P3P4P5P6P7
MSB
LSB
CSB
VTHVLSBVCSBVMSBVCSBVLSBVCSBVMSBVTHMarginVPASS111110100000010011001101(b)TLC SensingError after retention lossCellFigure 4: (a) 3D NAND block structure, and (b) 𝑉THdistribu-
tion of NAND TLC.
NAND, but still performs fine-grained INT8 re-ranking on the con-
troller. However, due to the limited precision of binary screening,
REIS requires both a large number of clusters (e.g., high 𝑁𝑛𝑙𝑖𝑠𝑡) and a
large re-ranking candidate set to maintain accuracy. Consequently,
large intermediate data must be fetched into the SSD controller for
final re-ranking, incurring high latency from TLC/QLC reads [ 76].
These limitations motivate D-NOVA, which performs entire IVF
three stages in storage using DTS-based in-array similarity evalu-
ation (Section 3.2) and in-array accumulation, ensuring that only
compact metadata—rather than raw embeddings or large interme-
diate candidate—leaves the array. By doing so, Fig. 3(b) shows that
D-NOVA achieves the highest overall and re-ranking stage data
movement reduction compared to prior designs.
2.3 Overview of 3D NAND flash and Challenges
Fig. 4(a) shows a 3D NAND block consisting of vertical strings in-
tersected by horizontal wordlines (WLs). A transistor cell is formed
at each WL–string intersection, and the cells within a string are
connected in series between a bitline (BL) at the top and a ground
select line (GSL) at the bottom. Each string is controlled by a string
select line (SSL) and a GSL, enabling selective activation during
read/program operations. Multiple strings are grouped into sub-
blocks, where BLs connect them to peripheral sense amplifiers (SAs)
and page buffers, which determine the page-level read granularity.
During a read operation, all BLs in one SSL group (strings sharing
the same SSL) are sensed in parallel while one WL is enabled at
a time. The selected WL is driven by a read voltage ( 𝑉read), while
unselected WLs are biased with a pass voltage ( 𝑉pass) to ensure they
remain conductive. If 𝑉readexceeds the selected cell’s threshold
voltage (𝑉TH), the cell conducts, and the string connects to the BL;
otherwise, the string remains disconnected. In modern multi-level
3D NAND, TLC (3 bits/cell) and QLC (4 bits/cell) encode2𝑏𝑉TH
states per transistor ( 𝑏=3or4), increasing density but requiring
multiple sensing steps to determine the stored value. Conceptually, a
𝑏-bit cell requires2𝑏−1𝑉reads (Fig. 4(b)), applied sequentially during
readout. In practice, controllers employ multi-level sensing and
read–retry schemes to reduce the average sensing cost per read [ 8].
Even with these optimizations, QLC readout remains several times
slower and more energy-intensive than an SLC sense.
One WL layer in a single SSL group (i.e., a sub-block), along with
its associated BLs, constitutes a logical page that can be read in a
single sensing cycle (e.g., 2 KB page with 16K BLs). Since all data in
a page are accessed simultaneously, each vector is mapped entirely
within a single page by distributing its embedding dimensions
across BLs in the same SSL-selected string group. Unlike ReRAM-
3

MICRO 2026, October 31–November 04, 2026, Athens, Greece Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S Rosing, and Mingu Kang
and PCM-based crossbar arrays, which implement analog PIM via
current accumulation from multiple cells connected in parallel to
each BL [ 21,71], NAND flash strings are serially connected and
operate in a digital manner, where each cell contributes a binary
conduction state (i.e., connected or disconnected). To accommodate
this structural and computational difference, D-NOVA introduces a
new distance metric,DTS, that enables accurate in-storage search
directly within 3D NAND arrays with negligible modification to
the existing structure, as described in Section 3.2.
3 D-NOVA
This section presents the overall D-NOVA architecture and explains
how the proposed in-storage DTS retrieval operates across the
plane. D-NOVA combines NAND-string threshold sensing, page-
buffer/peripheral deficit accumulation, and compact score/metadata
movement to execute the three-stage IVF retrieval flow inside
the storage device. We first describe the SSD-level organization—
including the controller, channels, dies, planes, and 3D NAND
blocks—that enables in-storage computation across NAND arrays
and the controller. We then detail the DTS mechanism and its over-
head, and outline how it integrates with the IVF-based three-stage
retrieval flow.
3.1 Overall Architecture
D-NOVA is implemented as an in-storage accelerator fully inte-
grated within an SSD system, as shown in Fig. 5. The system consists
of an SSD controller and a D-NOVA storage chip. The SSD con-
troller hosts flash controllers (one per channel) and an embedded
processor connected to SRAM, which performs lightweight sorting
operations such as quicksort [ 10,25,70] or quickselect [ 10,48,50]
on intermediate results from each retrieval stage, as well as meta-
data management for cluster centroids and Top- 𝐾candidate lists.
Each incoming query is broadcast from the controller to all active
channels for parallel retrieval. The D-NOVA storage system com-
prises𝑁𝐶independent channels, each connected to 𝑁𝐷dies within
a single MCP (multi-chip package) [ 36,52]. The channel-level par-
allelism enables concurrent execution of multiple queries, while
each die performs independent in-storage searching. Inside a die,
𝑁𝑃planes are organized via H-Tree bus [ 84], enabling balanced
signal distribution and low-latency communication across planes.
Each plane includes an array of 3D NAND blocks, which serve as
the fundamental computational storage units. Each NAND block
consists of 16K BLs, 128 WLs, and 16 SSLs, and multiple NAND
blocks within a plane share the same BL array, as shown in Fig. 4(a).
Each BL is connected to a shared SA and a local page buffer that
temporarily stores binary sensed outcomes.
Both embedding vectors and raw document chunks reside in
QLC, with vectors stored in a dedicated embedding region where
each cell encodes an INT4 value, and document chunks stored in a
separate document region. Each embedding maintains a compact
pointer to its corresponding document. After the Top- 𝐾embeddings
are selected, their identifiers are translated locally into document
addresses, and the corresponding text payloads are fetched directly
from the document region. This unified in-storage organization
enables D-NOVA to perform vector retrieval entirely within the
SSD, transferring only the resulting documents to the host.
(Top-K Selection)Embedded ProcessorSRAM(Address + Metadata)SSD Controller
……
D-NOVA SSD
…Plane #3Plane #NP-3Plane #NP-1Plane #4Plane #NP-2Plane #NPI/O Buffer(b) DieSSD Controller
Flash Ctrl. #NC…
Flash Ctrl. #2
Flash Ctrl. #1
WL DecoderPage BufferPeri. Logic(c) Plane(a)…
3D NANDBlock3D NANDBlockCuA
Plane #1Plane #2
Ch #1Ch #NC
NAND PKG(Multi-Dies by MCP)
Die #1Die #ND…
DRAM
3D NANDBlock
NAND PKG(Multi-Dies by MCP)NDFigure 5: (a) Overall architecture of D-NOVA SSD connected
with SSD controller, (b) die with multi-planes, and (c) plane
containing 3D NAND blocks, and peripheral circuitry.
3.2 Proposed Complement-Based DTS Scheme
For accurate in-storage similarity evaluation on dense INT4 embed-
dings, D-NOVA introduces a complement-basedDual-Bound Tight
Similarity Sensing (DTS)scheme tailored for the NAND string. DTS
narrows the candidates by applying upper and lower bounds (UBS
andComp UBS) without modifying the array structure. While sim-
ple bound-based search has been applied to highly error-tolerant
workloads such as mass spectrometry using hyperdimensional com-
puting [ 61], this naive strategy does not work in accuracy-sensitive
RAG, motivating the proposed DTS design.
In the D-NOVA configuration, unlike conventional NAND map-
pings where vectors are stored within a single page along the WL
direction, each embedding vector is mapped along a vertical string,
with its dimensions stored in NAND cells across successive WL
layers, as shown in Fig. 6. Since these embeddings are not conven-
tionally read as QLC pages during DTS scoring, D-NOVA maps
INT4 values to sequentially ordered 𝑉THlevels, rather than Gray-
coded logical page labels. In contrast, raw document chunks remain
stored in the conventional page-wise layout to enable fast fetch-
ing after Top- 𝐾retrieval, and can use standard Gray-coded QLC
encoding for normal NAND reads. During search, the query vec-
tor (QV) is given as WL voltages, each of which is proportional
to the magnitude of QV’s corresponding element. But, unlike the
conventional read operation, multiple ( 𝑚=1 - 8) WLs are enabled
at a time. 3D NAND devices already support multi-WL activation
modes [ 43,54,58,65,66] with a negligible area and latency over-
head. The details of the 𝑚configuration and the associated over-
heads are discussed in Section 4.1 and Section 6.7, respectively. As
an example in Fig. 6, we assume 𝑚=4and consider four consecutive
dimensions of an embedding vector (EV) [8, 4, 5, 8] and a QV [7,
7, 4, 7], which are stored as INT4 values, with the EVs encoded in
cell𝑉THlevels. These four dimensions are sensed together in one
cycle by enabling one SSL and four WLs, allowing the SA to detect
the combined binary conduction state of the string (on/off). This
fully digital sensing ensures robust operation compared to analog
PIMs; Noise effects from process variation and the impact of the
potential increase in on-resistance due to multi-WL enabling are
discussed in Section 6.4 and Section 6.7, respectively.
Upper-Bound Similarity Sensing (UBS):During UBS, each WL
is driven with a voltage slightly higher than the corresponding
query vector (QV) element, i.e., 𝑉WL=𝑞𝑖+𝛼, where𝛼is a tolerance
margin (set to 𝛼=2in Fig. 6(a)) and 𝑞𝑖denotes the𝑖-th element of the
QV. If the stored 𝑖-th embedding element ( 𝑟𝑖) satisfies𝑟𝑖<𝑞𝑖+𝛼, the
applied WL voltage exceeds the cell 𝑉TH, turning on the conduction
path. When all 𝑚cells in a group satisfy this condition (Fig. 6(a)), the
4

D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound
3D NAND-Optimized Similarity Search with Vector Adaptation MICRO 2026, October 31–November 04, 2026, Athens, Greece
VDD……V(7) + ⍺=V(7) + ⍺=SSL
GSLV(4) + ⍺=V(7) + ⍺=8458Pass condition: QV+ ⍺＞EVàOn
UBSscore= “1”Current flowing= PassmEV#1QV#1OnOnOnOn(a) UBS
(a)+(b)=Conv.InaccurateSensingL(a)+(c)=ProposedDTSw/⍺=2,m=4J
9>
9>
6>
9>Passif all OnPass(b) LBSVDD……V(7) -⍺V(7) -⍺SSL
GSLV(4) -⍺V(7) -⍺8458LBSscore= “1”No current= PassEV#1QV#1OffOnOffOff
5<
5<
2<
5<
❌Pass condition: QV-⍺< EVàOffPassif at least one OffPass
QV > 2ndcell, but still sensing as PassàInaccurate sensing LPass condition: QV+ ⍺＞EVàOnVDD……V(8) + ⍺=V(8) + ⍺=SSL
GSLV(11)+ ⍺=V(8) + ⍺=711107Comp UBSscore= “0”EV#1’QV#1’OnOffOnOnNo current= Fail(c) Comp UBS
Same logic as UBS w/ complement valuesFail àCorrect Sensing J
10>
10>
13>
10>
❌Passif all OnFail+vs.
Figure 6: Overview of (a) UBS, (b) LBS, and (c) Comp UBS.
string conducts, and the group is counted as a pass. This mechanism
ensures thatallelements of the stored vector remain below an upper
bound defined by the query, i.e., 𝑞𝑖+𝛼. Formally, the upper-bound
score (UBS score) is defined as:
UBS score𝑗=𝑚𝑗+𝑚−1Ö
𝑖=𝑚𝑗[𝑟𝑖<𝑞𝑖+𝛼].(1)
where𝑗denotes the index of the 𝑚-cell group sensed in parallel.
UBS thus enforces a strict all-or-nothing criterion, as all cells in the
group must satisfy the condition for the group to pass.
Lower-Bound Similarity Sensing (LBS):As UBS checks only the
upper bound, we employ LBS to complete the dual-bound check.
LBS applies 𝑉WL=𝑞𝑖−𝛼to check whether 𝑟𝑖>𝑞𝑖−𝛼. Since
the applied voltage is slightly lower than the corresponding QV
element, it falls below the threshold of matching cells, turning off
the conduction path. Due to the serial nature of the NAND string,
ifanyof the 𝑚cells satisfies this condition, the string becomes
non-conductive, and the group is marked as a pass ( LBS score=1).
Formally, the lower-bound score (LBS score) is defined as:
LBS score𝑗=1−𝑚𝑗+𝑚−1Ö
𝑖=𝑚𝑗[𝑟𝑖<𝑞𝑖−𝛼].(2)
Because the serial NAND string reports a “1” when even a single
element satisfies the bound, LBS is inherently much looser than UBS,
as illustrated in Fig. 6(b). This lenient behavior is tolerable in noise-
resilient workloads such as hyperdimensional computing [ 61,85],
where extremely high-dimensional vectors ( >10K dimensions) can
statistically absorb sensing errors, but it introduces false positives
in dense embeddings, degrading recall by over 10% compared to
cosine or L2 search, thereby motivating a new distance metric.
Complement Upper-Bound Similarity Sensing (Comp UBS):
To preserve strictness while extending coverage, D-NOVA replaces
the relaxed lower-bound path (LBS) with a complement-based sens-
ing mechanism. As shown in Fig. 6(c), each INT4 value 𝑟𝑖in the
embedding vector is stored with its 4 bits complement 𝑟′
𝑖=15−𝑟𝑖
(and similarly 𝑞′
𝑖=15−𝑞𝑖for the query). This complement transfor-
mation reverses the inequality, enabling the use of UBS in place of
LBS. Formally, the Comp UBS is written as:
Comp UBS score𝑗=𝑚𝑗+𝑚−1Ö
𝑖=𝑚𝑗[𝑟′
𝑖<𝑞′
𝑖+𝛼],(3)
This complement sensing eliminates false positives that LBS would
otherwise produce, restoring fine-grained discrimination across
INT4 embeddings. The data arrangement of normal and comple-
mentary embeddings and their potential data volume overhead are
discussed at the end of this section.
ScoreDTS> 64à7-bitsToo Large L
𝚫ScoreDTS< 16à4-bitsFit in DLJNormalized CountAccumulated Score per 128 WLs
Accumulated Score Deficit (𝚫)  per 128 WLsNormalized Count
Smax
High Score
Low 𝚫= High Score…SASASA
𝚫ScoreDTS
1
Pass1bUpdateAccumulator…
7 + 0
3 + 0
5 +1
3 + 04b
1
0
1Peri. Logic
1
1
0
1Adder(a)VDD……VDD……VDD……VDD……
EV#1
EV#2
EV#3
EV#NBL………………55335832351222014523QV #1 (t-1)SSL
GSLWL1WL2WL3WL4+⍺+⍺+⍺+⍺Simultaneous Row Check
Query Vector #1 (QV#1)…e.g.,) UBS w/⍺= 2, m = 4
❌
Embedding Vector #NBL(EV #NBL)SA
CL (t)
SL(t-1)
DL[3:0](t)
FailCuA (CMOS under Array)Sub-Block #1Sub-Block #256…Sub-Block #2
MiniLM, HotpotQAmpnet, HotpotQA(b)
(c)
4b
MiniLM, NQmpnet, NQ
High QuantileNormalized Count
Normalized CountAccumulated Score Deficit (𝚫)  per 128 WLsAccumulated Score Deficit (𝚫)  per 128 WLs(d)(e)
𝚫ScoreDTS< 16à4-bitsFit in DLJ
Low 𝚫= High Score
𝚫ScoreDTS< 16à4-bitsFit in DLJ
Low 𝚫= High ScoreFigure 7: (a) DTS in NAND and ΔScore DTSaccumulation, (b)
second stage accumulated score distributions, (c) correspond-
ing deficit ( 𝑆max−𝑆) distributions, and (d) first, and (e) third
stage deficit distributions under𝑚=(1,1,1).
Final Scoring with both metrics:The final dual-bound score is
obtained by aggregating UBS and Comp UBS results across all the
𝑚-cell groups as follows:
Score DTS=𝐷/𝑚∑︁
𝑗=1(UBS score𝑗+Comp UBS score𝑗)=𝐷/𝑚∑︁
𝑗=1DTS𝑗,(4)
where𝐷is the embedding dimension and 𝑚is the number of WLs
sensed in parallel. A higher Score DTSindicates that more groups fall
within valid bounds in both the original and complement domains,
providing finer discrimination for in-array matching. Beyond the
search capability itself, aggregation across multiple 𝑚-cell groups
serves as an important mechanism for providing robustness and
resilience. Even if individual groups fail due to process variation,
their impact is mitigated through accumulation, as errors are statis-
tically averaged out across groups. This aggregation preserves the
correctness of the search results despite device-level variability.
Delta Scoring for Compact Accumulation:In this design, the
scores are accumulated across all the 𝑚-cell groups, requiring wide
per-BL storage to hold the accumulated values. In the worst case
(𝑚= 1), accumulation over 128-WLs NAND string yields a maxi-
mum raw score of 𝑆max=128, requiring at least 7 bits per BL for
brute-force accumulation. Fig. 7(b) shows the second stage score
distributions across datasets and encoders, where accumulated val-
ues range from∼105 to 120, requiring 7-bit representation. Such
wide storage cannot be embedded within the page buffer and does
not fit within its storage budget, necessitating additional storage
or off-array aggregation. To address this challenge, D-NOVA in-
stead accumulates thescore deficitfrom the window-level maximum
(𝑆max), defined as ΔScore DTS=𝑆max−𝑆, where𝑆is the accumulated
score over the 128-WLs string. As shown in Fig. 7(c), the deficit
values remain small, ranging from 5 to 13 even at high quantiles,
indicating that ΔScore DTScan be represented within 4 bits. Fig. 7
(d) and (e) show that the fine-grained first and third stages also
5

MICRO 2026, October 31–November 04, 2026, Athens, Greece Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S Rosing, and Mingu Kang
NnlistàNnprobe(e.g., 1024 à64)Embeddings in NnprobeàTop-K2  (e.g., 30K à1K) …Top123K1…Top-K1Retrieval Set⓿WL on (Sensing)WL off (No Sensing)s1s2s3Embeddingsor CentroidsEmbeddings’orCentroids’(Complement)Dimension (D=384)NWL=128s1’s2’s3’s1s2s3Query Vectorss1's2's3'Query Vectors’(Complement)…
Accum.Peri. LogicSRAMMeta DataEmbedded Processor(S1) QuickselectSSD Controller❺❹SADL [3:0]Page Buffer❸……Plane  #1NBL
s1❶……Plane  #6NWL
s3’❶❷……
s2❶Plane  #2…(a)
WL Decoder
m = 2Accum.Peri. LogicSRAMMeta DataEmbedded Processor(S2) QuickselectSSD ControllerSADL [3:0]Page Buffer……Plane  #1NBL
s1
s1……Plane  #6NWL
s3’………
s2Plane  #2…Sub-Block #2 ... #N(Embeddings)(b)
m = 4(c)………Plane  #1Plane  #6…Plane  #2
Accum.Peri. LogicSRAMMeta DataEmbedded Processor(S3) Quicksort SSD ControllerSADL [3:0]Page Buffer…NBL
s1
s1…NWL
s3’…
s2…
m = 2WL Decoder
WL DecoderSub-Block #1(Centroids)1stCentroid(Same BL positionacross planes)NBLthCentroid
s3’
s2
s1
❼❻…NBL
Figure 8: Vector mapping and IVF data flow in D-NOVA for the (a) first, (b) second, and (c) third stages.
stay within the 4-bit range, with high quantiles of 7–12 and 7–9
under conservative 𝑚=( 1,1,1). Practical larger- 𝑚settings further
reduce the accumulated deficit range, increasing the margin for
4-bit storage.
We therefore reuse the existing 4-bit data latch (DL) for QLC in
the page buffer to store ΔScore DTS, as shown in Fig. 7(a), eliminating
the need for additional wide per-BL storage. For the score deficit
calculation, each 1-bit DTS outcome for an 𝑚-cell group is first
captured in the existing sensing latch (SL) and then transferred to
the cache latch (CL) in the page buffer. A lightweight peripheral
adder then updates the DL by accumulating the inverted CL value,
i.e., the logical complement of the 1-bit sensing outcome. Thus, a
passing group contributes "0" to the deficit, and a failing group
contributes "1". On the other hand, a 4-input NAND gate disables
further updates once the 4-bit deficit reaches 1111 to avoid overflow
and safely ignore low-score candidates, which does not affect the
final recall rate. After all WLs are processed, the compact delta
scores are forwarded to the controller’s embedded processor for
final ranking and aggregation. As the lightweight accumulators are
shared across multiple BLs in a time-multiplexed manner, it incurs
only minimal overhead, as discussed in Section 6.7.
DTS Complement Data Overhead and Speed Benefit:DTS
stores paired 4-bit values (EV and EV′) in different planes, preserv-
ing the same data volume as an INT8-based system NSP baseline
(described in Section 5.4) and REIS [ 10] while enabling parallel
search across planes. Unlike REIS, which allocates separate SLC
regions for binary vectors—thereby wasting three bits per QLC
cell—and additional INT8 data for re-ranking, D-NOVA maintains a
uniform 4-bit format aligned with QLC, eliminating redundant ca-
pacity overhead. In terms of latency, DTS replaces multi-level QLC
reads with a small, fixed number of binary sense operations, avoid-
ing the 7–15 iterative reads by sweeping 𝑉𝑟𝑒𝑎𝑑, which is required
for full QLC decoding shown in Fig. 4(b). The total sensing cycles
(𝑇) are given by 𝑇DTS=(𝐷/𝑚)× 2for DTS and 𝑇QLC=𝐷×𝑁 rdfor
conventional QLC reads, where the factor of 2 accounts for UBS and
Comp UBS, and 𝑁rdis the number of sense operations per multi-
level cell (2𝑏−1, e.g., 7 for TLC and 15 for QLC). Consequently, DTS
achieves up to a 9.3×reduction in sensing cycles and comparable
speedup over REIS without additional storage overhead.3.3 Three-Stage IVF Retrieval and Data Flow
D-NOVA implements an end-to-end in-storage IVF retrieval hi-
erarchy, storing all centroids, embeddings, and document chunks
directly within the SSD. Unlike prior ISP/NSP architectures such
as REIS [ 10] and RAGX [ 47], it executes the entire three-stage sim-
ilarity search within the array with minimal data transfer to the
embedded processor. Intermediate similarity results are accumu-
lated in plane-level peripherals, while only final scores and compact
metadata (e.g., addresses and centroid identifiers), which are much
smaller than raw embeddings, are stored in the controller memory,
minimizing data movement.
Fig. 8 shows the vector mapping and retrieval data flow for each
stage. For exposition purposes, assume each vector is represented
as a 384-𝐷INT4 vector along with its 4-bit complement for DTS
computation, divided into six 128- 𝐷segments to fit within 128
WLs. In the Fig. 8(a), the six segments—original (s1, s2, and s3) and
complement (s1’, s2’, and s3’)—are each vertically mapped onto
separate NAND strings located in different planes, but at the same
sub-block position in their planes. These six strings together form
a logical group that is sensed simultaneously within a single cycle.
During thefirst stage, for example, centroid data is deliberately
distributed across the first sub-block of each plane, and multiple
centroids are packed to fill the first sub-block of each plane as
shown in Fig. 8(a). The first sub-block stores up to 16K centroids
(considering 16K BLs), and the rest store embeddings or document
chunks. During DTS, 0the query vector is broadcast from the
SSD controller to all active channels for parallel processing. Each
channel delivers it to multiple dies and planes via the H-Tree rout-
ing network [ 34,71,84].1The query is applied as WL voltages
through the WL driver for DTS-based centroid search (Fig. 8(a)).
We use small 𝑚values (e.g., 1-4) for this stage to maintain high
precision centroid matching (the 𝑚-adaptive sensing mechanism
is detailed in Section 4.1). 2Each segment-level 1-bit outcome is
captured by SA and stored in the SL and transferred to the CL, and
then 3theΔScore DTSis incrementally accumulated in the page
buffer DL via a lightweight peripheral accumulator. Because the
centroid segments reside in different planes and each sub-block
holds multiple centroids’ segment, D-NOVA performs comparisons
over a large volume of centroids in parallel both across planes and
6

D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound
3D NAND-Optimized Similarity Search with Vector Adaptation MICRO 2026, October 31–November 04, 2026, Athens, Greece
within each sub-block. After score accumulation, 4the 4-bit accu-
mulated scores from each plane are merged at the SSD controller to
form a 6-bit ΔScore DTSper centroid, and only lightweight metadata
is temporarily stored in the controller’s on-chip SRAM. Each piece
of metadata consists of a log2(𝑁list)-bit centroid identifier (CID), a
6-bit distance value (DIST), and a compact address field (CADR)
including the physical plane, block, SSL, die, and channel indices
(∼37-bit in total). 5These {CID, DIST, CADR} entries are then
transmitted to the embedded processor for lightweight quickselect
reduction [ 10,48,50] to locate the top 𝑁𝑝clusters. With 𝑁𝑝=64,
the total SRAM footprint remains <0.5KB, which is negligible
compared to the controller’s capacity.
Thesecond stage(Fig. 8(b)) refines retrieval by evaluating only
the vectors within the chosen clusters to produce a coarse Top-
𝐾2candidate list. Centroid identifiers (CIDs) are translated into
physical NAND addresses using a small lookup table (cluster di-
rectory). Each directory entry maps CID and CADR to its phys-
ical coordinates—plane number, sub-block ID, and BL segment
group—allowing direct activation of those planes. This operation
remains compatible with the conventional NAND access pattern,
as only the addressed sub-blocks are enabled while all other struc-
tural and timing behaviors follow standard NAND operation. Each
activated sub-block performs DTS-based comparison using larger
𝑚values (e.g., 4–8) to increase throughput, as the coarse retrieval
stage accounts for over 95% of all similarity search operations. Par-
tial scores from each segment group are accumulated locally by
their corresponding accumulators and merged in the controller,
and the embedded processor identifies the Top- 𝐾2candidates using
the same 6quickselect procedure. Initially, each candidate entry
is represented in a {EID (embedding ID), DIST, EADR (embedding
address)} format. After candidate selection, to minimize memory
usage, the processor retains only the {EID, EADR} pair of the Top- 𝐾2
candidates. Even for a 1B-vector DB with 768- 𝐷embeddings, the
transient controller-side SRAM required to receive, aggregate, and
buffer compact candidate metadata during Top- 𝐾2selection remains
within a few megabytes, well within the available memory budget
(∼8 MB)[ 46]. In thethird stage(precise re-ranking, Fig. 8(c)), the
corresponding small set of planes and sub-blocks are reactivated
for fine-grained re-ranking for higher precision, using a smaller 𝑚
(e.g., 1–4) to increase sensing precision. After accumulation, 7the
embedded processor performs a final quicksort [ 25,70] to rank the
Top-𝐾1results, which contributes <0.1%of the total latency and
energy due to the highly limited candidate set.
This data flow avoids reading full embeddings out of NAND for
conventional distance computation. D-NOVA does not implement
cosine similarity or L2 distance inside the NAND array; instead,
each IVF stage uses DTS as a threshold-based retrieval score. The
NAND cell array performs query-dependent bound-check sensing
over ordered INT4 𝑉THlevels, page-buffer/peripheral logic accumu-
lates compact deficit scores, and the embedded controller performs
comparison-based Top- 𝐾selection over compact scores and meta-
data. Unlike controller- or ASIC-based distance computation, which
must first read QLC/TLC embeddings through multiple sensing
cycles and buffer them before arithmetic scoring (e.g., Hamming-
distance or dot-product scoring), D-NOVA’s binary DTS sensing
evaluates the threshold score directly in the storage path while
preserving high-density QLC storage.
ßVector DBPassagesPassagesPassagesPassages
PassagesPassagesPassagesEmbeddingVectorsDTS SearchingEncoder(e.g., all-MiniLM-L6-v2, all-mpnet-base-v2)Correct! JWrong LWrong LWrong L…Top123K…Top-K Retrieval SetNegative(K-1)PositiveAnchor (Adapter)
PositiveAnchor (Adapter)àInfoNCEloss (𝓛)Training AdapterInitial AdapterAfter TrainingTraining SetNegative(K-1)nn.LayerNorm(in_dim)nn.Linear(in_dim, out_dim)nn.Linear(out_dim, in_dim)Activation (e.g., nn.GELU())QueryProposed Adapter for DTS697698699700701702703704705706707708709710711712713714715716717718719720721722723724725726727728729730731732733734735736737738739740741742743744745746747748749750751752753754D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound3D NAND-Optimized Similarity Search with Vector Adaptation MICRO 2026, October 31–November 04, 2026, Athens, Greece755756757758759760761762763764765766767768769770771772773774775776777778779780781782783784785786787788789790791792793794795796797798799800801802803804805806807808809810811812NnlistàNprobe(e.g., 1024 à64)Embeddings in NnprobeàTop-K2  (e.g., 30K à1000) …Top123K1…Top-K1Retrieval Set⓿WL on (Sensing)WL off (No Sensing)s1s2s3Embeddings(Embeddings)’Dimension (D=384)NWL=128s1’s2’s3’s1s2s3Query Vectorss1's2's3'(Query Vectors)’…Centroids
Accum.Peri. LogicSRAMMeta DataEmbedded Processor(S1) QuickselectSSD Controller❺❹SADL [3:0]Page Buffer❸……Plane  #1NBL
s1
s1❶……Plane  #6NWL
s3’
s3’❶❷……
s2
s2❶Plane  #2…(a)
WL Decoder
m = 2Accum.Peri. LogicSRAMMeta DataEmbedded Processor(S2) QuickselectSSD ControllerSADL [3:0]Page Buffer……Plane  #1NBL
s1
s1……Plane  #6NWL
s3’………
s2Plane  #2…Embeddings(b)❻
m = 4(c)………Plane  #1Plane  #6…Plane  #2
Accum.Peri. LogicSRAMMeta DataEmbedded Processor(S3) Quicksort SSD ControllerSADL [3:0]Page Buffer…NBL
s1
s1…NWL
s3’…
s2…❼
m = 2WL Decoder
WL DecoderFigure 8: Vector mapping and IVF data!ow in D-NOVA for the (a)"rst, (b) second, and (c) third stages.All computations are performed fully in-place within the NANDarray—no raw INT4 data or partial scores are ever read outside thedie. In contrast to a controller- or ASIC-based distance calculationprocessor, where QLC/TLC read operations require multiple sens-ing cycles and large local bu!ers, D-NOVA’s binary DTS sensingoperates in a single pass per WL, even when embeddings are storedin high-density QLC mode with full in-storage execution.4 D-NOVA Throughput and AccuracyEnhancement TechniquesTo further improve D-NOVA, we introduce three complementarytechniques: stage-aware𝐿-adaptive sensing for higher parallelism,a DTS-optimized adapter to improve recall, and a locality-awaremapping strategy for e"cient vector placement.4.1 Stage-Aware𝐿-Adaptive SensingThe DTS exposes a direct architectural knob for controlling intra-string parallelism. We denote by𝐿the number of WLs, each cor-responding to one vector dimension, sensed in parallel under asingle SSL activation. As discussed in Section3.3, 3D NAND de-vices already support multi-WL activation modes [37,47,57,58], soadjusting𝐿adds<1%overhead over existing WL drivers. A larger𝐿improves per-cycle throughput since each sensing step evaluates𝐿dimensions in parallel along a NAND string and produces a sin-gle pass/fail bit for the group. However, increasing𝐿also increasesthe resistance of the serially connected string, even when using“all-pass” stacks, thereby increasing the accumulated resistance. Asa result, very large𝐿values increase the likelihood of false nega-tives (i.e., cells that are connected appear disconnected), whereassmall𝐿values preserve a sharper sensing margin but require moresensing cycles per vector, creating a trade-o!between accuracyand e"ciency.In practice, however, not all stages of the IVF hierarchy areequally sensitive to this accuracy trade-o!. Centroid selection (#rststage) and#nal Top-𝑀1re-ranking (third stage) directly determinewhich clusters are explored and how the#nal results are ordered;errors in these stages can irreversibly exclude relevant documents.In contrast, coarse Top-𝑀2retrieval (second stage) primarily prunesa large candidate pool into a manageable working set (typicallyVector DatabaseQuery
Embedded QueryPassagesPassagesPassagesPassages
PassagesPassagesPassagesEmbeddingVectorsDTS Searchingnn.LayerNorm(in_dim)nn.Linear(in_dim, out_dim)nn.Linear(out_dim, in_dim)Activation (e.g., nn.GELU())Encoder(e.g., all-MiniLM-L6-v2, all-mpnet-base-v2)Correct! JWrong LWrong LWrong L…Top123K…Top-K Retrieval SetNegative(K-1)PositiveAnchor (Adapter)
PositiveAnchor (Adapter)(InfoNCEloss)Training AdapterInitial AdapterAfter TrainingTraining SetNegative(K-1)Proposed Adapter for DTSFigure 9: Structure of the adapter model, and the trainingprocess based on the DTS retrieval using infoNCEloss.𝑀2→𝑀1) and can tolerate modest sensing noise. Accordingly, D-NOVA adopts a stage-aware con#guration in which𝐿1and𝐿3arekept small for accuracy-critical stages (e.g.,𝐿1,𝐿3=1↑2), while𝐿2is increased to maximize throughput during coarse retrieval (e.g.,𝐿2=4↑8). The performance and accuracy trends from sweeping𝐿are presented in Section6.3.4.2 Contrastive Adapter Co-DesignExisting text encoders such as all-MiniLM-L6-v2 [75] and all-mpnet-base-v2 [66] are optimized for continuous similarity metrics such ascosine similarity or L2 distance, whereas D-NOVA’s DTS operatesin a discrete, threshold-based metric space. Applying DTS directlyto these embeddings introduces a metric mismatch that degradesrecall, especially for semantically similar vectors separated by smallnumeric deviations. To mitigate this mismatch, we introduce alightweight query-side adapter that reshapes encoder outputs intoa DTS-aligned representation, as shown in Fig.9.Training uses a retrieval-aware InfoNCE loss [51,74] constructedfrom DTS retrieval results rather than random negatives. For a setof queries{𝑁𝐿}, each query retrieves a Top-𝑀candcandidate poolfrom DTS search. The ground-truth passage𝑂+𝐿forms the positivepair(𝑁𝐿,𝑂+𝐿), while the remaining candidates form negative pairs(𝑁𝐿,𝑂↑𝐿𝑀). The adapter is trained to minimize the InfoNCE lossL=↑1𝑃𝑁/√︄ummationdi√︄√︁lay.𝐿=1logexp(sim(q↓𝐿,p↓+𝐿)/𝑄)exp(sim(q↓𝐿,p↓+𝐿)/𝑂)+/√︄ummationtext.𝑀↑1𝑁=1exp(sim(q↓𝐿,p↓↑𝐿𝑁)/𝑂)(6)7InfoNCEloss (𝓛) =Embedded QueryFigure 9: Adapter architecture and retrieval-aware training.
4 Throughput and Accuracy Enhancement
To further improve D-NOVA, we introduce three complementary
techniques: stage-aware 𝑚-adaptive sensing for higher parallelism,
a DTS-optimized adapter to improve recall, and a locality-aware
mapping strategy for efficient vector placement.
4.1 Stage-Aware𝑚-Adaptive Sensing
DTS exposes an architectural knob for intra-string parallelism via
the multi-WL activation mechanism [ 43,58] described in Section 3.2.
With𝑚WLs sensed in parallel within a string under a single SSL ac-
tivation, where each WL corresponds to one vector dimension, each
sensing step produces a single pass/fail bit for 𝑚dimensions, im-
proving per-cycle throughput. However, increasing 𝑚yields more
coarse-grained decisions, since sensing produces only a group-level
pass/fail outcome rather than element-wise results, reducing sens-
ing accuracy. In contrast, smaller 𝑚values preserve fine-grained de-
cisions, enabling more accurate sensing at the cost of lower through-
put, reflecting the trade-off between accuracy and efficiency.
In practice, however, not all stages of the IVF are equally sen-
sitive to this accuracy trade-off. Centroid selection and Top- 𝐾1
re-ranking directly determine which clusters are explored and how
the final results are ordered, respectively; errors in these stages can
irreversibly exclude relevant documents. In contrast, coarse Top- 𝐾2
retrieval primarily prunes a large candidate pool into a manageable
working set (typically 𝐾2≫𝐾 1) and can tolerate modest sensing
noise. Accordingly, D-NOVA adopts a stage-aware configuration,
where𝑚𝑠denotes the parallel sensing degree at stage 𝑠:𝑚1and𝑚3
are kept small for the accuracy-critical centroid selection and final
re-ranking stages (e.g.,1–4), while 𝑚2is increased during coarse
Top-𝐾2retrieval (e.g.,4–8), where throughput dominates and mod-
est sensing noise can be tolerated. This stage-wise rule provides
the design guideline for selecting 𝑚. Section 6.2 shows this asym-
metric configuration preserves accuracy, while Section 6.6 shows
that it delivers 4.3–12.1 ×higher throughput than REIS; Sections 6.4
and 6.7 further confirm robust operation under noise with minimal
hardware overhead.
4.2 Contrastive Adapter Co-Design
Existing text encoders such as all-MiniLM-L6-v2 [ 82] and all-mpnet-
base-v2 [ 74] are optimized for continuous similarity metrics such as
cosine similarity or L2 distance, whereas D-NOVA’s DTS operates
in a discrete, threshold-based metric space. Applying DTS directly
to these embeddings introduces a metric mismatch that can degrade
recall, especially for semantically similar vectors separated by small
numeric deviations. To mitigate this mismatch, we introduce a light-
weight adapter applied to the query at runtime to reshape encoder
7

MICRO 2026, October 31–November 04, 2026, Athens, Greece Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S Rosing, and Mingu Kang
………………PlaneNBLNWL……3D NAND Block
…………NBLNWL……3D NAND BlockEmbeddingsCentroidsCluster Group #1
Cluster Group #2QuerySelect!………………PlaneNBLNWL……
……………NBLNWL……(b) Normal Mapping3D NAND Block
3D NAND BlockAfter 2ndStageAfter 2ndStage
(c) Locality-Aware Mapping(a) Cluster grouping4 sub-blocks reading L3 sub-blocks reading JSub-block
Figure 10: (a) Overview of cluster grouping, (b) normal vector
mapping, and (c) proposed locality-aware mapping.
outputs into a DTS-aligned representation, as shown in Fig. 9. Train-
ing uses a retrieval-aware InfoNCE loss [ 60,81] constructed from
DTS retrieval results rather than random negatives. For a set of
queries{𝑞𝑖}, let𝑝+
𝑖denote the ground-truth positive passage for
query𝑞𝑖, and let𝑝−
𝑖𝑗denote the 𝑗-th negative passage from the
Top-𝐾candcandidate pool retrieved by DTS search. The adapter is
trained on these query-passage pairs by minimizing the InfoNCE
loss (L) shown in Fig. 9. Here, 𝑞′
𝑖denotes the adapter-transformed
query embedding, 𝜏is the temperature parameter, and sim(·,·)
denotes cosine similarity. Unlike conventional contrastive train-
ing [ 23], which uses randomly sampled embeddings as negatives,
our method uses hard negatives obtained from DTS itself. These
DTS-mined negatives are more informative, because they empha-
size near-boundary false positives, enabling the adapter to better
align the query representation with DTS’s dual-bound metric.
Quantitatively, the adapter significantly improves recall by reduc-
ing the average Recall@100 gap to IVF from 3.77%–4.54% to 1.43%–
1.76%, as detailed in Section 6.1. Training is a one-timeofflinecost
during embedding generation and incurs no latency or computation
overhead during retrieval. At inference, D-NOVA applies only a
lightweight query-side adapter while keeping passage embeddings
fixed in 3D NAND. This adaptation adds only 0.5%–1% overhead to
query encoding time and is negligible in end-to-end LLM workloads.
The adapter itself is compact (0.4M parameters for 384- 𝐷embed-
dings) and can be efficiently trained or fine-tuned on standard GPUs,
while inference can be performed on either a GPU or a CPU host
during retrieval, enabling low-cost and flexible deployment. This
HW/SW co-design sustains near-FP32-IVF recall while preserving
D-NOVA’s zero-overhead in-storage execution. This query-side de-
sign preserves the RAG database-update path. New documents can
be added through normal index insertion: encode the document,
quantize it into D-NOVA’s INT4/complement representation, as-
sign it to the nearest centroid, and append it to that centroid’s IVF
list. This does not require retraining the LLM, rewriting existing
passage embeddings, or modifying the storage-side DTS pipeline.
If a large database update changes the query/workload distribution,
the compact adapter may be optionally refreshed offline as part of
index maintenance; ordinary document insertion does not require
this, and the in-storage execution flow remains unchanged.
4.3 Locality-Aware Mapping
To reduce sub-block activations per query, we propose locality-
aware mapping, which co-locates relevant data by reorganizing
IVF-list placement. This is especially important in the re-rankingstage, where a small Top- 𝐾1subset of vectors may otherwise be
scattered across strings, forcing many sub-block activations. The
baseline (Fig. 10(b)) uses a simple, first-fit layout that packs inverted
lists (the vectors assigned to each centroid) into sub-blocks by size,
allocating a new sub-block when capacity is exhausted. This lay-
out is implementation-friendly but ignores centroid geometry and
cross-list query locality. In contrast, the proposed locality-aware
mapping introduces a two-step procedure. First, it builds asimilar-
ity chainover centroids using a greedy nearest-neighbor walk [ 64]:
starting from the largest list (by size), it repeatedly appends the
unused centroid whose centroid vector has the smallest distance
to the last centroid in the chain. As described in Fig. 10(a), this
produces a one-dimensional ordering in which neighboring IVF
lists are geometrically close and likely to be co-accessed. The lists
are then packed into sub-blocks by traversing this chain and ap-
plying the same first-fit rule as in the baseline, naturally grouping
co-accessed lists into the same or nearby sub-blocks (Fig. 10(c)).
With locality-aware mapping, second- and third-stage candidate
accesses concentrate in fewer regions, reducing sensing operations
per query and improving throughput. As quantified in Section 6.5,
this lowers the median activation footprint by about 40%.
5 Experimental Setup
5.1 Benchmarks and Encoder
We evaluate D-NOVA on six retrieval benchmarks representative of
RAG-oriented dense retrieval: NQ (2.68M documents; open-domain
QA) [ 39], HotpotQA (5.23M; multi-hop QA) [ 86], FEVER (5.42M;
fact verification) [ 78], and FiQA (57.6K; financial QA) [ 49] from
BEIR [ 77], wiki_en (1.01M; open-domain QA) from AIR-Bench [ 9],
and MIRACL-English (32.9M; ad-hoc retrieval) from MIRACL [ 88].
All methods use the same text encoders for fair comparison: all-
MiniLM-L6-v2 (MiniLM, 384- 𝐷) [82] and all-mpnet-base-v2 (mp-
net, 768-𝐷) [74]. We use FP32-IVF as the full-precision software
accuracy reference: it uses the same IVF-Flat index, FP32 embed-
dings, and cosine similarity, without DTS quantization or NAND
sensing constraints. This FP32-IVF reference corresponds to the
CPU-IVF baseline whose system configuration is described in Sec-
tion 5.4. For D-NOVA and the DTS variants, we attach our pro-
posed adapter to the encoder outputs and train it onceoffline, as
described in Section 4.2, while keeping the adapter frozen during
retrieval. Unless otherwise noted, all systems use the same IVF-Flat
design with dataset-specific search parameters. We use 𝑁list=1024
for NQ, FiQA, and wiki_en,2048for HotpotQA and FEVER, and
8192for MIRACL-English. We evaluate 𝑁probe={64,128}for NQ,
{32,64}for FiQA and wiki_en, {128,256}for HotpotQA and FEVER,
and{256,512}for MIRACL-English. Across all datasets, we keep
𝐾2=1000coarse candidates and 𝐾1=100final results fixed for
both FP32-IVF and all DTS configurations. Therefore, the recall
results reflect the DTS metric, adapter, and sensing configuration
under the same candidate budgets, rather than differences in IVF
search effort.
5.2 In-House DTS & Adapter Simulator
We develop a cycle-accurate, event-driven simulator to model D-
NOVA’s DTS operation on a realistic 3D NAND array. Table 1
summarizes the simulator configuration. The simulator tracks WL
8

D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound
3D NAND-Optimized Similarity Search with Vector Adaptation MICRO 2026, October 31–November 04, 2026, Athens, Greece
Table 1: Simulator Configuration
Component Configuration
Read Energy 294.9 nJ/page [38]
Read Latency 30𝜇s (SLC), 140𝜇s (QLC) [76]
Circuit Simulation Spectre 19.1, PTM-MG 14nm [89]
String/Cell Config. 128 WLs / QLC within 8.5V window [15]
activations, page-buffer events, and controller-side reductions at
cycle granularity, translating them to latency and energy using mea-
sured NAND timing parameters and per-operation energy models
from prior works [ 38,76], rescaled to our block organization using
an RC-based 3D NAND model [ 69]. This includes the movement of
compact scores and metadata through page buffers and controller
SRAM, local deficit updates in the peripheral logic, and sub-block ac-
tivations determined by the selected IVF lists. To ground the system-
level abstraction in device-level behavior, we perform circuit-level
Monte Carlo simulations that incorporate both cell-level and pe-
ripheral non-idealities, including 𝑉𝑇𝐻mismatch and interconnect
parasitics. Each Spectre netlist instantiates a precharged bitline, a
128-cell NAND string, a sense amplifier, and an output buffer using
PTM-MG 14nm [ 89] models. In the scaled PTM voltage domain,
query-derived WL voltages are generated as 𝑉WL=0.1+(𝑞+𝛼)Δ𝑉 TH
over a 16-level QLC window, while unselected WLs are biased with
𝑉PASS=1.20V. The binary DTS sensing output is determined by
whether the bitline crosses the 0.40 V sense trip point during the
read window. 𝑉THvariations are injected based on QLC drift charac-
teristics from [ 56], modeling the open-block scenario in which 𝑉TH
distributions exhibit significant broadening and shifts. This captures
sensing uncertainty from process variation and retention/drift-like
threshold broadening rather than assuming ideal threshold lev-
els; the same variation model is used throughout the 𝛼sweep in
Sec. 6.4, so the reported sensing accuracy and Recall@100 reflect
perturbed QLC thresholds. To account for non-idealities introduced
by multi-WL enabling, the effective string resistance is scaled with
the number of simultaneously activated cells ( 𝑚), which reduces the
sensing resolution at larger 𝑚. In DTS, the tolerance 𝛼is added to
the query WL voltage used in bound checks: a larger 𝛼broadens the
acceptance region, while smaller values enforce stricter matches.
All experiments sweep 𝛼under noisy 𝑉𝑇𝐻conditions to reflect the
process variation and report accuracy with perturbed 𝑉𝑇𝐻from
limited retention. The simulator also models 𝑚-adaptive grouping
and locality-aware mapping, and incorporates the functionality of
one-time adapter training via an integrated end-to-end training
loop driven by DTS retrieval results.
5.3 D-NOVA Configuration
Table 2: D-NOVA Configuration
Component Configuration
Host Interface PCIe 4.0×4 (7.88 GB/s) [6]
Controller Embedded CPU [46],∼8 MB SRAM
Flash Channels 8 channels (2.0 GB/s/ch, 16 GB/s total) [14]
NAND PKG 8 dies/ch, 12 planes/die, 256 blocks/plane
Block Spec 128 WL, 16K BL, 16 SSL
Page Buffer 2 KB (SLC), 8 KB latch capacity (QLC) [35, 44]D-NOVA is implemented as a self-contained NVMe SSD sub-
system. Table 2 summarizes its overall system configuration. The
storage hierarchy exposes high internal bandwidth through multi-
channel and multi-plane parallelism. In particular, planes within
each die are interconnected via an H-Tree routing network [ 34,
71,84], enabling balanced signal propagation and high inter-plane
bandwidth (≈1 TB/s), which is critical for scalable in-storage par-
allel search. Although DTS relies on binary sensing by utilizing
simple pass/fail comparators, D-NOVA fully preserves compatibil-
ity with standard QLC read/program operations by reusing existing
page-buffer structures. D-NOVA employs a lightweight shared accu-
mulator in the peripheral logic for a group of 256 BLs to aggregate
CL outputs. The accumulator is implemented using digital logic and
synthesized using the ASAP7 7 nm [ 17] standard-cell library with
the OpenROAD flow [ 3], targeting a 1 GHz operating frequency.
The design choice and associated area overhead are discussed in
Section 6.7. The remaining on-die peripheral circuits, such as page
decoders and WL/BL drivers, follow the conventional NAND con-
figuration [ 10], preserving full compatibility with standard NAND
operations (e.g., ISPP/ISPE). Consequently, by using single-cycle
binary sensing instead of multi-level QLC reads, D-NOVA achieves
SLC-like read latency while maintaining QLC density.
5.4 Hardware Baselines and Metrics
We evaluate D-NOVA against both a CPU-based software baseline,
as they represent the software upper bound in accuracy and latency,
and state-of-the-art ISP/NSP accelerators representative of current
practices in large-scale vector retrieval, as described below.
1) CPU–IVF baselineuses FAISS IVF-Flat on a dual-socket 128-
core server [ 4] equipped with 1.5 TB DDR4 DRAM [ 51] and a PCIe
4.0 SSD [ 20]. All embeddings are stored in DRAM, and cosine simi-
larity is computed in FP32 precision, providing the software upper
bound unconstrained by NAND bandwidth or quantization effects.
System-level power is estimated using AMD 𝜇Prof [ 1] for CPU
components, while DRAM energy is modeled using CACTI7 [ 7], as
𝜇Prof does not account for main-memory power.
2) REIS baseline[ 10] performs cluster selection and binary
coarse search inside NAND, followed by INT8 re-ranking in the
controller. We scale its 𝑁nlistand𝑁nprobe parameters to align with
our datasets. Binary search energy and latency for the REIS baseline
are modeled using measured NAND SSD characteristics [ 76], identi-
cal to those used for D-NOVA. The re-ranking stage operates on the
controller’s embedded processor [ 46] with LPDDR4 memory [ 29],
where INT8 dot-product energy and latency are derived from [ 87]
and scaled to 7 nm following the methodology in [75].
3) NSP baselinemodels a near-storage processor similar to
RAGX [ 47]. It stores INT8 embeddings in QLC NAND, which are
fetched via normal NAND read operations. An ASIC accelerator
adjacent to the SSD controller executes the first- and second-stage
INT8 cosine similarity searches, while the third stage performs
only quicksort on the resulting candidates. For fair comparison,
we use identical 𝑁nlistand𝑁nprobe parameters and scaled processor
and DRAM specifications as in REIS. This baseline reflects a near-
data design that reduces host traffic but still requires full INT8
readouts and arithmetic computation outside the NAND array. For
datasets whose indexes fit entirely in DRAM, DRAM-PIM [ 11]
9

MICRO 2026, October 31–November 04, 2026, Athens, Greece Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S Rosing, and Mingu Kang
Figure 11: Recall@25 and Recall@100 for FP32-IVF, DTS111, and DTS242, each with and without the adapter, across six datasets.
could also be considered. However, it assumes a volatile DRAM-
resident index, while D-NOVA targets NAND-resident embeddings
and avoids full-embedding readout. Efficient DRAM-PIM ANN also
often changes the index to IVF-PQ/LUT scoring, introducing a
different accuracy/storage trade-off from our IVF-Flat reference.
For larger RAG corpora, DRAM-PIM must populate or maintain a
DRAM copy from SSD/NAND, whereas D-NOVA directly searches
the NAND-resident index.
All results include NAND reading, on-die reductions, locality-
aware mapping effects, and controller-side Top- 𝐾selection and
metadata handling. Unless otherwise specified, the DTS tolerance
margin is set to 𝛼=2. In our evaluation, GPU-based baselines are not
considered, as our study, consistent with prior work [ 10,12] in the
retrieval stage, indicates that memory and I/O are bandwidth bottle-
necks rather than computational throughput. We report Recall@25
& @100, latency & energy per query, and queries per second (QPS).
6 Evaluation Results
6.1 DTS Retrieval Accuracy Results
For brevity, we denote DTS configurations as DTS 𝑚1𝑚2𝑚3(e.g.,
DTS242 corresponds to 𝑚=( 2,4,2)). Fig. 11 reports Recall@25 (R@25)
and Recall@100 (R@100) for FP32-IVF baseline and DTS variants
under two configurations (DTS111 and DTS242), each evaluated
with and without the adapter. As shown in Fig. 11, DTS closely
tracks the FP32-IVF baseline, with only a small accuracy gap at
the largest𝑁nprobe . The gap is consistently smaller for mpnet than
for MiniLM, indicating that the denser encoder is less sensitive
to the discrete DTS metric. Increasing the sensing group size 𝑚
from DTS111 to DTS242 introduces a modest degradation, while
still maintaining stable accuracy across benchmarks and encoders.
Applying our proposed adapter significantly improves recall not
only for DTS, but also for the baseline IVF, with larger gains ob-
served for DTS as summarized in Table 3. Since the adapter is
trained with DTS-mined hard negatives, it reshapes the query rep-
resentation to separate relevant passages from DTS-confusing false
positives. The same trend holds across both encoders and all six
benchmarks, indicating that retrieval-aware adaptation effectively
compensates for the residual mismatch between the DTS metric
and the FP32 retrieval space. The adapter also reduces the difference
between DTS111 and DTS242, showing that part of the loss from
coarser sensing can be recovered in the embedding space.As an
additional NQ database-expansion stress test, we partition the cor-
pus into an 80% base database and a 20% added-document set. The
adapter is trained only on the base database; the added documentsTable 3: Average R@100 gap across 6 datasets & 2 encoders.
MethodGap to FP32-IVF (%)Adapter Gain (%)
w/o Adapter w/ Adapter
DTS111 3.79 1.45 (↓2.34) +11.48
DTS242 4.48 1.75 (↓2.73) +11.87
are then inserted into the IVF lists, and the frozen base adapter
is reused without retraining. Averaged over randomized database
splits, MiniLM/mpnet, and 𝑁probe∈{64,128}, the frozen adapter
reduces the expanded-DB DTS111-IVF R@100 gap from 4.95% to
2.93%, and the DTS242-IVF gap from 6.40% to 3.04%. This shows
that newly added documents can be searched using the existing
adapter while preserving the same storage-side DTS flow.
Overall, these results show that DTS preserves strong retrieval
quality across diverse tasks, domains, and encoder choices, while
retrieval-aware adaptation materially reduces the accuracy degra-
dation without altering the storage-side retrieval flow.
6.2𝑚-Adaptive Parallelism vs. Recall Accuracy
Fig. 12(a)–(d) show heatmaps sweeping 𝑚2along the x-axis (coarse
Top-𝐾 2stage) and tied𝑚 1=𝑚3along the y-axis (centroid selection
and re-ranking). Each cell reports the Recall@100 difference relative
to IVF on NQ. The trends are consistent across both encoders and
𝑁nprobe . Keeping the precision-critical stages narrow ( 𝑚1=𝑚3∈
{1,2,4}) maintains accuracy close to IVF while allowing the second
stage to run with a larger 𝑚2. With MiniLM, the gap stays within
about 3.5% at 𝑁nprobe=64and within about2% at 𝑁nprobe=128; with
mpnet, it is within about1% for both settings. However, once 𝑚1,𝑚3
approach 16, the degradation becomes substantial. For MiniLM in
particular, the Recall@100 drop exceeds 10%, reflecting that overly
wide sensing in the precision-critical stages severely harms fidelity.
Conversely, results show that the second stage is comparatively
insensitive: once 𝑚1and𝑚3are modest, increasing 𝑚2to8has little
recall degradation, enabling aggressive mid-stage parallelism. These
results support an asymmetric configuration—small 𝑚1and𝑚3for
fidelity, large 𝑚2in the second stage for throughput—leveraging
the array’s inherent parallelism without sacrificing recall.
6.3 Benefit of DTS vs. UBS+LBS
Fig. 13(a) and (b) show the advantage of DTS under DTS141 and
DTS441, respectively, where DTS applies a complement-based tighter
bound than the lenient UBS + LBS scheme as discussed in Section 3.2.
When𝑚= 1, the two schemes become equivalent (DTS = UBS+LBS),
10

D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound
3D NAND-Optimized Similarity Search with Vector Adaptation MICRO 2026, October 31–November 04, 2026, Athens, Greece
(a)MiniLM, nprobe = 64(c) mpnet, nprobe = 64(b) MiniLM, nprobe = 128(d) mpnet, nprobe = 128m1and m3m2Difference (%)m1and m3m2Difference (%)m1and m3m2Difference (%)m1and m3m2Difference (%)
Figure 12: (a)-(d) Recall@100 degradation heatmap across different encoders for the
NQ under𝑚-adaptive configuration and different stage settings (𝑚 1,𝑚2,𝑚3).
0.40%0.50%0.00%0.00%0.80%0.60%0.20%0.00%
4.81%3.42%1.51%1.04%
4.40%3.40%1.57%0.99%0.00%1.00%2.00%3.00%4.00%5.00%64128641281282561282566412864128128256128256NQHotpotQANQHotpotQAMiniLMmpnetMiniLMmpnetMiniLMmpnetMiniLMmpnetEncoderDatasetRecall@100 Difference (DTS –(UBS+LBS)) (a)(b)nprobeFigure 13: Benefit of (a) DTS141, and
(b) DTS441 vs. (UBS + LBS).
QLC Noise Region
Recall@100nprobe100806040200Sensing Accuracy (%)1.00.90.80.70.6Recall@100(DTS242, ⍺= 2, nprobe=128)0.700.750.800.850.900.95
32641283264128DTS w/o AdapterDTS w/ AdapterNQ, MiniLM, nlist= 1024, m = (1,1,1)⍺= 2⍺= 3
Noise Std (σmultiplier)0.10          0.15           0.20           0.25       0.30   0.35(a)(b)MiniLMw/o AdapterMiniLMw/Adaptermpnetw/o Adaptermpnetw/AdapterSensing Acc. (⍺= 1)Sensing Acc. (⍺= 2)Sensing Acc. (⍺= 3)
nprobe
Figure 14: (a) NAND sensing accuracy in QLC and Recall@100
under noise on NQ, and (b) Recall@100 at𝛼=2and3.
1.68x1.34x0.84x0.74x0.64x1.67x1.25x0.71x0.50x2ndStage w/o Locality-Aware2ndStage w/ Locality-Aware3rdStage w/ Locality-Aware3rdStage w/o Locality-Aware1.00.80.60.40.20.0102030405060Sub-blocks access per queryEmpirical Cumulative Distribution Function(ECDF)
Empirical Cumulative Distribution Function(ECDF)1.00.80.60.40.20.020406080100120Sub-blocks access per queryNQ, mpnet,nlist= 1024, nprobe =64HotpotQA,mpnet,nlist= 2048, nprobe =128Low QPSHigh QPSNRM_QPS = 1xLow QPSHigh QPSNRM_QPS = 1x0.63x
Figure 15: ECDF and normalized QPS of sub-block accesses
per query for normal vs. locality-aware mapping.
which explains why using 𝑚= 1in the first and third stages re-
sults in negligible differences. However, increasing only 𝑚1to 4
immediately enlarges the recall gap between the two schemes by
more than 4%, and this gap continues to grow as 𝑚increases. These
results demonstrate that our proposed DTS, which implements a
complement-based dual-bound sensing scheme, performs a more
precise search than the lenient UBS+LBS approach.
6.4 Noise-Aware Circuit and System Validations
Fig. 14(a) illustrates the binary sensing accuracy (i.e., UBS-like
bound checks) under varying 𝑉𝑇𝐻noise. We sweep the standard
deviation (𝜎) of Gaussian 𝑉𝑇𝐻noise and perform 1,000 Monte Carlo
trials per𝜎, covering a wide QLC noise regime (0 .1≤𝜎≤ 0.35) cen-
tered around 𝜎≈0.2, which corresponds to high-RBER conditions
(∼10−2) observed in state-of-the-art 3D QLC NAND [ 56]. In the crit-
ical QLC region, 𝑉𝑇𝐻drift causes the sensing accuracy for 𝛼=1to
collapse below 10%, rendering DTS ineffective. In contrast, margins
of𝛼≥ 2successfully compensate for these variations, restoring
sensing reliability across the targeted noise range. Fig. 14(b) sum-
marizes the impact of the DTS margin 𝛼on Recall@100 for NQ
with MiniLM under DTS111. We treat 𝛼=2as the nominal operating
point. Without the adapter, 𝛼=3stays within approximately0 .1%
of the reference accuracy. With the adapter, overall Recall@100
improves, and tighter bounds become more favorable, as the em-
bedding is adapted to the DTS sensing metric. As a result, 𝛼=2
offers a robust trade-off: the bound is tight enough to suppressfalse matches while remaining tolerant to realistic threshold varia-
tion. Although a larger margin ( 𝛼=3) remains accurate, it begins
to admit additional false positives, particularly at smaller 𝑁nprobe .
Therefore, we use 𝛼=2as the default setting for D-NOVA. This
sweep shows that DTS is hardened against QLC retention/drift-
like threshold broadening through the tolerance margin 𝛼, which
acts as a voltage-domain guardband during bound checking, and
provides a robustness/selectivity guideline for margin selection
without admitting excessive false positives.
6.5 Benefit of Locality-Aware Mapping
Building on Section 4.3, Fig. 15 illustrates how the mapping changes
the distribution of sub-block activations per query for the second
and third stage on NQ and HotpotQA with the mpnet. Each curve
is an ECDF over queries, so a leftward shift means that a typical
query touches fewer sub-blocks and requires fewer plane activa-
tions, directly implying higher QPS. On NQ, locality-aware map-
ping reduces the median footprint by ∼38 %in both the second
and third stages, so the same IVF index is retrieved from a much
tighter subset of the array. On HotpotQA, the median number of
sub-block accesses reduces by ∼44 %in the second and ∼42 %in
the third stage. Under incremental database updates, this locality
benefit is still preserved. Even if an additional 25% of the database
is appended using naive placement, the median sub-block-access
reduction remains about 30.4% on NQ and 33.6–35.2% on HotpotQA.
Therefore, locality-aware mapping confines the second- and third-
stage accesses to fewer sub-blocks and planes, reducing the number
of activations per query. Importantly, DTS performs fully paral-
lel sensing within a plane, so concentrating vectors into fewer
sub-blocks does not increase per-access latency. While repeated
accesses may introduce temporal contention, cluster placement
across planes distributes accesses and mitigates such effects.
6.6 E2E Latency, Energy, and Throughput
We evaluate three D-NOVA variants with DTS configurations: D-
NOVA (DTS242), D-NOVA-F (DTS282), and D-NOVA-XF (DTS484).
As shown in Section 6.2, DTS484 maintains Recall@100 within a
few percent of the IVF baseline, so we include D-NOVA-XF as an
aggressive throughput-oriented design while preserving accuracy.
On the other hand, D-NOVA is the conservative variant, achieving
high accuracy despite low parallelism.
Latency & Energy Improvement:Fig. 16(a) and (b) summarize
end-to-end latency and per-query energy, normalized to the CPU
baseline. Across datasets and encoders, all D-NOVA variants deliver
the lowest latency and energy. Even the conservative D-NOVA con-
figuration reduces latency by30 ×and energy by15–34 ×relative
to NSP, achieves33–41 ×lower latency and58–71 ×lower energy
11

MICRO 2026, October 31–November 04, 2026, Athens, Greece Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S Rosing, and Mingu Kang
0.0000.0020.0040.0060.0080.010
all-MiniLM-L6-v2all-mpnet-base-v2all-MiniLM-L6-v2all-mpnet-base-v2BEIR/NQHotpotQANormalized Latency1.000.761.000.791.000.771.000.820.0890.0300.0230.0150.0130.0890.0270.0210.0890.0890.0260.0210.0130.0120.0180.0240.0000.0050.0100.0150.020
all-MiniLM-L6-v2all-mpnet-base-v2all-MiniLM-L6-v2all-mpnet-base-v2BEIR/NQHotpotQANormalized Energy1.001.001.001.000.250.250.490.490.0140.0140.0140.0180.0150.0150.0150.0180.0160.0160.0160.0180.0170.0170.0170.017Total Latency or EnergyStage 1Stage 2Stage 3CPUNSP BaselineD-NOVAD-NOVA-FD-NOVA-XFREIS
MiniLMmpnetMiniLMmpnetNQHotpotQA(a)MiniLMmpnetMiniLMmpnetNQHotpotQA* D-NOVA: (m1, m2, m3) = (2, 4, 2), D-NOVA-F: (m1, m2, m3) = (2, 8, 2), D-NOVA-XF: (m1, m2, m3) = (4, 8, 4)  (b)
Figure 16: Normalized (a) latency and (b) energy breakdown of D-NOVA compared with CPU, NSP, and REIS baselines.
1111124814819611111125745741,1481111125575571,1141101126796791,359
01101001,000
all-MiniLM-L6-v2all-mpnet-base-v2all-MiniLM-L6-v2all-mpnet-base-v2BEIR/NQHotpotQANormalized ThroughputCPUNSP BaselineD-NOVAD-NOVA-FD-NOVA-XFREIS* D-NOVA: (m1, m2, m3) = (2, 4, 2), D-NOVA-F: (m1, m2, m3) = (2, 8, 2), D-NOVA-XF: (m1, m2, m3) = (4, 8, 4)  
MiniLMmpnetMiniLMmpnetNQHotpotQA
Figure 17: Normalized throughput of D-NOVA vs. baselines.
than the CPU software baseline, and further reduces latency by
3.01–3.79×(D-NOVA),3 .80–4.99×(D-NOVA-F), and5 .98–7.53×
(D-NOVA-XF) with additional energy savings of6–27%over REIS.
These results directly quantify the benefit of avoiding off-array final
re-ranking. D-NOVA keeps all embeddings in flash, performs both
coarse and fine scoring through in-array DTS-based binary sensing,
and transfers only compact scores/metadata to the controller, avoid-
ing full multi-level QLC candidate readout and controller-side INT8
re-ranking. In contrast, REIS is the closest baseline for this case:
it performs the earlier NAND-side binary search stages, but still
moves tens of thousands of QLC-resident candidate embeddings off-
array to the SSD controller for final INT8 re-ranking. NSP represents
a more general near-storage design that reads INT8 embeddings
from flash before external similarity computation. Eliminating these
off-array candidate accesses and processor computations removes
the dominant sources of latency and energy in prior designs. Be-
cause all D-NOVA variants activate nearly the same set of WLs for
a given query, varying 𝑚primarily reshapes the sensing schedule
(e.g., sequential vs. parallel) without changing the total number
of WL activations. As WL toggling dominates overall energy con-
sumption, the total per-query energy remains largely similar across
different𝑚configurations. Beyond eliminating re-ranking-time
candidate readout, D-NOVA further reduces energy by minimiz-
ing sub-block activations through locality-aware mapping. Overall,
although per-sensing energy is similar, D-NOVA achieves much
faster end-to-end latency and far lower total energy by requiring
fewer sensing cycles and eliminating almost all off-array DRAM
and compute overhead.
Throughput Improvement:Fig. 17 shows the corresponding
steady-state throughput (QPS) under large-batch operation, nor-
malized to the CPU baseline. With the pipeline saturated, D-NOVA
and D-NOVA-F sustain roughly481–679 ×of the CPU QPS and
about42.7–67.9×of the NSP baseline. Relative to REIS, D-NOVA
and D-NOVA-F improve throughput by about4 .3–6.1×, while D-
NOVA-XF reaches8 .6–12.1×higher QPS. In this configuration,
D-NOVA and D-NOVA-F exhibit similar QPS because the larger
𝑚2in D-NOVA-F shortens the coarse second stage, so the third
stage (with the same 𝑚3in both) becomes the dominant bottleneck.This trade-off can be re-tuned per deployment by shrinking 𝐾1or
adjusting𝑁nprobe and𝑚2to shift more work into the highly parallel
second stage while meeting the target accuracy.
6.7 Overhead of D-NOVA System
DTS is evaluated by page-buffer comparators that reuse existing SA
front-ends in binary pass/fail mode. To support delta-based accumu-
lation, D-NOVA introduces a lightweight shared accumulator in the
peripheral logic. Our synthesized design shows that sharing one ac-
cumulator across 256 BLs (64 accumulators/plane) results in a total
area of 0.023 mm2per plane, including multiplexer, corresponding
to <1.5% of the 1.476 mm2plane area [ 26]. This overhead is amor-
tized within existing CMOS-under-Array (CuA) regions [ 16,53,57],
resulting in negligible die-level impact. The accumulation is tightly
pipelined with DTS: while SL-level sensing is performed, CL out-
puts from previously sensed WLs are forwarded to the accumulator,
allowing score updates to proceed in parallel with ongoing sensing.
The accumulation latency (256 BLs @ 1 GHz ≈0.25𝜇s) is negligible
compared to sensing ( ≈30𝜇s) and is fully hidden by pipelined
execution. The BL grouping (e.g., 256 per accumulator) is chosen to
balance accumulator fan-in and multiplexer complexity, avoiding
long critical paths and excessive routing overhead in the peripheral
logic. Detailed configurations are provided in Section 5.3.
In addition, D-NOVA leverages multi-WL activation to increase
intra-string parallelism. To enable 𝑚-adaptive sensing, D-NOVA
leverages existing per-WL voltage control, where each WL already
supports𝑉readand𝑉passduring standard reads [ 24,42] along with
existing multi-WL activation modes [ 43,54,58,65,66]. We reuse the
same WL drivers for multi-WL activation. Enabling multiple WLs
can increase the effective on-resistance of the string, potentially
leading to increased latency. Using our in-house DTS circuit simula-
tor, we compare the BL discharge time against the reference 𝑡DISCH
reported in [ 59] and observe that even with 𝑚= 8, the sensing
latency differs by only 5.6%. Since 𝑡DISCH constitutes only a small
portion of the overall read latency budget, this difference translates
to negligible impact on end-to-end read performance, indicating
that the practical overhead of multi-WL activation in our operating
regime is minimal. Prior work [ 58] further confirms that activating
up to 48 WLs is feasible within a read timing margin of 24 𝜇s, based
on validation across multiple dies of real silicon.
7 Conclusion
D-NOVA demonstrates the feasibility of executing an entire re-
trieval pipeline fully inside the storage device, eliminating the
memory bandwidth bottleneck at the system level. We introduce a
12

D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound
3D NAND-Optimized Similarity Search with Vector Adaptation MICRO 2026, October 31–November 04, 2026, Athens, Greece
novel Dual-Bound Tight Similarity Sensing (DTS) mechanism inte-
grated into 3D NAND flash and a lightweight contrastive adapter,
trained onceoffline, that aligns embeddings to this discrete metric,
thereby preserving high recall accuracy. This hardware–software
co-designed in-storage architecture achieves4 .3–12.1×higher through-
put and1.1–1.26×lower energy than the state-of-the-art in-storage
accelerator [ 10], with feasibility validated through detailed circuit-
level noise simulations and overhead analysis. These results estab-
lish a new direction for efficient and scalable RAG systems.
Acknowledgments
This work was supported in part by the Center for Processing with
Intelligent Storage and Memory (PRISM) under Semiconductor
Research Corporation (SRC) grant 2023-JU-3135 and by CoCoSys,
both centers in JUMP 2.0, an SRC program sponsored by DARPA.
This work was also supported in part by NSF grants #2112665,
#2211386, #2052809, and #2112167.
References
[1]Advanced Micro Devices, Inc. 2025. AMD uProf: Performance Analysis Tool.
https://www.amd.com/en/developer/uprof.html.
[2]Mohiuddin Ahmed, Raihan Seraj, and Syed Mohammed Shamsul Islam. 2020.
The k-means algorithm: A comprehensive survey and performance evaluation.
Electronics9, 8 (2020), 1295.
[3]Toluwalope Ajayi et al .2019. OpenROAD: Toward a Self-Driving, Open-Source
Digital Layout Implementation Tool Chain. InDAC.
[4]AMD. 2023.AMD EPYC™9554. https://www.amd.com/en/products/processors/
server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9554.html Ac-
cessed: 2025-11-16.
[5]AMD Adaptive & Embedded Computing Group and Samsung. 2023.
SmartSSD®Computational Storage Drive: Product Brief. https:
//www.xilinx.com/publications/product-briefs/xilinx-smartssd-computational-
storage-drive-product-brief.pdf. Accessed Nov. 2025.
[6]Woorham Bae, Sung-Yong Cho, and Deog-Kyoon Jeong. 2021. A 1.93-pJ/Bit PCI
Express Gen4 PHY Transmitter with On-Chip Supply Regulators in 28 nm CMOS.
Electronics(2021). https://api.semanticscholar.org/CorpusID:234325987
[7]Rajeev Balasubramonian, Andrew B. Kahng, Naveen Muralimanohar, Ali Shafiee,
and Vaishnav Srinivas. 2017. CACTI 7: New Tools for Interconnect Exploration
in Innovative Off-Chip Memories.ACM Transactions on Architecture and Code
Optimization14, 2 (2017). doi:10.1145/3092639 Accessed: 2025-11-16.
[8]Yu Cai, Saugata Ghose, Erich F. Haratsch, Yixin Luo, and Onur Mutlu. 2017. Error
Characterization, Mitigation, and Recovery in Flash-Memory-Based Solid-State
Drives.Proc. IEEE105, 9 (2017), 1666–1704. doi:10.1109/JPROC.2017.2713127
[9]Jianlyu Chen, Nan Wang, Chaofan Li, Bo Wang, Shitao Xiao, Han Xiao, Hao
Liao, Defu Lian, and Zheng Liu. 2025. AIR-Bench: Automated Heterogeneous
Information Retrieval Benchmark. InProceedings of the 63rd Annual Meeting of
the Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang
Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.).
Association for Computational Linguistics, Vienna, Austria, 19991–20022. doi:10.
18653/v1/2025.acl-long.982
[10] Kangqi Chen, Rakesh Nadig, Manos Frouzakis, Nika Mansouri Ghiasi, Yu Liang,
Haiyu Mao, Jisung Park, Mohammad Sadrosadati, and Onur Mutlu. 2025. REIS:
A High-Performance and Energy-Efficient Retrieval System with In-Storage Pro-
cessing. InProceedings of the 52nd Annual International Symposium on Computer
Architecture (ISCA ’25). Association for Computing Machinery, New York, NY,
USA, 1171–1192. doi:10.1145/3695053.3731116
[11] Mingkai Chen, Tianhua Han, Cheng Liu, Shengwen Liang, Kuai Yu, Lei Dai,
Ziming Yuan, Ying Wang, Lei Zhang, Huawei Li, and Xiaowei Li. 2025. DRIM-
ANN: An Approximate Nearest Neighbor Search Engine based on Commercial
DRAM-PIMs. InProceedings of the International Conference for High Performance
Computing, Networking, Storage and Analysis (SC ’25). Association for Computing
Machinery, New York, NY, USA, 820–836. doi:10.1145/3712285.3759801
[12] Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li,
Mao Yang, and Jingdong Wang. 2021. SPANN: Highly-efficient Billion-scale
Approximate Nearest Neighbor Search. In35th Conference on Neural Information
Processing Systems (NeurIPS 2021).
[13] Sitian Chen, Amelie Chi Zhou, Yucheng Shi, Yusen Li, and Xin Yao. 2024. Me-
mANNS: Enhancing Billion-Scale ANNS Efficiency with Practical PIM Hardware.
arXiv:2410.23805. https://arxiv.org/abs/2410.23805[14] Jiho Cho, D. Chris Kang, Jongyeol Park, Sang-Wan Nam, Jung-Ho Song, Bong-Kil
Jung, Jaedoeg Lyu, Hogil Lee, Won-Tae Kim, Hongsoo Jeon, Sunghoon Kim, In-
Mo Kim, Jae-Ick Son, Kyoungtae Kang, Sang-Won Shim, JongChul Park, Eungsuk
Lee, Kyung-Min Kang, Sang-Won Park, Jaeyun Lee, Seung Hyun Moon, Pansuk
Kwak, ByungHoon Jeong, Cheon An Lee, Kisung Kim, Junyoung Ko, Tae-Hong
Kwon, Junha Lee, Yohan Lee, Chaehoon Kim, Myeong-Woo Lee, Jeong-yun
Yun, HoJun Lee, Yonghyuk Choi, Sanggi Hong, JongHoon Park, Yoonsung Shin,
Hojoon Kim, Hansol Kim, Chiweon Yoon, Dae Seok Byeon, Seungjae Lee, Jin-
Yub Lee, and Jaihyuk Song. 2021. 30.3 A 512Gb 3b/Cell 7th -Generation 3D-
NAND Flash Memory with 184MB/s Write Throughput and 2.0Gb/s Interface. In
2021 IEEE International Solid-State Circuits Conference (ISSCC), Vol. 64. 426–428.
doi:10.1109/ISSCC42613.2021.9366054
[15] Hwanheechan Choi, Hyungjun Jo, Sangmin Ahn, Insang Han, and Hyungcheol
Shin. 2026. Machine learning-based prediction of the impact of random grain
boundary Z-interference on Vt distribution in 3-D NAND flash memory.Journal
of Computational Electronics25, 1 (2026), 16.
[16] Myungjun Chun, Jaeyong Lee, Sanggu Lee, Myungsuk Kim, and Jihong Kim.
2022. PiF: in-flash acceleration for data-intensive applications. InProceedings of
the 14th ACM Workshop on Hot Topics in Storage and File Systems(Virtual Event)
(HotStorage ’22). Association for Computing Machinery, New York, NY, USA,
106–112. doi:10.1145/3538643.3539742
[17] Lawrence T Clark, Vinay Vashishtha, Lucian Shifren, Aditya Gujja, Saurabh
Sinha, Brian Cline, Chandarasekaran Ramamurthy, and Greg Yeric. 2016. ASAP7:
A 7-nm finFET predictive process design kit.Microelectronics Journal53 (July
2016), 105–115.
[18] Cohere. 2023. wikipedia-2023-11-embed-multilingual-v3. https://huggingface.
co/datasets/Cohere/wikipedia-2023-11-embed-multilingual-v3. Hugging Face
Datasets.
[19] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025.
THE FAISS LIBRARY.IEEE Transactions on Big Data(2025), 1–17. doi:10.1109/
TBDATA.2025.3618474
[20] Samsung Electronics. 2021.PM9A3 NVMe PCIe SSD. https://semiconductor.
samsung.com/ssd/datacenter-ssd/pm9a3/ Accessed: 2025-11-16.
[21] Keming Fan, Ashkan Moradifirouzabadi, Xiangjin Wu, Zheyu Li, Flavio Ponzina,
Anton Persson, Eric Pop, Tajana Rosing, and Mingu Kang. 2024. SpecPCM: A
Low-Power PCM-Based In-Memory Computing Accelerator for Full-Stack Mass
Spectrometry Analysis.IEEE Journal on Exploratory Solid-State Computational
Devices and Circuits10 (2024), 161–169. doi:10.1109/JXCDC.2024.3498837
[22] Jingqi Feng, Yukai Huang, Rui Zhang, Sicheng Liang, Ming Yan, and Jie Wu.
2025. WindServe: Efficient Phase-Disaggregated LLM Serving with Stream-based
Dynamic Scheduling. InProceedings of the 52nd Annual International Symposium
on Computer Architecture (ISCA ’25). Association for Computing Machinery, New
York, NY, USA, 1283–1295. doi:10.1145/3695053.3730999
[23] Hakim Hafidi, Mounir Ghogho, Philippe Ciblat, and Ananthram Swami. 2022.
Negative sampling strategies for contrastive self-supervised learning of graph
representations.Signal Processing190 (2022), 108310.
[24] Tsutomu Higuchi, Takuyo Kodama, Koji Kato, Ryo Fukuda, Naoya Tokiwa,
Mitsuhiro Abe, Teruo Takagiwa, Yuki Shimizu, Junji Musha, Katsuaki Sakurai,
Jumpei Sato, Tetsuaki Utsumi, Kazuhide Yoneya, Yasuhiro Suematsu, Toshifumi
Hashimoto, Takeshi Hioka, Kosuke Yanagidaira, Masatsugu Kojima, Junya Mat-
suno, Kei Shiraishi, Kensuke Yamamoto, Shintaro Hayashi, Tomoharu Hashiguchi,
Kazuko Inuzuka, Akio Sugahara, Mitsuaki Honma, Keiji Tsunoda, Kazumasa Ya-
mamoto, Takahiro Sugimoto, Tomofumi Fujimura, Mizuki Kaneko, Hiroki Date,
Osamu Kobayashi, Takatoshi Minamoto, Ryoichi Tachibana, Itaru Yamaguchi,
Juan Lee, Venky Ramachandra, Srinivas Rajendra, Tianyu Tang, Siddhesh Darne,
Jiwang Lee, Jason Li, Toru Miwa, Ryuji Yamashita, Hiroshi Sugawara, Naoki
Ookuma, Masahiro Kano, Hiroyuki Mizukoshi, Yuki Kuniyoshi, Mitsuyuki Watan-
abe, Kei Akiyama, Hirotoshi Mori, Akira Arimizu, Yoshito Katano, Masakazu
Ehama, Hiroshi Maejima, Koji Hosono, and Masahiro Yoshihara. 2021. 30.4
A 1Tb 3b/Cell 3D-Flash Memory in a 170+ Word-Line-Layer Technology. In
2021 IEEE International Solid-State Circuits Conference (ISSCC), Vol. 64. 428–430.
doi:10.1109/ISSCC42613.2021.9366003
[25] Charles AR Hoare. 1962. Quicksort.The computer journal5, 1 (1962), 10–16.
[26] Po-Kai Hsu, Weihong Xu, Tajana Rosing, and Shimeng Yu. 2023. An in-storage
processing architecture with 3d nand heterogeneous integration for spectra open
modification search. InProceedings of the International Symposium on Memory
Systems. 1–7.
[27] Zhengding Hu, Vibha Murthy, Zaifeng Pan, Wanlu Li, Xiaoyi Fang, Yufei Ding,
and Yuke Wang. 2025. HedraRAG: Co-Optimizing Generation and Retrieval
for Heterogeneous RAG Workflows. InProceedings of the ACM SIGOPS 31st
Symposium on Operating Systems Principles. 623–638.
[28] Bongjoon Hyun, Taehun Kim, Dongjae Lee, and Minsoo Rhu. 2024. Pathfinding
Future PIM Architectures by Demystifying a Commercial PIM Technology. In
2024 IEEE International Symposium on High-Performance Computer Architecture
(HPCA). 263–279. doi:10.1109/HPCA57654.2024.00029
[29] Micron Technology Inc. 2023.LPDDR4X/LPDDR4 SDRAM MT53E768M64D4,
MT53E1536M64D8, MT53E768M32D2, MT53E1536M32D4 Data Sheet. https://www.
13

MICRO 2026, October 31–November 04, 2026, Athens, Greece Chang Eun Song, Sumukh Pinge, Tianqi Zhang, Sung Eun Kim, Tajana S Rosing, and Mingu Kang
mouser.com/datasheet/2/671/z4bm_embedded_lpddr4x_lpddr4-3193428.pdf Rev.
D, March 2023.
[30] Yeonwoo Jeong, Hyunji Cho, Kyuri Park, Youngjae Kim, and Sungyong Park. 2025.
CALL: Context-Aware Low-Latency Retrieval in Disk-Based Vector Databases.
arXiv preprint arXiv:2509.18670(2025).
[31] Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir Yazdan-
bakhsh, and Vidushi Dadu. 2025. Rago: Systematic performance optimization
for retrieval-augmented generation serving. InProceedings of the 52nd Annual
International Symposium on Computer Architecture. 974–989.
[32] Chao Jin, Zili Zhang, Xuanlin Jiang, Fangyue Liu, Shufan Liu, Xuanzhe Liu, and
Xin Jin. 2024. Ragcache: Efficient knowledge caching for retrieval-augmented
generation.ACM Transactions on Computer Systems(2024).
[33] Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity
search with GPUs.IEEE Transactions on Big Data7, 3 (2019), 535–547.
[34] Norman P Jouppi, Doe Hyun Yoon, Matthew Ashcraft, Mark Gottscho, Thomas B
Jablin, George Kurian, James Laudon, Sheng Li, Peter Ma, Xiaoyu Ma, et al .2021.
Ten lessons from three generations shaped google’s tpuv4i: Industrial product. In
2021 ACM/IEEE 48th Annual International Symposium on Computer Architecture
(ISCA). IEEE, 1–14.
[35] Ali Khakifirooz, Sriram Balasubrahmanyam, Richard Fastow, Kristopher H
Gaewsky, Chang Wan Ha, Rezaul Haque, Owen W Jungroth, Steven Law, Alias-
gar S Madraswala, Binh Ngo, et al .2021. 30.2 a 1tb 4b/cell 144-tier floating-gate
3d-nand flash memory with 40mb/s program throughput and 13.8 gb/mm 2 bit
density. In2021 IEEE International Solid-State Circuits Conference (ISSCC), Vol. 64.
IEEE, 424–426.
[36] Hyun-Jin Kim, Jeong-Don Lim, Jang-Woo Lee, Dae-Hoon Na, Joon-Ho Shin,
Chae-Hoon Kim, Seung-Woo Yu, Ji-Yeon Shin, Seon-Kyoo Lee, Devraj Rajagopal,
et al.2015. 7.6 1gb/s 2tb nand flash multi-chip package with frequency-boosting
interface chip. In2015 IEEE International Solid-State Circuits Conference-(ISSCC)
Digest of Technical Papers. IEEE, 1–3.
[37] Ji-Hoon Kim, Yeo-Reum Park, Jaeyoung Do, Soo-Young Ji, and Joo-Young Kim.
2022. Accelerating large-scale graph-based nearest neighbor search on a compu-
tational storage platform.IEEE Trans. Comput.72, 1 (2022), 278–290.
[38] Kana Kudo, Yuta Aiba, Kazuma Hasegawa, Xu Li, Yuichi Sano, and Tomoya
Sanuki. 2025. Energy-Efficient In-Memory Computing using 3D Flash Memory
with Sequential Multi-Block Activation and Current Control Cell (CC cell). In
2025 IEEE International Memory Workshop (IMW). 1–4. doi:10.1109/IMW61990.
2025.11026979
[39] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur
Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee,
Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M.
Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A
Benchmark for Question Answering Research.Transactions of the Association for
Computational Linguistics7 (2019), 452–466. doi:10.1162/tacl_a_00276
[40] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
memory management for large language model serving with pagedattention. In
Proceedings of the 29th symposium on operating systems principles. 611–626.
[41] Joo Hwan Lee, Hui Zhang, Veronica Lagrange, Praveen Krishnamoorthy, Xi-
aodong Zhao, and Yang Seok Ki. 2020. SmartSSD: FPGA Accelerated Near-Storage
Data Analytics on SSD.IEEE Computer Architecture Letters19, 2 (2020), 110–113.
doi:10.1109/LCA.2020.3009347
[42] Kyungmin Lee, Gunwook Yoon, Seung Jae Baik, and Myounggon Kang. 2026.
Low-Power Stack-Level Programming Enabled by Optimized Dummy Word Line
Voltage in 3-D NAND Flash Memory.IEEE Journal of the Electron Devices Society
14 (2026), 102–106. doi:10.1109/JEDS.2026.3659350
[43] Peter Wung Lee. 2015. NAND array hierarchical bit-line structures for
multiple word-line and all-bit-line simultaneous erase, erase-verify, program,
program-verify, and read operations. https://patents.google.com/patent/
WO2015013689A2/en
[44] Nancy Leong, Sachit Chandra, and Hounien Chen. 2008. Random cache read
using a double memory. US Patent 7,423,915.
[45] Yinan Li, Bailu Ding, Ziyun Wei, Lukas M Maas, Momin Al-Ghosien, Spyros
Blanas, Nicolas Bruno, Carlo Curino, Matteo Interlandi, Craig Peeper, et al .2025.
Scaling GPU-Accelerated Databases beyond GPU Memory Size.Proceedings of
the VLDB Endowment18, 11 (2025), 4518–4531.
[46] Arm Ltd. 2016. Cortex-R8. https://www.arm.com/products/silicon-ip-cpu/cortex-
r/cortex-r8. Accessed: 2025-11-16.
[47] Rohan Mahapatra, Harsha Santhanam, Christopher Priebe, Hanyang Xu, and
Hadi Esmaeilzadeh. 2025. In-Storage Acceleration of Retrieval Augmented Gen-
eration as a Service. InProceedings of the 52nd Annual International Symposium
on Computer Architecture (ISCA ’25). Association for Computing Machinery, New
York, NY, USA, 450–466. doi:10.1145/3695053.3731032
[48] Hosam M Mahmoud, Reza Modarres, and Robert T Smythe. 1995. Analysis of
quickselect: An algorithm for order statistics.RAIRO-Theoretical Informatics and
Applications29, 4 (1995), 255–276.[49] Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott,
Manel Zarrouk, and Alexandra Balahur. 2018. WWW’18 Open Challenge: Finan-
cial Opinion Mining and Question Answering. InCompanion Proceedings of the
The Web Conference 2018. Lyon, France, 1941–1942. doi:10.1145/3184558.3192301
[50] Conrado Martínez and Salvador Roura. 2001. Optimal sampling strategies in
quicksort and quickselect.SIAM J. Comput.31, 3 (2001), 683–705.
[51] Inc. Micron Technology. 2025.DDR4 SDRAM. https://www.micron.com/products/
memory/dram-components/ddr4-sdram Accessed: 2025-11-16.
[52] Daehoon Na, Jang-woo Lee, Seon-Kyoo Lee, Hwasuk Cho, Junha Lee, Manjae
Yang, Eunjin Song, Anil Kavala, Tongsung Kim, Dong-Su Jang, et al .2021. A
1.8-Gb/s/pin 16-Tb NAND flash memory multi-chip package with F-chip for
high-performance and high-capacity storage.IEEE Journal of Solid-State Circuits
56, 4 (2021), 1129–1140.
[53] Seock-Hwan Noh, Hoyeon Lee, Junkyum Kim, Junsu Im, Jay H Park, Sungjin
Lee, Sam H Noh, Yeseong Kim, and Jaeha Kung. 2025. Flexible In-NAND Cryp-
tographic Processing for Secure Flash Storage.arXiv preprint arXiv:2508.03866
(2025).
[54] Yoshiaki Ogura et al .2003. Semiconductor memory device and method for
selecting multiple word lines. https://patents.google.com/patent/JP2003222422A
Laid-open patent application.
[55] Hiroyuki Ootomo, Akira Naruse, Corey Nolet, Ray Wang, Tamas Feher, and Yong
Wang. 2024. Cagra: Highly parallel graph construction and approximate nearest
neighbor search for gpus. In2024 IEEE 40th International Conference on Data
Engineering (ICDE). IEEE, 4236–4247.
[56] Nikolaos Papandreou, Haralampos Pozidis, Nikolas Ioannou, Thomas Parnell,
Roman Pletka, Milos Stanisavljevic, Radu Stoica, Sasa Tomic, Patrick Breen, Gary
Tressler, et al .2020. Open block characterization and read voltage calibration of
3D QLC NAND flash. In2020 IEEE International Reliability Physics Symposium
(IRPS). IEEE, 1–6.
[57] Krishna Parat and Chuck Dennison. 2015. A floating gate based 3D NAND
technology with CMOS under array. In2015 IEEE International Electron Devices
Meeting (IEDM). IEEE, 3–3.
[58] Jisung Park, Roknoddin Azizi, Geraldo F Oliveira, Mohammad Sadrosadati,
Rakesh Nadig, David Novo, Juan Gómez-Luna, Myungsuk Kim, and Onur Mutlu.
2022. Flash-cosmos: In-flash bulk bitwise operations using inherent computation
capability of nand flash memory. In2022 55th IEEE/ACM International Symposium
on Microarchitecture (MICRO). IEEE, 937–955.
[59] Jisung Park, Myungsuk Kim, Myoungjun Chun, Lois Orosa, Jihong Kim, and Onur
Mutlu. 2021. Reducing solid-state drive read latency by optimizing read-retry.
InProceedings of the 26th ACM International Conference on Architectural Support
for Programming Languages and Operating Systems(Virtual, USA)(ASPLOS ’21).
Association for Computing Machinery, New York, NY, USA, 702–716. doi:10.
1145/3445814.3446719
[60] Advait Parulekar, Liam Collins, Karthikeyan Shanmugam, Aryan Mokhtari, and
Sanjay Shakkottai. 2023. Infonce loss provably learns cluster-preserving rep-
resentations. InThe Thirty Sixth Annual Conference on Learning Theory. PMLR,
1914–1961.
[61] Sumukh Pinge, Ashkan Moradifirouzabadi, Keming Fan, Prasanna Venkatesan
Ravindran, Tanvir H Pantha, Po-Kai Hsu, Zheyu Li, Weihong Xu, Zihan Xia,
Flavio Ponzina, et al .2025. FeNOMS: Enhancing Open Modification Spectral
Library Search with In-Storage Processing on Ferroelectric NAND (FeNAND)
Flash. In2025 IEEE/ACM International Conference On Computer Aided Design
(ICCAD). IEEE, 1–9.
[62] Yubin Qin, Yang Wang, Dazheng Deng, Zhiren Zhao, Xiaolong Yang, Leibo Liu,
Shaojun Wei, Yang Hu, and Shouyi Yin. 2023. FACT: FFN-Attention Co-optimized
Transformer Architecture with Eager Correlation Prediction. InProceedings of
the 50th Annual International Symposium on Computer Architecture(Orlando, FL,
USA)(ISCA ’23). Association for Computing Machinery, New York, NY, USA,
Article 22, 14 pages. doi:10.1145/3579371.3589057
[63] Derrick Quinn, Mohammad Nouri, Neel Patel, John Salihu, Alireza Salemi,
Sukhan Lee, Hamed Zamani, and Mohammad Alian. 2025. Accelerating retrieval-
augmented generation. InProceedings of the 30th ACM International Conference on
Architectural Support for Programming Languages and Operating Systems, Volume
1. 15–32.
[64] D. J. Rosenkrantz, R. E. Stearns, and P. M. Lewis. 1977. An Analysis of Several
Heuristics for the Traveling Salesman Problem.SIAM J. Comput.6, 3 (1977),
563–581. doi:10.1137/0206041
[65] Sayed Ahmad Salehi. 2022. In-memory Bulk Bitwise Logic Operation for Multi-
level Cell Non-volatile Memories. InProceedings of the 2022 International Sympo-
sium on Memory Systems. 1–5.
[66] Eran Sharon et al .2014. Simultaneous sensing of multiple word-lines and detec-
tion of NAND failures. https://patents.google.com/patent/EP2737487A1/en
[67] Michael Shen, Muhammad Umar, Kiwan Maeng, G. Edward Suh, and Udit Gupta.
2025. Hermes: Algorithm-System Co-design for Efficient Retrieval-Augmented
Generation At-Scale. InProceedings of the 52nd Annual International Symposium
on Computer Architecture (ISCA ’25). Association for Computing Machinery, New
York, NY, USA, 958–973. doi:10.1145/3695053.3731076
14

D-NOVA: In-Storage Retrieval Accelerator via Dual-Bound
3D NAND-Optimized Similarity Search with Vector Adaptation MICRO 2026, October 31–November 04, 2026, Athens, Greece
[68] Joobo Shim, Jaewon Oh, Hongchan Roh, Jaeyoung Do, and Sang-Won Lee. 2025.
Turbocharging Vector Databases Using Modern SSDs.Proc. VLDB Endow.18, 11
(July 2025), 4710–4722. doi:10.14778/3749646.3749724
[69] Wonbo Shim, Hongwu Jiang, Xiaochen Peng, and Shimeng Yu. 2021. Architectural
Design of 3D NAND Flash based Compute-in-Memory for Inference Engine. In
Proceedings of the International Symposium on Memory Systems(Washington, DC,
USA)(MEMSYS ’20). Association for Computing Machinery, New York, NY, USA,
77–85. doi:10.1145/3422575.3422779
[70] Tinku Singh, Durgesh Kumar Srivastava, and Alok Aggarwal. 2017. A novel
approach for CPU utilization on a multicore paradigm using parallel quicksort. In
2017 3rd International Conference on Computational Intelligence & Communication
Technology (CICT). IEEE, 1–6.
[71] Chang Eun Song, Priyansh Bhatnagar, Zihan Xia, Nam Sung Kim, Tajana S Rosing,
and Mingu Kang. 2025. Hybrid SLC-MLC RRAM Mixed-Signal Processing-in-
Memory Architecture for Transformer Acceleration via Gradient Redistribution.
InProceedings of the 52nd Annual International Symposium on Computer Architec-
ture. 1155–1170.
[72] Chang Eun Song, Yidong Li, Amardeep Ramnani, Pulkit Agrawal, Purvi Agrawal,
Sung-Joon Jang, Sang-Seol Lee, Tajana Rosing, and Mingu Kang. 2024. 52.5
TOPS/W 1.7 GHz Reconfigurable XGBoost Inference Accelerator Based on
Modular-Unit-Tree with Dynamic Data and Compute Gating. In2024 IEEE Custom
Integrated Circuits Conference (CICC). IEEE, 1–2.
[73] Chang Eun Song, Ashkan Moradifirouzabadi, Tajana Rosing, and Mingu Kang.
2024. Efficient Transformer Acceleration via Reconfiguration for Encoder and
Decoder Models and Sparsity-Aware Algorithm Mapping. InProceedings of the
29th ACM/IEEE International Symposium on Low Power Electronics and Design
(Newport Beach, CA, USA)(ISLPED ’24). Association for Computing Machinery,
New York, NY, USA, 1–6. doi:10.1145/3665314.3670798
[74] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. MPNet: Masked
and Permuted Pre-training for Language Understanding. InAdvances in Neural
Information Processing Systems 33 (NeurIPS 2020). https://proceedings.neurips.
cc/paper/2020/file/c3a690be93aa602ee2dc0ccab5b7b67e-Paper.pdf
[75] A. Stillmaker and B. Baas. 2017. Scaling equations for the accurate prediction
of CMOS device performance from 180 nm to 7 nm.Integration, the VLSI Jour-
nal58 (2017), 74–81. http://vcl.ece.ucdavis.edu/pubs/2017.02.VLSIintegration.
TechScale/.
[76] Yoshiki Takai, Mamoru Fukuchi, Reika Kinoshita, Chihiro Matsui, and Ken
Takeuchi. 2019. Analysis on heterogeneous SSD configuration with quadruple-
level cell (QLC) NAND flash memory. In2019 IEEE 11th International Memory
Workshop (IMW). IEEE, 1–4.
[77] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna
Gurevych. 2021. BEIR: A Heterogenous Benchmark for Zero-Shot Evalua-
tion of Information Retrieval Models. InProceedings of the 2021 Conference
on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks
Track. https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/
65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf
[78] James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal.
2018. FEVER: a Large-scale Dataset for Fact Extraction and VERification. In
Proceedings of the 2018 Conference of the North American Chapter of the Association
for Computational Linguistics: Human Language Technologies, Volume 1 (Long
Papers), Marilyn Walker, Heng Ji, and Amanda Stent (Eds.). Association for
Computational Linguistics, New Orleans, Louisiana, 809–819. doi:10.18653/v1/
N18-1074
[79] Bing Tian, Haikun Liu, Zhuohui Duan, Xiaofei Liao, Hai Jin, and Yu Zhang. 2024.
Scalable billion-point approximate nearest neighbor search using SmartSSDs. In
Proceedings of the 2024 USENIX Conference on Usenix Annual Technical Conference
(Santa Clara, CA, USA)(USENIX ATC’24). USENIX Association, USA, Article 69,
16 pages.
[80] Bing Tian, Haikun Liu, Yuhang Tang, Shihai Xiao, Zhuohui Duan, Xiaofei Liao,
Hai Jin, Xuecang Zhang, Junhua Zhu, and Yu Zhang. 2025. Towards high-
throughput and low-latency billion-scale vector search via CPU/GPU collab-
orative filtering and re-ranking. InProceedings of the 23rd USENIX Conference
on File and Storage Technologies(Santa Clara, CA, USA)(FAST ’25). USENIX
Association, USA, Article 11, 15 pages.
[81] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation Learning
with Contrastive Predictive Coding. InAdvances in Neural Information Processing
Systems.
[82] Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou.
2020. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression
of Pre-Trained Transformers. arXiv:2002.10957 [cs.CL]
[83] Puqing Wu, Minhui Xie, Enrui Zhao, Dafang Zhang, Jing Wang, Xiao Liang, Kai
Ren, and Yunpeng Chai. 2025. Turbocharge ANNS on Real Processing-in-Memory
by Enabling Fine-Grained Per-PIM-Core Scheduling. InUSENIX Annual Technical
Conference (ATC). https://www.usenix.org/system/files/atc25-wu-puqing.pdf
[84] Weihong Xu, Junwei Chen, Po-Kai Hsu, Jaeyoung Kang, Minxuan Zhou, Sumukh
Pinge, Shimeng Yu, and Tajana Rosing. 2023. Proxima: Near-storage Accel-
eration for Graph-based Approximate Nearest Neighbor Search in 3D NAND.
arXiv:2312.04257 [cs.AR] doi:10.48550/arXiv.2312.04257 Submitted on 7 Dec 2023.[85] Haichao Yang, Chang Eun Song, Weihong Xu, Behnam Khaleghi, Uday Mallappa,
Monil Shah, Keming Fan, Mingu Kang, and Tajana Rosing. 2024. FSL-HDnn: A
5.7 TOPS/W End-to-end Few-shot Learning Classifier Accelerator with Feature
Extraction and Hyperdimensional Computing. In2024 IEEE European Solid-State
Electronics Research Conference (ESSERC). IEEE, 33–36. doi:10.1109/ESSERC60068.
2024.10719453
[86] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan
Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for
Diverse, Explainable Multi-hop Question Answering. InProceedings of the 2018
Conference on Empirical Methods in Natural Language Processing. Association for
Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-
1259
[87] Amir Yazdanbakhsh, Ashkan Moradifirouzabadi, Zheng Li, and Mingu Kang.
2022. Sparse attention acceleration with synergistic in-memory pruning and
on-chip recomputation. In2022 55th IEEE/ACM International Symposium on Mi-
croarchitecture (MICRO). IEEE, 744–762.
[88] Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David
Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy
Lin. 2023. MIRACL: A Multilingual Retrieval Dataset Covering 18 Di-
verse Languages.Transactions of the Association for Computational Lin-
guistics11 (09 2023), 1114–1131. arXiv:https://direct.mit.edu/tacl/article-
pdf/doi/10.1162/tacl_a_00595/2157340/tacl_a_00595.pdf doi:10.1162/tacl_a_00595
[89] Wei Zhao and Yu Cao. 2007. Predictive technology model for nano-CMOS design
exploration.ACM Journal on Emerging Technologies in Computing Systems (JETC)
3, 1 (2007), 1–es.
[90] Justin Zobel and Alistair Moffat. 2006. Inverted files for text search engines.ACM
computing surveys (CSUR)38, 2 (2006), 6–es.
15