# Energy-Efficient On-Device RAG on a Mobile NPU: System Design and Benchmark on Snapdragon X Elite

**Authors**: Zhiyuan Cheng, Longying Lai

**Published**: 2026-06-09 01:09:00

**PDF URL**: [https://arxiv.org/pdf/2606.11257v1](https://arxiv.org/pdf/2606.11257v1)

## Abstract
Retrieval-Augmented Generation (RAG) pipelines are compute-intensive, combining embedding, retrieval, reranking, and large language model (LLM) generation. Running them entirely on-device benefits privacy, latency, and offline use, but the energy cost of CPU inference is a major barrier. We present what is, to our knowledge, the first end-to-end RAG pipeline that runs all neural stages -- embedding, reranking, and LLM generation -- on the Qualcomm Hexagon NPU of the Snapdragon X Elite. Profiling on a Dell XPS 13 laptop, we compare NPU-accelerated RAG against CPU and OpenCL/Adreno GPU baselines on indexing and query workloads. On indexing, the NPU achieves 9.1x higher embedding throughput and 12.3x less system energy. On a 120-query Wikipedia-passage benchmark, it delivers 18.1x faster LLM prefilling, 4.0x lower end-to-end query latency, and 4.0x less system energy than the CPU baseline; the same workload on the integrated GPU is 1.7x slower than CPU and uses 6.5x more energy than the NPU. A GPT-4.1 LLM-as-judge evaluation finds NPU answer quality on par with CPU and GPU within evaluator noise (mean 9.32 vs. 8.95 vs. 9.03 on a 1-10 rubric), with 86.7% of queries scoring identically across all three backends. On the Snapdragon X Elite / Hexagon class of laptop SoC, the NPU thus enables practical, energy-efficient on-device RAG without quality regression -- a sustainable path toward green edge intelligence that we expect to generalize to comparable mobile NPUs (Apple Neural Engine, Intel NPU, MediaTek APU) as their software stacks mature.

## Full Text


<!-- PDF content starts -->

Energy-Efficient On-Device RAG on a Mobile
NPU: System Design and Benchmark on
Snapdragon X Elite
1stZhiyuan Cheng
School of Engineering
Stanford University
Stanford, CA, USA
zhycheng@stanford.edu2ndLongying Lai
Simon Business School
University of Rochester
Rochester, NY , USA
lai.longying@urgrad.rochester.edu
Abstract—Retrieval-Augmented Generation (RAG) pipelines
are compute-intensive, combining embedding, retrieval, rerank-
ing, and large language model (LLM) generation. Running them
entirely on-device benefits privacy, latency, and offline use, but the
energy cost of CPU inference is a major barrier. We present what
is, to our knowledge, the first end-to-end RAG pipeline that runs
all neural stages—embedding, reranking, and LLM generation—
on the Qualcomm Hexagon NPU of the Snapdragon X Elite.
Profiling on a Dell XPS 13 laptop, we compare NPU-accelerated
RAG against CPU and OpenCL/Adreno GPU baselines on index-
ing and query workloads. On indexing, the NPU achieves 9.1×
higher embedding throughput and 12.3×less system energy. On a
120-query Wikipedia-passage benchmark, it delivers 18.1×faster
LLM prefilling, 4.0×lower end-to-end query latency, and 4.0×
less system energy than the CPU baseline; the same workload on
the integrated GPU is 1.7×slower than CPU and uses 6.5×more
energy than the NPU. A GPT-4.1 LLM-as-judge evaluation finds
NPU answer quality on par with CPU and GPU within evaluator
noise (mean 9.32 vs. 8.95 vs. 9.03 on a 1–10 rubric), with 86.7%
of queries scoring identically across all three backends. On the
Snapdragon X Elite / Hexagon class of laptop SoC, the NPU thus
enables practical, energy-efficient on-device RAG without quality
regression—a sustainable path toward green edge intelligence
that we expect to generalize to comparable mobile NPUs (Apple
Neural Engine, Intel NPU, MediaTek APU) as their software
stacks mature.
Index Terms—Neural Processing Unit, Retrieval-Augmented
Generation, on-device inference, energy efficiency, edge AI, Snap-
dragon X Elite
I. INTRODUCTION
The proliferation of large language models (LLMs) has
driven demand for on-device AI systems that operate without
cloud connectivity, preserving user privacy, reducing latency,
and enabling offline functionality [1]. Retrieval-Augmented
Generation (RAG) [2] has emerged as the dominant paradigm
for grounding LLM outputs in external knowledge, mitigating
hallucination while adapting to domain-specific corpora. How-
ever, a complete RAG pipeline involves multiple compute-
intensive neural inference stages—embedding generation, se-
mantic retrieval, cross-encoder reranking, and autoregressive
LLM generation—making on-device deployment challengingunder the power and thermal constraints of mobile and laptop
platforms.
Modern systems-on-chip (SoCs) increasingly integrate ded-
icated Neural Processing Units (NPUs) alongside CPUs and
GPUs. Qualcomm’s Snapdragon X Elite, for example, features
the Hexagon NPU with up to 45 TOPS of INT8 through-
put, designed specifically for sustained neural inference at
low power [3]. While prior work has demonstrated NPU-
accelerated inference for individual models—particularly LLM
prefilling [4]—we are not aware of any prior study that has
implemented and benchmarked acompleteRAG pipeline on
a mobile NPU.
This gap is significant for two reasons. First, a RAG pipeline
loads multiple models simultaneously (embedding, reranker,
LLM), creating memory allocation and scheduling challenges
unique to the NPU’s static computation graph architecture.
Second, the energy savings from NPU offloading compound
across pipeline stages, but the magnitude of this benefit has
not been quantified for multi-model workloads. Understanding
these trade-offs is essential for designing sustainable on-device
AI systems [5].
We note that on the Snapdragon X Elite, the integrated
GPU (Adreno X1-85) is not a viable acceleration target for
this workload: although OpenCL offload is functional, end-to-
end query processing is 1.7×slower than the CPU baseline
and consumes 1.6×more system energy (Section V). This
is a hardware ceiling—the integrated GPU is small relative
to the X-Elite’s 12 high-performance cores—not an immature
software stack. The cross-encoder reranker additionally cannot
run on the OpenCL backend due to a batched-scoring defect,
so it stays on the NPU in all configurations. The NPU is
therefore the only practical on-chip accelerator for this class
of workload.
In this paper, we make the following contributions:
1)A complete NPU-accelerated RAG pipeline.We im-
plement what is, to our knowledge, the first end-
to-end RAG system in which all three neural in-
ference components—embedding generation (Embed-
dingGemma 300M [6]), cross-encoder reranking (JinaarXiv:2606.11257v1  [cs.CL]  9 Jun 2026

Reranker v2 [7]), and LLM generation (Qwen3-4B-
Instruct [8])—run on the Qualcomm Hexagon NPU
through the Qualcomm AI Runtime (QAIRT/QNN)
SDK [9]. We document critical engineering challenges
including model loading order constraints, context
length limitations imposed by static computation graphs,
and dependency issues in the Windows ARM64 environ-
ment.
2)Comprehensive performance and energy benchmark.
We profile both the indexing pipeline (document parsing,
embedding, FAISS index construction) and the query
pipeline (hybrid retrieval, reranking, LLM generation)
on identical hardware, comparing NPU against CPU
baselines. We instrument power draw via HWiNFO64
shared-memory telemetry at 500 ms granularity, measur-
ing per-component power and total system energy.
3)Quantitative sustainability analysis.We show that
NPU acceleration reduces total system energy by 12.3×
for indexing and 19.2×for query processing, with
energy savings that aresuper-linearrelative to speedup
because the NPU also lowers average system power
draw. We contextualize these savings in terms of daily
and annual energy budgets for edge RAG deployments,
connecting to the Green AI agenda [10].
II. RELATEDWORK
On-device LLM inference.The deployment of LLMs
on edge devices has been enabled by quantization, pruning,
and efficient runtime systems. llama.cpp [11] provides cross-
platform CPU/GPU inference for GGUF-quantized models
and is widely adopted for desktop and mobile deployment.
MLC-LLM [12] leverages machine learning compilation to
generate optimized kernels for diverse hardware backends
including Vulkan and Metal. ExecuTorch [13] targets mobile
deployment within the PyTorch ecosystem. The llm.npu sys-
tem [4] is the closest predecessor to our work: it demon-
strates NPU-accelerated LLMprefillingon mobile SoCs,
achieving 22.4×speedup and 30.7×energy savings over
baselines by employing prompt-level chunking, tensor-level
outlier extraction, and block-level heterogeneous scheduling.
However, llm.npu focuses exclusively on single-model LLM
inference and does not address the multi-model RAG setting.
Xu et al. [1] provide a comprehensive survey of on-device
language model techniques, covering efficient architectures,
compression, and hardware acceleration, but do not benchmark
NPU-based RAG pipelines. Orthogonal to hardware offload,
algorithmic efficiency techniques reduce the compute a model
needs and are complementary to the NPU acceleration we
study: reasoning pruning with skill-aware step decomposi-
tion [14], multi-teacher distillation [15], and pattern-aware
tool-integrated reasoning [16]. Small language models can
match or outperform larger ones on targeted tasks [17], even as
autonomous agents for efficient knowledge mining [18], and
data-efficiency and compression optimization further shrink
deployment cost [19]; efficient sequence architectures and
modular cross-domain adaptation reduce it further [20], [21];performance–efficiency trade-offs between traditional machine
learning and LLMs have also been characterized directly [22].
RAG systems.Since the introduction of RAG by Lewis
et al. [2], numerous frameworks have emerged for building
retrieval-augmented pipelines. LangChain [23] and LlamaIn-
dex [24] provide modular abstractions for document ingestion,
retrieval, and generation, but are designed primarily for server
and cloud environments. Gao et al. [25] survey the RAG
landscape and identify hybrid retrieval, reranking, and iterative
refinement as key design patterns. Recent work has sharpened
RAG’s reliability and retrieval quality: studies on knowledge-
conflict and context compliance diagnose when retrieval is
wrong [26], transparent knowledge-conflict handling makes
such failures interpretable [27], evidence-force calibration
improves citation faithfulness in cited RAG [28], and retrieval-
augmented architectures have been studied specifically under
tight on-device compute budgets [29]. Complementary efforts
mine and curate retrievable knowledge through web-scale
pipelines [30] and probe how prompt design shapes LLM rea-
soning over retrieved evidence [31]. Retrieval itself continues
to advance beyond text into composed image settings [32],
[33], and RAG and efficient-inference techniques are in-
creasingly applied in specialized domains such as financial
decision-making [34], multi-LLM financial sentiment fore-
casting [35], interpretable decision-support analytics [36], and
clinical and mental-health applications [37]. In the financial-
document setting we also evaluate, hybrid retrieval with
cross-encoder reranking improves answer correctness on 10-
K filings [38], and document-routed retrieval mitigates cross-
document chunk confusion in homogeneous corpora [39]. To
our knowledge, no prior work has implemented or bench-
marked a complete RAG pipeline targeting on-device NPU
execution.
NPU computing and hardware acceleration.Neural Pro-
cessing Units are increasingly integrated into consumer SoCs.
Qualcomm’s Hexagon NPU [3] powers on-device inference on
Snapdragon platforms, while Apple’s Neural Engine and In-
tel’s NPU serve similar roles on their respective architectures.
Qualcomm’s AI Runtime (QAIRT/QNN) SDK [9] exposes
the Hexagon NPU to applications, compiling models into
static computation graphs ahead of time; for CPU and GPU
execution, llama.cpp [11] is the de facto runtime for GGUF-
quantized models. Co-designing model architectures for NPU
constraints, such as integer-only vision-language inference,
can further narrow the hardware–model gap [40]. Despite
growing hardware capability, systematic benchmarks of multi-
model NPU workloads remain scarce.
Energy-efficient AI.Schwartz et al. [10] introduced the
Green AI framework, advocating for energy efficiency as a
first-class evaluation criterion alongside accuracy. Subsequent
work has measured the carbon footprint of training large
models [41], [42] and proposed carbon-aware scheduling for
cloud inference. However, energy measurement foron-device
NPU inference remains underexplored, particularly for multi-
stage pipelines. Our work addresses this gap by providing fine-
grained power telemetry for a complete RAG workflow on a

mobile NPU.
III. SYSTEMDESIGN
We design an end-to-end RAG system that executes all
neural inference on the Qualcomm Hexagon NPU. The sys-
tem comprises two pipelines—indexing and query—sharing
a common model infrastructure. Fig. 1 illustrates the overall
architecture.
A. Indexing Pipeline
The indexing pipeline transforms a corpus of documents
into a searchable vector index through three stages:
Stage 1: Document parsing and chunking.Documents
(PDF, text) are parsed into structured text using format-specific
parsers. The extracted text is then split into chunks using
recursive character-based splitting with a chunk size of 1,000
characters and 500-character overlap. These parameters are
constrained by the NPU’s limited context window (discussed
in Section III-D); a CPU-only deployment would use larger
chunks (2,500 characters, 1,250 overlap). Chunks and meta-
data are stored in a SQLite database with FTS5 full-text search
indexing.
Stage 2: Embedding generation.Each chunk is encoded
into a 1,024-dimensional vector using EmbeddingGemma
300M [6], a lightweight embedding model from the Gemma 3
family optimized for on-device deployment. The model runs
on the Hexagon NPU through the QAIRT/QNN runtime,
processing chunks in batches of up to 32. Embeddings are
L2-normalized before storage.
Stage 3: FAISS index construction.Embedding vectors
are added to a FAISSIndexFlatL2index wrapped with
IndexIDMap2for chunk-ID-based retrieval [43]. The flat
index enables exact nearest-neighbor search, which is feasible
given the on-device corpus scale (thousands to tens of thou-
sands of chunks).
B. Query Pipeline
The query pipeline processes user queries through four
stages:
Query rewriting.The raw user query is optionally rewritten
by the LLM (Qwen3-4B-Instruct [8]) to produce a clarified
query and extracted keywords. This step improves retrieval
quality by resolving ambiguity and expanding query terms.
Hybrid retrieval.The system performs two parallel re-
trieval passes: (1) keyword-based retrieval via SQLite FTS5
(up to 20 results), and (2) semantic retrieval via FAISS nearest-
neighbor search using the query embedding (up to 30 results).
Results are fused using Reciprocal Rank Fusion (RRF) [44]
withk=60.
Reranking.Fused candidates (up to 30) are reranked by a
cross-encoder model, Jina Reranker v2 Base Multilingual [7],
running on the NPU. The reranker assigns relevance scores;
candidates are filtered using both a minimum score threshold
(0.45) and a cliff-cutoff heuristic that detects sharp score drops
between consecutive candidates. The top 7 chunks are retained
for generation.Answer generation.The retained chunks are assembled
into a context prompt and passed to Qwen3-4B-Instruct [8] for
streaming answer generation on the NPU. The model operates
with a 4,096-token context window.
C. Inference Backends
Each neural component runs on one of two interchangeable
backends, allowing controlled comparison across compute tar-
gets. For NPU execution, models are compiled ahead of time
into static computation graphs and run through the Qualcomm
AI Runtime (QAIRT/QNN) SDK [9]. For CPU and GPU
execution, the same model architectures are run in GGUF-
quantized format through llama.cpp [11], with GPU offload
enabled by settingn_gpu_layersto the full layer count.
We wrap both backends behind a common interface so that the
embedding, reranking, and LLM stages can be independently
assigned to the NPU, CPU, or GPU without changing pipeline
logic.
D. Engineering Challenges
Deploying a multi-model RAG pipeline on the NPU re-
vealed several non-trivial engineering challenges:
Model loading order.The NPU requires models to be
loaded in descending order of size: LLM (4B parameters)
first, then embedding (300M), and finally reranker (278M).
Violating this order causes static computation graph memory
allocation failures, likely due to the NPU’s contiguous memory
allocation strategy. This constraint does not apply to CPU/GPU
backends.
Context length limitations.NPU static graphs impose a
fixed maximum context length, which is more restrictive than
the flexible KV-cache approach used in CPU/GPU inference.
This necessitates smaller chunk sizes during indexing (1,000
vs. 2,500 characters) and fewer assembled chunks during
generation (7 vs. 10), reducing the context available to the
LLM but remaining sufficient for our benchmarks.
ARM64 Windows ecosystem.The entire pipeline runs
in a native Windows ARM64 Python environment on the
Snapdragon X Elite platform. Many Python packages lack pre-
built ARM64 Windows wheels, requiring manual compilation
or alternative dependencies. This ecosystem immaturity adds
substantial development overhead but is essential for lever-
aging the NPU through the QAIRT/QNN Windows ARM64
runtime.
IV. EXPERIMENTALSETUP
A. Hardware Platform
All experiments are conducted on a Dell XPS 13 9345
laptop. Table I summarizes the hardware specifications. To
ensure measurement integrity, no other applications run during
experiments, and the system operates on AC power in battery
performance mode.

Indexing Pipeline
Query PipelineDocument
ParsingChunkingEmbedding
Generation
(NPU)FAISS Index
Construction
SQLite DB
(FTS5)
User
QueryQuery
Rewrite
(NPU LLM)Hybrid
Retrieval
(FTS+FAISS)Reranking
(NPU)Answer
Generation
(NPU LLM)Answer
NPU-accelerated CPU / I/O Storage
Fig. 1. Architecture of the NPU-accelerated RAG system. Blue-shaded stages execute neural inference on the Hexagon NPU; gray stages run on the CPU;
orange stages involve persistent storage. The indexing pipeline (top) processes documents into a searchable index; the query pipeline (bottom) retrieves relevant
chunks and generates answers.
TABLE I
HARDWARE SPECIFICATIONS OF THE TEST PLATFORM.
Component Specification
Device Dell XPS 13 9345
CPU Snapdragon X Elite X1E80100
(Qualcomm Oryon, 12 cores)
GPU Qualcomm Adreno X1-85
NPU Hexagon NPU (45 TOPS INT8)
Memory 64 GB LPDDR5x
OS Windows 11 ARM64
Power supply 100 W AC adapter
Power mode Battery performance mode
TABLE II
MODELS USED IN THERAGPIPELINE.
Component Model Params Dim
Embedding EmbeddingGemma [6] 300M 1024
Reranker Jina Reranker v2 [7] 278M —
LLM Qwen3-4B-Instruct [8] 4B —
B. Models
Table II lists the models used in the RAG pipeline. For
NPU execution, models are compiled ahead of time into the
QAIRT/QNN static-graph format. For the CPU and GPU
baselines, the same model architectures are used in GGUF-
quantized format, inferred via llama.cpp.
C. Dataset and Workload
Forindexing, we process a corpus of 10 documents (the
SEC 10-K filings of A, AAL, AAPL, ABBV , ABNB, ABT,
ACGL, ACN, ADBE, and ADI), producing 9,324 chunks com-
prising approximately 8.14 million characters (≈2.04 million
estimated tokens).
Forquery processing, we benchmark on two complemen-
tary 120-query test sets, each requiring full pipeline execution
(retrieval, reranking, and generation):•wiki minirag(general-knowledge factoid QA).
3,187 Wikipedia passages drawn from the
rag-mini-wikipediadataset [45]; 120 queries
deterministically sampled (every 7th row) from the
dataset’s 918-question test split. Wikipedia passages
contain the answer to every query, allowing the
benchmark to isolate backend-induced quality differences
from corpus-coverage artifacts. We use this set as the
primary workload for performance, energy, and answer-
quality evaluation.
•FinDER(domain-specialized financial QA). 120 queries
from the FinDER [46] test split, restricted to the 10 in-
dexed companies. Queries are heavily abbreviated (e.g.,
“ADI’s eng. talent ratio & turnover impact on comp.
innovation.”); we use this set as a stress test of correct
refusal behavior on a tiny domain-specialist corpus.
D. Profiling Methodology
We instrument profiling at three levels:
Stage-level timing.Wall-clock time for each pipeline
stage is measured using Python’stime.perf_counter(),
providing microsecond-resolution timestamps. Throughput is
computed as tokens processed per second for embedding
(prefilling speed) and LLM inference (prefilling and decoding
speeds separately).
Power and energy monitoring.A background moni-
toring thread samples system telemetry at 500 ms inter-
vals throughout each experiment. On the Windows plat-
form, we read HWiNFO64 sensor data from shared memory
(Global\HWiNFO_SENS_SM2), capturing per-component
power draw (CPU clusters 0/1/2, GPU, system total), NPU
utilization percentage, and GPU utilization. System-level re-
source metrics (CPU utilization, process memory) are col-
lected viapsutil. Total energy consumption (Joules) is
computed by numerical integration of instantaneous power
samples over the experiment duration.

LLM profiling.For query processing, we measure time-to-
first-token (TTFT) to isolate the prefilling phase, and tokens-
per-second during decoding. These metrics are extracted from
the runtime’s built-in profiling interface (QAIRT/QNN for the
NPU, llama.cpp for the CPU and GPU).
Answer-quality evaluation.We score every generated an-
swer against a per-query reference answer using GPT-4.1 as
an LLM-as-judge [47], [48], with temperature 0.0 and a 1–10
rubric where≥7 denotes a materially correct answer and 10
denotes a perfect match. The judge sees only the question,
the system’s answer, and the reference; it does not see which
backend produced the answer. We report mean score, failure
rate (s=1), correctness rate (s≥7), and perfect rate (s=10) for
each backend, plus paired exact Wilcoxon signed-rank tests on
the per-query score differences. We use a single judge in line
with established LLM-as-judge practice for systems papers;
this caveat is discussed in Section VI-B.
E. Baselines
Our experiments compare three on-chip backends on the
same Snapdragon X Elite SoC, using identical model archi-
tectures and identical pipeline code. The reranker stays on
the Hexagon NPU in all configurations (the OpenCL backend
exhibits a batched-scoring defect, and pinning the reranker
isolates the LLM as the only varying component). The three
backends differ only in where the LLM (and embedder)
executes:
•NPU.Embedding, reranking, and LLM generation all
execute on the Hexagon NPU through the QAIRT/QNN
SDK using ahead-of-time-compiled static-graph models.
•CPU.The LLM and embedder run on the X-Elite
CPU through llama.cpp withn_gpu_layers=0, using
GGUF Q4 KM weights for the LLM and BF16 weights
for the embedder. The reranker remains on the NPU.
•GPU.The LLM and embedder run on the Adreno
X1-85 integrated GPU through llama.cpp with full
OpenCL offload (n_gpu_layers=999), using the
same GGUF/BF16 weights as the CPU baseline. The
reranker remains on the NPU.
This design ensures that performance, energy, and quality
differences across the three rows are attributable to the LLM
compute backend rather than to model architecture, quantiza-
tion, or pipeline configuration.
V. RESULTS ANDANALYSIS
A. Indexing Performance
Table III presents the indexing benchmark results. The NPU
achieves a9.1×improvement in embedding throughput (3,325
vs. 367 tokens/s), which dominates the total pipeline time.
Stage 1 (parsing/chunking) and Stage 3 (FAISS construction)
show similar performance across backends, as expected for
CPU-bound and I/O-bound operations respectively.
The total pipeline speedup (7.9×) is lower than the
embedding-stage speedup (9.1×) because the parsing stage
constitutes a fixed overhead that is identical across backends.
With embedding dominating pipeline time (86% for NPU,TABLE III
INDEXING PERFORMANCE COMPARISON(10DOCUMENTS, 9,324
CHUNKS,≈2.04MTOKENS).
Metric NPU CPU Ratio
Stage Timing (seconds)
Parse & chunk 97.6 92.7 1.0×
Embedding 612.4 5,554.5 9.1×
FAISS build 0.5 0.8 1.6×
Total pipeline 710.7 5,648.2 7.9×
Embedding Throughput
Tokens/s 3,325 367 9.1×
Chunks/s 15.2 1.7 8.9×
Resource Utilization
Avg CPU util. (%) 12.6 98.5 —
Avg NPU util. (%) 84.2 0.0 —
Energy
Avg system power (W) 27.6 42.7 0.65×
Total system energy (J) 19,592 241,02012.3×
TABLE IV
QUERY PERFORMANCE ON WIKI MINIRAG(MEAN OVER120QUERIES).
RATIOS ARE TAKEN AGAINST THECPUBASELINE.
Metric NPU CPU GPU N/C G/C
Latency (seconds)
Retrieval time 3.09 3.09 3.10 1.0×1.0×
LLM TTFT 1.30 24.77 42.24 19.1×0.59×
Total query time 9.48 37.98 63.61 4.0×0.60×
LLM Throughput (tokens/s)
Prefilling speed 786.66 43.36 25.18 18.1×0.58×
Decoding speed 14.19 8.17 4.65 1.74×0.57×
Resource Utilization
Avg CPU util. (%) 55.84 79.10 41.86 — —
Avg NPU util. (%) 97.30 7.67 4.62 — —
Avg GPU D3D (%) 10.37 9.83 71.19 — —
Tail Latency (P95, seconds)
Total query time 17.90 69.56 107.40 3.9×0.65×
Energy (total over 120 queries)
Avg sys. power (W) 33.24 32.94 32.25 1.01×0.98×
Total sys. E (kJ) 37.83 150.12 246.144.0×0.61×
98% for CPU), the NPU effectively eliminates the embedding
bottleneck.
B. Query Performance
Table IV summarizes query processing results on the
wiki minirag benchmark, averaged over 120 queries, for all
three on-chip backends. The NPU delivers dramatic improve-
ments over both CPU and GPU baselines on the LLM-
intensive stages.
Prefilling speedis the standout result: the NPU achieves
786.7 tokens/s, an 18.1×improvement over the CPU’s 43.4 to-
kens/s and a 31.2×improvement over the GPU’s 25.2 to-
kens/s. This aligns with the NPU’s strength in dense matrix-
multiplication workloads that dominate the prefilling phase.
Decoding speedshows a more modest 1.74×improvement
over CPU (14.2 vs. 8.2 tokens/s) and 3.05×over GPU, consis-

TABLE V
PER-COMPONENT ENERGY BREAKDOWN OVER THE120-QUERY
WIKI MINIRAG BATCH(HWINFO64SENSORS, 500MS SAMPLING).
NPU CPU GPU
Component W kJ W kJ W kJ
CPU Cluster 0 4.11 4.67 5.33 24.30 3.33 25.44
CPU Cluster 1 4.86 5.53 5.48 24.99 3.89 29.71
CPU Cluster 2 4.63 5.27 6.46 29.46 4.11 31.40
GPU 0.08 0.10 0.07 0.33 2.15 16.40
System total 33.24 37.83 32.94 150.12 32.25 246.14
tent with the memory-bandwidth-bound, inherently sequential
nature of autoregressive token generation.
The Adreno X1-85 GPU is genuinely active but slower
than the X-Elite CPUon this 4B Q4 KM model. The GPU
configuration sustains 71.2% GPU D3D utilization throughout
the run (vs.≈10% in the NPU and CPU configurations), con-
firming OpenCL offload is functional. However, the integrated
GPU is small relative to the X-Elite’s 12 high-performance
cores, and decoding is 1.7×slower than CPU. This is a
hardware-ceiling result, not a runtime defect.
Retrieval timeis nearly identical across backends (≈3.1 s),
as expected: this stage involves FAISS vector search and
SQLite FTS5 queries, which are CPU-bound and do not
benefit from NPU or GPU acceleration.
Tail latency(P95) reflects the same ranking as the means:
17.9 s NPU vs. 69.6 s CPU vs. 107.4 s GPU. The NPU provides
consistent, predictable performance suitable for interactive use.
C. Energy Analysis
Table V shows the per-component energy breakdown dur-
ing query processing.Average system power is essentially
identical across all three backends(33.24, 32.94, 32.25 W);
the platform’s idle baseline dominates the system rail. The
energy ranking is therefore set almost entirely by wall-clock
time: NPU completes the 120-query batch in 19 min, CPU in
76 min, GPU in 127 min.
For indexing, the original 7.9×NPU/CPU speedup yields a
12.3×system-energy reduction through the more pronounced
super-linear dual effect we documented in earlier experiments:
with the NPU handling embedding, CPU clusters operate
at near-idle power, reducing average system power by 35%
(27.6 W vs. 42.7 W). Indexing is a substantially more compute-
dense workload than per-query LLM generation in the on-
device setting, which is why the system-power gap is wider
for indexing.
For query processing, the LLM-decoding phase is memory-
bandwidth-bound; the system power difference between back-
ends collapses to noise, so the energy ratio tracks the latency
ratio (4.0×NPU/CPU, 0.61×GPU/CPU, 6.5×NPU/GPU).
The OpenCL run consumes 16.4 kJ in the GPU domain
alone—165×the NPU run’s GPU energy—confirming Adreno
is genuinely active, but it does not translate into wall-clock
savings.19.6241.0
37.8150.1246.1
NPU CPU NPU CPU GPU
Indexing Query (120)12.3×
6.5×
Fig. 2. Total system energy in kJ (HWiNFO64). Indexing: NPU vs. CPU.
Query: NPU vs. CPU vs. GPU/OpenCL on the wiki minirag 120-query
benchmark.
TABLE VI
ANSWER QUALITY ON WIKI MINIRAG(GPT-4.1JUDGE, 1–10RUBRIC,
120QUERIES PER BACKEND).
Backend Avg Fail (s=1) Correct (s≥7) Perfect (s=10)
CPU 8.95 5.0% 87.5% 84.2%
GPU/OpenCL 9.03 6.7% 88.3% 85.0%
NPU 9.32 5.0% 92.5% 88.3%
Fig. 2 visualizes total system energy across indexing (NPU
vs. CPU) and query processing (all three backends).
D. RAG Answer Quality
The NPU runs a deliberately constrained configuration:
1,000-character chunks (vs. 2,500 on CPU/GPU), top-7 re-
trieved chunks (vs. top-10), and a 4,096-token context window.
A natural concern is whether this constraint degrades answer
quality. Table VI reports a GPT-4.1 LLM-as-judge evaluation
on the wiki minirag test set across all three backends.
Quality is on par across all three backends.104 of
120 queries (86.7%) receive identical scores from all three
backends. Pairwise paired exact Wilcoxon signed-rank tests
on the≤15 discordant pairs yieldp=0.039for NPU−CPU
(mean diff+0.367),p=0.121for NPU−GPU (+0.283), and
p=0.313for CPU−GPU (−0.083). A manual spot-check of
the largest disagreements shows the residual gap is domi-
nated by judge sensitivity rather than substantive backend
differences—e.g., on the query“What is the biggest city
in Finland?”the CPU and GPU emit nearly identical text,
yet are scored 2 and 8 respectively, a 6-point swing on
essentially the same answer; on“Who did Ford nominate for
Vice President?”all three backends correctly answer “Nelson
Rockefeller” (the figure named in the retrieved passage), but
the dataset’s reference is “Bob Dole” (a different event), so
two of the three direct answers receive a 1 from the judge. We
therefore report this asanswer-quality parity within evaluator
noise: the NPU’s constrained configuration does not materially

degrade quality, even as it delivers a 4.0×system-energy
reduction.
Refusal behavior on a domain-specialist corpus.The
FinDER queries are heavily abbreviated and target a tiny 10-
document corpus, so we use this dataset to evaluatecorrect
refusalrather than answer quality. Across the 120-query test
set, the NPU returns substantive answers for 9.2%, refusing
the rest with messages such as “the provided context does not
contain sufficient information”; the CPU returns substantive
answers for 44.2%. Among the substantive subset, mean GPT-
4.1 scores are 6.45 (NPU) and 7.47 (CPU). Both backends
therefore refuse rather than fabricate when retrieval evidence
is weak—the desired conservative behavior for an on-device
assistant. The substantive-rate gap reflects the CPU’s larger
context window (top-10 chunks of 2,500 characters) admitting
more borderline-relevant passages, not a quality difference;
in line with the wiki minirag finding, the answers that are
produced are of comparable quality.
VI. DISCUSSION
A. Sustainability Implications
The energy savings demonstrated by NPU-accelerated RAG
have meaningful implications for sustainable AI deployment
at the edge. Per-query system energy on the wiki minirag
benchmark is 315 J for the NPU, 1,251 J for the CPU, and
2,051 J for the GPU. Consider a scenario in which an on-
device RAG system processes 1,000 queries per day:
•NPU:1,000×315J= 315kJ/day≈87.5Wh/day
•CPU baseline:1,000×1,251J= 1,251kJ/day≈
347.5Wh/day
•GPU/OpenCL:1,000×2,051J= 2,051kJ/day≈
569.7Wh/day
Relative to the CPU baseline, the NPU saves approximately
260 Wh per day, or 94.9 kWh per year per device. For fleet
deployments across thousands of edge devices, these savings
scale proportionally, contributing to measurable reductions in
operational carbon footprint. The integrated GPU is, in this
regime, theworstof the three backends—an important data
point for practitioners who might otherwise default to GPU
offload because it is the conventional choice on x86 platforms.
Beyond energy, when LLM inference moves to the NPU
the CPU clusters drop to near-idle power (4.1–4.6 W average
vs. 5.3–6.5 W under CPU inference), freeing CPU capacity for
other on-device tasks. Lower thermal dissipation also reduces
fan noise and extends device longevity, both of which matter
for battery-powered laptop and tablet form factors.
B. Limitations and Future Work
Our study has several limitations that suggest directions for
future research:
Static computation graphs.The NPU’s requirement for
statically compiled models imposes fixed context length limits,
preventing dynamic context expansion. Future NPU runtimes
with dynamic shape support would alleviate the chunk size
and context window constraints documented in Section III-D.Model loading constraints.The observed largest-first
model loading order requirement reflects current NPU memory
management limitations. As NPU software stacks mature, we
expect more flexible memory allocation strategies to emerge.
Limited model ecosystem.Not all model architectures have
NPU-compiled variants available. Expanding the set of NPU-
compatible embedding, reranking, and LLM models would
broaden the applicability of NPU-accelerated RAG.
Single-device evaluation.Our benchmarks are conducted
on a single hardware platform (Snapdragon X Elite). Evaluat-
ing across multiple NPU architectures (Apple Neural Engine,
Intel NPU, MediaTek APU) and form factors (smartphones,
tablets) would strengthen generalizability. We expect the
qualitative finding—that an NPU accelerates the prefilling-
dominated phase of RAG enough to make end-to-end on-
device deployment energy-competitive—to carry across ar-
chitectures, but the specific speedup and energy ratios are
platform-dependent.
Single LLM-as-judge evaluator.Our quality evaluation
in Section V-D uses one judge (GPT-4.1, temperature 0.0)
and no human annotation. A multi-evaluator cross-check or
human-graded subset would harden the parity claim. The spot-
check in Section V-D shows that the residual NPU–CPU gap is
dominated by judge sensitivity rather than substantive backend
differences, which is consistent with the conclusion (parity
within evaluator noise) but does not by itself rule out judge
bias [48]. More broadly, ensuring LLM and vision-language
outputs are robust and safe remains an active concern [49],
[50], including the compositional risks that arise when agent
skills are combined [51].
VII. CONCLUSION
We have presented what is, to our knowledge, the first
end-to-end RAG pipeline that executes all neural inference
stages on a mobile NPU, demonstrating the viability of
NPU-accelerated on-device retrieval-augmented generation.
Our system integrates embedding generation, cross-encoder
reranking, and LLM-based answer generation on the Qual-
comm Hexagon NPU of the Snapdragon X Elite through the
Qualcomm AI Runtime (QAIRT/QNN) SDK, with llama.cpp
providing the CPU and GPU baselines.
Comprehensive benchmarks on a Dell XPS 13 laptop re-
veal that NPU acceleration delivers 9.1×higher embedding
throughput and 18.1×faster LLM prefilling compared to
CPU inference, while reducing total system energy by 12.3×
for indexing and 4.0×for query processing on a 120-query
Wikipedia-passage benchmark. A direct comparison against
the same Snapdragon X Elite’s integrated GPU shows that
OpenCL offload to the Adreno X1-85, while functional, is
1.7×slower than the CPU on this workload and consumes
6.5×more system energy than the NPU—an important data
point for practitioners who might default to GPU offload as
the conventional accelerator choice. A GPT-4.1 LLM-as-judge
evaluation confirms that the NPU’s constrained configuration
produces answers of equivalent quality to the CPU and GPU
within evaluator noise, indicating that the energy savings

come essentially for free on quality. These results position
the Hexagon-class mobile NPU as a compelling platform for
energy-efficient on-device AI on the Snapdragon X Elite, with
a system design that we expect to carry over to compara-
ble mobile NPU architectures (Apple Neural Engine, Intel
NPU, MediaTek APU) as their hardware capabilities, software
toolchains, and model ecosystems continue to mature.
Our source code will be made publicly available at https://
github.com/zhycheng614/NPU-RAG-public upon publication
of this work.
REFERENCES
[1] J. Xuet al., “On-device language models: A comprehensive review,”
arXiv preprint arXiv:2409.00088, 2024.
[2] P. Lewis, E. Perez, A. Piktus, F. Petroni, V . Karpukhin, N. Goyal,
H. K ¨uttler, M. Lewis, W.-t. Yih, T. Rockt ¨aschel, S. Riedel, and D. Kiela,
“Retrieval-augmented generation for knowledge-intensive NLP tasks,” in
Advances in Neural Information Processing Systems, vol. 33, 2020, pp.
9459–9474.
[3] Qualcomm Technologies, Inc., “Snapdragon X elite platform,”
https://www.qualcomm.com/products/mobile/snapdragon/
pcs-and-tablets/snapdragon-x-elite, 2024.
[4] D. Xu, H. Zhang, L. Yang, R. Liu, G. Huang, M. Xu, and X. Liu, “Fast
on-device LLM inference with NPUs,” inProceedings of the 30th ACM
International Conference on Architectural Support for Programming
Languages and Operating Systems (ASPLOS ’25), Volume 1. ACM,
2025, pp. 445–462.
[5] Z. Cheng, L. Lai, Y . Liu, and Y . Sun, “Toward sustainable on-device
intelligence: A survey on energy-efficient RAG systems with small
language models,”Available at SSRN 6698538, 2026.
[6] J. Leeet al., “EmbeddingGemma: Powerful and lightweight text repre-
sentations,”arXiv preprint arXiv:2509.20354, 2025.
[7] N. Sturuaet al., “jina-reranker-v2-base-multilingual,” https:
//huggingface.co/jinaai/jina-reranker-v2-base-multilingual, 2024,
jina AI.
[8] A. Yanget al., “Qwen3 technical report,”arXiv preprint
arXiv:2505.09388, 2025.
[9] Qualcomm Technologies, Inc., “Qualcomm AI runtime
(QAIRT) SDK,” https://www.qualcomm.com/developer/software/
qualcomm-ai-runtime-sdk, 2024.
[10] R. Schwartz, J. Dodge, N. A. Smith, and O. Etzioni, “Green AI,”
Communications of the ACM, vol. 63, no. 12, pp. 54–63, 2020.
[11] G. Gerganovet al., “llama.cpp: Inference of Meta’s LLaMA model in
pure C/C++,” https://github.com/ggerganov/llama.cpp, 2024.
[12] MLC AI, “MLC-LLM: Machine learning compilation for large language
models,” https://llm.mlc.ai/, 2024.
[13] Meta Platforms, Inc., “ExecuTorch: End-to-end solution for enabling
on-device inference,” https://github.com/pytorch/executorch, 2024.
[14] Y . Jiang, D. Li, and F. Ferraro, “DRP: Distilled reasoning pruning with
skill-aware step decomposition for efficient large reasoning models,”
2026. [Online]. Available: https://arxiv.org/abs/2505.13975
[15] H. Zhang, S. Yang, X. Liang, C. Shang, Y . Jiang, C. Tao, J. Xiong,
H. K.-H. So, R. Xie, A. X. Chang, and N. Wong, “Find your optimal
teacher: Personalized data synthesis via router-guided multi-teacher
distillation,” 2026. [Online]. Available: https://arxiv.org/abs/2510.10925
[16] N. Xu, Y . Jiang, S. R. Dipta, and H. Zhang, “Learning how to use
tools, not just when: Pattern-aware tool-integrated reasoning,” 2026.
[Online]. Available: https://arxiv.org/abs/2509.23292
[17] J. Cao, Y . Ma, X. Li, Q. Ren, and X. Chen, “Task-specific efficiency
analysis: When small language models outperform large language
models,” 2026. [Online]. Available: https://arxiv.org/abs/2603.21389
[18] S. Zhang, S. Lin, X. Wei, Y . Chen, P. Qian, S. Wang, and H. Xu, “Small
language model agents enable efficient and high-quality knowledge
mining,” 2026. [Online]. Available: https://arxiv.org/abs/2510.01427
[19] X. Li, Y . Ma, Y . Huang, X. Wang, Y . Lin, and C. Zhang, “Synergized
data efficiency and compression (SEC) optimization for large language
models,” in2024 4th International Conference on Electronic Information
Engineering and Computer Science (EIECS). IEEE, 2024, pp. 586–591.[20] X. Li, J. Cao, M. Wang, Y . Wu, L. Yan, Y . Zhou, Z. Sha, and
Y . Ma, “FAST: A synergistic framework of attention and state-space
models for spatiotemporal traffic prediction,” 2026. [Online]. Available:
https://arxiv.org/abs/2604.13453
[21] X. Li, Y . Lu, J. Cao, Y . Ma, Z. Li, and Y . Zhou, “CATCH: A modular
cross-domain adaptive template with hook,” inInternational Symposium
on Visual Computing. Springer, 2025, pp. 41–52.
[22] Y . Zhang, Z. Xiang, and H. Xu, “Performance-efficiency trade-offs
in human preference prediction: A comparative study of traditional
machine learning and large language models,” inProceedings of the 31st
IEEE Symposium on Computers and Communications (ISCC). IEEE,
2026.
[23] LangChain, Inc., “LangChain: Building applications with LLMs through
composability,” https://github.com/langchain-ai/langchain, 2024.
[24] LlamaIndex, “LlamaIndex: A data framework for LLM applications,”
https://github.com/run-llama/llama index, 2024.
[25] Y . Gao, Y . Xiong, X. Gao, K. Jia, J. Pan, Y . Bi, Y . Dai, J. Sun, and
H. Wang, “Retrieval-augmented generation for large language models:
A survey,”arXiv preprint arXiv:2312.10997, 2024.
[26] Y . Chen, P. Qian, S. Wang, S. Zhang, H. Xu, S. Lin, and X. Wei,
“Does RAG know when retrieval is wrong? diagnosing context
compliance under knowledge conflict,” 2026. [Online]. Available:
https://arxiv.org/abs/2605.14473
[27] H. Ye, S. Chen, Z. Zhong, C. Xiao, H. Zhang, Y . Wu, and F. Shen,
“Seeing through the conflict: Transparent knowledge conflict handling in
retrieval-augmented generation,” inProceedings of the AAAI Conference
on Artificial Intelligence, vol. 40, no. 40, 2026, pp. 34 423–34 431.
[28] P. Qian, S. Wang, X. Wang, Y . Chen, W. Xu, Q. Yu, S. Lin,
S. Zhang, J. You, and X. Wei, “Relevant is not warranted:
Evidence-force calibration for cited RAG,” 2026. [Online]. Available:
https://arxiv.org/abs/2605.28044
[29] J. Liu, J. Yang, X. Li, W. Yan, Y . Wu, P. Liang, and M. Yuan,
“Architecture matters more than scale: A comparative study of retrieval
and memory augmentation for financial QA under SME compute
constraints,” 2026. [Online]. Available: https://arxiv.org/abs/2604.17979
[30] Y . Qi, Y . Qi, and T. Wagh, “Coverage-aware web crawling for domain-
specific supplier discovery via a web–knowledge–web pipeline,” 2026.
[Online]. Available: https://arxiv.org/abs/2602.24262
[31] Y . Qi, X. Xu, and Y . Li, “When corrective hints hurt: Prompt design in
reasoner-guided repair of LLM overcaution on entailed negations under
OWL 2 DL,” 2026. [Online]. Available: https://arxiv.org/abs/2604.23398
[32] Z. Li, Z. Chen, H. Wen, Z. Fu, Y . Hu, and W. Guan, “Encoder: Entity
mining and modification relation binding for composed image retrieval,”
inProceedings of the AAAI Conference on Artificial Intelligence, vol. 39,
no. 5, 2025, pp. 5101–5109.
[33] Z. Li, Z. Fu, Y . Hu, Z. Chen, H. Wen, and L. Nie, “FineCIR: Explicit
parsing of fine-grained modification semantics for composed image
retrieval,”arXiv preprint arXiv:2503.21309, 2025.
[34] Z. Chen, M. Hu, J. Yi, and W. Sun, “Reinforcement learning for option
hedging: Static implied-volatility fit versus shortfall-aware performance,”
2026.
[35] Z. Zhang, R. Fu, Y . He, X. Shen, Y . Wang, X. Du, H. You, K. Jin,
J. Shi, and S. Fong, “FinSentLLM: Multi-LLM and structured semantic
signals for enhanced financial sentiment forecasting,” inICASSP 2026
– 2026 IEEE International Conference on Acoustics, Speech and Signal
Processing (ICASSP). IEEE, 2026, pp. 17 682–17 686.
[36] X. Qin, S. Li, Y . Cai, and L. Wang, “Enhancing counterfactual ex-
planations with feasibility and diversity,” in2025 IEEE International
Conference on Data Mining Workshops (ICDMW). IEEE, 2025, pp.
2310–2319.
[37] J. Yang, T. Liu, Y . T. Luo, T. Niu, P. Pang, A. Xiang, and Q. Yang,
“Exploring the application boundaries of LLMs in mental health: A
systematic scoping review,”Frontiers in Psychology, vol. 16, p. 1715306,
2026.
[38] Z. Cheng, L. Lai, Y . Liu, K. Cheng, and X. Qi, “Enhancing
financial report question-answering: A retrieval-augmented generation
system with reranking analysis,” 2026. [Online]. Available: https:
//arxiv.org/abs/2603.16877
[39] Z. Cheng, L. Lai, and Y . Liu, “Resolving the robustness-precision
trade-off in financial RAG through hybrid document-routed retrieval,”
2026. [Online]. Available: https://arxiv.org/abs/2603.26815
[40] W. Chen, L. Wu, Y . Hu, Z. Li, Z. Cheng, Y . Qian, L. Zhu, Z. Hu,
L. Liang, Q. Tang, Z. Liu, and H. Yang, “AutoNeural: Co-designing

vision-language models for NPU inference,” 2025. [Online]. Available:
https://arxiv.org/abs/2512.02924
[41] E. Strubell, A. Ganesh, and A. McCallum, “Energy and policy consid-
erations for deep learning in NLP,” inProceedings of the 57th Annual
Meeting of the Association for Computational Linguistics, 2019, pp.
3645–3650.
[42] D. Patterson, J. Gonzalez, Q. Le, C. Liang, L.-M. Munguia,
D. Rothchild, D. So, M. Texier, and J. Dean, “Carbon emissions and
large neural network training,”arXiv preprint arXiv:2104.10350, 2021.
[43] J. Johnson, M. Douze, and H. J ´egou, “Billion-scale similarity search
with GPUs,”IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–
547, 2021.
[44] G. V . Cormack, C. L. A. Clarke, and S. Buettcher, “Reciprocal rank
fusion outperforms condorcet and individual rank learning methods,”
inProceedings of the 32nd International ACM SIGIR Conference on
Research and Development in Information Retrieval. ACM, 2009, pp.
758–759.
[45] RAG Datasets contributors, “rag-mini-wikipedia: a small wikipedia-
passage benchmark for retrieval-augmented generation,” Hugging Face
dataset, https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia,
2024.
[46] Linq AI Research, “FinDER: a financial-domain question-answering
benchmark over SEC filings,” Hugging Face dataset, https://huggingface.
co/datasets/Linq-AI-Research/FinDER, 2024.
[47] L. Zheng, W.-L. Chiang, Y . Sheng, S. Zhuang, Z. Wu, Y . Zhuang, Z. Lin,
Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica,
“Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena,” in
Advances in Neural Information Processing Systems (NeurIPS) Datasets
and Benchmarks Track, 2023.
[48] D. Li, B. Jiang, L. Huang, A. Beigi, C. Zhao, Z. Tan, A. Bhattacharjee,
Y . Jiang, C. Chen, T. Wuet al., “From generation to judgment: Oppor-
tunities and challenges of LLM-as-a-Judge,” inProceedings of the 2025
Conference on Empirical Methods in Natural Language Processing,
2025, pp. 2757–2791.
[49] L. Lin, J. You, Y . Li, L. Lin, Y . Wang, Z. Zhang, and
M. Zheng, “Reflect-guard: Enhancing LLM safeguards against
adversarial prompts via logical self-reflection,” 2026. [Online].
Available: https://arxiv.org/abs/2605.24834
[50] Y . Wang, Y . Tang, Y . Qian, and C. Zhao, “VisualLeakBench:
Auditing the fragility of large vision-language models against
PII leakage and social engineering,” 2026. [Online]. Available:
https://arxiv.org/abs/2603.13385
[51] S. Wang, P. Qian, Y . Chen, J. You, X. Wang, X. Jiang,
L. Liu, H. Yu, and J. Xu, “When safe skills collide: Measuring
compositional risk in agent skill ecosystems,” 2026. [Online]. Available:
https://arxiv.org/abs/2606.00448