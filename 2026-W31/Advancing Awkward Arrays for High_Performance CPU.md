# Advancing Awkward Arrays for High-Performance CPU and GPU Processing

**Authors**: Ianna Osborne, Manasvi Goyal

**Published**: 2026-07-30 12:53:25

**PDF URL**: [https://arxiv.org/pdf/2607.28145v1](https://arxiv.org/pdf/2607.28145v1)

## Abstract
Awkward Array is a Python library for representing and processing nested, variable-length data that is widely used in high-energy physics. As HL-LHC analyses increasingly rely on accelerator hardware, efficient execution of irregular workloads has become essential. While dense numerical arrays map naturally to GPUs, nested and variable-length data structures remain significantly more difficult to accelerate because they require indirect indexing, segmented operations, and irregular memory access patterns. We present recent developments in the Awkward Array GPU backend, including CUDA implementations built on NVIDIA CUDA Core Compute Libraries (CCCL), optimized memory management, and segmented reduction algorithms for ragged arrays. These developments preserve the existing Python programming model while substantially improving GPU throughput on irregular workloads. We describe the backend architecture, automated validation framework, and benchmark results comparing CPU, CuPy, and CUDA implementations.

## Full Text


<!-- PDF content starts -->

Advancing Awkward Arrays for High-Performance CPU
and GPU Processing
Ianna Osborne𝑎,∗and Manasvi Goyal𝑏
𝑎Princeton University,
Princeton, NJ, USA
𝑏Harvard University,
Cambridge, MA, USA
E-mail:iosborne@princeton.edu
Awkward Array is a Python library for representing and processing nested, variable-length data
thatiswidelyusedinhigh-energyphysics. AsHL-LHCanalysesincreasinglyrelyonaccelerator
hardware,efficientexecutionofirregularworkloadshasbecomeessential. Whiledensenumerical
arraysmapnaturallytoGPUs,nestedandvariable-lengthdatastructuresremainsignificantlymore
difficult to accelerate because they require indirect indexing, segmented operations, and irregular
memory access patterns. We present recent developments in the Awkward Array GPU backend,
including CUDA implementations built on NVIDIA CUDA Core Compute Libraries (CCCL),
optimized memory management, and segmented reduction algorithms for ragged arrays. These
developments preserve the existing Python programming model while substantially improving
GPU throughput on irregular workloads. We describe the backend architecture, automated vali-
dation framework, and benchmark results comparing CPU, CuPy, and CUDA implementations.
23rd International Workshop on Advanced Computing and Analysis Techniques in Physics Research
(ACAT2025)
8–12 September 2025
Hamburg, Germany
∗Speaker
©Copyright owned by the author(s) under the terms of the Creative Commons Attribution-NonCommercial-NoDerivatives
4.0 International License (CC BY-NC-ND 4.0) All rights for text and data mining, AI training, and similar technologies for
commercial purposes, are reserved. ISSN 1824-8039 . Published by SISSA Medialab.https://pos.sissa.it/
arXiv:2607.28145v1  [cs.SE]  30 Jul 2026

Advancing Awkward Arrays for High-Performance CPU and GPU ProcessingIanna Osborne
1. Introduction
The High-Luminosity Large Hadron Collider (HL-LHC) [1] will significantly increase the
volumeandcomplexityofdataavailableforphysicsanalyses. Efficientprocessingofirregularevent
structureshasthereforebecomeincreasinglyimportantformodernhigh-energyphysicsworkflows.
Awkward Array provides a columnar representation for nested and variable-length data while
maintaining a NumPy-like interface [2]. It has become a key component of Python-based analysis
ecosystems, enabling efficient manipulation of detector and event data. Unlike dense multidi-
mensional arrays, irregular data structures require indirect indexing, variable-length segments,
and dynamic memory layouts, making efficient parallel execution challenging. As a result, many
optimization techniques developed for dense numerical arrays do not directly apply.
Thisworkpresentsrecentdevelopmentstowardhigh-performanceexecutionofAwkwardArray
acrossCPUandGPUbackends,includingCUDA[3]implementationsofirregularoperationsbased
on NVIDIA CUDA Core Compute Libraries (CCCL) [4], optimized GPU memory management,
an automated validation framework for ensuring backend consistency, and performance evaluation
on representative irregular workloads.
2. Backend Architecture
AwkwardArrayseparateshigh-leveloperationsfrombackend-specifickernelimplementations.
ThislayeredarchitectureallowsuserstoexecuteidenticalanalysiscodeoneitherCPUorGPUback-
ends. For example, switching execution from the CPU to the CUDA backend requires changing
only the backend argument, with no modifications to the analysis code. Backend implementations
expose the same high-level API while dispatching execution to backend-specific kernel libraries.
This separation allows new hardware backends to be developed independently without modifying
user analysis code. The CPU backend currently consists of approximately 144 specialized kernels.
Manyhigh-leveloperationsmapdirectlytoasingleCPUkernelimplementation. TheCUDAback-
end,however,reliesonCuPyformemorymanagementandkernellaunching[5]whilemaintaining
the same high-level Python API. The CPU backend serves as the correctness reference and shares
the same high-level execution model, illustrated in Figure 1.
3. GPU Acceleration with CUDA and CCCL
RecentdevelopmenteffortsfocusonimprovingGPUperformancethroughoptimizedmemory
management,increasedparallelism,andintegrationwithNVIDIACUDACoreComputeLibraries
(CCCL)[4]. GPUexecutionofirregularreductionsrequiresseveralpreprocessingstages,including
segment construction, temporary workspace allocation on the GPU, and invocation of segmented
reduction primitives. Consequently, a single logical Awkward operation may require multiple
CUDA kernel launches. Many GPU operations therefore follow a common execution pipeline that
constructssegmentoffsets,allocatesoutputandtemporarystorage,invokesCuPykernelsorCCCL
segmented primitives, and writes the resulting values back into Awkward layouts.
CCCL provides segmented GPU primitives that are well suited for these operations [4, 6, 7].
For example, segmented reductions compute independent reductions over each ragged list while
2

Advancing Awkward Arrays for High-Performance CPU and GPU ProcessingIanna Osborne
Figure 1:AwkwardArrayexecutionmodel. High-levelPythonoperationsaredispatchedtobackend-specific
CPU or CUDA implementations.
efficiently exploiting GPU parallelism. Replacing custom CUDA kernels with CCCL primitives
substantially simplifies the implementation while preserving the existing Python interface. For
the segmented minimum operation, approximately 70 lines of custom CuPy RawKernel code
are replaced by a three-line Python call to a CCCL segmented reduction primitive illustrated
in Figure 2. The work presented here focuses on segmented reduction operations, including
minimum, maximum, sum, product, and count. Together, these primitives provide the foundation
for accelerating a broad class of Awkward Array computations.
Figure 2:Current integration of CCCL within the GPU backend. The CCCL implementation simplifies
the segmented reduction code (left), while multiple software layers in the current backend contribute to
kernel-launch overhead (right).
4. Validation and Performance Evaluation
MaintainingidenticalbehavioracrossCPUandGPUimplementationsrequiresextensivevali-
dation. Kernel specifications are shared across backends, allowing identical correctness tests to be
generated for both CPU and CUDA implementations. The validation framework combines kernel-
level unit tests, backend consistency checks, integration tests for high-level Awkward operations,
andregressiontestsfornewlyintroducedCUDAkernels. Benchmarkswereperformedonasystem
equipped with an AMD EPYC 7H12 processor (128 physical cores) and an NVIDIA A100 PCIe
40GB GPU. The benchmark workload evaluates segmented reductions over irregular arrays with
3

Advancing Awkward Arrays for High-Performance CPU and GPU ProcessingIanna Osborne
increasing numbers of variable-length segments. Each benchmark was executed multiple times
after GPU warm-up, and the reported runtime corresponds to the minimum execution time.
Figure 3:Execution time and peak memory usage for the Awkwardak.minimplementation and the
underlying CCCL segmented reduction (left), and segmented mean performance comparing NumPy, CuPy,
andtheAwkwardCPUandGPUbackends(right). Atthelargestworkload,GPUak.minis5.3×fasterthan
the CPU implementation, while the low-level CCCL primitive achieves nearly 6×speedup.
Figure3(left)showsbothexecutiontimeandpeakmemoryusageforak.minandtheunderlying
CCCL segmented reduction. At the largest workload, GPUak.minachieves a 5.3×speedup over
theCPUimplementation,whilethelow-levelCCCLprimitivereachesnearly6×speedup,indicating
that most remaining overhead arises from the high-level Awkward execution pipeline. Peak GPU
memoryusageincreasesfromapproximately1GBto4GBacrossthebenchmarkrange,compared
with roughly 60MB to 2GB for the CPU backend, primarily owing to temporary storage required
for offset construction and segmented reduction operations. The segmented mean benchmark
(right) compares NumPy [8], CuPy [5], and the Awkward CPU and GPU backends. The GPU
implementation provides up to 12×speedup over the Awkward CPU backend and approximately
4×speedup over NumPy for the largest benchmark. Although it remains around 4×slower than
a hand-written CuPy implementation, the performance gap narrows with increasing problem size,
indicating that the remaining overhead is largely attributable to the generality of the high-level
Awkward execution model rather than the underlying GPU kernels.
5. Conclusions
AwkwardArrayprovidesanefficientframeworkforprocessingirregularscientificdataacross
both CPU and GPU architectures. Recent developments based on CUDA and CCCL substantially
improveGPUthroughputwhilepreservingthefamiliarPythonprogrammingmodel. ThenewGPU
backend combines optimized memory management with high-performance segmented primitives
for irregular data, enabling significant speedups over CPU implementations on large workloads.
Several challenges remain, including reducing kernel-launch overhead introduced by the current
execution pipeline and adapting to the evolving CCCL API. These developments establish the
foundation for compiler-driven optimization of high-level Awkward Array expressions, including
kernel fusion and portable execution across future accelerator platforms.
4

Advancing Awkward Arrays for High-Performance CPU and GPU ProcessingIanna Osborne
Acknowledgments
This work was supported by the National Science Foundation under Cooperative Agreements
OAC-1836650, PHY-2323298 and PHY-2121686.
References
[1] G. Apollinari et al.,High-Luminosity Large Hadron Collider (HL-LHC), World Scientific
(2015).
[2] J. Pivarski et al.,Awkward Array, Computing and Software for Big Science6(2022) 1.
[3] NVIDIA Corporation,CUDA Toolkit Documentation, https://docs.nvidia.com/cuda/.
[4] NVIDIA Corporation,CUDA Core Compute Libraries (CCCL),
https://github.com/NVIDIA/cccl.
[5] R. Okuta et al.,CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations, Pro-
ceedings of Workshop on Machine Learning Systems (2017).
[6] N. Merrill and M. Garland,Single-pass Parallel Prefix Scan with Decoupled Look-back,
NVIDIA Research (2016).
[7] N.BellandJ.Hoberock,Thrust: AProductivity-OrientedLibraryforCUDA,GPUComputing
Gems (2011).
[8] C. R. Harris et al.,Array programming with NumPy, Nature585(2020) 357.
5