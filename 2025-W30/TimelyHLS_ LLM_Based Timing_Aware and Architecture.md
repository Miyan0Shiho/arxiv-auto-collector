# TimelyHLS: LLM-Based Timing-Aware and Architecture-Specific FPGA HLS Optimization

**Authors**: Nowfel Mashnoor, Mohammad Akyash, Hadi Kamali, Kimia Azar

**Published**: 2025-07-23 22:08:15

**PDF URL**: [http://arxiv.org/pdf/2507.17962v1](http://arxiv.org/pdf/2507.17962v1)

## Abstract
Achieving timing closure and design-specific optimizations in FPGA-targeted
High-Level Synthesis (HLS) remains a significant challenge due to the complex
interaction between architectural constraints, resource utilization, and the
absence of automated support for platform-specific pragmas. In this work, we
propose TimelyHLS, a novel framework integrating Large Language Models (LLMs)
with Retrieval-Augmented Generation (RAG) to automatically generate and
iteratively refine HLS code optimized for FPGA-specific timing and performance
requirements. TimelyHLS is driven by a structured architectural knowledge base
containing FPGA-specific features, synthesis directives, and pragma templates.
Given a kernel, TimelyHLS generates HLS code annotated with both
timing-critical and design-specific pragmas. The synthesized RTL is then
evaluated using commercial toolchains, and simulation correctness is verified
against reference outputs via custom testbenches. TimelyHLS iteratively
incorporates synthesis logs and performance reports into the LLM engine for
refinement in the presence of functional discrepancies. Experimental results
across 10 FPGA architectures and diverse benchmarks show that TimelyHLS reduces
the need for manual tuning by up to 70%, while achieving up to 4x latency
speedup (e.g., 3.85x for Matrix Multiplication, 3.7x for Bitonic Sort) and over
50% area savings in certain cases (e.g., 57% FF reduction in Viterbi).
TimelyHLS consistently achieves timing closure and functional correctness
across platforms, highlighting the effectiveness of LLM-driven,
architecture-aware synthesis in automating FPGA design.

## Full Text


<!-- PDF content starts -->

(Special Session)
TimelyHLS: LLM-Based Timing-Aware and
Architecture-Specific FPGA HLS Optimization
Nowfel Mashnoor, Mohammad Akyash, Hadi Kamali, Kimia Azar
Department of Electrical and Computer Engineering (ECE), University of Central Florida, Orlando, FL 32816, USA
{nowfel.mashnoor, mohammad.akyash, kamali, azar }@ucf.edu
Abstract —Achieving timing closure and design-specific opti-
mizations in FPGA-targeted High-Level Synthesis (HLS) remains
a significant challenge due to the complex interaction between
architectural constraints, resource utilization, and the absence
of automated support for platform-specific pragmas. In this
work, we propose TimelyHLS, a novel framework integrating
Large Language Models (LLMs) with Retrieval-Augmented Gen-
eration (RAG) to automatically generate and iteratively refine
HLS code optimized for FPGA-specific timing and performance
requirements. TimelyHLS is driven by a structured architectural
knowledge base containing FPGA-specific features, synthesis
directives, and pragma templates. Given a kernel, TimelyHLS
generates HLS code annotated with both timing-critical and
design-specific pragmas. The synthesized RTL is then evaluated
using commercial toolchains, and simulation correctness is veri-
fied against reference outputs via custom testbenches. TimelyHLS
iteratively incorporates synthesis logs and performance reports
into the LLM engine for refinement in the presence of functional
discrepancies. Experimental results across 10 FPGA architectures
and diverse benchmarks show that TimelyHLS reduces the need
for manual tuning by up to 70%, while achieving up to 4× latency
speedup (e.g., 3.85× for Matrix Multiplication, 3.7× for Bitonic
Sort) and over 50% area savings in certain cases (e.g., 57% FF
reduction in Viterbi). TimelyHLS consistently achieves timing
closure and functional correctness across platforms, highlighting
the effectiveness of LLM-driven, architecture-aware synthesis in
automating FPGA design.
Index Terms —FPGA, Large Language Models, High-Level
Synthesis, Timing Closure, Retrieval-Augmented Generation
I. I NTRODUCTION
FPGAs provide a flexible platform for accelerating diverse
algorithms, but achieving timing closure on FPGAs remains a
persistent challenge [1]. Timing closure entails ensuring that
all signal paths meet the FPGA’s timing constraints (setup/hold
times, clock delays, etc.), and it is critical for correct operation
at the target clock frequency [2]. In practice, reaching timing
closure is an iterative and complex process, hindered by high
clock rates, large and interconnected designs, and physical
effects like routing delays [3]. Modern FPGA design flows
often demand manual tuning and optimization to meet timing,
especially when using High-Level Synthesis (HLS) tools to
generate hardware from C/C++ code [4], [5].
HLS tools such as Xilinx Vitis HLS [6] raise the design
abstraction to high-level code, but they still rely heavily on
user guidance to produce efficient, timing-compliant hardware
[7]. In particular, performance-critical optimizations (e.g. loop
pipelining, loop unrolling, memory partitioning) are typically
controlled by pragmas or directives embedded in the HLSsource [8]. Selecting the right combination of pragmas for a
given design and FPGA is a non-trivial task that often requires
deep hardware expertise [9].
Currently, there is a lack of automated support within HLS
tools for platform-specific pragmas (i.e. directives tailored to
a specific FPGA architecture or vendor). Each FPGA platform
introduces unique resources and constraints, which often ne-
cessitates different pragmas or coding styles. Designers must
manually adapt and tune their HLS code for each target device,
since a pragma that works well (or is even recognized) on
one toolchain may not apply on another [10]. For instance,
a designer targeting Xilinx FPGAs might use the PIPELINE
pragma to improve loop initiation intervals, while Intel’s
HLS compiler requires a different pragma (ivdep) or coding
convention to achieve similar results [11].
To ease the challenges of HLS optimization, researchers
have developed automated methods like Design Space Ex-
ploration (DSE) frameworks [12], [13], which search large
pragma spaces to find configurations with good Quality of
Results (QoR). Tools such as AutoDSE [14] use heuristic
searches but require many HLS runs, making them slow.
Analytical methods, like formulating pragma selection as
a non-linear optimization problem [8], reduce this cost by
pruning poor choices. ML-based tools (e.g., HARP [15])
predict performance to guide optimizations more efficiently.
However, models often struggle to generalize to new designs
or architectures [10]. Despite progress, fully automated, widely
adopted solutions remain lacking, especially for timing-driven
optimization, leaving designers to rely on manual tuning.
Recent advances in Large Language Models (LLMs) have
opened new possibilities in automating HLS optimization for
FPGA design [7], [16], [17]. Tools like LIFT [16] fine-tune
LLMs with Graph Neural Network (GNN) analyses to insert
pragmas, achieving up to 3.5× speedup over prior methods.
HLSPilot [7] uses in-context learning and retrieval from
vendor docs, combining DSE and profile-guided refinement
to produce results comparable to expert designs. However,
prompting general LLMs without domain-specific context can
significantly degrade performance [18], demonstrating the
need for integrated domain knowledge and feedback.
Despite recent progress, HLS-based FPGA design continues
to face two critical challenges: (i) ensuring that synthesized
designs meet strict timing requirements , which is complicated
by factors like deep logic pipelines, routing congestion, and
critical paths that are difficult to predict and optimize atarXiv:2507.17962v1  [cs.CR]  23 Jul 2025

a high level; and (ii) adapting optimizations totheunique
characteristics ofeach FPGA architecture , where vendor-
specific toolchains, resource constraints, and low-level features
often necessitate custom pragma configurations and design
patterns that do not generalize across platforms. These issues
contribute to a persistent “closure gap” that current tools
struggle to overcome without huge manual intervention.
To address these limitations, we propose TimelyHLS, a
framework that combines LLMs with retrieval-augmented gen-
eration (RAG) and iterative refinement. TimelyHLS is guided
by a structured knowledge base encoding FPGA-specific fea-
tures, pragmas, and optimization heuristics. The LLM queries
this knowledge during inference to generate HLS code tailored
to the target architecture. This initial generation is followed
by an iterative loop: synthesized RTL is evaluated using com-
mercial tools, testbenches verify functional correctness, and
synthesis logs are fed back into the model. The LLM then re-
vises the code based on this feedback until the design achieves
timing closure and correctness1. We evaluate TimelyHLS on
various FPGA devices and benchmark kernels. Results show
that TimelyHLS reduces manual iterations, consistently meets
timing constraints, and delivers performance comparable to
hand-tuned designs. Our contributions include:
(i) LLM + RAG for FPGA HLS: We propose TimelyHLS ,
the first framework to integrate an LLM with RAG for FPGA-
specific HLS code generation. By grounding the model in a
curated knowledge base of FPGA-specific architectural fea-
tures, synthesis directives, and pragma strategies, TimelyHLS
generates HLS code tailored to the target device.
(ii) Iterative Refinement with Tool Feedback: TimelyHLS
employs an iterative refinement loop where synthesis reports,
timing analysis, and functional verification feedback are rein-
tegrated into the LLM. This enables the model to progressively
resolve timing violations and improve design quality in a
closed-loop, minimizing the need for manual tuning.
(iii) Effective Timing Closure/Optimization: Extensive ex-
periments show that TimelyHLS achieves timing closure at an
acceptable oveheard (area) across diverse FPGAs while signif-
icantly reducing manual intervention. It delivers performance
and overall QoR (latency, area, timing) on par with, and in
many cases exceeding, expert-optimized HLS designs.
II. R ELATED WORKS
Early work on automating HLS optimization used heuristic
search and static modeling to explore the design space of
synthesis directives (pragmas) [19]2. OpenTuner [19], a gen-
eral auto-tuning framework adapted for HLS that orchestrates
many search strategies (greedy, genetic algorithms, simulated
annealing, etc.) via a multi-armed bandit approach. By dy-
namically choosing among strategies, OpenTuner effectively
1While TimelyHLS mainly focuses on improving timing closure with
functional correctness, the same flow can be used for power/area efficiency
(if targeted and needed by the design specification).
2Exhaustive search is infeasible as the combination of directives, e.g., loop
unrolling, pipelining, memory partitioning, etc. grows exponentially, so studies
applied heuristics to guide DSE without brute force.navigated large pragma spaces, often finding better solutions
than any single algorithm alone. AutoDSE [14], a DSE tool
that iteratively tunes one pragma at a time, always addressing
the current performance bottleneck. By focusing on the most
critical optimization step-by-step, AutoDSE achieved expert-
level results with far fewer directives. Many DSE frameworks
employed simulated annealing [20] or hill-climbing [21] to
automate pragma selection. These rule-based searches marked
a big step in reducing engineering effort. However, purely
heuristic approaches can miss global optima or get stuck in
local optima, especially in very large parameter spaces.
Complementing above search techniques, analytical model-
ing approaches sought to speed up exploration by predicting
HLS outcomes without full synthesis. Tools like Lin-Analyzer
[22], COMBA [23], and more recently ScaleHLS [24], use
static code analysis (e.g., loop dependence graphs, pipeline
initiation interval formulas) to estimate a design’s latency and
resource usage under different pragma choices. By evaluating
design points with these mathematical models, unpromising
configurations can be pruned orders of magnitude faster than
actually synthesizing them. In practice, these models often
need re-tuning for each tool or hardware target, and they
handle only a subset of possible pragmas or code structures (to
keep the formulas tractable). As a result, analytical estimation
is usually used as a component in a larger system.
Recent work has explored learning-based methods. ML
frameworks like HARP [15] use surrogate models, often
GNNs to predict performance from code and pragma config-
urations, enabling rapid DSE without full synthesis. Bayesian
optimization (BO) tools like Sherlock [17] further improve
efficiency by prioritizing high-potential candidates through
probabilistic modeling, excelling in multi-objective tuning.
Reinforcement learning (RL) frames HLS as sequential
decision-making, where agents learn to apply transformations
or insert pragmas to improve performance. Methods like
AutoAnnotate [25] report up to 4× speedup, though RL faces
challenges with large design spaces and training times. Hybrid
solutions like AutoHLS [26] combine neural prediction with
BO to prune poor candidates, achieving up to 70× speedup.
LLM-based approaches, which are heavily in use in Reg-
ister Transfer Language (RTL) design and verification [27]–
[29], aim to generate optimized HLS code directly. While
general-purpose models like GPT-4 struggle without domain
grounding, LIFT [16] improves results by fine-tuning on 40k+
annotated HLS samples with graph-based features, achieving
3.5× speedup over baselines. However, it lacks adaptability
to new platforms and toolchains. HLSPilot [7] uses RAG,
guiding the LLM with relevant HLS examples and vendor-
specific rules at inference time. It enables structural code
transformations and rivals expert performance.
Building on this, proposed TimelyHLS enhances adapt-
ability by dynamically querying an evolving, platform-aware
knowledge base during generation. This ensures that all opti-
mizations are context-aware, verifiable, and tailored to current
toolchains, avoiding hallucinated or outdated directives.

HLS Code
First Verification Stage
•Select benchmark (e.g., 
matrix mul , FIR filter)
•Load C++ and testbench
•Retrieve FPGA specs  (DSPs, 
LUTs, BRAMs)
•Format specs into 
structured input (e.g., 
JSON/dictionary)
•Tag benchmark with known 
synthesis issuesData 
Extract
Open -source•Select HLS benchmark and testbench
•Use RAG to fetch constraints
•Generate HLS code with pragmas
•Align code with target FPGA 
FPGA 
Specs
•Compile HLS code in Vitis
•Run C++ testbench
•Check functional correctness
HLS Code
•Export RTL from Vitis
•Synthesize with Vivado
•Run Verilog testbench
.v code
Vivado
Second Verification StagePrompting LLM
•Parse Vitis logs
•Detect Syntax/runtime errors
•Identify failed pragmas or loop
•Prompts LLM for Correction•Extract timing & synthesis logs
•Detect WNS/TNS violations
•Check RTL Sim Correctness
•Send Issues to LLM for fixFirst Feedback Second FeedbackHLS and RTL code
ready for FPGA deploymentFig. 1: Overview of TimelyHLS Framework.
III. O VERVIEW OF TIMELY HLS
The TimelyHLS framework automates timing-aware HLS
code generation through a combination of LLM, RAG, and
iterative synthesis feedback. Fig. 1, shows the top view of
TimelyHLS framework. The workflow operates in two main
verification stages, HLS-level and RTL-level, and leverages
FPGA-specific knowledge to guide the model toward platform-
compliant and timing-closure-friendly designs. To accomplish
optimization for speed-up, the TimelyHLS framework consists
of four main components, which are as follows:
A. Dataset Collection
To establish (and evaluate) our framework, we curated a
dataset of real-world HLS design examples gathered from
open-source repositories, e.g., CHStone [30], LegUp bench-
marks [31], and MachSuite [32]. These repositories provide
diverse C/C++ programs widely used in HLS, covering a
range of computational domains. We selected 10 representative
HLS applications that reflect common optimization bottle-
necks in FPGA synthesis, including timing violations, long
critical paths, inefficient pipelining, and suboptimal resource
allocation. A detailed description of each benchmark and its
associated synthesis challenge is provided in Table I.
For each benchmark, we developed corresponding HLS
C++ source files as well as custom testbenches to verify
functional correctness during simulation and synthesis. All
designs were compiled and analyzed using the Xilinx Vitis
HLS toolchain. To ensure architectural diversity and practical
relevance, we evaluated each design across 10 distinct FPGA
targets spanning multiple device families. For each FPGA
architecture, we collected synthesis reports, timing summaries,
and resource utilization logs. Additionally, we captured the
output of testbench simulations to verify functional correct-
ness. This comprehensive dataset, which includes source code,
testbenches, and tool-generated logs across multiple architec-
tures, forms the basis for evaluating the effectiveness of our
proposed LLM-based (prompting) TimelyHLS framework.
B. Prompting LLM for Initial Code Generation
For each sample in our dataset, we craft a task-specific
prompt that describes the functionality and performance ob-
jectives of the design (e.g., loop behavior, target throughput, orTABLE I: HLS Selected Benchmarks with Synthesis Chal-
lenges (Timing-wise) Targeted and Used in TimelyHLS.
Application Optimization Challenge
Matrix Multiplication Long critical path due to nested loops; loop pipelining
inefficiencies.
Convolution Timing violations caused by inefficient memory access
and computation overlap.
Vector Dot Product Underutilized resources and insufficient parallelism.
Vector Addition Suboptimal loop unrolling with moderate timing slack.
Bitonic Sort Deep logic pipelines leading to routing congestion and
critical path delays.
Viterbi Decoder Control dependencies causing resource contention.
Adaptive FIR Filter (LMS) Feedback loop latency and failed timing closure due to
iteration dependencies.
CORDIC Algorithm Inefficient pipelining due to iterative data dependencies.
Matrix-Vector Multiplication Memory partitioning bottlenecks leading to timing
degradation.
Needleman–Wunsch (DP) Irregular memory access patterns causing critical path
delay and low throughput.
memory access constraints). This prompt is paired with target
FPGA metadata (i.e. device family, number of DSPs, BRAMs,
LUTs, and timing constraints as part of a RAG pipeline). The
architectural specifications are embedded from datasheets and
vendor tool documentation into a structured knowledge base.
We then query the LLM (e.g., Code LLaMA or GPT-4) with
the prompt and retrieved FPGA constraints to generate HLS-
compliant C/C++ code. The model is expected to insert rel-
evant synthesis directives (pragmas) such as #pragma HLS
pipeline ,unroll , orarray_partition , aligned with
the resource capabilities of the target device.
C. HLS-Level Verification and Correction
The generated HLS code is compiled and simulated using
Xilinx Vitis HLS3, paired with the testbench we previously
crafted for each benchmark. This first stage ensures that the
model’s output is functionally correct and synthesizable at the
C level. If the compilation fails or the functional simulation
does not produce expected results, we extract relevant infor-
mation from Vitis logs (e.g., syntax errors, resource binding
issues, or pipeline depth violations) and return this feedback
3All prompts and scripts are configurable (parametrized) to be reusable for
different vendors (to be easily used with different toolsets.

to the LLM in the form of an augmented prompt. The LLM is
asked to revise its output to address the specific failures. This
process is repeated iteratively until the design passes both
HLS synthesis and functional simulation.
D. RTL-Level Verification and Timing Evaluation
Once the HLS design passes the first verification stage,
we export the RTL (Verilog) output and proceed with the
second stage using Xilinx Vivado. In this phase, we gener-
ate the corresponding Verilog testbenches and synthesize the
design for the selected FPGA target. Vivado’s post-synthesis
reports are then used to evaluate timing closure (e.g., Worst
Negative Slack (WNS), Total Negative Slack (TNS)), resource
utilization, and syntactic validity of the RTL. We also simulate
the design at the RTL level using the generated testbenches
to verify behavioral equivalence with the HLS-level outputs.
If the synthesizer (i.e., Xilinx Vivado) fails to synthesize the
design or simulation results deviate from the expected output,
we extract detailed logs, including synthesis errors, critical
path reports, and functional mismatches and pass them as
feedback to the LLM. This closes the second loop of iterative
refinement, allowing the model to correct deeper architectural
or low-level issues not observable during HLS.
This two-stage refinement loop continues until the design
satisfies these criteria: (i) Passes functional simulation in both
HLS and RTL levels; (ii) Is synthesizable by Vivado for
the target FPGA architecture; and (iii) Meets timing closure
requirements with no negative slack.
By integrating FPGA-specific architectural guidance into
generation and leveraging compiler logs as feedback, Time-
lyHLS transforms the traditional trial-and-error-based HLS
optimization into an automated, LLM-driven pipeline.
IV. E XPERIMENTAL SETUP
A. Experimental Environment and Tools
All experiments were conducted on a Linux-based devel-
opment environment (Ubuntu 24.04.2 LTS) using Xilinx Vitis
HLS (Version 2024.2) for HLS and Vivado (Version 2024.2)
for RTL synthesis and implementation. The experimental in-
frastructure consisted of 13th Gen Intel(R) Core(TM) i7-13700
Processor and 32GB of memory capacity to accommodate
the synthesis flows, parallel DSE, and iterative LLM infer-
ence processes. We evaluated TimelyHLS across 10 diverse
FPGA devices, including Artix-7, Spartan-7, Zynq, and Virtex
UltraScale+, to ensure comprehensive architectural coverage.
The selected devices represent a wide spectrum of resource
capacities, from low-cost embedded solutions to high-end
accelerators listed in Table II. All designs were synthesized
with constraints of achieving the maximum frequency.
B. Large Language Model Configuration
We conducted comparative experiments using two state-
of-the-art LLMs: OpenAI GPT-4 and Anthropic Claude-3.5-
Sonnet, both accessed via their respective APIs with defaultTABLE II: Targeted FPGA Families and their Applications.
FPGA Family Part Number(s) Typical Applications
Zynq xc7z020-clg484-1 Embedded applications
Zynq UltraScale+ xczu3eg-sbva484-1-e Heterogeneous computing
Artix/Kintex-7 xc7a200tfbg676-2, Cost-optimized designs
xc7k325tffg676-2
Spartan-7 xc7s50-ftgb196-2 Ultra-low-cost applications
Virtex UltraScale+ xcvu9p-flgb2104-2-e, High-performance computing
xcvu11p-flga2577-1-e,
xcvu9p-flgb2104-1-e
Kintex UltraScale+ xck26-sfvc784-2LV-c Balanced performance-power
Versal AI Edge xave2602-nsvh1369-1LJ-i-L AI/ML acceleration
temperature settings (0.7) to balance creativity and determin-
ism in code generation. The RAG knowledge base was con-
structed by extracting and structuring information from official
FPGA datasheets, vendor HLS user guides, and architectural
reference manuals for each target device family.
V. R ESULTS AND EVALUATION
To evaluate TimelyHLS, the benchmarks span a variety
of domains, e.g., linear algebra, signal processing, sorting,
and dynamic programming, each presenting unique challenges
such as long critical paths, deep loop dependencies, or inef-
ficient memory access. The evaluation focuses on four key
metric categories: timing and performance, resource utiliza-
tion, loop-level optimizations, and structural design changes.
A. Performance Improvements
TimelyHLS demonstrated significant performance gains
across several benchmarks. As illustrated in Fig. 2, speedup
values on Artix-7 reached up to 4×, with applications like
Matrix Multiplication, LMS Filter, and Bitonic Sort showing
the most notable improvements. These gains are primarily
the result of architecture-specific pipelining strategies and
effective insertion of pragmas such as pragma HLS pipeline
and loop unrolling. These changes reduce the initiation interval
(II) and enable higher parallelism.
B. Timing and Latency Analysis
The timing closure performance of TimelyHLS is substan-
tiated by latency and slack results presented in Table III.
Across multiple benchmarks, the framework not only resolved
0 0.5x 1x 1.5x 2x 2.5x 3x 3.5x 4xSpeed -upViterbi (up to 2.6x)
VecDoc  (up to 1.2x)
VecAdd  (up to 1.65x)
NW (up to 2.8x)
MatVecDot  (up to 1.15x) MatMul  (up to 3.85x)
Conv (up to 2.7x)
CORDIC (up to 2.2x)
Bitonic  (up to 3.7x)
LMS (up to 2.55x)Benchmarks under Test
Fig. 2: Performance Speedup Ratios for Artix 7.

TABLE III: Latency Comparison: Base vs. TimelyHLS.
Bitonic CORDIC Mat-Vec Mat Mul VecAdd VecDot Viterbi
Base (ns) 0.1 2.7 0.54 -0.08 2.7 -0.54 2.7
TimelyHLS (ns) 0.1 2.7 0.62 0.1 2.7 0.54 2.7
negative slack violations but also preserved optimal latency
characteristics when further improvement was infeasible or
unnecessary. As shown, in the Matrix Multiplication bench-
mark, the baseline design exhibited negative slack, a direct
result of deep loop nesting and non-optimized memory access,
which introduced excessive combinational delay. TimelyHLS
resolved this by restructuring loop hierarchies and selectively
applying pipelining with loop unrolling and array partitioning,
resulting in a fully timing-closed implementation with zero
slack. Similarly. for the Vector Dot Product, the baseline
implementation underutilized available DSP resources and suf-
fered from inefficient loop scheduling, leading to intermittent
timing violations. TimelyHLS addressed this by rebalancing
the loop-carried dependencies and improving the data flow
via pipelining and unrolling.
C. Resource Utilization and Performance Tradeoffs
Tables IV and V reflect the impact of TimelyHLS on
balancing hardware area consumption, measured through flip-
flop (FF) and lookup table (LUT) usage, with improvements
in latency and timing closure. The observed trade-offs under-
score the framework’s architecture-aware optimization strat-
egy, where moderate increases in resource utilization achieve
the latency improvements, while in other cases, aggressive area
savings are prioritized when latency is already near-optimal.
In benchmarks such as vector addition and matrix-vector
multiplication, TimelyHLS used deeper pipelining and mem-
ory partitioning to eliminate bottlenecks and sustain loop
throughput. While this increased LUT usage by 50–65%,
improvements in timing slack and loop initiation intervals
justified the added area (deliberate trade-off: reducing latency
with parallelism and logic duplication increases area). On
the other hand, for benchmarks like the Viterbi Decoder,
TABLE IV: Flip-Flop (FF) Usage across FPGA Families.
Benchmark Family Part FF Used FF Change (%)
Viterbi Artix-7 xc7a200t 247 -57.34
Viterbi Spartan-7 xc7s50 247 -57.34
CORDIC Spartan-7 xc7s50 329 -12.27
Vec Dot Artix-7 xc7a200t 572 22.75
Vec Add Zynq xc7z020 2479 38.11
Vec Add Spartan-7 xc7s50 2468 39.91
Vec Add Virtex-U+ xcvu11p 2479 38.11
TABLE V: LUTs Usage across FPGA Families.
Benchmark Family Part LUTs Used LUTs Change (%)
Viterbi Dec. Artix-7 xc7a200t 619 -48.24
Viterbi Dec. Spartan-7 xc7s50 647 -47.91
Vec. Dot Prod. Artix-7 xc7a200t 669 40.84
Vec. Addition Spartan-7 xc7s50 2460 46.43
Viterbi Versal AI xcvc2602 479 0.00
Vec. Addition Virtex-U+ xcvu11p 2578 52.82
Viterbi Dec. Virtex-U+ xcvu11p 1946 58.47
Mat-Vec Mult. Artix-7 xc7a200t 3132 65.18
Mat-Vec Mult. Spartan-7 xc7s50 3132 65.18TABLE VI: Impact of TimelyHLS on Loop II.
Project Base TimelyHLS Reason
Matrix Multiplication 16 1–2 Faster throughput.
Bitonic Sort Non-pip. All II=1 Loop pipelined.
Matrix-Vector Mult. Not pip. Pip. (II=1) Added pipelining.
Vector Dot Product 1 1 Fixed timing.
Vector Addition 1–2 1 Higher BRAM usage.
CORDIC 2 Unrolled Loop unrolled.
TimelyHLS reduced FF and LUT usage by over 50%. This
implies that the original design had redundant control logic
or non-optimized datapath that could be compacted without
affecting latency. These observations highlight the context-
sensitive nature of the latency–area trade-off. TimelyHLS
adapts its optimization to the designs’ needs, aggressively
optimizing for performance when needed, and concentrating
on compactness when further gains are unnecessary. All final
designs met FPGA resource constraints, showing both adapt-
ability and architectural feasibility.
D. Loop-Level Optimization Strategies
Table VI shows the performance impact of TimelyHLS
on key look initiation interval (II). As shown, in Matrix
Multiplication, the initiation interval (II) was reduced from
16 to 1–2, significantly enhancing throughput. Bitonic Sort,
previously limited by non-pipelined loops, was fully pipelined
with II=1, resolving the primary performance bottleneck. Sim-
ilarly, Matrix-Vector Multiplication and CORDIC benefited
from loop unrolling and pipelining, leading to reduced latency
and improved data movement. These results indicate that
TimelyHLS effectively identifies loop-carried dependencies
and applies appropriate directives to maximize hardware uti-
lization and scheduling efficiency. An example of such impact
has shown in Fig. 3 that represents code snippet of Vector Add
(base vs. TimelyHLS implementation).
E. Structural Optimization across FPGA Architectures
Evaluation across FPGA families (e.g., Artix-7) shows
that TimelyHLS often restructures modules and interfaces to
enhance performance. For instance, in Matrix Multiplication,
latency dropped significantly, from 16,531 to 4,277 cycles,
accompanied by the use of AXI interfaces for improved mod-
ularity. Similarly, Bitonic Sort saw a 3.6× speedup, enabled
by the addition of sparsemux units for faster scheduling. In
contrast, Vector Dot Product saw a modest latency increase
(519 to 640 cycles) but a dramatic rise in DSP usage (5 to
voidvector_add (
constunsigned int*in1,
constunsigned int*in2,
unsigned int*out,
 intsize)
{
#pragma HLSINTERFACE m_axiport=in1 depth=1024
#pragma HLSINTERFACE m_axiport=in2 depth=1024
#pragma HLSINTERFACE m_axiport=out depth=1024
#pragma HLSINTERFACE s_axilite port=size
#pragma HLSINTERFACE s_axilite port=return
 for(inti=0;i<size;i++){
out[i]=in1[i]+in2[i];
}}voidvector_add (
constunsigned int*in1,
constunsigned int*in2,
unsigned int*out,
 intsize)
{
#pragma HLSINTERFACE m_axiport=in1 offset=slave bundle= gmemdepth=1024
#pragma HLSINTERFACE m_axiport=in2 offset=slave bundle= gmemdepth=1024
#pragma HLSINTERFACE m_axiport=out offset=slave bundle= gmemdepth=1024
#pragma HLSINTERFACE s_axilite port=size bundle=control
#pragma HLSINTERFACE s_axilite port=return bundle=control
unsigned intlocal_in1[1024];
unsigned intlocal_in2[1024];
unsigned intlocal_out [1024];
#pragma HLSARRAY_PARTITION variable=local_in1 cyclic factor=16 dim=1
#pragma HLSARRAY_PARTITION variable=local_in2 cyclic factor=16 dim=1
#pragma HLSARRAY_PARTITION variable= local_out cyclic factor=16 dim=1
//Burstreadinputarrays
 for(inti=0;i<size;i++){
#pragma HLSPIPELINE II=1
local_in1[ i]=in1[i];
local_in2[ i]=in2[i];
}(a) HLS Code
(b) TimelyHLS  Code//Perform vector addition
 for(inti=0;i<size;i++){
#pragma HLSPIPELINE II=1
local_out [i]=local_in1[ i]+local_in2[ i];
}
//Burstwriteoutput array
 for(inti=0;i<size;i++){
#pragma HLSPIPELINE II=1
out[i]=local_out [i];
}} //Perform ...
Fig. 3: TimelyHLS Code Example with Optimization.

90% 91% 92% 93% 94% 95% 96% 97% 98% 99% 100%FPGA Families
Pass Rate (Synthesis and Functional)Kintex  Ultrascale +VersalZynq
Virtex  Ultrascale +
Spartan 7
Artix  7Fig. 4: Code Generation Success across FPGA Families.
160), indicating operator duplication to meet timing. Overall,
structural changes via TimelyHLS often led to performance
gains, though at the cost of complexity or resource overhead.
F . Architectural Adaptability and Design Quality
TimelyHLS demonstrates strong architectural adaptability,
consistently generating synthesizable HLS code across a wide
range of FPGA families, from low-cost devices like Spartan-
7 and Artix-7 to high-end platforms such as Zynq Ultra-
Scale+ and Virtex UltraScale+ (see Fig. 4). Most benchmarks
compiled successfully across nearly all devices, indicating
broad generalizability without the need for manual retargeting.
Failures were mostly limited to complex designs with irregular
memory access or feedback-heavy loops, which struggled on
resource-constrained FPGAs. In contrast, high-end devices like
the Virtex UltraScale+ consistently supported even the most
challenging designs. This suggests that TimelyHLS adapts
its code generation to match platform capabilities—favoring
compact, efficient structures on smaller FPGAs and leveraging
advanced features (e.g., AXI interfacing, etc.) on larger ones.
VI. C ONCLUSION AND FUTURE WORK
This paper presented TimelyHLS, a framework that com-
bines large language models, retrieval-augmented generation,
and synthesis feedback to automate timing-aware, architecture-
specific HLS code generation for FPGAs. By leveraging a
structured knowledge base, TimelyHLS produces functionally
correct, synthesizable designs that meet timing constraints
across a broad range of FPGA architectures. Experiments show
that TimelyHLS generalizes well to both low-end and high-
performance platforms, adapting its optimization strategies to
balance latency, area, and throughput. It automates complex
design transformation, while maintaining high synthesis suc-
cess rates even on resource-limited devices. Future work will
extend the framework to support multi-objective optimization
(e.g., power, area, performance trade-offs), integrate additional
toolchains and hardware platforms, and improve model gen-
eralization through fine-tuning and curriculum learning.
REFERENCES
[1] Q. Yanghua et al. ,“Improving classification accuracy of a machine
learning approach for FPGA timing closure” , in2016 IEEE 24th FCCM .
IEEE, 2016, pp. 80–83.
[2] J. Cong et al. ,”FPGA HLS today: successes, challenges, and opportu-
nities” ,ACM TRETS , vol. 15, no. 4, pp. 1–42, 2022.
[3] M. W. Numan et al. ,“Towards Automatic High-Level Code Deployment
on Reconfigurable Platforms: A Survey of High-Level Synthesis Tools
and Toolchains” ,IEEE Access , vol. 8, pp. 174692–174722, 2020.[4] E. Ustun et al. ,“LAMDA: Learning-assisted multi-stage autotuning for
FPGA design closure” , inFCCM . IEEE, 2019, pp. 74–77.
[5] Q. Sun et al. ,“Correlated multi-objective multi-fidelity optimization for
HLS directives design” ,ACM TODAES , vol. 27, no. 4, pp. 1–27, 2022.
[6] AMD, “AMD Vitis HLS, High-Level Synthesis Tool,”
https://www.amd.com/en/products/software/adaptive-socs-and-
fpgas/vitis/vitis-hls.html.
[7] C. Xiong et al. ,“HLSPilot: LLM-based high-level synthesis” , inProc.
of 43rd IEEE/ACM ICCAD , 2024, pp. 1–9.
[8] S. Pouget et al. ,“Automatic hardware pragma insertion in high-level
synthesis: A non-linear programming approach” ,ACM Tran. on Design
Auto. of Elec. Sys. , vol. 30, no. 2, pp. 1–44, 2025.
[9] Y . Chi et al. ,“Democratizing domain-specific computing” ,Commun. of
the ACM , vol. 66, no. 1, pp. 74–85, 2022.
[10] S. Lahti et al. ,”High-level Synthesis for FPGAs—A Hardware Engi-
neer’s Perspective” ,IEEE Access , 2025.
[11] AMD, #pragma HLS pipeline —Vitis HLS User Guide
(UG1399) , AMD Inc., Santa Clara, CA, USA. [Online]. Available:
https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-pipeline
[12] S. Liu et al. ,“Accelerating FPGA prototyping through predictive model-
based HLS design space exploration” , inDAC , 2019, pp. 1–6.
[13] L. Ferretti et al. ,“Lattice-Traversing Design Space Exploration for High
Level Synthesis” , in2018 IEEE 36th ICCD , 2018, pp. 210–217.
[14] A. Sohrabizadeh et al. ,“AutoDSE: Enabling software programmers to
design efficient FPGA accelerators” ,ACM TODAES , vol. 27, no. 4, pp.
1–27, 2022.
[15] A.Sohrabizadeh et al. ,“Robust GNN-based representation learning for
HLS” , in2023 IEEE/ACM ICCAD , IEEE, 2023, pp. 1–9.
[16] N. Prakriya et al. ,”LIFT: LLM-based pragma insertion for HLS via
GNN supervised fine-tuning” ,arXiv, preprint arXiv:2504.21187 , 2025.
[17] Q. Gautier et al. ,“Sherlock: A multi-objective design space exploration
framework” ,ACM TODAES , vol. 27, no. 4, pp. 1–20, 2022.
[18] B. Peng et al. ,“Check your facts and try again: Improving large
language models with external knowledge and automated feedback” ,
arXiv, preprint arXiv:2302.12813 , 2023.
[19] J. Ansel et al. ,“Opentuner: An extensible framework for program
autotuning” , in Proc. 23rd Int. Conf. on Par. arch. and comp. , 2014,
pp. 303–316.
[20] Z. Ding et al. ,“Efficient task transfer for HLS DSE” , in Proc. 43rd
IEEE/ACM ICCAD , 2024, pp. 1–9.
[21] N. K. Pham et al. ,“Exploiting loop-array dependencies to accelerate
the design space exploration with high level synthesis” , in2015 IEEE
DATE . IEEE, 2015, pp. 157–162.
[22] G. Zhong et al. ,“Lin-analyzer: A high-level performance analysis tool
for FPGA-based accelerators” , inProc. 53rd DAC , 2016, pp. 1–6.
[23] J. Zhao et al. ,“COMBA: A comprehensive model-based analysis frame-
work for high level synthesis of real applications” , in2017 IEEE/ACM
ICCAD . IEEE, 2017, pp. 430–437.
[24] H. Ye et al. ,“ScaleHLS: A new scalable high-level synthesis framework
on multi-level intermediate representation” , in2022 IEEE HPCA , IEEE,
2022, pp. 741–755.
[25] H. Shahzad et al. ,“Autoannotate: Reinforcement learning based code
annotation for high level synthesis” , in2024 25th ISQED , IEEE, 2024,
pp. 1–9.
[26] M. R. Ahmed et al. ,“AutoHLS: Learning to Accelerate Design Space
Exploration for HLS Designs” , in 2023 IEEE 66th MWSCAS , IEEE,
2023, pp. 491–495.
[27] M. Akyash et al. ,“Evolutionary large language models for hardware
security: A comparative survey” , inProc. of the great lakes symposium
on VLSI 2024 , 2024, pp. 496–501.
[28] N. Mashnoor et al. ,“LLM-IFT: LLM-Powered Information Flow Track-
ing for Secure Hardware” , in2025 IEEE 43rd VTS . IEEE, 2025, pp.
1–5.
[29] M. Akyash et al. ,“RTL++: Graph-enhanced LLM for RTL Code
Generation” ,arXiv , preprint arXiv:2505.13479, 2025.
[30] Y . Hara et al. ,“CHStone: A benchmark program suite for practical
C-based high-level synthesis” , in2008 IEEE ISCAS . IEEE, 2008, pp.
1192–1195.
[31] A. Canis et al. ,“LegUp: An open-source high-level synthesis tool for
FPGA-based processor/accelerator systems” ,ACM TECS , vol. 13, no.
2, pp. 1–27, 2013.
[32] B. Reagen et al. ,“MachSuite: Benchmarks for accelerator design and
customized architectures” , in2014 IEEE IISWC . IEEE, 2014, pp. 110–
119.