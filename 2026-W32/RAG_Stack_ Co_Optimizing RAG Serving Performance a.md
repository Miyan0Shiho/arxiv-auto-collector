# RAG-Stack: Co-Optimizing RAG Serving Performance and Quality

**Authors**: Haiqiang Zhang, Yuanqing Lei, Wanting Li, Tao Zhang, Wenqi Jiang

**Published**: 2026-08-04 11:23:19

**PDF URL**: [https://arxiv.org/pdf/2608.03487v1](https://arxiv.org/pdf/2608.03487v1)

## Abstract
Retrieval-augmented generation (RAG), which augments large language model (LLM) generation with information retrieved from databases, has become a widely used approach for knowledge-intensive applications. Modern RAG systems, however, expose many configuration choices, such as retrieval indexes, model selections, and how models invoke retrieval. Each configuration yields a different trade-off between answer quality and serving performance, making it challenging to choose the optimal setting for a specific application deployment. We present RAG-Stack, a framework for efficiently discovering quality-performance Pareto frontiers across diverse RAG applications and serving systems. RAG-Stack consists of RAG-PE, an iterative design-space exploration algorithm that selects the next RAG configuration to evaluate; RAG-IR, a workload abstraction for diverse RAG algorithms; and RAG-CM, a performance model that predicts the optimal deployment and serving performance on the given hardware. Together, these components allow RAG-Stack to search the joint algorithm-system configuration space without deploying every candidate and to transfer an existing Pareto frontier to a new serving system. Given the same number of optimization iterations across diverse datasets, the Pareto frontiers found by RAG-Stack cover 52.5% to 153.2% more of the normalized quality-performance space than those found by state-of-the-art configuration-search methods evaluated over the same RAG design space.

## Full Text


<!-- PDF content starts -->

RAG-Stack: Co-Optimizing RAG Serving Performance and Quality
Haiqiang Zhang
ETH Zurich
Zurich, Switzerland
zhaiqiang@ethz.chYuanqing Lei
Columbia University
New York, United States
yl5457@columbia.edu
Wanting Li
National University of Singapore
Singapore, Singapore
wantingli@u.nus.eduTao Zhang
ETH Zurich
Zurich, Switzerland
zhangta@ethz.chWenqi Jiang
National University of Singapore
Singapore, Singapore
wenqi.jiang@nus.edu.sg
Abstract
Retrieval-augmented generation (RAG), which augments LLM gen-
eration with information retrieved from databases, has become a
widely used approach for knowledge-intensive applications. Mod-
ern RAG systems, however, expose many configuration choices,
such as retrieval indexes, model selections, and how models invoke
retrievals. Each configuration yields a different trade-off between
answer quality and serving performance, making it challenging to
choose the optimal setting for a specific application deployment. In
this paper, we presentRAG-Stack, a framework for efficiently dis-
covering quality–performance Pareto frontiers across diverse RAG
applications and serving systems. Specifically, it consists ofRAG-PE,
an iterative design-space exploration algorithm that selects the next
RAG configuration to evaluate;RAG-IR, a workload abstraction for
diverse RAG algorithms; andRAG-CM, a performance model that pre-
dicts the optimal deployment and serving performance on the given
hardware. Together, these components allowRAG-Stackto search
the joint algorithm–system configuration space without deploying
every candidate and to transfer an existing Pareto frontier to a new
serving system. Given the same number of optimization iterations
across diverse datasets, the Pareto frontiers found byRAG-Stack
cover 52.5–153.2% more of the normalized quality–performance
space than those found by state-of-the-art configuration-search
methods evaluated over the same RAG design space.
Artifact Availability:
The source code, data, and/or other artifacts have been made available at
https://github.com/haiqiang-zhang/rag-stack.
1 Introduction
Retrieval-augmented generation (RAG) augments large language
models(LLMs)withinformationretrievedfromexternaldatasources[ 24,
43,64]. By combining models and databases, RAG addresses several
limitations of model-only systems. First, RAG can incorporate up-
to-date information that may not be encoded in a model’s param-
eters [ 64]. Second, by grounding generation in retrieved evidence,
RAG can improve factual accuracy and reduce hallucinations [ 60].
Third, RAG enables models to use private or domain-specific cor-
pora that were unavailable during pretraining [ 62,72]. Together,
these capabilities have made RAG a widely adopted approach for
knowledge-intensive applications [16, 72, 74].
Although the core idea of RAG is simple, modern RAG systems
expose a large design space spanning both (1) pipeline components
Regular LLM servingRetrievalPreﬁxDecodeRewrite (preﬁx)RerankRewrite(decode)Regular LLM servingPreﬁxDecodeRetrieval[query]
Finish[answer](a) Sequential RAG
(b) Agentic RAGRetrieval ServingEmbedModel
RetrievalRerankRetrieval ToolsEmbedModelnext thoughtQueryAnswer
×N parallel retrieval(Optional)Figure 1: Modern RAG system comprises diverse model
components and execution workflows.
and (2) execution workflows (Figure 1), inducing a broad spectrum
of trade-offs between answer quality and serving performance. At
the component level, a RAG pipeline contains a retrieval system and
a generative LLM and may additionally include stages such as query
rewriting and result reranking. Each RAG component can introduce
a unique choice of model, prompt, and stage-specific parameters. At
the workflow level, these components can be orchestrated either as a
(a) sequential pipeline, in which every query follows a fixed execution
order, or as an(b) agentic pipeline, in which an LLM-based controller
dynamically decides when to retrieve and whether additional re-
trieval rounds are needed [ 46]. Naively enabling every pipeline com-
ponent, selecting the most capable models, or maximizing the num-
ber of retrieval rounds may improve answer quality, but doing so also
increases serving cost and reduces system performance (worse la-
tency and throughput). In this paper, we ask:How can we efficiently ex-
plore this quality–performance trade-off space and identify the Pareto
frontier across diverse RAG applications and serving systems?
However, identifying the Pareto frontier for a given RAG appli-
cation and serving system requires solving three key problems.(P1)
Existing RAG configuration exploration algorithms either
neglect cross-stage interactions or overlook per-stage signals.
One approach to explore RAG configurations is to proceed stage by
stage: it identifies the best configuration for one stage, holds that
configuration fixed, and then optimizes the next stage [ 3]. However,
the best end-to-end configuration depends on interactions among
stages and therefore cannot generally be obtained by optimizing each
stage independently. For example, increasing the retrieval top- 𝑘can
improve quality when a strong reranker filters irrelevant passages or
a long-context LLM effectively uses the additional evidence, but can
1
arXiv:2608.03487v1  [cs.DB]  4 Aug 2026

RAG-PEExplore algorithm configsRAG-IRAbstract the workloadRAG-CMSearch + Predict the Performance
WorkflowSchemaTraceAlgorithm Config.PerformanceEvaluationQualityEvaluationRAG-StackAlgorithm Config. Space
System Config. SpaceInput 4:System ResourceInput1: Algorithm Conﬁg. SpaceInput2: RAG DatasetInput3: Optimization Target
Intermediate RepresentationUsers
OutputPareto-Optimal RAG ConﬁgurationsInput 2:New System ResourceRAG-Stack Mode 1 InputOptimizing RAG from Scratch
RAG-Stack Mode 2 InputMigrating to New SystemInput 1Figure 2: Overview ofRAG-Stackand its two operating modes.
degrade quality when noisy passages reach a weaker generator with-
out reranking. By contrast, global RAG optimization captures these
interactions by jointly tuning parameters across all stages using
end-to-end accuracy as the optimization signal [ 12]. However, this
end-to-end signal does not reveal which stage is responsible for a gain
or loss, potentially leading to poorly directed exploration and wasted
evaluations.(P2) Existing multi-objective RAG optimizers over-
look the system design space.Existing quality–performance op-
timizers search over algorithmic configurations while holding the
serving deployment fixed [ 12,57], even though different algorithms
may require different batching policies, parallelization strategies,
and stage placements to achieve their best serving performance [ 30].
(P3) Deployment-based serving-performance measurements
are costly and do not transfer across systems.Measuring serving
performance requires deploying and benchmarking each candidate
configuration, and the resulting measurements are specific to the
target system [ 45]. When the application is moved to a system with
different hardware, configurations on the original Pareto frontier
may no longer be Pareto-optimal, requiring the search to be repeated
to reconstruct the frontier [28, 50].
To address these problems, we presentRAG-Stack, an efficient
framework for discovering the quality–performance Pareto frontier
across arbitrary RAG applications and serving systems (Figure 2).
RAG-Stacktakes four inputs from the user: (1) an algorithm design
space, (2) a new RAG application (i.e., an evaluation dataset), (3) an
optimization target comprising quality and performance require-
ments, and (4) the available system resources.RAG-Stackcan either
discover a Pareto frontier from scratch or transfer an existing frontier
to a new system by reusing archived quality measurements.
For a new RAG application,RAG-Stackdiscovers the Pareto fron-
tier with an iterative optimization loop. First,RAG-PEproposes an
algorithm configuration for quality evaluation on the application
dataset. Next,RAG-IRabstracts the executed workflow, andRAG-CM
predicts its best achievable serving performance under the availableresources without requiring deployment on the target system. Fi-
nally,RAG-PEuses the measured quality and predicted performance
to select the next configuration. This loop continues until a user-
specified stopping criterion or search budget is reached. We now
introduce the main components inRAG-Stack.
First,RAG-PE( PlanExploration)isthecoremulti-objectiveBayesian
optimizer inRAG-Stack. It takes as input the algorithm design space
and feedback from previous trials, including quality measurements
on the dataset and performance estimates produced byRAG-CM. Its
output is the next algorithm configuration to evaluate, which is sent
through the RAG execution path and subsequently represented by
RAG-IR.RAG-PEaddressesP1by jointly optimizing end-to-end qual-
ity and serving performance while using intermediate stage-level
quality signals to guide exploration.
Second,RAG-IR( Intermediate Representation) bridgesRAG-PE
andRAG-CMby translating each RAG algorithm configuration into
a workload representation comprising a workflow schema and an
execution trace of stage invocations and input/output sizes. The
output ofRAG-IRis asystem-agnosticworkload representation that
decouples each stage’s logical work from its physical deployment,
enablingRAG-CMto explore deployment choices and re-estimate
the performance of the same workload on new hardware. Together
withRAG-CM, this representation addressesP2andP3.
Third,RAG-CM( CostModel) is an ML–analytical fusion perfor-
mance model. It takes as input the workload representation pro-
duced byRAG-IR, the system design space, and the user’s available
hardware resources. It then searches the system design space in-
ternally and returns the best predicted deployment and serving
performance for the current RAG configuration to RAG-PE, thus
effectively addressingP2. Moreover, RAG-CM can re-evaluate the
archived algorithm configurations under a new hardware configura-
tion without repeating their quality evaluations, enabling efficient
frontier transfer and further addressingP3.
We evaluateRAG-Stackon RAGEval [ 74] and MS MARCO [ 9] us-
ing two system configurations: one equipped with four NVIDIA H100
GPUs and the other with eight NVIDIA A100 GPUs. With the same
number of exploration iterations, the Pareto frontiers found byRAG-
Stack, averaged across seeds, cover 52.5% and 153.2% more of the nor-
malized quality–performance space on RAGEval and MS MARCO,
respectively, than those found by state-of-the-art configuration-
search methods evaluated over the same RAG design space. When
transferring a Pareto frontier to a new serving system,RAG-Stack
reuses quality measurements from previous evaluations and per-
forms only a small number of additional optimization iterations
to discover the new frontier. The resulting adapted frontier covers
182.2% more of the normalized quality–performance space than the
frontier obtained by re-optimizing from scratch on the new system.
In summary, the paper makes the followingcontributions:
•We presentRAG-Stack, an end-to-end framework that efficiently
discovers the Pareto frontier between RAG answer quality and
serving performance for arbitrary applications and systems.
•We designRAG-PE, a multi-objective, sub-metric-aware Bayesian
optimizer that uses both end-to-end and intermediate signals to
efficiently navigate the RAG configuration space.
2

•We introduceRAG-CM, a hybrid ML-analytical performance
model that predicts serving performance and searches optimal
deployment configurations on the given hardware.
2 Background and Motivation
2.1 Vector Search for Retrieval
RAG retrievers usevector searchto match queries with passages
based on semantic similarity rather than exact lexical overlap. Be-
fore serving, corpus passages are embedded and indexed. At query
time, the retriever embeds the query and returns the nearest indexed
passages. Exact search scans the entire corpus and guarantees the
true nearest neighbors under the chosen metric, but scales poorly.
Production systems therefore rely onapproximate nearest-neighbor
(ANN) indexes, which prune most candidates to reduce latency and
increase throughput, but may lower recall and consequently degrade
answer quality when relevant evidence is missed.
Two index families.ANN indexes are either clustering-based
or graph-based. TheIVF (inverted-file) familyis clustering-based: it
groups the vectors into many lists, and for each query scans only
the few lists closest to it. Within this family, IVF-Flat keeps the full
vectors and favors recall; IVF-PQ compresses each vector into a short
code to save memory and bandwidth; and IVF-PQ FastScan speeds
up the distance computation with SIMD-friendly kernels [ 19,29].
HNSWis graph-based: it links the vectors into a navigable graph and
answers a query by walking the graph greedily toward the nearest
ones. HNSW often reaches high recall at low latency, but it needs
extra memory to store the graph [51].
Knobstraderecallforspeed.Larger nprobe in IVF or efSearch
in HNSW examines more candidates, improving recall at the cost
of latency and throughput; IVF-PQ reduces memory use but may
lower recall. Because the best choice depends on the workload and
deployment, FAISS is well suited to studying these trade-offs: it di-
rectly exposes both index families and their CPU/GPU knobs [ 19],
unlike vector databases such as Milvus, whose serving layer hides
many low-level choices [65].
2.2 RAG Pipelines and Serving
A RAG pipeline is built from a few stages. The two core stages are
retrieval (§2.1) and generation: the retriever runs a vector search over
the corpus, and the retrieved passages—optionally after query rewrit-
ing or reranking—are placed in a prompt for an LLM that prefills
and decodes. The same stages can be wired into differentworkflows.
The simplest pipeline retrieves once and generates once. Iterative
and active-retrieval workflows interleave several rounds of retrieval
and generation to improve the answer [ 6]. An agentic pipeline goes
further: an LLM controller decides at run time which stages to run,
when to retrieve, and whether to iterate [ 13,46,61]. Choosing the
components and the workflow is already a quality-tuning problem,
and frameworks such as FlashRAG let users assemble, swap, and
evaluate these pipelines to raise answer quality [33].
The choices that raise quality also raise the work per request. A
larger top-𝑘, an added reranker, iterative retrieval, or a bigger gener-
ator each helps the answer, but each also adds retrieval work, model
calls, prompt tokens, or decoding time. Serving then adds a second
layer of choices that leave the answer unchanged but set how fast it
is produced: how the stages are placed on the hardware, how much
128256512 1K2K
Chunk size0.00.20.40.6Median quality 
(a)
Seq. vs ReActExpansion off/on Reranker off/onCompress. off/on0.1
0.00.10.2Median Quality 
(b)
020406080
Median QPS 
0.5×1×2×4×8×
Median QPS ratio 
Figure 3: Median quality–throughput tension across algo-
rithm choices. (a) By chunk size. (b) Single-choice contrasts
(quality: first−second; QPS: first /second); error bars: in-
terquartile range.
parallelism they use, how requests are batched, and how the index
and runtime knobs are configured.
A large body of systems work optimizes this serving layer for a
fixed RAG algorithm—tensor [ 59], pipeline [ 26,27], and data [ 44]
parallelism, continuous batching [ 70], and paged KV-cache manage-
ment [ 40]—together with RAG-specific techniques such as adaptive
pipeline parallelism [32] and disaggregated accelerators [31].
2.3 Motivation: Performance
and Quality Co-optimization for RAG
The discussion above of RAG pipelines and serving shows that co-
optimizing answer quality and serving performance better reflects
the practical needs of RAG serving. This view also accounts for
RAG serving-system design, because placement, batching, and par-
allelism determine how efficiently each pipeline runs. Manual co-
optimization is difficult because measurements reveal trade-offs
that intuition misses (Figure 3): agentic ReAct [ 69] can be faster
than a sequential pipeline at a modest quality cost, query expansion
sacrifices throughput for a quality gain that appears only in some
configurations, and even a larger generator does not always improve
quality [ 73]. A configuration that avoids such unrewarded cost can
deliver the same quality at higher throughput and thusdominate
alternatives; only non-dominated configurations are worth serving.
Finding the frontier is a search problem.Let Xdenote a can-
didate space and y(x)∈R𝑚the objective vector of x∈X , with
all objectives oriented so that larger values are better. An objective
pointydominates y′ify≥y′componentwise with at least one
strict inequality [ 18]. A candidate isnon-dominatedif no other can-
didate’s objective point dominates its own; the objective points of
all non-dominated candidates form thePareto frontier F. Thehyper-
volume HV(F ;r)is the volume of the objective-space region that is
dominated byFand dominates a fixedreference point r∈R𝑚that
lower-boundsF; a larger hypervolume indicates broader coverage
of desirable trade-offs [ 17]. Throughout, objectives are min–max
normalized per experiment and dataset, with r=0 . The frontier thus
provides a set of operating points from which users can choose. A
general configuration-search method finds it iteratively: it proposes
a candidate, evaluates its objective vector, and uses the observations
collected so far to select the next candidate. Because this search
3

150 250 3500.150.200.25Quality (ans. corr.)+32%
k=1k=8
-20%k=1
k=8chunk 512
chunk 256
25 30 35 40+16%
k=1k=16
-14%k=1
k=16no compression
compression 0.3
Performance (QPS)Figure 4: Cross-stage interactions in the algorithm design
space. Each arrow traces one configuration as retrieval top-𝑘
rises; annotated percentages are the resulting quality change.
requires onlyXand the evaluated objective values, it can in princi-
ple use any multi-objective optimization method. The next section
shows why such methods, applied directly to RAG, fall short.
2.4 Limitations of Existing Approaches
Existing RAG-specific optimizers cover only parts of the quality–
performance problem. A RAG configuration spans two design spaces:
thealgorithm design space, whose choices affect answer quality—
chunking, top- 𝑘, reranking, and the generator—and thesystem de-
sign space, whose choices affect how fast that answer is served—
placement, batching, parallelism, and hardware. Quality-side tools
search the algorithm design space against answer quality alone [ 3,22,
33]. System-side tools such as RAGO search the system design space
for serving performance, but only for a fixed RAG algorithm [ 30].
Online methods such as METIS adapt a small set of serving choices
per query, but they also operate within a fixed deployment [57].
A natural alternative is to treat RAG tuning as a generic multi-
objective black-box optimization problem. Recent work takes this
route by applying standard multi-objective Bayesian optimization,
including LogNEHVI, to search RAG hyperparameters for quality–
cost trade-offs [ 12]. More broadly, possible black-box optimizers
include plain sampling, Bayesian optimization [ 10,42,48,53,66],
evolutionary search [ 14,25], and multi-fidelity optimization with
cheaper proxy runs [ 35]. These optimizers can also be steered by
LLMs [ 1,41,52] or extended to the multi-objective setting through
scalarization [ 38,55], dominance [ 18,71], or hypervolume crite-
ria [4,17]. These general tools are our starting point. Applied directly
to RAG, however, they and the RAG-specific optimizers above run
into three problems.
(P1)Existing RAG optimizers either break cross-stage inter-
actions or lose stage-level signals.Figure 4 shows that increasing
retrieval top- 𝑘always reduces throughput, but its quality effect flips
with chunk size and compression, demonstrating that stage choices
cannot be optimized independently. Stage-wise optimizers such as
AutoRAG retain stage-level feedback but miss configurations that
work only through cross-stage combinations [ 3]; global optimization
accounts for these interactions by tuning complete pipelines, but
end-to-end feedback alone cannot reveal which stage drives a gain or
loss, leading to poorly directed exploration and wasted evaluations.
(P2)Existing multi-objective RAG optimizers overlook the
system design space.Existing quality–performance RAG optimiz-
ers search algorithm choices but measure performance on one fixed
deployment [ 12], while online methods adapt only a few servingparameters within that deployment [ 57]; neither finds an algorithm
configuration’s best achievable serving performance. Simply fold-
ing the full system design space into the multi-objective search is
wasteful because system parameters do not affect answer quality,
yet every variant would still incur a full quality evaluation.
(P3)Deployment-basedserving-performancemeasurements
are costly and non-transferable across systems.RAG serving
performance is typically obtained by deploying and timing each
configuration end to end, making dense deployment search prohib-
itively expensive [ 45,47]. Such measurements are system-specific:
a configuration on the original Pareto frontier may become domi-
nated on a new system, so migration requires remeasurement and
frontier reconstruction; candidate hardware also cannot be evalu-
ated before acquisition. Although prediction could avoid these costs,
existing RAG performance models cover only narrow portions of
the stack [30, 37], leaving no portable full-stack estimator.
3RAG-Stack: System Overview
We presentRAG-Stack(Fig. 2), an efficient framework that finds a
RAG system’s quality ( 𝑄)–performance ( 𝑃) Pareto frontier across
the full algorithm and system design space, without deploying each
candidate to measure its performance. This addresses the limitations
in §2.4, which leave RAG without a practical way to co-optimize
answer quality and serving performance across its full design space.
Inputs.In from-scratch mode, a user drivesRAG-Stackwith a
single declarative specification of four things:(i)thealgorithm de-
sign spaceto search, a hierarchical, conditional space of components
and parameters detailed in §4.1 (Fig. 5);(ii)an evaluationdataset
of queries with reference answers;(iii)anoptimization target—the
quality and performance metrics to trade off, plus serving SLOs; and
(iv)the availablesystem resources—GPUs, CPU, and interconnect—
bounding the system design spaceRAG-CMsearches (Table 1). In
system-transfer mode, the input is instead the previous Pareto result
and a new system-resource specification, as described underTwo
operating modesbelow.
Output.RAG-Stackreturns the Pareto-optimal configurations
it found—each a complete algorithm-and-system deployment—so
the user picks the operating point that fits their workload, hardware,
and SLO.
Overview.RAG-Stack(Fig. 2) rests on one split: a RAG config-
uration’s parameters divide into analgorithm design space, whose
choices change the produced answer and so move both quality and
performance, and asystem design space, whose choices change only
how that fixed computation is served.RAG-Stacksearches the two in
an iterative loop over three components. Each round,RAG-PE( Plan
Exploration, §4) proposes one algorithm configuration x; the quality
evaluator runs that pipeline on the dataset and returns its quality
𝑄(x) ;RAG-IR( Intermediate Representation, §5) abstracts the exe-
cuted run into a workload representation;RAG-CM( CostModel, §6)
searches the system design space for that representation and predicts
x’s best serving performance 𝑃(x) ; andRAG-PEuses the measured
𝑄(x) and predicted 𝑃(x) to choose the next configuration. The loop
stops when the optimization target is met or the budget is spent. Qual-
ity comes from this one evaluation run, but performance is always
predicted byRAG-CM—no candidate is ever deployed to measure it.
4

Two operating modes.The loop above optimizes a RAG sys-
temfrom scratch, discovering its frontier on the current hardware.
RAG-Stackalso supports asystem-transfermode: to retarget an
already-optimized system to new hardware, the user provides only
the previous Pareto result and a new system-resource specification.
RAG-Stackthen reuses the quality results from the first run, first
invokingRAG-CMto re-score the previous deployments under the
new resources and subsequently running a small number ofRAG-PE
polishing iterations to refine the transferred frontier.
Benefits.RAG-Stack’s design yields three benefits, each remov-
ing one limitation of prior optimizers.B1 (solving P1): stage-aware
multi-objective optimization without breaking cross-stage
interactions.RAG-PEstill optimizes the end-to-end (𝑄,𝑃) Pareto
frontier, so each trial is a complete pipeline configuration and cross-
stage interactions are preserved. At the same time, it reads stage-level
sub-metrics to guide exploration: for example, low context recall
with high faithfulness points to retrieval as the bottleneck rather
than generation. These sub-metrics guide whereRAG-PEsearches
next without becoming separate objectives, giving one optimizer
both global frontier optimization and stage-level diagnosis (§4.2).
B2 (solving P2): near wallclock-free exhaustive system search.
Given the algorithm/system split above,RAG-PEspends expensive
quality evaluations only on algorithm configurations, whileRAG-CM
exhaustively searches the system design space for each one inside
the cost model. Because this search uses predicted performance
rather than real deployments, it is nearly wall-clock-free, letting
RAG-Stackreport the best deployment-side operating point instead
of inheriting the performance of one fixed deployment (§4.1, §6).
B3 (solving P3): cheap system transfer and planning.Because
RAG-CMpredicts performance from a hardware description instead
of a real run, an optimized system is retargeted to new hardware by
re-scoring the quality archive, and candidate machines are compared
before purchase—neither step builds a real deployment.
We now detail the three components in turn:RAG-PE(§4),RAG-IR
(§5), andRAG-CM(§6).
4RAG-PE: Plan Exploration
RAG-PEpartitions the design space (§4.1), runs the RAG-specific op-
timizer (§4.2), and coordinates quality evaluation with performance
modeling (§5, §6).
4.1 Search Space
The space of RAG configurations is large and hierarchical.RAG-PE
splits it into an algorithm part and a system part, searches the algo-
rithm part directly, and delegates the system part toRAG-CM. We
define this split and then organize the partRAG-PEsearches as a
hierarchy.
4.1.1 Two Design Spaces: Algorithm and System Design Spaces.Let
Xfull=Î𝑛
𝑗=1Θ𝑗be the full configuration space of the RAG stack,
with each Θ𝑗denoting the domain of one parameter (e.g., nprobe ,
thread_count ,top_k ), and let𝑄(x) and𝑃(x) be the answer-quality
and serving-performance objectives under the user’s chosen met-
rics (e.g., latency, throughput, or SLO satisfaction). We partition the
parameters by what each one changes: thelogical computationthat
produces the answer, or only thephysical executionof that fixedcomputation. Analgorithm parameterchanges the logical compu-
tation (Fig. 5); conversely,a parameter belongs to the system design
space whenever it leaves the logical computation unchanged. System
parameters change only how that fixed computation is executed and
served (Table 1)—placement, batching, parallelism, thread counts,
and hardware. Batching, for example, merely groups identical per-
request computations and so leaves the answer unchanged; varying
parallelism or thread count may perturb the output slightly through
floating-point reduction order, but the logical computation remains
unchanged.
Formally, let 𝐼𝑌denote the parameters relevant to objective 𝑌∈
{𝑃,𝑄} . Any parameter relevant to answer quality 𝑄changes the logi-
calcomputationandmustalsobeconsideredforservingperformance
𝑃; therefore,𝐼𝑄⊆𝐼𝑃. This yieldsXfull=X algo×X cm, whereXalgo:=Î
𝑗∈𝐼𝑄Θ𝑗contains parameters that affect both 𝑄and𝑃, whileXcm:=Î
𝑗∈𝐼𝑃\𝐼𝑄Θ𝑗contains parameters that affect only 𝑃.RAG-Stackopti-
mizesXalgoagainst(𝑄,𝑃) and delegatesXcmtoRAG-CMfor 𝑃alone.
Table 1: System design space searched byRAG-CM.
Category Design choice Example range
Vector DBSearch threads{16,32,64}
Parallel mode {intra-query, inter-query}
LLM servingParallelism strategy {TP, PP, DP, hybrid}
GPUs per stage{1,2,3,4}
Prefill–decode
deployment{collocated, disaggregated}
Placement Stage-to-device
mappingPrefill→GPUs {0–7}
BatchingRequest batch size{1,2,...,256}
Decode batch size{16,32,64,128,256}
Dynamic
batching wait time{0,1,2,5,10}ms
Hardware
(Optional)CPU model {AMD EPYC, Intel Xeon}
GPU inventory {4×H100, 8×A100}
4.1.2 Search-space organization.RAG-PEparses the user’s declar-
ative specification into the two design spaces above. System-only
choices inXcmare sent toRAG-CM(§6), which exhaustively searches
them inside the cost model to return the best predicted performance
˜𝑃(x) for a given algorithm configuration.RAG-PEtherefore runs the
expensive ground-truth optimization only over Xalgo, whileRAG-CM
handles the internal system search. It organizes Xalgoas a hierarchi-
cal design space (Fig. 5) governed by two rules:conditional activation
andheuristic constraintsthat remove invalid or low-value trials. Un-
der conditional activation, a child parameter is active only when its
parent choice is active; for example, query-rewriter parameters are
searched only if query rewriting is enabled. The heuristic constraints
(starred nodes in Fig. 5) use smooth proxies for derived knobs: IVF
setsnlist=factor·√
𝑁for corpus size 𝑁, while PQ selects a valid
divisor𝑀of the embedding dimension𝐷.
4.2 The Optimizer
As discussed in §2.4, RAG’s algorithm design space is entangled by
cross-stage interactions: the best setting for one stage depends on the
choices made in the others (Fig. 4). Optimizing stages independently
5

Algo. Design Spacecorpus chunkerquery rewriterretrievalchunk sizechunk typeenableddisabledHyDEmulti query LLM modeltop-KindexIVFHNSWquantizationpqﬂatnlist factordsub (D/M)nbit…RAG workﬂowsequentialagentic
48632embedding model
…384D1024D…141282048
heuristic aware: nlist = factor · √N; N depends on corpus size & chunk size, only factor is tunedheuristic aware: M should be relative to the embedding dimension D — chosen so that dsub = D/M falls in the sweet-spot range (≈4–16), subject to D mod M = 0.Selected Retrieval-side ParametersSelected Model-side RAG Parameters1.5B14B……passage rerankerpassage compressormain LLMenableddisabled
tartColBERTtypestop-K…enableddisabledLLM modeltemperatureLLMLingua-2rate0.30.71.5B14B…
…temperature164…Figure 5: Algorithm design space [ 23,29,36,51,54] searched byRAG-PE. The highlightedselectednodes are one example, showing
how a single evaluation’s algorithm configuration is assembled: each trial fixes one value at every active branch of the hierarchy
(retrieval-side in purple, model-side in yellow), and the starred callouts mark heuristic-aware parameters coupled across branches.
can therefore miss configurations that work well only as a complete
pipeline.RAG-PEconsequently usesmulti-objective Bayesian opti-
mization(MOBO) as its global foundation, evaluating and ranking
complete configurations against the end-to-end (𝑄,𝑃) Pareto frontier.
MOBO also suits the small evaluation budget: its probabilistic surro-
gate and acquisition function use prior observations and uncertainty
to select the configuration expected to improve the frontier most.
A global optimizer alone, however, sees only the end-to-end ob-
jectives and lacks directional information about which stage should
change.RAG-PEtherefore augments the global MOBO with stage-
level feedback: stage diagnostics guide exploration toward promising
changes, while the global acquisition function still arbitrates com-
plete configurations according to how much they are expected to
improve the(𝑄,𝑃) frontier. Stage information thus directs the search
without becoming a separate objective or breaking cross-stage inter-
actions. We first introduce the MOBO foundation and then present
ourstage–global co-awareextensions.
4.2.1 Preliminary:multi-objectiveBayesianoptimization.Multi-objective
Bayesian optimization(MOBO) follows the general Pareto-search
formulation in §2.3. In our setting, X=X algoandy(x)= 𝑃(x),𝑄(x),
using the serving-performance and answer-quality objectives de-
fined above. Because evaluating quality is expensive, MOBO seeks
to identify the frontier using as few evaluations as possible.
MOBO consists of two components. The first is a cheap proba-
bilisticsurrogatefor each objective—a Gaussian process (GP) fitted
to all configurations evaluated so far. Because Xalgocontains many
categorical variables and has a hierarchical structure, each GP uses a
mixed kernel: a Matérn kernel for the numeric knobs and a Hamming
kernel for the categorical ones. Together, they provide an appropriate
notion of similarity for this design space [42].
The second ingredient is anacquisition function 𝛼that scores any
unevaluated configuration by its expected gain in this hypervolume.LetD𝑡denote the data after 𝑡rounds. Thehypervolume improve-
ment(HVI) of a candidate outcome yoverFisHVI(y|F,r)=
HV(F∪{y} ;r)−HV(F ;r). HVI alone cannot score a candidate,
because the outcome yis unknown before the evaluation.Expected
hypervolume improvement(EHVI) resolves this by averaging HVI
over the GP posterior prediction at x, measured against the fron-
tier of the outcomes observed so far [ 17]. EHVI, however, trusts
those observations to be exact, while our quality scores are noisy:
the observed outcomes need not form the true frontier.NoisyEHVI
(NEHVI) also treats the frontier as uncertain, averaging HVI over
posterior draws of the evaluated configurations and the candidate:
𝛼NEHVI(x)=E f∼𝑝(f|D 𝑡)[HVI(f(x)|F f,r)]. We estimate this expec-
tation using 𝑁Sobol draws from the joint GP posterior; each draw
f𝑗induces a frontierF 𝑗over the evaluated configurations [17].
In practice, we maximizeLogNEHVI, a numerically stabilized vari-
ant of NEHVI, using the BoTorch implementation [ 4,10]. Each round,
RAG-PEfits the GPs, picks the next configuration by maximizing the
acquisition over the whole space, x𝑡+1=argmaxx∈X algo𝛼LogNEHVI(x),
evaluates it, and appends the result.
TheseconsiderationsleadustochooseGP+LogNEHVIoverSMAC’s
combination of a random-forest (RF) surrogate and ParEGO [ 38,48].
LogNEHVI directly rewards expected improvement in the noisy
Pareto hypervolume, whereas ParEGO targets the frontier indirectly
through scalarized objectives [17]. The RF instead learns similarity
through tree partitions; its piecewise-constant predictions can create
plateaus in the acquisition landscape, leaving little signal for local
refinement.
Despite this principled acquisition rule, the standard loop is poorly
directed when applied to RAG’s large hierarchical design space. The
surrogate is fitted only on the two end-to-end objectives, so its pos-
terior uncertainty indicates where the search has not looked, but
notwhya configuration succeeds or which pipeline stage is worth
changing. Under a tight quality-evaluation budget, it can therefore
spend trials on configurations that are novel yet unlikely to be useful.
6

Agentic AnalyzerGaussian Processes (GPs)Discrete LocalSearch (DLS)Candidate PoolArbitrate124
5RetrievalMain LLMRewrite (preﬁx)RerankEmbedModele2e-score of Q & PEval. Result (Q & P)GPs know e2e-score only per-stage score of Q & PAgentic Analyzer know per-stage and e2e score
Next Algo. Conﬁg for EvaluationInputPareto-tension crossover3
Forced channelsLogNEHVIQuality reuseOutputFigure 6: The optimizer ofRAG-PE.
4.2.2 Our optimizer: stage–global co-aware MOBO.RAG-PEextends
conventional MOBO for RAG with stage-aware diagnostics, het-
erogeneous candidate generation, and global arbitration, making
exploration more directed while preserving end-to-end Pareto op-
timization. This design is necessary because stage effects are non-
separable—the benefit of changing one stage depends on the choices
made in the others (Fig. 4). Greedy stage-wise optimization can
therefore miss configurations that work well only as a complete
pipeline [ 3]. Composite-function BO also exploits intermediate out-
puts, but assumes that the final objective is a known function of those
outputs [ 7,39]. RAG provides useful per-stage diagnostics, but no
known function maps them to end-to-end quality and performance.
RAG-PEtherefore uses these diagnostics to direct candidate genera-
tion while leaving complete-configuration comparison to the global
MOBO objective.RAG-PEcombines three mechanisms, shown in
Fig. 6 and detailed below.
(1) Sub-metric Awareness. RAG-PEoptimizes only end-to-end
quality𝑄and performance 𝑃, while itsagentic analyzer(
1 ) uses
stage-level sub-metrics to diagnose where a configuration falls short.
The quality diagnostics are context recall and precision for retrieval
and answer faithfulness for the Main LLM; the performance diagnos-
tics areRAG-CM’s per-resource capacities 𝜅𝑟(§6.5), which expose
the bottleneck stage and remaining headroom. We record them as s𝑖
alongside each observation, yielding D𝑡={(x𝑖,𝑃𝑖,𝑄𝑖)}𝑡
𝑖=1for the GPs
andH𝑡={(x𝑖,𝑃𝑖,𝑄𝑖,s𝑖)}𝑡
𝑖=1for the analyzer. Reading H𝑡, the analyzer
forms𝑀𝑡stage-guided candidate configurations RLLM
𝑡={c 1,...,c𝑀𝑡}.
Eachc𝑚=b𝑚⊕𝜹𝑚,𝑚=1,...,𝑀𝑡, is constructed by applying a targeted
edit𝜹𝑚to a previously evaluated base b𝑚and snapping the result
to the valid grid;⊕overwrites only the edited parameters [ 41,49].
Thus, sub-metrics guide candidate generation without becoming
surrogate inputs or optimization objectives.
(2) Heterogeneous Candidate Pool.Standard acquisition optimiza-
tion uses multiple random restarts, but all candidates are refined
against the same surrogate and therefore inherit the same model bias.
RAG-PEinstead constructs C𝑡from four complementary channels:
Sobol coverage, stage-guided candidates, DLS-based acquisition-
guided local refinement, and Pareto-tension crossover (Fig. 6).
Two channels determine where to search:Sobol breadthdraws
a hierarchy-aware, space-filling candidate setB 𝑡, while theagentic
analyzersupplies stage-guided candidatesRLLM
𝑡(
1).The stage-guided candidates and quasi-random restarts Rrand
𝑡
seeddiscrete local search(DLS,
2 ). Let z(𝑘)denote the discrete
configuration after 𝑘DLS steps, and letN(z) denote its hierarchy-
valid one-parameter neighbors. Each c𝑚initializes a DLS trajectory
atz(0)=c𝑚. WritingN+(z)=N(z)∪{z} , DLS greedily applies
z(𝑘+1)=argmaxz∈N+(z(𝑘))𝛼LogNEHVI(z)until convergence. Thus,
the agent chooses a promising region and DLS refines the configu-
ration within it.
Pareto-tension crossover(
3 ) starts from the best observed config-
uration for each objective and applies the single-knob crossover that
most strongly moves it toward the other objective. For 𝑜∈{𝑃,𝑄} ,
denote that anchor by x★
𝑜=argmaxx∈D 𝑡𝑜(x) and the other objective
by¯𝑜. Letx𝑜,𝑗=x★
𝑜⊕𝑗𝑥★
¯𝑜,𝑗denote the hierarchy-valid crossover that re-
places only knob 𝑗with its value from x★
¯𝑜. It selects𝑗★
𝑜=argmax𝑗𝜏𝑜,𝑗,
where hats denote min–max normalization and 𝜏𝑜,𝑗=ˆ¯𝑜(x𝑜,𝑗)−
ˆ¯𝑜(x★
𝑜)−[ ˆ𝑜(x★
𝑜)−ˆ𝑜(x𝑜,𝑗)]+, yieldingR×
𝑡. The candidate pool is
C𝑡=B𝑡∪RLLM
𝑡∪DLS RLLM
𝑡∪Rrand
𝑡∪R×
𝑡, with DLS applied inde-
pendently to each seed and source labels retained for arbitration (
4 ).
(3) Arbitration.On ordinary rounds, LogNEHVI scores every can-
didate inC𝑡, andRAG-PEevaluates the candidate with the largest
expected hypervolume improvement, independent of its genera-
tor. When the number of consecutive ordinary evaluations with no
realized normalized-hypervolume gain reaches a stagnation thresh-
old𝜏𝑠,RAG-PEactivates itsforced channels. These channels target
the largest gap between adjacent points on the normalized realized
Pareto frontier and draw candidates from the agent and Pareto-
tension crossover. The candidates bypass LogNEHVI and are ranked
by the product of their range-normalized posterior standard devi-
ations, with hierarchy-aware novelty breaking ties. Finally,quality
reuseskips generation and judging when the retrieved passages and
quality-relevant downstream configuration match an earlier eval-
uation:RAG-PEreuses its quality score, recomputes performance
withRAG-CM, records the point, and repeats arbitration.
5RAG-IR: Intermediate Representation
RAG-IRbridges quality evaluation andRAG-CMthrough a common
workload representation with two properties. It issystem-agnostic,
separating logical work from deployment soRAG-CMcan search
deployment choices and re-cost the same workload on new hard-
ware without rerunning it. It isRAG-workflow-agnostic, encoding
sequential and agentic pipelines with a common schema soRAG-
PEcan explore workflows whileRAG-CMevaluates them without
workflow-specific modeling logic (Fig. 2).
RAG-IRrecords an order-free workflow schema for throughput
and per-request execution traces for latency (§6.5).
Workflow schema.An order-free summary of each stage’s
performance-relevant attributes and aggregate logical work, used
byRAG-CMto predict throughput.
Per-request trace.For each request 𝑞,RAG-IRrecords an execu-
tion DAGG(𝑞)
𝑖. Each node is one stage invocation and stores its stage
type and input/output token counts; edges record dependencies be-
tween calls. Repeated calls in iterative or agentic pipelines appear
as separate nodes, preserving their order and multiplicity.RAG-IR
forwards the schema and traces, together with x𝑖, toRAG-CM(§6).
7

6RAG-CM: Cost Model
RAG-CMis the performance-estimation component ofRAG-Stack.
For a fixed RAG algorithm configuration x𝑖, it estimates the best at-
tainable serving performance on a target system by searching the sys-
tem design spaceXcm, whose parameters affect serving performance
but not answer quality. Its inputs are the workload representation
produced byRAG-IR(a workflow schema and per-request execution
tracesG𝑖), the algorithm configuration x𝑖, and a resource specifica-
tion of the target system’s devices, communication topology, and
serving resources.
We denote by s∈X cmone concrete assignment of all system-
design parameters (Table 1). For each pair (x𝑖,s),RAG-CMpredicts
throughput and end-to-end latency, ˜p(x𝑖,s).
Whyacostmodelforperformance.Predicting performance in-
stead of measuring it serves four roles inRAG-Stack.(R1) A smaller
optimizer space.By moving the system design space intoRAG-
CM, the optimizer only searches the algorithm design space Xalgo,
so the scarce ground-truth budget is spent only on parameters that
affect quality (§4).(R2) RAG serving planning and transfer.RAG-
CMlets users plan RAG serving before acquiring the target system
and efficiently retarget it to new hardware by re-scoring the serv-
ing performance of previously identified Pareto configurations on
the new system.(R3) Faster optimization iterations.Because
RAG-CMpredicts performance without deploying each candidate or
benchmarking its performance, every optimization iteration avoids
this overhead, substantially reducing wall-clock time relative to
deployment-based optimizers.(R4) Exhaustive system search.
RAG-CMexhaustively searches the system design space for each
algorithm configuration, yielding higher performance and more
effective optimization.
The four-layer structure.RAG-CMcomputes ˜p, and hence ˜𝑃,
in four layers (Fig. 7): analgorithmlayer that produces one hardware-
agnostic Operator Work Profile per RAG stage (§6.1), aperformance
layer that maps each Operator Work Profile to time on the host (§6.3),
acommunicationlayer that prices the data moved between operators
(§6.4), and anassemblylayer that sweeps the system design space
Xcmand composes the other three layers into the throughput and
latency of each algorithm–system configuration pair (§6.5).
6.1 Algorithm Modeling
GivenRAG-IR’s workflow and execution trace, the algorithm layer
models every RAG stage and outputs one hardware-agnosticOp-
erator Work Profileper stage—a record of that stage’s per-phase
operation counts, data movement, and execution characteristics (
1
in Fig. 7; §6.2).RAG-CMuses GenZ [11] for model-inference opera-
tors; the following subsections present its retrieval models, grouped
into Clustering Indices (§6.1.1) and Graph Indices (§6.1.2).
6.1.1 ClusteringIndices.Allthreeclusteringindicesshareonesearch
skeleton: a query is run through acoarse quantizerthat selects the
𝑛probe inverted lists nearest to it, the candidate vectors in those lists
arescannedto compute distances, atop- 𝑘heapkeeps the best candi-
dates, and an optionalre-rankrecomputes exact distances on the sur-
vivors.Theyareanalytical:giventheconfiguration,theworkinevery
stage is fixed in closed form. Table 2 lists every stage, its compute
kernel, the peak it is bounded by, and which variants use it. For most
stages, the corresponding Operator Work Profile follows directly
Algorithm Modelling
Performance ModellingIVF Family OperatorHNSW OperatorPreﬁll Operator
Operator Work Proﬁlebytescompute modeopsparallel fractionCPU TopologyNUMAbig.LITTLEScatteredsequential MemoryEﬀective CacheParallel ModelRooﬂine ModelCommunication Modelling
1AMLA
LatencyThroughputOutputRetrievalPreﬁxDecodeRerankEmbedModelDecode OperatorA
bandwidthﬂopsopsbytesGPU     CPU / GPU     GPU↔↔Communication Type  (PCIe, NVLink …)5
344Input   RAG-IR
bytes locationaccess block
RAG-CM AssemblyGet E2E RAG PerformanceSearch System Design Space62Figure 7: Overview ofRAG-CM. A: analytical; ML: fused
machine learning and analytical.
Table 2: IVF-family search stages (Flat: IVF-Flat; PQ: IVF-PQ;
FS: IVF-PQ Fastscan; FSR: Fastscan with residual).
Stage Compute Mode Peak Used by
1. Coarse quantizer BLAS GEMM if per-thread batch ≥20, else
hand-vectorized SIMD distance kernelFP32 all
2. LUT build non-residual, built once/query
(FS); residual-dependent (PQ, FSR)FP32 PQ, FS, FSR
3a. PQ scan ADC table lookup (PQ) / vpshufb (FS, FSR) int PQ, FS, FSR
3b. Flat scan full-vector SIMD L2 int Flat
4. Top-𝑘heap branch-heavy integer int all
5. Refinement exact SIMD L2 on candidates FP32 optional
from the FAISS source: the ops, bytes, and access pattern of each ker-
nel are fixed by the configuration, so a closed-form formula gives the
profile for each such stage listed in Table 2, with variant-specific for-
mulaswherethekernelsdiffer;forexample,standardPQbuildsitsdis-
tance table per probed list, whereas Fastscan builds it once per query.
Only the scan stage requires special treatment because its work
cannot be read directly from the configuration, so we model it below.
PQ/flat scan: data-aware imbalance.The scan’s cost is driven by
𝑁scan, the number of database vectors a query touches—a vector
count, not an op or byte total. Under a fixed PQ configuration, scor-
ing each scanned vector requires reading its 𝑏code-byte PQ code
and summing one precomputed distance-table entry per subquan-
tizer, so the scan’s ops are 𝑁scantimes the per-vector work and its
bytes are𝑁scan𝑏code; this is how 𝑁scansets the scan operator’s pro-
file. IVF makes 𝑁scanfar smaller than the corpus size 𝑁: a query
scans only the 𝑛probe of𝑛listinverted lists nearest to it, a fraction
𝜌=𝑛 probe/𝑛listof all vectors. The naive estimate 𝑁scan=𝜌𝑁 as-
sumes equal-sized lists, but real embeddings cluster, so a query’s
nearest lists are larger than average and it scans more; we correct
with a cell-imbalance factor 𝑓measured from the built index, so
that𝑁scan=𝑓(𝑛 probe)𝜌𝑁, where𝑓is the ratio of the vectors a real
8

query actually scans—obtained from the index’s per-list sizes and
the nearest-list assignments of a query sample—to the balanced
𝜌𝑁. This𝑓is the only index-dependent input the model needs, and
it is a pure property of the corpus and embedding—never of the
hardware—while the remaining index knobs enter the cost model
as analytical parameters.RAG-CMtherefore caches 𝑓once per (cor-
pus, embedding)—only a handful of combinations across a whole
design space—and from then on predicts the scan count for any
configuration without ever building an index again.
6.1.2 Graph Indices.HNSW isML-analytical. A query first descends
greedily through the 𝐿≈log𝑀𝑁upper layers—a few hops each over
𝑀-neighbor lists—to reach a good entry point, then runs a bounded
best-first (beam) search of width ef𝑠at the base layer, which performs
almost all of the work. Where IVF’s scan count follows in closed form
from the configuration, the size of the base-layer neighborhood an
HNSW query explores is data-dependent—it grows with the data’s
intrinsic geometry and shrinks on clustered corpora where the beam
converges fast—and has no closed form. The model thereforepre-
dictsthe two quantities that drive the Operator Work Profile: the
per-query distance computations 𝑛disand graph hops 𝑛hops. These
fold into the profile’s 𝑊and𝑉exactly as IVF’s 𝑁scandoes (§6.2)— 𝑛dis
setting the dominant base-layer FLOPs and random-vector reads,
and𝑛hopsthe visited-set and heap overhead. Concretely, the HNSW
operator model estimates ˆ𝑛dis=𝑐𝑀ˆ𝑔dis(z)andˆ𝑛hops=ˆ𝑔hops(z)with
feature vector z=(𝑑,𝑀,ef𝑠,LID) : a learned predictor ˆ𝑔emits the
workload counts ˆn=( ˆ𝑛dis,ˆ𝑛hops), and a factor 𝑐𝑀obtained through
one-time per-corpus calibration fixes the scale of ˆ𝑛dis. These counts
determine the HNSW Operator Work Profile; the following para-
graph defines ˆ𝑔and its lightweight calibration.
Workload predictor.The predictor ˆ𝑔is two independent gradient-
boosted regressors, ˆ𝑔disand ˆ𝑔hops; both are predicted because their
ratio is not constant—it drifts with ef𝑠—and each contributes a dif-
ferent part of the Operator Work Profile. The lone data-dependent
feature is the corpus local intrinsic dimensionality LID, estimated
by an Amsaleg MLE on a query sample; it carries the geometry that
lets a single model generalize across corpora—from high-LID syn-
thetic training data to the low-LID, tightly clustered distributions
of real image and text embeddings, on which a configuration-only
estimate over-counts because the beam terminates sooner than the
ambient𝑑suggests. Crucially, every feature in zis either a search-
space knob or a property of the corpus itself: the predictor usesno
statistic of a constructed graph (degree, layer counts), so it scores
candidate configurations whose indexes have not been built—exactly
the regime the optimizer explores. The predictor requires only a one-
time calibration for each corpus, which is reused across all unbuilt
configurations and therefore adds little overhead.
6.2 Operator Work Profile
Each operator emits a hardware-agnosticOperator Work Profile, the
interface between the algorithm and performance layers. Computed
once per operator phase, it containswork (𝑊,𝑉) —the operation
count and bytes moved after cache reuse—and fourexecution char-
acteristics: (1) thecompute mode(e.g.blas/simd), which selects
compute efficiency; (2) thebytes location, determined by the working-
set size and unique byte volume 𝑉uniq≤𝑉, which selects the memorylevel; (3) theaccess-block size, which distinguishes sequential from
scattered bandwidth; and (4) theparallel fractionfor Amdahl scaling.
The performance layer maps these fields to effective compute and
memory rates(𝜋eff,𝛽eff). Data-dependent corrections modify 𝑊and
𝑉rather than the profile schema, allowing one performance model
to serve all operators.
6.3 Performance Modeling
The performance layer maps anOperator Work Profileto wall-clock
time in two composable steps (
3 ,
4in Fig. 7): arooflinebounds each
phase by the slower of its compute and memory time, andAmdahl’s
lawcomposes the per-phase times across threads under the chosen
parallel mode. The intermediate points below exist only to feed these
two: the hardware model supplies the roofline’s two rates ( 𝜋eff,𝛽eff),
and the parallel model sets how the per-phase results combine into
end-to-end latency and throughput.
Roofline kernel.For a phase with 𝑊operations and data volume 𝑉,
the model predicts 𝑇=𝑇 ovh+max
𝑊
𝜋eff,𝑉
𝛽eff
, where𝜋effand𝛽effare
the effective compute throughput and bandwidth, and 𝑇ovhcaptures
fixed per-call costs.
Hardware and execution factors.To derive these effective rates
and compose phase times,RAG-CMaccounts for (1)CPU topology,
including heterogeneous cores and NUMA effects; (2)effective cache
capacityunder sharing among concurrent threads; (3)scattered–
sequential memory access, which is sequential within a block but
random across blocks; and (4)parallel execution, using separate com-
pute and memory thread counts with Amdahl scaling.
6.4 Communication Modeling
Given a deployment,RAG-CM’s communication model estimates
the time to move every inter-stage and intra-LLM payload across its
selected devices (
5 in Fig. 7). It derives these costs from the concrete
topology: every GPU pair carries its own bandwidth and startup
latency(𝛽𝑖𝑗,ℓ𝑖𝑗), and GPU–CPU hops use a host-transfer class—
covering tensor-parallel (TP) and pipeline-parallel (PP) collectives,
data-parallel (DP) replicas, GPU–GPU stage handoffs such as the
prefill→decode KV-cache transfer under 1P/1D disaggregation, and
CPU-side boundaries. A cross-device edge 𝑒sums its fastest endpoint-
disjoint pair bandwidths into 𝛽𝑒and takes its slowest startup as
ℓ𝑒; LLM-internal TP/PP traffic folds into the prefill/decode service
time rather than adding assembly edges; and each explicit cross-
device edge is priced as ℓ𝑒+𝑀𝑒/𝛽𝑒over its payload 𝑀𝑒, with text
and dataframe boundaries (e.g., retrieval →reranker) priced as host
transfers.
6.5RAG-CMAssembly
The assembly layer (
6 in Fig. 7) has two functions: it enumerates
and materializes each s∈X cmas one deployment and execution plan,
then composes stage-level operator and communication predictions
into steady-state, end-to-end serving performance. For each candi-
date it prices, assembly predicts the saturation throughput ˜𝑇(s) and
the corresponding mean latency ˜𝐿(s) once serving reaches steady
state; startup behavior is outside the model.RAG-CMuses the perfor-
mance objective and SLOs in the user’s optimization target to select
9

among these configurations and exposes the selected performance
toRAG-PEas ˜𝑃(x𝑖).
6.5.1 Steady-state performance under saturated closed-loop serving.
We model a saturated closed-loop deployment with a fixed number of
concurrent clients, each issuing its next query after the previous re-
sponse. The resulting ˜𝑇is the maximum sustainable throughput, and
˜𝐿is the steady-state response latency at that operating point. This
target characterizes the deployment’s attainable capacity without
introducing an external arrival rate. An open-system latency instead
depends on the offered load and can make the same deployment
appear lightly loaded, near saturation, or unstable.
6.5.2 Trace-drivenassembly.Foreachalgorithm–systempair (x𝑖,s),
RAG-CMinstantiates a closed discrete-event model fromRAG-IR’s
workload representation and the operator and communication mod-
els (§6.4). The simulation captures four mechanisms: (1)Continuous
and dynamic batching:LLM stages use continuous batching, while
other stages dispatch batches by size or timeout; (2)Automatic pre-
fix caching:repeated calls prefill only the uncached prompt suffix
within KV-cache capacity; (3)Decoupled batch limits:genera-
tor decode and other stages use independent batch-size limits [ 30];
and (4)Shared-resource contention:collocated stages time-share
GPUs and CPUs. At concurrency 𝑁,b𝑇(s,𝑁) andb𝐿(s,𝑁) denote
steady-state throughput (QPS) and mean end-to-end latency, respec-
tively. To scale to thousands of candidates,RAG-CMuses a cheap
analytical model to shortlist the best batch settings per deployment
topology, then simulates only those candidates to throughput sat-
uration and reports only simulation results.
7 Evaluation
Our evaluation is organized around four questions.
RQ1 (End-to-end value).Across random seeds and under the
same evaluation budget, doesRAG-Stackconsistently find better
quality–performance Pareto frontiers than existing optimization
methods?
RQ2(Optimizerablation).With the rest ofRAG-Stackheld fixed,
does the optimizer insideRAG-PEoutperform existing alternatives?
RQ3 (Cost-model accuracy).How accurate isRAG-CM, in abso-
lute error and, more importantly, in ranking candidate deployments?
RQ4 (System transfer).After migrating to new hardware, how
much doesRAG-Stackimprove normalized hypervolume over re-
optimizing from scratch under the same evaluation budget?
7.1 Experimental Setup
Hardware.We use two NUMA servers as measured deployment
targets with different CPU and GPU resources:SysAhas two AMD
EPYC 9124 sockets, 32 CPU cores in total (16 cores per socket, 3.0 GHz
base and up to 3.7 GHz boost), two NUMA nodes, and four NVIDIA
H100 GPUs;SysBhas two AMD EPYC 7742 sockets, 128 CPU cores
in total (64 cores per socket, 2.25 GHz base and up to 3.4 GHz boost),
two NUMA nodes, and eight NVIDIA A100 80GB GPUs.
Datasets and Quality Metrics.For the end-to-end evaluation,
we use 100 queries each from RAGEval [ 74] to avoid contamination
from LLM training data [ 56,67], and MS MARCO [ 9] to evaluate on
a larger, more diverse dataset. TheRAG-CMaccuracy evaluationuses additional datasets, which we describe in §7.4. The quality ob-
jective𝑄is RAGAS answer correctness [ 20], which scores factual
agreement with the gold answer rather than lexical overlap—so valid
paraphrases are not penalized—in [0,1], averaged over queries. As
per-stage diagnostics,RAG-PEreads three RAGAS sub-metrics (§4.2):
context recall, context precision, and faithfulness. All LLM-as-judge
scoring uses DeepSeek-V4-Flash.
Algorithm Design Spaces.Our algorithm design space fol-
lows Fig. 5; here, we specify only the choices not detailed in the
figure, while all other components and parameter ranges remain
as shown. For RAGEval, the chunk size is chosen from {128,256,
512,1024,2048}; the embedding model is BGE-small [ 2] (384-d), all-
mpnet-base-v2 [ 63] (768-d), or BGE-M3 [ 15] (1024-d); the reranker is
TART [ 5], ColBERT [ 36], SentenceTransformer [ 58], or FlagEmbed-
ding [ 2,15]; and the query rewriter and main LLM independently use
a Qwen2.5 [ 8] model from{1.5,3,7,14}B. For MS MARCO, we fix the
chunk size at 2048 and the embedding model to all-mpnet-base-v2
to keep indexing its 8.84M passages tractable.
System Design Spaces.We instantiate the system axes of Table 1.
The end-to-end and optimizer-ablation experiments (§7.2, §7.3) share
the four-H100 SysA space:RAG-CMmaximizes throughput over col-
located and disaggregated GPU-stage placements, all feasible stage-
to-GPU mappings and LLM parallelism plans (at most four GPUs per
stage), request and decode batch sizes in {1,2,4,...,256}with𝐵decode≥
𝐵request , and dynamic-batching waits in {2,10,50}ms; FAISS threads
follow min(retrieval batch,physical cores) under inter-query par-
allelism. For transfer to the eight-A100 SysB (§7.5), only the GPU
budget and per-stage cap grow from four to eight.
Baseline system space.The baselines in §7.2 have no performance
model, so a fair comparison lets them search the algorithm design
space jointly with the system axes as one entangled space. But ev-
ery candidate they propose must be deployed and measured, and
the full system design space would spend most of this budget on
poor system variants. We therefore compress all system axes that
RAG-CMsearches into two categorical choices, expert-curated to be
strong on SysA: eightdeployment presets(collocated TP/PP plans and
1P/1D–2P/2D prefill/decode disaggregation, with non-main-LLM
stages either sharing GPUs with the main LLM or using GPUs left
unused by it) crossed with six𝐵 request×𝐵decode batch presets.
7.2 End-to-End Optimization
Figure 8 compares the measured quality–throughput Pareto fron-
tiers under the protocol in §7.1. Following §2.3, we compute hyper-
volume from linearly normalized quality and raw throughput per
dataset; throughput is log-scaled only in the plots for readability.
We then compute normalized hypervolume independently for each
seed and report the three-seed mean. Against GP+LogNEHVI [ 17],
the strongest baseline on both datasets,RAG-Stackaverages 0.514
versus 0.364 on RAGEval and 0.635 versus 0.259 on MS MARCO; the
corresponding per-seed relative improvements average 52.5% and
153.2%, respectively.RAG-Stack’s per-seed hypervolume exceeds
the strongest baseline’s in all six seed–dataset pairs.
As a secondary benefit,RAG-Stackcompletes the end-to-end
search in 4.59 hours on RAGEval and 3.60 hours on MS MARCO,
faster than all baselines except Greedy-Forward (Table 3). This ef-
ficiency holds even thoughRAG-CMevaluates 570–5,400 system
10

				
						/ (&.1 *-!+,,
	
									

	#,$+,) *!#. !'+/,-+%,##"1+,0 ,"1264537Figure 8: End-to-end quality–performance Pareto fron-
tiers (log-scaled performance). Translucent markers show
per-seed frontiers; each line pools seeds 43–45. Selected
RAG-Stackconfigurations are re-measured on SysA.
Table 3: Mean end-to-end cold-start runtime on SysA.
Optimizer RAGEval (hours) MS MARCO (hours)
Greedy-Forward [3]2.25±0.23 3.16±0.06
GP+LogNEHVI [17]5.60±0.40 5.68±0.51
SMAC [48]8.03±0.42 7.03±0.37
RAG-Stack(ours)4.59±0.51 3.60±1.79
configurations per candidate, because it prices them with the model
rather than deploying each one.
To explain these trade-offs, we inspect representative RAG con-
figurations in Figure 8 and attribute their gains to algorithm and
system choices.
On RAGEval (left),RAG-Stackpoint
2 dominates the baseline-
best point
1 with higher quality at81 ×the throughput. Both use
a 14B generator, but
1 adds 3B multi-query expansion to HNSW
(ef𝑠=128, top-𝑘=16) without reranking and uses a fixed 1P/2D plan
(one prefill and two decode replicas, batch16 ×32, with 14B prefill on
one H100).
2 disables expansion and uses 2048-character chunks
with ColBERT after top- 𝑘=8IVF-PQ to rank the retrieved documents
by relevance, raising answer correctness from 0.594 to 0.638, while
RAG-CMselects a faster serving plan.
GP+LogNEHVI is seed-dependent: only a few of its points rise
aboveRAG-Stack’s frontier, all of them from seed 43 and confined to
narrowthroughputranges,thewidestat
3 (245.6vs.204.1QPS;0.335
vs. 0.326 quality).RAG-Stackstill has higher seed-43 hypervolume,
and SMAC and Greedy-Forward never win on both axes.
3 replaces
RAG-Stack’s top- 𝑘=1plus ColBERT with reranker-free top- 𝑘=4re-
trieval over shorter chunks, then adds 1P/2D disaggregation and a
larger batch. Once seed 43 finds this region, LogNEHVI concentrates
later evaluations nearby; the other seeds never enter it—stochastic
discovery followed by exploitation, not a stable baseline advantage.
On MS MARCO (right),RAG-Stackpoint
5 dominates the quality-
best baseline point
4 with globally best quality at8 .6×the through-
put:
4 relies on top- 𝑘=64, whereas
5 uses top-𝑘=16and aRAG-
CM-selected disaggregated layout. At the throughput extreme,
6
dominates the baselines’ fastest point
7 with1.65×the throughput
and2.2×the quality. Both use a 1.5B generator;
6 retains accurate,
uncompressed HNSW and cuts only to top- 𝑘=1—sufficient because
010203040500.000.150.30Normalized HV RAGEval
010203040500.00.20.40.6
MSMARCO
Evaluation budgetRAG-PE (ours)
SMACGP+LogNEHVI
Greedy-ForwardGreedy-Lookback
Greedy-RAG-PEFigure 9: Optimizer ablation: dominated hypervolume versus
evaluation budget, holding the rest ofRAG-Stackfixed and
varying only the optimizer insideRAG-PE. Lines are means
over 3 seeds; bands show seed min/max.
each fixed 2048-character passage forms one chunk—then paral-
lelizes generation with TP2 ×PP2 and encoding with TP4.
7 instead
combines IVF-PQ-FS with ReAct, whose retrieval errors the small
generator cannot correct.
Across the overlapping quality range in Figure 8, serving at equal
quality is1.5–7.5×cheaper on MS MARCO than on RAGEval, likely
due to public-benchmark leakage into LLM training data [ 56,67].
Public benchmarks may therefore understate the serving cost of
quality, motivating RAGEval as a fresh-data control.
7.3 Optimizer Ablation
To isolate the contribution ofRAG-PE’s optimizer, we keep every
other part ofRAG-Stackunchanged and replace only this optimizer
with each baseline in turn. We compareRAG-PEwith SMAC [ 48],
GP+LogNEHVI[ 10],Greedy-Forward,andGreedy-Lookback[ 3];the
last alternates backward and forward passes after its initial forward
pass. Greedy-RAG-PEis a hybrid that switches from Greedy-Forward
toRAG-PEafter30of50evaluations.Eachmethodrunswithseeds43–
45, and Figure 9 reports mean normalized hypervolume by budget.
RAG-PEis the most consistent optimizer across both datasets:
against the four standalone baselines, it attains the highest mean
hypervolume at 77 of 80 post-warm-start budgets. At budget 50, it
reaches 0.375 versus Greedy-Forward’s 0.327 on RAGEval and 0.681
versus Greedy-Lookback’s 0.599 on MS MARCO, relative gains of
14.8% and 13.8%. No single baseline is consistently strongest: the
strongest method differs between the two datasets in this ablation,
while the end-to-end comparison identifies a different strongest base-
line (§7.2). In contrast,RAG-PEremains strong across settings and
contributes independently toRAG-Stack’s end-to-end advantage..
Because dominated hypervolume also credits configurations in
regions no deployment would use (extremely low quality or through-
put), Figure 10 re-reads the same archives from a deployment per-
spective: the best quality each optimizer can serve once a minimum
throughput is required.
Figure 11 shows that initialization is critical: the Sobol-plus-agent
warm start accounts for the largest share of the final hypervolume.
The remaining channels also make non-trivial contributions to the
final hypervolume.
11

1001011020.20.40.6Best qualityRAGEval
1011020.250.300.350.40
MSMARCO
Throughput requirement (QPS)RAG-PE (ours)
SMACGP+LogNEHVI
Greedy-ForwardGreedy-Lookback
Greedy-RAG-PEFigure 10: Best attainable quality versus throughput re-
quirement: per optimizer, the highest-quality configuration
meeting the requirement within the budget (lines: mean over
3 seeds; bands: seed min/max).
0 10 20
% of evaluationsDLS(agent)Agent proposalUnguided
(Sobol + DLS)Pareto-tension
crossoverInit (Sobol + agent)
19.916.623.819.217.9
010 20 30 40
% of final HV7.67.818.118.847.8Pool win Forced Init
Figure 11: Per-channel contributions toRAG-PE’s results
in §7.3: shares of all quality evaluations (left) and final
hypervolume (right).
Table 4:RAG-CMprediction accuracy. Error is MAPE; 𝜌𝑠and
𝑟denote Spearman and Pearson correlation; 𝑛is the number
of configurations.
SysA SysB
Component𝑛Err.↓𝜌 𝑠↑𝑟↑Err.↓𝜌 𝑠↑𝑟↑
IVF-Flat (Latency) 137 6.7% 0.998 1.000 11.4% 0.993 0.997
IVF-PQ (Latency) 137 11.3% 0.996 0.997 13.2% 0.996 0.998
IVF-PQ-FS (Latency) 137 17.0% 0.989 0.997 12.8% 0.994 0.998
HNSW (Latency) 1080 17.0% 0.986 0.971 20.1% 0.977 0.957
Sequential RAG (Latency) 201 11.9% 0.973 0.955 13.4% 0.965 0.950
Sequential RAG (QPS) 201 11.0% 0.975 0.969 14.6% 0.965 0.950
Agentic RAG (Latency) 30 8.1% 0.966 0.987 13.8% 0.984 0.979
Agentic RAG (QPS) 30 12.2% 0.969 0.974 20.3% 0.967 0.980
RAG Overall (Latency) 231 11.4% 0.978 0.958 13.5% 0.970 0.952
RAG Overall (QPS) 231 11.2% 0.978 0.952 15.3% 0.968 0.952
7.4RAG-CMAccuracy
Table 4 evaluates calibratedRAG-CMpredictions on SysA and SysB
at the operator and RAG pipeline levels. The operator study spans
four vector-search corpora (SIFT1M [ 29], GloVe, GIST1M [ 29], and
Deep1M [ 68]) and two embedded RAG corpora (ELI5 [ 21] and Triv-
iaQA [ 34]), varying batch size and CPU parallelism across HNSW
and IVF variants (full grids in the artifact). The RAG pipeline study
crosses 77 serving configurations with RAGEval [ 74], ELI5, and Triv-
iaQA, covering sequential and ReAct-style agentic workflows and
5 10 20 50 100 200
Performance (QPS)0.40.6Quality (ans. corr.)MSMARCOTransfer (calib., 20 evals)
Transfer (uncalib., 20 evals)From scratch (50 evals)
From scratch (20 evals)
Figure 12: System transfer from SysA to SysB. The transferred
Pareto configurations are re-measured end-to-end on SysB;
all points shown are measured.
varying models, indexes, batching, parallelism, colocation, disaggre-
gation, and dynamic batching; each deployment is compared against
measured saturated closed-loop QPS and mean end-to-end latency.
For guiding system space search, preserving the relative order of
configurations matters more than minimizing absolute prediction
error. Across both systems, calibratedRAG-CMattains Spearman cor-
relations of 0.968–0.978 for latency and QPS, while keeping overall
MAPE at 11.2–15.3%. Even for random-access-heavy IVF-PQ-FS and
HNSW, where MAPE rises to 20.1%, Spearman correlation remains at
least 0.977. Thus, the residual prediction error largely preserves con-
figuration ordering, providing the ranking signal thatRAG-PEneeds.
7.5 System Transfer
We evaluate system transfer by migrating optimized RAG configu-
rations from SysA to SysB.RAG-Stackre-scores the existing quality
archive withRAG-CMinstantiated for SysB, spends 20 additional
evaluations refining the frontier, and re-measures the selected config-
urationsonSysB.TheuncalibratedvariantusesonlySysB’shardware
description, whereas thecalibratedvariant additionally calibrates
RAG-CMon the target machine. Figure 12 compares both variants
with from-scratch optimization on SysB under 20- and 50-evaluation
budgets.
With the same 20-evaluation budget, uncalibrated and calibrated
transfer improve normalized HV over from-scratch optimization by
108.4% (0.331 vs. 0.159) and 182.2% (0.448 vs. 0.159), respectively.
8 Limitations and Discussion
Our current support for agentic RAG is trace-driven:RAG-IRmust ex-
ecute the pipeline to observe its runtime control flow beforeRAG-CM
can predict its performance. Consequently,RAG-PEcannot densely
sample the performance objective before quality evaluation—which
would let a multi-task Gaussian process (MTGP) exploit abundant
performance-only observations to improve sample efficiency—or
filter SLO-violating configurations before they enter the candidate
pool. We have implemented trace-free staticRAG-IRandRAG-CM
paths for sequential RAG, whose control flow is known from the
configuration alone; extending such pre-execution performance
prediction to agentic RAG remains future work.
9 Conclusion
We presentRAG-Stack, a system for efficiently discovering quality–
performance Pareto frontiers across RAG algorithms and serving
12

systems.RAG-StackcombinesRAG-PEfor sub-metric-aware multi-
objective exploration,RAG-IRfor representing executed sequential
and agentic workflows, andRAG-CMfor predicting serving perfor-
mance and searching system configurations without deploying every
candidate. Experiments show thatRAG-Stackimproves normalized
hypervolume over the strongest end-to-end baseline, with three-seed
mean gains of 52.5% on RAGEval and 153.2% on MS MARCO.
Acknowledgments
We sincerely thank Yongjun He and Gustavo Alonso for their in-
sightful discussions and invaluable suggestions throughout the de-
velopment of this work.
References
[1] 2026. Algorithmicsuperintelligence/Openevolve. Algorithmic SuperIntelligence
Labs.
[2] 2026. FlagOpen/FlagEmbedding. FlagOpen.
[3] 2026. Marker-Inc-Korea/AutoRAG. Markr.AI.
[4]Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat,
and Eytan Bakshy. 2025. Unexpected Improvements to Expected Improve-
ment for Bayesian Optimization. https://doi.org/10.48550/arXiv.2310.20708
arXiv:2310.20708 [cs]
[5] Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian
Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. 2023. Task-Aware Retrieval
with Instructions. InFindings of the Association for Computational Linguistics:
ACL 2023, Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.).
Association for Computational Linguistics, Toronto, Canada, 3650–3675.
https://doi.org/10.18653/v1/2023.findings-acl.225
[6] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.
https://doi.org/10.48550/arXiv.2310.11511 arXiv:2310.11511 [cs]
[7]Raul Astudillo and Peter Frazier. 2019. Bayesian Optimization of Composite
Functions. InProceedings of the 36th International Conference on Machine Learning.
PMLR, 354–363.
[8] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan,
Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji
Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men,
Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng
Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang,
Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng
Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang
Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen Technical
Report. https://doi.org/10.48550/arXiv.2309.16609 arXiv:2309.16609 [cs.CL]
[9]Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong
Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir
Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. MS
MARCO: A Human Generated MAchine Reading COmprehension Dataset.
https://doi.org/10.48550/arXiv.1611.09268 arXiv:1611.09268 [cs.CL]
[10] Maximilian Balandat, Brian Karrer, Daniel Jiang, Samuel Daulton, Ben Letham,
Andrew G Wilson, and Eytan Bakshy. 2020. BoTorch: A Framework for Efficient
Monte-Carlo Bayesian Optimization. InAdvances in Neural Information Processing
Systems, Vol. 33. Curran Associates, Inc., 21524–21538.
[11] Abhimanyu Bambhaniya, Ritik Raj, Geonhwa Jeong, Souvik Kundu, Sudarshan
Srinivasan, Suvinay Subramanian, Midhilesh Elavazhagan, Madhu Kumar, and
Tushar Krishna. 2025. Demystifying AI Platform Design for Distributed Inference
of Next-Generation LLM Models. https://doi.org/10.48550/arXiv.2406.01698
arXiv:2406.01698 [cs]
[12] Matthew Barker, Andrew Bell, Evan Thomas, James Carr, Thomas Andrews, and
Umang Bhatt. 2025. Faster, Cheaper, Better: Multi-Objective Hyperparameter Op-
timization for LLM and RAG Systems. https://doi.org/10.48550/arXiv.2502.18635
arXiv:2502.18635 [cs]
[13] Maciej Besta, Lorenzo Paleari, Jia Hao Andrea Jiang, Robert Gerstenberger, You
Wu, Jón Gunnar Hannesson, Patrick Iff, Ales Kubicek, Piotr Nyczyk, Diana Khimey,
Nils Blach, Haiqiang Zhang, Tao Zhang, Peiran Ma, Grzegorz Kwaśniewski, Marcin
Copik, Hubert Niewiadomski, and Torsten Hoefler. 2025. Affordable AI Assistants
with Knowledge Graph of Thoughts. https://doi.org/10.48550/arXiv.2504.02670
arXiv:2504.02670 [cs]
[14] Hans-Georg Beyer and Hans-Paul Schwefel. 2002. Evolution Strategies – A
Comprehensive Introduction.Natural Computing1, 1 (March 2002), 3–52.
https://doi.org/10.1023/A:1015059928466
[15] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu.
2024. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-GranularityText Embeddings Through Self-Knowledge Distillation. InFindings of the
Association for Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins,
and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok,
Thailand, 2318–2335. https://doi.org/10.18653/v1/2024.findings-acl.137
[16] Jian Chen, Peilin Zhou, Yining Hua, Loh Xin, Kehui Chen, Ziyuan Li, Bing Zhu,
and Junwei Liang. 2024. FinTextQA: A Dataset for Long-form Financial Question
Answering. InProceedings of the 62nd Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins,
and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok,
Thailand, 6025–6047. https://doi.org/10.18653/v1/2024.acl-long.328
[17] Samuel Daulton, Maximilian Balandat, and Eytan Bakshy. 2021. Parallel Bayesian
Optimization of Multiple Noisy Objectives with Expected Hypervolume Improve-
ment. https://doi.org/10.48550/arXiv.2105.08195 arXiv:2105.08195 [cs.LG]
[18] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. 2002. A Fast and Elitist
Multiobjective Genetic Algorithm: NSGA-II.IEEE Transactions on Evolutionary
Computation6, 2 (April 2002), 182–197. https://doi.org/10.1109/4235.996017
[19] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy,
Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025.
TheFaissLibrary. https://doi.org/10.48550/arXiv.2401.08281arXiv:2401.08281[cs]
[20] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs:
Automated Evaluation of Retrieval Augmented Generation. InProceedings of
the 18th Conference of the European Chapter of the Association for Computational
Linguistics: System Demonstrations, Nikolaos Aletras and Orphee De Clercq
(Eds.). Association for Computational Linguistics, St. Julians, Malta, 150–158.
https://doi.org/10.18653/v1/2024.eacl-demo.16
[21] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason We-
ston, and Michael Auli. 2019. ELI5: Long Form Question Answering.
https://doi.org/10.48550/arXiv.1907.09190 arXiv:1907.09190 [cs.CL]
[22] Jia Fu, Xiaoting Qin, Fangkai Yang, Lu Wang, Jue Zhang, Qingwei Lin, Yubo
Chen, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. 2024. AutoRAG-HP:
Automatic Online Hyper-Parameter Tuning for Retrieval-Augmented Generation.
https://doi.org/10.48550/arXiv.2406.19251 arXiv:2406.19251 [cs.CL]
[23] Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. 2022. Pre-
cise Zero-Shot Dense Retrieval without Relevance Labels. https:
//doi.org/10.48550/arXiv.2212.10496 arXiv:2212.10496 [cs.IR]
[24] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Ji-
awei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-Augmented Generation
for Large Language Models: A Survey. https://doi.org/10.48550/arXiv.2312.10997
arXiv:2312.10997 [cs.CL]
[25] Nikolaus Hansen. 2023. The CMA Evolution Strategy: A Tutorial.
https://doi.org/10.48550/arXiv.1604.00772 arXiv:1604.00772 [cs.LG]
[26] Aaron Harlap, Deepak Narayanan, Amar Phanishayee, Vivek Seshadri, Nikhil
Devanur, Greg Ganger, and Phil Gibbons. 2018. PipeDream: Fast and Efficient
Pipeline Parallel DNN Training. https://doi.org/10.48550/arXiv.1806.03377
arXiv:1806.03377 [cs.DC]
[27] Yanping Huang, Youlong Cheng, Ankur Bapna, Orhan Firat, Mia Xu Chen, Dehao
Chen, HyoukJoong Lee, Jiquan Ngiam, Quoc V. Le, Yonghui Wu, and Zhifeng
Chen. 2019. GPipe: Efficient Training of Giant Neural Networks Using Pipeline
Parallelism. https://doi.org/10.48550/arXiv.1811.06965 arXiv:1811.06965 [cs.CV]
[28] Pooyan Jamshidi, Miguel Velez, Christian Kästner, and Norbert Siegmund. 2018.
Learning to Sample: Exploiting Similarities across Environments to Learn Per-
formance Models for Configurable Systems. InProceedings of the 2018 26th ACM
Joint Meeting on European Software Engineering Conference and Symposium on the
Foundations of Software Engineering (ESEC/FSE 2018). Association for Computing
Machinery, New York, NY, USA, 71–82. https://doi.org/10.1145/3236024.3236074
[29] Herve Jégou, Matthijs Douze, and Cordelia Schmid. 2011. Product Quantization
for Nearest Neighbor Search.IEEE Transactions on Pattern Analysis and Machine
Intelligence33, 1 (Jan. 2011), 117–128. https://doi.org/10.1109/TPAMI.2010.57
[30] Wenqi Jiang, Suvinay Subramanian, Cat Graves, Gustavo Alonso, Amir
Yazdanbakhsh, and Vidushi Dadu. 2025. RAGO: Systematic Perfor-
mance Optimization for Retrieval-Augmented Generation Serving.
https://doi.org/10.48550/arXiv.2503.14649 arXiv:2503.14649 [cs]
[31] Wenqi Jiang, Marco Zeller, Roger Waleffe, Torsten Hoefler, and Gus-
tavo Alonso. 2025. Chameleon: A Heterogeneous and Disaggre-
gated Accelerator System for Retrieval-Augmented Language Models.
https://doi.org/10.48550/arXiv.2310.09949 arXiv:2310.09949 [cs]
[32] Wenqi Jiang, Shuai Zhang, Boran Han, Jie Wang, Bernie Wang, and Tim Kraska.
2025. PipeRAG: Fast Retrieval-Augmented Generation via Adaptive Pipeline
Parallelism. InProceedings of the 31st ACM SIGKDD Conference on Knowledge
Discovery and Data Mining V.1 (KDD ’25). Association for Computing Machinery,
New York, NY, USA, 589–600. https://doi.org/10.1145/3690624.3709194
[33] Jiajie Jin, Yutao Zhu, Guanting Dong, Yuyao Zhang, Xinyu Yang, Chenghao
Zhang, Tong Zhao, Zhao Yang, Zhicheng Dou, and Ji-Rong Wen. 2025. FlashRAG:
A Modular Toolkit for Efficient Retrieval-Augmented Generation Research.
InCompanion Proceedings of the ACM on Web Conference 2025. 737–740.
https://doi.org/10.1145/3701716.3715313. arXiv:2405.13576 [cs.CL]
[34] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. 2017. TriviaQA:
A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.
13

https://doi.org/10.48550/arXiv.1705.03551 arXiv:1705.03551 [cs.CL]
[35] Kirthevasan Kandasamy, Gautam Dasarathy, Jeff Schneider, and Barnabas Poczos.
2017. Multi-Fidelity Bayesian Optimisation with Continuous Approximations.
https://doi.org/10.48550/arXiv.1703.06240 arXiv:1703.06240 [stat]
[36] Omar Khattab and Matei Zaharia. 2020. ColBERT: Efficient and Ef-
fective Passage Search via Contextualized Late Interaction over BERT.
https://doi.org/10.48550/arXiv.2004.12832 arXiv:2004.12832 [cs.IR]
[37] Junkyum Kim and Divya Mahajan. 2026. VectorLiteRAG: Latency-
Aware and Fine-Grained Resource Partitioning for Efficient RAG.
https://doi.org/10.48550/arXiv.2504.08930 arXiv:2504.08930 [cs]
[38] J. Knowles. 2006. ParEGO: A Hybrid Algorithm with on-Line Landscape
Approximation for Expensive Multiobjective Optimization Problems.
IEEE Transactions on Evolutionary Computation10, 1 (Feb. 2006), 50–66.
https://doi.org/10.1109/TEVC.2005.851274
[39] Akshay Kudva, Wei-Ting Tang, and Joel A. Paulson. 2026. Multi-Objective
Bayesian Optimization for Networked Black-Box Systems: A Path to Greener
Profits and Smarter Designs. https://doi.org/10.48550/arXiv.2502.14121
arXiv:2502.14121 [stat]
[40] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng,
Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient
Memory Management for Large Language Model Serving with PagedAttention.
https://doi.org/10.48550/arXiv.2309.06180 arXiv:2309.06180 [cs.LG]
[41] Jiale Lao, Yibo Wang, Yufei Li, Jianping Wang, Yunjia Zhang, Zhiyuan Cheng,
Wanghu Chen, Mingjie Tang, and Jianguo Wang. 2025. GPTuner: An LLM-
Based Database Tuning System.SIGMOD Rec.54, 1 (April 2025), 101–110.
https://doi.org/10.1145/3733620.3733641
[42] Julien-Charles Lévesque, Audrey Durand, Christian Gagné, and Robert Sabourin.
2017. Bayesian Optimization for Conditional Hyperparameter Spaces. In
2017 International Joint Conference on Neural Networks (IJCNN). 286–293.
https://doi.org/10.1109/IJCNN.2017.7965867
[43] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation
for Knowledge-Intensive NLP Tasks. InProceedings of the 34th International
Conference on Neural Information Processing Systems (NIPS ’20). Curran Associates
Inc., Red Hook, NY, USA, 9459–9474.
[44] Shen Li, Yanli Zhao, Rohan Varma, Omkar Salpekar, Pieter Noordhuis, Teng Li,
Adam Paszke, Jeff Smith, Brian Vaughan, Pritam Damania, and Soumith Chintala.
2020. PyTorch Distributed: Experiences on Accelerating Data Parallel Training.
https://doi.org/10.48550/arXiv.2006.15704 arXiv:2006.15704 [cs.DC]
[45] Shaobo Li, Yirui Zhou, Yuan Xu, Kevin Chen, Daniel Waddington, Swaminathan
Sundararaman, Hubertus Franke, and Jian Huang. 2026. RAGPerf: An End-to-End
Benchmarking Framework for Retrieval-Augmented Generation Systems.
https://doi.org/10.48550/arXiv.2603.10765 arXiv:2603.10765 [cs.PF]
[46] Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh Huang, Yaozu Wu, Junyu
Luo, Yuanchen Bei, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit Chan,
Yankai Chen, Zhongfen Deng, Yinghui Li, Hai-Tao Zheng, Dongyuan Li, Renhe
Jiang, Ming Zhang, Yangqiu Song, and Philip S. Yu. 2025. Towards Agentic
RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs.
https://doi.org/10.48550/arXiv.2507.09477 arXiv:2507.09477 [cs]
[47] Ning Liang, Fabian Wenz, Jana Giceva, and Lisa Wu Wills. 2025. Athena: A
Plug-and-Play Advisor for Retrieval-Augmented Generation Using VectorDB. In
2025 IEEE International Symposium on Workload Characterization (IISWC). 28–41.
https://doi.org/10.1109/IISWC66894.2025.00013
[48] Marius Lindauer, Katharina Eggensperger, Matthias Feurer, André Biedenkapp,
Difan Deng, Carolin Benjamins, Tim Ruhopf, René Sass, and Frank Hutter.
2022. SMAC3: A Versatile Bayesian Optimization Package for Hyperparameter
Optimization. https://doi.org/10.48550/arXiv.2109.09831 arXiv:2109.09831 [cs]
[49] Tennison Liu, Nicolás Astorga, Nabeel Seedat, and Mihaela van der
Schaar. 2024. Large Language Models to Enhance Bayesian Optimization.
https://doi.org/10.48550/arXiv.2402.03921 arXiv:2402.03921 [cs]
[50] Zhichao Lu, Gautam Sreekumar, Erik Goodman, Wolfgang Banzhaf, Kalyanmoy
Deb, and Vishnu Naresh Boddeti. 2021. Neural Architecture Transfer.IEEE
Transactions on Pattern Analysis and Machine Intelligence43, 9 (Sept. 2021),
2971–2989. https://doi.org/10.1109/TPAMI.2021.3052758
[51] Yu A. Malkov and D. A. Yashunin. 2018. Efficient and Robust Approximate
Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.
https://doi.org/10.48550/arXiv.1603.09320 arXiv:1603.09320 [cs]
[52] Alexander Novikov, Ngân V ˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen
Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco
J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri,
George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej
Balog. 2025. AlphaEvolve: A Coding Agent for Scientific and Algorithmic
Discovery. https://doi.org/10.48550/arXiv.2506.13131 arXiv:2506.13131 [cs.AI]
[53] Miles Olson, Elizabeth Santorella, Louis C. Tiao, Sait Cakmak, Mia Garrard,
Samuel Daulton, Zhiyuan Jerry Lin, Sebastian Ament, Bernard Beckerman,
Eric Onofrey, Paschal Igusti, Cristian Lara, Benjamin Letham, Cesar Cardoso,
Shiyun Sunny Shen, Andy Chenyuan Lin, Matthew Grange, Elena Kashtelyan,David Eriksson, Maximilian Balandat, and Eytan Bakshy. 2025. Ax: A Platform for
Adaptive Experimentation. InProceedings of the Fourth International Conference
on Automated Machine Learning. PMLR, 21/1–25.
[54] Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang,
Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu,
and Dongmei Zhang. 2024. LLMLingua-2: Data Distillation for Efficient and
Faithful Task-Agnostic Prompt Compression. InFindings of the Association for
Computational Linguistics: ACL 2024, Lun-Wei Ku, Andre Martins, and Vivek
Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand,
963–981. https://doi.org/10.18653/v1/2024.findings-acl.57
[55] Biswajit Paria, Kirthevasan Kandasamy, and Barnabás Póczos. 2019. A Flex-
ible Framework for Multi-Objective Bayesian Optimization Using Random
Scalarizations. https://doi.org/10.48550/arXiv.1805.12168 arXiv:1805.12168 [cs]
[56] Zehan Qi, Rongwu Xu, Zhijiang Guo, Cunxiang Wang, Hao Zhang,
and Wei Xu. 2025. Long$^2$RAG: Evaluating Long-Context &
Long-Form Retrieval-Augmented Generation with Key Point Recall.
https://doi.org/10.48550/arXiv.2410.23000 arXiv:2410.23000 [cs]
[57] Siddhant Ray, Rui Pan, Zhuohan Gu, Kuntai Du, Shaoting Feng, Ganesh
Ananthanarayanan, Ravi Netravali, and Junchen Jiang. 2025. METIS:
Fast Quality-Aware RAG Systems with Configuration Adaptation.
https://doi.org/10.48550/arXiv.2412.10543 arXiv:2412.10543 [cs.LG]
[58] Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings
Using Siamese BERT-Networks. https://doi.org/10.48550/arXiv.1908.10084
arXiv:1908.10084 [cs.CL]
[59] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper,
and Bryan Catanzaro. 2020. Megatron-LM: Training Multi-Billion Parameter Lan-
guage Models Using Model Parallelism. https://doi.org/10.48550/arXiv.1909.08053
arXiv:1909.08053 [cs.CL]
[60] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021.
Retrieval Augmentation Reduces Hallucination in Conversation. InFindings
of the Association for Computational Linguistics: EMNLP 2021, Marie-Francine
Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (Eds.). Association
for Computational Linguistics, Punta Cana, Dominican Republic, 3784–3803.
https://doi.org/10.18653/v1/2021.findings-emnlp.320
[61] Aditi Singh, Abul Ehtesham, Saket Kumar, Tala Talaei Khoei, and Athanasios V.
Vasilakos. 2026. Agentic Retrieval-Augmented Generation: A Survey on Agentic
RAG. https://doi.org/10.48550/arXiv.2501.09136 arXiv:2501.09136 [cs]
[62] Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kalu-
arachchi, Rajib Rana, and Suranga Nanayakkara. 2023. Improving the Domain
Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain
Question Answering.Transactions of the Association for Computational Linguistics
11 (2023), 1–17. https://doi.org/10.1162/tacl_a_00530
[63] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. 2020. MP-
Net: Masked and Permuted Pre-training for Language Understanding.
https://doi.org/10.48550/arXiv.2004.09297 arXiv:2004.09297 [cs.CL]
[64] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei,
Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, and Thang Luong.
2024. FreshLLMs: Refreshing Large Language Models with Search Engine
Augmentation. InFindings of the Association for Computational Linguistics:
ACL 2024, Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Asso-
ciation for Computational Linguistics, Bangkok, Thailand, 13697–13720.
https://doi.org/10.18653/v1/2024.findings-acl.813
[65] Jianguo Wang, Xiaomeng Yi, Rentong Guo, Hai Jin, Peng Xu, Shengjun Li,
Xiangyu Wang, Xiangzhou Guo, Chengming Li, Xiaohai Xu, Kun Yu, Yuxing Yuan,
Yinghao Zou, Jiquan Long, Yudong Cai, Zhenxiang Li, Zhifeng Zhang, Yihua Mo,
Jun Gu, Ruiyi Jiang, Yi Wei, and Charles Xie. 2021. Milvus: A Purpose-Built Vector
Data Management System. InProceedings of the 2021 International Conference
on Management of Data (SIGMOD ’21). Association for Computing Machinery,
New York, NY, USA, 2614–2627. https://doi.org/10.1145/3448016.3457550
[66] Shuhei Watanabe. 2026. Tree-Structured Parzen Estimator: Understanding
Its Algorithm Components and Their Roles for Better Empirical Performance.
https://doi.org/10.48550/arXiv.2304.11127 arXiv:2304.11127 [cs.LG]
[67] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng,
Chongyang Tao, Qingwei Lin, and Daxin Jiang. 2025. WizardLM: Empow-
ering Large Pre-Trained Language Models to Follow Complex Instructions.
https://doi.org/10.48550/arXiv.2304.12244 arXiv:2304.12244 [cs]
[68] Artem Babenko Yandex and Victor Lempitsky. 2016. Efficient Indexing of Billion-
Scale Datasets of Deep Descriptors. In2016 IEEE Conference on Computer Vision and
Pattern Recognition (CVPR). 2055–2063. https://doi.org/10.1109/CVPR.2016.226
[69] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan,
and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language
Models. https://doi.org/10.48550/arXiv.2210.03629 arXiv:2210.03629 [cs.CL]
[70] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon
Chun. 2022. Orca: A Distributed Serving System for Transformer-Based
Generative Models. In16th USENIX Symposium on Operating Systems Design and
Implementation (OSDI 22). 521–538.
[71] Qingfu Zhang and Hui Li. 2007. MOEA/D: A Multiobjective Evolutionary Algo-
rithm Based on Decomposition.IEEE Transactions on Evolutionary Computation
14

11, 6 (Dec. 2007), 712–731. https://doi.org/10.1109/TEVC.2007.892759
[72] Tianyang Zhang, Zhuoxuan Jiang, Shengguang Bai, Tianrui Zhang, Lin Lin,
Yang Liu, and Jiawei Ren. 2024. RAG4ITOps: A Supervised Fine-Tunable
and Comprehensive RAG Framework for IT Operations and Maintenance. In
Proceedings of the 2024 Conference on Empirical Methods in Natural Language
Processing: Industry Track, Franck Dernoncourt, Daniel Preoţiuc-Pietro, and
Anastasia Shimorina (Eds.). Association for Computational Linguistics, Miami,
Florida, US, 738–754. https://doi.org/10.18653/v1/2024.emnlp-industry.56[73] Tao Zhang, Kaixian Qu, Zhibin Li, Jiajun Wu, Marco Hutter, Manling Li, and Fan Shi.
2026. Using Large Language Models for Embodied Planning Introduces Systematic
Safety Risks. https://doi.org/10.48550/arXiv.2604.18463 arXiv:2604.18463 [cs.AI]
[74] Kunlun Zhu, Yifan Luo, Dingling Xu, Yukun Yan, Zhenghao Liu, Shi Yu, Ruobing
Wang, Shuo Wang, Yishan Li, Nan Zhang, Xu Han, Zhiyuan Liu, and Maosong
Sun. 2025. RAGEval: Scenario Specific RAG Evaluation Dataset Generation
Framework. https://doi.org/10.48550/arXiv.2408.01262 arXiv:2408.01262 [cs]
15