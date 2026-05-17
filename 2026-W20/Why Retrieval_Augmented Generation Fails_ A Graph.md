# Why Retrieval-Augmented Generation Fails: A Graph Perspective

**Authors**: Kai Guo, Xinnan Dai, Zhibo Zhang, Nuohan Lin, Shenglai Zeng, Jie Ren, Haoyu Han, Jiliang Tang

**Published**: 2026-05-13 23:18:07

**PDF URL**: [https://arxiv.org/pdf/2605.14192v1](https://arxiv.org/pdf/2605.14192v1)

## Abstract
Retrieval-Augmented Generation (RAG) has become a powerful and widely used approach for improving large language models by grounding generation in retrieved evidence. However, RAG systems still produce incorrect answers in many cases. Why RAG fails despite having access to external information remains poorly understood. We present a model-internal study of retrieval-augmented generation that examines how retrieved evidence influences answer generation. Using circuit tracing, we construct attribution graphs that model the flow of information through transformer layers during decoding. These graphs represent interactions among retrieved context, intermediate model activations, and generated tokens, providing a graph, circuit-level view of how external evidence is integrated into the model's reasoning process across multiple question answering benchmarks, we observe consistent structural differences: correct predictions exhibit deeper reasoning paths, more distributed evidence flow, and a more structured pattern of local connectivity, while failed predictions show shallower, fragmented, and overly concentrated evidence flow. Building on these findings, we develop a graph-based error detection framework that uses attribution-graph topology features. Furthermore, we show that attribution graphs enable targeted interventions. By reinforcing question-constrained evidence grounding, we reshape internal routing so that answer generation remains guided by the question, leading to more effective integration of retrieved information and fewer errors.

## Full Text


<!-- PDF content starts -->

Why Retrieval-Augmented Generation Fails: A Graph Perspective
Kai Guo
Michigan State University
guokai1@msu.eduXinnan Dai
Michigan State University
daixinna@@msu.eduZhibo Zhang
Michigan State University
zhan2185@msu.edu
Nuohan Lin
Michigan State University
linnuoha@msu.eduShenglai Zeng
Michigan State University
zengshe1@msu.eduJie Ren
Massachusetts Institute of Technology
rjie@mit.edu
Haoyu Han
Michigan State University
hanhaoy1@msu.eduJiliang Tang
Michigan State University
tangjili@msu.edu
Abstract
Retrieval-Augmented Generation (RAG) has become a powerful
and widely used approach for improving large language models
by grounding generation in retrieved evidence. However, RAG sys-
tems still produce incorrect answers in many cases. Why RAG fails
despite having access to external information remains poorly under-
stood. We present a model-internal study of retrieval-augmented
generation that examines how retrieved evidence influences an-
swer generation. Using circuit tracing, we construct attribution
graphs that model the flow of information through transformer
layers during decoding. These graphs represent interactions among
retrieved context, intermediate model activations, and generated
tokens, providing a graph, circuit-level view of how external evi-
dence is integrated into the model’s reasoning process across mul-
tiple question answering benchmarks, we observe consistent struc-
tural differences: correct predictions exhibit deeper reasoning paths,
more distributed evidence flow, and a more structured pattern of
local connectivity, while failed predictions show shallower, frag-
mented, and overly concentrated evidence flow. Building on these
findings, we develop a graph-based error detection framework that
uses attribution-graph topology features. Furthermore, we show
that attribution graphs enable targeted interventions. By reinforc-
ing question-constrained evidence grounding, we reshape internal
routing so that answer generation remains guided by the question,
leading to more effective integration of retrieved information and
fewer errors.
CCS Concepts
•Computing methodologies→Neural networks.
Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
Conference acronym ’XX, Woodstock, NY
©2018 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 978-1-4503-XXXX-X/2018/06
https://doi.org/XXXXXXX.XXXXXXXKeywords
Retrieval-Augmented Generation, Attribution Graph, Large Lan-
guage Model
ACM Reference Format:
Kai Guo, Xinnan Dai, Zhibo Zhang, Nuohan Lin, Shenglai Zeng, Jie Ren,
Haoyu Han, and Jiliang Tang. 2018. Why Retrieval-Augmented Generation
Fails: A Graph Perspective. InProceedings of Make sure to enter the correct
conference title from your rights confirmation email (Conference acronym ’XX).
ACM, New York, NY, USA, 12 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Retrieval-Augmented Generation (RAG) has become a central para-
digm for improving large language models by grounding generation
in external evidence [ 6,18,21,25,33,45]. By retrieving relevant
documents at inference time and conditioning the model on this
information, RAG systems aim to reduce incorrect predictions and
improve factual reliability [ 3,4,23,29,31]. Despite these advan-
tages, incorrect outputs remain common even when the retrieved
passages contain the necessary evidence. This suggests that the
presence of evidence alone does not guarantee that it is faithfully
integrated into the model’s reasoning process [19, 20, 32, 38, 46].
Existing work to investigate RAG failures focuses primarily on
retrieval quality or consistency at the output-level [ 14,36]. Some
methods improve retrievers or re-rank retrieved documents, while
others detect errors using answer–document overlap or model con-
fidence [ 24,39,41]. Although these approaches provide useful diag-
nostic indicators, they offer limited insight into the model-internal
reasoning dynamics that lead to unfaithful generation. Recent stud-
ies have explored hidden-state representations as diagnostic signals
for knowledge checking [ 42]. However, such approaches typically
rely on representations from a single layer and only provide a
largely static view of the model’s internal state [ 26]. As a result,
they do not characterize how the retrieved evidence is propagated,
transformed, and combined across layers during decoding. This
highlights the need for a methodological framework that explicitly
captures internal evidence flow, enabling a granular understanding
of knowledge aggregation.
In this work, we take a graph perspective on RAG reasoning.
Instead of examining only inputs and outputs, we analyze how
retrieved evidence propagates through the model during decod-
ing. We utilize the circuit tracing technique [ 1] to build topologicalarXiv:2605.14192v1  [cs.CL]  13 May 2026

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
features to quantify how context tokens influence intermediate acti-
vations and final answer tokens. We then translate these attribution
signals into attribution graphs, which represent information flow
among retrieved tokens, intermediate components, and generated
outputs. This graph-based representation enables us to perform
direct structural analysis of reasoning processes across examples.
Therefore we conduct a systematic study of both correct and incor-
rect RAG predictions. We observe consistent structural differences
across datasets. Correct predictions exhibit deeper reasoning paths,
more distributed evidence flow, and a more structured local connec-
tivity. In contrast, incorrect predictions show shallower, fragmented,
and overly concentrated evidence flow.
To further provide a clear explanation of why failures occur, we
focus on a mixed-context setting in which retrieved passages con-
tain both supporting and distracting information. This scenario is
particularly diagnostic, as successful reasoning requires selectively
integrating the truly relevant evidence rather than relying on super-
ficial question–context overlap. Tracing internal information flow
under this condition reveals a recurring failure mode that we term
surface-aligned evidence grounding (SAEG): evidence only su-
perficially matches the question but lacks deep understanding of
the question and sustained influence from it, while generation be-
comes increasingly dominated by retrieved context. In contrast,
correct predictions often exhibitquestion-constrained evidence
grounding (QCEG), where the model places stronger emphasis
on understanding the question and retrieved evidence remains con-
sistently regulated by the question’s semantic constraints, forming
deeper and more integrated reasoning structures.
Overall, our study establishes attribution-graph structure as a
practical and interpretable lens for understanding evidence-grounding
failures in RAG systems. Building on the above insights, we develop
model-internal error detection methods and targeted inference-
time interventions that directly regulate internal routing dynamics.
These approaches not only detect incorrect predictions but can
also steer some failures toward correct outcomes, demonstrating
the practical utility of our mechanistic understanding. Our main
contributions are summarized as follows.
•We use circuit tracing to derive attribution graphs for RAG
models, enabling a graph-based analysis of evidence propa-
gation and influence.
•We identify consistent structural differences between cor-
rect and incorrect predictions, showing that many RAG er-
rors stem from insufficient question understanding and over-
reliance on retrieved context.
•We develop a graph-based error detection framework that
operates purely on internal model dynamics.
•We demonstrate that attribution-graph analysis enables tar-
geted inference-time interventions that promotequestion-
constrained evidence grounding (QCEG), thereby reduc-
ing incorrect predictions during generation.
2 Related Work
Due to space limitations, we provide a brief overview of the most rel-
evant prior work here and defer a more comprehensive discussion
to the Appendix A.1.Retrieval-Augmented Generation.Retrieval-Augmented Gen-
eration (RAG) improves the factuality and reasoning of large lan-
guage models by grounding generation in external knowledge [ 16,
43]. Prior work has explored dense and hybrid retrieval, multi-
hop evidence gathering, iterative retrieval–generation loops, and
query reformulation [ 5,28,34,36]. Other efforts enhance robust-
ness through context selection, reranking, compression, and prompt
engineering [2, 11].
Despite these advances, most approaches treat the language
model as a black box and focus on system-level improvements,
offering limited insight into how retrieved evidence is internally
processed. Consequently, they cannot fully explain why errors
persist even when relevant evidence is successfully retrieved. Some
recent work evaluates faithfulness and evidence usage [ 26,42], but
typically relies on representations from a single layer, providing
only a static and partial view of the model’s internal computation.
Interpretability and Circuit Analysis of LLMs.A parallel
line of research investigates transformer internals using atten-
tion analysis, Sparse Autoencoders, transcoders, and circuit trac-
ing [ 7,9,13,15,37]. These methods decompose neural representa-
tions into interpretable components and reveal that specific behav-
iors can often be attributed to distributed circuits spanning layers
and heads [ 17,30]. Attribution graphs have emerged as a useful
abstraction for modeling information flow within networks [27].
Although circuit-level analyses have shed light on reasoning in
standalone language models [ 10,44], they rarely consider retrieval-
augmented settings. As a result, how externally retrieved evidence
interacts with internal computational circuits in RAG remains
largely unexplored.
3 Background and Preliminaries
In this section, we formally define the attribution graph and describe
how it is constructed from the internal computation of a transformer
model.
3.1 Definition of Attribution Graphs
We represent token-level causal interactions inside the model in
a graph view. In particular, we model the interactions among the
activations as a directed attribution graph 𝐺=(𝑉,𝐸) that captures
how information flows between token representations across layers
during inference.
Each node𝑣𝑡,ℓ∈𝑉corresponds to the representation of token
position𝑡at transformer layer ℓ. A directed edge(𝑣𝑠,𝑘→𝑣𝑡,ℓ)∈𝐸
indicates that the token state at position 𝑠in layer𝑘contributes
to the token state at position 𝑡in layerℓ. The edge weight 𝑤mea-
sures the strength of this causal contribution. This graph-level view
allows us to analyze model reasoning as a structured computa-
tional process, revealing how evidence is integrated, propagated,
and transformed as representations evolve across layers.
3.2 Constructing Attribution Graphs
We now describe how token-level attribution graphs are constructed
from a transformer model.
Feature Decomposition as the Node Basis.Following prior work
on circuit tracing and attribution [ 13], we adopt transcoders to de-
compose residual stream activations when building the attribution

Why Retrieval-Augmented Generation Fails: A Graph Perspective Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
graph for a fixed target logit. At each layer ℓand token position 𝑡,
the residual stream vector is represented as a sparse set of learned
activation units, which serve as intermediate carriers of attribution
signals.
Attribution is computed at the level of activation units, reflecting
how each unit contributes—directly or indirectly—to the target logit
through the network. These activation-unit-level attributions are
then aggregated by token position, so that tokens in the prompt,
retrieved context, and generated output correspond to nodes in the
attribution graph.
Edge Construction via a Linearized Replacement Model.Edge
weights in the attribution graph are obtained using a locally lin-
earized replacement model, following existing circuit-tracing meth-
ods [ 1]. Specifically, we replace MLP blocks with their correspond-
ing transcoders while keeping attention modules unchanged, and fix
attention patterns and layer-normalization terms at their forward-
pass values. Under this setting, the network computation is linear
with respect to activation-unit activations.
This linearization allows attribution signals with respect to the
target logit to be decomposed into additive contributions between
activation units. These unit-level attributions are aggregated across
units associated with each token pair, yielding directed token-to-
token attribution scores, which are used as edge weights in the
attribution graph.
4 Circuit Analysis for RAG
In this section, we analyze the internal circuit structure of RAG
models to understand how retrieved evidence is integrated dur-
ing answer generation. We begin with a general retrieval setting,
where the retrieved context is treated as unconstrained. Under this
setting, we compare the attribution-graph structures ofcorrect
and incorrectpredictions to identify systematic differences in how
information flows through the model’s internal computation.
To probe the failure mechanism more directly, we introduce a
more challengingmixed-context settingin which retrieved pas-
sages intentionally include both supporting and non-supporting in-
formation. This scenario better reflects realistic retrieval conditions
and places stronger demands on the model’s ability to distinguish
relevant evidence from noise. Analyzing circuit behavior under this
mixed setting allows us to study how incorrect reasoning emerges
when the model fails to selectively ground its predictions in truly
supportive context.
4.1 Circuit Analysis of Correct and Incorrect
Predictions
This section uses attribution graphs to analyze differences between
correct and incorrect predictions in how models internally organize
and integrate information. We examine the structural properties of
the model’s internal computation during answer generation. These
structural patterns provide insight into the internal mechanisms
that distinguish successful from unsuccessful prediction use.
4.1.1 Graph Metrics.To understand why some predictions success-
fully integrate retrieved evidence while others do not, we examine
the structural organization of their attribution graphs. Given an
attribution graph 𝐺=(𝑉,𝐸) defined in Section 3.1, each exampleis summarized through a set of graph-level statistics. Rather than
characterizing individual graphs separately, our goal is to identify
systematic structural differences between correct and incorrect
reasoning circuits.
We hypothesize that correct and incorrect predictions differ
along three fundamental dimensions of internal evidence integra-
tion: (1) how far information propagates through the model, (2)
how strongly token representations interact with one another, and
(3) how information is organized across local and global structures.
We therefore design a set of graph metrics that quantify each of
these aspects.
Propagation depth.The first dimension concerns the depth of in-
formation propagation. Correct reasoning may require evidence to
travel through multiple intermediate representations, whereas shal-
low propagation may reflect shortcut or surface-level processing.
We measure this using the longest directed path length, DAG-L(𝑮)=
max𝜋∈P(𝐺)|𝜋|, whereP(𝐺) is the set of directed paths in 𝐺. Larger
values indicate longer multi-step propagation chains, suggesting
more compositional reasoning.
Interaction strength.The second dimension captures how strongly
token representations interact during computation. If evidence is
effectively integrated, we expect richer connectivity among tokens
rather than isolated or weakly connected fragments.
We measure this using two complementary metrics. The aver-
age degree AvgDeg(𝑮)=1
|𝑉|Í
𝑣∈𝑉(degin(𝑣)+degout(𝑣))captures
the typical number of interactions per token. Here, degin(𝑣)and
degout(𝑣)denote the in-degree and out-degree of node 𝑣. The di-
rected edge density Dens(𝑮)=|𝐸|
|𝑉|(|𝑉|−1)measures how densely
the reasoning circuit is connected overall. Higher values indicate
stronger and more widespread evidence interaction.
Structural organization across scales.We further characterize how
information is organized at both local and global scales.
Local fragmentation is captured by the fraction of disconnected
triads, 𝑻disc(𝑮)= #disc(𝐺)Í
𝜏#𝜏(𝐺) , where#𝑑𝑖𝑠𝑐 is a discon-
nected triad consists of three nodes with no edges among them.
Larger values indicate that nearby nodes fail to interact, suggesting
fragmented local structure.
Branching-style local aggregation is measured by 𝑻branch (𝑮)=
#branch(𝐺)Í
𝜏#𝜏(𝐺) , where a branch triad is a three-node pat-
tern in which two nodes both point to the same third node. Higher
values indicate that information from multiple sources tends to
merge into a single intermediate node, reflecting localized aggrega-
tion rather than linear propagation.
Finally, global concentration of information flow is captured by
MaxPR(𝑮)=max 𝑣∈𝑉PR(𝑣) , where PR(𝑣) is the PageRank score of
node𝑣. Larger values indicate that information flow is dominated
by a single hub, whereas lower values suggest more distributed
integration.
Together, these six metrics provide a structural signature of each
reasoning circuit. By comparing these signatures between correct
and incorrect predictions, we can identify how successful evidence
integration differs from failure at the level of internal computation.
4.1.2 A Study of Correct and Incorrect Question Answering.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Setup.We study QA benchmarks (HotpotQA [ 40], 2WikiMulti-
hopQA [ 22], and MuSiQue [ 35]), where answering requires compos-
ing evidence across multiple passages. For each query, we retrieve
a fixed-size context and generate chain-of-thought answers using
LLaMA-3 8B Instruct [ 12] with greedy decoding. We then assign a
binary label 𝑦∈{ 0,1}(incorrect or correct) using an external LLM-
based judge, Gemini-2.5-Flash-Lite [ 8]. Finally, for each dataset,
we construct a balanced set of attribution graphs comprising 500
incorrect and 500 correct predictions to enable directly comparable
structural analyses across classes.
Within each dataset, we construct balanced subsets consisting of
500 correct and 500 incorrect predictions, ensuring that structural
comparisons between attribution graphs are directly comparable
across classes.
Finding 1: Correct Answers Arise from Deeper, More Struc-
tured, and More Evenly Distributed Circuits.A consistent struc-
tural contrast emerges across datasets:correct reasoning circuits
are deep, densely interconnected, and broadly distributed,
whereas wrong circuits are shallow, sparse, and overly cen-
tralized.
Figure 1 shows this separation across all structural metrics. Cor-
rect answers are supported by attribution graphs that are deeper,
more structurally organized, and more evenly distributed in how
evidence is utilized. Incorrect answers, in contrast, arise from cir-
cuits that are shallow, fragmented, and overly concentrated around
a few dominant nodes. In addition to LLaMA-3 8B Instruct, we
also analyze Qwen-3 8B in Figure 8 in Appendix. The results show
consistent conclusions: correct and incorrect predictions exhibit
clearly different structural patterns.
Deeper vs. shallower propagation.Correct graphs exhibit
longer directed propagation depth (higher DAG-L), indicating that
evidence signals travel through multi-step internal routes before
reaching the final answer tokens. Incorrect predictions show shorter
directed paths, suggesting that reasoning is truncated and relies on
fewer intermediate transformations.
Structured vs. fragmented connectivity.Correct circuits are
more structurally organized at both global and local levels. Globally,
they display higher interaction richness, reflected in larger aver-
age degree (AvgDeg) and edge density (Dens), indicating stronger
cross-token coupling and more integrated evidence flow. Locally,
they contain fewer disconnected triads (lower 𝑇𝑑𝑖𝑠𝑐), meaning that
neighboring nodes are more likely to participate in coordinated
interactions. Incorrect circuits, by contrast, are sparser and more
fragmented, with many local node groups remaining structurally
isolated. In addition, higher branching motifs (higher 𝑇branch ) in-
dicate that intermediate states more often distribute information
to multiple downstream components, reflecting a more structured
and distributed reasoning process.
Distributed vs. concentrated evidence flow.Finally, correct
circuits make use of evidence in a more evenly distributed manner.
They exhibit lower maximum PageRank values (MaxPR), indicat-
ing that importance is spread across multiple nodes rather than
dominated by a single hub.
Taken together, these patterns show that correct reasoning emerges
from circuits that are deep, structurally coherent, and broadly dis-
tributed in their use of evidence. Incorrect answers, by contrast,
Figure 1: Radar comparison of attribution-graph structural
metrics between correct and wrong predictions across three
QA datasets (2Wiki, HotpotQA, and MuSiQue).
Figure 2: Layer-wise attribution mass for correct and wrong
predictions (left) and their difference on MuSiQue (right).
arise from shallow, fragmented, and overly centralized circuits in
which information either fails to propagate sufficiently or becomes
concentrated on a small set of dominant nodes.
Finding 2: Correct Predictions Use More Mid-Layer Process-
ing.In addition to graph structure, we analyze how attribution
mass is distributed across transformer layers. Figure 2 shows a clear
depth shift between correct and wrong predictions on MuSique.
Due to space constraints, the corresponding results for HotpotQA
and 2Wiki are provided in Figure 9 and Figure 10 in the Appendix.
Correct predictions allocate a larger fraction of the total activated
neurons to the middle layers (approximately layers 8–18), indicating
greater reliance on mid-layer reasoning computations. These layers
are where the model typically combines information from different
tokens and builds integrated representations. The higher activity
here suggests that correct answers depend on sustained internal
processing that brings together evidence from the question and
retrieved context.
Incorrect predictions follow a different pattern. They rely more
on early layers. Higher early-layer activity indicates that the model
may be matching surface-level patterns from the retrieved text
without deeper integration.
Overall, correct answers are associated with deeper and more
sustained information processing, while wrong answers tend to
reflect shallower decision dynamics.
4.2 Circuit Analysis under Mixed Context
The structural circuit analysis above establisheswhatdifferentiates
correct and incorrect predictions: correct reasoning is supported by
deeper, more connected, and more integrative attribution graphs,
whereas incorrect predictions rely on fragmented and weakly coor-
dinated structures. We now turn to a complementary question:how
do these structural differences emerge over the course of computation?

Why Retrieval-Augmented Generation Fails: A Graph Perspective Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Figure 3: Region-level attribution comparison between cor-
rect and wrong predictions in the mixed-context setting. Bars
show relative weights for𝑄→𝑄, and𝑄→Ans_EXT.
Figure 4: Layer-wise attribution comparison for 𝑄→
Ans_EXT and𝑄→𝑄 . Left: mean routing strength per layer
for correct and wrong predictions. Right: relative differences
(green = correct higher, red = wrong higher).
To address this, we examine the routing dynamics of information
across layers during decoding. Our analysis focuses on a mixed-
context setting. In this setting, each question is paired with re-
trieved context that intentionally contains both supporting and
non-supporting passages.
This scenario is particularly informative because the model must
not only leverage external evidence, but also distinguish relevant
signals from distractors while remaining aligned with the question.
As a result, it provides a controlled testbed for analyzing how inter-
nal routing dynamics differ when evidence selection and integration
become genuinely challenging.
We analyze these dynamics by grouping tokens into functional
regions and tracking how attribution mass flows between them
across layers. This provides a stage-wise view of how the model
balances question understanding, reliance on externally grounded
answer content, and internally composed answer representations
during reasoning.
4.2.1 Region-Level Routing Decomposition.We partition tokens
into three functional regions based on their roles in the reason-
ing process: 𝑄is the input question tokens; Ans_EXT indicates
answer tokens that are attributable to retrieved external context;
Figure 5: Layer-wise attribution weight for multiple region-
level edge types under mixed-context decoding, comparing
correct and wrong predictions.
andAns_INT denotes answer tokens that are generated internally
by the model and have no direct alignment with retrieved context.
For each transformer layer ℓ, we measure how attribution flows
between these regions. Let 𝑎(𝑖→𝑗) denote the weight from source
token𝑖to target token 𝑗at layerℓWe then aggregate attribution
mass at the region level as
𝐴(ℓ)
𝑋→𝑌=∑︁
𝑖∈𝑋∑︁
𝑗∈𝑌𝑎(𝑖→𝑗), 𝑋,𝑌∈{𝑄,Ans_EXT,Ans_INT},
(1)
where𝐴(ℓ)
𝑋→𝑌measures the total attribution routed from region 𝑋to
region𝑌at layerℓby summing over all source–target token pairs
(𝑖,𝑗) with𝑖∈𝑋 and𝑗∈𝑌 . This aggregation yields a layer-wise
routing profile that reveals how the model distributes computation
between understanding the question, leveraging externally aligned
answer content, and developing internally composed answer repre-
sentations.
4.2.2 A Study of Reasoning Patterns in the Mixed-Context Setting.
Setup.Our analysis centers on a mixed-context setting derived
from MuSiQue, which we termMix-MuSiQue. In this variant, each
question is paired with a retrieved context deliberately constructed
to include both supporting and non-supporting passages. This de-
sign creates a controlled mixed-evidence scenario that tests the
model’s ability to selectively use relevant information while ig-
noring distractors. The dataset comprises 667 questions under this
setting. Answers are generated using LLaMA-3 8B Instruct with
greedy decoding, and the resulting responses are evaluated by an
external LLM judge, Gemini-2.5-Flash-Lite.
Overall Pattern: Question-Guided Reasoning vs. External
Over-Reliance.Across layers, a clear global contrast emerges:Cor-
rect predictions emphasize question understanding, whereas
incorrect predictions over-rely on externally aligned answer
content.
As shown in Figure 3, correct samples consistently allocate more
routing mass to 𝑄→𝑄 . This indicates that the model places greater
emphasis on understanding the question, first building a stable in-
ternal representation of the reasoning objective and then continuing
to use it as a constraint while forming the answer.

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Incorrect samples, in contrast, show relatively weaker question
consolidation and stronger routing from 𝑄→Ans_EXT . This
suggests a shortcut strategy: instead of constructing the answer
through question-guided reasoning, the model leans heavily on
answer fragments that are directly supported by retrieved content.
As a result, the model tends to use context that is superficially
aligned with the question, yielding locally plausible reasoning steps,
while remaining misaligned with the deeper semantic constraints
required to correctly solve the problem.
Thus, the key difference is not simply how much external in-
formation is used, butwhether answer formation is anchored in a
well-formed question representation.
Layer-wise Distribution of Attribution Weight.While the
previous section examined the overall routing distribution, we now
provide a more detailed layer-wise analysis to better understand
how routing patterns evolve across the network, as shown in Fig-
ure 4 and Figure 5.
Low Layers (0-7): Establishing the Question Anchor.The diver-
gence begins in the lowest layers. Correct predictions devote sub-
stantially more routing to 𝑄→𝑄 , indicating stronger internal
consolidation of the question before heavy involvement of answer-
related representations. This early investment builds a stable se-
mantic anchor that guides downstream reasoning.
Wrong predictions, however, show weaker 𝑄→𝑄 routing and
relatively stronger early routing from 𝑄toward Ans_EXT . The
model starts linking the question directly to externally aligned
answer content before fully stabilizing the question representa-
tion itself. As a result, early processing is driven more by surface
alignment with retrieved information than by a structured internal
reasoning objective.
This early imbalance sets the stage for later errors: when question
understanding is shallow, answer formation becomes vulnerable to
external bias.
Higher Layers (8–31): Answer-Focused Refinement.In the higher
layers, routing patterns become broadly answer-centric forboth
correct and wrong predictions. The dominant activity shifts toward
refining internal answer representations, consistent with a late
stage where the model mainly stabilizes the emerging answer and
expresses it in fluent natural language. In this regime, the model pri-
marily elaborates and stabilizes the evolving answer state. Correct
predictions, however, still retain slightly stronger 𝑄→𝑄 routing
than wrong ones, indicating a small but persistent influence of the
question even at late stages.
Summary: A Depth-Wise Shift from Question Guidance to
External Drift.Correct and incorrect reasoning exhibit markedly
different depth-wise trajectories. Correct predictions begin with
strong comprehension of the question representation, followed
by answer formation that remains consistently constrained by
it—a pattern we termquestion-constrained evidence grounding
(QCEG). In contrast, incorrect predictions show weak early ques-
tion grounding, transition prematurely toward externally aligned
answer content, and ultimately refine answers under the domi-
nance of externally driven. We refer to this failure mode assurface-
aligned evidence grounding (SAEG).Thus, mixed-context errors arise from a progressive routing
shift: computation moves away from question-guided reasoning
and toward externally driven answer construction. Once this shift
occurs in early layers, later processing may tends to amplify the
misalignment.
5 Graph-Structural Detection of Unfaithful
Predictions
Our structural analysis reveals that correct and unfaithful predic-
tions differ in the internal organization of their reasoning circuits.
Correct answers tend to be supported by deeper, more integrated,
and more coherent attribution graphs, whereas unfaithful answers
arise from fragmented and weakly coordinated structures.
We leverage this insight by framing faithfulness detection as
agraph classificationproblem. If structural organization system-
atically differs between correct and unfaithful reasoning, then a
model should be able to predict answer faithfulness directly from
the structure of its attribution graph.
Concretely, given an attribution graph 𝐺=(𝑉,𝐸) constructed
from a model prediction, our goal is to estimate 𝑝(𝑦= 1|𝐺), the
probability that the prediction is faithful ( 𝑦=1) rather than unfaith-
ful (𝑦=0), using only internal structural information. To achieve
this, we employ a graph neural architecture that captures both local
evidence propagation patterns and global circuit organization.
5.1 Graph Features
Each attribution graph 𝐺contains nodes 𝑣∈𝑉 representing ques-
tion tokens, retrieved context tokens, intermediate activations, and
generated tokens, and directed edges (𝑢,𝑣)∈𝐸 representing causal
attribution links. Each edge carries a scalar weight 𝑤𝑢𝑣indicating
attribution strength.
Node features.Each node 𝑣is associated with a feature vector
x𝑣∈R𝑑𝑥, which concatenates a one-hot encoding of node type (e.g.,
question, context, answer, intermediate) with normalized structural
signals such as in-degree, out-degree, total degree, and PageRank
score. These features describe both the functional role and the local
structural importance of each node.
Edge features.Each directed edge (𝑢,𝑣) is assigned a one-dimensional
featuree𝑢𝑣=tanh(𝑤 𝑢𝑣), a bounded transformation of the attribu-
tion weight. This scalar encodes how strongly information flows
from𝑢to𝑣within the reasoning circuit.
Graph-level topology signatures.In addition to node- and edge-
level information, we compute a vector of global structural statistics
g(𝐺)∈R𝑑𝑔, including measures such as longest-path depth, aver-
age degree, triad ratios, graph density, and maximum PageRank,
as analyzed in Section 4.1.1 . These metrics summarize the overall
organizational patterns of the circuit.
5.2 Graph Transformer Encoder
To capture both local evidence propagation and long-range struc-
tural interactions, we use a graph transformer encoder that alter-
nates between message passing and attention. Leth(0)
𝑣=MLP in(x𝑣)
be the initial node embedding. Each layer ℓ=1,...,𝐿 then performs
two stages.

Why Retrieval-Augmented Generation Fails: A Graph Perspective Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
Local structural propagation.We first update node states using a
message passing operator that aggregates information from imme-
diate neighbors:
˜h(ℓ)
𝑣=h(ℓ−1)
𝑣+MPNN
𝑣,{h(ℓ−1)
𝑢,e𝑢𝑣}𝑢∈N(𝑣)
,
whereN(𝑣) denotes the neighbors of 𝑣. This step models how
evidence locally accumulates along attribution edges.
Global attention interaction.We then apply a graph attention
mechanism to allow non-local structural interactions:
h(ℓ)
𝑣=˜h(ℓ)
𝑣+∑︁
𝑢∈𝑉𝛼(ℓ)
𝑣𝑢W(ℓ)˜h(ℓ)
𝑢,
where attention weights 𝛼(ℓ)
𝑣𝑢are computed from node features and
edge features. This stage enables the model to capture long-range
coordination across different parts of the reasoning circuit. After
𝐿layers, each node has a final representationh(𝐿)
𝑣encoding both
local and global structural context.
5.3 Graph-Level Readout
We convert node embeddings into a graph representation using
simple multi-statistic pooling:
hpool(𝐺)=
mean𝑣∈𝑉h(𝐿)
𝑣∥sum𝑣∈𝑉h(𝐿)
𝑣∥max𝑣∈𝑉h(𝐿)
𝑣
.
Here∥denotes concatenation. Mean pooling captures the overall
structural tendency, sum pooling captures the total evidence mass,
and max pooling captures the strongest pathway signal.
We additionally embed a small set of global topology statistics
g(𝐺)∈R𝑑𝑔:
h𝑔(𝐺)=𝜙𝑔(g(𝐺)),
where𝜙𝑔is a small MLP. The final graph representation is simply
z(𝐺)=
hpool(𝐺)∥h𝑔(𝐺)
.
A final classifier produces𝑝(𝑦=1|𝐺)fromz(𝐺).
5.4 Unfaithful Prediction
A final classifier produces the probability that the prediction is
correct:
𝑝(𝑦=1|𝐺)=softmax MLP out(z(𝐺))
1.
We predict an output as unfaithful when𝑝(𝑦=1|𝐺)<0.5.
This detector checks whether the model’s internal reasoning
forms a coherent structure or a fragmented one. Local message
passing tracks how evidence flows along attribution paths, global
attention connects distant but related parts of the circuit, and graph-
level statistics summarize the overall depth and organization. To-
gether, these signals reveal reasoning failures that are not visible
from the answer text or retrieved documents alone.
5.5 Results
We now evaluate whether internal reasoning structure can be used
to reliably predict when a RAG model’s answer is correct.
Figure 6: Performance comparison across QA benchmarks.
5.5.1 Experimental Setup.We evaluate detection on attribution
graphs derived from HotpotQA, 2WikiMultihopQA, and MuSiQue.
For each dataset, we follow the fixed graph-based split defined dur-
ing graph construction: up to 500 wrong and 500 correct examples
are collected in deterministic filename order, 250 per class are sam-
pled for training/validation, and the remaining 500 examples form
a balanced test set. All methods are evaluated on the same fixed
indices to ensure direct comparability.
As a non-structural baseline, we use a logit-based self-judging
signal computed from the model’s own output distribution, without
access to gold answers. After generating an answer, the same model
is prompted to judge whether its prediction is correct given the
question, retrieved context, and its own reasoning trace, and is re-
stricted to a binary response (“Yes” or “No”). We compute log𝑝(Yes)
andlog𝑝(No) for the two continuations and predict correctness
whenlog𝑝(Yes)>log𝑝(No).
5.5.2 Graph Detector Training.The graph detector uses a graph
transformer encoder with 𝐿=2layer and hidden size 128, trained
with AdamW (learning rate10−4, batch size 32) and dropout 0.1.
For each dataset, we construct a balanced set of 1,000 graphs (500
incorrect and 500 correct) and use a fixed split with 250 graphs per
class for training/validation and 250 per class for testing.
5.5.3 Detection Performance.Figure 6 shows that the graph-structural
detector consistently outperforms the logit-based self-judging base-
line on all three QA benchmarks. Averaged across datasets, our
method improves accuracy by 11.53%.
These results demonstrate that modeling internal reasoning
structure provides a substantially more reliable signal of answer
correctness than relying on the model’s own output confidence.
While logit-based self-evaluation reflects surface-level uncertainty,
it cannot determine whether evidence has been integrated through
a coherent multi-step reasoning circuit. In contrast, the graph-
structural detector directly measures the organization of evidence
flow, enabling more accurate detection of unfaithful predictions
under challenging retrieval conditions.
6Attention Intervention for RAG Improvement
Section 4.2.2 shows that incorrect predictions under mixed retrieval
are not random errors but arise from a routing pattern: the model
under-invests in early question consolidation, over-commits to
surface-matched context, and later drifts into self-reinforcing de-
coding that is weakly constrained by the question. This section
turns that diagnosis into an actionable control mechanism. Instead
of changing parameters or re-retrieving, we intervene directly on

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
the attention computation during decoding to encourage the rout-
ing behavior characteristic of correct predictions.
6.1 Intervention as Layer-Wise Routing Control
The analysis above reveals that RAG errors arise from a systematic
shift in how information is routed across layers. Correct predictions
maintain strong question grounding throughout the computation,
whereas wrong predictions exhibit two coupled failures: (i) insuf-
ficient early consolidation of the question representation, and (ii)
progressively increasing reliance on externally information.
We therefore design an intervention that directly reshapes rout-
ing preferences inside the attention mechanism. The goal is not
to retrain the model, but to gently bias information flow toward
the routing regime associated with correct reasoning. We partition
token positions into three semantically meaningful regions:
•𝑄: question tokens,
•𝐸𝑥: external retrieved context tokens,
•𝐼𝑛: internally generated tokens.
6.1.1 Control 1: Strengthening Early Question Understanding.Sec-
tion 4.2.2 shows that wrong predictions underutilize 𝑄→𝑄 rout-
ing in early layers. As a result, the question representation is not
sufficiently understood before interacting with retrieved context.
To counteract this, we amplify attention among question tokens
in lower layers:
𝛼(ℓ)
𝑄→𝑄=𝛼 QQ forℓ∈L low, 𝛼 QQ>1.
This encourages deeper internal integration of question seman-
tics before heavy evidence mixing occurs.
6.1.2 Control 2: Suppressing Premature Context Reliance.We fur-
ther observe that incorrect predictions tend to route attention to-
ward external context too early, leading to brittle surface alignment
rather than deep reasoning.
We therefore down-weight attention whose *target* lies in the
external region during the same early stage:
𝛼(ℓ)
∗→𝐸𝑥=𝛼 ctx forℓ∈L low, 𝛼 ctx<1.
Here∗denotes any source region. This control reduces the in-
fluence of retrieved tokens before the question representation has
stabilized.
6.1.3 Control 3: Maintaining Question-Guided Decoding.In later
layers, answer tokens are iteratively refined. For correct predic-
tions, routing from the question to internal answer states remains
active, ensuring that decoding stays constrained by the original
task. Incorrect predictions, in contrast, show weakening 𝑄→𝐼𝑛
routing and increasing self-reinforcement among answer tokens.
To maintain question guidance during decoding, we strengthen
attention from question tokens to internally generated tokens in
higher layers:
𝛼(ℓ)
𝑄→𝐼𝑛=𝛼 QIn forℓ∈L high, 𝛼 QIn>1.
6.1.4 How is the Control Applied in the Model?The proposed in-
tervention does not alter any model parameters and requires no
retraining. Instead, it operates directly on the model’s internal at-
tention computation during inference. We implement this control
using forward hooks inserted into selected transformer layers.
Figure 7: Performance comparison on Mix-MusiQue.
A hook is a lightweight function inserted into the model’s for-
ward pass that intercepts and modifies intermediate activations
without changing model parameters. In our case, the hook inter-
cepts the attention pattern immediately before it is used to aggre-
gate value vectors. At chosen layers, we rescale specific groups
of attention weights based on the semantic regions of the source
and target tokens (question, retrieved context, or generated answer
tokens) as well as the layer index.
Because the modification occurs at the level of attention weights,
it changes the relative influence that different token groups exert
on one another, thereby steering information routing inside the
network. Importantly, all original model weights remain frozen:
the intervention introduces only a small amount of element-wise
scaling applied on-the-fly during the forward pass. As a result, the
computational overhead is negligible, and the model’s base behavior
can be fully restored by simply removing the hooks.
6.2 Results
We now evaluate whether the proposed layer-wise routing control
improves answer faithfulness in the mixed-context setting. We
focus on overall answer accuracy, which reflects whether the model
ultimately produces the correct answer under the presence of both
supporting and distracting evidence.
6.2.1 Setup.Our experiments focuses on a mixed-context set-
ting that we construct based on MuSiQue, which we refer to as
Mix-MuSiQue. In this dataset, each question is paired with a re-
trieved context that intentionally contains both supporting and
non-supporting passages. The final evaluation set consists of 667
questions under this mixed-evidence condition. Answers are gener-
ated using LLaMA-3 8B Instruct. We then evaluate the responses
using an external LLM judge, Gemini-2.5-Flash-Lite.
We compare two conditions: Before Control, the standard model
without intervention, and After Control, where region-aware atten-
tion reweighting is enabled in lower and higher layers. All other
decoding settings are kept identical to ensure a fair comparison.
For our method, we set 𝛼QQ=1.5, which promotes stronger un-
derstanding of question semantics before substantial interaction
with retrieved evidence. We set 𝛼ctx=0.5, mitigating premature
over-reliance on external information in early layers. Finally, we set
𝛼QIn=1.5, which biases later computation toward refining answer
representations under continued guidance from the question.
6.2.2 Intervention Results in the Mixed-Context Setting.Figure 7 re-
ports performance on the Mix-Musique setting, where the retrieved
context contains both supporting and non-supporting information.
This mixed-evidence scenario is particularly challenging because

Why Retrieval-Augmented Generation Fails: A Graph Perspective Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
the model must distinguish useful evidence from distractors while
maintaining alignment with the question.
Without intervention, the baseline model attains 56.5% accuracy.
Applying our attention control raises performance to 61.6%, repre-
senting a 9% improvement. This consistent gain demonstrates that
the proposed intervention effectively reshapes the model’s internal
information routing under mixed-context conditions.
The improvement suggests that errors in this setting are closely
tied to how the model allocates attention across question and con-
text tokens. By strengthening question-grounded routing and reduc-
ing early over-reliance on retrieved context, the intervention helps
the model better integrate relevant evidence while suppressing
distractors, leading to more faithful answer generation. In addition,
we present a case study in Appendix A.2 due to space limitations.
7 Conclusion
We present a graph-based perspective on why retrieval-augmented
generation can fail even when relevant evidence is available. Attri-
bution graphs reveal clear structural differences between success
and failure: correct predictions show deeper, more distributed, and
question-constrained evidence integration, while errors exhibit
shallow, context-dominated routing. These insights enable graph-
based error detection and targeted inference-time interventions
that reshape internal information flow.References
[1]Emmanuel Ameisen, Jack Lindsey, Adam Pearce, Wes Gurnee, Nicholas L. Turner,
Brian Chen, Craig Citro, David Abrahams, Shan Carter, Basil Hosmer, Jonathan
Marcus, Michael Sklar, Adly Templeton, Trenton Bricken, Callum McDougall,
Hoagy Cunningham, Thomas Henighan, Adam Jermyn, Andy Jones, Andrew
Persic, Zhenyi Qi, T. Ben Thompson, Sam Zimmerman, Kelley Rivoire, Thomas
Conerly, Chris Olah, and Joshua Batson. 2025. Circuit Tracing: Revealing Com-
putational Graphs in Language Models.Transformer Circuits Thread(2025).
https://transformer-circuits.pub/2025/attribution-graphs/methods.html
[2]Nicholas Ampazis. 2024. Improving RAG quality for large language models with
topic-enhanced reranking. InIFIP international conference on artificial intelligence
applications and innovations. Springer, 74–87.
[3]Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2023.
Self-rag: Self-reflective retrieval augmented generation. InNeurIPS 2023 workshop
on instruction tuning and instruction following.
[4] Orlando Ayala and Patrice Bechard. 2024. Reducing hallucination in structured
outputs via Retrieval-Augmented Generation. InProceedings of the 2024 Confer-
ence of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 6: Industry Track). 228–238.
[5]Chi-Min Chan, Chunpu Xu, Ruibin Yuan, Hongyin Luo, Wei Xue, Yike Guo,
and Jie Fu. 2024. Rq-rag: Learning to refine queries for retrieval augmented
generation.arXiv preprint arXiv:2404.00610(2024).
[6]Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024. Benchmarking large
language models in retrieval-augmented generation. InProceedings of the AAAI
Conference on Artificial Intelligence, Vol. 38. 17754–17762.
[7]Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D Manning.
2019. What does bert look at? an analysis of bert’s attention.arXiv preprint
arXiv:1906.04341(2019).
[8]Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen
Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen,
et al.2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multi-
modality, long context, and next generation agentic capabilities.arXiv preprint
arXiv:2507.06261(2025).
[9]Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey.
2023. Sparse autoencoders find highly interpretable features in language models.
arXiv preprint arXiv:2309.08600(2023).
[10] Xinnan Dai, Kai Guo, Chung-Hsiang Lo, Shenglai Zeng, Jiayuan Ding, Dongsheng
Luo, Subhabrata Mukherjee, and Jiliang Tang. 2025. GraphGhost: Tracing Struc-
tures Behind Large Language Models.arXiv preprint arXiv:2510.08613(2025).
[11] Jialin Dong, Bahare Fatemi, Bryan Perozzi, Lin F Yang, and Anton Tsitsulin.
2024. Don’t forget to connect! improving rag with graph-based reranking.arXiv
preprint arXiv:2405.18414(2024).
[12] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan,
et al. 2024. The llama 3 herd of models.arXiv e-prints(2024), arXiv–2407.
[13] Jacob Dunefsky, Philippe Chlenski, and Neel Nanda. 2024. Transcoders find
interpretable llm feature circuits.Advances in Neural Information Processing
Systems37 (2024), 24375–24410.
[14] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva
Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan
Larson. 2024. From local to global: A graph rag approach to query-focused
summarization.arXiv preprint arXiv:2404.16130(2024).
[15] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph,
Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova Das-
Sarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy
Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown,
Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2021. A Mathemat-
ical Framework for Transformer Circuits.Transformer Circuits Thread(2021).
https://transformer-circuits.pub/2021/framework/index.html.
[16] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin,
Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards
retrieval-augmented large language models. InProceedings of the 30th ACM
SIGKDD conference on knowledge discovery and data mining. 6491–6501.
[17] Javier Ferrando, Oscar Obeso, Senthooran Rajamanoharan, and Neel Nanda. 2024.
Do i know this entity? knowledge awareness and hallucinations in language
models.arXiv preprint arXiv:2411.14257(2024).
[18] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin
Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997
2, 1 (2023).
[19] Kai Guo, Harry Shomer, Shenglai Zeng, Haoyu Han, Yu Wang, and Jiliang Tang.
2025. Empowering graphrag with knowledge filtering and integration.arXiv
preprint arXiv:2503.13804(2025).
[20] Shailja Gupta, Rajesh Ranjan, and Surya Narayan Singh. 2024. A comprehensive
survey of retrieval-augmented generation (rag): Evolution, current landscape
and future directions.arXiv preprint arXiv:2410.12837(2024).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
[21] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Ma-
hantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al .
2024. Retrieval-augmented generation with graphs (graphrag).arXiv preprint
arXiv:2501.00309(2024).
[22] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020.
Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reason-
ing Steps. InProceedings of the 28th International Conference on Computational
Linguistics. 6609–6625.
[23] Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason Zhang, Xiaoyong Wei, and
Li Qing. 2025. Removal of hallucination on hallucination: Debate-augmented RAG.
InProceedings of the 63rd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 15839–15853.
[24] Dahyun Lee, Yongrae Jo, Haeju Park, and Moontae Lee. 2025. Shifting from
ranking to set selection for retrieval augmented generation. InProceedings of the
63rd Annual Meeting of the Association for Computational Linguistics (Volume 1:
Long Papers). 17606–17619.
[25] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin,
Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel,
et al.2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems33 (2020), 9459–9474.
[26] Zhining Liu, Rana Ali Amjad, Ravinarayana Adkathimar, Tianxin Wei, and Hang-
hang Tong. 2025. SelfElicit: Your language model secretly knows where is the
relevant evidence.arXiv preprint arXiv:2502.08767(2025).
[27] Samuel Marks, Can Rager, Eric J Michaud, Yonatan Belinkov, David Bau, and
Aaron Mueller. 2024. Sparse feature circuits: Discovering and editing interpretable
causal graphs in language models.arXiv preprint arXiv:2403.19647(2024).
[28] Jinming Nian, Zhiyuan Peng, Qifan Wang, and Yi Fang. 2025. W-rag: Weakly
supervised dense retrieval in rag for open-domain question answering. InPro-
ceedings of the 2025 International ACM SIGIR conference on innovative concepts
and theories in information retrieval (ICTIR). 136–146.
[29] Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong,
Juntong Song, and Tong Zhang. 2024. Ragtruth: A hallucination corpus for
developing trustworthy retrieval-augmented language models. InProceedings of
the 62nd Annual Meeting of the Association for Computational Linguistics (Volume
1: Long Papers). 10862–10878.
[30] Gonçalo Paulo, Alex Mallen, Caden Juang, and Nora Belrose. 2024. Automati-
cally interpreting millions of features in large language models.arXiv preprint
arXiv:2410.13928(2024).
[31] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan
Zhang, and Siliang Tang. 2025. Graph retrieval-augmented generation: A survey.
ACM Transactions on Information Systems44, 2 (2025), 1–52.
[32] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu
Chen. 2023. Enhancing retrieval-augmented large language models with iterative
retrieval-generation synergy.arXiv preprint arXiv:2305.15294(2023).
[33] Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning
Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. 2025. Parametric retrieval augmented
generation. InProceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval. 1240–1250.
[34] Yixuan Tang and Yi Yang. 2024. Multihop-rag: Benchmarking retrieval-
augmented generation for multi-hop queries.arXiv preprint arXiv:2401.15391
(2024).
[35] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2022. MuSiQue: Multihop Questions via Single-hop Question Composition.
Transactions of the Association for Computational Linguistics10 (2022), 539–554.
[36] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.
2023. Interleaving retrieval with chain-of-thought reasoning for knowledge-
intensive multi-step questions. InProceedings of the 61st annual meeting of the
association for computational linguistics (volume 1: long papers). 10014–10037.
[37] Jesse Vig and Yonatan Belinkov. 2019. Analyzing the structure of attention in a
transformer language model.arXiv preprint arXiv:1906.04284(2019).
[38] Zhiruo Wang, Jun Araki, Zhengbao Jiang, Md Rizwan Parvez, and Graham Neubig.
2023. Learning to filter context for retrieval-augmented generation.arXiv preprint
arXiv:2311.08377(2023).
[39] Wenlong Wu, Haofen Wang, Bohan Li, Peixuan Huang, Xinzhe Zhao, and Lei
Liang. 2025. Multirag: a knowledge-guided framework for mitigating hallucina-
tion in multi-source retrieval augmented generation. In2025 IEEE 41st Interna-
tional Conference on Data Engineering (ICDE). IEEE, 3070–3083.
[40] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan
Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for
diverse, explainable multi-hop question answering. InProceedings of the 2018
conference on empirical methods in natural language processing. 2369–2380.
[41] Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Moham-
mad Shoeybi, and Bryan Catanzaro. 2024. Rankrag: Unifying context ranking
with retrieval-augmented generation in llms.Advances in Neural Information
Processing Systems37 (2024), 121156–121184.
[42] Shenglai Zeng, Jiankun Zhang, Bingheng Li, Yuping Lin, Tianqi Zheng, Dante
Everaert, Hanqing Lu, Hui Liu, Yue Xing, Monica Xiao Cheng, et al .2025. To-
wards knowledge checking in retrieval-augmented generation: A representationperspective. InProceedings of the 2025 Conference of the Nations of the Ameri-
cas Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers). 2952–2969.
[43] Penghao Zhao, Hailin Zhang, Qinhan Yu, Zhengren Wang, Yunteng Geng,
Fangcheng Fu, Ling Yang, Wentao Zhang, Jie Jiang, and Bin Cui. 2026. Retrieval-
augmented generation for ai-generated content: A survey.Data Science and
Engineering(2026), 1–29.
[44] Zheng Zhao, Yeskendir Koishekenov, Xianjun Yang, Naila Murray, and Nicola
Cancedda. 2025. Verifying Chain-of-Thought Reasoning via Its Computational
Graph.arXiv preprint arXiv:2510.09312(2025).
[45] Xu Zheng, Ziqiao Weng, Yuanhuiyi Lyu, Lutao Jiang, Haiwei Xue, Bin Ren, Danda
Paudel, Nicu Sebe, Luc Van Gool, and Xuming Hu. 2025. Retrieval augmented
generation and understanding in vision: A survey and new outlook.arXiv preprint
arXiv:2503.18016(2025).
[46] Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Zheng Liu, Chaozhuo Li,
Zhicheng Dou, Tsung-Yi Ho, and Philip S Yu. 2024. Trustworthiness in retrieval-
augmented generation systems: A survey.arXiv preprint arXiv:2409.10102(2024).

Why Retrieval-Augmented Generation Fails: A Graph Perspective Conference acronym ’XX, June 03–05, 2018, Woodstock, NY
A Appendix
A.1 Related Work
Retrieval-Augmented Generation.Retrieval-Augmented Gener-
ation (RAG) has become a widely adopted paradigm for improving
the factuality and reasoning capabilities of large language models by
grounding generation in external knowledge sources [ 16,43]. Prior
work has explored a broad range of retrieval strategies, including
dense and hybrid retrievers [ 28], multi-hop retrieval [ 34], iterative
retrieval–generation loops [ 36], and query reformulation [ 5]. These
efforts have demonstrated that retrieval quality and evidence cover-
age play a critical role in downstream performance, and have led to
significant gains on question answering and knowledge-intensive
benchmarks.
More recent studies have focused on improving RAG robustness
through better context selection, reranking, compression, or prompt
engineering [ 2,11]. While these methods reduce failures at the
system level, they primarily treat the language model as a black box
and do not address how retrieved evidence is internally processed
once injected into the context. As a result, they offer limited insight
into why errors persist even when relevant evidence is successfully
retrieved.
In the context of RAG, several works have proposed methods to
assess faithfulness, evidence usage, and citation correctness [ 26,42].
However, these approaches typically rely on representations from
a single layer and provide only a largely static view of the model’s
internal state, lacking a comprehensive and dynamic perspective
for analyzing how evidence is integrated during computation.
Interpretability and Circuit Analysis of LLMsParallel to
advances in RAG, a growing body of work has investigated the
internal mechanisms of transformer models using interpretability
techniques such as attention analysis [ 7,37], Sparse Autoencoders
(SAEs) [ 9], transcoders [ 13], and circuit tracing [ 15]. These stud-
ies suggest that specific reasoning behaviors can often be traced
to localized subnetworks or circuits spanning multiple layers and
attention heads. A common theme across these methods is the de-
composition of dense neural representations into more interpretable
feature spaces [ 17,30], which in turn enables the construction of
attribution graphs that model the causal flow of information within
the network [27].
While circuit-level analyses have provided valuable insights into
question answering [ 10,44] by verifying chain-of-thought reason-
ing through computational graphs, most prior work has been lim-
ited to standalone language models without retrieval. As a result,
the interaction between external evidence and internal circuits in
RAG settings remains insufficiently understood.
Positioning of Our Work.Our work bridges these lines of
research by providing a mechanistic, circuit-level analysis of RAG
systems. Unlike prior RAG studies that emphasize retrieval quality
or prompt design, we focus on how retrieved evidence propagates
through internal model circuits. By introducing attribution graphs,
we move beyond scalar attribution scores and enable structural
analysis of information flow between faithful and hallucinated gen-
erations. This perspective allows us to identify integration failures
inside the model as a key source of RAG failures, complementing
and extending existing system-level analyses.A.2 Case Study
Figure 11 provides representative examples illustrating how the
intervention improves reasoning under mixed-context conditions.
In both examples, the baseline model fails to complete the full
reasoning chain: in one case it stops at a salient intermediate entity
without performing the final role mapping, and in the other it fails
to bridge a geographic clue to the relevant historical event. After
attention control is applied, the model follows a more complete
reasoning path, successfully connects intermediate evidence, and
produces the correct final answer.
These qualitative examples show that the intervention does not
merely alter surface-level outputs. Instead, it changes how the
model integrates evidence across reasoning steps, improving its
ability to link intermediate clues to the final answer.
Figure 8: Radar comparison of attribution-graph structural
metrics between correct and incorrect predictions across
three QA datasets (2Wiki, HotpotQA, and MuSiQue).
Received 20 February 2007; revised 12 March 2009; accepted 5 June 2009
Figure 9: Layer-wise attribution mass for correct and wrong
predictions (left) and their difference on HotpotQA (right).
Figure 10: Layer-wise attribution mass for correct and wrong
predictions (left) and their difference on 2Wiki (right).

Conference acronym ’XX, June 03–05, 2018, Woodstock, NY Trovato et al.
Successful Intervention Case 1
Question Who does the singer of May We All with Florida Georgia Line play in Country Strong?
Gold Answer James Canter
Baseline Model Out-
put (Before Control)The singer of “May We All” with Florida Georgia Line is Tim McGraw. He appears inCountry
Strong.Final answer: Tim McGraw.
Controlled Model
Output (After Con-
trol)The singer of “May We All” with Florida Georgia Line is Tim McGraw. InCountry Strong, Tim
McGraw plays the role of James Canter.Final answer: James Canter.
What Changed The baseline system stopped at the intermediate entity (the actor) and failed to perform the final
role-mapping step. After intervention, the model completed the reasoning chain and produced
the correct character name instead of the actor’s name.
Successful Intervention Case 2
Question When did allied troops land in the region sharing a border with Sahel?
Gold Answer 8 November 1942
Baseline Model Out-
put (Before Control)The context mentions North Africa and Operation Torch, but it does not explicitly state the
landing date for allied troops in the region bordering Sahel.Final answer: Not mentioned.
Controlled Model
Output (After Con-
trol)The region sharing a border with Sahel refers to North Africa. Operation Torch was the Allied
invasion of French North Africa, which began on 8 November 1942.Final answer: 8 November
1942.
What Changed The baseline system failed to link the geographic clue (Sahel) to North Africa and therefore
missed the historical event. After intervention, the model successfully bridged the geographic
reference to Operation Torch and retrieved the correct date.
Figure 11: Examples where routing control improves multi-hop reasoning. In both cases, the baseline model either stops at an
intermediate entity or fails to bridge a geographic clue to a historical event. After control, the model follows a more complete
reasoning path and produces the correct answer.