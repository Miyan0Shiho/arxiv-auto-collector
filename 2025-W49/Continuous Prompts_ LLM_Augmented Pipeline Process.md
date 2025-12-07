# Continuous Prompts: LLM-Augmented Pipeline Processing over Unstructured Streams

**Authors**: Shu Chen, Deepti Raghavan, Uğur Çetintemel

**Published**: 2025-12-03 02:41:45

**PDF URL**: [https://arxiv.org/pdf/2512.03389v1](https://arxiv.org/pdf/2512.03389v1)

## Abstract
Monitoring unstructured streams increasingly requires persistent, semantics-aware computation, yet today's LLM frameworks remain stateless and one-shot, limiting their usefulness for long-running analytics. We introduce Continuous Prompts (CPs), the first framework that brings LLM reasoning into continuous stream processing. CPs extend RAG to streaming settings, define continuous semantic operators, and provide multiple implementations, primarily focusing on LLM-based approaches but also reporting one embedding-based variants. Furthermore, we study two LLM-centric optimizations, tuple batching and operator fusion, to significantly improve efficiency while managing accuracy loss.
  Because these optimizations inherently trade accuracy for speed, we present a dynamic optimization framework that uses lightweight shadow executions and cost-aware multi-objective Bayesian optimization (MOBO) to learn throughput-accuracy frontiers and adapt plans under probing budgets.
  We implement CPs in the VectraFlow stream processing system. Using operator-level microbenchmarks and streaming pipelines on real datasets, we show that VectraFlow can adapt to workload dynamics, navigate accuracy-efficiency trade-offs, and sustain persistent semantic queries over evolving unstructured streams.

## Full Text


<!-- PDF content starts -->

Continuous Prompts: LLM-Augmented Pipeline Processing
over Unstructured Streams
Shu Chen
Brown University
Providence, USA
shu_chen@brown.eduDeepti Raghavan
Brown University
Providence, USA
deeptir@brown.eduUğur Çetintemel
Brown University
Providence, USA
ugur_cetintemel@brown.edu
ABSTRACT
Monitoring unstructured streams increasingly requires persistent,
semantics-aware computation, yet today’s LLM frameworks remain
stateless and one-shot, limiting their usefulness for long-running an-
alytics. We introduceContinuous Prompts(CPs), the first framework
that brings LLM reasoning into continuous stream processing. CPs
extend RAG to streaming settings, define continuous semantic op-
erators, and provide multiple implementations, primarily focusing
on LLM-based approaches but also reporting one embedding-based
variants. Furthermore, we study two LLM-centric optimizations, tu-
ple batching and operator fusion, to significantly improve efficiency
while managing accuracy loss.
Because these optimizations inherently trade accuracy for speed,
we present a dynamic optimization framework that uses lightweight
shadow executionsand cost-awaremulti-objective Bayesian optimiza-
tion(MOBO) to learn throughput–accuracy frontiers and adapt
plans under probing budgets.
We implement CPs in the VectraFlow stream processing system.
Using operator-level microbenchmarks and streaming pipelines
on real datasets, we show that VectraFlow can adapt to workload
dynamics, navigate accuracy–efficiency trade-offs, and sustain per-
sistent semantic queries over evolving unstructured streams.
1 INTRODUCTION
Large language models (LLMs) have become the de-facto tool for
interpreting diverse unstructured data, from text and images to
tables and semi-structured documents. Yet most LLM pipelines
today are stateless and episodic: they evaluate isolated prompts
over static corpora, with limited support for operator state, per-
sistence, or long-running adaptivity. Many emerging applications
(e.g., clinical monitoring, financial surveillance, regulatory compli-
ance) have fundamentally different needs: they require continuous,
semantics-aware computation over evolving data streams, often
under resource budgets that force explicit performance–accuracy
trade-offs.
We argue that supporting these workloads requires the LLM
analog of data-stream processing systems. To this end, we intro-
duceContinuous Prompts(CPs), which extend one-shot prompt
execution into stateful, long-lived, and adaptive computations over
unstructured streams. Just as continuous queries lift relational op-
erators to unbounded data, CPs lift LLM semantics into continuous,
pipeline-level operators. Building CPs requires: (i) LLM-native se-
mantic operators with explicit state and composable semantics; (ii)
continuous operator variants — windowing, grouping, and retrieval
— that track evolving content rather than fixed windows; and (iii)
runtime mechanisms that balance efficiency and accuracy under
dynamic conditions.Several recent systems (e.g., Palimpzest [ 12], Lotus [ 17], Do-
cETL [ 19]) introduced semantic operators for LLM-based analytics,
but they execute in batch or one-shot settings. They lack support
for unbounded streams and provide no mechanisms for optimiz-
ing LLM execution as workloads shift or as accuracy–throughput
trade-offs evolve. Our prior work [ 14] addresses streaming seman-
tics by operating over embedding-based operators such as vector
filters and vector topk, enabling continuous processing but only at
the vector level. In contrast, the present work brings continuous
processing to the LLM semantic layer, introducing continuous se-
mantic operators, and adaptive optimization techniques tailored to
the challenges of persistent LLM-based pipeline execution, which is
our focus. Embedding-based implementations, explored extensively
in VectraFlow and related systems, appear primarily as comparison
points in the accuracy–throughput design space.
We implement CPs on VectraFlow [ 14], extending the system
with LLM-based implementations of continuous semantic operators.
In addition to leveraging established techniques such as embedding-
based retrieval and indexing, we comprehensively investigate two
new LLM-specific optimizations.Tuple batchinggroups multiple
input tuples into a single LLM call to amortize invocation overhead,
andoperator fusioncombines multiple semantic operators into a sin-
gle prompt to reduce model invocations and exploit cross-operator
structure. Both techniques introduce performance–accuracy trade-
offs, which our system explicitly measures, models, and manages
through dynamic planning.
At runtime, VectraFlow’sdynamic optimization frameworkuni-
fies operator-level statistics and sensitivity profiles into a global
execution planner that continuously balances performance and
accuracy. The planner integrates offline operator-level microbench-
marks with live telemetry to generate, evaluate, and reconfigure ex-
ecution plans on the fly. It explores fusible operator chains, adaptive
batch sizes, and alternative implementations (e.g., embedding-based
variants), ranking candidate plans using predictive cost–accuracy
models. In summary, this paper makes the following contributions:
•We introducecontinuous prompts (CPs)—the LLM counterpart of
continuous queries—for stateful, adaptive processing of unstruc-
tured data streams, and extend prior semantic operator models
to continuous settings with new constructs such as semantic
windows, dynamic semantic group-by, and continuous RAG
(Section 3).
•We study two LLM-centric execution optimizations,tuple batch-
ingandoperator fusion, and quantify their performance–accuracy
trade-offs through operator sensitivity profiles (Section 4).
•We design a dynamic optimization framework that integrates
operator-level insights with cost models and online telemetry,
enabling adaptive plan selection at run-time (Section 5).arXiv:2512.03389v1  [cs.DB]  3 Dec 2025

•We propose a cost-aware multi-objective Bayesian optimization
(MOBO) algorithm that efficiently learns accuracy–throughput
trade-off curves and guides sample-efficient plan adjustments
under probing budgets (Section 6).
•We implement these ideas inVectraFlowand demonstrate that
it can effectively trade off performance and accuracy in re-
sponse to changing workload conditions on realistic streaming
pipelines (Section 7).
The rest of the paper is organized as follows. Section 2 introduces
the VectraFlow architecture, system model, and general semantic
operators. Section 3 presents our streaming-native semantic oper-
ators. Section 4 describes our execution-level optimizations and
analyzes their throughput–accuracy trade-offs. Section 5 describes
the dynamic planning framework, including its architecture and
cost and accuracy models. Section 6 details our cost-aware MOBO
algorithm for efficient frontier learning and adaptive plan selection.
Section 7 reports the evaluation setup and results, including two
end-to-end real-world streaming pipelines. Finally, Sections 8 and 9
review related work and conclude the paper.
2 VECTRAFLOW
2.1 System Model
The VectraFlow architecture [ 14] follows a traditional data-flow
system model. It formalizes continuous semantic stream processing
as a composition of stateful LLM-based operators over unstructured
data streams.
Unstructured Streams.The system ingests unbounded streams
of unstructured content such as clinical notes, financial news, or
social media posts. These inputs are heterogeneous in length, style,
and vocabulary, often lacking predefined schema or fixed attributes.
The goal is to continuously interpret and transform such data into
structured semantic representations.
Data Model.Incoming data are modeled as timestamped tuples
S={(𝑡𝑖,𝑥𝑖)|𝑖= 1,2,...} , where𝑡𝑖denotes the arrival time and 𝑥𝑖
is the tuple payload in an extended relational model, where tuples
can contain both conventional structured attributes and one or
more unstructured attributes (e.g., a document or message) that are
fed to semantic operators. Each tuple is processed incrementally
upon arrival, forming the atomic unit of streaming computation.
Operator Model.A query plan consists of a directed acyclic
graph ofsemantic operators O={𝑜 1,...,𝑜𝑘}. Each operator is
realized by an LLM or embedding-based function that implements
a continuous, stateful transformation such as semantic mapping,
filtering, segmentation (windowing, group-by), or ranking (top- 𝑘).
Operators maintain internal state 𝑠𝑗(𝑡)and update it as new tuples
arrive, enabling the system to adapt to semantic drift in the stream.
Execution Model.Tuples are processed either individually or
in mini-batches of size 𝑇to amortize LLM invocation overhead.
Adjacent operators may be fused into a single invocation to further
reduce redundant prefill costs. The system ensures order-preserving
propagation across operators and continuous state updates.
Cost Model.Each operator execution is associated with a cost
vector𝐶(𝑜𝑗)=⟨throughput(𝑜 𝑗),accuracy(𝑜 𝑗)⟩, capturing process-
ing efficiency in tuples per second and semantic fidelity. Thesemetrics form the foundation for dynamic plan optimization, guid-
ing runtime adjustments of batch size, fusion strategy, and operator
implementation.
2.2 General Semantic Operators
VectraFlow extends traditional relational data-flow models with a
suite of LLM-driven semantic operators that process unstructured
streams. Table 1 provides a complete overview of these operators.
The general semantic operators are designed to apply uniformly
in both batching and streaming execution modes. In particular,
operators such as semantic top-k and semantic aggregate include
incremental modes that maintain evolving state as new tuples arrive,
enabling streaming-native execution.
3 STREAMING-NATIVE SEMANTIC
OPERATORS
We introduce streaming-specific semantic operators that enable
adaptive, LLM-driven processing over unstructured data. Specif-
ically, we present three core operators and primarily LLM-based
implementations and analyze their behavior through microbench-
marks using the same experimental setup as inExperimental Setup
(§7.1). For completeness, we also evaluate embedding-based imple-
mentations when they provide a meaningful comparison.
3.1 Semantic Windows (𝜔 𝑠)
Stream processing systems usewindowsas logical boundaries to seg-
ment data for incremental computation. Conventional approaches
rely on time-based or count-based windows (e.g., “5-minute inter-
vals” or “100-event batches”), but these static strategies often fail to
align with evolving data semantics. To address this limitation, we
proposesemantic windows, which dynamically adjust their bound-
aries based on contextual meaning. By detecting signals such as
topic shifts, sentiment changes, or new entity mentions, semantic
windows can align execution with the intrinsic structure of the
stream rather than with arbitrary temporal or cardinal triggers.
Our design leverages a semantic windowing mechanism that
incrementally evaluates the semantic coherence of incoming tuples.
The LLM assigns acontinuity scoreto each new tuple, indicating
whether it belongs to the current window or should trigger the
start of a new one. Whenever this score falls below a threshold,
reflecting shifts such as topic drift, entity transitions, or narrative
changes, the operator can infer a semantic boundary. A typical
evaluation prompt might be:
“Given the tuples in the current window, should
the semantic window remain open? Analyze the
tuples for key shifts such as <topic drift> ,<new
entity reference> , or<narrative change> . If
one of these events occurs, return a continuity score
from 0 to 1, where 1 indicates high continuity and
0 signals that a new window should start.”
Implementation.We explore three strategies for realizing seman-
tic windowing: (1)Pairwise Semantic Window, which computes a
continuity score cont(𝑥𝑡,𝑥𝑡−1)between consecutive tuples; when
the score drops below a threshold 𝜏, a new window is opened,
2

Table 1: Overview of continuous semantic operators in VectraFlow.
Operator Description Example Use Case
General Semantic Operators (batching & streaming)
Semantic Filter(𝝈 𝒔) Applies an LLM-based predicate that returns a boolean de-
cision for each tuple; tuples evaluated astruepass the filter,
enabling semantic selection by topic, sentiment, or entitySelect tweets related toUkraineorCOVID-19from a
mixed news stream.
Semantic Map(𝝅 𝒔) Transforms unstructured text into structured records such
as entities, relations, or JSON key–value pairs.Extract company name and event from news head-
lines.
Semantic Aggregate(𝜸 𝒔) Computes summaries or trends over semantic windows;
supports both standard and incremental modes via init() ,
increment(), andfinalize().Continuously summarize market sentiment over the
last 100 tweets mentioning Apple ($AAPL).
Semantic Top-𝑘(𝝉 𝒔) Maintains the 𝑘most relevant tuples using a continuous
scoring function that evaluates impact, novelty, or topical
importance.Select the 10 most influential financial news items as
the market evolves.
Semantic Join(⊲⊳ 𝒔) Correlates tuples across streams based on semantic similarity
rather than strict key equality.Match analyst reports to relevant stock price move-
ments.
Streaming-Native Semantic Operators
Semantic Window(𝝎 𝒔) Dynamically adjusts window boundaries based on topic or
sentiment shifts to align computation with semantic changes.Detect when discourse shifts from “peace talks” to
“sanctions” in the Ukraine event stream.
Semantic Group-By(𝝁 𝒔) Groups tuples by meaning in streaming data, allowing cate-
gories to emerge, evolve, and dissolve over time.Track evolving news events and continuously update
group assignments as new information shifts the un-
derlying topics.
Continuous RAG(𝝆 𝒔) Maintains evolving prompts that continuously fetch relevant
context as query scope shifts, enabling real-time retrieval
from streams.Continuously monitor the financial news articles that
are relevant to my stock portfolio.
otherwise the current one is extended—this serves as our base-
line; (2)Summary-based Semantic Window, which maintains multi-
ple overlapping windows, each with an evolving summary 𝑆𝑖; for
each incoming tuple 𝑥𝑡, the system assigns it to the window with
the highest continuity score cont(𝑥𝑡,𝑆𝑖)above𝜏, updating the se-
lected summary incrementally via incremental aggregation; and (3)
Embedding-based Semantic Window, which maintains live clusters
with representatives and assigns 𝑥𝑡to the most similar cluster if
similarity≥𝜏, otherwise creating a new cluster. The threshold 𝜏is
tuned for the best observed performance.
Evaluation.We evaluate three semantic windowing strategies in
Figure 1 using the MiDe22 [ 21] dataset, which consists of 40 tem-
porally ordered events. The goal is to detect event shifts and assign
tweets from the same ground-truth event to a common window.
We constructoverlappingwindows with an expiry mechanism that
retires fading topics gracefully and prevents repeated splits un-
der gradual drift. We evaluate two dimensions:event groupingand
boundary detection. For grouping, we report F1 and ARI: F1 reflects
item-level precision and recall, while ARI captures pairwise agree-
ment between predicted and true event partitions. For boundary
detection, we use Boundary F1 and Purity: Boundary F1 measures
how accurately transition points are identified, and Purity reflects
the dominance of a single ground-truth event within each predicted
window. High Purity indicates strong internal consistency but may
also signalover-segmentation.
Takeaway.M3 (Embeddings) produces the most accurate bound-
aries, yielding the highest Boundary F1. M1 (Pairwise) achieves the
highest Purity by capturing fine-grained drift, though at the cost
F1 ARI Boundary F1 Purity0.00.20.40.60.81.0Score
(a) Metric Comparison
M1 Pairwise
M2 Summary
M3 Embedding
M1 M2 M30123Throughput (tuples/s)
(b) ThroughputFigure 1: Semantic window implementations on the MiDe22
dataset. Left: metric comparison (F1, ARI, Boundary F1, and
Purity), with ★indicating the best score. Right: throughput
in tuples/s.
of over-segmentation. M2 (Summary) delivers the strongest event-
grouping performance (F1 and ARI), maintaining long, coherent
windows but with less precise boundary placement.
3.2 Semantic Group-By (𝜇 𝑠)
We extend the conventional group-by abstraction to unstructured,
continuously arriving data. Unlike key-based grouping over explicit
attributes,semantic group-byclusters tuples by meaning, forming
groups that share a coherent topic or event. While prior systems
such as Lotus support offline semantic group-by over static corpora
(e.g., grouping ArXiv papers into 𝑘topics), our setting requires
clusters to evolve with the stream itself—topics drift, new entities
appear, and old ones fade.
3

F1 ARI Purity0.00.20.40.60.81.0Score
(a) Metric Comparison
M1 Basic LLM
M2 LLM+Refine
M3 Embedding
M1 M2 M30.000.250.500.751.001.25Throughput (tuples/s)
(b) ThroughputFigure 2: Semantic group-by implementations on the MiDe22
dataset. Left: metric comparison (F1, ARI, Boundary F1, and
Purity), with ★indicating the best score. Right: throughput
in tuples/s.
We therefore introduce adynamic semantic group-byopera-
tor that creates, refines, and retires categories on the fly, ensuring
groups remain aligned with the current semantic landscape. The op-
erator uses LLM-based comparisons or few-shot prompts to decide
membership incrementally, with embedding-based grouping as a
lighter alternative. For example, the operator tracks evolving news
events—merging early reports on “peace talks” and “sanctions” into
broader “Ukraine conflict” clusters as the story unfolds.
Implementation.Our dynamic semantic group-by design is in-
formed by the incremental clustering [ 4] that update cluster repre-
sentatives online as new data arrives. Building on these ideas, we
implement three approaches: (1)Basic LLM Group-By, where each
tuple is analyzed by the LLM and either assigned to an existing
group or used to create a new one, maintaining a dictionary of
group names and descriptions; this method is lightweight and fully
adaptive but may produce redundant or noisy groups; (2)LLM with
Refinement Group-By, which periodically revisits and restructures
the grouping—merging, splitting, or renaming categories to bet-
ter track topic drift, at the expense of additional LLM calls; and
(3)Embedding-based Group-By, which embeds tuples into a vector
space and performs incremental clustering, periodically sampling
a small number of items from each cluster and invoking an LLM
prompt to generate an updated, interpretable cluster name.
Evaluation.We compare the three approaches on a MiDe22 subset,
reporting F1, ARI, Purity, and throughput (tuples/s) in Figure 2.
The LLM with Refinement method periodically issues an additional
refinement prompt every 10 tuples.
Takeaway.Embedding-basedgrouping is fast and achieves high
item-level F1, but its over-segmentation produces fragmented events.
Basic LLMoffers moderate coherence and speed, whileLLM with Re-
finementdelivers the cleanest, most globally consistent clusters at a
modest throughput cost, making it the best option when preserving
event structure is the priority.
3.3 Continuous RAG
To support retrieval over evolving streams, we introducecontinuous
RAG, the continuous analog of traditional Retrieval-Augmented
Generation (RAG). While traditional RAG performs data retrieval
from a stored data store based on relevance to a one-time prompt,
continuous RAG retrieves relevant data from input streams based
on their relevance to a continuous prompt. More specifically, tradi-
tional RAG runs once: a fixed prompt retrieves relevant items from
Traditional RAG Data Stream 
Stored Prompts LLM Prompt Stream 
Stored Data LLM data chunk 1 data chunk 2 ... Check for relavant data chunks Check for matching tuples Continuous RAG 
prompt 1 prompt 2 ... Figure 3: Traditional RAG vs. Continuous RAG.
a stored corpus, whereas continuous RAG operates continuously,
using prompts to retrieve semantically relevant tuples from the live
stream. Figure 3 contrasts traditional RAG’s static retrieval model
with continuous RAG’s stream-oriented retrieval process.
Continuous RAG maintains a long-lived retrieval state that is
incrementally updated as new tuples arrive. In practice, VectraFlow
uses LLM-generated sub-prompts to track evolving semantic intent,
ensuring that retrieval remains aligned with current stream content
and user objectives. While we implement continuous RAG using a
cts_filterin our figures, acts_topkimplementation is equally
possible.
Implementation.To illustrate the behavior of our continuous
retrieval variants, consider a streaming analytics task where we
monitor a user’s stock portfolio. The system maintains a reference
table containing the user’s positions—e.g., NVDA ,AAPL ,MSFT —along
with metadata such as percentage allocation, descriptions, and ana-
lyst ratings. As news arrives in the stream, the retrieval operator
must continuously surface items relevant to these holdings. We
implement four variants of the continuous retrieval operator that
differ in how prompts and embeddings are maintained. (1) UP-
LLM maintains a single persistent retrieval prompt, which adapts
over time as the portfolio evolves. For example, the operator re-
peatedly issues a unified query such as “Find recent news that
impacts mystock_portfolio”, allowing the LLM to internally adjust
its scope as holdings change (e.g., adding or removing NVDA or
AAPL). (2) SP-LLM uses multiple LLM-generated sub-prompts to
track finer-grained intents. Under the portfolio example, SP-LLM
creates separate sub-prompts such as “Find news about NVDA”,
“Find news about AAPL”, and “Find news about MSFT”, ensuring
more precise and entity-aligned retrieval. (3) UP-Emb mirrors the
unified strategy of UP-LLM but performs retrieval in embedding
space: both the unified prompt and news items are embedded and
matched through vector similarity search. (4) SP-Emb combines sub-
prompting with embedding-based retrieval: each symbol-specific
prompt (e.g., NVDA, AAPL, MSFT) is encoded as a vector query,
enabling specialized, scalable matching across high-volume news
streams.
Evaluation.We conduct two experiments. (1) On MiDe22 [ 21],
we compare all fourcontinuous filtervariants across three topi-
cal categories (Ukraine, COVID-19, Refugees), reporting both F1
and throughput. (2) On FNSPID [ 7]—a financial news dataset with
aligned stock-price information for all S&P 500 companies. We
select 10 companies and vary the number of monitored companies
(i.e., number of predicates or sub-prompts) from 2 to 10, reporting
4

UP-LLM UP-Emb SP-LLM SP-Emb0.00.20.40.60.8F1 score
F1 Score
Throughput (tuples/s)
0246
Throughput (tuples/s)
Figure 4: Continuous RAG implementations on the MiDe22
dataset
.
2 4 6 8 10
# of Predicates0.60.70.80.9F1 Score
(a) F1 vs. Number of Predicates
UP-LLM
SP-LLM
UP-Emb
SP-Emb
2 4 6 8 10
# of Predicates12345Throughput (rec/s)
(b) Throughput vs. Number of Predicates
UP-LLM
SP-LLM
UP-Emb
SP-Emb
Figure 5: Continuous RAG under varying predicate counts
(2–10). Left: F1 versus # predicates. Right: throughput (rec/s)
versus # predicates.
F1 and throughput to test whether the trends remain consistent
under differing predicate counts.
Takeaway.Figure 4 reports performance across the four variants
of continuous filtering. As expected, SP–LLM achieves the highest
F1 because it leverages full LLM reasoning and decomposes the
predicate set into semantically focused subprompts. Conversely,
UP–Emb delivers the highest throughput, reflecting the efficiency
of unified prompting combined with lightweight embedding-based
retrieval. To further validate these assumptions, we vary the number
of predicates and examine how accuracy and throughput scale with
predicate complexity.
Figure 5 summarizes the trends as the number of predicates in-
creases. We note two observations: (i) SP–LLM consistently yields
the highest F1 across predicate counts, reflecting superior semantic
tracking as intent granularity increases. In contrast, UP–LLM ex-
periences mild accuracy degradation, consistent with instruction
interference when multiple predicates are fused into a single prompt.
(ii) UP–Emb and SP–Emb maintain high, stable throughput across
scales, demonstrating resilience to predicate growth, while LLM-
based variants show slower but more adaptive reasoning. Overall,
the results show that prompt factorization (SP) improves accuracy
under increasing predicate complexity, whereas embedding-based
retrieval preserves throughput scalability.
4 OPTIMIZATIONS
This section presents two key streaming-specific operator-level op-
timizations that improve the throughput of LLM-powered stream
processing:tuple batchingandoperator fusion. Although both meth-
ods can substantially increase throughput, they entail intrinsic
10 20 30
Tuple Batch Size (T)4.04.24.44.64.85.0Throughput (tuples/s)
Amazon Food Reviews
Throughput (tuples/s)
Accuracy0.600.650.700.750.800.850.900.95
Accuracy
(a) Amazon Food Reviews
10 20 30
Tuple Batch Size (T)4.44.64.85.05.2Throughput (tuples/s)
T weets
Throughput (tuples/s)
Accuracy0.600.650.700.750.80
Accuracy
 (b) Tweets
Figure 6: Tuple batching sensitivities for single operators on
short (Twitter) and longer (Amazon Fine Food Reviews) texts.
trade-offs: gains in efficiency commonly come at the cost of re-
duced accuracy or diminished semantic fidelity.
4.1 Tuple Batching
Tuple batching processes multiple input tuples within a single LLM
call, amortizing invocation cost and token usage across several data
items and thereby reducing the number of model calls. However,
batching introduces a trade-off: as more tuples are aggregated into
a single prompt, the prompt grows longer and more complex, which
can lead to quality loss if the model struggles to handle multiple
inputs simultaneously. While recent work onbatch prompting[ 5]
has explored joint processing of multiple inputs to improve LLM
inference efficiency, these techniques have not been studied in
the context ofsemantic relational pipelines. Our work examines
tuple batching as an operator-level optimization inside continuous,
stateful LLM pipelines, studying how batching interacts with se-
mantic operators, affects downstream accuracy, and contributes to
end-to-end throughput in streaming settings.
Implementation.We implement tuple batching by explicitly re-
structuring prompts to maximize their shared prefix and minimize
redundant tokens across tuples. Given a batch of 𝑇input tuples,
each operator starts from a logical per-tuple prompt template con-
sisting of: (i) a system prompt describing the role of the model,
(ii) an instruction prompt describing the task, and (iii) an output
schema specification. Instead of issuing 𝑇independent LLM calls,
we construct a single batched prompt as follows:
(1)Shared prefix construction: We move all shared content, includ-
ing the system prompt, task description, and schema, to the
beginning of the prompt, forming a common prefix that can be
reused across the entire batch.
(2)Tuple enumeration: We append the input stream tuples as num-
bered items, each labeled with a stable tuple identifier.
(3)Output specification: We ask the model to return a JSON list
whose𝑗-th entry corresponds to the input tuple 𝑗, so that out-
puts can be deterministically mapped back to the original tuples.
Evaluation.We evaluate how tuple batching balances efficiency
and accuracy, and characterize each operator’sbatching sensitiv-
ity—its performance response to varying batch size 𝑇and input
length. Using the sem_map sentiment classification operator, we
compare two datasets with contrasting input lengths—short Twitter
texts [ 6] and longer Amazon Fine Food Reviews [ 15]—to quantify
how throughput and accuracy evolve as 𝑇increases. Figure 6 illus-
trates the resulting throughput–accuracy trade-offs and highlights
the differing sensitivities of operators to batching.
5

Table 2: Operators, tasks, and evaluation metrics used in the
stock news dataset.
Operator Task Metric
map bi Binary sentiment classification (positive
vs.negative)F1
map multi Multi-class classification extracting the
referenced company (e.g., AAPL, TSLA)Macro-F1
map sum Sentence-level summarization of a sin-
gle news itemBERTScore-F1
filter Boolean semantic filter that keeps tu-
ples referring to a target companyF1
topk 𝑘 Selects the𝑘most impactful news items
(𝑘=1,3,9)Recall@k
agg Window-level summarization of news
content and sentimentBERTScore-F1
E2E Pipeline All operators jointly F1
4.2 Operator Fusion
Operator fusion[ 22] executes multiple logical operators within a sin-
gle LLM invocation by combining adjacent operators to eliminate
redundant prefill and per-request overhead. Rather than invoking
the model separately for each stage (e.g., filter ,map), fusion com-
poses consecutive operators into one prompt, thereby reducing
repeated initialization costs and token-level computation. A key
potential side effect issemantic interference, where combining multi-
ple operator intents into a single prompt can alter or degrade model
behavior unless the prompt is carefully structured and validated.
Implementation.To execute a chain of operators in a single LLM
call, VectraFlow constructs afused schemathat exposes all fields pro-
duced across the sequence. For a fusible chain Π=(𝑜𝑝 1,...,𝑜𝑝𝐿),
the fused schema is defined as
schema(fuse(Π))=schema(𝑜𝑝 1)∪···∪schema(𝑜𝑝 𝐿),
with attribute name collisions resolved through namespacing or
user-defined aliases. The LLM is then instructed, via a fused prompt,
to apply each operator’s logic step-by-step and output a single JSON
object adhering to the combined schema, thereby preserving the
pipeline’s semantics within one invocation.
Evaluation.We evaluate the operator fusion on a curated FN-
SPID [ 7] subset and report execution time (s), accuracy at both op-
erator and end-to-end (E2E) levels, and token usage (prompt/prefill
vs. generation). All stages use gpt-4o-mini . VectraFlow exposes
a comprehensive set of semantic operators and their fusible com-
binations, and we evaluate both individual operators and all valid
fused variants. To capture a range of task difficulties, we run each
operator on multiple settings (e.g., binary classification, multi-class
classification, and summarization for map). Table 2 summarizes
each operator, its task, and the evaluation metrics.
Filter-Aware Fusion.A filter behaves like a binary map whose
selectivity controls downstream load, so we evaluate it separately.
Table 3 compares unfused vs. fused orders for map andfilter : fusing
yields∼1.31×speedup for map→filter and∼1.08×forfilter→
map, with accuracy drops of0.065and0.047, respectively.
We vary filter selectivity 𝑠in Table 4 and observe that low selec-
tivity (dropping most tuples) makes fusion worse, since the fused
pipeline still executes the downstream operator on tuples the filterTable 3: Filter-involved fusion: time, accuracy, and token
usage. Tokens reported as P/G = prompt / generation.
Approach Time (s) Accuracy Operator Tokens (P/G)
Map→Filter 52.12 0.780Map 225/27
Filter 218/10
Fused(Map→Filter) 39.76 0.715 Fused Op 324/33
Filter→Map 42.83 0.791Filter 220/10
Map 240/28
Fused(Filter→Map) 39.64 0.744 Fused Op 324/33
Table 4: Relative speed gain ofFusedvs.Non-Fusedunder
filter selectivity𝑠.
Fusion Type 10% 30% 50% 80% 100%
Map→Filter (Fused vs. Non-Fused) 23.11% 23.40% 21.72% 21.16% 19.43%
Filter→Map (Fused vs. Non-Fused)− 10.35%−3.99% 3.21% 16.27% 21.17%
would have discarded, outweighing the benefits of eliminating an
extra call. As 𝑠increases, this penalty shrinks and the benefit of
eliminating an extra LLM call grows, making fusion increasingly
favorable and eventually dominant.
Filter-Free fusion.Table 5 reports fusion outcomes for non-filter
operator pairs using the formal operator labels. Each row pair shows
the baseline pipeline and its fused counterpart, along with the
measuredSpeedup,F1 loss. As a rule of thumb, smaller |Δ|(trade-off
ratio closer to0) indicates a more favorable performance–accuracy
balance.
Takeaway.These results suggest a rule of thumb for when fusion
is effective. Fusion is generally safe and beneficial for operator pairs
that apply lightweight transformations. This includes map→map ,
op→map , and op→filter , where we observe low sensitivity and a
trade-off ratio ( Δ) close to zero. Fusion becomes more fragile for
operators such as topk andagg, where the output depends heavily
on ranking quality or the reliability of the generated summary. In
these cases, accuracy is more sensitive to small LLM errors, and
speedups depend on parameters such as 𝑘or window size. When
𝑘approaches the window size, the operator effectively selects all
elements in the window, accuracy increases because the operator no
longer relies on fine-grained ranking. Fusion can be risky for topk
andagg, as its stability still depends on 𝑘, and should therefore be
validated empirically. Fusion is most effective when operators apply
light transformations and least stable when they require precise
ranking or summarization behavior.
5 DYNAMIC PLANNING FRAMEWORK
Our dynamic planning framework (Figure 7) comprises three com-
ponents: aplan generator, which enumerates candidate configura-
tions (batch sizes, operator variants, and fusion options); acost esti-
mator, which learns per-operator throughput and accuracy models
from sampled configurations; and aplan optimizer, which uses these
models to predict pipeline-level performance, construct the Pareto
frontier, and select the configuration that meets user-specified
throughput and accuracy targets.
6

Segoe 
Dynamic Planning Framework  Cost Estimator Plan Generator op1 op2 op3 op4 op1 fused (op2&3) op4 fused (op1&2&3) op4 op1 op2 op3 op4 T1=1 T2=1 T3=1 T4=1 
Op1.2 op2 Op3.2 op4 plan 1 plan 2 plan 3 plan 4 plan 5 T1=1 T2=1 T3=1 T4=2 T1=1 T2,3=2 T4=1 T1,2,,3=2 T4=1 T1=1 T2=2 T3=1 T4=4 
Input Data Streams Data-Flow Engine Filter 
Map 
Group-By 
Top-K 
Output Data Streams 
Throughput Model (y(T)) 
sample 
Accuracy Model (A(T)) Train Model Operator Proﬁle  
Maintain per-operator models as a function of  T A(T) y(T) A1(T) y1 (T) op1 A2(T)  y2 (T) op2 A12(T) y12 (T) fused12 A123(T) y123 (T) fused123 … … … Plan Optimizer 
x x x Acc 
Thr plan 1485 plan 1 plan 4 y* Select plan 4 that satisﬁes the throughput target y*! §E2E Throughput: bottleneck (min)  §E2E Accuracy: multiplicative Objective Function Model Training 
Shadow Execution Figure 7: Overview of the dynamic planning framework in VectraFlow.
Table 5: Baseline vs. fused operator performance
with throughput speedup and F1 loss. A smaller
Δ(F1 loss)/Δ(speedup) indicates a better throughput–
accuracy tradeoff. Values marked with★denote better
throughput–accuracy tradeoffs (smallerΔF1/ΔSpeedup).
Operator PairThroughput E2E F1𝚫F1/𝚫Speedup(base / fused) (base / fused)
map_multi→map_bi 0.65 / 1.13 0.7952 / 0.7730 0.038★
map_bi→map_multi 0.76 / 1.22 0.7879 / 0.7730 0.031★
map_sum→map_bi 0.51 / 0.62 0.6413 / 0.6237 0.130
map_bi→map_sum 0.51 / 0.71 0.6360 / 0.6296 0.027★
map→topk(k=3) 0.87 / 1.48 0.5386 / 0.5238 0.039★
map→topk(k=9) 0.82 / 1.00 0.7867 / 0.7325 0.308
map→agg 0.92 / 3.37 0.7280 / 0.0530 0.344
agg→map 1.02 / 2.49 0.8340 / 0.6530 0.173
topk(k=3)→map 1.13 / 1.31 0.6430 / 0.5430 0.943
topk(k=3)→agg 1.04 / 5.11 0.5900 / 0.3550 0.102
topk(k=3)→topk(k=1) 1.18 / 1.32 0.2860 / 0.2860 0.000★
topk(k=9)→map 0.63 / 1.09 0.9080 / 0.9010 0.011★
topk(k=9)→agg 0.96 / 3.35 0.6020 / 0.5760 0.022★
topk(k=9)→topk(k=1) 0.67 / 1.27 0.5260 / 0.5260 0.000★
5.1 Plan Generation and Pruning
Rather than relying solely on heuristics, VectraFlow collects ex-
ecution logs fromshadow runsacross the design space to train
predictive performance models. To explore this space, the system
employs aplan generatorthat automatically enumerates possible
execution strategies for a given pipeline. The generator considers
four families of plans: (1) a baseline plan without optimizations, (2)
fusion plans that combine consecutive operators to reduce invoca-
tion and prefill overhead, (3) batching plans that assign different
tuple batch sizes to operators to balance throughput and accuracy,
and (4) hybrid plans that integrate both fusion and batching. Beyond
these optimization strategies, the generator also explores alterna-
tive implementations ofcontinuous semantic operators, evaluating
both LLM-based (prompt-driven) and embedding-based variants
to capture different accuracy–throughput trade-offs. It supportspipelines of arbitrary length and enumerates all feasible combina-
tions up to a configurable maximum batch size, before applying
pruning rules to remove invalid or redundant configurations.
The plan generator prunes invalid configurations in the follow-
ing order: (1) Fusion infeasibility: remove plans that attempt to fuse
operators tied to different window contexts, since such operators
cannot be fused without contaminating the prompt context. (2)
Window constraints: discard plans where the batch size exceeds
the active window size ( 𝑇>𝑊 ), as a batch cannot contain more
tuples than the window admits. (3) Batching constraints: enforce
non-decreasing batch sizes across consecutive operators ( 𝑏𝑖+1≥𝑏𝑖)
to maintain balanced throughput, while allowing exceptions for
selective operators (e.g., filters), where downstream batch sizes may
shrink according to the operator’s selectivity 𝑠. Together, these
pruning rules substantially reduce the search space while preserv-
ing semantically valid and executable plans.
5.2 Cost-Aware Estimation
The planner buildsper-operatorpredictive models by sampling
data and plan configurations for each operator implementation or
variant. For every operator, we collect measurements across differ-
ent tuple-batch sizes 𝑇and train paired models that capture how
throughputandaccuracybehave as functions of 𝑇. Thethrough-
put modeldescribes how performance scales with batching, while
theaccuracy modelcharacterizes how quality changes as multiple
tuples are processed together.
Throughput Model.Tuple batching affects throughput by increas-
ing the size of the prompt and generated text in proportion to the
batch size. Processing a batch of 𝑇tuples requires constructing
a prompt that embeds all 𝑇items (linear in 𝑇) and generating 𝑇
corresponding outputs (also linear in 𝑇). Thus, the batch service
time for an operator can be well-approximated by an affine function
𝑠(𝑇)=𝑎𝑇+𝑏,
where𝑎captures the per-tuple inference cost and 𝑏reflects fixed
overhead such as boilerplate prompt text and model invocation
7

latency. The resulting per-operator throughput is
𝑦(𝑇)=𝑇
𝑠(𝑇)=𝑇
𝑎𝑇+𝑏tuple/s.(1)
This equation expresses an expected trade-off: throughput in-
creases rapidly for small 𝑇as fixed overhead is amortized, but
gradually saturates as the linear per-tuple cost becomes dominant.
This model matches the observed behavior across LLM-based op-
erators, providing a simple, operator-level predictor of batching
efficiency.
Accuracy Model.To understand how tuple batching affects op-
erator quality, we evaluate four representative semantic operators
drawn from two datasets. From the stock news dataset, we study (i)
asemantic group-byoperator that implements a company classifier,
and (ii) asemantic filterthat implements sentiment analysis. From
the Amazon Food Reviews dataset, we evaluate (iii) asemantic map
operator that summarizes user reviews and (iv) asemantic top- 𝑘
operator that rates the helpfulness of reviews. For each operator, we
sweep over a range of batch sizes 𝑇and run five trials with different
random seeds (1–5) to expose ordering and position effects. This
allows us to quantify how batching influences accuracy as more
tuples are packed into a single prompt.
In Figure 8, we observe that the same pattern emerges across
all operators: accuracy is highest at 𝑇=1and then consistently de-
creases as𝑇grows. The decline is steep for small increases in 𝑇
and becomes progressively shallower at larger batch sizes. This
behavior suggests anexponential decaytrend, which captures this
fast-initial-drop and slow-tail dynamic. Intuitively, batching intro-
duces several sources of degradation, including semantic interfer-
ence across items, position bias, and reduced attention allocated to
any single tuple, which accumulate quickly at first and then taper
off as prompts become saturated. We model the expected accuracy
at batch size𝑇as
𝐴(𝑇)=𝐴 max·𝑒−𝛽(𝑇−1).(2)
where𝐴maxcorresponds to the operator’s baseline accuracy at 𝑇=1
and𝛽is a decay parameter estimated from profiling runs. This for-
mulation aligns with findings from batch prompting studies [ 5,10],
which report that batching improves efficiency but can degrade
model performance due to order sensitivity and cross-item inter-
ference. Our exponential model provides a lightweight, empirically
grounded approximation of these trends, allowing the planner to
predict accuracy loss when exploring batching configurations.
5.3 Plan Optimizer
With per-operator models in place, the optimizer composes these
estimates into pipeline-level predictions and recommends plan con-
figurations that best satisfy user-defined throughput and accuracy
objectives. Given operator-level throughput 𝑦𝑖(𝑇)and accuracy
𝐴𝑖(𝑇)learned from profiling, the optimizer explores candidate
plans, predicts their end-to-end (E2E) performance, constructs the
Pareto frontier, and returns a configuration that matches the user’s
desired trade-off.
End-to-End Throughput.VectraFlow supports two execution
modes, each inducing a different composition rule for end-to-end
throughput𝑦 e2e(𝑥).
0 5 10 15 20 25 30 35
Tuple Batch Size (T)0.40.50.60.70.80.9Accuracy
Group-by
rs=1
rs=2
rs=3rs=4
rs=5
rs=0
Exp fit (mean), R² = 0.913(a) Company classifier
0 5 10 15 20 25 30 35
Tuple Batch Size (T)0.40.50.60.70.80.9Accuracy
Filter
rs=1
rs=2
rs=3rs=4
rs=5
rs=0
Exp fit (mean), R² = 0.903 (b) Sentiment analysis
0 5 10 15 20 25
Tuple Batch Size (T)0.860.880.900.920.940.960.981.00Accuracy
Map
s=1
s=2
s=3s=4
s=5
s=0
Exp fit (mean), R² = 0.801
(c) Review summarization
0 5 10 15 20 25
Tuple Batch Size (T)0.50.60.70.80.91.0Accuracy
T opK
s=1
s=2
s=3s=4
s=5
s=0
Exp fit (mean), R² = 0.858 (d) Review helpfulness
Figure 8: Effect of tuple batch size 𝑇on operator accuracy
across four operators: (a) company classifier, (b) sentiment
analysis, (c) review summarization, and (d) review helpful-
ness top-𝑘.
Pipeline-parallel execution.When operators run concurrently
with multiple in-flight batches, throughput is governed by the bot-
tleneck stage:
𝑦e2e(𝑥)=min
𝑖𝑦𝑖(𝑇𝑖).
Sequential execution.When batches are processed one at a time,
operator latencies accumulate. The corresponding end-to-end through-
put is the harmonic mean of operator throughputs:
𝑦e2e(𝑥)=1Í
𝑖1/𝑦𝑖(𝑇𝑖).
These formulations allow users to select the throughput objective
that best matches their execution semantics, or to specify custom
alternatives, as long as the objective can be expressed as a function
of per-operator throughput (e.g., weighted throughput). Objectives
that require joint modeling beyond individual operators are not
supported, since VectraFlow maintains learned performance models
at the operator level rather than for the entire pipeline.
End-to-End Accuracy.To estimate pipeline-level accuracy, Vec-
traFlow assumes that operators are independent. Under this simpli-
fying assumption, the E2E accuracy for a planTis approximated
as the product of per-operator accuracies:
𝐴E2E(T) ≈Ö
𝑖𝐴𝑖(𝑇𝑖).
The system also supports user-defined accuracy objectives that
can be expressed through per-operator models (e.g., weighted prod-
ucts, max/min). For more complex end-to-end metrics (e.g., F1),
VectraFlow can fall back to sparse pipeline-level sampling, using
those samples as targets for plan selection. Objectives that depend
on information unavailable from either operator models or pipeline
outputs are out of the scope of our current framework.
8

Plan Selection and Pareto Frontier.For each candidate plan, the
optimizer evaluates the chosen throughput metric together with
its predicted E2E accuracy, constructs the Pareto frontier of non-
dominated plans, and then returns a configuration that either meets
a user-specified throughput target with the highest possible accu-
racy or provides the best accuracy–performance trade-off along
the frontier. This approach allows the optimizer to efficiently navi-
gate a large plan space and tailor recommendations to the unique
constraints and objectives of each pipeline.
5.4 Extensibility and Modular Optimization
VectraFlow’s optimization framework is modular and extensible
by design. Rather than being tied to a fixed set of strategies, it
allows new optimization dimensions to be incorporated through
lightweight implementation rules. Beyond batching, fusion and
alternative operator implementations (LLM-based or embedding-
based), the planner can integrate additional layers such as model
selection (e.g., switching between lightweight and high-fidelity
LLMs) or query rewriting. Each module exposes a uniform inter-
face, enabling the planner to reason jointly about model capacity,
operator fidelity, and resource usage. This design allows VectraFlow
to remain adaptable and to incorporate future inference and opti-
mization techniques without modifying the core planning logic.
6 MULTI-OBJECTIVE BAYESIAN
OPTIMIZATION
Learning the throughput–accuracy Pareto frontier of an LLM-based
pipeline requires selectively probing operators under different tuple-
batch sizes𝑇and sampling rates 𝑠. Exhaustive enumeration is infea-
sible: even a four-operator pipeline with 𝑇≤ 10already yields over
20,000 plan configurations. Since we can only sample a small sub-
set of operators and configurations, the key challenge is deciding
whichoperators to probe andat whatsampling rates. This moti-
vates aCost-Aware Multi-Objective Bayesian Optimization(MOBO)
framework that leverages structural priors—such as the monotonic
and saturating relationships between batching, throughput, and
accuracy to guide exploration. By encoding these operator-level
trends as prior functions, MOBO learns the Pareto frontier far more
efficiently than naive random or bandit-based sampling, enabling
principled allocation of a limited probing budget.
6.1 Problem Statement
We aim to learn the throughput–accuracy Pareto frontier of a multi-
operator LLM pipeline under a fixed probing cost budget 𝐵. Each
probe evaluates a configuration (𝑖,𝑇,𝑠) —operator index, tuple batch
size, and sampling rate—and consumes part of this budget. The goal
is to discover high-quality pipeline plans that differ in batching
choices, fusion decisions, and operator variants while respecting
the total probing budget. Formally, we solve the multi-objective
optimization problem
max
𝑥∈X 𝑦e2e(𝑥),𝐴 e2e(𝑥)s.t.𝑁∑︁
𝑡=1cost(𝑖𝑡,𝑇𝑡,𝑠𝑡)≤𝐵,
whereXis the space of all feasible plan configurations, 𝑦e2e(𝑥)
denotes end-to-end throughput, and 𝐴e2e(𝑥)denotes end-to-end
accuracy. The goal is to recover the Pareto-optimal set.6.2 Algorithm
Phase I: Model Warm-Up and Priors.We begin with a light-
weight warm-up phase using a small sampling rate (e.g., 𝑠0=0.1).
Each operator is probed at a few representative batch sizes (e.g.,
𝑇∈{ 1,2,8}), producing initial measurements that seed the surro-
gate models. From these warm-up samples, we fit two parametric
priors that capture the empirical behavior described in Section 5.2:
the throughput prior (Eq. 1) and the accuracy prior (Eq. 2). These
priors reflect the observed sublinear saturation of throughput with
batch size and the exponential decay of accuracy as more tuples
share a prompt.
Using these priors, each operator maintains two independent
Gaussian-process (GP) surrogate models—for throughput and ac-
curacy—with mean functions 𝜇𝑦𝑖(𝑇)and𝜇𝐴𝑖(𝑇)and predictive
variances𝜎2
𝑦𝑖(𝑇)and𝜎2
𝐴𝑖(𝑇). To model the effect of sampling rate
𝑠, we augment the GP observation noise with a variance term pro-
portional to1/𝑠, reflecting that lower sampling rates yield noisier
estimates. We treat operators as conditionally independent given
the GPs. For any plan configuration 𝑥, we obtain a predictive distri-
bution over end-to-end throughput and accuracy by composing the
operator-level predictive Gaussians according to the user-specified
pipeline objective (e.g., bottleneck throughput and multiplicative
accuracy).
Phase II: Cost-Aware Acquisition and Iterative Probing.We
select the next probe by maximizing a cost-aware acquisition func-
tion:
𝑈(𝑖,𝑇,𝑠)=EHVI(𝑖,𝑇,𝑠)
cost(𝑖,𝑇,𝑠),(𝑖∗,𝑇∗,𝑠∗)=arg max
𝑖,𝑇,𝑠𝑈(𝑖,𝑇,𝑠).
Expected Hypervolume Improvement (EHVI).Let Fbe the current
non-dominated frontier and 𝑟a dominated reference point. For a
probe that updates operator𝑖at(𝑇,𝑠), we compute
EHVI(𝑖,𝑇,𝑠)=E
HVI (𝑦e2e,𝐴e2e),F;𝑟
,
using a Gaussian approximation to the predictive distribution from
operator-level surrogates (with the expectation evaluated via Monte
Carlo). End-to-end predictions are obtained by composing the per-
operator surrogate posterior means according to the user-specified
objectives. By default, we use bottleneck throughput 𝑦e2e(𝑥)=
min𝑖𝑦𝑖(𝑇𝑖)and multiplicative accuracy𝐴 e2e(𝑥)=Î
𝑖𝐴𝑖(𝑇𝑖).
Cost model.We approximate the cost of probing a configuration
(𝑖,𝑇,𝑠)as
cost(𝑖,𝑇,𝑠)=𝑛𝑠
ˆ𝑦𝑖(𝑇).
where𝑛is the number of tuples used for profiling, 𝑠is the sampling
rate,𝑛𝑠denotes the number of tuples actually processed during
the probe, ˆ𝑦𝑖(𝑇)denotes the posterior mean of the throughput
surrogate, yielding a simple and adaptive estimate of the probe’s
execution time.
After selecting a probe (𝑖∗,𝑇∗,𝑠∗), we execute it to obtain the ob-
served throughput and accuracy, update the corresponding operator-
level Gaussian surrogates, and refresh the non-dominated frontier
Fusing the updated end-to-end predictions. This cycle continues
until the probing budget 𝐵is exhausted, at which point we evaluate
all explored configurations under the final surrogate means and
extract pipeline’s resulting throughput–accuracy Pareto frontier.
9

7 EVALUATION
We evaluate VectraFlow on two representative streaming pipelines
to examine two key experimental claims:
•Effectiveness of optimization strategies.We show that
both operator-level and pipeline-level optimizations, includ-
ing tuple batching and operator fusion, can substantially
improve performance, while the planner selects configura-
tions that minimize accuracy degradation for the desired
throughput level.
•Sampling efficiency through MOBO.We show that cost-
aware sampling strategies, guided by multi-objective Bayesian
optimization, allow VectraFlow to identify Pareto-efficient
plans more efficiently than heuristic or random baselines.
7.1 Experimental Setup
To evaluate the performance characteristics of our proposed tech-
niques within the VectraFlow system, we conducted a series of
experiments. We evaluate our techniques using the Qwen/Qwen2.5-
7B-Instruct model, a 7B-parameter instruction-tuned language model
capable of handling complex, structured prompts. The model is de-
ployed using the vLLM [ 9] inference server, which provides efficient
batching and prefix caching. All experiments are conducted on a
single NVIDIA GeForce RTX 3090 GPU with 24GB of memory. For
reproducibility, we set temperature to t = 0 for all methods and
baselines, unless otherwise stated. This setup reflects a realistic
production-grade inference stack for streaming LLM applications.
We evaluate our dynamic planning framework in anofflineset-
ting using a train–test split of 100 data points each: 100 data points
are used to train the throughput and accuracy models, and another
100 are used to evaluate how well each method recovers the ground-
truth frontier. A larger training set is impractical because the plan
space exceeds 10,000 configurations, and fully evaluating all of
them requires more than 48 hours of end-to-end execution time.
Each method is then run on this offline workload and produces its
own predicted Pareto-optimal plans under a given budget, enabling
a direct comparison between the predicted and actual frontiers.
In practice, this offline frontier also enables online adaptation: at
runtime, the system maps the observed arrival rate to the corre-
sponding point on the precomputed frontier and selects the optimal
plan configuration.
We compare four strategies: (1)Heuristic-Guided Sampling Per-
Pipeline (Heuristic Pipe), which evaluates full pipeline configurations
using rule-driven heuristics derived from a small warm-up phase;
(2)Heuristic-Guided Sampling Per-Operator (Heuristic Op), which
applies the same heuristics but samples operator configurations in-
dependently; (3)MOBO (no warmup), our multi-objective Bayesian
optimization framework without the warm-up stage; and (4)MOBO,
which incorporates warm-up and prior-guided exploration. Both
heuristic-guided methods use statistics collected during the warm-
up phase to identify unpromising plans (e.g., fuse operators 1 and 2),
enabling the sampler to prune the search space before evaluation.
We also evaluatedRandom Sampling Per-Pipeline(Random Pipe)
andRandom Sampling Per-Operator(Random Op), but as both meth-
ods were consistently inferior to the heuristic-guided strategies, we
omit them from the subsequent results for clarity.
cts_filtersem_mapsem_groupbysem_topk…Fetch financial news relevant to my stock portfolioFinancial NewsClassify sentiment: {positive, negative}”Extract company symbol (AAPL, MSFT)Identify the five most impactful stock market news Output Stream(static)Figure 9: Stock News Monitoring Pipeline.
100 200 300 400 500
Cost Budget0.00.20.40.60.81.0Recall (fraction of targets)
Recall vs. Budget
Heuristic Pipe
Heuristic OpMOBO (no warmup)
MOBO
(a) Recall vs. Cost Budget.
100 200 300 400 500
Cost Budget0.00.20.40.60.81.0Precision (fraction of targets)
Precision vs. Budget
Heuristic Pipe
Heuristic OpMOBO (no warmup)
MOBO (b) Precision vs. Cost Budget.
Figure 10: Recall and Precision vs. Cost Budget for the Stock
News Monitoring Pipeline. MOBO consistently achieves
higher recall and precision across all budgets, saturating
after𝐵=300as it recovers nearly all Pareto-optimal plans.
7.2 Stock News Monitoring
Pipeline.Figure 9 shows our semantic streaming pipeline for
stock monitoring on the FNSPID [ 7] dataset. A live financial news
feed is first processed by the cts_filter operator, which continu-
ously retrieves and filters news articles relevant to a given stock
portfolio. The filtered stream is passed to a sem_map , which per-
forms data cleaning and structuring to normalize the input. Next,
asem_groupby groups news articles by their associated company
ticker (e.g., TSLA, AAPL). Within each group, the sem_topk oper-
ator selects the most impactful articles using a sem_window that
adapts to topic drift in financial news, ensuring that related develop-
ments are grouped together even when they unfold gradually across
multiple articles. Finally, sem_agg summarizes the top-ranked arti-
cles within a count-based window, producing concise portfolio-level
insights and recommendations.
Results.Figure 10 reports recall and precision under fixed cost
budgets (𝐵), measuring how many true Pareto frontier plans each
method successfully retrieves. Across all budgets, MOBO consis-
tently achieves the highest recall, converging near 𝐵=300with
diminishing improvements thereafter. Heuristic Op outperforms
Heuristic Pipe because full-pipeline evaluation wastes budget on
end-to-end measurements, whereas operator-level sampling allows
more focused exploration. The MOBO variant without warmup re-
trieves fewer frontier plans, showing that the warm-up phase is key
for stabilizing early model behavior and speeding up convergence.
Table 6 summarizes the prevalence of execution optimizations
among Pareto-efficient plans at 𝐵=300. Across all retrieved plans,
both tuple batching and operator fusion appear frequently. In partic-
ular, batching dominates across nearly all efficient pipelines, while
fusion appears more selectively.
Figure 11 shows a stepwise adoption of execution optimizations
along the Pareto frontier. Early gains come solely from tuple batch-
ing, which preserves high accuracy; higher-throughput regions
10

Table 6: Adoption of execution optimizations across Pareto-
efficient plans for the Stock News Monitoring Pipeline.
Optimization Type Pipeline-Level Operator-Level
Tuple Batching 8 / 9 (89%) 16 / 36 (44%)
Operator Fusion 3 / 9 (33%) 6 / 36 (17%)
Operator Variants 0 / 9 (0%) 0 / 36 (0%)
3.5 4.0 4.5
Throughput0.20.40.60.81.0Accuracy12
3
45
67
89Original pipeline
Batching only
Fusion only
Batching + Fusion
Figure 11: Stepwise adoption of optimization strategies along
the Pareto frontier for the Stock News Monitoring Pipeline.
introduce fusion for additional speedup; and the extreme end com-
bines batching and fusion to maximize throughput at the cost of
modest accuracy loss. This progression illustrates how VectraFlow
adaptively selects optimizations across throughput tiers. No fil-
ter variants appear among these efficient plans because the op-
timization objective was defined onbottleneck throughput: while
embedding-based filters increase total throughput, they marginally
reduce accuracy and do not improve the slowest stage in the pipeline.
Figure 12 reports throughput and accuracy as we simulate a
streaming workload by replaying 1,200 tuples with Poisson inter-
arrival times. The arrival rate 𝜆increases every 100 tuples, pro-
gressively stressing the system. In the throughput plot (Figure 12a),
thebaselineshows an almost flat curve: because its plan is fixed, it
cannot react to higher arrival rates. In contrast,MOBOdynamically
reconfigures the pipeline as 𝜆increases, allowing its throughput to
closely track—and eventually saturate at—the system’s maximum
achievable rate after leveraging all available execution optimiza-
tions (tuple batching, fusion, and operator variants). Theheuristic
strategy initially keeps up, but saturates much earlier and steadily
falls behind as the stream accelerates. In the accuracy plot (Fig-
ure 12b), thebaselinemaintains perfect accuracy because it never
changes configuration. Theheuristicpolicy suffers the steepest
drop: its aggressive reconfigurations fail to meet throughput de-
mands while substantially degrading accuracy.MOBOalso trades
accuracy for speed at high arrival rates, but does so in a controlled,
model-guided manner, preserving significantly more accuracy than
the heuristic in the overloaded regime.
7.3 Misinformation Event Monitoring
Pipeline.Figure 13 illustrates our end-to-end semantic streaming
pipeline for misinformation event monitoring. We continuously
ingest the MiDe22 stream, apply a sem_filter to discard items un-
likely to contain misinformation, dynamically group the remaining
tuples by semantic topic via sem_groupby , segment each topic’s
flow into dynamic event-context windows using sem_window , and
2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5
Arrival rate (tuples/s)2345Throughput (tuples/s)
Throughput vs. Arrival rate
Baseline throughput
MOBO throughput
Heuristic throughput(a) Throughput vs. Arrival Rate.
2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5
Arrival rate (tuples/s)0.00.20.40.60.81.0Accuracy
Accuracy vs. Arrival rate
Baseline accuracy
MOBO accuracy
Heuristic accuracy (b) Accuracy vs. Arrival Rate.
Figure 12: Throughput and Accuracy vs. Arrival Rate for the
Stock News Monitoring Pipeline.
sem_filtersem_groupbysem_windowsem_topk…Remove tweets containing misinformationNews StreamClassify the news according to its topicDetect event shifts over timeIdentify the three most urgent eventsOutput Stream(dynamic)
Figure 13: Misinformation Event Monitoring Pipeline.
800 1000 1200 1400
Cost Budget0.00.20.40.60.81.0Recall (fraction of targets)
Recall vs. Budget
Heuristic Pipe
Heuristic OpMOBO (no warmup)
MOBO
(a) Recall vs. Cost Budget
800 1000 1200 1400
Cost Budget0.00.20.40.60.81.0Precision (fraction of targets)
Precision vs. Budget
Heuristic Pipe
Heuristic OpMOBO (no warmup)
MOBO (b) Precision vs. Cost Budget
Figure 14: Recall and Precision vs. Cost Budget for the Misin-
formation Event Monitoring Pipeline. MOBO maintains su-
perior recall and precision, yielding higher-quality retrievals
than random baselines.
within every window compute sem_topk (𝑘=3) based on an ur-
gency scoring function. The pipeline produces a live feed of the
three most urgent misinformation events per window for down-
stream alerting, monitoring, or visualization.
Results.Figure 14 reports recall and precision under fixed cost bud-
gets (𝐵={800,1000,1200,1400}), measuring how many true–Pareto
frontier plans each method successfully retrieves. MOBO consis-
tently achieves the highest recall and precision, converging near
𝐵=1200with diminishing improvements thereafter. Heuristic Op
consistently surpasses Heuristic Pipe because Pipe allocates its bud-
get to complete end-to-end evaluations instead of focused operator-
level probes. The MOBO variant without warmup retrieves fewer
frontier plans, underscoring the importance of the warmup phase
for stabilizing initial model behavior and speeding up convergence.
Table 7 shows that tuple batching is universally adopted across all
efficient pipelines (excluding the baseline) and thus serves as the pri-
mary throughput optimization. Operator variants are also present
in every pipeline, with most pipelines relying on embedding-based
variants. In contrast, operator fusion appears only once, suggesting
that it offers benefits only in a narrow set of high-throughput cases
and is not broadly advantageous across the Pareto frontier.
11

Table 7: Adoption of execution optimizations across Pareto-
efficient plans for the Misinformation Event Monitoring
Pipeline.
Optimization Type Pipeline-Level Operator-Level
Tuple Batching 15 / 16 (94%) 48 / 64 (75%)
Operator Fusion 1 / 16 (6%) 2 / 64 (3%)
Operator Variants 15 / 16 (94%) 64 / 64 (100%)
sem_groupby (embedding) 7 / 16 (44%) 28 / 64 (44%)
sem_window (pairwise) 4 / 16 (25%) 16 / 64 (25%)
sem_window (clustering) 4 / 16 (25%) 16 / 64 (25%)
4 5 6 7 8
Throughput0.40.60.81.0Accuracy1
23
4567
8
9
1011
12
1314
15
16original pipeline
tuple batching + sem_groupby
(embedding)
tuple batching + sem_groupby
(embedding) + sem_window (pairwise)
tuple batching + sem_groupby
(embedding) + sem_window
(clustering)
tuple batching + sem_groupby
(embedding) + sem_window
(clustering) + fusion
Figure 15: Stepwise adoption of execution optimizations
along the Pareto frontier for the Misinformation Event Mon-
itoring Pipeline.
Figure 15 illustrates a clear stepwise adoption of optimizations
along the throughput–accuracy frontier. All Pareto-optimal pipelines
use tuple batching and sem_groupby (embedding) as a baseline,
while progressively stronger techniques are added with increasing
throughput: mid-range configurations incorporate sem_window
(pairwise), higher-throughput designs replace this with sem_window
(clustering), and the maximum-throughput setup additionally ap-
plies operator fusion over these variants.This pattern shows that
VectraFlow progressively adds execution optimizations to boost
throughput, accepting substantial accuracy degradation only in its
most aggressive, fusion-driven mode.
8 RELATED WORK
LLM-powered data processing systems and optimizations.A
growing line of systems extends the relational model with semantic
operators for processing unstructured data using LLMs [ 2,8,11,13,
17,19]. Lotus [ 17] introduces a declarative interface for semantic
pipelines and optimizes individual operators via model cascades
that combine a high-quality “gold” algorithm with a cheap proxy.
Palimpzest [ 12] and its successor Abacus [ 18] explore cost-based
optimization over models, prompts, and operator variants, formu-
lating pipeline selection as a multi-objective optimization problem
in offline, batch settings. DocETL [ 19] applies LLM-based query
rewriting to transform documents into structured representations,
using LLMs both to rewrite pipelines and validate the rewrites.
ZenDB [ 11] targets semi-structured document analytics with se-
mantic indexes and logical rewrites such as predicate reordering
and projection pull-up. These systems all operate in batch or one-
shot modes over static datasets. They do not address the challenges
of continuous execution over unbounded, evolving streams, nor dothey model or optimize the runtime performance–accuracy trade-
offs that arise when LLM operators must be executed persistently.
Our prior work (VectraFlow [ 14]) provides the vector-based ana-
logue of a continuous processing engine, supporting streaming
operations over embedding vectors. In contrast, this paper brings
continuous processing to the LLM layer: we introduce continuous
semantic operators, model their accuracy–throughput behavior,
and develop a dynamic optimization framework—including tuple
batching, operator fusion, embedding-based variants, and MOBO-
driven plan selection—for continuous LLM-powered pipelines.
Stream processing systems. Early systems (e.g., Aurora [ 1],
STREAM [ 16]) established the foundations of modern data-stream
management, introducing core techniques for continuous queries,
approximation, and adaptive resource management. These ideas
were later extended by open-source engines such as Apache Flink
[3] and Apache Storm [ 20], which advanced the field through uni-
fied batch/stream execution, low-latency pipelines, and scalable,
fault-tolerant processing over structured data streams.
In contrast, VectraFlow targets LLM-based data processing over
unstructured streams with novel continuous semantic operators.
It integrates LLM-native optimizations into a dynamic planning
framework. Whereas conventional systems emphasize standard
performance metrics, VectraFlow explicitly incorporates accuracy,
allowing for systematic performance–accuracy trade-offs that are
intrinsic to LLM-based processing.
9 CONCLUSIONS
This paper introducesContinuous Prompts, a new framework for
LLM-augmented stream processing that enables persistent, semantic-
aware queries over unstructured data. We extended RAG to stream-
ing settings, defined continuous semantic operators with several
practical implementations, and characterized LLM-specific execu-
tion optimizations that shape the performance–accuracy trade-offs
of continuous LLM pipelines. We further developed a dynamic
planning framework that models operator sensitivities and uses a
multi-objective Bayesian optimization (MOBO) strategy to learn
throughput–accuracy frontiers under limited probing budgets.
Our evaluation, combining operator-level microbenchmarks with
realistic end-to-end pipelines, shows that continuous prompts can
dynamically adapt to workload fluctuations and effectively navigate
accuracy–efficiency trade-offs in evolving unstructured streams.
Taken together, these contributions advance LLM-based stream
processing from static, one-shot pipelines to adaptive, continuously
optimized semantic computations over unbounded data.
ACKNOWLEDGMENTS
We gratefully acknowledge the support provided by a Brown Seed
Fund. We extend our special thanks to Weili Shi for his major
contributions to the initial VectraFlow prototype. We also thank the
rest of the team for their valuable feedback throughout this work.
REFERENCES
[1] D. J. Abadi, D. Carney, U. Çetintemel, M. Cherniack, C. Convey, S. Lee, M. Stone-
braker, N. Tatbul, and S. Zdonik. Aurora: A new model and architecture
for data stream management.The VLDB Journal, 12(2):120–139, 2003. doi:
10.1007/s00778-003-0095-z. URL https://doi.org/10.1007/s00778-003-0095-z.
[2]S. Arora, B. Yang, S. Eyuboglu, A. Narayan, A. Hojel, I. Trummer, and C. Ré.
Language models enable simple systems for generating structured views of
12

heterogeneous data lakes.Proceedings of the VLDB Endowment, 17(P92), 2024. doi:
10.14778/3626292.3626294. URL https://www.vldb.org/pvldb/vol17/p92-arora.
pdf.
[3]P. Carbone, A. Katsifodimos, S. Ewen, V. Markl, S. Haridi, and K. Tzoumas.
Apache Flink™: Stream and batch processing in a single engine.Bulletin of the
IEEE Computer Society Technical Committee on Data Engineering, 38(4):28–38,
2015.
[4] M. Charikar, C. Chekuri, T. Feder, and R. Motwani. Incremental clustering and
dynamic information retrieval.SIAM Journal on Discrete Mathematics, 17(2):
237–258, 2003.
[5]Z. Cheng, J. Kasai, and T. Yu. Batch prompting: Efficient inference with large
language model APIs. InProceedings of the 2023 Conference on Empirical Methods
in Natural Language Processing: Industry Track, pages 792–810, Singapore, 2023.
Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-industry.
74. URL https://aclanthology.org/2023.emnlp-industry.74/.
[6] K. Community. Twitter entity sentiment analysis dataset. https://www.kaggle.
com/datasets/jp797498e/twitter-entity-sentiment-analysis, 2024.
[7] Z. Dong, X. Fan, and Z. Peng. Fnspid: A comprehensive financial news dataset
in time series. InProceedings of the 30th ACM SIGKDD Conference on Knowledge
Discovery and Data Mining (KDD ’24), pages 4918–4927, 2024. doi: 10.1145/
3637528.3671629.
[8] O. Khattab, A. Singhvi, P. Maheshwari, Z. Zhang, K. Santhanam, S. Vardhamanan,
S. Haq, A. Sharma, T. T. Joshi, H. Moazam, et al. Dspy: Compiling declarative lan-
guage model calls into self-improving pipelines.arXiv preprint arXiv:2310.03714,
2023.
[9] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang,
and I. Stoica. Efficient memory management for large language model serving
with pagedattention. InProceedings of the 27th ACM SIGOPS Symposium on
Operating Systems Principles (SOSP) ’23, pages 506–522, 2023. doi: 10.1145/
3600006.3613165.
[10] J. Lin, M. Diesendruck, L. Du, and R. Abraham. Batchprompt: Accomplish more
with less. InInternational Conference on Learning Representations (ICLR 2024).
OpenReview.net, 2024. URL https://proceedings.iclr.cc/paper_files/paper/2024/
file/5d8c01de2dc698c54201c1c7d0b86974-Paper-Conference.pdf.
[11] Y. Lin, M. Hulsebos, R. Ma, S. Shankar, S. Zeighami, A. G. Parameswaran, and
E. Wu. Querying templatized document collections with large language models.
In41st IEEE International Conference on Data Engineering (ICDE 2025), pages
2422–2435. IEEE, 2025. doi: 10.1109/ICDE65448.2025.00183. URL https://doi.org/
10.1109/ICDE65448.2025.00183.
[12] C. Liu, M. Russo, M. Cafarella, L. Cao, P. B. Chen, Z. Chen, M. Franklin, T. Kraska,
S. Madden, R. Shahout, and G. Vitagliano. Palimpzest: Optimizing ai-powered
analytics with declarative query processing. InProceedings of the 15th Conference
on Innovative Data Systems Research (CIDR), 2025.
[13] S. Liu, J. Xu, W. Tjangnaka, S. J. Semnani, C. Yu, and M. S. Lam. Suql: Con-
versational search over structured and unstructured data with large language
models. InFindings of the Association for Computational Linguistics: NAACL 2024,
pages 4535–4555. Association for Computational Linguistics, June 2024. doi:
10.18653/v1/2024.findings-naacl.283. URL https://aclanthology.org/2024.findings-
naacl.283/.
[14] D. Lu, S. Feng, J. Zhou, F. Solleza, M. Schwarzkopf, and U. Çetintemel. Vectraflow:
Integrating vectors into stream processing. InProceedings of the 15th Annual
Conference on Innovative Data Systems Research (CIDR ’25), 2025.
[15] J. McAuley and J. Leskovec. From amateurs to connoisseurs: Modeling the
evolution of user expertise through online reviews. InProceedings of the 22nd
International Conference on World Wide Web (WWW ’13), pages 897–908. ACM,
2013. doi: 10.1145/2488388.2488477. URL https://snap.stanford.edu/data/web-
FineFoods.html.
[16] R. Motwani, J. Widom, A. Arasu, B. Babcock, S. Babu, M. Datar, G. S. Manku,
C. Olston, J. Rosenstein, and R. Varma. Query processing, approximation, and
resource management in a data stream management system. InProceedings of
the First Biennial Conference on Innovative Data Systems Research (CIDR), 2003.
[17] L. Patel, S. Jha, M. Pan, H. Gupta, P. Asawa, C. Guestrin, and M. Zaharia. Semantic
operators and their optimization: Enabling LLM-powered analytics.Proceedings
of the VLDB Endowment (PVLDB), 18(3):4171–4184, 2025. doi: 10.14778/3749646.
3749685.
[18] M. Russo, S. Sudhir, G. Vitagliano, C. Liu, T. Kraska, S. Madden, and M. Cafarella.
Abacus: A cost-based optimizer for semantic operator systems.arXiv preprint
arXiv:2505.14661, 2025. URL https://arxiv.org/abs/2505.14661.
[19] S. Shankar, T. Chambers, T. Shah, A. G. Parameswaran, and E. Wu. Docetl: Agentic
query rewriting and evaluation for complex document processing.Proceedings
of the VLDB Endowment, 18(9), 2025. doi: 10.14778/3746405.3746426.
[20] The Apache Software Foundation. Apache storm. https://storm.apache.org/.
Accessed: 2024-08-02.
[21] C. Toraman, O. Ozcelik, F. Sahinuç, and F. Can. MiDe22: An annotated multi-
event tweet dataset for misinformation detection. InProceedings of the 2024 Joint
International Conference on Computational Linguistics, Language Resources and
Evaluation (LREC-COLING 2024), pages 11283–11295. ELRA and ICCL, 2024. URL
https://aclanthology.org/2024.lrec-main.986.[22] U. Çetintemel, S. Chen, A. W. Lee, and D. Raghavan. Making prompts first-class
citizens for adaptive llm pipelines. InProceedings of the 15th Annual Conference
on Innovative Data Systems Research (CIDR’25), 2025. URL https://arxiv.org/abs/
2508.05012. To appear.
13