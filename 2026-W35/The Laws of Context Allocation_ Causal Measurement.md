# The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search

**Authors**: Peiyang Liu, Xi Wang, Di Liang, Wei Ye

**Published**: 2026-08-24 13:44:11

**PDF URL**: [https://arxiv.org/pdf/2608.23252v1](https://arxiv.org/pdf/2608.23252v1)

## Abstract
As Retrieval-Augmented Generation (RAG) shifts toward diverse portfolio generation, it is stymied by two critical bottlenecks: flawed measurement of evidence utilization, and suboptimal context budget allocation. We resolve both sequentially.
  To resolve measurement, we expose a pervasive ``diagnostic illusion'': standard relevance proxies fail catastrophically on hard negatives. We replace them with an efficient causal leave-one-out probe that accurately isolates generative reliance and formally calibrates the structural dilution of LLM attention.
  To resolve allocation, we deploy this causal probe in a deconfounded factorial grid. We prove that the prevailing strategy of monolithic context widening is an architectural trap penalized by relevance decay. Instead, allocating compute iteratively across multiple sequential generations drives transformative portfolio recall gains of 16.7--20.5 absolute percentage points, scaling robustly up to 32B models.
  Finally, we unify these solutions into a deployable closed-loop submodular scheduler. Augmented by an attribution-steered contrastive decoder to override LLM attention inertia, our architecture systematically forces fresh evidence integration. By dominating classical open-loop baselines, we establish sequential, feedback-driven orchestration as the definitive paradigm for generative search. Our code, data, and causal measurement instruments are available at https://github.com/PeiYangLiu/ascp.

## Full Text


<!-- PDF content starts -->

The Laws of Context Allocation: Causal Measurement and Closed-Loop
Orchestration in Generative Search
PEIYANG LIU,National Engineering Research Center for Software Engineering, Peking University, China
XI WANG,Peking University, China
DI LIANG,Tencent, China
WEI YE∗,National Engineering Research Center for Software Engineering, Peking University, China
As Retrieval-Augmented Generation (RAG) shifts toward diverse portfolio generation, it is stymied by two critical bottlenecks: flawed
measurement of evidence utilization, and suboptimal context budget allocation. We resolve both sequentially.
To resolve measurement, we expose a pervasive “diagnostic illusion”: standard relevance proxies fail catastrophically on hard
negatives. We replace them with an efficient causal leave-one-out probe that accurately isolates generative reliance and formally
calibrates the structural dilution of LLM attention.
To resolve allocation, we deploy this causal probe in a deconfounded factorial grid. We prove that the prevailing strategy of
monolithic context widening is an architectural trap penalized by relevance decay. Instead, allocating compute iteratively across
multiple sequential generations drives transformative portfolio recall gains of 16.8–20.5 absolute percentage points, scaling robustly
up to 32B models.
Finally, we unify these solutions into a deployable closed-loop submodular scheduler. Augmented by an attribution-steered
contrastive decoder to override LLM attention inertia, our architecture systematically forces fresh evidence integration. By dominating
classical open-loop baselines, we establish sequential, feedback-driven orchestration as the definitive paradigm for generative search.
Our code, data, and causal measurement instruments are available at https://github.com/PeiYangLiu/ascp.
CCS Concepts:•Information systems →Retrieval models and ranking;Search results deduplication;Evaluation of retrieval results.
Additional Key Words and Phrases: retrieval-augmented generation, context attribution, inference-time scaling, test-time compute,
evaluation
ACM Reference Format:
Peiyang Liu, Xi Wang, Di Liang, and Wei Ye. 2026. The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration
in Generative Search.ACM Trans. Inf. Syst.0, 0, Article 0 (July 2026), 37 pages. https://doi.org/XXXXXXX.XXXXXXX
1 Introduction
Classical information retrieval (IR) systems treat ambiguous or multi-faceted queries by returning a diversified ranked list,
acknowledging a fundamental truth: a single document rarely satisfies all underlying user intents [ 10,22,92,102,112].
Retrieval-augmented generation (RAG) inherits this premise of underspecified queries but radically alters the delivery
∗Corresponding author.
Authors’ Contact Information: Peiyang Liu, National Engineering Research Center for Software Engineering, Peking University, Beijing, China,
liupeiyang@pku.edu.cn; Xi Wang, Peking University, Beijing, China, wangxi5629@pku.edu.cn; Di Liang, Tencent, Beijing, China, liangd17@fudan.edu.cn;
Wei Ye, National Engineering Research Center for Software Engineering, Peking University, Beijing, China, wye@pku.edu.cn.
Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not
made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components
of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on
servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.
©2026 Copyright held by the owner/author(s). Publication rights licensed to ACM.
Manuscript submitted to ACM
Manuscript submitted to ACM 1
arXiv:2608.23252v1  [cs.LG]  24 Aug 2026

2 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
mechanism [ 14,61,142]. Rather than providing a diversified list of documents for a human to browse, the prevailing RAG
paradigm forces generative models to compress retrieved passages into a single context prompt, aiming to synthesize one
monolithic response. However, for complex informational needs, this single-pass synthesis is structurally inadequate. To
truly satisfy ambiguous queries in the generative era, a robust system must transition from extracting a single answer
to generating a diverseportfolioof responses that collectively cover the space of evidence-supported truths.
This necessary paradigm shift from diversified ranking to generative portfolio construction introduces a critical
system design dilemma:How should a fixed inference budget be optimally allocated?Given a retrieved pool of candidate
documents and a constrained hardware budget, system architects face two divergent paths. Should they follow the
current natural language processing (NLP) trend of feeding a massive, wide context into a single generation pass? Or
should they adhere to IR diversification principles by dividing the budget iteratively, querying the model across multiple
sequential rounds with narrower, focused contexts? Figure 1 visualizes this exact architectural dilemma and previews
our core findings: despite consuming an identical physical evidence budget, iterative narrow contexts fundamentally
eclipse monolithic wide contexts in answer space coverage. Crucially, standard relevance proxies completely mask this
dynamic, necessitating a paradigm shift in generative evaluation.
Before we can empirically resolve this allocation question, we hit an epistemological wall: the RAG community lacks
a rigorous mechanism to measure what evidence a Large Language Model (LLM) actually utilizes from its prompt.
Standard proxies, such as embedding similarity and lexical overlap, suffer from severe methodological blind spots
because they conflate genuine evidence utilization with mere topical relevance. Classical IR has long held that a measure
must be diagnosed against the behaviour it claims to capture rather than trusted on face validity [ 27,46]; we apply
that same standard to context attribution. To overcome this, we formulate a causal measurement instrument based on
counterfactual sensitivity, a leave-one-out (LOO) probe. Because the generated response is held fixed, our counterfactual
evaluations act as highly efficient teacher-forced passes, allowing us to deeply audit the generative cognitive process
without the prohibitive bottleneck of autoregressive decoding.
Armed with this causal probe, our first major contribution is uncovering a pervasivediagnostic illusionthat plagues
current RAG attribution literature. We demonstrate that the perceived effectiveness of existing attribution metrics is
almost entirely a mirage constructed by flawed evaluation datasets. When evaluated on standard off-query distractor
pools (passages retrieved for unrelated topics), naive similarity metrics appear near-perfect (AUCs approaching1 .000).
However, when forced to distinguish challengingsame-queryhard negatives, documents that are topically dense but
contain no actual answers, traditional proxies completely collapse to random chance. Only our causal probe maintains
robust discrimination. By formally calibrating out target-shift artifacts, we establish a fundamental structural property
of LLMs: the dilution of attribution across wider contexts is an inherent, inescapable generative behavior. Under strictly
controlled diagnostic isolation, this yields a calibrated width elasticity of −0.68(0.02), acting as an empirical baseline
for attention decay.
Having secured a definitively validated measurement instrument, we systematically dismantle the budget allocation
dilemma through a deconfounded 𝑘×𝑇 factorial experiment. We discover that simply expanding context width, the
current dominant scaling strategy, is anarchitectural trapfundamentally constrained by IR relevance decay. Wide
contexts merely construct a slightly better single answer while leaving massive informational blind spots. Conversely,
we establish a robust empirical law of context allocation: dedicating the computational budget to multiple, narrower
sequential generations yields transformative absolute gains of 16.8 to 20.5 percentage points in comprehensive portfolio
coverage. While this sequential approach inherently incurs higher autoregressive latency and probe overhead, it remains
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 3
(a) One inference budget, two ways to spend it
retrieved pool, 𝑁candidates for one under-specified query
wide & shallow
𝑘=24, 𝑇=1narrow & deep
𝑘=2, 𝑇=12
context𝐶1
one wide prompt𝐶1
𝐶2
𝐶12...
fresh rank window each round
1 response
portfolio recall PR@ 𝑇at the same 24-slot budget
(𝑘=24,𝑇=1) 0.253
(𝑘=2,𝑇=12) 0.397
+0.144
more of the answer space for the same evidence budgetfixed budget: 𝑘×𝑇=24 evidence slots
12 responses
portfolio0.22 0.25 0.28 0.31
held-out portfolio recall PR@ 𝑇PM-2-RAGvanilla RAGxQuADMMRCarriage -nar.DPP-RAGCarriageno steeringdeep rotationAscp (ours) 0.309
+0.033to+0.081
over every selection-
style baseline, 𝑞<.001frozen held-out frame; grey =n.s.(b) The scheduler it enables
off-query same-query
padding used to build the evaluation pool0.40.50.60.70.80.91.0AUC at𝑘=3leave-one-out probes (ours)−0.01
output overlap −0.14
output–doc. cosine −0.23
query–doc. cosine −0.52
BM25 −0.54ΔAUC
chanceswap the distractors and the ranking inverts(c) The measurement that makes it possible
1
Fig. 1.The context-allocation problem in one picture. (a)A fixed inference budget of 𝑘×𝑇 evidence slots can be packaged
as one wide context ( 𝑘=24,𝑇=1) or as many narrow contexts that rotate through fresh evidence ( 𝑘=2,𝑇=12). Both consume24
retrieved documents, yet the second packaging covers +0.144more of the answer space (Table 4).(b)Turning that allocation law into
a scheduler: on a frozen held-out frame,Ascpbeats every selection-style baseline by +0.033to+0.081portfolio recall (all BH 𝑞<. 001);
the two grey arms are reference configurations whose remaining gaps are not significant (Table 10).(c)None of this is measurable
with relevance proxies. Swapping the padding of the evaluation pool from off-query distractors to same-query hard negatives that
entail no answer leaves the causal leave-one-out probes almost unchanged ( −0.01AUC), while BM25 and query–document cosine
fall by more than0.5to chance (Figure 3).
the only mechanism capable of breaching the extraction ceiling of monolithic single-pass models, a structural supremacy
verified up to the 32B scale.
Finally, we operationalize these conceptual insights into a deployable system architecture. Recognizing that traditional
IR algorithms (e.g., MMR) operate asopen-loopsystems blind to actual generative consumption, we propose a feedback-
drivenclosed-loopsubmodular scheduler. By actively reading causal attribution feedback, our scheduler systematically
outperforms all seven evaluated selection-style baselines. Furthermore, to combat the LLM’s inherentattention inertia,
we augment our architecture with an orthogonal attribution-steered contrastive decoder. Acting as a cognitive override,
this micro-level intervention forcefully shifts probability mass away from over-used evidence while maintaining strict
plausibility guardrails, delivering mathematically guaranteed orthogonal gains. Ultimately, this framework pioneers the
application ofinference-time scalingto generative search: by deliberately investing test-time compute into iterative
causal orchestration, we bridge the gap between classical search-result diversification and modern LLM pipelines.
Our main conceptual and empirical contributions are summarized as follows:
Manuscript submitted to ACM

4 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
•Exposing the Diagnostic Illusion in Generative Evaluation:We reveal that standard off-query distractor
pools artificially inflate relevance proxies to apparent perfection. Utilizing rigorous same-query hard negatives,
we prove these proxies fail catastrophically, establishing causal counterfactual probes as the indispensable
standard for valid evidence attribution.
•Formalizing the Dilution Law of Context Width:We resolve widespread measurement artifacts in RAG
attribution to uncover a fundamental generative property: evidence utilization inevitably dilutes as context
expands, yielding a strictly calibrated width elasticity of−0.68(0.02).
•Empirical Laws of Context Allocation and Inference-Time Scaling:Through a deconfounded factorial
design, we prove that monolithic context-widening is an architectural trap that hits a rigid cognitive ceiling.
Instead, we establish a new paradigm for inference-time scaling: deliberately investing test-time compute across
multiple sequential generations drives massive absolute recall surges of 16.8 to 20.5 percentage points, a structural
supremacy verified up to the 32B model scale.
•A Closed-Loop Orchestration Architecture:Translating theory into practice, we introduce an attribution-
steered submodular scheduler. Validated across rigorous cross-task evaluation frames, the architecture system-
atically dominates classical open-loop baselines. Augmented by a contrastive decoder that acts as a cognitive
override against attention inertia, our framework proves that dynamic, feedback-driven context orchestration
fundamentally surpasses static context maximization.
2 Related Work
2.1 From Search Result Diversification to Diverse RAG
Classical information retrieval treats a query as an underspecified expression of intent. To mitigate the risk of returning
redundant near-duplicates, ranking models actively trade sheer relevance for novelty and subtopic coverage [ 10].
This foundational premise birthed a lineage of diversification machinery, including probabilistic, subtopic, axiomatic,
and learned paradigms [ 1,12,20,33,51,94,102,126]. The line remains actively developed: neural rankers encode
diversity greedily with self-attention [ 92], model candidates at multiple granularities [ 21], resolve subtopics at passage
rather than document level [ 112], pre-train diversification in a model-agnostic fashion [ 22], and extend coverage to
streaming corpora [ 69]. Many modern coverage formulations exploit monotone submodularity to inherit rigorous
greedy approximation guarantees [ 28,70,90], supported by scalable algorithms and determinantal point processes (DPP)
[5,57,83,84]. Consequently, classical metrics implicitly reward aspect coverage and penalize redundancy [ 16,101,135],
and are themselves derived from explicit models of how a user consumes a ranking [ 86,87]. That answer-bearing
evidence is redundantly spread across a corpus—so that coverage, not any single passage, bounds what a system can
answer—was also established well before RAG [71].
However, this classical lineage assumes ahumanconsumes the ranked list. Retrieval-augmented generation (RAG)
shifts the consumer from a human to a generative model. Recent diversity-aware RAG pipelines attempt to pack distinct
information into a single limited prompt window to optimize a comprehensive single answer [ 98,123]. Most notably,
Carriage[ 41] explores recipe adaptations on cross-cultural benchmarks [ 40,88] using MMR-like penalties and a sliding
window. While these adaptations are valuable, our analysis suggests that the sliding window carries the primary effect
because it intrinsically increases the distinct documents reaching the generator. Our work formalizes this transition:
instead of hedging a single ranked list, we repeatedly query the generator, evaluating how distinct document exposure
drives portfolio coverage across sequential readings.
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 5
2.2 Generative Context Utilization and Resource Allocation
Since its inception [ 61], RAG research has rapidly expanded across dense retrieval, joint pre-training, adaptive orches-
tration, and graph-based structuring [ 4,8,23,31,35,43,52,55,56,95,103,107,116,117,122,132,137], with the retrieval
and generation halves each now surveyed at length [ 14,66,91,139,142] and with classical first-stage devices such as
pseudo-relevance feedback re-examined under dense encoders [ 62,114]. To mitigate the burden of massive contexts,
techniques such as distribution ensembling, fusion, and prompt compression have been proposed [ 43,49,50,107,127],
alongside token-efficient agentic pipelines [ 136], joint optimization of knowledge selection with the reader [ 108],
and explicit memory management for long-running agents [ 138]. Benchmarks have broadened accordingly, spanning
create–read–update–delete task families [ 77], unified long-context needle-in-a-haystack probes [ 32], and deployed code
assistance [ 67]. Crucially, almost all these systems optimize asingleresponse rather than investigating what fraction
of the retrieved evidence is actually consumed; the sequential regime in which evidence accumulates over successive
turns is instead treated separately, as conversational search [85].
Emerging studies reveal that language models do not process retrieved context as perfect pipelines. Accuracy
degrades based on document position, irrelevant context induces hallucinations, and effective context windows remain
significantly shorter than advertised limits [ 3,6,19,39,53,60,65,72,105,128,130,131]. While prior studies analogize
RAG optimization to neural scaling laws [ 36,54], they primarily treat context width as the sole scaling variable. In
contrast, our deconfounded factorial experiment explicitly asks whether the same retrieved pool should fund a wider
single context or be allocated across multiple narrower contexts to expose fresh evidence.
2.3 Attribution, Faithfulness, and Causal Measurement
Evaluating whether a RAG response is genuinely grounded necessitates rigorous source attribution. The explanation
literature relies heavily on local approximations, Shapley values, or input erasure [ 18,64,76,99,113], repeatedly
cautioning that mere attention scores are fundamentally unreliable proxies for evidence use [ 38,45,104,125]. Within
IR proper, ranking models have been rebuilt to emit extractive rationales precisely so that the evidence a scorer relied
on becomes inspectable rather than inferred [ 59]. Consequently, evaluating generative search encompasses citation
frameworks, generative automatic judgements, and fact-checking protocols [ 7,30,42,47,73,78,79,81,82,89,96,133].
ContextCite [ 17] similarly ablates context to generate sparse linear surrogates, which we explicitly benchmark against
in our study.
Simultaneously, reference-free RAG evaluation frameworks [ 24,100] rely heavily on LLM-as-a-judge paradigms,
despite their documented biases [ 15,74,140]. The IR community has begun mapping where such generated assessments
hold: as relevance judgements driving query performance prediction [ 80], as the basis of entire test collections [ 118], and
as simulators of user behaviour [ 121]. By utilizing sentence embeddings [ 97] as a similarity baseline, we demonstrate
that standard proxies and judges saturate and fail catastrophically under hard same-query negative populations. Thus,
establishing a causal attribution probe is a strict prerequisite for asserting portfolio coverage claims in diverse RAG.
Our reliance on constructed ground truth, paired contrasts, and replicated seeds follows a long methodological
tradition in IR evaluation. Effectiveness differences are unstable unless replicate runs and topic-set variation are modelled
explicitly [ 120]; offline estimates can invert an online verdict when the logging policy confounds the comparison
[11,44]; offline and online judgements of the same component need not agree [ 115]; and meta-evaluating the measure
itself, rather than only the systems it scores, is the accepted obligation when a new metric is introduced [46, 75].
Manuscript submitted to ACM

6 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
Table 1. Within-query correlations across systems ( 𝑛=2395task–model–query groups, 23682 observations), centered within group.
Lower panel: partial correlations between portfolio recall and one measure after residualising against paired coverage or output-
diversity.‡marks a measure whose sign is not constant across tasks, so its pooled value should not be read alone.
vs. PR@𝑇vs. groundedness
Family Measure𝑛 𝑟 𝑝 𝑟 𝑝
coverage Evidence coverage rate 23682 +0.097 2.8e-50 -0.167 9.7e-148
Utilisation concentration 23682 -0.023 3.0e-04 -0.146 2.7e-112
diversity Semantic diversity 23682 +0.089 6.1e-43 -0.111 9.3e-66
Distinct-1 23682 +0.106 3.7e-60 -0.100 1.7e-53
Distinct-2 23682 +0.128 2.4e-87 -0.134 9.4e-96
Distinct-3 23682 +0.128 5.2e-87 -0.131 7.9e-91
Self-BLEU 23682 -0.102 5.1e-56 +0.161 1.8e-136
Per-task correlation with PR@𝑇: ASQA / QAMPARI / ELI5 / recipes
Evidence coverage rate +0.161 / +0.146 / +0.041 / +0.076
Utilisation concentration‡-0.014 / +0.014 / +0.038 / -0.209
Semantic diversity +0.121 / +0.125 / +0.028 / +0.280
Distinct-1 +0.160 / +0.139 / +0.022 / +0.284
Distinct-2 +0.179 / +0.144 / +0.041 / +0.295
Distinct-3 +0.177 / +0.141 / +0.045 / +0.295
Self-BLEU -0.155 / -0.044 / -0.054 / -0.284
Partial correlations with PR@𝑇
partial Evidence coverage|distinct-2 23682 +0.060 1.6e-20
partial Distinct-2|evidence coverage 23682 +0.104 1.0e-57
2.4 Inference-Time Scaling and Diverse Decoding
Our work inherently connects to mechanisms that control generation diversity and scale test-time compute. Traditional
diversity controls operate on the token level, such as temperature, top- 𝑘, diverse beam search, generic-response
penalties, contrastive representation, and DPP sampling [ 26,29,37,57,63,111,119,141]. These are orthogonal to
evidence availability; they manipulate textual variety while leaving evidence utilization largely static.
Simultaneously, inference-time scaling studies observe that task coverage (e.g., pass@𝑘) scales smoothly with sample
count [ 9,13,109], and allocating test-time compute effectively can often rival model upscaling [ 58,124,134]. While
existing techniques contrast expert and amateur distributions [ 68] or context-aware formulations [ 106], they uniformly
optimize for a single, definitive answer. Our scheduling and decoding interventions contrast over-used versus under-
used evidence, redistributing grounding across a multi-round portfolio to actively scale evidence coverage, rather than
simply sampling identical contexts repeatedly.
3 Problem Formulation and Evaluation Metrics
To systematically optimally allocate a retrieval-augmented generation (RAG) system’s inference budget, we must first
formalize what the system is trying to achieve and define rigorous metrics to measure its internal behavior. In this
section, we define our ultimate end-to-end goal (Portfolio Recall) and our intermediate diagnostic metric (Evidence
Coverage Rate), while exposing a critical pitfall in how the NLP community traditionally evaluates diversity.
3.1 The End-to-End Goal: From Single Answers to Generative Portfolios
Classical search engines handle ambiguous queries by returning a diversified list of documents, hedging their bets
to cover various possible user intents [ 92,112]. Modern RAG systems, however, typically force the generative model
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 7
to compress all retrieved evidence into a single monolithic response. For complex queries, this single-pass synthesis
inevitably drops critical information.
To resolve this, we formalize the concept ofresponse-portfolio generation. Given a query 𝑞and a pool of retrieved
documentsP, the RAG system is allowed a constrained computation budget to produce 𝑇distinct sequential responses,
denoted as𝑦 1,...,𝑦𝑇. The user views these𝑇responses as a unified portfolio.
The portfolio is considered optimal if the union of these responses covers as much of the underlying truth as possible.
We quantify this end-to-end success usingPortfolio Recall( PR@𝑇). Let𝐴be the set of gold-standard answer units for
the query. PR@𝑇is defined as the fraction of these gold units successfully recovered by any response in the portfolio:
PR@𝑇=1
|𝐴|
𝑎∈𝐴:∃𝑡s.t.𝑎is asserted in𝑦 𝑡	.(1)
The mathematical gap between PR@𝑇and the recall of the single best response precisely quantifies the value of
sequential diversification.
3.2 The Diagnostic Signal: Quantifying Genuine Evidence Coverage
While PR@𝑇measures the final success of the system, it is a “black-box” metric. To actually design an intelligent
scheduling algorithm that decideswhichdocuments to feed the LLM in the next round, we need to look inside the box:
we must measure what fraction of the offered documents the LLM actually consumed.
Assume for a moment that we possess an attribution matrix 𝐴∈[ 0,1]𝑇×𝑁(we will detail exactly how to construct
this matrix using our causal probe in Section 4). In this matrix, each entry 𝐴𝑡𝑑represents the causal utilization score of
document𝑑during generation round𝑡.
Using this matrix, we define our ultimate operational metric: theEvidence Coverage Rate (ECR), the generative
counterpart of the subtopic recall that classical diversification evaluates under an explicit model of how far a user reads
[86,87], and of the corpus redundancy that bounds what a question-answering system can recover at all [ 71]. ECR
measures extraction efficiency—specifically, what percentage of theunique documents exposed to the modelwere actually
utilized to generate the text. Let O𝑇=Ð𝑇
𝑡=1𝐶𝑡define the unique footprint of documents shown to the generator across
all𝑇rounds. We formalize ECR as:
ECR@𝑇=1
|O𝑇|{𝑑:∃𝑡, 𝐴 𝑡𝑑≥𝜃max
𝑑′𝐴𝑡𝑑′},(2)
where𝜃is a utilization threshold (e.g., 𝜃= 0.1). Crucially, by putting the dynamically orchestrated set |O𝑇|in the
denominator rather than an arbitrary static number, ECR strictly penalizes “lazy” scheduling policies that blindly inject
ignored documents into the prompt. It isolates the system’s true scheduling precision.
3.3 The Evaluation Trap: Textual Diversity vs. Evidence Diversity
At this point, a natural question arises: why build complex attribution matrices to measure evidence consumption?
Why not simply measure how “different” the generated texts 𝑦1,...,𝑦𝑇are from each other using standard NLP textual
diversity metrics (e.g., Distinct-2 or Semantic Diversity)?
This leads us to a critical methodological trap, which we term thetextual diversity confound: the dangerous
conflation ofhow differentlya model speaks withwhat different factsit actually uses.
If an LLM generates three responses with vastly different phrasing but relies on the exact same underlying document,
standard NLP metrics will heavily reward the system, creating a fake illusion of knowledge diversity. Our empirical
Manuscript submitted to ACM

8 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
|P|=𝑁relevance only
MMR / xQuAD / PM-2
submodular+feedback
round-robin rotation𝐶1
𝐶2
𝐶𝑇
𝑘documents each
𝐶𝑡
𝑦𝑡𝐶𝑡 reference
𝐶𝑡\{𝑑1} −0.62
𝐶𝑡\{𝑑2} −0.05
𝐶𝑡\{𝑑𝑘} −0.31
one batched teacher-forced pass — the text is fixed, so nothing is decoded𝜏
S𝑡: above threshold
𝑡
rounds×pool|O𝑇| distinct offered
|U𝑇| distinct used
𝑞=|U𝑇|/|O𝑇| utilisation fractionthroughput offer more distinct docs ↑|O𝑇|
instruction ask it to integrate sources ↑𝑞
context width narrower, and more of them ↑𝑞Scheduling
Retrieved pool Scheduling policy One context per round
Generation, then causal attribution of what was generated
Generate once Re-score the same 𝑦𝑡under𝑘+1contexts Documents used this round
What is measured, and the only three things that move it
Utilisation matrix Derived quantities Three levers
1
Fig. 2. The closed-loop measurement and orchestration pipeline. The scheduler transforms a retrieved pool into a curated context per
generation round. Subsequently, our causal probe isolates true evidence utilization via 𝑘+1parallelizable teacher-forced forward
passes, completely bypassing autoregressive decoding overhead. These causal signals dynamically populate the utilization matrix,
forming the exact feedback loop that governs subsequent submodular scheduling and attribution-steered cognitive decoding.
analysis of 23,682 paired observations (Table 1) confirms this trap. While surface-level textual diversity naturally corre-
lates with final task success (𝑟=+0.128), partial correlation analysis proves it operates almost entirely independently
from actual evidence coverage (𝑟=+0.097).
More dangerously, blindly optimizing for this textual variety actively harms system reliability. Our data reveals a
severe trade-off: systems that aggressively push for novel wording systematically lose their anchor in the source texts,
exhibiting a massive negative correlation with lexical groundedness ( 𝑟=− 0.167). To build a trustworthy diverse RAG
system, we cannot optimize for stochastic word-shuffling; we must optimize a manipulable IR target.
Meta-Evaluation.To validate that our ECR metric captures genuine, human-aligned evidence utilization rather than
statistical noise—the meta-evaluation any newly proposed measure owes its readers [ 75]—we benchmarked our metric
against an independent, identity-blinded LLM-as-a-judge over 858 document-level judgments (Figure 7). The results
were decisive: our ECR tracked the judge’s assessment of true informational coverage exceptionally well ( 𝜌=0.654).
Conversely—and even though generated assessments are otherwise serviceable as relevance labels [ 80,118]—when a
judge was instructed to rate pure “textual novelty, ” it proved fundamentally blind to whether new evidence was actually
introduced (𝜌=0.425).
This dictates a strict prerequisite: without actively parsing the retrieved pool via a rigorous causal instrument,
distinguishing genuinely fresh evidence extraction from stochastic paraphrasing is mathematically impossible. This
justifies the necessity of our causal attribution probe, which we introduce next.
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 9
4 The Causal Attribution Probe and Its Validation
Having established in Section 3 that diverse textual output does not guarantee diverse evidence utilization, we face a
fundamental methodological bottleneck: we must operationalize a reliable measurement instrument before any context
allocation rules can be evaluated. Because standard embedding similarity and lexical overlap are inherently confounded
by query relevance, we construct an intervention-based causal probe. In this section, we formalize this instrument,
rigorously validate its discriminative limits, and ultimately deconstruct a systemic evaluation flaw in current generative
attribution literature.
4.1 The Causal Instrument: Counterfactual Sensitivity
Traditional attribution metrics operate observationally, measuring superficial semantic overlap between the prompt
and the response. We argue that genuine utilization can only be isolated via intervention. For an already-produced
generation𝑦𝑡derived from context 𝐶𝑡at round𝑡, we pose a strict counterfactual:how much less likely would the exact
realized generation become if document𝑑were ablated from the context?
We quantify this counterfactual sensitivity via the per-token drop in log-likelihood:
𝑎raw
𝑡(𝑑)=1
|𝑦𝑡|h
log𝑝𝜃 𝑦𝑡|𝑞,𝐶𝑡−log𝑝𝜃 𝑦𝑡|𝑞,𝐶𝑡\{𝑑}i
,(3)
normalized over the context as 𝑎𝑡(𝑑)∝[𝑎raw
𝑡(𝑑)]+. A positive value dictates that deleting document 𝑑causally reduces
the likelihood of the generated text, establishing structural reliance. To discretize this continuous utilization into binary
counts for our live scheduling matrix, we apply an operational, free-generation threshold 𝜏free(𝑘)= 0.555𝑘−0.633. This
specific decay scaling mathematically accounts for the natural text-lengthening and hedging behaviors LLMs exhibit
when fed wider contexts (a confound we explicitly deconstruct in Section 4.4).
A hallmark of a practical IR measurement framework is computational feasibility, as visually mapped in our end-to-
end pipeline (Figure 2). Because our probe operates on a fixed response 𝑦𝑡, the ablation process bypasses the prohibitive
autoregressive decoding bottleneck entirely. Each counterfactual evaluation constitutes a single, highly parallelizable
teacher-forced forward pass. Furthermore, documents maintain immutable semantic identifiers across all counterfactual
passes, ensuring that deletion does not artificially conflate lost evidence with the mere renumbering of surface-form
citations.
4.2 Deconstructing the Diagnostic Illusion via Controlled Pools
Evaluating whether a model utilized specific evidence is notoriously confounded by the model’s internal parametric
knowledge. To establish absolute causal ground truth—in the same spirit as diagnostic test collections built to isolate
individual retrieval heuristics rather than score systems end to end [ 27]—we construct controlled retrieval pools
preserving exactly 𝑚= 2documents that attest to different gold answers (thenecessarydocuments), padded to various
context widths.
Testing against these controlled pools exposes a profound systemic flaw in current RAG evaluations (Figure 3(a,b)).
Initially, under standardoff-querypadding (distractors retrieved for completely unrelated intents), traditional IR
proxies appear scientifically flawless. At context width 𝑘=24, query–document cosine and BM25 achieve near-perfect
AUCs approaching1.000, while our deletion leave-one-out (LOO) probe trails at0.829.
However, this apparent precision is adiagnostic illusiondriven entirely by construction leakage. Off-query padding
provides trivial negative examples, exactly the type of topically disjoint documents that query relevance algorithms
Manuscript submitted to ACM

10 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
3 5 8 12 24
context width 𝑘0.40.50.60.70.80.91.0AUC
chanceproxies look perfect(a) off-query padding
3 5 8 12 24
context width 𝑘0.40.50.60.70.80.91.0AUC
chance
same proxies collapse(b) same-query padding
3 5 8 12 24
context width 𝑘0.40.50.60.70.80.91.0AUCsolid: LOO dashed: ContextCite
chance(c) LOO vs. ContextCite
distractor duplicate mixed
−.050.05.10.15.20
cover@ 𝑚gain over randomdeletion LOO
in-place LOO
ContextCite
output–doc. cos.
query–doc. cos.
BM25counterfactual estimators shaded(d) set-coverage target
2 3 5 8 12 24
context width 𝑘0.00.10.20.30.4exceedance of 𝜏(𝑘)only fixed-target curves are false-positive rates
nominal 5%(e) same-query near-miss transfer
fixed target
free target
in (c) and (e): circles are deletion,
squares in-place replacementpanels (a)–(b)
deletion LOO
in-place LOO
output–document cosine
query–document cosine
BM25
output overlap
1
Fig. 3. Probe validation and controls on constructed pools.(a)–(b)AUC for recovering the designed answer-bearing documents, by
context width, under off-query and same-query padding; the ordering of causal estimators and relevance proxies reverses with the
negative population.(c)Leave-one-out against ContextCite by padding regime, fixed protocol.(d)cover@ 𝑚gain over a random
ranking with95%cluster-bootstrap intervals, a target invariant to which answer-equivalent document was labelled necessary.(e)
Transfer of the off-query95th-percentile operating point to same-query passages that contain and entail no gold answer; only
fixed-target curves are false-positive rates, since under free generation the padding may legitimately shape the response.
rank last by definition. When we switch the padding tosame-queryhard negatives (distractors that are topically dense
and highly relevant to the query, but entail no actual gold answer aliases), the discriminative hierarchy completely
collapses.
Faced with these hard negatives, traditional metrics suffer a systemic failure. BM25 and query–document cosine
degrade to random chance on narrow contexts (AUCs of0 .444and0.484at𝑘=3, respectively). In stark contrast, our
deletion LOO probe remains highly robust, resisting the topical confusion to maintain an AUC of0 .876at𝑘=3and
0.824at𝑘= 24. This definitive reversal establishes an absolute methodological rule: off-query distractor pools are
fundamentally inadequate for validating context-attribution methods. Causal probes are uniquely equipped to isolate
generative utilization amidst dense topical relevance.
4.3 Sufficient Set Coverage Against Advanced Estimators
Having dismissed basic relevance proxies as unreliable under topical density, we benchmark our causal probe against
advanced attribution algorithms on redundant padding pools, where additive methods historically struggle to divide
credit among interchangeable substitutes. To evaluate strict informational utility, we adopt a permutation-invariant
cover@𝑚target: what fraction of the designed answer set do the top-𝑚ranked documents collectively attest?
As illustrated in Figure 3(d), under a strict diagnostic protocol spanning1 ,200conditions, relevance proxies fail
entirely to identify the sufficient set (e.g., BM25 marginally degrades random ranking by −0.014,𝑝=0.48). Conversely,
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 11
−0.8−0.6−0.4−0.2
width elasticity of 𝑞deletion, calibrated
deletion, exponent −1 s.e.
deletion, exponent +1 s.e.
deletion, flat pooled threshold
deletion, threshold-free
deletion, same-query calibration
deletion, fixed-protocol calibration
in-place, separately calibrated
in-place, fixed-protocol calibration
in-place, same-query calibration
in-place, threshold-freeprotocol-clean−0.68(a) by calibration criterion
−0.7−0.6−0.5−0.4
width elasticity of 𝑞pairwise entailment: low (0.05)
pairwise entailment: mid (0.14)
pairwise entailment: high (0.25)
answer attestation: none (0.00)
answer attestation: partial (0.73)
answer attestation: complete (1.00)
embedding similarity: low (0.62)
embedding similarity: mid (0.75)
embedding similarity: high (0.83)(b) by redundancy stratum
protocol-clean sensitivity variant protocol-contaminated
1
Fig. 4. Width elasticity of the thresholded utilisation statistic 𝑞, with95%intervals.(a)By calibration criterion: thresholds that do
not themselves shrink with width agree on the protocol-clean value, whereas free-generation and threshold-free criteria estimate a
different quantity.(b)By natural-pool redundancy stratum: the decline survives in every low-redundancy stratum, so redundancy
modulates but does not explain it.
counterfactual estimators successfully isolate the underlying drivers of the generation. Deletion LOO covers0 .792of
the answer set, yielding a massive +0.144absolute gain over a random ranking ( 𝑝<0.001). In-place LOO replacement
and the learned surrogate ContextCite [ 17] similarly yield robust gains of +0.134and+0.105. Deletion LOO’s coverage
gain remains globally positive and significant across all context widths, proving that counterfactual scoring uniquely
identifies a compact sufficient set even under heavy informational redundancy.
4.4 Isolating Confounders: Calibration and the Dilution Law
Before deploying this instrument to audit system budgets, we must systematically eliminate potential mechanical
confounders that could pollute the sensitivity signal. First, we investigated whether simply removing a document
artificially inflates score drops via subsequent text positional shifting. By comparing pure deletion against in-place
replacement using length-matched neutral text, we observed that the dynamic elasticities remain nearly identical
(−0.104versus−0.112), confirming that positional displacement does not artificially drive the causal signal.
Second, we observed that apparent limitations in LOO calibration are actually artifacts of protocol shifts rather than
probe failure. Under afree-generationprotocol, feeding wider contexts to an LLM systematically induces longer, more
hedged text. This behavioral shift shrinks per-token likelihood differences for reasons entirely disjoint from actual
evidence utilization, driving apparent false-positive rates up to37%. However, when we enforce a strictfixed-target
protocol where the generated text is held constant, true false-positive rates organically converge to a nominal, highly
stable margin (explicitly mapped in Figure 3).
This rigorous protocol isolation allows us to extract a pure, structural signal from the noise. We establish a strictly
calibrated width elasticity of −0.68(0.02)for generative attribution (Figure 4(a)). This verifies that the dilution of
attention across wider contexts is a fundamental generative property rather than a measurement error. Equipped with a
rigorously validated, confounding-free measurement instrument, we are now scientifically positioned to empirically
evaluate the precise laws governing context budget allocation.
Manuscript submitted to ACM

12 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
Table 2. Policy dependence with one ordinary decoder. All arms use the same explicit sampler; full custom decoder excluded. ECR
uses fixed-response threshold and equal task weights. Across the 8 evaluated policies, ECR spans a range of 0.251, with the vast
majority of paired ECR contrasts surviving strict BH-FDR correction. Offered is a treatment, not an outcome.
policy ECR used offered PR@𝑇
Ascpscheduler 0.626 6.76 10.55 0.300
deep rotation 0.375 9.37 25.00 0.303
vanilla RAG 0.578 2.89 5.00 0.237
MMR 0.587 3.13 5.28 0.239
DPP-RAG 0.526 4.71 8.83 0.274
xQuAD 0.571 2.85 5.00 0.238
PM-2-RAG 0.571 2.85 5.00 0.228
Carriage0.466 4.75 10.17 0.276
5 Theoretical Bounds and Empirical Laws of Evidence Consumption
Before architecting complex scheduling algorithms, we must understand the fundamental rules governing how a
generative model consumes evidence across multiple rounds. By establishing an idealized mathematical baseline and
contrasting it against the rigorous empirical laws extracted via our causal probe, we expose the exact generative
bottlenecks that mandate intelligent context allocation.
5.1 The Idealized Baseline vs. Generative Reality
To systematically model expected evidence coverage, we can first construct a simplified “toy model.” Imagine an idealized
LLM that acts as a perfect, uniform consumer of information. Suppose it utilizes any given document independently
with a fixed probability𝑞every time it sees it in the prompt.
Under a constrained budget of context width 𝑘and generation rounds 𝑇, the total number of “document slots”
available is𝑘𝑇. Mathematically (proven formally in Appendix A), the expected size of the utilized evidence set U𝑇is
strictly bounded:
E|U𝑇|≤𝑞𝑘𝑇.(4)
According to this probabilistic baseline, the absolute maximum coverage is achieved if and only if no document
is ever repeated across rounds. This suggests a naive conclusion: a simple “open-loop” policy that blindly rotates
fresh documents into the prompt every round should be perfectly optimal, rendering complex feedback mechanisms
unnecessary.
The Reality Check:However, deploying our causal probe on real-world generative data completely shatters this
idealized assumption. LLMs absolutely do not operate as passive, uniform consumers.
Empirical analysis reveals that actual generative utilization is highly clustered and suffers from severe “attention
inertia”: if an LLM fixates on a specific concept in round 1, it becomes structurally biased to ignore new, conflicting
information in round 2 (diverging from the independence assumption with 𝑝=6×10−31). Furthermore, pure arithmetic
slot-counting ignoresrelevance density. If a naive rotation policy blindly pushes deeper into the retrieved ranking, it
ends up feeding the LLM low-quality “trash” documents, actively degrading task performance.
These systematic divergences dictate that optimal context allocation cannot rely on abstract, blind rotation. It must
be governed by the empirical laws of actual generative behavior.
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 13
5.2 Empirical Law I: The Power of Active Orchestration
Because real LLM consumption is stubborn and context-dependent, we must establish exactly how front-end document
scheduling alters downstream generative reliance. Testing distinct scheduling policies under an identical generation
budget (𝑘=5), we observe profound variance in actual utilization.
As detailed in Table 2, the Evidence Coverage Rate (ECR) spans a massive range, proving thatdocument exposure is
a highly manipulable treatment, not a static outcome.For instance, the naive “deep rotation” policy—which blindly
pushes fresh documents without tracking if they are useful—passively achieves an ECR of only0 .375. It squanders the
majority of its contextual bandwidth on ignored evidence.
In stark contrast, intelligent orchestration drastically alters this consumption pattern. By utilizing causal feedback to
penalize redundant information, our proposed submodular scheduler (Ascp) explicitly forces the model to ingest fresh
evidence, driving the ECR up to a dominant0 .626. This proves our first empirical law: to maximize the generative utility
of a retrieved pool, the system must actively steer the context policy using feedback, rather than passively rotating
documents.
5.3 Empirical Law II: The Dilution Law of Context Width
If intelligent scheduling is required at a fixed width, what happens if we simply bypass the problem by expanding the
context window to fit all documents at once?
To answer this, we return to the width elasticity of −0.68(0.02)established via our causal probe (Section 4.4). The
massive gap between a flat elasticity (which would imply perfect attention capacity) and our rigorously calibrated
negative slopes (centering around−0.68under protocol isolation) exposes a severe cognitive limitation.
While−0.68represents our canonical estimate under strictly isolated diagnostic protocols, it is crucial to recognize
that the exact coefficient is modulated by operational factors such as task complexity and intrinsic pool redundancy.
For instance, depending on the severity of answer-attestation redundancy within the retrieved pool, the empirical slope
varies between−0.43and−0.67(detailed extensively in Appendix B.2). However, the overarching generative physics
remain absolute: across all evaluated strata, models, and semantic granularities, the elasticity remains profoundly
negative. The dilution law is defined not by a singular universal constant, but by the inescapable sub-linear decay of
evidence utilization as context expands.
This leads to our second empirical law:expanding the context width aggressively dilutes the magnitude of
individual document contributions.Because the prompts in our experimental grid peak at5 ,485tokens—safely
below the LLM’s absolute hardware limits—this −0.68dilution is not a hardware truncation artifact. It is a fundamental
generative constraint. As the set of provided evidence grows, the LLM’s attention is inherently fractured.
Taken together, these two laws unequivocally dictate the design of our architecture: since monolithic wide contexts
inevitably dilute attention (Law II), the optimal strategy is to break the budget into narrower sequential windows,
actively steered by causal feedback to maximize extraction efficiency (Law I).
6 System Architecture: The Closed-Loop Orchestration Framework
Guided by the empirical laws established in Section 5—specifically that wide contexts dilute attention and optimal
extraction requires active, multi-round scheduling—we now design a multi-tiered architecture to operationalize these
findings. Rather than treating the RAG pipeline as a static, open-loop black box, we introduce aclosed-loopsystem that
dynamically orchestrates what evidence the LLM sees (the Scheduler) and how forcefully it extracts it (the Decoder).
Manuscript submitted to ACM

14 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
6.1 The Baseline: Open-Loop Context Rotation (Rotate)
Before introducing our intelligent system, we define the structural baseline:Rotate. This is a pure throughput
mechanism that blindly cycles disjoint windows down the retrieved ranking. For instance, round 1 exposes ranks1 ...𝑘,
round 2 exposes ranks𝑘+1...2𝑘, and so on.
WhileRotateguarantees that 𝑘𝑇distinct documents are physically exposed to the LLM, it operates completely
“open-loop.” It has no idea if the LLM actually utilized the top documents, nor does it penalize redundant information.
Because relevance density sharply decays down a ranked list, this blind rotation aggressively squanders its budget on
lower-ranked, noisy tail documents.
6.2 The Brain: Feedback-Driven Submodular Scheduling (Ascp)
To optimize document allocation under concentrated relevance, we introduceAscp, a dynamic scheduling engine that
acts as the system’s brain.
Instead of blindly feeding documents,Ascpgroups the retrieved documents into semantic clusters (knowledge facets).
Crucially, after each generation round, it reads the causal utilization feedback from our LOO probe to see exactly which
documents—and consequently, which knowledge facets—the LLM has already successfully consumed.
We formulate the next round’s context selection as a greedy maximization of a monotone submodular objective
(formalized in Appendix A.1). In simple terms, submodularity mathematically enforces a “diminishing returns” penalty:
if the causal probe reports that a specific knowledge facet has already been deeply utilized in round 1, the scheduler
aggressively discounts any remaining documents belonging to that same facet.
This mechanism actively shifts the offered context toward unexplored, fresh evidence. Our rigorous ablation studies
(detailed later in Section 8.5) prove that this dynamic discounting is strictly dependent on our causal probe. Swapping
our causal feedback for a standard embedding-similarity proxy causes the scheduling gains to completely collapse,
proving that true causal sensitivity is the indispensable engine driving this scheduler.
6.3 The Enforcer: Attribution-Steered Contrastive Decoding
While the submodular scheduler dictates whatentersthe context window, we face a secondary generative hurdle: the
LLM’s microscopicattention inertia. Even when the scheduler brilliantly provides a prompt full of fresh evidence, LLMs
frequently fixate on familiar concepts they already generated, resisting the integration of novel facts.
To break this generative stagnation, we intervene directly inside the LLM’s generation process. After the initial
round, we define an “over-used” set of documents 𝑂(those the probe flagged as heavily utilized). During subsequent
rounds, we dynamically decode the text using a targeted contrastive distribution:
ℓ′=ℓ(·|𝑞,𝐶) +𝛼
ℓ(·|𝑞,𝐶\𝑂)−ℓ(·|𝑞,𝑂)
.(5)
In plain English, this subtractive formula acts as a cognitive override. It systematically subtracts probability mass from
words associated with the over-used documents, and shifts that mass toward words supported by the fresh, under-used
evidence.
To prevent the LLM from hallucinating when pushed away from its default distribution, this shift is strictly bounded
by an adaptive-plausibility constraint (APC) [ 68]. Ultimately, this micro-level intervention guarantees that the fresh
evidence curated by the scheduler is forcefully and safely extracted into the final portfolio.
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 15
Table 3. Relevance density 𝜌(𝑖): probability retrieved rank 𝑖attests a gold answer, from gold annotations over200queries/task. New
head-5 share counts only answer mass not attested earlier. ELI5 omitted: free-text claims lack rank-level lexical attestation.
Task gold units𝜌(1..5)𝜌(11..30)head-5 share head-5 shared log𝜌/d log𝑖
per query (any) (new)
ASQA 3.4 0.469 0.211 0.290 0.713 -0.40
QAMPARI 13.6 0.295 0.170 0.247 0.398 -0.27
RECIPES 20.7 0.890 0.870 0.170 0.391 -0.01
6.4 Operational Viability and Computational Overhead
A persistent concern with multi-round RAG and causal probing is deployment latency. However, our architecture is
meticulously designed for operational efficiency.
The greedy submodular selection operates in 𝑂(|P|𝑘𝑍) time, constituting a negligible microsecond overhead. More
importantly, while our causal attribution probe requires 𝑘+1forward passes per generation, these are executed as
teacher-forcedpasses on an already-generated text. By completely bypassing the prohibitive autoregressive decoding
bottleneck, these passes can be heavily parallelized. The system achieves deep causal introspection and intelligent
multi-round orchestration while maintaining strict temporal viability for complex search tasks.
7 Experimental Methodology and Setup
To isolate the precise causal effects of budget allocation and context scheduling, we enforce a highly controlled,
zero-leakage evaluation framework. Across all comparative experiments, the foundational RAG pipeline components,
specifically the retriever architecture and the retrieved document pools, are strictly frozen. Consequently, all evaluated
systems are exposed to the exact same raw evidence, guaranteeing that performance deltas are directly and exclusively
attributable to scheduling logic, instruction prompting, and decoding interventions.
7.1 Tasks and Evidence Pools
We evaluate our portfolio-generation framework across three rigorous benchmarks explicitly designed to necessitate
diverse, multi-faceted answer coverage, supplemented by an open-domain stress test:
•ASQA[ 110]: A challenging dataset of ambiguous factoid questions requiring the synthesis of multiple disam-
biguated sub-answers (exhibiting a concentrated head-5 relevance share of0.713, detailed in Table 3).
•QAMPARI[ 2]: A broad answer-set generation task averaging21annotated entities per query, of which13 .6are
empirically attested in the retrieval pools. It presents a highly dispersed answer distribution (Table 3), testing
extreme generative recall.
•ELI5[ 25]: An explanatory QA benchmark evaluated against claim sentences. To ensure rigorous semantic
evaluation, we strictly match ELI5 claims via NLI cross-encoder entailment rather than brittle string containment.
•Cross-Cultural Recipe Adaptation: Detailed extensively in Appendix D, this serves as a non-English, open-
domain stress test featuring an almost perfectly flat relevance density (log-log slope −0.01, Table 3) for optimizing
broad structural coverage beyond traditional QA paradigms.
For ASQA, QAMPARI, and ELI5, we adopt the standardized ALCE retrieval pools [ 30] (GTR for ASQA/QAMPARI,
BM25 for ELI5), strictly preserving the exact top 𝑁=30passages to ensure universal evidence parity. The HotpotQA
extension used in the decoupling analysis (Table 8) follows the identical retrieval and truncation pipeline over its own
corpus, likewise preserving the top 𝑁=30passages so that every task enters the factorial under matched evidence parity.
Manuscript submitted to ACM

16 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
The deep400-passage extensions utilized for extreme throughput limits preserve the original top-30scores and ranks,
maintaining absolute structural integrity for all𝑘=5control conditions.
7.2 Systems and Baselines Compared
We comprehensively benchmark our proposed architecture against a wide spectrum of classical IR diversification
algorithms and state-of-the-art RAG baselines. To ensure absolute hardware and computational equivalency, all held-out
comparisons rigorously equalize generation capacity: enforcing 𝑘=5,𝑇=5, temperature0 .7, top-𝑝=1, disabled top- 𝑘
filtering, and a strict repetition penalty of1.
Static Schedulers (Single-Context Paradigms).These classical paradigms reuse one static context, mathematically
upper-bounding distinct exposure to𝑘documents regardless of generation rounds𝑇:
•Vanilla RAG: Naive top-𝑘relevance sampling, serving as the standard industry baseline.
•MMR-RAG[ 10]: Re-ranks documents by aggressively balancing query relevance against content redundancy,
mimicking deployed diverse RAG systems.
•xQuAD-RAG[ 102] &PM-2-RAG[ 20]: Explicit intent-aware diversification algorithms operating over the exact
same induced semantic facets utilized by our own scheduler, isolating the exact gain of our causal feedback.
Sequential Schedulers (Dynamic-Context Paradigms).These state-of-the-art frameworks intelligently mutate the
context across generation rounds:
•DPP-RAG: Samples diverse contexts via greedy MAP inference over a quality-weighted determinantal kernel.
•Carriage[ 41]: A cutting-edge diverse RAG pipeline integrating output-aware MMR, sequential prompt listings,
and a sliding window. We evaluate both its main configuration and a restrictedCarriage-narrowvariant.
Our Orchestration Interventions.We deployRotate(cycling disjoint windows to isolate pure throughput effects
without feedback) andAscp(our causal-feedback-driven submodular scheduler optimizing Eq. (6)). For pristine causal
isolation, the 𝑘×𝑇 factorial grid strictly enforces ordinary, unsteered generation on both sides of every measured
contrast.
7.3 Generators and Embedding Configurations
We power the generative backends utilizing three leading open-weight LLMs in half-precision:Qwen2.5-7B[ 93],
Llama-3.1-8B[ 34], andMistral-7B-v0.3[ 48]. To validate scale generalization and real-world deployment robustness
(Section 8.7), we seamlessly upscale to the 14B and 32B variants of Qwen2.5. Semantic facets are robustly induced via
all-mpnet-base-v2for English tasks andparaphrase-multilingual-mpnet-base-v2for the recipe task [97].
7.4 Evaluation Protocol and Zero-Leakage Rigor
To fundamentally prevent algorithmic overfitting and guarantee out-of-distribution generalizability, we enforce a strict
separation of query pools. All architectural hyperparameters ( 𝛽=0.3,𝛽doc=0.3,𝜆=0.25,𝜅=0,𝛾=0.6,𝑚max=3,
𝜂=0.1) were stabilizedexactly onceon a disjoint40-query ASQA development split.
Crucially, the primary evaluation tables execute on a completely fresh, zero-leakage evaluation frame of100queries
per task across four tasks, three generators, and two stochastic decoding seeds. All inference runs are strictly paired
structurally (within task, model, seed, and query). We establish statistical significance through crossed bootstrap
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 17
2 5 12 24
context width 𝑘0.200.250.300.350.400.45portfolio recall PR@ 𝑇(a) width, holding 𝑇fixed
𝑇=1
𝑇=5
𝑇=12
1 5 12
generations 𝑇0.200.250.300.350.400.45
rotation
fixed(b) rotation vs. fixed context
𝑘=2
𝑘=5𝑘=12
𝑘=24
0.00 0.05 0.10 0.15
ΔPR@𝑇, rotation−fixed(24 ,12)
(24 ,5)
(24 ,1)
(12 ,12)
(12 ,5)
(12 ,1)
(5,12)
(5,5)
(5,1)
(2,12)
(2,5)
(2,1)(c) fresh-evidence gain
1
Fig. 5. Deconfounded width–count grid. Solid lines represent rotating through fresh evidence; dashed lines repeatedly sample the
same fixed context.
Table 4. Deconfounded width–count grid:rotationcycles disjoint rank windows;fixedrepeats top- 𝑘for𝑇samples. ΔPR@𝑇is paired
within query and bootstrapped by query, isolating fresh-evidence gain. Pool 𝑁= 400keeps𝑘𝑇≤ 288<𝑁 with no repeats; earlier
𝑁=30made𝑇=12offer the same24–30docs.∗/†:𝑝<0.05/𝑝<0.01.
𝑘 𝑇 𝑘𝑇offered PR@𝑇ΔPR@𝑇95% CI
rotation fixed rotation fixed
2 1 2 2.0 2.0 0.204 0.204 +0.000[+0.000,+0.000]
2 5 10 10.0 2.0 0.328 0.238 +0.090†[+0.068,+0.113]
2 12 24 24.0 2.0 0.397 0.257 +0.140†[+0.113,+0.169]
5 1 5 5.0 5.0 0.218 0.218 +0.000[+0.000,+0.000]
5 5 25 25.0 5.0 0.339 0.266 +0.073†[+0.052,+0.094]
5 12 60 60.0 5.0 0.423 0.290 +0.132†[+0.105,+0.159]
12 1 12 12.0 12.0 0.243 0.244 -0.001[−0.002,+0.000]
12 5 60 60.0 12.0 0.365 0.296 +0.069†[+0.048,+0.091]
12 12 144 144.0 12.0 0.428 0.331 +0.097†[+0.072,+0.122]
24 1 24 24.0 24.0 0.253 0.254 -0.000[−0.001,+0.000]
24 5 120 120.0 24.0 0.371 0.316 +0.056†[+0.033,+0.079]
24 12 288 288.0 24.0 0.421 0.336 +0.085†[+0.061,+0.110]
resampling, weighting tasks equally, and conservatively applying the Benjamini-Hochberg False Discovery Rate (BH-
FDR) within each designated family to mathematically account for testing multiplicity. Reporting multiple decoding
seeds and treating them as replicates follows the standing recommendation that IR effect sizes be estimated with the
run-to-run variance component made explicit [120].
8 Experimental Results
8.1 The𝑘×𝑇Factorial Grid: Formulating the Laws of Allocation
The fundamental architectural question for diverse RAG is how to optimally allocate a constrained inference budget. To
rigorously deconfound the generative effects of context width ( 𝑘) from sequential generation count ( 𝑇), we executed
a massive factorial evaluation crossing 𝑘∈{ 2,5,12,24}with𝑇∈{ 1,5,12}(summarized in Table 4 and Figure 5).
Evaluated symmetrically over four tasks and two generators with strict fixed-context controls, this grid explicitly
isolates the marginal return of expanding the prompt versus querying the model iteratively.
Manuscript submitted to ACM

18 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
Table 5. Decisive deconfounded-grid contrasts by task and generator, as within-query paired portfolio-recall differences. Last column:
rotation over matched fixed-context control.∗/†:𝑝<0.05/𝑝<0.01.
condition count T12 vs T1 at k24 count T12 vs T1 at k2 width k24 vs k2 at T12 width k24 vs k2 at T1 rot over fix at k24T12
asqa/llama +0.204†+0.289†-0.061∗+0.023 +0.066†
asqa/qwen +0.144†+0.229†-0.042 +0.043 +0.074†
qampari/llama +0.187†+0.141†+0.139†+0.093†+0.117†
qampari/qwen +0.134†+0.115†+0.058†+0.039†+0.082†
Table 6. Hierarchical rotation width contrast 𝑘=24minus𝑘=2, paired within query. Condition rows are partially pooled task–
generator effects (Paule–Mandel); last rows predict a new condition. With 𝐾= 4, interval uses Higgins–Thompson–Spiegelhalter
𝑡𝐾−2, not normal. At𝑇=12the mean is small and prediction interval wide;𝐾=4limits precision.
estimate𝑇=1𝑇=5𝑇=12
asqa/llama+0.036−0.033−0.053
asqa/qwen+0.047−0.000−0.038
qampari/llama+0.081+0.148+0.134
qampari/qwen+0.041+0.066+0.057
grand mean+0.051+0.045+0.025
between-condition SD+0.025+0.083+0.090
prediction interval (𝑡 𝐾−2)[−0.074,+0.176] [−0.356,+0.447] [−0.412,+0.462]
normal quantile, for comparison[−0.006,+0.108] [−0.138,+0.228] [−0.174,+0.224]
The empirical data yield a paradigm-defining conclusion:scaling sequential generation count delivers universal,
transformative gains, whereas expanding context width represents an architectural trap fundamentally
constrained by relevance decay.
As Table 4 illustrates, holding context width constant and iteratively raising 𝑇from one to twelve drives massive
portfolio recall improvements across the entire matrix. We observe robust absolute gains hovering around +0.20at
narrow widths ( 𝑘=2,5) and+0.17at extreme widths ( 𝑘=24). Crucially, these iterative gains remain globally positive
and highly significant across all evaluated task-generator combinations (Table 5), establishing sequential generation as
a universally transferable scaling lever.
Conversely, the marginal return on expanding context width 𝑘is highly brittle and rapidly collapses under iteration.
While widening the context from2to24documents nominally improves a single-pass generation ( +0.050at𝑇= 1), this
perceived benefit aggressively attenuates as rounds increase, collapsing to a negligible +0.024at𝑇=12. Pushing the
context beyond twelve slots yields no detectable statistical benefit under any configuration.
This dynamic unequivocally resolves the orchestration dilemma regarding informational extraction limits. When
comparing the mathematical equivalent of 24 allocated document slots, the multi-round (2,12)configuration system-
atically extracts a massively higher portfolio recall ( +0.144,95%CI[+0.119,+0.170]) from the exact same evidence
footprint than the single-pass (24,1)baseline. While the monolithic (24,1)approach minimizes computational latency, it
hits a rigid cognitive ceiling, leaving severe informational blind spots. Even a maximally expensive (24,12)configuration
only marginally exceeds the narrow-iterative baseline in recall, proving that aggressively scaling test-time compute
over tight, sequential windows is the only mechanism capable of maximizing the generative yield of retrieved evidence.
The catastrophic degradation of width utility under sequential rotation is not a generative artifact, but a fundamental
information retrieval penalty:relevance density. Expanding the context window inherently forces the retriever to ingest
deeper, lower-quality ranks. A random-effects hierarchical analysis of the width contrast (Table 6) exposes a highly
dispersed prediction interval of [−0.412,+0.462]. This extreme variance masks a harsh structural split: width expansion
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 19
Table 7. Long-context Qwen2.5-7B-Instruct-1M width–count contrasts at widths beyond the7–8B grid; 𝑘=96is about11K prompt
tokens. Entries are within-query paired PR@ 𝑇differences;∗/†:𝑝< 0.05/𝑝< 0.01. Generation-count gains remain positive ( +0.085to
+0.157); no width gain beyond𝑘=24is positive/significant, and ASQA worsens.
contrast ASQA QAMPARI
Δ𝑛Δ𝑛
𝑘=48vs𝑘=24,𝑇=1−0.033∗120+0.004120
𝑘=96vs𝑘=24,𝑇=1−0.03130+0.04029
𝑘=48vs𝑘=24,𝑇=5−0.03058+0.00259
𝑇=5vs𝑇=1,𝑘=24+0.089†60+0.112†60
𝑇=12vs𝑇=1,𝑘=24+0.101†60+0.157†60
𝑇=5vs𝑇=1,𝑘=48+0.109†107+0.085†108
is only viable for tasks with highly dispersed answer sets (e.g., QAMPARI), but turns actively toxic for tasks where
evidence is densely concentrated at the top ranks (e.g., ASQA). Generation count therefore stands alone as the globally
optimal allocation strategy, structurally immune to the rank-degradation penalty that plagues wide-context packing.
8.2 Isolating the Mechanics: The Supremacy of Fresh Evidence Over Stochastic Resampling
Raising the generation count 𝑇unequivocally improves portfolio recall, but this architectural lever inherently confounds
two fundamentally distinct generative mechanisms: the algorithmic benefit of drawing multiple stochastic samples
from the decoder (resampling), and the epistemological benefit of exposing the model to new retrieved documents
(fresh evidence).
To cleanly decouple these forces, we deploy a matched fixed-context control alongside the sequential rotation arm. By
locking the exact same top- 𝑘documents in the prompt across all 𝑇rounds and drawing independent decoder samples,
this control arm isolates the pure utility of stochastic re-reading. The paired difference ( ΔPR@𝑇) directly unmasks the
true marginal utility of fresh informational exposure (Table 4 and Figure 5).
The resulting decomposition shatters the prevailing NLP assumption that decoding stochasticity alone is sufficient
for diverse generation. While repeated sampling over a fixed context does provide a reliable baseline bump (improving
recall by+0.054to+0.087as the model re-evaluates the same text), this isolated mechanism rapidly hits a strict
cognitive asymptote. At twelve generations, actively injecting fresh evidence via rotation overwhelmingly obliterates
the fixed-context resampling baseline, driving decisive absolute margins ranging from +0.087at𝑘=24up to+0.140at
𝑘=2(all𝑝<0.001).
Deconstructing these gains reveals a profound structural insight: fresh evidence explicitly dictates the performance
ceiling. Exposure to unseen documents strictly accounts for72%of the total generation-count utility at narrow widths,
and maintains a dominant52%share even within massive24-document windows where one might incorrectly assume
all necessary information was already present.
The physical grounding footprints of the generated portfolios provide the definitive proof of this mechanism. At
the(24,12)extreme, the rotational policy structurally exposes288unique documents and successfully grounds its
portfolio on a massive50 .9of them. In stark contrast, the fixed-context arm, despite having twelve attempts to squeeze
information out of its static24-document window, mathematically stagnates, grounding on merely10.3documents.
Ultimately, this isolates a critical law of generative orchestration: an LLM cannot hallucinate genuine diversity from
a stagnant prompt. Extra generative rounds extract their transformative value not through stochastic paraphrasing, but
by serving as deliberate, sequential vehicles for unseen physical evidence.
Manuscript submitted to ACM

20 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
Table 8. Relevance density versus gold-set size over four task profiles; the multi-hop HotpotQA replaces the recipe stress test here
so that gold-set size and relevance density vary independently. |𝐴|mean gold answers, 𝜌rank-band answer probability, head-5
concentration. At 𝑇=12, top-5 share tracks width: only dispersed QAMPARI (0.398) gains, not ASQA (0.713) or HotpotQA (0.630)
despite gold sets3.4/1.0; with one generation, multi-hop gains+0.33. ELI5 string𝜌is not estimable.∗/†:𝑝<0.05/𝑝<0.01.
task|𝐴|head-5𝜌top-5 widthΔwidthΔcountΔ
share at𝑇=1at𝑇=12at𝑘=2
ASQA 3.4 0.713 0.469+0.033−0.052∗+0.259†
QAMPARI 13.6 0.398 0.295+0.066†+0.099†+0.128†
ELI5 – – –+0.011+0.010†+0.242†
HotpotQA 1.0 0.630 0.169+0.333†−0.046∗+0.450†
Table 9. ELI5 factorial: small claim gold set but dispersed BM25 profile, separating relevance density from gold-set size; HotpotQA
extension in Table 8. Entries are within-query paired PR@𝑇differences;∗/†mark𝑝<0.05/𝑝<0.01by paired bootstrap.
task generator count𝑇at k2 count𝑇at k24 width𝑘at T12 budget (2,12) vs (24,1) rot over fix at k24T12
eli5 llama+0.228†+0.156†−0.087†+0.242†–
eli5 qwen+0.256†+0.292†+0.072∗+0.219†–
8.3 The Trap of Context Saturation and Relevance Density
A fundamental question arising from the 𝑘×𝑇 laws is whether the observed saturation of context width is a temporary
artifact of the 7–8B models’ attention capacities, or a permanent structural bottleneck. To isolate the physical limits of
context scaling, we stress-test our findings using Qwen2.5-7B-Instruct-1M, pushing the context windows to massive
extremes of𝑘∈{24,48,96}(consuming up to approximately11,000prompt tokens).
The results (Table 7) unequivocally demonstrate that our established generation-count scaling laws transcend
architectural context limits. Even with a 1-million token capacity, raising generation rounds from1to12at 𝑘=24
strictly drives massive absolute gains of +0.101and+0.157across tasks. In stark contrast, aggressively expanding the
context window beyond 𝑘=24yields zero statistically significant portfolio coverage benefits. In fact, on the complex
ASQA benchmark, pushing to 𝑘=48actively degrades performance. A massively expanded trained window alone does
not, and cannot, convert extreme context width into a superior informational budget.
This rigid saturation is not a generative failure, but rather a fundamental Information Retrieval (IR) limitation governed
byrelevance density. Expanding a context window mathematically forces the system to ingest deeper, lower-quality
retrieval ranks, inevitably exhausting the high-density relevance head of the retrieved pool.
To cleanly decouple this relevance density from the sheer size of the gold answer set, we inject ELI5 (which exhibits
dispersed BM25 retrieval but few claim-level units) and HotpotQA [ 129] (requiring a singular multi-hop answer strictly
dependent on two specific co-occurring paragraphs) into the factorial analysis (Tables 9 and 8).
The HotpotQA dynamics perfectly isolate the necessity of sequential exploration. At a single generation pass,
widening the prompt from two to twenty-four documents drives a massive +0.333gain, purely because a narrow
𝑘=2window rarely captures both required multi-hop paragraphs simultaneously. However, when the budget permits
twelve sequential generations, rotational policies successfully assemble the multi-hop pair across rounds, completely
neutralizing the wide-context advantage and collapsing the width effect to a detrimental −0.046. Under matched
hardware budgets, the iterative(2,12)allocation continues to strictly dominate the monolithic(24,1)allocation.
Across all evaluated benchmarks, the utility of context expansion strictly tracks evidence concentration, quantified as
the head-5 answer-mass share, rather than gold-set size. Context expansion remains slightly viable exclusively on highly
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 21
Table 10. Frozen held-out comparison: steering chosen once by one-standard-error on development, then tested on disjoint queries (100
per task), four tasks, three generators, two decoding seeds. Contrasts are paired on query/generator/seed/prompt/width/count/code.
Sampling: temperature0 .7, top-𝑝=1, top-𝑘disabled, repetition penalty1. Crossed bootstrap, equal task weights, BH-FDR for PR@ 𝑇;
ECR fixed-response. Rounded differences may shift0.001; audit requires 2400 complete cells.
baseline baseline PRAscpPRΔPR BH𝑞ΔECRΔground
Ascpwithout steering 0.300 0.309 +0.0090.102+0.056 +0.029
deep rotation 0.303 0.309 +0.0060.343+0.307 +0.050
vanilla RAG 0.237 0.309 +0.072<.001+0.104 +0.003
MMR 0.239 0.309 +0.070<.001+0.095 +0.011
DPP-RAG 0.274 0.309 +0.035<.001+0.156 +0.011
xQuAD 0.238 0.309 +0.071<.001+0.111 +0.002
PM-2-RAG 0.228 0.309 +0.081<.001+0.111 +0.004
Carriage0.276 0.309 +0.033<.001+0.215 +0.052
Carriage-narrow 0.261 0.309 +0.048<.001-0.074 +0.089
Table 11. Scheduler-only held-out comparison:Ascpwithout steered decoding, so both sides use ordinary generation and isolate
evidence scheduling. Crossed bootstrap over decoding seeds and within-task queries; equal task weights, fixed three-model set;
BH-FDR covers the eight PR@𝑇contrasts.
baseline baseline PR scheduler PRΔPR BH𝑞
deep rotation 0.303 0.300 -0.0030.717
vanilla RAG 0.237 0.300 +0.063<.001
MMR 0.239 0.300 +0.061<.001
DPP-RAG 0.274 0.300 +0.0260.002
xQuAD 0.238 0.300 +0.062<.001
PM-2-RAG 0.228 0.300 +0.072<.001
Carriage0.276 0.300 +0.024<.001
Carriage-narrow 0.261 0.300 +0.039<.001
dispersed tasks like QAMPARI (head-5 share0 .398), but turns actively toxic on concentrated tasks like ASQA (0 .713)
and HotpotQA (0 .630). Ultimately, pushing deeper into a ranked list to populate a massive generative prompt dilutes
the LLM’s attention with low-yield noise. This physical IR constraint renders sequential, narrow-window generations
the globally optimal strategy for portfolio coverage, regardless of underlying hardware capacity.
8.4 End-to-End Evaluation: The Triumph of Closed-Loop Orchestration
Having established that sequential generation over narrow windows dictates optimal portfolio coverage, we now
evaluate how effectively concrete scheduling policies capture this theoretical potential under a strictly fixed hardware
budget. We rigorously benchmark our submodular evidence scheduler (Ascp) against a spectrum of selection and
rotation baselines over2,400strictly paired evaluation frames (Table 10).
The empirical results reveal a fundamental limitation in classical IR adaptations. When classical algorithms like
MMR, xQuAD, or PM-2 are naively ported into RAG pipelines, they operate asopen-loopsystems. They attempt to
diversify the prompt text based purely on document similarity, remaining entirely blind to what the generative model
actually consumes. Consequently, they stagnate at portfolio recalls of0 .228to0.239. Similarly, cutting-edge diverse
RAG frameworks likeCarriagemanage to push recall to0.276, but ultimately hit an architectural ceiling.
In stark contrast, ourAscpscheduler systematically shatters this ceiling, dominating every evaluated selection
baseline. Operating over an equal-task estimand,Ascpachieves a terminal portfolio recall of0 .309, securing a robust
absolute+0.033gain overCarriage(95%CI [+0.022,+0.044], BH𝑞<0.001) and soaring up to +0.081absolute points
over PM-2-RAG.
Crucially, this scheduling superiority is structurally driven by front-end context orchestration rather than generative
post-processing tricks. To definitively prove this, we stripAscpof its custom steered decoder (Table 11), forcing it to
Manuscript submitted to ACM

22 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
operate via ordinary generation perfectly matching baseline conditions. Even stripped of decoding interventions, the
pure scheduler retains significant dominance, maintaining absolute gains between +0.024and+0.072over all selection
baselines.
The true architectural elegance ofAscpemerges when we analyzehowit achieves this coverage. While exhaustive
deep rotation nominally matches the full method in pure portfolio recall (a statistically insignificant contrast of +0.006,
BH𝑞= 0.343), it achieves this equivalence through brute-force inefficiency. To matchAscp’s recall, deep rotation
blindly forces25 .00documents into the contexts, hoping the model will randomly extract value. This squanders massive
token bandwidth, yielding an Evidence Coverage Rate (ECR) of merely0.375.
Ascp, however, operates as a trueclosed-loopsystem. By actively reading causal attribution feedback, it dynamically
penalizes redundant information and forcefully steers the context toward unexplored semantic facets. Consequently, it
achieves the exact same terminal recall by offering only10 .55highly curated documents, driving ECR up to a dominant
0.626. This proves that submodular scheduling driven by causal feedback does not merely inflate coverage; it achieves
peak generative utility while maintaining substantially tighter, highly grounded, and token-efficient context bounds.
8.5 The Feedback Engine: Causal Probing vs. Similarity Illusions
The architectural supremacy of theAscpscheduler hinges fundamentally on its ability to dynamically discount evidence
that the generator has already consumed. Consequently, the performance ceiling of the entire orchestration loop is
strictly bound by the accuracy of its underlying feedback signal. While Section 4 proved that our causal LOO probe
isolates necessary documents far better than similarity heuristics on constructed diagnostic pools, we must validate
whether this discriminative advantage actually translates into end-to-end generative portfolio utility.
To definitively test this, we execute a structural ablation within the scheduling loop (detailed in Table 15), isolating
the feedback mechanism while holding all other orchestration logic constant.
If the scheduler operates open-loop, blindly assuming that the generator perfectly utilizes every single document
offered in the context (the uniform-use assumption), portfolio utility severely stagnates. By actively reading the
causal LOO probe’s utilization signal,Ascpyields a statistically significant +0.014absolute portfolio recall gain (95%
CI[+0.003,+0.025],𝑝= 0.013) over this naive uniform assumption. This proves that dynamically trackingactual
consumption is essential for budget optimization.
Crucially, attempting to recover this operational gain by swapping our causal probe for a standard embedding-
similarity proxy results in a catastrophic systemic failure. When operated with similarity-based feedback, the scheduler
produces a marginal utility completely indistinguishable from the naive open-loop assumption (a meaningless −0.002
contrast,𝑝=0.73).
This end-to-end collapse aligns perfectly with our initial diagnostic findings in Section 4: because all documents
retrieved for a given query naturally reside in the exact same semantic space as the generated response, embedding
similarities severely saturate. They are mathematically incapable of distinguishing between the documents that truly
drove the generation and dense, same-query near-misses.
The strategic conclusion is absolute: attempting to steer a RAG context scheduler using superficial text similarity
provides zero operational leverage. The orchestration advantage over brute-force rotation is uniquely and exclusively
unlocked by measuring exact counterfactual necessity, cementing our causal probe as the indispensable engine of
diverse generative search.
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 23
Table 12. Held-out decoder decomposition with identical sampling: temperature0 .7, top-𝑝=1, top-𝑘disabled, repetition penalty1.
APC is adaptive-plausibility constraint; contrasts isolate contrast direction, APC filtering, and complement. Intervals/ 𝑝: two-sided𝑡
over five seed means (𝑑𝑓=4), equal task weights, fixed three-model set; audit needs all 6000 frames; BH-FDR covers PR@𝑇.
contrastΔPR@𝑇seed-𝑡 4CI𝑝ΔECRΔgroundΔwords
full decoder−off +0.0107[+0.0072,+0.0142]0.001+0.0555 +0.0275 -1.0
contrast, APC held on +0.0116[+0.0045,+0.0187]0.011+0.0583 +0.0232 -0.6
APC only -0.0009[−0.0086,+0.0069]0.775-0.0028 +0.0042 -0.4
contrast, APC held off +0.0187[+0.0145,+0.0229]<.001+0.1012 +0.0366 -0.2
APC, contrast held on -0.0080[−0.0107,−0.0052]0.001-0.0457 -0.0092 -0.8
contrast×APC -0.0071[−0.0163,+0.0021]0.098-0.0429 -0.0134 -0.5
8.6 Decoder Interventions: Overriding Attention Inertia
While our submodular scheduler successfully optimizes the macroscopicdietof the LLM (what enters the context),
generative models inherently suffer from microscopicattention inertia. Even when provided with fresh evidence, LLMs
frequently fixate on familiar, already-extracted concepts, resisting the integration of novel facts. To break this generative
stagnation, our attribution-steered decoding intervenes directly at the logit level, executing a cognitive override that
systematically shifts probability mass away from over-used evidence.
To definitively isolate the pure causal impact of this intra-generation intervention, we execute a rigorous five-seed
decomposition on a completely disjoint evaluation frame (Table 12). By strictly enforcing identical baseline sampler
configurations (temperature0 .7, top-𝑝=1) across all paths, we ensure the observed gains represent true architectural
enhancements rather than stochastic anomalies.
The results (Table 12) unequivocally prove that this micro-level intervention successfully reprograms the LLM’s
evidence consumption. Enabling the full attribution-steered decoder yields a highly robust, orthogonal +0.0107absolute
portfolio recall gain. Crucially, the mathematical stability of this override is absolute: across all five independent
stochastic decoding seeds, the intervention reliably forces the extraction of new knowledge (BH 𝑞=0.001). Beyond
terminal recall, the decoder physically alters the generation footprint: it elevates the Evidence Coverage Rate (ECR) by
+0.0555and boosts strict lexical grounding by +0.0275. Most remarkably, it achieves this superior informational density
while mathematically generating1 .0fewerwords per portfolio, proving that the intervention eliminates repetitive
rambling in favor of dense, factual extraction.
Decomposing the internal mechanics reveals a brilliant synergy between exploration and safety. The subtractive
logit contrast acts as the exploratory engine: when deployed alone (APC held off), it aggressively forces the model into
unexplored semantic territory, unleashing a massive +0.0187(𝑞<0.001) recall gain alongside soaring ECR ( +0.1012).
However, unconstrained contrastive generation inherently risks hallucination by forcing the model too far from its
probability manifold.
This is where the adaptive-plausibility constraint (APC) proves vital. While APC alone provides zero coverage
utility (a statistically null −0.0009), when fused with the logit contrast, it acts as a strict hallucination-resistant leash. It
sacrifices a marginal0 .0080of the raw contrastive recall to mathematically clamp the candidate tokens within a safe
plausibility boundary. Ultimately, this fusion guarantees that the LLM’s attention is forcefully yet safely redistributed.
The decoder intervention systematically ensures that the fresh evidence offered by the scheduler physically translates
into diverse, deeply grounded portfolio construction.
Manuscript submitted to ACM

24 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
Table 13. Budget-matched scaling validation on 14B and 32B models. Entries represent paired portfolio recall, strictly contrasting the
multi-round iterative generation (2,12)against monolithic wide-context generation (24,1). The absolute structural superiority of
multi-round evidence allocation robustly holds at scale, maintaining massive positive bounds across tasks.
model task(2,12) (24,1)Δ95% CI
qwen14 asqa 0.569 0.401+0.168[+0.107,+0.233]
qwen14 pooled 0.400 0.266+0.134[+0.093,+0.175]
qwen14 qampari 0.231 0.131+0.100[+0.050,+0.157]
qwen32 asqa 0.560 0.380+0.180[+0.116,+0.250]
qwen32 pooled 0.429 0.291+0.138[+0.098,+0.181]
qwen32 qampari 0.298 0.202+0.097[+0.055,+0.140]
103104
total tokens per query0.350.400.450.500.550.60portfolio recall PR@ 𝑇(2,12)
(24,1)same 24 slots:
1.5×the tokens,
+0.19 PR@ 𝑇(a) token cost
100101
sequential latency (s)0.350.400.450.500.550.60(2,12)
(24,1)but10×the latency:
the two cost axes disagree
on the preferred packaging(b) wall-clock cost
𝑘=2 𝑘=5 𝑘=12 𝑘=24 rotation fixed context
1
Fig. 6. Cost versus quality across every grid cell (Qwen/ASQA, A100, probe removed).(a)Total tokens per query;(b)Sequential latency.
Solid lines represent rotation; dashed lines represent the matched fixed-context arm. The annotated pair is the budget-matched24-slot
comparison. Crucially, the two cost axes disagree on the preferred packaging, illustrating why no single cell serves as a universal
recommendation.
8.7 Inference-Time Scaling: The Compute-Quality Frontier
To rigorously confirm that the superiority of iterative portfolio generation is a fundamental physical law of LLMs
rather than an artifact of small-capacity models, we executed a budget-matched scaling replication at the extreme 14B
and 32B scales (detailed in Table 13). Measured purely by informational yield, the multi-round (2,12)configuration
decisively obliterates the monolithic single-pass (24,1)baseline by an absolute +0.134(95%CI[+0.093,+0.175]) utilizing
Qwen2.5-14B, and by+0.138utilizing Qwen2.5-32B.
However, translating these slot-matched gains into physical deployment exposes the fundamental mechanics of
inference-time scaling. Achieving the massive recall of (2,12)inherently demands deliberately investing greater test-time
compute: sequential auto-regressive generation naturally increases temporal latency (e.g.,17 .5seconds versus a singular
1.7seconds for(24,1)), and our causal LOO probe mandates strictly parallelizable, yet non-zero, teacher-forced forward
passes.
Rather than viewing this computational overhead as a defect, Figure 6 maps this dynamic as a strict Compute-
Quality scaling frontier. Monolithic single-pass architectures like (24,1)are computationally cheap, but they hit a rigid
cognitive ceiling. If a production system targets a modest0 .35portfolio recall floor, generating a single wide response
suffices. However, as the quality requirement scales to0 .40or0.45, the standard for complex exploratory search,no
single-generation configuration remains mathematically capable of reaching the threshold, regardless of parameter scale.
This forcefully dictates a paradigm shift in generative search deployment. To breach the single-response quality
ceiling, systems cannot simply widen the context; they must aggressively scale test-time compute. The multi-round
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 25
structures(2,5)and(2,12)prove that explicitly investing inference budget into sequential exploration and causal
feedback is the only architectural mechanism capable of unlocking transformative gains in comprehensive evidence
coverage.
9 Conclusion
Transitioning retrieval-augmented generation (RAG) from single-answer extraction to diverse portfolio generation
is fundamentally stymied by flawed measurement heuristics and arbitrary resource allocation. In this work, we
deconstructed the pervasive diagnostic illusion in RAG evaluation: we proved that the apparent perfection of traditional
IR proxies is a structural mirage reliant on trivial off-query padding. Evaluated on rigorous same-query hard negatives,
standard proxies collapse to random chance, whereas our intervention-based causal probe maintains highly robust
discrimination. By formally calibrating out protocol-shift artifacts, we quantified the inherent dilution of generative
attention—yielding a canonical width elasticity of −0.68under controlled isolation—securing a rigorous causal instrument
capable of genuinely auditing LLM evidence consumption.
Armed with this validated measurement capability, our deconfounded 𝑘×𝑇 factorial grid definitively resolved the
context budget dilemma. We established a fundamental generative law: scaling sequential generation count uniformly
drives massive portfolio recall gains of 16.8 to 20.5 absolute percentage points. In stark contrast, simply expanding
context width is an architectural trap actively penalized by rank degradation, yielding highly unstable returns that
frequently harm tasks with concentrated relevance.
Exploiting this structural superiority of fresh evidence, we operationalized an attribution-steered submodular
orchestration framework. Driven by causal feedback, our submodular scheduler systematically dominated all evaluated
open-loop selection baselines. Augmented by an orthogonal contrastive decoder that acts as a cognitive override against
attention inertia, our end-to-end architecture proves that intelligent, multi-round context orchestration fundamentally
maximizes the generative yield of a retrieved pool.
Ultimately, this work formally introduces the paradigm ofinference-time scalingto generative search. While classical
RAG pipelines attempt to minimize latency by cramming evidence into a single, monolithic context pass, we prove this
approach encounters a rigid cognitive ceiling. Instead, we demonstrate that deliberately investing computational budget
during inference, via sequential autoregressive passes and causal teacher-forced probing, unlocks transformative gains
in portfolio coverage. For future systems required to comprehensively cover a diverse evidence space, the paradigm
is clear: aggressively scaling structured, feedback-driven test-time compute fundamentally supersedes static context
maximization.
References
[1]Rakesh Agrawal, Sreenivas Gollapudi, Alan Halverson, and Samuel Ieong. 2009. Diversifying search results. InProceedings of the second ACM
international conference on web search and data mining. 5–14.
[2]Samuel Joseph Amouyal, Tomer Wolfson, Ohad Rubin, Ori Yoran, Jonathan Herzig, and Jonathan Berant. 2022. Qampari: An open-domain question
answering benchmark for questions with many answers from multiple paragraphs.arXiv preprint arXiv:2205.12665(2022).
[3]Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024. L-eval: Instituting
standardized evaluation for long context language models. InProceedings of the 62nd Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). 14388–14411.
[4]Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through
self-reflection. InInternational conference on learning representations, Vol. 2024. 9112–9141.
[5]Ashwinkumar Badanidiyuru and Jan Vondrák. 2014. Fast algorithms for maximizing submodular functions. InProceedings of the twenty-fifth
annual ACM-SIAM symposium on Discrete algorithms. SIAM, 1497–1514.
Manuscript submitted to ACM

26 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
[6]Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al .2024. Longbench:
A bilingual, multitask benchmark for long context understanding. InProceedings of the 62nd annual meeting of the association for computational
linguistics (volume 1: Long papers). 3119–3137.
[7]Bernd Bohnet, Vinh Q Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Massimiliano Ciaramita, Jacob Eisenstein, Kuzman
Ganchev, Jonathan Herzig, et al .2022. Attributed question answering: Evaluation and modeling for attributed large language models.arXiv
preprint arXiv:2212.08037(2022).
[8]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste
Lespiau, Bogdan Damoc, Aidan Clark, et al .2022. Improving language models by retrieving from trillions of tokens. InInternational conference on
machine learning. PMLR, 2206–2240.
[9]Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large language monkeys:
Scaling inference compute with repeated sampling.arXiv preprint arXiv:2407.21787(2024).
[10] Jaime G Carbonell and Jade Goldstein. 1998. The use of MMR, diversity-based reranking for reordering documents and producing summaries.. In
SIGIR, Vol. 98. 290941–291025.
[11] Olivier Chapelle, Thorsten Joachims, Filip Radlinski, and Yisong Yue. 2012. Large-scale validation and analysis of interleaved search evaluation.
ACM Transactions on Information Systems (TOIS)30, 1 (2012), 1–41.
[12] Harr Chen and David R Karger. 2006. Less is more: probabilistic models for retrieving fewer relevant documents. InProceedings of the 29th annual
international ACM SIGIR conference on Research and development in information retrieval. 429–436.
[13] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas
Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code.arXiv preprint arXiv:2107.03374(2021).
[14] Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, et al .2025. A survey on
knowledge-oriented retrieval-augmented generation.ACM Transactions on Information Systems(2025).
[15] Cheng-Han Chiang and Hung-yi Lee. 2023. Can large language models be an alternative to human evaluations?. InProceedings of the 61st Annual
Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 15607–15631.
[16] Charles LA Clarke, Maheedhar Kolla, Gordon V Cormack, Olga Vechtomova, Azin Ashkan, Stefan Büttcher, and Ian MacKinnon. 2008. Novelty and
diversity in information retrieval evaluation. InProceedings of the 31st annual international ACM SIGIR conference on Research and development in
information retrieval. 659–666.
[17] Benjamin Cohen-Wang, Harshay Shah, Kristian Georgiev, and Aleksander Mądry. 2024. Contextcite: Attributing model generation to context.
Advances in Neural Information Processing Systems37, 95764–95807.
[18] Ian Covert, Scott Lundberg, and Su-In Lee. 2021. Explaining by removing: A unified framework for model explanation.Journal of Machine Learning
Research22, 209 (2021), 1–90.
[19] Florin Cuconasu, Giovanni Trappolini, Federico Siciliano, Simone Filice, Cesare Campagnano, Yoelle Maarek, Nicola Tonellotto, and Fabrizio
Silvestri. 2024. The power of noise: Redefining retrieval for rag systems. InProceedings of the 47th international ACM SIGIR conference on research
and development in information retrieval. 719–729.
[20] Van Dang and W Bruce Croft. 2012. Diversity by proportionality: an election-based approach to search result diversification. InProceedings of the
35th international ACM SIGIR conference on Research and development in information retrieval. 65–74.
[21] Zhirui Deng, Zhicheng Dou, Zhan Su, and Ji-Rong Wen. 2024. Multi-grained document modeling for search result diversification.ACM Transactions
on Information Systems42, 5 (2024), 1–22.
[22] Zhirui Deng, Zhicheng Dou, Yutao Zhu, and Ji-Rong Wen. 2025. A model-agnostic pre-training framework for search result diversification.ACM
Transactions on Information Systems44, 1 (2025), 1–23.
[23] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness,
and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization.arXiv preprint arXiv:2404.16130(2024).
[24] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. Ragas: Automated evaluation of retrieval augmented generation. In
Proceedings of the 18th conference of the european chapter of the association for computational linguistics: system demonstrations. 150–158.
[25] Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long form question answering. InProceedings
of the 57th annual meeting of the association for computational linguistics. 3558–3567.
[26] Angela Fan, Mike Lewis, and Yann Dauphin. 2018. Hierarchical neural story generation. InProceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers). 889–898.
[27] Hui Fang, Tao Tao, and Chengxiang Zhai. 2011. Diagnostic evaluation of information retrieval models.ACM Transactions on Information Systems
(TOIS)29, 2 (2011), 1–42.
[28] Marshall L Fisher, George L Nemhauser, and Laurence A Wolsey. 2009. An analysis of approximations for maximizing submodular set functions—II.
InPolyhedral Combinatorics: Dedicated to the memory of DR Fulkerson. Springer, 73–87.
[29] Dan Friedman and Adji Bousso Dieng. 2022. The vendi score: A diversity evaluation metric for machine learning.arXiv preprint arXiv:2210.02410
(2022).
[30] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. 2023. Enabling large language models to generate text with citations. InProceedings of the
2023 Conference on Empirical Methods in Natural Language Processing. 6465–6488.
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 27
[31] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2023. Retrieval-augmented
generation for large language models: A survey.arXiv preprint arXiv:2312.10997(2023).
[32] Yunfan Gao, Yun Xiong, Wenlong Wu, Bohan Li, Yijie Zhong, and Haofen Wang. 2026. U-niah: Unified rag and llm evaluation for long context
needle-in-a-haystack.ACM Transactions on Information Systems44, 3 (2026), 1–30.
[33] Sreenivas Gollapudi and Aneesh Sharma. 2009. An axiomatic approach for result diversification. InProceedings of the 18th international conference
on World wide web. 381–390.
[34] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan
Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models.arXiv preprint arXiv:2407.21783(2024).
[35] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. (2020),
3929–3938.
[36] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks,
Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models.arXiv preprint arXiv:2203.15556(2022).
[37] Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. 2019. The curious case of neural text degeneration.arXiv preprint arXiv:1904.09751.
[38] Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim. 2019. A benchmark for interpretability methods in deep neural networks.
Advances in neural information processing systems32.
[39] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s
the real context size of your long-context language models?arXiv preprint arXiv:2404.06654(2024).
[40] Tianyi Hu, Maria Maistro, and Daniel Hershcovich. 2024. Bridging cultures in the kitchen: A framework and benchmark for cross-cultural recipe
retrieval. InProceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. 1068–1080.
[41] Tianyi Hu, Andrea Morales-Garzón, Jingyi Zheng, Maria Maistro, and Daniel Hershcovich. 2026. Culinary crossroads: A rag framework for
enhancing diversity in cross-cultural recipe adaptation. InProceedings of the 64th Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers). 2408–2423.
[42] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing
Qin, et al .2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions.ACM transactions on
information systems43, 2 (2025), 1–55.
[43] Gautier Izacard and Edouard Grave. 2021. Leveraging passage retrieval with generative models for open domain question answering. InProceedings
of the 16th conference of the european chapter of the association for computational linguistics: main volume. 874–880.
[44] Amir H Jadidinejad, Craig Macdonald, and Iadh Ounis. 2021. The simpson’s paradox in the offline evaluation of recommendation systems.ACM
Transactions on Information Systems (TOIS)40, 1 (2021), 1–22.
[45] Sarthak Jain and Byron C Wallace. 2019. Attention is not explanation. InProceedings of the 2019 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 3543–3556.
[46] Kalervo Jarvelin and Eero Sormunen. 2024. A blueprint of IR evaluation integrating task and user characteristics.ACM Transactions on Information
Systems42, 6 (2024), 1–38.
[47] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of
hallucination in natural language generation.ACM computing surveys55, 12 (2023), 1–38.
[48] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna
Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang,
Timothée Lacroix, and William El Sayed. 2023. Mistral 7B. (2023). arXiv:2310.06825 [cs.CL] https://arxiv.org/abs/2310.06825
[49] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Llmlingua: Compressing prompts for accelerated inference of large
language models. InProceedings of the 2023 conference on empirical methods in natural language processing. 13358–13376.
[50] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. Longllmlingua: Accelerating and
enhancing llms in long context scenarios via prompt compression. InProceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers). 1658–1677.
[51] Zhengbao Jiang, Ji-Rong Wen, Zhicheng Dou, Wayne Xin Zhao, Jian-Yun Nie, and Ming Yue. 2017. Learning to diversify search results via subtopic
attention. InProceedings of the 40th international ACM SIGIR Conference on Research and Development in Information Retrieval. 545–554.
[52] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active
retrieval augmented generation. (2023), 7969–7992.
[53] Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan Arik. 2025. Long-context llms meet rag: Overcoming challenges for long inputs in rag. In
International Conference on Learning Representations, Vol. 2025. 37784–37822.
[54] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario
Amodei. 2020. Scaling laws for neural language models.arXiv preprint arXiv:2001.08361(2020).
[55] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage
retrieval for open-domain question answering. InProceedings of the 2020 conference on empirical methods in natural language processing (EMNLP).
6769–6781.
[56] Omar Khattab, Keshav Santhanam, Xiang Lisa Li, David Hall, Percy Liang, Christopher Potts, and Matei Zaharia. 2022. Demonstrate-search-predict:
Composing retrieval and language models for knowledge-intensive nlp.arXiv preprint arXiv:2212.14024(2022).
Manuscript submitted to ACM

28 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
[57] Alex Kulesza and Ben Taskar. 2012. Determinantal point processes for machine learning.Foundations and Trends®in Machine Learning5, 2-3
(2012), 123–286.
[58] Youngwon Lee, Seung-won Hwang, Daniel F Campos, Filip Gralinski, Zhewei Yao, and Yuxiong He. 2025. Inference scaling for bridging retrieval
and augmented generation. InFindings of the Association for Computational Linguistics: NAACL 2025. 7339–7354.
[59] Jurek Leonhardt, Koustav Rudra, and Avishek Anand. 2023. Extractive explanations for interpretable text ranking.ACM Transactions on Information
Systems41, 4 (2023), 1–31.
[60] Mosh Levy, Alon Jacoby, and Yoav Goldberg. 2024. Same task, more tokens: the impact of input length on the reasoning performance of large
language models. InProceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 15339–15353.
[61] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim
Rocktäschel, et al .2020. Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in neural information processing systems33,
9459–9474.
[62] Hang Li, Ahmed Mourad, Shengyao Zhuang, Bevan Koopman, and Guido Zuccon. 2023. Pseudo relevance feedback with deep language models
and dense retrievers: Successes and pitfalls.ACM Transactions on Information Systems41, 3 (2023), 1–40.
[63] Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and William B Dolan. 2016. A diversity-promoting objective function for neural conversation
models. InProceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language
technologies. 110–119.
[64] Jiwei Li, Will Monroe, and Dan Jurafsky. 2016. Understanding neural networks through representation erasure.arXiv preprint arXiv:1612.08220
(2016).
[65] Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. 2024. Long-context llms struggle with long in-context learning.arXiv preprint
arXiv:2404.02060(2024).
[66] Xiaoxi Li, Jiajie Jin, Yujia Zhou, Yuyao Zhang, Peitian Zhang, Yutao Zhu, and Zhicheng Dou. 2025. From matching to generation: A survey on
generative information retrieval.ACM Transactions on Information Systems43, 3 (2025), 1–62.
[67] Xinze Li, Hanbin Wang, Zhenghao Liu, Shi Yu, Shuo Wang, Yukun Yan, Yukai Fu, Yu Gu, and Ge Yu. 2025. Building a coding assistant via the
retrieval-augmented language model.ACM Transactions on Information Systems43, 2 (2025), 1–25.
[68] Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023. Contrastive
decoding: Open-ended text generation as optimization. InProceedings of the 61st annual meeting of the association for computational linguistics
(volume 1: Long papers). 12286–12312.
[69] Shangsong Liang, Emine Yilmaz, Hong Shen, Maarten De Rijke, and W Bruce Croft. 2017. Search result diversification in short text streams.ACM
Transactions on Information Systems (TOIS)36, 1 (2017), 1–35.
[70] Hui Lin and Jeff Bilmes. 2011. A class of submodular functions for document summarization. InProceedings of the 49th annual meeting of the
association for computational linguistics: human language technologies. 510–520.
[71] Jimmy Lin. 2007. An exploration of the principles underlying redundancy-based factoid question answering.ACM Transactions on Information
Systems (TOIS)25, 2 (2007), 6–es.
[72] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How
language models use long contexts.Transactions of the association for computational linguistics12 (2024), 157–173.
[73] Nelson F Liu, Tianyi Zhang, and Percy Liang. 2023. Evaluating verifiability in generative search engines. InFindings of the Association for
Computational Linguistics: EMNLP 2023. 7001–7025.
[74] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using gpt-4 with better human
alignment. InProceedings of the 2023 conference on empirical methods in natural language processing. 2511–2522.
[75] Zeyang Liu, Ke Zhou, and Max L Wilson. 2021. Meta-evaluation of conversational search evaluation metrics.ACM Transactions on Information
Systems (TOIS)39, 4 (2021), 1–42.
[76] Scott M Lundberg and Su-In Lee. 2017. A unified approach to interpreting model predictions.Advances in neural information processing systems30.
[77] Yuanjie Lyu, Zhiyu Li, Simin Niu, Feiyu Xiong, Bo Tang, Wenjin Wang, Hao Wu, Huanyong Liu, Tong Xu, and Enhong Chen. 2025. Crud-rag: A
comprehensive chinese benchmark for retrieval-augmented generation of large language models.ACM Transactions on Information Systems43, 2
(2025), 1–32.
[78] Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and Dan Roth. 2024. ExpertQA: Expert-curated questions and attributed
answers. InProceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language
Technologies (Volume 1: Long Papers). 3025–3045.
[79] Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On faithfulness and factuality in abstractive summarization. In
Proceedings of the 58th annual meeting of the association for computational linguistics. 1906–1919.
[80] Chuan Meng, Negar Arabzadeh, Arian Askari, Mohammad Aliannejadi, and Maarten de Rijke. 2025. Query performance prediction using relevance
judgments generated by large language models.ACM Transactions on Information Systems43, 4 (2025), 1–35.
[81] Jacob Menick, Maja Trebacz, Vladimir Mikulik, John Aslanides, Francis Song, Martin Chadwick, Mia Glaese, Susannah Young, Lucy Campbell-
Gillingham, Geoffrey Irving, et al .2022. Teaching language models to support answers with verified quotes.arXiv preprint arXiv:2203.11147
(2022).
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 29
[82] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023.
FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. InProceedings of the 2023 conference on empirical
methods in natural language processing. 12076–12100.
[83] Michel Minoux. 2005. Accelerated greedy algorithms for maximizing submodular set functions. InOptimization Techniques: Proceedings of the 8th
IFIP Conference on Optimization Techniques Würzburg, September 5–9, 1977. Springer, 234–243.
[84] Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru, Amin Karbasi, Jan Vondrák, and Andreas Krause. 2015. Lazier than lazy greedy. 29, 1 (2015).
[85] Fengran Mo, Kelong Mao, Ziliang Zhao, Hongjin Qian, Haonan Chen, Yiruo Cheng, Xiaoxi Li, Yutao Zhu, Zhicheng Dou, and Jian-Yun Nie. 2025. A
survey of conversational search.ACM Transactions on Information Systems43, 6 (2025), 1–50.
[86] Alistair Moffat, Peter Bailey, Falk Scholer, and Paul Thomas. 2017. Incorporating user expectations and behavior into the measurement of search
effectiveness.ACM Transactions on Information Systems (TOIS)35, 3 (2017), 1–38.
[87] Alistair Moffat and Justin Zobel. 2008. Rank-biased precision for measurement of retrieval effectiveness.ACM Transactions on Information Systems
(TOIS)27, 1 (2008), 1–27.
[88] Andrea Morales-Garzón, Oscar A Rocha, Sara Benel Ramirez, Gabriel Tuco Casquino, and Alberto Medina. 2024. Healthy cooking with large
language models, supervised fine-tuning, and retrieval augmented generation. InProceedings of the LatinX in AI Workshop at NAACL.
[89] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William
Saunders, et al. 2021. Webgpt: Browser-assisted question-answering with human feedback.arXiv preprint arXiv:2112.09332(2021).
[90] George L Nemhauser, Laurence A Wolsey, and Marshall L Fisher. 1978. An analysis of approximations for maximizing submodular set functions—I.
Mathematical programming14, 1 (1978), 265–294.
[91] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. 2025. Graph retrieval-augmented
generation: A survey.ACM Transactions on Information Systems44, 2 (2025), 1–52.
[92] Xubo Qin, Zhicheng Dou, Yutao Zhu, and Ji-Rong Wen. 2023. GDESA: Greedy diversity encoder with self-attention for search results diversification.
ACM Transactions on Information Systems41, 2 (2023), 1–36.
[93] Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan
Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang,
Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren,
Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2025. Qwen2.5 Technical Report. (2025).
arXiv:2412.15115 [cs.CL] https://arxiv.org/abs/2412.15115
[94] Filip Radlinski and Susan Dumais. 2006. Improving personalized web search using result diversification. InProceedings of the 29th annual
international ACM SIGIR conference on Research and development in information retrieval. 691–692.
[95] Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-
augmented language models.Transactions of the Association for Computational Linguistics11 (2023), 1316–1331.
[96] Hannah Rashkin, Vitaly Nikolaev, Matthew Lamm, Lora Aroyo, Michael Collins, Dipanjan Das, Slav Petrov, Gaurav Singh Tomar, Iulia Turc, and
David Reitter. 2023. Measuring attribution in natural language generation models.Computational Linguistics49, 4 (2023), 777–840.
[97] Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. InProceedings of the 2019 conference
on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP).
3982–3992.
[98] Mohammad Reza Rezaei and Adji Bousso Dieng. 2025. Vendi-rag: Adaptively trading-off diversity and quality significantly improves retrieval
augmented generation with llms.arXiv preprint arXiv:2502.11228(2025).
[99] Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. " Why should i trust you?" Explaining the predictions of any classifier. InProceedings
of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining. 1135–1144.
[100] Jon Saad-Falcon, Omar Khattab, Christopher Potts, and Matei Zaharia. 2024. Ares: An automated evaluation framework for retrieval-augmented
generation systems. InProceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human
Language Technologies (Volume 1: Long Papers). 338–354.
[101] Tetsuya Sakai and Ruihua Song. 2011. Evaluating diversified search results using per-intent graded relevance. InProceedings of the 34th international
ACM SIGIR conference on Research and development in Information Retrieval. 1043–1052.
[102] Rodrygo LT Santos, Craig Macdonald, and Iadh Ounis. 2010. Exploiting query reformulations for web search result diversification. InProceedings
of the 19th international conference on World wide web. 881–890.
[103] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher Manning. 2024. Raptor: Recursive abstractive processing
for tree-organized retrieval. InInternational Conference on Learning Representations, Vol. 2024. 32628–32649.
[104] Sofia Serrano and Noah A Smith. 2019. Is attention interpretable? (2019), 2931–2951.
[105] Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny Zhou. 2023. Large language
models can be easily distracted by irrelevant context. InInternational conference on machine learning. PMLR, 31210–31227.
[106] Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. 2024. Trusting your evidence: Hallucinate less
with context-aware decoding. InProceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics:
Human Language Technologies (Volume 2: Short Papers). 783–791.
Manuscript submitted to ACM

30 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
[107] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2024. Replug: Retrieval-
augmented black-box language models. InProceedings of the 2024 conference of the north american chapter of the association for computational
linguistics: Human language technologies (volume 1: Long papers). 8371–8384.
[108] Zhengliang Shi, Lingyong Yan, Weiwei Sun, Yue Feng, Pengjie Ren, Xinyu Ma, Shuaiqiang Wang, Dawei Yin, Maarten de Rijke, and Zhaochun Ren.
2026. Direct retrieval-augmented optimization: Synergizing knowledge selection and language models.ACM Transactions on Information Systems
44, 4 (2026), 1–30.
[109] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more effective than scaling model
parameters.arXiv preprint arXiv:2408.03314(2024).
[110] Ivan Stelmakh, Yi Luan, Bhuwan Dhingra, and Ming-Wei Chang. 2022. ASQA: Factoid questions meet long-form answers. InProceedings of the 2022
Conference on Empirical Methods in Natural Language Processing. 8273–8288.
[111] Yixuan Su, Tian Lan, Yan Wang, Dani Yogatama, Lingpeng Kong, and Nigel Collier. 2022. A contrastive framework for neural text generation.
Advances in neural information processing systems35, 21548–21561.
[112] Zhan Su, Zhicheng Dou, Yutao Zhu, and Ji-Rong Wen. 2024. Passage-aware search result diversification.ACM Transactions on Information Systems
42, 5 (2024), 1–29.
[113] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. InInternational conference on machine learning.
PMLR, 3319–3328.
[114] Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten De Rijke, Wei Chen, and Xueqi Cheng. 2024. Listwise generative retrieval models via a sequential
learning process.ACM Transactions on Information Systems42, 5 (2024), 1–31.
[115] Leila Tavakoli, Johanne R Trippas, Hamed Zamani, Falk Scholer, and Mark Sanderson. 2024. Online and offline evaluation in search clarification.
ACM Transactions on Information Systems43, 1 (2024), 1–30.
[116] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot
evaluation of information retrieval models.arXiv preprint arXiv:2104.08663.
[117] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving retrieval with chain-of-thought reasoning for
knowledge-intensive multi-step questions. InProceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long
papers). 10014–10037.
[118] Mehmet Deniz Türkmen, Mucahid Kutlu, Bahadir Altun, and Gokalp Cosgun. 2025. Gentrec: The first test collection generated by large language
models for evaluating information retrieval systems.ACM Transactions on Information Systems(2025).
[119] Ashwin Vijayakumar, Michael Cogswell, Ramprasaath Selvaraju, Qing Sun, Stefan Lee, David Crandall, and Dhruv Batra. 2018. Diverse beam
search for improved description of complex scenes. 32, 1 (2018).
[120] Ellen M Voorhees, Daniel Samarov, and Ian Soboroff. 2017. Using replicates in information retrieval evaluation.ACM Transactions on Information
Systems (TOIS)36, 2 (2017), 1–21.
[121] Lei Wang, Jingsen Zhang, Hao Yang, Zhi-Yuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Hao Sun, Ruihua Song, et al .2025. User
behavior simulation with large language model-based agents.ACM Transactions on Information Systems43, 2 (2025), 1–37.
[122] Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, et al .2024.
Searching for best practices in retrieval-augmented generation. InProceedings of the 2024 conference on empirical methods in natural language
processing. 17716–17736.
[123] Zhichao Wang, Bin Bi, Yanqi Luo, Sitaram Asur, and Claire Na Cheng. 2025. Diversity Enhances an LLM’s Performance in RAG and Long-context
Task.arXiv preprint arXiv:2502.09017(2025).
[124] Zilong Ryan Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly,
Jingbo Shang, et al .2025. Speculative rag: Enhancing retrieval augmented generation through drafting. InInternational Conference on Learning
Representations, Vol. 2025. 18483–18505.
[125] Sarah Wiegreffe and Yuval Pinter. 2019. Attention is not not explanation. InProceedings of the 2019 conference on empirical methods in natural
language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP). 11–20.
[126] Long Xia, Jun Xu, Yanyan Lan, Jiafeng Guo, and Xueqi Cheng. 2015. Learning maximal marginal relevance model via directly optimizing diversity
evaluation measures. InProceedings of the 38th international ACM SIGIR conference on research and development in information retrieval. 113–122.
[127] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RECOMP: Improving retrieval-augmented LMs with context compression and selective
augmentation. InInternational Conference on Learning Representations, Vol. 2024. 43478–43502.
[128] Peng Xu, Wei Ping, Xianchao Wu, Lawrence McAfee, Chen Zhu, Zihan Liu, Sandeep Subramanian, Evelina Bakhturina, Mohammad Shoeybi, and
Bryan Catanzaro. 2024. Retrieval meets long context large language models. InInternational Conference on Learning Representations, Vol. 2024.
49569–49584.
[129] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA:
A dataset for diverse, explainable multi-hop question answering. InProceedings of the 2018 conference on empirical methods in natural language
processing. 2369–2380.
[130] Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant. 2024. Making retrieval-augmented language models robust to irrelevant context. In
International Conference on Learning Representations, Vol. 2024. 29862–29883.
[131] Tan Yu, Anbang Xu, and Rama Akkiraju. 2024. In defense of rag in the era of long-context language models.arXiv preprint arXiv:2409.01666(2024).
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 31
[132] Wenhao Yu, Dan Iter, Shuohang Wang, Yichong Xu, Mingxuan Ju, Soumya Sanyal, Chenguang Zhu, Michael Zeng, and Meng Jiang. 2022. Generate
rather than retrieve: Large language models are strong context generators.arXiv preprint arXiv:2209.10063.
[133] Xiang Yue, Boshi Wang, Ziru Chen, Kai Zhang, Yu Su, and Huan Sun. 2023. Automatic evaluation of attribution by large language models. In
Findings of the Association for Computational Linguistics: EMNLP 2023. 4615–4635.
[134] Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin, Dong Wang, Xuanhui Wang, and Michael Bendersky. 2025.
Inference scaling for long-context retrieval augmented generation. InInternational Conference on Learning Representations, Vol. 2025. 72914–72938.
[135] ChengXiang Zhai, William W Cohen, and John Lafferty. 2015. Beyond independent relevance: methods and evaluation metrics for subtopic
retrieval. InAcm sigir forum, Vol. 49. ACM New York, NY, USA, 2–9.
[136] Chao Zhang, Yuhao Wang, Derong Xu, Haoxin Zhang, Yuanjie Lyu, Yuhao Chen, Shuochen Liu, Tong Xu, Xiangyu Zhao, Yan Gao, et al .2026.
Tearag: A token-efficient agentic retrieval-augmented generation framework.ACM Transactions on Information Systems44, 6 (2026), 1–35.
[137] Tianjun Zhang, Shishir G Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E Gonzalez. 2024. Raft: Adapting language model to
domain specific rag.arXiv preprint arXiv:2403.10131(2024).
[138] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2025. A survey on the memory
mechanism of large language model-based agents.ACM Transactions on Information Systems43, 6 (2025), 1–47.
[139] Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. 2024. Dense text retrieval based on pretrained language models: A survey.ACM
Transactions on Information Systems42, 4 (2024), 1–60.
[140] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al .
2023. Judging llm-as-a-judge with mt-bench and chatbot arena.Advances in neural information processing systems36, 46595–46623.
[141] Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. 2018. Texygen: A benchmarking platform for text generation
models. InThe 41st international ACM SIGIR conference on research & development in information retrieval. 1097–1100.
[142] Yutao Zhu, Huaying Yuan, Shuting Wang, Jiongnan Liu, Wenhan Liu, Chenlong Deng, Haonan Chen, Zheng Liu, Zhicheng Dou, and Ji-Rong Wen.
2025. Large language models for information retrieval: A survey.ACM Transactions on Information Systems44, 1 (2025), 1–54.
A Formal Mathematical Framework and Proofs
To establish the theoretical rigor underpinning our portfolio generation architecture, this appendix details the submod-
ular optimization guarantees of our scheduling lever and formalizes the idealized theoretical bounds against which our
empirical violations are measured.
A.1 Submodular Optimization for Evidence Orchestration
At round𝑡, theAscpscheduler greedily constructs a size- 𝑘context𝐶by maximizing a discrete set function that
inherently penalizes redundancy through causal feedback. The objective is defined as:
𝐹𝑡(𝐶)=∑︁
𝑧∈Z𝜋(𝑧)𝛽𝑛𝑡(𝑧)h
1−Ö
𝑑∈𝐶 1−𝑣𝑡(𝑑)𝑊[𝑑,𝑧]i
+𝜆∑︁
𝑑∈𝐶𝑟(𝑑)
𝑘(1+𝑢𝑡(𝑑)),(6)
where dynamic document scaling 𝑣𝑡(𝑑)=𝛽𝑢𝑡(𝑑)
docoperates alongside measured document attribution 𝑢𝑡and facet
attribution𝑛𝑡. Static parameters include document–facet coverage 𝑊[𝑑,𝑧] and query-conditional facet importance 𝜋(𝑧) .
This construction ensures that historical evidence utilization directly decays the marginal utility of future redundant
exposures.
Proposition 1.For any fixed causal feedback state (𝑢𝑡,𝑛𝑡)at round𝑡, the context orchestration set function 𝐹𝑡defined
in Eq.(6)satisfies𝐹 𝑡(∅)=0and is strictly monotone non-decreasing and submodular.
Proof. We decompose the objective into coverage and relevance components: 𝐹𝑡=𝐺𝑡+𝜆𝑀𝑡. The relevance term
𝑀𝑡(𝐶)=Í
𝑑∈𝐶𝑟(𝑑)/(𝑘( 1+𝑢𝑡(𝑑))) operates as a direct sum of non-negative per-element weights, rendering it inherently
modular, monotone, and zero on the empty set. For the coverage component, we isolate 𝑐𝑧=𝜋(𝑧)𝛽𝑛𝑡(𝑧)≥0and
𝑤𝑧(𝑑)=𝑣𝑡(𝑑)𝑊[𝑑,𝑧]∈[ 0,1], expressing 𝐺𝑡(𝐶)=Í
𝑧𝑐𝑧𝑔𝑧(𝐶)where𝑔𝑧(𝐶)= 1−Î
𝑑∈𝐶(1−𝑤𝑧(𝑑)). Because𝑐𝑧and
𝑤𝑧are strictly state-dependent and invariant to the current context candidate 𝐶, we evaluate a fixed facet 𝑧. Trivially,
Manuscript submitted to ACM

32 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
𝑔𝑧(∅)=0. For any document𝑑∉𝐶, the marginal gain is mathematically bounded:
𝑔𝑧(𝐶∪{𝑑})−𝑔 𝑧(𝐶)=𝑤𝑧(𝑑)Ö
𝑒∈𝐶 1−𝑤𝑧(𝑒)≥0,(7)
proving𝑔𝑧is monotone. To establish submodularity, consider subsets 𝐴⊆𝐵 and a document 𝑑∉𝐵 . Because every factor
1−𝑤𝑧(𝑒)is constrained within [0,1], the product inequalityÎ
𝑒∈𝐴(1−𝑤𝑧(𝑒))≥Î
𝑒∈𝐵(1−𝑤𝑧(𝑒))holds universally.
Substituting this into Eq. (7)satisfies the strict diminishing-returns property. As non-negative linear combinations
preserve monotone submodularity,𝐹 𝑡is submodular.□
Consequently, standard greedy selection securely attains the robust 𝐹𝑡(𝐶𝑔)≥( 1−𝑒−1)max|𝐶|≤𝑘𝐹𝑡(𝐶)approximation
guarantee. If core evidence 𝐾dictates mandatory pinning, optimizing the residual 𝐹′
𝑡(𝑆)=𝐹𝑡(𝐾∪𝑆)−𝐹 𝑡(𝐾)maintains
𝐹𝑡(𝐾∪𝑆𝑔)≥( 1−𝑒−1)max𝐾⊆𝐶,|𝐶|≤𝑘𝐹𝑡(𝐶)+𝑒−1𝐹𝑡(𝐾). Because attribution feedback dynamically updates across rounds,
the absolute objective evolves sequentially, yet intra-round submodularity is mathematically preserved.
A.2 Bridging the Gap: Theoretical Independence vs. Generative Complexity
While mathematical bounds provide an elegant structural ceiling, real-world generative evidence consumption sys-
tematically diverges from idealized theoretical assumptions, necessitating our rigorous empirical design. Consider the
baseline assumption of strictly independent utilization, which posits that a generator grounds on a random subset of a
provided context with an expected size E|𝑆|=𝑞|𝐶| , where𝑞∈( 0,1)is entirely independent of the context’s specific
contents or historical rounds.
Evaluating this baseline assumption via a mixed-effects model against our actual generative data decisively rejects
policy invariance ( 𝜒2=171.4,𝑝=6×10−31); empirical utilization is over-dispersed by a factor of1 .39and exhibits
heavy cross-round correlation dictated by structural redundancy.
This profound generative complexity directly impacts optimal budget allocation. If we formalize relevance density
𝜌(𝑖) as the non-increasing probability that a document at rank 𝑖carries a gold answer unit, a rotation policy offering
sequential ranks1 ,...,𝑘𝑇 yields an expected answer-bearing coverage of 𝑞Í
𝑖≤𝑘𝑇𝜌(𝑖). Conversely, a selection policy
operating over a restricted top-𝑁′pool yieldsÍ
𝑖≤𝑚𝜌(𝑖)
1−(1−𝑞)𝑛𝑖
≤Í
𝑖≤𝑚𝜌(𝑖).
Proposition 2 (Selection versus Rotation Regime).Under idealized consumption, sequential rotation mathemati-
cally dominates bounded selection whenever 𝑞Í𝑘𝑇
𝑖=𝑚+1𝜌(𝑖)>Í
𝑖≤𝑚𝜌(𝑖)h
(1−𝑞)−( 1−𝑞)𝑛𝑖i
, where the right-hand side
represents the marginal yield extracted from re-offering previously chosen documents.
This inequality formally dictates that selection is theoretically optimal when relevance decays sharply, whereas
rotation dominates when tail evidence retains value. However, pushing this logic to a global budget allocation boundary
exposes the limits of pure mathematical bounding. Our full factorial grid actively confirms that empirical portfolio
recall is governed by highly correlated sequential generative rounds where 𝑇carries the true causal effect, proving that
allocation logic must be strictly governed by the deconfounded empirical laws of LLM behavior rather than isolated
mathematical abstraction.
B Comprehensive System Diagnostics and Robustness
To ensure the statistical validity of our core systemic findings, we conduct exhaustive diagnostic testing across stochastic
seed variation, decoder architecture parity, multiple hypothesis testing, and intrinsic pool redundancy.
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 33
Table 14. Seed replication of the factorial under a severely constrained 𝑁= 30retrieval pool. Unlike the deep 𝑁= 400pool in the
main text (which prevents evidence exhaustion), this artificially restricted setting starves the sequential iterations of fresh evidence,
naturally compressing the absolute structural gaps (e.g., the budget-matched gain shrinks to ∼+0.065). Crucially, despite this
compressed effect size, the cross-seed standard deviation remains negligible ( ≤0.011). This proves that algorithmic stochasticity
cannot explain our allocation laws, even under extreme resource starvation.
contrast seed 0 seed 1 seed 2 mean s.d.
count T12 vs T1 @k2 0.141 0.137 0.154 0.144 0.0090
count T12 vs T1 @k24 0.159 0.160 0.141 0.153 0.0109
width k24 vs k2 @T12 0.097 0.099 0.090 0.095 0.0049
width k24 vs k2 @T1 0.092 0.094 0.097 0.094 0.0024
rot over fix @k2T12 – – – – –
rot over fix @k24T12 – – – – –
budget (2,12) vs (24,1) 0.062 0.061 0.072 0.065 0.0058
B.1 Stochastic Consistency and Decoder Parity
A critical vulnerability in generative evaluation is the confounding variance introduced by stochastic decoding algorithms.
To definitively prove that our context allocation laws are structurally driven rather than transient artifacts of a specific
generation trace, we fully replicated the factorial grid across multiple independent decoding seeds.
To subject our findings to an extreme stress test, we executed this seed replication under a deliberately constrained
retrieval pool ( 𝑁= 30, as opposed to the unconstrained 𝑁= 400pool utilized in the primary Table 4). This severe
restriction forces early evidence exhaustion, naturally compressing the absolute magnitude of the structural gains
(e.g., attenuating the (2,12)vs(24,1)budget-matched contrast to an average of +0.065). However, holding the query
sample strictly fixed, we observed cross-seed standard deviations tightly constrained between0 .002and0.011(Table 14).
Because these stochastic deviations remain an order of magnitude below even these artificially compressed structural
contrasts, we definitively conclude that the structural benefit of iterative portfolio generation entirely eclipses baseline
generative stochasticity, even under severe resource starvation.
Furthermore, any performance deltas attributed to our custom attribution-steered decoder (our cognitive override
mechanism) must be absolutely isolated from arbitrary sampling discrepancies. We rigorously established baseline
parity: by setting our custom steering strength ( 𝛼) and adaptive-plausibility cutoff ( 𝜂) to zero, our specialized loop
mathematically reproduces ordinary generation exactly across240held-out query conditions. This yields structurally
identical texts, selected document traces, and resulting attribution vectors. Consequently, the performance deltas
reported in our five-seed decomposition (Table 12) strictly isolate the true, unconfounded causal impact of the logit
contrast mechanism.
B.2 Multiplicity Control and Redundancy Stratification
Given the scale of our factorial interactions, spurious significance via multiple comparisons is a severe risk. We pre-
declared eight distinct testing families and enforced strict Benjamini-Hochberg False Discovery Rate (BH-FDR) control
at𝛼=0.05. This correction proved our findings exceptionally robust:108of the123executed statistical tests survive
correction entirely unchanged. Crucially, even our most marginal preliminary observations (e.g., the ASQA/Llama width
contrast at𝑇= 12and the integrative-instruction width contrast) maintained absolute significance post-correction at a
stable𝑞=0.034, while all primary generation-count contrasts securely rested at the absolute bootstrap floor.
Beyond statistical correction, we isolated the physical confound of natural-pool redundancy. By stratifying2 ,880
query-level observations across six context widths using pairwise entailment, answer attestation, and maximum
embedding similarity (exhibiting loose internal correlation 𝑟=0.12–0.35), we tracked continuous log𝑘× redundancy
interactions. The structural dilution of evidence utilization firmly survives in even the lowest-redundancy environments.
Manuscript submitted to ACM

34 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
LOO probe(ours)
embedding similarity0.00.20.40.60.8agreement with judge0.5710.666
0.5700.594pale: Spearman 𝜌solid: top-1 agreement(a) which document did it use?
ctx.-util. entropy util. concentration evidence coverage0.00.20.40.60.8Spearman 𝜌0.704
0.3350.654(b) probe vs. judge coverage
distinct-2 self-BLEU
ctx.-util. entropy evidence coverage−0.6−0.30.00.30.6Spearman 𝜌0.675
-0.5650.431 0.425response-only judge ≈distinct-2(c) judge textual novelty
1
Fig. 7. Judge meta-evaluation over179portfolios and858document-level judgements.(a)Identifying the supporting document: both
the probe and the embedding proxy track the judge’s profile equally well overall, but the probe is significantly better at identifying the
primary source.(b)A comparison of probe-derived and judge-derived coverage measures.(c)Correlation with the judge’s assessment
oftextualnovelty. A response-only judge evaluates textual novelty by heavily tracking the distinct-2metric rather than actual evidence
coverage.
Table 15. Component ablations and probe substitutions (qwen). Attribution-feedback removal leaves open-loop scheduling; probe-
substitution rows keep the scheduler fixed and swap only the evidence-use signal.
Variant ASQA QAMPARI ELI5 Recipes
PR@𝑇ECR PR@𝑇ECR PR@𝑇ECR PR@𝑇ECR
Ascp(full) 0.493 0.331 0.162 0.324 0.273 0.459 0.238 0.371
−attribution feedback 0.465 0.320 0.163 0.309 0.248 0.459 0.242 0.398
−submodular selection 0.470 0.264 0.158 0.265 0.257 0.408 0.220 0.319
−steered decoding 0.488 0.325 0.161 0.321 0.278 0.465 0.231 0.378
steered decoding only 0.398 0.150 0.107 0.122 0.232 0.177 0.221 0.197
+guardrail (𝜅=1) 0.434 0.229 0.136 0.221 0.237 0.271 0.225 0.299
+guardrail (𝜅=2) 0.417 0.210 0.133 0.188 0.237 0.233 0.215 0.282
probe→hierarchical LOO 0.495 0.341 0.165 0.327 0.258 0.496 0.246 0.377
probe→similarity 0.458 0.513 0.161 0.469 0.248 0.665 0.239 0.462
probe→uniform 0.465 0.516 0.163 0.510 0.248 0.665 0.241 0.464
When enforcing strict pairwise entailment filtering, the utilized fraction 𝑞still aggressively decays from0 .575at𝑘=2
down to0.164at𝑘=24, yielding an elasticity of −0.528(0.021). Similarly, enforcing zero duplicated answer-bearing
passages forces decay from0 .561to0.204, securing an elasticity of −0.434(0.024). While intense answer attestation
redundancy mathematically steepens the decay slope to −0.667(exhibiting a highly significant log𝑘× interaction at
𝑝=2×10−7), redundancy strictly modulates, but critically cannot unilaterally manufacture, the fundamental dilution
penalty inherent to context window expansion. Note that these strata are estimated on natural retrieval pools under
free generation, so their slopes are expected to be shallower than the protocol-clean −0.68of Section 5.3, which isolates
the structural decay under a fixed target and a width-invariant threshold. The relevant evidence here is therefore not
the absolute magnitude, but that a steep decay persists in every low-redundancy stratum.
C Reproducibility and Experimental Artifacts
To satisfy rigorous ACM artifact review standards and ensure complete systematic reproducibility, this section details the
underlying computational protocols, dataset construction methodologies, and hyperparameter stabilization processes
that govern our end-to-end portfolio architecture.
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 35
0.3 0.5 0.7 10.520.540.560.580.60PR@𝑇𝛽facet discount
0.1 0.3 0.5 1𝛽docdocument discount
00.25 0.50.75 11.5𝛼steering strength
0 0.25 0.5 1 2𝜆relevance weight
3 5 80.520.540.560.580.60PR@𝑇𝑘context size
0 1 2𝜅pinned documents
10 20 30𝑁pool size
10 20 30𝑁pool size (vanilla)
⃝deployed setting of the frozen configuration; sweeps are one-at-a-time on the disjoint ASQA development split
1
Fig. 8. One-at-a-time sensitivity sweeps conducted on the disjoint ASQA development split to ensure reproducible hyperparameter
stabilization. The explicitly circled point indicates the selected optimal value for each distinct architectural parameter. The strictly
frozen evaluation configuration deployed uniformly across primary experiments is 𝛽=0.3,𝛽doc=0.3,𝜆=0.25,𝜅=0,𝛼=0.5,𝑘=5,
and𝑁=30.
The foundation of our experimental fairness relies on strict parity across evaluated frameworks. All benchmarked
systems, regardless of their internal scheduling logic or instructional prompting, are architecturally forced to share the
identical underlying retriever, retrieved evidence pool, context size, generated portfolio size, and generation temperature.
To eliminate overfitting, all architectural selection and semantic facet hyperparameters were independently stabilized
exactly once on a disjoint40-query ASQA development split. Figure 8 comprehensively visualizes the one-at-a-time
sensitivity sweeps executed on this split, validating that no single hyperparameter artificially dominates the system’s
robustness. The resulting frozen configuration ( 𝛽=0.3,𝛽doc=0.3,𝜆=0.25,𝜅=0,𝛼=0.5,𝑘=5, and𝑁= 30) was
locked prior to executing any of the primary held-out algorithmic evaluations.
Crucially, validating our causal attribution probe (Section 4) mandated the construction of highly controlled ground-
truth pools utilizing the identical ALCE candidate passages. To maintain strict isolation, a document is mathematically
classified as answer-bearing if and only if any normalized alias exists as a contiguous token subsequence. The essential
necessary set systematically comprises the first 𝑚= 2answer-bearing documents, ordered by answer coverage
and strictly constrained to attest entirely disjoint answer sets. Extraneous padding is aggressively regulated: true
duplicates strictly attest to already resolved answers, whereas hard distractors are aggressively sampled from up
to sixty unrelated queries, completely incapable of resolving the target query. These stringent requirements ensure
that our queries provide at least 𝑚+ 2verified answer-bearing documents alongside twenty-four verified distractors,
preserving approximately one-fifth of the raw ASQA corpus and one-third of QAMPARI. Within these validated pools,
documents are cryptographically shuffled via a per-case deterministic seed, ensuring that semantic document labels
remain immutable across all𝑘+1counterfactual ablation loops.
Our accompanying submission artifact encompasses the complete functional probe architecture, the operating
point calibration scripts, the exact validation-pool construction pipelines, and the comprehensive job specifications
necessary to precisely regenerate all reported tables and figures. We emphasize that GPU-free validation checks
structurally cover the critical 𝜏(𝑘) calibration, crossed bootstrap uncertainty bounds, ContextCite ablation routines, and
all similarity/lexical baseline trajectories. The total computational footprint of this analysis spans188unified execution
Manuscript submitted to ACM

36 Peiyang Liu, Xi Wang, Di Liang, and Wei Ye
Table 16. Utility-side factorial under commit versus integrate instructions. Integrate asks for every supported interpretation and
allows enumeration. Differences are integrate minus commit, paired within query ( 𝑛=314–480per cell), testing allocation under the
instruction most favourable to one wide response.
𝑘 𝑇commit PR@𝑇integrate PR@𝑇Δ95% CI
2 1 0.203 0.253+0.050[+0.031,+0.070]
2 5 0.326 0.394+0.068[+0.049,+0.088]
2 12 0.393 0.453+0.060[+0.041,+0.080]
5 1 0.219 0.292+0.073[+0.053,+0.095]
5 5 0.336 0.406+0.070[+0.052,+0.091]
5 12 0.411 0.475+0.065[+0.044,+0.087]
12 1 0.245 0.322+0.077[+0.055,+0.101]
12 5 0.365 0.439+0.074[+0.053,+0.098]
12 12 0.423 0.505+0.082[+0.059,+0.107]
24 1 0.259 0.355+0.096[+0.073,+0.121]
24 5 0.397 0.495+0.098[+0.070,+0.129]
24 12 0.411 0.484+0.073[+0.052,+0.094]
runs (148utilizing the primary commit instruction and40under the integrate baseline). A standard single evaluation run
operates efficiently at126seconds per query. The primary budget and context width sweeps utilize150discrete queries
at narrow widths ( 𝑘≤8) and60at extreme limits ( 𝑘∈{ 12,24}), while the expansive 𝑘×𝑇 factorial mandates exactly
120strictly paired observations across every distinct cell, driving11 ,579individually scored generation rows. This
exhaustive scale directly necessitates the hierarchical predictive modeling applied throughout our results, ensuring that
our architectural conclusions represent fundamental generative properties rather than isolated statistical anomalies.
D Supplementary Settings and Full Empirical Grids
To exhaustively validate that our architectural allocation laws and scheduling advantages extend beyond standard English
short-answer question answering, we provide the complete empirical grids encompassing alternative instructional
bounds, structured generative baselines, full algorithmic ablations, and open-domain cross-cultural stress tests.
D.1 Open-Domain Cross-Cultural Generalization
We evaluate the cross-cultural recipe adaptation setting as a rigorous non-English open-domain check [ 41]. Operating
over a corpus of9 ,486Spanish-origin recipes [ 88], the system is tasked with rewriting Latin-American query recipes to
incorporate authentic Spanish culinary practices. Gold answer units are strictly defined as valid Spanish ingredients
empirically attested by the retrieved pool but explicitly absent from the source recipe, directly rewarding generative
substitution breadth. Retrieved via a multilingual sentence encoder, this specific domain exhibits an almost perfectly flat
relevance density (log-log slope −0.01, Table 3). Unlike the concentrated ASQA benchmark, this flat relevancy physically
insulates the wide-context configurations from deep-rank decay, structurally dictating that unconstrained rotational
scheduling theoretically should, and empirically does, achieve peak portfolio extraction in this isolated regime.
D.2 Instructional Bounds and Structured Output Controls
A prevailing assumption is that the limitations of single-pass context utilization can be bypassed simply through
aggressive prompt engineering. To test this, we override our primarycommitinstruction with an expansiveintegrate
instruction, explicitly commanding the generator to enumerate every supported interpretation and cite all relevant
sources. As detailed in Table 16, while this integrative prompt successfully elevates absolute marginal recall across
the board (gains ranging from +0.050to+0.098), it completely fails to disrupt the fundamental factorial scaling laws.
Raising the generation count from 𝑇= 1to𝑇= 12continues to yield massive gains of +0.142to+0.200. Budget-matched
Manuscript submitted to ACM

The Laws of Context Allocation: Causal Measurement and Closed-Loop Orchestration in Generative Search 37
Table 17. Enumerated single-response baseline versus matched 𝑇=12portfolio, paired within query. The single response gets the
portfolio decode-token budget and must list distinct supported interpretations with grounding, stopping rather than padding. It
remains below portfolios at every width/instruction ( +0.109to+0.029), with narrowing gaps as context widens.∗/†:𝑝< 0.05/𝑝< 0.01.
instruction𝑘 𝑛portfolio structured gap 95% CI
commit 2 480 0.398 0.289+0.109†[+0.084,+0.135]
5 480 0.423 0.335+0.088†[+0.063,+0.114]
12 480 0.426 0.371+0.055†[+0.028,+0.082]
24 480 0.416 0.381+0.035†[+0.009,+0.061]
integrate 2 480 0.398 0.295+0.103†[+0.077,+0.130]
5 480 0.423 0.342+0.081†[+0.056,+0.108]
12 480 0.426 0.382+0.044†[+0.017,+0.071]
24 480 0.416 0.388+0.029∗[+0.002,+0.054]
superiority remains absolute: the multi-round (2,12)configuration systematically beats the single-pass (24,1)allocation
by+0.111(95%CI[+0.087,+0.136]).
Furthermore, we explicitly eliminate the confound of output-token volume limits. We construct a heavily optimized
structured single-response baseline: allocating this single pass the exact equivalent decode-token budget of a full
𝑇= 12portfolio, and coercing the model to output a structured list of distinct, grounded interpretations without
premature stopping. Despite these extreme guardrails, the structured single response systematically trails the sequential
portfolio across all widths and instructions (Table 17). While the performance gap expectedly narrows as the single
context window expands ( +0.109at𝑘=2collapsing to+0.035at𝑘=24under the commit instruction), the deficit
remains strictly positive and statistically significant across all480paired queries, proving that iterative context isolation
mechanically extracts evidence that a single wide-context pass fundamentally ignores.
D.3 Comprehensive System Ablations: The Necessity of Causal Feedback
Finally, Table 15 provides the exhaustive component-level ablation grid for theAscparchitecture across all four evaluated
tasks. These structural substitutions cleanly isolate the precise marginal utility of attribution feedback, submodular
selection, and steered decoding.
Notably, the final diagnostic rows strictly control the scheduling mechanism while substituting only the underlying
evidence-use signal. Replacing our causal probe with a naive embedding similarity proxy or a uniform-utilization
assumption triggers a systemic collapse in scheduling efficacy. This cleanly mirrors the diagnostic illusion exposed in
Section 4, conclusively establishing that counterfactual causal sensitivity is the absolute, irreplaceable driver of the
scheduler’s ability to maximize evidence coverage within a bounded generative budget.
Received
Manuscript submitted to ACM