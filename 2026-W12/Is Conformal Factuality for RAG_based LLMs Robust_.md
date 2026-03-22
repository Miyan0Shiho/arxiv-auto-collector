# Is Conformal Factuality for RAG-based LLMs Robust? Novel Metrics and Systematic Insights

**Authors**: Yi Chen, Daiwei Chen, Sukrut Madhav Chikodikar, Caitlyn Heqi Yin, Ramya Korlakai Vinayak

**Published**: 2026-03-17 17:20:08

**PDF URL**: [https://arxiv.org/pdf/2603.16817v1](https://arxiv.org/pdf/2603.16817v1)

## Abstract
Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged as potential ways to address this limitation. While RAG aims to ground responses in retrieved evidence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs across generation, scoring, calibration, robustness, and efficiency. We propose novel informativeness-aware metrics that better reflect task utility under conformal filtering. Across three benchmarks and multiple model families, we find that (i) conformal filtering suffers from low usefulness at high factuality levels due to vacuous outputs, (ii) conformal factuality guarantee is not robust to distribution shifts and distractors, highlighting the limitation that requires calibration data to closely match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform LLM-based model confidence scorers while requiring over $100\times$ fewer FLOPs. Overall, our results expose factuality-informativeness trade-offs and fragility of conformal filtering framework under distribution shifts and distractors, highlighting the need for new approaches for reliability with robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines that are both reliable and computationally efficient.

## Full Text


<!-- PDF content starts -->

ISCONFORMALFACTUALITY FORRAG-BASEDLLMSROBUST?
NOVELMETRICS ANDSYSTEMATICINSIGHTS
A PREPRINT
Yi Chen
University of Wisconsin-Madison
yi.chen@wisc.eduDaiwei Chen
University of Wisconsin-Madison
daiwei.chen@wisc.eduSukrut Madhav Chikodikar
University of Wisconsin-Madison
chikodikar@wisc.edu
Caitlyn Heqi Yin
University of Wisconsin-Madison
hyin66@wisc.eduRamya Korlakai Vinayak
University of Wisconsin-Madison
ramya@ece.wisc.edu
March 17, 2026
ABSTRACT
Large language models (LLMs) frequently hallucinate, limiting their reliability in knowledge-
intensive applications. Retrieval-augmented generation (RAG) and conformal factuality have emerged
as potential ways to address this limitation. While RAG aims to ground responses in retrieved evi-
dence, it provides no statistical guarantee that the final output is correct. Conformal factuality filtering
offers distribution-free statistical reliability by scoring and filtering atomic claims using a threshold
calibrated on held-out data, however, the informativeness of the final output is not guaranteed. We
systematically analyze the reliability and usefulness of conformal factuality for RAG-based LLMs
across generation, scoring, calibration, robustness, and efficiency. We proposenovel informativeness-
aware metricsthat better reflect task utility under conformal filtering. Across three benchmarks
and multiple model families, we find that (i) conformal filtering suffers fromlow usefulness at
high factuality levelsdue to vacuous outputs, (ii) conformal factuality guarantee isnot robust to
distribution shifts and distractors, highlighting the limitation that requires calibration data to closely
match deployment conditions, and (iii) lightweight entailment-based verifiers match or outperform
LLM-based model confidence scorers while requiring over 100× fewer FLOPs. Overall,our results
expose factuality–informativeness trade-offs and fragility of conformal filtering framework under
distribution shifts and distractors, highlighting the need for new approaches for reliability with
robustness and usefulness as key metrics, and provide actionable guidance for building RAG pipelines
that are both reliable and computationally efficient.
KeywordsRAG·LLM·hallucination mitigation·conformal prediction·factuality guarantee·calibration
1 Introduction
Large language models (LLMs) have demonstrated remarkable capabilities across open-domain question answering,
reasoning, and scientific discovery [Brown et al., 2020, Guo et al., 2025, Zhang et al., 2025]. Yet, a persistent barrier
to their reliable deployment is the phenomenon ofhallucinations: outputs that are fluent and confident but factually
incorrect [Ji et al., 2023, Nadeau et al., 2024, Huang et al., 2025]. Such errors are not merely cosmetic. In safety-critical
settings, such as medicine, law, or finance, a single fabricated claim can erode trust, propagate misinformation, and
incur high societal or financial costs. This makes hallucination mitigation one of the central challenges in advancing
trustworthy LLMs. A rich body of work has emerged to address this challenge along two main directions: (1) retrieval-
augmented generation (RAG) and (2) conformal methods. RAG aims to reduce hallucinations by grounding responses
in trusted external knowledge sources, typically by conditioning generation on retrieved passages [Lewis et al., 2020,
Gao et al., 2023, Siriwardhana et al., 2023]. While RAG reduces the likelihood of unsupported claims, it does not offerarXiv:2603.16817v1  [cs.AI]  17 Mar 2026

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Queryx∈ XReferenceR(x)
Response
GeneratorGOutputyConformal
Factuality
FrameworkCalibration
DataeX
Filtered
Outputy′
Input to LLMConformal Filtering
Figure 1: Overview of our framework. Given a query xand retrieved references R(x) , the Response Generator G
produces an output y. The conformal factuality framework utilizes a separate calibration data to determine a threshold
used to filter out information from the output yand yield y′. (See Figure 2 for the details of different stages involved in
conformal filtering.)
statistical guarantees on the factuality of the final response. Even with reference, LLMs can produce hallucinations in
the generated response [Huang et al., 2025]. The conformal prediction (CP) framework, on the other hand, aims to
provide a statistical guarantee on the final output, often via post-processing of the initial LLM response. CP frameworks
usually first decompose the initial LLM output into atomic claims, score each claim with a factuality scoring function,
and filter those falling below a threshold determined using a calibration dataset [Mohri and Hashimoto, 2024, Cherian
et al., 2024]. This procedure provides formal coverage guarantees but often at the expense of informativeness, since
aggressive filtering may yield empty or vacuous outputs. Furthermore, CP filtering cannot improve the usefulness or
accuracy of the LLM response; it can only remove hallucinations.
Despite their complementary strengths, it is unclear whether conformal prediction (CP) can improve reliability of
RAG-based LLMs. While several recent works integrate RAG with CP methods [Li et al., 2023, Rouzrokh et al.,
2024, Feng et al., 2025], they fall short of a systematic analysis that disentangles where gains come from and when
guarantees break down. A comprehensive understanding of the strengths and limitations of this combination requires
not only integrating the two frameworks, but also developing principled evaluation. In particular, standard metrics
such as empirical factuality can exhibit multiple failure modes and obscure real utility—for example, an empty answer
is trivially “factually correct” under such measures. Additionally, it remains unclear whether improved factuality
necessarilyrequires larger, more computationally expensive verifiers, or whether lightweight alternatives can achieve
comparable or superior performance. This raises a scaling-law-style question for factuality filtering: how do reliability
and utility scale with verifier capacity and inference cost, and where are the diminishing returns? Understanding the
relationship between factuality, model scale, and computational cost is critical for deploying reliable LLM systems in
practice, where both latency and compute budgets constrain end-to-end RAG pipelines. We address these issues in this
work.
Our Contributions: We systematically investigate the conformal factuality filtering framework for RAG-based LLMs
(Figure 1) making the following contributions:
•We proposenovel metricsto capture informativeness component that is often missed by traditional metrics:non-empty
rateandnon-vacuous empirical factuality, which jointly capture both correctness and information retention, and
sufficient correctnessandconditional sufficient correctness, which measure whether an output contains enough
correct information to infer the final answer to the query. These novel metrics capture the trade-off between factual
correctness and informativeness, providing practical insights and tools for future work on hallucination mitigation.
•We conduct comprehensive evaluation that spans diverse datasets (questions with free form answers, math and
natural question answering), various open-source model families and sizes (with and without reasoning), and scoring
functions (entailment-based and LLM-based scorers). Furthermore, we evaluate robustness against distribution shifts
and distractors, shedding light on both the capabilities and limitations of this approach. Our evaluation highlights the
key issues –limited usefulness at high factuality levelsandnon-robustness to distribution shifts and distractors. This
raises the need for new approaches for guaranteeing factuality with usefulness and robustness as important metrics.
•In addition, we analyze the trade-offs between verification accuracy and computational efficiency, demonstrating that
lightweight verifiers can outperform larger LLM-based scorers while requiring orders-of-magnitude fewer FLOPs.
2

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
2 Problem Setting, Datasets, Models, and Metrics
In this section, we describe the conformal filtering framework for RAG-based LLMs (Figure 1), introduce the scoring
functions, datasets and evaluation metrics used in this work.
Letx∈ X denote an input query. Let R(x) denote the reference material sufficient to answer x. We assume the
existence of an oracle retriever that can retrieve R(x) such that the true answer y⋆is either in or can be deduced
from R(x) . This enables us to focus on evaluating the effectiveness of conformal filtering of answers generated by
RAG-based LLMs by decoupling the effects of specific methods used for retrieval. A response generator G, instantiated
as a large language model (LLM), is prompted with bothxandR(x)to produce an outputy=G(x, R(x)). The goal
of the conformal filtering is to create a final output y′such that each statement in y′is factually correct at a user’s
expected level, e.g., 85%, while being useful in answering the query. Conformal filtering method as applied to this
system is described below.
Conformal Filtering.Let eXdenote a calibration set that is exchangeable with X. For each query ˜x∈eX, a set of
claims {ci}is obtained by parsing the corresponding yusing a parser P. These claims are then scored by a factuality
scoring function f. For each calibration query ˜x, the candidate threshold is defined as the smallest value of τsuch
that all claims with score above τare factual: inf{τ| ∀c∈F(τ), cis factual}, where F(τ) ={c|f(c)> τ}
denotes the set of claims scoring above τ. Given a specified error level α∈(0,1) , the conformal filtering framework
determines a threshold ταfrom the calibration set. Specifically, ταis chosen as the⌈(n+1)(1−α)⌉
nquantile of a set
of candidate thresholds, where n=|eX|is the size of calibration data. The resulting guarantee is that P 
∀c∈
F(τα), cis factual
≥1−α . At inference time, for each query x, the generated output yis decomposed into atomic
claims by a parser P, yielding C(y) ={c i}k
i=1. Each claim ciis scored with f, and those exceeding the threshold are
retained: C′(y) ={c i|f(c i)> τ α, i= 1, . . . , k} . Finally, the retained claims C′(y)are merged by a merger M
into a single filtered response:y′=M(C′(y)). Figure 2 illustrates the entire conformal filtering pipeline.
We usegpt-5-nano [gpt, 2025] in our experiments for tasks such as claim parsing, claim merging and factuality
labeling1.
InputxReference
Response
GeneratorOutputy ParserClaim 1
Claim 2
...
ClaimkScoring
functionf
Score 1
Score 2
...
Scorek
Conformal
Threshold
Calibration Data eXFiltered
Claim 1
Filtered
Claim 2
...
Filtered
Claimk′MergerFiltered
Outputy′
Figure 2: Given an input xand a reference text related to x, the Response Generator produces an output y, which is
then parsed by the Parser into a list of claims. Each individual claim is subsequently scored by the Scorer, conditioned
on the input xand, optionally, the reference text. These scores are passed to the conformal prediction algorithm, which
filters out claims whose scores fall below a learned threshold. Finally, the remaining claims are merged into a single
paragraph and returned to the user.
1We validate the factuality labeling quality of using a gpt-5-nano as a judge by comparing with human labeling (See Appendix
A.3 for details of the human evaluation).
3

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
2.1 Scoring Functions
Scoring functionfis a core component of the conformal factuality framework, the determination of which claims are
retained or filtered depends on the score. We study two main families of scoring functions:
(a)Entailment-based scorers are natural language inference (NLI) models to assess whether the reference text supports
a claim. We use both document-level and sentence-level variants.
1.Fordocument-level entailment, the entailment score is computed directly between the entire reference R(x) and
the claim ci. We use the entailment model from [Laurer et al., 2022] trained on the DocNLI dataset [Yin et al., 2021].
2.In thesentence-level entailmentsetting, entailment scores are computed between the claim ciand each sentence in
R(x), and then aggregated in two different ways:
(a)Conservative entailment, where a claim is marked as contradictory if any sentence in the reference contradicts
it. It is marked as entailed only if there are no contradictions and there is at least one sentence supporting it.
(b)Average entailment, which averages the scores of all non-neutral sentence-level comparisons. For sentence-
level entailment, we useroberta-large-mnlias the entailment model [Liu et al., 2019].
(b)LLM-based model confidence scorers. In this family, we prompt a language model to assign a factuality score
to each claim. We explore the design space of the prompt by varying five dimensions: (i) the inclusion of retrieved
references R(x) , (ii) evidence highlighting within R(x) , (iii) the use of Chain-of-Thought (CoT) reasoning, (iv) output
granularity for verbalizing (continuous [0,1] vs. Boolean) , and (v) evaluation consistency (single generation vs.
averaging over five independent generations). We refer to this scoring function asmodel confidence score.
2.2 Datasets
We perform evaluations on three datasets that span open-ended summarization, mathematical reasoning, and question
answering tasks. This diversity allows us to assess both the factual reliability and the task-level utility of our approach.
•FActScoredataset [Min et al., 2023] consists of 601 individuals, each paired with a Wikipedia page. Queries are: “Tell
me a paragraph biography about [person] ,” where the reference R(x) is the Wikipedia page of the person. Because
no canonical ground-truth answers are provided, this dataset is well-suited for evaluating factuality inopen-ended
generationand for testing whether models can produce faithful summaries grounded in external references. We
additionally considerFActScore Rare, a subset of 198 queries focusing on less well-known individuals. This subset
probes the robustness of models when the model’s parametric knowledge is not enough to answer the question, a
regime where hallucinations are more likely, if a reference is not given.
•MATHdataset [Hendrycks et al., 2021] contains 12,446 competition-style mathematics problems spanning five
difficulty levels and seven categories. Each problem provides a question xand a ground-truth answer y⋆. To construct
reference materials, we prompted gpt-5-nano to generate prerequisite knowledge relevant to solving each problem,
which serves asR(x).
•Natural Questions (NQ)dataset [Kwiatkowski et al., 2019] consists of 10K real-world queries collected from search
engines. Each query is annotated with both a long answer and a short answer; we use the long answer as the reference
R(x)and the short answer as the ground-truth.
Together, these datasets provide coverage over distinct capabilities: factual summarization (FActScore), mathematical
reasoning (MATH), and reference-based question answering (NQ).
2.3 Language Models
We evaluate our framework over several open-source language models in order to systematically evaluate different
components of the factuality and RAG pipeline under varying architectures, reasoning modes, and parameter scales.
Our open-source suite includes multiple families. The Qwen3 models [Yang et al., 2025] are evaluated both in their
base form and in a reasoning-enabled variant, Qwen3-Think , where reasoning with the <think></think> tag is
enabled. This contrast allows us to study whether reasoning-oriented training improves factuality scoring and filtering.
To broaden architectural diversity, we also include Llama-3.x-Instruct [Dubey et al., 2024], SmolLM2-Instruct
[Allal et al., 2025], and gpt-oss [Agarwal et al., 2025]. Table 1 summarizes the model families, parameter counts, and
architectures.
These models are chosen to probe three orthogonal dimensions: (i)model architecture, by comparing across families;
(ii)reasoning capability, by contrasting Qwen3 withQwen3-Think ; and (iii)model scale. This diversity allows us to
assess how each factor influences factuality scoring, and to highlight regimes where smaller, more efficient models
suffice.
4

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Family Model Params Arch. Activated
Qwen3Qwen3-0.6B0.6B Dense 0.6B
Qwen3Qwen3-4B4B Dense 4B
Qwen3Qwen3-8B8.2B Dense 8.2B
Qwen3Qwen3-30B-A3B30.5B MoE 3.3B
Qwen3Qwen3-32B32.8B Dense 32.8B
Llama-3.xLlama-3.2-1B-Instruct1.2B Dense 1.2B
Llama-3.xLlama-3.2-3B-Instruct3.2B Dense 3.2B
Llama-3.xLlama-3.1-8B-Instruct8B Dense 8B
SmolLM2SmolLM2-135M-Instruct135M Dense 135M
SmolLM2SmolLM2-360M-Instruct360M Dense 360M
SmolLM2SmolLM2-1.7B-Instruct1.7B Dense 1.7B
gpt-ossgpt-oss-20b21B MoE 3.6B
gpt-ossgpt-oss-120b117B MoE 5.1B
Table 1: Summary of evaluated open-source language models. For dense models, the number of activated parameters
equals the total parameter count. For MoE models, activated parameter counts are shown when known.
2.4 Evaluation Metrics
We evaluate the performance using both traditional factuality measures and proposed novel metrics that are designed to
capture aspects of informativeness that existing metrics overlook. We use the following commonly used criteria:
•Empirical Factuality (EF)measures the fraction of outputs y′in which all retained claims C′(y)are factual. By
convention, an empty claim set C′(y) =∅ is treated as factual, which can artificially inflate EF when filtering is
aggressive.Higher EF is better, as it indicates stronger factual reliability.
•Powerquantifies the average proportion of true claims retained.Higher Power is better, since it means fewer correct
claims are lost.
•False Positive Rate (FPR)measures the fraction of non-factual claims that survive.Lower FPR is better, as it reflects
stronger suppression of hallucinations.
•Correctnessmeasures the fraction of outputs equivalent to the ground-truth answer y⋆.Higher Correctness is better,
though this metric is intentionally strict and is most applicable on datasets with unambiguous ground-truth answers.
These metrics only provide limited insight into the overall usefulness of the final answer. Each of the statements
in an LLM response could be factually correct while still not being informative enough to answer the input query.
Furthermore, a vacuous or empty output is factually correct by definition, but is not useful. When the input reference
does contain information to provide a correct answer to the query, an empty final answer is an indication of failure.
While EF, Power, and FPR capture factuality and error rates, they fail to penalize vacuous but “factual” outputs. To
address this,we propose the following novel evaluation metrics.
•Non-empty Rate (NR), the fraction of outputs that preserve at least one claim.Higher NR is better, rewarding
informative responses rather than empty ones.
•Non-vacuous Empirical Factuality (NvEF), which computes EF only over non-empty outputs.Higher NvEF is
better, reflecting factuality conditional on informativeness.
•Sufficient Correctness (SC)evaluates whether an output to a given query x(initial output yor filtered output
y′) contains enough correct information—relative to a reference R(x) —to recover the correct answer.Higher SC
indicates better end-task utility.UnlikeCorrectness, which can be overly strict (e.g., penalizing partially correct but
still useful responses, or being inapplicable when there is no single canonical y⋆, e.g., open-ended summarization
task in FActScore), SC explicitly measures whether the content in the output is sufficient to answer the query.
•Conditional Sufficient Correctness (CSC)restricts evaluation of the filtered outputs y′whose unfiltered counterparts
yalready satisfy SC.A higher CSC reflects stronger fidelity of the filtering process.CSC isolates the effect of filtering
from generation quality. SC on the filtered output can drop either because the base model failed to include sufficient
information in the initial responseorbecause filtering removed it. Since CSC is conditioned on cases where the
unfiltered output has sufficient information and asks whether the filtered output preserves that sufficiency, it provides
a direct measurement of whether filtering maintains useful content rather than unnecessarily deleting it.
NR, NvEF are claim-level, while correctness and (conditional) sufficient correctness are at the final task-level outcome.
Together, these metrics balance factual reliability, informativeness, and task-level utility, ensuring that evaluation
reflects not only safety (removing hallucinations) but also usefulness (retaining an adequate signal for the end task).
5

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
We now design a series of experiments to systematically analyze: (i) the impact of references, (ii) the design of scoring
functions, (iii) robustness to distributional shifts and adversarial inputs and (iv) end-to-end evaluation with a focus on
overall computation involved.
3 Impact of References
We begin by isolating the role of references, asking:How much do retrieved references improve generation quality
before filtering is even applied?To study this, we evaluate outputs yproduced by the response generator Gunder
two conditions: query-only generation y=G(x) and query-plus-reference generation y=G(x, R(x)) . We then
measure sufficient correctness of the initial LLM response ywith respect to the reference R(x) usinggpt-5-nano
2for different datasets: FActScore, FActScore-Rare,MATH-200(a 200-example subset of MATH), andNQ-200(a
200-example subset of Natural Questions). These datasets help provide insights into whether conditioning on R(x)
improves different types of tasks: factual summarization, reasoning, and question answering.
Qwen3
0.6BQwen3
0.6B
ThinkQwen3
4BQwen3
4B
ThinkQwen3
8BQwen3
8B
Think
Model0.00.20.40.60.81.0Sufficient CorrectnessMATH 200
Reference
Yes
No
Qwen3
0.6BQwen3
0.6B
ThinkQwen3
4BQwen3
4B
ThinkQwen3
8BQwen3
8B
Think
ModelFActScore Rare
Qwen3
0.6BQwen3
0.6B
ThinkQwen3
4BQwen3
4B
ThinkQwen3
8BQwen3
8B
Think
ModelFActScore
Qwen3
0.6BQwen3
0.6B
ThinkQwen3
4BQwen3
4B
ThinkQwen3
8BQwen3
8B
Think
ModelNQ 200
Figure 3: Sufficient correctness (SC) of Qwen3 models (0.6B, 4B, 8B) on four datasets (MATH-200, FActScore-Rare,
FActScore, NQ-200), with and without access to references. Across model sizes and datasets, providing references
consistently improves generation quality.
Results.Figure 3 compares sufficient correctness (SC) with and without references for the Qwen3 family. On FActScore
and NQ, references play a major role, highlighting their importance when the model may not have memorized all
relevant information. On MATH, while the gap between the two settings decreases as model size increases, the SC
improves with the reference, suggesting that while larger models possess stronger reasoning abilities the reference
material still plays a role. For Qwen3-0.6B , enablingthinkreasoning produces a large jump in SC for MATH dataset,
highlighting the role of reasoning capacity for mathematical questions. In contrast, for FActScore and NQ, where
reasoning plays a smaller role, enablingthinkhas little effect. Lastly, we observe only marginal gains as model size
increases within the Qwen3 family: the improvement in SC from 4B to 8B is much smaller than the improvement from
0.6B to 4B. Overall, providing references consistently improves generation quality. We observe similar trends for
Llama-3.x ,SmolLM2 , and some frontier models (see Figures 14, 15, and 16 in Appendix A.1.1). On FActScore-Rare,
the performance of Qwen3 4B is comparable to that of Gemini 2.5 Pro and GPT-5.1 when reference is provided
(Figure 16). This demonstrates that having good reference enables even small and medium sized LLMs to generate good
outputs. The references are provided to the generator Gin all subsequent experiments which focus on effectiveness of
conformal filtering.
4 Design Choices for Factuality Scoring in Conformal Filtering
We now turn to the systematic evaluation of conformal filtering which provides statistical guarantees on factuality of
the final output. A key component of conformal filtering (Figure 2) is the factuality scoring function, which assigns a
score indicating how well a generated output is supported by the reference. The effectiveness of conformal filtering
therefore depends critically on these scores. In this section, we systematically study several design choices for factuality
scoring functions, including prompting strategies for LLM-based model confidence score, the role of references during
scoring, the choice of scorer model, and different families of scoring functions (entailment based scores compared
with LLM-based scores). Through experiments across multiple datasets and model families, we identify practical
configurations that improve filtering performance and calibration.
2more details in Appendix A.1.1
6

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
4.1 Prompting Strategies for LLM Model Confidence Scores
We begin by examining the effect of prompting variations for LLM-based scoring functions (model confidence score),
including: (i) highlighting supporting evidence, (ii) enabling chain-of-thought reasoning, (iii) scalar vs. Boolean scoring
for the verbalized scores, and (iv) consistency averaging (majority of multiple responses). We run the experiments
with different configurations of the above mentioned variations on FActScore, MATH-1K (a 1,050-example subset
of MATH), and NQ-1K (a 1,000-example subset of Natural Questions). References are always provided to both the
scoring functionfand the LLM generatorGthat produces the initial responsey(Figure 2) in these experiments3.
Results.Figure 4 summarizes the effect of prompting strategies on model confidence scores across three LLM families
for a subset of variations on FActScore dataset at level 1−α= 0.9 . While no single strategy emerges as universally
optimal, several consistent patterns are observed. First, instructing models to output numeric scores reliably outperforms
Boolean values. Second, sampling multiple responses provides consistent gains, indicating that aggregation reduces
variance and stabilizes confidence estimates.
Figure 4: Evaluation of various prompting strategies across different LLMs on the FActScore dataset (Section 4.1) at
level 1−α= 0.9 . Results demonstrate that: (i) prompting models to generate numeric scores consistently outperforms
Boolean scoring; (ii) sampling multiple responses uniformly improves performance; however, (iii) incorporating
chain-of-thought reasoning or evidence highlighting do not yield reliable performance gains across models.
4.2 Role of References on Model Confidence Scores
In this section, we evaluate how providing references to the LLM-based model confidence scoring function improves
scoring. Outputs y=G(x, R(x)) are generated using gpt-5-nano , while open-source models serve as scorers.
This is because API inference is faster than running models on our own GPUs. When varying reference access, we
hold all other prompt settings fixed, including chain-of-thought prompting, scalar scoring, and consistency averaging.
Because reference access changes across conditions, we accordingly adjust whether the model is instructed to highlight
supporting evidence from the reference. Experiments are conducted on FActScore, MATH-1K, and NQ-1K.
Results.Figure 5 shows the performance under various metrics for the model confidence scores with and without
reference provided to the model confidence scoring function for the MATH-1K dataset using Qwen3-4B as the scorer.
We observe that when a reference is introduced to the model confidence scoring function, the power is consistently
higher than in the case where no reference is given to the scoring function. Similarly, for the non-empty rate (NR), the
scoring functions with reference have an advantage when the target factuality is of a larger size. These results show the
benefit of feeding the reference to the LLM-based model confidence scoring function. Therefore, in the subsequent
experiments, we provide a reference to model confidence scoring function. We defer the results for FActScore and
NQ-1K dataset to Appendix A.1.3.
3Note that the sufficient correctness (SC) in this section is calculated on the filtered output y′by prompting gpt-5-nano with the
prompts in Appendix A.4.13.
7

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Figure 5: Performance of model confidence score on MATH-1K dataset with and without reference provided to scoring
functions usingQwen3-4Bas the LLM-based scorer.
4.3 Model Choice for LLM-based Scorers
In this section, we study the usage of different open-source models in model confidence scoring functions f, using
gpt-5-nano as the generator Gfor initial response. All prompts include references, require evidence highlighting and
chain-of-thought reasoning, produce scalar scores, and apply consistency averaging. This experiment assesses how the
scorer model family and scale affect factuality filtering.
Results.Our experiments reveal that scaling LLMs used in model confidence scorer does not guarantee improved
confidence calibration in conformal factuality, as shown in Figure 6. While among the Llama-3.x models, there is
an improvement in terms of power and sufficient correctness with larger models, this trend breaks with other model
families. The SmolLM2 models show no systematic benefit as we increase model size; and in the case of Qwen3 , scaling
even degrades performance. These experiments highlight that smaller models are competitive for scoring in model
confidence scores. This observation is practically useful especially since the model used has to be repeatedly called for
scoring each claim (for each query). We defer the results of different factuality levelαto Appendix A.1.4.
Figure 6: Experimental evaluation across three model families of various scales on the FActScore dataset at level
1−α= 0.9 . Results demonstrate that increasing model size does not consistently improve performance of conformal
factuality.
8

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
4.4 Comparison between Entailment-based and LLM-based Scoring Functions
Figure 7: Comparison between entailment-based scores against the model confidence score on the FActScore dataset.
Across our evaluated settings, the entailment-based scores match or exceed the model-confidence baseline, despite the
entailment model being substantially smaller than the target LLMs.
Beyond model confidence scores, where we utilize an LLM to generate a score, we also examine entailment-based
scoring functions. We consider three different entailment-based scoring functions (Section 2.1)): document entailment
score, average entailment score and conservative entailment score. We compare conformal filtering with entailment-
based filtering on FActScore and MATH-1K datasets to assess whether entailment signals offer additional advantages in
factuality filtering.
Results.Figure 7 presents a direct comparison between model confidence scores and entailment-based scoring
functions. Notably, entailment-based scores, in particular,document entailment score consistently match or exceeds
the performance of model confidence score, despite the entailment model being substantially smaller than the target
LLMs4. Together with our analysis in Section 4.3, provide practical insight that targeted, lightweight scorers (or
verifiers) can deliver both computational efficiency and superior performance. We also note that,at higher factuality
levels, the power is quite limited across all scoring functions.We defer the results on the MATH-1K dataset to
Appendix A.1.5.
Qwen3
0.6BQwen3
0.6B
ThinkonQwen3
4BQwen3
4B
ThinkonQwen3
8BQwen3
8B
ThinkonQwen3
30B
A3BQwen3
30B
A3B
ThinkonQwen3
32BQwen3
32B
Thinkon
Model0.00.10.20.30.40.50.60.70.8(Conditional) Sufficient Correctnessdataset = FActScore
Qwen3
0.6BQwen3
0.6B
ThinkonQwen3
4BQwen3
4B
ThinkonQwen3
8BQwen3
8B
ThinkonQwen3
30B
A3BQwen3
30B
A3B
ThinkonQwen3
32BQwen3
32B
Thinkon
Modeldataset = MATH-1K
Qwen3
0.6BQwen3
0.6B
ThinkonQwen3
4BQwen3
4B
ThinkonQwen3
8BQwen3
8B
ThinkonQwen3
30B
A3BQwen3
30B
A3B
ThinkonQwen3
32BQwen3
32B
Thinkon
Modeldataset = NQ-1K
Type
Cond. SC
SC
Figure 8: Sufficient Correctness (SC) and Conditional Sufficient Correctness (CSC) for the Qwen3 model family at
α= 0.05across FActScore, MATH-1K, and NQ-1K dataset.
4.5 Conditional Sufficient Correctness of the Filtered Output
Sufficient Correctness(SC) measures whether the output contains sufficient correct information, relative to a reference
R(x) , to recover the correct answer, with higher values indicating stronger end-task utility. When we measure SC on
the final filtered output, it can potentially conflate roles of conformal filtering and the quality of the LLM generator G
providing the initial unfiltered output y(see Figure 1). Note that conformal filtering can only remove claims in the
initial output but not add to them. So, when the filtered output y′fails to satisfy SC, it does not distinguish whether this
failure is caused by the conformal filtering process or by the fact that the original output yalready lacked sufficient
4We employ DeBERTa andRoBERTa to compute entailment scores, achieving computational efficiency gains of more than two
orders of magnitude compared to LLM-based confidence scoring methods. For detailed comparisons of model parameters and
computational complexity, see Table 2.
9

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
information. To decouple these effects, we proposeConditional Sufficient Correctness(CSC), which evaluates sufficient
correctness of the final output y′conditioned on the initial output ybeing sufficiently correct. This is done by evaluating
sufficient correctness only on instances where the unfiltered output already satisfies SC, thereby isolating how well
filtering preserves valid answers. For more detailed description on how SC and CSC is measured in Appendix A.1.6.
Results.Figure 8 reports SC and CSC for the Qwen3 family (used in the model confidence scorer) at α= 0.05
across FActScore, MATH-1K, and NQ-1K datasets. CSC consistently matches or exceeds SC across model scales and
reasoning variants. The gap between CSC and SC widens on MATH-1K and NQ-1K. Moreover, we observe that larger
models do not automatically yield better SC/CSC within the Qwen3 family: the smaller Qwen3-0.6B performs as well
as its 32B counterpart on all three datasets. This aligns with our findings in Section 4.3. A similar scaling trend is
observed for thegpt-ossfamily; results are deferred to Appendix A.1.6.
5 Robustness of Factuality Scoring Functions
Having examined the design and performance of different factuality scoring functions in Section 4.4, we now study their
robustness under distribution shifts. While conformal filtering provides the guarantee of factuality at the required level
1−α , the practical usefulness depends on whether it remains reliable when key assumptions are violated. In particular,
conformal filtering guarantee relies on the exchangeability between the claims in the calibration set and the test data. In
real-world deployments, the exchangeability assumption may fail due to distribution shifts (which can occur due to
various reasons, e.g., the phrasing of the input query, change in the LLM used to generate the initial answer or parser) or
adversarial perturbations. To evaluate the robustness of conformal factuality under such conditions, we perform a series
of stress tests that examine how scoring functions behave under calibration distribution shifts and distractor injection.
5.1 Robustness to Calibration Distribution Shift
We first study robustness to violations of the exchangeability assumption due to distribution shift in the test data when
compared to distribution of claims in the calibration data. Conformal factuality assumes that calibration and test data
are exchangeable. We evaluate robustness when this assumption is violated by using mismatched calibration data. To
achieve this, we use open-source models as the generator to produce yand parse them into claims. Then, we compare
the empirical factuality under two different calibration datasets:
•50 human-annotated claims from Mohri and Hashimoto [2024] ( gpt-4 -generated claims, denoted as MH).
Since these claims are generated by gpt-4 , they come from adifferent distributionthan claims generated by
the open-source model.
•50 randomly selected queries from the held-out test half for which the claims associated are generated by the
open-source model, and thus the claims follow thesame distributionas the test data.
We evaluate empirical factuality using Qwen3-4B ,SmolLM2-360M-Instruct , andLlama-3.2-3B-Instruct across
FActScore, MATH-1K, and NQ-1K datasets.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.40.60.81.0Empirical Factuality
NQ-1K
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
MATH-1K
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
FActScore
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
NQ-1K
Same Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
MATH-1K
Same Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
FActScore
Same Distribution
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 9: Empirical factuality (EF) on FActScore, MATH-1K, and NQ-1K, under two calibration settings: (i) calibration
claims generated in Mohri and Hashimoto [2024], which comes from a different distribution (ii) calibration claims
drawn from the same distribution as the test data. We use Qwen3-4B as the scoring function. The results show how the
distribution shift in the calibration set affects conformal factuality guarantees.
Results.Figure 9 compares empirical factuality (EF) under different sources of calibration data, with test outputs
generated and scored by Qwen3-4B . Figure 10 compares the power. When calibration data come from a different
10

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.00.20.40.60.81.0Power
NQ-1K
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
MATH-1K
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
FActScore
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
NQ-1K
Same Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
MATH-1K
Same Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
FActScore
Same Distribution
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 10: Power on FActScore, MATH-1K, and NQ-1K, under two calibration settings: (i) calibration claims generated
in Mohri and Hashimoto [2024], which comes from a different distribution (ii) calibration claims drawn from the same
distribution as the test data. We useQwen3-4Bas the scoring function.
distribution (generated by gpt-4 ), EF can fall below the target level (gray dashed line), especially at higher factuality
levels. The drop is significant for MATH-1K. While some entailment-based scorers seem to be robust to this distribution
shift, it comes at a steep cost on the power as shown in Figure 10. Furthermore, this robustness is not consistent across
language models: switching GandftoLlama-3.2-3B-Instruct orSmolLM2-360M-Instruct yields different
behaviors for the scoring functions that seem robust in the Qwen3-4B setting (see Figures 35 and 36 in Appendix A.1.7).
This shows that the conformal filtering is not robust to distribution shifts between the claims in the calibration set and
the test set.
5.2 Robustness to Distractors
Distribution shifts can also occur when LLM outputs contain irrelevant, misleading or hallucinated content. LLM
outputs may include claims that appear plausible yet are factually incorrect. This can arise because LLMs are susceptible
to being distracted by irrelevant information in the input [Shi et al., 2023]. To simulate such conditions, we replace a
proportion of factual claims in each test query with distractor statements generated bygpt-5-nano.
Specifically, for each query x, its associated reference text R(x) , and the set of claims {ci}, we prompt the LLM to
modify each ci, conditioning on xandR(x) , so that the result reflects a type of hallucination the model could plausibly
produce. The prompt design used to generate these hallucinated claims is documented in Appendix A.4.6. Our goal is
to create hallucinated claims that are sufficiently convincing to the model, such that it would judge them as potentially
originating from itself. To ensure this, after we generate a hallucinated claim, we ask the LLM to check if it thinks
that the claim might be generated or hallucinated by itself (prompt in Appendix A.4.7), given the same x, R(x) . If the
hallucination claim can cause the model to think that it is the one who generates it, given xandR(x) , then we keep this
hallucination claim. Otherwise, we repeat this process and generate a new hallucination claim.
We evaluate whether the different scoring functions used in the conformal factuality framework can reliably distinguish
correct claims from these plausible but incorrect claims. Experiments are performed on the FActScore, MATH-1K, and
NQ-1K datasets.
Results.Figure 11 (a)-(d) show that as the distractor rate increases, empirical factuality (EF) drops sharply. This
degradation occurs because adding distractors to the test set violates the exchangeability assumption underlying the
conformal factuality framework, causing EF to fall below the target level. Although EF can increase when the target
factuality is set very high, this comes at the cost of a substantial loss in power, as shown in Figure 11 (e). Our results
indicate that the conformal filtering with current scoring functions is not robust to distractors, underscoring the need for
improved scoring functions that maintain robustness in their presence.
5.3 Can Distraction-Aware Threshold Help?
A natural attempt to mitigate the effect of distractors is to anticipate their presence by using distraction-aware calibration.
To study the efficacy of this approach, we extend the previous setting by introducing distractors not only into the test set
but also into the calibration set. This models potential distribution shifts caused by distractor content and evaluates
whether conformal filtering remains reliable when calibration data include such claims. Since the true level of distractors
in practice is unknown, we vary the amount of distractors in calibration set keeping the fraction of distractors in the
11

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Empirical Factuality
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 11: (a) - (d) Empirical factuality versus target factuality (1- α) forQwen3-4B on the FActScore dataset. Each
panel corresponds to a different distractor injection rate in the test set (0.0, 0.1, 0.25, 0.5). The gray dashed diagonal
represents perfect calibration ( y=x ), where empirical factuality matches the target. (e) Power versus target factuality
(1 -α) when the injection rate is 0.1.
test sets to 0.25. This setup enables us to assess both under-estimation and over-estimation of distractor prevalence.
Experiments are conducted on FActScore, MATH-1K, and NQ-1K datasets.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Empirical Factuality
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Empirical Factuality
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Empirical Factuality
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 12: (a) - (c) Empirical factuality versus target factuality ( 1−α ) forQwen3-4B on the FActScore dataset when
the test set is injected with 25% distractors. We vary the proportion of distractors in the calibration set from (0.1, 0.25,
0.5). As the proportion matches, we can see that the empirical factuality rises to the y=x line. (d)-(e) Comparison
of non-empty rate when both the test and calibration sets contain no distractors and contain 25% of the distractors.
Although introducing distractors to the calibration set can achieve target factuality, the non-empty rate suffers.
Results.In Figure 12, we show the result for the setting with a test set with a distractor proportion of 0.25 and varying
the levels of distractors in the calibration set. When the fraction of distractors is underestimated, the EF is still far
below the target EF (Figure 12 (a)). From Figure 12 (b)-(c), we can see that introducing a large enough fraction of
distractors to the calibration set can bring up the empirical factuality. However, we note that this incurs a high cost on the
non-empty rate. As we see in Figure 12 (d)-(e), the non-empty rate drops significantly when distractors are introduced
to the calibration set. When both calibration and test contain no distractors, we have a much higher non-empty rate
compared to the case where we inject 25% distractors into both the calibration and test sets. This happens likely due to
the thresholds found by the conformal filtering framework becoming more stringent. Therefore, the scoring functions
cannot distinguish the distractors that are factually incorrect from the factually correct claims.
Overall,the experiments on robustness show that the conformal filtering framework is not robust to distribution
shifts and distractors.This calls attention to the need for re-thinking factuality guarantees for LLM outputs with
robustness as an important criteria.
6 Efficiency Evaluation of the Conformal Factuality Pipeline End-to-End
Providing factuality guarantees on the output of an LLM requires additional inference and therefore is necessarily
increases the expense of the final response. In this section, we evaluate the complete end-to-end pipeline, jointly
considering generation, scoring, and conformal filtering with a particular focus on efficiency measured by FLOPs. We
usegpt-oss-20b , a mixture-of-experts (MoE) model with 3.6B active parameters, as the response generator Gand
consider two options for the scoring function f: (i) using gpt-oss-20b itself as the scorer, reflecting the scenario in
which the same model is available for both generation and scoring; and (ii) using Qwen3-8B as the scorer, allowing us to
examine whether a dense model can serve as an alternative to a FLOPs-efficient MoE model. Finally, we compare these
12

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
LLM-based model confidence scorers with entailment-based scoring while analyzing the associated computational
costs. All experiments in this section are conducted on the FActScore dataset.
Model Total Tokens Total FLOPs
gpt-oss-20b(3.6B active) [Agarwal et al., 2025] 20001.44×1013
Qwen3-8B(8.19B active) [Yang et al., 2025] 20003.28×1013
DeepSeek-R1(37B active) [DeepSeek-AI, 2025] 20001.5×1014
DeBERTa(184M active) [He et al., 2020] 20004.9×1011
RoBERTa(356M active) [Liu et al., 2019] 20001.6×1012
Table 2: Estimated FLOPs for generating 1000 tokens with a 1000-token prompt (assuming KV caching).
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.00.51.0
Empirical
Factuality
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
Power
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
False
Positive
Rate
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
Non-Empty
Rate
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
Non-Vacuous
Empirical
Factuality
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
Sufficient
Correctness
Average Score Conservative Score Document Entailment ScoreModel Confidence Score
(GPT-OSS-20B)Model Confidence Score
(Qwen3-8B)y = x
Figure 13: Performance of entailment- and confidence-based scorers using gpt-oss-20b andQwen3-8B on FActScore.
Results.Figure 13 shows that confidence-based scoring with gpt-oss-20b achieves higher power and sufficient
correctness than Qwen3-8B , particularly at moderate target factuality levels. Despite activating substantially fewer
parameters, gpt-oss-20b consistently outperforms the dense Qwen3-8B , highlighting the advantages of the MoE
architecture for scalable factuality filtering. Table 2 further underscores this efficiency–performance trade-off. Notably,
the document entailment score based on DeBERTa lies at an even more favorable point on this frontier. It is over
100× more computationally efficient than gpt-oss-20b , yet achieves comparable non-empty rates and sufficient
correctness. This suggests thatlightweight entailment models can serve as strong surrogates for large LLM-based
scorers when computational budgets are constrained, without substantial losses in factuality performance.Together,
these results demonstrate that both parameter-efficient MoE models and compact entailment-based scorers offer
compelling alternatives to dense large models, enabling end-to-end factuality evaluation that is not only more effective
but also dramatically more economical.
7 Related Works
Many studies show that LLMs are prone to hallucinations [Nadeau et al., 2024, Huang et al., 2025] despite their
impressive capabilities in summarization, dialogue, and coding [Achiam et al., 2023, Zhang et al., 2024, Nam et al.,
2024]. RAG and conformal prediction based filtering methods have emerged as prominent methods to mitigate
hallucinations and provide factuality guarantees.
7.1 Retrieval-Augmented Generation
Retrieval-augmented generation (RAG) improves LLM performance on knowledge-intensive tasks by grounding
responses in retrieved external context [Lewis et al., 2020, Joren et al., 2025, Gao et al., 2023]. Formally, given a query
x, a retriever Rreturns context R(x) that supplements the model’s parametric knowledge, guiding generation toward
more factually correct outputs. In this work, we assume access to an oracle retriever that always provides relevant,
accurate references. While RAG is powerful, it does not provide statistical guarantees on the factuality of its outputs,
and generated responses can still contain hallucinations [Huang et al., 2025].
In addition, LLMs may not utilize references effectively, especially information appearing in the middle of the context—
a phenomenon known aslost-in-the-middle[Liu et al., 2023, Ravaut et al., 2023, Chen et al., 2023, Tang et al., 2023].
13

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
This may be exacerbated by positional encoding mechanisms such as rotary positional embeddings (RoPE) [Su et al.,
2024, Huang et al., 2025]. Another contributing factor is that, in pretraining data, salient information is often located
near the beginning or the end of a document rather than in the middle [Ravaut et al., 2023, Huang et al., 2025]. The
metric sufficient context is proposed in [Joren et al., 2025] to study the usefulness of the retrieved reference in a
RAG-based LLMs. In contrast to this metric which focuses on the quality of retrieval, sufficient correctness and
conditional sufficient correctness proposed in our work focuses on the quality of output from the RAG-based LLM
measured with respect to the reference.
7.2 Conformal Prediction
Conformal prediction provides another promising mitigation strategy by filtering non-factual content from LLM
outputs [V ovk et al., 2005, Angelopoulos and Bates, 2021, Mohri and Hashimoto, 2024, Cherian et al., 2024]. These
methods not only improve factuality but also offer a statistical guarantee, e.g., P(output is factual)≥1−α . For
instance, Mohri and Hashimoto [2024] introducedconformal factuality, which scores claims in the output and removes
those below a threshold calibrated on held-out data. The choice of scoring function fis therefore critical to the
effectiveness of the framework.
7.3 Conformal Prediction and RAG
Recent work has begun integrating conformal prediction into retrieval-augmented generation (RAG) to provide statistical
reliability guarantees. TRAQ [Li et al., 2023] applies conformal prediction at both the retriever and generator stages,
ensuring with high probability that a semantically correct answer is included in the output set. Conformal-RAG [Feng
et al., 2025] instead operates at the sub-claim level, filtering unreliable statements to guarantee factuality across domains.
Conflare [Rouzrokh et al., 2024] focuses on the retrieval stage, calibrating similarity thresholds so that retrieved
contexts contain the true answer with user-specified confidence. While these approaches provide important coverage
guarantees at different stages of the pipeline, they largely assess correctness in isolation. In contrast, our work provides
a systematic analysis of the performance of conformal filtering for guaranteeing factuality of RAG-based LLMs by
introducing new metrics—non-empty rate, non-vacuous empirical factuality, sufficient correctness and conditional
sufficient correctness—that explicitly capture the trade-off between correctness and informativeness, which existing
frameworks do not address. Furthermore, we perform robustness analysis that reveals the fragility of conformal filtering
under distribution shifts and in the presence of distractors.
8 Conclusion
We systematically investigate conformal filtering framework used in guaranteeing factuality of RAG-based LLM outputs.
Our experiments extensively study the importance of references for generation and scoring, various scoring functions,
sensitivity to calibration data, robustness to distractor-induced hallucinations and the efficiency of the end-to-end
pipeline. A key limitation we uncover is that standard factuality measures can overstate practical progress: because
they primarily rewardabsence of incorrect content, they can be optimized by filtering systems that abstain (returning
empty answers) or produce generic, non-committal responses that are technically “factual” yet unhelpful even when the
input reference has the information to answer the question. As a result, high empirical factuality may coincide with low
end-task utility, obscuring the correctness–informativeness trade-off that real deployments must navigate. To address
these limitations, we introduced novel metrics—non-empty rate,non-vacuous empirical factuality, and(conditional)
sufficient correctness—that explicitly measure whether filtered outputs remain informative and sufficient for answering
the query along with empirical factuality.
Our experiments span three datasets for diverse tasks – FActScore for open-ended summarization, MATH for mathemat-
ical queries, and Natural Questions and multiple model families, revealing several insights. In particular, we show that
stronger factuality does not require larger or more expensive verifiers: lightweight entailment-based verifiers consistently
outperform LLM-based confidence scorers while requiring orders of magnitude fewer FLOPs. Our comprehensive
analysis of robustness reveals that the current conformal filtering approaches are not robust under distribution shift and
distractors, which is an important limitation for practical usage in safety critical settings.
Overall, our findings provide actionable guidance for building reliable and efficient RAG systems and underscore the
need to rethink how factuality in LLMs is measured, enforced, and optimized under realistic deployment constraints
with a particular focus on robustness and usefulness.
14

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
References
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan,
Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners.Advances in neural
information processing systems, 33:1877–1901, 2020.
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong
Ma, Xiao Bi, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning.Nature, 645(8081):
633–638, 2025.
Yanbo Zhang, Sumeer A Khan, Adnan Mahmud, Huck Yang, Alexander Lavin, Michael Levin, Jeremy Frey, Jared
Dunnmon, James Evans, Alan Bundy, et al. Exploring the role of large language models in the scientific method:
from hypothesis to discovery.npj Artificial Intelligence, 1(1):14, 2025.
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and
Pascale Fung. Survey of hallucination in natural language generation.ACM computing surveys, 55(12):1–38, 2023.
David Nadeau, Mike Kroutikov, Karen McNeil, and Simon Baribeau. Benchmarking llama2, mistral, gemma and gpt
for factuality, toxicity, bias and propensity for hallucinations.arXiv preprint arXiv:2404.09785, 2024.
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua
Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy,
challenges, and open questions.ACM Transactions on Information Systems, 43(2):1–55, 2025.
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler,
Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks.
Advances in neural information processing systems, 33:9459–9474, 2020.
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang,
and Haofen Wang. Retrieval-augmented generation for large language models: A survey.arXiv preprint
arXiv:2312.10997, 2(1), 2023.
Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga
Nanayakkara. Improving the domain adaptation of retrieval augmented generation (rag) models for open domain
question answering.Transactions of the Association for Computational Linguistics, 11:1–17, 2023.
Christopher Mohri and Tatsunori Hashimoto. Language models with conformal factuality guarantees. InProceedings
of the 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024.
John Cherian, Isaac Gibbs, and Emmanuel Candes. Large language model validity via enhanced conformal prediction
methods.Advances in Neural Information Processing Systems, 37:114812–114842, 2024.
Shuo Li, Sangdon Park, Insup Lee, and Osbert Bastani. Traq: Trustworthy retrieval augmented question answering via
conformal prediction.arXiv preprint arXiv:2307.04642, 2023.
Pouria Rouzrokh, Shahriar Faghani, Cooper U Gamble, Moein Shariatnia, and Bradley J Erickson. Conflare: conformal
large language model retrieval.arXiv preprint arXiv:2404.04287, 2024.
Naihe Feng, Yi Sui, Shiyi Hou, Jesse C Cresswell, and Ga Wu. Response quality assessment for retrieval-augmented
generation via conditional conformal factuality. InProceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval, pages 2832–2836, 2025.
Aug 2025. URLhttps://openai.com/index/introducing-gpt-5/.
Moritz Laurer, Wouter van Atteveldt, Andreu Salleras Casas, and Kasper Welbers. Less annotating, more classifying –
addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI, June
2022. URLhttps://osf.io/74b8k. Preprint, Open Science Framework.
Wenpeng Yin, Dragomir Radev, and Caiming Xiong. Docnli: A large-scale dataset for document-level natural language
inference.arXiv preprint arXiv:2106.09449, 2021.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer,
and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining approach.arXiv preprint arXiv:1907.11692,
2019.
Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer,
and Hannaneh Hajishirzi. Factscore: Fine-grained atomic evaluation of factual precision in long form text generation.
arXiv preprint arXiv:2305.14251, 2023.
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob
Steinhardt. Measuring mathematical problem solving with the math dataset.arXiv preprint arXiv:2103.03874, 2021.
15

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle
Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering
research.Transactions of the Association for Computational Linguistics, 7:453–466, 2019.
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang,
Chenxu Lv, et al. Qwen3 technical report.arXiv preprint arXiv:2505.09388, 2025.
Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil
Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models.arXiv e-prints, pages arXiv–2407,
2024.
Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall,
Andrés Marafioti, Hynek Kydlí ˇcek, Agustín Piqueres Lajarín, Vaibhav Srivastav, et al. Smollm2: When smol goes
big–data-centric training of a small language model.arXiv preprint arXiv:2502.02737, 2025.
Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K Arora, Yu Bai,
Bowen Baker, Haiming Bao, et al. gpt-oss-120b & gpt-oss-20b model card.arXiv preprint arXiv:2508.10925, 2025.
Freda Shi, Xinyun Chen, Kanishka Misra, Nathan Scales, David Dohan, Ed H Chi, Nathanael Schärli, and Denny
Zhou. Large language models can be easily distracted by irrelevant context. InInternational Conference on Machine
Learning, pages 31210–31227. PMLR, 2023.
DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL
https://arxiv.org/abs/2501.12948.
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. Deberta: Decoding-enhanced bert with disentangled
attention.arXiv preprint arXiv:2006.03654, 2020.
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida,
Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report.arXiv preprint arXiv:2303.08774,
2023.
Yang Zhang, Hanlei Jin, Dan Meng, Jun Wang, and Jinghua Tan. A comprehensive survey on process-oriented automatic
text summarization with exploration of llm-based methods.arXiv preprint arXiv:2403.02901, 2024.
Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, and Brad Myers. Using an llm to help with
code understanding. InProceedings of the IEEE/ACM 46th International Conference on Software Engineering, pages
1–13, 2024.
Hailey Joren, Jianyi Zhang, Chun-Sung Ferng, Da-Cheng Juan, Ankur Taly, and Cyrus Rashtchian. Sufficient context:
A new lens on retrieval augmented generation systems. InThe Thirteenth International Conference on Learning
Representations, 2025. URLhttps://openreview.net/forum?id=Jjr2Odj8DJ.
Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in
the middle: How language models use long contexts.arXiv preprint arXiv:2307.03172, 2023.
Mathieu Ravaut, Aixin Sun, Nancy F Chen, and Shafiq Joty. On context utilization in summarization with large
language models.arXiv preprint arXiv:2310.10570, 2023.
Hung-Ting Chen, Fangyuan Xu, Shane Arora, and Eunsol Choi. Understanding retrieval augmentation for long-form
question answering.arXiv preprint arXiv:2310.12150, 2023.
Raphael Tang, Xinyu Zhang, Xueguang Ma, Jimmy Lin, and Ferhan Ture. Found in the middle: Permutation
self-consistency improves listwise ranking in large language models.arXiv preprint arXiv:2310.07712, 2023.
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with
rotary position embedding.Neurocomputing, 568:127063, 2024.
Vladimir V ovk, Alexander Gammerman, and Glenn Shafer.Algorithmic learning in a random world, volume 29.
Springer, 2005.
Anastasios N Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free
uncertainty quantification.arXiv preprint arXiv:2107.07511, 2021.
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk,
Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models.arXiv
preprint arXiv:2312.11805, 2023.
16

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A Appendix
This appendix provides supplementary material that supports and extends the analyses presented in the main paper. It is
organized as follows.
Appendix A.1presents extended experimental results that complement the main findings. Section A.1.1 broadens the
analysis of how retrieved references improve generation quality (Section 3) to additional model families ( Llama-3.x ,
SmolLM2 ) and frontier models. Section A.1.2 provides the full set of prompting-strategy comparisons across all target
factuality levels, extending the summary in Section 4.1. Section A.1.3 expands the reference-in-scoring analysis
(Section 4.2) across four model families and three datasets. Section A.1.4 supplements the scorer scaling study
(Section 4.3) with detailed radar plots at multiple factuality targets. Section A.1.6 reports sufficient correctness and
conditional sufficient correctness for model families beyond Qwen3 (Section 4.5). Section A.1.7 and Section A.1.8
extend the robustness analyses of Sections 5.1 and 5.2 to additional scoring functions and model families. Section A.1.9
complements Section 5.3 with full results on distraction-aware calibration.
Appendix A.2details how adversarial distractor claims are generated and verified, supporting the experimental protocol
in Section 5.2.
Appendix A.3reports the human evaluation used to validategpt-5-nanoas a factuality judge.
Appendix A.4collects all prompts used throughout our experiments, including those for generation, parsing, labeling,
scoring, merging, correctness evaluation, and distractor creation.
A.1 Extended Results
The following subsections present extended experimental results that complement the main-paper analyses. Each
subsection identifies the corresponding main-paper section and provides additional figures covering model families,
datasets, or configurations not shown in the main text.
A.1.1 Impact of References
Section 3 of the main paper demonstrates that providing retrieved references to the response generator consistently
improves sufficient correctness, using the Qwen3 family as the primary illustration. Here we first describe how sufficient
correctness is measured and then verify that this finding generalizes across model families and scales.
Measuring sufficient correctnessFor each original generated output y, we usegpt-5-nano with the prompt
provided in Appendix A.4.13 to assess whether yis sufficient correct. Concretely, we replace the {response}
placeholder in the prompt with the string representation of y. At the same time, we replace {query} and{reference}
with the corresponding query and reference, respectively.
Llama-3.x family.Figure 14 extends the reference-impact analysis to the Llama-3.x family. Consistent with the
Qwen3 results, all three model sizes—1B, 3B, and 8B—show clear gains in sufficient correctness when references
are provided, across all four datasets. The improvement is especially pronounced on FActScore and NQ-200, where
parametric knowledge alone is insufficient.
Llama-3.2
1B
InstructLlama-3.2
3B
InstructLlama-3.1
8B
Instruct
Model0.00.20.40.60.81.0Sufficient CorrectnessMATH 200
Reference
Yes
No
Llama-3.2
1B
InstructLlama-3.2
3B
InstructLlama-3.1
8B
Instruct
ModelFActScore Rare
Llama-3.2
1B
InstructLlama-3.2
3B
InstructLlama-3.1
8B
Instruct
ModelFActScore
Llama-3.2
1B
InstructLlama-3.2
3B
InstructLlama-3.1
8B
Instruct
ModelNQ 200
Figure 14: Sufficient correctness (SC) of Llama-3.x models on four datasets (MATH-200, FActScore-Rare, FActScore,
NQ-200), with and without access to references. Across model sizes and datasets, providing references consistently
improves generation quality.
17

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
SmolLM2 family.Figure 15 presents the same comparison for the SmolLM2 family. Despite their substantially
smaller parameter counts (135M–1.7B), these models also benefit from references, although the absolute level of
sufficient correctness remains lower than that of the larger families. This confirms that reference grounding is beneficial
even at very small scales.
SmolLM2
135M
InstructSmolLM2
360M
InstructSmolLM2
1.7B
Instruct
Model0.00.20.40.6Sufficient CorrectnessMATH 200
Reference
Yes
No
SmolLM2
135M
InstructSmolLM2
360M
InstructSmolLM2
1.7B
Instruct
ModelFActScore Rare
SmolLM2
135M
InstructSmolLM2
360M
InstructSmolLM2
1.7B
Instruct
ModelFActScore
SmolLM2
135M
InstructSmolLM2
360M
InstructSmolLM2
1.7B
Instruct
ModelNQ 200
Figure 15: Sufficient correctness (SC) of SmolLM2 models on four datasets (MATH-200, FActScore-Rare, FActScore,
NQ-200), with and without access to references. Across model sizes and datasets, providing references consistently
improves generation quality.
Frontier model comparison.Finally, Figure 16 compares the Qwen3 family against two frontier models—
gemini-2.5-pro [Team et al., 2023] and gpt-5.1 [gpt, 2025]—on the FActScore-Rare dataset. A noteworthy
finding is that Qwen3-4B , when given a reference, achieves sufficient correctness comparable to these frontier models.
This underscores that, with proper retrieval augmentation, even moderately sized models can match frontier-level factual
accuracy on knowledge-intensive tasks.
Qwen3
0.6BQwen3
0.6B
ThinkQwen3
4BQwen3
4B
ThinkQwen3
8BQwen3
8B
ThinkGemini 2.5 Pro GPT-5.1
Model0.00.20.40.60.81.0Sufficient Correctness
(a)FActScore Rare
Figure 16: Sufficient correctness (SC) of Qwen3 and frontier models on FActScore-Rare, with and without access
to references. Providing references consistently improves generation quality; notably, Qwen3-4B with references is
comparable to frontier models.
18

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.1.2 Prompting Strategies for Scorers
Figure 4 in Section 4.1 summarizes the effect of prompting strategies on LLM-based scoring functions using three
representative configurations per model. Here we present the full set of comparisons at five target factuality levels
(1−α∈ {0.75,0.8,0.85,0.9,0.95} ), enabling a more granular assessment of how each prompting dimension—
evidence highlighting, chain-of-thought reasoning, scalar vs. Boolean scoring, and consistency averaging—interacts
with the target factuality.
FActScore dataset.Figure 17 shows radar plots for all 16 prompting strategies across four LLMs on FActScore. The
plots confirm the main-paper finding that numeric scoring and consistency averaging are the most reliably beneficial
dimensions, while chain-of-thought and highlighting provide inconsistent gains that vary by model and factuality target.
MATH-1K dataset.Figure 18 repeats this analysis on MATH-1K. On mathematical reasoning tasks, the relative
advantage of numeric scoring is even more pronounced, likely because scalar confidence values better capture the
degree of certainty in multi-step derivations than binary labels.
19

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Figure 17: Overall performance comparison of model confidence scores across 16 prompting strategies at five target
factuality levels ( 1−α ) on the FActScore dataset. Each column corresponds to a different LLM scorer; each
row corresponds to a different target factuality. Numeric scoring and consistency averaging yield the most robust
improvements.
20

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Figure 18: Overall performance comparison of model confidence scores across various prompting strategies at five
target factuality levels ( 1−α ) on the MATH-1K dataset. The advantage of numeric over Boolean scoring is especially
clear for mathematical reasoning.
21

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.1.3 Role of References in Scoring
Section 4.2 demonstrates that providing references to the scoring function improves power and non-empty rate, using
Qwen3-4B on MATH-1K as the primary example. Here we extend this analysis to additional scorer models and datasets
to assess how broadly the benefit holds.
Qwen3-4B across datasets.Figures 19 and 20 show the reference effect for Qwen3-4B on FActScore and NQ-1K,
respectively. On both datasets, providing the reference to the scorer consistently improves power and non-empty rate,
mirroring the MATH-1K results in the main paper. The gains are largest at high target factuality, where the scoring
function must be most discriminating.
Figure 19: Performance of model confidence score on FActScore with and without reference provided to scoring
functions, using gpt-5-nano as generator and Qwen3-4B as scorer. Reference access improves power and non-empty
rate.
Figure 20: Performance of model confidence score on NQ-1K with and without reference provided to scoring functions,
usinggpt-5-nano as generator and Qwen3-4B as scorer. The benefit of reference access is consistent with the MATH-
1K and FActScore results.
22

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Qwen3-8B across datasets.Figures 21–23 repeat the analysis with Qwen3-8B . The trends are qualitatively similar to
Qwen3-4B : reference access yields consistent improvements, although the magnitude of the gain varies across datasets.
This suggests that the benefit of feeding references to the scorer is not an artifact of a particular model scale.
Figure 21: Performance of model confidence score on FActScore with and without reference provided to scoring
functions, usinggpt-5-nanoas generator andQwen3-8Bas scorer.
Figure 22: Performance of model confidence score on MATH-1K with and without reference provided to scoring
functions, usinggpt-5-nanoas generator andQwen3-8Bas scorer.
23

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Figure 23: Performance of model confidence score on NQ-1K with and without reference provided to scoring functions,
usinggpt-5-nanoas generator andQwen3-8Bas scorer.
24

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Llama-3.2-3B across datasets.Figures 24–26 present the same comparison for Llama-3.2-3B-Instruct . The
reference benefit is again observed, confirming that it is not specific to the Qwen3 family. Notably, on FActScore, the
improvement in sufficient correctness from adding a reference is among the largest we observe across all model–dataset
pairs.
Figure 24: Performance of model confidence score on FActScore with and without reference provided to scoring
functions, usinggpt-5-nanoas generator andLlama-3.2-3B-Instructas scorer.
Figure 25: Performance of model confidence score on MATH-1K with and without reference provided to scoring
functions, usinggpt-5-nanoas generator andLlama-3.2-3B-Instructas scorer.
25

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Figure 26: Performance of model confidence score on NQ-1K with and without reference provided to scoring functions,
usinggpt-5-nanoas generator andLlama-3.2-3B-Instructas scorer.
26

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
SmolLM2-1.7B across datasets.Finally, Figures 27–29 show results for SmolLM2-1.7B-Instruct , the smallest
LLM-based scorer in our study. Even at this scale, reference access improves scoring performance, although the
absolute gains are more modest. This is consistent with the hypothesis that smaller models have less capacity to leverage
long reference contexts, but still benefit from the additional grounding signal.
Figure 27: Performance of model confidence score on FActScore with and without reference provided to scoring
functions, usinggpt-5-nanoas generator andSmolLM2-1.7B-Instructas scorer.
Figure 28: Performance of model confidence score on MATH-1K with and without reference provided to scoring
functions, usinggpt-5-nanoas generator andSmolLM2-1.7B-Instructas scorer.
27

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Figure 29: Performance of model confidence score on NQ-1K with and without reference provided to scoring functions,
usinggpt-5-nanoas generator andSmolLM2-1.7B-Instructas scorer.
28

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.1.4 Model Choice for Scorers
Section 4.3 examines how the scorer model family and scale affect factuality filtering, using gpt-5-nano as the
generator. The main paper presents a condensed summary; here we provide the full set of radar plots at each target
factuality level (1−α∈ {0.75,0.8,0.85,0.9,0.95}) for all three model families (Llama-3.x,Qwen3, andSmolLM2).
Figure 30 reveals the heterogeneous scaling behaviors discussed in Section 4.3 in greater detail. While the Llama-3.x
family shows monotonic improvement with model size across most metrics and factuality targets, this pattern does
not hold for Qwen3 orSmolLM2 . ForQwen3 , the smallest model (0.6B) sometimes outperforms larger variants, and for
SmolLM2 , there is no systematic ordering by parameter count. These results reinforce the main-paper conclusion that
scaling alone does not guarantee improved conformal factuality.
Figure 30: Overall performance comparison of model confidence scores across different model scales and families at
five target factuality levels ( 1−α ) on the FActScore dataset. While Llama-3.x models show consistent gains with
scale,Qwen3andSmolLM2do not, highlighting that scaling alone is insufficient for improving conformal factuality.
29

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.1.5 Comparison between Entailment-based and LLM-based Scoring Functions
Section 4.4 compares model confidence score with entailment-based scoring functions on the FActScore dataset. Here,
we extend the results to the MATH-1K dataset.
Figure 31: Comparison between entailment-based scores against the model confidence score on the MATH-1K dataset.
As we can see in Figure 31, model confidence score yields a better power and non-empty rate. However, in terms
of empirical factuality and non-vacuous empirical factuality, the gap between the different scoring functions is close.
Moreover, with some models, the document entailment score has a better sufficient correctness and and accuracy
comparing to model confidence score.
30

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.1.6 Sufficient Correctness and Conditional Sufficient Correctness
Section 4.5 introduces Conditional Sufficient Correctness (CSC) and reports results for the Qwen3 family. Here, we first
describe how SC and CSC is measured. Then, we extend this analysis to three additional model families— Llama-3.x ,
SmolLM2, andgpt-oss—to assess whether the observed patterns generalize.
Measuring (conditional) sufficient correctness. Let yandy′denote the original and filtered outputs for the same
input, respectively. In this section, sufficient correctness measures whether the filtered output y′contains enough correct
information to recover the answer. To evaluate this, we use gpt-5-nano with the prompt in Appendix A.4.13, replacing
the placeholder {response} with the string representation of y′. Letndenote the number of examples in the dataset.
We then define sufficient correctness as
SC =1
nnX
i=1I[y′
iis sufficient correct].
Conditional sufficient correctness is defined by restricting attention to examples for which the original output yis
already sufficient correct. Formally,
CSC =Pn
i=1I[y′
iis sufficient correct]Pn
i=1I[yiis sufficient correct].
Llama-3.x family.Figure 32 shows SC and CSC for Llama-3.x atα= 0.05 . As with Qwen3 , CSC consistently
exceeds SC, confirming that a substantial portion of SC failures are attributable to the generator rather than the filter.
The gap is especially wide on MATH-1K, where the smaller models struggle to produce sufficiently correct unfiltered
outputs.
Llama
3.2
1B
InstructLlama
3.2
3B
InstructLlama
3.1
8B
Instruct
Model0.00.20.40.60.8(Conditional) Sufficient Correctnessdataset = FActScore
Llama
3.2
1B
InstructLlama
3.2
3B
InstructLlama
3.1
8B
Instruct
Modeldataset = MATH-1K
Llama
3.2
1B
InstructLlama
3.2
3B
InstructLlama
3.1
8B
Instruct
Modeldataset = NQ-1K
Type
Cond. SC
SC
Figure 32: Sufficient Correctness (SC) and Conditional Sufficient Correctness (CSC) for the Llama-3.x family at
α= 0.05 across FActScore, MATH-1K, and NQ-1K. CSC consistently exceeds SC, indicating that filtering largely
preserves useful content when the generator provides it.
SmolLM2 family.Figure 33 presents the same comparison for SmolLM2 . The absolute SC values are lower due to
the smaller model sizes, but the CSC–SC gap is qualitatively similar. Notably, there is no consistent improvement in SC
or CSC with model size within this family, echoing the main-paper finding that scaling the scorer does not reliably
improve conformal factuality.
SmolLM2
135M
InstructSmolLM2
360M
InstructSmolLM2
1.7B
Instruct
Model0.00.10.20.30.40.5(Conditional) Sufficient Correctnessdataset = FActScore
SmolLM2
135M
InstructSmolLM2
360M
InstructSmolLM2
1.7B
Instruct
Modeldataset = MATH-1K
SmolLM2
135M
InstructSmolLM2
360M
InstructSmolLM2
1.7B
Instruct
Modeldataset = NQ-1K
Type
Cond. SC
SC
Figure 33: Sufficient Correctness (SC) and Conditional Sufficient Correctness (CSC) for the SmolLM2 family at
α= 0.05across FActScore, MATH-1K, and NQ-1K. No consistent scaling benefit is observed within this family.
31

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
gpt-oss family.Figure 34 compares the two gpt-oss models (20B and 120B). Despite a roughly 6×difference in
total parameters, the larger model does not consistently outperform the smaller one in SC or CSC, further supporting
the conclusion that parameter count is not the primary driver of conformal factuality performance.
gpt
oss
20bgpt
oss
120b
Model0.00.20.40.60.8(Conditional) Sufficient Correctnessdataset = FActScore
gpt
oss
20bgpt
oss
120b
Modeldataset = MATH-1K
gpt
oss
20bgpt
oss
120b
Modeldataset = NQ-1K
Type
Cond. SC
SC
Figure 34: Sufficient Correctness (SC) and Conditional Sufficient Correctness (CSC) for the gpt-oss family at
α= 0.05 across FActScore, MATH-1K, and NQ-1K. The larger gpt-oss-120b does not consistently outperform
gpt-oss-20b.
32

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.1.7 Robustness to Calibration Distribution Shift
Section 5.1 studies the sensitivity of conformal factuality guarantees to distribution shift between calibration
and test data, using Qwen3-4B as the scorer. Here we extend this analysis to Llama-3.2-3B-Instruct and
SmolLM2-360M-Instruct to assess whether the robustness (or lack thereof) of different scoring functions is consistent
across model families.
Llama-3.2-3B-Instruct.Figure 35 shows empirical factuality under same-distribution and different-distribution
calibration for Llama-3.2-3B-Instruct . When calibration data come from a different distribution (left panels), the
factuality guarantee frequently fails across all three datasets. Interestingly, the entailment-based scorers, which appeared
robust to distribution shift under Qwen3-4B , show degraded performance here, indicating that robustness to distribution
shift is model-dependent rather than a universal property of any particular scoring function family.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.40.60.81.0Empirical Factuality
NQ-1K
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
MATH-1K
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
FActScore
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
NQ-1K
Same Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
MATH-1K
Same Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
FActScore
Same Distribution
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 35: Empirical factuality (EF) on FActScore, MATH-1K, and NQ-1K under same-distribution and different-
distribution calibration, using Llama-3.2-3B-Instruct as the scorer. Distribution shift causes the factuality guarantee
to break for several scoring functions.
SmolLM2-360M-Instruct.Figure 36 presents the same analysis for SmolLM2-360M-Instruct . The pattern is
consistent: different-distribution calibration leads to violations of the target factuality level, particularly on NQ-1K and
MATH-1K. Together with the Qwen3-4B andLlama-3.2-3B results, these findings underscore that calibration data
must be collected using the same generator and scorer that will be deployed, as even switching the underlying LLM for
the same dataset can break the exchangeability assumption.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.40.60.81.0Empirical Factuality
NQ-1K
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
MATH-1K
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
FActScore
Different Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
NQ-1K
Same Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
MATH-1K
Same Distribution
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
FActScore
Same Distribution
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 36: Empirical factuality (EF) on FActScore, MATH-1K, and NQ-1K under same-distribution and different-
distribution calibration, using SmolLM2-360M-Instruct as the scorer. The results confirm that distribution shift in the
calibration set undermines conformal factuality guarantees across model families.
33

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.1.8 Robustness to Adversarial Distractors
Section 5.2 examines how injecting adversarial distractor claims into the test set degrades empirical factual-
ity, using Qwen3-4B on FActScore as the primary example. Here we extend this analysis to additional scorers
(Llama-3.2-3B-Instruct andSmolLM2-1.7B-Instruct ) and datasets (MATH-1K and NQ-1K) to assess the gen-
erality of this vulnerability.
FActScore dataset.Figures 37 and 38 show that both Llama-3.2-3B-Instruct andSmolLM2-1.7B-Instruct
exhibit the same sharp degradation in empirical factuality as the distractor rate increases, consistent with the Qwen3-4B
results. This confirms that the vulnerability to adversarial distractors is not model-specific but a systemic issue with the
current conformal filtering framework.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 37: Empirical factuality under varying test distractor proportions ( 0.0to0.5) with fixed clean calibration data on
FActScore, usingLlama-3.2-3B-Instruct. Factuality degrades sharply as distractor rate increases.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 38: Empirical factuality under varying test distractor proportions ( 0.0to0.5) with fixed clean calibration data on
FActScore, usingSmolLM2-1.7B-Instruct. The degradation pattern is consistent with other model families.
34

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
MATH-1K dataset.Figures 39–41 extend the distractor robustness analysis to MATH-1K. Mathematical claims are
particularly susceptible to subtle numerical perturbations, and indeed we observe that empirical factuality drops rapidly
even at low distractor rates (e.g., 10%). This suggests that distractor robustness is an especially critical concern for
reasoning-intensive tasks.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 39: Empirical factuality under varying test distractor proportions on MATH-1K with Qwen3-4B . Mathematical
claims are especially vulnerable to subtle numerical perturbations.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 40: Empirical factuality under varying test distractor proportions on MATH-1K with Llama-3.2-3B-Instruct .
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 41: Empirical factuality under varying test distractor proportions on MATH-1K with SmolLM2-1.7B-Instruct .
35

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
NQ-1K dataset.Figures 42–44 complete the analysis on NQ-1K. The overall pattern is consistent across all three
datasets: as distractors are injected, empirical factuality drops below the target level, and recovering factuality by raising
the target threshold comes at a steep cost in power. This reinforces the main-paper conclusion that developing scoring
functions robust to adversarial content is a critical direction for future work.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 42: Empirical factuality under varying test distractor proportions on NQ-1K withQwen3-4B.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 43: Empirical factuality under varying test distractor proportions on NQ-1K withLlama-3.2-3B-Instruct.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(b) T est: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(c) T est: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00
(d) T est: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Power
(e) T est: 0.1
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 44: Empirical factuality under varying test distractor proportions on NQ-1K withSmolLM2-1.7B-Instruct.
36

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.1.9 Robustness with Adversarial Prepared Calibration
Section 5.3 investigates whether including distractors in the calibration set can restore factuality guarantees when
distractors are also present at test time. The main paper presents results for Qwen3-4B on FActScore. Here we provide
the complete set of results across all three datasets and three scorer models.
The key question addressed by these experiments is:If we anticipate adversarial distractors during deployment, can we
protect the factuality guarantee by injecting synthetic distractors into the calibration set?The answer, as the following
figures show, is nuanced: matching the distractor proportion in calibration to that in the test set does restore empirical
factuality to the target level, but at a severe cost to the non-empty rate. This trade-off arises because the conformal
threshold becomes more stringent when calibration data contain distractors, causing the filter to aggressively remove
content—including correct claims.
FActScore dataset.Figures 45 and 46 show the factuality–coverage trade-off for Llama-3.2-3B-Instruct and
SmolLM2-1.7B-Instruct on FActScore. When calibration distractor proportion is underestimated (panels a), empiri-
cal factuality remains below the target. Matching the proportion (panels b) restores the guarantee, and overestimating it
(panels c) yields conservative but highly restrictive filtering. Panels (d)–(e) illustrate the non-empty rate cost.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 45: Empirical factuality and non-empty rates under varying calibration distractor proportions on FActScore with
Llama-3.2-3B-Instruct . Matching calibration to test distractor levels restores factuality but reduces the non-empty
rate.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 46: Empirical factuality and non-empty rates under varying calibration distractor proportions on FActScore with
SmolLM2-1.7B-Instruct.
37

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
MATH-1K dataset.Figures 47–49 present the same analysis on MATH-1K. The trade-off is even starker here:
the non-empty rate drops precipitously when distractors are introduced into calibration, reflecting the difficulty of
distinguishing correct mathematical steps from plausible but incorrect ones.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 47: Empirical factuality and non-empty rates under varying calibration distractor proportions on MATH-1K with
Qwen3-4B.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 48: Empirical factuality and non-empty rates under varying calibration distractor proportions on MATH-1K with
Llama-3.2-3B-Instruct.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 49: Empirical factuality and non-empty rates under varying calibration distractor proportions on MATH-1K with
SmolLM2-1.7B-Instruct.
38

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
NQ-1K dataset.Figures 50–52 complete the analysis on NQ-1K. The overall pattern is consistent: distraction-aware
calibration can restore the statistical guarantee, but the resulting outputs are frequently empty, highlighting a fundamental
limitation of threshold-based filtering when scoring functions cannot reliably distinguish correct claims from adversarial
ones.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 50: Empirical factuality and non-empty rates under varying calibration distractor proportions on NQ-1K with
Qwen3-4B.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 51: Empirical factuality and non-empty rates under varying calibration distractor proportions on NQ-1K with
Llama-3.2-3B-Instruct.
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(a) T est: 0.25, Calib.: 0.1
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(b) T est: 0.25, Calib.: 0.25
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00EF
(c) T est: 0.25, Calib.: 0.5
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(d) T est: 0, Calib.: 0
0.75 0.80 0.85 0.90 0.95
Target Factuality (1 )
0.000.250.500.751.00Non-empty Rate
(e) T est: 0.25, Calib.: 0.25
Model Confidence Score Document Entailment Score Conservative Entailment Score Average Entailment Score
Figure 52: Empirical factuality and non-empty rates under varying calibration distractor proportions on NQ-1K with
SmolLM2-1.7B-Instruct.
39

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.2 Distractor Generation Protocol
A core assumption of the conformal factuality framework is that the test data are exchangeable with the calibration
dataset. In practice, however, this assumption may be mildly violated—for example, when LLM outputs contain
hallucinated claims that were not represented in calibration. To stress-test the robustness of scoring functions under
such conditions (Section 5.2), we construct adversarial distractor claims as follows.
For each query xi, the corresponding reference text R(x) , and the set of parsed claims {ci}, we prompt the LLM to
modify each ci(conditioned on xandR(x) ) so that the result resembles a plausible hallucination. The full prompt used
for this generation step is provided in Appendix A.4.6.
Crucially, we want these distractor claims to be sufficiently convincing that the model itself would believe it generated
them. To enforce this, we apply a verification step: after generating a hallucinated claim, we prompt the LLM to judge
whether the claim could plausibly be one of its own outputs given xandR(x) (prompt in Appendix A.4.7). If the model
identifies the claim as a plausible self-generated hallucination, we retain it; otherwise, we regenerate until a sufficiently
convincing distractor is found. This two-stage process ensures that the adversarial distractors are realistic and relevant
to the failure modes of modern LLMs.
40

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.3 Human Evaluation
To validate the use of gpt-5-nano as a factuality judge (Section 2.3), we conducted a human evaluation study. We
randomly sampled 200 claims from the FActScore dataset and had two graduate students (referred to as A and B)
independently label the factuality of each claim. We also obtained labels fromgpt-5-nanoon the same set.
Table 3 reports the pairwise agreement rates. The agreement between gpt-5-nano and each human annotator (76.5%
and 77.0%) is comparable to—and slightly exceeds—the inter-annotator agreement between the two humans (73.0%).
This result supports the use of gpt-5-nano as a factuality judge in our experimental pipeline, as it performs on par
with individual human annotators.
Pair Labelers Agreement Rate
Model–Humangpt-5-nanovs. Student A 76.5%
Model–Humangpt-5-nanovs. Student B 77.0%
Human–Human Student A vs. Student B 73.0%
Table 3: Agreement rates on factuality labels between gpt-5-nano and two human annotators, as well as between the
two human annotators themselves. The model–human agreement is comparable to inter-human agreement.
41

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.4 Prompts
This section collects all prompts used throughout our experiments. We organize them by their role in the pipeline:
generation (Sections A.4.1 and A.4.2), claim parsing (Section A.4.3), factuality labeling (Sections A.4.4 and A.4.5),
adversarial distractor generation and verification (Sections A.4.6 and A.4.7), model confidence scoring (Section A.4.8),
claim merging (Sections A.4.9–A.4.11), correctness evaluation (Section A.4.12), sufficient correctness evaluation
(Section A.4.13), and MATH reference generation (Section A.4.14). All prompts instruct the model to return structured
JSON5 output to facilitate automated parsing.
A.4.1 Generator (with reference)
You are a helpful assistant that answers queries strictly based on the provided
reference text.,→
Instructions:
1. You will be given:
- A reference text
- A query
2. Use only the information from the reference text to answer the query.
3. Do not include any information not supported by the reference text.
Output Requirements:
- Output ONLY a single VALID JSON5 object with EXACTLY this schema:
{
"response": "...answer strictly based on the reference text..."
}
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\.
- No trailing commas in objects or arrays.
- Use the exact top-level container specified and close it properly.
- Do not include comments, code fences, or any text outside the JSON5 output.
- Follow the schema exactly; do not add or omit keys.
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
- Any code block delimiters (e.g.,```json).
Input:
Reference Text: {reference}
Query: {query}
(Reiteration of the instruction)
Answer the query strictly using only the reference text, and return a single JSON5
object with the key "response" only. ,→
Output:
A.4.2 Generator (without reference)
You are a helpful assistant that answers queries.
Instructions:
1. You will be given:
- A query
42

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Output Requirements:
- Output ONLY a single VALID JSON5 object with EXACTLY this schema:
{
"response": "...answer..."
}
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\.
- No trailing commas in objects or arrays.
- Use the exact top-level container specified and close it properly.
- Do not include comments, code fences, or any text outside the JSON5 output.
- Follow the schema exactly; do not add or omit keys.
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
- Any code block delimiters (e.g.,```json).
Input:
Query: {query}
(Reiteration of the instruction)
Answer the query and return a single JSON5 object with the key "response" only.
Output:
A.4.3 Parser
You are an AI assistant tasked with breaking down input text into small,
self-contained claims for easy human verification. ,→
Instructions:
1. Parse the provided text into concise, independent, and non-overlapping
subclaims.,→
2. Ensure each subclaim is:
- As small and specific as possible.
- Independent and self-contained.
- Do not use pronouns like he, she, his, her, it, its, etc.
- Explicitly mention subjects.
- Factually complete without relying on context from other subclaims.
3. If the provided text is not a full sentence, use the provided text verbatim as
the subclaim,→
Output Requirements:
1. The result must be a VALID and COMPLETE JSON list of dictionaries.
2. Each dictionary must have the following structure:
{
"subclaim": "Subclaim text"
}
JSON Rules:
- Ensure the JSON is STRICTLY VALID:
- Use DOUBLE QUOTES ("") for all keys and string values.
- DO NOT include trailing commas after the LAST item in arrays or objects.
- Ensure ALL dictionaries are enclosed in curly braces {}.
43

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
- Ensure the JSON list is ENCLOSED in square brackets [].
- CLOSE the JSON list properly with a closing square bracket ].
- DO NOT include any code block delimiters.
- DO NOT include any additional text, explanations, or comments in the output—return
ONLY the JSON list.,→
Critical Requirement:
- Validate and ensure the JSON is complete, properly CLOSED, and parseable. The final
JSON MUST pass validation without errors. ,→
(Reiteration of the instruction)
1. Parse the provided text into concise, independent, and non-overlapping subclaims.
2. Ensure each subclaim is:
- As small and specific as possible.
- Independent and self-contained.
- Do not use pronouns like he, she, his, her, it, its, etc.
- Explicitly mention subjects.
- Factually complete without relying on context from other subclaims.
3. If the provided text is not a full sentence, use the provided text verbatim as the
subclaim,→
Input:
{input}
Output:
A.4.4 Labeler (without ground-truth answer)
You are an AI assistant tasked with assigning a label to a claim based on its
factuality.,→
Instructions:
1. You are given:
- A reference text for the query.
- A query.
- A claim made in response to the query.
2. Rate the factuality of the claim as a boolean:
- true -> The claim is well-supported by the reference text.
- false -> The claim contradicts the reference text, or is not well-supported by it.
- If it is ambiguous, answer False.
3. Provide:
- The part(s) of the reference text that directly support your decision.
- A reasoning statement describing your rationale.
4. You must assign either true or false. Never return null or None.
Output Requirements:
- Output ONLY a single VALID JSON5 object with EXACTLY these keys:
{
"highlighted_text": "Part(s) of the reference text that support the decision.",
"reasoning": "A reasoning statement describing your rationale.",
"answer": true
}
- "answer" must be a boolean (true/false).
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\.
44

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
- No trailing commas in objects or arrays.
- Use the exact top-level container specified and close it properly.
- Do not include comments, code fences, or any text outside the JSON5 output.
- Follow the schema exactly; do not add or omit keys.
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
- Any code block delimiters (e.g.,```json).
Examples:
Example Input:
Reference Text: Michael Scott is a fictional character in the NBC sitcom The Office,
portrayed by Steve Carell. Michael is the regional manager of the Scranton,
Pennsylvania branch of Dunder Mifflin, a paper company, for the majority of the
series. Like his counterpart in the original British version of the show, David
Brent, he is characterized as a largely incompetent, unproductive, unprofessional
boss, though he is depicted as kinder and occasionally shown to be effective at
his job in key moments.,→
,→
,→
,→
,→
,→
Query: Tell me a paragraph bio of Michael Scott.
Claim: The fictional character Michael Scott is the regional manager of a paper
company.,→
Example Output:
{
"highlighted_text": "Michael Scott is a fictional character in the NBC sitcom The
Office, portrayed by Steve Carell. Michael is the regional manager of the
Scranton, Pennsylvania branch of Dunder Mifflin, a paper company, for the
majority of the series.",,→
,→
,→
"reasoning": "Reference explicitly states Michael is regional manager at a paper
company.",,→
"answer": true
}
Example Input:
Reference Text: Michael Scott is a fictional character in the NBC sitcom The Office,
portrayed by Steve Carell. Michael is the regional manager of the Scranton,
Pennsylvania branch of Dunder Mifflin, a paper company, for the majority of the
series. Like his counterpart in the original British version of the show, David
Brent, he is characterized as a largely incompetent, unproductive, unprofessional
boss, though he is depicted as kinder and occasionally shown to be effective at
his job in key moments.,→
,→
,→
,→
,→
,→
Query: Tell me a paragraph bio of Michael Scott.
Claim: The portrayal of Michael Scott in the NBC sitcom The Office is similar to that
of David Brent, in the British version. ,→
Example Output:
{
"highlighted_text": "Like his counterpart in the original British version of the
show, David Brent, he is characterized as a largely incompetent, unproductive,
unprofessional boss, though he is depicted as kinder and occasionally shown to
be effective at his job in key moments.",,→
,→
,→
"reasoning": "Reference compares Michael Scott's characterization to David Brent's,
indicating similarity.",,→
"answer": true
}
Example Input:
45

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Reference Text: Michael Scott is a fictional character in the NBC sitcom The Office,
portrayed by Steve Carell. Michael is the regional manager of the Scranton,
Pennsylvania branch of Dunder Mifflin, a paper company, for the majority of the
series. Like his counterpart in the original British version of the show, David
Brent, he is characterized as a largely incompetent, unproductive, unprofessional
boss, though he is depicted as kinder and occasionally shown to be effective at
his job in key moments.,→
,→
,→
,→
,→
,→
Query: Tell me a paragraph bio of Michael Scott.
Claim: Michael Scott is the founder of The Michael Scott Paper Company.
Example Output:
{
"highlighted_text": "Michael is the regional manager of the Scranton, Pennsylvania
branch of Dunder Mifflin, a paper company, for the majority of the series.", ,→
"reasoning": "Reference mentions only Dunder Mifflin; founding another company is
not supported.",,→
"answer": false
}
Example Input:
Reference Text: Michael Scott is a fictional character in the NBC sitcom The Office,
portrayed by Steve Carell. Michael is the regional manager of the Scranton,
Pennsylvania branch of Dunder Mifflin, a paper company, for the majority of the
series. Like his counterpart in the original British version of the show, David
Brent, he is characterized as a largely incompetent, unproductive, unprofessional
boss, though he is depicted as kinder and occasionally shown to be effective at
his job in key moments.,→
,→
,→
,→
,→
,→
Query: Tell me a paragraph bio of Michael Scott.
Claim: Michael Scott is the CEO of Dunder Mifflin.
Example Output:
{
"highlighted_text": "Michael is the regional manager of the Scranton, Pennsylvania
branch of Dunder Mifflin, a paper company, for the majority of the series.", ,→
"reasoning": "Reference states regional manager, not CEO.",
"answer": false
}
Input:
Reference Text: {reference}
Query: {query}
Claim: {claim}
(Reiteration of the instruction)
Return a single JSON5 object with "highlighted_text", "reasoning", and "answer" (true
if supported; false if contradicted or unsupported). Assign true or false—never
null/None.,→
,→
Output:
A.4.5 Labeler (with ground-truth answer)
You are an AI assistant tasked with assigning a label to a claim based on its
factuality.,→
Instructions:
1. You are given:
46

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
- A reference text for the query.
- A query.
- A provided solution (final answer)
- A claim made in response to the query.
2. Rate the factuality of the claim as a boolean:
- true→The claim is well-supported by the reference text or match the given
provided solution (final answer).,→
- false→The claim contradicts the reference text or is not well-supported by it and
contradicts the provided solution (final answer). ,→
3. Provide:
- The part(s) of the reference text or the provided solution that directly support
your decision.,→
- A reasoning statement describing your rationale.
4. You must assign either true or false. Never return null or None.
Output Requirements:
- Output ONLY a single VALID JSON5 object with EXACTLY this schema:
{
"highlighted_text": "Part(s) of the reference text or the provided solution that
directly support the decision.",,→
"reasoning": "A reasoning statement describing your rationale.",
"answer": true
}
- "answer" must be a boolean (true/false).
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\.
- No trailing commas in objects or arrays.
- Use the exact top-level container specified and close it properly.
- Do not include comments, code fences, or any text outside the JSON5 output.
- Follow the schema exactly; do not add or omit keys.
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
- Any code block delimiters (e.g.,```json).
Examples:
Example Input:
Reference Text: "Paris is the capital of France."
Answer: "Paris"
Query: "What is the capital of France?"
Claim: "Paris is the capital of France."
Example Output:
{
"reasoning": "Claim matches reference and provided solution.",
"answer": true
}
Example Input:
Reference Text: "Michael is the regional manager of Dunder Mifflin."
Answer: "Regional manager"
Query: "What is Michael's role at Dunder Mifflin?"
Claim: "Michael is the CEO of Dunder Mifflin."
Example Output:
47

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
{
"reasoning": "Claim contradicts the reference and solution.",
"answer": false
}
Example Input:
Reference Text: "Compute 12×3 + 4."
Answer: "40"
Query: "Evaluate 12×3 + 4."
Claim: "The value is 40."
Example Output:
{
"reasoning": "Computation equals 40; claim matches the solution.",
"answer": true
}
Example Input:
Reference Text: "Compute 12×3 + 4."
Answer: "40"
Query: "Evaluate 12×3 + 4."
Claim: "The value is 42."
Example Output:
{
"reasoning": "The solution gives 40, not 42.",
"answer": false
}
Example Input:
Reference Text: "The sum of the interior angles of a triangle is 180 degrees."
Answer: "180 degrees"
Query: "What is the sum of the interior angles of a triangle?"
Claim: "The sum is 180 degrees."
Example Output:
{
"reasoning": "Claim agrees with the stated geometric fact and the solution.",
"answer": true
}
Input:
Reference Text: {reference}
Answer: {answer}
Query: {query}
Claim: {claim}
(Reiteration of the instruction)
Return a single JSON5 object with "highlighted_text", "reasoning", and "answer" (true
if supported; false if contradicted or unsupported). Assign true or false—never
null/None.,→
,→
Output:
48

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
A.4.6 Attacker
You are a helpful assistant that modifies a correct claim into a false but plausible
claim ("hallucination") given a reference text and a query. ,→
Instructions:
1. Read the reference text, the query, the original claim, and the two lists of prior
modified claims (that do not work / that do work). ,→
2. Generate exactly one false claim that:
- Directly answers the query.
- Contradicts or is not supported by the reference text.
- Appears plausible given the reference and query (e.g., near-miss number/date,
closely related entity, role swap). ,→
- Does not match or paraphrase any claim in either provided list.
3. Provide a reasoning statement describing why the false claim might look plausible.
Output Requirements:
- Output ONLY a single VALID JSON5 object with EXACTLY this schema:
{
"reasoning": "A reasoning statement describing why the false claim might look
plausible.",,→
"subclaim": "One-sentence false but plausible answer to the query."
}
- The "subclaim" must be a standalone sentence that answers the query and is false
with respect to the reference.,→
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\.
- No trailing commas in objects or arrays.
- Use the exact top-level container specified and close it properly.
- Do not include comments, code fences, or any text outside the JSON5 output.
- Follow the schema exactly; do not add or omit keys.
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
- Any code block delimiters (e.g.,```json).
Examples:
Example Input:
Modified claims that do not work: []
Modified claims that do work: []
Reference Text: "Gustave Eiffel oversaw the construction of the Eiffel Tower, which
opened in 1889. The tower's design is credited to Gustave Eiffel." ,→
Query: "Who designed the Eiffel Tower?"
Claim: "Gustave Eiffel designed the Eiffel Tower."
Example Output:
{
"reasoning": "Reference credits Eiffel; another associated architect name may look
credible.",,→
"subclaim": "Stephen Sauvestre designed the Eiffel Tower."
}
Example Input:
Modified claims that do not work: []
Modified claims that do work: []
Reference Text: "The capital of Australia is Canberra."
49

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Query: "What is the capital of Australia?"
Claim: "Canberra."
Example Output:
{
"reasoning": "Larger city misconception is common.",
"subclaim": "Sydney is the capital of Australia."
}
Input:
Modified claims that do not work: {claims_not_work}
Modified claims that do work: {claims_work}
Reference Text: {reference}
Query: {query}
Claim: {claim}
(Reiteration of the instruction)
Generate exactly one false but plausible claim answering the query, unsupported by or
contradicting the reference, not overlapping with provided lists; then return a
JSON5 object with "reasoning" and "subclaim".,→
,→
Output:
A.4.7 Confusee
You are a helpful assistant that evaluates whether a claim sounds like something you
(an LLM) would plausibly say—and potentially hallucinate—based on the provided
reference text and query.,→
,→
Instructions:
1. Read the reference text, the query, and the claim.
2. Decide whether the claim is a plausible LLM-style statement that could arise from
the reference and query even if it is not supported or is false. ,→
3. Return true if the claim is plausibly hallucinated; otherwise return false.
4. Provide a reasoning statement describing your rationale.
Output Requirements:
- Output ONLY a single VALID JSON5 object with EXACTLY this schema:
{
"reasoning": "A reasoning statement describing your rationale.",
"answer": (true|false)
}
- "answer" must be a boolean (true/false).
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\.
- No trailing commas in objects or arrays.
- Use the exact top-level container specified and close it properly.
- Do not include comments, code fences, or any text outside the JSON5 output.
- Follow the schema exactly; do not add or omit keys.
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
- Any code block delimiters (e.g.,```json).
50

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Examples:
Example Input:
Reference Text: "The capital of Australia is Canberra."
Query: "What is the capital of Australia?"
Claim: "Sydney is the capital of Australia."
Example Output:
{
"reasoning": "Common misconception favors the largest city over the capital.",
"answer": true
}
Example Input:
Reference Text: "Paris is the capital of France."
Query: "What is the capital of France?"
Claim: "Paris is the capital of France."
Example Output:
{
"reasoning": "Claim is directly supported by the reference; not a hallucination.",
"answer": false
}
Input:
Reference Text: {reference}
Query: {query}
Claim: {claim}
(Reiteration of the instruction)
Return a JSON5 object with "reasoning" and "answer" (true if plausibly hallucinated;
else false).,→
Output:
A.4.8 Model Confidence Score
You are an AI assistant tasked with assigning a confidence score to a claim based on
its factuality.,→
Instructions:
1. You are given:
- A reference text for the query.
- A query.
- A claim made in response to the query.
2. Rate the factuality of the claim with a numeric score in [0.0, 1.0]:
- 0.0 -> The claim is false, contradicts the reference, or is not well-supported.
- 1.0 -> The claim is factual and well-supported by the reference, or can be
logically deduced from the reference and query. ,→
3. Provide:
- A score.
- The parts of the reference text that directly support your scoring decision.
- A reasoning statement describing your rationale.
4. You must assign a numeric score. Never return null, None, or a non-numeric value.
Important: A claim should be considered true (score 1.0) if it is either:
- Directly stated in the reference text, OR
51

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
- Can be logically deduced or calculated from the information provided in the
reference text and query.,→
For mathematical claims, perform the necessary calculations based on the given data.
Output Requirements:
- Output ONLY a single VALID JSON5 object with EXACTLY these keys:
{
"highlighted_text": "Part(s) of the reference text that support the decision.",
"reasoning": "A reasoning statement describing your rationale.",
"score": 0.0-1.0
}
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\\.
- No trailing commas in objects or arrays.
- Follow the schema exactly.
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
Input:
Reference Text: {reference}
Query: {query}
Claim: {claim}
Reiteration of Instructions:
Return a single JSON5 object with "highlighted_text", "reasoning", "score". Assign a
numeric score—never null/None.,→
Output:
A.4.9 Merger (FActScore)
You will get an instruction and a set of facts that are true. Construct an answer
using ONLY the facts provided, and try to use all facts as long as its possible.
If the input facts are empty, output the empty string. Do not repeat the
instruction.,→
,→
,→
Input:
The facts: {claims}
The instruction: {query}
Remember, If the input facts are empty, output the empty string. Do not repeat the
instruction.,→
Output:
A.4.10 Merger (Natural Questions)
You will get a natural question and parts of an answer, which you are to merge into
coherent prose. Make sure to include all the parts in the answer. There may be
parts that are seemingly unrelated to the others, but DO NOT add additional
information or reasoning to merge them. If the input parts are empty, output the
empty string. Do not repeat the question.,→
,→
,→
,→
Input:
The parts: {claims}
The question: {query}
52

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Remember, DO NOT add any additional information or commentary, just combine the
parts. If the input parts are empty, output the empty string. Do not repeat the
question.,→
,→
Output:
A.4.11 Merger (MATH)
You will get a math problem and a set of steps that are true. Construct an answer
using ONLY the steps provided. Make sure to include all the steps in the answer,
and do not add any additional steps or reasoning. These steps may not fully solve
the problem, but merging them could assist someone in solving the problem. If the
input steps are empty, output the empty string. Do not repeat the math problem.,→
,→
,→
,→
Input:
The steps: {claims}
The math problem: {query}
Remember, do not do any additional reasoning, just combine the given steps. If the
input steps are empty, output the empty string. Do not repeat the math problem. ,→
Output:
A.4.12 Correctness
I need your help in evaluating an answer provided by an LLM against ground truth
answers. Your task is to determine if the LLM's response matches the ground truth
answers. Please analyze the provided data and make a decision.,→
,→
Instructions:
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answers".
2. Consider the substance of the answers – look for equivalent information or correct
answers. Do not focus on exact wording unless the exact wording is crucial to the
meaning.,→
,→
3. Your final decision should be based on whether the meaning and the vital facts of
the "Ground Truth Answers" are present in the "Predicted Answer." ,→
4. Categorize the answer as one of the following:
- "perfect": The answer is completely correct and matches the ground truth.
- "acceptable": The answer is partially correct or contains the main idea of the
ground truth.,→
- "incorrect": The answer is wrong or contradicts the ground truth.
- "missing": The answer is "I don't know", "invalid question", or similar responses
indicating lack of knowledge.,→
Output Requirements:
- Output ONLY a single VALID JSON5 object with EXACTLY this schema:
{
"reasoning": "A reasoning statement describing your rationale.",
"answer": "One of perfect, acceptable, incorrect, or missing"
}
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\.
- No trailing commas in objects or arrays.
- Use the exact top-level container specified and close it properly.
- Do not include comments, code fences, or any text outside the JSON5 output.
- Follow the schema exactly; do not add or omit keys.
53

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
- Any code block delimiters (e.g.,```json).
Input:
Query: {query}
Predicted Answer: {merged_string}
Ground Truth Answer: {answer}
Reiteration of Instructions (before output):
1. Carefully compare the "Predicted Answer" with the "Ground Truth Answers".
2. Consider the substance of the answers – look for equivalent information or correct
answers. Do not focus on exact wording unless the exact wording is crucial to the
meaning.,→
,→
3. Your final decision should be based on whether the meaning and the vital facts of
the "Ground Truth Answers" are present in the "Predicted Answer." ,→
4. Categorize the answer as one of the following:
- "perfect": The answer is completely correct and matches the ground truth.
- "acceptable": The answer is partially correct or contains the main idea of the
ground truth.,→
- "incorrect": The answer is wrong or contradicts the ground truth.
- "missing": The answer is "I don't know", "invalid question", or similar responses
indicating lack of knowledge.,→
Output:
A.4.13 Sufficient Correctness
You are an expert LLM evaluator that excels at evaluating a RESPONSE with respect to
a QUERY given a REFERENCE.,→
Consider the following criteria:
Sufficient Correctness:
1 IF the RESPONSE contains a sufficient amount of CORRECT information (verified
against the REFERENCE) to infer the answer to the QUERY. ,→
0 IF the RESPONSE does not contain a sufficient amount of CORRECT information to
infer the answer to the QUERY.,→
Important:
Judge only the correctness of the RESPONSE content according to the REFERENCE.
Do not infer correctness from external knowledge.
Output ONLY a single VALID JSON5 object with EXACTLY these keys:
{
"explanation": An explanation describing your rationale, with with you will make
your decision on sufficient_correctness, ,→
"sufficient_correctness": 1 IF the RESPONSE contains a sufficient amount of correct
information (verified against the REFERENCE) to infer the answer to the QUERY,
0 IF the RESPONSE does not contain a sufficient amount of correct information
to infer the answer to the QUERY.,→
,→
,→
}
- "sufficient_correctness" must be an integer (0/1).
JSON5 Rules:
- Use DOUBLE QUOTES (") for all keys and all string values.
- Escape double quotes inside string values as \".
- Escape backslashes as \\.
54

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
- No trailing commas in objects or arrays.
- Use the exact top-level container specified and close it properly.
- Do not include comments, code fences, or any text outside the JSON5 output.
- Follow the schema exactly; do not add or omit keys.
Do NOT include:
- Any text, explanations, comments, or formatting outside of the JSON5.
- Any code block delimiters (e.g.,```json).
Input:
### QUERY
{query}
### REFERENCE
{reference}
### RESPONSE
{response}
Output:
A.4.14 MATH reference
You are a helpful assistant that extracts the prerequisite mathematics knowledge
needed to answer a given question—without solving it. ,→
Instructions:
1. Read the question.
2. Identify only the minimal prerequisite items across concepts, definitions,
theorems/properties, formulas, techniques, notation, assumptions/conditions, and
common pitfalls required to answer the question.,→
,→
3. When you state a theorem, a definition, or a formula, write out its full, standard
statement (not just the name). For theorems, include hypotheses and conclusions;
for definitions, give the precise meaning; for formulas, write the exact
equation(s) in standard notation.,→
,→
,→
4. Do NOT provide the answer or partial solution steps.
5. Do NOT use examples that arise directly from the given question; keep statements
general and problem-agnostic.,→
Output Requirements:
- Produce PLAIN TEXT ONLY.
- Write free-form prose (no headings, lists, or numbering).
- Mention required topics, definitions, fully written theorems/properties, formulas,
techniques, notation conventions, assumptions/conditions, and common pitfalls in
sentences.,→
,→
- The output may be long; include complete statements where needed.
Do NOT include:
- Any answer, hints, or step-by-step solution.
- Any examples derived from the given question.
Reiteration of the instructions:
1. Read the question.
2. Identify only the minimal prerequisite items across concepts, definitions,
theorems/properties, formulas, techniques, notation, assumptions/conditions, and
common pitfalls required to answer the question.,→
,→
3. When you state a theorem, a definition, or a formula, write out its full, standard
statement (not just the name). For theorems, include hypotheses and conclusions;
for definitions, give the precise meaning; for formulas, write the exact
equation(s) in standard notation.,→
,→
,→
55

Is Conformal Factuality for RAG-based LLMs Robust?A PREPRINT
4. Do NOT provide the answer or partial solution steps.
5. Do NOT use examples that arise directly from the given question; keep statements
general and problem-agnostic.,→
Input:
{query}
Output:
56